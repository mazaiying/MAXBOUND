import os
import sys
import json
import time
import numpy as np
import torch
import faiss

sys.path.insert(0, os.path.expanduser("~/maxbound"))
from bench.common import write_result_json, audit_exact_topk, compute_ir_metrics
from src.oracle import exact_maxsim_oracle


def hadamard_matrix_128():
    H = np.array([[1.0]], dtype=np.float32)
    for _ in range(7):
        H = np.block([[H, H], [H, -H]]) / np.float32(np.sqrt(2.0))
    return H.astype(np.float32)


@torch.no_grad()
def batched_exact_maxsim(q, d_toks, d_offs, doc_ids, device=1):
    if len(doc_ids) == 0:
        return np.empty(0, dtype=np.float32), 0
    scores = np.empty(len(doc_ids), dtype=np.float32)
    m, dim = q.shape
    tok_ops = 0
    for s in range(0, len(doc_ids), 512):
        sub = doc_ids[s:s + 512]
        lens = d_offs[sub + 1] - d_offs[sub]
        tok_ops += int(m * lens.sum())
        toks_list = [d_toks[int(d_offs[d]):int(d_offs[d + 1])] for d in sub]
        cat_t = torch.cat(toks_list, dim=0).to(device=device, dtype=torch.float32)
        sim = q @ cat_t.t()
        seg_ids = torch.from_numpy(np.repeat(np.arange(len(sub), dtype=np.int64), lens)).to(device)
        out = torch.full((m, len(sub)), -1e9, device=device)
        out.scatter_reduce_(1, seg_ids.unsqueeze(0).expand(m, -1), sim, reduce="amax", include_self=True)
        scores[s:s + len(sub)] = out.sum(dim=0).cpu().numpy()
    return scores, tok_ops


@torch.no_grad()
def analyze_dataset_e0(ds_name, data_dir, device=1, topk=10, K_coarse=2048, P_pq=16, c_warm=64):
    print(f"\n=== E0 Analyzing {ds_name} ===", flush=True)
    t0 = time.time()
    d_toks = torch.from_numpy(np.load(os.path.join(data_dir, "doc_tokens.fp16.npy")))
    d_offs = np.load(os.path.join(data_dir, "doc_offsets.int64.npy"))
    q_toks = torch.from_numpy(np.load(os.path.join(data_dir, "query_tokens.fp16.npy")))
    q_offs = np.load(os.path.join(data_dir, "query_offsets.int64.npy"))
    with open(os.path.join(data_dir, "qrels.json")) as f:
        qrels = json.load(f)
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)

    N_docs = len(d_offs) - 1
    N_toks = d_toks.shape[0]
    dim = d_toks.shape[1]
    B = min(len(q_offs) - 1, 256)
    q_offs_sub = q_offs[:B + 1]
    q_toks_sub = q_toks[:int(q_offs_sub[-1])].float()

    oracle_ids, oracle_scores = exact_maxsim_oracle(
        q_toks_sub, q_offs_sub, d_toks, d_offs, topk=topk, doc_batch_size=2048, device=device
    )
    ir = compute_ir_metrics(oracle_ids, qrels, k=topk)
    print(f"  Oracle IR metrics ({time.time()-t0:.1f}s): {ir}", flush=True)

    lens = np.diff(d_offs)
    pad_waste_bs64 = float(np.mean([
        1.0 - float(lens[s:s + 64].sum()) / float(len(lens[s:s + 64]) * lens[s:s + 64].max())
        for s in range(0, N_docs, 64)
    ]))
    pad_waste_global = 1.0 - float(lens.sum()) / float(N_docs * lens.max())

    # 1. Fast GPU Coarse KMeans via PyTorch (no faiss GPU resource conflict!)
    K = min(K_coarse, max(256, int(N_toks // 20)))
    rng = np.random.default_rng(2027)
    n_sample = min(100000, N_toks)
    samp_idx = rng.choice(N_toks, size=n_sample, replace=False)
    x_samp_gpu = d_toks[samp_idx].to(device=device, dtype=torch.float32)
    perm_init = torch.randperm(n_sample, device=device)[:K]
    centroids = x_samp_gpu[perm_init].clone()
    for _ in range(15):
        # Assign in chunks
        d2 = 2.0 - 2.0 * (x_samp_gpu @ centroids.t())
        assign = d2.argmin(dim=1)
        new_c = torch.zeros_like(centroids)
        cnt = torch.zeros(K, device=device)
        new_c.index_add_(0, assign, x_samp_gpu)
        cnt.index_add_(0, assign, torch.ones(n_sample, device=device))
        valid = cnt > 0
        centroids[valid] = new_c[valid] / cnt[valid].unsqueeze(1)
    print(f"  Coarse KMeans K={K} done ({time.time()-t0:.1f}s)", flush=True)

    cids = torch.empty(N_toks, dtype=torch.int32, device="cpu")
    res_norms = torch.empty(N_toks, dtype=torch.float32, device="cpu")
    res_vecs = np.empty((N_toks, dim), dtype=np.float32)
    c_sq = (centroids ** 2).sum(dim=1)
    for s in range(0, N_toks, 131072):
        e = min(s + 131072, N_toks)
        xb = d_toks[s:e].to(device=device, dtype=torch.float32)
        dists = (xb ** 2).sum(dim=1, keepdim=True) + c_sq.unsqueeze(0) - 2.0 * (xb @ centroids.t())
        _, cid = dists.min(dim=1)
        res = xb - centroids[cid]
        cids[s:e] = cid.cpu().to(torch.int32)
        res_norms[s:e] = res.norm(dim=1).cpu()
        res_vecs[s:e] = res.cpu().numpy()

    # 2. Fast GPU Batched PQ training & encoding across all P_pq subspaces simultaneously!
    sub_d = dim // P_pq
    res_samp_gpu = torch.from_numpy(res_vecs[samp_idx]).to(device).view(n_sample, P_pq, sub_d).permute(1, 0, 2)  # [P, n_sample, sub_d]
    p_init = torch.randint(0, n_sample, (P_pq, 256), device=device)
    pq_c_t = torch.gather(res_samp_gpu, 1, p_init.unsqueeze(-1).expand(-1, -1, sub_d)).clone()  # [P, 256, sub_d]
    for _ in range(15):
        # Batched distance [P, n_sample, 256]
        for cs in range(0, n_sample, 32768):
            ce = min(cs + 32768, n_sample)
            sub_x = res_samp_gpu[:, cs:ce, :]  # [P, m, sub_d]
            x2 = (sub_x ** 2).sum(dim=-1, keepdim=True)
            c2 = (pq_c_t ** 2).sum(dim=-1).unsqueeze(1)
            d2 = x2 + c2 - 2.0 * torch.bmm(sub_x, pq_c_t.transpose(1, 2))
            ass = d2.argmin(dim=-1)  # [P, m]
            if cs == 0:
                acc = torch.zeros_like(pq_c_t)
                cnt = torch.zeros((P_pq, 256), device=device)
            for p_i in range(P_pq):
                acc[p_i].index_add_(0, ass[p_i], sub_x[p_i])
                cnt[p_i].index_add_(0, ass[p_i], torch.ones(ce - cs, device=device))
        valid = cnt > 0
        pq_c_t[valid] = acc[valid] / cnt[valid].unsqueeze(-1)

    # Encode all tokens with PQ & compute exact pq_rho and int4_rho
    pq_codes = np.empty((N_toks, P_pq), dtype=np.uint8)
    pq_rho = np.empty(N_toks, dtype=np.float32)
    int4_rho = np.empty(N_toks, dtype=np.float32)
    H = torch.from_numpy(hadamard_matrix_128()).to(device)
    c2 = (pq_c_t ** 2).sum(dim=-1).unsqueeze(1)
    for s in range(0, N_toks, 65536):
        e = min(s + 65536, N_toks)
        rb = torch.from_numpy(res_vecs[s:e]).to(device)  # [n, dim]
        sub_r = rb.view(e - s, P_pq, sub_d).permute(1, 0, 2)  # [P, n, sub_d]
        x2 = (sub_r ** 2).sum(dim=-1, keepdim=True)
        d2 = x2 + c2 - 2.0 * torch.bmm(sub_r, pq_c_t.transpose(1, 2))
        codes_b = d2.argmin(dim=-1).t().contiguous()  # [n, P]
        pq_codes[s:e] = codes_b.cpu().to(torch.uint8).numpy()

        xb = d_toks[s:e].to(device=device, dtype=torch.float32)
        cb = centroids[cids[s:e].to(device=device, dtype=torch.long)]
        recon = cb.clone()
        for p in range(P_pq):
            recon[:, p * sub_d:(p + 1) * sub_d] += pq_c_t[p][codes_b[:, p]]
        pq_rho[s:e] = ((xb - recon).norm(dim=1) + 1e-4).cpu().numpy()

        hr = rb @ H.t()
        scale = hr.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 7.0
        hr_hat = torch.round(hr / scale).clamp(-7, 7) * scale
        int4_rho[s:e] = ((hr - hr_hat).norm(dim=1) + 1e-4).cpu().numpy()
    print(f"  PQ-{P_pq} & INT4 encoding done ({time.time()-t0:.1f}s), mean_pq_rho={pq_rho.mean():.4f}, mean_int4_rho={int4_rho.mean():.4f}", flush=True)

    cids_np = cids.numpy()
    rn_np = res_norms.numpy()
    doc_g_cids, doc_g_r = [], []
    for d in range(N_docs):
        s, e = int(d_offs[d]), int(d_offs[d + 1])
        uniq, inv = np.unique(cids_np[s:e], return_inverse=True)
        rg = np.zeros(len(uniq), dtype=np.float32)
        np.maximum.at(rg, inv, rn_np[s:e])
        doc_g_cids.append(uniq)
        doc_g_r.append(rg + 1e-4)
    total_groups = sum(len(x) for x in doc_g_cids)
    g_lens = np.array([len(x) for x in doc_g_cids], dtype=np.int64)
    g_doc_idx = torch.from_numpy(np.repeat(np.arange(N_docs, dtype=np.int64), g_lens)).to(device)
    gc_t = torch.from_numpy(np.concatenate(doc_g_cids)).to(device=device, dtype=torch.long)
    gr_t = torch.from_numpy(np.concatenate(doc_g_r)).to(device=device, dtype=torch.float32)

    pq_rho_t = torch.from_numpy(pq_rho).to(device)
    int4_rho_t = torch.from_numpy(int4_rho).to(device)
    pq_codes_t = torch.from_numpy(pq_codes).to(device=device, dtype=torch.long)
    cids_t = cids.to(device=device, dtype=torch.long)

    B_eval = min(B, 64)
    l0_doc_surv = 0
    l1_pq_doc_surv = 0
    l1b_int4_doc_surv = 0
    l1_tok_ops_done = 0
    l1_tok_ops_full = 0
    exact_tok_ops_done = 0
    full_tok_ops = 0

    pred_ids = np.full((B_eval, topk), -1, dtype=np.int64)
    pred_scores = np.full((B_eval, topk), -1e9, dtype=np.float32)

    for qi in range(B_eval):
        qs, qe = int(q_offs_sub[qi]), int(q_offs_sub[qi + 1])
        q = q_toks_sub[qs:qe].to(device=device, dtype=torch.float32)
        m = q.shape[0]
        full_tok_ops += m * N_toks
        q_norm = q.norm(dim=1)
        S_c = q @ centroids.t()
        lut = torch.einsum("mpd,pkd->mpk", q.view(m, P_pq, sub_d), pq_c_t)

        sg = S_c[:, gc_t]
        ub_g = sg + q_norm[:, None] * gr_t[None, :]
        lb_g = sg - q_norm[:, None] * gr_t[None, :]
        idx_exp = g_doc_idx.view(1, -1).expand(m, -1)
        s0_m = torch.full((m, N_docs), -1e9, device=device)
        ub0_m = torch.full((m, N_docs), -1e9, device=device)
        lb0_m = torch.full((m, N_docs), -1e9, device=device)
        s0_m.scatter_reduce_(1, idx_exp, sg, reduce="amax", include_self=True)
        ub0_m.scatter_reduce_(1, idx_exp, ub_g, reduce="amax", include_self=True)
        lb0_m.scatter_reduce_(1, idx_exp, lb_g, reduce="amax", include_self=True)

        approx_s0 = s0_m.sum(dim=0).cpu().numpy()
        ub0 = ub0_m.sum(dim=0).cpu().numpy()

        warm_cands = np.argpartition(-approx_s0, min(c_warm, N_docs - 1))[:min(c_warm, N_docs)]
        warm_scores, w_ops = batched_exact_maxsim(q, d_toks, d_offs, warm_cands, device=device)
        exact_tok_ops_done += w_ops
        warm_set = set(int(x) for x in warm_cands)
        topk_list = sorted([(float(sc), int(d)) for sc, d in zip(warm_scores, warm_cands)], key=lambda x: (-x[0], x[1]))[:topk]
        tau = topk_list[-1][0]
        tau_safe = tau - 1e-3 * m

        surv_l0 = np.array([int(d) for d in np.where(ub0 >= tau_safe)[0] if int(d) not in warm_set], dtype=np.int64)
        if len(surv_l0) > 0:
            surv_l0 = surv_l0[np.argsort(-approx_s0[surv_l0])]
        l0_doc_surv += len(surv_l0) + len(warm_cands)

        for bs in range(0, len(surv_l0), 1024):
            batch_d = surv_l0[bs:bs + 1024]
            tau_safe = tau - 1e-3 * m
            batch_d = batch_d[ub0[batch_d] >= tau_safe]
            if len(batch_d) == 0:
                continue

            b_lens = lens[batch_d]
            l1_tok_ops_full += int(m * b_lens.sum())

            bd_t = torch.from_numpy(batch_d).to(device)
            cur_ub = ub0_m[:, bd_t].clone()
            cur_lb = lb0_m[:, bd_t].clone()
            alive = torch.ones(len(batch_d), dtype=torch.bool, device=device)

            t_idx_list = [torch.arange(int(d_offs[d]), int(d_offs[d + 1]), device=device) for d in batch_d]
            cat_tidx = torch.cat(t_idx_list)
            b_cids = cids_t[cat_tidx]
            b_codes = pq_codes_t[cat_tidx]
            b_rho = pq_rho_t[cat_tidx]
            seg_doc = torch.from_numpy(np.repeat(np.arange(len(batch_d), dtype=np.int64), b_lens)).to(device)

            q_order = torch.argsort((cur_ub - cur_lb).mean(dim=1), descending=True)
            for qb in range(0, m, 8):
                if not alive.any():
                    break
                q_idx = q_order[qb:qb + 8]
                alive_np = alive.cpu().numpy()
                l1_tok_ops_done += int(len(q_idx) * b_lens[alive_np].sum())

                s_hat = S_c[q_idx][:, b_cids]
                for p in range(P_pq):
                    s_hat = s_hat + lut[q_idx, p][:, b_codes[:, p]]
                err = q_norm[q_idx, None] * b_rho[None, :]
                seg_exp = seg_doc.unsqueeze(0).expand(len(q_idx), -1)
                ub_q = torch.full((len(q_idx), len(batch_d)), -1e9, device=device)
                lb_q = torch.full((len(q_idx), len(batch_d)), -1e9, device=device)
                ub_q.scatter_reduce_(1, seg_exp, s_hat + err, reduce="amax", include_self=True)
                lb_q.scatter_reduce_(1, seg_exp, s_hat - err, reduce="amax", include_self=True)

                cur_ub[q_idx] = torch.minimum(cur_ub[q_idx], ub_q)
                cur_lb[q_idx] = torch.maximum(cur_lb[q_idx], lb_q)
                running_ub = cur_ub.sum(dim=0)
                tau_safe = tau - 1e-3 * m
                alive = alive & (running_ub >= tau_safe)

            surv_idx = torch.nonzero(alive, as_tuple=False).squeeze(-1).cpu().numpy()
            if len(surv_idx) == 0:
                continue
            surv_docs = batch_d[surv_idx]
            l1_pq_doc_surv += len(surv_docs)

            ex_scores, ex_ops = batched_exact_maxsim(q, d_toks, d_offs, surv_docs, device=device)
            exact_tok_ops_done += ex_ops

            # Batched Level-1b INT4 check
            s_tidx = torch.cat([torch.arange(int(d_offs[d]), int(d_offs[d + 1]), device=device) for d in surv_docs])
            s_lens = lens[surv_docs]
            s_cat_np = np.concatenate([np.arange(int(d_offs[d]), int(d_offs[d + 1])) for d in surv_docs])
            rb = torch.from_numpy(res_vecs[s_cat_np]).to(device)
            hr = rb @ H.t()
            scale = hr.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 7.0
            hr_hat = torch.round(hr / scale).clamp(-7, 7) * scale
            d_hat_int4 = centroids[cids_t[s_tidx]] + (hr_hat @ H)
            s_hat_i4 = q @ d_hat_int4.t()
            err_i4 = q_norm[:, None] * int4_rho_t[s_tidx][None, :]
            seg_s = torch.from_numpy(np.repeat(np.arange(len(surv_docs), dtype=np.int64), s_lens)).to(device)
            ub_i4_m = torch.full((m, len(surv_docs)), -1e9, device=device)
            ub_i4_m.scatter_reduce_(1, seg_s.unsqueeze(0).expand(m, -1), s_hat_i4 + err_i4, reduce="amax", include_self=True)
            ub_i4 = ub_i4_m.sum(dim=0).cpu().numpy()
            l1b_int4_doc_surv += int((ub_i4 >= tau - 1e-3 * m).sum())

            for sc, did_val in zip(ex_scores, surv_docs):
                topk_list.append((float(sc), int(did_val)))
            topk_list.sort(key=lambda x: (-x[0], x[1]))
            topk_list = topk_list[:topk]
            tau = topk_list[-1][0]

        for r_idx, (sc, did) in enumerate(topk_list):
            pred_ids[qi, r_idx] = did
            pred_scores[qi, r_idx] = sc

    audit = audit_exact_topk(pred_ids, pred_scores, oracle_ids[:B_eval], oracle_scores[:B_eval])
    comp_den = total_groups * 4 + N_toks * (P_pq + 2)
    res = {
        "dataset": ds_name,
        "num_docs": N_docs,
        "total_tokens": N_toks,
        "avg_doc_len": round(float(meta["doc_len_mean"]), 2),
        "doc_len_cv": round(float(meta["doc_len_cv"]), 3),
        "avg_groups_per_doc": round(total_groups / N_docs, 2),
        "group_vs_token_ratio": round(total_groups / N_toks, 3),
        "pad_waste_bs64_pct": round(pad_waste_bs64 * 100, 2),
        "pad_waste_global_pct": round(pad_waste_global * 100, 2),
        "ir_metrics": ir,
        "K_coarse": K,
        "P_pq": P_pq,
        "mean_pq_rho": round(float(pq_rho.mean()), 4),
        "mean_int4_rho": round(float(int4_rho.mean()), 4),
        "l0_doc_prune_pct": round((1.0 - l0_doc_surv / (B_eval * N_docs)) * 100, 2),
        "l1a_pq_doc_prune_pct": round((1.0 - l1_pq_doc_surv / (B_eval * N_docs)) * 100, 2),
        "l1b_int4_doc_prune_pct": round((1.0 - l1b_int4_doc_surv / (B_eval * N_docs)) * 100, 2),
        "l1_prog_tok_skip_pct": round((1.0 - l1_tok_ops_done / max(l1_tok_ops_full, 1)) * 100, 2),
        "exact_tok_ops_pruned_pct": round((1.0 - exact_tok_ops_done / full_tok_ops) * 100, 3),
        "fp16_mb": round(N_toks * dim * 2 / (1024**2), 2),
        "maxbound_l0_l1a_mb": round(comp_den / (1024**2), 2),
        "compression_ratio": round((N_toks * dim * 2) / comp_den, 2),
        "audit": audit,
        "elapsed_s": round(time.time() - t0, 2),
    }
    print(f"  [{ds_name}] L0_prune={res['l0_doc_prune_pct']}% | L1a_PQ_prune={res['l1a_pq_doc_prune_pct']}% | L1b_INT4_prune={res['l1b_int4_doc_prune_pct']}% | Exact_Tok_Pruned={res['exact_tok_ops_pruned_pct']}% | PadWaste64={res['pad_waste_bs64_pct']}% | Comp={res['compression_ratio']}x | Audit={audit['agreement']} | {res['elapsed_s']}s", flush=True)
    return res


if __name__ == "__main__":
    data_root = os.path.expanduser("~/maxbound/data_mv")
    records = []
    for ds in ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "longbench", "hotpotqa"]:
        ddir = os.path.join(data_root, ds)
        if os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            records.append(analyze_dataset_e0(ds, ddir, device=1, K_coarse=2048, P_pq=16))
    write_result_json(os.path.expanduser("~/maxbound/results/e0_motivation.json"), records)
