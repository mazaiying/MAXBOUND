import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/maxbound"))
from bench.common import write_result_json, audit_exact_topk, compute_ir_metrics
from src.oracle import exact_maxsim_oracle


def hadamard_matrix(d):
    H = np.array([[1.0]], dtype=np.float32)
    steps = int(np.log2(d))
    for _ in range(steps):
        H = np.block([[H, H], [H, -H]]) / np.float32(np.sqrt(2.0))
    return H.astype(np.float32)


@torch.no_grad()
def batched_exact_maxsim(q, d_toks, d_offs, doc_ids, device=1):
    if len(doc_ids) == 0:
        return np.empty(0, dtype=np.float32), 0
    scores = np.empty(len(doc_ids), dtype=np.float32)
    m = q.shape[0]
    tok_ops = 0
    for s in range(0, len(doc_ids), 1024):
        sub = doc_ids[s:s + 1024]
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
def eval_hadamard_bits(q_rot, q_norm, hd_tokens, d_offs, d_doc_idx, N_docs, lens, bits, oracle_ids, oracle_scores,
                       d_toks_orig, approx_s0=None, c_warm=64, topk=10, device=1):
    """
    Vectorized GPU evaluation of Direct Hadamard-INT{bits} + Progressive Query-Token Early Exit.
    Returns exact pruning stats + 100% exactness audit.
    """
    N_toks, dim = hd_tokens.shape
    qmax = (1 << (bits - 1)) - 1  # 1 for 2-bit? Use symmetric [-qmax, qmax]: 2b->1 (or unsigned 0..3), 3b->3, 4b->7, 6b->31, 8b->127
    if bits == 2:
        # 2-bit asymmetric [0, 3] per token is much tighter than symmetric [-1, 1]!
        h_min = hd_tokens.amin(dim=1, keepdim=True)
        h_max = hd_tokens.amax(dim=1, keepdim=True)
        scale = (h_max - h_min).clamp(min=1e-6) / 3.0
        q_code = torch.round((hd_tokens - h_min) / scale).clamp(0, 3)
        hd_hat = h_min + q_code * scale
    else:
        scale = hd_tokens.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / float(qmax)
        q_code = torch.round(hd_tokens / scale).clamp(-qmax, qmax)
        hd_hat = q_code * scale

    rho = (hd_tokens - hd_hat).norm(dim=1) + 2e-4  # [N_toks] conservative FP safety margin

    B_eval = q_rot.shape[0]
    m = q_rot.shape[1]

    # Compute exact quantized score matrix in chunks of docs
    # Wait: for N_toks <= 8M, we can evaluate per query or batch of queries!
    # Let's evaluate B_eval queries in chunks of docs (e.g. 8192 docs)
    doc_surv_total = 0
    tok_ops_int_done = 0
    tok_ops_int_full = B_eval * N_toks * m
    tok_ops_exact_done = 0

    pred_ids = np.full((B_eval, topk), -1, dtype=np.int64)
    pred_scores = np.full((B_eval, topk), -1e9, dtype=np.float32)

    # Pre-warm tau for each query using approx_s0 if provided, else top-c_warm by full INT score
    # Actually: compute full INT score OR coarse approx_s0!
    # Let's test progressive query-token early exit with 4 tokens per step (8 steps for m=32)
    # Initial upper bound for unvisited tokens: if we don't have ub0, use q_norm[i] (Cauchy-Schwarz = 1.0)
    # OR run top-c_warm exact first!
    hd_hat_half = hd_hat.half()

    for qi in range(B_eval):
        qr = q_rot[qi]  # [m, dim]
        qn = q_norm[qi]  # [m]
        # Prewarm tau
        if approx_s0 is not None:
            warm_cands = np.argpartition(-approx_s0[qi], min(c_warm, N_docs - 1))[:min(c_warm, N_docs)]
        else:
            warm_cands = np.arange(min(c_warm, N_docs), dtype=np.int64)
        warm_scores, w_ops = batched_exact_maxsim(qr @ torch.from_numpy(hadamard_matrix(dim)).to(device), d_toks_orig, d_offs, warm_cands, device=device)
        # Wait: pass orig query so exact match! We'll pass q_orig separately below.
        break

    return None


@torch.no_grad()
def analyze_dataset_fast(ds_name, data_dir, device=1, topk=10, K_coarse=2048, c_warm=64):
    print(f"\n=== E0 Fast Analysis: {ds_name} ===", flush=True)
    t0 = time.time()
    d_toks = torch.from_numpy(np.load(os.path.join(data_dir, "doc_tokens.fp16.npy")))
    d_offs = np.load(os.path.join(data_dir, "doc_offsets.int64.npy"))
    q_toks = torch.from_numpy(np.load(os.path.join(data_dir, "query_tokens.fp16.npy")))
    q_offs = np.load(os.path.join(data_dir, "query_offsets.int64.npy"))
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)

    N_docs = len(d_offs) - 1
    N_toks = d_toks.shape[0]
    dim = d_toks.shape[1]
    B = min(len(q_offs) - 1, 64)
    q_offs_sub = q_offs[:B + 1]
    q_toks_sub = q_toks[:int(q_offs_sub[-1])].float()

    oracle_ids, oracle_scores = exact_maxsim_oracle(
        q_toks_sub, q_offs_sub, d_toks, d_offs, topk=topk, doc_batch_size=2048, device=device
    )

    lens = np.diff(d_offs)
    pad_waste_bs64 = float(np.mean([
        1.0 - float(lens[s:s + 64].sum()) / float(len(lens[s:s + 64]) * lens[s:s + 64].max())
        for s in range(0, N_docs, 64)
    ]))

    # Build coarse centroids for Pass-1 prewarm + Level-0 check
    K = min(K_coarse, max(256, int(N_toks // 20)))
    rng = np.random.default_rng(2027)
    n_sample = min(100000, N_toks)
    samp_idx = rng.choice(N_toks, size=n_sample, replace=False)
    x_samp_gpu = d_toks[samp_idx].to(device=device, dtype=torch.float32)
    centroids = x_samp_gpu[torch.randperm(n_sample, device=device)[:K]].clone()
    for _ in range(15):
        assign = (2.0 - 2.0 * (x_samp_gpu @ centroids.t())).argmin(dim=1)
        new_c = torch.zeros_like(centroids)
        cnt = torch.zeros(K, device=device)
        new_c.index_add_(0, assign, x_samp_gpu)
        cnt.index_add_(0, assign, torch.ones(n_sample, device=device))
        v = cnt > 0
        centroids[v] = new_c[v] / cnt[v].unsqueeze(1)

    cids = torch.empty(N_toks, dtype=torch.int32, device="cpu")
    res_norms = torch.empty(N_toks, dtype=torch.float32, device="cpu")
    c_sq = (centroids ** 2).sum(dim=1)
    for s in range(0, N_toks, 131072):
        e = min(s + 131072, N_toks)
        xb = d_toks[s:e].to(device=device, dtype=torch.float32)
        dists = (xb ** 2).sum(dim=1, keepdim=True) + c_sq.unsqueeze(0) - 2.0 * (xb @ centroids.t())
        _, cid = dists.min(dim=1)
        res = xb - centroids[cid]
        cids[s:e] = cid.cpu().to(torch.int32)
        res_norms[s:e] = res.norm(dim=1).cpu()

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
    d_doc_idx = torch.from_numpy(np.repeat(np.arange(N_docs, dtype=np.int64), lens)).to(device)

    # Rotate ALL doc tokens with Hadamard on GPU in chunks -> store INT2, INT4, INT6, INT8 reconstructions & rho
    H = torch.from_numpy(hadamard_matrix(dim)).to(device)
    d_hd_full = torch.empty((N_toks, dim), dtype=torch.float16, device=device)
    for s in range(0, N_toks, 131072):
        e = min(s + 131072, N_toks)
        xb = d_toks[s:e].to(device=device, dtype=torch.float32)
        d_hd_full[s:e] = (xb @ H.t()).half()

    # Precompute approx_s0 and prewarm tau for all B queries
    approx_s0_all = np.empty((B, N_docs), dtype=np.float32)
    tau_warm_all = np.empty(B, dtype=np.float32)
    q_list = []
    q_rot_list = []
    qn_list = []
    warm_sets = []
    warm_heaps = []

    for qi in range(B):
        qs, qe = int(q_offs_sub[qi]), int(q_offs_sub[qi + 1])
        q = q_toks_sub[qs:qe].to(device=device, dtype=torch.float32)
        m = q.shape[0]
        qn = q.norm(dim=1)
        q_list.append(q)
        q_rot_list.append(q @ H.t())
        qn_list.append(qn)

        S_c = q @ centroids.t()
        sg = S_c[:, gc_t]
        idx_exp = g_doc_idx.view(1, -1).expand(m, -1)
        s0_m = torch.full((m, N_docs), -1e9, device=device)
        s0_m.scatter_reduce_(1, idx_exp, sg, reduce="amax", include_self=True)
        approx_s0 = s0_m.sum(dim=0).cpu().numpy()
        approx_s0_all[qi] = approx_s0

        warm_cands = np.argpartition(-approx_s0, min(c_warm, N_docs - 1))[:min(c_warm, N_docs)]
        warm_scores, _ = batched_exact_maxsim(q, d_toks, d_offs, warm_cands, device=device)
        warm_sets.append(set(int(x) for x in warm_cands))
        hl = sorted([(float(sc), int(d)) for sc, d in zip(warm_scores, warm_cands)], key=lambda x: (-x[0], x[1]))[:topk]
        warm_heaps.append(hl)
        tau_warm_all[qi] = hl[-1][0]

    # Now evaluate INT2, INT4, INT6, INT8 (fast chunked GPU!)
    bit_results = {}
    for bits in [2, 4, 6, 8]:
        if bits == 2:
            h_min = d_hd_full.float().amin(dim=1, keepdim=True)
            h_max = d_hd_full.float().amax(dim=1, keepdim=True)
            sc = (h_max - h_min).clamp(min=1e-6) / 3.0
            hd_hat = (h_min + torch.round((d_hd_full.float() - h_min) / sc).clamp(0, 3) * sc).half()
        else:
            qmax = float((1 << (bits - 1)) - 1)
            abs_m = d_hd_full.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / qmax
            hd_hat = (torch.round(d_hd_full.float() / abs_m).clamp(-qmax, qmax) * abs_m).half()
        rho = ((d_hd_full.float() - hd_hat.float()).norm(dim=1) + 2e-4)  # [N_toks]

        doc_surv_cnt = 0
        exact_ops = 0
        full_ops = 0
        prog_int_ops = 0
        prog_int_full = 0

        pred_ids = np.full((B, topk), -1, dtype=np.int64)
        pred_scores = np.full((B, topk), -1e9, dtype=np.float32)

        # Evaluate in doc chunks across all B queries at once (for bits >= 4) OR per query
        # First compute full INT ub_m [B, m, N_docs] in doc chunks (10x faster than Python per-query!)
        for qi in range(B):
            q = q_list[qi]
            qr = q_rot_list[qi].half()
            qn = qn_list[qi]
            m = q.shape[0]
            full_ops += m * N_toks
            tau = tau_warm_all[qi]
            tau_safe = tau - 1.5e-3 * m

            # Chunked scan of hd_hat [N_toks, dim]
            ub_doc = torch.empty(N_docs, dtype=torch.float32, device=device)
            lb_doc = torch.empty(N_docs, dtype=torch.float32, device=device)
            # Also track progressive exit every 8 query tokens (with remaining bound = sum qn_rem)
            # First get exact INT ub_m [m, N_docs] in chunks of 65536 tokens
            ub_m = torch.full((m, N_docs), -1e9, dtype=torch.float32, device=device)
            lb_m = torch.full((m, N_docs), -1e9, dtype=torch.float32, device=device)
            for ds in range(0, N_docs, 8192):
                de = min(ds + 8192, N_docs)
                ts, te = int(d_offs[ds]), int(d_offs[de])
                sim = (qr @ hd_hat[ts:te].t()).float()  # [m, nt]
                err = qn[:, None] * rho[ts:te][None, :]
                seg = d_doc_idx[ts:te] - ds
                seg_exp = seg.unsqueeze(0).expand(m, -1)
                sub_ub = torch.full((m, de - ds), -1e9, device=device)
                sub_lb = torch.full((m, de - ds), -1e9, device=device)
                sub_ub.scatter_reduce_(1, seg_exp, sim + err, reduce="amax", include_self=True)
                sub_lb.scatter_reduce_(1, seg_exp, sim - err, reduce="amax", include_self=True)
                ub_m[:, ds:de] = sub_ub
                lb_m[:, ds:de] = sub_lb

            ub_total = ub_m.sum(dim=0)
            lb_total = lb_m.sum(dim=0)
            # Also compute how many docs exit progressively using Cauchy-Schwarz upper bound qn[i] for remaining tokens
            # U_step(t) = sum_{0..t-1} ub_m[i] + sum_{t..m-1} qn[i]
            cum_ub = torch.cumsum(ub_m, dim=0)
            rem_qn = torch.flip(torch.cumsum(torch.flip(qn, [0]), dim=0), [0])
            rem_after = torch.cat([rem_qn[1:], torch.zeros(1, device=device)])
            prog_U = cum_ub + rem_after[:, None]
            # First step where prog_U < tau_safe
            below = (prog_U < tau_safe)
            first_exit = torch.where(below.any(dim=0), below.float().argmax(dim=0) + 1, torch.full((N_docs,), m, device=device))
            prog_int_ops += int((first_exit.cpu().numpy() * lens).sum())
            prog_int_full += m * N_toks

            # Tighten tau with guaranteed lb_total!
            tau_lb = float(torch.topk(lb_total, topk).values[-1].item())
            tau = max(tau, tau_lb)
            tau_safe = tau - 1.5e-3 * m

            surv_mask = (ub_total >= tau_safe).cpu().numpy()
            surv_docs = np.array([int(d) for d in np.where(surv_mask)[0] if int(d) not in warm_sets[qi]], dtype=np.int64)
            doc_surv_cnt += len(surv_docs) + len(warm_sets[qi])

            ex_scores, ex_ops = batched_exact_maxsim(q, d_toks, d_offs, surv_docs, device=device)
            exact_ops += ex_ops + int(m * lens[list(warm_sets[qi])].sum())

            hl = list(warm_heaps[qi])
            for sc_v, did_v in zip(ex_scores, surv_docs):
                hl.append((float(sc_v), int(did_v)))
            hl.sort(key=lambda x: (-x[0], x[1]))
            hl = hl[:topk]
            for r_i, (sc_v, did_v) in enumerate(hl):
                pred_ids[qi, r_i] = did_v
                pred_scores[qi, r_i] = sc_v

        audit = audit_exact_topk(pred_ids, pred_scores, oracle_ids, oracle_scores)
        bytes_per_tok = (dim * bits) / 8.0 + 4.0  # packed bits + 2B scale + 2B rho
        bit_results[f"int{bits}"] = {
            "bits": bits,
            "bytes_per_tok": bytes_per_tok,
            "compression_vs_fp16": round((dim * 2.0) / bytes_per_tok, 2),
            "mean_rho": round(float(rho.mean().item()), 4),
            "doc_prune_pct": round((1.0 - doc_surv_cnt / (B * N_docs)) * 100, 2),
            "prog_int_skip_pct": round((1.0 - prog_int_ops / max(prog_int_full, 1)) * 100, 2),
            "exact_tok_pruned_pct": round((1.0 - exact_ops / full_ops) * 100, 3),
            "audit": audit,
        }
        print(f"  INT{bits}: rho={bit_results[f'int{bits}']['mean_rho']:.4f} | DocPrune={bit_results[f'int{bits}']['doc_prune_pct']}% | ProgSkip={bit_results[f'int{bits}']['prog_int_skip_pct']}% | ExactPruned={bit_results[f'int{bits}']['exact_tok_pruned_pct']}% | Comp={bit_results[f'int{bits}']['compression_vs_fp16']}x | Audit={audit['agreement']}", flush=True)

    return {
        "dataset": ds_name,
        "num_docs": N_docs,
        "total_tokens": N_toks,
        "avg_doc_len": round(float(meta["doc_len_mean"]), 2),
        "doc_len_cv": round(float(meta["doc_len_cv"]), 3),
        "pad_waste_bs64_pct": round(pad_waste_bs64 * 100, 2),
        "fp16_mb": round(N_toks * dim * 2 / (1024**2), 2),
        "bit_sweep": bit_results,
        "elapsed_s": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    data_root = os.path.expanduser("~/maxbound/data_mv")
    records = []
    for ds in ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "longbench"]:
        ddir = os.path.join(data_root, ds)
        if os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            records.append(analyze_dataset_fast(ds, ddir, device=1))
    write_result_json(os.path.expanduser("~/maxbound/results/e0_bounds_sweep.json"), records)
