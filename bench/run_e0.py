import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/maxbound"))
from bench.common import write_result_json
from src.oracle import exact_maxsim_oracle
from src.maxbound_engine import hadamard_matrix_torch


def measure_bounds_on_dataset(ds_name, device=1, B=64):
    ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds_name}")
    d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
    d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
    q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
    q_offs_all = np.load(os.path.join(ddir, "query_offsets.int64.npy"))
    B_eff = min(B, len(q_offs_all) - 1)
    q_offs = q_offs_all[:B_eff + 1]
    q_sub = q_toks[:int(q_offs[-1])]

    meta_file = os.path.join(ddir, "meta.json")
    ds_meta = {}
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            ds_meta = json.load(f)

    N_docs = len(d_offs) - 1
    N_toks = d_toks.shape[0]
    D = d_toks.shape[1]
    lens = np.diff(d_offs)
    len_max = int(lens.max())

    stats = {
        "dataset": ds_name,
        "source": ds_meta.get("source", ds_name),
        "encoder": ds_meta.get("encoder", "colbertv2.0" if D == 128 else "BAAI/bge-m3"),
        "num_docs": int(N_docs),
        "total_doc_tokens": int(N_toks),
        "num_queries": int(len(q_offs_all) - 1),
        "dim": int(D),
        "doc_len_mean": float(np.mean(lens)),
        "doc_len_std": float(np.std(lens)),
        "doc_len_cv": float(np.std(lens) / max(np.mean(lens), 1e-9)),
        "doc_len_p50": float(np.percentile(lens, 50)),
        "doc_len_p95": float(np.percentile(lens, 95)),
        "doc_len_max": int(len_max),
        "b1_padded_tokens": int(N_docs * len_max),
        "b1_padding_waste_pct": float(100.0 * (1.0 - N_toks / max(N_docs * len_max, 1))),
        "fp16_gb": float(N_toks * D * 2 / (1024 ** 3)),
        "int8_gb": float(N_toks * (D + 8) / (1024 ** 3)),
        "int6_gb": float(N_toks * (D * 0.75 + 8) / (1024 ** 3)) if D == 128 else None,
        "int4_gb": float(N_toks * (D * 0.5 + 8) / (1024 ** 3)) if D == 128 else None,
    }

    if ds_name not in ("scifact", "nfcorpus", "arguana", "scidocs", "fiqa"):
        return {"stats": stats, "bounds": []}

    torch.cuda.set_device(device)
    oracle_ids, oracle_scores = exact_maxsim_oracle(
        q_sub.float(), q_offs, d_toks, d_offs, topk=10, doc_batch_size=2048, device=device, use_fp64=True
    )
    tau = torch.from_numpy(oracle_scores[:, -1]).to(device)
    q_lens = np.diff(q_offs).astype(np.int32)
    m_max = int(q_lens.max())
    q_pad = torch.zeros((B_eff, m_max, D), dtype=torch.float32, device=device)
    for qi in range(B_eff):
        qs, qe = int(q_offs[qi]), int(q_offs[qi + 1])
        q_pad[qi, :qe - qs] = q_sub[qs:qe].float()
    q_norms = q_pad.norm(dim=-1)
    q_mask = (torch.arange(m_max, device=device).unsqueeze(0) < torch.from_numpy(q_lens).to(device).unsqueeze(1))

    d_gpu = d_toks.to(device=device, dtype=torch.float32)
    doc_seg = torch.from_numpy(np.repeat(np.arange(N_docs, dtype=np.int64), lens)).to(device)
    d_means = torch.zeros((N_docs, D), dtype=torch.float32, device=device)
    d_means.index_add_(0, doc_seg, d_gpu)
    cnt = torch.from_numpy(lens).to(device=device, dtype=torch.float32).unsqueeze(1).clamp(min=1.0)
    d_centroids = d_means / cnt

    tok_diffs = d_gpu - d_centroids[doc_seg]
    tok_rad = tok_diffs.norm(dim=1)
    doc_rad = torch.zeros(N_docs, dtype=torch.float32, device=device)
    doc_rad.scatter_reduce_(0, doc_seg, tok_rad, reduce="amax", include_self=False)

    K = min(1024, N_docs)
    gen = torch.Generator(device=device).manual_seed(2027)
    perm = torch.randperm(N_toks, generator=gen, device=device)[:K]
    C = F.normalize(d_gpu[perm], p=2, dim=1)
    for _ in range(8):
        assign = (d_gpu[:min(N_toks, 131072)] @ C.t()).argmax(dim=1)
        new_C = torch.zeros_like(C)
        new_C.index_add_(0, assign, d_gpu[:min(N_toks, 131072)])
        c_cnt = torch.bincount(assign, minlength=K).float().unsqueeze(1).clamp(min=1.0)
        C = F.normalize(new_C / c_cnt, p=2, dim=1)

    km_assign = torch.empty(N_toks, dtype=torch.long, device=device)
    km_err = torch.empty(N_toks, dtype=torch.float32, device=device)
    for s in range(0, N_toks, 131072):
        e = min(s + 131072, N_toks)
        a = (d_gpu[s:e] @ C.t()).argmax(dim=1)
        km_assign[s:e] = a
        km_err[s:e] = (d_gpu[s:e] - C[a]).norm(dim=1)

    H = hadamard_matrix_torch(D, device)
    d_rot = d_gpu @ H.t()
    q_rot = q_pad @ H.t()

    bounds_res = []

    def eval_bound(name, bits_or_method):
        ub_all = torch.empty((B_eff, N_docs), dtype=torch.float32, device=device)
        if bits_or_method == "L0_CoarseBall":
            dots = torch.matmul(q_pad.view(B_eff * m_max, D), d_centroids.t()).view(B_eff, m_max, N_docs)
            ub_tok = dots + q_norms[:, :, None] * doc_rad[None, None, :]
            ub_tok.masked_fill_(~q_mask[:, :, None], 0.0)
            ub_all = ub_tok.sum(dim=1)
        elif bits_or_method == "KMeans_AABB":
            q_c = torch.matmul(q_pad.view(B_eff * m_max, D), C.t())
            for ds in range(0, N_docs, 2048):
                de = min(ds + 2048, N_docs)
                ts, te = int(d_offs[ds]), int(d_offs[de])
                tok_ub = q_c[:, km_assign[ts:te]] + q_norms.view(B_eff * m_max, 1) * km_err[ts:te].unsqueeze(0)
                tok_ub = tok_ub.view(B_eff, m_max, te - ts)
                seg = doc_seg[ts:te] - ds
                doc_m = torch.full((B_eff, m_max, de - ds), -1e9, device=device)
                doc_m.scatter_reduce_(2, seg.view(1, 1, -1).expand(B_eff, m_max, -1), tok_ub, reduce="amax", include_self=False)
                doc_m.masked_fill_(~q_mask[:, :, None], 0.0)
                ub_all[:, ds:de] = doc_m.sum(dim=1)
        elif bits_or_method == "PQ16":
            M_sub = 8
            d_sub = D // M_sub
            K_pq = 256
            d_pq_err = torch.zeros(N_toks, dtype=torch.float32, device=device)
            codes_all, Cb_list = [], []
            for m_i in range(M_sub):
                xs = d_gpu[:, m_i * d_sub:(m_i + 1) * d_sub]
                samp = xs[:min(N_toks, 65536)]
                perm_p = torch.randperm(samp.shape[0], generator=gen, device=device)[:K_pq]
                Cb = samp[perm_p].clone()
                for _ in range(6):
                    a = torch.cdist(samp, Cb).argmin(dim=1)
                    nCb = torch.zeros_like(Cb)
                    nCb.index_add_(0, a, samp)
                    c_c = torch.bincount(a, minlength=K_pq).float().unsqueeze(1).clamp(min=1.0)
                    Cb = nCb / c_c
                a_full = torch.empty(N_toks, dtype=torch.long, device=device)
                for s in range(0, N_toks, 65536):
                    e = min(s + 65536, N_toks)
                    aa = torch.cdist(xs[s:e], Cb).argmin(dim=1)
                    a_full[s:e] = aa
                    d_pq_err[s:e] += (xs[s:e] - Cb[aa]).norm(dim=1).pow(2)
                codes_all.append(a_full)
                Cb_list.append(Cb)
            d_pq_err = torch.sqrt(d_pq_err) + 1e-4
            for ds in range(0, N_docs, 2048):
                de = min(ds + 2048, N_docs)
                ts, te = int(d_offs[ds]), int(d_offs[de])
                tok_dot = torch.zeros((B_eff * m_max, te - ts), dtype=torch.float32, device=device)
                for m_i in range(M_sub):
                    qt_m = q_pad[:, :, m_i * d_sub:(m_i + 1) * d_sub].reshape(B_eff * m_max, d_sub)
                    lut = qt_m @ Cb_list[m_i].t()
                    tok_dot += lut[:, codes_all[m_i][ts:te]]
                tok_ub = (tok_dot + q_norms.view(B_eff * m_max, 1) * d_pq_err[ts:te].unsqueeze(0)).view(B_eff, m_max, te - ts)
                seg = doc_seg[ts:te] - ds
                doc_m = torch.full((B_eff, m_max, de - ds), -1e9, device=device)
                doc_m.scatter_reduce_(2, seg.view(1, 1, -1).expand(B_eff, m_max, -1), tok_ub, reduce="amax", include_self=False)
                doc_m.masked_fill_(~q_mask[:, :, None], 0.0)
                ub_all[:, ds:de] = doc_m.sum(dim=1)
        else:
            qmax = float((1 << (bits_or_method - 1)) - 1) if bits_or_method > 2 else 1.5
            sc = d_rot.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / qmax
            if bits_or_method == 2:
                qi = torch.clamp(torch.round(d_rot / sc - 0.5) + 0.5, -1.5, 1.5)
            else:
                qi = torch.round(d_rot / sc).clamp(-qmax, qmax)
            d_hat = qi * sc
            rho = (d_rot - d_hat).norm(dim=1) + 5e-4
            for ds in range(0, N_docs, 2048):
                de = min(ds + 2048, N_docs)
                ts, te = int(d_offs[ds]), int(d_offs[de])
                tok_ub = (torch.matmul(q_rot.view(B_eff * m_max, D), d_hat[ts:te].t()) +
                          q_norms.view(B_eff * m_max, 1) * rho[ts:te].unsqueeze(0)).view(B_eff, m_max, te - ts)
                seg = doc_seg[ts:te] - ds
                doc_m = torch.full((B_eff, m_max, de - ds), -1e9, device=device)
                doc_m.scatter_reduce_(2, seg.view(1, 1, -1).expand(B_eff, m_max, -1), tok_ub, reduce="amax", include_self=False)
                doc_m.masked_fill_(~q_mask[:, :, None], 0.0)
                ub_all[:, ds:de] = doc_m.sum(dim=1)

        surv = (ub_all >= (tau[:, None] - 1e-3))
        surv_pct = float(surv.float().mean().item() * 100.0)
        prune_pct = 100.0 - surv_pct
        o_t = torch.from_numpy(oracle_ids).to(device)
        hits = torch.gather(surv, 1, o_t).all().item()
        bounds_res.append({
            "method": name,
            "prune_pct": round(prune_pct, 4),
            "survivor_pct": round(surv_pct, 4),
            "oracle_100pct_preserved": bool(hits),
        })
        print(f"  [{ds_name}] {name:18s}: prune={prune_pct:6.2f}% | surv={surv_pct:6.2f}% | safe={hits}")

    eval_bound("L0_CoarseBall", "L0_CoarseBall")
    eval_bound("KMeans_AABB_1024", "KMeans_AABB")
    eval_bound("PQ16_8x16", "PQ16")
    eval_bound("Hadamard_INT2", 2)
    eval_bound("Hadamard_INT4", 4)
    eval_bound("Hadamard_INT6", 6)
    eval_bound("Hadamard_INT8", 8)
    return {"stats": stats, "bounds": bounds_res}


@torch.no_grad()
def measure_e0b_bound_slack_quantiles(device=1):
    """
    E0B: Empirical Validation of Theorem 1 (Zero Bound Violations & Tightness Slack Quantiles)
    across > 4,500,000 (query, document) pairs spanning 128D (SciFact, FiQA) and 1024D (SciFact-BGEM3).
    Measures slack(q, D) = (UB_actual_kernel(q, D) + eps_buf[q]) - ExactFP64MaxSim(q, D).
    """
    from src.maxbound_engine import MaxBoundIndex
    torch.cuda.set_device(device)
    slack_rows = []

    for ds in ["scifact", "fiqa", "scifact_bgem3"]:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            continue
        d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
        d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
        q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
        B = 64
        q_offs = np.load(os.path.join(ddir, "query_offsets.int64.npy"))[:B + 1]
        q_sub = q_toks[:int(q_offs[-1])]
        N_docs = len(d_offs) - 1
        D = d_toks.shape[1]

        # Compute exact IEEE-754 FP64 full [B, N_docs] score matrix on GPU (supports variable-length queries!)
        q_lens = np.diff(q_offs).astype(np.int32)
        m_max = int(np.max(q_lens))
        q_pad64 = torch.zeros((B, m_max, D), dtype=torch.float64, device=device)
        qt64_all = q_sub.to(device=device, dtype=torch.float64)
        for qi in range(B):
            qs_i, qe_i = int(q_offs[qi]), int(q_offs[qi + 1])
            q_pad64[qi, :qe_i - qs_i] = qt64_all[qs_i:qe_i]
        q_mask = (torch.arange(m_max, device=device).unsqueeze(0) < torch.from_numpy(q_lens).to(device).unsqueeze(1))
        q_f64 = q_pad64.view(B * m_max, D)
        exact_f64 = np.empty((B, N_docs), dtype=np.float64)
        lens = np.diff(d_offs)
        for ds_i in range(0, N_docs, 1024):
            de_i = min(ds_i + 1024, N_docs)
            ts, te = int(d_offs[ds_i]), int(d_offs[de_i])
            dc = d_toks[ts:te].to(device=device, dtype=torch.float64)
            sim = torch.matmul(q_f64, dc.t()).view(B, m_max, -1)
            seg = torch.from_numpy(np.repeat(np.arange(de_i - ds_i, dtype=np.int64), lens[ds_i:de_i])).to(device)
            dm = torch.full((B, m_max, de_i - ds_i), -1e9, dtype=torch.float64, device=device)
            dm.scatter_reduce_(2, seg.view(1, 1, -1).expand(B, m_max, -1), sim, reduce="amax", include_self=True)
            dm.masked_fill_(~q_mask[:, :, None], 0.0)
            exact_f64[:, ds_i:de_i] = dm.sum(dim=1).cpu().numpy()

        modes = ["int4", "int6", "int8"] if D == 128 else ["int8"]
        for mode in modes:
            eng = MaxBoundIndex(d_toks, d_offs, device=device, mode=mode, sort_by_length=False)
            _, _, m_pad, _, q_fp16_pad, q_pack, q_sc, q_n, eps_buf = eng._prepare_queries(q_sub, q_offs, pad_mult=32)
            ub = torch.zeros((B, N_docs), dtype=torch.float32, device=device)
            lb = torch.zeros((B, N_docs), dtype=torch.float32, device=device)
            act = torch.ones((B, N_docs), dtype=torch.uint8, device=device)
            tau_dummy = torch.full((B,), -1e9, dtype=torch.float32, device=device)
            rem_z = torch.zeros(B, dtype=torch.float32, device=device)
            eps_sc = float(eps_buf.max().item())
            for p in range(m_pad // 32):
                if mode == "int8":
                    eng._exec_int8_pass32(
                        q_pack, q_sc, q_n, tau_dummy, ub, lb, rem_z, act, p * 32, eps_sc,
                        1 if p == 0 else 0, 0, force_w1=False, q_fp16_pad=q_fp16_pad,
                        enable_2slice=False, eps_buf=eps_buf, is_packed_hd=False
                    )
                elif mode == "int6":
                    eng.ext.fused_tc_int6_pass32(q_pack, q_sc, q_n, eng.d_hi4, eng.d_lo2, eng.d_scales, eng.d_rho, eng.d_pad_offs_gpu, tau_dummy, ub, lb, rem_z, act, p * 32, 0.0, 1 if p == 0 else 0, 0, eps_buf)
                else:
                    eng.ext.fused_tc_int4_pass32(q_pack, q_sc, q_n, eng.d_qdata, eng.d_scales, eng.d_rho, eng.d_pad_offs_gpu, tau_dummy, ub, lb, rem_z, act, p * 32, 0.0, 1 if p == 0 else 0, 0, eps_buf)

            ub_safe_f64 = (ub + eps_buf[:, None]).cpu().numpy().astype(np.float64)
            lb_safe_f64 = (lb - eps_buf[:, None]).cpu().numpy().astype(np.float64)
            ub_slack = (ub_safe_f64 - exact_f64).reshape(-1)
            lb_slack = (exact_f64 - lb_safe_f64).reshape(-1)
            ub_violations = int(np.sum(ub_slack < 0.0))
            lb_violations = int(np.sum(lb_slack < 0.0))
            assert ub_violations == 0 and lb_violations == 0, f"Bound violation on {ds} {mode}: ub_v={ub_violations}, lb_v={lb_violations}"
            mean_exact = float(np.mean(np.abs(exact_f64)))
            row = {
                "dataset": ds, "dim": D, "mode": mode,
                "num_qd_pairs_audited": int(ub_slack.size),
                "ub_violations": ub_violations,
                "lb_violations": lb_violations,
                "min_ub_slack": round(float(np.min(ub_slack)), 6),
                "p10_ub_slack": round(float(np.percentile(ub_slack, 10)), 4),
                "p50_ub_slack": round(float(np.percentile(ub_slack, 50)), 4),
                "p95_ub_slack": round(float(np.percentile(ub_slack, 95)), 4),
                "p99_ub_slack": round(float(np.percentile(ub_slack, 99)), 4),
                "relative_p50_slack_pct": round(float(np.percentile(ub_slack, 50) / max(mean_exact, 1e-6) * 100.0), 2),
            }
            slack_rows.append(row)
            print(f"  [E0B Slack {ds:13s} {mode}] pairs={ub_slack.size:,} | violations=0 | min={row['min_ub_slack']:.4f} | p50={row['p50_ub_slack']:.3f} ({row['relative_p50_slack_pct']:.1f}%) | p95={row['p95_ub_slack']:.3f}")
            del eng
            torch.cuda.empty_cache()

    return slack_rows


@torch.no_grad()
def measure_e0c_exactness_value_rare_entity(device=1):
    """
    E0C: Why Exactness Matters — Tail / Rare-Token / Low-Centroid-Density Failure Modes of Approximate Indexes.
    Stratifies official evaluation queries on FiQA and SCIDOCS into Head (top-20% closest to corpus KMeans centroids)
    vs. Tail / OOD Rare-Token queries (bottom-20% farthest from coarse centroids, e.g., rare tickers/entities/numbers),
    comparing Oracle Top-10 Recall & nDCG@10 across PLAID, Official FastPLAID, WARP, EMVB, MUVERA, and MAXBOUND.
    """
    from src.maxbound_engine import MaxBoundIndex
    from src.baselines.baselines import B4OfficialFastPlaid, B4PlaidGPU, B5MuveraGPU, B9WarpGPU, B10EmvbGPU
    from bench.common import compute_ir_metrics

    torch.cuda.set_device(device)
    rows = []
    for ds in ["fiqa", "scidocs"]:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            continue
        d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
        d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
        q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
        q_offs_all = np.load(os.path.join(ddir, "query_offsets.int64.npy"))
        with open(os.path.join(ddir, "qrels.json")) as f:
            qrels = json.load(f)

        nq = min(256, len(q_offs_all) - 1)
        q_offs = q_offs_all[:nq + 1]
        q_sub = q_toks[:int(q_offs[-1])]
        oracle_ids, _ = exact_maxsim_oracle(q_sub.float(), q_offs, d_toks, d_offs, topk=10, device=device, use_fp64=True)

        b4 = B4PlaidGPU(d_toks, d_offs, device=device, n_centroids=1024)
        # Measure each query's mean max-centroid similarity (low = rare/OOD entity tokens poorly covered by coarse centroids)
        q_cent_affinity = np.zeros(nq, dtype=np.float32)
        qt_gpu = q_sub.to(device=device, dtype=torch.float16)
        csim_all = (qt_gpu @ b4.centroids.t()).float().amax(dim=1).cpu().numpy()
        for qi in range(nq):
            q_cent_affinity[qi] = float(np.mean(csim_all[int(q_offs[qi]):int(q_offs[qi + 1])]))
        order_aff = np.argsort(q_cent_affinity)
        k_tail = max(16, nq // 5)
        tail_q = set(int(x) for x in order_aff[:k_tail])
        head_q = set(int(x) for x in order_aff[-k_tail:])

        def _eval_method(mname, fn):
            preds = []
            for s in range(0, nq, 64):
                e = min(s + 64, nq)
                so = q_offs[s:e + 1] - q_offs[s]
                st = q_sub[int(q_offs[s]):int(q_offs[e])]
                preds.append(fn(st, so)[0])
            pred_all = np.concatenate(preds, axis=0)

            def _rec_subset(qset):
                vals = []
                for qi in sorted(qset):
                    oset = set(int(x) for x in oracle_ids[qi] if x >= 0)
                    pset = set(int(x) for x in pred_all[qi] if x >= 0)
                    vals.append(len(oset & pset) / max(len(oset), 1))
                return round(float(np.mean(vals) * 100.0), 2)

            tail_qrels = {str(i): qrels[str(q)] for i, q in enumerate(sorted(tail_q)) if str(q) in qrels}
            tail_preds = pred_all[sorted(tail_q)]
            tail_ir = compute_ir_metrics(tail_preds, tail_qrels, k=10) if tail_qrels else {}
            r_all = _rec_subset(set(range(nq)))
            r_head = _rec_subset(head_q)
            r_tail = _rec_subset(tail_q)
            row = {
                "dataset": ds, "method": mname,
                "overall_top10_agreement_pct": r_all,
                "head_queries_top10_agreement_pct": r_head,
                "tail_rare_queries_top10_agreement_pct": r_tail,
                "tail_recall_drop_pct": round(r_head - r_tail, 2),
                "tail_ndcg_at_10": tail_ir.get("ndcg@10", 0.0),
            }
            rows.append(row)
            print(f"  [E0C {ds:8s} {mname:22s}] Overall={r_all:6.2f}% | Head={r_head:6.2f}% | Tail(OOD)={r_tail:6.2f}% (drop={r_head-r_tail:+5.2f}%)")

        _eval_method("B4_PLAID_GPU", lambda qt, qo: b4.search(qt, qo, nprobe=2, ncands=256, topk=10))
        del b4; torch.cuda.empty_cache()

        fp = B4OfficialFastPlaid(d_toks, d_offs, device=device, nbits=2, kmeans_niters=3)
        _eval_method("B4_Official_FastPLAID", lambda qt, qo: fp.search(qt, qo, n_ivf_probe=2, n_full_scores=256, topk=10))
        fp.cleanup(); del fp; torch.cuda.empty_cache()

        b9 = B9WarpGPU(d_toks, d_offs, device=device, n_centroids=1024)
        _eval_method("B9_WARP_GPU", lambda qt, qo: b9.search(qt, qo, t_prime=4, topk=10))
        del b9; torch.cuda.empty_cache()

        b10 = B10EmvbGPU(d_toks, d_offs, device=device, n_centroids=1024)
        _eval_method("B10_EMVB_GPU", lambda qt, qo: b10.search(qt, qo, nprobe=4, ncands=512, nrefine=64, topk=10))
        del b10; torch.cuda.empty_cache()

        b5 = B5MuveraGPU(d_toks, d_offs, device=device, r_reps=8, k_sim=4, d_proj=16)
        _eval_method("B5_MUVERA_GPU", lambda qt, qo: b5.search(qt, qo, ncands=256, topk=10))
        del b5; torch.cuda.empty_cache()

        mb = MaxBoundIndex(d_toks, d_offs, device=device, mode="int8")
        _eval_method("MAXBOUND_INT8", lambda qt, qo: mb.search(qt, qo, topk=10, c_warm=32)[:2])
        del mb; torch.cuda.empty_cache()

    return rows


def main():
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    datasets = [
        "scifact", "nfcorpus", "trec_covid", "arguana", "longbench",
        "scidocs", "fiqa", "hotpotqa", "msmarco", "scale_50m", "msmarco_2m",
        "scifact_bgem3", "fiqa_bgem3",
    ]
    records = []
    for ds in datasets:
        p = os.path.expanduser(f"~/maxbound/data_mv/{ds}/doc_offsets.int64.npy")
        if not os.path.exists(p):
            print(f"Skipping {ds} (not built yet)")
            continue
        print(f"\n=== E0 Motivation & Corpus Stats: {ds} ===")
        records.append(measure_bounds_on_dataset(ds, device=dev, B=64))

    print("\n=== E0B Theorem 1 Empirical Bound Safety & Slack Quantiles ===")
    slack_rows = measure_e0b_bound_slack_quantiles(device=dev)
    print("\n=== E0C Exactness Value on Rare-Entity / Tail Queries ===")
    e0c_rows = measure_e0c_exactness_value_rare_entity(device=dev)
    total_pairs = sum(r["num_qd_pairs_audited"] for r in slack_rows)
    out_path = os.path.expanduser("~/maxbound/results/e0_motivation.json")
    write_result_json(out_path, records, {
        "experiment": "E0_motivation_padding_bounds_slack_and_exactness_value",
        "device": dev,
        "total_qd_pairs_slack_audited": total_pairs,
        "total_bound_violations": 0,
        "bound_slack_quantiles": slack_rows,
        "e0c_rare_entity_exactness_value": e0c_rows,
    })
    print(f"\nSaved {out_path} ({len(records)} datasets, {total_pairs:,} pairs slack-audited with 0 violations)")


if __name__ == "__main__":
    main()
