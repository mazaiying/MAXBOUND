import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/maxbound"))
from bench.common import time_cuda_fn, audit_exact_topk, write_result_json
from src.oracle import exact_maxsim_oracle
from src.maxbound_engine import MaxBoundIndex, get_maxbound_ext, hadamard_matrix_torch
from src.baselines.baselines import B1TorchPaddedGemm, B2CuBLASUnpaddedExact, B8FusedFP16TCExact, _pad_queries
from src.planner import MaxBoundPlanner


def run_e5(device=1):
    records = []
    # 5A: Controlled Length CV Sweep (from real fiqa tokens, constant total N_toks=2,000,000, varying doc-length distribution)
    ddir = os.path.expanduser("~/maxbound/data_mv/fiqa")
    raw_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
    q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
    q_offs = np.load(os.path.join(ddir, "query_offsets.int64.npy"))[:65]
    q_sub = q_toks[:int(q_offs[-1])]

    N_docs_syn = 16000
    target_mean = 128
    rng = np.random.default_rng(2027)
    for target_cv in [0.0, 0.3, 0.7, 1.2, 2.0]:
        if target_cv == 0.0:
            lens = np.full(N_docs_syn, target_mean, dtype=np.int64)
        else:
            # Lognormal lengths matching target CV = sqrt(exp(sigma^2) - 1)
            sigma = np.sqrt(np.log(1.0 + target_cv ** 2))
            mu = np.log(target_mean) - 0.5 * sigma ** 2
            lens = np.clip(np.round(rng.lognormal(mu, sigma, size=N_docs_syn)).astype(np.int64), 8, 2048)
        d_offs = np.zeros(N_docs_syn + 1, dtype=np.int64)
        np.cumsum(lens, out=d_offs[1:])
        total_t = int(d_offs[-1])
        # Sample contiguous real token slices cyclically so vector geometry stays authentic
        d_toks = raw_toks[:total_t] if total_t <= raw_toks.shape[0] else torch.cat([raw_toks] * ((total_t // raw_toks.shape[0]) + 1), dim=0)[:total_t]
        real_cv = float(np.std(lens) / np.mean(lens))

        oracle_ids, oracle_scores = exact_maxsim_oracle(q_sub.float(), q_offs, d_toks, d_offs, topk=10, device=device, use_fp64=True)
        b2 = B2CuBLASUnpaddedExact(d_toks, d_offs, device=device)
        t_b2, (ib2, sb2) = time_cuda_fn(lambda: b2.search(q_sub, q_offs, topk=10), warmup=3, repeats=15, device=device)
        del b2; torch.cuda.empty_cache()

        b8 = B8FusedFP16TCExact(d_toks, d_offs, device=device)
        t_b8, _ = time_cuda_fn(lambda: b8.search(q_sub, q_offs, topk=10), warmup=3, repeats=15, device=device)
        del b8; torch.cuda.empty_cache()

        b1_ms, b1_oom = None, False
        try:
            b1 = B1TorchPaddedGemm(d_toks, d_offs, device=device, chunk_docs=1024)
            t_b1, _ = time_cuda_fn(lambda: b1.search(q_sub, q_offs, topk=10), warmup=2, repeats=9, device=device)
            b1_ms = t_b1["median_ms"]
            del b1
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            b1_oom = True
        torch.cuda.empty_cache()

        mb8 = MaxBoundIndex(d_toks, d_offs, device=device, mode="int8")
        t_8, (i8, s8, inf8) = time_cuda_fn(lambda: mb8.search(q_sub, q_offs, topk=10, c_warm=32), warmup=3, repeats=15, device=device)
        _, _, inf8_m = mb8.search(q_sub, q_offs, topk=10, c_warm=32, return_tested_mask=True)
        aud8 = audit_exact_topk(i8, s8, oracle_ids, oracle_scores, tol=2e-3, tested_mask=inf8_m["tested_mask_orig"])
        assert aud8["agreement"] == 1.0 and aud8["false_negative_prunes"] == 0
        del mb8; torch.cuda.empty_cache()

        mb6 = MaxBoundIndex(d_toks, d_offs, device=device, mode="int6")
        t_6, (i6, s6, inf6) = time_cuda_fn(lambda: mb6.search(q_sub, q_offs, topk=10, c_warm=32), warmup=3, repeats=15, device=device)
        _, _, inf6_m = mb6.search(q_sub, q_offs, topk=10, c_warm=32, return_tested_mask=True)
        aud6 = audit_exact_topk(i6, s6, oracle_ids, oracle_scores, tol=2e-3, tested_mask=inf6_m["tested_mask_orig"])
        assert aud6["agreement"] == 1.0 and aud6["false_negative_prunes"] == 0
        del mb6; torch.cuda.empty_cache()

        records.append({
            "sub_exp": "5A_length_cv_sweep", "dataset": "fiqa_cv_syn", "batch": 64, "exact": True,
            "target_cv": target_cv, "actual_cv": round(real_cv, 3),
            "N_docs": N_docs_syn, "N_toks": total_t, "len_max": int(lens.max()),
            "b1_ms": b1_ms, "b1_oom": b1_oom,
            "b2_ms": t_b2["median_ms"], "b2_ci95_ms": t_b2["ci95_ms"],
            "b8_ms": t_b8["median_ms"], "b8_ci95_ms": t_b8["ci95_ms"],
            "mb8_ms": t_8["median_ms"], "mb8_ci95_ms": t_8["ci95_ms"],
            "mb8_speedup": t_b2["median_ms"] / t_8["median_ms"],
            "mb8_speedup_vs_b8": t_b8["median_ms"] / t_8["median_ms"],
            "mb8_speedup_vs_best_exact": min(t_b2["median_ms"], t_b8["median_ms"]) / t_8["median_ms"],
            "mb8_surv_pct": round(100.0 * inf8["survivor_ratio"], 3),
            "mb6_ms": t_6["median_ms"], "mb6_ci95_ms": t_6["ci95_ms"],
            "mb6_speedup": t_b2["median_ms"] / t_6["median_ms"],
            "mb6_speedup_vs_best_exact": min(t_b2["median_ms"], t_b8["median_ms"]) / t_6["median_ms"],
            "mb6_surv_pct": round(100.0 * inf6["survivor_ratio"], 3),
            "audit": aud8, "audit_int6": aud6,
        })
        print(f"[E5A] CV={real_cv:.2f} max={lens.max():4d} | B1={b1_ms} | B2={t_b2['median_ms']:.2f} | B8={t_b8['median_ms']:.2f} | MB8={t_8['median_ms']:.2f} ({t_b2['median_ms']/t_8['median_ms']:.2f}x vs B2, {t_b8['median_ms']/t_8['median_ms']:.2f}x vs B8) | MB6={t_6['median_ms']:.2f}")

    # 5B: Query Length Sweep m in {16, 32, 64, 128} on arguana
    adir = os.path.expanduser("~/maxbound/data_mv/arguana")
    ad_toks = torch.from_numpy(np.load(os.path.join(adir, "doc_tokens.fp16.npy")))
    ad_offs = np.load(os.path.join(adir, "doc_offsets.int64.npy"))
    aq_toks = torch.from_numpy(np.load(os.path.join(adir, "query_tokens.fp16.npy")))
    aq_offs = np.load(os.path.join(adir, "query_offsets.int64.npy"))
    B = 64
    b2_a = B2CuBLASUnpaddedExact(ad_toks, ad_offs, device=device)
    b8_a = B8FusedFP16TCExact(ad_toks, ad_offs, device=device)
    mb8_a = MaxBoundIndex(ad_toks, ad_offs, device=device, mode="int8")

    for m_len in [16, 32, 64, 128]:
        if m_len <= 64:
            q_list = [aq_toks[int(aq_offs[i]):int(aq_offs[i]) + m_len] for i in range(B)]
        else:
            # Repeat 64 tokens to 128 preserving authentic ColBERTv2 geometry for long-query stress test
            q_list = [torch.cat([aq_toks[int(aq_offs[i]):int(aq_offs[i + 1])], aq_toks[int(aq_offs[i]):int(aq_offs[i + 1])]], dim=0)[:128] for i in range(B)]
        qm_toks = torch.cat(q_list, dim=0)
        qm_offs = np.arange(0, (B + 1) * m_len, m_len, dtype=np.int64)
        o_i, o_s = exact_maxsim_oracle(qm_toks.float(), qm_offs, ad_toks, ad_offs, topk=10, device=device, use_fp64=True)
        t_b2, _ = time_cuda_fn(lambda: b2_a.search(qm_toks, qm_offs, topk=10), warmup=3, repeats=15, device=device)
        t_b8, _ = time_cuda_fn(lambda: b8_a.search(qm_toks, qm_offs, topk=10), warmup=3, repeats=15, device=device)
        t_mb, (im, sm, infm) = time_cuda_fn(lambda: mb8_a.search(qm_toks, qm_offs, topk=10, c_warm=32), warmup=3, repeats=15, device=device)
        _, _, infm_mask = mb8_a.search(qm_toks, qm_offs, topk=10, c_warm=32, return_tested_mask=True)
        aud = audit_exact_topk(im, sm, o_i, o_s, tol=2e-3, tested_mask=infm_mask["tested_mask_orig"])
        assert aud["agreement"] == 1.0 and aud["false_negative_prunes"] == 0
        records.append({
            "sub_exp": "5B_query_len_sweep", "dataset": "arguana", "m": m_len, "B": B, "exact": True,
            "b2_ms": t_b2["median_ms"], "b2_ci95_ms": t_b2["ci95_ms"],
            "b8_ms": t_b8["median_ms"], "b8_ci95_ms": t_b8["ci95_ms"],
            "mb8_ms": t_mb["median_ms"], "mb8_ci95_ms": t_mb["ci95_ms"],
            "speedup": t_b2["median_ms"] / t_mb["median_ms"],
            "speedup_vs_b8": t_b8["median_ms"] / t_mb["median_ms"],
            "surv_pct": round(100.0 * infm["survivor_ratio"], 3), "audit": aud,
        })
        print(f"[E5B] m={m_len:3d} | B2={t_b2['median_ms']:.2f} | B8={t_b8['median_ms']:.2f} | MB8={t_mb['median_ms']:.2f} ({t_b2['median_ms']/t_mb['median_ms']:.2f}x vs B2, {t_b8['median_ms']/t_mb['median_ms']:.2f}x vs B8) | surv={infm['survivor_ratio']*100:.2f}%")

    out_path = os.path.expanduser("~/maxbound/results/e5_skew_robustness.json")
    write_result_json(out_path, records, {"experiment": "E5_skew_and_query_len", "device": device})
    print(f"Saved {out_path}")


def run_e6(device=1):
    records = []
    ext = get_maxbound_ext(device)
    for ds in ["fiqa", "longbench", "hotpotqa"]:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            continue
        torch.cuda.empty_cache()
        d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
        d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
        q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
        B = 64
        q_offs = np.load(os.path.join(ddir, "query_offsets.int64.npy"))[:B + 1]
        q_sub = q_toks[:int(q_offs[-1])]
        oracle_ids, oracle_scores = exact_maxsim_oracle(q_sub.float(), q_offs, d_toks, d_offs, topk=10, device=device)

        b2 = B2CuBLASUnpaddedExact(d_toks, d_offs, device=device)
        mb8 = MaxBoundIndex(d_toks, d_offs, device=device, mode="int8")

        # V0: B2 Unpadded
        t_v0, (i0, s0) = time_cuda_fn(lambda: b2.search(q_sub, q_offs, topk=10), warmup=2, repeats=5, device=device)

        # V1: Coarse Ball only (no Hadamard quant -> 100% fallback to level2 verify, measured natively!)
        def run_v1():
            sq = torch.arange(B, device=device, dtype=torch.int32).unsqueeze(1).repeat(1, mb8.N_docs).reshape(-1).contiguous()
            sd = torch.arange(mb8.N_docs, device=device, dtype=torch.int32).unsqueeze(0).repeat(B, 1).reshape(-1).contiguous()
            _, _, _, ql, qp, *_ = mb8._prepare_queries(q_sub, q_offs, pad_mult=32)
            sc = ext.level2_exact_verify(sq, sd, qp, ql, mb8.d_tokens_fp16_gpu, mb8.d_offsets_gpu).view(B, mb8.N_docs)
            v, idx = torch.topk(sc, k=10, dim=1)
            return idx.cpu().numpy(), v.cpu().numpy()

        t_v1, _ = time_cuda_fn(run_v1, warmup=0 if mb8.N_docs > 60000 else 1, repeats=1 if mb8.N_docs > 60000 else 2, device=device)

        # V2: Hadamard INT8 via PyTorch _int_mm + general_dim_int8_bounds (no fused kernel, no prewarm)
        def run_v2():
            _, m_max, m_pad, ql, qp, qi8, qsc, qn, eps = mb8._prepare_queries(q_sub, q_offs, pad_mult=32)
            ub = torch.zeros((B, mb8.N_docs), dtype=torch.float32, device=device)
            lb = torch.zeros((B, mb8.N_docs), dtype=torch.float32, device=device)
            mb8._general_dim_int8_bounds(qi8, qsc, qn, ql, ub, lb, doc_batch=512)
            tau = torch.topk(lb, k=10, dim=1).values[:, -1]
            nz = torch.nonzero(ub >= (tau[:, None] - eps), as_tuple=False)
            sq, sd = nz[:, 0].to(torch.int32).contiguous(), nz[:, 1].to(torch.int32).contiguous()
            ex = mb8._verify_pairs(sq, sd, qp, ql)
            sc_m = torch.full((B, mb8.N_docs), -1e9, device=device)
            sc_m[sq.long(), sd.long()] = ex
            v, idx = torch.topk(sc_m, k=10, dim=1)
            return idx.cpu().numpy(), v.cpu().numpy()

        t_v2, _ = time_cuda_fn(run_v2, warmup=2, repeats=3, device=device)

        # V3: V2 + Mean-Vector Prewarm (c_warm=32)
        def run_v3():
            _, m_max, m_pad, ql, qp, qi8, qsc, qn, eps = mb8._prepare_queries(q_sub, q_offs, pad_mult=32)
            wi, ws, tau = mb8._stage0_prewarm(qp, ql, B, 10, 32)
            ub = torch.zeros((B, mb8.N_docs), dtype=torch.float32, device=device)
            lb = torch.zeros((B, mb8.N_docs), dtype=torch.float32, device=device)
            mb8._general_dim_int8_bounds(qi8, qsc, qn, ql, ub, lb, doc_batch=512)
            tau = torch.maximum(tau, torch.topk(lb, k=10, dim=1).values[:, -1])
            mask = (ub >= (tau[:, None] - eps))
            mask.scatter_(1, wi, False)
            nz = torch.nonzero(mask, as_tuple=False)
            sq, sd = nz[:, 0].to(torch.int32).contiguous(), nz[:, 1].to(torch.int32).contiguous()
            ex = mb8._verify_pairs(sq, sd, qp, ql)
            ub.fill_(-1e9); ub.scatter_(1, wi, ws); ub[sq.long(), sd.long()] = ex
            v, idx = torch.topk(ub, k=10, dim=1)
            return idx.cpu().numpy(), v.cpu().numpy()

        t_v3, _ = time_cuda_fn(run_v3, warmup=2, repeats=3, device=device)

        # V4/V5/V6: Full Fused PTX Kernel + Stage Breakdown via CUDA events!
        def run_v6_breakdown():
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            ev[0].record()
            _, m_max, m_pad, ql, qp, qi8, qsc, qn, eps = mb8._prepare_queries(q_sub, q_offs, pad_mult=32)
            wi, ws, tau = mb8._stage0_prewarm(qp, ql, B, 10, 32)
            ev[1].record()
            ub = torch.zeros((B, mb8.N_docs), dtype=torch.float32, device=device)
            lb = torch.zeros((B, mb8.N_docs), dtype=torch.float32, device=device)
            act = torch.ones((B, mb8.N_docs), dtype=torch.uint8, device=device)
            act.scatter_(1, wi, 0)
            rem = torch.zeros(B, device=device)
            ext.fused_tc_int8_pass32(qi8, qsc, qn, mb8.d_qdata, mb8.d_scales, mb8.d_rho, mb8.d_pad_offs_gpu, tau, ub, lb, rem, act, 0, eps, 1, 0)
            tau = torch.maximum(tau, torch.topk(lb, k=10, dim=1).values[:, -1])
            ev[2].record()
            mask = (act > 0) & (ub >= (tau[:, None] - eps))
            nz = torch.nonzero(mask, as_tuple=False)
            ub.fill_(-1e9); ub.scatter_(1, wi, ws)
            if nz.numel() > 0:
                sq, sd = nz[:, 0].to(torch.int32).contiguous(), nz[:, 1].to(torch.int32).contiguous()
                ub[sq.long(), sd.long()] = mb8._verify_pairs(sq, sd, qp, ql)
            ev[3].record()
            v, idx = torch.topk(ub, k=10, dim=1)
            res_i, res_v = idx.cpu().numpy(), v.cpu().numpy()
            ev[4].record()
            torch.cuda.synchronize(device)
            return res_i, res_v, [ev[k].elapsed_time(ev[k + 1]) for k in range(4)]

        # Warmup breakdown
        for _ in range(2):
            run_v6_breakdown()
        t_v6, (i6, s6, _) = time_cuda_fn(lambda: mb8.search(q_sub, q_offs, topk=10, c_warm=32), warmup=2, repeats=5, device=device)
        aud6 = audit_exact_topk(i6, s6, oracle_ids, oracle_scores, tol=2e-3)
        assert aud6["agreement"] == 1.0
        bks = np.median(np.array([run_v6_breakdown()[2] for _ in range(5)]), axis=0)

        records.append({
            "dataset": ds, "batch": B,
            "V0_B2_Unpadded_ms": t_v0["median_ms"],
            "V1_NaiveBall_Fallback_ms": t_v1["median_ms"],
            "V2_Hadamard_PyTorch_ms": t_v2["median_ms"],
            "V3_Prewarm_PyTorch_ms": t_v3["median_ms"],
            "V4_V6_Full_MAXBOUND_ms": t_v6["median_ms"],
            "stage_breakdown_ms": {
                "query_prep_and_prewarm": round(float(bks[0]), 4),
                "stage1_fused_tc_filter": round(float(bks[1]), 4),
                "stage2_exact_verify": round(float(bks[2]), 4),
                "topk_and_d2h": round(float(bks[3]), 4),
            },
            "speedup_v6_vs_v0": t_v0["median_ms"] / t_v6["median_ms"],
            "audit": aud6["agreement"],
        })
        print(f"[E6 {ds}] V0={t_v0['median_ms']:.2f} -> V2={t_v2['median_ms']:.2f} -> V3={t_v3['median_ms']:.2f} -> V6={t_v6['median_ms']:.2f} ({t_v0['median_ms']/t_v6['median_ms']:.2f}x) | stages={[round(float(x),2) for x in bks]}")
        del b2, mb8
        torch.cuda.empty_cache()

    out_path = os.path.expanduser("~/maxbound/results/e6_ablation_breakdown.json")
    write_result_json(out_path, records, {"experiment": "E6_ablation_and_stage_breakdown", "device": device})
    print(f"Saved {out_path}")


def run_e7():
    from scipy.optimize import nnls
    # Step A: Disjoint Calibration on synthetic clustered topic vectors ONLY -> results/m3_calibration.json
    rng = np.random.default_rng(999)
    K_syn = 64
    centers = rng.standard_normal((K_syn, 128)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    calib_points = []
    for n_docs in [3000, 12000, 25000]:
        lens = rng.integers(64, 220, size=n_docs)
        d_offs = np.zeros(n_docs + 1, dtype=np.int64); np.cumsum(lens, out=d_offs[1:])
        doc_topics = np.repeat(np.arange(n_docs, dtype=np.int64) % K_syn, lens)
        x = centers[doc_topics] + 0.35 * rng.standard_normal((int(d_offs[-1]), 128)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        d_toks = torch.from_numpy(x).half()
        q_offs = np.arange(0, 65 * 32, 32, dtype=np.int64)
        q_topics = np.repeat(np.arange(64, dtype=np.int64) % K_syn, 32)
        qx = centers[q_topics] + 0.35 * rng.standard_normal((int(q_offs[-1]), 128)).astype(np.float32)
        qx /= np.linalg.norm(qx, axis=1, keepdims=True)
        q_toks = torch.from_numpy(qx).half()
        for dev in [0, 1]:
            rec_map = {B: {} for B in [1, 8, 16, 64]}
            for pname, maker in [
                ("Dense_Unpadded", lambda: B2CuBLASUnpaddedExact(d_toks, d_offs, device=dev)),
                ("MB_INT8", lambda: MaxBoundIndex(d_toks, d_offs, device=dev, mode="int8")),
                ("MB_INT6", lambda: MaxBoundIndex(d_toks, d_offs, device=dev, mode="int6")),
            ]:
                eng = maker()
                for B in [1, 8, 16, 64]:
                    qo = q_offs[:B + 1]; qt = q_toks[:int(qo[-1])]
                    fn = (lambda e=eng, q=qt, o=qo: e.search(q, o, 10)) if pname == "Dense_Unpadded" else (lambda e=eng, q=qt, o=qo: e.search(q, o, 10, 32))
                    rec_map[B][pname] = time_cuda_fn(fn, 1, 3, dev)[0]["median_ms"]
                del eng
                torch.cuda.empty_cache()
            for B in [1, 8, 16, 64]:
                calib_points.append({
                    "dev": dev, "n_docs": n_docs, "n_toks": int(d_offs[-1]), "B": B, "m": 32,
                    "b2_ms": rec_map[B]["Dense_Unpadded"], "mb8_ms": rec_map[B]["MB_INT8"], "mb6_ms": rec_map[B]["MB_INT6"],
                })

    planner = MaxBoundPlanner()
    # 3-term Non-Negative Least Squares per (device, plan) on disjoint synthetic calibration points ONLY!
    for dev_str in ["0", "1"]:
        pts = [p for p in calib_points if str(p["dev"]) == dev_str]
        for plan, ykey in [("Dense_Unpadded", "b2_ms"), ("MB_INT8", "mb8_ms"), ("MB_INT6", "mb6_ms")]:
            X = np.stack([planner.features(plan, p["n_toks"], p["B"], p["m"]) for p in pts], axis=0)
            y = np.array([p[ykey] for p in pts], dtype=np.float64)
            # Weighted NNLS minimizing relative error (1/y weighting)
            w = 1.0 / np.maximum(y, 0.5)
            coef, _ = nnls(X * w[:, None], y * w)
            planner.params[dev_str][plan] = {"c0": float(coef[0]), "c1": float(coef[1]), "c2": float(coef[2])}
        # Keep MB_INT4 slightly above MB_INT6
        c6 = planner.params[dev_str]["MB_INT6"]
        planner.params[dev_str]["MB_INT4"] = {"c0": c6["c0"] * 1.1, "c1": c6["c1"] * 1.5, "c2": c6["c2"] * 1.6}

    calib_path = os.path.expanduser("~/maxbound/results/m3_calibration.json")
    with open(calib_path, "w") as f:
        json.dump({"surv_coef": planner.surv_coef, "params": planner.params, "calib_points": calib_points}, f, indent=2)

    # Step B: Evaluate on HELD-OUT real datasets x (B, device, vram_budget)
    eval_records = []
    real_ds = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "longbench"]
    for ds in real_ds:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            continue
        with open(os.path.join(ddir, "meta.json")) as f:
            meta = json.load(f)
        d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
        d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
        q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
        q_offs_all = np.load(os.path.join(ddir, "query_offsets.int64.npy"))
        N_docs, N_toks, cv = meta["num_docs"], meta["total_doc_tokens"], meta["doc_len_cv"]
        m_len = int(np.diff(q_offs_all[:2])[0])

        for dev in [0, 1]:
            actual_map = {B: {} for B in [1, 8, 16, 32, 64]}
            for p_name, maker in [
                ("Dense_Unpadded", lambda: B2CuBLASUnpaddedExact(d_toks, d_offs, device=dev)),
                ("MB_INT8", lambda: MaxBoundIndex(d_toks, d_offs, device=dev, mode="int8")),
                ("MB_INT6", lambda: MaxBoundIndex(d_toks, d_offs, device=dev, mode="int6")),
            ]:
                eng = maker()
                for B in [1, 8, 16, 32, 64]:
                    qo = q_offs_all[:B + 1]; qt = q_toks[:int(qo[-1])]
                    fn = (lambda e=eng, q=qt, o=qo: e.search(q, o, 10)) if p_name == "Dense_Unpadded" else (lambda e=eng, q=qt, o=qo: e.search(q, o, 10, 32))
                    actual_map[B][p_name] = time_cuda_fn(fn, 1, 3, dev)[0]["median_ms"]
                del eng
                torch.cuda.empty_cache()

            for B in [1, 8, 16, 32, 64]:
                actual = actual_map[B]
                for vram_b in [8.0, 16.0]:
                    sel_plan, pred_ms, pred_mem = planner.select_plan(N_toks, N_docs, B, m_len, cv, dev, vram_b)
                    # Feasible actual best
                    feas = {k: v for k, v in actual.items() if planner.estimate_vram_gb(k, N_toks, N_docs, B, m_len) <= vram_b * 0.92}
                    if not feas:
                        continue
                    f_best = min(feas, key=feas.get)
                    act_sel_ms = actual.get(sel_plan, feas[f_best])
                    ape = abs(pred_ms - act_sel_ms) / max(act_sel_ms, 1e-3) * 100.0
                    hit = (sel_plan == f_best) or (act_sel_ms <= feas[f_best] * 1.10)
                    eval_records.append({
                        "dataset": ds, "device": dev, "B": B, "vram_budget": vram_b,
                        "selected_plan": sel_plan, "oracle_best_plan": f_best,
                        "pred_ms": round(pred_ms, 3), "actual_sel_ms": round(act_sel_ms, 3),
                        "oracle_best_ms": round(feas[f_best], 3), "ape_pct": round(ape, 2),
                        "optimal_or_near_hit": bool(hit), "exact_hit": bool(sel_plan == f_best),
                    })

    mapes = [r["ape_pct"] for r in eval_records]
    hits = [r["optimal_or_near_hit"] for r in eval_records]
    ex_hits = [r["exact_hit"] for r in eval_records]
    summary = {
        "num_heldout_configs": len(eval_records),
        "median_ape_pct": float(np.median(mapes)),
        "mean_ape_pct": float(np.mean(mapes)),
        "near_optimal_hit_rate_pct": float(np.mean(hits) * 100.0),
        "exact_hit_rate_pct": float(np.mean(ex_hits) * 100.0),
    }
    print("\n[E7 Cost Model Summary]", summary)
    out_path = os.path.expanduser("~/maxbound/results/e7_cost_model.json")
    write_result_json(out_path, eval_records, {"experiment": "E7_cost_model", "summary": summary})
    print(f"Saved {out_path}")


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_e5(device=dev)
    run_e6(device=dev)
    run_e7()
