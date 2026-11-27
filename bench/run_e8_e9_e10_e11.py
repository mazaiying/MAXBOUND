import os
import sys
import json
import time
import glob
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/maxbound"))
from bench.common import time_cuda_fn, audit_exact_topk, compute_ir_metrics, write_result_json
from src.oracle import exact_maxsim_oracle
from src.maxbound_engine import MaxBoundIndex
from src.baselines.baselines import B2CuBLASUnpaddedExact


def run_e8(device=1):
    records = []
    for ds in ["scifact", "nfcorpus", "trec_covid", "arguana", "longbench", "scidocs", "fiqa", "hotpotqa"]:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            continue
        torch.cuda.empty_cache()
        d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
        d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
        q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
        q_offs_all = np.load(os.path.join(ddir, "query_offsets.int64.npy"))
        B_eff = min(64, len(q_offs_all) - 1)
        q_offs = q_offs_all[:B_eff + 1]
        q_sub = q_toks[:int(q_offs[-1])]

        fp16_gb = d_toks.numel() * 2 / (1024 ** 3)
        b2 = B2CuBLASUnpaddedExact(d_toks, d_offs, device=device)
        t_b2 = time_cuda_fn(lambda: b2.search(q_sub, q_offs, 10), 2, 5, device)[0]["median_ms"] / float(B_eff)  # per query ms
        del b2; torch.cuda.empty_cache()

        for mode in ["int8", "int6", "int4"]:
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            eng = MaxBoundIndex(d_toks, d_offs, device=device, mode=mode)
            torch.cuda.synchronize(device)
            build_s = time.perf_counter() - t0
            if mode == "int8":
                idx_bytes = eng.d_qdata.numel() + (eng.d_scales.numel() + eng.d_rho.numel()) * 4 + eng.d_means.numel() * 2
            elif mode == "int6":
                idx_bytes = eng.d_hi4.numel() + eng.d_lo2.numel() + (eng.d_scales.numel() + eng.d_rho.numel()) * 4 + eng.d_means.numel() * 2
            else:
                idx_bytes = eng.d_qdata.numel() + (eng.d_scales.numel() + eng.d_rho.numel()) * 4 + eng.d_means.numel() * 2
            idx_gb = idx_bytes / (1024 ** 3)

            t_mb = time_cuda_fn(lambda: eng.search(q_sub, q_offs, 10, 32), 2, 5, device)[0]["median_ms"] / float(B_eff)
            gain_per_q_s = max((t_b2 - t_mb) / 1000.0, 1e-9)
            breakeven_q = int(np.ceil(build_s / gain_per_q_s)) if t_b2 > t_mb else -1
            records.append({
                "dataset": ds, "mode": mode, "N_docs": len(d_offs) - 1, "N_toks": d_toks.shape[0],
                "fp16_gb": round(fp16_gb, 4), "index_gb": round(idx_gb, 4),
                "compression_ratio": round(fp16_gb / max(idx_gb, 1e-6), 3),
                "build_s": round(build_s, 4),
                "b2_per_q_ms": round(t_b2, 4), "mb_per_q_ms": round(t_mb, 4),
                "breakeven_queries": breakeven_q,
            })
            print(f"[E8 {ds:10s} {mode}] build={build_s:.3f}s | comp={fp16_gb/idx_gb:.2f}x | break-even Q={breakeven_q}")
            del eng; torch.cuda.empty_cache()

    out_path = os.path.expanduser("~/maxbound/results/e8_build_amortization.json")
    write_result_json(out_path, records, {"experiment": "E8_build_and_amortization", "device": device})
    print(f"Saved {out_path}")


def run_e9():
    # Consolidate 100.000% exactness audits across ALL saved exact result JSONs
    results_dir = os.path.expanduser("~/maxbound/results")
    audit_rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        fname = os.path.basename(path)
        if fname in ("p0_hardware.json", "m3_calibration.json", "e9_exactness_audit.json"):
            continue
        with open(path) as f:
            data = json.load(f)
        for rec in data.get("records", []):
            aud = rec.get("audit")
            if isinstance(aud, dict) and rec.get("exact", True) and "agreement" in aud:
                method = rec.get("method") or rec.get("config") or "exact_method"
                if any(k in str(method) for k in ["B3", "B4", "B5", "PLAID", "MUVERA"]):
                    continue
                audit_rows.append({
                    "file": fname,
                    "dataset": rec.get("dataset", "scale_50m"),
                    "method": method,
                    "batch": rec.get("batch"),
                    "agreement": aud["agreement"],
                    "num_mismatch": aud["num_mismatch"],
                    "max_score_diff": aud.get("max_score_diff", 0.0),
                })
    all_pass = all(r["agreement"] == 1.0 and r["num_mismatch"] == 0 for r in audit_rows)
    out_path = os.path.expanduser("~/maxbound/results/e9_exactness_audit.json")
    write_result_json(out_path, audit_rows, {"experiment": "E9_exactness_audit", "all_exact_passed": all_pass, "total_checks": len(audit_rows)})
    print(f"[E9] Consolidated {len(audit_rows)} exactness checks, all_pass={all_pass} -> {out_path}")


def run_e10(device=1):
    records = []
    ddir = os.path.expanduser("~/maxbound/data_mv/fiqa")
    d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
    d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
    q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
    q_offs_all = np.load(os.path.join(ddir, "query_offsets.int64.npy"))

    mb8 = MaxBoundIndex(d_toks, d_offs, device=device, mode="int8")
    b2 = B2CuBLASUnpaddedExact(d_toks, d_offs, device=device)
    q_offs = q_offs_all[:65]; q_sub = q_toks[:int(q_offs[-1])]
    o_i, o_s = exact_maxsim_oracle(q_sub.float(), q_offs, d_toks, d_offs, topk=10, device=device)

    # 10A: c_warm sensitivity
    for cw in [0, 8, 16, 32, 64, 128]:
        t, (pi, ps, inf) = time_cuda_fn(lambda c=cw: mb8.search(q_sub, q_offs, topk=10, c_warm=max(c, 1) if c > 0 else 10), 2, 5, device)
        aud = audit_exact_topk(pi, ps, o_i, o_s, tol=2e-3)
        records.append({"sweep": "c_warm", "dataset": "fiqa", "val": cw, "ms": t["median_ms"], "qps": 64000.0 / t["median_ms"], "surv_pct": round(100 * inf["survivor_ratio"], 3), "audit": aud["agreement"]})

    # 10B: Batch size B sensitivity
    for B in [1, 8, 16, 32, 64]:
        qo = q_offs_all[:B + 1]; qt = q_toks[:int(qo[-1])]
        t2, _ = time_cuda_fn(lambda: b2.search(qt, qo, 10), 2, 5, device)
        t8, (_, _, i8) = time_cuda_fn(lambda: mb8.search(qt, qo, 10, 32), 2, 5, device)
        records.append({"sweep": "batch_B", "dataset": "fiqa", "val": B, "b2_ms": t2["median_ms"], "mb8_ms": t8["median_ms"], "speedup": t2["median_ms"] / t8["median_ms"], "qps": B * 1000.0 / t8["median_ms"]})

    # 10C: topk sensitivity
    for tk in [1, 10, 50, 100]:
        ok_i, ok_s = exact_maxsim_oracle(q_sub.float(), q_offs, d_toks, d_offs, topk=tk, device=device)
        t8, (pi, ps, i8) = time_cuda_fn(lambda k=tk: mb8.search(q_sub, q_offs, topk=k, c_warm=max(32, k)), 2, 5, device)
        aud = audit_exact_topk(pi, ps, ok_i, ok_s, tol=2e-3)
        assert aud["agreement"] == 1.0
        records.append({"sweep": "topk", "dataset": "fiqa", "val": tk, "ms": t8["median_ms"], "qps": 64000.0 / t8["median_ms"], "surv_pct": round(100 * i8["survivor_ratio"], 3), "audit": aud["agreement"]})

    # 10D: Quantization width sensitivity {int4, int6, int8}
    del b2, mb8; torch.cuda.empty_cache()
    for mode in ["int4", "int6", "int8"]:
        eng = MaxBoundIndex(d_toks, d_offs, device=device, mode=mode)
        t, (pi, ps, inf) = time_cuda_fn(lambda: eng.search(q_sub, q_offs, 10, 32), 2, 5, device)
        aud = audit_exact_topk(pi, ps, o_i, o_s, tol=2e-3)
        assert aud["agreement"] == 1.0
        records.append({"sweep": "quant_bits", "dataset": "fiqa", "val": mode, "ms": t["median_ms"], "qps": 64000.0 / t["median_ms"], "surv_pct": round(100 * inf["survivor_ratio"], 3), "audit": aud["agreement"]})
        del eng; torch.cuda.empty_cache()

    # 10E: High-dimensional BGE-M3 (d=1024) on scifact_bgem3
    bdir = os.path.expanduser("~/maxbound/data_mv/scifact_bgem3")
    if os.path.exists(os.path.join(bdir, "doc_tokens.fp16.npy")):
        bd = torch.from_numpy(np.load(os.path.join(bdir, "doc_tokens.fp16.npy")))
        bdo = np.load(os.path.join(bdir, "doc_offsets.int64.npy"))
        bq = torch.from_numpy(np.load(os.path.join(bdir, "query_tokens.fp16.npy")))
        bqo = np.load(os.path.join(bdir, "query_offsets.int64.npy"))[:33]
        bqs = bq[:int(bqo[-1])]
        boi, bos = exact_maxsim_oracle(bqs.float(), bqo, bd, bdo, topk=10, device=device)
        b2_m3 = B2CuBLASUnpaddedExact(bd, bdo, device=device)
        mb_m3 = MaxBoundIndex(bd, bdo, device=device, mode="int8")
        t_b2m, _ = time_cuda_fn(lambda: b2_m3.search(bqs, bqo, 10), 2, 5, device)
        t_mbm, (pm_i, pm_s, infm) = time_cuda_fn(lambda: mb_m3.search(bqs, bqo, 10, 32), 2, 5, device)
        aud_m = audit_exact_topk(pm_i, pm_s, boi, bos, tol=5e-3)
        assert aud_m["agreement"] == 1.0
        records.append({
            "sweep": "encoder_dim_1024", "dataset": "scifact_bgem3", "val": 1024, "B": 32,
            "b2_ms": t_b2m["median_ms"], "mb8_ms": t_mbm["median_ms"],
            "speedup": t_b2m["median_ms"] / t_mbm["median_ms"],
            "surv_pct": round(100 * infm["survivor_ratio"], 3), "audit": aud_m["agreement"],
        })
        print(f"[E10 BGE-M3 d=1024] B2={t_b2m['median_ms']:.2f} ms | MB8={t_mbm['median_ms']:.2f} ms ({t_b2m['median_ms']/t_mbm['median_ms']:.2f}x) | surv={infm['survivor_ratio']*100:.2f}%")
        del b2_m3, mb_m3; torch.cuda.empty_cache()

    out_path = os.path.expanduser("~/maxbound/results/e10_sensitivity.json")
    write_result_json(out_path, records, {"experiment": "E10_sensitivity", "device": device})
    print(f"Saved {out_path}")


def run_e11():
    # Architectural portability: A100 (sm_80, dev=1) vs RTX 4080 (sm_89, dev=0)
    records = []
    for ds in ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "longbench", "hotpotqa", "trec_covid"]:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            continue
        d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
        d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
        q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
        q_offs_all = np.load(os.path.join(ddir, "query_offsets.int64.npy"))
        B_eff = min(64, len(q_offs_all) - 1)
        q_offs = q_offs_all[:B_eff + 1]
        q_sub = q_toks[:int(q_offs[-1])]
        o_i, o_s = exact_maxsim_oracle(q_sub.float(), q_offs, d_toks, d_offs, topk=10, device=1)

        row = {"dataset": ds, "batch": B_eff}
        for dev, gname in [(1, "A100_sm80"), (0, "RTX4080_sm89")]:
            torch.cuda.empty_cache()
            offload_mb = False
            b2_ms = None
            try:
                b2 = B2CuBLASUnpaddedExact(d_toks, d_offs, device=dev)
                t2, _ = time_cuda_fn(lambda: b2.search(q_sub, q_offs, 10), 2, 5, dev)
                b2_ms = t2["median_ms"]
                del b2; torch.cuda.empty_cache()
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                torch.cuda.empty_cache()
                offload_mb = True

            try:
                mb8 = MaxBoundIndex(d_toks, d_offs, device=dev, mode="int8", host_offload_fp16=offload_mb)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                torch.cuda.empty_cache()
                offload_mb = True
                mb8 = MaxBoundIndex(d_toks, d_offs, device=dev, mode="int8", host_offload_fp16=True)

            t8, (pi, ps, inf) = time_cuda_fn(lambda: mb8.search(q_sub, q_offs, 10, 32), 2, 5, dev)
            aud = audit_exact_topk(pi, ps, o_i, o_s, tol=2e-3)
            assert aud["agreement"] == 1.0
            row[gname] = {
                "b2_ms": b2_ms, "mb8_ms": t8["median_ms"],
                "host_offload": offload_mb,
                "speedup": round(b2_ms / t8["median_ms"], 3) if b2_ms else None,
                "qps": round(B_eff * 1000.0 / t8["median_ms"], 1),
                "surv_pct": round(100.0 * inf["survivor_ratio"], 3),
                "audit": aud["agreement"],
            }
            del mb8; torch.cuda.empty_cache()
        print(f"[E11 {ds:10s}] A100={row['A100_sm80']['speedup']}x ({row['A100_sm80']['qps']} QPS) | 4080={row['RTX4080_sm89']['speedup']}x ({row['RTX4080_sm89']['qps']} QPS)")
        records.append(row)

    out_path = os.path.expanduser("~/maxbound/results/e11_portability.json")
    write_result_json(out_path, records, {"experiment": "E11_gpu_portability"})
    print(f"Saved {out_path}")


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_e8(device=dev)
    run_e10(device=dev)
    run_e11()
    run_e9()
