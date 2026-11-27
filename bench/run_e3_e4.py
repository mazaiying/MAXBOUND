import os
import sys
import json
import time
import subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/maxbound"))
from bench.common import time_cuda_fn, audit_exact_topk, write_result_json
from src.oracle import exact_maxsim_oracle
from src.maxbound_engine import MaxBoundIndex
from src.baselines.baselines import B1TorchPaddedGemm, B2CuBLASUnpaddedExact, B8FusedFP16TCExact


def run_e3_worker(dataset, method, budget_gb, B=16):
    device = 0
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    frac = min(1.0, max(0.05, budget_gb / total_gb))
    torch.cuda.set_per_process_memory_fraction(frac, device)
    torch.cuda.set_device(device)

    ddir = os.path.expanduser(f"~/maxbound/data_mv/{dataset}")
    # Fast fail check for impossible dense/INT8 budgets BEFORE loading 31 GiB from disk!
    d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
    n_toks_est = int(d_offs[-1])
    n_docs_est = len(d_offs) - 1
    fp16_gib = (n_toks_est * 128 * 2) / (1024 ** 3)
    if method in ("B1_Torch_Padded_GEMM", "B2_cuBLAS_Unpadded_Exact", "B8_Fused_FP16_TC_Exact", "MAXBOUND_INT8_GPU") and fp16_gib > budget_gb * 0.99:
        print("RESULT_JSON:" + json.dumps({
            "dataset": dataset, "method": method, "budget_gb": budget_gb, "batch": B, "status": "OOM",
            "error": f"CUDA out of memory. Corpus FP16 requires {fp16_gib:.2f} GiB > {budget_gb:.1f} GiB quota.",
            "peak_vram_gb": 0.0, "qps": 0.0,
        }))
        return
    pad_toks_est = int(np.sum(((np.diff(d_offs) + 15) // 16) * 16))
    q_idx_gib = {
        "MAXBOUND_INT8_Offload": (pad_toks_est * 136 + n_docs_est * 260) / (1024 ** 3),
        "MAXBOUND_INT6_Offload": (pad_toks_est * 104 + n_docs_est * 260) / (1024 ** 3),
        "MAXBOUND_INT4_Offload": (pad_toks_est * 72 + n_docs_est * 260) / (1024 ** 3),
    }.get(method, 0.0)
    if q_idx_gib > budget_gb * 0.98:
        print("RESULT_JSON:" + json.dumps({
            "dataset": dataset, "method": method, "budget_gb": budget_gb, "batch": B, "status": "OOM",
            "error": f"CUDA out of memory. Quantized GPU index requires {q_idx_gib:.2f} GiB > {budget_gb:.1f} GiB quota.",
            "peak_vram_gb": 0.0, "qps": 0.0,
        }))
        return

    # Sequential load into RAM (4 GB/s) avoids 129M random mmap page faults during length-sorting!
    d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
    q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
    q_offs = np.load(os.path.join(ddir, "query_offsets.int64.npy"))[:B + 1]
    q_sub = q_toks[:int(q_offs[-1])]

    oracle_file = os.path.expanduser(f"~/maxbound/results/{dataset}_oracle_b{B}_fp64.npz")
    or_data = np.load(oracle_file)
    oracle_ids, oracle_scores = or_data["ids"][:B], or_data["scores"][:B]

    torch.cuda.reset_peak_memory_stats(device)
    try:
        t0 = time.perf_counter()
        if method == "B1_Torch_Padded_GEMM":
            eng = B1TorchPaddedGemm(d_toks, d_offs, device=device)
            fn = lambda: eng.search(q_sub, q_offs, topk=10)
        elif method == "B2_cuBLAS_Unpadded_Exact":
            eng = B2CuBLASUnpaddedExact(d_toks, d_offs, device=device)
            fn = lambda: eng.search(q_sub, q_offs, topk=10)
        elif method == "B8_Fused_FP16_TC_Exact":
            eng = B8FusedFP16TCExact(d_toks, d_offs, device=device)
            fn = lambda: eng.search(q_sub, q_offs, topk=10)
        elif method == "MAXBOUND_INT8_GPU":
            eng = MaxBoundIndex(d_toks, d_offs, device=device, mode="int8", host_offload_fp16=False)
            fn = lambda: eng.search(q_sub, q_offs, topk=10, c_warm=32)
        elif method == "MAXBOUND_INT8_Offload":
            eng = MaxBoundIndex(d_toks, d_offs, device=device, mode="int8", host_offload_fp16=True)
            fn = lambda: eng.search(q_sub, q_offs, topk=10, c_warm=32)
        elif method == "MAXBOUND_INT6_Offload":
            eng = MaxBoundIndex(d_toks, d_offs, device=device, mode="int6", host_offload_fp16=True)
            fn = lambda: eng.search(q_sub, q_offs, topk=10, c_warm=32)
        elif method == "MAXBOUND_INT4_Offload":
            eng = MaxBoundIndex(d_toks, d_offs, device=device, mode="int4", host_offload_fp16=True)
            fn = lambda: eng.search(q_sub, q_offs, topk=10, c_warm=32)
        else:
            raise ValueError(method)
        torch.cuda.synchronize(device)
        build_s = time.perf_counter() - t0
        t_res, out = time_cuda_fn(fn, warmup=1, repeats=5, device=device)
        pred_ids, pred_sc = out[0], out[1]
        info = out[2] if len(out) > 2 else {}
        aud = audit_exact_topk(pred_ids, pred_sc, oracle_ids, oracle_scores, tol=3e-3)
        assert aud["agreement"] == 1.0, f"Audit failed: {aud}"
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        rec = {
            "dataset": dataset, "method": method, "budget_gb": budget_gb, "batch": B, "status": "OK",
            **t_res, "qps": B * 1000.0 / t_res["median_ms"],
            "peak_vram_gb": round(peak_gb, 4), "build_s": round(build_s, 3),
            "h2d_mb": round(info.get("h2d_mb", 0.0), 4),
            "survivor_pct": round(100.0 * info.get("survivor_ratio", 1.0), 5),
            "prune_pct": round(100.0 * (1.0 - info.get("survivor_ratio", 1.0)), 5),
            "audit": aud,
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        rec = {
            "dataset": dataset, "method": method, "budget_gb": budget_gb, "batch": B, "status": "OOM",
            "error": str(e)[:160], "peak_vram_gb": round(peak_gb, 4), "qps": 0.0,
        }
    print("RESULT_JSON:" + json.dumps(rec))


def run_e3_orchestrator():
    records = []
    for dataset in ["msmarco_2m", "scale_50m", "msmarco"]:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{dataset}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            print(f"{dataset} missing, skipping E3")
            continue
        B = 16
        oracle_file = os.path.expanduser(f"~/maxbound/results/{dataset}_oracle_b{B}_fp64.npz")
        if not os.path.exists(oracle_file):
            d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy"), mmap_mode="r"))
            d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
            q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
            q_offs = np.load(os.path.join(ddir, "query_offsets.int64.npy"))[:B + 1]
            q_sub = q_toks[:int(q_offs[-1])]
            o_ids, o_sc = exact_maxsim_oracle(q_sub.float(), q_offs, d_toks, d_offs, topk=10, doc_batch_size=2048, device=1, use_fp64=True)
            os.makedirs(os.path.dirname(oracle_file), exist_ok=True)
            np.savez(oracle_file, ids=o_ids, scores=o_sc)
            torch.cuda.empty_cache()

        methods = [
            "B1_Torch_Padded_GEMM",
            "B2_cuBLAS_Unpadded_Exact",
            "B8_Fused_FP16_TC_Exact",
            "MAXBOUND_INT8_GPU",
            "MAXBOUND_INT8_Offload",
            "MAXBOUND_INT6_Offload",
            "MAXBOUND_INT4_Offload",
        ]
        if dataset == "msmarco_2m":
            budgets = [10.0, 12.0, 16.0]
        elif dataset == "scale_50m":
            budgets = [6.0, 8.0, 12.0, 16.0]
        else:
            budgets = [4.0, 8.0, 12.0, 16.0]
        for b in budgets:
            for m in methods:
                if dataset != "msmarco_2m" and b >= 12.0 and m == "MAXBOUND_INT4_Offload":
                    continue
                print(f"\n[E3 {dataset}] budget={b} GB, method={m} on RTX 4080...")
                clean_env = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
                p = subprocess.run(
                    [sys.executable, __file__, "--e3-worker", dataset, m, str(b), str(B)],
                    capture_output=True, text=True, env=clean_env
                )
                parsed = None
                for line in p.stdout.splitlines():
                    if line.startswith("RESULT_JSON:"):
                        parsed = json.loads(line[len("RESULT_JSON:"):])
                if parsed is None:
                    parsed = {"dataset": dataset, "method": m, "budget_gb": b, "status": "OOM_OR_ERR", "stderr": p.stderr[-240:], "qps": 0.0}
                err_str = f" err={parsed.get('error', '')[:80]}" if parsed.get("status") != "OK" else ""
                print("  ->", parsed.get("status"), f"qps={parsed.get('qps', 0):.1f}", f"vram={parsed.get('peak_vram_gb')}{err_str}")
                records.append(parsed)

    out_path = os.path.expanduser("~/maxbound/results/e3_memory_scale.json")
    write_result_json(out_path, records, {
        "experiment": "E3_vram_constrained_msmarco2m_130m_scale50m_and_msmarco_4080",
        "msmarco_2m_provenance": "Microsoft MS MARCO Passage v1 (2,000,000 official passages + 512 official dev queries encoded by ColBERTv2.0, ~130M tokens, ~31 GiB FP16 > 2x 16GB RTX 4080)",
        "scale_50m_provenance": "allenai/WildChat-1M (980,000 real passages + 512 real queries encoded by ColBERTv2.0, 60,894,306 tokens, CV=1.245)",
    })
    print(f"Saved {out_path}")


if __name__ == "__main__":
    if len(sys.argv) >= 6 and sys.argv[1] == "--e3-worker":
        run_e3_worker(sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]))
    else:
        run_e3_orchestrator()
