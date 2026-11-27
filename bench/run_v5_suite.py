import os
import sys
import json
import time
import subprocess
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/maxbound"))
from bench.common import time_cuda_fn, audit_exact_topk, compute_ir_metrics, write_result_json
from bench.run_e1_e2 import time_cpu_fn, eval_ir_batched
from src.oracle import exact_maxsim_oracle
from src.maxbound_engine import MaxBoundIndex, get_maxbound_ext
from src.dual_gpu import DualGpuMaxBoundEngine
from src.baselines.baselines import (
    B1TorchPaddedGemm,
    B2CuBLASUnpaddedExact,
    B8FusedFP16TCExact,
    B3QuantizedDecompressGemm,
    B4OfficialFastPlaid,
    B4PlaidGPU,
    B5MuveraGPU,
    B9WarpGPU,
    B10EmvbGPU,
    B6CpuExactMaxSim,
)
from src.planner import MaxBoundPlanner, MaxBoundAutoEngine


ALL_11_DATASETS = [
    "scifact", "nfcorpus", "arguana", "scidocs", "longbench",
    "fiqa", "hotpotqa", "trec_covid", "msmarco", "msmarco_2m",
    "scifact_bgem3", "fiqa_bgem3"
]


def build_timed(fn, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    m0 = torch.cuda.memory_allocated(device)
    t0 = time.perf_counter()
    obj = fn()
    torch.cuda.synchronize(device)
    bt = time.perf_counter() - t0
    m1 = torch.cuda.memory_allocated(device)
    return obj, bt, max(0.0, (m1 - m0) / (1024 ** 3))


# ==============================================================================
# E1: 11-Dataset Main Evaluation (9 Baselines + 2 MAXBOUND Tiers + Hybrids + Rerank, 15-Run 95% CI, FP64 Zero-FN Audit)
# ==============================================================================
def run_e1_v5(device=1):
    records = []
    e1_path = os.path.expanduser("~/maxbound/results/e1_main_eval.json")

    for ds in ALL_11_DATASETS:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            print(f"[E1] Skipping missing {ds}")
            continue
        torch.cuda.empty_cache()
        d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
        d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
        q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
        q_offs_all = np.load(os.path.join(ddir, "query_offsets.int64.npy"))
        with open(os.path.join(ddir, "qrels.json")) as f:
            qrels = json.load(f)

        N_docs = len(d_offs) - 1
        N_toks = d_toks.shape[0]
        D = d_toks.shape[1]
        lens_ds = np.diff(d_offs)
        doc_cv = float(np.std(lens_ds) / max(np.mean(lens_ds), 1e-9))
        num_eval_q = min(512, len(q_offs_all) - 1)
        reps = 15 if N_toks < 20_000_000 else (9 if N_toks < 80_000_000 else 7)
        print(f"\n================ E1 v6 [{ds}] D={D} N_docs={N_docs} N_toks={N_toks} Q_eval={num_eval_q} ================")

        batches = [64, 16, 1] if ds in ("scifact", "fiqa", "longbench", "hotpotqa", "msmarco", "msmarco_2m", "scifact_bgem3", "fiqa_bgem3") else [64]
        tol = 5e-3 if D == 1024 else 2e-3

        # Precompute FP64 exact Top-K oracles per batch before building heavy GPU indices
        oracles = {}
        for B in batches:
            B_eff = min(B, len(q_offs_all) - 1)
            qo_b = q_offs_all[:B_eff + 1]
            qs_b = q_toks[:int(qo_b[-1])]
            oracles[B] = exact_maxsim_oracle(
                qs_b.float(), qo_b, d_toks, d_offs, topk=10,
                doc_batch_size=1024 if D == 1024 else 2048, device=device, use_fp64=True
            )
        torch.cuda.empty_cache()

        # Build & evaluate B2 strictly standalone, then free immediately so 130M-token msmarco_2m never exceeds 50 GB VRAM!
        b2, b2_bt, b2_gb = build_timed(lambda: B2CuBLASUnpaddedExact(d_toks, d_offs, device=device), device)
        ir_exact = eval_ir_batched(lambda qt, qo: b2.search(qt, qo, topk=10), q_toks, q_offs_all, qrels, num_eval_q)
        b2_ms_map = {}
        for B in batches:
            B_eff = min(B, len(q_offs_all) - 1)
            q_offs = q_offs_all[:B_eff + 1]; q_sub = q_toks[:int(q_offs[-1])]
            oracle_ids, oracle_scores = oracles[B]
            torch.cuda.reset_peak_memory_stats(device)
            t_b2, (i_b2, s_b2) = time_cuda_fn(lambda: b2.search(q_sub, q_offs, topk=10), warmup=3, repeats=reps, device=device)
            aud_b2 = audit_exact_topk(i_b2, s_b2, oracle_ids, oracle_scores, tol=tol)
            assert aud_b2["agreement"] == 1.0
            b2_ms_map[B] = t_b2["median_ms"]
            records.append({
                "dataset": ds, "dim": D, "batch": B_eff, "mode": "full_corpus", "method": "B2_cuBLAS_Unpadded_Exact",
                "exact": True, **t_b2, "qps": B_eff * 1000.0 / t_b2["median_ms"], "speedup_vs_b2": 1.0,
                "prune_pct": 0.0, "survivor_pct": 100.0, "build_s": round(b2_bt, 3), "index_gb": round(b2_gb, 4),
                "vram_peak_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 4),
                "ir": ir_exact, "audit": aud_b2, "N_docs": N_docs, "N_toks": N_toks,
            })
        del b2
        torch.cuda.empty_cache()

        # B8 Hand-Crafted Fused Tensor-Core FP16 Exact Baseline (build -> eval all batches -> free!)
        b8, b8_bt, b8_gb = build_timed(lambda: B8FusedFP16TCExact(d_toks, d_offs, device=device), device)
        b8_ms_map = {}
        for B in batches:
            B_eff = min(B, len(q_offs_all) - 1)
            q_offs = q_offs_all[:B_eff + 1]; q_sub = q_toks[:int(q_offs[-1])]
            oracle_ids, oracle_scores = oracles[B]
            torch.cuda.reset_peak_memory_stats(device)
            t_b8, (i_b8, s_b8) = time_cuda_fn(lambda: b8.search(q_sub, q_offs, topk=10), warmup=3, repeats=reps, device=device)
            aud_b8 = audit_exact_topk(i_b8, s_b8, oracle_ids, oracle_scores, tol=tol)
            assert aud_b8["agreement"] == 1.0, f"{ds} B8 B={B_eff} FAILED: {aud_b8}"
            b8_ms_map[B] = t_b8["median_ms"]
            records.append({
                "dataset": ds, "dim": D, "batch": B_eff, "mode": "full_corpus", "method": "B8_Fused_FP16_TC_Exact",
                "exact": True, **t_b8, "qps": B_eff * 1000.0 / t_b8["median_ms"],
                "speedup_vs_b2": b2_ms_map[B] / t_b8["median_ms"],
                "speedup_vs_best_exact": min(b2_ms_map[B], t_b8["median_ms"]) / t_b8["median_ms"],
                "prune_pct": 0.0, "survivor_pct": 100.0, "build_s": round(b8_bt, 3), "index_gb": round(b8_gb, 4),
                "vram_peak_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 4),
                "ir": ir_exact, "audit": aud_b8, "N_docs": N_docs, "N_toks": N_toks,
            })
        del b8
        torch.cuda.empty_cache()

        # MAXBOUND-INT6 (D=128: build -> eval BEFORE mb8 so mb6 & mb8 are NEVER co-resident on 130M-token msmarco_2m!)
        if D == 128:
            mb6, mb6_bt, mb6_gb = build_timed(lambda: MaxBoundIndex(d_toks, d_offs, device=device, mode="int6"), device)
            for B in batches:
                B_eff = min(B, len(q_offs_all) - 1)
                q_offs = q_offs_all[:B_eff + 1]; q_sub = q_toks[:int(q_offs[-1])]
                oracle_ids, oracle_scores = oracles[B]
                torch.cuda.reset_peak_memory_stats(device)
                t_mb6, (i_mb6, s_mb6, info6) = time_cuda_fn(
                    lambda: mb6.search(q_sub, q_offs, topk=10, c_warm=32), warmup=3, repeats=reps, device=device
                )
                peak_vram6 = round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 4)
                _, _, info6_m = mb6.search(q_sub, q_offs, topk=10, c_warm=32, return_tested_mask=True)
                aud_mb6 = audit_exact_topk(i_mb6, s_mb6, oracle_ids, oracle_scores, tol=tol, tested_mask=info6_m["tested_mask_orig"])
                assert aud_mb6["agreement"] == 1.0 and aud_mb6["num_mismatch"] == 0 and aud_mb6["false_negative_prunes"] == 0
                sp_b2 = b2_ms_map[B] / t_mb6["median_ms"]
                sp_b8 = b8_ms_map[B] / t_mb6["median_ms"]
                sp_best = min(b2_ms_map[B], b8_ms_map[B]) / t_mb6["median_ms"]
                records.append({
                    "dataset": ds, "dim": D, "batch": B_eff, "mode": "full_corpus", "method": "MAXBOUND_INT6",
                    "exact": True, **t_mb6, "qps": B_eff * 1000.0 / t_mb6["median_ms"],
                    "speedup_vs_b2": sp_b2, "speedup_vs_b8_fused": sp_b8, "speedup_vs_best_exact": sp_best,
                    "prune_pct": round(100.0 * (1.0 - info6["survivor_ratio"]), 4),
                    "survivor_pct": round(100.0 * info6["survivor_ratio"], 4),
                    "build_s": round(mb6_bt, 3), "index_gb": round(mb6_gb, 4),
                    "vram_peak_gb": peak_vram6,
                    "ir": ir_exact, "audit": aud_mb6, "N_docs": N_docs, "N_toks": N_toks,
                })
                print(f"  B={B_eff:2d} {'MAXBOUND_INT6':14s}: {t_mb6['median_ms']:6.2f}±{t_mb6['ci95_ms']:.3f} ms ({sp_b2:.2f}x vs B2={b2_ms_map[B]:.2f}, {sp_best:.2f}x vs BestExact) | surv={info6['survivor_ratio']*100:.3f}% | FN={aud_mb6['false_negative_prunes']}")
            del mb6
            torch.cuda.empty_cache()

        # MAXBOUND-INT8 + MAXBOUND-Auto (zero-copy M3 adaptive dispatcher)
        mb8, mb8_bt, mb8_gb = build_timed(lambda: MaxBoundIndex(d_toks, d_offs, device=device, mode="int8"), device)
        mb_auto = MaxBoundAutoEngine(mb8, cv=doc_cv)
        ir_mb8 = eval_ir_batched(lambda qt, qo: mb8.search(qt, qo, topk=10, c_warm=32)[:2], q_toks, q_offs_all, qrels, num_eval_q)

        def stream_mb8_all():
            for s in range(0, num_eval_q, 64):
                e = min(s + 64, num_eval_q)
                so = q_offs_all[s:e + 1] - q_offs_all[s]
                st = q_toks[int(q_offs_all[s]):int(q_offs_all[e])]
                mb8.search(st, so, topk=10, c_warm=32)

        t_stream_mb8, _ = time_cuda_fn(stream_mb8_all, warmup=1, repeats=5 if N_toks < 80_000_000 else 3, device=device)
        stream_qps_mb8 = round(num_eval_q * 1000.0 / t_stream_mb8["median_ms"], 2)

        for B in batches:
            B_eff = min(B, len(q_offs_all) - 1)
            q_offs = q_offs_all[:B_eff + 1]; q_sub = q_toks[:int(q_offs[-1])]
            oracle_ids, oracle_scores = oracles[B]
            torch.cuda.reset_peak_memory_stats(device)
            t_mb, (i_mb, s_mb, info) = time_cuda_fn(
                lambda: mb8.search(q_sub, q_offs, topk=10, c_warm=32), warmup=3, repeats=reps, device=device
            )
            peak_vram = round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 4)
            _, _, info_mask = mb8.search(q_sub, q_offs, topk=10, c_warm=32, return_tested_mask=True)
            aud_mb = audit_exact_topk(i_mb, s_mb, oracle_ids, oracle_scores, tol=tol, tested_mask=info_mask["tested_mask_orig"])
            assert aud_mb["agreement"] == 1.0 and aud_mb["num_mismatch"] == 0 and aud_mb["false_negative_prunes"] == 0, \
                f"{ds} MAXBOUND_INT8 B={B_eff} FAILED: {aud_mb}"
            sp_b2 = b2_ms_map[B] / t_mb["median_ms"]
            sp_b8 = b8_ms_map[B] / t_mb["median_ms"]
            sp_best = min(b2_ms_map[B], b8_ms_map[B]) / t_mb["median_ms"]
            rec_mb = {
                "dataset": ds, "dim": D, "batch": B_eff, "mode": "full_corpus", "method": "MAXBOUND_INT8",
                "exact": True, **t_mb, "qps": B_eff * 1000.0 / t_mb["median_ms"],
                "speedup_vs_b2": sp_b2, "speedup_vs_b8_fused": sp_b8, "speedup_vs_best_exact": sp_best,
                "prune_pct": round(100.0 * (1.0 - info["survivor_ratio"]), 4),
                "survivor_pct": round(100.0 * info["survivor_ratio"], 4),
                "build_s": round(mb8_bt, 3), "index_gb": round(mb8_gb, 4),
                "vram_peak_gb": peak_vram,
                "ir": ir_mb8, "audit": aud_mb, "N_docs": N_docs, "N_toks": N_toks,
            }
            if B == 64:
                rec_mb["streaming_512q_qps"] = stream_qps_mb8
                rec_mb["num_eval_queries_streamed"] = num_eval_q
            records.append(rec_mb)
            print(f"  B={B_eff:2d} {'MAXBOUND_INT8':14s}: {t_mb['median_ms']:6.2f}±{t_mb['ci95_ms']:.3f} ms ({sp_b2:.2f}x vs B2={b2_ms_map[B]:.2f}, {sp_best:.2f}x vs BestExact={min(b2_ms_map[B],b8_ms_map[B]):.2f}) | surv={info['survivor_ratio']*100:.3f}% | FN={aud_mb['false_negative_prunes']}")

            # Also evaluate MAXBOUND_Auto (M3 adaptive plan dispatcher)
            t_au, (i_au, s_au, info_au) = time_cuda_fn(
                lambda: mb_auto.search(q_sub, q_offs, topk=10, c_warm=32), warmup=3, repeats=reps, device=device
            )
            aud_au = audit_exact_topk(i_au, s_au, oracle_ids, oracle_scores, tol=tol)
            assert aud_au["agreement"] == 1.0
            records.append({
                "dataset": ds, "dim": D, "batch": B_eff, "mode": "full_corpus", "method": "MAXBOUND_Auto",
                "selected_plan": info_au.get("selected_plan"),
                "exact": True, **t_au, "qps": B_eff * 1000.0 / t_au["median_ms"],
                "speedup_vs_b2": b2_ms_map[B] / t_au["median_ms"],
                "speedup_vs_b8_fused": b8_ms_map[B] / t_au["median_ms"],
                "speedup_vs_best_exact": min(b2_ms_map[B], b8_ms_map[B]) / t_au["median_ms"],
                "prune_pct": round(100.0 * (1.0 - info_au["survivor_ratio"]), 4),
                "survivor_pct": round(100.0 * info_au["survivor_ratio"], 4),
                "build_s": round(mb8_bt, 3), "index_gb": round(mb8_gb, 4),
                "vram_peak_gb": peak_vram,
                "ir": ir_mb8, "audit": aud_au, "N_docs": N_docs, "N_toks": N_toks,
            })

        del mb_auto
        if N_toks > 80_000_000:
            # On 130M-token msmarco_2m, free mb8 before building approximate indices, then reload only for hybrid rerank!
            del mb8
            torch.cuda.empty_cache()
            mb8 = None

        # Remaining B=64 baselines built & freed strictly one-at-a-time to keep peak VRAM low!
        B_eff = min(64, len(q_offs_all) - 1)
        q_offs = q_offs_all[:B_eff + 1]
        q_sub = q_toks[:int(q_offs[-1])]
        oracle_ids, oracle_scores = oracles[64]
        b2_ms = b2_ms_map[64]
        n_cent = min(2048 if N_docs >= 100000 else 1024, max(64, N_docs // 4))

        # B1 Padded GEMM (skip building if N_docs*d_max*D*2 > 32 GB)
        max_doc_l = int(np.diff(d_offs).max())
        est_b1_gb = (N_docs * max_doc_l * D * 2) / (1024 ** 3)
        if est_b1_gb > 32.0:
            records.append({
                "dataset": ds, "dim": D, "batch": B_eff, "mode": "full_corpus", "method": "B1_Torch_Padded_GEMM",
                "exact": True, "status": "OOM", "oom": True, "qps": 0.0, "median_ms": None, "N_docs": N_docs, "N_toks": N_toks,
            })
        else:
            try:
                torch.cuda.empty_cache()
                b1, b1_bt, b1_gb = build_timed(lambda: B1TorchPaddedGemm(d_toks, d_offs, device=device), device)
                torch.cuda.reset_peak_memory_stats(device)
                t_b1, (i_b1, s_b1) = time_cuda_fn(lambda: b1.search(q_sub, q_offs, topk=10), warmup=2, repeats=min(reps, 7), device=device)
                aud_b1 = audit_exact_topk(i_b1, s_b1, oracle_ids, oracle_scores, tol=tol)
                assert aud_b1["agreement"] == 1.0
                records.append({
                    "dataset": ds, "dim": D, "batch": B_eff, "mode": "full_corpus", "method": "B1_Torch_Padded_GEMM",
                    "exact": True, **t_b1, "qps": B_eff * 1000.0 / t_b1["median_ms"],
                    "speedup_vs_b2": b2_ms / t_b1["median_ms"], "prune_pct": 0.0, "survivor_pct": 100.0,
                    "build_s": round(b1_bt, 3), "index_gb": round(b1_gb, 4),
                    "vram_peak_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 4),
                    "ir": ir_exact, "audit": aud_b1, "N_docs": N_docs, "N_toks": N_toks,
                })
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                records.append({
                    "dataset": ds, "dim": D, "batch": B_eff, "mode": "full_corpus", "method": "B1_Torch_Padded_GEMM",
                    "exact": True, "status": "OOM", "oom": True, "qps": 0.0, "median_ms": None, "N_docs": N_docs, "N_toks": N_toks,
                })
            finally:
                b1 = None
                torch.cuda.empty_cache()

        # B3 Quantized Decompress
        try:
            b3, b3_bt, b3_gb = build_timed(lambda: B3QuantizedDecompressGemm(d_toks, d_offs, device=device), device)
            ir_b3 = eval_ir_batched(lambda qt, qo: b3.search(qt, qo, topk=10), q_toks, q_offs_all, qrels, num_eval_q)
            torch.cuda.reset_peak_memory_stats(device)
            t_b3, (i_b3, s_b3) = time_cuda_fn(lambda: b3.search(q_sub, q_offs, topk=10), warmup=2, repeats=min(reps, 9), device=device)
            records.append({
                "dataset": ds, "dim": D, "batch": B_eff, "mode": "full_corpus", "method": "B3_Quant_Decompress",
                "exact": False, **t_b3, "qps": B_eff * 1000.0 / t_b3["median_ms"],
                "speedup_vs_b2": b2_ms / t_b3["median_ms"], "prune_pct": 0.0, "survivor_pct": 100.0,
                "build_s": round(b3_bt, 3), "index_gb": round(b3_gb, 4),
                "vram_peak_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 4),
                "ir": ir_b3, "audit": audit_exact_topk(i_b3, s_b3, oracle_ids, oracle_scores, tol=tol),
                "N_docs": N_docs, "N_toks": N_toks,
            })
        finally:
            b3 = None
            torch.cuda.empty_cache()

        # B6 CPU 48-thread Exact
        work_toks = N_toks * (D // 128)
        b_cpu = min(B_eff, 1 if work_toks > 80_000_000 else (2 if work_toks > 25_000_000 else (4 if work_toks > 12_000_000 else (8 if work_toks > 5_000_000 else 16))))
        q_cpu_offs = q_offs_all[:b_cpu + 1]
        q_cpu_sub = q_toks[:int(q_cpu_offs[-1])]
        o_cpu_ids, o_cpu_sc = exact_maxsim_oracle(q_cpu_sub.float(), q_cpu_offs, d_toks, d_offs, topk=10, device=device, use_fp64=True)
        b6 = B6CpuExactMaxSim(d_toks, d_offs)
        t_b6, (i_b6, s_b6) = time_cpu_fn(lambda: b6.search(q_cpu_sub, q_cpu_offs, topk=10), warmup=1 if work_toks <= 80_000_000 else 0, repeats=2 if work_toks <= 80_000_000 else 1)
        aud_b6 = audit_exact_topk(i_b6, s_b6, o_cpu_ids, o_cpu_sc, tol=tol)
        assert aud_b6["agreement"] == 1.0
        records.append({
            "dataset": ds, "dim": D, "batch": b_cpu, "mode": "full_corpus", "method": "B6_CPU_48T_Exact",
            "exact": True, **t_b6, "qps": b_cpu * 1000.0 / t_b6["median_ms"],
            "speedup_vs_b2": (b_cpu * 1000.0 / t_b6["median_ms"]) / (B_eff * 1000.0 / b2_ms),
            "prune_pct": 0.0, "survivor_pct": 100.0, "build_s": 0.0, "index_gb": 0.0,
            "ir": ir_exact, "audit": aud_b6, "N_docs": N_docs, "N_toks": N_toks,
        })
        del b6

        # Official FastPLAID (Rust+CUDA, D=128; skip 2M-doc disk build to stay well inside <100GB disk cap!)
        if D == 128 and N_docs <= 1_000_000:
            fp_off = None
            try:
                torch.cuda.empty_cache()
                fp_off, fp_bt, _ = build_timed(
                    lambda: B4OfficialFastPlaid(d_toks, d_offs, device=device, nbits=2, kmeans_niters=3), device
                )
                ir_fp = eval_ir_batched(lambda qt, qo: fp_off.search(qt, qo, n_ivf_probe=4, n_full_scores=512, topk=10), q_toks, q_offs_all, qrels, num_eval_q)
                t_fp, (i_fp, s_fp) = time_cuda_fn(
                    lambda: fp_off.search(q_sub, q_offs, n_ivf_probe=4, n_full_scores=512, topk=10),
                    warmup=2, repeats=min(reps, 9), device=device
                )
                records.append({
                    "dataset": ds, "dim": D, "batch": B_eff, "mode": "approx_first_stage", "method": "B4_Official_FastPLAID",
                    "exact": False, **t_fp, "qps": B_eff * 1000.0 / t_fp["median_ms"],
                    "speedup_vs_b2": b2_ms / t_fp["median_ms"], "build_s": round(fp_bt, 3),
                    "ir": ir_fp, "audit": audit_exact_topk(i_fp, s_fp, oracle_ids, oracle_scores, tol=tol),
                    "N_docs": N_docs, "N_toks": N_toks,
                })
            except Exception as ex:
                print(f"  [WARN] Official FastPlaid skipped on {ds}: {ex}")
            finally:
                if fp_off is not None:
                    fp_off.cleanup()
                fp_off = None
                torch.cuda.empty_cache()

        # B9 WARP (SIGIR'25 CUDA)
        torch.cuda.empty_cache()
        b9, b9_bt, _ = build_timed(lambda: B9WarpGPU(d_toks, d_offs, device=device, n_centroids=n_cent), device)
        ir_b9 = eval_ir_batched(lambda qt, qo: b9.search(qt, qo, t_prime=8, topk=10), q_toks, q_offs_all, qrels, num_eval_q)
        t_b9, (i_b9, s_b9) = time_cuda_fn(lambda: b9.search(q_sub, q_offs, t_prime=8, topk=10), warmup=2, repeats=min(reps, 9), device=device)
        records.append({
            "dataset": ds, "dim": D, "batch": B_eff, "mode": "approx_first_stage", "method": "B9_WARP_GPU",
            "exact": False, **t_b9, "qps": B_eff * 1000.0 / t_b9["median_ms"],
            "speedup_vs_b2": b2_ms / t_b9["median_ms"], "build_s": round(b9_bt, 3),
            "ir": ir_b9, "audit": audit_exact_topk(i_b9, s_b9, oracle_ids, oracle_scores, tol=tol),
            "N_docs": N_docs, "N_toks": N_toks,
        })
        del b9
        torch.cuda.empty_cache()

        # B10 EMVB (ECIR'24 Bit-Vector Popcount + PQ16 ADC + Exact Refinement)
        b10, b10_bt, _ = build_timed(lambda: B10EmvbGPU(d_toks, d_offs, device=device, n_centroids=n_cent), device)
        ir_b10 = eval_ir_batched(lambda qt, qo: b10.search(qt, qo, nprobe=8, ncands=1024, nrefine=128, topk=10), q_toks, q_offs_all, qrels, num_eval_q)
        t_b10, (i_b10, s_b10) = time_cuda_fn(lambda: b10.search(q_sub, q_offs, nprobe=8, ncands=1024, nrefine=128, topk=10), warmup=2, repeats=min(reps, 9), device=device)
        records.append({
            "dataset": ds, "dim": D, "batch": B_eff, "mode": "approx_first_stage", "method": "B10_EMVB_GPU",
            "exact": False, **t_b10, "qps": B_eff * 1000.0 / t_b10["median_ms"],
            "speedup_vs_b2": b2_ms / t_b10["median_ms"], "build_s": round(b10_bt, 3),
            "ir": ir_b10, "audit": audit_exact_topk(i_b10, s_b10, oracle_ids, oracle_scores, tol=tol),
            "N_docs": N_docs, "N_toks": N_toks,
        })
        del b10
        torch.cuda.empty_cache()

        # B4 PLAID + Hybrid PLAID_plus_MAXBOUND
        b4, b4_bt, _ = build_timed(lambda: B4PlaidGPU(d_toks, d_offs, device=device, n_centroids=n_cent), device)
        ir_b4 = eval_ir_batched(lambda qt, qo: b4.search(qt, qo, nprobe=4, ncands=512, topk=10), q_toks, q_offs_all, qrels, num_eval_q)
        t_b4, (i_b4, s_b4) = time_cuda_fn(lambda: b4.search(q_sub, q_offs, nprobe=4, ncands=512, topk=10), warmup=3, repeats=min(reps, 9), device=device)
        records.append({
            "dataset": ds, "dim": D, "batch": B_eff, "mode": "approx_first_stage", "method": "B4_PLAID_GPU",
            "exact": False, **t_b4, "qps": B_eff * 1000.0 / t_b4["median_ms"],
            "speedup_vs_b2": b2_ms / t_b4["median_ms"], "build_s": round(b4_bt, 3),
            "ir": ir_b4, "audit": audit_exact_topk(i_b4, s_b4, oracle_ids, oracle_scores, tol=tol),
            "N_docs": N_docs, "N_toks": N_toks,
        })
        plaid_cands_64 = b4.get_candidates(q_sub, q_offs, nprobe=8, ncands=min(2048, N_docs))
        plaid_cands_eval = [
            b4.get_candidates(q_toks[int(q_offs_all[s]):int(q_offs_all[min(s + 64, num_eval_q)])],
                              q_offs_all[s:min(s + 64, num_eval_q) + 1] - q_offs_all[s],
                              nprobe=8, ncands=min(2048, N_docs))
            for s in range(0, num_eval_q, 64)
        ]
        t_plaid_cand, _ = time_cuda_fn(lambda: b4.get_candidates(q_sub, q_offs, nprobe=8, ncands=min(2048, N_docs)), warmup=2, repeats=5, device=device)
        del b4
        torch.cuda.empty_cache()

        # B5 MUVERA + pre-extract candidates for Hybrid & Mode B Rerank so b5 and mb8 never collide on 130M tokens!
        b5, b5_bt, _ = build_timed(lambda: B5MuveraGPU(d_toks, d_offs, device=device, r_reps=8, k_sim=4, d_proj=16), device)
        ir_b5 = eval_ir_batched(lambda qt, qo: b5.search(qt, qo, ncands=512, topk=10), q_toks, q_offs_all, qrels, num_eval_q)
        t_b5, (i_b5, s_b5) = time_cuda_fn(lambda: b5.search(q_sub, q_offs, ncands=512, topk=10), warmup=3, repeats=min(reps, 9), device=device)
        records.append({
            "dataset": ds, "dim": D, "batch": B_eff, "mode": "approx_first_stage", "method": "B5_MUVERA_GPU",
            "exact": False, **t_b5, "qps": B_eff * 1000.0 / t_b5["median_ms"],
            "speedup_vs_b2": b2_ms / t_b5["median_ms"], "build_s": round(b5_bt, 3),
            "ir": ir_b5, "audit": audit_exact_topk(i_b5, s_b5, oracle_ids, oracle_scores, tol=tol),
            "N_docs": N_docs, "N_toks": N_toks,
        })
        muv_cands_64 = b5.get_candidates(q_sub, q_offs, ncands=min(2048, N_docs))
        muv_cands_eval = [
            b5.get_candidates(q_toks[int(q_offs_all[s]):int(q_offs_all[min(s + 64, num_eval_q)])],
                              q_offs_all[s:min(s + 64, num_eval_q) + 1] - q_offs_all[s],
                              ncands=min(2048, N_docs))
            for s in range(0, num_eval_q, 64)
        ]
        t_muv_cand, _ = time_cuda_fn(lambda: b5.get_candidates(q_sub, q_offs, ncands=min(2048, N_docs)), warmup=2, repeats=5, device=device)
        rerank_cands_map = {}
        if N_docs >= 1000:
            for C in ([1000, 4096] if N_docs >= 4096 else [1000]):
                rerank_cands_map[C] = b5.get_candidates(q_sub, q_offs, ncands=C)
        del b5
        torch.cuda.empty_cache()

        if mb8 is None:
            mb8 = MaxBoundIndex(d_toks, d_offs, device=device, mode="int8")

        t_hp_rr, (ihp, shp, infohp) = time_cuda_fn(
            lambda: mb8.search_rerank(q_sub, q_offs, plaid_cands_64, topk=10, c_warm=16),
            warmup=3, repeats=min(reps, 9), device=device
        )
        hp_preds = [
            mb8.search_rerank(q_toks[int(q_offs_all[s]):int(q_offs_all[min(s + 64, num_eval_q)])],
                              q_offs_all[s:min(s + 64, num_eval_q) + 1] - q_offs_all[s],
                              plaid_cands_eval[idx_c], topk=10, c_warm=16)[0]
            for idx_c, s in enumerate(range(0, num_eval_q, 64))
        ]
        ir_hp = compute_ir_metrics(np.concatenate(hp_preds, axis=0), {str(k): v for k, v in qrels.items() if int(k) < num_eval_q}, k=10)
        hp_tot_ms = t_plaid_cand["median_ms"] + t_hp_rr["median_ms"]
        records.append({
            "dataset": ds, "dim": D, "batch": B_eff, "mode": "hybrid_exact_rerank", "method": "PLAID_plus_MAXBOUND",
            "exact": False, "median_ms": hp_tot_ms, "mean_ms": hp_tot_ms, "ci95_ms": t_hp_rr["ci95_ms"], "p99_ms": hp_tot_ms,
            "qps": B_eff * 1000.0 / hp_tot_ms,
            "speedup_vs_b2": b2_ms / hp_tot_ms, "survivor_pct": round(100.0 * infohp["survivor_ratio"], 4),
            "ir": ir_hp, "audit": audit_exact_topk(ihp, shp, oracle_ids, oracle_scores, tol=tol),
            "N_docs": N_docs, "N_toks": N_toks,
        })

        t_hm_rr, (ihm, shm, infohm) = time_cuda_fn(
            lambda: mb8.search_rerank(q_sub, q_offs, muv_cands_64, topk=10, c_warm=16),
            warmup=3, repeats=min(reps, 9), device=device
        )
        hm_preds = [
            mb8.search_rerank(q_toks[int(q_offs_all[s]):int(q_offs_all[min(s + 64, num_eval_q)])],
                              q_offs_all[s:min(s + 64, num_eval_q) + 1] - q_offs_all[s],
                              muv_cands_eval[idx_c], topk=10, c_warm=16)[0]
            for idx_c, s in enumerate(range(0, num_eval_q, 64))
        ]
        ir_hm = compute_ir_metrics(np.concatenate(hm_preds, axis=0), {str(k): v for k, v in qrels.items() if int(k) < num_eval_q}, k=10)
        hm_tot_ms = t_muv_cand["median_ms"] + t_hm_rr["median_ms"]
        records.append({
            "dataset": ds, "dim": D, "batch": B_eff, "mode": "hybrid_exact_rerank", "method": "MUVERA_plus_MAXBOUND",
            "exact": False, "median_ms": hm_tot_ms, "mean_ms": hm_tot_ms, "ci95_ms": t_hm_rr["ci95_ms"], "p99_ms": hm_tot_ms,
            "qps": B_eff * 1000.0 / hm_tot_ms,
            "speedup_vs_b2": b2_ms / hm_tot_ms, "survivor_pct": round(100.0 * infohm["survivor_ratio"], 4),
            "ir": ir_hm, "audit": audit_exact_topk(ihm, shm, oracle_ids, oracle_scores, tol=tol),
            "N_docs": N_docs, "N_toks": N_toks,
        })

        mb_rr_saved = {}
        for C, cands in rerank_cands_map.items():
            t_rm, (irm, srm, infom) = time_cuda_fn(lambda c=cands: mb8.search_rerank(q_sub, q_offs, c, topk=10, c_warm=16), warmup=3, repeats=reps, device=device)
            mb_rr_saved[C] = (t_rm, irm, srm, infom)

        del mb8
        torch.cuda.empty_cache()

        if rerank_cands_map:
            b2_rr = B2CuBLASUnpaddedExact(d_toks, d_offs, device=device)
            for C, cands in rerank_cands_map.items():
                or_id, or_sc = exact_maxsim_oracle(
                    q_sub.float(), q_offs, d_toks, d_offs, topk=10, device=device,
                    candidate_ids=cands.cpu().numpy(), use_fp64=True
                )
                t_r2, (ir2, sr2) = time_cuda_fn(lambda c=cands: b2_rr.search_rerank(q_sub, q_offs, c, topk=10), warmup=3, repeats=reps, device=device)
                t_rm, irm, srm, infom = mb_rr_saved[C]
                aud_rm = audit_exact_topk(irm, srm, or_id, or_sc, tol=tol)
                assert aud_rm["agreement"] == 1.0
                sp_r = t_r2["median_ms"] / t_rm["median_ms"]
                print(f"  [Mode B Rerank C={C}] B2={t_r2['median_ms']:.2f} ms | MB={t_rm['median_ms']:.2f} ms ({sp_r:.2f}x, surv={infom['survivor_ratio']*100:.2f}%)")
                records.append({
                    "dataset": ds, "dim": D, "batch": B_eff, "mode": f"rerank_C{C}", "method": "B2_Rerank",
                    "exact": True, **t_r2, "qps": B_eff * 1000.0 / t_r2["median_ms"], "speedup_vs_b2": 1.0,
                    "audit": audit_exact_topk(ir2, sr2, or_id, or_sc, tol=tol), "N_docs": N_docs, "N_toks": N_toks,
                })
                records.append({
                    "dataset": ds, "dim": D, "batch": B_eff, "mode": f"rerank_C{C}", "method": "MAXBOUND_Rerank_INT8",
                    "exact": True, **t_rm, "qps": B_eff * 1000.0 / t_rm["median_ms"], "speedup_vs_b2": sp_r,
                    "prune_pct": round(100.0 * (1.0 - infom["survivor_ratio"]), 4),
                    "survivor_pct": round(100.0 * infom["survivor_ratio"], 4),
                    "audit": aud_rm, "N_docs": N_docs, "N_toks": N_toks,
                })
            del b2_rr
            torch.cuda.empty_cache()

        write_result_json(e1_path, records, {
            "experiment": "E1_main_eval_12datasets_10baselines_15run_ci95",
            "device": device,
            "oracle_precision": "IEEE-754 FP64",
        })

    write_result_json(e1_path, records, {
        "experiment": "E1_main_eval_12datasets_10baselines_15run_ci95",
        "device": device,
        "oracle_precision": "IEEE-754 FP64",
    })
    print(f"\nSaved {e1_path} ({len(records)} records across 12 datasets)")


# ==============================================================================
# E2: Pareto Frontier across 4 Real Benchmarks (FiQA, SCIDOCS, HotpotQA, MS MARCO!)
#     Includes Official FastPLAID (2-bit & 4-bit), WARP (SIGIR'25), PLAID, MUVERA, Hybrids, B8 Fused FP16, B2, MAXBOUND
# ==============================================================================
def run_e2_v5(device=1):
    records = []
    for ds in ["fiqa", "scidocs", "hotpotqa", "msmarco"]:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            continue
        torch.cuda.empty_cache()
        d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
        d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
        q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
        q_offs_all = np.load(os.path.join(ddir, "query_offsets.int64.npy"))
        with open(os.path.join(ddir, "qrels.json")) as f:
            qrels = json.load(f)

        N_docs = len(d_offs) - 1
        B = 64
        q_offs = q_offs_all[:B + 1]
        q_sub = q_toks[:int(q_offs[-1])]
        num_eval_q = min(256, len(q_offs_all) - 1)
        oracle_ids_256 = []
        for s in range(0, num_eval_q, B):
            e = min(s + B, num_eval_q)
            so = q_offs_all[s:e + 1] - q_offs_all[s]
            st = q_toks[int(q_offs_all[s]):int(q_offs_all[e])]
            oi, _ = exact_maxsim_oracle(st.float(), so, d_toks, d_offs, topk=10, device=device, use_fp64=True)
            oracle_ids_256.append(oi)
        oracle_ids_256 = np.concatenate(oracle_ids_256, axis=0)

        def exact_recall_vs_oracle(pred_ids_256):
            hits = []
            for qi in range(num_eval_q):
                o_set = set(int(x) for x in oracle_ids_256[qi] if x >= 0)
                p_set = set(int(x) for x in pred_ids_256[qi] if x >= 0)
                hits.append(len(o_set & p_set) / max(len(o_set), 1))
            return float(np.mean(hits))

        def eval_256(search_fn):
            preds = []
            for s in range(0, num_eval_q, B):
                e = min(s + B, num_eval_q)
                so = q_offs_all[s:e + 1] - q_offs_all[s]
                st = q_toks[int(q_offs_all[s]):int(q_offs_all[e])]
                preds.append(search_fn(st, so)[0])
            p256 = np.concatenate(preds, axis=0)
            return exact_recall_vs_oracle(p256), compute_ir_metrics(p256, {str(k): v for k, v in qrels.items() if int(k) < num_eval_q}, k=10)

        n_cent = 2048 if N_docs >= 100000 else 1024
        # Build & evaluate each engine sequentially in E2 so 500K-doc msmarco never exceeds 25 GB VRAM!
        for name, maker in [
            ("B2_Exact", lambda: B2CuBLASUnpaddedExact(d_toks, d_offs, device=device)),
            ("B8_Fused_FP16_Exact", lambda: B8FusedFP16TCExact(d_toks, d_offs, device=device)),
            ("MAXBOUND_INT6", lambda: MaxBoundIndex(d_toks, d_offs, device=device, mode="int6")),
        ]:
            eng = maker()
            fn = (lambda st, so, e=eng: e.search(st, so, topk=10)) if "MAXBOUND" not in name else (lambda st, so, e=eng: e.search(st, so, topk=10, c_warm=32)[:2])
            t_r, _ = time_cuda_fn(lambda f=fn: f(q_sub, q_offs), warmup=3, repeats=11, device=device)
            or_rec, ir = eval_256(fn)
            records.append({
                "dataset": ds, "method": name, "config": "exact", "qps": B * 1000.0 / t_r["median_ms"],
                "median_ms": t_r["median_ms"], "ci95_ms": t_r["ci95_ms"], "oracle_topk_recall": or_rec, "ir": ir,
            })
            del eng
            torch.cuda.empty_cache()

        mb8 = MaxBoundIndex(d_toks, d_offs, device=device, mode="int8")
        fn_mb8 = lambda st, so: mb8.search(st, so, topk=10, c_warm=32)[:2]
        t_r, _ = time_cuda_fn(lambda: fn_mb8(q_sub, q_offs), warmup=3, repeats=11, device=device)
        or_rec, ir = eval_256(fn_mb8)
        records.append({
            "dataset": ds, "method": "MAXBOUND_INT8", "config": "exact", "qps": B * 1000.0 / t_r["median_ms"],
            "median_ms": t_r["median_ms"], "ci95_ms": t_r["ci95_ms"], "oracle_topk_recall": or_rec, "ir": ir,
        })

        # Official FastPLAID (Rust+CUDA) curve
        fp2 = B4OfficialFastPlaid(d_toks, d_offs, device=device, nbits=2, kmeans_niters=3)
        for np_, nc_ in [(1, 128), (2, 256), (4, 512), (8, 1024), (16, 4096)]:
            t_r, _ = time_cuda_fn(lambda n1=np_, n2=nc_: fp2.search(q_sub, q_offs, n_ivf_probe=n1, n_full_scores=n2, topk=10), warmup=2, repeats=7, device=device)
            or_rec, ir = eval_256(lambda st, so, n1=np_, n2=nc_: fp2.search(st, so, n_ivf_probe=n1, n_full_scores=n2, topk=10))
            records.append({
                "dataset": ds, "method": "B4_Official_FastPLAID", "config": f"nprobe={np_},ndocs={nc_}",
                "qps": B * 1000.0 / t_r["median_ms"], "median_ms": t_r["median_ms"], "ci95_ms": t_r["ci95_ms"],
                "oracle_topk_recall": or_rec, "ir": ir,
            })
        fp2.cleanup()
        del fp2
        torch.cuda.empty_cache()

        # SIGIR'25 WARP curve
        b9 = B9WarpGPU(d_toks, d_offs, device=device, n_centroids=n_cent)
        for tp_ in [2, 4, 8, 16, 32]:
            t_r, _ = time_cuda_fn(lambda tp=tp_: b9.search(q_sub, q_offs, t_prime=tp, topk=10), warmup=2, repeats=7, device=device)
            or_rec, ir = eval_256(lambda st, so, tp=tp_: b9.search(st, so, t_prime=tp, topk=10))
            records.append({
                "dataset": ds, "method": "B9_WARP_GPU", "config": f"t_prime={tp_}",
                "qps": B * 1000.0 / t_r["median_ms"], "median_ms": t_r["median_ms"], "ci95_ms": t_r["ci95_ms"],
                "oracle_topk_recall": or_rec, "ir": ir,
            })
        del b9
        torch.cuda.empty_cache()

        # ECIR'24 EMVB curve (Bit-Vector Popcount + PQ16 ADC + Exact Refinement)
        b10 = B10EmvbGPU(d_toks, d_offs, device=device, n_centroids=n_cent)
        for np_, nc_, nr_ in [(2, 256, 32), (4, 512, 64), (8, 1024, 128), (16, 2048, 256), (32, 4096, 512)]:
            t_r, _ = time_cuda_fn(lambda n1=np_, n2=nc_, n3=nr_: b10.search(q_sub, q_offs, nprobe=n1, ncands=n2, nrefine=n3, topk=10), warmup=2, repeats=7, device=device)
            or_rec, ir = eval_256(lambda st, so, n1=np_, n2=nc_, n3=nr_: b10.search(st, so, nprobe=n1, ncands=n2, nrefine=n3, topk=10))
            records.append({
                "dataset": ds, "method": "B10_EMVB_GPU", "config": f"nprobe={np_},C={nc_},R={nr_}",
                "qps": B * 1000.0 / t_r["median_ms"], "median_ms": t_r["median_ms"], "ci95_ms": t_r["ci95_ms"],
                "oracle_topk_recall": or_rec, "ir": ir,
            })
        del b10
        torch.cuda.empty_cache()

        b4 = B4PlaidGPU(d_toks, d_offs, device=device, n_centroids=n_cent)
        for np_, nc_ in [(1, 64), (2, 128), (4, 256), (4, 512), (8, 1024), (16, 2048)]:
            t_r, _ = time_cuda_fn(lambda n1=np_, n2=nc_: b4.search(q_sub, q_offs, nprobe=n1, ncands=n2, topk=10), warmup=2, repeats=7, device=device)
            or_rec, ir = eval_256(lambda st, so, n1=np_, n2=nc_: b4.search(st, so, nprobe=n1, ncands=n2, topk=10))
            records.append({
                "dataset": ds, "method": "B4_PLAID_GPU", "config": f"nprobe={np_},C={nc_}",
                "qps": B * 1000.0 / t_r["median_ms"], "median_ms": t_r["median_ms"], "oracle_topk_recall": or_rec, "ir": ir,
            })
            if nc_ >= 1024:
                t_h, _ = time_cuda_fn(lambda n1=np_, n2=nc_: mb8.search_rerank(q_sub, q_offs, b4.get_candidates(q_sub, q_offs, nprobe=n1, ncands=n2), topk=10, c_warm=16), warmup=2, repeats=7, device=device)
                records.append({
                    "dataset": ds, "method": "PLAID_plus_MAXBOUND", "config": f"nprobe={np_},C={nc_}",
                    "qps": B * 1000.0 / t_h["median_ms"], "median_ms": t_h["median_ms"], "oracle_topk_recall": or_rec, "ir": ir,
                })
        del b4
        torch.cuda.empty_cache()

        b5 = B5MuveraGPU(d_toks, d_offs, device=device, r_reps=8, k_sim=4, d_proj=16)
        for nc_ in [64, 128, 256, 512, 1024, 2048, 4096]:
            t_r, _ = time_cuda_fn(lambda n2=nc_: b5.search(q_sub, q_offs, ncands=n2, topk=10), warmup=2, repeats=7, device=device)
            or_rec, ir = eval_256(lambda st, so, n2=nc_: b5.search(st, so, ncands=n2, topk=10))
            records.append({
                "dataset": ds, "method": "B5_MUVERA_GPU", "config": f"C={nc_}",
                "qps": B * 1000.0 / t_r["median_ms"], "median_ms": t_r["median_ms"], "oracle_topk_recall": or_rec, "ir": ir,
            })
            if nc_ >= 1024:
                t_h, _ = time_cuda_fn(lambda n2=nc_: mb8.search_rerank(q_sub, q_offs, b5.get_candidates(q_sub, q_offs, ncands=n2), topk=10, c_warm=16), warmup=2, repeats=7, device=device)
                records.append({
                    "dataset": ds, "method": "MUVERA_plus_MAXBOUND", "config": f"C={nc_}",
                    "qps": B * 1000.0 / t_h["median_ms"], "median_ms": t_h["median_ms"], "oracle_topk_recall": or_rec, "ir": ir,
                })
        del b5, mb8
        torch.cuda.empty_cache()

    out_path = os.path.expanduser("~/maxbound/results/e2_pareto_frontier.json")
    write_result_json(out_path, records, {"experiment": "E2_pareto_frontier_4datasets_with_fastplaid_and_warp", "device": device})
    print(f"Saved {out_path}")


# ==============================================================================
# E4: Heterogeneous Dual-GPU (A100 + RTX 4080) across FiQA, HotpotQA, MS MARCO, and WildChat-Scale50M!
# ==============================================================================
def run_e4_v5():
    datasets = [
        ("fiqa", 64, False),
        ("hotpotqa", 64, False),
        ("msmarco", 64, True),
        ("scale_50m", 16, True),
        ("msmarco_2m", 16, True),
    ]
    records = []

    for ds, B, offload_4080 in datasets:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            continue
        torch.cuda.empty_cache()
        d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
        d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
        q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
        q_offs = np.load(os.path.join(ddir, "query_offsets.int64.npy"))[:B + 1]
        q_sub = q_toks[:int(q_offs[-1])]
        N_docs = len(d_offs) - 1
        N_toks = d_toks.shape[0]
        print(f"\n=== E4 Dual-GPU [{ds}] B={B} N_docs={N_docs} N_toks={N_toks} ===")

        oracle_ids, oracle_scores = exact_maxsim_oracle(
            q_sub.float(), q_offs, d_toks, d_offs, topk=10, doc_batch_size=2048, device=1, use_fp64=True
        )

        def time_wall(fn, warmup=2, repeats=9):
            for _ in range(warmup):
                out = fn()
            ts = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                out = fn()
                ts.append((time.perf_counter() - t0) * 1000.0)
            arr = np.array(ts, dtype=np.float64)
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            return {
                "runs_ms": [round(float(x), 4) for x in ts],
                "median_ms": float(np.median(arr)),
                "mean_ms": float(np.mean(arr)),
                "ci95_ms": round(float(1.96 * std / np.sqrt(len(arr))), 4),
                "p99_ms": float(np.percentile(arr, 99)),
            }, out

        reps_e4 = 7 if N_toks > 80_000_000 else 9
        b2_1 = B2CuBLASUnpaddedExact(d_toks, d_offs, device=1)
        t_b2, (ib2, sb2) = time_cuda_fn(lambda: b2_1.search(q_sub, q_offs, topk=10), warmup=2, repeats=reps_e4, device=1)
        records.append({
            "dataset": ds, "batch": B, "config": "A100_Only_B2",
            **t_b2, "qps": B * 1000.0 / t_b2["median_ms"],
            "audit": audit_exact_topk(ib2, sb2, oracle_ids, oracle_scores, tol=3e-3),
        })
        del b2_1; torch.cuda.empty_cache()

        mb_1 = MaxBoundIndex(d_toks, d_offs, device=1, mode="int8")
        t_mb1, (im1, sm1, inf1) = time_cuda_fn(lambda: mb_1.search(q_sub, q_offs, topk=10, c_warm=32), warmup=2, repeats=reps_e4, device=1)
        records.append({
            "dataset": ds, "batch": B, "config": "A100_Only_MAXBOUND",
            **t_mb1, "qps": B * 1000.0 / t_mb1["median_ms"],
            "survivor_pct": round(100.0 * inf1["survivor_ratio"], 4),
            "audit": audit_exact_topk(im1, sm1, oracle_ids, oracle_scores, tol=3e-3),
        })
        del mb_1; torch.cuda.empty_cache()

        # On 130M-token msmarco_2m, INT8 (~18.8 GB) & INT6 (~14.5 GB + workspace) exceed 15.57 GB 4080, so single-4080 uses INT4_Offload!
        mode_4080_single = "int4" if N_toks > 80_000_000 else "int8"
        mb_0 = MaxBoundIndex(d_toks, d_offs, device=0, mode=mode_4080_single, host_offload_fp16=offload_4080)
        t_mb0, (im0, sm0, inf0) = time_cuda_fn(lambda: mb_0.search(q_sub, q_offs, topk=10, c_warm=32), warmup=2, repeats=reps_e4, device=0)
        records.append({
            "dataset": ds, "batch": B, "config": "4080_Only_MAXBOUND", "mode_4080": mode_4080_single,
            **t_mb0, "qps": B * 1000.0 / t_mb0["median_ms"],
            "survivor_pct": round(100.0 * inf0["survivor_ratio"], 4),
            "audit": audit_exact_topk(im0, sm0, oracle_ids, oracle_scores, tol=3e-3),
        })
        del mb_0; torch.cuda.empty_cache()

        # Compare 4 genuine dual-GPU schedulers:
        # 1) B7_Naive_DualGPU (50/50 contiguous split, no stealing)
        # 2) B7_Peak_TFLOPS_Split (0.615 hardware INT8 TOPS ratio: A100 624 vs 4080 390 TOPS)
        # 3) B7_HBM_BW_Split (0.730 hardware HBM BW ratio: A100 1935 vs 4080 717 GB/s)
        # 4) MAXBOUND_DualGPU (M3 zero-leakage analytical split + length-interleaved + offload work-stealing & tau sync!)
        dual_cfgs = [
            ("B7_Naive_DualGPU", 0.50, False, False, False),
            ("B7_Peak_TFLOPS_Split", 0.615, False, False, False),
            ("B7_HBM_BW_Split", 0.730, False, False, False),
            ("MAXBOUND_DualGPU", None, True, True, True),
        ]
        for cname, ratio_in, sync_tau, steal, interleave in dual_cfgs:
            est_r = ratio_in if ratio_in is not None else (0.65 if N_toks > 80_000_000 else 0.44)
            need_off0 = bool(offload_4080 and ((1.0 - est_r + (0.20 if steal else 0.0)) * (N_toks * 128 * 3.2 / (1024 ** 3)) > 10.5))
            deng = DualGpuMaxBoundEngine(
                d_toks, d_offs, split_ratio_gpu1=ratio_in,
                mode_gpu1="int8", mode_gpu0="int8",
                host_offload_gpu1=False, host_offload_gpu0=need_off0,
                enable_work_stealing=steal, interleave_by_doc=interleave
            )
            t_d, (id_d, sc_d, inf_d) = time_wall(
                lambda: deng.search(q_sub, q_offs, topk=10, c_warm=32, sync_tau=sync_tau),
                warmup=2, repeats=reps_e4
            )
            aud = audit_exact_topk(id_d, sc_d, oracle_ids, oracle_scores, tol=3e-3)
            assert aud["agreement"] == 1.0, f"E4 {ds} {cname} FAILED audit: {aud}"
            imbal = max(inf_d["t_gpu1_ms"], inf_d["t_gpu0_ms"]) / max(min(inf_d["t_gpu1_ms"], inf_d["t_gpu0_ms"]), 1e-3)
            records.append({
                "dataset": ds, "batch": B, "config": cname,
                "split_gpu1": inf_d["split_ratio_gpu1"], "sync_tau": sync_tau, "work_stealing": steal,
                "offload_gpu0": need_off0, **t_d, "qps": B * 1000.0 / t_d["median_ms"],
                "t_gpu1_ms": round(inf_d["t_gpu1_ms"], 3), "t_gpu0_ms": round(inf_d["t_gpu0_ms"], 3),
                "shards_stolen_gpu1": inf_d.get("shards_stolen_gpu1", 0),
                "shards_stolen_gpu0": inf_d.get("shards_stolen_gpu0", 0),
                "straggler_imbalance": round(imbal, 3),
                "survivor_pct": round(100.0 * inf_d["survivor_ratio"], 4),
                "audit": aud,
            })
            print(f"  {cname:22s}: {t_d['median_ms']:6.2f} ms ({B*1000/t_d['median_ms']:6.1f} QPS) | split={inf_d['split_ratio_gpu1']:.2f} | t1={inf_d['t_gpu1_ms']:.1f} t0={inf_d['t_gpu0_ms']:.1f} (imbal={imbal:.2f}) | stolen=({inf_d.get('shards_stolen_gpu1',0)},{inf_d.get('shards_stolen_gpu0',0)})")
            del deng
            torch.cuda.empty_cache()

    out_path = os.path.expanduser("~/maxbound/results/e4_dual_gpu.json")
    write_result_json(out_path, records, {
        "experiment": "E4_heterogeneous_dual_gpu_with_msmarco_and_wildchat_scale50m",
        "scale_50m_provenance": "allenai/WildChat-1M (980,000 real passages + 512 real queries encoded by ColBERTv2.0, 60,894,306 tokens, CV=1.245)",
    })
    print(f"Saved {out_path}")


# ==============================================================================
# E6: 5-Step Pure-CUDA Monotonic Ablation + Stage Breakdown + 11-Dataset 4-Slice Table
# ==============================================================================
def run_e6_v5(device=1):
    ablation_records = []
    for ds in ["fiqa", "longbench", "hotpotqa", "msmarco", "fiqa_bgem3"]:
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
        D = d_toks.shape[1]
        tol = 5e-3 if D == 1024 else 2e-3
        oracle_ids, oracle_scores = exact_maxsim_oracle(q_sub.float(), q_offs, d_toks, d_offs, topk=10, device=device, use_fp64=True)

        b2 = B2CuBLASUnpaddedExact(d_toks, d_offs, device=device)
        t_v0, (i0, s0) = time_cuda_fn(lambda: b2.search(q_sub, q_offs, topk=10), 3, 11, device)
        assert audit_exact_topk(i0, s0, oracle_ids, oracle_scores, tol=tol)["agreement"] == 1.0
        del b2; torch.cuda.empty_cache()

        mb_unsorted = MaxBoundIndex(d_toks, d_offs, device=device, mode="int8", sort_by_length=False)
        t_v1, (i1, s1, inf1) = time_cuda_fn(
            lambda: mb_unsorted.search(q_sub, q_offs, topk=10, c_warm=10, force_w1=True, disable_multipass=True, disable_lb_tighten=True), 2, 9, device
        )
        assert audit_exact_topk(i1, s1, oracle_ids, oracle_scores, tol=tol)["agreement"] == 1.0
        t_v2, (i2, s2, inf2) = time_cuda_fn(
            lambda: mb_unsorted.search(q_sub, q_offs, topk=10, c_warm=32, force_w1=True, disable_multipass=True, disable_lb_tighten=False), 2, 9, device
        )
        assert audit_exact_topk(i2, s2, oracle_ids, oracle_scores, tol=tol)["agreement"] == 1.0
        t_v3, (i3, s3, inf3) = time_cuda_fn(
            lambda: mb_unsorted.search(q_sub, q_offs, topk=10, c_warm=32, force_w1=False, disable_multipass=True), 2, 9, device
        )
        assert audit_exact_topk(i3, s3, oracle_ids, oracle_scores, tol=tol)["agreement"] == 1.0
        del mb_unsorted; torch.cuda.empty_cache()

        mb_sorted = MaxBoundIndex(d_toks, d_offs, device=device, mode="int8", sort_by_length=True)
        t_v4, (i4, s4, inf4) = time_cuda_fn(
            lambda: mb_sorted.search(q_sub, q_offs, topk=10, c_warm=32, force_w1=False, disable_multipass=False), 3, 11, device
        )
        _, _, inf4_m = mb_sorted.search(q_sub, q_offs, topk=10, c_warm=32, return_tested_mask=True)
        aud4 = audit_exact_topk(i4, s4, oracle_ids, oracle_scores, tol=tol, tested_mask=inf4_m["tested_mask_orig"])
        assert aud4["agreement"] == 1.0 and aud4["false_negative_prunes"] == 0

        def run_v4_breakdown():
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            ev[0].record()
            _, m_max, m_pad, ql, qp, qpk, qsc, qn, eps_b = mb_sorted._prepare_queries(q_sub, q_offs, pad_mult=32)
            wi, ws, tau = mb_sorted._stage0_prewarm(qp, ql, B, 10, 32)
            ev[1].record()
            ub, lb, act = mb_sorted._get_workspace(B)
            act.scatter_(1, wi, 0)
            n_passes = m_pad // 32
            for p in range(n_passes):
                rem = torch.zeros(B, device=device)
                mb_sorted._exec_int8_pass32(
                    qpk, qsc, qn, tau, ub, lb, rem, act, p * 32, 0.0,
                    1 if p == 0 else 0, 1 if n_passes > 1 else 0, force_w1=False,
                    q_fp16_pad=qp, enable_2slice=True, eps_buf=eps_b, is_packed_hd=False
                )
            lb_for_tau = lb.scatter(1, wi, ws)
            tau = torch.maximum(tau, torch.topk(lb_for_tau, k=10, dim=1).values[:, -1])
            ev[2].record()
            mask = (act > 0) & (ub >= (tau[:, None] - eps_b[:, None]))
            nz = torch.nonzero(mask, as_tuple=False)
            ub.fill_(-1e9); ub.scatter_(1, wi, ws)
            if nz.numel() > 0:
                sq, sd = nz[:, 0].to(torch.int32).contiguous(), nz[:, 1].to(torch.int32).contiguous()
                ub[sq.long(), sd.long()] = mb_sorted._verify_pairs(sq, sd, qp, ql)
            ev[3].record()
            v, idx = torch.topk(ub, k=10, dim=1)
            idx = mb_sorted.perm_to_orig_gpu[idx]
            _ = idx.cpu().numpy(), v.cpu().numpy()
            ev[4].record()
            torch.cuda.synchronize(device)
            return [ev[k].elapsed_time(ev[k + 1]) for k in range(4)]

        for _ in range(2):
            run_v4_breakdown()
        bks = np.median(np.array([run_v4_breakdown() for _ in range(7)]), axis=0)

        v0_ms = t_v0["median_ms"]
        rec = {
            "dataset": ds, "dim": D, "batch": B,
            "V0_B2_Unpadded_ms": v0_ms,
            "V1_TC_W1_NoPrewarm_ms": t_v1["median_ms"],
            "V2_TC_W1_Prewarm_ms": t_v2["median_ms"],
            "V3_TC_W16_Unsorted_ms": t_v3["median_ms"],
            "V4_Full_MAXBOUND_ms": t_v4["median_ms"],
            "V4_ci95_ms": t_v4["ci95_ms"],
            "speedup_steps": {
                "V0_Baseline": 1.000,
                "V1_plus_TC_Bound": round(v0_ms / t_v1["median_ms"], 3),
                "V2_plus_Prewarm": round(v0_ms / t_v2["median_ms"], 3),
                "V3_plus_SMEM_W16": round(v0_ms / t_v3["median_ms"], 3),
                "V4_plus_SkewSort": round(v0_ms / t_v4["median_ms"], 3),
            },
            "survivor_pct_steps": {
                "V1_no_prewarm": round(inf1["survivor_ratio"] * 100.0, 4),
                "V2_with_prewarm": round(inf2["survivor_ratio"] * 100.0, 4),
                "V4_full": round(inf4["survivor_ratio"] * 100.0, 4),
            },
            "stage_breakdown_ms": {
                "query_prep_and_prewarm": round(float(bks[0]), 4),
                "stage1_fused_tc_filter": round(float(bks[1]), 4),
                "stage2_exact_verify": round(float(bks[2]), 4),
                "topk_and_d2h": round(float(bks[3]), 4),
            },
            "audit": aud4,
            "exact": True,
        }
        ablation_records.append(rec)
        print(f"[E6A {ds:11s}] V0=1.00x ({v0_ms:.2f}ms) -> V1={rec['speedup_steps']['V1_plus_TC_Bound']:.2f}x -> V2={rec['speedup_steps']['V2_plus_Prewarm']:.2f}x -> V3={rec['speedup_steps']['V3_plus_SMEM_W16']:.2f}x -> V4={rec['speedup_steps']['V4_plus_SkewSort']:.2f}x ({t_v4['median_ms']:.2f}ms)")
        del mb_sorted
        torch.cuda.empty_cache()

    slice_records = []
    for ds in ALL_11_DATASETS:
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
        D = d_toks.shape[1]
        tol = 5e-3 if D == 1024 else 2e-3

        oi, os_ = exact_maxsim_oracle(q_sub.float(), q_offs, d_toks, d_offs, topk=10, device=device, use_fp64=True)
        mb = MaxBoundIndex(d_toks, d_offs, device=device, mode="int8")
        pi, ps, sl_info = mb.profile_slice_pruning(q_sub, q_offs, topk=10, c_warm=32)
        aud = audit_exact_topk(pi, ps, oi, os_, tol=tol)
        assert aud["agreement"] == 1.0
        slice_records.append({
            "dataset": ds, "dim": D, "batch": B_eff, "exact": True,
            **sl_info, "audit": aud,
        })
        print(f"[E6C Slice Table {ds:13s}] S1={sl_info['slice1_prune_pct']:5.2f}% | S2={sl_info['slice2_prune_pct']:5.2f}% | S3={sl_info['slice3_prune_pct']:5.2f}% | S4={sl_info['slice4_prune_pct']:6.3f}% | Avg={sl_info['avg_slice_prune_pct']:5.2f}% | TokSkip={sl_info['exact_tok_pair_skip_pct']:.3f}%")
        del mb
        torch.cuda.empty_cache()

    out_path = os.path.expanduser("~/maxbound/results/e6_ablation_breakdown.json")
    write_result_json(out_path, ablation_records, {
        "experiment": "E6_ablation_stage_and_11dataset_slice_table",
        "device": device,
        "slice_pruning_table": slice_records,
    })
    print(f"Saved {out_path}")


# ==============================================================================
# E7: Strictly Zero-Leakage Analytical Cost Model Validation
#     - Calibrated ONLY on synthetic multi-vector clusters (seed=1001) via NNLS
#     - Evaluated on 8 Development Datasets + 3 Strictly Unseen Blind Test Datasets (hotpotqa, trec_covid, msmarco)
#     - Reports BOTH deduplicated unique (dataset, GPU, B, feasible_regime) metrics AND full quota-grid metrics!
# ==============================================================================
def run_e7_zero_leakage():
    from scipy.optimize import nnls
    rng = np.random.default_rng(1001)
    calib_points = []
    offload_samples = {0: [], 1: []}

    print("\n=== E7 Phase 1: Synthetic-Only Calibration (seed=1001, ZERO real-dataset access) ===")
    for D in [128, 1024]:
        K_syn = 128
        centers = rng.standard_normal((K_syn, D)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        cfgs = (
            [(3000, 65, 32), (3000, 180, 64), (16000, 75, 32), (16000, 170, 32), (48000, 110, 32), (120000, 80, 32)]
            if D == 128 else
            [(3000, 85, 32), (9000, 110, 32), (24000, 95, 32), (45000, 90, 32)]
        )
        for n_docs, mean_l, m_syn in cfgs:
            lens = np.clip(rng.integers(max(16, mean_l - 40), mean_l + 45, size=n_docs), 16, 300)
            d_offs = np.zeros(n_docs + 1, dtype=np.int64); np.cumsum(lens, out=d_offs[1:])
            doc_topics = np.repeat(np.arange(n_docs, dtype=np.int64) % K_syn, lens)
            x = centers[doc_topics] + 0.36 * rng.standard_normal((int(d_offs[-1]), D)).astype(np.float32)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            d_toks = torch.from_numpy(x).half()
            q_offs = np.arange(0, 65 * m_syn, m_syn, dtype=np.int64)
            q_topics = np.repeat(np.arange(64, dtype=np.int64) % K_syn, m_syn)
            qx = centers[q_topics] + 0.36 * rng.standard_normal((int(q_offs[-1]), D)).astype(np.float32)
            qx /= np.linalg.norm(qx, axis=1, keepdims=True)
            q_toks = torch.from_numpy(qx).half()

            for dev in [0, 1]:
                if D == 1024 and n_docs > 12000 and dev == 0:
                    continue
                if D == 128 and n_docs > 100000 and dev == 0:
                    continue
                torch.cuda.empty_cache()
                rec_map = {B: {} for B in [1, 8, 16, 32, 64]}
                makers = [
                    ("Dense_Unpadded", lambda: B2CuBLASUnpaddedExact(d_toks, d_offs, device=dev)),
                    ("Fused_FP16_TC", lambda: B8FusedFP16TCExact(d_toks, d_offs, device=dev)),
                    ("MB_INT8", lambda: MaxBoundIndex(d_toks, d_offs, device=dev, mode="int8")),
                ]
                if D == 128:
                    makers.append(("MB_INT6", lambda: MaxBoundIndex(d_toks, d_offs, device=dev, mode="int6")))
                for pname, maker in makers:
                    eng = maker()
                    for B in [1, 8, 16, 32, 64]:
                        qo = q_offs[:B + 1]; qt = q_toks[:int(qo[-1])]
                        fn = (lambda e=eng, q=qt, o=qo: e.search(q, o, 10)) if pname in ("Dense_Unpadded", "Fused_FP16_TC") else (lambda e=eng, q=qt, o=qo: e.search(q, o, 10, 32))
                        rec_map[B][pname] = time_cuda_fn(fn, 2, 5, dev)[0]["median_ms"]
                    del eng
                    torch.cuda.empty_cache()

                if n_docs >= 16000 and D == 128:
                    eng_off = MaxBoundIndex(d_toks, d_offs, device=dev, mode="int8", host_offload_fp16=True)
                    for B in [8, 32, 64]:
                        qo = q_offs[:B + 1]; qt = q_toks[:int(qo[-1])]
                        t_off, (_, _, inf_o) = time_cuda_fn(lambda e=eng_off, q=qt, o=qo: e.search(q, o, 10, 32), 1, 4, dev)
                        extra_ms = max(0.05, t_off["median_ms"] - rec_map[B]["MB_INT8"])
                        h2d_bytes = inf_o["h2d_mb"] * 1024.0 * 1024.0
                        if h2d_bytes > 65536:
                            offload_samples[dev].append((h2d_bytes / 1e6) / extra_ms)
                    del eng_off
                    torch.cuda.empty_cache()

                for B in [1, 8, 16, 32, 64]:
                    calib_points.append({
                        "dev": dev, "D": D, "n_docs": n_docs, "n_toks": int(d_offs[-1]), "B": B, "m": m_syn,
                        "Dense_Unpadded": rec_map[B]["Dense_Unpadded"],
                        "Fused_FP16_TC": rec_map[B]["Fused_FP16_TC"],
                        "MB_INT8": rec_map[B]["MB_INT8"],
                        "MB_INT6": rec_map[B].get("MB_INT6"),
                    })

    planner = MaxBoundPlanner()
    model_specs = [
        ("Dense_Unpadded", "Dense_Unpadded", 128),
        ("Dense_Unpadded_HD", "Dense_Unpadded", 1024),
        ("Fused_FP16_TC", "Fused_FP16_TC", 128),
        ("Fused_FP16_TC_HD", "Fused_FP16_TC", 1024),
        ("MB_INT8", "MB_INT8", 128),
        ("MB_INT6", "MB_INT6", 128),
        ("MB_INT8_HD", "MB_INT8", 1024),
    ]
    for dev in [0, 1]:
        dev_str = str(dev)
        if offload_samples[dev]:
            planner.params[dev_str]["offload_eff_gbs"] = round(float(np.median(offload_samples[dev])), 4)
        pts_dev = [p for p in calib_points if p["dev"] == dev]
        for mkey, ykey, dim_filter in model_specs:
            sub_pts = [p for p in pts_dev if p["D"] == dim_filter and p.get(ykey) is not None]
            if not sub_pts:
                continue
            X = np.stack([planner.features(mkey, p["n_toks"], p["n_docs"], p["B"], p["m"], D=p["D"], device=dev) for p in sub_pts], axis=0)
            y = np.array([p[ykey] for p in sub_pts], dtype=np.float64)
            w_irwls = 1.0 / np.maximum(y, 0.25)
            coef, _ = nnls(X * w_irwls[:, None], y * w_irwls)
            planner.params[dev_str][mkey] = {"w": [round(float(c), 6) for c in coef]}

    # Freeze synthetic-only calibration BEFORE touching any real dataset!
    calib_path = os.path.expanduser("~/maxbound/results/m3_calibration.json")
    with open(calib_path, "w") as f:
        json.dump({
            "calibration_protocol": "strictly_synthetic_nnls_zero_real_data_leakage",
            "synthetic_seed": 1001,
            "num_synthetic_calib_points": len(calib_points),
            "surv_coef": planner.surv_coef,
            "params": planner.params,
        }, f, indent=2)

    print("=== E7 Phase 2: Frozen Model Evaluation on 8 Dev Datasets + 3 Strictly Unseen Blind Test Datasets ===")
    dev_datasets = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "longbench", "scifact_bgem3", "fiqa_bgem3"]
    blind_datasets = ["hotpotqa", "trec_covid", "msmarco"]
    all_eval_ds = [(ds, "dev") for ds in dev_datasets] + [(ds, "blind_test") for ds in blind_datasets]

    eval_records = []
    seen_unique_regimes = set()

    for ds, split_tag in all_eval_ds:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            continue
        with open(os.path.join(ddir, "meta.json")) as f:
            meta = json.load(f)
        d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
        d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
        q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
        q_offs_all = np.load(os.path.join(ddir, "query_offsets.int64.npy"))
        N_docs, N_toks, cv, D = meta["num_docs"], meta["total_doc_tokens"], meta["doc_len_cv"], meta["dim"]
        m_len = int(np.diff(q_offs_all[:2])[0])

        for dev in [0, 1]:
            torch.cuda.empty_cache()
            actual_map = {B: {} for B in [1, 8, 16, 32, 64]}
            plan_makers = [
                ("Dense_Unpadded", lambda: B2CuBLASUnpaddedExact(d_toks, d_offs, device=dev)),
                ("Fused_FP16_TC", lambda: B8FusedFP16TCExact(d_toks, d_offs, device=dev)),
                ("MB_INT8", lambda: MaxBoundIndex(d_toks, d_offs, device=dev, mode="int8")),
                ("MB_INT8_Offload", lambda: MaxBoundIndex(d_toks, d_offs, device=dev, mode="int8", host_offload_fp16=True)),
            ]
            if D == 128:
                plan_makers.append(("MB_INT6", lambda: MaxBoundIndex(d_toks, d_offs, device=dev, mode="int6")))

            phys_vram = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
            for p_name, maker in plan_makers:
                if planner.estimate_vram_gb(p_name, N_toks, N_docs, 64, m_len, D=D) > phys_vram * 0.96:
                    continue
                try:
                    eng = maker()
                    for B in [1, 8, 16, 32, 64]:
                        B_eff = min(B, len(q_offs_all) - 1)
                        qo = q_offs_all[:B_eff + 1]; qt = q_toks[:int(qo[-1])]
                        fn = (lambda e=eng, q=qt, o=qo: e.search(q, o, 10)) if p_name in ("Dense_Unpadded", "Fused_FP16_TC") else (lambda e=eng, q=qt, o=qo: e.search(q, o, 10, 32))
                        actual_map[B][p_name] = time_cuda_fn(fn, 2, 5, dev)[0]["median_ms"]
                    del eng
                    torch.cuda.empty_cache()
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    torch.cuda.empty_cache()

            for B in [1, 8, 16, 32, 64]:
                B_eff = min(B, len(q_offs_all) - 1)
                actual = actual_map[B]
                for vram_b in [8.0, 16.0, 80.0]:
                    feas = {k: v for k, v in actual.items() if planner.estimate_vram_gb(k, N_toks, N_docs, B_eff, m_len, D=D) <= vram_b * 0.92}
                    if not feas:
                        continue
                    sel_plan, pred_ms, _ = planner.select_plan(
                        N_toks, N_docs, B_eff, m_len, cv, dev, vram_b, D=D, allowed_plans=list(feas.keys())
                    )
                    f_best = min(feas, key=feas.get)
                    act_sel_ms = feas[sel_plan]
                    ape = abs(pred_ms - act_sel_ms) / max(act_sel_ms, 1e-3) * 100.0
                    hit = (sel_plan == f_best) or (act_sel_ms <= feas[f_best] * 1.10)
                    regime_sig = (ds, dev, B_eff, tuple(sorted(feas.keys())))
                    is_unique_regime = regime_sig not in seen_unique_regimes
                    seen_unique_regimes.add(regime_sig)

                    # Compare M3 analytical selector against 3 simple heuristic rules!
                    def _pick_first(prefs):
                        for pr in prefs:
                            if pr in feas:
                                return pr
                        return sel_plan

                    r1_plan = _pick_first(["MB_INT8", "MB_INT6", "MB_INT8_Offload", "Fused_FP16_TC", "Dense_Unpadded"])
                    r2_plan = _pick_first(["Fused_FP16_TC", "Dense_Unpadded", "MB_INT8", "MB_INT8_Offload"])
                    r3_plan = _pick_first(
                        ["Fused_FP16_TC", "MB_INT8", "MB_INT8_Offload"] if B_eff == 1 else ["MB_INT8", "MB_INT6", "MB_INT8_Offload", "Fused_FP16_TC"]
                    )
                    best_ms = max(feas[f_best], 1e-4)
                    eval_records.append({
                        "dataset": ds, "split": split_tag, "dim": D, "device": dev, "B": B_eff, "vram_budget": vram_b,
                        "is_unique_feasible_regime": is_unique_regime,
                        "feasible_plans": sorted(feas.keys()),
                        "selected_plan": sel_plan, "oracle_best_plan": f_best,
                        "pred_ms": round(pred_ms, 3), "actual_sel_ms": round(act_sel_ms, 3),
                        "oracle_best_ms": round(feas[f_best], 3), "ape_pct": round(ape, 2),
                        "m3_regret_pct": round((act_sel_ms - feas[f_best]) / best_ms * 100.0, 2),
                        "rule1_always_mb8_regret_pct": round((feas[r1_plan] - feas[f_best]) / best_ms * 100.0, 2),
                        "rule2_always_fp16tc_regret_pct": round((feas[r2_plan] - feas[f_best]) / best_ms * 100.0, 2),
                        "rule3_b1_fp16_else_mb8_regret_pct": round((feas[r3_plan] - feas[f_best]) / best_ms * 100.0, 2),
                        "optimal_or_near_hit": bool(hit), "exact_hit": bool(sel_plan == f_best),
                        "rule1_exact_hit": bool(r1_plan == f_best),
                        "rule2_exact_hit": bool(r2_plan == f_best),
                        "rule3_exact_hit": bool(r3_plan == f_best),
                    })

    def summarize_slice(rows):
        if not rows:
            return {}
        apes = [r["ape_pct"] for r in rows]
        return {
            "count": len(rows),
            "median_ape_pct": round(float(np.median(apes)), 3),
            "mean_ape_pct": round(float(np.mean(apes)), 3),
            "near_optimal_hit_rate_pct": round(float(np.mean([r["optimal_or_near_hit"] for r in rows]) * 100.0), 2),
            "exact_hit_rate_pct": round(float(np.mean([r["exact_hit"] for r in rows]) * 100.0), 2),
            "m3_mean_regret_pct": round(float(np.mean([r["m3_regret_pct"] for r in rows])), 2),
            "m3_p95_regret_pct": round(float(np.percentile([r["m3_regret_pct"] for r in rows], 95)), 2),
            "heuristic_comparison": {
                "M3_Analytical_Planner": {
                    "exact_hit_pct": round(float(np.mean([r["exact_hit"] for r in rows]) * 100.0), 2),
                    "mean_regret_pct": round(float(np.mean([r["m3_regret_pct"] for r in rows])), 2),
                    "p95_regret_pct": round(float(np.percentile([r["m3_regret_pct"] for r in rows], 95)), 2),
                },
                "Rule1_Always_MB_INT8": {
                    "exact_hit_pct": round(float(np.mean([r["rule1_exact_hit"] for r in rows]) * 100.0), 2),
                    "mean_regret_pct": round(float(np.mean([r["rule1_always_mb8_regret_pct"] for r in rows])), 2),
                    "p95_regret_pct": round(float(np.percentile([r["rule1_always_mb8_regret_pct"] for r in rows], 95)), 2),
                },
                "Rule2_Always_Fused_FP16": {
                    "exact_hit_pct": round(float(np.mean([r["rule2_exact_hit"] for r in rows]) * 100.0), 2),
                    "mean_regret_pct": round(float(np.mean([r["rule2_always_fp16tc_regret_pct"] for r in rows])), 2),
                    "p95_regret_pct": round(float(np.percentile([r["rule2_always_fp16tc_regret_pct"] for r in rows], 95)), 2),
                },
                "Rule3_Batch1_FP16_Else_MB8": {
                    "exact_hit_pct": round(float(np.mean([r["rule3_exact_hit"] for r in rows]) * 100.0), 2),
                    "mean_regret_pct": round(float(np.mean([r["rule3_b1_fp16_else_mb8_regret_pct"] for r in rows])), 2),
                    "p95_regret_pct": round(float(np.percentile([r["rule3_b1_fp16_else_mb8_regret_pct"] for r in rows], 95)), 2),
                },
            },
        }

    all_sum = summarize_slice(eval_records)
    dedup_sum = summarize_slice([r for r in eval_records if r["is_unique_feasible_regime"]])
    blind_all_sum = summarize_slice([r for r in eval_records if r["split"] == "blind_test"])
    blind_dedup_sum = summarize_slice([r for r in eval_records if r["split"] == "blind_test" and r["is_unique_feasible_regime"]])
    dev_dedup_sum = summarize_slice([r for r in eval_records if r["split"] == "dev" and r["is_unique_feasible_regime"]])

    summary = {
        "num_heldout_configs": all_sum["count"],
        "median_ape_pct": all_sum["median_ape_pct"],
        "mean_ape_pct": all_sum["mean_ape_pct"],
        "near_optimal_hit_rate_pct": all_sum["near_optimal_hit_rate_pct"],
        "exact_hit_rate_pct": all_sum["exact_hit_rate_pct"],
        "deduplicated_unique_regimes": dedup_sum,
        "blind_test_all_quotas": blind_all_sum,
        "blind_test_deduplicated": blind_dedup_sum,
        "dev_set_deduplicated": dev_dedup_sum,
    }
    print("\n[E7 Zero-Leakage Cost Model Summary]")
    print("  All Quotas   :", all_sum)
    print("  Deduplicated :", dedup_sum)
    print("  Blind Test   :", blind_dedup_sum)
    out_path = os.path.expanduser("~/maxbound/results/e7_cost_model.json")
    write_result_json(out_path, eval_records, {"experiment": "E7_zero_leakage_cost_model_11datasets", "summary": summary})


# ==============================================================================
# E8, E10 (with Crossover Boundary & Fused 1024D Heatmap), and E11 (11-Dataset Portability)
# ==============================================================================
def run_e8_e10_e11_v5(device=1):
    # E8 across all 11 datasets with build-stage breakdown + competitor index footprint!
    e8_records = []
    for ds in ALL_11_DATASETS:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            continue
        torch.cuda.empty_cache()
        d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
        d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
        q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
        q_offs_all = np.load(os.path.join(ddir, "query_offsets.int64.npy"))
        B_eff = min(64, len(q_offs_all) - 1)
        q_offs = q_offs_all[:B_eff + 1]; q_sub = q_toks[:int(q_offs[-1])]
        D = d_toks.shape[1]

        fp16_gb = d_toks.numel() * 2 / (1024 ** 3)
        b2 = B2CuBLASUnpaddedExact(d_toks, d_offs, device=device)
        t_b2 = time_cuda_fn(lambda: b2.search(q_sub, q_offs, 10), 2, 7 if ds == "msmarco_2m" else 9, device)[0]["median_ms"] / float(B_eff)
        del b2; torch.cuda.empty_cache()

        b8 = B8FusedFP16TCExact(d_toks, d_offs, device=device)
        t_b8 = time_cuda_fn(lambda: b8.search(q_sub, q_offs, 10), 2, 7 if ds == "msmarco_2m" else 9, device)[0]["median_ms"] / float(B_eff)
        del b8; torch.cuda.empty_cache()

        modes = ["int8", "int6", "int4"] if D == 128 else ["int8"]
        for mode in modes:
            eng = MaxBoundIndex(d_toks, d_offs, device=device, mode=mode)
            build_s = eng.build_breakdown_s["total_build_s"]
            if mode == "int8":
                idx_bytes = eng.d_qdata.numel() + (eng.d_scales.numel() + eng.d_rho.numel()) * 4 + eng.d_means.numel() * 2
            elif mode == "int6":
                idx_bytes = eng.d_hi4.numel() + eng.d_lo2.numel() + (eng.d_scales.numel() + eng.d_rho.numel()) * 4 + eng.d_means.numel() * 2
            else:
                idx_bytes = eng.d_qdata.numel() + (eng.d_scales.numel() + eng.d_rho.numel()) * 4 + eng.d_means.numel() * 2
            idx_gb = idx_bytes / (1024 ** 3)
            dual_res_gb = fp16_gb + idx_gb
            t_mb = time_cuda_fn(lambda: eng.search(q_sub, q_offs, 10, 32), 2, 7 if ds == "msmarco_2m" else 9, device)[0]["median_ms"] / float(B_eff)
            t_best_ex = min(t_b2, t_b8)
            gain_per_q_s = max((t_b2 - t_mb) / 1000.0, 1e-9)
            gain_best_s = max((t_best_ex - t_mb) / 1000.0, 1e-9)
            breakeven_q = int(np.ceil(build_s / gain_per_q_s)) if t_b2 > t_mb else -1
            breakeven_vs_best_q = int(np.ceil(build_s / gain_best_s)) if t_best_ex > t_mb else -1
            e8_records.append({
                "dataset": ds, "dim": D, "mode": mode, "N_docs": len(d_offs) - 1, "N_toks": d_toks.shape[0],
                "fp16_gb": round(fp16_gb, 4),
                "index_gb": round(idx_gb, 4),
                "offload_vram_gb": round(idx_gb, 4),
                "dual_resident_vram_gb": round(dual_res_gb, 4),
                "offload_vram_reduction_ratio": round(fp16_gb / max(idx_gb, 1e-6), 3),
                "dual_resident_overhead_ratio": round(dual_res_gb / max(fp16_gb, 1e-6), 3),
                "compression_ratio": round(fp16_gb / max(idx_gb, 1e-6), 3),
                "build_s": round(build_s, 4),
                "build_breakdown_s": eng.build_breakdown_s,
                "b2_per_q_ms": round(t_b2, 4), "b8_per_q_ms": round(t_b8, 4), "mb_per_q_ms": round(t_mb, 4),
                "breakeven_queries": breakeven_q,
                "breakeven_vs_best_exact_queries": breakeven_vs_best_q,
            })
            print(f"[E8 {ds:13s} {mode}] build={build_s:.3f}s | dualVRAM={dual_res_gb:.2f}GB ({dual_res_gb/fp16_gb:.2f}x FP16) | offloadVRAM={idx_gb:.2f}GB ({fp16_gb/idx_gb:.2f}x comp) | BE_B2={breakeven_q} | BE_Best={breakeven_vs_best_q}")
            del eng; torch.cuda.empty_cache()

    write_result_json(os.path.expanduser("~/maxbound/results/e8_build_amortization.json"), e8_records, {"experiment": "E8_build_stage_breakdown_and_amortization_11ds", "device": device})

    # E10: Sensitivity + (N_docs, B) Crossover Boundary + Fused 1024D 4x4 Heatmap!
    e10_records = []
    ddir = os.path.expanduser("~/maxbound/data_mv/fiqa")
    d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
    d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
    q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
    q_offs_all = np.load(os.path.join(ddir, "query_offsets.int64.npy"))
    mb8 = MaxBoundIndex(d_toks, d_offs, device=device, mode="int8")
    b2 = B2CuBLASUnpaddedExact(d_toks, d_offs, device=device)
    b8 = B8FusedFP16TCExact(d_toks, d_offs, device=device)
    q_offs = q_offs_all[:65]; q_sub = q_toks[:int(q_offs[-1])]
    o_i, o_s = exact_maxsim_oracle(q_sub.float(), q_offs, d_toks, d_offs, topk=10, device=device, use_fp64=True)

    for cw in [0, 8, 16, 32, 64, 128]:
        if cw == 0:
            t, (pi, ps, inf) = time_cuda_fn(lambda: mb8.search(q_sub, q_offs, topk=10, disable_prewarm=True), 2, 9, device)
        else:
            t, (pi, ps, inf) = time_cuda_fn(lambda c=cw: mb8.search(q_sub, q_offs, topk=10, c_warm=c), 2, 9, device)
        aud = audit_exact_topk(pi, ps, o_i, o_s, tol=2e-3)
        assert aud["agreement"] == 1.0
        e10_records.append({"sweep": "c_warm", "dataset": "fiqa", "val": cw, "ms": t["median_ms"], "ci95_ms": t["ci95_ms"], "qps": 64000.0 / t["median_ms"], "surv_pct": round(100 * inf["survivor_ratio"], 3), "audit": aud, "exact": True})

    for B in [1, 2, 4, 8, 16, 32, 64]:
        qo = q_offs_all[:B + 1]; qt = q_toks[:int(qo[-1])]
        t2, _ = time_cuda_fn(lambda: b2.search(qt, qo, 10), 3, 15, device)
        t_f8, _ = time_cuda_fn(lambda: b8.search(qt, qo, 10), 3, 15, device)
        t_mb, (_, _, i8) = time_cuda_fn(lambda: mb8.search(qt, qo, 10, 32), 3, 15, device)
        e10_records.append({
            "sweep": "batch_B", "dataset": "fiqa", "val": B,
            "b2_ms": t2["median_ms"], "b8_fused_fp16_ms": t_f8["median_ms"], "mb8_ms": t_mb["median_ms"],
            "mb8_ci95_ms": t_mb["ci95_ms"],
            "speedup": t2["median_ms"] / t_mb["median_ms"],
            "speedup_vs_b8": t_f8["median_ms"] / t_mb["median_ms"],
            "qps": B * 1000.0 / t_mb["median_ms"],
        })

    for tk in [1, 10, 50, 100]:
        ok_i, ok_s = exact_maxsim_oracle(q_sub.float(), q_offs, d_toks, d_offs, topk=tk, device=device, use_fp64=True)
        t8, (pi, ps, i8) = time_cuda_fn(lambda k=tk: mb8.search(q_sub, q_offs, topk=k, c_warm=max(32, k)), 2, 9, device)
        aud = audit_exact_topk(pi, ps, ok_i, ok_s, tol=2e-3)
        assert aud["agreement"] == 1.0
        e10_records.append({"sweep": "topk", "dataset": "fiqa", "val": tk, "ms": t8["median_ms"], "qps": 64000.0 / t8["median_ms"], "surv_pct": round(100 * i8["survivor_ratio"], 3), "audit": aud, "exact": True})

    del b2, b8, mb8; torch.cuda.empty_cache()

    # Honest (N_docs x B) Crossover Boundary Grid on FiQA sub-corpora (N_docs in {2K, 5K, 10K, 25K, 57.6K}, B in {1, 4, 16, 64})
    for n_sub in [2000, 5000, 10000, 25000, len(d_offs) - 1]:
        soffs = d_offs[:n_sub + 1]
        stoks = d_toks[:int(soffs[-1])]
        b2_s = B2CuBLASUnpaddedExact(stoks, soffs, device=device)
        b8_s = B8FusedFP16TCExact(stoks, soffs, device=device)
        mb_s = MaxBoundIndex(stoks, soffs, device=device, mode="int8")
        auto_s = MaxBoundAutoEngine(mb_s)
        for B_c in [1, 4, 16, 64]:
            qo_c = q_offs_all[:B_c + 1]; qt_c = q_toks[:int(qo_c[-1])]
            t2_c, _ = time_cuda_fn(lambda: b2_s.search(qt_c, qo_c, 10), 2, 11, device)
            tb8_c, _ = time_cuda_fn(lambda: b8_s.search(qt_c, qo_c, 10), 2, 11, device)
            tmb_c, _ = time_cuda_fn(lambda: mb_s.search(qt_c, qo_c, 10, 32), 2, 11, device)
            tau_c, (_, _, inf_au) = time_cuda_fn(lambda: auto_s.search(qt_c, qo_c, 10, 32), 2, 11, device)
            best_ex = min(t2_c["median_ms"], tb8_c["median_ms"])
            e10_records.append({
                "sweep": "crossover_N_x_B", "dataset": "fiqa_sub", "N_docs": int(n_sub), "B": B_c,
                "b2_ms": round(t2_c["median_ms"], 4),
                "b8_fused_ms": round(tb8_c["median_ms"], 4),
                "mb8_ms": round(tmb_c["median_ms"], 4),
                "mb_auto_ms": round(tau_c["median_ms"], 4),
                "mb_auto_plan": inf_au.get("selected_plan"),
                "speedup_vs_b2": round(t2_c["median_ms"] / tmb_c["median_ms"], 3),
                "speedup_vs_b8": round(tb8_c["median_ms"] / tmb_c["median_ms"], 3),
                "speedup_auto_vs_best_exact": round(best_ex / tau_c["median_ms"], 3),
            })
        del b2_s, b8_s, mb_s, auto_s; torch.cuda.empty_cache()

    for mode in ["int4", "int6", "int8"]:
        eng = MaxBoundIndex(d_toks, d_offs, device=device, mode=mode)
        t, (pi, ps, inf) = time_cuda_fn(lambda: eng.search(q_sub, q_offs, 10, 32), 2, 9, device)
        aud = audit_exact_topk(pi, ps, o_i, o_s, tol=2e-3)
        assert aud["agreement"] == 1.0
        e10_records.append({"sweep": "quant_bits", "dataset": "fiqa", "val": mode, "ms": t["median_ms"], "qps": 64000.0 / t["median_ms"], "surv_pct": round(100 * inf["survivor_ratio"], 3), "audit": aud, "exact": True})
        del eng; torch.cuda.empty_cache()

    for bds in ["scifact_bgem3", "fiqa_bgem3"]:
        bdir = os.path.expanduser(f"~/maxbound/data_mv/{bds}")
        if not os.path.exists(os.path.join(bdir, "doc_tokens.fp16.npy")):
            continue
        bd = torch.from_numpy(np.load(os.path.join(bdir, "doc_tokens.fp16.npy")))
        bdo = np.load(os.path.join(bdir, "doc_offsets.int64.npy"))
        bq = torch.from_numpy(np.load(os.path.join(bdir, "query_tokens.fp16.npy")))
        bqo = np.load(os.path.join(bdir, "query_offsets.int64.npy"))[:65]
        bqs = bq[:int(bqo[-1])]
        boi, bos = exact_maxsim_oracle(bqs.float(), bqo, bd, bdo, topk=10, device=device, use_fp64=True)
        b2_m3 = B2CuBLASUnpaddedExact(bd, bdo, device=device)
        mb_m3 = MaxBoundIndex(bd, bdo, device=device, mode="int8")
        t_b2m, _ = time_cuda_fn(lambda: b2_m3.search(bqs, bqo, 10), 3, 11, device)
        t_mbm, (pm_i, pm_s, infm) = time_cuda_fn(lambda: mb_m3.search(bqs, bqo, 10, 32), 3, 11, device)
        aud_m = audit_exact_topk(pm_i, pm_s, boi, bos, tol=5e-3)
        assert aud_m["agreement"] == 1.0
        e10_records.append({
            "sweep": "encoder_dim_1024", "dataset": bds, "val": 1024, "B": 64, "exact": True,
            "b2_ms": t_b2m["median_ms"], "mb8_ms": t_mbm["median_ms"],
            "speedup": t_b2m["median_ms"] / t_mbm["median_ms"],
            "surv_pct": round(100 * infm["survivor_ratio"], 3), "audit": aud_m,
        })
        print(f"[E10 Fused PTX BGE-M3 {bds}] B2={t_b2m['median_ms']:.2f} ms | MB8={t_mbm['median_ms']:.2f} ms ({t_b2m['median_ms']/t_mbm['median_ms']:.2f}x)")
        del b2_m3, mb_m3; torch.cuda.empty_cache()

        if bds == "fiqa_bgem3":
            for n_sub in [8000, 16000, 32000, len(bdo) - 1]:
                sub_offs = bdo[:n_sub + 1]
                sub_toks_1024 = bd[:int(sub_offs[-1])]
                for dim_sub in [128, 256, 512, 1024]:
                    sub_d = F.normalize(sub_toks_1024[:, :dim_sub].float(), p=2, dim=1).half().contiguous()
                    sub_q = F.normalize(bqs[:, :dim_sub].float(), p=2, dim=1).half().contiguous()
                    oi_h, os_h = exact_maxsim_oracle(sub_q.float(), bqo, sub_d, sub_offs, topk=10, device=device, use_fp64=True)
                    b2_h = B2CuBLASUnpaddedExact(sub_d, sub_offs, device=device)
                    mb_h = MaxBoundIndex(sub_d, sub_offs, device=device, mode="int8")
                    t2_h, _ = time_cuda_fn(lambda: b2_h.search(sub_q, bqo, 10), 3, 9, device)
                    tm_h, (pi_h, ps_h, inf_h) = time_cuda_fn(lambda: mb_h.search(sub_q, bqo, 10, 32), 3, 9, device)
                    aud_h = audit_exact_topk(pi_h, ps_h, oi_h, os_h, tol=5e-3)
                    assert aud_h["agreement"] == 1.0
                    sp_h = t2_h["median_ms"] / tm_h["median_ms"]
                    e10_records.append({
                        "sweep": "heatmap_N_x_D", "dataset": "fiqa_bgem3_sub",
                        "N_docs": int(n_sub), "dim": int(dim_sub), "B": 64, "exact": True,
                        "b2_ms": round(t2_h["median_ms"], 3), "mb8_ms": round(tm_h["median_ms"], 3),
                        "speedup": round(sp_h, 3), "surv_pct": round(inf_h["survivor_ratio"] * 100.0, 3),
                        "audit": aud_h,
                    })
                    print(f"  [Heatmap N={n_sub:5d} D={dim_sub:4d}] B2={t2_h['median_ms']:6.2f}ms | MB8={tm_h['median_ms']:6.2f}ms -> {sp_h:.2f}x")
                    del b2_h, mb_h; torch.cuda.empty_cache()

    write_result_json(os.path.expanduser("~/maxbound/results/e10_sensitivity.json"), e10_records, {"experiment": "E10_sensitivity_crossover_and_fused_2d_heatmap", "device": device})
    run_e11_v5(device=device)


def run_e11_v5(device=1):
    import gc
    # E11: Cross-Architecture Portability across ALL 11 real datasets (A100 sm_80 vs RTX 4080 sm_89)
    e11_records = []
    for ds in ALL_11_DATASETS:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            continue
        d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
        d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
        q_toks = torch.from_numpy(np.load(os.path.join(ddir, "query_tokens.fp16.npy")))
        q_offs_all = np.load(os.path.join(ddir, "query_offsets.int64.npy"))
        B_eff = min(16 if ds == "msmarco_2m" else 64, len(q_offs_all) - 1)
        q_offs = q_offs_all[:B_eff + 1]; q_sub = q_toks[:int(q_offs[-1])]
        D = d_toks.shape[1]
        tol = 5e-3 if D == 1024 else 2e-3
        o_i, o_s = exact_maxsim_oracle(q_sub.float(), q_offs, d_toks, d_offs, topk=10, device=1, use_fp64=True)

        row = {"dataset": ds, "dim": D, "batch": B_eff}
        for dev, gname in [(1, "A100_sm80"), (0, "RTX4080_sm89")]:
            gc.collect()
            torch.cuda.empty_cache()
            phys_gb = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
            fp16_gb = d_toks.numel() * 2 / (1024 ** 3)
            offload_mb = bool((fp16_gb * 1.65) > (phys_gb * 0.88))
            mode_p = "int4" if (dev == 0 and fp16_gb > 24.0) else "int8"
            b2 = None
            mb8 = None
            b2_ms = None
            b2_oom = False
            if fp16_gb > phys_gb * 0.96:
                b2_oom = True
                offload_mb = True
            else:
                try:
                    b2 = B2CuBLASUnpaddedExact(d_toks, d_offs, device=dev)
                    t2, _ = time_cuda_fn(lambda: b2.search(q_sub, q_offs, 10), 2, 9 if ds == "msmarco_2m" else 11, dev)
                    b2_ms = t2["median_ms"]
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    b2_oom = True
                    offload_mb = True
                finally:
                    b2 = None
                    gc.collect()
                    torch.cuda.empty_cache()

            try:
                mb8 = MaxBoundIndex(d_toks, d_offs, device=dev, mode=mode_p, host_offload_fp16=offload_mb)
                _ = mb8.search(q_sub, q_offs, 10, 32)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                mb8 = None
                gc.collect()
                torch.cuda.empty_cache()
                offload_mb = True
                mb8 = MaxBoundIndex(d_toks, d_offs, device=dev, mode=mode_p, host_offload_fp16=True)

            t8, (pi, ps, inf) = time_cuda_fn(lambda: mb8.search(q_sub, q_offs, 10, 32), 2, 9 if ds == "msmarco_2m" else 11, dev)
            _, _, inf_m = mb8.search(q_sub, q_offs, 10, 32, return_tested_mask=True)
            aud = audit_exact_topk(pi, ps, o_i, o_s, tol=tol, tested_mask=inf_m["tested_mask_orig"])
            assert aud["agreement"] == 1.0 and aud["false_negative_prunes"] == 0
            row[gname] = {
                "b2_ms": b2_ms, "b2_oom": b2_oom, "mb8_ms": t8["median_ms"], "mb8_ci95_ms": t8["ci95_ms"],
                "mode_used": mode_p,
                "offload_used": offload_mb,
                "speedup": (b2_ms / t8["median_ms"]) if b2_ms else None,
                "qps": B_eff * 1000.0 / t8["median_ms"],
                "surv_pct": round(100 * inf["survivor_ratio"], 3),
                "audit": aud,
            }
            mb8 = None
            gc.collect()
            torch.cuda.empty_cache()
        e11_records.append(row)
        s4080 = f"OOM->{row['RTX4080_sm89']['qps']:.1f}QPS" if row["RTX4080_sm89"]["b2_oom"] else f"{row['RTX4080_sm89']['speedup']:.2f}x"
        print(f"[E11 {ds:13s}] A100={row['A100_sm80']['speedup']:.2f}x | 4080={s4080}")

    write_result_json(os.path.expanduser("~/maxbound/results/e11_portability.json"), e11_records, {"experiment": "E11_portability_11datasets_sm80_vs_sm89"})


# ==============================================================================
# E14: End-to-End RAG Query Pipeline Evaluation (GPU Query Encoding + Retrieval)
#      Evaluates ColBERTv2 (128D) on FiQA, HotpotQA, MS MARCO & BGE-M3 (1024D) on FiQA-BGEM3
# ==============================================================================
def run_e14_end_to_end_rag(device=1):
    from scripts.build_datasets import ColBERTv2Encoder
    from transformers import AutoTokenizer, AutoModel

    torch.cuda.set_device(device)
    dev_str = f"cuda:{device}"
    colbert_enc = ColBERTv2Encoder(os.path.expanduser("~/maxbound/colbertv2"), device=dev_str, query_maxlen=32)

    bge_dir = os.environ.get("BGE_M3_DIR", "BAAI/bge-m3")
    bge_tok = AutoTokenizer.from_pretrained(bge_dir, local_files_only=True)
    bge_mod = AutoModel.from_pretrained(bge_dir, local_files_only=True).to(dev_str).half().eval()

    @torch.no_grad()
    def encode_colbert_gpu(texts):
        enc = colbert_enc.tok(texts, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
        ids = enc["input_ids"].to(dev_str)
        mask = enc["attention_mask"].to(dev_str)
        ids[:, 1] = colbert_enc.Q_marker_id
        ids[ids == colbert_enc.tok.pad_token_id] = colbert_enc.tok.mask_token_id
        mask[:] = 1
        out = colbert_enc.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        proj = F.normalize(colbert_enc.linear(out), p=2, dim=-1).half().contiguous()
        B = len(texts)
        return proj.view(B * 32, 128), np.arange(0, (B + 1) * 32, 32, dtype=np.int64)

    @torch.no_grad()
    def encode_bgem3_gpu(texts):
        enc = bge_tok(texts, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
        ids = enc["input_ids"].to(dev_str)
        mask = enc["attention_mask"].to(dev_str)
        out = bge_mod(input_ids=ids, attention_mask=mask).last_hidden_state
        proj = F.normalize(out, p=2, dim=-1).half()
        proj = proj * mask.unsqueeze(-1).half()
        B = len(texts)
        return proj.contiguous().view(B * 32, 1024), np.arange(0, (B + 1) * 32, 32, dtype=np.int64)

    sample_queries = [
        f"What are the primary financial risk factors and exact retrieval latency bounds for portfolio segment {i}?"
        for i in range(64)
    ]

    e14_records = []
    for ds in ["fiqa", "hotpotqa", "msmarco", "fiqa_bgem3"]:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            continue
        torch.cuda.empty_cache()
        d_toks = torch.from_numpy(np.load(os.path.join(ddir, "doc_tokens.fp16.npy")))
        d_offs = np.load(os.path.join(ddir, "doc_offsets.int64.npy"))
        D = d_toks.shape[1]
        N_docs = len(d_offs) - 1

        b2 = B2CuBLASUnpaddedExact(d_toks, d_offs, device=device)
        b8 = B8FusedFP16TCExact(d_toks, d_offs, device=device)
        mb8 = MaxBoundIndex(d_toks, d_offs, device=device, mode="int8")

        enc_fn = encode_colbert_gpu if D == 128 else encode_bgem3_gpu
        enc_name = "ColBERTv2.0 (FP16 GPU)" if D == 128 else "BAAI/BGE-M3 (FP16 GPU)"

        for B in [1, 16, 64]:
            q_txts = sample_queries[:B]
            t_enc, (qt_gpu, q_offs) = time_cuda_fn(lambda: enc_fn(q_txts), warmup=3, repeats=15, device=device)
            enc_ms = t_enc["median_ms"]

            t_b2_ret, _ = time_cuda_fn(lambda: b2.search(qt_gpu, q_offs, 10), warmup=3, repeats=15, device=device)
            t_b8_ret, _ = time_cuda_fn(lambda: b8.search(qt_gpu, q_offs, 10), warmup=3, repeats=15, device=device)
            t_mb_ret, _ = time_cuda_fn(lambda: mb8.search(qt_gpu, q_offs, 10, 32), warmup=3, repeats=15, device=device)

            # Also time true fused end-to-end (Encode + Search in single timed scope)
            t_e2e_b2, _ = time_cuda_fn(lambda: b2.search(*enc_fn(q_txts), 10), warmup=2, repeats=11, device=device)
            t_e2e_b8, _ = time_cuda_fn(lambda: b8.search(*enc_fn(q_txts), 10), warmup=2, repeats=11, device=device)
            t_e2e_mb, _ = time_cuda_fn(lambda: mb8.search(*enc_fn(q_txts), 10, 32), warmup=2, repeats=11, device=device)

            rec = {
                "dataset": ds, "dim": D, "N_docs": N_docs, "encoder": enc_name, "batch": B,
                "encode_median_ms": round(enc_ms, 3),
                "encode_ci95_ms": t_enc["ci95_ms"],
                "b2_retrieval_ms": round(t_b2_ret["median_ms"], 3),
                "b8_fused_retrieval_ms": round(t_b8_ret["median_ms"], 3),
                "mb8_retrieval_ms": round(t_mb_ret["median_ms"], 3),
                "e2e_b2_ms": round(t_e2e_b2["median_ms"], 3),
                "e2e_b8_fused_ms": round(t_e2e_b8["median_ms"], 3),
                "e2e_mb8_ms": round(t_e2e_mb["median_ms"], 3),
                "e2e_mb8_ci95_ms": t_e2e_mb["ci95_ms"],
                "b2_retrieval_share_pct": round(100.0 * t_b2_ret["median_ms"] / max(enc_ms + t_b2_ret["median_ms"], 1e-6), 2),
                "e2e_speedup_vs_b2": round(t_e2e_b2["median_ms"] / t_e2e_mb["median_ms"], 3),
                "e2e_speedup_vs_b8": round(t_e2e_b8["median_ms"] / t_e2e_mb["median_ms"], 3),
                "retrieval_only_speedup_vs_b2": round(t_b2_ret["median_ms"] / t_mb_ret["median_ms"], 3),
                "e2e_qps_mb8": round(B * 1000.0 / t_e2e_mb["median_ms"], 2),
            }
            e14_records.append(rec)
            print(f"[E14 E2E {ds:11s} B={B:2d}] Enc={enc_ms:5.2f}ms | Ret(B2={t_b2_ret['median_ms']:6.2f}, MB={t_mb_ret['median_ms']:6.2f}) | E2E(B2={t_e2e_b2['median_ms']:6.2f} -> MB={t_e2e_mb['median_ms']:6.2f}ms, {rec['e2e_speedup_vs_b2']:.2f}x E2E!)")

        del b2, b8, mb8; torch.cuda.empty_cache()

    del colbert_enc, bge_mod; torch.cuda.empty_cache()
    out_path = os.path.expanduser("~/maxbound/results/e14_end_to_end_rag.json")
    write_result_json(out_path, e14_records, {"experiment": "E14_end_to_end_gpu_query_encoding_and_retrieval", "device": device})
    print(f"Saved {out_path}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    run_all = len(only) == 0
    if run_all or "e1" in only:
        run_e1_v5(device=1)
    if run_all or "e2" in only:
        run_e2_v5(device=1)
    if run_all or "e4" in only:
        run_e4_v5()
    if run_all or "e6" in only:
        run_e6_v5(device=1)
    if run_all or "e7" in only:
        run_e7_zero_leakage()
    if run_all or "e8_10_11" in only:
        run_e8_e10_e11_v5(device=1)
    if "e11" in only:
        run_e11_v5(device=1)
    if run_all or "e14" in only:
        run_e14_end_to_end_rag(device=1)
