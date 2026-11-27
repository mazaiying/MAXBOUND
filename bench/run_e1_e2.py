import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/maxbound"))
from bench.common import time_cuda_fn, audit_exact_topk, compute_ir_metrics, write_result_json
from src.oracle import exact_maxsim_oracle
from src.maxbound_engine import MaxBoundIndex
from src.baselines.baselines import (
    B1TorchPaddedGemm,
    B2CuBLASUnpaddedExact,
    B3QuantizedDecompressGemm,
    B4PlaidGPU,
    B5MuveraGPU,
    B6CpuExactMaxSim,
)


def time_cpu_fn(fn, warmup=1, repeats=3):
    out = None
    for _ in range(warmup):
        out = fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(ts, dtype=np.float64)
    return {
        "runs_ms": [round(float(x), 4) for x in ts],
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "p99_ms": float(np.percentile(arr, 99)),
    }, out


def eval_ir_batched(search_fn, q_toks, q_offs, qrels, num_eval_q, batch_size=64, topk=10):
    if not qrels:
        return {"ndcg_at_10": None, "mrr_at_10": None, "recall_at_10": None, "num_eval_queries": 0}
    n_q = min(num_eval_q, len(q_offs) - 1)
    all_preds = np.full((n_q, topk), -1, dtype=np.int64)
    for s in range(0, n_q, batch_size):
        e = min(s + batch_size, n_q)
        sub_offs = q_offs[s:e + 1] - q_offs[s]
        sub_toks = q_toks[int(q_offs[s]):int(q_offs[e])]
        res = search_fn(sub_toks, sub_offs)
        all_preds[s:e] = res[0]
    sub_qrels = {str(k): v for k, v in qrels.items() if int(k) < n_q}
    return compute_ir_metrics(all_preds, sub_qrels, k=topk)


def run_e1(device=1):
    datasets = ["scifact", "nfcorpus", "trec_covid", "arguana", "longbench", "scidocs", "fiqa", "hotpotqa"]
    records = []

    for ds in datasets:
        ddir = os.path.expanduser(f"~/maxbound/data_mv/{ds}")
        if not os.path.exists(os.path.join(ddir, "doc_tokens.fp16.npy")):
            print(f"Skipping {ds}")
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
        num_eval_q = min(256, len(q_offs_all) - 1)
        print(f"\n================ E1 [{ds}] N_docs={N_docs} N_toks={N_toks} ================")

        # Build engines and record build time + memory
        def build_timed(fn):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            m0 = torch.cuda.memory_allocated(device)
            t0 = time.perf_counter()
            obj = fn()
            torch.cuda.synchronize(device)
            bt = time.perf_counter() - t0
            m1 = torch.cuda.memory_allocated(device)
            return obj, bt, (m1 - m0) / (1024 ** 3)

        b2, b2_bt, b2_gb = build_timed(lambda: B2CuBLASUnpaddedExact(d_toks, d_offs, device=device))
        mb8, mb8_bt, mb8_gb = build_timed(lambda: MaxBoundIndex(d_toks, d_offs, device=device, mode="int8"))
        mb6, mb6_bt, mb6_gb = build_timed(lambda: MaxBoundIndex(d_toks, d_offs, device=device, mode="int6"))
        b3, b3_bt, b3_gb = build_timed(lambda: B3QuantizedDecompressGemm(d_toks, d_offs, device=device))
        b4, b4_bt, b4_gb = build_timed(lambda: B4PlaidGPU(d_toks, d_offs, device=device, n_centroids=min(1024, max(64, N_docs // 4))))
        b5, b5_bt, b5_gb = build_timed(lambda: B5MuveraGPU(d_toks, d_offs, device=device, r_reps=8, k_sim=4, d_proj=16))

        # Test B1 (may OOM on large/long datasets)
        b1, b1_bt, b1_gb, b1_oom = None, 0.0, 0.0, False
        try:
            b1, b1_bt, b1_gb = build_timed(lambda: B1TorchPaddedGemm(d_toks, d_offs, device=device))
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            b1_oom = True
            torch.cuda.empty_cache()

        # IR quality on up to 256 official queries (evaluated once per method)
        ir_exact = eval_ir_batched(lambda qt, qo: b2.search(qt, qo, topk=10), q_toks, q_offs_all, qrels, num_eval_q)
        ir_mb8 = eval_ir_batched(lambda qt, qo: mb8.search(qt, qo, topk=10, c_warm=32)[:2], q_toks, q_offs_all, qrels, num_eval_q)
        ir_b3 = eval_ir_batched(lambda qt, qo: b3.search(qt, qo, topk=10), q_toks, q_offs_all, qrels, num_eval_q)
        ir_b4 = eval_ir_batched(lambda qt, qo: b4.search(qt, qo, nprobe=4, ncands=512, topk=10), q_toks, q_offs_all, qrels, num_eval_q)
        ir_b5 = eval_ir_batched(lambda qt, qo: b5.search(qt, qo, ncands=512, topk=10), q_toks, q_offs_all, qrels, num_eval_q)

        batches = [64, 16, 1] if ds in ("scifact", "fiqa", "longbench", "hotpotqa") else [64]
        for B in batches:
            B_eff = min(B, len(q_offs_all) - 1)
            q_offs = q_offs_all[:B_eff + 1]
            q_sub = q_toks[:int(q_offs[-1])]
            oracle_ids, oracle_scores = exact_maxsim_oracle(
                q_sub.float(), q_offs, d_toks, d_offs, topk=10, doc_batch_size=2048, device=device
            )

            # B2 baseline timing
            torch.cuda.reset_peak_memory_stats(device)
            t_b2, (i_b2, s_b2) = time_cuda_fn(lambda: b2.search(q_sub, q_offs, topk=10), warmup=3, repeats=5, device=device)
            aud_b2 = audit_exact_topk(i_b2, s_b2, oracle_ids, oracle_scores)
            assert aud_b2["agreement"] == 1.0
            b2_ms = t_b2["median_ms"]
            records.append({
                "dataset": ds, "batch": B_eff, "mode": "full_corpus", "method": "B2_cuBLAS_Unpadded_Exact",
                "exact": True, **t_b2, "qps": B_eff * 1000.0 / b2_ms, "speedup_vs_b2": 1.0,
                "prune_pct": 0.0, "survivor_pct": 100.0, "build_s": round(b2_bt, 3), "index_gb": round(b2_gb, 4),
                "vram_peak_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 4),
                "ir": ir_exact, "audit": aud_b2, "N_docs": N_docs, "N_toks": N_toks,
            })

            if B == 64:
                # B1
                if not b1_oom:
                    try:
                        torch.cuda.reset_peak_memory_stats(device)
                        t_b1, (i_b1, s_b1) = time_cuda_fn(lambda: b1.search(q_sub, q_offs, topk=10), warmup=2, repeats=5, device=device)
                        aud_b1 = audit_exact_topk(i_b1, s_b1, oracle_ids, oracle_scores)
                        assert aud_b1["agreement"] == 1.0
                        records.append({
                            "dataset": ds, "batch": B_eff, "mode": "full_corpus", "method": "B1_Torch_Padded_GEMM",
                            "exact": True, **t_b1, "qps": B_eff * 1000.0 / t_b1["median_ms"],
                            "speedup_vs_b2": b2_ms / t_b1["median_ms"], "prune_pct": 0.0, "survivor_pct": 100.0,
                            "build_s": round(b1_bt, 3), "index_gb": round(b1_gb, 4),
                            "vram_peak_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 4),
                            "ir": ir_exact, "audit": aud_b1, "N_docs": N_docs, "N_toks": N_toks,
                        })
                    except (torch.cuda.OutOfMemoryError, RuntimeError):
                        b1_oom = True
                        torch.cuda.empty_cache()
                if b1_oom:
                    records.append({
                        "dataset": ds, "batch": B_eff, "mode": "full_corpus", "method": "B1_Torch_Padded_GEMM",
                        "exact": True, "status": "OOM", "qps": 0.0, "median_ms": None, "N_docs": N_docs, "N_toks": N_toks,
                    })

                # B3 Quantized Decompress
                torch.cuda.reset_peak_memory_stats(device)
                t_b3, (i_b3, s_b3) = time_cuda_fn(lambda: b3.search(q_sub, q_offs, topk=10), warmup=2, repeats=5, device=device)
                aud_b3 = audit_exact_topk(i_b3, s_b3, oracle_ids, oracle_scores)
                records.append({
                    "dataset": ds, "batch": B_eff, "mode": "full_corpus", "method": "B3_Quant_Decompress",
                    "exact": False, **t_b3, "qps": B_eff * 1000.0 / t_b3["median_ms"],
                    "speedup_vs_b2": b2_ms / t_b3["median_ms"], "prune_pct": 0.0, "survivor_pct": 100.0,
                    "build_s": round(b3_bt, 3), "index_gb": round(b3_gb, 4),
                    "vram_peak_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 4),
                    "ir": ir_b3, "audit": aud_b3, "N_docs": N_docs, "N_toks": N_toks,
                })

                # B6 CPU 48-thread Exact (scale B_cpu by corpus size so each corpus completes cleanly)
                b_cpu = min(B_eff, 4 if N_toks > 15_000_000 else (8 if N_toks > 5_000_000 else 16))
                q_cpu_offs = q_offs_all[:b_cpu + 1]
                q_cpu_sub = q_toks[:int(q_cpu_offs[-1])]
                o_cpu_ids, o_cpu_sc = exact_maxsim_oracle(
                    q_cpu_sub.float(), q_cpu_offs, d_toks, d_offs, topk=10, doc_batch_size=2048, device=device
                )
                b6 = B6CpuExactMaxSim(d_toks, d_offs)
                t_b6, (i_b6, s_b6) = time_cpu_fn(lambda: b6.search(q_cpu_sub, q_cpu_offs, topk=10), warmup=1, repeats=2)
                aud_b6 = audit_exact_topk(i_b6, s_b6, o_cpu_ids, o_cpu_sc)
                assert aud_b6["agreement"] == 1.0
                records.append({
                    "dataset": ds, "batch": b_cpu, "mode": "full_corpus", "method": "B6_CPU_48T_Exact",
                    "exact": True, **t_b6, "qps": b_cpu * 1000.0 / t_b6["median_ms"],
                    "speedup_vs_b2": (b_cpu * 1000.0 / t_b6["median_ms"]) / (B_eff * 1000.0 / b2_ms),
                    "prune_pct": 0.0, "survivor_pct": 100.0, "build_s": 0.0, "index_gb": 0.0,
                    "ir": ir_exact, "audit": aud_b6, "N_docs": N_docs, "N_toks": N_toks,
                })
                del b6

            # MAXBOUND-INT8 & MAXBOUND-INT6
            for mname, eng, bt, gb in [("MAXBOUND_INT8", mb8, mb8_bt, mb8_gb), ("MAXBOUND_INT6", mb6, mb6_bt, mb6_gb)]:
                torch.cuda.reset_peak_memory_stats(device)
                t_mb, (i_mb, s_mb, info) = time_cuda_fn(
                    lambda e=eng: e.search(q_sub, q_offs, topk=10, c_warm=32), warmup=3, repeats=5, device=device
                )
                aud_mb = audit_exact_topk(i_mb, s_mb, oracle_ids, oracle_scores, tol=2e-3)
                assert aud_mb["agreement"] == 1.0 and aud_mb["num_mismatch"] == 0, f"{ds} {mname} B={B_eff} FAILED: {aud_mb}"
                sp = b2_ms / t_mb["median_ms"]
                records.append({
                    "dataset": ds, "batch": B_eff, "mode": "full_corpus", "method": mname,
                    "exact": True, **t_mb, "qps": B_eff * 1000.0 / t_mb["median_ms"], "speedup_vs_b2": sp,
                    "prune_pct": round(100.0 * (1.0 - info["survivor_ratio"]), 4),
                    "survivor_pct": round(100.0 * info["survivor_ratio"], 4),
                    "build_s": round(bt, 3), "index_gb": round(gb, 4),
                    "vram_peak_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 4),
                    "ir": ir_mb8, "audit": aud_mb, "N_docs": N_docs, "N_toks": N_toks,
                })
                print(f"  B={B_eff:2d} {mname:14s}: {t_mb['median_ms']:6.2f} ms ({sp:.2f}x vs B2={b2_ms:.2f} ms) | surv={info['survivor_ratio']*100:.2f}% | audit=1.0")

            # B4 PLAID, B5 MUVERA, and Hybrids + Mode B Rerank at B=64
            if B == 64:
                t_b4, (i_b4, s_b4) = time_cuda_fn(lambda: b4.search(q_sub, q_offs, nprobe=4, ncands=512, topk=10), warmup=3, repeats=5, device=device)
                aud_b4 = audit_exact_topk(i_b4, s_b4, oracle_ids, oracle_scores)
                records.append({
                    "dataset": ds, "batch": B_eff, "mode": "approx_first_stage", "method": "B4_PLAID_GPU",
                    "exact": False, **t_b4, "qps": B_eff * 1000.0 / t_b4["median_ms"],
                    "speedup_vs_b2": b2_ms / t_b4["median_ms"], "ir": ir_b4, "audit": aud_b4, "N_docs": N_docs, "N_toks": N_toks,
                })

                t_b5, (i_b5, s_b5) = time_cuda_fn(lambda: b5.search(q_sub, q_offs, ncands=512, topk=10), warmup=3, repeats=5, device=device)
                aud_b5 = audit_exact_topk(i_b5, s_b5, oracle_ids, oracle_scores)
                records.append({
                    "dataset": ds, "batch": B_eff, "mode": "approx_first_stage", "method": "B5_MUVERA_GPU",
                    "exact": False, **t_b5, "qps": B_eff * 1000.0 / t_b5["median_ms"],
                    "speedup_vs_b2": b2_ms / t_b5["median_ms"], "ir": ir_b5, "audit": aud_b5, "N_docs": N_docs, "N_toks": N_toks,
                })

                # Hybrids: PLAID+MAXBOUND and MUVERA+MAXBOUND (C=2048)
                for hname, cgen in [("PLAID_plus_MAXBOUND", lambda: b4.get_candidates(q_sub, q_offs, nprobe=8, ncands=min(2048, N_docs))),
                                    ("MUVERA_plus_MAXBOUND", lambda: b5.get_candidates(q_sub, q_offs, ncands=min(2048, N_docs)))]:
                    def _hybrid(cg=cgen):
                        cids = cg()
                        return mb8.search_rerank(q_sub, q_offs, cids, topk=10, c_warm=16)
                    t_h, (ih, sh, infoh) = time_cuda_fn(_hybrid, warmup=3, repeats=5, device=device)
                    ir_h = eval_ir_batched(
                        lambda qt, qo, cg_fn=(b4.get_candidates if "PLAID" in hname else b5.get_candidates): mb8.search_rerank(
                            qt, qo, cg_fn(qt, qo, ncands=min(2048, N_docs)) if "MUVERA" in hname else cg_fn(qt, qo, nprobe=8, ncands=min(2048, N_docs)),
                            topk=10, c_warm=16
                        )[:2],
                        q_toks, q_offs_all, qrels, num_eval_q
                    )
                    records.append({
                        "dataset": ds, "batch": B_eff, "mode": "hybrid_exact_rerank", "method": hname,
                        "exact": False, **t_h, "qps": B_eff * 1000.0 / t_h["median_ms"],
                        "speedup_vs_b2": b2_ms / t_h["median_ms"], "survivor_pct": round(100.0 * infoh["survivor_ratio"], 4),
                        "ir": ir_h, "audit": audit_exact_topk(ih, sh, oracle_ids, oracle_scores),
                        "N_docs": N_docs, "N_toks": N_toks,
                    })

                # Mode B Candidate Rerank comparison (C=1000, C=4096)
                if N_docs >= 1000:
                    for C in ([1000, 4096] if N_docs >= 4096 else [1000]):
                        cands = b5.get_candidates(q_sub, q_offs, ncands=C)
                        or_id, or_sc = exact_maxsim_oracle(q_sub.float(), q_offs, d_toks, d_offs, topk=10, device=device, candidate_ids=cands.cpu().numpy())
                        t_r2, (ir2, sr2) = time_cuda_fn(lambda c=cands: b2.search_rerank(q_sub, q_offs, c, topk=10), warmup=3, repeats=5, device=device)
                        t_rm, (irm, srm, infom) = time_cuda_fn(lambda c=cands: mb8.search_rerank(q_sub, q_offs, c, topk=10, c_warm=16), warmup=3, repeats=5, device=device)
                        aud_rm = audit_exact_topk(irm, srm, or_id, or_sc, tol=2e-3)
                        assert aud_rm["agreement"] == 1.0
                        sp_r = t_r2["median_ms"] / t_rm["median_ms"]
                        print(f"  [Mode B Rerank C={C}] B2={t_r2['median_ms']:.2f} ms | MB={t_rm['median_ms']:.2f} ms ({sp_r:.2f}x, surv={infom['survivor_ratio']*100:.2f}%)")
                        records.append({
                            "dataset": ds, "batch": B_eff, "mode": f"rerank_C{C}", "method": "B2_Rerank",
                            "exact": True, **t_r2, "qps": B_eff * 1000.0 / t_r2["median_ms"], "speedup_vs_b2": 1.0,
                            "audit": audit_exact_topk(ir2, sr2, or_id, or_sc), "N_docs": N_docs, "N_toks": N_toks,
                        })
                        records.append({
                            "dataset": ds, "batch": B_eff, "mode": f"rerank_C{C}", "method": "MAXBOUND_Rerank_INT8",
                            "exact": True, **t_rm, "qps": B_eff * 1000.0 / t_rm["median_ms"], "speedup_vs_b2": sp_r,
                            "prune_pct": round(100.0 * (1.0 - infom["survivor_ratio"]), 4),
                            "survivor_pct": round(100.0 * infom["survivor_ratio"], 4),
                            "audit": aud_rm, "N_docs": N_docs, "N_toks": N_toks,
                        })

        del b1, b2, b3, b4, b5, mb8, mb6
        torch.cuda.empty_cache()

    out_path = os.path.expanduser("~/maxbound/results/e1_main_eval.json")
    write_result_json(out_path, records, {"experiment": "E1_main_eval", "device": device})
    print(f"\nSaved {out_path}")


def run_e2(device=1):
    records = []
    for ds in ["fiqa", "scidocs", "hotpotqa"]:
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
            oi, _ = exact_maxsim_oracle(st.float(), so, d_toks, d_offs, topk=10, device=device)
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

        b2 = B2CuBLASUnpaddedExact(d_toks, d_offs, device=device)
        mb8 = MaxBoundIndex(d_toks, d_offs, device=device, mode="int8")
        mb6 = MaxBoundIndex(d_toks, d_offs, device=device, mode="int6")
        b4 = B4PlaidGPU(d_toks, d_offs, device=device, n_centroids=1024)
        b5 = B5MuveraGPU(d_toks, d_offs, device=device, r_reps=8, k_sim=4, d_proj=16)

        for name, fn in [("B2_Exact", lambda st, so: b2.search(st, so, topk=10)),
                         ("MAXBOUND_INT8", lambda st, so: mb8.search(st, so, topk=10, c_warm=32)[:2]),
                         ("MAXBOUND_INT6", lambda st, so: mb6.search(st, so, topk=10, c_warm=32)[:2])]:
            t_r, _ = time_cuda_fn(lambda f=fn: f(q_sub, q_offs), warmup=2, repeats=5, device=device)
            or_rec, ir = eval_256(fn)
            records.append({
                "dataset": ds, "method": name, "config": "exact", "qps": B * 1000.0 / t_r["median_ms"],
                "median_ms": t_r["median_ms"], "oracle_topk_recall": or_rec, "ir": ir,
            })

        for np_, nc_ in [(1, 64), (2, 128), (4, 256), (4, 512), (8, 1024), (16, 2048)]:
            t_r, _ = time_cuda_fn(lambda n1=np_, n2=nc_: b4.search(q_sub, q_offs, nprobe=n1, ncands=n2, topk=10), warmup=2, repeats=5, device=device)
            or_rec, ir = eval_256(lambda st, so, n1=np_, n2=nc_: b4.search(st, so, nprobe=n1, ncands=n2, topk=10))
            records.append({
                "dataset": ds, "method": "B4_PLAID_GPU", "config": f"nprobe={np_},C={nc_}",
                "qps": B * 1000.0 / t_r["median_ms"], "median_ms": t_r["median_ms"], "oracle_topk_recall": or_rec, "ir": ir,
            })
            if nc_ >= 1024:
                t_h, _ = time_cuda_fn(lambda n1=np_, n2=nc_: mb8.search_rerank(q_sub, q_offs, b4.get_candidates(q_sub, q_offs, nprobe=n1, ncands=n2), topk=10, c_warm=16), warmup=2, repeats=5, device=device)
                records.append({
                    "dataset": ds, "method": "PLAID_plus_MAXBOUND", "config": f"nprobe={np_},C={nc_}",
                    "qps": B * 1000.0 / t_h["median_ms"], "median_ms": t_h["median_ms"], "oracle_topk_recall": or_rec, "ir": ir,
                })

        for nc_ in [64, 128, 256, 512, 1024, 2048, 4096]:
            t_r, _ = time_cuda_fn(lambda n2=nc_: b5.search(q_sub, q_offs, ncands=n2, topk=10), warmup=2, repeats=5, device=device)
            or_rec, ir = eval_256(lambda st, so, n2=nc_: b5.search(st, so, ncands=n2, topk=10))
            records.append({
                "dataset": ds, "method": "B5_MUVERA_GPU", "config": f"C={nc_}",
                "qps": B * 1000.0 / t_r["median_ms"], "median_ms": t_r["median_ms"], "oracle_topk_recall": or_rec, "ir": ir,
            })
            if nc_ >= 1024:
                t_h, _ = time_cuda_fn(lambda n2=nc_: mb8.search_rerank(q_sub, q_offs, b5.get_candidates(q_sub, q_offs, ncands=n2), topk=10, c_warm=16), warmup=2, repeats=5, device=device)
                records.append({
                    "dataset": ds, "method": "MUVERA_plus_MAXBOUND", "config": f"C={nc_}",
                    "qps": B * 1000.0 / t_h["median_ms"], "median_ms": t_h["median_ms"], "oracle_topk_recall": or_rec, "ir": ir,
                })

        del b2, mb8, mb6, b4, b5
        torch.cuda.empty_cache()

    out_path = os.path.expanduser("~/maxbound/results/e2_pareto_frontier.json")
    write_result_json(out_path, records, {"experiment": "E2_pareto_frontier", "device": device})
    print(f"Saved {out_path}")


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_e1(device=dev)
    run_e2(device=dev)
