import json
import math
import os
import subprocess
from datetime import datetime
import numpy as np
import torch


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.expanduser("~/maxbound"),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "uncommitted"


def time_cuda_fn(fn, warmup=3, repeats=15, device=0):
    """
    Rigorous CUDA Event timer reporting median, mean, sample std, 95% confidence interval (ci95_ms),
    p95, p99, and min across `repeats` timed executions after `warmup` runs.
    """
    torch.cuda.set_device(device)
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize(device)
    times_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize(device)
        times_ms.append(float(start.elapsed_time(end)))
    arr = np.array(times_ms, dtype=np.float64)
    n = len(arr)
    std_ms = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci95_ms = float(1.96 * std_ms / math.sqrt(n)) if n > 1 else 0.0
    return {
        "runs_ms": [round(x, 4) for x in times_ms],
        "num_runs": int(n),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": round(std_ms, 5),
        "ci95_ms": round(ci95_ms, 5),
        "min_ms": float(np.min(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }, out


def audit_exact_topk(pred_ids, pred_scores, oracle_ids, oracle_scores, tol=1e-3, tested_mask=None):
    """
    Audits exact Top-K retrieval against IEEE-754 FP64/FP32 oracle AND optionally verifies
    zero false-negative prunes (`false_negative_prunes == 0`) when `tested_mask` [B, N_docs] is provided!
    """
    B, K = oracle_ids.shape
    pred_ids = np.asarray(pred_ids, dtype=np.int64)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)
    oracle_ids = np.asarray(oracle_ids, dtype=np.int64)
    oracle_scores = np.asarray(oracle_scores, dtype=np.float32)

    exact_match = 0
    false_negative_prunes = 0
    max_score_diff = 0.0
    mismatches = []
    for q in range(B):
        o_list = [int(x) for x in oracle_ids[q] if x >= 0]
        o_set = set(o_list)
        p_set = set(int(x) for x in pred_ids[q] if x >= 0)
        if tested_mask is not None:
            for od in o_list:
                if not bool(tested_mask[q, od]):
                    false_negative_prunes += 1
        diff = float(np.max(np.abs(np.sort(pred_scores[q]) - np.sort(oracle_scores[q]))))
        if diff > max_score_diff:
            max_score_diff = diff
        if o_set == p_set:
            exact_match += 1
        else:
            kth_o = float(oracle_scores[q, -1])
            kth_p = float(np.min(pred_scores[q]))
            if abs(kth_o - kth_p) <= tol and diff <= tol * 10:
                exact_match += 1
            else:
                mismatches.append({
                    "q": q,
                    "oracle_ids": oracle_ids[q].tolist(),
                    "pred_ids": pred_ids[q].tolist(),
                    "oracle_scores": [round(float(x), 5) for x in oracle_scores[q]],
                    "pred_scores": [round(float(x), 5) for x in pred_scores[q]],
                })
    res = {
        "num_queries": int(B),
        "k": int(K),
        "agreement": float(exact_match / max(B, 1)),
        "num_mismatch": int(len(mismatches)),
        "max_score_diff": float(max_score_diff),
        "sample_mismatches": mismatches[:5],
    }
    if tested_mask is not None:
        res["false_negative_prunes"] = int(false_negative_prunes)
        res["false_negative_rate"] = float(false_negative_prunes / max(B * K, 1))
    return res


def compute_ir_metrics(pred_ids, qrels, k=10):
    if not qrels:
        return {"ndcg_at_10": None, "mrr_at_10": None, "recall_at_10": None}
    ndcgs, mrrs, recalls = [], [], []
    for q_idx, rel_dict in qrels.items():
        qi = int(q_idx)
        if qi >= len(pred_ids) or not rel_dict:
            continue
        ranked = [int(d) for d in pred_ids[qi][:k] if d >= 0]
        rr = 0.0
        for rank, did in enumerate(ranked, start=1):
            if rel_dict.get(str(did), rel_dict.get(did, 0)) > 0:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
        dcg = 0.0
        for rank, did in enumerate(ranked, start=1):
            rel = float(rel_dict.get(str(did), rel_dict.get(did, 0)))
            if rel > 0:
                dcg += (2.0 ** rel - 1.0) / np.log2(rank + 1.0)
        ideal_rels = sorted([float(v) for v in rel_dict.values() if v > 0], reverse=True)[:k]
        idcg = sum((2.0 ** r - 1.0) / np.log2(idx + 2.0) for idx, r in enumerate(ideal_rels))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        num_rel = sum(1 for v in rel_dict.values() if v > 0)
        hits = sum(1 for did in ranked if rel_dict.get(str(did), rel_dict.get(did, 0)) > 0)
        recalls.append(hits / max(min(num_rel, k), 1))
    return {
        "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else None,
        "mrr_at_10": float(np.mean(mrrs)) if mrrs else None,
        "recall_at_10": float(np.mean(recalls)) if recalls else None,
        "num_eval_queries": len(ndcgs),
    }


def write_result_json(path, records, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "meta": {
            "git_commit": get_git_commit(),
            "seed": 2027,
            "timestamp": datetime.now().isoformat(),
            **(meta or {}),
        },
        "records": records,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
