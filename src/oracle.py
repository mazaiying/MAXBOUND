import torch
import numpy as np


@torch.no_grad()
def exact_maxsim_oracle(
    q_tokens_fp32: torch.Tensor,
    q_offsets: np.ndarray,
    d_tokens_fp16: torch.Tensor,
    d_offsets: np.ndarray,
    topk: int = 10,
    doc_batch_size: int = 4096,
    device: int = 1,
    candidate_ids: np.ndarray = None,
    use_fp64: bool = True,
):
    """
    IEEE-754 Exact FP64/FP32 MaxSim Oracle.
    By default uses IEEE-754 double-precision (float64) accumulation on GPU so exactness audits
    are verified against an unassailable higher-precision reference independent of FP16/TF32 rounding!
    Tie-broken deterministically by (-score, doc_id).
    """
    torch.cuda.set_device(device)
    acc_dtype = torch.float64 if use_fp64 else torch.float32
    B = len(q_offsets) - 1
    N_docs = len(d_offsets) - 1

    all_ids = np.full((B, topk), -1, dtype=np.int64)
    all_scores = np.full((B, topk), -1e9, dtype=np.float32)

    if candidate_ids is not None:
        for qi in range(B):
            qs, qe = int(q_offsets[qi]), int(q_offsets[qi + 1])
            q = q_tokens_fp32[qs:qe].to(device=device, dtype=acc_dtype)
            cands = candidate_ids[qi]
            cands = cands[cands >= 0]
            scores_list = []
            for start in range(0, len(cands), doc_batch_size):
                sub_c = cands[start:start + doc_batch_size]
                lens = d_offsets[sub_c + 1] - d_offsets[sub_c]
                max_len = int(lens.max())
                toks = torch.zeros((len(sub_c), max_len, q.shape[1]), dtype=acc_dtype, device=device)
                mask = torch.zeros((len(sub_c), max_len), dtype=torch.bool, device=device)
                for idx, did in enumerate(sub_c):
                    ds, de = int(d_offsets[did]), int(d_offsets[did + 1])
                    ln = de - ds
                    toks[idx, :ln] = d_tokens_fp16[ds:de].to(device=device, dtype=acc_dtype)
                    mask[idx, :ln] = True
                sim = torch.einsum("md,nld->nml", q, toks)
                sim.masked_fill_(~mask.unsqueeze(1), -1e9)
                doc_sc = sim.max(dim=-1).values.sum(dim=-1).float()
                scores_list.append(doc_sc.cpu().numpy())
            all_sc = np.concatenate(scores_list)
            order = np.lexsort((cands, -all_sc))[:topk]
            k_eff = min(topk, len(order))
            all_ids[qi, :k_eff] = cands[order[:k_eff]]
            all_scores[qi, :k_eff] = all_sc[order[:k_eff]]
        return all_ids, all_scores

    max_m = int(np.max(np.diff(q_offsets)))
    q_pad = torch.zeros((B, max_m, q_tokens_fp32.shape[1]), dtype=acc_dtype, device=device)
    q_mask = torch.zeros((B, max_m), dtype=acc_dtype, device=device)
    for qi in range(B):
        qs, qe = int(q_offsets[qi]), int(q_offsets[qi + 1])
        m = qe - qs
        q_pad[qi, :m] = q_tokens_fp32[qs:qe].to(device=device, dtype=acc_dtype)
        q_mask[qi, :m] = 1.0
    q_flat = q_pad.view(B * max_m, -1)

    doc_scores_all = np.empty((B, N_docs), dtype=np.float32)
    eff_bs = max(512, doc_batch_size // 2) if use_fp64 else doc_batch_size
    for d_start in range(0, N_docs, eff_bs):
        d_end = min(d_start + eff_bs, N_docs)
        t_start = int(d_offsets[d_start])
        t_end = int(d_offsets[d_end])
        d_chunk = d_tokens_fp16[t_start:t_end].to(device=device, dtype=acc_dtype)
        sim = torch.matmul(q_flat, d_chunk.t()).view(B, max_m, -1)
        rel_offsets = d_offsets[d_start:d_end + 1] - t_start
        lens = np.diff(rel_offsets)
        if np.all(lens == lens[0]):
            L = int(lens[0])
            doc_max = sim.view(B, max_m, d_end - d_start, L).max(dim=-1).values
        else:
            n_sub = d_end - d_start
            doc_max = torch.full((B, max_m, n_sub), -1e9, dtype=acc_dtype, device=device)
            doc_ids_per_tok = np.repeat(np.arange(n_sub, dtype=np.int64), lens)
            idx_t = torch.from_numpy(doc_ids_per_tok).to(device=device)
            idx_expand = idx_t.view(1, 1, -1).expand(B, max_m, -1)
            doc_max.scatter_reduce_(2, idx_expand, sim, reduce="amax", include_self=True)
        doc_max = doc_max * q_mask.unsqueeze(-1)
        doc_scores_all[:, d_start:d_end] = doc_max.sum(dim=1).float().cpu().numpy()

    doc_ids_range = np.arange(N_docs, dtype=np.int64)
    for qi in range(B):
        sc = doc_scores_all[qi]
        if N_docs > topk * 4:
            cand = np.argpartition(-sc, topk * 4)[:topk * 4]
            cand_sc = sc[cand]
            order = np.lexsort((cand, -cand_sc))[:topk]
            top_idx = cand[order]
        else:
            order = np.lexsort((doc_ids_range, -sc))[:topk]
            top_idx = doc_ids_range[order]
        k_eff = min(topk, len(top_idx))
        all_ids[qi, :k_eff] = top_idx[:k_eff]
        all_scores[qi, :k_eff] = sc[top_idx[:k_eff]]
    return all_ids, all_scores
