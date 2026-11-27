import os
import shutil
import tempfile
import ctypes
import numpy as np
import torch
import torch.nn.functional as F

from src.maxbound_engine import get_maxbound_ext, hadamard_matrix_torch


def _pad_queries(q_tokens_fp16, q_offsets, device, pad_mult=1):
    B = len(q_offsets) - 1
    q_lens = np.diff(q_offsets).astype(np.int32)
    m_max = int(q_lens.max())
    m_pad = ((m_max + pad_mult - 1) // pad_mult) * pad_mult
    D = q_tokens_fp16.shape[1]
    qt = q_tokens_fp16.to(device=device, dtype=torch.float16)
    if np.all(q_lens == m_max) and m_pad == m_max:
        q_pad = qt.view(B, m_max, D).contiguous()
    else:
        q_pad = torch.zeros((B, m_pad, D), dtype=torch.float16, device=device)
        if np.all(q_lens == m_max):
            q_pad[:, :m_max] = qt.view(B, m_max, D)
        else:
            for qi in range(B):
                qs, qe = int(q_offsets[qi]), int(q_offsets[qi + 1])
                q_pad[qi, :qe - qs] = qt[qs:qe]
    q_lens_gpu = torch.from_numpy(q_lens).to(device)
    return B, m_pad, D, q_pad, q_lens_gpu


class B1TorchPaddedGemm:
    """B1: Dense padded FP16 torch.matmul + masked max-reduce."""

    def __init__(self, d_tokens_fp16, d_offsets, device=1, chunk_docs=2048):
        self.device = device
        torch.cuda.set_device(device)
        self.N_docs = len(d_offsets) - 1
        self.D = d_tokens_fp16.shape[1]
        lens = np.diff(d_offsets)
        self.d_max = int(lens.max())
        self.chunk_docs = chunk_docs
        self.d_pad = torch.zeros((self.N_docs, self.d_max, self.D), dtype=torch.float16, device=device)
        self.d_mask = torch.zeros((self.N_docs, self.d_max), dtype=torch.bool, device=device)
        dt = d_tokens_fp16.to(device=device, dtype=torch.float16)
        for d in range(self.N_docs):
            ds, de = int(d_offsets[d]), int(d_offsets[d + 1])
            l = de - ds
            self.d_pad[d, :l] = dt[ds:de]
            self.d_mask[d, :l] = True

    @torch.no_grad()
    def search(self, q_tokens_fp16, q_offsets, topk=10):
        torch.cuda.set_device(self.device)
        B, m, D, q_pad, q_lens_gpu = _pad_queries(q_tokens_fp16, q_offsets, self.device)
        q_mask = (torch.arange(m, device=self.device).unsqueeze(0) < q_lens_gpu.unsqueeze(1))
        q_flat = q_pad.view(B * m, D)
        scores = torch.empty((B, self.N_docs), dtype=torch.float32, device=self.device)
        for ds in range(0, self.N_docs, self.chunk_docs):
            de = min(ds + self.chunk_docs, self.N_docs)
            nd = de - ds
            d_sub = self.d_pad[ds:de].view(nd * self.d_max, D)
            sim = torch.matmul(q_flat, d_sub.t()).float().view(B, m, nd, self.d_max)
            sim.masked_fill_(~self.d_mask[ds:de][None, None, :, :], -1e9)
            mx = sim.max(dim=-1).values
            mx.masked_fill_(~q_mask[:, :, None], 0.0)
            scores[:, ds:de] = mx.sum(dim=1)
        k_eff = min(topk, self.N_docs)
        vals, ids = torch.topk(scores, k=k_eff, dim=1)
        out_ids = np.full((B, topk), -1, dtype=np.int64)
        out_sc = np.full((B, topk), -1e9, dtype=np.float32)
        out_ids[:, :k_eff] = ids.cpu().numpy()
        out_sc[:, :k_eff] = vals.cpu().numpy()
        return out_ids, out_sc


class B2CuBLASUnpaddedExact:
    """B2: Length-unpadded cuBLAS FP16 GEMM + coalesced segmented CUDA reduction."""

    def __init__(self, d_tokens_fp16, d_offsets, device=1, chunk_docs=4096):
        self.device = device
        torch.cuda.set_device(device)
        self.ext = get_maxbound_ext(device)
        self.N_docs = len(d_offsets) - 1
        self.d_offs_np = np.asarray(d_offsets, dtype=np.int64)
        self.d_offs_gpu = torch.from_numpy(self.d_offs_np).to(device)
        self.d_tokens_gpu = d_tokens_fp16.to(device=device, dtype=torch.float16).contiguous()
        avg_len = max(1.0, float(d_tokens_fp16.shape[0]) / max(self.N_docs, 1))
        self.chunk_docs = min(chunk_docs, max(256, int(262144 / avg_len)))

    @torch.no_grad()
    def search(self, q_tokens_fp16, q_offsets, topk=10):
        torch.cuda.set_device(self.device)
        B, m, D, q_pad, q_lens_gpu = _pad_queries(q_tokens_fp16, q_offsets, self.device)
        q_flat = q_pad.view(B * m, D)
        scores = torch.empty((B, self.N_docs), dtype=torch.float32, device=self.device)
        for ds in range(0, self.N_docs, self.chunk_docs):
            de = min(ds + self.chunk_docs, self.N_docs)
            ts = int(self.d_offs_np[ds])
            te = int(self.d_offs_np[de])
            sim = torch.matmul(q_flat, self.d_tokens_gpu[ts:te].t())
            rel_offs = self.d_offs_gpu[ds:de + 1] - ts
            scores[:, ds:de] = self.ext.unpadded_cublas_reduce(sim, rel_offs, q_lens_gpu, B, m, de - ds)
        k_eff = min(topk, self.N_docs)
        vals, ids = torch.topk(scores, k=k_eff, dim=1)
        out_ids = np.full((B, topk), -1, dtype=np.int64)
        out_sc = np.full((B, topk), -1e9, dtype=np.float32)
        out_ids[:, :k_eff] = ids.cpu().numpy()
        out_sc[:, :k_eff] = vals.cpu().numpy()
        return out_ids, out_sc

    @torch.no_grad()
    def search_rerank(self, q_tokens_fp16, q_offsets, cand_ids, topk=10):
        """Mode B Exact Reranking via level2_exact_verify over all [B, C] candidates (no pruning)."""
        torch.cuda.set_device(self.device)
        if not torch.is_tensor(cand_ids):
            cand_ids = torch.from_numpy(np.asarray(cand_ids, dtype=np.int32))
        cand_ids = cand_ids.to(device=self.device, dtype=torch.int32).contiguous()
        B, C = cand_ids.shape
        _, m, _, q_pad, q_lens_gpu = _pad_queries(q_tokens_fp16, q_offsets, self.device)
        sq = torch.arange(B, device=self.device, dtype=torch.int32).unsqueeze(1).repeat(1, C).reshape(-1).contiguous()
        sd = cand_ids.reshape(-1).contiguous()
        sc = self.ext.level2_exact_verify(sq, sd, q_pad, q_lens_gpu, self.d_tokens_gpu, self.d_offs_gpu).view(B, C)
        k_eff = min(topk, C)
        vals, pos = torch.topk(sc, k=k_eff, dim=1)
        top_docs = torch.gather(cand_ids.long(), 1, pos)
        out_ids = np.full((B, topk), -1, dtype=np.int64)
        out_sc = np.full((B, topk), -1e9, dtype=np.float32)
        out_ids[:, :k_eff] = top_docs.cpu().numpy()
        out_sc[:, :k_eff] = vals.cpu().numpy()
        return out_ids, out_sc


class B8FusedFP16TCExact:
    """
    B8: Strong Hand-Crafted Fused Tensor-Core FP16 MaxSim Exact Baseline (WMMA hmma.16816.f32).
      - Uses length-sorted unpadded FP16 storage + W=8 SMEM doc-tile reuse across queries.
      - Computes exact MaxSim online in registers/SMEM with ZERO intermediate similarity matrix HBM I/O!
      - Isolates MAXBOUND's quantized bound-pruning speedup from pure kernel-fusion benefits.
    """

    def __init__(self, d_tokens_fp16, d_offsets, device=1, sort_by_length=True):
        self.device = device
        torch.cuda.set_device(device)
        self.ext = get_maxbound_ext(device)
        self.N_docs = len(d_offsets) - 1
        raw_offs = np.asarray(d_offsets, dtype=np.int64)
        raw_lens = np.diff(raw_offs)
        self.N_toks, self.D = d_tokens_fp16.shape
        self.sort_by_length = bool(sort_by_length and self.N_docs > 1)

        if self.sort_by_length:
            self.perm_to_orig_np = np.argsort(raw_lens, kind="stable").astype(np.int64)
            self.perm_to_orig_gpu = torch.from_numpy(self.perm_to_orig_np).to(device=device, dtype=torch.int64)
            lens = raw_lens[self.perm_to_orig_np]
            self.d_offs_np = np.zeros(self.N_docs + 1, dtype=np.int64)
            np.cumsum(lens, out=self.d_offs_np[1:])
            src_s = np.repeat(raw_offs[:-1][self.perm_to_orig_np], lens)
            dst_s = np.repeat(self.d_offs_np[:-1], lens)
            gather_idx = src_s + (np.arange(self.N_toks, dtype=np.int64) - dst_s)
            d_tokens_fp16 = d_tokens_fp16[torch.from_numpy(gather_idx)]
        else:
            self.perm_to_orig_gpu = None
            self.d_offs_np = raw_offs.copy()

        self.d_offs_cpu = torch.from_numpy(self.d_offs_np)
        self.d_offs_gpu = torch.from_numpy(self.d_offs_np).to(device=device, dtype=torch.int64)
        self.d_tokens_gpu = d_tokens_fp16.to(device=device, dtype=torch.float16).contiguous()
        self._sim_l2_buf = (
            torch.empty(12 * 1024 * 1024, dtype=torch.float16, device=device)
            if self.D > 128 else None
        )

    @torch.no_grad()
    def search(self, q_tokens_fp16, q_offsets, topk=10):
        torch.cuda.set_device(self.device)
        B, m_pad, D, q_pad, q_lens_gpu = _pad_queries(q_tokens_fp16, q_offsets, self.device, pad_mult=16)
        if D > 128 and B >= 4 and self._sim_l2_buf is not None:
            scores = self.ext.exec_b8_hd_l2_fp16_maxsim(
                q_pad, q_lens_gpu, self.d_tokens_gpu, self.d_offs_gpu, self.d_offs_cpu, self._sim_l2_buf
            )
        else:
            scores = self.ext.fused_tc_fp16_maxsim(q_pad, q_lens_gpu, self.d_tokens_gpu, self.d_offs_gpu)
        k_eff = min(topk, self.N_docs)
        vals, ids = torch.topk(scores, k=k_eff, dim=1)
        if self.sort_by_length:
            ids = self.perm_to_orig_gpu[ids]
        out_ids = np.full((B, topk), -1, dtype=np.int64)
        out_sc = np.full((B, topk), -1e9, dtype=np.float32)
        out_ids[:, :k_eff] = ids.cpu().numpy()
        out_sc[:, :k_eff] = vals.cpu().numpy()
        return out_ids, out_sc


class B3QuantizedDecompressGemm:
    """B3: Store INT8-compressed vectors in VRAM, decompress chunk-by-chunk -> cuBLAS FP16 GEMM (no pruning)."""

    def __init__(self, d_tokens_fp16, d_offsets, device=1, chunk_docs=4096):
        self.device = device
        torch.cuda.set_device(device)
        self.ext = get_maxbound_ext(device)
        self.N_docs = len(d_offsets) - 1
        self.N_toks, self.D = d_tokens_fp16.shape
        self.d_offs_np = np.asarray(d_offsets, dtype=np.int64)
        self.d_offs_gpu = torch.from_numpy(self.d_offs_np).to(device)
        self.chunk_docs = chunk_docs
        self.H = hadamard_matrix_torch(self.D, device)

        self.d_int8 = torch.empty((self.N_toks, self.D), dtype=torch.int8, device=device)
        self.d_scales = torch.empty(self.N_toks, dtype=torch.float16, device=device)
        for s in range(0, self.N_toks, 131072):
            e = min(s + 131072, self.N_toks)
            xb = d_tokens_fp16[s:e].to(device=device, dtype=torch.float32)
            hr = xb @ self.H.t()
            sc = hr.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 127.0
            self.d_int8[s:e] = torch.round(hr / sc).clamp(-127, 127).to(torch.int8)
            self.d_scales[s:e] = sc.squeeze(1).half()

    @torch.no_grad()
    def search(self, q_tokens_fp16, q_offsets, topk=10):
        torch.cuda.set_device(self.device)
        B, m, D, q_pad, q_lens_gpu = _pad_queries(q_tokens_fp16, q_offsets, self.device)
        q_rot = (q_pad.float() @ self.H.t()).half().view(B * m, D)
        scores = torch.empty((B, self.N_docs), dtype=torch.float32, device=self.device)
        for ds in range(0, self.N_docs, self.chunk_docs):
            de = min(ds + self.chunk_docs, self.N_docs)
            ts = int(self.d_offs_np[ds])
            te = int(self.d_offs_np[de])
            d_fp16 = self.d_int8[ts:te].half() * self.d_scales[ts:te].unsqueeze(1)
            sim = torch.matmul(q_rot, d_fp16.t())
            rel_offs = self.d_offs_gpu[ds:de + 1] - ts
            scores[:, ds:de] = self.ext.unpadded_cublas_reduce(sim, rel_offs, q_lens_gpu, B, m, de - ds)
        k_eff = min(topk, self.N_docs)
        vals, ids = torch.topk(scores, k=k_eff, dim=1)
        out_ids = np.full((B, topk), -1, dtype=np.int64)
        out_sc = np.full((B, topk), -1e9, dtype=np.float32)
        out_ids[:, :k_eff] = ids.cpu().numpy()
        out_sc[:, :k_eff] = vals.cpu().numpy()
        return out_ids, out_sc


class B4OfficialFastPlaid:
    """
    Official FastPLAID (LightOn fast-plaid 1.7.0 Rust + CUDA Engine) Baseline!
    Evaluates the genuine community-standard quantized PLAID engine (2-bit or 4-bit residual codes) on GPU.
    """

    def __init__(self, d_tokens_fp16, d_offsets, device=1, nbits=2, kmeans_niters=4, seed=42):
        from fast_plaid import search as fp_search
        self.device = device
        self.nbits = nbits
        self.N_docs = len(d_offsets) - 1
        self.D = d_tokens_fp16.shape[1]
        self.tmp_dir = tempfile.mkdtemp(prefix=f"maxbound_fp_b{nbits}_")
        self.engine = fp_search.FastPlaid(index=self.tmp_dir, device=f"cuda:{device}")
        dt_cpu = d_tokens_fp16.detach().cpu().half()
        offs = np.asarray(d_offsets, dtype=np.int64)
        docs_list = [dt_cpu[int(offs[i]):int(offs[i + 1])] for i in range(self.N_docs)]
        n_samp = min(self.N_docs, 32768)
        self.engine.create(
            documents_embeddings=docs_list,
            kmeans_niters=kmeans_niters,
            nbits=nbits,
            n_samples_kmeans=n_samp,
            seed=seed,
        )

    def cleanup(self):
        if getattr(self, "tmp_dir", None) and os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            self.tmp_dir = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    @torch.no_grad()
    def search(self, q_tokens_fp16, q_offsets, n_ivf_probe=8, n_full_scores=4096, topk=10):
        B = len(q_offsets) - 1
        q_lens = np.diff(q_offsets)
        m_max = int(q_lens.max())
        if np.all(q_lens == m_max):
            q_in = q_tokens_fp16.detach().to(device=f"cuda:{self.device}", dtype=torch.float16).view(B, m_max, self.D)
        else:
            qt = q_tokens_fp16.detach().to(device=f"cuda:{self.device}", dtype=torch.float16)
            q_in = [qt[int(q_offsets[i]):int(q_offsets[i + 1])] for i in range(B)]
        res = self.engine.search(
            queries_embeddings=q_in,
            top_k=min(topk, self.N_docs),
            n_ivf_probe=n_ivf_probe,
            n_full_scores=min(n_full_scores, self.N_docs),
            show_progress=False,
        )
        out_ids = np.full((B, topk), -1, dtype=np.int64)
        out_sc = np.full((B, topk), -1e9, dtype=np.float32)
        for qi, hits in enumerate(res):
            for rank, (did, sc) in enumerate(hits[:topk]):
                out_ids[qi, rank] = int(did)
                out_sc[qi, rank] = float(sc)
        return out_ids, out_sc


class B9WarpGPU:
    """
    B9: GPU-Native WARP (SIGIR'25 XTR/WARP: Centroid-Selective Implicit Residual Decompression).
    Stores tokens in centroid-clustered inverted lists with 8-bit/4-bit residual codes, evaluates ONLY
    tokens inside the top-t_prime probed centroids per query token via implicit centroid+residual scoring,
    and optionally refines top-C candidates or returns pure WARP implicit MaxSim scores!
    """

    def __init__(self, d_tokens_fp16, d_offsets, device=1, n_centroids=1024, kmeans_iters=10, seed=2027):
        self.device = device
        torch.cuda.set_device(device)
        self.ext = get_maxbound_ext(device)
        self.N_docs = len(d_offsets) - 1
        self.N_toks, self.D = d_tokens_fp16.shape
        self.d_offs_np = np.asarray(d_offsets, dtype=np.int64)
        self.d_offs_gpu = torch.from_numpy(self.d_offs_np).to(device)
        self.d_tokens_gpu = d_tokens_fp16.to(device=device, dtype=torch.float16).contiguous()
        self.K = min(n_centroids, max(32, self.N_docs // 2))

        rng = np.random.default_rng(seed)
        sample_n = min(self.N_toks, 131072)
        s_idx = torch.from_numpy(rng.choice(self.N_toks, size=sample_n, replace=False)).to(device)
        x_sample = self.d_tokens_gpu[s_idx].float()
        perm = torch.randperm(sample_n, device=device)[:self.K]
        C = F.normalize(x_sample[perm], p=2, dim=1)
        for _ in range(kmeans_iters):
            sim = x_sample @ C.t()
            assign = sim.argmax(dim=1)
            new_C = torch.zeros_like(C)
            new_C.index_add_(0, assign, x_sample)
            cnt = torch.bincount(assign, minlength=self.K).float().unsqueeze(1).clamp(min=1.0)
            C = F.normalize(new_C / cnt, p=2, dim=1)
        self.centroids = C.half().contiguous()

        lens = np.diff(self.d_offs_np)
        doc_seg = torch.from_numpy(np.repeat(np.arange(self.N_docs, dtype=np.int64), lens)).to(device=device, dtype=torch.int32)
        cids_all = torch.empty(self.N_toks, dtype=torch.int64, device=device)
        for s in range(0, self.N_toks, 131072):
            e = min(s + 131072, self.N_toks)
            cids_all[s:e] = (self.d_tokens_gpu[s:e] @ self.centroids.t()).argmax(dim=1)

        # Sort tokens by (centroid ID, doc_id) into contiguous inverted lists + 8-bit quantized residuals
        sort_key = cids_all * self.N_docs + doc_seg.long()
        sort_idx = torch.argsort(sort_key)
        sorted_cids = cids_all[sort_idx]
        self.inv_doc_ids = doc_seg[sort_idx].contiguous()
        c_counts = torch.bincount(sorted_cids, minlength=self.K)
        self.c_offs = torch.zeros(self.K + 1, dtype=torch.int64, device=device)
        torch.cumsum(c_counts, dim=0, out=self.c_offs[1:])
        self.c_offs_cpu = self.c_offs.cpu().numpy()

        # Store compressed INT8 residual + FP16 scale in centroid-sorted order for implicit decompression
        self.res_i8 = torch.empty((self.N_toks, self.D), dtype=torch.int8, device=device)
        self.res_sc = torch.empty(self.N_toks, dtype=torch.float16, device=device)
        for s in range(0, self.N_toks, 131072):
            e = min(s + 131072, self.N_toks)
            orig_tok = self.d_tokens_gpu[sort_idx[s:e]].float()
            cent_tok = self.centroids[sorted_cids[s:e]].float()
            res = orig_tok - cent_tok
            sc = res.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 127.0
            self.res_i8[s:e] = torch.round(res / sc).clamp(-127, 127).to(torch.int8)
            self.res_sc[s:e] = sc.squeeze(1).half()

        # Reusable L2-friendly atomicMax workspace (self-cleaning to 0 after every query!)
        q_chunk = min(64, max(8, int(32_000_000 // max(self.N_docs, 1))))
        self._ws_max_i32 = torch.zeros((q_chunk, self.N_docs), dtype=torch.int32, device=device)

    @torch.no_grad()
    def search(self, q_tokens_fp16, q_offsets, t_prime=8, topk=10):
        """Pure CUDA 2-pass WARP implicit centroid-residual decompression + missing-token floor estimation."""
        torch.cuda.set_device(self.device)
        B, m, D, q_pad, q_lens_gpu = _pad_queries(q_tokens_fp16, q_offsets, self.device)
        tp = min(t_prime, self.K)
        q_flat = q_pad.view(B * m, D).contiguous()
        c_scores = (q_flat @ self.centroids.t()).float()  # [B*m, K]
        top_c_sc, top_c_id = torch.topk(c_scores, k=tp, dim=1)  # [B*m, tp]

        # WARP implicit missing-score floor = tp-th centroid score per query token
        floors = top_c_sc[:, -1].view(B, m)
        q_mask = (torch.arange(m, device=self.device).unsqueeze(0) < q_lens_gpu.unsqueeze(1))
        floors = floors.masked_fill(~q_mask, 0.0).contiguous()

        doc_scores = floors.sum(dim=1, keepdim=True).expand(B, self.N_docs).contiguous().clone()
        self.ext.warp_implicit_maxsim(
            q_flat, q_lens_gpu, top_c_id.contiguous(), top_c_sc.contiguous(), floors.view(-1),
            self.c_offs, self.inv_doc_ids, self.res_i8, self.res_sc, self._ws_max_i32, doc_scores,
            B, m, tp
        )

        k_eff = min(topk, self.N_docs)
        vals, ids = torch.topk(doc_scores, k=k_eff, dim=1)
        out_ids = np.full((B, topk), -1, dtype=np.int64)
        out_sc = np.full((B, topk), -1e9, dtype=np.float32)
        out_ids[:, :k_eff] = ids.cpu().numpy()
        out_sc[:, :k_eff] = vals.cpu().numpy()
        return out_ids, out_sc


class B10EmvbGPU:
    """
    B10: GPU-Native EMVB (Nardini et al., ECIR 2024: Efficient Multi-Vector Retrieval via Bit-Vectors + PQ16 ADC).
      - Stage 1: Document centroid bit-vectors + hardware __popcll candidate prefiltering.
      - Stage 2 & 3: Bit-vector token filtering + vectorized uint4 16-subvector Product Quantization (PQ16) MaxSim.
      - Stage 4: Exact FP16 pair refinement on top-C_refine candidates.
    """

    def __init__(self, d_tokens_fp16, d_offsets, device=1, n_centroids=1024, m_pq=16, kmeans_iters=8, seed=2027):
        self.device = device
        torch.cuda.set_device(device)
        self.ext = get_maxbound_ext(device)
        self.N_docs = len(d_offsets) - 1
        self.N_toks, self.D = d_tokens_fp16.shape
        assert self.D % m_pq == 0 and m_pq == 16, f"EMVB PQ16 requires D={self.D} % 16 == 0"
        self.m_pq = m_pq
        self.d_sub = self.D // m_pq
        self.d_offs_np = np.asarray(d_offsets, dtype=np.int64)
        self.d_offs_gpu = torch.from_numpy(self.d_offs_np).to(device)
        self.d_tokens_gpu = d_tokens_fp16.to(device=device, dtype=torch.float16).contiguous()

        raw_k = min(n_centroids, max(64, ((self.N_docs // 2) // 64) * 64))
        self.K = max(64, (raw_k // 64) * 64)
        self.n_words = self.K // 64

        rng = np.random.default_rng(seed)
        sample_n = min(self.N_toks, 131072)
        s_idx = torch.from_numpy(rng.choice(self.N_toks, size=sample_n, replace=False)).to(device)
        x_sample = self.d_tokens_gpu[s_idx].float()
        perm = torch.randperm(sample_n, device=device)[:self.K]
        C = F.normalize(x_sample[perm], p=2, dim=1)
        for _ in range(kmeans_iters):
            assign = (x_sample @ C.t()).argmax(dim=1)
            new_C = torch.zeros_like(C)
            new_C.index_add_(0, assign, x_sample)
            cnt = torch.bincount(assign, minlength=self.K).float().unsqueeze(1).clamp(min=1.0)
            C = F.normalize(new_C / cnt, p=2, dim=1)
        self.centroids = C.half().contiguous()

        # Assign tokens to centroids and pack 64-bit document bit-vectors
        lens = np.diff(self.d_offs_np)
        doc_seg = torch.from_numpy(np.repeat(np.arange(self.N_docs, dtype=np.int64), lens)).to(device)
        self.tok_cids = torch.empty(self.N_toks, dtype=torch.int16, device=device)
        doc_bv = torch.zeros((self.N_docs, self.n_words), dtype=torch.int64, device=device)
        for s in range(0, self.N_toks, 131072):
            e = min(s + 131072, self.N_toks)
            cid = (self.d_tokens_gpu[s:e] @ self.centroids.t()).argmax(dim=1)
            self.tok_cids[s:e] = cid.to(torch.int16)
            # Pack into doc_bv via unique (doc, word, bit) triples in chunk
            dids = doc_seg[s:e]
            w_idx = cid >> 6
            b_idx = cid & 63
            # Bitwise OR via scatter_reduce_ ("amax" on individual bits or loop over 64 bit positions? Wait:
            # torch doesn't have bitwise_or scatter in older versions, so accumulate bitmask per (doc*n_words+w) cleanly!)
            dw = dids * self.n_words + w_idx
            # Sort by dw and reduce OR in vectorized segments, or CPU numpy bitor (very fast: 131K int64s = 0.4ms!)
            dw_np = dw.cpu().numpy()
            bit_np = (np.uint64(1) << b_idx.cpu().numpy().astype(np.uint64)).view(np.int64)
            doc_bv_np_view = doc_bv.cpu().numpy() if s == 0 else doc_bv_np_view
            np.bitwise_or.at(doc_bv_np_view.reshape(-1), dw_np, bit_np)
        self.doc_bitvecs = torch.from_numpy(doc_bv_np_view).to(device=device, dtype=torch.int64).contiguous()

        # Train M=16, K=256 Product Quantizer codebooks in parallel across all 16 subspaces!
        pq_samp = x_sample[:min(sample_n, 65536)].view(-1, self.m_pq, self.d_sub).permute(1, 0, 2).contiguous()  # [16, Ns, d_sub]
        k_pq = 256
        cb = pq_samp[:, :k_pq, :].clone()  # [16, 256, d_sub]
        for _ in range(6):
            dists = torch.cdist(pq_samp, cb)  # [16, Ns, 256]
            a = dists.argmin(dim=2)  # [16, Ns]
            new_cb = torch.zeros_like(cb)
            new_cb.scatter_add_(1, a.unsqueeze(2).expand(-1, -1, self.d_sub), pq_samp)
            ones = torch.ones((self.m_pq, pq_samp.size(1)), dtype=torch.float32, device=device)
            cnt_pq = torch.zeros((self.m_pq, k_pq), dtype=torch.float32, device=device)
            cnt_pq.scatter_add_(1, a, ones)
            cb = torch.where(cnt_pq.unsqueeze(2) > 0, new_cb / cnt_pq.unsqueeze(2).clamp(min=1.0), cb)
        self.pq_codebooks = cb.half().contiguous()  # [16, 256, d_sub]
        self.pq_codebooks_t = self.pq_codebooks.transpose(1, 2).contiguous()  # [16, d_sub, 256]

        self.pq_codes = torch.empty((self.N_toks, self.m_pq), dtype=torch.uint8, device=device)
        cb_f = self.pq_codebooks.float()
        for s in range(0, self.N_toks, 65536):
            e = min(s + 65536, self.N_toks)
            xs = self.d_tokens_gpu[s:e].float().view(-1, self.m_pq, self.d_sub).permute(1, 0, 2).contiguous()
            codes_s = torch.cdist(xs, cb_f).argmin(dim=2).t().contiguous()  # [e-s, 16]
            self.pq_codes[s:e] = codes_s.to(torch.uint8)
        self.bit_shifts = torch.arange(64, device=device, dtype=torch.int64)

    def _build_query_bitmasks(self, csim, q_mask, nprobe_doc, nprobe_tok):
        B, m, K = csim.shape
        np_d = min(nprobe_doc, K)
        np_t = min(nprobe_tok, K)
        _, top_t = torch.topk(csim, k=np_t, dim=-1)  # [B, m, np_t]
        bool_tok = torch.zeros((B, m, K), dtype=torch.int64, device=self.device)
        bool_tok.scatter_(2, top_t, 1)
        bool_tok.masked_fill_(~q_mask[:, :, None], 0)
        q_tok_mask = (bool_tok.view(B, m, self.n_words, 64) << self.bit_shifts).sum(dim=-1).contiguous()
        q_union_mask = (bool_tok.amax(dim=1).view(B, self.n_words, 64) << self.bit_shifts).sum(dim=-1).contiguous()

        if np_d == np_t:
            q_doc_mask = q_union_mask
        else:
            _, top_d = torch.topk(csim, k=np_d, dim=-1)
            bool_doc = torch.zeros((B, m, K), dtype=torch.int64, device=self.device)
            bool_doc.scatter_(2, top_d, 1)
            bool_doc.masked_fill_(~q_mask[:, :, None], 0)
            q_doc_mask = (bool_doc.amax(dim=1).view(B, self.n_words, 64) << self.bit_shifts).sum(dim=-1).contiguous()
        return q_doc_mask, q_union_mask, q_tok_mask

    @torch.no_grad()
    def search(self, q_tokens_fp16, q_offsets, nprobe=8, ncands=1024, nrefine=128, topk=10):
        torch.cuda.set_device(self.device)
        B, m, D, q_pad, q_lens_gpu = _pad_queries(q_tokens_fp16, q_offsets, self.device)
        q_mask = (torch.arange(m, device=self.device).unsqueeze(0) < q_lens_gpu.unsqueeze(1))
        csim = (q_pad.view(B * m, D) @ self.centroids.t()).view(B, m, self.K)
        q_doc_mask, q_union_mask, q_tok_mask = self._build_query_bitmasks(
            csim, q_mask, nprobe_doc=nprobe, nprobe_tok=min(self.K, nprobe * 4)
        )

        # Stage 1: Hardware __popcll bit-vector candidate prefiltering
        bv_scores = self.ext.emvb_bitvec_filter(q_doc_mask, self.doc_bitvecs)
        k_c = min(ncands, self.N_docs)
        _, cand_ids = torch.topk(bv_scores, k=k_c, dim=1)
        cand_ids = cand_ids.to(torch.int32).contiguous()

        # Stage 2 & 3: Bit-masked token filter + PQ16 ADC lookup
        q_sub = q_pad.view(B * m, self.m_pq, self.d_sub).permute(1, 0, 2).contiguous()  # [16, B*m, d_sub]
        lut = torch.bmm(q_sub, self.pq_codebooks_t).permute(1, 0, 2).float().reshape(B, m, 16, 256).contiguous()
        pq_scores = self.ext.emvb_pq16_maxsim(
            q_union_mask, q_tok_mask, lut, q_lens_gpu, cand_ids,
            self.d_offs_gpu, self.tok_cids, self.pq_codes
        )
        k_r = min(max(nrefine, topk), k_c)
        _, ref_pos = torch.topk(pq_scores, k=k_r, dim=1)
        ref_ids = torch.gather(cand_ids, 1, ref_pos).contiguous()

        # Stage 4: Exact FP16 refinement
        sq = torch.arange(B, device=self.device, dtype=torch.int32).unsqueeze(1).repeat(1, k_r).reshape(-1).contiguous()
        sd = ref_ids.reshape(-1).contiguous()
        sc = self.ext.level2_exact_verify(sq, sd, q_pad, q_lens_gpu, self.d_tokens_gpu, self.d_offs_gpu).view(B, k_r)
        k_eff = min(topk, k_r)
        vals, pos = torch.topk(sc, k=k_eff, dim=1)
        top_docs = torch.gather(ref_ids.long(), 1, pos)
        out_ids = np.full((B, topk), -1, dtype=np.int64)
        out_sc = np.full((B, topk), -1e9, dtype=np.float32)
        out_ids[:, :k_eff] = top_docs.cpu().numpy()
        out_sc[:, :k_eff] = vals.cpu().numpy()
        return out_ids, out_sc

    @torch.no_grad()
    def get_candidates(self, q_tokens_fp16, q_offsets, nprobe=8, ncands=512):
        torch.cuda.set_device(self.device)
        B, m, D, q_pad, q_lens_gpu = _pad_queries(q_tokens_fp16, q_offsets, self.device)
        q_mask = (torch.arange(m, device=self.device).unsqueeze(0) < q_lens_gpu.unsqueeze(1))
        csim = (q_pad.view(B * m, D) @ self.centroids.t()).view(B, m, self.K)
        q_doc_mask, _, _ = self._build_query_bitmasks(csim, q_mask, nprobe_doc=nprobe, nprobe_tok=nprobe)
        bv_scores = self.ext.emvb_bitvec_filter(q_doc_mask, self.doc_bitvecs)
        k_c = min(ncands, self.N_docs)
        _, cand_ids = torch.topk(bv_scores, k=k_c, dim=1)
        return cand_ids.to(torch.int32).contiguous()


class B4PlaidGPU:
    """
    B4: GPU-Native PLAID (Centroids + Inverted Lists + nprobe pruning + exact FP16 Stage-2 rerank).
    Also supports generating top-C candidate lists for hybrid PLAID+MAXBOUND!
    """

    def __init__(self, d_tokens_fp16, d_offsets, device=1, n_centroids=1024, kmeans_iters=12, seed=2027):
        self.device = device
        torch.cuda.set_device(device)
        self.ext = get_maxbound_ext(device)
        self.N_docs = len(d_offsets) - 1
        self.N_toks, self.D = d_tokens_fp16.shape
        self.d_offs_np = np.asarray(d_offsets, dtype=np.int64)
        self.d_offs_gpu = torch.from_numpy(self.d_offs_np).to(device)
        self.d_tokens_gpu = d_tokens_fp16.to(device=device, dtype=torch.float16).contiguous()
        self.K = min(n_centroids, max(32, self.N_docs // 2))

        rng = np.random.default_rng(seed)
        sample_n = min(self.N_toks, 131072)
        s_idx = torch.from_numpy(rng.choice(self.N_toks, size=sample_n, replace=False)).to(device)
        x_sample = self.d_tokens_gpu[s_idx].float()
        perm = torch.randperm(sample_n, device=device)[:self.K]
        C = F.normalize(x_sample[perm], p=2, dim=1)
        for _ in range(kmeans_iters):
            sim = x_sample @ C.t()
            assign = sim.argmax(dim=1)
            new_C = torch.zeros_like(C)
            new_C.index_add_(0, assign, x_sample)
            cnt = torch.bincount(assign, minlength=self.K).float().unsqueeze(1).clamp(min=1.0)
            C = F.normalize(new_C / cnt, p=2, dim=1)
        self.centroids = C.half().contiguous()

        lens = np.diff(self.d_offs_np)
        doc_seg = torch.from_numpy(np.repeat(np.arange(self.N_docs, dtype=np.int64), lens)).to(device)
        pairs = []
        for s in range(0, self.N_toks, 131072):
            e = min(s + 131072, self.N_toks)
            cid = (self.d_tokens_gpu[s:e] @ self.centroids.t()).argmax(dim=1)
            pairs.append(cid * self.N_docs + doc_seg[s:e])
        uniq_cd = torch.unique(torch.cat(pairs))
        cids = uniq_cd // self.N_docs
        dids = uniq_cd % self.N_docs
        order = torch.argsort(dids)
        self.inv_cids = cids[order].contiguous()
        d_counts = torch.bincount(dids, minlength=self.N_docs)
        self.d_c_offs = torch.zeros(self.N_docs + 1, dtype=torch.int64, device=device)
        torch.cumsum(d_counts, dim=0, out=self.d_c_offs[1:])
        self.doc_centroid_csr = torch.sparse_csr_tensor(
            self.d_c_offs, self.inv_cids.long(),
            torch.ones(self.inv_cids.numel(), dtype=torch.float32, device=device),
            size=(self.N_docs, self.K)
        )

    @torch.no_grad()
    def get_candidates(self, q_tokens_fp16, q_offsets, nprobe=4, ncands=512):
        torch.cuda.set_device(self.device)
        B, m, D, q_pad, q_lens_gpu = _pad_queries(q_tokens_fp16, q_offsets, self.device)
        q_mask = (torch.arange(m, device=self.device).unsqueeze(0) < q_lens_gpu.unsqueeze(1))
        csim = (q_pad.view(B * m, D) @ self.centroids.t()).view(B, m, self.K)
        top_c_val, top_c_idx = torch.topk(csim, k=min(nprobe, self.K), dim=-1)
        probe_mask = torch.zeros((B, m, self.K), dtype=torch.float32, device=self.device)
        probe_mask.scatter_(2, top_c_idx, top_c_val.clamp(min=0.0).float())
        probe_mask.masked_fill_(~q_mask[:, :, None], 0.0)
        q_centroid_sum = probe_mask.sum(dim=1)
        doc_approx = torch.sparse.mm(self.doc_centroid_csr, q_centroid_sum.t()).t()
        k_c = min(ncands, self.N_docs)
        _, cand_ids = torch.topk(doc_approx, k=k_c, dim=1)
        return cand_ids.to(torch.int32).contiguous()

    @torch.no_grad()
    def search(self, q_tokens_fp16, q_offsets, nprobe=4, ncands=512, topk=10):
        cand_ids = self.get_candidates(q_tokens_fp16, q_offsets, nprobe=nprobe, ncands=ncands)
        B, C = cand_ids.shape
        _, m, _, q_pad, q_lens_gpu = _pad_queries(q_tokens_fp16, q_offsets, self.device)
        sq = torch.arange(B, device=self.device, dtype=torch.int32).unsqueeze(1).repeat(1, C).reshape(-1).contiguous()
        sd = cand_ids.reshape(-1).contiguous()
        sc = self.ext.level2_exact_verify(sq, sd, q_pad, q_lens_gpu, self.d_tokens_gpu, self.d_offs_gpu).view(B, C)
        k_eff = min(topk, C)
        vals, pos = torch.topk(sc, k=k_eff, dim=1)
        top_docs = torch.gather(cand_ids.long(), 1, pos)
        out_ids = np.full((B, topk), -1, dtype=np.int64)
        out_sc = np.full((B, topk), -1e9, dtype=np.float32)
        out_ids[:, :k_eff] = top_docs.cpu().numpy()
        out_sc[:, :k_eff] = vals.cpu().numpy()
        return out_ids, out_sc


class B5MuveraGPU:
    """
    B5: GPU-Native MUVERA (NeurIPS'24 Fixed Dimensional Encoding FDE via SimHash partitions + projection -> exact Stage-2 rerank).
    Also provides get_candidates() for MUVERA+MAXBOUND!
    """

    def __init__(self, d_tokens_fp16, d_offsets, device=1, r_reps=8, k_sim=4, d_proj=16, seed=2027):
        self.device = device
        torch.cuda.set_device(device)
        self.ext = get_maxbound_ext(device)
        self.N_docs = len(d_offsets) - 1
        self.N_toks, self.D = d_tokens_fp16.shape
        self.d_offs_np = np.asarray(d_offsets, dtype=np.int64)
        self.d_offs_gpu = torch.from_numpy(self.d_offs_np).to(device)
        self.d_tokens_gpu = d_tokens_fp16.to(device=device, dtype=torch.float16).contiguous()

        self.r_reps = r_reps
        self.k_sim = k_sim
        self.n_part = 1 << k_sim
        self.d_proj = d_proj
        self.fde_dim = r_reps * self.n_part * d_proj

        gen = torch.Generator(device=device).manual_seed(seed)
        self.S_hash = torch.randn((r_reps, self.D, k_sim), generator=gen, device=device, dtype=torch.float16)
        self.P_proj = (torch.randn((r_reps, self.D, d_proj), generator=gen, device=device, dtype=torch.float16) / np.sqrt(d_proj))
        self.bit_weights = (1 << torch.arange(k_sim, device=device, dtype=torch.int64))

        lens = np.diff(self.d_offs_np)
        doc_seg = torch.from_numpy(np.repeat(np.arange(self.N_docs, dtype=np.int64), lens)).to(device)
        d_fde = torch.zeros((self.N_docs, self.fde_dim), dtype=torch.float32, device=device)
        d_cnt = torch.zeros((self.N_docs, r_reps * self.n_part), dtype=torch.float32, device=device)

        for s in range(0, self.N_toks, 65536):
            e = min(s + 65536, self.N_toks)
            xb = self.d_tokens_gpu[s:e]
            dids = doc_seg[s:e]
            for r in range(r_reps):
                bits = ((xb @ self.S_hash[r]) > 0).to(torch.int64)
                part = (bits * self.bit_weights).sum(dim=1) + r * self.n_part
                px = (xb @ self.P_proj[r]).float()
                bucket = dids * (r_reps * self.n_part) + part
                d_fde.view(self.N_docs * r_reps * self.n_part, d_proj).index_add_(0, bucket, px)
                d_cnt.view(-1).index_add_(0, bucket, torch.ones(e - s, device=device))

        denom = d_cnt.clamp(min=1.0).unsqueeze(-1)
        d_fde = (d_fde.view(self.N_docs, r_reps * self.n_part, d_proj) / denom).view(self.N_docs, self.fde_dim)
        self.d_fde = d_fde.half().contiguous()

    def _encode_queries_fde(self, q_pad, q_lens_gpu):
        B, m, D = q_pad.shape
        q_mask = (torch.arange(m, device=self.device).unsqueeze(0) < q_lens_gpu.unsqueeze(1))
        q_fde = torch.zeros((B, self.r_reps * self.n_part, self.d_proj), dtype=torch.float32, device=self.device)
        q_flat = q_pad.view(B * m, D)
        q_ids = torch.arange(B, device=self.device, dtype=torch.int64).unsqueeze(1).expand(B, m).reshape(-1)
        valid = q_mask.reshape(-1)
        qv = q_flat[valid]
        qids_v = q_ids[valid]
        for r in range(self.r_reps):
            bits = ((qv @ self.S_hash[r]) > 0).to(torch.int64)
            part = (bits * self.bit_weights).sum(dim=1) + r * self.n_part
            px = (qv @ self.P_proj[r]).float()
            bucket = qids_v * (self.r_reps * self.n_part) + part
            q_fde.view(B * self.r_reps * self.n_part, self.d_proj).index_add_(0, bucket, px)
        return q_fde.view(B, self.fde_dim).half().contiguous()

    @torch.no_grad()
    def get_candidates(self, q_tokens_fp16, q_offsets, ncands=512):
        torch.cuda.set_device(self.device)
        B, _, _, q_pad, q_lens_gpu = _pad_queries(q_tokens_fp16, q_offsets, self.device)
        q_fde = self._encode_queries_fde(q_pad, q_lens_gpu)
        approx = q_fde @ self.d_fde.t()
        k_c = min(ncands, self.N_docs)
        _, cand_ids = torch.topk(approx, k=k_c, dim=1)
        return cand_ids.to(torch.int32).contiguous()

    @torch.no_grad()
    def search(self, q_tokens_fp16, q_offsets, ncands=512, topk=10):
        cand_ids = self.get_candidates(q_tokens_fp16, q_offsets, ncands=ncands)
        B, C = cand_ids.shape
        _, _, _, q_pad, q_lens_gpu = _pad_queries(q_tokens_fp16, q_offsets, self.device)
        sq = torch.arange(B, device=self.device, dtype=torch.int32).unsqueeze(1).repeat(1, C).reshape(-1).contiguous()
        sd = cand_ids.reshape(-1).contiguous()
        sc = self.ext.level2_exact_verify(sq, sd, q_pad, q_lens_gpu, self.d_tokens_gpu, self.d_offs_gpu).view(B, C)
        k_eff = min(topk, C)
        vals, pos = torch.topk(sc, k=k_eff, dim=1)
        top_docs = torch.gather(cand_ids.long(), 1, pos)
        out_ids = np.full((B, topk), -1, dtype=np.int64)
        out_sc = np.full((B, topk), -1e9, dtype=np.float32)
        out_ids[:, :k_eff] = top_docs.cpu().numpy()
        out_sc[:, :k_eff] = vals.cpu().numpy()
        return out_ids, out_sc


class B6CpuExactMaxSim:
    """B6: 48-thread C++ / OpenMP / AVX SIMD Exact MaxSim CPU baseline."""

    def __init__(self, d_tokens_fp16, d_offsets, so_path=None):
        so_path = so_path or os.path.expanduser("~/maxbound/src/baselines/libcpu_exact.so")
        self.lib = ctypes.CDLL(so_path)
        self.lib.cpu_exact_maxsim_topk.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        self.d_fp32 = np.ascontiguousarray(d_tokens_fp16.cpu().numpy().astype(np.float32))
        self.d_offs = np.ascontiguousarray(d_offsets, dtype=np.int64)
        self.N_docs = len(self.d_offs) - 1
        self.D = self.d_fp32.shape[1]

    def search(self, q_tokens_fp16, q_offsets, topk=10):
        q_fp32 = np.ascontiguousarray(q_tokens_fp16.cpu().numpy().astype(np.float32))
        q_offs = np.ascontiguousarray(q_offsets, dtype=np.int64)
        B = len(q_offs) - 1
        out_ids = np.empty((B, topk), dtype=np.int64)
        out_scores = np.empty((B, topk), dtype=np.float32)
        self.lib.cpu_exact_maxsim_topk(
            q_fp32.ctypes.data, q_offs.ctypes.data,
            self.d_fp32.ctypes.data, self.d_offs.ctypes.data,
            B, self.N_docs, self.D, topk,
            out_ids.ctypes.data, out_scores.ctypes.data
        )
        return out_ids, out_scores
