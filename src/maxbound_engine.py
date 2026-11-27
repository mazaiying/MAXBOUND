import os
import sys
import math
import time
import numpy as np
import torch

_EXT_CACHE = {}


def get_maxbound_ext(device_id: int):
    if device_id in _EXT_CACHE:
        return _EXT_CACHE[device_id]
    from torch.utils.cpp_extension import load
    props = torch.cuda.get_device_properties(device_id)
    arch = f"{props.major}{props.minor}"
    build_dir = os.path.expanduser(f"~/maxbound/build_sm{arch}")
    src_dir = os.path.expanduser("~/maxbound/src/cuda")
    os.makedirs(build_dir, exist_ok=True)
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{props.major}.{props.minor}"
    ext = load(
        name=f"maxbound_ext_{arch}",
        sources=[
            os.path.join(src_dir, "bindings.cpp"),
            os.path.join(src_dir, "maxbound_kernels.cu"),
        ],
        build_directory=build_dir,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
        verbose=False,
    )
    _EXT_CACHE[device_id] = ext
    return ext


def hadamard_matrix_torch(d: int, device):
    blk = min(d, 128)
    H = np.array([[1.0]], dtype=np.float32)
    for _ in range(int(round(math.log2(blk)))):
        H = np.block([[H, H], [H, -H]]) / np.float32(np.sqrt(2.0))
    if d == blk:
        return torch.from_numpy(H.astype(np.float32)).to(device)
    assert d % 128 == 0
    full = np.zeros((d, d), dtype=np.float32)
    for i in range(0, d, 128):
        full[i:i + 128, i:i + 128] = H
    return torch.from_numpy(full).to(device)


class MaxBoundIndex:
    """
    MAXBOUND GPU-Native Exact Multi-Vector Retrieval Engine:
      M1: Length-Sorted Skew-Coalesced Layout + Hadamard-Rotated Multi-Tier Quantized Storage
          (INT8 / Bit-Sliced INT6 / INT4) + 16-token aligned CSR + L0 Centroid/Radius + Host-Pinned Offload
      M2: PTX m16n8k32 Register-Resident Online MaxSim (Operator P1 at D=128; Operator P2-Fused
          Shared-Memory Multi-Query pass16 + 2-Slice L0 Early Pruning at D>=256..1024+)
          + Zero-Sync Low-Latency B=1..4 Fast Path + Vectorized Exact Verification + Mode B Rerank
      Dynamic Updates: Zero-Rebuild GPU Slab Insertion & O(1) Tombstone Mask Deletion
    """

    def __init__(self, d_tokens_fp16: torch.Tensor, d_offsets: np.ndarray, device: int = 1,
                 mode: str = "int8", host_offload_fp16: bool = False, sort_by_length: bool = True):
        self.device = device
        torch.cuda.set_device(device)
        self.ext = get_maxbound_ext(device)
        torch.cuda.synchronize(device)
        t_b0 = time.perf_counter()

        self.N_docs = len(d_offsets) - 1
        raw_offsets = np.asarray(d_offsets, dtype=np.int64)
        raw_lens = np.diff(raw_offsets)
        self.N_toks, self.D = d_tokens_fp16.shape
        self.mode = mode
        self.host_offload_fp16 = host_offload_fp16
        self.sort_by_length = bool(sort_by_length and self.N_docs > 1)
        assert self.D % 128 == 0, f"D={self.D} must be a multiple of 128"
        if mode in ("int6", "int4"):
            assert self.D == 128, f"mode={mode} requires D=128"

        if self.sort_by_length:
            self.perm_to_orig_np = np.argsort(raw_lens, kind="stable").astype(np.int64)
            self.orig_to_perm_np = np.empty(self.N_docs, dtype=np.int64)
            self.orig_to_perm_np[self.perm_to_orig_np] = np.arange(self.N_docs, dtype=np.int64)
            lens = raw_lens[self.perm_to_orig_np]
            self.d_offsets_np = np.zeros(self.N_docs + 1, dtype=np.int64)
            np.cumsum(lens, out=self.d_offsets_np[1:])
        else:
            self.perm_to_orig_np = None
            self.orig_to_perm_np = None
            lens = raw_lens
            self.d_offsets_np = raw_offsets.copy()

        # Pre-allocate GPU quantized index tensors FIRST so constrained-VRAM OOMs fail in <0.02s
        # before spending any CPU time sorting/gathering 130M-token host arrays!
        pad_lens = ((lens + 15) // 16) * 16
        self.d_pad_offs_np = np.zeros(self.N_docs + 1, dtype=np.int64)
        np.cumsum(pad_lens, out=self.d_pad_offs_np[1:])
        self.d_pad_offs_cpu = torch.from_numpy(self.d_pad_offs_np)
        self.d_pad_offs_gpu = torch.from_numpy(self.d_pad_offs_np).to(device=device, dtype=torch.int64)
        total_pad_toks = int(self.d_pad_offs_np[-1])

        self.d_scales = torch.zeros(total_pad_toks, dtype=torch.float32, device=device)
        self.d_rho = torch.zeros(total_pad_toks, dtype=torch.float32, device=device)
        self.d_qdata = None
        self.d_hi4 = None
        self.d_lo2 = None
        if mode == "int8":
            self.d_qdata = torch.zeros((total_pad_toks, self.D), dtype=torch.int8, device=device)
            qmax = 127.0
        elif mode == "int6":
            self.d_hi4 = torch.zeros((total_pad_toks, self.D // 2), dtype=torch.uint8, device=device)
            self.d_lo2 = torch.zeros((total_pad_toks, self.D // 4), dtype=torch.uint8, device=device)
            qmax = 31.0
        elif mode == "int4":
            self.d_qdata = torch.zeros((total_pad_toks, self.D // 2), dtype=torch.uint8, device=device)
            qmax = 7.0
        else:
            raise ValueError(f"Unknown mode {mode}")

        if not host_offload_fp16:
            # Also reserve GPU FP16 buffer upfront before CPU gather so OOM is immediate!
            self.d_tokens_fp16_gpu = torch.empty((self.N_toks, self.D), dtype=torch.float16, device=device)
            self.d_tokens_fp16_host = None

        if self.sort_by_length:
            self.perm_to_orig_gpu = torch.from_numpy(self.perm_to_orig_np).to(device=device, dtype=torch.int64)
            self.orig_to_perm_gpu = torch.from_numpy(self.orig_to_perm_np).to(device=device, dtype=torch.int64)
            src_starts = np.repeat(raw_offsets[:-1][self.perm_to_orig_np], lens)
            dst_starts = np.repeat(self.d_offsets_np[:-1], lens)
            tok_gather_idx = src_starts + (np.arange(self.N_toks, dtype=np.int64) - dst_starts)
            d_tokens_fp16 = torch.index_select(d_tokens_fp16, 0, torch.from_numpy(tok_gather_idx))
        else:
            self.perm_to_orig_gpu = None
            self.orig_to_perm_gpu = None

        self.d_offsets_gpu = torch.from_numpy(self.d_offsets_np).to(device=device, dtype=torch.int64)
        self.has_deletions = False
        self.doc_alive = torch.ones(self.N_docs, dtype=torch.uint8, device=device)

        max_dn = 0.0
        for s in range(0, self.N_toks, 262144):
            e = min(s + 262144, self.N_toks)
            dn = float(d_tokens_fp16[s:e].to(device=device, dtype=torch.float32).norm(dim=1).max().item())
            if dn > max_dn:
                max_dn = dn
        self.max_d_norm = max_dn + 1e-3

        if host_offload_fp16:
            # Avoid cudaHostRegister page-locking 31 GiB upfront when N_toks > 35M; pin only the tiny ~2MB survivor slice!
            if self.N_toks <= 35_000_000:
                self.d_tokens_fp16_host = d_tokens_fp16.contiguous().pin_memory()
            else:
                self.d_tokens_fp16_host = d_tokens_fp16.contiguous()
            self.d_tokens_fp16_gpu = None
        else:
            self.d_tokens_fp16_gpu.copy_(d_tokens_fp16)

        self.H = hadamard_matrix_torch(self.D, device)
        self.H_t = self.H.t().contiguous()
        self.mma_flat_idx_16 = self._build_mma_flat_idx_16(device) if self.D > 128 else None
        self.mma_flat_idx_32 = self._build_mma_flat_idx_32(device) if self.D > 128 else None

        # Build L2-normalized mean vector + exact L0 bounding ball radius per document (chunked doc_seg saves ~2 GiB VRAM!)
        doc_seg_np = np.repeat(np.arange(self.N_docs, dtype=np.int64), lens)
        d_means_fp32 = torch.zeros((self.N_docs, self.D), dtype=torch.float32, device=device)
        corpus_sum = torch.zeros(self.D, dtype=torch.float32, device=device)
        for s in range(0, self.N_toks, 262144):
            e = min(s + 262144, self.N_toks)
            xb = d_tokens_fp16[s:e].to(device=device, dtype=torch.float32)
            seg_s = torch.from_numpy(doc_seg_np[s:e]).to(device)
            d_means_fp32.index_add_(0, seg_s, xb)
            corpus_sum += xb.sum(dim=0)
        d_means_fp32.div_(d_means_fp32.norm(dim=1, keepdim=True).clamp_(min=1e-12))
        self.d_means = d_means_fp32.half().contiguous()
        self.d_means_t = self.d_means.t().contiguous()
        self.corpus_mean = torch.nn.functional.normalize(corpus_sum, p=2, dim=0)

        self.d_radii = torch.zeros(self.N_docs, dtype=torch.float32, device=device)
        for s in range(0, self.N_toks, 262144):
            e = min(s + 262144, self.N_toks)
            xb = d_tokens_fp16[s:e].to(device=device, dtype=torch.float32)
            seg_s = torch.from_numpy(doc_seg_np[s:e]).to(device)
            rad = (xb - d_means_fp32[seg_s]).norm(dim=1) + 1e-3
            self.d_radii.scatter_reduce_(0, seg_s, rad, reduce="amax", include_self=True)
        del d_means_fp32, doc_seg_np

        torch.cuda.synchronize(device)
        t_b1 = time.perf_counter()

        pad_starts = np.repeat(self.d_pad_offs_np[:-1], lens)
        orig_starts = np.repeat(self.d_offsets_np[:-1], lens)
        orig_to_pad = pad_starts + (np.arange(self.N_toks, dtype=np.int64) - orig_starts)

        for s in range(0, self.N_toks, 131072):
            e = min(s + 131072, self.N_toks)
            xb = d_tokens_fp16[s:e].to(device=device, dtype=torch.float32)
            hr = xb @ self.H_t
            sc = hr.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / qmax
            qi = torch.round(hr / sc).clamp(-qmax, qmax)
            hr_hat = qi * sc
            rho = (hr - hr_hat).norm(dim=1) + 5e-4
            pidx = torch.from_numpy(orig_to_pad[s:e]).to(device)
            self.d_scales[pidx] = sc.squeeze(1)
            self.d_rho[pidx] = rho
            if mode == "int8":
                self.d_qdata[pidx] = qi.to(torch.int8)
            elif mode == "int6":
                u6 = qi.to(torch.int32) + 32
                hi4 = (u6 >> 2).view(-1, self.D // 2, 2)
                lo2 = (u6 & 3).view(-1, self.D // 4, 4)
                self.d_hi4[pidx] = (hi4[:, :, 0] | (hi4[:, :, 1] << 4)).to(torch.uint8)
                self.d_lo2[pidx] = (lo2[:, :, 0] | (lo2[:, :, 1] << 2) | (lo2[:, :, 2] << 4) | (lo2[:, :, 3] << 6)).to(torch.uint8)
            else:
                u = (qi.to(torch.int32) + 8).view(-1, self.D // 2, 2)
                self.d_qdata[pidx] = (u[:, :, 0] | (u[:, :, 1] << 4)).to(torch.uint8)

        torch.cuda.synchronize(device)
        t_b2 = time.perf_counter()
        self.build_breakdown_s = {
            "sort_and_l0_meta_s": round(t_b1 - t_b0, 5),
            "hadamard_quant_pack_s": round(t_b2 - t_b1, 5),
            "total_build_s": round(t_b2 - t_b0, 5),
        }

        # Reusable low-latency GPU caches to eliminate per-query mallocs and host syncs
        self._qlens_cache = {}
        self._wq_cache = {}
        self._ws_cache = {}
        self._qprep_cache = {}
        self._zero_128 = torch.zeros(128, dtype=torch.float32, device=device)
        self.corpus_mean_fp16 = self.corpus_mean.half().contiguous()
        max_surv_cap = min(max(65536, self.N_docs * 16), 4_000_000) if not host_offload_fp16 else 0
        self._max_surv_ws = max_surv_cap
        self._ws_surv_q = torch.empty(max_surv_cap, dtype=torch.int32, device=device) if max_surv_cap > 0 else None
        self._ws_surv_d = torch.empty(max_surv_cap, dtype=torch.int32, device=device) if max_surv_cap > 0 else None
        self._d_num_surv = torch.zeros(1, dtype=torch.int32, device=device) if max_surv_cap > 0 else None
        self._sim_l2_buf = (
            torch.empty(32 * 1024 * 1024, dtype=torch.int32, device=device)
            if (self.D > 128 and mode == "int8") else None
        )

    def _invalidate_ws_cache(self):
        self._ws_cache.clear()
        self.d_means_t = self.d_means.t().contiguous()
        if not self.host_offload_fp16:
            max_surv_cap = min(max(65536, self.N_docs * 16), 4_000_000)
            if max_surv_cap > self._max_surv_ws:
                self._max_surv_ws = max_surv_cap
                self._ws_surv_q = torch.empty(max_surv_cap, dtype=torch.int32, device=self.device)
                self._ws_surv_d = torch.empty(max_surv_cap, dtype=torch.int32, device=self.device)

    def _get_workspace(self, B: int):
        ws = self._ws_cache.get((B, self.N_docs))
        if ws is None and B <= 64 and (B * self.N_docs * 9) <= (256 * 1024 * 1024):
            ub = torch.empty((B, self.N_docs), dtype=torch.float32, device=self.device)
            lb = torch.empty((B, self.N_docs), dtype=torch.float32, device=self.device)
            act = torch.empty((B, self.N_docs), dtype=torch.uint8, device=self.device)
            ws = (ub, lb, act)
            self._ws_cache[(B, self.N_docs)] = ws
        if ws is not None:
            ub, lb, act = ws
            ub.zero_()
            lb.zero_()
            if self.has_deletions:
                act.copy_(self.doc_alive.unsqueeze(0).expand(B, -1))
                lb.masked_fill_(self.doc_alive.unsqueeze(0) == 0, -1e9)
                ub.masked_fill_(self.doc_alive.unsqueeze(0) == 0, -1e9)
            else:
                act.fill_(1)
            return ub, lb, act
        ub = torch.zeros((B, self.N_docs), dtype=torch.float32, device=self.device)
        lb = torch.zeros((B, self.N_docs), dtype=torch.float32, device=self.device)
        if self.has_deletions:
            act = self.doc_alive.unsqueeze(0).expand(B, -1).contiguous().clone()
            lb.masked_fill_(self.doc_alive.unsqueeze(0) == 0, -1e9)
            ub.masked_fill_(self.doc_alive.unsqueeze(0) == 0, -1e9)
        else:
            act = torch.ones((B, self.N_docs), dtype=torch.uint8, device=self.device)
        return ub, lb, act

    @torch.no_grad()
    def delete_docs(self, orig_doc_ids):
        """O(1) GPU tombstone deletion without moving any vectors or rebuilding the index."""
        torch.cuda.set_device(self.device)
        ids = np.asarray(orig_doc_ids, dtype=np.int64)
        if len(ids) == 0:
            return
        internal_ids = self.orig_to_perm_np[ids] if self.sort_by_length else ids
        idx_gpu = torch.from_numpy(internal_ids).to(self.device)
        self.doc_alive[idx_gpu] = 0
        self.has_deletions = True

    @torch.no_grad()
    def insert_docs(self, new_tokens_fp16: torch.Tensor, new_offsets: np.ndarray):
        """Zero-rebuild incremental GPU append of new documents (preserving 100% bound exactness)."""
        torch.cuda.set_device(self.device)
        assert self.mode == "int8" and not self.host_offload_fp16
        new_offsets = np.asarray(new_offsets, dtype=np.int64)
        n_new = len(new_offsets) - 1
        if n_new <= 0:
            return
        new_lens = np.diff(new_offsets)
        n_new_toks = int(new_offsets[-1])

        xb = new_tokens_fp16.to(device=self.device, dtype=torch.float32)
        dn = float(xb.norm(dim=1).max().item())
        if dn + 1e-3 > self.max_d_norm:
            self.max_d_norm = dn + 1e-3

        doc_seg = torch.from_numpy(np.repeat(np.arange(n_new, dtype=np.int64), new_lens)).to(self.device)
        m_fp32 = torch.zeros((n_new, self.D), dtype=torch.float32, device=self.device)
        m_fp32.index_add_(0, doc_seg, xb)
        m_norm = torch.nn.functional.normalize(m_fp32, p=2, dim=1)
        rad_new = torch.zeros(n_new, dtype=torch.float32, device=self.device)
        rad_tok = (xb - m_norm[doc_seg]).norm(dim=1) + 1e-3
        rad_new.scatter_reduce_(0, doc_seg, rad_tok, reduce="amax", include_self=True)

        pad_lens = ((new_lens + 15) // 16) * 16
        new_pad_offs = np.zeros(n_new + 1, dtype=np.int64)
        np.cumsum(pad_lens, out=new_pad_offs[1:])
        tot_pad_new = int(new_pad_offs[-1])

        qdata_new = torch.zeros((tot_pad_new, self.D), dtype=torch.int8, device=self.device)
        scales_new = torch.zeros(tot_pad_new, dtype=torch.float32, device=self.device)
        rho_new = torch.zeros(tot_pad_new, dtype=torch.float32, device=self.device)

        pad_starts = np.repeat(new_pad_offs[:-1], new_lens)
        orig_starts = np.repeat(new_offsets[:-1], new_lens)
        pidx = torch.from_numpy(pad_starts + (np.arange(n_new_toks, dtype=np.int64) - orig_starts)).to(self.device)

        hr = xb @ self.H_t
        sc = hr.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 127.0
        qi = torch.round(hr / sc).clamp(-127.0, 127.0)
        rho = (hr - qi * sc).norm(dim=1) + 5e-4
        qdata_new[pidx] = qi.to(torch.int8)
        scales_new[pidx] = sc.squeeze(1)
        rho_new[pidx] = rho

        old_n_docs = self.N_docs
        old_n_toks = self.N_toks
        old_pad_toks = int(self.d_pad_offs_np[-1])

        self.d_tokens_fp16_gpu = torch.cat([self.d_tokens_fp16_gpu, xb.half().contiguous()], dim=0)
        self.d_means = torch.cat([self.d_means, m_norm.half().contiguous()], dim=0)
        self.d_radii = torch.cat([self.d_radii, rad_new], dim=0)
        self.d_qdata = torch.cat([self.d_qdata, qdata_new], dim=0)
        self.d_scales = torch.cat([self.d_scales, scales_new], dim=0)
        self.d_rho = torch.cat([self.d_rho, rho_new], dim=0)
        self.doc_alive = torch.cat([self.doc_alive, torch.ones(n_new, dtype=torch.uint8, device=self.device)], dim=0)

        self.d_offsets_np = np.concatenate([self.d_offsets_np, new_offsets[1:] + old_n_toks])
        self.d_offsets_gpu = torch.from_numpy(self.d_offsets_np).to(device=self.device, dtype=torch.int64)
        self.d_pad_offs_np = np.concatenate([self.d_pad_offs_np, new_pad_offs[1:] + old_pad_toks])
        self.d_pad_offs_cpu = torch.from_numpy(self.d_pad_offs_np)
        self.d_pad_offs_gpu = torch.from_numpy(self.d_pad_offs_np).to(device=self.device, dtype=torch.int64)

        if self.sort_by_length:
            new_orig_ids = np.arange(old_n_docs, old_n_docs + n_new, dtype=np.int64)
            self.perm_to_orig_np = np.concatenate([self.perm_to_orig_np, new_orig_ids])
            self.orig_to_perm_np = np.concatenate([self.orig_to_perm_np, np.arange(old_n_docs, old_n_docs + n_new, dtype=np.int64)])
            self.perm_to_orig_gpu = torch.from_numpy(self.perm_to_orig_np).to(device=self.device, dtype=torch.int64)
            self.orig_to_perm_gpu = torch.from_numpy(self.orig_to_perm_np).to(device=self.device, dtype=torch.int64)

        self.N_docs += n_new
        self.N_toks += n_new_toks
        self._invalidate_ws_cache()

    @staticmethod
    def _build_mma_flat_idx_16(device):
        row_offs = (0, 8, 0, 8)
        col_offs = (0, 0, 4, 4)
        idx = np.empty(512, dtype=np.int64)
        for r in range(16):
            ks = r >> 2
            sub = r & 3
            for lane in range(32):
                group_id = lane >> 2
                tid_in_g = lane & 3
                row = group_id + row_offs[sub]
                col = ks * 8 + tid_in_g + col_offs[sub]
                idx[r * 32 + lane] = row * 32 + col
        return torch.from_numpy(idx).to(device=device, dtype=torch.long)

    def _pack_q_mma_16(self, q_int8: torch.Tensor) -> torch.Tensor:
        B, m_pad, D = q_int8.shape
        n_slices = m_pad >> 4
        n_dblks = D >> 7
        q_i32 = (
            q_int8.contiguous()
            .view(torch.int32)
            .reshape(B, n_slices, 16, n_dblks, 32)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, n_slices, n_dblks, 512)
        )
        return (
            torch.index_select(q_i32, 3, self.mma_flat_idx_16)
            .contiguous()
            .view(torch.int8)
            .reshape(B, m_pad, D)
        )

    def _compute_l0_slice_bound(self, q_fp16_slice: torch.Tensor, q_norms_slice: torch.Tensor) -> torch.Tensor:
        B, slen, D = q_fp16_slice.shape
        out = torch.empty((B, self.N_docs), dtype=torch.float32, device=self.device)
        q_flat = q_fp16_slice.contiguous().view(B * slen, D)
        qn_cap = (q_norms_slice * self.max_d_norm).unsqueeze(2)
        qn_exp = q_norms_slice.unsqueeze(2)
        for dc in range(0, self.N_docs, 16384):
            de = min(dc + 16384, self.N_docs)
            dots = torch.matmul(q_flat, self.d_means[dc:de].t()).float().view(B, slen, de - dc)
            l0 = torch.minimum(qn_cap, dots + qn_exp * self.d_radii[dc:de].view(1, 1, -1))
            out[:, dc:de] = l0.sum(dim=1)
        return out

    @staticmethod
    def _build_mma_flat_idx_32(device):
        row_offs = (0, 8, 0, 8, 16, 24, 16, 24)
        col_offs = (0, 0, 4, 4, 0, 0, 4, 4)
        idx = np.empty(1024, dtype=np.int64)
        for r in range(32):
            ks = r >> 3
            s = r & 7
            for lane in range(32):
                group_id = lane >> 2
                tid_in_g = lane & 3
                row = group_id + row_offs[s]
                col = ks * 8 + tid_in_g + col_offs[s]
                idx[r * 32 + lane] = row * 32 + col
        return torch.from_numpy(idx).to(device=device, dtype=torch.long)

    def _pack_q_mma_32(self, q_int8: torch.Tensor) -> torch.Tensor:
        B, m_pad, D = q_int8.shape
        n_passes = m_pad >> 5
        n_dblks = D >> 7
        q_i32 = (
            q_int8.contiguous()
            .view(torch.int32)
            .reshape(B, n_passes, 32, n_dblks, 32)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, n_passes, n_dblks, 1024)
        )
        return (
            torch.index_select(q_i32, 3, self.mma_flat_idx_32)
            .contiguous()
            .view(torch.int8)
            .reshape(B, m_pad, D)
        )

    def _exec_int8_pass32(self, q_int8: torch.Tensor, q_scales: torch.Tensor, q_norms: torch.Tensor,
                          tau_buf: torch.Tensor, ub_accum: torch.Tensor, lb_accum: torch.Tensor,
                          rem: torch.Tensor, doc_active: torch.Tensor, q_tok_off: int,
                          eps_safe_scalar: float, is_first: int, do_prune: int, force_w1: bool = False,
                          q_fp16_pad: torch.Tensor = None, enable_2slice: bool = False,
                          eps_buf: torch.Tensor = None, is_packed_hd: bool = False):
        if self.D == 128:
            self.ext.fused_tc_int8_pass32(
                q_int8, q_scales, q_norms,
                self.d_qdata, self.d_scales, self.d_rho,
                self.d_pad_offs_gpu, tau_buf, ub_accum, lb_accum, rem, doc_active,
                q_tok_off, eps_safe_scalar, is_first, do_prune, 1 if force_w1 else 0, eps_buf, 0
            )
            return

        # Operator P2 (D >= 256):
        #   - For B >= 4 (not force_w1): Hardware Persisting-L2 cuBLASLt INT8 Tensor-Core IMMA (624 TOPS!)
        #     + 2-Slice (16+16) L0 Early Pruning + Sparse PTX Rerank on Surviving Pairs!
        #   - For B < 4 or force_w1: Register/SMEM PTX m16n8k32 pass16!
        B = q_int8.size(0)
        has_tail16 = (q_fp16_pad is None) or (q_fp16_pad.size(1) > q_tok_off + 16)

        if B >= 4 and (not force_w1) and (self._sim_l2_buf is not None) and (not is_packed_hd):
            do_2slice = bool(enable_2slice and (q_fp16_pad is not None) and has_tail16 and (self.N_docs >= 256))
            if not has_tail16:
                q_sl = q_int8[:, q_tok_off:q_tok_off + 16, :].contiguous().view(B * 16, self.D)
                t_toks = max(8192, min(65536, int(self._sim_l2_buf.numel() // max(B * 16, 1))))
                self.ext.exec_hd_l2_int8_pass(
                    q_sl, self.d_qdata, self.d_scales, self.d_rho,
                    self.d_pad_offs_gpu, self.d_pad_offs_cpu, tau_buf,
                    ub_accum, lb_accum, rem, doc_active, q_scales, q_norms,
                    self._sim_l2_buf, t_toks, q_tok_off, eps_safe_scalar,
                    is_first, do_prune, 1, eps_buf
                )
                return

            if not do_2slice:
                q_sl32 = q_int8[:, q_tok_off:q_tok_off + 32, :].contiguous().view(B * 32, self.D)
                t_toks = max(8192, min(32768, int(self._sim_l2_buf.numel() // max(B * 32, 1))))
                self.ext.exec_hd_l2_int8_pass(
                    q_sl32, self.d_qdata, self.d_scales, self.d_rho,
                    self.d_pad_offs_gpu, self.d_pad_offs_cpu, tau_buf,
                    ub_accum, lb_accum, rem, doc_active, q_scales, q_norms,
                    self._sim_l2_buf, t_toks, q_tok_off, eps_safe_scalar,
                    is_first, do_prune, 1, eps_buf
                )
                return

            # Stage 1A (Slice 0: first 16 query tokens via 624-TOPS Persisting-L2 INT8 IMMA + L0 tail bound!)
            s1_end = min(q_tok_off + 32, q_fp16_pad.size(1))
            l0_rem16 = self._compute_l0_slice_bound(
                q_fp16_pad[:, q_tok_off + 16:s1_end, :],
                q_norms[:, q_tok_off + 16:s1_end]
            )
            if do_prune:
                l0_rem16 = l0_rem16 + rem.unsqueeze(1)
            q_sl0 = q_int8[:, q_tok_off:q_tok_off + 16, :].contiguous().view(B * 16, self.D)
            t_toks = max(8192, min(65536, int(self._sim_l2_buf.numel() // max(B * 16, 1))))
            self.ext.exec_hd_l2_int8_pass(
                q_sl0, self.d_qdata, self.d_scales, self.d_rho,
                self.d_pad_offs_gpu, self.d_pad_offs_cpu, tau_buf,
                ub_accum, lb_accum, l0_rem16, doc_active, q_scales, q_norms,
                self._sim_l2_buf, t_toks, q_tok_off, eps_safe_scalar,
                is_first, 1, 1, eps_buf
            )

            # Stage 1B (Slice 1: remaining 16 query tokens via Sparse PTX Rerank when pruned >= 65%!)
            act_bool = (doc_active > 0)
            counts = act_bool.sum(dim=1, dtype=torch.int64)
            max_c = int(counts.max().item())
            if max_c == 0:
                return
            if max_c <= max(256, self.N_docs // 3):
                nz = torch.nonzero(act_bool, as_tuple=False)
                row_starts = torch.cumsum(counts, dim=0) - counts
                q_idx = nz[:, 0]
                d_idx = nz[:, 1]
                col_idx = torch.arange(nz.size(0), device=self.device, dtype=torch.int64) - row_starts[q_idx]
                cand_ids = torch.full((B, max_c), -1, dtype=torch.int32, device=self.device)
                cand_ids[q_idx, col_idx] = d_idx.to(torch.int32)

                q_i8_tail = torch.zeros((B, 32, self.D), dtype=torch.int8, device=self.device)
                q_sc_tail = torch.zeros((B, 32), dtype=torch.float32, device=self.device)
                q_nm_tail = torch.zeros((B, 32), dtype=torch.float32, device=self.device)
                q_i8_tail[:, :16, :] = q_int8[:, q_tok_off + 16:q_tok_off + 32, :]
                q_sc_tail[:, :16] = q_scales[:, q_tok_off + 16:q_tok_off + 32]
                q_nm_tail[:, :16] = q_norms[:, q_tok_off + 16:q_tok_off + 32]

                ub_r = torch.zeros((B, max_c), dtype=torch.float32, device=self.device)
                lb_r = torch.zeros((B, max_c), dtype=torch.float32, device=self.device)
                self.ext.fused_tc_rerank_int8(
                    q_i8_tail, q_sc_tail, q_nm_tail,
                    self.d_qdata, self.d_scales, self.d_rho,
                    self.d_pad_offs_gpu, cand_ids, ub_r, lb_r, 0, 1
                )
                ub_new = ub_accum[q_idx, d_idx] + ub_r[q_idx, col_idx]
                lb_new = lb_accum[q_idx, d_idx] + lb_r[q_idx, col_idx]
                ub_accum[q_idx, d_idx] = ub_new
                lb_accum[q_idx, d_idx] = lb_new
                if do_prune:
                    doc_active[q_idx, d_idx] = (ub_new + rem[q_idx] >= tau_buf[q_idx] - eps_buf[q_idx]).to(torch.uint8)
            else:
                q_sl1 = q_int8[:, q_tok_off + 16:q_tok_off + 32, :].contiguous().view(B * 16, self.D)
                self.ext.exec_hd_l2_int8_pass(
                    q_sl1, self.d_qdata, self.d_scales, self.d_rho,
                    self.d_pad_offs_gpu, self.d_pad_offs_cpu, tau_buf,
                    ub_accum, lb_accum, rem, doc_active, q_scales, q_norms,
                    self._sim_l2_buf, t_toks, q_tok_off + 16, eps_safe_scalar,
                    0, do_prune, 1, eps_buf
                )
            return

        # Small-batch (B < 4) or force_w1: Shared-Memory PTX m16n8k32 pass16
        q_pack16 = q_int8 if is_packed_hd else self._pack_q_mma_16(q_int8)
        fw1_flag = 1 if force_w1 else 0
        do_2slice = bool(enable_2slice and (q_fp16_pad is not None) and has_tail16 and (self.N_docs >= 128) and (not force_w1))

        if not has_tail16:
            self.ext.fused_tc_int8_hd_pass16(
                q_pack16, q_scales, q_norms,
                self.d_qdata, self.d_scales, self.d_rho,
                self.d_pad_offs_gpu, tau_buf, ub_accum, lb_accum, rem, doc_active,
                q_tok_off, eps_safe_scalar, is_first, do_prune, fw1_flag, eps_buf
            )
            return

        if not do_2slice:
            self.ext.fused_tc_int8_hd_pass16(
                q_pack16, q_scales, q_norms,
                self.d_qdata, self.d_scales, self.d_rho,
                self.d_pad_offs_gpu, tau_buf, ub_accum, lb_accum, rem, doc_active,
                q_tok_off, eps_safe_scalar, is_first, 0, fw1_flag, eps_buf
            )
            self.ext.fused_tc_int8_hd_pass16(
                q_pack16, q_scales, q_norms,
                self.d_qdata, self.d_scales, self.d_rho,
                self.d_pad_offs_gpu, tau_buf, ub_accum, lb_accum, rem, doc_active,
                q_tok_off + 16, eps_safe_scalar, 0, do_prune, fw1_flag, eps_buf
            )
            return

        s1_end = min(q_tok_off + 32, q_fp16_pad.size(1))
        l0_rem16 = self._compute_l0_slice_bound(
            q_fp16_pad[:, q_tok_off + 16:s1_end, :],
            q_norms[:, q_tok_off + 16:s1_end]
        )
        if do_prune:
            l0_rem16 = l0_rem16 + rem.unsqueeze(1)
        self.ext.fused_tc_int8_hd_pass16(
            q_pack16, q_scales, q_norms,
            self.d_qdata, self.d_scales, self.d_rho,
            self.d_pad_offs_gpu, tau_buf, ub_accum, lb_accum, l0_rem16, doc_active,
            q_tok_off, eps_safe_scalar, is_first, 1, fw1_flag, eps_buf
        )
        self.ext.fused_tc_int8_hd_pass16(
            q_pack16, q_scales, q_norms,
            self.d_qdata, self.d_scales, self.d_rho,
            self.d_pad_offs_gpu, tau_buf, ub_accum, lb_accum, rem, doc_active,
            q_tok_off + 16, eps_safe_scalar, 0, do_prune, fw1_flag, eps_buf
        )

    def _prepare_queries(self, q_tokens_fp16: torch.Tensor, q_offsets: np.ndarray, pad_mult: int = 32, for_rerank: bool = False):
        B = len(q_offsets) - 1
        total_qt = int(q_offsets[-1])
        if B == 1:
            m_max = total_qt
            uniform = True
        else:
            q_lens_np = np.diff(q_offsets).astype(np.int32)
            m_max = int(q_lens_np.max())
            uniform = bool(total_qt == B * m_max and np.all(q_lens_np == m_max))

        m_pad = ((m_max + pad_mult - 1) // pad_mult) * pad_mult

        if self.D <= 1024:
            if uniform:
                q_lens_gpu = self._qlens_cache.get((B, m_max))
                if q_lens_gpu is None:
                    q_lens_gpu = torch.full((B,), m_max, dtype=torch.int32, device=self.device)
                    if B <= 128:
                        self._qlens_cache[(B, m_max)] = q_lens_gpu
                if q_tokens_fp16.is_cuda and q_tokens_fp16.device.index == self.device and q_tokens_fp16.dtype == torch.float16:
                    q_fp16_pad = q_tokens_fp16.view(B, m_max, self.D).contiguous()
                else:
                    q_fp16_pad = q_tokens_fp16.to(device=self.device, dtype=torch.float16).view(B, m_max, self.D).contiguous()
            else:
                q_lens_gpu = torch.from_numpy(q_lens_np).to(self.device)
                q_fp16_gpu = q_tokens_fp16.to(device=self.device, dtype=torch.float16)
                q_fp16_pad = torch.zeros((B, m_max, self.D), dtype=torch.float16, device=self.device)
                for qi in range(B):
                    qs, qe = int(q_offsets[qi]), int(q_offsets[qi + 1])
                    q_fp16_pad[qi, :qe - qs] = q_fp16_gpu[qs:qe]

            if (not for_rerank) and (m_pad > 32 or (self.D > 128 and m_max >= 20)) and uniform:
                sim_c = torch.matmul(q_fp16_pad, self.corpus_mean_fp16)
                order = torch.argsort(sim_c, dim=1)
                q_fp16_pad = torch.gather(q_fp16_pad, 1, order.unsqueeze(2).expand(-1, -1, self.D)).contiguous()

            qp_bufs = self._qprep_cache.get((B, m_pad))
            if qp_bufs is None:
                q_int8 = torch.empty((B, m_pad, self.D), dtype=torch.int8, device=self.device)
                q_scales = torch.empty((B, m_pad), dtype=torch.float32, device=self.device)
                q_norms = torch.empty((B, m_pad), dtype=torch.float32, device=self.device)
                eps_buf = torch.empty((B,), dtype=torch.float32, device=self.device)
                if B <= 64 and m_pad <= 128:
                    self._qprep_cache[(B, m_pad)] = (q_int8, q_scales, q_norms, eps_buf)
            else:
                q_int8, q_scales, q_norms, eps_buf = qp_bufs

            self.ext.prepare_queries_hadamard_int8(
                q_fp16_pad, q_lens_gpu, q_int8, q_scales, q_norms, eps_buf, self.max_d_norm
            )
            if (not for_rerank) and self.D > 128 and self.mode == "int8" and B < 4:
                q_int8_out = self._pack_q_mma_16(q_int8)
            else:
                q_int8_out = q_int8
            return B, m_max, m_pad, q_lens_gpu, q_fp16_pad, q_int8_out, q_scales, q_norms, eps_buf

        if uniform:
            q_lens_gpu = self._qlens_cache.get((B, m_max))
            if q_lens_gpu is None:
                q_lens_gpu = torch.full((B,), m_max, dtype=torch.int32, device=self.device)
                if B <= 128:
                    self._qlens_cache[(B, m_max)] = q_lens_gpu
            q_fp16_gpu = q_tokens_fp16.to(device=self.device, dtype=torch.float16)
            qt_all = q_fp16_gpu.float()
            q_fp16_pad = q_fp16_gpu.view(B, m_max, self.D).contiguous()
            qt_rot_all = qt_all @ self.H_t
            qt_norm_all = qt_all.norm(dim=1)
            if m_pad == m_max:
                q_rot = qt_rot_all.view(B, m_pad, self.D)
                q_norms = qt_norm_all.view(B, m_pad)
            else:
                q_rot = torch.zeros((B, m_pad, self.D), dtype=torch.float32, device=self.device)
                q_norms = torch.zeros((B, m_pad), dtype=torch.float32, device=self.device)
                q_rot[:, :m_max] = qt_rot_all.view(B, m_max, self.D)
                q_norms[:, :m_max] = qt_norm_all.view(B, m_max)
        else:
            q_lens_gpu = torch.from_numpy(q_lens_np).to(self.device)
            q_fp16_gpu = q_tokens_fp16.to(device=self.device, dtype=torch.float16)
            qt_all = q_fp16_gpu.float()
            qt_rot_all = qt_all @ self.H_t
            qt_norm_all = qt_all.norm(dim=1)
            q_fp16_pad = torch.zeros((B, m_max, self.D), dtype=torch.float16, device=self.device)
            q_rot = torch.zeros((B, m_pad, self.D), dtype=torch.float32, device=self.device)
            q_norms = torch.zeros((B, m_pad), dtype=torch.float32, device=self.device)
            for qi in range(B):
                qs, qe = int(q_offsets[qi]), int(q_offsets[qi + 1])
                mq = qe - qs
                q_fp16_pad[qi, :mq] = q_fp16_gpu[qs:qe]
                q_rot[qi, :mq] = qt_rot_all[qs:qe]
                q_norms[qi, :mq] = qt_norm_all[qs:qe]

        if (not for_rerank) and (m_pad > 32 or (self.D > 128 and m_max >= 20)) and uniform:
            sim_c = torch.matmul(qt_all.view(B, m_max, self.D), self.corpus_mean)
            order = torch.argsort(sim_c, dim=1)
            order_exp = order.unsqueeze(2).expand(-1, -1, self.D)
            q_fp16_pad = torch.gather(q_fp16_pad, 1, order_exp).contiguous()
            q_rot[:, :m_max] = torch.gather(q_rot[:, :m_max], 1, order_exp)
            q_norms[:, :m_max] = torch.gather(q_norms[:, :m_max], 1, order)

        q_sc = q_rot.abs().amax(dim=2, keepdim=True).clamp(min=1e-6) / 127.0
        q_int8 = torch.round(q_rot / q_sc).clamp(-127, 127).to(torch.int8).contiguous()
        q_scales = q_sc.squeeze(2).contiguous()
        q_err = (q_rot - q_int8.float() * q_sc).norm(dim=2)
        eps_buf = (q_err.sum(dim=1) * self.max_d_norm + 2e-3 * q_norms.sum(dim=1) + 1e-3).contiguous()
        if (not for_rerank) and self.D > 128 and self.mode == "int8":
            q_int8 = self._pack_q_mma_16(q_int8)
        return B, m_max, m_pad, q_lens_gpu, q_fp16_pad, q_int8, q_scales, q_norms, eps_buf

    def _stage0_prewarm(self, q_fp16_pad, q_lens_gpu, B, topk, c_warm):
        q_means = torch.nn.functional.normalize(q_fp16_pad.float().sum(dim=1), p=2, dim=1).half()
        approx_s0 = torch.matmul(q_means, self.d_means_t)
        if self.has_deletions:
            approx_s0 = approx_s0.float().masked_fill(self.doc_alive.unsqueeze(0) == 0, -1e9)
            n_avail = int(self.doc_alive.sum().item())
        else:
            n_avail = self.N_docs
        c_w = min(max(c_warm, topk), max(n_avail, 1))
        _, warm_idx = torch.topk(approx_s0, k=c_w, dim=1)
        wq = self._wq_cache.get((B, c_w))
        if wq is None:
            wq = torch.arange(B, device=self.device, dtype=torch.int32).unsqueeze(1).repeat(1, c_w).reshape(-1).contiguous()
            if B <= 128:
                self._wq_cache[(B, c_w)] = wq
        wd = warm_idx.to(torch.int32).reshape(-1).contiguous()
        warm_scores = self._verify_pairs(wq, wd, q_fp16_pad, q_lens_gpu).view(B, c_w)
        k_eff = min(topk, max(n_avail, 1))
        warm_topk_scores = torch.topk(warm_scores, k=min(k_eff, c_w), dim=1).values
        tau_init = warm_topk_scores[:, -1].contiguous()
        return warm_idx, warm_scores, tau_init

    @torch.no_grad()
    def search(self, q_tokens_fp16: torch.Tensor, q_offsets: np.ndarray, topk: int = 10,
               c_warm: int = 16, external_tau: torch.Tensor = None, prewarm_tuple=None,
               prepared_tuple=None, force_w1: bool = False, disable_prewarm: bool = False,
               disable_multipass: bool = False, disable_lb_tighten: bool = False,
               return_tested_mask: bool = False):
        torch.cuda.set_device(self.device)
        self._last_h2d_bytes = 0
        if prepared_tuple is None:
            B, m_max, m_pad, q_lens_gpu, q_fp16_pad, q_int8_pack, q_scales, q_norms, eps_buf = \
                self._prepare_queries(q_tokens_fp16, q_offsets, pad_mult=32, for_rerank=False)
        else:
            B, m_max, m_pad, q_lens_gpu, q_fp16_pad, q_int8_pack, q_scales, q_norms, eps_buf = prepared_tuple

        n_avail = int(self.doc_alive.sum().item()) if self.has_deletions else self.N_docs
        k_eff = min(topk, max(n_avail, 1))

        if disable_prewarm:
            warm_idx = None
            warm_scores = None
            tau_buf = torch.full((B,), -1e9, dtype=torch.float32, device=self.device)
        elif prewarm_tuple is None:
            warm_idx, warm_scores, tau_buf = self._stage0_prewarm(q_fp16_pad, q_lens_gpu, B, topk, c_warm)
        else:
            warm_idx, warm_scores, tau_buf = prewarm_tuple
            tau_buf = tau_buf.clone()

        if external_tau is not None:
            tau_buf = torch.maximum(tau_buf, external_tau.to(self.device))

        ub_accum, lb_accum, doc_active = self._get_workspace(B)
        if warm_idx is not None:
            doc_active.scatter_(1, warm_idx, 0)

        n_passes = m_pad // 32
        if n_passes > 1:
            pass_norm_sums = torch.stack([
                q_norms[:, p * 32:(p + 1) * 32].sum(dim=1) * self.max_d_norm for p in range(n_passes)
            ], dim=0)
        else:
            pass_norm_sums = None

        # All modes (INT8, INT6, INT4) now consume per-query GPU eps_buf directly with ZERO host sync!
        eps_safe_scalar = 0.0
        is_packed_hd = bool(self.D > 128 and self.mode == "int8" and B < 4 and (not force_w1))

        for p in range(n_passes):
            if n_passes > 1 and p + 1 < n_passes:
                rem = pass_norm_sums[p + 1:].sum(dim=0)
            else:
                rem = self._zero_128[:B] if B <= 128 else torch.zeros(B, dtype=torch.float32, device=self.device)
            do_prune = 1 if (n_passes > 1 and not disable_multipass and not disable_prewarm) else 0
            is_first = 1 if (p == 0) else 0
            if self.mode == "int8":
                self._exec_int8_pass32(
                    q_int8_pack, q_scales, q_norms, tau_buf, ub_accum, lb_accum, rem, doc_active,
                    p * 32, eps_safe_scalar, is_first, do_prune, force_w1=force_w1,
                    q_fp16_pad=q_fp16_pad, enable_2slice=(not disable_multipass and not disable_prewarm),
                    eps_buf=eps_buf, is_packed_hd=is_packed_hd
                )
            elif self.mode == "int6":
                self.ext.fused_tc_int6_pass32(
                    q_int8_pack, q_scales, q_norms,
                    self.d_hi4, self.d_lo2, self.d_scales, self.d_rho,
                    self.d_pad_offs_gpu, tau_buf, ub_accum, lb_accum, rem, doc_active,
                    p * 32, eps_safe_scalar, is_first, do_prune, eps_buf
                )
            else:
                self.ext.fused_tc_int4_pass32(
                    q_int8_pack, q_scales, q_norms,
                    self.d_qdata, self.d_scales, self.d_rho,
                    self.d_pad_offs_gpu, tau_buf, ub_accum, lb_accum, rem, doc_active,
                    p * 32, eps_safe_scalar, is_first, do_prune, eps_buf
                )
            if p == n_passes - 1 and not disable_lb_tighten:
                lb_accum.masked_fill_(doc_active == 0, -1e9)
                if self.has_deletions:
                    lb_accum.masked_fill_(self.doc_alive.unsqueeze(0) == 0, -1e9)
                if warm_idx is not None:
                    lb_for_tau = lb_accum.scatter(1, warm_idx, warm_scores)
                else:
                    lb_for_tau = lb_accum
                lb_kth = torch.topk(lb_for_tau, k=k_eff, dim=1).values[:, -1]
                tau_buf = torch.maximum(tau_buf, lb_kth)

        scores_mat = ub_accum
        tested_mask_orig = None

        if (not self.host_offload_fp16) and (not return_tested_mask) and (self._ws_surv_q is not None):
            # Zero-host-sync Stage-2: GPU ballot compaction + scatter exact verification directly into scores_mat!
            ub_backup = ub_accum.clone() if (B * self.N_docs > self._max_surv_ws) else None
            self.ext.fused_stage2_verify(
                doc_active, scores_mat, tau_buf, eps_buf,
                warm_idx.contiguous() if warm_idx is not None else None,
                warm_scores.contiguous() if warm_scores is not None else None,
                self._ws_surv_q, self._ws_surv_d, self._d_num_surv,
                q_fp16_pad, q_lens_gpu, self.d_tokens_fp16_gpu, self.d_offsets_gpu
            )
            num_surv = int(self._d_num_surv.item())
            if num_surv > self._max_surv_ws and ub_backup is not None:
                surv_mask = (doc_active > 0) & (ub_backup >= (tau_buf[:, None] - eps_buf[:, None]))
                scores_mat.fill_(-1e9)
                if warm_idx is not None:
                    scores_mat.scatter_(1, warm_idx, warm_scores)
                nz = torch.nonzero(surv_mask, as_tuple=False)
                num_surv = int(nz.size(0))
                if num_surv > 0:
                    surv_q = nz[:, 0].to(torch.int32).contiguous()
                    surv_doc = nz[:, 1].to(torch.int32).contiguous()
                    scores_mat[surv_q.long(), surv_doc.long()] = self._verify_pairs(surv_q, surv_doc, q_fp16_pad, q_lens_gpu)
        else:
            surv_mask = (doc_active > 0) & (ub_accum >= (tau_buf[:, None] - eps_buf[:, None]))
            if return_tested_mask:
                tested_internal = surv_mask.clone()
                if warm_idx is not None:
                    tested_internal.scatter_(1, warm_idx, True)
                if self.sort_by_length:
                    tested_mask_orig = tested_internal[:, self.orig_to_perm_gpu].cpu().numpy()
                else:
                    tested_mask_orig = tested_internal.cpu().numpy()
            scores_mat.fill_(-1e9)
            if warm_idx is not None:
                scores_mat.scatter_(1, warm_idx, warm_scores)
            nz = torch.nonzero(surv_mask, as_tuple=False)
            num_surv = int(nz.size(0))
            if num_surv > 0:
                surv_q = nz[:, 0].to(torch.int32).contiguous()
                surv_doc = nz[:, 1].to(torch.int32).contiguous()
                scores_mat[surv_q.long(), surv_doc.long()] = self._verify_pairs(surv_q, surv_doc, q_fp16_pad, q_lens_gpu)

        top_vals, top_idx = torch.topk(scores_mat, k=k_eff, dim=1)
        if self.sort_by_length:
            top_idx = self.perm_to_orig_gpu[top_idx]

        top_idx_np = top_idx.cpu().numpy()
        top_vals_np = top_vals.cpu().numpy()
        if k_eff == topk:
            all_ids = top_idx_np
            all_scores = top_vals_np
        else:
            all_ids = np.full((B, topk), -1, dtype=np.int64)
            all_scores = np.full((B, topk), -1e9, dtype=np.float32)
            all_ids[:, :k_eff] = top_idx_np
            all_scores[:, :k_eff] = top_vals_np

        info = {
            "num_survivors": num_surv,
            "survivor_ratio": float(num_surv / max(B * n_avail, 1)),
            "tau_buf": tau_buf,
            "h2d_mb": float(self._last_h2d_bytes / (1024.0 * 1024.0)),
        }
        if tested_mask_orig is not None:
            info["tested_mask_orig"] = tested_mask_orig
        return all_ids, all_scores, info

    @torch.no_grad()
    def profile_slice_pruning(self, q_tokens_fp16: torch.Tensor, q_offsets: np.ndarray,
                              topk: int = 10, c_warm: int = 32):
        torch.cuda.set_device(self.device)
        B = len(q_offsets) - 1
        q_lens = np.diff(q_offsets).astype(np.int32)
        m_max = max(4, int(q_lens.max()))
        q_lens_gpu = torch.from_numpy(q_lens).to(self.device)
        qt_flat = q_tokens_fp16.to(device=self.device, dtype=torch.float32)
        if np.all(q_lens == m_max):
            qt = qt_flat.view(B, m_max, self.D)
        else:
            qt = torch.zeros((B, m_max, self.D), dtype=torch.float32, device=self.device)
            for qi in range(B):
                qs, qe = int(q_offsets[qi]), int(q_offsets[qi + 1])
                qt[qi, :qe - qs] = qt_flat[qs:qe]
        valid_mask = torch.arange(m_max, device=self.device).unsqueeze(0) < q_lens_gpu.unsqueeze(1)
        sim_c = torch.matmul(qt, self.corpus_mean).masked_fill(~valid_mask, 1e9)
        order = torch.argsort(sim_c, dim=1)
        qt = torch.gather(qt, 1, order.unsqueeze(2).expand(-1, -1, self.D)).contiguous()
        q_fp16 = qt.half()

        warm_idx, warm_scores, tau_init = self._stage0_prewarm(q_fp16, q_lens_gpu, B, topk, c_warm)

        slice_edges = [0, m_max // 4, m_max // 2, (3 * m_max) // 4, m_max]
        q_rot = torch.matmul(qt, self.H_t)
        q_norms = qt.norm(dim=2)
        q_sc = q_rot.abs().amax(dim=2, keepdim=True).clamp(min=1e-6) / 127.0
        q_i8 = torch.round(q_rot / q_sc).clamp(-127, 127).to(torch.int8)
        q_err = (q_rot - q_i8.float() * q_sc).norm(dim=2)
        eps_buf = (q_err.sum(dim=1) * self.max_d_norm + 2e-3 * q_norms.sum(dim=1) + 1e-3).contiguous()
        eps_safe = float(eps_buf.max().item())

        slice_ub = [torch.zeros((B, self.N_docs), dtype=torch.float32, device=self.device) for _ in range(4)]
        slice_lb = [torch.zeros((B, self.N_docs), dtype=torch.float32, device=self.device) for _ in range(4)]
        slice_l0 = [torch.zeros((B, self.N_docs), dtype=torch.float32, device=self.device) for _ in range(4)]

        d_means_f = self.d_means.float()
        for s_idx in range(4):
            s0, s1 = slice_edges[s_idx], slice_edges[s_idx + 1]
            slen = s1 - s0
            q_sl = qt[:, s0:s1, :].contiguous()
            qn_sl = q_norms[:, s0:s1].contiguous()
            for dc in range(0, self.N_docs, 8192):
                de = min(dc + 8192, self.N_docs)
                dots_m = torch.matmul(q_sl.reshape(B * slen, self.D), d_means_f[dc:de].t()).reshape(B, slen, de - dc)
                l0_tok = torch.minimum(
                    (qn_sl * self.max_d_norm).unsqueeze(2),
                    dots_m + qn_sl.unsqueeze(2) * self.d_radii[dc:de].view(1, 1, -1)
                )
                slice_l0[s_idx][:, dc:de] = l0_tok.sum(dim=1)

            qi8_pad = torch.zeros((B, 32, self.D), dtype=torch.int8, device=self.device)
            qsc_pad = torch.zeros((B, 32), dtype=torch.float32, device=self.device)
            qn_pad = torch.zeros((B, 32), dtype=torch.float32, device=self.device)
            qi8_pad[:, :slen, :] = q_i8[:, s0:s1, :]
            qsc_pad[:, :slen] = q_sc.squeeze(2)[:, s0:s1]
            qn_pad[:, :slen] = qn_sl

            act_tmp = torch.ones((B, self.N_docs), dtype=torch.uint8, device=self.device)
            rem_zero = torch.zeros(B, dtype=torch.float32, device=self.device)
            self._exec_int8_pass32(
                qi8_pad.contiguous(), qsc_pad.contiguous(), qn_pad.contiguous(),
                tau_init, slice_ub[s_idx], slice_lb[s_idx],
                rem_zero, act_tmp, 0, eps_safe, 1, 0, force_w1=False,
                enable_2slice=False, eps_buf=eps_buf, is_packed_hd=False
            )

        total_pairs = float(B * self.N_docs)
        doc_lens_gpu = (self.d_offsets_gpu[1:] - self.d_offsets_gpu[:-1]).float()
        q_lens_f = q_lens_gpu.float()
        total_tok_pairs = float(q_lens_f.sum().item() * float(doc_lens_gpu.sum().item()))

        active = torch.ones((B, self.N_docs), dtype=torch.bool, device=self.device)
        active.scatter_(1, warm_idx, False)
        pruned_pcts = []
        cum_ub = torch.zeros_like(slice_ub[0])
        cum_lb = torch.zeros_like(slice_lb[0])

        for s_idx in range(4):
            cum_ub = cum_ub + slice_ub[s_idx]
            cum_lb = cum_lb + slice_lb[s_idx]
            rem_l0 = sum(slice_l0[r] for r in range(s_idx + 1, 4)) if s_idx < 3 else 0.0
            if s_idx < 3:
                tau_cur = tau_init
            else:
                lb_full = cum_lb.scatter(1, warm_idx, warm_scores)
                tau_cur = torch.maximum(tau_init, torch.topk(lb_full, k=min(topk, self.N_docs), dim=1).values[:, -1])
            active = active & ((cum_ub + rem_l0) >= (tau_cur[:, None] - eps_buf[:, None]))
            pruned_pcts.append(100.0 * (1.0 - float(active.sum().item()) / total_pairs))

        nz = torch.nonzero(active, as_tuple=False)
        scores_mat = torch.full((B, self.N_docs), -1e9, dtype=torch.float32, device=self.device)
        scores_mat.scatter_(1, warm_idx, warm_scores)
        verified_tok_pairs = float((doc_lens_gpu[warm_idx] * q_lens_f[:, None]).sum().item())
        if nz.numel() > 0:
            sq = nz[:, 0].to(torch.int32).contiguous()
            sd = nz[:, 1].to(torch.int32).contiguous()
            scores_mat[sq.long(), sd.long()] = self._verify_pairs(sq, sd, q_fp16, q_lens_gpu)
            verified_tok_pairs += float((doc_lens_gpu[sd.long()] * q_lens_f[sq.long()]).sum().item())

        top_vals, top_idx = torch.topk(scores_mat, k=min(topk, self.N_docs), dim=1)
        if self.sort_by_length:
            top_idx = self.perm_to_orig_gpu[top_idx]

        tok_pair_skip_pct = 100.0 * (1.0 - verified_tok_pairs / max(total_tok_pairs, 1.0))
        return top_idx.cpu().numpy(), top_vals.cpu().numpy(), {
            "slice1_prune_pct": round(pruned_pcts[0], 2),
            "slice2_prune_pct": round(pruned_pcts[1], 2),
            "slice3_prune_pct": round(pruned_pcts[2], 2),
            "slice4_prune_pct": round(pruned_pcts[3], 4),
            "avg_slice_prune_pct": round(float(np.mean(pruned_pcts)), 2),
            "exact_tok_pair_skip_pct": round(tok_pair_skip_pct, 4),
        }

    @torch.no_grad()
    def search_rerank(self, q_tokens_fp16: torch.Tensor, q_offsets: np.ndarray,
                      cand_ids: torch.Tensor, topk: int = 10, c_warm: int = 16):
        """Mode B: Exact MaxSim Reranking over per-query candidate sets [B, C] (any D % 128 == 0!)."""
        torch.cuda.set_device(self.device)
        assert self.mode == "int8"
        if not torch.is_tensor(cand_ids):
            cand_ids = torch.from_numpy(np.asarray(cand_ids, dtype=np.int64))
        orig_cands_gpu = cand_ids.to(device=self.device, dtype=torch.int64).contiguous()
        if self.sort_by_length:
            int_cands = torch.where(orig_cands_gpu >= 0, self.orig_to_perm_gpu[orig_cands_gpu.clamp(min=0)], orig_cands_gpu).to(torch.int32).contiguous()
        else:
            int_cands = orig_cands_gpu.to(torch.int32).contiguous()

        B, C = int_cands.shape
        _, m_max, m_pad, q_lens_gpu, q_fp16_pad, q_int8, q_scales, q_norms, eps_buf = \
            self._prepare_queries(q_tokens_fp16, q_offsets, pad_mult=32, for_rerank=True)

        c_w = min(max(c_warm, topk), C)
        wq = self._wq_cache.get((B, c_w))
        if wq is None:
            wq = torch.arange(B, device=self.device, dtype=torch.int32).unsqueeze(1).repeat(1, c_w).reshape(-1).contiguous()
            if B <= 128:
                self._wq_cache[(B, c_w)] = wq
        wd = int_cands[:, :c_w].reshape(-1).contiguous()
        warm_scores = self._verify_pairs(wq, wd, q_fp16_pad, q_lens_gpu).view(B, c_w)
        k_eff = min(topk, C)
        tau_buf = torch.topk(warm_scores, k=k_eff, dim=1).values[:, -1].contiguous()

        ub = torch.zeros((B, C), dtype=torch.float32, device=self.device)
        lb = torch.zeros((B, C), dtype=torch.float32, device=self.device)
        n_passes = m_pad // 32
        for p in range(n_passes):
            self.ext.fused_tc_rerank_int8(
                q_int8, q_scales, q_norms,
                self.d_qdata, self.d_scales, self.d_rho,
                self.d_pad_offs_gpu, int_cands, ub, lb, p * 32, 1 if p == 0 else 0
            )
        lb_kth = torch.topk(lb, k=k_eff, dim=1).values[:, -1]
        tau_buf = torch.maximum(tau_buf, lb_kth)

        surv_mask = (ub >= (tau_buf[:, None] - eps_buf[:, None]))
        surv_mask[:, :c_w] = False
        nz = torch.nonzero(surv_mask, as_tuple=False)
        num_surv = int(nz.shape[0])

        cand_scores = torch.full((B, C), -1e9, dtype=torch.float32, device=self.device)
        cand_scores[:, :c_w] = warm_scores
        if num_surv > 0:
            sq = nz[:, 0].to(torch.int32).contiguous()
            sc_pos = nz[:, 1].to(torch.long)
            sd = int_cands[sq.long(), sc_pos].contiguous()
            ex = self._verify_pairs(sq, sd, q_fp16_pad, q_lens_gpu)
            cand_scores[sq.long(), sc_pos] = ex

        vals, pos = torch.topk(cand_scores, k=k_eff, dim=1)
        top_docs = torch.gather(orig_cands_gpu, 1, pos)
        all_ids = np.full((B, topk), -1, dtype=np.int64)
        all_scores = np.full((B, topk), -1e9, dtype=np.float32)
        all_ids[:, :k_eff] = top_docs.cpu().numpy()
        all_scores[:, :k_eff] = vals.cpu().numpy()
        return all_ids, all_scores, {
            "num_survivors": num_surv,
            "survivor_ratio": float(num_surv / max(B * C, 1)),
            "tau_buf": tau_buf,
        }

    def _verify_pairs(self, surv_q, surv_doc, q_fp16_pad, q_lens_gpu):
        if not self.host_offload_fp16:
            return self.ext.level2_exact_verify(
                surv_q, surv_doc, q_fp16_pad, q_lens_gpu,
                self.d_tokens_fp16_gpu, self.d_offsets_gpu
            )
        uniq_docs, inv = torch.unique(surv_doc, return_inverse=True)
        u_np = uniq_docs.cpu().numpy()
        chunks = [self.d_tokens_fp16_host[int(self.d_offsets_np[d]):int(self.d_offsets_np[d + 1])] for d in u_np]
        lens = np.array([len(c) for c in chunks], dtype=np.int64)
        offs = np.zeros(len(u_np) + 1, dtype=np.int64)
        np.cumsum(lens, out=offs[1:])
        self._last_h2d_bytes = getattr(self, "_last_h2d_bytes", 0) + int(lens.sum()) * self.D * 2 + len(offs) * 8
        cat_cpu = torch.cat(chunks, dim=0)
        if not self.d_tokens_fp16_host.is_pinned():
            cat_cpu = cat_cpu.pin_memory()
        d_sub_gpu = cat_cpu.to(device=self.device, non_blocking=True)
        offs_gpu = torch.from_numpy(offs).to(self.device)
        return self.ext.level2_exact_verify(
            surv_q, inv.to(torch.int32), q_fp16_pad, q_lens_gpu, d_sub_gpu, offs_gpu
        )
