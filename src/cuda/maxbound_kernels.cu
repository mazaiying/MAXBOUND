#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <math.h>

using namespace nvcuda;

#define WARP_SIZE 32
#define SH_STRIDE 144

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void mma_m16n8k32_s8(
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int32_t c0, int32_t c1, int32_t c2, int32_t c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
}

// ============================================================================
// Baseline B2: Length-Unpadded cuBLAS FP16 GEMM + Coalesced Segmented Reduce
// ============================================================================
__global__ void unpadded_cublas_reduce_kernel(
    const half* __restrict__ sim,
    const int64_t* __restrict__ d_offs,
    const int* __restrict__ q_lens,
    float* __restrict__ out_scores,
    int B, int m, int N_docs, int64_t total_toks)
{
    int doc_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    int q = blockIdx.y;
    int lane = threadIdx.x & 31;
    if (doc_idx >= N_docs || q >= B) return;

    int64_t ds = d_offs[doc_idx];
    int64_t de = d_offs[doc_idx + 1];
    int mq = q_lens[q];
    float total = 0.0f;

    for (int i = 0; i < mq; ++i) {
        const half* row = sim + ((int64_t)(q * m + i) * total_toks);
        float best = -1e9f;
        for (int64_t j = ds + lane; j < de; j += WARP_SIZE) {
            best = fmaxf(best, __half2float(row[j]));
        }
        best = warp_reduce_max(best);
        if (lane == 0) total += best;
    }
    if (lane == 0) {
        out_scores[(int64_t)q * N_docs + doc_idx] = total;
    }
}

// ============================================================================
// M2 Core Kernel (D=128 Fast Register-Resident Query Path):
// Adaptive 32-Query-Token Single-Pass PTX m16n8k32 (INT8)
// ============================================================================
template <int DOCS_PER_CTA, int WARPS_PER_CTA>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32, (WARPS_PER_CTA == 1) ? 8 : 2) fused_tc_int8_pass32_kernel(
    const int8_t* __restrict__ q_int8,      // [B, m_pad, 128]
    const float* __restrict__ q_scales,     // [B, m_pad]
    const float* __restrict__ q_norms,      // [B, m_pad]
    const int8_t* __restrict__ d_int8,      // [total_pad_toks, 128]
    const float* __restrict__ d_scales,     // [total_pad_toks]
    const float* __restrict__ d_rho,        // [total_pad_toks]
    const int64_t* __restrict__ d_pad_offs, // [N_docs + 1]
    const float* __restrict__ tau_buf,      // [B]
    float* __restrict__ ub_accum,           // [B, N_docs]
    float* __restrict__ lb_accum,           // [B, N_docs]
    const float* __restrict__ rem_bound,    // [B]
    const float* __restrict__ eps_buf,      // [B] or nullptr
    uint8_t* __restrict__ doc_active,       // [B, N_docs]
    int B, int m_pad, int N_docs, int q_tok_off, float eps_safe, int is_first_pass, int do_prune)
{
    int doc_base = blockIdx.x * DOCS_PER_CTA;
    int q_base = blockIdx.y * WARPS_PER_CTA;
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;
    int tid_in_g = lane & 3;
    int q = q_base + warp_id;
    constexpr int BLOCK_TH = WARPS_PER_CTA * 32;

    __shared__ int8_t sh_d[32 * SH_STRIDE];
    __shared__ float sh_ds[32];
    __shared__ float sh_dr[32];
    __shared__ int any_warp_needs;

    uint32_t a0[4], a1[4], a2[4], a3[4];
    uint32_t b0_q[4], b1_q[4], b2_q[4], b3_q[4];
    float qs0 = 0.0f, qs1 = 0.0f, qs2 = 0.0f, qs3 = 0.0f;
    float qn0 = 0.0f, qn1 = 0.0f, qn2 = 0.0f, qn3 = 0.0f;
    float tau_val = -1e9f, rem_b = 0.0f, eps_val = eps_safe;
    bool q_valid = (q < B);

    if (q_valid) {
        int qtb = q * m_pad + q_tok_off;
        const int8_t* qp0 = q_int8 + (int64_t)qtb * 128;
        const int8_t* qp1 = qp0 + 16 * 128;
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            int k_off = ks * 32 + tid_in_g * 4;
            a0[ks] = *reinterpret_cast<const uint32_t*>(qp0 + group_id * 128 + k_off);
            a1[ks] = *reinterpret_cast<const uint32_t*>(qp0 + (group_id + 8) * 128 + k_off);
            a2[ks] = *reinterpret_cast<const uint32_t*>(qp0 + group_id * 128 + k_off + 16);
            a3[ks] = *reinterpret_cast<const uint32_t*>(qp0 + (group_id + 8) * 128 + k_off + 16);

            b0_q[ks] = *reinterpret_cast<const uint32_t*>(qp1 + group_id * 128 + k_off);
            b1_q[ks] = *reinterpret_cast<const uint32_t*>(qp1 + (group_id + 8) * 128 + k_off);
            b2_q[ks] = *reinterpret_cast<const uint32_t*>(qp1 + group_id * 128 + k_off + 16);
            b3_q[ks] = *reinterpret_cast<const uint32_t*>(qp1 + (group_id + 8) * 128 + k_off + 16);
        }
        qs0 = q_scales[qtb + group_id];      qs1 = q_scales[qtb + group_id + 8];
        qs2 = q_scales[qtb + 16 + group_id]; qs3 = q_scales[qtb + 24 + group_id];
        qn0 = q_norms[qtb + group_id];       qn1 = q_norms[qtb + group_id + 8];
        qn2 = q_norms[qtb + 16 + group_id];  qn3 = q_norms[qtb + 24 + group_id];
        if (do_prune) {
            tau_val = tau_buf[q];
            rem_b = rem_bound[q];
            if (eps_buf != nullptr) eps_val = eps_buf[q];
        }
    }

    int doc_end = min(doc_base + DOCS_PER_CTA, N_docs);
    for (int d = doc_base; d < doc_end; ++d) {
        bool active = q_valid && (doc_active[(int64_t)q * N_docs + d] != 0);
        if constexpr (WARPS_PER_CTA > 1) {
            if (threadIdx.x == 0) any_warp_needs = 0;
            __syncthreads();
            if (lane == 0 && active) atomicOr(&any_warp_needs, 1);
            __syncthreads();
            if (any_warp_needs == 0) continue;
        } else {
            if (!active) continue;
        }

        int64_t p_start = d_pad_offs[d];
        int64_t p_end = d_pad_offs[d + 1];

        float ub0 = (qn0 > 0.0f) ? -1e9f : 0.0f, lb0 = ub0;
        float ub1 = (qn1 > 0.0f) ? -1e9f : 0.0f, lb1 = ub1;
        float ub2 = (qn2 > 0.0f) ? -1e9f : 0.0f, lb2 = ub2;
        float ub3 = (qn3 > 0.0f) ? -1e9f : 0.0f, lb3 = ub3;

        for (int64_t t = p_start; t < p_end; t += 32) {
            int n_tok = (int)min((int64_t)32, p_end - t);
            for (int idx = threadIdx.x; idx < n_tok * 8; idx += BLOCK_TH) {
                int tok_i = idx >> 3;
                int vec_i = idx & 7;
                const int4* src = reinterpret_cast<const int4*>(d_int8 + (t + tok_i) * 128);
                *reinterpret_cast<int4*>(sh_d + tok_i * SH_STRIDE + vec_i * 16) = src[vec_i];
            }
            if (threadIdx.x < n_tok) {
                sh_ds[threadIdx.x] = d_scales[t + threadIdx.x];
                sh_dr[threadIdx.x] = d_rho[t + threadIdx.x];
            } else if (threadIdx.x < 32) {
                sh_ds[threadIdx.x] = 0.0f;
                sh_dr[threadIdx.x] = 0.0f;
            }
            if constexpr (WARPS_PER_CTA > 1) __syncthreads(); else __syncwarp();

            if (active) {
                for (int n_sub = 0; n_sub < n_tok; n_sub += 8) {
                    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    const int8_t* dp = sh_d + (n_sub + group_id) * SH_STRIDE;
                    #pragma unroll
                    for (int ks = 0; ks < 4; ++ks) {
                        int k_off = ks * 32 + tid_in_g * 4;
                        uint32_t b0 = *reinterpret_cast<const uint32_t*>(dp + k_off);
                        uint32_t b1 = *reinterpret_cast<const uint32_t*>(dp + k_off + 16);
                        mma_m16n8k32_s8(c0, c1, c2, c3, a0[ks], a1[ks], a2[ks], a3[ks], b0, b1, c0, c1, c2, c3);
                        mma_m16n8k32_s8(d0, d1, d2, d3, b0_q[ks], b1_q[ks], b2_q[ks], b3_q[ks], b0, b1, d0, d1, d2, d3);
                    }
                    int j0 = n_sub + tid_in_g * 2;
                    int j1 = j0 + 1;
                    float ds0 = sh_ds[j0], ds1 = sh_ds[j1];
                    float dr0 = sh_dr[j0], dr1 = sh_dr[j1];

                    if (ds0 > 0.0f) {
                        if (qn0 > 0.0f) { float dt = (float)c0 * qs0 * ds0, er = qn0 * dr0; ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); }
                        if (qn1 > 0.0f) { float dt = (float)c2 * qs1 * ds0, er = qn1 * dr0; ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(lb1, dt - er); }
                        if (qn2 > 0.0f) { float dt = (float)d0 * qs2 * ds0, er = qn2 * dr0; ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); }
                        if (qn3 > 0.0f) { float dt = (float)d2 * qs3 * ds0, er = qn3 * dr0; ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); }
                    }
                    if (ds1 > 0.0f) {
                        if (qn0 > 0.0f) { float dt = (float)c1 * qs0 * ds1, er = qn0 * dr1; ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); }
                        if (qn1 > 0.0f) { float dt = (float)c3 * qs1 * ds1, er = qn1 * dr1; ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(lb1, dt - er); }
                        if (qn2 > 0.0f) { float dt = (float)d1 * qs2 * ds1, er = qn2 * dr1; ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); }
                        if (qn3 > 0.0f) { float dt = (float)d3 * qs3 * ds1, er = qn3 * dr1; ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); }
                    }
                }
            }
            if constexpr (WARPS_PER_CTA > 1) __syncthreads(); else __syncwarp();
        }

        if (active) {
            #define RED4(v) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1)); v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
            RED4(ub0); RED4(ub1); RED4(ub2); RED4(ub3);
            RED4(lb0); RED4(lb1); RED4(lb2); RED4(lb3);
            #undef RED4
            float sum_ub = ub0 + ub1 + ub2 + ub3;
            float sum_lb = lb0 + lb1 + lb2 + lb3;
            sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 4);
            sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 8);
            sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 16);
            sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 4);
            sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 8);
            sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 16);

            if (lane == 0) {
                int64_t idx = (int64_t)q * N_docs + d;
                float prev_ub = is_first_pass ? 0.0f : ub_accum[idx];
                float prev_lb = is_first_pass ? 0.0f : lb_accum[idx];
                float new_ub = prev_ub + sum_ub;
                float new_lb = prev_lb + sum_lb;
                ub_accum[idx] = new_ub;
                lb_accum[idx] = new_lb;
                if (do_prune) {
                    doc_active[idx] = (new_ub + rem_b >= tau_val - eps_val) ? 1 : 0;
                }
            }
        }
    }
}


// ============================================================================
// M2 Operator P2-Fused (D >= 256..1024+ True Shared-Memory Multi-Query PTX m16n8k32):
//   - Processes 16 query tokens per slice (pass16) across WQ queries per CTA
//     (WQ=8 on A100 sm_80 in 138 KB dynamic SMEM; WQ=4 on RTX 4080 sm_89 in 72 KB SMEM)
//   - Preloads all WQ queries' packed MMA int8 slices ONCE per CTA into sh_q!
//   - Streams 32 doc tokens x 128 dims into sh_d ONCE per WQ queries, reuses d_int8
//     register fragments across all WQ queries, holds int32 dot accumulators in registers
//     across db = 0..n_dblks-1 with ZERO HBM similarity matrix materialization!
// ============================================================================
template <int DOCS_PER_CTA, int WQ>
__global__ void __launch_bounds__(128, 1) fused_tc_int8_hd_pass16_shm_kernel(
    const int8_t* __restrict__ q_pack_i8,   // uint32_t [B, n_slices16, n_dblks, 512]
    const float* __restrict__ q_scales,     // [B, m_pad]
    const float* __restrict__ q_norms,      // [B, m_pad]
    const int8_t* __restrict__ d_int8,      // [total_pad_toks, D]
    const float* __restrict__ d_scales,     // [total_pad_toks]
    const float* __restrict__ d_rho,        // [total_pad_toks]
    const int64_t* __restrict__ d_pad_offs, // [N_docs + 1]
    const float* __restrict__ tau_buf,      // [B]
    float* __restrict__ ub_accum,           // [B, N_docs]
    float* __restrict__ lb_accum,           // [B, N_docs]
    const float* __restrict__ rem_bound,    // [B] or [B, N_docs] when rem_is_2d == 1
    const float* __restrict__ eps_buf,      // [B] or nullptr
    uint8_t* __restrict__ doc_active,       // [B, N_docs]
    int B, int m_pad, int N_docs, int D, int q_tok_off, float eps_safe,
    int is_first_pass, int do_prune, int rem_is_2d)
{
    int q_base = blockIdx.x * WQ;
    if (q_base >= B) return;
    int doc_base = blockIdx.y * DOCS_PER_CTA;
    int doc_end = min(doc_base + DOCS_PER_CTA, N_docs);

    int warp_id = threadIdx.x >> 5;   // 0..3
    int lane = threadIdx.x & 31;      // 0..31
    int group_id = lane >> 2;         // 0..7
    int tid_in_g = lane & 3;          // 0..3
    int warp_tok_base = warp_id * 8;

    int n_dblks = D >> 7;
    int n_slices = m_pad >> 4;
    int slice_idx = q_tok_off >> 4;
    const uint32_t* q_pack_all = reinterpret_cast<const uint32_t*>(q_pack_i8);

    extern __shared__ char smem_raw[];
    int sh_q_words_per_q = (n_dblks <= 8) ? (n_dblks * 512) : 0;
    uint32_t* sh_q = reinterpret_cast<uint32_t*>(smem_raw);
    int8_t* sh_d = reinterpret_cast<int8_t*>(sh_q + WQ * sh_q_words_per_q);
    float* sh_ds = reinterpret_cast<float*>(sh_d + 32 * SH_STRIDE);
    float* sh_dr = sh_ds + 32;
    float* sh_qs = sh_dr + 32;
    float* sh_qn = sh_qs + WQ * 16;
    float* sh_ub = sh_qn + WQ * 16;
    float* sh_lb = sh_ub + WQ * 4 * 16;
    int* sh_q_act = reinterpret_cast<int*>(sh_lb + WQ * 4 * 16);
    int* p_any = sh_q_act + WQ;

    // Fast CTA-level early exit on subsequent slices when ALL (WQ x DOCS_PER_CTA) pairs are already pruned!
    if (!is_first_pass) {
        if (threadIdx.x == 0) *p_any = 0;
        __syncthreads();
        for (int idx = threadIdx.x; idx < WQ * DOCS_PER_CTA; idx += 128) {
            int qi = idx / DOCS_PER_CTA;
            int di = doc_base + (idx % DOCS_PER_CTA);
            int q_id = q_base + qi;
            if (q_id < B && di < doc_end && doc_active[(int64_t)q_id * N_docs + di] != 0) {
                atomicOr(p_any, 1);
                break;
            }
        }
        __syncthreads();
        if (*p_any == 0) return;
    }

    // Preload scales, norms, and packed query slices into dynamic SMEM ONCE per CTA
    for (int qi = 0; qi < WQ; ++qi) {
        int q_id = q_base + qi;
        if (q_id < B) {
            if (threadIdx.x < 16) {
                sh_qs[qi * 16 + threadIdx.x] = q_scales[q_id * m_pad + q_tok_off + threadIdx.x];
                sh_qn[qi * 16 + threadIdx.x] = q_norms[q_id * m_pad + q_tok_off + threadIdx.x];
            }
            if (n_dblks <= 8) {
                const uint32_t* src_q = q_pack_all + ((int64_t)q_id * n_slices + slice_idx) * sh_q_words_per_q;
                uint32_t* dst_q = sh_q + qi * sh_q_words_per_q;
                for (int idx = threadIdx.x; idx < sh_q_words_per_q; idx += 128) {
                    dst_q[idx] = src_q[idx];
                }
            }
        } else {
            if (threadIdx.x < 16) {
                sh_qs[qi * 16 + threadIdx.x] = 0.0f;
                sh_qn[qi * 16 + threadIdx.x] = 0.0f;
            }
            if (n_dblks <= 8) {
                uint32_t* dst_q = sh_q + qi * sh_q_words_per_q;
                for (int idx = threadIdx.x; idx < sh_q_words_per_q; idx += 128) {
                    dst_q[idx] = 0u;
                }
            }
        }
    }
    __syncthreads();

    for (int d = doc_base; d < doc_end; ++d) {
        if (threadIdx.x == 0) *p_any = 0;
        __syncthreads();
        if (threadIdx.x < WQ) {
            int q_id = q_base + threadIdx.x;
            int act = (q_id < B && doc_active[(int64_t)q_id * N_docs + d] != 0) ? 1 : 0;
            sh_q_act[threadIdx.x] = act;
            if (act) atomicOr(p_any, 1);
        }
        __syncthreads();
        if (*p_any == 0) continue;

        int64_t p_start = d_pad_offs[d];
        int64_t p_end = d_pad_offs[d + 1];

        float ub0[WQ], ub1[WQ], lb0[WQ], lb1[WQ];
        #pragma unroll
        for (int qi = 0; qi < WQ; ++qi) {
            float qn_a = sh_qn[qi * 16 + group_id];
            float qn_b = sh_qn[qi * 16 + group_id + 8];
            ub0[qi] = (qn_a > 0.0f) ? -1e9f : 0.0f; lb0[qi] = ub0[qi];
            ub1[qi] = (qn_b > 0.0f) ? -1e9f : 0.0f; lb1[qi] = ub1[qi];
        }

        for (int64_t t = p_start; t < p_end; t += 32) {
            int n_tok = (int)min((int64_t)32, p_end - t);
            if (threadIdx.x < n_tok) {
                sh_ds[threadIdx.x] = d_scales[t + threadIdx.x];
                sh_dr[threadIdx.x] = d_rho[t + threadIdx.x];
            } else if (threadIdx.x < 32) {
                sh_ds[threadIdx.x] = 0.0f;
                sh_dr[threadIdx.x] = 0.0f;
            }

            int32_t c0[WQ], c1[WQ], c2[WQ], c3[WQ];
            #pragma unroll
            for (int qi = 0; qi < WQ; ++qi) {
                c0[qi] = 0; c1[qi] = 0; c2[qi] = 0; c3[qi] = 0;
            }

            for (int db = 0; db < n_dblks; ++db) {
                int d_off = db * 128;
                for (int idx = threadIdx.x; idx < n_tok * 8; idx += 128) {
                    int tok_i = idx >> 3;
                    int vec_i = idx & 7;
                    const int4* src = reinterpret_cast<const int4*>(d_int8 + (t + tok_i) * (int64_t)D + d_off);
                    *reinterpret_cast<int4*>(sh_d + tok_i * SH_STRIDE + vec_i * 16) = src[vec_i];
                }
                __syncthreads();

                if (warp_tok_base < n_tok) {
                    const int8_t* dp = sh_d + (warp_tok_base + group_id) * SH_STRIDE;
                    uint32_t b0_arr[4], b1_arr[4];
                    #pragma unroll
                    for (int ks = 0; ks < 4; ++ks) {
                        int k_off = ks * 32 + tid_in_g * 4;
                        b0_arr[ks] = *reinterpret_cast<const uint32_t*>(dp + k_off);
                        b1_arr[ks] = *reinterpret_cast<const uint32_t*>(dp + k_off + 16);
                    }

                    #pragma unroll
                    for (int qi = 0; qi < WQ; ++qi) {
                        if (sh_q_act[qi]) {
                            const uint32_t* qp = ((n_dblks <= 8)
                                ? (sh_q + qi * sh_q_words_per_q)
                                : (q_pack_all + ((int64_t)(q_base + qi) * n_slices + slice_idx) * n_dblks * 512))
                                + db * 512 + lane;
                            #pragma unroll
                            for (int ks = 0; ks < 4; ++ks) {
                                uint32_t a0 = qp[(ks * 4 + 0) * 32];
                                uint32_t a1 = qp[(ks * 4 + 1) * 32];
                                uint32_t a2 = qp[(ks * 4 + 2) * 32];
                                uint32_t a3 = qp[(ks * 4 + 3) * 32];
                                mma_m16n8k32_s8(c0[qi], c1[qi], c2[qi], c3[qi],
                                                a0, a1, a2, a3, b0_arr[ks], b1_arr[ks],
                                                c0[qi], c1[qi], c2[qi], c3[qi]);
                            }
                        }
                    }
                }
                __syncthreads();
            }

            if (warp_tok_base < n_tok) {
                int j0 = warp_tok_base + tid_in_g * 2;
                int j1 = j0 + 1;
                float ds0 = sh_ds[j0], ds1 = sh_ds[j1];
                float dr0 = sh_dr[j0], dr1 = sh_dr[j1];
                #pragma unroll
                for (int qi = 0; qi < WQ; ++qi) {
                    if (sh_q_act[qi]) {
                        float qs_a = sh_qs[qi * 16 + group_id],     qs_b = sh_qs[qi * 16 + group_id + 8];
                        float qn_a = sh_qn[qi * 16 + group_id],     qn_b = sh_qn[qi * 16 + group_id + 8];
                        if (ds0 > 0.0f) {
                            if (qn_a > 0.0f) { float dt = (float)c0[qi] * qs_a * ds0, er = qn_a * dr0; ub0[qi] = fmaxf(ub0[qi], dt + er); lb0[qi] = fmaxf(lb0[qi], dt - er); }
                            if (qn_b > 0.0f) { float dt = (float)c2[qi] * qs_b * ds0, er = qn_b * dr0; ub1[qi] = fmaxf(ub1[qi], dt + er); lb1[qi] = fmaxf(lb1[qi], dt - er); }
                        }
                        if (ds1 > 0.0f) {
                            if (qn_a > 0.0f) { float dt = (float)c1[qi] * qs_a * ds1, er = qn_a * dr1; ub0[qi] = fmaxf(ub0[qi], dt + er); lb0[qi] = fmaxf(lb0[qi], dt - er); }
                            if (qn_b > 0.0f) { float dt = (float)c3[qi] * qs_b * ds1, er = qn_b * dr1; ub1[qi] = fmaxf(ub1[qi], dt + er); lb1[qi] = fmaxf(lb1[qi], dt - er); }
                        }
                    }
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int qi = 0; qi < WQ; ++qi) {
            if (sh_q_act[qi]) {
                ub0[qi] = fmaxf(ub0[qi], __shfl_xor_sync(0xffffffff, ub0[qi], 1));
                ub0[qi] = fmaxf(ub0[qi], __shfl_xor_sync(0xffffffff, ub0[qi], 2));
                ub1[qi] = fmaxf(ub1[qi], __shfl_xor_sync(0xffffffff, ub1[qi], 1));
                ub1[qi] = fmaxf(ub1[qi], __shfl_xor_sync(0xffffffff, ub1[qi], 2));
                lb0[qi] = fmaxf(lb0[qi], __shfl_xor_sync(0xffffffff, lb0[qi], 1));
                lb0[qi] = fmaxf(lb0[qi], __shfl_xor_sync(0xffffffff, lb0[qi], 2));
                lb1[qi] = fmaxf(lb1[qi], __shfl_xor_sync(0xffffffff, lb1[qi], 1));
                lb1[qi] = fmaxf(lb1[qi], __shfl_xor_sync(0xffffffff, lb1[qi], 2));
                if (tid_in_g == 0) {
                    int base_off = (qi * 4 + warp_id) * 16;
                    sh_ub[base_off + group_id]     = ub0[qi];
                    sh_ub[base_off + group_id + 8] = ub1[qi];
                    sh_lb[base_off + group_id]     = lb0[qi];
                    sh_lb[base_off + group_id + 8] = lb1[qi];
                }
            }
        }
        __syncthreads();

        for (int qi = warp_id; qi < WQ; qi += 4) {
            int q_id = q_base + qi;
            if (q_id < B && sh_q_act[qi]) {
                float u_max = 0.0f, l_max = 0.0f;
                if (lane < 16) {
                    int b0_i = (qi * 4 + 0) * 16 + lane;
                    int b1_i = (qi * 4 + 1) * 16 + lane;
                    int b2_i = (qi * 4 + 2) * 16 + lane;
                    int b3_i = (qi * 4 + 3) * 16 + lane;
                    u_max = fmaxf(fmaxf(sh_ub[b0_i], sh_ub[b1_i]), fmaxf(sh_ub[b2_i], sh_ub[b3_i]));
                    l_max = fmaxf(fmaxf(sh_lb[b0_i], sh_lb[b1_i]), fmaxf(sh_lb[b2_i], sh_lb[b3_i]));
                }
                #pragma unroll
                for (int off = 8; off > 0; off >>= 1) {
                    u_max += __shfl_xor_sync(0xffffffff, u_max, off);
                    l_max += __shfl_xor_sync(0xffffffff, l_max, off);
                }
                if (lane == 0) {
                    int64_t idx = (int64_t)q_id * N_docs + d;
                    float new_ub = (is_first_pass ? 0.0f : ub_accum[idx]) + u_max;
                    float new_lb = (is_first_pass ? 0.0f : lb_accum[idx]) + l_max;
                    ub_accum[idx] = new_ub;
                    lb_accum[idx] = new_lb;
                    if (do_prune) {
                        float rem_v = rem_is_2d ? rem_bound[idx] : rem_bound[q_id];
                        float ep_v = (eps_buf != nullptr) ? eps_buf[q_id] : eps_safe;
                        doc_active[idx] = (new_ub + rem_v >= tau_buf[q_id] - ep_v) ? 1 : 0;
                    }
                }
            }
        }
        __syncthreads();
    }
}


// ============================================================================
// Strong Exact Baseline B8: Fused Tensor-Core FP16 MaxSim Kernel (WMMA hmma.16816.f32)
//   - Eliminates ALL intermediate similarity matrix HBM reads/writes!
//   - Shares 16-doc-token FP16 tiles in SMEM across WARPS_PER_CTA queries!
// ============================================================================
#define FP16_SH_STRIDE 136
template <int DOCS_PER_CTA, int WARPS_PER_CTA>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32, 2) fused_tc_fp16_maxsim_kernel(
    const half* __restrict__ q_pad,      // [B, m_pad, D] (m_pad % 16 == 0)
    const int* __restrict__ q_lens,      // [B]
    const half* __restrict__ d_tokens,   // [N_toks, D]
    const int64_t* __restrict__ d_offs,  // [N_docs + 1]
    float* __restrict__ out_scores,      // [B, N_docs]
    int B, int m_pad, int N_docs, int D)
{
    int doc_base = blockIdx.x * DOCS_PER_CTA;
    int doc_end = min(doc_base + DOCS_PER_CTA, N_docs);
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int q = blockIdx.y * WARPS_PER_CTA + warp_id;
    constexpr int BLOCK_TH = WARPS_PER_CTA * 32;

    __shared__ half sh_d[16 * FP16_SH_STRIDE];
    __shared__ float sh_c[WARPS_PER_CTA][16 * 16];
    __shared__ float sh_best[WARPS_PER_CTA][32];

    int mq = (q < B) ? q_lens[q] : 0;
    int row_in_tile = lane >> 1;   // 0..15
    int col_par = lane & 1;        // 0..1

    // Register-resident query fast path for D == 128 across ANY m_pad (16, 32, 64, 128):
    // Loops q_chunk in outer 32-token slices so Q fragments stay 100% in registers across all documents!
    if (D == 128) {
        for (int q_chunk = 0; q_chunk < m_pad; q_chunk += 32) {
            int has_second_16 = (q_chunk + 16 < m_pad) ? 1 : 0;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a0_reg[8];
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a1_reg[8];
            if (q < B) {
                const half* qp0 = q_pad + ((int64_t)q * m_pad + q_chunk) * 128;
                const half* qp1 = qp0 + 16 * 128;
                #pragma unroll
                for (int ks = 0; ks < 8; ++ks) {
                    wmma::load_matrix_sync(a0_reg[ks], qp0 + ks * 16, 128);
                    if (has_second_16) {
                        wmma::load_matrix_sync(a1_reg[ks], qp1 + ks * 16, 128);
                    }
                }
            }

            for (int d = doc_base; d < doc_end; ++d) {
                int64_t ds = d_offs[d];
                int64_t de = d_offs[d + 1];
                if (q < B) {
                    sh_best[warp_id][lane] = (q_chunk + lane < mq) ? -1e9f : 0.0f;
                }
                __syncwarp();

                for (int64_t t = ds; t < de; t += 16) {
                    int n_tok = (int)min((int64_t)16, de - t);
                    for (int idx = threadIdx.x; idx < 16 * 16; idx += BLOCK_TH) {
                        int tok_i = idx >> 4;
                        int vec_i = idx & 15;
                        int4 val = (tok_i < n_tok)
                            ? *reinterpret_cast<const int4*>(d_tokens + (t + tok_i) * 128 + vec_i * 8)
                            : make_int4(0, 0, 0, 0);
                        *reinterpret_cast<int4*>(sh_d + tok_i * FP16_SH_STRIDE + vec_i * 8) = val;
                    }
                    if constexpr (WARPS_PER_CTA > 1) __syncthreads(); else __syncwarp();

                    if (q < B) {
                        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0;
                        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc1;
                        wmma::fill_fragment(acc0, 0.0f);
                        if (has_second_16) wmma::fill_fragment(acc1, 0.0f);
                        #pragma unroll
                        for (int ks = 0; ks < 8; ++ks) {
                            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                            wmma::load_matrix_sync(b_frag, sh_d + ks * 16, FP16_SH_STRIDE);
                            wmma::mma_sync(acc0, a0_reg[ks], b_frag, acc0);
                            if (has_second_16) {
                                wmma::mma_sync(acc1, a1_reg[ks], b_frag, acc1);
                            }
                        }
                        wmma::store_matrix_sync(sh_c[warp_id], acc0, 16, wmma::mem_row_major);
                        __syncwarp();
                        float rm0 = -1e9f;
                        for (int c = col_par; c < n_tok; c += 2) rm0 = fmaxf(rm0, sh_c[warp_id][row_in_tile * 16 + c]);
                        rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffff, rm0, 1));
                        if (col_par == 0 && (q_chunk + row_in_tile) < mq) {
                            sh_best[warp_id][row_in_tile] = fmaxf(sh_best[warp_id][row_in_tile], rm0);
                        }
                        __syncwarp();

                        if (has_second_16) {
                            wmma::store_matrix_sync(sh_c[warp_id], acc1, 16, wmma::mem_row_major);
                            __syncwarp();
                            float rm1 = -1e9f;
                            for (int c = col_par; c < n_tok; c += 2) rm1 = fmaxf(rm1, sh_c[warp_id][row_in_tile * 16 + c]);
                            rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffff, rm1, 1));
                            if (col_par == 0 && (q_chunk + 16 + row_in_tile) < mq) {
                                sh_best[warp_id][16 + row_in_tile] = fmaxf(sh_best[warp_id][16 + row_in_tile], rm1);
                            }
                            __syncwarp();
                        }
                    }
                    if constexpr (WARPS_PER_CTA > 1) __syncthreads(); else __syncwarp();
                }

                if (q < B) {
                    float v = sh_best[warp_id][lane];
                    v = warp_reduce_sum(v);
                    if (lane == 0) {
                        int64_t oidx = (int64_t)q * N_docs + d;
                        out_scores[oidx] = (q_chunk == 0) ? v : (out_scores[oidx] + v);
                    }
                }
                __syncwarp();
            }
        }
        return;
    }

    for (int d = doc_base; d < doc_end; ++d) {
        int64_t ds = d_offs[d];
        int64_t de = d_offs[d + 1];
        float total_q = 0.0f;

        for (int q_chunk = 0; q_chunk < m_pad; q_chunk += 32) {
            int has_second_16 = (q_chunk + 16 < m_pad) ? 1 : 0;
            if (q < B) {
                sh_best[warp_id][lane] = (q_chunk + lane < mq) ? -1e9f : 0.0f;
            }
            __syncwarp();

            for (int64_t t = ds; t < de; t += 16) {
                int n_tok = (int)min((int64_t)16, de - t);
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc1;
                wmma::fill_fragment(acc0, 0.0f);
                if (has_second_16) wmma::fill_fragment(acc1, 0.0f);

                for (int db = 0; db < D; db += 128) {
                    for (int idx = threadIdx.x; idx < 16 * 16; idx += BLOCK_TH) {
                        int tok_i = idx >> 4;
                        int vec_i = idx & 15;
                        int4 val = make_int4(0, 0, 0, 0);
                        if (tok_i < n_tok) {
                            val = *reinterpret_cast<const int4*>(d_tokens + (t + tok_i) * (int64_t)D + db + vec_i * 8);
                        }
                        *reinterpret_cast<int4*>(sh_d + tok_i * FP16_SH_STRIDE + vec_i * 8) = val;
                    }
                    if constexpr (WARPS_PER_CTA > 1) __syncthreads(); else __syncwarp();

                    if (q < B) {
                        const half* q_ptr0 = q_pad + ((int64_t)q * m_pad + q_chunk) * D + db;
                        const half* q_ptr1 = q_ptr0 + (int64_t)16 * D;
                        #pragma unroll
                        for (int ks = 0; ks < 8; ++ks) {
                            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                            wmma::load_matrix_sync(b_frag, sh_d + ks * 16, FP16_SH_STRIDE);
                            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a0_frag;
                            wmma::load_matrix_sync(a0_frag, q_ptr0 + ks * 16, D);
                            wmma::mma_sync(acc0, a0_frag, b_frag, acc0);
                            if (has_second_16) {
                                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a1_frag;
                                wmma::load_matrix_sync(a1_frag, q_ptr1 + ks * 16, D);
                                wmma::mma_sync(acc1, a1_frag, b_frag, acc1);
                            }
                        }
                    }
                    if constexpr (WARPS_PER_CTA > 1) __syncthreads(); else __syncwarp();
                }

                if (q < B) {
                    wmma::store_matrix_sync(sh_c[warp_id], acc0, 16, wmma::mem_row_major);
                    __syncwarp();
                    float rm0 = -1e9f;
                    for (int c = col_par; c < n_tok; c += 2) {
                        rm0 = fmaxf(rm0, sh_c[warp_id][row_in_tile * 16 + c]);
                    }
                    rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffff, rm0, 1));
                    if (col_par == 0 && (q_chunk + row_in_tile) < mq) {
                        sh_best[warp_id][row_in_tile] = fmaxf(sh_best[warp_id][row_in_tile], rm0);
                    }
                    __syncwarp();

                    if (has_second_16) {
                        wmma::store_matrix_sync(sh_c[warp_id], acc1, 16, wmma::mem_row_major);
                        __syncwarp();
                        float rm1 = -1e9f;
                        for (int c = col_par; c < n_tok; c += 2) {
                            rm1 = fmaxf(rm1, sh_c[warp_id][row_in_tile * 16 + c]);
                        }
                        rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffff, rm1, 1));
                        if (col_par == 0 && (q_chunk + 16 + row_in_tile) < mq) {
                            sh_best[warp_id][16 + row_in_tile] = fmaxf(sh_best[warp_id][16 + row_in_tile], rm1);
                        }
                        __syncwarp();
                    }
                }
            }

            if (q < B) {
                float v = sh_best[warp_id][lane];
                v = warp_reduce_sum(v);
                if (lane == 0) total_q += v;
            }
            __syncwarp();
        }

        if (q < B && lane == 0) {
            out_scores[(int64_t)q * N_docs + d] = total_q;
        }
    }
}


// ============================================================================
// M2 General-Dimension Tensor-Core Kernel (D = 128, 256, 512, 1024, ...):
// Uses Warp-Coalesced Lane-Major MMA Query Layout q_pack [B, P, n_dblks, 32, 32]
// + 16-doc-token SMEM tiles (2.3 KB SMEM) + zero-spill launch bounds!
// ============================================================================
template <int DOCS_PER_CTA, int WARPS_PER_CTA, bool SWIZZLE_Q = true>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32, (WARPS_PER_CTA == 1) ? 4 : 2) fused_tc_int8_gendim_pass32_kernel(
    const int8_t* __restrict__ q_pack_i8,   // reinterpret as uint32_t [B, P, n_dblks, 32, 32]
    const float* __restrict__ q_scales,     // [B, m_pad]
    const float* __restrict__ q_norms,      // [B, m_pad]
    const int8_t* __restrict__ d_int8,      // [total_pad_toks, D]
    const float* __restrict__ d_scales,     // [total_pad_toks]
    const float* __restrict__ d_rho,        // [total_pad_toks]
    const int64_t* __restrict__ d_pad_offs, // [N_docs + 1]
    const float* __restrict__ tau_buf,      // [B]
    float* __restrict__ ub_accum,           // [B, N_docs]
    float* __restrict__ lb_accum,           // [B, N_docs]
    const float* __restrict__ rem_bound,    // [B] or [B, N_docs]
    const float* __restrict__ eps_buf,      // [B] or nullptr
    uint8_t* __restrict__ doc_active,       // [B, N_docs]
    int B, int m_pad, int N_docs, int D, int q_tok_off, float eps_safe,
    int is_first_pass, int do_prune, int rem_is_2d, int slice_mode)
{
    int doc_base = SWIZZLE_Q ? (blockIdx.y * DOCS_PER_CTA) : (blockIdx.x * DOCS_PER_CTA);
    int q_base   = SWIZZLE_Q ? (blockIdx.x * WARPS_PER_CTA) : (blockIdx.y * WARPS_PER_CTA);
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;
    int tid_in_g = lane & 3;
    int q = q_base + warp_id;
    constexpr int BLOCK_TH = WARPS_PER_CTA * 32;
    int n_dblks = D >> 7;
    int n_passes = m_pad >> 5;
    int pass_idx = q_tok_off >> 5;
    const uint32_t* q_pack = reinterpret_cast<const uint32_t*>(q_pack_i8);

    __shared__ int8_t sh_d[8 * 16 * SH_STRIDE];
    __shared__ float sh_ds[16];
    __shared__ float sh_dr[16];
    __shared__ int any_warp_needs;

    float qs0 = 0.0f, qs1 = 0.0f, qs2 = 0.0f, qs3 = 0.0f;
    float qn0 = 0.0f, qn1 = 0.0f, qn2 = 0.0f, qn3 = 0.0f;
    float tau_val = -1e9f, rem_b = 0.0f;
    bool q_valid = (q < B);
    int qtb = q * m_pad + q_tok_off;

    if (q_valid) {
        if (slice_mode != 2) {
            qs0 = q_scales[qtb + group_id];      qs1 = q_scales[qtb + group_id + 8];
            qn0 = q_norms[qtb + group_id];       qn1 = q_norms[qtb + group_id + 8];
        }
        if (slice_mode != 1) {
            qs2 = q_scales[qtb + 16 + group_id]; qs3 = q_scales[qtb + 24 + group_id];
            qn2 = q_norms[qtb + 16 + group_id];  qn3 = q_norms[qtb + 24 + group_id];
        }
        if (do_prune) {
            tau_val = tau_buf[q];
            if (!rem_is_2d) rem_b = rem_bound[q];
        }
    }

    int doc_end = min(doc_base + DOCS_PER_CTA, N_docs);
    for (int d = doc_base; d < doc_end; ++d) {
        bool active = q_valid && (doc_active[(int64_t)q * N_docs + d] != 0);
        if constexpr (WARPS_PER_CTA > 1) {
            if (threadIdx.x == 0) any_warp_needs = 0;
            __syncthreads();
            if (lane == 0 && active) atomicOr(&any_warp_needs, 1);
            __syncthreads();
            if (any_warp_needs == 0) continue;
        } else {
            if (!active) continue;
        }

        int64_t p_start = d_pad_offs[d];
        int64_t p_end = d_pad_offs[d + 1];

        float ub0 = (qn0 > 0.0f) ? -1e9f : 0.0f, lb0 = ub0;
        float ub1 = (qn1 > 0.0f) ? -1e9f : 0.0f, lb1 = ub1;
        float ub2 = (qn2 > 0.0f) ? -1e9f : 0.0f, lb2 = ub2;
        float ub3 = (qn3 > 0.0f) ? -1e9f : 0.0f, lb3 = ub3;

        for (int64_t t = p_start; t < p_end; t += 16) {
            int n_tok = (int)min((int64_t)16, p_end - t);
            int n_subs = (n_tok + 7) >> 3;

            int32_t acc_c0_0 = 0, acc_c1_0 = 0, acc_c2_0 = 0, acc_c3_0 = 0;
            int32_t acc_d0_0 = 0, acc_d1_0 = 0, acc_d2_0 = 0, acc_d3_0 = 0;
            int32_t acc_c0_1 = 0, acc_c1_1 = 0, acc_c2_1 = 0, acc_c3_1 = 0;
            int32_t acc_d0_1 = 0, acc_d1_1 = 0, acc_d2_1 = 0, acc_d3_1 = 0;

            if (threadIdx.x < n_tok) {
                sh_ds[threadIdx.x] = d_scales[t + threadIdx.x];
                sh_dr[threadIdx.x] = d_rho[t + threadIdx.x];
            } else if (threadIdx.x < 16) {
                sh_ds[threadIdx.x] = 0.0f;
                sh_dr[threadIdx.x] = 0.0f;
            }

            for (int db_base = 0; db_base < n_dblks; db_base += 8) {
                int nb = min(8, n_dblks - db_base);
                int v_per_tok = nb * 8;
                int tot_v = n_tok * v_per_tok;
                for (int idx = threadIdx.x; idx < tot_v; idx += BLOCK_TH) {
                    int tok_i = (nb == 8) ? (idx >> 6) : (idx / v_per_tok);
                    int rem_v = idx - tok_i * v_per_tok;
                    int db_local = rem_v >> 3;
                    int vec_i = rem_v & 7;
                    const int4* src = reinterpret_cast<const int4*>(d_int8 + (t + tok_i) * (int64_t)D + db_base * 128);
                    *reinterpret_cast<int4*>(sh_d + (db_local * 16 + tok_i) * SH_STRIDE + vec_i * 16) = src[rem_v];
                }
                if constexpr (WARPS_PER_CTA > 1) __syncthreads(); else __syncwarp();

                if (active) {
                    for (int db_local = 0; db_local < nb; ++db_local) {
                        int db = db_base + db_local;
                        const uint32_t* qp = q_pack + (((int64_t)q * n_passes + pass_idx) * n_dblks + db) * 1024 + lane;
                        const int8_t* dp0 = sh_d + (db_local * 16 + group_id) * SH_STRIDE;
                        const int8_t* dp1 = dp0 + 8 * SH_STRIDE;

                        if (slice_mode == 1) {
                            #pragma unroll
                            for (int ks = 0; ks < 4; ++ks) {
                                uint32_t a0 = qp[(ks * 8 + 0) * 32];
                                uint32_t a1 = qp[(ks * 8 + 1) * 32];
                                uint32_t a2 = qp[(ks * 8 + 2) * 32];
                                uint32_t a3 = qp[(ks * 8 + 3) * 32];
                                int k_off = ks * 32 + tid_in_g * 4;
                                uint32_t b0_0 = *reinterpret_cast<const uint32_t*>(dp0 + k_off);
                                uint32_t b1_0 = *reinterpret_cast<const uint32_t*>(dp0 + k_off + 16);
                                mma_m16n8k32_s8(acc_c0_0, acc_c1_0, acc_c2_0, acc_c3_0,
                                                a0, a1, a2, a3, b0_0, b1_0,
                                                acc_c0_0, acc_c1_0, acc_c2_0, acc_c3_0);
                                if (n_subs > 1) {
                                    uint32_t b0_1 = *reinterpret_cast<const uint32_t*>(dp1 + k_off);
                                    uint32_t b1_1 = *reinterpret_cast<const uint32_t*>(dp1 + k_off + 16);
                                    mma_m16n8k32_s8(acc_c0_1, acc_c1_1, acc_c2_1, acc_c3_1,
                                                    a0, a1, a2, a3, b0_1, b1_1,
                                                    acc_c0_1, acc_c1_1, acc_c2_1, acc_c3_1);
                                }
                            }
                        } else if (slice_mode == 2) {
                            #pragma unroll
                            for (int ks = 0; ks < 4; ++ks) {
                                uint32_t bq0 = qp[(ks * 8 + 4) * 32];
                                uint32_t bq1 = qp[(ks * 8 + 5) * 32];
                                uint32_t bq2 = qp[(ks * 8 + 6) * 32];
                                uint32_t bq3 = qp[(ks * 8 + 7) * 32];
                                int k_off = ks * 32 + tid_in_g * 4;
                                uint32_t b0_0 = *reinterpret_cast<const uint32_t*>(dp0 + k_off);
                                uint32_t b1_0 = *reinterpret_cast<const uint32_t*>(dp0 + k_off + 16);
                                mma_m16n8k32_s8(acc_d0_0, acc_d1_0, acc_d2_0, acc_d3_0,
                                                bq0, bq1, bq2, bq3, b0_0, b1_0,
                                                acc_d0_0, acc_d1_0, acc_d2_0, acc_d3_0);
                                if (n_subs > 1) {
                                    uint32_t b0_1 = *reinterpret_cast<const uint32_t*>(dp1 + k_off);
                                    uint32_t b1_1 = *reinterpret_cast<const uint32_t*>(dp1 + k_off + 16);
                                    mma_m16n8k32_s8(acc_d0_1, acc_d1_1, acc_d2_1, acc_d3_1,
                                                    bq0, bq1, bq2, bq3, b0_1, b1_1,
                                                    acc_d0_1, acc_d1_1, acc_d2_1, acc_d3_1);
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int ks = 0; ks < 4; ++ks) {
                                uint32_t a0 = qp[(ks * 8 + 0) * 32];
                                uint32_t a1 = qp[(ks * 8 + 1) * 32];
                                uint32_t a2 = qp[(ks * 8 + 2) * 32];
                                uint32_t a3 = qp[(ks * 8 + 3) * 32];
                                uint32_t bq0 = qp[(ks * 8 + 4) * 32];
                                uint32_t bq1 = qp[(ks * 8 + 5) * 32];
                                uint32_t bq2 = qp[(ks * 8 + 6) * 32];
                                uint32_t bq3 = qp[(ks * 8 + 7) * 32];

                                int k_off = ks * 32 + tid_in_g * 4;
                                uint32_t b0_0 = *reinterpret_cast<const uint32_t*>(dp0 + k_off);
                                uint32_t b1_0 = *reinterpret_cast<const uint32_t*>(dp0 + k_off + 16);
                                mma_m16n8k32_s8(acc_c0_0, acc_c1_0, acc_c2_0, acc_c3_0,
                                                a0, a1, a2, a3, b0_0, b1_0,
                                                acc_c0_0, acc_c1_0, acc_c2_0, acc_c3_0);
                                mma_m16n8k32_s8(acc_d0_0, acc_d1_0, acc_d2_0, acc_d3_0,
                                                bq0, bq1, bq2, bq3, b0_0, b1_0,
                                                acc_d0_0, acc_d1_0, acc_d2_0, acc_d3_0);

                                if (n_subs > 1) {
                                    uint32_t b0_1 = *reinterpret_cast<const uint32_t*>(dp1 + k_off);
                                    uint32_t b1_1 = *reinterpret_cast<const uint32_t*>(dp1 + k_off + 16);
                                    mma_m16n8k32_s8(acc_c0_1, acc_c1_1, acc_c2_1, acc_c3_1,
                                                    a0, a1, a2, a3, b0_1, b1_1,
                                                    acc_c0_1, acc_c1_1, acc_c2_1, acc_c3_1);
                                    mma_m16n8k32_s8(acc_d0_1, acc_d1_1, acc_d2_1, acc_d3_1,
                                                    bq0, bq1, bq2, bq3, b0_1, b1_1,
                                                    acc_d0_1, acc_d1_1, acc_d2_1, acc_d3_1);
                                }
                            }
                        }
                    }
                }
                if constexpr (WARPS_PER_CTA > 1) __syncthreads(); else __syncwarp();
            }

            if (active) {
                {
                    int j0 = tid_in_g * 2, j1 = j0 + 1;
                    float ds0 = sh_ds[j0], ds1 = sh_ds[j1];
                    float dr0 = sh_dr[j0], dr1 = sh_dr[j1];
                    if (ds0 > 0.0f) {
                        if (qn0 > 0.0f) { float dt = (float)acc_c0_0 * qs0 * ds0, er = qn0 * dr0; ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); }
                        if (qn1 > 0.0f) { float dt = (float)acc_c2_0 * qs1 * ds0, er = qn1 * dr0; ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(lb1, dt - er); }
                        if (qn2 > 0.0f) { float dt = (float)acc_d0_0 * qs2 * ds0, er = qn2 * dr0; ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); }
                        if (qn3 > 0.0f) { float dt = (float)acc_d2_0 * qs3 * ds0, er = qn3 * dr0; ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); }
                    }
                    if (ds1 > 0.0f) {
                        if (qn0 > 0.0f) { float dt = (float)acc_c1_0 * qs0 * ds1, er = qn0 * dr1; ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); }
                        if (qn1 > 0.0f) { float dt = (float)acc_c3_0 * qs1 * ds1, er = qn1 * dr1; ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(lb1, dt - er); }
                        if (qn2 > 0.0f) { float dt = (float)acc_d1_0 * qs2 * ds1, er = qn2 * dr1; ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); }
                        if (qn3 > 0.0f) { float dt = (float)acc_d3_0 * qs3 * ds1, er = qn3 * dr1; ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); }
                    }
                }
                if (n_subs > 1) {
                    int j0 = 8 + tid_in_g * 2, j1 = j0 + 1;
                    float ds0 = sh_ds[j0], ds1 = sh_ds[j1];
                    float dr0 = sh_dr[j0], dr1 = sh_dr[j1];
                    if (ds0 > 0.0f) {
                        if (qn0 > 0.0f) { float dt = (float)acc_c0_1 * qs0 * ds0, er = qn0 * dr0; ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); }
                        if (qn1 > 0.0f) { float dt = (float)acc_c2_1 * qs1 * ds0, er = qn1 * dr0; ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(lb1, dt - er); }
                        if (qn2 > 0.0f) { float dt = (float)acc_d0_1 * qs2 * ds0, er = qn2 * dr0; ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); }
                        if (qn3 > 0.0f) { float dt = (float)acc_d2_1 * qs3 * ds0, er = qn3 * dr0; ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); }
                    }
                    if (ds1 > 0.0f) {
                        if (qn0 > 0.0f) { float dt = (float)acc_c1_1 * qs0 * ds1, er = qn0 * dr1; ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); }
                        if (qn1 > 0.0f) { float dt = (float)acc_c3_1 * qs1 * ds1, er = qn1 * dr1; ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(lb1, dt - er); }
                        if (qn2 > 0.0f) { float dt = (float)acc_d1_1 * qs2 * ds1, er = qn2 * dr1; ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); }
                        if (qn3 > 0.0f) { float dt = (float)acc_d3_1 * qs3 * ds1, er = qn3 * dr1; ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); }
                    }
                }
            }
            if constexpr (WARPS_PER_CTA > 1) __syncthreads(); else __syncwarp();
        }

        if (active) {
            #define RED4(v) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1)); v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
            RED4(ub0); RED4(ub1); RED4(ub2); RED4(ub3);
            RED4(lb0); RED4(lb1); RED4(lb2); RED4(lb3);
            #undef RED4
            float sum_ub = ub0 + ub1 + ub2 + ub3;
            float sum_lb = lb0 + lb1 + lb2 + lb3;
            sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 4);
            sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 8);
            sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 16);
            sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 4);
            sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 8);
            sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 16);

            if (lane == 0) {
                int64_t idx = (int64_t)q * N_docs + d;
                float prev_ub = is_first_pass ? 0.0f : ub_accum[idx];
                float prev_lb = is_first_pass ? 0.0f : lb_accum[idx];
                float new_ub = prev_ub + sum_ub;
                float new_lb = prev_lb + sum_lb;
                ub_accum[idx] = new_ub;
                lb_accum[idx] = new_lb;
                if (do_prune) {
                    float r_val = rem_is_2d ? rem_bound[idx] : rem_b;
                    float ep_v = (eps_buf != nullptr) ? eps_buf[q] : eps_safe;
                    doc_active[idx] = (new_ub + r_val >= tau_val - ep_v) ? 1 : 0;
                }
            }
        }
    }
}




// ============================================================================
// M2 General-Dimension Cooperative 4-Warp SMEM-Query L2-Swizzled Kernel:
//   - Grid: dim3(B, ceil(N_docs / DOCS_PER_CTA)) -> q = blockIdx.x (fast axis)
//     so all B queries hit the exact same active document working set in L2!
//   - Block: 128 threads (4 warps) sharing 1 query preloaded ONCE into sh_q[32 KB],
//     cooperating across 32 doc tokens (8 doc tokens per warp) with zero L1 thrash!
// ============================================================================
template <int DOCS_PER_CTA>
__global__ void __launch_bounds__(128, 2) fused_tc_int8_gendim_coop4_kernel(
    const int8_t* __restrict__ q_pack_i8,   // uint32_t [B, P, n_dblks, 32, 32]
    const float* __restrict__ q_scales,     // [B, m_pad]
    const float* __restrict__ q_norms,      // [B, m_pad]
    const int8_t* __restrict__ d_int8,      // [total_pad_toks, D]
    const float* __restrict__ d_scales,     // [total_pad_toks]
    const float* __restrict__ d_rho,        // [total_pad_toks]
    const int64_t* __restrict__ d_pad_offs, // [N_docs + 1]
    const float* __restrict__ tau_buf,      // [B]
    float* __restrict__ ub_accum,           // [B, N_docs]
    float* __restrict__ lb_accum,           // [B, N_docs]
    const float* __restrict__ rem_bound,    // [B]
    uint8_t* __restrict__ doc_active,       // [B, N_docs]
    int B, int m_pad, int N_docs, int D, int q_tok_off, float eps_safe, int is_first_pass, int do_prune)
{
    int q = blockIdx.x;
    if (q >= B) return;
    int doc_base = blockIdx.y * DOCS_PER_CTA;
    int doc_end = min(doc_base + DOCS_PER_CTA, N_docs);

    int warp_id = threadIdx.x >> 5;   // 0..3
    int lane = threadIdx.x & 31;      // 0..31
    int group_id = lane >> 2;         // 0..7
    int tid_in_g = lane & 3;          // 0..3

    int n_dblks = D >> 7;
    int n_passes = m_pad >> 5;
    int pass_idx = q_tok_off >> 5;
    const uint32_t* q_pack_base = reinterpret_cast<const uint32_t*>(q_pack_i8)
                                  + ((int64_t)q * n_passes + pass_idx) * n_dblks * 1024;

    __shared__ uint32_t sh_q[8 * 1024];        // 32 KB: holds full D<=1024 query pass!
    __shared__ int8_t   sh_d[32 * SH_STRIDE];  // 4.5 KB: 32 doc tokens x 144 B
    __shared__ float    sh_ds[32];
    __shared__ float    sh_dr[32];
    __shared__ float    sh_ub[4][32];
    __shared__ float    sh_lb[4][32];

    // Preload query pass into shared memory ONCE per CTA (outside document loop!)
    if (n_dblks <= 8) {
        int q_words = n_dblks * 1024;
        for (int idx = threadIdx.x; idx < q_words; idx += 128) {
            sh_q[idx] = q_pack_base[idx];
        }
        __syncthreads();
    }

    int qtb = q * m_pad + q_tok_off;
    float qs0 = q_scales[qtb + group_id],      qs1 = q_scales[qtb + group_id + 8];
    float qs2 = q_scales[qtb + 16 + group_id], qs3 = q_scales[qtb + 24 + group_id];
    float qn0 = q_norms[qtb + group_id],       qn1 = q_norms[qtb + group_id + 8];
    float qn2 = q_norms[qtb + 16 + group_id],  qn3 = q_norms[qtb + 24 + group_id];
    float tau_val = do_prune ? tau_buf[q] : -1e9f;
    float rem_b   = do_prune ? rem_bound[q] : 0.0f;

    int warp_tok_base = warp_id * 8;

    for (int d = doc_base; d < doc_end; ++d) {
        if (doc_active[(int64_t)q * N_docs + d] == 0) continue;

        int64_t p_start = d_pad_offs[d];
        int64_t p_end = d_pad_offs[d + 1];

        float ub0 = (qn0 > 0.0f) ? -1e9f : 0.0f, lb0 = ub0;
        float ub1 = (qn1 > 0.0f) ? -1e9f : 0.0f, lb1 = ub1;
        float ub2 = (qn2 > 0.0f) ? -1e9f : 0.0f, lb2 = ub2;
        float ub3 = (qn3 > 0.0f) ? -1e9f : 0.0f, lb3 = ub3;

        for (int64_t t = p_start; t < p_end; t += 32) {
            int n_tok = (int)min((int64_t)32, p_end - t);
            if (threadIdx.x < n_tok) {
                sh_ds[threadIdx.x] = d_scales[t + threadIdx.x];
                sh_dr[threadIdx.x] = d_rho[t + threadIdx.x];
            } else if (threadIdx.x < 32) {
                sh_ds[threadIdx.x] = 0.0f;
                sh_dr[threadIdx.x] = 0.0f;
            }

            int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;

            for (int db = 0; db < n_dblks; ++db) {
                int d_off = db * 128;
                // 32 * 8 = 256 int4s loaded across 128 threads (2 int4s/thread, 100% coalesced)
                for (int idx = threadIdx.x; idx < n_tok * 8; idx += 128) {
                    int tok_i = idx >> 3;
                    int vec_i = idx & 7;
                    const int4* src = reinterpret_cast<const int4*>(d_int8 + (t + tok_i) * (int64_t)D + d_off);
                    *reinterpret_cast<int4*>(sh_d + tok_i * SH_STRIDE + vec_i * 16) = src[vec_i];
                }
                __syncthreads();

                if (warp_tok_base < n_tok) {
                    const uint32_t* qp = ((n_dblks <= 8) ? sh_q : q_pack_base) + db * 1024 + lane;
                    const int8_t* dp = sh_d + (warp_tok_base + group_id) * SH_STRIDE;

                    #pragma unroll
                    for (int ks = 0; ks < 4; ++ks) {
                        uint32_t a0  = qp[(ks * 8 + 0) * 32];
                        uint32_t a1  = qp[(ks * 8 + 1) * 32];
                        uint32_t a2  = qp[(ks * 8 + 2) * 32];
                        uint32_t a3  = qp[(ks * 8 + 3) * 32];
                        uint32_t bq0 = qp[(ks * 8 + 4) * 32];
                        uint32_t bq1 = qp[(ks * 8 + 5) * 32];
                        uint32_t bq2 = qp[(ks * 8 + 6) * 32];
                        uint32_t bq3 = qp[(ks * 8 + 7) * 32];

                        int k_off = ks * 32 + tid_in_g * 4;
                        uint32_t b0 = *reinterpret_cast<const uint32_t*>(dp + k_off);
                        uint32_t b1 = *reinterpret_cast<const uint32_t*>(dp + k_off + 16);
                        mma_m16n8k32_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
                        mma_m16n8k32_s8(d0, d1, d2, d3, bq0, bq1, bq2, bq3, b0, b1, d0, d1, d2, d3);
                    }
                }
                __syncthreads();
            }

            if (warp_tok_base < n_tok) {
                int j0 = warp_tok_base + tid_in_g * 2;
                int j1 = j0 + 1;
                float ds0 = sh_ds[j0], ds1 = sh_ds[j1];
                float dr0 = sh_dr[j0], dr1 = sh_dr[j1];
                if (ds0 > 0.0f) {
                    if (qn0 > 0.0f) { float dt = (float)c0 * qs0 * ds0, er = qn0 * dr0; ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); }
                    if (qn1 > 0.0f) { float dt = (float)c2 * qs1 * ds0, er = qn1 * dr0; ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(lb1, dt - er); }
                    if (qn2 > 0.0f) { float dt = (float)d0 * qs2 * ds0, er = qn2 * dr0; ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); }
                    if (qn3 > 0.0f) { float dt = (float)d2 * qs3 * ds0, er = qn3 * dr0; ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); }
                }
                if (ds1 > 0.0f) {
                    if (qn0 > 0.0f) { float dt = (float)c1 * qs0 * ds1, er = qn0 * dr1; ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); }
                    if (qn1 > 0.0f) { float dt = (float)c3 * qs1 * ds1, er = qn1 * dr1; ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(lb1, dt - er); }
                    if (qn2 > 0.0f) { float dt = (float)d1 * qs2 * ds1, er = qn2 * dr1; ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); }
                    if (qn3 > 0.0f) { float dt = (float)d3 * qs3 * ds1, er = qn3 * dr1; ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); }
                }
            }
            __syncthreads();
        }

        // Reduce within each 4-lane subgroup first
        #define RED4(v) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1)); v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
        RED4(ub0); RED4(ub1); RED4(ub2); RED4(ub3);
        RED4(lb0); RED4(lb1); RED4(lb2); RED4(lb3);
        #undef RED4

        if (tid_in_g == 0) {
            sh_ub[warp_id][group_id]      = ub0;
            sh_ub[warp_id][group_id + 8]  = ub1;
            sh_ub[warp_id][group_id + 16] = ub2;
            sh_ub[warp_id][group_id + 24] = ub3;
            sh_lb[warp_id][group_id]      = lb0;
            sh_lb[warp_id][group_id + 8]  = lb1;
            sh_lb[warp_id][group_id + 16] = lb2;
            sh_lb[warp_id][group_id + 24] = lb3;
        }
        __syncthreads();

        if (warp_id == 0) {
            float u_max = fmaxf(fmaxf(sh_ub[0][lane], sh_ub[1][lane]), fmaxf(sh_ub[2][lane], sh_ub[3][lane]));
            float l_max = fmaxf(fmaxf(sh_lb[0][lane], sh_lb[1][lane]), fmaxf(sh_lb[2][lane], sh_lb[3][lane]));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                u_max += __shfl_xor_sync(0xffffffff, u_max, off);
                l_max += __shfl_xor_sync(0xffffffff, l_max, off);
            }
            if (lane == 0) {
                int64_t idx = (int64_t)q * N_docs + d;
                float new_ub = (is_first_pass ? 0.0f : ub_accum[idx]) + u_max;
                float new_lb = (is_first_pass ? 0.0f : lb_accum[idx]) + l_max;
                ub_accum[idx] = new_ub;
                lb_accum[idx] = new_lb;
                if (do_prune) {
                    doc_active[idx] = (new_ub + rem_b >= tau_val - eps_safe) ? 1 : 0;
                }
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// M2 Bit-Sliced INT6 (4-bit High + 2-bit Low = 96 B/tok at D=128)
// ============================================================================
template <int DOCS_PER_CTA, int WARPS_PER_CTA>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32, (WARPS_PER_CTA == 1) ? 8 : 2) fused_tc_int6_pass32_kernel(
    const int8_t* __restrict__ q_int8,
    const float* __restrict__ q_scales,
    const float* __restrict__ q_norms,
    const uint8_t* __restrict__ d_hi4,
    const uint8_t* __restrict__ d_lo2,
    const float* __restrict__ d_scales,
    const float* __restrict__ d_rho,
    const int64_t* __restrict__ d_pad_offs,
    const float* __restrict__ tau_buf,
    float* __restrict__ ub_accum,
    float* __restrict__ lb_accum,
    const float* __restrict__ rem_bound,
    const float* __restrict__ eps_buf,
    uint8_t* __restrict__ doc_active,
    int B, int m_pad, int N_docs, int q_tok_off, float eps_safe, int is_first_pass, int do_prune)
{
    int doc_base = blockIdx.x * DOCS_PER_CTA;
    int q_base = blockIdx.y * WARPS_PER_CTA;
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;
    int tid_in_g = lane & 3;
    int q = q_base + warp_id;
    constexpr int BLOCK_TH = WARPS_PER_CTA * 32;

    __shared__ int8_t sh_d[32 * SH_STRIDE];
    __shared__ float sh_ds[32];
    __shared__ float sh_dr[32];
    __shared__ int any_warp_needs;

    uint32_t a0[4], a1[4], a2[4], a3[4];
    uint32_t b0_q[4], b1_q[4], b2_q[4], b3_q[4];
    float qs0 = 0.0f, qs1 = 0.0f, qs2 = 0.0f, qs3 = 0.0f;
    float qn0 = 0.0f, qn1 = 0.0f, qn2 = 0.0f, qn3 = 0.0f;
    float tau_val = -1e9f, rem_b = 0.0f, eps_val = eps_safe;
    bool q_valid = (q < B);

    if (q_valid) {
        int qtb = q * m_pad + q_tok_off;
        const int8_t* qp0 = q_int8 + (int64_t)qtb * 128;
        const int8_t* qp1 = qp0 + 16 * 128;
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            int k_off = ks * 32 + tid_in_g * 4;
            a0[ks] = *reinterpret_cast<const uint32_t*>(qp0 + group_id * 128 + k_off);
            a1[ks] = *reinterpret_cast<const uint32_t*>(qp0 + (group_id + 8) * 128 + k_off);
            a2[ks] = *reinterpret_cast<const uint32_t*>(qp0 + group_id * 128 + k_off + 16);
            a3[ks] = *reinterpret_cast<const uint32_t*>(qp0 + (group_id + 8) * 128 + k_off + 16);

            b0_q[ks] = *reinterpret_cast<const uint32_t*>(qp1 + group_id * 128 + k_off);
            b1_q[ks] = *reinterpret_cast<const uint32_t*>(qp1 + (group_id + 8) * 128 + k_off);
            b2_q[ks] = *reinterpret_cast<const uint32_t*>(qp1 + group_id * 128 + k_off + 16);
            b3_q[ks] = *reinterpret_cast<const uint32_t*>(qp1 + (group_id + 8) * 128 + k_off + 16);
        }
        qs0 = q_scales[qtb + group_id];      qs1 = q_scales[qtb + group_id + 8];
        qs2 = q_scales[qtb + 16 + group_id]; qs3 = q_scales[qtb + 24 + group_id];
        qn0 = q_norms[qtb + group_id];       qn1 = q_norms[qtb + group_id + 8];
        qn2 = q_norms[qtb + 16 + group_id];  qn3 = q_norms[qtb + 24 + group_id];
        if (do_prune) {
            tau_val = tau_buf[q];
            rem_b = rem_bound[q];
            if (eps_buf != nullptr) eps_val = eps_buf[q];
        }
    }

    int doc_end = min(doc_base + DOCS_PER_CTA, N_docs);
    for (int d = doc_base; d < doc_end; ++d) {
        bool active = q_valid && (doc_active[(int64_t)q * N_docs + d] != 0);
        if constexpr (WARPS_PER_CTA > 1) {
            if (threadIdx.x == 0) any_warp_needs = 0;
            __syncthreads();
            if (lane == 0 && active) atomicOr(&any_warp_needs, 1);
            __syncthreads();
            if (any_warp_needs == 0) continue;
        } else {
            if (!active) continue;
        }

        int64_t p_start = d_pad_offs[d];
        int64_t p_end = d_pad_offs[d + 1];

        float ub0 = (qn0 > 0.0f) ? -1e9f : 0.0f, lb0 = ub0;
        float ub1 = (qn1 > 0.0f) ? -1e9f : 0.0f, lb1 = ub1;
        float ub2 = (qn2 > 0.0f) ? -1e9f : 0.0f, lb2 = ub2;
        float ub3 = (qn3 > 0.0f) ? -1e9f : 0.0f, lb3 = ub3;

        for (int64_t t = p_start; t < p_end; t += 32) {
            int n_tok = (int)min((int64_t)32, p_end - t);
            for (int idx = threadIdx.x; idx < n_tok * 16; idx += BLOCK_TH) {
                int tok_i = idx >> 4;
                int g16 = idx & 15;
                uint32_t hi = reinterpret_cast<const uint32_t*>(d_hi4 + (t + tok_i) * 64)[g16];
                uint32_t lo = (uint32_t)(reinterpret_cast<const uint16_t*>(d_lo2 + (t + tok_i) * 32)[g16]);
                uint32_t* dst32 = reinterpret_cast<uint32_t*>(sh_d + tok_i * SH_STRIDE + g16 * 8);
                uint32_t p0 = 0u, p1 = 0u;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    uint32_t v0 = ((((hi >> (b * 4)) & 0xFu) << 2) | ((lo >> (b * 2)) & 0x3u)) - 32u;
                    uint32_t v1 = ((((hi >> ((b + 4) * 4)) & 0xFu) << 2) | ((lo >> ((b + 4) * 2)) & 0x3u)) - 32u;
                    p0 |= (v0 & 0xFFu) << (b * 8);
                    p1 |= (v1 & 0xFFu) << (b * 8);
                }
                dst32[0] = p0;
                dst32[1] = p1;
            }
            if (threadIdx.x < n_tok) {
                sh_ds[threadIdx.x] = d_scales[t + threadIdx.x];
                sh_dr[threadIdx.x] = d_rho[t + threadIdx.x];
            } else if (threadIdx.x < 32) {
                sh_ds[threadIdx.x] = 0.0f;
                sh_dr[threadIdx.x] = 0.0f;
            }
            if constexpr (WARPS_PER_CTA > 1) __syncthreads(); else __syncwarp();

            if (active) {
                for (int n_sub = 0; n_sub < n_tok; n_sub += 8) {
                    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    const int8_t* dp = sh_d + (n_sub + group_id) * SH_STRIDE;
                    #pragma unroll
                    for (int ks = 0; ks < 4; ++ks) {
                        int k_off = ks * 32 + tid_in_g * 4;
                        uint32_t b0 = *reinterpret_cast<const uint32_t*>(dp + k_off);
                        uint32_t b1 = *reinterpret_cast<const uint32_t*>(dp + k_off + 16);
                        mma_m16n8k32_s8(c0, c1, c2, c3, a0[ks], a1[ks], a2[ks], a3[ks], b0, b1, c0, c1, c2, c3);
                        mma_m16n8k32_s8(d0, d1, d2, d3, b0_q[ks], b1_q[ks], b2_q[ks], b3_q[ks], b0, b1, d0, d1, d2, d3);
                    }
                    int j0 = n_sub + tid_in_g * 2;
                    int j1 = j0 + 1;
                    float ds0 = sh_ds[j0], ds1 = sh_ds[j1];
                    float dr0 = sh_dr[j0], dr1 = sh_dr[j1];

                    if (ds0 > 0.0f) {
                        if (qn0 > 0.0f) { float dt = (float)c0 * qs0 * ds0, er = qn0 * dr0; ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); }
                        if (qn1 > 0.0f) { float dt = (float)c2 * qs1 * ds0, er = qn1 * dr0; ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(lb1, dt - er); }
                        if (qn2 > 0.0f) { float dt = (float)d0 * qs2 * ds0, er = qn2 * dr0; ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); }
                        if (qn3 > 0.0f) { float dt = (float)d2 * qs3 * ds0, er = qn3 * dr0; ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); }
                    }
                    if (ds1 > 0.0f) {
                        if (qn0 > 0.0f) { float dt = (float)c1 * qs0 * ds1, er = qn0 * dr1; ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); }
                        if (qn1 > 0.0f) { float dt = (float)c3 * qs1 * ds1, er = qn1 * dr1; ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(lb1, dt - er); }
                        if (qn2 > 0.0f) { float dt = (float)d1 * qs2 * ds1, er = qn2 * dr1; ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); }
                        if (qn3 > 0.0f) { float dt = (float)d3 * qs3 * ds1, er = qn3 * dr1; ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); }
                    }
                }
            }
            if constexpr (WARPS_PER_CTA > 1) __syncthreads(); else __syncwarp();
        }

        if (active) {
            #define RED4(v) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1)); v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
            RED4(ub0); RED4(ub1); RED4(ub2); RED4(ub3);
            RED4(lb0); RED4(lb1); RED4(lb2); RED4(lb3);
            #undef RED4
            float sum_ub = ub0 + ub1 + ub2 + ub3;
            float sum_lb = lb0 + lb1 + lb2 + lb3;
            sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 4);
            sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 8);
            sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 16);
            sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 4);
            sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 8);
            sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 16);

            if (lane == 0) {
                int64_t idx = (int64_t)q * N_docs + d;
                float prev_ub = is_first_pass ? 0.0f : ub_accum[idx];
                float prev_lb = is_first_pass ? 0.0f : lb_accum[idx];
                float new_ub = prev_ub + sum_ub;
                float new_lb = prev_lb + sum_lb;
                ub_accum[idx] = new_ub;
                lb_accum[idx] = new_lb;
                if (do_prune) {
                    doc_active[idx] = (new_ub + rem_b >= tau_val - eps_val) ? 1 : 0;
                }
            }
        }
    }
}



// ============================================================================
// M2 Tier-1 INT4 Pass-32 Kernel (64 B/tok = 4x compression for 8GB VRAM)
// ============================================================================
template <int DOCS_PER_CTA, int WARPS_PER_CTA>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32, (WARPS_PER_CTA == 1) ? 8 : 2) fused_tc_int4_pass32_kernel(
    const int8_t* __restrict__ q_int8,
    const float* __restrict__ q_scales,
    const float* __restrict__ q_norms,
    const uint8_t* __restrict__ d_int4,
    const float* __restrict__ d_scales,
    const float* __restrict__ d_rho,
    const int64_t* __restrict__ d_pad_offs,
    const float* __restrict__ tau_buf,
    float* __restrict__ ub_accum,
    float* __restrict__ lb_accum,
    const float* __restrict__ rem_bound,
    const float* __restrict__ eps_buf,
    uint8_t* __restrict__ doc_active,
    int B, int m_pad, int N_docs, int q_tok_off, float eps_safe, int is_first_pass, int do_prune)
{
    int doc_base = blockIdx.x * DOCS_PER_CTA;
    int q_base = blockIdx.y * WARPS_PER_CTA;
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;
    int tid_in_g = lane & 3;
    int q = q_base + warp_id;
    constexpr int BLOCK_TH = WARPS_PER_CTA * 32;

    __shared__ int8_t sh_d[32 * SH_STRIDE];
    __shared__ float sh_ds[32];
    __shared__ float sh_dr[32];
    __shared__ int any_warp_needs;

    uint32_t a0[4], a1[4], a2[4], a3[4];
    uint32_t b0_q[4], b1_q[4], b2_q[4], b3_q[4];
    float qs0 = 0.0f, qs1 = 0.0f, qs2 = 0.0f, qs3 = 0.0f;
    float qn0 = 0.0f, qn1 = 0.0f, qn2 = 0.0f, qn3 = 0.0f;
    float tau_val = -1e9f, rem_b = 0.0f, eps_val = eps_safe;
    bool q_valid = (q < B);

    if (q_valid) {
        int qtb = q * m_pad + q_tok_off;
        const int8_t* qp0 = q_int8 + (int64_t)qtb * 128;
        const int8_t* qp1 = qp0 + 16 * 128;
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            int k_off = ks * 32 + tid_in_g * 4;
            a0[ks] = *reinterpret_cast<const uint32_t*>(qp0 + group_id * 128 + k_off);
            a1[ks] = *reinterpret_cast<const uint32_t*>(qp0 + (group_id + 8) * 128 + k_off);
            a2[ks] = *reinterpret_cast<const uint32_t*>(qp0 + group_id * 128 + k_off + 16);
            a3[ks] = *reinterpret_cast<const uint32_t*>(qp0 + (group_id + 8) * 128 + k_off + 16);

            b0_q[ks] = *reinterpret_cast<const uint32_t*>(qp1 + group_id * 128 + k_off);
            b1_q[ks] = *reinterpret_cast<const uint32_t*>(qp1 + (group_id + 8) * 128 + k_off);
            b2_q[ks] = *reinterpret_cast<const uint32_t*>(qp1 + group_id * 128 + k_off + 16);
            b3_q[ks] = *reinterpret_cast<const uint32_t*>(qp1 + (group_id + 8) * 128 + k_off + 16);
        }
        qs0 = q_scales[qtb + group_id];      qs1 = q_scales[qtb + group_id + 8];
        qs2 = q_scales[qtb + 16 + group_id]; qs3 = q_scales[qtb + 24 + group_id];
        qn0 = q_norms[qtb + group_id];       qn1 = q_norms[qtb + group_id + 8];
        qn2 = q_norms[qtb + 16 + group_id];  qn3 = q_norms[qtb + 24 + group_id];
        if (do_prune) {
            tau_val = tau_buf[q];
            rem_b = rem_bound[q];
            if (eps_buf != nullptr) eps_val = eps_buf[q];
        }
    }

    int doc_end = min(doc_base + DOCS_PER_CTA, N_docs);
    for (int d = doc_base; d < doc_end; ++d) {
        bool active = q_valid && (doc_active[(int64_t)q * N_docs + d] != 0);
        if constexpr (WARPS_PER_CTA > 1) {
            if (threadIdx.x == 0) any_warp_needs = 0;
            __syncthreads();
            if (lane == 0 && active) atomicOr(&any_warp_needs, 1);
            __syncthreads();
            if (any_warp_needs == 0) continue;
        } else {
            if (!active) continue;
        }

        int64_t p_start = d_pad_offs[d];
        int64_t p_end = d_pad_offs[d + 1];

        float ub0 = (qn0 > 0.0f) ? -1e9f : 0.0f, lb0 = ub0;
        float ub1 = (qn1 > 0.0f) ? -1e9f : 0.0f, lb1 = ub1;
        float ub2 = (qn2 > 0.0f) ? -1e9f : 0.0f, lb2 = ub2;
        float ub3 = (qn3 > 0.0f) ? -1e9f : 0.0f, lb3 = ub3;

        for (int64_t t = p_start; t < p_end; t += 32) {
            int n_tok = (int)min((int64_t)32, p_end - t);
            for (int idx = threadIdx.x; idx < n_tok * 16; idx += BLOCK_TH) {
                int tok_i = idx >> 4;
                int u32_i = idx & 15;
                uint32_t packed = reinterpret_cast<const uint32_t*>(d_int4 + (t + tok_i) * 64)[u32_i];
                uint32_t* dst32 = reinterpret_cast<uint32_t*>(sh_d + tok_i * SH_STRIDE + u32_i * 8);
                uint32_t p0 = 0u, p1 = 0u;
                #pragma unroll
                for (int b = 0; b < 2; ++b) {
                    uint8_t bv0 = (packed >> (b * 8)) & 0xFFu;
                    uint8_t bv1 = (packed >> ((b + 2) * 8)) & 0xFFu;
                    p0 |= (((uint32_t)((bv0 & 0x0Fu) - 8) & 0xFFu) << (b * 16))
                        | (((uint32_t)((bv0 >> 4) - 8) & 0xFFu) << (b * 16 + 8));
                    p1 |= (((uint32_t)((bv1 & 0x0Fu) - 8) & 0xFFu) << (b * 16))
                        | (((uint32_t)((bv1 >> 4) - 8) & 0xFFu) << (b * 16 + 8));
                }
                dst32[0] = p0;
                dst32[1] = p1;
            }
            if (threadIdx.x < n_tok) {
                sh_ds[threadIdx.x] = d_scales[t + threadIdx.x];
                sh_dr[threadIdx.x] = d_rho[t + threadIdx.x];
            } else if (threadIdx.x < 32) {
                sh_ds[threadIdx.x] = 0.0f;
                sh_dr[threadIdx.x] = 0.0f;
            }
            if constexpr (WARPS_PER_CTA > 1) __syncthreads(); else __syncwarp();

            if (active) {
                for (int n_sub = 0; n_sub < n_tok; n_sub += 8) {
                    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    const int8_t* dp = sh_d + (n_sub + group_id) * SH_STRIDE;
                    #pragma unroll
                    for (int ks = 0; ks < 4; ++ks) {
                        int k_off = ks * 32 + tid_in_g * 4;
                        uint32_t b0 = *reinterpret_cast<const uint32_t*>(dp + k_off);
                        uint32_t b1 = *reinterpret_cast<const uint32_t*>(dp + k_off + 16);
                        mma_m16n8k32_s8(c0, c1, c2, c3, a0[ks], a1[ks], a2[ks], a3[ks], b0, b1, c0, c1, c2, c3);
                        mma_m16n8k32_s8(d0, d1, d2, d3, b0_q[ks], b1_q[ks], b2_q[ks], b3_q[ks], b0, b1, d0, d1, d2, d3);
                    }
                    int j0 = n_sub + tid_in_g * 2;
                    int j1 = j0 + 1;
                    float ds0 = sh_ds[j0], ds1 = sh_ds[j1];
                    float dr0 = sh_dr[j0], dr1 = sh_dr[j1];

                    if (ds0 > 0.0f) {
                        if (qn0 > 0.0f) { float dt = (float)c0 * qs0 * ds0, er = qn0 * dr0; ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); }
                        if (qn1 > 0.0f) { float dt = (float)c2 * qs1 * ds0, er = qn1 * dr0; ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(lb1, dt - er); }
                        if (qn2 > 0.0f) { float dt = (float)d0 * qs2 * ds0, er = qn2 * dr0; ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); }
                        if (qn3 > 0.0f) { float dt = (float)d2 * qs3 * ds0, er = qn3 * dr0; ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); }
                    }
                    if (ds1 > 0.0f) {
                        if (qn0 > 0.0f) { float dt = (float)c1 * qs0 * ds1, er = qn0 * dr1; ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); }
                        if (qn1 > 0.0f) { float dt = (float)c3 * qs1 * ds1, er = qn1 * dr1; ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(lb1, dt - er); }
                        if (qn2 > 0.0f) { float dt = (float)d1 * qs2 * ds1, er = qn2 * dr1; ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); }
                        if (qn3 > 0.0f) { float dt = (float)d3 * qs3 * ds1, er = qn3 * dr1; ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); }
                    }
                }
            }
            if constexpr (WARPS_PER_CTA > 1) __syncthreads(); else __syncwarp();
        }

        if (active) {
            #define RED4(v) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1)); v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
            RED4(ub0); RED4(ub1); RED4(ub2); RED4(ub3);
            RED4(lb0); RED4(lb1); RED4(lb2); RED4(lb3);
            #undef RED4
            float sum_ub = ub0 + ub1 + ub2 + ub3;
            float sum_lb = lb0 + lb1 + lb2 + lb3;
            sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 4);
            sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 8);
            sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 16);
            sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 4);
            sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 8);
            sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 16);

            if (lane == 0) {
                int64_t idx = (int64_t)q * N_docs + d;
                float prev_ub = is_first_pass ? 0.0f : ub_accum[idx];
                float prev_lb = is_first_pass ? 0.0f : lb_accum[idx];
                float new_ub = prev_ub + sum_ub;
                float new_lb = prev_lb + sum_lb;
                ub_accum[idx] = new_ub;
                lb_accum[idx] = new_lb;
                if (do_prune) {
                    doc_active[idx] = (new_ub + rem_b >= tau_val - eps_val) ? 1 : 0;
                }
            }
        }
    }
}



// ============================================================================
// Mode B Candidate Rerank Kernel (General D % 128 == 0, fast-pathed for D=128)
// ============================================================================
__global__ void __launch_bounds__(128, 4) fused_tc_rerank_int8_kernel(
    const int8_t* __restrict__ q_int8,      // [B, m_pad, D]
    const float* __restrict__ q_scales,     // [B, m_pad]
    const float* __restrict__ q_norms,      // [B, m_pad]
    const int8_t* __restrict__ d_int8,      // [total_pad_toks, D]
    const float* __restrict__ d_scales,     // [total_pad_toks]
    const float* __restrict__ d_rho,        // [total_pad_toks]
    const int64_t* __restrict__ d_pad_offs, // [N_docs + 1]
    const int32_t* __restrict__ cand_ids,   // [B, C]
    float* __restrict__ out_ub,             // [B, C]
    float* __restrict__ out_lb,             // [B, C]
    int B, int C, int m_pad, int D, int q_tok_off, int is_first_pass)
{
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;
    int tid_in_g = lane & 3;
    int c = blockIdx.x * 4 + warp_id;
    int q = blockIdx.y;
    if (c >= C || q >= B) return;

    int doc = cand_ids[(int64_t)q * C + c];
    if (doc < 0) return;

    __shared__ int8_t sh_d_all[4 * 8 * SH_STRIDE];
    __shared__ float sh_ds_all[4 * 8];
    __shared__ float sh_dr_all[4 * 8];

    int8_t* sh_d = sh_d_all + warp_id * 8 * SH_STRIDE;
    float* sh_ds = sh_ds_all + warp_id * 8;
    float* sh_dr = sh_dr_all + warp_id * 8;

    int qtb = q * m_pad + q_tok_off;
    float qs0 = q_scales[qtb + group_id],      qs1 = q_scales[qtb + group_id + 8];
    float qs2 = q_scales[qtb + 16 + group_id], qs3 = q_scales[qtb + 24 + group_id];
    float qn0 = q_norms[qtb + group_id],       qn1 = q_norms[qtb + group_id + 8];
    float qn2 = q_norms[qtb + 16 + group_id],  qn3 = q_norms[qtb + 24 + group_id];

    float ub0 = (qn0 > 0.0f) ? -1e9f : 0.0f, lb0 = ub0;
    float ub1 = (qn1 > 0.0f) ? -1e9f : 0.0f, lb1 = ub1;
    float ub2 = (qn2 > 0.0f) ? -1e9f : 0.0f, lb2 = ub2;
    float ub3 = (qn3 > 0.0f) ? -1e9f : 0.0f, lb3 = ub3;

    int64_t p_start = d_pad_offs[doc];
    int64_t p_end = d_pad_offs[doc + 1];
    int n_dblks = D >> 7;

    for (int64_t t = p_start; t < p_end; t += 8) {
        int n_tok = (int)min((int64_t)8, p_end - t);
        if (lane < n_tok) {
            sh_ds[lane] = d_scales[t + lane];
            sh_dr[lane] = d_rho[t + lane];
        } else if (lane < 8) {
            sh_ds[lane] = 0.0f;
            sh_dr[lane] = 0.0f;
        }

        int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;

        for (int db = 0; db < n_dblks; ++db) {
            int d_off = db * 128;
            for (int idx = lane; idx < n_tok * 8; idx += 32) {
                int tok_i = idx >> 3;
                int vec_i = idx & 7;
                const int4* src = reinterpret_cast<const int4*>(d_int8 + (t + tok_i) * (int64_t)D + d_off);
                *reinterpret_cast<int4*>(sh_d + tok_i * SH_STRIDE + vec_i * 16) = src[vec_i];
            }
            __syncwarp();

            const int8_t* qp0 = q_int8 + (int64_t)qtb * D + d_off;
            const int8_t* qp1 = qp0 + (int64_t)16 * D;
            const int8_t* dp = sh_d + group_id * SH_STRIDE;
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                int k_off = ks * 32 + tid_in_g * 4;
                uint32_t a0 = *reinterpret_cast<const uint32_t*>(qp0 + (int64_t)group_id * D + k_off);
                uint32_t a1 = *reinterpret_cast<const uint32_t*>(qp0 + (int64_t)(group_id + 8) * D + k_off);
                uint32_t a2 = *reinterpret_cast<const uint32_t*>(qp0 + (int64_t)group_id * D + k_off + 16);
                uint32_t a3 = *reinterpret_cast<const uint32_t*>(qp0 + (int64_t)(group_id + 8) * D + k_off + 16);

                uint32_t bq0 = *reinterpret_cast<const uint32_t*>(qp1 + (int64_t)group_id * D + k_off);
                uint32_t bq1 = *reinterpret_cast<const uint32_t*>(qp1 + (int64_t)(group_id + 8) * D + k_off);
                uint32_t bq2 = *reinterpret_cast<const uint32_t*>(qp1 + (int64_t)group_id * D + k_off + 16);
                uint32_t bq3 = *reinterpret_cast<const uint32_t*>(qp1 + (int64_t)(group_id + 8) * D + k_off + 16);

                uint32_t b0 = *reinterpret_cast<const uint32_t*>(dp + k_off);
                uint32_t b1 = *reinterpret_cast<const uint32_t*>(dp + k_off + 16);
                mma_m16n8k32_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
                mma_m16n8k32_s8(d0, d1, d2, d3, bq0, bq1, bq2, bq3, b0, b1, d0, d1, d2, d3);
            }
            __syncwarp();
        }

        int j0 = tid_in_g * 2;
        int j1 = j0 + 1;
        float ds0 = sh_ds[j0], ds1 = sh_ds[j1];
        float dr0 = sh_dr[j0], dr1 = sh_dr[j1];

        if (ds0 > 0.0f) {
            if (qn0 > 0.0f) { float dt = (float)c0 * qs0 * ds0, er = qn0 * dr0; ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); }
            if (qn1 > 0.0f) { float dt = (float)c2 * qs1 * ds0, er = qn1 * dr0; ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(ub1, dt - er); }
            if (qn2 > 0.0f) { float dt = (float)d0 * qs2 * ds0, er = qn2 * dr0; ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); }
            if (qn3 > 0.0f) { float dt = (float)d2 * qs3 * ds0, er = qn3 * dr0; ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); }
        }
        if (ds1 > 0.0f) {
            if (qn0 > 0.0f) { float dt = (float)c1 * qs0 * ds1, er = qn0 * dr1; ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); }
            if (qn1 > 0.0f) { float dt = (float)c3 * qs1 * ds1, er = qn1 * dr1; ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(lb1, dt - er); }
            if (qn2 > 0.0f) { float dt = (float)d1 * qs2 * ds1, er = qn2 * dr1; ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); }
            if (qn3 > 0.0f) { float dt = (float)d3 * qs3 * ds1, er = qn3 * dr1; ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); }
        }
        __syncwarp();
    }

    #define RED4(v) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1)); v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    RED4(ub0); RED4(ub1); RED4(ub2); RED4(ub3);
    RED4(lb0); RED4(lb1); RED4(lb2); RED4(lb3);
    #undef RED4
    float sum_ub = ub0 + ub1 + ub2 + ub3;
    float sum_lb = lb0 + lb1 + lb2 + lb3;
    sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 4);
    sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 8);
    sum_ub += __shfl_xor_sync(0xffffffff, sum_ub, 16);
    sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 4);
    sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 8);
    sum_lb += __shfl_xor_sync(0xffffffff, sum_lb, 16);

    if (lane == 0) {
        int64_t idx = (int64_t)q * C + c;
        out_ub[idx] = (is_first_pass ? 0.0f : out_ub[idx]) + sum_ub;
        out_lb[idx] = (is_first_pass ? 0.0f : out_lb[idx]) + sum_lb;
    }
}

// ============================================================================
// M2 Stage 3A: Pair-Parallel Vectorized Exact FP16->FP32 Verification Kernel
// Supports vectorized int4 (half8) reads for ANY D % 128 == 0 (incl. D=1024!)
// ============================================================================
__global__ void level2_exact_verify_kernel(
    const int32_t* __restrict__ surv_q,
    const int32_t* __restrict__ surv_doc,
    const half* __restrict__ q_tokens,
    const int* __restrict__ q_lens,
    const half* __restrict__ d_tokens,
    const int64_t* __restrict__ d_offs,
    float* __restrict__ out_scores,
    int num_pairs, int m_max, int D)
{
    int pair_idx = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (pair_idx >= num_pairs) return;

    int q = surv_q[pair_idx];
    int doc = surv_doc[pair_idx];
    int mq = q_lens[q];
    int64_t ds = d_offs[doc];
    int64_t de = d_offs[doc + 1];

    const half* q_base = q_tokens + (int64_t)q * m_max * D;
    float total = 0.0f;

    if (D == 128) {
        int tok_sub = lane >> 4;
        int dim_off = (lane & 15) * 8;
        int i = 0;
        for (; i + 3 < mq; i += 4) {
            int4 qv0 = *reinterpret_cast<const int4*>(q_base + (i + 0) * 128 + dim_off);
            int4 qv1 = *reinterpret_cast<const int4*>(q_base + (i + 1) * 128 + dim_off);
            int4 qv2 = *reinterpret_cast<const int4*>(q_base + (i + 2) * 128 + dim_off);
            int4 qv3 = *reinterpret_cast<const int4*>(q_base + (i + 3) * 128 + dim_off);
            const half* qh0 = reinterpret_cast<const half*>(&qv0);
            const half* qh1 = reinterpret_cast<const half*>(&qv1);
            const half* qh2 = reinterpret_cast<const half*>(&qv2);
            const half* qh3 = reinterpret_cast<const half*>(&qv3);

            float b0 = -1e9f, b1 = -1e9f, b2 = -1e9f, b3 = -1e9f;
            for (int64_t j_base = ds; j_base < de; j_base += 2) {
                int64_t j = j_base + tok_sub;
                float s0 = -1e9f, s1 = -1e9f, s2 = -1e9f, s3 = -1e9f;
                if (j < de) {
                    int4 dv = *reinterpret_cast<const int4*>(d_tokens + j * 128 + dim_off);
                    const half* dh = reinterpret_cast<const half*>(&dv);
                    s0 = 0.0f; s1 = 0.0f; s2 = 0.0f; s3 = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        float d_val = __half2float(dh[k]);
                        s0 += __half2float(qh0[k]) * d_val;
                        s1 += __half2float(qh1[k]) * d_val;
                        s2 += __half2float(qh2[k]) * d_val;
                        s3 += __half2float(qh3[k]) * d_val;
                    }
                }
                for (int off = 8; off > 0; off >>= 1) {
                    s0 += __shfl_xor_sync(0xffffffff, s0, off);
                    s1 += __shfl_xor_sync(0xffffffff, s1, off);
                    s2 += __shfl_xor_sync(0xffffffff, s2, off);
                    s3 += __shfl_xor_sync(0xffffffff, s3, off);
                }
                if (j < de) {
                    b0 = fmaxf(b0, s0); b1 = fmaxf(b1, s1);
                    b2 = fmaxf(b2, s2); b3 = fmaxf(b3, s3);
                }
            }
            b0 = fmaxf(b0, __shfl_xor_sync(0xffffffff, b0, 16));
            b1 = fmaxf(b1, __shfl_xor_sync(0xffffffff, b1, 16));
            b2 = fmaxf(b2, __shfl_xor_sync(0xffffffff, b2, 16));
            b3 = fmaxf(b3, __shfl_xor_sync(0xffffffff, b3, 16));
            if (lane == 0) total += (b0 + b1 + b2 + b3);
        }
        for (; i < mq; ++i) {
            int4 qv = *reinterpret_cast<const int4*>(q_base + i * 128 + dim_off);
            const half* qh = reinterpret_cast<const half*>(&qv);
            float b = -1e9f;
            for (int64_t j_base = ds; j_base < de; j_base += 2) {
                int64_t j = j_base + tok_sub;
                float s = -1e9f;
                if (j < de) {
                    int4 dv = *reinterpret_cast<const int4*>(d_tokens + j * 128 + dim_off);
                    const half* dh = reinterpret_cast<const half*>(&dv);
                    s = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) s += __half2float(qh[k]) * __half2float(dh[k]);
                }
                for (int off = 8; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
                if (j < de) b = fmaxf(b, s);
            }
            b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, 16));
            if (lane == 0) total += b;
        }
    } else if ((D & 127) == 0) {
        int tok_sub = lane >> 4;
        int dim_off = (lane & 15) * 8;
        int n_dblks = D >> 7;
        int i = 0;
        for (; i + 7 < mq; i += 8) {
            float b0 = -1e9f, b1 = -1e9f, b2 = -1e9f, b3 = -1e9f;
            float b4 = -1e9f, b5 = -1e9f, b6 = -1e9f, b7 = -1e9f;
            for (int64_t j_base = ds; j_base < de; j_base += 2) {
                int64_t j = j_base + tok_sub;
                float s0 = -1e9f, s1 = -1e9f, s2 = -1e9f, s3 = -1e9f;
                float s4 = -1e9f, s5 = -1e9f, s6 = -1e9f, s7 = -1e9f;
                if (j < de) {
                    const half* dp = d_tokens + j * (int64_t)D;
                    s0 = 0.0f; s1 = 0.0f; s2 = 0.0f; s3 = 0.0f;
                    s4 = 0.0f; s5 = 0.0f; s6 = 0.0f; s7 = 0.0f;
                    for (int db = 0; db < n_dblks; ++db) {
                        int d_off = db * 128 + dim_off;
                        int4 dv = *reinterpret_cast<const int4*>(dp + d_off);
                        const half* dh = reinterpret_cast<const half*>(&dv);
                        float df[8];
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) df[k] = __half2float(dh[k]);

                        #define ACC_Q8(q_idx, s_var) { \
                            int4 qv = *reinterpret_cast<const int4*>(q_base + (int64_t)(i + q_idx) * D + d_off); \
                            const half* qh = reinterpret_cast<const half*>(&qv); \
                            for (int k = 0; k < 8; ++k) s_var += __half2float(qh[k]) * df[k]; \
                        }
                        ACC_Q8(0, s0); ACC_Q8(1, s1); ACC_Q8(2, s2); ACC_Q8(3, s3);
                        ACC_Q8(4, s4); ACC_Q8(5, s5); ACC_Q8(6, s6); ACC_Q8(7, s7);
                        #undef ACC_Q8
                    }
                }
                for (int off = 8; off > 0; off >>= 1) {
                    s0 += __shfl_xor_sync(0xffffffff, s0, off);
                    s1 += __shfl_xor_sync(0xffffffff, s1, off);
                    s2 += __shfl_xor_sync(0xffffffff, s2, off);
                    s3 += __shfl_xor_sync(0xffffffff, s3, off);
                    s4 += __shfl_xor_sync(0xffffffff, s4, off);
                    s5 += __shfl_xor_sync(0xffffffff, s5, off);
                    s6 += __shfl_xor_sync(0xffffffff, s6, off);
                    s7 += __shfl_xor_sync(0xffffffff, s7, off);
                }
                if (j < de) {
                    b0 = fmaxf(b0, s0); b1 = fmaxf(b1, s1); b2 = fmaxf(b2, s2); b3 = fmaxf(b3, s3);
                    b4 = fmaxf(b4, s4); b5 = fmaxf(b5, s5); b6 = fmaxf(b6, s6); b7 = fmaxf(b7, s7);
                }
            }
            b0 = fmaxf(b0, __shfl_xor_sync(0xffffffff, b0, 16));
            b1 = fmaxf(b1, __shfl_xor_sync(0xffffffff, b1, 16));
            b2 = fmaxf(b2, __shfl_xor_sync(0xffffffff, b2, 16));
            b3 = fmaxf(b3, __shfl_xor_sync(0xffffffff, b3, 16));
            b4 = fmaxf(b4, __shfl_xor_sync(0xffffffff, b4, 16));
            b5 = fmaxf(b5, __shfl_xor_sync(0xffffffff, b5, 16));
            b6 = fmaxf(b6, __shfl_xor_sync(0xffffffff, b6, 16));
            b7 = fmaxf(b7, __shfl_xor_sync(0xffffffff, b7, 16));
            if (lane == 0) total += (b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7);
        }
        for (; i < mq; ++i) {
            const half* qp = q_base + (int64_t)i * D;
            float b = -1e9f;
            for (int64_t j_base = ds; j_base < de; j_base += 2) {
                int64_t j = j_base + tok_sub;
                float s = -1e9f;
                if (j < de) {
                    const half* dp = d_tokens + j * (int64_t)D;
                    s = 0.0f;
                    for (int db = 0; db < n_dblks; ++db) {
                        int d_off = db * 128 + dim_off;
                        int4 qv = *reinterpret_cast<const int4*>(qp + d_off);
                        int4 dv = *reinterpret_cast<const int4*>(dp + d_off);
                        const half* qh = reinterpret_cast<const half*>(&qv);
                        const half* dh = reinterpret_cast<const half*>(&dv);
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            s += __half2float(qh[k]) * __half2float(dh[k]);
                        }
                    }
                }
                for (int off = 8; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
                if (j < de) b = fmaxf(b, s);
            }
            b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, 16));
            if (lane == 0) total += b;
        }
    } else {
        for (int i = 0; i < mq; ++i) {
            const half* qp = q_base + (int64_t)i * D;
            float best = -1e9f;
            for (int64_t j = ds; j < de; ++j) {
                const half* dp = d_tokens + j * D;
                float acc = 0.0f;
                for (int d = lane; d < D; d += 32) {
                    acc += __half2float(qp[d]) * __half2float(dp[d]);
                }
                float dot = warp_reduce_sum(acc);
                best = fmaxf(best, dot);
            }
            if (lane == 0) total += best;
        }
    }
    if (lane == 0) out_scores[pair_idx] = total;
}

// ============================================================================
// M2 Stage 3B: Document-Grouped Vectorized Exact Verify Kernel (D == 128)
// ============================================================================
__global__ void __launch_bounds__(256, 2) level2_doc_grouped_verify_kernel(
    const int32_t* __restrict__ surv_docs,  // [n_sd]
    const uint8_t* __restrict__ surv_mask,  // [B, N_docs]
    const half* __restrict__ q_tokens,      // [B, m_max, 128]
    const int* __restrict__ q_lens,         // [B]
    const half* __restrict__ d_tokens,      // [N_toks, 128]
    const int64_t* __restrict__ d_offs,     // [N_docs + 1]
    float* __restrict__ out_scores,         // [B, N_docs]
    int n_sd, int B, int m_max, int N_docs)
{
    int sd_idx = blockIdx.x;
    if (sd_idx >= n_sd) return;
    int doc = surv_docs[sd_idx];
    int64_t ds = d_offs[doc];
    int64_t de = d_offs[doc + 1];

    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    constexpr int NUM_WARPS = 8;

    __shared__ half sh_doc[64 * 128];     // 16 KB
    __shared__ float sh_best[4096];       // 16 KB (max B * m_max <= 4096)
    __shared__ int sh_qs[128];
    __shared__ int sh_cnt;

    if (threadIdx.x == 0) {
        int cnt = 0;
        for (int q = 0; q < B && cnt < 128; ++q) {
            if (surv_mask[(int64_t)q * N_docs + doc]) {
                sh_qs[cnt++] = q;
            }
        }
        sh_cnt = cnt;
    }
    __syncthreads();
    int cnt = sh_cnt;
    if (cnt == 0) return;

    int total_qtoks = cnt * m_max;
    for (int idx = threadIdx.x; idx < total_qtoks; idx += 256) {
        sh_best[idx] = -1e9f;
    }
    __syncthreads();

    int tok_sub = lane >> 4;
    int dim_off = (lane & 15) * 8;

    for (int64_t t = ds; t < de; t += 64) {
        int n_t = (int)min((int64_t)64, de - t);
        for (int idx = threadIdx.x; idx < n_t * 16; idx += 256) {
            int ti = idx >> 4;
            int vi = idx & 15;
            *reinterpret_cast<int4*>(sh_doc + ti * 128 + vi * 8) =
                *reinterpret_cast<const int4*>(d_tokens + (t + ti) * 128 + vi * 8);
        }
        __syncthreads();

        for (int qtok_i = warp_id; qtok_i < total_qtoks; qtok_i += NUM_WARPS) {
            int iq = qtok_i / m_max;
            int i = qtok_i - iq * m_max;
            int q = sh_qs[iq];
            if (i >= q_lens[q]) continue;

            int4 qv = *reinterpret_cast<const int4*>(q_tokens + ((int64_t)q * m_max + i) * 128 + dim_off);
            const half* qh = reinterpret_cast<const half*>(&qv);
            float b = -1e9f;
            for (int j_base = 0; j_base < n_t; j_base += 2) {
                int j = j_base + tok_sub;
                float s = -1e9f;
                if (j < n_t) {
                    int4 dv = *reinterpret_cast<const int4*>(sh_doc + j * 128 + dim_off);
                    const half* dh = reinterpret_cast<const half*>(&dv);
                    s = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) s += __half2float(qh[k]) * __half2float(dh[k]);
                }
                for (int off = 8; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
                if (j < n_t) b = fmaxf(b, s);
            }
            b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, 16));
            if (lane == 0) {
                sh_best[qtok_i] = fmaxf(sh_best[qtok_i], b);
            }
        }
        __syncthreads();
    }

    for (int iq = warp_id; iq < cnt; iq += NUM_WARPS) {
        int q = sh_qs[iq];
        int mq = q_lens[q];
        float acc = 0.0f;
        for (int i = lane; i < mq; i += 32) {
            acc += sh_best[iq * m_max + i];
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            out_scores[(int64_t)q * N_docs + doc] = acc;
        }
    }
}

extern "C" {
void launch_unpadded_cublas_reduce(
    const half* sim, const int64_t* d_offs, const int* q_lens, float* out_scores,
    int B, int m, int N_docs, int64_t total_toks, cudaStream_t stream)
{
    int warps_per_block = 8;
    dim3 block(warps_per_block * WARP_SIZE);
    dim3 grid((N_docs + warps_per_block - 1) / warps_per_block, B);
    unpadded_cublas_reduce_kernel<<<grid, block, 0, stream>>>(
        sim, d_offs, q_lens, out_scores, B, m, N_docs, total_toks);
}

void launch_fused_tc_int8_pass32(
    const int8_t* q_int8, const float* q_scales, const float* q_norms,
    const int8_t* d_int8, const float* d_scales, const float* d_rho,
    const int64_t* d_pad_offs, const float* tau_buf,
    float* ub_accum, float* lb_accum, const float* rem_bound, const float* eps_buf, uint8_t* doc_active,
    int B, int m_pad, int N_docs, int D, int q_tok_off, float eps_safe,
    int is_first_pass, int do_prune, int force_w1, int rem_is_2d, int slice_mode, cudaStream_t stream)
{
    if (D == 128) {
        if (B == 1 && !force_w1) {
            constexpr int DPC = 1; constexpr int W = 1;
            fused_tc_int8_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
        } else if (force_w1) {
            constexpr int DPC = 4; constexpr int W = 1;
            fused_tc_int8_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, B), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
        } else if (B <= 4) {
            if (N_docs <= 32768) {
                constexpr int DPC = 4; constexpr int W = 4;
                fused_tc_int8_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
                    q_int8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                    B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
            } else {
                constexpr int DPC = 8; constexpr int W = 4;
                fused_tc_int8_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
                    q_int8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                    B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
            }
        } else if (B <= 8) {
            if (N_docs <= 32768) {
                constexpr int DPC = 4; constexpr int W = 8;
                fused_tc_int8_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
                    q_int8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                    B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
            } else {
                constexpr int DPC = 16; constexpr int W = 8;
                fused_tc_int8_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
                    q_int8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                    B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
            }
        } else {
            if (N_docs <= 32768) {
                constexpr int DPC = 4; constexpr int W = 16;
                fused_tc_int8_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, (B + W - 1) / W), dim3(W * 32), 0, stream>>>(
                    q_int8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                    B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
            } else {
                constexpr int DPC = 16; constexpr int W = 16;
                fused_tc_int8_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, (B + W - 1) / W), dim3(W * 32), 0, stream>>>(
                    q_int8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                    B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
            }
        }
    } else {
        if (force_w1) {
            constexpr int DPC = 4; constexpr int W = 1;
            fused_tc_int8_gendim_pass32_kernel<DPC, W, false><<<dim3((N_docs + DPC - 1) / DPC, B), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, D, q_tok_off, eps_safe, is_first_pass, do_prune, rem_is_2d, slice_mode);
        } else if (B == 1) {
            constexpr int DPC = 1; constexpr int W = 1;
            fused_tc_int8_gendim_pass32_kernel<DPC, W, true><<<dim3(1, (N_docs + DPC - 1) / DPC), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, D, q_tok_off, eps_safe, is_first_pass, do_prune, rem_is_2d, slice_mode);
        } else if (B <= 4) {
            constexpr int DPC = 4; constexpr int W = 4;
            fused_tc_int8_gendim_pass32_kernel<DPC, W, true><<<dim3(1, (N_docs + DPC - 1) / DPC), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, D, q_tok_off, eps_safe, is_first_pass, do_prune, rem_is_2d, slice_mode);
        } else if (B <= 8) {
            constexpr int DPC = 4; constexpr int W = 8;
            fused_tc_int8_gendim_pass32_kernel<DPC, W, true><<<dim3(1, (N_docs + DPC - 1) / DPC), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, D, q_tok_off, eps_safe, is_first_pass, do_prune, rem_is_2d, slice_mode);
        } else {
            constexpr int DPC = 4; constexpr int W = 16;
            fused_tc_int8_gendim_pass32_kernel<DPC, W, true><<<dim3((B + W - 1) / W, (N_docs + DPC - 1) / DPC), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, D, q_tok_off, eps_safe, is_first_pass, do_prune, rem_is_2d, slice_mode);
        }
    }
}

void launch_fused_tc_int8_hd_pass16(
    const int8_t* q_pack_i8, const float* q_scales, const float* q_norms,
    const int8_t* d_int8, const float* d_scales, const float* d_rho,
    const int64_t* d_pad_offs, const float* tau_buf,
    float* ub_accum, float* lb_accum, const float* rem_bound, const float* eps_buf, uint8_t* doc_active,
    int B, int m_pad, int N_docs, int D, int q_tok_off, float eps_safe,
    int is_first_pass, int do_prune, int rem_is_2d, int force_w1, cudaStream_t stream)
{
    int dev_id = 0;
    cudaGetDevice(&dev_id);
    int max_smem = 49152;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id);

    int n_dblks = D >> 7;
    auto calc_smem = [&](int wq) -> size_t {
        size_t q_words = (n_dblks <= 8) ? ((size_t)n_dblks * 512) : 0;
        size_t sz = (size_t)wq * q_words * 4 + 32 * SH_STRIDE + 256 + (size_t)wq * 640 + (size_t)(wq + 4) * 4;
        return (sz + 127) & ~((size_t)127);
    };

    if (force_w1 || B == 1) {
        constexpr int DPC = 4; constexpr int WQ = 1;
        size_t smem = calc_smem(WQ);
        auto kfn = fused_tc_int8_hd_pass16_shm_kernel<DPC, WQ>;
        if (smem > 49152) cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        kfn<<<dim3((B + WQ - 1) / WQ, (N_docs + DPC - 1) / DPC), dim3(128), smem, stream>>>(
            q_pack_i8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf,
            ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
            B, m_pad, N_docs, D, q_tok_off, eps_safe, is_first_pass, do_prune, rem_is_2d);
    } else if (B >= 8 && calc_smem(8) <= (size_t)max_smem) {
        constexpr int DPC = 4; constexpr int WQ = 8;
        size_t smem = calc_smem(WQ);
        auto kfn = fused_tc_int8_hd_pass16_shm_kernel<DPC, WQ>;
        if (smem > 49152) cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        kfn<<<dim3((B + WQ - 1) / WQ, (N_docs + DPC - 1) / DPC), dim3(128), smem, stream>>>(
            q_pack_i8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf,
            ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
            B, m_pad, N_docs, D, q_tok_off, eps_safe, is_first_pass, do_prune, rem_is_2d);
    } else if (B >= 4 && calc_smem(4) <= (size_t)max_smem) {
        constexpr int DPC = 4; constexpr int WQ = 4;
        size_t smem = calc_smem(WQ);
        auto kfn = fused_tc_int8_hd_pass16_shm_kernel<DPC, WQ>;
        if (smem > 49152) cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        kfn<<<dim3((B + WQ - 1) / WQ, (N_docs + DPC - 1) / DPC), dim3(128), smem, stream>>>(
            q_pack_i8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf,
            ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
            B, m_pad, N_docs, D, q_tok_off, eps_safe, is_first_pass, do_prune, rem_is_2d);
    } else {
        constexpr int DPC = 4; constexpr int WQ = 2;
        size_t smem = calc_smem(WQ);
        auto kfn = fused_tc_int8_hd_pass16_shm_kernel<DPC, WQ>;
        if (smem > 49152) cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        kfn<<<dim3((B + WQ - 1) / WQ, (N_docs + DPC - 1) / DPC), dim3(128), smem, stream>>>(
            q_pack_i8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, tau_buf,
            ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
            B, m_pad, N_docs, D, q_tok_off, eps_safe, is_first_pass, do_prune, rem_is_2d);
    }
}

void launch_fused_tc_fp16_maxsim(
    const half* q_pad, const int* q_lens, const half* d_tokens, const int64_t* d_offs,
    float* out_scores, int B, int m_pad, int N_docs, int D, cudaStream_t stream)
{
    if (B == 1) {
        constexpr int DPC = 2; constexpr int W = 1;
        fused_tc_fp16_maxsim_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
            q_pad, q_lens, d_tokens, d_offs, out_scores, B, m_pad, N_docs, D);
    } else if (B <= 4) {
        constexpr int DPC = 4; constexpr int W = 4;
        fused_tc_fp16_maxsim_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
            q_pad, q_lens, d_tokens, d_offs, out_scores, B, m_pad, N_docs, D);
    } else if (D == 128 && B >= 16) {
        if (N_docs <= 32768) {
            constexpr int DPC = 4; constexpr int W = 16;
            fused_tc_fp16_maxsim_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, (B + W - 1) / W), dim3(W * 32), 0, stream>>>(
                q_pad, q_lens, d_tokens, d_offs, out_scores, B, m_pad, N_docs, D);
        } else {
            constexpr int DPC = 16; constexpr int W = 16;
            fused_tc_fp16_maxsim_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, (B + W - 1) / W), dim3(W * 32), 0, stream>>>(
                q_pad, q_lens, d_tokens, d_offs, out_scores, B, m_pad, N_docs, D);
        }
    } else {
        constexpr int DPC = 8; constexpr int W = 8;
        fused_tc_fp16_maxsim_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, (B + W - 1) / W), dim3(W * 32), 0, stream>>>(
            q_pad, q_lens, d_tokens, d_offs, out_scores, B, m_pad, N_docs, D);
    }
}

void launch_fused_tc_int4_pass32(
    const int8_t* q_int8, const float* q_scales, const float* q_norms,
    const uint8_t* d_int4, const float* d_scales, const float* d_rho,
    const int64_t* d_pad_offs, const float* tau_buf,
    float* ub_accum, float* lb_accum, const float* rem_bound, const float* eps_buf, uint8_t* doc_active,
    int B, int m_pad, int N_docs, int q_tok_off, float eps_safe, int is_first_pass, int do_prune, cudaStream_t stream)
{
    if (B == 1) {
        constexpr int DPC = 1; constexpr int W = 1;
        fused_tc_int4_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
            q_int8, q_scales, q_norms, d_int4, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
            B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
    } else if (B <= 4) {
        if (N_docs <= 32768) {
            constexpr int DPC = 4; constexpr int W = 4;
            fused_tc_int4_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_int4, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
        } else {
            constexpr int DPC = 8; constexpr int W = 4;
            fused_tc_int4_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_int4, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
        }
    } else if (B <= 8) {
        if (N_docs <= 32768) {
            constexpr int DPC = 4; constexpr int W = 8;
            fused_tc_int4_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_int4, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
        } else {
            constexpr int DPC = 16; constexpr int W = 8;
            fused_tc_int4_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_int4, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
        }
    } else {
        if (N_docs <= 32768) {
            constexpr int DPC = 4; constexpr int W = 16;
            fused_tc_int4_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, (B + W - 1) / W), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_int4, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
        } else {
            constexpr int DPC = 16; constexpr int W = 16;
            fused_tc_int4_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, (B + W - 1) / W), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_int4, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
        }
    }
}

void launch_fused_tc_int6_pass32(
    const int8_t* q_int8, const float* q_scales, const float* q_norms,
    const uint8_t* d_hi4, const uint8_t* d_lo2, const float* d_scales, const float* d_rho,
    const int64_t* d_pad_offs, const float* tau_buf,
    float* ub_accum, float* lb_accum, const float* rem_bound, const float* eps_buf, uint8_t* doc_active,
    int B, int m_pad, int N_docs, int q_tok_off, float eps_safe, int is_first_pass, int do_prune, cudaStream_t stream)
{
    if (B == 1) {
        constexpr int DPC = 1; constexpr int W = 1;
        fused_tc_int6_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
            q_int8, q_scales, q_norms, d_hi4, d_lo2, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
            B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
    } else if (B <= 4) {
        if (N_docs <= 32768) {
            constexpr int DPC = 4; constexpr int W = 4;
            fused_tc_int6_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_hi4, d_lo2, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
        } else {
            constexpr int DPC = 8; constexpr int W = 4;
            fused_tc_int6_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_hi4, d_lo2, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
        }
    } else if (B <= 8) {
        if (N_docs <= 32768) {
            constexpr int DPC = 4; constexpr int W = 8;
            fused_tc_int6_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_hi4, d_lo2, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
        } else {
            constexpr int DPC = 16; constexpr int W = 8;
            fused_tc_int6_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, 1), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_hi4, d_lo2, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
        }
    } else {
        if (N_docs <= 32768) {
            constexpr int DPC = 4; constexpr int W = 16;
            fused_tc_int6_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, (B + W - 1) / W), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_hi4, d_lo2, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
        } else {
            constexpr int DPC = 16; constexpr int W = 16;
            fused_tc_int6_pass32_kernel<DPC, W><<<dim3((N_docs + DPC - 1) / DPC, (B + W - 1) / W), dim3(W * 32), 0, stream>>>(
                q_int8, q_scales, q_norms, d_hi4, d_lo2, d_scales, d_rho, d_pad_offs, tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
                B, m_pad, N_docs, q_tok_off, eps_safe, is_first_pass, do_prune);
        }
    }
}

void launch_fused_tc_rerank_int8(
    const int8_t* q_int8, const float* q_scales, const float* q_norms,
    const int8_t* d_int8, const float* d_scales, const float* d_rho,
    const int64_t* d_pad_offs, const int32_t* cand_ids,
    float* out_ub, float* out_lb, int B, int C, int m_pad, int D, int q_tok_off, int is_first_pass, cudaStream_t stream)
{
    dim3 block(128);
    dim3 grid((C + 3) / 4, B);
    fused_tc_rerank_int8_kernel<<<grid, block, 0, stream>>>(
        q_int8, q_scales, q_norms, d_int8, d_scales, d_rho, d_pad_offs, cand_ids,
        out_ub, out_lb, B, C, m_pad, D, q_tok_off, is_first_pass);
}

void launch_level2_exact_verify(
    const int32_t* surv_q, const int32_t* surv_doc,
    const half* q_tokens, const int* q_lens,
    const half* d_tokens, const int64_t* d_offs,
    float* out_scores, int num_pairs, int m_max, int D, cudaStream_t stream)
{
    if (num_pairs <= 0) return;
    int warps_per_block = 4;
    dim3 block(warps_per_block * WARP_SIZE);
    dim3 grid((num_pairs + warps_per_block - 1) / warps_per_block);
    level2_exact_verify_kernel<<<grid, block, 0, stream>>>(
        surv_q, surv_doc, q_tokens, q_lens, d_tokens, d_offs, out_scores, num_pairs, m_max, D);
}

// ============================================================================
// M2 Operator P2 (High-Dimension D >= 256 L2-Tiled Tensor-Core IMMA Bound Reduce):
// Vectorized int4/float4 L2-resident MaxSim upper/lower bound reduction
// ============================================================================
__global__ void __launch_bounds__(256, 4) int8_gemm_bound_reduce_kernel(
    const int32_t* __restrict__ sim_i32,    // [B * s_len, chunk_toks]
    const float* __restrict__ q_scales,     // [B, m_pad]
    const float* __restrict__ q_norms,      // [B, m_pad]
    const float* __restrict__ d_scales,     // [total_pad_toks]
    const float* __restrict__ d_rho,        // [total_pad_toks]
    const int64_t* __restrict__ d_pad_offs, // [N_docs + 1]
    const float* __restrict__ tau_buf,      // [B]
    float* __restrict__ ub_accum,           // [B, N_docs]
    float* __restrict__ lb_accum,           // [B, N_docs]
    const float* __restrict__ rem_bound,    // [B] or [B, N_docs]
    const float* __restrict__ eps_buf,      // [B] or nullptr
    uint8_t* __restrict__ doc_active,       // [B, N_docs]
    int B, int s_len, int m_pad, int N_docs, int doc_start, int nd, int chunk_toks,
    int64_t ts, int q_tok_off, float eps_safe, int is_first_pass, int do_prune, int rem_is_2d)
{
    int doc_local = blockIdx.x * 8 + (threadIdx.x >> 5);
    int q = blockIdx.y;
    int lane = threadIdx.x & 31;
    if (doc_local >= nd || q >= B) return;

    int d = doc_start + doc_local;
    int64_t out_idx = (int64_t)q * N_docs + d;
    if (doc_active[out_idx] == 0) return;

    int rel_s = (int)(d_pad_offs[d] - ts);
    int rel_e = (int)(d_pad_offs[d + 1] - ts);
    int vec_s = rel_s >> 2;
    int vec_e = rel_e >> 2;

    int qtb = q * m_pad + q_tok_off;
    const float4* ds_v = reinterpret_cast<const float4*>(d_scales + ts);
    const float4* dr_v = reinterpret_cast<const float4*>(d_rho + ts);

    float sum_ub = 0.0f, sum_lb = 0.0f;

    for (int i = 0; i < s_len; i += 4) {
        float qs0 = q_scales[qtb + i + 0], qn0 = q_norms[qtb + i + 0];
        float qs1 = q_scales[qtb + i + 1], qn1 = q_norms[qtb + i + 1];
        float qs2 = q_scales[qtb + i + 2], qn2 = q_norms[qtb + i + 2];
        float qs3 = q_scales[qtb + i + 3], qn3 = q_norms[qtb + i + 3];

        float ub0 = (qn0 > 0.0f) ? -1e9f : 0.0f, lb0 = ub0;
        float ub1 = (qn1 > 0.0f) ? -1e9f : 0.0f, lb1 = ub1;
        float ub2 = (qn2 > 0.0f) ? -1e9f : 0.0f, lb2 = ub2;
        float ub3 = (qn3 > 0.0f) ? -1e9f : 0.0f, lb3 = ub3;

        const int4* r0 = reinterpret_cast<const int4*>(sim_i32 + (int64_t)(q * s_len + i + 0) * chunk_toks);
        const int4* r1 = reinterpret_cast<const int4*>(sim_i32 + (int64_t)(q * s_len + i + 1) * chunk_toks);
        const int4* r2 = reinterpret_cast<const int4*>(sim_i32 + (int64_t)(q * s_len + i + 2) * chunk_toks);
        const int4* r3 = reinterpret_cast<const int4*>(sim_i32 + (int64_t)(q * s_len + i + 3) * chunk_toks);

        for (int v = vec_s + lane; v < vec_e; v += 32) {
            float4 ds4 = ds_v[v];
            float4 dr4 = dr_v[v];
            int4 c0 = r0[v];
            int4 c1 = r1[v];
            int4 c2 = r2[v];
            int4 c3 = r3[v];

            #define UPD_GEMM_LANE(dsc, drc, v0, v1, v2, v3) \
                if ((dsc) > 0.0f) { \
                    if (qn0 > 0.0f) { float dt = (float)(v0) * qs0 * (dsc), er = qn0 * (drc); ub0 = fmaxf(ub0, dt + er); lb0 = fmaxf(lb0, dt - er); } \
                    if (qn1 > 0.0f) { float dt = (float)(v1) * qs1 * (dsc), er = qn1 * (drc); ub1 = fmaxf(ub1, dt + er); lb1 = fmaxf(lb1, dt - er); } \
                    if (qn2 > 0.0f) { float dt = (float)(v2) * qs2 * (dsc), er = qn2 * (drc); ub2 = fmaxf(ub2, dt + er); lb2 = fmaxf(lb2, dt - er); } \
                    if (qn3 > 0.0f) { float dt = (float)(v3) * qs3 * (dsc), er = qn3 * (drc); ub3 = fmaxf(ub3, dt + er); lb3 = fmaxf(lb3, dt - er); } \
                }
            UPD_GEMM_LANE(ds4.x, dr4.x, c0.x, c1.x, c2.x, c3.x);
            UPD_GEMM_LANE(ds4.y, dr4.y, c0.y, c1.y, c2.y, c3.y);
            UPD_GEMM_LANE(ds4.z, dr4.z, c0.z, c1.z, c2.z, c3.z);
            UPD_GEMM_LANE(ds4.w, dr4.w, c0.w, c1.w, c2.w, c3.w);
            #undef UPD_GEMM_LANE
        }

        ub0 = warp_reduce_max(ub0); lb0 = warp_reduce_max(lb0);
        ub1 = warp_reduce_max(ub1); lb1 = warp_reduce_max(lb1);
        ub2 = warp_reduce_max(ub2); lb2 = warp_reduce_max(lb2);
        ub3 = warp_reduce_max(ub3); lb3 = warp_reduce_max(lb3);
        if (lane == 0) {
            sum_ub += (ub0 + ub1 + ub2 + ub3);
            sum_lb += (lb0 + lb1 + lb2 + lb3);
        }
    }

    if (lane == 0) {
        float new_ub = (is_first_pass ? 0.0f : ub_accum[out_idx]) + sum_ub;
        float new_lb = (is_first_pass ? 0.0f : lb_accum[out_idx]) + sum_lb;
        ub_accum[out_idx] = new_ub;
        lb_accum[out_idx] = new_lb;
        if (do_prune) {
            float rem_v = rem_is_2d ? rem_bound[out_idx] : rem_bound[q];
            float ep_v = (eps_buf != nullptr) ? eps_buf[q] : eps_safe;
            doc_active[out_idx] = (new_ub + rem_v >= tau_buf[q] - ep_v) ? 1 : 0;
        }
    }
}

void launch_level2_doc_grouped_verify(
    const int32_t* surv_docs, const uint8_t* surv_mask,
    const half* q_tokens, const int* q_lens,
    const half* d_tokens, const int64_t* d_offs,
    float* out_scores, int n_sd, int B, int m_max, int N_docs, cudaStream_t stream)
{
    if (n_sd <= 0) return;
    level2_doc_grouped_verify_kernel<<<dim3(n_sd), dim3(256), 0, stream>>>(
        surv_docs, surv_mask, q_tokens, q_lens, d_tokens, d_offs, out_scores, n_sd, B, m_max, N_docs);
}

void set_l2_persisting_window(void* ptr, size_t num_bytes, cudaStream_t stream)
{
    static bool s_inited[8] = {false};
    static size_t s_max_win[8] = {0};

    int dev_id = 0;
    cudaGetDevice(&dev_id);
    if (dev_id < 0 || dev_id >= 8) return;

    if (!s_inited[dev_id]) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev_id);
        if (prop.persistingL2CacheMaxSize > 0) {
            size_t reserve = min((size_t)(28ULL * 1024ULL * 1024ULL), (size_t)prop.persistingL2CacheMaxSize);
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, reserve);
            s_max_win[dev_id] = min(reserve, (size_t)prop.accessPolicyMaxWindowSize);
        }
        s_inited[dev_id] = true;
    }
    if (s_max_win[dev_id] == 0) return;

    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));
    if (ptr != nullptr && num_bytes > 0) {
        attr.accessPolicyWindow.base_ptr = ptr;
        attr.accessPolicyWindow.num_bytes = min(num_bytes, s_max_win[dev_id]);
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    } else {
        attr.accessPolicyWindow.num_bytes = 0;
    }
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
}

void launch_int8_gemm_bound_reduce(
    const int32_t* sim_i32, const float* q_scales, const float* q_norms,
    const float* d_scales, const float* d_rho, const int64_t* d_pad_offs,
    const float* tau_buf, float* ub_accum, float* lb_accum, const float* rem_bound,
    const float* eps_buf, uint8_t* doc_active, int B, int s_len, int m_pad, int N_docs,
    int doc_start, int nd, int chunk_toks, int64_t ts, int q_tok_off, float eps_safe,
    int is_first_pass, int do_prune, int rem_is_2d, cudaStream_t stream)
{
    if (nd <= 0 || B <= 0) return;
    dim3 block(256);
    dim3 grid((nd + 7) / 8, B);
    int8_gemm_bound_reduce_kernel<<<grid, block, 0, stream>>>(
        sim_i32, q_scales, q_norms, d_scales, d_rho, d_pad_offs,
        tau_buf, ub_accum, lb_accum, rem_bound, eps_buf, doc_active,
        B, s_len, m_pad, N_docs, doc_start, nd, chunk_toks, ts, q_tok_off,
        eps_safe, is_first_pass, do_prune, rem_is_2d);
}

// ============================================================================
// V6 Kernel Group 1: Fused 7-Stage Butterfly FWHT-128 + INT8 Query Quantizer
// Eliminates ~14 PyTorch small-kernel launches at B=1..64 (D <= 1024, D % 128 == 0)
// ============================================================================
__global__ void __launch_bounds__(128, 4) prepare_queries_hadamard_int8_kernel(
    const half* __restrict__ q_fp16_pad,  // [B, m_max, D]
    const int32_t* __restrict__ q_lens,   // [B]
    int8_t* __restrict__ q_int8_out,      // [B, m_pad, D]
    float* __restrict__ q_scales_out,     // [B, m_pad]
    float* __restrict__ q_norms_out,      // [B, m_pad]
    float* __restrict__ eps_buf_out,      // [B]
    float max_d_norm, int B, int m_max, int m_pad, int D)
{
    int q = blockIdx.x;
    if (q >= B) return;
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int mq = q_lens[q];
    int n_dblks = D >> 7;

    __shared__ float sh_rot[4][1024];
    __shared__ float sh_werr[4];
    __shared__ float sh_wnorm[4];

    float w_err_sum = 0.0f;
    float w_norm_sum = 0.0f;
    constexpr float INV_SQRT128 = 0.08838834764831845f; // 1.0 / sqrt(128.0)

    for (int tok = warp_id; tok < m_pad; tok += 4) {
        int64_t out_tok_idx = (int64_t)q * m_pad + tok;
        if (tok >= mq || tok >= m_max) {
            for (int d = lane; d < D; d += 32) {
                q_int8_out[out_tok_idx * D + d] = 0;
            }
            if (lane == 0) {
                q_scales_out[out_tok_idx] = 1e-6f / 127.0f;
                q_norms_out[out_tok_idx] = 0.0f;
            }
        } else {
            const half* q_src = q_fp16_pad + ((int64_t)q * m_max + tok) * D;
            float* buf = sh_rot[warp_id];
            float norm_sq = 0.0f;
            float abs_max = 0.0f;

            for (int db = 0; db < n_dblks; ++db) {
                float* blk = buf + db * 128;
                const half* sblk = q_src + db * 128;
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    int idx = lane * 4 + c;
                    float v = __half2float(sblk[idx]);
                    norm_sq += v * v;
                    blk[idx] = v;
                }
                __syncwarp();

                #pragma unroll
                for (int s = 0; s < 7; ++s) {
                    int k = 1 << s;
                    int p0 = lane;
                    int lo0 = ((p0 >> s) << (s + 1)) | (p0 & (k - 1));
                    int hi0 = lo0 | k;
                    float u0 = blk[lo0], v0 = blk[hi0];
                    blk[lo0] = u0 + v0;
                    blk[hi0] = u0 - v0;

                    int p1 = lane + 32;
                    int lo1 = ((p1 >> s) << (s + 1)) | (p1 & (k - 1));
                    int hi1 = lo1 | k;
                    float u1 = blk[lo1], v1 = blk[hi1];
                    blk[lo1] = u1 + v1;
                    blk[hi1] = u1 - v1;
                    __syncwarp();
                }

                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    int idx = lane * 4 + c;
                    float rv = blk[idx] * INV_SQRT128;
                    blk[idx] = rv;
                    abs_max = fmaxf(abs_max, fabsf(rv));
                }
                __syncwarp();
            }

            norm_sq = warp_reduce_sum(norm_sq);
            abs_max = warp_reduce_max(abs_max);
            float q_norm = sqrtf(norm_sq);
            float sc = fmaxf(abs_max, 1e-6f) / 127.0f;
            float inv_sc = 1.0f / sc;

            float err_sq = 0.0f;
            for (int d = lane; d < D; d += 32) {
                float rv = buf[d];
                float qi = fminf(fmaxf(nearbyintf(rv * inv_sc), -127.0f), 127.0f);
                q_int8_out[out_tok_idx * D + d] = static_cast<int8_t>(static_cast<int>(qi));
                float diff = rv - qi * sc;
                err_sq += diff * diff;
            }
            err_sq = warp_reduce_sum(err_sq);
            if (lane == 0) {
                q_scales_out[out_tok_idx] = sc;
                q_norms_out[out_tok_idx] = q_norm;
                w_err_sum += sqrtf(err_sq);
                w_norm_sum += q_norm;
            }
            __syncwarp();
        }
    }

    if (lane == 0) {
        sh_werr[warp_id] = w_err_sum;
        sh_wnorm[warp_id] = w_norm_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot_err = sh_werr[0] + sh_werr[1] + sh_werr[2] + sh_werr[3];
        float tot_nrm = sh_wnorm[0] + sh_wnorm[1] + sh_wnorm[2] + sh_wnorm[3];
        eps_buf_out[q] = tot_err * max_d_norm + 2e-3f * tot_nrm + 1e-3f;
    }
}

void launch_prepare_queries_hadamard_int8(
    const half* q_fp16_pad, const int32_t* q_lens,
    int8_t* q_int8_out, float* q_scales_out, float* q_norms_out, float* eps_buf_out,
    float max_d_norm, int B, int m_max, int m_pad, int D, cudaStream_t stream)
{
    if (B <= 0) return;
    prepare_queries_hadamard_int8_kernel<<<dim3(B), dim3(128), 0, stream>>>(
        q_fp16_pad, q_lens, q_int8_out, q_scales_out, q_norms_out, eps_buf_out,
        max_d_norm, B, m_max, m_pad, D);
}

// ============================================================================
// V6 Kernel Group 2: Zero-Host-Sync Fused Stage-2 Compaction + Scatter Verify
// Eliminates mid-pipeline torch.nonzero() cudaStreamSynchronize stall!
// ============================================================================
__global__ void prepare_scores_and_compact_survivors_kernel(
    const uint8_t* __restrict__ doc_active,  // [B, N_docs]
    float* __restrict__ ub_scores,           // [B, N_docs] (in: ub_accum, out: -1e9f)
    const float* __restrict__ tau_buf,       // [B]
    const float* __restrict__ eps_buf,       // [B]
    int32_t* __restrict__ surv_q,            // [max_surv]
    int32_t* __restrict__ surv_doc,          // [max_surv]
    int32_t* __restrict__ d_num_surv,        // [1]
    int max_surv, int B, int N_docs)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y;
    int lane = threadIdx.x & 31;
    if (q >= B) return;

    bool is_surv = false;
    if (d < N_docs) {
        int64_t idx = (int64_t)q * N_docs + d;
        uint8_t act = doc_active[idx];
        float ub = ub_scores[idx];
        ub_scores[idx] = -1e9f;
        is_surv = (act != 0) && (ub >= (tau_buf[q] - eps_buf[q]));
    }
    uint32_t mask = __ballot_sync(0xffffffff, is_surv);
    if (mask != 0u) {
        int base = 0;
        if (lane == 0) {
            base = atomicAdd(d_num_surv, __popc(mask));
        }
        base = __shfl_sync(0xffffffff, base, 0);
        int pos = base + __popc(mask & ((1u << lane) - 1u));
        if (is_surv && pos < max_surv) {
            surv_q[pos] = q;
            surv_doc[pos] = d;
        }
    }
}

__global__ void scatter_warm_scores_kernel(
    float* __restrict__ scores_mat,          // [B, N_docs]
    const int64_t* __restrict__ warm_idx,    // [B, c_w]
    const float* __restrict__ warm_scores,   // [B, c_w]
    int B, int c_w, int N_docs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * c_w;
    if (idx >= total) return;
    int q = idx / c_w;
    int64_t doc = warm_idx[idx];
    if (doc >= 0 && doc < N_docs) {
        scores_mat[(int64_t)q * N_docs + doc] = warm_scores[idx];
    }
}

__global__ void level2_exact_verify_scatter_kernel(
    const int32_t* __restrict__ surv_q,
    const int32_t* __restrict__ surv_doc,
    const int32_t* __restrict__ d_num_surv,
    const half* __restrict__ q_tokens,
    const int* __restrict__ q_lens,
    const half* __restrict__ d_tokens,
    const int64_t* __restrict__ d_offs,
    float* __restrict__ scores_mat,
    int max_surv, int N_docs, int m_max, int D)
{
    int num_pairs = *d_num_surv;
    if (num_pairs <= 0) return;
    if (num_pairs > max_surv) num_pairs = max_surv;

    int warps_per_block = blockDim.x >> 5;
    int total_warps = gridDim.x * warps_per_block;
    int global_warp = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;

    for (int pair_idx = global_warp; pair_idx < num_pairs; pair_idx += total_warps) {
        int q = surv_q[pair_idx];
        int doc = surv_doc[pair_idx];
        int mq = q_lens[q];
        int64_t ds = d_offs[doc];
        int64_t de = d_offs[doc + 1];

        const half* q_base = q_tokens + (int64_t)q * m_max * D;
        float total = 0.0f;

        if (D == 128) {
            int tok_sub = lane >> 4;
            int dim_off = (lane & 15) * 8;
            int i = 0;
            for (; i + 3 < mq; i += 4) {
                int4 qv0 = *reinterpret_cast<const int4*>(q_base + (i + 0) * 128 + dim_off);
                int4 qv1 = *reinterpret_cast<const int4*>(q_base + (i + 1) * 128 + dim_off);
                int4 qv2 = *reinterpret_cast<const int4*>(q_base + (i + 2) * 128 + dim_off);
                int4 qv3 = *reinterpret_cast<const int4*>(q_base + (i + 3) * 128 + dim_off);
                const half* qh0 = reinterpret_cast<const half*>(&qv0);
                const half* qh1 = reinterpret_cast<const half*>(&qv1);
                const half* qh2 = reinterpret_cast<const half*>(&qv2);
                const half* qh3 = reinterpret_cast<const half*>(&qv3);

                float b0 = -1e9f, b1 = -1e9f, b2 = -1e9f, b3 = -1e9f;
                for (int64_t j_base = ds; j_base < de; j_base += 2) {
                    int64_t j = j_base + tok_sub;
                    float s0 = -1e9f, s1 = -1e9f, s2 = -1e9f, s3 = -1e9f;
                    if (j < de) {
                        int4 dv = *reinterpret_cast<const int4*>(d_tokens + j * 128 + dim_off);
                        const half* dh = reinterpret_cast<const half*>(&dv);
                        s0 = 0.0f; s1 = 0.0f; s2 = 0.0f; s3 = 0.0f;
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            float d_val = __half2float(dh[k]);
                            s0 += __half2float(qh0[k]) * d_val;
                            s1 += __half2float(qh1[k]) * d_val;
                            s2 += __half2float(qh2[k]) * d_val;
                            s3 += __half2float(qh3[k]) * d_val;
                        }
                    }
                    for (int off = 8; off > 0; off >>= 1) {
                        s0 += __shfl_xor_sync(0xffffffff, s0, off);
                        s1 += __shfl_xor_sync(0xffffffff, s1, off);
                        s2 += __shfl_xor_sync(0xffffffff, s2, off);
                        s3 += __shfl_xor_sync(0xffffffff, s3, off);
                    }
                    if (j < de) {
                        b0 = fmaxf(b0, s0); b1 = fmaxf(b1, s1);
                        b2 = fmaxf(b2, s2); b3 = fmaxf(b3, s3);
                    }
                }
                b0 = fmaxf(b0, __shfl_xor_sync(0xffffffff, b0, 16));
                b1 = fmaxf(b1, __shfl_xor_sync(0xffffffff, b1, 16));
                b2 = fmaxf(b2, __shfl_xor_sync(0xffffffff, b2, 16));
                b3 = fmaxf(b3, __shfl_xor_sync(0xffffffff, b3, 16));
                if (lane == 0) total += (b0 + b1 + b2 + b3);
            }
            for (; i < mq; ++i) {
                int4 qv = *reinterpret_cast<const int4*>(q_base + i * 128 + dim_off);
                const half* qh = reinterpret_cast<const half*>(&qv);
                float b = -1e9f;
                for (int64_t j_base = ds; j_base < de; j_base += 2) {
                    int64_t j = j_base + tok_sub;
                    float s = -1e9f;
                    if (j < de) {
                        int4 dv = *reinterpret_cast<const int4*>(d_tokens + j * 128 + dim_off);
                        const half* dh = reinterpret_cast<const half*>(&dv);
                        s = 0.0f;
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) s += __half2float(qh[k]) * __half2float(dh[k]);
                    }
                    for (int off = 8; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
                    if (j < de) b = fmaxf(b, s);
                }
                b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, 16));
                if (lane == 0) total += b;
            }
        } else {
            int tok_sub = lane >> 4;
            int dim_off = (lane & 15) * 8;
            int n_dblks = D >> 7;
            int i = 0;
            for (; i + 7 < mq; i += 8) {
                float b0 = -1e9f, b1 = -1e9f, b2 = -1e9f, b3 = -1e9f;
                float b4 = -1e9f, b5 = -1e9f, b6 = -1e9f, b7 = -1e9f;
                for (int64_t j_base = ds; j_base < de; j_base += 2) {
                    int64_t j = j_base + tok_sub;
                    float s0 = -1e9f, s1 = -1e9f, s2 = -1e9f, s3 = -1e9f;
                    float s4 = -1e9f, s5 = -1e9f, s6 = -1e9f, s7 = -1e9f;
                    if (j < de) {
                        const half* dp = d_tokens + j * (int64_t)D;
                        s0 = 0.0f; s1 = 0.0f; s2 = 0.0f; s3 = 0.0f;
                        s4 = 0.0f; s5 = 0.0f; s6 = 0.0f; s7 = 0.0f;
                        for (int db = 0; db < n_dblks; ++db) {
                            int d_off = db * 128 + dim_off;
                            int4 dv = *reinterpret_cast<const int4*>(dp + d_off);
                            const half* dh = reinterpret_cast<const half*>(&dv);
                            float df[8];
                            #pragma unroll
                            for (int k = 0; k < 8; ++k) df[k] = __half2float(dh[k]);
                            #define ACC_Q8_SC(q_idx, s_var) { \
                                int4 qv = *reinterpret_cast<const int4*>(q_base + (int64_t)(i + q_idx) * D + d_off); \
                                const half* qh = reinterpret_cast<const half*>(&qv); \
                                for (int k = 0; k < 8; ++k) s_var += __half2float(qh[k]) * df[k]; \
                            }
                            ACC_Q8_SC(0, s0); ACC_Q8_SC(1, s1); ACC_Q8_SC(2, s2); ACC_Q8_SC(3, s3);
                            ACC_Q8_SC(4, s4); ACC_Q8_SC(5, s5); ACC_Q8_SC(6, s6); ACC_Q8_SC(7, s7);
                            #undef ACC_Q8_SC
                        }
                    }
                    for (int off = 8; off > 0; off >>= 1) {
                        s0 += __shfl_xor_sync(0xffffffff, s0, off);
                        s1 += __shfl_xor_sync(0xffffffff, s1, off);
                        s2 += __shfl_xor_sync(0xffffffff, s2, off);
                        s3 += __shfl_xor_sync(0xffffffff, s3, off);
                        s4 += __shfl_xor_sync(0xffffffff, s4, off);
                        s5 += __shfl_xor_sync(0xffffffff, s5, off);
                        s6 += __shfl_xor_sync(0xffffffff, s6, off);
                        s7 += __shfl_xor_sync(0xffffffff, s7, off);
                    }
                    if (j < de) {
                        b0 = fmaxf(b0, s0); b1 = fmaxf(b1, s1); b2 = fmaxf(b2, s2); b3 = fmaxf(b3, s3);
                        b4 = fmaxf(b4, s4); b5 = fmaxf(b5, s5); b6 = fmaxf(b6, s6); b7 = fmaxf(b7, s7);
                    }
                }
                b0 = fmaxf(b0, __shfl_xor_sync(0xffffffff, b0, 16));
                b1 = fmaxf(b1, __shfl_xor_sync(0xffffffff, b1, 16));
                b2 = fmaxf(b2, __shfl_xor_sync(0xffffffff, b2, 16));
                b3 = fmaxf(b3, __shfl_xor_sync(0xffffffff, b3, 16));
                b4 = fmaxf(b4, __shfl_xor_sync(0xffffffff, b4, 16));
                b5 = fmaxf(b5, __shfl_xor_sync(0xffffffff, b5, 16));
                b6 = fmaxf(b6, __shfl_xor_sync(0xffffffff, b6, 16));
                b7 = fmaxf(b7, __shfl_xor_sync(0xffffffff, b7, 16));
                if (lane == 0) total += (b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7);
            }
            for (; i < mq; ++i) {
                const half* qp = q_base + (int64_t)i * D;
                float b = -1e9f;
                for (int64_t j_base = ds; j_base < de; j_base += 2) {
                    int64_t j = j_base + tok_sub;
                    float s = -1e9f;
                    if (j < de) {
                        const half* dp = d_tokens + j * (int64_t)D;
                        s = 0.0f;
                        for (int db = 0; db < n_dblks; ++db) {
                            int d_off = db * 128 + dim_off;
                            int4 qv = *reinterpret_cast<const int4*>(qp + d_off);
                            int4 dv = *reinterpret_cast<const int4*>(dp + d_off);
                            const half* qh = reinterpret_cast<const half*>(&qv);
                            const half* dh = reinterpret_cast<const half*>(&dv);
                            #pragma unroll
                            for (int k = 0; k < 8; ++k) s += __half2float(qh[k]) * __half2float(dh[k]);
                        }
                    }
                    for (int off = 8; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
                    if (j < de) b = fmaxf(b, s);
                }
                b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, 16));
                if (lane == 0) total += b;
            }
        }
        if (lane == 0) {
            scores_mat[(int64_t)q * N_docs + doc] = total;
        }
    }
}

void launch_fused_stage2_verify(
    const uint8_t* doc_active, float* ub_scores, const float* tau_buf, const float* eps_buf,
    const int64_t* warm_idx, const float* warm_scores, int c_w,
    int32_t* surv_q, int32_t* surv_doc, int32_t* d_num_surv, int max_surv,
    const half* q_tokens, const int* q_lens, const half* d_tokens, const int64_t* d_offs,
    int B, int N_docs, int m_max, int D, cudaStream_t stream)
{
    cudaMemsetAsync(d_num_surv, 0, sizeof(int32_t), stream);
    dim3 block_c(256);
    dim3 grid_c((N_docs + 255) / 256, B);
    prepare_scores_and_compact_survivors_kernel<<<grid_c, block_c, 0, stream>>>(
        doc_active, ub_scores, tau_buf, eps_buf, surv_q, surv_doc, d_num_surv, max_surv, B, N_docs);

    if (warm_idx != nullptr && warm_scores != nullptr && c_w > 0) {
        int tot_w = B * c_w;
        scatter_warm_scores_kernel<<<dim3((tot_w + 127) / 128), dim3(128), 0, stream>>>(
            ub_scores, warm_idx, warm_scores, B, c_w, N_docs);
    }

    int grid_v = min(1024, max(32, B * 32));
    level2_exact_verify_scatter_kernel<<<dim3(grid_v), dim3(128), 0, stream>>>(
        surv_q, surv_doc, d_num_surv, q_tokens, q_lens, d_tokens, d_offs,
        ub_scores, max_surv, N_docs, m_max, D);
}

// ============================================================================
// V6 Kernel Group 3: GPU-Native SIGIR'25 WARP 2-Pass Implicit MaxSim Kernel
// Zero host sync, zero repeat_interleave, zero torch.unique!
// ============================================================================
__global__ void warp_pass1_atomic_max_kernel(
    const half* __restrict__ q_flat,         // [B * m, D]
    const int32_t* __restrict__ q_lens,      // [B]
    const int64_t* __restrict__ top_c_id,    // [B * m, tp]
    const float* __restrict__ top_c_sc,      // [B * m, tp]
    const float* __restrict__ floors,        // [B * m]
    const int64_t* __restrict__ c_offs,      // [K + 1]
    const int32_t* __restrict__ inv_doc_ids, // [N_toks]
    const int8_t* __restrict__ res_i8,       // [N_toks, D]
    const half* __restrict__ res_sc,         // [N_toks]
    int32_t* __restrict__ ws_max_i32,        // [q_chunk, N_docs]
    int qt_base, int n_qt, int m, int tp, int N_docs, int D)
{
    int qt_local = blockIdx.x;
    int p_idx = blockIdx.y;
    if (qt_local >= n_qt || p_idx >= tp) return;

    int qt_global = qt_base + qt_local;
    int q_id = qt_global / m;
    int m_id = qt_global - q_id * m;
    if (m_id >= q_lens[q_id]) return;

    int64_t probe_flat = (int64_t)qt_global * tp + p_idx;
    int64_t cid = top_c_id[probe_flat];
    float c_sc = top_c_sc[probe_flat];
    float fl = floors[qt_global];

    int64_t ls = c_offs[cid];
    int64_t le = c_offs[cid + 1];
    if (ls >= le) return;

    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = blockDim.x >> 5;
    int global_warp_in_list = blockIdx.z * num_warps + warp_id;
    int list_stride = gridDim.z * num_warps;

    const half* qp = q_flat + (int64_t)qt_global * D;
    for (int64_t tok = ls + global_warp_in_list; tok < le; tok += list_stride) {
        const int8_t* rp = res_i8 + tok * (int64_t)D;
        float dot = 0.0f;
        for (int d = lane * 4; d < D; d += 128) {
            char4 r4 = *reinterpret_cast<const char4*>(rp + d);
            half2 q01 = *reinterpret_cast<const half2*>(qp + d);
            half2 q23 = *reinterpret_cast<const half2*>(qp + d + 2);
            float2 f01 = __half22float2(q01);
            float2 f23 = __half22float2(q23);
            dot += f01.x * (float)r4.x + f01.y * (float)r4.y + f23.x * (float)r4.z + f23.y * (float)r4.w;
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            float imp = c_sc + dot * __half2float(res_sc[tok]);
            float delta = imp - fl;
            if (delta > 0.0f) {
                int doc_id = inv_doc_ids[tok];
                atomicMax(&ws_max_i32[(int64_t)qt_local * N_docs + doc_id], __float_as_int(delta));
            }
        }
    }
}

__global__ void warp_pass2_collect_and_reset_kernel(
    const int32_t* __restrict__ q_lens,      // [B]
    const int64_t* __restrict__ top_c_id,    // [B * m, tp]
    const int64_t* __restrict__ c_offs,      // [K + 1]
    const int32_t* __restrict__ inv_doc_ids, // [N_toks]
    int32_t* __restrict__ ws_max_i32,        // [q_chunk, N_docs]
    float* __restrict__ doc_scores,          // [B, N_docs]
    int qt_base, int n_qt, int m, int tp, int N_docs)
{
    int qt_local = blockIdx.x;
    int p_idx = blockIdx.y;
    if (qt_local >= n_qt || p_idx >= tp) return;

    int qt_global = qt_base + qt_local;
    int q_id = qt_global / m;
    int m_id = qt_global - q_id * m;
    if (m_id >= q_lens[q_id]) return;

    int64_t cid = top_c_id[(int64_t)qt_global * tp + p_idx];
    int64_t ls = c_offs[cid];
    int64_t le = c_offs[cid + 1];

    for (int64_t tok = ls + threadIdx.x; tok < le; tok += blockDim.x) {
        int doc_id = inv_doc_ids[tok];
        int64_t slot = (int64_t)qt_local * N_docs + doc_id;
        if (ws_max_i32[slot] > 0) {
            int prev = atomicExch(&ws_max_i32[slot], 0);
            if (prev > 0) {
                atomicAdd(&doc_scores[(int64_t)q_id * N_docs + doc_id], __int_as_float(prev));
            }
        }
    }
}

void launch_warp_implicit_maxsim(
    const half* q_flat, const int32_t* q_lens, const int64_t* top_c_id, const float* top_c_sc,
    const float* floors, const int64_t* c_offs, const int32_t* inv_doc_ids,
    const int8_t* res_i8, const half* res_sc, int32_t* ws_max_i32, float* doc_scores,
    int B, int m, int tp, int N_docs, int D, int q_chunk, cudaStream_t stream)
{
    int total_qt = B * m;
    for (int qt_base = 0; qt_base < total_qt; qt_base += q_chunk) {
        int n_qt = min(q_chunk, total_qt - qt_base);
        dim3 grid_p1(n_qt, tp, 8);
        dim3 grid_p2(n_qt, tp);
        warp_pass1_atomic_max_kernel<<<grid_p1, dim3(128), 0, stream>>>(
            q_flat, q_lens, top_c_id, top_c_sc, floors, c_offs, inv_doc_ids,
            res_i8, res_sc, ws_max_i32, qt_base, n_qt, m, tp, N_docs, D);
        warp_pass2_collect_and_reset_kernel<<<grid_p2, dim3(128), 0, stream>>>(
            q_lens, top_c_id, c_offs, inv_doc_ids, ws_max_i32, doc_scores,
            qt_base, n_qt, m, tp, N_docs);
    }
}

// ============================================================================
// V6 Kernel Group 4: GPU-Native ECIR'24 EMVB (Bit-Vector Popcount + PQ16 ADC)
// ============================================================================
__global__ void emvb_bitvec_filter_kernel(
    const uint64_t* __restrict__ q_doc_mask,   // [B, n_words]
    const uint64_t* __restrict__ doc_bitvecs,  // [N_docs, n_words]
    float* __restrict__ out_scores,            // [B, N_docs]
    int B, int N_docs, int n_words)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y;
    if (d >= N_docs || q >= B) return;
    const uint64_t* qm = q_doc_mask + (int64_t)q * n_words;
    const uint64_t* dm = doc_bitvecs + (int64_t)d * n_words;
    int pop = 0;
    for (int w = 0; w < n_words; ++w) {
        pop += __popcll(qm[w] & dm[w]);
    }
    out_scores[(int64_t)q * N_docs + d] = static_cast<float>(pop);
}

__global__ void emvb_pq16_maxsim_kernel(
    const uint64_t* __restrict__ q_union_mask, // [B, n_words]
    const uint64_t* __restrict__ q_tok_mask,   // [B, m, n_words]
    const float* __restrict__ pq_lut,          // [B, m, 16, 256]
    const int32_t* __restrict__ q_lens,        // [B]
    const int32_t* __restrict__ cand_ids,      // [B, C]
    const int64_t* __restrict__ d_offs,        // [N_docs + 1]
    const uint16_t* __restrict__ tok_cids,     // [N_toks]
    const uint8_t* __restrict__ pq_codes,      // [N_toks, 16]
    float* __restrict__ out_pq_scores,         // [B, C]
    int B, int m, int C, int n_words)
{
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int c = blockIdx.x * 4 + warp_id;
    int q = blockIdx.y;
    if (c >= C || q >= B) return;

    int doc = cand_ids[(int64_t)q * C + c];
    if (doc < 0) {
        if (lane == 0) out_pq_scores[(int64_t)q * C + c] = -1e9f;
        return;
    }

    int mq = min(m, min(64, q_lens[q]));
    float best[64];
    for (int i = 0; i < mq; ++i) best[i] = -1e9f;

    const uint64_t* qu = q_union_mask + (int64_t)q * n_words;
    const uint64_t* qt_m = q_tok_mask + (int64_t)q * m * n_words;
    const float* q_lut_base = pq_lut + (int64_t)q * m * 4096;

    int64_t ds = d_offs[doc];
    int64_t de = d_offs[doc + 1];

    for (int64_t t = ds + lane; t < de; t += 32) {
        uint16_t cid = tok_cids[t];
        int w = cid >> 6;
        uint64_t bit = 1ULL << (cid & 63);
        if ((qu[w] & bit) != 0ULL) {
            uint4 code_u4 = *reinterpret_cast<const uint4*>(pq_codes + t * 16);
            const uint8_t* cb = reinterpret_cast<const uint8_t*>(&code_u4);
            for (int i = 0; i < mq; ++i) {
                if ((qt_m[(int64_t)i * n_words + w] & bit) != 0ULL) {
                    const float* lut_i = q_lut_base + (int64_t)i * 4096;
                    float sim = 0.0f;
                    #pragma unroll
                    for (int sub = 0; sub < 16; ++sub) {
                        sim += lut_i[sub * 256 + cb[sub]];
                    }
                    best[i] = fmaxf(best[i], sim);
                }
            }
        }
    }

    float total = 0.0f;
    for (int i = 0; i < mq; ++i) {
        float b_max = warp_reduce_max(best[i]);
        if (lane == 0 && b_max > -1e8f) total += b_max;
    }
    if (lane == 0) {
        out_pq_scores[(int64_t)q * C + c] = total;
    }
}

void launch_emvb_bitvec_filter(
    const uint64_t* q_doc_mask, const uint64_t* doc_bitvecs, float* out_scores,
    int B, int N_docs, int n_words, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((N_docs + 255) / 256, B);
    emvb_bitvec_filter_kernel<<<grid, block, 0, stream>>>(
        q_doc_mask, doc_bitvecs, out_scores, B, N_docs, n_words);
}

void launch_emvb_pq16_maxsim(
    const uint64_t* q_union_mask, const uint64_t* q_tok_mask, const float* pq_lut,
    const int32_t* q_lens, const int32_t* cand_ids, const int64_t* d_offs,
    const uint16_t* tok_cids, const uint8_t* pq_codes, float* out_pq_scores,
    int B, int m, int C, int n_words, cudaStream_t stream)
{
    dim3 block(128);
    dim3 grid((C + 3) / 4, B);
    emvb_pq16_maxsim_kernel<<<grid, block, 0, stream>>>(
        q_union_mask, q_tok_mask, pq_lut, q_lens, cand_ids, d_offs,
        tok_cids, pq_codes, out_pq_scores, B, m, C, n_words);
}
}
