#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <algorithm>

extern "C" {
void launch_unpadded_cublas_reduce(
    const half* sim, const int64_t* d_offs, const int* q_lens, float* out_scores,
    int B, int m, int N_docs, int64_t total_toks, cudaStream_t stream);

void launch_fused_tc_int8_pass32(
    const int8_t* q_int8, const float* q_scales, const float* q_norms,
    const int8_t* d_int8, const float* d_scales, const float* d_rho,
    const int64_t* d_pad_offs, const float* tau_buf,
    float* ub_accum, float* lb_accum, const float* rem_bound, const float* eps_buf, uint8_t* doc_active,
    int B, int m_pad, int N_docs, int D, int q_tok_off, float eps_safe,
    int is_first_pass, int do_prune, int force_w1, int rem_is_2d, int slice_mode, cudaStream_t stream);

void launch_fused_tc_int8_hd_pass16(
    const int8_t* q_pack_i8, const float* q_scales, const float* q_norms,
    const int8_t* d_int8, const float* d_scales, const float* d_rho,
    const int64_t* d_pad_offs, const float* tau_buf,
    float* ub_accum, float* lb_accum, const float* rem_bound, const float* eps_buf, uint8_t* doc_active,
    int B, int m_pad, int N_docs, int D, int q_tok_off, float eps_safe,
    int is_first_pass, int do_prune, int rem_is_2d, int force_w1, cudaStream_t stream);

void launch_fused_tc_fp16_maxsim(
    const half* q_pad, const int* q_lens, const half* d_tokens, const int64_t* d_offs,
    float* out_scores, int B, int m_pad, int N_docs, int D, cudaStream_t stream);

void launch_fused_tc_int4_pass32(
    const int8_t* q_int8, const float* q_scales, const float* q_norms,
    const uint8_t* d_int4, const float* d_scales, const float* d_rho,
    const int64_t* d_pad_offs, const float* tau_buf,
    float* ub_accum, float* lb_accum, const float* rem_bound, const float* eps_buf, uint8_t* doc_active,
    int B, int m_pad, int N_docs, int q_tok_off, float eps_safe, int is_first_pass, int do_prune, cudaStream_t stream);

void launch_fused_tc_int6_pass32(
    const int8_t* q_int8, const float* q_scales, const float* q_norms,
    const uint8_t* d_hi4, const uint8_t* d_lo2, const float* d_scales, const float* d_rho,
    const int64_t* d_pad_offs, const float* tau_buf,
    float* ub_accum, float* lb_accum, const float* rem_bound, const float* eps_buf, uint8_t* doc_active,
    int B, int m_pad, int N_docs, int q_tok_off, float eps_safe, int is_first_pass, int do_prune, cudaStream_t stream);

void launch_prepare_queries_hadamard_int8(
    const half* q_fp16_pad, const int32_t* q_lens,
    int8_t* q_int8_out, float* q_scales_out, float* q_norms_out, float* eps_buf_out,
    float max_d_norm, int B, int m_max, int m_pad, int D, cudaStream_t stream);

void launch_fused_stage2_verify(
    const uint8_t* doc_active, float* ub_scores, const float* tau_buf, const float* eps_buf,
    const int64_t* warm_idx, const float* warm_scores, int c_w,
    int32_t* surv_q, int32_t* surv_doc, int32_t* d_num_surv, int max_surv,
    const half* q_tokens, const int* q_lens, const half* d_tokens, const int64_t* d_offs,
    int B, int N_docs, int m_max, int D, cudaStream_t stream);

void launch_warp_implicit_maxsim(
    const half* q_flat, const int32_t* q_lens, const int64_t* top_c_id, const float* top_c_sc,
    const float* floors, const int64_t* c_offs, const int32_t* inv_doc_ids,
    const int8_t* res_i8, const half* res_sc, int32_t* ws_max_i32, float* doc_scores,
    int B, int m, int tp, int N_docs, int D, int q_chunk, cudaStream_t stream);

void launch_emvb_bitvec_filter(
    const uint64_t* q_doc_mask, const uint64_t* doc_bitvecs, float* out_scores,
    int B, int N_docs, int n_words, cudaStream_t stream);

void launch_emvb_pq16_maxsim(
    const uint64_t* q_union_mask, const uint64_t* q_tok_mask, const float* pq_lut,
    const int32_t* q_lens, const int32_t* cand_ids, const int64_t* d_offs,
    const uint16_t* tok_cids, const uint8_t* pq_codes, float* out_pq_scores,
    int B, int m, int C, int n_words, cudaStream_t stream);

void launch_fused_tc_rerank_int8(
    const int8_t* q_int8, const float* q_scales, const float* q_norms,
    const int8_t* d_int8, const float* d_scales, const float* d_rho,
    const int64_t* d_pad_offs, const int32_t* cand_ids,
    float* out_ub, float* out_lb, int B, int C, int m_pad, int D, int q_tok_off, int is_first_pass, cudaStream_t stream);

void launch_level2_exact_verify(
    const int32_t* surv_q, const int32_t* surv_doc,
    const half* q_tokens, const int* q_lens,
    const half* d_tokens, const int64_t* d_offs,
    float* out_scores, int num_pairs, int m_max, int D, cudaStream_t stream);

void launch_level2_doc_grouped_verify(
    const int32_t* surv_docs, const uint8_t* surv_mask,
    const half* q_tokens, const int* q_lens,
    const half* d_tokens, const int64_t* d_offs,
    float* out_scores, int n_sd, int B, int m_max, int N_docs, cudaStream_t stream);

void set_l2_persisting_window(void* ptr, size_t num_bytes, cudaStream_t stream);

void launch_int8_gemm_bound_reduce(
    const int32_t* sim_i32, const float* q_scales, const float* q_norms,
    const float* d_scales, const float* d_rho, const int64_t* d_pad_offs,
    const float* tau_buf, float* ub_accum, float* lb_accum, const float* rem_bound,
    const float* eps_buf, uint8_t* doc_active, int B, int s_len, int m_pad, int N_docs,
    int doc_start, int nd, int chunk_toks, int64_t ts, int q_tok_off, float eps_safe,
    int is_first_pass, int do_prune, int rem_is_2d, cudaStream_t stream);
}

torch::Tensor unpadded_cublas_reduce(
    torch::Tensor sim, torch::Tensor d_offs, torch::Tensor q_lens, int B, int m, int N_docs) {
    auto out = torch::empty({B, N_docs}, torch::TensorOptions().dtype(torch::kFloat32).device(sim.device()));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_unpadded_cublas_reduce(
        reinterpret_cast<const half*>(sim.data_ptr<at::Half>()),
        d_offs.data_ptr<int64_t>(), q_lens.data_ptr<int32_t>(),
        out.data_ptr<float>(), B, m, N_docs, sim.size(1), stream);
    return out;
}

torch::Tensor fused_tc_fp16_maxsim(
    torch::Tensor q_pad, torch::Tensor q_lens, torch::Tensor d_tokens, torch::Tensor d_offs) {
    int B = q_pad.size(0);
    int m_pad = q_pad.size(1);
    int D = q_pad.size(2);
    int N_docs = d_offs.size(0) - 1;
    auto out = torch::empty({B, N_docs}, torch::TensorOptions().dtype(torch::kFloat32).device(q_pad.device()));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fused_tc_fp16_maxsim(
        reinterpret_cast<const half*>(q_pad.data_ptr<at::Half>()),
        q_lens.data_ptr<int32_t>(),
        reinterpret_cast<const half*>(d_tokens.data_ptr<at::Half>()),
        d_offs.data_ptr<int64_t>(),
        out.data_ptr<float>(), B, m_pad, N_docs, D, stream);
    return out;
}

void fused_tc_int8_pass32(
    torch::Tensor q_int8, torch::Tensor q_scales, torch::Tensor q_norms,
    torch::Tensor d_int8, torch::Tensor d_scales, torch::Tensor d_rho,
    torch::Tensor d_pad_offs, torch::Tensor tau_buf,
    torch::Tensor ub_accum, torch::Tensor lb_accum, torch::Tensor rem_bound, torch::Tensor doc_active,
    int q_tok_off, double eps_safe, int is_first_pass, int do_prune, int force_w1 = 0,
    c10::optional<torch::Tensor> eps_buf = c10::nullopt, int slice_mode = 0) {
    int B = q_int8.size(0);
    int m_pad = q_int8.size(1);
    int D = q_int8.size(2);
    int N_docs = d_pad_offs.size(0) - 1;
    int rem_is_2d = (rem_bound.dim() == 2) ? 1 : 0;
    const float* eps_ptr = (eps_buf.has_value() && eps_buf->defined() && eps_buf->numel() > 0)
        ? eps_buf->data_ptr<float>() : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fused_tc_int8_pass32(
        q_int8.data_ptr<int8_t>(), q_scales.data_ptr<float>(), q_norms.data_ptr<float>(),
        d_int8.data_ptr<int8_t>(), d_scales.data_ptr<float>(), d_rho.data_ptr<float>(),
        d_pad_offs.data_ptr<int64_t>(), tau_buf.data_ptr<float>(),
        ub_accum.data_ptr<float>(), lb_accum.data_ptr<float>(), rem_bound.data_ptr<float>(),
        eps_ptr, doc_active.data_ptr<uint8_t>(),
        B, m_pad, N_docs, D, q_tok_off, static_cast<float>(eps_safe),
        is_first_pass, do_prune, force_w1, rem_is_2d, slice_mode, stream);
}

void fused_tc_int8_hd_pass16(
    torch::Tensor q_pack_i8, torch::Tensor q_scales, torch::Tensor q_norms,
    torch::Tensor d_int8, torch::Tensor d_scales, torch::Tensor d_rho,
    torch::Tensor d_pad_offs, torch::Tensor tau_buf,
    torch::Tensor ub_accum, torch::Tensor lb_accum, torch::Tensor rem_bound, torch::Tensor doc_active,
    int q_tok_off, double eps_safe, int is_first_pass, int do_prune, int force_w1 = 0,
    c10::optional<torch::Tensor> eps_buf = c10::nullopt) {
    int B = q_pack_i8.size(0);
    int m_pad = q_pack_i8.size(1);
    int D = q_pack_i8.size(2);
    int N_docs = d_pad_offs.size(0) - 1;
    int rem_is_2d = (rem_bound.dim() == 2) ? 1 : 0;
    const float* eps_ptr = (eps_buf.has_value() && eps_buf->defined() && eps_buf->numel() > 0)
        ? eps_buf->data_ptr<float>() : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fused_tc_int8_hd_pass16(
        q_pack_i8.data_ptr<int8_t>(), q_scales.data_ptr<float>(), q_norms.data_ptr<float>(),
        d_int8.data_ptr<int8_t>(), d_scales.data_ptr<float>(), d_rho.data_ptr<float>(),
        d_pad_offs.data_ptr<int64_t>(), tau_buf.data_ptr<float>(),
        ub_accum.data_ptr<float>(), lb_accum.data_ptr<float>(), rem_bound.data_ptr<float>(),
        eps_ptr, doc_active.data_ptr<uint8_t>(),
        B, m_pad, N_docs, D, q_tok_off, static_cast<float>(eps_safe),
        is_first_pass, do_prune, rem_is_2d, force_w1, stream);
}

void fused_tc_int4_pass32(
    torch::Tensor q_int8, torch::Tensor q_scales, torch::Tensor q_norms,
    torch::Tensor d_int4, torch::Tensor d_scales, torch::Tensor d_rho,
    torch::Tensor d_pad_offs, torch::Tensor tau_buf,
    torch::Tensor ub_accum, torch::Tensor lb_accum, torch::Tensor rem_bound, torch::Tensor doc_active,
    int q_tok_off, double eps_safe, int is_first_pass, int do_prune,
    c10::optional<torch::Tensor> eps_buf = c10::nullopt) {
    int B = q_int8.size(0);
    int m_pad = q_int8.size(1);
    int N_docs = d_pad_offs.size(0) - 1;
    const float* eps_ptr = (eps_buf.has_value() && eps_buf->defined() && eps_buf->numel() > 0)
        ? eps_buf->data_ptr<float>() : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fused_tc_int4_pass32(
        q_int8.data_ptr<int8_t>(), q_scales.data_ptr<float>(), q_norms.data_ptr<float>(),
        d_int4.data_ptr<uint8_t>(), d_scales.data_ptr<float>(), d_rho.data_ptr<float>(),
        d_pad_offs.data_ptr<int64_t>(), tau_buf.data_ptr<float>(),
        ub_accum.data_ptr<float>(), lb_accum.data_ptr<float>(), rem_bound.data_ptr<float>(),
        eps_ptr, doc_active.data_ptr<uint8_t>(),
        B, m_pad, N_docs, q_tok_off, static_cast<float>(eps_safe), is_first_pass, do_prune, stream);
}

void fused_tc_int6_pass32(
    torch::Tensor q_int8, torch::Tensor q_scales, torch::Tensor q_norms,
    torch::Tensor d_hi4, torch::Tensor d_lo2, torch::Tensor d_scales, torch::Tensor d_rho,
    torch::Tensor d_pad_offs, torch::Tensor tau_buf,
    torch::Tensor ub_accum, torch::Tensor lb_accum, torch::Tensor rem_bound, torch::Tensor doc_active,
    int q_tok_off, double eps_safe, int is_first_pass, int do_prune,
    c10::optional<torch::Tensor> eps_buf = c10::nullopt) {
    int B = q_int8.size(0);
    int m_pad = q_int8.size(1);
    int N_docs = d_pad_offs.size(0) - 1;
    const float* eps_ptr = (eps_buf.has_value() && eps_buf->defined() && eps_buf->numel() > 0)
        ? eps_buf->data_ptr<float>() : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fused_tc_int6_pass32(
        q_int8.data_ptr<int8_t>(), q_scales.data_ptr<float>(), q_norms.data_ptr<float>(),
        d_hi4.data_ptr<uint8_t>(), d_lo2.data_ptr<uint8_t>(), d_scales.data_ptr<float>(), d_rho.data_ptr<float>(),
        d_pad_offs.data_ptr<int64_t>(), tau_buf.data_ptr<float>(),
        ub_accum.data_ptr<float>(), lb_accum.data_ptr<float>(), rem_bound.data_ptr<float>(),
        eps_ptr, doc_active.data_ptr<uint8_t>(),
        B, m_pad, N_docs, q_tok_off, static_cast<float>(eps_safe), is_first_pass, do_prune, stream);
}

void fused_tc_rerank_int8(
    torch::Tensor q_int8, torch::Tensor q_scales, torch::Tensor q_norms,
    torch::Tensor d_int8, torch::Tensor d_scales, torch::Tensor d_rho,
    torch::Tensor d_pad_offs, torch::Tensor cand_ids,
    torch::Tensor out_ub, torch::Tensor out_lb, int q_tok_off, int is_first_pass) {
    int B = q_int8.size(0);
    int C = cand_ids.size(1);
    int m_pad = q_int8.size(1);
    int D = q_int8.size(2);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fused_tc_rerank_int8(
        q_int8.data_ptr<int8_t>(), q_scales.data_ptr<float>(), q_norms.data_ptr<float>(),
        d_int8.data_ptr<int8_t>(), d_scales.data_ptr<float>(), d_rho.data_ptr<float>(),
        d_pad_offs.data_ptr<int64_t>(), cand_ids.data_ptr<int32_t>(),
        out_ub.data_ptr<float>(), out_lb.data_ptr<float>(),
        B, C, m_pad, D, q_tok_off, is_first_pass, stream);
}

torch::Tensor level2_exact_verify(
    torch::Tensor surv_q, torch::Tensor surv_doc,
    torch::Tensor q_tokens, torch::Tensor q_lens,
    torch::Tensor d_tokens, torch::Tensor d_offs) {
    surv_q = surv_q.contiguous();
    surv_doc = surv_doc.contiguous();
    q_tokens = q_tokens.contiguous();
    q_lens = q_lens.contiguous();
    d_tokens = d_tokens.contiguous();
    d_offs = d_offs.contiguous();
    int num_pairs = surv_q.size(0);
    int m_max = q_tokens.size(1);
    int D = d_tokens.size(1);
    auto out = torch::empty({num_pairs}, torch::TensorOptions().dtype(torch::kFloat32).device(q_tokens.device()));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_level2_exact_verify(
        surv_q.data_ptr<int32_t>(), surv_doc.data_ptr<int32_t>(),
        reinterpret_cast<const half*>(q_tokens.data_ptr<at::Half>()),
        q_lens.data_ptr<int32_t>(),
        reinterpret_cast<const half*>(d_tokens.data_ptr<at::Half>()),
        d_offs.data_ptr<int64_t>(),
        out.data_ptr<float>(), num_pairs, m_max, D, stream);
    return out;
}

void level2_doc_grouped_verify(
    torch::Tensor surv_docs, torch::Tensor surv_mask,
    torch::Tensor q_tokens, torch::Tensor q_lens,
    torch::Tensor d_tokens, torch::Tensor d_offs,
    torch::Tensor out_scores) {
    int n_sd = surv_docs.size(0);
    int B = surv_mask.size(0);
    int N_docs = surv_mask.size(1);
    int m_max = q_tokens.size(1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_level2_doc_grouped_verify(
        surv_docs.data_ptr<int32_t>(), surv_mask.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(q_tokens.data_ptr<at::Half>()),
        q_lens.data_ptr<int32_t>(),
        reinterpret_cast<const half*>(d_tokens.data_ptr<at::Half>()),
        d_offs.data_ptr<int64_t>(),
        out_scores.data_ptr<float>(), n_sd, B, m_max, N_docs, stream);
}

void int8_gemm_bound_reduce(
    torch::Tensor sim_i32, torch::Tensor q_scales, torch::Tensor q_norms,
    torch::Tensor d_scales, torch::Tensor d_rho, torch::Tensor d_pad_offs,
    torch::Tensor tau_buf, torch::Tensor ub_accum, torch::Tensor lb_accum,
    torch::Tensor rem_bound, torch::Tensor doc_active,
    int doc_start, int nd, int64_t ts, int q_tok_off, double eps_safe,
    int is_first_pass, int do_prune, c10::optional<torch::Tensor> eps_buf = c10::nullopt) {
    int B = q_scales.size(0);
    int s_len = sim_i32.size(0) / B;
    int m_pad = q_scales.size(1);
    int N_docs = d_pad_offs.size(0) - 1;
    int chunk_toks = sim_i32.size(1);
    int rem_is_2d = (rem_bound.dim() == 2) ? 1 : 0;
    const float* eps_ptr = (eps_buf.has_value() && eps_buf->defined() && eps_buf->numel() > 0)
        ? eps_buf->data_ptr<float>() : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_int8_gemm_bound_reduce(
        sim_i32.data_ptr<int32_t>(), q_scales.data_ptr<float>(), q_norms.data_ptr<float>(),
        d_scales.data_ptr<float>(), d_rho.data_ptr<float>(), d_pad_offs.data_ptr<int64_t>(),
        tau_buf.data_ptr<float>(), ub_accum.data_ptr<float>(), lb_accum.data_ptr<float>(),
        rem_bound.data_ptr<float>(), eps_ptr, doc_active.data_ptr<uint8_t>(),
        B, s_len, m_pad, N_docs, doc_start, nd, chunk_toks, ts, q_tok_off,
        static_cast<float>(eps_safe), is_first_pass, do_prune, rem_is_2d, stream);
}

void exec_hd_l2_int8_pass(
    torch::Tensor q_slice,
    torch::Tensor d_qdata,
    torch::Tensor d_scales,
    torch::Tensor d_rho,
    torch::Tensor d_pad_offs_gpu,
    torch::Tensor d_pad_offs_cpu,
    torch::Tensor tau_buf,
    torch::Tensor ub_accum,
    torch::Tensor lb_accum,
    torch::Tensor rem_bound,
    torch::Tensor doc_active,
    torch::Tensor q_scales,
    torch::Tensor q_norms,
    torch::Tensor sim_buf_1d,
    int64_t target_toks,
    int q_tok_off,
    double eps_safe,
    int is_first_pass,
    int do_prune,
    int use_l2_persist,
    c10::optional<torch::Tensor> eps_buf = c10::nullopt) {
    int B = q_scales.size(0);
    int m_pad = q_scales.size(1);
    int64_t M_rows = q_slice.size(0);
    int s_len = static_cast<int>(M_rows / B);
    int N_docs = static_cast<int>(d_pad_offs_cpu.size(0) - 1);
    int rem_is_2d = (rem_bound.dim() == 2) ? 1 : 0;
    const float* eps_ptr = (eps_buf.has_value() && eps_buf->defined() && eps_buf->numel() > 0)
        ? eps_buf->data_ptr<float>() : nullptr;
    const int64_t* offs = d_pad_offs_cpu.data_ptr<int64_t>();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (use_l2_persist) {
        size_t win_bytes = std::min<size_t>(
            static_cast<size_t>(sim_buf_1d.numel()) * sizeof(int32_t),
            static_cast<size_t>(28ULL * 1024ULL * 1024ULL));
        set_l2_persisting_window(sim_buf_1d.data_ptr(), win_bytes, stream);
    }

    int ds = 0;
    while (ds < N_docs) {
        int64_t ts = offs[ds];
        int64_t limit_tok = ts + target_toks;
        const int64_t* it = std::upper_bound(offs + ds + 1, offs + N_docs + 1, limit_tok);
        int de = static_cast<int>((it - offs) - 1);
        if (de <= ds) de = ds + 1;
        int64_t te = offs[de];
        int64_t n_toks = te - ts;
        int nd = de - ds;
        if (n_toks > 0) {
            int64_t needed = M_rows * n_toks;
            torch::Tensor sim_flat = (needed <= sim_buf_1d.numel())
                ? sim_buf_1d.narrow(0, 0, needed)
                : torch::empty({needed}, sim_buf_1d.options());
            auto sim_view = sim_flat.view({M_rows, n_toks});
            auto d_chunk_t = d_qdata.narrow(0, ts, n_toks).t();
            at::_int_mm_out(sim_view, q_slice, d_chunk_t);
            launch_int8_gemm_bound_reduce(
                sim_view.data_ptr<int32_t>(),
                q_scales.data_ptr<float>(), q_norms.data_ptr<float>(),
                d_scales.data_ptr<float>(), d_rho.data_ptr<float>(),
                d_pad_offs_gpu.data_ptr<int64_t>(),
                tau_buf.data_ptr<float>(),
                ub_accum.data_ptr<float>(), lb_accum.data_ptr<float>(),
                rem_bound.data_ptr<float>(), eps_ptr,
                doc_active.data_ptr<uint8_t>(),
                B, s_len, m_pad, N_docs, ds, nd, static_cast<int>(n_toks), ts,
                q_tok_off, static_cast<float>(eps_safe), is_first_pass, do_prune, rem_is_2d, stream);
        }
        ds = de;
    }

    if (use_l2_persist) {
        set_l2_persisting_window(nullptr, 0, stream);
    }
}

torch::Tensor exec_b8_hd_l2_fp16_maxsim(
    torch::Tensor q_pad,
    torch::Tensor q_lens,
    torch::Tensor d_tokens,
    torch::Tensor d_offs_gpu,
    torch::Tensor d_offs_cpu,
    torch::Tensor sim_buf_1d) {
    int B = q_pad.size(0);
    int m = q_pad.size(1);
    int D = q_pad.size(2);
    int64_t M_rows = static_cast<int64_t>(B) * m;
    int N_docs = static_cast<int>(d_offs_cpu.size(0) - 1);
    auto out = torch::empty({B, N_docs}, torch::TensorOptions().dtype(torch::kFloat32).device(q_pad.device()));
    auto q_flat = q_pad.contiguous().view({M_rows, D});
    const int64_t* offs = d_offs_cpu.data_ptr<int64_t>();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Pin intermediate similarity tile in Persisting L2 (<= 24 MB) so ZERO sim bytes spill to HBM!
    int64_t target_toks = std::max<int64_t>(4096, std::min<int64_t>(32768, (12LL * 1024LL * 1024LL) / std::max<int64_t>(M_rows, 1)));
    size_t win_bytes = std::min<size_t>(
        static_cast<size_t>(sim_buf_1d.numel()) * sizeof(at::Half),
        static_cast<size_t>(24ULL * 1024ULL * 1024ULL));
    set_l2_persisting_window(sim_buf_1d.data_ptr(), win_bytes, stream);

    int ds = 0;
    while (ds < N_docs) {
        int64_t ts = offs[ds];
        int64_t limit_tok = ts + target_toks;
        const int64_t* it = std::upper_bound(offs + ds + 1, offs + N_docs + 1, limit_tok);
        int de = static_cast<int>((it - offs) - 1);
        if (de <= ds) de = ds + 1;
        int64_t te = offs[de];
        int64_t n_toks = te - ts;
        int nd = de - ds;
        if (n_toks > 0) {
            int64_t needed = M_rows * n_toks;
            torch::Tensor sim_flat = (needed <= sim_buf_1d.numel())
                ? sim_buf_1d.narrow(0, 0, needed)
                : torch::empty({needed}, sim_buf_1d.options());
            auto sim_view = sim_flat.view({M_rows, n_toks});
            auto d_chunk_t = d_tokens.narrow(0, ts, n_toks).t();
            at::mm_out(sim_view, q_flat, d_chunk_t);
            auto out_slice = out.narrow(1, ds, nd);
            // Note: unpadded_cublas_reduce_kernel reads d_offs[d] - d_offs[0] when we pass shifted offsets or we pass d_offs_gpu+ds?
            // Wait: unpadded_cublas_reduce_kernel reads d_offs[d], expecting 0-based relative to sim! Let's check!
            auto rel_offs = d_offs_gpu.narrow(0, ds, nd + 1) - ts;
            auto sub_out = torch::empty({B, nd}, out.options());
            launch_unpadded_cublas_reduce(
                reinterpret_cast<const half*>(sim_view.data_ptr<at::Half>()),
                rel_offs.data_ptr<int64_t>(), q_lens.data_ptr<int32_t>(),
                sub_out.data_ptr<float>(), B, m, nd, n_toks, stream);
            out_slice.copy_(sub_out);
        }
        ds = de;
    }
    set_l2_persisting_window(nullptr, 0, stream);
    return out;
}

void prepare_queries_hadamard_int8(
    torch::Tensor q_fp16_pad, torch::Tensor q_lens,
    torch::Tensor q_int8_out, torch::Tensor q_scales_out,
    torch::Tensor q_norms_out, torch::Tensor eps_buf_out,
    double max_d_norm) {
    int B = q_fp16_pad.size(0);
    int m_max = q_fp16_pad.size(1);
    int D = q_fp16_pad.size(2);
    int m_pad = q_int8_out.size(1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_prepare_queries_hadamard_int8(
        reinterpret_cast<const half*>(q_fp16_pad.data_ptr<at::Half>()),
        q_lens.data_ptr<int32_t>(),
        q_int8_out.data_ptr<int8_t>(),
        q_scales_out.data_ptr<float>(),
        q_norms_out.data_ptr<float>(),
        eps_buf_out.data_ptr<float>(),
        static_cast<float>(max_d_norm), B, m_max, m_pad, D, stream);
}

void fused_stage2_verify(
    torch::Tensor doc_active, torch::Tensor ub_scores,
    torch::Tensor tau_buf, torch::Tensor eps_buf,
    c10::optional<torch::Tensor> warm_idx,
    c10::optional<torch::Tensor> warm_scores,
    torch::Tensor surv_q, torch::Tensor surv_doc, torch::Tensor d_num_surv,
    torch::Tensor q_tokens, torch::Tensor q_lens,
    torch::Tensor d_tokens, torch::Tensor d_offs) {
    int B = doc_active.size(0);
    int N_docs = doc_active.size(1);
    int max_surv = surv_q.size(0);
    int m_max = q_tokens.size(1);
    int D = q_tokens.size(2);
    int c_w = (warm_idx.has_value() && warm_idx->defined() && warm_idx->numel() > 0) ? warm_idx->size(1) : 0;
    const int64_t* w_idx_ptr = (c_w > 0) ? warm_idx->data_ptr<int64_t>() : nullptr;
    const float* w_sc_ptr = (c_w > 0 && warm_scores.has_value() && warm_scores->defined()) ? warm_scores->data_ptr<float>() : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fused_stage2_verify(
        doc_active.data_ptr<uint8_t>(), ub_scores.data_ptr<float>(),
        tau_buf.data_ptr<float>(), eps_buf.data_ptr<float>(),
        w_idx_ptr, w_sc_ptr, c_w,
        surv_q.data_ptr<int32_t>(), surv_doc.data_ptr<int32_t>(), d_num_surv.data_ptr<int32_t>(), max_surv,
        reinterpret_cast<const half*>(q_tokens.data_ptr<at::Half>()),
        q_lens.data_ptr<int32_t>(),
        reinterpret_cast<const half*>(d_tokens.data_ptr<at::Half>()),
        d_offs.data_ptr<int64_t>(),
        B, N_docs, m_max, D, stream);
}

void warp_implicit_maxsim(
    torch::Tensor q_flat, torch::Tensor q_lens,
    torch::Tensor top_c_id, torch::Tensor top_c_sc, torch::Tensor floors,
    torch::Tensor c_offs, torch::Tensor inv_doc_ids,
    torch::Tensor res_i8, torch::Tensor res_sc,
    torch::Tensor ws_max_i32, torch::Tensor doc_scores,
    int B, int m, int tp) {
    int N_docs = doc_scores.size(1);
    int D = q_flat.size(1);
    int q_chunk = ws_max_i32.size(0);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_warp_implicit_maxsim(
        reinterpret_cast<const half*>(q_flat.data_ptr<at::Half>()),
        q_lens.data_ptr<int32_t>(),
        top_c_id.data_ptr<int64_t>(),
        top_c_sc.data_ptr<float>(),
        floors.data_ptr<float>(),
        c_offs.data_ptr<int64_t>(),
        inv_doc_ids.data_ptr<int32_t>(),
        res_i8.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(res_sc.data_ptr<at::Half>()),
        ws_max_i32.data_ptr<int32_t>(),
        doc_scores.data_ptr<float>(),
        B, m, tp, N_docs, D, q_chunk, stream);
}

torch::Tensor emvb_bitvec_filter(torch::Tensor q_doc_mask, torch::Tensor doc_bitvecs) {
    int B = q_doc_mask.size(0);
    int n_words = q_doc_mask.size(1);
    int N_docs = doc_bitvecs.size(0);
    auto out = torch::empty({B, N_docs}, torch::TensorOptions().dtype(torch::kFloat32).device(q_doc_mask.device()));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_emvb_bitvec_filter(
        reinterpret_cast<const uint64_t*>(q_doc_mask.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(doc_bitvecs.data_ptr<int64_t>()),
        out.data_ptr<float>(), B, N_docs, n_words, stream);
    return out;
}

torch::Tensor emvb_pq16_maxsim(
    torch::Tensor q_union_mask, torch::Tensor q_tok_mask, torch::Tensor pq_lut,
    torch::Tensor q_lens, torch::Tensor cand_ids, torch::Tensor d_offs,
    torch::Tensor tok_cids, torch::Tensor pq_codes) {
    int B = cand_ids.size(0);
    int C = cand_ids.size(1);
    int m = q_tok_mask.size(1);
    int n_words = q_union_mask.size(1);
    auto out = torch::empty({B, C}, torch::TensorOptions().dtype(torch::kFloat32).device(pq_lut.device()));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_emvb_pq16_maxsim(
        reinterpret_cast<const uint64_t*>(q_union_mask.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(q_tok_mask.data_ptr<int64_t>()),
        pq_lut.data_ptr<float>(),
        q_lens.data_ptr<int32_t>(),
        cand_ids.data_ptr<int32_t>(),
        d_offs.data_ptr<int64_t>(),
        reinterpret_cast<const uint16_t*>(tok_cids.data_ptr<int16_t>()),
        pq_codes.data_ptr<uint8_t>(),
        out.data_ptr<float>(), B, m, C, n_words, stream);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unpadded_cublas_reduce", &unpadded_cublas_reduce, "Unpadded cuBLAS reduce");
    m.def("fused_tc_fp16_maxsim", &fused_tc_fp16_maxsim, "Fused TensorCore FP16 MaxSim Exact Baseline (B8)");
    m.def("exec_b8_hd_l2_fp16_maxsim", &exec_b8_hd_l2_fp16_maxsim, "Persisting-L2 Tiled FP16 TensorCore MaxSim for D>=256 (B8)");
    m.def("int8_gemm_bound_reduce", &int8_gemm_bound_reduce, "L2-tiled INT8 IMMA bound reduce");
    m.def("exec_hd_l2_int8_pass", &exec_hd_l2_int8_pass, "Hardware L2-Persisting IMMA + Bound Reduce Pass (D>=256)",
          pybind11::arg("q_slice"), pybind11::arg("d_qdata"), pybind11::arg("d_scales"), pybind11::arg("d_rho"),
          pybind11::arg("d_pad_offs_gpu"), pybind11::arg("d_pad_offs_cpu"), pybind11::arg("tau_buf"),
          pybind11::arg("ub_accum"), pybind11::arg("lb_accum"), pybind11::arg("rem_bound"),
          pybind11::arg("doc_active"), pybind11::arg("q_scales"), pybind11::arg("q_norms"),
          pybind11::arg("sim_buf_1d"), pybind11::arg("target_toks"), pybind11::arg("q_tok_off"),
          pybind11::arg("eps_safe"), pybind11::arg("is_first_pass"), pybind11::arg("do_prune"),
          pybind11::arg("use_l2_persist") = 1, pybind11::arg("eps_buf") = c10::nullopt);
    m.def("fused_tc_int8_pass32", &fused_tc_int8_pass32, "Adaptive TensorCore INT8 32-token pass",
          pybind11::arg("q_int8"), pybind11::arg("q_scales"), pybind11::arg("q_norms"),
          pybind11::arg("d_int8"), pybind11::arg("d_scales"), pybind11::arg("d_rho"),
          pybind11::arg("d_pad_offs"), pybind11::arg("tau_buf"),
          pybind11::arg("ub_accum"), pybind11::arg("lb_accum"), pybind11::arg("rem_bound"), pybind11::arg("doc_active"),
          pybind11::arg("q_tok_off"), pybind11::arg("eps_safe"), pybind11::arg("is_first_pass"),
          pybind11::arg("do_prune"), pybind11::arg("force_w1") = 0,
          pybind11::arg("eps_buf") = c10::nullopt, pybind11::arg("slice_mode") = 0);
    m.def("fused_tc_int8_hd_pass16", &fused_tc_int8_hd_pass16, "Fused Shared-Memory Multi-Query TensorCore INT8 16-token pass (D>=256)",
          pybind11::arg("q_pack_i8"), pybind11::arg("q_scales"), pybind11::arg("q_norms"),
          pybind11::arg("d_int8"), pybind11::arg("d_scales"), pybind11::arg("d_rho"),
          pybind11::arg("d_pad_offs"), pybind11::arg("tau_buf"),
          pybind11::arg("ub_accum"), pybind11::arg("lb_accum"), pybind11::arg("rem_bound"), pybind11::arg("doc_active"),
          pybind11::arg("q_tok_off"), pybind11::arg("eps_safe"), pybind11::arg("is_first_pass"),
          pybind11::arg("do_prune"), pybind11::arg("force_w1") = 0,
          pybind11::arg("eps_buf") = c10::nullopt);
    m.def("fused_tc_int4_pass32", &fused_tc_int4_pass32, "Adaptive TensorCore INT4 32-token pass",
          pybind11::arg("q_int8"), pybind11::arg("q_scales"), pybind11::arg("q_norms"),
          pybind11::arg("d_int4"), pybind11::arg("d_scales"), pybind11::arg("d_rho"),
          pybind11::arg("d_pad_offs"), pybind11::arg("tau_buf"),
          pybind11::arg("ub_accum"), pybind11::arg("lb_accum"), pybind11::arg("rem_bound"), pybind11::arg("doc_active"),
          pybind11::arg("q_tok_off"), pybind11::arg("eps_safe"), pybind11::arg("is_first_pass"),
          pybind11::arg("do_prune"), pybind11::arg("eps_buf") = c10::nullopt);
    m.def("fused_tc_int6_pass32", &fused_tc_int6_pass32, "Adaptive TensorCore Bit-Sliced INT6 32-token pass",
          pybind11::arg("q_int8"), pybind11::arg("q_scales"), pybind11::arg("q_norms"),
          pybind11::arg("d_hi4"), pybind11::arg("d_lo2"), pybind11::arg("d_scales"), pybind11::arg("d_rho"),
          pybind11::arg("d_pad_offs"), pybind11::arg("tau_buf"),
          pybind11::arg("ub_accum"), pybind11::arg("lb_accum"), pybind11::arg("rem_bound"), pybind11::arg("doc_active"),
          pybind11::arg("q_tok_off"), pybind11::arg("eps_safe"), pybind11::arg("is_first_pass"),
          pybind11::arg("do_prune"), pybind11::arg("eps_buf") = c10::nullopt);
    m.def("fused_tc_rerank_int8", &fused_tc_rerank_int8, "Mode B Candidate Rerank INT8 (general D)");
    m.def("level2_exact_verify", &level2_exact_verify, "Pair-parallel exact verify (general D)");
    m.def("level2_doc_grouped_verify", &level2_doc_grouped_verify, "Doc-grouped exact verify");
    m.def("prepare_queries_hadamard_int8", &prepare_queries_hadamard_int8, "Fused 7-stage FWHT-128 + INT8 query prep");
    m.def("fused_stage2_verify", &fused_stage2_verify, "Zero-host-sync Stage-2 compaction + scatter exact verify");
    m.def("warp_implicit_maxsim", &warp_implicit_maxsim, "GPU-native SIGIR'25 WARP 2-pass implicit MaxSim");
    m.def("emvb_bitvec_filter", &emvb_bitvec_filter, "GPU-native ECIR'24 EMVB bit-vector popcount filter");
    m.def("emvb_pq16_maxsim", &emvb_pq16_maxsim, "GPU-native ECIR'24 EMVB bit-masked PQ16 ADC MaxSim");
}
