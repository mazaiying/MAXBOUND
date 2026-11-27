#include <stdint.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <omp.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

static inline float dot_f32(const float* a, const float* b, int D) {
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    int k = 0;
    for (; k + 15 < D; k += 16) {
        __m512 va = _mm512_loadu_ps(a + k);
        __m512 vb = _mm512_loadu_ps(b + k);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    float res = _mm512_reduce_add_ps(acc);
    for (; k < D; ++k) res += a[k] * b[k];
    return res;
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 acc = _mm256_setzero_ps();
    int k = 0;
    for (; k + 7 < D; k += 8) {
        __m256 va = _mm256_loadu_ps(a + k);
        __m256 vb = _mm256_loadu_ps(b + k);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float res = _mm_cvtss_f32(sum);
    for (; k < D; ++k) res += a[k] * b[k];
    return res;
#else
    float res = 0.0f;
    for (int k = 0; k < D; ++k) res += a[k] * b[k];
    return res;
#endif
}

extern "C" {
void cpu_exact_maxsim_topk(
    const float* q_tokens,   // [total_q_toks, D]
    const int64_t* q_offs,   // [B + 1]
    const float* d_tokens,   // [total_d_toks, D]
    const int64_t* d_offs,   // [N_docs + 1]
    int B, int N_docs, int D, int topk,
    int64_t* out_ids,        // [B, topk]
    float* out_scores)       // [B, topk]
{
    int k_eff = (topk < N_docs) ? topk : N_docs;
    int n_threads = omp_get_max_threads();

    for (int q = 0; q < B; ++q) {
        int64_t qs = q_offs[q], qe = q_offs[q + 1];
        int mq = (int)(qe - qs);
        std::vector<float> scores(N_docs, 0.0f);

        #pragma omp parallel for schedule(dynamic, 64) num_threads(n_threads)
        for (int doc = 0; doc < N_docs; ++doc) {
            int64_t ds = d_offs[doc], de = d_offs[doc + 1];
            float total = 0.0f;
            for (int i = 0; i < mq; ++i) {
                const float* qp = q_tokens + (qs + i) * D;
                float best = -1e9f;
                for (int64_t j = ds; j < de; ++j) {
                    float s = dot_f32(qp, d_tokens + j * D, D);
                    if (s > best) best = s;
                }
                total += best;
            }
            scores[doc] = total;
        }

        std::vector<int> idx(N_docs);
        for (int i = 0; i < N_docs; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + k_eff, idx.end(),
            [&](int a, int b) {
                if (scores[a] != scores[b]) return scores[a] > scores[b];
                return a < b;
            });
        for (int r = 0; r < topk; ++r) {
            if (r < k_eff) {
                out_ids[(int64_t)q * topk + r] = idx[r];
                out_scores[(int64_t)q * topk + r] = scores[idx[r]];
            } else {
                out_ids[(int64_t)q * topk + r] = -1;
                out_scores[(int64_t)q * topk + r] = -1e9f;
            }
        }
    }
}
}
