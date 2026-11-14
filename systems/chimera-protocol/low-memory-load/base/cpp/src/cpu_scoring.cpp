#include "chimera/chimera_index.cuh"
#include "chimera/config.cuh"
#include "chimera_index_internal.cuh"
#include "rabitqlib/utils/space.hpp"
#include "cpu_gemv_neon.hpp"
#include <limits>
namespace Chimera {
void chimera_index::compute_full_bit_scores(
    const int* refined_doc_ids,
    const int* full_bit_candidate_indices,
    int num_full_bit_candidates,
    const float* queries_flat,
    const QueryState* queries,
    std::pair<float, int>* full_bit_scores
) {
    const size_t full_bit_codes_stride = PADDED_DIM * (1 + ex_bits) / 8;
    const auto& full_bit_codes = full_bit_codes_;
    const auto& full_bit_factors = full_bit_factors_;
    alignas(64) float cbex_sumq_arr[Q_DOCLEN];
    for (size_t j = 0; j < Q_DOCLEN; j++)
        cbex_sumq_arr[j] = queries[j].cbex_sumq;

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < num_full_bit_candidates; ++i) {
        int doc_id = refined_doc_ids[full_bit_candidate_indices[i]];
        size_t doc_start = doc_ptrs_[doc_id];
        size_t n_tok = doc_ptrs_[doc_id + 1] - doc_start;

        alignas(64) float decoded[PADDED_DIM];
        alignas(64) float full_ip[Q_DOCLEN];
        alignas(64) float max_ts[Q_DOCLEN];
        __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        for (size_t j = 0; j < Q_DOCLEN; j += 16)
            _mm512_store_ps(&max_ts[j], neg_inf);

        for (size_t t = 0; t < n_tok; t++) {
            size_t tid = doc_start + t;
            unpack_func_(
                reinterpret_cast<const uint8_t*>(
                    &full_bit_codes[tid * full_bit_codes_stride]),
                decoded, PADDED_DIM);
            if (t + 1 < n_tok) {
                __builtin_prefetch(
                    &full_bit_codes[(tid + 1) * full_bit_codes_stride], 0, 3);
                __builtin_prefetch(&full_bit_factors[tid + 1], 0, 1);
            }
#if defined(__aarch64__)
            rabitqlib::gemv_batch4_neon(
#else
            rabitqlib::gemv_batch8_avx512(
#endif
                queries_flat, decoded,
                full_ip,
                Q_DOCLEN, PADDED_DIM);

            const float full_bit_factor = full_bit_factors[tid];
            const __m512 full_bit_factor_vector =
                _mm512_set1_ps(full_bit_factor);

            for (size_t j = 0; j < Q_DOCLEN; j += 16) {
                __m512 full_ip_v = _mm512_load_ps(&full_ip[j]);
                __m512 cbex    = _mm512_load_ps(&cbex_sumq_arr[j]);
                __m512 combined = _mm512_mul_ps(
                    _mm512_sub_ps(full_ip_v, cbex),
                    full_bit_factor_vector);
                _mm512_store_ps(&max_ts[j],
                    _mm512_max_ps(_mm512_load_ps(&max_ts[j]), combined));
            }
        }

        __m512 sum = _mm512_load_ps(&max_ts[0]);
        for (size_t j = 16; j < Q_DOCLEN; j += 16)
            sum = _mm512_add_ps(sum, _mm512_load_ps(&max_ts[j]));
        full_bit_scores[i] = {_mm512_reduce_add_ps(sum), doc_id};
    }
}

} // namespace Chimera
