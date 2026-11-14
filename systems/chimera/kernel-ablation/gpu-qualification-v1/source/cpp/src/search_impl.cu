#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>

#include "chimera/chimera_index.cuh"

#define EIGEN_NO_CUDA

#include "chimera/config.cuh"
#include "chimera/chunk_partition.hpp"
#include "chimera/profiling.hpp"
#include "chimera/ivf_pg.hpp"
#include "chimera/kernels.cuh"
#include "chimera_index_internal.cuh"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

#include <algorithm>
#include "rabitqlib/utils/chimera_simd.hpp"
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Chimera {

namespace {

struct cast_int_size_t {
    __host__ __device__ size_t operator()(int value) const {
        return static_cast<size_t>(value);
    }
};

}  // namespace

std::vector<size_t> chimera_index::search(
    const std::vector<float>& query,
    size_t k) {
    return search(query.data(), query.size(), k);
}

std::vector<size_t> chimera_index::search(
    const float* query,
    size_t query_size,
    size_t k) {
    if (!has_index_) {
        throw std::runtime_error("Cannot search an empty Chimera index");
    }
    if (query == nullptr || query_size != Q_DOCLEN * d) {
        throw std::invalid_argument(
            "A Chimera query must have shape [" +
            std::to_string(Q_DOCLEN) + ", " + std::to_string(d) + "]");
    }
    if (k == 0) {
        throw std::invalid_argument("k must be positive");
    }
    if (!search_ready_) {
        try {
            initialize_search_state();
        } catch (...) {
            release_resources();
            throw;
        }
    }
    return search_one(query, k);
}

std::vector<size_t> chimera_index::search_one(const float* queries, size_t k) {
    for (size_t i = 0; i < Q_DOCLEN; ++i) {
        rotator_->rotate(&queries[i * d], &ws_->h_pinned_queries[i * PADDED_DIM]);
    }

    std::vector<QueryState> query_states(Q_DOCLEN);
    for (size_t i = 0; i < Q_DOCLEN; ++i) {
        query_states[i] =
            QueryState(&ws_->h_pinned_queries[i * PADDED_DIM], PADDED_DIM, ex_bits);
        ws_->h_pinned_cb1_sumq[i] = query_states[i].cb1_sumq;
    }

    check_cuda(cudaMemcpyAsync(ws_->d_queries, ws_->h_pinned_queries,
                               Q_DOCLEN * PADDED_DIM * sizeof(float),
                               cudaMemcpyHostToDevice, ws_->stream_h2d));
    check_cuda(cudaMemcpyAsync(ws_->d_cb1_sumq, ws_->h_pinned_cb1_sumq,
                               Q_DOCLEN * sizeof(float),
                               cudaMemcpyHostToDevice, ws_->stream_h2d));
    check_cuda(cudaEventRecord(ws_->event_h2d_done, ws_->stream_h2d));
    check_cuda(cudaStreamWaitEvent(ws_->stream_compute, ws_->event_h2d_done));

    if (options_.use_lut) {
    precompute_one_bit_lut_kernel<<<Q_DOCLEN, 256, 0, ws_->stream_compute>>>(
        ws_->d_queries, ws_->d_lut);
    check_cuda(cudaGetLastError());
    }


    int num_refined_candidates = 0;
    generate_and_refine_candidates(
        options_.nprobe,
        options_.k_refine,
        num_refined_candidates,
        ws_->stream_compute);


    std::vector<size_t> result;
    collaborative_document_scoring(
        num_refined_candidates,
        k,
        options_.k_full_bit,
        query_states.data(),
        result);


    return result;
}

void chimera_index::generate_and_refine_candidates(
    size_t nprobe,
    size_t k_refine,
    int& num_refined_candidates,
    cudaStream_t stream) {
    ProfileRange range(profiling_enabled_, "Chimera/candidate_generation_refinement");
    check_cuda(cudaMemsetAsync(d_candidate_bitmap_, 0,
                               candidate_bitmap_bucket_count_ * sizeof(candidate_bitmap_bucket_t),
                               ws_->stream_d2h));
    check_cuda(cudaMemsetAsync(d_candidate_bitmap_offsets_, 0,
                               candidate_bitmap_offset_count_ * sizeof(candidate_bitmap_offset_t),
                               ws_->stream_d2h));
    check_cuda(cudaEventRecord(ws_->event_h2d_done, ws_->stream_d2h));

    {
    ProfileRange route(profiling_enabled_, "Chimera/route");
    ivf->search_centroids(
        *cagra_res_,
        ws_->d_queries,
        Q_DOCLEN,
        nprobe,
        ws_->d_cagra_dists,
        ws_->d_cagra_labels,
        stream,
        static_cast<size_t>(options_.cagra_itopk_size));
    }

    check_cuda(cudaStreamWaitEvent(stream, ws_->event_h2d_done));

    const int threads_per_block = 256;
    const int blocks_x = std::max(
        (max_cluster_size + threads_per_block - 1) / threads_per_block,
        16);
    const dim3 grid(blocks_x, Q_DOCLEN);

    mark_candidate_documents_kernel<<<grid, threads_per_block, 0, stream>>>(
        ws_->d_cagra_labels,
        d_cluster_pos_,
        d_clustered_doc_ids_,
        d_candidate_bitmap_,
        num_docs,
        static_cast<int>(nprobe),
        ivf->n_clusters);
    check_cuda(cudaGetLastError());

    count_candidate_bitmap_kernel<<<
        (candidate_bitmap_bucket_count_ + threads_per_block - 1) / threads_per_block,
        threads_per_block,
        0,
        stream>>>(
            d_candidate_bitmap_,
            candidate_bitmap_bucket_count_,
            d_candidate_bitmap_offsets_);
    check_cuda(cudaGetLastError());

    size_t scan_temp = ws_->cub_temp_storage_bytes;
    cub::DeviceScan::ExclusiveSum(
        ws_->d_cub_temp_storage,
        scan_temp,
        d_candidate_bitmap_offsets_,
        d_candidate_bitmap_offsets_,
        candidate_bitmap_offset_count_,
        stream);

    candidate_bitmap_offset_t num_candidate_docs_u32 = 0;
    check_cuda(cudaMemcpyAsync(&num_candidate_docs_u32,
                               d_candidate_bitmap_offsets_ + candidate_bitmap_offset_count_ - 1,
                               sizeof(candidate_bitmap_offset_t),
                               cudaMemcpyDeviceToHost,
                               stream));
    check_cuda(cudaStreamSynchronize(stream));

    const int num_candidate_docs = static_cast<int>(num_candidate_docs_u32);
    if (num_candidate_docs == 0) {
        num_refined_candidates = 0;
        return;
    }
    if (static_cast<size_t>(num_candidate_docs) > max_candidate_docs_) {
        throw std::runtime_error(
            "Candidate document count " + std::to_string(num_candidate_docs) +
            " exceeded workspace bound " + std::to_string(max_candidate_docs_));
    }

    ensure_candidate_capacity(static_cast<size_t>(num_candidate_docs));
    check_cuda(cudaMemsetAsync(ws_->d_partial_query_scores,
                               0,
                               static_cast<size_t>(num_candidate_docs) * Q_DOCLEN * sizeof(float),
                               stream));

    materialize_candidate_documents_kernel<<<
        (candidate_bitmap_bucket_count_ + threads_per_block - 1) / threads_per_block,
        threads_per_block,
        0,
        stream>>>(
            d_candidate_bitmap_,
            candidate_bitmap_bucket_count_,
            d_candidate_bitmap_offsets_,
            ws_->d_materialized_candidate_doc_ids);
    check_cuda(cudaGetLastError());

    if (options_.use_aggregation) {
        if (options_.use_lut) {
    partial_score_kernel<<<grid, threads_per_block, 0, stream>>>(
        ws_->d_lut,
        d_clustered_one_bit_codes_,
        d_clustered_one_bit_factors_,
        ws_->d_cb1_sumq,
        ws_->d_cagra_labels,
        d_cluster_pos_,
        d_clustered_doc_ids_,
        ws_->d_partial_query_scores,
        d_candidate_bitmap_,
        d_candidate_bitmap_offsets_,
        num_candidate_docs,
        num_docs,
        static_cast<int>(nprobe),
        ivf->n_clusters);
    check_cuda(cudaGetLastError());
        } else {
    partial_score_direct_kernel<<<grid, threads_per_block, 0, stream>>>(
        ws_->d_queries,
        d_clustered_one_bit_codes_,
        d_clustered_one_bit_factors_,
        ws_->d_cb1_sumq,
        ws_->d_cagra_labels,
        d_cluster_pos_,
        d_clustered_doc_ids_,
        ws_->d_partial_query_scores,
        d_candidate_bitmap_,
        d_candidate_bitmap_offsets_,
        num_candidate_docs,
        num_docs,
        static_cast<int>(nprobe),
        ivf->n_clusters);
    check_cuda(cudaGetLastError());
        }
    } else {
        retrieved_token_counts_kernel<<<1,Q_DOCLEN,0,stream>>>(
            ws_->d_cagra_labels,d_cluster_pos_,ws_->d_ablation_offsets+1,
            static_cast<int>(nprobe),ivf->n_clusters);
        check_cuda(cudaGetLastError());
        check_cuda(cudaMemcpyAsync(ws_->h_ablation_offsets+1,ws_->d_ablation_offsets+1,
            Q_DOCLEN*sizeof(size_t),cudaMemcpyDeviceToHost,stream));
        check_cuda(cudaStreamSynchronize(stream));
        size_t total=0;
        for(int q=0;q<Q_DOCLEN;++q) {
            const size_t count=ws_->h_ablation_offsets[q+1];
            if(count>static_cast<size_t>(std::numeric_limits<int>::max()) ||
                total>std::numeric_limits<size_t>::max()-count)
                throw std::overflow_error("Routed token offsets exceed supported capacity");
            ws_->h_ablation_offsets[q]=total;total+=count;
        }
        ws_->h_ablation_offsets[Q_DOCLEN]=total;
        if(total>std::numeric_limits<size_t>::max()/sizeof(float) ||
           total>std::numeric_limits<size_t>::max()/sizeof(int))
            throw std::overflow_error("Packed materialization size overflow");
        if(total>ws_->ablation_token_capacity) {
            float* scores=nullptr;int* docs=nullptr;
            check_cuda(cudaMalloc(&scores,total*sizeof(float)));
            try {check_cuda(cudaMalloc(&docs,total*sizeof(int)));}
            catch(...) {(void)cudaFree(scores);throw;}
            check_cuda(cudaFree(ws_->d_ablation_scores));
            check_cuda(cudaFree(ws_->d_ablation_doc_ids));
            ws_->d_ablation_scores=scores;ws_->d_ablation_doc_ids=docs;
            ws_->ablation_token_capacity=total;
        }
        check_cuda(cudaMemcpyAsync(ws_->d_ablation_offsets,ws_->h_ablation_offsets,
            (Q_DOCLEN+1)*sizeof(size_t),cudaMemcpyHostToDevice,stream));
        ProfileRange materialized(profiling_enabled_,"Chimera/naive_CR_materialized_tokens",total);
        if(options_.use_lut) {
    partial_score_materialize_lut_kernel<<<grid, threads_per_block, 0, stream>>>(
        ws_->d_lut,
        d_clustered_one_bit_codes_,
        d_clustered_one_bit_factors_,
        ws_->d_cb1_sumq,
        ws_->d_cagra_labels,
        d_cluster_pos_,
        d_clustered_doc_ids_,
        ws_->d_partial_query_scores,
        d_candidate_bitmap_,
        d_candidate_bitmap_offsets_,
        num_candidate_docs,
        num_docs,
        static_cast<int>(nprobe),
        ivf->n_clusters,
        ws_->d_ablation_offsets,
        ws_->d_ablation_scores,
        ws_->d_ablation_doc_ids);
    check_cuda(cudaGetLastError());
        } else {
    partial_score_materialize_direct_kernel<<<grid, threads_per_block, 0, stream>>>(
        ws_->d_queries,
        d_clustered_one_bit_codes_,
        d_clustered_one_bit_factors_,
        ws_->d_cb1_sumq,
        ws_->d_cagra_labels,
        d_cluster_pos_,
        d_clustered_doc_ids_,
        ws_->d_partial_query_scores,
        d_candidate_bitmap_,
        d_candidate_bitmap_offsets_,
        num_candidate_docs,
        num_docs,
        static_cast<int>(nprobe),
        ivf->n_clusters,
        ws_->d_ablation_offsets,
        ws_->d_ablation_scores,
        ws_->d_ablation_doc_ids);
    check_cuda(cudaGetLastError());
        }
        aggregate_materialized_partial_kernel<<<grid,threads_per_block,0,stream>>>(
            ws_->d_ablation_scores,ws_->d_ablation_doc_ids,ws_->d_ablation_offsets,
            ws_->d_partial_query_scores,d_candidate_bitmap_,d_candidate_bitmap_offsets_,
            num_candidate_docs,num_docs);
        check_cuda(cudaGetLastError());
    }

    num_refined_candidates =
        std::min(static_cast<int>(k_refine), num_candidate_docs);

    const int thread_count = PARTIAL_SCORE_WARPS_PER_BLOCK * 32;
    const int score_blocks =
        (num_candidate_docs + PARTIAL_SCORE_WARPS_PER_BLOCK - 1) /
        PARTIAL_SCORE_WARPS_PER_BLOCK;
    sum_partial_scores_kernel<<<score_blocks, thread_count, 0, stream>>>(
        ws_->d_partial_query_scores,
        ws_->d_materialized_candidate_doc_ids,
        ws_->d_partial_scores,
        ws_->d_candidate_doc_ids,
        num_candidate_docs);
    check_cuda(cudaGetLastError());
    cub::DeviceRadixSort::SortPairsDescending(
        ws_->d_cub_temp_storage,
        ws_->cub_temp_storage_bytes,
        ws_->d_partial_scores,
        ws_->d_sorted_partial_scores,
        ws_->d_candidate_doc_ids,
        ws_->d_sorted_candidate_doc_ids,
        num_candidate_docs,
        0,
        32,
        stream);
    check_cuda(cudaGetLastError());
    check_cuda(cudaMemcpyAsync(ws_->d_candidate_doc_ids,
                               ws_->d_sorted_candidate_doc_ids,
                               num_refined_candidates * sizeof(int),
                               cudaMemcpyDeviceToDevice,
                               stream));
}

void chimera_index::collaborative_document_scoring(
    int num_refined_candidates,
    size_t k,
    size_t k_full_bit,
    QueryState* queries,
    std::vector<size_t>& result
) {
    ProfileRange range(profiling_enabled_, "Chimera/collaborative_scoring", num_refined_candidates);
    if (num_refined_candidates == 0) {
        result.clear();
        last_result_scores_.clear();
        return;
    }

    cudaStream_t stream = ws_->stream_compute;
    cudaStream_t stream_d2h = ws_->stream_d2h;
    const int full_bit_target = static_cast<int>(
        std::min(k_full_bit, static_cast<size_t>(num_refined_candidates)));

    // Prepare the refined candidate documents for collaborative scoring.
    int threads = 256;
    int blocks = (num_refined_candidates + threads - 1) / threads;

    gather_refined_doc_lengths_kernel<<<blocks, threads, 0, stream>>>(
        ws_->d_candidate_doc_ids, d_doc_ptrs_,
        ws_->d_refined_doc_lengths,
        num_refined_candidates
    );
    check_cuda(cudaGetLastError());

    check_cuda(cudaMemsetAsync(ws_->d_refined_doc_offsets, 0, sizeof(size_t), stream));
    {
        thrust::device_ptr<int> len_ptr(ws_->d_refined_doc_lengths);
        auto xform_iter = thrust::make_transform_iterator(len_ptr, cast_int_size_t());
        size_t scan_temp = ws_->cub_temp_storage_bytes;
        cub::DeviceScan::InclusiveScan(
            ws_->d_cub_temp_storage, scan_temp,
            xform_iter, ws_->d_refined_doc_offsets + 1,
            thrust::plus<size_t>(), num_refined_candidates, stream);
    }

    gather_refined_token_positions_kernel<<<
        num_refined_candidates, 256, 0, stream>>>(
        ws_->d_candidate_doc_ids,
        d_doc_ptrs_,
        d_token_to_cluster_pos_,
        ws_->d_refined_doc_offsets,
        ws_->d_refined_token_positions,
        num_refined_candidates
    );
    check_cuda(cudaGetLastError());

    size_t total_tokens;
    check_cuda(cudaMemcpyAsync(
        ws_->h_num_refined_tokens,
        ws_->d_refined_doc_offsets + num_refined_candidates,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        stream));
    check_cuda(cudaMemcpyAsync(
        ws_->h_pinned_refined_doc_offsets,
        ws_->d_refined_doc_offsets,
        (num_refined_candidates + 1) * sizeof(size_t),
        cudaMemcpyDeviceToHost,
        stream));

    check_cuda(cudaMemcpyAsync(
        ws_->h_refined_doc_ids,
        ws_->d_candidate_doc_ids,
        num_refined_candidates * sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));

    check_cuda(cudaStreamSynchronize(stream));
    total_tokens = *ws_->h_num_refined_tokens;
    const int* refined_doc_ids = ws_->h_refined_doc_ids;

    // Score the chunks on the GPU while completed chunks stream to the CPU.
    int score_threads = 128;
    while (score_threads < Q_DOCLEN && score_threads < 256) score_threads *= 2;
    score_threads = std::min(score_threads, 256);

    const auto partition = ceil_chunk_partition(num_refined_candidates, options_.num_chunks);
    const int num_chunks = partition.first;
    const int chunk_size = partition.second;

    std::vector<cudaEvent_t> chunk_compute_done(num_chunks);
    std::vector<cudaEvent_t> chunk_d2h_done(num_chunks);
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        check_cuda(cudaEventCreateWithFlags(
            &chunk_compute_done[chunk], cudaEventDisableTiming));
        check_cuda(cudaEventCreateWithFlags(
            &chunk_d2h_done[chunk], cudaEventDisableTiming));
    }

    auto launch_chunk_compute = [&](int chunk) {
        ProfileRange gpu_range(profiling_enabled_, "Chimera/gpu_chunk_submission", chunk);
        const auto bounds = clamped_chunk_bounds(num_refined_candidates, chunk_size, chunk);
        const int chunk_start = bounds.first;
        const int chunk_end = bounds.second;
        const int chunk_count = chunk_end - chunk_start;
        const size_t token_start =
            ws_->h_pinned_refined_doc_offsets[chunk_start];
        const size_t token_end =
            ws_->h_pinned_refined_doc_offsets[chunk_end];
        const size_t token_count = token_end - token_start;

        if (token_count > 0) {
            const int token_blocks = (token_count + 255) / 256;
            const size_t lut_shared_memory =
                DOCUMENT_SCORING_LUT_SMEM_FLOATS * sizeof(float) +
                DOCUMENT_SCORING_LUT_TILE_Q * sizeof(float);
            if(options_.use_lut) {
            one_bit_token_score_kernel<<<
                token_blocks, 256, lut_shared_memory, stream>>>(
                ws_->d_lut,
                d_clustered_one_bit_codes_,
                d_clustered_one_bit_factors_,
                ws_->d_cb1_sumq,
                ws_->d_refined_token_positions + token_start,
                ws_->d_one_bit_token_scores + token_start,
                total_tokens,
                token_count);
            check_cuda(cudaGetLastError());
            } else {
            one_bit_token_score_direct_kernel<<<token_blocks,256,0,stream>>>(
                ws_->d_queries,d_clustered_one_bit_codes_,d_clustered_one_bit_factors_,
                ws_->d_cb1_sumq,ws_->d_refined_token_positions+token_start,
                ws_->d_one_bit_token_scores+token_start,total_tokens,token_count);
            check_cuda(cudaGetLastError());
            }
        }

        if (chunk_count > 0) {
        if(options_.use_aggregation) {
        one_bit_document_score_kernel<<<
            chunk_count, score_threads, 0, stream>>>(
            ws_->d_one_bit_token_scores,
            ws_->d_refined_doc_offsets + chunk_start,
            ws_->d_one_bit_document_scores + chunk_start,
            total_tokens,
            chunk_count);
        check_cuda(cudaGetLastError());
        } else {
        one_bit_document_score_naive_kernel<<<(chunk_count+255)/256,256,0,stream>>>(
            ws_->d_one_bit_token_scores,ws_->d_refined_doc_offsets+chunk_start,
            ws_->d_one_bit_document_scores+chunk_start,total_tokens,chunk_count);
        check_cuda(cudaGetLastError());
        }
        }

        check_cuda(cudaEventRecord(chunk_compute_done[chunk], stream));

        check_cuda(cudaStreamWaitEvent(
            stream_d2h, chunk_compute_done[chunk], 0));
        if (chunk_count > 0) check_cuda(cudaMemcpyAsync(
            ws_->h_mapped_one_bit_document_scores + chunk_start,
            ws_->d_one_bit_document_scores + chunk_start,
            chunk_count * sizeof(float),
            cudaMemcpyDeviceToHost, stream_d2h));
        check_cuda(cudaEventRecord(chunk_d2h_done[chunk], stream_d2h));
    };

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        launch_chunk_compute(chunk);
    }

    std::priority_queue<std::pair<float, size_t>> full_bit_result_heap;
    auto& seen_doc_map = ws_->seen_doc_map;
    std::vector<int> seen_doc_list;
    seen_doc_list.reserve(full_bit_target * 2);
    int previous_order_size = 0;

    auto& one_bit_score_order = ws_->one_bit_score_order;
    auto& full_bit_candidate_indices = ws_->full_bit_candidate_indices;
    auto& full_bit_scores = ws_->full_bit_scores;

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const auto bounds = clamped_chunk_bounds(num_refined_candidates, chunk_size, chunk);
        const int chunk_start = bounds.first;
        const int chunk_end = bounds.second;
        const int chunk_count = chunk_end - chunk_start;
        {
            ProfileRange wait(profiling_enabled_, "Chimera/wait_chunk", chunk);
            check_cuda(cudaEventSynchronize(chunk_d2h_done[chunk]));
        }

        std::iota(
            one_bit_score_order.begin() + previous_order_size,
            one_bit_score_order.begin() + previous_order_size + chunk_count,
            chunk_start);
        const int merge_size = previous_order_size + chunk_count;
        const int carry_size = std::min(full_bit_target, merge_size);
        const float* one_bit_scores = ws_->h_mapped_one_bit_document_scores;
        auto descending_one_bit_score = [one_bit_scores](int a, int b) {
            return one_bit_scores[a] > one_bit_scores[b];
        };
        std::nth_element(
            one_bit_score_order.begin(),
            one_bit_score_order.begin() + carry_size,
            one_bit_score_order.begin() + merge_size,
            descending_one_bit_score);
        const int cumulative_full_bit_target = std::min(
            int((int64_t(full_bit_target) * (chunk + 1)) / num_chunks), carry_size);
        std::nth_element(
            one_bit_score_order.begin(),
            one_bit_score_order.begin() + cumulative_full_bit_target,
            one_bit_score_order.begin() + carry_size,
            descending_one_bit_score);
        previous_order_size = carry_size;

        std::vector<std::pair<float, int>> newly_selected_candidates;
        newly_selected_candidates.reserve(cumulative_full_bit_target);
        for (int i = 0; i < cumulative_full_bit_target; ++i) {
            const int refined_index = one_bit_score_order[i];
            const int doc_id = refined_doc_ids[refined_index];
            if (!seen_doc_map[doc_id]) {
                newly_selected_candidates.emplace_back(
                    one_bit_scores[refined_index], refined_index);
            }
        }

        if (newly_selected_candidates.empty()) {
            continue;
        }
        std::sort(
            newly_selected_candidates.begin(),
            newly_selected_candidates.end(),
            std::greater<>());
        const int num_full_bit_candidates =
            static_cast<int>(newly_selected_candidates.size());
        for (const auto& candidate : newly_selected_candidates) {
            const int doc_id = refined_doc_ids[candidate.second];
            seen_doc_map[doc_id] = true;
            seen_doc_list.push_back(doc_id);
        }

        for (int i = 0; i < num_full_bit_candidates; ++i) {
            full_bit_candidate_indices[i] =
                newly_selected_candidates[i].second;
        }

        compute_full_bit_scores(
            refined_doc_ids,
            full_bit_candidate_indices.data(),
            num_full_bit_candidates,
            ws_->h_pinned_queries,
            queries,
            full_bit_scores.data());

        for (int i = 0; i < num_full_bit_candidates; ++i) {
            full_bit_result_heap.emplace(
                full_bit_scores[i].first,
                static_cast<size_t>(full_bit_scores[i].second));
        }
    }

    result.clear();
    last_result_scores_.clear();
    if (capture_scores_) last_result_scores_.reserve(k);
    for (size_t i = 0; i < k && !full_bit_result_heap.empty(); ++i) {
        result.push_back(full_bit_result_heap.top().second);
        if (capture_scores_) last_result_scores_.push_back(full_bit_result_heap.top().first);
        full_bit_result_heap.pop();
    }

    for (int doc_id : seen_doc_list) seen_doc_map[doc_id] = false;

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        check_cuda(cudaEventDestroy(chunk_compute_done[chunk]));
        check_cuda(cudaEventDestroy(chunk_d2h_done[chunk]));
    }
}


}  // namespace Chimera
