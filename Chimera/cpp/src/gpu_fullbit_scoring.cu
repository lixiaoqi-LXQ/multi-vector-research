#include "chimera/fullbit_gpu.cuh"
#include "chimera/chimera_index.cuh"
#include "chimera/config.cuh"
#include "chimera/profiling.hpp"
#include "chimera_index_internal.cuh"
#include <iostream>
#include <math_constants.h>
#include <stdexcept>

namespace Chimera {
namespace {
// One CTA per document, one 16-lane tile per query token. Each lane owns
// precisely the same eight dimensions/FP32 FMA order as the CPU NEON scorer.
__global__ void full_bit_maxsim_kernel(const char* codes, const float* factors,
    const int* ptrs, const int* ids, const float* queries, const float* cbex,
    float* scores) {
    const int lane = threadIdx.x & 15;
    const int q = threadIdx.x >> 4;
    const int doc = ids[blockIdx.x];
    __shared__ float max_per_query[32];
    __shared__ float query[32 * 128];
    for (int i = threadIdx.x; i < 32 * 128; i += blockDim.x)
        query[i] = queries[i];
    __syncthreads();
    float maximum = -CUDART_INF_F;
    for (int token = ptrs[doc]; token < ptrs[doc + 1]; ++token) {
        const auto* packed = reinterpret_cast<const unsigned char*>(codes)
                             + static_cast<size_t>(token) * 64;
        float dot = 0.0f;
        #pragma unroll
        for (int group = 0; group < 8; ++group) {
            const unsigned byte = packed[group * 8 + (lane & 7)];
            const float decoded = static_cast<float>((byte >> ((lane >> 3) * 4)) & 15);
            dot = __fmaf_rn(query[q * 128 + group * 16 + lane], decoded, dot);
        }
        // Preserve the exact 16-lane CPU addition tree (8,4,2,1).
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1)
            dot = __fadd_rn(dot, __shfl_down_sync(0xffffffffu, dot, offset, 16));
        if (lane == 0) {
            const float value = __fmul_rn(__fsub_rn(dot, cbex[q]), factors[token]);
            maximum = maximum > value ? maximum : value;
        }
    }
    if (lane == 0) max_per_query[q] = maximum;
    __syncthreads();
    // CPU first adds query maxima [0:16]+[16:32], then reduces 16 lanes.
    if (threadIdx.x < 16) {
        float sum = __fadd_rn(max_per_query[lane], max_per_query[lane + 16]);
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1)
            sum = __fadd_rn(sum, __shfl_down_sync(0x0000ffffu, sum, offset, 16));
        if (lane == 0) scores[blockIdx.x] = sum;
    }
}

template<class T>
const T* gpu_visible(const T* host, size_t bytes, bool ats, bool readonly, void*& registered) {
    if (ats) return host;
    const unsigned flags = cudaHostRegisterMapped | (readonly ? cudaHostRegisterReadOnly : 0);
    check_cuda(cudaHostRegister(const_cast<T*>(host), bytes, flags),
               "register full-bit UMA storage");
    registered = const_cast<T*>(host);
    void* device = nullptr;
    check_cuda(cudaHostGetDevicePointer(&device, registered, 0), "map full-bit storage");
    return static_cast<const T*>(device);
}
}

void launch_full_bit_gpu(const char* codes, const float* factors,
    const int* ptrs, const int* ids, const float* queries,
    const float* cbex, int count, float* scores, cudaStream_t stream) {
    if (count <= 0) return;
    full_bit_maxsim_kernel<<<count, 512, 0, stream>>>(
        codes, factors, ptrs, ids, queries, cbex, scores);
    check_cuda(cudaGetLastError(), "full-bit MaxSim kernel");
}

void chimera_index::initialize_gpu_full_bit() {
    if (ex_bits != 3 || PADDED_DIM != 128 || Q_DOCLEN != 32)
        throw std::runtime_error("Chimera+ GPU full-bit scorer requires 4bit, D128, Q32");
    int device = 0, pageable = 0, host_tables = 0;
    check_cuda(cudaGetDevice(&device));
    check_cuda(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device));
    check_cuda(cudaDeviceGetAttribute(&host_tables,
               cudaDevAttrPageableMemoryAccessUsesHostPageTables, device));
    const bool ats = pageable && host_tables;
    ws_->gpu_full_codes = gpu_visible(full_bit_codes_data(), n * 64, ats, low_memory_load_,
                                      ws_->registered_full_codes);
    ws_->gpu_full_factors = gpu_visible(full_bit_factors_.data(), n * sizeof(float), ats, false,
                                        ws_->registered_full_factors);
    const size_t count = ws_->max_full_bit_candidates;
    check_cuda(cudaMalloc(&ws_->d_full_doc_ids, count * sizeof(int)));
    check_cuda(cudaMalloc(&ws_->d_full_cbex, 32 * sizeof(float)));
    check_cuda(cudaMalloc(&ws_->d_full_scores, count * sizeof(float)));
    check_cuda(cudaMallocHost(&ws_->h_full_doc_ids, count * sizeof(int)));
    check_cuda(cudaMallocHost(&ws_->h_full_cbex, 32 * sizeof(float)));
    check_cuda(cudaMallocHost(&ws_->h_full_scores, count * sizeof(float)));
    std::cout << "[FULLBIT] backend=gpu access=" << (ats ? "ATS_host_page_tables" : "mapped_host")
              << " precision=fp32 codes_bytes=" << n * 64 << " corpus_copy_bytes=0\n";
}

void chimera_index::release_gpu_full_bit() noexcept {
    if (!ws_) return;
    (void)cudaFree(ws_->d_full_doc_ids);
    (void)cudaFree(ws_->d_full_cbex);
    (void)cudaFree(ws_->d_full_scores);
    (void)cudaFreeHost(ws_->h_full_doc_ids);
    (void)cudaFreeHost(ws_->h_full_cbex);
    (void)cudaFreeHost(ws_->h_full_scores);
    if (ws_->registered_full_codes) (void)cudaHostUnregister(ws_->registered_full_codes);
    if (ws_->registered_full_factors) (void)cudaHostUnregister(ws_->registered_full_factors);
}

void chimera_index::compute_gpu_full_bit_scores(const int* refined_ids,
    const int* candidate_indices, int count, const QueryState* queries,
    std::pair<float, int>* scores) {
    ProfileRange range(profiling_enabled_, "Chimera/gpu_fullbit", count);
    if (count <= 0) return;
    if (static_cast<size_t>(count) > ws_->max_full_bit_candidates)
        throw std::runtime_error("Full-bit candidate capacity exceeded");
    for (int i = 0; i < count; ++i) ws_->h_full_doc_ids[i] = refined_ids[candidate_indices[i]];
    for (int j = 0; j < 32; ++j) ws_->h_full_cbex[j] = queries[j].cbex_sumq;
    auto stream = ws_->stream_compute;
    check_cuda(cudaMemcpyAsync(ws_->d_full_doc_ids, ws_->h_full_doc_ids,
        count * sizeof(int), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(ws_->d_full_cbex, ws_->h_full_cbex,
        32 * sizeof(float), cudaMemcpyHostToDevice, stream));
    launch_full_bit_gpu(ws_->gpu_full_codes, ws_->gpu_full_factors,
        d_doc_ptrs_, ws_->d_full_doc_ids, ws_->d_queries, ws_->d_full_cbex,
        count, ws_->d_full_scores, stream);
    check_cuda(cudaMemcpyAsync(ws_->h_full_scores, ws_->d_full_scores,
        count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    check_cuda(cudaStreamSynchronize(stream));
    for (int i = 0; i < count; ++i) scores[i] = {ws_->h_full_scores[i], ws_->h_full_doc_ids[i]};
}
} // namespace Chimera
