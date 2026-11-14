#include "chimera/flat_l2_router.cuh"
#include "chimera/chimera_index.cuh"
#include <cublas_v2.h>
#include <raft/matrix/select_k.cuh>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
namespace Chimera {
namespace {
void blas_check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed, cuBLAS status " + std::to_string(int(status)));
}
__global__ void row_norms(const float* matrix, float* norms, int rows, int dimension, int stride) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int d = lane; d < dimension; d += 32) {
        const float v = matrix[size_t(row) * stride + d];
        sum = __fmaf_rn(v, v, sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, offset));
    if (lane == 0) norms[row] = sum;
}
__global__ void expanded_l2(float* dots, const float* centroid_norms,
                           const float* query_norms, size_t rows, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float norm_sum = __fadd_rn(centroid_norms[i % rows], query_norms[i / rows]);
        dots[i] = fmaxf(0.0f, __fadd_rn(norm_sum, __fmul_rn(-2.0f, dots[i])));
    }
}
}
FlatL2Router::FlatL2Router(const float* centroids, size_t rows, size_t dimension,
                         size_t stride, size_t queries)
    : centroids_(centroids), rows_(rows), dimension_(dimension), stride_(stride), queries_(queries) {
    if (!centroids || !rows || !dimension || stride < dimension || !queries ||
        rows > size_t(std::numeric_limits<int>::max()) || stride > size_t(std::numeric_limits<int>::max()) ||
        dimension > size_t(std::numeric_limits<int>::max()) || queries > size_t(std::numeric_limits<int>::max()) ||
        rows > std::numeric_limits<size_t>::max() / queries / sizeof(float))
        throw std::invalid_argument("Invalid flat route matrix shape or allocation overflow");
    cublasHandle_t handle = nullptr;
    try {
        check_cuda(cudaMalloc(&scores_, rows * queries * sizeof(float)));
        check_cuda(cudaMalloc(&centroid_norms_, rows * sizeof(float)));
        check_cuda(cudaMalloc(&query_norms_, queries * sizeof(float)));
        blas_check(cublasCreate(&handle), "Creating exact flat cuBLAS handle");
        blas_ = handle;
        blas_check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "Disabling reduced precision math");
        blas_check(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED), "Disabling cuBLAS atomics");
    } catch (...) {
        if (handle) (void)cublasDestroy(handle);
        (void)cudaFree(scores_); (void)cudaFree(centroid_norms_); (void)cudaFree(query_norms_);
        throw;
    }
}
FlatL2Router::~FlatL2Router() {
    if (blas_) (void)cublasDestroy(static_cast<cublasHandle_t>(blas_));
    (void)cudaFree(scores_); (void)cudaFree(centroid_norms_); (void)cudaFree(query_norms_);
}
void FlatL2Router::search(raft::resources& resources, const float* queries, size_t query_rows,
                          size_t k, float* distances, uint32_t* labels, cudaStream_t stream) {
    if (query_rows != queries_ || !queries || !k || k > rows_ || !distances || !labels)
        throw std::invalid_argument("Invalid exact flat query or top-k shape");
    auto handle = static_cast<cublasHandle_t>(blas_);
    blas_check(cublasSetStream(handle, stream), "Setting exact flat stream");
    raft::resource::set_cuda_stream(resources, rmm::cuda_stream_view(stream));
    if (!norms_ready_) {
        row_norms<<<(rows_ + 3) / 4, 128, 0, stream>>>(centroids_, centroid_norms_, int(rows_), int(dimension_), int(stride_));
        check_cuda(cudaGetLastError()); norms_ready_ = true;
    }
    row_norms<<<(queries_ + 3) / 4, 128, 0, stream>>>(queries, query_norms_, int(queries_), int(dimension_), int(dimension_));
    check_cuda(cudaGetLastError());
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major [Q,C] is column-major [C,Q]. A is row-major [C,D].
    blas_check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        int(rows_), int(queries_), int(dimension_), &alpha,
        centroids_, CUDA_R_32F, int(stride_), queries, CUDA_R_32F, int(dimension_),
        &beta, scores_, CUDA_R_32F, int(rows_), CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT),
        "Exact FP32 flat centroid GEMM");
    const size_t count = rows_ * queries_;
    expanded_l2<<<(count + 255) / 256, 256, 0, stream>>>(scores_, centroid_norms_, query_norms_, rows_, count);
    check_cuda(cudaGetLastError());
    raft::matrix::select_k<float, uint32_t>(resources,
        raft::make_device_matrix_view<const float, int64_t>(scores_, queries_, rows_),
        std::nullopt,
        raft::make_device_matrix_view<float, int64_t>(distances, queries_, k),
        raft::make_device_matrix_view<uint32_t, int64_t>(labels, queries_, k),
        true, true, raft::matrix::SelectAlgo::kAuto);
}
}
