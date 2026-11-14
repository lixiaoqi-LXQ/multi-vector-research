#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
namespace raft { class resources; }
namespace Chimera {
// Exact online route over the stored rotated FP32 centroid matrix. No graph search.
class FlatL2Router {
 public:
    FlatL2Router(const float* centroids, size_t rows, size_t dimension, size_t stride, size_t queries);
    ~FlatL2Router();
    FlatL2Router(const FlatL2Router&) = delete;
    FlatL2Router& operator=(const FlatL2Router&) = delete;
    void search(raft::resources& resources, const float* queries, size_t query_rows,
                size_t k, float* distances, uint32_t* labels, cudaStream_t stream);
    size_t scratch_bytes() const { return sizeof(float) * (rows_ * queries_ + rows_ + queries_); }
 private:
    const float* centroids_;
    size_t rows_, dimension_, stride_, queries_;
    float* scores_ = nullptr;
    float* centroid_norms_ = nullptr;
    float* query_norms_ = nullptr;
    void* blas_ = nullptr;
    bool norms_ready_ = false;
};
}
