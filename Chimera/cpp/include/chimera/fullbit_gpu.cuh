#pragma once
#include <cuda_runtime.h>

namespace Chimera {
// Same packed 4-bit layout and FP32 arithmetic as the ARM CPU full-bit scorer.
// All pointers must be accessible on the selected CUDA device.
void launch_full_bit_gpu(const char* codes, const float* factors,
    const int* doc_ptrs, const int* doc_ids, const float* queries,
    const float* cbex, int count, float* scores, cudaStream_t stream);
}
