#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
namespace Chimera {
// Preserve original chunk count, ceil size and cumulative-budget denominator.
inline std::pair<int,int> ceil_chunk_partition(int documents, int requested_chunks) {
    if (documents < 0 || requested_chunks <= 0) throw std::invalid_argument("Invalid document chunk shape");
    if (!documents) return {0, 0};
    const int count = std::min(requested_chunks, documents);
    return {count, documents / count + (documents % count != 0)};
}
inline std::pair<int,int> clamped_chunk_bounds(int documents, int size, int chunk) {
    if (documents < 0 || size <= 0 || chunk < 0) throw std::invalid_argument("Invalid document chunk range");
    const int start = int(std::min(int64_t(documents), int64_t(chunk) * size));
    const int end = int(std::min(int64_t(documents), int64_t(start) + size));
    return {start, end};
}
}
