#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace Chimera {

// IEEE binary16 -> binary32, including NumPy-compatible quiet NaN payloads.
// Always writes the exact converted bits; returns false if source has Inf/NaN.
// Implementation is ordinary C++, with integer NEON only in a CXX translation unit.
bool expand_fp16_to_fp32(const uint16_t* source, float* destination,
                         size_t count) noexcept;

class FloatInputView {
 public:
    FloatInputView(const float* source, size_t count)
        : fp32_(source), fp16_(nullptr), count_(count) {}
    FloatInputView(const uint16_t* source, size_t count)
        : fp32_(nullptr), fp16_(source), count_(count) {}
    size_t size() const { return count_; }
    bool empty() const { return count_ == 0 || (fp32_ == nullptr && fp16_ == nullptr); }
    bool is_fp16() const { return fp16_ != nullptr; }
    const float* float_data() const { return fp32_; }
    void copy_to(float* destination, size_t offset, size_t count) const {
        if (offset > count_ || count > count_ - offset) {
            throw std::out_of_range("Embedding input view read exceeds source");
        }
        if (fp16_ != nullptr) {
            if (!expand_fp16_to_fp32(fp16_ + offset, destination, count)) {
                throw std::invalid_argument("FP16 input contains Inf or NaN");
            }
        } else if (count != 0) {
            std::memcpy(destination, fp32_ + offset, count * sizeof(float));
        }
    }
    // encode_embeddings runs only after assignment has checked every FP16 row.
    // No exceptions inside its OpenMP region; the source is immutable during build.
    const float* row_data(size_t offset, size_t count, float* scratch) const noexcept {
        if (fp16_ == nullptr) return fp32_ + offset;
        (void)expand_fp16_to_fp32(fp16_ + offset, scratch, count);
        return scratch;
    }
 private:
    const float* fp32_;
    const uint16_t* fp16_;
    size_t count_;
};

}  // namespace Chimera
