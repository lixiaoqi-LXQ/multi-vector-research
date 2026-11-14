#include "chimera/fp16_input.hpp"
#include <limits>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace Chimera {
namespace {
uint32_t scalar_bits(uint16_t input) noexcept {
    const uint32_t sign = static_cast<uint32_t>(input & 0x8000u) << 16;
    const uint32_t exponent = (input >> 10) & 31u;
    uint32_t mantissa = input & 1023u;
    if (exponent == 31) {
        return sign | 0x7f800000u | (mantissa << 13) |
               (mantissa != 0 ? 0x00400000u : 0u);
    }
    if (exponent != 0) return sign | ((exponent + 112u) << 23) | (mantissa << 13);
    if (mantissa == 0) return sign;
    int shift = 0;
    while ((mantissa & 1024u) == 0) { mantissa <<= 1; ++shift; }
    return sign | (static_cast<uint32_t>(113 - shift) << 23) | ((mantissa & 1023u) << 13);
}
}

bool expand_fp16_to_fp32(const uint16_t* source, float* destination,
                         size_t count) noexcept {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "Input conversion requires IEEE binary32");
    size_t i = 0;
    bool finite = true;
#if defined(__aarch64__)
    const uint32x4_t zero = vdupq_n_u32(0);
    const uint32x4_t mask = vdupq_n_u32(1023u);
    uint32x4_t invalid = zero;
    for (; i + 4 <= count; i += 4) {
        const uint32x4_t input = vmovl_u16(vld1_u16(source + i));
        const uint32x4_t sign = vshlq_n_u32(vandq_u32(input, vdupq_n_u32(0x8000u)), 16);
        const uint32x4_t exponent = vandq_u32(vshrq_n_u32(input, 10), vdupq_n_u32(31u));
        const uint32x4_t mantissa = vandq_u32(input, mask);
        const uint32x4_t normal = vorrq_u32(vshlq_n_u32(vaddq_u32(exponent, vdupq_n_u32(112u)), 23),
                                           vshlq_n_u32(mantissa, 13));
        const uint32x4_t shift = vsubq_u32(vclzq_u32(mantissa), vdupq_n_u32(21u));
        uint32x4_t subnormal = vorrq_u32(
            vshlq_n_u32(vsubq_u32(vdupq_n_u32(113u), shift), 23),
            vshlq_n_u32(vandq_u32(vshlq_u32(mantissa, vreinterpretq_s32_u32(shift)), mask), 13));
        subnormal = vbslq_u32(vceqq_u32(mantissa, zero), zero, subnormal);
        const uint32x4_t quiet = vandq_u32(vmvnq_u32(vceqq_u32(mantissa, zero)), vdupq_n_u32(0x00400000u));
        const uint32x4_t special = vorrq_u32(vdupq_n_u32(0x7f800000u),
                                           vorrq_u32(vshlq_n_u32(mantissa, 13), quiet));
        uint32x4_t result = vbslq_u32(vceqq_u32(exponent, zero), subnormal, normal);
        const uint32x4_t is_special = vceqq_u32(exponent, vdupq_n_u32(31u));
        result = vorrq_u32(sign, vbslq_u32(is_special, special, result));
        vst1q_f32(destination + i, vreinterpretq_f32_u32(result));
        invalid = vorrq_u32(invalid, is_special);
    }
    finite = vmaxvq_u32(invalid) == 0;
#endif
    for (; i < count; ++i) {
        const uint32_t bits = scalar_bits(source[i]);
        std::memcpy(destination + i, &bits, sizeof(bits));
        finite = finite && ((source[i] & 0x7c00u) != 0x7c00u);
    }
    return finite;
}

}  // namespace Chimera
