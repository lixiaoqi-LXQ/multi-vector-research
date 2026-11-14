#pragma once
#if defined(__aarch64__)
#if defined(__CUDACC__)
#define SIMDE_NO_NATIVE
#endif
#define SIMDE_ENABLE_NATIVE_ALIASES
#include "simde-0.8.2/simde/x86/avx512.h"

inline __mmask16 _cvtu32_mask16(uint32_t v) { return static_cast<__mmask16>(v); }
inline __m512i _mm512_cvtepu8_epi32(__m128i a) {
    const auto x=simde__m128i_to_private(a); simde__m512i_private y;
    for(int i=0;i<16;++i) y.i32[i]=x.u8[i]; return simde__m512i_from_private(y);
}
inline __m512i _mm512_cvtepi8_epi32(__m128i a) {
    const auto x=simde__m128i_to_private(a); simde__m512i_private y;
    for(int i=0;i<16;++i) y.i32[i]=x.i8[i]; return simde__m512i_from_private(y);
}
inline __m128i _mm512_cvtepi32_epi8(__m512i a) {
    const auto x=simde__m512i_to_private(a); simde__m128i_private y;
    for(int i=0;i<16;++i) y.u8[i]=static_cast<uint8_t>(x.i32[i]); return simde__m128i_from_private(y);
}
inline __m256i _mm512_cvtepi32_epi16(__m512i a) {
    const auto x=simde__m512i_to_private(a); simde__m256i_private y;
    for(int i=0;i<16;++i) y.u16[i]=static_cast<uint16_t>(x.i32[i]); return simde__m256i_from_private(y);
}
inline void _mm_storeu_epi8(void* p, __m128i a) {simde_mm_storeu_si128(p,a);}
inline float _mm512_reduce_add_ps(__m512 a) {
    const auto x=simde__m512_to_private(a);
    // AVX512 reduction tree: 512->256->128->64->32.
    float q[8]; for(int i=0;i<8;++i) q[i]=x.f32[i]+x.f32[i+8];
    float r[4]; for(int i=0;i<4;++i) r[i]=q[i]+q[i+4];
    return (r[0]+r[2])+(r[1]+r[3]);
}

#define CHIMERA_SIMDE_AVX512 1
#else
#include <immintrin.h>
#endif
