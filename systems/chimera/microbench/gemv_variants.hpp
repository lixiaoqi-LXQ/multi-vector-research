#pragma once
#include "rabitqlib/utils/space.hpp"
namespace rabitqlib {
inline void gemv_batch4_simde(
    const float* __restrict__ queries_flat,   // [n_queries * dim], row-major
    const float* __restrict__ decoded,        // [dim], 64-byte aligned
    float* __restrict__ results,              // [n_queries]
    size_t n_queries,
    size_t dim
) {
    size_t j = 0;
    for (; j + 4 <= n_queries; j += 4) {
        const float* q0 = queries_flat + (j+0) * dim;
        const float* q1 = queries_flat + (j+1) * dim;
        const float* q2 = queries_flat + (j+2) * dim;
        const float* q3 = queries_flat + (j+3) * dim;
        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps(),
               s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
        for (size_t d = 0; d < dim; d += 16) {
            __m512 v = _mm512_load_ps(&decoded[d]);
            s0 = _mm512_fmadd_ps(v, _mm512_loadu_ps(&q0[d]), s0);
            s1 = _mm512_fmadd_ps(v, _mm512_loadu_ps(&q1[d]), s1);
            s2 = _mm512_fmadd_ps(v, _mm512_loadu_ps(&q2[d]), s2);
            s3 = _mm512_fmadd_ps(v, _mm512_loadu_ps(&q3[d]), s3);
        }
        results[j+0] = _mm512_reduce_add_ps(s0);
        results[j+1] = _mm512_reduce_add_ps(s1);
        results[j+2] = _mm512_reduce_add_ps(s2);
        results[j+3] = _mm512_reduce_add_ps(s3);
    }
    for (; j < n_queries; j++) {
        results[j] = dot_f32_avx512(queries_flat + j * dim, decoded, dim);
    }
}

}

namespace rabitqlib {
inline float reduce16_neon(float32x4_t a0,float32x4_t a1,float32x4_t a2,float32x4_t a3) {
 const float32x4_t v=vaddq_f32(vaddq_f32(a0,a2),vaddq_f32(a1,a3));
 return (vgetq_lane_f32(v,0)+vgetq_lane_f32(v,2))+(vgetq_lane_f32(v,1)+vgetq_lane_f32(v,3));
}
inline void gemv_batch4_neon(const float* __restrict__ query,const float* __restrict__ decoded,float* __restrict__ result,size_t nq,size_t dim) {
 size_t j=0;
 for(;j+4<=nq;j+=4) {
 const float* q0=query+(j+0)*dim;
 float32x4_t s00=vdupq_n_f32(0);
 float32x4_t s01=vdupq_n_f32(0);
 float32x4_t s02=vdupq_n_f32(0);
 float32x4_t s03=vdupq_n_f32(0);
 const float* q1=query+(j+1)*dim;
 float32x4_t s10=vdupq_n_f32(0);
 float32x4_t s11=vdupq_n_f32(0);
 float32x4_t s12=vdupq_n_f32(0);
 float32x4_t s13=vdupq_n_f32(0);
 const float* q2=query+(j+2)*dim;
 float32x4_t s20=vdupq_n_f32(0);
 float32x4_t s21=vdupq_n_f32(0);
 float32x4_t s22=vdupq_n_f32(0);
 float32x4_t s23=vdupq_n_f32(0);
 const float* q3=query+(j+3)*dim;
 float32x4_t s30=vdupq_n_f32(0);
 float32x4_t s31=vdupq_n_f32(0);
 float32x4_t s32=vdupq_n_f32(0);
 float32x4_t s33=vdupq_n_f32(0);
 for(size_t d=0;d<dim;d+=16) {
 const float32x4_t v0=vld1q_f32(decoded+d+0);
 const float32x4_t v1=vld1q_f32(decoded+d+4);
 const float32x4_t v2=vld1q_f32(decoded+d+8);
 const float32x4_t v3=vld1q_f32(decoded+d+12);
 s00=vfmaq_f32(s00,v0,vld1q_f32(q0+d+0));
 s01=vfmaq_f32(s01,v1,vld1q_f32(q0+d+4));
 s02=vfmaq_f32(s02,v2,vld1q_f32(q0+d+8));
 s03=vfmaq_f32(s03,v3,vld1q_f32(q0+d+12));
 s10=vfmaq_f32(s10,v0,vld1q_f32(q1+d+0));
 s11=vfmaq_f32(s11,v1,vld1q_f32(q1+d+4));
 s12=vfmaq_f32(s12,v2,vld1q_f32(q1+d+8));
 s13=vfmaq_f32(s13,v3,vld1q_f32(q1+d+12));
 s20=vfmaq_f32(s20,v0,vld1q_f32(q2+d+0));
 s21=vfmaq_f32(s21,v1,vld1q_f32(q2+d+4));
 s22=vfmaq_f32(s22,v2,vld1q_f32(q2+d+8));
 s23=vfmaq_f32(s23,v3,vld1q_f32(q2+d+12));
 s30=vfmaq_f32(s30,v0,vld1q_f32(q3+d+0));
 s31=vfmaq_f32(s31,v1,vld1q_f32(q3+d+4));
 s32=vfmaq_f32(s32,v2,vld1q_f32(q3+d+8));
 s33=vfmaq_f32(s33,v3,vld1q_f32(q3+d+12));
 }
 result[j+0]=reduce16_neon(s00,s01,s02,s03);
 result[j+1]=reduce16_neon(s10,s11,s12,s13);
 result[j+2]=reduce16_neon(s20,s21,s22,s23);
 result[j+3]=reduce16_neon(s30,s31,s32,s33);
 }
 for(;j<nq;++j) result[j]=dot_f32_avx512(query+j*dim,decoded,dim);
}
}
