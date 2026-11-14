#pragma once
#include <cstdint>
#include <cstddef>
#include <cfloat>
#include <cmath>
#include "chimera/config.cuh"
#ifdef __CUDACC__
#define CHIMERA_IP_INLINE __host__ __device__ __forceinline__
#else
#define CHIMERA_IP_INLINE inline
#endif
namespace Chimera {
// Paper no-LUT baseline: direct bit scan and conditional FP32 accumulation.
CHIMERA_IP_INLINE float direct_binary_ip(const float* query, const uint32_t* words) {
    float ip = 0.0f;
#ifdef __CUDACC__
#pragma unroll
#endif
    for (int dim = 0; dim < PADDED_DIM; ++dim)
        if ((words[dim >> 5] >> (dim & 31)) & 1u) ip += query[dim];
    return ip;
}
// CPU oracle for the production nibble-table grouping; not a GPU replacement.
inline float reference_lut_binary_ip(const float* query,const uint32_t* words) {
    float ip=0.0f;
    for(int chunk=0;chunk<PADDED_DIM/4;++chunk) {
        const unsigned nibble=(words[chunk/8]>>((chunk%8)*4))&15u;
        float item=0.0f;
        for(int bit=0;bit<4;++bit) if(nibble&(1u<<bit)) item+=query[chunk*4+bit];
        ip+=item;
    }
    return ip;
}
CHIMERA_IP_INLINE float naive_document_score_body(const float* scores,size_t begin,size_t end,size_t total_tokens) {
    float maxima[Q_DOCLEN];
#ifdef __CUDACC__
#pragma unroll
#endif
    for(int q=0;q<Q_DOCLEN;++q) {
        float value=-FLT_MAX;
        for(size_t t=begin;t<end;++t)value=fmaxf(value,scores[static_cast<size_t>(q)*total_tokens+t]);
        float query_sum=0.0f;
        query_sum+=value; // Match the Full kernel local_sum, including signed zero.
        maxima[q]=query_sum;
    }
#ifdef __CUDACC__
#pragma unroll
#endif
    for(int stride=Q_DOCLEN/2;stride>0;stride>>=1) {
#ifdef __CUDACC__
#pragma unroll
#endif
        for(int q=0;q<stride;++q)maxima[q]+=maxima[q+stride];
    }
    return maxima[0];
}

}
#undef CHIMERA_IP_INLINE
