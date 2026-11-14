#include "chimera/fullbit_gpu.cuh"
#include "chimera/chimera_index.cuh"
#include "rabitqlib/utils/space.hpp"
#include "cpu_gemv_neon.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
using Chimera::check_cuda;
float cpu_score(const std::vector<unsigned char>& codes, const std::vector<float>& factors,
                const std::vector<int>& ptrs, int doc, const float* query, const float* cb) {
    alignas(64) float maxima[32],decoded[128],dots[32];
    std::fill_n(maxima,32,-std::numeric_limits<float>::infinity());
    auto unpack=rabitqlib::select_excode_unpackfunc(4);
    for(int t=ptrs[doc];t<ptrs[doc+1];++t) {
        unpack(codes.data()+t*64,decoded,128);
#if defined(__aarch64__)
        rabitqlib::gemv_batch4_neon(query,decoded,dots,32,128);
#else
        rabitqlib::gemv_batch8_avx512(query,decoded,dots,32,128);
#endif
        for(int j=0;j<32;j++) {
            float v=(dots[j]-cb[j])*factors[t];
            maxima[j]=maxima[j]>v?maxima[j]:v;
        }
    }
    __m512 sum=_mm512_add_ps(_mm512_load_ps(maxima),_mm512_load_ps(maxima+16));
    return _mm512_reduce_add_ps(sum);
}
template<class T>T* upload(const T* p,size_t n) {
 T* d=nullptr;check_cuda(cudaMalloc(&d,n*sizeof(T)));check_cuda(cudaMemcpy(d,p,n*sizeof(T),cudaMemcpyHostToDevice));return d;
}
int main() {
    std::mt19937 rng(217);
    const std::vector<int> lens={1,2,3,31,32,33,127,128,129,180,300};
    std::vector<int> ptrs={0};for(int l:lens)ptrs.push_back(ptrs.back()+l);
    std::vector<unsigned char> codes(ptrs.back()*64);
    std::vector<float> factors(ptrs.back());
    alignas(64) std::array<float,32*128> query;
    alignas(64) std::array<float,32> cb;
    std::vector<int> ids={10,0,8,1,7,2,6,3,5,4,9,10,0};
    size_t checked=0;
    for(int fixture=0;fixture<8;fixture++) {
        for(auto&v:codes)v=static_cast<unsigned char>(rng());
        if(fixture==1)std::fill(codes.begin(),codes.end(),0);
        if(fixture==2)std::fill(codes.begin(),codes.end(),255);
        if(fixture==3)for(size_t i=0;i<codes.size();i++)codes[i]=static_cast<unsigned char>((i&15)|((15-(i&15))<<4));
        for(auto&v:factors)v=(float(int(rng()%1000)+1)/1000.0f);
        for(auto&v:query)v=float(int(rng()%2049)-1024)/512.f;
        if(fixture==2)for(auto&v:query)v=-std::abs(v); // all token scores negative
        if(fixture==4)query.fill(0.f);
        if(fixture==5)for(auto&v:factors)v=0.f; // signed zero ties
        for(int q=0;q<32;q++)cb[q]=std::accumulate(query.data()+q*128,query.data()+(q+1)*128,0.f)*15.f/2.f;
        // Assert nibble order at every dimension before any GPU comparison.
        alignas(64) float decoded[128];auto unpack=rabitqlib::select_excode_unpackfunc(4);
        for(size_t t=0;t<factors.size();t++) {
            unpack(codes.data()+t*64,decoded,128);
            for(int d=0;d<128;d++) {
                float expected=(codes[t*64+(d/16)*8+(d%8)]>>((d%16)/8*4))&15;
                if(decoded[d]!=expected)throw std::runtime_error("packed nibble mismatch");
            }
        }
        auto dc=upload(codes.data(),codes.size());auto df=upload(factors.data(),factors.size());
        auto dp=upload(ptrs.data(),ptrs.size());auto di=upload(ids.data(),ids.size());
        auto dq=upload(query.data(),query.size());auto db=upload(cb.data(),cb.size());
        float* out=nullptr;check_cuda(cudaMalloc(&out,ids.size()*sizeof(float)));
        Chimera::launch_full_bit_gpu(reinterpret_cast<char*>(dc),df,dp,di,dq,db,0,out,nullptr);
        Chimera::launch_full_bit_gpu(reinterpret_cast<char*>(dc),df,dp,di,dq,db,ids.size(),out,nullptr);
        std::vector<float> actual(ids.size());check_cuda(cudaMemcpy(actual.data(),out,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for(size_t i=0;i<ids.size();i++) {
            float expected=cpu_score(codes,factors,ptrs,ids[i],query.data(),cb.data());
            if(std::memcmp(&expected,&actual[i],4)) {
                std::cerr<<"fixture="<<fixture<<" doc="<<ids[i]<<" CPU="<<std::hexfloat<<expected<<" GPU="<<actual[i]<<"\n";
                throw std::runtime_error("FP32 score not bitwise equal");
            }
            checked++;
        }
        for(void* v:{static_cast<void*>(dc),static_cast<void*>(df),static_cast<void*>(dp),static_cast<void*>(di),static_cast<void*>(dq),static_cast<void*>(db),static_cast<void*>(out)})check_cuda(cudaFree(v));
    }
    check_cuda(cudaDeviceSynchronize());std::cout<<"PASS: "<<checked<<" CPU/GPU FP32 scores bitwise equal; nibble layout, negative scores, zero/ties, lengths1..300, repeated/reordered IDs, empty candidates\n";
}
