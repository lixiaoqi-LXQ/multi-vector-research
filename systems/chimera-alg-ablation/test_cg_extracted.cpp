#include <cstddef>
#include <cstdint>
#include <vector>
#include <set>
#include <numeric>
#include <cassert>
#include <iostream>
using candidate_bitmap_bucket_t=uint32_t;
using candidate_bitmap_offset_t=uint32_t;
constexpr int kCandidateBitmapCompressionRatio=8,kCandidateBitmapBucketWidth=32;
struct Shape{size_t x;};Shape blockIdx,threadIdx,blockDim;
#define __global__
inline unsigned __popc(unsigned x){return __builtin_popcount(x);}
inline int __ffs(unsigned x){return __builtin_ffs(x);}
template<class T>inline T __ldg(const T*p){return *p;}
__global__ void count_candidate_bitmap_kernel(
    const candidate_bitmap_bucket_t* __restrict__ d_candidate_bitmap,
    size_t num_buckets,
    candidate_bitmap_offset_t* __restrict__ d_candidate_bitmap_offsets
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_buckets) {
        return;
    }

    if ((idx % kCandidateBitmapCompressionRatio) == 0) {
        candidate_bitmap_offset_t count = 0;
        const size_t group_bucket_start = idx;
        #pragma unroll
        for (int offset = 0; offset < kCandidateBitmapCompressionRatio; ++offset) {
            const size_t bucket_idx = group_bucket_start + offset;
            if (bucket_idx >= num_buckets) {
                break;
            }
            count += __popc(d_candidate_bitmap[bucket_idx]);
        }
        d_candidate_bitmap_offsets[idx / kCandidateBitmapCompressionRatio] = count;
    }
}

__global__ void materialize_candidate_documents_kernel(
    const candidate_bitmap_bucket_t* __restrict__ d_candidate_bitmap,
    size_t num_buckets,
    const candidate_bitmap_offset_t* __restrict__ d_candidate_bitmap_offsets,
    int* __restrict__ d_candidate_doc_ids
) {
    const size_t bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= num_buckets) {
        return;
    }

    const size_t group_idx = bucket_idx / kCandidateBitmapCompressionRatio;
    const size_t group_bucket_start = group_idx * kCandidateBitmapCompressionRatio;
    uint32_t out_idx = d_candidate_bitmap_offsets[group_idx];

    #pragma unroll
    for (int offset = 0; offset < kCandidateBitmapCompressionRatio; ++offset) {
        const size_t prev_bucket = group_bucket_start + offset;
        if (prev_bucket >= bucket_idx) {
            break;
        }
        out_idx += __popc(__ldg(&d_candidate_bitmap[prev_bucket]));
    }

    candidate_bitmap_bucket_t bits = __ldg(&d_candidate_bitmap[bucket_idx]);
    while (bits != 0u) {
        const int bit = __ffs(bits) - 1;
        d_candidate_doc_ids[out_idx++] =
            static_cast<int>(bucket_idx * kCandidateBitmapBucketWidth + bit);
        bits &= (bits - 1);
    }
}


int main(){int cases=0;
for(int n:{0,1,11,31,32,33,255,256,257,513,1001})for(int pattern=0;pattern<4;++pattern){
 size_t buckets=(((size_t(n)+31)/32+7)/8)*8;std::vector<uint32_t> bits(buckets),offsets(buckets/8+1);std::set<int> expected;
 for(int i=0;i<n;++i)if(pattern==1||pattern==2&&(i%3==0)||pattern==3&&(i==0||i==n-1||i==31||i==32||i==255||i==256)){expected.insert(i);bits[i/32]|=1u<<(i%32);bits[i/32]|=1u<<(i%32);}
 blockDim.x=256;for(size_t i=0;i<buckets;++i){blockIdx.x=i/256;threadIdx.x=i%256;count_candidate_bitmap_kernel(bits.data(),buckets,offsets.data());}
 uint32_t total=0;for(auto&v:offsets){auto count=v;v=total;total+=count;}assert(offsets.back()==expected.size());
 std::vector<int> output(expected.size());for(size_t i=0;i<buckets;++i){blockIdx.x=i/256;threadIdx.x=i%256;materialize_candidate_documents_kernel(bits.data(),buckets,offsets.data(),output.data());}
 assert(output==std::vector<int>(expected.begin(),expected.end()));++cases;
}
std::cout<<"PASS extracted_actual_CG_materialization_cases="<<cases<<"\n";}
