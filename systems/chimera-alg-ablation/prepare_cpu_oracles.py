from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent
source=R/'source/cpp/src/kernels.cu'
s=source.read_text();start=s.index('__global__ void count_candidate_bitmap_kernel(');end=s.index('__global__ void sum_partial_scores_kernel(',start)
body=s[start:end]
prefix='''#include <cstddef>
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
'''
suffix='''
int main(){int cases=0;
for(int n:{0,1,11,31,32,33,255,256,257,513,1001})for(int pattern=0;pattern<4;++pattern){
 size_t buckets=(((size_t(n)+31)/32+7)/8)*8;std::vector<uint32_t> bits(buckets),offsets(buckets/8+1);std::set<int> expected;
 for(int i=0;i<n;++i)if(pattern==1||pattern==2&&(i%3==0)||pattern==3&&(i==0||i==n-1||i==31||i==32||i==255||i==256)){expected.insert(i);bits[i/32]|=1u<<(i%32);bits[i/32]|=1u<<(i%32);}
 blockDim.x=256;for(size_t i=0;i<buckets;++i){blockIdx.x=i/256;threadIdx.x=i%256;count_candidate_bitmap_kernel(bits.data(),buckets,offsets.data());}
 uint32_t total=0;for(auto&v:offsets){auto count=v;v=total;total+=count;}assert(offsets.back()==expected.size());
 std::vector<int> output(expected.size());for(size_t i=0;i<buckets;++i){blockIdx.x=i/256;threadIdx.x=i%256;materialize_candidate_documents_kernel(bits.data(),buckets,offsets.data(),output.data());}
 assert(output==std::vector<int>(expected.begin(),expected.end()));++cases;
}
std::cout<<"PASS extracted_actual_CG_materialization_cases="<<cases<<"\\n";}
'''
(R/'test_cg_extracted.cpp').write_text(prefix+body+suffix)
(R/'cg-extraction-provenance.json').write_text(json.dumps({'original_kernel_source':str(source),'original_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'extracted_functions':['count_candidate_bitmap_kernel','materialize_candidate_documents_kernel'],'body_transform':'none; CPU fake launch indices and popc/ffs/ldg intrinsics only','body_sha256':hashlib.sha256(body.encode()).hexdigest(),'test_source_sha256':hashlib.sha256((prefix+body+suffix).encode()).hexdigest(),'GPU_used':False},indent=2)+'\n')
