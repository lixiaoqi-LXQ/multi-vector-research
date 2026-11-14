#include "chimera/fp16_input.hpp"
#include "mapped_fp16_npy.hpp"
#include "mapped_embeddings.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
using FloatView=Chimera::FloatInputView;
constexpr uint64_t kCentroidSampleSeed=42;
size_t checked_product(size_t a,size_t b,const char*){if(b&&a>SIZE_MAX/b)throw std::overflow_error("overflow");return a*b;}
#include "original-sample-test.inc"
#include "sorted-sample-test.inc"
void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
int main(int argc,char**argv){try{
 need(argc==3,"Usage test_centroids <half.npy> <fp32.bin>");MappedFp16Npy half(argv[1]);MappedEmbeddings full(argv[2]);need(half.dimension()==128&&full.dimension()==128,"D128");
 const std::vector<std::pair<size_t,size_t>> cases={{1,1},{2,1},{2,2},{17,1},{17,9},{17,17},{63,31},{63,63},{257,1},{257,128},{257,257},{4099,256},{4099,4099}};
 size_t matrices=0,ids=0,repeats=0,forced=0;
 for(int input=0;input<2;++input){
  for(auto [n,c]:cases){FloatView view=input==0?FloatView(half.data(),n*128):FloatView(full.data(),n*128);original::forced_ids=sorted::forced_ids=nullptr;auto a=original::sample_centroids(view,n,128,c);auto b=sorted::sample_centroids(view,n,128,c);need(a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0,"entire centroid matrix bit mismatch");++matrices;need(original::captured_ids==sorted::captured_ids,"original reservoir IDs changed");++ids;auto again=sorted::sample_centroids(view,n,128,c);need(std::memcmp(b.data(),again.data(),b.size()*4)==0,"repeat mismatch");++repeats;}
  FloatView view=input==0?FloatView(half.data(),4099*128):FloatView(full.data(),4099*128);
  for(const std::vector<size_t>& selection:std::vector<std::vector<size_t>>{{4098,0,4098,1},{0,16,15,4098},{4098,4098,4098,4098}}){original::forced_ids=sorted::forced_ids=&selection;auto a=original::sample_centroids(view,4099,128,selection.size());auto b=sorted::sample_centroids(view,4099,128,selection.size());need(std::memcmp(a.data(),b.data(),a.size()*4)==0,"forced duplicates/tail matrix mismatch");need(original::captured_ids==selection&&sorted::captured_ids==selection,"forced IDs were reordered");++forced;}
 }
 std::cout<<"{\"status\":\"PASS_EXACT_CENTROID_BITS\",\"matrix_cases\":"<<matrices<<",\"reservoir_id_cases\":"<<ids<<",\"repeat_cases\":"<<repeats<<",\"forced_duplicate_tail_cases\":"<<forced<<",\"input_modes\":[\"mapped_FP16\",\"mapped_FP32\"],\"GPU_used\":false}"<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
