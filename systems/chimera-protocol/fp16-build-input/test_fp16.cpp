#include "chimera/fp16_input.hpp"
#include "mapped_fp16_npy.hpp"
#include <cassert>
#include <array>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "test_sampling.inc"

template <typename T> std::vector<T> raw(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);assert(f.good());const size_t bytes=f.tellg();assert(bytes%sizeof(T)==0);
    std::vector<T> values(bytes/sizeof(T));f.seekg(0);f.read(reinterpret_cast<char*>(values.data()),bytes);assert(f.good());return values;
}
template<typename F> void rejects(F fn){bool fail=false;try{fn();}catch(const std::exception&){fail=true;}assert(fail);}
int main(int argc,char**argv){
 assert(argc==2);const std::string root(argv[1]);
 const auto expected=raw<uint32_t>(root+"/numpy-all-f32-bits.bin");assert(expected.size()==65536);
 MappedFp16Npy all(root+"/all-fp16-bits.npy");assert(all.size()==65536&&all.dimension()==128);
 std::vector<float> expanded(all.size()),single(all.size());
 assert(!Chimera::expand_fp16_to_fp32(all.data(),expanded.data(),all.size()));
 assert(std::memcmp(expanded.data(),expected.data(),expected.size()*4)==0);
 for(size_t i=0;i<all.size();++i){
  const bool finite=Chimera::expand_fp16_to_fp32(all.data()+i,single.data()+i,1);
  assert(finite==((all.data()[i]&0x7c00u)!=0x7c00u));
 }
 assert(std::memcmp(single.data(),expected.data(),expected.size()*4)==0);
 // Exercise vector/tail and unaligned-source cases separately.
 for(size_t count:{1u,2u,3u,4u,5u,7u,17u,65535u}){
  Chimera::expand_fp16_to_fp32(all.data()+1,expanded.data()+1,count);
  assert(std::memcmp(expanded.data()+1,expected.data()+1,count*4)==0);
 }
 std::cout<<"all_65536_FP16_patterns_match_numpy_bitwise_bulk_NEON_and_scalar_including_signed_zero_subnormal_Inf_NaN\n";
 const auto finite32=raw<float>(root+"/finite-f32.bin");
 for(const std::string filename:{"finite-u16.npy","finite-f16.npy","finite-v2.npy"}){
  MappedFp16Npy mapped(root+"/"+filename);assert(mapped.size()==finite32.size());
  std::vector<float> output(mapped.size());assert(Chimera::expand_fp16_to_fp32(mapped.data(),output.data(),mapped.size()));
  assert(std::memcmp(output.data(),finite32.data(),finite32.size()*4)==0);
 }
 MappedFp16Npy prefix(root+"/partial.npy",16);assert(prefix.size()==16*128);
 for(const std::string filename:{"fortran.npy","wrong-f32.npy","wrong-rank.npy","big-endian.npy","empty.npy","partial.npy","trailing.npy","missing.npy"}) rejects([&]{MappedFp16Npy invalid(root+"/"+filename);});
 rejects([&]{MappedFp16Npy invalid(root+"/finite-u16.npy",1000000);});
 std::cout<<"NPY_u16_f16_v1_v2_and_complete_prefix_ok_invalid_layout_dtype_length_endian_rejected\n";
 MappedFp16Npy finite(root+"/finite-u16.npy");
 Chimera::FloatInputView half(finite.data(),finite.size());
 Chimera::FloatInputView full(finite32.data(),finite32.size());
 std::vector<float> copy(finite.size());
 for(size_t begin=0;begin<finite.size();begin+=131){size_t count=std::min(size_t(131),finite.size()-begin);half.copy_to(copy.data()+begin,begin,count);}
 assert(std::memcmp(copy.data(),finite32.data(),copy.size()*4)==0);
 std::array<float,128> row;
 for(size_t i=0;i<finite.size();i+=128){assert(std::memcmp(half.row_data(i,128,row.data()),finite32.data()+i,512)==0);assert(full.row_data(i,128,row.data())==finite32.data()+i);}
 auto before=original::sample_centroids(original::FloatView(finite32.data(),finite32.size()),finite.size()/128,128,37);
 auto after=updated::sample_centroids(half,finite.size()/128,128,37);assert(before.size()==after.size());assert(std::memcmp(before.data(),after.data(),before.size()*4)==0);
 auto after32=updated::sample_centroids(full,finite.size()/128,128,37);assert(std::memcmp(before.data(),after32.data(),before.size()*4)==0);
 rejects([&]{half.copy_to(copy.data(),finite.size(),1);});
 Chimera::FloatInputView nonfinite(all.data(),all.size());rejects([&]{nonfinite.copy_to(expanded.data(),0,all.size());});
 std::cout<<"seed42_original_centroid_sampling_exact; assignment_chunk_and_every_encode_row_exact; checked_copy_rejects_Inf_NaN\n";
}
