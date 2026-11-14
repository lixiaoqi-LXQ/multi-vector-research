#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/quantization/pack_excode.hpp"
#include <random>
#include <stdexcept>
#include <iostream>
int main() {
 std::mt19937 g(42); std::uniform_real_distribution<float> f(-1,1);
 size_t checks=0;
 for(size_t d: {64,128,256}) for(size_t b=1;b<=8;++b) for(int rep=0;rep<20;++rep) {
  std::vector<uint8_t> raw(d),packed(d*b/8); std::vector<float> decoded(d);
  for(auto& x:raw) x=g()% (1<<b);
  rabitqlib::quant::rabitq_impl::ex_bits::packing_rabitqplus_code(raw.data(),packed.data(),d,b);
  rabitqlib::select_excode_unpackfunc(b)(packed.data(),decoded.data(),d);
  for(size_t i=0;i<d;++i) {if(decoded[i]!=raw[i]) throw std::runtime_error("packed roundtrip mismatch b="+std::to_string(b)+" i="+std::to_string(i)); ++checks;}
 }
 for(int ld=1;ld<=11;++ld) {
  size_t d=1<<ld; std::vector<float> x(d),y;for(auto&v:x)v=f(g);y=x;
  fht_float(y.data(),ld);
  for(size_t i=0;i<d;++i){double ref=0;for(size_t j=0;j<d;++j) ref+=(__builtin_parity(i&j)?-1:1)*double(x[j]); if(std::abs(ref-y[i])>1e-3)throw std::runtime_error("fht mismatch");++checks;}
 }
 for(int rep=0;rep<20;++rep){alignas(64) float x[128],q[32*128],out[32]; for(auto& v:x)v=f(g);for(auto&v:q)v=f(g);
 rabitqlib::gemv_batch8_avx512(q,x,out,32,128);for(int i=0;i<32;++i){double ref=0;for(int j=0;j<128;++j)ref+=double(q[i*128+j])*x[j];if(std::abs(ref-out[i])>1e-4)throw std::runtime_error("gemv mismatch"); ++checks;}}
 std::cout << "PASS "<<checks<<" checks: pack/unpack bits1-8 dims64/128/256; FHT logdim1-11; batched GEMV32x128"<<std::endl;
}
