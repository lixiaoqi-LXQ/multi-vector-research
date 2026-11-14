#include "gemv_variants.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <array>
#include <cmath>
#include <vector>
#include <iomanip>
using F=void(*)(const float*,const float*,float*,size_t,size_t);
__attribute__((noinline)) void batch8(const float*q,const float*d,float*out,size_t n,size_t dim){rabitqlib::gemv_batch8_avx512(q,d,out,n,dim);}
__attribute__((noinline)) void batch4(const float*q,const float*d,float*out,size_t n,size_t dim){rabitqlib::gemv_batch4_simde(q,d,out,n,dim);}
__attribute__((noinline)) void batch4native(const float*q,const float*d,float*out,size_t n,size_t dim){rabitqlib::gemv_batch4_neon(q,d,out,n,dim);}
int main(){
 constexpr size_t D=128,Q=32,N=512,R=100000;
 std::mt19937 g(42);std::uniform_real_distribution<float> rnd(-1,1);
 alignas(64) float query[Q*D],decoded[N*D],out[Q],ref8[Q],ref4[Q],refn[Q];
 for(auto&v:query)v=rnd(g);for(auto&v:decoded)v=rnd(g);
 double e8=0,e4=0,en=0,e48=0,en8=0;
 for(size_t i=0;i<N;++i){batch8(query,decoded+i*D,ref8,Q,D);batch4(query,decoded+i*D,ref4,Q,D);batch4native(query,decoded+i*D,refn,Q,D);
 for(size_t j=0;j<Q;++j){double ref=0;for(size_t d=0;d<D;++d)ref+=double(query[j*D+d])*decoded[i*D+d];e8=std::max(e8,std::abs(ref-ref8[j]));e4=std::max(e4,std::abs(ref-ref4[j]));e48=std::max(e48,std::abs(double(ref4[j])-ref8[j]));en=std::max(en,std::abs(ref-refn[j]));en8=std::max(en8,std::abs(double(refn[j])-ref8[j]));}}
 std::cout<<std::setprecision(12)<<"{\"type\":\"correctness\",\"max_error8\":"<<e8<<",\"max_error4\":"<<e4<<",\"max_difference4vs8\":"<<e48<<",\"max_error_native\":"<<en<<",\"max_difference_native_vs8\":"<<en8<<"}\n";
 for(int round=0;round<6;++round)for(int order=0;order<3;++order){int choice=(order+round)%3;F fn=choice==2?batch4native:(choice?batch4:batch8);volatile double checksum=0;auto t0=std::chrono::steady_clock::now();for(size_t i=0;i<R;++i){fn(query,decoded+(i%N)*D,out,Q,D);checksum+=out[i%Q];}auto t1=std::chrono::steady_clock::now();double ns=std::chrono::duration<double,std::nano>(t1-t0).count()/R;std::cout<<"{\"type\":\"timing\",\"round\":"<<round<<",\"variant\":\""<<(choice==2?"batch4native":(choice?"batch4":"batch8"))<<"\",\"ns_per_token\":"<<ns<<",\"checksum\":"<<checksum<<"}\n";}
 return e4<1e-4&&e8<1e-4&&e48==0&&en<1e-4&&en8==0?0:1;
}
