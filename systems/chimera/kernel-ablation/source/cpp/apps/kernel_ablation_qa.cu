#include "chimera/kernels.cuh"
#include "chimera/binary_ip_reference.hpp"
#include <array>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <unistd.h>
using namespace Chimera;
void ck(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
template<class T>struct Dev{T*p=nullptr;size_t n;Dev(size_t count):n(count){if(n)ck(cudaMalloc(&p,n*sizeof(T)));}~Dev(){if(p)cudaFree(p);}void put(const std::vector<T>&x){need(x.size()==n,"host/device size");if(n)ck(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T>get(){std::vector<T>x(n);if(n)ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}void zero(){if(n)ck(cudaMemset(p,0,n*sizeof(T)));}};
uint32_t bits(float x){uint32_t v;memcpy(&v,&x,4);return v;}
double gamma_n(int n){double u=std::numeric_limits<float>::epsilon()/2;return n*u/(1-n*u);}
size_t virtual_pos(size_t start,size_t n,size_t i){size_t head=start%8?8-start%8:0;if(!head||head>=n)return start+i;size_t run=n-head;return i<run?start+head+i:start+i-run;}
int main(int argc,char**argv){if(argc!=2||std::string(argv[1])!="--run"){std::cout<<"--run (requires UMA_RUN_GPU_TESTS=1)\n";return 0;}need(std::getenv("UMA_RUN_GPU_TESTS")&&std::string(std::getenv("UMA_RUN_GPU_TESTS"))=="1","GPU test marker missing");std::cout<<"[PID] "<<getpid()<<std::endl;
 const int D=15,C=17,NP=4;std::vector<int>lens={255,256,257,299,300,1,7,8,9,31,32,33,127,128,129};std::vector<size_t>off(1,0);for(int x:lens)off.push_back(off.back()+x);size_t N=off.back();std::mt19937 gen(42);std::uniform_real_distribution<float>rnd(-1,1);std::vector<std::array<uint32_t,4>>logical(N);std::vector<float>lf(N);std::vector<int>ld(N);for(int d=0;d<D;++d)for(size_t t=off[d];t<off[d+1];++t){for(auto&w:logical[t])w=gen();lf[t]=std::abs(rnd(gen))*3;ld[t]=d;if(d==0){logical[t].fill(0);lf[t]=1;}if(d==1){logical[t].fill(~0u);lf[t]=1;}if(d==2)lf[t]=0;}
 std::vector<std::vector<size_t>>clusters(C);for(size_t t=0;t<N;++t)clusters[1+(t*7)%16].push_back(t);std::vector<size_t>pos(1,0),physical;for(auto&x:clusters){physical.insert(physical.end(),x.begin(),x.end());pos.push_back(physical.size());}std::vector<std::array<uint32_t,4>>codes(N);std::vector<float>factor(N);std::vector<int>docs(N);std::vector<uint32_t>inverse(N);for(size_t i=0;i<N;++i){size_t t=physical[i];codes[i]=logical[t];factor[i]=lf[t];docs[i]=ld[t];inverse[t]=i;}docs[inverse[0]]=-1; // invalid CR sentinel, DS still uses logical offsets.
 std::vector<float>queries(32*128),cb(32);std::array<double,32>cb64{},absall{};for(int q=0;q<32;++q){float sum=0;for(int d=0;d<128;++d){float x=q==0?1:q==1?-1:q==2?0:rnd(gen)*std::ldexp(1.0f,q%15-7);queries[q*128+d]=x;sum+=x;cb64[q]+=.5*double(x);absall[q]+=std::abs(double(x));}cb[q]=sum*.5f;}
 Dev<std::array<uint32_t,4>>dc(N);dc.put(codes);Dev<float>df(N),dq(queries.size()),dcb(32),dlut(LUT_TOTAL_FLOATS);df.put(factor);dq.put(queries);dcb.put(cb);Dev<int>dd(N);dd.put(docs);Dev<size_t>dp(C+1),doff(off.size());dp.put(pos);doff.put(off);Dev<uint32_t>dinv(N);dinv.put(inverse);std::vector<uint32_t>bitmap(8,0);bitmap[0]=(1u<<D)-1;Dev<uint32_t>dbm(8),dbmo(2);dbm.put(bitmap);dbmo.put({0,D});Dev<float>partial(D*32),scores(N*32),dresult(D);Dev<size_t>packedoff(33),counts(32);size_t comparisons=0;double maxratio=0;
 precompute_one_bit_lut_kernel<<<32,256>>>(dq.p,dlut.p);ck(cudaGetLastError());
 for(int scenario=0;scenario<2;++scenario){std::vector<uint32_t>labels(32*NP);std::array<std::vector<size_t>,32>route;std::vector<size_t>packed(33,0);for(int q=0;q<32;++q){for(int p=0;p<NP;++p){uint32_t c=(q+p*5)%18;if(q==3||scenario==1)c=~0u;labels[q*NP+p]=c;if(c<C)for(size_t i=0;i<pos[c+1]-pos[c];++i)route[q].push_back(virtual_pos(pos[c],pos[c+1]-pos[c],i));}packed[q+1]=packed[q]+route[q].size();}size_t M=packed.back();Dev<uint32_t>labels_d(labels.size());labels_d.put(labels);retrieved_token_counts_kernel<<<1,32>>>(labels_d.p,dp.p,counts.p,NP,C);ck(cudaGetLastError());auto actualcounts=counts.get();for(int q=0;q<32;++q)need(actualcounts[q]==route[q].size(),"GPU routed counts");packedoff.put(packed);Dev<float>material(M);Dev<int>materialdocs(M);std::vector<float>reference_partial,reference_ds,reference_matrix,direct_partial,direct_ds,direct_matrix;
 #define CRARGS(INPUT) INPUT,reinterpret_cast<const char*>(dc.p),df.p,dcb.p,labels_d.p,dp.p,dd.p,partial.p,dbm.p,dbmo.p,D,D,NP,C
 for(int direct=0;direct<2;++direct)for(int naive=0;naive<2;++naive){partial.zero();dim3 grid(16,32);
  if(!naive){if(!direct)partial_score_kernel<<<grid,256>>>(CRARGS(dlut.p));else partial_score_direct_kernel<<<grid,256>>>(CRARGS(dq.p));}
  else{if(!direct)partial_score_materialize_lut_kernel<<<grid,256>>>(CRARGS(dlut.p),packedoff.p,material.p,materialdocs.p);else partial_score_materialize_direct_kernel<<<grid,256>>>(CRARGS(dq.p),packedoff.p,material.p,materialdocs.p);aggregate_materialized_partial_kernel<<<grid,256>>>(material.p,materialdocs.p,packedoff.p,partial.p,dbm.p,dbmo.p,D,D);}
  ck(cudaGetLastError());auto cr=partial.get();
  if(!direct)one_bit_token_score_kernel<<<(N+255)/256,256,DOCUMENT_SCORING_LUT_SMEM_FLOATS*sizeof(float)+DOCUMENT_SCORING_LUT_TILE_Q*sizeof(float)>>>(dlut.p,reinterpret_cast<const char*>(dc.p),df.p,dcb.p,dinv.p,scores.p,N,N);
  else one_bit_token_score_direct_kernel<<<(N+255)/256,256>>>(dq.p,reinterpret_cast<const char*>(dc.p),df.p,dcb.p,dinv.p,scores.p,N,N);
  if(!naive)one_bit_document_score_kernel<<<D,32>>>(scores.p,doff.p,dresult.p,N,D);else one_bit_document_score_naive_kernel<<<1,256>>>(scores.p,doff.p,dresult.p,N,D);ck(cudaGetLastError());auto ds=dresult.get();auto matrix=scores.get();
  if(!direct&&!naive){reference_partial=cr;reference_ds=ds;reference_matrix=matrix;}
  if(direct&&!naive){direct_partial=cr;direct_ds=ds;direct_matrix=matrix;}
  if(direct&&naive){need(memcmp(cr.data(),direct_partial.data(),cr.size()*4)==0,"Direct CR aggregation bitwise mismatch");need(memcmp(ds.data(),direct_ds.data(),ds.size()*4)==0,"Direct DS aggregation bitwise mismatch");need(memcmp(matrix.data(),direct_matrix.data(),matrix.size()*4)==0,"Direct token matrix mismatch");}
  if(!direct&&naive){need(memcmp(cr.data(),reference_partial.data(),cr.size()*4)==0,"LUT CR aggregation bitwise mismatch");need(memcmp(ds.data(),reference_ds.data(),ds.size()*4)==0,"LUT DS aggregation bitwise mismatch");need(memcmp(matrix.data(),reference_matrix.data(),matrix.size()*4)==0,"LUT token matrix mismatch");}
  // FP64 score reference and data-scaled roundoff bounds for all four modes.
  std::vector<double>cr64(D*32,0),crbound(D*32,0),ds64(D*32,-INFINITY),dsbound(D*32,0);
  for(int q=0;q<32;++q)for(size_t t=0;t<N;++t){double ip=0,absip=0;for(int d=0;d<128;++d)if((logical[t][d/32]>>(d%32))&1u){double v=queries[q*128+d];ip+=v;absip+=std::abs(v);}double v=(ip-cb64[q])*lf[t];double bound=(gamma_n(128)*(absip+.5*absall[q])+gamma_n(3)*(absip+std::abs(cb64[q])))*std::abs(lf[t]);double error=std::abs(double(matrix[q*N+t])-v);need(error<=bound,"GPU token FP64 scaled error");if(bound)maxratio=std::max(maxratio,error/bound);++comparisons;int doc=ld[t];ds64[doc*32+q]=std::max(ds64[doc*32+q],v);dsbound[doc*32+q]=std::max(dsbound[doc*32+q],bound);}
  for(int q=0;q<32;++q)for(size_t i:route[q]){int doc=docs[i];if(doc<0||doc>=D)continue;size_t t=physical[i];double ip=0,absip=0;for(int d=0;d<128;++d)if((logical[t][d/32]>>(d%32))&1u){ip+=queries[q*128+d];absip+=std::abs(double(queries[q*128+d]));}double v=(ip-cb64[q])*factor[i];double bound=(gamma_n(128)*(absip+.5*absall[q])+gamma_n(3)*(absip+std::abs(cb64[q])))*std::abs(factor[i]);cr64[doc*32+q]=std::max(cr64[doc*32+q],v);crbound[doc*32+q]=std::max(crbound[doc*32+q],bound);}
  for(size_t i=0;i<cr.size();++i){need(std::abs(double(cr[i])-cr64[i])<=crbound[i],"GPU CR doc/query FP64 error");++comparisons;}
  for(int doc=0;doc<D;++doc){double sum=0,scale=0,bound=0;for(int q=0;q<32;++q){sum+=ds64[doc*32+q];scale+=std::abs(ds64[doc*32+q]);bound+=dsbound[doc*32+q];}bound+=gamma_n(32)*(scale+bound);need(std::abs(double(ds[doc])-sum)<=bound,"GPU DS document FP64 scaled error");++comparisons;}
  if(naive){auto mat=material.get();auto md=materialdocs.get();for(int q=0;q<32;++q)for(size_t j=0;j<route[q].size();++j){size_t phys=route[q][j],logical_id=physical[phys];need(md[packed[q]+j]==docs[phys],"Materialized doc mapping");need(bits(mat[packed[q]+j])==bits(matrix[q*N+logical_id]),"Materialized token/DS score mismatch");++comparisons;}}
  std::cout<<"[CASE] scenario="<<scenario<<" direct="<<direct<<" no_aggregation="<<naive<<" packed_tokens="<<M<<" PASS"<<std::endl;
 }
 #undef CRARGS
 }
 ck(cudaDeviceSynchronize());std::cout<<"{\"passed\":true,\"pid\":"<<getpid()<<",\"comparisons\":"<<comparisons<<",\"max_error_over_scaled_bound\":"<<maxratio<<",\"LUT_aggregation_bitwise\":true,\"DS_lengths\":[255,256,257,299,300],\"CR_zero_route_and_invalid_doc\":true}"<<std::endl;
}
