#include "chimera/binary_ip_reference.hpp"
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cassert>
#include <cstring>
#include <limits>
using namespace Chimera;
struct Token {std::array<uint32_t,4> bits;float factor;int doc;};
uint32_t raw(float x){uint32_t r;memcpy(&r,&x,4);return r;}
double gamma_n(int n){double u=std::numeric_limits<float>::epsilon()/2;return n*u/(1-n*u);}
float tree(std::array<float,32> x){for(int d=16;d;d>>=1)for(int q=0;q<d;++q)x[q]+=x[q+d];return x[0];}
size_t virtual_pos(size_t start,size_t size,size_t local){size_t mis=start&7,head=mis?8-mis:0;if(!head||head>=size)return start+local;size_t run=size-head;return local<run?start+head+local:start+local-run;}
int main(){std::mt19937 gen(42);std::uniform_real_distribution<float> rand(-1,1);size_t cases=24,token_checks=0,agg_checks=0,topdiff=0,zero_clips=0,doc_checks=0;double max_doc_ratio=0,max_doc_error=0,max_ratio=0,max_abs_lut=0,max_abs_direct=0;
 for(size_t c=0;c<cases;++c){std::vector<int>lens={1,7,8,9,31,32,33,127,128,129,255,256,257,299,300};std::rotate(lens.begin(),lens.begin()+c%lens.size(),lens.end());lens[0]=300;lens[1]=257;lens[2]=255;std::vector<size_t>offsets(1,0);for(int n:lens)offsets.push_back(offsets.back()+n);size_t N=offsets.back(),D=lens.size();std::vector<Token>tok(N);for(size_t d=0;d<D;++d)for(size_t t=offsets[d];t<offsets[d+1];++t){tok[t].doc=d;for(auto&w:tok[t].bits)w=gen();tok[t].factor=std::abs(rand(gen))*3;if(d==0){tok[t].bits.fill(0);tok[t].factor=1;}if(d==1){tok[t].bits.fill(~0u);tok[t].factor=1;}if(d==2)tok[t].factor=0;}
 std::array<std::array<float,128>,32>q{};std::array<float,32>cb{};std::array<double,32>cb64{},sumabs{};
 for(int j=0;j<32;++j){float sum=0;for(int d=0;d<128;++d){q[j][d]=j==0?1:j==1?-1:j==2?0:rand(gen)*std::ldexp(1.0f,j%15-7);sum+=q[j][d];cb64[j]+=.5*double(q[j][d]);sumabs[j]+=std::abs(double(q[j][d]));}cb[j]=sum*.5f;}
 std::vector<std::vector<size_t>>cluster(17);for(size_t t=0;t<N;++t)cluster[1+gen()%16].push_back(t);std::vector<size_t>starts(1,0),physical;for(auto&v:cluster){physical.insert(physical.end(),v.begin(),v.end());starts.push_back(physical.size());}std::array<std::vector<size_t>,32>routes;
 for(int j=0;j<32;++j)for(int p=0;p<7;++p){int label=(j*5+p*3)%18;if(label>=17 || j==3)continue;size_t n=starts[label+1]-starts[label];for(size_t i=0;i<n;++i)routes[j].push_back(physical[virtual_pos(starts[label],n,i)]);}
 std::vector<float>cr[2][2],ds[2][2];std::vector<double>ds64(D*32,-std::numeric_limits<double>::infinity()),cr64(D*32,0),bounds(N*32);std::vector<float>allscore[2];for(auto&v:allscore)v.resize(N*32);
 for(int j=0;j<32;++j)for(size_t t=0;t<N;++t){double ip=0,absip=0;for(int d=0;d<128;++d)if((tok[t].bits[d/32]>>(d%32))&1u){ip+=double(q[j][d]);absip+=std::abs(double(q[j][d]));}double value=(ip-cb64[j])*tok[t].factor;double bound=(gamma_n(128)*(absip+.5*sumabs[j])+gamma_n(3)*(absip+std::abs(cb64[j])))*std::abs(tok[t].factor);bounds[j*N+t]=bound;ds64[tok[t].doc*32+j]=std::max(ds64[tok[t].doc*32+j],value);
  for(int direct=0;direct<2;++direct){float v=((direct?direct_binary_ip(q[j].data(),tok[t].bits.data()):reference_lut_binary_ip(q[j].data(),tok[t].bits.data()))-cb[j])*tok[t].factor;allscore[direct][j*N+t]=v;double error=std::abs(double(v)-value);assert(error<=bound);if(bound)max_ratio=std::max(max_ratio,error/bound);(direct?max_abs_direct:max_abs_lut)=std::max(direct?max_abs_direct:max_abs_lut,error);++token_checks;}}
 for(int direct=0;direct<2;++direct)for(int naive=0;naive<2;++naive){cr[direct][naive].assign(D*32,0);ds[direct][naive].resize(D);std::vector<float>material;std::vector<int>docs;std::array<size_t,33>packed{};
  for(int j=0;j<32;++j){packed[j]=material.size();for(size_t t:routes[j]){float v=allscore[direct][j*N+t];if(naive){material.push_back(v);docs.push_back(tok[t].doc);}else if(v>0)cr[direct][naive][tok[t].doc*32+j]=std::fmax(cr[direct][naive][tok[t].doc*32+j],v);if(direct==0&&naive==0){double ip=0;for(int d=0;d<128;++d)if((tok[t].bits[d/32]>>(d%32))&1u)ip+=q[j][d];cr64[tok[t].doc*32+j]=std::max(cr64[tok[t].doc*32+j],(ip-cb64[j])*tok[t].factor);}}}packed[32]=material.size();
  if(naive)for(int j=0;j<32;++j)for(size_t t=packed[j];t<packed[j+1];++t)if(material[t]>0)cr[direct][naive][docs[t]*32+j]=std::fmax(cr[direct][naive][docs[t]*32+j],material[t]);
  for(size_t d=0;d<D;++d){std::array<float,32>maxima{};for(int j=0;j<32;++j){maxima[j]=-FLT_MAX;for(size_t base=offsets[d];base<offsets[d+1];base+=8)for(size_t t=base;t<std::min(base+8,offsets[d+1]);++t)maxima[j]=std::fmax(maxima[j],allscore[direct][j*N+t]);}for(auto&v:maxima){float sum=0.0f;sum+=v;v=sum;}ds[direct][naive][d]=naive?naive_document_score_body(allscore[direct].data(),offsets[d],offsets[d+1],N):tree(maxima);if(d==0)assert(maxima[0]<0);}
 }
 for(int direct=0;direct<2;++direct){for(size_t i=0;i<D*32;++i){assert(raw(cr[direct][0][i])==raw(cr[direct][1][i]));++agg_checks;}for(size_t d=0;d<D;++d){assert(raw(ds[direct][0][d])==raw(ds[direct][1][d]));++agg_checks;}}
 for(int direct=0;direct<2;++direct)for(size_t d=0;d<D;++d)for(int j=0;j<32;++j){double bound=0;for(size_t t=offsets[d];t<offsets[d+1];++t)bound=std::max(bound,bounds[j*N+t]);assert(std::abs(double(cr[direct][0][d*32+j])-cr64[d*32+j])<=bound);if(j==3)assert(cr[direct][0][d*32+j]==0);if(j==0&&d==0){assert(cr[direct][0][0]==0);++zero_clips;}}
 for(int direct=0;direct<2;++direct)for(size_t d=0;d<D;++d){double expected=0,sumabsmax=0,sumerr=0;for(int j=0;j<32;++j){expected+=ds64[d*32+j];sumabsmax+=std::abs(ds64[d*32+j]);double bound=0;for(size_t t=offsets[d];t<offsets[d+1];++t)bound=std::max(bound,bounds[j*N+t]);sumerr+=bound;}double bound=sumerr+gamma_n(32)*(sumabsmax+sumerr);double error=std::abs(double(ds[direct][0][d])-expected);assert(error<=bound);if(bound)max_doc_ratio=std::max(max_doc_ratio,error/bound);max_doc_error=std::max(max_doc_error,error);++doc_checks;}
 auto order=[&](int direct){std::vector<int>x(D);std::iota(x.begin(),x.end(),0);std::sort(x.begin(),x.end(),[&](int a,int b){return std::pair<float,int>(ds[direct][0][a],a)>std::pair<float,int>(ds[direct][0][b],b);});x.resize(10);std::sort(x.begin(),x.end());return x;};topdiff+=order(0)!=order(1);
 }
 std::vector<float>negative_zero_matrix(300*32,-0.0f);assert(raw(naive_document_score_body(negative_zero_matrix.data(),0,300,300))==0u);
 std::cout<<"{\"CPU_only\":true,\"cases\":"<<cases<<",\"token_FP64_bound_checks\":"<<token_checks<<",\"aggregation_bitwise_checks\":"<<agg_checks<<",\"CR_negative_zero_clips\":"<<zero_clips<<",\"max_error_over_scaled_bound\":"<<max_ratio<<",\"max_LUT_token_abs_error\":"<<max_abs_lut<<",\"max_direct_token_abs_error\":"<<max_abs_direct<<",\"document_FP64_scaled_bound_checks\":"<<doc_checks<<",\"max_document_error_over_bound\":"<<max_doc_ratio<<",\"max_document_abs_error\":"<<max_doc_error<<",\"synthetic_DS_top10_set_changes\":"<<topdiff<<",\"DS_negative_qmax_verified\":true}"<<std::endl;
}
