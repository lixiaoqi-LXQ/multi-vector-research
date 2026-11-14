#include "chimera/ivf_pg.hpp"
#include "chimera/fp16_input.hpp"
#include "mapped_fp16_npy.hpp"
#include <openssl/evp.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
namespace fs=std::filesystem;
using namespace Chimera;
void need(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Identity {
 dev_t device;ino_t inode;off_t bytes;timespec mtime,ctime;
 bool operator==(const Identity& x)const{return device==x.device&&inode==x.inode&&bytes==x.bytes&&mtime.tv_sec==x.mtime.tv_sec&&mtime.tv_nsec==x.mtime.tv_nsec&&ctime.tv_sec==x.ctime.tv_sec&&ctime.tv_nsec==x.ctime.tv_nsec;}
};
Identity identity(const fs::path& p){struct stat s{};need(::stat(p.c_str(),&s)==0,"file stat failed");return {s.st_dev,s.st_ino,s.st_size,s.st_mtim,s.st_ctim};}
struct Digest {
 std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx{EVP_MD_CTX_new(),EVP_MD_CTX_free};
 Digest(){need(bool(ctx)&&EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"SHA init failed");}
 void add(const void* data,size_t bytes){need(EVP_DigestUpdate(ctx.get(),data,bytes)==1,"SHA update failed");}
 std::string finish(){unsigned char result[EVP_MAX_MD_SIZE];unsigned n=0;need(EVP_DigestFinal_ex(ctx.get(),result,&n)==1&&n==32,"SHA finalize failed");std::ostringstream s;for(unsigned i=0;i<n;++i)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(result[i]);return s.str();}
};
std::string file_sha(const fs::path& p){std::ifstream f(p,std::ios::binary);need(bool(f),"SHA input open failed");Digest d;std::vector<char>b(8*1024*1024);while(f){f.read(b.data(),b.size());d.add(b.data(),size_t(f.gcount()));}need(f.eof(),"SHA read failed");return d.finish();}
std::string stat_json(const Identity& s){std::ostringstream o;o<<"{\"device\":"<<s.device<<",\"inode\":"<<s.inode<<",\"bytes\":"<<s.bytes<<",\"mtime_ns\":"<<(uint64_t(s.mtime.tv_sec)*1000000000+s.mtime.tv_nsec)<<",\"ctime_ns\":"<<(uint64_t(s.ctime.tv_sec)*1000000000+s.ctime.tv_nsec)<<"}";return o.str();}
struct Device {void* p=nullptr;explicit Device(size_t bytes){ck(cudaMalloc(&p,bytes));}~Device(){if(p)cudaFree(p);}template<class T>T* as(){return static_cast<T*>(p);}};
int run(int argc,char**argv){
 const bool inspect=argc>1&&std::string(argv[1])=="--inspect-inputs";
 if((inspect&&argc!=6)||(!inspect&&argc!=6)){std::cout<<"Usage: ms_q1_restore_audit [--inspect-inputs] <centroids.f32> <assignment.cagra> <full documents.npy> <labels/0.u32> [fresh output dir]\n";return argc==1?0:2;}
 const int shift=inspect?1:0;fs::path cached=argv[1+shift],graph_path=argv[2+shift],raw=argv[3+shift],labels=argv[4+shift];
 constexpr size_t rows=4194304,dim=128,batch=65536,chunkrows=65536;
 auto cs=identity(cached),gs=identity(graph_path),rs=identity(raw),ls=identity(labels);
 need(cs.bytes==off_t(rows*dim*sizeof(float)),"Expected full C4M x128 FP32 centroid matrix (2GiB)");need(gs.bytes==3221227188LL,"Expected Q1 full C4M assignment graph bytes");need(ls.bytes==off_t(batch*sizeof(uint32_t)),"Expected complete 65536-label Q1 first chunk");need(rs.bytes==153064939392LL,"Expected complete pinned MS corpus NPY bytes");
 MappedFp16Npy source(raw.string(),batch);need(source.dimension()==dim,"Expected FP16 NPY dimension128");
 if(inspect){std::cout<<"{\"status\":\"PASS_INPUT_METADATA_ONLY\",\"GPU_used\":false,\"rows\":"<<rows<<",\"dimension\":"<<dim<<",\"query_rows\":"<<batch<<"}\n";return 0;}
 fs::path out=argv[5];need(!fs::exists(out),"refuse existing output directory");need(std::getenv("UMA_RUN_GPU_TESTS")&&std::string(std::getenv("UMA_RUN_GPU_TESTS"))=="1","explicit UMA_RUN_GPU_TESTS=1 required");
 need(fs::create_directory(out),"exclusive output reservation failed");
 const auto graph_sha=file_sha(graph_path),label_sha=file_sha(labels);
 PG_CAGRA loaded(rows,dim);loaded.load(graph_path.string());ck(cudaDeviceSynchronize());auto dataset=loaded.index_cagra->dataset();
 need(size_t(loaded.index_cagra->size())==rows&&loaded.index_cagra->dim()==dim&&loaded.index_cagra->graph_degree()==64,"loaded graph shape/degree mismatch");need(dataset.stride(0)>=dim&&dataset.stride(1)==1,"invalid dataset stride");cudaPointerAttributes attrs{};ck(cudaPointerGetAttributes(&attrs,dataset.data_handle()));need(attrs.type==cudaMemoryTypeDevice,"loaded dataset must be device-owned");
 size_t mismatches=0,checked=0;Digest expected_hash,actual_hash;{
  std::vector<float> expected(chunkrows*dim),actual(chunkrows*dim);std::ifstream f(cached,std::ios::binary);need(bool(f),"centroid open failed");
  while(checked<rows){size_t n=std::min(chunkrows,rows-checked),bytes=n*dim*sizeof(float);f.read(reinterpret_cast<char*>(expected.data()),bytes);need(bool(f),"short centroid read");ck(cudaMemcpy2D(actual.data(),dim*sizeof(float),dataset.data_handle()+checked*dataset.stride(0),dataset.stride(0)*sizeof(float),dim*sizeof(float),n,cudaMemcpyDeviceToHost));
   expected_hash.add(expected.data(),bytes);actual_hash.add(actual.data(),bytes);for(size_t i=0;i<n*dim;++i)mismatches+=std::memcmp(&expected[i],&actual[i],sizeof(float))!=0;checked+=n;
  }
 }
 auto expected_sha=expected_hash.finish(),actual_sha=actual_hash.finish();need(mismatches==0&&expected_sha==actual_sha,"loaded dataset differs from original unrotated centroids");
 std::vector<float> query(batch*dim),dist(batch);std::vector<uint32_t> actual_labels(batch),expected_labels(batch);need(expand_fp16_to_fp32(source.data(),query.data(),query.size()),"nonfinite raw query prefix");for(float v:query)need(std::isfinite(v),"nonfinite expanded query");Digest qh;qh.add(query.data(),query.size()*sizeof(float));auto query_sha=qh.finish();
 std::ifstream lf(labels,std::ios::binary);lf.read(reinterpret_cast<char*>(expected_labels.data()),batch*sizeof(uint32_t));need(bool(lf),"short labels read");Device dq(query.size()*sizeof(float)),dd(batch*sizeof(float)),dl(batch*sizeof(uint32_t));ck(cudaMemcpy(dq.p,query.data(),query.size()*sizeof(float),cudaMemcpyHostToDevice));loaded.search_centroids(dq.as<float>(),batch,1,dd.as<float>(),dl.as<uint32_t>(),nullptr,150);ck(cudaDeviceSynchronize());ck(cudaMemcpy(dist.data(),dd.p,batch*sizeof(float),cudaMemcpyDeviceToHost));ck(cudaMemcpy(actual_labels.data(),dl.p,batch*sizeof(uint32_t),cudaMemcpyDeviceToHost));
 size_t label_errors=0;for(size_t i=0;i<batch;++i){need(actual_labels[i]<rows&&expected_labels[i]<rows,"label out of range");need(std::isfinite(dist[i]),"nonfinite search distance");label_errors+=actual_labels[i]!=expected_labels[i];}Digest ah,dh;ah.add(actual_labels.data(),batch*sizeof(uint32_t));dh.add(dist.data(),batch*sizeof(float));auto actual_label_sha=ah.finish(),distance_sha=dh.finish();need(label_errors==0&&actual_label_sha==label_sha,"restored same-graph assignment labels differ from Q1");
 need(identity(cached)==cs&&identity(graph_path)==gs&&identity(raw)==rs&&identity(labels)==ls,"input stat changed during audit");std::ofstream f(out/"result.json");f<<"{\n\"status\":\"PASS_FULL_MS_Q1_LARGE_RESTORE\",\n\"pid\":"<<getpid()<<",\n\"rows\":"<<rows<<",\n\"checked_rows\":"<<checked<<",\n\"dimension\":"<<dim<<",\n\"graph_degree\":64,\n\"dataset_stride\":"<<dataset.stride(0)<<",\n\"dataset_memory_type\":"<<int(attrs.type)<<",\n\"dataset_device\":"<<attrs.device<<",\n\"centroid_sha256\":\""<<expected_sha<<"\",\n\"loaded_dataset_sha256\":\""<<actual_sha<<"\",\n\"assignment_graph_sha256\":\""<<graph_sha<<"\",\n\"float32_bit_mismatches\":"<<mismatches<<",\n\"query_rows\":65536,\n\"query_prefix_fp32_sha256\":\""<<query_sha<<"\",\n\"nprobe\":1,\n\"itopk\":150,\n\"batch\":65536,\n\"label_mismatches\":"<<label_errors<<",\n\"q1_labels_sha256\":\""<<label_sha<<"\",\n\"restored_labels_sha256\":\""<<actual_label_sha<<"\",\n\"finite_distances\":65536,\n\"distance_sha256\":\""<<distance_sha<<"\",\n\"centroid_stat\":"<<stat_json(cs)<<",\n\"graph_stat\":"<<stat_json(gs)<<",\n\"raw_stat\":"<<stat_json(rs)<<",\n\"labels_stat\":"<<stat_json(ls)<<",\n\"source_stat_preserved\":true,\n\"host_compare_capacity_bytes\":67108864,\n\"same_loaded_graph_used_for_dataset_and_assignment\":true\n}\n";f.close();need(bool(f),"receipt write failed");std::cout<<"PASS fullC4M dataset and 65536 labels\n";return 0;
}
int main(int argc,char**argv){try{return run(argc,argv);}catch(const std::exception&e){std::cerr<<"[ERROR] "<<e.what()<<"\n";return 2;}}
