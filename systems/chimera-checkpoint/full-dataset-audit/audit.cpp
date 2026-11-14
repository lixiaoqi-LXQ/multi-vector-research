// Qualification helper only. Same Eigen arithmetic policy as production CUDA TU.
#define EIGEN_DONT_VECTORIZE
#include "chimera/ivf_pg.hpp"
#include "chimera/build_checkpoint.hpp"
#include "chimera/config.cuh"
#define EIGEN_NO_CUDA
#include "rabitqlib/utils/rotator.hpp"
#include <openssl/evp.h>
#include <cuda_runtime_api.h>
#include <algorithm>
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
 dev_t device;ino_t inode;off_t bytes;timespec mtime;
 bool operator==(const Identity& x)const{return device==x.device&&inode==x.inode&&bytes==x.bytes&&mtime.tv_sec==x.mtime.tv_sec&&mtime.tv_nsec==x.mtime.tv_nsec;}
};
Identity identity(const fs::path& p){struct stat s{};need(::stat(p.c_str(),&s)==0,"file stat failed");return {s.st_dev,s.st_ino,s.st_size,s.st_mtim};}
struct Digest {
 std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx{EVP_MD_CTX_new(),EVP_MD_CTX_free};
 Digest(){need(bool(ctx)&&EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"SHA init failed");}
 void add(const void* data,size_t bytes){need(EVP_DigestUpdate(ctx.get(),data,bytes)==1,"SHA update failed");}
 std::string finish(){unsigned char result[EVP_MAX_MD_SIZE];unsigned n=0;need(EVP_DigestFinal_ex(ctx.get(),result,&n)==1&&n==32,"SHA finalize failed");std::ostringstream s;for(unsigned i=0;i<n;++i)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(result[i]);return s.str();}
};
size_t rows_from_bytes(off_t bytes){need(bytes>0&&bytes%(128*sizeof(float))==0,"centroid size must be positive whole Nx128 FP32 rows");const size_t rows=size_t(bytes)/(128*sizeof(float));need(rows<=size_t(std::numeric_limits<int64_t>::max()),"too many centroid rows");return rows;}
int run(int argc,char**argv){
 if(argc==3&&std::string(argv[1])=="--inspect-centroids"){auto s=identity(argv[2]);std::cout<<"{\"rows\":"<<rows_from_bytes(s.bytes)<<",\"dimension\":128,\"bytes\":"<<s.bytes<<",\"GPU_used\":false}\n";return 0;}
 if(argc!=4){std::cout<<"Usage: chimera_saved_dataset_audit <cached FP32 Nx128 centroids.f32> <final centroids.carga> <fresh output directory>\nRows inferred from exact file size; supports prefix/c2097152/c4194304.\n";return argc==1?0:2;}
 need(std::getenv("UMA_RUN_GPU_TESTS")&&std::string(std::getenv("UMA_RUN_GPU_TESTS"))=="1","explicit UMA_RUN_GPU_TESTS=1 required");
 fs::path cached=argv[1],graph_path=argv[2],out=argv[3];need(!fs::exists(out),"refuse existing output directory");
 const auto source_id=identity(cached),graph_id=identity(graph_path);constexpr size_t dimension=128,batch_rows=65536;
 const size_t rows=rows_from_bytes(source_id.bytes);
 // Graph SHA is taken before native load and its exact stat identity is held through validation.
 const auto graph_sha=checkpoint_file_sha256(graph_path);
 PG_CAGRA loaded(rows,PADDED_DIM);loaded.load(graph_path.string());ck(cudaDeviceSynchronize());auto dataset=loaded.index_cagra->dataset();
 need(size_t(loaded.index_cagra->size())==rows&&loaded.index_cagra->dim()==PADDED_DIM&&loaded.index_cagra->graph_degree()==64,"final graph shape/default degree differs from centroid matrix");
 cudaPointerAttributes attrs{};ck(cudaPointerGetAttributes(&attrs,dataset.data_handle()));
 auto rotator=std::unique_ptr<rabitqlib::Rotator<float>>(rabitqlib::choose_rotator_with_seed<float>(dimension,rabitqlib::RotatorType::FhtKacRotator,PADDED_DIM,42));
 // Three bounded 32MiB host chunks; no corpus-sized expected/actual host allocation.
 const size_t capacity=std::min(rows,batch_rows);std::vector<float> source(capacity*dimension),expected(capacity*PADDED_DIM),actual(capacity*PADDED_DIM);
 Digest source_digest,expected_digest,actual_digest;size_t bit_mismatches=0,checked=0;
 std::ifstream input(cached,std::ios::binary);need(bool(input),"centroid source open failed");
 while(checked<rows){
  const size_t count=std::min(batch_rows,rows-checked);input.read(reinterpret_cast<char*>(source.data()),count*dimension*sizeof(float));need(bool(input),"short centroid chunk read");source_digest.add(source.data(),count*dimension*sizeof(float));
  #pragma omp parallel for schedule(static)
  for(int64_t c=0;c<int64_t(count);++c)rotator->rotate(source.data()+size_t(c)*dimension,expected.data()+size_t(c)*PADDED_DIM);
  ck(cudaMemcpy2D(actual.data(),PADDED_DIM*sizeof(float),dataset.data_handle()+checked*dataset.stride(0),dataset.stride(0)*sizeof(float),PADDED_DIM*sizeof(float),count,cudaMemcpyDefault));
  for(size_t i=0;i<count*PADDED_DIM;++i)bit_mismatches+=std::memcmp(&expected[i],&actual[i],sizeof(float))!=0;
  expected_digest.add(expected.data(),count*PADDED_DIM*sizeof(float));actual_digest.add(actual.data(),count*PADDED_DIM*sizeof(float));checked+=count;
  std::cout<<"[AUDIT] rows="<<checked<<"/"<<rows<<" float32_bit_mismatches="<<bit_mismatches<<std::endl;
 }
 need(identity(cached)==source_id&&identity(graph_path)==graph_id,"audit source/graph changed during validation");
 const auto source_sha=source_digest.finish(),expected_sha=expected_digest.finish(),actual_sha=actual_digest.finish();const bool passed=checked==rows&&expected_sha==actual_sha&&bit_mismatches==0&&attrs.type==cudaMemoryTypeDevice;
 fs::create_directories(out);std::ofstream f(out/"result.json");f<<"{\n\"status\":\""<<(passed?"PASS_POST_ENCODE_SAVE_DATASET":"FAIL_POST_ENCODE_SAVE_DATASET")<<"\",\n\"pid\":"<<getpid()<<",\n\"rows\":"<<rows<<",\n\"checked_rows\":"<<checked<<",\n\"dimension\":"<<PADDED_DIM<<",\n\"rotation_seed\":42,\n\"rotator\":\"FhtKacRotator\",\n\"source_rows_inferred_from_exact_Nx128_f32_bytes\":true,\n\"centroid_source_bytes\":"<<source_id.bytes<<",\n\"centroid_source_sha256\":\""<<source_sha<<"\",\n\"final_cagra_file_sha256\":\""<<graph_sha<<"\",\n\"expected_rotated_dataset_sha256\":\""<<expected_sha<<"\",\n\"actual_saved_dataset_sha256\":\""<<actual_sha<<"\",\n\"float32_bit_mismatches\":"<<bit_mismatches<<",\n\"dataset_stride\":"<<dataset.stride(0)<<",\n\"dataset_memory_type\":"<<int(attrs.type)<<",\n\"dataset_device\":"<<attrs.device<<",\n\"host_chunk_rows\":"<<batch_rows<<",\n\"source_and_graph_stat_identity_preserved\":true,\n\"Eigen_vectorization_disabled_like_production\":true\n}\n";f.close();need(bool(f),"receipt output failed");std::cout<<"[RESULT] "<<(passed?"PASS":"FAIL")<<" post_encode_save_dataset rows="<<rows<<std::endl;return passed?0:1;
}
int main(int argc,char**argv){try{return run(argc,argv);}catch(const std::exception& e){std::cerr<<"[ERROR] "<<e.what()<<"\n";return 2;}}
