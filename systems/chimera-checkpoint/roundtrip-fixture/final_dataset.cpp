// Match the production CUDA translation unit's Eigen arithmetic policy.
#define EIGEN_DONT_VECTORIZE
// Qualification only: inspect the saved graph after the complete encode/save path.
#include "chimera/ivf_pg.hpp"
#include "chimera/build_checkpoint.hpp"
#include "chimera/config.cuh"
#define EIGEN_NO_CUDA
#include "rabitqlib/utils/rotator.hpp"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <unistd.h>
namespace fs=std::filesystem;
using namespace Chimera;
void need(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int run(int argc,char** argv){
 if(argc!=4){std::cout<<"Usage: checkpoint_final_dataset <cached8192x128 centroids.f32> <final centroids.carga> <fresh output directory>\n";return argc==1?0:2;}
 need(std::getenv("UMA_RUN_GPU_TESTS")&&std::string(std::getenv("UMA_RUN_GPU_TESTS"))=="1","explicit UMA_RUN_GPU_TESTS=1 required");
 const fs::path cached=argv[1],graph_path=argv[2],out=argv[3];need(!fs::exists(out),"refuse existing output directory");
 need(fs::file_size(cached)==8192ull*128*sizeof(float),"expected c8192x128 centroid matrix");
 std::vector<float> centroids(8192*128),expected(8192*PADDED_DIM);std::ifstream in(cached,std::ios::binary);in.read(reinterpret_cast<char*>(centroids.data()),centroids.size()*sizeof(float));need(bool(in),"short centroid read");
 auto rotator=std::unique_ptr<rabitqlib::Rotator<float>>(rabitqlib::choose_rotator_with_seed<float>(128,rabitqlib::RotatorType::FhtKacRotator,PADDED_DIM,42));
 // Same cluster row order and seed42 operation as build_index.cu::rotate_centroids.
 #pragma omp parallel for schedule(static)
 for(int64_t c=0;c<8192;++c)rotator->rotate(centroids.data()+c*128,expected.data()+c*PADDED_DIM);
 const auto expected_sha=checkpoint_sha256(expected.data(),expected.size()*sizeof(float));
 PG_CAGRA loaded(8192,PADDED_DIM);loaded.load(graph_path.string());ck(cudaDeviceSynchronize());auto dataset=loaded.index_cagra->dataset();
 need(loaded.index_cagra->size()==8192&&loaded.index_cagra->dim()==PADDED_DIM&&loaded.index_cagra->graph_degree()==64,"unexpected final graph shape/defaultdegree");
 std::vector<float> actual(expected.size());ck(cudaMemcpy2D(actual.data(),PADDED_DIM*sizeof(float),dataset.data_handle(),dataset.stride(0)*sizeof(float),PADDED_DIM*sizeof(float),8192,cudaMemcpyDefault));
 const auto actual_sha=checkpoint_sha256(actual.data(),actual.size()*sizeof(float));
 size_t bit_mismatches=0;for(size_t i=0;i<actual.size();++i)bit_mismatches+=std::memcmp(&expected[i],&actual[i],sizeof(float))!=0;
 cudaPointerAttributes a{};ck(cudaPointerGetAttributes(&a,dataset.data_handle()));
 const bool passed=expected_sha==actual_sha&&bit_mismatches==0&&a.type==cudaMemoryTypeDevice;
 fs::create_directories(out);std::ofstream receipt(out/"result.json");receipt<<"{\n\"status\":\""<<(passed?"PASS_POST_ENCODE_SAVE_DATASET":"FAIL_POST_ENCODE_SAVE_DATASET")<<"\",\n\"pid\":"<<getpid()<<",\n\"rows\":8192,\n\"dimension\":"<<PADDED_DIM<<",\n\"rotation_seed\":42,\n\"rotator\":\"FhtKacRotator\",\n\"centroid_source_sha256\":\""<<checkpoint_file_sha256(cached)<<"\",\n\"final_cagra_file_sha256\":\""<<checkpoint_file_sha256(graph_path)<<"\",\n\"expected_rotated_dataset_sha256\":\""<<expected_sha<<"\",\n\"actual_saved_dataset_sha256\":\""<<actual_sha<<"\",\n\"float32_bit_mismatches\":"<<bit_mismatches<<",\n\"dataset_stride\":"<<dataset.stride(0)<<",\n\"dataset_memory_type\":"<<int(a.type)<<",\n\"dataset_device\":"<<a.device<<"\n}\n";receipt.close();need(bool(receipt),"receipt write failed");
 std::cout<<"[RESULT] "<<(passed?"PASS":"FAIL")<<" post_encode_save_dataset\n";return passed?0:1;
}
int main(int argc,char**argv){try{return run(argc,argv);}catch(const std::exception& e){std::cerr<<"[ERROR] "<<e.what()<<"\n";return 2;}}
