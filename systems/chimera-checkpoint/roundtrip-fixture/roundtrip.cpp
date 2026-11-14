#include "chimera/ivf_pg.hpp"
#include "chimera/fp16_input.hpp"
#include "chimera/build_checkpoint.hpp"
#include "mapped_fp16_npy.hpp"
#include <cuda_runtime_api.h>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <utility>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <sstream>
#include <unistd.h>
namespace fs=std::filesystem;
using namespace Chimera;
void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void ck(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
template<class T> struct Device {
 T* p=nullptr;
 explicit Device(size_t n){ck(cudaMalloc(&p,n*sizeof(T)));}
 ~Device(){if(p)cudaFree(p);}
 Device(const Device&)=delete;
};
template<class T> void save(const fs::path& file,const std::vector<T>& values){
 need(!fs::exists(file),"refuse existing fixture payload");std::ofstream f(file,std::ios::binary);f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));f.close();need(bool(f),"fixture output failed");
}
struct PointerAttributes {
 const void* pointer=nullptr;int status=0,type=-1,device=-2;void* device_pointer=nullptr;void* host_pointer=nullptr;
};
PointerAttributes pointer_attributes(const void* pointer){
 cudaPointerAttributes a{};auto code=cudaPointerGetAttributes(&a,pointer);
 if(code!=cudaSuccess){cudaGetLastError();return {pointer,int(code),-1,-2,nullptr,nullptr};}
 return {pointer,int(code),int(a.type),a.device,a.devicePointer,a.hostPointer};
}
void json_attributes(std::ostream& f,const PointerAttributes& a){
 f<<"{\"pointer\":\""<<a.pointer<<"\",\"status\":"<<a.status<<",\"type\":"<<a.type<<",\"device\":"<<a.device<<",\"devicePointer\":\""<<a.device_pointer<<"\",\"hostPointer\":\""<<a.host_pointer<<"\"}";
}
struct Snapshot{size_t rows,dimension,degree,stride;std::string dataset_sha,graph_sha;PointerAttributes dataset_attributes;};
Snapshot snapshot(PG_CAGRA& engine){
 auto& index=*engine.index_cagra;auto dataset=index.dataset();auto graph=index.graph();
 need(dataset.extent(0)==index.size()&&dataset.extent(1)==index.dim()&&graph.extent(0)==index.size(),"unexpected index view shape");
 std::vector<float> data(index.size()*index.dim());std::vector<uint32_t> edges(index.size()*index.graph_degree());
 ck(cudaMemcpy2D(data.data(),index.dim()*sizeof(float),dataset.data_handle(),dataset.stride(0)*sizeof(float),index.dim()*sizeof(float),index.size(),cudaMemcpyDefault));
 ck(cudaMemcpy2D(edges.data(),index.graph_degree()*sizeof(uint32_t),graph.data_handle(),graph.stride(0)*sizeof(uint32_t),index.graph_degree()*sizeof(uint32_t),index.size(),cudaMemcpyDefault));
 return {size_t(index.size()),size_t(index.dim()),size_t(index.graph_degree()),size_t(dataset.stride(0)),checkpoint_sha256(data.data(),data.size()*sizeof(float)),checkpoint_sha256(edges.data(),edges.size()*sizeof(uint32_t)),pointer_attributes(dataset.data_handle())};
}
struct Ranking{std::vector<uint32_t> labels;std::vector<float> distances;double elapsed_ms=0;};
Ranking search(PG_CAGRA& engine,const MappedFp16Npy& raw,size_t start,size_t count){
 std::vector<float> queries(count*128);need(expand_fp16_to_fp32(raw.data()+start*128,queries.data(),queries.size()),"nonfinite query fixture");
 Device<float> q(queries.size()),d(count);Device<uint32_t> labels(count);
 ck(cudaMemcpy(q.p,queries.data(),queries.size()*sizeof(float),cudaMemcpyHostToDevice));
 const auto started=std::chrono::steady_clock::now();
 // Exactly the production assignment API, k1 / itopk150 / original batch boundary.
 engine.search_centroids(q.p,count,1,d.p,labels.p,nullptr,150);
 Ranking result{std::vector<uint32_t>(count),std::vector<float>(count)};
 ck(cudaMemcpy(result.labels.data(),labels.p,count*sizeof(uint32_t),cudaMemcpyDeviceToHost));
 ck(cudaMemcpy(result.distances.data(),d.p,count*sizeof(float),cudaMemcpyDeviceToHost));
 result.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
 for(size_t i=0;i<count;++i)need(result.labels[i]<engine.n&&std::isfinite(result.distances[i]),"invalid rank result");
 return result;
}
struct Difference{size_t labels=0,distances=0;float max_error=0;};
Difference compare(const Ranking& a,const Ranking& b){
 need(a.labels.size()==b.labels.size(),"ranking shapes differ");Difference r;
 for(size_t i=0;i<a.labels.size();++i){r.labels+=a.labels[i]!=b.labels[i];r.distances+=std::memcmp(&a.distances[i],&b.distances[i],sizeof(float))!=0;r.max_error=std::max(r.max_error,std::abs(a.distances[i]-b.distances[i]));}return r;
}
void json_difference(std::ostream& f,const Difference& d){f<<"{\"label_mismatches\":"<<d.labels<<",\"distance_bit_mismatches\":"<<d.distances<<",\"max_distance_error\":"<<d.max_error<<"}";}
int run_fixture(int argc,char**argv){
 if(argc!=4){std::cout<<"Usage: checkpoint_graph_roundtrip <cached centroids.f32> <qualified1087049-token FP16 NPY> <fresh output dir>\n";return argc==1?0:2;}
 need(std::getenv("UMA_RUN_GPU_TESTS")&&std::string(std::getenv("UMA_RUN_GPU_TESTS"))=="1","explicit UMA_RUN_GPU_TESTS=1 required");
 fs::path centroids_path=argv[1],query_path=argv[2],output=argv[3];need(!fs::exists(output),"refuse existing output");need(fs::file_size(centroids_path)==8192ull*128*sizeof(float),"expected c8192 cached FP32 matrix");
 MappedFp16Npy raw(query_path.string());need(raw.dimension()==128&&raw.size()/128==1087049,"expected exact qualified prefix input");
 fs::create_directories(output);std::vector<float> centroids(8192*128);std::ifstream in(centroids_path,std::ios::binary);in.read(reinterpret_cast<char*>(centroids.data()),centroids.size()*sizeof(float));need(bool(in),"centroid input short");
 const auto expected_dataset_sha=checkpoint_sha256(centroids.data(),centroids.size()*sizeof(float));
 const auto original_host_pointer=centroids.data();
 std::cout<<"[PHASE] live_graph_build_start pid="<<getpid()<<std::endl;PG_CAGRA live(8192,128);live.build_index(centroids.data());ck(cudaDeviceSynchronize());auto before=snapshot(live);need(before.rows==8192&&before.dimension==128&&before.degree==64,"unexpected live graph defaults");
 need(before.dataset_sha==expected_dataset_sha,"built graph dataset differs from exact caller values");
 need(before.dataset_attributes.status==0&&before.dataset_attributes.type==int(cudaMemoryTypeDevice),"live dataset must be device allocated");
 need(before.dataset_attributes.pointer!=original_host_pointer,"live dataset aliases caller host vector");
 need(live.explicit_dataset_.get()==before.dataset_attributes.pointer,"live graph does not reference the RAII-owned device dataset");
 std::vector<float>().swap(centroids);
 // Same-sized live allocations overwrite reusable caller storage; keep churn alive.
 std::vector<std::vector<float>> churn;
 for(size_t i=0;i<16;++i)churn.emplace_back(8192*128,-1000.0f-float(i));
 ck(cudaDeviceSynchronize());auto after_caller_free=snapshot(live);
 need(after_caller_free.dataset_sha==expected_dataset_sha&&after_caller_free.graph_sha==before.graph_sha,"dataset/graph changed after caller free and allocator churn");
 std::cout<<"[PHASE] caller_freed_allocator_churn_verified first_assignment_before"<<std::endl;auto first_before=search(live,raw,0,65536);auto tail_before=search(live,raw,1048576,38473);
 // Native save/load, including the candidate's pre-created empty temporary file behavior.
 auto graph_path=output/"same-live-graph.cagra";std::ofstream(graph_path,std::ios::binary).close();live.save(graph_path.string());ck(cudaDeviceSynchronize());
 std::cout<<"[PHASE] graph_serialized load_same_graph"<<std::endl;PG_CAGRA loaded(8192,128);loaded.load(graph_path.string());ck(cudaDeviceSynchronize());
 // Reload the same object, exercising old-index-before-owner cleanup.
 loaded.load(graph_path.string());ck(cudaDeviceSynchronize());need(!loaded.explicit_dataset_,"loaded graph retained unused explicit build allocation");auto after=snapshot(loaded),live_after=snapshot(live);
 need(after.dataset_attributes.status==0&&after.dataset_attributes.type==int(cudaMemoryTypeDevice),"loaded dataset must be device allocated");
 auto first_loaded=search(loaded,raw,0,65536);auto tail_loaded=search(loaded,raw,1048576,38473);auto first_repeat=search(live,raw,0,65536);auto tail_repeat=search(live,raw,1048576,38473);
 // Exercise load on an object that currently owns an explicit build allocation.
 live.load(graph_path.string());ck(cudaDeviceSynchronize());auto live_reloaded=snapshot(live);
 need(!live.explicit_dataset_,"loading built object retained old explicit allocation");
 need(live_reloaded.dataset_sha==expected_dataset_sha&&live_reloaded.graph_sha==before.graph_sha,"build-to-load object transition changed payloads");
 bool payloads=before.dataset_sha==expected_dataset_sha&&after_caller_free.dataset_sha==expected_dataset_sha&&after_caller_free.graph_sha==before.graph_sha&&before.rows==after.rows&&before.dimension==after.dimension&&before.degree==after.degree&&before.dataset_sha==after.dataset_sha&&before.graph_sha==after.graph_sha&&before.dataset_sha==live_after.dataset_sha&&before.graph_sha==live_after.graph_sha;
 auto first_diff=compare(first_before,first_loaded),tail_diff=compare(tail_before,tail_loaded),repeat_first_diff=compare(first_before,first_repeat),repeat_tail_diff=compare(tail_before,tail_repeat);
 std::ofstream placement(output/"placement-and-query-regions.json");
 placement<<"{\n\"live_dataset_attributes\":";json_attributes(placement,before.dataset_attributes);
 placement<<",\n\"loaded_dataset_attributes\":";json_attributes(placement,after.dataset_attributes);
 placement<<",\n\"live_dataset_pointer_equals_original_host_centroids\":"<<(before.dataset_attributes.pointer==original_host_pointer?"true":"false");
 placement<<",\n\"loaded_dataset_pointer_equals_original_host_centroids\":"<<(after.dataset_attributes.pointer==original_host_pointer?"true":"false");
 placement<<",\n\"logical_payload_snapshot_copy_kind\":\"cudaMemcpyDefault (UVA infers host/device direction)\",\n\"timer_scope\":\"wall search_centroids through synchronous labels/distances return; query preparation/device allocation/upload excluded; host-result allocation included\",\n\"diagnostic_only_not_formal_performance\":true,\n\"query_region_ms\":{\"first_before\":"<<first_before.elapsed_ms<<",\"tail_before\":"<<tail_before.elapsed_ms<<",\"first_loaded\":"<<first_loaded.elapsed_ms<<",\"tail_loaded\":"<<tail_loaded.elapsed_ms<<",\"first_live_repeat\":"<<first_repeat.elapsed_ms<<",\"tail_live_repeat\":"<<tail_repeat.elapsed_ms<<"}\n}\n";
 placement.close();need(bool(placement),"placement metadata output failed");
 for(auto pair:{std::pair<const char*,const Ranking*>{"first-before",&first_before},{"tail-before",&tail_before},{"first-loaded",&first_loaded},{"tail-loaded",&tail_loaded},{"first-live-repeat",&first_repeat},{"tail-live-repeat",&tail_repeat}}){save(output/(std::string(pair.first)+".u32"),pair.second->labels);save(output/(std::string(pair.first)+".f32"),pair.second->distances);}
 bool exact=payloads&&first_diff.labels==0&&first_diff.distances==0&&tail_diff.labels==0&&tail_diff.distances==0&&repeat_first_diff.labels==0&&repeat_first_diff.distances==0&&repeat_tail_diff.labels==0&&repeat_tail_diff.distances==0;
 std::ofstream result(output/"result.json");result<<"{\n\"status\":\""<<(exact?"PASS_SAME_LIVE_GRAPH_ROUNDTRIP":"FAIL_SAME_LIVE_GRAPH_ROUNDTRIP")<<"\",\n\"pid\":"<<getpid()<<",\n\"rows\":8192,\n\"dimension\":128,\n\"degree\":64,\n\"assignment_k\":1,\n\"itopk\":150,\n\"first_query_count\":65536,\n\"tail_start\":1048576,\n\"tail_query_count\":38473,\n\"logical_dataset_graph_payloads_exact\":"<<(payloads?"true":"false")<<",\n\"caller_host_buffer_freed_before_queries\":true,\n\"allocator_churn_live_allocations\":16,\n\"build_owned_object_then_load_cleanup_checked\":true,\n\"expected_dataset_sha256\":\""<<expected_dataset_sha<<"\",\n\"dataset_sha_after_caller_free_churn\":\""<<after_caller_free.dataset_sha<<"\",\n\"dataset_stride_before\":"<<before.stride<<",\n\"dataset_stride_loaded\":"<<after.stride<<",\n\"dataset_sha_before\":\""<<before.dataset_sha<<"\",\n\"dataset_sha_loaded\":\""<<after.dataset_sha<<"\",\n\"graph_sha_before\":\""<<before.graph_sha<<"\",\n\"graph_sha_loaded\":\""<<after.graph_sha<<"\",\n\"cached_centroids_sha256\":\""<<checkpoint_file_sha256(centroids_path)<<"\",\n\"serialized_graph_sha256\":\""<<checkpoint_file_sha256(graph_path)<<"\",\n\"first_live_vs_loaded\":";json_difference(result,first_diff);result<<",\n\"tail_live_vs_loaded\":";json_difference(result,tail_diff);result<<",\n\"first_live_repeat\":";json_difference(result,repeat_first_diff);result<<",\n\"tail_live_repeat\":";json_difference(result,repeat_tail_diff);result<<"\n}\n";result.close();need(bool(result),"result output failed");std::cout<<"[RESULT] "<<(exact?"PASS":"FAIL")<<" same_live_graph_roundtrip"<<std::endl;return exact?0:1;
}

int main(int argc,char**argv){try{return run_fixture(argc,argv);}catch(const std::exception& error){std::cerr<<"[ERROR] "<<error.what()<<std::endl;return 2;}}
