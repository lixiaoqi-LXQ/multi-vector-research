#define EIGEN_DONT_VECTORIZE
#define EIGEN_NO_CUDA
#include "chimera/chimera_index.cuh"
#include "chimera/ivf_pg.hpp"
#include "chimera/config.cuh"
#include "chimera/profiling.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include <cuda_profiler_api.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>
using namespace Chimera;
namespace fs=std::filesystem;
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
namespace Chimera {
struct AlgorithmAblationQualification {
 static std::vector<double> quantized_maxsim(const chimera_index& index,const float* query){
    std::vector<float> rotated(Q_DOCLEN*PADDED_DIM);std::vector<float> cb(Q_DOCLEN);
    for(int q=0;q<Q_DOCLEN;++q){index.rotator_->rotate(query+q*index.d,rotated.data()+q*PADDED_DIM);cb[q]=std::accumulate(rotated.data()+q*PADDED_DIM,rotated.data()+(q+1)*PADDED_DIM,0.f)*float((1<<(index.ex_bits+1))-1)/2.f;}
    std::vector<double> result(index.num_docs);const size_t stride=PADDED_DIM*(index.ex_bits+1)/8;const char*codes=index.full_bit_codes_data();
    for(size_t doc=0;doc<index.num_docs;++doc){double total=0;for(int q=0;q<Q_DOCLEN;++q){double best=-std::numeric_limits<double>::infinity();for(int token=index.doc_ptrs_[doc];token<index.doc_ptrs_[doc+1];++token){alignas(64)float decoded[PADDED_DIM];index.unpack_func_(reinterpret_cast<const uint8_t*>(codes+size_t(token)*stride),decoded,PADDED_DIM);double dot=0;for(int j=0;j<PADDED_DIM;++j)dot+=double(decoded[j])*rotated[q*PADDED_DIM+j];best=std::max(best,(dot-double(cb[q]))*index.full_bit_factors_[token]);}total+=best;}result[doc]=total;}
    return result;
 }
 static void empty_gpu_postings(chimera_index&index,bool empty){
    if(empty)check_cuda(cudaMemset(index.d_cluster_pos_,0,(index.n_clusters+1)*sizeof(size_t)));
    else check_cuda(cudaMemcpy(index.d_cluster_pos_,index.ivf->cluster_pos.data(),(index.n_clusters+1)*sizeof(size_t),cudaMemcpyHostToDevice));
 }
};
}
int cpu_invalid_build(){
    int count=0;
    for(int mode=0;mode<6;++mode){chimera_index index;std::vector<float>x(128,1.f);std::vector<int> lens{1};size_t d=128,c=1,b=3;if(mode==0)c=2;if(mode==1)x.clear();if(mode==2)lens.clear();if(mode==3)c=0;if(mode==4)d=0;if(mode==5)b=0;try{index.build(x,d,lens,c,b);}catch(const std::invalid_argument&){++count;}}
    need(count==6,"invalid build input reached CUDA or was accepted");std::cout<<"PASS CPU_INVALID_BUILD_SHAPES cases="<<count<<"\n";return 0;
}
int main(int argc,char**argv){try{
    if(argc==1||std::string(argv[1])=="--help"){std::cout<<"qa_pipeline --cpu-only | <fresh synthetic-data-dir> <fresh result-dir> [--profile] [--variant full|no-cr|no-ivf-pg|neither]\n";return 0;}
    if(std::string(argv[1])=="--cpu-only")return cpu_invalid_build();
    need(argc>=3,"invalid qualification arguments");
    bool profile=false;int selected=-1;
    for(int a=3;a<argc;++a){std::string flag=argv[a];if(flag=="--profile")profile=true;else if(flag=="--variant"&&a+1<argc){std::string value=argv[++a];if(value=="full")selected=0;else if(value=="no-cr")selected=1;else if(value=="no-ivf-pg")selected=2;else if(value=="neither")selected=3;else throw std::invalid_argument("Unknown qualification variant");}else throw std::invalid_argument("Unknown qualification flag");}need(std::getenv("UMA_RUN_GPU_TESTS")&&std::string(std::getenv("UMA_RUN_GPU_TESTS"))=="1","explicit GPU qualification required");
    fs::path data=argv[1],out=argv[2];need(!fs::exists(data)&&!fs::exists(out),"refuse existing synthetic data/output");fs::create_directories(data);fs::create_directories(out);
    const std::vector<int> lengths{1,3,7,33,63,127,255,256,257,299,300};const int docs=lengths.size(),dim=128,clusters=256,nq=32;const size_t tokens=std::accumulate(lengths.begin(),lengths.end(),size_t(0));
    std::mt19937 gen(381);std::normal_distribution<float> normal(0,1);std::vector<float>x(tokens*dim),queries(size_t(nq)*Q_DOCLEN*dim);
    auto fill=[&](std::vector<float>&a){for(size_t start=0;start<a.size();start+=dim){double norm=0;for(int j=0;j<dim;++j){a[start+j]=normal(gen);norm+=double(a[start+j])*a[start+j];}for(int j=0;j<dim;++j)a[start+j]/=float(std::sqrt(norm));}};fill(x);fill(queries);
    auto rawwrite=[](const fs::path&p,const void*v,size_t bytes){std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(v),bytes);need(bool(f),"could not write synthetic data");};
    rawwrite(data/"embeddings.f32",x.data(),x.size()*sizeof(float));rawwrite(data/"queries.f32",queries.data(),queries.size()*sizeof(float));rawwrite(data/"doclens.i32",lengths.data(),lengths.size()*sizeof(int));
    SearchOptions build_options;build_options.nprobe=clusters;build_options.k_refine=docs;build_options.k_full_bit=docs;build_options.num_chunks=10;build_options.cagra_itopk_size=clusters;
    {chimera_index index;index.build(x,dim,lengths,clusters,3,build_options);index.save((data/"index").string());}
    std::vector<std::vector<size_t>> full_ids(nq);std::vector<std::vector<float>> full_scores(nq);std::ofstream rows(out/"rows.jsonl");int query_id=0,zero_cases=0;double max_error=0;
    {auto opt=build_options;opt.num_chunks=1;chimera_index reference;reference.load((data/"index").string(),opt);reference.set_capture_scores(true);for(int q=0;q<nq;++q){full_ids[q]=reference.search(queries.data()+size_t(q)*Q_DOCLEN*dim,Q_DOCLEN*dim,docs);full_scores[q]=reference.last_result_scores();}}
    if(profile)check_cuda(cudaProfilerStart());
    for(auto mode:{AlgorithmVariant::Full,AlgorithmVariant::NoCR,AlgorithmVariant::NoIVFPG,AlgorithmVariant::Neither})for(int chunks:{1,2,4,6,8,10})for(int budget:{docs,3}){
        if(selected>=0&&selected!=int(mode))continue;
        SearchOptions opt=build_options;opt.algorithm_variant=mode;opt.expected_coarse_clusters=uses_flat_centroid_route(mode)?clusters:0;opt.num_chunks=chunks;opt.k_full_bit=budget;if(skips_candidate_refinement(mode))opt.k_refine=1;
        chimera_index index;index.load((data/"index").string(),opt);index.set_capture_scores(true);index.set_profiling_enabled(profile);
        for(int q=0;q<nq;++q){const float*query=queries.data()+size_t(q)*Q_DOCLEN*dim;auto expected=AlgorithmAblationQualification::quantized_maxsim(index,query);
            ProfileRange query_range(profile,"Chimera/query_local_row",query_id);
            auto ids=index.search(query,Q_DOCLEN*dim,budget);auto scores=index.last_result_scores();need(ids.size()==size_t(budget)&&scores.size()==ids.size(),"wrong pipeline completed result count");std::set<size_t> seen;
            for(size_t i=0;i<ids.size();++i){need(ids[i]<size_t(docs)&&seen.insert(ids[i]).second,"invalid pipeline document IDs");need(std::isfinite(scores[i]),"nonfinite pipeline score");if(i)need(scores[i]<=scores[i-1],"pipeline scores not descending");double error=std::abs(double(scores[i])-expected[ids[i]]);max_error=std::max(max_error,error);need(error<=2e-4,"CPU quantized FP64 MaxSim score error exceeds tolerance");}
            if(budget==docs){need(seen.size()==size_t(docs),"all-document fullbit budget did not score all docs");for(int i=1;i<docs;++i)need(expected[ids[i-1]]+4e-4>=expected[ids[i]],"all-document rank differs from high precision quantized MaxSim");need(ids==full_ids[q]&&scores==full_scores[q],"all-document results differ across chunk/mode");}
            rows<<"{\"query_row\":"<<query_id<<",\"original_query\":"<<q<<",\"mode\":"<<int(mode)<<",\"k_refine\":"<<opt.k_refine<<",\"requested_chunks\":"<<chunks<<",\"k_full\":"<<budget<<",\"returned\":"<<ids.size()<<",\"fullbit_count_source\":\"NVTX cpu_fullbit INT64; profile required for actual Khat\"}\n";++query_id;
        }
        AlgorithmAblationQualification::empty_gpu_postings(index,true);auto empty=index.search(queries.data(),Q_DOCLEN*dim,docs);need(empty.empty()&&index.last_result_scores().empty(),"empty CG retained stale results/scores");AlgorithmAblationQualification::empty_gpu_postings(index,false);check_cuda(cudaDeviceSynchronize());++zero_cases;
    }
    if(profile)check_cuda(cudaProfilerStop());
    bool mismatch=false;try{auto opt=build_options;opt.algorithm_variant=AlgorithmVariant::NoIVFPG;opt.expected_coarse_clusters=257;chimera_index rejected;rejected.load((data/"index").string(),opt);}catch(const std::invalid_argument&){mismatch=true;}need(mismatch,"coarse centroid count mismatch accepted");
    std::ofstream result(out/"result.json");result<<"{\"status\":\"PASS_SYNTHETIC_ALGORITHM_PIPELINE\",\"documents\":11,\"tokens\":"<<tokens<<",\"queries_per_condition\":32,\"conditions\":"<<(selected<0?48:12)<<",\"completed_searches\":"<<query_id<<",\"zero_CG_cases\":"<<zero_cases<<",\"max_quantized_MaxSim_FP64_error\":"<<max_error<<",\"all_document_ID_score_rows_exact_across_modes_chunks\":true,\"coarse_cluster_mismatch_rejected\":true,\"profiled\":"<<(profile?"true":"false")<<",\"scope\":\"synthetic stress; full Fig8 uses independently rebuilt actual coarse index, this same tiny index is qualification only\"}\n";
    std::cout<<"PASS synthetic algorithm pipeline searches="<<query_id<<" zeroCG="<<zero_cases<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
