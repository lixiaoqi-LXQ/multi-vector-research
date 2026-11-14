#include "chimera/flat_l2_router.cuh"
#include "chimera/chimera_index.cuh"
#include <raft/core/resources.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <vector>
using namespace Chimera;
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char**argv) {
 try {
    if(argc!=2||std::string(argv[1])=="--help"){std::cout<<"GPU qualification: qa_flat <fresh-result.json> (UMA_RUN_GPU_TESTS=1 required)\n";return argc==1||std::string(argv[1])=="--help"?0:2;}
    need(std::getenv("UMA_RUN_GPU_TESTS") && std::string(std::getenv("UMA_RUN_GPU_TESTS"))=="1","explicit GPU qualification required");
    need(!std::filesystem::exists(argv[1]),"refuse existing flat QA result");
    std::mt19937 rng(127);std::uniform_real_distribution<float> gen(-3.f,3.f);int cases=0,queries_done=0;double max_error=0;
    raft::resources resources;cudaStream_t stream;check_cuda(cudaStreamCreate(&stream));
    for(int rows:{1,2,3,33,129,257})for(int nq:{1,2,32}) {
        const int d=128,stride=131;std::vector<float> x(size_t(rows)*stride,std::numeric_limits<float>::quiet_NaN()),q(nq*d);
        for(int i=0;i<rows;++i)for(int j=0;j<d;++j)x[size_t(i)*stride+j]=gen(rng);
        if(rows>1)for(int j=0;j<d;++j)x[stride+j]=x[j]; // exact duplicate/tie
        for(float& value:q)value=gen(rng);
        for(int j=0;j<d;++j)q[j]=x[j];
        float *dx=nullptr,*dq=nullptr,*ds=nullptr;uint32_t*di=nullptr;
        check_cuda(cudaMalloc(&dx,x.size()*sizeof(float)));check_cuda(cudaMalloc(&dq,q.size()*sizeof(float)));
        check_cuda(cudaMemcpy(dx,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));check_cuda(cudaMemcpy(dq,q.data(),q.size()*sizeof(float),cudaMemcpyHostToDevice));
        check_cuda(cudaMalloc(&ds,size_t(rows)*nq*sizeof(float)));check_cuda(cudaMalloc(&di,size_t(rows)*nq*sizeof(uint32_t)));
        {
            FlatL2Router router(dx,rows,d,stride,nq);
            for(int k:{1,std::min(7,rows),rows}) {
                router.search(resources,dq,nq,k,ds,di,stream);check_cuda(cudaStreamSynchronize(stream));
                std::vector<float> scores(size_t(nq)*k);std::vector<uint32_t> ids(size_t(nq)*k);
                check_cuda(cudaMemcpy(scores.data(),ds,scores.size()*sizeof(float),cudaMemcpyDeviceToHost));check_cuda(cudaMemcpy(ids.data(),di,ids.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
                for(int qi=0;qi<nq;++qi){
                    std::vector<double> expected(rows),tol(rows);
                    for(int c=0;c<rows;++c){double dist=0,cn=0,qn=0,dot=0;for(int j=0;j<d;++j){double a=x[size_t(c)*stride+j],b=q[qi*d+j];dist+=(a-b)*(a-b);cn+=a*a;qn+=b*b;dot+=a*b;}expected[c]=dist;tol[c]=64*std::numeric_limits<float>::epsilon()*(cn+qn+2*std::abs(dot)+1);}
                    auto sorted=expected;std::sort(sorted.begin(),sorted.end());double cutoff=sorted[k-1];std::set<uint32_t> selected;
                    for(int rank=0;rank<k;++rank){size_t pos=size_t(qi)*k+rank;need(ids[pos]<uint32_t(rows)&&selected.insert(ids[pos]).second,"duplicate/out-of-range flat ID");need(std::isfinite(scores[pos])&&scores[pos]>=0,"nonfinite/negative flat distance");if(rank)need(scores[pos]>=scores[pos-1],"flat output not ascending");double err=std::abs(scores[pos]-expected[ids[pos]]);max_error=std::max(max_error,err);need(err<=tol[ids[pos]],"flat FP32 distance exceeds FP64-derived tolerance");need(expected[ids[pos]]<=cutoff+2*tol[ids[pos]],"flat selected a strictly farther centroid");}
                    for(int c=0;c<rows;++c)if(expected[c]<cutoff-2*tol[c])need(selected.count(c),"flat omitted a strictly nearer centroid");
                    ++queries_done;
                }
                ++cases;
            }
            bool rejected=false;try{router.search(resources,dq,nq,size_t(rows)+1,ds,di,stream);}catch(const std::invalid_argument&){rejected=true;}need(rejected,"flat k>rows accepted");
        }
        check_cuda(cudaFree(dx));check_cuda(cudaFree(dq));check_cuda(cudaFree(ds));check_cuda(cudaFree(di));
    }
    check_cuda(cudaStreamDestroy(stream));std::ofstream out(argv[1],std::ios::out);need(bool(out),"could not write QA result");
    out<<"{\"status\":\"PASS_GPU_EXACT_FLAT\",\"cases\":"<<cases<<",\"query_cases\":"<<queries_done<<",\"max_distance_error\":"<<max_error<<",\"TF32\":false,\"scope\":\"all centroid FP32_PEDANTIC expanded L2 plus exact selectK; FP64/tie-aware oracle, logicalD128 stride131\"}\n";
    std::cout<<"PASS GPU exact flat cases="<<cases<<" querycases="<<queries_done<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
