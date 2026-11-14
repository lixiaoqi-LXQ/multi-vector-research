#include "utils.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
using namespace Chimera;
int main(int argc,char**argv) {
    assert(argc==2); const std::string ix=argv[1];std::filesystem::create_directories(ix);std::ofstream(ix+"/doclens.bin").put(0);
    auto parse=[&](std::vector<std::string> extra){std::vector<std::string>a={"test","--index",ix,"--query","dummy","--gt","dummy"};a.insert(a.end(),extra.begin(),extra.end());std::vector<char*> v;for(auto& s:a)v.push_back(s.data());return parse_search_args(v.size(),v.data());};
    assert(parse({}).options.algorithm_variant==AlgorithmVariant::Full);
    assert(parse({"--kernel-mode","full"}).options.algorithm_variant==AlgorithmVariant::Full);
    assert(parse({"--algorithm-variant","no-cr","--k-refine","1","--k-full-bit","400"}).options.algorithm_variant==AlgorithmVariant::NoCR);
    for(auto v:{"no-ivf-pg","neither"}) assert(uses_flat_centroid_route(parse({"--algorithm-variant",v,"--ablation-coarse-clusters","131072"}).options.algorithm_variant));
    int rejected=0;
    for(auto args:std::vector<std::vector<std::string>>{
        {"--kernel-mode","no-lut"},{"--algorithm-variant","pretend"},{"--algorithm-variant","no-ivf-pg"},
        {"--algorithm-variant","neither","--ablation-coarse-clusters","0"},
        {"--algorithm-variant","neither","--ablation-coarse-clusters","-1"},
        {"--algorithm-variant","full","--ablation-coarse-clusters","131072"},
        {"--algorithm-variant","no-cr","--ablation-coarse-clusters","131072"},
        {"--algorithm-variant","full","--k-refine","1","--k-full-bit","400"},
        {"--algorithm-variant","no-cr","--num-chunks","0"}}) {
        try{parse(args);}catch(const std::exception&){++rejected;}
    }
    assert(rejected==9);std::cout<<"PASS cli_valid=5 cli_invalid="<<rejected<<"\n";
}
