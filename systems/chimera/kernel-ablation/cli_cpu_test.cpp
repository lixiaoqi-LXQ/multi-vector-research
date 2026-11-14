#include "utils.hpp"
#include <cassert>
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <fstream>
Chimera::SearchCliArgs parse(const std::string& k,const std::string& algorithm="full") {
 std::vector<std::string>s={"test","--index","kernel-ablation/cli-fixture","--query","dummy","--gt","dummy","--kernel-mode",k,"--algorithm-variant",algorithm};std::vector<char*>p;for(auto&x:s)p.push_back(x.data());return Chimera::parse_search_args(p.size(),p.data());
}
int main(){std::filesystem::create_directories("kernel-ablation/cli-fixture");std::ofstream("kernel-ablation/cli-fixture/doclens.bin").put(0);int passed=0;for(auto k:{"full","no-lut","no-aggregation","neither"}){auto x=parse(k);assert(x.options.use_lut==(std::string(k)=="full"||std::string(k)=="no-aggregation"));assert(x.options.use_aggregation==(std::string(k)=="full"||std::string(k)=="no-lut"));++passed;}for(auto x:{std::pair<std::string,std::string>{"invalid","full"},{"full","no-cr"},{"neither","neither"}}){bool rejected=false;try{parse(x.first,x.second);}catch(...){rejected=true;}assert(rejected);++passed;}std::cout<<"PASS "<<passed<<" actual CLI mode/unsupported-family checks; CPU-only\n";}
