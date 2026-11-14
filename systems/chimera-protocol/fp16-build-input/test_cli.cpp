#include "utils.hpp"
#include <cassert>
#include <vector>
#include <string>
using namespace Chimera;
BuildCliArgs parse(std::vector<std::string> words){
 std::vector<char*> argv;for(auto& word:words) argv.push_back(word.data());
 return parse_build_args(static_cast<int>(argv.size()),argv.data());
}
int main(){
 auto base=std::vector<std::string>{"gpu_build","--index","index","--doclens","doclens","--n-clusters","32","--ex-bits","7"};
 auto half=base;half.insert(half.end(),{"--fp16-npy","documents.npy"});auto a=parse(half);assert(a.fp16_npy_file=="documents.npy"&&a.data_file.empty());
 auto full=base;full.insert(full.end(),{"--data","embeddings.bin"});auto b=parse(full);assert(b.data_file=="embeddings.bin"&&b.fp16_npy_file.empty());
 auto both=half;both.insert(both.end(),{"--data","embeddings.bin"});bool fail=false;try{parse(both);}catch(const std::exception&){fail=true;}assert(fail);
 fail=false;try{parse(base);}catch(const std::exception&){fail=true;}assert(fail);
}
