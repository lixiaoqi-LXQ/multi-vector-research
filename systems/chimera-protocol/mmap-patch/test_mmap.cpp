#include "cpp/apps/mapped_embeddings.hpp"
#include <cassert>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string root(argv[1]);
    const std::string valid = root + "/valid.bin";
    { std::ofstream f(valid, std::ios::binary); int32_t shape[2]={2,3}; float values[6]={1,2,3,4,5,6}; f.write(reinterpret_cast<char*>(shape),8); f.write(reinterpret_cast<char*>(values),24); }
    { MappedEmbeddings data(valid); assert(data.size()==6 && data.dimension()==3); for(int i=0;i<6;++i) assert(data.data()[i]==float(i+1)); }
    bool caught=false;
    try { MappedEmbeddings missing(root+"/absent.bin"); } catch(const std::runtime_error&) {caught=true;} assert(caught);
    { std::ofstream f(root+"/short.bin",std::ios::binary); int32_t shape[2]={2,3}; f.write(reinterpret_cast<char*>(shape),8); }
    caught=false;try { MappedEmbeddings truncated(root+"/short.bin"); }catch(const std::runtime_error&){caught=true;}assert(caught);
    { std::ofstream f(root+"/negative.bin",std::ios::binary); int32_t shape[2]={-1,128};f.write(reinterpret_cast<char*>(shape),8); }
    caught=false;try { MappedEmbeddings negative(root+"/negative.bin"); }catch(const std::runtime_error&){caught=true;}assert(caught);
    { std::ofstream f(root+"/trailing.bin",std::ios::binary); int32_t shape[2]={1,1};float values[2]={1,2};f.write(reinterpret_cast<char*>(shape),8);f.write(reinterpret_cast<char*>(values),8); }
    caught=false;try { MappedEmbeddings trailing(root+"/trailing.bin"); }catch(const std::runtime_error&){caught=true;}assert(caught);
}
