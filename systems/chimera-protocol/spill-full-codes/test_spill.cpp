#include "chimera/spilled_full_codes.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
using Chimera::SpilledFullCodes;
namespace fs=std::filesystem;
template<class F> void rejects(F f) { bool caught=false;try{f();}catch(const std::exception&){caught=true;}assert(caught); }
int main(int argc,char**argv) {
 assert(argc==2); fs::path root=argv[1];fs::create_directories(root);
 const auto temp=root/"owned.tmp", dest=root/"index", saved=dest/"doc_full.bin";
 std::vector<char> reference(131200);for(size_t i=0;i<reference.size();++i)reference[i]=char((i*17)%251);
 {
  SpilledFullCodes spill(temp.string(),reference.size(),65536);
  assert(spill.rows_per_flush(128)==512);
  for(size_t i=0;i<reference.size();++i)assert(spill.data()[i]==0);
  rejects([&]{SpilledFullCodes duplicate(temp.string(),1);});
  rejects([&]{spill.preflight_save_directory(root.string());});
  spill.preflight_save_directory(dest.string());
  for(size_t first=0;first<reference.size();first+=65536){size_t n=std::min(size_t(65536),reference.size()-first);std::memcpy(spill.data()+first,reference.data()+first,n);spill.sync_and_drop(first,n);}
  assert(std::memcmp(spill.data(),reference.data(),reference.size())==0);
  fs::create_directories(dest);spill.publish_hardlink(saved.string());
  struct stat a{},b{};assert(stat(temp.c_str(),&a)==0&&stat(saved.c_str(),&b)==0);assert(a.st_ino==b.st_ino&&a.st_dev==b.st_dev);
  rejects([&]{spill.publish_hardlink(saved.string());});
 }
 assert(!fs::exists(temp)&&fs::file_size(saved)==reference.size());
 std::ifstream input(saved,std::ios::binary);std::vector<char> read(reference.size());input.read(read.data(),read.size());assert(read==reference);
 {
  SpilledFullCodes spill(temp.string(),4096);
  fs::rename(temp,root/"renamed.tmp");std::ofstream(temp)<<"replacement";
 }
 assert(fs::exists(temp)&&fs::file_size(temp)==11);
 rejects([&]{SpilledFullCodes invalid((root/"zero").string(),0);});
 std::cout<<"PASS: exact bytes, bounded flush, same-inode save, ownership-safe cleanup, preflight and error guards\n";
}
