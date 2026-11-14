#include "chimera/build_checkpoint.hpp"
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>
namespace fs=std::filesystem;
static bool record_sync=false;
struct SyncEvent { std::string path; bool directory; };
static std::vector<SyncEvent> sync_events;
extern "C" int __real_fsync(int);
extern "C" int __wrap_fsync(int fd) {
 if(record_sync){char link[128],target[4096];snprintf(link,sizeof(link),"/proc/self/fd/%d",fd);auto n=readlink(link,target,sizeof(target)-1);struct stat st{};if(n>=0&&fstat(fd,&st)==0){target[n]=0;sync_events.push_back({target,bool(S_ISDIR(st.st_mode))});}}
 return __real_fsync(fd);
}
using namespace Chimera;
void require(bool x,const char* m) { if(!x)throw std::runtime_error(m); }
std::string text(const fs::path& p) {std::ifstream f(p);return {std::istreambuf_iterator<char>(f),{}};}
void put(const fs::path& p,const std::string& s) {std::ofstream f(p,std::ios::binary);f<<s;}
int main(int argc,char**argv) {
 require(argc==2,"fixture directory required");fs::path root=argv[1];require(!fs::exists(root),"refuse fixture overwrite");fs::create_directories(root);
 AssignmentCheckpointConfig cfg;cfg.identity_sha256=std::string(64,'a');cfg.doclens_sha256=std::string(64,'b');cfg.tokens=65539;cfg.dimension=64;cfg.clusters=2;cfg.ex_bits=3;
 std::vector<float> centroids(128,.125f);std::vector<uint32_t> labels(cfg.tokens);for(size_t i=0;i<labels.size();++i)labels[i]=i%2;
 int tests=0;
 record_sync=true;
 {AssignmentCheckpoint c((root/"parent-durability").string(),cfg,false);}
 record_sync=false;
 require(!sync_events.empty()&&sync_events.front().directory&&sync_events.front().path==root.string(),"cache parent entry was not synced before initial state");
 bool own_dir=false;for(const auto& e:sync_events)own_dir|=e.directory&&e.path==(root/"parent-durability").string();require(own_dir,"cache state directory sync missing");
 ++tests;std::cout<<"PASS actual cache parent-directory fsync before initial state commit\n";
 auto expect_reject=[&](const std::function<void()>& f,const char* name){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}require(rejected,name);++tests;std::cout<<"PASS reject "<<name<<'\n';};
 auto make_graph=[&](AssignmentCheckpoint& c){c.save_centroids(centroids);auto t=c.reserve_graph_temporary();put(t,"CPU fixture fake graph bytes, not cuVS validation");c.commit_graph(t);};
 auto cache=root/"good";
 {AssignmentCheckpoint c(cache.string(),cfg,false);make_graph(c);c.append_labels(0,labels.data(),65536);require(c.completed_tokens()==65536,"prefix commit");expect_reject([&]{AssignmentCheckpoint second(cache.string(),cfg,true);},"concurrent cache lock");}
 {AssignmentCheckpoint c(cache.string(),cfg,true);require(c.has_graph()&&c.has_centroids(),"resume artifacts");require(c.load_centroids()==centroids,"centroid bytes");std::vector<uint32_t> loaded(cfg.tokens);c.load_labels(loaded);require(std::equal(loaded.begin(),loaded.begin()+65536,labels.begin()),"prefix bytes");c.append_labels(65536,labels.data()+65536,3);require(c.completed_tokens()==cfg.tokens,"full commit");}
 {AssignmentCheckpoint c(cache.string(),cfg,true);std::vector<uint32_t> loaded(cfg.tokens);c.load_labels(loaded);require(loaded==labels,"complete labels bytes");}++tests;std::cout<<"PASS stop/resume full digest and exact labels\n";
 expect_reject([&]{AssignmentCheckpoint c(cache.string(),cfg,false);},"existing cache without resume");
 auto missing=root/"missing";expect_reject([&]{AssignmentCheckpoint c(missing.string(),cfg,true);},"missing cache resume");require(!fs::exists(missing),"missing resume created directory");
 for(int kind=0;kind<4;++kind){auto bad=cfg;if(kind==0)bad.identity_sha256[0]='c';if(kind==1)bad.doclens_sha256[0]='c';if(kind==2)bad.ex_bits=7;if(kind==3)bad.tokens++;expect_reject([&]{AssignmentCheckpoint c(cache.string(),bad,true);},"changed identity/parameters");}
 auto copy=[&](const char* name){auto p=root/name;fs::copy(cache,p,fs::copy_options::recursive);return p;};
 for(auto name:{"centroids.f32","assignment.cagra","labels/0.u32"}) {auto p=copy((std::string("corrupt-")+std::to_string(tests)).c_str());auto f=p/name;std::fstream io(f,std::ios::binary|std::ios::in|std::ios::out);char x;io.read(&x,1);x^=1;io.seekp(0);io.write(&x,1);io.close();expect_reject([&]{AssignmentCheckpoint c(p.string(),cfg,true);},name);}
 {auto p=copy("truncated-labels");fs::resize_file(p/"labels/0.u32",4);expect_reject([&]{AssignmentCheckpoint c(p.string(),cfg,true);},"truncated label bytes");}
 {auto p=copy("trailing-state");std::ofstream(p/"state.txt",std::ios::app)<<"unexpected\n";expect_reject([&]{AssignmentCheckpoint c(p.string(),cfg,true);},"trailing metadata");}
 {auto p=copy("full-sha-mutation");auto st=text(p/"state.txt");auto pos=st.rfind("65539 ");require(pos!=std::string::npos,"locate done field");pos+=6;st.replace(pos,64,std::string(64,'0'));put(p/"state.txt",st);expect_reject([&]{AssignmentCheckpoint c(p.string(),cfg,true);},"full digest mutation");}
 {auto p=root/"range";AssignmentCheckpoint c(p.string(),cfg,false);make_graph(c);auto bad=labels;bad[0]=2;expect_reject([&]{c.append_labels(0,bad.data(),65536);},"out of range label");require(!fs::exists(p/"labels/0.u32"),"bad labels published");expect_reject([&]{c.append_labels(1,labels.data(),65536);},"noncontiguous start");expect_reject([&]{c.append_labels(0,labels.data(),1);},"wrong batch size");}
 {auto p=root/"orphan";{AssignmentCheckpoint c(p.string(),cfg,false);put(p/"centroids.f32","uncommitted centroid evidence");make_graph(c);put(p/"labels/0.u32","uncommitted labels evidence");}AssignmentCheckpoint c(p.string(),cfg,true);c.append_labels(0,labels.data(),65536);bool found=false;for(auto& e:fs::directory_iterator(p/"labels"))if(e.path().filename().string().find("orphan-")==0)found=text(e.path())=="uncommitted labels evidence";require(found,"orphan bytes not retained");require(c.completed_tokens()==65536,"new orphan prefix");}++tests;std::cout<<"PASS orphan retention/requery\n";
 std::cout<<"CPU_CHECKPOINT_TESTS_PASS "<<tests<<'\n';
}
