#include "chimera/build_checkpoint.hpp"
#include <openssl/evp.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace Chimera {
namespace {
namespace fs = std::filesystem;
void need(bool value, const std::string& message) {
    if (!value) throw std::runtime_error("Build checkpoint: " + message);
}
struct Digest {
    EVP_MD_CTX* p = EVP_MD_CTX_new();
    Digest() { need(p && EVP_DigestInit_ex(p, EVP_sha256(), nullptr) == 1, "SHA init"); }
    ~Digest() { EVP_MD_CTX_free(p); }
    void update(const void* data, size_t count) { need(EVP_DigestUpdate(p, data, count) == 1, "SHA update"); }
    std::string finish() {
        unsigned char bytes[EVP_MAX_MD_SIZE]; unsigned count = 0;
        need(EVP_DigestFinal_ex(p, bytes, &count) == 1 && count == 32, "SHA final");
        std::ostringstream out;
        for (unsigned i = 0; i < count; ++i) out << std::hex << std::setfill('0') << std::setw(2) << unsigned(bytes[i]);
        return out.str();
    }
};
bool digest_string(const std::string& value) {
    return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == std::string::npos;
}
void sync_directory(const fs::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
    need(fd >= 0, "open directory for sync");
    int result = ::fsync(fd); ::close(fd); need(result == 0, "directory sync");
}
fs::path temporary(const fs::path& dir, const std::string& name) {
    std::string value = (dir / (".tmp-" + name + "-XXXXXX")).string();
    std::vector<char> chars(value.begin(), value.end()); chars.push_back(0);
    int fd = ::mkstemp(chars.data()); need(fd >= 0, "create temporary"); ::close(fd);
    return chars.data();
}
void sync_file(const fs::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY); need(fd >= 0, "open artifact for sync");
    int result = ::fsync(fd); ::close(fd); need(result == 0, "artifact sync");
}
void publish_exclusive(const fs::path& temp, const fs::path& destination) {
    sync_file(temp);
    need(::link(temp.c_str(), destination.c_str()) == 0, "refusing existing artifact " + destination.string());
    sync_directory(destination.parent_path());
    need(::unlink(temp.c_str()) == 0, "release own temporary link");
}
void retain_orphan(const fs::path& path) {
    if (!fs::exists(path)) return;
    need(fs::is_regular_file(fs::symlink_status(path)), "uncommitted artifact is not regular");
    for (size_t i = 0;; ++i) {
        auto orphan = path.parent_path() / ("orphan-" + path.filename().string() + "-" + std::to_string(::getpid()) + "-" + std::to_string(i));
        if (::link(path.c_str(), orphan.c_str()) == 0) {
            sync_directory(path.parent_path()); need(::unlink(path.c_str()) == 0, "retain orphan"); return;
        }
        need(errno == EEXIST, "orphan publication failed");
    }
}
void write_bytes(const fs::path& path, const void* bytes, size_t size) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    need(bool(out), "open artifact output"); out.write(static_cast<const char*>(bytes), size);
    out.flush(); need(bool(out), "write artifact output"); out.close(); need(bool(out), "close artifact output");
}
void read_bytes(const fs::path& path, void* bytes, size_t size) {
    need(fs::is_regular_file(fs::symlink_status(path)) && fs::file_size(path) == size, "artifact size/type mismatch " + path.string());
    std::ifstream in(path, std::ios::binary); need(bool(in), "open artifact input");
    in.read(static_cast<char*>(bytes), size); need(size == 0 || size_t(in.gcount()) == size, "short artifact input");
}
std::string concat_sha(const fs::path& root, const std::vector<std::pair<size_t,size_t>>& chunks) {
    Digest hash; std::array<char,1 << 20> buffer;
    for (auto [start,count] : chunks) {
        auto path = root / "labels" / (std::to_string(start) + ".u32");
        std::ifstream in(path, std::ios::binary); need(bool(in), "open labels for full SHA");
        while (in) { in.read(buffer.data(), buffer.size()); auto n=in.gcount(); if(n>0) hash.update(buffer.data(), n); }
        need(in.eof(), "read labels full SHA");
    }
    return hash.finish();
}
}
std::string checkpoint_sha256(const void* bytes, size_t count) { Digest hash; hash.update(bytes,count); return hash.finish(); }
std::string checkpoint_file_sha256(const fs::path& path) {
    need(fs::is_regular_file(fs::symlink_status(path)), "artifact must be regular file");
    Digest hash; std::array<char,1 << 20> buffer; std::ifstream in(path, std::ios::binary); need(bool(in), "open SHA input");
    while (in) { in.read(buffer.data(), buffer.size()); auto n=in.gcount(); if(n>0) hash.update(buffer.data(), n); }
    need(in.eof(), "read SHA input"); return hash.finish();
}
AssignmentCheckpoint::AssignmentCheckpoint(const std::string& directory, const AssignmentCheckpointConfig& config, bool resume)
  : root_(fs::absolute(directory)), config_(config) {
    need(digest_string(config.identity_sha256) && digest_string(config.doclens_sha256), "explicit identity/doclens SHA required");
    need(config.tokens>0 && config.dimension>0 && config.clusters>0 && config.clusters<=config.tokens && config.ex_bits>=1 && config.ex_bits<=7 && config.clusters<=std::numeric_limits<uint32_t>::max() && config.batch_size==65536 && config.assignment_k==1 && config.assignment_itopk==150, "invalid assignment configuration");
    need(config.dimension<=std::numeric_limits<size_t>::max()/sizeof(float)/config.clusters, "centroid size overflow");
    need(!resume || fs::is_directory(root_), "resume cache does not exist");
    bool created=resume ? false : fs::create_directory(root_);
    need(!fs::is_symlink(fs::symlink_status(root_)) && fs::is_directory(root_), "checkpoint directory required");
    need(resume || created, "existing cache requires explicit resume");
    try {
        lock_fd_=::open((root_/".lock").c_str(), O_RDWR|O_CREAT|O_NOFOLLOW,0600);
        need(lock_fd_>=0 && ::flock(lock_fd_,LOCK_EX|LOCK_NB)==0, "cache already locked");
        if(resume) { need(!created && fs::exists(root_/"state.txt"), "resume requires committed state"); read_state(); validate(); }
        else { fs::create_directory(root_/"labels"); write_state(); }
    } catch(...) { if(lock_fd_>=0) ::close(lock_fd_); lock_fd_=-1; throw; }
}
AssignmentCheckpoint::~AssignmentCheckpoint() { if(lock_fd_>=0) ::close(lock_fd_); }
bool AssignmentCheckpoint::has_centroids() const { return centroids_.sha!="-"; }
bool AssignmentCheckpoint::has_graph() const { return graph_.sha!="-"; }
fs::path AssignmentCheckpoint::chunk_path(size_t start) const { return root_/"labels"/(std::to_string(start)+".u32"); }
void AssignmentCheckpoint::write_state() {
    auto temp=temporary(root_,"state"); std::ofstream out(temp);
    out << "CHIMERA_ASSIGNMENT_CHECKPOINT_V1\n" << config_.identity_sha256 << '\n' << config_.doclens_sha256 << '\n'
        << config_.tokens << ' ' << config_.dimension << ' ' << config_.clusters << ' ' << config_.ex_bits << ' '
        << config_.batch_size << ' ' << config_.sample_seed << ' ' << config_.rotator_seed << ' '
        << config_.assignment_k << ' ' << config_.assignment_itopk << '\n'
        << centroids_.bytes << ' ' << centroids_.sha << '\n' << graph_.bytes << ' ' << graph_.sha << '\n'
        << done_ << ' ' << labels_sha_ << ' ' << chunks_.size() << '\n';
    for(const auto& c:chunks_) out<<c.start<<' '<<c.count<<' '<<c.sha<<'\n';
    out.flush(); need(bool(out),"state write"); out.close(); sync_file(temp);
    need(::rename(temp.c_str(),(root_/"state.txt").c_str())==0,"atomic state publication"); sync_directory(root_);
}
void AssignmentCheckpoint::read_state() {
    auto state=root_/"state.txt"; need(fs::is_regular_file(fs::symlink_status(state)) && fs::file_size(state)<16*1024*1024,"state type/size invalid");
    std::ifstream in(state); std::string magic, identity, doclens; AssignmentCheckpointConfig actual; size_t count=0;
    in>>magic>>identity>>doclens>>actual.tokens>>actual.dimension>>actual.clusters>>actual.ex_bits>>actual.batch_size>>actual.sample_seed>>actual.rotator_seed>>actual.assignment_k>>actual.assignment_itopk;
    need(bool(in) && magic=="CHIMERA_ASSIGNMENT_CHECKPOINT_V1" && identity==config_.identity_sha256 && doclens==config_.doclens_sha256,"state identity/schema mismatch");
    need(actual.tokens==config_.tokens && actual.dimension==config_.dimension && actual.clusters==config_.clusters && actual.ex_bits==config_.ex_bits && actual.batch_size==config_.batch_size && actual.sample_seed==config_.sample_seed && actual.rotator_seed==config_.rotator_seed && actual.assignment_k==config_.assignment_k && actual.assignment_itopk==config_.assignment_itopk,"state parameter mismatch");
    in>>centroids_.bytes>>centroids_.sha>>graph_.bytes>>graph_.sha>>done_>>labels_sha_>>count;
    need(bool(in) && count<=1+(config_.tokens-1)/config_.batch_size,"state prefix count invalid");
    for(size_t i=0;i<count;i++) { Chunk c; in>>c.start>>c.count>>c.sha; need(bool(in),"truncated state"); chunks_.push_back(c); }
    std::string extra; need(!(in>>extra),"unexpected state fields");
}
void AssignmentCheckpoint::validate() const {
    need(done_<=config_.tokens && (has_centroids() || (!has_graph() && done_==0 && chunks_.empty())),"invalid checkpoint stage");
    if(has_centroids()) need(digest_string(centroids_.sha) && centroids_.bytes==config_.clusters*config_.dimension*sizeof(float) && fs::file_size(root_/"centroids.f32")==centroids_.bytes && checkpoint_file_sha256(root_/"centroids.f32")==centroids_.sha,"centroid checksum mismatch");
    else need(centroids_.bytes==0,"unexpected centroid bytes");
    if(has_graph()) need(digest_string(graph_.sha) && graph_.bytes>0 && fs::file_size(root_/"assignment.cagra")==graph_.bytes && checkpoint_file_sha256(root_/"assignment.cagra")==graph_.sha,"graph checksum mismatch");
    else need(graph_.bytes==0 && done_==0 && chunks_.empty(),"labels without committed graph");
    size_t cursor=0; std::vector<std::pair<size_t,size_t>> paths;
    for(const auto& c:chunks_) {
        need(c.start==cursor && c.count==std::min(config_.batch_size,config_.tokens-cursor) && digest_string(c.sha),"noncontiguous labels");
        std::vector<uint32_t> labels(c.count);read_bytes(chunk_path(c.start),labels.data(),labels.size()*sizeof(uint32_t));
        need(checkpoint_sha256(labels.data(),labels.size()*sizeof(uint32_t))==c.sha,"label checksum mismatch");
        for(auto label:labels) need(label<config_.clusters,"label outside centroid range");
        cursor+=c.count;paths.emplace_back(c.start,c.count);
    }
    need(cursor==done_,"prefix completion mismatch");
    if(done_==config_.tokens) need(digest_string(labels_sha_) && concat_sha(root_,paths)==labels_sha_,"full assignment checksum mismatch");
    else need(labels_sha_=="-","premature full assignment digest");
}
std::vector<float> AssignmentCheckpoint::load_centroids() const {
    need(has_centroids(),"centroids not committed"); std::vector<float> out(config_.clusters*config_.dimension);
    read_bytes(root_/"centroids.f32",out.data(),out.size()*sizeof(float));
    need(checkpoint_sha256(out.data(),out.size()*sizeof(float))==centroids_.sha,"centroids changed during load");return out;
}
void AssignmentCheckpoint::save_centroids(const std::vector<float>& values) {
    need(!has_centroids() && !has_graph() && done_==0 && values.size()==config_.clusters*config_.dimension,"invalid centroid publication");
    retain_orphan(root_/"centroids.f32");auto temp=temporary(root_,"centroids");write_bytes(temp,values.data(),values.size()*sizeof(float));publish_exclusive(temp,root_/"centroids.f32");
    centroids_={values.size()*sizeof(float),checkpoint_sha256(values.data(),values.size()*sizeof(float))};write_state();
}
fs::path AssignmentCheckpoint::reserve_graph_temporary() { need(has_centroids() && !has_graph(),"graph stage invalid");return temporary(root_,"graph"); }
void AssignmentCheckpoint::commit_graph(const fs::path& temp) {
    need(has_centroids() && !has_graph() && temp.parent_path()==root_,"graph temporary outside cache");
    Artifact artifact{fs::file_size(temp),checkpoint_file_sha256(temp)};need(artifact.bytes>0,"empty graph");retain_orphan(root_/"assignment.cagra");publish_exclusive(temp,root_/"assignment.cagra");graph_=artifact;write_state();
}
std::string AssignmentCheckpoint::graph_path() const { need(has_graph(),"graph not committed");return (root_/"assignment.cagra").string(); }
void AssignmentCheckpoint::load_labels(std::vector<uint32_t>& values) const {
    need(values.size()==config_.tokens,"assignment vector size mismatch");
    for(const auto& c:chunks_) {auto* dest=values.data()+c.start;read_bytes(chunk_path(c.start),dest,c.count*sizeof(uint32_t));need(checkpoint_sha256(dest,c.count*sizeof(uint32_t))==c.sha,"labels changed during load");}
}
void AssignmentCheckpoint::append_labels(size_t start,const uint32_t* labels,size_t count) {
    need(has_graph() && start==done_ && count>0 && count==std::min(config_.batch_size,config_.tokens-start),"assignment batch mismatch");
    for(size_t i=0;i<count;i++) need(labels[i]<config_.clusters,"invalid assignment label");
    auto destination=chunk_path(start);retain_orphan(destination);auto temp=temporary(root_/"labels","labels");write_bytes(temp,labels,count*sizeof(uint32_t));publish_exclusive(temp,destination);
    chunks_.push_back({start,count,checkpoint_sha256(labels,count*sizeof(uint32_t))});done_+=count;
    if(done_==config_.tokens) {std::vector<std::pair<size_t,size_t>> paths;for(auto& c:chunks_)paths.emplace_back(c.start,c.count);labels_sha_=concat_sha(root_,paths);}
    write_state();
}
}  // namespace Chimera
