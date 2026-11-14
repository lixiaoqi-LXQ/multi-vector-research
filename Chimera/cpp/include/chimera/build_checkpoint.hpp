#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace Chimera {
struct AssignmentCheckpointConfig {
    std::string identity_sha256;
    std::string doclens_sha256;
    size_t tokens = 0, dimension = 0, clusters = 0, ex_bits = 0;
    size_t batch_size = 65536, sample_seed = 42, rotator_seed = 42;
    size_t assignment_k = 1, assignment_itopk = 150;
};
std::string checkpoint_sha256(const void*, size_t);
std::string checkpoint_file_sha256(const std::filesystem::path&);
// Metadata/files are exclusive to one builder. Failed uncommitted files are retained as orphans.
class AssignmentCheckpoint {
 public:
    AssignmentCheckpoint(const std::string&, const AssignmentCheckpointConfig&, bool resume);
    ~AssignmentCheckpoint();
    AssignmentCheckpoint(const AssignmentCheckpoint&) = delete;
    AssignmentCheckpoint& operator=(const AssignmentCheckpoint&) = delete;
    bool has_centroids() const;
    bool has_graph() const;
    size_t completed_tokens() const { return done_; }
    std::vector<float> load_centroids() const;
    void save_centroids(const std::vector<float>&);
    std::filesystem::path reserve_graph_temporary();
    void commit_graph(const std::filesystem::path&);
    std::string graph_path() const;
    void load_labels(std::vector<uint32_t>&) const;
    void append_labels(size_t start, const uint32_t*, size_t count);
 private:
    struct Artifact { size_t bytes = 0; std::string sha = "-"; };
    struct Chunk { size_t start, count; std::string sha; };
    void write_state();
    void read_state();
    void validate() const;
    std::filesystem::path chunk_path(size_t) const;
    std::filesystem::path root_;
    AssignmentCheckpointConfig config_;
    Artifact centroids_, graph_;
    size_t done_ = 0;
    std::string labels_sha_ = "-";
    std::vector<Chunk> chunks_;
    int lock_fd_ = -1;
};
}  // namespace Chimera
