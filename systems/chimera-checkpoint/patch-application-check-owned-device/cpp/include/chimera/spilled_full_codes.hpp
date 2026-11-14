#pragma once
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace Chimera {

// Owns exactly one O_EXCL-created temporary name. Saved hardlinks are never removed.
class SpilledFullCodes {
 public:
    SpilledFullCodes(const std::string& path, size_t bytes,
                     size_t flush_bytes = 64ull * 1024ull * 1024ull)
        : path_(std::filesystem::absolute(path).lexically_normal().string()),
          bytes_(bytes), flush_bytes_(flush_bytes) {
        if (bytes == 0 || bytes > static_cast<uintmax_t>(std::numeric_limits<off_t>::max()) || flush_bytes == 0) {
            throw std::invalid_argument("Spill size and flush window must be positive and fit off_t");
        }
        fd_ = ::open(path_.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd_ < 0) fail("exclusive spill create");
        try {
            struct stat state {};
            if (::fstat(fd_, &state) != 0) fail("spill fstat");
            device_ = state.st_dev; inode_ = state.st_ino; identity_known_ = true;
            // Fail ENOSPC before encoding; reserved extents read as zero like vector<char>.
            const int error = ::posix_fallocate(fd_, 0, static_cast<off_t>(bytes_));
            if (error != 0) { errno = error; fail("spill disk reservation"); }
            mapping_ = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
            if (mapping_ == MAP_FAILED) fail("shared spill mmap");
        } catch (...) { cleanup(); throw; }
    }
    ~SpilledFullCodes() { cleanup(); }
    SpilledFullCodes(const SpilledFullCodes&) = delete;
    SpilledFullCodes& operator=(const SpilledFullCodes&) = delete;
    char* data() { return static_cast<char*>(mapping_); }
    const char* data() const { return static_cast<const char*>(mapping_); }
    size_t size() const { return bytes_; }
    const std::string& temporary_path() const { return path_; }

    size_t rows_per_flush(size_t stride) const {
        if (stride == 0) throw std::invalid_argument("Full-code stride must be positive");
        const size_t page = page_size();
        const size_t alignment_rows = page / std::gcd(page, stride);
        const size_t rows = std::max<size_t>(1, flush_bytes_ / stride);
        return std::max(alignment_rows, rows / alignment_rows * alignment_rows);
    }
    void sync_and_drop(size_t offset, size_t count) {
        if (offset > bytes_ || count > bytes_ - offset) throw std::out_of_range("Spill flush exceeds mapping");
        if (count == 0) return;
        const size_t aligned = offset / page_size() * page_size();
        void* begin = data() + aligned;
        const size_t length = offset - aligned + count;
        if (::msync(begin, length, MS_SYNC) != 0) fail("spill writeback");
        if (::madvise(begin, length, MADV_DONTNEED) != 0) fail("spill clean-page release");
    }

    // Read-only preflight, invoked before the original save touches output files.
    void preflight_save_directory(const std::string& index_dir) const {
        validate_name();
        const auto destination = std::filesystem::weakly_canonical(std::filesystem::absolute(index_dir));
        const auto source = std::filesystem::weakly_canonical(path_);
        for (auto parent = source.parent_path(); !parent.empty();) {
            if (parent == destination) {
                throw std::invalid_argument("Spill temporary must be outside the saved index directory");
            }
            const auto next = parent.parent_path();
            if (next == parent) break;
            parent = next;
        }
        auto existing = destination;
        while (!std::filesystem::exists(existing)) {
            const auto next = existing.parent_path();
            if (next == existing || next.empty()) throw std::runtime_error("No existing save ancestor");
            existing = next;
        }
        struct stat state {};
        if (::stat(existing.c_str(), &state) != 0) fail("save ancestor stat");
        if (!S_ISDIR(state.st_mode) || state.st_dev != device_) {
            throw std::runtime_error("Spill and index must be on the same filesystem for hardlink save");
        }
    }
    void publish_hardlink(const std::string& target) const {
        validate_name();
        if (::msync(mapping_, bytes_, MS_SYNC) != 0 || ::fdatasync(fd_) != 0) fail("spill final sync");
        // Does not replace an existing name; normal save prepares the destination.
        if (::link(path_.c_str(), target.c_str()) != 0) fail("spill hardlink publish");
        const auto parent = std::filesystem::path(target).parent_path();
        const int directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory < 0) fail("saved index directory open");
        const int status = ::fsync(directory);
        const int error = errno;
        (void)::close(directory);
        if (status != 0) { errno = error; fail("saved index directory sync"); }
    }
 private:
    static size_t page_size() {
        const long value = ::sysconf(_SC_PAGESIZE);
        return value > 0 ? static_cast<size_t>(value) : 4096;
    }
    void validate_name() const {
        struct stat state {};
        if (::lstat(path_.c_str(), &state) != 0 || !S_ISREG(state.st_mode) ||
            state.st_dev != device_ || state.st_ino != inode_ ||
            state.st_size < 0 || static_cast<uintmax_t>(state.st_size) != bytes_) {
            throw std::runtime_error("Spill temporary name/identity/length changed: " + path_);
        }
    }
    [[noreturn]] void fail(const char* operation) const {
        const int error = errno;
        throw std::runtime_error(std::string(operation) + " failed for " + path_ + ": " + std::strerror(error));
    }
    void cleanup() noexcept {
        if (mapping_ != MAP_FAILED) { (void)::munmap(mapping_, bytes_); mapping_ = MAP_FAILED; }
        if (fd_ >= 0) { (void)::close(fd_); fd_ = -1; }
        struct stat state {};
        if (identity_known_ && ::lstat(path_.c_str(), &state) == 0 && S_ISREG(state.st_mode) &&
            state.st_dev == device_ && state.st_ino == inode_) {
            (void)::unlink(path_.c_str());
        }
    }
    std::string path_;
    size_t bytes_ = 0;
    size_t flush_bytes_ = 0;
    int fd_ = -1;
    void* mapping_ = MAP_FAILED;
    dev_t device_ {};
    ino_t inode_ {};
    bool identity_known_ = false;
};

}  // namespace Chimera
