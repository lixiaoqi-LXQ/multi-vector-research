#pragma once

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

// Same little-endian [int32 N, int32 D, float32 N*D] format as load_data.
// The input must remain an immutable regular file until this object is destroyed.
class MappedEmbeddings {
 public:
    explicit MappedEmbeddings(const std::string& path) {
        static_assert(sizeof(float) == 4, "FP32 input requires four-byte floats");
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "MappedEmbeddings currently requires a little-endian host"
#endif
        struct FileDescriptor {
            int value;
            ~FileDescriptor() { if (value >= 0) (void)::close(value); }
        } fd{::open(path.c_str(), O_RDONLY | O_CLOEXEC)};
        if (fd.value < 0) fail("open", path);
        struct stat state {};
        if (::fstat(fd.value, &state) != 0) fail("fstat", path);
        if (!S_ISREG(state.st_mode) || state.st_size < 8) {
            throw std::runtime_error("Embedding input must be a regular file with an 8-byte header: " + path);
        }
        int32_t header[2];
        if (::pread(fd.value, header, sizeof(header), 0) != sizeof(header)) {
            throw std::runtime_error("Could not read complete embedding header: " + path);
        }
        if (header[0] <= 0 || header[1] <= 0) {
            throw std::runtime_error("Embedding N and D must be positive: " + path);
        }
        dimension_ = static_cast<size_t>(header[1]);
        const size_t rows = static_cast<size_t>(header[0]);
        if (rows > (std::numeric_limits<size_t>::max() - 8) / sizeof(float) / dimension_) {
            throw std::runtime_error("Embedding shape overflows host size_t: " + path);
        }
        values_ = rows * dimension_;
        bytes_ = 8 + values_ * sizeof(float);
        if (static_cast<uintmax_t>(state.st_size) != static_cast<uintmax_t>(bytes_)) {
            throw std::runtime_error("Embedding file size disagrees with its N*D header: " + path);
        }
        mapping_ = ::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd.value, 0);
        if (mapping_ == MAP_FAILED) fail("mmap", path);
        // No anonymous corpus copy, mlock, or MAP_POPULATE: clean pages are reclaimable.
        // Sequential advice allows the kernel to reclaim earlier pages during long scans.
        (void)::madvise(mapping_, bytes_, MADV_SEQUENTIAL);
    }
    ~MappedEmbeddings() { if (mapping_ != MAP_FAILED) (void)::munmap(mapping_, bytes_); }
    MappedEmbeddings(const MappedEmbeddings&) = delete;
    MappedEmbeddings& operator=(const MappedEmbeddings&) = delete;
    const float* data() const {
        return reinterpret_cast<const float*>(static_cast<const char*>(mapping_) + 8);
    }
    size_t size() const { return values_; }
    size_t dimension() const { return dimension_; }
 private:
    [[noreturn]] static void fail(const char* operation, const std::string& path) {
        const int code = errno;
        throw std::runtime_error(std::string(operation) + " failed for " + path + ": " + std::strerror(code));
    }
    void* mapping_ = MAP_FAILED;
    size_t bytes_ = 0;
    size_t values_ = 0;
    size_t dimension_ = 0;
};
