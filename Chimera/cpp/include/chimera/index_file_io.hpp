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
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace Chimera {

// A raw, immutable index payload, not a CUDA-mapped allocation.
class ReadOnlyMappedFile {
 public:
    ReadOnlyMappedFile(const std::string& path, size_t expected_bytes) {
        struct Descriptor {
            int value;
            ~Descriptor() { if (value >= 0) (void)::close(value); }
        } fd{::open(path.c_str(), O_RDONLY | O_CLOEXEC)};
        if (fd.value < 0) fail("open", path);
        struct stat state {};
        if (::fstat(fd.value, &state) != 0) fail("fstat", path);
        if (!S_ISREG(state.st_mode) || expected_bytes == 0 || state.st_size < 0 ||
            static_cast<uintmax_t>(state.st_size) != static_cast<uintmax_t>(expected_bytes)) {
            throw std::runtime_error("Index payload size/type mismatch: " + path);
        }
        bytes_ = expected_bytes;
        mapping_ = ::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd.value, 0);
        if (mapping_ == MAP_FAILED) fail("mmap", path);
        // CPU random full-bit scoring reads ordinary file-backed pages.
        (void)::madvise(mapping_, bytes_, MADV_RANDOM);
    }
    ~ReadOnlyMappedFile() {
        if (mapping_ != MAP_FAILED) (void)::munmap(mapping_, bytes_);
    }
    ReadOnlyMappedFile(const ReadOnlyMappedFile&) = delete;
    ReadOnlyMappedFile& operator=(const ReadOnlyMappedFile&) = delete;
    const char* data() const { return static_cast<const char*>(mapping_); }
    size_t size() const { return bytes_; }
    // Match the original eager file read before query timing. Pages remain
    // reclaimable; this is not mlock or guaranteed residency under memory pressure.
    void prefault() const {
        const long configured = ::sysconf(_SC_PAGESIZE);
        const size_t page = configured > 0 ? static_cast<size_t>(configured) : 4096;
        const volatile char* data = static_cast<const char*>(mapping_);
        for (size_t offset = 0; offset < bytes_; offset += page) {
            const char value = data[offset];
            (void)value;
        }
    }
 private:
    [[noreturn]] static void fail(const char* operation, const std::string& path) {
        const int error = errno;
        throw std::runtime_error(std::string(operation) + " failed for " + path + ": " + std::strerror(error));
    }
    void* mapping_ = MAP_FAILED;
    size_t bytes_ = 0;
};

inline void reject_readonly_index_save(bool low_memory_load) {
    if (low_memory_load) {
        throw std::logic_error("save() is unavailable after low-memory load; reload with default LoadOptions before saving");
    }
}

// Read each serialized segment into a single bounded host buffer. consume receives
// (segment index, segment-relative offset, bytes, count); it must finish reading
// the buffer before returning. cudaMemcpy's synchronous H2D form satisfies this.
template <typename Consumer>
void stream_index_segments(const std::string& path,
                           const std::vector<size_t>& segment_bytes,
                           size_t buffer_bytes, Consumer consume) {
    if (buffer_bytes == 0 || buffer_bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::invalid_argument("Index streaming buffer size must be positive and fit streamsize");
    }
    size_t total = 0;
    for (const size_t count : segment_bytes) {
        if (count > std::numeric_limits<size_t>::max() - total) {
            throw std::overflow_error("Index segment sizes overflow size_t");
        }
        total += count;
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() < 0 || static_cast<uintmax_t>(input.tellg()) != static_cast<uintmax_t>(total)) {
        throw std::runtime_error("Clustered index payload size mismatch: " + path);
    }
    input.seekg(0);
    std::vector<char> buffer(std::min(total, buffer_bytes));
    for (size_t segment = 0; segment < segment_bytes.size(); ++segment) {
        for (size_t offset = 0; offset < segment_bytes[segment];) {
            const size_t count = std::min(buffer.size(), segment_bytes[segment] - offset);
            input.read(buffer.data(), static_cast<std::streamsize>(count));
            if (!input) throw std::runtime_error("Short clustered index read: " + path);
            consume(segment, offset, buffer.data(), count);
            offset += count;
        }
    }
}

}  // namespace Chimera
