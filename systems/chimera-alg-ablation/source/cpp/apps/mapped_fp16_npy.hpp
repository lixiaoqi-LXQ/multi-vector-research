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
#include <regex>
#include <stdexcept>
#include <string>

// Read-only C-order NPY v1/v2, dtype <u2 raw FP16 bits or <f2, shape [N,D].
// Optional token prefix maps only complete required rows, enabling .partial use.
class MappedFp16Npy {
 public:
    explicit MappedFp16Npy(const std::string& path, size_t requested_rows = 0) {
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "MappedFp16Npy requires a little-endian host"
#endif
        struct Descriptor { int value; ~Descriptor() { if (value >= 0) (void)::close(value); } } fd{
            ::open(path.c_str(), O_RDONLY | O_CLOEXEC)};
        if (fd.value < 0) fail("open", path);
        struct stat state {};
        if (::fstat(fd.value, &state) != 0) fail("fstat", path);
        if (!S_ISREG(state.st_mode) || state.st_size < 12) {
            throw std::runtime_error("NPY input must be a regular file: " + path);
        }
        uint8_t prefix[12];
        read_at(fd.value, prefix, sizeof(prefix), 0, path);
        const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
        if (std::memcmp(prefix, magic, sizeof(magic)) != 0 || prefix[7] != 0 ||
            (prefix[6] != 1 && prefix[6] != 2)) {
            throw std::runtime_error("Expected NPY format v1/v2: " + path);
        }
        size_t header_start = prefix[6] == 1 ? 10 : 12;
        size_t header_bytes = static_cast<size_t>(prefix[8]) | (static_cast<size_t>(prefix[9]) << 8);
        if (prefix[6] == 2) header_bytes |= (static_cast<size_t>(prefix[10]) << 16) | (static_cast<size_t>(prefix[11]) << 24);
        if (header_bytes == 0 || header_bytes > 1024 * 1024 ||
            header_start + header_bytes > static_cast<uintmax_t>(state.st_size)) {
            throw std::runtime_error("Invalid/truncated NPY header: " + path);
        }
        std::string header(header_bytes, '\0');
        read_at(fd.value, header.data(), header_bytes, static_cast<off_t>(header_start), path);
        const std::regex dtype(R"re(["']descr["']\s*:\s*["']<(u2|f2)["'])re");
        const std::regex order(R"re(["']fortran_order["']\s*:\s*False)re");
        const std::regex shape(R"re(["']shape["']\s*:\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,?\s*\))re");
        std::smatch match;
        if (!std::regex_search(header, dtype) || !std::regex_search(header, order) ||
            !std::regex_search(header, match, shape)) {
            throw std::runtime_error("Expected C-order NPY <u2/<f2 matrix [N,D]: " + path);
        }
        const auto parsed_rows = std::stoull(match[1].str());
        const auto parsed_dimension = std::stoull(match[2].str());
        if (parsed_rows == 0 || parsed_dimension == 0 ||
            parsed_rows > std::numeric_limits<size_t>::max() ||
            parsed_dimension > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("Invalid NPY matrix dimensions: " + path);
        }
        const size_t rows = static_cast<size_t>(parsed_rows);
        dimension_ = static_cast<size_t>(parsed_dimension);
        offset_ = header_start + header_bytes;
        if (offset_ % alignof(uint16_t) != 0 ||
            dimension_ > (std::numeric_limits<size_t>::max() - offset_) / sizeof(uint16_t) / rows) {
            throw std::runtime_error("NPY matrix size/alignment invalid: " + path);
        }
        const size_t expected_bytes = offset_ + rows * dimension_ * sizeof(uint16_t);
        const size_t mapped_rows = requested_rows == 0 ? rows : requested_rows;
        if (mapped_rows > rows) throw std::runtime_error("Requested token prefix exceeds NPY matrix: " + path);
        values_ = mapped_rows * dimension_;
        bytes_ = offset_ + values_ * sizeof(uint16_t);
        if (static_cast<uintmax_t>(state.st_size) > static_cast<uintmax_t>(expected_bytes) ||
            static_cast<uintmax_t>(state.st_size) < static_cast<uintmax_t>(bytes_)) {
            throw std::runtime_error("NPY file does not contain the requested complete prefix, or has trailing bytes: " + path);
        }
        mapping_ = ::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd.value, 0);
        if (mapping_ == MAP_FAILED) fail("mmap", path);
        (void)::madvise(mapping_, bytes_, MADV_SEQUENTIAL);
    }
    ~MappedFp16Npy() { if (mapping_ != MAP_FAILED) (void)::munmap(mapping_, bytes_); }
    MappedFp16Npy(const MappedFp16Npy&) = delete;
    MappedFp16Npy& operator=(const MappedFp16Npy&) = delete;
    const uint16_t* data() const {
        return reinterpret_cast<const uint16_t*>(static_cast<const char*>(mapping_) + offset_);
    }
    size_t size() const { return values_; }
    size_t dimension() const { return dimension_; }
 private:
    static void read_at(int fd, void* buffer, size_t count, off_t offset, const std::string& path) {
        if (::pread(fd, buffer, count, offset) != static_cast<ssize_t>(count)) {
            throw std::runtime_error("Failed reading NPY header: " + path);
        }
    }
    [[noreturn]] static void fail(const char* operation, const std::string& path) {
        const int error = errno;
        throw std::runtime_error(std::string(operation) + " failed for " + path + ": " + std::strerror(error));
    }
    void* mapping_ = MAP_FAILED;
    size_t bytes_ = 0;
    size_t values_ = 0;
    size_t dimension_ = 0;
    size_t offset_ = 0;
};
