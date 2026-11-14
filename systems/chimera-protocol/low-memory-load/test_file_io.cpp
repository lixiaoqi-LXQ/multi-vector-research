#include "cpp/include/chimera/index_file_io.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>

using namespace Chimera;
static void write(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    assert(f.good());
}
static std::vector<char> read(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
template <typename F> static void rejects(F function) {
    bool error = false;
    try { function(); } catch (const std::exception&) { error = true; }
    assert(error);
}
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string root(argv[1]);
    std::filesystem::create_directories(root);
    std::mt19937 generator(123);
    for (const size_t bits : {4u, 8u}) {
        const size_t n = 123;
        std::vector<char> full(n * 128 * bits / 8);
        for (auto& value : full) value = static_cast<char>(generator() & 255);
        const auto path = root + "/full" + std::to_string(bits) + ".bin";
        write(path, full);
        {
            ReadOnlyMappedFile mapped(path, full.size());
            assert(mapped.size() == full.size());
            mapped.prefault();
            assert(std::memcmp(mapped.data(), full.data(), full.size()) == 0);
            // Same token stride and byte pointer used by the real CPU scorer.
            const size_t stride = 128 * bits / 8;
            for (size_t token = 0; token < n; ++token) {
                assert(std::memcmp(mapped.data() + token * stride, full.data() + token * stride, stride) == 0);
            }
        }
        assert(read(path) == full);
        rejects([&] { ReadOnlyMappedFile too_short(path, full.size() - 1); });
        rejects([&] { ReadOnlyMappedFile too_long(path, full.size() + 1); });
        std::cout << "mapped_full_codes_" << bits << "bit_exact_and_prefault_ok\n";
    }
    rejects([&] { ReadOnlyMappedFile missing(root + "/missing", 4); });
    rejects([&] { ReadOnlyMappedFile directory(root, 4); });
    std::vector<size_t> sizes = {16*123, 4*123, 4*123, 4*123};
    std::vector<char> serialized(std::accumulate(sizes.begin(), sizes.end(), size_t(0)));
    for (auto& value : serialized) value = static_cast<char>(generator() & 255);
    const auto path = root + "/cluster.bin";
    write(path, serialized);
    for (const size_t chunk : {1u, 7u, 64u, 1024u, 65536u}) {
        std::vector<std::vector<char>> destinations;
        for (const auto size : sizes) destinations.emplace_back(size);
        size_t largest_chunk = 0;
        stream_index_segments(path, sizes, chunk,
            [&](size_t segment, size_t offset, const char* bytes, size_t count) {
                assert(offset + count <= destinations[segment].size());
                largest_chunk = std::max(largest_chunk, count);
                std::memcpy(destinations[segment].data() + offset, bytes, count);
            });
        assert(largest_chunk <= chunk);
        size_t begin = 0;
        for (size_t segment = 0; segment < sizes.size(); ++segment) {
            assert(std::memcmp(destinations[segment].data(), serialized.data() + begin, sizes[segment]) == 0);
            begin += sizes[segment];
        }
    }
    rejects([&] { stream_index_segments(path, sizes, 0, [](auto...){ }); });
    auto invalid = sizes; invalid.back()--;
    rejects([&] { stream_index_segments(path, invalid, 7, [](auto...){ }); });
    write(root + "/empty.bin", {});
    rejects([&] { ReadOnlyMappedFile empty(root + "/empty.bin", 0); });
    const auto old_file = root + "/existing-index.bin";
    write(old_file, serialized);
    rejects([&] { reject_readonly_index_save(true); write(old_file, {}); });
    assert(read(old_file) == serialized);
    reject_readonly_index_save(false);
    // Check the exact seek distance used to skip CPU-unused doc1bit arrays.
    const size_t n = 123, prefix = 37;
    std::vector<char> onebit(prefix + n*16 + n*4 + n*4, char(0));
    std::vector<float> factors(n);
    for (size_t i=0; i<n; ++i) factors[i]=static_cast<float>(i)*.25f;
    std::memcpy(onebit.data()+prefix+n*20, factors.data(), n*4);
    write(root + "/doc1bit.bin", onebit);
    std::ifstream stream(root + "/doc1bit.bin", std::ios::binary);
    stream.seekg(prefix);
    stream.seekg(static_cast<std::streamoff>(n*(128/8+sizeof(float))), std::ios::cur);
    std::vector<float> loaded(n);
    stream.read(reinterpret_cast<char*>(loaded.data()), static_cast<std::streamsize>(n*4));
    assert(stream.good() && loaded == factors);
    std::cout << "four_segment_stream_exact_with_5_chunk_sizes; rejects_truncation_and_wrong_sizes\n";
    std::cout << "save_guard_preserves_existing_file; factor_seek_exact\n";
}
