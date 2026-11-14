from pathlib import Path
import difflib
r=Path(__file__).parent
paths=[str(p.relative_to(r/'base')) for p in (r/'base').rglob('*') if p.is_file()]
original={p:(r/'base'/p).read_text() for p in paths};changed=dict(original)
p='cpp/include/chimera/chimera_index.cuh';s=changed[p]
s=s.replace('struct IVF_PG;', '''struct IVF_PG;
class ReadOnlyMappedFile;

struct LoadOptions {
    bool low_memory = false;
    size_t transfer_chunk_bytes = 64ull * 1024ull * 1024ull;
    // True reads mapped CPU pages once during load, before query timing.
    bool prefault_full_codes = true;
};''')
needle='''        const SearchOptions& options = {});

    chimera_index(const chimera_index&)'''
replacement='''        const SearchOptions& options = {});
    void load(
        const std::string& index_dir,
        const SearchOptions& options,
        const LoadOptions& load_options);

    chimera_index(const chimera_index&)'''
assert needle in s;s=s.replace(needle,replacement)
s=s.replace('    std::vector<char> full_bit_codes_;', '''    std::vector<char> full_bit_codes_;
    std::unique_ptr<ReadOnlyMappedFile> mapped_full_bit_codes_;
    const char* mapped_full_bit_data_ = nullptr;
    LoadOptions load_options_;
    bool low_memory_load_ = false;
    std::string load_directory_;''');changed[p]=s
p='cpp/src/chimera_index.cu';s=changed[p]
s=s.replace('#include "chimera/io.hpp"','#include "chimera/io.hpp"\n#include "chimera/index_file_io.hpp"')
needle='''    const SearchOptions& options) {
    validate_search_options(options);
    const std::filesystem::path directory(index_dir);'''
replacement='''    const SearchOptions& options) {
    load(index_dir, options, LoadOptions {});
}

void chimera_index::load(
    const std::string& index_dir,
    const SearchOptions& options,
    const LoadOptions& load_options) {
    validate_search_options(options);
    if (load_options.transfer_chunk_bytes == 0) {
        throw std::invalid_argument("Load transfer_chunk_bytes must be positive");
    }
    const std::filesystem::path directory(index_dir);'''
assert needle in s;s=s.replace(needle,replacement)
needle='''    begin_initialization(options);

    const auto doclens_path'''
replacement='''    begin_initialization(options);
    load_options_ = load_options;
    low_memory_load_ = load_options.low_memory;
    load_directory_ = std::filesystem::absolute(directory).string();
    if (low_memory_load_) {
        // release_resources clears vector sizes; drop capacities from a previous
        // ordinary load too so switching modes on the same object saves memory.
        std::vector<char>().swap(full_bit_codes_);
        std::vector<uint64_t>().swap(one_bit_codes_);
        std::vector<float>().swap(one_bit_factors_);
        std::vector<char>().swap(clustered_data_);
        std::vector<float>().swap(full_bit_factors_);
    }

    const auto doclens_path'''
assert needle in s;s=s.replace(needle,replacement)
start=s.index('    one_bit_codes_.resize(n * (PADDED_DIM / 64));');end=s.index('    ivf = new IVF_PG',start)
s=s[:start]+'''    full_bit_factors_.resize(n);
    if (low_memory_load_) {
        // doc_1bit.bin after its rotator stores codes[16*N], one-bit
        // factors[4*N], then CPU full-bit factors[4*N]. Search uses only
        // the last segment on CPU; GPU 1-bit data comes from cluster_1bit.bin.
        const size_t skip = n * (PADDED_DIM / 8 + sizeof(float));
        inf.seekg(static_cast<std::streamoff>(skip), std::ios::cur);
    } else {
        one_bit_codes_.resize(n * (PADDED_DIM / 64));
        one_bit_factors_.resize(n);
        inf.read(reinterpret_cast<char*>(one_bit_codes_.data()),
                 static_cast<std::streamsize>(one_bit_codes_.size() * sizeof(uint64_t)));
        inf.read(reinterpret_cast<char*>(one_bit_factors_.data()),
                 static_cast<std::streamsize>(one_bit_factors_.size() * sizeof(float)));
    }
    inf.read(reinterpret_cast<char*>(full_bit_factors_.data()),
             static_cast<std::streamsize>(full_bit_factors_.size() * sizeof(float)));
    if (!inf) {
        throw std::runtime_error("Failed to read quantized payload from " + doc_one_bit_path);
    }
    const size_t full_code_bytes = n * PADDED_DIM * (1 + ex_bits) / 8;
    if (low_memory_load_) {
        mapped_full_bit_codes_ = std::make_unique<ReadOnlyMappedFile>(doc_full_path, full_code_bytes);
        mapped_full_bit_data_ = mapped_full_bit_codes_->data();
        if (load_options_.prefault_full_codes) mapped_full_bit_codes_->prefault();
    } else {
        full_bit_codes_.resize(full_code_bytes);
        std::ifstream full_bit_codes_file(doc_full_path, std::ios::binary);
        full_bit_codes_file.read(full_bit_codes_.data(), full_bit_codes_.size());
        if (!full_bit_codes_file) {
            throw std::runtime_error("Failed to read full-code payload from " + doc_full_path);
        }
    }
'''+s[end:]
needle='''    clustered_data_.resize(clustered_bytes);
    clustered_file.seekg(0);
    clustered_file.read(
        clustered_data_.data(),
        static_cast<std::streamsize>(clustered_data_.size()));
    if (!clustered_file) {
        throw std::runtime_error(
            "Failed to read clustered index: " + cluster_one_bit_path);
    }'''
replacement='''    if (!low_memory_load_) {
        clustered_data_.resize(clustered_bytes);
        clustered_file.seekg(0);
        clustered_file.read(clustered_data_.data(),
                            static_cast<std::streamsize>(clustered_data_.size()));
        if (!clustered_file) {
            throw std::runtime_error("Failed to read clustered index: " + cluster_one_bit_path);
        }
    }'''
assert needle in s;s=s.replace(needle,replacement)
s=s.replace('''    if (clustered_data_.size() != expected_clustered_bytes) {''', '''    if (!low_memory_load_ && clustered_data_.size() != expected_clustered_bytes) {''')
s=s.replace('if (free_mem <= clustered_data_.size()) {','if (free_mem <= expected_clustered_bytes) {').replace('std::to_string(clustered_data_.size()) + " bytes, have " +','std::to_string(expected_clustered_bytes) + " bytes, have " +')
start=s.index('    const char* clustered_code = clustered_data_.data();');end=s.index('    compute_candidate_bounds();',start)
s=s[:start]+'''    check_cuda(cudaMalloc(&d_clustered_one_bit_codes_, n * code_per_vec));
    check_cuda(cudaMalloc(&d_clustered_one_bit_factors_, n * sizeof(float)));
    check_cuda(cudaMalloc(&d_clustered_doc_ids_, n * sizeof(int)));
    check_cuda(cudaMalloc(&d_token_to_cluster_pos_, n * sizeof(uint32_t)));

    if (low_memory_load_) {
        const auto path = (std::filesystem::path(load_directory_) / "cluster_1bit.bin").string();
        const std::vector<size_t> sizes = {
            n * code_per_vec, n * sizeof(float), n * sizeof(int), n * sizeof(uint32_t)};
        char* destinations[] = {
            d_clustered_one_bit_codes_,
            reinterpret_cast<char*>(d_clustered_one_bit_factors_),
            reinterpret_cast<char*>(d_clustered_doc_ids_),
            reinterpret_cast<char*>(d_token_to_cluster_pos_)};
        stream_index_segments(path, sizes, load_options_.transfer_chunk_bytes,
            [&](size_t segment, size_t offset, const char* data, size_t count) {
                check_cuda(cudaMemcpy(destinations[segment] + offset, data, count,
                                      cudaMemcpyHostToDevice));
            });
    } else {
        const char* clustered_code = clustered_data_.data();
        const auto* clustered_factor = reinterpret_cast<const float*>(
            clustered_code + n * code_per_vec);
        const auto* clustered_doc_ids = reinterpret_cast<const int*>(
            reinterpret_cast<const char*>(clustered_factor) + n * sizeof(float));
        const auto* token_to_cluster_pos = reinterpret_cast<const uint32_t*>(
            reinterpret_cast<const char*>(clustered_doc_ids) + n * sizeof(int));
        check_cuda(cudaMemcpy(d_clustered_one_bit_codes_, clustered_code,
                              n * code_per_vec, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_clustered_one_bit_factors_, clustered_factor,
                              n * sizeof(float), cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_clustered_doc_ids_, clustered_doc_ids,
                              n * sizeof(int), cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_token_to_cluster_pos_, token_to_cluster_pos,
                              n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

'''+s[end:];changed[p]=s
p='cpp/src/utils.cu';s=changed[p].replace('#include "chimera/kernels.cuh"','#include "chimera/kernels.cuh"\n#include "chimera/index_file_io.hpp"')
needle='''    full_bit_codes_.clear();'''
replacement='''    mapped_full_bit_data_ = nullptr;
    mapped_full_bit_codes_.reset();
    low_memory_load_ = false;
    load_options_ = LoadOptions {};
    load_directory_.clear();
    full_bit_codes_.clear();'''
assert needle in s;s=s.replace(needle,replacement,1);changed[p]=s
p='cpp/src/cpu_scoring.cpp';s=changed[p];needle='    const auto& full_bit_codes = full_bit_codes_;'
replacement='''    const char* full_bit_codes = mapped_full_bit_data_ != nullptr
        ? mapped_full_bit_data_ : full_bit_codes_.data();'''
assert needle in s;s=s.replace(needle,replacement);changed[p]=s
p='cpp/src/build_index.cu';s=changed[p].replace('#include "chimera/config.cuh"','#include "chimera/config.cuh"\n#include "chimera/index_file_io.hpp"')
needle='''void chimera_index::save(const std::string& index_dir) const {
''';replacement=needle+'''    // Reject before prepare_output_directory can remove existing payloads.
    reject_readonly_index_save(low_memory_load_);
''';assert needle in s;s=s.replace(needle,replacement);changed[p]=s
p='cpp/apps/utils.hpp';s=changed[p].replace('''    SearchOptions options;
};''','''    SearchOptions options;
    LoadOptions load_options;
};''');changed[p]=s
p='cpp/apps/utils.cpp';s=changed[p]
needle='''        if (arg == "--output") {''';replacement='''        if (arg == "--low-memory") {
            args.load_options.low_memory = true;
        } else if (arg == "--output") {''';assert needle in s;s=s.replace(needle,replacement,1)
needle='''        << "  [--output <results.tsv>]''';replacement='''        << "  [--low-memory]                 Read-only CPU full-code mmap and bounded clustered upload; save disabled.\\n"
        << "  [--output <results.tsv>]''';assert needle in s;s=s.replace(needle,replacement,1);changed[p]=s
p='cpp/apps/search.cpp';s=changed[p];needle='''    index.load(args.index_dir, args.options);''';replacement='''    index.load(args.index_dir, args.options, args.load_options);
    std::cout << "[LOAD] low_memory=" << args.load_options.low_memory
              << " prefault_full_codes=" << args.load_options.prefault_full_codes
              << " transfer_chunk_bytes=" << args.load_options.transfer_chunk_bytes << '\\n';''';assert needle in s;s=s.replace(needle,replacement);changed[p]=s
for p,s in changed.items():
 dst=r/'modified'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(s)
patch=[]
for p in sorted(paths):patch.extend(difflib.unified_diff(original[p].splitlines(True),changed[p].splitlines(True),fromfile='a/'+p,tofile='b/'+p))
p='cpp/include/chimera/index_file_io.hpp';patch.extend(difflib.unified_diff([],(r/'modified'/p).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+p))
(r/'low-memory-load.patch').write_text(''.join(patch))
