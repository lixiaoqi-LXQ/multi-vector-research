from pathlib import Path
import difflib
src=Path('/home/lxq/graduate/ai-workspace/idea/multi-query/sources/Chimera-main')
out=Path(__file__).parent/'mmap-patch'
paths=['cpp/include/chimera/chimera_index.cuh','cpp/src/build_index.cu','cpp/apps/build.cpp']
original={p:(src/p).read_text() for p in paths};changed=dict(original)
p=paths[0];s=changed[p];needle='    std::vector<size_t> search(\n';pos=s.index(needle)
s=s[:pos]+'''    // Non-owning FP32 input; keep storage alive until this synchronous call returns.
    // embedding_values is the number of floats, not the number of token vectors.
    void build_from_ptr(
        const float* embeddings,
        size_t embedding_values,
        size_t dimension,
        const std::vector<int>& doc_lens,
        size_t n_clusters,
        size_t ex_bits,
        const SearchOptions& options = {});

'''+s[pos:];changed[p]=s
p=paths[1];s=changed[p]
s=s.replace('constexpr std::uint64_t kCentroidSampleSeed = 42;', '''// Minimal non-owning view shared by vector and mmap-backed callers.
class FloatView {
 public:
    FloatView(const float* data, size_t size) : data_(data), size_(size) {}
    const float* data() const { return data_; }
    size_t size() const { return size_; }
    bool empty() const { return data_ == nullptr || size_ == 0; }
 private:
    const float* data_;
    size_t size_;
};

template <typename T>
void release_vector(std::vector<T>& data) {
    std::vector<T>().swap(data);
}

constexpr std::uint64_t kCentroidSampleSeed = 42;''')
# Only non-owning embedding input changes; centroids remain owning vectors.
assert s.count('const std::vector<float>& embeddings')==5
s=s.replace('const std::vector<float>& embeddings','FloatView embeddings',4)
s=s.replace('std::vector<uint8_t> full_codes;', 'std::vector<char> full_codes;').replace('std::vector<uint8_t>(checked_product(\n            num_embeddings, full_code_bytes_per_vector, "full codes"))','std::vector<char>(checked_product(\n            num_embeddings, full_code_bytes_per_vector, "full codes"))')
s=s.replace('encoded.full_codes.data() + index * full_code_bytes_per_vector,','reinterpret_cast<uint8_t*>(encoded.full_codes.data()) + index * full_code_bytes_per_vector,')
start=s.index('struct ClusteredDocuments {');end=s.index('std::ofstream open_binary_output',start)
s=s[:start]+'''// Fill the final serialized layout directly, avoiding a second 28*N-byte copy.
std::vector<char> reorder_by_cluster(
    const EncodedDocuments& encoded,
    const IVF_PG& ivf,
    const std::vector<int>& token_doc_ids) {
    const size_t num_embeddings = ivf.inv_list.size();
    const size_t code_bytes_per_vector = PADDED_DIM / 8;
    const size_t code_bytes = checked_product(num_embeddings, code_bytes_per_vector, "clustered one-bit codes");
    const size_t factor_bytes = checked_product(num_embeddings, sizeof(float), "clustered factors");
    const size_t doc_bytes = checked_product(num_embeddings, sizeof(int), "clustered document IDs");
    const size_t position_bytes = checked_product(num_embeddings, sizeof(uint32_t), "clustered token positions");
    std::vector<char> output(code_bytes + factor_bytes + doc_bytes + position_bytes);
    char* codes = output.data();
    char* factors = codes + code_bytes;
    char* docs = factors + factor_bytes;
    char* positions = docs + doc_bytes;
    const auto* source_codes = reinterpret_cast<const char*>(encoded.one_bit_codes.data());
#pragma omp parallel for schedule(static)
    for (std::int64_t position = 0;
         position < static_cast<std::int64_t>(num_embeddings); ++position) {
        const size_t cluster_position = static_cast<size_t>(position);
        const uint32_t token_id = ivf.inv_list[cluster_position];
        std::memcpy(codes + cluster_position * code_bytes_per_vector,
                    source_codes + static_cast<size_t>(token_id) * code_bytes_per_vector,
                    code_bytes_per_vector);
        std::memcpy(factors + cluster_position * sizeof(float),
                    &encoded.one_bit_factors[token_id], sizeof(float));
        std::memcpy(docs + cluster_position * sizeof(int),
                    &token_doc_ids[token_id], sizeof(int));
        const auto compact_position = static_cast<uint32_t>(cluster_position);
        std::memcpy(positions + static_cast<size_t>(token_id) * sizeof(uint32_t),
                    &compact_position, sizeof(uint32_t));
    }
    return output;
}

'''+s[end:]
needle='''    const SearchOptions& options) {
    const size_t num_embeddings = validate_build_inputs('''
replace='''    const SearchOptions& options) {
    build_from_ptr(embeddings.data(), embeddings.size(), dimension, doc_lens,
                   n_clusters, ex_bits, options);
}

void chimera_index::build_from_ptr(
    const float* embedding_data,
    size_t embedding_values,
    size_t dimension,
    const std::vector<int>& doc_lens,
    size_t n_clusters,
    size_t ex_bits,
    const SearchOptions& options) {
    const FloatView embeddings(embedding_data, embedding_values);
    const size_t num_embeddings = validate_build_inputs('''
assert needle in s;s=s.replace(needle,replace,1)
needle='''    ivf->build_from_assignments(assignments.data(), num_embeddings);

    auto encoded'''
replace='''    ivf->build_from_assignments(assignments.data(), num_embeddings);
    release_vector(assignments);
    release_vector(centroids);
    release_vector(rotated_centroids);

    auto encoded'''
assert needle in s;s=s.replace(needle,replace)
s=s.replace('const auto token_doc_ids = make_token_doc_ids','auto token_doc_ids = make_token_doc_ids')
s=s.replace('auto clustered = reorder_by_cluster(encoded, *ivf, token_doc_ids);','auto clustered = reorder_by_cluster(encoded, *ivf, token_doc_ids);\n    release_vector(token_doc_ids);')
start=s.index('    full_bit_codes_.assign(');end=s.index('    has_index_ = true;',start)
s=s[:start]+'''    full_bit_codes_ = std::move(encoded.full_codes);
    clustered_data_ = std::move(clustered);

'''+s[end:];changed[p]=s
p=paths[2];s=changed[p].replace('#include <iostream>','#include <iostream>\n#include "mapped_embeddings.hpp"')
s=s.replace('''    size_t num_embeddings = 0;
    size_t dimension = 0;
    auto embeddings = load_data(num_embeddings, dimension, args.data_file);''','''    MappedEmbeddings embeddings(args.data_file);
    const size_t dimension = embeddings.dimension();''').replace('''    index.build(
        embeddings,
        dimension,''','''    index.build_from_ptr(
        embeddings.data(),
        embeddings.size(),
        dimension,''');changed[p]=s
for p,s in changed.items():(out/p).write_text(s)
patch=[]
for p in paths:patch.extend(difflib.unified_diff(original[p].splitlines(True),changed[p].splitlines(True),fromfile='a/'+p,tofile='b/'+p))
new=out/'cpp/apps/mapped_embeddings.hpp'
if new.exists():patch.extend(difflib.unified_diff([],new.read_text().splitlines(True),fromfile='/dev/null',tofile='b/cpp/apps/mapped_embeddings.hpp'))
(out/'mmap-build.patch').write_text(''.join(patch))
