from pathlib import Path
import difflib
r=Path(__file__).parent;paths=[str(p.relative_to(r/'base')) for p in (r/'base').rglob('*') if p.is_file()]
old={p:(r/'base'/p).read_text() for p in paths};new=dict(old)
p='CMakeLists.txt';s=new[p].replace('    cpp/src/io.cpp\n','    cpp/src/io.cpp\n    cpp/src/fp16_input.cpp\n',1);new[p]=s
p='cpp/include/chimera/chimera_index.cuh';s=new[p].replace('struct IVF_PG;','struct IVF_PG;\nclass FloatInputView;',1)
needle='''    std::vector<size_t> search(
''';pos=s.index(needle);s=s[:pos]+'''    // IEEE FP16 bit patterns. Expanded exactly to FP32 in bounded CPU buffers.
    void build_from_f16_ptr(
        const uint16_t* embeddings,
        size_t embedding_values,
        size_t dimension,
        const std::vector<int>& doc_lens,
        size_t n_clusters,
        size_t ex_bits,
        const SearchOptions& options = {});

'''+s[pos:]
needle='''  private:
''';replacement=needle+'''    void build_from_view(
        const FloatInputView& embeddings,
        size_t dimension,
        const std::vector<int>& doc_lens,
        size_t n_clusters,
        size_t ex_bits,
        const SearchOptions& options);

''';assert needle in s;s=s.replace(needle,replacement,1);new[p]=s
p='cpp/src/build_index.cu';s=new[p].replace('#include "chimera/config.cuh"','#include "chimera/config.cuh"\n#include "chimera/fp16_input.hpp"',1)
start=s.index('// Minimal non-owning view shared');end=s.index('template <typename T>',start)
s=s[:start]+'using FloatView = FloatInputView;\n\n'+s[end:]
needle='''        std::memcpy(
            centroids.data() + cluster * dimension,
            embeddings.data() + sampled_ids[cluster] * dimension,
            dimension * sizeof(float));'''
replacement='''        embeddings.copy_to(
            centroids.data() + cluster * dimension,
            sampled_ids[cluster] * dimension, dimension);''';assert needle in s;s=s.replace(needle,replacement)
needle='''    std::vector<uint32_t> assignments(num_embeddings);
    for (size_t start = 0;'''
replacement='''    // Only FP16 input allocates an additional FP32 host assignment batch.
    std::vector<float> expanded_batch;
    if (embeddings.is_fp16()) expanded_batch.resize(batch_capacity * dimension);
    std::vector<uint32_t> assignments(num_embeddings);
    for (size_t start = 0;''';assert needle in s;s=s.replace(needle,replacement)
needle='''        const size_t count = std::min(batch_capacity, num_embeddings - start);
        check_cuda('''
replacement='''        const size_t count = std::min(batch_capacity, num_embeddings - start);
        const float* source = nullptr;
        if (embeddings.is_fp16()) {
            embeddings.copy_to(expanded_batch.data(), start * dimension, count * dimension);
            source = expanded_batch.data();
        } else {
            source = embeddings.float_data() + start * dimension;
        }
        check_cuda(''';assert needle in s;s=s.replace(needle,replacement,1)
s=s.replace('''                embeddings.data() + start * dimension,
''','''                source,
''',1)
needle='''        std::vector<float> rotated(PADDED_DIM);

#pragma omp for'''
replacement='''        std::vector<float> rotated(PADDED_DIM);
        std::array<float, PADDED_DIM> expanded_row;

#pragma omp for''';assert needle in s;s=s.replace(needle,replacement,1)
s=s.replace('''                embeddings.data() + index * dimension,
''','''                embeddings.row_data(index * dimension, dimension, expanded_row.data()),
''',1)
needle='''    validate_search_options(options);
    const FloatView embeddings(embedding_data, embedding_values);
    const size_t num_embeddings = validate_build_inputs('''
replacement='''    build_from_view(FloatInputView(embedding_data, embedding_values), dimension,
                    doc_lens, n_clusters, ex_bits, options);
}

void chimera_index::build_from_f16_ptr(
    const uint16_t* embedding_data,
    size_t embedding_values,
    size_t dimension,
    const std::vector<int>& doc_lens,
    size_t n_clusters,
    size_t ex_bits,
    const SearchOptions& options) {
    build_from_view(FloatInputView(embedding_data, embedding_values), dimension,
                    doc_lens, n_clusters, ex_bits, options);
}

void chimera_index::build_from_view(
    const FloatInputView& embeddings,
    size_t dimension,
    const std::vector<int>& doc_lens,
    size_t n_clusters,
    size_t ex_bits,
    const SearchOptions& options) {
    validate_search_options(options);
    const size_t num_embeddings = validate_build_inputs(''';assert needle in s;s=s.replace(needle,replacement)
assert 'embeddings.data()' in s # retained only original vector entry point
import re
assert len(re.findall(r'(?<!\w)embeddings\.data\(\)',s))==1
new[p]=s
p='cpp/apps/utils.hpp';s=new[p];needle='''    std::string data_file;
''';assert needle in s;s=s.replace(needle,needle+'''    std::string fp16_npy_file;
''',1);new[p]=s
p='cpp/apps/utils.cpp';s=new[p]
needle='''        } else if (arg == "--data") {
            args.data_file = require_value(argc, argv, i, arg);'''
replacement=needle+'''
        } else if (arg == "--fp16-npy") {
            args.fp16_npy_file = require_value(argc, argv, i, arg);''';assert needle in s;s=s.replace(needle,replacement,1)
needle='''    if (args.index_dir.empty() || args.doclens_file.empty() ||
        args.data_file.empty() || args.n_clusters == 0 || args.ex_bits == 0) {'''
replacement='''    if (args.data_file.empty() == args.fp16_npy_file.empty()) {
        throw std::runtime_error("Provide exactly one of --data (raw FP32) or --fp16-npy");
    }
    if (args.index_dir.empty() || args.doclens_file.empty() ||
        args.n_clusters == 0 || args.ex_bits == 0) {''';assert needle in s;s=s.replace(needle,replacement)
s=s.replace('''        << " --index <index_dir> --doclens <doclens> --data <data>"''','''        << " --index <index_dir> --doclens <doclens> (--data <FP32-bin> | --fp16-npy <NPY>)"''',1)
s=s.replace('''        << "  --data           Input token-embedding file.\\n"''','''        << "  --data           Original raw FP32 token-embedding file.\\n"
        << "  --fp16-npy       Read-only C-order NPY <u2 raw FP16 bits or <f2 matrix.\\n"''',1);new[p]=s
p='cpp/apps/build.cpp';s=new[p].replace('#include "mapped_embeddings.hpp"','#include "mapped_embeddings.hpp"\n#include "mapped_fp16_npy.hpp"\n#include <limits>')
start=s.index('    MappedEmbeddings embeddings(args.data_file);');end=s.index('    index.save(args.index_dir);',start)
s=s[:start]+'''    chimera_index index;
    if (!args.fp16_npy_file.empty()) {
        if (doc_lens.empty()) throw std::invalid_argument("Document lengths must not be empty");
        size_t tokens = 0;
        for (int length : doc_lens) {
            if (length <= 0 || static_cast<size_t>(length) > std::numeric_limits<size_t>::max() - tokens) {
                throw std::invalid_argument("Invalid or overflowing document lengths");
            }
            tokens += static_cast<size_t>(length);
        }
        MappedFp16Npy embeddings(args.fp16_npy_file, tokens);
        std::cout << "[BUILD] input=FP16-NPY values=" << embeddings.size()
                  << " dimension=" << embeddings.dimension() << '\\n';
        index.build_from_f16_ptr(embeddings.data(), embeddings.size(), embeddings.dimension(),
                                 doc_lens, args.n_clusters, args.ex_bits);
    } else {
        MappedEmbeddings embeddings(args.data_file);
        index.build_from_ptr(embeddings.data(), embeddings.size(), embeddings.dimension(),
                             doc_lens, args.n_clusters, args.ex_bits);
    }
'''+s[end:];new[p]=s
for p,s in new.items():
 q=r/'modified'/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(s)
patch=[]
for p in sorted(paths):patch.extend(difflib.unified_diff(old[p].splitlines(True),new[p].splitlines(True),fromfile='a/'+p,tofile='b/'+p))
for p in ['cpp/include/chimera/fp16_input.hpp','cpp/src/fp16_input.cpp','cpp/apps/mapped_fp16_npy.hpp']:
 patch.extend(difflib.unified_diff([],(r/'modified'/p).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+p))
(r/'fp16-input.patch').write_text(''.join(patch))
