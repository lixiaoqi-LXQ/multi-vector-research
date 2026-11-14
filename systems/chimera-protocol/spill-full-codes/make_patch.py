from pathlib import Path
import difflib
root=Path(__file__).resolve().parent
base=root/'base'; out=root/'modified'
def edit(name, fn):
 p=base/name; q=out/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(fn(p.read_text()))
def header(s):
 s=s.replace('class ReadOnlyMappedFile;', 'class ReadOnlyMappedFile;\nclass SpilledFullCodes;\n\nstruct BuildOptions {\n    std::string spill_full_codes_path;\n    size_t spill_flush_bytes = 64ull * 1024ull * 1024ull;\n};')
 # Preserve existing overloads and add overloads explicitly accepting build options.
 start=s.index('    void build('); end=s.index('    std::vector<size_t> search(',start)
 block=s[start:end]
 import re
 extra=re.sub(r'const SearchOptions& options = \{\}\);', 'const SearchOptions& options,\n        const BuildOptions& build_options);', block)
 s=s[:end]+extra+s[end:]
 s=s.replace('const SearchOptions& options);\n\n    struct QueryState;', 'const SearchOptions& options,\n        const BuildOptions& build_options = {});\n    const char* full_bit_codes_data() const;\n\n    struct QueryState;')
 return s.replace('    std::vector<char> full_bit_codes_;','    std::vector<char> full_bit_codes_;\n    std::unique_ptr<SpilledFullCodes> spilled_full_bit_codes_;')
edit('cpp/include/chimera/chimera_index.cuh',header)
def build(s):
 s='#include "chimera/spilled_full_codes.hpp"\n'+s
 s=s.replace('    std::vector<char> full_codes;','    std::vector<char> full_codes;\n    std::unique_ptr<SpilledFullCodes> spilled_codes;')
 s=s.replace('const Rotator<float>& rotator) {\n    const size_t one_bit_words', 'const Rotator<float>& rotator,\n    const BuildOptions& build_options) {\n    const size_t one_bit_words')
 s=s.replace('std::vector<char>(checked_product(\n            num_embeddings, full_code_bytes_per_vector, "full codes")),','std::vector<char>(build_options.spill_full_codes_path.empty() ? checked_product(\n            num_embeddings, full_code_bytes_per_vector, "full codes") : 0),\n        nullptr,')
 a=s.index('#pragma omp parallel\n',s.index('EncodedDocuments encode_embeddings(')); b=s.index('\n    return encoded;',a)
 body=s[a:b]
 body=body.replace('row = 0;', 'row = static_cast<std::int64_t>(first);').replace('static_cast<std::int64_t>(num_embeddings);','static_cast<std::int64_t>(last);').replace('encoded.full_codes.data()', 'full_codes')
 pre='''    if (!build_options.spill_full_codes_path.empty()) {
        encoded.spilled_codes = std::make_unique<SpilledFullCodes>(
            build_options.spill_full_codes_path,
            checked_product(num_embeddings, full_code_bytes_per_vector, "full codes"),
            build_options.spill_flush_bytes);
    }
    char* full_codes = encoded.spilled_codes ? encoded.spilled_codes->data() : encoded.full_codes.data();
    const size_t rows_per_flush = encoded.spilled_codes
        ? encoded.spilled_codes->rows_per_flush(full_code_bytes_per_vector) : num_embeddings;
    for (size_t first = 0; first < num_embeddings;) {
        const size_t last = first + std::min(rows_per_flush, num_embeddings - first);
'''
 post='''
        if (encoded.spilled_codes) encoded.spilled_codes->sync_and_drop(
            first * full_code_bytes_per_vector, (last - first) * full_code_bytes_per_vector);
        first = last;
    }
'''
 s=s[:a]+pre+body+post+s[b:]
 # New overloads reuse existing bodies; originals delegate with default BuildOptions.
 start=s.index('void chimera_index::build('); end=s.index('void chimera_index::build_from_view(',start)
 block=s[start:end]
 extra=block.replace('const SearchOptions& options) {','const SearchOptions& options,\n    const BuildOptions& build_options) {').replace('n_clusters, ex_bits, options);','n_clusters, ex_bits, options, build_options);')
 s=s[:end]+extra+s[end:]
 s=s.replace('size_t ex_bits,\n    const SearchOptions& options) {\n    validate_search_options', 'size_t ex_bits,\n    const SearchOptions& options,\n    const BuildOptions& build_options) {\n    validate_search_options')
 s=s.replace('        *rotator);\n    auto token_doc_ids', '        *rotator, build_options);\n    auto token_doc_ids')
 s=s.replace('    full_bit_codes_ = std::move(encoded.full_codes);','    full_bit_codes_ = std::move(encoded.full_codes);\n    spilled_full_bit_codes_ = std::move(encoded.spilled_codes);')
 s=s.replace('    prepare_output_directory(index_dir);','    if (spilled_full_bit_codes_) spilled_full_bit_codes_->preflight_save_directory(index_dir);\n    prepare_output_directory(index_dir);')
 a=s.index('    auto full_code_output = open_binary_output(doc_full_path);');b=s.index('\n\n    auto clustered_output',a)
 old=s[a:b]
 s=s[:a]+'    if (spilled_full_bit_codes_) {\n        spilled_full_bit_codes_->publish_hardlink(doc_full_path);\n    } else {\n'+old+'\n    }'+s[b:]
 return s
edit('cpp/src/build_index.cu',build)
def utils(s):
 s='#include "chimera/spilled_full_codes.hpp"\n'+s
 s=s.replace('    mapped_full_bit_data_ = nullptr;', '    spilled_full_bit_codes_.reset();\n    mapped_full_bit_data_ = nullptr;')
 s+='\nnamespace Chimera {\nconst char* chimera_index::full_bit_codes_data() const {\n    if (spilled_full_bit_codes_) return spilled_full_bit_codes_->data();\n    return mapped_full_bit_data_ ? mapped_full_bit_data_ : full_bit_codes_.data();\n}\n}\n'
 return s
edit('cpp/src/utils.cu',utils)
edit('cpp/src/cpu_scoring.cpp',lambda s:s.replace('mapped_full_bit_data_ != nullptr\n        ? mapped_full_bit_data_ : full_bit_codes_.data()', 'full_bit_codes_data()'))
# unique_ptr's destructor requires the complete type wherever constructor/destructor are defined.
edit('cpp/src/chimera_index.cu',lambda s:'#include "chimera/spilled_full_codes.hpp"\n'+s)
edit('cpp/apps/utils.hpp',lambda s:s.replace('    size_t ex_bits = 0;', '    size_t ex_bits = 0;\n    std::string spill_full_codes_path;'))
edit('cpp/apps/utils.cpp',lambda s:s.replace('        } else if (arg == "--fp16-npy") {','        } else if (arg == "--spill-full-codes") {\n            args.spill_full_codes_path = require_value(argc, argv, i, arg);\n        } else if (arg == "--fp16-npy") {').replace('        << "  --fp16-npy', '        << "  --spill-full-codes <path>  Exclusive temporary file; same filesystem as index.\\n"\n        << "  --fp16-npy'))
edit('cpp/apps/build.cpp',lambda s:s.replace('    chimera_index index;', '    chimera_index index;\n    Chimera::BuildOptions build_options;\n    build_options.spill_full_codes_path = args.spill_full_codes_path;').replace('doc_lens, args.n_clusters, args.ex_bits);','doc_lens, args.n_clusters, args.ex_bits, {}, build_options);'))
patch=[]
for q in sorted(out.rglob('*')):
 if not q.is_file():continue
 name=str(q.relative_to(out));p=base/name
 patch.extend(difflib.unified_diff(p.read_text().splitlines(True) if p.exists() else [],q.read_text().splitlines(True),fromfile='a/'+name if p.exists() else '/dev/null',tofile='b/'+name))
(root/'spill-full-codes.patch').write_text(''.join(patch))
