#include "utils.hpp"
#include "chimera/chimera_index.cuh"
#include "chimera/io.hpp"

#include <iostream>
#include "mapped_embeddings.hpp"
#include "mapped_fp16_npy.hpp"
#include <limits>

using namespace Chimera;

int main(int argc, char* argv[]) {
    BuildCliArgs args;
    try {
        args = parse_build_args(argc, argv);
    } catch (const HelpRequested&) {
        print_build_help(argv[0]);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n\n";
        print_build_help(argv[0]);
        return 1;
    }

    auto doc_lens = load_doclens(args.doclens_file);
    chimera_index index;
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
                  << " dimension=" << embeddings.dimension() << '\n';
        index.build_from_f16_ptr(embeddings.data(), embeddings.size(), embeddings.dimension(),
                                 doc_lens, args.n_clusters, args.ex_bits);
    } else {
        MappedEmbeddings embeddings(args.data_file);
        index.build_from_ptr(embeddings.data(), embeddings.size(), embeddings.dimension(),
                             doc_lens, args.n_clusters, args.ex_bits);
    }
    index.save(args.index_dir);
    return 0;
}
