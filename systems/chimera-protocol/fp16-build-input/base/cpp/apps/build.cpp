#include "utils.hpp"
#include "chimera/chimera_index.cuh"
#include "chimera/io.hpp"

#include <iostream>
#include "mapped_embeddings.hpp"

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
    MappedEmbeddings embeddings(args.data_file);
    const size_t dimension = embeddings.dimension();
    chimera_index index;
    index.build_from_ptr(
        embeddings.data(),
        embeddings.size(),
        dimension,
        doc_lens,
        args.n_clusters,
        args.ex_bits);
    index.save(args.index_dir);
    return 0;
}
