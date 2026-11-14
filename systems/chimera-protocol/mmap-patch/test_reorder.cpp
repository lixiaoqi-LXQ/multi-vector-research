#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>
#define PADDED_DIM 128
size_t checked_product(size_t a,size_t b,const char*){return a*b;}
struct IVF_PG {std::vector<uint32_t> inv_list;};
struct EncodedDocuments {
    std::vector<uint64_t> one_bit_codes;
    std::vector<uint8_t> full_codes;
    std::vector<float> one_bit_factors;
    std::vector<float> full_code_factors;
};

namespace old {
struct ClusteredDocuments {
    std::vector<char> one_bit_codes;
    std::vector<float> one_bit_factors;
    std::vector<int> doc_ids;
    std::vector<uint32_t> token_to_cluster_position;
};

ClusteredDocuments reorder_by_cluster(
    const EncodedDocuments& encoded,
    const IVF_PG& ivf,
    const std::vector<int>& token_doc_ids) {
    const size_t num_embeddings = ivf.inv_list.size();
    const size_t code_bytes_per_vector = PADDED_DIM / 8;
    ClusteredDocuments clustered {
        std::vector<char>(checked_product(
            num_embeddings, code_bytes_per_vector, "clustered one-bit codes")),
        std::vector<float>(num_embeddings),
        std::vector<int>(num_embeddings),
        std::vector<uint32_t>(num_embeddings),
    };

    const auto* source_codes =
        reinterpret_cast<const char*>(encoded.one_bit_codes.data());
#pragma omp parallel for schedule(static)
    for (std::int64_t position = 0;
         position < static_cast<std::int64_t>(num_embeddings);
         ++position) {
        const size_t cluster_position = static_cast<size_t>(position);
        const uint32_t token_id = ivf.inv_list[cluster_position];
        std::memcpy(
            clustered.one_bit_codes.data() +
                cluster_position * code_bytes_per_vector,
            source_codes + static_cast<size_t>(token_id) * code_bytes_per_vector,
            code_bytes_per_vector);
        clustered.one_bit_factors[cluster_position] =
            encoded.one_bit_factors[token_id];
        clustered.doc_ids[cluster_position] = token_doc_ids[token_id];
        clustered.token_to_cluster_position[token_id] =
            static_cast<uint32_t>(cluster_position);
    }

    return clustered;
}

}
namespace newer {
// Fill the final serialized layout directly, avoiding a second 28*N-byte copy.
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

}
int main(){
 EncodedDocuments e; const size_t n=123;
 e.one_bit_codes.resize(n*2);e.one_bit_factors.resize(n);
 std::vector<int> docs(n); IVF_PG ivf;ivf.inv_list.resize(n);
 for(size_t i=0;i<n;++i){e.one_bit_factors[i]=float(i)*.125f;docs[i]=int(i/3);ivf.inv_list[i]=uint32_t((i*7)%n);e.one_bit_codes[2*i]=i*17;e.one_bit_codes[2*i+1]=~(i*17);}
 auto a=old::reorder_by_cluster(e,ivf,docs);auto b=newer::reorder_by_cluster(e,ivf,docs);
 size_t offset=0;assert(b.size()==n*28);
 assert(std::memcmp(b.data(),a.one_bit_codes.data(),n*16)==0);offset+=n*16;
 assert(std::memcmp(b.data()+offset,a.one_bit_factors.data(),n*4)==0);offset+=n*4;
 assert(std::memcmp(b.data()+offset,a.doc_ids.data(),n*4)==0);offset+=n*4;
 assert(std::memcmp(b.data()+offset,a.token_to_cluster_position.data(),n*4)==0);
}
