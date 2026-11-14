#include "chimera/spilled_full_codes.hpp"
#include "chimera/build_checkpoint.hpp"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "chimera/chimera_index.cuh"

#define EIGEN_NO_CUDA

#include <omp.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "chimera/config.cuh"
#include "chimera/fp16_input.hpp"
#include "chimera/index_file_io.hpp"
#include "chimera/ivf_pg.hpp"
#include "chimera/quantization.hpp"
#include "rabitqlib/utils/rotator.hpp"

namespace Chimera {

using namespace rabitqlib;

namespace {

using FloatView = FloatInputView;

template <typename T>
void release_vector(std::vector<T>& data) {
    std::vector<T>().swap(data);
}

constexpr std::uint64_t kCentroidSampleSeed = 42;
constexpr std::uint64_t kRotatorSeed = 42;
constexpr size_t kAssignmentBatchSize = 65536;
constexpr RotatorType kRotatorType = RotatorType::FhtKacRotator;

template <typename T>
class DeviceBuffer {
   public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                "cudaMalloc");
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    T* data() { return data_; }

   private:
    T* data_ = nullptr;
};

size_t checked_product(size_t lhs, size_t rhs, const char* description) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::runtime_error(
            std::string("Size overflow while allocating ") + description);
    }
    return lhs * rhs;
}

size_t validate_build_inputs(
    FloatView embeddings,
    size_t dimension,
    const std::vector<int>& doc_lens,
    size_t n_clusters,
    size_t ex_bits) {
    if (dimension < 64 || dimension > PADDED_DIM) {
        throw std::invalid_argument(
            "Embedding dimension must be in [64, " +
            std::to_string(PADDED_DIM) + "]");
    }
    if (embeddings.empty() || embeddings.size() % dimension != 0) {
        throw std::invalid_argument(
            "Embedding storage must contain a non-empty whole number of vectors");
    }

    const size_t num_embeddings = embeddings.size() / dimension;
    if (num_embeddings > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "Embedding count exceeds the int32 index format limit");
    }
    if (n_clusters == 0 || n_clusters > num_embeddings) {
        throw std::invalid_argument(
            "n_clusters must be between 1 and the number of embeddings");
    }
    if (ex_bits < 1 || ex_bits > 7) {
        throw std::invalid_argument(
            "ex_bits must be in [1, 7] so full codes fit in 2-8 bits per dimension");
    }
    if (doc_lens.empty()) {
        throw std::invalid_argument("doc_lens must not be empty");
    }

    size_t token_count = 0;
    for (const int doc_len : doc_lens) {
        if (doc_len <= 0) {
            throw std::invalid_argument("Every document length must be positive");
        }
        const size_t length = static_cast<size_t>(doc_len);
        if (token_count > num_embeddings || length > num_embeddings - token_count) {
            throw std::invalid_argument(
                "Document lengths exceed the number of embeddings");
        }
        token_count += length;
    }
    if (token_count != num_embeddings) {
        throw std::invalid_argument(
            "Document lengths do not sum to the number of embeddings");
    }

    return num_embeddings;
}

void prepare_output_directory(const std::string& index_dir) {
    std::error_code error;
    std::filesystem::create_directories(index_dir, error);
    if (error) {
        throw std::runtime_error(
            "Failed to create index directory " + index_dir + ": " +
            error.message());
    }

    const std::array<const char*, 6> filenames = {
        "doclens.bin",
        "doc_1bit.bin",
        "doc_full.bin",
        "cluster_1bit.bin",
        "ivf.bin",
        "centroids.carga",
    };

    for (const char* filename : filenames) {
        const auto output =
            (std::filesystem::path(index_dir) / filename).string();
        error.clear();
        std::filesystem::remove(output, error);
        if (error) {
            throw std::runtime_error(
                "Failed to remove stale index output " + output + ": " +
                error.message());
        }
    }
}

std::vector<float> sample_centroids(
    FloatView embeddings,
    size_t num_embeddings,
    size_t dimension,
    size_t n_clusters) {
    std::vector<size_t> sampled_ids(n_clusters);
    std::iota(sampled_ids.begin(), sampled_ids.end(), size_t {0});

    std::mt19937_64 generator(kCentroidSampleSeed);
    for (size_t index = n_clusters; index < num_embeddings; ++index) {
        std::uniform_int_distribution<size_t> distribution(0, index);
        const size_t position = distribution(generator);
        if (position < n_clusters) {
            sampled_ids[position] = index;
        }
    }

    std::vector<float> centroids(checked_product(n_clusters, dimension, "centroids"));
    // Preserve reservoir IDs and centroid slots; gather immutable source rows in order.
    std::vector<size_t> read_order(n_clusters);
    std::iota(read_order.begin(), read_order.end(), size_t {0});
    std::sort(read_order.begin(), read_order.end(), [&](size_t a, size_t b) {
        return sampled_ids[a] != sampled_ids[b] ? sampled_ids[a] < sampled_ids[b] : a < b;
    });
    for (size_t cluster : read_order) {
        embeddings.copy_to(
            centroids.data() + cluster * dimension,
            sampled_ids[cluster] * dimension, dimension);
    }
    return centroids;
}

std::vector<uint32_t> assign_embeddings(
    FloatView embeddings,
    size_t num_embeddings,
    size_t dimension,
    const std::vector<float>& centroids,
    size_t n_clusters,
    AssignmentCheckpoint* checkpoint = nullptr,
    size_t stop_after_batches = 0) {
    PG_CAGRA assignment_index(n_clusters, dimension);
    if (checkpoint && checkpoint->has_graph()) {
        std::cerr << "[BUILD_CHECKPOINT] assignment_graph_load_start" << std::endl;
        assignment_index.load(checkpoint->graph_path());
        std::cerr << "[BUILD_CHECKPOINT] assignment_graph_loaded" << std::endl;
    } else {
        if (checkpoint) std::cerr << "[BUILD_CHECKPOINT] assignment_graph_build_start" << std::endl;
        assignment_index.build_index(centroids.data());
        if (checkpoint) {
            auto temporary = checkpoint->reserve_graph_temporary();
            assignment_index.save(temporary.string());
            check_cuda(cudaDeviceSynchronize(), "Completing checkpoint graph serialization");
            checkpoint->commit_graph(temporary);
            std::cerr << "[BUILD_CHECKPOINT] assignment_graph_committed" << std::endl;
        }
    }

    if (checkpoint && (assignment_index.index_cagra->size() != n_clusters ||
                       assignment_index.index_cagra->dim() != dimension ||
                       assignment_index.index_cagra->graph_degree() != 64)) {
        throw std::runtime_error("Checkpoint graph shape/default-degree mismatch");
    }
    const size_t batch_capacity = std::min(num_embeddings, kAssignmentBatchSize);
    DeviceBuffer<float> device_embeddings(
        checked_product(batch_capacity, dimension, "assignment batch"));
    DeviceBuffer<uint32_t> device_labels(batch_capacity);
    DeviceBuffer<float> device_distances(batch_capacity);

    // Only FP16 input allocates an additional FP32 host assignment batch.
    std::vector<float> expanded_batch;
    if (embeddings.is_fp16()) expanded_batch.resize(batch_capacity * dimension);
    std::vector<uint32_t> assignments(num_embeddings);
    if (checkpoint) checkpoint->load_labels(assignments);
    const size_t first = checkpoint ? checkpoint->completed_tokens() : 0;
    size_t new_batches = 0;
    const auto assignment_started = std::chrono::steady_clock::now();
    if (checkpoint) std::cerr << "[BUILD_CHECKPOINT] assignment_start completed_tokens=" << first
                              << " total_tokens=" << num_embeddings << " batch_size=" << batch_capacity << std::endl;
    for (size_t start = first; start < num_embeddings; start += batch_capacity) {
        const size_t count = std::min(batch_capacity, num_embeddings - start);
        const float* source = nullptr;
        if (embeddings.is_fp16()) {
            embeddings.copy_to(expanded_batch.data(), start * dimension, count * dimension);
            source = expanded_batch.data();
        } else {
            source = embeddings.float_data() + start * dimension;
        }
        check_cuda(
            cudaMemcpy(
                device_embeddings.data(),
                source,
                count * dimension * sizeof(float),
                cudaMemcpyHostToDevice),
            "Copying embeddings to the GPU");

        assignment_index.search_centroids(
            device_embeddings.data(),
            count,
            1,
            device_distances.data(),
            device_labels.data(),
            nullptr);

        check_cuda(
            cudaMemcpy(
                assignments.data() + start,
                device_labels.data(),
                count * sizeof(uint32_t),
                cudaMemcpyDeviceToHost),
            "Copying centroid assignments to the host");
        if (checkpoint) {
            checkpoint->append_labels(start, assignments.data() + start, count);
            ++new_batches;
            std::cerr << "[BUILD_CHECKPOINT] assignment_committed completed_tokens=" << start + count
                      << " total_tokens=" << num_embeddings << " new_batches=" << new_batches
                      << " elapsed_s=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - assignment_started).count()
                      << std::endl;
            if (stop_after_batches && new_batches >= stop_after_batches) throw BuildCheckpointStop{};
        }
    }
    return assignments;
}

std::vector<float> rotate_centroids(
    const std::vector<float>& centroids,
    size_t n_clusters,
    size_t dimension,
    const Rotator<float>& rotator) {
    std::vector<float> rotated(
        checked_product(n_clusters, static_cast<size_t>(PADDED_DIM), "rotated centroids"));

#pragma omp parallel for schedule(static)
    for (std::int64_t cluster = 0;
         cluster < static_cast<std::int64_t>(n_clusters);
         ++cluster) {
        rotator.rotate(
            centroids.data() + static_cast<size_t>(cluster) * dimension,
            rotated.data() + static_cast<size_t>(cluster) * PADDED_DIM);
    }
    return rotated;
}

struct EncodedDocuments {
    std::vector<uint64_t> one_bit_codes;
    std::vector<char> full_codes;
    std::unique_ptr<SpilledFullCodes> spilled_codes;
    std::vector<float> one_bit_factors;
    std::vector<float> full_code_factors;
};

EncodedDocuments encode_embeddings(
    FloatView embeddings,
    size_t num_embeddings,
    size_t dimension,
    size_t ex_bits,
    const Rotator<float>& rotator,
    const BuildOptions& build_options) {
    const size_t one_bit_words_per_vector = PADDED_DIM / 64;
    const size_t full_code_bytes_per_vector = PADDED_DIM * (1 + ex_bits) / 8;

    EncodedDocuments encoded {
        std::vector<uint64_t>(checked_product(
            num_embeddings, one_bit_words_per_vector, "one-bit codes")),
        std::vector<char>(build_options.spill_full_codes_path.empty() ? checked_product(
            num_embeddings, full_code_bytes_per_vector, "full codes") : 0),
        nullptr,
        std::vector<float>(num_embeddings),
        std::vector<float>(num_embeddings),
    };

    if (!build_options.spill_full_codes_path.empty()) {
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
#pragma omp parallel
    {
        std::vector<float> rotated(PADDED_DIM);
        std::array<float, PADDED_DIM> expanded_row;

#pragma omp for schedule(static)
        for (std::int64_t row = static_cast<std::int64_t>(first);
             row < static_cast<std::int64_t>(last);
             ++row) {
            const size_t index = static_cast<size_t>(row);
            rotator.rotate(
                embeddings.row_data(index * dimension, dimension, expanded_row.data()),
                rotated.data());
            encode_one_bit(
                rotated.data(),
                PADDED_DIM,
                encoded.one_bit_codes.data() + index * one_bit_words_per_vector,
                &encoded.one_bit_factors[index]);
            encode_full_code(
                rotated.data(),
                PADDED_DIM,
                ex_bits,
                reinterpret_cast<uint8_t*>(full_codes) + index * full_code_bytes_per_vector,
                &encoded.full_code_factors[index]);
        }
    }

        if (encoded.spilled_codes) encoded.spilled_codes->sync_and_drop(
            first * full_code_bytes_per_vector, (last - first) * full_code_bytes_per_vector);
        first = last;
    }

    return encoded;
}

std::vector<int> make_token_doc_ids(
    const std::vector<int>& doc_lens,
    size_t num_embeddings) {
    std::vector<int> token_doc_ids(num_embeddings);
    size_t token = 0;
    for (size_t doc_id = 0; doc_id < doc_lens.size(); ++doc_id) {
        const size_t next = token + static_cast<size_t>(doc_lens[doc_id]);
        std::fill(
            token_doc_ids.begin() + static_cast<std::ptrdiff_t>(token),
            token_doc_ids.begin() + static_cast<std::ptrdiff_t>(next),
            static_cast<int>(doc_id));
        token = next;
    }
    return token_doc_ids;
}

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

std::ofstream open_binary_output(const std::string& path) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    return output;
}

void finish_output(std::ofstream& output, const std::string& path) {
    output.flush();
    if (!output) {
        throw std::runtime_error("Failed to write output file: " + path);
    }
}

void write_doc_lens(
    const std::vector<int>& doc_lens,
    const std::string& path) {
    auto output = open_binary_output(path);
    const int count = static_cast<int>(doc_lens.size());
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    output.write(
        reinterpret_cast<const char*>(doc_lens.data()),
        static_cast<std::streamsize>(doc_lens.size() * sizeof(int)));
    finish_output(output, path);
}

}  // namespace

void chimera_index::build(
    const std::vector<float>& embeddings,
    size_t dimension,
    const std::vector<int>& doc_lens,
    size_t n_clusters,
    size_t ex_bits,
    const SearchOptions& options) {
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
    build_from_view(FloatInputView(embedding_data, embedding_values), dimension,
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

void chimera_index::build(
    const std::vector<float>& embeddings,
    size_t dimension,
    const std::vector<int>& doc_lens,
    size_t n_clusters,
    size_t ex_bits,
    const SearchOptions& options,
    const BuildOptions& build_options) {
    build_from_ptr(embeddings.data(), embeddings.size(), dimension, doc_lens,
                   n_clusters, ex_bits, options, build_options);
}

void chimera_index::build_from_ptr(
    const float* embedding_data,
    size_t embedding_values,
    size_t dimension,
    const std::vector<int>& doc_lens,
    size_t n_clusters,
    size_t ex_bits,
    const SearchOptions& options,
    const BuildOptions& build_options) {
    build_from_view(FloatInputView(embedding_data, embedding_values), dimension,
                    doc_lens, n_clusters, ex_bits, options, build_options);
}

void chimera_index::build_from_f16_ptr(
    const uint16_t* embedding_data,
    size_t embedding_values,
    size_t dimension,
    const std::vector<int>& doc_lens,
    size_t n_clusters,
    size_t ex_bits,
    const SearchOptions& options,
    const BuildOptions& build_options) {
    build_from_view(FloatInputView(embedding_data, embedding_values), dimension,
                    doc_lens, n_clusters, ex_bits, options, build_options);
}

void chimera_index::build_from_view(
    const FloatInputView& embeddings,
    size_t dimension,
    const std::vector<int>& doc_lens,
    size_t n_clusters,
    size_t ex_bits,
    const SearchOptions& options,
    const BuildOptions& build_options) {
    validate_search_options(options);
    if (build_options.checkpoint_dir.empty() != build_options.checkpoint_identity_sha256.empty()) {
        throw std::invalid_argument("Checkpoint directory and explicit identity SHA must be supplied together");
    }
    if (build_options.checkpoint_dir.empty() &&
        (build_options.resume_checkpoint || build_options.stop_after_assignment_batches)) {
        throw std::invalid_argument("Checkpoint stop/resume requires a checkpoint directory");
    }
    const size_t num_embeddings = validate_build_inputs(
        embeddings,
        dimension,
        doc_lens,
        n_clusters,
        ex_bits);

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "Querying CUDA devices");
    if (device_count == 0) {
        throw std::runtime_error("No CUDA device is available");
    }
    check_cuda(cudaSetDevice(0), "Selecting CUDA device 0");

    begin_initialization(options);
    n = num_embeddings;
    d = dimension;
    this->n_clusters = n_clusters;
    this->ex_bits = ex_bits;
    doc_lens_ = doc_lens;

    std::unique_ptr<AssignmentCheckpoint> checkpoint;
    if (!build_options.checkpoint_dir.empty()) {
        AssignmentCheckpointConfig config;
        config.identity_sha256 = build_options.checkpoint_identity_sha256;
        config.doclens_sha256 = checkpoint_sha256(doc_lens.data(), doc_lens.size() * sizeof(int));
        config.tokens = num_embeddings; config.dimension = dimension;
        config.clusters = n_clusters; config.ex_bits = ex_bits;
        config.sample_seed = kCentroidSampleSeed; config.rotator_seed = kRotatorSeed;
        checkpoint = std::make_unique<AssignmentCheckpoint>(build_options.checkpoint_dir, config, build_options.resume_checkpoint);
    } else if (build_options.resume_checkpoint || build_options.stop_after_assignment_batches) {
        throw std::invalid_argument("Checkpoint stop/resume requires a checkpoint directory");
    }
    std::vector<float> centroids;
    if (checkpoint && checkpoint->has_centroids()) {
        centroids = checkpoint->load_centroids();
        std::cerr << "[BUILD_CHECKPOINT] sampled_centroids_loaded" << std::endl;
    } else {
        if (checkpoint) std::cerr << "[BUILD_CHECKPOINT] centroid_sample_start" << std::endl;
        centroids = sample_centroids(embeddings, num_embeddings, dimension, n_clusters);
        if (checkpoint) {
            checkpoint->save_centroids(centroids);
            std::cerr << "[BUILD_CHECKPOINT] sampled_centroids_committed" << std::endl;
        }
    }
    auto assignments = assign_embeddings(embeddings, num_embeddings, dimension, centroids,
                                         n_clusters, checkpoint.get(), build_options.stop_after_assignment_batches);
    if (checkpoint) std::cerr << "[BUILD_CHECKPOINT] assignment_complete" << std::endl;
    auto rotator = std::unique_ptr<Rotator<float>>(
        choose_rotator_with_seed<float>(
            dimension,
            kRotatorType,
            PADDED_DIM,
            kRotatorSeed));

    auto rotated_centroids = rotate_centroids(
        centroids,
        n_clusters,
        dimension,
        *rotator);
    auto centroid_graph = std::make_unique<PG_CAGRA>(n_clusters, PADDED_DIM);
    if (checkpoint) std::cerr << "[BUILD_CHECKPOINT] rotated_graph_build_start" << std::endl;
    centroid_graph->build_index(rotated_centroids.data());
    if (checkpoint) std::cerr << "[BUILD_CHECKPOINT] rotated_graph_complete" << std::endl;

    ivf = new IVF_PG(n_clusters, dimension);
    delete ivf->pg_index;
    ivf->pg_index = centroid_graph.release();
    ivf->build_from_assignments(assignments.data(), num_embeddings);
    release_vector(assignments);
    release_vector(centroids);
    release_vector(rotated_centroids);

    if (checkpoint) std::cerr << "[BUILD_CHECKPOINT] encoding_start" << std::endl;
    auto encoded = encode_embeddings(
        embeddings,
        num_embeddings,
        dimension,
        ex_bits,
        *rotator, build_options);
    auto token_doc_ids = make_token_doc_ids(doc_lens, num_embeddings);
    auto clustered = reorder_by_cluster(encoded, *ivf, token_doc_ids);
    release_vector(token_doc_ids);

    rotator_ = rotator.release();
    one_bit_codes_ = std::move(encoded.one_bit_codes);
    one_bit_factors_ = std::move(encoded.one_bit_factors);
    full_bit_factors_ = std::move(encoded.full_code_factors);
    full_bit_codes_ = std::move(encoded.full_codes);
    spilled_full_bit_codes_ = std::move(encoded.spilled_codes);
    clustered_data_ = std::move(clustered);

    has_index_ = true;
    if (checkpoint) std::cerr << "[BUILD_CHECKPOINT] index_ready_for_save" << std::endl;
}

void chimera_index::save(const std::string& index_dir) const {
    // Reject before prepare_output_directory can remove existing payloads.
    reject_readonly_index_save(low_memory_load_);
    if (!has_index_) {
        throw std::runtime_error("Cannot save an empty Chimera index");
    }

    if (spilled_full_bit_codes_) spilled_full_bit_codes_->preflight_save_directory(index_dir);
    prepare_output_directory(index_dir);
    const std::filesystem::path output_dir(index_dir);
    const auto doclens_path = (output_dir / "doclens.bin").string();
    const auto doc_one_bit_path = (output_dir / "doc_1bit.bin").string();
    const auto doc_full_path = (output_dir / "doc_full.bin").string();
    const auto cluster_one_bit_path =
        (output_dir / "cluster_1bit.bin").string();
    const auto ivf_path = (output_dir / "ivf.bin").string();
    const auto centroids_path = (output_dir / "centroids.carga").string();

    write_doc_lens(doc_lens_, doclens_path);
    ivf->save(ivf_path, centroids_path);

    const IndexHeader header {
        n,
        d,
        this->n_clusters,
        this->ex_bits,
        PADDED_DIM,
    };
    auto one_bit_output = open_binary_output(doc_one_bit_path);
    one_bit_output.write(
        reinterpret_cast<const char*>(&header), sizeof(header));
    rotator_->save(one_bit_output);
    one_bit_output.write(
        reinterpret_cast<const char*>(one_bit_codes_.data()),
        static_cast<std::streamsize>(
            one_bit_codes_.size() * sizeof(uint64_t)));
    one_bit_output.write(
        reinterpret_cast<const char*>(one_bit_factors_.data()),
        static_cast<std::streamsize>(
            one_bit_factors_.size() * sizeof(float)));
    one_bit_output.write(
        reinterpret_cast<const char*>(full_bit_factors_.data()),
        static_cast<std::streamsize>(
            full_bit_factors_.size() * sizeof(float)));
    finish_output(one_bit_output, doc_one_bit_path);

    if (spilled_full_bit_codes_) {
        spilled_full_bit_codes_->publish_hardlink(doc_full_path);
    } else {
    auto full_code_output = open_binary_output(doc_full_path);
    full_code_output.write(
        full_bit_codes_.data(),
        static_cast<std::streamsize>(full_bit_codes_.size()));
    finish_output(full_code_output, doc_full_path);
    }

    auto clustered_output = open_binary_output(cluster_one_bit_path);
    clustered_output.write(
        clustered_data_.data(),
        static_cast<std::streamsize>(clustered_data_.size()));
    finish_output(clustered_output, cluster_one_bit_path);
}

}  // namespace Chimera
