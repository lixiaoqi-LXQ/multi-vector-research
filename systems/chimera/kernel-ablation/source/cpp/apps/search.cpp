#include "utils.hpp"
#include "chimera/config.cuh"
#include "chimera/profiling.hpp"
#include <cuda_profiler_api.h>
#include <omp.h>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/resource.h>
#include <unistd.h>
#include <sstream>
#include "chimera/io.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>

using namespace Chimera;

namespace {

void report_recalls(
    const std::vector<std::vector<size_t>>& eval_ground_truth,
    const std::vector<std::vector<size_t>>& results,
    int result_k) {
    std::cout << "[SEARCH] recall@" << result_k << '='
              << compute_recall(eval_ground_truth, results, result_k)
              << '\n';
    if (result_k == 100) {
        std::cout << "[SEARCH] recall@10="
                  << compute_recall(eval_ground_truth, results, 10)
                  << '\n';
    }
}

void report_latency(
    const std::vector<double>& query_latencies_ms) {
    const auto latency = summarize_query_latencies_ms(query_latencies_ms);
    std::cout << std::fixed << std::setprecision(3)
              << "[SEARCH] latency_ms"
              << " mean=" << latency.mean_ms
              << " min=" << latency.min_ms
              << " p50=" << latency.p50_ms
              << " p90=" << latency.p90_ms
              << " p95=" << latency.p95_ms
              << " p99=" << latency.p99_ms
              << " max=" << latency.max_ms
              << " stddev=" << latency.stddev_ms
              << '\n';
}


void report_thread_affinity() {
#pragma omp parallel
    {
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        const int status = sched_getaffinity(0, sizeof(allowed), &allowed);
        std::ostringstream line;
        line << "[THREAD] omp_thread=" << omp_get_thread_num()
             << " linux_tid=" << static_cast<long>(syscall(SYS_gettid))
             << " cpu=" << sched_getcpu() << " allowed=";
        bool first = true;
        if (status == 0) for (int core = 0; core < CPU_SETSIZE; ++core) {
            if (CPU_ISSET(core, &allowed)) {
                if (!first) line << ',';
                line << core;
                first = false;
            }
        }
        if (status != 0) line << "ERROR";
#pragma omp critical
        std::cout << line.str() << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    SearchCliArgs args;

    try {
        args = parse_search_args(argc, argv);
    } catch (const HelpRequested&) {
        print_search_help(argv[0]);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n\n";
        print_search_help(argv[0]);
        return 1;
    }

    size_t num_q = 0;
    size_t d = 0;
    size_t query_length = 0;
    std::vector<float> queries =
        load_query(query_length, num_q, d, args.query_file);
    auto ground_truth = read_gt_tsv(num_q, 1000, args.gt_file);

    if (query_length != Q_DOCLEN) {
        std::cerr << "Query length " << query_length
                  << " does not match compiled Q_DOCLEN=" << Q_DOCLEN << '\n';
        return 1;
    }

    const int warmup_queries =
        std::min<int>(args.warmup, static_cast<int>(num_q));
    const int run_queries =
        (args.nq < 0)
            ? static_cast<int>(num_q)
            : std::min<int>(args.nq, static_cast<int>(num_q));
    if (run_queries == 0) {
        std::cerr << "No evaluation queries selected.\n";
        return 1;
    }

    if (args.profile_range && run_queries != 32 && run_queries != 100) {
        throw std::invalid_argument("--profile-range requires --nq 32 or --nq 100");
    }
    std::vector<std::vector<size_t>> eval_ground_truth(
        ground_truth.begin(), ground_truth.begin() + run_queries);

    chimera_index index;
    index.load(args.index_dir, args.options, args.load_options);
    const char* kernel_mode = args.options.use_lut
        ? (args.options.use_aggregation ? "full" : "no-aggregation")
        : (args.options.use_aggregation ? "no-lut" : "neither");
    std::cout << "[VARIANT] algorithm_variant=full kernel_mode=" << kernel_mode
              << " use_lut=" << args.options.use_lut
              << " use_aggregation=" << args.options.use_aggregation << '\n';
    std::cout << "[LOAD] low_memory=" << args.load_options.low_memory
              << " prefault_full_codes=" << args.load_options.prefault_full_codes
              << " transfer_chunk_bytes=" << args.load_options.transfer_chunk_bytes << '\n';
    index.set_capture_scores(!args.output_file.empty());

    for (int i = 0; i < warmup_queries; ++i) {
        index.search(&queries[i * Q_DOCLEN * d], Q_DOCLEN * d, args.k);
    }

    if (args.thread_info) report_thread_affinity();
    std::vector<std::vector<size_t>> results(run_queries);
    std::vector<double> query_latencies_ms(run_queries);
    std::vector<std::vector<float>> scores(args.output_file.empty() ? 0 : run_queries);
    {
        size_t free_bytes = 0, total_bytes = 0;
        check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "Querying CUDA memory view");
        std::cout << "[MEMORY] phase=after_warmup cudaMemGetInfo_free_bytes=" << free_bytes
                  << " cudaMemGetInfo_total_bytes=" << total_bytes << '\n';
    }
    if (args.profile_range) {
        index.set_profiling_enabled(true);
        check_cuda(cudaProfilerStart(), "Starting diagnostic capture");
    }
    rusage usage_before{}, usage_after{};
    if (getrusage(RUSAGE_SELF, &usage_before) != 0)
        throw std::runtime_error("getrusage before query loop failed");
    const auto loop_start = std::chrono::steady_clock::now();
    for (int i = 0; i < run_queries; ++i) {
        ProfileRange query_range(args.profile_range, "Chimera/query_local_row", i);
        const auto query_start = std::chrono::high_resolution_clock::now();
        results[i] =
            index.search(&queries[i * Q_DOCLEN * d], Q_DOCLEN * d, args.k);
        const auto query_end = std::chrono::high_resolution_clock::now();
        query_latencies_ms[i] =
            std::chrono::duration<double, std::milli>(
                query_end - query_start).count();
        if (!args.output_file.empty()) scores[i] = index.last_result_scores();
    }

    const double loop_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loop_start).count();
    if (getrusage(RUSAGE_SELF, &usage_after) != 0)
        throw std::runtime_error("getrusage after query loop failed");
    if (args.profile_range) {
        check_cuda(cudaProfilerStop(), "Stopping diagnostic capture");
        index.set_profiling_enabled(false);
        std::cout << "[PROFILE] metric_scope=diagnostic_profile_not_QPS\n";
    }
    if (!args.output_file.empty()) {
        std::ofstream output(args.output_file);
        if (!output) throw std::runtime_error("Could not open result output: " + args.output_file);
        output << "query_id\trank\tdocument_id\tquantized_score\tlatency_ms\n" << std::setprecision(10);
        for (int i = 0; i < run_queries; ++i)
            for (size_t j = 0; j < results[i].size(); ++j)
                output << i << '\t' << j+1 << '\t' << results[i][j] << '\t'
                       << scores[i][j] << '\t' << query_latencies_ms[i] << '\n';
        if (!output) throw std::runtime_error("Failed writing result output");
    }
    {
        size_t free_bytes = 0, total_bytes = 0;
        check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "Querying CUDA memory view");
        std::cout << "[MEMORY] phase=after_queries cudaMemGetInfo_free_bytes=" << free_bytes
                  << " cudaMemGetInfo_total_bytes=" << total_bytes << '\n';
    }
    std::cout << "[PAGEFAULTS] scope=query_loop"
              << " major_before=" << usage_before.ru_majflt
              << " major_after=" << usage_after.ru_majflt
              << " major_delta=" << usage_after.ru_majflt - usage_before.ru_majflt
              << " minor_before=" << usage_before.ru_minflt
              << " minor_after=" << usage_after.ru_minflt
              << " minor_delta=" << usage_after.ru_minflt - usage_before.ru_minflt << '\n';
    const double total_ms = std::accumulate(query_latencies_ms.begin(), query_latencies_ms.end(), 0.0);
    std::cout << std::setprecision(10) << "[SEARCH] queries=" << run_queries
              << " total_search_ms=" << total_ms << " qps=" << run_queries*1000.0/total_ms
              << " loop_wall_ms=" << loop_ms << " loop_qps=" << run_queries*1000.0/loop_ms << '\n';
    report_latency(query_latencies_ms);
    report_recalls(eval_ground_truth, results, args.k);

    return 0;
}
