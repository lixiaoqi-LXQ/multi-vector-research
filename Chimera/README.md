# Chimera+

Chimera+ replaces only the CPU full-bit document scorer in the accepted ARM Chimera baseline with a CUDA scorer. Routing, candidate generation/refinement, 1bit scoring, chunk selection and the final CPU heap are unchanged. The existing index is loaded without conversion or rebuilding.

The current GPU scorer supports the experiment's 4bit, D128, 32-query-token representation. It preserves the CPU nibble layout, FP32 FMA/addition tree, correction factors, per-query-token maxima and score ties. One CUDA block handles each selected document. On GB10, ATS lets the GPU read the existing full-code/factor arrays through host page tables; no second corpus allocation or FP32 expansion is made. Other supported CUDA mappings use registered host storage.

- `--full-bit-device gpu` is the Chimera+ default.
- `--full-bit-device cpu` invokes the unchanged CPU scoring body for comparison.
- `--verify-full-bit` additionally replays exactly the same selected candidates on CPU and requires bitwise equal FP32 scores. This diagnostic must be disabled for performance measurements.

`BASELINE.json` identifies the imported experimental baseline. Git branch `baseline` preserves it; `Chimera+` contains this change. Source changes are limited to scoring, its workspace/lifetime and CLI integration, a standalone CMake guard, and numerical tests.

Build outside the source tree with CUDA architecture 121, the existing cuVS/RMM toolchain and `CHIMERA_BUILD_PYTHON=OFF`. Targets: `gpu_search`, `fullbit_gpu_qualification`. The qualification compares actual CPU SIMD scoring with GPU output on nibble boundaries, reordered/repeated IDs, document lengths 1–300, negative scores, zero/tied scores and empty candidate batches.

Heavy builds and all GPU validation/performance measurements use an exclusive nonblocking `~/device.lock`, held until all associated work completes. Build directories, logs, test specifications and outputs live in `/home/lxq/multi-query/ai-workspace/`; they are not source artifacts.

The four frozen LoTTE configurations (CPU20, chunks1) remain:

| Returned k | Recall target | nprobe | refine | full-bit | CAGRA itopk |
|---:|---:|---:|---:|---:|---:|
| 10 | .90 | 32 | 1000 | 20 | 64 |
| 10 | .95 | 128 | 1000 | 50 | 160 |
| 100 | .90 | 64 | 3000 | 300 | 96 |
| 100 | .95 | 256 | 3000 | 300 | 288 |

The single experiment report remains `/home/lxq/multi-query/reports/lotte-r10-brief.md`.
