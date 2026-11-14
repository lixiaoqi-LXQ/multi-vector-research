# Chimera Figure8 algorithm candidate

The private candidate implements `full`, `no-cr`, `no-ivf-pg`, and `neither`. CPU-only preparation and compilation pass; GPU qualification has not run. Active full LoTTE construction, accepted source995b, all existing indexes and formal binaries are unchanged.

```text
full/no-cr: fine posting lists + original CAGRA route
no-ivf-pg/neither: independently rebuilt coarse posting lists + exhaustive FP32 flat route

CG bitmap → dense complete candidate IDs → original CR → original DS   (CR on)
CG bitmap → dense complete candidate IDs ────────────→ original DS   (CR off)
```

`--algorithm-variant full|no-cr|no-ivf-pg|neither` selects the mode. Flat modes require `--ablation-coarse-clusters C` matching the loaded index, while other modes reject that declaration. `--kernel-mode full` is accepted; this binary rejects non-full kernel modes. The shared `[VARIANT]` line records `algorithm_variant`, `kernel_mode=full`, `use_lut=1`, and `use_aggregation=1`; a detailed `[ALGORITHM_ABLATION]` line records the actual router and CR behavior. `[ALGORITHM_WORKSPACE]` records conservative CG/DS capacity and flat scratch.

Flat routing uses cuBLAS FP32_PEDANTIC dot products plus FP32 norms and RAFT exact minimum selection. It never calls the online graph search. The deserialized CAGRA graph remains a dataset container and occupies memory; this candidate does not claim graph-free construction or capacity. Formal no-IVF_PG uses rebuilt posting lists at the actual PLAID cluster count. The tiny fixture reuses one C256 index strictly for functional stress.

No-CR retains all materialized CG documents regardless of k_refine and bypasses partial score, sum and CR TopK. DS bounds use all possible CG candidates and their largest document lengths. Worst LoTTE token-score capacity is34.074GB. The independent chunk repair preserves requested chunk count, original ceil-sized nonempty ranges and cumulative budget denominator, clamps empty tails and records their event chain without zero-grid launches. Empty CPU carry steps still grow the budget. Zero CG also clears stale captured scores. These repairs are separate patches.

CPU checks:5290 chunk partitions/12342 empty ranges/3 invalid shapes;5 valid/9 rejected CLI cases;6 invalid build shapes with CUDA hidden;44 cases of extracted actual CG bitmap kernels;45 expanded-L2 FP64 tolerance cases. Normal Full search computation, after removing explicit variant branches and the two safety repairs, matches the qualified baseline text; original CUDA scoring, CPU scoring, quantization, graph and build math files match its SHA.

`gpu-qualification-plan.json` contains seven serialized bounded stages. Every stage uses `run_qualification.py PHASE --guardian-pid ACTUAL_HOLDER --run` only after root authorizes the small correctness slot. It uses the previously qualified exact-child supervisor, checked PDEATHSIG guard, inherited output lock,170s timeout plus10s cleanup and fresh outputs. `--run` is never implied. All generated synthetic data belongs under `/home/lxq/graduate/data/multivector-colbertv2/qualification`.

The pipeline fixture uses11 documents,1601 tokens with lengths1…300, C256 (above default extended NN-descent degree192),32 queries per condition, six chunk settings and two full-bit budgets. No-CR modes set k_refine=1 while the all-document budget is11: every document must still be returned with its CPU quantized MaxSim score. Full-budget results must match the Full/chunk1 reference bit-for-bit across every mode/chunk. Zero-CG event paths and coarse-count rejection are included. Smaller full-bit budget results are checked for valid scores/order and later NVTX counts, without assuming identical selected IDs across chunkings.

Actual Khat is measured from the existing `Chimera/cpu_fullbit` INT64 NVTX payload. Its NSYS observation remains separate from QPS. Production prefix equivalence against the frozen original engine and every GPU fixture remain required before promotion. `freeze.json` binds source, binaries, tests, plan and supervisor dependencies.
