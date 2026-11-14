# Figure8 kernel candidates (private)

`--kernel-mode full|no-lut|no-aggregation|neither` controls the two independent optimizations. `--algorithm-variant full` is accepted for family guards; every other algorithm variant rejects during parsing. Original active, formal and observation sources/binaries are unchanged.

| Mode | CR binary IP | CR document/query max | DS token IP | DS document reduction |
|---|---|---|---|---|
| full | original LUT kernel | original fused/warp aggregation | original tiled LUT | original cooperative tile |
| no-lut | direct128-bit conditional sum | original fused/warp aggregation | direct128-bit conditional sum | original cooperative tile |
| no-aggregation | original LUT grouping | materialized scores+doc IDs, then separate atomic max | original tiled LUT | thread per document, scattered scan |
| neither | direct conditional sum | materialized, separate max | direct conditional sum | thread per document |

The no-LUT path skips LUT allocation/precompute/shared copy, while retaining CPU QueryState cb1_sumq and its upload. Direct IP uses the existing binary representation and `(ip-cb1_sumq)*factor`, never original FP32 document GEMM. The Full GPU kernel bodies remain text-identical to the qualified base.

No-aggregation obtains32 actual routed-list lengths on GPU, copies them to persistent pinned host offsets, and allocates/grows packed scratch from actual M=Σq|Lq|. Requested GPU scratch is8M+264 bytes; pinned host offsets add264 bytes. Capacity retains the largest actual observed M to avoid repeated allocation. Counts/offset transfer and allocation are timed work for this variant. Overflow rejects; CUDA allocation failure is not replaced by another algorithm. CR maxima start+0 and accept only positive values. DS maxima start−FLT_MAX and inspect every token; its one-thread final32-query sum follows the original tree/local_sum pattern, including signed zero.

Both private safety patches are shared with the algorithm candidate: clamp empty ceil chunks while preserving C/budget denominator/events/CPU carry, and clear captured scores for zero candidates. They are not installed in the active base. Existing5290-case chunk CPU receipt is reused; CLI tests exercise all four modes and unsupported family rejection.

CPU oracle covers D128/Q32, lengths255/256/257/299/300, negative CR clipping, negative DS maxima, zero route, empty/invalid lists, ties, zero factors and signed-zero reduction. It compares LUT versus direct arithmetic to FP64 using per-token magnitude/factor gamma128 bounds, then propagates max/sum bounds per document. Aggregation on/off is bitwise for each IP path. Synthetic top10 changes are recorded, not interpreted as real recall.

`kernel_ablation_qa` is a compiled GPU helper, not yet run: direct calls to genuine production/new kernels compare token matrices, packed doc mappings, partial maxima and final DS scores; all four combinations, invalid-doc/zero-route and length/tail cases. It requires explicit `--run` and `UMA_RUN_GPU_TESTS=1`. The separate planned real prefix runs compare original versus candidate Full, all modes' actual outputs/recall, and Nref11/chunks10 safety. No formal QPS claim before GPU qualification and later exclusive measurements.
