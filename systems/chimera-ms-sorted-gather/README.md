# Future MS: source-sorted centroid gather, isolated candidate

Status: CPU exact-bit oracles and private `gpu_build` compilation PASS. GPU/full-corpus execution has not run. Root review is required before future MS use. Active LoTTE builder8d95, source and checkpoint cache were not changed.

The candidate inherits the qualified owned-device centroid lifetime and checkpoint/supervision source. Exactly one engine file differs: `cpp/src/build_index.cu`, only the final gather loop inside sample_centroids. RNG/reservoir code, seed42, sampled_ids, assignment65536/k1/itopk150, FHT, CAGRA, kernels and quantization remain unchanged. `candidate-policy.json` and `source-manifest.json` explicitly bind the new default read order.

After sampling, an extra permutation0..C−1 is sorted by `(sampled_ids[cluster],cluster)`. Source rows are read in increasing order, while every copied/expanded row is stored in its original centroid slot. sampled_ids itself is never sorted. Each output element therefore receives the same immutable source half/float bit pattern through the same conversion/copy function, with no changed arithmetic or reduction. Extra order memory is16MiB for2^21 centroids and32MiB for2^22. The existing adapter's MADV_SEQUENTIAL is unchanged so the candidate isolates gather order.

## CPU proof checks

`prepare_cpu_tests.py` extracts the actual baseline and candidate sample_centroids functions. It adds only test hooks to inject custom IDs and capture IDs after gather; production sources are not instrumented. `cpu-extraction-provenance.json` binds each original function body. Native FP16 conversion is linked from the candidate's unchanged ordinary-CXX fp16_input.cpp.

`test_centroids` reads small mapped FP16 NPY and FP32 binary fixtures (3,148,168B total). It covers13 N/C pairs per mode, fromN1/C1 throughN4099/C4099, checking:

-26 complete centroid matrices by memcmp of every FP32 byte;
-26 actual reservoir-ID vectors against baseline;
-26 repeated deterministic results;
-6 custom duplicate/tail/cross-page selections, preserving original ID slots.

Fixtures include signed zeros, subnormals, finite extrema and FP32 values not representable in FP16. Their hashes remain unchanged after tests. `cpu-tests.log` records PASS_EXACT_CENTROID_BITS. These checks establish exact matrix equivalence in the tested cases and exercise the row-permutation proof; they are not a claim of recall approximation or measured full-corpus IO speedup.

## Build / freeze

`compile.sh` masks CUDA_VISIBLE_DEVICES=-1, uses CPU17–18/two jobs, architecture121, and builds only target gpu_build; Python bindings and gpu_search targets are not built. Supervisor/exec_guard files remain byte-identical to the qualified checkpoint baseline. `--help` and supervisor `--plan` passed without CUDA execution or full input hashing.

`validation.json` and `freeze.json` bind the exact candidate binary, source, patch, CPU evidence and operation metadata. Binary SHA:

`ecb75fba520e82df4600f9537033ea9a7349d5652ccf69ea7887caf571e04537`

Freeze SHA:

`90891d19d7ab1d49db6bbd60345cea38787e496da1ce5d143e561471e81614e3`

## Future MS invocation

`future-ms-plan.json` and `ms-provenance.json` are prepared only. The input is verified released FP16 directly; no153GB rehash or expansion occurred in preparation. The plan uses a new source-sorted cache/index namespace, with C4194304/ex_bits3 and20physicalCPU runtime budget under root scheduling. The actual checkpoint supervisor independently binds the new binary/source manifest, raw input, libraries, environment and affinity on execution. Existing LoTTE or other-binary caches must not be resumed.

Future build timing should record sampling wall/read_bytes/major faults and centroid digest separately from subsequent graph/assignment/encode stages. Matrix bits are preserved; GPU graph construction can retain its own scheduling nondeterminism. Full performance benefit remains to be measured in the authorized MS build.
