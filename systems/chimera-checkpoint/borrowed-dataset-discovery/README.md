# Optional construction checkpoint/resume — isolated candidate

Status: isolated implementation and CPU compilation/checks. **GPU construction, serialized CAGRA round-trip, and stop/resume equivalence remain unqualified.** Nothing was applied to `chimera/arm-source`, its build, active runner, or running builder. All source changes are in `source/`; `cpp/third_party` is a read-only symlink to the baseline dependencies.

The optional path retains the original sampled centroids, first CAGRA default-parameter graph, fixed assignment batches65536, k1, itopk150, seeds42, float input conversion, rotated second graph, quantization and full-code spill handling. It adds progress and durable bookkeeping. Without checkpoint flags, the original sample/build/assignment/encode flow runs with the same algorithm/parameters; no search or scoring kernel was changed.

## Durable artifacts and guards

`--build-checkpoint-dir DIR --build-checkpoint-identity-sha256 SHA` creates an exclusive cache; existing caches require `--resume-build-checkpoint`.

```
cache.identity.json   immutable input/binary/libraries/source/env binding (wrapper)
cache/
  .lock               one checkpoint writer at a time
  state.txt           atomic V1 metadata: config, artifact SHA, completed prefix
  centroids.f32       original sampled FP32 matrix in original order + SHA
  assignment.cagra    first graph including dataset, native cuVS serialization + SHA
  labels/<start>.u32  immutable original-batch assignment labels + per-chunk SHA
```

Metadata binds token count, D, cluster count, bitwidth, assignment batch/k/itopk and both seeds, plus raw document-length-vector SHA and the supervisor's identity SHA. Resume verifies centroid/graph files and every committed label chunk, contiguous original batch boundaries, exact byte lengths and all label ranges. Completion stores/verifies the concatenated full-label SHA. After deserialization, the core also checks actual graph size/dimension/default degree64. Files are fsynced before atomic metadata publication and directory fsync. Creation of the cache itself is followed by fsync of its parent directory before initial state publication. Unreferenced canonical artifacts from a crash window are hard-linked to `orphan-*` names and then regenerated; old evidence is retained, never adopted without a committed digest. Failure before the initial state commit requires a fresh cache path.

`--stop-after-assignment-batches N` commits N **new** original batches and returns exit3 before final index publication. On resume, existing committed batches are loaded and never searched again. A stop on the final batch still returns3; the next invocation validates the full prefix and continues the original remaining pipeline. The cache remains available after final save. Construction cache hits/resume are recorded separately from independent fresh build time.

## Identity supervisor

`run_checkpoint_build.py --plan ...` prints a plan without data hashing, directory creation or GPU use. `--run` requires `UMA_RUN_GPU_TESTS=1`; the caller/coordinator must own the GPU lease or explicitly authorize an offline cohort. The supervisor does not start a hidden benchmark or acquire a second device lock.

Before launch, the supervisor hashes the actual raw input, doclens, frozen binary, resolved `ldd` libraries, any extra identity files and compiled source manifest; it checks the pinned source identity and expected input hashes in the supplied provenance JSON. It binds CPU affinity and relevant OpenMP/CUDA/library environment. Resume requires the exact same canonical binding. Do not edit/move/recompile the frozen binary, exec guard, supervision module, dependencies, source manifest or binding between invocations. The wrapper holds an exclusive output lock and refuses an existing index/spill pathname. The GPU child inherits that locked file descriptor. SIGTERM/SIGINT requests terminate only the exact owned child, escalate to SIGKILL after grace, and reap it before lock release; ordinary exceptions take the same cleanup path. A tiny checked-parent-PID exec guard installs Linux PDEATHSIG(SIGKILL), protecting hard-parent-death startup and execution. Graceful interruption/failure writes a durable supervision receipt. A hard SIGKILL can prevent a final parent receipt; its child is killed and retains the inherited lock until terminal. Failed index outputs must be retained and a new final-output pathname used.

Provenance JSON shape:

```
{
  "source_identity": {"repository":"...", "revision":"<40hex>",
                      "documents_url":"https://huggingface.co/datasets/.../resolve/<40hex>/documents.npy"},
  "input_sha256":"<actual pinned input SHA>",
  "doclens_sha256":"<doclens binary SHA>",
  "source_manifest_sha256":"<this isolated compiled source-manifest.json SHA>"
}
```

Use the existing fixed-version source derivation receipt when the input is a prefix fixture. When `input_derivation_receipt` / `input_derivation_receipt_sha256` are declared, both are required and the actual receipt contents are hashed, verified and tracked across the run/resume. Full-data hashes run only in a scheduled offline invocation; none were run while preparing this implementation. `source-manifest.json`, binary/library hashes and run receipts identify the actual implementation. On successful final save the wrapper writes `build-provenance.json` with actual `vector_input`, `input_mode=fp16_npy` (for FP16), and `input_files.vector_input` path/device/inode/size/mtime/SHA from the bound input. This preserves the qualified source4→derived8 adapter interface. It is compatible with the existing basic search identity fields, explicitly labeling checkpoint construction and not claiming independent fresh build time.

Example invocation (paths and provenance must be prepared/frozen by the coordinator):

```
UMA_RUN_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=20 ... \
python3 run_checkpoint_build.py --run \
 --binary build/gpu_build --input-mode fp16-npy --input /home/lxq/graduate/data/.../documents.npy \
 --doclens /home/lxq/graduate/data/.../doclens.bin --clusters 8192 --ex-bits 3 \
 --cache /home/lxq/graduate/data/.../checkpoint-qualification \
 --index /home/lxq/graduate/data/.../final-qualification \
 --provenance-json prepared-provenance.json --source-manifest source-manifest.json \
 --receipt-dir /home/lxq/graduate/ai-workspace/.../first \
 --library-dir /home/lxq/miniconda3/envs/cuvs_benchmarks/lib \
 --library-dir /usr/local/cuda-13.0/lib64 --stop-after-batches 1
```

The wrapper, like the underlying builder, returns exit3 on an intentional committed stop. Then repeat with `--resume`, a new receipt directory and no stop flag. A spill path can be passed unchanged from original construction (`--spill-full-codes`), but it must be absent at the start of any invocation that will encode. Source manifests, source arrays, doclens and saved checkpoint graph remain immutable.

## CPU validation and compile

`compile.sh` uses the baseline cuVS/toolchain/sm121 CMake setup, adds OpenSSL SHA256, disables Python bindings, masks CUDA devices and limits compilation to cores15-16 / two jobs. `compile-parent-durability.log` records the latest complete builder link, including cache parent-directory fsync. No kernel was launched during compilation.

`test_checkpoint.cpp` uses tiny CPU-only fake graph bytes solely to test checkpoint I/O and metadata. `cpu-tests-parent-durability.log` has19 cases including actual parent-directory fsync before initial cache state; it covers exact full-label stop/resume bytes, identity/parameter mismatch, concurrent cache access, missing/new caches, centroid/graph/label corruption, truncation, trailing metadata, full digest corruption, invalid/noncontiguous batches and orphan preservation. These checks do not establish cuVS serialization or GPU equivalence. `--help` and wrapper plan/preflight checks exercise the CLI without GPU execution. `test_supervision.py` / `cpu-supervision-tests-2.log` additionally cover SIGTERM/SIGINT, escalation with a live-child lock check, post-spawn exception cleanup, inherited lock fd, hard-parent-death via PDEATHSIG, normal exit3, exact parent-directory fsync, derivation receipt content mutation, and the derived8 adapter raw-stat metadata access. These fake children are CPU-only.

## Required GPU qualification before full use

Under an authorized window, use the existing10k/c8192 fixed FP16 qualification input and frozen candidate builder.

1. Run a small original no-checkpoint build as comparison if needed; preserve raw input/doclens/parameter/source identities.
2. Checkpoint build, stop after1 batch; assert exit3, no final index, valid immutable matrix/graph/labels metadata.
3. Before continuation branches, run the separate `roundtrip-fixture/build/checkpoint_graph_roundtrip` fixture with the committed `centroids.f32`, the exact FP16 qualification NPY and a fresh output directory. It builds **one** live graph, searches the first65536 and tail38473 token rows with k1/itopk150, serializes that graph using original PG_CAGRA save/load, and compares loaded labels/distances bitwise plus logical dataset/graph SHA against the still-live original. It also repeats queries on the original live graph to expose any native nondeterminism. No independently rebuilt graphs are compared. The fixture is compile-only until root authorizes GPU execution.
4. Clone that committed checkpoint and its binding into two fresh cache paths. From the same first graph/prefix: branchA resumes to completion; branchB resumes for2 new batches, stops and resumes to completion. This holds graph identity fixed across the continuation comparison.
5. Verify all preserved prefix chunk bytes, complete label count/range/digests and full-label equality; compare doclens/doc1bit/docfull payloads. Checkpoint corruption must reject before continuation. The independently rebuilt rotated final CAGRA graphs can have their existing scheduling nondeterminism; preserve their actual identities rather than silently replacing them.
6. Load final indexes through the original engine and check complete query outputs, finite/sorted/legal IDs, actual quality and native/default/lowmem constraints. Qualify any mismatch explicitly; small-prefix checks do not establish full-corpus acceptance.

Root review and this GPU gate are required before replacing or restarting a full construction attempt. The currently running default builder is unaffected.

The C++ candidate builder remains byte-identical after the wrapper supervision fixes. `roundtrip-fixture/compile.sh` builds its standalone test using the original isolated PG_CAGRA/fp16 input methods; it does not edit/relink the candidate builder. Set the same conda/CUDA library path when running the fixture. Logical dataset rows are copied with their actual stride; padding bytes are excluded from the equality claim.
