# CPU requantize from a fixed Chimera 4-bit index

## Scope and qualification

Source analysis confirms feasibility. The independent host-only executable compiled successfully using the same CUDA-TU rotator arithmetic policy and existing native quantizer/FHT/half-expansion objects. It contains no CUDA API calls or kernel launches. **CPU payload qualification passed** on a 10,000-document /1,087,049-token prefix against a separately built fresh8 index. All139,142,272 bytes of doc_full.bin and26,089,280 bytes of doc_1bit.bin match exactly, including rotator, onebit data and full8 factors. Four reused files share source4 inodes, and source4 payload hashes remain unchanged. Corrupted source-row, wrong raw SHA and existing-output negative checks passed. See `qualification-result.json`; normal-loader GPU query QA is owned separately by the Chimera agent. No full workload was requantized here.

`requantize.cu` and `run_requantize.py` live independently; no engine files were changed. Compile via `compile.sh ARM_SOURCE MATCHING_BUILD_DIR`. Keep the exact matching native objects/source hashes with the executable; the script's tool binary SHA binds their linked implementation, while the full build provenance should retain their origin.

## Why reuse is valid

In `build_index.cu`, ex_bits enters `validate_build_inputs`, final `encode_full_code`, stored member and `IndexHeader.ex_bits`. It does not enter seed42 centroid sampling, embedding→centroid CAGRA assignment, seed42 FHT rotator, rotated centroid CAGRA graph, IVF inverted lists, encode_one_bit or reorder_by_cluster. The cluster layout contains only 1bit codes/factors plus doc IDs/token positions. Therefore a fixed 4-bit index's graph/IVF/1bit payload can be preserved for 8-bit scoring.

The tool reads the serialized source rotator, expands raw FP16 to FP32 with the existing exact converter, and calls the original `rotate` + `encode_full_code(ex_bits=7)`. Every row also recomputes and bitwise checks source 1bit codes/factors **and** source 4bit full codes/factors. Any mismatch/nonfinite raw input rejects the staging output. This catches arithmetic-policy, source-row and serialized-rotator disagreement before publishing the derived index.

## Files and storage

- Hardlink unchanged `doclens.bin`, `ivf.bin`, `cluster_1bit.bin`, `centroids.carga`.
- Write a distinct `doc_1bit.bin`: identical header except ex_bits3→7, byte-identical serialized rotator/onebit codes/onebit factors, freshly generated full8 factors.
- Write distinct `doc_full.bin` at128 bytes/token using the existing safe SpilledFullCodes MAP_SHARED implementation, 64MiB flush/drop windows and hardlink publication.
- The new doc1bit payload is also spilled, avoiding an extra fullfactor anonymous array. Fixed source bytes are copied in64MiB windows. Ordinary small per-thread encoder scratch remains.

For MS597909919 tokens, additional physical disk is approximately(128+24)N bytes=90.88GB, plus small metadata; full8 alone is76.53GB. Graph/IVF/cluster/doclens consume no duplicate payload blocks on the same filesystem. Existing raw data, source4 index and prior FP32 files remain intact. Source hardlinks update inode link-count/ctime, not size/mtime/content; current Chimera stat guards use size/mtime/inode/device and remain compatible.

## Binding / execution interface

The public entry is the Python wrapper, not the internal executable. It requires an explicit binding JSON:

```
{
  "schema": 1,
  "source_build_receipt": "/absolute/source-build-receipt.json",
  "source_build_receipt_sha256": "...",
  "raw_identity_confirmed_against_source_build": true,
  "raw_npy_sha256": "...",
  "source_files_sha256": {
    "doclens.bin": "...", "ivf.bin": "...", "cluster_1bit.bin": "...",
    "centroids.carga": "...", "doc_1bit.bin": "...", "doc_full.bin": "..."
  }
}
```

The binding must be prepared from the accepted source build receipt and fixed raw provenance, not invented from an unrelated file that happens to pass onebit checks. For full MS, the full raw SHA should equal the pinned HF documents.npy SHA. All six index files, receipt and raw file are hashed before running; file stat identities are checked afterwards. Inputs are immutable during the run. The wrapper requires same filesystem, rejects existing destinations, retains failed staging diagnostics, hashes the outputs and exclusively publishes the completed directory using renameat2(RENAME_NOREPLACE). The receipt explicitly labels CPU-derived requantization and `fresh_8bit_build=false`.

```
python3 run_requantize.py --source /data/source4 --raw-fp16-npy /data/documents.npy \
  --binding binding.json --tool ./requantize --output /data/derived8
```

No resume is implemented. Failed partial staging is retained with explicit identity, while owned temporary mappings remove only their own temporary names. This simple tool requires a fresh output directory for every attempt.

## Required small-prefix acceptance

Use matching4/8 builds with identical raw prefix, doclens, dimension, cluster count, source/compiler arithmetic policy and source seed. Invoke the wrapper on the4bit source, then compare derived vs fresh8:

1. doc_full.bin all bytes identical;
2. doc1bit full8 factors all bytes identical; header/rotator/onebit codes/factors likewise;
3. source4 vs derived8 four reused files same inode/content; source4 payload hashes unchanged;
4. mutate raw/source4 bytes in a private test fixture: reject before publication; existing output is never overwritten;
5. load derived8 via the normal low-memory loader and validate query IDs/scores in the separate engine QA window.

CAGRA construction can have independent scheduling nondeterminism. The payload comparison is the decisive quantization check; graph identity is deliberately inherited from source4. Record derived8 construction time separately from independent fresh8 build time. This method does not measure the latter.

## Completed qualification identity

`qualification-binding.json` binds source4 build receipt and the complete278MB FP16 prefix provenance. `qualification_compare.py` performs full byte comparison, inode/source preservation and mutation tests; `qualification-result.json` records their PASS and all three index receipt digests. Derived output is `/home/lxq/graduate/data/multivector-colbertv2/indexes/chimera/lotte-prefix-10000-c8192-b8-requantized-qualification`. The3.94s CPU requantization/output-hash interval excludes input prehash and is not an independent fresh8 build measurement.
