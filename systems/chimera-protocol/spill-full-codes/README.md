# MS8 full-code spill build patch

Independent optional storage patch; `base/` is a read-only ARM-port snapshot. No GPU execution or active-source modifications are part of this artifact.

The MS8 8-bit build has 597,909,919 token vectors. Existing simultaneous encoded arrays cost 188 bytes/token (104.69 GiB): full codes 128, one-bit codes 16, two factors 8, token-doc IDs 4 and cluster layout 28, plus IVF offsets/IDs. With approximately 12 GiB baseline and 3 GiB CAGRA, only about 2 GiB remain on this 121.7 GiB host before context and allocator costs. FP16 input mapping alone therefore leaves an unsafe capacity margin.

`--spill-full-codes /absolute/ai-workspace/tempfile` selects exclusively created MAP_SHARED full-code output; all quantization, seeds, CPU scoring and GPU cluster arrays remain the existing algorithm. Old API overloads and no-option CLI retain vector storage. The remaining core anonymous peak is approximately 60 bytes/token (33.41 GiB). Full-code file pages are written in 64 MiB windows, synchronously flushed, then dropped from this mapping. Clean file cache is reclaimable by Linux; this is a bounded anonymous-memory model, not a promise of constant MemAvailable. Disk writeback adds build cost and must be recorded separately.

The temporary file reserves the full payload before encoding (8-bit: 76,532,469,632 bytes; 4-bit: 38,266,234,816 bytes). Input FP16 and all existing datasets stay intact. Save preflights that the temp lies outside the index directory and shares its filesystem, before existing index files can be removed. `doc_full.bin` becomes a hardlink to the same fully flushed inode, requiring no second full-code disk allocation. Destructor unlinks only the exclusively owned original temporary name after checking inode identity; saved links remain. Low-memory loaded-index save refusal remains unchanged. A process crash can leave the temporary name for manual identity-checked cleanup.

Patch integration: apply `spill-full-codes.patch` to the matching ARM snapshot, review any concurrently modified context, then compile normally. `BuildOptions` adds opt-in overloads for vector, FP32 pointer and FP16 pointer input. The constructor/destructor translation units include the complete spill type. `full_bit_codes_data()` selects spilled build, read-only loaded mapping, then default vector.

CPU validation command (run only outside formal timing):

```
c++ -std=c++17 -O1 -I modified/cpp/include test_spill.cpp -o test_spill
./test_spill ./cpu-test-files
```

The test covers byte equality across multiple flush windows, preallocated zero initialization, same-inode hardlink persistence after destruction, no overwrite, original-name ownership protection and preflight rejection. GPU quantizer/build equivalence still requires the integrating agent's existing small-prefix regression.

CPU validation completed successfully; see `cpu-validation.json`. Both snapshot and then-current ARM source pass `git apply --check`. The full CUDA translation units have not been compiled by this child agent.
