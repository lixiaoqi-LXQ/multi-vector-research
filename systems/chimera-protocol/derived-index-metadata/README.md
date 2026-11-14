# Derived8 metadata adapter (separate from frozen requantization)

`publish_metadata.py` verifies an accepted CPU-requantized8 index and prepares a normal search-runner-compatible `build-provenance.json`. Default behavior writes only the explicitly named preview outside the index. `--publish` exclusively links the metadata into the index after verification; an existing manifest is rejected. Six payload files, the frozen requantize tool, its receipt and qualification evidence remain unchanged.

Verification binds the caller-pinned requantize receipt SHA, source4 build receipt SHA and its six actual payload hashes, actual derived six hashes/sizes, original raw FP16 file SHA and stat identity, qualified binary/qualification receipt SHA, all-row source4 invariant checks, document counts/lengths and exact8bit header. The four reused payloads must still share source inodes. The compatible metadata records operation `CPU requantize from fixed 4-bit index to 8-bit`, `fresh_8bit_build=false`, correct docs/tokens/D/ex_bits7/total_bits8, inherited pinned HF source identity and actual index identity. It reports the separate CPU-derived interval without labeling it fresh8 build time.

Tests use tiny synthetic identity fixtures and never run GPU or mutate real indexes. `cpu-tests.log`:22 executed fixture checks (13 distinct cases, including inherited repetitions) passed in0.073s. Cases cover publication/file preservation, existing metadata, receipt SHA, source/raw/derived mutation, unapproved tool, hardlink identity, publication-time recheck and explicit runner acceptance. A real qualification-index dry run produced `qualification-build-provenance.preview.json`; no build-provenance.json was written into that index.

`run-search-derived-acceptance.patch` is an unapplied independent runner patch. Fresh-build behavior is unchanged. Derived indexes require these explicit future spec fields:

```
"allow_derived_index": true,
"approved_requantize_tool_sha256": "<qualified binary SHA>",
"approved_requantize_qualification_sha256": "<qualification-result.json SHA>"
```

The runner rechecks the derived receipt/qualification hashes and their binding to current index identity, all-row verification and approved tool. `run_search_grid.derived.py` is a review copy only; active runner source was not modified. ARM agent confirmed no conflicting runner edits. Root should review the adapter, preview and runner patch before any publication/integration. Full raw rehash occurs only on an actual adapter invocation in a scheduled offline window; none was performed for full LoTTE/MS during preparation.

Reproduce the qualification preview using `qualification-preview-command.json`. To publish after review, use the same argv with a new preview pathname and append `--publish`. The index directory otherwise stays byte-identical.

## Root-reviewed publication completed

Root reviewed and authorized publication. The qualification derived8 index now has exclusively published compatible metadata, and the reviewed runner copy is installed for future freezes. `publication-receipt.json` records exact before/after hashes; all six payloads and the original requantization receipt remain identical. `post-publication-validation.json` records successful explicit-derived acceptance, implicit-derived rejection, fresh4/fresh8 identity equality with the baseline runner and AST equality of every original function except the intended identity_checks hook. Existing frozen builders and search snapshots remain intact. `future-derived-spec-fields.json` contains the exact approved qualification/tool identities for future8 specs.
