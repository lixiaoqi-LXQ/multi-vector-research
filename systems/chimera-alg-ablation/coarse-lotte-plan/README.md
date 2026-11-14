# Full LoTTE coarse posting-list construction for Figure8

Prepared only; root reviews `spec.json` before GPU launch. This constructs all266,205,513 tokens /2,428,854 documents at C131072 and4bit, matching the actual full PLAID partition count. It does not stop after Q1.

```text
same LoTTE FP16 NPY → seed42 reservoir / sorted source gather → original CAGRA assignment
 → new coarse posting lists + unchanged global quantization → encoded index
 → saved rotated centroid bit audit → online exact-flat algorithm variants
```

The builder is the qualified sorted-owned binary `ecb75fba…` with identical seed42/sample slots, assignment batch65536/k1/itopk150 and ex_bits3. Its source manifest is `ca5ea1c7…`. Offline assignment and serialized graph construction remain CAGRA. The online algorithm candidate avoids graph search in no-IVF-PG/neither but retains the graph as a matrix container; this plan makes no graph-free offline or memory claim.

Input is the canonical68,148,611,456-byte LoTTE NPY. CPU preflight binds its unchanged device/inode/size/mtime to the already completed full-SHA measurement in the live fine-build receipt, checks its exact NPY header, checks all doc lengths/count/sum and the accepted FP16↔FP32 complete integrity receipt. The wrapper still rehashes the full input before launch. It expands only bounded CPU buffers; the136GB FP32 archive stays archived.

New index and durable cache live under `/home/lxq/graduate/data/multivector-colbertv2`; receipts, spill and compilation caches stay under the AI workspace. Expected incremental serialized index is32.045GB, checkpoint1.232GB. Full-code spill is17.037GB but publication hardlinks the same inode, avoiding a second payload. New nonreclaimable memory budget24GiB; preflight requires64GiB MemAvailable and64GiB disk free. These are preparation models; actual peak memory and I/O are recorded independently.

`launch_full.py`, `guard_full.py`, `start_detached.py`, exact-child wrapper/supervisor and checked PDEATHSIG guard are byte copies of the previously CPU/GPU-validated full-MS Q1 chain. Only the dataset-specific `preflight.py` was adapted to full LoTTE/C131072/no-stop. Nine CPU preflight tests and three private-lock fake-process launcher tests pass. They never use the real device lock or launch CUDA. The original legacy field `existing_GT_identity` points to the required live fine-LoTTE predecessor; MS GT is an optional observed job, so its normal completion does not block this independent construction.

After root authorizes the exact spec SHA, launch with:

```sh
CUDA_VISIBLE_DEVICES=-1 /home/lxq/graduate/.venv/bin/python start_detached.py --run --approved-spec-sha256 SPEC_SHA
```

The bootstrap detaches only the top-level launcher; checked parent-death supervision continues through wrapper/native. The actual global-lock predecessor is verified, successor queues before work starts, and subsequent root-coordinated measurements wait until all offline jobs are terminal. No automatic search/calibration begins after construction.

Start `monitor_coarse.py` with CUDA hidden after the job directory exists. It reads process identities, RSS/HWM/fault/I/O counters, MemAvailable, GPU views and1s checkpoint events. Sampling time is collection time; offline cohort timing is not formal throughput. Final acceptance requires complete wrapper/native exit0, every label batch and index payload identities, the generic saved rotated centroid bit audit, then the normal routing sidecar. Failed artifacts are retained; explicit resume uses the same immutable cache/source/binary/input/library/environment binding with fresh attempt receipts and spill names.
