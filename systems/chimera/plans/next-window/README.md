# Previous window plan — superseded by centroid ownership qualification

Status: prefix/fresh4/fresh8/CPU requantization/storage QA ran; subsequent full construction was interrupted by root for a confirmed borrowed-centroid dataset lifetime defect on ATS. Do not execute the old full-build command. See state.json and review/centroid-ownership/affected-indexes.json. New explicit-owned-device candidate qualification is isolated under chimera-checkpoint; old frozen evidence is preserved.

1. After explicit release, generate complete 1,087,049×128 uint16 NPY prefix under `/home/lxq/graduate/data/multivector-colbertv2/chimera-inputs/lotte-prefix-10000/documents-prefix-qualification.npy`. Preserve bit patterns, bind source fixed HF revision plus source header/data-prefix SHA and rows; do not hash/copy full68GB for this QA.
2. Run fresh4 and fresh8 builds below. Each runner obtains the shared flock and verifies10 seconds compute idle, freezes binary, records UMA_RUN_GPU_TESTS=1, OMP20 placements and six payload SHA. Both use opt-in spill. Reject existing output/temp paths.
3. Protocol agent runs CPU requantize wrapper from fresh4 to separate `lotte-prefix-10000-c8192-b8-requantized-qualification`, with fresh4 source-build receipt and exact prefix SHA binding. Compare all bytes of doc_full and doc_1bit against fresh8; check inherited four files same inode/hash and source unchanged. Mutation/rejection tests in private fixtures. Failure stops full construction.
4. Main GPU QA loads fresh4/fresh8/derived8 default and lowmem;32 queries, warmup5, k100/nprobe128/refine10000/full400/chunks1/itopk160. Compare default/lowmem all IDs+scores exactly; fresh8/derived quantization bytes exact (graphs may differ), validate finite sorted unique outputs and GT. Verify20 actual worker placements, profile-range absent, pagefault receipts present and major_delta=0, lowmem prefault bytes/time. Investigate any major fault before full QPS. QA is diagnostic.
5. Release QA lease, record results, then run full LoTTE c2^21/ex3 build below once QA accepted. Full c2^21 build is independently timed and retains its own frozen builder/source/input identity. No concurrent PLAID/GPU work.

```sh
cd /home/lxq/graduate/ai-workspace/research/multivector/chimera
CUDA_VISIBLE_DEVICES=-1 python3 prepare_qualification_prefix.py --run
python3 run_build.py --spec plans/next-window/fresh4.json --output runs/next-window-fresh4 --run
python3 run_build.py --spec plans/next-window/fresh8.json --output runs/next-window-fresh8 --run
# Protocol agent: binding, CPU requantize, byte/inode/mutation checks.
# GPU QA: bounded32-query storage and pagefault checks; no performance claim.
python3 run_build.py --spec plans/next-window/full-build.json --output runs/lotte-full-c2097152-b4-build --run
```

Canonical full input136297222664 bytes and source integrity receipt already accepted by parent. Full main CPU policy20 cores0-19, explicit fast-first OMP_PLACES/master7. Fresh qualification indexes are separate from old frozen QA artifacts.
