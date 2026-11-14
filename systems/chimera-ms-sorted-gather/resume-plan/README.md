# Full MS resume attempt002 (prepared, not started)

Target stays all 597,909,919 tokens / 8,841,823 documents, C4,194,304, 4bit. Resume the accepted Q1 cache identity0544ef5a… from65,536 committed labels through all assignments, rotated graph, encoding and save. The original Q1 binary/wrapper/provenance/source-manifest/guard retain exact paths, inodes, contents, environment and all20 affinity. Only remove stop1, add --resume, choose new receipt/spill paths. The unchanged wrapper hashes153GB raw once and verifies complete cache digests during resume.

New standalone detached launcher uses the validated continuation guardian. Concurrent offline launch requires exact live LoTTE native3233295/start245395633 and actual WRITE holder3228697; queued owners never satisfy it. Otherwise it waits for its own guardian to acquire the global lease. Gates: MemAvailable72GiB, disk128GiB. Reserved new-job nonreclaimable budget48GiB; spill codes38.27GB are file-backed/reclaimable and hardlinked into final index. No automatic search/calibration/8bit conversion.

Safe stop during assignment:

```
verify job launcher PID+startticks+cmdline
          → SIGTERM exact launcher
          → TERM wrapper → TERM/reap native
          → audit last atomically committed prefix
          → cancel only own queued nonholder guardian
```

The frozen native has no finish-current-batch signal handler: TERM retains the last complete checkpoint; an in-flight batch may be repeated. Do not SIGSTOP or leave CUDA context paused. Wait for wrapper/native terminal and supervision receipt, verify process identities and both cache/output locks are free, then run `check_stopped_cache.py --hash-artifacts` (CPU-only). It verifies full fixed artifacts plus every committed chunk's continuity, size, SHA and label range, checks state stability, and reports uncommitted files without deleting them. A future attempt must use a fresh receipt/spill name; preserve orphan spills. Prefer stopping while assignment is running. If final save has begun, retain any partial index and review its disposition before resuming: existing index paths are deliberately rejected.

`check_stopped_cache.py` already passed on the accepted Q1 state without rehashing its5GiB artifacts (full-content hashes were freshly verified by Q1 and the large restore audit). Nine preflight CPU tests and three copied real-launcher/fake-child fixtures pass. These tests use private locks and no GPU. `monitor_resume.py` is the read-only1s stage/OS-memory observer for the eventual authorized job; collection gaps are retained.
