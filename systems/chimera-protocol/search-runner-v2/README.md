# Future Chimera search runner v2

Use `chimera/run_search_grid_v2.py`, `select_points_v2.py`, and `accept_run_v2.py` for future runs. Old entrypoints and frozen results remain intact.

Each plan/run requires a fresh output directory. The runner freezes executable, exec guard, and Python sources. Its exact native child inherits the global flock descriptor; TERM/INT, monitoring exceptions, timeout, and parent death retain the lease through native termination. Supervision records terminal/reaped state before release. GPU polling is bounded and does not change search timing.

Specs accept `kernel_mode`: full / no-lut / no-aggregation / neither and `algorithm_variant`: full / no-cr / no-ivf-pg / neither. Explicit fields produce the corresponding engine flags. Flat algorithm modes require `expected_coarse_clusters` equal to the actual index; CAGRA modes reject it. An absent field binds metadata to full and omits its flag for the original CLI. Use only a binary whose frozen capabilities support the selected combination.

Calibration selection rejects mixed mode, binary, six-file index identity, bit width, K, query/GT/split identity, CPU affinity/OpenMP, load mode, or warmup. Search parameters within one family remain selectable. `accept_run_v2.py` checks the same family, frozen sources/guard, each child supervision record, existing routing guard and output contract.

CPU evidence: 13 supervision/durability fixtures including TERM, INT, escalation, started/poll exceptions and hard parent death; 8 contract methods including original scientific-function AST equivalence, 16 mode combinations, family mutations, actual selector rejection, explicit command preparation and existing-output preservation; 14 routing and 22 metadata tests. No GPU run was performed. The accepted original identity/recall/thread/latency functions are AST-identical; only GPU observer calls gain a 5s timeout.

Explicit mode specs require exactly one `[VARIANT]` log line matching the declared algorithm/kernel and LUT/aggregation booleans; both producer agents agreed on this log contract.

No-CR modes ignore positive `k_refine`; their actual refined count equals query-dependent CG count (bound 0..docs), and target is min(k_full_bit,actual_CG). Only CR-enabled modes enforce k_full_bit<=k_refine. All modes enforce positive CLI fields, nprobe capacity and k_full_bit<=docs as an explicit memory-budget constraint. No-CR helper validation covers kref1/kfull11/docs11; mixed CPU budgets remain separate calibration families.
