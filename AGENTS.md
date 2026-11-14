# Multi-query / Chimera+ workspace

- Canonical project root: `/home/lxq/multi-query`; datasets/queries/groundtruth: `/home/lxq/data/multivector-colbertv2`; indexes: project `indexes/`.
- Track source, static configuration and current final reports with Git. Keep datasets, indexes, environments, raw experiment outputs and receipts local and ignored.
- All new temporary scripts, logs, builds, raw results and review receipts go under `ai-workspace/`. Existing frozen experiment namespaces remain available through compatibility links; preserve their files and identities.
- Coordinate CPU/GPU heavy work through `~/device.lock`: exclusive nonblocking lock for builds and performance measurements, held through all child processes and synchronization. Only one index build at a time. Check external compute interference; do not stop other users' processes.
- Verify exact PID/start time before stopping owned jobs. Preserve resumable checkpoints and interrupted evidence. Never use broad process kills or remove lock files to unlock.
- MS MARCO is paused for migration at 142344192/597909919 committed tokens. Do not resume it or start PLAID unless the user authorizes continuation.
- Keep one concise performance report at `reports/lotte-r10-brief.md`; separate Chimera+ flow research is `reports/chimera-plus.md`. Report official PLAID GPU only.
- Do not edit frozen sources/binaries/specifications just to rewrite historical paths. Old graduate paths are compatibility links; update active documentation/configuration when needed.
- Use diagrams for workflows and comparisons. Embed generated diagrams in Markdown; do not emit raw Mermaid flowcharts in TUI replies.
