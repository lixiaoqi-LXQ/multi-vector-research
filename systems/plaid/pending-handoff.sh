#!/usr/bin/env bash
# Manual entry after root grants an exclusive GPU window. Caller holds flock.
set -euo pipefail
R=/home/lxq/graduate/ai-workspace/research/multivector/plaid
D=/home/lxq/graduate/data/multivector-colbertv2
cd "$R"
export PATH="$R/.venv/bin:$PATH" PYTHONPATH="$R/faiss-gpu-env/lib/python3.13/site-packages" LD_LIBRARY_PATH="$R/faiss-gpu-env/lib"
export CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST=12.1 TORCH_EXTENSIONS_DIR="$R/torch-extensions-cuda" MAX_JOBS=2 UMA_RUN_GPU_TESTS=1
unset GOMP_CPU_AFFINITY PLAID_SEARCH_PHASE_MANIFEST
export OMP_NUM_THREADS=20 OMP_THREAD_LIMIT=20 OPENBLAS_NUM_THREADS=20 MKL_NUM_THREADS=20 OMP_DYNAMIC=FALSE OMP_PROC_BIND=true UMA_SEARCH_CPU_BUDGET=0-19
export OMP_PLACES='{7},{5},{6},{8},{9},{15},{16},{17},{18},{19},{0},{1},{2},{3},{4},{10},{11},{12},{13},{14}'
I="$D/indexes/plaid-lotte-full-b4-km4"
J="$R/scale-runs/lotte-full-b4-km4-gpu-all-payload"
O="$R/controls/lotte-next-exclusive-newendpoint"
mkdir -p "$O"
common=(--index "$I" --data "$D/lotte-pooled-dev" --groundtruth "$D/groundtruth/lotte-pooled-dev-full")
case "${1:-help}" in
 qa32)
  python placement_control_isolated.py "${common[@]}" --selection "$J/grid/selection-frozen.json" --target 'R100>=0.9' --output "$O/qa32" --threads 20 --cpus 0-19 --resident-mode gpu-all-payload --skip-full ;;
 native)
  python native_organization_driver.py "${common[@]}" --split "$D/splits/lotte-calibration-seed42" --quality-source "$J/grid/calibration-results.json" --target 100:.90 --output "$O/native-resource10090" ;;
 r10095|r1095)
  if [[ "$1" == r10095 ]]; then target='R100>=0.95'; else target='R10>=0.95'; fi
  python guarded_search_phase.py --output "$O/$1-guard" -- taskset -c 0-19 python additional95_worker.py "${common[@]}" --split "$D/splits/lotte-calibration-seed42" --selection-plan "$J/additional95-cal-only-selection-frozen.json" --target "$target" --index-placement gpu-all-payload --threads 20 --output "$O/$1" ;;
 lotte8build)
  python future_phase_driver.py --docs 2428854 --tag lotte-full --bits 8 --kmeans-iters 4 --data "$D/lotte-pooled-dev" --groundtruth "$D/groundtruth/lotte-pooled-dev-full" --split "$D/splits/lotte-calibration-seed42" --codes-from4bit "$I" --index-placement gpu-all-payload --run-label reuse8-preflight-build --build-only ;;
 ms4build)
  python future_phase_driver.py --docs 8841823 --tag msmarco-full --bits 4 --kmeans-iters 4 --data "$D/msmarco-v1-passage" --groundtruth "$D/groundtruth/msmarco-v1-passage-full" --split "$D/splits/msmarco-calibration-seed42" --index-placement gpu-all-payload --run-label offline-build --build-only ;;
 ms8build)
  python future_phase_driver.py --docs 8841823 --tag msmarco-full --bits 8 --kmeans-iters 4 --data "$D/msmarco-v1-passage" --groundtruth "$D/groundtruth/msmarco-v1-passage-full" --split "$D/splits/msmarco-calibration-seed42" --codes-from4bit "$D/indexes/plaid-msmarco-full-b4-km4" --index-placement gpu-all-payload --run-label reuse8-preflight-build --build-only ;;
 *) echo 'Manual phases: qa32 native r10095 r1095 lotte8build ms4build ms8build. Root GPU handoff + outer flock required.' ;;
esac
