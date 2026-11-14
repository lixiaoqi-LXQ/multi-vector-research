#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
export CUDA_VISIBLE_DEVICES=-1
cmake=/home/lxq/graduate/ai-workspace/research/multivector/chimera/build-env/bin/cmake
ninja=/home/lxq/graduate/ai-workspace/research/multivector/chimera/build-env/bin/ninja
taskset -c 15-16 "$cmake" -S "$root" -B "$root/build" -G Ninja -DCMAKE_MAKE_PROGRAM="$ninja"
taskset -c 15-16 "$cmake" --build "$root/build" -j2
