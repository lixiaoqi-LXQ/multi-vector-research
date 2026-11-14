#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
export CUDA_VISIBLE_DEVICES=-1
export TMPDIR="$root/tmp"
mkdir -p "$TMPDIR"
taskset -c 15-16 /home/lxq/graduate/ai-workspace/research/multivector/chimera/build-env/bin/cmake \
 -S "$root/source" -B "$root/build" -G Ninja \
 -DCHIMERA_TOOLCHAIN_ROOT=/home/lxq/miniconda3/envs/cuvs_benchmarks \
 -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
 -DCMAKE_CUDA_ARCHITECTURES=121 \
 -DCUDA_nvJitLink_LIBRARY=/home/lxq/miniconda3/envs/cuvs_benchmarks/lib/libnvJitLink.so.13 \
 -DCMAKE_MAKE_PROGRAM=/home/lxq/graduate/ai-workspace/research/multivector/chimera/build-env/bin/ninja \
 -DCHIMERA_BUILD_PYTHON=OFF
taskset -c 15-16 /home/lxq/graduate/ai-workspace/research/multivector/chimera/build-env/bin/cmake --build "$root/build" --target gpu_build -j2

taskset -c 15-16 cc -O2 -Wall -Wextra "$root/exec_guard.c" -o "$root/exec_guard"
