#!/usr/bin/env bash
set -euo pipefail
exec 9>>"$HOME/device.lock"
flock -xn 9 || { exec 9>&-; exit 75; }
trap 'flock -u 9; exec 9>&-' EXIT
export CUDA_VISIBLE_DEVICES=-1
export TMPDIR=/home/lxq/graduate/ai-workspace/builds/chimera-plus/tmp
mkdir -p "$TMPDIR"
cmake_bin=/home/lxq/graduate/ai-workspace/research/multivector/chimera/build-env/bin/cmake
taskset -c 17-18 "$cmake_bin" -S /home/lxq/graduate/new-idea/multi-vector/Chimera -B /home/lxq/graduate/ai-workspace/builds/chimera-plus -G Ninja -DCHIMERA_TOOLCHAIN_ROOT=/home/lxq/miniconda3/envs/cuvs_benchmarks -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=121 -DCUDA_nvJitLink_LIBRARY=/home/lxq/miniconda3/envs/cuvs_benchmarks/lib/libnvJitLink.so.13 -DCMAKE_MAKE_PROGRAM=/home/lxq/graduate/ai-workspace/research/multivector/chimera/build-env/bin/ninja -DCHIMERA_BUILD_PYTHON=OFF
taskset -c 17-18 "$cmake_bin" --build /home/lxq/graduate/ai-workspace/builds/chimera-plus --target gpu_search -j2
