#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
export TMPDIR="$root/tmp"
mkdir -p "$TMPDIR"
"$root/build-env/bin/cmake" -S "$root/arm-source" -B "$root/build" -G Ninja \
 -DCHIMERA_TOOLCHAIN_ROOT=/home/lxq/miniconda3/envs/cuvs_benchmarks \
 -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
 -DCMAKE_CUDA_ARCHITECTURES=121 \
 -DCUDA_nvJitLink_LIBRARY=/home/lxq/miniconda3/envs/cuvs_benchmarks/lib/libnvJitLink.so.13 \
 -DCMAKE_MAKE_PROGRAM="$root/build-env/bin/ninja" \
 -DCHIMERA_BUILD_PYTHON=ON \
 -DPython_EXECUTABLE="$root/build-env/bin/python" \
 -Dpybind11_DIR="$root/build-env/lib/python3.13/site-packages/pybind11/share/cmake/pybind11"
"$root/build-env/bin/cmake" --build "$root/build" -j 3
