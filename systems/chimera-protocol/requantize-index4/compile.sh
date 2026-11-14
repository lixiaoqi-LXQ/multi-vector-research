#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source_root=${1:?Provide ARM source snapshot path}
objects_root=${2:?Provide existing matching build directory}
# Reuse the exact native CXX quantizer / FHT / half converter objects.
# Compile the rotator-calling TU with the original CUDA-side Eigen scalar policy.
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -DNDEBUG -DEIGEN_DONT_VECTORIZE \
  -Xcompiler=-fopenmp,-march=native,-ftree-vectorize,-fexceptions \
  --diag-suppress=68,177,186 -arch=sm_121 \
  -I "$source_root/cpp/include" -I "$source_root/cpp/third_party" -I "$source_root/cpp/apps" \
  -isystem /home/lxq/miniconda3/envs/cuvs_benchmarks/include \
  requantize.cu \
  "$objects_root/CMakeFiles/chimera_core.dir/cpp/src/quantization.cpp.o" \
  "$objects_root/CMakeFiles/chimera_core.dir/cpp/src/fp16_input.cpp.o" \
  "$objects_root/CMakeFiles/chimera_core.dir/cpp/third_party/rabitqlib/utils/fht_avx.cpp.o" \
  -o requantize
