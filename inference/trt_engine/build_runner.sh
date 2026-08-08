#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$root/bin"
g++ -std=c++17 -O2 "$root/run_omnivla_engine_dynamic.cpp" -o "$root/bin/run_omnivla_engine_dynamic" \
  -I"${CUDA_HOME:-/usr/local/cuda}/include" -L"${TENSORRT_LIB_DIR:-/usr/lib/$(uname -m)-linux-gnu}" \
  -lnvinfer -lcudart -ldl
