# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CUDA GEMM (General Matrix Multiply) optimization learning project. It demonstrates progressive optimization steps from a naive implementation toward cuBLAS-level performance.

## Build Commands

```bash
# Build all targets
mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Build a specific target
cmake --build build --target opt00
```

## Running

### Single Optimization Version
```bash
./build/opt00 <M> <N> <K> [warmup] [iters] [repeats]
# Example: ./build/opt00 2048 2048 2048
```

Parameters:
- M, N, K: Matrix dimensions (required)
- warmup: Warmup iterations, not counted (default: 2)
- iters: Kernel executions per measurement (default: 5)
- repeats: Measurement repetitions, median taken (default: 3)

### Benchmark Suite
```bash
./build/gemm_benchmark <start> <stop> <step> [warmup] [iters] [repeats]
# Example: ./build/gemm_benchmark 256 4096 256

# Or use the convenience script:
./examples/full_benchmark.sh 512 2048 256
```

### Convenience Scripts
```bash
./examples/single_run.sh <version> [M N K] [warmup iters repeats]
# version: 0=Naive, 1=Shared Memory, 2=Register Blocking
```

## Profiling

```bash
# Nsight Compute
ncu --set full -o report.ncu-rep ./build/opt02 2048 2048 2048
ncu-ui report.ncu-rep

# Nsight Systems
nsys profile --output=report.nsys-rep ./build/gemm_benchmark 256 2048 256
```

## Architecture

```
include/
  common.h          # Shared utilities: CHECK_CUDA, CHECK_CUBLAS macros,
                    # init_matrix, allclose, gflops, measure_median_ms

src/
  optimizations/    # Numbered optimization versions
    00_naive/       # Baseline implementation
    01_shared_memory/  # Tiling with shared memory
    02_register_blocking/  # Register-level tiling

  benchmarks/       # Multi-version benchmark runner (gemm_bench.cu)
  localtest/        # Development test files
  plot/             # Python plotting (plot.py)
```

### Key Configuration Constants (in common.h)
- `TILE_SIZE = 16` - Block tile dimension for shared memory
- `BM = 64, BN = 64, BK = 8` - Block-level tiling dimensions
- `TM = 8, TN = 8` - Thread-level register tiling dimensions

### Adding a New Optimization Version
1. Create directory: `src/optimizations/03_new_technique/`
2. Add `gemm.cu` with kernel and main function
3. Add `README.md` explaining the optimization
4. CMake will auto-compile to `build/opt03`

## Dependencies
- CUDA Toolkit 11.0+
- CMake 3.18+
- cuBLAS (linked automatically)
- CUTLASS at `/home/chenmin/3rdparty/cutlass/include`
- Python 3.7+ with pandas, matplotlib, numpy for plotting

## Target Architecture
CUDA compute capability 8.6 (Ampere)