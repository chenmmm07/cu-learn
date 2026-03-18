# AGENTS.md - Guide for AI Coding Agents

This document provides essential information for AI coding agents working on the cu-learn CUDA GEMM optimization project.

## Project Overview

cu-learn is a CUDA-based project for learning GEMM (General Matrix Multiply) optimization techniques. It contains multiple optimization implementations progressing from naive to register blocking approaches, with performance benchmarking against cuBLAS.

## Build Commands

```bash
# Build the project
mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Clean build
rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)

# Build specific target
make opt00  # Build only opt00

# Install (optional)
make install
```

## Running Tests and Benchmarks

### Run Single Optimization Version

```bash
# Basic usage
./build/opt00 <M> <N> <K> [warmup] [iters] [repeats]

# Examples
./build/opt00 2048 2048 2048           # Naive GEMM
./build/opt01 2048 2048 2048           # Shared Memory
./build/opt02 2048 2048 2048           # Register Blocking

# With custom benchmark config
./build/opt00 1024 1024 1024 5 10 7   # Custom warmup/iters/repeats

# Using the convenience script
./examples/single_run.sh 0 2048 2048 2048 10 50 9
```

### Run Full Benchmark Suite

```bash
# Default: sizes 256..4096 step 256
./build/gemm_benchmark 256 4096 256

# Custom range
./build/gemm_benchmark <start> <stop> <step> [warmup] [iters] [repeats]

# Using the convenience script
./examples/full_benchmark.sh 512 2048 256
```

### Performance Profiling with Nsight

```bash
# Nsight Compute - detailed kernel analysis
ncu --set full -o report ./build/opt02 2048 2048 2048
ncu-ui report.ncu-rep

# Nsight Systems - system-level profiling
nsys profile --output=report ./build/gemm_benchmark 256 2048 256
```

## Code Style Guidelines

### File Structure

- Optimization implementations: `src/optimizations/<number>_<name>/gemm.cu`
- Benchmark code: `src/benchmarks/`
- Local test files: `src/localtest/`
- Common headers: `include/common.h`

### Naming Conventions

```cpp
// Kernel functions: snake_case with descriptive prefix
__global__ void gemm_naive(...)
__global__ void gemm_shared(...)
__global__ void gemm_regblock(...)

// Constants: UPPER_CASE macros
#define TILE_SIZE 16
#define BM 64
#define BN 64

// Variables: snake_case
int thread_row, block_col;
float sum = 0.0f;

// Structs/Contexts: PascalCase
struct KernelCtx { ... };

// Utility namespace: snake_case
namespace gemm_utils { ... }
```

### CUDA-Specific Patterns

```cpp
// Always use __restrict__ for pointer parameters
__global__ void kernel(const float* __restrict__ A,
                       float* __restrict__ C)

// Use #pragma unroll for small fixed-size loops
#pragma unroll
for (int i = 0; i < TM; i++) {
    // ...
}

// Always check CUDA errors with macros
CHECK_CUDA(cudaMalloc(&dA, bytes));
CHECK_CUBLAS(cublasCreate(&handle));

// Initialize pointers to nullptr
float *dA = nullptr, *dB = nullptr;

// Use size_t for large memory calculations
size_t bytes = (size_t)M * N * sizeof(float);
```

### Error Handling

```cpp
// Use provided macros from common.h
CHECK_CUDA(cudaMalloc(...));      // CUDA API calls
CHECK_CUDA(cudaGetLastError());    // After kernel launches
CHECK_CUBLAS(cublasSgemm(...));   // cuBLAS calls

// Return 0 for success, 1 for failure (based on correctness check)
return ok ? 0 : 1;
```

### Memory Management

```cpp
// Host vectors
std::vector<float> hA((size_t)M * K);

// Device memory - always initialize to nullptr, always check
float *dA = nullptr;
CHECK_CUDA(cudaMalloc(&dA, bytes));

// Always free at the end
CHECK_CUDA(cudaFree(dA));
```

### Performance Measurement Pattern

```cpp
// Use the measure_median_ms utility for consistent timing
struct Ctx {
    const float* dA;
    const float* dB;
    float* dC;
    int M, N, K;
    dim3 grid, block;
} ctx = {dA, dB, dC, M, N, K, grid, block};

auto kernel_op = [](void* p) {
    Ctx* ctx = (Ctx*)p;
    my_kernel<<<ctx->grid, ctx->block>>>(...);
    CHECK_CUDA(cudaGetLastError());
};

float ms = gemm_utils::measure_median_ms(kernel_op, &ctx, warmup, iters, repeats);
```

### Correctness Verification

```cpp
// Always verify against cuBLAS reference
std::vector<float> hC_ref((size_t)M * N);
// ... run cuBLAS and copy result to hC_ref ...

bool ok = gemm_utils::allclose(hC, hC_ref);
std::printf("Correctness: %s\n", ok ? "PASS" : "FAIL");
```

### Output Formatting

```cpp
// Use std::printf for consistent output
std::printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
std::printf("Median time: %.3f ms\n", ms);
std::printf("Performance: %.2f GFLOPS\n", gflops);
```

### Shared Memory and Synchronization

```cpp
// Declare shared memory tiles
__shared__ float As[TILE_SIZE][TILE_SIZE];

// Always synchronize after loading shared memory
__syncthreads();

// And before next iteration
__syncthreads();
```

### Thread/Block Indexing

```cpp
// Standard pattern
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// Always check bounds
if (row < M && col < N) {
    // ...
}
```

## Project-Specific Notes

- This is an educational project for learning CUDA GEMM optimizations
- Each optimization step builds on the previous one
- Performance is measured as GFLOPS and compared against cuBLAS
- Use cuBLAS as the reference for correctness verification
- The benchmark outputs CSV format for plotting

## Dependencies

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compiler
- cuBLAS (part of CUDA Toolkit)
- Python 3.7+ with pandas, matplotlib, numpy (for plotting)
- CUTLASS (included via external path in CMakeLists.txt)