cmake_minimum_required(VERSION 3.18)
project(cu-learn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# CUDA 架构配置
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architecture")
endif()

# 查找依赖
find_package(CUDAToolkit REQUIRED)

# 包含目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Optimization 0: Naive
add_executable(opt00_naive src/optimizations/00_naive/gemm.cu)
target_link_libraries(opt00_naive cublas)

# Optimization 1: Shared Memory
add_executable(opt01_shared src/optimizations/01_shared_memory/gemm.cu)
target_link_libraries(opt01_shared cublas)

# Optimization 2: Loop Unrolling
add_executable(opt02_unroll src/optimizations/02_loop_unrolling/gemm.cu)
target_link_libraries(opt02_unroll cublas)

# Optimization 3: Register Blocking
add_executable(opt03_regblock src/optimizations/03_register_blocking/gemm.cu)
target_link_libraries(opt03_regblock cublas)

# Full Benchmark
add_executable(gemm_benchmark src/benchmarks/gemm_bench.cu)
target_link_libraries(gemm_benchmark cublas)

# 安装
install(TARGETS gemm_benchmark opt00_naive opt01_shared opt02_unroll opt03_regblock DESTINATION bin)
install(FILES src/plot/plot.py DESTINATION bin/plot)
