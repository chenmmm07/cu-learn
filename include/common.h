#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>

// CUDA错误检查宏
#define CHECK_CUDA(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        std::fprintf(stderr, "CUDA Error %s:%d: %s\n",          \
            __FILE__, __LINE__, cudaGetErrorString(err));      \
        std::exit(EXIT_FAILURE);                               \
    }                                                          \
} while(0)

#define CHECK_CUBLAS(call) do {                                \
    cublasStatus_t st = (call);                                 \
    if (st != CUBLAS_STATUS_SUCCESS) {                          \
        std::fprintf(stderr, "cuBLAS Error %s:%d: %d\n",        \
            __FILE__, __LINE__, (int)st);                       \
        std::exit(EXIT_FAILURE);                                \
    }                                                          \
} while(0)

// 配置常量
#define TILE_SIZE 16
#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8
#define PADDING 1

// 工具函数
namespace gemm_utils {

// 初始化矩阵（确定性随机数）
inline void init_matrix(std::vector<float>& h, unsigned seed) {
    std::srand(seed);
    for (size_t i = 0; i < h.size(); i++) {
        h[i] = (float)(std::rand() % 100) / 10.0f;
    }
}

// 验证两个矩阵是否接近（对称的 allclose）
inline bool allclose(const std::vector<float>& a,
                     const std::vector<float>& b,
                     float rtol = 1e-4f, float atol = 1e-2f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        float x = a[i], y = b[i];
        float diff = std::fabs(x - y);
        float scale = std::max(std::fabs(x), std::fabs(y));
        float tol = atol + rtol * scale;
        if (diff > tol) {
            std::fprintf(stderr,
                "Mismatch at %zu: a=%f b=%f diff=%f tol=%f\n",
                i, x, y, diff, tol);
            return false;
        }
    }
    return true;
}

// 计算中位数
inline float median_ms(std::vector<float>& xs) {
    std::sort(xs.begin(), xs.end());
    size_t n = xs.size();
    if (n == 0) return 0.0f;
    if (n % 2) return xs[n / 2];
    return 0.5f * (xs[n/2 - 1] + xs[n/2]);
}

// 计算 GFLOPS
inline double gflops(int M, int N, int K, float ms) {
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double sec = (double)ms * 1e-3;
    return (flops * 1e-9) / sec;
}

// 测量kernel执行时间的中位数
inline float measure_median_ms(void (*op)(void*), void* ctx,
                               int warmup, int iters, int repeats) {
    std::vector<float> ms(repeats);
    for (int r = 0; r < repeats; r++) {
        // warmup
        for (int i = 0; i < warmup; i++) {
            op(ctx);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // timing
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            op(ctx);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float total = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&total, start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        ms[r] = total / iters;
    }
    return median_ms(ms);
}

} // namespace gemm_utils
