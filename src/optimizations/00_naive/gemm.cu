// 优化 0: Naive GEMM Implementation
// 最基础的矩阵乘法实现，直接映射到全局内存

#include "common.h"

__global__ void gemm_naive(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    // 每个线程计算C矩阵的一个元素
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (row < M && col < N) {
        float sum = 0.0f;
        // 遍历K维度进行点积运算
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    int M = 2048, N = 2048, K = 2048;
    int warmup = 2, iters = 5, repeats = 3;

    // 命令行参数
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (argc >= 5) warmup = std::atoi(argv[4]);
    if (argc >= 6) iters = std::atoi(argv[5]);
    if (argc >= 7) repeats = std::atoi(argv[6]);

    std::printf("优化 0: Naive GEMM\n");
    std::printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    std::printf("Benchmark config: warmup=%d, iters=%d, repeats=%d\n", warmup, iters, repeats);

    // 分配内存
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    std::vector<float> hA((size_t)M * K);
    std::vector<float> hB((size_t)K * N);
    std::vector<float> hC((size_t)M * N);

    // 初始化矩阵
    gemm_utils::init_matrix(hA, 123);
    gemm_utils::init_matrix(hB, 456);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    // 配置kernel参数
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    // Warmup
    gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 使用 cuBLAS 作为参考
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    int m = N, n = M, k = K;
    int lda = N, ldb = K, ldc = N;

    // cuBLAS参考实现
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, k, &alpha, dB, lda, dA, ldb, &beta, dC, ldc));

    std::vector<float> hC_ref((size_t)M * N);
    CHECK_CUDA(cudaMemcpy(hC_ref.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    // 性能测量
    struct Ctx {
        const float* dA;
        const float* dB;
        float* dC;
        int M, N, K;
        dim3 grid, block;
    } ctx = {dA, dB, dC, M, N, K, grid, block};

    auto kernel_op = [](void* p) {
        Ctx* ctx = (Ctx*)p;
        gemm_naive<<<ctx->grid, ctx->block>>>(
            ctx->dA, ctx->dB, ctx->dC, ctx->M, ctx->N, ctx->K);
        CHECK_CUDA(cudaGetLastError());
    };

    float ms = gemm_utils::measure_median_ms(kernel_op, &ctx, warmup, iters, repeats);
    double gflops = gemm_utils::gflops(M, N, K, ms);

    std::printf("\n=== Performance Results ===\n");
    std::printf("Median time: %.3f ms\n", ms);
    std::printf("Performance: %.2f GFLOPS\n", gflops);

    // 验证正确性
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    bool ok = gemm_utils::allclose(hC, hC_ref);
    std::printf("Correctness: %s\n", ok ? "PASS" : "FAIL");

    // 清理
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUBLAS(cublasDestroy(handle));

    return ok ? 0 : 1;
}
