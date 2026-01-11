// 优化 2: Loop Unrolling
// 使用循环展开优化指令级并行

#include "common.h"

__global__ void gemm_unrolled(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // 循环展开：减少循环开销，暴露指令级并行
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    int M = 2048, N = 2048, K = 2048;
    int warmup = 2, iters = 5, repeats = 3;

    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (argc >= 5) warmup = std::atoi(argv[4]);
    if (argc >= 6) iters = std::atoi(argv[5]);
    if (argc >= 7) repeats = std::atoi(argv[6]);

    std::printf("优化 2: Loop Unrolling GEMM\n");
    std::printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    std::printf("Benchmark config: warmup=%d, iters=%d, repeats=%d\n", warmup, iters, repeats);

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    std::vector<float> hA((size_t)M * K);
    std::vector<float> hB((size_t)K * N);
    std::vector<float> hC((size_t)M * N);

    gemm_utils::init_matrix(hA, 123);
    gemm_utils::init_matrix(hB, 456);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    float alpha = 1.0f, beta = 0.0f;
    int m = N, n = M, k = K;
    int lda = N, ldb = K, ldc = N;

    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, k, &alpha, dB, lda, dA, ldb, &beta, dC, ldc));

    std::vector<float> hC_ref((size_t)M * N);
    CHECK_CUDA(cudaMemcpy(hC_ref.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    struct Ctx {
        const float* dA;
        const float* dB;
        float* dC;
        int M, N, K;
        dim3 grid, block;
    } ctx = {dA, dB, dC, M, N, K, grid, block};

    auto kernel_op = [](void* p) {
        Ctx* ctx = (Ctx*)p;
        gemm_unrolled<<<ctx->grid, ctx->block>>>(
            ctx->dA, ctx->dB, ctx->dC, ctx->M, ctx->N, ctx->K);
        CHECK_CUDA(cudaGetLastError());
    };

    float ms = gemm_utils::measure_median_ms(kernel_op, &ctx, warmup, iters, repeats);
    double gflops = gemm_utils::gflops(M, N, K, ms);

    std::printf("\n=== Performance Results ===\n");
    std::printf("Median time: %.3f ms\n", ms);
    std::printf("Performance: %.2f GFLOPS\n", gflops);

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    bool ok = gemm_utils::allclose(hC, hC_ref);
    std::printf("Correctness: %s\n", ok ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUBLAS(cublasDestroy(handle));

    return ok ? 0 : 1;
}
