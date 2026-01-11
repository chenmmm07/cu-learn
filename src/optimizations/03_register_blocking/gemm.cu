// 优化 3: Register Blocking (Register Tiling)
// 使用寄存器分块大幅提升算术强度

#include "common.h"

__global__ void gemm_regblock(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    // Block-level shared memory tiles
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;  // 0..(BN/TN-1)
    int ty = threadIdx.y;  // 0..(BM/TM-1)

    // 每个线程计算 TM×TN 的子块
    float regC[TM][TN];

    // 初始化累加器
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            regC[i][j] = 0.0f;

    // 寄存器缓存 A 和 B 的片段
    float regA[TM];
    float regB[TN];

    int threadRow = ty * TM;
    int threadCol = tx * TN;

    // 遍历 K 维度的块
    for (int k0 = 0; k0 < K; k0 += BK) {
        // 协作加载 A tile 到 shared memory
        for (int i = 0; i < BM; i += blockDim.y) {
            for (int j = 0; j < BK; j += blockDim.x) {
                int row = by * BM + i + ty;
                int col = k0 + j + tx;
                As[i + ty][j + tx] = (row < M && col < K) ? A[row * K + col] : 0.0f;
            }
        }

        // 协作加载 B tile 到 shared memory
        for (int i = 0; i < BK; i += blockDim.y) {
            for (int j = 0; j < BN; j += blockDim.x) {
                int row = k0 + i + ty;
                int col = bx * BN + j + tx;
                Bs[i + ty][j + tx] = (row < K && col < N) ? B[row * N + col] : 0.0f;
            }
        }

        __syncthreads();

        // 计算当前块
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // 从 shared memory 加载到寄存器
            #pragma unroll
            for (int i = 0; i < TM; i++) regA[i] = As[threadRow + i][kk];
            #pragma unroll
            for (int j = 0; j < TN; j++) regB[j] = Bs[kk][threadCol + j];

            // 更新 regC: 外积展开
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    regC[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // 写回结果到全局内存
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int row = by * BM + threadRow + i;
            int col = bx * BN + threadCol + j;
            if (row < M && col < N) C[row * N + col] = regC[i][j];
        }
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

    std::printf("优化 3: Register Blocking GEMM\n");
    std::printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    std::printf("Block config: BM=%d, BN=%d, BK=%d, TM=%d, TN=%d\n", BM, BN, BK, TM, TN);
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

    dim3 block(BN / TN, BM / TM);  // 8×8 threads per block
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

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
        gemm_regblock<<<ctx->grid, ctx->block>>>(
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
