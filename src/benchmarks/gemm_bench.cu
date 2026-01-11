// gemm_bench_multi_cublas.cu
#include "common.h"
// ------------------------- config -------------------------
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif
#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8
// ------------------------- helpers -------------------------
static void init_matrix(std::vector<float>& h, unsigned seed) {
    std::srand(seed);
    for (size_t i = 0; i < h.size(); i++) {
        h[i] = (float)(std::rand() % 100) / 10.0f;
    }
}
// symmetric allclose: |x-y| <= atol + rtol*max(|x|,|y|)
static bool allclose(const std::vector<float>& a,
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
static float median_ms(std::vector<float>& xs) {
    std::sort(xs.begin(), xs.end());
    size_t n = xs.size();
    if (n == 0) return 0.0f;
    if (n % 2) return xs[n / 2];
    return 0.5f * (xs[n/2 - 1] + xs[n/2]);
}
static double gflops(int M, int N, int K, float ms) {
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double sec = (double)ms * 1e-3;
    return (flops * 1e-9) / sec;
}
// Measure an "op()" that launches work on GPU stream 0.
// - warmup launches are not timed
// - timed launches are measured by cudaEvent and averaged over iters
static float measure_median_ms(void (*op)(void*), void* ctx,
                               int warmup, int iters, int repeats) {
    std::vector<float> ms(repeats);
    for (int r = 0; r < repeats; r++) {
        for (int i = 0; i < warmup; i++) {
            op(ctx);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
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
// ------------------------- kernels (V0..V4) -------------------------
__global__ void gemm_v0_naive(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
__global__ void gemm_v1_shared(const float* __restrict__ A,
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
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
__global__ void gemm_v2_unrolled(const float* __restrict__ A,
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
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
__global__ void gemm_v3_register_blocking(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; // 0..(BN/TN-1)
    int ty = threadIdx.y; // 0..(BM/TM-1)
    float regC[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            regC[i][j] = 0.0f;
    float regA[TM];
    float regB[TN];
    int threadRow = ty * TM;
    int threadCol = tx * TN;
    for (int k0 = 0; k0 < K; k0 += BK) {
        // load A tile
        for (int i = 0; i < BM; i += blockDim.y) {
            for (int j = 0; j < BK; j += blockDim.x) {
                int row = by * BM + i + ty;
                int col = k0 + j + tx;
                As[i + ty][j + tx] = (row < M && col < K) ? A[row * K + col] : 0.0f;
            }
        }
        // load B tile
        for (int i = 0; i < BK; i += blockDim.y) {
            for (int j = 0; j < BN; j += blockDim.x) {
                int row = k0 + i + ty;
                int col = bx * BN + j + tx;
                Bs[i + ty][j + tx] = (row < K && col < N) ? B[row * N + col] : 0.0f;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) regA[i] = As[threadRow + i][kk];
            #pragma unroll
            for (int j = 0; j < TN; j++) regB[j] = Bs[kk][threadCol + j];
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
// ------------------------- launch contexts -------------------------
struct KernelCtx {
    const float* dA;
    const float* dB;
    float* dC;
    int M, N, K;
    dim3 grid, block;
    int which; // 0=v0,1=v1,2=v2,3=v3
};
static void op_kernel(void* p) {
    KernelCtx* ctx = (KernelCtx*)p;
    switch (ctx->which) {
        case 0:
            gemm_v0_naive<<<ctx->grid, ctx->block>>>(ctx->dA, ctx->dB, ctx->dC, ctx->M, ctx->N, ctx->K);
            break;
        case 1:
            gemm_v1_shared<<<ctx->grid, ctx->block>>>(ctx->dA, ctx->dB, ctx->dC, ctx->M, ctx->N, ctx->K);
            break;
        case 2:
            gemm_v2_unrolled<<<ctx->grid, ctx->block>>>(ctx->dA, ctx->dB, ctx->dC, ctx->M, ctx->N, ctx->K);
            break;
        case 3:
            gemm_v3_register_blocking<<<ctx->grid, ctx->block>>>(ctx->dA, ctx->dB, ctx->dC, ctx->M, ctx->N, ctx->K);
            break;
        default:
            break;
    }
    CHECK_CUDA(cudaGetLastError());
}
// cuBLAS ctx and op
struct CublasCtx {
    cublasHandle_t handle;
    const float* dA;
    const float* dB;
    float* dC;
    int M, N, K;
};
static void op_cublas(void* p) {
    CublasCtx* ctx = (CublasCtx*)p;
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major C(MxN) = A(MxK)*B(KxN)
    // Compute as column-major: C^T(NxM) = B^T(NxK) * A^T(KxM)
    int m = ctx->N;
    int n = ctx->M;
    int k = ctx->K;
    int lda = ctx->N; // B_col is (N x K)
    int ldb = ctx->K; // A_col is (K x M)
    int ldc = ctx->N; // C_col is (N x M)
    CHECK_CUBLAS(cublasSgemm(ctx->handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                            &alpha,
                             ctx->dB, lda,
                             ctx->dA, ldb,
                            &beta,
                             ctx->dC, ldc));
}
// ------------------------- main -------------------------
int main(int argc, char** argv) {
    // Defaults: square sizes
    int start = 256, stop = 4096, step = 256;
    int warmup = 2;
    int iters = 5;
    int repeats = 4;
    // Usage:
    // ./bench [start stop step warmup iters repeats]
    if (argc >= 4) {
        start = std::atoi(argv[1]);
        stop  = std::atoi(argv[2]);
        step  = std::atoi(argv[3]);
    }
    if (argc >= 5) warmup  = std::atoi(argv[4]);
    if (argc >= 6) iters   = std::atoi(argv[5]);
    if (argc >= 7) repeats = std::atoi(argv[6]);
    std::fprintf(stderr,
        "Benchmark sizes: %d..%d step %d, warmup=%d, iters=%d, repeats=%d\n",
        start, stop, step, warmup, iters, repeats);
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    // CSV header (stdout ONLY)
    std::printf("size,method,ms,GFLOPS,ok\n");
    const char* methods[] = {
        "cublasSgemm",
        "v0_naive",
        "v1_shared",
        "v2_unrolled",
        "v3_regblock"
    };
    for (int sz = start; sz <= stop; sz += step) {
        int M = sz, N = sz, K = sz;
        size_t bytesA = (size_t)M * K * sizeof(float);
        size_t bytesB = (size_t)K * N * sizeof(float);
        size_t bytesC = (size_t)M * N * sizeof(float);
        std::vector<float> hA((size_t)M * K);
        std::vector<float> hB((size_t)K * N);
        init_matrix(hA, 123);
        init_matrix(hB, 456);
        float *dA=nullptr, *dB=nullptr, *dC=nullptr;
        CHECK_CUDA(cudaMalloc(&dA, bytesA));
        CHECK_CUDA(cudaMalloc(&dB, bytesB));
        CHECK_CUDA(cudaMalloc(&dC, bytesC));
        CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
        // --- cuBLAS (reference + timing) ---
        CublasCtx cb{handle, dA, dB, dC, M, N, K};
        float cb_med = measure_median_ms(op_cublas, &cb, warmup, iters, repeats);
        std::vector<float> hC_ref((size_t)M * N);
        CHECK_CUDA(cudaMemcpy(hC_ref.data(), dC, bytesC, cudaMemcpyDeviceToHost));
        std::printf("%d,%s,%.6f,%.2f,1\n", sz, methods[0], cb_med, gflops(M,N,K,cb_med));
        // --- V0..V3 ---
        for (int which = 0; which <= 3; which++) {
            KernelCtx kc;
            kc.dA = dA; kc.dB = dB; kc.dC = dC;
            kc.M = M; kc.N = N; kc.K = K;
            kc.which = which;
            if (which == 0) { // v0 naive
                kc.block = dim3(16, 16);
                kc.grid  = dim3((N + kc.block.x - 1) / kc.block.x,
                                (M + kc.block.y - 1) / kc.block.y);
            } else if (which == 3) { // v3 register blocking
                kc.block = dim3(BN / TN, BM / TM); // (8,8)
                kc.grid  = dim3((N + BN - 1) / BN,
                                (M + BM - 1) / BM);
            } else { // v1 shared, v2 unrolled
                kc.block = dim3(TILE_SIZE, TILE_SIZE);
                kc.grid  = dim3((N + TILE_SIZE - 1) / TILE_SIZE,
                                (M + TILE_SIZE - 1) / TILE_SIZE);
            }
            float med = measure_median_ms(op_kernel, &kc, warmup, iters, repeats);
            std::vector<float> hC((size_t)M * N);
            CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
            bool ok = allclose(hC, hC_ref, 1e-4f, 1e-2f);
            std::printf("%d,%s,%.6f,%.2f,%d\n",
                        sz, methods[which + 1], med, gflops(M,N,K,med), ok ? 1 : 0);
        }
        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));
    }
    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}
