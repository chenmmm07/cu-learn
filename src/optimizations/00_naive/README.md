# 优化 0: Naive GEMM Implementation

## 期望性能

- **GFLOPS**: ~50-100
- **相对 cuBLAS**: ~0.5-1%

## 代码关键点

```cpp
// 边界检查：处理非方阵或非 TILE_SIZE 整数倍的情况
if (row < M && col < N) {
    // 计算点积
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
```

## 使用方法

```bash
# 编译
mkdir build && cd build
cmake ..
make opt00_naive

# 运行
./build/opt00_naive [M N K]

# 示例
./build/opt00_naive 2048 2048 2048
```

## 练习题

1. 修改代码，使用不同的 block size（如 8×8, 32×32），观察性能变化
2. 使用 Nsight Compute 分析内存吞吐量和 cache 命中率
3. 为什么 A 矩阵的访问模式比 B 矩阵更高效？
