# 优化 1: Shared Memory Optimization

## 学习目标

理解 Shared Memory 的工作原理，学会使用分块（tiling）技术减少全局内存访问。

## 优化原理

### Shared Memory 特性

- **位置**: 位于 GPU 芯片内，速度远快于全局内存
- **作用域**: 每个 block 内共享
- **延迟**: ~30 cycles（全局内存 ~300 cycles）

### 分块（Tiling）策略

将大矩阵分成小块（tile），每个 block 处理一个 tile：

```
原始访问: 每次迭代从全局内存加载
优化后: 一次加载一个 tile 到 shared memory，重复使用
```

## 代码关键点

### 1. Shared Memory 声明

```cpp
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];
```

- 每个 block 有独立的 shared memory
- 所有线程共享同一块内存

### 2. 协作加载

```cpp
As[threadIdx.y][threadIdx.x] =
    (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
```

- 每个 thread 加载一个元素
- 所有线程协作加载整个 tile

### 3. 同步机制

```cpp
__syncthreads();  // 等待所有线程完成加载
// ... 使用 shared memory 计算 ...
__syncthreads();  // 等待所有线程完成计算
```

**重要**: 访问 shared memory 前必须同步！

### 4. 边界处理

```cpp
(row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
```

- 处理非 TILE_SIZE 整数倍的矩阵
- 填充 0 保证正确性

## 使用方法

```bash
# 编译
cmake --build build --target opt01_shared

# 运行
./build/opt01_shared 2048 2048 2048

# 性能分析
ncu --set full ./build/opt01_shared 2048 2048 2048
```
