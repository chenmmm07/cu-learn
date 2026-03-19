# CuTe DSL 学习笔记

CuTe (CUDA Tensor Algebra) 是 CUTLASS 提供的张量代数 DSL，用于简化高性能 CUDA kernel 的编写。

---

## 目录

1. [Layout 基础](#1-layout-基础)
2. [Tensor 基础](#2-tensor-基础)
3. [Swizzle 与 composition](#3-swizzle-与-composition)
4. [Tile 与 logical_divide](#4-tile-与-logical_divide)
5. [local_partition](#5-local_partition)
6. [Copy_Atom 与 TiledCopy](#6-copy_atom-与-tiledcopy)
7. [MMA_Atom 与 TiledMMA](#7-mma_atom-与-tiledmma)
8. [GEMM 综合示例](#8-gemm-综合示例)

---

## 1. Layout 基础

Layout 描述了逻辑坐标到线性地址的映射关系。

### 1.1 make_layout

```cpp
#include <cute/layout.hpp>
using namespace cute;

// 方式1: 使用 Shape 和 Stride 模板参数
using Layout1 = Layout<Shape<_16, _32>, Stride<_32, _1>>;
auto layout1 = Layout1{};

// 方式2: 使用 make_layout 函数
auto layout2 = make_layout(Shape<_16, _32>{}, Stride<_32, _1>{});

// 方式3: 使用 GenRowMajor / GenColMajor 生成器
auto layout3 = make_layout(Shape<_16, _32>{}, GenRowMajor{});
auto layout4 = make_layout(Shape<_16, _32>{}, GenColMajor{});
```

### 1.2 Layout 的含义

对于 `Layout<Shape<M, N>, Stride<stride_m, stride_n>>`:
- 逻辑坐标 `(i, j)` 映射到线性地址: `i * stride_m + j * stride_n`
- Row-major: `Stride<N, 1>` → `addr = i*N + j`
- Col-major: `Stride<1, M>` → `addr = i + j*M`

### 1.3 查询 Layout 属性

```cpp
// 获取维度
auto m = size<0>(layout);  // M
auto n = size<1>(layout);  // N
auto total = size(layout); // M * N

// 打印 layout
print_layout(layout);

// 访问元素
int linear_idx = layout(i, j);  // 逻辑坐标 → 线性索引
```

---

## 2. Tensor 基础

Tensor = Pointer + Layout，描述了数据的存储位置和组织方式。

### 2.1 make_gmem_ptr

```cpp
#include <cute/pointer.hpp>

// 创建全局内存指针
int data[128];
auto gmem_ptr = make_gmem_ptr(data);

// 对于 const 数据
const int const_data[128];
auto const_gmem_ptr = make_gmem_ptr(const_data);
```

### 2.2 make_tensor

```cpp
#include <cute/tensor.hpp>

// 方式1: 从指针和 layout 创建
int data[128];
auto layout = make_layout(Shape<_16, _8>{}, GenRowMajor{});
auto tensor = make_tensor(make_gmem_ptr(data), layout);

// 方式2: 简化写法
auto tensor = make_tensor(make_gmem_ptr(data), Shape<_16, _8>{}, GenRowMajor{});

// 访问元素
tensor(i, j) = value;     // 写
auto val = tensor(i, j);  // 读
```

### 2.3 Tensor 操作

```cpp
// 获取 tensor 的属性
auto shape = tensor.shape();   // shape
auto layout = tensor.layout(); // layout
auto ptr = tensor.data();      // pointer

// 获取维度
auto m = size<0>(tensor);
auto n = size<1>(tensor);
auto total = size(tensor);

// 打印 tensor
print_tensor(tensor);
```

---

## 3. Swizzle 与 composition

### 3.1 Swizzle

Swizzle 通过 XOR 操作重排地址，用于消除 shared memory bank conflicts。

```cpp
#include <cute/swizzle.hpp>

// Swizzle<B, M, S> 参数含义:
// B: 混洗的行数 = 2^B
// M: 固定的列数 = 2^M
// S: 混洗位数 = 2^S
auto swizzle = Swizzle<3, 0, 3>{};
```

### 3.2 composition

composition 将两个 layout 组合，生成新的 layout。

```cpp
// 基础 layout
auto base_layout = make_layout(Shape<_16, _32>{}, GenRowMajor{});

// 应用 swizzle
auto swizzled_layout = composition(Swizzle<3, 0, 3>{}, base_layout);

// 对比访问
// base_layout(0, 0) = 0
// swizzled_layout(0, 0) 可能不等于 0（被 swizzle 重排）
```

### 3.3 Swizzled Tensor 示例

```cpp
int a[128], b[128];

// 普通 tensor
auto base_layout = make_layout(Shape<_8, _16>{}, GenRowMajor{});
auto tensor_a = make_tensor(make_gmem_ptr(a), base_layout);

// Swizzled tensor
auto swizzled_layout = composition(Swizzle<2, 1, 2>{}, base_layout);
auto tensor_b = make_tensor(make_gmem_ptr(b), swizzled_layout);

// 两者逻辑坐标相同，但物理存储不同
for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 16; ++j) {
        tensor_a(i, j) = i * 16 + j;
        tensor_b(i, j) = i * 16 + j;  // 逻辑上相同
    }
}
// 但实际内存布局不同，可用 print_tensor 查看
```

---

## 4. Tile 与 logical_divide

### 4.1 make_tile

```cpp
// make_tile 创建一个 tile shape
auto tile = make_tile(Int<4>{}, Int<4>{});  // 4x4 的 tile

// 等价于
auto tile = Shape<_4, _4>{};
```

### 4.2 logical_divide

logical_divide 将大 layout 分割成多个小 tile。

```cpp
// 原始 16x16 layout
auto global_layout = make_layout(Shape<_16, _16>{}, GenRowMajor{});

// 分割成 4x4 的 tiles
auto tile_shape = make_tile(Int<4>{}, Int<4>{});
auto tiled_layout = logical_divide(global_layout, tile_shape);

// tiled_layout 现在是两层结构: (tile内坐标, tile索引)
// 访问: tiled_layout(tile_m, tile_n, inner_i, inner_j)
```

### 4.3 local_tile

local_tile 是 logical_divide 的别名，语义上更清晰。

```cpp
auto tiled_mn = local_tile(global_layout, tile_shape);
// 返回的 layout 可以理解为: (inner_tile, tile_coords)
```

---

## 5. local_partition

local_partition 将 tensor 按线程布局分区，每个线程获得自己的数据片段。

```cpp
// 1D 示例
int data[16];
auto tensor = make_tensor(make_gmem_ptr(data), Shape<_16>{});

auto thread_layout = make_layout(Shape<_4>{});  // 4 个线程

// 获取每个线程的分区
auto partition_0 = local_partition(tensor, thread_layout, Int<0>{});  // thread 0
auto partition_1 = local_partition(tensor, thread_layout, Int<1>{});  // thread 1
// 每个分区有 4 个元素

// 2D 示例
int data2d[64];
auto tensor2d = make_tensor(make_gmem_ptr(data2d), Shape<_8, _8>{});
auto thread_layout_2d = make_layout(Shape<_2, _2>{});  // 2x2 线程网格

auto part_00 = local_partition(tensor2d, thread_layout_2d, Int<0>{}, Int<0>{});  // 线程(0,0)
auto part_11 = local_partition(tensor2d, thread_layout_2d, Int<1>{}, Int<1>{});  // 线程(1,1)
```

---

## 6. Copy_Atom 与 TiledCopy

### 6.1 Copy_Atom

Copy_Atom 封装单个硬件拷贝指令。

```cpp
#include <cute/atom/copy_atom.hpp>

// UniversalCopy: 最简单的拷贝（直接赋值）
using CopyAtom1 = Copy_Atom<UniversalCopy<float>, float>;

// SM75_U32x4_LDSM_N: ldmatrix.x4.m8n8 指令（shared → register）
using CopyAtom2 = Copy_Atom<SM75_U32x4_LDSM_N, cutlass::half_t>;

// SM80_CP_ASYNC: 异步拷贝（global → shared）
using CopyAtom3 = Copy_Atom<SM80_CP_ASYNC, float>;
```

### 6.2 TiledCopy

TiledCopy 将 Copy_Atom tile 化，支持多线程协作。

```cpp
#include <cute/atom/copy_atom.hpp>

using copy_atom = Copy_Atom<UniversalCopy<float>, float>;

// 创建 TiledCopy
auto tiled_copy = make_tiled_copy(
    copy_atom{},
    Layout<Shape<_16, _4>>{}  // 线程布局
);

// 获取线程视图
auto thr_copy = tiled_copy.get_slice(thread_id);

// 分区源和目标 tensor
auto src_partition = thr_copy.partition_S(src_tensor);
auto dst_partition = thr_copy.partition_D(dst_tensor);

// 执行拷贝
copy(tiled_copy, src_partition, dst_partition);
```

### 6.3 常用 Copy_Atom 类型

| Copy_Atom | 功能 | 架构要求 |
|-----------|------|---------|
| `UniversalCopy<T>` | 通用赋值 | 全部 |
| `SM75_U32x4_LDSM_N` | ldmatrix (shared→reg) | SM75+ |
| `SM80_CP_ASYNC` | 异步拷贝 (global→shared) | SM80+ |
| `AutoVectorizingCopy` | 自动向量化 | 全部 |

---

## 7. MMA_Atom 与 TiledMMA

### 7.1 MMA_Atom

MMA_Atom 封装矩阵乘累加指令。

```cpp
#include <cute/atom/mma_atom.hpp>

// SM80 16x8x16 FP32=FP16×FP16+FP32 (TN 布局)
using mma_op = SM80_16x8x16_F32F16F16F32_TN;
using mma_atom = MMA_Atom<mma_op>;

// 其他常见 MMA 操作:
// SM70_16x16x8_F32F16F16F32_TN - Volta
// SM75_16x8x8_F32F16F16F32_TN  - Turing
// SM80_16x8x16_F32F16F16F32_TN - Ampere
```

### 7.2 TiledMMA

TiledMMA 将 MMA_Atom tile 化，支持更大的矩阵块计算。

```cpp
using mma_op = SM80_16x8x16_F32F16F16F32_TN;
using mma_atom = MMA_Atom<mma_op>;

// 不重复
using TiledMMA_1x1x1 = TiledMMA<mma_atom, Layout<Shape<_1, _1, _1>>>;

// 2x2x1 重复
using TiledMMA_2x2x1 = TiledMMA<mma_atom, Layout<Shape<_2, _2, _1>>>;

TiledMMA_2x2x1 mma;
```

### 7.3 partition_fragment

```cpp
auto mma = TiledMMA_2x2x1{};

// 获取线程视图
auto thr_mma = mma.get_slice(thread_id);

// 创建寄存器 fragment
auto tCrA = thr_mma.partition_fragment_A(Shape<_32, _16>{});  // A: M×K
auto tCrB = thr_mma.partition_fragment_B(Shape<_16, _16>{});  // B: K×N
auto tCrC = thr_mma.partition_fragment_C(Shape<_32, _16>{});  // C: M×N (累加器)
```

### 7.4 执行 MMA

```cpp
// C = A × B + C
gemm(mma, tCrA, tCrB, tCrC);
```

---

## 8. GEMM 综合示例

完整的 CuTe GEMM 数据流：

```
Global Memory (gA, gB, gC)
       │
       │ TiledCopy (cp.async) - Global → Shared
       ▼
Shared Memory (sA, sB) [Swizzled Layout]
       │
       │ TiledCopy (ldmatrix) - Shared → Register
       ▼
Register (rA, rB, rC)
       │
       │ TiledMMA (mma.sync) - 计算
       ▼
Result in rC
       │
       │ Store - Register → Global
       ▼
Global Memory (gC)
```

### 代码框架

```cpp
// 1. 定义 layout
auto gA_layout = make_layout(make_shape(M, K), GenRowMajor{});
auto sA_layout = composition(Swizzle<3,0,3>{}, make_layout(Shape<_64, _8>{}));

// 2. 创建 tensor
auto gA = make_tensor(make_gmem_ptr(A_ptr), gA_layout);
auto sA = make_tensor(make_smem_ptr(sA_ptr), sA_layout);

// 3. Global → Shared (cp.async)
using g2s_atom = Copy_Atom<SM80_CP_ASYNC, half_t>;
auto g2s_copy = make_tiled_copy(g2s_atom{}, thread_layout);
auto thr_g2s = g2s_copy.get_slice(tid);
copy(g2s_copy, thr_g2s.partition_S(gA), thr_g2s.partition_D(sA));

// 4. Shared → Register (ldmatrix)
using s2r_atom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
auto s2r_copy = make_tiled_copy(s2r_atom{}, thread_layout);

// 5. MMA 计算
using mma_op = SM80_16x8x16_F32F16F16F32_TN;
auto mma = TiledMMA<MMA_Atom<mma_op>, tile_shape>{};
gemm(mma, rA, rB, rC);

// 6. 写回结果
copy(rC, gC);
```

---

## 参考资源

- [CUTLASS CuTe Documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)
- [CuTe Tutorial (GTC 2023)](https://developer.nvidia.com/gtc/2023/video/s51891)
- 本地示例: `src/demo/cute_test.cu`