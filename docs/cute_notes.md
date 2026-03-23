# CuTe DSL 学习笔记

CuTe (CUDA Tensor Algebra) 是 CUTLASS 提供的张量代数 DSL，用于用更清晰的方式描述矩阵布局、切片、分区和数据搬运。

---

> 面向初学者的小提示：CuTe 的核心不是记很多 API，而是先建立这三个概念：
> 1) 用 `Layout` 描述逻辑坐标到线性地址的映射；
> 2) 用 `Tensor` 把指针和布局绑定起来，再用 `(i, j, ...)` 访问；
> 3) 用 `logical_divide`、`local_tile`、`local_partition` 把大问题切成小视图，再配合 `copy` 和 `gemm` 完成数据搬运与计算。

## 目录

1. [先理解 CuTe 的三个核心对象](#1-先理解-cute-的三个核心对象)
2. [Layout 基础](#2-layout-基础)
3. [Tensor 基础](#3-tensor-基础)
4. [Swizzle 与 composition](#4-swizzle-与-composition)
5. [Tile 与 logical_divide](#5-tile-与-logical_divide)
6. [local_tile 与 local_partition](#6-local_tile-与-local_partition)
7. [Copy_Atom 与 TiledCopy](#7-copy_atom-与-tiledcopy)
8. [MMA_Atom 与 TiledMMA](#8-mma_atom-与-tiledmma)
9. [GEMM 综合示例](#9-gemm-综合示例)

---

## 1. 先理解 CuTe 的三个核心对象

建议先记住这条主线：`Layout` 负责“怎么算地址”，`Tensor` 负责“怎么访问数据”，`Tile/Partition` 负责“怎么把大矩阵切给 block / thread”。

- `Layout`：逻辑坐标到线性地址的映射规则。
- `Tensor`：指针 + Layout 绑定后的对象，像“带坐标系的数组”。
- `Tile / Partition`：把矩阵切成块，再把块分给 CTA 或线程。

如果是第一次接触 CuTe，我自己一般也会先用这三件事去对照后面的 API。

---

## 2. Layout 基础

Layout 这部分我理解成：先说清楚“坐标怎么落到内存上”。

### 2.1 make_layout

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

### 2.2 Layout 的含义

对于 `Layout<Shape<M, N>, Stride<stride_m, stride_n>>`:
- 逻辑坐标 `(i, j)` 映射到线性地址: `i * stride_m + j * stride_n`
- Row-major: `Stride<N, 1>` → `addr = i*N + j`
- Col-major: `Stride<1, M>` → `addr = i + j*M`

这里的 `i/j` 是逻辑坐标，`Stride` 决定的是这些坐标在内存里的展开方式。这个区别一开始我容易混，后面基本就按“坐标”和“地址”两层去看。

### 2.3 查询 Layout 属性

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

### 2.4 make_coord 创建坐标

`make_coord` 主要是把坐标写得更明确，尤其是遇到嵌套 layout 的时候会更好读。

```cpp
#include <cute/coordinate.hpp>

// 1D 坐标
auto coord1d = make_coord(5);  // 单个坐标

// 2D 坐标
auto coord2d = make_coord(2, 3);  // (row=2, col=3)

// 嵌套坐标 (用于 tiled layout)
auto nested = make_coord(make_coord(1, 2), make_coord(3, 4));
// 结构: ((inner_row, tile_row), (inner_col, tile_col))

// 用坐标访问 layout
auto layout = make_layout(Shape<_8, _16>{}, GenRowMajor{});
layout(make_coord(2, 3));  // 等价于 layout(2, 3)
```

嵌套坐标通常是在 `logical_divide` 之后才会更常见，我这里先记住“内层坐标 + 外层坐标”这个结构就够了：

```cpp
// 12x16 layout 分割成 4x8 的 tiles
auto global = make_layout(Shape<_12, _16>{}, GenRowMajor{});
auto tile = make_tile(Int<4>{}, Int<8>{});
auto tiled = logical_divide(global, tile);

// 访问: tiled(make_coord(inner_row, tile_row), make_coord(inner_col, tile_col))
tiled(make_coord(0, 0), make_coord(0, 0));  // tile(0,0) 内的 (0,0) -> 0
tiled(make_coord(0, 1), make_coord(0, 0));  // tile(1,0) 内的 (0,0) = row4,col0 -> 64
tiled(make_coord(0, 0), make_coord(0, 1));  // tile(0,1) 内的 (0,0) = row0,col8 -> 8
```

---

## 3. Tensor 基础

Tensor 可以先粗暴地记成 `Pointer + Layout`，也就是“指向哪块内存”加上“怎么理解这块内存”。

### 3.1 make_gmem_ptr

```cpp
#include <cute/pointer.hpp>

// 创建全局内存指针
int data[128];
auto gmem_ptr = make_gmem_ptr(data);

// 对于 const 数据
const int const_data[128];
auto const_gmem_ptr = make_gmem_ptr(const_data);
```

### 3.2 make_tensor

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

### 3.3 Tensor 操作

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

## 4. Swizzle 与 composition

### 4.1 Swizzle

这一段我一开始看得比较晕，后来把它当成“为了让 shared memory 更顺手而做的重排”就好理解一些。

Swizzle 通过 XOR 重排地址，常见目的就是减少 shared memory bank conflicts。

```cpp
#include <cute/swizzle.hpp>

// Swizzle<B, M, S> 参数含义:
// B: 混洗的行数 = 2^B
// M: 固定的列数 = 2^M
// S: 混洗位数 = 2^S
auto swizzle = Swizzle<3, 0, 3>{};
```

### 4.2 composition

composition 可以理解成把一个变换叠到 layout 上，最后得到一个新的 layout。

```cpp
// 基础 layout
auto base_layout = make_layout(Shape<_16, _32>{}, GenRowMajor{});

// 应用 swizzle
auto swizzled_layout = composition(Swizzle<3, 0, 3>{}, base_layout);

// 对比访问
// base_layout(0, 0) = 0
// swizzled_layout(0, 0) 可能不等于 0（被 swizzle 重排）
```

### 4.3 Swizzled Tensor 示例

```cpp
int a[48], b[48];

// 普通 tensor
auto base_layout = make_layout(Shape<_4, _12>{}, GenRowMajor{});
auto tensor_a = make_tensor(make_gmem_ptr(a), base_layout);

// Swizzled tensor
auto swizzled_layout = composition(Swizzle<2, 1, 2>{}, base_layout);
auto tensor_b = make_tensor(make_gmem_ptr(b), swizzled_layout);

// 两者逻辑坐标看起来一样，但物理存储会不一样
for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 12; ++j) {
        tensor_a(i, j) = j;
        tensor_b(i, j) = j;  // 逻辑上相同
    }
}
// 但实际内存布局不同，可用 print_tensor 查看
```

---

## 5. Tile 与 logical_divide

### 5.1 make_tile

`make_tile` 这部分我会先把它理解成“切块的尺寸”。

```cpp
#include <cute/layout.hpp>

// 2D tile: 4行 x 8列
auto tile_2d = make_tile(Int<4>{}, Int<8>{});

// 3D tile: 2x3x4
auto tile_3d = make_tile(Int<2>{}, Int<3>{}, Int<4>{});

// 查询 tile 属性
auto rows = size<0>(tile_2d);      // 4
auto cols = size<1>(tile_2d);      // 8
auto total = size(tile_2d);        // 32 (总元素数)

// make_tile 返回的是 Shape，可以直接用于其他操作
auto layout = make_layout(tile_2d, GenRowMajor{});  // 创建 4x8 layout
```

### 5.2 logical_divide

`logical_divide` 就是在 layout 这层把大块拆成小块，结果会变成嵌套结构。

```cpp
// 1D 示例: 12 元素分割成 4 元素的 tiles
auto layout_1d = make_layout(Shape<_12>{});
auto tile_1d = make_tile(Int<4>{});
auto tiled_1d = logical_divide(layout_1d, tile_1d);

// 原始 layout: (_12):(_1)
// 分割后 layout: ((_4,_3)):((_1,_4))
// 结构: (_4,_3) -> 内层 4 元素, 外层 3 个 tile

// 2D 示例: 12x16 分割成 4x8 的 tiles
auto global_layout = make_layout(Shape<_12, _16>{}, GenRowMajor{});
auto tile_shape = make_tile(Int<4>{}, Int<8>{});
auto tiled_layout = logical_divide(global_layout, tile_shape);

// 结果: 3x2 个 tile (12/4=3, 16/8=2)
// 结构: ((_4,_3),(_8,_2)) - 嵌套的 Shape
```

### 5.3 logical_divide 2D 实际示例

```cpp
auto global_layout = make_layout(Shape<_12, _16>{}, GenRowMajor{});
auto tile_shape = make_tile(Int<4>{}, Int<8>{});
auto tiled_layout = logical_divide(global_layout, tile_shape);

// tiled_layout 结构: ((_4,_3),(_8,_2)):((_16,_64),(_1,_8))
// 第一组 (_4,_3): inner_row in [0,4), tile_row in [0,3)
// 第二组 (_8,_2): inner_col in [0,8), tile_col in [0,2)

// 访问方式: tiled_layout(make_coord(inner_row, tile_row), make_coord(inner_col, tile_col))
tiled_layout(make_coord(0, 0), make_coord(0, 0));  // tile(0,0)内(0,0) -> 0
tiled_layout(make_coord(0, 1), make_coord(0, 0));  // tile(1,0)内(0,0) = row4,col0 -> 64
tiled_layout(make_coord(0, 0), make_coord(0, 1));  // tile(0,1)内(0,0) = row0,col8 -> 8
tiled_layout(make_coord(3, 2), make_coord(7, 1));  // tile(2,1)内(3,7) = row11,col15 -> 191
```

### 5.4 zipped_divide

zipped_divide 将内层和外层坐标"拉链"合并，生成扁平化的索引结构。

#### 与 logical_divide 的关键区别

对于 12x16 矩阵分成 4x8 tiles：

```cpp
auto global_layout = make_layout(Shape<_12, _16>{}, GenRowMajor{});
auto tile_shape = make_tile(Int<4>{}, Int<8>{});

auto tiled = logical_divide(global_layout, tile_shape);
// 结果: ((_4,_3),(_8,_2)):((_16,_64),(_1,_8))
// 访问: tiled(make_coord(inner_row, tile_row), make_coord(inner_col, tile_col))

auto zipped = zipped_divide(global_layout, tile_shape);
// 结果: ((_4,_8),(_3,_2)):((_16,_1),(_64,_8))
// 访问: zipped(inner_idx, tile_idx)
```

**结构对比：**

| 属性 | logical_divide | zipped_divide |
|------|----------------|---------------|
| 内层 Shape | `(_4, _8)` 分开 | `(_4, _8)` 按顺序拉平 |
| 外层 Shape | `(_3, _2)` 分开 | `(_3, _2)` 按顺序拉平 |
| 坐标形式 | `((inner_row, tile_row), (inner_col, tile_col))` | `(inner_idx, tile_idx)` |
| 内层范围 | inner_row: 0..3, inner_col: 0..7 | inner_idx: 0..31 |
| 外层范围 | tile_row: 0..2, tile_col: 0..1 | tile_idx: 0..5 |

#### 为什么访问方式不同？

**关键：Shape 的组织方式不同！**

```
logical_divide Shape: ((_4,_3), (_8,_2))
                        ^^^^^^^^  ^^^^^^^^
                        行维度     列维度
每个 () 内保留 inner/tile 分离：
  (_4,_3) = (inner_row=4, tile_row=3)
  (_8,_2) = (inner_col=8, tile_col=2)
→ 需要 make_coord(inner, tile) 访问每个维度

zipped_divide Shape: ((_4,_8), (_3,_2))
                        ^^^^^^^^  ^^^^^^^^
                        inner合并  tile合并
"zipped" 把同类维度拉在一起：
  (_4,_8) = inner_row × inner_col = 32 (所有 inner 合并)
  (_3,_2) = tile_row × tile_col = 6   (所有 tile 合并)
→ 只需要 (inner_idx, tile_idx) 两个标量
```

**比喻：**
- `logical_divide` 按**维度**组织：行相关放一起，列相关放一起
- `zipped_divide` 按**类型**组织：所有 inner 放一起，所有 tile 放一起

#### zipped_divide 计算公式

**这里先记住：`zipped_divide` 的标量坐标展开方式和 `logical_divide` 不一样，调试时最好结合实际输出看。**

```cpp
// zipped(inner_idx, tile_idx) 的计算过程:
// 假设 tile_shape = (height, width) = (4, 8), num_tiles = (3, 2)

// 1. inner_idx -> (inner_row, inner_col) [column-major]
inner_row = inner_idx % tile_height;   // tile_height = 4
inner_col = inner_idx / tile_height;   // 除以行数

// 2. tile_idx -> (tile_row, tile_col) [column-major]
tile_row = tile_idx % num_tile_rows;   // num_tile_rows = 3
tile_col = tile_idx / num_tile_rows;   // 除以 tile 行数

// 3. 全局坐标
global_row = inner_row + tile_row * tile_height;
global_col = inner_col + tile_col * tile_width;

// 4. 线性索引 (global layout 是 row-major: stride_row=16, stride_col=1)
linear_idx = global_row * 16 + global_col;
```

#### 计算示例

```cpp
// zipped(9, 0) 的计算:
// inner_idx=9: inner_row=9%4=1, inner_col=9/4=2
// tile_idx=0:  tile_row=0%3=0, tile_col=0/3=0
// global_row=1+0*4=1, global_col=2+0*8=2
// linear_idx=1*16+2=18 ✓

// zipped(31, 5) 的计算 (最后一个元素):
// inner_idx=31: inner_row=31%4=3, inner_col=31/4=7
// tile_idx=5:   tile_row=5%3=2,   tile_col=5/3=1
// global_row=3+2*4=11, global_col=7+1*8=15
// linear_idx=11*16+15=191 ✓
```

#### 使用场景

**logical_divide 适用于：**
- 需要按行列语义访问 tile 内元素
- 保持 2D 结构便于后续操作
- 例如：加载 tile 的某一行到寄存器

**zipped_divide 适用于：**
- 需要线性遍历 tile 内所有元素
- 与硬件指令配合（如 ldmatrix 按行加载）
- 例如：遍历 tile 内每个元素执行操作

```cpp
// 使用 zipped_divide 遍历 tile 内元素
for (int i = 0; i < size<0>(zipped); ++i) {
    // 处理第 i 个元素
    auto val = zipped(i, tile_id);
}
```

## 6. local_tile 与 local_partition

`local_tile` 和 `local_partition` 都是在“切片”，但前者偏 block 视角，后者偏 thread 视角。

### 6.1 local_tile

`local_tile` 更像是“直接把某个 block 的那一块切出来”。

```cpp
#include <cute/tensor.hpp>

// 创建全局 tensor
int data[192];
auto g_tensor = make_tensor(make_gmem_ptr(data), Shape<_12, _16>{});

// 定义 tile 形状
auto tile_shape = make_tile(Int<4>{}, Int<8>{});  // 4x8 tile

// 获取指定 tile 的视图
auto tile_00 = local_tile(g_tensor, tile_shape, make_coord(0, 0));  // 第(0,0)个tile
auto tile_21 = local_tile(g_tensor, tile_shape, make_coord(2, 1));  // 第(2,1)个tile

// tile_00 的 shape = (4, 8)，对应 rows [0,4), cols [0,8)
// tile_21 的 shape = (4, 8)，对应 rows [8,12), cols [8,16)
```

#### 参数说明

```cpp
local_tile(tensor, tile_shape, tile_coord)
```

| 参数 | 说明 |
|------|------|
| `tensor` | 输入 tensor |
| `tile_shape` | tile 形状，如 `make_tile(Int<4>{}, Int<8>{})` |
| `tile_coord` | tile 坐标，如 `make_coord(tile_row, tile_col)` |

#### 与 logical_divide 的区别

| 操作 | logical_divide | local_tile |
|------|----------------|------------|
| 输入 | Layout | Tensor |
| 输出 | 完整的分割后 layout | 指定 tile 的 tensor 视图 |
| 访问方式 | 需要额外用坐标访问 | 直接返回目标 tile |
| 典型场景 | 需要访问多个 tile | 获取单个 block 数据 |

#### 在 kernel 中的典型用法

```cpp
__global__ void gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // 创建全局 tensor
    auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K));
    auto gB = make_tensor(make_gmem_ptr(B), make_shape(K, N));
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(M, N));

    // 定义 block tile 形状
    auto cta_tile = make_tile(Int<64>{}, Int<64>{});  // 64x64 block

    // 获取当前 block 负责的 tile
    auto tA = local_tile(gA, cta_tile, make_coord(blockIdx.x, _));  // A 的 K 维度完整保留
    auto tB = local_tile(gB, cta_tile, make_coord(_, blockIdx.y));  // B 的 K 维度完整保留
    auto tC = local_tile(gC, cta_tile, make_coord(blockIdx.x, blockIdx.y));

    // tC 的 shape = (64, 64)，当前 block 的 C 矩阵块
}
```

---

### 6.2 local_partition

`local_partition` 一般理解成“按线程把数据分掉”，每个线程拿到自己的那一份。

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
int data2d[96];
auto tensor2d = make_tensor(make_gmem_ptr(data2d), Shape<_6, _16>{});
auto thread_layout_2d = make_layout(Shape<_2, _4>{});  // 2x4 线程网格

auto part_00 = local_partition(tensor2d, thread_layout_2d, Int<0>{}, Int<0>{});  // 线程(0,0)
auto part_13 = local_partition(tensor2d, thread_layout_2d, Int<1>{}, Int<3>{});  // 线程(1,3)

// Thread(0,0) 负责 (row 0-2, col 0-4) 的元素
// Thread(1,3) 负责 (row 3-5, col 12-16) 的元素
```

### 6.3 local_partition 与 logical_divide 的关系

`local_partition` 的思路可以先粗略记成：先切，再取对应线程的那一份。

```cpp
// local_partition 的内部逻辑 (伪代码)
template <class Tensor, class Layout, class... Coords>
auto local_partition(Tensor&& tensor, Layout&& layout, Coords... coords) {
    auto divided = logical_divide(tensor, layout);
    // 返回指定坐标的内层 slice
    return make_tensor(divided.data(), get<1>(divided.layout()));
}
```

**为什么需要编译时常量：**

1. **类型安全**：CuTe 的 Layout 类型在编译期完全确定，不同线程 ID 对应的返回类型可能不同
2. **编译器优化**：编译时常量更容易让编译器做内联、展开和消分支
3. **零开销抽象**：这类访问模式更容易直接变成高效机器码

**实际 kernel 中的用法：**

在 CUDA kernel 中，`threadIdx` 是运行时值。CuTe 提供了 `get_slice()` 方法处理这种情况：

```cpp
// 在 kernel 中
__global__ void my_kernel(float* data, int N) {
    // TiledCopy 或 TiledMMA 提供 get_slice 方法
    auto thr_copy = tiled_copy.get_slice(threadIdx.x);  // 运行时索引
    auto src_part = thr_copy.partition_S(src_tensor);
    auto dst_part = thr_copy.partition_D(dst_tensor);

    copy(tiled_copy, src_part, dst_part);
}
```

**等价性示例：**

```cpp
// 这两种写法效果相同：
auto partition = local_partition(tensor, thread_layout, Int<1>{});

// 等价于：
auto divided = logical_divide(tensor, thread_layout);
// 分割后结构: (inner, outer)
// inner: 每个线程的数据
// outer: 线程数量
auto slice = divided(make_coord(_, 1));  // 取第 1 个线程的数据
```

### 5.2 local_tile vs local_partition

两者都是分割 tensor，但用途和结果完全不同。
`local_tile` 偏“块视角”，`local_partition` 偏“线程视角”。

#### 对比示例

```cpp
// 8x8 tensor
int data[64];
auto tensor = make_tensor(make_gmem_ptr(data), Shape<_8, _8>{}, GenRowMajor{});

// local_tile: 按 4x4 tile 分割
auto tile_shape = make_tile(Int<4>{}, Int<4>{});
auto tile_00 = local_tile(tensor, tile_shape, make_coord(0, 0));
// 结果: 4x4 tensor, stride=(8,1) -> 连续存储
//       0  1  2  3
//       8  9 10 11
//      16 17 18 19
//      24 25 26 27

// local_partition: 按 2x2 线程网格分割
auto thread_layout = make_layout(Shape<_2, _2>{});
auto thr0 = local_partition(tensor, thread_layout, Int<0>{});
// 结果: 4x4 tensor, stride=(16,2) -> 间隔存储
//       0  2  4  6
//      16 18 20 22
//      32 34 36 38
//      48 50 52 54
```

#### 关键区别

| 属性 | local_tile | local_partition |
|------|------------|-----------------|
| **分割依据** | tile_shape (元素个数) | thread_layout (线程数量) |
| **元素分布** | 连续存储 (stride 小) | 分散存储 (stride 大) |
| **使用场景** | block 级并行 | thread 级并行 |
| **典型用法** | 获取 block 的数据 | 获取 thread 的数据 |

#### 图解

```
原始 8x8 tensor:          local_tile (4x4):       local_partition (2x2):
 0  1  2  3  4  5  6  7    tile(0,0): [0-3,0-3]    Thread 0: 0,2,4,6,...
 8  9 10 11 12 13 14 15    tile(0,1): [0-3,4-7]    Thread 1: 8,10,12,...
16 17 18 19 20 21 22 23    tile(1,0): [4-7,0-3]    Thread 2: 1,3,5,7,...
24 25 26 27 28 29 30 31    tile(1,1): [4-7,4-7]    Thread 3: 9,11,13,...
32 33 34 35 36 37 38 39
40 41 42 43 44 45 46 47
48 49 50 51 52 53 54 55
56 57 58 59 60 61 62 63
```

#### 典型配合使用

```cpp
// GEMM kernel 中的典型模式
__global__ void gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // 1. 先用 local_tile 获取当前 block 的数据
    auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K));
    auto block_A = local_tile(gA, BlockShape{}, make_coord(blockIdx.x, _));

    // 2. 再用 local_partition 分给当前 thread
    auto thr_A = local_partition(block_A, ThreadLayout{}, threadIdx.x);

    // 现在 thr_A 是当前线程需要处理的数据
}
```

---

## 7. Copy_Atom 与 TiledCopy

### 6.1 Copy_Atom

Copy_Atom 可以先理解为底层搬数据的基本单元。

```cpp
#include <cute/atom/copy_atom.hpp>

// 基本模板参数
template <class CopyOp, class T>
struct Copy_Atom;

// 示例
using CopyAtom1 = Copy_Atom<UniversalCopy<float>, float>;
```

**模板参数说明：**

| 参数 | 说明 | 示例 |
|------|------|------|
| `CopyOp` | 拷贝操作类型 | `UniversalCopy<T>`, `SM80_CP_ASYNC` |
| `T` | 数据类型 | `float`, `half_t`, `int` |

**常用 Copy_Atom 类型：**

| Copy_Atom | 功能 | 架构要求 |
|-----------|------|---------|
| `UniversalCopy<T>` | 通用赋值 | 全部 |
| `SM75_U32x4_LDSM_N` | ldmatrix (shared→reg) | SM75+ |
| `SM80_CP_ASYNC` | 异步拷贝 (global→shared) | SM80+ |
| `AutoVectorizingCopy` | 自动向量化 | 全部 |

### 6.2 TiledCopy

TiledCopy 是把 Copy_Atom 再组织一下，让多线程能协作做拷贝。

**创建 TiledCopy：**

```cpp
#include <cute/atom/copy_atom.hpp>

auto tiled_copy = make_tiled_copy(
    Copy_Atom<UniversalCopy<float>, float>{},  // copy_atom
    Layout<Shape<_16, _4>>{}                    // thread_layout
);
```

**make_tiled_copy 参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `copy_atom` | `Copy_Atom<CopyOp, T>` | 底层拷贝操作 |
| `thread_layout` | `Layout<Shape<...>>` | 线程布局，决定数据如何分布到线程 |

**TiledCopy 完整使用流程：**

```cpp
// 1. 创建 TiledCopy
using copy_atom = Copy_Atom<UniversalCopy<float>, float>;
auto tiled_copy = make_tiled_copy(copy_atom{}, Layout<Shape<_16>>{});

// 2. get_slice 获取当前线程的切片视图 (返回 ThrCopy 类型)
auto thr_copy = tiled_copy.get_slice(threadIdx.x);

// 3. 使用 ThrCopy 的 partition_S/partition_D 分区数据
auto src_part = thr_copy.partition_S(src_tensor);  // 源数据分区
auto dst_part = thr_copy.partition_D(dst_tensor);  // 目标数据分区

// 4. 执行拷贝
copy(tiled_copy, src_part, dst_part);
```

**TiledCopy 主要方法：**

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `get_slice(thread_id)` | `ThrCopy` | 获取指定线程的切片视图 |
| `size()` | `int` | 总线程数 |

**ThrCopy（get_slice 的返回类型）：**

`ThrCopy` 不是独立的执行接口，而是 `TiledCopy.get_slice()` 的返回类型，封装了当前线程的数据分布信息。

| 方法 | 说明 |
|------|------|
| `partition_S(tensor)` | 返回当前线程负责读取的源数据 |
| `partition_D(tensor)` | 返回当前线程负责写入的目标数据 |

```cpp
// ThrCopy 使用示例
auto thr_copy = tiled_copy.get_slice(threadIdx.x);  // threadIdx.x in [0, 16)

// 每个线程得到 tensor_size / num_threads 个元素
auto src_part = thr_copy.partition_S(src_tensor);
auto dst_part = thr_copy.partition_D(dst_tensor);
```

**ThrCopy 与 local_partition 的等价性：**

```cpp
// 两种方式等价:
auto src_part = tiled_copy.get_slice(tid).partition_S(tensor);
auto thr_data = local_partition(tensor, thread_layout, tid);
```

### 6.3 cute::copy

`cute::copy` 是执行数据拷贝的核心函数。

**函数签名：**

```cpp
// 基本形式
copy(TiledCopy const& tiled_copy, Tensor const& src, Tensor& dst);

// 简化形式 (不使用 TiledCopy)
copy(Tensor const& src, Tensor& dst);
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `tiled_copy` | TiledCopy 对象，定义拷贝方式 |
| `src` | 源 tensor (通过 partition_S 获取) |
| `dst` | 目标 tensor (通过 partition_D 获取) |

**使用示例：**

```cpp
// 示例 1: 使用 TiledCopy 多线程协作拷贝
__global__ void copy_kernel(float* src, float* dst, int N) {
    auto g_src = make_tensor(make_gmem_ptr(src), make_shape(N));
    auto g_dst = make_tensor(make_gmem_ptr(dst), make_shape(N));

    auto tiled_copy = make_tiled_copy(
        Copy_Atom<UniversalCopy<float>, float>{},
        Layout<Shape<_128>>{}
    );

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);
    copy(tiled_copy,
         thr_copy.partition_S(g_src),
         thr_copy.partition_D(g_dst));
}

// 示例 2: 简单拷贝 (单线程或已分区数据)
auto src_tensor = make_tensor(make_gmem_ptr(src), Shape<_64>{});
auto dst_tensor = make_tensor(make_gmem_ptr(dst), Shape<_64>{});
copy(src_tensor, dst_tensor);  // 直接拷贝所有元素
```

**copy 的内部行为：**

```cpp
// copy 内部会:
// 1. 遍历 src tensor 的每个元素
// 2. 使用 TiledCopy 定义的 Copy_Atom 执行实际拷贝
// 3. 根据 ThrCopy 的分区结果，每个线程只处理自己的数据
```

### 6.4 get_slice、partition_S、partition_D 详解

这三个 API 是 TiledCopy 在 kernel 里最常见的一组组合。

#### 概念解释

```cpp
// TiledCopy 定义了多线程协作拷贝的方式
auto tiled_copy = make_tiled_copy(copy_atom{}, thread_layout);

// get_slice(thread_id): 获取指定线程的"切片器"
// - thread_id 可以是运行时值 (如 threadIdx.x)
// - 返回一个对象，包含该线程的数据分布信息
auto thr_copy = tiled_copy.get_slice(threadIdx.x);

// partition_S(tensor): 对源 tensor 进行分区
// - S = Source
// - 返回当前线程负责读取的数据
auto src_part = thr_copy.partition_S(src_tensor);

// partition_D(tensor): 对目标 tensor 进行分区
// - D = Destination
// - 返回当前线程负责写入的数据
auto dst_part = thr_copy.partition_D(dst_tensor);
```

#### 与 local_partition 的关系

```cpp
// 这两种写法等价:
auto src_part = tiled_copy.get_slice(tid).partition_S(tensor);
auto thr_part = local_partition(tensor, thread_layout, tid);
```

#### get_slice 的优势

| 特性 | get_slice | local_partition |
|------|-----------|-----------------|
| thread_id | 支持运行时值 | 需要编译时常量 |
| 分区方式 | 同时提供 S 和 D | 只有单一分区 |
| 使用场景 | 配合 copy() 函数 | 通用数据分区 |

#### 在 kernel 中的完整用法

```cpp
__global__ void copy_kernel(float* src, float* dst, int N) {
    // 1. 创建 tensor
    auto g_src = make_tensor(make_gmem_ptr(src), make_shape(N));
    auto g_dst = make_tensor(make_gmem_ptr(dst), make_shape(N));

    // 2. 定义 TiledCopy
    using copy_atom = Copy_Atom<UniversalCopy<float>, float>;
    auto tiled_copy = make_tiled_copy(copy_atom{}, Layout<Shape<_128>>{});

    // 3. 获取当前线程的分区 (运行时 threadIdx)
    auto thr_copy = tiled_copy.get_slice(threadIdx.x);
    auto src_part = thr_copy.partition_S(g_src);
    auto dst_part = thr_copy.partition_D(g_dst);

    // 4. 执行拷贝
    copy(tiled_copy, src_part, dst_part);
}
```

---

## 8. MMA_Atom 与 TiledMMA

### 7.1 MMA_Atom

MMA_Atom 可以先理解成“单条矩阵乘累加指令”的封装。

```cpp
#include <cute/atom/mma_atom.hpp>

// SM80 16x8x16 FP32=FP16×FP16+FP32 (TN 布局: A转置, B不转置)
using MMA_Op = SM80_16x8x16_F32F16F16F32_TN;
using MMA_Atom_T = MMA_Atom<MMA_Op>;
```

**命名规则解析：**

```
SM80_16x8x16_F32F16F16F32_TN
│      │     │  │  │  │  │
│      │     │  │  │  │  └── Layout: TN (A转置, B不转置)
│      │     │  │  │  └───── C的精度: F32 (FP32)
│      │     │  │  └──────── B的精度: F16 (FP16)
│      │     │  └─────────── A的精度: F16 (FP16)
│      │     └────────────── 输出精度: F32 (FP32)
│      └──────────────────── 形状: M=16, N=8, K=16
└─────────────────────────── 架构: SM80 (Ampere)
```

**常见 MMA_Atom 类型：**

| MMA_Atom | 架构 | 形状 (MxNxK) | A/B精度 | C精度 |
|----------|------|--------------|---------|-------|
| SM80_16x8x16_F32F16F16F32_TN | Ampere | 16x8x16 | FP16 | FP32 |
| SM80_16x8x8_F32F16F16F32_TN | Ampere | 16x8x8 | FP16 | FP32 |
| SM75_16x8x8_F32F16F16F32_TN | Turing | 16x8x8 | FP16 | FP32 |
| SM70_16x16x8_F32F16F16F32_TN | Volta | 16x16x8 | FP16 | FP32 |

**GPU 架构与 MMA 兼容性：**

| 架构 | 计算能力 | GPU 示例 | MMA 支持 |
|------|----------|----------|----------|
| Volta | SM70 | V100 | SM70 MMA |
| Turing | SM75 | RTX 2080, T4 | SM75+ MMA |
| Ampere | SM80 | A100, A30 | SM80+ MMA |
| Ampere | SM86 | RTX 3090/3080/3070/3060 | **兼容 SM80 MMA** |

> **注意：** RTX 3060/3070/3080/3090 使用 SM86 架构，完全兼容 SM80 的所有 MMA 操作。你可以直接使用 `SM80_16x8x16_F32F16F16F32_TN` 等 SM80 MMA 类型。

**Layout 后缀含义：**
- `TN`: A转置(Transposed), B正常(Normal) → A是KxM, B是KxN
- `TT`: A转置, B转置 → A是KxM, B是NxK
- `NT`: A正常, B正常 → A是MxK, B是KxN
- `NN`: A正常, B转置 → A是MxK, B是NxK

**查询 MMA_Atom 属性：**

```cpp
// 获取 MMA_Atom 的维度大小
constexpr int m_dim = size<0>(MMA_Atom_T{});  // M = 16
constexpr int n_dim = size<1>(MMA_Atom_T{});  // N = 8
constexpr int k_dim = size<2>(MMA_Atom_T{});  // K = 16
```

**RTX 3060 12G 推荐配置：**

```cpp
// RTX 3060: SM86 架构, 计算能力 8.6
// 完全兼容 SM80 的 MMA 操作

using MMA_Op = SM80_16x8x16_F32F16F16F32_TN;
using MMA_Atom_T = MMA_Atom<MMA_Op>;

// 推荐 TiledMMA 配置 (适合 3060 的 28 SMs)
using TiledMMA = TiledMMA<MMA_Atom_T, Layout<Shape<_2, _2, _1>>>;
// 计算大小: 32x16x16, 每个 block 需要 128 线程 (4 warps)
```

### 7.2 TiledMMA

TiledMMA 将 MMA_Atom 在 M、N、K 维度上复制，支持更大的矩阵块计算。

**创建 TiledMMA：**

```cpp
#include <cute/atom/mma_atom.hpp>

using MMA_Op = SM80_16x8x16_F32F16F16F32_TN;
using MMA_Atom_T = MMA_Atom<MMA_Op>;

// 1x1x1: 不复制，使用单个 MMA_Atom
using TiledMMA_1x1x1 = TiledMMA<MMA_Atom_T, Layout<Shape<_1, _1, _1>>>;

// 2x2x1: M维度x2, N维度x2, K维度x1
// 实际计算大小: (16*2) x (8*2) x (16*1) = 32x16x16
using TiledMMA_2x2x1 = TiledMMA<MMA_Atom_T, Layout<Shape<_2, _2, _1>>>;

// 2x2x2: M维度x2, N维度x2, K维度x2
// 实际计算大小: (16*2) x (8*2) x (16*2) = 32x16x32
using TiledMMA_2x2x2 = TiledMMA<MMA_Atom_T, Layout<Shape<_2, _2, _2>>>;
```

**TiledMMA 配置与计算大小：**

| TiledMMA Shape | MMA_Atom | 实际 M | 实际 N | 实际 K | 线程数 |
|----------------|----------|--------|--------|--------|--------|
| Shape<_1,_1,_1> | 16x8x16 | 16 | 8 | 16 | 32 |
| Shape<_2,_1,_1> | 16x8x16 | 32 | 8 | 16 | 64 |
| Shape<_1,_2,_1> | 16x8x16 | 16 | 16 | 16 | 64 |
| Shape<_2,_2,_1> | 16x8x16 | 32 | 16 | 16 | 128 |
| Shape<_2,_2,_2> | 16x8x16 | 32 | 16 | 32 | 128 |

**注意：** TiledMMA 需要足够多的线程来覆盖所有 MMA 指令。通常一个 warp（32 线程）是最常见的起点。

### 7.3 partition_fragment

`partition_fragment_*` 这部分我一般理解成：把数据整理成寄存器里要用的 fragment。

**核心 API：**

```cpp
// 在 kernel 中
__global__ void kernel(...) {
    // 创建 TiledMMA 实例
    TiledMMA_2x2x1 tiled_mma;

    // 获取当前线程的 MMA slice
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    // 创建寄存器 fragment (实际分配寄存器内存)
    auto tCrA = thr_mma.partition_fragment_A(sA);  // A fragment
    auto tCrB = thr_mma.partition_fragment_B(sB);  // B fragment
    auto tCrC = thr_mma.partition_fragment_C(gC);  // C accumulator
}
```

**关键概念：**

1. **get_slice(thread_id)**: 获取指定线程的 MMA 视图，决定该线程执行哪些 MMA 指令
2. **partition_fragment_A/B/C**: 创建寄存器 tensor 视图，指向线程私有的寄存器空间
3. **tCrA/tCrB**: 输入 fragments，需要从 shared/global 内存加载数据
4. **tCrC**: 累加器 fragment，需要初始化为 0

**命名约定解析：**

CuTe 使用前缀编码表示 Tensor 的属性：

```
tCrA
│││└─ A矩阵 (GEMM中的A)
││└── r = Register (寄存器存储)
│└─── C = CTA/Current (当前CTA层级)
└──── t = Thread (线程私有)
```

| 前缀 | 含义 | 说明 |
|------|------|------|
| 第1位 | `t` = thread<br>`g` = global<br>`s` = shared | 数据并行层级 |
| 第2位 | `C` = CTA/Current<br>`M` = multi-CTA | CTA 层级标识 |
| 第3位 | `r` = register<br>`m` = memory (gmem/smem) | 存储位置 |
| 第4位 | `A/B/C` | GEMM 矩阵标识 |

**常见命名：**

| 命名 | 全称 | 含义 |
|------|------|------|
| `tCrA` | thread CTA Register A | 线程私有的A矩阵寄存器数据 |
| `tCrB` | thread CTA Register B | 线程私有的B矩阵寄存器数据 |
| `tCrC` | thread CTA Register C | 线程私有的C矩阵累加器 |
| `gA` | global A | 全局内存中的A矩阵 |
| `sA` | shared A | 共享内存中的A矩阵 |
| `tCgC` | thread CTA Global C | 线程负责的C矩阵全局内存视图 |
| `tCsA` | thread CTA Shared A | 线程负责的A矩阵共享内存视图 |

**与 Copy 的 partition 的区别：**

| 操作 | 源/目标 | 用途 | 典型场景 |
|------|---------|------|----------|
| `partition_S/D` | Global/Shared | 数据移动 | cp.async, ldmatrix |
| `partition_fragment_A/B/C` | Register | MMA 计算 | mma.sync |

### 7.4 执行 MMA 计算

**gemm 函数：**

```cpp
// C = A × B + C (累加)
gemm(tiled_mma, tCrA, tCrB, tCrC);
```

**完整 Kernel 示例：**

```cpp
__global__ void mma_kernel(half_t* A, half_t* B, float* C, int M, int N, int K) {
    using namespace cute;

    // 1. 定义 TiledMMA
    using MMA_Op = SM80_16x8x16_F32F16F16F32_TN;
    using MMA_Atom_T = MMA_Atom<MMA_Op>;
    using TiledMMA = TiledMMA<MMA_Atom_T, Layout<Shape<_2, _2, _1>>>;
    TiledMMA tiled_mma;

    // 2. 创建全局内存 tensor
    auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), GenRowMajor{});
    auto gB = make_tensor(make_gmem_ptr(B), make_shape(K, N), GenRowMajor{});
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), GenRowMajor{});

    // 3. 定义 CTA tile (与 TiledMMA 输出匹配: 32x16)
    auto cta_tile = make_tile(Int<32>{}, Int<16>{});
    auto tCgC = local_tile(gC, cta_tile, make_coord(blockIdx.x, blockIdx.y));

    // 4. 获取当前线程的 MMA slice
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    // 5. 创建寄存器 fragment
    // tCrA: 形状与 A tile 匹配 (32x16)
    // tCrB: 形状与 B tile 匹配 (16x16)
    // tCrC: 形状与 C tile 匹配 (32x16)
    auto tCrA = thr_mma.partition_fragment_A(gA);
    auto tCrB = thr_mma.partition_fragment_B(gB);
    auto tCrC = thr_mma.partition_fragment_C(tCgC);

    // 6. 初始化累加器为 0
    clear(tCrC);

    // 7. 从 shared 内存加载 A 和 B (假设 sA, sB 已准备好)
    // copy(s2r_copy, sA_part, tCrA);
    // copy(s2r_copy, sB_part, tCrB);

    // 8. 执行 MMA 计算
    gemm(tiled_mma, tCrA, tCrB, tCrC);

    // 9. 将结果写回全局内存
    copy(tCrC, tCgC);
}
```

### 7.5 MMA 数据流总结

完整的 GEMM pipeline 数据流：

```
┌─────────────────────────────────────────────────────────────┐
│                     MMA 数据流                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Global Memory (gA, gB)                                     │
│         │                                                   │
│         │ TiledCopy (cp.async)                              │
│         ▼                                                   │
│  Shared Memory (sA, sB) ──[Swizzled Layout]                 │
│         │                                                   │
│         │ TiledCopy (ldmatrix / UniversalCopy)              │
│         ▼                                                   │
│  Register (rA, rB) ──────[Thread-private]                   │
│         │                                                   │
│         │ TiledMMA (mma.sync)                               │
│         ▼                                                   │
│  Register (rC accumulators)                                 │
│         │                                                   │
│         │ Store (UniversalCopy)                             │
│         ▼                                                   │
│  Global Memory (gC)                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**典型 GEMM Kernel 结构：**

```cpp
__global__ void gemm_kernel(...) {
    // 阶段 1: 定义硬件抽象
    using TiledMMA = ...;      // MMA 计算配置
    using G2S_Copy = ...;      // Global→Shared 拷贝配置
    using S2R_Copy = ...;      // Shared→Register 拷贝配置

    // 阶段 2: 获取线程视图
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto thr_g2s = g2s_copy.get_slice(threadIdx.x);
    auto thr_s2r = s2r_copy.get_slice(threadIdx.x);

    // 阶段 3: 创建寄存器 fragment
    auto tCrA = thr_mma.partition_fragment_A(sA);
    auto tCrB = thr_mma.partition_fragment_B(sB);
    auto tCrC = thr_mma.partition_fragment_C(gC);
    clear(tCrC);

    // 阶段 4: 主循环 (K 维度分块)
    for (int k = 0; k < K; k += kBlock) {
        // 4.1 Global → Shared (异步拷贝)
        copy(g2s_copy, gA_part, sA_part);
        copy(g2s_copy, gB_part, sB_part);
        cp_async_wait_group<0>();

        // 4.2 Shared → Register (ldmatrix)
        copy(s2r_copy, sA_part, tCrA);
        copy(s2r_copy, sB_part, tCrB);

        // 4.3 MMA 计算
        gemm(tiled_mma, tCrA, tCrB, tCrC);
    }

    // 阶段 5: 写回结果
    copy(tCrC, gC_part);
}
```

---

## 9. GEMM 综合示例

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
