// CuTe DSL 学习示例
// 按照由浅入深的顺序介绍 CuTe 常用函数
// 配套文档: docs/cute_notes.md

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/stride.hpp>
#include <cute/swizzle.hpp>
#include <cute/util/print.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

// ============================================================================
// Section 1: Layout 基础
// ============================================================================
void section_01_layout_basics() {
    printf("\n========== Section 1: Layout 基础 ==========\n");

    // 1.1 使用模板参数创建 Layout
    printf("\n--- 1.1 Layout<Shape, Stride> ---\n");
    using Layout1 = Layout<Shape<_16, _32>, Stride<_32, _1>>;
    auto layout1 = Layout1{};
    printf("Layout<Shape<16,32>, Stride<32,1>> (row-major 16x32):\n");
    print_layout(layout1);

    // 1.2 使用 make_layout 函数
    printf("\n--- 1.2 make_layout() ---\n");
    auto layout2 = make_layout(Shape<_8, _8>{}, Stride<_8, _1>{});  // row-major 8x8
    printf("make_layout(Shape<8,8>, Stride<8,1>):\n");
    print_layout(layout2);

    // 1.3 使用 GenRowMajor / GenColMajor
    printf("\n--- 1.3 GenRowMajor / GenColMajor ---\n");
    auto layout_rm = make_layout(Shape<_8, _8>{}, GenRowMajor{});
    auto layout_cm = make_layout(Shape<_8, _8>{}, GenColMajor{});
    printf("Row-major (Stride<8,1>):\n");
    print_layout(layout_rm);
    printf("Col-major (Stride<1,8>):\n");
    print_layout(layout_cm);

    // 1.4 查询 Layout 属性
    printf("\n--- 1.4 查询属性 ---\n");
    printf("size<0>(layout_rm) = %d\n", int(size<0>(layout_rm)));
    printf("size<1>(layout_rm) = %d\n", int(size<1>(layout_rm)));
    printf("size(layout_rm) = %d\n", int(size(layout_rm)));
    printf("layout_rm(2, 3) = %d (row=2, col=3 的线性索引)\n", layout_rm(2, 3));
}

// ============================================================================
// Section 2: Tensor 基础
// ============================================================================
void section_02_tensor_basics() {
    printf("\n========== Section 2: Tensor 基础 ==========\n");

    // 2.1 make_gmem_ptr
    printf("\n--- 2.1 make_gmem_ptr ---\n");
    int data[64];
    auto gmem_ptr = make_gmem_ptr(data);
    printf("make_gmem_ptr(data) 创建全局内存指针\n");

    // 2.2 make_tensor
    printf("\n--- 2.2 make_tensor ---\n");
    auto layout = make_layout(Shape<_8, _8>{}, GenRowMajor{});
    auto tensor = make_tensor(gmem_ptr, layout);
    printf("Tensor shape: (%d, %d)\n", int(size<0>(tensor)), int(size<1>(tensor)));

    // 2.3 读写 tensor
    printf("\n--- 2.3 读写 tensor ---\n");
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            tensor(i, j) = i * 8 + j;  // 写
        }
    }
    printf("tensor 写入后:\n");
    print_tensor(tensor);

    // 2.4 简化创建方式
    printf("\n--- 2.4 简化创建 ---\n");
    float data2[16];
    auto tensor2 = make_tensor(make_gmem_ptr(data2), Shape<_4, _4>{}, GenRowMajor{});
    printf("make_tensor(ptr, Shape<4,4>, GenRowMajor) 直接创建\n");
}

// ============================================================================
// Section 3: Swizzle 与 composition
// ============================================================================
void section_03_swizzle_and_composition() {
    printf("\n========== Section 3: Swizzle 与 composition ==========\n");

    // 3.1 Swizzle 基础
    printf("\n--- 3.1 Swizzle<B,M,S> ---\n");
    printf("Swizzle<3,0,3>: B=3(8行), M=0(1列固定), S=3(8位混洗)\n");
    printf("用于消除 shared memory bank conflicts\n");

    // 3.2 composition
    printf("\n--- 3.2 composition(Swizzle, Layout) ---\n");
    auto base_layout = make_layout(Shape<_16, _32>{}, GenRowMajor{});
    auto swizzled_layout = composition(Swizzle<3, 0, 3>{}, base_layout);

    printf("原始 layout (前4行):\n");
    for (int i = 0; i < 4; ++i) {
        printf("row %d: ", i);
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", base_layout(i, j));
        }
        printf("...\n");
    }

    printf("\nSwizzled layout (前4行):\n");
    for (int i = 0; i < 4; ++i) {
        printf("row %d: ", i);
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", swizzled_layout(i, j));
        }
        printf("...\n");
    }

    // 3.3 Swizzled Tensor 示例
    printf("\n--- 3.3 Swizzled Tensor 实际效果 ---\n");
    int a[64], b[64];
    auto base = make_layout(Shape<_8, _8>{}, GenRowMajor{});
    auto swz = composition(Swizzle<2, 1, 2>{}, base);

    auto ta = make_tensor(make_gmem_ptr(a), base);
    auto tb = make_tensor(make_gmem_ptr(b), swz);

    // 逻辑上写入相同数据
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            ta(i, j) = j;
            tb(i, j) = j;
        }
    }

    printf("普通 tensor (按 layout 读取):\n");
    print_tensor(ta);
    printf("\nSwizzled tensor (按 layout 读取，物理存储不同):\n");
    // 用 base layout 读取 swizzled 内存，可以看到实际存储差异
    auto tb_view = make_tensor(make_gmem_ptr(b), base);
    print_tensor(tb_view);
}

// ============================================================================
// Section 4: Tile 与 logical_divide
// ============================================================================
void section_04_tile_and_logical_divide() {
    printf("\n========== Section 4: Tile 与 logical_divide ==========\n");

    // 4.1 make_tile
    printf("\n--- 4.1 make_tile ---\n");
    printf("make_tile(Int<4>{}, Int<4>{}) = Shape<4,4> tile\n");
    printf("  用于定义 tile 的形状\n");

    // 4.2 logical_divide 概念说明
    printf("\n--- 4.2 logical_divide ---\n");
    printf("logical_divide 将大 layout 分割成小 tiles:\n");
    printf("  原始: 16x16 row-major\n");
    printf("  Tile: 4x4\n");
    printf("  结果: 4x4 个 tile，每个 tile 是 4x4\n");
    printf("  用法: auto tiled = logical_divide(global_layout, tile_shape);\n");

    // 4.3 local_tile 示例
    printf("\n--- 4.3 local_tile (在 kernel 中的用法) ---\n");
    printf("在 GEMM kernel 中常用模式:\n");
    printf("  1. 使用 logical_divide 将全局 tensor 分成 tiles\n");
    printf("  2. 每个 block 负责一个或多个 tiles\n");
    printf("  3. 使用 local_partition 获取当前 block 的数据\n");
}

// ============================================================================
// Section 5: local_partition
// ============================================================================
void section_05_local_partition() {
    printf("\n========== Section 5: local_partition ==========\n");

    // 5.1 1D partition
    printf("\n--- 5.1 1D local_partition ---\n");
    int data[16];
    for (int i = 0; i < 16; ++i) data[i] = i * 10;

    auto tensor = make_tensor(make_gmem_ptr(data), Shape<_16>{});
    auto thread_layout = make_layout(Shape<_4>{});  // 4 个线程

    printf("原始数据: ");
    for (int i = 0; i < 16; ++i) printf("%d ", data[i]);
    printf("\n\n各线程分区:\n");

    // 注意: local_partition 的第三个参数需要编译时常量
    auto partition_0 = local_partition(tensor, thread_layout, Int<0>{});
    auto partition_1 = local_partition(tensor, thread_layout, Int<1>{});
    auto partition_2 = local_partition(tensor, thread_layout, Int<2>{});
    auto partition_3 = local_partition(tensor, thread_layout, Int<3>{});

    printf("Thread 0: ");
    for (int i = 0; i < size(partition_0); ++i) printf("%d ", partition_0(i));
    printf("\n");

    printf("Thread 1: ");
    for (int i = 0; i < size(partition_1); ++i) printf("%d ", partition_1(i));
    printf("\n");

    printf("Thread 2: ");
    for (int i = 0; i < size(partition_2); ++i) printf("%d ", partition_2(i));
    printf("\n");

    printf("Thread 3: ");
    for (int i = 0; i < size(partition_3); ++i) printf("%d ", partition_3(i));
    printf("\n");

    // 5.2 2D partition
    printf("\n--- 5.2 2D local_partition ---\n");
    int data2d[64];
    for (int i = 0; i < 64; ++i) data2d[i] = i;

    auto tensor2d = make_tensor(make_gmem_ptr(data2d), Shape<_8, _8>{});
    auto thread_layout_2d = make_layout(Shape<_2, _2>{});  // 2x2 线程网格

    printf("8x8 tensor, 2x2 线程网格:\n");

    // 使用编译时常量
    auto part_00 = local_partition(tensor2d, thread_layout_2d, Int<0>{}, Int<0>{});
    auto part_01 = local_partition(tensor2d, thread_layout_2d, Int<0>{}, Int<1>{});
    auto part_10 = local_partition(tensor2d, thread_layout_2d, Int<1>{}, Int<0>{});
    auto part_11 = local_partition(tensor2d, thread_layout_2d, Int<1>{}, Int<1>{});

    printf("Thread(0,0) 负责的元素 (左上角): %d %d %d %d\n",
           part_00(0, 0), part_00(0, 1), part_00(1, 0), part_00(1, 1));
    printf("Thread(0,1) 负责的元素: %d %d %d %d\n",
           part_01(0, 0), part_01(0, 1), part_01(1, 0), part_01(1, 1));
    printf("Thread(1,0) 负责的元素: %d %d %d %d\n",
           part_10(0, 0), part_10(0, 1), part_10(1, 0), part_10(1, 1));
    printf("Thread(1,1) 负责的元素 (右下角): %d %d %d %d\n",
           part_11(0, 0), part_11(0, 1), part_11(1, 0), part_11(1, 1));
}

// ============================================================================
// Section 6: Copy_Atom 与 TiledCopy
// ============================================================================
void section_06_copy_atom_and_tiled_copy() {
    printf("\n========== Section 6: Copy_Atom 与 TiledCopy ==========\n");

    // 6.1 Copy_Atom 类型
    printf("\n--- 6.1 Copy_Atom 类型 ---\n");
    printf("常用 Copy_Atom:\n");
    printf("  - UniversalCopy<T>: 通用赋值\n");
    printf("  - SM75_U32x4_LDSM_N: ldmatrix.x4 (shared→register)\n");
    printf("  - SM80_CP_ASYNC: 异步拷贝 (global→shared)\n");

    // 6.2 TiledCopy 示例
    printf("\n--- 6.2 TiledCopy 示例 ---\n");
    using copy_atom = Copy_Atom<UniversalCopy<float>, float>;
    auto tiled_copy = make_tiled_copy(
        copy_atom{},
        Layout<Shape<_16>>{}  // 16 个线程
    );

    printf("TiledCopy 创建成功\n");
    printf("  - 线程布局: Layout<Shape<16>>\n");

    // 6.3 执行 copy
    printf("\n--- 6.3 执行 copy ---\n");
    float src[64], dst[64];
    for (int i = 0; i < 64; ++i) { src[i] = i; dst[i] = 0; }

    auto src_tensor = make_tensor(make_gmem_ptr(src), Shape<_64>{});
    auto dst_tensor = make_tensor(make_gmem_ptr(dst), Shape<_64>{});

    // 单线程模拟 copy
    auto thr_copy = tiled_copy.get_slice(0);
    auto src_part = thr_copy.partition_S(src_tensor);
    auto dst_part = thr_copy.partition_D(dst_tensor);

    copy(tiled_copy, src_part, dst_part);

    printf("Copy 前: dst[0..7] = ");
    for (int i = 0; i < 8; ++i) printf("%.0f ", dst[i]);
    printf("\nCopy 后: dst[0..7] = ");
    for (int i = 0; i < 8; ++i) printf("%.0f ", dst[i]);
    printf("\n");
}

// ============================================================================
// Section 7: MMA_Atom 与 TiledMMA
// ============================================================================
void section_07_mma_atom_and_tiled_mma() {
    printf("\n========== Section 7: MMA_Atom 与 TiledMMA ==========\n");

    // 7.1 MMA_Atom
    printf("\n--- 7.1 MMA_Atom ---\n");
    printf("MMA_Atom 封装硬件矩阵乘累加指令:\n");
    printf("  - SM80_16x8x16_F32F16F16F32_TN:\n");
    printf("    输入 A: 16x16 FP16, 输入 B: 16x8 FP16, 输出 C: 16x8 FP32\n");
    printf("  - SM70_16x16x8_F32F16F16F32_TN: Volta 架构\n");
    printf("  - SM75_16x8x8_F32F16F16F32_TN: Turing 架构\n");

    // 7.2 TiledMMA 概念
    printf("\n--- 7.2 TiledMMA ---\n");
    printf("TiledMMA 将 MMA_Atom tile 化，支持更大矩阵块:\n");
    printf("  TiledMMA<MMA_Atom, Layout<Shape<M,N,K>>>\n");
    printf("  - Shape<2,2,1>: M维x2, N维x2, K维x1\n");
    printf("  - 总计算量: (16*2)x(8*2)x16 = 32x16x16\n");

    // 7.3 partition_fragment 概念
    printf("\n--- 7.3 partition_fragment ---\n");
    printf("在 kernel 中，每个线程需要自己的数据片段:\n");
    printf("  auto thr_mma = mma.get_slice(thread_id);\n");
    printf("  auto tCrA = thr_mma.partition_A(gA);  // gA 是 tensor\n");
    printf("  auto tCrB = thr_mma.partition_B(gB);\n");
    printf("  auto tCrC = thr_mma.partition_C(gC);\n");
    printf("  gemm(mma, tCrA, tCrB, tCrC);  // 执行 MMA\n");

    printf("\n注意: MMA 相关 API 需要在 CUDA kernel 中使用\n");
}

// ============================================================================
// Section 8: GEMM 综合示例
// ============================================================================
void section_08_gemm_overview() {
    printf("\n========== Section 8: GEMM 综合示例 ==========\n");

    printf("\nCuTe GEMM 数据流:\n");
    printf("\n");
    printf("  Global Memory (gA, gB, gC)\n");
    printf("         │\n");
    printf("         │ TiledCopy (cp.async)\n");
    printf("         ▼\n");
    printf("  Shared Memory (sA, sB) [Swizzled]\n");
    printf("         │\n");
    printf("         │ TiledCopy (ldmatrix)\n");
    printf("         ▼\n");
    printf("  Register (rA, rB, rC)\n");
    printf("         │\n");
    printf("         │ TiledMMA (mma.sync)\n");
    printf("         ▼\n");
    printf("  Result in rC → Store → gC\n");

    printf("\n关键 API 调用顺序:\n");
    printf("  1. make_layout() + make_tensor()      # 创建 tensor\n");
    printf("  2. composition(Swizzle{}, layout)      # 优化 shared memory\n");
    printf("  3. make_tiled_copy() + partition_S/D() # 数据移动\n");
    printf("  4. TiledMMA + partition_fragment_*()   # 准备计算\n");
    printf("  5. gemm(mma, rA, rB, rC)               # 执行计算\n");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("CuTe DSL 学习示例\n");
    printf("配套文档: docs/cute_notes.md\n");

    section_01_layout_basics();       // Layout 基础
    section_02_tensor_basics();       // Tensor 基础
    section_03_swizzle_and_composition();  // Swizzle
    section_04_tile_and_logical_divide();  // Tile
    section_05_local_partition();     // Partition
    section_06_copy_atom_and_tiled_copy(); // Copy
    section_07_mma_atom_and_tiled_mma();   // MMA
    section_08_gemm_overview();       // GEMM 概览

    printf("\n========== 所有示例完成 ==========\n");
    return 0;
}