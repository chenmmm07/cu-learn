// CuTe DSL 学习示例
// 按照由浅入深的顺序介绍 CuTe 常用函数
// 配套文档: docs/cute_notes.md

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/stride.hpp>
#include <cute/swizzle.hpp>
#include <cute/util/print.hpp>
#include <cute/util/print_tensor.hpp>
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
    using Layout1 = Layout<Shape<_8, _16>, Stride<_16, _1>>;
    auto layout1 = Layout1{};
    printf("Layout<Shape<8,16>, Stride<16,1>> (row-major 8x16):\n");
    print_layout(layout1);

    // 1.2 使用 make_layout 函数
    printf("\n--- 1.2 make_layout() ---\n");
    auto layout2 = make_layout(Shape<_4, _16>{}, Stride<_16, _1>{});  // row-major 4x16
    printf("make_layout(Shape<4,16>, Stride<16,1>):\n");
    print_layout(layout2);

    // 1.3 使用 GenRowMajor / GenColMajor
    printf("\n--- 1.3 GenRowMajor / GenColMajor ---\n");
    auto layout_rm = make_layout(Shape<_8, _16>{}, GenRowMajor{});
    auto layout_cm = make_layout(Shape<_8, _16>{}, GenColMajor{});
    printf("Row-major (Stride<16,1>):\n");
    print_layout(layout_rm);
    printf("Col-major (Stride<1,16>):\n");
    print_layout(layout_cm);

    // 1.4 查询 Layout 属性
    printf("\n--- 1.4 查询属性 ---\n");
    printf("size<0>(layout_rm) = %d, size<1>(layout_rm) = %d \n", int(size<0>(layout_rm)), int(size<1>(layout_rm)));
    printf("size(layout_rm) = %d\n", int(size(layout_rm)));
    printf("layout_rm(2, 3) = %d (row=2, col=3 的线性索引)\n", layout_rm(2, 3));
    printf("layout_cm(2, 3) = %d (row=2, col=3 的线性索引)\n", layout_cm(2, 3));

    // 1.5 make_coord 创建坐标
    printf("\n--- 1.5 make_coord 创建坐标 ---\n");
    printf("make_coord 用于创建多维坐标:\n");

    // 1D 坐标
    auto coord1d = make_coord(5);
    print(coord1d);
    printf("\n  make_coord(5) -> 1D坐标\n");

    // 2D 坐标
    auto coord2d = make_coord(2, 3);
    print(coord2d);
    printf("\n  make_coord(2, 3) -> 2D坐标: (%d, %d)\n", get<0>(coord2d), get<1>(coord2d));

    // 嵌套坐标 (用于 tiled layout)
    auto nested = make_coord(make_coord(1, 2), make_coord(3, 4));
    printf("  make_coord(make_coord(1,2), make_coord(3,4)) -> 嵌套坐标, 用于访问 tiled layout 的分层结构\n");
    print(nested);
    printf("\n");

    // 用 coord 访问 layout
    printf("\n用坐标访问 layout:\n");
    printf("  layout_rm(make_coord(2, 3)) = %d\n", layout_rm(make_coord(2, 3)));
    printf("  等价于 layout_rm(2, 3) = %d\n", layout_rm(2, 3));
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
    auto layout = make_layout(Shape<_4, _16>{}, GenRowMajor{});
    auto tensor = make_tensor(gmem_ptr, layout);
    printf("Tensor shape: (%d, %d)\n", int(size<0>(tensor)), int(size<1>(tensor)));

    // 2.3 读写 tensor
    printf("\n--- 2.3 读写 tensor ---\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 16; ++j) {
            tensor(i, j) = i * 16 + j;  // 写
        }
    }
    printf("tensor 写入后:\n");
    print_tensor(tensor);

    // 2.4 简化创建方式
    printf("\n--- 2.4 简化创建 ---\n");
    float data2[48];
    auto tensor2 = make_tensor(make_gmem_ptr(data2), Shape<_4, _12>{}, GenRowMajor{});
    printf("make_tensor(ptr, Shape<4,12>, GenRowMajor) 直接创建\n");
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
    int a[48], b[48];
    auto base = make_layout(Shape<_4, _12>{}, GenRowMajor{});
    auto swz = composition(Swizzle<2, 1, 2>{}, base);

    auto ta = make_tensor(make_gmem_ptr(a), base);
    auto tb = make_tensor(make_gmem_ptr(b), swz);

    // 逻辑上写入相同数据
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 12; ++j) {
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
    auto tile1 = make_tile(Int<4>{}, Int<8>{});  // 4x8 tile
    auto tile2 = make_tile(Int<2>{}, Int<3>{}, Int<4>{});  // 2x3x4 tile (3D)

    printf("make_tile 创建 tile 形状:\n");
    printf("  tile1 = make_tile(Int<4>{}, Int<8>{})\n");
    printf("  tile2 = make_tile(Int<2>{}, Int<3>{}, Int<4>{})\n");
    printf("\n实际值:\n");
    printf("  size<0>(tile1) = %d, size<1>(tile1) = %d\n", int(size<0>(tile1)), int(size<1>(tile1)));
    printf("  size(tile1) = %d (总元素数)\n", int(size(tile1)));
    printf("  size<0>(tile2) = %d, size<1>(tile2) = %d, size<2>(tile2) = %d\n",
           int(size<0>(tile2)), int(size<1>(tile2)), int(size<2>(tile2)));
    printf("  size(tile2) = %d (总元素数)\n", int(size(tile2)));

    // 4.2 logical_divide 基础示例
    printf("\n--- 4.2 logical_divide ---\n");

    // 简单的 1D 示例
    auto layout_1d = make_layout(Shape<_12>{});
    auto tile_1d = make_tile(Int<4>{});
    auto tiled_1d_divide = logical_divide(layout_1d, tile_1d);

    printf("1D 示例: 12 元素分割成 4 元素的 tiles\n");
    printf("  原始 layout: ");
    print(layout_1d);
    printf("\n");
    printf("  原始访问: layout_1d(0)=%d, layout_1d(4)=%d, layout_1d(8)=%d\n",
           int(layout_1d(0)), int(layout_1d(4)), int(layout_1d(8)));
    printf("  分割后 layout: ");
    print(tiled_1d_divide);
    printf("\n");
    printf("  结构说明: (_4,_3) 表示内层4元素，外层3个tile\n");

    // 对比 row-major 和 column-major 的 divide
    printf("\n=== divide 与原始 layout 的 row/col-major 无关 ===\n");
    auto layout_rm = make_layout(Shape<_8, _8>{}, GenRowMajor{});  // row-major
    auto layout_cm = make_layout(Shape<_8, _8>{}, GenColMajor{});  // col-major
    auto tile_2d = make_tile(Int<4>{}, Int<4>{});

    auto tiled_rm = logical_divide(layout_rm, tile_2d);
    auto tiled_cm = logical_divide(layout_cm, tile_2d);

    printf("Row-major layout: ");
    print(layout_rm);
    printf(" -> divide: ");
    print(tiled_rm);
    printf("\n");

    printf("Col-major layout: ");
    print(layout_cm);
    printf(" -> divide: ");
    print(tiled_cm);
    printf("\n\n");

    printf("关键点:\n");
    printf("  1. logical_divide 的 Shape 结构相同: ((_4,_2),(_4,_2))\n");
    printf("  2. 差异在 Stride: RM是((_4,_16),(_1,_4)), CM是((_1,_4),(_4,_16))\n");
    printf("  3. divide 操作在'坐标空间'进行，与'地址映射'正交\n");
    printf("  4. logical_divide 不需要'编码'——每个维度独立访问\n\n");

    // zipped_divide 的 column-major 编码是固定的
    auto zipped_rm = zipped_divide(layout_rm, tile_2d);
    auto zipped_cm = zipped_divide(layout_cm, tile_2d);

    printf("zipped_divide 对比:\n");
    printf("  RM zipped: ");
    print(zipped_rm);
    printf("\n  CM zipped: ");
    print(zipped_cm);
    printf("\n\n");

    printf("zipped_divide 的 inner_idx 编码始终是 column-major:\n");
    printf("  inner_row = inner_idx %% tile_height\n");
    printf("  inner_col = inner_idx / tile_height\n");
    printf("  这是 CuTe 的固定设计，不可更改\n");

    // 2D 示例
    auto global_layout = make_layout(Shape<_12, _16>{}, GenRowMajor{});
    auto tile_shape = make_tile(Int<4>{}, Int<8>{});

    printf("\n2D 示例: 12x16 分割成 4x8 的 tiles\n");
    printf("  原始 layout: ");
    print(global_layout);
    printf("\n");
    printf("  Tile 形状: 4x8\n");
    printf("  结果: 3x2 个 tile (12/4=3, 16/8=2)\n");

    // 4.3 logical_divide 2D 实际示例
    printf("\n--- 4.3 logical_divide 2D 实际示例 ---\n");
    auto tiled_layout = logical_divide(global_layout, tile_shape);

    printf("原始 layout 访问:\n");
    print_layout(global_layout);
    print("\n");
    printf("  global_layout(0,0) -> 线性索引 %d\n", int(global_layout(0, 0)));
    printf("  global_layout(1,0) -> 线性索引 %d\n", int(global_layout(1, 0)));
    printf("  global_layout(0,1) -> 线性索引 %d\n", int(global_layout(0, 1)));
    printf("  global_layout(4,0) -> 线性索引 %d (第2个tile行开始)\n", int(global_layout(4, 0)));
    printf("  global_layout(0,8) -> 线性索引 %d (第2个tile列开始)\n", int(global_layout(0, 8)));

    printf("\nlogical_divide 后的结构:\n");
    printf("  完整 layout: ");
    print(tiled_layout);
    printf("\n");
    printf("  总大小 size(tiled_layout) = %d\n", int(size(tiled_layout)));

    // 使用 coordinate 访问
    printf("\n通过 make_coord 访问:\n");
    printf("  结构: (inner_row, tile_row), (inner_col, tile_col)\n");
    printf("  tiled_layout(make_coord(0,0), make_coord(0,0)) = %d (tile(0,0)内(0,0)) = global_layout(0 + 0 * size<0>(tile_shape), 0 + 0 * size<1>(tile_shape))\n", int(tiled_layout(make_coord(0, 0), make_coord(0, 0))));
    printf("  tiled_layout(make_coord(0,1), make_coord(0,0)) = %d (tile(1,0)内(0,0)) = global_layout(0 + 1 * size<0>(tile_shape), 0 + 0 * size<1>(tile_shape))\n", int(tiled_layout(make_coord(0, 1), make_coord(0, 0))));
    printf("  tiled_layout(make_coord(0,0), make_coord(0,1)) = %d (tile(0,1)内(0,0)) = global_layout(0 + 0 * size<0>(tile_shape), 0 + 1 * size<1>(tile_shape))\n", int(tiled_layout(make_coord(0, 0), make_coord(0, 1))));
    printf("  tiled_layout(make_coord(3,2), make_coord(7,1)) = %d (tile(2,1)内(3,7)) = global_layout(3 + 2 * size<0>(tile_shape), 7 + 1 * size<1>(tile_shape))\n", int(tiled_layout(make_coord(3, 2), make_coord(7, 1))));

    // 4.4 zipped_divide 示例
    printf("\n--- 4.4 zipped_divide (简化坐标访问) ---\n");
    auto zipped = zipped_divide(global_layout, tile_shape);

    printf("=== Tile(0,0) 内容展示 ===\n");
    printf("原始 global_layout (row 0-3, col 0-7):\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", global_layout(i, j));
        }
        printf("\n");
    }

    printf("\nzipped(inner_idx, 0) 按 column-major 遍历:\n");
    printf("  inner_idx -> (inner_row=idx%%4, inner_col=idx/4)\n\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            // column-major: inner_idx = inner_col * tile_height + inner_row
            // 即 inner_idx = j * 4 + i
            int inner_idx = j * 4 + i;  // column-major 编码
            printf("%3d ", zipped(inner_idx, 0));
        }
        printf("\n");
    }

    printf("\n对比: 如果用 row-major 编码 (i*8+j) 会得到错乱结果:\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            int inner_idx = i * 8 + j;  // row-major 编码 (错误!)
            printf("%3d ", zipped(inner_idx, 0));
        }
        printf("\n");
    }

    printf("\nzipped_divide 将内层和外层坐标合并:\n");
    printf("  global layout: ");
    print(global_layout);
    printf("\n  tile_shape: ");
    print(tile_shape);
    printf("\n  zipped layout: ");
    print(zipped);
    printf("\n");
    printf("  size<0>(zipped) = %d (tile内元素数 = 4*8=32)\n", int(size<0>(zipped)));
    printf("  size<1>(zipped) = %d (tile数 = 3*2=6)\n", int(size<1>(zipped)));
    
    // 与 logical_divide 对比结构
    printf("\n=== zipped_divide vs logical_divide 结构对比 ===\n");
    printf("logical_divide 结果: ((_4,_3),(_8,_2)):((_16,_64),(_1,_8))\n");
    printf("  Shape 分解:\n");
    printf("    第一组 (_4,_3): inner_row∈[0,4), tile_row∈[0,3)\n");
    printf("    第二组 (_8,_2): inner_col∈[0,8), tile_col∈[0,2)\n");
    printf("  访问方式: layout(make_coord(inner_row, tile_row), make_coord(inner_col, tile_col))\n");
    printf("\nzipped_divide 结果: ((_4,_8),(_3,_2)):((_16,_1),(_64,_8))\n");
    printf("  Shape 分解:\n");
    printf("    第一组 (_4,_8): inner_idx∈[0,32) - tile内扁平化索引\n");
    printf("    第二组 (_3,_2): tile_idx∈[0,6) - tile编号\n");
    printf("  访问方式: layout(inner_idx, tile_idx)\n");

    // 详细解释为什么访问方式不同
    printf("\n=== 为什么访问方式不同？===\n");
    printf("关键: Shape 的组织方式不同！\n\n");

    printf("logical_divide Shape: ((_4,_3), (_8,_2))\n");
    printf("                       ^^^^^^^^  ^^^^^^^^\n");
    printf("                       行维度     列维度\n");
    printf("  每个()内保留 inner/tile 分离:\n");
    printf("    (_4,_3) = (inner_row=4, tile_row=3)\n");
    printf("    (_8,_2) = (inner_col=8, tile_col=2)\n");
    printf("  所以需要 make_coord(inner, tile) 来访问每个维度\n\n");

    printf("zipped_divide Shape: ((_4,_8), (_3,_2))\n");
    printf("                       ^^^^^^^^  ^^^^^^^^\n");
    printf("                       inner合并  tile合并\n");
    printf("  \"zipped\" 把同类维度拉在一起:\n");
    printf("    (_4,_8) = inner_row × inner_col = 32 (所有 inner 合并)\n");
    printf("    (_3,_2) = tile_row × tile_col = 6   (所有 tile 合并)\n");
    printf("  所以只需要 (inner_idx, tile_idx) 两个标量\n\n");

    printf("比喻: logical_divide 像\"按维度组织\", zipped_divide 像\"按类型组织\"\n");

    // 详细计算过程
    printf("\n=== zipped_divide 计算过程详解 ===\n");
    printf("CuTe 使用 column-major 布局分解标量坐标:\n");
    printf("\nzipped(inner_idx, tile_idx) 的计算:\n");
    printf("  1. inner_idx 分解为 (inner_row, inner_col) [column-major]:\n");
    printf("     inner_row = inner_idx %% tile_height = inner_idx %% 4\n");
    printf("     inner_col = inner_idx / tile_height = inner_idx / 4\n");
    printf("  2. tile_idx 分解为 (tile_row, tile_col) [column-major]:\n");
    printf("     tile_row = tile_idx %% num_tile_rows = tile_idx %% 3\n");
    printf("     tile_col = tile_idx / num_tile_rows = tile_idx / 3\n");
    printf("  3. 全局坐标:\n");
    printf("     global_row = inner_row + tile_row * tile_height\n");
    printf("     global_col = inner_col + tile_col * tile_width\n");
    printf("  4. 线性索引 (global layout 是 row-major):\n");
    printf("     linear_idx = global_row * 16 + global_col * 1\n");

    // 具体计算示例
    printf("\n=== 具体计算示例 ===\n");
    // 示例1: zipped(9, 0)
    printf("zipped(9, 0) 的计算:\n");
    printf("  inner_idx=9: inner_row=9%%4=1, inner_col=9/4=2\n");
    printf("  tile_idx=0: tile_row=0%%3=0, tile_col=0/3=0\n");
    printf("  global_row=1+0*4=1, global_col=2+0*8=2\n");
    printf("  linear_idx=1*16+2*1=18\n");
    printf("  实际值: zipped(9, 0) = %d\n", int(zipped(9, 0)));

    // 示例2: zipped(0, 3)
    printf("\nzipped(0, 3) 的计算:\n");
    printf("  inner_idx=0: inner_row=0%%4=0, inner_col=0/4=0\n");
    printf("  tile_idx=3: tile_row=3%%3=0, tile_col=3/3=1\n");
    printf("  global_row=0+0*4=0, global_col=0+1*8=8\n");
    printf("  linear_idx=0*16+8*1=8\n");
    printf("  实际值: zipped(0, 3) = %d\n", int(zipped(0, 3)));

    // 示例3: zipped(15, 5)
    printf("\nzipped(15, 5) 的计算:\n");
    printf("  inner_idx=15: inner_row=15%%4=3, inner_col=15/4=3\n");
    printf("  tile_idx=5: tile_row=5%%3=2, tile_col=5/3=1\n");
    printf("  global_row=3+2*4=11, global_col=3+1*8=11\n");
    printf("  linear_idx=11*16+11*1=187\n");
    printf("  实际值: zipped(15, 5) = %d\n", int(zipped(15, 5)));

    // 示例4: zipped(31, 5) - 最后一个元素
    printf("\nzipped(31, 5) 的计算:\n");
    printf("  inner_idx=31: inner_row=31%%4=3, inner_col=31/4=7\n");
    printf("  tile_idx=5: tile_row=5%%3=2, tile_col=5/3=1\n");
    printf("  global_row=3+2*4=11, global_col=7+1*8=15\n");
    printf("  linear_idx=11*16+15*1=191\n");
    printf("  实际值: zipped(31, 5) = %d\n", int(zipped(31, 5)));

    // 使用场景对比
    printf("\n=== 使用场景对比 ===\n");
    printf("logical_divide 适用场景:\n");
    printf("  - 需要按行列语义访问 tile 内元素\n");
    printf("  - 需要保持 2D 结构进行后续操作\n");
    printf("  - 例如: 加载 tile 某一行到寄存器\n");
    printf("\nzipped_divide 适用场景:\n");
    printf("  - 需要线性遍历 tile 内所有元素\n");
    printf("  - 与硬件指令配合 (如 ldmatrix 按行加载)\n");
    printf("  - 例如: 遍历 tile 内每个元素执行操作\n");

    // 4.5 local_tile 示例
    printf("\n--- 4.5 local_tile ---\n");
    printf("local_tile 返回 tensor 的指定 tile 视图:\n");
    printf("  参数: (tensor, tile_shape, tile_coord)\n");
    printf("  tile_coord: 指定要获取的 tile 坐标\n\n");

    // 创建一个 tensor 用于演示
    int tensor_data[192];
    for (int i = 0; i < 192; ++i) tensor_data[i] = i;
    auto g_tensor = make_tensor(make_gmem_ptr(tensor_data), global_layout);
    print_tensor(g_tensor);
    print("\n");
    // 获取不同 tile 的视图
    // local_tile(tensor, tile_shape, coord) - coord 是 tile 的坐标
    auto tile_00 = local_tile(g_tensor, tile_shape, make_coord(0, 0));  // 第(0,0)个tile
    auto tile_21 = local_tile(g_tensor, tile_shape, make_coord(2, 1));  // 第(2,1)个tile

    printf("=== local_tile(tensor, tile_shape, coord) ===\n\n");
    printf("Tile(0,0) - rows [0,4), cols [0,8):\n");
    print_tensor(tile_00);
    printf("\nTile(2,1) - rows [8,12), cols [8,16):\n");
    print_tensor(tile_21);

    // Tile 划分示意
    printf("\n=== Tile 划分示意图 (12x16 矩阵, 4x8 tiles) ===\n");
    printf("  tile_shape: 4行 × 8列\n");
    printf("  tile 数量: 3行 × 2列\n\n");
    printf("         tile_col: 0      1\n");
    printf("  tile_row 0:  [0,0]    [0,1]\n");
    printf("  tile_row 1:  [1,0]    [1,1]\n");
    printf("  tile_row 2:  [2,0]    [2,1]\n\n");

    printf("坐标到全局范围的映射:\n");
    printf("  tile(r,c) -> rows [%d, %d), cols [%d, %d)\n",
           0, 4, 0, 8);
    printf("  tile(r,c) -> rows [%d + r*4, %d + r*4), cols [%d + c*8, %d + c*8)\n",
           0, 4, 0, 8);

    // 与 logical_divide 的关系
    printf("\n=== local_tile vs logical_divide ===\n");
    printf("logical_divide(layout, tile):\n");
    printf("  - 返回分割后的完整 layout\n");
    printf("  - 包含所有 tile 信息\n");
    printf("  - 需要额外用坐标访问\n\n");
    printf("local_tile(tensor, tile, coord):\n");
    printf("  - 直接返回指定 tile 的 tensor 视图\n");
    printf("  - 一步完成分割和选择\n");
    printf("  - 常用于 kernel 中获取 block 数据\n\n");

    // 在 kernel 中的典型用法
    printf("=== 在 kernel 中的典型用法 ===\n");
    printf("__global__ void gemm_kernel(float* A, int M, int N) {\n");
    printf("  // 创建全局 tensor\n");
    printf("  auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, N));\n");
    printf("  // 获取当前 block 负责的 tile\n");
    printf("  auto tA = local_tile(gA, TileShape{}, make_coord(blockIdx.x, blockIdx.y));\n");
    printf("  // tA 的 shape = TileShape (当前 block 的数据)\n");
    printf("}\n");
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
    int data2d[96];
    for (int i = 0; i < 96; ++i) data2d[i] = i;

    // 不指定stride则默认ColMajor
    auto tensor2d = make_tensor(make_gmem_ptr(data2d), Shape<_6, _16>{});
    auto thread_layout_2d = make_layout(Shape<_2, _4>{});  // 2x4 线程网格

    print_tensor(tensor2d);
    printf("\n6x16 tensor, 2x4 线程网格:\n");
    // 使用编译时常量 - 展示不同线程负责的区域
    auto part_00 = local_partition(tensor2d, thread_layout_2d, Int<0>{}, Int<0>{});
    auto part_01 = local_partition(tensor2d, thread_layout_2d, Int<0>{}, Int<1>{});
    auto part_02 = local_partition(tensor2d, thread_layout_2d, Int<0>{}, Int<2>{});
    auto part_10 = local_partition(tensor2d, thread_layout_2d, Int<1>{}, Int<0>{});
    auto part_13 = local_partition(tensor2d, thread_layout_2d, Int<1>{}, Int<3>{});

    // 5.3 local_partition 与 logical_divide 的关系
    printf("\n--- 5.3 local_partition 与 logical_divide 的关系 ---\n");
    printf("local_partition 内部实现:\n");
    printf("  1. 先对 tensor 执行 logical_divide(tensor, thread_layout)\n");
    printf("  2. 返回 logical_divide 后的内层 slice (对应指定线程的数据)\n");
    printf("\n为什么需要编译时常量:\n");
    printf("  - CuTe 的 Layout 是编译期确定的类型\n");

    // 5.4 local_tile vs local_partition 详细对比
    printf("\n--- 5.4 local_tile vs local_partition 详细对比 ---\n");

    // 准备一个 8x8 的 tensor 用于对比
    int cmp_data[64];
    for (int i = 0; i < 64; ++i) cmp_data[i] = i;
    auto cmp_tensor = make_tensor(make_gmem_ptr(cmp_data), Shape<_8, _8>{}, GenRowMajor{});

    printf("原始 8x8 tensor:\n");
    print_tensor(cmp_tensor);
    printf("\n");

    // local_tile 示例
    printf("=== local_tile: 按 tile 形状分割 ===\n");
    printf("tile_shape = 4x4, 每个 tile 包含 16 个连续元素\n\n");
    auto tile_shape_cmp = make_tile(Int<4>{}, Int<4>{});

    auto tile_00_cmp = local_tile(cmp_tensor, tile_shape_cmp, make_coord(0, 0));
    auto tile_11_cmp = local_tile(cmp_tensor, tile_shape_cmp, make_coord(1, 1));

    printf("local_tile(tensor, 4x4, (0,0)) -> 左上 4x4:\n");
    print_tensor(tile_00_cmp);
    printf("\nlocal_tile(tensor, 4x4, (1,1)) -> 右下 4x4:\n");
    print_tensor(tile_11_cmp);
    printf("\n");

    // local_partition 示例
    printf("=== local_partition: 按线程布局分割 ===\n");
    printf("thread_layout = 2x2, 4个线程，每个线程负责分散的元素\n\n");
    auto thr_layout_cmp = make_layout(Shape<_2, _2>{});

    auto thr0 = local_partition(cmp_tensor, thr_layout_cmp, Int<0>{});
    auto thr1 = local_partition(cmp_tensor, thr_layout_cmp, Int<1>{});
    auto thr2 = local_partition(cmp_tensor, thr_layout_cmp, Int<2>{});
    auto thr3 = local_partition(cmp_tensor, thr_layout_cmp, Int<3>{});

    printf("Thread 0 (左上 4x4):\n");
    print_tensor(thr0);
    printf("\nThread 1 (右上 4x4):\n");
    print_tensor(thr1);
    printf("\nThread 2 (左下 4x4):\n");
    print_tensor(thr2);
    printf("\nThread 3 (右下 4x4):\n");
    print_tensor(thr3);
    printf("\n");

    // 关键区别总结
    printf("=== 关键区别总结 ===\n\n");
    printf("1. 分割依据:\n");
    printf("   local_tile:     按 tile_shape (元素个数) 分割\n");
    printf("   local_partition: 按 thread_layout (线程数量) 分割\n\n");

    printf("2. 元素分布:\n");
    printf("   local_tile:     每个 tile 内元素连续存储\n");
    printf("   local_partition: 每个线程元素可能分散 (stride > 1)\n\n");

    printf("3. 使用场景:\n");
    printf("   local_tile:     一个 block 处理一个 tile (block 级并行)\n");
    printf("   local_partition: 一个 thread 处理一部分数据 (thread 级并行)\n\n");

    printf("4. 典型配合:\n");
    printf("   // 先用 local_tile 获取 block 数据\n");
    printf("   auto block_data = local_tile(g_tensor, BlockShape{}, block_coord);\n");
    printf("   // 再用 local_partition 分给线程\n");
    printf("   auto thread_data = local_partition(block_data, ThreadLayout{}, thread_id);\n");
    printf("  - local_partition 返回的 tensor 类型取决于线程 ID\n");
    printf("  - 编译时常量允许编译器推导返回类型并优化\n");
    printf("\n实际 kernel 中的用法:\n");
    printf("  // 在 kernel 中, threadIdx 是运行时值\n");
    printf("  // 但 CuTe 提供了 get_slice() 方法处理运行时索引\n");
    printf("  auto thr_slice = tiled_copy.get_slice(threadIdx.x);\n");
    printf("  auto src_part = thr_slice.partition_S(src_tensor);\n");
    printf("  auto dst_part = thr_slice.partition_D(dst_tensor);\n");

    // 展示与 logical_divide 的等价性
    printf("\n等价示例:\n");
    printf("  // 这两种写法等价:\n");
    printf("  auto partition = local_partition(tensor, layout, Int<1>{});\n");
    printf("  auto divided = logical_divide(tensor, layout);\n");
    printf("  auto slice = make_tensor(divided.data(), get<1>(divided.layout()));\n");

    printf("Thread(0,0) 负责 (row 0-2, col 0-3) 的元素, 具体如下\n");
    print_tensor(part_00);
    print("\n");
    printf("Thread(0,1) 负责 (row 0-2, col 4-7) 的元素, 具体如下\n");
    print_tensor(part_01);
    print("\n");
    printf("Thread(0,2) 负责 (row 0-2, col 8-11) 的元素\n");
    printf("Thread(1,0) 负责 (row 3-5, col 0-3) 的元素\n");
    printf("Thread(1,3) 负责 (row 3-5, col 12-15) 的元素\n");
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
        Layout<Shape<_4, _4>>{}  // 4x4 = 16 个线程
    );

    printf("TiledCopy 创建成功\n");
    printf("  - Copy_Atom: UniversalCopy<float>\n");
    printf("  - 线程布局: Layout<Shape<4, 4>> = 16 个线程\n");

    // 6.3 get_slice, partition_S, partition_D 详解
    printf("\n--- 6.3 get_slice, partition_S, partition_D 详解 ---\n");

    // 准备数据
    float src_data[64], dst_data[64];
    for (int i = 0; i < 64; ++i) { src_data[i] = i; dst_data[i] = 0; }
    auto src_tensor = make_tensor(make_gmem_ptr(src_data), Shape<_8, _8>{});
    auto dst_tensor = make_tensor(make_gmem_ptr(dst_data), Shape<_8, _8>{});

    printf("=== 概念解释 ===\n");
    printf("TiledCopy 定义了多线程协作拷贝的方式:\n");
    printf("  - 每个线程负责哪些元素?\n");
    printf("  - 数据如何分布在线程间?\n\n");

    printf("get_slice(thread_id): 获取指定线程的\"切片器\"\n");
    printf("  - thread_id 可以是运行时值 (如 threadIdx.x)\n");
    printf("  - 返回一个对象，包含该线程的数据分布信息\n\n");

    printf("partition_S(tensor): 对源 tensor 进行分区\n");
    printf("  - S = Source\n");
    printf("  - 返回当前线程负责读取的数据\n\n");

    printf("partition_D(tensor): 对目标 tensor 进行分区\n");
    printf("  - D = Destination\n");
    printf("  - 返回当前线程负责写入的数据\n\n");

    // 展示不同线程的分区结果
    printf("=== 不同线程的分区结果 ===\n\n");

    auto thr0 = tiled_copy.get_slice(0);
    auto thr5 = tiled_copy.get_slice(5);

    auto src0 = thr0.partition_S(src_tensor);
    auto dst0 = thr0.partition_D(dst_tensor);
    auto src5 = thr5.partition_S(src_tensor);
    auto dst5 = thr5.partition_D(dst_tensor);

    printf("Thread 0 (thread_id=0):\n");
    printf("  partition_S 结果 shape: ");
    print(src0.shape());
    printf("\n  partition_D 结果 shape: ");
    print(dst0.shape());
    printf("\n  src0 layout: ");
    print(src0.layout());
    printf("\n\n");

    printf("Thread 5 (thread_id=5):\n");
    printf("  partition_S 结果 shape: ");
    print(src5.shape());
    printf("\n  partition_D 结果 shape: ");
    print(dst5.shape());
    printf("\n  src5 layout: ");
    print(src5.layout());
    printf("\n\n");

    // 展示 ThrCopy 返回的数据内容
    printf("=== ThrCopy 分区数据详解 ===\n");
    printf("TiledCopy 线程布局: 4x4 = 16 线程\n");
    printf("Tensor 大小: 8x8 = 64 元素\n");
    printf("每个线程负责: 64/16 = 4 元素\n\n");

    printf("Thread 0 负责的源数据索引 (通过 partition_S):\n");
    printf("  src0(0) 对应全局索引: %d\n", int(src0(0)));
    printf("  src0(1) 对应全局索引: %d\n", int(src0(1)));
    printf("  src0(2) 对应全局索引: %d\n", int(src0(2)));
    printf("  src0(3) 对应全局索引: %d\n", int(src0(3)));
    printf("\n");

    printf("Thread 5 负责的源数据索引:\n");
    printf("  src5(0) 对应全局索引: %d\n", int(src5(0)));
    printf("  src5(1) 对应全局索引: %d\n", int(src5(1)));
    printf("  src5(2) 对应全局索引: %d\n", int(src5(2)));
    printf("  src5(3) 对应全局索引: %d\n", int(src5(3)));
    printf("\n");

    // 与 local_partition 的关系
    printf("=== 与 local_partition 的关系 ===\n");
    printf("tiled_copy.get_slice(tid).partition_S(tensor)\n");
    printf("  等价于\n");
    printf("local_partition(tensor, thread_layout, tid)\n\n");
    printf("get_slice 的优势:\n");
    printf("  1. 支持运行时 thread_id (local_partition 需要编译时常量)\n");
    printf("  2. 同时提供 partition_S 和 partition_D (源和目标可能不同)\n");
    printf("  3. 与 copy() 函数无缝配合\n");

    // 6.4 cute::copy 详解
    printf("\n--- 6.4 cute::copy ---\n");
    printf("copy 是执行数据拷贝的核心函数\n\n");

    printf("=== 函数签名 ===\n");
    printf("copy(TiledCopy const& tiled_copy, Tensor const& src, Tensor& dst);\n");
    printf("copy(Tensor const& src, Tensor& dst);  // 简化形式\n\n");

    printf("=== 参数说明 ===\n");
    printf("  tiled_copy: TiledCopy 对象，定义拷贝方式\n");
    printf("  src: 源 tensor (通过 partition_S 获取)\n");
    printf("  dst: 目标 tensor (通过 partition_D 获取)\n\n");

    printf("=== copy 内部行为 ===\n");
    printf("  1. 遍历 src tensor 的每个元素\n");
    printf("  2. 使用 TiledCopy 定义的 Copy_Atom 执行实际拷贝\n");
    printf("  3. 根据 ThrCopy 的分区结果，每个线程只处理自己的数据\n\n");

    // 演示 copy 的两种用法
    printf("=== copy 使用示例 ===\n\n");

    // 示例 1: 使用 TiledCopy
    printf("// 示例 1: 使用 TiledCopy 多线程协作拷贝\n");
    copy(tiled_copy, src0, dst0);
    printf("  copy(tiled_copy, src_part, dst_part) 完成\n");

    // 示例 2: 简单拷贝
    float simple_src[8], simple_dst[8];
    for (int i = 0; i < 8; ++i) { simple_src[i] = i * 10; simple_dst[i] = 0; }
    auto s_src = make_tensor(make_gmem_ptr(simple_src), Shape<_8>{});
    auto s_dst = make_tensor(make_gmem_ptr(simple_dst), Shape<_8>{});

    printf("\n// 示例 2: 简单拷贝 (不使用 TiledCopy)\n");
    printf("  拷贝前: ");
    for (int i = 0; i < 8; ++i) printf("%.0f ", simple_dst[i]);
    printf("\n");

    copy(s_src, s_dst);

    printf("  copy(src, dst) 后: ");
    for (int i = 0; i < 8; ++i) printf("%.0f ", simple_dst[i]);
    printf("\n\n");

    // 在 kernel 中的完整用法
    printf("=== 在 kernel 中的完整用法 ===\n");
    printf("__global__ void copy_kernel(float* src, float* dst, int N) {\n");
    printf("  auto g_src = make_tensor(make_gmem_ptr(src), make_shape(N));\n");
    printf("  auto g_dst = make_tensor(make_gmem_ptr(dst), make_shape(N));\n");
    printf("\n");
    printf("  // 定义 TiledCopy\n");
    printf("  using copy_atom = Copy_Atom<UniversalCopy<float>, float>;\n");
    printf("  auto tiled_copy = make_tiled_copy(copy_atom{}, Layout<Shape<_128>>{});\n");
    printf("\n");
    printf("  // 获取当前线程的分区\n");
    printf("  auto thr_copy = tiled_copy.get_slice(threadIdx.x);\n");
    printf("  auto src_part = thr_copy.partition_S(g_src);\n");
    printf("  auto dst_part = thr_copy.partition_D(g_dst);\n");
    printf("\n");
    printf("  // 执行拷贝\n");
    printf("  copy(tiled_copy, src_part, dst_part);\n");
    printf("}\n");
}

// ============================================================================
// Section 7: MMA_Atom 与 TiledMMA
// ============================================================================

// 7.4 完整的 MMA Kernel 示例 (API 演示)
// 适用于: SM80 (A100) 和 SM86 (RTX 3060/3070/3080/3090)
// RTX 3060 12G 计算能力: 8.6 (Ampere 架构)
#if 0  // 条件编译禁用，仅作示例参考
__global__ void mma_kernel(half_t* A, half_t* B, float* C) {
    using namespace cute;

    // 定义 MMA_Atom: SM80 16x8x16 FP32=FP16*FP16+FP32
    // 注意: SM86 (RTX 3060/3070/3080/3090) 完全兼容 SM80 的 MMA 操作
    using MMA_Op = SM80_16x8x16_F32F16F16F32_TN;
    using MMA_Atom_T = MMA_Atom<MMA_Op>;

    // 定义 TiledMMA: 2x2x1 的重复模式
    // 实际计算大小: M=16*2=32, N=8*2=16, K=16*1=16
    using TiledMMA_2x2x1 = TiledMMA<MMA_Atom_T, Layout<Shape<_2, _2, _1>>>;
    TiledMMA_2x2x1 tiled_mma;

    // 定义静态形状的 tensor (实际 kernel 中可能来自 shared memory)
    auto sA = make_tensor(make_smem_ptr(A), Layout<Shape<_32, _16>>{});  // 32x16
    auto sB = make_tensor(make_smem_ptr(B), Layout<Shape<_16, _16>>{});  // 16x16
    auto sC = make_tensor(make_smem_ptr(C), Layout<Shape<_32, _16>>{});  // 32x16

    // 获取当前线程的 MMA slice
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    // 创建寄存器 Fragment
    // partition_fragment_* 返回寄存器 tensor，供 MMA 使用
    auto tCrA = thr_mma.partition_fragment_A(sA);  // A fragment
    auto tCrB = thr_mma.partition_fragment_B(sB);  // B fragment
    auto tCrC = thr_mma.partition_fragment_C(sC);  // C 累加器

    // 初始化累加器为 0
    clear(tCrC);

    // 从 shared 加载数据到寄存器 fragment
    // copy(s2r_copy, sA_part, tCrA);
    // copy(s2r_copy, sB_part, tCrB);

    // GEMM 计算: C = A * B + C
    gemm(tiled_mma, tCrA, tCrB, tCrC);

    // 将结果写回 shared/global
    copy(tCrC, sC);
}
#endif

void section_07_mma_atom_and_tiled_mma() {
    printf("\n========== Section 7: MMA_Atom 与 TiledMMA ==========\n");

    // 7.1 MMA_Atom
    printf("\n--- 7.1 MMA_Atom ---\n");
    printf("MMA_Atom 封装单个硬件矩阵乘累加指令:\n");
    printf("  模板: MMA_Atom<MMA_Op>\n");
    printf("\n");
    printf("  常见 MMA_Op 类型 (TN 表示 A转置, B不转置):\n");
    printf("  ┌─────────────────────────────────────┬──────────────┬─────────────────────────┐\n");
    printf("  │ MMA_Op                              │ 架构         │ 形状 (MxNxK)            │\n");
    printf("  ├─────────────────────────────────────┼──────────────┼─────────────────────────┤\n");
    printf("  │ SM80_16x8x16_F32F16F16F32_TN        │ Ampere       │ 16x8x16, FP32=FP16*FP16 │\n");
    printf("  │ SM80_16x8x8_F32F16F16F32_TN         │ Ampere       │ 16x8x8, FP32=FP16*FP16  │\n");
    printf("  │ SM75_16x8x8_F32F16F16F32_TN         │ Turing       │ 16x8x8, FP32=FP16*FP16  │\n");
    printf("  │ SM70_16x16x8_F32F16F16F32_TN        │ Volta        │ 16x16x8, FP32=FP16*FP16 │\n");
    printf("  └─────────────────────────────────────┴──────────────┴─────────────────────────┘\n");
    printf("\n");
    printf("  GPU 架构对照:\n");
    printf("    - SM80: A100, A30, A10\n");
    printf("    - SM86: RTX 3090/3080/3070/3060 (Ampere消费级)\n");
    printf("    - SM86 完全兼容 SM80 的所有 MMA 操作\n");
    printf("\n");
    printf("  命名规则: SM<arch>_<M>x<N>x<K>_<CType><AType><BType><CType>_<Layout>\n");
    printf("  - M, N, K: 矩阵维度\n");
    printf("  - Types: F32=FP32, F16=FP16\n");
    printf("  - Layout: TN=A转置, TT=都转置, NT=都不转置, NN=B转置\n");
    printf("\n");
    printf("  [你的GPU] RTX 3060 12G: SM86 架构, 计算能力 8.6\n");
    printf("           推荐使用: SM80_16x8x16_F32F16F16F32_TN\n");

    // 代码示例
    printf("\n  代码示例:\n");
    printf("  ```cpp\n");
    printf("  using MMA_Op = SM80_16x8x16_F32F16F16F32_TN;\n");
    printf("  using MMA_Atom_T = MMA_Atom<MMA_Op>;\n");
    printf("  \n");
    printf("  // 查询 MMA_Atom 属性\n");
    printf("  constexpr int mm = size<0>(MMA_Atom_T{});  // M = 16\n");
    printf("  constexpr int nn = size<1>(MMA_Atom_T{});  // N = 8\n");
    printf("  constexpr int kk = size<2>(MMA_Atom_T{});  // K = 16\n");
    printf("  ```\n");

    // 7.2 TiledMMA
    printf("\n--- 7.2 TiledMMA ---\n");
    printf("TiledMMA 将 MMA_Atom tile 化，支持更大的矩阵块:\n");
    printf("\n");
    printf("  模板: TiledMMA<MMA_Atom, Layout<Shape<M_Rep,N_Rep,K_Rep>>>\n");
    printf("  - M_Rep, N_Rep, K_Rep: 在 M/N/K 维度的重复次数\n");
    printf("  - 实际计算大小 = (M*M_Rep) x (N*N_Rep) x (K*K_Rep)\n");
    printf("\n");
    printf("  TiledMMA 配置示例:\n");
    printf("  ┌─────────────────────┬────────────────┬────────────────────────┐\n");
    printf("  │ TiledMMA Shape      │ MMA Atom       │ 实际计算大小           │\n");
    printf("  ├─────────────────────┼────────────────┼────────────────────────┤\n");
    printf("  │ Shape<_1,_1,_1>     │ 16x8x16        │ 16x8x16 (单指令)       │\n");
    printf("  │ Shape<_2,_1,_1>     │ 16x8x16        │ 32x8x16  (M扩展)       │\n");
    printf("  │ Shape<_1,_2,_1>     │ 16x8x16        │ 16x16x16 (N扩展)       │\n");
    printf("  │ Shape<_2,_2,_1>     │ 16x8x16        │ 32x16x16 (M,N扩展)     │\n");
    printf("  │ Shape<_2,_2,_2>     │ 16x8x16        │ 32x16x32 (M,N,K扩展)   │\n");
    printf("  └─────────────────────┴────────────────┴────────────────────────┘\n");

    // 代码示例
    printf("\n  代码示例:\n");
    printf("  ```cpp\n");
    printf("  using MMA_Op = SM80_16x8x16_F32F16F16F32_TN;\n");
    printf("  using MMA_Atom_T = MMA_Atom<MMA_Op>;\n");
    printf("  \n");
    printf("  // 定义 TiledMMA: 2x2x1 重复\n");
    printf("  using TiledMMA_2x2x1 = TiledMMA<MMA_Atom_T, Layout<Shape<_2, _2, _1>>>;\n");
    printf("  TiledMMA_2x2x1 tiled_mma;\n");
    printf("  \n");
    printf("  // 查询 TiledMMA 属性\n");
    printf("  constexpr int warp_size = size(TiledMMA_2x2x1{});  // warp 大小 (通常为32)\n");
    printf("  ```\n");

    // 7.3 partition_fragment
    printf("\n--- 7.3 partition_fragment 详解 ---\n");
    printf("partition_fragment_* 将全局/共享内存 tensor 分区到寄存器 fragment:\n");
    printf("\n");
    printf("  API 说明:\n");
    printf("  ```cpp\n");
    printf("  auto thr_mma = tiled_mma.get_slice(threadIdx.x);\n");
    printf("  \n");
    printf("  // 创建寄存器 fragment (实际分配寄存器)\n");
    printf("  auto tCrA = thr_mma.partition_fragment_A(tensor_A);  // A fragment\n");
    printf("  auto tCrB = thr_mma.partition_fragment_B(tensor_B);  // B fragment\n");
    printf("  auto tCrC = thr_mma.partition_fragment_C(tensor_C);  // C accumulator\n");
    printf("  ```\n");
    printf("\n");
    printf("  关键概念:\n");
    printf("  - get_slice(thread_id): 获取指定线程的 MMA 视图\n");
    printf("  - partition_fragment_*: 返回 Tensor 视图，指向线程私有的寄存器\n");
    printf("  - tCrA/tCrB: 输入数据，需要从 shared/global 加载\n");
    printf("  - tCrC: 累加器，需要初始化为 0\n");

    // 7.4 完整 GEMM 流程
    printf("\n--- 7.4 完整 GEMM 流程 (kernel 内) ---\n");
    printf("  ```cpp\n");
    printf("  __global__ void gemm_kernel(half_t* A, half_t* B, float* C) {\n");
    printf("    // 1. 定义 TiledMMA\n");
    printf("    using MMA_Op = SM80_16x8x16_F32F16F16F32_TN;\n");
    printf("    using TiledMMA = TiledMMA<MMA_Atom<MMA_Op>, Layout<Shape<_2,_2,_1>>>;\n");
    printf("    TiledMMA tiled_mma;\n");
    printf("    \n");
    printf("    // 2. 创建全局内存 tensor\n");
    printf("    auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K));\n");
    printf("    auto gB = make_tensor(make_gmem_ptr(B), make_shape(K, N));\n");
    printf("    auto gC = make_tensor(make_gmem_ptr(C), make_shape(M, N));\n");
    printf("    \n");
    printf("    // 3. 获取当前 block 的 tile\n");
    printf("    auto tCgC = local_tile(gC, cta_tile, block_coord);\n");
    printf("    \n");
    printf("    // 4. 获取线程的 MMA slice\n");
    printf("    auto thr_mma = tiled_mma.get_slice(threadIdx.x);\n");
    printf("    \n");
    printf("    // 5. 创建寄存器 fragment\n");
    printf("    auto tCrA = thr_mma.partition_fragment_A(sA);  // sA=shared memory\n");
    printf("    auto tCrB = thr_mma.partition_fragment_B(sB);\n");
    printf("    auto tCrC = thr_mma.partition_fragment_C(tCgC);\n");
    printf("    \n");
    printf("    // 6. 初始化累加器\n");
    printf("    clear(tCrC);\n");
    printf("    \n");
    printf("    // 7. 执行 MMA 计算\n");
    printf("    gemm(tiled_mma, tCrA, tCrB, tCrC);\n");
    printf("    \n");
    printf("    // 8. 写回结果\n");
    printf("    copy(tCrC, tCgC);\n");
    printf("  }\n");
    printf("  ```\n");

    // 7.5 数据流总结
    printf("\n--- 7.5 MMA 数据流总结 ---\n");
    printf("  ```\n");
    printf("  Global Memory (gA, gB)        Register (rC accumulators)\n");
    printf("         │                              ▲\n");
    printf("         │ TiledCopy (cp.async)         │\n");
    printf("         ▼                              │\n");
    printf("  Shared Memory (sA, sB) ──TiledCopy───┘\n");
    printf("         │         (ldmatrix)\n");
    printf("         ▼\n");
    printf("  Register (rA, rB)\n");
    printf("         │\n");
    printf("         │ TiledMMA (mma.sync)\n");
    printf("         ▼\n");
    printf("  Register (rC updated)\n");
    printf("  ```\n");
    printf("\n");
    printf("  完整 GEMM pipeline:\n");
    printf("  1. Global → Shared: cp.async 异步拷贝 + Swizzle 优化\n");
    printf("  2. Shared → Register: ldmatrix 加载到寄存器\n");
    printf("  3. MMA 计算: gemm(tiled_mma, rA, rB, rC)\n");
    printf("  4. 循环 K 维度，累加结果\n");
    printf("  5. Register → Global: 写回最终结果\n");

    printf("\n  [提示] 完整 kernel 示例代码见 mma_kernel 函数 (本文件第 ~830行)\n");
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