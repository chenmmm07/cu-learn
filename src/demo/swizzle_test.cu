// CUTE headers
#include <cute/tensor.hpp>
#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/stride.hpp"
#include "cute/tensor_impl.hpp"
#include "cute/util/print_tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"

using namespace cute;

// ============================================================================
// 示例1: composition之后的layout详解
// ============================================================================
void example_composition_layout() {
    printf("\n=== Example 1: Composition Layout ===\n");

    using BaseLayout = Layout<Shape<_16,_32>, Stride<_32,_1>>;
    auto base_layout = BaseLayout{};
    printf("Base Layout (8x8, row-major):\n");
    print_layout(base_layout);

    // 应用Swizzle变换来优化shared memory访问
    // Swizzle<2,0,3>: 对地址位进行XOR操作，减少bank conflict
    //   - 第一个2: 你要混洗的行数为2^2 = 4行
    //   - 第二个0: 你要固定的列数为2^0 = 1列
    //   - 第三个3: 
    // 这会将原本的线性地址映射到一个"混洗"后的地址，避免多个线程访问同一bank
    auto swizzled_layout = composition(Swizzle<2,3,3>{}, base_layout);
    printf("\nSwizzled Layout (减少bank conflicts):\n");
    print_layout(swizzled_layout);

    // 3. composition的作用：
    // - 将两个layout组合，新的layout = Swizzle(BaseLayout(coord))
    // - 用于实现复杂的地址映射，如shared memory的swizzle优化
    // - 使得访问模式更加高效，避免bank conflicts

    // 测试几个坐标映射
    printf("\n坐标映射示例 (row, col) -> linear_index:\n");
    printf("  (0,0) -> %d (base) vs %d (swizzled)\n",
           base_layout(0,0), swizzled_layout(0,0));
    printf("  (0,1) -> %d (base) vs %d (swizzled)\n",
           base_layout(0,1), swizzled_layout(0,1));
    printf("  (1,0) -> %d (base) vs %d (swizzled)\n",
           base_layout(1,0), swizzled_layout(1,0));
    printf("  可以看到swizzle改变了地址映射关系\n");
}

void example_composition_layout_2() {
    // 基础 row-major layout：offset = row*32 + col
    constexpr int M = 8;
    constexpr int N = 8;

    // 逻辑 shape 相同
    auto shape = Shape<Int<M>, Int<N>>{};

    // 普通 row-major：offset = m*N + n
    auto rm = make_layout(shape, Stride<Int<N>, Int<1>>{});

    // 加 swizzle（示例参数随便取一个能工作的形态；真实项目要按 bank/向量化选参）
    auto swz_layout = composition(Swizzle<3,0,3>{}, rm);

    // 两块不同的物理内存
    alignas(16) int a[M*N];
    alignas(16) int b[M*N];

    // 构造两个 tensor，注意：shape 一样，layout 不一样
    auto t_rm  = make_tensor(make_gmem_ptr(a), rm);
    auto t_swz = make_tensor(make_gmem_ptr(b), swz_layout);

    // 写入：用法完全一致。在指定layout之后，我们使用tensor其实就可以不用关心其物理上是如何排列的了
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            t_rm(m,n)  = 100*m + n;
            t_swz(m,n) = 100*m + n;
        }
    }
}

void example_swizzle_layout() {
    // 基础 row-major layout：offset = row*32 + col
    constexpr int M = 8;
    constexpr int N = 8;
    int a[M*N];
    int b[M*N];
    // 逻辑 shape 相同
    auto shape = Shape<Int<M>, Int<N>>{};
    auto base_layout = make_layout(shape, GenRowMajor{});
    auto a_tensor = make_tensor(make_gmem_ptr(a), base_layout);
    // print_layout(base_layout);

    auto swizzled_layout = composition(Swizzle<2,1,2>{}, base_layout);
    auto b_tensor = make_tensor(make_gmem_ptr(b), swizzled_layout);
    // print_layout(swizzled_layout);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            a_tensor(m,n) = n;
            b_tensor(m,n) = n;
        }
    }

    auto c_tensor = make_tensor(make_gmem_ptr(b), base_layout);
    print_tensor(a_tensor);
    print_tensor(c_tensor);
}
// ============================================================================
// 示例2: local_tile详解
// ============================================================================
void example_local_tile() {
    printf("\n=== Example 2: local_tile (logical_divide) ===\n");

    // 创建一个大的layout: 16x16，并填充实际数据
    constexpr int M = 16;
    constexpr int N = 16;
    int data[M*N];
    for(int i=0; i<M*N; i++) data[i] = i;

    auto global_layout = make_layout(make_shape(Int<M>{}, Int<N>{}), GenRowMajor{});
    auto global_tensor = make_tensor(make_gmem_ptr(data), global_layout);

    printf("Global Tensor (16x16):\n");
    print_tensor(global_tensor);

    // logical_divide将大的layout分割成小的tile
    // 这里将16x16分割成4x4的tiles
    // 参数: 原layout, tile的shape (使用tuple形式)
    auto tile_shape = make_tile(Int<4>{}, Int<4>{});
    auto tiled_layout = logical_divide(global_layout, tile_shape);
    printf("\nTiled Layout (分割成4x4的tiles):\n");
    print_layout(tiled_layout);

    // 实际效果演示：使用tiled_layout访问数据
    // tiled_layout现在是一个两层结构：(4,4) x (4,4)
    // 第一层(4,4): tile内部坐标
    // 第二层(4,4): tile的索引
    printf("\n=== 实际效果演示 ===\n");

    // 遍历所有tiles
    printf("\n通过tiled_layout遍历tile (tile_m, tile_n):\n");
    for(int tile_m = 0; tile_m < 4; ++tile_m) {
        for(int tile_n = 0; tile_n < 4; ++tile_n) {
            printf("Tile (%d,%d): ", tile_m, tile_n);
            // 打印这个tile的前4个元素
            for(int elem = 0; elem < 4; ++elem) {
                // 使用tile坐标访问：(tile_m,tile_n, 0,elem)
                int linear_idx = tiled_layout(tile_m, tile_n, 0, elem);
                printf("%3d ", linear_idx);
            }
            printf("...\n");
        }
    }

    // logical_divide的效果：
    // - 原来的16x16被分成了 (4,4) x (4,4) 的结构
    // - 第一层(4,4): 内部tile的大小 (4x4)
    // - 第二层(4,4): tile的数量 (16/4 = 4个tiles在每个维度)
    // - 这样可以方便地迭代处理每个tile

    printf("\n应用场景: 将大矩阵分解成小块进行处理\n");
    printf("  - 例如将128x128的矩阵分成16x16个8x8的tile\n");
    printf("  - 每个thread block处理一个或多个tile\n");
    printf("  - logical_divide是实现tiling的核心函数\n");
}

// ============================================================================
// 示例3: local_partition详解
// ============================================================================
void example_local_partition() {
    printf("\n=== Example 3: local_partition ===\n");

    // 创建一个tensor
    int data[16];
    for(int i=0; i<16; i++) data[i] = i*10;  // 填充数据: 0, 10, 20, ..., 150
    auto tensor = make_tensor(make_gmem_ptr(data), make_shape(Int<16>{}), GenRowMajor{});

    printf("Original Tensor (16 elements):\n");
    print_tensor(tensor);

    // local_partition将tensor按照某个layout进行分区
    // 假设我们有4个线程，每个线程处理4个元素
    auto thread_layout = make_layout(make_shape(Int<4>{}), GenRowMajor{}); // 4个线程

    // local_partition的作用：
    // - 将tensor按照thread_layout进行划分
    // - 每个线程得到一个连续的子区域
    auto partitioned = local_partition(tensor, thread_layout, Int<0>{}); // thread 0
    printf("\nThread 0's partition (连续4个元素):\n");
    print_tensor(partitioned);

    auto partitioned_1 = local_partition(tensor, thread_layout, Int<1>{}); // thread 1
    printf("\nThread 1's partition (接下来的4个元素):\n");
    print_tensor(partitioned_1);

    auto partitioned_2 = local_partition(tensor, thread_layout, Int<2>{}); // thread 2
    printf("\nThread 2's partition (接下来的4个元素):\n");
    print_tensor(partitioned_2);

    auto partitioned_3 = local_partition(tensor, thread_layout, Int<3>{}); // thread 3
    printf("\nThread 3's partition (最后的4个元素):\n");
    print_tensor(partitioned_3);

    // 实际效果演示：修改某个线程的分区
    printf("\n=== 修改Thread 1的分区数据 ===\n");
    for(int i=0; i<size(partitioned_1); ++i) {
        partitioned_1(i) = 999;  // 将thread 1的所有元素设为999
    }

    printf("修改后的Original Tensor (可以看到Thread 1负责的位置变成999):\n");
    print_tensor(tensor);

    // 二维tensor的partition示例
    printf("\n=== 二维tensor的local_partition示例 ===\n");
    int data2d[4*4];
    for(int i=0; i<16; i++) data2d[i] = i;
    auto tensor2d = make_tensor(make_gmem_ptr(data2d), make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});

    printf("Original 2D Tensor (4x4):\n");
    print_tensor(tensor2d);

    // 假设有2x2的线程grid
    auto thread_layout_2d = make_layout(make_shape(Int<2>{}, Int<2>{}), GenRowMajor{});

    // (0,0)线程: 负责左上角的2x2子块
    auto part_2d_00 = local_partition(tensor2d, thread_layout_2d, Int<0>{}, Int<0>{});
    printf("\nThread (0,0)'s partition (左上角2x2):\n");
    print_tensor(part_2d_00);

    // (1,1)线程: 负责右下角的2x2子块
    auto part_2d_11 = local_partition(tensor2d, thread_layout_2d, Int<1>{}, Int<1>{});
    printf("\nThread (1,1)'s partition (右下角2x2):\n");
    print_tensor(part_2d_11);

    printf("\n应用场景:\n");
    printf("  - 在cooperative groups中，将数据分配给不同的线程\n");
    printf("  - 每个线程知道自己负责哪部分数据\n");
    printf("  - 实现SIMT并行处理\n");
}

// ============================================================================
// 示例4: TiledMMA详解
// ============================================================================
void example_tiled_mma() {
    printf("\n=== Example 4: TiledMMA ===\n");

    // TiledMMA是CuTe中用于描述矩阵乘法的核心概念
    // 它将MMA (Matrix Multiply-Accumulate)指令tile化

    // 使用SM80的MMA指令: 16x8x16 (M x N x K)
    // 这表示一次MMA操作处理:
    //   - A矩阵: 16x16 (MxK)
    //   - B矩阵: 16x8  (KxN)
    //   - C矩阵: 16x8  (MxN)
    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_atom = MMA_Atom<mma_op>;

    // TiledMMA将MMA_Atom进一步tile化
    // 例如，一个warp可以执行多个MMA指令
    // 参数说明:
    //   - MMA_Atom: 基础的MMA操作单元
    //   - Layout: 描述如何在多个线程间重复这个atom
    using TiledMMA_Type = TiledMMA<
        mma_atom,
        Layout<Shape<_1,_1,_1>>  // 1x1x1 的tile重复 (不重复，使用单个MMA)
    >;

    TiledMMA_Type mma;

    printf("TiledMMA结构:\n");
    printf("  - 基础MMA操作: 16x8x16 (MxNxK)\n");
    printf("  - Tile重复: 1x1x1 (使用单个MMA指令)\n");

    // 实际效果演示：查询TiledMMA的各种属性
    printf("\n=== TiledMMA属性查询 ===\n");
    printf("Thread Layout (线程如何组织):\n");
    print_layout(mma.get_thr_layout());

    printf("\nValLayout A (A矩阵每个线程的数据布局):\n");
    print_layout(mma.get_val_layout_A());

    printf("\nValLayout B (B矩阵每个线程的数据布局):\n");
    print_layout(mma.get_val_layout_B());

    printf("\nValLayout C (C矩阵每个线程的数据布局):\n");
    print_layout(mma.get_val_layout_C());

    // 使用不同的tile重复模式
    printf("\n=== 2x2x1 Tile重复示例 ===\n");
    using TiledMMA_Type_2x2x1 = TiledMMA<
        mma_atom,
        Layout<Shape<_2,_2,_1>>  // M维x2, N维x2, K维x1
    >;
    TiledMMA_Type_2x2x1 mma_2x2x1;

    printf("Thread Layout (2x2x1重复):\n");
    print_layout(mma_2x2x1.get_thr_layout());

    printf("\n总计算量:\n");
    printf("  - 基础MMA: 16x8x16\n");
    printf("  - Tile重复: 2x2x1\n");
    printf("  - 总计: (16*2)x(8*2)x16 = 32x16x16 per iteration\n");

    printf("\nTiledMMA的核心概念:\n");
    printf("  - MMA_Atom: 封装单个硬件MMA指令 (如mma.sync.m16n8k16)\n");
    printf("  - TiledMMA: 将多个MMA_Atom组合，形成更大的计算tile\n");
    printf("  - 自动计算线程到数据的映射关系\n");

    printf("\nTiledMMA提供的关键操作:\n");
    printf("  - partition_fragment_A: 创建A矩阵的寄存器fragment\n");
    printf("  - partition_fragment_B: 创建B矩阵的寄存器fragment\n");
    printf("  - partition_fragment_C: 创建C矩阵的累加器fragment\n");
    printf("  - 这些fragment用于实际的矩阵乘法计算\n");

    printf("\n应用场景: GEMM kernel的核心\n");
    printf("  - TiledMMA封装了底层的mma.sync指令\n");
    printf("  - 自动处理线程到数据的映射关系\n");
    printf("  - 支持不同的MMA指令(SM70, SM75, SM80, SM90等)\n");
}

// ============================================================================
// 示例5: partition_fragment_A详解
// ============================================================================
void example_partition_fragment() {
    printf("\n=== Example 5: partition_fragment_A ===\n");

    // partition_fragment_A是TiledMMA的成员函数
    // 它为当前线程创建一个fragment (寄存器中的小tensor)

    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_atom = MMA_Atom<mma_op>;
    using TiledMMA_Type = TiledMMA<mma_atom, Layout<Shape<_1,_1,_1>>>;

    TiledMMA_Type mma;

    // 创建一个A矩阵的tensor (global memory模拟)
    // Shape: 16x16 (匹配MMA的M和K维度)
    cutlass::half_t A_data[16*16];
    for(int i=0; i<16*16; i++) A_data[i] = cutlass::half_t(i);
    auto A_tensor = make_tensor(make_gmem_ptr(A_data),
                                make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{}));

    printf("A矩阵原始数据 (16x16 MxK维度):\n");
    print_tensor(A_tensor);

    // 使用TiledMMA的get_slice获取线程视图
    int thread_id = 0; // 假设是thread 0

    // partition_fragment_A的作用：
    // 1. 根据MMA指令的要求，确定该线程需要哪些A矩阵元素
    // 2. 创建一个register fragment来存储这些元素
    // 3. 返回一个适合该线程的tensor view

    auto thr_mma = mma.get_slice(thread_id);
    auto tCrA = thr_mma.partition_fragment_A(make_shape(Int<16>{}, Int<16>{}));

    printf("\n=== 实际效果演示 ===\n");
    printf("Thread %d的A fragment (寄存器布局):\n", thread_id);
    print_layout(tCrA.layout());

    printf("\nFragment shape: (%d, %d)\n", size<0>(tCrA), size<1>(tCrA));
    printf("Fragment总元素数: %d\n", size(tCrA));

    // 查询fragment属性
    printf("\nFragment属性:\n");
    printf("  - 每个线程需要从A矩阵中取 %d 个元素\n", size(tCrA));
    printf("  - 这些元素排列在寄存器中，布局如上所示\n");
    printf("  - fragment的shape由MMA指令和线程布局决定\n");

    // 展示不同线程的fragment布局
    printf("\n=== 不同线程的fragment布局对比 ===\n");
    for(int tid = 0; tid < 8; ++tid) {
        auto thr_mma_tid = mma.get_slice(tid);
        auto tCrA_tid = thr_mma_tid.partition_fragment_A(make_shape(Int<16>{}, Int<16>{}));
        printf("Thread %d: fragment shape = (%d, %d)\n",
               tid, size<0>(tCrA_tid), size<1>(tCrA_tid));
    }

    // partition_fragment_B的示例
    printf("\n=== partition_fragment_B示例 ===\n");
    cutlass::half_t B_data[16*8];  // KxN = 16x8
    for(int i=0; i<16*8; i++) B_data[i] = cutlass::half_t(i*0.5f);
    auto B_tensor = make_tensor(make_gmem_ptr(B_data),
                                make_layout(make_shape(Int<16>{}, Int<8>{}), GenRowMajor{}));

    printf("B矩阵原始数据 (16x8 KxN维度):\n");
    print_tensor(B_tensor);

    auto tCrB = thr_mma.partition_fragment_B(make_shape(Int<16>{}, Int<8>{}));
    printf("\nThread %d的B fragment (寄存器布局):\n", thread_id);
    print_layout(tCrB.layout());
    printf("Fragment shape: (%d, %d)\n", size<0>(tCrB), size<1>(tCrB));

    // partition_fragment_C的示例
    printf("\n=== partition_fragment_C示例 ===\n");
    float C_data[16*8];  // MxN = 16x8
    for(int i=0; i<16*8; i++) C_data[i] = 0.0f;
    auto C_tensor = make_tensor(make_gmem_ptr(C_data),
                                make_layout(make_shape(Int<16>{}, Int<8>{}), GenRowMajor{}));

    auto tCrC = thr_mma.partition_fragment_C(make_shape(Int<16>{}, Int<8>{}));
    printf("Thread %d的C fragment (累加器寄存器布局):\n", thread_id);
    print_layout(tCrC.layout());
    printf("Fragment shape: (%d, %d)\n", size<0>(tCrC), size<1>(tCrC));

    printf("\npartition_fragment系列函数:\n");
    printf("  - partition_fragment_A: 为A矩阵创建寄存器fragment\n");
    printf("  - partition_fragment_B: 为B矩阵创建寄存器fragment\n");
    printf("  - partition_fragment_C: 为C矩阵创建累加器fragment\n");

    printf("\n应用场景:\n");
    printf("  - GEMM kernel中，从shared memory加载数据到寄存器\n");
    printf("  - 确保每个线程加载正确的数据片段\n");
    printf("  - 与copy操作配合，实现高效的数据移动\n");
    printf("  - fragment是执行MMA计算的输入/输出\n");
}

// ============================================================================
// 示例6: Copy_Traits和Copy_Atom详解
// ============================================================================
void example_copy_traits_and_atom() {
    printf("\n=== Example 6: Copy_Traits & Copy_Atom ===\n");

    // Copy_Atom是CuTe中描述硬件copy指令的基本单元
    // Copy_Traits定义了copy操作的特性

    // 例1: 使用UniversalCopy (最简单的copy，直接赋值)
    // UniversalCopy可以处理任意类型的数据
    using copy_atom_universal = Copy_Atom<UniversalCopy<float>, float>;

    printf("=== Copy_Atom: UniversalCopy<float> ===\n");
    printf("  - 操作: 通用的直接赋值 (dst = src)\n");
    printf("  - 适用于: 任意内存空间的copy\n");

    // 查询Copy_Traits
    using traits_universal = Copy_Traits<copy_atom_universal>;
    printf("\nCopy_Traits<UniversalCopy<float>>提供的信息:\n");
    printf("  - ThrID layout (线程布局):\n");
    print_layout(traits_universal::ThrLayout{});
    printf("  - ValLayout Src (源数据布局):\n");
    print_layout(traits_universal::ValLayoutSrc{});
    printf("  - ValLayout Dst (目标数据布局):\n");
    print_layout(traits_universal::ValLayoutDst{});

    // 例2: 使用ldmatrix指令 (从shared memory加载到寄存器)
    // ldmatrix.x4.m8n8 一次加载8x8的矩阵
    using copy_atom_ldmatrix = Copy_Atom<SM75_U32x4_LDSM_N, cutlass::half_t>;

    printf("\n=== Copy_Atom: SM75_U32x4_LDSM_N ===\n");
    printf("  - 操作: 从shared memory加载数据 (load matrix)\n");
    printf("  - 硬件指令: ldmatrix.sync.aligned.x4.m8n8\n");
    printf("  - 一次加载: 4个32bit值 = 128bit = 8个half_t\n");
    printf("  - 线程协作: 多个线程协同加载一个8x8 tile\n");

    // Copy_Traits描述了这个copy操作的特征:
    using traits_ldmatrix = Copy_Traits<copy_atom_ldmatrix>;
    printf("\nCopy_Traits<SM75_U32x4_LDSM_N>提供的信息:\n");
    printf("  - ThrID layout (线程布局):\n");
    print_layout(traits_ldmatrix::ThrLayout{});
    printf("  - ValLayout Src (源数据布局):\n");
    print_layout(traits_ldmatrix::ValLayoutSrc{});
    printf("  - ValLayout Dst (目标数据布局):\n");
    print_layout(traits_ldmatrix::ValLayoutDst{});

    // 实际效果演示
    printf("\n=== 实际效果演示 ===\n");
    printf("ThrLayout的维度: (%d, %d)\n",
           size<0>(traits_ldmatrix::ThrLayout{}),
           size<1>(traits_ldmatrix::ThrLayout{}));
    printf("ValLayoutSrc的维度: (%d, %d)\n",
           size<0>(traits_ldmatrix::ValLayoutSrc{}),
           size<1>(traits_ldmatrix::ValLayoutSrc{}));
    printf("说明:\n");
    printf("  - ThrLayout: 定义了参与copy的线程如何组织 (如8x4 = 32个线程)\n");
    printf("  - ValLayoutSrc: 每个线程在源数据中看到的数据排列\n");
    printf("  - ValLayoutDst: 每个线程在目标数据中看到的数据排列\n");

    printf("\n应用场景:\n");
    printf("  - Copy_Atom: 封装不同的硬件copy指令\n");
    printf("  - Copy_Traits: 查询copy操作的特性，用于构造TiledCopy\n");
    printf("  - 自动处理线程协作和数据对齐\n");
}

// ============================================================================
// 示例7: make_tiled_copy详解
// ============================================================================
void example_make_tiled_copy() {
    printf("\n=== Example 7: make_tiled_copy ===\n");

    // make_tiled_copy将Copy_Atom tile化，创建TiledCopy
    // TiledCopy描述了多个线程协同执行的copy操作

    // 例子1: 使用UniversalCopy (最简单的copy)
    // UniversalCopy可以处理任意类型的数据，使用直接赋值
    using copy_atom = Copy_Atom<UniversalCopy<float>, float>;

    // 创建一个TiledCopy: 16个线程，每个线程copy 4个float
    auto tiled_copy = make_tiled_copy(
        copy_atom{},
        Layout<Shape<_16, _4>>{}  // 16个线程，每个处理4个元素
    );

    printf("TiledCopy结构:\n");
    printf("  - Copy_Atom: UniversalCopy<float> (通用的赋值操作)\n");
    printf("  - Thread layout: Shape<16, 4> (16x4=64个工作项)\n");
    printf("  - 总吞吐: 16 * 4 = 64 个float per copy iteration\n");

    // 实际效果演示：查询TiledCopy的属性
    printf("\n=== TiledCopy属性查询 ===\n");
    printf("Thread Layout:\n");
    print_layout(tiled_copy.get_TiledLayout());

    // 创建源和目标数据
    float src_data[64];
    float dst_data[64];
    for(int i=0; i<64; i++) src_data[i] = i * 0.5f;
    for(int i=0; i<64; i++) dst_data[i] = 0.0f;

    auto src_tensor = make_tensor(make_gmem_ptr(src_data), make_shape(Int<64>{}));
    auto dst_tensor = make_tensor(make_gmem_ptr(dst_data), make_shape(Int<64>{}));

    printf("\n源数据 (前16个元素):\n");
    for(int i=0; i<16; i++) printf("%.1f ", src_data[i]);
    printf("...\n");

    // 执行copy操作
    printf("\n=== 实际copy操作演示 ===\n");
    for(int tid = 0; tid < 16; ++tid) {
        auto thr_copy = tiled_copy.get_slice(tid);
        auto src_partition = thr_copy.partition_S(src_tensor);
        auto dst_partition = thr_copy.partition_D(dst_tensor);

        // 执行copy
        copy(tiled_copy, src_partition, dst_partition);
    }

    printf("Copy完成后的目标数据:\n");
    for(int i=0; i<16; i++) printf("%.1f ", dst_data[i]);
    printf("...\n");

    // 验证每个线程的分区
    printf("\n=== 不同线程的分区验证 ===\n");
    for(int tid = 0; tid < 4; ++tid) {
        auto thr_copy = tiled_copy.get_slice(tid);
        auto src_partition = thr_copy.partition_S(src_tensor);

        printf("Thread %d 的源分区: ", tid);
        for(int i=0; i<size(src_partition) && i<4; ++i) {
            printf("%.1f ", src_partition(i));
        }
        printf("\n");
    }

    printf("\nmake_tiled_copy的参数:\n");
    printf("  1. Copy_Atom: 定义基础copy操作 (如ldmatrix, cp.async等)\n");
    printf("  2. Thread Layout: 定义线程如何组织和分配工作\n");
    printf("  3. (可选) Value Layout: 定义每个线程处理的数据排列\n");

    // TiledCopy提供的关键函数:
    printf("\nTiledCopy的关键函数:\n");
    printf("  1. get_slice(thread_id): 获取某个线程的视图\n");
    printf("  2. partition_S(src_tensor): 将源tensor分区给各线程\n");
    printf("  3. partition_D(dst_tensor): 将目标tensor分区给各线程\n");
    printf("  4. copy(): 执行实际的拷贝操作\n");

    printf("\n不同类型的Copy_Atom:\n");
    printf("  - UniversalCopy: 通用赋值，适用于任何情况\n");
    printf("  - SM75_U32x4_LDSM_N: ldmatrix指令，从shared memory加载\n");
    printf("  - SM80_CP_ASYNC: cp.async指令，异步copy到shared memory\n");
    printf("  - AutoVectorizingCopy: 自动向量化的copy\n");

    printf("\n应用场景:\n");
    printf("  - Global -> Shared memory copy (使用cp.async)\n");
    printf("  - Shared -> Register copy (使用ldmatrix)\n");
    printf("  - Register -> Shared/Global memory (使用store)\n");
    printf("  - 自动向量化和合并访问\n");
}

// ============================================================================
// 示例8: partition_S详解
// ============================================================================
void example_partition_S() {
    printf("\n=== Example 8: partition_S ===\n");

    // partition_S是TiledCopy的成员函数
    // S代表Source，即对源tensor进行分区

    // 创建一个源tensor (global memory中的数据)
    float src_data[128];
    for(int i=0; i<128; i++) src_data[i] = i * 0.1f;

    auto src_tensor = make_tensor(make_gmem_ptr(src_data),
                                  make_shape(Int<128>{}));

    printf("Source Tensor (128 elements, 前16个):\n");
    for(int i=0; i<16; i++) printf("%.1f ", src_data[i]);
    printf("...\n");

    // 创建一个TiledCopy: 16个线程
    using copy_atom = Copy_Atom<UniversalCopy<float>, float>;
    auto tiled_copy = make_tiled_copy(
        copy_atom{},
        Layout<Shape<_16>>{}  // 16 threads
    );

    // 获取thread 0的视图
    int tid = 0;
    auto thr_copy = tiled_copy.get_slice(tid);

    // partition_S的作用:
    // 1. 根据TiledCopy的thread layout，将源tensor分区
    // 2. 返回当前线程负责的源数据部分
    // 3. 返回的是一个tensor view，指向正确的内存位置
    auto src_partition = thr_copy.partition_S(src_tensor);

    printf("\n=== 实际效果演示 ===\n");
    printf("Thread %d的源分区 (partition_S):\n", tid);
    printf("  - 该线程负责copy源tensor中的某些元素\n");
    printf("  - 分区大小: %d 个元素\n", size(src_partition));
    printf("  - 分区数据: ");
    for(int i=0; i<size(src_partition) && i<8; ++i) {
        printf("%.1f ", src_partition(i));
    }
    printf("...\n");

    // 对比不同线程的partition
    auto thr_copy_1 = tiled_copy.get_slice(1);
    auto src_partition_1 = thr_copy_1.partition_S(src_tensor);

    printf("\nThread 1的源分区:\n");
    printf("  - 与thread 0的partition不重叠\n");
    printf("  - 分区大小: %d 个元素\n", size(src_partition_1));
    printf("  - 分区数据: ");
    for(int i=0; i<size(src_partition_1) && i<8; ++i) {
        printf("%.1f ", src_partition_1(i));
    }
    printf("...\n");

    // 展示所有线程的分区
    printf("\n=== 所有线程的分区索引范围 ===\n");
    for(int t = 0; t < 16; ++t) {
        auto thr = tiled_copy.get_slice(t);
        auto part = thr.partition_S(src_tensor);
        printf("Thread %2d: 分区大小 = %2d, 首元素 = %.1f\n",
               t, size(part), part(0));
    }

    // 使用partition_S和partition_D完成copy
    printf("\n=== 完整copy操作演示 ===\n");
    float dst_data[128];
    for(int i=0; i<128; i++) dst_data[i] = 0.0f;
    auto dst_tensor = make_tensor(make_gmem_ptr(dst_data), make_shape(Int<128>{}));

    // 执行copy
    for(int t = 0; t < 16; ++t) {
        auto thr = tiled_copy.get_slice(t);
        auto src_part = thr.partition_S(src_tensor);
        auto dst_part = thr.partition_D(dst_tensor);
        copy(tiled_copy, src_part, dst_part);
    }

    printf("Copy完成后的目标数据 (前16个):\n");
    for(int i=0; i<16; i++) printf("%.1f ", dst_data[i]);
    printf("...\n");

    printf("\n应用场景:\n");
    printf("  - 多线程协同copy时，确定每个线程的工作范围\n");
    printf("  - 避免数据竞争和重复copy\n");
    printf("  - 与partition_D配对使用，完成src->dst的映射\n");

    // 完整的copy流程:
    printf("\n完整流程:\n");
    printf("  1. 创建TiledCopy (定义copy策略)\n");
    printf("  2. get_slice(tid) (获取当前线程的视图)\n");
    printf("  3. partition_S(src) (划分源数据)\n");
    printf("  4. partition_D(dst) (划分目标数据)\n");
    printf("  5. copy(tiled_copy, src_part, dst_part) (执行copy)\n");
}

// ============================================================================
// 综合示例: 在GEMM中的实际应用
// ============================================================================
void example_gemm_usage() {
    printf("\n=== Comprehensive Example: GEMM Usage ===\n");
    printf("\n这些概念如何在GEMM kernel中协同工作:\n\n");

    printf("1. Layout & Composition:\n");
    printf("   - 创建swizzled shared memory layout减少bank conflicts\n");
    printf("   - auto smem_layout = composition(Swizzle<3,0,3>{}, base_layout);\n\n");

    printf("2. TiledCopy (Global -> Shared):\n");
    printf("   - auto g2s_copy = make_tiled_copy(...);\n");
    printf("   - auto g2s_thr = g2s_copy.get_slice(threadIdx.x);\n");
    printf("   - auto gA_partition = g2s_thr.partition_S(gA); // global A\n");
    printf("   - auto sA_partition = g2s_thr.partition_D(sA); // shared A\n");
    printf("   - copy(g2s_copy, gA_partition, sA_partition);\n\n");

    printf("3. TiledMMA:\n");
    printf("   - TiledMMA<...> mma;\n");
    printf("   - auto thr_mma = mma.get_slice(threadIdx.x);\n\n");

    printf("4. Partition Fragments:\n");
    printf("   - auto tCrA = mma.partition_fragment_A(sA); // shared A -> register\n");
    printf("   - auto tCrB = mma.partition_fragment_B(sB); // shared B -> register\n");
    printf("   - auto tCrC = mma.partition_fragment_C(sC); // accumulator\n\n");

    printf("5. TiledCopy (Shared -> Register):\n");
    printf("   - auto s2r_copy = make_tiled_copy(...);\n");
    printf("   - auto s2r_thr = s2r_copy.get_slice(threadIdx.x);\n");
    printf("   - auto sA_partition = s2r_thr.partition_S(sA);\n");
    printf("   - auto rA_partition = s2r_thr.partition_D(tCrA);\n");
    printf("   - copy(s2r_copy, sA_partition, rA_partition);\n\n");

    printf("6. MMA Computation:\n");
    printf("   - gemm(mma, tCrA, tCrB, tCrC); // C = A * B\n\n");

    printf("7. local_tile & local_partition:\n");
    printf("   - 将大矩阵分成tiles: local_tile(global_layout, tile_shape)\n");
    printf("   - 将tiles分配给线程: local_partition(tensor, layout, tid)\n\n");

    printf("总结:\n");
    printf("  - Copy操作: 管理数据在不同memory层次间的移动\n");
    printf("  - MMA操作: 管理矩阵乘法计算\n");
    printf("  - Layout操作: 管理数据的组织和映射\n");
    printf("  - partition操作: 管理线程间的工作划分\n");
    printf("  - 这些概念共同构成了高效GEMM kernel的基础\n");
}

int main() {
    // 运行所有示例
    // example_composition_layout();
    // example_swizzle_layout();
    example_local_tile();
    example_local_partition();
    example_tiled_mma();
    example_partition_fragment();
    example_copy_traits_and_atom();
    example_make_tiled_copy();
    example_partition_S();
    example_gemm_usage();

    printf("\n=== All Examples Completed ===\n");
    return 0;
}
