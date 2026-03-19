# cu-learn: CUDA GEMM Optimization Journey

记录自己从零开始学习 CUDA 矩阵乘法优化，逐步实现接近 cuBLAS 的性能。

## 🎯 项目简介

本项目通过实际代码演示如何从最基本的矩阵乘法实现开始，逐步应用各种 CUDA 优化技术，最终实现高性能 GEMM kernel。

### 性能提升
待补充

## 📚 优化步骤

1. **[优化 0](src/optimizations/00_naive/)**: Naive GEMM - 理解基础实现和性能瓶颈
2. **[优化 1](src/optimizations/01_shared_memory/)**: Shared Memory - 优化全局内存访问
3. **[优化 2](src/optimizations/02_register_blocking/)**: Register Blocking - 提升算术强度

## 📖 示例代码

`src/demo/` 目录包含工具和接口的使用示例：

- **[cute_test.cu](src/demo/cute_test.cu)**: CuTe DSL 常见函数示例 (配套文档: [cute_notes.md](docs/cute_notes.md))

运行示例：
```bash
./build/demo_cute_test
```

## 🚀 快速开始

### 环境要求

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++14 编译器
- Python 3.7+ (用于绘图)

### 安装依赖

```bash
# Python 依赖
pip install -r requirements.txt
```

### 编译项目

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 运行示例

#### 单独运行某个优化版本

```bash
# 基本用法
./build/opt00 <M> <N> <K> [warmup] [iters] [repeats]

# 参数说明:
#   M, N, K: 矩阵维度 (必需)
#   warmup:  预热次数，不计入统计 (默认: 2)
#   iters:   每次测量执行的kernel次数 (默认: 5)
#   repeats:  重复测量次数，取中位数 (默认: 4)
#
# 注意: 总kernel调用次数 = repeats × (warmup + iters)
# 默认配置下: 4 × (2 + 5) = 28 次调用
# 运行时间 ≈ 总调用次数 × 单次kernel时间
```

#### 各版本运行示例

```bash
# 优化 0: Naive
./build/opt00 2048 2048 2048

# 优化 1: Shared Memory
./build/opt01 2048 2048 2048

# 优化 2: Register Blocking
./build/opt02 2048 2048 2048
```

#### 运行完整 benchmark
目前benchmark是包含所有版本的，可以通过参数指定运行哪个版本

```bash
# 运行完整 benchmark (size: 256..4096, step: 256)
./build/gemm_benchmark 256 4096 256

# 自定义参数
./build/gemm_benchmark <start> <stop> <step> [warmup] [iters] [repeats]

# 示例
./build/gemm_benchmark 512 2048 256

```

#### 绘制性能曲线

```bash
# 生成 results.csv 后
cd src/plot
python plot.py

# 默认会生成:
# - time_ms.png: 时间对比图
# - gflops.png: GFLOPS 对比图
# - percent_of_cublas.png: 相对 cuBLAS 的性能百分比

# 如需只绘制特定版本，编辑 plot.py 中的 selected_methods
```
或使用便捷脚本运行完整 benchmark 并生成图表
```bash
./examples/full_benchmark.sh 512 2048 256
```


## 📊 性能分析工具

### Nsight Compute

```bash
# 分析某个 kernel
ncu --set full -o report.ncu-rep ./build/opt02 2048 2048 2048
# 可视化report文件
ncu-ui report.ncu-rep
```

### Nsight Systems

```bash
# 分析系统级性能
nsys profile --output=report.nsys-rep ./build/gemm_benchmark 256 2048 256
```


## 📖 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight Compute CLI Guide](https://docs.nvidia.com/nsight-compute/)
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！


**Happy Learning!** 🚀
