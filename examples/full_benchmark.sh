#!/bin/bash
# 运行完整的 benchmark 并生成 CSV 和图表

START=${1:-256}
STOP=${2:-4096}
STEP=${3:-256}
WARMUP=${4:-2}
ITERS=${5:-5}
REPEATS=${6:-4}

echo "=========================================="
echo "GEMM Full Benchmark"
echo "Size range: $START..$STOP step $STEP"
echo "Benchmark config: warmup=$WARMUP, iters=$ITERS, repeats=$REPEATS"
echo "=========================================="

# 确保结果目录存在
mkdir -p results

# 运行 benchmark
./build/gemm_benchmark $START $STOP $STEP $WARMUP $ITERS $REPEATS | tee results/benchmark.csv

# 生成图表
echo ""
echo "=========================================="
echo "Generating plots..."
echo "=========================================="
cd src/plot
python plot.py

echo ""
echo "Done! Results saved to:"
echo "  - results/benchmark.csv"
echo "  - src/plot/time_ms.png"
echo "  - src/plot/gflops.png"
echo "  - src/plot/speedup_vs_cublas.png"
