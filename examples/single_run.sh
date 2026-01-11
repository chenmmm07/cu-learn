#!/bin/bash
# 运行单个版本的 GEMM 实现并显示结果

VERSION=$1
M=${2:-2048}
N=${3:-2048}
K=${4:-2048}
WARMUP=${5:-10}
ITERS=${6:-50}
REPEATS=${7:-9}

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version> [M N K] [warmup iters repeats]"
    echo "Available versions: 0, 1, 2, 3"
    echo "Example: $0 0 2048 2048 2048 10 50 9"
    exit 1
fi

case $VERSION in
    0)
        BINARY="build/opt00_naive"
        NAME="Naive"
        ;;
    1)
        BINARY="build/opt01_shared"
        NAME="Shared Memory"
        ;;
    2)
        BINARY="build/opt02_unroll"
        NAME="Loop Unrolling"
        ;;
    3)
        BINARY="build/opt03_regblock"
        NAME="Register Blocking"
        ;;
    *)
        echo "Error: Invalid version. Use 0-3."
        exit 1
esac

if [ ! -f "$BINARY" ]; then
    echo "Error: $BINARY not found. Please build the project first."
    exit 1
fi

echo "=========================================="
echo "Running: $NAME"
echo "Matrix size: M=$M, N=$N, K=$K"
echo "Benchmark config: warmup=$WARMUP, iters=$ITERS, repeats=$REPEATS"
echo "=========================================="
$BINARY $M $N $K $WARMUP $ITERS $REPEATS
