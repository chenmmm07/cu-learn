import pandas as pd
import matplotlib.pyplot as plt
 
import os
csv_path = os.path.join(os.path.dirname(__file__), "../../results/benchmark.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    # 旧路径兼容
    df = pd.read_csv("results.csv")
 
# 基本检查
required = {"size", "method", "ms", "GFLOPS", "ok"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Columns are: {list(df.columns)}")
 
# 只画 ok==1 的曲线（避免错误结果污染）
df_ok = df[df["ok"] == 1].copy()
 
# pivot 成宽表
ms = df_ok.pivot(index="size", columns="method", values="ms").sort_index()
gf = df_ok.pivot(index="size", columns="method", values="GFLOPS").sort_index()
 
# ===== 在这里选择要绘制的版本 =====
# 从下面的方法列表中选择你想绘制的，注释掉不需要的
selected_methods = [
    "cublasSgemm",
    "v0_naive",
    "v1_shared",
    "v2_unrolled",
    "v3_regblock",
]

# 按照列表顺序筛选列
cols = [m for m in selected_methods if m in ms.columns]
ms = ms[cols]
gf = gf[cols]
 
# --- Time plot ---
plt.figure(figsize=(9,5))
for col in ms.columns:
    plt.plot(ms.index, ms[col], marker="o", linewidth=1.8, label=col)
plt.xlabel("Matrix size (M=N=K)")
plt.ylabel("Median time (ms)")
plt.title("GEMM Benchmark: custom kernels vs cuBLAS (time)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("time_ms.png", dpi=200)
 
# --- GFLOPS plot ---
plt.figure(figsize=(9,5))
for col in gf.columns:
    plt.plot(gf.index, gf[col], marker="o", linewidth=1.8, label=col)
plt.xlabel("Matrix size (M=N=K)")
plt.ylabel("GFLOPS")
plt.title("GEMM Benchmark: custom kernels vs cuBLAS (GFLOPS)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("gflops.png", dpi=200)
 
# --- Speedup vs cuBLAS ---
if "cublasSgemm" in ms.columns:
    speedup = ms["cublasSgemm"].to_frame().join(ms, rsuffix="_x")
    # speedup(method) = cublas_ms / method_ms
    plt.figure(figsize=(9,5))
    for col in ms.columns:
        if col == "cublasSgemm":
            continue
        plt.plot(ms.index, ms["cublasSgemm"] / ms[col], marker="o", linewidth=1.8, label=col)
    plt.axhline(1.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xlabel("Matrix size (M=N=K)")
    plt.ylabel("Speedup vs cuBLAS ( >1 means faster than cuBLAS )")
    plt.title("Speedup relative to cuBLAS")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("speedup_vs_cublas.png", dpi=200)
 
print("Saved: time_ms.png, gflops.png, speedup_vs_cublas.png")
