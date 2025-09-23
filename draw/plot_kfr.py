import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========= 配置（把路径改成你的实际清晰设置下的 CSV） =========
csv_paths = {
    "ER":   "../logs/kfr_rhl/ER,cifar10,m200mbs64sbs10blurry500/run0/blurry_kfr.csv",
    # "Ours": "../logs/kfr_rhl/ER_EMA,cifar10,m200mbs64sbs10/run0/task0/task0_kfr.csv",
}
TASK_SIZE = 1000
SMOOTH_WINDOW = 9               # 5/7/9 建议奇数 + center=True
USE_SHADE = True                # True=浅灰底条；False=竖线
SAVE_FIG = True
SAVE_SUMMARY = True

# 线型与透明度（保持：原始=虚线半透明；平滑=粗实线）
RAW_LINESTYLE = (0, (5, 3))
RAW_LINEWIDTH = 1.2
RAW_ALPHA = 0.55

SMOOTH_LINESTYLE = "-"
SMOOTH_LINEWIDTH = 2.4
SMOOTH_ALPHA = 0.95

# ========= 工具函数 =========
def load_and_process_one(path, smooth_window=SMOOTH_WINDOW):
    df = pd.read_csv(path, sep=None, engine="python")
    for col in ["global_step", "KFR_20", "KFR_40"]:
        if col not in df.columns:
            raise ValueError(f"{path} 缺少列：{col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (df.dropna(subset=["global_step"])
            .groupby("global_step", as_index=False)
            .agg({"KFR_20":"mean", "KFR_40":"mean"})
            .sort_values("global_step"))

    def smooth(series, window=smooth_window):
        return series.rolling(window=window, center=True, min_periods=max(1, window//2)).mean()

    df["KFR_20_smooth"]  = smooth(df["KFR_20"])
    df["KFR_40_smooth"] = smooth(df["KFR_40"])
    return df

# 读取多个方法的数据
dfs = {}
for name, path in csv_paths.items():
    if not Path(path).exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    dfs[name] = load_and_process_one(path)

# ========= 统一任务段边界（按所有方法的 step 范围） =========
global_min = min(int(np.floor(dfs[n]["global_step"].min() / TASK_SIZE) * TASK_SIZE) for n in dfs)
global_max = max(int(np.ceil((dfs[n]["global_step"].max() + 1) / TASK_SIZE) * TASK_SIZE) for n in dfs)
edges = np.arange(global_min, global_max + TASK_SIZE, TASK_SIZE)
labels = [f"{int(edges[i])}–{int(edges[i+1]-1)}" for i in range(len(edges)-1)]

# 给每个 df 分配统一的 task_bin
for name in dfs:
    dfs[name]["task_bin"] = pd.cut(dfs[name]["global_step"], bins=edges, right=False, labels=labels)

# ========= 汇总表（分方法、分任务段均值） =========
summary_list = []
for name, df in dfs.items():
    s = (df.groupby("task_bin", observed=True)
           .agg(KFR_50_mean=("KFR_20","mean"),
                KFR_100_mean=("KFR_40","mean"),
                count=("global_step","size"))
           .reset_index())
    s.insert(0, "method", name)
    summary_list.append(s)
summary = pd.concat(summary_list, ignore_index=True)

# ========= 绘图 =========
fig, ax = plt.subplots(figsize=(10, 5))

# 背景：浅灰底条 或 竖线
if USE_SHADE:
    for i in range(len(edges)-1):
        if i % 2 == 1:
            ax.axvspan(edges[i], edges[i+1], facecolor="0.92", alpha=0.3, zorder=0)
else:
    for b in edges[1:]:
        ax.axvline(b, linestyle="--", linewidth=1, alpha=0.6, zorder=1)

# 叠加每个方法的曲线
y_candidates = []
for name, df in dfs.items():
    # 原始（虚线、半透明）
    ax.plot(df["global_step"], df["KFR_20"],
            label=f"{name} KFR_20 (raw)", linestyle=RAW_LINESTYLE,
            linewidth=RAW_LINEWIDTH, alpha=RAW_ALPHA, zorder=2)
    ax.plot(df["global_step"], df["KFR_40"],
            label=f"{name} KFR_40 (raw)", linestyle=RAW_LINESTYLE,
            linewidth=RAW_LINEWIDTH, alpha=RAW_ALPHA, zorder=2)

    # 平滑（实线、较粗）
    ax.plot(df["global_step"], df["KFR_20_smooth"],
            label=f"{name} KFR_20 (smooth-{SMOOTH_WINDOW})", linestyle=SMOOTH_LINESTYLE,
            linewidth=SMOOTH_LINEWIDTH, alpha=SMOOTH_ALPHA, zorder=3)
    ax.plot(df["global_step"], df["KFR_40_smooth"],
            label=f"{name} KFR_100 (smooth-{SMOOTH_WINDOW})", linestyle=SMOOTH_LINESTYLE,
            linewidth=SMOOTH_LINEWIDTH, alpha=SMOOTH_ALPHA, zorder=3)

    y_candidates += [
        df["KFR_20"].max(), df["KFR_40"].max(),
        df["KFR_20_smooth"].max(), df["KFR_40_smooth"].max()
    ]

# 任务段均值标注（每段两行：上 ER、下 Ours；若方法名不同顺序也可自动对应）
y_top = np.nanmax(y_candidates)
y_pos_top = y_top * 0.93 if np.isfinite(y_top) else 0.0
line_gap = (y_top * 0.06) if np.isfinite(y_top) else 0.05  # 两行之间的垂直间隔

# for i in range(len(edges)-1):
#     left, right = edges[i], edges[i+1]
#     mid = (left + right) / 2.0
#
#     # 用 method 的顺序稳定输出（与 legend 顺序一致）
#     row_idx = 0
#     for name in csv_paths.keys():
#         df = dfs[name]
#         seg = df[(df["global_step"] >= left) & (df["global_step"] < right)]
#         if len(seg) == 0:
#             continue
#         m50 = seg["KFR_50"].mean()
#         m100 = seg["KFR_100"].mean()
#         ax.text(mid, y_pos_top - row_idx * line_gap, f"{name}: {m50:.3f}/{m100:.3f}",
#                 ha="center", va="top", fontsize=7.5, alpha=0.9)
#         row_idx += 1

ax.set_xlabel("Step")
ax.set_ylabel("KFR")
# ax.set_title("ER vs. PDFK (ours): KFR_50/100 对比（含任务边界与平滑趋势）")
ax.grid(True, alpha=0.3)

# 让图例更紧凑（可选：loc='upper right', ncol=2）
# ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
#           frameon=False, fontsize=9, borderaxespad=0.0)
# plt.savefig("out.png", dpi=200, bbox_inches="tight")

plt.tight_layout()

# 保存
if SAVE_FIG:
    out_name = "kfr_compare_ER_vs_ours.png"
    plt.savefig(out_name, dpi=300,bbox_inches="tight")
    print(f"已保存图像：{out_name}")
plt.show()

# ========= 导出汇总表 =========
def parse_range(s):
    a, b = s.split("–")
    return int(a), int(b)

summary["task_start"] = summary["task_bin"].astype(str).apply(lambda s: parse_range(s)[0])
summary["task_end"]   = summary["task_bin"].astype(str).apply(lambda s: parse_range(s)[1])
summary = summary[["method", "task_bin", "task_start", "task_end", "count", "KFR_20_mean", "KFR_40_mean"]]

print("\n每个任务段的 KFR 均值（按方法）：")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

if SAVE_SUMMARY:
    out_csv = "kfr_task_summary_compare.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\n已保存：{out_csv}")
