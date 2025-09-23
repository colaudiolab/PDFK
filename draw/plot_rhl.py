import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========= 配置（把路径改为你的文件） =========
csv_paths = {
    "ER":            "../logs/kfr_rhl/ER,cifar10,m200mbs64sbs10blurry500/run0/blurry_retain_curve.csv",
    "PDFK (ours)":   "../logs/kfr_rhl/ER_EMA,cifar10,m200mbs64sbs10blurry500/run0/blurry_retain_curve.csv",
}

X_COL = "delta_steps"          # 横轴列名
Y_COL = "retention"            # 纵轴列名
SMOOTH_WINDOW = 3              # 滑动平均窗口（奇数更好；基于“点数”）
SAVE_FIG = True

# 可选：在这些 delta 位置画竖线参考（留空 [] 即不画）
MILESTONES = []  # 例如 [100, 200, 500]

# —— 配色（Okabe–Ito）——
COLORS = {
    "ER": "#0072B2",           # 蓝
    "PDFK (ours)": "#D55E00",  # 橙红
}

# 线型与透明度
RAW_LINESTYLE = (0, (5, 3))    # 虚线
RAW_LINEWIDTH = 1.2
RAW_ALPHA = 0.45               # 半透明

SMOOTH_LINESTYLE = "-"         # 实线
SMOOTH_LINEWIDTH = 2.6
SMOOTH_ALPHA = 0.98

# ========= 读取与预处理 =========
def load_one(path):
    df = pd.read_csv(path, sep=None, engine="python")
    if X_COL not in df.columns or Y_COL not in df.columns:
        raise ValueError(f"{path} 缺少列：{X_COL} / {Y_COL}")
    df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce")
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")
    df = (df.dropna(subset=[X_COL])
            .groupby(X_COL, as_index=False)[Y_COL].mean()
            .sort_values(X_COL))
    # 滑动平均（基于点数窗口）
    df[f"{Y_COL}_smooth"] = df[Y_COL].rolling(
        window=SMOOTH_WINDOW, center=True, min_periods=max(1, SMOOTH_WINDOW//2)
    ).mean()
    return df

dfs = {}
for name, path in csv_paths.items():
    if not Path(path).exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    dfs[name] = load_one(path)

# ========= 绘图 =========
fig, ax = plt.subplots(figsize=(10, 5))
LINE_COLORS = {
    ("ER", "raw"): "#BFDFD2",  # ER raw
    ("ER", "smooth"): "#5BAA99",  # ER smooth（示例色，可改）
    ("PDFK (ours)", "raw"): "#F4B183",  # EREMA raw（示例色，可改）
    ("PDFK (ours)", "smooth"): "#D55E00",  # EREMA smooth（示例色，可改）
}

# 可选：参考竖线
for m in MILESTONES:
    ax.axvline(m, linestyle="--", linewidth=1, alpha=0.4, zorder=1)

# 曲线
for name, df in dfs.items():
    # 颜色映射：按 (方法名, 曲线类型) 精确指定
    # 可以把这些 HEX 改成自己想要的；我先用 #BFDFD2 给 ER raw

    # raw
    ax.plot(df[X_COL], df[Y_COL],
            label=f"{name} (raw)",
            linestyle=RAW_LINESTYLE, linewidth=RAW_LINEWIDTH, alpha=RAW_ALPHA,
            color=LINE_COLORS[(name, "raw")], zorder=2)
    # smooth
    ax.plot(df[X_COL], df[f"{Y_COL}_smooth"],
            label=f"{name} (smooth-{SMOOTH_WINDOW})",
            linestyle=SMOOTH_LINESTYLE, linewidth=SMOOTH_LINEWIDTH, alpha=SMOOTH_ALPHA,
            color=LINE_COLORS[(name, "smooth")], zorder=3)

ax.set_xlabel("Delta steps")
ax.set_ylabel("Retention")
ax.grid(True, alpha=0.3)

# 图例：只保留 smooth，放图外上方以免遮挡
handles, labels = ax.get_legend_handles_labels()
keep_idx = [i for i, lab in enumerate(labels) if "smooth" in lab]
handles = [handles[i] for i in keep_idx]
labels  = [labels[i]  for i in keep_idx]
# ax.legend(handles, labels, loc="right", bbox_to_anchor=(0.5, 1.18),
#           ncol=2, frameon=False, fontsize=9, handlelength=2.5,
#           columnspacing=1.4, borderaxespad=0.0)

plt.tight_layout()

if SAVE_FIG:
    out_name = "retention_compare_ER_vs_ours_blurry.png"
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    print(f"已保存图像：{out_name}")

plt.show()
