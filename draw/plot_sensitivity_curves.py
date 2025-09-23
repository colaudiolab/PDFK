# -*- coding: utf-8 -*-
"""
绘制 ΔAcc–ε / ΔLoss–ε（Baseline vs Ours），并计算 AUC 与近零 Slope
- 读入你贴的 sensitivity_agg 格式 CSV（列名与顺序均已适配）
- x 轴为 log(ε)，曲线用 PCHIP 平滑连线（无额外拟合，保形插值）
- 阴影为 95% 置信区间（基于 std/√R），R 默认 12，可在顶部改
- 输出 PNG+PDF 两种格式，适合论文/幻灯片使用

仅需把 CSV_PATH 改成你的文件路径即可。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 必改：你的 CSV 路径 =========
CSV_PATH = "../outputs/sensitivity_final_only/sensitivity_agg.csv"

# ========= 可选：基本绘图与统计参数（无需改也能直接用） =========
OUT_DIR = "./figs_sensitivity"
DATASET_FILTER = "cifar10"     # 过滤 dataset 列（若 CSV 中只有一个数据集可不改）
T_FILTER = "final"             # 只画最终模型
METHODS = ["Baseline", "Ours"] # 曲线顺序
REPEATS = 20                   # 用于从 std 估计 95%CI：CI ≈ 1.96 * std / sqrt(REPEATS)
SMALL_EPS_K = 3                # 用前 k 个最小 ε 点估近零斜率
USE_PCHIP = True               # 若环境无 scipy，会自动退化到线性连线
SAVE_DPI = 300
FONT_SIZE = 12
LINE_WIDTH = 2.0
MARKER_SIZE = 5
ALPHA_FILL = 0.18              # 阴影透明度
# 可选：限制最大 ε（例如只画到 0.006），None 表示不裁剪
EPS_MAX = None  # e.g., EPS_MAX = 0.006

# 颜色（色弱友好，论文常用配色）
COLORS = {
    "Baseline": "#1f77b4",  # 蓝
    "Ours":     "#d62728",  # 红
    "Gap":      "#2ca02c",  # 绿
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_and_filter(csv_path):
    df = pd.read_csv(csv_path)
    # 列清洗：防止意外空格/大小写
    df.columns = [c.strip() for c in df.columns]
    # 过滤
    if "dataset" in df.columns:
        df = df[df["dataset"].astype(str) == str(DATASET_FILTER)]
    if "t" in df.columns:
        df = df[df["t"].astype(str) == str(T_FILTER)]
    # 只保留关键信息
    need = ["method", "eps", "acc0", "loss0",
            "delta_acc_mean", "delta_acc_std",
            "delta_loss_mean", "delta_loss_std"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}")
    # 转数值
    for c in ["eps", "acc0", "loss0", "delta_acc_mean", "delta_acc_std",
              "delta_loss_mean", "delta_loss_std"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 可选裁剪 ε
    if EPS_MAX is not None:
        df = df[df["eps"] <= float(EPS_MAX)]
    return df

def pchip_or_linear(x, y):
    """返回可调用的插值函数：优先 PCHIP，退化到线性"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    try:
        from scipy.interpolate import PchipInterpolator
        return PchipInterpolator(x, y, extrapolate=False)
    except Exception:
        # 线性插值退化（不外推）
        def _lin(xx):
            return np.interp(xx, x, y, left=np.nan, right=np.nan)
        return _lin

def trapz_auc_norm(xs, ys):
    """归一化 AUC: ∫|y|dx / eps_max"""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2:
        return np.nan
    auc = np.trapz(np.abs(ys), xs)
    return float(auc / max(xs.max(), 1e-12))

def near_zero_slope(xs, ys, k=3):
    """近零斜率：取最小的 k 个 ε 点，对 |Δ| 线性回归的斜率"""
    xs = np.asarray(xs, dtype=float)
    ys = np.abs(np.asarray(ys, dtype=float))
    order = np.argsort(xs)
    xs = xs[order][:max(2, k)]
    ys = ys[order][:max(2, k)]
    x_mean = xs.mean()
    y_mean = ys.mean()
    denom = ((xs - x_mean) ** 2).sum()
    if denom < 1e-20:
        return np.nan
    slope = float(((xs - x_mean) * (ys - y_mean)).sum() / denom)
    return slope

def prepare_series(df, method):
    sub = df[df["method"] == method].copy()
    sub = sub.sort_values("eps")
    x = sub["eps"].to_numpy(dtype=float)
    ya = sub["delta_acc_mean"].to_numpy(dtype=float)
    ya_std = sub["delta_acc_std"].to_numpy(dtype=float)
    yl = sub["delta_loss_mean"].to_numpy(dtype=float)
    yl_std = sub["delta_loss_std"].to_numpy(dtype=float)
    # 95%CI（基于 std/√R）
    ci_a = 1.96 * ya_std / max(REPEATS, 1)**0.5
    ci_l = 1.96 * yl_std / max(REPEATS, 1)**0.5
    return x, ya, ci_a, yl, ci_l

def annotate_stats_outside(ax, method_stats):
    """在图像右上角统一展示所有方法的 AUC / Slope，不遮挡曲线"""
    lines = []
    for method, (auc, slope) in method_stats.items():
        lines.append(f"{method}:\nAUC={auc:.3e}\nSlope₀={slope:.2e}")
    full_text = "\n\n".join(lines)
    ax.text(1.02, 1.0, full_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=FONT_SIZE-1,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#666666", alpha=0.95))


def pretty_ax(ax, ylabel):
    ax.set_xscale("log")
    ax.set_xlabel("ε (log scale)", fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    for tick in ax.get_xticklabels()+ax.get_yticklabels():
        tick.set_fontsize(FONT_SIZE-1)

def plot_acc(df):
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=SAVE_DPI)
    xs_dense = None
    method_stats = {}
    for m in METHODS:
        x, y, ci, _, _ = prepare_series(df, m)
        # 插值曲线
        f = pchip_or_linear(x, y) if USE_PCHIP else None
        if xs_dense is None:
            # 在 log 空间等距采样再映回
            xs_dense = np.geomspace(x.min(), x.max(), 400)
        y_smooth = f(xs_dense) if f is not None else None

        # 阴影：95%CI
        ax.fill_between(x, y - ci, y + ci, color=COLORS[m], alpha=ALPHA_FILL, linewidth=0)
        # 原始点
        ax.plot(x, y, marker="o", ms=MARKER_SIZE, lw=0, color=COLORS[m], label=f"{m}")
        # 平滑连线
        if y_smooth is not None:
            ax.plot(xs_dense, y_smooth, lw=LINE_WIDTH, color=COLORS[m])

        # 统计数据暂存
        auc = trapz_auc_norm(x, y)
        slope = near_zero_slope(x, y, k=SMALL_EPS_K)
        method_stats[m] = (auc, slope)

    # annotate_stats_outside(ax, method_stats)
    pretty_ax(ax, ylabel="ΔAcc (pp)")
    ax.legend(frameon=True, fontsize=FONT_SIZE-1, loc="lower left")
    plt.tight_layout()
    ensure_dir(OUT_DIR)
    fig.savefig(os.path.join(OUT_DIR, "delta_acc_vs_eps.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "delta_acc_vs_eps.pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_loss(df):
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=SAVE_DPI)
    xs_dense = None
    method_stats = {}
    for m in METHODS:
        x, _, _, y, ci = prepare_series(df, m)
        f = pchip_or_linear(x, y) if USE_PCHIP else None
        if xs_dense is None:
            xs_dense = np.geomspace(x.min(), x.max(), 400)
        y_smooth = f(xs_dense) if f is not None else None

        ax.fill_between(x, y - ci, y + ci, color=COLORS[m], alpha=ALPHA_FILL, linewidth=0)
        ax.plot(x, y, marker="s", ms=MARKER_SIZE, lw=0, color=COLORS[m], label=f"{m}")
        if y_smooth is not None:
            ax.plot(xs_dense, y_smooth, lw=LINE_WIDTH, color=COLORS[m])

        # 统计数据暂存
        auc = trapz_auc_norm(x, y)
        slope = near_zero_slope(x, y, k=SMALL_EPS_K)
        method_stats[m] = (auc, slope)
    # annotate_stats_outside(ax, method_stats)
    pretty_ax(ax, ylabel="ΔLoss")
    ax.legend(frameon=True, fontsize=FONT_SIZE-1, loc="upper left")
    plt.tight_layout()
    ensure_dir(OUT_DIR)
    fig.savefig(os.path.join(OUT_DIR, "delta_loss_vs_eps.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "delta_loss_vs_eps.pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_gap(df):
    """可选：画 Ours−Baseline 的 ΔAcc 差值（负值表示 Ours 更平缓）"""
    # 对齐两方法的 eps
    dfb = df[df["method"] == "Baseline"].copy().sort_values("eps")
    dfo = df[df["method"] == "Ours"].copy().sort_values("eps")
    eps = np.intersect1d(dfb["eps"].values, dfo["eps"].values)
    if len(eps) < 2:
        return
    dfb = dfb[dfb["eps"].isin(eps)]
    dfo = dfo[dfo["eps"].isin(eps)]
    x = eps.astype(float)
    y_gap = dfo["delta_acc_mean"].to_numpy() - dfb["delta_acc_mean"].to_numpy()
    # 合成 CI（近似独立）：sem_gap^2 = sem_o^2 + sem_b^2
    sem_b = (dfb["delta_acc_std"].to_numpy() / max(REPEATS,1)**0.5)
    sem_o = (dfo["delta_acc_std"].to_numpy() / max(REPEATS,1)**0.5)
    ci_gap = 1.96 * np.sqrt(sem_b**2 + sem_o**2)

    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=SAVE_DPI)
    f = pchip_or_linear(x, y_gap) if USE_PCHIP else None
    xs_dense = np.geomspace(x.min(), x.max(), 400)
    y_smooth = f(xs_dense) if f is not None else None

    ax.axhline(0.0, color="#444444", lw=1, ls="--", alpha=0.6)
    ax.fill_between(x, y_gap - ci_gap, y_gap + ci_gap, color=COLORS["Gap"], alpha=ALPHA_FILL, linewidth=0)
    ax.plot(x, y_gap, marker="D", ms=MARKER_SIZE, lw=0, color=COLORS["Gap"], label="Ours − Baseline")
    if y_smooth is not None:
        ax.plot(xs_dense, y_smooth, lw=LINE_WIDTH, color=COLORS["Gap"])

    pretty_ax(ax, ylabel="ΔAcc Gap (pp)")
    ax.legend(frameon=True, fontsize=FONT_SIZE-1, loc="best")
    plt.tight_layout()
    ensure_dir(OUT_DIR)
    fig.savefig(os.path.join(OUT_DIR, "delta_acc_gap_vs_eps.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "delta_acc_gap_vs_eps.pdf"), bbox_inches="tight")
    plt.close(fig)

def main():
    df = load_and_filter(CSV_PATH)
    print(f"[Info] Loaded rows: {len(df)} (dataset={DATASET_FILTER}, t={T_FILTER})")
    # 计算并打印全表的 AUC / Slope（两方法分别）
    for m in METHODS:
        x, ya, _, yl, _ = prepare_series(df, m)
        auc_a = trapz_auc_norm(x, ya)
        slope_a = near_zero_slope(x, ya, k=SMALL_EPS_K)
        auc_l = trapz_auc_norm(x, yl)
        slope_l = near_zero_slope(x, yl, k=SMALL_EPS_K)
        print(f"[{m}] ΔAcc: AUC={auc_a:.4e}, Slope₀={slope_a:.4e} | ΔLoss: AUC={auc_l:.4e}, Slope₀={slope_l:.4e}")

    plot_acc(df)
    plot_loss(df)
    plot_gap(df)  # 可按需注释掉

if __name__ == "__main__":
    main()
