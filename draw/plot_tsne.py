import numpy as np, argparse, pathlib
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

# ===== 图形样式设置 =====
mpl.rcParams["pdf.fonttype"] = mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 10
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["font.family"] = "serif"

# ===== 获取调色板 =====
def get_palette(n):
    cmap = cm.get_cmap('tab20', n)
    return [cmap(i) for i in range(n)]

# ===== 加载原始特征数据 =====
def load_data(tag):
    fdir = pathlib.Path("../tsne")
    X = np.load(fdir / f"features_{tag}.npy")
    y = np.load(fdir / f"labels_{tag}.npy")
    return X, y

# ===== 下采样：每类最多保留 per_class 个样本 =====
def subsample(X, y, per_class=500, seed=0):
    keep_idx = []
    rng = np.random.default_rng(seed)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        keep_idx.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    keep_idx = np.array(keep_idx)
    return X[keep_idx], y[keep_idx]

# ===== 执行 t-SNE 降维 =====
def run_tsne(X, perplexity=100, seed=0):
    X_std = StandardScaler().fit_transform(X)
    tsne = TSNE(
        n_components=2, perplexity=perplexity, init="pca",
        learning_rate="auto", n_iter=1000, metric="euclidean",
        random_state=seed
    )
    return tsne.fit_transform(X_std)

# ===== 绘图主函数 =====
def plot(Z, y, out, tag):
    num_classes = len(np.unique(y))
    palette = get_palette(num_classes)

    fig, ax = plt.subplots(figsize=(3, 2.5))
    scatter_kw = dict(s=24, alpha=0.88, edgecolors="k", linewidths=0.25)

    for c in range(num_classes):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], color=palette[c], label=f"C{c}", **scatter_kw)

    ax.set_xticks(np.linspace(Z[:, 0].min(), Z[:, 0].max(), 5))
    ax.set_yticks(np.linspace(Z[:, 1].min(), Z[:, 1].max(), 5))
    ax.tick_params(labelleft=False, labelbottom=False)

    ax.axis("equal")
    # ax.margins(0.05)


    # 设置主网格与次网格
    ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.4)

    # 图例放右边
    # ax.legend(
    #     loc="center left", bbox_to_anchor=(1.02, 0.5),
    #     frameon=False, fontsize=9,
    #     borderpad=0.2, labelspacing=0.4,
    #     handletextpad=0.4, markerscale=1.0
    # )
    fig.subplots_adjust(right=0.80)
    # ax.legend(
    #     loc='center left',
    #     bbox_to_anchor=(1.01, 0.5),
    #     frameon=False,
    #     fontsize=9,
    #     markerscale=1.2
    # )

    # plt.tight_layout(rect=[0, 0, 0.83, 1])
    # 自动适应紧凑布局
    plt.tight_layout()

    # 去除额外边距
    # ax.set_xlim(Z[:, 0].min(), Z[:, 0].max())
    # ax.set_ylim(Z[:, 1].min(), Z[:, 1].max())

    plt.savefig(out, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved to {out}")

# ===== 主程序入口 =====
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="er_ema", help="features_<tag>.npy")
    ap.add_argument("--perplexity", type=float, default=100)
    ap.add_argument("--out", default="tsne_ours.pdf")
    ap.add_argument("--num-classes", type=int, default=10, help="Number of classes to visualize")#目前只能画10个类
    args = ap.parse_args()

    # ===== 加载数据 =====
    X, y = load_data(args.tag.lower())

    # ===== 筛选前 N 类样本并重映射标签 =====
    selected_classes = np.arange(args.num_classes)
    mask = np.isin(y, selected_classes)
    X = X[mask]
    y = y[mask]

    # 标签重映射为 [0, N-1]
    unique_classes = np.unique(y)
    class_mapping = {v: i for i, v in enumerate(unique_classes)}
    y = np.array([class_mapping[yi] for yi in y])

    # ===== 下采样、t-SNE、绘图 =====
    X, y = subsample(X, y)
    Z = run_tsne(X, perplexity=args.perplexity)
    plot(Z, y, args.out, args.tag)
