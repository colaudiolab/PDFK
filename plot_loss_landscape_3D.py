# -*- coding: utf-8 -*-
import os
import gc
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------------- 基本配置 -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_root = "./data"
n_tasks = 2
batch_size = 128
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# 方法对比路径（按需修改）
method_paths = {
    "ER": "checkpoints/ER,cifar10,m200mbs64sbs10ER,cifar10,m200mbs64sbs10/0",
    "ER_EMA": "checkpoints/ER_EMA,cifar10,m200mbs64sbs10ER_EMA,cifar10,m200mbs64sbs10/0"
}

# === 按原定范围 ===
alphas = np.linspace(-5.0, 5.0, 21)
betas  = np.linspace(-5.0, 5.0, 21)

# 稳健Z轴裁剪（百分位）。设为 None 则不裁剪
z_clip = (2, 98)

# ----------------------- 模型与数据 -----------------------
from src.models.resnet import ResNet18
def get_model():
    # ret_feat=True 时 forward 可能返回 (logits, feat)
    return ResNet18(nclasses=10, dim_in=512, nf=64, bias=True, ret_feat=True).to(device)

def get_seen_tasks_testloader(task_id, labels_order, n_classes_per_task=10):
    seen_labels = labels_order[: (task_id + 1) * n_classes_per_task]
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
    idx = [i for i, (_, label) in enumerate(testset) if label in seen_labels]
    subset = torch.utils.data.Subset(testset, idx)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# ----------------------- 参数空间中的随机方向 -----------------------
def get_random_directions(model):
    d1, d2 = [], []
    with torch.no_grad():
        for p in model.parameters():
            r1 = torch.randn_like(p)
            r2 = torch.randn_like(p)
            d1.append(r1 / (r1.norm() + 1e-12))
            d2.append(r2 / (r2.norm() + 1e-12))
    return d1, d2

def perturb_model(base_model, d1, d2, alpha, beta):
    new_model = deepcopy(base_model)
    with torch.no_grad():
        for p, d1_, d2_ in zip(new_model.parameters(), d1, d2):
            p.add_(alpha * d1_ + beta * d2_)
    return new_model

# ----------------------- 计算损失曲面 -----------------------
@torch.no_grad()
def compute_loss_surface(model, loader, d1, d2, alphas, betas, device):
    criterion = nn.CrossEntropyLoss()
    surface = np.zeros((len(alphas), len(betas)), dtype=np.float64)

    for i, alpha in enumerate(tqdm(alphas, desc="Alpha", leave=False)):
        for j, beta in enumerate(betas):
            perturbed = perturb_model(model, d1, d2, alpha, beta).to(device)
            perturbed.eval()

            loss_sum, count = 0.0, 0
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = perturbed(x)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                loss = criterion(logits, y)
                bs = x.size(0)
                loss_sum += loss.item() * bs
                count += bs

            surface[i, j] = loss_sum / max(count, 1)

            del perturbed
            torch.cuda.empty_cache()

    return surface

# ----------------------- 可视化（3D叠加） -----------------------
def robust_zlim(arrays, clip_percentiles=(2, 98)):
    if clip_percentiles is None:
        vmin = min(a.min() for a in arrays)
        vmax = max(a.max() for a in arrays)
        return vmin, vmax
    low, high = clip_percentiles
    merged = np.concatenate([a.reshape(-1) for a in arrays], axis=0)
    return np.percentile(merged, low), np.percentile(merged, high)

def visualize_overlay_3d_per_task():
    os.makedirs("figures3d", exist_ok=True)
    labels_order = list(range(10))
    methods = list(method_paths.keys())
    assert len(methods) == 2, "当前代码假定正好对比两种方法。"

    for task_id in range(n_tasks):
        print(f"\n=== Task {task_id+1} / {n_tasks} ===")

        # 载入两种方法的模型
        models = {}
        for m in methods:
            model_path = os.path.join(method_paths[m], f"task_{task_id + 1}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"{m} 的模型不存在: {model_path}")
            model = get_model()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models[m] = model

        # 同一任务共用 1 对随机方向
        d1, d2 = get_random_directions(models[methods[0]])

        # 数据加载器（已见类别）
        loader = get_seen_tasks_testloader(task_id, labels_order)

        # 计算两种方法的曲面
        surfaces = {}
        for m in methods:
            print(f"-> 计算 {m} 的损失曲面 ...")
            surfaces[m] = compute_loss_surface(models[m], loader, d1, d2, alphas, betas, device)

        # Z 轴稳健截断
        zmin, zmax = robust_zlim([surfaces[m] for m in methods], z_clip)

        # 绘制叠加 3D
        X, Y = np.meshgrid(betas, alphas)
        fig = plt.figure(figsize=(7.2, 5.2), dpi=180)
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X, Y, np.clip(surfaces[methods[0]], zmin, zmax),
                        cmap=plt.cm.Blues, linewidth=0, antialiased=True, alpha=0.85, shade=True)
        ax.plot_surface(X, Y, np.clip(surfaces[methods[1]], zmin, zmax),
                        cmap=plt.cm.Reds, linewidth=0, antialiased=True, alpha=0.85, shade=True)

        # 轴与视角
        ax.set_xlabel("X-axis", labelpad=8)
        ax.set_ylabel("Y-axis", labelpad=8)
        ax.set_zlabel("Loss Value", labelpad=6)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(zmin, zmax)
        ax.view_init(elev=25, azim=-60)

        # 网格外观
        ax.xaxis._axinfo["grid"]['linewidth'] = 0.3
        ax.yaxis._axinfo["grid"]['linewidth'] = 0.3
        ax.zaxis._axinfo["grid"]['linewidth'] = 0.3

        legend_handles = [
            Patch(facecolor=plt.cm.Blues(0.6), label=f"{methods[0]} (Blues)"),
            Patch(facecolor=plt.cm.Reds(0.6),  label=f"{methods[1]} (Reds)")
        ]
        # ax.legend(handles=legend_handles, loc="upper right", title="Legend", frameon=True)
        ax.set_title(f"Task {task_id + 1}", pad=12)

        plt.tight_layout()
        png_path = f"figures3d/loss3d_overlay_task{task_id + 1}.png"
        pdf_path = f"figures3d/loss3d_overlay_task{task_id + 1}.pdf"
        plt.savefig(png_path, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        # 清理
        del models, loader, surfaces, d1, d2
        torch.cuda.empty_cache()
        gc.collect()

    print("✅ 全部任务的3D叠加图已保存到 figures3d/ 目录。")

# ----------------------- 主入口 -----------------------
if __name__ == "__main__":
    visualize_overlay_3d_per_task()
