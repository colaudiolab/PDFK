import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from torchvision import datasets, transforms
import pandas as pd
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_root = "./data"
n_tasks = 5
batch_size = 128

# 方法对比路径
method_paths = {
    "ER": "checkpoints/ER,cifar10,m200mbs64sbs10ER,cifar10,m200mbs64sbs10/0",
    "ER_EMA": "checkpoints/ER_EMA,cifar10,m200mbs64sbs10ER_EMA,cifar10,m200mbs64sbs10/0"
    # "ER-P2": "checkpoints/ER_EMA,cifar10,m200mbs64sbs10ER_P2,cifar10,m200mbs64sbs10/0"
}

from src.models.resnet import ResNet18
def get_model():
    return ResNet18(nclasses=10, dim_in=512, nf=64, bias=True, ret_feat=True).to(device)


def find_global_loss_range():
    alphas = np.linspace(-5, 5, 21)
    betas = np.linspace(-5, 5, 21)
    labels_order = list(range(10))
    global_min, global_max = float("inf"), float("-inf")

    for method_name, model_root in method_paths.items():
        for task_id in range(n_tasks):
            model_path = os.path.join(model_root, f"task_{task_id + 1}.pth")
            model = get_model()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            loader = get_seen_tasks_testloader(task_id, labels_order)
            d1, d2 = get_random_directions(model)
            surface = compute_loss_surface(model, loader, d1, d2, alphas, betas)

            global_min = min(global_min, surface.min())
            global_max = max(global_max, surface.max())

            del model, loader, surface
            torch.cuda.empty_cache()
            gc.collect()

    return global_min, global_max

def get_seen_tasks_testloader(task_id, labels_order, n_classes_per_task=10):
    seen_labels = labels_order[: (task_id + 1) * n_classes_per_task]
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
    idx = [i for i, (_, label) in enumerate(testset) if label in seen_labels]
    subset = torch.utils.data.Subset(testset, idx)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)

def get_random_directions(model):
    d1, d2 = [], []
    for p in model.parameters():
        r1 = torch.randn_like(p)
        r2 = torch.randn_like(p)
        d1.append(r1 / r1.norm())
        d2.append(r2 / r2.norm())
    return d1, d2

def perturb_model(model, d1, d2, alpha, beta):
    new_model = deepcopy(model)
    with torch.no_grad():
        for p, d1_, d2_ in zip(new_model.parameters(), d1, d2):
            p.add_(alpha * d1_ + beta * d2_)
    return new_model

def compute_loss_surface(model, loader, d1, d2, alphas, betas):
    criterion = nn.CrossEntropyLoss()
    surface = np.zeros((len(alphas), len(betas)))
    for i, alpha in enumerate(tqdm(alphas, desc="Alpha")):
        for j, beta in enumerate(betas):
            perturbed = perturb_model(model, d1, d2, alpha, beta)
            perturbed.eval()
            loss_sum, count = 0.0, 0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    logits = perturbed(x)
                    loss = criterion(logits, y)
                    loss_sum += loss.item() * x.size(0)
                    count += x.size(0)
            surface[i][j] = loss_sum / count
    return surface

def compute_flatness_metric(surface):
    """定义平坦度指标：最大值 - 最小值（或方差）"""
    return {
        "range": np.max(surface) - np.min(surface),
        "std": np.std(surface),
        "mean": np.mean(surface)
    }

def visualize_all_methods():
    alphas = np.linspace(-5, 5, 21)
    betas = np.linspace(-5, 5, 21)
    labels_order = list(range(10))
    results = []

    # 手动设定每个 task 的 (vmin, vmax)
    task_loss_ranges = [
        (0, 60),  # Task 1
        (0, 50),  # Task 2
        (0, 20),  # Task 3
        (0, 25),  # Task 4
        (0, 40),  # Task 5
    ]

    for method_name, model_root in method_paths.items():
        fig, axes = plt.subplots(1, n_tasks, figsize=(n_tasks * 3.2, 3.2))
        for task_id in range(n_tasks):
            vmin, vmax = task_loss_ranges[task_id]

            model_path = os.path.join(model_root, f"task_{task_id + 1}.pth")
            model = get_model()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            loader = get_seen_tasks_testloader(task_id, labels_order)
            d1, d2 = get_random_directions(model)
            surface = compute_loss_surface(model, loader, d1, d2, alphas, betas)

            X, Y = np.meshgrid(betas, alphas)
            cp = axes[task_id].contourf(
                X, Y, surface,
                levels=np.linspace(vmin, vmax, 70),  # 增加为 100 等级
                cmap='cividis',
                vmin=vmin,
                vmax=vmax,
                extend='neither'  # 关闭两端尖头
            )

            axes[task_id].set_title(f"Task {task_id + 1}")
            plt.colorbar(cp, ax=axes[task_id])

            metrics = compute_flatness_metric(surface)
            metrics.update({
                "method": method_name,
                "task": task_id + 1
            })
            results.append(metrics)

            del model, loader, surface
            torch.cuda.empty_cache()
            gc.collect()

        plt.tight_layout()
        os.makedirs("../figures3", exist_ok=True)
        plt.savefig(f"figures3/loss_landscape_{method_name}.pdf")
        plt.close()

    df = pd.DataFrame(results)
    os.makedirs("../results", exist_ok=True)
    df.to_csv("results/oursflatness_metrics.csv", index=False)
    print("✅ Loss surface & metrics saved.")


if __name__ == "__main__":
    visualize_all_methods()
