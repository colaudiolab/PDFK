# -*- coding: utf-8 -*-
"""
Use the installed `loss_landscapes` library to draw loss landscapes
for CIFAR-10 5 tasks, Baseline vs Ours (overlay, single view).
"""
import os, sys, gc, random, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch

# ---- ensure we can import your src/ ----
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ====== user config ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

DATASET_ROOT = "./data"
NUM_CLASSES = 10
N_TASKS = 5
N_CLASSES_PER_TASK = 2               # CIFAR-10 -> 5 tasks
SAMPLES_PER_TASK = 512               # small fixed batch for speed

GRID_SIZE = 21                       # steps for the lib; 21/31/41 ...
DISTANCE = 0.3                       # +/-0.3 (roughly与之前一致)
VIEW_ELEV, VIEW_AZIM = 28, -55

method_paths = {
    "Baseline": "checkpoints/ER,cifar10,m200mbs64sbs10ER,cifar10,m200mbs64sbs10/0",
    "Ours":     "checkpoints/ER_EMA,cifar10,m200mbs64sbs10ER_EMA,cifar10,m200mbs64sbs10/0"
}

SAVE_DIR = "figures3d_llib"
os.makedirs(SAVE_DIR, exist_ok=True)

# ====== import the library ======
from loss_landscapes import loss_landscapes

# ====== your model ======
from src.models.resnet import ResNet18
def build_model():
    # 与训练时保持一致；ret_feat=True 时 forward 可能返回 (feat, logits)
    return ResNet18(nclasses=NUM_CLASSES, dim_in=512, nf=64, bias=True, ret_feat=True).to(device).eval()

# ----- helpers -----
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def forward_logits(model, x):
    out = model(x)
    # 兼容 ret_feat=True 等多返回值
    if isinstance(out, (tuple, list)):
        for t in out:
            if isinstance(t, torch.Tensor) and t.ndim == 2 and t.size(1) == NUM_CLASSES:
                return t
        for t in out:
            if isinstance(t, torch.Tensor): return t
        raise RuntimeError("No logits tensor found.")
    if isinstance(out, dict):
        for v in out.values():
            if isinstance(v, torch.Tensor) and v.ndim == 2 and v.size(1) == NUM_CLASSES:
                return v
        for v in out.values():
            if isinstance(v, torch.Tensor): return v
        raise RuntimeError("No logits tensor found.")
    return out

class SoftmaxWrapper(nn.Module):
    """
    Wrap model->softmax for MSE(onehot) fallback when the lib doesn't
    accept a custom loss function.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        logits = forward_logits(self.model, x)
        return F.softmax(logits, dim=1)  # [N, C]

def get_fixed_batch_for_task(task_id, labels_order):
    # 按你的要求：不打开 Normalize，仅 ToTensor()
    tfm = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root=DATASET_ROOT, train=False, download=True, transform=tfm)
    seen = set(labels_order[: (task_id + 1) * N_CLASSES_PER_TASK])
    idx = [i for i, (_, y) in enumerate(testset) if y in seen]
    set_seed(2025 + task_id)
    if len(idx) > SAMPLES_PER_TASK:
        idx = random.sample(idx, SAMPLES_PER_TASK)
    xs, ys = [], []
    for i in idx:
        x, y = testset[i]; xs.append(x); ys.append(y)
    X = torch.stack(xs, dim=0).to(device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return X, y

def to_onehot(y, num_classes=NUM_CLASSES):
    return F.one_hot(y, num_classes=num_classes).float()

def run_landscape_with_lib(model, X, y):
    """
    使用 loss_landscapes 库计算二维切片。
    - 优先尝试 CrossEntropy（若库支持 loss_fn 参数）
    - 否则使用 Softmax + onehot + MSE 的退路
    返回：loss_grid (numpy array, shape [steps, steps])
    """
    steps = GRID_SIZE
    distance = DISTANCE

    # 需要传一个 optimizer，但不会真正训练；SGD 即可
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0, momentum=0.0)

    # 方案 A：库支持 loss_fn
    try:
        loss_fn = nn.CrossEntropyLoss()
        # 某些库版本要求 y 为 Long；我们这里已经是 Long（类别）
        Z = loss_landscapes(model, optimizer, X, y, distance=distance, steps=steps, loss_fn=loss_fn)
        if isinstance(Z, torch.Tensor):
            Z = Z.detach().cpu().numpy()
        return np.array(Z)
    except TypeError:
        # 方案 B：退回 MSE(onehot, softmax)
        y_oh = to_onehot(y)                          # [N, C]
        wrap = SoftmaxWrapper(model)                  # 输出 [N, C]
        optimizer = torch.optim.SGD(wrap.parameters(), lr=0.0)
        Z = loss_landscapes(wrap, optimizer, X, y_oh, distance=distance, steps=steps)
        if isinstance(Z, torch.Tensor):
            Z = Z.detach().cpu().numpy()
        return np.array(Z)

def plot_overlay(Xgrid, Ygrid, Z_base, Z_ours, title, path, elev=VIEW_ELEV, azim=VIEW_AZIM):
    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(111, projection='3d')
    zmin, zmax = float(min(Z_base.min(), Z_ours.min())), float(max(Z_base.max(), Z_ours.max()))
    ax.set_zlim(zmin, zmax)
    ax.plot_surface(Xgrid, Ygrid, Z_base, cmap=cm.Blues, alpha=0.6, linewidth=0, antialiased=True)
    ax.plot_surface(Xgrid, Ygrid, Z_ours,  cmap=cm.Reds,  alpha=0.6, linewidth=0, antialiased=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X (parameter perturbation)")
    ax.set_ylabel("Y (parameter perturbation)")
    ax.set_zlabel("Loss value")
    ax.legend([
        Patch(facecolor=cm.Blues(0.6), edgecolor='none', label='Baseline (Blues)'),
        Patch(facecolor=cm.Reds(0.6),  edgecolor='none', label='Ours (Reds)'),
    ], loc='upper left', frameon=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def run_one_task(task_id, labels_order, seed_base=2025):
    # 固定一个小批次
    X, y = get_fixed_batch_for_task(task_id, labels_order)

    # 为确保 Baseline/Ours 共享同一组随机方向：对同一 task 固定相同种子
    seed_for_plane = seed_base + task_id * 1000

    Zs = {}
    for name, root in method_paths.items():
        ckpt = os.path.join(root, f"task_{task_id+1}.pth")
        model = build_model()
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state); model.eval()

        # 关键：同一 task、两模型共享方向 —— 依赖库内部用随机数采样方向
        set_seed(seed_for_plane)

        Z = run_landscape_with_lib(model, X, y)  # 由库完成平面采样与计算
        Zs[name] = Z

        del model, state; torch.cuda.empty_cache(); gc.collect()

    # 网格坐标（仅用于绘制轴刻度）
    grid = np.linspace(-DISTANCE, DISTANCE, GRID_SIZE)
    Xg, Yg = np.meshgrid(grid, grid)

    out_path = os.path.join(SAVE_DIR, f"loss3d_task{task_id+1}_overlay.png")
    title = f"CIFAR-10 Task {task_id+1} (batch={X.size(0)}, grid={GRID_SIZE}x{GRID_SIZE})"
    plot_overlay(Xg, Yg, Zs["Baseline"], Zs["Ours"], title, out_path)
    print(f"[Task {task_id+1}] Saved: {out_path}")

def main():
    labels_order = list(range(NUM_CLASSES))
    assert "Baseline" in method_paths and "Ours" in method_paths and len(method_paths) == 2
    for t in range(N_TASKS):
        run_one_task(t, labels_order)
    print("🎯 All tasks done.")

if __name__ == "__main__":
    main()
