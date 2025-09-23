# -*- coding: utf-8 -*-
"""
傻瓜式采集脚本（仅最终模型，Baseline vs Ours）
- 固定方向扫 ε（每个 repeat 只采一次方向，所有 ε 复用），曲线更平滑；
- ε 网格加密（默认 16 点，可切 20 点），REPEATS=12，误差带更窄；
- 自动探测预处理（Normalize vs Plain），强制使用 model.logits(x) 取分类分数；
- 大 ε 自适应做 BN 统计轻校准（避免极端扰动下 BN 失配带来的噪声）；
- 仅输出数据：sensitivity_detail.csv、sensitivity_agg.csv。

使用：
1) 仅修改 CKPT_BASELINE 与 CKPT_OURS 为你的最终权重；
2) 运行：python sensitivity_collect_final_only_smooth.py
"""

import os, time, copy, random
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# =========================
# 你只需改这两行
# =========================
CKPT_BASELINE = "checkpoints/ER,cifar10,m200mbs64sbs10ER,cifar10,m200mbs64sbs10/0/task_5.pth"    # ← 改成你的 Baseline 最终权重
CKPT_OURS     = "checkpoints/ER_EMA,cifar10,m200mbs64sbs10ER_EMA,cifar10,m200mbs64sbs10/0/task_5.pth"  # ← 改成你的 Ours 最终权重

# =========================
# 硬编码配置（无需改）
# =========================
OUTPUT_DIR   = "./outputs/sensitivity_final_only"
DATA_ROOT    = "./data"
DATASET      = "cifar10"        # cifar10 / cifar100
BATCH_SIZE   = 256
NUM_WORKERS  = 0
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ε 网格：16 点（log 间距；更平滑）。如需 20 点到 1e-2，改用 EPS_LIST_EXTENDED。
EPS_LIST_16 = [1.0e-4, 1.3e-4, 1.7e-4, 2.2e-4, 2.9e-4, 3.8e-4, 5.0e-4, 6.6e-4,
               8.6e-4, 1.1e-3, 1.5e-3, 2.0e-3, 2.6e-3, 3.4e-3, 4.5e-3, 6.0e-3]
# 可选：20 点（覆盖到 1e-2）
EPS_LIST_EXTENDED = [1.0e-4, 1.3e-4, 1.7e-4, 2.2e-4, 2.9e-4, 3.8e-4, 5.0e-4, 6.6e-4,
                     8.6e-4, 1.1e-3, 1.5e-3, 2.0e-3, 2.6e-3, 3.4e-3, 4.5e-3, 6.0e-3,
                     8.0e-3, 1.0e-2, 1.3e-2, 1.6e-2]  # 如显卡/时间允许

EPS_LIST = EPS_LIST_EXTENDED      # ← 默认 16 点；想更顺滑改为 EPS_LIST_EXTENDED

REPEATS  = 20               # 重复次数（方向数）；越大误差带越窄
MIN_FILTER_NORM = 1e-12
INCLUDE_BIAS    = False
SEED = 0

# 大 ε 时的 BN 轻校准设置
BN_EPS_THRESHOLD = float('inf')    # 当 eps ≥ 6e-3 时触发 BN 校准
BN_CALIBRATE_STEPS_BIG = 20  # 触发时的前向批次数（仅统计刷新，无梯度）

# =========================
# 随机 & 设备
# =========================
def set_global_seed(seed: int = 0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_str: Optional[str] = None) -> torch.device:
    return torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 模型构建与加载（不改你模型）
# =========================
def infer_num_classes_from_ckpt(ckpt_path: str) -> int:
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }
    for key in ["linear.weight", "fc.weight", "classifier.weight", "head.weight"]:
        if key in sd and sd[key].dim() == 2:
            return int(sd[key].shape[0])
    cands = [(k, v.shape[0]) for k,v in sd.items() if torch.is_tensor(v) and v.dim()==2]
    if cands:
        cands.sort(key=lambda x: x[1])
        return int(cands[0][1])
    return 10

def build_model(num_classes: int) -> nn.Module:
    # 优先用你工程内的实现（包含 logits 方法、ret_feat=True 兼容）
    try:
        from src.models.resnet import ResNet18
        try:
            m = ResNet18(nclasses=num_classes, dim_in=512, nf=64, bias=True, ret_feat=True)
            return m
        except Exception:
            # 如果你的项目没有 ResNet18 的该签名，请按需调整；作为兜底回退到 torchvision
            pass
    except Exception:
        pass
    from torchvision import models
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def load_checkpoint_strict(model: nn.Module, ckpt_path: str, device: torch.device):
    sd = torch.load(ckpt_path, map_location=device)
    sd = sd.get("state_dict", sd)
    sd = { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }

    model_keys = list(model.state_dict().keys())
    def find_head_prefix(keys):
        for pref in ["linear", "fc", "classifier", "head"]:
            if f"{pref}.weight" in keys or f"{pref}.bias" in keys:
                return pref
        return None
    model_head = find_head_prefix(model_keys)
    ckpt_head  = find_head_prefix(list(sd.keys()))

    if ckpt_head and model_head and ckpt_head != model_head:
        renamed = {}
        for k,v in sd.items():
            if k.startswith(ckpt_head + "."):
                renamed[k.replace(ckpt_head + ".", model_head + ".", 1)] = v
            else:
                renamed[k] = v
        sd = renamed

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if model_head and (f"{model_head}.weight" in missing or f"{model_head}.bias" in missing):
        raise RuntimeError(f"Classifier head '{model_head}.*' 未成功加载，请检查维度/前缀映射。")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

# =========================
# 评测：强制拿 logits（不改模型）
# =========================
@torch.no_grad()
def safe_logits(model: nn.Module, x, num_classes: int):
    # 优先 model.logits
    if hasattr(model, "logits") and callable(getattr(model, "logits")):
        return model.logits(x)
    out = model(x)
    if isinstance(out, (tuple, list)):
        for z in out:
            if torch.is_tensor(z) and z.dim() >= 2 and z.shape[-1] == num_classes:
                return z
        return out[0] if torch.is_tensor(out[0]) else out
    else:
        z = out
        if torch.is_tensor(z) and z.dim() >= 2 and z.shape[-1] == num_classes:
            return z
        if hasattr(model, "linear") and isinstance(model.linear, nn.Linear):
            return model.linear(z)
        return z

@torch.no_grad()
def evaluate_with_logits(model: nn.Module, loader, device: torch.device, num_classes: int) -> Tuple[float, float]:
    model.eval()
    n_ok = 0; n_all = 0; losses = []
    for x,y in loader:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = safe_logits(model, x, num_classes)
        loss = F.cross_entropy(logits, y, reduction="mean")
        losses.append(loss.item())
        pred = logits.argmax(1)
        n_ok += (pred==y).sum().item(); n_all += y.numel()
    acc = 100.0 * n_ok / max(1,n_all)
    return acc, float(np.mean(losses)) if losses else 0.0

# =========================
# 数据加载（含预处理自动探测）
# =========================
from torchvision import datasets, transforms

def build_eval_loaders(dataset: str, data_root: str, batch_size: int, num_workers: int = 4):
    if dataset.lower() == "cifar10":
        norm = transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
        tf_norm = transforms.Compose([transforms.ToTensor(), norm])
        tf_plain= transforms.Compose([transforms.ToTensor()])
        ds_norm  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_norm)
        ds_plain = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_plain)
        nc = 10
    elif dataset.lower() == "cifar100":
        norm = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
        tf_norm = transforms.Compose([transforms.ToTensor(), norm])
        tf_plain= transforms.Compose([transforms.ToTensor()])
        ds_norm  = datasets.CIFAR100(root=data_root, train=False, download=True, transform=tf_norm)
        ds_plain = datasets.CIFAR100(root=data_root, train=False, download=True, transform=tf_plain)
        nc = 100
    else:
        raise NotImplementedError("仅示例支持 cifar10/cifar100。")

    ld_norm  = torch.utils.data.DataLoader(ds_norm,  batch_size=batch_size, shuffle=False,
                                           num_workers=num_workers, pin_memory=True)
    ld_plain = torch.utils.data.DataLoader(ds_plain, batch_size=batch_size, shuffle=False,
                                           num_workers=num_workers, pin_memory=True)
    return (ld_norm, ld_plain, nc)

@torch.no_grad()
def auto_pick_loader(model, ld_norm, ld_plain, device, num_classes):
    acc_n, _ = evaluate_with_logits(model, ld_norm,  device, num_classes)
    acc_p, _ = evaluate_with_logits(model, ld_plain, device, num_classes)
    if acc_n >= acc_p:
        print(f"[Preproc] 选择 Normalize（acc≈{acc_n:.2f}% ≥ {acc_p:.2f}%）")
        return ld_norm
    else:
        print(f"[Preproc] 选择 Plain（acc≈{acc_p:.2f}% > {acc_n:.2f}%）")
        return ld_plain

# =========================
# 扰动器（filter-wise 归一化）
# =========================
class FilterwisePerturber:
    def __init__(self, included=(nn.Conv2d, nn.Linear), min_norm: float = 1e-12, include_bias: bool = False):
        self.included = included; self.min_norm = float(min_norm); self.include_bias = bool(include_bias)

    def _unit_noise_like(self, shape, rng: torch.Generator, device):
        n = torch.randn(shape, generator=rng, device=device)
        n_flat = n.view(n.size(0), -1)
        n_unit = (n_flat / torch.norm(n_flat, p=2, dim=1, keepdim=True).clamp_min(1e-12)).view_as(n)
        return n_unit

    def _filter_norms(self, w: torch.Tensor):
        w_flat = w.view(w.size(0), -1)
        return torch.norm(w_flat, p=2, dim=1).clamp_min(self.min_norm)

    def make_direction(self, model: nn.Module, base_seed: int, tag: str, device: torch.device):
        """
        生成与 'tag' 绑定的方向；同一 tag 重复生成可得到相同方向。
        注意：为实现“固定方向扫 ε”，我们在循环中用相同 tag（不含 eps）。
        """
        dir_dict = {}; layer_idx = 0
        for m in model.modules():
            if isinstance(m, self.included) and hasattr(m, "weight") and m.weight is not None:
                w = m.weight
                rng = torch.Generator(device=device)
                rng.manual_seed(_hash_to_seed(base_seed, layer_idx, tuple(w.shape), tag))
                dir_dict[w] = self._unit_noise_like(w.shape, rng, device=w.device)
                if self.include_bias and hasattr(m,"bias") and m.bias is not None:
                    b = m.bias
                    rng_b = torch.Generator(device=device)
                    rng_b.manual_seed(_hash_to_seed(base_seed, layer_idx, ("bias", b.numel()), tag))
                    nb = torch.randn_like(b, generator=rng_b)
                    dir_dict[b] = nb / nb.norm(p=2).clamp_min(1e-12)
                layer_idx += 1
        return dir_dict

    @torch.no_grad()
    def apply_inplace(self, model: nn.Module, direction: Dict[torch.Tensor, torch.Tensor], eps: float):
        for m in model.modules():
            if isinstance(m, self.included) and hasattr(m,"weight") and m.weight is not None:
                w = m.weight
                if w not in direction: continue
                n_unit = direction[w]
                norms = self._filter_norms(w)  # [out]
                scale = norms.view(w.size(0), *([1]*(w.dim()-1)))
                w.add_(eps * scale * n_unit)
                if self.include_bias and hasattr(m,"bias") and m.bias is not None and m.bias in direction:
                    b = m.bias
                    b.add_(eps * b.norm(p=2).clamp_min(self.min_norm) * direction[b])

def _hash_to_seed(base_seed: int, layer_idx: int, shape_tuple, tag: str) -> int:
    s = f"{base_seed}|{layer_idx}|{shape_tuple}|{tag}"
    h = hash(s)
    return (base_seed ^ (h & 0xFFFFFFFF)) & 0xFFFFFFFF

# =========================
# BN 小批校准（自适应）
# =========================
@torch.no_grad()
def bn_recalibration(model: nn.Module, loader, steps: int, device: torch.device):
    if steps <= 0: return
    model.train()
    n = 0
    for x,_ in loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        n += 1
        if n >= steps: break
    model.eval()

def bn_steps_for_eps(eps: float) -> int:
    return BN_CALIBRATE_STEPS_BIG if eps >= BN_EPS_THRESHOLD else 0

# =========================
# 主流程
# =========================
def main():
    set_global_seed(SEED)
    device = get_device(DEVICE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 推断类别数（从 Baseline ckpt）
    num_classes = infer_num_classes_from_ckpt(CKPT_BASELINE)
    print(f"[Info] 推断 num_classes = {num_classes}")

    # 构建 Baseline 并选择预处理
    model_probe = build_model(num_classes=num_classes).to(device)
    load_checkpoint_strict(model_probe, CKPT_BASELINE, device=device)

    from torchvision import datasets, transforms
    ld_norm, ld_plain, nc_dataset = build_eval_loaders(DATASET, DATA_ROOT, BATCH_SIZE, NUM_WORKERS)
    if nc_dataset != num_classes:
        print(f"[WARN] 数据集类别数({nc_dataset})与模型({num_classes})不一致，评测可能失真。")
    eval_loader = auto_pick_loader(model_probe, ld_norm, ld_plain, device, num_classes)

    # 耗时估计（仅用于打印）
    t0 = time.time()
    _ = evaluate_with_logits(model_probe, eval_loader, device, num_classes)
    Teval = time.time() - t0
    perturber = FilterwisePerturber(min_norm=MIN_FILTER_NORM, include_bias=INCLUDE_BIAS)
    t1 = time.time()
    mp = copy.deepcopy(model_probe).to(device); mp.eval()
    # 用固定 tag（不含 eps）以模拟固定方向
    direction_probe = perturber.make_direction(mp, base_seed=SEED, tag=f"probe|rep=0", device=device)
    perturber.apply_inplace(mp, direction_probe, eps=EPS_LIST[0])
    Tover = time.time() - t1
    print(f"[Probe] Teval≈{Teval:.3f}s, Tover≈{Tover:.3f}s; N_eps={len(EPS_LIST)}, R={REPEATS}")

    # 真正评测
    methods = {
        "Baseline": CKPT_BASELINE,
        "Ours":     CKPT_OURS,
    }

    detail_rows, agg_rows = [], []
    for method_name, ckpt in methods.items():
        print(f"\n=== {method_name} ===")
        model = build_model(num_classes=num_classes).to(device)
        load_checkpoint_strict(model, ckpt, device=device)

        acc0, loss0 = evaluate_with_logits(model, eval_loader, device, num_classes)
        print(f"[Base] acc0={acc0:.2f}%, loss0={loss0:.4f}")

        # 固定方向扫 ε：每个 repeat 只生成一次方向，所有 ε 复用该方向
        for r in range(REPEATS):
            tag = f"final|rep={r}"  # 不含 eps
            # 对于不同 eps，我们会为每个 mp 重新生成“相同 tag”的 direction（与模型实例绑定，不保存引用）
            for eps in EPS_LIST:
                mp = copy.deepcopy(model).to(device); mp.eval()
                direction = perturber.make_direction(mp, base_seed=SEED, tag=tag, device=device)
                perturber.apply_inplace(mp, direction, eps=eps)

                # 自适应 BN 轻校准（大 ε 才做）
                bn_recalibration(mp, eval_loader, steps=bn_steps_for_eps(eps), device=device)

                acc_eps, loss_eps = evaluate_with_logits(mp, eval_loader, device, num_classes)
                da, dl = (acc_eps - acc0), (loss_eps - loss0)

                detail_rows.append({
                    "dataset": DATASET, "method": method_name, "t": "final",
                    "ckpt": ckpt, "eps": eps, "repeat": r,
                    "acc0": acc0, "loss0": loss0,
                    "acc_eps": acc_eps, "loss_eps": loss_eps,
                    "delta_acc": da, "delta_loss": dl
                })

        # 聚合（对同一 eps 汇总 R 次）
        df_tmp = pd.DataFrame([row for row in detail_rows if row["method"]==method_name])
        for eps in EPS_LIST:
            sub = df_tmp[df_tmp["eps"]==eps]
            da_m, da_s = float(sub["delta_acc"].mean()), float(sub["delta_acc"].std(ddof=1) if len(sub)>1 else 0.0)
            dl_m, dl_s = float(sub["delta_loss"].mean()), float(sub["delta_loss"].std(ddof=1) if len(sub)>1 else 0.0)
            agg_rows.append({
                "dataset": DATASET, "method": method_name, "t": "final", "eps": eps,
                "acc0": acc0, "loss0": loss0,
                "delta_acc_mean": da_m, "delta_acc_std": da_s,
                "delta_loss_mean": dl_m, "delta_loss_std": dl_s
            })

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_detail = pd.DataFrame(detail_rows)
    df_agg    = pd.DataFrame(agg_rows)
    p_detail = os.path.join(OUTPUT_DIR, "sensitivity_detail.csv")
    p_agg    = os.path.join(OUTPUT_DIR, "sensitivity_agg.csv")
    df_detail.to_csv(p_detail, index=False)
    df_agg.to_csv(p_agg, index=False)
    print(f"\n[Saved] {p_detail}")
    print(f"[Saved] {p_agg}")
    print("[Done] 固定方向 + 加密 ε + 提高 repeat 的数据采集完成。可直接据此画更平滑的 ΔAcc–ε / ΔLoss–ε。")

if __name__ == "__main__":
    main()
