# metrics_kfr_rhl.py
import math, os, csv
from collections import deque, defaultdict
from typing import List, Dict, Optional, Callable
import numpy as np
import torch
from torch.utils.data import DataLoader
from contextlib import nullcontext  # 新增，可让 AMP/NO-AMP 选择更干净


@torch.no_grad()
def evaluate_correct_mask(model: torch.nn.Module,
                          loader: DataLoader,
                          device: torch.device,
                          predict_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                          preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                          use_autocast: bool = True) -> np.ndarray:
    """
    返回布尔向量：每个样本是否预测正确（顺序=loader顺序）。
    predict_fn: 输入张量 -> logits；若为 None，将尝试 model.logits 或 model(x)。
    preprocess_fn: 评测前的图像变换函数（如 learner.transform_test），可为 None。
    """
    was_training = model.training
    model.eval()
    masks = []

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_autocast and torch.cuda.is_available()) else nullcontext()

    with amp_ctx:
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[:2]
            else:
                x, y = batch["x"], batch["y"]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if preprocess_fn is not None:
                x = preprocess_fn(x)

            if predict_fn is not None:
                logits = predict_fn(x)
            elif hasattr(model, "logits"):
                logits = model.logits(x)
            else:
                logits = model(x)

            pred = logits.argmax(dim=1)
            masks.append((pred == y).cpu().numpy().astype(np.bool_))

    if was_training: model.train()
    return np.concatenate(masks, axis=0)


def attach_eval_loader(dataset, batch_size=256, num_workers=4):
    # 评测必须固定顺序
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


class KFRRHLMeter:
    def __init__(self, K_list_updates: List[int], eval_interval: int,
                 dataset_size: int, save_dir: str, run_name: str = "default",
                 retain_max_frames: int = 50):   # <<< 新增参数：保留曲线窗口（帧数），默认 50 帧

        assert eval_interval >= 1
        self.eval_interval = eval_interval
        self.K_list = sorted(K_list_updates)
        self.offset_by_K = {K: max(1, math.ceil(K / eval_interval)) for K in self.K_list}
        self.max_offset = max(self.offset_by_K.values()) if self.offset_by_K else 1

        # --- KFR 的短历史（只需覆盖到最大的 K 所需帧数） ---
        self.history = deque(maxlen=self.max_offset + 1)
        self.history_steps = deque(maxlen=self.max_offset + 1)

        # --- RHL 的长历史（与 KFR 解耦，能看到更远的步数） ---
        self.retain_history = deque(maxlen=retain_max_frames)   # <<< 新增
        self.retain_steps = deque(maxlen=retain_max_frames)     # <<< 新增
        self.retain_max_frames = retain_max_frames              # <<< 新增

        self.dataset_size = dataset_size

        self.kfr_series: Dict[int, list] = {K: [] for K in self.K_list}
        self.kfr_sum = {K: 0.0 for K in self.K_list}
        self.kfr_cnt = {K: 0 for K in self.K_list}

        self.retain_num = defaultdict(int)
        self.retain_den = defaultdict(int)

        # --- 自增评测编号，避免 eval_index 被短 deque 卡死 ---
        self.eval_counter = -1                                   # <<< 新增

        os.makedirs(save_dir, exist_ok=True)
        self.kfr_csv = os.path.join(save_dir, f"{run_name}_kfr.csv")
        self.retain_csv = os.path.join(save_dir, f"{run_name}_retain_curve.csv")
        with open(self.kfr_csv, "w", newline="") as f:
            csv.writer(f).writerow(["eval_index", "global_step"] + [f"KFR_{K}" for K in self.K_list])

    def update(self, correct_mask: np.ndarray, global_step: int):
        assert correct_mask.ndim == 1 and correct_mask.size == self.dataset_size
        cmask = correct_mask.astype(np.bool_)

        # -- 推进编号 --
        self.eval_counter += 1
        eval_index = self.eval_counter

        # -- KFR 的短历史 --
        self.history.append(cmask)
        self.history_steps.append(global_step)

        # -- RHL 的长历史 --
        self.retain_history.append(cmask)  # <<< 新增
        self.retain_steps.append(global_step)  # <<< 新增

        # 1) KFR
        row = [eval_index, global_step]
        for K in self.K_list:
            off = self.offset_by_K[K]
            if len(self.history) >= off + 1:
                prev = self.history[-(off + 1)]
                now = self.history[-1]
                denom = prev.sum()
                if denom > 0:
                    kfr = float((prev & (~now)).sum()) / float(denom)
                    self.kfr_sum[K] += kfr
                    self.kfr_cnt[K] += 1
                else:
                    kfr = float("nan")
            else:
                kfr = float("nan")
            self.kfr_series[K].append({"eval_index": eval_index, "global_step": global_step, "value": kfr})
            row.append(kfr)
        with open(self.kfr_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # 2) 保留曲线累计：用“长历史”的每个起点与当前帧对齐
        for i, base in enumerate(self.retain_history):  # <<< 改为 retain_history
            delta_frames = len(self.retain_history) - 1 - i
            if delta_frames <= 0:
                continue
            base_size = base.sum()
            if base_size == 0:
                continue
            now = self.retain_history[-1]
            retain = (base & now).sum()
            self.retain_num[delta_frames] += int(retain)
            self.retain_den[delta_frames] += int(base_size)

    def summary(self, rhl_threshold: float = 0.5, return_lower_bound: bool = True) -> Dict:
        # 平均 KFR 同原逻辑
        kfr_mean = {K: (self.kfr_sum[K] / self.kfr_cnt[K]) if self.kfr_cnt[K] > 0 else float("nan")
                    for K in self.K_list}

        # 导出保留曲线
        rows = []
        for d in sorted(self.retain_den.keys()):
            den = self.retain_den[d]
            if den > 0:
                r = self.retain_num[d] / den
                rows.append((d * self.eval_interval, r))
        rows.sort()
        with open(self.retain_csv, "w", newline="") as f:
            w = csv.writer(f);
            w.writerow(["delta_steps", "retention"]);
            w.writerows(rows)

        # 半衰期
        rhl_steps = None
        for s, r in rows:
            if r <= rhl_threshold:
                rhl_steps = s
                break
        rhl_ge = None
        if rhl_steps is None and return_lower_bound and rows:
            rhl_ge = rows[-1][0]  # 至少大于等于这个步数

        return {"kfr_mean": kfr_mean, "rhl_steps": rhl_steps, "rhl_ge": rhl_ge}


class OnlineEvalHook:
    """
    训练时每 eval_interval 步触发一次评测，自动更新 meter。
    predict_fn: logits = predict_fn(x)（一般传 lambda x: learner.model.logits(x)）
    preprocess_fn: x = preprocess_fn(x)（一般传 learner.transform_test）
    """
    def __init__(self, model: torch.nn.Module, val_loader: DataLoader, device: torch.device,
                 meter: KFRRHLMeter,
                 predict_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 use_autocast: bool = True):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.meter = meter
        self.eval_interval = meter.eval_interval
        self.predict_fn = predict_fn
        self.preprocess_fn = preprocess_fn
        self.use_autocast = use_autocast

    def maybe(self, model: torch.nn.Module, global_step: int):
        if global_step % self.eval_interval != 0:
            return
        mask = evaluate_correct_mask(model=self.model,
                                     loader=self.val_loader,
                                     device=self.device,
                                     predict_fn=self.predict_fn,
                                     preprocess_fn=self.preprocess_fn,
                                     use_autocast=self.use_autocast)
        self.meter.update(mask, global_step)
