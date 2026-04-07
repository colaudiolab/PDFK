# PDFK: Perturbing to Defend Fragile Knowledge (OCL)

> Official implementation of **PDFK**, a lightweight temporal–spatial defense for **Online Continual Learning (OCL)**.  
> Key idea: (i) **EMA** smooths temporal oscillations; (ii) **targeted parameter perturbation + consistency** flattens sharp regions and enlarges the *fragility radius*.  
> Plug-and-play for replay pipelines; no task boundaries, no extra buffers.

---

## TL;DR

- **+4–8%** avg ACC over strong replay baselines on CIFAR-100 / Tiny-ImageNet (incl. **blurry** boundaries).  
- **−15–20%** forgetting on average.  
- **<8%** training-time overhead, **no** extra memory for model copies (uses a temporary proxy).

---

## Repo Layout

```
PDFK/
├─ config/
│  └─ aaai26/        # YAML/JSON configs for datasets & schedules
├─ src/
│  ├─ buffers/       # Replay buffers (e.g., reservoir, ring)
│  ├─ datasets/      # CIFAR/Tiny-IN loaders, blurry samplers
│  ├─ learners/      # OCL learners (ER, MKD, PDFK-ER, etc.)
│  ├─ models/        # Backbones (ResNet/ViT) & heads
│  └─ utils/         # Logging, metrics, schedulers, seeds
```

The **perturbation module** lives inside `learners` (via a `Perturber` class) and is invoked during each step when mixing current data with replay.

---

## Environment

```bash
# Python 3.9+ / PyTorch 2.1+ recommended
conda create -n pdfk python=3.9 -y
conda activate pdfk

# Install PyTorch (choose your CUDA version on pytorch.org)
pip install torch torchvision torchaudio

# Common dependencies
pip install -r requirements.txt
# (If the file is not present, typical deps include: numpy, pyyaml, tqdm, scikit-learn, matplotlib, faiss-cpu (optional))
```

---

## Quick Start

### 1) CIFAR-100 (Stream OCL, replay with ER + MKD + PDFK)

```bash
python -m src.learners.train   --dataset cifar100   --buffer-size 500   --batch-size 32   --arch resnet18   --optimizer sgd --lr 0.1 --momentum 0.9 --weight-decay 5e-4   --ema 0.999    --use-mkd   --use-pdfk   --p-steps 1 --p-gamma 0.05 --p-lam 0.01   --seed 1
```

### 2) Tiny-ImageNet (Blurry boundaries)

```bash
python -m src.learners.train   --dataset tinyimagenet   --blurry --blurry-sigma 0.15   --buffer-size 1000   --batch-size 32   --arch resnet18   --optimizer sgd --lr 0.1 --momentum 0.9 --weight-decay 5e-4   --ema 0.999   --use-mkd   --use-pdfk   --p-steps 1 --p-gamma 0.05 --p-lam 0.01   --seed 1
```

> Tip: configs under `config/aaai26/` provide ready-to-run YAMLs. You can pass `--cfg config/aaai26/cifar100_pdfk.yaml` to load all arguments.

---

## PDFK Arguments

| Flag | Meaning | Default |
|---|---|---|
| `--use-pdfk` | Enable spatial perturbation + consistency | off |
| `--p-steps` | # of inner ascent steps for the proxy | `1` |
| `--p-gamma` | Step size / how far to move from original weights | `0.05` |
| `--p-lam` | Weight for KL consistency loss | `0.01` |

**Inner step (proxy, per batch):**
- Copy weights to a **proxy** model; add tiny **layer-wise noise** (proportional to layer norms).
- Do `p-steps` of **normalized gradient ascent** to **maximize**  
  \[
  \mathrm{KL}(q_{\theta+\delta}(x) \,\|\, q_\theta(x))
  \]
  (masked on currently correct replay samples).
- Normalize gradients layer-wise and step the proxy with LR `p-gamma/p-steps`.

**Outer step (main model):**
- Compute **KL consistency** between main model and proxy outputs (on the same batch), multiply by `p-lam`.
- Backprop this KL **together** with the standard OCL loss (CE + MKD).
- **Restore** main weights (remove added diff) after backward to avoid drift; update with SGD/AdamW.

---

## Blurry Task Boundaries

Enable with `--blurry`. The stream is a **time-varying mixture** of latent tasks:
\[
(x_t,y_t)\sim\sum_k \pi_k(t)\,\mathcal{P}^{(k)} ,\quad
\pi_k(t)=\frac{\phi((t-\mu_k)/\sigma)}{\sum_j \phi((t-\mu_j)/\sigma)}.
\]
Control overlap via `--blurry-sigma`. This stresses **temporal robustness** (no boundary labels).

---

## Metrics & Logging

- **ACC** (final average accuracy), **Forgetting** (avg drop), **BWT/FWT**, per-task accuracy.  
- **Fragile fraction** (optional): proportion of $(\varepsilon,n)$-fragile samples over time.  
- CSV logs + tensorboard in `runs/…`.

---

## Reproducing Main Numbers

1. Set seeds `{1,2,3}` and average.  
2. Buffers: `{200, 500, 1000}` for CIFAR-100; `{500,1000}` for Tiny-IN.  
3. Compare `ER`, `ER+MKD`, `ER+MKD+PDFK` under **equal compute** (same #forward/backward, proxy 1 step).  
4. For SAM/AWP baselines, match extra passes to our proxy step.

---

## Results Snapshot

- CIFAR-100 (buffer 500): **ACC +4–5%**, **Forgetting −15–18%** vs. strong replay.  
- Tiny-IN (blurry): **ACC +6–8%**; sharpness proxy drops by **30%+**, validating flattening.

*(Exact tables depend on seeds/config; see `config/aaai26/` and logs.)*

---

## Acknowledgements

The perturbation proxy is inspired by **AWP** (Adversarial Weight Perturbation). We adapted and simplified the implementation for OCL.

---

## License

This project is for academic research. See `LICENSE` for details.
