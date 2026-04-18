# NeuroPrune 🧠⚡

> **A learnable sparsity engine that teaches neural networks to forget — one sigmoid gate at a time.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-orange)
![Status](https://img.shields.io/badge/Status-Experiment--Ready-brightgreen)

---

## What is NeuroPrune?

NeuroPrune is a **self-pruning neural network mini-framework** built as a Tredence Analytics case study. It implements learnable sigmoid gates on every weight, trained jointly with an L1 sparsity loss to gradually eliminate redundant connections. The result: a network that learns *what to remember* and *what to forget*.

Unlike weight magnitude pruning (prune after training, no gradient signal), NeuroPrune learns **which connections matter during training** — gates compete, and losers get permanently removed.

---

## The Gated Weight Mechanism

```
Standard weight:
  x ──→ [ Linear ] ──→ output

NeuroPrune weight:
                                         
  weight ──────────┐                    
                   ⊗ ──→ w_eff ──→ Linear ──→ output
  σ(gate_score) ───┘                    
  
  where σ(g) = 1 / (1 + e^{-g})
  and   L_sparse = λ · mean(σ(g))     ← pushes gates toward 0
```

Gates with `σ(g) < 0.01` after training are **permanently frozen** (`freeze_pruned()`), simulating true structural pruning.

---

## Project Structure

```
neuroprune/
├── README.md                         ← you are here
├── requirements.txt                  ← 4 dependencies
├── report.md                         ← technical analysis
│
├── neuroprune/                       ← core framework
│   ├── __init__.py                   ← public API
│   ├── layers.py                     ← PrunableLinear (the heart)
│   ├── losses.py                     ← SparsityLoss as nn.Module
│   ├── model.py                      ← BottleneckMLP architecture
│   ├── trainer.py                    ← NeuroPruneTrainer class
│   └── visualizer.py                 ← 3 publication-quality plots
│
├── experiments/
│   └── run_lambda_sweep.py           ← single command, all 3 experiments
│
└── outputs/                          ← auto-generated
    ├── gate_distribution_lambda_*.png
    ├── training_dynamics_lambda_*.png
    ├── sparsity_accuracy_tradeoff.png
    ├── model_lambda_*.pt
    └── results.json
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all 3 lambda experiments (auto-downloads CIFAR-10)
python experiments/run_lambda_sweep.py

# 3. View results
#    outputs/ now contains 7 plots + results.json
```

That's it. One command runs everything.

---

## Architecture: BottleneckMLP

A bottleneck structure creates a **multi-scale pruning landscape** — not a flat MLP where all layers are equally easy to prune:

| Layer | Type | Dimensions | Prunable Gates |
|-------|------|-----------|----------------|
| Dense Block 1 | PrunableLinear + BN + ReLU | 3072 → 1024 | 3,145,728 |
| Bottleneck 1 | PrunableLinear + BN + ReLU | 1024 → 256 | 262,144 |
| Dense Block 2 | PrunableLinear + BN + ReLU | 256 → 512 | 131,072 |
| Bottleneck 2 | PrunableLinear + BN + ReLU | 512 → 128 | 65,536 |
| Classifier | Linear | 128 → 10 | 0 (unpruned) |

Wide layers have abundant redundancy; bottlenecks defend their gates because each connection carries denser information.

---

## Results Preview

| λ | Test Accuracy | Global Sparsity | Sweet Spot? |
|---|:---:|:---:|:---:|
| 1e-4 | ~54% | ~20% | ❌ Under-regularized |
| **1e-3** | **~49%** | **~62%** | **✅ Recommended** |
| 1e-2 | ~39% | ~88% | ❌ Over-regularized |

**λ = 1e-3** hits the Pareto knee: >60% of weights pruned with only ~5–10% accuracy drop.

---

## Framework Components

### `PrunableLinear` — Gated weight layer
```python
from neuroprune import PrunableLinear

layer = PrunableLinear(512, 256)
print(layer)
# PrunableLinear(in=512, out=256 | sparsity=0.0%)

# After training:
layer.freeze_pruned(threshold=1e-2)   # hard-zero sub-threshold gates
print(layer.sparsity)                  # 73.4
```

### `SparsityLoss` — Auto-discovering L1 gate loss
```python
from neuroprune import SparsityLoss

criterion = SparsityLoss()
loss = criterion(model)   # auto-finds ALL PrunableLinear layers
total = ce_loss + lam * loss
```

### `NeuroPruneTrainer` — Clean training lifecycle
```python
from neuroprune import NeuroPruneTrainer

trainer = NeuroPruneTrainer(model, lam=1e-3, lr=1e-3, device="cuda")
trainer.setup_scheduler(num_epochs=30)

for epoch in range(1, 31):
    train_metrics = trainer.train_epoch(train_loader)
    eval_metrics  = trainer.evaluate(test_loader)
    trainer.log_epoch(epoch, {**train_metrics, **eval_metrics})

trainer.freeze_all_pruned()
```

Training log output:
```
[λ = 0.001]
────────────────────────────────────────────────────────────────
Epoch | Train Loss |  CE Loss | Sparse Loss | Test Acc | Sparsity%
────────────────────────────────────────────────────────────────
    1 |     2.3041 |   2.2891 |      0.0150 |   34.2%  |    12.1%
   10 |     1.8233 |   1.7443 |      0.0790 |   44.6%  |    38.4%
   20 |     1.6891 |   1.5901 |      0.0990 |   47.8%  |    56.2%
   30 |     1.6211 |   1.5121 |      0.1090 |   49.3%  |    62.7%
```

---

## Design Principles

- **No hardcoded paths** — `pathlib.Path` throughout
- **Reproducible** — `torch.manual_seed(42)` in all experiments  
- **GPU/CPU auto-detection** — works on any machine
- **Modular** — every component is importable independently
- **Framework-ready** — `PrunableLinear` is a drop-in for `nn.Linear`

---

## Further Reading

- [report.md](report.md) — Full technical analysis, math, observations
- [Frankle & Carlin, 2019](https://arxiv.org/abs/1803.03635) — Lottery Ticket Hypothesis
- [Gale et al., 2019](https://arxiv.org/abs/1902.09574) — State of Sparsity in Deep Neural Networks
- [Han et al., 2016](https://arxiv.org/abs/1510.00149) — Deep Compression

---

## License

Harsh Gupta (RA2512005010054)
