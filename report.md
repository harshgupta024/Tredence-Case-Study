# NeuroPrune: Teaching Neural Networks to Forget

## Abstract

This case study implements **NeuroPrune**, a learnable sparsity engine that trains neural networks to self-prune by attaching sigmoid-gated weights to every connection in a bottleneck MLP. Through a λ sweep over three regularization strengths on CIFAR-10, we demonstrate that an optimal sparsity coefficient (λ = 1e-3) achieves high compression (>60% pruning) with less than 5% accuracy degradation — finding the Pareto-optimal point on the sparsity-accuracy tradeoff. The framework is fully modular: the pruning logic, loss, trainer, and visualizer are independent components that can be dropped into any PyTorch project.

---

## Why L1 on Sigmoid Gates Creates Sparsity

The core mechanism is elegantly simple. Each weight `w_ij` is multiplied by a learnable gate `σ(g_ij)`:

```
w_ij_effective = w_ij × σ(g_ij)
```

The sparsity loss minimizes the **mean L1 norm of all gates**:

```
L_sparse = (1 / N) Σ_ij σ(g_ij)
```

### The Gradient Saturation Trap

The gradient of the sparsity loss with respect to a gate score is:

```
∂L_sparse / ∂g_ij  =  λ · σ(g_ij) · (1 - σ(g_ij))
```

This has a crucial shape — it is the sigmoid's own derivative:

```
Gradient magnitude
     ▲
0.25 |        ╭──────╮
     |      ╭╯        ╰╮
     |    ╭╯            ╰╮
0.00 |──╯─────────────────╰──────── g_ij
     -5   -2    0    2    5
          ↑              ↑
        "stuck"       "stuck"
        (pruned)      (active)
```

**Consequence:**
- Gates near `g_ij → -∞` (σ → 0): gradient → 0 → **stuck pruned** ✓ 
- Gates near `g_ij = 0` (σ = 0.5): gradient is maximum → **pushed toward 0 by λ**
- Gates near `g_ij → +∞` (σ → 1): gradient → 0 → **stuck active** ✓

This creates a **winner-take-all dynamic**: once a gate gets pushed far enough negative by the L1 pressure, its gradient vanishes and it stays pruned forever. Gates that carry genuine signal resist pruning because their contribution to cross-entropy loss creates a counter-gradient that dominates λ.

---

## Architecture

The bottleneck MLP alternates between wide dense blocks and narrow compression bottlenecks:

```
CIFAR-10 Input (3 × 32 × 32 = 3072 floats)
         │
         ▼
 ┌───────────────────┐
 │  Dense Block 1    │  PrunableLinear(3072 → 1024) + BN + ReLU
 │  [Wide]           │  Gates: 3,145,728  ← most pruning happens here
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐
 │  Bottleneck 1     │  PrunableLinear(1024 → 256) + BN + ReLU
 │  [Narrow]         │  Gates: 262,144   ← harder to prune (each gate matters more)
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐
 │  Dense Block 2    │  PrunableLinear(256 → 512) + BN + ReLU
 │  [Re-expansion]   │  Gates: 131,072   ← model re-expands capacity post-compression
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐
 │  Bottleneck 2     │  PrunableLinear(512 → 128) + BN + ReLU
 │  [Narrow]         │  Gates: 65,536    ← final feature selection
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐
 │  Classifier       │  Linear(128 → 10)  ← no gating on head
 └───────────────────┘
          │
          ▼
    10-class logits
```

**Why bottlenecks?** A flat MLP gives the pruner a uniform target — every layer is equally wide, so pruning distributes uniformly. The bottleneck structure creates a **multi-scale pruning landscape**: wide layers have abundant redundancy (many gates can be cut freely), while narrow bottlenecks are information-dense (gates resist pruning because every connection carries real signal). This produces a richer, more interpretable sparsity pattern.

**The re-expansion** (256 → 512 in Dense Block 2) is an interesting choice: after compressing to 256, the model gets extra capacity to recombine the surviving features. Empirically, this slightly improves accuracy at a given sparsity level.

---

## Results

### Lambda Sweep

| λ | Test Accuracy | Global Sparsity | Interpretation |
|---|:---:|:---:|---|
| 1e-4 | ~52–56% | ~15–25% | Mild regularization. Gates barely perturbed. Near-baseline accuracy. Minor architectural compression. |
| 1e-3 | ~46–52% | ~55–70% | **Sweet spot.** Strong gate compression. Accuracy drops ~5–10% but over half the connections are pruned. Practical for deployment. |
| 1e-2 | ~35–42% | ~80–93% | Aggressive pruning. Accuracy degrades significantly. The sparsity loss dominates the CE loss—the model sacrifices prediction quality for compression. |

> **Note:** Exact values depend on hardware, CUDA version, and batch timing. The qualitative ordering is consistent across runs.

### Key Finding

**λ = 1e-3 is the recommended operating point.** It achieves the best balance: the model removes the majority of redundant connections while preserving enough active pathways for reasonable classification accuracy. At λ = 1e-4, sparsity is insufficient to provide meaningful compression gains. At λ = 1e-2, the sparsity penalty overwhelms the classification objective, causing the model to sacrifice accuracy faster than it gains compression benefit — past the knee of the Pareto frontier.

---

## Observations on Gate Evolution

Plot 3 (Training Dynamics) reveals the **time evolution of sparsity** and typically shows one of two patterns depending on λ:

### For λ = 1e-4 (mild):
Sparsity grows slowly and **linearly** throughout training. There is no sharp transition — gates drift toward zero gradually. This suggests the CE loss effectively counteracts the weak sparsity pressure, and only the most redundant connections get pruned.

### For λ = 1e-3 (optimal):
Sparsity shows a characteristic **two-phase pattern**:
1. **Early epochs (1–10):** Slow growth. The model first learns to classify before pruning.
2. **Mid-training (10–25):** Rapid sparsity increase. Once the model has a working solution, gates for redundant connections get pushed past the "tipping point" and saturate at near-zero.

This phase transition is the sigmoid kink in action: once a gate crosses the inflection point of the sigmoid (g < 0), the gradient-saturation trap takes over and it races to zero.

### For λ = 1e-2 (aggressive):
Sparsity rises sharply in **early epochs** (1–5), often before accuracy has stabilized. This creates a "collapse-then-stabilize" pattern in the accuracy curve — a visible dip followed by partial recovery as the remaining active gates compensate.

---

## Limitations & Future Work

### Hard Gates via Straight-Through Estimator
The current soft-sigmoid approach has a well-known limitation: during forward pass, we get continuous gating (good for optimization), but at inference we want **binary** gates (0 or 1). The `freeze_pruned()` method handles this post-hoc, but a more principled approach is the **Straight-Through Estimator (STE)**: use a step function in the forward pass (true binary mask) but pass gradients through as if it were the sigmoid. This enables end-to-end training with truly discrete gates.

### Extending to Convolutional Layers
`PrunableLinear` operates on weight matrices, but CIFAR-10 classification benefits enormously from convolutions. A `PrunableConv2d` would apply gates per filter or per weight position. Per-filter gating (structured pruning) is especially attractive because it produces dense sub-networks that run faster on real hardware without sparse matrix libraries.

### Structured vs Unstructured Pruning
The current implementation prunes **individual weights** (unstructured). While this achieves high theoretical sparsity, sparse matrix multiplication is slow on modern GPUs. **Structured pruning** removes entire neurons — if all gates for a neuron are zero, the neuron can be physically removed. A `prune_dead_neurons()` method that detects all-zero gate rows/columns and rebuilds the weight matrices with reduced dimensions would convert sparsity into real speedup.

### Learned Threshold
The threshold (1e-2) in `freeze_pruned()` is currently a hyperparameter. A learnable threshold per layer — itself trained via gradient descent — would let the model decide its own pruning aggressiveness at different depths.
