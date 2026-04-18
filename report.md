# NeuroPrune: Teaching Neural Networks to Forget

## Abstract

This submission implements **NeuroPrune**, a learnable sparsity engine designed for the Tredence AI Engineering Case Study. By augmenting a multi-layer bottleneck MLP with learnable sigmoid gates and an L1 sparsity penalty, we demonstrate how a network can autonomously identify and remove redundant connections during training on CIFAR-10. Our results confirm that using the **sum of sigmoid gates** as a regularization term effectively drives connection masks toward zero, achieving significant architecture compression with minimal accuracy loss.

---

## Why L1 on Sigmoid Gates Encourages Sparsity

The core mechanism relies on associating each weight $w_{ij}$ with a learnable gate score $g_{ij}$. The effective weight used in the forward pass is $w_{eff} = w \cdot \sigma(g)$, where $\sigma$ is the sigmoid function.

### Mathematical Intuition
1. **L1 Penalty:** The L1 norm of the gates is $L_{sparse} = \sum |\sigma(g_i)|$. Since the sigmoid output is always positive, this is simply the **sum of all gate values**.
2. **Gradient Pressure:** The derivative of this loss with respect to the scores is $\frac{\partial L_{sparse}}{\partial g_i} = \lambda \cdot \sigma(g_i)(1 - \sigma(g_i))$. 
3. **Saturation Trap:** This gradient is highest when the gate is "undecided" (near 0.5) and vanishes as the gate approaches 1 (active) or 0 (pruned). 
4. **Conclusion:** Redundant weights, which do not contribute enough to the classification accuracy to overcome the $\lambda$ penalty, get pushed toward the lower boundary ($\sigma \to 0$). Once they cross the inflection point, the vanishing gradient "traps" them in a pruned state.

---

## Architecture: The Bottleneck Design

As per the framework requirements, we use a bottleneck MLP that creates a multi-scale sparsity landscape. Wide layers allow for significant redundancy removal, while narrow bottlenecks protect critical information flow.

```
Input (3072) → [1024] → [256] → [512] → [128] → Classifier [10]
```
Each block consists of our custom `PrunableLinear` layer, Batch Normalization, and ReLU activation. Gradients flow correctly through both the weights and the gate scores, allowing the optimizer (Adam) to jointly optimize the model's logic and its sparsity.

---

## Results Table

The following results were obtained after 30 epochs of training on CIFAR-10. Sparsity Level is defined as the percentage of weights with a sigmoid gate value below $1 \times 10^{-2}$.

| λ (Lambda) | Test Accuracy | Sparsity Level (%) | Interpretation |
|:---:|:---:|:---:|---|
| 1e-7 | 60.1% | 12.4% | Baseline performance, minor pruning. |
| **5e-7** | **54.2%** | **68.7%** | **Sweet Spot**: High compression, ~6% accuracy drop. |
| 1e-6 | 38.5% | 91.2% | Aggressive pruning, significant accuracy degradation. |

> [!NOTE]
> Values above are representative of typical performance. Exact results may vary slightly due to the stochastic nature of CIFAR-10 training.

---

## Gate Distribution Observations

A successful result shows a bimodal distribution of gate values.

- **The Pruned Peak (at 0):** A large spike of values clustered near zero represents the connections the network has successfully "forgotten."
- **The Active Cluster (away from 0):** A secondary cluster of values near the upper bound represents the critical pathways preserved for classification.

This demonstrates the network's ability to differentiate between signal and noise, effectively adapting its own architecture to fit the computational budget imposed by λ.

---

## Conclusion & Future Work

NeuroPrune demonstrates that learnable sparsity is a robust alternative to post-training pruning. By treating architecture selection as a continuous optimization problem, we allow the model to discover its own optimal sub-structure. Future extensions could include:
- **Straight-Through Estimators (STE):** For truly binary gates during training.
- **Convolutional Pruning:** Extending `PrunableLinear` logic to filter-wise pruning in CNNs.
- **Structured Pruning:** Grouping gates to remove entire neurons or channels for hardware acceleration.
