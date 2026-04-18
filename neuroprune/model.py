"""
model.py — BottleneckMLP: A multi-scale prunable architecture for CIFAR-10.

Architecture Design Rationale
─────────────────────────────
Instead of a flat MLP (which gives uniform pruning pressure everywhere),
we use a "bottleneck" structure that creates two regimes:

  1. Wide Dense Blocks (1024, 512): High capacity, many prunable connections.
     These are where most pruning happens — the model has redundancy to spare.

  2. Narrow Bottlenecks (256, 128): Information compression points.
     Gates here are harder to prune: each weight carries more signal.

This creates an interesting multi-scale sparsity distribution:
  - Dense blocks: potentially 70-90% prunable
  - Bottlenecks: typically 20-50% prunable (gates protect signal)

The bottleneck structure also means the model can learn WHICH paths to
preserve, not just whether to prune at all — a richer optimization landscape.

Architecture:
  Input (3072) 
    → [DenseBlock1: PrunableLinear(3072→1024) + BN + ReLU]
    → [Bottleneck1: PrunableLinear(1024→256) + BN + ReLU]
    → [DenseBlock2: PrunableLinear(256→512) + BN + ReLU]   ← expansion
    → [Bottleneck2: PrunableLinear(512→128) + BN + ReLU]
    → [Classifier:  Linear(128→10)]                         ← no gating on head
"""

import torch
import torch.nn as nn

from neuroprune.layers import PrunableLinear


class DenseBlock(nn.Module):
    """
    A single prunable dense block: PrunableLinear + BatchNorm1d + ReLU.

    Args:
        in_features  (int): Input dimensionality.
        out_features (int): Output dimensionality.
        dropout      (float): Dropout rate after activation. Default: 0.1.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear = PrunableLinear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.bn(self.linear(x))))

    def __repr__(self) -> str:
        return (
            f"DenseBlock({self.linear.in_features}→{self.linear.out_features} "
            f"| sparsity={self.linear.sparsity:.1f}%)"
        )


class BottleneckMLP(nn.Module):
    """
    Bottleneck MLP for CIFAR-10 with learnable sparsity gates.

    The architecture alternates between wide dense blocks (learning capacity)
    and narrow bottlenecks (information compression), creating a rich
    multi-scale pruning landscape.

    CIFAR-10 input: 3×32×32 = 3072 floats (flattened).
    Output: 10-class logits (no sigmoid/softmax — use nn.CrossEntropyLoss).

    Args:
        dropout (float): Dropout in dense/bottleneck blocks. Default: 0.1.
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()

        # Stage 1: Wide expansion
        self.dense1 = DenseBlock(3072, 1024, dropout=dropout)

        # Stage 2: First compression bottleneck
        self.bottle1 = DenseBlock(1024, 256, dropout=dropout)

        # Stage 3: Re-expansion (gives the model a second capacity surge)
        self.dense2 = DenseBlock(256, 512, dropout=dropout)

        # Stage 4: Final compression bottleneck
        self.bottle2 = DenseBlock(512, 128, dropout=dropout)

        # Classification head: plain Linear (we don't prune the head)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dims: (B, 3, 32, 32) → (B, 3072)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.bottle1(x)
        x = self.dense2(x)
        x = self.bottle2(x)
        return self.classifier(x)

    def get_prunable_layers(self) -> list:
        """Returns all PrunableLinear layers in order."""
        return [
            self.dense1.linear,
            self.bottle1.linear,
            self.dense2.linear,
            self.bottle2.linear,
        ]

    def global_sparsity(self) -> float:
        """
        Computes global sparsity across ALL prunable weights.
        Weighted by layer size so large layers dominate the metric.
        """
        total_pruned = 0
        total_weights = 0
        for layer in self.get_prunable_layers():
            n = layer.gate_scores.numel()
            pruned = int((torch.sigmoid(layer.gate_scores) * layer.hard_mask < 1e-2)
                         .float().sum().item())
            total_pruned += pruned
            total_weights += n
        if total_weights == 0:
            return 0.0
        return (total_pruned / total_weights) * 100.0

    def parameter_count(self) -> dict:
        """Returns total, prunable, and active parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        gate_params = sum(
            l.gate_scores.numel() for l in self.get_prunable_layers()
        )
        return {
            "total": total,
            "gate_params": gate_params,
            "weight_params": total - gate_params,
        }

    def __repr__(self) -> str:
        lines = ["BottleneckMLP("]
        for name, layer in [
            ("dense1 ", self.dense1),
            ("bottle1", self.bottle1),
            ("dense2 ", self.dense2),
            ("bottle2", self.bottle2),
        ]:
            lines.append(f"  ({name}): {layer}")
        lines.append(f"  (head  ): Linear(128→10)")
        lines.append(
            f"  Global Sparsity: {self.global_sparsity():.1f}%"
        )
        lines.append(")")
        return "\n".join(lines)
