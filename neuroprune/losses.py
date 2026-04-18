"""
losses.py — SparsityLoss: An nn.Module wrapping the L1 gate penalty.

Mathematical Intuition
─────────────────────
Given gate scores g, the effective gate is σ(g) = 1/(1+e^{-g}).
L1 loss on σ(g) minimizes Σ |σ(g_i)|, which drives gate values toward 0.

Why sigmoid + L1 creates sparsity:
  ∂L_sparse/∂g = λ · σ(g) · (1 - σ(g))

At g → -∞: σ(g) → 0, gradient → 0        (gate is "stuck" pruned)
At g →  0:  σ(g) = 0.5, gradient is max   (undecided gates get pushed)
At g → +∞: σ(g) → 1, gradient → 0         (gate is "stuck" active)

This creates a "win-lose" dynamic: active gates survive (gradient saturates),
undecided gates get pushed toward 0 by the L1 pressure from λ.
"""

import torch
import torch.nn as nn

from neuroprune.layers import PrunableLinear


class SparsityLoss(nn.Module):
    """
    L1 penalty on sigmoid gates across all PrunableLinear layers in a model.

    Encourages hard zeros through gradient saturation at sigmoid boundaries.
    Auto-discovers all PrunableLinear layers via isinstance checks — works
    with any model architecture passed to forward().

    Usage:
        sparsity_criterion = SparsityLoss()
        loss_sparse = sparsity_criterion(model)  # scalar tensor
        total_loss = ce_loss + λ * loss_sparse

    Design note:
        The L1 is applied to sigmoid(gate_scores), NOT the raw scores.
        This ensures the penalty is bounded in [0, 1] and the gradient
        has the desired saturation behavior at the extremes.
    """

    def __init__(self):
        super().__init__()

    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute L1 sparsity loss across all PrunableLinear layers.

        Args:
            model (nn.Module): Any model containing PrunableLinear layers.

        Returns:
            torch.Tensor: Scalar mean-L1 of all gate values. Returns 0 if
                          no PrunableLinear layers are found (safe fallback).
        """
        gate_values = []

        # Auto-discover all PrunableLinear layers (works for nested modules)
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                # Apply sigmoid to gate scores and flatten
                gates = torch.sigmoid(module.gate_scores)
                gate_values.append(gates.view(-1))

        if not gate_values:
            # No prunable layers found — return zero loss safely
            return torch.tensor(0.0, requires_grad=True)

        # Concatenate all gates and compute the sum (L1 norm for positive sigmoid gates)
        all_gates = torch.cat(gate_values)
        return all_gates.sum()

    def extra_repr(self) -> str:
        return "penalty=L1, gates=sigmoid(gate_scores), auto_discover=True"
