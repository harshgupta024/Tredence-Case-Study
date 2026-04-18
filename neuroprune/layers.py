"""
layers.py — PrunableLinear: A Linear layer with learnable sigmoid gates.

Each weight in this layer is gated by a learnable scalar score passed through
a sigmoid. During training, L1 regularization on the gate scores drives them
toward 0 (sigmoid→0.5) and lower. With sufficient λ, gates collapse to near-0,
effectively pruning those connections.

After training, `freeze_pruned()` hard-zeroes sub-threshold gates and detaches
them from the computational graph, simulating permanent structural pruning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear with learnable per-weight sparsity gates.

    Each weight w_ij has an associated gate score g_ij. The effective weight is:
        w_ij_eff = w_ij * sigmoid(g_ij)

    The SparsityLoss applies L1 on sigmoid(g_ij), pushing gates toward zero.

    Args:
        in_features  (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias         (bool): If set to False, the layer will not learn an additive bias.
                             Default: True.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard learnable weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Gate scores: one per weight, initialized to 0 → sigmoid(0) = 0.5
        # (neutral: neither pruned nor fully active at start)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Mask for hard-pruned gates (non-trainable buffer)
        self.register_buffer(
            "hard_mask",
            torch.ones(out_features, in_features, dtype=torch.float32),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming uniform init for weights (same as nn.Linear default)."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: apply gated weights.
            effective_weight = weight * sigmoid(gate_scores) * hard_mask
        """
        gates = torch.sigmoid(self.gate_scores) * self.hard_mask
        effective_weight = self.weight * gates
        return F.linear(x, effective_weight, self.bias)

    @property
    def sparsity(self) -> float:
        """
        Returns the percentage of weights considered pruned (gate < 1e-2).

        A weight is 'pruned' if its sigmoid gate value is below 1e-2,
        meaning it contributes <1% of its potential magnitude.
        """
        with torch.no_grad():
            gate_values = torch.sigmoid(self.gate_scores) * self.hard_mask
            pruned = (gate_values < 1e-2).float().sum()
            total = gate_values.numel()
        return (pruned / total).item() * 100.0

    def freeze_pruned(self, threshold: float = 1e-2) -> int:
        """
        Permanently zeros out gates below `threshold` and detaches them from
        the computational graph. Simulates true hard structural pruning
        post-training — these weights will never recover.

        Args:
            threshold (float): Gates with sigmoid(score) < threshold are frozen.

        Returns:
            int: Number of newly frozen gates.
        """
        with torch.no_grad():
            gate_values = torch.sigmoid(self.gate_scores)
            newly_pruned_mask = (gate_values < threshold).float()

            # Update the hard mask: once pruned, always pruned
            self.hard_mask *= (1.0 - newly_pruned_mask)

            # Force gate scores for pruned weights to a very negative value
            # (sigmoid(-10) ≈ 4.5e-5, effectively zero)
            self.gate_scores.data = torch.where(
                self.hard_mask.bool(),
                self.gate_scores.data,
                torch.full_like(self.gate_scores.data, -10.0),
            )

            newly_pruned_count = int(newly_pruned_mask.sum().item())

        return newly_pruned_count

    def gate_values(self) -> torch.Tensor:
        """Returns sigmoid(gate_scores) * hard_mask as a detached tensor."""
        with torch.no_grad():
            return (torch.sigmoid(self.gate_scores) * self.hard_mask).detach().cpu()

    def __repr__(self) -> str:
        sparsity_pct = self.sparsity
        return (
            f"PrunableLinear("
            f"in={self.in_features}, "
            f"out={self.out_features} | "
            f"sparsity={sparsity_pct:.1f}%)"
        )
