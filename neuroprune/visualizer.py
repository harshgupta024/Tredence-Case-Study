"""
visualizer.py — NeuroPruneVisualizer: Three publication-quality plots.

Plots:
  1. Gate Distribution Histogram   — shows sparsity structure of best model
  2. Sparsity vs Accuracy Tradeoff — key insight: the λ sweet spot
  3. Training Dynamics Dual-Axis   — shows how sparsity evolves over training

All plots use a consistent dark theme and high-res output (150 dpi).
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works headless)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn

from neuroprune.layers import PrunableLinear

# ── Global Style ──────────────────────────────────────────────────────────────
plt.style.use("dark_background")

PALETTE = {
    "pruned": "#FF4C6A",      # vivid red for pruned gates
    "active": "#4C9FFF",      # electric blue for active gates
    "threshold": "#FFD166",   # amber for threshold line
    "accent": "#06D6A0",      # teal for secondary axis
    "grid": "#333333",
    "text": "#E0E0E0",
    "lambdas": ["#FF6B6B", "#FFD166", "#4ECDC4"],  # per-λ colors
}

DPI = 150
FIGSIZE_SINGLE = (10, 5)
FIGSIZE_WIDE = (12, 6)


class NeuroPruneVisualizer:
    """
    Generates and saves all NeuroPrune diagnostic plots.

    Args:
        output_dir (Path): Directory where PNG files are saved.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Plot 1: Gate Distribution Histogram
    # ──────────────────────────────────────────────────────────────────────────

    def plot_gate_distribution(
        self,
        model: nn.Module,
        lam: float,
        threshold: float = 1e-2,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Histogram of all sigmoid(gate_score) values across the model.

        Pruned gates (< threshold) are shown in red; active gates in blue.
        A vertical amber line marks the pruning threshold.

        Args:
            model     (nn.Module): Trained model with PrunableLinear layers.
            lam       (float):     λ value used during training (for title).
            threshold (float):     Gate value below which a weight is 'pruned'.
            filename  (str):       Output filename (auto-generated if None).

        Returns:
            Path to saved PNG.
        """
        all_gates = []
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                vals = (torch.sigmoid(module.gate_scores) * module.hard_mask)
                all_gates.append(vals.detach().cpu().numpy().ravel())

        if not all_gates:
            raise ValueError("Model has no PrunableLinear layers.")

        gates = np.concatenate(all_gates)
        pruned_mask = gates < threshold
        pruned_gates = gates[pruned_mask]
        active_gates = gates[~pruned_mask]

        pruned_pct = 100.0 * pruned_mask.mean()

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        fig.patch.set_facecolor("#0D0D0D")
        ax.set_facecolor("#111111")

        bins = np.linspace(0, 1, 80)

        # Plot active first (blue), then pruned on top (red)
        ax.hist(
            active_gates,
            bins=bins,
            color=PALETTE["active"],
            alpha=0.85,
            label=f"Active  ({100 - pruned_pct:.1f}%)",
            edgecolor="none",
        )
        ax.hist(
            pruned_gates,
            bins=bins,
            color=PALETTE["pruned"],
            alpha=0.90,
            label=f"Pruned  ({pruned_pct:.1f}%)",
            edgecolor="none",
        )

        # Threshold line
        ax.axvline(
            threshold,
            color=PALETTE["threshold"],
            linewidth=1.8,
            linestyle="--",
            label=f"Threshold = {threshold}",
        )

        # Annotation: total gates
        ax.text(
            0.97, 0.93,
            f"Total gates: {len(gates):,}",
            transform=ax.transAxes,
            ha="right", va="top",
            color=PALETTE["text"],
            fontsize=10,
            alpha=0.7,
        )

        ax.set_xlabel("Gate Value  σ(g)", color=PALETTE["text"], fontsize=12)
        ax.set_ylabel("Count", color=PALETTE["text"], fontsize=12)
        ax.set_title(
            f"Gate Value Distribution — λ = {lam}",
            color="white", fontsize=14, fontweight="bold", pad=14,
        )
        ax.legend(framealpha=0.2, fontsize=10)
        ax.tick_params(colors=PALETTE["text"])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

        plt.tight_layout()
        fname = filename or f"gate_distribution_lambda_{lam}.png"
        out = self.output_dir / fname
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Visualizer] Saved: {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Plot 2: Sparsity vs Accuracy Tradeoff
    # ──────────────────────────────────────────────────────────────────────────

    def plot_sparsity_accuracy_tradeoff(
        self,
        results: Dict[float, Dict[str, float]],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Scatter + line plot showing the tradeoff between sparsity and accuracy.

        Each λ appears as a labeled point. A dashed line connects them to show
        the Pareto frontier of the sparsity-accuracy tradeoff.

        Args:
            results (dict): {λ: {"test_acc": float, "sparsity": float}} for each λ.
            filename (str): Output filename (auto-generated if None).

        Returns:
            Path to saved PNG.
        """
        lambdas = sorted(results.keys())
        sparsities = [results[l]["sparsity"] for l in lambdas]
        accuracies = [results[l]["test_acc"] for l in lambdas]
        colors = PALETTE["lambdas"][: len(lambdas)]

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        fig.patch.set_facecolor("#0D0D0D")
        ax.set_facecolor("#111111")

        # Dashed connecting line
        ax.plot(
            sparsities, accuracies,
            color="#555555", linestyle="--", linewidth=1.5, zorder=1,
        )

        for i, (lam, sp, acc, col) in enumerate(zip(lambdas, sparsities, accuracies, colors)):
            ax.scatter(
                sp, acc,
                color=col,
                s=180,
                zorder=3,
                edgecolors="white",
                linewidths=1.2,
                label=f"λ = {lam}",
            )
            # Label above/below alternately to avoid overlap
            offset = 1.0 if i % 2 == 0 else -1.8
            ax.annotate(
                f"λ={lam}\n{acc:.1f}% acc",
                xy=(sp, acc),
                xytext=(sp + 0.5, acc + offset),
                color=col,
                fontsize=9,
                ha="left",
            )

        # Shade the "sweet spot" region
        if len(lambdas) >= 2:
            best_idx = np.argmax(
                [acc - sp * 0.3 for sp, acc in zip(sparsities, accuracies)]
            )
            ax.annotate(
                "★ Sweet Spot",
                xy=(sparsities[best_idx], accuracies[best_idx]),
                xytext=(sparsities[best_idx] - 8, accuracies[best_idx] + 2),
                color=PALETTE["threshold"],
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color=PALETTE["threshold"], lw=1.5),
            )

        ax.set_xlabel("Global Sparsity (%)", color=PALETTE["text"], fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", color=PALETTE["text"], fontsize=12)
        ax.set_title(
            "Sparsity vs Accuracy Tradeoff — λ Sweep",
            color="white", fontsize=14, fontweight="bold", pad=14,
        )
        ax.legend(framealpha=0.2, fontsize=10, loc="lower left")
        ax.tick_params(colors=PALETTE["text"])
        ax.grid(True, color=PALETTE["grid"], linestyle="--", alpha=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

        plt.tight_layout()
        fname = filename or "sparsity_accuracy_tradeoff.png"
        out = self.output_dir / fname
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Visualizer] Saved: {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Plot 3: Training Dynamics (Dual Y-axis)
    # ──────────────────────────────────────────────────────────────────────────

    def plot_training_dynamics(
        self,
        history: Dict[str, List[float]],
        lam: float,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Dual Y-axis plot showing how accuracy and sparsity co-evolve.

        Left axis:  Test Accuracy (%) over epochs  — electric blue
        Right axis: Global Sparsity (%) over epochs — teal

        This reveals whether sparsity grows gradually (soft pruning) or
        in discrete jumps (gates snapping through the sigmoid kink).

        Args:
            history (dict): Must contain 'test_acc' and 'sparsity' lists.
            lam     (float): λ value for plot title.
            filename (str):  Output filename.

        Returns:
            Path to saved PNG.
        """
        epochs = list(range(1, len(history["test_acc"]) + 1))
        test_acc = history["test_acc"]
        sparsity = history["sparsity"]

        fig, ax1 = plt.subplots(figsize=FIGSIZE_WIDE)
        fig.patch.set_facecolor("#0D0D0D")
        ax1.set_facecolor("#111111")

        # Left axis: Accuracy
        color_acc = PALETTE["active"]
        ax1.set_xlabel("Epoch", color=PALETTE["text"], fontsize=12)
        ax1.set_ylabel("Test Accuracy (%)", color=color_acc, fontsize=12)
        line1, = ax1.plot(
            epochs, test_acc,
            color=color_acc,
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="Test Accuracy",
            zorder=3,
        )
        ax1.fill_between(epochs, test_acc, alpha=0.12, color=color_acc)
        ax1.tick_params(axis="y", labelcolor=color_acc)
        ax1.tick_params(axis="x", colors=PALETTE["text"])
        ax1.set_ylim(bottom=0)

        # Right axis: Sparsity
        ax2 = ax1.twinx()
        color_sp = PALETTE["accent"]
        ax2.set_ylabel("Global Sparsity (%)", color=color_sp, fontsize=12)
        line2, = ax2.plot(
            epochs, sparsity,
            color=color_sp,
            linewidth=2.2,
            linestyle="--",
            marker="s",
            markersize=4,
            label="Sparsity",
            zorder=3,
        )
        ax2.fill_between(epochs, sparsity, alpha=0.10, color=color_sp)
        ax2.tick_params(axis="y", labelcolor=color_sp)
        ax2.set_ylim(0, 100)

        # Unified legend
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", framealpha=0.2, fontsize=10)

        ax1.set_title(
            f"Training Dynamics — λ = {lam}",
            color="white", fontsize=14, fontweight="bold", pad=14,
        )
        ax1.grid(True, color=PALETTE["grid"], linestyle="--", alpha=0.35)

        for spine in ax1.spines.values():
            spine.set_edgecolor("#333333")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#333333")

        plt.tight_layout()
        fname = filename or f"training_dynamics_lambda_{lam}.png"
        out = self.output_dir / fname
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Visualizer] Saved: {out}")
        return out
