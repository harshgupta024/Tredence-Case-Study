"""
trainer.py — NeuroPruneTrainer: A clean, self-contained training engine.

Encapsulates the full training lifecycle:
  - Mixed loss optimization (CrossEntropy + λ·SparsityLoss)
  - Per-epoch metric logging with a pretty table
  - Evaluation with accuracy computation
  - Global sparsity tracking
  - Post-training pruning finalization via freeze_all_pruned()
"""

import time
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuroprune.losses import SparsityLoss
from neuroprune.layers import PrunableLinear


class NeuroPruneTrainer:
    """
    Training engine for NeuroPrune models.

    Handles the combined loss:
        L_total = L_CE + λ · L_sparse

    where L_CE is CrossEntropyLoss and L_sparse is the mean L1 on sigmoid gates.

    Args:
        model  (nn.Module):  The model to train (should contain PrunableLinear layers).
        lam    (float):      Sparsity regularization coefficient λ.
        lr     (float):      Learning rate for Adam optimizer.
        device (str):        'cuda' or 'cpu'.
    """

    HEADER = (
        f"{'Epoch':>5} | {'Train Loss':>10} | {'CE Loss':>9} | "
        f"{'Sparse Loss':>11} | {'Test Acc':>8} | {'Sparsity%':>9}"
    )
    SEPARATOR = "─" * len(HEADER)

    def __init__(
        self,
        model: nn.Module,
        lam: float,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.lam = lam
        self.device = device

        self.ce_loss = nn.CrossEntropyLoss()
        self.sparsity_loss = SparsityLoss()

        # Adam with mild weight decay for stability
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4
        )

        # Cosine annealing — lets lr decay gently over training
        self.scheduler = None  # Set via setup_scheduler() if desired

        self._header_printed = False

    def setup_scheduler(self, num_epochs: int) -> None:
        """Attach a cosine annealing scheduler."""
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-5
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Core Training
    # ──────────────────────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """
        Run one full training epoch over the data loader.

        Returns:
            dict with keys: train_loss, ce_loss, sparse_loss
        """
        self.model.train()
        total_loss = ce_total = sparse_total = 0.0
        n_batches = len(loader)

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(images)
            ce = self.ce_loss(logits, labels)
            sparse = self.sparsity_loss(self.model)
            loss = ce + self.lam * sparse

            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            ce_total += ce.item()
            sparse_total += sparse.item()

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "train_loss":  total_loss  / n_batches,
            "ce_loss":     ce_total    / n_batches,
            "sparse_loss": sparse_total / n_batches,
        }

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate accuracy on the given data loader.

        Returns:
            dict with keys: test_acc, test_loss
        """
        self.model.eval()
        correct = total = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)
                test_loss += self.ce_loss(logits, labels).item()

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return {
            "test_acc":  (correct / total) * 100.0,
            "test_loss": test_loss / len(loader),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Sparsity Management
    # ──────────────────────────────────────────────────────────────────────────

    def get_global_sparsity(self) -> float:
        """
        Computes the global sparsity (%) across all PrunableLinear layers.
        A weight is 'pruned' if sigmoid(gate_score) * hard_mask < 1e-2.
        """
        total = pruned = 0
        for module in self.model.modules():
            if isinstance(module, PrunableLinear):
                vals = torch.sigmoid(module.gate_scores) * module.hard_mask
                pruned += (vals < 1e-2).sum().item()
                total += vals.numel()
        return (pruned / total * 100.0) if total > 0 else 0.0

    def freeze_all_pruned(self, threshold: float = 1e-2) -> None:
        """
        Permanently freezes all gates below threshold across the entire model.
        After this call, pruned weights never recover — simulating hard pruning.

        Args:
            threshold (float): Gates with sigmoid(score) < threshold are frozen.
        """
        total_frozen = 0
        for module in self.model.modules():
            if isinstance(module, PrunableLinear):
                frozen = module.freeze_pruned(threshold=threshold)
                total_frozen += frozen
        print(f"[NeuroPrune] Frozen {total_frozen:,} gate(s) at threshold={threshold}")

    # ──────────────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────────────

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Prints a clean, aligned table row for the current epoch.

        Args:
            epoch   (int):  Current epoch number (1-indexed).
            metrics (dict): Must contain: train_loss, ce_loss, sparse_loss,
                            test_acc. Sparsity is computed internally.
        """
        if not self._header_printed:
            print(f"\n[λ = {self.lam}]")
            print(self.SEPARATOR)
            print(self.HEADER)
            print(self.SEPARATOR)
            self._header_printed = True

        sparsity = self.get_global_sparsity()
        print(
            f"{epoch:>5} | "
            f"{metrics['train_loss']:>10.4f} | "
            f"{metrics['ce_loss']:>9.4f} | "
            f"{metrics['sparse_loss']:>11.4f} | "
            f"{metrics['test_acc']:>7.2f}% | "
            f"{sparsity:>8.1f}%"
        )

    def print_footer(self) -> None:
        """Print a separator after the last epoch."""
        print(self.SEPARATOR)
