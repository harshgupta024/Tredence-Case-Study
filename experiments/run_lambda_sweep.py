"""
run_lambda_sweep.py — One command to run all 3 λ experiments.

Usage:
    python experiments/run_lambda_sweep.py

Runs training for λ ∈ {1e-4, 1e-3, 1e-2} sequentially, saves all plots,
logs results to outputs/results.json, and prints a final comparison table.

All experiments use:
  - Seed: 42 (reproducibility)
  - Dataset: CIFAR-10 (auto-downloaded to data/)
  - Epochs: 30
  - Learning rate: 1e-3
  - Batch size: 256
  - Device: CUDA if available, else CPU
"""

import json
import sys
import time
from pathlib import Path

# Add project root to sys.path so `neuroprune` package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from neuroprune import BottleneckMLP, NeuroPruneTrainer, NeuroPruneVisualizer

# ── Configuration ─────────────────────────────────────────────────────────────
# With SparsityLoss using .sum(), λ values must be much smaller than with .mean()
# For ~4M parameters, λ=1e-6 applies a global pressure similar to λ=4.0 with mean.
LAMBDAS = [1e-7, 5e-7, 1e-6] 
NUM_EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
SEED = 42
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

# ── Setup ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[NeuroPrune] Device: {DEVICE.upper()}")
print(f"[NeuroPrune] Output: {OUTPUT_DIR}")
print(f"[NeuroPrune] Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"[NeuroPrune] λ sweep: {LAMBDAS}\n")


# ── Data Loaders ──────────────────────────────────────────────────────────────

def build_dataloaders():
    """
    Build CIFAR-10 train and test loaders with standard augmentation.

    Train: Random horizontal flip + crop (standard CIFAR-10 augmentation).
    Test:  Normalize only (no augmentation).

    CIFAR-10 mean/std are dataset statistics for proper normalization.
    """
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )

    num_workers = 2 if DEVICE == "cuda" else 0

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=(DEVICE == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=(DEVICE == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    return train_loader, test_loader


# ── Single Experiment ─────────────────────────────────────────────────────────

def run_experiment(lam: float, train_loader, test_loader) -> dict:
    """
    Full training run for a single λ value.

    Args:
        lam          (float): Sparsity coefficient.
        train_loader:         Training data loader.
        test_loader:          Test data loader.

    Returns:
        dict with 'test_acc', 'sparsity', 'history', 'model'.
    """
    # Fresh model + trainer for each λ (independent experiments)
    torch.manual_seed(SEED)
    model = BottleneckMLP(dropout=0.1)

    trainer = NeuroPruneTrainer(
        model=model,
        lam=lam,
        lr=LEARNING_RATE,
        device=DEVICE,
    )
    trainer.setup_scheduler(num_epochs=NUM_EPOCHS)

    history = {
        "train_loss":  [],
        "ce_loss":     [],
        "sparse_loss": [],
        "test_acc":    [],
        "sparsity":    [],
    }

    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = trainer.train_epoch(train_loader)
        eval_metrics  = trainer.evaluate(test_loader)
        sparsity      = trainer.get_global_sparsity()

        metrics = {**train_metrics, **eval_metrics}
        trainer.log_epoch(epoch, metrics)

        history["train_loss"].append(train_metrics["train_loss"])
        history["ce_loss"].append(train_metrics["ce_loss"])
        history["sparse_loss"].append(train_metrics["sparse_loss"])
        history["test_acc"].append(eval_metrics["test_acc"])
        history["sparsity"].append(sparsity)

    trainer.print_footer()

    elapsed = time.time() - t_start
    print(f"  ↳ Training time: {elapsed:.1f}s | Final sparsity: {history['sparsity'][-1]:.1f}%\n")

    # Hard-prune at end
    trainer.freeze_all_pruned(threshold=1e-2)

    final_acc = eval_metrics["test_acc"]
    final_sparsity = history["sparsity"][-1]

    # Save model checkpoint
    ckpt_path = OUTPUT_DIR / f"model_lambda_{lam}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[NeuroPrune] Checkpoint saved: {ckpt_path}")

    return {
        "test_acc":  final_acc,
        "sparsity":  final_sparsity,
        "history":   history,
        "model":     model,
        "lam":       lam,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    visualizer = NeuroPruneVisualizer(OUTPUT_DIR)
    train_loader, test_loader = build_dataloaders()

    all_results = {}

    for lam in LAMBDAS:
        print(f"\n{'═' * 70}")
        print(f"  EXPERIMENT: λ = {lam}")
        print(f"{'═' * 70}")
        result = run_experiment(lam, train_loader, test_loader)
        all_results[lam] = result

        # Plot 1: Gate distribution for each λ
        visualizer.plot_gate_distribution(
            model=result["model"],
            lam=lam,
        )

        # Plot 3: Training dynamics for each λ
        visualizer.plot_training_dynamics(
            history=result["history"],
            lam=lam,
        )

    # Plot 2: Tradeoff curve (uses all λ results)
    tradeoff_data = {
        lam: {"test_acc": r["test_acc"], "sparsity": r["sparsity"]}
        for lam, r in all_results.items()
    }
    visualizer.plot_sparsity_accuracy_tradeoff(results=tradeoff_data)

    # ── Final Summary Table ───────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'═' * 70}")
    print(f"{'λ':>8} | {'Test Accuracy':>14} | {'Global Sparsity':>16}")
    print(f"{'─' * 8}─|─{'─' * 14}─|─{'─' * 16}")

    serializable = {}
    for lam in LAMBDAS:
        r = all_results[lam]
        print(f"{lam:>8} | {r['test_acc']:>13.2f}% | {r['sparsity']:>15.1f}%")
        serializable[str(lam)] = {
            "test_acc": round(r["test_acc"], 3),
            "sparsity": round(r["sparsity"], 3),
            "history": {
                k: [round(v, 4) for v in vals]
                for k, vals in r["history"].items()
            },
        }

    print(f"{'═' * 70}\n")

    # Save results JSON
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[NeuroPrune] Results saved to: {results_path}")
    print(f"[NeuroPrune] Plots saved to:   {OUTPUT_DIR}/")
    print("\n✓ Lambda sweep complete.\n")


if __name__ == "__main__":
    main()
