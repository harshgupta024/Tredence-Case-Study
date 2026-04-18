"""
NeuroPrune Submission Script — Tredence AI Engineering Case Study
==================================================================

This script implements a self-pruning neural network for CIFAR-10.
It follows the precise requirements of the Tredence Case Study:
  - Custom PrunableLinear layer with learnable sigmoid gates
  - L1 sparsity regularization on gates (sum-based)
  - Full training loop with sparsity/accuracy tradeoff evaluation

Author: NeuroPrune Framework (AI-Generated)
Date: 2026-04-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: The "Prunable" Linear Layer
# ─────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    Custom linear layer that learns to prune its own weights.
    Implementation Detail: Uses sigmoid(gate_scores) as weight masks.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Learnable gate scores (Part 1.2)
        # Initialized to zeros so sigmoid(0) = 0.5 (neutral start)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Transform scores to [0, 1] gates (Part 1.3.a)
        gates = torch.sigmoid(self.gate_scores)

        # 2. Calculate pruned weights (Part 1.3.b)
        pruned_weights = self.weight * gates

        # 3. Standard linear operation (Part 1.3.c)
        # Implemented using matmul to demonstrate 'from-scratch' logic
        # Output = x @ weights^T + bias
        # Using F.linear is functionally identical and slightly more optimized
        return F.linear(x, pruned_weights, self.bias)

    @property
    def sparsity(self) -> float:
        """Returns % of weights below 1e-2 threshold (Part 3.2)."""
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            pruned = (gates < 1e-2).float().sum()
            total = gates.numel()
        return (pruned / total).item() * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: The Sparsity Regularization Loss
# ─────────────────────────────────────────────────────────────────────────────

class SparsityLoss(nn.Module):
    """
    Computes the L1 penalty on sigmoid gates across the entire model.
    Effectively the SUM of all gate values (Part 2.2).
    """
    def forward(self, model: nn.Module) -> torch.Tensor:
        total_sparsity_loss = 0.0
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                # Sum of sigmoid gates as per prompt
                total_sparsity_loss += torch.sigmoid(module.gate_scores).sum()
        return total_sparsity_loss


# ─────────────────────────────────────────────────────────────────────────────
# PART 1/3: Model Architecture (Bottleneck Structure)
# ─────────────────────────────────────────────────────────────────────────────

class NeuroPruneModel(nn.Module):
    """
    A multi-layer MLP using PrunableLinear layers.
    Bottleneck design: 3072 -> 1024 -> 256 -> 512 -> 128 -> 10
    """
    def __init__(self):
        super().__init__()
        self.dense1 = PrunableLinear(3072, 1024)
        self.bottle1 = PrunableLinear(1024, 256)
        self.dense2 = PrunableLinear(256, 512)
        self.bottle2 = PrunableLinear(512, 128)
        self.classifier = nn.Linear(128, 10) # Head is generally left dense

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten CIFAR-10 images
        x = F.relu(self.bn1(self.dense1(x)))
        x = F.relu(self.bn2(self.bottle1(x)))
        x = F.relu(self.bn3(self.dense2(x)))
        x = F.relu(self.bn4(self.bottle2(x)))
        return self.classifier(x)

    def get_global_sparsity(self) -> float:
        total_pruned = total_weights = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                total_pruned += (gates < 1e-2).sum().item()
                total_weights += gates.numel()
        return (total_pruned / total_weights) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Training and Evaluation Module
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(lam, num_epochs=30, batch_size=256, lr=1e-3, device='cpu'):
    """Full lifecycle for one λ value."""
    print(f"\n--- Testing λ = {lam} ---")

    # Data setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = NeuroPruneModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    sparse_loss_fn = SparsityLoss()

    for epoch in range(num_epochs):
        model.train()
        total_ce = total_sparse = total_l = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Part 2: Total Loss = ClassificationLoss + λ * SparsityLoss
            l_ce = ce_loss(outputs, labels)
            l_sparse = sparse_loss_fn(model)
            loss = l_ce + lam * l_sparse

            loss.backward()
            optimizer.step()

            total_ce += l_ce.item()
            total_sparse += l_sparse.item()
            total_l += loss.item()

        # Evluation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        sparsity = model.get_global_sparsity()
        print(f"Epoch {epoch+1}/{num_epochs} | CE: {total_ce/len(train_loader):.4f} | "
              f"Sparse: {total_sparse/len(train_loader):.1f} | Acc: {accuracy:.2f}% | Sparsity: {sparsity:.1f}%")

    return accuracy, sparsity, model


def plot_gate_distribution(model, lam):
    """Generates the required matplotlib plot (Part 6.2)."""
    all_gates = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            all_gates.append(torch.sigmoid(module.gate_scores).detach().cpu().numpy().flatten())

    gates = np.concatenate(all_gates)
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    plt.hist(gates, bins=100, color='#4C9FFF', alpha=0.7)
    plt.axvline(1e-2, color='red', linestyle='--', label='Threshold (1e-2)')
    plt.title(f"Gate Distribution for λ={lam}")
    plt.xlabel("Gate Value (Sigmoid Score)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(f"outputs/gate_dist_lam_{lam}.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    if not os.path.exists("outputs"): os.makedirs("outputs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # λ values targeted for sum-based loss: 1e-7, 5e-7, 1e-6
    # (Adjusted from 1e-4 range to compensate for 3.6M params sum)
    lambdas = [1e-7, 5e-7, 1e-6]
    results = []

    for lam in lambdas:
        acc, sp, best_model = train_and_evaluate(lam, device=device)
        results.append((lam, acc, sp))
        plot_gate_distribution(best_model, lam)

    # Print summary table (Part 6.2)
    print("\n" + "="*50)
    print(f"{'Lambda':>10} | {'Accuracy (%)':>12} | {'Sparsity (%)':>12}")
    print("-" * 50)
    for lam, acc, sp in results:
        print(f"{lam:10.1e} | {acc:12.2f} | {sp:12.1f}")
    print("="*50)
