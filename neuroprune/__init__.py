"""
NeuroPrune: A Learnable Sparsity Engine for CIFAR-10
=====================================================
A mini-framework for training self-pruning neural networks
using learnable sigmoid gates with L1 sparsity regularization.
"""

__version__ = "1.0.0"
__author__ = "NeuroPrune"

from neuroprune.layers import PrunableLinear
from neuroprune.losses import SparsityLoss
from neuroprune.model import BottleneckMLP
from neuroprune.trainer import NeuroPruneTrainer
from neuroprune.visualizer import NeuroPruneVisualizer

__all__ = [
    "PrunableLinear",
    "SparsityLoss",
    "BottleneckMLP",
    "NeuroPruneTrainer",
    "NeuroPruneVisualizer",
]
