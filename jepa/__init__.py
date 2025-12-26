"""
JEPA (Joint Embedding Predictive Architecture) for Sepsis Prediction
=====================================================================

A PyTorch implementation of JEPA for learning patient physiological dynamics
through self-supervised pre-training on clinical time-series data.

Architecture:
    - Transformer Encoder for temporal representation learning
    - MLP Predictor for future state prediction
    - VICReg loss for representation regularization

Usage:
    from jepa import SepsisJEPA, SepsisClassifier, VICRegLoss
    from jepa.config import JEPAConfig, TrainingConfig
"""

from jepa.model import SepsisJEPA, SepsisClassifier
from jepa.loss import VICRegLoss
from jepa.config import JEPAConfig, TrainingConfig

__version__ = "0.1.0"
__all__ = [
    "SepsisJEPA",
    "SepsisClassifier",
    "VICRegLoss",
    "JEPAConfig",
    "TrainingConfig",
]
