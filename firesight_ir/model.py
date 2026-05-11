"""
firesight_ir.model
==================

Multi-branch physics-informed neural network for satellite IR wildfire
classification. Four input branches (CNN over IR patches, MLP over ERA5
atmospheric features, MLP over MODIS+OSM surface features, MLP over
derived physics features) join into a 208-dimensional embedding, fuse
through a two-layer MLP, and split into a 3-class softmax head and a
sigmoid head predicting atmospheric transmittance for the Beer-Lambert
constraint.

Total trainable parameters: 202,228.

Lifted from notebooks/03a_pinn_training_final.ipynb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two-layer residual MLP block with batch norm, ReLU, and dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        self.proj = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + self.proj(x))


class CNNBranch(nn.Module):
    """Convolutional encoder for 4-channel 32x32 IR patches.

    Three Conv-BN-ReLU stacks with max-pool between, followed by global
    average pooling. Output dimensionality is 128.
    """

    def __init__(self, in_channels: int = 4, dropout: float = 0.2):
        super().__init__()
        # Attribute name must stay `enc` to match the trained checkpoint's
        # state_dict keys (cnn.enc.*).
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.gap(self.enc(x)).flatten(1))


class FireSightPINN(nn.Module):
    """Four-branch fusion PINN for wildfire classification.

    Inputs:
        patch:    (B, 4, 32, 32) brightness-temperature patches with channels
                  [BT_I4, BT_I5, BTD, fire_mask], normalised
        atm:      (B, n_atm)     ERA5 atmospheric features
        srf:      (B, n_srf)     surface features (MODIS land cover, OSM
                                 proximity, solar geometry)
        der:      (B, n_derived) derived physics features

    Outputs:
        logits:        (B, n_classes) raw classification logits
        transmittance: (B, 1) sigmoid output for atmospheric transmittance
    """

    def __init__(
        self,
        n_atm: int = 16,
        n_srf: int = 20,
        n_derived: int = 6,
        n_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn = CNNBranch(4, dropout)
        self.atm = nn.Sequential(
            ResidualBlock(n_atm, 64), ResidualBlock(64, 32)
        )
        self.srf = nn.Sequential(
            ResidualBlock(n_srf, 64), ResidualBlock(64, 32)
        )
        self.der = nn.Sequential(
            nn.Linear(n_derived, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(208, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(64, n_classes)
        self.trans = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        patch: torch.Tensor,
        atm: torch.Tensor,
        srf: torch.Tensor,
        der: torch.Tensor,
    ):
        joint = torch.cat(
            [self.cnn(patch), self.atm(atm), self.srf(srf), self.der(der)],
            dim=1,
        )
        feat = self.fusion(joint)
        return self.cls(feat), self.trans(feat)

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        map_location: str = "cpu",
        **kwargs,
    ) -> "FireSightPINN":
        """Load a trained checkpoint into a fresh model instance.

        kwargs are forwarded to the constructor (n_atm, n_srf, n_derived,
        n_classes, dropout). Defaults match the trained checkpoint.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**kwargs)
        state = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        model.load_state_dict(state)
        model.eval()
        return model

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
