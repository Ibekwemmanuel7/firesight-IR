"""
FireSight-IR
============

A physics-informed neural network for wildfire detection and false-alarm
rejection on satellite multispectral infrared data.

Quick start
-----------

    >>> from firesight_ir import FireSightPINN, predict
    >>> model = FireSightPINN.from_checkpoint('models/firesight_pinn_best.pt')
    >>> preds, probs, transmittance = predict(
    ...     model, patches, atm, srf, derived
    ... )

See the project README for the expected input format and pre-processing.
"""

from .data import FireSightDataset
from .inference import CLASS_NAMES, predict
from .losses import (
    BT_I4_DYNAMIC_RANGE_K,
    BTD_THERMAL_REALISM_K,
    DEFAULT_CLASS_WEIGHTS,
    LAMBDA_BL,
    LAMBDA_DR,
    LAMBDA_TH,
    PINNLoss,
    TCWV_BL_COEFFICIENT,
)
from .model import CNNBranch, FireSightPINN, ResidualBlock

__version__ = "0.1.0"
__author__ = "Emmanuel Ibekwe"

__all__ = [
    # Model
    "FireSightPINN",
    "ResidualBlock",
    "CNNBranch",
    # Dataset
    "FireSightDataset",
    # Loss
    "PINNLoss",
    "DEFAULT_CLASS_WEIGHTS",
    "LAMBDA_BL",
    "LAMBDA_DR",
    "LAMBDA_TH",
    "BTD_THERMAL_REALISM_K",
    "BT_I4_DYNAMIC_RANGE_K",
    "TCWV_BL_COEFFICIENT",
    # Inference
    "predict",
    "CLASS_NAMES",
]
