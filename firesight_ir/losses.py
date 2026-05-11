"""
firesight_ir.losses
===================

Composite physics-informed loss used to train FireSight-IR.

    L_total = L_CE
            + lambda_BL * L_BeerLambert
            + lambda_DR * L_DynamicRange
            + lambda_TH * L_ThermalRealism

where:

    L_CE             weighted cross-entropy across {Non-fire, Wildfire,
                     False-alarm}, with class weights inversely proportional
                     to class frequency
    L_BeerLambert    MSE between the model's transmittance head output and
                     the proxy target tau ~ exp(-0.05 * TCWV)
    L_DynamicRange   penalises wildfire predictions on cold pixels
                     (BT_I4 < 310 K)
    L_ThermalRealism penalises wildfire predictions on pixels with low
                     MWIR/LWIR contrast (BTD < 10 K)

Default lambdas: BL=0.10, DR=0.05, TH=0.05.

Lifted from notebooks/03a_pinn_training_final.ipynb.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# Class weights used during training. The non-fire and false-alarm classes
# are upweighted relative to the dominant wildfire class. False-alarm is
# penalised about 27 times more than wildfire to prevent the model from
# ignoring the rare class.
DEFAULT_CLASS_WEIGHTS: Tuple[float, float, float] = (6.90, 0.36, 9.81)

LAMBDA_BL: float = 0.10
LAMBDA_DR: float = 0.05
LAMBDA_TH: float = 0.05

# Physics-constraint thresholds (Kelvin)
BTD_THERMAL_REALISM_K: float = 10.0
BT_I4_DYNAMIC_RANGE_K: float = 310.0

# TCWV coefficient in the Beer-Lambert proxy: tau ~ exp(-0.05 * TCWV)
TCWV_BL_COEFFICIENT: float = 0.05


class PINNLoss(nn.Module):
    """Composite physics-informed loss for FireSight-IR.

    Parameters
    ----------
    weights : sequence of three floats
        Per-class cross-entropy weights for (Non-fire, Wildfire, False-alarm).
    lambda_bl, lambda_dr, lambda_th : floats
        Weights on the three physics regularisation terms.
    bt_min : float
        Minimum BT_I4 (Kelvin) for a wildfire prediction to be unpenalised.
    btd_min : float
        Minimum BTD (Kelvin) for a wildfire prediction to be unpenalised.
    device : str
        Device on which to allocate the class-weight tensor.

    Forward inputs
    --------------
    logits : (B, 3) raw classification logits
    transmittance : (B, 1) sigmoid output of the physics head
    labels : (B,) integer class indices
    aux : (B, 3) auxiliary tensor with columns
              [BT_I4_center, BTD_center, transmittance_target]

    Returns
    -------
    total_loss : torch.Tensor (scalar)
    components : dict mapping component name to scalar value
                 ('ce', 'bl', 'dr', 'th')
    """

    def __init__(
        self,
        weights: Sequence[float] = DEFAULT_CLASS_WEIGHTS,
        lambda_bl: float = LAMBDA_BL,
        lambda_dr: float = LAMBDA_DR,
        lambda_th: float = LAMBDA_TH,
        bt_min: float = BT_I4_DYNAMIC_RANGE_K,
        btd_min: float = BTD_THERMAL_REALISM_K,
        device: str = "cpu",
    ):
        super().__init__()
        self.lambda_bl = lambda_bl
        self.lambda_dr = lambda_dr
        self.lambda_th = lambda_th
        self.bt_min = bt_min
        self.btd_min = btd_min
        w = torch.tensor(weights, dtype=torch.float32).to(device)
        self.ce = nn.CrossEntropyLoss(weight=w)
        self.mse = nn.MSELoss()

    def forward(
        self,
        logits: torch.Tensor,
        transmittance: torch.Tensor,
        labels: torch.Tensor,
        aux: torch.Tensor,
    ):
        L_ce = self.ce(logits, labels)
        L_bl = self.mse(transmittance, aux[:, 2:3])

        p_wf = F.softmax(logits, dim=1)[:, 1]
        L_dr = (p_wf * (aux[:, 0] < self.bt_min).float()).mean()
        L_th = (p_wf * (aux[:, 1] < self.btd_min).float()).mean()

        total = (
            L_ce
            + self.lambda_bl * L_bl
            + self.lambda_dr * L_dr
            + self.lambda_th * L_th
        )
        components = {
            "ce": L_ce.item(),
            "bl": L_bl.item(),
            "dr": L_dr.item(),
            "th": L_th.item(),
        }
        return total, components
