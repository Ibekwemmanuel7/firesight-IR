"""
firesight_ir.inference
======================

Convenience wrappers for running the trained FireSight-IR model on new
data without going through the full notebook pipeline.

Public surface:

    from firesight_ir import FireSightPINN, predict
    model = FireSightPINN.from_checkpoint('models/firesight_pinn_best.pt')
    preds, probs, transmittance = predict(model, patches, atm, srf, derived)

All inputs may be torch tensors or numpy arrays. Outputs are numpy arrays.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch


# Class index meanings, matching the trained checkpoint.
CLASS_NAMES: Tuple[str, str, str] = ("Non-fire", "Wildfire", "False-alarm")


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_tensor(x: ArrayLike, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    patches: ArrayLike,
    atm: ArrayLike,
    srf: ArrayLike,
    derived: ArrayLike,
    batch_size: int = 1024,
    device: Union[str, torch.device, None] = None,
):
    """Run batched inference and return predictions, probabilities, and
    predicted transmittance.

    Parameters
    ----------
    model : torch.nn.Module
        A FireSightPINN instance, typically from FireSightPINN.from_checkpoint.
    patches : array-like, shape (N, 4, 32, 32)
        Brightness-temperature patches, normalised the same way as in
        training (BT_I4 zero-centred at 300 K and scaled by 50, BT_I5
        zero-centred at 290 K and scaled by 20, BTD scaled by 40, fire
        mask scaled by 9).
    atm : array-like, shape (N, n_atm)
        ERA5 atmospheric features, scaled to match training.
    srf : array-like, shape (N, n_srf)
        Surface features (MODIS land cover, OSM proximity, solar geometry).
    derived : array-like, shape (N, n_derived)
        Derived physics features.
    batch_size : int
        Inference batch size.
    device : str or torch.device, optional
        Device to run inference on. If None, inferred from model.

    Returns
    -------
    preds : numpy.ndarray, shape (N,) int64
        Argmax class index in {0, 1, 2} mapping to CLASS_NAMES.
    probs : numpy.ndarray, shape (N, 3) float32
        Softmax probabilities over the three classes.
    transmittance : numpy.ndarray, shape (N,) float32
        Predicted atmospheric transmittance from the physics head.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    patches_t = _to_tensor(patches)
    atm_t = _to_tensor(atm)
    srf_t = _to_tensor(srf)
    derived_t = _to_tensor(derived)

    n = len(patches_t)
    if not (len(atm_t) == len(srf_t) == len(derived_t) == n):
        raise ValueError(
            f"Inputs disagree on N: patches={len(patches_t)}, atm={len(atm_t)}, "
            f"srf={len(srf_t)}, derived={len(derived_t)}"
        )

    logits_chunks = []
    trans_chunks = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        logits, trans = model(
            patches_t[start:end].to(device),
            atm_t[start:end].to(device),
            srf_t[start:end].to(device),
            derived_t[start:end].to(device),
        )
        logits_chunks.append(logits.cpu())
        trans_chunks.append(trans.cpu())

    logits = torch.cat(logits_chunks)
    trans = torch.cat(trans_chunks).squeeze(-1)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    return preds.numpy(), probs.numpy(), trans.numpy()
