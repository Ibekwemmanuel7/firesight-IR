"""
firesight_ir.data
=================

PyTorch Dataset for FireSight-IR. Reads pre-cached numpy memmaps for
fast iteration on Colab. The memmap caches are produced by the feature
engineering notebook from the master HDF5 archive.

Lifted from notebooks/03a_pinn_training_final.ipynb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Union

import numpy as np
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]


class FireSightDataset(Dataset):
    """Dataset over the pre-cached FireSight-IR memmaps.

    Parameters
    ----------
    cache_files : Mapping[str, PathLike]
        Dictionary with keys 'patches', 'atm', 'srf', 'derived', 'labels',
        'aux' pointing at .npy memmap files. Shapes:
            patches : (N, 4, 32, 32)  float32
            atm     : (N, n_atm)       float32
            srf     : (N, n_srf)       float32
            derived : (N, n_derived)   float32
            labels  : (N,)             uint8   (0=non-fire, 1=wildfire, 2=FA)
            aux     : (N, 3)           float32 [BT_I4_center, BTD_center,
                                                transmittance_target]
    indices : array-like
        Sample indices to expose through __getitem__.
    augment : bool
        If True, apply random rotations (90/180/270 degrees) and horizontal
        flips to patches. Used during training only.
    zero_atm, zero_srf, zero_derived : bool
        Ablation flags. If set, the corresponding feature vector is replaced
        with zeros at __getitem__ time. Used by the ablation study to test
        each branch's contribution without retraining.

    Returns from __getitem__
    -------------------------
    tuple of tensors:
        (patch, atm, srf, derived, label, aux)

    where patch is (4, 32, 32), atm/srf/derived are 1D feature vectors,
    label is a scalar long tensor, and aux is a 3-element float tensor.
    """

    def __init__(
        self,
        cache_files: Mapping[str, PathLike],
        indices,
        augment: bool = False,
        zero_atm: bool = False,
        zero_srf: bool = False,
        zero_derived: bool = False,
    ):
        self.indices = np.sort(np.asarray(indices))
        self.augment = augment
        self.zero_atm = zero_atm
        self.zero_srf = zero_srf
        self.zero_derived = zero_derived

        self.patches = np.load(cache_files["patches"], mmap_mode="r")
        self.atm = np.load(cache_files["atm"], mmap_mode="r")
        self.srf = np.load(cache_files["srf"], mmap_mode="r")
        self.derived = np.load(cache_files["derived"], mmap_mode="r")
        self.labels = np.load(cache_files["labels"], mmap_mode="r")
        self.aux = np.load(cache_files["aux"], mmap_mode="r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        patch = self.patches[idx].copy()

        if self.augment:
            k = np.random.randint(0, 4)
            if k:
                patch = np.rot90(patch, k, axes=(1, 2)).copy()
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=2).copy()

        atm = (
            np.zeros_like(self.atm[idx])
            if self.zero_atm
            else self.atm[idx].copy()
        )
        srf = (
            np.zeros_like(self.srf[idx])
            if self.zero_srf
            else self.srf[idx].copy()
        )
        der = (
            np.zeros_like(self.derived[idx])
            if self.zero_derived
            else self.derived[idx].copy()
        )

        return (
            torch.from_numpy(patch),
            torch.from_numpy(atm),
            torch.from_numpy(srf),
            torch.from_numpy(der),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
            torch.from_numpy(self.aux[idx].copy()),
        )
