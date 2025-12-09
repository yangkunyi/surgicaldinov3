from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import h5py  # type: ignore
import numpy as np
import lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from .utils.transform import NormalizeImage, PrepareForNet, Resize


DEFAULT_H5 = "/bd_byta6000i0/users/surgical_depth/SCARED_fixed/scared.hdf5"


def _ensure_group(h5: h5py.File, path: str) -> h5py.Group:
    if path not in h5:
        raise KeyError(f"Group '{path}' not found in HDF5 file.")
    return h5[path]  # type: ignore[return-value]


def _sorted_keys(g: h5py.Group) -> List[str]:
    keys = sorted(list(g.keys()))
    if not keys:
        raise RuntimeError(f"Group '{g.name}' has no children.")
    return keys


@logger.catch(onerror=lambda _: sys.exit(1))
def list_all_frames(h5: h5py.File, dataset_group: str) -> List[Tuple[str, str]]:
    """List all (keyframe_group, frame_id) pairs within a dataset group.

    This scans every immediate child group under the dataset (e.g., keyframe1, keyframe2, ...)
    and returns the union of all frames.
    """
    g = _ensure_group(h5, dataset_group)
    pairs: List[Tuple[str, str]] = []
    for kf in _sorted_keys(g):
        sub = g[kf]
        if not isinstance(sub, h5py.Group):
            continue
        for fr in _sorted_keys(sub):
            pairs.append((kf, fr))
    if not pairs:
        raise RuntimeError(f"No frames found under '{dataset_group}'.")
    return pairs


class DepthH5Dataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path = DEFAULT_H5,
        datasets: Sequence[int] = (8, 9),
        input_size: int | None = 512,
    ) -> None:
        super().__init__()
        self.h5_path = str(h5_path)
        self.datasets = list(datasets)
        self.index: List[Tuple[str, str, str]] = []  # (ds_group, keyframe_group, frame)

        with h5py.File(self.h5_path, "r") as f:
            for ds in self.datasets:
                ds_group = f"dataset{ds}"
                for kf, fr in list_all_frames(f, ds_group):
                    self.index.append((ds_group, kf, fr))

        logger.info(
            "DepthH5Dataset initialized from %s with datasets %s (%d frames)",
            self.h5_path,
            self.datasets,
            len(self.index),
        )

        # Build transforms; if input_size is None, skip resizing
        if input_size is None:
            self.transform = Compose(
                [
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                ]
            )
        else:
            self.transform = Compose(
                [
                    Resize(
                        width=input_size,
                        height=input_size,
                        resize_target=True,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=16,
                        resize_method="lower_bound",
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                ]
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        ds_group, kf_group, fr = self.index[i]
        with h5py.File(self.h5_path, "r") as f:
            grp = f[ds_group][kf_group][fr]
            if "image" not in grp or "gt" not in grp:
                raise KeyError(f"Expected 'image' and 'gt' datasets at '{ds_group}/{kf_group}/{fr}'.")
            img_np = grp["image"][...]
            depth_np = grp["gt"][...]

        if img_np.ndim != 3 or img_np.shape[-1] != 3:
            raise ValueError(f"Unexpected image shape {img_np.shape} at '{ds_group}/{kf_group}/{fr}'.")
        if depth_np.ndim not in (2, 3):
            raise ValueError(f"Unexpected depth shape {depth_np.shape} at '{ds_group}/{kf_group}/{fr}'.")
        if depth_np.ndim == 3:
            depth_np = depth_np[..., 0]

        sample = {"image": img_np.astype(np.float32) / 255.0, "depth": depth_np.astype(np.float32)}
        sample = self.transform(sample)

        image = torch.from_numpy(sample["image"])  # CHW float32
        depth = torch.from_numpy(sample["depth"])  # HW float32
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        valid_mask = depth > 0

        return {
            "image": image,
            "depth": depth,
            "valid_mask": valid_mask,
            "id": f"{ds_group}/{kf_group}/{fr}",
        }


class DepthDataModule(pl.LightningDataModule):
    """DataModule wrapper around the HDF5 SCARED dataset.

    Expects a Hydra-style ``cfg`` where ``cfg.data`` contains HDF5-related
    fields (e.g. ``h5_path``, ``train_datasets``, ``val_datasets``,
    ``input_size``, ``batch_size``, ``num_workers``).
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.train_ds: DepthH5Dataset | None = None
        self.val_ds: DepthH5Dataset | None = None

    @logger.catch(onerror=lambda _: sys.exit(1))
    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            logger.info("DepthDataModule.setup(stage=%s): building train/val datasets", stage)
            self.train_ds = DepthH5Dataset(
                h5_path=self.cfg.data.h5_path,
                datasets=self.cfg.data.train_datasets,
                input_size=self.cfg.data.input_size,
            )
            self.val_ds = DepthH5Dataset(
                h5_path=self.cfg.data.h5_path,
                datasets=self.cfg.data.val_datasets,
                input_size=self.cfg.data.input_size,
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
        )
