"""Data module for SCARED depth estimation using WebDataset.

Each sample in the shards is expected to contain the following keys:

* ``"png"``  - RGB image bytes (PNG)
* ``"depth"`` - depth / scene-points bytes (``.npy`` preferred, otherwise
  ``.tiff``)
* ``"json"``  - JSON sidecar with metadata (not required by this module)

The SCARED WebDataset shards can be created with ``utility/create_scared_wds.py``.

This module exposes a ``ScaredDepthDataModule`` that returns batches with the
following structure used by ``DinoDPTDepthModule``:

    {
        "image": FloatTensor[B, 3, H, W],  # normalized RGB
        "depth": FloatTensor[B, H, W],     # depth map in arbitrary units
        "valid_mask": BoolTensor[B, H, W]  # optional, True where depth > 0
    }
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader

import lightning as pl

import webdataset as wds
from torchvision.transforms import v2 as T


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _decode_depth_bytes(depth_bytes: bytes) -> np.ndarray:
    """Decode depth from raw bytes.

    Currently we assume the depth is stored as a NumPy ``.npy`` array and
    load it directly from a bytes buffer.
    """

    return np.load(io.BytesIO(depth_bytes))



def _make_image_transform(image_size: int) -> T.Compose:
    """Return RGB image transform using torchvision v2 transforms.

    ``image_size`` is the shorter side passed to ``T.Resize``; aspect ratio
    is preserved.
    """

    size = int(image_size)
    return T.Compose(
        [
            T.ToImage(),
            T.Resize(size),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def _make_depth_transform(image_size: int) -> T.Compose:
    """Return depth transform for raw depth maps.

    Uses nearest-neighbor resize on a single-channel tensor and preserves the
    original depth scale (no normalization or rescaling).
    """

    size = int(image_size)
    return T.Resize(size, interpolation=T.InterpolationMode.NEAREST)
class ScaredDepthDataModule(pl.LightningDataModule):
    """LightningDataModule for depth supervision on SCARED WebDataset shards.

    This module assumes an iterable-style WebDataset (``resampled=True``), so
    the training loop should be driven by ``max_steps`` rather than
    ``max_epochs``.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg

        self.train_image_transform = _make_image_transform(cfg.image_size)
        self.val_image_transform = _make_image_transform(cfg.image_size)
        self.train_depth_transform = _make_depth_transform(cfg.image_size)
        self.val_depth_transform = _make_depth_transform(cfg.image_size)

        self._train_ds = None
        self._val_ds = None

        logger.info(
            "ScaredDepthDataModule initialized (train_shards=%s, val_shards=%s, image_size=%s)",
            self.cfg.train_shards,
            self.cfg.val_shards,
            self.cfg.image_size,
        )

    def _build_dataset(self, shards_pattern: str, *, is_train: bool) -> wds.WebDataset:
        logger.info("Building %s WebDataset from pattern: %s", "train" if is_train else "val", shards_pattern)
        img_transform = self.train_image_transform if is_train else self.val_image_transform
        depth_transform = self.train_depth_transform if is_train else self.val_depth_transform

        def _decode_depth(sample: Dict[str, Any]) -> Dict[str, Any]:
            sample["depth"] = _decode_depth_bytes(sample["depth"])
            return sample

        def _to_tensors(sample: Dict[str, Any]) -> Dict[str, Any]:
            # RGB image (v2 pipeline)
            img_t: Tensor = img_transform(sample["png"])

            # Depth map: keep as raw numeric values, only resize spatially
            depth_arr = np.asarray(sample["depth"])
            if depth_arr.ndim == 3:
                # If 3 channels, assume last channel stores depth (Z)
                depth_arr = depth_arr[..., -1]

            depth_t = torch.from_numpy(depth_arr).float().unsqueeze(0)  # [1, H, W]
            depth_t = depth_transform(depth_t).squeeze(0)  # [H, W]

            valid_mask = (depth_t >= self.cfg.min_depth) & (depth_t <= self.cfg.max_depth)

            return {
                "image": img_t,
                "depth": depth_t,
                "valid_mask": valid_mask,
            }

        dataset = (
            wds.WebDataset(shards_pattern, shardshuffle=is_train, resampled=True)
            .decode("pil")
            .map(_decode_depth)
            .map(_to_tensors)
        )

        # Shuffle per-epoch for training; light shuffle for validation
        if is_train:
            dataset = dataset.shuffle(self.cfg.shuffle_buffer)

        return dataset

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        if stage in (None, "fit"):
            if self._train_ds is None:
                self._train_ds = self._build_dataset(self.cfg.train_shards, is_train=True)
            if self._val_ds is None:
                self._val_ds = self._build_dataset(self.cfg.val_shards, is_train=False)

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._train_ds is not None, "DataModule.setup('fit') must be called before train_dataloader()"
        return DataLoader(
            self._train_ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._val_ds is not None, "DataModule.setup('fit') must be called before val_dataloader()"
        return DataLoader(
            self._val_ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
