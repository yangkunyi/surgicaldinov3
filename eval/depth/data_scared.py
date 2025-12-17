"""Data module for SCARED depth estimation using WebDataset.
Refactored to use per-sample transformation via .map() instead of .batched(collate_fn).
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, default_collate

import lightning as pl

import webdataset as wds
from torchvision.transforms import v2

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class SampleTransformer:
    """处理单个样本的 Transform 类。
    用于 WebDataset 的 .map() 管道中。
    """

    def __init__(self, cfg, is_train: bool):
        self.cfg = cfg
        self.is_train = is_train

        # --- Image Transform ---
        self.img_transform = v2.Compose(
            [
                # 注意：这里输入已经是 Tensor (C, H, W)
                v2.Resize(
                    size=self.cfg.image_size,
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

        # --- Depth Transform ---
        self.depth_transform = v2.Compose(
            [
                v2.Resize(
                    size=self.cfg.image_size,
                    interpolation=v2.InterpolationMode.NEAREST,
                    antialias=False,
                ),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入: 从 WebDataset 解码后的单个样本字典
        {'png': ndarray (H, W, 3), 'depth': bytes, ...}
        """

        # 1. 处理图像
        # wds.decode("rgb8") 产生的是 (H, W, 3) 的 uint8 numpy array
        # 我们需要先转换为 Tensor 并 permute 到 (C, H, W)
        image = torch.from_numpy(sample["png"]).permute(2, 0, 1)  # (3, H, W)
        image = self.img_transform(image)

        # 2. 处理深度
        # 深度数据通常存储为 bytes (npy 格式)，需要手动加载

        depth_bytes = sample["depth"]
        with io.BytesIO(depth_bytes) as f:
            depth_np = np.load(f)  # (H, W)

        # 转换为 Tensor 并增加 Channel 维度以便 Resize: (1, H, W)
        depth = torch.from_numpy(depth_np).unsqueeze(0)
        depth = self.depth_transform(depth)

        # Resize 后去掉 Channel 维度: (H, W)
        depth = depth.squeeze(0)


        # 3. 生成 Mask
        valid_mask = (depth > self.cfg.min_depth) & (depth < self.cfg.max_depth)

        # 返回处理后的字典，只包含需要的字段
        return {
            "image": image,  # FloatTensor[3, H, W]
            "depth": depth,  # FloatTensor[H, W]
            "valid_mask": valid_mask,  # BoolTensor[H, W]
            "__key__": sample.get("__key__", ""),  # 保留 key 用于 debug
        }


class ScaredDepthDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self._train_ds = None
        self._val_ds = None

        logger.info(
            f"ScaredDepthDataModule initialized (train_shards={self.cfg.train_shards}, "
            f"val_shards={self.cfg.val_shards}, image_size={self.cfg.image_size})"
        )

    def _build_dataset(self, shards_pattern: str, *, is_train: bool) -> wds.WebDataset:
        phase = "train" if is_train else "val"
        logger.info(f"Building {phase} WebDataset from pattern: {shards_pattern}")

        # 实例化 Transform
        transform_fn = SampleTransformer(self.cfg, is_train=is_train)

        dataset = (
            wds.WebDataset(shards_pattern, shardshuffle=42 if is_train else 0)
            .shuffle(1000)
            .decode("rgb8")  # 自动解码 'png' -> numpy array, 'depth' (bytes) 保持原样
            .map(transform_fn)  # <--- 关键更改：在此处并行处理单个样本
            .batched(
                self.cfg.batch_size,
                partial=False,
                collation_fn=default_collate,  # <--- 关键更改：使用默认 collate 简单堆叠
            )
        )

        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if self._train_ds is None:
                self._train_ds = self._build_dataset(
                    self.cfg.train_shards, is_train=True
                )
            if self._val_ds is None:
                self._val_ds = self._build_dataset(self.cfg.val_shards, is_train=False)

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=None,  # WebDataset 已经处理了 batching
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            prefetch_factor=4,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=None,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
