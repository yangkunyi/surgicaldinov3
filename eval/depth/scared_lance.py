from __future__ import annotations

from typing import Any, Dict, Sequence

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import lance
import torch

import cv2
import lightning as pl


class LanceMapDataset(Dataset):
    """Map-style Dataset for SCARED Lance tables."""

    def __init__(
        self,
        lance_path: str,
        *,
        allowed_ids: Sequence[int | str] = (),
        image_size: int,
        min_depth: float,
        max_depth: float,
        columns: Sequence[str] = ("image", "depth"),
    ) -> None:
        super().__init__()

        self.lance_path = lance_path
        self.ds = lance.dataset(self.lance_path)

        self.allowed_ids = [str(v) for v in allowed_ids]
        self.allowed_ids = [
            v if v.startswith("dataset") else f"dataset{v}" for v in self.allowed_ids
        ]

        total_rows = self.ds.count_rows()
        if self.allowed_ids:
            full_table = self.ds.to_table(columns=["dataset_id"])
            df = full_table.to_pandas()
            self.global_indices = df[
                df["dataset_id"].isin(self.allowed_ids)
            ].index.values.tolist()
        else:
            self.global_indices = list(range(total_rows))

        self.image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(image_size, interpolation=v2.InterpolationMode.BILINEAR),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.depth_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(image_size, interpolation=v2.InterpolationMode.NEAREST),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.columns = list(columns)

    def __len__(self) -> int:
        return len(self.global_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_row_idx = self.global_indices[idx]
        table = self.ds.take([real_row_idx], columns=self.columns)

        image_np = table["image"][0].to_numpy().copy()
        depth_np = table["depth"][0].to_numpy().copy()

        image = self.image_transform(image_np)
        depth = self.depth_transform(depth_np).squeeze(0)
        valid_mask = (depth >= self.min_depth) & (depth <= self.max_depth)
        return {
            "image": image,
            "depth": depth,
            "valid_mask": valid_mask,
        }


class ScaredLanceDataModule(pl.LightningDataModule):
    """Lightning DataModule for SCARED depth stored in a Lance table."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg

        self.train_ds = LanceMapDataset(
            cfg.lance_path,
            allowed_ids=cfg.train_datasets,
            image_size=cfg.image_size,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            columns=cfg.columns,
        )
        self.val_ds = LanceMapDataset(
            cfg.lance_path,
            allowed_ids=cfg.val_datasets,
            image_size=cfg.image_size,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            columns=cfg.columns,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
