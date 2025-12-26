from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from colormap import IMAGENET_MEAN, IMAGENET_STD, color_mask_to_class_index


class CholecSeg8kDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: Sequence[tuple[Path, Path, str]],
        transforms: Any,
        *,
        debug_unknown_masks_dir: str,
    ) -> None:
        super().__init__()
        self.samples = list(samples)
        self.transforms = transforms
        self.debug_unknown_masks_dir = debug_unknown_masks_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_path, mask_path, video_id = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask_img = Image.open(mask_path)
        debug_save_path = ""
        if len(self.debug_unknown_masks_dir) > 0:
            debug_dir = Path(self.debug_unknown_masks_dir)
            debug_save_path = str(
                debug_dir / video_id / mask_path.parent.name / mask_path.name
            )
        mask_arr = color_mask_to_class_index(mask_img, debug_save_path=debug_save_path)
        mask = tv_tensors.Mask(torch.from_numpy(mask_arr).to(torch.int64))
        image, mask = self.transforms(image, mask)
        return {
            "image": image,
            "mask": mask,
            "video_id": video_id,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def _build_train_transforms(cfg: Any) -> Any:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomShortestSize(
                min_size=cfg.train.min_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            v2.RandomCrop(size=(cfg.train.crop_size, cfg.train.crop_size)),
            v2.RandomHorizontalFlip(p=cfg.train.flip_prob),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _build_val_transforms(cfg: Any) -> Any:
    transforms = [
        v2.ToImage(),
        v2.Resize(
            size=cfg.val.short_side,
            interpolation=InterpolationMode.BILINEAR,
            max_size=cfg.val.max_size,
            antialias=True,
        ),
    ]
    if cfg.val.center_crop:
        transforms.append(v2.CenterCrop(size=(cfg.val.crop_size, cfg.val.crop_size)))
    transforms.extend(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return v2.Compose(transforms)


def _collect_video_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _build_samples_for_videos(
    video_dirs: Sequence[Path],
) -> list[tuple[Path, Path, str]]:
    samples: list[tuple[Path, Path, str]] = []
    for video_dir in video_dirs:
        video_id = video_dir.name
        image_paths = sorted(video_dir.rglob("*_endo.png"))
        for image_path in image_paths:
            mask_name = image_path.name.replace("_endo.png", "_endo_color_mask.png")
            mask_path = image_path.with_name(mask_name)
            samples.append((image_path, mask_path, video_id))
    return samples


class CholecSeg8kDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_transforms = _build_train_transforms(cfg)
        self.val_transforms = _build_val_transforms(cfg)
        self.train_dataset: CholecSeg8kDataset
        self.val_dataset: CholecSeg8kDataset

    def setup(self, stage: str | None = None) -> None:
        root = Path(self.cfg.root)
        video_dirs = _collect_video_dirs(root)
        video_names = [p.name for p in video_dirs]
        rng = random.Random(self.cfg.split_seed)
        rng.shuffle(video_names)

        if len(self.cfg.val_videos) > 0:
            val_video_names = list(self.cfg.val_videos)
        else:
            val_count = int(len(video_names) * float(self.cfg.val_ratio))
            val_video_names = video_names[:val_count]

        val_video_set = set(val_video_names)
        train_video_dirs = [
            root / name for name in video_names if name not in val_video_set
        ]
        val_video_dirs = [root / name for name in video_names if name in val_video_set]

        train_samples = _build_samples_for_videos(train_video_dirs)
        val_samples = _build_samples_for_videos(val_video_dirs)

        self.train_dataset = CholecSeg8kDataset(
            train_samples,
            transforms=self.train_transforms,
            debug_unknown_masks_dir=self.cfg.debug_unknown_masks_dir,
        )
        self.val_dataset = CholecSeg8kDataset(
            val_samples,
            transforms=self.val_transforms,
            debug_unknown_masks_dir=self.cfg.debug_unknown_masks_dir,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
            drop_last=self.cfg.drop_last,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
            drop_last=False,
        )
