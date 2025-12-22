from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py  # type: ignore
import lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


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
        self.input_size = input_size
        self.index: List[Tuple[str, str, str]] = []  # (ds_group, keyframe_group, frame)
        self._h5: h5py.File | None = None

        with h5py.File(self.h5_path, "r") as f:
            for ds in self.datasets:
                ds_group = f"dataset{ds}"
                for kf, fr in list_all_frames(f, ds_group):
                    self.index.append((ds_group, kf, fr))

        logger.info(
            f"DepthH5Dataset initialized from {self.h5_path} "
            f"with datasets {self.datasets} ({len(self.index)} frames)"
        )

        # Torchvision v2 transforms (mirrors eval/depth/scared_lance.py).
        # Keep a "with resize" and "no resize" path so pre-resized datasets
        # don't pay Resize cost in every __getitem__.
        image_tfms_post: List[Any] = [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        depth_tfms_post: List[Any] = [v2.ToDtype(torch.float32, scale=False)]

        if input_size is not None:
            self.image_transform_with_resize = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(
                        input_size,
                        interpolation=v2.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                    *image_tfms_post,
                ]
            )
            self.depth_transform_with_resize = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(
                        input_size,
                        interpolation=v2.InterpolationMode.NEAREST,
                        antialias=False,
                    ),
                    *depth_tfms_post,
                ]
            )
        else:
            self.image_transform_with_resize = None
            self.depth_transform_with_resize = None

        self.image_transform_no_resize = v2.Compose([v2.ToImage(), *image_tfms_post])
        self.depth_transform_no_resize = v2.Compose([v2.ToImage(), *depth_tfms_post])

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_h5"] = None
        return state

    def __del__(self) -> None:
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        ds_group, kf_group, fr = self.index[i]

        grp = self._get_h5()[ds_group][kf_group][fr]
        img_np = grp["image"][...]
        depth_np = grp["gt"][...]
        if depth_np.ndim == 3:
            depth_np = depth_np[..., 0]

        # Skip Resize when the frame is already resized to torchvision's
        # Resize(size=int) convention (shorter side == input_size).
        if self.input_size is not None:
            assert self.image_transform_with_resize is not None
            assert self.depth_transform_with_resize is not None
            do_resize = min(int(img_np.shape[0]), int(img_np.shape[1])) != int(self.input_size)
            if do_resize:
                image = self.image_transform_with_resize(img_np)  # FloatTensor[3, H, W]
                depth = self.depth_transform_with_resize(depth_np).squeeze(0)  # FloatTensor[H, W]
            else:
                image = self.image_transform_no_resize(img_np)  # FloatTensor[3, H, W]
                depth = self.depth_transform_no_resize(depth_np).squeeze(0)  # FloatTensor[H, W]
        else:
            image = self.image_transform_no_resize(img_np)  # FloatTensor[3, H, W]
            depth = self.depth_transform_no_resize(depth_np).squeeze(0)  # FloatTensor[H, W]

        # depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        valid_mask = (depth > 0.0001) & (depth < 150.0)
        # print("depth")
        # print(depth.min())
        # print(depth.max())
        # print("valid_mask")
        # print(valid_mask.sum())

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
            logger.info(
                f"DepthDataModule.setup(stage={stage}): building train/val datasets"
            )
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
