from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import lightning as pl
import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


@dataclass(frozen=True)
class MegainfoRecord:
    index: str
    dataset_number: int
    keyframe_number: int
    frame_id: str
    image_dtype: str | None = None
    image_shape: tuple[int, ...] | None = None
    depth_dtype: str | None = None
    depth_shape: tuple[int, ...] | None = None


def _parse_shape(value: object) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(x) for x in value)
    if isinstance(value, list):
        return tuple(int(x) for x in value)
    # CSV stores JSON string like "[1024,1280,3]".
    s = str(value)
    return tuple(int(x) for x in json.loads(s))


def _normalize_allowed_dataset_numbers(allowed: Sequence[int | str]) -> set[int]:
    out: set[int] = set()
    for v in allowed:
        if isinstance(v, int):
            out.add(v)
            continue
        s = str(v)
        if s.startswith("dataset"):
            s = s[len("dataset") :]
        out.add(int(s))
    return out


def _read_megainfo(path: str | Path) -> List[MegainfoRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Megainfo file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records: List[MegainfoRecord] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(
                    MegainfoRecord(
                        index=str(obj["index"]),
                        dataset_number=int(obj["dataset_number"]),
                        keyframe_number=int(obj["keyframe_number"]),
                        frame_id=str(obj.get("frame_id", "")),
                        image_dtype=(str(obj["image_dtype"]) if "image_dtype" in obj else None),
                        image_shape=(
                            _parse_shape(obj["image_shape"]) if "image_shape" in obj else None
                        ),
                        depth_dtype=(str(obj["depth_dtype"]) if "depth_dtype" in obj else None),
                        depth_shape=(
                            _parse_shape(obj["depth_shape"]) if "depth_shape" in obj else None
                        ),
                    )
                )
        return records

    # Default: CSV
    records = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"index", "dataset_number", "keyframe_number"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Megainfo CSV missing required columns {sorted(required)}; got {reader.fieldnames}"
            )
        for row in reader:
            records.append(
                MegainfoRecord(
                    index=str(row["index"]),
                    dataset_number=int(row["dataset_number"]),
                    keyframe_number=int(row["keyframe_number"]),
                    frame_id=str(row.get("frame_id", "")),
                    image_dtype=(row.get("image_dtype") or None),
                    image_shape=(
                        _parse_shape(row["image_shape"]) if row.get("image_shape") else None
                    ),
                    depth_dtype=(row.get("depth_dtype") or None),
                    depth_shape=(
                        _parse_shape(row["depth_shape"]) if row.get("depth_shape") else None
                    ),
                )
            )
    return records


class LmdbMapDataset(Dataset):
    """Map-style Dataset for SCARED stored in a single LMDB.

    Assumes LMDB keys:
      {index}-image
      {index}-depth
    where `index` is the zero-padded string from the megainfo file.

    Supports two encodings:
    - Raw bytes (`np.ndarray.tobytes`) + dtype/shape stored in megainfo.
    - Legacy `.npy` bytes (if dtype/shape not present in megainfo).
    """

    def __init__(
        self,
        lmdb_path: str,
        *,
        megainfo_path: str,
        allowed_datasets: Sequence[int | str] = (),
        image_size: int,
        min_depth: float,
        max_depth: float,
        readahead: bool = False,
        meminit: bool = False,
        buffers: bool = True,
    ) -> None:
        super().__init__()
        self.lmdb_path = str(lmdb_path)
        self.megainfo_path = str(megainfo_path)
        self.allowed_dataset_numbers = _normalize_allowed_dataset_numbers(allowed_datasets)

        self._env: lmdb.Environment | None = None

        all_records = _read_megainfo(self.megainfo_path)
        if self.allowed_dataset_numbers:
            self.records = [
                r for r in all_records if r.dataset_number in self.allowed_dataset_numbers
            ]
        else:
            self.records = all_records

        if not self.records:
            raise RuntimeError(
                "No records selected from megainfo. "
                f"allowed_datasets={sorted(self.allowed_dataset_numbers) if self.allowed_dataset_numbers else 'ALL'}"
            )

        self._lmdb_open_opts = {
            "subdir": False,
            "readonly": True,
            "lock": False,
            "readahead": bool(readahead),
            "meminit": bool(meminit),
            "max_dbs": 1,
        }

        # If True, txn.get returns a buffer object (zero-copy) instead of bytes.
        # Safe here because we keep a persistent read txn per worker process.
        self._lmdb_buffers = bool(buffers)

        self.image_size = int(image_size)

        # When the dataset has already been resized offline, we want to avoid
        # paying the Resize cost in every __getitem__. We keep two transform
        # pipelines and select based on the sample's current spatial size.
        self.image_transform_with_resize = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(image_size, interpolation=v2.InterpolationMode.BILINEAR),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.image_transform_no_resize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.depth_transform_with_resize = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(image_size, interpolation=v2.InterpolationMode.NEAREST),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )
        self.depth_transform_no_resize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )

        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_env"] = None
        return state

    def __del__(self) -> None:
        try:
            if self._env is not None:
                self._env.close()
        except Exception:
            pass

    def _get_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, **self._lmdb_open_opts)
        return self._env

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load_npy_bytes(buf: bytes) -> np.ndarray:
        return np.load(io.BytesIO(buf), allow_pickle=False)

    @staticmethod
    def _load_raw_bytes(buf: bytes, *, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
        # Zero-copy view over `buf` (typically a `bytes` object returned by lmdb).
        # Note: the resulting array may be read-only if `buf` is immutable.
        return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        index_str = rec.index
        k_img = f"{index_str}-image".encode("utf-8")
        k_depth = f"{index_str}-depth".encode("utf-8")

        env = self._get_env()
        with env.begin(write=False, buffers=self._lmdb_buffers) as txn:
            v_img = txn.get(k_img)
            v_depth = txn.get(k_depth)

        if v_img is None or v_depth is None:
            raise KeyError(
                f"Missing LMDB keys for index={index_str} (image={v_img is not None}, depth={v_depth is not None})"
            )

        if rec.image_dtype and rec.image_shape:
            image_np = self._load_raw_bytes(
                v_img, dtype=rec.image_dtype, shape=rec.image_shape
            ).copy()
        else:
            image_np = self._load_npy_bytes(v_img)

        if rec.depth_dtype and rec.depth_shape:
            depth_np = self._load_raw_bytes(
                v_depth, dtype=rec.depth_dtype, shape=rec.depth_shape
            ).copy()
        else:
            depth_np = self._load_npy_bytes(v_depth)

        if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
            depth_np = depth_np[..., 0]

        # torchvision Resize(size=int) makes the shorter side == image_size.
        # If the sample is already resized to that convention, skip Resize.
        do_resize = min(int(image_np.shape[0]), int(image_np.shape[1])) != self.image_size
        if do_resize:
            image = self.image_transform_with_resize(image_np)
            depth = self.depth_transform_with_resize(depth_np).squeeze(0)

        else:
            image = self.image_transform_no_resize(image_np)
            depth = self.depth_transform_no_resize(depth_np).squeeze(0)

        valid_mask = (depth >= self.min_depth) & (depth <= self.max_depth)

        return {
            "image": image,
            "depth": depth,
            "valid_mask": valid_mask,
            "id": f"dataset{rec.dataset_number}/keyframe{rec.keyframe_number}/{rec.frame_id}",
        }


class ScaredLmdbDataModule(pl.LightningDataModule):
    """Lightning DataModule for SCARED depth stored in LMDB.

    Expects cfg to provide:
      - lmdb_path
      - megainfo_path
      - train_datasets, val_datasets
      - image_size, batch_size, num_workers, min_depth, max_depth
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg

        self.train_ds = LmdbMapDataset(
            cfg.lmdb_path,
            megainfo_path=cfg.megainfo_path,
            allowed_datasets=cfg.train_datasets,
            image_size=cfg.image_size,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
        )
        self.val_ds = LmdbMapDataset(
            cfg.lmdb_path,
            megainfo_path=cfg.megainfo_path,
            allowed_datasets=cfg.val_datasets,
            image_size=cfg.image_size,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
