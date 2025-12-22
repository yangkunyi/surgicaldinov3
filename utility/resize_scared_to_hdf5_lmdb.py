#!/usr/bin/env python3
"""Resize the fixed SCARED dataset once and save outputs to HDF5 + LMDB.

Motivation
----------
SCARED samples are high-resolution (e.g. 1024x1280). Many training pipelines
resize in ``__getitem__`` (see ``eval/depth/scared_lmdb.py``), which adds CPU
overhead and keeps IO heavy. This script pre-resizes both RGB and depth using
the same semantics as torchvision ``Resize(size=int)``: match the *shorter*
side to ``--image_size`` while preserving aspect ratio.

Input HDF5 layout (SCARED_fixed)
-------------------------------
  /dataset{N}/keyframe{K}/{frame_id}/image   uint8  (H, W, 3)
  /dataset{N}/keyframe{K}/{frame_id}/gt      float32 (H, W, 1) or (H, W)

Outputs
-------
1) Resized HDF5 (same nested group structure):
  /dataset{N}/keyframe{K}/{frame_id}/image   uint8  (H', W', 3)
  /dataset{N}/keyframe{K}/{frame_id}/gt      float32 (H', W', 1)

2) Resized LMDB + megainfo (CSV or JSONL):
  Keys:
    {index}-image  -> raw bytes of uint8 (H', W', 3)
    {index}-depth  -> raw bytes of float32 (H', W')

Example
-------
python utility/resize_scared_to_hdf5_lmdb.py \
  --input_h5 /bd_byta6000i0/users/surgical_depth/SCARED_fixed/scared.hdf5 \
  --output_h5 /bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/data/SCARED-HDF5/SCARED_256.hdf5 \
  --output_lmdb /bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/data/SCARED-LMDB/SCARED_256.lmdb \
  --megainfo_path /bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/data/SCARED-LMDB/SCARED_256.csv \
  --image_size 256 \
  --map_size_gb 128 \
  --commit_interval 256
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


_DATASET_RE = re.compile(r"^dataset(?P<num>\d+)$")
_KEYFRAME_RE = re.compile(r"^keyframe(?P<num>\d+)$")


@dataclass(frozen=True)
class FrameRef:
    ds_group: str
    kf_group: str
    frame_id: str

    @property
    def dataset_number(self) -> int:
        m = _DATASET_RE.match(self.ds_group)
        if not m:
            raise ValueError(f"Unexpected dataset group name: {self.ds_group}")
        return int(m.group("num"))

    @property
    def keyframe_number(self) -> int:
        m = _KEYFRAME_RE.match(self.kf_group)
        if not m:
            raise ValueError(f"Unexpected keyframe group name: {self.kf_group}")
        return int(m.group("num"))


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resize SCARED_fixed HDF5 to smaller HDF5 + LMDB.",
    )
    p.add_argument(
        "--input_h5",
        type=str,
        default="/bd_byta6000i0/users/surgical_depth/SCARED_fixed/scared.hdf5",
        help="Input HDF5 path (SCARED_fixed/scared.hdf5).",
    )
    p.add_argument(
        "--output_h5",
        type=str,
        required=True,
        help="Output resized HDF5 path.",
    )
    p.add_argument(
        "--output_lmdb",
        type=str,
        required=True,
        help="Output resized LMDB path (single file; subdir=False).",
    )
    p.add_argument(
        "--megainfo_path",
        type=str,
        required=True,
        help="Output megainfo path (csv or jsonl).",
    )
    p.add_argument(
        "--megainfo_format",
        type=str,
        choices=("csv", "jsonl"),
        default="csv",
        help="Megainfo format for LMDB (default: csv).",
    )
    p.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Shorter side after resize (matches torchvision Resize(size=int)).",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset numbers to include (empty = all).",
    )
    p.add_argument(
        "--keyframes",
        type=str,
        default="",
        help="Comma-separated keyframe numbers to include (empty = all).",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle frame order before writing LMDB indices.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed (used only with --shuffle).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of frames to write (0 = no limit).",
    )
    p.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="Starting index for LMDB keys (default: 1).",
    )
    p.add_argument(
        "--index_width",
        type=int,
        default=8,
        help="Zero-padding width for LMDB index keys (default: 8 -> 00000001).",
    )
    p.add_argument(
        "--commit_interval",
        type=int,
        default=256,
        help="Commit transaction every N frames (default: 256).",
    )
    p.add_argument(
        "--map_size_gb",
        type=float,
        default=256.0,
        help=(
            "LMDB map size in GB (virtual address space; must exceed final DB size). "
            "Default: 256."
        ),
    )
    p.add_argument(
        "--h5_compression",
        type=str,
        default="",
        help="Optional HDF5 compression, e.g. 'gzip' or 'lzf' (default: none).",
    )
    p.add_argument(
        "--h5_compression_level",
        type=int,
        default=4,
        help="Compression level for gzip (ignored for non-gzip).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they exist.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only scan input and report how many frames would be written.",
    )
    p.add_argument(
        "--verify",
        type=int,
        default=0,
        help="Randomly verify N LMDB entries after writing.",
    )
    return p.parse_args(argv)


def _parse_int_list(spec: str) -> Optional[set[int]]:
    if not spec.strip():
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _iter_frame_refs(h5) -> Iterator[FrameRef]:
    """Yield FrameRef from an opened h5py.File.

    Filters out non-group nodes and frames without both 'image' and 'gt'.
    """

    import h5py  # local import

    for ds_group in sorted(h5.keys()):
        ds_obj = h5[ds_group]
        if not isinstance(ds_obj, h5py.Group):
            continue
        if not ds_group.startswith("dataset"):
            continue

        for kf_group in sorted(ds_obj.keys()):
            kf_obj = ds_obj[kf_group]
            if not isinstance(kf_obj, h5py.Group):
                continue
            if not kf_group.startswith("keyframe"):
                continue

            for frame_id in sorted(kf_obj.keys()):
                fr_obj = kf_obj[frame_id]
                if not isinstance(fr_obj, h5py.Group):
                    continue
                if "image" not in fr_obj or "gt" not in fr_obj:
                    continue
                yield FrameRef(ds_group=ds_group, kf_group=kf_group, frame_id=frame_id)


def _select_frame_refs(
    all_refs: Iterable[FrameRef],
    dataset_numbers: Optional[set[int]],
    keyframe_numbers: Optional[set[int]],
) -> list[FrameRef]:
    selected: list[FrameRef] = []
    for ref in all_refs:
        if dataset_numbers is not None and ref.dataset_number not in dataset_numbers:
            continue
        if keyframe_numbers is not None and ref.keyframe_number not in keyframe_numbers:
            continue
        selected.append(ref)
    return selected


def _prepare_output_paths(
    *,
    output_h5: Path,
    output_lmdb: Path,
    megainfo_path: Path,
    overwrite: bool,
) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    output_lmdb.parent.mkdir(parents=True, exist_ok=True)
    megainfo_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[Path] = []
    for p in (output_h5, output_lmdb, megainfo_path):
        if p.exists():
            existing.append(p)

    lock_path = output_lmdb.with_name(output_lmdb.name + "-lock")
    if lock_path.exists():
        existing.append(lock_path)

    if existing and not overwrite:
        raise FileExistsError(
            "Output already exists. Use --overwrite to replace: "
            + ", ".join(str(p) for p in existing)
        )

    if overwrite:
        for p in existing:
            p.unlink()


def _shape_to_str(shape: tuple[int, ...]) -> str:
    return json.dumps(list(shape), separators=(",", ":"))


def _dtype_to_str(dtype) -> str:
    import numpy as np

    return str(np.dtype(dtype))


def _to_raw_bytes(arr) -> bytes:
    import numpy as np

    arr = np.ascontiguousarray(arr)
    return arr.tobytes(order="C")


def _compute_resized_hw(h: int, w: int, short_side: int) -> tuple[int, int]:
    """Match torchvision Resize(size=int): resize shorter side to short_side."""
    if short_side <= 0:
        raise ValueError(f"short_side must be positive, got {short_side}")
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size: h={h}, w={w}")

    if h <= w:
        new_h = short_side
        new_w = int(round(w * (short_side / h)))
    else:
        new_w = short_side
        new_h = int(round(h * (short_side / w)))

    new_h = max(1, new_h)
    new_w = max(1, new_w)
    return new_h, new_w


def _resize_image_uint8_hwc(image_hwc, *, short_side: int):
    import numpy as np
    import torch
    import torch.nn.functional as F

    if not (isinstance(image_hwc, np.ndarray) and image_hwc.ndim == 3 and image_hwc.shape[2] == 3):
        raise ValueError(f"Expected image ndarray (H,W,3), got {type(image_hwc)} shape={getattr(image_hwc, 'shape', None)}")
    if image_hwc.dtype != np.uint8:
        image_hwc = np.asarray(image_hwc, dtype=np.uint8)

    h, w, _ = image_hwc.shape
    new_h, new_w = _compute_resized_hw(h, w, short_side)
    if (new_h, new_w) == (h, w):
        return image_hwc

    image = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    try:
        resized = F.interpolate(
            image,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        # Older torch versions may not support antialias.
        resized = F.interpolate(
            image,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

    resized_u8 = resized.clamp(0.0, 255.0).round().to(torch.uint8)
    out = resized_u8.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return out


def _resize_depth_f32_hw(depth, *, short_side: int):
    import numpy as np
    import torch
    import torch.nn.functional as F

    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth ndarray (H,W) or (H,W,1), got shape={depth.shape}")
    depth = np.asarray(depth, dtype=np.float32)

    h, w = depth.shape
    new_h, new_w = _compute_resized_hw(h, w, short_side)
    if (new_h, new_w) == (h, w):
        return depth

    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
    resized = F.interpolate(depth_t, size=(new_h, new_w), mode="nearest")
    out = resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    return out


def _write_megainfo_header_csv(writer: csv.writer) -> None:
    writer.writerow(
        [
            "index",
            "dataset_number",
            "keyframe_number",
            "frame_id",
            "image_dtype",
            "image_shape",
            "depth_dtype",
            "depth_shape",
        ]
    )


def _estimate_map_size_bytes(map_size_gb: float) -> int:
    return int(map_size_gb * (1024**3))


def _verify_lmdb(
    *,
    lmdb_path: Path,
    megainfo_path: Path,
    megainfo_format: str,
    start_index: int,
    num_frames: int,
    index_width: int,
    n: int,
) -> None:
    import lmdb
    import numpy as np

    if num_frames <= 0:
        return

    rng = random.Random(123)
    indices = [start_index + rng.randrange(0, num_frames) for _ in range(min(n, num_frames))]
    wanted = {f"{idx:0{index_width}d}" for idx in indices}

    meta: dict[str, dict[str, object]] = {}
    if megainfo_format == "jsonl" or megainfo_path.suffix.lower() == ".jsonl":
        with megainfo_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                idx = str(obj.get("index", ""))
                if idx in wanted:
                    meta[idx] = obj
                    if len(meta) == len(wanted):
                        break
    else:
        with megainfo_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = str(row.get("index", ""))
                if idx in wanted:
                    meta[idx] = row
                    if len(meta) == len(wanted):
                        break

    def _parse_shape(value: object) -> tuple[int, ...]:
        if isinstance(value, list):
            return tuple(int(x) for x in value)
        s = str(value)
        return tuple(int(x) for x in json.loads(s))

    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_dbs=1,
    )
    try:
        with env.begin(write=False) as txn:
            for idx in indices:
                index_str = f"{idx:0{index_width}d}"
                m = meta.get(index_str)
                if m is None:
                    raise RuntimeError(f"Missing megainfo for index={index_str}")

                img_dtype = np.dtype(str(m["image_dtype"]))
                img_shape = _parse_shape(m["image_shape"])  # type: ignore[arg-type]
                depth_dtype = np.dtype(str(m["depth_dtype"]))
                depth_shape = _parse_shape(m["depth_shape"])  # type: ignore[arg-type]

                k_img = f"{index_str}-image".encode("utf-8")
                k_depth = f"{index_str}-depth".encode("utf-8")
                v_img = txn.get(k_img)
                v_depth = txn.get(k_depth)
                if v_img is None or v_depth is None:
                    raise RuntimeError(f"Missing LMDB keys for index={index_str}")

                img = np.frombuffer(v_img, dtype=img_dtype).reshape(img_shape)
                depth = np.frombuffer(v_depth, dtype=depth_dtype).reshape(depth_shape)
                if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
                    raise RuntimeError(f"Unexpected image decoded shape={img.shape} dtype={img.dtype}")
                if depth.dtype != np.float32 or depth.ndim != 2:
                    raise RuntimeError(f"Unexpected depth decoded shape={depth.shape} dtype={depth.dtype}")

        logging.info("LMDB verify: OK (%d samples)", len(indices))
    finally:
        env.close()


def main(argv: Optional[list[str]] = None) -> None:
    _setup_logging()
    args = _parse_args(argv)

    input_h5 = Path(args.input_h5)
    output_h5 = Path(args.output_h5)
    output_lmdb = Path(args.output_lmdb)
    megainfo_path = Path(args.megainfo_path)

    if not input_h5.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {input_h5}")

    _prepare_output_paths(
        output_h5=output_h5,
        output_lmdb=output_lmdb,
        megainfo_path=megainfo_path,
        overwrite=bool(args.overwrite),
    )

    dataset_numbers = _parse_int_list(args.datasets)
    keyframe_numbers = _parse_int_list(args.keyframes)

    logging.info("Opening input HDF5: %s", input_h5)
    import h5py

    with h5py.File(str(input_h5), "r") as h5_in:
        logging.info("Scanning HDF5 index...")
        all_refs = list(_iter_frame_refs(h5_in))
        selected_refs = _select_frame_refs(all_refs, dataset_numbers, keyframe_numbers)

        if not selected_refs:
            raise RuntimeError("No frames found after applying dataset/keyframe filters.")

        if args.shuffle:
            rng = random.Random(int(args.seed))
            rng.shuffle(selected_refs)

        if args.limit and int(args.limit) > 0:
            selected_refs = selected_refs[: int(args.limit)]

        num_frames = len(selected_refs)
        logging.info("Frames to write: %d", num_frames)

        # Dry-run: sample one frame to report resized shape.
        sample_ref = selected_refs[0]
        sample_grp = h5_in[sample_ref.ds_group][sample_ref.kf_group][sample_ref.frame_id]
        sample_image = sample_grp["image"][...]
        sample_depth = sample_grp["gt"][...]
        resized_image = _resize_image_uint8_hwc(sample_image, short_side=int(args.image_size))
        resized_depth = _resize_depth_f32_hw(sample_depth, short_side=int(args.image_size))
        logging.info(
            "Resize preview: image %s -> %s, depth %s -> %s",
            tuple(sample_image.shape),
            tuple(resized_image.shape),
            tuple(sample_depth.shape),
            tuple(resized_depth.shape),
        )

        if args.dry_run:
            logging.info("Dry run complete; nothing written.")
            return

        import lmdb
        import numpy as np

        map_size = _estimate_map_size_bytes(float(args.map_size_gb))
        logging.info(
            "Opening output LMDB: %s (map_size=%.2f GB)",
            output_lmdb,
            map_size / (1024**3),
        )
        env = lmdb.open(
            str(output_lmdb),
            subdir=False,
            map_size=map_size,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_dbs=1,
        )

        compression = str(args.h5_compression).strip() or None
        print(f"compression={compression}")
        compression_opts = None
        if compression == "gzip":
            compression_opts = int(args.h5_compression_level)

        logging.info("Opening output HDF5: %s", output_h5)
        with h5py.File(str(output_h5), "w") as h5_out:
            h5_out.attrs["source_h5"] = str(input_h5)
            h5_out.attrs["resized_short_side"] = int(args.image_size)
            h5_out.attrs["image_interp"] = "bilinear"
            h5_out.attrs["depth_interp"] = "nearest"
            if compression is not None:
                h5_out.attrs["compression"] = compression

            if args.megainfo_format == "csv":
                megainfo_f = megainfo_path.open("w", newline="", encoding="utf-8")
                writer = csv.writer(megainfo_f)
                _write_megainfo_header_csv(writer)
            else:
                megainfo_f = megainfo_path.open("w", encoding="utf-8")
                writer = None

            with megainfo_f:
                txn = env.begin(write=True)
                frames_in_txn = 0

                for offset, ref in enumerate(selected_refs):
                    index_num = int(args.start_index) + offset
                    index_str = f"{index_num:0{int(args.index_width)}d}"

                    grp_in = h5_in[ref.ds_group][ref.kf_group][ref.frame_id]
                    image = grp_in["image"][...]
                    depth = grp_in["gt"][...]

                    image_r = _resize_image_uint8_hwc(image, short_side=int(args.image_size))
                    depth_r = _resize_depth_f32_hw(depth, short_side=int(args.image_size))
                    depth_r = np.asarray(depth_r, dtype=np.float32)

                    # ---- write HDF5 ----
                    grp_out = h5_out.require_group(ref.ds_group).require_group(ref.kf_group).require_group(ref.frame_id)
                    if "image" in grp_out or "gt" in grp_out:
                        raise RuntimeError(f"Output HDF5 already has datasets under {grp_out.name}")
                    grp_out.create_dataset(
                        "image",
                        data=image_r,
                        dtype="uint8",
                        compression=compression,
                        compression_opts=compression_opts,
                        chunks=True if compression is not None else None,
                    )
                    grp_out.create_dataset(
                        "gt",
                        data=depth_r[..., None],
                        dtype="float32",
                        compression=compression,
                        compression_opts=compression_opts,
                        chunks=True if compression is not None else None,
                    )

                    # ---- write LMDB ----
                    image_bytes = _to_raw_bytes(image_r)
                    depth_bytes = _to_raw_bytes(depth_r)

                    k_img = f"{index_str}-image".encode("utf-8")
                    k_depth = f"{index_str}-depth".encode("utf-8")
                    txn.put(k_img, image_bytes)
                    txn.put(k_depth, depth_bytes)

                    if args.megainfo_format == "csv":
                        assert writer is not None
                        writer.writerow(
                            [
                                index_str,
                                ref.dataset_number,
                                ref.keyframe_number,
                                ref.frame_id,
                                _dtype_to_str(image_r.dtype),
                                _shape_to_str(tuple(image_r.shape)),
                                _dtype_to_str(depth_r.dtype),
                                _shape_to_str(tuple(depth_r.shape)),
                            ]
                        )
                    else:
                        rec = {
                            "index": index_str,
                            "dataset_number": ref.dataset_number,
                            "keyframe_number": ref.keyframe_number,
                            "frame_id": ref.frame_id,
                            "h5_group": f"{ref.ds_group}/{ref.kf_group}/{ref.frame_id}",
                            "image_dtype": _dtype_to_str(image_r.dtype),
                            "image_shape": list(image_r.shape),
                            "depth_dtype": _dtype_to_str(depth_r.dtype),
                            "depth_shape": list(depth_r.shape),
                        }
                        megainfo_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    frames_in_txn += 1
                    if int(args.commit_interval) > 0 and frames_in_txn >= int(args.commit_interval):
                        txn.commit()
                        txn = env.begin(write=True)
                        frames_in_txn = 0

                    if (offset + 1) % 100 == 0 or (offset + 1) == num_frames:
                        logging.info("Written %d/%d frames", offset + 1, num_frames)

                txn.commit()

        env.sync()
        env.close()

    if args.verify and int(args.verify) > 0:
        _verify_lmdb(
            lmdb_path=output_lmdb,
            megainfo_path=megainfo_path,
            megainfo_format=str(args.megainfo_format),
            start_index=int(args.start_index),
            num_frames=num_frames,
            index_width=int(args.index_width),
            n=int(args.verify),
        )

    logging.info("Done. Resized HDF5: %s", output_h5)
    logging.info("Done. Resized LMDB: %s", output_lmdb)
    logging.info("Done. Megainfo: %s", megainfo_path)


if __name__ == "__main__":
    main()
