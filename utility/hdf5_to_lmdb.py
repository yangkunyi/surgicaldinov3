#!/usr/bin/env python3
"""Convert SCARED-style HDF5 into a single LMDB + megainfo file.

Expected HDF5 layout (matches `utility/hdf5_to_lance.ipynb` and
`eval/depth/dataset.py`):

  /dataset{N}/keyframe{K}/{frame_id}/image   uint8  (H, W, 3)
  /dataset{N}/keyframe{K}/{frame_id}/gt      float32 (H, W, 1) or (H, W)

Output:
  - One LMDB environment stored as a single file (subdir=False).
  - One megainfo file mapping each LMDB index to (dataset_number, keyframe_number).

LMDB keys (index is 1-based by default):
  0000001-image
  0000001-depth

Values are stored as raw contiguous bytes (via numpy `.tobytes(order='C')`).
Array `dtype` and `shape` are written to the megainfo file so readers can
reconstruct arrays with `np.frombuffer(..., dtype=...).reshape(shape)`.

Example:
  /bd_byta6000i0/users/surgicaldinov2/miniforge3/bin/conda run -n da3 \
    python utility/hdf5_to_lmdb.py \
      --h5_path /bd_byta6000i0/users/surgical_depth/SCARED_fixed/scared.hdf5 \
      --output_lmdb /bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/data/scared.lmdb \
      --megainfo_path /bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/data/scared.megainfo.csv \
      --map_size_gb 512
"""

from __future__ import annotations

import argparse
import csv
import io
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
    p = argparse.ArgumentParser(description="Convert SCARED HDF5 to LMDB + megainfo")
    p.add_argument(
        "--h5_path",
        type=str,
        default="/bd_byta6000i0/users/surgical_depth/SCARED_fixed/scared.hdf5",
        help="Input HDF5 path (e.g. SCARED_fixed/scared.hdf5)",
    )
    p.add_argument(
        "--output_lmdb",
        type=str,
        default="/bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/data/SCARED-LMDB/SCARED.lmdb",
        help="Output LMDB file path (single file; subdir=False).",
    )
    p.add_argument(
        "--megainfo_path",
        type=str,
        default="/bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/data/SCARED-LMDB/SCARED.csv",
        help="Output megainfo file path (csv or jsonl depending on --megainfo_format).",
    )
    p.add_argument(
        "--megainfo_format",
        type=str,
        choices=("csv", "jsonl"),
        default="csv",
        help="Megainfo output format (default: csv).",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset numbers to include, e.g. '1,2,3'. Empty = all.",
    )
    p.add_argument(
        "--keyframes",
        type=str,
        default="",
        help="Comma-separated keyframe numbers to include, e.g. '1,2'. Empty = all.",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle frame order before writing (requires pre-scan list).",
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
        help="Zero-padding width for index keys (default: 7 -> 0000001).",
    )
    p.add_argument(
        "--commit_interval",
        type=int,
        default=128,
        help="Commit transaction every N frames (default: 128).",
    )
    p.add_argument(
        "--map_size_gb",
        type=float,
        default=1024.0,
        help=(
            "LMDB map size in GB (default: 1024). This is virtual address space, "
            "not preallocated disk, but must be larger than the final DB size."
        ),
    )
    p.add_argument(
        "--keep_depth_channel",
        action="store_true",
        help="Keep depth as (H, W, 1) if present; default squeezes to (H, W).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they exist.",
    )
    p.add_argument(
        "--verify",
        type=int,
        default=0,
        help="After writing, randomly verify N indices by reading back and loading npy.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only scan HDF5 and report how many frames would be written; do not write outputs.",
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

    import h5py  # local import to allow module import without h5py

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


def _to_raw_bytes(arr) -> bytes:
    import numpy as np

    arr = np.ascontiguousarray(arr)
    return arr.tobytes(order="C")


def _shape_to_str(shape: tuple[int, ...]) -> str:
    return json.dumps(list(shape), separators=(",", ":"))


def _dtype_to_str(dtype) -> str:
    # Use NumPy's canonical dtype string.
    import numpy as np

    return str(np.dtype(dtype))


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


def _prepare_output_paths(output_lmdb: Path, megainfo_path: Path, overwrite: bool) -> None:
    output_lmdb.parent.mkdir(parents=True, exist_ok=True)
    megainfo_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if output_lmdb.exists():
        existing.append(output_lmdb)
    if megainfo_path.exists():
        existing.append(megainfo_path)
    # lmdb may create a lock file alongside the main file when lock=True; we use lock=False,
    # but still guard against stale lock file if present.
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


def _estimate_map_size_bytes(num_frames: int, map_size_gb: float) -> int:
    # If user provided map_size_gb explicitly, trust it.
    # Convert GB to bytes (GiB-like).
    return int(map_size_gb * (1024**3))


def convert_hdf5_to_lmdb(
    *,
    h5_path: Path,
    output_lmdb: Path,
    megainfo_path: Path,
    megainfo_format: str,
    dataset_numbers: Optional[set[int]],
    keyframe_numbers: Optional[set[int]],
    shuffle: bool,
    seed: int,
    limit: int,
    start_index: int,
    index_width: int,
    commit_interval: int,
    map_size_gb: float,
    keep_depth_channel: bool,
    verify: int,
) -> None:
    import h5py
    import lmdb
    import numpy as np

    # Output paths are validated/created by main(); keep this function focused on conversion.
    output_lmdb.parent.mkdir(parents=True, exist_ok=True)
    megainfo_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Opening HDF5: %s", h5_path)
    with h5py.File(str(h5_path), "r") as h5:
        logging.info("Scanning HDF5 index...")
        all_refs = list(_iter_frame_refs(h5))
        selected_refs = _select_frame_refs(all_refs, dataset_numbers, keyframe_numbers)

        if not selected_refs:
            raise RuntimeError("No frames found after applying dataset/keyframe filters.")

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(selected_refs)

        if limit and limit > 0:
            selected_refs = selected_refs[:limit]

        num_frames = len(selected_refs)
        logging.info("Frames to write: %d", num_frames)

        map_size = _estimate_map_size_bytes(num_frames, map_size_gb)
        logging.info("Opening LMDB: %s (map_size=%.2f GB)", output_lmdb, map_size / (1024**3))

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

        try:
            if megainfo_format == "csv":
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
                    index_num = start_index + offset
                    index_str = f"{index_num:0{index_width}d}"

                    grp = h5[ref.ds_group][ref.kf_group][ref.frame_id]
                    image = grp["image"][...]
                    depth = grp["gt"][...]

                    if not keep_depth_channel and depth.ndim == 3 and depth.shape[-1] == 1:
                        depth = depth[..., 0]

                    depth = np.asarray(depth, dtype=np.float32)

                    image_bytes = _to_raw_bytes(image)
                    depth_bytes = _to_raw_bytes(depth)

                    k_img = f"{index_str}-image".encode("utf-8")
                    k_depth = f"{index_str}-depth".encode("utf-8")
                    txn.put(k_img, image_bytes)
                    txn.put(k_depth, depth_bytes)

                    if megainfo_format == "csv":
                        assert writer is not None
                        writer.writerow(
                            [
                                index_str,
                                ref.dataset_number,
                                ref.keyframe_number,
                                ref.frame_id,
                                _dtype_to_str(image.dtype),
                                _shape_to_str(image.shape),
                                _dtype_to_str(depth.dtype),
                                _shape_to_str(depth.shape),
                            ]
                        )
                    else:
                        rec = {
                            "index": index_str,
                            "dataset_number": ref.dataset_number,
                            "keyframe_number": ref.keyframe_number,
                            "frame_id": ref.frame_id,
                            "h5_group": f"{ref.ds_group}/{ref.kf_group}/{ref.frame_id}",
                            "image_dtype": _dtype_to_str(image.dtype),
                            "image_shape": list(image.shape),
                            "depth_dtype": _dtype_to_str(depth.dtype),
                            "depth_shape": list(depth.shape),
                        }
                        megainfo_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    frames_in_txn += 1
                    if commit_interval > 0 and frames_in_txn >= commit_interval:
                        txn.commit()
                        txn = env.begin(write=True)
                        frames_in_txn = 0

                    if (offset + 1) % 100 == 0 or (offset + 1) == num_frames:
                        logging.info("Written %d/%d frames", offset + 1, num_frames)

                txn.commit()

        finally:
            env.sync()
            env.close()

    if verify and verify > 0:
        _verify_lmdb(
            lmdb_path=output_lmdb,
            megainfo_path=megainfo_path,
            megainfo_format=megainfo_format,
            start_index=start_index,
            num_frames=num_frames,
            index_width=index_width,
            n=verify,
        )


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

    # Load just enough metadata to reconstruct arrays.
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
                k_img = f"{index_str}-image".encode("utf-8")
                k_depth = f"{index_str}-depth".encode("utf-8")
                v_img = txn.get(k_img)
                v_depth = txn.get(k_depth)
                if v_img is None or v_depth is None:
                    raise RuntimeError(f"Missing keys for index {index_str}")

                m = meta.get(index_str)
                if m is None:
                    raise RuntimeError(f"Missing megainfo entry for index {index_str}")

                img_dtype = np.dtype(str(m.get("image_dtype")))
                img_shape = _parse_shape(m.get("image_shape"))
                depth_dtype = np.dtype(str(m.get("depth_dtype")))
                depth_shape = _parse_shape(m.get("depth_shape"))

                img = np.frombuffer(v_img, dtype=img_dtype).reshape(img_shape)
                depth = np.frombuffer(v_depth, dtype=depth_dtype).reshape(depth_shape)
                logging.info(
                    "Verified %s: image=%s/%s depth=%s/%s",
                    index_str,
                    img.shape,
                    img.dtype,
                    depth.shape,
                    depth.dtype,
                )
    finally:
        env.close()


def main(argv: Optional[list[str]] = None) -> int:
    _setup_logging()
    args = _parse_args(argv)

    h5_path = Path(args.h5_path).expanduser()
    output_lmdb = Path(args.output_lmdb).expanduser()
    megainfo_path = Path(args.megainfo_path).expanduser()

    dataset_numbers = _parse_int_list(args.datasets)
    keyframe_numbers = _parse_int_list(args.keyframes)

    if not h5_path.exists():
        logging.error("Input HDF5 does not exist: %s", h5_path)
        return 2

    if args.dry_run:
        import h5py

        with h5py.File(str(h5_path), "r") as h5:
            all_refs = list(_iter_frame_refs(h5))
            selected_refs = _select_frame_refs(all_refs, dataset_numbers, keyframe_numbers)
        if args.limit and args.limit > 0:
            selected_refs = selected_refs[: args.limit]
        logging.info("Dry-run: would write %d frames", len(selected_refs))
        return 0

    try:
        _prepare_output_paths(output_lmdb, megainfo_path, overwrite=args.overwrite)
    except FileExistsError as e:
        logging.error(str(e))
        return 2

    try:
        convert_hdf5_to_lmdb(
            h5_path=h5_path,
            output_lmdb=output_lmdb,
            megainfo_path=megainfo_path,
            megainfo_format=args.megainfo_format,
            dataset_numbers=dataset_numbers,
            keyframe_numbers=keyframe_numbers,
            shuffle=args.shuffle,
            seed=args.seed,
            limit=args.limit,
            start_index=args.start_index,
            index_width=args.index_width,
            commit_interval=args.commit_interval,
            map_size_gb=args.map_size_gb,
            keep_depth_channel=args.keep_depth_channel,
            verify=args.verify,
        )
    except Exception as e:
        logging.exception("Conversion failed: %s", e)
        return 1

    logging.info("Done. LMDB: %s", output_lmdb)
    logging.info("Done. Megainfo: %s", megainfo_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
