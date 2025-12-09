#!/usr/bin/env python3
"""Create WebDataset shards from the fixed SCARED dataset.

The script expects the following on-disk layout (as in ``SCARED_fixed``::

    SCARED_fixed/
        dataset1/
            keyframe1/
                image_02/
                    data/
                        0000000001.png
                        0000000002.png
                        ...
                        groundtruth/
                            scene_points000000.npy
                            scene_points000000.tiff
                            scene_points000001.npy
                            scene_points000001.tiff
                            ...
        dataset2/
        ...
        dataset9/

Each sample in the resulting WebDataset contains exactly three payload keys:

    - ``png``: RGB image bytes from ``*.png``
    - ``depth``: depth / scene points bytes (``.npy`` preferred, otherwise
      ``.tiff``), with the actual type recorded in the JSON metadata
    - ``json``: a JSON sidecar describing the sample, including dataset /
      keyframe ids, relative paths, and ``depth_type`` (``"npy"`` or
      ``"tiff"``)

Shards are written to ``output_dir / "shards"`` by default, using a
pattern like ``shard-000000.tar``. All datasets are mixed together but
remain distinguishable via the ``__key__`` and JSON metadata.

Example
-------

.. code-block:: bash

    python utility/create_scared_wds.py \
        --scared_root /bd_byta6000i0/users/surgical_depth/SCARED_fixed \
        --output_dir /bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/data/SCARED \
        --maxcount 4000 --overwrite

You can then load the shards with WebDataset, e.g.::

    import webdataset as wds

    shards = \
        "/bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/" \
        "data/SCARED/shards/shard-{000000..000999}.tar"  # adjust upper bound

    dataset = (
        wds.WebDataset(shards, shardshuffle=42)
        .decode("pil")  # decodes the PNG image; depth stays as raw bytes
        .shuffle(4000)
        .to_tuple("png", "depth", "json", "__key__")
    )

"""

import argparse
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import webdataset as wds


@dataclass
class ScaredEntry:
    """Metadata for a single SCARED sample.

    Only paths and small identifiers are stored here; actual bytes are read
    later when writing shards.
    """

    dataset_id: str
    keyframe_id: str
    frame_index: int  # 1-based index from image filename
    image_path: Path
    npy_path: Optional[Path]
    tiff_path: Optional[Path]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create WebDataset shards from SCARED_fixed structure.",
    )
    parser.add_argument(
        "--scared_root",
        type=str,
        default="/bd_byta6000i0/users/surgical_depth/SCARED_fixed",
        help="Root directory of SCARED_fixed (containing dataset1..dataset9)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/data/SCARED",
        help="Directory where WebDataset shards will be written.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help=(
            "Optional comma-separated list of dataset names to include, "
            "e.g. 'dataset1,dataset2'. If empty, all 'dataset*' directories "
            "under scared_root are used."
        ),
    )
    parser.add_argument(
        "--camera_dir",
        type=str,
        default="image_02",
        help=(
            "Camera subdirectory name under each keyframe (default: 'image_02'). "
            "Change if your SCARED_fixed layout uses a different camera folder."
        ),
    )
    parser.add_argument(
        "--maxcount",
        type=int,
        default=50,
        help="Maximum number of samples per shard (default: 4000).",
    )
    parser.add_argument(
        "--shard_suffix",
        type=str,
        default=".tar",
        help=(
            "Shard filename suffix, e.g. '.tar' or '.tar.gz' (default: '.tar'). "
            "The base pattern is 'shard-%%06d', so filenames look like "
            "'shard-000000.tar'."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing shard files matching 'shard-*<shard_suffix>' "
            "if they already exist."
        ),
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="",
        help=(
            "Optional compression (e.g., 'gz' for gzip). Usually implied by "
            "shard_suffix like '.tar.gz'."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for global shuffling. Set to -1 to disable "
            "deterministic seeding."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Number of parallel workers for reading input files. "
            "0 = auto (use a reasonable value based on CPU count), "
            "1 = no parallelism."
        ),
    )
    return parser.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _iter_dataset_dirs(scared_root: Path, include_names: Optional[List[str]]) -> Iterable[Path]:
    """Yield dataset directories under ``scared_root``.

    If ``include_names`` is provided, only directories whose name is in that
    list are yielded.
    """

    if not scared_root.exists() or not scared_root.is_dir():
        raise FileNotFoundError(
            f"SCARED root directory does not exist or is not a directory: {scared_root}"
        )

    ds_dirs = sorted(p for p in scared_root.iterdir() if p.is_dir())
    if include_names:
        include_set = set(include_names)
        ds_dirs = [p for p in ds_dirs if p.name in include_set]

    for d in ds_dirs:
        if not d.name.startswith("dataset"):
            # Ignore any non-dataset directories such as 'scared.hdf5'.
            logging.debug("Skipping non-dataset directory: %s", d)
            continue
        yield d


def gather_scared_entries(
    scared_root: Path,
    dataset_names: Optional[List[str]] = None,
    camera_dir: str = "image_02",
) -> List[ScaredEntry]:
    """Collect metadata for all usable SCARED samples.

    This walks ``dataset*/keyframe*/<camera_dir>/data`` and matches each
    ``*.png`` (or ``*.jpg``/``*.jpeg``) with the corresponding
    ``groundtruth/scene_pointsXXXXXX.(npy|tiff)``.
    """

    entries: List[ScaredEntry] = []

    for dataset_dir in _iter_dataset_dirs(scared_root, dataset_names):
        dataset_id = dataset_dir.name
        logging.info("Scanning %s", dataset_id)

        keyframe_dirs = sorted(dataset_dir.glob("keyframe*"))
        if not keyframe_dirs:
            logging.warning("No keyframe* directories found under %s", dataset_dir)
            continue

        for keyframe_dir in keyframe_dirs:
            keyframe_id = keyframe_dir.name
            img_root = keyframe_dir / camera_dir / "data"
            gt_root = img_root / "groundtruth"

            if not img_root.exists() or not img_root.is_dir():
                logging.warning("Missing image data directory: %s", img_root)
                continue

            if not gt_root.exists() or not gt_root.is_dir():
                logging.warning("Missing groundtruth directory: %s", gt_root)
                continue

            image_files = sorted(
                p
                for p in img_root.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )

            if not image_files:
                logging.warning("No image files found in %s", img_root)
                continue

            for img_path in image_files:
                stem = img_path.stem
                try:
                    frame_index = int(stem)
                except ValueError:
                    logging.warning("Skipping image with non-integer stem %s", img_path)
                    continue

                # Scene points indices appear to be 0-based in the filenames,
                # while RGB frames are 1-based (0000000001.png -> scene_points000000).
                depth_idx = frame_index - 1
                depth_stem = f"scene_points{depth_idx:06d}"

                npy_path = gt_root / f"{depth_stem}.npy"
                tiff_path = gt_root / f"{depth_stem}.tiff"

                if not npy_path.exists() and not tiff_path.exists():
                    logging.warning(
                        "No groundtruth found for %s (expected %s.[npy|tiff])",
                        img_path,
                        depth_stem,
                    )
                    continue

                # Only store existing paths; missing ones stay as None.
                if not npy_path.exists():
                    npy_path = None
                if not tiff_path.exists():
                    tiff_path = None

                entries.append(
                    ScaredEntry(
                        dataset_id=dataset_id,
                        keyframe_id=keyframe_id,
                        frame_index=frame_index,
                        image_path=img_path,
                        npy_path=npy_path,
                        tiff_path=tiff_path,
                    )
                )

    return entries


def _entry_to_sample(entry: ScaredEntry) -> Optional[dict]:
    """Convert a single ScaredEntry into a WebDataset sample dict.

    This function is pure w.r.t. global state and safe to call from
    multiple threads. Logging is still emitted from within the function.
    """

    key = f"{entry.dataset_id}_{entry.keyframe_id}_{entry.frame_index:06d}"

    # Read RGB image bytes (always stored under "png" in the shard).
    img_suffix = entry.image_path.suffix.lower()
    if img_suffix != ".png":
        logging.warning(
            "Image %s does not have .png extension (found %s)",
            entry.image_path,
            img_suffix,
        )

    with entry.image_path.open("rb") as f:
        img_bytes = f.read()

    # Read depth bytes, preferring NPY over TIFF, and store them
    # under a unified "depth" key.
    depth_bytes = None

    if entry.npy_path is not None:
        with entry.npy_path.open("rb") as f:
            depth_bytes = f.read()
    elif entry.tiff_path is not None:
        with entry.tiff_path.open("rb") as f:
            depth_bytes = f.read()

    if depth_bytes is None:
        logging.warning("No depth data for %s; skipping sample", key)
        return None

    sample = {"__key__": key, "png": img_bytes, "depth": depth_bytes}

    # Lightweight JSON metadata to make it easy to distinguish
    # between datasets/keyframes later on.
    meta = {
        "dataset": entry.dataset_id,
        "keyframe": entry.keyframe_id,
        "frame_index": entry.frame_index,
    }

    sample["json"] = json.dumps(meta).encode("utf-8")
    return sample


def write_shards(
    entries: List[ScaredEntry],
    scared_root: Path,
    output_dir: Path,
    shard_suffix: str,
    maxcount: int,
    overwrite: bool,
    compression: str,
    num_workers: int,
) -> int:
    """Write globally shuffled entries into WebDataset shards.

    Shards live under ``output_dir / "shards"`` and are named using the
    pattern ``shard-%06d<shard_suffix>``.
    """

    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # existing_shards = sorted(shards_dir.glob(f"shard-*{shard_suffix}"))
    # if existing_shards:
    #     if not overwrite:
    #         logging.info(
    #             "Found %d existing shards matching 'shard-*%s'; "
    #             "use --overwrite to regenerate.",
    #             len(existing_shards),
    #             shard_suffix,
    #         )
    #         return 0

    #     logging.info(
    #         "Overwriting %d existing shards matching 'shard-*%s'.",
    #         len(existing_shards),
    #         shard_suffix,
    #     )
    #     for shard in existing_shards:
    #         shard.unlink()

    writer_opts = {}
    if compression:
        writer_opts["compress"] = compression

    pattern_str = str(shards_dir / ("shard-val-%06d" + shard_suffix))

    total_written = 0
    with wds.ShardWriter(pattern_str, maxcount=maxcount, **writer_opts) as sink:
        # Use a thread pool to parallelize reading image/depth bytes while
        # keeping writes to the ShardWriter single-threaded (it is not
        # guaranteed to be thread-safe).
        if num_workers is None or num_workers <= 1:
            for sample in map(_entry_to_sample, entries):
                if sample is None:
                    continue
                sink.write(sample)
                total_written += 1
        else:
            logging.info("Using %d parallel workers for sample preparation", num_workers)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for sample in executor.map(_entry_to_sample, entries):
                    if sample is None:
                        continue
                    sink.write(sample)
                    total_written += 1

    logging.info(
        "Done writing SCARED shards. Total samples written: %d", total_written
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)

    scared_root = Path(args.scared_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    shard_suffix = args.shard_suffix
    maxcount = args.maxcount
    overwrite = args.overwrite
    compression = args.compression
    seed = args.seed
    camera_dir = args.camera_dir
    num_workers = args.num_workers

    dataset_names = (
        [s.strip() for s in args.datasets.split(",") if s.strip()]
        if args.datasets
        else None
    )

    logging.info("SCARED root: %s", scared_root)
    logging.info("Output dir: %s", output_dir)
    logging.info("Shard suffix: %s", shard_suffix)
    logging.info("Max samples per shard: %d", maxcount)
    logging.info("Camera directory: %s", camera_dir)
    logging.info("Num workers: %s", "auto" if num_workers == 0 else num_workers)
    if dataset_names:
        logging.info("Restricting to datasets: %s", ", ".join(dataset_names))
    if seed is not None and seed >= 0:
        logging.info("Shuffle seed: %d", seed)
    if compression:
        logging.info("Compression: %s", compression)

    try:
        entries = gather_scared_entries(
            scared_root=scared_root,
            dataset_names=dataset_names,
            camera_dir=camera_dir,
        )
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1

    if not entries:
        logging.warning("No SCARED entries found; nothing to write.")
        return 0

    # Global shuffle before sharding.
    if seed is not None and seed >= 0:
        rng = random.Random(seed)
        rng.shuffle(entries)
    else:
        random.shuffle(entries)

    # Resolve automatic worker count (0) to a concrete number here so that
    # the rest of the code does not have to branch on this special value.
    if num_workers == 0:
        try:
            import os

            cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = 1
        # Cap to a small, reasonable value to avoid oversubscribing the
        # machine while still benefiting from I/O parallelism.
        num_workers = min(cpu_count, 8)
        logging.info("Resolved num_workers=0 to %d based on CPU count", num_workers)

    return write_shards(
        entries=entries,
        scared_root=scared_root,
        output_dir=output_dir,
        shard_suffix=shard_suffix,
        maxcount=maxcount,
        overwrite=overwrite,
        compression=compression,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
