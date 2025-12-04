#!/usr/bin/env python3
"""Create globally shuffled WebDataset shards from extracted video frames.

This script assumes frames have already been extracted with
`utility/extract_frames.py`, which creates one subdirectory per video under
`--frames_root` and writes frames using a pattern like `frame_000001.jpg`.

Behavior:
- Load metadata for all matching frames across all videos.
- Shuffle the full list of frames globally.
- Write WebDataset shards with a fixed maximum number of samples per shard,
  using names like `shard-000000.tar`.

Each sample stores:
- The JPEG image bytes under the `jpg` field.
- A JSON sidecar with `video_id`, `frame_index` (1-based within its video
  according to sorted filename order), and `frame_path` relative to
  `--frames_root`.

Example:
    python utility/create_wds_shards_global.py \
        --frames_root /path/to/frames \
        --output_dir /path/to/shards \
        --pattern "frame_*.jpg" \
        --maxcount 10000
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import webdataset as wds


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create globally shuffled WebDataset shards from extracted video frames."
        )
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="/bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/data/cholec80/frames",
        help="Root directory containing per-video frame subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/data/cholec80/shards",
        help="Directory where WebDataset shards will be written.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="frame_*.jpg",
        help=(
            "Glob pattern for frames inside each video directory "
            "(default: 'frame_*.jpg')."
        ),
    )
    parser.add_argument(
        "--shard_suffix",
        type=str,
        default=".tar",
        help=(
            "Shard filename suffix, e.g. '.tar' or '.tar.gz' (default: '.tar'). "
            "The base pattern is 'shard-%06d', so filenames look like "
            "'shard-000000.tar'."
        ),
    )
    parser.add_argument(
        "--maxcount",
        type=int,
        default=4000,
        help="Maximum number of samples per shard (default: 10000).",
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
            "Optional random seed for shuffling. If not set, a different "
            "shuffle order will be used each run."
        ),
    )
    return parser.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def gather_frame_entries(
    frames_root: Path, pattern: str
) -> List[Tuple[str, int, Path]]:
    """Return a list of (video_id, frame_index, frame_path) for all frames.

    The `frame_index` is 1-based and reflects the order of sorted filenames
    within each video directory. This captures the "origin frame order" per
    video while still allowing us to globally shuffle the resulting list.
    """

    if not frames_root.exists() or not frames_root.is_dir():
        raise FileNotFoundError(
            f"Frames root directory does not exist or is not a directory: {frames_root}"
        )

    entries: List[Tuple[str, int, Path]] = []
    video_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])

    if not video_dirs:
        logging.warning("No subdirectories found under %s", frames_root)
        return entries

    for video_dir in video_dirs:
        video_id = video_dir.name
        frame_paths = sorted(video_dir.glob(pattern))
        if not frame_paths:
            logging.warning(
                "No frames matching pattern '%s' found in %s; skipping.",
                pattern,
                video_dir,
            )
            continue

        for idx, frame_path in enumerate(frame_paths, start=1):
            entries.append((video_id, idx, frame_path))

    return entries


def write_global_shards(
    entries: List[Tuple[str, int, Path]],
    frames_root: Path,
    output_dir: Path,
    shard_suffix: str,
    maxcount: int,
    overwrite: bool,
    compression: str,
) -> int:
    """Write globally shuffled entries into WebDataset shards.

    Shards are named using the pattern `shard-%06d<shard_suffix>`, for example
    `shard-000000.tar`.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle existing shards.
    existing_shards = sorted(output_dir.glob(f"shard-*{shard_suffix}"))
    if existing_shards:
        if not overwrite:
            logging.info(
                "Found %d existing shards matching 'shard-*%s'; "
                "use --overwrite to regenerate.",
                len(existing_shards),
                shard_suffix,
            )
            return 0

        logging.info(
            "Overwriting %d existing shards matching 'shard-*%s'.",
            len(existing_shards),
            shard_suffix,
        )
        for shard in existing_shards:
            shard.unlink()

    writer_opts = {}
    if compression:
        writer_opts["compress"] = compression

    # Pattern for ShardWriter; it will expand %06d automatically.
    pattern_str = str(output_dir / ("shard-%06d" + shard_suffix))

    total_written = 0
    with wds.ShardWriter(pattern_str, maxcount=maxcount, **writer_opts) as sink:
        for video_id, frame_index, frame_path in entries:
            key = f"{video_id}_{frame_index:06d}"
            with frame_path.open("rb") as f:
                img_bytes = f.read()
            sample = {
                "__key__": key,
                "jpg": img_bytes,
            }

            sink.write(sample)
            total_written += 1

    logging.info(
        "Done writing globally shuffled shards. Total samples written: %d",
        total_written,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)

    frames_root = Path(args.frames_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    pattern = args.pattern
    shard_suffix = args.shard_suffix
    maxcount = args.maxcount
    overwrite = args.overwrite
    compression = args.compression
    seed = args.seed

    logging.info("Frames root: %s", frames_root)
    logging.info("Output dir: %s", output_dir)
    logging.info("Pattern: %s", pattern)
    logging.info("Shard suffix: %s", shard_suffix)
    logging.info("Max samples per shard: %d", maxcount)
    if seed is not None:
        logging.info("Shuffle seed: %d", seed)
    if compression:
        logging.info("Compression: %s", compression)

    try:
        entries = gather_frame_entries(frames_root, pattern)
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1

    if not entries:
        logging.warning(
            "No frames found under %s matching pattern '%s'", frames_root, pattern
        )
        return 0

    # Shuffle all entries globally before writing any shards.
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(entries)
    else:
        random.shuffle(entries)

    return write_global_shards(
        entries=entries,
        frames_root=frames_root,
        output_dir=output_dir,
        shard_suffix=shard_suffix,
        maxcount=maxcount,
        overwrite=overwrite,
        compression=compression,
    )


if __name__ == "__main__":
    import sys

    sys.exit(main())
