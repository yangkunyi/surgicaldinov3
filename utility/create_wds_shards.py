#!/usr/bin/env python3
"""
Create WebDataset shards from extracted video frames.

For each subdirectory under --frames_root (one per video), this script creates
a single WebDataset shard (tar file) containing all frames matching the given
pattern.

Example:
    python utility/create_wds_shards.py \
        --frames_root /bd_byt4090i1/users/dataset/cholec80/frames \
        --output_dir /bd_byt4090i1/users/dataset/cholec80/shards \
        --pattern "frame_*.jpg" \
        --shard_suffix ".tar"

Each shard will be named <video_id><shard_suffix>, e.g. VIDEO001.tar, and will
contain samples with keys like VIDEO001_000001, VIDEO001_000002, etc.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import webdataset as wds


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create WebDataset shards from extracted video frames."
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        required=True,
        help="Root directory containing per-video frame subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where WebDataset shards will be written.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="frame_*.jpg",
        help='Glob pattern for frames inside each video directory (default: "frame_*.jpg").',
    )
    parser.add_argument(
        "--shard_suffix",
        type=str,
        default=".tar",
        help='Shard filename suffix, e.g. ".tar" or ".tar.gz" (default: ".tar").',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard files if they already exist.",
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
    return parser.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_video_dirs(frames_root: Path) -> List[Path]:
    if not frames_root.exists() or not frames_root.is_dir():
        raise FileNotFoundError(f"Frames root directory does not exist or is not a directory: {frames_root}")
    dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])
    return dirs


def create_shard_for_video(
    video_dir: Path,
    output_dir: Path,
    pattern: str,
    shard_suffix: str,
    overwrite: bool,
    compression: str,
) -> None:
    video_id = video_dir.name
    shard_name = f"{video_id}{shard_suffix}"
    shard_path = output_dir / shard_name

    frame_paths = sorted(video_dir.glob(pattern))
    if not frame_paths:
        logging.warning(
            "No frames matching pattern '%s' found in %s; skipping.",
            pattern,
            video_dir,
        )
        return

    if shard_path.exists() and not overwrite:
        logging.info(
            "Shard %s already exists; skipping (use --overwrite to regenerate).",
            shard_path,
        )
        return

    # If overwriting, remove old shard first to avoid mixing content.
    if shard_path.exists():
        shard_path.unlink()

    logging.info(
        "Creating shard %s from %d frames in %s",
        shard_path.name,
        len(frame_paths),
        video_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    writer_opts = {}
    if compression:
        writer_opts["compress"] = compression

    # ShardWriter will create exactly one shard at shard_path since we pass a
    # concrete filename and set maxcount to len(frame_paths) (plus metadata).
    pattern_str = str(shard_path)
    with wds.ShardWriter(pattern_str, maxcount=len(frame_paths) + 1, **writer_opts) as sink:
        for idx, frame_path in enumerate(frame_paths, start=1):
            key = f"{video_id}_{idx:06d}"
            with frame_path.open("rb") as f:
                img_bytes = f.read()
            sample = {
                "__key__": key,
                "jpg": img_bytes,
            }
            sink.write(sample)

        # Optional: write a small metadata sample per video.
        meta_key = f"{video_id}_meta"
        metadata = {
            "video_id": video_id,
            "num_frames": len(frame_paths),
            "pattern": pattern,
        }
        sink.write(
            {
                "__key__": meta_key,
                "json": json.dumps(metadata).encode("utf-8"),
            }
        )


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)

    frames_root = Path(args.frames_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    pattern = args.pattern
    shard_suffix = args.shard_suffix
    overwrite = args.overwrite
    compression = args.compression

    logging.info("Frames root: %s", frames_root)
    logging.info("Output dir: %s", output_dir)
    logging.info("Pattern: %s", pattern)
    logging.info("Shard suffix: %s", shard_suffix)
    if compression:
        logging.info("Compression: %s", compression)

    try:
        video_dirs = find_video_dirs(frames_root)
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1

    if not video_dirs:
        logging.warning("No subdirectories found under %s", frames_root)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0

    for video_dir in video_dirs:
        shard_name = f"{video_dir.name}{shard_suffix}"
        shard_path = output_dir / shard_name

        if shard_path.exists() and not overwrite:
            logging.info(
                "Shard %s already exists; skipping (use --overwrite to regenerate).",
                shard_path,
            )
            skipped += 1
            continue

        try:
            create_shard_for_video(
                video_dir=video_dir,
                output_dir=output_dir,
                pattern=pattern,
                shard_suffix=shard_suffix,
                overwrite=overwrite,
                compression=compression,
            )
            processed += 1
        except Exception as e:
            logging.error("Failed to create shard for %s: %s", video_dir, e)
            failed += 1

    logging.info(
        "Done creating shards. Shards created: %d, Skipped: %d, Failed: %d",
        processed,
        skipped,
        failed,
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
