#!/usr/bin/env python3
"""
Extract frames from videos using ffmpeg.

For each .mp4 file in the input directory, this script creates a subdirectory
under the output root and writes frames at the specified FPS, preserving the
original resolution (no scaling filter applied).

Example:
    python utility/extract_frames.py \
        --input_dir /bd_byt4090i1/users/dataset/cholec80/videos \
        --output_root /bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/data/cholec80/frames \
        --fps 1
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from joblib import Parallel, delayed

DEFAULT_INPUT_DIR = "/bd_byt4090i1/users/dataset/cholec80/videos/"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from .mp4 videos using ffmpeg."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing input .mp4 videos (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory where extracted frames will be stored.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1.0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frame directories if they already exist.",
    )
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="Path to ffmpeg executable (default: 'ffmpeg').",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers to use when extracting frames (default: 4).",
    )
    return parser.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_videos(input_dir: Path) -> List[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist or is not a directory: {input_dir}"
        )
    videos = sorted(input_dir.glob("*.mp4"))
    return videos


def ensure_ffmpeg_available(ffmpeg_bin: str) -> None:
    # If user provided a path, trust it; otherwise, check PATH.
    if ffmpeg_bin == "ffmpeg":
        if shutil.which(ffmpeg_bin) is None:
            raise RuntimeError(
                "ffmpeg executable not found in PATH. "
                "Install ffmpeg or provide a path using --ffmpeg."
            )


def run_ffmpeg(
    ffmpeg_bin: str,
    input_video: Path,
    output_dir: Path,
    fps: float,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Output filenames: frame_000001.jpg, frame_000002.jpg, ...
    output_pattern = str(output_dir / "frame_%06d.jpg")

    ffmpeg_cmd = [
        ffmpeg_bin,
        "-i",
        str(input_video),
        "-vf",
        f"fps={fps}",
        "-qscale:v",
        "2",  # quality parameter for JPEG
        output_pattern,
    ]

    # If overwrite: remove existing matching frames before running ffmpeg
    if overwrite:
        for existing in output_dir.glob("frame_*.jpg"):
            existing.unlink()

    logging.info("Running ffmpeg for %s", input_video.name)
    logging.debug("Command: %s", " ".join(ffmpeg_cmd))

    try:
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        if result.stdout:
            logging.debug("ffmpeg stdout for %s:\n%s", input_video.name, result.stdout)
        if result.stderr:
            logging.debug("ffmpeg stderr for %s:\n%s", input_video.name, result.stderr)
    except subprocess.CalledProcessError as e:
        logging.error(
            "ffmpeg failed for %s with return code %s",
            input_video.name,
            e.returncode,
        )
        if e.stderr:
            logging.error("ffmpeg stderr:\n%s", e.stderr)
        raise


def extract_frames(
    input_dir: Path,
    output_root: Path,
    ffmpeg_bin: str,
    fps: float,
    overwrite: bool,
    max_workers: int = 4,
) -> int:
    """Extract frames for all videos under input_dir in parallel.

    Behavior matches the original sequential loop but uses joblib-powered
    multiprocessing to run multiple ffmpeg commands concurrently.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        videos = find_videos(input_dir)
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1

    if not videos:
        logging.warning("No .mp4 videos found in %s", input_dir)
        return 0

    processed = 0
    skipped = 0
    failed = 0

    # Helper to keep per-video logic identical to the previous implementation.
    def _process_single_video(video_path: Path) -> tuple[bool, bool]:
        video_id = video_path.stem  # filename without extension
        video_output_dir = output_root / video_id

        if (
            video_output_dir.exists()
            and any(video_output_dir.iterdir())
            and not overwrite
        ):
            logging.info(
                "Output directory %s already exists and is non-empty; skipping %s "
                "(use --overwrite to re-generate).",
                video_output_dir,
                video_path.name,
            )
            return False, False  # skipped, not failed

        try:
            run_ffmpeg(
                ffmpeg_bin=ffmpeg_bin,
                input_video=video_path,
                output_dir=video_output_dir,
                fps=fps,
                overwrite=overwrite,
            )
            return True, False  # processed, not failed
        except Exception:
            return False, True  # not processed, failed

    # Run extraction in parallel across videos using joblib for robust
    # multi-process execution with consistent error handling semantics.
    results = Parallel(n_jobs=max_workers, prefer="processes")(
        delayed(_process_single_video)(video_path) for video_path in videos
    )

    for processed_flag, failed_flag in results:
        if processed_flag:
            processed += 1
        elif failed_flag:
            failed += 1
        else:
            skipped += 1

    logging.info(
        "Done extracting frames. Processed: %d, Skipped: %d, Failed: %d",
        processed,
        skipped,
        failed,
    )
    return 0 if failed == 0 else 1


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)

    input_dir = Path(args.input_dir).expanduser()
    output_root = Path(args.output_root).expanduser()
    ffmpeg_bin = args.ffmpeg
    fps = args.fps
    overwrite = args.overwrite
    num_workers = args.num_workers

    logging.info("Input directory: %s", input_dir)
    logging.info("Output root: %s", output_root)
    logging.info("FPS: %s", fps)
    logging.info("ffmpeg binary: %s", ffmpeg_bin)
    logging.info("Num workers: %s", num_workers)

    try:
        ensure_ffmpeg_available(ffmpeg_bin)
    except RuntimeError as e:
        logging.error(str(e))
        return 1

    # Use CLI-provided parallelism; callers can override by importing and
    # calling extract_frames directly.
    return extract_frames(
        input_dir=input_dir,
        output_root=output_root,
        ffmpeg_bin=ffmpeg_bin,
        fps=fps,
        overwrite=overwrite,
        max_workers=num_workers,
    )


if __name__ == "__main__":
    sys.exit(main())
