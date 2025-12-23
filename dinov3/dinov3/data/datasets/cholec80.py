# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Cholec80 dataset backed by frames on disk.

Expected directory structure under ``root``:

.. code-block:: text

    root/
      frames/
        video01/
          frame_000001.jpg
          frame_000002.jpg
          ...
        video02/
          ...

This is a standard map-style dataset (``__len__`` / ``__getitem__``) returning
``(image, target)`` pairs. The target is always ``None``.
"""

import os
import random
from typing import Any, Callable, List, Optional

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset


class Cholec80(ExtendedVisionDataset):
    def __init__(
        self,
        *,
        root: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_clip: bool = False,
        clip_max_frames: int = 8,
        clip_stride: int = 1,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )

        self.return_clip = return_clip
        self.clip_max_frames = clip_max_frames
        self.clip_stride = clip_stride

        frames_root = os.path.join(self.root, "frames")
        video_dirnames = sorted(os.listdir(frames_root))

        image_paths: List[str] = []
        global_to_video: List[int] = []
        global_to_frame: List[int] = []
        video_frame_filenames: List[List[str]] = []

        for video_idx, video_dirname in enumerate(video_dirnames):
            video_root = os.path.join(frames_root, video_dirname)
            frame_filenames = sorted(os.listdir(video_root))
            video_frame_filenames.append(frame_filenames)
            for frame_idx, frame_filename in enumerate(frame_filenames):
                image_paths.append(os.path.join("frames", video_dirname, frame_filename))
                global_to_video.append(video_idx)
                global_to_frame.append(frame_idx)

        self.image_paths = image_paths
        self._video_dirnames = video_dirnames
        self._video_frame_filenames = video_frame_filenames
        self._global_to_video = global_to_video
        self._global_to_frame = global_to_frame

    def get_image_data(self, index: int) -> bytes:
        image_relpath = self.image_paths[index]
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        return None

    def __getitem__(self, index: int):
        if not self.return_clip:
            return super().__getitem__(index)

        video_idx = self._global_to_video[index]
        frame_idx = self._global_to_frame[index]
        video_dirname = self._video_dirnames[video_idx]
        frame_filenames = self._video_frame_filenames[video_idx]

        n_frames = len(frame_filenames)
        stride = self.clip_stride
        clip_len = self.clip_max_frames

        max_start = n_frames - (clip_len - 1) * stride - 1
        anchor_min = (frame_idx - max_start + stride - 1) // stride
        if anchor_min < 0:
            anchor_min = 0
        anchor_max = frame_idx // stride
        if anchor_max > clip_len - 1:
            anchor_max = clip_len - 1

        if anchor_min > anchor_max:
            anchor = 0
            clip_frame_indices = [(frame_idx + i * stride) % n_frames for i in range(clip_len)]
        else:
            anchor = random.randint(anchor_min, anchor_max)
            start = frame_idx - anchor * stride
            clip_frame_indices = [start + i * stride for i in range(clip_len)]

        clip_relpaths = [
            os.path.join("frames", video_dirname, frame_filenames[frame_index]) for frame_index in clip_frame_indices
        ]

        image_data = self.get_image_data(index)
        image = self.image_decoder(image_data).decode()

        clip_images = []
        for relpath in clip_relpaths:
            with open(os.path.join(self.root, relpath), mode="rb") as f:
                clip_data = f.read()
            clip_images.append(self.image_decoder(clip_data).decode())

        image = {"image": image, "clip": clip_images, "mum_anchor": anchor}
        target = self.get_target(index)
        target = self.target_decoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.image_paths)
