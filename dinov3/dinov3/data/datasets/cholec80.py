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
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )

        frames_root = os.path.join(self.root, "frames")
        video_dirnames = sorted(os.listdir(frames_root))

        image_paths: List[str] = []
        for video_dirname in video_dirnames:
            video_root = os.path.join(frames_root, video_dirname)
            frame_filenames = sorted(os.listdir(video_root))
            for frame_filename in frame_filenames:
                image_paths.append(os.path.join("frames", video_dirname, frame_filename))

        self.image_paths = image_paths

    def get_image_data(self, index: int) -> bytes:
        image_relpath = self.image_paths[index]
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        return None

    def __len__(self) -> int:
        return len(self.image_paths)

