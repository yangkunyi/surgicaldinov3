# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Cholec80 dataset implemented as a single WebDataset-backed class.

Frames are expected to have been extracted and sharded into tar files named
``shard-<NNNNNN>.tar`` (for example using ``utility/create_wds_shards_global.py``).

This module exposes one class, :class:`Cholec80`, which:
- builds a ``webdataset.WebDataset`` pipeline internally, and
- applies a user-provided ``transform`` inside that pipeline, so that each
  sample yielded by the dataset is a *transformed* image.

Example
-------

.. code-block:: python

    from dinov3.data.datasets.cholec80 import Cholec80
    from torchvision import transforms

    shards = \
        "/bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/" \
        "data/cholec80/shards/shard-{000001..000046}.tar"

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataset = Cholec80(shards=shards, transform=transform)

    for img in dataset:
        # img is already transformed
        ...
"""

from typing import Callable, Iterator, Literal, Optional

import webdataset as wds
from torch.utils.data import IterableDataset


DecodeMode = Literal["pil", "rgb"]


class Cholec80(IterableDataset):
    """IterableDataset wrapper for Cholec80 WebDataset shards.

    The dataset is backed by :class:`webdataset.WebDataset` and configured
    entirely from the constructor arguments. The provided ``transform`` is
    integrated into the WebDataset pipeline itself, so it is applied as part
    of the data reading/decoding stage.

    Each iteration yields a single (optionally transformed) image; there is no
    separate target.
    """

    def __init__(
        self,
        *,
        root: str,
        transform: Optional[Callable] = None,
        shuffle_buffer: int = 4000,
        shardshuffle: int = 42,        
        **kwargs
    ) -> None:
        """Initialize the Cholec80 dataset.

        Args:
            shards: Glob / brace pattern pointing to shard tars, e.g.
                ``".../shards/shard-{000001..000046}.tar"`` or
                ``".../shards/shard-*.tar"``.
            transform: Callable applied to each decoded image (e.g. torchvision
                transforms). If ``None``, images are yielded as decoded by
                WebDataset.
            decode: Either ``"pil"`` (PIL.Image) or ``"rgb"`` (NumPy RGB
                arrays) to use with ``WebDataset.decode``.
            shuffle_buffer: Per-worker sample shuffle buffer size.
            shardshuffle: Seed / flag controlling shard order shuffling in
                ``wds.WebDataset``.
        """

        super().__init__()
        # Base WebDataset pipeline: read shards, decode images, local shuffle,
        # and keep only the JPEG field as `(image,)`.
        ds = wds.WebDataset(root, shardshuffle=shardshuffle, resampled=True)
        ds = ds.decode("pil").shuffle(shuffle_buffer).to_tuple("jpg","__key__")

        # If a transform is provided, integrate it into the WebDataset pipeline
        # so that it runs as part of the iterable processing.
        if transform is not None:
            ds = ds.map_tuple(transform)

        self._dataset = ds

    def __iter__(self) -> Iterator:
        """Yield transformed images from the underlying WebDataset pipeline."""

        return iter(self._dataset)


