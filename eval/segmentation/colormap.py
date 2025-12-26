from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

COLOR_CLASS_MAPPING: Dict[Tuple[int, int, int], int] = {
    (127, 127, 127): 0,
    (210, 140, 140): 1,
    (255, 114, 114): 2,
    (231, 70, 156): 3,
    (186, 183, 75): 4,
    (170, 255, 0): 5,
    (255, 85, 0): 6,
    (255, 0, 0): 7,
    (255, 255, 0): 8,
    (169, 255, 184): 9,
    (255, 160, 165): 10,
    (0, 50, 128): 11,
    (111, 74, 0): 12,
}

CLASS_NAME_MAPPING: Dict[int, str] = {
    0: "Black Background",
    1: "Abdominal Wall",
    2: "Liver",
    3: "Gastrointestinal Tract",
    4: "Fat",
    5: "Grasper",
    6: "Connective Tissue",
    7: "Blood",
    8: "Cystic Duct",
    9: "L-hook Electrocautery",
    10: "Gallbladder",
    11: "Hepatic Vein",
    12: "Liver Ligament",
}


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def color_mask_to_class_index(
    mask: Image.Image, *, debug_save_path: str = ""
) -> np.ndarray:
    rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
    class_index = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, class_id in COLOR_CLASS_MAPPING.items():
        matches = (
            (rgb[:, :, 0] == color[0])
            & (rgb[:, :, 1] == color[1])
            & (rgb[:, :, 2] == color[2])
        )
        class_index[matches] = class_id
    unknown = class_index == 255
    unknown_count = int(unknown.sum())
    if unknown_count > 0:
        if len(debug_save_path) > 0:
            print(unknown_count)
            debug_path = Path(debug_save_path)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            overlay = rgb.copy()
            overlay[unknown] = np.asarray((255, 0, 255), dtype=np.uint8)
            Image.fromarray(overlay).save(debug_path)
    return class_index
