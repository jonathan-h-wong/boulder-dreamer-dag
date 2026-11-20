"""
Hold-proposal mask storage (processed/{video_id}/holds/proposals/masks.npz).

Full-frame masks would be N x H x W (hundreds of marks x millions of pixels), so each
mask is stored cropped to its bbox:
    "<mark_id>"   bool array, shape (y1 - y0, x1 - x0)
and marks.json gives each mark's bbox [x0, y0, x1, y1] (exclusive x1/y1) in plate pixels.
"""

from __future__ import annotations

import io

import numpy as np


def encode_masks(crops: dict[int, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **{str(k): v.astype(bool) for k, v in crops.items()})
    return buf.getvalue()


def decode_masks(data: bytes) -> dict[int, np.ndarray]:
    with np.load(io.BytesIO(data)) as z:
        return {int(k): z[k] for k in z.files}


def full_mask(crop: np.ndarray, bbox, image_size) -> np.ndarray:
    """Paste a bbox crop into a full-frame bool mask. image_size = [w, h]."""
    w, h = image_size
    x0, y0, x1, y1 = bbox
    out = np.zeros((h, w), bool)
    out[y0:y1, x0:x1] = crop
    return out
