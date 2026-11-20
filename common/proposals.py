"""
Readers for extract_hold_proposals' outputs (processed/{video_id}/holds/proposals/), shared
by the two hold-perception agents and the set-of-marks benchmark.

  marks.json   {"image_size": [w, h], "overlay_size": [w, h], "tiles": [[x0, y0, x1, y1]],
                "marks": {"<id>": {"bbox", "area", "centroid", "label_point", "score"}}}
               bbox/area/centroid in plate px; label_point/tiles in overlay px
  masks.npz    per-mark bool crops at plate resolution (common/masks.py)
  plate.jpg    the plate at SAM's resolution
"""

from __future__ import annotations

import json

from common.masks import decode_masks
from common.s3 import download_bytes, get_client


def proposals_prefix(video_id: str, config: dict) -> str:
    return f"{config['s3']['prefixes']['processed']}{video_id}/holds/proposals/"


def load_proposals(video_id: str, config: dict, masks: bool = True, plate: bool = True):
    """-> (marks.json dict, {mark_id: bool crop} | None, plate.jpg bytes | None)"""
    s3, bucket = get_client(config["s3"]["region"]), config["s3"]["bucket"]
    base = proposals_prefix(video_id, config)
    marks = json.loads(download_bytes(s3, bucket, base + "marks.json"))
    return (marks,
            decode_masks(download_bytes(s3, bucket, base + "masks.npz")) if masks else None,
            download_bytes(s3, bucket, base + "plate.jpg") if plate else None)
