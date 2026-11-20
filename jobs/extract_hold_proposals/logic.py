"""
extract_hold_proposals logic — class-agnostic hold proposals for set-of-marks labeling.

Steps:
1. Download processed/{video_id}/bkg_rgb.png. SAM2 sees it downscaled to
   hold_proposals.max_side_px on the long side (only 4K plates are affected).
2. SAM2 automatic mask generation (points_per_side, crop_n_layers for small footholds,
   min_mask_region_area = min_mask_area_px removes specks/holes). Thresholds from config.
3. Drop masks below min_mask_area_px or above max_mask_area_frac of the image (wall,
   floor, mats); de-duplicate with greedy NMS on mask IoU (nms_mask_iou), best
   predicted IoU first. A hold on a volume survives: its IoU with the volume is small.
4. Number surviving masks 1..N, top-to-bottom then left-to-right by centroid (row bands
   of ROW_BAND_FRAC of the image height, so a row of holds reads left to right).
5. Render the overlay (mask contour + number on a white box at the mask's interior point =
   distance-transform maximum, so it lands inside even for C-shaped holds), upscaled so its
   long side is overlay_long_side_px while labels keep a fixed pixel size: on a 720p plate a
   label at native size covers a whole foothold. Cut it into overlapping tiles (tile_px,
   tile_overlap_px); the overlap exceeds the label size, so every label is whole in at
   least one tile.
6. Replace processed/{video_id}/holds/proposals/:
     masks.npz    per-mark bool crops at plate resolution (common/masks.py)
     marks.json   {"image_size": [w, h], "marks": {"<id>": {"bbox", "area", "centroid",
                   "label_point", "score"}}, "overlay_size": [w, h], "tiles": [[x0,y0,x1,y1]]}
                  bbox/area/centroid in plate pixels; label_point/tiles in overlay pixels
     marks.png    full overlay, for humans
     plate.jpg    the plate at SAM's resolution, for the agent (a 4K PNG exceeds the
                  API's 5 MB image limit)
     tiles/{i:02d}.png
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np

from common.masks import encode_masks
from common.overlay import render_overlay
from common.s3 import download_bytes, get_client, list_keys

ROW_BAND_FRAC = 0.04

_GENERATOR = None   # SAM2 generator, reused across calls in a warm container


# ---------------------------------------------------------------------------
# SAM2
# ---------------------------------------------------------------------------

def _generator(cfg: dict, checkpoint: str):
    global _GENERATOR
    if _GENERATOR is None:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        _GENERATOR = SAM2AutomaticMaskGenerator.from_pretrained(
            checkpoint,
            points_per_side=cfg["points_per_side"],
            points_per_batch=128,
            pred_iou_thresh=cfg["pred_iou_thresh"],
            stability_score_thresh=cfg["stability_score_thresh"],
            crop_n_layers=cfg["crop_n_layers"],
            min_mask_region_area=cfg["min_mask_area_px"],
            output_mode="binary_mask",
        )
    return _GENERATOR


def sam_masks(img_bgr: np.ndarray, cfg: dict, checkpoint: str) -> list[dict]:
    import torch
    gen = _generator(cfg, checkpoint)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return gen.generate(rgb)


# ---------------------------------------------------------------------------
# Pure post-processing (unit-tested without a GPU)
# ---------------------------------------------------------------------------

def downscale(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """-> (image, scale) with scale = processed px / plate px (<= 1)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return img, 1.0
    size = (round(w * scale), round(h * scale))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), scale


def to_proposals(masks: list[dict], image_shape, min_area: int, max_area_frac: float) -> list[dict]:
    """SAM2 output -> [{"crop", "bbox" [x0,y0,x1,y1) , "area", "score"}], filtered by area."""
    h, w = image_shape[:2]
    out = []
    for m in masks:
        seg = m["segmentation"]
        area = int(seg.sum())
        if area < min_area or area > max_area_frac * h * w:
            continue
        ys, xs = np.nonzero(seg.any(axis=1))[0], np.nonzero(seg.any(axis=0))[0]
        x0, y0, x1, y1 = int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1
        out.append({"crop": seg[y0:y1, x0:x1].copy(), "bbox": [x0, y0, x1, y1], "area": area,
                    "score": float(m.get("predicted_iou", 0.0))})
    return out


def mask_iou(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = a["bbox"]
    bx0, by0, bx1, by1 = b["bbox"]
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    ca = a["crop"][y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
    cb = b["crop"][y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    inter = int(np.logical_and(ca, cb).sum())
    return inter / (a["area"] + b["area"] - inter) if inter else 0.0


def mask_nms(props: list[dict], iou_thresh: float) -> list[dict]:
    kept: list[dict] = []
    for p in sorted(props, key=lambda p: (-p["score"], -p["area"])):
        if all(mask_iou(p, k) < iou_thresh for k in kept):
            kept.append(p)
    return kept


def interior_point(crop: np.ndarray) -> tuple[int, int]:
    """(x, y) in crop coords of the pixel farthest from the mask boundary."""
    padded = np.pad(crop.astype(np.uint8), 1)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x) - 1, int(y) - 1


def centroid(p: dict) -> tuple[float, float]:
    ys, xs = np.nonzero(p["crop"])
    return float(xs.mean() + p["bbox"][0]), float(ys.mean() + p["bbox"][1])


def order_marks(props: list[dict], image_h: int) -> list[dict]:
    band = max(1.0, ROW_BAND_FRAC * image_h)
    for p in props:
        p["centroid"] = centroid(p)
    return sorted(props, key=lambda p: (int(p["centroid"][1] // band), p["centroid"][0]))


def tile_boxes(w: int, h: int, tile: int, overlap: int) -> list[list[int]]:
    """Overlapping tiles covering [0, w) x [0, h), evenly spaced, row-major."""
    def starts(length):
        t = min(tile, length)
        n = 1 if length <= t else math.ceil((length - overlap) / (t - overlap))
        return t, ([0] if n == 1 else [round(i * (length - t) / (n - 1)) for i in range(n)])
    tw, xs = starts(w)
    th, ys = starts(h)
    return [[x, y, x + tw, y + th] for y in ys for x in xs]


def scale_proposal(p: dict, u: float, w: int, h: int) -> dict:
    """Copy of a proposal (crop/bbox/label_point) scaled by u into a w x h image."""
    if u == 1.0:
        return p
    x0, y0, x1, y1 = p["bbox"]
    bb = [math.floor(x0 * u), math.floor(y0 * u), min(w, math.ceil(x1 * u)), min(h, math.ceil(y1 * u))]
    crop = cv2.resize(p["crop"].astype(np.uint8), (bb[2] - bb[0], bb[3] - bb[1]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    lx, ly = p["label_point"]
    return {**p, "crop": crop, "bbox": bb, "label_point": (round(lx * u), round(ly * u))}


def to_plate_space(p: dict, scale: float, plate_w: int, plate_h: int) -> tuple[list[int], np.ndarray]:
    """Bbox + crop of a processed-space proposal at plate resolution."""
    if scale == 1.0:
        return list(p["bbox"]), p["crop"]
    x0, y0, x1, y1 = p["bbox"]
    bx = [math.floor(x0 / scale), math.floor(y0 / scale),
          min(plate_w, math.ceil(x1 / scale)), min(plate_h, math.ceil(y1 / scale))]
    crop = cv2.resize(p["crop"].astype(np.uint8), (bx[2] - bx[0], bx[3] - bx[1]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    return bx, crop


def build_outputs(plate: np.ndarray, masks: list[dict], cfg: dict, img: np.ndarray, scale: float):
    """-> (marks_json, mask_crops, overlay, tiles) from raw SAM2 masks in img space."""
    ph, pw = plate.shape[:2]
    h, w = img.shape[:2]
    props = to_proposals(masks, img.shape, cfg["min_mask_area_px"], cfg["max_mask_area_frac"])
    props = order_marks(mask_nms(props, cfg["nms_mask_iou"]), h)
    u = max(1.0, cfg["overlay_long_side_px"] / max(h, w))
    ow, oh = round(w * u), round(h * u)
    marks_json = {"image_size": [pw, ph], "overlay_size": [ow, oh], "marks": {}}
    crops: dict[int, np.ndarray] = {}
    numbered = []
    for mid, p in enumerate(props, start=1):
        ix, iy = interior_point(p["crop"])
        p["label_point"] = (p["bbox"][0] + ix, p["bbox"][1] + iy)
        numbered.append((mid, scale_proposal(p, u, ow, oh)))
        bbox, crop = to_plate_space(p, scale, pw, ph)
        crops[mid] = crop
        cx, cy = p["centroid"]
        marks_json["marks"][str(mid)] = {
            "bbox": bbox, "area": int(crop.sum()),
            "centroid": [round(cx / scale, 1), round(cy / scale, 1)],
            "label_point": list(numbered[-1][1]["label_point"]), "score": round(p["score"], 4),
        }
    big = img if u == 1.0 else cv2.resize(img, (ow, oh), interpolation=cv2.INTER_CUBIC)
    overlay = render_overlay(big, numbered)
    boxes = tile_boxes(ow, oh, cfg["tile_px"], cfg["tile_overlap_px"])
    marks_json["tiles"] = boxes
    tiles = [overlay[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes]
    return marks_json, crops, overlay, tiles


# ---------------------------------------------------------------------------
# Job entry
# ---------------------------------------------------------------------------

def _png(img) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def run(video_id: str, config: dict):
    s3 = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    cfg = config["hold_proposals"]
    base = f"{config['s3']['prefixes']['processed']}{video_id}/"
    out = base + "holds/proposals/"

    data = np.frombuffer(download_bytes(s3, bucket, base + "bkg_rgb.png"), np.uint8)
    plate = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if plate is None:
        raise ValueError("could not decode bkg_rgb.png")
    img, scale = downscale(plate, cfg["max_side_px"])
    masks = sam_masks(img, cfg, config["models"]["sam2"]["checkpoint"])
    marks_json, crops, overlay, tiles = build_outputs(plate, masks, cfg, img, scale)
    del masks
    if not crops:
        raise ValueError("SAM2 produced no proposals within the area limits")

    old = list_keys(s3, bucket, out)            # replace, never mix (tile count can shrink)
    if old:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in old]})
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    puts = [("masks.npz", encode_masks(crops), "application/octet-stream"),
            ("marks.png", _png(overlay), "image/png"),
            ("plate.jpg", jpg.tobytes(), "image/jpeg"),
            *((f"tiles/{i:02d}.png", _png(t), "image/png") for i, t in enumerate(tiles))]
    for name, body, ctype in puts:
        s3.put_object(Bucket=bucket, Key=out + name, Body=body, ContentType=ctype)
    # marks.json last: the agent reads it first, so it only exists with a complete set
    s3.put_object(Bucket=bucket, Key=out + "marks.json", Body=json.dumps(marks_json).encode(),
                  ContentType="application/json")
    print(f"[{video_id}] {len(crops)} marks, {len(tiles)} tiles, scale {scale:.3f}")
