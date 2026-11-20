"""
Hold detection agent (set-of-marks, pass 1 of 2) — one Claude batch request per video,
dispatched together by agents.base.run_batch_stage.

Each SAM2 proposal gets its own close-up, so small footholds fill the frame instead of
hiding under an overlay's outlines and numbers: a square window centered on the mark's
bbox (closeup_scale x its larger side, at least closeup_min_px plate.jpg pixels), enlarged
to closeup_px, shown as photographed and again with the mark outlined. Enlarging every mark
to one size removes its scale, so this pass only detects; extract_hold_descriptions judges
type and size on the whole plate at one scale.

Input (from extract_hold_proposals):
  processed/{video_id}/holds/proposals/{plate.jpg, marks.json, masks.npz}
Output:
  processed/{video_id}/holds/hold_detections.json    HoldDetections (every mark exactly once)

Coverage: an answer may skip a few marks (they become non-holds, listed as unaccounted);
more than max_missing_frac of them fails the video, which is retried next run. Unknown
numbers are dropped; a mark listed twice keeps its first hold (holds before non-holds).
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from agents.base import image_block, output_config, parse_structured
from agents.detect_holds.prompts import SYSTEM_PROMPT, build_detection_prompt
from common.proposals import load_proposals
from common.s3 import get_client
from common.schemas import HoldDetectionResponse, HoldDetections

OUTLINE_BGR = (255, 0, 255)


def closeup(plate: np.ndarray, padded: np.ndarray, pad: int, bbox: list[int], crop: np.ndarray,
            s: float, cfg: dict) -> bytes:
    """[as photographed | outlined] JPEG for one mark. bbox/crop are in plate px; s maps
    plate px to plate.jpg px; `padded` is plate.jpg with a `pad`-px dark border."""
    size, scale, min_px = cfg["closeup_px"], cfg["closeup_scale"], cfg["closeup_min_px"]
    x0, y0, x1, y1 = (c * s for c in bbox)
    side = int(round(max(min_px, scale * max(x1 - x0, y1 - y0))))
    wx, wy = int(round((x0 + x1 - side) / 2)), int(round((y0 + y1 - side) / 2))
    window = padded[wy + pad: wy + pad + side, wx + pad: wx + pad + side]
    # the mask, resized to plate.jpg px and placed in the window
    mw, mh = max(1, round((bbox[2] - bbox[0]) * s)), max(1, round((bbox[3] - bbox[1]) * s))
    mk = cv2.resize(crop.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((side, side), np.uint8)
    ox, oy = round(bbox[0] * s) - wx, round(bbox[1] * s) - wy
    ys, xs = slice(max(0, oy), min(side, oy + mh)), slice(max(0, ox), min(side, ox + mw))
    if ys.start < ys.stop and xs.start < xs.stop:
        canvas[ys, xs] = mk[ys.start - oy: ys.stop - oy, xs.start - ox: xs.stop - ox]
    clean = cv2.resize(window, (size, size), interpolation=cv2.INTER_CUBIC)
    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST)
    outlined = clean.copy()
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outlined, contours, -1, OUTLINE_BGR, 2)
    pair = np.hstack([clean, np.full((size, 6, 3), 255, np.uint8), outlined])
    return cv2.imencode(".jpg", pair, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes()


def build_request(video_id: str, marks: dict, masks: dict[int, np.ndarray], plate_jpg: bytes,
                  config: dict) -> dict:
    """Messages API params from loaded proposals (also used by the benchmark)."""
    cfg = config["agents"]["detect_holds"]
    plate = cv2.imdecode(np.frombuffer(plate_jpg, np.uint8), cv2.IMREAD_COLOR)
    h, w = plate.shape[:2]
    pw, ph = marks["image_size"]
    s = w / pw
    pad = max(h, w)
    padded = cv2.copyMakeBorder(plate, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    ids = sorted(int(k) for k in marks["marks"])
    content = [image_block(plate_jpg, "image/jpeg")]
    for mid in ids:
        m = marks["marks"][str(mid)]
        cx, cy = m["centroid"]
        content.append({"type": "text", "text": f"Mark {mid}, at {round(100 * cx / pw)}% across "
                                                f"and {round(100 * cy / ph)}% down the plate:"})
        content.append(image_block(closeup(plate, padded, pad, m["bbox"], masks[mid], s, cfg), "image/jpeg"))
    content.append({"type": "text", "text": build_detection_prompt(video_id, ids)})
    return {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content}],
        "output_config": output_config(HoldDetectionResponse, cfg.get("effort")),
    }


def resolve(resp: HoldDetectionResponse, mark_ids: set[int], max_missing_frac: float) -> HoldDetections:
    """Every mark exactly once; raises if the answer skipped too many marks."""
    seen: set[int] = set()
    holds = []
    for h in resp.holds:
        g = sorted({m for m in h.mark_ids if m in mark_ids} - seen)
        if g:
            holds.append(g)
            seen |= set(g)
    non_holds = sorted({m for m in resp.non_hold_marks if m in mark_ids} - seen)
    seen |= set(non_holds)
    missing = sorted(mark_ids - seen)
    if len(missing) > max(1, int(max_missing_frac * len(mark_ids))):
        raise ValueError(f"detection skipped {len(missing)} of {len(mark_ids)} marks")
    return HoldDetections(holds=sorted(holds, key=min), non_hold_marks=sorted(non_holds + missing),
                          unaccounted=missing)


def build_params(video_id: str, config: dict) -> dict:
    marks, masks, plate_jpg = load_proposals(video_id, config)
    return build_request(video_id, marks, masks, plate_jpg, config)


def handle_result(video_id: str, message, config: dict) -> None:
    marks, _, _ = load_proposals(video_id, config, masks=False, plate=False)
    det = resolve(parse_structured(message, HoldDetectionResponse), {int(k) for k in marks["marks"]},
                  config["agents"]["detect_holds"]["max_missing_frac"])
    s3 = get_client(config["s3"]["region"])
    key = f"{config['s3']['prefixes']['processed']}{video_id}/holds/hold_detections.json"
    s3.put_object(Bucket=config["s3"]["bucket"], Key=key, Body=det.model_dump_json(indent=2).encode(),
                  ContentType="application/json")
    print(f"[{video_id}] {len(det.holds)} holds detected, {len(det.non_hold_marks)} non-hold marks"
          + (f" ({len(det.unaccounted)} skipped)" if det.unaccounted else ""))
