"""
Set-of-marks drawing, shared by extract_hold_proposals (the numbered overlay of every
proposal) and extract_hold_descriptions (the overlay of only the detected holds), so both
look the same to the model.

A drawable region is (number, {"bbox": [x0, y0, x1, y1], "crop": bool mask of bbox size,
"label_point": (x, y)}) in the target image's pixels.
"""

from __future__ import annotations

import cv2
import numpy as np

PALETTE = [(255, 64, 64), (64, 255, 64), (64, 160, 255), (255, 255, 0), (255, 0, 255),
           (0, 255, 255), (255, 160, 0), (160, 64, 255)]        # BGR, bright on walls
LABEL_FONT_SCALE, LABEL_THICKNESS, CONTOUR_PX = 0.55, 1, 2   # fixed overlay pixels


def render_overlay(img: np.ndarray, marks: list[tuple[int, dict]]) -> np.ndarray:
    """Contour of every region, then its number on a white box (labels on top of all contours)."""
    out = img.copy()
    h, w = img.shape[:2]
    for mid, p in marks:
        x0, y0 = p["bbox"][:2]
        contours, _ = cv2.findContours(p["crop"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, PALETTE[mid % len(PALETTE)], CONTOUR_PX, offset=(x0, y0))
    for mid, p in marks:
        text = str(mid)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS)
        cx, cy = p["label_point"]
        bx0 = int(np.clip(cx - tw // 2 - 2, 0, w - tw - 4))
        by0 = int(np.clip(cy - th // 2 - 2, 0, h - th - base - 4))
        cv2.rectangle(out, (bx0, by0), (bx0 + tw + 4, by0 + th + base + 4), (255, 255, 255), -1)
        cv2.putText(out, text, (bx0 + 2, by0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
                    (0, 0, 0), LABEL_THICKNESS, cv2.LINE_AA)
    return out


def paste_marks(marks: dict, masks: dict[int, np.ndarray], mark_ids, scale: float):
    """Union of several marks' bbox crops (plate px) as one region scaled by `scale`:
    -> (bbox [x0, y0, x1, y1], bool crop) in the scaled image."""
    boxes = [marks[str(m)]["bbox"] for m in mark_ids]
    bx = [round(min(b[0] for b in boxes) * scale), round(min(b[1] for b in boxes) * scale),
          round(max(b[2] for b in boxes) * scale), round(max(b[3] for b in boxes) * scale)]
    canvas = np.zeros((max(1, bx[3] - bx[1]), max(1, bx[2] - bx[0])), np.uint8)
    for m in mark_ids:
        b = marks[str(m)]["bbox"]
        ox, oy = round(b[0] * scale) - bx[0], round(b[1] * scale) - bx[1]
        mw, mh = max(1, round((b[2] - b[0]) * scale)), max(1, round((b[3] - b[1]) * scale))
        mk = cv2.resize(masks[m].astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST)
        ys = slice(max(0, oy), min(canvas.shape[0], oy + mh))
        xs = slice(max(0, ox), min(canvas.shape[1], ox + mw))
        if ys.start < ys.stop and xs.start < xs.stop:
            canvas[ys, xs] |= mk[ys.start - oy: ys.stop - oy, xs.start - ox: xs.stop - ox]
    return bx, canvas.astype(bool)
