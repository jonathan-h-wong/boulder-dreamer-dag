"""
CPU tests for the plate + proposal post-processing and the benchmark scorer (no GPU, no
S3, no SAM2: SAM2's output is faked as a list of mask dicts).

    python tests/test_hold_proposals.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.masks import decode_masks, encode_masks, full_mask                        # noqa: E402
from jobs.extract_bkg_plate.logic import median_plate, stride_for                     # noqa: E402
from jobs.extract_hold_proposals import logic as hp                                   # noqa: E402
from scripts.benchmark_set_of_marks import coverage, score                            # noqa: E402

failures = 0


def check(label, cond):
    global failures
    print(("PASS " if cond else "FAIL ") + label)
    failures += not cond


# --- plate ------------------------------------------------------------------
check("stride: rate when short", stride_for(300, 10, 64) == 10)
check("stride: grows to cap frames", stride_for(3000, 10, 64) == 47)
wall = np.full((40, 30, 3), 100, np.uint8)
frames = [wall.copy() for _ in range(9)]
for i in range(4):                       # a 'climber' in a different spot in 4 of 9 frames
    frames[i][i * 8:(i + 1) * 8, 5:15] = 255
check("median removes a moving climber", np.array_equal(median_plate(frames), wall))


# --- proposals --------------------------------------------------------------
def rect(h, w, x0, y0, x1, y1, score=0.9):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return {"segmentation": m, "predicted_iou": score}


H, W = 200, 300
masks = [
    rect(H, W, 10, 10, 30, 30, 0.95),     # A
    rect(H, W, 11, 10, 31, 30, 0.90),     # near-duplicate of A -> NMS drops it
    rect(H, W, 200, 12, 240, 40, 0.9),    # B: same row band as A, to its right
    rect(H, W, 50, 120, 120, 180, 0.9),   # volume
    rect(H, W, 70, 140, 90, 160, 0.9),    # hold on the volume -> kept (small IoU)
    rect(H, W, 0, 0, 300, 200, 0.99),     # whole image -> above max area
    rect(H, W, 150, 100, 152, 102, 0.9),  # speck -> below min area
]
props = hp.to_proposals(masks, (H, W), min_area=20, max_area_frac=0.2)
check("area filter drops speck and whole-image mask", len(props) == 5)
kept = hp.mask_nms(props, 0.7)
check("NMS drops the near-duplicate, keeps hold-on-volume", len(kept) == 4)
ordered = hp.order_marks(kept, H)
cents = [tuple(round(c) for c in p["centroid"]) for p in ordered]
check("order: top row left to right, then down", cents[0][0] < cents[1][0] and cents[0][1] < 50
      and cents[1][1] < 50 and min(c[1] for c in cents[2:]) > 100)

c = np.zeros((20, 20), bool); c[2:18, 2:6] = True; c[2:6, 2:18] = True; c[14:18, 2:18] = True
x, y = hp.interior_point(c)
check("interior point lies inside a C shape", c[y, x])

boxes = hp.tile_boxes(720, 1280, 1024, 128)
check("tiles: 720x1280 -> 2 full-width tiles", boxes == [[0, 0, 720, 1024], [0, 256, 720, 1280]])
boxes = hp.tile_boxes(1080, 1920, 1024, 128)
check("tiles: 1080x1920 -> 2x2 covering the image", len(boxes) == 4
      and max(b[2] for b in boxes) == 1080 and max(b[3] for b in boxes) == 1920)
ok = all(min(b1[2], b2[2]) - max(b1[0], b2[0]) >= 128 or min(b1[3], b2[3]) - max(b1[1], b2[1]) >= 128
         for b1 in boxes for b2 in boxes if b1 != b2 and (b1[0] == b2[0] or b1[1] == b2[1]))
check("tiles: neighbours overlap by >= tile_overlap_px", ok)

cfg = {"min_mask_area_px": 20, "max_mask_area_frac": 0.2, "nms_mask_iou": 0.7,
       "tile_px": 128, "tile_overlap_px": 32, "overlay_long_side_px": 0}
img = np.full((H, W, 3), 90, np.uint8)
marks_json, crops, overlay, tiles = hp.build_outputs(img, masks, cfg, img, 1.0)
check("build_outputs: 4 marks numbered 1..4", sorted(crops) == [1, 2, 3, 4]
      and sorted(marks_json["marks"]) == ["1", "2", "3", "4"])
check("overlay drawn (differs from plate), tiles cover overlay",
      not np.array_equal(overlay, img) and len(tiles) == len(marks_json["tiles"]) > 1)
m1 = marks_json["marks"]["1"]
check("marks.json bbox/area in plate px", m1["bbox"] == [10, 10, 30, 30] and m1["area"] == 400)

# Downscaled processing (4K plate): bbox/crops come back at plate resolution
plate = np.zeros((2 * H, 2 * W, 3), np.uint8)
mj2, crops2, _, _ = hp.build_outputs(plate, masks, cfg, img, 0.5)
b = mj2["marks"]["1"]["bbox"]
check("scale 0.5: bbox and crop at plate resolution",
      b == [20, 20, 60, 60] and crops2[1].shape == (40, 40) and mj2["image_size"] == [600, 400])

mj3, crops3, ov3, _ = hp.build_outputs(img, masks, {**cfg, "overlay_long_side_px": 600}, img, 1.0)
check("overlay upscaled 2x; plate-space outputs unchanged",
      ov3.shape[:2] == (400, 600) and mj3["overlay_size"] == [600, 400]
      and mj3["marks"]["1"]["bbox"] == [10, 10, 30, 30]
      and mj3["marks"]["1"]["label_point"] == [2 * c for c in marks_json["marks"]["1"]["label_point"]])

rt = decode_masks(encode_masks(crops))
check("masks.npz round trip", all(np.array_equal(rt[k], crops[k]) for k in crops))
fm = full_mask(rt[1], m1["bbox"], marks_json["image_size"])
check("full_mask pastes at bbox", fm.sum() == 400 and fm[10:30, 10:30].all())


# --- benchmark scoring --------------------------------------------------------
def hold(ids, t="crimp"):
    return {"mark_ids": ids, "hold_type": t, "outline": "o", "profile": "p", "lip": "l",
            "opening": "o", "symmetry": "s", "relative_size": "r", "depth_class": "d",
            "description": ""}


ids = {1, 2, 3, 4, 5}
ref = {"holds": [hold([1, 2]), hold([3], "jug")], "non_hold_marks": [4, 5]}
check("identical answer scores 1.0", (lambda s: s["hold_f1"] == 1 and s["group_f1"] == 1
      and s["attr_accuracy"]["hold_type"] == 1)(score(ref, ref, ids)))
pred = {"holds": [hold([1]), hold([2]), hold([3]), hold([4])], "non_hold_marks": [5]}
s = score(pred, ref, ids)
check("split hold -> group F1 0; extra hold -> precision 3/4",
      s["group_f1"] == 0 and s["hold_precision"] == 0.75 and s["hold_recall"] == 1.0)
check("type mismatch counted per mark", s["attr_accuracy"]["hold_type"] == 2 / 3)
check("coverage counts missing/unknown/duplicated",
      coverage({"holds": [hold([1, 9])], "non_hold_marks": [1]}, ids)
      == {"missing": 4, "unknown": 1, "duplicated": 1})

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
