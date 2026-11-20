"""
Hold description agent (set-of-marks, pass 2 of 2) — one Claude batch request per video,
dispatched together by agents.base.run_batch_stage. Its output also feeds the global hold
catalog.

detect_holds already decided which marks are holds and how they group; this pass judges
the fields for each detected hold with the whole wall at one scale: the clean plate plus
the standard overlay tiles (marks.json "tiles", overlay px), redrawn with only the detected
holds outlined, one number per hold (its smallest mark id, at its largest mark's label
point), each tile followed by the same tile without outlines.

Input:
  processed/{video_id}/holds/proposals/{plate.jpg, marks.json, masks.npz}
  processed/{video_id}/holds/hold_detections.json     (detect_holds)
Output:
  processed/{video_id}/holds/hold_labels.json        HoldLabelResponse: detections joined
                                                     with descriptions
  processed/{video_id}/holds/hold_descriptions.csv   HoldRecord rows; hold_id and bbox are
                                                     computed here from the marks, on_route /
                                                     mounted_on stay empty (extract_hold_contacts
                                                     writes those to hold_contacts.csv)
Every detected hold must be described (a truncated answer fails the video, retried next run).
"""

from __future__ import annotations

import csv
import io
import json

import cv2
import numpy as np

from agents.base import image_block, output_config, parse_structured
from agents.extract_hold_descriptions.prompts import SYSTEM_PROMPT, build_hold_description_prompt
from common.overlay import paste_marks, render_overlay
from common.proposals import load_proposals
from common.s3 import download_bytes, get_client
from common.schemas import (HoldDescriptionResponse, HoldDetections, HoldLabel, HoldLabelResponse,
                            HoldRecord)


def _holds_prefix(video_id: str, config: dict) -> str:
    return f"{config['s3']['prefixes']['processed']}{video_id}/holds/"


def load_detections(video_id: str, config: dict) -> HoldDetections:
    s3 = get_client(config["s3"]["region"])
    return HoldDetections.model_validate_json(download_bytes(
        s3, config["s3"]["bucket"], _holds_prefix(video_id, config) + "hold_detections.json"))


def detected_overlay_tiles(marks: dict, masks: dict[int, np.ndarray], plate_jpg: bytes,
                           holds: list[list[int]]) -> list[tuple[bytes, bytes]]:
    """[(overlay tile PNG, clean tile JPEG)] over marks["tiles"], only detected holds drawn."""
    plate = cv2.imdecode(np.frombuffer(plate_jpg, np.uint8), cv2.IMREAD_COLOR)
    ow, oh = marks["overlay_size"]
    u = ow / marks["image_size"][0]                     # plate px -> overlay px
    big = cv2.resize(plate, (ow, oh), interpolation=cv2.INTER_CUBIC)
    regions = []
    for g in holds:
        bbox, crop = paste_marks(marks["marks"], masks, g, u)
        biggest = max(g, key=lambda m: marks["marks"][str(m)]["area"])
        regions.append((g[0], {"bbox": bbox, "crop": crop,
                               "label_point": tuple(marks["marks"][str(biggest)]["label_point"])}))
    overlay = render_overlay(big, regions)
    return [(cv2.imencode(".png", overlay[y0:y1, x0:x1])[1].tobytes(),
             cv2.imencode(".jpg", big[y0:y1, x0:x1], [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tobytes())
            for x0, y0, x1, y1 in marks["tiles"]]


def build_request(video_id: str, marks: dict, masks: dict[int, np.ndarray], plate_jpg: bytes,
                  detections: HoldDetections, config: dict) -> dict:
    """Messages API params from loaded proposals + detections (also used by the benchmark)."""
    cfg = config["agents"]["extract_hold_descriptions"]
    holds = sorted((sorted(g) for g in detections.holds), key=min)
    if not holds:
        raise ValueError("no detected holds to describe")
    tiles = detected_overlay_tiles(marks, masks, plate_jpg, holds)
    content = [image_block(plate_jpg, "image/jpeg")]
    for overlay_png, clean_jpg in tiles:
        content += [image_block(overlay_png), image_block(clean_jpg, "image/jpeg")]
    content.append({"type": "text", "text": build_hold_description_prompt(video_id, len(tiles), holds)})
    return {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content}],
        "output_config": output_config(HoldDescriptionResponse, cfg.get("effort")),
    }


def join(detections: HoldDetections, resp: HoldDescriptionResponse) -> HoldLabelResponse:
    """One HoldLabel per detected hold; raises if any detected hold went undescribed."""
    by_number: dict[int, dict] = {}
    for d in resp.holds:
        by_number.setdefault(d.hold, d.model_dump(exclude={"hold"}))   # first entry wins
    holds = [sorted(g) for g in detections.holds]
    missing = [g[0] for g in holds if g[0] not in by_number]
    if missing:
        raise ValueError(f"{len(missing)} of {len(holds)} detected holds not described: {missing[:10]}")
    return HoldLabelResponse(holds=[HoldLabel(mark_ids=g, **by_number[g[0]]) for g in holds],
                             non_hold_marks=detections.non_hold_marks)


def _union_bbox(boxes: list[list[int]]) -> list[int]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def build_records(labels: HoldLabelResponse, marks: dict[int, dict]) -> list[HoldRecord]:
    """Assign stable hold_ids (ordered by smallest mark id) and bboxes from the marks."""
    labels.check_coverage(set(marks))
    ordered = sorted(labels.holds, key=lambda h: min(h.mark_ids))
    return [
        HoldRecord(
            hold_id=f"h{i:03d}",
            mark_ids=sorted(h.mark_ids),
            bbox=_union_bbox([marks[m]["bbox"] for m in h.mark_ids]),
            **h.model_dump(exclude={"mark_ids"}),
        )
        for i, h in enumerate(ordered)
    ]


def records_to_csv(records: list[HoldRecord]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(HoldRecord.model_fields))
    writer.writeheader()
    for r in records:
        row = r.model_dump(mode="json")
        row["mark_ids"] = json.dumps(row["mark_ids"])
        row["bbox"] = json.dumps(row["bbox"])
        writer.writerow(row)
    return buf.getvalue()


def build_params(video_id: str, config: dict) -> dict:
    marks, masks, plate_jpg = load_proposals(video_id, config)
    return build_request(video_id, marks, masks, plate_jpg, load_detections(video_id, config), config)


def handle_result(video_id: str, message, config: dict) -> None:
    labels = join(load_detections(video_id, config), parse_structured(message, HoldDescriptionResponse))
    marks, _, _ = load_proposals(video_id, config, masks=False, plate=False)
    records = build_records(labels, {int(k): v for k, v in marks["marks"].items()})

    s3 = get_client(config["s3"]["region"])
    bucket, prefix = config["s3"]["bucket"], _holds_prefix(video_id, config)
    s3.put_object(Bucket=bucket, Key=prefix + "hold_labels.json",
                  Body=labels.model_dump_json(indent=2).encode(), ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=prefix + "hold_descriptions.csv",
                  Body=records_to_csv(records).encode(), ContentType="text/csv")
    print(f"[{video_id}] {len(records)} holds described, {len(labels.non_hold_marks)} non-hold marks")
