"""
Hold description agent (set-of-marks) — one Claude batch request per video, dispatched
together by agents.base.run_batch_stage. Its output also feeds the global hold catalog.

Input (from extract_hold_proposals):
  processed/{video_id}/bkg_rgb.png
  processed/{video_id}/holds/proposals/marks.json
      {"image_size": [w, h], "marks": {"<mark_id>": {"bbox": [x0, y0, x1, y1], "area": int,
                                                     "centroid": [x, y]}}}
  processed/{video_id}/holds/proposals/tiles/*.png   (numbered overlay tiles)
Output:
  processed/{video_id}/holds/hold_labels.json        validated HoldLabelResponse (raw judgment)
  processed/{video_id}/holds/hold_descriptions.csv   HoldRecord rows; hold_id and bbox are
                                                     computed here from the marks, on_route /
                                                     mounted_on stay empty (extract_hold_contacts
                                                     writes those to hold_contacts.csv)
"""

import base64
import csv
import io
import json

from agents.base import output_config, parse_structured
from agents.extract_hold_descriptions.prompts import SYSTEM_PROMPT, build_hold_description_prompt
from common.s3 import download_bytes, get_client, list_keys
from common.schemas import HoldLabelResponse, HoldRecord


def _holds_prefix(video_id: str, config: dict) -> str:
    return f"{config['s3']['prefixes']['processed']}{video_id}/holds/"


def _load_marks(video_id: str, config: dict) -> dict[int, dict]:
    s3 = get_client(config["s3"]["region"])
    raw = download_bytes(s3, config["s3"]["bucket"], _holds_prefix(video_id, config) + "proposals/marks.json")
    return {int(k): v for k, v in json.loads(raw)["marks"].items()}


def _image_block(data: bytes, media_type: str = "image/png") -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                        "data": base64.b64encode(data).decode()}}


def build_params(video_id: str, config: dict) -> dict:
    s3 = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    marks = _load_marks(video_id, config)
    tile_keys = sorted(k for k in list_keys(s3, bucket, _holds_prefix(video_id, config) + "proposals/tiles/")
                       if k.endswith(".png"))
    if not tile_keys:
        raise FileNotFoundError(f"no overlay tiles for {video_id}")
    plate = download_bytes(s3, bucket, f"{config['s3']['prefixes']['processed']}{video_id}/bkg_rgb.png")

    content = [_image_block(plate)]
    content += [_image_block(download_bytes(s3, bucket, k)) for k in tile_keys]
    content.append({"type": "text",
                    "text": build_hold_description_prompt(video_id, len(marks), len(tile_keys))})
    return {
        "model": config["agents"]["extract_hold_descriptions"]["model"],
        "max_tokens": config["agents"]["extract_hold_descriptions"]["max_tokens"],
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content}],
        "output_config": output_config(HoldLabelResponse, config["agents"]["extract_hold_descriptions"].get("effort")),
    }


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


def handle_result(video_id: str, message, config: dict) -> None:
    labels = parse_structured(message, HoldLabelResponse)
    records = build_records(labels, _load_marks(video_id, config))

    s3 = get_client(config["s3"]["region"])
    bucket, prefix = config["s3"]["bucket"], _holds_prefix(video_id, config)
    s3.put_object(Bucket=bucket, Key=prefix + "hold_labels.json",
                  Body=labels.model_dump_json(indent=2).encode(), ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=prefix + "hold_descriptions.csv",
                  Body=records_to_csv(records).encode(), ContentType="text/csv")
    print(f"[{video_id}] {len(records)} holds, {len(labels.non_hold_marks)} non-hold marks")
