"""
Filter Flow — decides which raw videos enter the dataset.

  1. extract_filter_frames  (Modal .map over all raw videos with climber height)
  2. filter_video           (one Claude batch over all videos with frames)
  3. aggregate              -> filtered_videos/filter_results.csv   (one row per video)
                               filtered_videos/passing_video_ids.json
                               filtered_videos/_runs/{utc_ts}.json (summary history)

Videos without climber height in raw/metadata/*.csv are rejected up front, without
spending a Claude request on them.

`sample=N` runs on a seeded random sample of N raw videos (e.g. a first pass to check
the filter, or a subset for the set-of-marks benchmark). A subset run updates only its
own rows of filter_results.csv; the passing list and the summary always cover the whole
table, i.e. every video filtered so far.
"""

from __future__ import annotations

import json
from collections import Counter
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from botocore.exceptions import ClientError
from prefect import flow, get_run_logger
from prefect.artifacts import create_markdown_artifact

from flows.filter_report import rows_from_csv, rows_to_csv, summarize, to_markdown
from common.s3 import download_bytes
from flows.runtime import (
    S3_WORKERS, build_context, load_config, publish_report, run_agent_stage, run_modal_stage,
)


def _get_json(ctx, key):
    try:
        return json.loads(download_bytes(ctx.s3, ctx.bucket, key))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


@flow(name="filter-videos", log_prints=True)
def filter_flow(config_path: str = "configs/pipeline.yaml", sample: int | None = None,
                seed: int = 0, video_ids: list[str] | None = None):
    logger = get_run_logger()
    config = load_config(config_path)
    ctx = build_context(config, video_ids)
    if sample is not None and sample < len(ctx.video_ids):
        keep = set(random.Random(seed).sample(sorted(ctx.video_ids), sample))
        ctx.inputs = {v: d for v, d in ctx.inputs.items() if v in keep}
    logger.info(f"Filtering {len(ctx.video_ids)} raw videos.")

    has_metadata = [v for v in ctx.video_ids if ctx.inputs[v]["metadata"] is not None]
    run_modal_stage(ctx, "extract_filter_frames", has_metadata)
    run_agent_stage(ctx, "filter_video")
    failed = {r.stage: r.failed for r in ctx.reports}

    prefix = config["s3"]["prefixes"]["filtered_videos"]
    max_motion = float(config["filter"]["max_camera_motion_px"])

    def row(vid: str) -> dict:
        d = ctx.inputs[vid]
        r = {"video_id": vid, "gym": d.get("gym"), "no_metadata_row": not d.get("has_metadata_row", True),
             "missing_climber_metadata": d["metadata"] is None}
        if r["missing_climber_metadata"]:
            return {**r, "status": "rejected", "notes": "no climber height in raw/metadata"}
        info = _get_json(ctx, f"{prefix}{vid}/video_info.json") or {}
        r.update({k: info.get(k) for k in ("camera_motion_px", "duration_s", "width", "height")})
        motion = info.get("camera_motion_px")
        r["camera_motion"] = motion is not None and motion > max_motion   # None: no evidence
        if vid not in ctx.ok["filter_video"]:
            reason = (failed.get("extract_filter_frames", {}).get(vid)
                      and "frame extraction: " + failed["extract_filter_frames"][vid]) \
                or (failed.get("filter_video", {}).get(vid)
                    and "filter agent: " + failed["filter_video"][vid]) \
                or "upstream step not current"
            return {**r, "status": "not_evaluated", "not_evaluated_reason": reason}
        res = _get_json(ctx, f"{prefix}{vid}/filter_result.json")   # Claude's judgment
        r.update({k: res[k] for k in ("too_many_climbers", "camera_blur", "moving_camera",
                                      "no_floor_plane", "notes")})
        rejected = res["too_many_climbers"] or res["camera_blur"] or res["moving_camera"]             or res["no_floor_plane"] or r["camera_motion"]
        return {**r, "status": "rejected" if rejected else "passed"}

    with ThreadPoolExecutor(S3_WORKERS) as pool:
        run_rows = list(pool.map(row, ctx.video_ids))

    # Merge into the table of every video filtered so far.
    table_key = f"{prefix}filter_results.csv"
    try:
        table = {r["video_id"]: r for r in rows_from_csv(download_bytes(ctx.s3, ctx.bucket, table_key))}
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("NoSuchKey", "404"):
            raise
        table = {}
    table.update({r["video_id"]: r for r in run_rows})
    table_bytes = rows_to_csv(list(table.values()))
    rows = rows_from_csv(table_bytes)          # normalized (string) form for the summary
    ctx.s3.put_object(Bucket=ctx.bucket, Key=table_key, Body=table_bytes, ContentType="text/csv")

    passing = sorted(r["video_id"] for r in rows if r["status"] == "passed")
    ctx.s3.put_object(Bucket=ctx.bucket, Key=f"{prefix}passing_video_ids.json",
                      Body=json.dumps(passing).encode(), ContentType="application/json")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = summarize(rows)
    run = {"run_ts": run_ts, "scope": len(run_rows), "sample": sample, "seed": seed,
           "git_sha": ctx.git_sha, "model": config["agents"]["filter_video"]["model"],
           "max_camera_motion_px": max_motion,
           "run_status": dict(Counter(r["status"] for r in run_rows))}
    ctx.s3.put_object(Bucket=ctx.bucket, Key=f"{prefix}_runs/{run_ts}.json",
                      Body=json.dumps({"run": run, "summary": summary}, indent=2).encode(),
                      ContentType="application/json")

    logger.info(f"{len(passing)}/{len(rows)} filtered videos pass "
                f"(this run: {run['run_status']}).")
    publish_report(ctx, "filter-stages")
    create_markdown_artifact(key="filter-summary", markdown=to_markdown(summary, run),
                             description="Video filter statistics")
    return summary
