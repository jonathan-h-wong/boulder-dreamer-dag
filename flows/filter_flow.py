"""
Filter Flow — decides which raw videos enter the dataset.

  1. extract_filter_frames  (Modal .map over all raw videos)
  2. filter_video           (one Claude batch over all videos with frames + metadata)
  3. aggregate              -> filtered_videos/passing_video_ids.json

Videos without climber height in raw/metadata/*.csv are rejected up front, without
spending a Claude request on them.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from prefect import flow, get_run_logger
from prefect.artifacts import create_table_artifact

from common.s3 import download_bytes
from common.schemas import VideoFilterResult
from flows.runtime import (
    S3_WORKERS, build_context, load_config, publish_report, run_agent_stage, run_modal_stage,
)


@flow(name="filter-videos", log_prints=True)
def filter_flow(config_path: str = "configs/pipeline.yaml"):
    logger = get_run_logger()
    config = load_config(config_path)
    ctx = build_context(config)
    logger.info(f"Found {len(ctx.video_ids)} raw videos.")

    has_metadata = [v for v in ctx.video_ids if ctx.inputs[v]["metadata"] is not None]
    run_modal_stage(ctx, "extract_filter_frames", has_metadata)
    run_agent_stage(ctx, "filter_video")

    prefix = config["s3"]["prefixes"]["filtered_videos"]

    def result(vid: str) -> dict:
        if ctx.inputs[vid]["metadata"] is None:
            return VideoFilterResult(video_id=vid, passed=False, missing_climber_metadata=True,
                                     notes="no climber height in raw/metadata").model_dump()
        if vid not in ctx.ok["filter_video"]:
            return VideoFilterResult(video_id=vid, passed=False,
                                     notes="not evaluated (failed upstream; rerun to retry)").model_dump()
        raw = download_bytes(ctx.s3, ctx.bucket, f"{prefix}{vid}/filter_result.json")
        return json.loads(raw)

    with ThreadPoolExecutor(S3_WORKERS) as pool:
        results = list(pool.map(result, ctx.video_ids))

    passing = sorted(r["video_id"] for r in results if r["passed"])
    ctx.s3.put_object(Bucket=ctx.bucket, Key=f"{prefix}passing_video_ids.json",
                      Body=json.dumps(passing).encode(), ContentType="application/json")
    logger.info(f"{len(passing)}/{len(results)} videos passed filtering.")

    publish_report(ctx, "filter-stages")
    create_table_artifact(key="filter-results", table=results, description="Video filter results")
    return passing
