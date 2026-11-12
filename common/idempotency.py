"""
Stage completion markers carrying cache keys.

Layout (one JSON file per stage per video, next to that video's outputs):
    {root}{video_id}/_markers/{stage}.json      per-video stages
    {root}_markers/{stage}.json                  global stages (hold catalog)

A stage is *current* when its marker exists and marker["cache_key"] equals the key the
orchestrator computes now (common/stages.py). A stale marker is simply overwritten on
the next successful run — nothing has to be cleared by hand.

Who writes:
  - Modal jobs write their own marker as the final step, after outputs are uploaded
    (the orchestrator passes the fully-built `spec` in). A crashed orchestrator never
    loses a completed result.
  - Agent stages (Claude batches) have no job, so the orchestrator writes after
    validating and uploading each result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError


def marker_key(prefix: str, stage: str, video_id: str | None = None) -> str:
    base = f"{prefix}{video_id}/" if video_id is not None else prefix
    return f"{base}_markers/{stage}.json"


def marker_spec(*, bucket: str, region: str, key: str, cache_key: str, body: dict) -> dict:
    """Everything needed to write a marker, serializable to a Modal container."""
    return {"bucket": bucket, "region": region, "key": key,
            "body": {"cache_key": cache_key, **body}}


def write_marker(spec: dict, client=None, **extra) -> None:
    client = client or boto3.client("s3", region_name=spec["region"])
    body = {**spec["body"], **extra, "completed_at": datetime.now(timezone.utc).isoformat()}
    client.put_object(
        Bucket=spec["bucket"],
        Key=spec["key"],
        Body=json.dumps(body, indent=2, default=str).encode(),
        ContentType="application/json",
    )


def read_marker(client, bucket: str, key: str) -> dict | None:
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise  # permissions / throttling must not masquerade as "not done"
    return json.loads(resp["Body"].read())
