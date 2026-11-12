"""
Filter Flow — dispatches filter_video agent jobs for a list of raw video IDs.
Writes passing video IDs to S3 for downstream flows to consume.
"""

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_table_artifact
import yaml
import json
import boto3

from common.idempotency import is_done, mark_done
from common.s3 import get_client
from agents.filter_video.agent import submit_filter_batch, poll_filter_batch


@task
def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@task
def list_raw_videos(client, bucket: str) -> list[str]:
    """Return video IDs from the raw upload prefix that have not been filtered yet."""
    # TODO: scan s3://{bucket}/raw_videos/ and return video IDs
    raise NotImplementedError


@task
def run_filter_agent(video_ids: list[str], config: dict) -> list[dict]:
    """Submit Claude batch job for video filtering and poll until complete."""
    batch_id = submit_filter_batch(video_ids, config)
    return poll_filter_batch(batch_id, config)


@task
def write_filter_results(client, bucket: str, results: list[dict], config: dict):
    passing = [r["video_id"] for r in results if r["passed"]]
    key = f"{config['s3']['prefixes']['filtered_videos']}passing_video_ids.json"
    client.put_object(Bucket=bucket, Key=key, Body=json.dumps(passing).encode())
    return passing


@flow(name="filter-videos", log_prints=True)
def filter_flow(config_path: str = "configs/pipeline.yaml"):
    logger = get_run_logger()
    config = load_config(config_path)
    client = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]

    video_ids = list_raw_videos(client, bucket)
    logger.info(f"Found {len(video_ids)} raw videos to filter.")

    results = run_filter_agent(video_ids, config)
    passing = write_filter_results(client, bucket, results, config)
    logger.info(f"{len(passing)}/{len(video_ids)} videos passed filtering.")

    create_table_artifact(
        key="filter-results",
        table=results,
        description="Video filter results",
    )
