"""
Dataset Flow — per-video processing pipeline for all filtered videos.

Stages:
  1. Scene Perception: bkg plate, depth, poses
  2. Hold Perception: hold descriptions (agent), hold crops (GPU)
  3. Reconstruction: grounded poses, wall mesh, scenegraph
  4. Grounding: post-processing (floor plane, metric normalization)
"""

from prefect import flow, task, get_run_logger
from prefect.futures import wait
import yaml
import json
import modal

from common.idempotency import is_done, mark_done
from common.s3 import get_client, download_bytes


@task
def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@task
def get_filtered_video_ids(config: dict) -> list[str]:
    client = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    key = config["s3"]["prefixes"]["filtered_videos"] + "passing_video_ids.json"
    return json.loads(download_bytes(client, bucket, key))


# ---------------------------------------------------------------------------
# Stage 1: Scene Perception
# ---------------------------------------------------------------------------

@task
def stage1_scene_perception(video_id: str, config: dict):
    from jobs.extract_bkg_plate.job import app as bkg_app, extract_bkg_plate
    from jobs.extract_depth.job import app as depth_app, extract_depth
    from jobs.extract_poses.job import app as poses_app, extract_poses

    client = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    prefix = config["s3"]["prefixes"]["processed"] + f"{video_id}/"

    for name, app, fn in [
        ("bkg_plate", bkg_app, extract_bkg_plate),
        ("depth", depth_app, extract_depth),
        ("poses", poses_app, extract_poses),
    ]:
        if is_done(client, bucket, prefix + name + "/"):
            continue
        fn.remote(video_id, config)
        mark_done(client, bucket, prefix + name + "/")


# ---------------------------------------------------------------------------
# Stage 2: Hold Perception
# ---------------------------------------------------------------------------

@task
def stage2_hold_perception(video_id: str, config: dict):
    from agents.extract_hold_descriptions.agent import (
        submit_hold_description_batch,
        poll_hold_description_batch,
    )
    from jobs.extract_hold_crops.job import app, extract_hold_crops

    client = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    prefix = config["s3"]["prefixes"]["processed"] + f"{video_id}/holds/"

    if not is_done(client, bucket, prefix + "descriptions/"):
        batch_id = submit_hold_description_batch(video_id, config)
        poll_hold_description_batch(batch_id, config)
        mark_done(client, bucket, prefix + "descriptions/")

    if not is_done(client, bucket, prefix + "crops/"):
        extract_hold_crops.remote(video_id, config)
        mark_done(client, bucket, prefix + "crops/")


# ---------------------------------------------------------------------------
# Stage 3: Reconstruction
# ---------------------------------------------------------------------------

@task
def stage3_reconstruction(video_id: str, config: dict):
    from jobs.extract_grounded_poses.job import app as gp_app, extract_grounded_poses
    from jobs.extract_wall_mesh.job import app as wm_app, extract_wall_mesh
    from jobs.extract_scenegraph.job import app as sg_app, extract_scenegraph

    client = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    prefix = config["s3"]["prefixes"]["processed"] + f"{video_id}/"

    for name, fn in [
        ("output/grounded_poses", extract_grounded_poses),
        ("input/wall_mesh", extract_wall_mesh),
        ("output/scenegraph", extract_scenegraph),
    ]:
        if is_done(client, bucket, prefix + name + "/"):
            continue
        fn.remote(video_id, config)
        mark_done(client, bucket, prefix + name + "/")


# ---------------------------------------------------------------------------
# Stage 4: Grounding
# ---------------------------------------------------------------------------

@task
def stage4_grounding(video_id: str, config: dict):
    from jobs.post_processing.job import app, post_processing

    client = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    prefix = config["s3"]["prefixes"]["processed"] + f"{video_id}/grounded/"

    if is_done(client, bucket, prefix):
        return
    post_processing.remote(video_id, config)
    mark_done(client, bucket, prefix)


# ---------------------------------------------------------------------------
# Top-level flow
# ---------------------------------------------------------------------------

@flow(name="produce-dataset", log_prints=True)
def dataset_flow(config_path: str = "configs/pipeline.yaml"):
    logger = get_run_logger()
    config = load_config(config_path)
    video_ids = get_filtered_video_ids(config)
    logger.info(f"Processing {len(video_ids)} videos.")

    for video_id in video_ids:
        logger.info(f"[{video_id}] Stage 1: Scene Perception")
        stage1_scene_perception(video_id, config)

        logger.info(f"[{video_id}] Stage 2: Hold Perception")
        stage2_hold_perception(video_id, config)

        logger.info(f"[{video_id}] Stage 3: Reconstruction")
        stage3_reconstruction(video_id, config)

        logger.info(f"[{video_id}] Stage 4: Grounding")
        stage4_grounding(video_id, config)

    logger.info("Dataset production complete.")
