"""
Hold Catalog Flow — builds the global, route-agnostic hold catalog.

Stages:
  1. Extract Hold Descriptions (Claude agent batch job)
  2. Extract Hold Crops (GPU: Marigold V2 + SAM2)
  3. Curate Hold Catalog (embedding, HDBSCAN clustering, parametric hold generation)
"""

from prefect import flow, task, get_run_logger
import yaml

from common.idempotency import is_done, mark_done
from common.s3 import get_client


@task
def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@task
def run_extract_hold_descriptions(config: dict):
    from agents.extract_hold_descriptions_catalog.agent import (
        submit_hold_description_batch,
        poll_hold_description_batch,
    )
    client = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    prefix = config["s3"]["prefixes"]["hold_catalog"] + "holds/"

    if is_done(client, bucket, prefix + "descriptions/"):
        return
    batch_id = submit_hold_description_batch(config)
    poll_hold_description_batch(batch_id, config)
    mark_done(client, bucket, prefix + "descriptions/")


@task
def run_extract_hold_crops(config: dict):
    import modal
    from jobs.extract_hold_crops_catalog.job import app, extract_hold_crops

    client = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    prefix = config["s3"]["prefixes"]["hold_catalog"] + "holds/"

    if is_done(client, bucket, prefix + "crops/"):
        return
    with modal.runner.deploy_app(app):
        extract_hold_crops.remote(config)
    mark_done(client, bucket, prefix + "crops/")


@task
def run_curate_hold_catalog(config: dict):
    import modal
    from jobs.curate_hold_catalog.job import app, curate_hold_catalog

    client = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    prefix = config["s3"]["prefixes"]["hold_catalog"]

    if is_done(client, bucket, prefix + "catalog/"):
        return
    with modal.runner.deploy_app(app):
        curate_hold_catalog.remote(config)
    mark_done(client, bucket, prefix + "catalog/")


@flow(name="curate-hold-catalog", log_prints=True)
def hold_catalog_flow(config_path: str = "configs/pipeline.yaml"):
    logger = get_run_logger()
    config = load_config(config_path)

    logger.info("Stage 1: Extract Hold Descriptions")
    run_extract_hold_descriptions(config)

    logger.info("Stage 2: Extract Hold Crops")
    run_extract_hold_crops(config)

    logger.info("Stage 3: Curate Hold Catalog")
    run_curate_hold_catalog(config)

    logger.info("Hold catalog complete.")
