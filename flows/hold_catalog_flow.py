"""
Hold Catalog Flow — standalone (re)build of the global, route-agnostic hold catalog,
e.g. after changing the embedding / clustering config ("re-embedding").

The catalog has no detection pass of its own: it aggregates the per-video hold
perception outputs (processed/{video_id}/holds/) of every video whose
extract_wall_mesh is current. The dataset flow calls the same ensure_hold_catalog
step inline; use this flow to rebuild without running the rest of the dataset.
"""

import json

from prefect import flow, get_run_logger

from common.s3 import download_bytes, get_client
from flows.runtime import build_context, ensure_hold_catalog, load_config, refresh_stage


@flow(name="curate-hold-catalog", log_prints=True)
def hold_catalog_flow(config_path: str = "configs/pipeline.yaml", rebuild: bool = True):
    logger = get_run_logger()
    config = load_config(config_path)
    key = config["s3"]["prefixes"]["filtered_videos"] + "passing_video_ids.json"
    video_ids = json.loads(download_bytes(get_client(config["s3"]["region"]),
                                          config["s3"]["bucket"], key))
    ctx = build_context(config, video_ids)

    refresh_stage(ctx, "extract_wall_mesh")
    catalog_key = ensure_hold_catalog(ctx, rebuild=rebuild)
    logger.info(f"Hold catalog {catalog_key} "
                f"({len(ctx.ok['extract_wall_mesh'])} candidate member videos).")
    return catalog_key
