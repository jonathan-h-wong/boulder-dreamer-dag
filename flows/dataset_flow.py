"""
Dataset Flow — stage-major processing of every filtered video.

  1. Scene perception     bkg_plate | depth                 (concurrent Modal maps)
                          poses | hold_proposals            (poses need depth's K;
                                                             proposals need the plate)
  2. Hold perception      hold_descriptions                 (one Claude batch, all videos)
                          hold_crops | hold_contacts | grounded_poses   (concurrent)
                          wall_mesh                         (+ bare-wall depth, hold frames)
  -- barrier --           hold catalog                      (reuse an existing one, or
                                                             rebuild with rebuild_catalog;
                                                             fits with jobs/hold_fit.py)
  3. Reconstruction       scenegraph                        (same hold_fit procedure,
                                                             warm-started from the catalog)
  4. Grounding            post_processing                   (Modal map)
  (optional)              release manifest                  manifests/{release}.json

Each stage runs only on videos whose upstream stages are current, and skips videos
whose marker already matches the computed cache key.

If no hold catalog exists yet, the flow stops at the barrier instead of building one, so
the catalog can be inspected before anything is fitted against it: build it with the
hold-catalog flow (or pass rebuild_catalog=True), then rerun this flow. Everything up to
the barrier is cached, so the rerun picks up at reconstruction.
"""

import json

from prefect import flow, get_run_logger

from common.s3 import download_bytes, get_client
from flows.runtime import (
    build_context, ensure_hold_catalog, load_config, publish_report, run_agent_stage,
    run_modal_stage, write_climber_info, write_release,
)


def _concurrently(ctx, stages: list[str]):
    futures = [run_modal_stage.submit(ctx, s) for s in stages]
    for f in futures:
        f.result()


@flow(name="produce-dataset", log_prints=True)
def dataset_flow(
    config_path: str = "configs/pipeline.yaml",
    video_ids: list[str] | None = None,
    rebuild_catalog: bool = False,
    release: str | None = None,
):
    """
    video_ids:        subset to process (default: filtered_videos/passing_video_ids.json)
    rebuild_catalog:  (re)build the global hold catalog from all current videos before
                      reconstruction (invalidates every scenegraph downstream). Without
                      it, a missing catalog stops the flow at the barrier.
    release:          if set, write an immutable manifests/{release}.json at the end
    """
    logger = get_run_logger()
    config = load_config(config_path)
    if video_ids is None:
        key = config["s3"]["prefixes"]["filtered_videos"] + "passing_video_ids.json"
        video_ids = json.loads(download_bytes(get_client(config["s3"]["region"]),
                                              config["s3"]["bucket"], key))
    ctx = build_context(config, video_ids)
    logger.info(f"Processing {len(ctx.video_ids)} videos (git {ctx.git_sha}).")

    # Stage 1: Scene Perception (+ hold proposals, which only need the clean plate)
    _concurrently(ctx, ["extract_bkg_plate", "extract_depth"])
    _concurrently(ctx, ["extract_poses", "extract_hold_proposals"])

    # Stage 2: Hold Perception
    run_agent_stage(ctx, "extract_hold_descriptions")
    _concurrently(ctx, ["extract_hold_crops", "extract_hold_contacts", "extract_grounded_poses"])
    run_modal_stage(ctx, "extract_wall_mesh")

    # Barrier: global hold catalog (fit on the same hold frames the scenegraph uses)
    if ensure_hold_catalog(ctx, rebuild=rebuild_catalog, build_if_missing=False) is None:
        publish_report(ctx, "dataset-stages")
        logger.warning(
            f"Stopped at the hold-catalog barrier: no catalog exists. "
            f"{len(ctx.ok['extract_wall_mesh'])} videos have a current wall mesh. Build the "
            "catalog with the hold-catalog flow, inspect it, then rerun this flow.")
        return

    # Stage 3: Reconstruction
    run_modal_stage(ctx, "extract_scenegraph")

    # Stage 4: Grounding
    run_modal_stage(ctx, "post_processing", before_dispatch=write_climber_info)

    publish_report(ctx, "dataset-stages")
    done = len(ctx.ok["post_processing"])
    logger.info(f"Dataset production complete: {done}/{len(ctx.video_ids)} videos current.")

    if release:
        write_release(ctx, release)
