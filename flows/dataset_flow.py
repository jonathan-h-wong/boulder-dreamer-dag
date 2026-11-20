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
import random

from prefect import flow, get_run_logger

from common.s3 import download_bytes, get_client
from flows.runtime import (
    build_context, ensure_hold_catalog, load_config, publish_report, refresh_stage,
    run_agent_stage, run_modal_stage, write_climber_info, write_release,
)
from flows.stages import STAGES


def _step(ctx, stages: list[str], only: set[str] | None, **kw) -> bool:
    """Run `stages` concurrently (they are independent by construction). Stages outside
    `only` are not run, just refreshed from their markers so downstream eligibility still
    works. Returns True once every stage in `only` has been handled: time to stop."""
    futures = []
    for s in stages:
        if only is not None and s not in only:
            futures.append(refresh_stage.submit(ctx, s))
        elif STAGES[s].runner == "agent":
            futures.append(run_agent_stage.submit(ctx, s))
        else:
            futures.append(run_modal_stage.submit(ctx, s, **kw))
    for f in futures:
        f.result()
    return only is not None and only <= set(ctx.ok)


@flow(name="produce-dataset", log_prints=True)
def dataset_flow(
    config_path: str = "configs/pipeline.yaml",
    video_ids: list[str] | None = None,
    sample: int | None = None,
    seed: int = 0,
    stages: list[str] | None = None,
    rebuild_catalog: bool = False,
    release: str | None = None,
):
    """
    video_ids:        subset to process (default: filtered_videos/passing_video_ids.json)
    sample, seed:     process a seeded random sample of those videos instead
    stages:           run only these per-video stages (others are read from their markers,
                      not run) and stop after the last of them, e.g.
                      ["extract_bkg_plate", "extract_hold_proposals"]
    rebuild_catalog:  (re)build the global hold catalog from all current videos before
                      reconstruction (invalidates every scenegraph downstream). Without
                      it, a missing catalog stops the flow at the barrier.
    release:          if set, write an immutable manifests/{release}.json at the end
    """
    logger = get_run_logger()
    config = load_config(config_path)
    only = set(stages) if stages else None
    if only is not None:
        bad = only - {n for n, st in STAGES.items() if st.scope == "video"}
        if bad:
            raise ValueError(f"not per-video stages: {sorted(bad)} (the catalog has its own flow)")
    if video_ids is None:
        key = config["s3"]["prefixes"]["filtered_videos"] + "passing_video_ids.json"
        video_ids = json.loads(download_bytes(get_client(config["s3"]["region"]),
                                              config["s3"]["bucket"], key))
    if sample:
        video_ids = random.Random(seed).sample(sorted(video_ids), min(sample, len(video_ids)))
    ctx = build_context(config, video_ids)
    logger.info(f"Processing {len(ctx.video_ids)} videos (git {ctx.git_sha})"
                + (f", stages {sorted(only)}." if only else "."))

    def done() -> bool:
        publish_report(ctx, "dataset-stages")
        logger.info(f"Requested stages done for {len(ctx.video_ids)} videos.")
        return True

    # Stage 1: Scene Perception (+ hold proposals, which only need the clean plate)
    if _step(ctx, ["extract_bkg_plate", "extract_depth"], only) and done():
        return
    if _step(ctx, ["extract_poses", "extract_hold_proposals"], only) and done():
        return

    # Stage 2: Hold Perception (two Claude batches: detect, then describe)
    if _step(ctx, ["detect_holds"], only) and done():
        return
    if _step(ctx, ["extract_hold_descriptions"], only) and done():
        return
    if _step(ctx, ["extract_hold_crops", "extract_hold_contacts", "extract_grounded_poses"],
             only) and done():
        return
    if _step(ctx, ["extract_wall_mesh"], only) and done():
        return

    # Barrier: global hold catalog (fit on the same hold frames the scenegraph uses)
    if ensure_hold_catalog(ctx, rebuild=rebuild_catalog, build_if_missing=False) is None:
        publish_report(ctx, "dataset-stages")
        logger.warning(
            f"Stopped at the hold-catalog barrier: no catalog exists. "
            f"{len(ctx.ok['extract_wall_mesh'])} videos have a current wall mesh. Build the "
            "catalog with the hold-catalog flow, inspect it, then rerun this flow.")
        return

    # Stage 3: Reconstruction
    if _step(ctx, ["extract_scenegraph"], only) and done():
        return

    # Stage 4: Grounding
    if _step(ctx, ["post_processing"], only, before_dispatch=write_climber_info) and done():
        return

    publish_report(ctx, "dataset-stages")
    n_done = len(ctx.ok["post_processing"])
    logger.info(f"Dataset production complete: {n_done}/{len(ctx.video_ids)} videos current.")

    if release:
        write_release(ctx, release)
