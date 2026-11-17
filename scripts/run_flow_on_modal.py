"""
Run a Prefect flow in a Modal container instead of on your machine (or, later, a RunPod pod).

This is NOT a second orchestrator: the flow is the orchestrator, and this script only
rents it a computer. The container gets the same secrets as the jobs (AWS, HF,
Anthropic), so Claude batch stages run without any local keys. Prefect runs in
ephemeral mode (no server, no UI); results land in S3 as usual and logs are in the
Modal app logs. Jobs are called by name, so deploy them first (scripts/deploy_jobs.py).

    modal run --detach scripts/run_flow_on_modal.py --flow filter --sample 300
    modal run --detach scripts/run_flow_on_modal.py --flow filter            # all videos
    modal run --detach scripts/run_flow_on_modal.py --flow dataset           # stops before
                                                                             # the catalog if none exists
    modal run --detach scripts/run_flow_on_modal.py --flow catalog           # build/rebuild it
    modal run --detach scripts/run_flow_on_modal.py --flow dataset --release r1

--detach keeps the run going if this terminal closes (Claude batches can take hours).
"""

from __future__ import annotations

import modal

from common.modal_images import SECRETS, orchestrator

app = modal.App("boulderdreamer-orchestrator", image=orchestrator)

CONFIG = "configs/pipeline.yaml"


@app.function(secrets=SECRETS, timeout=24 * 3600, cpu=2, memory=4096)
def run_flow(flow: str, kwargs: dict, git_sha: str):
    import os
    os.environ["GIT_SHA"] = git_sha
    if flow == "filter":
        from flows.filter_flow import filter_flow
        return filter_flow(config_path=CONFIG, **kwargs)
    if flow == "dataset":
        from flows.dataset_flow import dataset_flow
        return dataset_flow(config_path=CONFIG, **kwargs)
    if flow == "catalog":
        from flows.hold_catalog_flow import hold_catalog_flow
        return hold_catalog_flow(config_path=CONFIG, **kwargs)
    raise ValueError(f"unknown flow {flow!r}")


@app.local_entrypoint()
def main(flow: str = "filter", sample: int = 0, seed: int = 0,
         rebuild_catalog: bool = False, release: str = ""):
    import json

    from flows.runtime import git_sha

    kwargs = {
        "filter": {"sample": sample or None, "seed": seed},
        "dataset": {"rebuild_catalog": rebuild_catalog, "release": release or None},
        "catalog": {},
    }[flow]
    result = run_flow.remote(flow, kwargs, git_sha())
    print("FLOW RESULT")
    print(json.dumps(result, indent=2, default=str))
