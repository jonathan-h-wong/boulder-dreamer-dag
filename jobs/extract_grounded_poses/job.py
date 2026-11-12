"""extract_grounded_poses — CPU Modal job (best-fit human depth alignment)."""

import modal
from common.modal_images import cpu_base

app = modal.App("boulderdreamer-extract-grounded-poses")


@app.function(
    image=cpu_base,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    cpu=8,
    timeout=1800,
)
def extract_grounded_poses(video_id: str, config: dict):
    from jobs.extract_grounded_poses.logic import run
    run(video_id, config)
