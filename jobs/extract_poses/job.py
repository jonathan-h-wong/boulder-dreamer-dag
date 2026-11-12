"""extract_poses — GPU Modal job (GVHMR full-sequence pose estimation)."""

import modal
from common.modal_images import gpu_gvhmr

app = modal.App("boulderdreamer-extract-poses")


@app.function(
    image=gpu_gvhmr,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    gpu="A10G",
    timeout=3600,
)
def extract_poses(video_id: str, config: dict):
    from jobs.extract_poses.logic import run
    run(video_id, config)
