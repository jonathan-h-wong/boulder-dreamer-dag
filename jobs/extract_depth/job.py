"""extract_depth — GPU Modal job (MoGe-2 depth + intrinsics per downsampled frame)."""

import modal
from common.modal_images import gpu_moge

app = modal.App("boulderdreamer-extract-depth")


@app.function(
    image=gpu_moge,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    gpu="A10G",
    timeout=3600,
)
def extract_depth(video_id: str, config: dict):
    from jobs.extract_depth.logic import run
    run(video_id, config)
