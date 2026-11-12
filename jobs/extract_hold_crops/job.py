"""extract_hold_crops — GPU Modal job (Marigold V2 + SAM2, per-video version)."""

import modal
from common.modal_images import gpu_sam2_marigold

app = modal.App("boulderdreamer-extract-hold-crops")


@app.function(
    image=gpu_sam2_marigold,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    gpu="A10G",
    timeout=1800,
)
def extract_hold_crops(video_id: str, config: dict):
    from jobs.extract_hold_crops.logic import run
    run(video_id, config)
