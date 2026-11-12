"""extract_hold_proposals — GPU Modal job (SAM2 automatic masks + set-of-marks overlay)."""

import modal

from common.modal_images import gpu_sam2, WEIGHTS_MOUNT, weights_volume
from common.modal_jobs import app_name

app = modal.App(app_name("extract_hold_proposals"))


@app.function(
    image=gpu_sam2,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    volumes={WEIGHTS_MOUNT: weights_volume},
    gpu="A10G",
    timeout=1800,
)
def extract_hold_proposals(video_id: str, marker: dict, config: dict):
    from common.idempotency import write_marker
    from jobs.extract_hold_proposals.logic import run

    run(video_id, config)
    write_marker(marker)  # last step: only reached if outputs were fully written
