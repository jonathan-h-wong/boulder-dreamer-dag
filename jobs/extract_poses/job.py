"""extract_poses — GPU Modal job (GVHMR full-sequence pose estimation)."""

import modal

from common.modal_images import SECRETS, gpu_gvhmr, WEIGHTS_MOUNT, weights_volume
from common.modal_jobs import app_name

app = modal.App(app_name("extract_poses"))


@app.function(
    image=gpu_gvhmr,
    secrets=SECRETS,
    volumes={WEIGHTS_MOUNT: weights_volume},
    gpu="A10G",
    timeout=3600,
)
def extract_poses(video_id: str, marker: dict, config: dict):
    from common.idempotency import write_marker
    from jobs.extract_poses.logic import run

    run(video_id, config)
    write_marker(marker)  # last step: only reached if outputs were fully written
