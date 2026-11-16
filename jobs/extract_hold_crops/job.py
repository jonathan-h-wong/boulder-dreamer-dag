"""extract_hold_crops — GPU Modal job (Marigold V2 normals + per-hold crops)."""

import modal

from common.modal_images import SECRETS, gpu_marigold, WEIGHTS_MOUNT, weights_volume
from common.modal_jobs import app_name

app = modal.App(app_name("extract_hold_crops"))


@app.function(
    image=gpu_marigold,
    secrets=SECRETS,
    volumes={WEIGHTS_MOUNT: weights_volume},
    gpu="A10G",  # 24 GB covers Marigold V2 at 1024^2; use "L40S" for 2048^2
    timeout=1800,
)
def extract_hold_crops(video_id: str, marker: dict, config: dict):
    from common.idempotency import write_marker
    from jobs.extract_hold_crops.logic import run

    run(video_id, config)
    write_marker(marker)  # last step: only reached if outputs were fully written
