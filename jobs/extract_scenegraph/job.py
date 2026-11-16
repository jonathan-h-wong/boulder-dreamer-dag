"""extract_scenegraph — GPU Modal job (hold_fit: differentiable SDF ray marching)."""

import modal

from common.modal_images import SECRETS, gpu_fit, WEIGHTS_MOUNT, weights_volume
from common.modal_jobs import app_name

app = modal.App(app_name("extract_scenegraph"))


@app.function(
    image=gpu_fit,
    secrets=SECRETS,
    volumes={WEIGHTS_MOUNT: weights_volume},
    gpu="A100",
    timeout=7200,
)
def extract_scenegraph(video_id: str, marker: dict, config: dict):
    from common.idempotency import write_marker
    from jobs.extract_scenegraph.logic import run

    run(video_id, config)
    write_marker(marker)  # last step: only reached if outputs were fully written
