"""extract_depth — GPU Modal job (MoGe-3 depth + intrinsics per downsampled frame)."""

import modal

from common.modal_images import gpu_moge, WEIGHTS_MOUNT, weights_volume
from common.modal_jobs import app_name

app = modal.App(app_name("extract_depth"))


@app.function(
    image=gpu_moge,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    volumes={WEIGHTS_MOUNT: weights_volume},
    gpu="A10G",
    timeout=3600,
)
def extract_depth(video_id: str, marker: dict, config: dict):
    from common.idempotency import write_marker
    from jobs.extract_depth.logic import run

    run(video_id, config)
    write_marker(marker)  # last step: only reached if outputs were fully written
