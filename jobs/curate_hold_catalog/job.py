"""curate_hold_catalog — GPU Modal job (embedding, HDBSCAN, programs, hold_fit)."""

import modal

from common.modal_images import gpu_fit, WEIGHTS_MOUNT, weights_volume
from common.modal_jobs import app_name

app = modal.App(app_name("curate_hold_catalog"))


@app.function(
    image=gpu_fit,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    volumes={WEIGHTS_MOUNT: weights_volume},
    gpu="A100",
    memory=32768,
    timeout=6 * 3600,
)
def curate_hold_catalog(video_ids: list[str], marker: dict, config: dict):
    from common.idempotency import write_marker
    from jobs.curate_hold_catalog.logic import run

    run(video_ids, config)
    write_marker(marker)  # last step: only reached if outputs were fully written
