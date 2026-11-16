"""extract_grounded_poses — CPU Modal job (best-fit human depth alignment)."""

import modal

from common.modal_images import SECRETS, cpu_geometry, WEIGHTS_MOUNT, weights_volume
from common.modal_jobs import app_name

app = modal.App(app_name("extract_grounded_poses"))


@app.function(
    image=cpu_geometry,
    secrets=SECRETS,
    volumes={WEIGHTS_MOUNT: weights_volume},
    cpu=8,
    timeout=1800,
)
def extract_grounded_poses(video_id: str, marker: dict, config: dict):
    from common.idempotency import write_marker
    from jobs.extract_grounded_poses.logic import run

    run(video_id, config)
    write_marker(marker)  # last step: only reached if outputs were fully written
