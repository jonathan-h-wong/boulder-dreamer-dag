"""post_processing — CPU Modal job (floor plane fit + metric normalization)."""

import modal

from common.modal_images import SECRETS, cpu_geometry
from common.modal_jobs import app_name

app = modal.App(app_name("post_processing"))


@app.function(
    image=cpu_geometry,
    secrets=SECRETS,
    cpu=4,
    timeout=900,
)
def post_processing(video_id: str, marker: dict, config: dict):
    from common.idempotency import write_marker
    from jobs.post_processing.logic import run

    run(video_id, config)
    write_marker(marker)  # last step: only reached if outputs were fully written
