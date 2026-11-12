"""extract_filter_frames — CPU Modal job (sample frames for the filter agent)."""

import modal

from common.modal_images import cpu_base
from common.modal_jobs import app_name

app = modal.App(app_name("extract_filter_frames"))


@app.function(
    image=cpu_base,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    cpu=2,
    timeout=600,
)
def extract_filter_frames(video_id: str, marker: dict, config: dict):
    from common.idempotency import write_marker
    from jobs.extract_filter_frames.logic import run

    run(video_id, config)
    write_marker(marker)  # last step: only reached if outputs were fully written
