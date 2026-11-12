"""post_processing — CPU Modal job (floor plane fit + metric normalization)."""

import modal
from common.modal_images import cpu_base

app = modal.App("boulderdreamer-post-processing")


@app.function(
    image=cpu_base,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    cpu=4,
    timeout=900,
)
def post_processing(video_id: str, config: dict):
    from jobs.post_processing.logic import run
    run(video_id, config)
