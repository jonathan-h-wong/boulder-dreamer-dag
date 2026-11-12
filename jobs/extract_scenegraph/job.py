"""extract_scenegraph — GPU Modal job (differentiable SDF + gaussian splatting)."""

import modal
from common.modal_images import gpu_scenegraph

app = modal.App("boulderdreamer-extract-scenegraph")


@app.function(
    image=gpu_scenegraph,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    gpu="A100",
    timeout=7200,
)
def extract_scenegraph(video_id: str, config: dict):
    from jobs.extract_scenegraph.logic import run
    run(video_id, config)
