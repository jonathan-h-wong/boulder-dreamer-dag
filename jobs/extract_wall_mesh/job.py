"""extract_wall_mesh — CPU Modal job (depth lifting + wall mesh generation)."""

import modal
from common.modal_images import cpu_base

app = modal.App("boulderdreamer-extract-wall-mesh")


@app.function(
    image=cpu_base,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    cpu=4,
    timeout=1200,
)
def extract_wall_mesh(video_id: str, config: dict):
    from jobs.extract_wall_mesh.logic import run
    run(video_id, config)
