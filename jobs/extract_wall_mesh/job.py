"""extract_wall_mesh — CPU Modal job (depth lifting + wall mesh generation)."""

import modal

from common.modal_images import SECRETS, cpu_geometry
from common.modal_jobs import app_name

app = modal.App(app_name("extract_wall_mesh"))


@app.function(
    image=cpu_geometry,
    secrets=SECRETS,
    cpu=4,
    timeout=1200,
)
def extract_wall_mesh(video_id: str, marker: dict, config: dict):
    from common.idempotency import write_marker
    from jobs.extract_wall_mesh.logic import run

    run(video_id, config)
    write_marker(marker)  # last step: only reached if outputs were fully written
