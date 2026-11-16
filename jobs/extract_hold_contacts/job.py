"""extract_hold_contacts — CPU Modal job (pose contacts -> on_route, volume mounting)."""

import modal

from common.modal_images import SECRETS, cpu_geometry, WEIGHTS_MOUNT, weights_volume
from common.modal_jobs import app_name

app = modal.App(app_name("extract_hold_contacts"))


@app.function(
    image=cpu_geometry,
    secrets=SECRETS,
    volumes={WEIGHTS_MOUNT: weights_volume},
    cpu=4,
    timeout=1200,
)
def extract_hold_contacts(video_id: str, marker: dict, config: dict):
    from common.idempotency import write_marker
    from jobs.extract_hold_contacts.logic import run

    run(video_id, config)
    write_marker(marker)  # last step: only reached if outputs were fully written
