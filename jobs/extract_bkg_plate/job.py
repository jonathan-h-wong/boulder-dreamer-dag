"""extract_bkg_plate — CPU Modal job (median filter over downsampled frames)."""

import modal

from common.modal_images import SECRETS, cpu_base
from common.modal_jobs import app_name

app = modal.App(app_name("extract_bkg_plate"))


@app.function(
    image=cpu_base,
    secrets=SECRETS,
    cpu=4,
    memory=8192,      # 64 frames of 4K + the median's working copy
    timeout=900,
)
def extract_bkg_plate(video_id: str, marker: dict, config: dict):
    from common.idempotency import write_marker
    from jobs.extract_bkg_plate.logic import run

    run(video_id, config)
    write_marker(marker)  # last step: only reached if outputs were fully written
