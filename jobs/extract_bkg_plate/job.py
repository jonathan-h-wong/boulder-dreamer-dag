"""extract_bkg_plate — CPU Modal job (median filter over downsampled frames)."""

import modal
from common.modal_images import cpu_base

app = modal.App("boulderdreamer-extract-bkg-plate")


@app.function(
    image=cpu_base,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    cpu=4,
    timeout=900,
)
def extract_bkg_plate(video_id: str, config: dict):
    from jobs.extract_bkg_plate.logic import run
    run(video_id, config)
