"""extract_hold_crops_catalog — GPU Modal job (Marigold V2 + SAM2, catalog version)."""

import modal
from common.modal_images import gpu_sam2_marigold

app = modal.App("boulderdreamer-extract-hold-crops-catalog")


@app.function(
    image=gpu_sam2_marigold,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    gpu="A10G",
    timeout=1800,
)
def extract_hold_crops(config: dict):
    from jobs.extract_hold_crops_catalog.logic import run
    run(config)
