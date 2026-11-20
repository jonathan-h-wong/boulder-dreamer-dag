"""extract_hold_proposals — GPU Modal job (SAM2 automatic masks + set-of-marks overlay).

Weights: SAM2 downloads its checkpoint through the HF cache on the weights volume
(HF_HOME=/weights/hf). Populate it once, before the first map, so containers don't each
download it and race to commit the volume:

    modal run -m jobs.extract_hold_proposals.job::prefetch_weights
"""

import modal

from common.modal_images import SECRETS, WEIGHTS_MOUNT, gpu_sam2, weights_volume
from common.modal_jobs import app_name

app = modal.App(app_name("extract_hold_proposals"))


@app.function(
    image=gpu_sam2,
    secrets=SECRETS,
    volumes={WEIGHTS_MOUNT: weights_volume},
    gpu="A10G",
    memory=16384,      # SAM2 returns every mask full-frame before they are cropped
    timeout=1800,
)
def extract_hold_proposals(video_id: str, marker: dict, config: dict):
    from common.idempotency import write_marker
    from jobs.extract_hold_proposals.logic import run

    run(video_id, config)
    write_marker(marker)  # last step: only reached if outputs were fully written


@app.function(image=gpu_sam2, secrets=SECRETS, volumes={WEIGHTS_MOUNT: weights_volume},
              timeout=1800)
def prefetch_weights(checkpoint: str = "facebook/sam2.1-hiera-large"):
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES

    _, ckpt_name = HF_MODEL_ID_TO_FILENAMES[checkpoint]
    path = hf_hub_download(repo_id=checkpoint, filename=ckpt_name)
    weights_volume.commit()
    print(f"cached {checkpoint} at {path}")
