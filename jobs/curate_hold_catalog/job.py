"""curate_hold_catalog — CPU Modal job (embedding, HDBSCAN, parametric hold generation)."""

import modal
from common.modal_images import cpu_base

app = modal.App("boulderdreamer-curate-hold-catalog")


@app.function(
    image=cpu_base,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    cpu=8,
    memory=16384,
    timeout=3600,
)
def curate_hold_catalog(config: dict):
    from jobs.curate_hold_catalog.logic import run
    run(config)
