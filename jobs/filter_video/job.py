"""
filter_video — Modal job wrapper.
Agentic CPU job; actual filtering is handled by the Claude batch agent.
This job handles video downloading, frame extraction, and result writing.
"""

import modal
from common.modal_images import cpu_base

app = modal.App("boulderdreamer-filter-video")


@app.function(
    image=cpu_base,
    secrets=[modal.Secret.from_name("boulderdreamer-secrets")],
    cpu=2,
    timeout=600,
)
def filter_video(video_id: str, config: dict):
    from jobs.filter_video.logic import run
    run(video_id, config)
