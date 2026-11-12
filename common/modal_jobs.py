"""
Naming convention + lookup for deployed Modal jobs.

Each jobs/<job>/job.py defines `app = modal.App(app_name("<job>"))` and a function
named `<job>`. Jobs are deployed once (scripts/deploy_jobs.py); flows never import
job modules, they look up the deployed function by name:

    job_fn("extract_depth").remote(video_id, config)
"""

import modal

APP_PREFIX = "boulderdreamer"


def app_name(job: str) -> str:
    return f"{APP_PREFIX}-{job.replace('_', '-')}"


def job_fn(job: str) -> modal.Function:
    """Reference to a deployed job function. Lazy: resolved on first call."""
    return modal.Function.from_name(app_name(job), job)
