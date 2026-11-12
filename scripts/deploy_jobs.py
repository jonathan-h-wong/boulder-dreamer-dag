"""
Deploy every Modal job app (jobs/*/job.py). Run from the repo root:

    python scripts/deploy_jobs.py                  # all jobs
    python scripts/deploy_jobs.py extract_depth    # selected jobs
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(selected: list[str]) -> int:
    jobs = sorted(p.parent.name for p in (ROOT / "jobs").glob("*/job.py"))
    unknown = set(selected) - set(jobs)
    if unknown:
        print(f"Unknown jobs: {sorted(unknown)}. Available: {jobs}")
        return 1
    failed = []
    for job in selected or jobs:
        print(f"==> deploying {job}")
        if subprocess.run(["modal", "deploy", "-m", f"jobs.{job}.job"], cwd=ROOT).returncode:
            failed.append(job)
    if failed:
        print(f"Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
