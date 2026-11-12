"""
Shared Claude Batch API base for BoulderDreamer agent jobs.

All agentic jobs inherit this pattern:
  1. Build a list of batch requests (one per item)
  2. Submit via client.messages.batches.create()
  3. Store batch_id (in S3 or Prefect state)
  4. Poll until complete
  5. Parse results
"""

import time
import anthropic
from typing import Any


def get_client() -> anthropic.Anthropic:
    # API key picked up from ANTHROPIC_API_KEY env var (set via Modal secret)
    return anthropic.Anthropic()


def poll_batch(client: anthropic.Anthropic, batch_id: str, poll_interval: int = 30) -> list[Any]:
    """
    Poll a Claude batch until complete, then return all results.
    poll_interval: seconds between status checks.
    """
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        if status == "ended":
            break
        print(f"Batch {batch_id} status: {status} — waiting {poll_interval}s...")
        time.sleep(poll_interval)

    results = []
    for result in client.messages.batches.results(batch_id):
        results.append(result)
    return results
