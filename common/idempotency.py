"""
Idempotency helpers using S3 _SUCCESS marker files.

Usage pattern:
    if is_done(client, bucket, success_key):
        return  # skip — already completed
    ... do work ...
    mark_done(client, bucket, success_key)
"""

import json
from datetime import datetime, timezone
from common.s3 import get_client


SUCCESS_MARKER = "_SUCCESS"


def success_key(prefix: str) -> str:
    """Return the S3 key for the success marker under a given output prefix."""
    return f"{prefix.rstrip('/')}/{SUCCESS_MARKER}"


def is_done(client, bucket: str, prefix: str) -> bool:
    """Check whether a _SUCCESS marker exists for this output prefix."""
    key = success_key(prefix)
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.ClientError:
        return False


def mark_done(client, bucket: str, prefix: str, metadata: dict | None = None):
    """Write a _SUCCESS marker to S3 with optional metadata (e.g. git sha, version)."""
    payload = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **(metadata or {}),
    }
    client.put_object(
        Bucket=bucket,
        Key=success_key(prefix),
        Body=json.dumps(payload).encode(),
        ContentType="application/json",
    )


def clear_done(client, bucket: str, prefix: str):
    """Remove a _SUCCESS marker to force re-processing."""
    client.delete_object(Bucket=bucket, Key=success_key(prefix))
