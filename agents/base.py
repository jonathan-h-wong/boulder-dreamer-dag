"""
Resumable Claude Message Batches runner shared by all agent stages.

One call dispatches *every* pending item at once (chunked only to respect the API's
per-batch limits). Each submitted batch is recorded in S3 before polling:

    {batches_prefix}{stage}/{batch_id}.json
        {"batch_id", "submitted_at", "items": {custom_id: {"item": video_id, "cache_key": ...}}}

so if the orchestrator dies during the (up to 24h) wait, a rerun re-attaches to the
in-flight batch instead of re-submitting and paying twice.

An agent module plugs in with two functions:
    build_params(item: str, config: dict) -> dict            # Messages API params
    handle_result(item: str, message, config: dict) -> None  # validate + upload outputs
handle_result raising marks that item failed; it's retried on the next run.
"""

from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, TypeVar

import anthropic
from pydantic import BaseModel

from common.idempotency import write_marker

M = TypeVar("M", bound=BaseModel)

# API limits are 100k requests / 256 MB per batch; stay under with margin.
MAX_REQUESTS_PER_BATCH = 50_000
MAX_BYTES_PER_BATCH = 200 * 1024 * 1024
IO_WORKERS = 32          # build_params / handle_result are S3-bound
IO_WINDOW = 256          # items in flight at once (bounds request memory)


def get_client() -> anthropic.Anthropic:
    # API key from ANTHROPIC_API_KEY (Modal secret / orchestrator env)
    return anthropic.Anthropic()


def image_block(data: bytes, media_type: str = "image/png") -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                        "data": base64.b64encode(data).decode()}}


def message_text(message) -> str:
    return "".join(b.text for b in message.content if b.type == "text")


def output_config(model: type[BaseModel], effort: str | None = None) -> dict:
    """
    Structured outputs: constrain the response to the JSON schema of a Pydantic model.
    transform_schema converts the model's schema to the subset the API enforces (moving
    unsupported constraints like max_length into descriptions), so parse_structured
    re-validates with the full Pydantic model.
    """
    cfg: dict = {"format": {"type": "json_schema", "schema": anthropic.transform_schema(model)}}
    if effort:
        cfg["effort"] = effort
    return cfg


def parse_structured(message, model: type[M]) -> M:
    if message.stop_reason != "end_turn":
        # max_tokens -> truncated JSON; refusal -> no answer. Either way, retry later.
        raise ValueError(f"stop_reason={message.stop_reason}")
    return model.model_validate_json(message_text(message))


@dataclass
class BatchOutcome:
    succeeded: set[str] = field(default_factory=set)
    failed: dict[str, str] = field(default_factory=dict)  # item -> reason


def run_batch_stage(
    *,
    stage: str,
    items: dict[str, dict],          # item -> marker spec (spec["body"]["cache_key"] = target)
    build_params: Callable[[str, dict], dict],
    handle_result: Callable[[str, object, dict], None],
    config: dict,
    s3,
    poll_interval: int = 60,
    log: Callable[[str], None] = print,
) -> BatchOutcome:
    client = get_client()
    bucket = config["s3"]["bucket"]
    pending_prefix = f"{config['s3']['prefixes']['batches']}{stage}/"
    outcome = BatchOutcome()

    # 1. Re-attach to batches submitted by a previous run.
    pending = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=pending_prefix):
        for obj in page.get("Contents", []):
            rec = json.loads(s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read())
            pending[obj["Key"]] = rec

    def is_current(entry: dict) -> bool:
        spec = items.get(entry["item"])
        return spec is not None and spec["body"]["cache_key"] == entry["cache_key"]

    for key, rec in list(pending.items()):
        if not any(is_current(e) for e in rec["items"].values()):
            # Every item is stale (config/code changed or no longer requested): stop paying.
            try:
                client.messages.batches.cancel(rec["batch_id"])
            except anthropic.APIError:
                pass  # already ended
            s3.delete_object(Bucket=bucket, Key=key)
            del pending[key]
    in_flight = {e["item"] for rec in pending.values() for e in rec["items"].values()
                 if is_current(e)}
    if pending:
        log(f"[{stage}] re-attaching to {len(pending)} in-flight batch(es), {len(in_flight)} items")

    # 2. Submit everything else, chunked by request count / payload size.
    to_submit = [i for i in items if i not in in_flight]
    chunk, chunk_bytes = [], 0

    def flush():
        nonlocal chunk, chunk_bytes
        if not chunk:
            return
        batch = client.messages.batches.create(
            requests=[{"custom_id": cid, "params": params} for cid, _, params in chunk]
        )
        rec = {
            "batch_id": batch.id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            # custom_id must match ^[a-zA-Z0-9_-]{1,64}$, so video ids map via this table
            "items": {cid: {"item": item, "cache_key": items[item]["body"]["cache_key"]}
                      for cid, item, _ in chunk},
        }
        key = f"{pending_prefix}{batch.id}.json"
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(rec).encode())
        pending[key] = rec
        log(f"[{stage}] submitted batch {batch.id} with {len(chunk)} requests")
        chunk, chunk_bytes = [], 0

    def build(item):
        try:
            return build_params(item, config), None
        except Exception as e:  # missing upstream file etc. — skip this item only
            return None, f"build_params: {e!r}"

    with ThreadPoolExecutor(IO_WORKERS) as pool:
        for start in range(0, len(to_submit), IO_WINDOW):
            window = to_submit[start:start + IO_WINDOW]
            for n, item, (params, err) in zip(range(start, start + len(window)), window,
                                              pool.map(build, window)):
                if err:
                    outcome.failed[item] = err
                    continue
                size = len(json.dumps(params))
                if chunk and (len(chunk) >= MAX_REQUESTS_PER_BATCH
                              or chunk_bytes + size > MAX_BYTES_PER_BATCH):
                    flush()
                chunk.append((f"r{n}", item, params))
                chunk_bytes += size
    flush()

    # 3. Poll every pending batch to completion, then collect results.
    waiting = dict(pending)
    while waiting:
        for key, rec in list(waiting.items()):
            batch = client.messages.batches.retrieve(rec["batch_id"])
            if batch.processing_status != "ended":
                continue
            def collect(result, batch_id=rec["batch_id"], rec=rec):
                entry = rec["items"].get(result.custom_id)
                if entry is None or not is_current(entry):
                    return None  # stale result: never let it overwrite current outputs
                item = entry["item"]
                if result.result.type != "succeeded":
                    return item, f"batch result {result.result.type}"
                try:
                    handle_result(item, result.result.message, config)
                except Exception as e:
                    return item, f"handle_result: {e!r}"
                write_marker(items[item], client=s3, batch_id=batch_id)
                return item, None

            window = []
            with ThreadPoolExecutor(IO_WORKERS) as pool:
                def drain():
                    for res in pool.map(collect, window):
                        if res is None:
                            continue
                        item, err = res
                        if err:
                            outcome.failed[item] = err
                        else:
                            outcome.succeeded.add(item)
                    window.clear()
                for result in client.messages.batches.results(rec["batch_id"]):
                    window.append(result)
                    if len(window) >= IO_WINDOW:
                        drain()
                drain()
            s3.delete_object(Bucket=bucket, Key=key)
            del waiting[key]
        if waiting:
            log(f"[{stage}] {len(waiting)} batch(es) still processing; next check in {poll_interval}s")
            time.sleep(poll_interval)

    return outcome
