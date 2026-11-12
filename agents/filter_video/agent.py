"""
Video filter agent — submits a Claude batch job to evaluate video clips.
"""

import json
import base64
from pathlib import Path

import anthropic

from agents.base import get_client, poll_batch
from agents.filter_video.prompts import SYSTEM_PROMPT, build_filter_prompt
from common.schemas import VideoFilterResult
from common.s3 import get_client as get_s3_client, list_keys, download_bytes


def _sample_frames_b64(video_id: str, config: dict, n_frames: int = 8) -> list[str]:
    """
    Download downsampled frames for a video from S3 and return as base64 JPEG strings.
    TODO: implement frame extraction from raw video or pre-extracted frame directory.
    """
    raise NotImplementedError


def submit_filter_batch(video_ids: list[str], config: dict) -> str:
    """Build and submit a Claude batch request for all video IDs. Returns batch_id."""
    client = get_client()
    requests = []

    for video_id in video_ids:
        frame_b64s = _sample_frames_b64(video_id, config)
        # Build content blocks: text prompt + image blocks
        content = [{"type": "text", "text": build_filter_prompt(video_id, frame_b64s)}]
        for b64 in frame_b64s:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })

        requests.append(
            anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
                custom_id=video_id,
                params={
                    "model": config["agents"]["model"],
                    "max_tokens": config["agents"]["max_tokens"],
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": content}],
                },
            )
        )

    batch = client.messages.batches.create(requests=requests)
    return batch.id


def poll_filter_batch(batch_id: str, config: dict) -> list[dict]:
    """Poll until complete, parse results into VideoFilterResult dicts."""
    client = get_client()
    raw_results = poll_batch(client, batch_id)

    results = []
    for result in raw_results:
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text
            try:
                data = json.loads(text)
                results.append(VideoFilterResult(**data).model_dump())
            except Exception as e:
                print(f"Failed to parse result for {result.custom_id}: {e}")
                results.append({"video_id": result.custom_id, "passed": False, "notes": str(e)})
        else:
            results.append({"video_id": result.custom_id, "passed": False, "notes": "batch error"})

    return results
