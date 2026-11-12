"""
Hold description extraction agent — per-video version.
Runs on clean plate + pose overlay, produces holds/hold_descriptions.csv per video.
"""

import json
import csv
import io

from agents.base import get_client, poll_batch
from agents.extract_hold_descriptions.prompts import SYSTEM_PROMPT, build_hold_description_prompt
from common.schemas import HoldDescriptionWithRoute
from common.s3 import get_client as get_s3_client


def _get_video_bkg_b64(video_id: str, config: dict) -> str:
    """Download per-video bkg plate and return as base64 JPEG."""
    # TODO: implement
    raise NotImplementedError


def _get_pose_overlay_b64(video_id: str, config: dict) -> str | None:
    """Download pose overlay image (bkg plate + keypose visualization) if available."""
    # TODO: implement
    return None


def submit_hold_description_batch(video_id: str, config: dict) -> str:
    """Submit a Claude batch request for a single video's hold detection."""
    client = get_client()
    image_b64 = _get_video_bkg_b64(video_id, config)
    pose_b64 = _get_pose_overlay_b64(video_id, config)

    content = [
        {"type": "text", "text": build_hold_description_prompt(video_id)},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
    ]
    if pose_b64:
        content.append({"type": "text", "text": "Pose overlay:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": pose_b64}})

    import anthropic
    requests = [
        anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
            custom_id=video_id,
            params={
                "model": config["agents"]["model"],
                "max_tokens": config["agents"]["max_tokens"],
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": content}],
            },
        )
    ]

    batch = client.messages.batches.create(requests=requests)
    return batch.id


def poll_hold_description_batch(batch_id: str, config: dict):
    """Poll until complete, validate schema, write hold_descriptions.csv to S3."""
    client = get_client()
    s3 = get_s3_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]

    raw_results = poll_batch(client, batch_id)

    for result in raw_results:
        video_id = result.custom_id
        if result.result.type != "succeeded":
            print(f"Batch result failed for {video_id}")
            continue

        text = result.result.message.content[0].text
        data = json.loads(text)
        holds = [HoldDescriptionWithRoute(**h).model_dump() for h in data.get("holds", [])]

        buf = io.StringIO()
        if holds:
            writer = csv.DictWriter(buf, fieldnames=holds[0].keys())
            writer.writeheader()
            writer.writerows(holds)

        prefix = config["s3"]["prefixes"]["processed"]
        s3.put_object(
            Bucket=bucket,
            Key=f"{prefix}{video_id}/holds/hold_descriptions.csv",
            Body=buf.getvalue().encode(),
            ContentType="text/csv",
        )
        print(f"[{video_id}] Wrote {len(holds)} hold descriptions.")
