"""
Hold description extraction agent — global catalog version.
Runs on the hold catalog background plate, produces hold_descriptions.csv.
"""

import json
import csv
import io

from agents.base import get_client, poll_batch
from agents.extract_hold_descriptions_catalog.prompts import SYSTEM_PROMPT, build_hold_description_prompt
from common.schemas import HoldDescription
from common.s3 import get_client as get_s3_client, download_bytes, upload_bytes


def _get_catalog_bkg_b64(config: dict) -> str:
    """Download the catalog background plate and return as base64 JPEG."""
    # TODO: implement — download from s3://{bucket}/hold_catalog/bkg_rgb.jpg
    raise NotImplementedError


def submit_hold_description_batch(config: dict) -> str:
    """Submit a single-item Claude batch for the catalog background plate."""
    client = get_client()
    image_b64 = _get_catalog_bkg_b64(config)

    import anthropic
    requests = [
        anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
            custom_id="catalog_bkg",
            params={
                "model": config["agents"]["model"],
                "max_tokens": config["agents"]["max_tokens"],
                "system": SYSTEM_PROMPT,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_hold_description_prompt(image_b64)},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                    ],
                }],
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

    holds = []
    for result in raw_results:
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text
            data = json.loads(text)
            for h in data.get("holds", []):
                holds.append(HoldDescription(**h).model_dump())

    # Write CSV to S3
    buf = io.StringIO()
    if holds:
        writer = csv.DictWriter(buf, fieldnames=holds[0].keys())
        writer.writeheader()
        writer.writerows(holds)

    prefix = config["s3"]["prefixes"]["hold_catalog"]
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}holds/hold_descriptions.csv",
        Body=buf.getvalue().encode(),
        ContentType="text/csv",
    )
    print(f"Wrote {len(holds)} hold descriptions to S3.")
