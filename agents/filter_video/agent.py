"""
Video filter agent — one Claude batch request per video, dispatched together by
agents.base.run_batch_stage. Sees only a handful of sampled frames.

Input:  filtered_videos/{video_id}/frames/*.jpg   (from the extract_filter_frames job)
Output: filtered_videos/{video_id}/filter_result.json   (VideoFilterResult)

Claude returns a FilterJudgment (schema-enforced); `passed` is computed in code, and
missing_climber_metadata is set by the filter flow from climber height in raw/metadata/*.csv.
"""

import base64

from agents.base import output_config, parse_structured
from agents.filter_video.prompts import SYSTEM_PROMPT, build_filter_prompt
from common.s3 import download_bytes, get_client, list_keys
from common.schemas import FilterJudgment, VideoFilterResult


def _video_prefix(video_id: str, config: dict) -> str:
    return f"{config['s3']['prefixes']['filtered_videos']}{video_id}/"


def _frames_b64(video_id: str, config: dict) -> list[str]:
    s3 = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    keys = sorted(k for k in list_keys(s3, bucket, _video_prefix(video_id, config) + "frames/")
                  if k.endswith(".jpg"))
    if not keys:
        raise FileNotFoundError(f"no frames for {video_id}")
    return [base64.b64encode(download_bytes(s3, bucket, k)).decode() for k in keys]


def build_params(video_id: str, config: dict) -> dict:
    frames = _frames_b64(video_id, config)
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
        for b64 in frames
    ]
    content.append({"type": "text", "text": build_filter_prompt(video_id, len(frames))})
    return {
        "model": config["agents"]["filter_video"]["model"],
        "max_tokens": config["agents"]["filter_video"]["max_tokens"],
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content}],
        "output_config": output_config(FilterJudgment, config["agents"]["filter_video"].get("effort")),
    }


def handle_result(video_id: str, message, config: dict) -> None:
    judgment = parse_structured(message, FilterJudgment)
    result = VideoFilterResult.from_judgment(video_id, judgment)
    s3 = get_client(config["s3"]["region"])
    s3.put_object(
        Bucket=config["s3"]["bucket"],
        Key=_video_prefix(video_id, config) + "filter_result.json",
        Body=result.model_dump_json().encode(),
        ContentType="application/json",
    )
