"""
Prompts for the video filter agent.
"""

SYSTEM_PROMPT = """You are a computer vision analyst reviewing climbing gym footage.
Your job is to evaluate short video clips and determine whether each clip is suitable
for downstream 3D reconstruction and motion analysis. Be concise and return structured JSON."""


def build_filter_prompt(video_id: str, frame_b64s: list[str]) -> str:
    """
    video_id: identifier string
    frame_b64s: list of base64-encoded JPEG frames (downsampled from video)
    Returns a user message string.
    """
    return f"""Evaluate the following sampled frames from video '{video_id}' for these criteria:

1. too_many_climbers: Are there 2 or more climbers visible simultaneously at any point?
2. camera_blur: Is the camera significantly blurry or out of focus?
3. moving_camera: Is the camera panning, tilting, or otherwise moving? (not a static mount)
4. no_floor_plane: Is the gym floor NOT visible in any frame?
5. missing_climber_metadata: (Do not evaluate — set to false, handled externally)

Return a JSON object with this exact schema:
{{
  "video_id": "{video_id}",
  "passed": <bool>,
  "too_many_climbers": <bool>,
  "camera_blur": <bool>,
  "moving_camera": <bool>,
  "no_floor_plane": <bool>,
  "missing_climber_metadata": false,
  "notes": "<optional short explanation>"
}}

A video passes only if ALL criteria are false."""
