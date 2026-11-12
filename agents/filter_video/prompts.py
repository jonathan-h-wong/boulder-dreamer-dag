"""
Prompts for the video filter agent. The response format is enforced separately by the
FilterJudgment JSON schema (structured outputs); the prompt only defines the criteria.
"""

SYSTEM_PROMPT = """You review short climbing-gym videos to decide whether each is usable for
3D reconstruction of a single climber on a single wall. You see a handful of frames sampled
evenly across the whole video, in order. Judge each criterion independently, and answer in
the required structured format."""


def build_filter_prompt(video_id: str, n_frames: int) -> str:
    return f"""The {n_frames} images are frames sampled evenly across video '{video_id}', in order.

Mark a criterion true only if the frames show it clearly:
- too_many_climbers: two or more people are on the wall or climbing at the same time in any
  frame. Downstream pose extraction assumes exactly one climber. Spectators or belayers
  standing on the floor do not count.
- camera_blur: frames are significantly blurry or out of focus, enough to hide hold edges.
  Motion blur on the climber alone does not count.
- moving_camera: the camera pans, tilts, zooms, or is handheld. Compare the wall and floor
  across frames; the background plate is recovered by median filtering, which needs a static
  camera.
- no_floor_plane: the gym floor or mat is not visible in any frame. The floor sets the
  gravity-aligned world frame.

In notes, give one short sentence naming what failed, or leave it empty if nothing did."""
