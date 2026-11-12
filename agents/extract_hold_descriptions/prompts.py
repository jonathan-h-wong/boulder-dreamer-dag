"""
Prompts for per-video hold description extraction.
Extends catalog version with on-route annotation using the clean plate + pose overlays.
"""

SYSTEM_PROMPT = """You are a climbing hold identification expert and route reading specialist.
You will be given an image of a climbing wall — the clean background plate — with an optional
overlay showing keypose detections of the climber. Identify all holds and annotate whether
each hold is on-route (touched by the climber)."""


SCHEMA_DESCRIPTION = """
For each detected hold or volume, return an entry with these fields:
- hold_id: unique string identifier (e.g. "hold_001")
- hold_type: one of [jug, crimp_edge, rail, sloper, pinch, pocket, horn_knob, foot_chip,
  volume_pyramid, volume_dome, volume_other, macro_slab, wedge_chip, flake, other]
- outline: one of [round, oval, elongated, rectangular, triangular, irregular, crescent]
- profile: one of [convex_dome, flat_top, dished, hollowed, ridge, faceted, mixed]
- lip: one of [none, rounded, sharp, incut]
- opening: one of [none, single_finger, two_finger, three_finger, hand_scoop, slot]
- symmetry: one of [bilateral, radial, none]
- relative_size: one of [tiny, small, medium, large, volume]
- depth_class: one of [shallow, medium, deep, very_deep]
- description: free text, maximum two sentences
- bbox: [x_min, y_min, x_max, y_max] in pixel coordinates
- on_route: boolean — true if the climber's hands or feet touch this hold or volume.
  If a hold is mounted on a volume and the hold is touched, the volume is also on_route.
"""


def build_hold_description_prompt(video_id: str) -> str:
    return f"""Identify all the climbing holds and volumes in the following image for video '{video_id}'.
Focus on the holds and volumes on the closest wall the camera is focusing on. Ignore contents on further walls.
Be comprehensive. Detect small footholds, and volumes being occluded by mounted holds.
Using the pose overlay (if present), determine which holds are on-route.

{SCHEMA_DESCRIPTION}

Return a JSON object with a single key "holds" containing a list of hold entries.
Example: {{"holds": [{{...}}, {{...}}]}}"""
