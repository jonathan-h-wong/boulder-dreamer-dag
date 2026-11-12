"""
Prompts for hold description extraction — global catalog version.
Applied to background plate images (no climber, no on-route annotation needed).
"""

SYSTEM_PROMPT = """You are a climbing hold identification expert with deep knowledge of
climbing hold types, shapes, and morphology. You will be given an image of a climbing wall
and must identify every hold and volume present."""


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
"""


def build_hold_description_prompt(image_b64: str) -> str:
    return f"""Identify all the climbing holds and volumes in the following image.
Focus on the holds and volumes on the closest wall the camera is focusing on. Ignore contents on further walls.
Be comprehensive. Detect small footholds, and volumes being occluded by mounted holds.

{SCHEMA_DESCRIPTION}

Return a JSON object with a single key "holds" containing a list of hold entries.
Example: {{"holds": [{{...}}, {{...}}]}}"""
