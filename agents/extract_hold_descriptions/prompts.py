"""
Prompt for the description pass of set-of-marks hold perception. The holds are already
detected (detect_holds); Claude sees the clean plate plus overlay tiles, all at one scale,
in which only the detected holds are outlined and numbered, each tile followed by a clean
copy, and fills in the schema fields for every detected hold. Seeing every hold at the
same scale is what lets it tell a foot chip from a jug; the written hold-type definitions
are what make the types agree (both measured in the set-of-marks experiments).
"""

SYSTEM_PROMPT = """You are an expert in climbing holds and route setting. You describe climbing
holds that are outlined and numbered on a photo of a bouldering wall. Answer in the required
structured format."""

HOLD_TYPES = """Hold types (choose by shape):
- foot_chip: small, low-profile hold too small or shallow to grab with the hand
- crimp_edge: thin edge the fingertips pull on
- jug: deep, incut hold the whole hand wraps
- sloper: rounded hold with no edge, held by friction
- pinch: gripped by squeezing from both sides
- pocket: a hole the fingers go into
- rail: long, narrow edge
- horn_knob: protruding knob or horn to wrap
- wedge_chip: small angular wedge
- flake: thin, plate-like flake
- volume_pyramid, volume_dome, volume_other: large shapes other holds mount on
- macro_slab: large, flat macro hold
- other: none of these"""


def build_hold_description_prompt(video_id: str, n_tiles: int, holds: list[list[int]]) -> str:
    listing = "\n".join(f"- hold {g[0]}: regions {', '.join(map(str, g))}" for g in holds)
    return f"""The first image is the clean background plate of video '{video_id}' (no climber).
The next {2 * n_tiles} images are {n_tiles} overlapping tiles of the same plate, all at the same
scale. Each tile is shown first with the climbing holds and volumes outlined and labeled with
their hold number, then again without outlines so their shape is unobscured. A hold near a
tile edge may appear in more than one tile with the same number.

These {len(holds)} holds and volumes on the wall were already detected; each is listed as its
hold number followed by the segmenter regions it is made of:
{listing}

Describe every one of them: give exactly one entry per hold, with its hold number. Every tile
is at the same scale, so judge size and depth against the whole wall and the neighboring
holds, not against the frame.

Fill in the schema fields from what is visible. relative_size is relative to a typical adult
hand. depth_class is how far the hold protrudes from the wall. The description is at most two
sentences about shape and grip.

{HOLD_TYPES}"""
