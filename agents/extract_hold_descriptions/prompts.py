"""
Prompts for set-of-marks hold labeling. Claude sees the clean plate plus overlay tiles in
which every SAM2 proposal is outlined and numbered; it classifies numbered regions and
never outputs coordinates. The response format is enforced by the HoldLabelResponse JSON
schema (structured outputs), so the prompt defines only the task and field semantics.
"""

SYSTEM_PROMPT = """You are an expert in climbing holds and route setting. You label regions
proposed by an automatic image segmenter on a photo of a bouldering wall. Each region is
outlined and marked with its number. Answer in the required structured format."""


def build_hold_description_prompt(video_id: str, n_marks: int, n_tiles: int) -> str:
    return f"""The first image is the clean background plate of video '{video_id}' (no climber).
The next {n_tiles} images are overlapping tiles of the same plate in which {n_marks} candidate
regions are outlined and numbered; a region near a tile edge may appear in more than one tile
with the same number.

Decide which numbered regions are climbing holds or volumes:
- Only the holds and volumes on the closest wall the camera is focused on count. Regions on
  further walls, bare wall, tape, labels, bolts, mats and people go in non_hold_marks.
- Be comprehensive: small footholds and foot chips are holds, and so are volumes that are
  partly hidden by holds mounted on them.
- If one hold or volume is split across several numbered regions, list all of those
  numbers together in a single entry's mark_ids. A hold mounted on a volume is its own
  entry, separate from the volume.
- Account for every number from the overlay exactly once, either in one hold's mark_ids
  or in non_hold_marks.

For each hold or volume, fill in the schema fields from what is visible. relative_size is
relative to a typical adult hand. depth_class is how far the hold protrudes from the wall.
The description is at most two sentences about shape and grip."""
