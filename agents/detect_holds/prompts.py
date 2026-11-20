"""
Prompt for the detection pass of set-of-marks hold perception. Claude sees the clean plate,
then one close-up per SAM2 proposal (the mark centered, shown as photographed and again
outlined), and decides which proposals are holds and which ones belong to the same hold.
Only detection here; the fields are judged in extract_hold_descriptions, at one scale.
"""

SYSTEM_PROMPT = """You are an expert in climbing holds and route setting. You judge regions
proposed by an automatic image segmenter on a photo of a bouldering wall. Answer in the
required structured format."""


def build_detection_prompt(video_id: str, mark_ids: list[int]) -> str:
    return f"""The first image is the clean background plate of video '{video_id}' (no climber).
After it come {len(mark_ids)} close-ups, one per candidate region found by an automatic
segmenter. Each close-up is introduced by its mark number and its position on the plate, and
shows the same window twice: on the left as photographed, on the right with the candidate
region outlined in magenta. The region is centered in the window.

Decide which of these numbered regions are climbing holds or volumes: {", ".join(map(str, mark_ids))}.
- Only the holds and volumes on the closest wall the camera is focused on count. Regions on
  further walls, bare wall, tape, labels, bolts, mats and people go in non_hold_marks. Use
  the plate and each mark's position to judge which wall a region is on.
- Be comprehensive: small footholds and foot chips are holds, and so are volumes that are
  partly hidden by holds mounted on them.
- If one hold or volume is split across several numbered regions (their close-ups show parts
  of the same object), list all of those numbers together in a single entry's mark_ids. A
  hold mounted on a volume is its own entry, separate from the volume.
- Account for each of your {len(mark_ids)} numbers exactly once, either in one hold's
  mark_ids or in non_hold_marks."""
