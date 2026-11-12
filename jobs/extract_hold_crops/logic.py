"""
extract_hold_crops logic — Marigold V2 normals + per-hold crops for every labeled hold.
Feeds both the per-video reconstruction and the global hold catalog.

Steps:
1. Download bkg_rgb.png, holds/proposals/masks.npz, holds/hold_descriptions.csv
2. Run Marigold V2 (normals checkpoint, models.marigold_v2.resolution) on bkg_rgb.png ->
   camera-space normal map; upload processed/{video_id}/bkg_normals.png (reused by
   extract_scenegraph as normal supervision, so the 20B model runs once per video)
3. For each hold: mask = union of its mark_ids' proposal masks (SAM2 already ran in
   extract_hold_proposals, so no prompting here); crop RGB, normal and mask at its bbox
4. Upload to processed/{video_id}/holds/hold_crops/{rgb,normal,segmentation}/{hold_id}.png
"""


def run(video_id: str, config: dict):
    # TODO: implement
    raise NotImplementedError
