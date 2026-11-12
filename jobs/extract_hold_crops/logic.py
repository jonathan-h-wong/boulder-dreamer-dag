"""
extract_hold_crops logic — per-video version.
Mirrors catalog version but reads from processed/{video_id}/ paths.

Steps:
1. Download bkg_rgb.png from processed/{video_id}/
2. Load holds/hold_descriptions.csv
3. Run Marigold V2 on bkg_rgb.png -> normal map
4. For each hold bbox:
   a. Crop RGB, Normal
   b. Run SAM2 with bbox prompt -> segmentation mask
   c. Upload to processed/{video_id}/holds/hold_crops/{rgb,normal,segmentation}/
"""


def run(video_id: str, config: dict):
    # TODO: implement — mirrors extract_hold_crops_catalog with different S3 paths
    raise NotImplementedError
