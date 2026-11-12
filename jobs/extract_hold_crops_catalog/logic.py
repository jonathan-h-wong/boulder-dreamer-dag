"""
extract_hold_crops_catalog logic — runs Marigold V2 (normals) + SAM2 (segmentation)
on the catalog background plate, crops {rgb, normal, segmentation} per detected hold,
and uploads to hold_catalog/holds/hold_crops/.
"""


def run(config: dict):
    # TODO:
    # 1. Download bkg_rgb.jpg from S3
    # 2. Load hold_descriptions.csv from S3
    # 3. Run Marigold V2 on bkg_rgb.jpg -> normal map
    # 4. For each hold bbox in hold_descriptions.csv:
    #    a. Crop RGB, Normal
    #    b. Run SAM2 with bbox prompt -> segmentation mask
    #    c. Upload crops to hold_catalog/holds/hold_crops/{rgb,normal,segmentation}/
    raise NotImplementedError
