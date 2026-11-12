"""
extract_hold_proposals logic — class-agnostic hold proposals for set-of-marks labeling.

Steps:
1. Download processed/{video_id}/bkg_rgb.png
2. SAM2 automatic mask generation (config hold_proposals: points_per_side, crop_n_layers
   for small footholds, min_mask_area_px)
3. Drop masks below min area; de-duplicate with NMS on mask IoU (nms_mask_iou)
4. Number surviving masks 1..N (stable order: top-to-bottom, left-to-right by centroid)
5. Render the overlay (mask contour + number at each mask's interior point) and split it
   into overlapping tiles (tile_px, tile_overlap_px) so small marks stay legible after
   the API downscales images; a mark is drawn in every tile that contains it
6. Upload processed/{video_id}/holds/proposals/:
     masks.npz   (N x H x W bool, or RLE)       marks.png   (full overlay, for humans)
     marks.json  {"image_size": [w, h], "marks": {"<id>": {"bbox", "area", "centroid"}}}
     tiles/{i:02d}.png
"""


def run(video_id: str, config: dict):
    # TODO: implement
    raise NotImplementedError
