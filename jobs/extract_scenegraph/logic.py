"""
extract_scenegraph logic — fits this video's on-route holds with the same procedure as
the catalog (common/hold_fit.py), so parameters are directly comparable.

Steps:
1. Load hold_descriptions.csv + hold_contacts.csv (on_route, mounted_on), hold_crops,
   hold_frames.json, bkg_depth.png, bkg_normals.png, intrinsics.json from S3
2. Load embeddings.npy + knn_labels.parquet + parametric_holds.py from hold_catalog/
3. For each on-route hold:
   a. Same embedding construction as the catalog -> kNN -> cluster + program;
      intrinsics init = distance-weighted mean of the neighbors' fitted intrinsics
   b. Extrinsics init from this video's own hold frame (never from neighbors):
      theta multi-start by symmetry, tilt 0, offset 0
   c. hold_fit.fit_hold: differentiable SDF ray marching, supervised by the hold mask,
      bkg_normals.png and bkg_depth inside the mask
4. Export scenegraph.json (per hold: hold_id, cluster, program, frame, extrinsics,
   intrinsics, score) to processed/{video_id}/output/scenegraph.json
"""


def run(video_id: str, config: dict):
    # TODO: implement (fitting lives in common/hold_fit.py)
    raise NotImplementedError
