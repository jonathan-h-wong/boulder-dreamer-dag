"""
extract_wall_mesh logic — wall geometry shared by the hold catalog and the scenegraph.

Inputs: bkg_depth.png, intrinsics.json, bkg_normals.png, holds/hold_descriptions.csv,
holds/hold_contacts.csv (on_route, mounted_on), holds/hold_crops/segmentation/.

Steps:
1. Bare-wall depth: inpaint (linear interpolation) the depth under EVERY hold and volume
   mask -> bare_wall_depth.npy. This is the surface hold frames sit on.
2. Hold frames (the fitting init for common/hold_fit.py, used by both the catalog and
   the scenegraph): for each hold,
     origin  = mask centroid lifted onto the bare-wall surface with K
     normal  = local plane fit of the bare wall around the hold; for a hold with
               mounted_on set, plane fit of that volume's own depth instead
     tangent = mask principal axis (2D PCA) projected into the plane
   -> holds/hold_frames.json {hold_id: {origin, normal, tangent, source: wall|volume}}
3. Input mesh (dataset input): linearly interpolate depth under untouched holds
   (on_route False), back-project with K, mesh (Delaunay on the image grid)
   -> input/wall_mesh.glb
"""


def run(video_id: str, config: dict):
    # TODO: implement
    raise NotImplementedError
