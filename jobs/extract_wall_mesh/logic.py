"""
extract_wall_mesh logic — wall geometry shared by the hold catalog and the scenegraph.

Inputs: bkg_depth.png, intrinsics.json, bkg_normals.png, holds/hold_descriptions.csv,
holds/hold_contacts.csv (on_route, mounted_on), holds/hold_crops/segmentation/.

Steps:
1. Bare-wall depth: inpaint (linear interpolation) the depth under EVERY hold and volume
   mask -> bare_wall_depth.npy. This is the surface hold frames sit on.
2. Hold frames (the fitting init for jobs/hold_fit.py, used by both the catalog and
   the scenegraph): for each hold,
     origin  = mask centroid lifted onto the bare-wall surface with K
     normal  = local plane fit of the bare wall around the hold; for a hold with
               mounted_on set, plane fit of that volume's own depth instead
     tangent = mask principal axis (2D PCA) projected into the plane
   -> holds/hold_frames.json {hold_id: {origin, normal, tangent, source: wall|volume}}
3. Input mesh (dataset input): the clean wall with no hold bumps, i.e. bare_wall_depth
   back-projected with K and meshed (Delaunay on the image grid) -> input/wall_mesh.glb.
   The bumps are not lost: bkg_depth / bkg_normals keep them as fitting supervision.
(hold_contacts.csv is still needed for mounted_on, which picks volume-face frames.)
"""


def run(video_id: str, config: dict):
    # TODO: implement
    raise NotImplementedError
