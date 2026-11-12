"""
extract_wall_mesh logic — lifts depth map to 3D wall mesh, excluding climber region.

Steps:
1. Load bkg_depth.png, intrinsics, hold_descriptions.csv (untouched holds) from S3
2. Linearly interpolate depth values under untouched hold regions to smooth wall
3. Back-project depth map to 3D point cloud using intrinsics
4. Mesh the point cloud (Poisson or Delaunay on the projected grid)
5. Export wall_mesh.glb to processed/{video_id}/input/wall_mesh.glb
"""

import numpy as np


def run(video_id: str, config: dict):
    # TODO: implement depth inpainting + meshing
    raise NotImplementedError
