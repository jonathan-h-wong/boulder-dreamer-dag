"""
post_processing logic — floor plane fit, coordinate frame normalization, metric scaling.

Steps:
1. Load wall_mesh.glb, output/poses.npy, scenegraph.json, climber.json
   ({"height_cm", "ape_index_cm"}, written by the orchestrator from raw/metadata) from S3
2. Fit floor plane to wall mesh base vertices (RANSAC plane fit)
3. Compute gravity-aligned world frame:
   - Floor plane normal -> up axis
   - Origin: center of wall mesh bounding box, snapped to the ground plane
4. Transform wall_mesh, poses, and hold world positions into this shared frame
5. Metric scaling: use known climber height from metadata to compute scale factor
   and apply uniformly to the scene
6. Overwrite scenegraph.json + poses.npy with grounded, metric-scaled versions
"""

import numpy as np


def run(video_id: str, config: dict):
    # TODO: implement floor plane RANSAC + coordinate normalization + metric scaling
    raise NotImplementedError
