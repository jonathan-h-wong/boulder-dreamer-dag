"""
extract_scenegraph logic — constructs the optimized scenegraph for a video.

Steps:
1. Load hold_descriptions.csv, hold_crops, bkg_rgb.png, bkg_depth.png from S3
2. Load embeddings.npy + knn_labels from hold_catalog
3. For each on-route hold:
   a. Query KNN catalog embedding -> nearest cluster + warm-start parametric params
   b. Initialize parametric hold on 2D bbox + lifted 3D depth from bkg_depth
   c. Run differentiable SDF ray marching for depth and normals
   d. Supervision:
      - Normal: Marigold V2 on-the-fly normal map (untouched holds linearly interpolated)
      - Depth: untouched hold depth linear interpolation
   e. Optimize hold intrinsic parameters
4. Run gaussian splatting on hold segmentation masks + bkg plate (no color)
5. Export scenegraph.json to processed/{video_id}/output/scenegraph.json
"""


def run(video_id: str, config: dict):
    # TODO: implement differentiable SDF optimization + gsplat
    raise NotImplementedError
