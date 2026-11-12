"""
curate_hold_catalog logic — builds embedding matrix, clusters holds, generates parametric holds.

Built by aggregating per-video hold perception — there is no separate catalog detection
pass. Membership is the explicit `video_ids` list (recorded in the marker); videos added
later query this catalog without being in it until it is rebuilt.

Pipeline:
1. For each member video, load processed/{video_id}/holds/hold_descriptions.csv,
   hold_crops/{rgb, normal, segmentation}, hold_frames.json, intrinsics.json,
   bkg_normals.png and bkg_depth.png. Holds are all labeled holds (on- and off-route);
   the catalog is route agnostic.
2. Construct embedding:
   - One-hot encode categorical schema fields
   - Encode free-text description with a small text transformer
   - Compute normal histogram from normal crop (instead of DINO)
   - Concatenate / fuse into single embedding vector per hold
3. PCA -> HDBSCAN -> cluster assignment
4. For each cluster: sample up to 50 farthest-point samples
5. Claude agent: feed {RGB, Normal} crops + schema descriptions ->
   generate procedural program for hold shape (SDF / parametric params)
6. Fit with common/hold_fit.py (the same procedure extract_scenegraph uses): frames from
   hold_frames.json, extrinsic residuals + intrinsics fitted, accepted at
   hold_fit.min_score. Exemplars first (to settle each cluster's program), then every
   remaining member hold, warm-started from its cluster's exemplars.
7. Export:
   - embeddings.npy (N x D embedding matrix)
   - knn_labels.parquet (video_id, hold_id, cluster_id, intrinsics, extrinsics, score)
     Only intrinsics are used as a warm start downstream; extrinsics are kept for analysis.
   - parametric_holds.py (generated procedural programs)
"""


def run(video_ids: list[str], config: dict):
    # TODO: implement full pipeline described above
    raise NotImplementedError
