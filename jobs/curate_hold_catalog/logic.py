"""
curate_hold_catalog logic — builds embedding matrix, clusters holds, generates parametric holds.

Pipeline:
1. Load hold_descriptions.csv + hold crops ({rgb, normal, segmentation}) from S3
2. Construct embedding:
   - One-hot encode categorical schema fields
   - Encode free-text description with a small text transformer
   - Compute normal histogram from normal crop (instead of DINO)
   - Concatenate / fuse into single embedding vector per hold
3. PCA -> HDBSCAN -> cluster assignment
4. For each cluster: sample up to 50 farthest-point samples
5. Claude agent: feed {RGB, Normal} crops + schema descriptions ->
   generate procedural program for hold shape (SDF / parametric params)
   Optimization target: segmentation IoU + normal alignment > 80%
6. Fit remaining holds with warm-started parametric parameters
7. Export:
   - embeddings.npy (N x D embedding matrix)
   - knn_labels.parquet (hold_id, cluster_id, fit_params...)
   - parametric_holds.py (generated procedural programs)
"""


def run(config: dict):
    # TODO: implement full pipeline described above
    raise NotImplementedError
