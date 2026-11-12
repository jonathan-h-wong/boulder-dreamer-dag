"""
extract_poses logic — GVHMR full-sequence pose estimation + downsampling.

Steps:
1. Download raw video from S3
2. Run GVHMR on the full video sequence -> per-frame SMPL body poses
3. Subsample output to keypose_rate
4. Upload poses.npy to processed/{video_id}/poses.npy
   Shape: (N_keyposes, ...) — SMPL pose params + global translation
"""


def run(video_id: str, config: dict):
    # TODO: implement GVHMR inference + subsampling
    raise NotImplementedError
