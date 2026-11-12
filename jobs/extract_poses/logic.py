"""
extract_poses logic — GVHMR full-sequence pose estimation + downsampling.

Steps:
1. Download raw video + processed/{video_id}/intrinsics.json (from extract_depth)
2. Run GVHMR on the full video sequence with the known intrinsics (not GVHMR's own focal
   estimate) -> per-frame SMPL body poses, in the same camera as the depth maps
3. Subsample output to keypose_rate
4. Upload poses.npy to processed/{video_id}/poses.npy
   Shape: (N_keyposes, ...) — SMPL pose params + global translation
"""


def run(video_id: str, config: dict):
    # TODO: implement GVHMR inference + subsampling
    raise NotImplementedError
