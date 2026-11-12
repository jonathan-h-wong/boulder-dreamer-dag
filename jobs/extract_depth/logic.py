"""
extract_depth logic — MoGe-2 depth model, per-frame depth + intrinsics.

Steps:
1. Download raw video from S3
2. Sample every Nth frame (depth_rate)
3. Run MoGe-2 on each frame -> (depth_map, intrinsics)
4. Compute median over all depth maps -> bkg_depth.png
5. Upload:
   - processed/{video_id}/bkg_depth.png
   - processed/{video_id}/depth_samples/imgs/{frame_id}.png
   - processed/{video_id}/depth_samples/intrinsics/{frame_id}.json
"""


def run(video_id: str, config: dict):
    # TODO: implement MoGe-2 inference loop
    raise NotImplementedError
