"""
extract_depth logic — MoGe-3 depth model, per-frame depth + intrinsics.

Steps:
1. Download raw/videos/{video_id}.mp4
2. Sample every Nth frame (depth_rate)
3. Run MoGe-3 on each frame -> (depth_map, intrinsics)
4. Compute median over all depth maps -> bkg_depth.png
5. Static camera (guaranteed by filtering): per-parameter median of the per-frame
   intrinsics -> one K for the whole video (intrinsics.json). GVHMR, grounded poses, hold
   contacts and the wall mesh all use this K.
6. Upload:
   - processed/{video_id}/bkg_depth.png
   - processed/{video_id}/depth_samples/imgs/{frame_id}.png
   - processed/{video_id}/depth_samples/intrinsics/{frame_id}.json
   - processed/{video_id}/intrinsics.json
"""


def run(video_id: str, config: dict):
    # TODO: implement MoGe-3 inference loop
    raise NotImplementedError
