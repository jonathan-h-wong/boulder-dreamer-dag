"""
extract_filter_frames logic — samples frames for the filter agent.

Steps:
1. Download raw/videos/{video_id}.mp4
2. Extract filter.n_frames evenly spaced frames across the full duration, downscaled
3. Upload to filtered_videos/{video_id}/frames/{idx:04d}.jpg
"""

from common.s3 import get_client


def run(video_id: str, config: dict):
    s3 = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    n_frames = config["filter"]["n_frames"]

    # TODO: download raw video from s3://{bucket}/{raw_videos}{video_id}.mp4
    # TODO: extract n_frames evenly spaced frames with cv2
    # TODO: upload to s3://{bucket}/{filtered_videos}{video_id}/frames/
    raise NotImplementedError
