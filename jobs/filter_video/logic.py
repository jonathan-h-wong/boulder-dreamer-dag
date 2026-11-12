"""
filter_video logic — frame extraction for the Claude batch agent.
Downloads raw video from S3, extracts evenly-spaced downsampled frames,
uploads frame directory for the agent to consume.
"""

import cv2
import boto3
import tempfile
from pathlib import Path
from common.s3 import get_client, upload_image


def run(video_id: str, config: dict):
    s3 = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    rate = config["downsampling"]["bkg_plate_rate"]

    # TODO: download raw video from s3://{bucket}/raw_videos/{video_id}.mp4
    # TODO: extract every Nth frame with cv2
    # TODO: upload frames to s3://{bucket}/filtered_videos/{video_id}/frames/
    raise NotImplementedError
