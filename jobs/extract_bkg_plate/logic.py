"""
extract_bkg_plate logic — median filter over downsampled video frames.

Steps:
1. Download raw video from S3
2. Sample every Nth frame (bkg_plate_rate from config)
3. Stack frames and compute per-pixel median -> background RGB plate
4. Upload bkg_rgb.png to processed/{video_id}/bkg_rgb.png
"""

import cv2
import numpy as np
from common.s3 import get_client, upload_image
import tempfile
from pathlib import Path


def run(video_id: str, config: dict):
    s3 = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    rate = config["downsampling"]["bkg_plate_rate"]
    prefix = config["s3"]["prefixes"]["processed"]

    # TODO:
    # 1. Download raw video
    # 2. Extract frames at 'rate' interval with cv2.VideoCapture
    # 3. np.median(frames, axis=0) -> bkg_rgb
    # 4. cv2.imencode('.png', bkg_rgb) and upload
    raise NotImplementedError
