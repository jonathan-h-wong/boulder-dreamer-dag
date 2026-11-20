"""
extract_bkg_plate logic — clean background plate (the wall without the climber).

The camera is static (enforced by the filter), so every pixel sees bare wall in most
frames and the climber in a few; the per-pixel temporal median keeps the wall.

Steps:
1. Download raw/videos/{video_id}.mp4 to local scratch.
2. Decode front to back (sequential grab(): seeking is inexact on H.264), keeping every
   Nth frame: N = downsampling.bkg_plate_rate, raised if needed so that at most
   downsampling.bkg_plate_max_frames frames are kept (bounds memory on long/4K videos).
   OpenCV applies the phone's rotation metadata, so frames come out upright.
3. Per-pixel median over the kept frames, in row bands (np.median copies its input).
4. Upload processed/{video_id}/bkg_rgb.png at the video's native resolution, so plate
   pixels line up with every other per-frame output, and bkg_plate.json (frames used).

Known limit: a climber who stays in one spot for more than half of the kept frames
(long rest) leaves a ghost in the plate.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from common.s3 import get_client

ROW_BAND = 64


def frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:                              # container doesn't report it: count by decoding
        while cap.grab():
            n += 1
    cap.release()
    return n


def stride_for(n_total: int, rate: int, max_frames: int) -> int:
    return max(int(rate), math.ceil(n_total / max_frames))


def read_frames(path: str, stride: int, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("unreadable video (OpenCV could not open it)")
    frames, idx = [], 0
    while len(frames) < max_frames and cap.grab():
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(frame)
        idx += 1
    cap.release()
    return frames


def median_plate(frames: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(frames)                # T x H x W x 3 uint8
    frames.clear()                          # don't hold two copies (4K: ~1.6 GB each)
    out = np.empty(stack.shape[1:], np.uint8)
    for y in range(0, stack.shape[1], ROW_BAND):
        out[y:y + ROW_BAND] = np.median(stack[:, y:y + ROW_BAND], axis=0)
    return out


def run(video_id: str, config: dict):
    s3 = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    prefixes = config["s3"]["prefixes"]
    ds = config["downsampling"]
    out_prefix = f"{prefixes['processed']}{video_id}/"

    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "video.mp4")
        s3.download_file(bucket, f"{prefixes['raw_videos']}{video_id}.mp4", path)
        n_total = frame_count(path)
        if n_total <= 0:
            raise ValueError("video has no decodable frames")
        stride = stride_for(n_total, ds["bkg_plate_rate"], ds["bkg_plate_max_frames"])
        frames = read_frames(path, stride, ds["bkg_plate_max_frames"])
    if len(frames) < 3:
        raise ValueError(f"only {len(frames)} frames decoded; need >= 3 for a median plate")

    plate = median_plate(frames)
    ok, buf = cv2.imencode(".png", plate)
    if not ok:
        raise RuntimeError("PNG encode failed")
    s3.put_object(Bucket=bucket, Key=out_prefix + "bkg_rgb.png", Body=buf.tobytes(),
                  ContentType="image/png")
    info = {"frames_used": len(frames), "stride": stride, "frame_count": n_total,
            "width": int(plate.shape[1]), "height": int(plate.shape[0])}
    s3.put_object(Bucket=bucket, Key=out_prefix + "bkg_plate.json",
                  Body=json.dumps(info).encode(), ContentType="application/json")
