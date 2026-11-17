"""
extract_filter_frames logic — samples frames for the filter agent.

Steps:
1. Download raw/videos/{video_id}.mp4 to local scratch.
2. Decode once, front to back, keeping filter.n_frames frames at the centers of n equal
   time slices (skips the first/last instants, where the camera is often still being set
   down). Sequential grab() is used instead of seeking: seeking is inexact on H.264.
   OpenCV applies the phone's rotation metadata, so frames come out upright.
3. Downscale to filter.max_side_px on the long side (~450 image tokens per frame).
4. Score camera motion on the downscaled frames (camera_motion below).
5. Replace filtered_videos/{video_id}/frames/ with {idx:02d}.jpg and write
   filtered_videos/{video_id}/video_info.json (fps, frame count, duration, size,
   camera_motion_px). The motion THRESHOLD is applied by the filter flow when it
   aggregates, so changing it never reruns this job or the Claude batch.

An unreadable video raises; the flow records it as "not evaluated" with the reason.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2

from common.s3 import get_client, list_keys


def frame_targets(n_total: int, n: int) -> list[int]:
    """Centers of n equal slices of [0, n_total), unique and sorted."""
    return sorted({min(n_total - 1, int((i + 0.5) * n_total / n)) for i in range(n)})


def _count_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def sample_frames(path: str, n: int, max_side: int) -> tuple[list, dict]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("unreadable video (OpenCV could not open it)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if n_total <= 0:                       # container doesn't report it: count by decoding
        n_total = _count_frames(path)
    if n_total <= 0:
        raise ValueError("video has no decodable frames")

    targets = frame_targets(n_total, n)
    wanted, frames = set(targets), []
    cap = cv2.VideoCapture(path)
    idx = 0
    while len(frames) < len(targets) and cap.grab():
        if idx in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames.append((idx, frame))
        idx += 1
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded (reported {n_total})")

    h, w = frames[0][1].shape[:2]
    scale = min(1.0, max_side / max(h, w))
    out = [(i, cv2.resize(f, (round(f.shape[1] * scale), round(f.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA) if scale < 1 else f)
           for i, f in frames]
    info = {
        "fps": round(fps, 3),
        "frame_count": n_total,
        "duration_s": round(n_total / fps, 2) if fps > 0 else None,
        "width": w,
        "height": h,
        "sampled_frames": [i for i, _ in frames],
    }
    return out, info


def camera_motion(frames: list) -> dict:
    """
    Cheap static-camera check on the sampled frames. ORB features of every frame are
    matched to the first frame, and a similarity transform (rotation + scale +
    translation) is fitted with RANSAC. The wall is static and dominates the features,
    so the climber's motion falls out as outliers. Motion of a frame = the largest
    displacement of an image corner under that transform, in pixels of these
    (downscaled) frames; the video's score is the maximum over frames.

    Returns {"camera_motion_px": float | None, "motion_per_frame": [...]}. None means too
    few reliable matches (e.g. a blank or very blurry frame) — no evidence either way.
    """
    import numpy as np

    orb = cv2.ORB_create(nfeatures=2000)
    feats = [orb.detectAndCompute(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None) for f in frames]
    h, w = frames[0].shape[:2]
    corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kp0, des0 = feats[0]
    per_frame: list[float | None] = []
    for kp, des in feats[1:]:
        if des0 is None or des is None:
            per_frame.append(None)
            continue
        matches = matcher.match(des, des0)
        if len(matches) < 30:
            per_frame.append(None)
            continue
        src = np.float32([kp[m.queryIdx].pt for m in matches])
        dst = np.float32([kp0[m.trainIdx].pt for m in matches])
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                 ransacReprojThreshold=2.0)
        if M is None or int(inliers.sum()) < 20:
            per_frame.append(None)
            continue
        moved = corners @ M[:, :2].T + M[:, 2]
        per_frame.append(round(float(np.linalg.norm(moved - corners, axis=1).max()), 2))
    valid = [d for d in per_frame if d is not None]
    return {"camera_motion_px": max(valid) if valid else None, "motion_per_frame": per_frame}


def run(video_id: str, config: dict):
    s3 = get_client(config["s3"]["region"])
    bucket = config["s3"]["bucket"]
    prefixes = config["s3"]["prefixes"]
    cfg = config["filter"]
    out_prefix = f"{prefixes['filtered_videos']}{video_id}/"

    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "video.mp4")
        s3.download_file(bucket, f"{prefixes['raw_videos']}{video_id}.mp4", path)
        frames, info = sample_frames(path, cfg["n_frames"], cfg["max_side_px"])
    info.update(camera_motion([f for _, f in frames]))

    # Replace, never mix: a smaller n_frames must not leave old frames behind.
    old = list_keys(s3, bucket, out_prefix + "frames/")
    if old:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in old]})
    quality = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg["jpeg_quality"])]
    for n, (_, frame) in enumerate(frames):
        ok, buf = cv2.imencode(".jpg", frame, quality)
        if not ok:
            raise RuntimeError("JPEG encode failed")
        s3.put_object(Bucket=bucket, Key=f"{out_prefix}frames/{n:02d}.jpg",
                      Body=buf.tobytes(), ContentType="image/jpeg")
    s3.put_object(Bucket=bucket, Key=out_prefix + "video_info.json",
                  Body=json.dumps(info).encode(), ContentType="application/json")
