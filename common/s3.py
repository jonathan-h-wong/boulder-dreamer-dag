"""
Shared S3 helpers for reading/writing pipeline artifacts.
All paths follow the convention:
  s3://{bucket}/processed/{video_id}/{artifact}
  s3://{bucket}/hold_catalog/{artifact}
"""

import boto3
from pathlib import PurePosixPath
from typing import Union
import numpy as np
import io


def get_client(region: str = "us-east-1"):
    return boto3.client("s3", region_name=region)


def s3_path(bucket: str, *parts: str) -> str:
    return "s3://" + str(PurePosixPath(bucket, *parts))


def upload_bytes(client, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream"):
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def download_bytes(client, bucket: str, key: str) -> bytes:
    resp = client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def upload_numpy(client, bucket: str, key: str, arr: np.ndarray):
    buf = io.BytesIO()
    np.save(buf, arr)
    upload_bytes(client, bucket, key, buf.getvalue(), content_type="application/octet-stream")


def download_numpy(client, bucket: str, key: str) -> np.ndarray:
    data = download_bytes(client, bucket, key)
    return np.load(io.BytesIO(data))


def upload_image(client, bucket: str, key: str, image_bytes: bytes):
    ext = key.rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "application/octet-stream")
    upload_bytes(client, bucket, key, image_bytes, content_type=mime)


def list_keys(client, bucket: str, prefix: str) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys
