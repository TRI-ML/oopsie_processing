"""Read-only S3 access helpers.

Everything here is non-destructive: GET/HEAD and presign only. Small h5 files
are downloaded to a temp dir and the caller deletes them immediately. mp4
integrity is probed over a presigned URL with ffprobe reading only the container
header (no -count_frames, so no full download) or a bounded byte range.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile

import boto3
from botocore.config import Config

for _n in ("boto3", "botocore", "s3transfer", "urllib3"):
    logging.getLogger(_n).setLevel(logging.ERROR)


def make_client(region: str, max_pool: int):
    return boto3.client(
        "s3", region_name=region,
        config=Config(max_pool_connections=max_pool, retries={"max_attempts": 3}),
    )


def download_to_temp(s3, bucket: str, key: str, scratch_dir: str, suffix: str = ".h5") -> str:
    os.makedirs(scratch_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=scratch_dir, suffix=suffix)
    os.close(fd)
    s3.download_file(bucket, key, path)
    return path


def rm(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def head_size(s3, bucket: str, key: str) -> int:
    return int(s3.head_object(Bucket=bucket, Key=key)["ContentLength"])


def range_bytes(s3, bucket: str, key: str, n: int) -> bytes:
    """First n bytes of an object via HTTP Range (no full download)."""
    resp = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{n-1}")
    return resp["Body"].read()


def presign(s3, bucket: str, key: str, expires: int) -> str:
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)


def ffprobe_video(s3, bucket: str, key: str, expires: int, timeout: int) -> dict:
    """Header-only probe of the first video stream over a presigned URL.

    Returns dict with nb_frames (container metadata), r_frame_rate, duration,
    width, height; or {'error': ...}. Does NOT decode/count frames, so it reads
    only the moov atom via range requests rather than the whole file.
    """
    url = presign(s3, bucket, key, expires)
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=nb_frames,r_frame_rate,avg_frame_rate,duration,width,height,codec_name",
        "-of", "json", url,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "ffprobe_timeout"}
    if r.returncode != 0:
        return {"error": "ffprobe_rc_%d" % r.returncode, "stderr": r.stderr.strip()[:200]}
    try:
        streams = json.loads(r.stdout).get("streams", [])
    except json.JSONDecodeError:
        return {"error": "ffprobe_badjson"}
    if not streams:
        return {"error": "no_video_stream"}
    s = streams[0]

    def _rate(v):
        if not v or v == "0/0":
            return None
        try:
            num, den = v.split("/")
            den = float(den)
            return float(num) / den if den else None
        except Exception:
            return None

    nbf = s.get("nb_frames")
    try:
        nbf = int(nbf) if nbf not in (None, "N/A") else None
    except ValueError:
        nbf = None
    dur = s.get("duration")
    try:
        dur = float(dur) if dur not in (None, "N/A") else None
    except ValueError:
        dur = None
    fps = _rate(s.get("r_frame_rate")) or _rate(s.get("avg_frame_rate"))
    # If the container omits nb_frames, derive from duration*fps as a fallback.
    nb_derived = None
    if nbf is None and dur is not None and fps:
        nb_derived = round(dur * fps)
    return {
        "nb_frames": nbf,
        "nb_frames_derived": nb_derived,
        "fps": fps,
        "duration_s": dur,
        "width": s.get("width"),
        "height": s.get("height"),
        "codec": s.get("codec_name"),
    }
