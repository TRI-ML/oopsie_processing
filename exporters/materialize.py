"""Materialize the cleaned-subset manifest into a local canonical staging tree.

Reads the stage-10 manifest (one row per clean episode) and stages each selected
episode locally as:

    <raw_stage_dir>/<lab>/<episode_id>/
        episode.h5          # copy of the raw h5 with observations/video_paths
                            # REWRITTEN to canonical role pointers (base.mp4 / ...)
        base.mp4            # canonical base view
        left_wrist.mp4      # canonical left-wrist view
        right_wrist.mp4     # canonical right-wrist view (omitted when it duplicates base)

Why rewrite video_paths: donor labs vary camera naming ACROSS episodes (e.g.
BONN_MB names its single external view 'left' in some episodes and 'global' in
others). Stage 6 (resolve_roles) already resolved each episode's native names to
the three canonical roles; materialize bakes that resolution into the staged h5 so
BOTH exporters (LeRobot + vla_foundry WebDataset, which use a single fixed per-lab
camera_map) see one consistent schema. Raw S3 is read-only and untouched.

This is READ-ONLY on S3: it only GETs the selected h5 + mp4 objects. No raw object
is ever deleted or mutated.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oopsie_processing.common import config, report, s3io  # noqa: E402

# canonical role -> staged filename
ROLE_FILE = {"base": "base.mp4", "left_wrist": "left_wrist.mp4", "right_wrist": "right_wrist.mp4"}


def _load_manifest(cfg) -> list[dict]:
    name = config.require(cfg, "manifest_name")
    path = os.path.join(report.durable_dir(cfg), f"{name}.jsonl")
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _rewrite_video_paths(h5_path: str, pointers: dict) -> None:
    """Replace observations/video_paths with canonical {role: filename} pointers."""
    sdt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(h5_path, "a") as f:
        obs = f.require_group("observations")
        if "video_paths" in obs:
            del obs["video_paths"]
        g = obs.create_group("video_paths")
        for role, fname in pointers.items():
            g.create_dataset(role, data=np.array(fname.encode("utf-8"), dtype=sdt))


def _stage_one(s3, bucket, scratch, row, raw_root) -> dict:
    ep_dir = os.path.join(raw_root, row["lab_id"], row["episode_id"])
    os.makedirs(ep_dir, exist_ok=True)
    cam_keys = row["camera_s3_keys"]  # {role: s3_key}
    base_key = cam_keys["base"]

    # Download each UNIQUE source mp4 once, named by the role it first fills; a role
    # whose s3 key equals the base key is a duplicate (single-arm wrist slot) and
    # points back at base.mp4 instead of re-downloading.
    pointers = {}
    downloaded = {}  # s3_key -> staged filename
    for role in ("base", "left_wrist", "right_wrist"):
        if role not in cam_keys:
            continue
        key = cam_keys[role]
        if role != "base" and key == base_key:
            pointers[role] = ROLE_FILE["base"]  # dup base; no extra file
            continue
        if key in downloaded:
            pointers[role] = downloaded[key]
            continue
        fname = ROLE_FILE[role]
        tmp = s3io.download_to_temp(s3, bucket, key, scratch, suffix=".mp4")
        shutil.move(tmp, os.path.join(ep_dir, fname))
        pointers[role] = fname
        downloaded[key] = fname

    # Copy the h5 and rewrite its camera pointers to the canonical filenames.
    tmp_h5 = s3io.download_to_temp(s3, bucket, row["h5_key"], scratch, suffix=".h5")
    staged_h5 = os.path.join(ep_dir, "episode.h5")
    shutil.move(tmp_h5, staged_h5)
    _rewrite_video_paths(staged_h5, pointers)
    return {"lab": row["lab_id"], "episode_id": row["episode_id"], "dir": ep_dir}


def stage_rows(cfg: dict, rows: list[dict]) -> dict:
    """Stage an EXPLICIT row list into raw_stage_dir WITHOUT wiping the tree.

    Used by the disk-safe streamer (exporters/stream_to_s3.py) to stage one lab
    at a time. Returns {staged, errors} for just these rows. The caller owns
    cleanup (it deletes each lab's staged dir after exporting + uploading it)."""
    bucket = config.require(cfg, "bucket")
    region = config.require(cfg, "region")
    scratch = config.require(cfg, "scratch_dir")
    n_workers = config.require(cfg, "n_workers")
    raw_root = config.require(cfg, "raw_stage_dir")
    os.makedirs(raw_root, exist_ok=True)

    s3 = s3io.make_client(region, config.require(cfg, "max_pool_connections"))
    staged, errors = [], []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_stage_one, s3, bucket, scratch, r, raw_root): r for r in rows}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                staged.append(fut.result())
            except Exception as e:  # fail loudly per-episode, keep the rest
                errors.append({"episode_id": r["episode_id"], "error": f"{type(e).__name__}: {e}"})
    return {"staged": staged, "errors": errors}


def run(stage_cfg_name: str = "subset") -> dict:
    cfg = config.load_merged(stage_cfg_name)
    raw_root = config.require(cfg, "raw_stage_dir")

    rows = _load_manifest(cfg)
    if os.path.isdir(raw_root):
        shutil.rmtree(raw_root)
    os.makedirs(raw_root, exist_ok=True)

    res = stage_rows(cfg, rows)
    staged, errors = res["staged"], res["errors"]

    per_lab = {}
    for s in staged:
        per_lab[s["lab"]] = per_lab.get(s["lab"], 0) + 1
    summary = {"stage": "materialize", "raw_stage_dir": raw_root,
               "n_staged": len(staged), "n_errors": len(errors),
               "per_lab": per_lab, "errors": errors[:20]}
    report.write_json(cfg, "materialize_summary.json", summary)
    print(f"[materialize] staged={len(staged)} errors={len(errors)} per_lab={per_lab}")
    if errors:
        print(f"[materialize] FIRST ERRORS: {errors[:5]}")
    return summary


if __name__ == "__main__":
    run()
