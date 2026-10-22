"""Export the cleaned subset to a LeRobot v2.1 dataset for pi05/openpi training.

Self-contained port of dsrl_openpi_flip's convert_oopsie_to_lerobot.py, but driven
by the stage-10 manifest + the local canonical staging tree (materialize.py) instead
of re-listing S3. Emits exactly the columns the pi05_oopsie_smoke / _clean_subset
transform stack consumes:

    observation.state   [18] f32        action [18] f32      action_mask [18] f32
    meta_prompt (string)  success [1] f32  speed_bucket [1] i64  embodiment_id [1] i64
    observation.images.{base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb}  (video)
    task = language_instruction

meta_prompt is rebuilt from the manifest, identical to the proven converter:
    "<opt=success|failure> <speed=0|1|2> <emb=NAME> <op=OP> "  (trailing space)

Action/state are re-harmonized from the STAGED h5 via stage5 (the single source of
truth for the common-18 layout + mask). Run under the openpi uv env (has lerobot):

    cd /home/anthonyliang/Documents/dsrl_openpi_flip
    uv run --with boto3 python /home/anthonyliang/Documents/oopsie_processing/exporters/export_lerobot.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import cv2
import h5py
import numpy as np

_THIS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_THIS, "..", "src"))

from oopsie_processing import stage5_harmonize as s5  # noqa: E402
from oopsie_processing.common import config, embodiments, report  # noqa: E402

CANONICAL_CAMS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
ROLE_OF_CAM = {"base_0_rgb": "base", "left_wrist_0_rgb": "left_wrist", "right_wrist_0_rgb": "right_wrist"}


def _decode_cam(mp4_path: str, num_frames: int, hw: int) -> np.ndarray:
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while len(frames) < num_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (hw, hw), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {mp4_path}")
    return np.stack(frames, axis=0)


def _meta_prompt(row: dict) -> str:
    opt = "success" if float(row["success"]) >= 0.5 else "failure"
    op = (row.get("operator_name") or "unknown").split()[0] if row.get("operator_name") else "unknown"
    return (f"<opt={opt}> <speed={int(row['speed_bucket'])}> "
            f"<emb={row['embodiment_name']}> <op={op}> ")


def _load_manifest(cfg) -> list[dict]:
    name = config.require(cfg, "manifest_name")
    with open(os.path.join(report.durable_dir(cfg), f"{name}.jsonl")) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def create_ds(cfg: dict, reg):
    """Create a fresh (empty) consolidated LeRobotDataset for the whole subset.

    Returns (ds, out_dir, hw). The streamer calls this once, then appends one lab
    at a time with add_rows() so only one lab's raw mp4s ever sit on local disk."""
    repo_id = config.require(cfg, "lerobot_repo_id")
    lerobot_home = config.require(cfg, "lerobot_home")
    hw = int(config.require(cfg, "image_hw"))
    fps = int(config.require(cfg, "fps"))
    # Set BEFORE importing lerobot: HF_LEROBOT_HOME is bound into a module-level
    # constant at import time, so setting it afterward is silently ignored (the
    # dataset would be written under ~/.cache instead of our scratch dir).
    os.environ["HF_LEROBOT_HOME"] = lerobot_home

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    out_dir = os.path.join(lerobot_home, repo_id)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    features = {
        "observation.state": {"dtype": "float32", "shape": (reg.common_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (reg.common_dim,), "names": None},
        "action_mask": {"dtype": "float32", "shape": (reg.common_dim,), "names": None},
        "meta_prompt": {"dtype": "string", "shape": (1,), "names": None},
        "success": {"dtype": "float32", "shape": (1,), "names": None},
        "speed_bucket": {"dtype": "int64", "shape": (1,), "names": None},
        "embodiment_id": {"dtype": "int64", "shape": (1,), "names": None},
    }
    for cam in CANONICAL_CAMS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (3, hw, hw), "names": ["channels", "height", "width"]}

    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, root=out_dir, robot_type="oopsie_multi", features=features,
        use_videos=True, tolerance_s=1.0, image_writer_processes=0, image_writer_threads=4)
    return ds, out_dir, hw


def add_rows(ds, cfg: dict, reg, rows: list[dict], hw: int) -> dict:
    """Append an explicit row list (e.g. one lab) to an existing dataset.

    Reads each episode's STAGED canonical h5 + mp4s under raw_stage_dir. Returns
    {n_ok, per_lab, errors}. Harmonization is re-derived from the staged h5 via
    stage5 (single source of truth for the common-18 layout + mask)."""
    raw_root = config.require(cfg, "raw_stage_dir")
    n_ok, per_lab, errors = 0, {}, []
    for row in rows:
        ep_dir = os.path.join(raw_root, row["lab_id"], row["episode_id"])
        h5_path = os.path.join(ep_dir, "episode.h5")
        try:
            emb = embodiments.require_embodiment(reg, row["lab_id"])
            with h5py.File(h5_path, "r") as f:
                action, mask, sig = s5.harmonize_action(f, emb, reg.common_dim)
                t = action.shape[0]
                state = s5.harmonize_state(f, emb, t, reg.common_dim)
            t = min(t, state.shape[0])
            # staged canonical mp4s: right_wrist may be absent (dup base).
            cam_file = {"base": "base.mp4", "left_wrist": "left_wrist.mp4", "right_wrist": "right_wrist.mp4"}
            decoded = {}
            for role, fname in cam_file.items():
                p = os.path.join(ep_dir, fname)
                if os.path.exists(p):
                    decoded[role] = _decode_cam(p, t, hw)
            if "right_wrist" not in decoded:
                decoded["right_wrist"] = decoded["base"]
            if "left_wrist" not in decoded:
                decoded["left_wrist"] = decoded["base"]
            n = min([t] + [v.shape[0] for v in decoded.values()])
            mp = _meta_prompt(row)
            for i in range(n):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "action_mask": mask,
                    "meta_prompt": mp,
                    "success": np.array([float(row["success"])], dtype=np.float32),
                    "speed_bucket": np.array([int(row["speed_bucket"])], dtype=np.int64),
                    "embodiment_id": np.array([int(row["embodiment_id"])], dtype=np.int64),
                    "task": row["language_instruction"],
                }
                for cam in CANONICAL_CAMS:
                    frame[f"observation.images.{cam}"] = decoded[ROLE_OF_CAM[cam]][i]
                ds.add_frame(frame)
            ds.save_episode()
            n_ok += 1
            per_lab[row["lab_id"]] = per_lab.get(row["lab_id"], 0) + 1
        except Exception as e:
            errors.append({"episode_id": row["episode_id"], "error": f"{type(e).__name__}: {e}"})
    return {"n_ok": n_ok, "per_lab": per_lab, "errors": errors}


def run(stage_cfg_name: str = "subset") -> dict:
    cfg = config.load_merged(stage_cfg_name)
    reg = embodiments.load_registry()
    repo_id = config.require(cfg, "lerobot_repo_id")

    rows = _load_manifest(cfg)
    ds, out_dir, hw = create_ds(cfg, reg)
    res = add_rows(ds, cfg, reg, rows, hw)

    summary = {"stage": "export_lerobot", "repo_id": repo_id, "out_dir": out_dir,
               "n_episodes": res["n_ok"], "n_errors": len(res["errors"]),
               "per_lab": res["per_lab"], "total_episodes": int(ds.meta.total_episodes),
               "errors": res["errors"][:20]}
    report.write_json(cfg, "export_lerobot_summary.json", summary)
    print(f"[export_lerobot] episodes={res['n_ok']} errors={len(res['errors'])} "
          f"per_lab={res['per_lab']} -> {out_dir}")
    if res["errors"]:
        print(f"[export_lerobot] FIRST ERRORS: {res['errors'][:5]}")
    return summary


if __name__ == "__main__":
    run()
