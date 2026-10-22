"""Export the cleaned subset to WebDataset shards via vla_foundry's own preprocessing.

Rather than re-implement vla_foundry's shard/stats/manifest writer (and risk drift in
the percentile_1/99 per-timestep stats the Qwen trainer needs), this DRIVES
vla_foundry's preprocess_robotics_to_tar entrypoint over the local canonical staging
tree (exporters/materialize.py). It:

  1. aliases the staged Enatic_OpenArm dir to Enatic_OpenArm_clean so it maps to the
     canonical-camera LabSpec (added UNCOMMITTED to vla_foundry converters/oopsie.py),
     leaving the native-name Enatic smoke spec untouched;
  2. writes a preprocessing preset (oopsie type) pointing source_episodes at the five
     staged lab dirs, with the same metadata-conditioning params as the proven
     config_presets/preprocessing/oopsie_metacond_smoke.yaml;
  3. invokes vla_foundry's .venv python on preprocess_robotics_to_tar --type oopsie.

Outputs: <wds_out_dir>/shards/{shard_*.tar, manifest.jsonl, stats.json} +
<wds_out_dir>/episodes/.  Run from the oopsie_processing repo under any env with pyyaml:

    cd /home/anthonyliang/Documents/dsrl_openpi_flip
    uv run --with boto3 python /home/anthonyliang/Documents/oopsie_processing/exporters/export_webdataset.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oopsie_processing.common import config, report  # noqa: E402

VLA_FOUNDRY = "/home/anthonyliang/Documents/vla_foundry"
# lab dir name -> LabSpec key in vla_foundry converters/oopsie.py (Enatic aliased).
LAB_TO_SPEC = {
    "Charles_UFAL": "Charles_UFAL",
    "BONN_MB": "BONN_MB",
    "Araya": "Araya",
    "Enatic_OpenArm": "Enatic_OpenArm_clean",
    "CMU_Aire": "CMU_Aire",
}


def _preset(cfg, source_dirs, out_dir) -> dict:
    """Mirror oopsie_metacond_smoke.yaml, repointed at the staged subset."""
    img_idx = list(config.require(cfg, "wds_image_indices"))
    return {
        "type": "oopsie",
        "source_episodes": source_dirs,
        "output_dir": out_dir,
        # (per-lab runs point source_episodes at ONE staged lab dir -> one dataset)
        # Fixed-path copy only runs for s3->s3; a local value just avoids a None crash.
        "output_dir_fixed_path": os.path.join(out_dir, "_fixed"),
        "common_dim": 18,
        "episodes_per_lab": -1,  # take ALL staged (clean) episodes
        "fps_default": float(config.require(cfg, "fps")),
        "camera_names": ["base_camera", "left_wrist_camera", "right_wrist_camera"],
        "past_lowdim_steps": 0,
        "future_lowdim_steps": int(config.require(cfg, "wds_lowdim_future")),
        "image_indices": img_idx,
        "stride": 1,
        "max_padding_left": 1,
        "max_padding_right": int(config.require(cfg, "wds_lowdim_future")),
        "padding_strategy": "copy",
        "filter_still_samples": False,
        "samples_per_shard": int(config.require(cfg, "wds_samples_per_shard")),
        "num_workers": int(config.require(cfg, "n_workers")),
        "resize_images_size": [int(config.require(cfg, "image_hw"))] * 2,
        "image_resizing_method": "resize_no_crop",
        "jpeg_quality": int(config.require(cfg, "wds_jpeg_quality")),
        "compute_statistics": True,
        "auto_tag": False,
        "skip_git_tagging": True,
        "db_logging": False,
        "fail_on_nan": True,
        "ray_num_cpus": int(config.require(cfg, "n_workers")),
    }


def _staged_lab_dir(raw_root: str, lab: str, spec_key: str) -> str:
    """Resolve the staged source dir for one lab. Enatic is exposed under a
    canonical-camera alias dir so the converter picks the _clean LabSpec."""
    if spec_key == "Enatic_OpenArm_clean":
        alias = os.path.join(raw_root, "Enatic_OpenArm_clean")
        target = os.path.join(raw_root, "Enatic_OpenArm")
        if os.path.islink(alias):
            os.unlink(alias)
        if not os.path.isdir(target):
            raise FileNotFoundError(f"staged lab dir missing: {target}")
        os.symlink(target, alias)
        return alias
    d = os.path.join(raw_root, lab)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"staged lab dir missing: {d}")
    return d


def _drive_preprocess(cfg, source_dirs, out_dir) -> str:
    """Write a preset and drive vla_foundry preprocessing. Returns shards dir.

    Hands the subprocess a CLEAN env pinned to vla_foundry/.venv: if we inherit a
    uv-managed VIRTUAL_ENV (e.g. launched via `uv run`), ray re-execs its workers
    in a uv ephemeral env that lacks h5py. Pinning VIRTUAL_ENV + PATH to
    vla_foundry/.venv and dropping UV_*/PYTHONPATH makes ray workers use it."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    preset = _preset(cfg, source_dirs, out_dir)
    preset_path = os.path.join(out_dir, "oopsie_subset_preprocess.yaml")
    with open(preset_path, "w") as fh:
        yaml.safe_dump(preset, fh, sort_keys=False)

    venv = os.path.join(VLA_FOUNDRY, ".venv")
    py = os.path.join(venv, "bin", "python")
    # Build a CLEAN whitelist env, NOT a copy of os.environ. If this module is run
    # under `uv run`, the inherited UV_*/VIRTUAL_ENV make ray/uv re-exec the
    # preprocessing workers in a uv ephemeral .venv that lacks h5py. Pin VIRTUAL_ENV
    # + PATH to vla_foundry/.venv and carry over only AWS/locale/system vars.
    keep = ("HOME", "USER", "LOGNAME", "TERM", "LANG", "LC_ALL", "TMPDIR",
            "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
    env = {k: v for k, v in os.environ.items()
           if k in keep or k.startswith("AWS_")}
    env["VIRTUAL_ENV"] = venv
    env["PATH"] = os.path.join(venv, "bin") + ":/usr/bin:/bin"
    # Ray (>=2.40) auto-detects a `uv run` ANCESTOR in the process tree and forces its
    # workers to relaunch via `uv run`, which (a) isn't on this scrubbed PATH and (b)
    # would build an ephemeral venv lacking h5py. Disabling the uv-run runtime env makes
    # ray launch workers with the driver's own interpreter (vla_foundry/.venv, has h5py).
    env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    cmd = [py, "-m", "vla_foundry.data.preprocessing.preprocess_robotics_to_tar",
           "--type", "oopsie", "--config_path", preset_path]
    print(f"[export_wds] running: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=VLA_FOUNDRY, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"vla_foundry preprocessing failed (rc={r.returncode})")
    return os.path.join(out_dir, "shards")


def export_lab(cfg: dict, lab: str) -> dict:
    """Export ONE staged lab to its own WDS dataset: <wds_out_dir>/<lab>/shards.

    Per-lab datasets are the disk-safe unit: the streamer builds one, uploads it
    to s3://.../wds/<lab>/, then deletes it before the next lab. Each lab's shards
    + manifest.jsonl + stats.json self-contained (vla_foundry's dataset_manifest
    is a LIST, so each lab is one weighted dataset at train time)."""
    raw_root = config.require(cfg, "raw_stage_dir")
    out_root = config.require(cfg, "wds_out_dir")
    if lab not in LAB_TO_SPEC:
        raise KeyError(f"lab '{lab}' not in LAB_TO_SPEC (known: {list(LAB_TO_SPEC)})")
    src = _staged_lab_dir(raw_root, lab, LAB_TO_SPEC[lab])
    lab_out = os.path.join(out_root, lab)
    shards_dir = _drive_preprocess(cfg, [src], lab_out)
    shards = [f for f in os.listdir(shards_dir) if f.endswith(".tar")] if os.path.isdir(shards_dir) else []
    return {"lab": lab, "out_dir": lab_out, "shards_dir": shards_dir,
            "n_shard_tars": len(shards),
            "manifest": os.path.join(shards_dir, "manifest.jsonl"),
            "stats": os.path.join(shards_dir, "stats.json")}


def run(stage_cfg_name: str = "subset") -> dict:
    """Standalone whole-subset export (NOT disk-safe for large caps: materializes
    every lab's shards locally at once). The disk-safe path is stream_to_s3.py,
    which calls export_lab() per lab and uploads+deletes between labs."""
    cfg = config.load_merged(stage_cfg_name)
    labs = list(config.require(cfg, "labs").keys())
    per_lab = {}
    for lab in labs:
        per_lab[lab] = export_lab(cfg, lab)
    summary = {"stage": "export_webdataset", "out_dir": config.require(cfg, "wds_out_dir"),
               "per_lab": per_lab,
               "n_shard_tars_total": sum(d["n_shard_tars"] for d in per_lab.values())}
    report.write_json(cfg, "export_webdataset_summary.json", summary)
    print(f"[export_wds] per-lab shards={ {k: v['n_shard_tars'] for k, v in per_lab.items()} }")
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default=None,
                    help="export only this lab to <wds_out_dir>/<lab>/shards; "
                         "default exports the whole subset (NOT disk-safe at scale)")
    ap.add_argument("--stage-cfg", default="subset")
    args = ap.parse_args()
    if args.lab:
        cfg = config.load_merged(args.stage_cfg)
        res = export_lab(cfg, args.lab)
        print("[export_wds] " + __import__("json").dumps(res))
    else:
        run(args.stage_cfg)
