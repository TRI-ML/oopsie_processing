"""Disk-safe streaming materializer: manifest -> both formats -> S3, one lab at a time.

The prior Phase-B run filled puget's root disk by staging all labs and accumulating
~98 GB of per-frame-JPEG WebDataset tars locally before any upload. This orchestrator
makes the materialization STREAMING and bounded:

    create ONE consolidated LeRobotDataset (compact re-encoded video; grows slowly)
    for each lab:
        1. materialize ONLY this lab's clean episodes  -> phaseB2/raw/<lab>
        2. append this lab to the LeRobot dataset       (reads staged files)
        3. export this lab to its own WDS dataset        -> phaseB2/wds/<lab>/shards
        4. aws s3 sync  phaseB2/wds/<lab>/  ->  s3://.../oopsie_clean/wds/<lab>/
        5. DELETE phaseB2/raw/<lab> + phaseB2/wds/<lab>   (local footprint back to ~LeRobot only)
        6. assert free disk >= df_floor_gb and phaseB2 tree <= phaseB2_cap_gb
    aws s3 sync  lerobot_home/<repo_id>/  ->  s3://.../oopsie_clean/lerobot/
    DELETE lerobot_home
    upload manifest.parquet + manifest.jsonl -> s3://.../oopsie_clean/

Per-lab WDS datasets are the natural disk-safe unit AND the natural train-time unit:
vla_foundry's dataset_manifest is a LIST, so each lab becomes one weighted dataset with
its own stats.json. Run under the openpi/lerobot env (has boto3/h5py/cv2/lerobot); the
WDS step shells out to vla_foundry's own venv internally.

    cd /home/anthonyliang/Documents/dsrl_openpi_flip
    uv run --with boto3 python /home/anthonyliang/Documents/oopsie_processing/exporters/stream_to_s3.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))  # sibling exporter modules by bare name

from oopsie_processing.common import config, embodiments, report  # noqa: E402
import materialize as _materialize  # noqa: E402
import export_lerobot as _lerobot  # noqa: E402
import export_webdataset as _wds  # noqa: E402


def _free_gb(mount: str) -> float:
    return shutil.disk_usage(mount).free / 1e9


def _dir_bytes(path: str) -> int:
    if not os.path.exists(path):
        return 0
    out = subprocess.run(["du", "-sb", path], capture_output=True, text=True)
    try:
        return int(out.stdout.split()[0])
    except (ValueError, IndexError):
        return 0


def _s3_sync(local: str, s3_uri: str) -> None:
    """Upload a local dir to S3, errors-only output (context hygiene)."""
    r = subprocess.run(
        ["aws", "s3", "sync", local, s3_uri, "--no-progress", "--only-show-errors"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"aws s3 sync {local} -> {s3_uri} failed (rc={r.returncode}): "
                           f"{r.stderr.strip()[:300]}")


def _s3_cp(local: str, s3_uri: str) -> None:
    r = subprocess.run(["aws", "s3", "cp", local, s3_uri, "--only-show-errors"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"aws s3 cp {local} -> {s3_uri} failed (rc={r.returncode}): "
                           f"{r.stderr.strip()[:300]}")


def _load_manifest(cfg) -> list[dict]:
    name = config.require(cfg, "manifest_name")
    path = os.path.join(report.durable_dir(cfg), f"{name}.jsonl")
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def run(stage_cfg_name: str = "subset") -> dict:
    cfg = config.load_merged(stage_cfg_name)
    reg = embodiments.load_registry()
    labs = list(config.require(cfg, "labs").keys())
    raw_root = config.require(cfg, "raw_stage_dir")
    wds_root = config.require(cfg, "wds_out_dir")
    lerobot_home = config.require(cfg, "lerobot_home")
    repo_id = config.require(cfg, "lerobot_repo_id")
    s3_wds = config.require(cfg, "s3_wds_prefix").rstrip("/") + "/"
    s3_lerobot = config.require(cfg, "s3_lerobot_prefix").rstrip("/") + "/"
    s3_clean = config.require(cfg, "s3_clean_prefix").rstrip("/") + "/"
    mount = config.require(cfg, "df_mount")
    floor_gb = float(config.require(cfg, "df_floor_gb"))
    cap_gb = float(config.require(cfg, "phaseB2_cap_gb"))
    phaseB2_root = config.require(cfg, "durable_dir")

    free0 = _free_gb(mount)
    print(f"[stream] PREFLIGHT free({mount})={free0:.1f} GB", flush=True)
    if free0 < 150:
        raise RuntimeError(f"ABORT: free disk {free0:.1f} GB < 150 GB preflight floor")

    rows = _load_manifest(cfg)
    by_lab = {}
    for r in rows:
        by_lab.setdefault(r["lab_id"], []).append(r)

    # Fresh WDS + LeRobot local trees; KEEP the pre-staged raw (small h5+mp4 that
    # BOTH exporters read, ~10 MB/ep so well under the phaseB2 cap even for all
    # labs at once). Stage any lab whose raw is missing. Only the per-frame-JPEG
    # WDS shards (the storage-explosive part) are cycled per lab.
    for d in (wds_root, lerobot_home):
        if os.path.isdir(d):
            shutil.rmtree(d)
    os.makedirs(raw_root, exist_ok=True)
    os.makedirs(wds_root, exist_ok=True)
    for lab in labs:
        if not os.path.isdir(os.path.join(raw_root, lab)):
            print(f"[stream] staging missing raw for {lab}", flush=True)
            sres = _materialize.stage_rows(cfg, by_lab.get(lab, []))
            if sres["errors"]:
                print(f"[stream] {lab} stage errors: {sres['errors'][:3]}", flush=True)

    ds, out_dir, hw = _lerobot.create_ds(cfg, reg)

    peak_phaseB2 = 0
    lab_stats = {}
    for lab in labs:
        lab_rows = by_lab.get(lab, [])
        lab_raw = os.path.join(raw_root, lab)
        lab_wds = os.path.join(wds_root, lab)
        try:
            if not lab_rows:
                raise RuntimeError(f"no manifest rows for lab {lab}")
            print(f"[stream] === {lab}: {len(lab_rows)} episodes ===", flush=True)
            raw_bytes = _dir_bytes(lab_raw)

            # 1. append to the consolidated LeRobot dataset (compact video)
            lres = _lerobot.add_rows(ds, cfg, reg, lab_rows, hw)
            if lres["errors"]:
                print(f"[stream] {lab} lerobot errors: {lres['errors'][:3]}", flush=True)

            # 2. export this lab's own WDS dataset (subprocess under vla_foundry venv)
            wres = _wds.export_lab(cfg, lab)
            wds_bytes = _dir_bytes(wres["shards_dir"])

            # 3. upload ONLY this lab's shards/ (shard tars + manifest.jsonl +
            #    stats.json + preprocessing_config.yaml) to S3. The frames/ and
            #    episodes/ sibling dirs are vla_foundry intermediates (~2x the shard
            #    bytes) that training never reads — neither uploaded nor kept.
            _s3_sync(wres["shards_dir"], f"{s3_wds}{lab}/shards/")

            peak_phaseB2 = max(peak_phaseB2, _dir_bytes(phaseB2_root))
            lab_stats[lab] = {
                "episodes": len(lab_rows),
                "lerobot_episodes": lres["n_ok"],
                "wds_shard_tars": wres["n_shard_tars"],
                "raw_bytes": raw_bytes,
                "wds_bytes": wds_bytes,
                "raw_mb": round(raw_bytes / 1e6, 1),
                "wds_mb": round(wds_bytes / 1e6, 1),
                "raw_mb_per_ep": round(raw_bytes / 1e6 / max(1, len(lab_rows)), 2),
                "wds_mb_per_ep": round(wds_bytes / 1e6 / max(1, lres["n_ok"]), 2),
            }
        finally:
            # ALWAYS reclaim the big per-frame-JPEG WDS shards (+ the Enatic alias
            # symlink), success or failure. KEEP raw: it is small and shared by both
            # exporters, and is deleted once in one shot after the lab loop.
            for d in (lab_wds, os.path.join(raw_root, "Enatic_OpenArm_clean")):
                if os.path.islink(d):
                    os.unlink(d)
                elif os.path.isdir(d):
                    shutil.rmtree(d)
        # disk guards
        free = _free_gb(mount)
        pbytes = _dir_bytes(phaseB2_root)
        print(f"[stream] {lab} done: raw={lab_stats.get(lab, {}).get('raw_mb')}MB "
              f"wds={lab_stats.get(lab, {}).get('wds_mb')}MB | free={free:.1f}GB "
              f"phaseB2={pbytes/1e9:.2f}GB", flush=True)
        if free < floor_gb:
            raise RuntimeError(f"ABORT after {lab}: free {free:.1f} GB < floor {floor_gb} GB")
        if pbytes / 1e9 > cap_gb:
            raise RuntimeError(f"ABORT after {lab}: phaseB2 {pbytes/1e9:.2f} GB > cap {cap_gb} GB")

    # Raw is no longer needed once every lab is in both LeRobot + WDS: reclaim it.
    if os.path.isdir(raw_root):
        shutil.rmtree(raw_root)

    # LeRobot: upload the whole consolidated dataset once, then reclaim.
    lerobot_bytes = _dir_bytes(out_dir)
    peak_phaseB2 = max(peak_phaseB2, _dir_bytes(phaseB2_root))
    _s3_sync(out_dir, s3_lerobot)
    shutil.rmtree(lerobot_home)

    # manifest to the clean root.
    name = config.require(cfg, "manifest_name")
    for ext in ("parquet", "jsonl"):
        p = os.path.join(report.durable_dir(cfg), f"{name}.{ext}")
        if os.path.exists(p):
            _s3_cp(p, f"{s3_clean}{name}.{ext}")

    free_end = _free_gb(mount)
    total_wds_mb = sum(s["wds_mb"] for s in lab_stats.values())
    summary = {
        "stage": "stream_to_s3",
        "labs": labs,
        "n_episodes": sum(s["lerobot_episodes"] for s in lab_stats.values()),
        "lerobot_total_episodes": int(ds.meta.total_episodes),
        "lerobot_bytes": lerobot_bytes,
        "lerobot_mb": round(lerobot_bytes / 1e6, 1),
        "wds_total_mb": round(total_wds_mb, 1),
        "per_lab": lab_stats,
        "peak_phaseB2_gb": round(peak_phaseB2 / 1e9, 2),
        "free_gb_preflight": round(free0, 1),
        "free_gb_end": round(free_end, 1),
        "s3_lerobot": s3_lerobot,
        "s3_wds": s3_wds,
        "s3_manifest": f"{s3_clean}{name}.parquet",
    }
    report.write_json(cfg, "stream_to_s3_summary.json", summary)
    print(f"[stream] DONE: {summary['n_episodes']} eps | lerobot={summary['lerobot_mb']}MB "
          f"wds={summary['wds_total_mb']}MB | peak phaseB2={summary['peak_phaseB2_gb']}GB "
          f"free {free0:.1f}->{free_end:.1f}GB", flush=True)
    return summary


if __name__ == "__main__":
    run()
