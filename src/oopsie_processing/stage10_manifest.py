"""Stage 10 — metadata manifest for the cleaned SUBSET.

One row per clean episode. Selects CLEAN episodes for the configured subset labs
(quarantine paths skipped, per-episode alignment mismatches skipped, optional
path-prefix restriction, optional success/failure balancing), downloads each small
h5, and applies the stage-5/6/8/9 harmonization to emit the manifest axes needed
for metadata/suboptimality-conditioned pretraining.

Outputs manifest.parquet + manifest.jsonl (one row per episode). The exporters read
this manifest to materialize the LeRobot and WebDataset training formats.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

from . import stage5_harmonize as s5
from . import stage6_cameras as s6
from . import stage8_optimality as s8
from . import stage9_instructions as s9
from .common import config, embodiments, h5io, listing, report, s3io, sampling


def _load_quarantine(path: str) -> set[str]:
    out = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.add(json.loads(line)["path"])
    return out


def _load_alignment_skips(path: str) -> set[str]:
    """h5 keys flagged as temporally misaligned in the Phase-A alignment report."""
    out = set()
    data = json.load(open(path))
    for _lab, d in data.get("per_lab", {}).items():
        for ex in d.get("examples", []):
            if ex.get("h5"):
                out.add(ex["h5"])
    return out


def _select_keys(lab, keys, cap, path_filter, outcome_balanced, dataset_prefix):
    """Deterministic, bounded selection for one lab."""
    if path_filter:
        pref = dataset_prefix + path_filter[len(dataset_prefix):] if path_filter.startswith(dataset_prefix) else dataset_prefix + path_filter
        keys = [k for k in keys if k.startswith(pref)]
    if lab in outcome_balanced:
        succ = [k for k in keys if "/success/" in k]
        fail = [k for k in keys if "/failure/" in k]
        half = max(1, cap // 2)
        return sampling.evenly_spaced(succ, half) + sampling.evenly_spaced(fail, cap - half) \
            if (succ and fail) else sampling.evenly_spaced(keys, cap)
    return sampling.evenly_spaced(keys, cap)


def _build_row(s3, bucket, scratch, key, lab, emb, reg, folder_fallback):
    p = s3io.download_to_temp(s3, bucket, key, scratch)
    try:
        with h5py.File(p, "r") as f:
            action, mask, sig = s5.harmonize_action(f, emb, reg.common_dim)
            t = int(action.shape[0])
            if t < 2:
                return {"key": key, "lab": lab, "skip": "lt2_frames"}
            cf, _ = h5io.control_freq(f)
            vpaths = h5io.video_paths(f)
            roles = s6.resolve_roles(vpaths, emb, reg)
            opt = s8.consolidate(f, key, lab, folder_fallback)
            display, norm_key = s9.normalize(h5io.attr(f, "language_instruction"))
            operator = h5io.attr(f, "operator_name")
            rp, _ = h5io.parse_robot_profile(f)
        valid = sig["valid"]
        dact = np.abs(np.diff(action[:, :valid], axis=0))
        speed = float(dact.mean() * (cf or 1.0))
        h5_dir = key.rsplit("/", 1)[0]
        cam_keys = {role: f"{h5_dir}/{vpaths[native]}" for role, native in roles.items()}
        return {
            "key": key,
            "lab_id": lab,
            "embodiment_id": emb.embodiment_id,
            "embodiment_name": emb.embodiment_name,
            "robot_name": (rp.get("robot_name") if rp else None) or emb.robot_name,
            "is_biarm": bool(rp.get("is_biarm")) if (rp and rp.get("is_biarm") is not None) else emb.is_biarm,
            "gripper_name": (rp.get("gripper_name") if rp else None) or emb.gripper_name,
            "operator_name": operator,
            "success": opt["success"],
            "taxonomy_outcome": opt["taxonomy_outcome"],
            "failure_category": opt["failure_category"],
            "severity": opt["severity"],
            "annotator": opt["annotator"],
            "label_source": opt["label_source"],
            "speed": speed,
            "language_instruction": display,
            "instruction_norm_key": norm_key,
            "camera_roles": roles,
            "camera_s3_keys": cam_keys,
            "native_action_signature": sig,
            "T": t,
            "fps": cf,
            "h5_key": key,
        }
    except Exception as e:
        return {"key": key, "lab": lab, "error": type(e).__name__ + ":" + str(e)[:160]}
    finally:
        s3io.rm(p)


def run(stage_cfg_name: str = "subset") -> dict:
    cfg = config.load_merged(stage_cfg_name)
    reg = embodiments.load_registry()
    bucket = config.require(cfg, "bucket")
    region = config.require(cfg, "region")
    prefix = config.require(cfg, "dataset_prefix")
    listing_path = config.require(cfg, "listing_file")
    scratch = config.require(cfg, "scratch_dir")
    n_workers = config.require(cfg, "n_workers")
    labs_cfg = config.require(cfg, "labs")
    path_filter = cfg.get("lab_path_filter", {})
    outcome_balanced = set(config.require(cfg, "outcome_balanced_labs"))
    quarantine = _load_quarantine(config.require(cfg, "quarantine_file"))
    align_skips = _load_alignment_skips(config.require(cfg, "alignment_file"))
    folder_fallback = set(config.require(config.load_merged("stage8"), "folder_fallback_labs"))

    h5_by_lab = listing.h5_keys_by_lab(listing_path, prefix, set(labs_cfg))
    s3 = s3io.make_client(region, config.require(cfg, "max_pool_connections"))

    rows = []
    errors = []
    skipped = {"quarantine": 0, "alignment": 0, "lt2_frames": 0}
    for lab, cap in labs_cfg.items():
        emb = embodiments.require_embodiment(reg, lab)
        keys = [k for k in h5_by_lab.get(lab, [])
                if k not in quarantine and k not in align_skips]
        n_q = sum(1 for k in h5_by_lab.get(lab, []) if k in quarantine)
        n_a = sum(1 for k in h5_by_lab.get(lab, []) if k in align_skips)
        skipped["quarantine"] += n_q
        skipped["alignment"] += n_a
        sel = _select_keys(lab, keys, cap, path_filter.get(lab), outcome_balanced, prefix)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            lab_rows = list(ex.map(
                lambda k: _build_row(s3, bucket, scratch, k, lab, emb, reg, folder_fallback), sel))
        ok = 0
        for r in lab_rows:
            if r.get("error"):
                errors.append(r)
            elif r.get("skip") == "lt2_frames":
                skipped["lt2_frames"] += 1
            else:
                rows.append(r)
                ok += 1
        print(f"[stage10] {lab:16s} selected={len(sel):4d} ok={ok:4d} "
              f"quarantined={n_q} align_skip={n_a}", flush=True)

    # speed_bucket tertiles over ALL selected clean episodes (matches converters).
    speeds = np.asarray([r["speed"] for r in rows]) if rows else np.zeros(0)
    if speeds.size:
        lo, hi = np.quantile(speeds, [1 / 3, 2 / 3])
    else:
        lo = hi = 0.0
    for r in rows:
        s = r["speed"]
        r["speed_bucket"] = 0 if s <= lo else (1 if s <= hi else 2)
        r.pop("key", None)
        r.pop("episode_id", None)

    # set episode_id from the h5 basename (stable, matches converters' episode_id)
    for r in rows:
        stem = r["h5_key"].rsplit("/", 1)[-1]
        r["episode_id"] = f"{r['lab_id']}_{stem[:-3] if stem.endswith('.h5') else stem}"

    report.write_jsonl(cfg, f"{config.require(cfg, 'manifest_name')}.jsonl", rows)
    try:
        _write_parquet(cfg, rows)
    except Exception as e:  # parquet needs pandas+pyarrow; jsonl is the source of truth
        print(f"[stage10] WARNING: parquet not written ({type(e).__name__}: {e}); "
              f"jsonl is complete. Run `manifest_parquet` in an env with pandas+pyarrow.")

    summary = {
        "stage": "10_manifest",
        "n_episodes": len(rows),
        "n_errors": len(errors),
        "skipped": skipped,
        "speed_tertiles": {"lo": float(lo), "hi": float(hi)},
        "per_lab_counts": {lab: sum(1 for r in rows if r["lab_id"] == lab) for lab in labs_cfg},
        "per_embodiment_counts": _counts(rows, "embodiment_name"),
        "success_counts": _counts(rows, "success"),
        "label_source_counts": _counts(rows, "label_source"),
        "errors": errors[:20],
    }
    report.write_json(cfg, "manifest_summary.json", summary)
    print(f"[stage10] manifest rows={len(rows)} errors={len(errors)} "
          f"per_lab={summary['per_lab_counts']}")
    print(f"[stage10] success={summary['success_counts']} label_source={summary['label_source_counts']}")
    return summary


def _counts(rows, field):
    out = {}
    for r in rows:
        out[str(r.get(field))] = out.get(str(r.get(field)), 0) + 1
    return out


def _write_parquet(cfg, rows):
    import pandas as pd
    rd = report.reports_dir(cfg)
    name = config.require(cfg, "manifest_name") + ".parquet"
    path = os.path.join(rd, name)
    df = pd.DataFrame([_flatten(r) for r in rows])
    df.to_parquet(path, index=False)
    import shutil
    shutil.copy2(path, os.path.join(report.durable_dir(cfg), name))
    return path


def _flatten(r):
    """JSON-encode nested dict columns so parquet stays columnar + portable."""
    out = dict(r)
    for k in ("camera_roles", "camera_s3_keys", "native_action_signature", "failure_category"):
        if k in out and not isinstance(out[k], (str, type(None))):
            out[k] = json.dumps(out[k])
    return out
