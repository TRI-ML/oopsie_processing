"""Shared bounded h5 scan feeding stage-4 (schema) and data-quality.

Downloads each sampled h5 exactly once, computes schema-validation signals and
data-quality signals (no raw arrays retained), and caches the compact
per-episode records to the durable dir so the two consumer stages (and re-runs)
do not re-download. Read-only; temp files deleted immediately.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

from . import config, h5io, listing, report, s3io, sampling

CACHE_NAME = "_scan_cache.json"


def _per_dim_constant(arr: np.ndarray, rel_eps: float) -> int:
    """Count columns that are (near-)constant over time. arr is (T, D) or (T,)."""
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] < 2:
        return 0
    finite = np.isfinite(arr)
    rng = np.full(arr.shape[1], 0.0)
    for d in range(arr.shape[1]):
        col = arr[finite[:, d], d]
        if col.size < 2:
            continue
        rng[d] = float(np.max(col) - np.min(col))
    scale = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 0.0
    thresh = rel_eps * (scale if scale > 0 else 1.0)
    return int(np.sum(rng <= thresh))


def _analyze_group(f, group_path, rel_eps):
    """Return dict: n_slots, dims, has_nan, has_inf, constant_dims, min, max."""
    has_nan = has_inf = False
    total_dims = 0
    const_dims = 0
    gmin = None
    gmax = None
    slots = {}
    for name, arr in h5io.read_arrays(f, group_path):
        arr = np.asarray(arr)
        if not np.issubdtype(arr.dtype, np.number):
            continue
        d = 1 if arr.ndim == 1 else int(np.prod(arr.shape[1:]))
        total_dims += d
        slots[name] = list(arr.shape)
        if np.isnan(arr).any():
            has_nan = True
        if np.isinf(arr).any():
            has_inf = True
        const_dims += _per_dim_constant(arr.reshape(arr.shape[0], -1), rel_eps)
        fin = arr[np.isfinite(arr)]
        if fin.size:
            lo, hi = float(fin.min()), float(fin.max())
            gmin = lo if gmin is None else min(gmin, lo)
            gmax = hi if gmax is None else max(gmax, hi)
    return {"n_slots": len(slots), "slots": slots, "dims": total_dims,
            "has_nan": has_nan, "has_inf": has_inf,
            "constant_dims": const_dims, "min": gmin, "max": gmax}


def _scan_one(s3, bucket, key, lab, scratch, mp4_basenames, rel_eps):
    p = s3io.download_to_temp(s3, bucket, key, scratch)
    rec = {"key": key, "lab": lab}
    try:
        with h5py.File(p, "r") as f:
            # ---- schema signals ----
            present = {a: (h5io.attr(f, a) is not None) for a in h5io.ROOT_ATTRS_REQUIRED}
            rp, rp_err = h5io.parse_robot_profile(f)
            vpaths = h5io.video_paths(f)
            resolved = unresolved = 0
            for cam, ptr in vpaths.items():
                if not ptr:
                    unresolved += 1
                    continue
                base = str(ptr).rsplit("/", 1)[-1]
                if base in mp4_basenames:
                    resolved += 1
                else:
                    unresolved += 1
            rs = h5io.robot_states(f)
            ac = h5io.actions(f)
            T, t_src = h5io.get_T(f)
            cf, cf_src = h5io.control_freq(f)
            instr = h5io.attr(f, "language_instruction")

            violations = []
            for a, ok in present.items():
                if not ok:
                    violations.append(f"missing_attr:{a}")
            if rp_err:
                violations.append(f"robot_profile:{rp_err}")
            if not rs:
                violations.append("robot_states_empty")
            if vpaths and unresolved:
                violations.append(f"video_ptr_unresolved:{unresolved}")
            if not vpaths:
                violations.append("no_video_paths")

            rec["schema"] = {
                "schema_attr": h5io.attr(f, "schema"),
                "attrs_present": present,
                "robot_profile_ok": rp_err is None,
                "robot_profile_error": rp_err,
                "n_state_keys": len(rs),
                "n_action_slots": len(ac),
                "n_video_ptr": len(vpaths),
                "video_resolved": resolved,
                "video_unresolved": unresolved,
                "violations": violations,
                "valid": len(violations) == 0,
            }

            # ---- data-quality signals ----
            act = _analyze_group(f, "actions", rel_eps)
            sta = _analyze_group(f, "observations/robot_states", rel_eps)
            robot_name = rp.get("robot_name") if rp else None
            is_biarm = rp.get("is_biarm") if rp else None
            dur = (T / cf) if (T and cf) else None
            degenerate = (act["dims"] > 0 and act["constant_dims"] == act["dims"])
            rec["dq"] = {
                "T": T, "control_freq": cf, "duration_s": dur,
                "robot_name": robot_name, "is_biarm": is_biarm,
                "instruction_present": bool(instr),
                "action": act, "state": sta,
                "degenerate_action": degenerate,
                "has_nan": act["has_nan"] or sta["has_nan"],
                "has_inf": act["has_inf"] or sta["has_inf"],
            }
        return rec
    except Exception as e:
        rec["error"] = type(e).__name__ + ":" + str(e)[:120]
        return rec
    finally:
        s3io.rm(p)


def scan_real_labs(cfg: dict, sample_per_lab: int, rel_eps: float,
                   use_cache: bool = True) -> list[dict]:
    durable = report.durable_dir(cfg)
    cache_path = os.path.join(durable, CACHE_NAME)
    if use_cache and os.path.exists(cache_path):
        meta = json.load(open(cache_path))
        if meta.get("sample_per_lab") == sample_per_lab and meta.get("rel_eps") == rel_eps:
            print(f"[scan] reusing cache {cache_path} "
                  f"({len(meta['records'])} records)")
            return meta["records"]

    bucket = config.require(cfg, "bucket")
    region = config.require(cfg, "region")
    prefix = config.require(cfg, "dataset_prefix")
    listing_path = config.require(cfg, "listing_file")
    scratch = config.require(cfg, "scratch_dir")
    n_workers = config.require(cfg, "n_workers")

    stats = listing.scan_prefixes(listing_path, prefix)
    real_labs = sorted(n for n, st in stats.items()
                       if listing.classify(n, st)[0] == "real_lab")
    h5_by_lab = listing.h5_keys_by_lab(listing_path, prefix, set(real_labs))
    mp4_by_lab = listing.mp4_basenames_by_lab(listing_path, prefix, set(real_labs))

    s3 = s3io.make_client(region, config.require(cfg, "max_pool_connections"))

    records: list[dict] = []
    for i, lab in enumerate(real_labs, 1):
        keys = h5_by_lab.get(lab, [])
        if not keys:
            continue
        sample = sampling.evenly_spaced(keys, sample_per_lab)
        mp4_names = set(mp4_by_lab.get(lab, {}).keys())
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            lab_recs = list(ex.map(
                lambda k: _scan_one(s3, bucket, k, lab, scratch, mp4_names, rel_eps),
                sample))
        records += lab_recs
        n_err = sum(1 for r in lab_recs if "error" in r)
        print(f"[scan] [{i:2d}/{len(real_labs)}] {lab:32s} "
              f"sampled={len(sample):4d} err={n_err}", flush=True)

    json.dump({"sample_per_lab": sample_per_lab, "rel_eps": rel_eps,
               "records": records}, open(cache_path, "w"))
    print(f"[scan] wrote cache {cache_path} ({len(records)} records)")
    return records
