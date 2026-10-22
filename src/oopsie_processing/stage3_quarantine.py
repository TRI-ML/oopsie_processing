"""Stage 3 — cruft & defect quarantine.

Quarantine = an exclude-from-clean list; nothing is deleted from raw. Two
sources: (1) cruft enumerated directly from the object listing
(`*.hidden` shadow copies, `*.h5.bak`, `.DS_Store`, `*.zip`), and (2)
defects found by bounded probing (undersized/zero-frame mp4 containers,
truncated/empty h5). Emits reports/quarantine.jsonl (path + reason).
"""
from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor

import h5py

from .common import config, h5io, listing, report, s3io, sampling


def run(stage_cfg_name: str = "stage3") -> dict:
    cfg = config.load_merged(stage_cfg_name)
    bucket = config.require(cfg, "bucket")
    region = config.require(cfg, "region")
    prefix = config.require(cfg, "dataset_prefix")
    listing_path = config.require(cfg, "listing_file")
    scratch = config.require(cfg, "scratch_dir")
    n_workers = config.require(cfg, "n_workers")
    mp4_probe = config.require(cfg, "mp4_probe_per_lab")
    mp4_min = config.require(cfg, "mp4_min_valid_bytes")
    h5_probe = config.require(cfg, "h5_truncation_probe_per_lab")
    expires = config.require(cfg, "presign_expires_s")
    ff_timeout = config.require(cfg, "ffprobe_timeout_s")

    s3 = s3io.make_client(region, config.require(cfg, "max_pool_connections"))

    rows = []  # quarantine jsonl rows: {path, reason, lab, detail}
    cruft_counts = collections.Counter()
    by_lab_cruft = collections.defaultdict(lambda: collections.Counter())
    undersize_mp4 = []  # (key, size, lab)

    # ---- (1) cruft + undersized mp4 straight from the listing ----
    for size, key in listing.iter_rows(listing_path, prefix):
        if "/.cache/" in key:
            continue
        lab = listing.prefix_of(key, prefix)
        base = key.rsplit("/", 1)[-1]
        reason = None
        if key.endswith(".hidden"):
            reason = "cruft:hidden_shadow_copy"
        elif key.endswith(".h5.bak") or key.endswith(".hdf5.bak"):
            reason = "cruft:h5_bak"
        elif base == ".DS_Store":
            reason = "cruft:ds_store"
        elif key.endswith(".zip"):
            reason = "cruft:zip"
        if reason:
            cruft_counts[reason] += 1
            by_lab_cruft[lab][reason] += 1
            rows.append({"path": key, "reason": reason, "lab": lab, "bytes": size})
            continue
        if key.endswith(".mp4") and size < mp4_min:
            undersize_mp4.append((key, size, lab))
            rows.append({"path": key, "reason": "defect:mp4_undersize_container",
                         "lab": lab, "bytes": size})

    # ---- (2) bounded defect probing per real lab ----
    stats = listing.scan_prefixes(listing_path, prefix)
    real_labs = sorted(n for n, st in stats.items()
                       if listing.classify(n, st)[0] == "real_lab")
    h5_by_lab = listing.h5_keys_by_lab(listing_path, prefix, set(real_labs))
    mp4_by_lab_keys = collections.defaultdict(list)
    for _s, key in listing.iter_rows(listing_path, prefix):
        if "/.cache/" in key or not key.endswith(".mp4"):
            continue
        lab = listing.prefix_of(key, prefix)
        if lab in h5_by_lab:  # real lab
            mp4_by_lab_keys[lab].append(key)
    for v in mp4_by_lab_keys.values():
        v.sort()

    def probe_mp4(key):
        r = s3io.ffprobe_video(s3, bucket, key, expires, ff_timeout)
        if "error" in r:
            return key, "defect:mp4_unreadable", r
        nb = r.get("nb_frames") or r.get("nb_frames_derived")
        if not nb or nb == 0:
            return key, "defect:mp4_zero_frame", r
        return key, None, r

    def probe_h5(key):
        p = s3io.download_to_temp(s3, bucket, key, scratch)
        try:
            with h5py.File(p, "r") as f:
                T, src = h5io.get_T(f)
                rs = h5io.robot_states(f)
                ac = h5io.actions(f)
            if T is None or T == 0:
                return key, "defect:h5_no_trajectory", {"T": T}
            if not rs and not ac:
                return key, "defect:h5_empty_groups", {"T": T}
            return key, None, {"T": T, "src": src}
        except Exception as e:
            return key, "defect:h5_unreadable", {"err": type(e).__name__}
        finally:
            s3io.rm(p)

    mp4_probed = mp4_defects = 0
    mp4_defect_by_lab = collections.Counter()
    mp4_targets = []
    for lab in real_labs:
        mp4_targets += sampling.evenly_spaced(mp4_by_lab_keys.get(lab, []), mp4_probe) \
            if mp4_by_lab_keys.get(lab) else []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for key, reason, detail in ex.map(probe_mp4, mp4_targets):
            mp4_probed += 1
            if reason:
                mp4_defects += 1
                lab = listing.prefix_of(key, prefix)
                mp4_defect_by_lab[lab] += 1
                rows.append({"path": key, "reason": reason, "lab": lab, "detail": detail})

    h5_probed = h5_defects = 0
    h5_defect_by_lab = collections.Counter()
    h5_targets = []
    for lab in real_labs:
        h5_targets += sampling.evenly_spaced(h5_by_lab.get(lab, []), h5_probe) \
            if h5_by_lab.get(lab) else []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for key, reason, detail in ex.map(probe_h5, h5_targets):
            h5_probed += 1
            if reason:
                h5_defects += 1
                lab = listing.prefix_of(key, prefix)
                h5_defect_by_lab[lab] += 1
                rows.append({"path": key, "reason": reason, "lab": lab, "detail": detail})

    report.write_jsonl(cfg, "quarantine.jsonl", rows)

    cmu = [k for k, s, l in undersize_mp4 if l == "CMU_Auton"]
    summary = {
        "stage": "3_quarantine",
        "total_quarantined_paths": len(rows),
        "cruft_counts": dict(cruft_counts),
        "cruft_by_lab": {l: dict(c) for l, c in sorted(by_lab_cruft.items())},
        "undersize_mp4_total": len(undersize_mp4),
        "undersize_mp4_by_lab": dict(collections.Counter(l for _k, _s, l in undersize_mp4)),
        "cmu_auton_undersize_mp4": {"count": len(cmu),
                                    "example": (cmu[0] if cmu else None),
                                    "example_bytes": (next((s for k, s, l in undersize_mp4
                                                            if k == cmu[0]), None) if cmu else None)},
        "mp4_probe": {"probed": mp4_probed, "defects": mp4_defects,
                      "by_lab": dict(mp4_defect_by_lab)},
        "h5_probe": {"probed": h5_probed, "defects": h5_defects,
                     "by_lab": dict(h5_defect_by_lab)},
    }
    report.write_json(cfg, "quarantine_summary.json", summary)
    print(f"[stage3] quarantined paths={len(rows)} cruft={dict(cruft_counts)}")
    print(f"[stage3] undersize mp4={len(undersize_mp4)} "
          f"(CMU_Auton={len(cmu)}, example bytes={summary['cmu_auton_undersize_mp4']['example_bytes']})")
    print(f"[stage3] mp4 probe {mp4_probed} -> {mp4_defects} defect; "
          f"h5 probe {h5_probed} -> {h5_defects} defect")
    return summary
