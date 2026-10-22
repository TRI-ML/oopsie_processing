"""Stage 7 — temporal alignment verification.

Per real lab, over a bounded episode sample: for each camera, assert mp4 frame
count == trajectory T and video fps == control_freq. Frame count and fps are
read from the mp4 container header over a presigned URL (no full download).
Mismatches are recorded (Phase B quarantines them).
"""
from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor

import h5py

from .common import config, h5io, listing, report, s3io, sampling


def run(stage_cfg_name: str = "stage7") -> dict:
    cfg = config.load_merged(stage_cfg_name)
    bucket = config.require(cfg, "bucket")
    region = config.require(cfg, "region")
    prefix = config.require(cfg, "dataset_prefix")
    listing_path = config.require(cfg, "listing_file")
    scratch = config.require(cfg, "scratch_dir")
    n_workers = config.require(cfg, "n_workers")
    per_lab_n = config.require(cfg, "sample_episodes_per_lab")
    max_cams = config.require(cfg, "max_cams_per_episode")
    fps_tol = config.require(cfg, "fps_tolerance")
    frame_tol = config.require(cfg, "frame_tolerance")
    expires = config.require(cfg, "presign_expires_s")
    ff_timeout = config.require(cfg, "ffprobe_timeout_s")

    s3 = s3io.make_client(region, config.require(cfg, "max_pool_connections"))

    stats = listing.scan_prefixes(listing_path, prefix)
    real_labs = sorted(n for n, st in stats.items()
                       if listing.classify(n, st)[0] == "real_lab")
    h5_by_lab = listing.h5_keys_by_lab(listing_path, prefix, set(real_labs))
    mp4_by_lab = listing.mp4_basenames_by_lab(listing_path, prefix, set(real_labs))

    def align_episode(args):
        lab, h5key = args
        p = s3io.download_to_temp(s3, bucket, h5key, scratch)
        res = {"lab": lab, "h5": h5key, "cams": [], "episode_ok": None}
        try:
            with h5py.File(p, "r") as f:
                T, _ = h5io.get_T(f)
                cf, _ = h5io.control_freq(f)
                vpaths = h5io.video_paths(f)
            res["T"] = T
            res["control_freq"] = cf
            names = mp4_by_lab.get(lab, {})
            cam_results = []
            for cam, ptr in list(vpaths.items())[:max_cams]:
                base = str(ptr).rsplit("/", 1)[-1] if ptr else None
                key = names.get(base) if base else None
                if key is None:
                    cam_results.append({"cam": cam, "status": "ptr_unresolved", "ptr": ptr})
                    continue
                pr = s3io.ffprobe_video(s3, bucket, key, expires, ff_timeout)
                if "error" in pr:
                    cam_results.append({"cam": cam, "status": "probe_error", "err": pr["error"]})
                    continue
                nb = pr.get("nb_frames") or pr.get("nb_frames_derived")
                fps = pr.get("fps")
                frame_ok = (nb is not None and T is not None and abs(nb - T) <= frame_tol)
                fps_ok = (fps is not None and cf is not None and cf > 0
                          and abs(fps - cf) <= fps_tol * cf)
                cam_results.append({
                    "cam": cam, "nb_frames": nb, "fps": fps,
                    "frame_ok": frame_ok, "fps_ok": fps_ok,
                    "status": "ok" if (frame_ok and fps_ok) else "mismatch",
                })
            res["cams"] = cam_results
            probed = [c for c in cam_results if c["status"] in ("ok", "mismatch")]
            if probed:
                res["episode_ok"] = all(c["status"] == "ok" for c in probed)
            return res
        except Exception as e:
            res["error"] = type(e).__name__
            return res
        finally:
            s3io.rm(p)

    targets = []
    for lab in real_labs:
        for k in sampling.evenly_spaced(h5_by_lab.get(lab, []), per_lab_n):
            targets.append((lab, k))

    per_lab = collections.defaultdict(lambda: {
        "episodes": 0, "episodes_ok": 0, "episodes_mismatch": 0,
        "cams_probed": 0, "cams_ok": 0, "frame_mismatch": 0, "fps_mismatch": 0,
        "ptr_unresolved": 0, "probe_errors": 0, "h5_errors": 0, "examples": []})
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for r in ex.map(align_episode, targets):
            d = per_lab[r["lab"]]
            if r.get("error"):
                d["h5_errors"] += 1
                continue
            d["episodes"] += 1
            if r["episode_ok"] is True:
                d["episodes_ok"] += 1
            elif r["episode_ok"] is False:
                d["episodes_mismatch"] += 1
                if len(d["examples"]) < 5:
                    d["examples"].append({"h5": r["h5"], "T": r.get("T"),
                                          "cf": r.get("control_freq"), "cams": r["cams"]})
            for c in r["cams"]:
                if c["status"] == "ptr_unresolved":
                    d["ptr_unresolved"] += 1
                elif c["status"] == "probe_error":
                    d["probe_errors"] += 1
                else:
                    d["cams_probed"] += 1
                    if c["status"] == "ok":
                        d["cams_ok"] += 1
                    else:
                        if not c["frame_ok"]:
                            d["frame_mismatch"] += 1
                        if not c["fps_ok"]:
                            d["fps_mismatch"] += 1

    per_lab = {l: dict(d) for l, d in sorted(per_lab.items())}
    tot_ep = sum(d["episodes"] for d in per_lab.values())
    tot_ok = sum(d["episodes_ok"] for d in per_lab.values())
    tot_cam = sum(d["cams_probed"] for d in per_lab.values())
    tot_cam_ok = sum(d["cams_ok"] for d in per_lab.values())
    out = {
        "stage": "7_alignment",
        "method": "mp4 container-header nb_frames/fps (presigned URL, no full download) "
                  "vs h5 trajectory T and control_freq",
        "corpus": {
            "episodes_checked": tot_ep,
            "episodes_fully_aligned": tot_ok,
            "episode_pass_rate": round(tot_ok / tot_ep, 4) if tot_ep else None,
            "cams_probed": tot_cam,
            "cams_aligned": tot_cam_ok,
            "cam_pass_rate": round(tot_cam_ok / tot_cam, 4) if tot_cam else None,
            "labs_with_mismatch": sorted(l for l, d in per_lab.items()
                                         if d["episodes_mismatch"] or d["frame_mismatch"]
                                         or d["fps_mismatch"]),
        },
        "per_lab": per_lab,
    }
    report.write_json(cfg, "alignment.json", out)
    c = out["corpus"]
    print(f"[stage7] episodes {c['episodes_fully_aligned']}/{c['episodes_checked']} "
          f"aligned = {c['episode_pass_rate']}; cam pass {c['cam_pass_rate']}; "
          f"mismatch labs={c['labs_with_mismatch']}")
    return out
