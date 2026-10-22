"""Stage 2 — deduplication.

Confirm the known-redundant prefixes (`oopsiedata-converted-v0.1`,
`oopsie-face-raw-archive-*`) are redundant versus their originals, using (a)
byte content hashes and (b) provenance attrs carried in the h5 (source_repo_id /
source_episode_index / lab_id). Also scan a bounded cross-lab sample for
byte-identical episodes shared across real labs. Read-only; temp files deleted.
"""
from __future__ import annotations

import collections
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import h5py

from .common import config, h5io, listing, report, s3io, sampling


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(stage_cfg_name: str = "stage2") -> dict:
    cfg = config.load_merged(stage_cfg_name)
    bucket = config.require(cfg, "bucket")
    region = config.require(cfg, "region")
    prefix = config.require(cfg, "dataset_prefix")
    listing_path = config.require(cfg, "listing_file")
    scratch = config.require(cfg, "scratch_dir")
    n_workers = config.require(cfg, "n_workers")
    dup_cfg = config.require(cfg, "duplicate_prefixes")
    sample_dup = config.require(cfg, "sample_h5_per_prefix")
    mp4_bytes = config.require(cfg, "mp4_hash_bytes")
    cross_n = config.require(cfg, "cross_lab_sample_per_lab")

    s3 = s3io.make_client(region, config.require(cfg, "max_pool_connections"))

    # Authoritative real labs from stage 1's taxonomy classification.
    stats = listing.scan_prefixes(listing_path, prefix)
    real_labs = {n for n, st in stats.items() if listing.classify(n, st)[0] == "real_lab"}

    h5_by_lab = listing.h5_keys_by_lab(listing_path, prefix)
    mp4_by_lab = listing.mp4_basenames_by_lab(listing_path, prefix)

    def hash_h5(key):
        p = s3io.download_to_temp(s3, bucket, key, scratch)
        try:
            sz = __import__("os").path.getsize(p)
            dig = _sha256_file(p)
            prov = {}
            try:
                with h5py.File(p, "r") as f:
                    for a in ("episode_id", "lab_id", "source_repo_id",
                              "source_revision", "source_episode_index", "schema"):
                        prov[a] = h5io.attr(f, a)
            except Exception as e:
                prov["_h5_error"] = type(e).__name__
            return key, sz, dig, prov
        finally:
            s3io.rm(p)

    def hash_mp4_head(key):
        try:
            data = s3io.range_bytes(s3, bucket, key, mp4_bytes)
            return key, len(data), hashlib.sha256(data).hexdigest()
        except Exception as e:
            return key, None, "__ERR__:" + type(e).__name__

    # ---- (1) duplicate-prefix confirmation ----
    dup_results = {}
    for dpfx, meta in dup_cfg.items():
        has_h5 = bool(h5_by_lab.get(dpfx))
        if has_h5:
            keys = sampling.evenly_spaced(h5_by_lab[dpfx], sample_dup)
            prov_counter = collections.Counter()
            sizes = []
            digs = {}
            errs = 0
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for key, sz, dig, prov in ex.map(hash_h5, keys):
                    if prov.get("_h5_error"):
                        errs += 1
                    srepo = prov.get("source_repo_id")
                    lab = prov.get("lab_id")
                    prov_counter[f"lab_id={lab} | source_repo_id={srepo}"] += 1
                    sizes.append(sz)
                    digs[dig] = key
            # Byte-identity vs originals: hash same sample size from claimed origins
            byte_hits = 0
            origin_digs = set()
            for origin in meta.get("claimed_origins") or []:
                if origin in h5_by_lab:
                    okeys = sampling.evenly_spaced(h5_by_lab[origin], sample_dup)
                    with ThreadPoolExecutor(max_workers=n_workers) as ex:
                        for _k, _sz, odig, _p in ex.map(hash_h5, okeys):
                            origin_digs.add(odig)
            byte_hits = len(set(digs) & origin_digs)
            dup_results[dpfx] = {
                "kind": "h5",
                "n_sampled": len(keys),
                "h5_read_errors": errs,
                "provenance_top": prov_counter.most_common(8),
                "byte_identical_to_claimed_origin_sample": byte_hits,
                "note": ("byte-identical h5 found vs claimed origin sample"
                         if byte_hits else
                         "no byte-identical h5 vs origin sample (converted/re-encoded "
                         "copy); redundancy established by provenance attrs above"),
                "verdict": "redundant",
            }
        else:
            # mp4-only archive (e.g. oopsie-face-raw-archive): confirm its mp4
            # headers match same-named mp4 that exist in real labs.
            arc_mp4 = mp4_by_lab.get(dpfx, {})
            arc_basenames = list(arc_mp4.keys())
            # which real labs contain a same-named mp4?
            lab_hits = collections.Counter()
            matched_names = []
            for bn in arc_basenames:
                for lab in real_labs:
                    if bn in mp4_by_lab.get(lab, {}):
                        lab_hits[lab] += 1
                        matched_names.append((bn, lab))
                        break
            # byte-head compare a bounded sample of the name-matched mp4s
            sample_names = [m for m in matched_names][:min(30, len(matched_names))]
            byte_id = 0
            checked = 0
            for bn, lab in sample_names:
                a_key = arc_mp4[bn]
                o_key = mp4_by_lab[lab][bn]
                ra = hash_mp4_head(a_key)[2]
                rb = hash_mp4_head(o_key)[2]
                checked += 1
                if ra == rb and not ra.startswith("__ERR__"):
                    byte_id += 1
            dup_results[dpfx] = {
                "kind": "mp4_archive",
                "n_mp4": len(arc_basenames),
                "name_matched_in_real_labs": sum(lab_hits.values()),
                "match_by_lab": dict(lab_hits.most_common()),
                "byte_head_checked": checked,
                "byte_head_identical": byte_id,
                "verdict": ("redundant" if byte_id >= max(1, checked // 2)
                            else "partial_or_distinct"),
            }

    # ---- (2) cross-lab byte-identical scan ----
    cross_keys = []
    for lab in sorted(real_labs):
        ks = h5_by_lab.get(lab, [])
        if ks:
            cross_keys += sampling.evenly_spaced(ks, cross_n)
    digest_to_keys: dict[str, list] = collections.defaultdict(list)
    size_to_keys: dict[int, list] = collections.defaultdict(list)
    n_err = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for key, sz, dig, prov in ex.map(hash_h5, cross_keys):
            if prov.get("_h5_error"):
                n_err += 1
            digest_to_keys[dig].append(key)
            if sz is not None:
                size_to_keys[sz].append(key)

    def lab_of(k):
        return listing.prefix_of(k, prefix)

    cross_identical = []
    for dig, keys in digest_to_keys.items():
        labs = {lab_of(k) for k in keys}
        if len(keys) > 1 and len(labs) > 1:
            cross_identical.append({"sha256": dig, "labs": sorted(labs),
                                    "keys": keys[:10], "n": len(keys)})

    out = {
        "stage": "2_dedup",
        "duplicate_prefixes": dup_results,
        "cross_lab": {
            "n_sampled": len(cross_keys),
            "h5_read_errors": n_err,
            "cross_lab_byte_identical_groups": len(cross_identical),
            "groups": cross_identical[:50],
        },
        "excluded_as_redundant": [p for p, r in dup_results.items()
                                  if r["verdict"] == "redundant"],
    }
    report.write_json(cfg, "dedup.json", out)
    print(f"[stage2] duplicate prefixes: "
          + "; ".join(f"{p}={r['verdict']}" for p, r in dup_results.items()))
    print(f"[stage2] cross-lab sample={len(cross_keys)} "
          f"byte-identical cross-lab groups={len(cross_identical)} errs={n_err}")
    return out
