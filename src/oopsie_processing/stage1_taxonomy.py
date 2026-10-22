"""Stage 1 — scope & prefix taxonomy.

Classify every top-level prefix in the object listing into real_lab /
source_origin / duplicate / preview / staging / empty_placeholder, reconcile the
known audit discrepancy (per_lab has 65 keys, 4 of which are infra), and emit an
authoritative include/exclude list. Read-only: listing + structure + audit.
"""
from __future__ import annotations

import json

from .common import config, listing, report


def run(stage_cfg_name: str = "stage1") -> dict:
    cfg = config.load_merged(stage_cfg_name)
    listing_path = config.require(cfg, "listing_file")
    prefix = config.require(cfg, "dataset_prefix")
    structure = json.load(open(config.require(cfg, "structure_file")))
    audit = json.load(open(config.require(cfg, "audit_file")))
    infra_in_per_lab = set(config.require(cfg, "audit_infra_in_per_lab"))
    exp_real = config.require(cfg, "expected_real_labs")
    exp_placeholders = config.require(cfg, "expected_empty_placeholders")

    stats = listing.scan_prefixes(listing_path, prefix)

    prefixes = {}
    by_cat: dict[str, list] = {}
    for name in sorted(stats):
        st = stats[name]
        cat, reason = listing.classify(name, st)
        rec = {
            "category": cat, "reason": reason,
            "n_obj": st.n_obj, "bytes": st.bytes, "gb": round(st.bytes / 1e9, 3),
            "h5": st.h5, "mp4": st.mp4,
            "ext_hidden": st.ext_hidden, "h5_bak": st.h5_bak,
            "ds_store": st.ds_store, "zip": st.zip,
            "gitattributes": st.gitattributes, "other": st.other,
            "cache_objs": st.cache,
        }
        prefixes[name] = rec
        by_cat.setdefault(cat, []).append(name)

    real = sorted(by_cat.get("real_lab", []))
    include = real
    exclude = {c: sorted(v) for c, v in by_cat.items() if c != "real_lab"}

    # Reconcile against the audit per_lab map.
    per_lab = set(audit.get("per_lab", {}).keys())
    reconciled_real = sorted(per_lab - infra_in_per_lab)
    recon = {
        "audit_per_lab_count": len(per_lab),
        "audit_infra_in_per_lab": sorted(infra_in_per_lab),
        "audit_real_labs_derived": len(reconciled_real),
        "listing_real_labs": len(real),
        "real_match": sorted(reconciled_real) == real,
        "in_listing_not_audit": sorted(set(real) - per_lab),
        "in_audit_not_listing": sorted((per_lab - infra_in_per_lab) - set(real)),
    }

    checks = {
        "real_labs_eq_expected": len(real) == exp_real,
        "placeholders_eq_expected":
            len(by_cat.get("empty_placeholder", [])) == exp_placeholders,
        "listing_vs_structure_prefix_count": {
            "listing": len(stats), "structure": structure.get("n_labs"),
        },
    }

    include_bytes = sum(prefixes[n]["bytes"] for n in real)
    exclude_bytes = sum(prefixes[n]["bytes"] for c in exclude for n in exclude[c])

    out = {
        "stage": "1_taxonomy",
        "authoritative_real_lab_count": len(real),
        "category_counts": {c: len(v) for c, v in sorted(by_cat.items())},
        "include_labs": include,
        "include_total_bytes": include_bytes,
        "include_total_gb": round(include_bytes / 1e9, 2),
        "exclude": exclude,
        "exclude_total_bytes": exclude_bytes,
        "exclude_total_gb": round(exclude_bytes / 1e9, 2),
        "reconciliation": recon,
        "checks": checks,
        "prefixes": prefixes,
    }
    report.write_json(cfg, "prefix_taxonomy.json", out)

    print(f"[stage1] real labs={len(real)} (expected {exp_real}, "
          f"audit-derived {len(reconciled_real)}, match={recon['real_match']})")
    print(f"[stage1] categories: {out['category_counts']}")
    print(f"[stage1] include {out['include_total_gb']} GB / "
          f"exclude {out['exclude_total_gb']} GB")
    for c, names in exclude.items():
        print(f"[stage1]   exclude[{c}] ({len(names)}): {', '.join(names) if len(names)<=20 else str(len(names))+' prefixes'}")
    return out
