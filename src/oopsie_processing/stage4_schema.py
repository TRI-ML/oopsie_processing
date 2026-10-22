"""Stage 4 — schema validation against oopsiedata_format_v1.

Per real lab, over a bounded evenly-spaced h5 sample: required root attrs
present, robot_profile parses, video_paths pointers resolve to listed mp4 keys,
robot_states populated. Emits a per-lab validation record.
"""
from __future__ import annotations

import collections

from .common import config, report, scan


def run(stage_cfg_name: str = "stage4", records=None) -> dict:
    cfg = config.load_merged(stage_cfg_name)
    sample_per_lab = config.require(cfg, "sample_h5_per_lab")
    # rel_eps only matters for DQ; schema uses the same scan. Pull DQ's value so
    # the shared cache key is consistent across stages.
    dq_cfg = config.load_merged("data_quality")
    rel_eps = config.require(dq_cfg, "constant_traj_rel_range")

    if records is None:
        records = scan.scan_real_labs(cfg, sample_per_lab, rel_eps)

    per_lab = {}
    labs = collections.defaultdict(list)
    for r in records:
        labs[r["lab"]].append(r)

    tot_sampled = tot_valid = tot_err = 0
    for lab in sorted(labs):
        recs = labs[lab]
        n = len(recs)
        n_err = sum(1 for r in recs if "error" in r)
        good = [r for r in recs if "error" not in r]
        n_valid = sum(1 for r in good if r["schema"]["valid"])
        viol = collections.Counter()
        attr_missing = collections.Counter()
        vid_unresolved_eps = 0
        rp_bad = 0
        rs_empty = 0
        for r in good:
            s = r["schema"]
            for v in s["violations"]:
                viol[v.split(":")[0]] += 1
            for a, ok in s["attrs_present"].items():
                if not ok:
                    attr_missing[a] += 1
            if not s["robot_profile_ok"]:
                rp_bad += 1
            if s["video_unresolved"]:
                vid_unresolved_eps += 1
            if s["n_state_keys"] == 0:
                rs_empty += 1
        per_lab[lab] = {
            "n_sampled": n, "n_read_errors": n_err,
            "n_valid": n_valid,
            "pass_rate": round(n_valid / n, 4) if n else None,
            "schema_attr_values": sorted({r["schema"]["schema_attr"]
                                          for r in good if r["schema"]["schema_attr"]}),
            "attr_missing_counts": dict(attr_missing),
            "robot_profile_parse_failures": rp_bad,
            "robot_states_empty": rs_empty,
            "episodes_with_unresolved_video_ptr": vid_unresolved_eps,
            "violation_type_counts": dict(viol),
        }
        tot_sampled += n
        tot_valid += n_valid
        tot_err += n_err

    out = {
        "stage": "4_schema",
        "corpus": {
            "n_labs": len(per_lab),
            "n_sampled": tot_sampled,
            "n_valid": tot_valid,
            "n_read_errors": tot_err,
            "overall_pass_rate": round(tot_valid / tot_sampled, 4) if tot_sampled else None,
            "labs_100pct": sorted(l for l, d in per_lab.items() if d["pass_rate"] == 1.0),
            "labs_below_99pct": {l: d["pass_rate"] for l, d in per_lab.items()
                                 if d["pass_rate"] is not None and d["pass_rate"] < 0.99},
        },
        "per_lab": per_lab,
    }
    report.write_json(cfg, "schema_validation.json", out)
    c = out["corpus"]
    print(f"[stage4] schema pass {c['n_valid']}/{c['n_sampled']} "
          f"= {c['overall_pass_rate']}; labs<99%={list(c['labs_below_99pct'].keys())}")
    return out
