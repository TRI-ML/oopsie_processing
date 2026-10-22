"""Data-quality checks over the bounded h5 sample.

NaN/Inf in actions/robot_states, degenerate/constant trajectories, action value
ranges per embodiment, proprio dims, duration/frequency outliers, and
instruction coverage (reused from the precomputed instruction_counts.json).
"""
from __future__ import annotations

import collections
import json

from .common import config, report, scan


def run(stage_cfg_name: str = "data_quality", records=None) -> dict:
    cfg = config.load_merged(stage_cfg_name)
    sample_per_lab = config.require(cfg, "sample_h5_per_lab")
    rel_eps = config.require(cfg, "constant_traj_rel_range")
    short_s = config.require(cfg, "duration_outlier_short_s")
    long_s = config.require(cfg, "duration_outlier_long_s")
    instr_counts = json.load(open(config.require(cfg, "instruction_counts_file")))

    if records is None:
        records = scan.scan_real_labs(cfg, sample_per_lab, rel_eps)

    labs = collections.defaultdict(list)
    for r in records:
        labs[r["lab"]].append(r)

    per_lab = {}
    tot = collections.Counter()
    for lab in sorted(labs):
        good = [r for r in labs[lab] if "error" not in r]
        n = len(good)
        nan = sum(1 for r in good if r["dq"]["has_nan"])
        inf = sum(1 for r in good if r["dq"]["has_inf"])
        degen = sum(1 for r in good if r["dq"]["degenerate_action"])
        durs = [r["dq"]["duration_s"] for r in good if r["dq"]["duration_s"]]
        short = sum(1 for d in durs if d < short_s)
        lng = sum(1 for d in durs if d > long_s)
        act_dims = sorted({r["dq"]["action"]["dims"] for r in good if r["dq"]["action"]["dims"]})
        state_dims = sorted({r["dq"]["state"]["dims"] for r in good if r["dq"]["state"]["dims"]})
        robots = sorted({str(r["dq"]["robot_name"]) for r in good if r["dq"]["robot_name"]})
        freqs = sorted({r["dq"]["control_freq"] for r in good if r["dq"]["control_freq"]})
        amins = [r["dq"]["action"]["min"] for r in good if r["dq"]["action"]["min"] is not None]
        amaxs = [r["dq"]["action"]["max"] for r in good if r["dq"]["action"]["max"] is not None]
        # episodes with a partially-constant action vector (some, not all, dims flat)
        part_const = sum(1 for r in good
                         if 0 < r["dq"]["action"]["constant_dims"] < r["dq"]["action"]["dims"])
        ic = instr_counts.get(lab, {})
        per_lab[lab] = {
            "n_analyzed": n,
            "nan_episodes": nan, "inf_episodes": inf,
            "degenerate_action_episodes": degen,
            "partially_constant_action_episodes": part_const,
            "duration_short_outliers": short, "duration_long_outliers": lng,
            "action_dims_seen": act_dims, "state_dims_seen": state_dims,
            "control_freqs_seen": freqs,
            "robot_names": robots,
            "mixed_action_dims": len(act_dims) > 1,
            "action_value_range": [min(amins) if amins else None,
                                   max(amaxs) if amaxs else None],
            "instruction_distinct": ic.get("distinct_instructions"),
            "instruction_attr_present": ic.get("attr_present"),
            "instruction_total_h5": ic.get("n_total_h5"),
        }
        tot["n"] += n
        tot["nan"] += nan
        tot["inf"] += inf
        tot["degen"] += degen
        tot["short"] += short
        tot["long"] += lng

    flags = {
        "labs_with_nan": {l: d["nan_episodes"] for l, d in per_lab.items() if d["nan_episodes"]},
        "labs_with_inf": {l: d["inf_episodes"] for l, d in per_lab.items() if d["inf_episodes"]},
        "labs_with_degenerate": {l: d["degenerate_action_episodes"]
                                 for l, d in per_lab.items() if d["degenerate_action_episodes"]},
        "labs_mixed_action_dims": {l: d["action_dims_seen"]
                                   for l, d in per_lab.items() if d["mixed_action_dims"]},
        "labs_with_duration_outliers": {l: [d["duration_short_outliers"], d["duration_long_outliers"]]
                                        for l, d in per_lab.items()
                                        if d["duration_short_outliers"] or d["duration_long_outliers"]},
        "single_instruction_labs": sorted(l for l, d in per_lab.items()
                                          if d["instruction_distinct"] == 1),
    }
    out = {
        "stage": "data_quality",
        "corpus_totals": dict(tot),
        "flags": flags,
        "per_lab": per_lab,
    }
    report.write_json(cfg, "data_quality.json", out)
    print(f"[dq] analyzed={tot['n']} nan_eps={tot['nan']} inf_eps={tot['inf']} "
          f"degenerate={tot['degen']} shortOutliers={tot['short']} longOutliers={tot['long']}")
    print(f"[dq] mixed-action-dim labs={list(flags['labs_mixed_action_dims'].keys())}")
    return out
