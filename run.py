#!/usr/bin/env python3
"""Phase-A CLI. Invoke one stage or the full read-only inventory.

Usage:
    python run.py stage1          # prefix taxonomy
    python run.py stage2          # dedup
    python run.py stage3          # cruft & defect quarantine
    python run.py stage4          # schema validation
    python run.py stage7          # alignment
    python run.py dq              # data-quality
    python run.py phaseA          # 1,2,3,7 + a single shared h5 scan for 4 & dq

Every stage reads its YAML config (configs/<stage>.yaml + common.yaml). Missing
config keys are hard errors (no silent defaults).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from oopsie_processing import (  # noqa: E402
    stage1_taxonomy, stage2_dedup, stage3_quarantine,
    stage4_schema, stage7_alignment, data_quality,
)
from oopsie_processing.common import config, scan  # noqa: E402

STAGES = {
    "stage1": stage1_taxonomy.run,
    "stage2": stage2_dedup.run,
    "stage3": stage3_quarantine.run,
    "stage4": stage4_schema.run,
    "stage7": stage7_alignment.run,
    "dq": data_quality.run,
}


def phaseA():
    stage1_taxonomy.run()
    stage2_dedup.run()
    stage3_quarantine.run()
    stage7_alignment.run()
    # Single shared h5 scan feeds both schema validation and data-quality.
    s4cfg = config.load_merged("stage4")
    dqcfg = config.load_merged("data_quality")
    sample_per_lab = config.require(s4cfg, "sample_h5_per_lab")
    rel_eps = config.require(dqcfg, "constant_traj_rel_range")
    records = scan.scan_real_labs(s4cfg, sample_per_lab, rel_eps)
    stage4_schema.run(records=records)
    data_quality.run(records=records)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd == "phaseA":
        phaseA()
    elif cmd in STAGES:
        STAGES[cmd]()
    else:
        print(f"unknown stage: {cmd}")
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
