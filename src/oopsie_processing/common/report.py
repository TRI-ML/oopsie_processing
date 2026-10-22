"""Report writer: JSON / JSONL into reports/ and a durable mirror.

Machine-readable reports go to the gitignored reports/ dir and are also copied
to a durability directory (outside the repo) so Phase B can consume them.
"""
from __future__ import annotations

import json
import os
import shutil

from . import config


def _ensure(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def reports_dir(cfg: dict) -> str:
    rd = config.require(cfg, "reports_dir")
    if not os.path.isabs(rd):
        rd = os.path.join(config.REPO_ROOT, rd)
    return _ensure(rd)


def durable_dir(cfg: dict) -> str:
    return _ensure(config.require(cfg, "durable_dir"))


def write_json(cfg: dict, name: str, obj) -> str:
    rd = reports_dir(cfg)
    path = os.path.join(rd, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    shutil.copy2(path, os.path.join(durable_dir(cfg), name))
    return path


def write_jsonl(cfg: dict, name: str, rows) -> str:
    rd = reports_dir(cfg)
    path = os.path.join(rd, name)
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    shutil.copy2(path, os.path.join(durable_dir(cfg), name))
    return path
