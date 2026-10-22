"""Config loading with NO silent defaults.

Hard project rule: if a value a stage needs is absent from its YAML, we fail
loudly. `require()` is the only accessor stages use for mandatory values.
"""
from __future__ import annotations

import os
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")


class ConfigError(RuntimeError):
    pass


def load(name: str) -> dict:
    """Load a YAML config by bare name (e.g. 'stage1') from configs/."""
    path = name if os.path.isabs(name) else os.path.join(CONFIG_DIR, name)
    if not path.endswith((".yaml", ".yml")):
        path += ".yaml"
    if not os.path.exists(path):
        raise ConfigError(f"config file not found: {path}")
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"config {path} did not parse to a mapping")
    return data


def require(cfg: dict, key: str) -> Any:
    """Return cfg[key] or raise. Dotted keys descend nested mappings.

    Never returns a default. An absent or null value is a hard error.
    """
    cur: Any = cfg
    parts = key.split(".")
    for i, p in enumerate(parts):
        if not isinstance(cur, dict) or p not in cur:
            raise ConfigError(
                f"required config key missing: '{key}' (stopped at '{'.'.join(parts[:i+1])}')"
            )
        cur = cur[p]
    if cur is None:
        raise ConfigError(f"required config key '{key}' is present but null")
    return cur


def load_merged(stage_name: str) -> dict:
    """Load common.yaml and the stage config, merged (stage overrides common).

    Both files must exist. No defaults are injected.
    """
    common = load("common")
    stage = load(stage_name)
    merged = dict(common)
    merged.update(stage)
    return merged
