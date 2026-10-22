"""Stage 6 — camera / modality standardization.

Map each lab's non-standard camera names onto the canonical pi05 roles
(base / left_wrist / right_wrist), preserving the native names. The role->native
mapping is explicit per embodiment (configs/embodiments.yaml); a single-arm lab's
missing wrist slot duplicates the resolved base camera (the `dup_base_sentinel`),
matching the proven converters. NO silent defaults: a declared source camera that
is absent from an episode's observations/video_paths is a hard error.
"""
from __future__ import annotations

from .common import config, embodiments


def _resolve_candidate(lab_id, role, src, vpaths, reg):
    """Resolve one role's native camera name against this episode's video_paths.

    `src` may be a single native name or a priority list of acceptable native
    names (labs that used more than one camera-naming convention across episodes,
    e.g. BONN_MB base = [left, global]). Returns the first candidate present. NO
    silent default: if none of the declared candidates is present it is a hard
    error."""
    cands = list(src) if isinstance(src, (list, tuple)) else [src]
    for c in cands:
        if c == reg.dup_base_sentinel:
            raise config.ConfigError(
                f"{lab_id}: dup-base sentinel cannot appear in a candidate list for role {role}")
        if c in vpaths:
            return c
    raise ValueError(
        f"{lab_id}: no camera for role {role} (tried {cands}) in video_paths {list(vpaths)}")


def resolve_roles(vpaths: dict, emb: embodiments.Embodiment, reg: embodiments.Registry) -> dict:
    """canonical role -> native camera name, validated against this episode's
    video_paths. Returns e.g. {"base": "image", "left_wrist": "wrist_image",
    "right_wrist": "image"}."""
    base_src = emb.cameras["base"]
    if base_src == reg.dup_base_sentinel:
        raise config.ConfigError(f"{emb.lab_id}: base camera cannot be the dup-base sentinel")
    base_use = _resolve_candidate(emb.lab_id, "base", base_src, vpaths, reg)
    roles = {}
    for role in reg.canonical_camera_roles:
        src = emb.cameras[role]
        if src == reg.dup_base_sentinel:
            roles[role] = base_use
        else:
            roles[role] = _resolve_candidate(emb.lab_id, role, src, vpaths, reg)
    return roles
