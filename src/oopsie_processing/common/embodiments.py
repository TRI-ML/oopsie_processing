"""Embodiment + camera registry loader (configs/embodiments.yaml).

Shared by stage5 (action harmonization), stage6 (camera roles) and both exporters
so every stage harmonizes identically. NO silent defaults: a lab with no registry
entry is a hard error at the point of use.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class Embodiment:
    lab_id: str
    embodiment_id: int
    embodiment_name: str
    robot_name: str
    is_biarm: bool
    gripper_name: str
    motion_keys: tuple[str, ...]
    gripper_keys: tuple[str, ...]
    cameras: dict  # canonical role -> native cam name (or dup-base sentinel)


@dataclass(frozen=True)
class Registry:
    common_dim: int
    canonical_camera_roles: tuple[str, ...]
    dup_base_sentinel: str
    embodiments: dict  # lab_id -> Embodiment


def load_registry(name: str = "embodiments") -> Registry:
    cfg = config.load(name)
    common_dim = config.require(cfg, "common_dim")
    roles = tuple(config.require(cfg, "canonical_camera_roles"))
    dup = config.require(cfg, "dup_base_sentinel")
    emb_cfg = config.require(cfg, "embodiments")
    embs = {}
    for lab, e in emb_cfg.items():
        cams = config.require(e, "cameras")
        for role in roles:
            if role not in cams:
                raise config.ConfigError(
                    f"embodiment '{lab}' cameras missing canonical role '{role}'")
        embs[lab] = Embodiment(
            lab_id=lab,
            embodiment_id=config.require(e, "embodiment_id"),
            embodiment_name=config.require(e, "embodiment_name"),
            robot_name=config.require(e, "robot_name"),
            is_biarm=config.require(e, "is_biarm"),
            gripper_name=config.require(e, "gripper_name"),
            motion_keys=tuple(config.require(e, "motion_keys")),
            gripper_keys=tuple(config.require(e, "gripper_keys")),
            cameras=dict(cams),
        )
    return Registry(common_dim=common_dim, canonical_camera_roles=roles,
                    dup_base_sentinel=dup, embodiments=embs)


def require_embodiment(reg: Registry, lab_id: str) -> Embodiment:
    if lab_id not in reg.embodiments:
        raise config.ConfigError(
            f"no embodiment registry entry for lab '{lab_id}' "
            f"(known: {sorted(reg.embodiments)})")
    return reg.embodiments[lab_id]
