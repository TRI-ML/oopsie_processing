"""Stage 5 — action-space harmonization.

Resolve the heterogeneous native action spaces to a common padded vector
(`common_dim`, default 18) plus a per-episode `action_dim_valid` mask, keyed by
embodiment. Layout (identical to the proven pi05 + vla_foundry converters):

    [0, motion_dim)                 = motion channel  (joint_ or cartesian_position)
    [motion_dim, motion_dim+grip)   = gripper channel
    [motion_dim+grip, common_dim)   = zero padding
    mask[:valid] = 1.0, rest = 0.0  (valid = motion_dim + gripper_dim)

The motion/gripper slots are picked PER EPISODE from the embodiment's ordered
`motion_keys` / `gripper_keys` candidate lists (first populated wins), so labs that
mix action spaces internally (e.g. CMU_Aire: cartesian xarm7 vs joint yam) are
harmonized per episode rather than forced to one signature. Labs whose episodes
span more than one native motion signature are FLAGGED (never silently merged).

`run()` emits an action-space mapping report over the Phase-A h5 scan sample.
"""
from __future__ import annotations

import collections
import math

import numpy as np

from .common import config, embodiments, h5io, report, scan


def _dim_of(shape) -> int:
    if shape is None or len(shape) == 0:
        return 0
    if len(shape) == 1:
        return 1
    return int(math.prod(shape[1:]))


def pick_slot(populated: dict, candidates) -> tuple[str, int] | None:
    """First (key, dim) whose key is populated, in candidate priority order."""
    for key in candidates:
        if key in populated:
            return key, _dim_of(populated[key])
    return None


def action_signature(action_slots: dict, emb: embodiments.Embodiment) -> dict | None:
    """Native (motion,gripper) signature for an episode, or None if unresolvable."""
    m = pick_slot(action_slots, emb.motion_keys)
    g = pick_slot(action_slots, emb.gripper_keys)
    if m is None or g is None:
        return None
    return {"motion_key": m[0], "motion_dim": m[1],
            "gripper_key": g[0], "gripper_dim": g[1], "valid": m[1] + g[1]}


def harmonize_action(f, emb: embodiments.Embodiment, common_dim: int):
    """Return (action [T, common_dim] f32, mask [common_dim] f32, signature dict).

    Raises if the episode's action slots cannot be resolved against the embodiment
    or exceed common_dim (fail loudly; never truncate silently).
    """
    slots = h5io.actions(f)
    sig = action_signature(slots, emb)
    if sig is None:
        raise ValueError(
            f"{emb.lab_id}: no populated motion/gripper slot among "
            f"motion={emb.motion_keys} gripper={emb.gripper_keys}; present={list(slots)}")
    if sig["valid"] > common_dim:
        raise ValueError(
            f"{emb.lab_id}: valid action dim {sig['valid']} exceeds common_dim {common_dim}")
    motion = np.asarray(f[f"actions/{sig['motion_key']}"][()], dtype=np.float32).reshape(
        -1, sig["motion_dim"])
    grip = np.asarray(f[f"actions/{sig['gripper_key']}"][()], dtype=np.float32).reshape(
        -1, sig["gripper_dim"])
    t = min(motion.shape[0], grip.shape[0])
    act = np.zeros((t, common_dim), dtype=np.float32)
    act[:, : sig["motion_dim"]] = motion[:t]
    act[:, sig["motion_dim"] : sig["valid"]] = grip[:t]
    mask = np.zeros((common_dim,), dtype=np.float32)
    mask[: sig["valid"]] = 1.0
    return act, mask, sig


def harmonize_state(f, emb: embodiments.Embodiment, t: int, common_dim: int) -> np.ndarray:
    """State vector built from the SAME motion+gripper slots under robot_states,
    concatenated and zero-padded to common_dim. Raises if unresolvable."""
    slots = h5io.robot_states(f)
    m = pick_slot(slots, emb.motion_keys)
    g = pick_slot(slots, emb.gripper_keys)
    if m is None or g is None:
        raise ValueError(
            f"{emb.lab_id}: robot_states missing motion/gripper slot; present={list(slots)}")
    motion = np.asarray(f[f"observations/robot_states/{m[0]}"][()], dtype=np.float32).reshape(
        -1, m[1])
    grip = np.asarray(f[f"observations/robot_states/{g[0]}"][()], dtype=np.float32).reshape(
        -1, g[1])
    tt = min(t, motion.shape[0], grip.shape[0])
    cat = np.concatenate([motion[:tt], grip[:tt]], axis=1)
    if cat.shape[1] > common_dim:
        raise ValueError(f"{emb.lab_id}: state dim {cat.shape[1]} exceeds common_dim {common_dim}")
    state = np.zeros((tt, common_dim), dtype=np.float32)
    state[:, : cat.shape[1]] = cat
    return state


def run(stage_cfg_name: str = "stage5") -> dict:
    """Report the native action-space signature per registered lab over the
    Phase-A scan sample; flag labs whose episodes span >1 motion signature."""
    cfg = config.load_merged(stage_cfg_name)
    reg = embodiments.load_registry()
    sample_per_lab = config.require(cfg, "sample_h5_per_lab")
    dq_cfg = config.load_merged("data_quality")
    rel_eps = config.require(dq_cfg, "constant_traj_rel_range")
    records = scan.scan_real_labs(cfg, sample_per_lab, rel_eps)

    by_lab = collections.defaultdict(list)
    for r in records:
        if "error" not in r:
            by_lab[r["lab"]].append(r)

    per_lab = {}
    for lab in sorted(reg.embodiments):
        emb = reg.embodiments[lab]
        sigs = collections.Counter()
        for r in by_lab.get(lab, []):
            slots = r["dq"]["action"]["slots"]
            sig = action_signature(slots, emb)
            if sig is None:
                sigs["unresolved"] += 1
            else:
                sigs[f"{sig['motion_key']}:{sig['motion_dim']}+{sig['gripper_key']}:{sig['gripper_dim']}=>{sig['valid']}"] += 1
        distinct = [s for s in sigs if s != "unresolved"]
        per_lab[lab] = {
            "embodiment_id": emb.embodiment_id,
            "embodiment_name": emb.embodiment_name,
            "common_dim": reg.common_dim,
            "n_sampled": sum(sigs.values()),
            "signatures": dict(sigs),
            "internal_mixed": len(distinct) > 1,
        }

    out = {
        "stage": "5_harmonize",
        "common_dim": reg.common_dim,
        "registered_labs": sorted(reg.embodiments),
        "internal_mixed_labs": sorted(l for l, d in per_lab.items() if d["internal_mixed"]),
        "per_lab": per_lab,
    }
    report.write_json(cfg, "action_space_mapping.json", out)
    print(f"[stage5] registered labs={len(per_lab)} common_dim={reg.common_dim} "
          f"internal_mixed={out['internal_mixed_labs']}")
    return out
