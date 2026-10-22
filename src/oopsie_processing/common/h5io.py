"""h5 readers for oopsiedata_format_v1.

Root attrs: episode_id, lab_id, language_instruction, operator_name,
robot_profile (JSON: control_freq, robot_name, is_biarm, gripper_name,
camera_names, action_space ...), schema.
Groups: actions/<slot>, observations/robot_states/<key>,
observations/video_paths/<cam> (string pointer), optional
episode_annotations/<annotator> (attrs success, taxonomy).
"""
from __future__ import annotations

import json

import h5py
import numpy as np

ROOT_ATTRS_REQUIRED = [
    "episode_id", "lab_id", "language_instruction", "operator_name", "robot_profile",
]


def decode(val):
    if val is None:
        return None
    if isinstance(val, bytes):
        return val.decode("utf-8", "replace")
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return None
        return decode(val.reshape(-1)[0])
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def attr(f, name):
    return decode(f.attrs.get(name))


def parse_robot_profile(f):
    """Return (dict_or_None, error_or_None)."""
    rp = attr(f, "robot_profile")
    if rp is None:
        return None, "missing"
    if isinstance(rp, str):
        try:
            return json.loads(rp), None
        except json.JSONDecodeError as e:
            return None, f"json_error:{e}"
    if isinstance(rp, dict):
        return rp, None
    return None, "unexpected_type"


def _len0(obj):
    try:
        if obj is None or obj.shape is None or len(obj.shape) == 0:
            return 0
        return int(obj.shape[0])
    except Exception:
        return 0


def populated_slots(group):
    """{slot_name: shape_list} for datasets in a group with a nonzero length."""
    out = {}
    if not isinstance(group, h5py.Group):
        return out
    for k in group.keys():
        o = group.get(k)
        if isinstance(o, h5py.Dataset) and o.shape is not None and len(o.shape) > 0 and o.shape[0] > 0:
            out[k] = list(o.shape)
    return out


def robot_states(f):
    return populated_slots(f.get("observations/robot_states"))


def actions(f):
    return populated_slots(f.get("actions"))


def video_paths(f):
    """{cam_name: pointer_string}."""
    out = {}
    g = f.get("observations/video_paths")
    if isinstance(g, h5py.Group):
        for k in g.keys():
            out[k] = decode(g[k][()])
    return out


def get_T(f):
    """Trajectory length from robot_states, falling back to actions."""
    g = f.get("observations/robot_states")
    if isinstance(g, h5py.Group):
        for k in g.keys():
            L = _len0(g.get(k))
            if L > 0:
                return L, f"robot_states/{k}"
    a = f.get("actions")
    if isinstance(a, h5py.Group):
        for k in a.keys():
            L = _len0(a.get(k))
            if L > 0:
                return L, f"actions/{k}"
    return None, None


def control_freq(f):
    rp, _ = parse_robot_profile(f)
    if rp and rp.get("control_freq"):
        try:
            return float(rp["control_freq"]), "robot_profile"
        except (TypeError, ValueError):
            pass
    for a in ("control_freq", "control_frequency"):
        v = attr(f, a)
        if v is not None:
            try:
                return float(v), "root_attr"
            except (TypeError, ValueError):
                pass
    return None, None


def annotations(f):
    """List of {annotator, success, taxonomy} from episode_annotations/*."""
    out = []
    g = f.get("episode_annotations")
    if isinstance(g, h5py.Group):
        for k in g.keys():
            sub = g[k]
            rec = {"annotator": k}
            rec["success"] = decode(sub.attrs.get("success"))
            rec["taxonomy"] = decode(sub.attrs.get("taxonomy"))
            out.append(rec)
    return out


def read_arrays(f, group_path):
    """Yield (name, numpy array) for populated datasets under group_path."""
    g = f.get(group_path)
    if not isinstance(g, h5py.Group):
        return
    for k in g.keys():
        o = g.get(k)
        if isinstance(o, h5py.Dataset) and o.shape is not None and len(o.shape) > 0 and o.shape[0] > 0:
            yield k, o[()]
