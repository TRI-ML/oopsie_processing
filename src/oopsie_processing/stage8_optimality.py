"""Stage 8 — optimality-label consolidation.

Consolidate the scattered success/failure signal into one uniform per-episode
record. Precedence (NEVER infer):
  1. episode_annotations/<annotator>.success  (authoritative 0/1 float) + its
     taxonomy (outcome / failure_category / severity).
  2. For labs in `folder_fallback_labs` (CMU_Aire): the success|failure parent
     folder name, when no annotation success is present.
  3. Otherwise optimality = unknown (success=None, label_source="unknown").
"""
from __future__ import annotations

import json

from .common import config, h5io


def _first_annotation_success(f):
    """Return (success_float, annotator, taxonomy_dict) from the first annotator
    that carries a `success` attr, or (None, None, None)."""
    g = f.get("episode_annotations")
    if g is None:
        return None, None, None
    for ann in g.keys():
        sub = g[ann]
        s = sub.attrs.get("success")
        if s is None:
            continue
        tax_raw = h5io.decode(sub.attrs.get("taxonomy"))
        tax = {}
        if isinstance(tax_raw, str) and tax_raw.strip():
            try:
                tax = json.loads(tax_raw)
            except json.JSONDecodeError:
                tax = {}
        return float(s), ann, tax
    return None, None, None


def _folder_outcome(h5_key: str):
    """success|failure from the h5's parent folder name, else None."""
    parts = h5_key.split("/")
    for p in reversed(parts[:-1]):
        if p == "success":
            return 1.0
        if p == "failure":
            return 0.0
    return None


def consolidate(f, h5_key: str, lab_id: str, folder_fallback_labs) -> dict:
    succ, annotator, tax = _first_annotation_success(f)
    if succ is not None:
        return {
            "success": succ,
            "taxonomy_outcome": tax.get("outcome"),
            "failure_category": tax.get("failure_category"),
            "severity": tax.get("severity"),
            "annotator": annotator,
            "label_source": "annotation",
        }
    if lab_id in folder_fallback_labs:
        fo = _folder_outcome(h5_key)
        if fo is not None:
            return {
                "success": fo,
                "taxonomy_outcome": "success" if fo >= 0.5 else "failure",
                "failure_category": None,
                "severity": None,
                "annotator": None,
                "label_source": "folder",
            }
    return {
        "success": None,
        "taxonomy_outcome": None,
        "failure_category": None,
        "severity": None,
        "annotator": None,
        "label_source": "unknown",
    }
