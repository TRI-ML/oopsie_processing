"""Parse the precomputed S3 object listing and classify top-level prefixes.

The listing is the authoritative object inventory produced by a prior read-only
sweep. Lines are ``date time size key`` and keys start with the dataset prefix
(``anthonyliang/oopsie/``). We never re-list S3 here.
"""
from __future__ import annotations

import collections
import json
import os
from dataclasses import dataclass, field
from typing import Iterator

# Explicit, authoritative classification of the non-lab prefixes. Everything
# else that carries data is a real contributing lab. These names are stable
# facts from the audit, not guesses.
EXPLICIT_NONLAB = {
    "SOAR_origin": "source_origin",
    "AutoEval_origin": "source_origin",
    "oopsiedata-converted-v0.1": "duplicate",
    "roboarena_upload": "staging",
    "WebVisualizer": "preview",
    "Visualizer_Previews": "preview",
    ".cache": "hf_cache",
}
# name-prefix rules (dated archive suffixes)
EXPLICIT_NONLAB_PREFIXES = {
    "oopsie-face-raw-archive": "duplicate",
}


def _row(line: str):
    """Return (size:int, key:str) or None for a listing line."""
    parts = line.split(None, 3)
    if len(parts) < 4:
        return None
    try:
        size = int(parts[2])
    except ValueError:
        return None
    return size, parts[3].strip()


@dataclass
class PrefixStat:
    name: str
    n_obj: int = 0
    bytes: int = 0
    h5: int = 0
    mp4: int = 0
    ext_hidden: int = 0
    h5_bak: int = 0
    ds_store: int = 0
    zip: int = 0
    gitattributes: int = 0
    other: int = 0
    cache: int = 0  # .cache/huggingface bookkeeping objects


def iter_rows(listing_path: str, dataset_prefix: str) -> Iterator[tuple[int, str]]:
    with open(listing_path) as fh:
        for line in fh:
            r = _row(line)
            if r is None:
                continue
            size, key = r
            if not key.startswith(dataset_prefix):
                continue
            yield size, key


def prefix_of(key: str, dataset_prefix: str) -> str:
    rest = key[len(dataset_prefix):]
    if not rest:
        return ""
    return rest.split("/", 1)[0]


def scan_prefixes(listing_path: str, dataset_prefix: str) -> dict[str, PrefixStat]:
    """One streaming pass over the listing -> per-top-level-prefix stats."""
    stats: dict[str, PrefixStat] = {}
    for size, key in iter_rows(listing_path, dataset_prefix):
        name = prefix_of(key, dataset_prefix)
        if not name:
            continue
        st = stats.get(name)
        if st is None:
            st = stats[name] = PrefixStat(name=name)
        is_cache = "/.cache/" in key or key.endswith("/.cache")
        if is_cache:
            st.cache += 1
            continue
        st.n_obj += 1
        st.bytes += size
        low = key.lower()
        base = key.rsplit("/", 1)[-1]
        if key.endswith(".h5") or key.endswith(".hdf5"):
            st.h5 += 1
        elif key.endswith(".mp4"):
            st.mp4 += 1
        elif key.endswith(".hidden"):
            st.ext_hidden += 1
        elif key.endswith(".h5.bak") or key.endswith(".hdf5.bak"):
            st.h5_bak += 1
        elif base == ".DS_Store":
            st.ds_store += 1
        elif key.endswith(".zip"):
            st.zip += 1
        elif base == ".gitattributes":
            st.gitattributes += 1
        else:
            st.other += 1
    return stats


def classify(name: str, st: PrefixStat) -> tuple[str, str]:
    """Return (category, reason). Categories: real_lab, source_origin,
    duplicate, preview, staging, empty_placeholder."""
    if name in EXPLICIT_NONLAB:
        cat = EXPLICIT_NONLAB[name]
        return cat, f"explicit non-lab prefix ({cat})"
    for pfx, cat in EXPLICIT_NONLAB_PREFIXES.items():
        if name.startswith(pfx):
            return cat, f"explicit non-lab archive ({cat})"
    if st.h5 == 0 and st.mp4 == 0:
        return "empty_placeholder", "no h5 and no mp4 objects (placeholder prefix)"
    return "real_lab", "carries h5 + mp4 trajectory data"


def h5_keys_by_lab(listing_path: str, dataset_prefix: str,
                   labs: set[str] | None = None) -> dict[str, list[str]]:
    """lab -> sorted list of h5 keys (excludes .cache). Optionally restrict."""
    out: dict[str, list[str]] = collections.defaultdict(list)
    for _size, key in iter_rows(listing_path, dataset_prefix):
        if "/.cache/" in key:
            continue
        if not (key.endswith(".h5") or key.endswith(".hdf5")):
            continue
        lab = prefix_of(key, dataset_prefix)
        if labs is not None and lab not in labs:
            continue
        out[lab].append(key)
    for v in out.values():
        v.sort()
    return out


def mp4_basenames_by_lab(listing_path: str, dataset_prefix: str,
                         labs: set[str] | None = None) -> dict[str, dict[str, str]]:
    """lab -> {mp4 basename: full key}. Used to resolve video_paths pointers."""
    out: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for _size, key in iter_rows(listing_path, dataset_prefix):
        if "/.cache/" in key or not key.endswith(".mp4"):
            continue
        lab = prefix_of(key, dataset_prefix)
        if labs is not None and lab not in labs:
            continue
        out[lab][key.rsplit("/", 1)[-1]] = key
    return out


def sizes_by_key(listing_path: str, dataset_prefix: str,
                 keys: set[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    want = set(keys)
    for size, key in iter_rows(listing_path, dataset_prefix):
        if key in want:
            out[key] = size
    return out
