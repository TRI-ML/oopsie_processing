"""Deterministic bounded sampling over sorted key lists."""
from __future__ import annotations


def evenly_spaced(keys, cap: int):
    """Up to `cap` keys, evenly spaced across the sorted input (inclusive ends).

    Deterministic so a re-run samples the same episodes.
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    n = len(keys)
    if n <= cap:
        return list(keys)
    if cap == 1:
        return [keys[0]]
    return [keys[round(i * (n - 1) / (cap - 1))] for i in range(cap)]
