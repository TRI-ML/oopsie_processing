"""Stage 9 — instruction normalization.

Normalize `language_instruction`: strip + collapse internal whitespace for a
clean display string, and derive a case-folded normalized key for grouping /
distinct-instruction accounting. No auto-merge of near-duplicates; the key is a
pure deterministic normalization.
"""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def normalize(raw) -> tuple[str | None, str | None]:
    """Return (display, norm_key). (None, None) if there is no instruction."""
    if raw is None:
        return None, None
    s = _WS.sub(" ", str(raw)).strip()
    if not s:
        return None, None
    return s, s.lower()
