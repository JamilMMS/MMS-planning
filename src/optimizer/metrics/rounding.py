"""Display rounding for Nielsen metrics - reporter use only.

All calculations run at full precision (``cfg['rounding']['internal'] == 'full_precision'``).
Rounding is applied once, at display time, with the single rule in ``cfg['rounding']``:

    rounding:
      display: round_half_up        # or: truncate
      decimals: {rating_pct: 2, grp_pct: 1, reach_pct: 1, ots: 2, cpm: 2, usd: 0}

Why configurable: the Nielsen PDFs are inconsistent (see docs/NIELSEN_DEFINITIONS.md,
"Evidence from the PDFs (Phase 2)"): some printed values are truncated (OTS 4.026 -> 4.02,
MDT p92; Frequency 1.7389 -> 1.73, MDT p64) and others rounded (Share 2.138 -> 2.14,
MDT p10; Reach % 0.023875 -> 0.024, MDT p19). Confirm against real eTAM exports.

Never call these functions inside a calculation.
"""
from __future__ import annotations

import math
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Any, Mapping

_MODES = {"round_half_up": ROUND_HALF_UP, "truncate": ROUND_DOWN}


def _quantize(value: float, decimals: int, mode: str) -> float:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return value
    # str() gives the shortest repr, so 2.675 is treated as the decimal 2.675 (not 2.67499..)
    q = Decimal(1).scaleb(-int(decimals))
    return float(Decimal(str(value)).quantize(q, rounding=_MODES[mode]))


def round_half_up(value: float, decimals: int) -> float:
    """Round half away from zero (commercial rounding) to ``decimals`` places."""
    return _quantize(value, decimals, "round_half_up")


def truncate(value: float, decimals: int) -> float:
    """Truncate toward zero to ``decimals`` places (the convention seen in MDT p64, p92)."""
    return _quantize(value, decimals, "truncate")


def display_round(value: Any, kind: str, cfg: Mapping) -> Any:
    """Round ``value`` for display using ``cfg['rounding']``.

    ``kind`` selects the number of decimals from ``cfg['rounding']['decimals']`` (e.g.
    'rating_pct', 'grp_pct', 'reach_pct', 'ots', 'cpm', 'usd'); ``cfg['rounding']['display']``
    selects the rule ('round_half_up' or 'truncate'). Works element-wise on lists.
    """
    rcfg = cfg["rounding"]
    mode = rcfg["display"]
    if mode not in _MODES:
        raise ValueError(f"unknown rounding.display {mode!r}; expected one of {sorted(_MODES)}")
    decimals_map = rcfg["decimals"]
    if kind not in decimals_map:
        raise KeyError(f"no rounding.decimals entry for kind {kind!r}; known: {sorted(decimals_map)}")
    decimals = int(decimals_map[kind])
    if isinstance(value, (list, tuple)):
        return type(value)(_quantize(float(v), decimals, mode) for v in value)
    return _quantize(float(value), decimals, mode)
