"""Small helpers shared by the reach modules (no data, no config side effects)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import project_root


def abs_path(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def flight_days(cfg: dict) -> int:
    """Number of days in the flight (cfg flight.start .. flight.end inclusive)."""
    s = pd.Timestamp(cfg["flight"]["start"]).date()
    e = pd.Timestamp(cfg["flight"]["end"]).date()
    return (e - s).days + 1


def daypart_of(start_min: float, dayparts: dict[str, list]) -> str:
    """Daypart name for a broadcast-day minute (03:00 = 180 ... 26:59 = 1619)."""
    m = float(start_min)
    for name, (lo, hi) in dayparts.items():
        if lo <= m < hi:
            return name
    raise ValueError(f"start_min {start_min} not covered by reach.estimate.dayparts")
