"""EXACT mode: respondent-level reach using the Nielsen functions in
:mod:`optimizer.metrics.nielsen` (no formula is re-implemented here).

Respondent file (DATA_SPEC C.3 option A), CSV or parquet, one row per viewing record:

    panelist_id, date, daily_weight, channel, view_start, view_end

* ``date``          broadcast date (the 03:00-26:59 day the record belongs to).
* ``daily_weight``  the panelist's weight on ``date``. Every in-tab day of every panelist must
                    appear at least once; a day with no viewing is a row with empty
                    channel / view_start / view_end (weight-only row).
* ``view_start`` / ``view_end``  either seconds from the broadcast date's midnight (values
                    >= 86,400 = after midnight, same convention as breaks.parquet
                    ``start_sec``) or datetimes; end is exclusive (a record 10:00:00-10:00:30
                    is 30 seconds).

Exposure of a panelist to a spot = seconds of overlap of their viewing of the spot's channel
with [start, start + spot_length) on the spot's (mapped) date, summed over records, and
counted when >= ``reach.min_seconds`` (:func:`nielsen.exposure_counts`).
Reach / Reach N+ / Frequency use common weights (:func:`nielsen.common_weights`, rule
``reach.common_weight_rule``) over the period of analysis = all dates in the respondent file
(extract the file for exactly the analysis period, as eTAM does). GRP Absolute uses the
daily weight of the spot's day (:func:`nielsen.grp_abs`, GL p38).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..metrics import nielsen

RESPONDENT_COLUMNS = ["panelist_id", "date", "daily_weight", "channel", "view_start", "view_end"]


class RespondentFileError(ValueError):
    pass


def _to_seconds(col: pd.Series, dates: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(col):
        return col.to_numpy(float)
    if pd.api.types.is_timedelta64_dtype(col):
        return col.dt.total_seconds().to_numpy(float)
    dt = pd.to_datetime(col, errors="coerce")
    if dt.notna().sum() == 0 and col.notna().any():
        td = pd.to_timedelta(col, errors="coerce")
        return td.dt.total_seconds().to_numpy(float)
    return (dt - dates).dt.total_seconds().to_numpy(float)


def load_respondent_file(path: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load and validate a respondent-level viewing file. Raises RespondentFileError that
    lists every missing column."""
    if isinstance(path, pd.DataFrame):
        df = path.copy()
        src = "<DataFrame>"
    else:
        p = Path(path)
        src = str(p)
        if not p.exists():
            raise FileNotFoundError(f"respondent file not found: {p}")
        df = pd.read_parquet(p) if p.suffix.lower() in (".parquet", ".pq") else pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in RESPONDENT_COLUMNS if c not in df.columns]
    if missing:
        raise RespondentFileError(
            f"{src}: respondent file is missing required column(s) {missing}; "
            f"required: {RESPONDENT_COLUMNS} (see DATA_SPEC C.3 option A)")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["daily_weight"] = pd.to_numeric(df["daily_weight"], errors="raise").astype(float)
    if (df["daily_weight"] < 0).any():
        raise RespondentFileError(f"{src}: negative daily_weight")
    df["view_start_sec"] = _to_seconds(df["view_start"], df["date"])
    df["view_end_sec"] = _to_seconds(df["view_end"], df["date"])
    viewing = df["channel"].notna() & df["view_start_sec"].notna() & df["view_end_sec"].notna()
    bad = viewing & (df["view_end_sec"] < df["view_start_sec"])
    if bad.any():
        raise RespondentFileError(f"{src}: {int(bad.sum())} rows with view_end < view_start")
    w = df.groupby(["panelist_id", "date"])["daily_weight"].nunique()
    if (w > 1).any():
        raise RespondentFileError(f"{src}: panelist has two different daily weights on one date: "
                                  f"{list(w[w > 1].index[:5])}")
    df["is_viewing"] = viewing
    df.attrs["source"] = src
    return df


@dataclass
class SpotExposure:
    """Cached per-spot exposure data (person indices and overlap seconds)."""
    date: pd.Timestamp
    persons: np.ndarray          # person indices with overlap > 0
    seconds: np.ndarray          # overlap seconds
    viewers: np.ndarray          # person indices with overlap >= min_seconds
    viewer_daily_w: np.ndarray   # their daily weight on the spot's date


@dataclass
class RespondentPanel:
    df: pd.DataFrame
    min_seconds: float
    rule: str
    spot_len: float
    date_mapping: str = "identity"
    flight_start: pd.Timestamp | None = None
    persons: np.ndarray = field(init=False)
    common_w: np.ndarray = field(init=False)
    common_w_map: dict = field(init=False)
    dates: list = field(init=False)

    def __post_init__(self) -> None:
        if self.min_seconds is None:
            raise ValueError("reach.min_seconds is null: the eTAM reach minimum-seconds threshold "
                             "must be set in config before EXACT mode can run (DATA_SPEC C.4)")
        self.min_seconds = float(self.min_seconds)
        self.persons = np.array(sorted(self.df["panelist_id"].unique(), key=str), dtype=object)
        self.pidx = {p: i for i, p in enumerate(self.persons)}
        dw_rows = self.df.drop_duplicates(["panelist_id", "date"])[["panelist_id", "date", "daily_weight"]]
        self.daily_weights: dict = {}
        for p, d, w in dw_rows.itertuples(index=False):
            self.daily_weights.setdefault(p, {})[d] = float(w)
        self.dates = sorted(dw_rows["date"].unique())
        cw = nielsen.common_weights(self.daily_weights, self.rule, period_days=self.dates)
        self.common_w_map = cw
        self.common_w = np.array([cw.get(p, 0.0) for p in self.persons], dtype=float)
        self.daily_w = {d: g.set_index("panelist_id")["daily_weight"].to_dict()
                        for d, g in dw_rows.groupby("date")}
        v = self.df[self.df["is_viewing"]]
        self.viewing = {key: (np.array([self.pidx[p] for p in g["panelist_id"]], dtype=np.int64),
                              g["view_start_sec"].to_numpy(float), g["view_end_sec"].to_numpy(float))
                        for key, g in v.groupby(["channel", "date"])}
        self._cache: dict = {}

    # -- universe -------------------------------------------------------------------------
    def panel_universe(self) -> float:
        """Average daily universe = mean over dates of the sum of daily weights (nielsen.universe)."""
        return float(np.mean([nielsen.universe(list(w.values())) for w in self.daily_w.values()]))

    # -- dates ----------------------------------------------------------------------------
    def map_date(self, d: Any) -> pd.Timestamp:
        d = pd.Timestamp(d).normalize()
        if self.date_mapping == "identity":
            if d not in self.daily_w:
                raise KeyError(f"spot date {d.date()} not in respondent file dates "
                               f"({self.dates[0].date()}..{self.dates[-1].date()}); "
                               "set reach.exact.date_mapping: cyclic_weekday to plan future dates")
            return d
        if self.date_mapping == "cyclic_weekday":
            cands = [x for x in self.dates if pd.Timestamp(x).weekday() == d.weekday()]
            if not cands:
                raise KeyError(f"no respondent date with weekday {d.day_name()}")
            start = pd.Timestamp(self.flight_start) if self.flight_start is not None else d
            week = max(0, (d - start).days // 7)
            return pd.Timestamp(cands[week % len(cands)])
        raise ValueError(f"unknown reach.exact.date_mapping {self.date_mapping!r}")

    # -- exposures ------------------------------------------------------------------------
    def spot_exposure(self, channel: str, air_date: Any, start_min: float) -> SpotExposure:
        d = self.map_date(air_date)
        s0 = float(start_min) * 60.0
        key = (channel, d, s0)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        rec = self.viewing.get((channel, d))
        if rec is None:
            persons = np.zeros(0, np.int64)
            secs = np.zeros(0)
        else:
            pi, vs, ve = rec
            ov = np.clip(np.minimum(ve, s0 + self.spot_len) - np.maximum(vs, s0), 0.0, None)
            tot = np.bincount(pi, weights=ov, minlength=len(self.persons))
            persons = np.nonzero(tot > 0)[0]
            secs = tot[persons]
        viewers = persons[secs >= self.min_seconds]
        dw = self.daily_w[d]
        vdw = np.array([dw.get(self.persons[i], 0.0) for i in viewers], dtype=float)
        out = SpotExposure(d, persons, secs, viewers, vdw)
        self._cache[key] = out
        return out
