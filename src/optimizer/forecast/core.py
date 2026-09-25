"""Per-slot audience forecast — pure functions (METHODOLOGY section 2).

A *target* is anything we forecast the audience of ONE 30" spot for: an October grid row, or a
back-test hold-out slot. The expected audience of one spot placed in a slot is the (robust)
average ``rating_abs`` of the breaks in that slot window — breaks are never summed.
``rating_abs`` in breaks.parquet is already ``TRP_Absolute * 60 / break_sec`` (CLAUDE.md).

Evidence hierarchy (2.1) — every level uses only non-event breaks (``is_event`` rows are dropped
by the caller when ``etam.exclude_event_days``):

1. ``program``: same eTAM title (``etam_title``, only when the history is on the SAME channel),
   same run type (grid ``is_rerun`` <-> break ``first_run``), break start within
   ``[start - tol, end + tol)`` (tol = ``forecast.time_tolerance_min``), any weekday.
2. ``slot``: same channel, same weekday group (``forecast.weekday_groups``), break start within
   ``[start, end)``, any programme. Generic-slot rows whose pool rule says "Prefer rerun
   breaks" use the rerun-only sub-pool when it has >= ``min_breaks``.
3. ``channel_daypart``: same channel, dayparts overlapped by the window, same day-type
   (``forecast.day_types``); low-sample channels use ``forecast.low_sample_policy`` (pooled over
   all days by default) and ONLY this level. If this level has < ``min_breaks`` it widens to all
   day-types, then to the whole channel (noted in ``evidence_desc``).

A level is *usable* when it has >= ``forecast.min_breaks`` breaks. The forecast is the most
specific usable level, shrunk toward the next usable level (empirical-Bayes partial pooling
with a pseudo-count k = ``forecast.shrinkage_k``)::

    est_L = (n_L * x_L + k * est_parent) / (n_L + k)

where ``x_L`` is the robust estimate (``forecast.estimator``) of level L's breaks and
``est_parent`` is the (itself shrunk) estimate of the next usable level. The top level is not
shrunk. With k = 6 a program with 6 breaks is a 50/50 blend, 54 breaks -> 90% own history.

Estimators (``forecast.estimator``): ``median``; ``mean``; ``trimmed_mean_P`` = mean after
trimming P% of the total weight at EACH end (fractional trimming, so it is exactly defined for
the weighted bootstrap and equals ``scipy.stats.trim_mean`` whenever P% x n is an integer).

Uncertainty (2.3): p50 = the point estimate above. p10/p90 are a predictive interval for the
mean audience of the slot on one airing day, from B = ``forecast.bootstrap_n`` draws of
``theta_b * r_b``:

* ``theta_b`` — day-cluster bootstrap: resample the evidence *days* with replacement (breaks of
  the same day stay together), re-estimate, re-shrink toward the fixed parent estimate;
* ``r_b`` — slot-day noise: evidence breaks are grouped into (date, start // w) cells, w =
  max(target window, ``forecast.predictive_min_window_min``); r_b is a random cell's mean
  divided by the level's raw estimate (additive deviation if that estimate is 0).

Quantiles ``forecast.quantiles``; the interval is then forced to contain p50 and widened for
flagged slots: ``p10' = max(0, p50 - w*(p50 - p10))``, ``p90' = p50 + w*(p90 - p50)`` with
w = max over the slot's flags of ``forecast.interval_widening`` (new_program / live_special /
low_sample). Seeds: numpy ``default_rng([random_seed, crc32(evidence key)])`` — deterministic.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
LEVELS = ("program", "slot", "channel_daypart")


# --------------------------------------------------------------------------- parameters
@dataclass
class ForecastParams:
    min_breaks: int = 6
    time_tolerance_min: int = 30
    weekday_groups: list[list[str]] = field(default_factory=lambda: [["Sun", "Mon", "Tue", "Wed"], ["Thu"], ["Fri"], ["Sat"]])
    estimator: str = "trimmed_mean_10"
    shrinkage_k: float = 6.0
    min_match_confidence: float = 0.0
    day_types: dict[str, list[str]] = field(default_factory=lambda: {"weekday": ["Sun", "Mon", "Tue", "Wed", "Thu"], "weekend": ["Fri", "Sat"]})
    dayparts: dict[str, list[int]] = field(default_factory=lambda: {"early": [180, 360], "morning": [360, 720], "daytime": [720, 1080], "prime": [1080, 1440], "late": [1440, 1680]})
    low_sample_policy: str = "channel_daypart"
    live_special_patterns: list[str] = field(default_factory=lambda: ["KHALEEJI", r"\bCUP\b", r"\bMATCH\b", "FIFA", "FOOTBALL"])
    quantiles: list[float] = field(default_factory=lambda: [0.10, 0.50, 0.90])
    random_seed: int = 42
    bootstrap_n: int = 1000
    interval_widening: dict[str, float] = field(default_factory=lambda: {"new_program": 1.5, "live_special": 2.0, "low_sample": 1.5})
    high_uncertainty_rel_width: float = 2.0
    predictive_min_window_min: int = 30

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any], **overrides: Any) -> "ForecastParams":
        f = dict(cfg.get("forecast", {}))
        kw = {k: f[k] for k in cls.__dataclass_fields__ if k in f}
        kw.update(overrides)
        return cls(**kw)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def weekday_group_map(groups: list[list[str]]) -> dict[str, int]:
    """weekday name -> group index; every weekday must appear exactly once."""
    out: dict[str, int] = {}
    for i, g in enumerate(groups):
        for d in g:
            if d in out:
                raise ValueError(f"weekday {d} in two weekday_groups")
            out[d] = i
    missing = set(DAYS) - set(out)
    if missing:
        raise ValueError(f"weekday_groups miss {sorted(missing)}")
    return out


def day_type_map(day_types: dict[str, list[str]]) -> dict[str, str]:
    out = {d: name for name, days in day_types.items() for d in days}
    missing = set(DAYS) - set(out)
    if missing:
        raise ValueError(f"day_types miss {sorted(missing)}")
    return out


def dayparts_overlapping(start_min: int, end_min: int, dayparts: dict[str, list[int]]) -> tuple[str, ...]:
    """Names of dayparts whose [a, b) intersects [start_min, end_min)."""
    names = tuple(n for n, (a, b) in dayparts.items() if start_min < b and end_min > a)
    if not names:  # outside every daypart: nearest one
        mid = (start_min + end_min) / 2
        names = (min(dayparts, key=lambda n: min(abs(mid - dayparts[n][0]), abs(mid - dayparts[n][1]))),)
    return names


def is_live_special(title: str, is_live: bool, patterns: list[str]) -> bool:
    """Live sport / one-off special: an is_live row whose title matches a live_special pattern."""
    if not is_live:
        return False
    t = str(title).upper()
    return any(re.search(p, t) for p in patterns)


# --------------------------------------------------------------------------- estimators
def _trim_alpha(estimator: str) -> float | None:
    if estimator == "mean":
        return 0.0
    if estimator == "median":
        return None
    m = re.fullmatch(r"trimmed_mean_(\d+(?:\.\d+)?)", estimator)
    if not m:
        raise ValueError(f"unknown forecast.estimator {estimator!r}")
    a = float(m.group(1)) / 100.0
    if not 0 <= a < 0.5:
        raise ValueError("trim share must be in [0, 50)")
    return a


def weighted_estimate(x_sorted: np.ndarray, w: np.ndarray, estimator: str) -> np.ndarray:
    """Robust estimate of sorted values ``x_sorted`` (n,) under weights ``w`` (B, n) or (n,).
    Returns shape (B,) (or scalar array for 1-D w). Weights are day-cluster bootstrap counts
    (unit weights = the plain estimator)."""
    w2 = np.atleast_2d(np.asarray(w, dtype=float))
    x = np.asarray(x_sorted, dtype=float)
    T = w2.sum(axis=1, keepdims=True)
    cw = np.cumsum(w2, axis=1)
    alpha = _trim_alpha(estimator)
    if alpha is None:  # (weighted) median: average of the two middle order statistics
        half = T / 2.0
        i1 = np.argmax(cw >= half - 1e-9, axis=1)
        i2 = np.argmax(cw > half + 1e-9, axis=1)
        out = (x[i1] + x[i2]) / 2.0
    else:
        lo, hi = alpha * T, (1 - alpha) * T
        c_hi = np.clip(cw, lo, hi)
        c_lo = np.clip(np.concatenate([np.zeros_like(T), cw[:, :-1]], axis=1), lo, hi)
        out = ((c_hi - c_lo) * x).sum(axis=1) / (hi - lo)[:, 0]
    return out if np.ndim(w) == 2 else out[0]


def robust_estimate(values, estimator: str) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    if x.size == 0:
        return float("nan")
    return float(weighted_estimate(x, np.ones(x.size), estimator))


def shrink(x_level: float, n_level: int, parent: float | None, k: float) -> float:
    """Empirical-Bayes partial pooling: (n*x + k*parent)/(n+k); no parent -> x."""
    if parent is None or not np.isfinite(parent) or k <= 0:
        return float(x_level)
    return float((n_level * x_level + k * parent) / (n_level + k))


# --------------------------------------------------------------------------- break index
class BreakIndex:
    """Column arrays of the (event-free, in-window) evidence breaks for fast masking."""

    def __init__(self, breaks: pd.DataFrame, p: ForecastParams):
        b = breaks.reset_index(drop=True)
        self.p = p
        self.df = b
        self.channel = b["channel"].astype(str).to_numpy()
        self.program = b["program"].astype(str).to_numpy()
        self.first_run = b["first_run"].astype(bool).to_numpy()
        self.minute = (b["start_sec"].to_numpy() // 60).astype(int)
        self.weekday = b["weekday"].astype(str).to_numpy()
        wg = weekday_group_map(p.weekday_groups)
        self.wkgrp = np.array([wg[d] for d in self.weekday])
        dt = day_type_map(p.day_types)
        self.daytype = np.array([dt[d] for d in self.weekday])
        dp_names = list(p.dayparts)
        dp = np.full(len(b), "", dtype=object)
        for n in dp_names:
            a, e = p.dayparts[n]
            dp[(self.minute >= a) & (self.minute < e)] = n
        self.daypart = dp.astype(str)
        self.date = pd.to_datetime(b["broadcast_date"]).dt.strftime("%Y-%m-%d").to_numpy()
        self.y = b["rating_abs"].to_numpy(dtype=float)
        self.break_id = b["break_id"].astype(str).to_numpy()
        self._by_channel = {c: (self.channel == c) for c in np.unique(self.channel)}

    def chan(self, ch: str) -> np.ndarray:
        m = self._by_channel.get(ch)
        return m if m is not None else np.zeros(len(self.y), dtype=bool)


# --------------------------------------------------------------------------- evidence
def evidence_masks(t: dict[str, Any], bi: BreakIndex) -> dict[str, tuple[np.ndarray, str]]:
    """Boolean masks (over bi) for each hierarchy level of target ``t`` + a text description.

    Target keys: channel, weekday, start_min, end_min, is_rerun, etam_title ('' if none),
    use_program (bool), prefer_rerun_pool, low_sample.
    """
    p = bi.p
    ch = bi.chan(t["channel"])
    s, e = int(t["start_min"]), int(t["end_min"])
    out: dict[str, tuple[np.ndarray, str]] = {}
    wg = weekday_group_map(p.weekday_groups)[t["weekday"]]
    dtype = day_type_map(p.day_types)[t["weekday"]]
    dps = dayparts_overlapping(s, e, p.dayparts)

    if t.get("low_sample"):
        by_type = p.low_sample_policy == "channel_daypart_daytype"
        m3 = ch & np.isin(bi.daypart, dps) & ((bi.daytype == dtype) if by_type else True)
        desc = f"{t['channel']} x {'+'.join(dps)}" + (f" x {dtype}" if by_type else " x all days") + " (low-sample pooled)"
        out["channel_daypart"] = _widen_top(m3, ch, bi, dps, desc)
        return out

    if t.get("use_program") and t.get("etam_title"):
        tol = p.time_tolerance_min
        m1 = (ch & (bi.program == t["etam_title"]) & (bi.first_run == (not bool(t["is_rerun"])))
              & (bi.minute >= s - tol) & (bi.minute < e + tol))
        run = "rerun" if t["is_rerun"] else "first-run"
        out["program"] = (m1, f"{t['etam_title']} ({run}) on {t['channel']}, start within window +/-{tol}min")

    m2 = ch & (bi.wkgrp == wg) & (bi.minute >= s) & (bi.minute < e)
    desc2 = f"{t['channel']} wkgrp {'/'.join(p.weekday_groups[wg])} breaks in window"
    if t.get("prefer_rerun_pool"):
        m2r = m2 & ~bi.first_run
        if m2r.sum() >= p.min_breaks:
            m2, desc2 = m2r, desc2 + " (rerun breaks only)"
    out["slot"] = (m2, desc2)

    m3 = ch & np.isin(bi.daypart, dps) & (bi.daytype == dtype)
    out["channel_daypart"] = _widen_top(m3, ch, bi, dps, f"{t['channel']} x {'+'.join(dps)} x {dtype}")
    return out


def _widen_top(m3, ch, bi, dps, desc):
    if m3.sum() >= bi.p.min_breaks:
        return m3, desc
    m = ch & np.isin(bi.daypart, dps)
    if m.sum() >= bi.p.min_breaks:
        return m, desc + " -> widened to all days"
    return ch, desc + " -> widened to whole channel"


def pool_top_programs(mask: np.ndarray, bi: BreakIndex, top: int = 2) -> str:
    if not mask.any():
        return ""
    vc = pd.Series(bi.program[mask]).value_counts(normalize=True).head(top)
    return ", ".join(f"{k} {v:.0%}" for k, v in vc.items())


# --------------------------------------------------------------------------- one target
def forecast_one(t: dict[str, Any], bi: BreakIndex, with_intervals: bool = True) -> dict[str, Any]:
    p = bi.p
    masks = evidence_masks(t, bi)
    raw: dict[str, tuple[float, int]] = {}
    for lvl in LEVELS:
        if lvl in masks:
            m = masks[lvl][0]
            raw[lvl] = (robust_estimate(bi.y[m], p.estimator), int(m.sum()))
    usable = [l for l in LEVELS if l in raw and raw[l][1] >= p.min_breaks]
    if "channel_daypart" not in usable:  # top level always usable (it was widened to >= min or whole channel)
        usable.append("channel_daypart")
    # shrink from the top down
    est: dict[str, float] = {}
    parent: float | None = None
    for lvl in reversed(usable):
        x, n = raw[lvl]
        est[lvl] = shrink(x, n, parent, p.shrinkage_k) if lvl != usable[-1] else x
        parent = est[lvl]
    level = usable[0]
    parent_of_level = est[usable[1]] if len(usable) > 1 else None
    p50 = est[level]
    res: dict[str, Any] = {
        "forecast_level": level, "evidence_n": raw[level][1], "p50": p50,
        "raw_level_est": raw[level][0], "parent_est": parent_of_level,
        "evidence_desc": masks[level][1],
    }
    for lvl in LEVELS:
        res[f"est_{lvl}"] = raw[lvl][0] if lvl in raw else np.nan
        res[f"n_{lvl}"] = raw[lvl][1] if lvl in raw else 0
    lvl_mask = masks[level][0]
    res["pool_top_programs"] = pool_top_programs(lvl_mask, bi) if level != "program" else t.get("etam_title", "")
    res["_mask"] = lvl_mask
    if with_intervals:
        key = "|".join(str(t.get(k, "")) for k in ("channel", "weekday", "start_min", "end_min", "is_rerun", "etam_title", "low_sample")) + "|" + level
        p10, p90 = predictive_interval(bi, lvl_mask, p50, raw[level][0], raw[level][1], parent_of_level,
                                       int(t["end_min"]) - int(t["start_min"]), key)
        res["p10_raw"], res["p90_raw"] = p10, p90
    return res


def predictive_interval(bi: BreakIndex, mask: np.ndarray, p50: float, raw_est: float, n: int,
                        parent: float | None, window_min: int, key: str) -> tuple[float, float]:
    """Day-cluster bootstrap x slot-day noise; returns (q_lo, q_hi) forced to bracket p50."""
    p = bi.p
    q_lo, q_hi = p.quantiles[0], p.quantiles[-1]
    y = bi.y[mask]
    if y.size == 0:
        return p50, p50
    rng = np.random.default_rng([int(p.random_seed), zlib.crc32(key.encode("utf-8"))])
    dates = bi.date[mask]
    order = np.argsort(y, kind="stable")
    ys, ds = y[order], dates[order]
    udays, day_idx = np.unique(ds, return_inverse=True)
    D, B = len(udays), int(p.bootstrap_n)
    counts = rng.multinomial(D, np.full(D, 1.0 / D), size=B)          # (B, D) day draws
    w = counts[:, day_idx]                                               # (B, n) break weights
    theta = weighted_estimate(ys, w, p.estimator)
    nb = w.sum(axis=1)
    if parent is not None and p.shrinkage_k > 0:
        theta = (nb * theta + p.shrinkage_k * parent) / (nb + p.shrinkage_k)
    # slot-day noise cells
    width = max(int(window_min), int(p.predictive_min_window_min))
    cell = pd.Series(bi.y[mask]).groupby([bi.date[mask], bi.minute[mask] // width]).mean().to_numpy()
    pick = cell[rng.integers(0, cell.size, size=B)]
    if raw_est > 0:
        draw = theta * (pick / raw_est)
    else:
        draw = np.maximum(theta + (pick - raw_est), 0.0)
    lo, hi = np.quantile(draw, [q_lo, q_hi])
    return float(min(lo, p50)), float(max(hi, p50))


def widen(p50: float, p10: float, p90: float, w: float) -> tuple[float, float]:
    return max(0.0, p50 - w * (p50 - p10)), p50 + w * (p90 - p50)


# --------------------------------------------------------------------------- many targets
TARGET_KEYS = ("channel", "weekday", "start_min", "end_min", "is_rerun", "etam_title", "use_program",
               "prefer_rerun_pool", "low_sample")


def forecast_targets(targets: pd.DataFrame, breaks: pd.DataFrame, p: ForecastParams,
                     with_intervals: bool = True, keep_evidence: bool = False
                     ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Forecast every row of ``targets`` (see ``evidence_masks`` for the required columns plus
    ``target_id``, ``is_new``, ``live_special``). ``breaks`` must already be event-free and
    restricted to the history window. Returns (forecast frame, evidence frame or None)."""
    bi = BreakIndex(breaks, p)
    cache: dict[tuple, dict[str, Any]] = {}
    rows, ev = [], []
    for t in targets.to_dict("records"):
        key = tuple(t.get(k) for k in TARGET_KEYS)
        if key not in cache:
            cache[key] = forecast_one(t, bi, with_intervals)
        r = dict(cache[key])
        mask = r.pop("_mask")
        r["target_id"] = t["target_id"]
        if with_intervals:
            flags = [f for f, on in (("new_program", t.get("is_new")), ("live_special", t.get("live_special")),
                                     ("low_sample", t.get("low_sample"))) if on]
            wf = max([p.interval_widening.get(f, 1.0) for f in flags], default=1.0)
            r["widening"] = wf
            r["p10"], r["p90"] = widen(r["p50"], r["p10_raw"], r["p90_raw"], wf)
        if keep_evidence:
            idx = np.flatnonzero(mask)
            ev.append(pd.DataFrame({"target_id": t["target_id"], "forecast_level": r["forecast_level"],
                                    "break_id": bi.break_id[idx], "rating_abs": bi.y[idx]}))
        rows.append(r)
    out = pd.DataFrame(rows)
    evid = pd.concat(ev, ignore_index=True) if ev else None
    return out, evid
