"""Reach engine: one interface, three methods (EXACT | CALIBRATED | ESTIMATE).

    engine = ReachEngine.from_config(cfg, forecast=None)
    out = engine.reach(schedule_df)            # dict, see RESULT_KEYS
    st = engine.new_state()
    engine.add(st, spot); engine.remove(st, spot)
    engine.marginal_reach(st, candidate)       # incremental Reach 1+ (people), O(1) / O(#cells)
    engine.marginal_reach_n(st, candidate, 3)  # incremental Reach 3+ (people)
    engine.evaluate(st)                        # same keys as reach()
    engine.sensitivity(schedule_df)            # ESTIMATE / CALIBRATED: k x (1-s, 1, 1+s)

A *spot* is any mapping / namedtuple / Series with at least ``channel`` and ``aud_abs``
(per-spot forecast audience, p50) plus ``air_date`` and ``start_min`` (EXACT mode and
daypart cells) and optionally ``daypart`` / ``slot_id``. For speed, pre-compute spots once
with ``engine.prepare(spot)`` / ``engine.prepare_many(df)`` and pass the prepared objects.

States are mutated in place by add/remove (and returned, for chaining); use
``state.copy()`` to branch.

GRP Absolute = sum of aud_abs (impressions); GRP % = GRP Absolute / universe x 100 with
universe = ``target.universe``; OTS = GRP Absolute / Reach 1+. EXACT mode computes GRP from
the respondent data with daily weights (as eTAM) and also reports ``grp_abs_forecast``.
Every result carries ``method``.
"""
from __future__ import annotations

import copy
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ..metrics import nielsen
from . import curves
from .common import abs_path, daypart_of

METHODS = ("EXACT", "CALIBRATED", "ESTIMATE")
RESULT_KEYS = ("reach_1plus_abs", "reach_1plus_pct", "reach_3plus_abs", "reach_3plus_pct",
               "reach_n_dist", "reach_n_plus_abs", "reach_n_plus_pct", "grp_abs", "grp_pct",
               "ots", "n_spots", "universe", "method", "params_summary")


_abs = abs_path


def _get(spot: Any, key: str, default: Any = None) -> Any:
    if isinstance(spot, Mapping):
        return spot.get(key, default)
    if isinstance(spot, pd.Series):
        return spot.get(key, default)
    return getattr(spot, key, default)


def _records(schedule: pd.DataFrame | Iterable) -> list:
    if isinstance(schedule, pd.DataFrame):
        return schedule.to_dict("records")
    return list(schedule)


# =============================================================================== base
class ReachEngine:
    method: str = ""

    @staticmethod
    def from_config(cfg: dict, forecast: pd.DataFrame | None = None, *,
                    params: dict | None = None, respondents: pd.DataFrame | str | None = None,
                    ) -> "ReachEngine":
        """Build the engine for ``cfg['reach']['mode']``.

        forecast : optional DataFrame with ``slot_id`` and ``aud_abs`` (or ``aud_abs_p50``)
            used to fill ``aud_abs`` for schedules that only carry slot_id.
        params : override the parameter dict (ESTIMATE / CALIBRATED) instead of reading JSON.
        respondents : override the respondent file (EXACT) with a path or DataFrame.
        """
        mode = str(cfg["reach"]["mode"]).upper()
        if mode not in METHODS:
            raise ValueError(f"reach.mode must be one of {METHODS}, got {mode!r}")
        if mode == "EXACT":
            return ExactReachEngine(cfg, forecast=forecast, respondents=respondents)
        if params is None:
            key = cfg["reach"]["params_path"] if mode == "ESTIMATE" else cfg["reach"]["calibration"]["params_path"]
            p = _abs(key)
            if not p.exists():
                hint = ("python -m optimizer.reach.fit --config config/plan_config.yaml" if mode == "ESTIMATE"
                        else "python -m optimizer.reach.calibrate --config config/plan_config.yaml")
                raise FileNotFoundError(f"{mode} reach parameters not found at {p}; run `{hint}`")
            params = json.loads(p.read_text(encoding="utf-8"))
            params.setdefault("_path", str(p))
            if mode == "ESTIMATE":
                _warn_if_stale(params)
        if params.get("method") != mode:
            raise ValueError(f"parameter file method {params.get('method')!r} does not match reach.mode {mode!r}")
        return CurveReachEngine(cfg, params, forecast=forecast)

    # ---- shared helpers -----------------------------------------------------------------
    def _init_common(self, cfg: dict, forecast: pd.DataFrame | None) -> None:
        self.cfg = cfg
        self.universe = float(cfg["target"]["universe"])
        self.kmax = int(cfg["reach"]["max_frequency"])
        if self.kmax < 3:
            raise ValueError("reach.max_frequency must be >= 3 (Reach 3+ is a KPI)")
        self._fc = None
        if forecast is not None:
            col = "aud_abs" if "aud_abs" in forecast.columns else "aud_abs_p50"
            self._fc = dict(zip(forecast["slot_id"], forecast[col].astype(float)))

    def _aud(self, spot: Any) -> float:
        a = _get(spot, "aud_abs")
        if (a is None or (isinstance(a, float) and math.isnan(a))) and self._fc is not None:
            a = self._fc.get(_get(spot, "slot_id"))
        if a is None or (isinstance(a, float) and math.isnan(a)):
            raise ValueError(f"spot has no aud_abs (slot_id={_get(spot, 'slot_id')!r}) and no forecast value")
        a = float(a)
        if a < 0:
            raise ValueError(f"negative aud_abs {a} for slot_id={_get(spot, 'slot_id')!r}")
        return a

    def prepare_many(self, schedule: pd.DataFrame | Iterable) -> list:
        return [self.prepare(s) for s in _records(schedule)]

    def reach(self, schedule: pd.DataFrame | Iterable) -> dict:
        st = self.new_state()
        for s in _records(schedule):
            self.add(st, s)
        return self.evaluate(st)

    def _result(self, reach1: float, nplus_abs: np.ndarray, grp_abs: float, n_spots: int,
                params_summary: dict, extra: dict | None = None) -> dict:
        U = self.universe
        nplus_abs = np.asarray(nplus_abs, float)
        exact_n = np.append(nplus_abs[:-1] - nplus_abs[1:], nplus_abs[-1])
        out = {
            "reach_1plus_abs": float(reach1),
            "reach_1plus_pct": float(reach1) / U * 100.0,
            "reach_3plus_abs": float(nplus_abs[2]),
            "reach_3plus_pct": float(nplus_abs[2]) / U * 100.0,
            "reach_n_dist": {n + 1: float(v) for n, v in enumerate(exact_n)},
            "reach_n_plus_abs": {n + 1: float(v) for n, v in enumerate(nplus_abs)},
            "reach_n_plus_pct": {n + 1: float(v) / U * 100.0 for n, v in enumerate(nplus_abs)},
            "max_frequency_bucket": self.kmax,
            "grp_abs": float(grp_abs),
            "grp_pct": nielsen.grp_pct(float(grp_abs), U) if grp_abs else 0.0,
            "ots": nielsen.ots(float(grp_abs), float(reach1)) if reach1 > 0 else None,
            "n_spots": int(n_spots),
            "universe": U,
            "method": self.method,
            "params_summary": params_summary,
        }
        if extra:
            out.update(extra)
        check_sanity(out)
        return out

    def sensitivity(self, schedule: pd.DataFrame | Iterable, k_multipliers: Iterable[float] | None = None) -> list[dict]:
        raise NotImplementedError


def check_sanity(out: dict, tol: float = 1e-6) -> None:
    """Hard sanity constraints on every result (raise AssertionError on violation)."""
    U = out["universe"]
    r1, r3 = out["reach_1plus_abs"], out["reach_3plus_abs"]
    assert out["method"] in METHODS, "result without a valid method label"
    assert -tol <= r1 <= U * (1 + tol), f"reach 1+ {r1} outside [0, universe]"
    assert r3 <= r1 * (1 + tol) + tol, f"reach 3+ {r3} > reach 1+ {r1}"
    nplus = list(out["reach_n_plus_abs"].values())
    assert all(a + tol * max(1.0, a) >= b for a, b in zip(nplus, nplus[1:])), "Reach N+ not non-increasing"
    # EXACT: GRP uses daily weights and reach common weights (GL p38 / p16), so reach 1+ may
    # exceed GRP Absolute by the weight difference; the model modes enforce it strictly.
    if out["method"] != "EXACT":
        assert r1 <= out["grp_abs"] * (1 + tol) + tol, f"reach 1+ {r1} > GRP Absolute {out['grp_abs']}"
        if out["ots"] is not None:
            assert out["ots"] >= 1 - tol, f"OTS {out['ots']} < 1"


def _warn_if_stale(params: dict) -> None:
    src, sha = params.get("breaks_path"), params.get("breaks_sha256")
    if not src or not sha:
        return
    import hashlib
    p = _abs(src)
    if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() != sha:
        warnings.warn(f"{p} changed since the ESTIMATE reach parameters were fitted "
                      f"({params.get('generated_at')}); re-run `python -m optimizer.reach.fit`",
                      stacklevel=3)


# =============================================================================== curve modes
@dataclass(frozen=True)
class PreparedSpot:
    cell: int
    g_pct: float
    aud_abs: float
    slot_id: Any = None


class CurveState:
    """Per-cell accumulators for the curve model (ESTIMATE / CALIBRATED)."""
    __slots__ = ("g", "n", "r", "sum_r", "grp_abs", "n_spots", "q_excl", "pair_excl",
                 "max_excl", "reach_frac", "reach_abs", "version", "_pmfs", "_others", "_eval")

    def __init__(self, n_cells: int):
        self.g = np.zeros(n_cells)
        self.n = np.zeros(n_cells, dtype=np.int64)
        self.r = np.zeros(n_cells)
        self.sum_r = 0.0
        self.grp_abs = 0.0
        self.n_spots = 0
        self.q_excl = np.ones(n_cells)
        self.pair_excl = np.ones(n_cells)
        self.max_excl = np.zeros(n_cells)
        self.reach_frac = 0.0
        self.reach_abs = 0.0
        self.version = 0
        self._pmfs: dict = {}
        self._others: dict = {}
        self._eval = None

    def copy(self) -> "CurveState":
        return copy.deepcopy(self)


class CurveReachEngine(ReachEngine):
    """Per-cell reach curves R_c(g) combined across cells (ESTIMATE / CALIBRATED)."""

    def __init__(self, cfg: dict, params: dict, forecast: pd.DataFrame | None = None,
                 k_multiplier: float = 1.0):
        self._init_common(cfg, forecast)
        self.method = params["method"]
        self.params = params
        self.form = params["form"]
        self._fid = curves.form_id(self.form)
        self.k_multiplier = float(k_multiplier)
        cells = params["cells"]
        self.cell_keys = list(cells.keys())
        self.cell_index = {k: i for i, k in enumerate(self.cell_keys)}
        self.rmax = np.array([float(cells[k]["rmax_pct"]) for k in self.cell_keys])
        self.k = np.array([float(cells[k]["k"]) for k in self.cell_keys]) * self.k_multiplier
        self.granularity = params.get("cell_granularity", "channel")
        self.dayparts = cfg["reach"].get("estimate", {}).get("dayparts", {})
        self.phi = None
        dup = params.get("duplication")
        if cfg["reach"].get("duplication", "sainsbury") == "matrix":
            if not dup:
                raise ValueError("reach.duplication = matrix but the parameter file has no duplication matrix")
            order = dup["cells"]
            M = np.asarray(dup["phi"], float)
            idx = [order.index(k) for k in self.cell_keys]
            self.phi = M[np.ix_(idx, idx)]
        self._rmax_l = self.rmax.tolist()
        self._k_l = self.k.tolist()

    # ---- spots ---------------------------------------------------------------------------
    def cell_key(self, spot: Any) -> str:
        ch = _get(spot, "channel")
        if self.granularity == "channel":
            key = ch
        else:
            dp = _get(spot, "daypart")
            if dp is None or dp not in self.dayparts:
                dp = daypart_of(float(_get(spot, "start_min")), self.dayparts)
            key = f"{ch}|{dp}"
        if key not in self.cell_index:
            raise KeyError(f"no reach parameters for cell {key!r} (known: {self.cell_keys})")
        return key

    def prepare(self, spot: Any) -> PreparedSpot:
        if isinstance(spot, PreparedSpot):
            return spot
        a = self._aud(spot)
        return PreparedSpot(self.cell_index[self.cell_key(spot)], a / self.universe * 100.0, a,
                            _get(spot, "slot_id"))

    # ---- state -----------------------------------------------------------------------------
    def new_state(self) -> CurveState:
        return CurveState(len(self.cell_keys))

    def _r_of(self, c: int, g: float) -> float:
        return curves.curve_scalar(g, self._rmax_l[c], self._k_l[c], self._fid) / 100.0

    def _refresh(self, st: CurveState) -> None:
        r = st.r
        q = 1.0 - r
        C = len(r)
        # product of the other cells' (1 - r)
        pre = np.concatenate([[1.0], np.cumprod(q)[:-1]])
        suf = np.concatenate([np.cumprod(q[::-1])[::-1][1:], [1.0]])
        st.q_excl = pre * suf
        st.sum_r = float(r.sum())
        if self.phi is not None:
            F = np.ones((C, C))
            for i in range(C):
                for j in range(i + 1, C):
                    F[i, j] = F[j, i] = curves.pair_factor(r[i], r[j], self.phi[i, j])
            tot = np.prod(F[np.triu_indices(C, 1)])
            rowp = np.prod(F, axis=1)
            st.pair_excl = tot / np.where(rowp > 0, rowp, 1.0)
            for i in range(C):
                st.max_excl[i] = np.max(np.delete(r, i)) if C > 1 else 0.0
        frac = curves.combine_pairwise(r, self.phi)
        st.reach_frac = curves.cap_reach(frac, st.sum_r, st.grp_abs / self.universe)
        st.reach_abs = st.reach_frac * self.universe
        st.version += 1
        st._others = {}
        st._eval = None

    def add(self, st: CurveState, spot: Any) -> CurveState:
        p = self.prepare(spot)
        c = p.cell
        st.g[c] += p.g_pct
        st.n[c] += 1
        st.r[c] = self._r_of(c, st.g[c])
        st.grp_abs += p.aud_abs
        st.n_spots += 1
        st._pmfs.pop(c, None)
        self._refresh(st)
        return st

    def remove(self, st: CurveState, spot: Any) -> CurveState:
        p = self.prepare(spot)
        c = p.cell
        if st.n[c] <= 0:
            raise ValueError(f"cannot remove a spot from empty cell {self.cell_keys[c]!r}")
        st.n[c] -= 1
        st.g[c] = 0.0 if st.n[c] == 0 else max(st.g[c] - p.g_pct, 0.0)
        st.r[c] = self._r_of(c, st.g[c])
        st.grp_abs = max(st.grp_abs - p.aud_abs, 0.0) if st.n_spots > 1 else 0.0
        st.n_spots -= 1
        st._pmfs.pop(c, None)
        self._refresh(st)
        return st

    # ---- marginal ------------------------------------------------------------------------
    def _reach_frac_with(self, st: CurveState, c: int, r_new: float, grp_abs_new: float) -> float:
        if self.phi is None:
            frac = 1.0 - st.q_excl[c] * (1.0 - r_new)
        else:
            p_none = st.q_excl[c] * (1.0 - r_new) * st.pair_excl[c]
            ph, r = self.phi[c], st.r
            for j in range(len(r)):
                if j != c:
                    p_none *= curves.pair_factor(r_new, r[j], ph[j])
            sum_r = st.sum_r - st.r[c] + r_new
            lo = max(0.0, 1.0 - sum_r)
            hi = 1.0 - max(st.max_excl[c], r_new)
            frac = 1.0 - min(max(p_none, lo), hi)
        sum_r = st.sum_r - st.r[c] + r_new
        return curves.cap_reach(frac, sum_r, grp_abs_new / self.universe)

    def marginal_reach(self, st: CurveState, candidate: Any) -> float:
        """Incremental Reach 1+ (people) if ``candidate`` were added. O(1) under Sainsbury,
        O(#cells) with a duplication matrix. Does not modify the state."""
        p = candidate if isinstance(candidate, PreparedSpot) else self.prepare(candidate)
        c = p.cell
        r_new = self._r_of(c, st.g[c] + p.g_pct)
        frac = self._reach_frac_with(st, c, r_new, st.grp_abs + p.aud_abs)
        return frac * self.universe - st.reach_abs

    # ---- frequency / evaluate -----------------------------------------------------------
    def _cell_pmf(self, st: CurveState, c: int, g: float | None = None, n: int | None = None) -> np.ndarray:
        if g is None:
            pm = st._pmfs.get(c)
            if pm is None:
                pm = curves.cell_frequency_pmf(st.g[c], st.r[c] * 100.0, int(st.n[c]), self.kmax)
                st._pmfs[c] = pm
            return pm
        return curves.cell_frequency_pmf(g, self._r_of(c, g) * 100.0, n, self.kmax)

    def _others_pmf(self, st: CurveState, c: int) -> np.ndarray:
        pm = st._others.get(c)
        if pm is None:
            pm = np.zeros(self.kmax + 1)
            pm[0] = 1.0
            for j in range(len(self.cell_keys)):
                if j != c and st.n[j] > 0:
                    pm = curves.convolve_capped(pm, self._cell_pmf(st, j), self.kmax)
            st._others[c] = pm
        return pm

    def _nplus(self, reach_frac: float, pmf: np.ndarray) -> np.ndarray:
        nplus = curves.n_plus_from_pmf(pmf)
        nplus = reach_frac * nplus / nplus[0] if nplus[0] > 0 else np.zeros(self.kmax)
        nplus[0] = reach_frac
        return np.minimum.accumulate(nplus)

    def marginal_reach_n(self, st: CurveState, candidate: Any, n: int = 3) -> float:
        """Incremental Reach N+ (people) if ``candidate`` were added (O(K^2) + one NBD solve)."""
        if n == 1:
            return self.marginal_reach(st, candidate)
        if not 1 <= n <= self.kmax:
            raise ValueError(f"n must be in 1..{self.kmax}")
        p = candidate if isinstance(candidate, PreparedSpot) else self.prepare(candidate)
        c = p.cell
        cur = self._current_nplus(st)[n - 1]
        g_new = st.g[c] + p.g_pct
        r_new = self._r_of(c, g_new)
        frac = self._reach_frac_with(st, c, r_new, st.grp_abs + p.aud_abs)
        cp = curves.cell_frequency_pmf(g_new, r_new * 100.0, int(st.n[c]) + 1, self.kmax)
        pmf = curves.convolve_capped(self._others_pmf(st, c), cp, self.kmax)
        return float(self._nplus(frac, pmf)[n - 1] * self.universe - cur)

    def _current_nplus(self, st: CurveState) -> np.ndarray:
        if st._eval is None:
            active = [j for j in range(len(self.cell_keys)) if st.n[j] > 0]
            pmf = np.zeros(self.kmax + 1)
            pmf[0] = 1.0
            for j in active:
                pmf = curves.convolve_capped(pmf, self._cell_pmf(st, j), self.kmax)
            st._eval = self._nplus(st.reach_frac, pmf) * self.universe
        return st._eval

    def params_summary(self) -> dict:
        p = self.params
        return {
            "method": self.method, "form": self.form, "cell_granularity": self.granularity,
            "k_multiplier": self.k_multiplier,
            "duplication": "matrix" if self.phi is not None else "sainsbury",
            "cells": {k: {"rmax_pct": float(self.rmax[i]), "k": float(self.k[i]),
                          "initial_slope": float(self.rmax[i] / self.k[i])}
                      for i, k in enumerate(self.cell_keys)},
            "params_path": p.get("_path"), "generated_at": p.get("generated_at"),
            "assumptions": p.get("assumptions"),
            "universe_is_official": bool(self.cfg["target"].get("universe_is_official", False)),
            "note": ("ESTIMATE: model only, not validated against measured reach"
                     if self.method == "ESTIMATE" else "CALIBRATED: fitted to eTAM R&F test schedules"),
        }

    def evaluate(self, st: CurveState) -> dict:
        nplus = self._current_nplus(st)
        per_cell = {self.cell_keys[j]: {"grp_pct": float(st.g[j]), "n_spots": int(st.n[j]),
                                        "reach_pct": float(st.r[j] * 100.0)}
                    for j in range(len(self.cell_keys)) if st.n[j] > 0}
        return self._result(st.reach_abs, nplus, st.grp_abs, st.n_spots, self.params_summary(),
                            {"per_cell": per_cell})

    # ---- sensitivity ---------------------------------------------------------------------
    def with_k_multiplier(self, mult: float) -> "CurveReachEngine":
        return CurveReachEngine(self.cfg, self.params, k_multiplier=self.k_multiplier * mult)

    def sensitivity(self, schedule: pd.DataFrame | Iterable, k_multipliers: Iterable[float] | None = None) -> list[dict]:
        """Reach 1+/3+ of ``schedule`` with every cell's k multiplied by each multiplier
        (default [1-s, 1, 1+s], s = reach.estimate_sensitivity_k)."""
        if k_multipliers is None:
            s = float(self.cfg["reach"]["estimate_sensitivity_k"])
            k_multipliers = [1.0 - s, 1.0, 1.0 + s]
        recs = [self.prepare(r) for r in _records(schedule)]
        out = []
        for m in k_multipliers:
            res = self.with_k_multiplier(m).reach(recs)
            out.append({"k_multiplier": float(m), "method": res["method"],
                        "reach_1plus_abs": res["reach_1plus_abs"], "reach_1plus_pct": res["reach_1plus_pct"],
                        "reach_3plus_abs": res["reach_3plus_abs"], "reach_3plus_pct": res["reach_3plus_pct"],
                        "ots": res["ots"], "grp_pct": res["grp_pct"]})
        return out


# =============================================================================== EXACT
@dataclass(frozen=True)
class ExactSpot:
    uid: int
    channel: str
    date: pd.Timestamp
    start_min: float
    aud_abs: float | None
    slot_id: Any = None


class ExactState:
    __slots__ = ("counts", "reach_abs", "grp_abs", "grp_fc", "spots", "_eval")

    def __init__(self, n_persons: int):
        self.counts = np.zeros(n_persons, dtype=np.int64)
        self.reach_abs = 0.0
        self.grp_abs = 0.0
        self.grp_fc = 0.0
        self.spots: dict[int, ExactSpot] = {}
        self._eval = None

    def copy(self) -> "ExactState":
        return copy.deepcopy(self)


class ExactReachEngine(ReachEngine):
    """Respondent-level reach (Nielsen definitions via optimizer.metrics.nielsen)."""
    method = "EXACT"

    def __init__(self, cfg: dict, forecast: pd.DataFrame | None = None,
                 respondents: pd.DataFrame | str | None = None):
        from .exact import RespondentPanel, load_respondent_file
        self._init_common(cfg, forecast)
        rc = cfg["reach"]
        src = respondents if respondents is not None else rc["exact"]["respondent_path"]
        if src is None:
            raise FileNotFoundError("reach.mode = EXACT but reach.exact.respondent_path is not set "
                                    "(respondent-level data, DATA_SPEC C.3 option A)")
        df = load_respondent_file(src if isinstance(src, pd.DataFrame) else _abs(src))
        self.panel = RespondentPanel(df, rc["min_seconds"], nielsen.common_weight_rule_from_config(cfg),
                                     float(cfg["grid"]["spot_length_sec"]),
                                     rc["exact"].get("date_mapping", "identity"),
                                     pd.Timestamp(cfg["flight"]["start"]))
        if rc["exact"].get("universe_from_panel"):
            self.universe = self.panel.panel_universe()
        self._uid = 0

    def prepare(self, spot: Any) -> ExactSpot:
        if isinstance(spot, ExactSpot):
            return spot
        a = _get(spot, "aud_abs")
        a = None if a is None or (isinstance(a, float) and math.isnan(a)) else float(a)
        self._uid += 1
        return ExactSpot(self._uid, _get(spot, "channel"), pd.Timestamp(_get(spot, "air_date")).normalize(),
                         float(_get(spot, "start_min")), a, _get(spot, "slot_id"))

    def _exp(self, s: ExactSpot):
        return self.panel.spot_exposure(s.channel, s.date, s.start_min)

    def new_state(self) -> ExactState:
        return ExactState(len(self.panel.persons))

    def add(self, st: ExactState, spot: Any) -> ExactState:
        s = self.prepare(spot)
        if s.uid in st.spots:
            raise ValueError("spot already in state (prepare a new spot to buy the same airtime twice)")
        e = self._exp(s)
        w = self.panel.common_w
        new = e.viewers[st.counts[e.viewers] == 0]
        st.reach_abs += float(w[new].sum())
        st.counts[e.viewers] += 1
        st.grp_abs += float(e.viewer_daily_w.sum())
        st.grp_fc += s.aud_abs or 0.0
        st.spots[s.uid] = s
        st._eval = None
        return st

    def remove(self, st: ExactState, spot: Any) -> ExactState:
        s = spot if isinstance(spot, ExactSpot) else None
        if s is None:   # match an unprepared spot by channel/date/time
            key = (_get(spot, "channel"), pd.Timestamp(_get(spot, "air_date")).normalize(), float(_get(spot, "start_min")))
            s = next((x for x in st.spots.values() if (x.channel, x.date, x.start_min) == key), None)
        if s is None or s.uid not in st.spots:
            raise ValueError("spot not in state")
        e = self._exp(s)
        st.counts[e.viewers] -= 1
        gone = e.viewers[st.counts[e.viewers] == 0]
        st.reach_abs -= float(self.panel.common_w[gone].sum())
        st.grp_abs -= float(e.viewer_daily_w.sum())
        st.grp_fc -= s.aud_abs or 0.0
        del st.spots[s.uid]
        st._eval = None
        return st

    def marginal_reach(self, st: ExactState, candidate: Any) -> float:
        s = candidate if isinstance(candidate, ExactSpot) else self.prepare(candidate)
        e = self._exp(s)
        v = e.viewers
        return float(self.panel.common_w[v[st.counts[v] == 0]].sum())

    def marginal_reach_n(self, st: ExactState, candidate: Any, n: int = 3) -> float:
        s = candidate if isinstance(candidate, ExactSpot) else self.prepare(candidate)
        v = self._exp(s).viewers
        return float(self.panel.common_w[v[st.counts[v] == n - 1]].sum())

    def evaluate(self, st: ExactState) -> dict:
        """Recompute everything from the state's spots with the Nielsen functions."""
        panel = self.panel
        viewing = []
        per_spot_w = []
        for uid, s in st.spots.items():
            e = self._exp(s)
            viewing.extend((panel.persons[i], uid, sec) for i, sec in zip(e.persons, e.seconds))
            per_spot_w.append(list(e.viewer_daily_w))
        counts = nielsen.exposure_counts(viewing, panel.min_seconds)
        w = panel.common_w_map
        nplus = np.array([nielsen.reach_n_plus(w, counts, n) if counts else 0.0
                          for n in range(1, self.kmax + 1)])
        grp = nielsen.grp_abs(per_spot_w) if per_spot_w else 0.0
        freq = nielsen.frequency(w, counts) if counts and nplus[0] > 0 else None
        extra = {"grp_abs_forecast": st.grp_fc, "frequency": freq,
                 "grp_basis": "respondent daily weights (GL p38)"}
        return self._result(nplus[0], nplus, grp, len(st.spots), self.params_summary(), extra)

    def params_summary(self) -> dict:
        rc = self.cfg["reach"]
        return {"method": "EXACT", "min_seconds": self.panel.min_seconds,
                "common_weight_rule": self.panel.rule, "date_mapping": self.panel.date_mapping,
                "respondent_file": self.panel.df.attrs.get("source"),
                "n_panelists": int(len(self.panel.persons)), "n_days": len(self.panel.dates),
                "universe_from_panel": bool(rc["exact"].get("universe_from_panel"))}

    def sensitivity(self, schedule, k_multipliers=None) -> list[dict]:
        res = self.reach(schedule)
        return [{"k_multiplier": None, "method": "EXACT", "note": "no model parameter in EXACT mode",
                 "reach_1plus_abs": res["reach_1plus_abs"], "reach_1plus_pct": res["reach_1plus_pct"],
                 "reach_3plus_abs": res["reach_3plus_abs"], "reach_3plus_pct": res["reach_3plus_pct"],
                 "ots": res["ots"], "grp_pct": res["grp_pct"]}]
