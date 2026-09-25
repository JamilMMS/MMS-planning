"""Candidate set and the integer-indexed optimisation problem.

Every parameter comes from the config:

* candidates = in-flight slots (``flight.start`` .. ``flight.end`` on the broadcast-day
  ``air_date``) of ``plan.forecast_path`` minus ``constraints.excluded_programs`` (title_en or
  program_name, case-insensitive) and ``constraints.excluded_slots`` (slot_id);
* audience = ``plan.audience_column`` (aud_abs_p50); cost = ``rate_usd`` held internally in
  integer US cents (round half-up), ``rate_aed`` carried alongside;
* each slot is expanded into ``caps.max_spots_per_slot`` purchasable *units* (copy 0..cap-1);
* flight weeks are 7-day blocks from ``flight.start`` (week 1 = start .. start+6).

Money is integer cents throughout the search and the checks, so budget / share constraints
never suffer float round-off.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import project_root

SHARE_DENOM_LIMIT = 100000


def abs_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else project_root() / p


def to_cents(usd: Any) -> np.ndarray | int:
    """USD -> integer cents, round half-up (rounding.display = round_half_up)."""
    a = np.floor(np.asarray(usd, dtype=float) * 100.0 + 0.5).astype(np.int64)
    return int(a) if a.ndim == 0 else a


def share_fraction(x: float) -> Fraction:
    return Fraction(str(x)).limit_denominator(SHARE_DENOM_LIMIT)


def with_overrides(cfg: dict, *, budget_total_usd: float | None = None,
                   min_trp_pct: float | None = None) -> dict:
    """Deep copy of ``cfg`` with a different budget (frontier) and/or TRP floor."""
    c = copy.deepcopy(cfg)
    if budget_total_usd is not None:
        c["budget"]["total_usd"] = float(budget_total_usd)
    if min_trp_pct is not None:
        c["constraints"]["min_trp_pct"] = float(min_trp_pct)
    return c


# ----------------------------------------------------------------------------- candidates
def flight_week(air_date: pd.Series, cfg: dict) -> pd.Series:
    start = pd.Timestamp(cfg["flight"]["start"])
    return ((pd.to_datetime(air_date).dt.normalize() - start).dt.days // 7 + 1).astype(int)


def n_flight_weeks(cfg: dict) -> int:
    start, end = pd.Timestamp(cfg["flight"]["start"]), pd.Timestamp(cfg["flight"]["end"])
    return int((end - start).days // 7 + 1)


def flags_of(df: pd.DataFrame) -> pd.Series:
    """One readable flags string per slot from the forecast's flag columns + assumption tags."""
    cols = [c for c in ("is_synthetic", "is_new_program", "is_live", "live_special", "low_sample",
                        "high_uncertainty", "has_conflict") if c in df.columns]
    parts = []
    for _, r in df.iterrows():
        f = [c for c in cols if bool(r[c])]
        a = r.get("assumption")
        if isinstance(a, str) and a.strip():
            f += [t for t in a.split(";") if t and t not in f]
        parts.append(";".join(f))
    return pd.Series(parts, index=df.index, dtype=object)


def build_candidates(cfg: dict, forecast: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Slot-level candidate table (one row per purchasable slot) + a log of what was dropped."""
    pc = cfg["plan"]
    fc = forecast if forecast is not None else pd.read_parquet(abs_path(pc["forecast_path"]))
    fc = fc.copy()
    fc["air_date"] = pd.to_datetime(fc["air_date"]).dt.normalize()
    audc = pc.get("audience_column", "aud_abs_p50")
    log: dict[str, Any] = {"n_forecast_rows": int(len(fc))}
    start, end = pd.Timestamp(cfg["flight"]["start"]), pd.Timestamp(cfg["flight"]["end"])
    inflight = fc["air_date"].between(start, end)
    log["dropped_out_of_flight"] = fc.loc[~inflight, "slot_id"].tolist()
    fc = fc[inflight]
    cons = cfg.get("constraints", {})
    ex_prog = {str(x).strip().upper() for x in (cons.get("excluded_programs") or [])}
    ex_slot = {str(x) for x in (cons.get("excluded_slots") or [])}
    title_u = fc["title_en"].astype(str).str.strip().str.upper()
    pname_u = fc["program_name"].astype(str).str.strip().str.upper() if "program_name" in fc else title_u
    excl = title_u.isin(ex_prog) | pname_u.isin(ex_prog) | fc["slot_id"].astype(str).isin(ex_slot)
    log["dropped_excluded"] = fc.loc[excl, "slot_id"].tolist()
    fc = fc[~excl]
    bad = fc[audc].isna() | fc["rate_usd"].isna() | (fc["rate_usd"] <= 0) | (fc[audc] < 0)
    log["dropped_missing_aud_or_rate"] = fc.loc[bad, "slot_id"].tolist()
    fc = fc[~bad]
    if not fc["slot_id"].is_unique:
        raise ValueError("slot_id is not unique in the forecast")
    fc = fc.sort_values(["air_date", "channel", "start_min", "slot_id"], kind="mergesort").reset_index(drop=True)
    fc["aud"] = fc[audc].astype(float)
    fc["cost_cents"] = to_cents(fc["rate_usd"])
    fc["week"] = flight_week(fc["air_date"], cfg)
    if "flags" not in fc.columns:
        fc["flags"] = flags_of(fc)
    log["n_candidates"] = int(len(fc))
    return fc, log


def load_conflicts(cfg: dict, conflicts: pd.DataFrame | None = None) -> pd.DataFrame:
    if conflicts is not None:
        return conflicts
    p = abs_path(cfg["plan"]["conflicts_path"])
    if not p.exists():
        return pd.DataFrame(columns=["slot_a", "slot_b"])
    return pd.read_parquet(p)


# ----------------------------------------------------------------------------- limits
@dataclass
class Limits:
    """All bounds of one scenario (money in integer cents)."""
    budget: int
    min_spend: int
    slot_cap: int
    ch_min_share: np.ndarray          # (C,) floats
    ch_max_share: np.ndarray          # (C,) floats (inf = none)
    ch_basis: str                     # budget | spend
    wk_min_share: float
    wk_max_share: float
    wk_basis: str
    chday_cap: dict                   # channel -> int | None
    prog_cap: int | None
    prog_scope: str
    min_trp_pct: float = 0.0
    min_impressions: float = 0.0


def limits_from_config(cfg: dict, channels: list[str], *, min_impressions: float = 0.0) -> Limits:
    b = cfg["budget"]
    budget = to_cents(b["total_usd"])
    min_spend = int(math.ceil(budget * float(share_fraction(b["min_utilisation"])) - 1e-9))
    cons = cfg.get("constraints", {})
    mins, maxs = cons.get("channel_min_share") or {}, cons.get("channel_max_share") or {}
    pc = cfg.get("plan", {})
    caps = cfg["caps"]
    return Limits(
        budget=budget, min_spend=min_spend, slot_cap=int(caps.get("max_spots_per_slot", 1)),
        ch_min_share=np.array([float(mins.get(c, 0.0)) for c in channels]),
        ch_max_share=np.array([float(maxs[c]) if c in maxs and maxs[c] is not None else np.inf for c in channels]),
        ch_basis=pc.get("channel_share_basis", "budget"),
        wk_min_share=float(cfg["flight"].get("weekly_share_min", 0.0) or 0.0),
        wk_max_share=float(cfg["flight"].get("weekly_share_max", 1.0) or 1.0),
        wk_basis=pc.get("weekly_share_basis", "spend"),
        chday_cap={c: (int(v) if v is not None else None) for c, v in (caps.get("max_spots_per_channel_per_day") or {}).items()},
        prog_cap=(int(caps["max_spots_per_program_per_day"]) if caps.get("max_spots_per_program_per_day") is not None else None),
        prog_scope=pc.get("program_cap_scope", "channel"),
        min_trp_pct=float(cons.get("min_trp_pct", 0.0) or 0.0),
        min_impressions=float(min_impressions or 0.0),
    )


# ----------------------------------------------------------------------------- problem
@dataclass
class Problem:
    """Integer-indexed purchasable units (slot copies) with group memberships."""
    cfg: dict
    slots: pd.DataFrame               # slot-level candidate table
    units: pd.DataFrame               # one row per unit: slot row index, copy
    lim: Limits
    channels: list[str]
    n: int
    slot_of: np.ndarray               # unit -> slot row
    ch: np.ndarray
    wk: np.ndarray                    # 0-based week index
    W: int
    chday: np.ndarray                 # unit -> (channel, day) group
    chday_cap: np.ndarray             # group -> cap (large if none)
    prog: np.ndarray                  # unit -> (programme, day) group
    n_prog: int
    cost: np.ndarray                  # int64 cents
    aud: np.ndarray                   # float persons
    trp: np.ndarray                   # rating % (TRP contribution)
    partners: list                    # unit -> np.ndarray of conflicting units
    conflict_pairs: np.ndarray        # (m, 2) unit pairs
    log: dict = field(default_factory=dict)

    @property
    def C(self) -> int:
        return len(self.channels)

    def unit_frame(self, units: np.ndarray) -> pd.DataFrame:
        """Slot rows of the given units (one row per spot), in unit order."""
        units = np.sort(np.asarray(units, dtype=int))
        df = self.slots.iloc[self.slot_of[units]].copy()
        df.insert(0, "spot_copy", self.units["copy"].to_numpy()[units])
        return df.reset_index(drop=True)


def build_problem(cfg: dict, forecast: pd.DataFrame | None = None, conflicts: pd.DataFrame | None = None,
                  *, min_impressions: float = 0.0, slots: pd.DataFrame | None = None,
                  cand_log: dict | None = None) -> Problem:
    if slots is None:
        slots, cand_log = build_candidates(cfg, forecast)
    cand_log = dict(cand_log or {})
    channels = list(dict.fromkeys(list((cfg.get("constraints", {}).get("channel_min_share") or {}).keys())
                                  + sorted(slots["channel"].unique())))
    channels = [c for c in channels if c in set(slots["channel"])] + \
        [c for c in channels if c not in set(slots["channel"])]
    lim = limits_from_config(cfg, channels, min_impressions=min_impressions)
    cap = lim.slot_cap
    S = len(slots)
    slot_of = np.repeat(np.arange(S), cap)
    copy_ix = np.tile(np.arange(cap), S)
    units = pd.DataFrame({"slot": slot_of, "copy": copy_ix})
    ch_ix = {c: i for i, c in enumerate(channels)}
    s_ch = slots["channel"].map(ch_ix).to_numpy()
    W = n_flight_weeks(cfg)
    s_wk = slots["week"].to_numpy() - 1
    day = slots["air_date"].dt.strftime("%Y-%m-%d")
    chday_key = slots["channel"] + "|" + day
    chday_codes, chday_uni = pd.factorize(chday_key, sort=True)
    big = 10 ** 9
    chday_cap = np.array([lim.chday_cap.get(k.split("|")[0]) if lim.chday_cap.get(k.split("|")[0]) is not None else big
                          for k in chday_uni], dtype=np.int64)
    title = slots["title_en"].astype(str).str.strip().str.upper()
    prog_key = (slots["channel"] + "|" if lim.prog_scope == "channel" else "") + title + "|" + day
    prog_codes, prog_uni = pd.factorize(prog_key, sort=True)
    # conflicts: slot pairs -> unit pairs (every copy of a with every copy of b)
    sid_ix = {s: i for i, s in enumerate(slots["slot_id"])}
    cf = load_conflicts(cfg, conflicts)
    pairs = []
    for a, b in zip(cf["slot_a"], cf["slot_b"]):
        if a in sid_ix and b in sid_ix and a != b:
            for ka in range(cap):
                for kb in range(cap):
                    pairs.append((sid_ix[a] * cap + ka, sid_ix[b] * cap + kb))
    pairs_arr = np.array(sorted(set(pairs)), dtype=np.int64).reshape(-1, 2)
    n = S * cap
    part: list[list[int]] = [[] for _ in range(n)]
    for a, b in pairs_arr:
        part[a].append(int(b))
        part[b].append(int(a))
    cand_log["n_units"] = n
    cand_log["n_conflict_pairs_units"] = int(len(pairs_arr))
    return Problem(
        cfg=cfg, slots=slots, units=units, lim=lim, channels=channels, n=n, slot_of=slot_of,
        ch=s_ch[slot_of].astype(np.int64), wk=s_wk[slot_of].astype(np.int64), W=W,
        chday=chday_codes[slot_of].astype(np.int64), chday_cap=chday_cap,
        prog=prog_codes[slot_of].astype(np.int64), n_prog=len(prog_uni),
        cost=slots["cost_cents"].to_numpy(np.int64)[slot_of],
        aud=slots["aud"].to_numpy(float)[slot_of],
        trp=slots["rating_pct_p50"].to_numpy(float)[slot_of],
        partners=[np.array(sorted(p), dtype=np.int64) for p in part], conflict_pairs=pairs_arr, log=cand_log,
    )
