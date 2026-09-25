"""verify_constraints: the separate, independent check of a finished plan (METHODOLOGY 4.2).

It recomputes everything from the plan's slot_ids joined to the candidate table (never from
the optimiser's internal state) with pandas group-bys and exact integer-cent arithmetic.
A plan is valid only if every check is PASS.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .problem import (build_candidates, flight_week, load_conflicts, n_flight_weeks,
                      share_fraction, to_cents)

FLOAT_TOL = 1e-9


def _ge_share(amount_c: int, share: float, basis_c: int) -> bool:
    """amount >= share x basis, exactly (share as a fraction)."""
    f = share_fraction(share)
    return amount_c * f.denominator >= f.numerator * basis_c


def _le_share(amount_c: int, share: float, basis_c: int) -> bool:
    f = share_fraction(share)
    return amount_c * f.denominator <= f.numerator * basis_c


def _chk(ok: bool, value: Any, bound: Any, offending: list | None = None, note: str | None = None) -> dict:
    d = {"status": "PASS" if ok else "FAIL", "value": value, "bound": bound, "offending": offending or []}
    if note:
        d["note"] = note
    return d


def verify_constraints(plan_df: pd.DataFrame, candidates: pd.DataFrame | None, cfg: dict, *,
                       conflicts: pd.DataFrame | None = None, min_impressions: float | None = None) -> dict:
    """Check ``plan_df`` (one row per spot, needs ``slot_id``) against every constraint in ``cfg``.

    candidates : slot-level candidate table from :func:`build_candidates` (built from cfg if None).
    min_impressions : scenario-specific impressions floor (S4), checked when > 0.
    Returns {"status": PASS|FAIL, "checks": {name: {status, value, bound, offending}}, "failed": [...]}.
    """
    if candidates is None:
        candidates, _ = build_candidates(cfg)
    b = cfg["budget"]
    caps = cfg["caps"]
    cons = cfg.get("constraints", {}) or {}
    pc = cfg.get("plan", {}) or {}
    budget_c = to_cents(b["total_usd"])
    audc = pc.get("audience_column", "aud_abs_p50")
    checks: dict[str, dict] = {}

    ids = plan_df["slot_id"].astype(str).tolist()
    cand = candidates.set_index("slot_id", drop=False)
    unknown = [s for s in ids if s not in cand.index]
    checks["slots_are_candidates"] = _chk(not unknown, len(ids) - len(unknown), len(ids), unknown,
                                          "every spot is an in-flight, non-excluded candidate slot")
    P = cand.loc[[s for s in ids if s in cand.index]].reset_index(drop=True)
    P["air_date"] = pd.to_datetime(P["air_date"]).dt.normalize()
    cost_c = to_cents(P["rate_usd"]) if len(P) else np.zeros(0, np.int64)
    P["cost_c"] = cost_c

    # data consistency: plan's own rate / audience columns match the candidate table
    bad = []
    for col in ("rate_usd", audc):
        if col in plan_df.columns and len(P) == len(plan_df):
            diff = ~np.isclose(plan_df[col].astype(float).to_numpy(), P[col].astype(float).to_numpy(), rtol=0, atol=1e-6)
            bad += [f"{plan_df['slot_id'].iloc[i]}:{col}" for i in np.flatnonzero(diff)]
    checks["plan_matches_candidates"] = _chk(not bad, None, None, bad)

    # flight + exclusions
    start, end = pd.Timestamp(cfg["flight"]["start"]), pd.Timestamp(cfg["flight"]["end"])
    out = P.loc[~P["air_date"].between(start, end), "slot_id"].tolist()
    checks["in_flight"] = _chk(not out, f"{start.date()}..{end.date()}", None, out)
    ex_prog = {str(x).strip().upper() for x in (cons.get("excluded_programs") or [])}
    ex_slot = {str(x) for x in (cons.get("excluded_slots") or [])}
    # checked on the plan's own rows (an excluded slot is also not a candidate)
    src = plan_df if "title_en" in plan_df.columns else P
    title_u = src["title_en"].astype(str).str.strip().str.upper()
    pname_u = src["program_name"].astype(str).str.strip().str.upper() if "program_name" in src else title_u
    exc = src.loc[(title_u.isin(ex_prog) | pname_u.isin(ex_prog) | src["slot_id"].astype(str).isin(ex_slot)).to_numpy(),
                  "slot_id"].tolist()
    checks["exclusions"] = _chk(not exc, None, {"programs": sorted(ex_prog), "slots": sorted(ex_slot)}, exc)

    # budget
    spend_c = int(P["cost_c"].sum())
    checks["budget_max"] = _chk(spend_c <= budget_c, spend_c / 100, budget_c / 100,
                                [{"excess_usd": (spend_c - budget_c) / 100}] if spend_c > budget_c else [])
    minu = float(b["min_utilisation"])
    min_c = int(math.ceil(budget_c * float(share_fraction(minu)) - 1e-9))
    checks["budget_min_utilisation"] = _chk(spend_c >= min_c, spend_c / 100, min_c / 100,
                                            [{"shortfall_usd": (min_c - spend_c) / 100}] if spend_c < min_c else [],
                                            note=f"spend >= {minu} x budget")

    # per-slot cap
    cap = int(caps.get("max_spots_per_slot", 1))
    cnt = P["slot_id"].value_counts()
    over = cnt[cnt > cap]
    checks["per_slot_cap"] = _chk(over.empty, int(cnt.max()) if len(cnt) else 0, cap,
                                  [{"slot_id": k, "spots": int(v)} for k, v in over.items()])

    # conflicts
    cf = load_conflicts(cfg, conflicts)
    bought = set(P["slot_id"])
    both = [{"slot_a": a, "slot_b": bb} for a, bb in zip(cf["slot_a"], cf["slot_b"]) if a in bought and bb in bought]
    checks["conflict_pairs"] = _chk(not both, len(both), 0, both, "overlapping slots: never both bought")

    # channel shares
    ch_basis = pc.get("channel_share_basis", "budget")
    basis_ch = budget_c if ch_basis == "budget" else spend_c
    ch_sp = P.groupby("channel")["cost_c"].sum()
    mins = cons.get("channel_min_share") or {}
    offend = []
    vals = {}
    for c, s in mins.items():
        v = int(ch_sp.get(c, 0))
        vals[c] = v / basis_ch if basis_ch else None
        if not _ge_share(v, float(s), basis_ch):
            offend.append({"channel": c, "spend_usd": v / 100, "share": vals[c], "min_share": float(s)})
    checks["channel_min_share"] = _chk(not offend, vals, dict(mins), offend,
                                       f"share of {ch_basis}; every channel >= its minimum (use all channels)")
    maxs = cons.get("channel_max_share") or {}
    offend = []
    for c, s in maxs.items():
        if s is None:
            continue
        v = int(ch_sp.get(c, 0))
        if not _le_share(v, float(s), basis_ch):
            offend.append({"channel": c, "spend_usd": v / 100, "share": v / basis_ch if basis_ch else None, "max_share": float(s)})
    checks["channel_max_share"] = _chk(not offend, {c: int(ch_sp.get(c, 0)) / basis_ch if basis_ch else None for c in maxs},
                                       dict(maxs), offend, f"share of {ch_basis}")

    # spots per channel per day (broadcast day = air_date)
    cd = P.groupby(["channel", "air_date"]).size()
    cdcap = caps.get("max_spots_per_channel_per_day") or {}
    offend = [{"channel": c, "air_date": str(d.date()), "spots": int(v), "cap": int(cdcap[c])}
              for (c, d), v in cd.items() if cdcap.get(c) is not None and v > int(cdcap[c])]
    checks["max_spots_per_channel_per_day"] = _chk(not offend, int(cd.max()) if len(cd) else 0, dict(cdcap), offend)

    # spots per programme per day
    pcap = caps.get("max_spots_per_program_per_day")
    scope = pc.get("program_cap_scope", "channel")
    keys = (["channel"] if scope == "channel" else []) + ["title_u", "air_date"]
    P["title_u"] = P["title_en"].astype(str).str.strip().str.upper()
    pd_cnt = P.groupby(keys).size()
    offend = []
    if pcap is not None:
        for k, v in pd_cnt.items():
            if v > int(pcap):
                kk = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
                kk["air_date"] = str(pd.Timestamp(kk["air_date"]).date())
                offend.append({**kk, "spots": int(v), "cap": int(pcap)})
    checks["max_spots_per_program_per_day"] = _chk(not offend, int(pd_cnt.max()) if len(pd_cnt) else 0, pcap, offend,
                                                   f"scope={scope} (title_en per broadcast day)")

    # weekly phasing
    W = n_flight_weeks(cfg)
    wk = flight_week(P["air_date"], cfg) if len(P) else pd.Series([], dtype=int)
    wk_sp = P["cost_c"].groupby(wk.to_numpy()).sum() if len(P) else pd.Series(dtype=np.int64)
    wmin = float(cfg["flight"].get("weekly_share_min", 0.0) or 0.0)
    wmax = float(cfg["flight"].get("weekly_share_max", 1.0) or 1.0)
    wb = pc.get("weekly_share_basis", "spend")
    basis_w = spend_c if wb == "spend" else budget_c
    offend, vals = [], {}
    for w in range(1, W + 1):
        v = int(wk_sp.get(w, 0))
        vals[w] = v / basis_w if basis_w else None
        if not (_ge_share(v, wmin, basis_w) and _le_share(v, wmax, basis_w)):
            offend.append({"week": w, "spend_usd": v / 100, "share": vals[w]})
    checks["weekly_phasing"] = _chk(not offend, vals, [wmin, wmax], offend,
                                    f"share of {wb}; week w = flight.start + 7(w-1) .. +6")

    # TRP floor
    trp = float(P["rating_pct_p50"].sum()) if len(P) else 0.0
    ftrp = float(cons.get("min_trp_pct", 0.0) or 0.0)
    if ftrp > 0:
        checks["min_trp_pct"] = _chk(trp >= ftrp * (1 - FLOAT_TOL), trp, ftrp, note="TRP % = sum(rating_pct_p50)")
    if min_impressions:
        imp = float(P[audc].sum()) if len(P) else 0.0
        checks["min_impressions"] = _chk(imp >= float(min_impressions) * (1 - FLOAT_TOL), imp, float(min_impressions))

    failed = [k for k, v in checks.items() if v["status"] != "PASS"]
    return {"status": "PASS" if not failed else "FAIL", "failed": failed, "checks": checks,
            "n_spots": int(len(plan_df)), "spend_usd": spend_c / 100}
