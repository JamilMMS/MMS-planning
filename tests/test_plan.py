"""Optimizer (src/optimizer/plan) on tiny synthetic candidate sets with known optima."""
from __future__ import annotations

import copy
import itertools

import numpy as np
import pandas as pd
import pytest

from optimizer.config import load_config
from optimizer.plan.constraints import verify_constraints
from optimizer.plan.milp import solve_max_impressions
from optimizer.plan.problem import build_problem
from optimizer.plan.run import Runner
from optimizer.plan.search import Objective, corrected_greedy, local_search, solve_reach
from optimizer.reach.engine import ReachEngine

U = 100_000_000.0   # large universe: reach ~ additive at these audiences


def make_cfg(*, budget=200.0, min_util=0.0, start="2026-10-04", end="2026-10-10", wmin=0.0, wmax=1.0,
             ch_min=None, ch_max=None, chday=None, prog_cap=None, channels=("A", "B", "C")):
    cfg = copy.deepcopy(load_config())
    cfg["budget"].update(total_usd=budget, min_utilisation=min_util)
    cfg["flight"].update(start=start, end=end, weekly_share_min=wmin, weekly_share_max=wmax)
    cfg["target"]["universe"] = U
    cfg["reach"]["mode"] = "ESTIMATE"
    cfg["caps"] = {"max_spots_per_slot": 1,
                   "max_spots_per_channel_per_day": chday or {c: 50 for c in channels},
                   "max_spots_per_program_per_day": prog_cap}
    cfg["constraints"] = {"min_trp_pct": 0, "channel_min_share": ch_min or {c: 0.0 for c in channels},
                          "channel_max_share": ch_max or {}, "excluded_programs": [], "excluded_slots": []}
    cfg["plan"].update(milp_num_workers=1, milp_time_limit_sec=20, milp_deterministic_time=10,
                       local_search_max_iters=500)
    cfg["_config_sha256"] = "test"
    return cfg


def make_engine(cfg, channels=("A", "B", "C")):
    params = {"method": "ESTIMATE", "form": "hyperbolic", "cell_granularity": "channel",
              "cells": {c: {"rmax_pct": 60.0, "k": 60.0} for c in channels}}
    return ReachEngine.from_config(cfg, params=params)


def slots(rows):
    """rows: (channel, date, start_min, cost_usd, aud[, title])"""
    out = []
    for i, r in enumerate(rows):
        ch, d, sm, cost, aud = r[:5]
        title = r[5] if len(r) > 5 else f"P{i}"
        out.append({"slot_id": f"{ch}|{d}|{i:04d}", "channel": ch, "air_date": pd.Timestamp(d), "weekday": pd.Timestamp(d).day_name()[:3],
                    "start_min": float(sm), "end_min": float(sm) + 30, "tier": "Regular", "rate_usd": float(cost),
                    "rate_aed": float(cost) * 3.6725, "title_en": title, "program_name": title,
                    "aud_abs_p10": aud * 0.8, "aud_abs_p50": float(aud), "aud_abs_p90": aud * 1.2,
                    "rating_pct_p50": aud / U * 100, "forecast_level": "program", "high_uncertainty": False})
    return pd.DataFrame(out)


NO_CONF = pd.DataFrame(columns=["slot_a", "slot_b"])


def plan_of(prob, units):
    return prob.unit_frame(units)


def brute_force_best(fc, budget, key="aud_abs_p50", forbid=()):
    best, arg = -1, None
    for k in range(1, len(fc) + 1):
        for comb in itertools.combinations(range(len(fc)), k):
            if fc["rate_usd"].iloc[list(comb)].sum() > budget + 1e-9:
                continue
            ids = set(fc["slot_id"].iloc[list(comb)])
            if any(a in ids and b in ids for a, b in forbid):
                continue
            v = fc[key].iloc[list(comb)].sum()
            if v > best:
                best, arg = v, ids
    return best, arg


# --------------------------------------------------------------------------------- S3 MILP
def test_s3_picks_max_impression_pair():
    fc = slots([("A", "2026-10-04", 600, 100, 50_000), ("A", "2026-10-05", 600, 100, 80_000),
                ("B", "2026-10-05", 700, 100, 30_000), ("B", "2026-10-06", 600, 100, 90_000),
                ("C", "2026-10-07", 600, 100, 70_000)])
    cfg = make_cfg(budget=200, min_util=0.98)
    prob = build_problem(cfg, fc, NO_CONF)
    r = solve_max_impressions(prob, cfg)
    assert r["status"] == "OPTIMAL"
    got = set(prob.slots["slot_id"].iloc[prob.slot_of[r["units"]]])
    assert got == {fc.slot_id[1], fc.slot_id[3]}
    assert verify_constraints(plan_of(prob, r["units"]), prob.slots, cfg, conflicts=NO_CONF)["status"] == "PASS"


def test_s3_matches_brute_force_knapsack():
    rng = np.random.default_rng(7)
    rows = [(("A", "B", "C")[i % 3], f"2026-10-0{4 + i % 5}", 600 + 30 * i, int(rng.integers(50, 200)), int(rng.integers(10_000, 90_000)))
            for i in range(10)]
    fc = slots(rows)
    cfg = make_cfg(budget=400)
    prob = build_problem(cfg, fc, NO_CONF)
    r = solve_max_impressions(prob, cfg)
    best, _ = brute_force_best(prob.slots, 400)
    assert r["status"] == "OPTIMAL"
    assert prob.aud[r["units"]].sum() == pytest.approx(best)


# --------------------------------------------------------------------------------- conflicts
def test_conflict_pair_never_both_bought():
    fc = slots([("A", "2026-10-04", 600, 100, 90_000), ("A", "2026-10-04", 615, 100, 85_000),
                ("B", "2026-10-05", 600, 100, 40_000), ("C", "2026-10-06", 600, 100, 30_000)])
    conf = pd.DataFrame({"slot_a": [fc.slot_id[0]], "slot_b": [fc.slot_id[1]]})
    cfg = make_cfg(budget=200)
    prob = build_problem(cfg, fc, conf)
    r = solve_max_impressions(prob, cfg)
    ids = set(prob.slots["slot_id"].iloc[prob.slot_of[r["units"]]])
    assert not {fc.slot_id[0], fc.slot_id[1]} <= ids
    assert ids == {fc.slot_id[0], fc.slot_id[2]}
    eng = make_engine(cfg)
    run = Runner(cfg, forecast=fc, conflicts=conf, engine=eng, write=False, log=lambda *_: None)
    for name in ("max_reach_1plus", "max_reach_3plus", "max_impressions", "balanced"):
        out = run.scenario(name)
        ids = set(out["plan"]["slot_id"])
        assert not {fc.slot_id[0], fc.slot_id[1]} <= ids, name
        assert out["constraint_report"]["checks"]["conflict_pairs"]["status"] == "PASS"


# --------------------------------------------------------------------------------- channel minimum
def test_channel_minimum_forces_low_efficiency_channel():
    rows = [("A", f"2026-10-0{4 + d}", 600 + 60 * k, 100, 80_000) for d in range(4) for k in range(3)]
    rows += [("C", "2026-10-05", 600, 100, 1_000), ("C", "2026-10-06", 600, 100, 900)]   # terrible CPM
    fc = slots(rows)
    base = make_cfg(budget=500, channels=("A", "C"))
    eng = make_engine(base, channels=("A", "C"))
    free = Runner(base, forecast=fc, conflicts=NO_CONF, engine=eng, write=False, log=lambda *_: None)
    for name in ("max_reach_1plus", "max_impressions"):
        assert "C" not in set(free.scenario(name)["plan"]["channel"])
    cfg = make_cfg(budget=500, channels=("A", "C"), ch_min={"A": 0.0, "C": 0.2})
    run = Runner(cfg, forecast=fc, conflicts=NO_CONF, engine=eng, write=False, log=lambda *_: None)
    for name in ("max_reach_1plus", "max_reach_3plus", "max_impressions"):
        out = run.scenario(name)
        spend_c = out["plan"].loc[out["plan"].channel == "C", "rate_usd"].sum()
        assert spend_c >= 0.2 * 500 - 1e-9, name
        assert out["constraint_report"]["status"] == "PASS", (name, out["constraint_report"]["failed"])


# --------------------------------------------------------------------------------- weekly phasing
def test_weekly_phasing_respected():
    rows = [("A", f"2026-10-0{4 + d}", 600 + 60 * k, 100, 90_000) for d in range(3) for k in range(4)]     # week 1: great
    rows += [("B", f"2026-10-{11 + d}", 600 + 60 * k, 100, 20_000) for d in range(3) for k in range(4)]    # week 2: poor
    fc = slots(rows)
    cfg = make_cfg(budget=1000, min_util=0.9, end="2026-10-17", wmin=0.4, wmax=0.6)
    eng = make_engine(cfg)
    run = Runner(cfg, forecast=fc, conflicts=NO_CONF, engine=eng, write=False, log=lambda *_: None)
    for name in ("max_reach_1plus", "max_reach_3plus", "max_impressions", "balanced"):
        out = run.scenario(name)
        p = out["plan"]
        sp = p["rate_usd"].sum()
        w2 = p.loc[p.week == 2, "rate_usd"].sum() / sp
        assert 0.4 - 1e-9 <= w2 <= 0.6 + 1e-9, (name, w2)
        assert out["constraint_report"]["checks"]["weekly_phasing"]["status"] == "PASS"
        assert out["constraint_report"]["status"] == "PASS", (name, out["constraint_report"]["failed"])


# --------------------------------------------------------------------------------- verify
def test_verify_flags_hand_built_violations():
    fc = slots([("A", "2026-10-04", 600, 100, 50_000, "SHOW"), ("A", "2026-10-04", 660, 100, 50_000, "SHOW"),
                ("A", "2026-10-04", 720, 100, 50_000, "SHOW"), ("B", "2026-10-05", 600, 100, 10_000),
                ("C", "2026-10-06", 600, 100, 10_000)])
    conf = pd.DataFrame({"slot_a": [fc.slot_id[0]], "slot_b": [fc.slot_id[1]]})
    cfg = make_cfg(budget=250, min_util=0.9, chday={"A": 2, "B": 5, "C": 5}, prog_cap=2,
                   ch_min={"A": 0.0, "B": 0.0, "C": 0.2}, wmin=0.0, wmax=0.5)
    cfg["constraints"]["excluded_slots"] = [fc.slot_id[3]]
    prob = build_problem(cfg, fc, conf)
    bad = pd.concat([fc.iloc[[0, 1, 2, 0]], fc.iloc[[3]]])        # dup slot, conflict, caps, over budget, excluded
    rep = verify_constraints(bad, prob.slots, cfg, conflicts=conf)
    assert rep["status"] == "FAIL"
    for k in ("budget_max", "per_slot_cap", "conflict_pairs", "max_spots_per_channel_per_day",
              "max_spots_per_program_per_day", "channel_min_share", "slots_are_candidates", "weekly_phasing",
              "exclusions"):
        assert rep["checks"][k]["status"] == "FAIL", k
        assert rep["checks"][k]["offending"], k
    under = fc.iloc[[4]]
    rep = verify_constraints(under, prob.slots, cfg, conflicts=conf)
    assert rep["checks"]["budget_min_utilisation"]["status"] == "FAIL"
    cfg2 = copy.deepcopy(cfg)
    cfg2["constraints"]["min_trp_pct"] = 1.0
    assert verify_constraints(under, prob.slots, cfg2, conflicts=conf)["checks"]["min_trp_pct"]["status"] == "FAIL"


# --------------------------------------------------------------------------------- local search
def test_local_search_escapes_greedy_trap():
    # ratio greedy takes A (best ratio) then B; C no longer fits. Optimum B + C (swap A -> C).
    fc = slots([("A", "2026-10-04", 600, 60, 70_000), ("B", "2026-10-05", 600, 150, 170_000),
                ("C", "2026-10-06", 600, 150, 170_000)])
    cfg = make_cfg(budget=300)
    eng = make_engine(cfg)
    prob = build_problem(cfg, fc, NO_CONF)
    P = eng.prepare_many(prob.unit_frame(np.arange(prob.n)).assign(aud_abs=lambda d: d.aud))
    obj = Objective("reach1", eng)
    g, _ = corrected_greedy(prob, eng, P, obj, 0.0, 0)
    vg = obj.value(g.est)
    gset = set(prob.slots.slot_id.iloc[prob.slot_of[g.units]])
    assert fc.slot_id[0] in gset and len(gset) == 2          # trapped: A + one of B/C
    s, st = local_search(g.copy(), obj, cfg)
    assert obj.value(s.est) > vg + 1000
    assert set(prob.slots.slot_id.iloc[prob.slot_of[s.units]]) == {fc.slot_id[1], fc.slot_id[2]}
    assert st["moves"]["swap"] + st["moves"]["drop_add"] >= 1


def _random_instance(seed=3, n=60):
    rng = np.random.default_rng(seed)
    rows = [(("A", "B", "C")[int(rng.integers(0, 3))], f"2026-10-{4 + int(rng.integers(0, 14)):02d}",
             int(rng.integers(180, 1500)), int(rng.choice([50, 80, 120, 300, 600])),
             int(rng.integers(5_000, 400_000)), f"T{int(rng.integers(0, 12))}") for _ in range(n)]
    return slots(rows)


def test_greedy_plus_local_search_at_least_greedy():
    fc = _random_instance()
    cfg = make_cfg(budget=4000, min_util=0.95, end="2026-10-17", wmin=0.3, wmax=0.7,
                   chday={"A": 3, "B": 3, "C": 3}, prog_cap=2, ch_min={"A": 0.1, "B": 0.1, "C": 0.1})
    eng = make_engine(cfg)
    prob = build_problem(cfg, fc, NO_CONF)
    P = eng.prepare_many(prob.unit_frame(np.arange(prob.n)).assign(aud_abs=lambda d: d.aud))
    for kind in ("reach1", "reach3"):
        r = solve_reach(prob, eng, P, kind, cfg)
        assert r.penalty == 0
        assert r.objective >= r.stats["greedy_objective"] - 1e-6
        assert verify_constraints(plan_of(prob, r.units), prob.slots, cfg, conflicts=NO_CONF)["status"] == "PASS"


def test_balanced_meets_impressions_floor_and_trp_floor():
    fc = _random_instance(seed=5)
    cfg = make_cfg(budget=4000, min_util=0.9, end="2026-10-17", wmin=0.2, wmax=0.8, chday={"A": 4, "B": 4, "C": 4})
    eng = make_engine(cfg)
    run = Runner(cfg, forecast=fc, conflicts=NO_CONF, engine=eng, write=False, log=lambda *_: None)
    s3 = run.scenario("max_impressions")
    s4 = run.scenario("balanced")
    assert s4["kpis"]["impressions"] >= 0.9 * s3["kpis"]["impressions"] * (1 - 1e-9)
    assert s4["constraint_report"]["checks"]["min_impressions"]["status"] == "PASS"
    cfg_t = copy.deepcopy(cfg)
    cfg_t["constraints"]["min_trp_pct"] = 0.95 * s3["kpis"]["trp_pct"]
    run_t = Runner(cfg_t, forecast=fc, conflicts=NO_CONF, engine=eng, write=False, log=lambda *_: None)
    s1 = run_t.scenario("max_reach_1plus")
    assert s1["constraint_report"]["status"] == "PASS"
    assert s1["kpis"]["trp_pct"] >= cfg_t["constraints"]["min_trp_pct"] * (1 - 1e-9)


# --------------------------------------------------------------------------------- determinism
def test_determinism_two_runs_identical():
    fc = _random_instance(seed=11)
    cfg = make_cfg(budget=3000, min_util=0.95, end="2026-10-17", wmin=0.3, wmax=0.7, chday={"A": 3, "B": 3, "C": 3})
    outs = []
    for _ in range(2):
        eng = make_engine(cfg)
        run = Runner(cfg, forecast=fc, conflicts=NO_CONF, engine=eng, write=False, log=lambda *_: None)
        res = {n: run.scenario(n) for n in ("max_impressions", "max_reach_1plus", "max_reach_3plus", "balanced")}
        outs.append({n: (tuple(r["plan"]["slot_id"]), r["kpis"]["reach_1plus_abs"], r["kpis"]["impressions"])
                     for n, r in res.items()})
    assert outs[0] == outs[1]


# --------------------------------------------------------------------------------- frontier
def test_frontier_points_pass_and_reach_grows_with_budget():
    fc = _random_instance(seed=13)
    cfg = make_cfg(budget=3000, min_util=0.9, end="2026-10-17", wmin=0.2, wmax=0.8, chday={"A": 4, "B": 4, "C": 4})
    cfg["frontier"]["budget_levels"] = [0.5, 1.0]
    cfg["plan"]["frontier_trp_floors"] = [0.0, 0.99, 1.0]
    run = Runner(cfg, forecast=fc, conflicts=NO_CONF, engine=make_engine(cfg), write=False, log=lambda *_: None)
    run.scenario("max_impressions")
    run.scenario("max_reach_1plus")
    fr = run.frontier()
    assert (fr["constraints"] == "PASS").all()
    b = fr[fr.frontier == "budget"].set_index("level")
    assert b.loc[1.0, "reach_1plus_pct"] >= b.loc[0.5, "reach_1plus_pct"]
    t = fr[fr.frontier == "trp_floor"].set_index("level")
    assert t.loc[1.0, "trp_pct"] >= run.s3()["trp_pct"] * (1 - 1e-9)
