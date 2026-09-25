"""S3 max_impressions: exact MILP with OR-Tools CP-SAT (METHODOLOGY 4.2).

One binary per unit; objective = sum of integer audiences (persons, round half-up of the
forecast); money in integer cents; every constraint linear:

* budget: sum cost x <= B and >= ceil(min_utilisation x B)
* conflicts: x_a + x_b <= 1 for every conflicting pair (all copies)
* spots per channel per day, spots per programme per day: sum x <= cap
* channel shares (budget basis): sum cost x >= ceil(min x B), <= floor(max x B);
  (spend basis): den x ch_spend >= num x spend
* weekly phasing (spend basis): den x week_spend >= num_min x spend, <= num_max x spend
  (budget basis: against B)
* TRP floor: sum round(rating_pct x 1e6) x >= ceil(min_trp_pct x 1e6)
* impressions floor (optional): sum aud_int x >= ceil(floor)
* symmetry breaking between copies of one slot: x_k >= x_{k+1}

Determinism: fixed random_seed, fixed num_workers with interleave_search, and a
deterministic-time cap (plus a wall-clock cap).
"""
from __future__ import annotations

import math
import time

import numpy as np
from ortools.sat.python import cp_model

from .problem import Problem, share_fraction

TRP_SCALE = 10 ** 6


def solve_max_impressions(prob: Problem, cfg: dict, hint_units: np.ndarray | None = None) -> dict:
    pc = cfg["plan"]
    L = prob.lim
    n = prob.n
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{j}") for j in range(n)]
    cost = [int(c) for c in prob.cost]
    aud_int = np.floor(prob.aud + 0.5).astype(np.int64)
    spend = sum(cost[j] * x[j] for j in range(n))
    m.Add(spend <= L.budget)
    m.Add(spend >= L.min_spend)
    for a, b in prob.conflict_pairs:
        m.Add(x[int(a)] + x[int(b)] <= 1)
    groups: dict[int, list[int]] = {}
    for j, g in enumerate(prob.chday):
        groups.setdefault(int(g), []).append(j)
    for g, js in groups.items():
        cap = int(prob.chday_cap[g])
        if cap < len(js):
            m.Add(sum(x[j] for j in js) <= cap)
    if L.prog_cap is not None:
        groups = {}
        for j, g in enumerate(prob.prog):
            groups.setdefault(int(g), []).append(j)
        for js in groups.values():
            if L.prog_cap < len(js):
                m.Add(sum(x[j] for j in js) <= L.prog_cap)
    # channel shares
    for c in range(prob.C):
        js = np.flatnonzero(prob.ch == c)
        chs = sum(cost[j] * x[j] for j in js)
        lo, hi = L.ch_min_share[c], L.ch_max_share[c]
        if L.ch_basis == "budget":
            if lo > 0:
                m.Add(chs >= int(math.ceil(float(share_fraction(lo)) * L.budget - 1e-9)))
            if np.isfinite(hi):
                m.Add(chs <= int(math.floor(float(share_fraction(hi)) * L.budget + 1e-9)))
        else:
            if lo > 0:
                f = share_fraction(lo)
                m.Add(f.denominator * chs >= f.numerator * spend)
            if np.isfinite(hi):
                f = share_fraction(hi)
                m.Add(f.denominator * chs <= f.numerator * spend)
    # weekly phasing
    fmin, fmax = share_fraction(L.wk_min_share), share_fraction(L.wk_max_share)
    for w in range(prob.W):
        js = np.flatnonzero(prob.wk == w)
        wks = sum(cost[j] * x[j] for j in js) if len(js) else 0
        basis = spend if L.wk_basis == "spend" else L.budget
        m.Add(fmin.denominator * wks >= fmin.numerator * basis)
        m.Add(fmax.denominator * wks <= fmax.numerator * basis)
    if L.min_trp_pct > 0:
        tr = np.floor(prob.trp * TRP_SCALE + 0.5).astype(np.int64)
        m.Add(sum(int(tr[j]) * x[j] for j in range(n)) >= int(math.ceil(L.min_trp_pct * TRP_SCALE - 1e-6)))
    if L.min_impressions > 0:
        m.Add(sum(int(aud_int[j]) * x[j] for j in range(n)) >= int(math.ceil(L.min_impressions)))
    for s in range(len(prob.slots)):
        js = np.flatnonzero(prob.slot_of == s)
        for a, b in zip(js[:-1], js[1:]):
            m.Add(x[int(a)] >= x[int(b)])
    m.Maximize(sum(int(aud_int[j]) * x[j] for j in range(n)))
    if hint_units is not None:
        hs = set(int(u) for u in hint_units)
        for j in range(n):
            m.AddHint(x[j], 1 if j in hs else 0)
    solver = cp_model.CpSolver()
    prm = solver.parameters
    prm.random_seed = int(pc.get("random_seed", 0))
    prm.num_workers = int(pc.get("milp_num_workers", 8))
    prm.interleave_search = bool(pc.get("milp_interleave_search", True))
    prm.max_time_in_seconds = float(pc.get("milp_time_limit_sec", 120))
    if pc.get("milp_deterministic_time"):
        prm.max_deterministic_time = float(pc["milp_deterministic_time"])
    t = time.perf_counter()
    st = solver.Solve(m)
    wall = time.perf_counter() - t
    status = solver.status_name(st)
    out = {"status": status, "wall_sec": wall, "deterministic_time": solver.deterministic_time,
           "num_workers": prm.num_workers, "interleave_search": prm.interleave_search,
           "random_seed": prm.random_seed, "objective_unit": "persons (integer-rounded aud)"}
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        units = np.array([j for j in range(n) if solver.value(x[j])], dtype=np.int64)
        obj, bnd = solver.objective_value, solver.best_objective_bound
        out.update(units=units, objective=obj, best_bound=bnd,
                   gap=(bnd - obj) / obj if obj else None, conflicts=solver.num_conflicts,
                   branches=solver.num_branches)
    else:
        out.update(units=None, objective=None, best_bound=None, gap=None)
    return out
