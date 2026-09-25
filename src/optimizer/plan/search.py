"""Reach objectives (S1 max Reach 1+, S2 max Reach 3+, S4 balanced): lazy greedy on marginal
objective per dollar + best-single-item correction + local search (METHODOLOGY 4.2).

Greedy
  1. channel-minimum pass: for each channel (config order) add that channel's best
     marginal-objective-per-dollar spots until its minimum share is met;
  2. global lazy greedy (max-heap of stale scores, re-evaluated on pop; an item whose fresh
     score still beats the next stale score is taken).
  Every step is checked by ``SearchState.greedy_ok`` (hard constraints + conservative share
  bounds + reservation of budget for outstanding channel / week minimums). All those checks are
  monotone while adding, so an item that fails once is dropped for the rest of the pass.
  Score = (d objective + lambda x audience) / cost; lambda > 0 only to meet a TRP / impressions
  floor (smallest lambda found by doubling + bisection). S2 adds w x d Reach 1+ as a tie-break
  (Reach 3+ gain is 0 for the first spots of a channel) and refreshes all scores every
  ``plan.s2_full_refresh_every`` picks (Reach 3+ is not submodular, so stale scores can be low).

Cost-benefit correction + greedy family
  Ratio greedy; ratio greedy seeded with the best single item (largest absolute score
  numerator on the empty plan); and score = gain / cost^alpha for alpha in
  plan.greedy_cost_exponents (alpha = 0 is the pure-gain greedy). The per-channel-per-day caps
  bind before the budget on this grid, and plain ratio greedy then spends the capped spot
  slots on cheap low-audience spots; the family covers that. The best plan is kept.

Multi-start
  When the S3 MILP optimum for the same budget is available it is also used as a local-search
  start (always for S4 and TRP floors, and for S1 / S2 at the base budget).

Local search (until no improving move, or plan.local_search_max_iters accepted moves)
  * add: best feasible single addition;
  * swap: remove i, add j (best j over a per-channel shortlist: top-N feasible by audience,
    audience/cost and cost; exact for channel-cell reach models, whose marginal reach is
    increasing in the spot's audience within a channel);
  * drop-add: remove an expensive i and refill greedily from a per-channel shortlist;
  * upgrade: swap i for a larger, pricier unit of the same channel (typically the same
    capped channel-day), fund it by dropping the lowest loss-per-dollar spots (loss ranking
    refreshed every sweep), refill leftover (plan.upgrade_candidates_per_spot targets per i).
  A move is accepted if it lowers the constraint penalty, or keeps it (0 stays 0) and raises
  the objective by more than plan.local_search_min_gain.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from .problem import Problem
from .state import GreedyBounds, SearchState

PEN_TOL = 1e-6


# ----------------------------------------------------------------------------- objectives
class Objective:
    def __init__(self, kind: str, engine, tiebreak_w: float = 0.0):
        if kind not in ("reach1", "reach3"):
            raise ValueError(kind)
        self.kind, self.engine, self.w = kind, engine, float(tiebreak_w)
        self.key = "reach_1plus_abs" if kind == "reach1" else "reach_3plus_abs"

    def value(self, est) -> float:
        return float(self.engine.evaluate(est)[self.key])

    def gain(self, est, p) -> float:
        if self.kind == "reach1":
            return self.engine.marginal_reach(est, p)
        return self.engine.marginal_reach_n(est, p, 3)

    def greedy_gain(self, est, p) -> float:
        if self.kind == "reach1":
            return self.engine.marginal_reach(est, p)
        g = self.engine.marginal_reach_n(est, p, 3)
        return g + self.w * self.engine.marginal_reach(est, p) if self.w else g


@dataclass
class SearchResult:
    units: np.ndarray
    objective: float
    penalty: float
    stats: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- greedy
def _lazy_pass(s: SearchState, obj: Objective, gb: GreedyBounds, members: np.ndarray, lam: float,
               stop=None, refresh_every: int = 0, alpha: float = 1.0) -> int:
    p = s.prob
    P = s.P
    aud = p.aud
    cost = p.cost.astype(float) ** alpha

    def score(j: int) -> float:
        return (obj.greedy_gain(s.est, P[j]) + lam * aud[j]) / cost[j]

    def build(items):
        h = [(-score(int(j)), int(j)) for j in items if s.greedy_ok(int(j), gb) == 0]
        heapq.heapify(h)
        return h

    heap = build(members)
    picks = 0
    deferred: list[int] = []
    while True:
        spend_at_defer = s.spend
        while heap:
            if stop is not None and stop():
                return picks
            _, j = heapq.heappop(heap)
            ok = s.greedy_ok(j, gb)
            if ok == 1:
                continue
            if ok == 2:
                deferred.append(j)
                continue
            sc = score(j)
            if heap and sc < -heap[0][0] - 1e-12 * max(1.0, abs(sc)):
                heapq.heappush(heap, (-sc, j))
                continue
            s.add(j)
            picks += 1
            if refresh_every and picks % refresh_every == 0:
                heap = build([h[1] for h in heap] + deferred)
                deferred = []
        # items blocked only by a spend-basis upper share: retry once spend has grown
        if not deferred or s.spend == spend_at_defer or (stop is not None and stop()):
            return picks
        heap, deferred = build(deferred), []
        if not heap:
            return picks


def greedy(prob: Problem, engine, P: list, obj: Objective, lam: float = 0.0, seed_unit: int | None = None,
           refresh_every: int = 0, alpha: float = 1.0) -> SearchState:
    s = SearchState(prob, engine, P)
    gb = GreedyBounds(prob)
    if seed_unit is not None and s.greedy_ok(seed_unit, gb) == 0:
        s.add(seed_unit)
    allu = np.arange(prob.n)
    for c in range(prob.C):
        if gb.ch_lo[c] <= 0:
            continue
        mem = allu[prob.ch == c]
        _lazy_pass(s, obj, gb, mem, lam, stop=lambda c=c: s.ch_spend[c] >= gb.ch_lo[c], refresh_every=refresh_every,
                   alpha=alpha)
    _lazy_pass(s, obj, gb, allu, lam, refresh_every=refresh_every, alpha=alpha)
    return s


def best_single_unit(prob: Problem, engine, P: list, obj: Objective, lam: float) -> int | None:
    s = SearchState(prob, engine, P)
    gb = GreedyBounds(prob)
    best, bv = None, -np.inf
    for j in range(prob.n):
        if s.greedy_ok(j, gb) != 0:
            continue
        v = obj.greedy_gain(s.est, P[j]) + lam * prob.aud[j]
        if v > bv + 1e-9:
            best, bv = j, v
    return best


def _key(pen: float, val: float) -> tuple:
    return (-round(pen, 6), val)


def corrected_greedy(prob, engine, P, obj, lam, refresh_every, alphas=(1.0,)) -> tuple[SearchState, dict]:
    """Best of: ratio greedy (alpha = 1), ratio greedy seeded with the best single item (the
    cost-benefit correction), and the greedy family score = gain / cost^alpha for the other
    ``alphas`` (plan.greedy_cost_exponents; alpha < 1 favours high-audience spots, which matters
    when the per-channel-per-day caps bind before the budget does)."""
    t = time.perf_counter()
    runs: dict[str, SearchState] = {}
    runs["ratio"] = a = greedy(prob, engine, P, obj, lam, None, refresh_every, 1.0)
    seed = best_single_unit(prob, engine, P, obj, lam)
    if seed is not None and not a.sel[seed]:
        runs["seeded_best_single"] = greedy(prob, engine, P, obj, lam, seed, refresh_every, 1.0)
    for al in alphas:
        if float(al) != 1.0:
            runs[f"alpha_{float(al):g}"] = greedy(prob, engine, P, obj, lam, None, refresh_every, float(al))
    info = {"best_single_unit": seed, "best_single_unit_in_ratio_plan": bool(seed is not None and a.sel[seed]),
            "greedy_runs": {}}
    best_name, best_k = None, None
    for name, st in runs.items():
        v, pn = obj.value(st.est), st.penalty()
        info["greedy_runs"][name] = {"objective": v, "penalty": pn, "spots": int(st.sel.sum()), "spend_cents": int(st.spend)}
        k = _key(pn, v)
        if best_k is None or k > best_k:
            best_name, best_k = name, k
    info["greedy_choice"] = best_name
    info["greedy_sec"] = time.perf_counter() - t
    return runs[best_name], info


# ----------------------------------------------------------------------------- local search
class _Orders:
    """Per-channel unit orders (static): by audience desc, audience/cost desc, cost desc."""

    def __init__(self, prob: Problem):
        self.by = []
        for c in range(prob.C):
            u = np.flatnonzero(prob.ch == c)
            a, co = prob.aud[u], prob.cost[u].astype(float)
            self.by.append([u[np.lexsort((u, -a))], u[np.lexsort((u, -(a / co)))], u[np.lexsort((u, -co))]])


def _shortlist(orders: _Orders, ok: np.ndarray, n: int, which=(0, 1, 2)) -> np.ndarray:
    out = []
    for lists in orders.by:
        for k in which:
            o = lists[k]
            f = o[ok[o]][:n]
            out.append(f)
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)


def _better(pen_new, obj_new, pen_cur, obj_cur, min_gain) -> bool:
    if pen_new < pen_cur - PEN_TOL:
        return True
    return pen_new <= pen_cur + PEN_TOL and obj_new > obj_cur + min_gain


def _best_move(s: SearchState, obj: Objective, orders: _Orders, i: int | None, pen_cur: float, obj_cur: float,
               N: int, min_gain: float):
    prob = s.prob
    allu = np.arange(prob.n)
    hard, pen = s.eval_moves(i, allu)
    if pen_cur <= PEN_TOL:
        ok = hard & (pen <= PEN_TOL)
    else:
        ok = hard & (pen <= pen_cur + PEN_TOL)
    est = s.est
    if i is not None:
        est = est.copy()
        s.engine.remove(est, s.P[i])
    base = obj.value(est) if i is not None else obj_cur
    J = _shortlist(orders, ok, N)
    if pen_cur > PEN_TOL:
        cand = np.flatnonzero(ok)
        if len(cand):
            J = np.union1d(J, cand[np.lexsort((cand, pen[cand]))][:N * prob.C])
    best = None
    for j in J:
        v = base + obj.gain(est, s.P[j])
        key = (-(pen[j]), v)
        if best is None or key > best[0]:
            best = (key, int(j), float(pen[j]), v)
    if i is not None and pen_cur > PEN_TOL:   # pure drop can repair an upper-bound violation
        pd_ = s.penalty_without(i)
        if best is None or (pd_, -base) < (best[2], -best[3]):
            best = ((-pd_, base), None, pd_, base)
    if best is None:
        return None
    _, j, pn, v = best
    return (j, pn, v) if _better(pn, v, pen_cur, obj_cur, min_gain) else None


def _fill(t: SearchState, obj: Objective, orders: _Orders, N_fill: int) -> list[int]:
    """Greedy refill (best gain per cost over a per-channel shortlist) keeping the penalty non-increasing."""
    allu = np.arange(t.prob.n)
    cost = t.prob.cost
    pen_t = t.penalty()
    added = []
    while True:
        hard, pen = t.eval_moves(None, allu)
        ok = hard & (pen <= pen_t + PEN_TOL)
        J = _shortlist(orders, ok, N_fill, which=(1,))
        if not len(J):
            break
        g = np.array([obj.gain(t.est, t.P[j]) for j in J])
        r = g / cost[J]
        k = int(np.lexsort((J, -r))[0])
        if g[k] <= 0 and pen[J[k]] >= pen_t - PEN_TOL:
            break
        j = int(J[k])
        t.add(j)
        added.append(j)
        pen_t = float(pen[j])
    return added


def _drop_add(s: SearchState, obj: Objective, orders: _Orders, i: int, pen_cur: float, obj_cur: float,
              N_fill: int, min_gain: float):
    t = s.copy()
    t.remove(i)
    added = _fill(t, obj, orders, N_fill)
    if not added:
        return None
    v, pn = obj.value(t.est), t.penalty()
    return (t, added, pn, v) if _better(pn, v, pen_cur, obj_cur, min_gain) else None


def _loss_rank(s: SearchState, obj: Objective, obj_cur: float) -> list[int]:
    """Selected units by ascending (objective lost if removed) / cost."""
    u = s.units
    loss = np.empty(len(u))
    for n_, k in enumerate(u):
        e = s.est.copy()
        s.engine.remove(e, s.P[int(k)])
        loss[n_] = obj_cur - obj.value(e)
    r = loss / s.prob.cost[u]
    return [int(x) for x in u[np.lexsort((u, r))]]


def _upgrade(s: SearchState, obj: Objective, orders: _Orders, i: int, pen_cur: float, obj_cur: float,
             rank: list[int], N_up: int, N_fill: int, min_gain: float):
    """Swap i for a larger, pricier unit j of the same channel that the budget cannot absorb,
    fund it by dropping the lowest loss-per-dollar spots (keeping the penalty at its level),
    then refill any leftover."""
    prob = s.prob
    B = prob.lim.budget
    by_aud = orders.by[prob.ch[i]][0]
    hard, pen = s.eval_moves(i, by_aud, ignore_budget=True)
    need = s.spend - int(prob.cost[i]) + prob.cost[by_aud] - B
    okm = hard & (pen <= pen_cur + PEN_TOL) & (prob.aud[by_aud] > prob.aud[i]) & (need > 0)
    best = None
    for j in by_aud[okm][:N_up]:
        t = s.copy()
        t.remove(i)
        t.add(int(j))
        base_pen = t.penalty()
        for k in rank:
            if t.spend <= B:
                break
            if k == i or k == j or not t.sel[k]:
                continue
            if t.penalty_without(k) > base_pen + PEN_TOL:
                continue
            t.remove(k)
        if t.spend > B:
            continue
        _fill(t, obj, orders, N_fill)
        v, pn = obj.value(t.est), t.penalty()
        if _better(pn, v, pen_cur, obj_cur, min_gain) and (best is None or _key(pn, v) > _key(best[2], best[3])):
            best = (t, int(j), pn, v)
    return best


def local_search(s: SearchState, obj: Objective, cfg: dict) -> tuple[SearchState, dict]:
    pc = cfg["plan"]
    max_iters = int(pc["local_search_max_iters"])
    max_sweeps = int(pc["local_search_max_sweeps"])
    min_gain = float(pc["local_search_min_gain"])
    N = int(pc["local_search_shortlist_per_cell"])
    N_fill = int(pc["local_search_fill_shortlist_per_cell"])
    N_up = int(pc.get("upgrade_candidates_per_spot", 0))
    prob = s.prob
    da_min = float(pc["drop_add_min_cost_ratio"]) * float(prob.cost.min())
    orders = _Orders(prob)
    t0 = time.perf_counter()
    obj_cur, pen_cur = obj.value(s.est), s.penalty()
    stats = {"start_objective": obj_cur, "start_penalty": pen_cur,
             "moves": {"add": 0, "swap": 0, "drop": 0, "drop_add": 0, "upgrade": 0},
             "sweeps": 0, "stopped": "no_improving_move"}
    n_moves = 0
    for sweep in range(max_sweeps):
        stats["sweeps"] = sweep + 1
        improved = False
        rank = _loss_rank(s, obj, obj_cur) if N_up > 0 else []
        for i in [None] + [int(u) for u in s.units]:
            if n_moves >= max_iters:
                break
            if i is not None and not s.sel[i]:
                continue
            mv = _best_move(s, obj, orders, i, pen_cur, obj_cur, N, min_gain)
            if mv is not None:
                j, pn, v = mv
                if i is not None:
                    s.remove(i)
                if j is not None:
                    s.add(j)
                kind = "add" if i is None else ("drop" if j is None else "swap")
                stats["moves"][kind] += 1
                n_moves += 1
                improved = True
                obj_cur, pen_cur = obj.value(s.est), s.penalty()
                continue
            if i is not None and prob.cost[i] >= da_min:
                r = _drop_add(s, obj, orders, i, pen_cur, obj_cur, N_fill, min_gain)
                if r is not None:
                    s = r[0]
                    stats["moves"]["drop_add"] += 1
                    n_moves += 1
                    improved = True
                    obj_cur, pen_cur = obj.value(s.est), s.penalty()
                    continue
            if i is not None and N_up > 0:
                r = _upgrade(s, obj, orders, i, pen_cur, obj_cur, rank, N_up, N_fill, min_gain)
                if r is not None:
                    s = r[0]
                    stats["moves"]["upgrade"] += 1
                    n_moves += 1
                    improved = True
                    obj_cur, pen_cur = obj.value(s.est), s.penalty()
        if n_moves >= max_iters:
            stats["stopped"] = "move_budget"
            break
        if not improved:
            break
    else:
        stats["stopped"] = "sweep_budget"
    stats.update(end_objective=obj_cur, end_penalty=pen_cur, accepted_moves=n_moves,
                 local_search_sec=time.perf_counter() - t0)
    return s, stats


# ----------------------------------------------------------------------------- scenario driver
def _floors_met(s: SearchState) -> bool:
    return s.penalty() <= PEN_TOL


def solve_reach(prob: Problem, engine, P: list, kind: str, cfg: dict,
                start_units: np.ndarray | None = None) -> SearchResult:
    """Max Reach 1+ (kind='reach1') or Reach 3+ ('reach3') under every constraint of ``prob``.

    start_units : an extra feasible start (the S3 optimum under the same budget); local search is
    run from the best greedy plan and from this start, and the better plan is kept. With a TRP /
    impressions floor near 100 % of the MILP optimum this start is what makes the floor reachable.
    """
    pc = cfg["plan"]
    obj = Objective(kind, engine, float(pc.get("s2_reach1_tiebreak_weight", 0.0)) if kind == "reach3" else 0.0)
    refresh = int(pc.get("s2_full_refresh_every", 0)) if kind == "reach3" else 0
    alphas = tuple(float(a) for a in pc.get("greedy_cost_exponents", [1.0]))
    L = prob.lim
    floors = L.min_trp_pct > 0 or L.min_impressions > 0
    stats: dict = {"objective": kind}
    lam = 0.0
    if not floors:
        s, ginfo = corrected_greedy(prob, engine, P, obj, 0.0, refresh, alphas)
        lam_trace = []
    else:
        # smallest lambda (score = (gain + lambda x aud) / cost) whose ratio greedy meets the floors
        s0, _ = corrected_greedy(prob, engine, P, obj, 0.0, refresh)
        lam_trace = [(0.0, s0.penalty())]
        found = _floors_met(s0)
        if not found:
            lo, hi = 0.0, float(pc["floor_lambda_start"])
            while hi <= float(pc["floor_lambda_max"]):
                cand, _ = corrected_greedy(prob, engine, P, obj, hi, refresh)
                lam_trace.append((hi, cand.penalty()))
                if _floors_met(cand):
                    found = True
                    break
                lo, hi = hi, hi * 2
            if found:
                for _ in range(int(pc["floor_lambda_bisect_steps"])):
                    mid = 0.5 * (lo + hi)
                    cand, _ = corrected_greedy(prob, engine, P, obj, mid, refresh)
                    lam_trace.append((mid, cand.penalty()))
                    if _floors_met(cand):
                        hi = mid
                    else:
                        lo = mid
                lam = hi
        s, ginfo = corrected_greedy(prob, engine, P, obj, lam, refresh, alphas)
        stats["floor_met_by_greedy"] = found
    stats.update(ginfo)
    stats["lambda"] = lam
    stats["lambda_trace"] = lam_trace
    stats["greedy_objective"] = obj.value(s.est)
    stats["greedy_penalty"] = s.penalty()
    s, ls = local_search(s, obj, cfg)
    stats["local_search"] = ls
    best = s
    starts = {"greedy": (obj.value(s.est), s.penalty())}
    if start_units is not None:
        t = SearchState(prob, engine, P)
        for u in start_units:
            t.add(int(u))
        starts_s3_before = (obj.value(t.est), t.penalty())
        t, ls2 = local_search(t, obj, cfg)
        stats["local_search_from_s3"] = ls2
        v2, p2 = obj.value(t.est), t.penalty()
        starts["s3_start"] = (v2, p2)
        stats["s3_start_before_local_search"] = {"objective": starts_s3_before[0], "penalty": starts_s3_before[1]}
        vb, pb = starts["greedy"]
        if _key(p2, v2) > _key(pb, vb + 1e-9):
            best = t
    stats["starts"] = {k: {"objective": v[0], "penalty": v[1]} for k, v in starts.items()}
    stats["chosen_start"] = "s3_start" if best is not s else "greedy"
    return SearchResult(units=best.units, objective=obj.value(best.est), penalty=best.penalty(), stats=stats)
