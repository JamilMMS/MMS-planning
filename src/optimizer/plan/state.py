"""Incremental plan state: selected units, constraint aggregates and the reach-engine state.

Constraint handling
-------------------
* **Hard** (never violated by any move): unit not yet bought, no conflicting partner bought,
  spots per channel per day, spots per programme per day, spend <= budget.
* **Penalised** (exact integer-cent violation amount, 0 = satisfied): min utilisation,
  channel min / max share, weekly min / max share, TRP floor, impressions floor. The greedy
  builder enforces conservative versions of these at every step (see ``GreedyBounds``); the
  local search only accepts moves that keep the penalty at 0 once it is 0, or reduce it.

Penalty 0 <=> the corresponding checks of :func:`verify_constraints` PASS (same exact
fraction arithmetic), so the heuristics and the independent check agree at the boundaries.
"""
from __future__ import annotations

import math

import numpy as np

from .problem import Problem, share_fraction

FLOOR_REL_TOL = 1e-9
BIG = 10 ** 15


class ShareBounds:
    """Exact integer bounds amount >= num/den x basis (lo) and amount <= num/den x basis (hi)."""

    def __init__(self, lo_shares, hi_shares):
        lo = [share_fraction(x) for x in lo_shares]
        hi = [share_fraction(x) if np.isfinite(x) else None for x in hi_shares]
        self.lo_num = np.array([f.numerator for f in lo], np.int64)
        self.lo_den = np.array([f.denominator for f in lo], np.int64)
        self.has_hi = np.array([f is not None for f in hi])
        self.hi_num = np.array([f.numerator if f is not None else 0 for f in hi], np.int64)
        self.hi_den = np.array([f.denominator if f is not None else 1 for f in hi], np.int64)

    def violation(self, amounts: np.ndarray, basis) -> np.ndarray:
        """amounts (G, m) int64, basis scalar or (m,) int64 -> total violation (m,) in cents."""
        basis = np.asarray(basis, np.int64)
        lo = np.maximum(0, self.lo_num[:, None] * basis - self.lo_den[:, None] * amounts) / self.lo_den[:, None]
        v = lo.sum(axis=0)
        if self.has_hi.any():
            h = self.has_hi
            hi = np.maximum(0, self.hi_den[h, None] * amounts[h] - self.hi_num[h, None] * basis) / self.hi_den[h, None]
            v = v + hi.sum(axis=0)
        return v


class GreedyBounds:
    """Conservative per-step bounds for the constructive greedy, valid for any final spend in
    [max(min_spend, spend so far), budget]: lower shares on the largest basis (budget), upper
    shares on the smallest admissible basis (budget, or max(min_spend, spend after the add) for
    spend-basis shares); plus the reservation rule budget - spend >= max(channel shortfall,
    week shortfall)."""

    def __init__(self, prob: Problem):
        L = prob.lim
        B = L.budget
        self.M = L.min_spend
        self.ch_spend_basis = L.ch_basis != "budget"
        self.wk_spend_basis = L.wk_basis != "budget"
        self.ch_hi_frac = [share_fraction(s) if np.isfinite(s) else None for s in L.ch_max_share]
        self.wk_hi_frac = share_fraction(L.wk_max_share)
        self.ch_hi = np.array([math.floor(float(f) * B + 1e-9) if f is not None else BIG for f in self.ch_hi_frac], np.int64)
        self.ch_lo = np.array([math.ceil(float(share_fraction(s)) * B - 1e-9) for s in L.ch_min_share], np.int64)
        self.wk_hi = int(math.floor(float(self.wk_hi_frac) * B + 1e-9))
        self.wk_lo = int(math.ceil(float(share_fraction(L.wk_min_share)) * B - 1e-9))

    def ch_upper(self, ch: int, spend_after: int) -> int:
        f = self.ch_hi_frac[ch]
        if f is None:
            return BIG
        if not self.ch_spend_basis:
            return int(self.ch_hi[ch])
        return (f.numerator * max(self.M, spend_after)) // f.denominator

    def wk_upper(self, spend_after: int) -> int:
        if not self.wk_spend_basis:
            return self.wk_hi
        f = self.wk_hi_frac
        if f >= 1:
            return BIG
        return (f.numerator * max(self.M, spend_after)) // f.denominator


class SearchState:
    def __init__(self, prob: Problem, engine, prepared: list):
        self.prob, self.engine, self.P = prob, engine, prepared
        n = prob.n
        self.sel = np.zeros(n, dtype=bool)
        self.spend = 0
        self.ch_spend = np.zeros(prob.C, np.int64)
        self.wk_spend = np.zeros(prob.W, np.int64)
        self.chday_cnt = np.zeros(len(prob.chday_cap), np.int64)
        self.prog_cnt = np.zeros(prob.n_prog, np.int64)
        self.conf_cnt = np.zeros(n, np.int64)
        self.trp = 0.0
        self.aud = 0.0
        self.est = engine.new_state()
        L = prob.lim
        self._chb = ShareBounds(L.ch_min_share, L.ch_max_share)
        self._wkb = ShareBounds([L.wk_min_share] * prob.W, [L.wk_max_share] * prob.W)
        self._prog_cap = L.prog_cap if L.prog_cap is not None else BIG

    # ---- mutation ------------------------------------------------------------------------
    def copy(self) -> "SearchState":
        o = object.__new__(SearchState)
        o.prob, o.engine, o.P = self.prob, self.engine, self.P
        for k in ("sel", "ch_spend", "wk_spend", "chday_cnt", "prog_cnt", "conf_cnt"):
            setattr(o, k, getattr(self, k).copy())
        o.spend, o.trp, o.aud = self.spend, self.trp, self.aud
        o.est = self.est.copy()
        o._chb, o._wkb, o._prog_cap = self._chb, self._wkb, self._prog_cap
        return o

    def add(self, j: int) -> None:
        p = self.prob
        assert not self.sel[j]
        c = int(p.cost[j])
        self.sel[j] = True
        self.spend += c
        self.ch_spend[p.ch[j]] += c
        self.wk_spend[p.wk[j]] += c
        self.chday_cnt[p.chday[j]] += 1
        self.prog_cnt[p.prog[j]] += 1
        self.conf_cnt[p.partners[j]] += 1
        self.trp += p.trp[j]
        self.aud += p.aud[j]
        self.engine.add(self.est, self.P[j])

    def remove(self, j: int) -> None:
        p = self.prob
        assert self.sel[j]
        c = int(p.cost[j])
        self.sel[j] = False
        self.spend -= c
        self.ch_spend[p.ch[j]] -= c
        self.wk_spend[p.wk[j]] -= c
        self.chday_cnt[p.chday[j]] -= 1
        self.prog_cnt[p.prog[j]] -= 1
        self.conf_cnt[p.partners[j]] -= 1
        self.trp -= p.trp[j]
        self.aud -= p.aud[j]
        self.engine.remove(self.est, self.P[j])

    @property
    def units(self) -> np.ndarray:
        return np.flatnonzero(self.sel)

    # ---- penalties -------------------------------------------------------------------------
    def _penalty(self, spend, chs, wks, trp, aud) -> np.ndarray:
        L = self.prob.lim
        spend = np.asarray(spend, np.int64)
        pen = np.maximum(0, L.min_spend - spend).astype(float)
        pen = pen + self._chb.violation(chs, L.budget if L.ch_basis == "budget" else spend)
        pen = pen + self._wkb.violation(wks, L.budget if L.wk_basis == "budget" else spend)
        if L.min_trp_pct > 0:
            short = np.maximum(0.0, L.min_trp_pct * (1 - FLOOR_REL_TOL) - np.asarray(trp, float))
            pen = pen + short * (L.budget / L.min_trp_pct)
        if L.min_impressions > 0:
            short = np.maximum(0.0, L.min_impressions * (1 - FLOOR_REL_TOL) - np.asarray(aud, float))
            pen = pen + short * (L.budget / L.min_impressions)
        return pen

    def penalty(self) -> float:
        return float(self._penalty(np.array([self.spend]), self.ch_spend[:, None], self.wk_spend[:, None],
                                   np.array([self.trp]), np.array([self.aud]))[0])

    def penalty_without(self, i: int) -> float:
        p = self.prob
        c = int(p.cost[i])
        chs = self.ch_spend.copy()
        chs[p.ch[i]] -= c
        wks = self.wk_spend.copy()
        wks[p.wk[i]] -= c
        return float(self._penalty(np.array([self.spend - c]), chs[:, None], wks[:, None],
                                   np.array([self.trp - p.trp[i]]), np.array([self.aud - p.aud[i]]))[0])

    def eval_moves(self, i: int | None, J: np.ndarray, ignore_budget: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Remove ``i`` (or nothing) and add each unit of ``J``: (hard-feasible mask, penalty after;
        +inf where not hard-feasible). ignore_budget: leave spend <= budget out of the mask (the
        caller repairs the budget)."""
        p = self.prob
        J = np.asarray(J)
        ci = int(p.cost[i]) if i is not None else 0
        cj_all = p.cost[J]
        hard = ~self.sel[J]
        if not ignore_budget:
            hard &= (self.spend - ci + cj_all) <= p.lim.budget
        chd = self.chday_cnt[p.chday[J]] + 1
        prg = self.prog_cnt[p.prog[J]] + 1
        conf = self.conf_cnt[J]
        if i is not None:
            hard &= J != i
            chd = chd - (p.chday[J] == p.chday[i])
            prg = prg - (p.prog[J] == p.prog[i])
            if len(p.partners[i]):
                conf = conf - np.isin(J, p.partners[i])
        hard &= (conf == 0) & (chd <= p.chday_cap[p.chday[J]]) & (prg <= self._prog_cap)
        pen = np.full(len(J), np.inf)
        idx = np.flatnonzero(hard)
        if len(idx):
            H = J[idx]
            m = len(H)
            cj = p.cost[H]
            spend = self.spend - ci + cj
            chs = np.repeat(self.ch_spend[:, None], m, axis=1)
            wks = np.repeat(self.wk_spend[:, None], m, axis=1)
            trp = self.trp + p.trp[H]
            aud = self.aud + p.aud[H]
            if i is not None:
                chs[p.ch[i]] -= ci
                wks[p.wk[i]] -= ci
                trp = trp - p.trp[i]
                aud = aud - p.aud[i]
            ar = np.arange(m)
            chs[p.ch[H], ar] += cj
            wks[p.wk[H], ar] += cj
            pen[idx] = self._penalty(spend, chs, wks, trp, aud)
        return hard, pen

    # ---- greedy step check ------------------------------------------------------------------
    def greedy_ok(self, j: int, gb: GreedyBounds) -> int:
        """0 = can add; 1 = cannot, permanently while only adding (hard constraints, budget-basis
        shares, reservation for outstanding minimums - all monotone); 2 = blocked only by a
        spend-basis upper share, which loosens as spend grows (retry later)."""
        p = self.prob
        if self.sel[j] or self.conf_cnt[j] > 0:
            return 1
        c = int(p.cost[j])
        sp = self.spend + c
        if sp > p.lim.budget:
            return 1
        if self.chday_cnt[p.chday[j]] + 1 > p.chday_cap[p.chday[j]] or self.prog_cnt[p.prog[j]] + 1 > self._prog_cap:
            return 1
        ch, wk = p.ch[j], p.wk[j]
        if self.ch_spend[ch] + c > gb.ch_hi[ch] and not gb.ch_spend_basis:
            return 1
        if self.wk_spend[wk] + c > gb.wk_hi and not gb.wk_spend_basis:
            return 1
        ch_sf = np.maximum(0, gb.ch_lo - self.ch_spend)
        ch_sf_tot = int(ch_sf.sum()) - int(min(c, ch_sf[ch]))
        wk_sf = np.maximum(0, gb.wk_lo - self.wk_spend)
        wk_sf_tot = int(wk_sf.sum()) - int(min(c, wk_sf[wk]))
        if p.lim.budget - sp < max(ch_sf_tot, wk_sf_tot):
            return 1
        if self.ch_spend[ch] + c > gb.ch_upper(ch, sp) or self.wk_spend[wk] + c > gb.wk_upper(sp):
            return 2
        return 0
