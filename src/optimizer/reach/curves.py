"""Reach-curve building blocks shared by the ESTIMATE and CALIBRATED modes.

Everything here is a pure function of explicit parameters (no config reads, no data).

Units
-----
* ``g``    GRPs of one cell in % of universe (GRP % = sum of spot audiences / U x 100).
* ``R``    reach 1+ of one cell in % of universe.
* ``Rmax`` asymptotic reach of the cell in % of universe.
* ``k``    shape parameter in GRP %; for both forms the initial slope dR/dg at g = 0 is
           ``Rmax / k`` (reach points per GRP point).

Forms
-----
* ``hyperbolic``: R(g) = Rmax * g / (k + g)
* ``negexp``    : R(g) = Rmax * (1 - exp(-g / k))

Cross-cell combination
----------------------
* Sainsbury (independence): P(none) = prod_c (1 - r_c), r_c = R_c / 100.
* Pairwise duplication adjustment (CALIBRATED with a duplication matrix): the pair
  factor c_ij = P(not i, not j) / ((1-r_i)(1-r_j)) = 1 + (phi_ij - 1) r_i r_j / ((1-r_i)(1-r_j)),
  with phi_ij = duplication index (1 = independent), and P(none) = prod(1-r_c) * prod_{i<j} c_ij.
  Exact for two cells; a pairwise (Kirkwood-type) approximation for more; the result is
  clipped to the Frechet bounds [max(0, 1 - sum r), 1 - max r].

Frequency distribution (per cell)
---------------------------------
Given mean exposures per universe member ``m = g/100`` and P(0) = 1 - r, the cell's
exposure count follows a negative binomial (NBD) whose dispersion is solved so that
P(0) = 1 - r exactly; if r exceeds the Poisson limit 1 - exp(-m) (only possible when the
curve's initial slope is > 1, e.g. in the k-30% sensitivity run), a zero-modified,
zero-truncated Poisson with mean OTS = m / r is used instead. Mass above the number of spots
bought in the cell (a person cannot see more spots than were bought) and above the reporting
cap K is lumped into the top bucket. Cells are combined by convolution (independence,
consistent with Sainsbury for reach 1+).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import betaln, gammaln

FORMS = ("hyperbolic", "negexp")


# ----------------------------------------------------------------------------- curves
def curve(g: float | np.ndarray, rmax: float, k: float, form: str) -> float | np.ndarray:
    """Reach 1+ (% of universe) of one cell at ``g`` GRP %."""
    g = np.maximum(np.asarray(g, dtype=float), 0.0)
    if form == "hyperbolic":
        out = rmax * g / (k + g)
    elif form == "negexp":
        out = rmax * -np.expm1(-g / k)
    else:
        raise ValueError(f"unknown reach curve form {form!r}; expected one of {FORMS}")
    return float(out) if out.ndim == 0 else out


def curve_scalar(g: float, rmax: float, k: float, form_id: int) -> float:
    """Fast scalar version for the optimizer hot path (form_id 0 = hyperbolic, 1 = negexp)."""
    if g <= 0.0:
        return 0.0
    if form_id == 0:
        return rmax * g / (k + g)
    return -rmax * math.expm1(-g / k)


def form_id(form: str) -> int:
    if form not in FORMS:
        raise ValueError(f"unknown reach curve form {form!r}; expected one of {FORMS}")
    return FORMS.index(form)


# ----------------------------------------------------------------------------- day cume
def beta_day_cume(daily_reach: float, rho: float, n_days: int) -> float:
    """Cume reach (fraction) over ``n_days`` from an average daily reach (fraction) when each
    person's daily viewing probability p ~ Beta with mean ``daily_reach`` and intra-person
    correlation ``rho`` (rho -> 0: days independent; rho -> 1: same people every day).

    cume = 1 - E[(1-p)^N] = 1 - B(a, b + N) / B(a, b), a = d (1/rho - 1), b = (1-d)(1/rho - 1).
    Next-day repeat rate implied: P(view day t+1 | view day t) = d + rho (1 - d).
    """
    d = float(daily_reach)
    if not 0.0 < d < 1.0:
        raise ValueError(f"daily_reach must be in (0,1), got {d}")
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho must be in (0,1), got {rho}")
    s = 1.0 / rho - 1.0
    a, b = d * s, (1.0 - d) * s
    return float(1.0 - math.exp(betaln(a, b + n_days) - betaln(a, b)))


# ----------------------------------------------------------------------------- combination
def sainsbury(r: Sequence[float] | np.ndarray) -> float:
    """Combined reach (fraction) of independent cells with reaches ``r`` (fractions)."""
    r = np.clip(np.asarray(r, dtype=float), 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - r))


def pair_factor(ri: float, rj: float, phi: float) -> float:
    """c_ij = P(not i & not j) / ((1-ri)(1-rj)) for duplication index ``phi``."""
    if phi == 1.0 or ri <= 0.0 or rj <= 0.0:
        return 1.0
    den = (1.0 - ri) * (1.0 - rj)
    if den <= 0.0:
        return 1.0
    # P(i & j) = phi ri rj must lie within Frechet bounds
    both = min(phi * ri * rj, ri, rj)
    both = max(both, ri + rj - 1.0, 0.0)
    return (1.0 - ri - rj + both) / den


def combine_pairwise(r: Sequence[float] | np.ndarray, phi: np.ndarray | None) -> float:
    """Combined reach (fraction) with a pairwise duplication-index matrix (None = Sainsbury)."""
    r = np.clip(np.asarray(r, dtype=float), 0.0, 1.0)
    if phi is None:
        return sainsbury(r)
    p_none = float(np.prod(1.0 - r))
    n = len(r)
    for i in range(n):
        for j in range(i + 1, n):
            p_none *= pair_factor(r[i], r[j], float(phi[i, j]))
    lo = max(0.0, 1.0 - float(r.sum()))
    hi = 1.0 - float(r.max()) if n else 1.0
    p_none = min(max(p_none, lo), hi)
    return 1.0 - p_none


# ----------------------------------------------------------------------------- frequency
def _nbd_p0(m: float, kk: float) -> float:
    return math.exp(-kk * math.log1p(m / kk))


def solve_nbd_k(m: float, p0: float) -> float | None:
    """NBD shape ``k`` such that P(0) = p0 for mean ``m``; None if p0 is below the Poisson
    limit exp(-m) (reach above Poisson)."""
    if p0 >= 1.0 or m <= 0:
        return None
    if p0 <= math.exp(-m) * (1 + 1e-12):
        return None
    target = math.log(p0)
    f = lambda lk: -math.exp(lk) * math.log1p(m / math.exp(lk)) - target  # noqa: E731
    lo, hi = -30.0, 30.0
    if f(hi) > 0:      # essentially Poisson
        return math.exp(hi)
    if f(lo) < 0:
        return math.exp(lo)
    return math.exp(brentq(f, lo, hi, xtol=1e-12))


def cell_frequency_pmf(g_pct: float, reach_pct: float, n_spots: int, kmax: int) -> np.ndarray:
    """P(exactly j exposures), j = 0..kmax (kmax = kmax or more) for one cell.

    P(0) = 1 - reach exactly; mean = g/100 before lumping mass above min(n_spots, kmax).
    """
    out = np.zeros(kmax + 1)
    r = reach_pct / 100.0
    m = g_pct / 100.0
    if n_spots <= 0 or r <= 0.0 or m <= 0.0:
        out[0] = 1.0
        return out
    r = min(r, 1.0 - 1e-12)
    top = min(n_spots, kmax)
    j = np.arange(0, top + 1, dtype=float)
    kk = solve_nbd_k(m, 1.0 - r)
    if kk is not None:
        logp = (gammaln(kk + j) - gammaln(kk) - gammaln(j + 1)
                + kk * math.log(kk / (kk + m)) + j * math.log(m / (kk + m)))
        pmf = np.exp(logp)
        pmf[0] = 1.0 - r
    else:
        # zero-modified, zero-truncated Poisson with conditional mean OTS = m / r (>= 1)
        mu = max(m / r, 1.0 + 1e-9)
        lam = brentq(lambda x: x / -math.expm1(-x) - mu, 1e-9, mu + 50.0)
        logp = -lam + j * math.log(lam) - gammaln(j + 1)
        pmf = np.exp(logp) / -math.expm1(-lam) * r
        pmf[0] = 1.0 - r
    # lump the tail (>= top) into the top bucket so the pmf sums to 1 and P(0) is untouched
    pmf[top] = max(0.0, 1.0 - pmf[:top].sum())
    out[: top + 1] = pmf
    return out


def convolve_capped(a: np.ndarray, b: np.ndarray, kmax: int) -> np.ndarray:
    """Distribution of the sum of two independent counts, top bucket = kmax or more."""
    full = np.convolve(a, b)
    out = full[: kmax + 1].copy()
    out[kmax] += full[kmax + 1:].sum()
    return out


def n_plus_from_pmf(pmf: np.ndarray) -> np.ndarray:
    """Reach N+ (fractions) for N = 1..kmax from a pmf over 0..kmax."""
    tail = np.cumsum(pmf[::-1])[::-1]   # tail[j] = P(X >= j)
    return tail[1:]


# ----------------------------------------------------------------------------- full model
def cap_reach(frac: float, sum_r: float, grp_frac: float) -> float:
    """Sanity caps on combined reach 1+ (fractions): 0 <= reach <= min(1, sum of cell reaches
    (<= sum of Rmax), GRP as a fraction (reach 1+ <= impressions, i.e. OTS >= 1))."""
    return max(0.0, min(frac, sum_r, grp_frac, 1.0))


def model_reach(g_pct: np.ndarray, n_spots: np.ndarray, rmax: np.ndarray, k: np.ndarray,
                form: str, phi: np.ndarray | None, kmax: int,
                cell_pmfs: list[np.ndarray] | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the curve model for a schedule summarised per cell.

    Returns (reach_1plus_frac, reach_n_plus_frac[0..kmax-1] for N = 1..kmax, pmf[0..kmax]).
    reach 1+ = capped combination (Sainsbury or pairwise duplication) of the cell curves;
    reach N+ (N >= 2) = reach 1+ x the conditional share P(X >= N | X >= 1) of the
    independent convolution of the cell NBD frequency distributions (identical to the plain
    convolution under Sainsbury when no cap binds; guarantees reach N+ <= reach 1+).
    """
    g_pct = np.asarray(g_pct, float)
    r = np.array([curve(g, rm, kk, form) for g, rm, kk in zip(g_pct, rmax, k)]) / 100.0
    reach1 = cap_reach(combine_pairwise(r, phi), float(r.sum()), float(g_pct.sum()) / 100.0)
    pmf = np.zeros(kmax + 1)
    pmf[0] = 1.0
    for i in range(len(r)):
        cp = cell_pmfs[i] if cell_pmfs is not None else cell_frequency_pmf(g_pct[i], r[i] * 100.0, int(n_spots[i]), kmax)
        pmf = convolve_capped(pmf, cp, kmax)
    nplus = n_plus_from_pmf(pmf)
    if nplus[0] > 0:
        nplus = reach1 * nplus / nplus[0]
    else:
        nplus = np.zeros(kmax)
    nplus[0] = reach1
    nplus = np.minimum.accumulate(nplus)
    return reach1, nplus, pmf
