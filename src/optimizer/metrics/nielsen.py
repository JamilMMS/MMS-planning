"""Nielsen KSA eTAM data-type formulas as pure functions.

Sources (authoritative):
  * "eTAM Data Types - Main Data Types Explained" (Main Data Types, MDT), 93 pp.
  * "KSA TAM Data Types Glossary" (Glossary, GL), 92 pp.
Page numbers below are PHYSICAL PDF pages (1-based), not the printed TOC labels.

Conventions
-----------
* Full precision everywhere. No function here rounds; display rounding lives in
  :mod:`optimizer.metrics.rounding` and is applied only by the reporter.
* Two input levels:
    - respondent level (EXACT mode): per-person weights, viewed time, exposure counts,
      per-day weights -> every data type can be computed;
    - aggregated level (Rating Absolute, GRP Absolute, reach values already computed by
      eTAM) -> only data types that are sums/ratios of aggregates are valid. Reach,
      Reach N+, Frequency and Cume Reach need duplication information and raise
      :class:`ReachFromAggregatesError` when called on aggregates. Use
      ``optimizer.reach`` (CALIBRATED / ESTIMATE) for modelled reach.
* No config value is hard-coded: the common-weight rule, the reach minimum-seconds
  threshold and the Rate Duration Factors table are always parameters.

Respondent-level data structures
--------------------------------
``weights``      Mapping person_id -> weight (a pandas Series works too). For reach over a
                 multi-day period these are *common* weights, see :func:`common_weights`.
``exposures``    Mapping person_id -> number of schedule items (spots, dayparts,
                 programs) the person viewed for at least the reach threshold
                 (see :func:`exposure_counts`).
``daily_weights`` Mapping person_id -> Mapping day -> daily weight on the days the person
                 is in the panel (in-tab) during the period of analysis.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from typing import Any, Hashable

import numpy as np
import pandas as pd

Person = Hashable

# Rate Duration Factors: the real eTAM table (Rate Card option) has NOT been supplied yet
# (docs/OPEN_ISSUES.md, docs/DATA_SPEC.md section C.4). Until it is, only the 30" base
# length is known with certainty (factor 1.0 by definition of the 30"-equivalent).
DEFAULT_RATE_DURATION_FACTORS: dict[int, float] = {30: 1.0}

COMMON_WEIGHT_RULES = ("average", "middle_day")


class ReachFromAggregatesError(ValueError, NotImplementedError):
    """Raised when a reach-type data type is requested from aggregated inputs.

    Reach (and Reach N+, Reach N, Frequency, Cume Reach) depends on how audiences
    overlap across items. Aggregated Rating/GRP values carry no duplication
    information, so the value cannot be derived from them. Supply respondent-level
    exposures (EXACT) or use the reach engine in ``optimizer.reach`` (CALIBRATED /
    ESTIMATE, always labelled as such).
    """


_AGG_MSG = (
    "{what} cannot be computed from aggregated ratings/GRPs: they contain no audience "
    "duplication information. Provide respondent-level inputs (weights + per-person "
    "exposures, EXACT mode) or use optimizer.reach (CALIBRATED/ESTIMATE model)."
)


# ----------------------------------------------------------------------------- helpers
def _is_number(x: Any) -> bool:
    return isinstance(x, (Real, np.number)) and not isinstance(x, bool)


def _as_mapping(obj: Any, what: str) -> dict:
    """Mapping or pandas Series -> dict. Anything else is treated as aggregated input."""
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, pd.Series):
        return dict(obj.items())
    raise ReachFromAggregatesError(_AGG_MSG.format(what=what))


def _flat(values: Any) -> np.ndarray:
    """Flatten a scalar / sequence / sequence of sequences (e.g. per-day lists) to 1-D."""
    if _is_number(values):
        return np.array([float(values)])
    parts = [np.atleast_1d(np.asarray(v, dtype=float)).ravel() for v in values]
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def _total(values: Any) -> float:
    return float(_flat(values).sum())


def _check_positive(x: float, name: str) -> None:
    if not x > 0:
        raise ValueError(f"{name} must be > 0 (got {x!r})")


# ----------------------------------------------------------------------------- universe
def universe(weights: Any) -> float:
    """Universe = sum of weights of all panel members in the target demographic.

    Formula (GL p86): ``U = sum_{n in U} w_n``, w_n = weight of member n in the period.
    Accepts a sequence or a Mapping/Series of weights. For daily metrics eTAM uses the
    *average daily universe*; for reach, the *average universe in the period* (GL p9,
    p17): see :func:`average_universe`.
    """
    if isinstance(weights, Mapping):
        return _total(list(weights.values()))
    if _is_number(weights):
        return float(weights)
    return float(np.asarray(weights, dtype=float).sum())  # list / ndarray / Series


def average_universe(period_universes: Sequence[float]) -> float:
    """Average of daily (or weekly) universes over the period (GL p9, p17, p21)."""
    arr = _flat(period_universes)
    if arr.size == 0:
        raise ValueError("average_universe: empty input")
    return float(arr.mean())


# ----------------------------------------------------------------------------- ratings
def rating_abs(weights: Any, time_viewed: Any, duration: Any) -> float:
    """Rating Absolute (average-minute audience), respondent level.

    Formula (GL p8; MDT p5-6): ``sum_{n in V} (w_n * t_n) / D``
      V = people viewing >= 1 second of the event, w_n = DAILY weight, t_n = time viewed,
      D = length of the event. Multi-day / multi-channel: pass per-day lists for
      ``weights`` and ``time_viewed`` and either the total analysed duration or a list of
      per-day(-per-channel) durations, which are summed (MDT p6 note: two channels x two
      days x 60 min -> D = 60*4). ``time_viewed`` and ``duration`` must share a unit.
    """
    w = _flat(weights)
    t = _flat(time_viewed)
    if w.shape != t.shape:
        raise ValueError(f"weights ({w.size}) and time_viewed ({t.size}) differ in length")
    d = _total(duration)
    _check_positive(d, "duration")
    viewing = t > 0  # V = viewers with >= 1 s; zero-time rows contribute nothing anyway
    return float(np.sum(w[viewing] * t[viewing]) / d)


def rating_abs_combine(ratings: Sequence[float], durations: Sequence[float]) -> float:
    """Combine Rating Absolute values of several events/days (aggregated inputs).

    GL p8: "In case of multiple days or events, the total Rating Absolute is the average,
    weighted by the duration of each event, of the individual Rating Absolute values."
    ``sum(R_e * D_e) / sum(D_e)``.
    """
    r = _flat(ratings)
    d = _flat(durations)
    if r.shape != d.shape:
        raise ValueError("ratings and durations differ in length")
    _check_positive(float(d.sum()), "sum of durations")
    return float(np.sum(r * d) / d.sum())


def rating_abs_from_break_trp(trp_abs: Any, break_seconds: Any) -> Any:
    """Per-break audience from the eTAM break file (project invariant, PRELIMINARY_FINDINGS).

    In the eTAM break export, TRP Absolute = Rating Absolute x break length in minutes, so
    ``rating_abs = TRP_Absolute * 60 / break_seconds``. Vectorised (numpy/pandas).
    """
    sec = np.asarray(break_seconds, dtype=float)
    if np.any(sec <= 0):
        raise ValueError("break_seconds must be > 0")
    out = np.asarray(trp_abs, dtype=float) * 60.0 / sec
    if isinstance(trp_abs, pd.Series):  # keep pandas index
        return trp_abs.__class__(out, index=trp_abs.index)
    return float(out) if out.ndim == 0 else out


def rating_pct(rating_abs_value: float, universe_value: float) -> float:
    """Rating % = Rating Absolute / Universe x 100 (GL p9; MDT p7).

    Universe = average daily demographic universe.
    """
    _check_positive(universe_value, "universe")
    return float(rating_abs_value) / float(universe_value) * 100.0


def trp_abs(rating_abs_values: Iterable[float]) -> float:
    """TRP Absolute = sum of Rating Absolute over the events E (GL p10)."""
    return _total(list(rating_abs_values))


def trp_pct(rating_pct_values: Iterable[float]) -> float:
    """TRP % = sum of Rating % over the events E (GL p11; MDT p8, p46)."""
    return _total(list(rating_pct_values))


def average_daily_trp_abs(daily_trp_abs: Sequence[float]) -> float:
    """Average Daily TRP Absolute = sum of daily TRP Absolute / number of days (MDT p36)."""
    arr = _flat(daily_trp_abs)
    if arr.size == 0:
        raise ValueError("empty input")
    return float(arr.mean())


def average_weekly_trp_abs(weekly_trp_abs: Sequence[float]) -> float:
    """Average Weekly TRP Absolute = sum of weekly TRP Absolute / number of weeks (MDT p38).

    NB MDT p38's worked example is not reproducible (5 values listed, divided by 4, and
    the printed result matches neither); see docs/NIELSEN_DEFINITIONS.md.
    """
    arr = _flat(weekly_trp_abs)
    if arr.size == 0:
        raise ValueError("empty input")
    return float(arr.mean())


def average_daily_trp_pct(avg_daily_trp_abs: float, universe_value: float) -> float:
    """Average Daily TRP % = Average Daily TRP Absolute / Universe x 100 (MDT p37)."""
    return rating_pct(avg_daily_trp_abs, universe_value)


def average_weekly_trp_pct(avg_weekly_trp_abs: float, universe_value: float) -> float:
    """Average Weekly TRP % = Average Weekly TRP Absolute / Universe x 100 (MDT p39).

    Erratum: MDT p39 prints 15,628.01 (copied from the daily page p37); the stated inputs
    give ~113,075.
    """
    return rating_pct(avg_weekly_trp_abs, universe_value)


def share_of_audience(rating_abs_value: float, total_tv_rating_abs: float) -> float:
    """Share of Audience % = Rating Abs / sum of Rating Abs over all sources S x 100.

    GL p12 (S = all measured channels and non-broadcast activities = Total TV); MDT p10.
    """
    _check_positive(total_tv_rating_abs, "total_tv_rating_abs")
    return float(rating_abs_value) / float(total_tv_rating_abs) * 100.0


def share_to_selected(rating_abs_value: float, selected_rating_abs: Sequence[float]) -> float:
    """Share to Selected % = Rating Abs / sum of Rating Abs of the selected channels C x 100.

    GL p13; MDT p11. ``selected_rating_abs`` must INCLUDE the channel itself (MDT p11 lists
    1,625 in the denominator).
    """
    denom = _total(selected_rating_abs)
    _check_positive(denom, "sum of selected ratings")
    return float(rating_abs_value) / denom * 100.0


def profile_pct(rating_abs_target: float, rating_abs_base: float) -> float:
    """Profile % = Rating Abs(target) / Rating Abs(base demo) x 100 (GL p14)."""
    _check_positive(rating_abs_base, "rating_abs_base")
    return float(rating_abs_target) / float(rating_abs_base) * 100.0


def profile_index(rating_pct_target: float, rating_pct_base: float) -> float:
    """Profile Index = Rating %(target) / Rating %(base demo) x 100 (GL p15)."""
    _check_positive(rating_pct_base, "rating_pct_base")
    return float(rating_pct_target) / float(rating_pct_base) * 100.0


def loyalty_pct(rating_abs_loyals: float, rating_abs_all: float) -> float:
    """Loyalty % = Rating Abs(viewers above the loyalty threshold) / Rating Abs(all) x 100.

    GL p64. The threshold is an eTAM Options setting (MDT p77-78: Manual / Auto / LMH); no
    numeric worked example exists in the PDFs.
    """
    _check_positive(rating_abs_all, "rating_abs_all")
    return float(rating_abs_loyals) / float(rating_abs_all) * 100.0


# ----------------------------------------------------------------------------- time
def total_duration(durations: Sequence[float]) -> float:
    """Total Duration = sum of event lengths D_n (GL p36; MDT p35)."""
    return _total(durations)


def tsv_universe_daily(rating_abs_value: float, duration: float, universe_value: float) -> float:
    """TSV Universe (Daily) = Rating Abs / Universe x D (GL p32; MDT p28-29).

    Multi-day: pass the average of the daily Rating Absolute values (MDT p29) and the
    per-day event length D.
    """
    _check_positive(universe_value, "universe")
    return float(rating_abs_value) * float(duration) / float(universe_value)


def tsv_viewers_daily(rating_abs_value: float, duration: float, avg_daily_reach: float) -> float:
    """TSV Viewers (Daily) = Rating Abs / Average Daily Reach x D (GL p31; MDT p30-31, p65)."""
    _check_positive(avg_daily_reach, "avg_daily_reach")
    return float(rating_abs_value) * float(duration) / float(avg_daily_reach)


def tsv_universe_weekly(rating_abs_values: Sequence[float], durations: Sequence[float],
                        universe_value: float) -> float:
    """TSV Universe (Weekly) = sum_n (Rating Abs_n x D_n) / Universe (GL p34; MDT p32).

    NB for a multi-week period this is the TOTAL time over all weeks (MDT p32 multiplies
    the average weekly rating by 60 min x 21 days), not a per-week average.
    """
    r, d = _flat(rating_abs_values), _flat(durations)
    if r.shape != d.shape:
        raise ValueError("rating_abs_values and durations differ in length")
    _check_positive(universe_value, "universe")
    return float(np.sum(r * d) / universe_value)


def tsv_viewers_weekly(rating_abs_values: Sequence[float], durations: Sequence[float],
                       avg_weekly_reach: float) -> float:
    """TSV Viewers (Weekly) = sum_n (Rating Abs_n x D_n) / Average Weekly Reach (GL p33; MDT p33)."""
    r, d = _flat(rating_abs_values), _flat(durations)
    if r.shape != d.shape:
        raise ValueError("rating_abs_values and durations differ in length")
    _check_positive(avg_weekly_reach, "avg_weekly_reach")
    return float(np.sum(r * d) / avg_weekly_reach)


def completion_rate(rating_abs_value: float, avg_daily_reach: float) -> float:
    """Completion Rate % = Rating Abs / Average Daily Reach x 100 (GL p35; MDT p34, p66)."""
    _check_positive(avg_daily_reach, "avg_daily_reach")
    return float(rating_abs_value) / float(avg_daily_reach) * 100.0


# ----------------------------------------------------------------------------- weights
def common_weights(daily_weights: Any, rule: str, period_days: Sequence | None = None
                   ) -> dict[Person, float]:
    """Common weight of each person over the period of analysis (reach-type data types).

    GL p16/p22/p26/p28: "calculated with a common weight across the entire period of
    analysis (average or middle day weight, depending on the official calculation rules in
    place)". The Glossary does NOT say which rule KSA uses; every multi-day worked example
    in MDT (p18, p21, p24, p27, p58, p64, p84, p87, p90) uses the AVERAGE of the person's
    daily weights, and p24/p62/p64/p87/p90 label them "AVERAGE WEIGHTS".

    Parameters
    ----------
    daily_weights : Mapping person -> Mapping day -> weight, or a DataFrame with columns
        ``person, day, weight``. Supply the days the person is in the panel (in-tab).
        The PDF examples only ever show weights on days the person viewed, so whether eTAM
        averages over in-tab days or viewing days cannot be determined from the PDFs.
    rule : 'average' | 'middle_day' — take it from ``cfg['reach']['common_weight_rule']``.
        * average    -> mean of the person's supplied daily weights;
        * middle_day -> the person's weight on the middle day of the period
          (``sorted(period_days)[(len-1)//2]``, i.e. the lower middle for an even count).
          A person with no weight on that day gets weight 0 (not in the middle-day sample).
          This tie-break and exclusion are implementation choices, not stated in the PDFs.
    period_days : optional explicit list of days in the period (middle_day only);
        defaults to the union of days present in ``daily_weights``.
    """
    if rule not in COMMON_WEIGHT_RULES:
        raise ValueError(f"unknown common_weight_rule {rule!r}; expected one of {COMMON_WEIGHT_RULES}")
    if isinstance(daily_weights, pd.DataFrame):  # columns person, day, weight
        dw: dict = {}
        for p, d, w in daily_weights[["person", "day", "weight"]].itertuples(index=False):
            dw.setdefault(p, {})[d] = float(w)
    else:
        dw = {p: dict(v) for p, v in daily_weights.items()}
    if rule == "average":
        return {p: float(np.mean(list(days.values()))) for p, days in dw.items() if days}
    days_all = sorted(period_days) if period_days is not None else sorted({d for v in dw.values() for d in v})
    if not days_all:
        raise ValueError("no days in period")
    middle = days_all[(len(days_all) - 1) // 2]
    return {p: float(days.get(middle, 0.0)) for p, days in dw.items()}


def common_weight_rule_from_config(cfg: Mapping) -> str:
    """Read the common-weight rule from the loaded config (``reach.common_weight_rule``)."""
    return str(cfg["reach"]["common_weight_rule"])


def exposure_counts(viewing: Iterable[tuple[Person, Hashable, float]], min_seconds: float
                    ) -> dict[Person, int]:
    """Count, per person, the schedule items viewed for at least ``min_seconds``.

    ``viewing`` yields (person, item_id, seconds_viewed) records; several records for the
    same (person, item) are summed first. An item (spot / daypart / program airing) counts
    once per person. ``min_seconds`` is the eTAM Options-filter reach threshold (GL p16:
    "for at least a specified minimum amount of seconds (as defined in the Options filter)");
    its KSA value is not stated in the PDFs, so it must be passed explicitly.
    """
    per_item: dict[tuple, float] = {}
    for person, item, sec in viewing:
        per_item[(person, item)] = per_item.get((person, item), 0.0) + float(sec)
    counts: dict[Person, int] = {}
    for (person, _item), sec in per_item.items():
        if sec > 0 and sec >= min_seconds:
            counts[person] = counts.get(person, 0) + 1
    return counts


def exposures_from_items(items: Sequence[Iterable[Person]]) -> dict[Person, int]:
    """Exposure counts from a list of items, each given as the set of persons who viewed it
    (already above threshold). Convenience for schedule tables like MDT p23."""
    counts: dict[Person, int] = {}
    for viewers in items:
        for p in set(viewers):
            counts[p] = counts.get(p, 0) + 1
    return counts


# ----------------------------------------------------------------------------- reach
def _respondent(weights: Any, exposures: Any, what: str) -> tuple[dict, dict]:
    if exposures is None or _is_number(weights):
        raise ReachFromAggregatesError(_AGG_MSG.format(what=what))
    w = _as_mapping(weights, what)
    f = _as_mapping(exposures, what)
    missing = [p for p, k in f.items() if k > 0 and p not in w]
    if missing:
        raise KeyError(f"{what}: no weight for exposed persons {missing[:5]}")
    return w, f


def reach_from_aggregates(*_args: Any, **_kwargs: Any) -> float:
    """Always raises: reach cannot be derived from aggregated ratings/GRPs."""
    raise ReachFromAggregatesError(_AGG_MSG.format(what="Reach"))


def reach_n_plus(weights: Any, exposures: Any, n: int) -> float:
    """Reach N+ = sum of common weights of people with >= N exposures (GL p26; MDT p23-24,
    p61-62, p89-90). ``n=1`` gives Unduplicated Reach."""
    if n < 1:
        raise ValueError("n must be >= 1")
    w, f = _respondent(weights, exposures, f"Reach {n}+")
    return float(sum(w[p] for p, k in f.items() if k >= n))


def reach_n(weights: Any, exposures: Any, n: int) -> float:
    """Reach N (exact) = sum of common weights of people with exactly N exposures (GL p28;
    MDT p60 - whose printed total is an erratum, see docs)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    w, f = _respondent(weights, exposures, f"Reach {n}")
    return float(sum(w[p] for p, k in f.items() if k == n))


def unduplicated_reach(weights: Any, exposures: Any = None) -> float:
    """Unduplicated Reach = sum of common weights w_n over people viewing >= 1 item for at
    least the threshold (GL p16; MDT p17-19, p83-84).

    ``exposures`` may be a Mapping person -> count, or an iterable of person ids reached.
    Multi-day: build ``weights`` with :func:`common_weights`.
    """
    if exposures is not None and not isinstance(exposures, (Mapping, pd.Series)) \
            and not _is_number(exposures):
        exposures = {p: 1 for p in exposures}
    return reach_n_plus(weights, exposures, 1)


def reach_pct(reach_value: float, universe_value: float) -> float:
    """Any reach-type % (Unduplicated / Cume / N+ / N / Incremental / Duplication /
    Exclusive / Average Daily / Average Weekly) = reach / Universe x 100 (GL p10, p17, p23,
    p25, p27, p29). Universe: average universe in the period (reach) or average daily /
    weekly universe (average daily / weekly reach)."""
    _check_positive(universe_value, "universe")
    return float(reach_value) / float(universe_value) * 100.0


def frequency(weights: Any, exposures: Any) -> float:
    """Average Frequency = sum(w_n * f_n) / sum(w_n) over people with f_n >= 1, using average
    (common) weights in the period (GL p30; MDT p26-27, p64)."""
    w, f = _respondent(weights, exposures, "Frequency")
    num = sum(w[p] * k for p, k in f.items() if k >= 1)
    den = sum(w[p] for p, k in f.items() if k >= 1)
    _check_positive(den, "reach")
    return float(num / den)


def frequency_from_totals(weighted_exposures: float, reach_value: float) -> float:
    """Frequency from already-aggregated totals sum(w*f) and sum(w) (MDT p27 bottom line)."""
    _check_positive(reach_value, "reach")
    return float(weighted_exposures) / float(reach_value)


def average_daily_reach(daily: Sequence[Any]) -> float:
    """Average Daily Reach = mean over days of the daily reach computed with DAILY weights
    (GL p11; MDT p12-13, p49-50).

    Each element is either an already-computed daily reach (aggregated, valid: it is a plain
    average) or a Mapping person -> daily weight of that day's viewers (respondent level).
    """
    if len(daily) == 0:
        raise ValueError("empty input")
    vals = [_total(list(d.values())) if isinstance(d, Mapping) else float(d) for d in daily]
    return float(np.mean(vals))


def average_weekly_reach(weekly: Sequence[Any]) -> float:
    """Average Weekly Reach = mean over weeks of the weekly reach computed with WEEKLY
    weights (GL p13; MDT p15, p52).

    Each element is either a weekly reach value, or a Mapping person -> weekly weight, or
    person -> list of that person's daily weights in the week (weekly weight = mean of the
    daily weights, as in the MDT p15 "AVERAGE WEIGHTS WEEK n" column).
    """
    if len(weekly) == 0:
        raise ValueError("empty input")
    vals = []
    for wk in weekly:
        if isinstance(wk, Mapping):
            vals.append(sum(float(np.mean(_flat(v))) for v in wk.values()))
        else:
            vals.append(float(wk))
    return float(np.mean(vals))


def cume_reach_rf(lines: Sequence[Iterable[Person]], weights: Any) -> list[float]:
    """Cume Reach (RF): running Unduplicated Reach line by line (GL p22; MDT p20-21, p57-58,
    p86-87). The last line equals Unduplicated Reach.

    Parameters
    ----------
    lines : sequence of report lines, each the iterable of person ids reached by that line.
    weights : a single Mapping person -> common weight used for every line (spot reports,
        MDT p87), OR a sequence (one per line) of Mappings giving the weights to apply when
        evaluating line n (MDT p21/p58 show line 1 of a two-day report evaluated with the
        first day's weights and the last line with the two-day average weights).
        The PDFs are inconsistent on which applies; see docs/NIELSEN_DEFINITIONS.md.
    Aggregated per-line reach numbers cannot be cumulated (duplication unknown): passing
    numbers raises :class:`ReachFromAggregatesError`; use
    :func:`cume_reach_rf_from_increments` if you hold Incremental Reach values.
    """
    if any(_is_number(x) for x in lines):
        raise ReachFromAggregatesError(_AGG_MSG.format(what="Cume Reach (RF)"))
    if isinstance(weights, (Mapping, pd.Series)):
        per_line = [_as_mapping(weights, "Cume Reach (RF)")] * len(lines)
    else:
        per_line = [_as_mapping(w, "Cume Reach (RF)") for w in weights]
        if len(per_line) != len(lines):
            raise ValueError("need one weights mapping per line")
    seen: set = set()
    out = []
    for viewers, w in zip(lines, per_line):
        seen |= set(viewers)
        out.append(float(sum(w[p] for p in seen)))
    return out


def cume_reach_rf_from_increments(increments: Sequence[float]) -> list[float]:
    """Cume Reach (RF) lines from Incremental Reach values (valid on aggregates)."""
    return [float(x) for x in np.cumsum(_flat(increments))]


def incremental_reach(cume_lines: Sequence[float]) -> list[float]:
    """Incremental Reach_n = Cume Reach(RF)_n - Cume Reach(RF)_(n-1) (GL p24); the line above
    the first line is 0."""
    arr = _flat(cume_lines)
    return [float(x) for x in np.diff(np.concatenate([[0.0], arr]))]


def duplication_cume_reach(weights: Any, viewers_1: Iterable[Person], viewers_2: Iterable[Person]) -> float:
    """Duplication Cume Reach = sum of common weights over V1 intersect V2 (GL p76; MDT p74)."""
    w = _as_mapping(weights, "Duplication Cume Reach")
    return float(sum(w[p] for p in set(viewers_1) & set(viewers_2)))


def exclusive_cume_reach(weights: Any, viewers_1: Iterable[Person], viewers_2: Iterable[Person]) -> float:
    """Exclusive Cume Reach = sum of common weights over V1 minus V2 (GL p78; MDT p75-76)."""
    w = _as_mapping(weights, "Exclusive Cume Reach")
    return float(sum(w[p] for p in set(viewers_1) - set(viewers_2)))


# ----------------------------------------------------------------------------- spots
def grp_abs(spots: Sequence[Any]) -> float:
    """GRP Absolute = sum over spots s of sum over viewers n of w_{n,s} (GL p38; MDT p81).

    w_{n,s} = DAILY weight of viewer n on the day spot s aired. Each element of ``spots`` is
    either the list of viewer daily weights of that spot (respondent level) or the spot's
    audience (Rating Absolute of the spot, aggregated) - both are valid since GRP is a sum.
    """
    return _total([s for s in spots])


def grp_abs_from_exposures(daily_weights: Any, exposures: Any) -> float:
    """GRP Absolute for a single-day schedule from per-person exposure counts:
    sum(w_n * f_n) (MDT p92 numerator). For multi-day schedules use :func:`grp_abs` with
    per-spot daily weights, since each exposure uses the weight of the spot's day."""
    w = _as_mapping(daily_weights, "GRP Absolute")
    f = _as_mapping(exposures, "GRP Absolute")
    return float(sum(w[p] * k for p, k in f.items()))


def grp_pct(grp_abs_value: Any, universe_value: float) -> float:
    """GRP % = GRP Absolute / Universe x 100, Universe = average daily universe (GL p39;
    MDT p82). ``grp_abs_value`` may be a per-spot list (summed)."""
    _check_positive(universe_value, "universe")
    return _total(grp_abs_value) / float(universe_value) * 100.0


def ots(grp_abs_value: Any, reach_value: float) -> float:
    """OTS = TRP (GRP) Absolute / Unduplicated Reach (GL p40; MDT p92).

    ``reach_value`` must itself come from respondent data or a labelled model.
    """
    _check_positive(reach_value, "reach")
    return _total(grp_abs_value) / float(reach_value)


def cpm(cost: Any, grp_abs_value: Any) -> float:
    """Cost Per Thousand = sum(Cost_n) / sum(GRP Absolute_n / 1000) (GL p44)."""
    g = _total(grp_abs_value)
    _check_positive(g, "GRP Absolute")
    return _total(cost) / (g / 1000.0)


def cost_per_rating_pct(cost: Any, grp_pct_value: Any) -> float:
    """Spot Cost per Rating % (CPP) = sum(Cost_n) / sum(GRP %_n) (GL p45)."""
    g = _total(grp_pct_value)
    _check_positive(g, "GRP %")
    return _total(cost) / g


# ----------------------------------------------------------------------------- weighted (30")
def eq_factor(length_sec: Any, factors: Mapping[int, float]) -> np.ndarray:
    """Rate duration factor(s) for spot length(s) from a Rate Duration Factors table.

    ``factors`` maps spot length in seconds -> EqFactor (GL p47-51; Rate Card option of
    eTAM). The real KSA table is pending (OPEN_ISSUES #10 / DATA_SPEC C.4); callers pass
    :data:`DEFAULT_RATE_DURATION_FACTORS` (30" only) until it arrives. Unknown lengths
    raise KeyError rather than being interpolated.
    """
    lengths = np.atleast_1d(np.asarray(length_sec))
    out = []
    for L in lengths:
        key = int(L)
        if key not in factors:
            raise KeyError(f"no Rate Duration Factor for {key}s spots (table has {sorted(factors)}); "
                           "supply the eTAM Rate Duration Factors table (DATA_SPEC C.4)")
        out.append(float(factors[key]))
    return np.array(out)


def weighted_grp_abs(grp_abs_per_spot: Sequence[float], lengths_sec: Sequence[int],
                     factors: Mapping[int, float]) -> float:
    """Weighted Rating Absolute (30"-equivalent GRP Abs) = sum(GRP Abs_n x EqFactor_n) (GL p47)."""
    g = _flat(grp_abs_per_spot)
    return float(np.sum(g * eq_factor(lengths_sec, factors)))


def weighted_grp_pct(grp_pct_per_spot: Sequence[float], lengths_sec: Sequence[int],
                     factors: Mapping[int, float]) -> float:
    """Weighted Rating % (30"-equivalent GRP %) = sum(GRP %_n x EqFactor_n) (GL p48)."""
    g = _flat(grp_pct_per_spot)
    return float(np.sum(g * eq_factor(lengths_sec, factors)))


def weighted_spot_cost(costs: Sequence[float], lengths_sec: Sequence[int],
                       factors: Mapping[int, float]) -> float:
    """Weighted Spot Cost = sum(Cost_n / EqFactor_n) (GL p49 - note DIVISION, whereas the
    weighted ratings multiply by EqFactor)."""
    c = _flat(costs)
    return float(np.sum(c / eq_factor(lengths_sec, factors)))


def weighted_cpm(costs: Sequence[float], grp_abs_per_spot: Sequence[float], lengths_sec: Sequence[int],
                 factors: Mapping[int, float]) -> float:
    """Weighted Cost Per Thousand = sum(Weighted Spot Cost_n) / sum(GRP Abs_n / 1000) (GL p50).
    The denominator is the ACTUAL (unweighted) GRP Absolute as printed in GL p50."""
    return cpm(weighted_spot_cost(costs, lengths_sec, factors), grp_abs_per_spot)


def weighted_cost_per_rating_pct(costs: Sequence[float], grp_pct_per_spot: Sequence[float],
                                 lengths_sec: Sequence[int], factors: Mapping[int, float]) -> float:
    """Weighted Spot Cost per Rating % = sum(Weighted Spot Cost_n) / sum(GRP %_n).

    GL p51 prints the denominator as sum(GRP %_n / 1000); that looks copied from the CPM
    formula (GL p50) and contradicts the unweighted Spot Cost per Rating % (GL p45, no
    /1000). This implementation follows GL p45 (no /1000); flagged as ambiguous in
    docs/NIELSEN_DEFINITIONS.md.
    """
    return cost_per_rating_pct(weighted_spot_cost(costs, lengths_sec, factors), grp_pct_per_spot)
