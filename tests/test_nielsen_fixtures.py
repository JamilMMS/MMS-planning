"""One test per row of tests/fixtures/nielsen_fixtures.csv (parametrised by fixture_id).

Each fixture row maps to a small compute function registered with ``@fixture("Fxx")``;
its docstring states what is rebuilt and, for ``status=erratum`` rows, the PDF
discrepancy. Erratum rows assert the CORRECTED math, never the printed number.
Tolerance: 0.5% relative (plus 1e-9 absolute for zero values) unless the row's note
contains ``tol=<fraction>``.

Page numbers are physical PDF pages of the Nielsen KSA eTAM "Main Data Types" (MDT)
and "Data Types Glossary" (GL) documents.
"""
from __future__ import annotations

import copy
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from optimizer.config import load_config
from optimizer.metrics import (
    DEFAULT_RATE_DURATION_FACTORS,
    ReachFromAggregatesError,
    average_daily_reach,
    average_daily_trp_abs,
    average_daily_trp_pct,
    average_weekly_reach,
    average_weekly_trp_pct,
    common_weights,
    completion_rate,
    cost_per_rating_pct,
    cpm,
    cume_reach_rf,
    cume_reach_rf_from_increments,
    display_round,
    duplication_cume_reach,
    exclusive_cume_reach,
    exposure_counts,
    exposures_from_items,
    frequency,
    frequency_from_totals,
    grp_abs,
    grp_abs_from_exposures,
    grp_pct,
    incremental_reach,
    loyalty_pct,
    ots,
    rating_abs,
    rating_abs_combine,
    rating_abs_from_break_trp,
    rating_pct,
    reach_from_aggregates,
    reach_n,
    reach_n_plus,
    reach_pct,
    share_of_audience,
    share_to_selected,
    total_duration,
    trp_pct,
    tsv_universe_daily,
    tsv_universe_weekly,
    tsv_viewers_daily,
    tsv_viewers_weekly,
    unduplicated_reach,
    universe,
    weighted_cost_per_rating_pct,
    weighted_cpm,
    weighted_grp_abs,
    weighted_grp_pct,
    weighted_spot_cost,
)

FIXTURES = Path(__file__).parent / "fixtures" / "nielsen_fixtures.csv"
DEFAULT_TOL = 0.005
with open(FIXTURES, newline="", encoding="utf-8") as fh:
    ROWS = list(csv.DictReader(fh))

COMPUTE: dict = {}


def fixture(fid):
    def deco(fn):
        COMPUTE[fid] = fn
        return fn
    return deco


# ------------------------------------------------------------------ input parsing
def _scalar(x: str):
    x = x.strip()
    if "|" in x:
        return [p.strip() for p in x.split("|")]
    if x == "_":
        return None
    try:
        return float(x)
    except ValueError:
        return x


def _value(v: str):
    v = v.strip()
    if v.startswith("["):
        return [_scalar(x) for x in v[1:-1].split(",")]
    if v.startswith("{"):
        out = {}
        for kv in v[1:-1].split(","):
            k, x = kv.split(":")
            vals = [_scalar(p) for p in x.split("/")]
            out[k.strip()] = vals if len(vals) > 1 else vals[0]
        return out
    return _scalar(v)


def parse_inputs(s: str) -> dict:
    """Mini-grammar used by rows F24+: ``key=value; key=value``. Values: number,
    ``[list]``, ``{P:w,...}`` (``/`` separates per-day weights, ``_`` = not in panel that
    day), and ``A|B|C`` = the set of persons viewing an item."""
    out = {}
    for part in s.split(";"):
        if part.strip():
            k, v = part.split("=", 1)
            out[k.strip()] = _value(v)
    return out


def parse_expected(s: str):
    s = s.strip()
    if s.startswith("["):
        return [float(x) for x in s[1:-1].split(",")]
    if "=" in s:
        return {k.strip(): float(v) for k, v in (p.split("=") for p in s.split(";"))}
    return float(s)


def _persons(x):
    return x if isinstance(x, list) else [x]


def _daily(dw: dict) -> dict:
    """{P: [w_day0, w_day1 or None]} -> {P: {day: w}} (days where the person is shown)."""
    out = {}
    for p, ws in dw.items():
        ws = ws if isinstance(ws, list) else [ws]
        out[p] = {d: w for d, w in enumerate(ws) if w is not None}
    return out


def _cw(inp: dict) -> dict:
    return common_weights(_daily(inp["daily_weights"]), rule=inp["rule"])


def _reach_levels(weights, exposures, keys, fn):
    return {k: fn(weights, exposures, int(k.rstrip("+"))) for k in keys}


def _two_day_program(inp_str: str):
    """Parse the MDT p65/p66 input string (hand-coded format)."""
    days = re.findall(r"weights=\[([^\]]*)\] minutes=\[([^\]]*)\]", inp_str)
    w = [[float(x) for x in a.split(",")] for a, _ in days]
    t = [[float(x) for x in b.split(",")] for _, b in days]
    d = float(re.search(r"duration=(\d+)", inp_str).group(1))
    reach = [float(x) for x in re.search(r"daily_reach=\[([^\]]*)\]", inp_str).group(1).split(",")]
    daily_ratings = [rating_abs(wi, ti, d) for wi, ti in zip(w, t)]
    return float(np.mean(daily_ratings)), d, average_daily_reach(reach)


# ------------------------------------------------------------------ F01-F23 (hand-coded)
@fixture("F01")
def f01(_):
    """Rating Absolute single day, MDT p5: sum(w*t)/D = 97,500/60."""
    return rating_abs([900, 1100, 1000, 800], [15, 10, 45, 35], 60)


@fixture("F02")
def f02(_):
    """Rating Absolute multi day, MDT p6: per-day lists, D = 60 + 60."""
    return rating_abs([[900, 1100, 1000, 800], [950, 1200, 1300, 900]],
                      [[15, 10, 45, 35], [20, 40, 9, 36]], [60, 60])


@fixture("F03")
def f03(_):
    """Rating %, MDT p7."""
    return rating_pct(1625, 20_000_000)


@fixture("F04")
def f04(_):
    """Share of Audience %, MDT p10 (printed 2.14 = rounded)."""
    return share_of_audience(1625, 76_000)


@fixture("F05")
def f05(_):
    """Share to Selected %, MDT p11 (denominator includes the channel itself)."""
    return share_to_selected(1625, [1540, 3960, 2105, 3640, 2835, 3005, 1625, 450, 5255, 1060,
                                    1200, 1105, 2010, 2350, 905])


@fixture("F06")
def f06(_):
    """Average Daily Reach from daily reach values (aggregated path is valid), MDT p13."""
    return average_daily_reach([3800, 4350])


@fixture("F07")
def f07(_):
    """Average Weekly Reach from weekly reach values, MDT p15."""
    return average_weekly_reach([4900, 5650])


@fixture("F08")
def f08(_):
    """Frequency single day, MDT p26: sum(w*f)/sum(w) = 7,150/5,050."""
    w = dict(enumerate([900, 1200, 850, 700, 1400]))
    f = dict(enumerate([2, 2, 1, 1, 1]))
    return frequency(w, f)


@fixture("F09")
def f09(_):
    """Frequency multi day from totals, MDT p27: 12,450/6,150."""
    return frequency_from_totals(12450, 6150)


@fixture("F10")
def f10(_):
    """TSV Universe (Daily), MDT p28: Rating Abs x D / Universe."""
    return tsv_universe_daily(1158.33, 60, 20_000_000)


@fixture("F11")
def f11(_):
    """TSV Viewers (Daily), MDT p30: Rating Abs x D / Average Daily Reach. PDF prints 23.16
    because it uses the rating rounded to 1,158 (1,158 x 60 / 3,000 = 23.16 exactly)."""
    return tsv_viewers_daily(1158.33, 60, 3000)


@fixture("F12")
def f12(_):
    """Completion Rate, MDT p34: 1,625/3,800 x 100."""
    return completion_rate(1625, 3800)


@fixture("F13")
def f13(_):
    """Average Daily TRP %, MDT p37."""
    return average_daily_trp_pct(3_178_009_782, 20_335_350)


@fixture("F14")
def f14(_):
    """ERRATUM - Average Weekly TRP %, MDT p39.

    The PDF prints 15,628.01, which is the DAILY result copied from p37. With the stated
    inputs (22,994,225,336 / 20,335,350 x 100) the correct value is ~113,075.1. (The
    22,994,225,336 input itself is not reproducible from the five weekly values on p38.)
    """
    return average_weekly_trp_pct(22_994_225_336, 20_335_350)


@fixture("F15")
def f15(_):
    """Program Rating Absolute single day, MDT p43: 64,000/30."""
    return rating_abs([900, 1200, 1100, 1000], [15, 20, 15, 10], 30)


@fixture("F16")
def f16(_):
    """Program Rating Absolute multi day, MDT p44: sum(w*t) = 112,000 over D = 30 + 30.
    The row gives only the total, passed as a single weight-minute product."""
    return rating_abs([1.0], [112_000], [30, 30])


@fixture("F17")
def f17(_):
    """Program Share of Audience %, MDT p47."""
    return share_of_audience(2133, 64_000)


@fixture("F18")
def f18(_):
    """Reach N+ single day, rebuilt from the MDT p23 viewing table (image read manually).

    Persons (average weights): A=700 (teal), B=1,000 (purple), C=800 (red), D=900 (orange).
    Dayparts and viewing (seconds from the printed viewing times, inclusive):
      08:00-11:59  08:10:00-08:45:59 (2,160 s)  A, B, C
      12:00-15:59  12:20:00-12:24:59 (300 s)    B, C
      16:00-17:59  16:30:00-17:59:59 (5,400 s)  C, D
      18:00-19:59  19:30:00-19:45:59 (960 s)    B, D
    Exposures: A=1, B=3, C=3, D=2 -> 1+ = 3,400, 2+ = 2,700, 3+ = 1,800.
    The reach threshold is not stated in the PDF; 1 s is used (every viewing qualifies).
    """
    w = {"A": 700, "B": 1000, "C": 800, "D": 900}
    dayparts = [(2160, "ABC"), (300, "BC"), (5400, "CD"), (960, "BD")]
    viewing = [(p, i, sec) for i, (sec, ps) in enumerate(dayparts) for p in ps]
    f = exposure_counts(viewing, min_seconds=1)
    assert f == {"A": 1, "B": 3, "C": 3, "D": 2}
    return {f"{n}+": reach_n_plus(w, f, n) for n in (1, 2, 3)}


@fixture("F19")
def f19(_):
    """GRP Absolute on spots with the DAILY weight of each spot's day, MDT p81."""
    return grp_abs([[900, 1200, 1100, 1000], [950, 1100, 1300, 1050]])


@fixture("F20")
def f20(_):
    """GRP %, MDT p82: 532,040 / 20,000,000 x 100."""
    return grp_pct([60800, 120800, 150000, 200440], 20_000_000)


@fixture("F21")
def f21(_):
    """ERRATUM (rounding) - OTS, MDT p92: 15,300 / 3,800 = 4.0263.

    The PDF prints 4.02: that is TRUNCATION to 2 dp (round-half-up would give 4.03). The
    test asserts the full-precision value; display rounding is a reporter concern
    (config.rounding).
    """
    return ots(15300, 3800)


@fixture("F22")
def f22(_):
    """Unduplicated Reach % multi day, MDT p19: 4,775 / 20,000,000 x 100."""
    return reach_pct(4775, 20_000_000)


@fixture("F23")
def f23(_):
    """Project invariant (PRELIMINARY_FINDINGS): eTAM break-file TRP Absolute = Rating Abs x
    break minutes, so rating_abs = 125,241 x 60 / 140 (break 03:06:09-03:08:28, 140 s
    inclusive) = 53,674.7."""
    return rating_abs_from_break_trp(125241, 140)


# ------------------------------------------------------------------ F24+ (generic grammar)
@fixture("F24")
def f24(inp):
    """ERRATUM - TRP % daypart, MDT p8. The PDF prints 255.33, but its own 24 formula terms
    sum to 255.12 (and its table, which shows 13.02 instead of 13.82 for 14:00, to 254.32).
    Asserts the sum of the formula terms."""
    return trp_pct(parse_inputs(inp)["rating_pct"])


@fixture("F25")
def f25(inp):
    """Average Daily Reach %, MDT p14."""
    i = parse_inputs(inp)
    return reach_pct(average_daily_reach(i["reach"]), i["universe"])


@fixture("F26")
def f26(inp):
    """Unduplicated Reach, spots single day, MDT p83 (respondent level)."""
    i = parse_inputs(inp)
    return unduplicated_reach(i["weights"], exposures_from_items([_persons(x) for x in i["items"]]))


@fixture("F27")
def f27(inp):
    """Unduplicated Reach multi day with common (average) weights, MDT p18:
    (900+950)/2 + (1,100+1,200)/2 + (1,000+1,300)/2 + (800+900)/2 + 700 = 4,775."""
    i = parse_inputs(inp)
    return unduplicated_reach(_cw(i), exposures_from_items([_persons(x) for x in i["items"]]))


@fixture("F28")
def f28(inp):
    """ERRATUM - Unduplicated Reach spots multi day, MDT p84. The PDF prints 4,825 because it
    averages (1,100 + 1,300)/2 for the red person, whose weights are 1,000 (20 May) and
    1,300 (21 May) per the legend; the identical data on p18 gives 4,775. Asserts 4,775."""
    return f27(inp)


def _cume(inp):
    i = parse_inputs(inp)
    lines = [_persons(x) for x in i["lines"]]
    if "daily_weights" in i:
        daily = _daily(i["daily_weights"])
        assert i.get("scope") == "to_date"
        per_line = []
        for n in range(len(lines)):  # line n covers days 0..n
            upto = {p: {d: w for d, w in ds.items() if d <= n} for p, ds in daily.items()}
            per_line.append(common_weights({p: d for p, d in upto.items() if d}, rule=i["rule"]))
        return cume_reach_rf(lines, per_line)
    return cume_reach_rf(lines, i["weights"])


@fixture("F29")
def f29(inp):
    """Cume Reach (RF) multi day (one line per day), MDT p21: 3,800 then
    (900+950)/2 + (1,100+1,200)/2 + 1,000 + 800 + 1,300 + 900 = 6,075. Line 1 uses only the
    first day's weights (the 'to date' convention; contrast F34/p87)."""
    return _cume(inp)


@fixture("F30")
def f30(inp):
    """Cume Reach % (RF), MDT p22."""
    i = parse_inputs(inp)
    return reach_pct(i["reach"], i["universe"])


@fixture("F31")
def f31(inp):
    """Cume Reach (RF) program single day, MDT p57: 3,100 / 4,200 / 4,200 / 5,600."""
    return _cume(inp)


@fixture("F32")
def f32(inp):
    """Incremental Reach (GL p24) from the MDT p57 cume lines."""
    return incremental_reach(parse_inputs(inp)["cume"])


@fixture("F33")
def f33(inp):
    """Cume Reach (RF) program multi day, MDT p58: day-1 reach 5,600 (daily weights), grand
    summary 6,775 = 925 + 1,050 + 1,000 + 1,100 + 1,400 + 1,300. The PDF formula text omits
    the 1,400 term but prints the correct total."""
    return _cume(inp)


@fixture("F34")
def f34(inp):
    """Cume Reach (RF) spots multi day, MDT p87: two-day AVERAGE weights applied from line 1."""
    return _cume(inp)


@fixture("F35")
def f35(inp):
    """Incremental Reach from the MDT p87 cume lines; round-trips through
    cume_reach_rf_from_increments."""
    inc = incremental_reach(parse_inputs(inp)["cume"])
    assert cume_reach_rf_from_increments(inc) == pytest.approx(parse_inputs(inp)["cume"])
    return inc


@fixture("F36")
def f36(inp):
    """ERRATUM - Reach N+ multi day, MDT p24 (and p62). Average weights A=725, B=1,050,
    C=825, D=925; exposures from the drawn viewing table: A=3, B=5, C=6, D=3. The PDF prints
    4+ = 725 + 1,050 + 825 = 2,600, but A has only 3 exposures, so 4+ = B + C = 1,875.
    1+/2+/3+ = 3,525 as printed."""
    i = parse_inputs(inp)
    f = exposures_from_items([_persons(x) for x in i["items"]])
    assert f == {"A": 3, "B": 5, "C": 6, "D": 3}
    return _reach_levels(_cw(i), f, ["1+", "2+", "3+", "4+"], reach_n_plus)


@fixture("F37")
def f37(inp):
    """ERRATUM - Reach N (exact frequency), MDT p60. Exposures A=1, B=3, C=3, D=2 give
    Reach 1 = 700, Reach 2 = 900, Reach 3 = 1,800. The PDF instead prints "Total 9,900" as
    the sum of per-program reach 2,500 + 1,800 + 1,700 + 1,900 - which is (a) not Reach N
    (it is gross contacts) and (b) mis-added (the terms sum to 7,900)."""
    i = parse_inputs(inp)
    f = exposures_from_items([_persons(x) for x in i["items"]])
    out = {k: reach_n(i["weights"], f, int(k)) for k in ("1", "2", "3")}
    assert sum(out.values()) == unduplicated_reach(i["weights"], f)
    return out


@fixture("F38")
def f38(inp):
    """ERRATUM - Reach N+ spots single day, MDT p89. 2+ = 1,100 + 1,000 + 800 = 2,900; the PDF
    prints 2,800 (arithmetic slip, propagated to 2+% = 0.014% on p91). Exposure 4's weight
    list also shows a spurious '1,800' next to the two drawn viewers."""
    i = parse_inputs(inp)
    f = exposures_from_items([_persons(x) for x in i["items"]])
    return _reach_levels(i["weights"], f, ["1+", "2+", "3+"], reach_n_plus)


@fixture("F39")
def f39(inp):
    """Reach N+ spots multi day, MDT p90 (average weights; printed values correct)."""
    i = parse_inputs(inp)
    f = exposures_from_items([_persons(x) for x in i["items"]])
    return _reach_levels(i["weights"], f, ["1+", "2+", "3+"], reach_n_plus)


@fixture("F40")
def f40(inp):
    """Reach N+ %, MDT p25."""
    i = parse_inputs(inp)
    return {f"{n}+": reach_pct(r, i["universe"]) for n, r in zip((1, 2, 3), i["reach"])}


@fixture("F41")
def f41(inp):
    """Frequency multi day rebuilt at respondent level with average weights, MDT p27:
    (950x4 + 1,150x4 + 850 + 700 + 1,400 + 1,100) / 6,150 = 12,450 / 6,150."""
    i = parse_inputs(inp)
    return frequency(_cw(i), i["exposures"])


@fixture("F42")
def f42(inp):
    """ERRATUM - Frequency program multi day, MDT p64. Average weights 925, 1,050, 900, 1,050
    (sum 3,925); sum(w*f) = 925x3 + 1,050x2 + 900 + 1,050 = 6,825, but the PDF table prints
    9,825. 6,825 / 3,925 = 1.7389, printed as 1.73 (TRUNCATION; rounding gives 1.74)."""
    i = parse_inputs(inp)
    return frequency(_cw(i), i["exposures"])


@fixture("F43")
def f43(inp):
    """Average Daily Reach from per-day respondent weights, MDT p50: (4,200 + 4,550)/2."""
    i = parse_inputs(inp)
    return average_daily_reach([i["day1"], i["day2"]])


@fixture("F44")
def f44(inp):
    """Average Weekly Reach with weekly weights = mean of each person's daily weights in the
    week (MDT p15 "AVERAGE WEIGHTS WEEK n"): (4,900 + 5,650)/2."""
    i = parse_inputs(inp)
    weeks = [{p: _persons(v) for p, v in i[k].items()} for k in ("week1", "week2")]
    return average_weekly_reach(weeks)


@fixture("F45")
def f45(inp):
    """Average Weekly Reach %, MDT p16 (printed 0.02% = truncated)."""
    i = parse_inputs(inp)
    return reach_pct(average_weekly_reach(i["reach"]), i["universe"])


@fixture("F46")
def f46(inp):
    """TSV Universe (Daily) multi day, MDT p29: mean daily Rating Abs x 60 / Universe."""
    i = parse_inputs(inp)
    r = np.mean([rating_abs([1.0], [s], i["duration"]) for s in i["day_sum_wt"]])
    return tsv_universe_daily(r, i["duration"], i["universe"])


@fixture("F47")
def f47(inp):
    """TSV Viewers (Daily) multi day, MDT p31: 988.9 x 60 / 2,950. PDF prints 20.12 because
    it uses the rounded rating 989."""
    i = parse_inputs(inp)
    r = np.mean([rating_abs([1.0], [s], i["duration"]) for s in i["day_sum_wt"]])
    return tsv_viewers_daily(r, i["duration"], average_daily_reach(i["daily_reach"]))


@fixture("F48")
def f48(inp):
    """TSV Universe (Weekly), MDT p32 / GL p34: sum(Rating Abs_w x 420) / Universe. PDF prints
    1.07 (truncated). For 3 weeks this is the total over the period, not per week."""
    i = parse_inputs(inp)
    return tsv_universe_weekly(i["rating_abs"], i["duration"], i["universe"])


@fixture("F49")
def f49(inp):
    """TSV Viewers (Weekly), MDT p33 / GL p33: sum(Rating Abs_w x 420) / Average Weekly Reach."""
    i = parse_inputs(inp)
    return tsv_viewers_weekly(i["rating_abs"], i["duration"], average_weekly_reach(i["weekly_reach"]))


@fixture("F50")
def f50(inp):
    """TSV Viewers (Daily) program multi day, MDT p65: [(1,000 + 1,120)/2] / 3,350 x 120.
    (p65 writes '700 * 12 = 70', omitting the '/120', but the value 70 is right.)"""
    r, d, adr = _two_day_program(inp)
    return tsv_viewers_daily(r, d, adr)


@fixture("F51")
def f51(inp):
    """Completion Rate program multi day, MDT p66: 1,060 / 3,350 x 100."""
    r, _d, adr = _two_day_program(inp)
    return completion_rate(r, adr)


@fixture("F52")
def f52(inp):
    """ERRATUM - Total Duration, MDT p35. Program 3 runs 16:30:00-17:59:59 = 90 min, but the
    PDF lists 30 min and totals 36 + 5 + 30 + 16 = 77. Correct: 36 + 5 + 90 + 16 = 147."""
    return total_duration(parse_inputs(inp)["durations_min"])


@fixture("F53")
def f53(inp):
    """Average Daily TRP Absolute, MDT p36 (printed ...782 = rounded from ...781.71)."""
    return average_daily_trp_abs(parse_inputs(inp)["daily_trp_abs"])


@fixture("F54")
def f54(inp):
    """TRP % programs, MDT p46."""
    return trp_pct(parse_inputs(inp)["rating_pct"])


@fixture("F55")
def f55(inp):
    """Program Rating %, MDT p45: (64,000/30) / 20,000,000 x 100."""
    i = parse_inputs(inp)
    return rating_pct(rating_abs([1.0], [i["sum_wt"]], i["duration"]), i["universe"])


@fixture("F56")
def f56(inp):
    """ERRATUM - Program Share to Selected %, MDT p48. The table lists Programs 1-14 as 1,300,
    1,100, 900, 850, 2,133, 1,083, 900, 1,300, 1,650, 800, 2,400, 2,000, 850, 1,150
    (sum 18,416) -> 11.58%. The printed formula uses 1,000 / 2,100 / 1,300 in place of
    1,100 / 2,133 / 1,083 (-> 11.66%). Printed result 11.5% is the table value truncated."""
    i = parse_inputs(inp)
    return share_to_selected(i["channel"], i["selected"])


@fixture("F57")
def f57(inp):
    """Duplication Cume Reach, MDT p74 (and Program 1 reach). The p74 formula image shows
    V1 \\ V2 (the exclusive formula); the correct set is V1 intersect V2 (GL p76)."""
    i = parse_inputs(inp)
    return {"dup": duplication_cume_reach(i["weights"], i["p1"], i["p2"]),
            "reach_p1": unduplicated_reach(i["weights"], i["p1"])}


@fixture("F58")
def f58(inp):
    """Exclusive Cume Reach, 3 programs, MDT p76 (cell row r / column c = viewers of r not c).
    Viewers as drawn: P1 = {900, 1,100, 1,000}, P2 = {1,000, 800}, P3 = {1,100, 1,000, 1,300}.
    Note: the weights column of the 10:30 row lists '1,100, 1,000, 800' while only two
    viewers are drawn; the printed matrix is consistent with the drawing only."""
    i = parse_inputs(inp)
    out = {}
    for a in ("p1", "p2", "p3"):
        for b in ("p1", "p2", "p3"):
            if a != b:
                out[f"{a}-{b}"] = exclusive_cume_reach(i["weights"], i[a], i[b])
    return out


@fixture("F59")
def f59(inp):
    """GRP Absolute from per-person exposure counts, MDT p92: 900x3 + 1,100x2 + 1,000x8 + 800x3."""
    i = parse_inputs(inp)
    return grp_abs_from_exposures(i["weights"], i["exposures"])


@fixture("F60")
def f60(inp):
    """Rating Absolute combined from daily ratings (GL p8 duration-weighted average) - the
    aggregated path reproduces F02 (MDT p6)."""
    i = parse_inputs(inp)
    return rating_abs_combine(i["ratings"], i["durations"])


EXPECTED_OVERRIDE = {
    # F23's expected cell is a sentence; the asserted number is its final value.
    "F23": 53675.0,
}


# ------------------------------------------------------------------ the parametrised test
def _tol(row) -> float:
    m = re.search(r"tol=([\d.]+)", row["note"] or "")
    return float(m.group(1)) if m else DEFAULT_TOL


def _assert_close(got, exp, tol, fid):
    if isinstance(exp, dict):
        assert set(got) == set(exp), f"{fid}: keys {sorted(got)} != {sorted(exp)}"
        for k in exp:
            assert got[k] == pytest.approx(exp[k], rel=tol, abs=1e-9), f"{fid} [{k}]"
    elif isinstance(exp, list):
        assert len(got) == len(exp), f"{fid}: length"
        assert list(got) == pytest.approx(exp, rel=tol, abs=1e-9), fid
    else:
        assert got == pytest.approx(exp, rel=tol, abs=1e-9), fid


def test_every_fixture_row_has_a_test():
    assert {r["fixture_id"] for r in ROWS} == set(COMPUTE), "fixture rows and tests out of sync"


def test_fixture_csv_format():
    header = ["fixture_id", "data_type", "source_pdf", "inputs", "expected", "status", "note"]
    with open(FIXTURES, newline="", encoding="utf-8") as fh:
        assert next(csv.reader(fh)) == header
    assert all(len(r) == 7 for r in ROWS)
    assert {r["status"] for r in ROWS} <= {"ok", "erratum"}


@pytest.mark.parametrize("row", ROWS, ids=[r["fixture_id"] for r in ROWS])
def test_fixture(row):
    fid = row["fixture_id"]
    got = COMPUTE[fid](row["inputs"])
    exp = EXPECTED_OVERRIDE.get(fid, None)
    if exp is None:
        exp = parse_expected(row["expected"])
    if row["status"] == "erratum":
        assert COMPUTE[fid].__doc__ and "ERRATUM" in COMPUTE[fid].__doc__, \
            f"{fid}: erratum test must document the PDF discrepancy in its docstring"
    _assert_close(got, exp, _tol(row), fid)


# ------------------------------------------------------------------ non-fixture unit tests
# (no worked example exists in the PDFs for these; synthetic values, formula from GL)
@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_reach_from_aggregates_raises():
    with pytest.raises(ReachFromAggregatesError, match="duplication"):
        unduplicated_reach(53675.0)
    with pytest.raises(NotImplementedError):
        reach_n_plus([1000.0, 2000.0], None, 3)
    with pytest.raises(ValueError):
        frequency(15300.0, None)
    with pytest.raises(ReachFromAggregatesError):
        cume_reach_rf([3000.0, 2900.0], None)
    with pytest.raises(ReachFromAggregatesError):
        reach_from_aggregates(rating_abs=1625)


def test_common_weight_rules(cfg):
    dw = {"A": {1: 900, 2: 1000, 3: 1100}, "B": {1: 800, 3: 1000}}
    assert common_weights(dw, rule="average") == {"A": 1000.0, "B": 900.0}
    # middle day of {1,2,3} is 2; B has no weight that day -> excluded (0)
    assert common_weights(dw, rule="middle_day") == {"A": 1000.0, "B": 0.0}
    df = pd.DataFrame([(p, d, w) for p, ds in dw.items() for d, w in ds.items()],
                      columns=["person", "day", "weight"])
    rule = cfg["reach"]["common_weight_rule"]
    assert common_weights(df, rule=rule) == common_weights(dw, rule=rule)
    with pytest.raises(ValueError):
        common_weights(dw, rule="first_day")


def test_exposure_threshold():
    viewing = [("A", 1, 5), ("A", 1, 4), ("A", 2, 3), ("B", 1, 2)]
    assert exposure_counts(viewing, min_seconds=5) == {"A": 1}   # A item1 = 9 s, item2 = 3 s
    assert exposure_counts(viewing, min_seconds=1) == {"A": 2, "B": 1}


def test_cost_metrics():
    assert cpm([1000, 3000], [100_000, 300_000]) == pytest.approx(10.0)          # GL p44
    assert cost_per_rating_pct([1000, 3000], [0.5, 1.5]) == pytest.approx(2000)  # GL p45
    f = {30: 1.0, 15: 0.6}  # illustrative only - real table pending (DATA_SPEC C.4)
    assert weighted_grp_abs([1000, 1000], [30, 15], f) == pytest.approx(1600)    # GL p47
    assert weighted_grp_pct([1.0, 1.0], [30, 15], f) == pytest.approx(1.6)       # GL p48
    assert weighted_spot_cost([600, 600], [30, 15], f) == pytest.approx(1600)    # GL p49 (divides)
    assert weighted_cpm([600, 600], [1000, 1000], [30, 15], f) == pytest.approx(800)  # GL p50
    assert weighted_cost_per_rating_pct([600, 600], [1.0, 1.0], [30, 15], f) == pytest.approx(800)
    assert weighted_grp_abs([500], [30], DEFAULT_RATE_DURATION_FACTORS) == 500
    with pytest.raises(KeyError, match="Rate Duration"):
        weighted_grp_abs([500], [20], DEFAULT_RATE_DURATION_FACTORS)


def test_universe_and_loyalty():
    assert universe([900, 1100, 1000]) == 3000                       # GL p86
    assert universe({"a": 900, "b": 1100}) == 2000
    assert loyalty_pct(900, 1200) == pytest.approx(75.0)             # GL p64


def test_break_rating_vectorised():
    s = pd.Series([125241.0, 60000.0], index=["x", "y"])
    out = rating_abs_from_break_trp(s, pd.Series([140, 60], index=["x", "y"]))
    assert list(out.index) == ["x", "y"]
    assert out["y"] == pytest.approx(60000)


def test_display_rounding(cfg):
    """OTS 4.0263 -> 4.03 under round_half_up, 4.02 under truncate (the PDF's MDT p92 value)."""
    v = ots(15300, 3800)
    assert display_round(v, "ots", cfg) == (4.03 if cfg["rounding"]["display"] == "round_half_up" else 4.02)
    c2 = copy.deepcopy(cfg)
    c2["rounding"]["display"] = "truncate"
    assert display_round(v, "ots", c2) == 4.02
    c2["rounding"]["display"] = "round_half_up"
    assert display_round(2.675, "ots", c2) == 2.68                   # decimal, not binary, half-up
    assert display_round([2.138, 4.9175], "rating_pct", c2) == [2.14, 4.92]
    with pytest.raises(KeyError):
        display_round(1.0, "no_such_kind", c2)
