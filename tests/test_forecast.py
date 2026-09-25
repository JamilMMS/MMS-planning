"""Tests for optimizer.forecast (Phase 4): pure-function tests on tiny synthetic frames, plus
regression tests on the real processed files when they exist."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import trim_mean

from optimizer.config import load_config, project_root
from optimizer.forecast.core import (ForecastParams, forecast_one, forecast_targets, robust_estimate, shrink,
                                     weighted_estimate, BreakIndex, is_live_special, widen)
from optimizer.forecast.run import assemble, build_october_targets, load_history, low_sample_channels

ROOT = project_root()


# --------------------------------------------------------------------------- helpers
def mk_breaks(rows):
    """rows: (channel, program, date 'YYYY-MM-DD', minute, first_run, rating_abs[, is_event])"""
    recs = []
    for i, r in enumerate(rows):
        ch, prog, d, minute, fr, y = r[:6]
        ev = r[6] if len(r) > 6 else False
        ts = pd.Timestamp(d)
        recs.append({"channel": ch, "program": prog, "broadcast_date": ts, "weekday": ts.strftime("%a"),
                     "start_sec": minute * 60 + 5, "first_run": fr, "rating_abs": float(y), "is_event": ev,
                     "break_id": f"{ch}|{d}|{minute * 60 + 5}|{i}", "target": "TP Arabs 15+",
                     "low_sample_channel": False})
    return pd.DataFrame(recs)


def target(**kw):
    t = dict(target_id="T", channel="C1", weekday="Sun", start_min=1200, end_min=1260, is_rerun=False,
             etam_title="", use_program=False, prefer_rerun_pool=False, low_sample=False, is_new=False,
             live_special=False)
    t.update(kw)
    return t


SUNDAYS = ["2026-09-06", "2026-09-13", "2026-09-20"]
MONTUE = ["2026-09-07", "2026-09-08", "2026-09-14", "2026-09-15"]


def base_history():
    rows = []
    # programme SHOW first run at 20:10 on Sundays (3 days x 3 breaks = 9 breaks, audience ~100)
    for d in SUNDAYS:
        for k, y in enumerate([90, 100, 110]):
            rows.append(("C1", "SHOW", d, 1210 + 10 * k, True, y))
    # SHOW reruns at 20:15 on Mon/Tue (audience ~40), 8 breaks
    for d in MONTUE:
        for k, y in enumerate([38, 42]):
            rows.append(("C1", "SHOW", d, 1215 + 10 * k, False, y))
    # other programmes in prime on the channel (audience ~60), all weekdays
    for d in SUNDAYS + MONTUE:
        for k in range(4):
            rows.append(("C1", "OTHER", d, 1080 + 30 * k, True, 60))
    return mk_breaks(rows)


P = ForecastParams(min_breaks=6, shrinkage_k=6, estimator="trimmed_mean_10", bootstrap_n=300,
                   weekday_groups=[["Sun", "Mon", "Tue", "Wed"], ["Thu"], ["Fri"], ["Sat"]])


# --------------------------------------------------------------------------- estimators
def test_trimmed_mean_matches_scipy_when_integer_trim():
    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, 1000.0, size=50)
    assert robust_estimate(x, "trimmed_mean_10") == pytest.approx(trim_mean(x, 0.10), rel=1e-12)
    assert robust_estimate(x, "median") == pytest.approx(np.median(x))
    assert robust_estimate(x[:7], "median") == pytest.approx(np.median(x[:7]))
    assert robust_estimate(x, "mean") == pytest.approx(x.mean())


def test_weighted_estimate_equals_repeated_values():
    x = np.sort(np.array([0.0, 0.0, 1.0, 5.0, 9.0, 30.0]))
    w = np.array([2, 0, 1, 3, 1, 1])
    rep = np.repeat(x, w)
    for est in ("median", "mean", "trimmed_mean_10", "trimmed_mean_25"):
        got = float(weighted_estimate(x, w, est))
        assert got == pytest.approx(robust_estimate(rep, est)), est


def test_robust_to_outlier():
    x = [100.0] * 19 + [100000.0]
    assert robust_estimate(x, "trimmed_mean_10") == pytest.approx(100.0)
    assert robust_estimate(x, "median") == pytest.approx(100.0)


# --------------------------------------------------------------------------- shrinkage
def test_shrinkage_direction_and_weight():
    assert shrink(100, 6, 50, 6) == pytest.approx(75)
    assert 50 < shrink(100, 60, 50, 6) < 100
    assert shrink(100, 60, 50, 6) > shrink(100, 6, 50, 6)      # more evidence -> closer to own level
    assert shrink(100, 6, None, 6) == 100
    assert shrink(100, 6, 50, 0) == 100


def test_program_level_shrinks_toward_slot():
    bi = BreakIndex(base_history(), P)
    r = forecast_one(target(etam_title="SHOW", use_program=True), bi, with_intervals=False)
    assert r["forecast_level"] == "program" and r["evidence_n"] == 9
    raw, parent = r["est_program"], r["parent_est"]
    assert raw == pytest.approx(100.0)
    assert min(raw, parent) < r["p50"] < max(raw, parent)
    assert r["p50"] == pytest.approx((9 * raw + 6 * parent) / 15)


# --------------------------------------------------------------------------- hierarchy fallback
def test_fallback_program_to_slot_when_thin():
    b = base_history()
    b = b[~((b["program"] == "SHOW") & b["first_run"] & (b["broadcast_date"] > "2026-09-06"))]  # 3 first-run breaks left
    bi = BreakIndex(b, P)
    r = forecast_one(target(etam_title="SHOW", use_program=True), bi, with_intervals=False)
    assert r["n_program"] == 3
    assert r["forecast_level"] == "slot"


def test_fallback_slot_to_channel_daypart_when_thin():
    bi = BreakIndex(base_history(), P)
    # Thursday: nothing aired on a Thursday -> slot level empty -> channel x daypart x day-type
    r = forecast_one(target(weekday="Thu"), bi, with_intervals=False)
    assert r["n_slot"] == 0 and r["forecast_level"] == "channel_daypart"
    assert r["evidence_n"] >= P.min_breaks


def test_low_sample_uses_pooled_level_only():
    bi = BreakIndex(base_history(), P)
    r = forecast_one(target(etam_title="SHOW", use_program=True, low_sample=True), bi, with_intervals=False)
    assert r["forecast_level"] == "channel_daypart" and r["n_program"] == 0 and r["n_slot"] == 0
    assert "all days" in r["evidence_desc"]


def test_low_sample_estimator_override():
    p = ForecastParams(**{**P.as_dict(), "low_sample_estimator": "mean"})
    assert p.estimator_for(True) == "mean" and p.estimator_for(False) == P.estimator


# --------------------------------------------------------------------------- rerun vs first run
def test_rerun_uses_rerun_history_only():
    bi = BreakIndex(base_history(), P)
    fr = forecast_one(target(etam_title="SHOW", use_program=True, is_rerun=False, weekday="Mon"), bi, False)
    rr = forecast_one(target(etam_title="SHOW", use_program=True, is_rerun=True, weekday="Mon"), bi, False)
    assert fr["est_program"] == pytest.approx(100.0)
    assert rr["est_program"] == pytest.approx(40.0)
    assert rr["n_program"] == 8 and fr["n_program"] == 9


# --------------------------------------------------------------------------- event exclusion
def test_event_rows_excluded_from_history(tmp_path):
    rows = [("C1", "SHOW", d, 1210, True, 100) for d in SUNDAYS] + [("C1", "MATCH", "2026-09-19", 1100, True, 999999, True)]
    b = mk_breaks(rows)
    (tmp_path / "b.parquet").write_bytes(b"")
    b.to_parquet(tmp_path / "b.parquet", index=False)
    cfg = {"forecast": {"paths": {"breaks": "b.parquet"}}, "etam": {"exclude_event_days": True},
           "target": {"buying_target": "TP Arabs 15+"}}
    h = load_history(cfg, tmp_path)
    assert len(h) == 3 and not h["is_event"].any()
    cfg["etam"]["exclude_event_days"] = False
    assert len(load_history(cfg, tmp_path)) == 4


def test_event_spike_would_contaminate_but_excluded():
    rows = [("C1", "X", d, 1090 + 5 * k, True, 50) for d in ["2026-09-05", "2026-09-12", "2026-09-19"] for k in range(4)]
    rows += [("C1", "MATCH", "2026-09-19", 1100 + k, True, 300000, True) for k in range(3)]
    b = mk_breaks(rows)
    t = target(weekday="Sat", start_min=1080, end_min=1140)
    fc_clean, _ = forecast_targets(pd.DataFrame([t]), b[~b["is_event"]], P, with_intervals=False)
    assert fc_clean["p50"].iloc[0] == pytest.approx(50.0)


# --------------------------------------------------------------------------- intervals
def test_quantiles_ordered_and_widening():
    b = base_history()
    ts = pd.DataFrame([target(target_id="a", etam_title="SHOW", use_program=True),
                       target(target_id="b", weekday="Mon", is_new=True),
                       target(target_id="c", weekday="Thu", live_special=True),
                       target(target_id="d", low_sample=True)])
    fc, _ = forecast_targets(ts, b, P, with_intervals=True)
    assert (fc["p10"] <= fc["p50"] + 1e-9).all() and (fc["p50"] <= fc["p90"] + 1e-9).all()
    assert (fc["p10"] >= 0).all()
    w = dict(zip(fc["target_id"], fc["widening"]))
    assert w["a"] == 1.0 and w["b"] == P.interval_widening["new_program"] and w["c"] == P.interval_widening["live_special"]
    lo, hi = widen(100, 80, 130, 2.0)
    assert (lo, hi) == (60, 160)
    assert widen(100, 10, 130, 2.0)[0] == 0.0


def test_deterministic():
    b = base_history()
    ts = pd.DataFrame([target(target_id="a", etam_title="SHOW", use_program=True)])
    f1, _ = forecast_targets(ts, b, P)
    f2, _ = forecast_targets(ts, b, P)
    assert f1[["p10", "p50", "p90"]].equals(f2[["p10", "p50", "p90"]])


def test_live_special_patterns():
    pats = ForecastParams().live_special_patterns
    assert is_live_special("NADEENA (KHALEEJI 27)", True, pats)
    assert is_live_special("EXTREME H WORLD CUP (2026)DAY 1LIVE", True, pats)
    assert not is_live_special("MBC NEWS LIVE", True, pats)
    assert not is_live_special("NADEENA (KHALEEJI 27)", False, pats)


# --------------------------------------------------------------------------- October assembly (tiny)
def tiny_grid():
    def row(sid, date, syn, title="SHOW"):
        d = pd.Timestamp(date)
        return {"slot_id": sid, "channel": "C1", "air_date": d, "weekday": d.strftime("%a"), "start_min": 1200,
                "end_min": 1260, "start_hhmm": 2000, "end_hhmm": 2059, "tier": "Prime", "rate_usd": 1000.0,
                "rate_aed": 3672.5, "title_en": title, "program_name": title, "is_rerun": False, "is_live": False,
                "is_synthetic": syn, "has_conflict": False, "assumption": "ASSUMPTION_MBC1_WK4" if syn else "",
                "in_flight": True}
    return pd.DataFrame([row("C1|2026-10-18|2000", "2026-10-18", False), row("C1|2026-10-25|2000", "2026-10-25", True),
                         row("C1|2026-10-19|2000", "2026-10-19", False, "NEWBIE")])


def tiny_map():
    return pd.DataFrame([
        {"channel": "C1", "grid_title": "SHOW", "etam_title": "SHOW", "etam_channel": "C1", "match_type": "exact",
         "confidence": 1.0, "slot_pool_rule": np.nan, "flags": np.nan},
        {"channel": "C1", "grid_title": "NEWBIE", "etam_title": "NONE", "etam_channel": np.nan, "match_type": "new program",
         "confidence": 0.95, "slot_pool_rule": "per October slot: ...", "flags": np.nan}])


def test_carried_forward_rows_equal_source_and_one_row_per_slot():
    g, pm = tiny_grid(), tiny_map()
    t = build_october_targets(g, pm, P, set())
    assert len(t) == 2  # synthetic row not forecast separately
    fc, _ = forecast_targets(t, base_history(), P)
    cfg = {"target": {"universe": 1_000_000}}
    out = assemble(g, t, fc, cfg, P)
    assert len(out) == 3 and out["slot_id"].is_unique
    src = out.set_index("slot_id").loc["C1|2026-10-18|2000"]
    syn = out.set_index("slot_id").loc["C1|2026-10-25|2000"]
    for c in ["aud_abs_p10", "aud_abs_p50", "aud_abs_p90", "forecast_level", "evidence_n", "cpm_p50"]:
        assert syn[c] == src[c], c
    assert syn["source_slot_id"] == "C1|2026-10-18|2000"
    assert "ASSUMPTION_MBC1_WK4" in syn["assumption"] and "CARRIED_FORWARD" in syn["assumption"]
    new = out.set_index("slot_id").loc["C1|2026-10-19|2000"]
    assert bool(new["is_new_program"]) and new["forecast_level"] in ("slot", "channel_daypart")
    assert src["cpm_p50"] == pytest.approx(1000.0 / (src["aud_abs_p50"] / 1000.0))
    assert src["rating_pct_p50"] == pytest.approx(src["aud_abs_p50"] / 1_000_000 * 100)


def test_cross_channel_match_not_used_at_program_level():
    g = tiny_grid().iloc[:1]
    pm = tiny_map().iloc[:1].assign(etam_channel="C2")
    t = build_october_targets(g, pm, P, set())
    assert not t["use_program"].iloc[0] and "CROSS_CHANNEL" in ";".join(t["assumption_codes"].iloc[0])


# --------------------------------------------------------------------------- real-data regression
REAL = all((ROOT / p).exists() for p in ["data/processed/grid.parquet", "data/processed/breaks.parquet",
                                          "data/processed/program_map.csv"])


@pytest.fixture(scope="module")
def october():
    if not REAL:
        pytest.skip("processed parquet files not present")
    cfg = load_config()
    p = ForecastParams.from_cfg(cfg)
    grid = pd.read_parquet(ROOT / cfg["forecast"]["paths"]["grid"])
    pm = pd.read_csv(ROOT / cfg["forecast"]["paths"]["program_map"])
    hist = load_history(cfg, ROOT)
    t = build_october_targets(grid, pm, p, low_sample_channels(ROOT, cfg))
    fc, _ = forecast_targets(t, hist, p)
    return grid, assemble(grid, t, fc, cfg, p), hist


def test_real_one_forecast_row_per_in_flight_grid_row(october):
    grid, out, _ = october
    gi = grid[grid["in_flight"]]
    assert len(out) == len(gi)
    assert set(out["slot_id"]) == set(gi["slot_id"]) and out["slot_id"].is_unique
    assert (out["aud_abs_p10"] <= out["aud_abs_p50"] + 1e-9).all() and (out["aud_abs_p50"] <= out["aud_abs_p90"] + 1e-9).all()
    assert (out["aud_abs_p50"] > 0).all()


def test_real_history_has_no_event_rows(october):
    _, _, hist = october
    assert not hist["is_event"].any()


def test_real_mr_bean_action_saturday_1800_cpm(october):
    """Naive event-contaminated value was $1.91 CPM (PRELIMINARY_FINDINGS 2); must be >= $30."""
    _, out, _ = october
    r = out[(out["channel"] == "MBC ACTION") & (out["title_en"] == "MR. BEAN") & (out["weekday"] == "Sat")
            & (out["start_hhmm"] == 1800)]
    assert len(r) >= 1
    assert (r["cpm_p50"] >= 30).all(), r[["slot_id", "aud_abs_p50", "cpm_p50"]]


def test_real_carried_forward_equal_source(october):
    _, out, _ = october
    syn = out[out["is_synthetic"]]
    if syn.empty:
        pytest.skip("no synthetic rows")
    src = out.set_index("slot_id").loc[syn["source_slot_id"]]
    for c in ["aud_abs_p10", "aud_abs_p50", "aud_abs_p90", "forecast_level"]:
        assert (syn[c].to_numpy() == src[c].to_numpy()).all(), c
