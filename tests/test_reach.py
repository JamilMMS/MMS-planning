"""Reach engine tests: interface contract (EXACT / CALIBRATED / ESTIMATE), ESTIMATE model
properties, Nielsen fixture reproduction through the EXACT engine, CALIBRATED recovery."""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest

from optimizer.config import load_config
from optimizer.reach import curves
from optimizer.reach.calibrate import (CalibrationFileError, fit_calibrated, load_rf_schedules,
                                       synthetic_rf)
from optimizer.reach.common import abs_path
from optimizer.reach.engine import RESULT_KEYS, ReachEngine, check_sanity
from optimizer.reach.exact import RespondentFileError, load_respondent_file
from optimizer.reach.fit import fit_estimate_params, fit_turnover

CHANNELS = ["MBC 1", "MBC 2", "MBC 4", "MBC ACTION", "MBC BOLLYWOOD", "MBC DRAMA", "MBC MAX"]


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _params(method="ESTIMATE", form="hyperbolic"):
    rmax = [50.0, 16.0, 22.0, 2.0, 12.0, 26.0, 3.0]
    return {"method": method, "form": form, "cell_granularity": "channel",
            "generated_at": "test", "assumptions": {"synthetic": True},
            "cells": {c: {"rmax_pct": r, "k": r} for c, r in zip(CHANNELS, rmax)}}


def _cfg_mode(cfg, mode, **reach):
    c = copy.deepcopy(cfg)
    c["reach"]["mode"] = mode
    for k, v in reach.items():
        c["reach"][k] = v
    return c


@pytest.fixture(scope="module")
def est(cfg):
    return ReachEngine.from_config(_cfg_mode(cfg, "ESTIMATE"), params=_params())


def _schedule(n=300, seed=0):
    rng = np.random.default_rng(seed)
    med = {"MBC 1": 80e3, "MBC DRAMA": 37e3, "MBC 4": 22e3, "MBC BOLLYWOOD": 19e3,
           "MBC 2": 14e3, "MBC ACTION": 2.8e3, "MBC MAX": 2e3}
    ch = rng.choice(CHANNELS, n)
    return pd.DataFrame({
        "slot_id": [f"{c}|{i}" for i, c in enumerate(ch)], "channel": ch,
        "air_date": pd.Timestamp("2026-10-04") + pd.to_timedelta(rng.integers(0, 28, n), "D"),
        "start_min": rng.integers(180, 1620, n),
        "aud_abs": [med[c] * rng.uniform(0.3, 1.8) for c in ch]})


# ----------------------------------------------------------------------------- contract
def _contract(out, method):
    for k in ("reach_1plus_abs", "reach_1plus_pct", "reach_3plus_abs", "reach_3plus_pct",
              "reach_n_dist", "grp_abs", "grp_pct", "ots", "method", "params_summary"):
        assert k in out, k
    assert set(RESULT_KEYS) <= set(out)
    assert out["method"] == method
    assert out["reach_1plus_pct"] == pytest.approx(out["reach_1plus_abs"] / out["universe"] * 100)
    assert out["reach_3plus_abs"] <= out["reach_1plus_abs"] + 1e-9
    assert sum(out["reach_n_dist"].values()) == pytest.approx(out["reach_1plus_abs"], rel=1e-9, abs=1e-6)
    check_sanity(out)


def test_contract_estimate(est):
    s = _schedule()
    out = est.reach(s)
    _contract(out, "ESTIMATE")
    assert out["grp_abs"] == pytest.approx(s["aud_abs"].sum())
    assert out["grp_pct"] == pytest.approx(s["aud_abs"].sum() / est.universe * 100)
    assert out["ots"] == pytest.approx(out["grp_abs"] / out["reach_1plus_abs"])
    st = est.new_state()
    for r in s.to_dict("records"):
        est.add(st, r)
    ev = est.evaluate(st)
    assert ev["reach_1plus_abs"] == pytest.approx(out["reach_1plus_abs"])
    assert ev["method"] == "ESTIMATE"


def test_contract_calibrated(cfg, tmp_path):
    p = _params("CALIBRATED")
    import json
    f = tmp_path / "cal.json"
    f.write_text(json.dumps(p))
    c = _cfg_mode(cfg, "CALIBRATED")
    c["reach"]["calibration"] = {**c["reach"]["calibration"], "params_path": str(f)}
    eng = ReachEngine.from_config(c)
    _contract(eng.reach(_schedule()), "CALIBRATED")
    # a parameter file of the wrong method is refused
    with pytest.raises(ValueError):
        ReachEngine.from_config(c, params=_params("ESTIMATE"))


def test_contract_calibrated_missing_file_is_clear(cfg, tmp_path):
    c = _cfg_mode(cfg, "CALIBRATED")
    c["reach"]["calibration"] = {**c["reach"]["calibration"], "params_path": str(tmp_path / "none.json")}
    with pytest.raises(FileNotFoundError, match="optimizer.reach.calibrate"):
        ReachEngine.from_config(c)


def test_contract_exact(cfg):
    eng, spots = _exact_engine(cfg, _F26_ITEMS, {0: _F26_W})
    out = eng.reach(spots)
    _contract(out, "EXACT")


def test_bad_mode(cfg):
    with pytest.raises(ValueError):
        ReachEngine.from_config(_cfg_mode(cfg, "GUESS"))


def test_empty_schedule(est):
    out = est.evaluate(est.new_state())
    assert out["reach_1plus_abs"] == 0 and out["ots"] is None and out["method"] == "ESTIMATE"


def test_forecast_fills_aud_abs(cfg):
    s = _schedule(20)
    fc = s[["slot_id", "aud_abs"]].rename(columns={"aud_abs": "aud_abs_p50"})
    eng = ReachEngine.from_config(_cfg_mode(cfg, "ESTIMATE"), forecast=fc, params=_params())
    a = eng.reach(s.drop(columns="aud_abs"))
    b = eng.reach(s)
    assert a["reach_1plus_abs"] == pytest.approx(b["reach_1plus_abs"])


# ----------------------------------------------------------------------------- ESTIMATE properties
@pytest.mark.parametrize("form", ["hyperbolic", "negexp"])
def test_monotone_and_marginal_equals_difference(cfg, form):
    eng = ReachEngine.from_config(_cfg_mode(cfg, "ESTIMATE"), params=_params(form=form))
    st = eng.new_state()
    prev = 0.0
    for sp in eng.prepare_many(_schedule(250, seed=3)):
        before = eng.evaluate(st)["reach_1plus_abs"]
        m = eng.marginal_reach(st, sp)
        m3 = eng.marginal_reach_n(st, sp, 3)
        b3 = eng.evaluate(st)["reach_3plus_abs"]
        eng.add(st, sp)
        after = eng.evaluate(st)
        assert m >= -1e-9
        assert after["reach_1plus_abs"] >= prev - 1e-9
        assert m == pytest.approx(after["reach_1plus_abs"] - before, abs=1e-6)
        assert m3 == pytest.approx(after["reach_3plus_abs"] - b3, abs=1e-6)
        prev = after["reach_1plus_abs"]


def test_remove_restores(est):
    sp = est.prepare_many(_schedule(60, seed=5))
    st = est.new_state()
    for p in sp[:40]:
        est.add(st, p)
    base = est.evaluate(st)
    for p in sp[40:]:
        est.add(st, p)
    for p in sp[40:]:
        est.remove(st, p)
    again = est.evaluate(st)
    assert again["reach_1plus_abs"] == pytest.approx(base["reach_1plus_abs"], rel=1e-9)
    assert again["reach_3plus_abs"] == pytest.approx(base["reach_3plus_abs"], rel=1e-9)
    assert again["grp_abs"] == pytest.approx(base["grp_abs"], rel=1e-9)


@pytest.mark.parametrize("form", ["hyperbolic", "negexp"])
def test_concave_in_grps(cfg, form):
    eng = ReachEngine.from_config(_cfg_mode(cfg, "ESTIMATE"), params=_params(form=form))
    st = eng.new_state()
    spot = {"channel": "MBC 1", "aud_abs": 100_000.0}
    inc = []
    for _ in range(200):
        inc.append(eng.marginal_reach(st, spot))
        eng.add(st, spot)
    assert all(a >= b - 1e-9 for a, b in zip(inc, inc[1:]))
    # multi-channel proportional scaling is concave too
    s = _schedule(70, seed=9)
    vals = [eng.reach(s.assign(aud_abs=s["aud_abs"] * f))["reach_1plus_abs"] for f in np.linspace(0.5, 5, 10)]
    d = np.diff(vals)
    assert np.all(d >= -1e-6) and np.all(np.diff(d) <= 1e-6)


def test_sanity_constraints(est):
    rmax_sum = sum(c["rmax_pct"] for c in _params()["cells"].values()) / 100 * est.universe
    for seed in range(15):
        s = _schedule(int(np.random.default_rng(seed).integers(1, 1500)), seed=seed)
        for mult in (0.7, 1.0, 1.3):
            out = est.with_k_multiplier(mult).reach(s)
            assert 0 <= out["reach_1plus_abs"] <= est.universe
            assert out["reach_3plus_abs"] <= out["reach_1plus_abs"]
            assert out["reach_1plus_abs"] <= out["grp_abs"] + 1e-6
            assert out["ots"] >= 1 - 1e-12
            assert out["reach_1plus_abs"] <= rmax_sum
    # a single tiny spot with a steep curve: the GRP cap keeps OTS >= 1
    out = est.with_k_multiplier(0.5).reach([{"channel": "MBC 1", "aud_abs": 50_000.0}])
    assert out["reach_1plus_abs"] <= 50_000.0 + 1e-6 and out["ots"] >= 1 - 1e-12


def test_sainsbury_combination(est):
    a = [{"channel": "MBC 1", "aud_abs": 90_000.0}] * 30
    b = [{"channel": "MBC DRAMA", "aud_abs": 40_000.0}] * 25
    ra = est.reach(a)["reach_1plus_pct"] / 100
    rb = est.reach(b)["reach_1plus_pct"] / 100
    rab = est.reach(a + b)["reach_1plus_pct"] / 100
    assert rab == pytest.approx(1 - (1 - ra) * (1 - rb), rel=1e-12)
    assert curves.sainsbury([ra, rb]) == pytest.approx(rab)


def test_single_channel_curve_values(est):
    # hyperbolic R = Rmax g / (k + g), k = Rmax (slope 1) -> 50 * 40 / 90
    spots = [{"channel": "MBC 1", "aud_abs": 0.01 * est.universe}] * 40  # 40 GRPs
    assert est.reach(spots)["reach_1plus_pct"] == pytest.approx(50 * 40 / 90)


def test_sensitivity_ordering(est):
    s = _schedule(400, seed=11)
    rows = est.sensitivity(s)
    assert [r["k_multiplier"] for r in rows] == pytest.approx([0.7, 1.0, 1.3])
    assert all(r["method"] == "ESTIMATE" for r in rows)
    assert rows[0]["reach_1plus_abs"] > rows[1]["reach_1plus_abs"] > rows[2]["reach_1plus_abs"]
    base = est.reach(s)
    assert rows[1]["reach_1plus_abs"] == pytest.approx(base["reach_1plus_abs"])


def test_frequency_pmf_properties():
    pmf = curves.cell_frequency_pmf(120.0, 40.0, 200, 10)
    assert pmf.sum() == pytest.approx(1.0)
    assert pmf[0] == pytest.approx(0.6)
    assert (pmf * np.arange(11)).sum() <= 1.2 + 1e-9          # top bucket lumps the tail
    # above-Poisson reach falls back to a zero-truncated Poisson with the right OTS
    pmf = curves.cell_frequency_pmf(10.0, 9.9, 50, 10)
    assert pmf[0] == pytest.approx(1 - 0.099)
    assert (pmf * np.arange(11)).sum() == pytest.approx(0.10, rel=1e-6)
    # never more exposures than spots
    pmf = curves.cell_frequency_pmf(3.0, 2.9, 2, 10)
    assert pmf[3:].sum() == 0


def test_duplication_matrix_two_cells(cfg):
    p = _params("CALIBRATED")
    cells = list(p["cells"])
    phi = np.ones((7, 7))
    phi[0, 5] = phi[5, 0] = 2.0
    p["duplication"] = {"cells": cells, "phi": phi.tolist()}
    eng = ReachEngine.from_config(_cfg_mode(cfg, "CALIBRATED", duplication="matrix"), params=p)
    a = [{"channel": "MBC 1", "aud_abs": 90_000.0}] * 30
    b = [{"channel": "MBC DRAMA", "aud_abs": 40_000.0}] * 25
    ra, rb = eng.reach(a)["reach_1plus_pct"] / 100, eng.reach(b)["reach_1plus_pct"] / 100
    assert eng.reach(a + b)["reach_1plus_pct"] / 100 == pytest.approx(ra + rb - 2.0 * ra * rb)
    # marginal still matches evaluate difference with the matrix
    st = eng.new_state()
    for sp in eng.prepare_many(_schedule(120, seed=2)):
        before = st.reach_abs
        m = eng.marginal_reach(st, sp)
        eng.add(st, sp)
        assert m == pytest.approx(st.reach_abs - before, abs=1e-6)
        assert m >= -1e-9


def test_real_params_file_if_present(cfg):
    p = abs_path(cfg["reach"]["params_path"])
    if not p.exists():
        pytest.skip("reach_params.json not fitted yet")
    eng = ReachEngine.from_config(_cfg_mode(cfg, "ESTIMATE"))
    out = eng.reach(_schedule(300))
    _contract(out, "ESTIMATE")
    assert set(out["params_summary"]["cells"]) == set(CHANNELS)


# ----------------------------------------------------------------------------- fit step
def test_fit_turnover_recovers_lambda():
    rng = np.random.default_rng(0)
    a = rng.uniform(5e3, 2e5, 2000)
    L = rng.integers(10, 400, 2000).astype(float)
    df = pd.DataFrame({"rating_abs": a, "break_sec": L, "reach_abs": a * (1.01 + 4e-4 * L)})
    t = fit_turnover(df)
    assert t["r0"] == pytest.approx(1.01, rel=1e-6)
    assert t["lambda_per_sec"] == pytest.approx(4e-4, rel=1e-6)


def test_fit_estimate_params_synthetic(cfg):
    rng = np.random.default_rng(1)
    rows = []
    for ch, aud in [("X", 60e3), ("Y", 10e3)]:
        for d in pd.date_range("2026-09-01", "2026-09-14"):
            for h in range(3, 27):
                for j in range(3):
                    L = float(rng.integers(30, 300))
                    a = aud * rng.uniform(0.5, 1.5)
                    rows.append({"channel": ch, "broadcast_date": d, "start_sec": h * 3600 + j * 1000,
                                 "break_sec": L, "rating_abs": a, "reach_abs": a * (1 + 5e-4 * L),
                                 "is_event": False})
    b = pd.DataFrame(rows)
    b.attrs["path"] = "synthetic"
    p = fit_estimate_params(cfg, b)
    assert p["method"] == "ESTIMATE" and p["form"] in curves.FORMS
    x = p["cells"]["X"]
    assert x["tau_min"] == pytest.approx(1 / 5e-4 / 60, rel=1e-3)
    assert x["initial_slope"] <= cfg["reach"]["estimate"]["initial_slope_cap"]
    assert x["k"] == pytest.approx(x["rmax_pct"] / x["initial_slope"])
    assert x["daily_cume_lb_pct"] <= x["daily_cume_pct"] <= x["daily_cume_ub_pct"]
    assert x["rmax_pct"] > p["cells"]["Y"]["rmax_pct"]
    assert x["rmax_pct"] >= x["daily_cume_pct"]


def test_beta_day_cume_limits():
    assert curves.beta_day_cume(0.1, 0.999, 28) == pytest.approx(0.1, abs=5e-3)
    assert curves.beta_day_cume(0.1, 1e-6, 28) == pytest.approx(1 - 0.9 ** 28, rel=1e-3)


# ----------------------------------------------------------------------------- EXACT (Nielsen fixtures)
_F26_W = {"A": 900, "B": 1100, "C": 1000, "D": 800}
_F26_ITEMS = [(0, "A|B|C"), (0, "B|C|D"), (0, "C|D"), (0, "C|D")]
# MDT p18 / p84 / p90 viewing table: day 0 then day 1
_P84_ITEMS = [(0, "A|B|C"), (0, "B|C|D"), (0, "C|D"), (0, "C|D"),
              (1, "A|B|C"), (1, "B|C"), (1, "C|D"), (1, "D|E")]
_F27_W = {0: {"A": 900, "B": 1100, "C": 1000, "D": 800},
          1: {"A": 950, "B": 1200, "C": 1300, "D": 900, "E": 700}}
# F39 (MDT p90) average weights A 925, B 1100, C 1150, D 850, E 700
_F39_W = {0: {"A": 900, "B": 1000, "C": 1000, "D": 800},
          1: {"A": 950, "B": 1200, "C": 1300, "D": 900, "E": 700}}
DAY0 = pd.Timestamp("2026-10-04")


def _respondents(items, weights, seconds=30.0):
    """Tiny respondent file: one weight-only row per (person, day) + one viewing row per
    (person, spot); spot i airs on 'MBC 1' at minute 600 + 10 i of its day."""
    rows = []
    for d, ws in weights.items():
        for p, w in ws.items():
            rows.append({"panelist_id": p, "date": DAY0 + pd.Timedelta(days=d), "daily_weight": w,
                         "channel": None, "view_start": None, "view_end": None})
    spots = []
    for i, (d, viewers) in enumerate(items):
        start = 600 + 10 * i
        spots.append({"slot_id": f"s{i}", "channel": "MBC 1", "air_date": DAY0 + pd.Timedelta(days=d),
                      "start_min": start, "aud_abs": 1.0})
        for p in viewers.split("|"):
            rows.append({"panelist_id": p, "date": DAY0 + pd.Timedelta(days=d),
                         "daily_weight": weights[d][p], "channel": "MBC 1",
                         "view_start": start * 60.0, "view_end": start * 60.0 + seconds})
    return pd.DataFrame(rows), pd.DataFrame(spots)


def _exact_engine(cfg, items, weights, min_seconds=1, rule="average", seconds=30.0):
    df, spots = _respondents(items, weights, seconds)
    c = _cfg_mode(cfg, "EXACT", min_seconds=min_seconds, common_weight_rule=rule)
    return ReachEngine.from_config(c, respondents=df), spots


def test_exact_F26_F38_single_day(cfg):
    eng, spots = _exact_engine(cfg, _F26_ITEMS, {0: _F26_W})
    out = eng.reach(spots)
    assert out["reach_1plus_abs"] == pytest.approx(3800)              # F26 (MDT p83)
    assert out["reach_n_plus_abs"][2] == pytest.approx(2900)          # F38 (MDT p89, erratum 2800)
    assert out["reach_3plus_abs"] == pytest.approx(1800)              # F38
    assert out["grp_abs"] == pytest.approx(3000 + 2900 + 1800 + 1800)  # daily weights (GL p38)
    assert out["ots"] == pytest.approx(9500 / 3800)
    assert out["method"] == "EXACT"


def test_exact_F27_multi_day_common_weights(cfg):
    eng, spots = _exact_engine(cfg, _P84_ITEMS, _F27_W)
    assert eng.reach(spots)["reach_1plus_abs"] == pytest.approx(4775)  # F27 (MDT p18)


def test_exact_F39_reach_n_plus_multi_day(cfg):
    eng, spots = _exact_engine(cfg, _P84_ITEMS, _F39_W)
    out = eng.reach(spots)
    assert out["reach_1plus_abs"] == pytest.approx(4725)               # F39 (MDT p90)
    assert out["reach_n_plus_abs"][2] == pytest.approx(4025)
    assert out["reach_3plus_abs"] == pytest.approx(3100)


def test_exact_incremental_matches_evaluate(cfg):
    eng, spots = _exact_engine(cfg, _P84_ITEMS, _F39_W)
    st = eng.new_state()
    for sp in eng.prepare_many(spots):
        before = eng.evaluate(st)
        m, m3 = eng.marginal_reach(st, sp), eng.marginal_reach_n(st, sp, 3)
        eng.add(st, sp)
        after = eng.evaluate(st)
        assert m == pytest.approx(after["reach_1plus_abs"] - before["reach_1plus_abs"])
        assert m3 == pytest.approx(after["reach_3plus_abs"] - before["reach_3plus_abs"])
    last = list(st.spots.values())[-1]
    eng.remove(st, last)
    assert st.reach_abs == pytest.approx(4725 - 700)


def test_exact_min_seconds_threshold(cfg):
    eng, spots = _exact_engine(cfg, _F26_ITEMS, {0: _F26_W}, min_seconds=15, seconds=10.0)
    assert eng.reach(spots)["reach_1plus_abs"] == 0
    eng, spots = _exact_engine(cfg, _F26_ITEMS, {0: _F26_W}, min_seconds=10, seconds=10.0)
    assert eng.reach(spots)["reach_1plus_abs"] == pytest.approx(3800)


def test_exact_requires_min_seconds(cfg):
    df, _ = _respondents(_F26_ITEMS, {0: _F26_W})
    with pytest.raises(ValueError, match="min_seconds"):
        ReachEngine.from_config(_cfg_mode(cfg, "EXACT", min_seconds=None), respondents=df)


def test_load_respondent_file_lists_missing_columns(tmp_path):
    f = tmp_path / "r.csv"
    pd.DataFrame({"panelist_id": [1], "date": ["2026-10-04"], "channel": ["MBC 1"]}).to_csv(f, index=False)
    with pytest.raises(RespondentFileError) as e:
        load_respondent_file(f)
    for c in ("daily_weight", "view_start", "view_end"):
        assert c in str(e.value)


def test_load_respondent_file_roundtrip(tmp_path):
    df, _ = _respondents(_F26_ITEMS, {0: _F26_W})
    f = tmp_path / "r.parquet"
    df.to_parquet(f)
    out = load_respondent_file(f)
    assert out["is_viewing"].sum() == 10


# ----------------------------------------------------------------------------- CALIBRATED
def test_calibrated_recovers_synthetic(cfg):
    truth = {"MBC 1": {"rmax_pct": 55.0, "k": 60.0}, "MBC 4": {"rmax_pct": 25.0, "k": 20.0},
             "MBC DRAMA": {"rmax_pct": 30.0, "k": 45.0}}
    s, sp = synthetic_rf(truth, "hyperbolic", cfg["target"]["universe"], 40, seed=3)
    p = fit_calibrated(cfg, s, sp, forms=("hyperbolic",))
    for ch, t in truth.items():
        assert p["cells"][ch]["rmax_pct"] == pytest.approx(t["rmax_pct"], rel=1e-3)
        assert p["cells"][ch]["k"] == pytest.approx(t["k"], rel=1e-3)
    assert p["n_train"] == 32 and p["n_valid"] == 8
    assert p["validation_error"]["reach_1plus"]["max_abs_points"] < 1e-3
    eng = ReachEngine.from_config(_cfg_mode(cfg, "CALIBRATED"), params=p)
    _contract(eng.reach(sp[sp.schedule_id == s.schedule_id[0]]), "CALIBRATED")


def test_calibrated_recovers_with_noise_and_matrix(cfg):
    truth = {"MBC 1": {"rmax_pct": 50.0, "k": 40.0}, "MBC DRAMA": {"rmax_pct": 28.0, "k": 30.0}}
    phi = np.array([[1.0, 1.8], [1.8, 1.0]])
    c = copy.deepcopy(cfg)
    c["reach"]["calibration"]["fit_reach_levels"] = [1, 2, 3]
    s, sp = synthetic_rf(truth, "negexp", cfg["target"]["universe"], 50, seed=4, phi=phi, noise_points=0.2)
    p = fit_calibrated(c, s, sp, phi=phi)
    assert p["form"] == "negexp"
    for ch, t in truth.items():
        assert p["cells"][ch]["rmax_pct"] == pytest.approx(t["rmax_pct"], rel=0.1)
    assert p["validation_error"]["reach_1plus"]["mae_points"] < 0.5
    assert p["duplication"]["phi"][0][1] == pytest.approx(1.8)


def test_calibration_loader_errors():
    s = pd.DataFrame({"schedule_id": ["a"], "grp_pct": [10.0]})
    sp = pd.DataFrame({"schedule_id": ["a"], "channel": ["MBC 1"], "aud_abs": [1.0]})
    with pytest.raises(CalibrationFileError, match="reach_1plus_pct"):
        load_rf_schedules(s, sp)
    s["reach_1plus_pct"] = 5.0
    with pytest.raises(CalibrationFileError, match="unknown schedule_id"):
        load_rf_schedules(s, pd.concat([sp, sp.assign(schedule_id="b")]))


def test_channel_daypart_cells(cfg):
    c = copy.deepcopy(cfg)
    c["reach"]["estimate"]["cell_granularity"] = "channel_daypart"
    rng = np.random.default_rng(2)
    rows = []
    for d in pd.date_range("2026-09-01", "2026-09-07"):
        for h in range(3, 27):
            for j in range(3):
                L = float(rng.integers(30, 300))
                a = 40e3 * rng.uniform(0.5, 1.5) * (2 if 18 <= h < 24 else 1)
                rows.append({"channel": "MBC 1", "broadcast_date": d, "start_sec": h * 3600 + j * 1000,
                             "break_sec": L, "rating_abs": a, "reach_abs": a * (1 + 4e-4 * L), "is_event": False})
    b = pd.DataFrame(rows)
    b.attrs["path"] = "synthetic"
    p = fit_estimate_params(c, b)
    assert set(p["cells"]) == {f"MBC 1|{dp}" for dp in c["reach"]["estimate"]["dayparts"]}
    assert p["cells"]["MBC 1|prime"]["rmax_pct"] > p["cells"]["MBC 1|early"]["rmax_pct"]
    eng = ReachEngine.from_config(c, params=p)
    out = eng.reach([{"channel": "MBC 1", "start_min": 1200, "aud_abs": 80e3},
                     {"channel": "MBC 1", "start_min": 400, "aud_abs": 40e3},
                     {"channel": "MBC 1", "start_min": 1300, "daypart": "prime", "aud_abs": 80e3}])
    assert set(out["per_cell"]) == {"MBC 1|prime", "MBC 1|morning"}
    assert out["per_cell"]["MBC 1|prime"]["n_spots"] == 2


def test_exact_cyclic_weekday_mapping(cfg):
    df, spots = _respondents(_F26_ITEMS, {0: _F26_W})    # panel day = Sun 2026-10-04
    c = _cfg_mode(cfg, "EXACT", min_seconds=1)
    c["reach"]["exact"] = {**c["reach"]["exact"], "date_mapping": "cyclic_weekday"}
    eng = ReachEngine.from_config(c, respondents=df)
    future = spots.assign(air_date=pd.Timestamp("2026-10-25"))  # a Sunday three weeks later
    assert eng.reach(future)["reach_1plus_abs"] == pytest.approx(3800)
    ident = ReachEngine.from_config(_cfg_mode(cfg, "EXACT", min_seconds=1), respondents=df)
    with pytest.raises(KeyError, match="cyclic_weekday"):
        ident.reach(future)
