"""Tests for optimizer.ingest.breaks.

Pure-function tests always run. Real-file tests skip if
data/raw/etam/etam_breaks_2026-09-01_to_09-21_arabs15plus.xlsx is absent.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from optimizer.config import load_config, project_root
from optimizer.ingest.breaks import infer_universe, load_breaks, to_seconds, trp_invariant_ratio

ROOT = project_root()
BREAKS_PATH = ROOT / "data" / "raw" / "etam" / "etam_breaks_2026-09-01_to_09-21_arabs15plus.xlsx"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _skip_if_no_breaks():
    if not BREAKS_PATH.exists():
        pytest.skip("data/raw/etam/etam_breaks_2026-09-01_to_09-21_arabs15plus.xlsx not present")


# --------------------------------------------------------------------------- pure functions

def test_to_seconds_time_before_midnight():
    """datetime.time 03:06:09 -> 11169 s (the DATA_SPEC round-trip example)."""
    assert to_seconds(dt.time(3, 6, 9)) == 11169


def test_to_seconds_timedelta_after_midnight():
    """eTAM renders after-midnight-same-broadcast-day times as e.g. '1 day, 0:41:34';
    pandas reads them back as a Timedelta. -> 88894 s (the DATA_SPEC round-trip example)."""
    td = pd.Timedelta("1 day, 0:41:34")
    assert to_seconds(td) == 88894


def test_to_seconds_unparseable_returns_nan():
    import math
    assert math.isnan(to_seconds("not a time"))
    assert math.isnan(to_seconds(None))


def test_rating_abs_formula_synthetic():
    """The non-negotiable rule: rating_abs = trp_abs * 60 / break_sec (NOT trp_abs itself)."""
    df = pd.DataFrame({"trp_abs": [125241.0, 60000.0], "break_sec": [139, 60]})
    rating_abs = df["trp_abs"] * 60 / df["break_sec"]
    assert rating_abs.iloc[0] == pytest.approx(125241 * 60 / 139)
    assert rating_abs.iloc[1] == pytest.approx(60000.0)  # 60s break -> rating_abs == trp_abs


def test_break_sec_inclusive_formula():
    start_sec, end_sec = 11169, 11288  # 03:06:09 -> 03:08:08
    assert end_sec - start_sec + 1 == 120


def test_infer_universe_synthetic():
    df = pd.DataFrame({
        "rating_pct": [0.5, 0.5, 0.1],  # last row excluded (< 0.3)
        "reach_abs": [50000, 55000, 1000],
        "reach_pct": [0.5, 0.55, 0.01],
    })
    U = infer_universe(df)
    assert U["n"] == 2
    assert U["median"] == pytest.approx(10000000, rel=0.01)


# --------------------------------------------------------------------------- real-file tests

def test_load_breaks_row_count_and_target(cfg):
    _skip_if_no_breaks()
    df = load_breaks(BREAKS_PATH, cfg)
    assert len(df) == 12876
    assert set(df["target"].unique()) == {"TP Arabs 15+"}
    assert str(df["broadcast_date"].min().date()) == "2026-09-01"
    assert str(df["broadcast_date"].max().date()) == "2026-09-21"


def test_load_breaks_length_stats(cfg):
    _skip_if_no_breaks()
    df = load_breaks(BREAKS_PATH, cfg)
    assert int(df["break_sec"].median()) == 134
    assert round(float(df["break_sec"].mean()), 1) == pytest.approx(131.3, abs=0.2)
    assert int(df["break_sec"].min()) == 2
    assert int(df["break_sec"].max()) == 796
    assert (df["break_sec"] > 0).all()


def test_trp_invariant_ratio_real(cfg):
    _skip_if_no_breaks()
    df = load_breaks(BREAKS_PATH, cfg)
    U = infer_universe(df)
    ratio = trp_invariant_ratio(df, U["median"])
    assert len(ratio) == 3211
    assert abs(ratio.median() - 1.00) <= 0.01
    assert ratio.std() < 0.01


def test_rating_abs_trp_identity_real(cfg):
    """rating_abs * break_minutes == trp_abs, exactly (the defining identity), on every row."""
    _skip_if_no_breaks()
    df = load_breaks(BREAKS_PATH, cfg)
    recomputed = df["rating_abs"] * (df["break_sec"] / 60)
    assert (recomputed - df["trp_abs"]).abs().max() < 1e-6


def test_reach_lt_rating_count(cfg):
    _skip_if_no_breaks()
    df = load_breaks(BREAKS_PATH, cfg)
    assert int(df["reach_lt_rating"].sum()) == 51
    gap = (df["rating_pct"] - df["reach_pct"])[df["reach_lt_rating"]]
    assert round(float(gap.max()), 2) == 0.43


def test_low_sample_channel_flags(cfg):
    _skip_if_no_breaks()
    df = load_breaks(BREAKS_PATH, cfg)
    flagged = set(df.loc[df["low_sample_channel"], "channel"].unique())
    assert flagged == {"MBC ACTION", "MBC MAX"}


def test_is_event_flags_both_channels(cfg):
    """The 19 Sep FIFA/NADEENA football special aired simultaneously on MBC 1 AND
    MBC ACTION -- both must be flagged, not just MBC ACTION. Per-channel policy (CHANGE 1):
    MBC 1 (match_only) gets just the 8 match breaks; MBC ACTION (window) gets the whole
    15-break anomalous-audience window built around those same match breaks."""
    _skip_if_no_breaks()
    df = load_breaks(BREAKS_PATH, cfg)
    event = df[df["is_event"]]
    assert len(event) == 23
    assert set(event["channel"].unique()) == {"MBC 1", "MBC ACTION"}
    assert set(event["broadcast_date"].dt.date.astype(str).unique()) == {"2026-09-19"}
    assert int((event["channel"] == "MBC 1").sum()) == 8
    assert int((event["channel"] == "MBC ACTION").sum()) == 15

    mbc1_event = event[event["channel"] == "MBC 1"]
    assert set(mbc1_event["event_reason"].unique()) == {"SIMULCAST_MATCH"}
    action_event = event[event["channel"] == "MBC ACTION"]
    assert set(action_event["event_reason"].unique()) == {"SIMULCAST_WINDOW"}
    assert set(df.loc[~df["is_event"], "event_reason"].unique()) == {""}


def test_event_window_attrs_for_mbc_action(cfg):
    """The computed MBC ACTION event window (CHANGE 1) matches the worked example in
    etam_breaks_report.md: ~14:59:46-19:24:52, 15 breaks, 5 programmes."""
    _skip_if_no_breaks()
    df = load_breaks(BREAKS_PATH, cfg)
    windows = df.attrs["event_windows"]
    assert len(windows) == 1
    w = windows[0]
    assert w["channel"] == "MBC ACTION"
    assert w["broadcast_date"] == "2026-09-19"
    assert w["n_breaks"] == 15
    assert w["window_start_sec"] == 53986   # 14:59:46
    assert w["window_end_sec"] == 69892     # 19:24:52
    assert set(w["programmes"]) == {
        "NADEENA", "FIFA AFRICAN ASIAN PACIFIC CUP 2026", "RIDICULOUSNESS",
        "BUNDESLIGA CLUB PROFILE", "BUNDESLIGA 2026/2027",
    }


def test_break_id_unique_per_target(cfg):
    _skip_if_no_breaks()
    df = load_breaks(BREAKS_PATH, cfg)
    assert not df.duplicated(subset=["break_id", "target"]).any()


def test_nonstring_program_episode_cast(cfg):
    """At least one MBC MAX/MBC 2 programme title is a bare number in the raw file; must be
    cast to str, never raise, and never appear as e.g. numpy int64 in the output column."""
    _skip_if_no_breaks()
    df = load_breaks(BREAKS_PATH, cfg)
    assert df.attrs["n_nonstr_program"] > 0
    assert (df["program"].map(type) == str).all()
