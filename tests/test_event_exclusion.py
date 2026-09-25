"""Regression test for Gate 1 CHANGE 1 (per-channel 19-Sep event exclusion policy).

Proves the MBC ACTION event-window exclusion actually bites on a real downstream number: the
naive slot match (same method as reference_scripts/naive_slot_match.py, which produced the
PRELIMINARY_FINDINGS "$1.91 CPM" figure) for October's "MR. BEAN S1 ®" MBC ACTION Saturday
18:00 slot must look like normal Saturday-evening ACTION CPM once is_event rows are excluded,
and must reproduce the original (bogus) $1.91 figure when they are not.

Skips if data/raw/grid or data/raw/etam is absent.
"""
from __future__ import annotations

import pandas as pd
import pytest

from optimizer.config import load_config, project_root
from optimizer.ingest.breaks import load_breaks
from optimizer.ingest.grid import load_grid

ROOT = project_root()
GRID_PATH = ROOT / "data" / "raw" / "grid" / "october_grid.xlsx"
BREAKS_PATH = ROOT / "data" / "raw" / "etam" / "etam_breaks_2026-09-01_to_09-21_arabs15plus.xlsx"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _skip_if_no_data():
    if not GRID_PATH.exists() or not BREAKS_PATH.exists():
        pytest.skip("data/raw/grid or data/raw/etam not present")


def _mr_bean_naive_cpm(g: pd.DataFrame, b: pd.DataFrame, exclude_events: bool) -> tuple[float, int]:
    """Naive slot match per reference_scripts/naive_slot_match.py: same channel + weekday,
    break start within [slot start, slot end) (the actual grid slot window, looked up from
    grid.parquet -- not a hand-picked hour band). Returns (cpm_usd, n_breaks_matched)."""
    row = g[
        (g["channel"] == "MBC ACTION")
        & (g["weekday"] == "Sat")
        & (g["title_en"].str.contains("MR. BEAN", case=False, na=False))
        & (g["start_hhmm"] == 1800)
    ].iloc[0]

    b = b.copy()
    b["start_min"] = b["start_sec"] / 60
    m = b[
        (b["channel"] == "MBC ACTION")
        & (b["weekday"] == "Sat")
        & (b["start_min"] >= row["start_min"])
        & (b["start_min"] < row["end_min"])
    ]
    if not exclude_events:
        matched = m
    else:
        matched = m[~m["is_event"]]

    mean_aud = matched["rating_abs"].mean()
    cpm = row["rate_usd"] / (mean_aud / 1000)
    return float(cpm), int(len(matched))


def test_mr_bean_naive_cpm_normal_after_event_exclusion(cfg):
    _skip_if_no_data()
    g = load_grid(GRID_PATH, cfg)
    b = load_breaks(BREAKS_PATH, cfg)

    cpm_excl, n_excl = _mr_bean_naive_cpm(g, b, exclude_events=True)
    cpm_incl, n_incl = _mr_bean_naive_cpm(g, b, exclude_events=False)

    # Excluding is_event rows: normal Saturday-evening MBC ACTION CPM (Regular tier, per
    # PRELIMINARY_FINDINGS section 3 the ACTION Regular median CPM is ~$155) -- comfortably
    # above a $30 floor.
    assert cpm_excl >= 30.0, f"naive CPM with event exclusion too low: ${cpm_excl:.2f}"

    # Without exclusion: reproduces the PRELIMINARY_FINDINGS "$1.91" figure (the FIFA/NADEENA
    # 19-Sep break inflates the average by ~2 orders of magnitude) -- proves the exclusion
    # actually bites, it isn't a no-op.
    assert cpm_incl < 5.0, f"naive CPM without exclusion should reproduce the ~$1.91 bogus figure, got ${cpm_incl:.2f}"
    assert cpm_incl == pytest.approx(1.91, abs=0.05)
    assert cpm_excl > cpm_incl * 5  # exclusion must materially change the number, not just nudge it


def test_no_action_event_rows_remain_inside_window_bounds(cfg):
    """Every MBC ACTION break on 19-Sep whose start falls inside the detected event window
    bounds must be flagged is_event -- i.e. excluding is_event rows removes the window
    completely, not just the original 8 match rows."""
    _skip_if_no_data()
    b = load_breaks(BREAKS_PATH, cfg)
    windows = {w["channel"]: w for w in b.attrs["event_windows"]}
    w = windows["MBC ACTION"]

    day = b[(b["channel"] == "MBC ACTION") & (b["broadcast_date"] == pd.Timestamp(w["broadcast_date"]))]
    in_window = day[(day["start_sec"] >= w["window_start_sec"]) & (day["start_sec"] <= w["window_end_sec"])]
    assert len(in_window) == w["n_breaks"]
    assert in_window["is_event"].all(), "every break inside the detected window must be is_event=True"

    remaining_non_event = in_window[~in_window["is_event"]]
    assert len(remaining_non_event) == 0


def test_mbc1_still_has_non_event_breaks_on_19_sep(cfg):
    """MBC 1's policy is match_only (CHANGE 1): only the 8 FIFA/NADEENA match breaks are
    excluded, its other 19-Sep programming stays in baselines."""
    _skip_if_no_data()
    b = load_breaks(BREAKS_PATH, cfg)
    mbc1_19sep = b[(b["channel"] == "MBC 1") & (b["broadcast_date"] == pd.Timestamp("2026-09-19"))]
    assert len(mbc1_19sep) > 8
    non_event = mbc1_19sep[~mbc1_19sep["is_event"]]
    assert len(non_event) > 0
    assert len(non_event) == len(mbc1_19sep) - 8
