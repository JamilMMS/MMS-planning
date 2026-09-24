"""Tests for optimizer.ingest.grid.

Pure-function tests always run (no data/raw dependency). Real-file tests skip if
data/raw/grid/october_grid.xlsx is absent.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from optimizer.config import load_config, project_root
from optimizer.ingest.grid import (
    _assign_overlap_groups,
    _find_carry_forward_week,
    _split_title,
    find_overlaps,
    hhmm_to_min,
    load_grid,
)

ROOT = project_root()
GRID_PATH = ROOT / "data" / "raw" / "grid" / "october_grid.xlsx"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _skip_if_no_grid():
    if not GRID_PATH.exists():
        pytest.skip("data/raw/grid/october_grid.xlsx not present")


# --------------------------------------------------------------------------- pure functions

def test_hhmm_to_min():
    assert hhmm_to_min(900) == 9 * 60
    assert hhmm_to_min(2000) == 20 * 60
    assert hhmm_to_min(2630) == 26 * 60 + 30  # broadcast-day time past midnight


def test_split_title_standard_separator():
    en, ar = _split_title("AHLA NASEEB S1 / أحلى نصيب")
    assert en == "AHLA NASEEB S1"
    assert ar == "أحلى نصيب"


@pytest.mark.parametrize("raw,expected_en", [
    ("AL LIQAA MIN AL SIFR S8 /اللقاء من الصفر", "AL LIQAA MIN AL SIFR S8"),
    ("BIG TIME PODCAST S1 /كريم عبد العزيز", "BIG TIME PODCAST S1"),
    ("EL MADDAH: OSTOURET EL ISHQ /المداح: أسطورة العشق", "EL MADDAH: OSTOURET EL ISHQ"),
    ("MALEK BEL TAWEELA S8 /مالك بالطويلة", "MALEK BEL TAWEELA S8"),
])
def test_split_title_no_space_separator(raw, expected_en):
    """The reference parser's ' / ' literal split leaves the Arabic half inside title_en for
    these 4 real programme names (no space between '/' and the Arabic text)."""
    en, ar = _split_title(raw)
    assert en == expected_en
    assert ar is not None
    assert not any("؀" <= ch <= "ۿ" for ch in en), "Arabic leaked into title_en"


@pytest.mark.parametrize("raw", [
    "KINGDOM S4/S5",
    "POLICE 24/7 S1",
    "BUNDESLIGA HIGHLIGHTS   (2026/27)",
    "THE EVOLUTION OF SPORT S5/S6/S7",
])
def test_split_title_non_arabic_slash_untouched(raw):
    """Slashes with no Arabic on the right (season ranges, "24/7", year ranges) must not be
    split -- there is no Arabic half to extract."""
    en, ar = _split_title(raw)
    assert en == raw
    assert ar is None


def test_no_arabic_in_title_en_synthetic():
    """Pure synthetic-frame version of the Arabic-leak assertion in load_grid."""
    titles = pd.Series([
        "AL LIQAA MIN AL SIFR S8 /اللقاء من الصفر",
        "KINGDOM S4/S5",
        "AHLA NASEEB S1 / أحلى نصيب",
    ])
    for t in titles:
        en, _ = _split_title(t)
        assert not any("؀" <= ch <= "ۿ" for ch in en)


def _tiny_grid_frame():
    """A minimal synthetic frame with the columns load_grid's internals expect, used to test
    _assign_overlap_groups and _find_carry_forward_week in isolation."""
    return pd.DataFrame({
        "channel": ["X", "X", "X", "Y", "Y"],
        "air_date": pd.to_datetime(["2026-10-01"] * 3 + ["2026-10-01"] * 2),
        "start_min": [0, 30, 200, 0, 500],
        "end_min": [60, 90, 260, 100, 560],
        "slot_id": ["a", "b", "c", "d", "e"],
    })


def test_assign_overlap_groups_pure():
    df = _tiny_grid_frame()
    groups = _assign_overlap_groups(df)
    # a [0,60) and b [30,90) overlap -> same group; c [200,260) overlaps nothing -> -1
    assert groups.loc[df.index[df.slot_id == "a"]].iloc[0] == groups.loc[df.index[df.slot_id == "b"]].iloc[0]
    assert groups.loc[df.index[df.slot_id == "a"]].iloc[0] != -1
    assert groups.loc[df.index[df.slot_id == "c"]].iloc[0] == -1
    # d and e (different channel Y, non-overlapping) -> both -1
    assert groups.loc[df.index[df.slot_id == "d"]].iloc[0] == -1
    assert groups.loc[df.index[df.slot_id == "e"]].iloc[0] == -1


def test_assign_overlap_groups_containment():
    """A containing interval overlapping several nested rows -- not just adjacent-in-sort-
    order -- must all land in one group (the MBC BOLLYWOOD 'compilation block' pattern)."""
    df = pd.DataFrame({
        "channel": ["X"] * 4,
        "air_date": pd.to_datetime(["2026-10-01"] * 4),
        "start_min": [0, 10, 20, 30],
        "end_min": [100, 20, 30, 40],  # row0 spans [0,100) and contains rows 1,2,3
        "slot_id": ["big", "n1", "n2", "n3"],
    })
    groups = _assign_overlap_groups(df)
    assert len(set(groups)) == 1  # all 4 rows in one group
    assert -1 not in set(groups)


def test_find_carry_forward_week_pure():
    g = pd.DataFrame({
        "week": [20261004, 20261004, 20261011, 20261011],
        "channel": ["MBC 1", "MBC 2", "MBC 2", "MBC 1"],
    })
    # MBC 1 has 20261004 and 20261011 -> no gap
    assert _find_carry_forward_week(g) is None

    g2 = pd.DataFrame({
        "week": [20261004, 20261004, 20261011],
        "channel": ["MBC 1", "MBC 2", "MBC 2"],  # MBC 1 missing 20261011
    })
    found = _find_carry_forward_week(g2)
    assert found == (20261011, 20261004)


def test_wrap_day_time_conversion():
    """Broadcast-day round-trip: grid HHMM 2630 on air_date D -> real clock datetime
    D+1 02:30 (see CLAUDE.md 'Broadcast day runs 03:00-26:59')."""
    air_date = pd.Timestamp("2026-10-08")
    start_min = hhmm_to_min(2630)
    start_dt = air_date + pd.to_timedelta(start_min, unit="m")
    assert start_dt == pd.Timestamp("2026-10-09 02:30:00")


# --------------------------------------------------------------------------- real-file tests

def test_load_grid_row_counts(cfg):
    _skip_if_no_grid()
    g = load_grid(GRID_PATH, cfg)
    orig = g[~g["is_synthetic"]]
    assert len(orig) == 3760
    expected = {"MBC 1": 603, "MBC 2": 295, "MBC 4": 755, "MBC ACTION": 779,
                "MBC BOLLYWOOD": 392, "MBC DRAMA": 672, "MBC MAX": 264}
    for ch, n in expected.items():
        assert int((orig["channel"] == ch).sum()) == n, ch


def test_load_grid_rerun_live_cost(cfg):
    _skip_if_no_grid()
    g = load_grid(GRID_PATH, cfg)
    orig = g[~g["is_synthetic"]]
    assert int(orig["is_rerun"].sum()) == 2472
    assert int(orig["is_live"].sum()) == 75
    assert round(float(orig["rate_usd"].sum()), 2) == 3201960.52
    assert round(float(orig[orig["channel"] == "MBC 1"]["rate_usd"].sum()), 2) == 2266603.13


def test_load_grid_slot_id_unique(cfg):
    _skip_if_no_grid()
    g = load_grid(GRID_PATH, cfg)
    assert g["slot_id"].is_unique


def test_load_grid_no_arabic_in_title_en(cfg):
    _skip_if_no_grid()
    g = load_grid(GRID_PATH, cfg)
    assert not g["title_en"].str.contains(r"[؀-ۿ]", regex=True, na=False).any()


def test_load_grid_rate_aed_integral(cfg):
    _skip_if_no_grid()
    g = load_grid(GRID_PATH, cfg)
    off_by = (g["rate_aed"] - g["rate_aed"].round(0)).abs()
    assert off_by.max() < 0.01


def test_load_grid_overlap_groups_count(cfg):
    _skip_if_no_grid()
    g = load_grid(GRID_PATH, cfg)
    n_groups = int(g.loc[g["overlap_group"] >= 0, "overlap_group"].nunique())
    assert n_groups == 8
    ov = find_overlaps(g)
    assert len(ov) > 0
    # every slot_id appearing in an overlap pair actually has overlap_group != -1
    involved = set(ov["slot_a"]).union(ov["slot_b"])
    assert (g.set_index("slot_id").loc[list(involved), "overlap_group"] >= 0).all()


def test_load_grid_carry_forward_matches_week3(cfg):
    _skip_if_no_grid()
    g = load_grid(GRID_PATH, cfg)
    synthetic = g[g["is_synthetic"]]
    assert (synthetic["assumption"] == "ASSUMPTION_MBC1_WK4").all()
    missing_week, source_week = g.attrs["carry_forward_weeks"]
    week3_count = int(((~g["is_synthetic"]) & (g["channel"] == "MBC 1") & (g["week"] == source_week)).sum())
    assert len(synthetic) == week3_count
    assert len(synthetic) > 0


def test_load_grid_pre_flight_rows_kept_not_dropped(cfg):
    _skip_if_no_grid()
    g = load_grid(GRID_PATH, cfg)
    pre_flight = g[~g["in_flight"]]
    assert len(pre_flight) == g.attrs["n_pre_flight_rows"]
    assert len(pre_flight) == 2  # week 20260927 rows, kept not dropped
    assert (pre_flight["week"] == 20260927).all()


def test_load_grid_day_mask_exactly_one_day(cfg):
    _skip_if_no_grid()
    raw = pd.read_excel(GRID_PATH)
    masks = raw["day_mask"].map(lambda m: sum(1 for c in str(m) if c != "_"))
    assert (masks == 1).all()
