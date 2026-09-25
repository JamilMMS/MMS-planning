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
    _assign_overlap_components,
    _find_carry_forward_week,
    _split_title,
    find_slot_conflicts,
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
    find_slot_conflicts / _assign_overlap_components and _find_carry_forward_week in isolation."""
    return pd.DataFrame({
        "channel": ["X", "X", "X", "Y", "Y"],
        "air_date": pd.to_datetime(["2026-10-01"] * 3 + ["2026-10-01"] * 2),
        "start_min": [0, 30, 200, 0, 500],
        "end_min": [60, 90, 260, 100, 560],
        "slot_id": ["a", "b", "c", "d", "e"],
    })


def test_find_slot_conflicts_pure():
    df = _tiny_grid_frame()
    conflicts = find_slot_conflicts(df)
    # a [0,60) and b [30,90) overlap -> one pair; c [200,260) overlaps nothing -> no pair
    pairs = set(zip(conflicts["slot_a"], conflicts["slot_b"]))
    assert pairs == {("a", "b")}
    # d and e (different channel Y, non-overlapping) -> no pairs at all involving them
    assert not any("d" in p or "e" in p for p in pairs)
    assert not any("c" in p for p in pairs)


def test_assign_overlap_components_pure():
    df = _tiny_grid_frame()
    conflicts = find_slot_conflicts(df)
    comps = _assign_overlap_components(df, conflicts)
    assert comps.loc[df.index[df.slot_id == "a"]].iloc[0] == comps.loc[df.index[df.slot_id == "b"]].iloc[0]
    assert comps.loc[df.index[df.slot_id == "a"]].iloc[0] != -1
    assert comps.loc[df.index[df.slot_id == "c"]].iloc[0] == -1
    assert comps.loc[df.index[df.slot_id == "d"]].iloc[0] == -1
    assert comps.loc[df.index[df.slot_id == "e"]].iloc[0] == -1


def test_find_slot_conflicts_containment_is_pairwise_not_transitive():
    """A containing interval overlapping several nested rows -- not just adjacent-in-sort-
    order -- must produce one conflict PAIR per (umbrella, nested-row); the nested rows must
    NOT conflict with each other (the MBC BOLLYWOOD 'compilation block' pattern, CHANGE 2)."""
    df = pd.DataFrame({
        "channel": ["X"] * 4,
        "air_date": pd.to_datetime(["2026-10-01"] * 4),
        "start_min": [0, 10, 20, 30],
        "end_min": [100, 20, 30, 40],  # row0 spans [0,100) and contains rows 1,2,3
        "slot_id": ["big", "n1", "n2", "n3"],
    })
    conflicts = find_slot_conflicts(df)
    pairs = set(zip(conflicts["slot_a"], conflicts["slot_b"]))
    assert pairs == {("big", "n1"), ("big", "n2"), ("big", "n3")}  # umbrella-vs-each only
    assert (conflicts["kind"] == "contains").all()
    # a single connected component groups all 4 for reporting, but that is NOT the
    # exclusivity rule -- n1/n2/n3 do not conflict with each other, per `pairs` above.
    comps = _assign_overlap_components(df, conflicts)
    assert len(set(comps)) == 1
    assert -1 not in set(comps)


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


def test_load_grid_overlap_locations_count(cfg):
    _skip_if_no_grid()
    g = load_grid(GRID_PATH, cfg)
    conflicts = g.attrs["slot_conflicts"]
    n_locations = conflicts.groupby(["channel", "air_date"]).ngroups
    assert n_locations == 8  # PRELIMINARY_FINDINGS "8 overlapping slots" -- distinct locations
    assert len(conflicts) > 0
    # every slot_id appearing in a conflict pair actually has has_conflict=True and a
    # non-(-1) overlap_component
    involved = set(conflicts["slot_a"]).union(conflicts["slot_b"])
    gi = g.set_index("slot_id")
    assert gi.loc[list(involved), "has_conflict"].all()
    assert (gi.loc[list(involved), "overlap_component"] >= 0).all()


def test_bollywood_compilation_pairwise_conflicts(cfg):
    """CHANGE 2 regression: the BOLLYWOOD 'WEEKEND DRAMA COMPILATION' umbrella conflicts with
    every episode row it contains, but no two contained episode rows conflict with each other
    -- computed strictly pairwise, not as a transitive-closure group. The 6-hour Thursday
    occurrence (Oct 8) has 5 contained episodes, the most informative case; the 5-hour
    Saturday occurrence (Oct 10) has only 1 (PARINEETI)."""
    _skip_if_no_grid()
    g = load_grid(GRID_PATH, cfg)
    conflicts = g.attrs["slot_conflicts"]

    for air_date, expected_episode_count in [("2026-10-08", 5), ("2026-10-10", 1)]:
        day_rows = g[(g["channel"] == "MBC BOLLYWOOD") & (g["air_date"] == air_date)]
        # There can be more than one "WEEKEND DRAMA COMPILATION" row on the same day (e.g. a
        # separate late-night block with no episodes overlapping it) -- the one under test is
        # the one that actually has conflicts.
        umbrella = day_rows[
            day_rows["title_en"].str.contains("WEEKEND DRAMA COMPILATION", na=False) & day_rows["has_conflict"]
        ]
        assert len(umbrella) == 1, air_date
        umbrella_id = umbrella["slot_id"].iloc[0]

        day_conflicts = conflicts[(conflicts["channel"] == "MBC BOLLYWOOD") & (conflicts["air_date"] == air_date)]
        pairs = set(zip(day_conflicts["slot_a"], day_conflicts["slot_b"]))

        # the episodes CONTAINED in the umbrella are exactly the rows it conflicts with (the
        # channel/day has other, non-overlapping BOLLYWOOD programming too -- day_rows is not
        # restricted to the compilation block).
        conflict_partner_ids = {sid for pair in pairs for sid in pair if umbrella_id in pair} - {umbrella_id}
        episodes = day_rows[day_rows["slot_id"].isin(conflict_partner_ids)]
        assert len(episodes) == expected_episode_count, air_date

        # umbrella conflicts with every contained episode
        for ep_id in episodes["slot_id"]:
            pair = tuple(sorted([umbrella_id, ep_id]))
            assert pair in pairs, f"{air_date}: umbrella does not conflict with episode {ep_id}"

        # no two episodes conflict with each other
        ep_ids = sorted(episodes["slot_id"])
        for i in range(len(ep_ids)):
            for j in range(i + 1, len(ep_ids)):
                pair = tuple(sorted([ep_ids[i], ep_ids[j]]))
                assert pair not in pairs, f"{air_date}: episodes {pair} must not conflict"

        # exactly one pair per episode -- no extra pairs for this (channel, air_date)
        assert len(day_conflicts) == expected_episode_count, air_date

    # total conflict pairs: 4 Thursdays x 5 + 3 Saturdays x 1 (BOLLYWOOD) + 1 (MBC 2) = 24
    assert len(conflicts) == 24


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
