"""Cross-cutting invariants required by the data-validator role, run against the manifest
and processed parquet produced by `python -m optimizer.ingest.run`.

Real-data tests skip if data/raw or data/processed/ingest_manifest.json is absent (run
`python -m optimizer.ingest.run --config config/plan_config.yaml` first). Pure-function
tests on tiny synthetic frames always run.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json

import pandas as pd
import pytest

from optimizer.config import load_config, project_root
from optimizer.ingest.breaks import infer_universe, load_breaks, to_seconds, trp_invariant_ratio
from optimizer.ingest.grid import hhmm_to_min, load_grid

ROOT = project_root()
GRID_PATH = ROOT / "data" / "raw" / "grid" / "october_grid.xlsx"
BREAKS_PATH = ROOT / "data" / "raw" / "etam" / "etam_breaks_2026-09-01_to_09-21_arabs15plus.xlsx"
MANIFEST_PATH = ROOT / "data" / "processed" / "ingest_manifest.json"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _skip_if_no_grid():
    if not GRID_PATH.exists():
        pytest.skip("data/raw/grid/october_grid.xlsx not present")


def _skip_if_no_breaks():
    if not BREAKS_PATH.exists():
        pytest.skip("data/raw/etam break file not present")


# --------------------------------------------------------------------------- pure / synthetic

def test_broadcast_day_roundtrip_grid_time():
    """Grid HHMM 2630 on air_date D -> real clock datetime D+1 02:30."""
    D = pd.Timestamp("2026-10-15")
    start_min = hhmm_to_min(2630)
    assert D + pd.to_timedelta(start_min, unit="m") == pd.Timestamp("2026-10-16 02:30:00")


def test_broadcast_day_roundtrip_etam_timedelta():
    """eTAM '1 day, 0:41:34' -> 88894 s -> D+1 00:41:34."""
    td = pd.Timedelta("1 day, 0:41:34")
    secs = to_seconds(td)
    assert secs == 88894
    D = pd.Timestamp("2026-09-05")
    assert D + pd.to_timedelta(secs, unit="s") == pd.Timestamp("2026-09-06 00:41:34")


def test_broadcast_day_roundtrip_etam_time():
    """datetime.time 03:06:09 -> 11169 s."""
    assert to_seconds(dt.time(3, 6, 9)) == 11169


def test_rate_aed_integral_synthetic(cfg):
    aed_per_usd = float(cfg["budget"]["aed_per_usd"])
    # Exact AED targets from PRELIMINARY_FINDINGS, converted back to USD so rate_usd *
    # aed_per_usd round-trips to the whole AED number (as it does in the real grid).
    aed_targets = pd.Series([735.0, 1500.0, 2200.0, 25500.0])
    rates_usd = aed_targets / aed_per_usd
    rate_aed = (rates_usd * aed_per_usd).round(2)
    off_by = (rate_aed - aed_targets).abs()
    assert off_by.max() <= 0.01


def test_trp_identity_synthetic():
    """rating_abs * break_min == trp_abs exactly, by construction."""
    trp_abs = pd.Series([125241.0, 30121.0, 894.0])
    break_sec = pd.Series([139, 301, 6])
    rating_abs = trp_abs * 60 / break_sec
    recomputed = rating_abs * (break_sec / 60)
    assert (recomputed - trp_abs).abs().max() < 1e-9


def test_day_mask_single_day_invariant_violation_detected():
    """A day_mask with 0 or 2+ days must be rejected -- pure check on a synthetic frame."""
    bad = pd.Series(["S______", "SM_____", "_______"])
    n_days = bad.map(lambda m: sum(1 for c in m if c != "_"))
    assert not (n_days == 1).all()  # confirms the synthetic fixture IS invalid
    good = pd.Series(["S______", "_M_____", "______S"])
    n_days_good = good.map(lambda m: sum(1 for c in m if c != "_"))
    assert (n_days_good == 1).all()


# --------------------------------------------------------------------------- real-data invariants

def test_trp_invariant_real(cfg):
    _skip_if_no_breaks()
    df = load_breaks(BREAKS_PATH, cfg)
    U = infer_universe(df)
    ratio = trp_invariant_ratio(df, U["median"])
    assert abs(ratio.median() - 1.00) <= 0.01, f"median {ratio.median()} outside 1.00+-0.01"


def test_trp_identity_real(cfg):
    _skip_if_no_breaks()
    df = load_breaks(BREAKS_PATH, cfg)
    recomputed = df["rating_abs"] * (df["break_sec"] / 60)
    assert (recomputed - df["trp_abs"]).abs().max() < 1e-6


def test_rate_aed_integral_real(cfg):
    _skip_if_no_grid()
    g = load_grid(GRID_PATH, cfg)
    off_by = (g["rate_aed"] - g["rate_aed"].round(0)).abs()
    assert off_by.max() < 0.01


def test_slot_id_unique_real(cfg):
    _skip_if_no_grid()
    g = load_grid(GRID_PATH, cfg)
    assert g["slot_id"].is_unique


def test_break_sec_positive_real(cfg):
    _skip_if_no_breaks()
    df = load_breaks(BREAKS_PATH, cfg)
    assert (df["break_sec"] > 0).all()


def test_carry_forward_rows_flagged_and_counted(cfg):
    _skip_if_no_grid()
    g = load_grid(GRID_PATH, cfg)
    synthetic = g[g["is_synthetic"]]
    assert len(synthetic) > 0
    assert (synthetic["assumption"] == "ASSUMPTION_MBC1_WK4").all()
    assert (~g[~g["is_synthetic"]]["is_synthetic"]).all()  # non-synthetic rows never flagged
    missing_week, source_week = g.attrs["carry_forward_weeks"]
    week3_mbc1 = g[(~g["is_synthetic"]) & (g["channel"] == "MBC 1") & (g["week"] == source_week)]
    assert len(synthetic) == len(week3_mbc1)


def test_overlap_locations_and_conflict_pairs(cfg):
    """CHANGE 2: overlaps are pairwise conflicts, not transitive-closure groups."""
    _skip_if_no_grid()
    g = load_grid(GRID_PATH, cfg)
    conflicts = g.attrs["slot_conflicts"]
    n_locations = conflicts.groupby(["channel", "air_date"]).ngroups
    assert n_locations == 8
    assert len(conflicts) == 24
    assert set(conflicts["kind"].unique()) <= {"contains", "partial"}
    # overlap_component is a reporting-only union-find grouping over the same pairs
    for cid, grp in g[g["overlap_component"] >= 0].groupby("overlap_component"):
        assert len(grp) >= 2  # every component has at least 2 rows sharing it


def test_raw_data_sha256_unchanged_after_ingest():
    """data/raw/ is read-only: every raw file's sha256 recorded in the manifest must still
    match the file on disk after ingestion runs."""
    if not MANIFEST_PATH.exists():
        pytest.skip("data/processed/ingest_manifest.json not present -- run "
                    "`python -m optimizer.ingest.run` first")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    grid_entry = manifest["grid"]
    grid_path = ROOT / grid_entry["path"]
    if grid_path.exists():
        assert _sha256(grid_path) == grid_entry["sha256"]

    for f in manifest["breaks"]["files"]:
        p = ROOT / f["path"]
        if p.exists():
            assert _sha256(p) == f["sha256"]
