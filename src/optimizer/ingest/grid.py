"""Ingest the October grid (buying-system export).

Hardened from ``reference_scripts/ingest_grid.py`` per ``docs/DATA_SPEC.md`` section A.
Never modifies ``data/raw/``; the caller is responsible for writing the returned frame to
``data/processed/grid.parquet``.

Parsing rules (see DATA_SPEC.A):
  - ``air_date`` = week (YYYYMMDD, a Sunday) + index of the non-underscore char in
    ``day_mask`` (0=Sun .. 6=Sat). Exactly one day per row (asserted).
  - HHMM -> minutes; ``end_min`` is exclusive (raw end minute + 1). If the slot wraps past
    midnight (end <= start once made exclusive) 24h is added to ``end_min``.
  - ``rate_usd`` = rate; ``rate_aed`` = round(rate x cfg.budget.aed_per_usd, 2), asserted to
    be integral to within 0.01 AED.
  - ``Program_name`` -> ``title_en`` / ``title_ar`` / ``season`` / ``is_rerun`` (contains
    "(R)"-style circled-R glyph) / ``is_live`` ("LIVE" in the name).
    The English/Arabic separator is a forward slash that may or may not have a preceding
    and/or following space (" / ", " /", "/ ", "/"). We split only at a slash that is
    directly followed (after optional whitespace) by an Arabic character -- slashes that are
    part of a season token ("S4/S5"), a running total ("24/7") or a year range ("2026/27")
    are left alone. This is a hardening fix over the reference script, which split on the
    literal substring " / " only and silently dropped the Arabic half of 44 rows (7 distinct
    programs) that use "  /" or "/" with no surrounding space.
  - Some ``Program_name`` cells are non-string when the raw programme title is purely
    numeric (observed in the eTAM file, not the grid, but guarded here defensively too);
    every string column is cast with ``.astype(str)`` before any ``.str`` accessor is used.
  - Overlaps (Gate 1 CHANGE 2): computed strictly **pairwise**, not as transitive-closure
    groups. Two rows on the same channel + air_date conflict iff their ``[start_min, end_min)``
    windows truly intersect (not just adjacent-in-sort-order neighbours -- the grid contains
    "compilation" container rows, e.g. MBC BOLLYWOOD "WEEKEND DRAMA COMPILATION", whose 6-hour
    window fully contains several separately-listed component-episode rows). Every conflicting
    *pair* is emitted as one row of ``data/processed/slot_conflicts.parquet`` (columns:
    slot_a, slot_b, channel, air_date, overlap_minutes, kind), so an umbrella row conflicts
    with each episode row it contains, but the contained episode rows do NOT conflict with
    each other (their own windows don't intersect -- they're back-to-back inside the umbrella).
    ``kind`` = ``'contains'`` when one window fully contains the other, else ``'partial'``.
    ``grid.parquet`` also gets a ``has_conflict`` bool (True iff the row appears in any pair)
    and an ``overlap_component`` id -- a cheap union-find connected-component grouping of rows
    that share *some* conflict, kept only for reporting/visualisation. It is explicitly NOT the
    exclusivity rule: two rows in the same component do not necessarily conflict with each
    other (see the BOLLYWOOD compilation case above) -- always consult ``slot_conflicts.parquet``
    / ``has_conflict`` for buying decisions, never ``overlap_component`` membership.
  - MBC 1 week-4 gap: if ``cfg.grid.mbc1_week4 == "carry_forward_wk3"``, synthetic MBC 1 rows
    are created for the missing week by copying the most recent week MBC 1 *does* have data
    for (found generically as "the week 7 days before the missing week", not hard-coded) onto
    the missing week, same weekday, with new ``slot_id`` values, ``is_synthetic=True`` and
    ``assumption='ASSUMPTION_MBC1_WK4'``. If ``"exclude"``, nothing is added.
  - Rows outside the flight window (``cfg.flight.start``..``cfg.flight.end``) are KEPT in the
    output with ``in_flight=False`` -- they are never dropped.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

# Arabic-script unicode ranges (Arabic + Arabic Supplement + Arabic Presentation Forms).
_ARABIC_RE = r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]"
# Split only at a "/" that is followed (after optional whitespace) by an Arabic character.
_TITLE_SPLIT_RE = re.compile(r"/(?=\s*" + _ARABIC_RE + ")")
_SEASON_RE = re.compile(r"\b(S\d+(?:/S\d+)*)\b")
_RERUN_RE = re.compile(r"\xae|®|®")  # circled-R glyph, however it round-trips through openpyxl


def hhmm_to_min(t: int) -> int:
    """HHMM integer (may be >= 2400 for the broadcast day) -> minutes from 00:00."""
    return (t // 100) * 60 + t % 100


def _split_title(raw: str) -> tuple[str, str | None]:
    """Split a raw ``Program_name`` (with any rerun glyph already stripped) into
    (title_en, title_ar). Returns title_ar=None when there is no Arabic half."""
    parts = _TITLE_SPLIT_RE.split(raw, maxsplit=1)
    title_en = parts[0].strip()
    title_ar = parts[1].strip() if len(parts) > 1 else None
    return title_en, title_ar


def find_slot_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Strictly pairwise slot-window conflicts within the same channel + air_date (Gate 1
    CHANGE 2). This is NOT a transitive-closure grouping: a pair conflicts iff their
    ``[start_min, end_min)`` windows actually intersect, computed independently for every
    pair -- so an umbrella "compilation" row conflicts with each episode row it contains, but
    two contained episode rows do NOT conflict with each other (their own windows are
    back-to-back inside the umbrella, not overlapping).

    Returns one row per conflicting pair: ``slot_a``/``slot_b`` (sorted so a pair appears
    once), ``channel``, ``air_date``, ``overlap_minutes`` (the intersection length), and
    ``kind`` = ``'contains'`` when one window fully contains the other, else ``'partial'``.
    """
    rows = []
    for (channel, air_date), grp in df.groupby(["channel", "air_date"], sort=False):
        items = list(grp[["slot_id", "start_min", "end_min"]].itertuples(index=False))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a.start_min < b.end_min and b.start_min < a.end_min:
                    a_contains_b = a.start_min <= b.start_min and b.end_min <= a.end_min
                    b_contains_a = b.start_min <= a.start_min and a.end_min <= b.end_min
                    kind = "contains" if (a_contains_b or b_contains_a) else "partial"
                    overlap_minutes = min(a.end_min, b.end_min) - max(a.start_min, b.start_min)
                    slot_a, slot_b = sorted([a.slot_id, b.slot_id])
                    rows.append((slot_a, slot_b, channel, air_date, int(overlap_minutes), kind))
    return pd.DataFrame(
        rows, columns=["slot_a", "slot_b", "channel", "air_date", "overlap_minutes", "kind"]
    )


def _assign_overlap_components(df: pd.DataFrame, conflicts: pd.DataFrame) -> pd.Series:
    """Cheap union-find connected-component id over the pairwise conflicts in ``conflicts``
    (see ``find_slot_conflicts``) -- for reporting/visualisation ONLY. This is explicitly NOT
    the exclusivity rule: two rows sharing a component do not necessarily conflict with each
    other (e.g. two episode rows inside the same BOLLYWOOD compilation umbrella share a
    component with the umbrella and with each other, but the two episodes themselves do not
    conflict -- always use ``find_slot_conflicts`` / ``has_conflict`` for buying decisions).
    Returns an int Series aligned to ``df.index``: a shared non-negative id per connected
    component, -1 for rows in no conflict at all."""
    parent: dict[str, str] = {sid: sid for sid in df["slot_id"]}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in conflicts.itertuples():
        union(r.slot_a, r.slot_b)

    roots = df["slot_id"].map(find)
    sizes = roots.value_counts()
    real_roots = sizes[sizes > 1].index
    comp_id = {root: cid for cid, root in enumerate(sorted(real_roots))}
    return pd.Series([comp_id.get(r, -1) for r in roots], index=df.index, dtype="int64")


def _find_carry_forward_week(g: pd.DataFrame, channel_col: str = "channel") -> tuple[int, int] | None:
    """Find (missing_week, source_week) for the MBC 1 week-4 gap generically: a week that
    every other channel has rows for, within the flight, that MBC 1 does not, whose
    "week minus 7 days" MBC 1 *does* have. Returns None if there is no such gap."""
    all_weeks = sorted(g["week"].unique())
    mbc1_weeks = set(g.loc[g[channel_col] == "MBC 1", "week"].unique())
    other_weeks = set(g.loc[g[channel_col] != "MBC 1", "week"].unique())
    missing = sorted(w for w in other_weeks if w not in mbc1_weeks)
    for missing_week in missing:
        missing_dt = pd.to_datetime(str(missing_week), format="%Y%m%d")
        source_dt = missing_dt - pd.Timedelta(days=7)
        source_week = int(source_dt.strftime("%Y%m%d"))
        if source_week in mbc1_weeks and missing_week in all_weeks:
            return int(missing_week), int(source_week)
    return None


def load_grid(path: str | Path, cfg: dict[str, Any]) -> pd.DataFrame:
    """Load and fully parse the October grid per DATA_SPEC section A.

    Parameters
    ----------
    path: path to the grid .xlsx (read-only; never modified).
    cfg: the loaded ``config/plan_config.yaml`` dict (see ``optimizer.config.load_config``).

    Returns
    -------
    DataFrame, one row per original grid row plus any synthetic MBC 1 week-4 carry-forward
    rows. See module docstring / DELIVER item 1 in the data-validator role for the full
    column contract.
    """
    aed_per_usd = float(cfg["budget"]["aed_per_usd"])
    flight_start = pd.Timestamp(cfg["flight"]["start"])
    flight_end = pd.Timestamp(cfg["flight"]["end"])
    mbc1_week4_mode = cfg["grid"]["mbc1_week4"]

    raw = pd.read_excel(path)
    g = raw.copy()

    # --- defensive string casts (pandas 3.0 string columns may already be 'str' dtype,
    # but cast explicitly so a stray numeric cell never breaks a .str accessor) ---
    for col in ["Station_name", "day_mask", "tags", "available", "Program_name"]:
        g[col] = g[col].astype(str)

    g["channel"] = g["Station_name"].str.upper().str.strip()
    g["station_code"] = g["station_code"]

    # --- day_mask -> day_idx / weekday / air_date ---
    masks = g["day_mask"].map(lambda m: [i for i, c in enumerate(m) if c != "_"])
    n_days = masks.map(len)
    assert (n_days == 1).all(), (
        f"day_mask must contain exactly one day; {int((n_days != 1).sum())} rows violate this"
    )
    g["day_idx"] = masks.str[0].astype("int64")
    g["weekday"] = g["day_idx"].map(lambda i: DAYS[i])
    week_start = pd.to_datetime(g["week"].astype(str), format="%Y%m%d")
    g["air_date"] = week_start + pd.to_timedelta(g["day_idx"], unit="D")

    # --- times ---
    g["start_hhmm"] = g["start_time"].astype("int64")
    g["end_hhmm"] = g["end_time"].astype("int64")
    g["start_min"] = g["start_hhmm"].map(hhmm_to_min)
    g["end_min"] = g["end_hhmm"].map(hhmm_to_min) + 1  # exclusive
    wraps = g["end_min"] <= g["start_min"]
    g.loc[wraps, "end_min"] = g.loc[wraps, "end_min"] + 24 * 60
    g["wraps_day"] = wraps
    g["slot_minutes"] = g["end_min"] - g["start_min"]
    g["start_dt"] = g["air_date"] + pd.to_timedelta(g["start_min"], unit="m")
    g["end_dt"] = g["air_date"] + pd.to_timedelta(g["end_min"], unit="m")

    # --- rates ---
    g["rate_usd"] = g["rate"].astype("float64")
    g["rate_aed"] = (g["rate_usd"] * aed_per_usd).round(2)
    off_by = (g["rate_aed"] - g["rate_aed"].round(0)).abs()
    assert off_by.max() < 0.01, (
        f"rate_aed not integral to 0.01 AED for {int((off_by >= 0.01).sum())} rows "
        f"(max off-by {off_by.max():.4f})"
    )

    # --- program name parsing ---
    pn_raw = g["Program_name"]
    g["is_rerun"] = pn_raw.str.contains(_RERUN_RE, regex=True, na=False)
    g["is_live"] = pn_raw.str.upper().str.contains("LIVE", na=False)
    pn_clean = pn_raw.str.replace(_RERUN_RE, "", regex=True).str.strip()
    split = pn_clean.map(_split_title)
    g["title_en"] = split.map(lambda t: t[0])
    g["title_ar"] = split.map(lambda t: t[1])
    has_arabic_in_en = g["title_en"].str.contains(_ARABIC_RE, regex=True, na=False)
    assert not has_arabic_in_en.any(), (
        f"{int(has_arabic_in_en.sum())} rows still have Arabic characters in title_en "
        "after splitting"
    )
    g["season"] = g["title_en"].str.extract(_SEASON_RE, expand=False)
    g["title_en"] = g["title_en"].str.replace(_SEASON_RE, "", regex=True).str.strip()

    # --- categorical / passthrough columns ---
    g["tier"] = g["tags"]
    g["program_name"] = pn_raw
    g["spot_length"] = g["spot_length"].astype("int64")
    g["daypart_code"] = g["daypart_code"].astype("int64")
    for col in ["available", "free_time", "num_ratings", "demo_code", "audience", "universe"]:
        g[col] = g[col]  # kept as-is (placeholder columns from the buying-system template)

    # --- slot_id ---
    g["slot_id"] = (
        g["channel"] + "|" + g["air_date"].dt.strftime("%Y-%m-%d") + "|"
        + g["start_hhmm"].astype(str).str.zfill(4)
    )
    assert g["slot_id"].is_unique, "slot_id must be unique before synthetic rows are added"

    g["in_flight"] = (g["air_date"] >= flight_start) & (g["air_date"] <= flight_end)
    g["is_synthetic"] = False
    g["assumption"] = ""

    n_pre_flight = int((~g["in_flight"]).sum())

    # --- MBC 1 week-4 carry-forward ---
    carry_forward_rows = 0
    carry_forward_weeks = None
    if mbc1_week4_mode == "carry_forward_wk3":
        found = _find_carry_forward_week(g)
        if found is not None:
            carry_forward_weeks = found
            missing_week, source_week = found
            src = g[(g["channel"] == "MBC 1") & (g["week"] == source_week)].copy()
            carry_forward_rows = len(src)
            src["week"] = missing_week
            src["air_date"] = src["air_date"] + pd.Timedelta(days=7)
            src["start_dt"] = src["air_date"] + pd.to_timedelta(src["start_min"], unit="m")
            src["end_dt"] = src["air_date"] + pd.to_timedelta(src["end_min"], unit="m")
            src["slot_id"] = (
                src["channel"] + "|" + src["air_date"].dt.strftime("%Y-%m-%d") + "|"
                + src["start_hhmm"].astype(str).str.zfill(4)
            )
            src["in_flight"] = (src["air_date"] >= flight_start) & (src["air_date"] <= flight_end)
            src["is_synthetic"] = True
            src["assumption"] = "ASSUMPTION_MBC1_WK4"
            g = pd.concat([g, src], ignore_index=True)
    elif mbc1_week4_mode == "exclude":
        pass
    else:
        raise ValueError(f"Unknown grid.mbc1_week4 mode: {mbc1_week4_mode!r}")

    assert g["slot_id"].is_unique, "slot_id must remain unique after synthetic rows are added"

    # --- conflicts (recomputed after any synthetic rows, since they land on new dates) ---
    # Strictly pairwise (Gate 1 CHANGE 2) -- see find_slot_conflicts / _assign_overlap_components
    # docstrings. slot_conflicts is the exclusivity table; overlap_component/has_conflict on
    # grid.parquet are derived from it for convenience.
    slot_conflicts = find_slot_conflicts(g)
    g["overlap_component"] = _assign_overlap_components(g, slot_conflicts)
    conflicted_ids = set(slot_conflicts["slot_a"]).union(slot_conflicts["slot_b"])
    g["has_conflict"] = g["slot_id"].isin(conflicted_ids)

    g.attrs["n_pre_flight_rows"] = n_pre_flight
    g.attrs["n_carry_forward_rows"] = carry_forward_rows
    g.attrs["carry_forward_weeks"] = carry_forward_weeks
    g.attrs["slot_conflicts"] = slot_conflicts

    return g.reset_index(drop=True)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from optimizer.config import load_config

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/grid/october_grid.xlsx"
    cfg = load_config()
    g = load_grid(path, cfg)
    print("rows", len(g))
    print(pd.crosstab(g["channel"], g["week"]))
    conflicts = g.attrs["slot_conflicts"]
    print("conflict pairs", len(conflicts), "distinct components",
          int(g.loc[g["overlap_component"] >= 0, "overlap_component"].nunique()))
    print("reruns", int(g.is_rerun.sum()), "live", int(g.is_live.sum()))
    print("total cost USD", round(g.rate_usd.sum(), 2))
    print("pre-flight rows kept (in_flight=False)", g.attrs["n_pre_flight_rows"])
    print("carry-forward rows", g.attrs["n_carry_forward_rows"])
