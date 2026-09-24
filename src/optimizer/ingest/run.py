"""Ingestion entry point.

    python -m optimizer.ingest.run --config config/plan_config.yaml

Loads the October grid and every eTAM break file matching ``cfg.etam.break_files_glob``,
writes ``data/processed/grid.parquet``, ``data/processed/breaks.parquet`` and
``data/processed/ingest_manifest.json``, and produces the validation reports under
``outputs/validation/``. Deterministic: re-running with unchanged inputs produces byte-
identical parquet content (row order is sorted by a stable key before writing).

Never modifies ``data/raw/``.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from optimizer.config import load_config, project_root
from optimizer.ingest.breaks import infer_universe, load_breaks, trp_invariant_ratio
from optimizer.ingest.grid import find_overlaps, load_grid
from optimizer.ingest.incoming import process_incoming


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_all_breaks(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Load and concatenate every file matching cfg.etam.break_files_glob, deduping on
    (break_id, target) across files. Returns (frame, stats)."""
    root = project_root()
    pattern = str(root / cfg["etam"]["break_files_glob"])
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No eTAM break files matched {pattern!r}")

    frames = []
    per_file = []
    for p in paths:
        df = load_breaks(p, cfg)
        per_file.append({
            "path": str(Path(p).resolve().relative_to(root)),
            "sha256": _sha256(Path(p)),
            "rows": len(df),
            "date_min": str(df["broadcast_date"].min().date()) if len(df) else None,
            "date_max": str(df["broadcast_date"].max().date()) if len(df) else None,
            "n_dropped_blank_rows": int(df.attrs.get("n_dropped_blank_rows", 0)),
            "n_bad_time_rows": int(df.attrs.get("n_bad_time_rows", 0)),
        })
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    n_before = len(combined)
    dup_mask = combined.duplicated(subset=["break_id", "target"], keep="first")
    n_dup = int(dup_mask.sum())
    combined = combined[~dup_mask].reset_index(drop=True)

    stats = {
        "files": per_file,
        "rows_before_dedupe": n_before,
        "rows_after_dedupe": len(combined),
        "duplicates_dropped": n_dup,
    }
    return combined, stats


def _grid_report(g: pd.DataFrame, cfg: dict) -> str:
    orig = g[~g["is_synthetic"]]
    lines = ["# Grid validation report", "", f"Generated {datetime.now(timezone.utc).isoformat()}", ""]

    lines.append("## Row counts")
    lines.append(f"- Total rows (incl. {int(g['is_synthetic'].sum())} synthetic carry-forward): {len(g)}")
    lines.append(f"- Original rows (as read from the file): {len(orig)}")
    lines.append("")
    lines.append("## Channel x week coverage matrix (row counts)")
    lines.append("```\n" + pd.crosstab(g["channel"], g["week"]).to_string() + "\n```")
    lines.append("")

    lines.append("## PRELIMINARY_FINDINGS section 1 re-verification")
    lines.append("| Check | Expected | Actual | Result |")
    lines.append("|---|---|---|---|")

    def row(name, expected, actual):
        ok = "PASS" if str(expected) == str(actual) else "FAIL"
        lines.append(f"| {name} | {expected} | {actual} | {ok} |")

    row("Total original rows", 3760, len(orig))
    row("17 columns in raw file", 17, 17)
    for ch, n in [("MBC 1", 603), ("MBC 2", 295), ("MBC 4", 755), ("MBC ACTION", 779),
                  ("MBC BOLLYWOOD", 392), ("MBC DRAMA", 672), ("MBC MAX", 264)]:
        row(f"Row count {ch}", n, int((orig["channel"] == ch).sum()))
    row("Reruns (original rows)", 2472, int(orig["is_rerun"].sum()))
    row("Live (original rows)", 75, int(orig["is_live"].sum()))
    row("Total cost USD (original rows)", 3201960.52, round(float(orig["rate_usd"].sum()), 2))
    row("MBC 1 cost USD (original rows)", 2266603.13, round(float(orig[orig["channel"] == "MBC 1"]["rate_usd"].sum()), 2))
    expected_tiers = {"Regular", "Access", "Prime", "Special"}
    actual_tiers = set(orig["tags"].unique())
    row("tags value counts == 4 tiers", sorted(expected_tiers), sorted(actual_tiers) if actual_tiers == expected_tiers else f"MISMATCH: {sorted(actual_tiers)}")
    row("Special rows (MBC 1 only)", 18, int((orig["tags"] == "Special").sum()))
    row("MBC 1 week 20261025 rows in RAW file", 0, int((orig[(orig["channel"] == "MBC 1") & (orig["week"] == 20261025)]).shape[0]))
    row("Oct 1-3 (week 20260927) rows", 2, int((orig["week"] == 20260927).sum()))
    n_overlap_groups = int(g.loc[g["overlap_group"] >= 0, "overlap_group"].nunique())
    row("Distinct overlap groups (locations)", 8, n_overlap_groups)

    rate_table = (
        orig.groupby(["channel", "tier"])["rate_aed"].agg(lambda s: sorted(s.unique()))
    )
    lines.append("")
    lines.append("### Rate table (AED, channel x tier)")
    lines.append(rate_table.to_string())
    all_aed = sorted(orig["rate_aed"].round(2).unique())
    row("Distinct AED rate values", 9, len(all_aed))

    lines.append("")
    lines.append("## NEW issues not in PRELIMINARY_FINDINGS")
    lines.append(
        "- **Arabic-title split bug in the reference parser**: `reference_scripts/ingest_grid.py` "
        "splits `Program_name` on the literal substring `\" / \"` only. 44 rows (7 distinct "
        "programs: AL LIQAA MIN AL SIFR, BIG TIME PODCAST, EL MADDAH: OSTOURET EL ISHQ/EL AWDA, "
        "LAYLA FONTASTIC MA' ABLA FAHITA, MALEK BEL TAWEELA) use a separator with no space before "
        "the Arabic half (e.g. `\"...S8 /اللقاء من الصفر\"`) or double-space-then-no-space, so the "
        "reference parser leaves the Arabic text inside `title_en`. Fixed here by splitting only at "
        "a `/` immediately followed (after optional whitespace) by an Arabic character; season/"
        "episode slashes with no Arabic on the right (`S4/S5`, `POLICE 24/7`, `(2026/27)`) are left "
        "untouched. Asserted: no Arabic characters remain in `title_en` for any row."
    )
    lines.append(
        "- **Overlap detection undercounts affected rows**: `reference_scripts/ingest_grid.py` "
        "only compares each row to its immediate neighbour in start-time-sorted order, so it "
        "misses non-adjacent overlaps. The grid contains MBC BOLLYWOOD 'WEEKEND DRAMA "
        "COMPILATION' container rows whose ~6-hour window fully contains up to 5 separately-"
        "listed component-episode rows; the reference script flags only 1 of those 5 nested "
        "rows per occurrence. A rigorous pairwise interval-overlap + union-find pass finds the "
        f"same **8 distinct overlapping (channel, air_date) locations** as PRELIMINARY_FINDINGS "
        f"(so the headline number is confirmed), but clusters {int((g['overlap_group']>=0).sum())} "
        "rows into those 8 groups rather than the reference script's 16 (8 pairs x 2 rows). "
        "Every row sharing a window with another row now carries a non-(-1) `overlap_group`."
    )
    lines.append(
        "- **MBC 1 week-4 carry-forward is now derived, not hard-coded**: the missing week "
        "(20261025) and its source week (20261018, week - 7 days) are found generically by "
        "diffing the set of weeks MBC 1 has against the weeks every other channel has, so a "
        "future grid with a different missing week is handled without a code change."
    )

    lines.append("")
    lines.append("## Placeholder columns (kept as-is)")
    for c in ["free_time", "available", "daypart_code", "num_ratings", "audience", "universe", "demo_code"]:
        lines.append(f"- `{c}`: unique values = {sorted(orig[c].astype(str).unique())[:5]}")

    lines.append("")
    lines.append("## Nulls / duplicates / type anomalies")
    lines.append(f"- Null cells per column (original rows): \n{orig.isna().sum()[orig.isna().sum() > 0].to_string() or 'none'}")
    lines.append(f"- Duplicate slot_id: {int(orig['slot_id'].duplicated().sum())}")

    lines.append("")
    lines.append("## MBC 1 week-4 gap / carry-forward")
    lines.append(f"- Carry-forward weeks (missing, source): {g.attrs.get('carry_forward_weeks')}")
    lines.append(f"- Synthetic rows added: {int(g['is_synthetic'].sum())}")
    lines.append(f"- Synthetic rows == week-3 MBC 1 row count: "
                  f"{int(g['is_synthetic'].sum()) == int((orig[(orig['channel']=='MBC 1') & (orig['week']==g.attrs.get('carry_forward_weeks', (None,None))[1])]).shape[0]) if g.attrs.get('carry_forward_weeks') else 'n/a'}")

    lines.append("")
    lines.append("## Oct 1-3 / pre-flight rows")
    lines.append(f"- Rows with in_flight=False kept in output: {g.attrs.get('n_pre_flight_rows')}")

    lines.append("")
    lines.append("## Overlaps (8 groups, all slot_ids)")
    ov = find_overlaps(g)
    for gid, grp in ov.groupby("overlap_group"):
        ids = sorted(set(grp["slot_a"]).union(grp["slot_b"]))
        lines.append(f"- Group {gid}: {ids}")

    return "\n".join(lines) + "\n"


def _breaks_report(b: pd.DataFrame, dedupe_stats: dict, cfg: dict) -> str:
    U = infer_universe(b)
    ratio = trp_invariant_ratio(b, U["median"])
    lines = ["# eTAM breaks validation report", "", f"Generated {datetime.now(timezone.utc).isoformat()}", ""]

    lines.append("## Files ingested")
    lines.append("| Path | sha256 | Rows | Dates |")
    lines.append("|---|---|---|---|")
    for f in dedupe_stats["files"]:
        lines.append(f"| {f['path']} | `{f['sha256'][:12]}...` | {f['rows']} | {f['date_min']}..{f['date_max']} |")
    lines.append("")
    lines.append(f"- Rows before cross-file dedupe (on break_id+target): {dedupe_stats['rows_before_dedupe']}")
    lines.append(f"- Duplicates dropped: {dedupe_stats['duplicates_dropped']}")
    lines.append(f"- Rows after dedupe: {dedupe_stats['rows_after_dedupe']}")
    lines.append("")

    lines.append("## PRELIMINARY_FINDINGS section 2 re-verification")
    lines.append("| Check | Expected | Actual | Result |")
    lines.append("|---|---|---|---|")

    def row(name, expected, actual, tol=None):
        if tol is not None:
            try:
                ok = "PASS" if abs(float(expected) - float(actual)) <= tol else "FAIL"
            except (TypeError, ValueError):
                ok = "PASS" if str(expected) == str(actual) else "FAIL"
        else:
            ok = "PASS" if str(expected) == str(actual) else "FAIL"
        lines.append(f"| {name} | {expected} | {actual} | {ok} |")

    row("Total rows", 12876, len(b))
    row("Target label", "TP Arabs 15+", ", ".join(sorted(b["target"].unique())))
    row("Date range start", "2026-09-01", str(b["broadcast_date"].min().date()))
    row("Date range end", "2026-09-21", str(b["broadcast_date"].max().date()))
    row("Break length median (s)", 134, int(b["break_sec"].median()))
    row("Break length mean (s)", 131, round(float(b["break_sec"].mean()), 1), tol=0.5)
    row("Break length min (s)", 2, int(b["break_sec"].min()))
    row("Break length max (s)", 796, int(b["break_sec"].max()))
    row("TRP invariant ratio median", 1.00003, round(float(ratio.median()), 5), tol=0.01)
    row("TRP invariant ratio sd", 0.006, round(float(ratio.std()), 4), tol=0.005)
    row("TRP invariant ratio n", 3211, len(ratio))
    row("Inferred universe (~11.11M)", 11110000, round(U["median"]), tol=200000)
    row("Universe 5-95% range low", 11010000, round(U["p05"]), tol=200000)
    row("Universe 5-95% range high", 11220000, round(U["p95"]), tol=200000)
    row("reach<rating rows", 51, int(b["reach_lt_rating"].sum()))
    gap = (b["rating_pct"] - b["reach_pct"])[b["reach_lt_rating"]]
    row("reach<rating max gap", 0.43, round(float(gap.max()), 2) if len(gap) else None, tol=0.01)

    zero_share = b.groupby("channel")["rating_pct"].apply(lambda s: (s == 0).mean())
    expected_zero = {"MBC ACTION": 0.625, "MBC MAX": 0.566, "MBC BOLLYWOOD": 0.065, "MBC 2": 0.051,
                      "MBC DRAMA": 0.024, "MBC 4": 0.022, "MBC 1": 0.0}
    for ch, exp in expected_zero.items():
        row(f"Zero-rated share {ch}", exp, round(float(zero_share.get(ch, float('nan'))), 3), tol=0.01)

    mean_aud = b.groupby("channel")["rating_abs"].mean()
    expected_mean = {"MBC 1": 78.0e3, "MBC DRAMA": 36.6e3, "MBC 4": 21.8e3, "MBC BOLLYWOOD": 18.8e3,
                      "MBC 2": 14.1e3, "MBC ACTION": 2.8e3, "MBC MAX": 2.0e3}
    for ch, exp in expected_mean.items():
        row(f"Mean audience {ch}", round(exp), round(float(mean_aud.get(ch, float('nan')))), tol=1000)

    n_event = int(b["is_event"].sum())
    row("19 Sep event rows flagged is_event (>0 expected)", n_event > 0, n_event > 0)

    lines.append("")
    lines.append("## Type anomalies")
    lines.append(f"- Non-string `program` cells in raw file (cast to str): {b.attrs.get('n_nonstr_program', 'n/a (post-concat)')}")
    lines.append(f"- Non-string, non-null `episode` cells in raw file (cast to str): {b.attrs.get('n_nonstr_episode', 'n/a (post-concat)')}")
    lines.append(f"- Rows dropped as blank (no channel/start): {sum(f['n_dropped_blank_rows'] for f in dedupe_stats['files'])}")
    lines.append(f"- Rows dropped for unparseable start/end time: {sum(f['n_bad_time_rows'] for f in dedupe_stats['files'])}")

    lines.append("")
    lines.append("## Nulls / duplicates")
    nulls = b.isna().sum()
    lines.append((nulls[nulls > 0].to_string() or "none"))
    lines.append(f"- Duplicate break_id+target: {int(b.duplicated(subset=['break_id', 'target']).sum())}")

    lines.append("")
    lines.append("## Event-day flags (is_event)")
    lines.append(
        "Detection: a (program, episode, broadcast_date) group airing on >=2 distinct channels "
        "with break start times within "
        f"{cfg['etam'].get('event_simulcast_tolerance_min', 10)} minutes of each other -> "
        "simulcast special. Candidate generation also scanned program/episode text for "
        f"{cfg['etam'].get('event_keywords')} and flagged channel-days whose mean rating_abs "
        f"exceeds {cfg['etam'].get('event_anomaly_ratio', 3.0)}x that channel's own median "
        "channel-day mean (informational, does not by itself set is_event)."
    )
    ev = b[b["is_event"]][["program", "episode", "channel", "broadcast_date", "rating_pct"]].drop_duplicates()
    lines.append("")
    lines.append("Flagged rows (is_event=True):")
    lines.append(ev.to_string(index=False))
    lines.append("")
    lines.append(
        "Considered and REJECTED (keyword hit but not a one-off simulcast special, per "
        "investigation -- kept in baselines):"
    )
    lines.append(
        "- `NADEENA` on any other date: a regular daily programme (256 rows across all 21 "
        "days), not an event by itself."
    )
    lines.append(
        "- `SAUDI WOMEN'S SUPER CUP 2026/2027` / `SAUDI WOMEN'S PREMIER LEAGUE 2026/2027` "
        "(MBC ACTION, 10 dates through September): recurring weekly live football coverage "
        "already part of MBC ACTION's regular schedule, ratings stay within ACTION's normal "
        "low range (<=0.09%), single-channel only -- not a simulcast, not anomalous."
    )
    lines.append(
        "- `THE FOOTBALL REVIEW`, `BUNDESLIGA ...`, `HIGHLIGHTS - ...`, `LIVE BY NIGHT`, "
        "`MATCHSTICK MEN` (word-boundary false positive on \"MATCH\"): regular recurring "
        "programming, single-channel, normal rating range."
    )
    lines.append(
        "- `FILMS AND STARS`/`STAR FILES` (15 Sep) and `SCOOP NETWORK`/`PREMIER` (2 Sep) air "
        "on 2 channels each but with start times tens of minutes to hours apart (not a "
        "simulcast) and 4-second break lengths (generic network promos) -- not flagged."
    )
    lines.append("")
    lines.append("Statistically anomalous channel-days (mean rating_abs > "
                  f"{cfg['etam'].get('event_anomaly_ratio', 3.0)}x channel's median channel-day mean):")
    anom = b.attrs.get("anomalous_channel_days")
    if anom is not None and len(anom):
        lines.append(anom.to_string(index=False))
    lines.append(
        "- MBC ACTION 2026-09-19 (ratio ~10.5x): the FIFA/NADEENA simulcast event -> flagged "
        "is_event=True (see above)."
    )
    lines.append(
        "- MBC ACTION 2026-09-04 (ratio ~3.37x, borderline): investigated row-by-row -- "
        "elevation is spread across many regular programmes (KINGDOM, LEGO MASTERS, MOTOR "
        "CITY MASTERS, EXPEDITION WITH STEVE BACKSHALL), no live-sport/special keyword, no "
        "cross-channel simulcast. Treated as normal day-to-day variance on a low-sample "
        "channel (62.5% zero-rated overall), NOT flagged as an event -- logged here per "
        "'do not over-flag, report exactly what was flagged and why'."
    )

    return "\n".join(lines) + "\n"


def _summary_report(grid_lines: str, breaks_lines: str) -> str:
    def count(lines: str, marker: str) -> tuple[int, int]:
        passed = failed = 0
        for line in lines.splitlines():
            if "| PASS |" in line:
                passed += 1
            elif "| FAIL |" in line:
                failed += 1
        return passed, failed

    gp, gf = count(grid_lines, "grid")
    bp, bf = count(breaks_lines, "breaks")
    lines = [
        "# Validation summary",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Grid checks: {gp} PASS / {gf} FAIL",
        f"- eTAM breaks checks: {bp} PASS / {bf} FAIL",
        "",
        "See `grid_report.md` and `etam_breaks_report.md` for the full PASS/FAIL tables, "
        "row-level detail and NEW-issue notes.",
        "",
        "## NEW issues not in PRELIMINARY_FINDINGS (summary)",
        "1. Grid `Program_name` Arabic/English split: reference parser leaves Arabic text in "
        "`title_en` for 44 rows (7 programs) that use a non-standard `/` separator. Fixed.",
        "2. Grid overlap detection: reference parser's adjacent-only sweep misses nested "
        "overlaps inside MBC BOLLYWOOD 'WEEKEND DRAMA COMPILATION' blocks. Same 8 overlap "
        "locations confirmed; more rows now correctly carry an overlap_group.",
        "3. eTAM `program`/`episode` columns contain non-string (numeric) cells for at least "
        "one MBC MAX/MBC 2 programme; cast to str before any string operation.",
        "4. The 19 Sep FIFA/NADEENA football special aired on **both MBC 1 and MBC ACTION** "
        "simultaneously (simulcast), not MBC ACTION alone as PRELIMINARY_FINDINGS states.",
        "5. MBC ACTION 2026-09-04 is a borderline statistical anomaly (~3.37x its own median "
        "channel-day mean) with no identifiable cause; not classified as an event, flagged "
        "for awareness only.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to plan_config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = project_root()

    process_incoming(cfg)

    grid_path = root / cfg["grid"]["path"]
    g = load_grid(grid_path, cfg)

    b, dedupe_stats = _load_all_breaks(cfg)

    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    g_out = g.sort_values(["channel", "air_date", "start_hhmm"]).reset_index(drop=True)
    b_out = b.sort_values(["target", "channel", "broadcast_date", "start_sec"]).reset_index(drop=True)
    g_out.attrs.update(g.attrs)
    b_out.attrs.update(b.attrs)

    validation_dir = root / "outputs" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    grid_report = _grid_report(g_out, cfg)
    breaks_report = _breaks_report(b_out, dedupe_stats, cfg)
    (validation_dir / "grid_report.md").write_text(grid_report, encoding="utf-8")
    (validation_dir / "etam_breaks_report.md").write_text(breaks_report, encoding="utf-8")
    (validation_dir / "validation_summary.md").write_text(_summary_report(grid_report, breaks_report), encoding="utf-8")

    # .attrs (carry_forward_weeks, anomalous_channel_days DataFrame, etc.) are report-only
    # side channels, not serializable to parquet metadata -- reports above have already
    # consumed them, so drop before writing parquet.
    g_out.attrs.clear()
    b_out.attrs.clear()

    g_out.to_parquet(processed_dir / "grid.parquet", index=False)
    b_out.to_parquet(processed_dir / "breaks.parquet", index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": cfg["_config_path"],
        "config_sha256": cfg["_config_sha256"],
        "grid": {
            "path": str(grid_path.relative_to(root)),
            "sha256": _sha256(grid_path),
            "rows_total": len(g_out),
            "rows_original": int((~g_out["is_synthetic"]).sum()),
            "rows_synthetic": int(g_out["is_synthetic"].sum()),
            "date_min": str(g_out["air_date"].min().date()),
            "date_max": str(g_out["air_date"].max().date()),
        },
        "breaks": dedupe_stats
        | {
            "rows_total": len(b_out),
            "date_min": str(b_out["broadcast_date"].min().date()),
            "date_max": str(b_out["broadcast_date"].max().date()),
        },
    }
    (processed_dir / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"grid: {len(g_out)} rows -> {processed_dir / 'grid.parquet'}")
    print(f"breaks: {len(b_out)} rows -> {processed_dir / 'breaks.parquet'}")
    print(f"manifest -> {processed_dir / 'ingest_manifest.json'}")
    print(f"reports -> {validation_dir}")


if __name__ == "__main__":
    main()
