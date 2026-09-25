"""Programme matching entry point (Phase 3 -> Gate 2).

    python -m optimizer.match.run --config config/plan_config.yaml

Reads data/processed/grid.parquet + breaks.parquet, writes
  * data/processed/program_map.csv                        (one row per in-flight channel+title_en)
  * outputs/validation/program_matches_review.xlsx        (review / all_matches / unmatched_etam)
  * outputs/validation/program_matches_review.csv         (review sheet only)
  * outputs/validation/program_matching_report.md
Paths come from cfg.match.paths. Deterministic.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openpyxl.utils import get_column_letter

from optimizer.config import load_config, project_root
from optimizer.match.matching import (CHANNEL_ORDER, build_program_map, match_params,
                                      review_table)


def _write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for i, col in enumerate(df.columns, start=1):
                vals = [str(col)] + [str(v) for v in df[col].head(300).tolist()]
                width = min(80, max(8, max(len(v) for v in vals) + 2))
                ws.column_dimensions[get_column_letter(i)].width = width


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in r) + " |")
    return "\n".join(lines)


def write_report(pm: pd.DataFrame, rv: pd.DataFrame, ue: pd.DataFrame, cfg: dict, path: Path) -> None:
    p = match_params(cfg)
    thr = p["review_threshold"]
    matched = pm[pm["etam_title"] != "NONE"]
    hi = matched[matched["confidence"] >= thr]
    tot_usd = pm["usd_at_stake"].sum()
    by_type = pm.groupby("match_type").agg(titles=("grid_title", "size"), usd=("usd_at_stake", "sum"))
    by_type["usd"] = by_type["usd"].round(0).astype(int)
    by_ch = pd.crosstab(pm["channel"], pm["match_type"]).reindex(
        [c for c in CHANNEL_ORDER if c in set(pm["channel"])])
    top = pm.sort_values("usd_at_stake", ascending=False).head(20)
    top_t = pd.DataFrame({"channel": top["channel"], "grid_title": top["grid_title"],
                          "usd": top["usd_at_stake"].round(0).astype(int),
                          "etam_title": top["etam_title"] + top["etam_channel"].where(
                              top["etam_channel"].ne(top["channel"]) & top["etam_channel"].ne(""), "").map(
                              lambda c: f" [{c}]" if c else ""),
                          "match_type": top["match_type"], "confidence": top["confidence"]})
    tr = matched[(matched["match_type"] == "fuzzy") & (matched["label_class"] == "title")
                 & ~matched["flags"].fillna("").str.contains("sibling")]
    sem = matched[matched["match_type"] == "time-slot inferred"]
    sib = matched[matched["flags"].fillna("").str.contains("sibling")]
    cross = matched[matched["etam_channel"] != matched["channel"]]
    comp = matched[matched["label_class"] == "compilation"]
    L = []
    L.append("# Programme matching report (Phase 3, Gate 2)\n")
    L.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
             f"`python -m optimizer.match.run`; config sha256 `{cfg.get('_config_sha256', '')[:12]}`.\n")
    L.append(f"- Distinct in-flight grid programmes (channel + title_en): **{len(pm)}**, "
             f"USD {tot_usd:,.0f} of in-flight grid value.")
    L.append(f"- Mapped to an eTAM title (any confidence): **{len(matched)}**; with confidence >= {thr}: "
             f"**{len(hi)}** titles covering USD {hi['usd_at_stake'].sum():,.0f} "
             f"({100 * hi['usd_at_stake'].sum() / tot_usd:.1f}% of grid value).")
    L.append(f"- Rows in the Gate 2 review sheet: **{len(rv)}** (confidence < {thr} or match_type in "
             f"time-slot inferred / generic-slot / new program), USD {rv['usd_at_stake'].sum():,.0f}.")
    L.append(f"- eTAM titles with >= {p['min_etam_breaks']} Sep breaks left unmatched: {len(ue)} "
             f"({int((ue['likely_role'] == 'movie in a generic-slot pool').sum()) if len(ue) else 0} are movies "
             f"inside generic movie-slot windows).\n")
    L.append("## Counts by match_type\n")
    L.append(_md_table(by_type.reset_index()) + "\n")
    L.append("## Counts by channel x match_type\n")
    L.append(_md_table(by_ch.reset_index()) + "\n")
    L.append("## Top-20 programmes by USD at stake\n")
    L.append(_md_table(top_t) + "\n")
    L.append("## Normalisation rules applied\n")
    L.append("1. NFKC, upper case; for 'EN S1 / AR ®' names keep the English part.\n"
             "2. Remove `®`, trailing `LIVE` (also glued, e.g. `DAY 1LIVE`), `14:00 GMT` labels, year tags "
             "`(2026)`, `(2026/27)`, `2026/2027`, `DAY n`, season tags `S1`, `S4/S5`, `SEASON n`, the word `REPEAT`.\n"
             "3. Parenthetical non-year text is kept aside as a qualifier (e.g. `KHALEEJI 27`) and caps confidence.\n"
             "4. `&` -> AND; apostrophes deleted (MALA'EB -> MALAEB); other punctuation -> space.\n"
             "5. Standalone articles AL / EL / AR dropped; glued articles tolerated at word level (ELBALAD ~ BALAD).\n"
             "6. eTAM titles also lose a trailing channel label (`AL AKHBAR - MBC 1` -> `AKHBAR`).\n"
             "7. Word similarity: rapidfuzz ratio, or 0.95 when first letter and consonant skeleton agree "
             "(KH/SH/TH/DH/GH/PH/Q folded, vowels/W/Y removed, double letters collapsed, trailing -H after a vowel dropped).\n"
             "8. Compilation labels (`TURKISH DRAMA - COMPILATION<title>`, `WEEKEND DRAMA COMPILATION: <title>`) "
             "are matched through the component; the eTAM `<title> MARATHON` block is preferred when it has >= "
             f"{p['min_etam_breaks']} breaks.\n"
             "9. Generic movie-strand labels (config `match.generic_patterns`) -> generic-slot, with a slot pool rule.\n"
             "10. Arabic titles (MBC 1 / DRAMA) are reduced to a consonant skeleton and compared with eTAM titles; "
             "such semantic matches are accepted only with time-slot evidence (match_type = time-slot inferred).\n")
    L.append("## Transliteration pairs accepted (same channel, fuzzy)\n")
    if len(tr):
        L.append(_md_table(pd.DataFrame({"channel": tr["channel"], "grid": tr["grid_title"],
                                         "eTAM": tr["etam_title"], "title_score": tr["title_score"],
                                         "confidence": tr["confidence"]})) + "\n")
    L.append("## Judgement calls for the human (all below the review threshold)\n")
    if len(sem):
        L.append("**Semantic / time-slot inferred** (Arabic-title transliteration or alias + time evidence):\n")
        L.append(_md_table(pd.DataFrame({"channel": sem["channel"], "grid": sem["grid_title"],
                                         "eTAM": sem["etam_title"], "confidence": sem["confidence"],
                                         "time_overlap": sem["time_overlap"]})) + "\n")
    if len(cross):
        L.append("**Cross-channel history** (audience level on the other channel is not transferable):\n")
        L.append(_md_table(pd.DataFrame({"channel": cross["channel"], "grid": cross["grid_title"],
                                         "eTAM": cross["etam_title"], "history_on": cross["etam_channel"],
                                         "confidence": cross["confidence"]})) + "\n")
    if len(sib):
        L.append("**Sibling / spin-off proxies** (different season subtitle or format; proxy only):\n")
        L.append(_md_table(pd.DataFrame({"channel": sib["channel"], "grid": sib["grid_title"],
                                         "eTAM": sib["etam_title"], "confidence": sib["confidence"]})) + "\n")
    if len(comp):
        L.append("**Compilations** mapped to component / MARATHON block:\n")
        L.append(_md_table(pd.DataFrame({"channel": comp["channel"], "grid": comp["grid_title"],
                                         "eTAM": comp["etam_title"], "confidence": comp["confidence"]})) + "\n")
    gen = pm[pm["match_type"] == "generic-slot"]
    L.append(f"**Generic slots**: {len(gen)} labels (USD {gen['usd_at_stake'].sum():,.0f}) on "
             f"{', '.join(sorted(set(gen['channel'])))} — etam_title = NONE, forecaster uses `slot_pool_rule`.\n")
    L.append("Matching never reads `is_event` / overlap columns; `n_etam_breaks` excludes event breaks.\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    root = project_root()
    paths = (cfg.get("match") or {}).get("paths") or {}
    grid = pd.read_parquet(root / "data" / "processed" / "grid.parquet")
    breaks = pd.read_parquet(root / "data" / "processed" / "breaks.parquet")
    pm, ue = build_program_map(grid, breaks, cfg)
    rv = review_table(pm, cfg)

    pm_path = root / paths.get("program_map", "data/processed/program_map.csv")
    pm_path.parent.mkdir(parents=True, exist_ok=True)
    pm.to_csv(pm_path, index=False, encoding="utf-8")
    rv_csv = root / paths.get("review_csv", "outputs/validation/program_matches_review.csv")
    rv_csv.parent.mkdir(parents=True, exist_ok=True)
    rv.to_csv(rv_csv, index=False, encoding="utf-8")
    _write_xlsx(root / paths.get("review_xlsx", "outputs/validation/program_matches_review.xlsx"),
                {"review": rv, "all_matches": pm, "unmatched_etam": ue})
    write_report(pm, rv, ue, cfg, root / paths.get("report_md", "outputs/validation/program_matching_report.md"))

    thr = match_params(cfg)["review_threshold"]
    hi = pm[(pm["etam_title"] != "NONE") & (pm["confidence"] >= thr)]
    print(f"program_map: {len(pm)} grid programmes -> {pm_path}")
    print(pm["match_type"].value_counts().to_string())
    print(f"confidence >= {thr}: {len(hi)} titles, USD {hi['usd_at_stake'].sum():,.0f} of "
          f"{pm['usd_at_stake'].sum():,.0f}; review rows: {len(rv)}; unmatched eTAM: {len(ue)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
