"""October per-slot audience forecast (Phase 4).

    python -m optimizer.forecast.run --config config/plan_config.yaml [--final]

Reads data/processed/{grid.parquet, breaks.parquet, program_map.csv}; writes
``forecast.paths.out_parquet`` (one row per in-flight grid row, synthetic rows included),
``forecast.paths.out_evidence`` (slot_id -> evidence break_id, for traceability),
``forecast.paths.out_app_json`` and refreshes this phase's rows in outputs/assumptions_log.md.
Without ``--final`` the run is marked provisional (program_map not yet signed off at Gate 2).
Method: see optimizer/forecast/core.py (module docstring) and METHODOLOGY section 2.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from optimizer.config import load_config, project_root
from optimizer.forecast.core import ForecastParams, forecast_targets, is_live_special

FORECAST_COLS = ["etam_title_used", "forecast_level", "evidence_n", "aud_abs_p10", "aud_abs_p50", "aud_abs_p90",
                 "rating_pct_p50", "cpm_p50", "evidence_desc", "pool_top_programs", "est_program", "n_program",
                 "est_slot", "n_slot", "est_channel_daypart", "n_channel_daypart", "raw_level_est", "parent_est",
                 "widening", "is_new_program", "is_live", "live_special", "low_sample", "high_uncertainty",
                 "assumption", "notes", "match_type", "match_confidence"]


def _p(root: Path, rel: str) -> Path:
    q = Path(rel)
    return q if q.is_absolute() else root / q


def load_history(cfg: dict[str, Any], root: Path, date_from=None, date_to=None) -> pd.DataFrame:
    """breaks.parquet for the configured target, events excluded when etam.exclude_event_days."""
    b = pd.read_parquet(_p(root, cfg["forecast"]["paths"]["breaks"]))
    tgt = cfg.get("target", {}).get("buying_target")
    if tgt and "target" in b.columns and (b["target"] == tgt).any():
        b = b[b["target"] == tgt]
    if cfg.get("etam", {}).get("exclude_event_days", True):
        b = b[~b["is_event"].astype(bool)]
    if date_from is not None:
        b = b[b["broadcast_date"] >= pd.Timestamp(date_from)]
    if date_to is not None:
        b = b[b["broadcast_date"] <= pd.Timestamp(date_to)]
    return b.reset_index(drop=True)


def low_sample_channels(root: Path, cfg: dict[str, Any]) -> set[str]:
    b = pd.read_parquet(_p(root, cfg["forecast"]["paths"]["breaks"]), columns=["channel", "low_sample_channel"])
    return set(b.loc[b["low_sample_channel"].astype(bool), "channel"].unique())


def build_october_targets(grid: pd.DataFrame, pmap: pd.DataFrame, p: ForecastParams, low_sample: set[str]
                          ) -> pd.DataFrame:
    """One target per in-flight, non-synthetic grid row, from grid + program_map."""
    g = grid[grid["in_flight"] & ~grid["is_synthetic"]].copy()
    pm = pmap.rename(columns={"grid_title": "title_en", "is_rerun": "map_is_rerun", "is_live": "map_is_live"})
    g = g.merge(pm[["channel", "title_en", "etam_title", "etam_channel", "match_type", "confidence",
                    "slot_pool_rule", "flags"]], on=["channel", "title_en"], how="left", validate="many_to_one")
    if g["match_type"].isna().any():
        missing = g.loc[g["match_type"].isna(), ["channel", "title_en"]].drop_duplicates()
        raise ValueError(f"grid titles missing from program_map: {missing.to_dict('records')}")
    recs = []
    for r in g.to_dict("records"):
        et = r["etam_title"] if isinstance(r["etam_title"], str) and r["etam_title"] != "NONE" else ""
        notes, codes = [], []
        same_ch = et and r["etam_channel"] == r["channel"]
        if et and not same_ch:
            notes.append(f"cross-channel match {et} on {r['etam_channel']} not used (other channel's audience level)")
            codes.append("CROSS_CHANNEL_HISTORY_NOT_USED")
        conf_ok = float(r["confidence"]) >= p.min_match_confidence
        if et and same_ch and not conf_ok:
            notes.append(f"match confidence {r['confidence']:.2f} < min_match_confidence")
        ls = r["channel"] in low_sample
        live_sp = is_live_special(r["title_en"], bool(r["is_live"]), p.live_special_patterns)
        is_new = r["match_type"] == "new program"
        generic = r["match_type"] == "generic-slot"
        use_prog = bool(et and same_ch and conf_ok and not live_sp and not ls)
        if is_new:
            codes.append("NEW_PROGRAM_SLOT_BASELINE_NO_GENRE_FACTOR")
        if generic:
            codes.append("GENERIC_SLOT_POOL")
        if live_sp:
            codes.append("LIVE_SPECIAL_SLOT_BASELINE_EXCL_EVENTS")
        if ls:
            codes.append("LOW_SAMPLE_CHANNEL_POOLED")
        rule = r["slot_pool_rule"] if isinstance(r["slot_pool_rule"], str) else ""
        recs.append({
            "target_id": r["slot_id"], "channel": r["channel"], "weekday": r["weekday"],
            "start_min": int(r["start_min"]), "end_min": int(r["end_min"]), "is_rerun": bool(r["is_rerun"]),
            "etam_title": et if use_prog else "", "use_program": use_prog,
            "prefer_rerun_pool": "prefer rerun" in rule.lower(), "low_sample": ls, "is_new": is_new,
            "live_special": live_sp, "match_type": r["match_type"], "match_confidence": float(r["confidence"]),
            "assumption_codes": codes, "notes": "; ".join(notes),
        })
    return pd.DataFrame(recs)


def assemble(grid: pd.DataFrame, targets: pd.DataFrame, fc: pd.DataFrame, cfg: dict[str, Any], p: ForecastParams
             ) -> pd.DataFrame:
    U = float(cfg["target"]["universe"])
    t = targets.merge(fc, on="target_id", how="left", validate="one_to_one")
    t = t.rename(columns={"target_id": "slot_id", "p10": "aud_abs_p10", "p50": "aud_abs_p50", "p90": "aud_abs_p90",
                          "is_new": "is_new_program"})
    t["etam_title_used"] = np.where(t["forecast_level"] == "program", t["etam_title"], "")
    base_cols = ["slot_id", "channel", "air_date", "weekday", "start_min", "end_min", "start_hhmm", "end_hhmm", "tier",
                 "rate_usd", "rate_aed", "title_en", "program_name", "is_rerun", "is_live", "is_synthetic",
                 "has_conflict"]
    gi = grid[grid["in_flight"]][base_cols + ["assumption"]].copy()
    real = gi[~gi["is_synthetic"]].merge(t.drop(columns=["channel", "weekday", "start_min", "end_min", "is_rerun"]),
                                         on="slot_id", how="left", validate="one_to_one")
    # carried-forward MBC 1 week-4 rows: identical forecast to the week-3 source row
    syn = gi[gi["is_synthetic"]].copy()
    if len(syn):
        syn["_src_date"] = syn["air_date"] - pd.Timedelta(days=7)
        src = real.rename(columns={"air_date": "_src_date"})
        keep = ["channel", "_src_date", "start_min", "title_en", "slot_id"] + [
            c for c in real.columns if c not in base_cols + ["assumption"]]
        syn = syn.merge(src[keep].rename(columns={"slot_id": "source_slot_id"}),
                        on=["channel", "_src_date", "start_min", "title_en"], how="left", validate="one_to_one")
        if syn["source_slot_id"].isna().any():
            raise ValueError(f"{syn['source_slot_id'].isna().sum()} synthetic rows without a week-3 source row")
        syn = syn.drop(columns=["_src_date"])
        syn["assumption_codes"] = syn["assumption_codes"].apply(lambda c: list(c) + ["CARRIED_FORWARD_WK3_FORECAST"])
    real["source_slot_id"] = ""
    out = pd.concat([real, syn], ignore_index=True)
    out["assumption"] = [";".join([a] if isinstance(a, str) and a else []) for a in out["assumption"]]
    out["assumption"] = [";".join(x for x in [a, *codes] if x) for a, codes in zip(out["assumption"], out["assumption_codes"])]
    out = out.drop(columns=["assumption_codes", "etam_title", "use_program", "prefer_rerun_pool"])
    out["is_live"] = out["is_live"].astype(bool)
    out["rating_pct_p50"] = out["aud_abs_p50"] / U * 100.0
    out["cpm_p50"] = np.where(out["aud_abs_p50"] > 0, out["rate_usd"] / (out["aud_abs_p50"] / 1000.0), np.inf)
    relw = (out["aud_abs_p90"] - out["aud_abs_p10"]) / out["aud_abs_p50"].where(out["aud_abs_p50"] > 0)
    out["rel_width"] = relw
    out["high_uncertainty"] = out["live_special"].astype(bool) | (relw.fillna(np.inf) > p.high_uncertainty_rel_width)
    out = out.sort_values(["channel", "air_date", "start_min", "slot_id"]).reset_index(drop=True)
    assert out["slot_id"].is_unique
    return out


def app_json(out: pd.DataFrame, cfg: dict[str, Any], p: ForecastParams, hist: pd.DataFrame, provisional: bool) -> dict:
    def hhmm(v):
        v = int(v)
        return f"{v // 100:02d}:{v % 100:02d}"

    rows = []
    for r in out.to_dict("records"):
        flags = [n for n, on in (("new_program", r["is_new_program"]), ("live", r["is_live"]),
                                 ("live_special", r["live_special"]), ("low_sample", r["low_sample"]),
                                 ("high_uncertainty", r["high_uncertainty"]), ("carried_forward", r["is_synthetic"]),
                                 ("overlap_conflict", r["has_conflict"])) if on]
        cpm = r["cpm_p50"]
        rows.append({
            "slot_id": r["slot_id"], "channel": r["channel"], "air_date": pd.Timestamp(r["air_date"]).strftime("%Y-%m-%d"),
            "weekday": r["weekday"], "start_hhmm": hhmm(r["start_hhmm"]), "end_hhmm": hhmm(r["end_hhmm"]),
            "title_en": r["title_en"], "tier": r["tier"], "rate_usd": round(float(r["rate_usd"]), 2),
            "aud_p10": round(float(r["aud_abs_p10"])), "aud_p50": round(float(r["aud_abs_p50"])),
            "aud_p90": round(float(r["aud_abs_p90"])), "rating_pct_p50": round(float(r["rating_pct_p50"]), 4),
            "cpm_p50": round(float(cpm), 2) if np.isfinite(cpm) else None,
            "forecast_level": r["forecast_level"], "evidence_n": int(r["evidence_n"]), "flags": flags,
            "assumption": r["assumption"],
        })
    meta = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "method": (f"hierarchical robust baseline ({p.estimator}; program -> slot -> channel x daypart), "
                   f"empirical-Bayes shrinkage k={p.shrinkage_k}, min_breaks={p.min_breaks}, "
                   f"day-cluster bootstrap p10/p90 (B={p.bootstrap_n}); per-spot audience = rating_abs = "
                   f"TRP_abs*60/break_sec; event breaks excluded"),
        "target": cfg["target"]["buying_target"],
        "universe": cfg["target"]["universe"], "universe_is_official": bool(cfg["target"]["universe_is_official"]),
        "history_window": [hist["broadcast_date"].min().strftime("%Y-%m-%d"), hist["broadcast_date"].max().strftime("%Y-%m-%d")],
        "provisional": bool(provisional),
        "config_sha256": cfg.get("_config_sha256"),
        "n_slots": len(rows),
    }
    return {"meta": meta, "slots": rows}


def update_assumptions_log(root: Path, out: pd.DataFrame, p: ForecastParams, provisional: bool) -> None:
    path = root / "outputs" / "assumptions_log.md"
    if not path.exists():
        return
    today = dt.date.today().isoformat()
    n = len(out)
    cnt = lambda m: int(m.sum())
    usd = lambda m: float(out.loc[m, "rate_usd"].sum())
    rows = [
        ("forecast:new", "#9", f"New October programmes ({cnt(out.is_new_program)} slots, ${usd(out.is_new_program):,.0f}) forecast from the slot baseline (what aired in that window in Sep = predecessor); no genre factor (not measurable with 3 weeks of data); interval widened x{p.interval_widening.get('new_program')}", "forecast.interval_widening.new_program", "flag is_new_program"),
        ("forecast:live", "#12", f"Live sport/special rows ({cnt(out.live_special)} slots: is_live and title matches forecast.live_special_patterns, e.g. NADEENA (KHALEEJI 27), EXTREME H WORLD CUP LIVE) = slot baseline excluding event breaks, never extrapolated from 19 Sep; interval widened x{p.interval_widening.get('live_special')}. Recurring live studio shows (MBC NEWS LIVE, THE MORNING SHOW, NADEENA, MBC IN A WEEK) use the normal hierarchy", "forecast.live_policy, forecast.live_special_patterns", "flag high_uncertainty"),
        ("forecast:lowsample", "—", f"Low-sample channels (low_sample_channel: MBC ACTION, MBC MAX; {cnt(out.low_sample)} slots) use only the pooled channel x daypart level ({p.low_sample_policy}); interval widened x{p.interval_widening.get('low_sample')}", "forecast.low_sample_policy", "flag low_sample"),
        ("forecast:xchannel", "—", f"Cross-channel program matches (history on another channel) are NOT used at program level (audience level differs by channel): {cnt(out.assumption.str.contains('CROSS_CHANNEL'))} slots, ${usd(out.assumption.str.contains('CROSS_CHANNEL')):,.0f}, fall back to the slot level", "—", "assumption CROSS_CHANNEL_HISTORY_NOT_USED"),
        ("forecast:wk4", "#4", f"MBC 1 week-4 carried-forward rows ({cnt(out.is_synthetic)}) get exactly the forecast of their week-3 source row", "grid.mbc1_week4", "assumption ASSUMPTION_MBC1_WK4"),
        ("forecast:season", "#9", "No seasonality / trend (forecast.seasonality_trend=false): September level assumed for October", "forecast.seasonality_trend", "revisit when August lands"),
    ]
    if provisional:
        rows.append(("forecast:provisional", "—", "Forecast is PROVISIONAL: built on program_map.csv before Gate 2 sign-off; re-run after corrections", "—", "meta.provisional=true"))
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if "[forecast:" not in l]
    while lines and not lines[-1].strip():
        lines.pop()
    for tag, issue, text, key, impact in rows:
        lines.append(f"| {today} | 4 | {issue} | {text} [{tag}] | {key} | {impact} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _ = n


def run(cfg: dict[str, Any], provisional: bool = True, root: Path | None = None) -> pd.DataFrame:
    root = root or project_root()
    fcfg = cfg["forecast"]
    if fcfg.get("seasonality_trend"):
        raise NotImplementedError("forecast.seasonality_trend=true needs August history (METHODOLOGY 2.2)")
    p = ForecastParams.from_cfg(cfg)
    grid = pd.read_parquet(_p(root, fcfg["paths"]["grid"]))
    pmap = pd.read_csv(_p(root, fcfg["paths"]["program_map"]))
    hist = load_history(cfg, root)
    targets = build_october_targets(grid, pmap, p, low_sample_channels(root, cfg))
    fc, evid = forecast_targets(targets, hist, p, with_intervals=True, keep_evidence=True)
    out = assemble(grid, targets, fc, cfg, p)
    n_if = int(grid["in_flight"].sum())
    assert len(out) == n_if, (len(out), n_if)
    assert (out["aud_abs_p10"] <= out["aud_abs_p50"] + 1e-9).all() and (out["aud_abs_p50"] <= out["aud_abs_p90"] + 1e-9).all()
    cols = ["slot_id", "channel", "air_date", "weekday", "start_min", "end_min", "start_hhmm", "end_hhmm", "tier",
            "rate_usd", "rate_aed", "title_en", "program_name", "is_rerun", "is_synthetic", "source_slot_id",
            "has_conflict", "rel_width"] + FORECAST_COLS
    out = out[cols]
    op = _p(root, fcfg["paths"]["out_parquet"])
    op.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(op, index=False)
    if evid is not None:
        evid.rename(columns={"target_id": "slot_id"}).to_parquet(_p(root, fcfg["paths"]["out_evidence"]), index=False)
    jp = _p(root, fcfg["paths"]["out_app_json"])
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(app_json(out, cfg, p, hist, provisional), ensure_ascii=False, indent=1), encoding="utf-8")
    update_assumptions_log(root, out, p, provisional)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--final", action="store_true", help="mark the output non-provisional (after Gate 2 sign-off)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = run(cfg, provisional=not a.final)
    usd = out.groupby("forecast_level")["rate_usd"].sum()
    print(f"slot_forecast rows: {len(out)} (synthetic {int(out.is_synthetic.sum())}); provisional={not a.final}")
    print(pd.DataFrame({"slots": out["forecast_level"].value_counts(), "usd": usd.round(0)}))
    print(out.groupby("channel").agg(p50_median=("aud_abs_p50", "median"), cpm_median=("cpm_p50", "median")).round(1))


if __name__ == "__main__":
    main()
