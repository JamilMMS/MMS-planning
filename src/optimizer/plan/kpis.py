"""Plan KPIs (Nielsen formulas from optimizer.metrics.nielsen only) and plan / KPI files."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..metrics import nielsen
from .problem import abs_path

PLAN_COLUMNS = ["slot_id", "channel", "air_date", "weekday", "start_min", "end_min", "start_hhmm", "end_hhmm",
                "title_en", "program_name", "tier", "rate_usd", "rate_aed", "aud_abs_p10", "aud_abs_p50",
                "aud_abs_p90", "rating_pct_p50", "cpm", "forecast_level", "flags", "week", "spot_copy"]


def _hhmm(m: float) -> str:
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


def plan_frame(prob, units: np.ndarray) -> pd.DataFrame:
    """One row per spot. cpm = nielsen.cpm(rate_usd, audience) per spot."""
    df = prob.unit_frame(units)
    if "start_hhmm" not in df.columns:
        df["start_hhmm"] = df["start_min"].map(_hhmm)
    if "end_hhmm" not in df.columns:
        df["end_hhmm"] = df["end_min"].map(_hhmm)
    audc = prob.cfg["plan"].get("audience_column", "aud_abs_p50")
    df["cpm"] = [nielsen.cpm(c, a) if a > 0 else None for c, a in zip(df["rate_usd"], df[audc])]
    for c in PLAN_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df.sort_values(["air_date", "channel", "start_min", "slot_id", "spot_copy"], kind="mergesort")
    df["air_date"] = pd.to_datetime(df["air_date"]).dt.strftime("%Y-%m-%d")
    return df[PLAN_COLUMNS].reset_index(drop=True)


def _block(d: pd.DataFrame, audc: str, U: float, total_spend: float) -> dict:
    sp = float(d["rate_usd"].sum())
    imp = nielsen.grp_abs(d[audc].astype(float).tolist()) if len(d) else 0.0
    gp = nielsen.grp_pct(imp, U) if imp else 0.0
    return {"spend_usd": sp, "spend_aed": float(d["rate_aed"].sum()), "share_of_spend": sp / total_spend if total_spend else None,
            "spots": int(len(d)), "impressions": imp, "grp_pct": gp,
            "cpm": nielsen.cpm(sp, imp) if imp > 0 else None,
            "cpp": nielsen.cost_per_rating_pct(sp, gp) if gp > 0 else None}


def compute_kpis(plan: pd.DataFrame, engine, cfg: dict) -> dict:
    audc = cfg["plan"].get("audience_column", "aud_abs_p50")
    U = float(cfg["target"]["universe"])
    spend = float(plan["rate_usd"].sum())
    imp = nielsen.grp_abs(plan[audc].astype(float).tolist()) if len(plan) else 0.0
    gp = nielsen.grp_pct(imp, U) if imp else 0.0
    trp = nielsen.trp_pct(plan["rating_pct_p50"].astype(float).tolist()) if len(plan) else 0.0
    sched = plan.assign(aud_abs=plan[audc].astype(float))
    r = engine.reach(sched) if len(plan) else None
    reach1 = r["reach_1plus_abs"] if r else 0.0
    out: dict[str, Any] = {
        "target": cfg["target"]["buying_target"], "universe": U,
        "universe_is_official": bool(cfg["target"].get("universe_is_official", False)),
        "spend_usd": spend, "spend_aed": float(plan["rate_aed"].sum()),
        "budget_usd": float(cfg["budget"]["total_usd"]), "budget_utilisation": spend / float(cfg["budget"]["total_usd"]),
        "spots": int(len(plan)),
        "impressions": imp, "grp_abs": imp, "grp_pct": gp, "trp_pct": trp,
        "trp_note": "buying target = base audience (target.*): TRP % = GRP %",
        "reach_1plus_abs": reach1, "reach_1plus_pct": r["reach_1plus_pct"] if r else 0.0,
        "reach_3plus_abs": r["reach_3plus_abs"] if r else 0.0, "reach_3plus_pct": r["reach_3plus_pct"] if r else 0.0,
        "reach_n_plus_pct": r["reach_n_plus_pct"] if r else {},
        "reach_method": r["method"] if r else engine.method,
        "ots": nielsen.ots(imp, reach1) if reach1 > 0 else None,
        "cpm": nielsen.cpm(spend, imp) if imp > 0 else None,
        "cpp": nielsen.cost_per_rating_pct(spend, gp) if gp > 0 else None,
        "impressions_p10": float(plan["aud_abs_p10"].sum()), "impressions_p90": float(plan["aud_abs_p90"].sum()),
    }
    if r:
        out["reach_params_summary"] = r.get("params_summary")
    per_ch = {}
    for ch, d in plan.groupby("channel", sort=True):
        b = _block(d, audc, U, spend)
        rc = engine.reach(d.assign(aud_abs=d[audc].astype(float)))
        b.update(reach_1plus_pct=rc["reach_1plus_pct"], reach_3plus_pct=rc["reach_3plus_pct"], reach_method=rc["method"])
        per_ch[ch] = b
    out["per_channel"] = per_ch
    out["per_week"] = {int(w): _block(d, audc, U, spend) for w, d in plan.groupby("week", sort=True)}
    out["per_tier"] = {t: _block(d, audc, U, spend) for t, d in plan.groupby("tier", sort=True)}
    out["per_forecast_level"] = {t: _block(d, audc, U, spend) for t, d in plan.groupby("forecast_level", sort=True)}
    flagged = plan["flags"].fillna("").str.contains("high_uncertainty")
    out["high_uncertainty_share_of_spend"] = float(plan.loc[flagged, "rate_usd"].sum()) / spend if spend else None
    return out


# ----------------------------------------------------------------------------- metadata
def sha256_file(p: Path) -> str | None:
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_metadata(cfg: dict) -> dict:
    pc = cfg["plan"]
    fc_p = abs_path(pc["forecast_path"])
    rp = abs_path(cfg["reach"]["params_path"] if str(cfg["reach"]["mode"]).upper() == "ESTIMATE"
                  else cfg["reach"]["calibration"]["params_path"])
    cf = abs_path(pc["conflicts_path"])
    meta_p = abs_path(pc.get("forecast_meta_path", "outputs/slot_forecast_for_app.json"))
    fmeta = {}
    if meta_p.exists():
        try:
            fmeta = json.loads(meta_p.read_text(encoding="utf-8")).get("meta", {})
        except Exception:  # noqa: BLE001
            fmeta = {}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=str(abs_path("."))).stdout.strip() or None
    except Exception:  # noqa: BLE001
        commit = None
    try:
        import ortools
        ov = ortools.__version__
    except Exception:  # noqa: BLE001
        ov = None
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_path": cfg.get("_config_path"), "config_sha256": cfg.get("_config_sha256"),
        "slot_forecast_path": str(fc_p), "slot_forecast_sha256": sha256_file(fc_p),
        "reach_params_path": str(rp), "reach_params_sha256": sha256_file(rp),
        "slot_conflicts_path": str(cf), "slot_conflicts_sha256": sha256_file(cf),
        "forecast_meta_path": str(meta_p), "forecast_provisional": fmeta.get("provisional"),
        "forecast_generated_at": fmeta.get("generated_at"), "forecast_config_sha256": fmeta.get("config_sha256"),
        "reach_mode": str(cfg["reach"]["mode"]).upper(), "random_seed": cfg["plan"].get("random_seed"),
        "git_commit": commit, "python": platform.python_version(), "ortools": ov,
    }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)
    raise TypeError(type(o))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default, ensure_ascii=False), encoding="utf-8")
