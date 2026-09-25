"""Fit the ESTIMATE-mode reach parameters from the eTAM break file (re-runnable).

    python -m optimizer.reach.fit --config config/plan_config.yaml

Reads ``data/processed/breaks.parquet`` (whatever version is current: the is_event flags
may be refined by the ingestion agent - just re-run), writes ``reach.params_path``
(default ``data/processed/reach_params.json``) and refreshes the numbers table in
``src/optimizer/reach/ESTIMATE_METHOD.md`` between the PARAMS markers.

Derivation (see ESTIMATE_METHOD.md for the argument and the caveats):

1. Turnover. For every break, reach_abs / rating_abs = r0 + lambda * L (L = break seconds):
   people who touch a break of length L vs its average audience. Fitted per channel by
   non-negative weighted least squares on reach_abs = a (r0 + lambda L), weights 1/a
   (Poisson-type variance). lambda = inflow rate of new viewers per second per viewer
   => mean viewing-session length tau = 1 / lambda.
2. Daily audience-minutes AM_c: hourly break-audience profile (time-weighted mean audience
   of breaks in each broadcast hour, averaged over days) x 60, summed over the day.
3. Daily sessions S_c = AM_c / tau_c; daily cume D_c = S_c / m (m = sessions per daily
   viewer, ASSUMPTION), bounded below by the mean daily maximum break reach and above by
   S_c (m = 1) and the universe.
4. Rmax_c = flight cume from D_c with a Beta day-to-day model (rho = intra-person
   correlation of daily viewing, ASSUMPTION), capped at ``rmax_cap_pct``.
5. Initial slope s0 = min(r_spot, initial_slope_cap), r_spot = r0 + lambda * spot_len
   (reach of one spot per GRP of it); k = Rmax / s0.
6. Curve form: both saturating forms are fitted to the only accumulation data available,
   reach vs break length (reach_abs = a (r0 + c h(L; T))); the lower pooled SSE wins, ties
   (relative difference below ``form_tie_tolerance``) go to the hyperbolic form.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

from ..config import load_config
from .common import abs_path, daypart_of, flight_days
from .curves import beta_day_cume

METHOD_DOC = Path(__file__).with_name("ESTIMATE_METHOD.md")
BEGIN, END = "<!-- PARAMS:BEGIN -->", "<!-- PARAMS:END -->"


# ----------------------------------------------------------------------------- helpers
_abs = abs_path


def estimate_cfg(cfg: dict) -> dict:
    return cfg["reach"]["estimate"]


def load_breaks(cfg: dict, path: str | Path | None = None) -> pd.DataFrame:
    p = _abs(path or cfg["reach"]["estimate"]["breaks_path"])
    df = pd.read_parquet(p)
    need = {"channel", "broadcast_date", "start_sec", "break_sec", "rating_abs", "reach_abs", "is_event"}
    miss = sorted(need - set(df.columns))
    if miss:
        raise ValueError(f"{p}: missing columns {miss}")
    df.attrs["path"] = str(p)
    return df


def _usable(breaks: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Non-event breaks (zero-rated breaks KEPT: they are real audience-minutes of 0)."""
    b = breaks
    if cfg["etam"].get("exclude_event_days", True):
        b = b[~b["is_event"].astype(bool)]
    return b[b["rating_abs"].notna() & b["reach_abs"].notna()].copy()


# ----------------------------------------------------------------------------- step 1
def fit_turnover(b: pd.DataFrame) -> dict[str, float]:
    """reach_abs = a (r0 + lambda L), NNLS with weights 1/a. Returns r0, lambda, n."""
    a = b["rating_abs"].to_numpy(float)
    L = b["break_sec"].to_numpy(float)
    R = b["reach_abs"].to_numpy(float)
    w = 1.0 / np.sqrt(a)
    X = np.c_[a, a * L] * w[:, None]
    coef, _ = nnls(X, R * w)
    return {"r0": float(coef[0]), "lambda_per_sec": float(coef[1]), "n_breaks": int(len(b))}


# ----------------------------------------------------------------------------- step 2
def audience_minutes(b: pd.DataFrame, hours: tuple[int, int] = (3, 27)) -> float:
    """Daily break-audience minutes of one cell: mean over days of the time-weighted mean
    break audience in each broadcast hour, x 60, summed over the hours present.
    Hours with no breaks on any day are linearly interpolated from neighbouring hours."""
    x = b.assign(h=b["start_sec"] // 3600, aL=b["rating_abs"] * b["break_sec"])
    dh = x.groupby(["broadcast_date", "h"]).agg(aL=("aL", "sum"), L=("break_sec", "sum"))
    prof = (dh["aL"] / dh["L"]).groupby(level="h").mean()
    lo, hi = hours
    wanted = [h for h in range(lo, hi) if (prof.index.min() <= h <= prof.index.max())]
    prof = prof.reindex(wanted).interpolate(limit_area="inside").fillna(0.0)
    return float(prof.sum() * 60.0)


def daily_max_reach(b: pd.DataFrame, all_breaks: pd.DataFrame) -> float:
    """Mean over days of the largest single-break reach (a hard lower bound on daily cume)."""
    days = all_breaks["broadcast_date"].nunique()
    per_day = b.groupby("broadcast_date")["reach_abs"].max()
    return float(per_day.sum() / max(days, 1))


# ----------------------------------------------------------------------------- step 6
def _sat(L: np.ndarray, c: float, T: float, form: str) -> np.ndarray:
    if form == "hyperbolic":
        return c * L / (T + L)
    return c * -np.expm1(-L / T)


def fit_form_on_breaks(b: pd.DataFrame) -> dict[str, float]:
    """Weighted SSE of reach_abs = a (r0 + c h(L;T)) for both saturation forms."""
    a = b["rating_abs"].to_numpy(float)
    L = b["break_sec"].to_numpy(float)
    R = b["reach_abs"].to_numpy(float)
    w = 1.0 / np.sqrt(a)
    out = {}
    for form in ("hyperbolic", "negexp"):
        def res(p, form=form):
            r0, lc, lT = p
            return (R - a * (r0 + _sat(L, math.exp(lc), math.exp(lT), form))) * w
        best = None
        for T0 in (60.0, 600.0, 6000.0, 60000.0):
            x0 = [1.0, math.log(max(4e-4 * T0, 1e-3)), math.log(T0)]
            r = least_squares(res, x0, bounds=([0.5, -12, 0], [2.0, 8, 16]))
            if best is None or r.cost < best.cost:
                best = r
        r0, lc, lT = best.x
        out[form] = {"sse": float(2 * best.cost), "r0": float(r0), "c": float(math.exp(lc)),
                     "T_sec": float(math.exp(lT))}
    return out


# ----------------------------------------------------------------------------- main fit
def fit_estimate_params(cfg: dict, breaks: pd.DataFrame | None = None) -> dict[str, Any]:
    ecfg = estimate_cfg(cfg)
    U = float(cfg["target"]["universe"])
    spot_len = float(cfg["grid"]["spot_length_sec"])
    n_days = flight_days(cfg)
    m_sess = float(ecfg["sessions_per_daily_viewer"])
    m_sess_dp = float(ecfg.get("sessions_per_daily_viewer_daypart", 1.0))
    rho = float(ecfg["day_to_day_rho"])
    s0_cap = float(ecfg["initial_slope_cap"])
    rmax_cap = float(ecfg["rmax_cap_pct"])
    min_aud = float(ecfg["min_rating_abs_for_ratio"])
    min_n = int(ecfg["min_breaks_for_channel_fit"])
    granularity = ecfg["cell_granularity"]
    dayparts = ecfg["dayparts"]
    override = ecfg.get("daily_reach_override") or {}

    if breaks is None:
        breaks = load_breaks(cfg)
    b_all = _usable(breaks, cfg)
    b_ratio = b_all[b_all["rating_abs"] >= min_aud]

    pooled = fit_turnover(b_ratio)
    turn = {}
    for ch, g in b_ratio.groupby("channel"):
        t = fit_turnover(g)
        t["source"] = "channel"
        if t["n_breaks"] < min_n or t["lambda_per_sec"] <= 0:
            t = {**pooled, "source": "pooled_all_channels", "n_breaks_channel": t["n_breaks"]}
        turn[ch] = t

    # curve form
    forms = {}
    for ch, g in b_ratio.groupby("channel"):
        if turn[ch]["source"] == "channel":
            forms[ch] = fit_form_on_breaks(g)
    sse_h = sum(v["hyperbolic"]["sse"] for v in forms.values())
    sse_e = sum(v["negexp"]["sse"] for v in forms.values())
    rel = abs(sse_h - sse_e) / max(min(sse_h, sse_e), 1e-12)
    tol = float(ecfg["form_tie_tolerance"])
    if ecfg["model_form"] in ("hyperbolic", "negexp"):
        chosen, why = ecfg["model_form"], "forced by config reach.estimate.model_form"
    elif rel < tol:
        chosen, why = "hyperbolic", (f"tie: pooled SSE differ by {rel:.4%} < tolerance {tol:.2%}; "
                                     "hyperbolic chosen (conservative: lower reach at high GRP)")
    else:
        chosen = "hyperbolic" if sse_h < sse_e else "negexp"
        why = f"lower pooled weighted SSE on reach-vs-break-length ({rel:.2%} better)"

    # cells
    if granularity == "channel":
        keys = [(ch, None) for ch in sorted(b_all["channel"].unique())]
    elif granularity == "channel_daypart":
        b_all = b_all.assign(daypart=[daypart_of(s / 60.0, dayparts) for s in b_all["start_sec"]])
        keys = [(ch, dp) for ch in sorted(b_all["channel"].unique()) for dp in dayparts]
    else:
        raise ValueError(f"unknown reach.estimate.cell_granularity {granularity!r}")

    cells = {}
    for ch, dp in keys:
        g = b_all[b_all["channel"] == ch]
        hours = (3, 27)
        m_use = m_sess
        if dp is not None:
            g = g[g["daypart"] == dp]
            lo, hi = dayparts[dp]
            hours = (int(lo // 60), int(math.ceil(hi / 60)))
            m_use = m_sess_dp
        if g.empty:
            continue
        t = turn[ch]
        tau_min = 1.0 / t["lambda_per_sec"] / 60.0
        am = audience_minutes(g, hours)
        sessions = am / tau_min
        lb = daily_max_reach(g, b_all[b_all["channel"] == ch])
        ub = min(sessions, U)
        key = ch if dp is None else f"{ch}|{dp}"
        if key in override:
            d_abs, d_src = float(override[key]), "OBSERVED override (reach.estimate.daily_reach_override)"
        else:
            d_abs, d_src = sessions / m_use, "ASSUMPTION sessions_per_daily_viewer"
        d_abs = min(max(d_abs, lb), ub if key not in override else U)
        d = d_abs / U
        rmax_unc = beta_day_cume(d, rho, n_days) * 100.0
        rmax = min(rmax_unc, rmax_cap)
        r_spot = t["r0"] + t["lambda_per_sec"] * spot_len
        s0 = min(r_spot, s0_cap)
        cells[key] = {
            "channel": ch, "daypart": dp,
            "rmax_pct": rmax, "k": rmax / s0, "initial_slope": s0,
            "ratio_r0": t["r0"], "lambda_per_sec": t["lambda_per_sec"], "tau_min": tau_min,
            "reach_rating_ratio_spot": r_spot,
            "reach_rating_ratio_median_break": t["r0"] + t["lambda_per_sec"] * float(
                b_ratio.loc[b_ratio["channel"] == ch, "break_sec"].median()),
            "turnover_source": t["source"], "n_breaks_ratio": t.get("n_breaks_channel", t["n_breaks"]),
            "audience_minutes_per_capita": am / U,
            "daily_sessions_pct": sessions / U * 100.0,
            "daily_cume_pct": d * 100.0, "daily_cume_source": d_src,
            "daily_cume_lb_pct": lb / U * 100.0, "daily_cume_ub_pct": ub / U * 100.0,
            "time_spent_per_daily_viewer_min": am / d_abs,
            "next_day_repeat_rate": d + rho * (1 - d),
            "rmax_pct_uncapped": rmax_unc,
            # duplication index of two small equal spots in this cell implied by the curve
            # (hyperbolic: 2 s0^2 / Rmax, negexp: s0^2 / Rmax; Rmax as a fraction)
            "implied_same_cell_dup_index": (2.0 if chosen == "hyperbolic" else 1.0) * s0 ** 2 / (rmax / 100.0),
        }

    # sensitivity of Rmax to the two assumptions (informational)
    grid = {}
    for mm in ecfg["report_sessions_grid"]:
        for rr in ecfg["report_rho_grid"]:
            row = {}
            for key, c in cells.items():
                d = min(max(c["daily_sessions_pct"] / mm, c["daily_cume_lb_pct"]), c["daily_cume_ub_pct"]) / 100
                row[key] = min(beta_day_cume(d, rr, n_days) * 100.0, rmax_cap)
            grid[f"m={mm}, rho={rr}"] = row

    src = Path(breaks.attrs.get("path", ecfg["breaks_path"]))
    sha = hashlib.sha256(_abs(src).read_bytes()).hexdigest() if _abs(src).exists() else None
    return {
        "method": "ESTIMATE",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": cfg.get("_config_path"), "config_sha256": cfg.get("_config_sha256"),
        "breaks_path": str(src), "breaks_sha256": sha,
        "n_breaks_total": int(len(breaks)), "n_breaks_used": int(len(b_all)),
        "n_breaks_ratio": int(len(b_ratio)),
        "universe": U, "universe_is_official": bool(cfg["target"].get("universe_is_official", False)),
        "flight_days": n_days, "spot_length_sec": spot_len,
        "form": chosen,
        "form_selection": {"reason": why, "pooled_sse_hyperbolic": sse_h, "pooled_sse_negexp": sse_e,
                           "relative_difference": rel, "per_channel": forms},
        "assumptions": {
            "sessions_per_daily_viewer": m_sess, "sessions_per_daily_viewer_daypart": m_sess_dp,
            "day_to_day_rho": rho, "initial_slope_cap": s0_cap, "rmax_cap_pct": rmax_cap,
            "turnover_pooled": pooled,
        },
        "cell_granularity": granularity,
        "cells": cells,
        "rmax_sensitivity_grid": grid,
    }


# ----------------------------------------------------------------------------- doc table
def params_table_md(p: dict) -> str:
    lines = [
        f"_Generated {p['generated_at']} by `python -m optimizer.reach.fit` from "
        f"`{Path(p['breaks_path']).name}` ({p['n_breaks_used']} non-event breaks incl. zero-rated; "
        f"{p['n_breaks_ratio']} with audience >= threshold used for ratios). Universe "
        f"{p['universe']:,.0f} ({'official' if p['universe_is_official'] else 'INFERRED'}); "
        f"flight {p['flight_days']} days. Method label: **ESTIMATE**._",
        "",
        f"**Curve form chosen: `{p['form']}`** — {p['form_selection']['reason']} "
        f"(pooled SSE hyperbolic {p['form_selection']['pooled_sse_hyperbolic']:.4g}, "
        f"negexp {p['form_selection']['pooled_sse_negexp']:.4g}).",
        "",
        "| Cell | r0 | lambda (1e-4/s) | tau (min) | reach/rating spot (30s) | reach/rating median break "
        "| AM/capita (min/day) | daily sessions % | daily cume % [LB, UB] | TSV/viewer (min) "
        "| repeat rate | Rmax % | k (GRP %) | slope s0 | dup index |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, c in p["cells"].items():
        src = "" if c["turnover_source"] == "channel" else " (pooled)"
        lines.append(
            f"| {key} | {c['ratio_r0']:.3f} | {c['lambda_per_sec'] * 1e4:.2f}{src} | {c['tau_min']:.1f} "
            f"| {c['reach_rating_ratio_spot']:.3f} | {c['reach_rating_ratio_median_break']:.3f} "
            f"| {c['audience_minutes_per_capita']:.2f} | {c['daily_sessions_pct']:.2f} "
            f"| {c['daily_cume_pct']:.2f} [{c['daily_cume_lb_pct']:.2f}, {c['daily_cume_ub_pct']:.2f}] "
            f"| {c['time_spent_per_daily_viewer_min']:.0f} | {c['next_day_repeat_rate']:.2f} "
            f"| {c['rmax_pct']:.1f} | {c['k']:.1f} | {c['initial_slope']:.3f} "
            f"| {c['implied_same_cell_dup_index']:.1f} |")
    lines += ["", "Rmax % under alternative assumptions (m = sessions per daily viewer, rho = day-to-day correlation):", ""]
    keys = list(p["cells"].keys())
    lines.append("| m / rho | " + " | ".join(keys) + " |")
    lines.append("|---|" + "---|" * len(keys))
    for lab, row in p["rmax_sensitivity_grid"].items():
        lines.append(f"| {lab} | " + " | ".join(f"{row[k]:.1f}" for k in keys) + " |")
    return "\n".join(lines)


PARAMS_REPORT = "outputs/validation/reach_estimate_params.md"
POINTER = ("_Fitted parameter tables are written by `python -m optimizer.reach.fit` to "
           "`outputs/validation/reach_estimate_params.md` (git-ignored: derived from confidential "
           "Nielsen data). The committed doc holds only the method._")


def update_method_doc(p: dict, doc: Path = METHOD_DOC) -> None:
    """Write the fitted tables to outputs/validation/ and keep only a pointer in the committed doc."""
    report = _abs(PARAMS_REPORT)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# ESTIMATE reach parameters (auto-generated)\n\n" + params_table_md(p) + "\n",
                      encoding="utf-8")
    if not doc.exists():
        return
    txt = doc.read_text(encoding="utf-8")
    if BEGIN not in txt or END not in txt:
        return
    new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END),
                 BEGIN + "\n" + POINTER + "\n" + END, txt, flags=re.S)
    doc.write_text(new, encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="override reach.params_path")
    ap.add_argument("--no-doc", action="store_true", help="do not refresh ESTIMATE_METHOD.md")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    p = fit_estimate_params(cfg)
    out = _abs(args.out or cfg["reach"]["params_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")
    if not args.no_doc:
        update_method_doc(p)
    print(f"ESTIMATE reach params -> {out}  (form={p['form']}; {p['form_selection']['reason']})")
    print(params_table_md(p))
    return p


if __name__ == "__main__":
    main()
