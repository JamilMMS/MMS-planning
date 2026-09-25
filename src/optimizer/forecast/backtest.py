"""Gate 3 back-test of the per-slot audience forecast (METHODOLOGY 2.4).

    python -m optimizer.forecast.backtest --config config/plan_config.yaml
           [--fit 2026-09-01 2026-09-14] [--holdout 2026-09-15 2026-09-21]

Fit on ``forecast.backtest.fit`` breaks, predict ``forecast.backtest.holdout`` (defaults Sep 1–14
-> Sep 15–21; when August lands pass --fit 2026-08-01 2026-08-31 --holdout 2026-09-01 2026-09-30).
Event breaks (``is_event``) are excluded from BOTH sides.

Hold-out "grid-like" slots
--------------------------
For every channel x broadcast date, consecutive breaks of the same eTAM programme form one
*block* (a programme airing). Its window is the block's break-start span, floored / ceiled to
``backtest.window_round_min`` (15 min, the grid's granularity): ``[floor(first/15)*15,
ceil((last+1)/15)*15)``. ``actual`` = mean rating_abs of the block's breaks (= expected audience
of one spot in that airing). Run type = rerun if most breaks are reruns. The block is forecast
with the SAME code as October (optimizer.forecast.core) from fit-window breaks only: program
level via the exact eTAM title on the same channel (if it aired in the fit window), otherwise
slot level (new programme), then channel x daypart; low-sample channels pooled.

Tier mapping (hold-out slot -> October grid tier and rate)
----------------------------------------------------------
The October grid's in-flight, non-synthetic rows on the same channel and weekday whose
``[start_min, end_min)`` contains the block's first break minute; tier = most frequent tier among
them, rate = median rate_usd of those rows with that tier. No covering row -> nearest grid row
of that channel/weekday by time distance (``tier_match='nearest'``).

Mock plans (plan-level impressions error = sum(p50) / sum(actual) - 1)
----------------------------------------------------------------------
* ``mock`` — one spot in every hold-out slot mapped to Access / Prime / Special, plus a
  random ``backtest.regular_sample_share`` of Regular slots (seed ``forecast.random_seed``);
  reported count-weighted and cost-weighted (sum rate*p50 / sum rate*actual - 1).
* ``selection`` — winner's-curse check: the cheapest predicted-CPM slots up to
  ``backtest.selection_budget_share`` of the total hold-out cost (what an optimiser would pick).

Settings compared: full factorial of ``backtest.alternatives`` (weekday grouping x estimator x
time tolerance); best = smallest |mock count-weighted plan error|, tie-break (within 0.5 pt)
WMAPE. The config file is never modified — the report states the recommended change.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from optimizer.config import load_config, project_root
from optimizer.forecast.core import ForecastParams, forecast_targets, is_live_special
from optimizer.forecast.run import _p, load_history, low_sample_channels

TIER_ORDER = ["Special", "Prime", "Access", "Regular"]


# --------------------------------------------------------------------------- hold-out slots
def build_holdout_slots(hold: pd.DataFrame, round_min: int = 15) -> pd.DataFrame:
    h = hold.sort_values(["channel", "broadcast_date", "start_sec"]).reset_index(drop=True)
    new_block = (h["program"] != h["program"].shift()) | (h["channel"] != h["channel"].shift()) | (
        h["broadcast_date"] != h["broadcast_date"].shift())
    h["block"] = new_block.cumsum()
    h["minute"] = h["start_sec"] // 60
    g = h.groupby("block").agg(channel=("channel", "first"), broadcast_date=("broadcast_date", "first"),
                               weekday=("weekday", "first"), program=("program", "first"),
                               first_min=("minute", "min"), last_min=("minute", "max"), n_breaks=("rating_abs", "size"),
                               actual=("rating_abs", "mean"), rerun_share=("first_run", lambda s: 1 - s.mean()))
    g["start_min"] = (g["first_min"] // round_min) * round_min
    g["end_min"] = -(-(g["last_min"] + 1) // round_min) * round_min
    g["is_rerun"] = g["rerun_share"] > 0.5
    g["target_id"] = g["channel"] + "|" + g["broadcast_date"].dt.strftime("%Y-%m-%d") + "|" + g["first_min"].astype(str)
    return g.reset_index(drop=True)


def map_tiers(slots: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    gr = grid[grid["in_flight"] & ~grid["is_synthetic"]]
    tiers, rates, how = [], [], []
    by = {k: d for k, d in gr.groupby(["channel", "weekday"])}
    for r in slots.itertuples():
        d = by.get((r.channel, r.weekday))
        if d is None or d.empty:
            tiers.append("Regular"); rates.append(np.nan); how.append("none"); continue
        cov = d[(d["start_min"] <= r.first_min) & (d["end_min"] > r.first_min)]
        kind = "cover"
        if cov.empty:
            dist = np.minimum((d["start_min"] - r.first_min).abs(), (d["end_min"] - 1 - r.first_min).abs())
            cov = d[dist == dist.min()]
            kind = "nearest"
        vc = cov["tier"].value_counts()
        top = vc[vc == vc.max()].index
        tier = min(top, key=TIER_ORDER.index)
        tiers.append(tier); rates.append(float(cov.loc[cov["tier"] == tier, "rate_usd"].median())); how.append(kind)
    out = slots.copy()
    out["tier"], out["rate_usd"], out["tier_match"] = tiers, rates, how
    return out


def holdout_targets(slots: pd.DataFrame, fit: pd.DataFrame, p: ForecastParams, low_sample: set[str]) -> pd.DataFrame:
    seen = set(zip(fit["channel"], fit["program"]))
    t = slots.copy()
    t["low_sample"] = t["channel"].isin(low_sample)
    t["live_special"] = [is_live_special(x, True, p.live_special_patterns) for x in t["program"]]
    t["is_new"] = [(c, pr) not in seen for c, pr in zip(t["channel"], t["program"])]
    t["use_program"] = ~t["is_new"] & ~t["low_sample"] & ~t["live_special"]
    t["etam_title"] = np.where(t["use_program"], t["program"], "")
    t["prefer_rerun_pool"] = False
    return t


# --------------------------------------------------------------------------- metrics
def err_stats(d: pd.DataFrame) -> dict[str, float]:
    a, pr = d["actual"].to_numpy(), d["p50"].to_numpy()
    nz = a > 0
    return {
        "n": len(d), "n_zero_actual": int((~nz).sum()),
        "MAPE": float(np.mean(np.abs(pr[nz] - a[nz]) / a[nz])) if nz.any() else np.nan,
        "WMAPE": float(np.abs(pr - a).sum() / a.sum()) if a.sum() > 0 else np.nan,
        "bias": float((pr - a).sum() / a.sum()) if a.sum() > 0 else np.nan,
        "cover_p10_p90": float(((a >= d["p10"]) & (a <= d["p90"])).mean()) if "p10" in d else np.nan,
        "actual_mean": float(a.mean()), "pred_mean": float(pr.mean()),
    }


def plan_errors(d: pd.DataFrame, p: ForecastParams, reg_share: float, sel_share: float) -> dict[str, Any]:
    rng = np.random.default_rng(p.random_seed)
    prem = d["tier"].isin(["Access", "Prime", "Special"]).to_numpy()
    reg = ~prem & (rng.random(len(d)) < reg_share)
    mock = d[prem | reg]
    rate = mock["rate_usd"].fillna(mock["rate_usd"].median())
    out = {
        "mock_n": len(mock), "mock_cost": float(rate.sum()),
        "mock_pred": float(mock["p50"].sum()), "mock_actual": float(mock["actual"].sum()),
        "mock_err": float(mock["p50"].sum() / mock["actual"].sum() - 1),
        "mock_cost_err": float((rate * mock["p50"]).sum() / (rate * mock["actual"]).sum() - 1),
    }
    s = d[d["p50"] > 0].assign(cpm=lambda x: x["rate_usd"] / x["p50"]).sort_values(["cpm", "target_id"])
    s = s[s["rate_usd"].cumsum() <= sel_share * d["rate_usd"].sum()]
    out.update(sel_n=len(s), sel_pred=float(s["p50"].sum()), sel_actual=float(s["actual"].sum()),
               sel_err=float(s["p50"].sum() / s["actual"].sum() - 1) if len(s) else np.nan)
    per_ch = {c: float(g["p50"].sum() / g["actual"].sum() - 1) if g["actual"].sum() > 0 else np.nan
              for c, g in mock.groupby("channel")}
    out["mock_err_by_channel"] = per_ch
    return out


def interval_calibration(d: pd.DataFrame, p: ForecastParams) -> tuple[str, dict[str, float]]:
    """Coverage of the raw (unwidened) and widened p10-p90 intervals per group, and the
    multiplicative widening w that would give nominal coverage (q_hi - q_lo) on the back-test."""
    nominal = p.quantiles[-1] - p.quantiles[0]
    grp = np.select([d["low_sample"], d["live_special"], d["is_new"]], ["low_sample", "live_special", "new_program"],
                    "level=" + d["forecast_level"].astype(str))
    rows = [f"Nominal coverage {nominal:.0%}. w = factor on the raw interval half-widths around p50 that reaches nominal on this hold-out week.\n",
            "| group | n | raw coverage | widening in config | widened coverage | w for nominal |", "|---|---:|---:|---:|---:|---:|"]
    need_by: dict[str, float] = {}
    for k in sorted(set(grp)):
        g = d[grp == k]
        a = g["actual"].to_numpy()
        lo_r, hi_r, m = g["p10_raw"].to_numpy(), g["p90_raw"].to_numpy(), g["p50"].to_numpy()
        cov = lambda w: float(((a >= np.maximum(0, m - w * (m - lo_r))) & (a <= m + w * (hi_r - m))).mean())
        need = next((w for w in np.arange(0.5, 5.01, 0.05) if cov(w) >= nominal), np.nan)
        need_by[k] = float(need)
        rows.append(f"| {k} | {len(g)} | {cov(1.0):.1%} | {g['widening'].iloc[0]:.2f} | "
                    f"{float(((a >= g['p10']) & (a <= g['p90'])).mean()):.1%} | {need:.2f} |")
    return "\n".join(rows), need_by


def recommended_widening(need_by: dict[str, float], p: ForecastParams) -> dict[str, float]:
    """base = w needed by the unflagged program-level slots (the bulk); each flag factor = its own
    need / base (>= 1). live_special has no hold-out evidence -> kept."""
    base = need_by.get("level=program", float(p.interval_widening.get("base", 1.0)))
    if not np.isfinite(base):
        base = float(p.interval_widening.get("base", 1.0))
    rec = {"base": round(base, 2)}
    for f in ("new_program", "low_sample"):
        v = need_by.get(f, np.nan)
        rec[f] = round(max(1.0, v / base), 2) if np.isfinite(v) else p.interval_widening.get(f, 1.0)
    rec["live_special"] = p.interval_widening.get("live_special", 1.0)
    return rec


# --------------------------------------------------------------------------- driver
def run_setting(slots: pd.DataFrame, fit: pd.DataFrame, p: ForecastParams, low_sample: set[str]) -> pd.DataFrame:
    t = holdout_targets(slots, fit, p, low_sample)
    fc, _ = forecast_targets(t, fit, p, with_intervals=True)
    return t.merge(fc, on="target_id", validate="one_to_one")


def run_backtest(cfg: dict[str, Any], fit_win=None, hold_win=None, root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    bt = cfg["forecast"]["backtest"]
    fit_win = fit_win or bt["fit"]
    hold_win = hold_win or bt["holdout"]
    fit = load_history(cfg, root, *fit_win)
    hold = load_history(cfg, root, *hold_win)
    grid = pd.read_parquet(_p(root, cfg["forecast"]["paths"]["grid"]))
    low = low_sample_channels(root, cfg)
    slots = map_tiers(build_holdout_slots(hold, int(bt["window_round_min"])), grid)
    alts = bt["alternatives"]
    base = ForecastParams.from_cfg(cfg)
    results = []
    cfg_wg_name = next((n for n, g in alts["weekday_groups"].items() if g == base.weekday_groups), "config")
    ls_opts = alts.get("low_sample_estimator", [base.low_sample_estimator])
    for (wg_name, wg), est, tol, ls_est in itertools.product(alts["weekday_groups"].items(), alts["estimator"],
                                                             alts["time_tolerance_min"], ls_opts):
        p = ForecastParams.from_cfg(cfg, weekday_groups=wg, estimator=est, time_tolerance_min=int(tol),
                                    low_sample_estimator=ls_est)
        d = run_setting(slots, fit, p, low)
        st = err_stats(d)
        pe = plan_errors(d, p, float(bt["regular_sample_share"]), float(bt["selection_budget_share"]))
        is_cfg = (wg == base.weekday_groups and est == base.estimator and int(tol) == base.time_tolerance_min
                  and ls_est == base.low_sample_estimator)
        results.append({"weekday_groups": wg_name, "estimator": est, "time_tolerance_min": int(tol),
                        "low_sample_estimator": ls_est or "inherit",
                        "is_current_config": is_cfg, **st, **{k: v for k, v in pe.items() if k != "mock_err_by_channel"},
                        "_d": d, "_pe": pe, "_p": p})
    cmp_ = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    cmp_["abs_plan_err"] = cmp_["mock_err"].abs()
    best_abs = cmp_["abs_plan_err"].min()
    cand = cmp_[cmp_["abs_plan_err"] <= best_abs + float(bt.get("tie_band", 0.0)) + 1e-12]
    best_i = int(cand.sort_values(["WMAPE", "abs_plan_err"]).index[0])
    return {"results": results, "comparison": cmp_, "best": best_i, "slots": slots, "fit_win": fit_win,
            "hold_win": hold_win, "cfg_wg_name": cfg_wg_name, "n_fit": len(fit), "n_hold": len(hold)}


# --------------------------------------------------------------------------- report
def _fmt_pct(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.1f}%"


def _fmt_p(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:.1f}%"


def _table(df: pd.DataFrame, by: str) -> str:
    rows = ["| " + by + " | n slots | zero actual | MAPE (actual>0) | WMAPE | bias | p10–p90 coverage | mean actual | mean p50 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    groups = list(df.groupby(by, sort=True)) + [("**All**", df)]
    for k, g in groups:
        s = err_stats(g)
        rows.append(f"| {k} | {s['n']} | {s['n_zero_actual']} | {_fmt_p(s['MAPE'])} | {_fmt_p(s['WMAPE'])} | "
                    f"{_fmt_pct(s['bias'])} | {_fmt_p(s['cover_p10_p90'])} | {s['actual_mean']:,.0f} | {s['pred_mean']:,.0f} |")
    return "\n".join(rows)


def write_report(res: dict[str, Any], cfg: dict[str, Any], path: Path) -> str:
    cmp_ = res["comparison"]
    best = res["results"][res["best"]]
    d, pe, p = best["_d"], best["_pe"], best["_p"]
    cur = next(r for r in res["results"] if r["is_current_config"]) if any(r["is_current_config"] for r in res["results"]) else None
    bt = cfg["forecast"]["backtest"]
    target = float(bt["target_abs_error"])
    L = []
    L.append("# Back-test report — per-slot audience forecast (Gate 3)\n")
    L.append(f"Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by `python -m optimizer.forecast.backtest`. "
             f"Target: {cfg['target']['buying_target']}; metric = per-spot audience `rating_abs` "
             "(= TRP_Absolute x 60 / break_sec, never TRP Absolute itself).\n")
    L.append(f"- **Fit**: {res['fit_win'][0]} .. {res['fit_win'][1]} ({res['n_fit']:,} non-event breaks). "
             f"**Hold-out**: {res['hold_win'][0]} .. {res['hold_win'][1]} ({res['n_hold']:,} non-event breaks). "
             "Event breaks (`is_event`, incl. the 19 Sep MBC ACTION window and MBC 1 match breaks) excluded from both sides.")
    L.append(f"- Hold-out slots: {len(d):,} programme blocks (channel x date x consecutive breaks of one programme); "
             f"window = break span rounded to {bt['window_round_min']} min; actual = mean rating_abs of the block's breaks.")
    tm = res["slots"]["tier_match"].value_counts().to_dict()
    L.append(f"- Tier mapping: grid rows (in-flight, non-synthetic) on the same channel + weekday covering the block's first break "
             f"minute; most frequent tier, median rate. covered={tm.get('cover', 0)}, nearest-row fallback={tm.get('nearest', 0)}.")
    L.append(f"- Chosen setting: **weekday_groups={best['weekday_groups']}, estimator={best['estimator']}, "
             f"time_tolerance_min={best['time_tolerance_min']}, low_sample_estimator={best['low_sample_estimator']}**, "
             f"min_breaks={p.min_breaks}, shrinkage_k={p.shrinkage_k}, low_sample_policy={p.low_sample_policy}, "
             f"bootstrap_n={p.bootstrap_n}, interval_widening={p.interval_widening}.\n")

    ok = abs(pe["mock_err"]) <= target
    L.append("## Headline — plan-level total impressions error (chosen setting)\n")
    L.append("| Plan | spots | cost USD | predicted impressions | actual impressions | error |")
    L.append("|---|---:|---:|---:|---:|---:|")
    L.append(f"| Mock plan, count-weighted (all Access/Prime/Special + {bt['regular_sample_share']:.0%} of Regular) | {pe['mock_n']} | "
             f"{pe['mock_cost']:,.0f} | {pe['mock_pred']:,.0f} | {pe['mock_actual']:,.0f} | **{_fmt_pct(pe['mock_err'])}** |")
    L.append(f"| Same mock plan, cost-weighted (sum rate x aud) | {pe['mock_n']} | {pe['mock_cost']:,.0f} | | | **{_fmt_pct(pe['mock_cost_err'])}** |")
    L.append(f"| Selection plan (cheapest predicted CPM up to {bt['selection_budget_share']:.0%} of cost; winner's-curse check) | "
             f"{pe['sel_n']} | | {pe['sel_pred']:,.0f} | {pe['sel_actual']:,.0f} | {_fmt_pct(pe['sel_err'])} |")
    L.append(f"\nGate 3 target |error| <= {target:.0%} on the mock plan: **{'MET' if ok else 'NOT MET'}**.\n")
    L.append("Mock-plan error by channel (count-weighted): " + ", ".join(
        f"{c} {_fmt_pct(v)}" for c, v in sorted(pe["mock_err_by_channel"].items())) + "\n")

    L.append("## Slot-level accuracy by channel (chosen setting)\n")
    L.append(_table(d, "channel") + "\n")
    L.append("## Slot-level accuracy by grid tier (chosen setting)\n")
    L.append(_table(d, "tier") + "\n")
    L.append("## Channel x tier\n")
    d2 = d.assign(channel_tier=d["channel"] + " / " + d["tier"])
    L.append(_table(d2, "channel_tier") + "\n")
    L.append("## By forecast level and flags\n")
    L.append(_table(d, "forecast_level") + "\n")
    d3 = d.assign(flag=np.select([d["low_sample"], d["is_new"], d["live_special"]], ["low_sample", "new_program", "live_special"], "none"))
    L.append(_table(d3, "flag") + "\n")
    L.append("MAPE is over slots with actual > 0 only (zero-rated blocks are common on MBC ACTION / MBC MAX: "
             "a 1-panelist change moves them from 0 to thousands); WMAPE = sum|p50-actual| / sum actual; "
             "bias = sum(p50-actual) / sum actual; coverage = share of actuals inside [p10, p90].\n")

    band = float(bt.get("tie_band", 0.0))
    L.append(f"## Settings comparison (full factorial; best = smallest |mock plan error|, settings within {band * 100:.1f} pt of it are tied -> lowest WMAPE)\n")
    L.append("| weekday groups | estimator | tol min | low-sample estimator | current cfg | mock plan err | cost-wtd err | selection err | WMAPE | MAPE | bias | coverage | ACTION+MAX bias |")
    L.append("|---|---|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in cmp_.iterrows():
        mark = " **(chosen)**" if i == res["best"] else ""
        dd = res["results"][i]["_d"]
        lsb = err_stats(dd[dd["low_sample"]])["bias"] if dd["low_sample"].any() else np.nan
        L.append(f"| {r['weekday_groups']}{mark} | {r['estimator']} | {r['time_tolerance_min']} | {r['low_sample_estimator']} | {'yes' if r['is_current_config'] else ''} | "
                 f"{_fmt_pct(r['mock_err'])} | {_fmt_pct(r['mock_cost_err'])} | {_fmt_pct(r['sel_err'])} | {_fmt_p(r['WMAPE'])} | "
                 f"{_fmt_p(r['MAPE'])} | {_fmt_pct(r['bias'])} | {_fmt_p(r['cover_p10_p90'])} | {_fmt_pct(lsb)} |")
    spread = cmp_["mock_err"].max() - cmp_["mock_err"].min()
    L.append(f"\nSpread of the mock-plan error across all {len(cmp_)} settings: {spread * 100:.1f} pt "
             f"(WMAPE {cmp_['WMAPE'].min() * 100:.1f}–{cmp_['WMAPE'].max() * 100:.1f}%). With one hold-out week, differences "
             "below ~1 pt are not meaningful; the median estimator is also degenerate on zero-inflated channels (p50 = 0 -> infinite CPM), "
             "which the impressions error barely registers because ACTION + MAX are ~1% of impressions.")
    wg = cfg["forecast"]["backtest"]["alternatives"]["weekday_groups"]
    L.append("\nweekday groups: " + "; ".join(f"`{k}` = {v}" for k, v in wg.items()) + "\n")
    _, _need0 = interval_calibration(d, p)
    rec_w = recommended_widening(_need0, p)
    L.append("## Recommendation for config/plan_config.yaml (NOT applied — lead to decide)\n")
    changes = []
    if cur is None or cur["weekday_groups"] != best["weekday_groups"]:
        changes.append(f"`forecast.weekday_groups: {wg[best['weekday_groups']]}`")
    if cur is None or cur["estimator"] != best["estimator"]:
        changes.append(f"`forecast.estimator: {best['estimator']}`")
    if cur is None or cur["time_tolerance_min"] != best["time_tolerance_min"]:
        changes.append(f"`forecast.time_tolerance_min: {best['time_tolerance_min']}`")
    if cur is None or cur["low_sample_estimator"] != best["low_sample_estimator"]:
        v = best["low_sample_estimator"]
        changes.append(f"`forecast.low_sample_estimator: {'null' if v == 'inherit' else v}`")
    if changes:
        L.append("Change " + ", ".join(changes) + ". Current config scored mock plan error "
                 f"{_fmt_pct(cur['mock_err']) if cur else 'n/a'}, WMAPE {_fmt_p(cur['WMAPE']) if cur else 'n/a'} "
                 f"vs chosen {_fmt_pct(best['mock_err'])}, {_fmt_p(best['WMAPE'])}.\n")
    else:
        L.append("No change: the current config is the best setting.\n")
    L.append(f"Also set `forecast.interval_widening: {rec_w}` (see Interval calibration) so p10–p90 has ~nominal coverage.\n")
    L.append("## Interval calibration (chosen setting)\n")
    cal_text, _ = interval_calibration(d, p)
    L.append(cal_text + "\n")
    L.append(f"Recommended `forecast.interval_widening`: {rec_w} (w = base x flag factor; live_special has no hold-out "
             "evidence and is kept). Raw intervals capture day-to-day noise inside the fit window but not the week-to-week "
             "level drift of the hold-out week, hence base > 1.\n")
    L.append("## Caveats\n")
    L.append("- One hold-out week (7 days) — the error estimates themselves are noisy; repeat with August -> September when August lands "
             "(`--fit 2026-08-01 2026-08-31 --holdout 2026-09-01 2026-09-30`).")
    L.append("- Hold-out 'new programmes' are titles absent from Sep 1–14 on that channel (mostly rotating movies on MBC 2 / MAX / BOLLYWOOD); "
             "October's truly new series may behave differently (premieres).")
    L.append("- Hold-out windows are built from actual break spans, so the slot level sees exactly the breaks of that airing's window; "
             "October grid windows are the scheduled programme windows (similar, but not identical).")
    L.append("- Only 2 fit days per weekday: with `each_day` grouping the slot level often has < min_breaks and falls back.\n")
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--fit", nargs=2, default=None)
    ap.add_argument("--holdout", nargs=2, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    root = project_root()
    res = run_backtest(cfg, a.fit, a.holdout, root)
    write_report(res, cfg, _p(root, cfg["forecast"]["paths"]["backtest_report"]))
    best = res["results"][res["best"]]
    cols = ["target_id", "channel", "broadcast_date", "weekday", "program", "start_min", "end_min", "n_breaks", "tier",
            "rate_usd", "tier_match", "is_rerun", "is_new", "low_sample", "live_special", "forecast_level", "evidence_n",
            "actual", "p10", "p50", "p90"]
    best["_d"][cols].to_csv(_p(root, cfg["forecast"]["paths"]["backtest_slots"]), index=False)
    print(res["comparison"][["weekday_groups", "estimator", "time_tolerance_min", "low_sample_estimator", "is_current_config", "mock_err",
                             "mock_cost_err", "sel_err", "WMAPE", "MAPE", "bias", "cover_p10_p90"]].round(4).to_string())
    print("best:", best["weekday_groups"], best["estimator"], best["time_tolerance_min"], best["low_sample_estimator"])


if __name__ == "__main__":
    main()
