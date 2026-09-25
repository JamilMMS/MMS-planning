"""Plan pipeline: scenarios S1-S4, constraint verification, KPIs, scenario comparison, frontier.

    python -m optimizer.plan.run --config config/plan_config.yaml [--scenarios S1 S3 ...] [--no-frontier]

Outputs (plan.out_dir, default outputs/plans/):
  <scenario>.csv            one row per spot
  <scenario>_kpis.json      KPIs, splits, constraint report, reach sensitivity, solver stats, run metadata
  scenario_comparison.csv   the scenarios of this run side by side
  frontier.csv              S1 across frontier.budget_levels and plan.frontier_trp_floors (x S3's TRP %)
  frontier/<point>.csv      the spot list behind every frontier point
Re-runnable: everything is re-derived from the config + processed data; deterministic.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import load_config
from ..reach.engine import ReachEngine
from .constraints import verify_constraints
from .kpis import compute_kpis, plan_frame, run_metadata, write_json
from .milp import solve_max_impressions
from .problem import Problem, abs_path, build_candidates, build_problem, limits_from_config, with_overrides
from .search import solve_reach

SCENARIOS = {  # name -> (id, kind)
    "max_reach_1plus": ("S1", "reach1"),
    "max_reach_3plus": ("S2", "reach3"),
    "max_impressions": ("S3", "milp"),
    "balanced": ("S4", "reach1"),
}
ID_TO_NAME = {v[0]: k for k, v in SCENARIOS.items()}


def _variant(prob: Problem, cfg_v: dict, min_impressions: float = 0.0) -> Problem:
    lim = limits_from_config(cfg_v, prob.channels, min_impressions=min_impressions)
    return dataclasses.replace(prob, cfg=cfg_v, lim=lim)


def _summary_row(sid: str, name: str, k: dict, report: dict, channels: list[str]) -> dict:
    row = {"scenario_id": sid, "scenario": name, "constraints": report["status"],
           "spend_usd": k["spend_usd"], "spend_aed": k["spend_aed"], "budget_utilisation": k["budget_utilisation"],
           "spots": k["spots"], "impressions": k["impressions"], "grp_pct": k["grp_pct"], "trp_pct": k["trp_pct"],
           "reach_1plus_abs": k["reach_1plus_abs"], "reach_1plus_pct": k["reach_1plus_pct"],
           "reach_3plus_abs": k["reach_3plus_abs"], "reach_3plus_pct": k["reach_3plus_pct"],
           "ots": k["ots"], "cpm": k["cpm"], "cpp": k["cpp"], "reach_method": k["reach_method"]}
    for c in channels:
        row[f"share_{c}"] = k["per_channel"].get(c, {}).get("share_of_spend", 0.0)
    return row


class Runner:
    def __init__(self, cfg: dict, *, forecast: pd.DataFrame | None = None, conflicts: pd.DataFrame | None = None,
                 engine=None, out_dir: str | Path | None = None, write: bool = True, log=print):
        self.cfg = cfg
        self.log = log
        self.write = write
        self.out = abs_path(out_dir or cfg["plan"]["out_dir"])
        slots, clog = build_candidates(cfg, forecast)
        self.prob = build_problem(cfg, conflicts=conflicts, slots=slots, cand_log=clog)
        self.conflicts = conflicts
        self.engine = engine if engine is not None else ReachEngine.from_config(cfg)
        uf = self.prob.unit_frame(np.arange(self.prob.n))
        uf["aud_abs"] = uf["aud"].astype(float)
        self.P = self.engine.prepare_many(uf)
        self.meta = run_metadata(cfg) if write else {}
        self.results: dict[str, dict] = {}
        self._s3: dict | None = None

    # ---- building blocks ---------------------------------------------------------------------
    def s3(self) -> dict:
        if self._s3 is None:
            t = time.perf_counter()
            r = solve_max_impressions(self.prob, self.cfg)
            if r["units"] is None:
                raise RuntimeError(f"S3 MILP found no feasible plan (status {r['status']}): constraints infeasible")
            r["impressions"] = float(self.prob.aud[r["units"]].sum())
            r["trp_pct"] = float(self.prob.trp[r["units"]].sum())
            r["sec"] = time.perf_counter() - t
            self._s3 = r
            self.log(f"S3 MILP: {r['status']} obj={r['objective']:.0f} gap={r['gap']:.2e} ({r['wall_sec']:.1f}s)")
        return self._s3

    def evaluate(self, prob: Problem, units: np.ndarray, cfg_v: dict, min_impressions: float = 0.0) -> dict:
        plan = plan_frame(prob, units)
        rep = verify_constraints(plan, prob.slots, cfg_v, conflicts=self.conflicts, min_impressions=min_impressions)
        k = compute_kpis(plan, self.engine, cfg_v)
        return {"plan": plan, "report": rep, "kpis": k}

    # ---- scenarios -----------------------------------------------------------------------------
    def scenario(self, name: str) -> dict:
        sid, kind = SCENARIOS[name]
        cfg = self.cfg
        t = time.perf_counter()
        min_imp = 0.0
        if kind == "milp":
            r = self.s3()
            units = r["units"]
            stats = {k: v for k, v in r.items() if k != "units"}
            prob = self.prob
        else:
            prob = self.prob
            start = None
            if name == "balanced":
                min_imp = float(cfg["objective"]["balanced_min_impressions_ratio"]) * self.s3()["impressions"]
                prob = _variant(self.prob, cfg, min_impressions=min_imp)
                start = self.s3()["units"]
            else:
                start = self.s3()["units"]     # multi-start: S3 optimum under the same constraints
            res = solve_reach(prob, self.engine, self.P, kind, cfg, start_units=start)
            units, stats = res.units, res.stats
        ev = self.evaluate(prob, units, cfg, min_imp)
        stats["total_sec"] = time.perf_counter() - t
        sens = self.engine.sensitivity(ev["plan"].assign(aud_abs=ev["plan"][cfg["plan"]["audience_column"]].astype(float)))
        out = {"scenario_id": sid, "scenario": name, "objective": kind,
               "recommended": name == cfg["objective"].get("recommended"),
               "min_impressions_floor": min_imp or None,
               "kpis": ev["kpis"], "constraint_report": ev["report"], "reach_sensitivity_k": sens,
               "solver": stats, "candidates": {k: (v if not isinstance(v, list) else {"n": len(v), "slot_ids": v})
                                                for k, v in self.prob.log.items()},
               "run_metadata": self.meta, "plan": ev["plan"]}
        self.results[name] = out
        k = ev["kpis"]
        self.log(f"{sid} {name}: {ev['report']['status']} spend=${k['spend_usd']:,.0f} spots={k['spots']} "
                 f"GRP={k['grp_pct']:.1f} R1+={k['reach_1plus_pct']:.2f}% R3+={k['reach_3plus_pct']:.2f}% "
                 f"({stats['total_sec']:.1f}s)")
        if self.write:
            self.out.mkdir(parents=True, exist_ok=True)
            ev["plan"].to_csv(self.out / f"{name}.csv", index=False)
            write_json(self.out / f"{name}_kpis.json", {k2: v for k2, v in out.items() if k2 != "plan"})
        return out

    def comparison(self) -> pd.DataFrame:
        rows = [_summary_row(r["scenario_id"], n, r["kpis"], r["constraint_report"], self.prob.channels)
                for n, r in sorted(self.results.items(), key=lambda kv: kv[1]["scenario_id"])]
        df = pd.DataFrame(rows)
        if self.write and len(df):
            df.to_csv(self.out / "scenario_comparison.csv", index=False)
        return df

    # ---- frontier ------------------------------------------------------------------------------
    def frontier(self) -> pd.DataFrame:
        cfg = self.cfg
        base_total = float(cfg["budget"]["total_usd"])
        rows = []
        s1 = self.results.get("max_reach_1plus")
        s3 = self.s3()
        points = [("budget", float(l), with_overrides(cfg, budget_total_usd=float(l) * base_total), s3["units"])
                  for l in cfg["frontier"]["budget_levels"]]
        points += [("trp_floor", float(f), with_overrides(cfg, min_trp_pct=float(f) * s3["trp_pct"]), s3["units"])
                   for f in cfg["plan"]["frontier_trp_floors"]]
        fdir = self.out / "frontier"
        for kind, level, cfg_v, start in points:
            t = time.perf_counter()
            floor_v = float(cfg_v["constraints"].get("min_trp_pct", 0) or 0)
            floor_0 = float(cfg["constraints"].get("min_trp_pct", 0) or 0)
            # S1's plan is feasible for a same-budget floor it already meets, and the floored
            # optimum can only be lower -> the S1 plan is the answer (floor not binding)
            same_as_s1 = (s1 is not None and cfg_v["budget"]["total_usd"] == base_total and floor_v >= floor_0
                          and s1["kpis"]["trp_pct"] >= floor_v * (1 - 1e-9))
            if same_as_s1:
                plan = s1["plan"]
                ev = {"plan": plan, "kpis": s1["kpis"],
                      "report": verify_constraints(plan, self.prob.slots, cfg_v, conflicts=self.conflicts)}
                st = s1["solver"]
            else:
                prob = _variant(self.prob, cfg_v)
                st_start = start if cfg_v["budget"]["total_usd"] == base_total else None
                res = solve_reach(prob, self.engine, self.P, "reach1", cfg_v, start_units=st_start)
                ev = self.evaluate(prob, res.units, cfg_v)
                st = res.stats
            k, rep = ev["kpis"], ev["report"]
            pid = f"{kind}_{level:g}"
            rows.append({"frontier": kind, "level": level, "point_id": pid,
                         "budget_usd": float(cfg_v["budget"]["total_usd"]),
                         "min_trp_pct": float(cfg_v["constraints"].get("min_trp_pct", 0) or 0),
                         "constraints": rep["status"], "failed": ";".join(rep["failed"]),
                         "spend_usd": k["spend_usd"], "spots": k["spots"], "impressions": k["impressions"],
                         "grp_pct": k["grp_pct"], "trp_pct": k["trp_pct"],
                         "reach_1plus_pct": k["reach_1plus_pct"], "reach_3plus_pct": k["reach_3plus_pct"],
                         "ots": k["ots"], "cpm": k["cpm"], "cpp": k["cpp"], "reach_method": k["reach_method"],
                         "greedy_objective": st.get("greedy_objective"), "lambda": st.get("lambda"),
                         "chosen_start": st.get("chosen_start"), "reused_s1": same_as_s1,
                         "floor_binding": (not same_as_s1) if kind == "trp_floor" else None,
                         "sec": time.perf_counter() - t})
            self.log(f"frontier {pid}: {rep['status']} spend=${k['spend_usd']:,.0f} GRP={k['grp_pct']:.1f} "
                     f"R1+={k['reach_1plus_pct']:.2f}%")
            if self.write:
                fdir.mkdir(parents=True, exist_ok=True)
                ev["plan"].to_csv(fdir / f"{pid}.csv", index=False)
        df = pd.DataFrame(rows)
        if self.write:
            df.insert(0, "config_sha256", self.meta.get("config_sha256"))
            df.insert(1, "slot_forecast_sha256", self.meta.get("slot_forecast_sha256"))
            df.insert(2, "forecast_provisional", self.meta.get("forecast_provisional"))
            df.to_csv(self.out / "frontier.csv", index=False)
        return df


def update_assumptions_log(cfg: dict, runner: Runner) -> None:
    path = abs_path("outputs/assumptions_log.md")
    if not path.exists():
        return
    pc = cfg["plan"]
    today = dt.date.today().isoformat()
    rows = [
        ("plan:weeks", "—", f"Weekly phasing: flight weeks = 7-day blocks from flight.start; each week's spend within "
         f"[{cfg['flight']['weekly_share_min']}, {cfg['flight']['weekly_share_max']}] x the plan's {pc['weekly_share_basis']}",
         "plan.weekly_share_basis", "config comment says '% of spend'"),
        ("plan:channels", "#7", f"Channel min/max shares are shares of the {pc['channel_share_basis']} (budget.total_usd)",
         "plan.channel_share_basis", "every channel >= its minimum"),
        ("plan:progcap", "—", f"max_spots_per_program_per_day counts (title_en, broadcast day) per {pc['program_cap_scope']} "
         "(generic strands such as MOVIE air on several channels)", "plan.program_cap_scope", "—"),
        ("plan:s3", "—", "S3 MILP objective uses the forecast audience rounded to whole persons (reported impressions use full precision)",
         "plan.milp_*", "< 0.5 person per spot"),
        ("plan:reach", "#8", f"Scenario reach figures are {str(cfg['reach']['mode']).upper()} (model); plans optimised on the p50 forecast",
         "reach.mode, plan.audience_column", "sensitivity k ±" + str(cfg["reach"]["estimate_sensitivity_k"])),
    ]
    if runner.meta.get("forecast_provisional"):
        rows.append(("plan:provisional", "—", "Plans built on the PROVISIONAL forecast (pre Gate 2); re-run after the forecast is final",
                     "—", "run_metadata.forecast_provisional=true"))
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if "[plan:" not in l]
    while lines and not lines[-1].strip():
        lines.pop()
    for tag, issue, text, key, impact in rows:
        lines.append(f"| {today} | 6 | {issue} | {text} [{tag}] | {key} | {impact} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: dict, scenarios: list[str] | None = None, frontier: bool = True, out_dir: str | None = None) -> Runner:
    names = scenarios or list(cfg["objective"]["run_scenarios"])
    names = [ID_TO_NAME.get(n.upper(), n) if n.upper() in ID_TO_NAME else n for n in names]
    for n in names:
        if n not in SCENARIOS:
            raise SystemExit(f"unknown scenario {n!r}; choose from {list(SCENARIOS)} or {list(ID_TO_NAME)}")
    order = sorted(names, key=lambda n: {"max_impressions": 0}.get(n, 1))   # S3 first (S4 / floors use it)
    r = Runner(cfg, out_dir=out_dir)
    for n in order:
        r.scenario(n)
    r.comparison()
    if frontier:
        if "max_reach_1plus" not in r.results:
            r.log("frontier: S1 not in --scenarios; frontier points are solved from scratch")
        r.frontier()
    update_assumptions_log(cfg, r)
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scenarios", nargs="+", default=None, help="names or ids (S1..S4); default objective.run_scenarios")
    ap.add_argument("--no-frontier", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    r = run(cfg, a.scenarios, not a.no_frontier, a.out_dir)
    bad = [n for n, x in r.results.items() if x["constraint_report"]["status"] != "PASS"]
    if bad:
        print(f"CONSTRAINT FAILURES: {bad}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
