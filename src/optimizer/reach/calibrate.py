"""CALIBRATED mode: fit the per-channel curve parameters (Rmax_c, k_c) to eTAM R&F outputs
for test schedules (DATA_SPEC C.3 option B), optionally with a channel duplication matrix
replacing Sainsbury.

    python -m optimizer.reach.calibrate --config config/plan_config.yaml
    python -m optimizer.reach.calibrate --self-test      # synthetic recovery check

Inputs (paths in ``reach.calibration``):
* schedules file: schedule_id, grp_pct, reach_1plus_pct [, reach_2plus_pct .. reach_5plus_pct]
  (eTAM Cume Reach RF / Reach N+ of each test schedule, % of universe);
* spots file: schedule_id, channel, aud_abs [, air_date, start_min] (one row per spot of the
  test schedule; aud_abs = the spot's GRP Absolute from eTAM);
* optional duplication matrix: channel_i, channel_j, dup_reach_abs, reach_i_abs, reach_j_abs
  (eTAM Duplication Cume Reach of the pair and each channel's cume reach) ->
  duplication index phi_ij = dup * U / (reach_i * reach_j).

Fit: least squares on reach N+ (levels ``fit_reach_levels``) in points, on a seeded
``train_share`` split of schedules; the rest validates (MAE / RMSE / max error in points).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..config import load_config
from . import curves
from .common import abs_path

SCHEDULE_COLUMNS = ["schedule_id", "grp_pct", "reach_1plus_pct"]
SPOT_COLUMNS = ["schedule_id", "channel", "aud_abs"]
DUP_COLUMNS = ["channel_i", "channel_j", "dup_reach_abs", "reach_i_abs", "reach_j_abs"]


class CalibrationFileError(ValueError):
    pass


def _read(path: str | Path | pd.DataFrame, required: list[str], what: str) -> pd.DataFrame:
    if isinstance(path, pd.DataFrame):
        df, src = path.copy(), "<DataFrame>"
    else:
        p = abs_path(path)
        src = str(p)
        if not p.exists():
            raise FileNotFoundError(f"{what} file not found: {p}")
        df = pd.read_parquet(p) if p.suffix.lower() in (".parquet", ".pq") else pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise CalibrationFileError(f"{src}: {what} file missing column(s) {miss}; required {required}")
    return df


def load_rf_schedules(schedules: Any, spots: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the R&F test-schedule table and its spot list; validate keys."""
    s = _read(schedules, SCHEDULE_COLUMNS, "R&F schedules")
    sp = _read(spots, SPOT_COLUMNS, "R&F schedule spots")
    unknown = set(sp["schedule_id"]) - set(s["schedule_id"])
    if unknown:
        raise CalibrationFileError(f"spots reference unknown schedule_id(s) {sorted(unknown)[:5]}")
    empty = set(s["schedule_id"]) - set(sp["schedule_id"])
    if empty:
        raise CalibrationFileError(f"schedule(s) without spots: {sorted(empty)[:5]}")
    return s, sp


def load_duplication_matrix(path: Any, cells: list[str], universe: float) -> np.ndarray:
    """phi[i, j] = Duplication Cume Reach_ij x U / (Cume_i x Cume_j); missing pairs -> 1."""
    d = _read(path, DUP_COLUMNS, "duplication matrix")
    idx = {c: i for i, c in enumerate(cells)}
    phi = np.ones((len(cells), len(cells)))
    for r in d.itertuples(index=False):
        if r.channel_i in idx and r.channel_j in idx and r.channel_i != r.channel_j:
            v = float(r.dup_reach_abs) * universe / (float(r.reach_i_abs) * float(r.reach_j_abs))
            phi[idx[r.channel_i], idx[r.channel_j]] = phi[idx[r.channel_j], idx[r.channel_i]] = v
    return phi


def summarise(spots: pd.DataFrame, cells: list[str], universe: float) -> dict[Any, tuple[np.ndarray, np.ndarray]]:
    """Per schedule: (GRP % per cell, spot count per cell). Cells = channels."""
    idx = {c: i for i, c in enumerate(cells)}
    out = {}
    for sid, g in spots.groupby("schedule_id", sort=False):
        gv = np.zeros(len(cells))
        nv = np.zeros(len(cells), dtype=np.int64)
        for ch, a in zip(g["channel"], g["aud_abs"].astype(float)):
            gv[idx[ch]] += a / universe * 100.0
            nv[idx[ch]] += 1
        out[sid] = (gv, nv)
    return out


def predict(gv, nv, rmax, k, form, phi, kmax, levels) -> np.ndarray:
    _, nplus, _ = curves.model_reach(gv, nv, rmax, k, form, phi, kmax)
    return np.array([nplus[n - 1] * 100.0 for n in levels])


def _observed(row: pd.Series, levels: list[int]) -> np.ndarray:
    return np.array([float(row[f"reach_{n}plus_pct"]) for n in levels])


def fit_calibrated(cfg: dict, schedules: pd.DataFrame, spots: pd.DataFrame,
                   phi: np.ndarray | None = None, init: dict | None = None,
                   forms: tuple[str, ...] = curves.FORMS) -> dict:
    cc = cfg["reach"]["calibration"]
    U = float(cfg["target"]["universe"])
    kmax = int(cfg["reach"]["max_frequency"])
    levels = [int(x) for x in cc["fit_reach_levels"]]
    for n in levels:
        if f"reach_{n}plus_pct" not in schedules.columns:
            raise CalibrationFileError(f"fit_reach_levels includes {n} but reach_{n}plus_pct is missing")
    cells = sorted(spots["channel"].unique())
    summ = summarise(spots, cells, U)
    rng = np.random.default_rng(int(cc["random_seed"]))
    ids = list(schedules["schedule_id"])
    perm = rng.permutation(len(ids))
    n_train = max(1, int(round(float(cc["train_share"]) * len(ids))))
    train = [ids[i] for i in perm[:n_train]]
    valid = [ids[i] for i in perm[n_train:]]
    rows = schedules.set_index("schedule_id")

    C = len(cells)
    x0 = np.empty(2 * C)
    for i, c in enumerate(cells):
        cinit = (init or {}).get(c, {})
        x0[i] = np.log(float(cinit.get("rmax_pct", 30.0)))
        x0[C + i] = np.log(float(cinit.get("k", 30.0)))
    lb = np.r_[np.full(C, np.log(0.1)), np.full(C, np.log(0.05))]
    ub = np.r_[np.full(C, np.log(99.0)), np.full(C, np.log(5000.0))]
    x0 = np.clip(x0, lb + 1e-9, ub - 1e-9)

    def residuals(x, form, sids):
        rmax, k = np.exp(x[:C]), np.exp(x[C:])
        return np.concatenate([predict(*summ[s], rmax, k, form, phi, kmax, levels) - _observed(rows.loc[s], levels)
                               for s in sids])

    fits = {}
    for form in forms:
        r = least_squares(residuals, x0, args=(form, train), bounds=(lb, ub), x_scale="jac")
        fits[form] = r
    form = min(fits, key=lambda f: fits[f].cost)
    x = fits[form].x
    rmax, k = np.exp(x[:C]), np.exp(x[C:])

    def metrics(sids):
        if not sids:
            return None
        res = residuals(x, form, sids).reshape(len(sids), len(levels))
        return {f"reach_{n}plus": {"mae_points": float(np.mean(np.abs(res[:, j]))),
                                   "rmse_points": float(np.sqrt(np.mean(res[:, j] ** 2))),
                                   "max_abs_points": float(np.max(np.abs(res[:, j])))}
                for j, n in enumerate(levels)}

    return {
        "method": "CALIBRATED",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "form": form,
        "form_costs": {f: float(2 * r.cost) for f, r in fits.items()},
        "cell_granularity": "channel",
        "universe": U,
        "fit_reach_levels": levels,
        "cells": {c: {"channel": c, "daypart": None, "rmax_pct": float(rmax[i]), "k": float(k[i]),
                      "initial_slope": float(rmax[i] / k[i])} for i, c in enumerate(cells)},
        "duplication": ({"cells": cells, "phi": phi.tolist()} if phi is not None else None),
        "n_schedules": len(ids), "n_train": len(train), "n_valid": len(valid),
        "train_ids": train, "valid_ids": valid,
        "train_error": metrics(train), "validation_error": metrics(valid),
    }


# ----------------------------------------------------------------------------- synthetic
def synthetic_rf(true_cells: dict[str, dict], form: str, universe: float, n_schedules: int = 50,
                 seed: int = 0, kmax: int = 10, phi: np.ndarray | None = None, noise_points: float = 0.0,
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate test schedules from a known curve model (for self-tests). Spot audiences are
    drawn around 0.3-1.5% of universe; schedules mix 1-7 channels at 5-400 GRPs."""
    rng = np.random.default_rng(seed)
    cells = sorted(true_cells)
    rmax = np.array([true_cells[c]["rmax_pct"] for c in cells])
    k = np.array([true_cells[c]["k"] for c in cells])
    srows, prow = [], []
    for s in range(n_schedules):
        sid = f"S{s:03d}"
        chans = rng.choice(len(cells), size=rng.integers(1, len(cells) + 1), replace=False)
        for ci in chans:
            n = int(rng.integers(1, 60))
            aud = rng.uniform(0.003, 0.015, n) * universe * rng.uniform(0.3, 1.5)
            for a in aud:
                prow.append({"schedule_id": sid, "channel": cells[ci], "aud_abs": float(a)})
        sp = pd.DataFrame([p for p in prow if p["schedule_id"] == sid])
        gv, nv = summarise(sp, cells, universe)[sid]
        obs = predict(gv, nv, rmax, k, form, phi, kmax, [1, 2, 3, 4, 5])
        obs = obs + rng.normal(0.0, noise_points, size=obs.shape) if noise_points else obs
        obs = np.minimum.accumulate(np.clip(obs, 0, 100))
        srows.append({"schedule_id": sid, "grp_pct": float(gv.sum()),
                      **{f"reach_{n}plus_pct": float(obs[n - 1]) for n in range(1, 6)}})
    return pd.DataFrame(srows), pd.DataFrame(prow)


def main(argv: list[str] | None = None) -> dict | None:
    ap = argparse.ArgumentParser(description="Fit CALIBRATED reach parameters to eTAM R&F outputs")
    ap.add_argument("--config", default=None)
    ap.add_argument("--self-test", action="store_true", help="synthetic recovery check only")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cc = cfg["reach"]["calibration"]
    U = float(cfg["target"]["universe"])
    if args.self_test:
        truth = {"A": {"rmax_pct": 55.0, "k": 60.0}, "B": {"rmax_pct": 25.0, "k": 20.0},
                 "C": {"rmax_pct": 12.0, "k": 15.0}}
        s, sp = synthetic_rf(truth, "hyperbolic", U, 50, seed=1)
        p = fit_calibrated(cfg, s, sp)
        for c, t in truth.items():
            print(f"{c}: true Rmax {t['rmax_pct']:.1f} k {t['k']:.1f} -> fitted "
                  f"Rmax {p['cells'][c]['rmax_pct']:.2f} k {p['cells'][c]['k']:.2f}")
        print("validation error:", p["validation_error"])
        return p
    if not cc.get("schedules_path") or not cc.get("spots_path"):
        print("No eTAM R&F calibration data configured (reach.calibration.schedules_path / spots_path); "
              "CALIBRATED mode unavailable - staying in ESTIMATE. Nothing written.")
        return None
    s, sp = load_rf_schedules(cc["schedules_path"], cc["spots_path"])
    cells = sorted(sp["channel"].unique())
    phi = load_duplication_matrix(cc["duplication_matrix_path"], cells, U) if cc.get("duplication_matrix_path") else None
    init = None
    est = abs_path(cfg["reach"]["params_path"])
    if est.exists():
        init = json.loads(est.read_text())["cells"]
    p = fit_calibrated(cfg, s, sp, phi=phi, init=init)
    out = abs_path(cc["params_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")
    print(f"CALIBRATED reach params -> {out}; form {p['form']}; validation {p['validation_error']}")
    return p


if __name__ == "__main__":
    main()
