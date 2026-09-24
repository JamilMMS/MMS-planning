# Acceptance Criteria

## Deliverables (in `outputs/`)
1. `October_Plan_MBC.xlsx`
   - **Summary**: KPIs of the recommended scenario — budget used (USD + AED), spots,
     Impressions (GRP Absolute), GRP %, TRP %, Reach 1+ / 3+ (with method label), OTS,
     CPM, CPP; plus a scenario comparison table (S1–S4).
   - **Spot list**: one row per spot — date, weekday, channel, program, slot start–end,
     tier, rate USD, rate AED, forecast audience p50 (and p10/p90), rating %, CPM,
     forecast_level, flags (new program / live / low sample / assumption).
   - **By channel**, **by week**, **by daypart/tier**: spots, spend, share of spend,
     impressions, GRP %, CPM, CPP.
   - **Frontier**: reach vs budget and reach vs GRP tables (+ native Excel charts).
   - **Assumptions & flags**: every default used, every data gap.
   - **Validation**: back-test accuracy, audit result.
2. `october_grid_filled.xlsx` — the original grid layout with `audience` (forecast rating_abs),
   `universe`, `num_ratings` populated and a `planned_spots` column, so it can be re-imported.
3. `plan_spots.csv` — machine-readable plan.
4. `REPORT.md` — 1–2 page plain-English summary for the planner: what the plan does, why,
   key risks, what would change with the pending data.
5. `validation/` — ingestion reports, match review sheet, back-test report, audit report.

## Quality gates
- `pytest -q` green, including every fixture in `tests/fixtures/nielsen_fixtures.csv`
  (erratum fixtures test the corrected math).
- Invariant tests: TRP_abs / (rating_abs x minutes) ≈ 1; rate x 3.6725 integral;
  no spot outside grid availability; no constraint violated; budget ≤ 1,000,000.00.
- Auditor report: PASS on all KPIs.
- Back-test report produced; plan-level impressions error stated.
- Every reach number labelled EXACT / CALIBRATED / ESTIMATE.
- Re-running the full pipeline from `data/raw` + config reproduces the same plan
  (deterministic seeds).

## Commands
- `make all` (or `python -m optimizer.run --config config/plan_config.yaml`) rebuilds
  everything end-to-end.
- `make test`, `make audit`, `make report` as separate targets.
