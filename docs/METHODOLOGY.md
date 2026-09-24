# Methodology

## 1. Unit of planning
One **candidate spot** = one 30" spot in one grid row (slot_id). Default cap: at most
`caps.max_spots_per_slot` (1) per grid row. Overlapping grid rows on the same channel/date
share one cap.

## 2. Audience forecast (per slot, per target)
Goal: expected per-spot audience `aud_abs` (= forecast rating_abs of a break in that slot).

### 2.1 Baseline construction from history (breaks.parquet, events excluded)
Hierarchy — use the most specific level with enough evidence (`forecast.min_breaks`, default 6),
and shrink toward the next level (empirical Bayes / partial pooling):
1. **Program level**: same matched program title, same channel, same run type
   (first run vs rerun), similar time window (±`forecast.time_tolerance_min`).
2. **Slot level**: same channel, same weekday group (Sun–Wed, Thu, Fri, Sat — test whether
   finer weekday grouping improves the back-test), break starts within the slot window.
3. **Channel x daypart x day-type** level.
Use medians / trimmed means (robust to outliers). Record which level was used per slot
(`forecast_level`) and the evidence count.

### 2.2 Adjustments
- **New October programs** (no history): slot-level baseline x channel genre factor if
  measurable; flag `is_new_program`, widen uncertainty.
- **Reruns (®)**: use rerun history of the same title if present, else slot baseline.
- **Live sport / specials** (`is_live`, event titles): separate model; default = slot
  baseline excluding event days, flagged `high_uncertainty`. Never extrapolate from a single
  event day.
- **Low-sample channels** (ACTION, MAX): use pooled channel x daypart level only; flag.
- **Seasonality**: when August data arrives, estimate an Aug->Sep trend per channel x daypart
  and project half a step forward for October (configurable, default off until tested).

### 2.3 Uncertainty
For every slot produce `aud_abs_p10`, `aud_abs_p50`, `aud_abs_p90` (bootstrap over history).
The optimizer uses p50; the report shows the range.

### 2.4 Back-test (Gate 3)
- With Sep 1–21 only: fit on Sep 1–14, predict Sep 15–21 slots (grid-like windows built from
  the September schedule), compare predicted vs actual rating_abs.
- When August arrives: fit on August, predict September.
- Report: MAPE and weighted MAPE (by audience) by channel x tier, bias, and the error of the
  **plan-level total impressions** for a mock plan (the number that matters most).
- Target: plan-level impressions error ≤ ±10%; slot-level WMAPE reported, not gated.

## 3. Reach engine
Three modes; the engine must expose the same interface for all:
`reach(schedule) -> {reach_1plus, reach_3plus, reach_n_dist, grp_abs, grp_pct, ots, method}`.

- **EXACT** (respondent data available): implement Nielsen exactly — per-person exposure
  counts across spots with the common weight rule; reproduce eTAM figures to ≤0.1%.
- **CALIBRATED** (eTAM R&F outputs for test schedules + duplication matrix): fit a reach
  model (e.g. beta-binomial / Hofmans per channel x daypart, combined across cells with the
  duplication matrix or Sainsbury as a fallback). Fit on ~80% of test schedules, validate on
  the rest; report reach error in points.
- **ESTIMATE** (current state: break-level data only): per-channel reach curve
  R(g) = Rmax · g / (k + g) or negative-exponential form, with Rmax and k anchored on what
  the data does show (per-break Unduplicated Reach vs rating ratios, channel daily audience
  levels) and cross-channel combination via Sainsbury. **Label output ESTIMATE and never
  present it as measured.** Show the sensitivity of the plan to ±30% on k.

Marginal reach of adding a spot must be computable quickly (the optimizer calls it a lot).

## 4. Optimizer
### 4.1 Objectives (run all scenarios; config chooses the recommended one)
- S1 `max_reach_1plus` (default recommended): maximize Reach 1+ s.t. budget and constraints,
  with a TRP floor (`constraints.min_trp_pct`).
- S2 `max_reach_3plus`: effective reach.
- S3 `max_impressions`: maximize GRP Absolute (linear -> exact MILP).
- S4 `balanced`: maximize Reach 1+ subject to impressions ≥ 90% of S3's optimum.

### 4.2 Algorithm
- Linear objectives (S3): MILP with OR-Tools CP-SAT or SCIP; integer costs in AED fils or
  USD cents to avoid float issues.
- Reach objectives (submodular): lazy-greedy on marginal reach per dollar (with the
  cost-benefit correction: best of greedy-by-ratio vs best single-item), then local search
  (swap/drop-add) to escape greedy traps. Constraints enforced at every step.
- Always verify the final plan against every constraint in a separate check function.

### 4.3 Constraints (all from config)
Budget (≤ total, ≥ `budget.min_utilisation`); per-slot cap; per-channel min/max share of
budget; per-channel min spots; max spots per channel per day; max spots per program title
per day; weekly phasing (min/max share per week); daypart mix bounds (optional); exclusions
list. Every channel must receive ≥ its minimum (the "use all channels" requirement).

### 4.4 Frontier
Run S1 across budget levels 25%/50%/75%/100%/125% and across TRP floors to produce:
reach vs budget curve, reach vs GRP frontier. These go into the report.

## 5. Audit (independent)
The auditor re-implements KPI calculations from scratch in `audit/` using only the Nielsen
definitions doc, the processed data and the plan file — never importing `src/optimizer`.
Any difference > 0.1% on cost, impressions, GRP %, CPM, CPP, or > 0.1 reach points
(same method) blocks delivery.
