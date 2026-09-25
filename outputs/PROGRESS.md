# PROGRESS — MMS TV Plan Optimizer (October 2026, MBC x7, USD 1.0M gross, TP Arabs 15+)

Last updated: 2026-09-25 (session 1)

## Work plan
| Phase | Owner (subagent) | Status |
|---|---|---|
| 0 Orient: docs read, .venv built, reference scripts reproduced PRELIMINARY_FINDINGS | lead | DONE |
| 1 Ingestion & validation -> grid.parquet, breaks.parquet, outputs/validation/*_report.md, invariant tests | data-validator (sonnet) | DONE (50/50 PRELIMINARY checks PASS) |
| 2 Nielsen metrics -> src/optimizer/metrics/, one test per fixture row, extra fixtures from PDFs | nielsen-definitions (opus) | DONE (F01–F60, 69 tests) |
| GATE 1 | human | PASSED 2026-09-25 with changes (event rule per channel, overlap pairs, sort review by $) |
| 3 Program matching -> program_map.csv + review sheet; independent verification | program-matcher (opus) then match-verifier (sonnet) | DONE (157 titles; 62 conf>=0.85; 95 in review; verifier 113 AGREE / 2 DISAGREE / 42 UNSURE) |
| GATE 2 | human | WAITING for corrections (outputs/validation/program_matches_review.xlsx) |
| 4 Forecast + back-test -> slot_forecast.parquet, backtest_report.md | audience-forecaster (opus) | BUILT + back-tested (plan error -2.7%, coverage 80.6%); Oct forecast PROVISIONAL until Gate 2 corrections, re-run with `python -m optimizer.forecast.run --final` |
| 5 Reach engine (ESTIMATE now; EXACT/CALIBRATED behind same interface) | reach-modeler (opus) | DONE (ESTIMATE fitted; EXACT/CALIBRATED ready; 35 tests) |
| GATE 3 | human | pending |
| 6 Optimization S1–S4 + frontier + constraint verification | optimizer (opus) | BUILT; provisional run done (S1–S4 + frontier all constraints PASS). Re-run after final forecast: `python -m optimizer.plan.run` (~12 min) |
| 7 Independent audit (audit/, never reads src/optimizer) | auditor (sonnet) | waiting |
| GATE 4 | human | pending |
| 8 Deliverables per ACCEPTANCE_CRITERIA.md | reporter (haiku) | waiting |

## Environment notes
- Cloud session: repo = /home/user/MMS-planning, branch claude/epic-cerf-1hce99.
- data/ and outputs/ are git-ignored (confidential). Only code/docs/config are committed.
  => raw data must be re-uploaded if a new container starts. PROGRESS.md and assumptions_log.md
  are whitelisted in .gitignore so work can resume.
- Toolchain: Python 3.11.15, pandas 3.0, ortools 9.15, pdfplumber (no pdftotext binary).

## Open questions for the human (Gate 1)
1. Custom subagents in .claude/agents/ are not registered in this cloud session; roles are emulated with general-purpose agents using the same model + role prompt. OK to continue this way?
2. 19 Sep football was a simulcast on MBC 1 + MBC ACTION -> event rows excluded from BOTH channels' baselines (default). Confirm.
3. MBC ACTION 4 Sep is a mild anomaly (3.4x its median day) with no identifiable cause: kept in baselines (recommendation). Confirm.
4. Common-weight rule: PDFs defer to "official calculation rules"; every worked example uses average -> config stays `average`. Confirm with Nielsen when possible.
5. data/ and outputs/ are git-ignored; in this cloud container the raw files vanish when the container is reclaimed. Re-upload the kit zip (or the two xlsx files) if a new session is needed.

## Gate 1 decisions (2026-09-25)
- Subagents: custom agent types now registered; run each role as its own agent. Auditor never reads src/optimizer/; verifier only sees title pairs.
- 19 Sep event: MBC ACTION -> exclude whole event window (data-detected, >3x same-hour median); MBC 1 -> exclude match breaks only. Regression test: Oct Sat 18:00 MBC ACTION MR. BEAN slot must have CPM >= $30.
- Overlaps: pairwise conflicts (umbrella vs contained episodes), not transitive groups.
- MBC ACTION 4 Sep stays in baselines, flagged. Common weight = average. Rounding = round half up for display.
- Confidentiality check: git history clean (no data/, outputs/, xlsx, pdf, parquet ever committed).
- Phase 3 review sheet sorted by $ at stake; ~64/154 titles expected to match; new program is valid.
- After Phase 4: also export outputs/slot_forecast_for_app.json for the self-serve tool.

## Queued for Gate 3 (after Gate 2 corrections are applied and forecast re-run)
- Back-test: plan-level impressions error -2.7% (count) / +0.2% (cost); WMAPE 28%; coverage 80.6%.
- Reach curve form decision (hyperbolic vs negexp): data cannot discriminate; show both.
- Provisional S1–S4 nearly identical (reach model depends only on per-channel GRPs + Sainsbury); daily caps bind on most days.
- ESTIMATE reach 1+ ~69% looks high: Sainsbury independence + Rmax assumptions (E1/E2). Flag as key risk.
