You are the lead engineer and project lead for the MMS TV Plan Optimizer. This folder contains
everything you need. Work autonomously between gates, delegate to the subagents defined in
.claude/agents/, and stop at each gate for my review.

## Goal
Build an optimizer and produce an optimized October 2026 TV spot plan across the 7 MBC
channels in the grid (MBC 1, MBC 2, MBC 4, MBC ACTION, MBC BOLLYWOOD, MBC DRAMA, MBC MAX)
for a budget of USD 1,000,000 gross, maximizing reach, GRPs/TRPs and impressions for the
target TP Arabs 15+, using every channel. All KPIs must follow Nielsen KSA eTAM definitions.

## Step 0 — Orient (do this before writing any code)
1. Read CLAUDE.md, then every file in docs/ in the order listed in CLAUDE.md, then
   config/plan_config.yaml and reference_scripts/.
2. Confirm the subagents in .claude/agents/ are loaded (run /agents). If any are missing,
   tell me.
3. Set up a Python virtualenv (.venv), install requirements.txt, run the three reference
   scripts and confirm their output matches docs/PRELIMINARY_FINDINGS.md.
4. Give me a short plan of the work (phases, which agent does what) and then start.
   Do not wait for approval on the plan unless something in the docs is contradictory.

## Phases
Phase 1 — Ingestion & validation (data-validator)
- Build src/optimizer/ingest/ from the reference scripts, per docs/DATA_SPEC.md.
- Produce data/processed/grid.parquet, data/processed/breaks.parquet and validation reports
  in outputs/validation/.
- Invariant tests (TRP Absolute = rating_abs x minutes; AED rates integral; time conversion).

Phase 2 — Nielsen metrics (nielsen-definitions), can run in parallel with Phase 1
- src/optimizer/metrics/ + one test per row of tests/fixtures/nielsen_fixtures.csv.
- Read both PDFs yourself and add any extra worked examples as fixtures.

>>> GATE 1: show me the validation summary, test results, and any new issues. Wait.

Phase 3 — Program matching (program-matcher, then match-verifier independently)
- data/processed/program_map.csv; review sheet with low-confidence matches and disagreements.

>>> GATE 2: show me the match review sheet. Wait for my corrections, then apply them.

Phase 4 — Forecast + back-test (audience-forecaster)
- data/processed/slot_forecast.parquet (p10/p50/p90 per slot) + backtest_report.md.

>>> GATE 3: show me back-test accuracy by channel and tier and the plan-level error. Wait.

Phase 5 — Reach engine (reach-modeler), can start in parallel with Phase 4
- ESTIMATE mode now (label it), EXACT and CALIBRATED ready for when data arrives.

Phase 6 — Optimization (optimizer)
- Scenarios S1–S4 from config, frontier, constraint verification.

Phase 7 — Audit (auditor) — independent recomputation; any FAIL blocks delivery; fix and re-audit.

>>> GATE 4: show me the scenario comparison, frontier and key risks. I choose the final plan.

Phase 8 — Deliverables (reporter) per docs/ACCEPTANCE_CRITERIA.md.

## Working rules
- Follow CLAUDE.md non-negotiable rules at all times (especially: per-spot audience =
  TRP_Absolute x 60 / break_seconds; never invent data; label reach method; raw data read-only).
- All parameters come from config/plan_config.yaml. Where a decision is pending, use the
  default in docs/OPEN_ISSUES.md and log it in outputs/assumptions_log.md.
- Use subagents for their roles; the auditor must never read src/optimizer/.
- Run pytest before every gate. Keep a running outputs/PROGRESS.md (what's done, what's next,
  open questions) so work can resume after interruptions.
- When new files appear in data/incoming/ (August, Sep 22–30, universe, reach data, MBC 1 week 4),
  process them via the data-validator and re-run the affected phases.
- If you are uncertain about a Nielsen definition, say so and show the evidence from the PDF;
  do not guess silently.
- At every gate, be concise: results, issues, decisions needed (with your recommendation).

Start with Step 0 now.
