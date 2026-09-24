# MMS TV Plan Optimizer — Project Memory

## Mission
Build an optimizer that produces an October 2026 TV spot plan across 7 MBC channels
(MBC 1, MBC 2, MBC 4, MBC ACTION, MBC BOLLYWOOD, MBC DRAMA, MBC MAX) for a USD 1,000,000
gross budget, using Nielsen KSA eTAM data, and delivering the highest reach, GRPs/TRPs and
impressions. Every KPI must reconcile with Nielsen eTAM definitions.

## Read these first (in order)
1. `docs/PRELIMINARY_FINDINGS.md` — what is already known about the data (verified).
2. `docs/DATA_SPEC.md` — file layouts, parsing rules, known traps.
3. `docs/NIELSEN_DEFINITIONS.md` — formulas, KPI mapping, PDF errata.
4. `docs/METHODOLOGY.md` — forecast, reach and optimization approach.
5. `docs/ACCEPTANCE_CRITERIA.md` — what "done" means.
6. `docs/OPEN_ISSUES.md` — decisions pending, with the defaults to use meanwhile.
7. `config/plan_config.yaml` — every tunable parameter. Never hard-code these.

## Non-negotiable rules
- **Never invent data.** If something is missing, use the documented default in
  OPEN_ISSUES.md, log it in `outputs/assumptions_log.md`, and flag it in the final report.
- **TRP Absolute in the eTAM break file is NOT spot impressions.**
  It equals Rating Absolute x break length in minutes.
  Per-spot audience: `rating_abs = TRP_Absolute * 60 / break_seconds`.
- **Rates in the grid are USD converted from AED at 3.6725.** Keep both.
- **Broadcast day runs 03:00–26:59.** Times >= 24:00 belong to the previous calendar date's
  broadcast day. Convert to real datetimes before any join.
- **Raw data is read-only.** Never modify files in `data/raw/`. Write derived data to
  `data/processed/`, outputs to `outputs/`.
- **Nielsen data is confidential.** Never commit `data/` to git; never upload it anywhere.
- Every number in the final plan must be traceable to a source row + a formula.
- Label every reach figure with its method: `EXACT` (respondent data), `CALIBRATED`
  (fitted to eTAM R&F outputs) or `ESTIMATE` (model only, not validated).

## Team (subagents in `.claude/agents/`)
| Agent | Model | Role |
|---|---|---|
| data-validator | sonnet | Ingest + validate every file, produce validation reports |
| nielsen-definitions | opus | Implement Nielsen formulas as tested functions |
| program-matcher | opus | Map October grid programs to eTAM program history |
| match-verifier | sonnet | Independently verify program matches (never sees matcher reasoning) |
| audience-forecaster | opus | Forecast per-slot October audience + back-test |
| reach-modeler | opus | Reach / frequency engine (EXACT / CALIBRATED / ESTIMATE modes) |
| optimizer | opus | Build the plan + scenarios + frontier |
| auditor | sonnet | Independent recomputation of every KPI; blocks delivery on mismatch |
| reporter | haiku | Excel/markdown output formatting only (no calculations) |

The main session is the **lead**: it plans, delegates, reviews outputs and enforces gates.
Calculations are never done by the reporter.

## Tech
- Python 3.11+, pandas, numpy, pyarrow, openpyxl, scipy, OR-Tools (`ortools`), pytest, pyyaml,
  rapidfuzz (fuzzy matching). Use a virtualenv (`.venv`).
- Code in `src/optimizer/`, tests in `tests/`, run `pytest -q` before every gate.
- Reference implementations already verified on the real files: `reference_scripts/`.

## Gates (stop and ask the human)
- **Gate 1** after ingestion: show validation reports + list of issues.
- **Gate 2** after program matching: show low-confidence matches for human review.
- **Gate 3** after back-test: show forecast accuracy before optimizing.
- **Gate 4** before final delivery: show scenarios + frontier; human picks the final plan.
