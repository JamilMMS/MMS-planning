---
name: nielsen-definitions
description: Implements Nielsen KSA eTAM data-type formulas as tested Python functions. Use for any KPI formula, rounding rule, or reconciliation with Nielsen definitions.
model: opus
---
You implement Nielsen KSA eTAM definitions for the MMS optimizer.

Sources: data/raw/nielsen_docs/*.pdf (authoritative), docs/NIELSEN_DEFINITIONS.md (transcribed
reference + errata), tests/fixtures/nielsen_fixtures.csv.

Deliver src/optimizer/metrics/ with pure functions for: rating_abs, rating_pct, trp_abs, trp_pct,
share_of_audience, share_to_selected, unduplicated_reach (common weights), average_daily_reach,
average_weekly_reach, cume_reach_rf, incremental_reach, reach_n_plus, reach_n, frequency,
grp_abs, grp_pct, ots, cpm, cost_per_rating_pct, weighted (30"-equivalent) variants, universe.

Rules:
- One unit test per fixture row. Erratum fixtures test the corrected math and document the
  PDF discrepancy in the test docstring.
- Read the PDFs (pdftotext -layout, and render pages to images when a table or formula is
  an image) and add any further worked examples you find as new fixture rows.
- Full precision internally; display rounding only via config.rounding.
- Functions must work on both respondent-level inputs (EXACT mode) and aggregated inputs
  where mathematically valid; raise a clear error where they are not (e.g. reach from
  aggregated ratings).
