# Open Issues — Decisions Pending (use the default, log it, keep going)

| # | Issue | Default until decided | Config key |
|---|---|---|---|
| 1 | Primary objective | S1 max Reach 1+ with TRP floor; also produce S2–S4 | `objective.recommended` |
| 2 | Buying target / base audience | TP Arabs 15+ for both (only target in data) | `target.*` |
| 3 | Official universe | 11,110,000 inferred; replace with eTAM Universe | `target.universe` |
| 4 | MBC 1 grid week 25–31 Oct missing | Carry forward week of 18 Oct (same weekdays), flag every carried slot `ASSUMPTION_MBC1_WK4` | `grid.mbc1_week4` |
| 5 | Oct 1–3 not in grid | Flight = 4–31 Oct; ignore the 2 rows of week 20260927 | `flight.*` |
| 6 | Real inventory / availability | `free_time` ignored; 1 spot per slot | `caps.max_spots_per_slot` |
| 7 | Channel minimums ("use all channels") | ≥3% of budget each for MBC 1/2/4/DRAMA/BOLLYWOOD; ≥1% each for ACTION/MAX (low-sample) | `constraints.channel_min_share` |
| 8 | Reach data (respondent or eTAM R&F) | Mode ESTIMATE, clearly labelled | `reach.mode` |
| 9 | August + Sep 22–30 data | Use Sep 1–21 only; re-run when files land in `data/incoming/` | — |
| 10 | Common-weight rule / reach threshold / spot exposure rule | Average weight; threshold n/a in ESTIMATE mode | `reach.*` |
| 11 | Rates gross vs net, agency commission, discounts | Treat grid rates as gross USD; no discounts | `budget.*` |
| 12 | Live sport (NADEENA / KHALEEJI 27 LIVE) | Slot baseline excluding event days, `high_uncertainty` flag | `forecast.live_policy` |
| 13 | Rounding convention (truncate vs round) | Full precision internally; round half-up for display | `rounding` |
| 14 | Overlapping grid slots (8) | Max one spot across overlapping rows | automatic |

When the human answers any of these, update the config, remove the row's "default" status
here, and re-run from the affected phase.
