# Nielsen KSA eTAM Definitions — Implementation Reference

Source of truth: the two PDFs in `data/raw/nielsen_docs/`. Formulas in the PDFs are images;
the descriptions and worked examples below were transcribed and checked. If anything here
conflicts with the PDFs, the PDFs win — but where the PDF worked examples are internally
inconsistent (see Errata), real eTAM exports win.

## Notation
w_n = weight of viewer n; t_n = seconds/minutes viewed; D = event length; U = universe
(sum of weights of all panel members in the target; "average daily universe" for daily
metrics, "average universe in the period" for reach).

## Core formulas
| Data type | Formula | Notes |
|---|---|---|
| Rating Absolute | Σ(w_n · t_n) / D over people viewing ≥1 s | Multi-day/multi-channel: denominator = total analysed duration across all days x channels |
| Rating % | Rating Absolute / U x 100 | U = average daily universe |
| TRP Absolute | Σ Rating Absolute over events | In break reports: equals Rating Abs x break minutes (verified) |
| TRP % | Σ Rating % over events | |
| Share of Audience % | Rating Abs(channel) / Rating Abs(Total TV) x 100 | |
| Share to Selected % | Rating Abs(channel) / Σ Rating Abs(selected) x 100 | |
| Unduplicated Reach | Σ w̄_n over people viewing ≥1 item for ≥ threshold seconds | **common weight** w̄_n across the period (average or middle-day — confirm KSA rule) |
| Average Daily Reach | mean over days of daily reach (daily weights) | |
| Average Weekly Reach | mean over weeks of weekly reach (weekly weights) | |
| Cume Reach (RF) | running Unduplicated Reach line by line | final line = Unduplicated Reach |
| Incremental Reach | Cume Reach RF_n − Cume Reach RF_(n−1) | |
| Reach N+ / Reach N | Σ w̄_n over people with ≥N / exactly N exposures | 1+ = Unduplicated Reach |
| Frequency | Σ(w̄_n · f_n) / Σ w̄_n | average weights |
| GRP Absolute (spots) | Σ_spots Σ_viewers w_{n,s} | daily weight of the spot's day |
| GRP % | GRP Absolute / U x 100 | U = average daily universe |
| OTS | GRP Absolute / Unduplicated Reach | |
| CPM | Σ Cost / Σ GRP Absolute x 1000 | |
| Spot Cost per Rating % (CPP) | Σ Cost / Σ GRP % | |
| Weighted (30"-equivalent) | value x EqFactor(spot length) | Rate Duration Factors table |
| Universe | Σ w_n over target members | |

## KPI mapping for this project
| Client KPI | Implementation |
|---|---|
| Impressions | GRP Absolute of the plan = Σ spot audience (rating_abs) |
| GRPs | GRP % on base audience (default: TP Arabs 15+ until a separate base is given) |
| TRPs | GRP % on the buying target (default: TP Arabs 15+) |
| Reach 1+ / 3+ | Reach N+ with common weights (method label EXACT/CALIBRATED/ESTIMATE) |
| Avg frequency | OTS = GRP Absolute / Reach 1+ |
| Efficiency | CPM, CPP (Spot Cost per Rating %) |

## Rounding
eTAM appears to **truncate** in at least one example (OTS 15,300 / 3,800 = 4.026 shown as 4.02).
Compute at full precision; apply display rounding only in the reporter, using a single
configurable rule (`config.rounding`). Confirm against real eTAM exports.

## Errata found in the PDF worked examples
- Main Data Types, "Average Weekly TRP Absolute": lists 5 weekly values but divides by 4, and
  "Average Weekly TRP %" repeats the daily result (15,628.01). Correct weekly % for the stated
  inputs = 22,994,225,336 / 20,335,350 x 100 ≈ 113,075.
- OTS example 4.02 vs computed 4.026 (rounding convention, see above).
Mark these fixtures `status=erratum` — test the correct math, not the printed number.

## Test fixtures
`tests/fixtures/nielsen_fixtures.csv` holds the verified worked examples. Every formula
function must reproduce them (tolerance 0.5% unless stated). The nielsen-definitions agent
should add any further examples it extracts from the PDFs.
