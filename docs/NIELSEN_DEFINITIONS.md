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

## Evidence from the PDFs (Phase 2)

Implementation: `src/optimizer/metrics/nielsen.py` (formulas), `src/optimizer/metrics/rounding.py`
(display only), tests `tests/test_nielsen_fixtures.py` (one test per fixture row, F01–F60).
Page numbers are **physical PDF pages** (MDT = Main Data Types, GL = Glossary). The printed
table-of-contents labels in the MDT program section are off by one, so F15–F17 were corrected
from p44/p45/p48 to p43/p44/p47. GL printed page numbers are 7 lower than the physical page
(e.g. Unduplicated Reach is printed "9" on physical page 16).

### 1. Common-weight rule (Unduplicated Reach, Cume Reach, Reach N+/N, Frequency)
- **The Glossary does not state which rule KSA uses.** GL p16 (Unduplicated Reach), and with the
  same wording GL p17, p22, p23, p26–p29 and Duplication/Exclusive Cume Reach GL p76/p78:
  > "The figures are calculated with a common weight across the entire period of analysis
  > (average or middle day weight, depending on the official calculation rules in place)."
  > — "w_n = common weight of viewer n in the period of analysis"
- GL p30 (Frequency): "calculated for each row of the layout and using **average weights** in the
  period of analysis" — "w_n = average weight of viewer n in the period of analysis".
- **Every multi-day worked example uses the average of the person's daily weights**: MDT p18/p19
  and p55/p56 ((900+950)/2 + …), p21/p22, p24 and p62 (legend "AVERAGE WEIGHTS": 725 = (700+750)/2 —
  a middle-day rule could only give 700 or 750), p27, p58, p64 ("AVERAGE WEIGHT"), p84, p87, p90.
- Weekly weight for Average Weekly Reach (GL p20: "calculated with weekly weights") is shown on
  MDT p15 as the mean of the person's daily weights within the week.
- Not determinable from the PDFs: whether the average runs over all in-tab days or only the days
  the person viewed (the examples only show weights on viewing days; a person seen on one day
  gets that day's weight, e.g. 700 on MDT p18).
- Implemented: `common_weights(daily_weights, rule)` with `rule = cfg.reach.common_weight_rule`.
  `average` = mean of the supplied daily weights (supply in-tab days). `middle_day` = weight on
  `sorted(days)[(n-1)//2]`, person with no weight that day gets 0 — **both the lower-middle
  tie-break and the exclusion are implementation choices, not in the PDFs.**
- Verdict: evidence supports the config default `average`; still confirm with Nielsen
  (OPEN_ISSUES #10) because the Glossary explicitly defers to "official calculation rules".

### 2. Reach minimum-seconds threshold and spot-exposure rule
- **No numeric reach threshold is stated.** GL p16/p22/p26/p28: "for at least a specified minimum
  amount of seconds (as defined in the Options filter)". GL p18/p20: "a specified minimum amount of
  seconds (as defined in the Options filter)". MDT p12: "viewed a minimum amount of the program or
  time band".
- Rating Absolute uses **≥ 1 second** (GL p8: "V = people watching at least 1 second of the event").
- GL p10/p11 (TRP): "The standard is 1-minute viewing" — ambiguous (reads as the minute-level basis
  of the rating, not a reach threshold).
- MDT p70 (Lead In/Out) shows Options examples (10-min qualifying period with 5-min criterion;
  10-min lead-in with 3-min criterion) — these are Lead In/Out settings, not the reach threshold.
- Spot exposure: GL p38 "V_s = people watching the s-th spot" — how much of the spot must be seen
  is not defined. `exposure_counts(viewing, min_seconds)` therefore takes the threshold as a
  required argument.

### 3. GRP on spots — daily weights
- GL p38: "Total number of exposure, calculated with **daily weights**, cumulated for all days of the
  analysis and all spots" — "w_{n,s} = daily weight of viewer n in the day where the s-th spot aired".
- MDT p81: Spot Y on 12 May = 4,200 (12-May weights) + 13 May = 4,400 (13-May weights) = 8,600.
- GRP % denominator: "average daily demographic universe" (GL p39). Reach % denominator:
  "average demographic universe … in the period of analysis" (GL p17). They can differ; the
  project currently has one inferred universe (config `target.universe`).
- OTS = **TRP Absolute** / Unduplicated Reach (GL p40 formula; MDT p92 labels the same number
  "GRP ABSOLUTE" in the figure and "TRP Absolute" in the table) — numerator uses daily weights,
  denominator common weights, by design.

### 4. Cume Reach (RF) line weighting — ambiguous
- MDT p21 and p58 (dayparts/programs, one line per day): line 1 = the first day's reach with
  **that day's** weights (3,800; 5,600), the final line with two-day average weights (6,075; 6,775).
- MDT p87 (spots, multi-day): **average two-day weights are used from line 1** (3,175 = 925 + 1,100 + 1,150).
- `cume_reach_rf(lines, weights)` supports both: one mapping for all lines (p87) or one mapping per
  line (p21/p58). The reach-modeler must choose; for spot schedules p87 is the relevant example.

### 5. Rounding evidence (printed vs full precision)
The PDFs are **inconsistent**: some values are truncated, others rounded, and several are computed
from intermediate values already rounded to integers.

| Page | Data type | Full precision | Printed | Convention |
|---|---|---|---|---|
| MDT p92 | OTS | 4.0263 | 4.02 | truncation |
| MDT p64 | Frequency | 1.7389 | 1.73 | truncation |
| MDT p48 | Share to Selected % | 11.582 | 11.5 | truncation (table values) |
| MDT p16, p53 | Avg Weekly Reach % | 0.026375 | 0.02 | truncation |
| MDT p32 | TSV Universe (Weekly) | 1.0765 | 1.07 | truncation |
| MDT p10 | Share of Audience % | 2.1382 | 2.14 | rounding |
| MDT p19, p56 | Unduplicated Reach % | 0.023875 | 0.024 | rounding |
| MDT p44 | Rating Abs (program, multi-day) | 1,866.67 | 1,867 | rounding |
| MDT p45 | Rating % (program) | 0.010667 | 0.011 | rounding |
| MDT p36 | Avg Daily TRP Absolute | 3,178,009,781.71 | 3,178,009,782 | rounding |
| MDT p37 | Avg Daily TRP % | 15,628.0063 | 15,628.01 | rounding |
| MDT p28 | TSV Universe (Daily) | 0.0034750 | 0.0035 | rounding |
| MDT p29 | TSV Universe (Daily, 3 days) | 0.0029667 | 0.0030 | rounding |
| MDT p88 | Cume Reach % (RF) | 0.019 | 0.02 | rounding |
| MDT p30 | TSV Viewers (Daily) | 23.1667 | 23.16 | intermediate rounding (Rating Abs 1,158) |
| MDT p31 | TSV Viewers (Daily, 3 days) | 20.1130 | 20.12 | intermediate rounding (989) then rounding |
| MDT p33 | TSV Viewers (Weekly) | 216.25 | 216 | intermediate rounding (1,044) |
| MDT p6, p11, p14, p22, p26, p27, p34, p51, p66 | various | — | — | consistent with either rule |

Conclusion: neither rule can be confirmed from the PDFs. Keep full precision internally and the
single configurable display rule (`rounding.display`: `round_half_up` default, `truncate`
available); confirm against real eTAM exports (OPEN_ISSUES #13).

### 6. New errata found (tests assert the corrected math)
| Fixture | Page | Printed | Correct | Issue |
|---|---|---|---|---|
| F24 | MDT p8 | TRP % 255.33 | 255.12 | formula terms sum to 255.12; the table (14:00 = 13.02 vs 13.82 in the formula) sums to 254.32 |
| F28 | MDT p84 | Undup. Reach 4,825 | 4,775 | uses (1,100+1,300)/2 for the person weighing 1,000 / 1,300; identical data on p18 = 4,775 |
| F36 | MDT p24, p62 | Reach 4+ 2,600 | 1,875 | includes person A (725) who has only 3 exposures; p62 legend also shows 920 for 925 |
| F37 | MDT p60 | Reach N "Total 9,900" | 1 = 700, 2 = 900, 3 = 1,800 | shows gross contacts (sum of per-program reach), and mis-adds them (terms sum to 7,900) |
| F38 | MDT p89 (and p91) | Reach 2+ 2,800 | 2,900 | 1,100+1,000+800 = 2,900; propagated to 2+% 0.014% on p91; spurious "1,800" in exposure 4 weights |
| F42 | MDT p64 | Σw·f 9,825 | 6,825 | table typo; 6,825/3,925 = 1.7389 (printed 1.73) |
| F52 | MDT p35 | Total Duration 77 min | 147 min | 16:30:00–17:59:59 is 90 min, listed as 30 |
| F56 | MDT p48 | formula terms | table values | formula lists 1,000/2,100/1,300 where the table has 1,100/2,133/1,083 |
| — | MDT p38 | Avg Weekly TRP Abs 22,994,225,336 | not reproducible | 5 weekly values for a 4-week period, divided by 4; neither Σ/4 (28.69bn), Σ/5 (22.95bn) nor any 4-value subset gives the printed value (dropping the near-duplicate 22,977,986,703 gives 22,946,725,336 — close but not equal). No fixture added. |
| — | MDT p74 | formula image Σ over V1\V2 | V1∩V2 | Duplication page shows the Exclusive formula (GL p76 is correct) |
| — | MDT p44, p65 | minor labels | — | p44 "1,300 10:00:00–10:14:59 (10 minutes) = 19,500" (15 min); p65 "700 * 12 = 70" omits "/120" |
| — | MDT p58 | formula text | — | omits the 1,400 term but the printed 6,775 is right |
| — | MDT p90 | formula labels | — | copied from p24 (725+1,050…); printed values correct |

### 7. Other ambiguities (explicit, not guessed)
- **Weighted Spot Cost per Rating %** (GL p51) prints the denominator as Σ(GRP %_n / 1000), which
  looks copied from CPM (GL p50) and contradicts Spot Cost per Rating % (GL p45, no /1000).
  Implemented without /1000 — confirm.
- **Weighted Spot Cost** (GL p49) *divides* cost by EqFactor while Weighted Rating (GL p47/p48)
  *multiplies* GRP by EqFactor; Weighted CPM/CPP (GL p50/p51) divide weighted cost by the
  *actual* (unweighted) GRP. Implemented exactly as printed. No worked example exists.
- **Rate Duration Factors** table is not in the PDFs (GL p47: "can be defined in the Rate Duration
  Factors table of the Rate Card option of the application"). Default `{30: 1.0}` only; any other
  length raises until the table arrives (DATA_SPEC C.4).
- **TSV (Weekly)** (GL p33/p34, MDT p32/p33): Σ(Rating Abs_n × D_n) over all weeks — for a 3-week
  period this is the total, not a per-week average, despite the "weekly" label.
- **Exclusive Cume Reach** MDT p75/p76: the 10:30 weights column lists "1,100, 1,000, 800" but only
  two viewers are drawn; the printed matrix (Program 2 not Program 1 = 800) matches the drawing
  only. p74 draws the same slot with three viewers.
- **Share of Audience** denominator (GL p12): "S = all measured channels and non-broadcast
  activities" — Total TV may include non-broadcast screen use.
- **Loyalty %, Peak/Floor, Lead In/Out, Universe, Sample Size, CPM, CPP, Weighted**: formulas only,
  no numeric worked examples in either PDF (MDT p77–78 show Loyalty % screenshots without inputs);
  covered by synthetic unit tests, not fixtures.
