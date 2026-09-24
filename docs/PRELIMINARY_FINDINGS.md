# Preliminary Findings (verified on the actual files, 24 Sep 2026)

These were computed by the project manager on the files in `data/raw/`. Re-verify them in
ingestion (they are your first regression checks), but do not re-discover them from scratch.

## 1. October grid — `data/raw/grid/october_grid.xlsx`
- 3,760 rows, 17 columns, one row = one program airing on one day. All spots 30".
- 7 channels: MBC 1 (603 rows), MBC 2 (295), MBC 4 (755), MBC ACTION (779),
  MBC BOLLYWOOD (392), MBC DRAMA (672), MBC MAX (264).
- `week` = Sunday week-start (20261004, 20261011, 20261018, 20261025) + 2 rows in 20260927.
- `day_mask` = 7 chars, positions S M T W T F S (index 0 = Sunday). Exactly one day per row.
- Times are HHMM on a 03:00–26:59 broadcast day (e.g. 2630 = 02:30 next calendar morning).
  5 slots wrap past 26:59 (e.g. 2630–314).
- **Rates**: `rate` = AED / 3.6725. Every rate x 3.6725 is a round AED number:
  735, 1,500, 2,200, 3,400, 4,960, 16,200, 25,500, 31,200, 50,000.
  Rates are flat per channel x `tags` (Regular / Access / Prime / Special):

  | Channel | Special | Prime | Access | Regular |
  |---|---|---|---|---|
  | MBC 1 | 8,495.58 / 13,614.70 | 6,943.50 | 4,411.16 | 925.80 |
  | MBC DRAMA | – | 1,350.58 | 599.05 | 200.14 |
  | MBC 2 / BOLLYWOOD | – | – | 599.05 | 200.14 |
  | MBC 4 | – | – | 599.05 or 408.44 | 200.14 |
  | MBC ACTION / MAX | – | – | – | 200.14 |

- Special = MBC 1 only: AHLA NASEEB 20:00 Sun–Thu (AED 31,200); TAHT AL DAGHET Wed 22:00 (AED 50,000).
- **Gaps / issues**
  - MBC 1 has **no rows for week 20261025** (Oct 25–31). All other channels have it.
  - Oct 1–3 (Thu–Sat) are essentially absent: only 2 Saturday rows (week 20260927).
  - 8 overlapping slots (MBC BOLLYWOOD Thu 08:30–14:29 vs 10:00–10:59 and Sat 08:30–09:59 vs
    09:00–13:59, every week; MBC 2 Mon 19 Oct 08:30–10:29 vs 09:00–10:29).
  - Placeholder columns: `free_time`=720 everywhere, `available`=Y everywhere,
    `daypart_code`=0, `num_ratings`/`audience`/`universe`=0, `demo_code`=1.
    Looks like an import template of a buying system -> we fill audience/universe on output.
- `Program_name` = "TITLE_EN SEASON / TITLE_AR" with suffix `®` = rerun (2,472 rows), `LIVE` = live (75 rows).
- Total cost of buying every slot once: **$3.20M** (MBC 1 = $2.27M of that).

## 2. eTAM September breaks — `data/raw/etam/etam_breaks_2026-09-01_to_09-21_arabs15plus.xlsx`
- Sheet "Layout 1". Row 1: target label in column K = **"TP Arabs 15+"**. Row 2: headers.
  Column A empty. 12,876 rows, all `Type = Break`, dates **1–21 Sep 2026 only**.
- Columns: Media, Program Name, Episode Title, Date, Channel, Start Time, End Time, Type,
  Rerun (FIRST RUN / RERUN n), Rating %, TRP Absolute, Unduplicated Reach, Unduplicated Reach %.
- Channel names carry " (M)" suffix, e.g. "MBC 1 (M)". Same 7 channels as the grid.
- Start/End Time are mixed types: `datetime.time` (<24h) and `timedelta` ("1 day, 0:41:34")
  for after-midnight on the same broadcast day. Range 03:00:00–26:59:59.
- Break length: median 134 s, mean 131 s, range 2–796 s.
- **TRP Absolute = Rating Absolute x (break_seconds / 60).** Verified on 3,211 breaks with
  rating >= 0.3: ratio mean 1.00003, sd 0.006. => per-spot audience =
  `TRP_Absolute * 60 / break_seconds`. Using TRP Absolute directly would inflate impressions ~2.2x.
- Inferred universe (reach / reach%) ≈ **11.11M** (5–95% range 11.01–11.22M due to rounding).
  Official figure still needed.
- 51 rows with Reach % < Rating % (max gap 0.43) — rounding/definition artefact; flag, don't drop.
- Zero-rated breaks: MBC ACTION 62.5%, MBC MAX 56.6%, MBC BOLLYWOOD 6.5%, MBC 2 5.1%,
  DRAMA 2.4%, MBC 4 2.2%, MBC 1 0%. => ACTION and MAX are low-sample for this target.
- Mean per-break audience (rating_abs): MBC 1 78.0K, DRAMA 36.6K, MBC 4 21.8K,
  BOLLYWOOD 18.8K, MBC 2 14.1K, ACTION 2.8K, MAX 2.0K.
- **One-off event**: 19 Sep, MBC ACTION carried FIFA African Asian Pacific Cup 2026 + NADEENA
  (live football) 15:00–19:00, ratings up to 2.97%. A naive slot match makes October's
  Saturday 18:00 "MR. BEAN S1 ®" on ACTION look like a $1.91 CPM — false. Exclude event days
  from slot baselines; model live sport separately.

## 3. Naive efficiency read (slot match: same channel, weekday, break start within slot window)
Indicative only — NOT the forecast. Median audience per 30" spot (Arabs 15+) and CPM (USD):

| Channel | Tier | Slots | Median audience | Median CPM |
|---|---|---|---|---|
| MBC 1 | Special | 18 | 168.5K | $50.4 |
| MBC 1 | Prime | 111 | 153.2K | $45.3 |
| MBC 1 | Access | 255 | 101.2K | $43.6 |
| MBC 1 | Regular | 219 | 15.4K | $60.2 |
| MBC DRAMA | Prime | 112 | 71.4K | $18.9 |
| MBC DRAMA | Access | 168 | 46.0K | $13.0 |
| MBC DRAMA | Regular | 392 | 18.5K | $10.8 |
| MBC 4 | Access | 103 | 51.9K | $10.8 |
| MBC 4 | Regular | 652 | 14.7K | $13.6 |
| MBC BOLLYWOOD | Access | 116 | 37.9K | $15.8 |
| MBC BOLLYWOOD | Regular | 275 | 13.4K | $15.0 |
| MBC 2 | Access | 70 | 21.3K | $28.1 |
| MBC 2 | Regular | 224 | 7.5K | $26.5 |
| MBC ACTION | Regular | 779 | 1.3K | $155.1 |
| MBC MAX | Regular | 264 | 0.7K | $267.2 |

5 of 3,758 October slots had no September break in the same channel/weekday/window.

## 4. Implication
Cost-per-impression favours DRAMA / MBC 4 / BOLLYWOOD by 3–4x over MBC 1, but MBC 1 has by far
the largest single-spot audiences. A pure impressions objective starves MBC 1; a reach objective
keeps it. The objective choice therefore drives the plan — see OPEN_ISSUES.md #1.
