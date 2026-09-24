# Data Specification

## Folder contract
```
data/raw/grid/           October grid (buying system export) — read-only
data/raw/etam/           eTAM exports — read-only
data/raw/nielsen_docs/   Nielsen KSA TAM definitions PDFs — read-only
data/incoming/           Drop zone for NEW exports (August, Sep 22–30, reach data, universe)
data/processed/          Everything derived (parquet). Safe to delete and rebuild.
outputs/                 Plans, reports, logs
```
When a new file appears in `data/incoming/`, the data-validator identifies its type from its
header row and layout, validates it, and moves a copy to the right `data/raw/` subfolder.

## A. October grid (have it)
Columns: station_code, Station_name, week, start_time, end_time, day_mask, spot_length,
Program_name, daypart_code, tags, rate, available, free_time, num_ratings, demo_code,
audience, universe.

Parsing rules:
- `air_date` = week (YYYYMMDD, a Sunday) + index of the non-underscore char in day_mask
  (0=Sun ... 6=Sat).
- HHMM -> minutes; start >= 2400 means next calendar day (broadcast day 03:00–26:59).
  If end < start, the slot wraps: add 24h to end.
- Slot window = [start:00, end:59].
- `rate_usd` = rate; `rate_aed` = round(rate x 3.6725, 2). Assert rate_aed is a whole number.
- Program_name -> title_en, season, title_ar, is_rerun (`®`), is_live (`LIVE`).
- Overlaps: keep both rows but flag; optimizer may buy at most one of two overlapping rows
  on the same channel/date (they are the same airtime).
- Output: `data/processed/grid.parquet` with a stable `slot_id` (channel|air_date|start).

## B. eTAM break report (have Sep 1–21)
Layout: sheet "Layout 1"; row 1 = target label above the metric columns; row 2 = headers;
column A blank. Columns: Media (As Selected), Program Name, Episode Title, Date, Channel,
Start Time, End Time, Type, Rerun, Rating %, TRP Absolute, Unduplicated Reach,
Unduplicated Reach %.

Parsing rules:
- Read target from row 1. If multiple targets appear (repeated metric blocks), unpivot to
  long format with a `target` column.
- Channel: strip " (M)".
- Times: `datetime.time` -> seconds; `timedelta` -> total seconds (may exceed 86,400 = after
  midnight on the same broadcast day). Keep `broadcast_date` = Date column as given.
- `break_sec` = end - start + 1.
- **`rating_abs` = TRP_Absolute x 60 / break_sec** (per-spot average audience).
- `rating_pct_exact` = rating_abs / universe x 100 (more precise than the rounded Rating %).
- Flags: `is_event` (live sport / one-off specials), `low_sample_channel`
  (share of zero-rated breaks > 50%), `reach_lt_rating`.
- Output: `data/processed/breaks.parquet`.

## C. Still to be extracted by the client (drop into data/incoming/)
Build the pipeline so each of these plugs in without code changes:
1. **August 1–31 and September 22–30** — identical layout to B. (Highest priority.)
2. **Universe + Sample Size** for each target, per week (eTAM data types "Universe",
   "Sample Size").
3. **Reach data** — one of:
   - Option A (EXACT): respondent-level: panelist_id, date, daily_weight (every day),
     demographics, channel, viewing start/end to the second.
   - Option B (CALIBRATED): (i) Duplication Cume Reach + Exclusive Cume Reach for every pair
     of channel x daypart cells; (ii) Cume Reach RF, Reach N+ (1+..5+), Incremental Reach and
     GRP for 30–50 test schedules built in eTAM's schedule tool; (iii) Average Daily Reach and
     Average Weekly Reach by channel x daypart.
4. **eTAM Options settings**: reach minimum-seconds threshold, spot-exposure rule, common
   weight rule (average vs middle-day), Rate Duration Factors table.
5. **MBC 1 grid for week 20261025** and (if needed) Oct 1–3 for all channels.
6. Optional: spot-level log (Report type Spots) for all advertisers on the 7 channels, with
   GRP Absolute per spot and position in break — improves spot-vs-break accuracy.
