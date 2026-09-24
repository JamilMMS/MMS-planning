"""Naive slot match used for PRELIMINARY_FINDINGS section 3 (NOT the forecast):
mean rating_abs of September breaks on same channel + weekday whose start falls in the slot window."""
import pandas as pd
from ingest_grid import load_grid
from ingest_etam_breaks import load_breaks

g = load_grid("data/raw/grid/october_grid.xlsx")
g = g[g.week >= 20261004].copy()
b = load_breaks("data/raw/etam/etam_breaks_2026-09-01_to_09-21_arabs15plus.xlsx")
b["start_min"] = b.start_sec / 60
aud = []
for r in g.itertuples():
    m = b[(b.channel == r.Station_name) & (b.weekday == r.weekday) &
          (b.start_min >= r.start_min) & (b.start_min < r.end_min)]
    aud.append(m.rating_abs.mean() if len(m) else float("nan"))
g["naive_aud"] = aud
g["naive_cpm"] = g.rate_usd / (g.naive_aud / 1000)
print(g.groupby(["Station_name", "tags"]).agg(slots=("slot_id", "size"),
      median_aud=("naive_aud", "median"), median_cpm=("naive_cpm", "median")).round(1))
print("unmatched slots", int(g.naive_aud.isna().sum()))
