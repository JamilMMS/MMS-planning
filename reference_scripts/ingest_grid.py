"""Reference parser for the October grid (verified on the real file, 24 Sep 2026).
Harden into src/optimizer/ingest/ — do not treat as production code."""
import sys
import pandas as pd

AED_PER_USD = 3.6725
DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def hhmm_to_min(t: int) -> int:
    return (t // 100) * 60 + t % 100


def load_grid(path: str) -> pd.DataFrame:
    g = pd.read_excel(path)
    g["Station_name"] = g["Station_name"].str.upper().str.strip()
    masks = g["day_mask"].map(lambda m: [i for i, c in enumerate(m) if c != "_"])
    assert masks.map(len).eq(1).all(), "day_mask must contain exactly one day"
    g["day_idx"] = masks.str[0]                       # 0 = Sunday
    g["weekday"] = g["day_idx"].map(lambda i: DAYS[i])
    week_start = pd.to_datetime(g["week"].astype(str), format="%Y%m%d")
    g["air_date"] = week_start + pd.to_timedelta(g["day_idx"], unit="D")   # broadcast date
    g["start_min"] = g["start_time"].map(hhmm_to_min)                      # minutes from broadcast-day 00:00
    g["end_min"] = g["end_time"].map(hhmm_to_min) + 1                      # exclusive end
    wrap = g["end_min"] <= g["start_min"]
    g.loc[wrap, "end_min"] += 24 * 60
    g["wraps_day"] = wrap
    g["slot_minutes"] = g["end_min"] - g["start_min"]
    g["start_dt"] = g["air_date"] + pd.to_timedelta(g["start_min"], unit="m")  # real clock datetime
    g["rate_usd"] = g["rate"]
    g["rate_aed"] = (g["rate"] * AED_PER_USD).round(2)
    assert (g["rate_aed"] - g["rate_aed"].round(0)).abs().max() < 0.01, "AED rates not integral"
    pn = g["Program_name"].astype(str)
    g["is_rerun"] = pn.str.contains("®")
    g["is_live"] = pn.str.upper().str.contains("LIVE")
    parts = pn.str.replace("®", "", regex=False).str.split(" / ", n=1, expand=True)
    g["title_en_raw"] = parts[0].str.strip()
    g["title_ar"] = parts[1].str.strip() if parts.shape[1] > 1 else None
    g["season"] = g["title_en_raw"].str.extract(r"\b(S\d+(?:/S\d+)?)\b", expand=False)
    g["title_en"] = g["title_en_raw"].str.replace(r"\bS\d+(?:/S\d+)?\b", "", regex=True).str.strip()
    g["slot_id"] = g["Station_name"] + "|" + g["air_date"].dt.strftime("%Y-%m-%d") + "|" + g["start_time"].astype(str).str.zfill(4)
    assert g["slot_id"].is_unique
    return g


def find_overlaps(g: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, x in g.sort_values("start_min").groupby(["Station_name", "air_date"]):
        rows = list(x.itertuples())
        for a, b in zip(rows, rows[1:]):
            if b.start_min < a.end_min:
                out.append((a.slot_id, b.slot_id))
    return pd.DataFrame(out, columns=["slot_a", "slot_b"])


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/grid/october_grid.xlsx"
    g = load_grid(path)
    print("rows", len(g))
    print(pd.crosstab(g["Station_name"], g["week"]))
    print("overlaps", len(find_overlaps(g)))
    print("reruns", int(g.is_rerun.sum()), "live", int(g.is_live.sum()))
    print("total cost USD", round(g.rate_usd.sum(), 2))
