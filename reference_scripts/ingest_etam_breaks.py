"""Reference parser for eTAM break reports ("Layout 1"), verified on Sep 1-21 file.
Harden into src/optimizer/ingest/."""
import sys
import datetime as dt
import numpy as np
import pandas as pd

COLS = ["media", "program", "episode", "date", "channel", "start", "end", "type",
        "rerun", "rating_pct", "trp_abs", "reach_abs", "reach_pct"]


def to_seconds(x):
    if isinstance(x, dt.time):
        return x.hour * 3600 + x.minute * 60 + x.second
    if isinstance(x, (dt.timedelta, pd.Timedelta)):
        return int(pd.Timedelta(x).total_seconds())   # >86400 = after midnight, same broadcast day
    return np.nan


def load_breaks(path: str) -> pd.DataFrame:
    head = pd.read_excel(path, header=None, nrows=1)
    target = next(v for v in head.iloc[0].tolist() if isinstance(v, str))
    raw = pd.read_excel(path, header=1)
    df = raw.iloc[:, 1:14].copy()
    df.columns = COLS
    df = df[df["channel"].notna() & df["start"].notna()].copy()
    df["target"] = target
    df["channel"] = df["channel"].str.replace(" (M)", "", regex=False).str.upper().str.strip()
    df["broadcast_date"] = pd.to_datetime(df["date"])
    df["start_sec"] = df["start"].map(to_seconds)
    df["end_sec"] = df["end"].map(to_seconds)
    df["break_sec"] = df["end_sec"] - df["start_sec"] + 1
    assert (df["break_sec"] > 0).all()
    # Per-spot audience. TRP Absolute = Rating Absolute x break minutes in this report.
    df["rating_abs"] = df["trp_abs"] * 60 / df["break_sec"]
    df["weekday"] = df["broadcast_date"].dt.day_name().str[:3]
    df["reach_lt_rating"] = df["reach_pct"] < df["rating_pct"]
    df["first_run"] = df["rerun"].eq("FIRST RUN")
    return df


def infer_universe(df: pd.DataFrame) -> float:
    d = df[(df.rating_pct >= 0.3) & (df.reach_pct > 0)]
    return float((d.reach_abs / (d.reach_pct / 100)).median())


def trp_invariant(df: pd.DataFrame, U: float) -> pd.Series:
    d = df[df.rating_pct >= 0.3]
    return d.trp_abs / (d.rating_pct / 100 * U * d.break_sec / 60)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/etam/etam_breaks_2026-09-01_to_09-21_arabs15plus.xlsx"
    df = load_breaks(path)
    U = infer_universe(df)
    r = trp_invariant(df, U)
    print("target", df.target.iloc[0], "rows", len(df), "dates", df.broadcast_date.min().date(), df.broadcast_date.max().date())
    print("universe_inferred", round(U))
    print("TRP invariant ratio median %.5f sd %.4f n %d" % (r.median(), r.std(), len(r)))
    print(df.groupby("channel").agg(breaks=("rating_abs", "size"),
                                    zero_share=("rating_pct", lambda s: (s == 0).mean()),
                                    mean_aud=("rating_abs", "mean")).round(3))
