"""Ingest eTAM break reports ("Layout 1" sheet).

Hardened from ``reference_scripts/ingest_etam_breaks.py`` per ``docs/DATA_SPEC.md`` section B.
Never modifies ``data/raw/``.

Layout (DATA_SPEC.B): sheet "Layout 1"; row 1 = target label(s) above the metric columns;
row 2 = headers; column A blank. The current file has a single target ("TP Arabs 15+") but
the format supports several *repeated metric blocks* -- each block is the same 4 metric
columns (Rating %, TRP Absolute, Unduplicated Reach, Unduplicated Reach %) with its own
target label in row 1, sharing one set of descriptive columns (Media, Program, Episode,
Date, Channel, Start, End, Type, Rerun). We detect blocks by locating every run of 4
consecutive row-2 headers equal to that metric-name sequence, rather than assuming a fixed
column count, so a file with 2+ targets is unpivoted into long format with a ``target``
column with no code changes.

Non-negotiable rule (CLAUDE.md): TRP Absolute is NOT spot impressions.
``rating_abs`` (per-spot audience) = ``trp_abs * 60 / break_sec``.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DESC_NAMES = ["media", "program", "episode", "date", "channel", "start", "end", "type", "rerun"]
METRIC_HEADER_SEQ = ["Rating %", "TRP Absolute", "Unduplicated Reach", "Unduplicated Reach %"]
METRIC_NAMES = ["rating_pct", "trp_abs", "reach_abs", "reach_pct"]
DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

# Event-window growth (etam.event_policy == "window"): once a channel-day's simulcast-anchor
# breaks are found, we grow the excluded window outward through breaks whose rating_abs still
# exceeds `event_window_ratio` x the channel's own same-hour baseline, but only while
# consecutive breaks stay close together in time. Real within-programme break gaps in this
# data run ~150-900s; a gap this much larger marks an actual programme/audience-level boundary
# (verified against the 19-Sep MBC ACTION event: the ~52min gap before 73021s is exactly where
# growth should stop -- see etam_breaks_report.md "Event window" section for the worked example).
EVENT_WINDOW_MAX_GAP_SEC = 45 * 60


def to_seconds(x: Any) -> float:
    """``datetime.time`` -> seconds since broadcast-day 00:00. ``timedelta`` (pandas renders
    "after midnight, same broadcast day" times as e.g. "1 day, 0:41:34") -> total seconds,
    which may exceed 86,400. Anything else -> NaN (logged as a parse failure by the caller)."""
    if isinstance(x, dt.time):
        return x.hour * 3600 + x.minute * 60 + x.second
    if isinstance(x, (dt.timedelta, pd.Timedelta)):
        return float(pd.Timedelta(x).total_seconds())
    return np.nan


def _find_metric_blocks(row1: list[Any]) -> list[int]:
    """Column indices where a 4-column run matching METRIC_HEADER_SEQ starts."""
    starts = []
    n = len(row1)
    for i in range(n - len(METRIC_HEADER_SEQ) + 1):
        window = [str(row1[i + j]).strip() for j in range(len(METRIC_HEADER_SEQ))]
        if window == METRIC_HEADER_SEQ:
            starts.append(i)
    return starts


def _target_for_block(row0: list[Any], start: int, next_start: int | None) -> str:
    """The target label in row 1 (0-indexed row0) that applies to the metric block starting
    at column `start`: the first non-null string found at or after `start` and before the
    next block's start (or end of row)."""
    hi = next_start if next_start is not None else len(row0)
    for v in row0[start:hi]:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "UNKNOWN_TARGET"


def _event_policy_for(channel: str, event_policy_cfg: dict[str, Any]) -> str:
    """Per-channel event-exclusion policy from cfg.etam.event_policy: 'window' (exclude the
    whole anomalous audience window -- pre/post-match spillover included) or 'match_only'
    (exclude just the simulcast-detected rows themselves). Falls back to the config's
    'default' entry, then to 'match_only' if the key is absent entirely."""
    if channel in event_policy_cfg:
        return str(event_policy_cfg[channel]).lower()
    return str(event_policy_cfg.get("default", "match_only")).lower()


def _grow_event_window(
    day_breaks: pd.DataFrame,
    anchor_start: int,
    anchor_end: int,
    hour_baseline: dict[int, float],
    ratio_threshold: float,
    max_gap_sec: int = EVENT_WINDOW_MAX_GAP_SEC,
) -> pd.DataFrame:
    """Grow a simulcast anchor (the match breaks themselves, `[anchor_start, anchor_end]`)
    outward, on one channel-day's breaks sorted by start_sec, to the full anomalous-audience
    window: keep including the next/previous break while (a) it is still elevated -- its
    rating_abs exceeds `ratio_threshold` x that hour's baseline for the channel -- and (b) the
    gap since the window's current edge is <= `max_gap_sec` (a real programme boundary, not
    just a normal inter-break gap). Returns the sub-frame of `day_breaks` in the resulting
    window (inclusive of the anchor)."""
    day_breaks = day_breaks.sort_values("start_sec").reset_index(drop=True)

    def ratio_of(row) -> float:
        base = hour_baseline.get(int(row.start_sec // 3600), 0.0)
        if base > 0:
            return row.rating_abs / base
        return float("inf") if row.rating_abs > 0 else 0.0

    in_anchor = day_breaks.index[
        (day_breaks["start_sec"] >= anchor_start) & (day_breaks["start_sec"] <= anchor_end)
    ]
    if len(in_anchor) == 0:
        return day_breaks.iloc[0:0]
    idx_lo, idx_hi = int(in_anchor.min()), int(in_anchor.max())
    n = len(day_breaks)

    while idx_hi + 1 < n:
        nxt = day_breaks.iloc[idx_hi + 1]
        gap = nxt.start_sec - day_breaks.iloc[idx_hi].end_sec
        if gap <= max_gap_sec and ratio_of(nxt) > ratio_threshold:
            idx_hi += 1
        else:
            break
    while idx_lo - 1 >= 0:
        prv = day_breaks.iloc[idx_lo - 1]
        gap = day_breaks.iloc[idx_lo].start_sec - prv.end_sec
        if gap <= max_gap_sec and ratio_of(prv) > ratio_threshold:
            idx_lo -= 1
        else:
            break

    return day_breaks.iloc[idx_lo : idx_hi + 1]


def load_breaks(path: str | Path, cfg: dict[str, Any]) -> pd.DataFrame:
    """Load and fully parse an eTAM break report per DATA_SPEC section B.

    Parameters
    ----------
    path: path to the eTAM .xlsx (read-only).
    cfg: the loaded ``config/plan_config.yaml`` dict.

    Returns
    -------
    Long-format DataFrame, one row per (break, target). See module docstring / DELIVER
    item 2 in the data-validator role for the full column contract.
    """
    universe = float(cfg["target"]["universe"])
    zero_share_thresh = float(cfg["etam"]["low_sample_zero_share"])
    simulcast_tol_min = float(cfg["etam"].get("event_simulcast_tolerance_min", 10))
    anomaly_ratio = float(cfg["etam"].get("event_anomaly_ratio", 3.0))

    head = pd.read_excel(path, header=None, nrows=2)
    row0 = head.iloc[0].tolist()
    row1 = head.iloc[1].tolist()
    blocks = _find_metric_blocks(row1)
    if not blocks:
        raise ValueError(f"{path}: could not find the expected metric-column header sequence")
    desc_cols = list(range(1, blocks[0]))  # column 0 is the blank column A
    if len(desc_cols) != len(DESC_NAMES):
        raise ValueError(
            f"{path}: expected {len(DESC_NAMES)} descriptive columns before the first metric "
            f"block, found {len(desc_cols)}"
        )

    body = pd.read_excel(path, header=None, skiprows=2)

    frames = []
    for bi, start in enumerate(blocks):
        next_start = blocks[bi + 1] if bi + 1 < len(blocks) else None
        target = _target_for_block(row0, start, next_start)
        cols = desc_cols + [start, start + 1, start + 2, start + 3]
        block = body.iloc[:, cols].copy()
        block.columns = DESC_NAMES + METRIC_NAMES
        block["target"] = target
        frames.append(block)
    df = pd.concat(frames, ignore_index=True)

    # Drop fully-blank trailing rows (no channel and no start time).
    n_before = len(df)
    df = df[df["channel"].notna() & df["start"].notna()].copy()
    n_dropped_blank = n_before - len(df)

    # --- defensive string casts: at least one program/episode cell in the real file is a
    # bare number (a movie titled by a number). Cast to str before any .str op. ---
    n_nonstr_program = int(df["program"].map(lambda x: not isinstance(x, str)).sum())
    n_nonstr_episode = int(df["episode"].map(lambda x: not isinstance(x, str) and pd.notna(x)).sum())
    df["program"] = df["program"].astype(str)
    df["episode"] = df["episode"].astype(object).where(df["episode"].notna(), None)
    df["episode"] = df["episode"].map(lambda x: x if x is None else str(x))
    df["media"] = df["media"].astype(str)

    df["source_file"] = Path(path).name
    df["channel"] = df["channel"].astype(str).str.replace(" (M)", "", regex=False).str.upper().str.strip()
    df["broadcast_date"] = pd.to_datetime(df["date"])

    df["start_sec"] = df["start"].map(to_seconds)
    df["end_sec"] = df["end"].map(to_seconds)
    n_bad_time = int(df["start_sec"].isna().sum() + df["end_sec"].isna().sum())
    df = df[df["start_sec"].notna() & df["end_sec"].notna()].copy()
    df["start_sec"] = df["start_sec"].astype("int64")
    df["end_sec"] = df["end_sec"].astype("int64")
    df["break_sec"] = df["end_sec"] - df["start_sec"] + 1
    assert (df["break_sec"] > 0).all(), "break_sec must be > 0 for every row"

    df["start_dt"] = df["broadcast_date"] + pd.to_timedelta(df["start_sec"], unit="s")
    df["end_dt"] = df["broadcast_date"] + pd.to_timedelta(df["end_sec"], unit="s")
    df["weekday"] = df["broadcast_date"].dt.dayofweek.map(lambda i: DAYS[(i + 1) % 7])

    df["rating_pct"] = df["rating_pct"].astype("float64")
    df["trp_abs"] = df["trp_abs"].astype("float64")
    df["reach_abs"] = df["reach_abs"].astype("float64")
    df["reach_pct"] = df["reach_pct"].astype("float64")

    # THE non-negotiable rule: TRP Absolute is NOT per-spot audience.
    df["rating_abs"] = df["trp_abs"] * 60 / df["break_sec"]
    df["rating_pct_exact"] = df["rating_abs"] / universe * 100
    df["reach_lt_rating"] = df["reach_pct"] < df["rating_pct"]
    df["first_run"] = df["rerun"].astype(str).eq("FIRST RUN")

    df["break_id"] = df["channel"] + "|" + df["broadcast_date"].dt.strftime("%Y-%m-%d") + "|" + df["start_sec"].astype(str)

    # --- low_sample_channel: per-channel share of zero-rated breaks > threshold ---
    zero_share = df.groupby("channel")["rating_pct"].apply(lambda s: (s == 0).mean())
    low_sample_channels = set(zero_share[zero_share > zero_share_thresh].index)
    df["low_sample_channel"] = df["channel"].isin(low_sample_channels)

    # --- is_event: simulcast detection (primary) ---
    # A (program, episode, broadcast_date) group airing on >=2 distinct channels with break
    # start times within `simulcast_tol_min` minutes of each other is treated as a one-off
    # simulcast special (this generalises the known 19-Sep FIFA/NADEENA football special,
    # which aired simultaneously on MBC 1 and MBC ACTION -- see grid_report / etam_breaks
    # report for the full candidate list considered and rejected).
    #
    # Per-channel policy (cfg.etam.event_policy, CLAUDE.md CHANGE 1): the simulcast rows found
    # here are always the "anchor" -- the match breaks themselves. What gets excluded around
    # that anchor then depends on the channel's policy:
    #   - 'match_only' (default): exclude just the anchor rows -- event_reason='SIMULCAST_MATCH'.
    #   - 'window': the anchor is grown outward (see `_grow_event_window`) to the full
    #     anomalous-audience window (pre/post-match spillover included) and every break in that
    #     window is excluded -- event_reason='SIMULCAST_WINDOW'.
    df["is_event"] = False
    df["event_reason"] = ""
    event_policy_cfg = cfg["etam"].get("event_policy", {}) or {}
    event_window_ratio = float(cfg["etam"].get("event_window_ratio", 3.0))
    tol_sec = simulcast_tol_min * 60
    anchor_idx_by_channel_date: dict[tuple[str, Any], set[int]] = {}
    for (_, _, _), grp in df.groupby(["program", "episode", "broadcast_date"], sort=False):
        channels = grp["channel"].unique()
        if len(channels) < 2:
            continue
        starts_by_channel = {c: grp.loc[grp["channel"] == c, "start_sec"].to_numpy() for c in channels}
        chans = list(channels)
        flagged_idx = set()
        for i in range(len(chans)):
            for j in range(i + 1, len(chans)):
                a, b = starts_by_channel[chans[i]], starts_by_channel[chans[j]]
                diffs = np.abs(a[:, None] - b[None, :])
                if (diffs <= tol_sec).any():
                    rows_a = grp.index[grp["channel"] == chans[i]]
                    rows_b = grp.index[grp["channel"] == chans[j]]
                    flagged_idx.update(rows_a)
                    flagged_idx.update(rows_b)
        if flagged_idx:
            df.loc[list(flagged_idx), "is_event"] = True
            for idx in flagged_idx:
                key = (df.at[idx, "channel"], df.at[idx, "broadcast_date"])
                anchor_idx_by_channel_date.setdefault(key, set()).add(idx)

    event_windows: list[dict[str, Any]] = []
    for (channel, event_date), anchor_idx in anchor_idx_by_channel_date.items():
        policy = _event_policy_for(channel, event_policy_cfg)
        anchor_start = int(df.loc[list(anchor_idx), "start_sec"].min())
        anchor_end = int(df.loc[list(anchor_idx), "end_sec"].max())
        if policy == "match_only":
            df.loc[list(anchor_idx), "event_reason"] = "SIMULCAST_MATCH"
            continue
        if policy != "window":
            raise ValueError(
                f"Unknown etam.event_policy value {policy!r} for channel {channel!r} "
                "(expected 'window' or 'match_only')"
            )

        # Same-hour baseline for this channel, computed over every OTHER date (all dates that
        # are not this channel's own anchor date(s) -- a channel can in principle have more
        # than one incident, though only MBC ACTION 19-Sep does today).
        own_anchor_dates = {d for (c, d) in anchor_idx_by_channel_date if c == channel}
        chan_df = df[df["channel"] == channel]
        baseline_df = chan_df[~chan_df["broadcast_date"].isin(own_anchor_dates)]
        hour_baseline = (
            (baseline_df["start_sec"] // 3600).astype(int).to_frame("hour").assign(
                rating_abs=baseline_df["rating_abs"].to_numpy()
            ).groupby("hour")["rating_abs"].median().to_dict()
        )

        day_breaks = chan_df[chan_df["broadcast_date"] == event_date]
        window = _grow_event_window(day_breaks, anchor_start, anchor_end, hour_baseline, event_window_ratio)
        # `window` was produced by reset_index(drop=True) inside _grow_event_window, so map
        # back to the real df index via break_id (unique per channel/date/start_sec) rather
        # than relying on the now-meaningless positional index.
        idx_to_flag = set(df.index[df["break_id"].isin(window["break_id"])])
        df.loc[list(idx_to_flag), "is_event"] = True
        df.loc[list(idx_to_flag), "event_reason"] = "SIMULCAST_WINDOW"

        event_windows.append({
            "channel": channel,
            "broadcast_date": str(pd.Timestamp(event_date).date()),
            "policy": policy,
            "window_start_sec": int(window["start_sec"].min()),
            "window_end_sec": int(window["end_sec"].max()),
            "n_breaks": int(len(window)),
            "programmes": sorted(window["program"].unique().tolist()),
            "event_window_ratio": event_window_ratio,
            "max_gap_sec": EVENT_WINDOW_MAX_GAP_SEC,
        })

    # --- channel-day statistical anomaly, for the report (does NOT alone set is_event) ---
    day_mean = df.groupby(["channel", "broadcast_date"])["rating_abs"].mean().reset_index(name="day_mean")
    chan_median = day_mean.groupby("channel")["day_mean"].median().rename("chan_median")
    day_mean = day_mean.merge(chan_median, on="channel")
    day_mean["ratio"] = day_mean["day_mean"] / day_mean["chan_median"].replace(0, np.nan)
    anomalous_days = day_mean[day_mean["ratio"] > anomaly_ratio]

    df.attrs["n_dropped_blank_rows"] = n_dropped_blank
    df.attrs["n_bad_time_rows"] = n_bad_time
    df.attrs["n_nonstr_program"] = n_nonstr_program
    df.attrs["n_nonstr_episode"] = n_nonstr_episode
    df.attrs["anomalous_channel_days"] = anomalous_days
    df.attrs["event_windows"] = event_windows
    df.attrs["targets"] = sorted({_target_for_block(row0, s, blocks[i + 1] if i + 1 < len(blocks) else None) for i, s in enumerate(blocks)})

    keep_cols = [
        "source_file", "target", "media", "program", "episode", "broadcast_date", "channel",
        "start_sec", "end_sec", "break_sec", "start_dt", "end_dt", "weekday", "type", "rerun",
        "first_run", "rating_pct", "trp_abs", "reach_abs", "reach_pct", "rating_abs",
        "rating_pct_exact", "reach_lt_rating", "is_event", "event_reason", "low_sample_channel",
        "break_id",
    ]
    out = df[keep_cols].reset_index(drop=True)
    out.attrs.update(df.attrs)
    return out


def infer_universe(df: pd.DataFrame) -> dict[str, float]:
    """Median of reach_abs/(reach_pct/100) on rating_pct>=0.3 & reach_pct>0, plus the 5-95%
    range, per DATA_SPEC section C.2."""
    d = df[(df["rating_pct"] >= 0.3) & (df["reach_pct"] > 0)]
    implied = d["reach_abs"] / (d["reach_pct"] / 100)
    return {
        "median": float(implied.median()),
        "p05": float(implied.quantile(0.05)),
        "p95": float(implied.quantile(0.95)),
        "n": int(len(implied)),
    }


def trp_invariant_ratio(df: pd.DataFrame, universe: float) -> pd.Series:
    """trp_abs / (rating_pct/100 * U * break_sec/60), on rating_pct>=0.3 rows. Should be
    ~1.00 -- this is the real check that TRP Absolute = Rating Absolute x break minutes."""
    d = df[df["rating_pct"] >= 0.3]
    return d["trp_abs"] / (d["rating_pct"] / 100 * universe * d["break_sec"] / 60)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from optimizer.config import load_config

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/etam/etam_breaks_2026-09-01_to_09-21_arabs15plus.xlsx"
    cfg = load_config()
    df = load_breaks(path, cfg)
    U = infer_universe(df)
    r = trp_invariant_ratio(df, U["median"])
    print("targets", df.attrs["targets"], "rows", len(df))
    print("dates", df.broadcast_date.min().date(), df.broadcast_date.max().date())
    print("universe_inferred", U)
    print("TRP invariant ratio median %.5f sd %.4f n %d" % (r.median(), r.std(), len(r)))
    print("is_event rows", int(df.is_event.sum()))
    print(df.groupby("channel").agg(breaks=("rating_abs", "size"),
                                    zero_share=("rating_pct", lambda s: (s == 0).mean()),
                                    mean_aud=("rating_abs", "mean")).round(3))
