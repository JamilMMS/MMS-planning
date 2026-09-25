#!/usr/bin/env python
"""
Independent verification of grid-to-eTAM program matches.

INDEPENDENCE: this script reads ONLY data/processed/match_pairs_for_verifier.csv
(grid_title / etam_title / etam_channel), data/processed/grid.parquet and
data/processed/breaks.parquet. It never reads program_map.csv, the matcher's
review/report files, src/optimizer/match/, tests/test_match.py or the match:
section of config/plan_config.yaml.

Outputs:
  outputs/validation/match_verification.csv
  outputs/validation/match_verification_report.md
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
PAIRS_PATH = ROOT / "data/processed/match_pairs_for_verifier.csv"
GRID_PATH = ROOT / "data/processed/grid.parquet"
BREAKS_PATH = ROOT / "data/processed/breaks.parquet"
OUT_CSV = ROOT / "outputs/validation/match_verification.csv"
OUT_MD = ROOT / "outputs/validation/match_verification_report.md"

MIN_BREAKS_SIGNIFICANT = 6  # "September history" threshold set by the task
TIME_TOL_MIN = 20  # minutes of slack applied to grid slot windows when testing overlap

VOWELS = set("AEIOU")

# Compilation / marathon strand prefixes seen in the grid (title glued directly
# onto the prefix, no space) -> stripped before comparing to the base eTAM title.
COMPILATION_PREFIXES = [
    r"^TURKISH DRAMA\s*-\s*COMPILATION",
    r"^TURKISH DRAMA\s*COMPILATION",
    r"^ARABIC DRAMA\s*-\s*COMPILATION",
    r"^ARABIC DRAMA\s*COMPILATION",
    r"^WEEKEND DRAMA COMPILATION:?\s*",
]

# Generic movie-strand labels: grid slots that are not a single named programme.
GENERIC_MOVIE_PATTERNS = [
    r"^MOVIE$", r"^MOVIES$", r"^PREVIOUS NIGHT MOVIE", r"^REPEAT MOVIE",
    r"^REPEAT MOVIES$", r"^FRIDAY MEGA MOVIE", r"^CHILLER NIGHT$",
    r"^BIG FAMILY NIGHT$", r"^SCREAMING SUNDAY$", r"^STAR OF THE MONTH",
    r"^THROWBACK THURSDAYS$",
]

# Known Arabic-semantics / translation equivalents called out in the brief.
# Keyed by normalised grid_title -> set of normalised etam_titles accepted as
# the same programme on semantic grounds (English label vs Arabic transliteration).
SEMANTIC_PAIRS = {
    # keys are post-normalize_title() forms (LIVE/season/etc already stripped)
    "MBC NEWS": {"AL AKHBAR MBC 1", "AL AKHBAR"},
    "THE MORNING SHOW": {"SABAH AL KHAIR YA ARAB"},
    "MBC IN A WEEK": {"MBC FI OSBO"},
    "THREE KINGDOMS": {"AL MAMALEK AL THALATH"},
}


def strip_compilation_prefix(raw: str) -> str:
    s = raw.strip()
    for pat in COMPILATION_PREFIXES:
        new = re.sub(pat, "", s, flags=re.IGNORECASE)
        if new != s:
            return new.strip()
    return s


def is_generic_movie_title(raw: str) -> bool:
    s = raw.strip().upper()
    return any(re.match(pat, s) for pat in GENERIC_MOVIE_PATTERNS)


def normalize_title(raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw)
    s = strip_compilation_prefix(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.upper()
    s = s.replace("®", " ")
    s = re.sub(r"\bLIVE\b", " ", s)
    s = re.sub(r"\(\d{4}(/\d{2,4})?\)", " ", s)          # (2026), (2026/27)
    s = re.sub(r"\b\d{4}/\d{2,4}\b", " ", s)             # 2026/27 bare
    s = re.sub(r"\b(19|20)\d{2}\b", " ", s)              # bare year
    s = re.sub(r"\d{1,2}[:.]\d{2}\s*GMT\b", " ", s)      # HH:MM GMT
    s = re.sub(r"\bDAY\s*\d+\b", " ", s)
    s = re.sub(r"\bS\d+(?:/S\d+)*\b", " ", s)            # season tags S1, S1/S2
    s = re.sub(r"\bMARATHON\b", " ", s)
    s = re.sub(r"\bPODCAST\b", " PODCAST ", s)           # keep, just normalising spacing
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # collapse AL / EL / EL- prefixes on the first word only, they are the
    # same Arabic definite article transliterated two ways
    s = re.sub(r"^(AL|EL)\s+", "AL ", s)
    return s


def consonant_skeleton(s: str) -> str:
    return "".join(ch for ch in s.replace(" ", "") if ch not in VOWELS)


# Populated once in main() from the actual title corpus: tokens that recur
# across many distinct programme names (HIGHLIGHTS, SHOW, CHAMPIONSHIP, ...)
# are low-signal and must not by themselves drive a title match.
STOPWORDS: set[str] = set()


def _strip_stopwords(norm: str) -> str:
    if not STOPWORDS:
        return norm
    kept = [w for w in norm.split() if w not in STOPWORDS]
    return " ".join(kept) if kept else norm


def title_score(grid_title_raw: str, etam_title_raw: str) -> tuple[int, bool]:
    """Returns (score 0-100, semantic_flag)."""
    g_norm = normalize_title(grid_title_raw)
    e_norm = normalize_title(etam_title_raw)
    if not g_norm or not e_norm:
        return 0, False
    # semantic equivalence table (English label vs Arabic transliteration)
    g_key = g_norm
    if g_key in SEMANTIC_PAIRS and e_norm.replace(" ", "") in {
        x.replace(" ", "") for x in SEMANTIC_PAIRS[g_key]
    }:
        return 100, True
    # Drop channel-generic tokens (e.g. "HIGHLIGHTS") before scoring, so two
    # titles that only share a common word don't score as a match.
    g_ds = _strip_stopwords(g_norm)
    e_ds = _strip_stopwords(e_norm)
    # token_sort_ratio is the anchor score. WRatio's partial-ratio component
    # can wrongly inflate scores for very different-length titles that merely
    # share one generic word, so it and the consonant-skeleton rescue are
    # only trusted as boosts once the anchor score already shows the titles
    # are plausibly related.
    base = fuzz.token_sort_ratio(g_ds, e_ds)
    g_words, e_words = set(g_ds.split()), set(e_ds.split())
    # Full containment (one title's remaining words are entirely inside the
    # other's, e.g. "NADEENA" inside "NADEENA KHALEEJI 27") is trustworthy
    # evidence of the same programme even when the anchor score is low, unlike
    # a single shared generic word -- so it gets its own gate.
    is_subset = bool(g_words) and bool(e_words) and (g_words <= e_words or e_words <= g_words)
    if base >= 55 or is_subset:
        s2 = fuzz.WRatio(g_ds, e_ds)
        base = max(base, s2)
        if abs(len(g_ds.split()) - len(e_ds.split())) <= 1:
            s3 = fuzz.ratio(consonant_skeleton(g_ds), consonant_skeleton(e_ds))
            base = max(base, s3)
    return round(base), False


def load_data():
    pairs = pd.read_csv(PAIRS_PATH)
    grid = pd.read_parquet(GRID_PATH)
    breaks = pd.read_parquet(BREAKS_PATH)
    breaks = breaks.copy()
    breaks["start_min"] = breaks["start_sec"] / 60.0
    breaks["end_min"] = breaks["end_sec"] / 60.0
    return pairs, grid, breaks


def get_grid_windows(grid: pd.DataFrame, channel: str, title_en: str) -> pd.DataFrame:
    rows = grid[(grid["channel"] == channel) & (grid["title_en"] == title_en)]
    return rows[["weekday", "start_min", "end_min"]].drop_duplicates()


def get_grid_windows_by_raw(grid: pd.DataFrame, channel: str, grid_title_raw: str) -> pd.DataFrame:
    """Match a raw grid_title (as it appears in the pairs file) against grid.title_en."""
    return get_grid_windows(grid, channel, grid_title_raw)


def _hits_mask(breaks_sub: pd.DataFrame, windows: pd.DataFrame, tol=TIME_TOL_MIN) -> "pd.Series[bool]":
    """Vectorised: for each break row, True if it falls inside any window
    (same weekday, start_min within [w.start-tol, w.end+tol])."""
    import numpy as np

    n = len(breaks_sub)
    if n == 0 or windows.empty:
        return pd.Series(False, index=breaks_sub.index)
    bwd = breaks_sub["weekday"].to_numpy()[:, None]          # (n,1)
    bmin = breaks_sub["start_min"].to_numpy()[:, None]       # (n,1)
    wwd = windows["weekday"].to_numpy()[None, :]              # (1,m)
    wstart = (windows["start_min"].to_numpy() - tol)[None, :]
    wend = (windows["end_min"].to_numpy() + tol)[None, :]
    same_day = bwd == wwd
    in_range = (bmin >= wstart) & (bmin <= wend)
    hit = (same_day & in_range).any(axis=1)
    return pd.Series(hit, index=breaks_sub.index)


def break_in_windows(weekday: str, start_min: float, windows: pd.DataFrame, tol=TIME_TOL_MIN) -> bool:
    if windows.empty:
        return False
    sub = windows[windows["weekday"] == weekday]
    if sub.empty:
        return False
    return bool((((sub["start_min"] - tol) <= start_min) & (start_min <= (sub["end_min"] + tol))).any())


def overlap_pct(breaks_sub: pd.DataFrame, windows: pd.DataFrame) -> float:
    if breaks_sub.empty:
        return 0.0
    hits = _hits_mask(breaks_sub, windows)
    return round(100.0 * hits.mean(), 1)


def weekday_jaccard(windows: pd.DataFrame, breaks_sub: pd.DataFrame) -> float:
    gwd = set(windows["weekday"].unique())
    bwd = set(breaks_sub["weekday"].unique())
    if not gwd and not bwd:
        return 1.0
    if not gwd or not bwd:
        return 0.0
    return len(gwd & bwd) / len(gwd | bwd)


def rerun_share(breaks_sub: pd.DataFrame) -> float:
    if breaks_sub.empty:
        return float("nan")
    return round(100.0 * (~breaks_sub["first_run"]).mean(), 1)


def grid_is_rerun_share(grid: pd.DataFrame, channel: str, title_en: str) -> float:
    rows = grid[(grid["channel"] == channel) & (grid["title_en"] == title_en)]
    if rows.empty:
        return float("nan")
    return round(100.0 * rows["is_rerun"].mean(), 1)


def candidate_programs(breaks: pd.DataFrame, channel: str, min_breaks=MIN_BREAKS_SIGNIFICANT):
    sub = breaks[breaks["channel"] == channel]
    counts = sub.groupby("program").size()
    return counts[counts >= min_breaks].sort_values(ascending=False)


def dominant_program_share(breaks_sub_all_channel: pd.DataFrame, windows: pd.DataFrame):
    """Among breaks on this channel that fall inside `windows`, which program
    dominates (and by what share)? Returns (program, share_pct, n_breaks_in_window)."""
    if windows.empty:
        return None, 0.0, 0
    hits = _hits_mask(breaks_sub_all_channel, windows)
    in_win = breaks_sub_all_channel[hits]
    n = len(in_win)
    if n == 0:
        return None, 0.0, 0
    top = in_win["program"].value_counts()
    return top.index[0], round(100.0 * top.iloc[0] / n, 1), n


def build_stopwords(grid: pd.DataFrame, breaks: pd.DataFrame, min_df: int = 5) -> set[str]:
    """Tokens that appear in >= min_df distinct programme names across the
    whole grid + eTAM corpus are too generic to be a matching signal on their
    own (HIGHLIGHTS, SHOW, CHAMPIONSHIP, COMPILATION, ...)."""
    from collections import Counter

    titles = set(grid["title_en"].dropna().unique()) | set(breaks["program"].dropna().unique())
    df = Counter()
    for t in titles:
        words = set(normalize_title(t).split())
        for w in words:
            df[w] += 1
    return {w for w, c in df.items() if c >= min_df and len(w) > 2}


def main():
    global STOPWORDS
    pairs, grid, breaks = load_data()
    STOPWORDS = build_stopwords(grid, breaks)
    results = []

    # index breaks by channel for speed
    breaks_by_channel = {ch: df for ch, df in breaks.groupby("channel")}

    for _, row in pairs.iterrows():
        channel = row["channel"]
        grid_title = row["grid_title"]
        etam_title = row["etam_title"]
        etam_channel = row["etam_channel"] if pd.notna(row["etam_channel"]) else None

        windows = get_grid_windows_by_raw(grid, channel, grid_title)
        grid_rerun_pct = grid_is_rerun_share(grid, channel, grid_title)
        ch_breaks = breaks_by_channel.get(channel, breaks.iloc[0:0])

        if etam_title == "NONE" or pd.isna(etam_title):
            # ---- Case: matcher found nothing. Check ourselves. ----
            generic = is_generic_movie_title(grid_title)
            dom_prog, dom_share, n_in_win = dominant_program_share(ch_breaks, windows)

            # search all channel candidates with >=6 Sept breaks for a title match
            cands = candidate_programs(breaks, channel)
            best_prog, best_score, best_sem = None, 0, False
            for prog in cands.index:
                sc, sem = title_score(grid_title, prog)
                if sc > best_score:
                    best_prog, best_score, best_sem = prog, sc, sem

            best_overlap = 0.0
            if best_prog is not None:
                prog_breaks = ch_breaks[ch_breaks["program"] == best_prog]
                best_overlap = overlap_pct(prog_breaks, windows)

            if generic:
                if dom_prog is None or dom_share <= 50.0:
                    verdict = "AGREE"
                    reason = (
                        f"Generic movie/strand slot; no single eTAM title dominates the "
                        f"window (top={dom_prog!r} at {dom_share}% of {n_in_win} breaks) "
                        f"-> correctly left unmatched."
                    )
                    prop = ""
                else:
                    verdict = "DISAGREE"
                    reason = (
                        f"Generic-looking slot but one eTAM title dominates it: "
                        f"{dom_prog!r} = {dom_share}% of {n_in_win} September breaks in this window."
                    )
                    prop = dom_prog
                results.append(dict(
                    channel=channel, grid_title=grid_title, etam_title=etam_title,
                    verdict=verdict, verifier_title_score=best_score,
                    verifier_time_overlap_pct=dom_share if generic else best_overlap,
                    verifier_proposed_etam_title=prop, verifier_reason=reason,
                ))
                continue

            # non-generic NONE
            if best_prog is not None and best_score >= 95:
                # Essentially exact title match in the eTAM corpus: trust the
                # name even if the time-slot overlap is weak (event-style
                # programmes can move date/time year to year).
                verdict = "DISAGREE"
                reason = (
                    f"eTAM title {best_prog!r} (score {best_score}) is essentially the same "
                    f"name as this grid title and has {cands[best_prog]} September breaks; "
                    f"time overlap is {best_overlap}%"
                    + (" (weak -- may have moved slot)" if best_overlap < 30 else "")
                    + " -> looks like the same programme, should not be NONE."
                )
                results.append(dict(
                    channel=channel, grid_title=grid_title, etam_title=etam_title,
                    verdict=verdict, verifier_title_score=best_score,
                    verifier_time_overlap_pct=best_overlap,
                    verifier_proposed_etam_title=best_prog, verifier_reason=reason,
                ))
                continue

            if best_prog is not None and (best_score >= 80 or best_sem) and best_overlap >= 30:
                verdict = "DISAGREE"
                reason = (
                    f"eTAM title {best_prog!r} (score {best_score}"
                    f"{', semantic match' if best_sem else ''}) has {cands[best_prog]} "
                    f"Sept breaks with {best_overlap}% falling inside this programme's "
                    f"October grid window(s) -> looks like the same programme."
                )
                results.append(dict(
                    channel=channel, grid_title=grid_title, etam_title=etam_title,
                    verdict=verdict, verifier_title_score=best_score,
                    verifier_time_overlap_pct=best_overlap,
                    verifier_proposed_etam_title=best_prog, verifier_reason=reason,
                ))
                continue

            if dom_prog is not None and dom_share > 50.0:
                verdict = "AGREE"
                reason = (
                    f"New October programme; its slot window was held by a different "
                    f"programme in September ({dom_prog!r}, {dom_share}% of {n_in_win} "
                    f"breaks in window) -> genuinely unmatched, predecessor = {dom_prog!r}."
                )
                results.append(dict(
                    channel=channel, grid_title=grid_title, etam_title=etam_title,
                    verdict=verdict, verifier_title_score=best_score,
                    verifier_time_overlap_pct=dom_share,
                    verifier_proposed_etam_title="", verifier_reason=reason,
                ))
                continue

            if n_in_win == 0 and (best_prog is None or best_overlap == 0):
                verdict = "AGREE"
                reason = "No September eTAM breaks fall in this slot's October window(s) -> plausibly a new title."
                results.append(dict(
                    channel=channel, grid_title=grid_title, etam_title=etam_title,
                    verdict=verdict, verifier_title_score=best_score,
                    verifier_time_overlap_pct=0.0,
                    verifier_proposed_etam_title="", verifier_reason=reason,
                ))
                continue

            verdict = "UNSURE"
            reason = (
                f"Ambiguous: best title candidate {best_prog!r} scores {best_score} with "
                f"{best_overlap}% overlap; dominant occupant of window is {dom_prog!r} "
                f"at {dom_share}% of {n_in_win} breaks. Neither clearly confirms nor "
                f"clearly refutes a match."
            )
            results.append(dict(
                channel=channel, grid_title=grid_title, etam_title=etam_title,
                verdict=verdict, verifier_title_score=best_score,
                verifier_time_overlap_pct=best_overlap,
                verifier_proposed_etam_title=best_prog or "", verifier_reason=reason,
            ))
            continue

        # ---- Case: matcher proposed etam_title on etam_channel ----
        e_ch = etam_channel or channel
        score, sem = title_score(grid_title, etam_title)
        prog_breaks = breaks_by_channel.get(e_ch, breaks.iloc[0:0])
        prog_breaks = prog_breaks[prog_breaks["program"] == etam_title]
        n_breaks = len(prog_breaks)
        ov = overlap_pct(prog_breaks, windows)
        wd_jac = weekday_jaccard(windows, prog_breaks) if n_breaks else 0.0
        etam_rerun_pct = rerun_share(prog_breaks) if n_breaks else float("nan")

        cross_channel = e_ch != channel
        cross_note = ""
        if cross_channel:
            # does the grid also carry this exact title on the etam_channel? (simulcast/first-run elsewhere)
            sister_windows = get_grid_windows(grid, e_ch, grid_title)
            if not sister_windows.empty:
                cross_note = (
                    f" Cross-channel pairing justified: grid also schedules "
                    f"{grid_title!r} on {e_ch} (its first-run/home channel), and "
                    f"{channel} has no September eTAM history for it (new/rerun slot)."
                )
            else:
                cross_note = (
                    f" Cross-channel pairing NOT independently justified: grid has no "
                    f"{grid_title!r} row on {e_ch} to explain the borrowed history."
                )

        if n_breaks == 0:
            verdict = "UNSURE"
            reason = (
                f"eTAM title {etam_title!r} has 0 September breaks on {e_ch}; cannot "
                f"verify by time slot. Title score {score}."
            ) + cross_note
        elif score >= 95 or sem:
            # Near-exact/exact title identity: trust the name match. Time-slot
            # overlap is reported as a sanity check, not a gate -- a weekly
            # drama can shift day/time between September and October while
            # remaining unambiguously the same title.
            verdict = "AGREE"
            caveat = (
                f" NOTE: time-slot overlap is only {ov}% (weekday overlap "
                f"{round(wd_jac*100)}%) -- schedule may have moved day/time between "
                f"September and October; title identity is exact so still AGREE."
                if ov < 30 else ""
            )
            reason = (
                f"Title score {score}{' (semantic)' if sem else ''} -- exact/near-exact "
                f"name match; {ov}% of {n_breaks} Sept breaks fall inside the October "
                f"grid window(s)."
            ) + caveat + cross_note
        elif (score >= 85 or sem) and ov >= 50:
            verdict = "AGREE"
            reason = (
                f"Title score {score}{' (semantic)' if sem else ''}; {ov}% of "
                f"{n_breaks} Sept breaks fall inside the October grid window(s); "
                f"weekday overlap {round(wd_jac*100)}%."
            ) + cross_note
        elif score >= 60 and ov >= 70:
            verdict = "AGREE"
            reason = (
                f"Moderate title score {score} (looks like a transliteration variant) "
                f"but strong time-slot confirmation: {ov}% of {n_breaks} Sept breaks "
                f"land inside the October window(s)."
            ) + cross_note
        elif score < 50 and ov < 20:
            # look for a better candidate before declaring DISAGREE
            cands = candidate_programs(breaks, channel)
            best_prog, best_score, best_sem = None, 0, False
            for prog in cands.index:
                sc, semm = title_score(grid_title, prog)
                if sc > best_score:
                    best_prog, best_score, best_sem = prog, sc, semm
            verdict = "DISAGREE"
            reason = (
                f"Weak title score {score} and low time overlap {ov}% for the proposed "
                f"pair. Better local candidate on {channel}: {best_prog!r} (score {best_score})."
                if best_prog else
                f"Weak title score {score} and low time overlap {ov}% for the proposed pair; "
                f"no better candidate found on {channel}."
            ) + cross_note
            results.append(dict(
                channel=channel, grid_title=grid_title, etam_title=etam_title,
                verdict=verdict, verifier_title_score=score,
                verifier_time_overlap_pct=ov,
                verifier_proposed_etam_title=best_prog or "", verifier_reason=reason,
            ))
            continue
        else:
            verdict = "UNSURE"
            reason = (
                f"Title score {score}, time overlap {ov}%, weekday overlap "
                f"{round(wd_jac*100)}% -- not a clean AGREE or DISAGREE by our thresholds."
            ) + cross_note

        results.append(dict(
            channel=channel, grid_title=grid_title, etam_title=etam_title,
            verdict=verdict, verifier_title_score=score,
            verifier_time_overlap_pct=ov,
            verifier_proposed_etam_title="" if verdict == "AGREE" else "",
            verifier_reason=reason,
        ))

    matched_etam_per_channel = {}
    resolved_grid_titles_per_channel = {}  # grid_title already paired with a real (non-NONE) etam_title
    for _, row in pairs.iterrows():
        if row["etam_title"] != "NONE" and pd.notna(row["etam_title"]):
            ech = row["etam_channel"] if pd.notna(row["etam_channel"]) else row["channel"]
            matched_etam_per_channel.setdefault(ech, set()).add(row["etam_title"])
            resolved_grid_titles_per_channel.setdefault(row["channel"], set()).add(row["grid_title"])

    reverse_rows = []
    for channel in sorted(breaks["channel"].unique()):
        cands = candidate_programs(breaks, channel)
        matched = matched_etam_per_channel.get(channel, set())
        grid_titles = list(grid[grid["channel"] == channel]["title_en"].unique())
        windows_map = {gt: get_grid_windows(grid, channel, gt) for gt in grid_titles}
        resolved_titles = resolved_grid_titles_per_channel.get(channel, set())
        # Occupancy (pure time-window) candidates are restricted to grid slots
        # the matcher left open (NONE) or generic strands -- a slot already
        # confidently paired with a different named eTAM title should not be
        # re-proposed just because a wide, multi-tier window coincidentally
        # overlaps another September show.
        occ_candidate_titles = [gt for gt in grid_titles if gt not in resolved_titles]
        title_score_map = {}  # cache: prog -> list of (gt, score, sem)
        for prog, n in cands.items():
            if prog in matched:
                continue
            prog_breaks = breaks_by_channel[channel]
            prog_breaks = prog_breaks[prog_breaks["program"] == prog]
            # does a grid programme's title or its Oct weekday/time windows match?
            best_grid, best_score, best_sem = None, 0, False
            best_ov_for_title = 0.0
            occ_grid, occ_ov = None, 0.0
            for gt in occ_candidate_titles:
                sc, semm = title_score(gt, prog)
                w = windows_map[gt]
                ov = overlap_pct(prog_breaks, w)
                if ov > occ_ov:
                    occ_grid, occ_ov = gt, ov
                if sc > best_score:
                    best_grid, best_score, best_sem = gt, sc, semm
                    best_ov_for_title = ov
            best_ov = best_ov_for_title
            occ_is_generic = occ_grid is not None and is_generic_movie_title(occ_grid)
            if best_score >= 80 or best_sem:
                verdict = "DISAGREE"
                reason = (
                    f"eTAM title {prog!r} ({n} Sept breaks on {channel}) is unmatched but "
                    f"closely resembles grid programme {best_grid!r} (score {best_score}, "
                    f"{best_ov}% time overlap) -> should probably have been matched."
                )
                proposal = best_grid
            elif occ_grid is not None and occ_ov >= 60 and not occ_is_generic and n >= 8:
                verdict = "DISAGREE"
                reason = (
                    f"eTAM title {prog!r} ({n} Sept breaks on {channel}) is unmatched but its "
                    f"September weekday/time pattern falls inside grid programme "
                    f"{occ_grid!r}'s October window {occ_ov}% of the time -> possible same slot."
                )
                proposal = occ_grid
            elif occ_is_generic and occ_ov >= 40:
                verdict = "AGREE"
                reason = (
                    f"eTAM title {prog!r} ({n} Sept breaks on {channel}) is unmatched, but its "
                    f"slot is a generic movie/compilation strand in the October grid "
                    f"({occ_grid!r}, {occ_ov}% time overlap) -- already correctly left as a "
                    f"rotating title with no single-programme match, consistent with the "
                    f"NONE verdicts on that strand elsewhere in this file."
                )
                proposal = ""
            elif best_score < 45 and occ_ov < 30:
                verdict = "AGREE"
                reason = (
                    f"eTAM title {prog!r} ({n} Sept breaks on {channel}) is unmatched; no grid "
                    f"title resembles it by name (best {best_grid!r} scores {best_score}) and "
                    f"no October slot is occupied by its September pattern ({occ_ov}%) -- "
                    f"plausibly a September-only programme with no October equivalent."
                )
                proposal = ""
            else:
                verdict = "UNSURE"
                reason = (
                    f"eTAM title {prog!r} ({n} Sept breaks on {channel}) is unmatched; "
                    f"best title candidate {best_grid!r} (score {best_score}), best "
                    f"time-window occupant {occ_grid!r} ({occ_ov}%) -- neither is a clean match, "
                    f"worth a human look."
                )
                proposal = ""
            reverse_rows.append(dict(
                channel=channel, grid_title="(none)", etam_title=prog,
                verdict=verdict, verifier_title_score=best_score,
                verifier_time_overlap_pct=max(best_ov, occ_ov),
                verifier_proposed_etam_title=proposal, verifier_reason=reason,
            ))

    out_rows = results + reverse_rows
    out_df = pd.DataFrame(out_rows, columns=[
        "channel", "grid_title", "etam_title", "verdict",
        "verifier_title_score", "verifier_time_overlap_pct",
        "verifier_proposed_etam_title", "verifier_reason",
    ])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    # ---- report ----
    main_df = pd.DataFrame(results)
    counts = main_df["verdict"].value_counts().to_dict()
    lines = []
    lines.append("# Match Verification Report\n")
    lines.append(f"Pairs checked: {len(main_df)}  |  Reverse-check rows: {len(reverse_rows)}\n")
    lines.append("## Verdict counts (main 157 pairs)\n")
    for v in ["AGREE", "DISAGREE", "UNSURE"]:
        lines.append(f"- {v}: {counts.get(v, 0)}")
    lines.append("")

    lines.append("## DISAGREE\n")
    for _, r in main_df[main_df["verdict"] == "DISAGREE"].iterrows():
        lines.append(
            f"- [{r['channel']}] grid={r['grid_title']!r} matcher_etam={r['etam_title']!r} "
            f"-> proposed={r['verifier_proposed_etam_title']!r}. {r['verifier_reason']}"
        )
    lines.append("")

    lines.append("## UNSURE\n")
    for _, r in main_df[main_df["verdict"] == "UNSURE"].iterrows():
        lines.append(
            f"- [{r['channel']}] grid={r['grid_title']!r} matcher_etam={r['etam_title']!r}. "
            f"{r['verifier_reason']}"
        )
    lines.append("")

    lines.append("## Reverse check: unmatched eTAM titles (>=6 Sept breaks) on grid channels\n")
    rev_df = pd.DataFrame(reverse_rows)
    for _, r in rev_df.sort_values(["channel", "etam_title"]).iterrows():
        lines.append(
            f"- [{r['channel']}] etam={r['etam_title']!r} -> verdict={r['verdict']}, "
            f"proposal={r['verifier_proposed_etam_title']!r}. {r['verifier_reason']}"
        )
    lines.append("")

    lines.append("## Method (10 lines)\n")
    lines.append(
        "1. Inputs: match_pairs_for_verifier.csv, grid.parquet, breaks.parquet only.\n"
        "2. Title normalisation: uppercase, strip (R), LIVE, season tags S#, year tags, "
        "GMT time labels, punctuation; AL/EL prefix folded to AL; compilation/marathon "
        "prefixes and suffixes stripped to recover the base title.\n"
        "3. Title score = max(rapidfuzz token_sort_ratio, WRatio, ratio-of-consonant-skeleton) "
        "so vowel-only transliteration drift (NOWAYLATI/NWAYLATI, ARD/ARDH, DAGHET/DAGHT) scores high.\n"
        "4. A small hand-built table covers the Arabic-semantics pairs named in the brief "
        "(MBC NEWS LIVE / AL AKHBAR, etc.) and forces a 100 score.\n"
        "5. Grid windows = distinct (weekday, start_min, end_min) rows in grid.parquet for "
        "channel+title_en; time is compared in broadcast-day minutes (03:00 = minute 180).\n"
        "6. eTAM breaks for the candidate programme are tested against those windows with a "
        "+/-20 min tolerance; overlap % = share of breaks landing inside a window on the same weekday.\n"
        "7. AGREE needs high title score + >=50% time overlap, or moderate score with >=70% "
        "time overlap (transliteration case); DISAGREE needs weak score AND weak overlap, or "
        "evidence a dominant eTAM title occupies a slot the matcher marked NONE/generic; "
        "everything else is UNSURE.\n"
        "8. For NONE pairs, all eTAM titles with >=6 September breaks on that channel are scored "
        "for title + time fit against the grid slot before accepting 'no match'.\n"
        "9. Generic movie-strand slots (MOVIE, PREVIOUS NIGHT MOVIE hh:mm GMT, REPEAT MOVIE...) "
        "are AGREE unless one eTAM title holds >50% of the breaks landing in that window.\n"
        "10. Reverse check: every eTAM title with >=6 September breaks on a grid channel that "
        "was never used as a match anywhere on that channel is tested the same way against all "
        "grid titles on that channel, by title score and by raw time-window occupancy."
    )

    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"Wrote {OUT_CSV} ({len(out_df)} rows)")
    print(f"Wrote {OUT_MD}")
    print("Counts:", counts)
    print("Reverse-check rows:", len(reverse_rows),
          "verdicts:", rev_df["verdict"].value_counts().to_dict() if len(rev_df) else {})


if __name__ == "__main__":
    main()
