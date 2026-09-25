"""Grid programme -> eTAM programme matching (Phase 3, Gate 2).

Maps every distinct October grid programme (channel + ``title_en``, in-flight rows including
the synthetic MBC 1 week-4 rows) to the title under which its history appears in the
September eTAM break file — or to ``NONE`` when there is no such history. "new program" is a
valid, expected answer; nothing is ever forced.

Pipeline
--------
1. **Classify the grid label** (rules in ``cfg.match``):
   * ``tbc_patterns`` (TBC/TBA/TBD) -> ``new program`` (content unknown).
   * ``generic_patterns`` (MOVIE, PREVIOUS NIGHT MOVIE 14:00 GMT, FRIDAY MEGA MOVIE, STAR OF
     THE MONTH ...) -> ``generic-slot``: a container whose title rotates; ``etam_title=NONE``
     and ``slot_pool_rule`` tells the forecaster which break pool to use.
   * ``compilation_pattern`` ("TURKISH DRAMA - COMPILATIONAL MOTAWAHESH", "WEEKEND DRAMA
     COMPILATION: SIRR AL HANEEN") -> matched through its *component* title; an eTAM
     "<component> MARATHON" on the same channel is preferred (same block format).
2. **Normalise** (``normalise``): NFKC + upper case; keep only the English part of
   "EN / AR" names; drop ``®``, trailing ``LIVE``, "(2026)"/"2026/27" year tags, "14:00 GMT"
   labels, "DAY n", season tags (S1, S4/S5, SEASON 2), the word REPEAT; ``&`` -> AND;
   apostrophes deleted (MALA'EB -> MALAEB); other punctuation -> space; standalone articles
   AL/EL/AR dropped. A parenthetical that is not a year ("(KHALEEJI 27)") is removed from the
   key and kept as a *qualifier*. eTAM titles additionally lose a trailing channel label
   ("AL AKHBAR - MBC 1" -> "AKHBAR").
3. **Title score** (``title_score``, 0–100): exact key or same words in another order = 100.
   Otherwise every word is aligned to its best counterpart; word similarity is the max of
   rapidfuzz ``ratio``, the ratio after removing a glued article (ELBALAD ~ BALAD), and a
   transliteration skeleton rule (same first letter and identical consonant skeleton after
   KH/SH/TH/DH/GH/PH/Q folding, vowel/W/Y removal, doubled-letter collapse and trailing-H
   removal -> 0.95; this accepts NOWAYLATI~NWAYLATI, ARD~ARDH, DAGHET~DAGHT, HOFRA~HOFFRAH,
   FI~FE, WEST~WAST but rejects AWDA~WADI). Score = mean word similarity over both titles.
   Kinds: ``exact``; ``translit`` (every word aligned, >= token_match); ``variant`` (eTAM has
   only extra ``variant_suffixes`` words, i.e. MARATHON); ``sibling`` (some word unaligned,
   e.g. THE VOICE vs THE VOICE KIDS, EL MADDAH ... AWDA vs ... WADI). rapidfuzz
   token_set_ratio / partial_ratio / WRatio are reported for transparency but are not used for
   acceptance because they score subset titles (THE VOICE ⊂ THE VOICE KIDS) at 100.
4. **Arabic-title transliteration** (``arabic_skeleton_tokens``): for MBC 1 / DRAMA the grid
   carries the Arabic title; its consonant skeleton is compared with the Latin skeleton of
   every eTAM title on the channel (catches THE MORNING SHOW = صباح الخير ياعرب = SABAH AL
   KHAIR YA ARAB, MBC NEWS LIVE = الاخبار = AL AKHBAR, MBC IN A WEEK = في اسبوع MBC = MBC FI
   OSBO', THREE KINGDOMS = الممالك الثلاث = AL MAMALEK AL THALATH). ``cfg.match.aliases`` adds
   hand-listed semantic pairs. Both are accepted only with time-slot evidence.
5. **Time evidence** (``time_evidence``): share of October slots with a September break of the
   candidate starting within [slot start - tol, slot end + tol) (any weekday, and same weekday
   group), share of the candidate's breaks inside the grid windows, weekday sets, and rerun
   share (grid ``is_rerun`` vs eTAM ``first_run``). Also the *slot predecessor* — which eTAM
   programme held the grid windows in September — reported, never used as a match by itself.
6. **Search order**: same channel first (pass 1 exact/translit; pass 2 variants/siblings,
   excluding eTAM titles already claimed by an exact/translit grid match on that channel),
   then Arabic/alias semantic proposals, then other channels (``cross_channel_penalty``);
   the proposal with the highest confidence wins, runners-up are listed as ``alt``.

Confidence calibration (``etam_title`` != NONE: confidence that the history belongs to the
same programme)
------------------------------------------------------------------------------------------
* exact, same channel, time overlap >= ``min_time_overlap``: 0.95 + 0.05 x overlap
  (0.95–1.00); exact with a different time window: 0.88.
* translit >= ``fuzzy_accept`` with time overlap: 0.85 + 0.05 x (score-90)/10 + 0.05 x overlap
  (0.85–0.95); without time evidence 0.82; score in [fuzzy_review, fuzzy_accept): 0.60–0.80.
* compilation -> component (or MARATHON): capped at ``compilation_cap`` (0.80).
* qualifier titles (NADEENA (KHALEEJI 27)): capped at ``qualifier_cap`` (0.80).
* sibling / spin-off proxy (needs time overlap): min(``sibling_cap``, 0.35 + 0.3 x overlap).
* Arabic-transliteration / alias (time-slot inferred, needs ``semantic_min_time_overlap``):
  ``semantic_confidence`` + 0.05 x overlap (<= 0.80).
* cross-channel: same formula on the other channel's breaks, minus ``cross_channel_penalty``,
  capped at ``cross_channel_cap`` (0.80).
* penalties: -``rerun_mismatch_penalty`` when rerun shares differ by > ``rerun_mismatch_gap``;
  -``ambiguity_penalty`` when a runner-up of the same kind is within ``ambiguity_gap`` points;
  history thinner than ``min_etam_breaks`` breaks caps at 0.90.
For ``etam_title`` = NONE the confidence expresses how sure we are that *no* September history
exists: generic-slot 0.95; new program 0.95 (best title score anywhere < 70), 0.85 (70–80),
0.75 (a near-miss >= fuzzy_review) or 0.70 (a sibling/semantic hint was rejected for lack of
time evidence); TBC 0.50. Everything below ``review_threshold`` (0.85) and every
time-slot inferred / generic-slot / new program row goes to the Gate 2 review sheet.

Matching never reads ``is_event`` or overlap columns (being refined concurrently); ``is_event``
is used only to *count* history (``n_etam_breaks`` excludes event breaks).
Deterministic: all iteration is over sorted keys; no randomness.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
CHANNEL_ORDER = ["MBC 1", "MBC DRAMA", "MBC 4", "MBC 2", "MBC BOLLYWOOD", "MBC ACTION", "MBC MAX"]

DEFAULTS: dict[str, Any] = {
    "review_threshold": 0.85, "fuzzy_accept": 90, "fuzzy_review": 80, "token_match": 80,
    "arabic_accept": 90, "arabic_min_skeleton_len": 3, "time_tolerance_min": 30,
    "min_time_overlap": 0.30, "semantic_min_time_overlap": 0.50, "cross_channel_penalty": 0.20,
    "cross_channel_cap": 0.80, "semantic_confidence": 0.75, "sibling_cap": 0.65,
    "compilation_cap": 0.80, "qualifier_cap": 0.80, "ambiguity_penalty": 0.05,
    "ambiguity_gap": 3, "rerun_mismatch_penalty": 0.02, "rerun_mismatch_gap": 0.50,
    "min_etam_breaks": 6, "articles": ["AL", "EL", "AR"], "variant_suffixes": ["MARATHON"],
    "sibling_max_diff_words": 1,
    "generic_words": ["THE", "OF", "AND", "A", "IN", "ON", "WITH", "SHOW", "HIGHLIGHTS", "WORLD",
                      "CHAMPIONSHIP", "CUP", "SERIES", "LEAGUE", "MBC", "MOVIE", "MOVIES", "DRAMA"],
    "etam_channel_suffix": r"\s*-\s*MBC\s*\w*\s*$",
    "compilation_pattern": r"^(?:(?:ARABIC|TURKISH|WEEKEND)\s+DRAMA\s*-?\s*COMPILATION)\s*:?\s*(.*)$",
    "tbc_patterns": [r"^(TBC|TBA|TBD)$"],
    "generic_patterns": [r"^(REPEAT\s+)?MOVIES?(\s+CONTINUES)?$", r"PREVIOUS NIGHT MOVIE",
                         r"MEGA MOVIE"],
    "aliases": [],
}


def match_params(cfg: dict | None) -> dict[str, Any]:
    """cfg['match'] merged over DEFAULTS (plus forecast.weekday_groups)."""
    p = dict(DEFAULTS)
    if cfg:
        p.update(cfg.get("match") or {})
        p["weekday_groups"] = (cfg.get("forecast") or {}).get(
            "weekday_groups", [["Sun", "Mon", "Tue", "Wed"], ["Thu"], ["Fri"], ["Sat"]])
    else:
        p["weekday_groups"] = [["Sun", "Mon", "Tue", "Wed"], ["Thu"], ["Fri"], ["Sat"]]
    return p


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------
_ARABIC_CHARS = re.compile(r"[؀-ۿ]")
_PAREN_YEAR = re.compile(r"\(\s*(?:19|20)\d{2}(?:\s*/\s*(?:19|20)?\d{2})?\s*\)")
_BARE_YEAR = re.compile(r"(?<![0-9])(?:19|20)\d{2}(?:\s*/\s*(?:19|20)?\d{2})?(?![0-9])")
_GMT = re.compile(r"\b\d{1,2}[:.]\d{2}\s*GMT\b")
_SEASON = re.compile(r"\bS\d{1,2}(?:\s*/\s*S\d{1,2})*\b|\bSEASON\s*\d+\b")
_DAY_N = re.compile(r"\bDAY\s*\d+\b")
_LIVE_TAIL = re.compile(r"\s*LIVE\s*$")
_PAREN = re.compile(r"\(([^)]*)\)")
_APOS = re.compile(r"['’‘`´]")


@dataclass(frozen=True)
class CleanTitle:
    raw: str
    key: str                 # normalised matching key
    qualifier: str = ""      # parenthetical content that is not a year tag
    rerun_marker: bool = False
    live_marker: bool = False


def clean_title(raw: str, articles=("AL", "EL", "AR"), channel_suffix: str | None = None) -> CleanTitle:
    """Normalise a programme label (grid or eTAM). See module docstring, step 2."""
    s = unicodedata.normalize("NFKC", str(raw or "")).upper().strip()
    # "EN S1 / AR ®": keep the English part when the right-hand side is Arabic
    if " / " in s:
        left, right = s.split(" / ", 1)
        if _ARABIC_CHARS.search(right):
            s = left + (" ®" if "®" in right else "") + (" LIVE" if right.rstrip("® ").endswith("LIVE") else "")
    rerun = "®" in s or bool(re.search(r"\bREPEAT\b", s))
    s = s.replace("®", " ").strip()
    live = bool(_LIVE_TAIL.search(s))
    while _LIVE_TAIL.search(s):
        s = _LIVE_TAIL.sub("", s)
    if channel_suffix:
        s = re.sub(channel_suffix, "", s)
    s = _GMT.sub(" ", s)
    s = _PAREN_YEAR.sub(" ", s)
    s = _BARE_YEAR.sub(" ", s)
    quals = [q.strip() for q in _PAREN.findall(s) if q.strip()]
    s = _PAREN.sub(" ", s)
    s = _DAY_N.sub(" ", s)
    s = _SEASON.sub(" ", s)
    s = re.sub(r"\bREPEAT\b", " ", s)
    s = s.replace("&", " AND ")
    s = _APOS.sub("", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    toks = s.split()
    arts = set(articles)
    kept = [t for t in toks if t not in arts] or toks
    qual = " ".join(" ".join(re.sub(r"[^A-Z0-9 ]", " ", _APOS.sub("", q)).split()) for q in quals)
    return CleanTitle(raw=str(raw), key=" ".join(kept), qualifier=qual, rerun_marker=rerun,
                      live_marker=live)


def normalise(raw: str, articles=("AL", "EL", "AR"), channel_suffix: str | None = None) -> str:
    """Normalised matching key of a programme label."""
    return clean_title(raw, articles, channel_suffix).key


def classify_label(raw: str, params: dict | None = None) -> tuple[str, str]:
    """Return (label_class, component). label_class in {'tbc','generic','compilation','title'}.
    ``component`` is the component title for compilations, else ''."""
    p = params or DEFAULTS
    s = " ".join(unicodedata.normalize("NFKC", str(raw or "")).upper().replace("®", " ").split())
    s_nolive = _LIVE_TAIL.sub("", s)
    for pat in p["tbc_patterns"]:
        if re.search(pat, s_nolive):
            return "tbc", ""
    for pat in p["generic_patterns"]:
        if re.search(pat, s_nolive):
            return "generic", ""
    m = re.match(p["compilation_pattern"], s_nolive)
    if m and m.group(1).strip():
        return "compilation", m.group(1).strip()
    return "title", ""


# --------------------------------------------------------------------------------------
# Title similarity
# --------------------------------------------------------------------------------------
_DIGRAPHS = (("KH", "K"), ("SH", "S"), ("TH", "T"), ("DH", "D"), ("GH", "G"), ("PH", "F"))


def latin_skeleton(tok: str, drop_trailing_h: bool = True) -> str:
    """Consonant skeleton of a Latin-script token (transliteration-insensitive)."""
    t = tok.upper()
    if drop_trailing_h and len(t) > 2 and t.endswith("H") and t[-2] in "AEIOU":
        t = t[:-1]
    for a, b in _DIGRAPHS:
        t = t.replace(a, b)
    t = t.replace("Q", "K")
    t = re.sub(r"[AEIOUYW']", "", t)
    return re.sub(r"(.)\1+", r"\1", t)


def token_sim(a: str, b: str, articles=("AL", "EL", "AR")) -> float:
    """Similarity (0–1) of two normalised words. See module docstring, step 3."""
    if a == b:
        return 1.0
    best = fuzz.ratio(a, b) / 100.0
    for x, y in ((a, b), (b, a)):
        for art in articles:
            if x.startswith(art) and len(x) - len(art) >= 3:
                best = max(best, fuzz.ratio(x[len(art):], y) / 100.0)
    if a[:1] == b[:1] and a.isalpha() and b.isalpha() and latin_skeleton(a) == latin_skeleton(b):
        best = max(best, 0.95)
    return best


@dataclass(frozen=True)
class TitleScore:
    score: float
    kind: str                     # exact | translit | variant | sibling
    grid_unmatched: tuple = ()
    etam_unmatched: tuple = ()
    note: str = ""


def title_score(g: str, e: str, token_match: float = 80, articles=("AL", "EL", "AR"),
                variant_suffixes=("MARATHON",)) -> TitleScore:
    """Score two *normalised* keys. See module docstring, step 3."""
    if not g or not e:
        return TitleScore(0.0, "sibling")
    if g == e:
        return TitleScore(100.0, "exact")
    gt, et = g.split(), e.split()
    if sorted(gt) == sorted(et):
        return TitleScore(100.0, "exact", note="word order")
    gbest = [max(token_sim(a, b, articles) for b in et) for a in gt]
    ebest = [max(token_sim(b, a, articles) for a in gt) for b in et]
    score = 100.0 * (sum(gbest) + sum(ebest)) / (len(gt) + len(et))
    thr = token_match / 100.0
    g_un = tuple(a for a, s in zip(gt, gbest) if s < thr)
    e_un = tuple(b for b, s in zip(et, ebest) if s < thr)
    if not g_un and not e_un:
        kind = "translit"
    elif not g_un and all(t in set(variant_suffixes) for t in e_un):
        kind = "variant"
    else:
        kind = "sibling"
    return TitleScore(round(score, 1), kind, g_un, e_un)


def rapidfuzz_scores(g: str, e: str) -> str:
    """token_set / partial / WRatio, for the evidence text only."""
    return (f"tset={fuzz.token_set_ratio(g, e):.0f} part={fuzz.partial_ratio(g, e):.0f} "
            f"W={fuzz.WRatio(g, e):.0f}")


# Arabic -> Latin consonant skeleton
_AR_MAP = {"ب": "B", "ت": "T", "ث": "T", "ج": "J", "ح": "H", "خ": "K", "د": "D", "ذ": "D",
           "ر": "R", "ز": "Z", "س": "S", "ش": "S", "ص": "S", "ض": "D", "ط": "T", "ظ": "D",
           "غ": "G", "ف": "F", "ق": "K", "ك": "K", "ل": "L", "م": "M", "ن": "N", "ه": "H",
           "ة": "H", "پ": "B", "چ": "J", "گ": "G", "ڤ": "F"}
_AR_DIACRITICS = re.compile(r"[ً-ْٰـ]")


def arabic_skeleton_tokens(ar: str) -> list[str]:
    """Consonant-skeleton tokens of an Arabic title (Latin runs such as 'MBC' kept)."""
    if not isinstance(ar, str) or not ar.strip():
        return []
    s = _AR_DIACRITICS.sub("", unicodedata.normalize("NFKC", ar))
    s = re.sub(r"LIVE\s*$", "", s.strip(), flags=re.I)
    s = re.sub(r"(?i)live(?=[؀-ۿ\s]|$)", " ", s)
    out = []
    for t in re.findall(r"[؀-ۿ]+|[A-Za-z]+", s):
        if _ARABIC_CHARS.match(t):
            if t.startswith("ال") and len(t) > 3:
                t = t[2:]
            sk = "".join(_AR_MAP.get(c, "") for c in t)
            sk = re.sub(r"(.)\1+", r"\1", sk)
        else:
            sk = latin_skeleton(t, drop_trailing_h=False)
        if sk:
            out.append(sk)
    return out


def latin_skeleton_tokens(key: str) -> list[str]:
    return [sk for sk in (latin_skeleton(t, drop_trailing_h=False) for t in key.split()
                          if not t.isdigit()) if sk]


def arabic_score(ar_tokens: list[str], lat_tokens: list[str], min_len: int = 3) -> float:
    if not ar_tokens or not lat_tokens or len("".join(ar_tokens)) < min_len:
        return 0.0
    return float(fuzz.token_sort_ratio(" ".join(ar_tokens), " ".join(lat_tokens)))


# --------------------------------------------------------------------------------------
# Time evidence / window summaries
# --------------------------------------------------------------------------------------
def _hhmm(m: int) -> str:
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"


def fmt_days(days) -> str:
    ds = [d for d in WEEKDAYS if d in set(days)]
    if len(ds) == 7:
        return "daily"
    idx = [WEEKDAYS.index(d) for d in ds]
    if len(ds) >= 3 and idx == list(range(idx[0], idx[0] + len(idx))):
        return f"{ds[0]}–{ds[-1]}"
    return ",".join(ds)


def summarise_grid_windows(wd, start, end, max_items: int = 6) -> str:
    """'Sun–Wed 21:30–21:59; Fri 22:00–22:44' from grid rows (end exclusive)."""
    df = pd.DataFrame({"wd": list(wd), "s": list(start), "e": list(end)})
    if df.empty:
        return ""
    by_time = df.groupby(["s", "e"])["wd"].apply(lambda x: frozenset(x)).reset_index()
    by_days: dict[frozenset, list] = {}
    for r in by_time.sort_values(["s", "e"]).itertuples():
        by_days.setdefault(r.wd, []).append((r.s, r.e))
    items = []
    for days, rng in sorted(by_days.items(), key=lambda kv: (min(s for s, _ in kv[1]), -len(kv[0]))):
        times = ", ".join(f"{_hhmm(s)}–{_hhmm(max(s, e - 1))}" for s, e in rng)
        items.append((len(rng), f"{fmt_days(days)} {times}"))
    txt = [t for _, t in items]
    if len(txt) > max_items:
        return "; ".join(txt[:max_items]) + f"; +{len(txt) - max_items} more"
    return "; ".join(txt)


def summarise_break_windows(wd, start_min, max_items: int = 6) -> str:
    """Hour-binned summary of break start times: 'daily 20:00–21:59; Fri 22:00–22:59'."""
    df = pd.DataFrame({"wd": list(wd), "h": [int(m) // 60 for m in start_min]})
    if df.empty:
        return ""
    n = len(df)
    cnt = df.groupby("h").size()
    keep = cnt[cnt >= (1 if n < 10 else max(2, int(np.ceil(0.03 * n))))].index
    df = df[df["h"].isin(keep)]
    hours = df.groupby("h")["wd"].apply(lambda x: frozenset(x)).sort_index()
    runs: list[tuple[frozenset, int, int]] = []
    for h, days in hours.items():
        if runs and runs[-1][0] == days and runs[-1][2] == h - 1:
            runs[-1] = (days, runs[-1][1], h)
        else:
            runs.append((days, h, h))
    by_days: dict[frozenset, list] = {}
    for days, a, b in runs:
        by_days.setdefault(days, []).append(f"{a:02d}:00–{b:02d}:59")
    txt = [f"{fmt_days(d)} {', '.join(r)}" for d, r in
           sorted(by_days.items(), key=lambda kv: (kv[1][0], -len(kv[0])))]
    if len(txt) > max_items:
        return "; ".join(txt[:max_items]) + f"; +{len(txt) - max_items} more"
    return "; ".join(txt)


def weekday_group_index(groups) -> dict[str, int]:
    return {d: i for i, grp in enumerate(groups) for d in grp}


@dataclass
class TimeEvidence:
    cov_any: float = 0.0      # share of Oct slots with a Sep airing within tolerance (any weekday)
    cov_wd: float = 0.0       # same, restricted to the same weekday group
    etam_in_grid: float = 0.0 # share of the candidate's Sep breaks inside the grid windows
    wd_jaccard: float = 0.0


def time_evidence(g_wd, g_s, g_e, b_wd, b_s, tol: int, wd_idx: dict) -> TimeEvidence:
    g_s = np.asarray(g_s, dtype=float); g_e = np.asarray(g_e, dtype=float)
    b_s = np.asarray(b_s, dtype=float)
    if len(g_s) == 0 or len(b_s) == 0:
        return TimeEvidence()
    within = (b_s[None, :] >= g_s[:, None] - tol) & (b_s[None, :] < g_e[:, None] + tol)
    gg = np.array([wd_idx.get(d, -1) for d in g_wd]); bg = np.array([wd_idx.get(d, -1) for d in b_wd])
    within_wd = within & (gg[:, None] == bg[None, :])
    inside = (b_s[None, :] >= g_s[:, None]) & (b_s[None, :] < g_e[:, None])
    gset, bset = set(g_wd), set(b_wd)
    return TimeEvidence(cov_any=float(within.any(1).mean()), cov_wd=float(within_wd.any(1).mean()),
                        etam_in_grid=float(inside.any(0).mean()),
                        wd_jaccard=len(gset & bset) / max(1, len(gset | bset)))


# --------------------------------------------------------------------------------------
# Programme map
# --------------------------------------------------------------------------------------
@dataclass
class Proposal:
    etam_title: str
    etam_channel: str
    match_type: str
    confidence: float
    kind: str
    score: float
    te: TimeEvidence
    notes: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    priority: int = 0            # lower wins before confidence (compilation -> MARATHON block)


def _etam_index(breaks: pd.DataFrame, p: dict) -> dict:
    """channel -> {program -> info}. Matching uses ALL breaks; counts exclude events."""
    b = breaks.copy()
    b["smin"] = (b["start_sec"] // 60).astype(int)
    ev = b["is_event"].fillna(False).astype(bool) if "is_event" in b.columns else pd.Series(False, index=b.index)
    b["_ev"] = ev.values
    idx: dict[str, dict] = {}
    for (ch, prog), d in b.groupby(["channel", "program"], sort=True):
        ct = clean_title(prog, p["articles"], p["etam_channel_suffix"])
        ne = d[~d["_ev"]]
        idx.setdefault(ch, {})[prog] = {
            "key": ct.key, "qualifier": ct.qualifier,
            "wd": d["weekday"].tolist(), "smin": d["smin"].tolist(),
            "n_all": int(len(d)), "n": int(len(ne)),
            "rerun_share": float(1 - d["first_run"].astype(bool).mean()),
            "mean_rating_abs": float(ne["rating_abs"].mean()) if len(ne) else float("nan"),
            "windows": summarise_break_windows(ne["weekday"], ne["smin"]) if len(ne) else
            summarise_break_windows(d["weekday"], d["smin"]),
            "lat_skel": latin_skeleton_tokens(ct.key),
            "episodes": " ".join(sorted(set(map(str, d["episode"].dropna())))).upper(),
        }
    return idx


def _grid_titles(grid: pd.DataFrame) -> pd.DataFrame:
    g = grid[grid["in_flight"].astype(bool)].copy()
    rows = []
    for (ch, t), d in g.groupby(["channel", "title_en"], sort=True):
        ar = d["title_ar"].dropna()
        seas = sorted(set(d["season"].dropna().astype(str)))
        rr = float(d["is_rerun"].astype(bool).mean())
        rows.append({
            "channel": ch, "grid_title": t, "grid_title_ar": ar.iloc[0] if len(ar) else "",
            "season": "|".join(seas), "rerun_share": rr,
            "is_rerun": "yes" if rr == 1 else ("no" if rr == 0 else "mixed"),
            "is_live": bool(d["is_live"].astype(bool).any()),
            "tiers": ",".join(sorted(set(d["tier"].dropna().astype(str)))),
            "n_grid_slots": int(len(d)), "usd_at_stake": float(d["rate_usd"].sum()),
            "n_synthetic_slots": int(d["is_synthetic"].astype(bool).sum()),
            "_wd": d["weekday"].tolist(), "_s": d["start_min"].astype(int).tolist(),
            "_e": d["end_min"].astype(int).tolist(),
            "grid_time_windows": summarise_grid_windows(d["weekday"], d["start_min"], d["end_min"]),
        })
    return pd.DataFrame(rows)


def _slot_predecessor(gr: dict, eidx_ch: dict, wd_idx: dict) -> str:
    counts: dict[str, int] = {}
    total = 0
    gs, ge = np.array(gr["_s"]), np.array(gr["_e"])
    gg = np.array([wd_idx.get(d, -1) for d in gr["_wd"]])
    for prog, info in eidx_ch.items():
        bs = np.array(info["smin"]); bg = np.array([wd_idx.get(d, -1) for d in info["wd"]])
        if len(bs) == 0:
            continue
        hit = ((bs[None, :] >= gs[:, None]) & (bs[None, :] < ge[:, None]) & (gg[:, None] == bg[None, :])).any(0)
        c = int(hit.sum())
        if c:
            counts[prog] = c
            total += c
    if not total:
        return ""
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
    return ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in top)


def _conf_title(kind: str, score: float, te: TimeEvidence, p: dict) -> float:
    fa, fr, mt = p["fuzzy_accept"], p["fuzzy_review"], p["min_time_overlap"]
    if kind == "exact":
        return 0.95 + 0.05 * te.cov_any if te.cov_any >= mt else 0.88
    if kind == "translit":
        if score >= fa:
            if te.cov_any >= mt:
                return 0.85 + 0.05 * (score - fa) / max(1e-9, 100 - fa) + 0.05 * te.cov_any
            return 0.82
        return 0.60 + 0.20 * (score - fr) / max(1e-9, fa - fr)
    return 0.0


def _time_text(te: TimeEvidence, tol: int) -> str:
    return (f"time {100 * te.cov_any:.0f}% of Oct slots within ±{tol}m of Sep airings "
            f"(same wd-group {100 * te.cov_wd:.0f}%), {100 * te.etam_in_grid:.0f}% of Sep breaks inside grid windows")


def build_program_map(grid: pd.DataFrame, breaks: pd.DataFrame, cfg: dict | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (program_map, unmatched_etam). Pure function of its inputs + cfg."""
    p = match_params(cfg)
    arts = tuple(p["articles"]); tol = int(p["time_tolerance_min"])
    wd_idx = weekday_group_index(p["weekday_groups"])
    eidx = _etam_index(breaks, p)
    gt = _grid_titles(grid)
    all_channels = sorted(eidx)

    def ts(a, b):
        return title_score(a, b, p["token_match"], arts, tuple(p["variant_suffixes"]))

    def tev(gr, info):
        return time_evidence(gr["_wd"], gr["_s"], gr["_e"], info["wd"], info["smin"], tol, wd_idx)

    def penalties(prop: Proposal, gr: dict, info: dict):
        # eTAM flags MARATHON blocks as first run, so rerun status is not comparable for them
        if prop.kind != "variant" and abs(gr["rerun_share"] - info["rerun_share"]) > p["rerun_mismatch_gap"]:
            prop.confidence -= p["rerun_mismatch_penalty"]; prop.flags.append("rerun_mismatch")
        if info["n_all"] < p["min_etam_breaks"]:   # n_all: matching must not depend on is_event
            prop.confidence = min(prop.confidence, 0.90); prop.flags.append("thin_history")

    # ---- pre-compute label classes and keys
    recs = []
    for gr in gt.to_dict("records"):
        cls, comp = classify_label(gr["grid_title"], p)
        ct = clean_title(comp if cls == "compilation" else gr["grid_title"], arts)
        gr.update(label_class=cls, component=comp, key=ct.key, qualifier=ct.qualifier)
        recs.append(gr)

    # ---- pass 1: same-channel exact / translit (claims eTAM titles)
    claimed: dict[tuple[str, str], list[str]] = {}
    for gr in recs:
        gr["_cands"] = []
        if gr["label_class"] in ("tbc", "generic"):
            continue
        for prog, info in sorted(eidx.get(gr["channel"], {}).items()):
            s = ts(gr["key"], info["key"])
            if s.kind in ("exact", "translit") and s.score >= p["fuzzy_review"] or s.kind == "exact":
                gr["_cands"].append((prog, s))
        if gr["label_class"] == "title":
            for prog, s in gr["_cands"]:
                claimed.setdefault((gr["channel"], prog), []).append(gr["grid_title"])

    rows = []
    for gr in recs:
        ch = gr["channel"]
        out = {k: v for k, v in gr.items() if not k.startswith("_")}
        props: list[Proposal] = []
        hints: list[str] = []
        best_any = (0.0, "", "")  # (score, title, channel)

        if gr["label_class"] == "tbc":
            out.update(etam_title="NONE", etam_channel="", match_type="new program", confidence=0.50,
                       evidence="TBC: content unknown — no title to match; forecast from slot pool",
                       flags="tbc")
            rows.append((out, gr)); continue
        if gr["label_class"] == "generic":
            out.update(etam_title="NONE", etam_channel="", match_type="generic-slot", confidence=0.95,
                       evidence="generic movie-strand label (title rotates) — no single eTAM title",
                       flags="generic")
            rows.append((out, gr)); continue

        chan = eidx.get(ch, {})
        # same channel: score every title once
        same_scores = {prog: ts(gr["key"], info["key"]) for prog, info in sorted(chan.items())}
        for prog, s in same_scores.items():
            if s.score > best_any[0]:
                best_any = (s.score, prog, ch)

        # (a) same-channel exact / translit
        for prog, s in same_scores.items():
            info = chan[prog]
            if not (s.kind == "exact" or (s.kind == "translit" and s.score >= p["fuzzy_review"])):
                continue
            te = tev(gr, info)
            c = _conf_title(s.kind, s.score, te, p)
            mt = "exact" if s.kind == "exact" else "fuzzy"
            notes = [f"title={s.score:.0f} {s.kind}" + (f" ({s.note})" if s.note else "")]
            if s.kind != "exact":
                notes.append(f"translit {gr['key']}~{info['key']}; {rapidfuzz_scores(gr['key'], info['key'])}")
            pr = Proposal(prog, ch, mt, c, s.kind, s.score, te, notes, [])
            if gr["label_class"] == "compilation":
                pr.confidence = min(pr.confidence, p["compilation_cap"]); pr.match_type = "fuzzy"
                pr.notes.insert(0, f"compilation label -> component '{gr['component']}'")
                pr.flags.append("compilation")
            if gr["qualifier"]:
                pr.confidence = min(pr.confidence, p["qualifier_cap"]); pr.flags.append("qualifier")
            penalties(pr, gr, info)
            props.append(pr)

        # (b) same-channel variant (MARATHON) / sibling
        for prog, s in same_scores.items():
            info = chan[prog]
            if s.kind not in ("variant", "sibling") or s.score < p["fuzzy_review"]:
                continue
            te = tev(gr, info)
            if gr["label_class"] == "compilation" and s.kind == "variant":
                c = min(p["compilation_cap"], _conf_title("exact", 100, te, p))
                pr = Proposal(prog, ch, "fuzzy", c, s.kind, s.score, te,
                              [f"compilation label -> component '{gr['component']}'; eTAM block format "
                               f"'{prog}' preferred over the regular episode title"], ["compilation"],
                              priority=-1 if info["n_all"] >= p["min_etam_breaks"] else 0)
                penalties(pr, gr, info); props.append(pr); continue
            if gr["label_class"] == "compilation":
                continue
            gw = set(p["generic_words"])
            shared = [t for t in gr["key"].split() if t not in s.grid_unmatched and t not in gw]
            if (len(s.grid_unmatched) > p["sibling_max_diff_words"] or len(s.etam_unmatched) >
                    p["sibling_max_diff_words"] or not shared):
                continue
            others = [t for t in claimed.get((ch, prog), []) if t != gr["grid_title"]]
            diff = f"{'/'.join(s.grid_unmatched) or '-'} vs {'/'.join(s.etam_unmatched) or '-'}"
            if others:
                hints.append(f"sibling '{prog}' already matched to grid '{others[0]}'")
                continue
            if te.cov_any < p["min_time_overlap"]:
                hints.append(f"sibling '{prog}' (differs {diff}) rejected: time overlap {100 * te.cov_any:.0f}%")
                continue
            c = min(p["sibling_cap"], 0.35 + 0.3 * te.cov_any)
            pr = Proposal(prog, ch, "fuzzy", c, s.kind, s.score, te,
                          [f"title={s.score:.0f} SIBLING/spin-off (differs {diff}) — proxy only"],
                          ["sibling"])
            penalties(pr, gr, info); props.append(pr)

        # (c) Arabic-title transliteration + config aliases (time-slot inferred)
        ar_tok = arabic_skeleton_tokens(gr["grid_title_ar"])
        if ar_tok:
            for prog, info in sorted(chan.items()):
                a = arabic_score(ar_tok, info["lat_skel"], p["arabic_min_skeleton_len"])
                if a < p["arabic_accept"] or same_scores[prog].kind in ("exact", "translit"):
                    continue
                te = tev(gr, info)
                if te.cov_any < p["semantic_min_time_overlap"]:
                    hints.append(f"AR-title ~ '{prog}' (skel {a:.0f}) rejected: time overlap {100 * te.cov_any:.0f}%")
                    continue
                c = min(0.80, p["semantic_confidence"] + 0.05 * te.cov_any)
                pr = Proposal(prog, ch, "time-slot inferred", c, "semantic", a, te,
                              [f"semantic: Arabic title '{gr['grid_title_ar']}' transliterates to eTAM "
                               f"'{prog}' (skeleton {' '.join(ar_tok)} ~ {' '.join(info['lat_skel'])}, {a:.0f}); "
                               f"EN title differs"], ["semantic"])
                penalties(pr, gr, info); props.append(pr)
        for al in p.get("aliases") or []:
            if al.get("channel") != ch or normalise(al.get("grid", ""), arts) != normalise(gr["grid_title"], arts):
                continue
            prog = al.get("etam")
            if prog not in chan:
                hints.append(f"alias '{prog}' not in Sep {ch} history"); continue
            te = tev(gr, chan[prog])
            if te.cov_any < p["semantic_min_time_overlap"]:
                hints.append(f"possible alias '{prog}' ({al.get('basis', '')}) rejected: time overlap "
                             f"{100 * te.cov_any:.0f}%")
                continue
            c = min(0.80, p["semantic_confidence"] + 0.05 * te.cov_any)
            pr = Proposal(prog, ch, "time-slot inferred", c, "alias", 0.0, te,
                          [f"alias ({al.get('basis', '')})"], ["alias"])
            penalties(pr, gr, chan[prog]); props.append(pr)

        # (d) other channels (only when nothing strong on the same channel)
        if not any(pr.kind in ("exact", "translit") for pr in props):
            for och in all_channels:
                if och == ch:
                    continue
                for prog, info in sorted(eidx[och].items()):
                    s = ts(gr["key"], info["key"])
                    if s.score > best_any[0]:
                        best_any = (s.score, prog, och)
                    if not (s.kind == "exact" or (s.kind == "translit" and s.score >= p["fuzzy_review"])):
                        continue
                    te = tev(gr, info)
                    base = _conf_title(s.kind, s.score, te, p)
                    c = min(p["cross_channel_cap"], base - p["cross_channel_penalty"])
                    pr = Proposal(prog, och, s.kind if s.kind == "exact" else "fuzzy", c, s.kind, s.score, te,
                                  [f"CROSS-CHANNEL: history on {och} (title={s.score:.0f} {s.kind}); "
                                   f"audience level not transferable"], ["cross_channel"])
                    if gr["label_class"] == "compilation":
                        pr.confidence = min(pr.confidence, p["compilation_cap"]); pr.flags.append("compilation")
                    same_title_other = [r for r in recs if r["channel"] == och and r["key"] == gr["key"]]
                    if same_title_other:
                        pr.notes.append(f"{och} grid carries the same title as season "
                                        f"'{same_title_other[0]['season'] or '-'}' vs '{gr['season'] or '-'}' here")
                    penalties(pr, gr, info); props.append(pr)

        if props:
            props.sort(key=lambda x: (x.priority, -round(x.confidence, 6), -x.score, -x.te.cov_any, x.etam_title))
            best = props[0]
            alts = [q for q in props[1:] if q.etam_title != best.etam_title or q.etam_channel != best.etam_channel]
            same_kind = [q for q in alts if q.kind == best.kind and q.etam_channel == best.etam_channel
                         and abs(q.score - best.score) <= p["ambiguity_gap"] and best.kind != "exact"]
            if same_kind:
                best.confidence -= p["ambiguity_penalty"]; best.flags.append("ambiguous")
            elif (best.kind in ("exact", "translit") and best.te.cov_any < p["min_time_overlap"]
                  and any(q.te.cov_any >= p["min_time_overlap"] and q.etam_channel == best.etam_channel
                          for q in alts)):
                best.confidence -= p["ambiguity_penalty"]; best.flags.append("ambiguous_time")
            info = eidx[best.etam_channel][best.etam_title]
            ev = list(best.notes)
            ev.append(_time_text(best.te, tol))
            ev.append(f"wd Oct {fmt_days(set(gr['_wd']))} / Sep {fmt_days(set(info['wd']))}")
            ev.append(f"rerun Oct {100 * gr['rerun_share']:.0f}% / Sep {100 * info['rerun_share']:.0f}%")
            ev.append(f"{info['n']} Sep breaks")
            if ar_tok and best.kind in ("exact", "translit", "variant"):
                a = arabic_score(ar_tok, info["lat_skel"], p["arabic_min_skeleton_len"])
                if a >= p["arabic_accept"]:
                    ev.append("AR title agrees")
            if gr["qualifier"]:
                q = gr["qualifier"]
                seen = any(q.split()[0] in (inf["key"] + " " + inf["episodes"])
                           for chx in eidx.values() for inf in chx.values())
                ev.append(f"qualifier '{q}'" + ("" if seen else " NOT in Sep history (special edition; use "
                                                               "base-title history only as a proxy)"))
            if gr["is_live"]:
                ev.append("LIVE")
                best.flags.append("live")
            if alts:
                ev.append("alt: " + ", ".join(f"{q.etam_title}@{q.etam_channel} {q.confidence:.2f}" for q in alts[:2]))
            if hints:
                ev.append("; ".join(hints[:2]))
            if best.te.cov_any < p["min_time_overlap"] and best.kind in ("exact", "translit"):
                best.flags.append("time_window_differs")
            out.update(etam_title=best.etam_title, etam_channel=best.etam_channel, match_type=best.match_type,
                       confidence=round(float(np.clip(best.confidence, 0, 1)), 3), evidence="; ".join(ev),
                       flags=",".join(dict.fromkeys(best.flags)), title_score=best.score,
                       time_overlap=round(best.te.cov_any, 3), time_overlap_wdgroup=round(best.te.cov_wd, 3))
        else:
            # never force: new program
            sc, bt, bch = best_any
            if hints:
                conf = 0.70
            elif sc >= p["fuzzy_review"]:
                conf = 0.75
            elif sc >= 70:
                conf = 0.85
            else:
                conf = 0.95
            ev = [f"no eTAM title >= {p['fuzzy_review']} on any channel" if sc < p["fuzzy_review"]
                  else f"near-miss only (not accepted)",
                  f"best: '{bt}'@{bch} {sc:.0f}" if bt else "no candidate"]
            if gr["label_class"] == "compilation":
                ev.insert(0, f"compilation of '{gr['component']}' — component not in Sep history")
            ev += hints[:2]
            if gr["is_live"]:
                ev.append("LIVE")
            out.update(etam_title="NONE", etam_channel="", match_type="new program", confidence=conf,
                       evidence="; ".join(ev), flags="live" if gr["is_live"] else "")
        rows.append((out, gr))

    # ---- assemble
    final = []
    for out, gr in rows:
        ech, et = out.get("etam_channel") or "", out["etam_title"]
        info = eidx.get(ech, {}).get(et) if et != "NONE" else None
        out["n_etam_breaks"] = int(info["n"]) if info else 0
        out["etam_time_windows"] = info["windows"] if info else ""
        out["etam_mean_rating_abs"] = round(info["mean_rating_abs"], 1) if info else np.nan
        out["slot_predecessor"] = _slot_predecessor(gr, eidx.get(gr["channel"], {}), wd_idx)
        if out["match_type"] in ("generic-slot", "new program"):
            rer = " Prefer rerun breaks (first_run=False) of the pool if >= min_breaks." if gr["rerun_share"] == 1 else ""
            what = "any programme (movie title rotates)" if out["match_type"] == "generic-slot" else \
                "any programme (no own history)"
            out["slot_pool_rule"] = (f"per October slot: {gr['channel']} breaks on the same weekday group whose "
                                     f"start falls within that slot's own [start, end) window, {what}; "
                                     f"events excluded.{rer} Windows: {gr['grid_time_windows']}")
        else:
            out["slot_pool_rule"] = ""
        if out["evidence"] and out["slot_predecessor"] and out["match_type"] in ("new program", "generic-slot"):
            out["evidence"] += f"; slot predecessor: {out['slot_predecessor']}"
        final.append(out)

    pm = pd.DataFrame(final)
    for c in ("title_score", "time_overlap", "time_overlap_wdgroup"):
        if c not in pm:
            pm[c] = np.nan
    cols = ["channel", "grid_title", "grid_title_ar", "season", "is_rerun", "rerun_share", "is_live",
            "etam_title", "etam_channel", "match_type", "confidence", "evidence", "n_grid_slots",
            "n_synthetic_slots", "usd_at_stake", "n_etam_breaks", "etam_mean_rating_abs",
            "etam_time_windows", "grid_time_windows", "slot_pool_rule", "slot_predecessor", "tiers",
            "label_class", "component", "qualifier", "title_score", "time_overlap",
            "time_overlap_wdgroup", "flags"]
    pm = pm[cols]
    pm["_co"] = pm["channel"].map({c: i for i, c in enumerate(CHANNEL_ORDER)}).fillna(99)
    pm = pm.sort_values(["_co", "usd_at_stake", "grid_title"], ascending=[True, False, True]).drop(columns="_co")
    pm = pm.reset_index(drop=True)

    # ---- unmatched eTAM titles
    used = {(r.etam_channel, r.etam_title) for r in pm.itertuples() if r.etam_title != "NONE"}
    generic_rows = [gr for _, gr in rows if gr["label_class"] == "generic"]
    un = []
    for ch in all_channels:
        grid_keys = [(r["grid_title"], r["key"]) for r in recs if r["channel"] == ch and r["label_class"] != "generic"]
        gen = [g for g in generic_rows if g["channel"] == ch]
        for prog, info in sorted(eidx[ch].items()):
            if info["n"] < p["min_etam_breaks"] or (ch, prog) in used:
                continue
            best = max(((ts(k, info["key"]).score, t) for t, k in grid_keys), default=(0.0, ""))
            share = 0.0
            if gen:
                bs = np.array(info["smin"]); bg = np.array([wd_idx.get(d, -1) for d in info["wd"]])
                hit = np.zeros(len(bs), dtype=bool)
                for g in gen:
                    gs, ge = np.array(g["_s"]), np.array(g["_e"])
                    gg = np.array([wd_idx.get(d, -1) for d in g["_wd"]])
                    hit |= ((bs[None, :] >= gs[:, None]) & (bs[None, :] < ge[:, None]) & (gg[:, None] == bg[None, :])).any(0)
                share = float(hit.mean())
            un.append({"channel": ch, "etam_title": prog, "n_sep_breaks": info["n"],
                       "mean_rating_abs": round(info["mean_rating_abs"], 1),
                       "rerun_share": round(info["rerun_share"], 2), "etam_time_windows": info["windows"],
                       "in_generic_slot_share": round(share, 2),
                       "likely_role": _likely_role(ch, prog, info, share, used, recs),
                       "closest_grid_title": best[1], "closest_grid_score": best[0]})
    ue = pd.DataFrame(un)
    if len(ue):
        ue["_co"] = ue["channel"].map({c: i for i, c in enumerate(CHANNEL_ORDER)}).fillna(99)
        ue["_gen"] = ue["likely_role"].eq("movie in a generic-slot pool")
        ue = ue.sort_values(["_gen", "_co", "n_sep_breaks", "etam_title"], ascending=[True, True, False, True])
        ue = ue.drop(columns=["_co", "_gen"]).reset_index(drop=True)
    return pm, ue


def _likely_role(ch: str, prog: str, info: dict, share: float, used: set, recs: list) -> str:
    if share >= 0.5:
        return "movie in a generic-slot pool"
    if any(prog.startswith(u[1] + " ") for u in used if u[0] == ch):
        return "variant/marathon of a matched title"
    other = sorted({r["channel"] for r in recs if r["channel"] != ch and r["key"] == info["key"]})
    if other:
        return "same title is in the October grid on " + ", ".join(other)
    return "ended / not in October grid?"


REVIEW_COLUMNS = ["channel", "grid_title", "proposed_etam_title", "tier(s)", "n_october_slots",
                  "usd_at_stake", "n_sep_breaks", "confidence", "match_type", "evidence",
                  "verifier_verdict", "human_decision", "human_etam_title"]


def review_table(pm: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    p = match_params(cfg)
    m = (pm["confidence"] < p["review_threshold"]) | pm["match_type"].isin(
        ["time-slot inferred", "generic-slot", "new program"])
    r = pm[m].copy()
    r = r.sort_values(["usd_at_stake", "channel", "grid_title"], ascending=[False, True, True])
    out = pd.DataFrame({
        "channel": r["channel"], "grid_title": r["grid_title"], "proposed_etam_title": r["etam_title"],
        "tier(s)": r["tiers"], "n_october_slots": r["n_grid_slots"],
        "usd_at_stake": r["usd_at_stake"].round(2), "n_sep_breaks": r["n_etam_breaks"],
        "confidence": r["confidence"], "match_type": r["match_type"],
        "evidence": [e + (f" [history on {c}]" if c and c != ch else "")
                     for e, c, ch in zip(r["evidence"], r["etam_channel"], r["channel"])],
        "verifier_verdict": "", "human_decision": "", "human_etam_title": ""})
    return out[REVIEW_COLUMNS].reset_index(drop=True)
