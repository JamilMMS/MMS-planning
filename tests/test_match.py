"""Programme matching (Phase 3). Pure-function tests always run; the coverage tests run only
when data/processed/{grid,breaks}.parquet and program_map.csv exist."""
from __future__ import annotations

import pandas as pd
import pytest

from optimizer.config import load_config, project_root
from optimizer.match.matching import (arabic_score, arabic_skeleton_tokens, build_program_map,
                                      classify_label, clean_title, latin_skeleton_tokens,
                                      match_params, normalise, review_table, title_score)

ROOT = project_root()
GRID = ROOT / "data" / "processed" / "grid.parquet"
BREAKS = ROOT / "data" / "processed" / "breaks.parquet"
PMAP = ROOT / "data" / "processed" / "program_map.csv"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def p(cfg):
    return match_params(cfg)


def _ts(a, b, p):
    return title_score(normalise(a), normalise(b, channel_suffix=p["etam_channel_suffix"]),
                       p["token_match"], tuple(p["articles"]), tuple(p["variant_suffixes"]))


# ---------------- normaliser
@pytest.mark.parametrize("raw,key", [
    ("AHLA NASEEB S1 / أحلى نصيب ®", "AHLA NASEEB"),
    ("AHLA NASEEB", "AHLA NASEEB"),
    ("MALA'EB MA' FAISAL ALJAFAN (2026)", "MALAEB MA FAISAL ALJAFAN"),
    ("BUNDESLIGA HIGHLIGHTS   (2026/27)", "BUNDESLIGA HIGHLIGHTS"),
    ("FIA KARTING CHAMPIONSHIP 2026", "FIA KARTING CHAMPIONSHIP"),
    ("EXTREME H WORLD CUP (2026)DAY 1LIVE", "EXTREME H WORLD CUP"),
    ("FILMS & STARS", "FILMS AND STARS"),
    ("POLICE 24/7", "POLICE 24 7"),
    ("KINGDOM S4/S5", "KINGDOM"),
    ("MBC NEWS LIVE", "MBC NEWS"),
    ("HABIBATI… MAN TAKOUN", "HABIBATI MAN TAKOUN"),
    ("EL MADDAH: OSTOURET EL ISHQ", "MADDAH OSTOURET ISHQ"),
])
def test_normalise(raw, key):
    assert normalise(raw) == key


def test_qualifier_and_markers():
    ct = clean_title("NADEENA (KHALEEJI 27)")
    assert ct.key == "NADEENA" and ct.qualifier == "KHALEEJI 27"
    assert clean_title("SET SHABAB S2 / ست شباب ®").rerun_marker
    assert clean_title("FRIDAY MEGA MOVIE REPEAT").rerun_marker
    assert normalise("AL AKHBAR - MBC 1", channel_suffix=r"\s*-\s*MBC\s*\w*\s*$") == "AKHBAR"


@pytest.mark.parametrize("a,b", [
    ("AL NOWAYLATI", "AL NWAYLATI"), ("ARD AL MILLION", "ARDH AL MILLION"),
    ("TAHT AL DAGHET", "TAHT AL DAGHT"), ("AL HOFRA", "AL HOFFRAH"), ("MA ZELT FI 17", "MA ZELT FE 17"),
    ("WEST ELBALAD", "WAST AL BALAD"), ("ALA SADA AL KHALKHAL", "ALA SADA AL KHELKHAL"),
    ("PARINEETI", "PARINEETII"),
])
def test_transliteration_pairs_score_high(a, b, p):
    s = _ts(a, b, p)
    assert s.score >= 85 and s.kind in ("exact", "translit")


def test_nowaylati_threshold(p):
    assert _ts("AL NOWAYLATI", "AL NWAYLATI", p).score >= 85


@pytest.mark.parametrize("a,b", [
    ("THE VOICE", "THE VOICE KIDS"), ("EL MADDAH: OSTOURET EL AWDA", "EL MADDAH OSTOURET AL WADI"),
])
def test_siblings_are_not_translit(a, b, p):
    assert _ts(a, b, p).kind == "sibling"


def test_marathon_is_variant(p):
    assert _ts("AL MOTAWAHESH", "AL MOTAWAHESH MARATHON", p).kind == "variant"


@pytest.mark.parametrize("label,cls", [
    ("PREVIOUS NIGHT MOVIE 14:00 GMT", "generic"), ("MOVIE", "generic"), ("MOVIES", "generic"),
    ("REPEAT MOVIE CONTINUES", "generic"), ("FRIDAY MEGA MOVIE", "generic"),
    ("STAR OF THE MONTH:BRAD PITT", "generic"), ("TBC", "tbc"), ("AHLA NASEEB", "title"),
    ("SCOOP WITH RAYA", "title"),
])
def test_classify_label(label, cls, p):
    assert classify_label(label, p)[0] == cls


@pytest.mark.parametrize("label,comp", [
    ("WEEKEND DRAMA COMPILATION: SIRR AL HANEEN", "SIRR AL HANEEN"),
    ("TURKISH DRAMA - COMPILATIONAL MOTAWAHESH", "AL MOTAWAHESH"),
    ("TURKISH DRAMA COMPILATIONWOROUD WA THONOUB", "WOROUD WA THONOUB"),
    ("ARABIC DRAMA - COMPILATIONESH ESH", "ESH ESH"),
])
def test_compilation_component(label, comp, p):
    assert classify_label(label, p) == ("compilation", comp)


def test_arabic_skeleton_semantic_pairs(p):
    pairs = [("صباح الخير ياعربLIVE", "SABAH AL KHAIR YA ARAB"), ("الاخبار", "AL AKHBAR - MBC 1"),
             ("في اسبوع MBCLIVE", "MBC FI OSBO'"), ("الممالك الثلاث", "AL MAMALEK AL THALATH")]
    for ar, en in pairs:
        lat = latin_skeleton_tokens(normalise(en, channel_suffix=p["etam_channel_suffix"]))
        assert arabic_score(arabic_skeleton_tokens(ar), lat) >= p["arabic_accept"], (ar, en)
    assert arabic_score(arabic_skeleton_tokens("أحلى نصيب"), latin_skeleton_tokens("AHLA MA TASH")) < 90


# ---------------- build_program_map on tiny synthetic frames
def _grid(rows):
    base = dict(title_ar=None, season=None, is_rerun=False, is_live=False, tier="Regular", rate_usd=100.0,
                in_flight=True, is_synthetic=False, weekday="Sun", start_min=1200, end_min=1230)
    return pd.DataFrame([{**base, **r} for r in rows])


def _breaks(rows):
    base = dict(episode="EP1", broadcast_date=pd.Timestamp("2026-09-06"), weekday="Sun", start_sec=1200 * 60,
                end_sec=1200 * 60 + 60, first_run=True, rerun="FIRST RUN", rating_abs=1000.0, is_event=False)
    return pd.DataFrame([{**base, **r} for r in rows])


def test_no_candidate_returns_new_program(cfg):
    g = _grid([{"channel": "MBC 1", "title_en": "COMPLETELY NEW SHOW"}])
    b = _breaks([{"channel": "MBC 1", "program": "ALI KLAY"}] * 8)
    pm, _ = build_program_map(g, b, cfg)
    r = pm.iloc[0]
    assert r["match_type"] == "new program" and r["etam_title"] == "NONE" and r["n_etam_breaks"] == 0


def test_exact_and_generic_and_fuzzy(cfg):
    g = _grid([{"channel": "MBC 1", "title_en": "ALI KLAY"},
               {"channel": "MBC 1", "title_en": "AL NOWAYLATI", "start_min": 600, "end_min": 645},
               {"channel": "MBC 2", "title_en": "PREVIOUS NIGHT MOVIE 14:00 GMT", "rate_usd": 50.0}])
    b = _breaks([{"channel": "MBC 1", "program": "ALI KLAY"}] * 8
                + [{"channel": "MBC 1", "program": "AL NWAYLATI", "start_sec": 610 * 60}] * 8
                + [{"channel": "MBC 2", "program": "SOME FILM"}] * 8)
    pm, _ = build_program_map(g, b, cfg)
    pm = pm.set_index("grid_title")
    assert pm.loc["ALI KLAY", "match_type"] == "exact" and pm.loc["ALI KLAY", "confidence"] >= 0.95
    assert pm.loc["AL NOWAYLATI", "etam_title"] == "AL NWAYLATI"
    assert 0.85 <= pm.loc["AL NOWAYLATI", "confidence"] <= 0.95
    gen = pm.loc["PREVIOUS NIGHT MOVIE 14:00 GMT"]
    assert gen["match_type"] == "generic-slot" and gen["etam_title"] == "NONE" and gen["slot_pool_rule"]


def test_cross_channel_is_below_review(cfg):
    g = _grid([{"channel": "MBC 1", "title_en": "AL MADEENA AL BA'EEDA"}])
    b = _breaks([{"channel": "MBC 4", "program": "AL MADEENA AL BA'EEDA"}] * 8
                + [{"channel": "MBC 1", "program": "ALI KLAY"}] * 8)
    pm, _ = build_program_map(g, b, cfg)
    r = pm.iloc[0]
    assert r["etam_channel"] == "MBC 4" and r["confidence"] < match_params(cfg)["review_threshold"]
    assert len(review_table(pm, cfg)) == 1


def test_matching_ignores_is_event(cfg):
    g = _grid([{"channel": "MBC 1", "title_en": "ALI KLAY"}])
    b1 = _breaks([{"channel": "MBC 1", "program": "ALI KLAY"}] * 8)
    b2 = b1.assign(is_event=True)
    a, _ = build_program_map(g, b1, cfg)
    c, _ = build_program_map(g, b2, cfg)
    assert a.loc[0, "etam_title"] == c.loc[0, "etam_title"] and a.loc[0, "confidence"] == c.loc[0, "confidence"]
    assert c.loc[0, "n_etam_breaks"] == 0 and a.loc[0, "n_etam_breaks"] == 8


# ---------------- real data coverage
@pytest.mark.skipif(not (GRID.exists() and BREAKS.exists() and PMAP.exists()),
                    reason="run python -m optimizer.match.run first")
def test_program_map_covers_grid_exactly_once():
    g = pd.read_parquet(GRID)
    g = g[g["in_flight"].astype(bool)]
    pm = pd.read_csv(PMAP, keep_default_na=False)
    keys = set(map(tuple, g[["channel", "title_en"]].drop_duplicates().values))
    pk = list(map(tuple, pm[["channel", "grid_title"]].values))
    assert len(pk) == len(set(pk)), "duplicate channel+title rows"
    assert set(pk) == keys
    assert pm["usd_at_stake"].astype(float).sum() == pytest.approx(g["rate_usd"].sum(), rel=1e-9)
    assert pm["n_grid_slots"].astype(int).sum() == len(g)
    assert set(pm["match_type"]) <= {"exact", "fuzzy", "time-slot inferred", "generic-slot", "new program"}
    none = pm["etam_title"] == "NONE"
    assert (pm.loc[none, "match_type"].isin(["generic-slot", "new program"])).all()
    assert pm["confidence"].astype(float).between(0, 1).all()


@pytest.mark.skipif(not (GRID.exists() and BREAKS.exists()), reason="processed parquet missing")
def test_real_data_known_pairs(cfg):
    pm, _ = build_program_map(pd.read_parquet(GRID), pd.read_parquet(BREAKS), cfg)
    m = pm.set_index(["channel", "grid_title"])
    assert m.loc[("MBC 1", "AL NOWAYLATI"), "etam_title"] == "AL NWAYLATI"
    assert m.loc[("MBC 1", "AHLA NASEEB"), "match_type"] == "new program"
    assert m.loc[("MBC 1", "MBC NEWS LIVE"), "match_type"] == "time-slot inferred"
    assert m.loc[("MBC 1", "MBC NEWS LIVE"), "confidence"] < 0.85
    assert m.loc[("MBC 2", "PREVIOUS NIGHT MOVIE 14:00 GMT"), "match_type"] == "generic-slot"
    assert m.loc[("MBC 1", "TBC"), "match_type"] == "new program"
