"""Classify and file new exports dropped into ``data/incoming/``.

Per DATA_SPEC.md: "When a new file appears in data/incoming/, the data-validator identifies
its type from its header row and layout, validates it, and moves a copy to the right
data/raw/ subfolder."

We never guess a layout we have not seen. Classification is done purely by keyword presence
in the first 3 rows of the file (case-insensitive), documented per type below:

  - grid          : header row contains both "station_code" and "day_mask" (the October grid
                    buying-system export columns).
  - etam_breaks   : header row contains both "TRP Absolute" and "Type" (the break-report
                    metric columns from DATA_SPEC.B).
  - etam_universe : any of the first 3 rows contains "Universe" and "Sample Size" (the eTAM
                    "Universe" / "Sample Size" data types named in DATA_SPEC.C.2).
  - etam_reach_rf : any of the first 3 rows contains "Cume Reach", "Reach 1+" or
                    "Incremental" (the eTAM reach & frequency schedule-tool outputs named in
                    DATA_SPEC.C.3 option B(ii)).
  - etam_duplication: any of the first 3 rows contains "Duplication" and "Exclusive" (the
                    Duplication/Exclusive Cume Reach cells named in DATA_SPEC.C.3 option
                    B(i)).
  - respondent_level: header row contains a panelist/respondent id column AND a weight
                    column (e.g. "panelist_id"/"respondent_id" + "daily_weight"/"weight"),
                    the DATA_SPEC.C.3 option A layout.
  - unknown       : none of the above matched; the file is still copied to
                    ``data/raw/unclassified/`` and reported, never silently dropped.

Only Phase 1 parsers exist (grid, etam_breaks) -- everything else is filed and reported as
"parser pending" so the pipeline plugs it in later without re-classifying it.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_EXT = {".xlsx", ".xls", ".csv", ".tsv"}

DEST_SUBDIR = {
    "grid": "grid",
    "etam_breaks": "etam",
    "etam_universe": "etam_universe",
    "etam_reach_rf": "etam_reach_rf",
    "etam_duplication": "etam_duplication",
    "respondent_level": "respondent_level",
    "unknown": "unclassified",
}


def _peek_rows(path: Path, n: int = 3) -> list[list[Any]]:
    """First n rows of the file as lists of cell values, tolerant of the file being an
    Excel workbook or a delimited text file. Returns [] if the file cannot be read at all
    (reported as unknown / unreadable, never raises past this function)."""
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path, header=None, nrows=n)
        else:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(path, header=None, nrows=n, sep=sep, engine="python")
        return [row.tolist() for _, row in df.iterrows()]
    except Exception:
        return []


def _row_text(rows: list[list[Any]]) -> str:
    cells = []
    for row in rows:
        for v in row:
            if isinstance(v, str):
                cells.append(v)
    return " | ".join(cells).lower()


def identify_file(path: str | Path) -> str:
    """Classify a file by keyword presence in its first 3 rows. See module docstring for the
    exact rule per type. Pure function: does not touch data/raw/ or data/incoming/."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXT:
        return "unknown"

    rows = _peek_rows(path, n=3)
    if not rows:
        return "unknown"
    text = _row_text(rows)

    header_row_text = " | ".join(str(v) for v in rows[0]).lower() if rows else ""
    header_row2_text = " | ".join(str(v) for v in rows[1]).lower() if len(rows) > 1 else ""
    header_candidates = header_row_text + " | " + header_row2_text

    if "station_code" in header_candidates and "day_mask" in header_candidates:
        return "grid"

    if "trp absolute" in header_candidates and "type" in header_candidates:
        return "etam_breaks"

    if "universe" in text and "sample size" in text:
        return "etam_universe"

    if "duplication" in text and "exclusive" in text:
        return "etam_duplication"

    if ("cume reach" in text or "reach 1+" in text or "incremental" in text):
        return "etam_reach_rf"

    has_id_col = any(k in header_candidates for k in ["panelist_id", "panelist id", "respondent_id", "respondent id"])
    has_weight_col = any(k in header_candidates for k in ["daily_weight", "daily weight", "weight"])
    if has_id_col and has_weight_col:
        return "respondent_level"

    return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dest_path(raw_root: Path, file_type: str, src: Path) -> Path:
    """A destination path under data/raw/<subdir>/ that never overwrites a file whose
    content hash differs from src's; identical-content files reuse the existing path."""
    subdir = raw_root / DEST_SUBDIR[file_type]
    subdir.mkdir(parents=True, exist_ok=True)
    candidate = subdir / src.name
    if not candidate.exists():
        return candidate
    if _sha256(candidate) == _sha256(src):
        return candidate  # identical content already filed; no-op copy target
    stem, suffix = src.stem, src.suffix
    i = 2
    while True:
        candidate = subdir / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
        if _sha256(candidate) == _sha256(src):
            return candidate
        i += 1


def process_incoming(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and file every new export in ``data/incoming/``.

    - Classifies each file with :func:`identify_file`.
    - grid / etam_breaks files are validated by actually parsing them with the Phase-1
      ingest modules (a parse failure is reported, the file is still filed since it may just
      need a future hardening fix, but the validation report calls it out).
    - Every other recognised type is filed under ``data/raw/<type>/`` with a note that no
      parser exists yet (Phase 1 scope).
    - ``unknown`` files are filed under ``data/raw/unclassified/`` -- never silently dropped.
    - Never overwrites a file whose content differs; a name clash gets a ``__2``, ``__3``, ...
      suffix.
    - Writes ``outputs/validation/incoming_report.md``.
    - Works on an empty ``data/incoming`` (only README.md) with no error, returning [].

    Returns a list of per-file result dicts (also embedded in the report).
    """
    from optimizer.config import project_root

    root = project_root()
    incoming_dir = root / "data" / "incoming"
    raw_root = root / "data" / "raw"
    report_path = root / "outputs" / "validation" / "incoming_report.md"

    results: list[dict[str, Any]] = []
    if incoming_dir.exists():
        candidates = sorted(
            p for p in incoming_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
        )
    else:
        candidates = []

    for src in candidates:
        file_type = identify_file(src)
        entry: dict[str, Any] = {
            "file": src.name,
            "type": file_type,
            "sha256": _sha256(src),
            "size_bytes": src.stat().st_size,
            "validation": "not attempted",
            "dest": None,
        }
        if file_type == "grid":
            try:
                from optimizer.ingest.grid import load_grid
                g = load_grid(src, cfg)
                entry["validation"] = f"PASS: parsed {len(g)} rows"
            except Exception as e:
                entry["validation"] = f"FAIL: {type(e).__name__}: {e}"
        elif file_type == "etam_breaks":
            try:
                from optimizer.ingest.breaks import load_breaks
                b = load_breaks(src, cfg)
                dates = f"{b.broadcast_date.min().date()}..{b.broadcast_date.max().date()}" if len(b) else "n/a"
                entry["validation"] = f"PASS: parsed {len(b)} rows, dates {dates}"
            except Exception as e:
                entry["validation"] = f"FAIL: {type(e).__name__}: {e}"
        elif file_type == "unknown":
            entry["validation"] = "UNKNOWN TYPE: filed to data/raw/unclassified/ for manual review"
        else:
            entry["validation"] = "FILED: no Phase-1 parser yet for this type (structural check only)"

        dest = _dest_path(raw_root, file_type, src)
        already_filed = dest.exists() and _sha256(dest) == entry["sha256"]
        if not already_filed:
            shutil.copy2(src, dest)
        entry["dest"] = str(dest.relative_to(root))
        entry["already_filed"] = already_filed
        results.append(entry)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# data/incoming processing report",
        "",
        f"Files found: {len(candidates)}",
        "",
    ]
    if not results:
        lines.append("No new files to process (data/incoming/ is empty or contains only README.md).")
    else:
        lines.append("| File | Type | Validation | Destination | sha256 |")
        lines.append("|---|---|---|---|---|")
        for r in results:
            lines.append(
                f"| {r['file']} | {r['type']} | {r['validation']} | {r['dest']} | `{r['sha256'][:12]}...` |"
            )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return results
