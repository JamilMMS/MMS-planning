Verified reference parsers (run from the project root):
  python reference_scripts/ingest_grid.py
  python reference_scripts/ingest_etam_breaks.py
  PYTHONPATH=reference_scripts python reference_scripts/naive_slot_match.py
Expected output is documented in docs/PRELIMINARY_FINDINGS.md.
These are starting points — harden them into src/optimizer/ingest/ with tests.
