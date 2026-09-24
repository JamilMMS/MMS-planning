# MMS TV Plan Optimizer — Claude Code Kit

## How to start (5 minutes)
1. Unzip this folder somewhere on your computer, e.g. `Documents/MMS-planning`.
   Keep the folder structure exactly as it is (the `.claude` folder is hidden on Mac/Linux —
   that's normal, don't delete it).
2. Open the folder in Claude Code:
   - **Desktop app** -> Code tab -> select this folder, or
   - **Terminal** -> `cd` into the folder -> run `claude`.
3. Open `PROMPT.md`, copy everything, paste it as your first message. Choose Opus as the
   main model.
4. Claude Code will stop at 4 gates and ask for your review. Answer, and it continues.

## What's inside
| Path | What it is |
|---|---|
| `PROMPT.md` | The kickoff prompt to paste into Claude Code |
| `CLAUDE.md` | Project memory — Claude Code reads it automatically every session |
| `.claude/agents/` | 9 specialist subagents with assigned models (Opus / Sonnet / Haiku) |
| `config/plan_config.yaml` | Budget, target, flight, caps, channel minimums, objective — edit here |
| `docs/` | Findings, data spec, Nielsen definitions, methodology, acceptance criteria, open issues |
| `data/raw/` | October grid, eTAM September breaks, Nielsen PDFs |
| `data/incoming/` | Drop new eTAM exports here later |
| `tests/fixtures/` | Nielsen worked examples used as unit tests |
| `reference_scripts/` | Verified parsers for the grid and eTAM file |

## Still needed from you (the plan runs without them, using flagged defaults)
1. eTAM break report for **August 1–31** and **September 22–30** (same layout as September).
2. Official **Universe** and **Sample Size** for TP Arabs 15+.
3. **Reach data**: respondent-level export (best) or eTAM R&F outputs for test schedules +
   duplication matrix (see `docs/DATA_SPEC.md` section C).
4. **MBC 1 grid for 25–31 October**.
5. Your answers to `docs/OPEN_ISSUES.md` (objective, channel minimums, caps...).
Drop files into `data/incoming/` and tell Claude Code "process the new files in data/incoming".

## Confidentiality
The Nielsen data here is confidential. Keep this folder local. `.gitignore` already excludes
`data/` and `outputs/` if you ever put the code on GitHub.
