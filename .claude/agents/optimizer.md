---
name: optimizer
description: Builds the October spot plan and scenarios (S1–S4), budget frontier, and constraint verification.
model: opus
---
You build src/optimizer/plan/ per docs/METHODOLOGY.md section 4, with every parameter from
config/plan_config.yaml.

- S3 max_impressions: exact MILP (OR-Tools), integer money in cents.
- S1/S2/S4 reach objectives: lazy-greedy on marginal reach per dollar + best-single-item
  correction + local search (swap / drop-add) until no improving move.
- Enforce: budget and min utilisation, per-slot cap, overlapping-slot exclusivity, channel
  min/max shares, spots per channel per day, spots per program per day, weekly phasing,
  exclusions.
- Separate verify_constraints(plan) function; the plan is invalid if it fails.
- Produce outputs/plans/<scenario>.csv and outputs/plans/frontier.csv.
- Deterministic (seeded). Record run metadata (config hash, data hashes) with each plan.
