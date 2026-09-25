# ESTIMATE reach model: derivation, parameters, assumptions

**Status: ESTIMATE, a model only, not validated against measured reach.** Every reach
figure this mode produces carries `method = "ESTIMATE"`. It is superseded automatically
when eTAM R&F outputs (CALIBRATED) or respondent data (EXACT) arrive (DATA_SPEC C.3). The
switch is `reach.mode` in `config/plan_config.yaml`, and the interface stays the same.

Re-run after any change to `data/processed/breaks.parquet` or to `reach.estimate.*`:

    .venv/bin/python -m optimizer.reach.fit --config config/plan_config.yaml

This writes `data/processed/reach_params.json` and refreshes the table at the bottom of this
file. The engine warns if breaks.parquet changed since the last fit.

## 1. What the break data can and cannot tell us
The eTAM break file gives, for each commercial break, its average audience
(`rating_abs = TRP_abs × 60 / break_sec`) and its **Unduplicated Reach**: the people who
viewed at least the reach threshold of that break. It gives **no** joint reach of two
breaks. So the duplication between spots (the thing a reach curve is about) is not
observed. The data can anchor three things:

| Evidence in the data | What it anchors |
|---|---|
| reach_abs / rating_abs as a function of break length L | audience turnover rate λ, which gives the mean viewing-session length τ = 1/λ |
| hourly break-audience profile (all breaks, including zero-rated ones) | daily audience-minutes of each channel (AM) |
| largest single-break reach of each day | hard lower bound on each channel's daily cume |

Everything beyond that is an **ASSUMPTION** with a config key, a default and a sensitivity
table (§6).

## 2. Per-channel curve
R_c(g) = Rmax_c · g / (k_c + g) (hyperbolic) or Rmax_c · (1 − e^(−g/k_c)) (negative
exponential). Here g = GRP % of the channel's spots in the schedule (Σ aud_abs / U × 100),
R and Rmax are in % of universe, and for both forms the initial slope is dR/dg(0) = Rmax/k.

### 2.1 Turnover (data)
For each channel c, fit reach_abs = a · (r0 + λ·L) by non-negative weighted least
squares (weights 1/a, Poisson-type variance). The fit uses non-event breaks with
a ≥ `min_rating_abs_for_ratio` (5,000, about 4 panelists at the typical ~1,150 weight;
below that the ratio is pure panel granularity). λ is the inflow of new viewers per second
per current viewer. In a stationary on/off viewing process this equals 1/τ, where τ is the
mean viewing-session length. A channel with fewer than `min_breaks_for_channel_fit` usable
breaks takes the pooled λ.

### 2.2 Daily cume (data plus ASSUMPTION m)
* AM_c = Σ over broadcast hours 03–26 of the time-weighted mean break audience in that
  hour (averaged over days) × 60 min. Zero-rated breaks count as zero audience.
* Daily viewing sessions S_c = AM_c / τ_c, since each session lasts τ on average.
* **Daily cume D_c = S_c / m**, where m = `sessions_per_daily_viewer` (**ASSUMPTION**,
  default 1.5) is the average number of separate sessions a daily viewer has with the
  channel. D_c is bounded below by the mean daily maximum single-break reach (LB, a hard
  bound from the data) and above by S_c (m = 1, every session a different person) and U.
  The table reports the implied time spent per daily viewer (AM/D) as a plausibility check.
* Bias note: τ is measured on breaks, where switching is higher than in programme content.
  This shortens τ and inflates S_c. m partly absorbs that bias, and it is one reason m > 1
  is the default.
* If eTAM *Average Daily Reach* by channel becomes available (DATA_SPEC C.3 iii), put it in
  `daily_reach_override` and it replaces the estimate, labelled as OBSERVED in the JSON.

### 2.3 Rmax: flight cume (ASSUMPTION rho)
Each person's probability of viewing the channel on a given day is p ~ Beta, with mean D_c/U
and intra-person correlation ρ = `day_to_day_rho` (**ASSUMPTION**, default 0.45). Then
Rmax_c = 1 − B(α, β+N)/B(α, β) over N = flight days (28). The implied next-day repeat rate
d + ρ(1 − d) is about 0.45–0.54, which is plausible for habitual channel viewing. It is capped
at `rmax_cap_pct` (95). Rmax is an estimate of the channel's flight cume, the asymptote a
spot schedule on that channel approaches. The data does **not** contain it.

### 2.4 k: initial slope (data, capped)
The reach of a single 30" spot per GRP of it is r_spot = r0 + λ·30, which is 1.01–1.05 on
the data. Under Nielsen spot definitions, one spot's GRP Absolute equals its reach (OTS = 1).
Our GRP proxy (average audience) understates it by that same 1–5%. The initial slope is
therefore s0 = min(r_spot, `initial_slope_cap` = 1.0), and k_c = Rmax_c / s0. The cap keeps
reach ≤ GRP (OTS ≥ 1), and the raw ratio is reported alongside. With s0 = 1 the curve's
shape is set by Rmax alone. That is why the ±30% k sensitivity is the key robustness check.

### 2.5 Curve form (data, weak)
The only accumulation data in the file is reach against break length. Both saturating forms
are fitted as reach_abs = a·(r0 + c·h(L; T)), and the pooled weighted SSE decides. Within
breaks of 2–800 s, turnover is nearly linear, so the forms are practically
indistinguishable. A relative SSE difference below `form_tie_tolerance` counts as a tie and
defaults to **hyperbolic**, which is the conservative choice (lower reach at high GRP, and a
Hofmans/Agostini-type form widely used for TV). `model_form` can force either form.
Plausibility check: for two small equal spots in the same channel, the curve implies a
duplication index of 2·s0²/Rmax (hyperbolic) or s0²/Rmax (negexp) relative to random. This
is reported per cell ("dup index"). Values of about 4–17 for the five main channels are
at or above the classic same-channel "duplication of viewing" constants (about 1.5–3 in
UK/US studies) for the big channels, and higher for smaller ones. ACTION and MAX get very
large indices (their few viewers are reached again and again), which follows from their tiny
Rmax.

## 3. Cross-channel combination
Sainsbury (independence): R = 1 − Π_c (1 − R_c). The duplication hook (`reach.duplication:
matrix`, CALIBRATED only) replaces it with a pairwise duplication-index adjustment built from
eTAM Duplication Cume Reach (see `curves.combine_pairwise`). Channel × daypart cells
(`cell_granularity: channel_daypart`) are also combined by Sainsbury. That **overstates**
reach, because dayparts of one channel share viewers, so the default is `channel`.

Sanity caps enforced on every result (`engine.check_sanity`, `curves.cap_reach`):
0 ≤ reach ≤ U, reach 1+ ≤ Σ_c R_c ≤ Σ_c Rmax_c, reach 1+ ≤ GRP Absolute (OTS ≥ 1),
reach 3+ ≤ reach 1+, and Reach N+ non-increasing in N.

## 4. Frequency distribution (reach 3+)
Per cell, the exposure count is negative binomial with mean m = g/100 (exposures per
universe member). Its dispersion is solved so that P(0) = 1 − R_c(g), which makes it
consistent with the curve. If R_c is above the Poisson limit 1 − e^(−m) (possible only
when k is scaled down), the model uses a zero-modified, zero-truncated Poisson with
OTS = m/R_c. Mass above the number of spots bought in the cell, or above
`reach.max_frequency` (K = 10), is lumped into the top bucket. Cells are convolved under
independence. Reach N+ = reach 1+ × P(X ≥ N | X ≥ 1) from the convolution, which is
identical to the plain convolution under Sainsbury. **ESTIMATE.**
Note: a larger k means more duplication, so it lowers reach 1+ but raises frequency among
those reached. Reach 3+ is therefore *not* monotone in k (see the sensitivity output).

## 5. Speed
`marginal_reach` is O(1) under Sainsbury. The state caches, per cell, the product of the
other cells' (1 − R). With a duplication matrix it is O(#cells). `marginal_reach_n`
(reach N+) costs one NBD solve plus one length-K convolution against a cached "all other
cells" distribution.

## 6. Parameters (auto-generated, do not edit by hand)
Symbols: AM = daily break-audience minutes; LB/UB = daily-cume bounds; TSV = time spent per
daily viewer; s0 = initial slope (reach per GRP at g = 0); dup index = implied
same-cell duplication index of two small spots (§2.5).

<!-- PARAMS:BEGIN -->
_Fitted parameter tables are written by `python -m optimizer.reach.fit` to `outputs/validation/reach_estimate_params.md` (git-ignored: derived from confidential Nielsen data). The committed doc holds only the method._
<!-- PARAMS:END -->

## 7. Assumptions register (all labelled ESTIMATE)
| # | Assumption | Config key | Default | Evidence / why | Effect of going wrong |
|---|---|---|---|---|---|
| E1 | Sessions per daily viewer | `reach.estimate.sessions_per_daily_viewer` | 1.5 | Not in data. The implied TSV of 40–75 min per daily viewer is plausible | Rmax scales with it (see grid, m = 1…2) |
| E2 | Day-to-day correlation ρ of channel viewing | `reach.estimate.day_to_day_rho` | 0.45 | Not in data. Implied repeat rate is about 0.5 | Rmax shifts about ±25% over ρ 0.3–0.6 |
| E3 | Initial slope capped at 1 | `reach.estimate.initial_slope_cap` | 1.0 | Data ratio 1.01–1.05. The cap guarantees OTS ≥ 1 | ≤ 5% at low GRP only |
| E4 | Channels independent (Sainsbury) | `reach.duplication` | sainsbury | No duplication data yet | Typically overstates combined reach when channels share viewers |
| E5 | Curve form | `reach.estimate.model_form` | auto → hyperbolic (tie) | Break data cannot discriminate | negexp gives higher reach at high GRP |
| E6 | Break-based turnover equals session turnover | none | none | Only breaks are observed | τ too short means S too high; partly absorbed by E1 |
| E7 | Universe | `target.universe` | 11,110,000 (inferred) | OPEN_ISSUES #3 | All % scale with U |
| E8 | Reach threshold / exposure rule | `reach.min_seconds` | null | Not in the PDFs (NIELSEN_DEFINITIONS §2) | Only affects EXACT; ESTIMATE inherits eTAM's break-reach rule |
