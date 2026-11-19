# Validation — gta6-trophy-rarity-forecast-01 entry

Built 2026-11-19. The kit ships no answers and no scorer, so practice loss
cannot be measured exactly. Validation is: (a) shape/determinism checks,
(b) scenario backtests against the refresh the rules describe, (c) a
synthetic final-like case with several refreshes to exercise the fit path,
(d) sanity checks on the medians.

## Model (summary)

`main.py` implements dilution + player progression (prep §7). An Exophase
rate is earners / players tracked; both are modelled as power laws in
`u = 1 + hours since the first read`, fitted by OLS on log–log change
points and shrunk hierarchically (N exponent toward the prior 1.0; per-trophy
earner exponent toward its tier group's fitted mean, falling back to a class
prior from the trophy's description when the page is frozen). The forecast
rate is `last rate × (u_target / u_anchor)^(beta − alpha)`, where the
anchor is the last refresh. The first trophy (100%) decays slowly toward
~99%. Spreads combine a random-walk term, an exponent-uncertainty term and
a refresh-lumpiness term. Full design notes are in the `main.py` docstring.

## (a) Shape + determinism (practice cases p00/p01)

Command: `MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=<out> python3 main.py`
(run twice; `diff -r` clean; runtime ~0.08 s for both cases, limit 8 s/case).

- 55 rows per output, exactly one key `trophies`, each row exactly
  `{"id": n, "q": [q05, q25, q50, q75, q95]}`.
- All quantiles finite, in [0.01, 99.99], non-decreasing.
- Outputs byte-identical across two runs (deterministic).
- Output size ~2.8 KB per case (limit 65,536 bytes).

## (b) Scenario backtests (p00/p01)

Simulated answers: the refresh the rules describe — N jumps 638 → 3,868,
earners grow g per trophy, id 2 stays at 100% (the rules say every rate but
the 100% one fell). Mean pinball loss on the logit scale, exactly as scored.

| case | scenario | this model | starter (house baseline) | model 90% cov | starter 90% cov |
|------|----------|-----------:|-------------------------:|--------------:|----------------:|
| p00 | earners +0% | 0.244 | 1.688 | 0.98 | 0.00 |
| p00 | earners +5% | 0.234 | 1.640 | 0.98 | 0.00 |
| p00 | earners +10% | 0.225 | 1.594 | 0.98 | 0.00 |
| p00 | earners +15% | 0.217 | 1.550 | 0.98 | 0.00 |
| p00 | earners +20% | 0.218 | 1.508 | 0.98 | 0.00 |
| p00 | earners +25% | 0.220 | 1.468 | 0.98 | 0.00 |
| p00 | earners +30% | 0.226 | 1.430 | 0.98 | 0.00 |
| p00 | earners ~U[0,30%] (3 seeds) | 0.224–0.225 | 1.538–1.558 | 0.98 | 0.00 |
| p01 | earners +0% | 0.234 | 1.688 | 0.98 | 0.00 |
| p01 | earners +5% | 0.225 | 1.640 | 0.98 | 0.00 |
| p01 | earners +10% | 0.223 | 1.594 | 0.98 | 0.00 |
| p01 | earners +15% | 0.226 | 1.550 | 0.98 | 0.00 |
| p01 | earners +20% | 0.232 | 1.508 | 0.98 | 0.00 |
| p01 | earners +25% | 0.239 | 1.468 | 0.98 | 0.00 |
| p01 | earners +30% | 0.247 | 1.430 | 0.98 | 0.00 |
| p01 | earners ~U[0,30%] (3 seeds) | 0.229–0.233 | 1.538–1.558 | 0.98 | 0.00 |

The model beats the starter's ~1.51 in **20/20 scenarios** (worst model
loss 0.247). Its 90% band covers 54/55 trophies in every scenario (the one
miss is id 2, whose true 100% sits above the ~99.5% top quantile — the
honest price of not forecasting exactly 100%). The starter covers 0/55:
the refresh takes every rate outside its narrow band.

## (c) Synthetic final-like case

A synthetic f00-like case (120 reads over 2.5 days, N growing 638 → 64,008
as u^1.15 with lumpy ~6 h refreshes, rates decaying as tier power laws,
horizon 6,510 min) exercises the fit path the frozen practice data cannot.
Result: valid output, 55 rows, monotone, deterministic; mean pinball
~0.30 against the synthetic truth, 90% coverage 0.98, runtime ~0.09 s.

## (d) Sanity checks (p01 medians vs last rates)

- Platinum id 1: 0.63% → 0.10% (falls sharply; launch-night rate was 4/638
  reviewers, diluted away).
- Gold id 14 (100% completion): 0.63% → 0.11%.
- Gold id 7 (final mission): 1.10% → 0.19%.
- Early story id 3 (Chapter 1): 1.25% → 0.26%; id 4 (Chapter 2):
  1.10% → 0.22% — fall least, as the launch cohort keeps earning them.
- First trophy id 2: 100% → median 98.90% (q05–q95: 97.72–99.47).
- Highest-rate bronze id 20: 2.04% → 0.38%.

## Caveats

- Practice loss is simulated, not measured: the kit holds no answers.
- The final's history will span ~2.5 days with real refreshes, so the
  fitted exponents (not the priors) will drive the f00 forecast; the
  priors only dominate when the page is frozen, as in practice.
- The prize gate needs beating all three house entries on the final,
  including the house model forecast; the margin above is against the
  house baseline only, on simulated practice refreshes.
