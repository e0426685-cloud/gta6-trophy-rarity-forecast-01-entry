# gta6-trophy-rarity-forecast-01-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/e0426685-cloud/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/e0426685-cloud/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A deterministic Python entry for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty. `main.py` forecasts every GTA VI PlayStation 5 trophy's earn rate on Exophase at the case's target, as five quantiles (5/25/50/75/95%) per trophy.

## Model

An Exophase rate is `earners / players tracked`, and both move — while the page itself only refreshes in lumps, so the last read is a stale snapshot. The model is **dilution + player progression**:

- **Players tracked N(t)** grows as a power law in `1 + hours since the first read`, fitted from the reads where N changed (shrunk toward a prior exponent of 1, which the launch-night refresh — 638 → 3,868 tracked players — implies). The anchor is the last refresh, not the checkpoint.
- **Earners per trophy** grow as a power law too, fitted per trophy and shrunk hierarchically toward its tier group's mean (platinum/gold/silver/bronze), falling back to a class prior read from the trophy's own description when the page is frozen: story chapters earn fast, grind/collectibles slow, platinum ~flat (nobody new finishes a ~200-hour game in week one).
- **Rate forecast**: last rate × (time ratio)^(earner exponent − N exponent). Dilution wins unless earners keep pace.
- **First trophy** ("Just Like Paradise", 100% at launch night) is special: every tracked player has finished the first mission, so its median only sags slowly, to ~99% — never exactly 100.
- **Spreads** are honest: a random-walk term, an exponent-uncertainty term (wider when the page is frozen and the fit is prior-only), and a refresh-lumpiness term, targeting ~0.9 coverage of the 90% band.

Everything is fitted from the case the program is given — no hard-coded practice answers, no network, no wall-clock reads, no randomness. See `VALIDATION.md` for the local checks and scenario backtests.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. Runs in ~0.1 s per case, well under the 8 s limit, and two runs are byte-identical.

## Layout

- `main.py` — the whole entry (single file)
- `VALIDATION.md` — model notes, shape/determinism checks, scenario backtests
- `LICENSE` — MIT
- `.makefun/owner.json` — the entry's wallet

---

<sub>Entered on [make.fun](https://make.fun/?ref=52532a96&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
