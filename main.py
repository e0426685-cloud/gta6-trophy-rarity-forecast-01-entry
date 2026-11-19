"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecasts each GTA VI trophy's earn rate on Exophase at the case's target,
for the make.fun bounty "Forecast how many GTA VI players earn each trophy
in week one".

Model: dilution + player progression (see VALIDATION.md for the design notes
and the local scenario backtests).

An Exophase rate is r_i(t) = earners_i(t) / N(t): the tracked players who
earned trophy i divided by the players Exophase tracks. Both move, and the
page only refreshes in lumps, so the last read is a stale snapshot:

* N(t) -- players tracked. Modelled as a power law in u = 1 + hours since
  the case's first read: N(u) ~ u**alpha. alpha is fitted by OLS on log N
  vs log u over the reads where N changed, shrunk toward the prior alpha=1
  (a launch-week tracked population growing roughly linearly in time; the
  launch-night refresh the rules describe -- 638 -> 3,868 tracked players
  between the first read and the 04:30Z read -- implies alpha ~= 1.04).
  The anchor is the last read where N changed (the last refresh); N_last
  is the tracked count as of that refresh, not as of the checkpoint.
* earners_i(t) -- modelled as earners_i(u) ~ u**beta_i. beta_i is fitted
  per trophy by OLS on log earners vs log u over its change points, then
  shrunk hierarchically toward its tier group's fitted mean (platinum /
  gold / silver / bronze), falling back to a class prior from the trophy's
  own description when there is nothing to fit (a frozen page). Class
  priors: story chapters earn fast (beta ~0.08-0.14), one-off feats and
  side missions slower (~0.06-0.10), grind/collectibles slowest (~0.04),
  gold/100%-completion ~0.02-0.03, platinum ~0.0 (nobody new finishes a
  ~200-hour game in week one). beta is clamped to [0, alpha]: earners
  cannot outgrow the tracked population for long.
* Rate forecast: r_i(target) = r_i(last) * (u_target / u_anchor) **
  (beta_i - alpha). The exponent (beta_i - alpha) <= 0 is the rate's decay:
  dilution wins unless earners keep pace with the tracked population.
* The first trophy (id 2, "Just Like Paradise", 100% at launch night) is
  special: every tracked player has finished the first mission, so its
  rate can only sag slowly as new tracked players join who have not yet.
  Its median decays a few points with log N growth, never to exactly 100.
* Spread: quantiles from the median on the logit scale,
  q_tau = 100 / (1 + exp(-(logit(m) + z_tau * s_i))), with
  s_i = sqrt(s_rw^2 + s_exp^2 + s_refresh^2): a random-walk term
  (0.5 * sqrt(horizon days), as in the house baseline), an exponent
  -uncertainty term (sigma_beta * ln(u_target / u_anchor), larger when
  the page is frozen and beta is prior-only), and a refresh-lumpiness
  term (Exophase refreshes arrive in jumps; the answering read is the
  page as of its last refresh before the target hour). Spreads are honest:
  the 90% band is meant to cover about 90% of trophies.

Everything is derived from the case JSON at run time: no hard-coded
practice answers, no network, no wall-clock reads, no randomness, so two
runs are byte-identical. Runtime is milliseconds for 55 trophies.

How the grader runs it: `python3 -P -s /bot/main.py` in python:3.13-slim.
Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input/), writes one
<id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output/): JSON with
exactly one key `trophies`, one row per trophy, each exactly
{"id": n, "q": [q05, q25, q50, q75, q95]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- model constants (priors, not fitted to any practice answer) ----------
ALPHA_PRIOR = 1.0        # N(u) ~ u**alpha; launch-night refresh implies ~1.04
ALPHA_SHRINK_K = 2.0     # prior weight (in change points) for alpha
BETA_SHRINK_K = 2.0      # prior weight (in change points) for beta
SIGMA_RW = 0.5           # random-walk spread, logit units per sqrt day
S_REFRESH = 0.35         # refresh lumpiness, logit units
SIGMA_BETA_PRIOR = 0.30  # exponent uncertainty when beta is prior-only
SIGMA_BETA_FLOOR = 0.08  # exponent uncertainty floor when beta is fitted
# First-trophy (rate ~100%) median decay: points lost per log N growth
# and per horizon day.
FIRST_TROPHY_DECAY_LOGN = 0.6
FIRST_TROPHY_DECAY_DAY = 0.15


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it. Whole numbers come back as
    ints, so json.dumps writes 100 where JavaScript writes 100."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def ols_slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope and its standard error. Returns (slope, se);
    se is inf when the slope is not identified."""
    n = len(xs)
    if n < 2:
        return 0.0, math.inf
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, math.inf
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    if n > 2:
        rss = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys))
        se = math.sqrt((rss / (n - 2)) / sxx) if rss > 0 else 0.0
    else:
        se = math.inf
    return slope, se


# --- trophy classification from the case's own descriptions ---------------
def classify(trophy: dict) -> tuple[str, float, float]:
    """Return (class, beta prior, spread multiplier) for one trophy.

    Classification uses only the trophy's tier and description text, so it
    works on whatever case the program is given.
    """
    tier = str(trophy.get("tier", "")).lower()
    desc = str(trophy.get("description", ""))
    if tier == "platinum":
        return "platinum", 0.00, 1.15
    if "100% Completion" in desc:
        return "completion", 0.02, 1.15
    if desc.startswith("Complete Chapter"):
        # "Complete Chapter N." -- earlier chapters are earned sooner.
        try:
            ch = int(desc.split("Chapter")[1].strip().split()[0].strip("."))
        except (IndexError, ValueError):
            ch = 2
        return "story-chapter", max(0.06, 0.16 - 0.02 * ch), 0.9
    if desc.startswith("Complete the final mission."):
        return "story-final", 0.03, 1.0
    if desc.startswith("Complete the final mission for"):
        return "story-side", 0.06, 1.0
    if desc.startswith("Complete ") and not any(c.isdigit() for c in desc):
        # "Complete <mission name>." -- a named story/side mission.
        return "story-mission", 0.10, 0.9
    grind_words = ("every", "all ", "each type", "Win ", "Discover ",
                   "Rob ", "Earn $", "Spend $", "Purchase ", "Register ",
                   "Meditate ", "Hunt ")
    if any(w in desc for w in grind_words) or any(c.isdigit() for c in desc):
        return "grind", 0.04, 1.15
    if tier == "silver":
        return "silver-misc", 0.06, 1.05
    if tier == "gold":
        return "gold-misc", 0.03, 1.1
    return "misc", 0.08, 1.0


def forecast_case(case: dict, name: str) -> dict:
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    if not history:
        # No reads at all: fall back to a flat, wide forecast rather than
        # crashing (an invalid case ranks below every valid entry).
        return {"trophies": [
            {"id": t["id"], "q": [round4(v) for v in
                                  (0.13, 0.46, 1.0, 2.17, 7.7)]}
            for t in trophies]}

    # --- time axis: u = 1 + hours since the first read -------------------
    t0 = parse_time(history[0]["at"])
    times = [parse_time(r["at"]) for r in history]
    us = [1.0 + (t - t0).total_seconds() / 3600.0 for t in times]
    target = parse_time(case["target"])
    u_target = 1.0 + (target - t0).total_seconds() / 3600.0
    if u_target <= 1.0:
        u_target = 1.0 + max(horizon_days * 24.0, 0.0)

    n_reads = len(history)
    n_tracked = [float(r.get("players_tracked") or 0) for r in history]

    # --- anchor: last read where N changed (the last refresh) ------------
    anchor_idx = 0
    for j in range(1, n_reads):
        if n_tracked[j] != n_tracked[j - 1]:
            anchor_idx = j
    u_anchor = us[anchor_idx]
    n_last = n_tracked[-1] if n_tracked[-1] > 0 else n_tracked[anchor_idx]
    if n_last <= 0:
        n_last = 1.0
    log_ratio = math.log(max(u_target, 1.0) / max(u_anchor, 1.0))
    if log_ratio < 0:
        log_ratio = 0.0

    # --- fit alpha (N growth exponent) ------------------------------------
    change_pts = [0] + [j for j in range(1, n_reads)
                       if n_tracked[j] != n_tracked[j - 1]]
    xs = [math.log(us[j]) for j in change_pts if n_tracked[j] > 0]
    ys = [math.log(n_tracked[j]) for j in change_pts if n_tracked[j] > 0]
    alpha_fit, _ = ols_slope(xs, ys)
    if len(xs) >= 2 and math.isfinite(alpha_fit):
        alpha_fit = min(2.0, max(0.0, alpha_fit))
        alpha = ((len(xs) * alpha_fit + ALPHA_SHRINK_K * ALPHA_PRIOR)
                 / (len(xs) + ALPHA_SHRINK_K))
    else:
        alpha = ALPHA_PRIOR

    # --- per-trophy beta fits ----------------------------------------------
    n_tr = len(trophies)
    # last non-null rate per trophy (most recent evidence)
    last_rate: list[float | None] = [None] * n_tr
    for r in history:
        rates = r.get("rates") or []
        for i in range(min(n_tr, len(rates))):
            if rates[i] is not None:
                last_rate[i] = float(rates[i])

    beta_fit: list[float | None] = [None] * n_tr
    sigma_beta: list[float] = [SIGMA_BETA_PRIOR] * n_tr
    for i in range(n_tr):
        pts_x: list[float] = []
        pts_y: list[float] = []
        prev_earn = None
        for j, r in enumerate(history):
            rates = r.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            earn = max(n_tracked[j] * float(rates[i]) / 100.0, 0.5)
            changed = (prev_earn is None or earn != prev_earn
                       or (j > 0 and n_tracked[j] != n_tracked[j - 1]))
            if changed or j == 0:
                pts_x.append(math.log(us[j]))
                pts_y.append(math.log(earn))
            prev_earn = earn
        if len(pts_x) >= 3:
            b, se = ols_slope(pts_x, pts_y)
            if math.isfinite(b):
                beta_fit[i] = min(alpha, max(0.0, b))
                sigma_beta[i] = min(0.5, (se if math.isfinite(se) else 0.30)
                                    + SIGMA_BETA_FLOOR)

    # Tier-group means of the fitted betas, for hierarchical shrinkage.
    tier_fitted: dict[str, list[float]] = {}
    for i, t in enumerate(trophies):
        if beta_fit[i] is not None:
            tier_fitted.setdefault(str(t.get("tier", "")).lower(),
                                   []).append(beta_fit[i])
    tier_mean = {k: sum(v) / len(v) for k, v in tier_fitted.items()
                 if len(v) >= 2}

    classes = [classify(t) for t in trophies]
    spread_rw = SIGMA_RW * math.sqrt(max(horizon_days, 0.0))
    rows = []
    for i, trophy in enumerate(trophies):
        cls, beta_prior, spread_mult = classes[i]
        tier = str(trophy.get("tier", "")).lower()
        if beta_fit[i] is not None:
            prior = tier_mean.get(tier, beta_prior)
            # At least 3 points by construction; weight the fit modestly.
            n_pts = 3
            beta = ((n_pts * beta_fit[i] + BETA_SHRINK_K * prior)
                    / (n_pts + BETA_SHRINK_K))
        else:
            beta = beta_prior
        beta = min(alpha, max(0.0, beta))

        rate = last_rate[i]
        if rate is None:
            rate = 1.0  # never seen: neutral small rate, wide spread below

        if rate >= 99.0:
            # First-trophy effect: essentially every tracked player has it;
            # it can only sag slowly as new tracked players join.
            decay = (FIRST_TROPHY_DECAY_LOGN * log_ratio
                     + FIRST_TROPHY_DECAY_DAY * horizon_days)
            median = 100.0 - min(8.0, max(0.5, decay))
            s = 0.45
        else:
            decay_exp = alpha - beta  # >= 0: the rate's decay exponent
            median = rate * math.exp(-decay_exp * log_ratio)
            median = clamp_pct(median)
            s_exp = sigma_beta[i] * log_ratio * spread_mult
            s = math.sqrt(spread_rw ** 2 + s_exp ** 2 + S_REFRESH ** 2)
            s = min(2.0, max(0.4, s))

        med_logit = logit(median)
        qs = []
        for z in Z:
            q = clamp_pct(inv_logit(med_logit + z * s))
            qs.append(round4(q))
        # Guarantee non-decreasing after rounding/clamping.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_forecast(case: dict) -> dict:
    """A valid-shaped forecast used only if a case fails unexpectedly."""
    rows = []
    for t in case.get("trophies", []):
        rows.append({"id": t["id"], "q": [0.13, 0.46, 1.0, 2.17, 7.7]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception as exc:  # never leave a case without an output
            print(f"warning: {path.name}: {exc}; using fallback",
                  file=sys.stderr)
            forecast = fallback_forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
