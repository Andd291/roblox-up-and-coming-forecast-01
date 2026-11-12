# Roblox forecaster — round 1 (v2, shipped config)

## What it is
`main.py` — Python 3.13 stdlib only, no network, deterministic (seed 20261112).
Joint Monte Carlo over the chart's games, two components blended, forecast =
time average over the window's 72 hourly slots, scaled per place by
(1 − newcomer_share_k).

- **MC 1 — OU in log-rank (weight 0.3):** per-game mean reversion toward its
  own equilibrium (recency-weighted, shrunk mean log-rank, θ = 0.010/h),
  short-horizon momentum (velocity, decay 18h, cap 0.05), rank-dependent
  volatility σ(r) = 0.13·1.4 / r^0.35, rank-dependent exit hazard.
- **MC 2 — empirical transition kernel (weight 0.7):** per lag bin
  (0-12/12-24/24-48/48-72h) and start-rank band, the histogram of end-ranks
  from all history read pairs, smoothed + floored (0.0005/rank). This carries
  the heavy tails MC 1 misses.
- **Newcomer share** per place estimated from the history's own pairs
  (×1.5 scale). **Rows:** top 25 per place, p ≥ 0.001.

## Why this shape — the three findings that drove it
1. **Playing-count growth does not predict rank change** (corr −0.017).
   `playing` is ignored; only ranks matter.
2. **The OU's Gaussian tails are far too thin.** A rank-16 game reaches #1
   ~1.5–2% of the time at 48–72h lags in the data; the OU gave it ~0.0005.
   In v1 this produced −8.8 per-place scores at late lags (near the −9.21
   floor) whenever a climber took a top place. The empirical kernel fixed it
   (K=24 cut: −56.93 → −44.32).
3. **Naming fewer games per place is better.** The unnamed share is divided
   by the number of unnamed games on a read's chart, so a dropped tail game
   gets a *better* probability unnamed (~0.002) than named at p = 0.0002.
   ROW_CAP 100 → 40 → 25 improved the backtest; 25 ≈ 15 (plateau), so 25
   was chosen as the middle of the plateau, not the extreme.

## Backtest (exact scorer, held-out reads within 72h of each cut)
Local backtest is a proxy only — 34 held-out reads total, and the tuning used
those same reads (mild overfitting risk; the final choice sits on a plateau,
not a peak).

| cut | held reads | bot | house baseline | diff |
|-----|-----------|-----|------|------|
| K=12 | 10 | −40.940 | −54.177 | +13.24 |
| K=18 | 10 | −36.826 | −47.436 | +10.61 |
| K=24 | 10 | −44.323 | −56.911 | +12.59 |
| K=30 | 4 | −28.861 | −30.008 | +1.15 |
| **mean** | **34** | **−37.738** | **−47.133** | **+9.40** |

Top-10 hits / agreement equal the baseline's at every cut (5.9/6.2/4.4/7.5
hits; the top10 list is persistence-ordered — the win is in the per-place
probabilities, not the ordering). v1 (OU only) mean was −44.75 (+2.38).

Sweep (one-at-a-time, cuts 12/18/24): BLEND_OU 0.3 > 0.5 > 0.7;
ROW_CAP 40 > 60 > 100; SIGMA_SCALE 1.4 > 1.0; MIN_ROW_P 0.001 > 0.0001;
NEWCOMER_SCALE 1.5 > 1.0; THETA 0.020 < 0.010; TAIL_FLOOR 0.002 ≈ 0.0005.
Combined winners: −37.96; CAP 25 instead of 40: −37.74; CAP 15: −37.76
(plateau — chose 25).

## Verification (full 34-snapshot dev history)
- Two runs byte-identical, sha256
  `67de7ad0352fce1329c81d775d364f8d1f76c6a0c7972685e3e583c5c2f44589`
- Runtime ~29 s, peak RSS ~24 MB (limits 300 s / 1024 MB)
- Format valid: 250 rows (25/place), per-place sums 0.56–0.76 (unnamed
  0.24–0.44), top10 = 10 distinct ids, 12,893 bytes

## Honest expectation
- If the +9.4 backtest edge transferred directly to the official practice
  window (baseline −43.51), the practice score would be ~−34 — around the
  best public entry (−34.7) and above the house model (−37.2). **Do not
  count on that:** the edge varies wildly by cut (+1.1 to +13.2), and the
  K=30 cut — closest to the real practice setup — shows only +1.15.
  Realistic official practice guess: **−36 to −40** (house model −37.2 is
  roughly the bar; beating it is plausible, not certain).
- The final is a different game again (72 hourly reads, denser history with
  recorder reads + `details`/`created` that this build ignores). Age-based
  exit (games aging past 28 days) is the biggest unmodeled effect for the
  final and the natural v3 direction — but it cannot be validated on the dev
  data (details are all null in the archive), so it was left out of v1.
- Files: `roblox-bot/main.py`, `roblox-eval/score.py`,
  `roblox-eval/backtest.py`, `roblox-eval/sweep2.py`, this file.
