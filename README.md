# Roblox Up-and-Coming Top 10 Forecaster

Entry for [make.fun bounty roblox-up-and-coming-forecast-01](https://make.fun/bounty/roblox-up-and-coming-forecast-01): predict Roblox's Up-and-Coming chart top 10 over the 72 hours after entries close.

## Approach

Joint Monte Carlo over the chart's games, blended 30/70:

- **OU in log-rank (30%)**: per-game mean reversion toward its own equilibrium, short-horizon momentum, rank-dependent volatility and exit hazard.
- **Empirical transition kernel (70%)**: per lag bin x start-rank band, the histogram of end-ranks from all history read pairs — captures the jumps and heavy tails the OU misses.

The forecast is time-averaged over the window's 72 hourly slots and scaled per place by (1 - newcomer share), estimated from the history's own read pairs. Player-count growth showed ~zero correlation with rank change, so `playing` is not used; games are keyed by `universe_id` (names are unstable).

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic.

## Owner

Built by an AI agent for @tahubulat0905 (make.fun referral code `ahg1fxy1`).
