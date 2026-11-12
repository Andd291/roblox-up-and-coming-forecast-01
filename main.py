#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster v2 (make.fun roblox-up-and-coming-forecast-01).

Two Monte Carlos, blended, over the chart's games; the forecast is the time
average over the window's hourly slots of the simulated place frequencies,
scaled per place by (1 - newcomer_share_k), where newcomer_share_k is
estimated from the history's own read pairs.

MC 1 — OU in log-rank (per-game signal): each game's log-rank mean-reverts
toward its own equilibrium (recency-weighted, shrunk mean of its observed
log-ranks) with short-horizon momentum (a per-game log-rank velocity that
decays in the simulation) and rank-dependent volatility; games exit the chart
with a rank-dependent hazard. This carries persistence and each game's own
trajectory, but its Gaussian tails are far too thin: a rank-16 game essentially
never reaches #1 in it, while the data says it does ~1.5-2% of the time at
48-72h lags.

MC 2 — empirical transition kernel (heavy tails): for each lag bin and
start-rank band, the histogram of end-ranks from all history read pairs,
smoothed over ranks with a floor on every rank. This directly encodes the
jumps (7% of game-reads climb >= 10 ranks), the mean reversion, the exits
and the heavy upper tail that MC 1 misses. It conditions only on the current
rank, not on the game's own history.

Blend: p = BLEND_OU * p_OU + (1 - BLEND_OU) * p_emp. The unnamed mass covers
brand-new games (a read's unnamed games share it equally, so naming every
plausible game matters: rows are kept down to MIN_ROW_P so the unnamed pool
is only true newcomers, not dropped tail games).

What the data said NOT to use: player-count growth has ~zero correlation
(-0.017) with rank change, so `playing` is ignored. Names are unstable
(77 universes changed name); universe_id is the key.

Stdlib only, no network, deterministic (fixed seed, fixed iteration order).
Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR (else
/output), window/places from task.json, never hardcoded.
"""
import bisect
import json
import math
import os
import random
from datetime import datetime

# --- tunables (checked on the local backtest; see ROUND1.md) ---
THETA = 0.010          # mean reversion per hour, log-rank (OU)
SIGMA_A = 0.13         # sigma(r) = SIGMA_A / r**SIGMA_B
SIGMA_B = 0.35
SIGMA_SCALE = 1.4
VEL_HALF_LIFE_H = 24.0  # recency weighting when estimating velocity
VEL_DECAY_H = 18.0      # momentum decay inside the OU simulation
VEL_CAP = 0.05          # log-rank per hour
MU_HALF_LIFE_H = 72.0   # recency weighting for equilibrium log-rank
MU_SHRINK_K = 2.0       # pseudo-observations pulling mu toward log(49)
MU_PRIOR = math.log(49.0)
N_PATHS = 1200
SEED = 20261112
NEWCOMER_SCALE = 1.5   # multiplies the estimated per-place newcomer share
MIN_ROW_P = 0.001     # keep named rows down to this (tail games stay named)
ROW_CAP = 25          # at most this many rows per place (smallest p's dropped)
BLEND_OU = 0.3         # weight of the OU MC in the blend (rest: empirical)
TAIL_FLOOR = 0.0005    # minimum mass per end-rank in the empirical kernel
# exit hazard per hour by rank band (from measured P(gone) at ~36h)
HAZARD = ((10, 0.0003), (20, 0.0014), (40, 0.0030), (10 ** 9, 0.0090))
# games absent from the last snapshot but seen within this many hours may re-enter
REENTER_WINDOW_H = 48.0
# start-rank bands for the empirical kernel: (lo, hi, center log-rank)
BANDS = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 40), (41, 10 ** 9)]
LAG_BINS = [(0.0, 12.0), (12.0, 24.0), (24.0, 48.0), (48.0, 72.0)]
LAG_BIN_HOURS = [12.0, 12.0, 24.0, 24.0]
GONE_RANK = 121  # sentinel end-rank: game left the chart
MAX_RANK = 120


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def sigma_of(rank):
    return SIGMA_A * SIGMA_SCALE / max(rank, 1.0) ** SIGMA_B


def hazard_of(rank):
    for bound, h in HAZARD:
        if rank <= bound:
            return h
    return HAZARD[-1][1]


def band_index(rank):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= rank <= hi:
            return i
    return len(BANDS) - 1


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(in_dir, "task.json")) as f:
        task = json.load(f)
    places = int(task["places"])
    window = task["window"]
    n_hours = int(window["slots"])  # hourly slots
    out_name = task.get("output", "forecast.json")

    snaps = []
    with open(os.path.join(in_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    snaps.sort(key=lambda s: s["at"])
    times = [parse_time(s["at"]) for s in snaps]
    last_t = times[-1]
    last_chart = snaps[-1]["up_and_coming"]
    last_rank = {g["universe_id"]: g["rank"] for g in last_chart}

    t0 = times[0]
    obs = {}  # uid -> list of (hours since first, log rank)
    for t, s in zip(times, snaps):
        h = (t - t0).total_seconds() / 3600.0
        for g in s["up_and_coming"]:
            obs.setdefault(g["universe_id"], []).append((h, math.log(g["rank"])))
    last_h = (last_t - t0).total_seconds() / 3600.0

    rank_maps = [{g["universe_id"] for g in s["up_and_coming"]} for s in snaps]
    rank_of = [{g["universe_id"]: g["rank"] for g in s["up_and_coming"]} for s in snaps]
    top_maps = [[g["universe_id"] for g in s["up_and_coming"][:places]] for s in snaps]
    n_snaps = len(snaps)

    # --- newcomer share per place, from all read pairs, lag-binned ---
    cnt = [[0] * (places + 1) for _ in LAG_BINS]
    new = [[0] * (places + 1) for _ in LAG_BINS]
    # --- empirical kernel histograms: hist[bin][band][end_rank] ---
    hist = [[[0.0] * (MAX_RANK + 2) for _ in BANDS] for _ in LAG_BINS]
    for i in range(n_snaps):
        for j in range(i + 1, n_snaps):
            lag = (times[j] - times[i]).total_seconds() / 3600.0
            b = None
            for bi, (lo, hi) in enumerate(LAG_BINS):
                if lo <= lag < hi:
                    b = bi
                    break
            if b is None:
                continue
            chart_i = rank_maps[i]
            for k in range(1, places + 1):
                cnt[b][k] += 1
                if top_maps[j][k - 1] not in chart_i:
                    new[b][k] += 1
            for uid, r in rank_of[i].items():
                e = rank_of[j].get(uid, GONE_RANK)
                if e > MAX_RANK:
                    e = GONE_RANK
                hist[b][band_index(r)][e] += 1.0
    newcomer = [0.0] * (places + 1)
    for k in range(1, places + 1):
        tot = 0.0
        for b in range(len(LAG_BINS)):
            n = cnt[b][k]
            overall = sum(new[b][1:]) / max(1, sum(cnt[b][1:]))
            rate = (new[b][k] + 30.0 * overall) / (n + 30.0) if n else overall
            tot += LAG_BIN_HOURS[b] * rate
        newcomer[k] = min(0.6, max(0.01, tot / 72.0) * NEWCOMER_SCALE)
    sm = newcomer[:]
    for k in range(1, places + 1):
        lo = newcomer[k - 1] if k > 1 else newcomer[k]
        hi = newcomer[k + 1] if k < places else newcomer[k]
        sm[k] = 0.25 * lo + 0.5 * newcomer[k] + 0.25 * hi
    newcomer = sm

    # --- per-game state ---
    games = []  # (uid, x0, mu, v0, start_rank_for_kernel)
    for uid in sorted(obs):
        o = obs[uid]
        last_seen_h, last_logr = o[-1]
        if uid in last_rank:
            x0 = math.log(last_rank[uid])
            start_rank = last_rank[uid]
            seen_recently = True
        else:
            if last_h - last_seen_h > REENTER_WINDOW_H:
                continue
            x0 = math.log(105.0)
            start_rank = 105
            seen_recently = False
        wsum = 0.0
        vsum = 0.0
        for h, lr in o:
            w = 0.5 ** ((last_h - h) / MU_HALF_LIFE_H)
            wsum += w
            vsum += w * lr
        mu = (vsum + MU_SHRINK_K * MU_PRIOR) / (wsum + MU_SHRINK_K)
        v = 0.0
        if len(o) >= 2 and seen_recently:
            wsum = 0.0
            vsum = 0.0
            for (h1, lr1), (h2, lr2) in zip(o, o[1:]):
                dh = h2 - h1
                if dh <= 0:
                    continue
                w = 0.5 ** ((last_h - h2) / VEL_HALF_LIFE_H)
                wsum += w
                vsum += w * (lr2 - lr1) / dh
            if wsum > 0:
                v = vsum / wsum
        v = max(-VEL_CAP, min(VEL_CAP, v))
        games.append((uid, x0, mu, v, start_rank))

    n_games = len(games)
    uids = [g[0] for g in games]
    x0s = [g[1] for g in games]
    mus = [g[2] for g in games]
    v0s = [g[3] for g in games]
    start_ranks = [g[4] for g in games]

    # --- empirical kernel: per game, per lag bin, a CDF over end-rank ---
    # smooth each band histogram over end-rank, add a floor, normalise
    band_cdf = []  # band_cdf[bin][band] = (ranks list, cdf list)
    for b in range(len(LAG_BINS)):
        per_band = []
        for bi in range(len(BANDS)):
            h = hist[b][bi]
            tot = sum(h)
            sm_h = [0.0] * (MAX_RANK + 2)
            for r in range(1, MAX_RANK + 2):
                left = h[r - 1] if r > 1 else 0.0
                right = h[r + 1] if r < MAX_RANK + 1 else 0.0
                sm_h[r] = 0.25 * left + 0.5 * h[r] + 0.25 * right + TAIL_FLOOR
            s = sum(sm_h[1:])
            cdf = []
            acc = 0.0
            ranks = list(range(1, MAX_RANK + 2))
            for r in ranks:
                acc += sm_h[r] / s
                cdf.append(acc)
            cdf[-1] = 1.0
            per_band.append((ranks, cdf))
        band_cdf.append(per_band)
    # interpolate between adjacent bands by log start-rank
    band_centers = [math.log(math.sqrt(lo * hi)) for lo, hi in BANDS]
    game_cdf = []  # game_cdf[game][bin] = (ranks, cdf)
    for sr in start_ranks:
        lx = math.log(max(sr, 1))
        if lx <= band_centers[0]:
            b0, b1, w = 0, 0, 0.0
        elif lx >= band_centers[-1]:
            b0, b1, w = len(BANDS) - 1, len(BANDS) - 1, 0.0
        else:
            b0 = 0
            while band_centers[b0 + 1] < lx:
                b0 += 1
            b1 = b0 + 1
            w = (lx - band_centers[b0]) / (band_centers[b1] - band_centers[b0])
        per_game = []
        for b in range(len(LAG_BINS)):
            r0, c0 = band_cdf[b][b0]
            r1, c1 = band_cdf[b][b1]
            cdf = [a * (1.0 - w) + c * w for a, c in zip(c0, c1)]
            cdf[-1] = 1.0
            per_game.append((r0, cdf))
        game_cdf.append(per_game)

    # --- Monte Carlos ---
    rng = random.Random(SEED)
    gauss = rng.gauss
    rnd = rng.random
    counts_ou = [[0] * n_games for _ in range(places + 1)]
    counts_emp = [[0] * n_games for _ in range(places + 1)]
    vel_decay = math.exp(-1.0 / VEL_DECAY_H)

    def lag_bin(hour):
        if hour <= 12:
            return 0
        if hour <= 24:
            return 1
        if hour <= 48:
            return 2
        return 3

    for _path in range(N_PATHS):
        xs = x0s[:]
        vs = v0s[:]
        alive = [True] * n_games
        for hour in range(1, n_hours + 1):
            order = []
            for i in range(n_games):
                if not alive[i]:
                    continue
                r = math.exp(xs[i])
                if rnd() < hazard_of(r):
                    alive[i] = False
                    continue
                xs[i] += vs[i] - THETA * (xs[i] - mus[i]) + sigma_of(r) * gauss(0.0, 1.0)
                vs[i] *= vel_decay
                order.append((xs[i], i))
            order.sort()
            for k in range(1, min(places, len(order)) + 1):
                counts_ou[k][order[k - 1][1]] += 1
            # empirical draw for this hour's lag bin
            b = lag_bin(hour)
            eranks = []
            for i in range(n_games):
                ranks, cdf = game_cdf[i][b]
                er = ranks[bisect.bisect_left(cdf, rnd())]
                if er < GONE_RANK:
                    eranks.append((er, i))
            eranks.sort()
            for k in range(1, min(places, len(eranks)) + 1):
                counts_emp[k][eranks[k - 1][1]] += 1

    denom = float(N_PATHS * n_hours)
    probabilities = []
    expected_place = {}
    for k in range(1, places + 1):
        keep = (1.0 - newcomer[k])
        for i in range(n_games):
            c = BLEND_OU * counts_ou[k][i] + (1.0 - BLEND_OU) * counts_emp[k][i]
            if c <= 0:
                continue
            p = (c / denom) * keep
            if p < MIN_ROW_P:
                continue
            probabilities.append({"universe_id": uids[i], "place": k, "p": round(p, 6)})
            expected_place[uids[i]] = expected_place.get(uids[i], 0.0) + k * p
    total_p = {}
    for row in probabilities:
        total_p[row["universe_id"]] = total_p.get(row["universe_id"], 0.0) + row["p"]
    ranked = sorted(
        uids,
        key=lambda u: (
            -total_p.get(u, 0.0),
            expected_place.get(u, 0.0) / total_p[u] if total_p.get(u, 0.0) > 0 else 999.0,
        ),
    )
    top10 = ranked[:places]

    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    # cap: at most 100 rows per place (drop the smallest p's; they rejoin the
    # unnamed pool, whose per-game share stays healthy because only true
    # newcomers are unnamed once ~100 games are named)
    capped = []
    for k in range(1, places + 1):
        rows_k = [r for r in probabilities if r["place"] == k]
        capped.extend(rows_k[:ROW_CAP])
    probabilities = capped
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, out_name), "w") as f:
        f.write(json.dumps({"probabilities": probabilities, "top10": top10},
                           separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
