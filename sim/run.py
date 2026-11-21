"""Play local matches and summarize how one bot places.

    python3 -m sim.run --bot agent --matches 48 --jobs 10
    python3 -m sim.run --bot example --matches 48          # baseline: the kit's example bot
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def factory(kind: str):
    if kind == "agent":
        from brain.agent import Agent
        return Agent
    from sim.house import ExampleBot, HouseBot
    return {"house": HouseBot, "example": ExampleBot}[kind]


def play(job):
    seed, us, opp, overrides = job
    from sim.engine import Match
    if overrides:
        os.environ["AGENT_PARAMS"] = json.dumps(overrides)
    slot = seed % 16
    lineup = []
    for s in range(16):
        if s == slot:
            lineup.append((us, factory(us)))
        else:
            kind = opp if opp != "mix" else ("example" if s % 4 == 0 else "house")
            lineup.append((kind, factory(kind)))
    t0 = time.perf_counter()
    res = Match(lineup, seed, timed=(slot,)).run()
    me = res["mechs"][slot]
    me["wall_s"] = time.perf_counter() - t0
    me["seed"] = seed
    me["match_t"] = res["t"]
    return me


def summarize(rows, label=""):
    pts = [16 - r["place"] for r in rows]
    n = len(pts)
    mean = statistics.mean(pts)
    se = statistics.stdev(pts) / math.sqrt(n) if n > 1 else 0.0
    places = Counter(r["place"] for r in rows)
    causes = Counter(r["cause"] or "alive" for r in rows)
    deaths = [r["death_t"] for r in rows if r["death_t"] is not None]
    dmg = Counter()
    for r in rows:
        dmg.update(r["dmg_taken"])
    print(f"== {label} matches={n} points={mean:.2f} ±{se:.2f} (place {16 - mean:.2f})")
    print("   place histogram:", " ".join(f"{p}:{places[p]}" for p in sorted(places)))
    print("   death cause:", dict(causes))
    if deaths:
        print(f"   death time: median {statistics.median(deaths):.0f}s  min {min(deaths):.0f}s")
    print("   damage taken per match:", {k: round(v / n) for k, v in dmg.most_common()})
    print(f"   kills/match {statistics.mean(r['kills'] for r in rows):.2f}  cubes/match {statistics.mean(r['cubes'] for r in rows):.1f}")
    print(f"   act ms: mean {statistics.mean(r['act_ms_mean'] for r in rows):.2f}  p99 {max(r['act_ms_p99'] for r in rows):.1f}"
          f"  max {max(r['act_ms_max'] for r in rows):.1f}  late {sum(r['late'] for r in rows)}  errors {sum(r['errors'] for r in rows)}")
    print(f"   sim wall s/match {statistics.mean(r['wall_s'] for r in rows):.1f}")
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="agent", choices=["agent", "house", "example"])
    ap.add_argument("--opp", default="house", choices=["house", "example", "mix"])
    ap.add_argument("--matches", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--params", default="", help="JSON dict of agent parameter overrides")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    overrides = json.loads(args.params) if args.params else {}
    jobs = [(args.seed + k, args.bot, args.opp, overrides) for k in range(args.matches)]
    t0 = time.perf_counter()
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            rows = list(pool.imap_unordered(play, jobs))
    else:
        rows = [play(j) for j in jobs]
    rows.sort(key=lambda r: r["seed"])
    summarize(rows, f"{args.bot} vs {args.opp}")
    print(f"   total wall {time.perf_counter() - t0:.0f}s")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh)


if __name__ == "__main__":
    main()
