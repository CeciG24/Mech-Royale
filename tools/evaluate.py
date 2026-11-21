"""Paired local evaluation. This simulator is an approximation, not the judge."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='11,12,13,14,15,16')
    ap.add_argument('--out', required=True)
    ap.add_argument('--opp', choices=['house', 'mix'], default='house')
    ap.add_argument('--squad', action='store_true')
    args = ap.parse_args()
    from bot import Bot
    from sim.engine import Match
    from sim.house import HouseBot, ExampleBot

    class Measured(Bot):
        def __init__(self, hello):
            super().__init__(hello)
            self.outside = self.stalled = self.decisions = 0
            self.last_pos = self.last_move = None
            self.latencies = []

        def act(self, obs):
            t0 = time.perf_counter()
            a = super().act(obs)
            self.latencies.append((time.perf_counter() - t0) * 1000)
            self.decisions += 1
            self.outside += not obs['self']['in_zone']
            p = obs['self']['pos'][:2]
            if self.last_pos is not None and self.last_move is not None:
                self.stalled += (math.hypot(*self.last_move) > .5 and
                                 math.dist(p, self.last_pos) < .1 and obs['self']['grounded'])
            self.last_pos, self.last_move = p, a.get('move', [0, 0])
            return a

    rows = []
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for seed in map(int, args.seeds.split(',')):
        slot = seed % 16
        lineup = [('candidate', Measured) if s == slot else
                  ('example', ExampleBot) if args.opp == 'mix' and s % 4 == 0 else
                  ('house', HouseBot) for s in range(16)]
        t0 = time.perf_counter()
        match = Match(lineup, seed, timed=(slot,), squad_prey=slot if args.squad else None)
        # Initial conditions fingerprint proves identical spawns, loadouts and zone per pair.
        initial = {'positions': [m.pos.tolist() for m in match.mechs],
                   'pickups': match.pickups, 'crates': match.crates}
        fingerprint = hashlib.sha256(json.dumps(initial, sort_keys=True, default=lambda x: x.tolist()).encode()).hexdigest()
        result = match.run()['mechs'][slot]
        b = match.bots[slot]
        result.update(seed=seed, points=16-result['place'], initial_sha256=fingerprint,
                      outside_ticks=b.outside, stalled_ticks=b.stalled, decisions=b.decisions,
                      fallback_init=b.agent is None, fallback_ticks=b.act_errors,
                      wall_s=time.perf_counter()-t0)
        rows.append(result)
        print(json.dumps({'seed':seed, 'points':result['points'], 'kills':result['kills'],
                          'fallback':b.act_errors, 'late':result['late']}), flush=True)
    summary = {'sha':sha, 'seeds':args.seeds, 'opp':args.opp, 'squad':args.squad,
               'mean_points':statistics.mean(r['points'] for r in rows),
               'late':sum(r['late'] for r in rows), 'errors':sum(r['errors'] for r in rows),
               'fallback_ticks':sum(r['fallback_ticks'] for r in rows),
               'fallback_init':sum(r['fallback_init'] for r in rows),
               'outside_ticks':sum(r['outside_ticks'] for r in rows),
               'stalled_ticks':sum(r['stalled_ticks'] for r in rows),
               'max_act_p99_ms':max(r['act_ms_p99'] for r in rows)}
    Path(args.out).write_text(json.dumps({'summary':summary, 'matches':rows}, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
