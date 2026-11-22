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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bot-root', required=True)
    ap.add_argument('--engine-root', required=True)
    ap.add_argument('--fixed-work-diagnostic', action='store_true')
    ap.add_argument('--seeds', default='11,12,13,14,15,16')
    ap.add_argument('--out', required=True)
    ap.add_argument('--opp', choices=['house', 'mix'], default='house')
    ap.add_argument('--squad', action='store_true')
    args = ap.parse_args()
    root = Path(args.bot_root).resolve()
    engine_root = Path(args.engine_root).resolve()
    sys.path.insert(0, str(root))
    from bot import Bot
    import sim
    sim.__path__ = [str(engine_root/'sim')]
    from sim.engine import Match
    from sim.house import HouseBot, ExampleBot
    if args.fixed_work_diagnostic:
        # Diagnostic only: remove wall-clock-driven search/coasting differences.
        # The normal benchmark and the Docker checker retain their real budgets.
        from types import SimpleNamespace
        import brain.agent as policy
        import brain.nav as navigation
        policy.time = SimpleNamespace(perf_counter=lambda: 0.0)
        original_run = navigation.NavField.run
        def fixed_run(self, budget_s=None, max_iters=100000):
            if budget_s is not None:
                max_iters = min(max_iters, self.iters + 16)
            return original_run(self, budget_s=None, max_iters=max_iters)
        navigation.NavField.run = fixed_run

    class Measured(Bot):
        def __init__(self, hello):
            super().__init__(hello)
            self.outside = self.stalled = self.decisions = 0
            self.last_pos = self.last_move = None
            self.latencies = []
            self.first_weapon_t = None
            self.weapon_ticks = 0
            self.grab_ticks = self.blocked_grab_ticks = self.take_commands = 0
            self.energy_sum = self.low_energy_ticks = self.boost_ticks = self.boost_switches = self.retreat_ticks = 0
            self.previous_boost = False

        def act(self, obs):
            t0 = time.perf_counter()
            a = super().act(obs)
            self.latencies.append((time.perf_counter() - t0) * 1000)
            self.decisions += 1
            self.outside += not obs['self']['in_zone']
            self.energy_sum += obs['self']['energy']
            self.low_energy_ticks += obs['self']['energy'] < 10
            boosted = bool(a.get('boost'))
            self.boost_ticks += boosted
            self.boost_switches += boosted != self.previous_boost
            self.previous_boost = boosted
            self.retreat_ticks += bool(getattr(self.agent, 'retreating', False))
            armed = any(w and w['kind'] != 'gun' for w in obs['self']['weapons'])
            self.weapon_ticks += armed
            if armed and self.first_weapon_t is None:
                self.first_weapon_t = obs['t']
            self.take_commands += 'take' in a
            grab = getattr(self.agent, 'grab_pt', None) if self.agent else None
            if grab is not None:
                self.grab_ticks += 1
                x,y = obs['self']['pos'][:2]
                self.blocked_grab_ticks += not self.agent.nav.clear_line(x,y,*grab)
            p = obs['self']['pos'][:2]
            if self.last_pos is not None and self.last_move is not None:
                self.stalled += (math.hypot(*self.last_move) > .5 and
                                 math.dist(p, self.last_pos) < .1 and obs['self']['grounded'])
            self.last_pos, self.last_move = p, a.get('move', [0, 0])
            return a

    rows = []
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    for seed in map(int, args.seeds.split(',')):
        slot = seed % 16
        lineup = [('candidate', Measured) if s == slot else
                  ('example', ExampleBot) if args.opp == 'mix' and s % 4 == 0 else
                  ('house', HouseBot) for s in range(16)]
        t0 = time.perf_counter()
        match = Match(lineup, seed, timed=(slot,), squad_prey=slot if args.squad else None,
                      enforce_budget=not args.fixed_work_diagnostic)
        # Match the historical fingerprint: mech positions, pickups and crates.
        initial = {'positions': [m.pos.tolist() for m in match.mechs],
                   'pickups': match.pickups, 'crates': match.crates}
        fingerprint = hashlib.sha256(json.dumps(initial, sort_keys=True, default=lambda x: x.tolist()).encode()).hexdigest()
        result = match.run()['mechs'][slot]
        b = match.bots[slot]
        result.update(seed=seed, points=16-result['place'], initial_sha256=fingerprint,
                      outside_ticks=b.outside, stalled_ticks=b.stalled, decisions=b.decisions,
                      fallback_init=b.agent is None, fallback_ticks=b.act_errors,
                      fixed_work_diagnostic=args.fixed_work_diagnostic,
                      compute_ticks_over_50ms=sum(ms > 50 for ms in b.latencies),
                      first_weapon_t=b.first_weapon_t,weapon_ticks=b.weapon_ticks,
                      grab_ticks=b.grab_ticks,blocked_grab_ticks=b.blocked_grab_ticks,
                      take_commands=b.take_commands,
                      wall_s=time.perf_counter()-t0,
                      energy_mean=float(b.energy_sum/max(1,b.decisions)), low_energy_ticks=int(b.low_energy_ticks),
                      boost_ticks=int(b.boost_ticks), boost_switches=int(b.boost_switches),retreat_ticks=int(b.retreat_ticks))
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
