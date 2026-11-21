"""Play one match and print a per-second timeline of our agent (mode, HP, threats, damage).

    python3 -m sim.trace --seed 5
"""
from __future__ import annotations

import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from brain.agent import Agent  # noqa: E402
from sim.engine import Match  # noqa: E402
from sim.house import HouseBot  # noqa: E402


class Traced(Agent):
    def act(self, obs):
        a = super().act(obs)
        me = obs["self"]
        dmg = [e for e in obs["events"] if e["type"] == "damaged"]
        line = None
        if dmg or obs["tick"] % 20 == 0 or any(e["type"] == "hit" for e in obs["events"]):
            vis = [(m["id"], round(math.dist(m["pos"][:2], me["pos"][:2])), "".join(w[0] for w in m.get("weapons", [])),
                    "C" if m.get("charging") else "") for m in obs["mechs"] if m["kind"] == "mech"]
            z = obs["zone"]
            dn = math.dist(me["pos"][:2], z["next_center"]) / max(1, z["next_radius"])
            line = (f"t={obs['t']:6.1f} alive={obs['alive']:2d} hp={me['hp']:6.0f}/{me['max_hp']:.0f} en={me['energy']:3.0f} "
                    f"mode={self.mode:6s} inz={int(me['in_zone'])} dnext={dn:4.2f} urg={self.zone_urgency():.2f} "
                    f"vis={vis} thr={len(self.threats)} cubes={me['cubes']}")
            hits = [e for e in obs["events"] if e["type"] == "hit"]
            if hits:
                line += " HIT " + ",".join(f"{e['weapon']}:{e['amount']:.0f}>{e['target']}" for e in hits)
            line += f" act=mv({a.get('move',[0,0])[0]:.1f},{a.get('move',[0,0])[1]:.1f}) b={int(bool(a.get('boost')))} f={a.get('fire')} w={[w['kind'] if w else None for w in me['weapons']]}"
            if dmg:
                line += " DMG " + ", ".join(f"{e['weapon']}:{e['amount']:.0f}{'R' if e.get('rear') else ''} by {e['by']}" for e in dmg)
        if line:
            print(line)
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    slot = args.seed % 16
    lineup = [("agent", Traced) if s == slot else ("house", HouseBot) for s in range(16)]
    res = Match(lineup, args.seed, timed=(slot,), squad_prey=slot).run()
    me = res["mechs"][slot]
    print("RESULT place", me["place"], "death", me["death_t"], me["cause"], "dmg", me["dmg_taken"], "cubes", me["cubes"])


if __name__ == "__main__":
    main()
