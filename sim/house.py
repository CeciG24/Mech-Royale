"""Opponents for the local simulator.

HouseBot approximates the built-in bots seen in the kit's recorded match: it loots a second
weapon early, keeps a preferred range (≈80-150 m), fires the laser at full charge, uses the
gun with some lead, rotates with the zone, shoots crates and NPCs when nothing else is around,
and turns toward whatever hits it. ExampleBot is the kit's example_bot.py, unchanged in spirit.
"""
from __future__ import annotations

import math
import random
from collections import OrderedDict

from brain.aim import lead, mortar_pitch, wrap, yaw_pitch
from brain.nav import NavField
from sim.engine import get_map

_FIELDS: "OrderedDict[tuple, NavField]" = OrderedDict()


def zone_field(nav, cx: float, cy: float, r: float) -> NavField:
    key = (round(cx, 1), round(cy, 1), round(r, 1))
    f = _FIELDS.get(key)
    if f is None:
        f = NavField(nav, nav.circle_mask(cx, cy, max(8.0, r * 0.6)))
        f.run()
        _FIELDS[key] = f
        while len(_FIELDS) > 24:
            _FIELDS.popitem(last=False)
    return f


class HouseBot:
    name = "house"

    def __init__(self, hello: dict):
        self.cfg = hello["config"]
        self.rng = random.Random(hello["seed"])
        self.world, self.nav = get_map(self.cfg)
        self.prey = hello.get("prey")
        role = hello.get("role")
        # Practice squad: 4 easy, 4 medium, 4 hard, 3 expert. Easy closes in and guns;
        # expert holds laser range and leads shots.
        roles = {
            "easy": (45.0, 0.45, 0.0, False, "lightning"),
            "medium": (90.0, 0.7, 0.5, True, "laser"),
            "hard": (120.0, 0.9, 1.0, True, "laser"),
            "expert": (150.0, 1.0, 1.0, True, "laser"),
        }
        if role in roles:
            self.pref, self.aggr, self.lead_frac, self.use_laser, self.want = roles[role]
        else:
            self.pref = self.rng.uniform(80, 150)
            self.aggr = self.rng.uniform(0.6, 1.0)
            self.lead_frac = self.rng.choice([0.0, 0.5, 1.0])
            self.use_laser = True
            self.want = self.rng.choice(["laser", "laser", "lightning", "mortar"])
        self.strafe = 1.0
        self.strafe_t = 0.0
        self.last = None
        self.stuck = 0.0
        self.unstick_until = -1.0
        self.unstick_dir = (1.0, 0.0)
        self.wander = None
        self.look = None
        self.hunt = None

    # ------------------------------------------------------------------ helpers
    def _goto(self, x, y, gx, gy):
        L = math.hypot(gx - x, gy - y) or 1.0
        ux, uy = (gx - x) / L, (gy - y) / L
        if self.nav.clear_line(x, y, gx, gy):
            return ux, uy
        best, bs = (ux, uy), -1e9
        for k in range(16):
            ang = 2 * math.pi * k / 16
            cx, cy = math.cos(ang), math.sin(ang)
            if not self.nav.clear_line(x, y, x + cx * 12, y + cy * 12):
                continue
            s = cx * ux + cy * uy
            if s > bs:
                best, bs = (cx, cy), s
        return best

    def act(self, obs: dict) -> dict:
        me = obs["self"]
        x, y, z = me["pos"]
        t = obs["t"]
        a: dict = {}
        if obs.get("offer"):
            a["pick"] = self.rng.randrange(len(obs["offer"]["choices"]))
        zone = obs["zone"]
        nc, nr = zone["next_center"], zone["next_radius"]
        weapons = me["weapons"]
        kinds = [w["kind"] if w else None for w in weapons]
        eye = (x, y, z + me["stats"].get("muzzle_height", 7))
        fire = [False, False]
        move = None
        boost = False

        for e in obs["events"]:
            if e["type"] == "damaged" and e.get("bearing") is not None:
                self.look = (e["bearing"], t)
        for s in obs["sounds"]:
            if s["kind"] in ("gun", "laser", "lightning", "mortar", "explosion") and s["dist"] != "far":
                if self.look is None or t - self.look[1] > 2.0:
                    self.look = (s["bearing"], t)
                if self.hunt is None and self.rng.random() < 0.15 * self.aggr:
                    d = 40 if s["dist"] == "near" else 110
                    self.hunt = (x + d * math.cos(s["bearing"]), y + d * math.sin(s["bearing"]), t)

        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        if self.prey is not None:
            enemies = [m for m in enemies if m["id"] == self.prey]
        enemies.sort(key=lambda m: math.dist(m["pos"][:2], (x, y)))
        npcs = sorted((m for m in obs["mechs"] if m["kind"] == "npc"),
                      key=lambda m: math.dist(m["pos"][:2], (x, y)))
        hit_by_npc = any(e["type"] == "damaged" and (e.get("by") or 0) >= 100 for e in obs["events"])
        target = None
        chase = 2500 if self.prey is not None else 230
        if enemies and math.dist(enemies[0]["pos"][:2], (x, y)) < chase:
            target = enemies[0]
        elif npcs and (hit_by_npc or (math.dist(npcs[0]["pos"][:2], (x, y)) < 70 and self.rng.random() < self.aggr)):
            target = npcs[0]
        crate = None
        if target is None and obs["crates"]:
            c = min(obs["crates"], key=lambda c: math.dist(c["pos"][:2], (x, y)))
            if math.dist(c["pos"][:2], (x, y)) < 60:
                crate = c

        # ---- loot second weapon / cubes
        if weapons[1] is None or weapons[0] is None:
            best = None
            for p in obs["pickups"]:
                if p["kind"] in kinds:
                    continue
                d = math.dist(p["pos"][:2], (x, y))
                score = d - (40 if p["kind"] == self.want else 0)
                if d < 120 and (best is None or score < best[0]):
                    best = (score, p)
            if best is not None:
                p = best[1]
                if math.dist(p["pos"][:2], (x, y)) < 3.5 and abs(p["pos"][2] - z) < 4:
                    a["take"] = p["id"]
                elif target is None:
                    move = self._goto(x, y, p["pos"][0], p["pos"][1])
        if move is None and target is None and obs["cubes"]:
            c = min(obs["cubes"], key=lambda c: math.dist(c["pos"][:2], (x, y)))
            if math.dist(c["pos"][:2], (x, y)) < 80 and abs(c["pos"][2] - z) < 4:
                move = self._goto(x, y, c["pos"][0], c["pos"][1])

        # ---- aim and fire
        aim = None
        if target is not None or crate is not None:
            tp = target["pos"] if target is not None else crate["pos"]
            tv = target["vel"] if target is not None else (0, 0, 0)
            th = 4.5 if (target is not None and target["kind"] == "mech") else 1.5
            d = math.dist(tp[:2], (x, y))
            centre = (tp[0], tp[1], tp[2] + th)
            slot_l = kinds.index("laser") if "laser" in kinds else None
            slot_z = kinds.index("lightning") if "lightning" in kinds else None
            slot_m = kinds.index("mortar") if "mortar" in kinds else None
            slot_g = kinds.index("gun") if "gun" in kinds else None
            use = None
            if slot_z is not None and d < 42 and me["energy"] >= 3:
                use = slot_z
            elif self.use_laser and slot_l is not None and d < 270 and (weapons[slot_l]["cooldown"] <= 0 or weapons[slot_l].get("charge", 0) > 0):
                use = slot_l
            elif slot_m is not None and 30 < d < 220 and weapons[slot_m]["cooldown"] <= 0 and target is not None:
                use = slot_m
            elif slot_g is not None:
                use = slot_g
            if use is not None:
                kind = kinds[use]
                st = weapons[use]["stats"]
                if kind == "gun":
                    p = lead(eye, centre, tv, st["speed"], self.lead_frac)
                    aim = list(yaw_pitch(eye, p))
                elif kind == "mortar":
                    yaw, _ = yaw_pitch(eye, centre)
                    pit = mortar_pitch(d, tp[2] - (z + 9.5), st["speed"], st["gravity"])
                    aim = [yaw, pit if pit is not None else 0.6]
                else:
                    aim = list(yaw_pitch(eye, centre))
                err = abs(wrap(aim[0] - me["aim"][0])) + abs(aim[1] - me["aim"][1])
                tol = max(0.012, math.atan2(2.0, max(d, 1.0)))
                if kind == "laser":
                    ch = weapons[use].get("charge", 0)
                    full = ch >= st["charge"] * 0.97
                    fire[use] = not (full and err < tol)
                elif kind == "lightning":
                    fire[use] = abs(wrap(aim[0] - me["aim"][0])) < math.radians(st["cone_deg"]) / 2
                elif kind == "mortar":
                    fire[use] = err < 0.05
                else:
                    fire[use] = err < tol * 1.5
                if kind == "gun" and weapons[use].get("ammo", 1) == 0:
                    fire[use] = False
            if crate is not None and move is None:
                d = math.dist(crate["pos"][:2], (x, y))
                if d > 25:
                    move = self._goto(x, y, crate["pos"][0], crate["pos"][1])
            if target is not None and target["kind"] == "mech":
                d = math.dist(tp[:2], (x, y))
                to = math.atan2(tp[1] - y, tp[0] - x)
                if t > self.strafe_t:
                    self.strafe = -self.strafe
                    self.strafe_t = t + self.rng.uniform(0.8, 2.2)
                side = (math.cos(to + self.strafe * math.pi / 2), math.sin(to + self.strafe * math.pi / 2))
                radial = 0.0
                if d < self.pref * 0.7:
                    radial = -1.0
                elif d > self.pref * 1.25:
                    radial = 0.8 * self.aggr
                mvx = side[0] * 0.8 + math.cos(to) * radial
                mvy = side[1] * 0.8 + math.sin(to) * radial
                L = math.hypot(mvx, mvy) or 1.0
                move = (mvx / L, mvy / L)
                boost = me["energy"] > 50 and self.rng.random() < 0.3

        # ---- zone and wandering
        dn = math.dist((x, y), nc)
        late = t > zone["shrink_start"] - 30
        if (not me["in_zone"]) or (dn > nr * 0.8 and late) or (move is None and dn > nr * 0.6):
            f = zone_field(self.nav, nc[0], nc[1], nr)
            d = f.steer(x, y)
            if d is not None and (move is None or not me["in_zone"] or dn > nr * 0.9):
                move = d
            boost = boost or not me["in_zone"]
        if move is None:
            if self.hunt is not None and t - self.hunt[2] < 20 and math.dist(self.hunt[:2], (x, y)) > 10 \
                    and math.dist(self.hunt[:2], nc) < nr:
                move = self._goto(x, y, self.hunt[0], self.hunt[1])
            else:
                self.hunt = None
                if self.wander is None or math.dist(self.wander, (x, y)) < 8 or self.rng.random() < 0.005:
                    ang, r = self.rng.uniform(0, 2 * math.pi), nr * 0.6 * math.sqrt(self.rng.random())
                    self.wander = (nc[0] + r * math.cos(ang), nc[1] + r * math.sin(ang))
                move = self._goto(x, y, *self.wander)

        # ---- unstick
        if self.last is not None and math.hypot(x - self.last[0], y - self.last[1]) < 0.3 and move is not None:
            self.stuck += 0.1
        else:
            self.stuck = 0.0
        self.last = (x, y)
        if self.stuck > 1.0:
            ang = self.rng.uniform(0, 2 * math.pi)
            self.unstick_dir = (math.cos(ang), math.sin(ang))
            self.unstick_until = t + 1.0
            self.stuck = 0.0
        jump = False
        if t < self.unstick_until:
            move = self.unstick_dir
            jump = me["grounded"]
            boost = me["energy"] > 20

        if aim is None:
            if self.look is not None and t - self.look[1] < 3.0:
                aim = [self.look[0], 0.0]
            elif move is not None:
                aim = [math.atan2(move[1], move[0]), 0.0]
            else:
                aim = list(me["aim"])
        a.update(move=list(move) if move else [0.0, 0.0], boost=boost, jump=jump, aim=aim, fire=fire)
        return a


class ExampleBot:
    """The kit's example_bot.py."""
    name = "example-py"

    def __init__(self, hello: dict):
        self.reach = hello["config"]["loot"]["pickup_radius"]

    def act(self, obs: dict) -> dict:
        me = obs["self"]
        x, y, z = me["pos"]
        goal = obs["zone"]["next_center"]
        act: dict = {"tick": obs["tick"]}
        if obs["offer"]:
            act["pick"] = 0
        near = [p for p in obs["pickups"] if math.dist(p["pos"][:2], (x, y)) < self.reach]
        if me["weapons"][1] is None and near and near[0]["kind"] != "gun":
            act["take"] = near[0]["id"]
        enemies = sorted((m for m in obs["mechs"] if m["kind"] == "mech"), key=lambda m: math.dist(m["pos"], me["pos"]))
        yaw = math.atan2(goal[1] - y, goal[0] - x)
        act.update(move=[math.cos(yaw), math.sin(yaw)], aim=[yaw, 0.0], boost=not me["in_zone"], jump=obs["tick"] % 40 == 0)
        if enemies:
            t = enemies[0]["pos"]
            aim = [math.atan2(t[1] - y, t[0] - x), math.atan2(t[2] - z - 2.5, math.dist(t[:2], (x, y)))]
            act.update(aim=aim, fire=[True, True])
        return act
