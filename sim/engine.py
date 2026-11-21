"""Local mechfight/1 match simulator, written from protocol.md.

It is not the official server: it approximates the rules closely enough to compare bot
policies offline. Every number comes from a hello.config (the recorded one in the kit by
default). Bots are in-process objects with act(obs) -> action, built from a hello dict, so
the real bot class runs unchanged.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
import time
import traceback
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from brain.nav import NavGrid
from brain.world import World

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, "mechfight-kit")
ARENA = os.path.join(KIT, "arena")
MAP_FILES = ["v2/world.json", "v2/terrain.i16"]

_MAP_CACHE: Dict[str, Any] = {}


def load_config() -> dict:
    with gzip.open(os.path.join(KIT, "check", "stream.jsonl.gz"), "rt") as fh:
        return json.loads(fh.readline())["config"]


def get_map(cfg: dict) -> Tuple[World, NavGrid]:
    if "world" not in _MAP_CACHE:
        w = World(ARENA, MAP_FILES)
        _MAP_CACHE["world"] = w
        _MAP_CACHE["nav"] = NavGrid(w, cfg["mech"])
    return _MAP_CACHE["world"], _MAP_CACHE["nav"]


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def dir3(yaw: float, pitch: float) -> np.ndarray:
    cp = math.cos(pitch)
    return np.array([cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)])


# ---------------------------------------------------------------------- entities
class Weapon:
    __slots__ = ("kind", "cooldown", "ammo", "reload", "charge", "charging", "full_t", "pending", "equip")

    def __init__(self, kind: str, stats: dict):
        self.kind = kind
        self.cooldown = 0.0
        self.ammo = int(stats.get("mag", 0)) if kind == "gun" else 0
        self.reload = 0.0
        self.charge = 0.0
        self.charging = False
        self.full_t = 0.0
        self.pending = False
        self.equip = 0.0


class Mech:
    def __init__(self, mid: int, name: str, pos, yaw: float, cfg: dict):
        self.id = mid
        self.name = name
        self.cfg = cfg
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(3)
        self.aim = [yaw, 0.0]
        self.cmd_aim = [yaw, 0.0]
        self.cubes = 0
        self.upg: Dict[str, int] = {}
        self.upg_list: List[str] = []
        self.recompute(heal=False)
        self.hp = self.max_hp
        self.energy = self.max_energy
        self.grounded = True
        self.weapons: List[Optional[Weapon]] = [Weapon("gun", self.ws["gun"]), None]
        self.alive = True
        self.place: Optional[int] = None
        self.death_t: Optional[float] = None
        self.death_cause: Optional[str] = None
        self.kills = 0
        self.shots: Dict[str, int] = {}
        self.dealt: Dict[str, float] = {}
        self.last_combat = -1e9
        self.last_spend = -1e9
        self.move = [0.0, 0.0]
        self.boost = False
        self.fire = [False, False]
        self.oneshot: Dict[str, Any] = {}
        self.miss_t = 0.0
        self.events: List[dict] = []
        self.sounds: List[dict] = []
        self.beams: List[dict] = []
        self.memory: Dict[int, float] = {}
        self.vis_items: set = set()
        self.offers: List[int] = []
        self.offer: Optional[dict] = None
        self.step_acc = 0.0
        self.jet_acc = 0.0
        self.short_until = -1e9
        self.in_zone = True
        self.dmg_taken: Counter = Counter()
        self.act_ms: List[float] = []
        self.errors = 0
        self.late = 0

    def recompute(self, heal: bool = True) -> None:
        cfg = self.cfg
        m = dict(cfg["mech"])
        ws = {k: dict(v) for k, v in cfg["weapons"].items()}
        upgs = {u["id"]: u for u in cfg["upgrades"]}
        for uid, n in self.upg.items():
            for _ in range(n):
                for e in upgs[uid]["effects"]:
                    parts = e["path"].split(".")
                    tgt, key = (m, parts[1]) if parts[0] == "mech" else (ws[parts[1]], parts[2])
                    if "mul" in e:
                        tgt[key] = tgt.get(key, 0.0) * e["mul"]
                    if "add" in e:
                        tgt[key] = tgt.get(key, 0.0) + e["add"]
        cc = cfg["cubes"]
        c = min(self.cubes, cc["stat_cap"])
        old_max = getattr(self, "max_hp", None)
        old_en = getattr(self, "max_energy", None)
        self.m, self.ws = m, ws
        self.max_hp = m["hp"] * (1 + cc["hp_per_cube"] * c)
        self.dmg_mult = 1 + cc["damage_per_cube"] * c
        s = 1 + cc["size_per_cube"] * self.cubes
        self.r, self.h = m["radius"] * s, m["height"] * s
        self.eye, self.muzzle = m["eye_height"] * s, m["muzzle_height"] * s
        self.max_energy = m["energy"]
        if heal and old_max is not None and self.max_hp > old_max:
            self.hp += self.max_hp - old_max
        if heal and old_en is not None and self.max_energy > old_en:
            self.energy += self.max_energy - old_en

    def holds(self, kind: str) -> bool:
        return any(w is not None and w.kind == kind for w in self.weapons)

    def charging(self) -> bool:
        return any(w is not None and w.kind == "laser" and w.charging for w in self.weapons)


class NPC:
    def __init__(self, nid: int, home, cfg: dict):
        self.id = nid
        self.home = np.array(home, dtype=float)
        self.pos = np.array(home, dtype=float)
        self.vel = np.zeros(3)
        self.aim = [0.0, 0.0]
        n = cfg["npc"]
        self.hp = float(n["hp"])
        self.max_hp = float(n["hp"])
        self.r, self.h = float(n["radius"]), float(n["height"])
        self.target: Optional[int] = None
        self.cooldown = 0.0
        self.goal = np.array(home, dtype=float)
        self.alive = True


class Proj:
    __slots__ = ("id", "kind", "owner", "pos", "vel", "ttl", "dmg_mult", "origin")

    def __init__(self, pid, kind, owner, pos, vel, ttl, dmg_mult):
        self.id, self.kind, self.owner = pid, kind, owner
        self.pos, self.vel, self.ttl, self.dmg_mult = pos, vel, ttl, dmg_mult
        self.origin = pos.copy()


# ---------------------------------------------------------------------- zone
class Zone:
    def __init__(self, zc: dict, rng: random.Random, world: World):
        c = tuple(zc["start_center"])
        r = float(zc["start_radius"])
        self.circles = [(c, r)]
        self.times = []
        T = 0.0
        x0, y0, x1, y1 = world.bounds
        for ph in zc["phases"]:
            rn = float(ph["radius"])
            room = max(0.0, r - rn)
            for _ in range(50):
                a, d = rng.uniform(0, 2 * math.pi), room * math.sqrt(rng.random())
                cn = (c[0] + d * math.cos(a), c[1] + d * math.sin(a))
                if x0 + rn + 20 <= cn[0] <= x1 - rn - 20 and y0 + rn + 20 <= cn[1] <= y1 - rn - 20:
                    break
            self.circles.append((cn, rn))
            hs = T + ph["hold_s"]
            se = hs + ph["shrink_s"]
            self.times.append((T, hs, se, float(ph["dps"])))
            T = se
            c, r = cn, rn
        self.end = T

    def state(self, t: float):
        for k, (t0, ss, se, dps) in enumerate(self.times):
            if t < se or k == len(self.times) - 1:
                (c0, r0), (c1, r1) = self.circles[k], self.circles[k + 1]
                if t <= ss:
                    c, r = c0, r0
                elif t < se:
                    f = (t - ss) / (se - ss)
                    c = (c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f)
                    r = r0 + (r1 - r0) * f
                else:
                    c, r = c1, r1
                return c, r, c1, r1, ss, se, dps
        raise AssertionError


# ---------------------------------------------------------------------- match
BotFactory = Callable[[dict], Any]


class Match:
    def __init__(self, bots: List[Tuple[str, BotFactory]], seed: int, cfg: Optional[dict] = None,
                 timed: Tuple[int, ...] = (), enforce_budget: bool = True, verbose: bool = False,
                 squad_prey: Optional[int] = None):
        self.cfg = cfg or load_config()
        self.rng = random.Random(seed)
        self.nrng = np.random.default_rng(seed)
        self.world, self.nav = get_map(self.cfg)
        self.t = 0.0
        self.tick = 0
        self.dt = 1.0 / self.cfg["match"]["tick_hz"]
        self.decide_every = int(self.cfg["match"]["decide_every"])
        self.zone = Zone(self.cfg["zone"], self.rng, self.world)
        self.timed = set(timed)
        self.enforce_budget = enforce_budget
        self.verbose = verbose
        self.squad_prey = squad_prey
        self.squad_seen: Optional[Tuple[float, dict]] = None
        self.next_id = {"proj": 1, "crate": 1000, "cube": 5000, "pickup": 8000, "drop": 20000}
        self.projs: List[Proj] = []
        self.crates: Dict[int, dict] = {}
        self.cubes: Dict[int, np.ndarray] = {}
        self.pickups: Dict[int, dict] = {}
        self.drops: Dict[int, dict] = {}
        self.drop_calls = list(self.cfg.get("drops", {}).get("times", []))
        self.sound_q: List[Tuple[str, np.ndarray, float, int]] = []
        self.beam_q: List[dict] = []
        self.npcs: Dict[int, NPC] = {}
        self.mechs: List[Mech] = []
        self.bots: List[Any] = []
        self.offer_seq = 0
        self._spawn_all(bots)

    # ------------------------------------------------------------------ setup
    def _nid(self, kind: str) -> int:
        n = self.next_id[kind]
        self.next_id[kind] += 1
        return n

    def _open_point(self, center=None, radius=None, margin=30.0, tries=4000):
        x0, y0, x1, y1 = self.world.bounds
        for _ in range(tries):
            if center is not None:
                a, d = self.rng.uniform(0, 2 * math.pi), radius * math.sqrt(self.rng.random())
                x, y = center[0] + d * math.cos(a), center[1] + d * math.sin(a)
            else:
                x, y = self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)
            if not self.world.in_bounds(x, y, margin) or not self.nav.walkable_at(x, y):
                continue
            z = self.world.ground_at(x, y)
            i, j = self.world.ij(x, y)
            if self.world.b_lo[j, i] < z + 10:
                continue
            return np.array([x, y, z])
        return None

    def _roof_point(self, tries=4000):
        w = self.world
        x0, y0, x1, y1 = w.bounds
        for _ in range(tries):
            x, y = self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)
            i, j = w.ij(x, y)
            top = float(w.s_hi[j, i])
            if top < w.ground[j, i] + 3 or top > 60:
                continue
            win = w.s_hi[max(0, j - 3): j + 4, max(0, i - 3): i + 4]
            if win.shape != (7, 7) or np.abs(win - top).max() > 0.3:
                continue
            return np.array([x, y, top])
        return None

    def _spawn_all(self, bots) -> None:
        cfg = self.cfg
        mc = cfg["match"]
        zc = cfg["zone"]
        sep = float(mc["spawn_min_separation"])
        rad = float(zc["start_radius"]) * float(mc["spawn_zone_fraction"])
        spots: List[np.ndarray] = []
        while len(spots) < len(bots):
            p = None
            for _ in range(3000):
                q = self._open_point(zc["start_center"], rad, mc["arena_margin"])
                if q is not None and all(np.hypot(*(q[:2] - s[:2])) >= sep for s in spots):
                    p = q
                    break
            if p is None:
                sep *= 0.9
                continue
            spots.append(p)
        for slot, ((name, factory), p) in enumerate(zip(bots, spots)):
            m = Mech(slot, name, p, self.rng.uniform(-math.pi, math.pi), cfg)
            self.mechs.append(m)
        lw = cfg["loot"]["weights"]
        kinds = [k for k in ("gun", "mortar", "laser", "lightning") if lw.get(k, 0) > 0]
        weights = [lw[k] for k in kinds]
        for _ in range(int(cfg["loot"]["floor_count"])):
            roof = self.rng.random() < cfg["loot"]["floor_roof_fraction"]
            p = self._roof_point() if roof else None
            p = p if p is not None else self._open_point(margin=20)
            if p is not None:
                self._add_pickup(self.rng.choices(kinds, weights)[0], p)
        for _ in range(int(cfg["crates"]["count"])):
            roof = self.rng.random() < cfg["crates"]["roof_fraction"]
            p = self._roof_point() if roof else None
            p = p if p is not None else self._open_point(margin=20)
            if p is not None:
                self.crates[self._nid("crate")] = {"pos": p, "hp": float(cfg["crates"]["hp"])}
        for n in range(int(cfg["npc"]["count"])):
            p = self._open_point(margin=40)
            if p is not None:
                self.npcs[100 + n] = NPC(100 + n, p, cfg)
        roles = ["easy"] * 4 + ["medium"] * 4 + ["hard"] * 4 + ["expert"] * 3
        house_i = 0
        self.role_rank: Dict[int, int] = {}
        for slot, (name, factory) in enumerate(bots):
            hello = {"type": "hello", "protocol": "mechfight/1", "slot": slot,
                     "seed": self.rng.randrange(2 ** 32),
                     "map": {"dir": ARENA, "files": list(MAP_FILES)}, "config": cfg}
            if self.squad_prey is not None and slot != self.squad_prey:
                hello["squad"] = True
                hello["prey"] = self.squad_prey
                hello["role"] = roles[min(house_i, len(roles) - 1)]
                self.role_rank[slot] = house_i
                house_i += 1
            t0 = time.perf_counter()
            bot = factory(hello)
            if slot in self.timed and self.verbose:
                print(f"[sim] slot {slot} hello took {time.perf_counter() - t0:.2f}s")
            self.bots.append(bot)

    def _add_pickup(self, kind: str, pos, weapon: Optional[Weapon] = None) -> int:
        pid = self._nid("pickup")
        self.pickups[pid] = {"kind": kind, "pos": np.array(pos, dtype=float), "w": weapon}
        return pid

    def _add_cube(self, pos) -> None:
        self.cubes[self._nid("cube")] = np.array(pos, dtype=float)

    # ------------------------------------------------------------------ helpers
    def alive_mechs(self) -> List[Mech]:
        return [m for m in self.mechs if m.alive]

    def sound(self, kind: str, pos, src: int) -> None:
        self.sound_q.append((kind, np.array(pos, dtype=float), self.t, src))

    def spend(self, m: Mech, amount: float) -> None:
        m.energy = max(0.0, m.energy - amount)
        m.last_spend = self.t

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        dur = float(self.cfg["match"]["duration_s"])
        while True:
            alive = self.alive_mechs()
            if len(alive) <= 1 or self.t >= dur - 1e-9:
                break
            if self.squad_prey is not None and not self.mechs[self.squad_prey].alive:
                break
            if self.tick % self.decide_every == 0:
                self.decide()
            self.step()
        alive = self.alive_mechs()
        alive.sort(key=lambda m: -m.hp)
        for n, m in enumerate(alive):
            m.place = n + 1
        return self.results()

    def results(self) -> dict:
        out = []
        for m in self.mechs:
            out.append({"slot": m.id, "name": m.name, "place": m.place, "kills": m.kills,
                        "shots": dict(m.shots), "dealt": dict(m.dealt),
                        "death_t": m.death_t, "cause": m.death_cause, "cubes": m.cubes,
                        "dmg_taken": dict(m.dmg_taken), "upgrades": list(m.upg_list),
                        "act_ms_max": max(m.act_ms) if m.act_ms else 0.0,
                        "act_ms_p99": float(np.percentile(m.act_ms, 99)) if m.act_ms else 0.0,
                        "act_ms_mean": float(np.mean(m.act_ms)) if m.act_ms else 0.0,
                        "late": m.late, "errors": m.errors,
                        "weapons": [w.kind if w else None for w in m.weapons]})
        return {"t": self.t, "mechs": out}

    # ------------------------------------------------------------------ decisions
    def _hunter_count(self, alive_n: int, n_house: int) -> int:
        """Practice squad: one hunter early, more as the field thins, every remaining bot late."""
        if alive_n <= 6 or self.t >= 300:
            return n_house
        if alive_n <= 9 or self.t >= 200:
            return min(n_house, 6)
        if alive_n <= 13 or self.t >= 90:
            return min(n_house, 3)
        return 1

    def _share_squad(self, obs: Dict[int, dict], alive: List[Mech]) -> None:
        """House bots share sight of the player, ignore each other, and only the active hunters
        are told where the player is."""
        prey = self.mechs[self.squad_prey]
        house = [m for m in alive if m.id != self.squad_prey]
        if not house or not prey.alive:
            return
        seen = False
        for m in house:
            if any(e.get("id") == prey.id for e in obs[m.id]["mechs"]):
                seen = True
                break
        if seen:
            snap = {"id": prey.id, "kind": "mech",
                    "pos": [round(float(v), 2) for v in prey.pos],
                    "vel": [round(float(v), 2) for v in prey.vel],
                    "aim": [round(prey.aim[0], 3), round(prey.aim[1], 3)],
                    "hp": round(prey.hp, 2), "max_hp": round(prey.max_hp, 2), "cubes": prey.cubes,
                    "weapons": [w.kind for w in prey.weapons if w is not None],
                    "charging": prey.charging()}
            self.squad_seen = (self.t, snap)
        remembered = self.squad_seen is not None and self.t - self.squad_seen[0] <= 8.0
        n = self._hunter_count(len(alive), len(house))
        # rules: the squad hunts "weakest first" (easy, then medium, hard, expert)
        ranked = sorted(house, key=lambda m: (self.role_rank.get(m.id, 99),
                                              float(np.hypot(*(m.pos[:2] - prey.pos[:2])))))
        hunters = {m.id for m in ranked[:n]}
        for m in house:
            kept = [e for e in obs[m.id]["mechs"] if e.get("kind") != "mech"]
            if m.id in hunters and remembered:
                kept.append(self.squad_seen[1])
            obs[m.id]["mechs"] = kept

    def decide(self) -> None:
        alive = self.alive_mechs()
        obs = self.build_obs(alive)
        if self.squad_prey is not None:
            self._share_squad(obs, alive)
        budget = self.cfg["bot"]["tick_budget_ms"]
        coast = self.cfg["bot"]["coast_s"]
        for m in alive:
            o = obs[m.id]
            t0 = time.perf_counter()
            try:
                act = self.bots[m.id].act(o) or {}
            except Exception:
                m.errors += 1
                if m.errors <= 3:
                    traceback.print_exc()
                act = {}
            ms = (time.perf_counter() - t0) * 1000.0
            if m.id in self.timed:
                m.act_ms.append(ms)
            late = self.enforce_budget and m.id in self.timed and ms > budget
            if late:
                m.late += 1
                m.miss_t += self.dt * self.decide_every
                m.fire = [False, False]
                m.oneshot = {}
                if m.miss_t > coast:
                    m.move, m.boost = [0.0, 0.0], False
                continue
            m.miss_t = 0.0
            self.apply_action(m, act)

    def apply_action(self, m: Mech, a: dict) -> None:
        mv = a.get("move", [0.0, 0.0]) or [0.0, 0.0]
        mx, my = float(mv[0]), float(mv[1])
        L = math.hypot(mx, my)
        if L > 1:
            mx, my = mx / L, my / L
        m.move = [mx, my]
        m.boost = bool(a.get("boost", False))
        if "aim" in a and a["aim"] is not None:
            ms = m.m
            m.cmd_aim = [wrap(float(a["aim"][0])), min(max(float(a["aim"][1]), ms["min_pitch"]), ms["max_pitch"])]
        f = a.get("fire") or [False, False]
        m.fire = [bool(f[0]) if len(f) > 0 else False, bool(f[1]) if len(f) > 1 else False]
        m.oneshot = {k: a[k] for k in ("jump", "reload", "pick", "take", "drop") if k in a and a[k] is not None}
        if "pick" in m.oneshot and m.offer is not None:
            p = int(m.oneshot["pick"])
            if 0 <= p < len(m.offer["choices"]):
                self.apply_upgrade(m, m.offer["choices"][p]["id"])
        if "take" in m.oneshot or "drop" in m.oneshot:
            self.do_take(m, m.oneshot.get("take"), m.oneshot.get("drop"))

    def do_take(self, m: Mech, take, drop) -> None:
        drop = int(drop) if drop is not None and int(drop) in (0, 1) else None
        if take is None:
            if drop is not None and m.weapons[drop] is not None:
                w = m.weapons[drop]
                if w.kind == "laser" and w.charging:
                    return
                pid = self._add_pickup(w.kind, m.pos, w)
                m.weapons[drop] = None
                m.events.append({"type": "dropped", "id": pid, "kind": w.kind, "slot": drop})
            return
        take = int(take)
        pk = self.pickups.get(take)
        if pk is None:
            m.events.append({"type": "take_failed", "id": take, "reason": "gone"})
            return
        d = pk["pos"] - m.pos
        if math.hypot(d[0], d[1]) > self.cfg["loot"]["pickup_radius"] or abs(d[2]) > 4.0:
            m.events.append({"type": "take_failed", "id": take, "reason": "range"})
            return
        if any(w is not None and w.kind == pk["kind"] and (drop is None or m.weapons[drop] is not w)
               for w in m.weapons):
            m.events.append({"type": "take_failed", "id": take, "reason": "have"})
            return
        slot = drop if drop is not None else (0 if m.weapons[0] is None else 1 if m.weapons[1] is None else None)
        if slot is None:
            m.events.append({"type": "take_failed", "id": take, "reason": "full"})
            return
        old = m.weapons[slot]
        if old is not None and old.kind == "laser" and old.charging:
            m.events.append({"type": "take_failed", "id": take, "reason": "busy"})
            return
        del self.pickups[take]
        if old is not None:
            pid = self._add_pickup(old.kind, m.pos, old)
            m.events.append({"type": "dropped", "id": pid, "kind": old.kind, "slot": slot})
        w = pk["w"] or Weapon(pk["kind"], m.ws[pk["kind"]])
        w.charging, w.charge, w.pending = False, 0.0, False
        w.equip = max(self.cfg["loot"]["equip_s"], m.ws[w.kind].get("cooldown", 0.0))
        m.weapons[slot] = w
        m.events.append({"type": "took", "id": take, "kind": w.kind, "slot": slot})

    # ------------------------------------------------------------------ upgrades
    def apply_upgrade(self, m: Mech, uid: str) -> None:
        m.upg[uid] = m.upg.get(uid, 0) + 1
        m.upg_list.append(uid)
        m.recompute()
        m.events.append({"type": "upgrade", "id": uid})
        m.offer = None
        self.next_offer(m)

    def next_offer(self, m: Mech) -> None:
        if m.offer is not None or not m.offers:
            return
        m.offers.pop(0)
        elig = []
        for u in self.cfg["upgrades"]:
            if m.upg.get(u["id"], 0) >= u["max"]:
                continue
            if u.get("requires") and not m.holds(u["requires"]):
                continue
            elig.append(u)
        if not elig:
            self.next_offer(m)
            return
        ch = self.rng.sample(elig, min(3, len(elig)))
        self.offer_seq += 1
        m.offer = {"id": self.offer_seq, "expires": self.t + self.cfg["bot"]["offer_timeout_s"],
                   "choices": [{"id": u["id"], "name": u["name"], "desc": u["desc"]} for u in ch]}

    def gain_cubes(self, m: Mech, n: int) -> None:
        th = self.cfg["cubes"]["upgrade_thresholds"]
        for _ in range(n):
            m.cubes += 1
            if m.cubes in th:
                m.offers.append(m.cubes)
        m.recompute()
        self.next_offer(m)

    # ------------------------------------------------------------------ physics
    def step(self) -> None:
        dt = self.dt
        zc, zr, _, _, _, _, dps = self.zone.state(self.t)
        for m in self.alive_mechs():
            self.step_mech(m, dt)
        self.step_npcs(dt)
        self.step_projs(dt)
        self.step_world(dt, zc, zr, dps)
        self.t = round(self.t + dt, 6)
        self.tick += 1

    def step_mech(self, m: Mech, dt: float) -> None:
        ms = m.m
        w = self.world
        yaw, pitch = m.aim
        dy = wrap(m.cmd_aim[0] - yaw)
        lim = ms["turn_rate"] * dt
        yaw = wrap(yaw + max(-lim, min(lim, dy)))
        dp = m.cmd_aim[1] - pitch
        plim = ms["pitch_rate"] * dt
        pitch = pitch + max(-plim, min(plim, dp))
        m.aim = [yaw, pitch]
        for slot in (0, 1):
            self.weapon_tick(m, slot, dt)
        if not m.alive:
            return
        if m.oneshot.pop("reload", False):
            for wp in m.weapons:
                if wp is not None and wp.kind == "gun" and wp.reload <= 0 and wp.ammo < m.ws["gun"]["mag"]:
                    wp.reload = m.ws["gun"]["reload"]
                    self.sound("reload", m.pos, m.id)
        mx, my = m.move
        mag = math.hypot(mx, my)
        boosting = m.boost and m.energy > 0
        speed = ms["walk_speed"]
        x, y, z = m.pos
        vx, vy, vz = m.vel
        if m.grounded and boosting and mag > 0.05:
            speed = ms["boost_speed"]
            self.spend(m, ms["boost_drain"] * dt)
        if m.charging():
            speed *= ms["charge_move_mult"]
        tvx, tvy = mx * speed, my * speed
        acc = (ms["ground_accel"] if m.grounded else ms["air_accel"]) * dt
        ddx, ddy = tvx - vx, tvy - vy
        L = math.hypot(ddx, ddy)
        if L > acc:
            ddx, ddy = ddx * acc / L, ddy * acc / L
        vx, vy = vx + ddx, vy + ddy
        if m.oneshot.pop("jump", False) and m.grounded:
            vz = ms["jump_speed"]
            m.grounded = False
        jetting = False
        if not m.grounded:
            vz -= ms["gravity"] * dt
            if boosting:
                jetting = True
                if vz < ms["max_rise_speed"]:
                    vz = min(vz + ms["jet_accel"] * dt, ms["max_rise_speed"])
                self.spend(m, ms["jet_drain"] * dt)
        # horizontal move with wall sliding
        nx_, ny_ = x + vx * dt, y + vy * dt
        x0, y0, x1, y1 = w.bounds
        if not (x0 + m.r <= nx_ <= x1 - m.r):
            nx_, vx = min(max(nx_, x0 + m.r), x1 - m.r), 0.0
        if not (y0 + m.r <= ny_ <= y1 - m.r):
            ny_, vy = min(max(ny_, y0 + m.r), y1 - m.r), 0.0
        su = ms["step_up"]
        blocked, nwx, nwy = w.wall_contact(nx_, ny_, z, m.r, m.h, su)
        if blocked:
            vn = vx * nwx + vy * nwy
            if vn < 0:
                vx, vy = vx - vn * nwx, vy - vn * nwy
            nx_, ny_ = x + vx * dt, y + vy * dt
            if w.wall_contact(nx_, ny_, z, m.r, m.h, su)[0]:
                if not w.wall_contact(x + vx * dt, y, z, m.r, m.h, su)[0]:
                    nx_, ny_, vy = x + vx * dt, y, 0.0
                elif not w.wall_contact(x, y + vy * dt, z, m.r, m.h, su)[0]:
                    nx_, ny_, vx = x, y + vy * dt, 0.0
                else:
                    nx_, ny_, vx, vy = x, y, 0.0, 0.0
        if w.ground_at(nx_, ny_) > z + su:
            nx_, ny_, vx, vy = x, y, 0.0, 0.0
        x, y = nx_, ny_
        sfc = w.surface(x, y, z, su)
        if m.grounded:
            if sfc >= z - 0.35:
                z, vz = sfc, 0.0
            else:
                m.grounded = False
        if not m.grounded:
            zn = z + vz * dt
            if vz > 0 and w.ceiling_hit(x, y, z, zn, m.r, m.h):
                vz, zn = 0.0, z
            sfc = w.surface(x, y, max(z, zn), su)
            if zn <= sfc:
                z, vz, m.grounded = sfc, 0.0, True
            else:
                if vz < 0:
                    top, ex, ey = w.edge_support(x, y, z, zn, m.r)
                    if top is not None and top > sfc:
                        zn, vz = top, 0.0
                        sx, sy = x + ex * 0.6, y + ey * 0.6
                        if not w.wall_contact(sx, sy, zn, m.r, m.h, su)[0]:
                            x, y = sx, sy
                z = zn
        m.pos = np.array([x, y, z])
        m.vel = np.array([vx, vy, vz])
        spd = math.hypot(vx, vy)
        sc = self.cfg["sound"]
        if m.grounded and spd > sc["step_speed"]:
            m.step_acc += dt
            if m.step_acc >= sc["step_every"]:
                m.step_acc = 0.0
                self.sound("boost" if (boosting and mag > 0.05) else "step", m.pos, m.id)
        else:
            m.step_acc = 0.0
        if jetting:
            m.jet_acc += dt
            if m.jet_acc >= sc["step_every"]:
                m.jet_acc = 0.0
                self.sound("jet", m.pos, m.id)

    # ------------------------------------------------------------------ weapons
    def muzzle(self, m: Mech, height: float):
        d = dir3(*m.aim)
        base = m.pos + np.array([0.0, 0.0, height])
        tip = base + d * (m.r + 0.5)
        f = float(self.world.first_block([base], [tip], step=0.25)[0])
        if f < 1.0:
            tip = base + d * (m.r + 0.5) * max(0.0, f - 0.05)
        return tip, d

    def weapon_tick(self, m: Mech, slot: int, dt: float) -> None:
        wp = m.weapons[slot]
        if wp is None:
            return
        st = m.ws[wp.kind]
        trig = m.fire[slot]
        wp.equip = max(0.0, wp.equip - dt)
        wp.cooldown -= dt
        if not trig and wp.kind != "laser":
            wp.cooldown = max(wp.cooldown, 0.0)
        if wp.kind == "gun":
            if wp.reload > 0:
                wp.reload -= dt
                wp.cooldown = max(wp.cooldown, 0.0)
                if wp.reload <= 0:
                    wp.reload, wp.ammo = 0.0, int(st["mag"])
                return
            if trig and wp.cooldown <= 1e-9 and wp.equip <= 0 and wp.ammo > 0:
                m.shots["gun"] = m.shots.get("gun", 0) + 1
                o, d = self.muzzle(m, m.muzzle)
                sp = math.radians(st["spread_deg"])
                d = dir3(m.aim[0] + self.rng.gauss(0, sp), m.aim[1] + self.rng.gauss(0, sp))
                self.projs.append(Proj(self._nid("proj"), "gun", m.id, o, d * st["speed"],
                                       st["range"] / st["speed"], m.dmg_mult))
                self.sound("gun", m.pos, m.id)
                wp.ammo -= 1
                wp.cooldown += st["cooldown"]
                if wp.ammo <= 0:
                    wp.reload = st["reload"]
                    self.sound("reload", m.pos, m.id)
            else:
                wp.cooldown = max(wp.cooldown, 0.0)
        elif wp.kind == "mortar":
            if trig and wp.cooldown <= 1e-9 and wp.equip <= 0:
                o, d = self.muzzle(m, m.h + 0.5)
                self.projs.append(Proj(self._nid("proj"), "mortar", m.id, o, d * st["speed"], st["ttl"], m.dmg_mult))
                self.sound("mortar", m.pos, m.id)
                wp.cooldown = st["cooldown"]
            wp.cooldown = max(wp.cooldown, 0.0)
        elif wp.kind == "lightning":
            if trig and wp.cooldown <= 1e-9 and wp.equip <= 0 and m.energy >= st["energy"]:
                self.spend(m, st["energy"])
                self.fire_lightning(m, st)
                wp.cooldown += st["cooldown"]
            wp.cooldown = max(wp.cooldown, 0.0)
        elif wp.kind == "laser":
            wp.cooldown = max(wp.cooldown, 0.0)
            if self.t < m.short_until:
                wp.charging, wp.charge, wp.pending = False, 0.0, False
                return
            if wp.charging:
                wp.charge = min(st["charge"], wp.charge + dt)
                if wp.charge >= st["charge"] - 1e-9:
                    wp.full_t += dt
                if not trig:
                    wp.pending = True
                if (wp.pending and wp.charge >= st["min_charge"] - 1e-9) or wp.full_t >= st["hold_s"]:
                    frac = st["min_frac"] + (1 - st["min_frac"]) * min(1.0, wp.charge / st["charge"])
                    self.fire_laser(m, st, frac)
                    wp.charging, wp.charge, wp.pending, wp.full_t = False, 0.0, False, 0.0
                    wp.cooldown = st["cooldown"]
            elif trig and wp.cooldown <= 1e-9 and wp.equip <= 0:
                wp.charging, wp.charge, wp.pending, wp.full_t = True, 0.0, False, 0.0
                self.sound("laser_charge", m.pos, m.id)

    def targets(self, exclude: Optional[int] = None):
        """Arrays of every damageable body: (ids, x, y, z, r, h, kinds)."""
        ids, xs, ys, zs, rs, hs, ks = [], [], [], [], [], [], []
        for m in self.mechs:
            if m.alive and m.id != exclude:
                ids.append(m.id); xs.append(m.pos[0]); ys.append(m.pos[1]); zs.append(m.pos[2])
                rs.append(m.r); hs.append(m.h); ks.append(0)
        for n in self.npcs.values():
            if n.alive and n.id != exclude:
                ids.append(n.id); xs.append(n.pos[0]); ys.append(n.pos[1]); zs.append(n.pos[2])
                rs.append(n.r); hs.append(n.h); ks.append(1)
        cr = self.cfg["crates"]
        for cid, c in self.crates.items():
            ids.append(cid); xs.append(c["pos"][0]); ys.append(c["pos"][1]); zs.append(c["pos"][2])
            rs.append(cr["radius"]); hs.append(cr["height"]); ks.append(2)
        return (ids, np.array(xs), np.array(ys), np.array(zs), np.array(rs), np.array(hs), ks)

    @staticmethod
    def seg_hits(P0, P1, T, pr: float):
        """s along each segment where it first meets each target cylinder (inf = miss), shape (n, m)."""
        _, tx, ty, tz, tr, th, _ = T
        d = P1 - P0
        dd = np.maximum(d[:, 0:1] ** 2 + d[:, 1:2] ** 2, 1e-12)
        rx = tx[None, :] - P0[:, 0:1]
        ry = ty[None, :] - P0[:, 1:2]
        s = np.clip((rx * d[:, 0:1] + ry * d[:, 1:2]) / dd, 0.0, 1.0)
        cx = P0[:, 0:1] + s * d[:, 0:1] - tx[None, :]
        cy = P0[:, 1:2] + s * d[:, 1:2] - ty[None, :]
        hit = cx * cx + cy * cy <= (tr[None, :] + pr) ** 2
        zz = P0[:, 2:3] + s * d[:, 2:3]
        hit &= (zz >= tz[None, :] - pr) & (zz <= tz[None, :] + th[None, :] + pr)
        return np.where(hit, s, np.inf)

    def fire_laser(self, m: Mech, st: dict, frac: float) -> None:
        o, d = self.muzzle(m, m.muzzle)
        end = o + d * st["range"]
        fw = float(self.world.first_block([o], [end], step=1.0)[0])
        T = self.targets(exclude=m.id)
        if T[0]:
            S = self.seg_hits(o[None, :], end[None, :], T, 0.3)[0]
            order = [k for k in np.argsort(S) if S[k] < fw]
        else:
            order = []
        cap = 1 + int(st.get("pierce", 0))
        hits = order[:cap]
        for k in hits:
            self.damage(T[0][k], T[6][k], st["damage"] * frac * m.dmg_mult, "laser", m.id, o, effect_ok=True)
        stop = float(S[hits[-1]]) if len(hits) == cap else fw
        tip = o + d * st["range"] * stop
        self.sound("laser", m.pos, m.id)
        self.beam_q.append({"by": m.id, "kind": "laser", "path": [o.tolist(), tip.tolist()], "t": self.t})

    def fire_lightning(self, m: Mech, st: dict) -> None:
        o, d = self.muzzle(m, m.muzzle)
        T = self.targets(exclude=m.id)
        ids, tx, ty, tz, tr, th, ks = T
        path = [o.tolist()]
        if ids:
            cz = tz + th * 0.5
            dist = np.sqrt((tx - o[0]) ** 2 + (ty - o[1]) ** 2 + (cz - o[2]) ** 2)
            yaw = np.arctan2(ty - o[1], tx - o[0])
            dyaw = np.abs((yaw - m.aim[0] + np.pi) % (2 * np.pi) - np.pi)
            cand = np.nonzero((dist <= st["range"] + tr) & (dyaw <= math.radians(st["cone_deg"]) / 2))[0]
            cand = cand[np.argsort(dist[cand])]
            first = None
            if len(cand):
                pts = np.stack([tx[cand], ty[cand], cz[cand]], 1)
                ok = self.world.los_batch(np.repeat(o[None, :], len(cand), 0), pts)
                good = cand[ok]
                first = int(good[0]) if len(good) else None
            if first is not None:
                hit = [first]
                dmg = st["damage"] * m.dmg_mult
                self.damage(ids[first], ks[first], dmg, "lightning", m.id, o, effect_ok=True)
                cur = first
                path.append([float(tx[cur]), float(ty[cur]), float(cz[cur])])
                for _ in range(int(st["chain"])):
                    dc = np.sqrt((tx - tx[cur]) ** 2 + (ty - ty[cur]) ** 2 + (cz - cz[cur]) ** 2)
                    dc[hit] = np.inf
                    nxt = int(np.argmin(dc))
                    if not np.isfinite(dc[nxt]) or dc[nxt] > st["chain_range"]:
                        break
                    a = np.array([tx[cur], ty[cur], cz[cur]])
                    b = np.array([tx[nxt], ty[nxt], cz[nxt]])
                    if not self.world.los(a, b):
                        break
                    dmg *= st["chain_falloff"]
                    self.damage(ids[nxt], ks[nxt], dmg, "lightning", m.id, a, effect_ok=True)
                    hit.append(nxt)
                    cur = nxt
                    path.append(b.tolist())
        if len(path) == 1:
            end = o + d * st["range"]
            f = float(self.world.first_block([o], [end])[0])
            path.append((o + d * st["range"] * f).tolist())
        self.sound("lightning", m.pos, m.id)
        self.beam_q.append({"by": m.id, "kind": "lightning", "path": path, "t": self.t})

    # ------------------------------------------------------------------ damage
    def damage(self, tid: int, kind: int, amount: float, weapon: str, by: Optional[int], src,
               effect_ok: bool = False, effect: Optional[str] = None) -> float:
        src = np.asarray(src, dtype=float)
        if kind == 2:
            c = self.crates.get(tid)
            if c is None:
                return 0.0
            c["hp"] -= amount
            if by is not None and by < 100:
                att = self.mechs[by]
                att.last_combat = self.t
                att.events.append({"type": "hit", "target": tid, "amount": round(amount, 2), "weapon": weapon})
            if c["hp"] <= 0:
                del self.crates[tid]
                for _ in range(int(self.cfg["crates"]["cubes"])):
                    self._add_cube(c["pos"])
                if self.rng.random() < self.cfg["loot"]["crate_weapon_chance"]:
                    lw = self.cfg["loot"]["weights"]
                    ks = [k for k in lw if lw[k] > 0]
                    self._add_pickup(self.rng.choices(ks, [lw[k] for k in ks])[0], c["pos"] + np.array([1.0, 0, 0]))
                self.sound("crate", c["pos"], -1)
            return amount
        if kind == 1:
            n = self.npcs.get(tid)
            if n is None or not n.alive:
                return 0.0
            n.hp -= amount
            if by is not None and by < 100:
                att = self.mechs[by]
                att.last_combat = self.t
                att.events.append({"type": "hit", "target": tid, "amount": round(amount, 2), "weapon": weapon})
                if self.mechs[by].alive:
                    n.target = by
            if n.hp <= 0:
                n.alive = False
                del self.npcs[tid]
                for _ in range(int(self.cfg["npc"]["cubes"])):
                    self._add_cube(n.pos + np.array([self.rng.uniform(-1, 1), self.rng.uniform(-1, 1), 0]))
                self.sound("death", n.pos, -1)
            return amount
        t = self.mechs[tid]
        if not t.alive:
            return 0.0
        rear = False
        bearing = None
        if weapon != "zone":
            dx, dy = src[0] - t.pos[0], src[1] - t.pos[1]
            if math.hypot(dx, dy) > 1e-6:
                bearing = math.atan2(dy, dx)
                rear = abs(wrap(bearing - t.aim[0])) > math.radians(t.m["front_arc_deg"]) / 2
            if rear:
                amount *= t.m["rear_mult"]
            if effect_ok and weapon == "laser" and t.holds("mortar"):
                amount *= 1 + self.cfg["weapons"]["laser"]["cookoff"]
                effect = "cookoff"
            if effect_ok and weapon == "lightning" and t.holds("laser"):
                effect = "short"
                t.short_until = self.t + self.cfg["weapons"]["lightning"]["short_s"]
                for w in t.weapons:
                    if w is not None and w.kind == "laser":
                        w.charging, w.charge, w.pending = False, 0.0, False
            amount *= max(0.0, 1 - min(0.9, t.m.get("armor", 0.0)))
        t.hp -= amount
        t.last_combat = self.t
        t.dmg_taken[weapon] += amount
        if by is not None and by < 100 and by != t.id:
            att_m = self.mechs[by]
            att_m.dealt[weapon] = att_m.dealt.get(weapon, 0.0) + amount
        ev = {"type": "damaged", "amount": round(amount, 2), "weapon": weapon,
              "bearing": bearing, "by": by}
        if rear:
            ev["rear"] = True
        t.events.append(ev)
        if effect is not None:
            t.events.append({"type": "effect", "effect": effect, "by": by})
        if by is not None and by < 100 and by != tid:
            att = self.mechs[by]
            att.last_combat = self.t
            hev = {"type": "hit", "target": tid, "amount": round(amount, 2), "weapon": weapon}
            if effect:
                hev["effect"] = effect
            if rear:
                hev["rear"] = True
            att.events.append(hev)
        if t.hp <= 0:
            self.kill(t, by, weapon)
        return amount

    def kill(self, t: Mech, by: Optional[int], weapon: str) -> None:
        t.alive = False
        t.hp = 0.0
        t.place = len([m for m in self.mechs if m.alive]) + 1
        same_tick = [m for m in self.mechs if not m.alive and m.death_t == self.t and m is not t]
        if same_tick:
            t.place = max(t.place, max(m.place for m in same_tick))
            for m in same_tick:
                m.place = t.place
        t.death_t = self.t
        t.death_cause = weapon
        for _ in range(t.cubes + int(self.cfg["cubes"]["death_drop_extra"])):
            self._add_cube(t.pos + np.array([self.rng.uniform(-2, 2), self.rng.uniform(-2, 2), 0]))
        for w in t.weapons:
            if w is not None and w.kind != "gun":
                w.charging, w.charge = False, 0.0
                self._add_pickup(w.kind, t.pos + np.array([self.rng.uniform(-2, 2), self.rng.uniform(-2, 2), 0]), w)
        self.sound("death", t.pos, t.id)
        if by is not None and by < 100 and by != t.id and self.mechs[by].alive:
            self.mechs[by].kills += 1
            self.mechs[by].events.append({"type": "kill", "target": t.id})

    def explode(self, p: Proj, at: np.ndarray, direct: Optional[Tuple[int, int]]) -> None:
        st = self.mechs[p.owner].ws["mortar"] if p.owner < 100 else self.cfg["weapons"]["mortar"]
        R = st["splash_radius"]
        edge = st.get("splash_edge", 0.3)
        self.sound("explosion", at, -1)
        if direct is not None:
            self.damage(direct[0], direct[1], st["damage"] * p.dmg_mult, "mortar", p.owner, at, effect_ok=True)
        T = self.targets()
        ids, tx, ty, tz, tr, th, ks = T
        if not ids:
            return
        hd = np.maximum(0.0, np.hypot(tx - at[0], ty - at[1]) - tr)
        vd = np.maximum(0.0, np.maximum(tz - at[2], at[2] - (tz + th)))
        dist = np.hypot(hd, vd)
        near = np.nonzero(dist <= R)[0]
        if len(near) == 0:
            return
        back = at - p.vel / max(1e-6, np.linalg.norm(p.vel)) * 0.4
        pts = np.stack([tx[near], ty[near], tz[near] + th[near] * 0.5], 1)
        ok = self.world.los_batch(np.repeat(back[None, :], len(near), 0), pts)
        for k, vis in zip(near, ok):
            if not vis:
                continue
            fall = 1.0 - (1.0 - edge) * dist[k] / R
            dmg = st["splash"] * fall * p.dmg_mult
            tid, kind = ids[k], ks[k]
            if kind == 0 and tid == p.owner:
                dmg *= self.cfg["mech"]["self_damage_mult"]
            self.damage(tid, kind, dmg, "mortar", p.owner, at, effect_ok=False)
            if kind == 0:
                m = self.mechs[tid]
                if m.alive:
                    dx, dy = m.pos[0] - at[0], m.pos[1] - at[1]
                    L = math.hypot(dx, dy) or 1.0
                    m.vel = m.vel + np.array([dx / L * st["knock"] * fall, dy / L * st["knock"] * fall, st["knock_up"] * fall])
                    m.grounded = False
                    m.energy = max(0.0, m.energy - st["drain"] * fall)
                    m.events.append({"type": "effect", "effect": "concuss", "by": p.owner})

    def step_projs(self, dt: float) -> None:
        if not self.projs:
            return
        g = self.cfg["weapons"]["mortar"]["gravity"]
        P0 = np.array([p.pos for p in self.projs])
        V = np.array([p.vel for p in self.projs])
        isM = np.array([p.kind == "mortar" for p in self.projs])
        V1 = V.copy()
        V1[isM, 2] -= g * dt
        P1 = P0 + (V + V1) * 0.5 * dt
        fw = self.world.first_block(P0, P1, step=0.5)
        T = self.targets()
        S = self.seg_hits(P0, P1, T, self.cfg["projectile"].get("gun_radius", 0.3)) if T[0] else None
        col = {tid: k for k, tid in enumerate(T[0])}
        keep = []
        for n, p in enumerate(self.projs):
            sh, k = np.inf, -1
            if S is not None:
                row = S[n]
                own = col.get(p.owner)
                if own is not None and np.isfinite(row[own]):
                    row = row.copy()
                    row[own] = np.inf
                k = int(np.argmin(row))
                sh = row[k]
            if sh < fw[n]:
                at = P0[n] + (P1[n] - P0[n]) * sh
                came_from = P0[n] - V[n] * 0.2
                if p.kind == "mortar":
                    p.vel = V1[n]
                    self.explode(p, at, (T[0][k], T[6][k]))
                elif p.kind == "gun":
                    self.damage(T[0][k], T[6][k], self.mechs[p.owner].ws["gun"]["damage"] * p.dmg_mult,
                                "gun", p.owner, came_from)
                elif T[6][k] == 0:
                    self.damage(T[0][k], 0, self.cfg["npc"]["damage"], "npc", p.owner, came_from)
                continue
            if fw[n] < 1.0:
                if p.kind == "mortar":
                    p.vel = V1[n]
                    self.explode(p, P0[n] + (P1[n] - P0[n]) * max(0.0, fw[n] - 0.02), None)
                continue
            p.pos, p.vel = P1[n], V1[n]
            p.ttl -= dt
            if p.ttl <= 0:
                if p.kind == "mortar":
                    self.explode(p, p.pos, None)
                continue
            keep.append(p)
        self.projs = keep

    # ------------------------------------------------------------------ NPCs
    def step_npcs(self, dt: float) -> None:
        nc = self.cfg["npc"]
        for n in list(self.npcs.values()):
            if not n.alive:
                continue
            n.cooldown = max(0.0, n.cooldown - dt)
            tgt = self.mechs[n.target] if n.target is not None else None
            home_d = math.hypot(*(n.pos[:2] - n.home[:2]))
            if tgt is not None:
                d = math.hypot(*(tgt.pos[:2] - n.pos[:2]))
                if not tgt.alive or d > nc["deaggro_range"] or home_d > nc["chase_leash"]:
                    n.target, tgt = None, None
                    n.goal = n.home.copy()
            if tgt is not None:
                d = math.hypot(*(tgt.pos[:2] - n.pos[:2]))
                n.goal = tgt.pos.copy() if d > 20 else n.pos.copy()
                aimp = tgt.pos + np.array([0, 0, tgt.h * 0.5])
                eye = n.pos + np.array([0, 0, n.h * 0.85])
                n.aim = [math.atan2(aimp[1] - eye[1], aimp[0] - eye[0]),
                         math.atan2(aimp[2] - eye[2], max(1e-6, math.hypot(*(aimp[:2] - eye[:2]))))]
                if n.cooldown <= 0 and (self.tick % 2 == 0) and self.world.los(eye, aimp):
                    sp = math.radians(nc["spread_deg"])
                    dv = dir3(n.aim[0] + self.rng.gauss(0, sp), n.aim[1] + self.rng.gauss(0, sp))
                    self.projs.append(Proj(self._nid("proj"), "npc", n.id, eye + dv * (n.r + 0.4),
                                           dv * nc["projectile_speed"], 2.0, 1.0))
                    self.sound("npc_gun", n.pos, n.id)
                    n.cooldown = nc["cooldown"]
            elif math.hypot(*(n.goal[:2] - n.pos[:2])) < 1.5:
                a, r = self.rng.uniform(0, 2 * math.pi), nc["leash"] * math.sqrt(self.rng.random())
                n.goal = n.home + np.array([r * math.cos(a), r * math.sin(a), 0])
            dx, dy = n.goal[0] - n.pos[0], n.goal[1] - n.pos[1]
            L = math.hypot(dx, dy)
            if L > 0.5:
                sp = nc["speed"] * (1.0 if tgt is not None else 0.5)
                st = min(L, sp * dt)
                nx_, ny_ = n.pos[0] + dx / L * st, n.pos[1] + dy / L * st
                z = n.pos[2]
                if not self.world.wall_contact(nx_, ny_, z, n.r, n.h, 1.2)[0] and abs(self.world.ground_at(nx_, ny_) - z) < 1.5:
                    n.pos = np.array([nx_, ny_, self.world.surface(nx_, ny_, z, 1.2)])
                    n.vel = np.array([dx / L * sp, dy / L * sp, 0.0])
                else:
                    n.goal = n.pos.copy()
                    n.vel = np.zeros(3)
            else:
                n.vel = np.zeros(3)

    def npc_aggro(self, alive: List[Mech]) -> None:
        nc = self.cfg["npc"]
        rays_a, rays_b, pairs = [], [], []
        for n in self.npcs.values():
            if n.target is not None or math.hypot(*(n.pos[:2] - n.home[:2])) > nc["leash"]:
                continue
            best, bd = None, nc["aggro_range"]
            for m in alive:
                d = float(np.linalg.norm(m.pos - n.pos))
                if d < bd:
                    best, bd = m, d
            if best is not None:
                rays_a.append(n.pos + np.array([0, 0, n.h * 0.85]))
                rays_b.append(best.pos + np.array([0, 0, best.h * 0.5]))
                pairs.append((n, best))
        if pairs:
            ok = self.world.los_batch(rays_a, rays_b)
            for (n, m), v in zip(pairs, ok):
                if v:
                    n.target = m.id

    # ------------------------------------------------------------------ world tick
    def step_world(self, dt: float, zc, zr: float, dps: float) -> None:
        mc = self.cfg["mech"]
        for m in self.alive_mechs():
            d = math.hypot(m.pos[0] - zc[0], m.pos[1] - zc[1])
            m.in_zone = d <= zr
            if not m.in_zone:
                self.damage(m.id, 0, dps * dt, "zone", None, m.pos)
                if not m.alive:
                    continue
            elif self.t - m.last_combat >= m.m["regen_delay"] and m.hp < m.max_hp:
                m.hp = min(m.max_hp, m.hp + m.m["regen_per_s"] * dt)
            if self.t - m.last_spend >= m.m["energy_regen_delay"]:
                m.energy = min(m.max_energy, m.energy + m.m["energy_regen"] * dt)
            if m.offer is not None and self.t >= m.offer["expires"]:
                self.apply_upgrade(m, m.offer["choices"][0]["id"])
        cr = self.cfg["cubes"]["pickup_radius"]
        alive = self.alive_mechs()
        if alive and self.cubes:
            for cid in list(self.cubes.keys()):
                c = self.cubes[cid]
                best, bd = None, None
                for m in alive:
                    dh = math.hypot(m.pos[0] - c[0], m.pos[1] - c[1])
                    if dh <= cr and abs(m.pos[2] - c[2]) <= 4.0:
                        if bd is None or dh < bd - 1e-9:
                            best, bd = m, dh
                if best is not None:
                    del self.cubes[cid]
                    self.gain_cubes(best, 1)
                    best.events.append({"type": "pickup", "cubes": best.cubes})
        self.step_drops(dt)

    def step_drops(self, dt: float) -> None:
        dc = self.cfg.get("drops")
        if not dc:
            return
        while self.drop_calls and self.t >= self.drop_calls[0]:
            self.drop_calls.pop(0)
            _, _, nc, nr, _, _, _ = self.zone.state(self.t)
            p = self._open_point(nc, nr * dc.get("spread", 0.7), margin=20)
            if p is None:
                p = self._open_point(nc, nr, margin=10)
            if p is not None:
                self.drops[self._nid("drop")] = {"pos": p, "land_t": self.t + dc["fall_s"], "holder": None,
                                                 "progress": 0.0, "landed": False}
        for did in list(self.drops.keys()):
            d = self.drops[did]
            if self.t < d["land_t"]:
                continue
            if not d["landed"]:
                d["landed"] = True
                self.sound("drop", d["pos"], -1)
            inside = [m for m in self.alive_mechs()
                      if math.hypot(*(m.pos[:2] - d["pos"][:2])) <= dc["radius"] and abs(m.pos[2] - d["pos"][2]) < 6]
            if len(inside) == 1:
                m = inside[0]
                if d["holder"] != m.id:
                    d["holder"], d["progress"] = m.id, 0.0
                d["progress"] += dt
                if d["progress"] >= dc["capture_s"]:
                    del self.drops[did]
                    self.gain_cubes(m, int(dc["cubes"]))
                    m.hp = max(m.hp, m.max_hp * dc.get("repair", 1.0))
                    m.energy = m.max_energy
                    m.events.append({"type": "drop", "id": did, "cubes": m.cubes})
            elif len(inside) == 0:
                d["holder"], d["progress"] = None, 0.0

    # ------------------------------------------------------------------ observations
    def build_obs(self, alive: List[Mech]) -> Dict[int, dict]:
        cfg = self.cfg
        vc = cfg["vision"]
        sc = cfg["sound"]
        self.npc_aggro(alive)
        actors: List[Tuple[int, Any]] = [(m.id, m) for m in alive] + [(n.id, n) for n in self.npcs.values() if n.alive]
        A = np.array([a.pos for _, a in actors]) if actors else np.zeros((0, 3))
        AH = np.array([a.h for _, a in actors]) if actors else np.zeros(0)
        visible: Dict[int, set] = {m.id: set() for m in alive}
        rays_a, rays_b, rays_c, owners = [], [], [], []
        geo = {}
        for m in alive:
            eye = m.pos + np.array([0, 0, m.eye])
            near = vc["near_range"] + m.m.get("near_range_bonus", 0.0)
            rng_ = vc["range"] + m.m.get("vision_bonus", 0.0)
            half = math.radians(vc["fov_deg"]) / 2
            geo[m.id] = (eye, near, rng_, half)
            if len(A):
                d = np.hypot(A[:, 0] - eye[0], A[:, 1] - eye[1])
                yaw = np.arctan2(A[:, 1] - eye[1], A[:, 0] - eye[0])
                dy = np.abs((yaw - m.aim[0] + np.pi) % (2 * np.pi) - np.pi)
                cand = np.nonzero((d <= near) | ((d <= rng_) & (dy <= half)))[0]
                for k in cand:
                    aid = actors[k][0]
                    if aid == m.id:
                        continue
                    rays_a.append(eye)
                    rays_b.append(A[k] + np.array([0, 0, AH[k] * 0.5]))
                    rays_c.append(A[k] + np.array([0, 0, AH[k]]))
                    owners.append((m.id, aid))
        if owners:
            ok = self.world.los_batch(rays_a, rays_b, vision=True, step=1.5)
            miss = np.nonzero(~ok)[0]
            if len(miss):
                ok2 = self.world.los_batch([rays_a[k] for k in miss], [rays_c[k] for k in miss], vision=True, step=1.5)
                ok[miss[ok2]] = True
            for (oid, aid), v in zip(owners, ok):
                if v:
                    visible[oid].add(aid)
        # items: refreshed every third decision per observer
        dec = self.tick // self.decide_every
        items = [(cid, c["pos"] + np.array([0, 0, 1.2])) for cid, c in self.crates.items()] + \
                [(cid, p + np.array([0, 0, 0.6])) for cid, p in self.cubes.items()] + \
                [(pid, p["pos"] + np.array([0, 0, 0.6])) for pid, p in self.pickups.items()]
        if items:
            IP = np.array([p for _, p in items])
            ra, rb, ow = [], [], []
            for m in alive:
                if (dec + m.id) % 3:
                    continue
                eye, near, rng_, half = geo[m.id]
                d = np.hypot(IP[:, 0] - eye[0], IP[:, 1] - eye[1])
                yaw = np.arctan2(IP[:, 1] - eye[1], IP[:, 0] - eye[0])
                dy = np.abs((yaw - m.aim[0] + np.pi) % (2 * np.pi) - np.pi)
                cand = np.nonzero((d <= near) | ((d <= rng_) & (dy <= half)))[0]
                m.vis_items = set()
                for k in cand:
                    ra.append(eye)
                    rb.append(IP[k])
                    ow.append((m, items[k][0]))
            if ow:
                ok = self.world.los_batch(ra, rb, vision=True, step=1.5)
                for (m, iid), v in zip(ow, ok):
                    if v:
                        m.vis_items.add(iid)
        # sounds
        heard: Dict[int, List[dict]] = defaultdict(list)
        sigma0 = math.radians(sc["bearing_noise_deg"])
        for kind, pos, ts, src in self.sound_q:
            rr = sc["range"].get(kind, 0)
            for m in alive:
                if m.id == src:
                    continue
                d = math.hypot(pos[0] - m.pos[0], pos[1] - m.pos[1])
                if d > rr:
                    continue
                b = math.atan2(pos[1] - m.pos[1], pos[0] - m.pos[0]) + self.rng.gauss(0, sigma0 * m.m.get("hearing", 1.0))
                heard[m.id].append({"kind": kind, "bearing": round(wrap(b), 3),
                                    "dist": "near" if d < sc["near"] else "mid" if d < sc["mid"] else "far",
                                    "age": round(max(0.0, self.t - ts), 2)})
        self.sound_q = []
        beams = self.beam_q
        self.beam_q = []
        zc, zr, nc, nr, ss, se, dps = self.zone.state(self.t)
        zone = {"center": [round(zc[0], 2), round(zc[1], 2)], "radius": round(zr, 2),
                "next_center": [round(nc[0], 2), round(nc[1], 2)], "next_radius": nr,
                "shrink_start": ss, "shrink_end": se, "dps": dps}
        dc = cfg.get("drops") or {}
        drops = [{"id": did, "pos": [round(v, 2) for v in d["pos"]], "radius": dc.get("radius", 8),
                  "lands_in": round(max(0.0, d["land_t"] - self.t), 2), "capture_s": dc.get("capture_s", 4),
                  "progress": round(d["progress"], 2), "holder": d["holder"]} for did, d in self.drops.items()]
        out = {}
        for m in alive:
            eye, near, rng_, half = geo[m.id]
            for aid in visible[m.id]:
                m.memory[aid] = self.t
            ms = []
            for aid, a in actors:
                if aid == m.id:
                    continue
                seen = m.memory.get(aid)
                if seen is None or self.t - seen > vc.get("memory_s", 5):
                    continue
                e = {"id": aid, "kind": "mech" if aid < 100 else "npc",
                     "pos": [round(v, 2) for v in a.pos], "vel": [round(v, 2) for v in a.vel],
                     "aim": [round(a.aim[0], 3), round(a.aim[1], 3)]}
                if vc.get("show_enemy_hp", True):
                    e["hp"], e["max_hp"] = round(a.hp, 2), round(a.max_hp, 2)
                if aid < 100:
                    e["cubes"] = a.cubes
                    e["weapons"] = [w.kind for w in a.weapons if w is not None]
                    e["charging"] = a.charging()
                ms.append(e)
            projs = []
            for p in self.projs:
                if p.owner == m.id:
                    continue
                d = math.hypot(p.pos[0] - eye[0], p.pos[1] - eye[1])
                yawp = math.atan2(p.pos[1] - eye[1], p.pos[0] - eye[0])
                if d <= near or (d <= rng_ and abs(wrap(yawp - m.aim[0])) <= half):
                    projs.append({"id": p.id, "kind": p.kind, "pos": [round(v, 2) for v in p.pos],
                                  "vel": [round(v, 2) for v in p.vel]})
            bs = []
            for b in beams:
                if b["by"] == m.id:
                    continue
                pts = [np.array(b["path"][0]), np.array(b["path"][-1]), (np.array(b["path"][0]) + np.array(b["path"][-1])) / 2]
                for q in pts:
                    d = math.hypot(q[0] - eye[0], q[1] - eye[1])
                    yq = math.atan2(q[1] - eye[1], q[0] - eye[0])
                    if d <= near or (d <= rng_ and abs(wrap(yq - m.aim[0])) <= half):
                        bs.append({"by": b["by"], "kind": b["kind"], "path": [[round(v, 2) for v in pt] for pt in b["path"]],
                                   "age": round(self.t - b["t"], 2)})
                        break
            weapons = []
            for w in m.weapons:
                if w is None:
                    weapons.append(None)
                    continue
                wo = {"kind": w.kind, "cooldown": round(max(w.cooldown, w.equip, 0.0), 3), "stats": dict(m.ws[w.kind])}
                if w.kind == "gun":
                    wo["ammo"], wo["reload"] = w.ammo, round(max(0.0, w.reload), 3)
                if w.kind == "laser":
                    wo["charge"] = round(w.charge if w.charging else 0.0, 3)
                weapons.append(wo)
            stats = dict(m.m)
            stats.update({"radius": m.r, "height": m.h, "eye_height": m.eye, "muzzle_height": m.muzzle})
            vi = m.vis_items
            o = {"type": "obs", "tick": self.tick, "t": round(self.t, 2), "alive": len(alive),
                 "self": {"id": m.id, "pos": [round(v, 3) for v in m.pos], "vel": [round(v, 3) for v in m.vel],
                          "aim": [round(m.aim[0], 4), round(m.aim[1], 4)], "hp": round(m.hp, 2),
                          "max_hp": round(m.max_hp, 2), "energy": round(m.energy, 2), "max_energy": m.max_energy,
                          "grounded": m.grounded, "cubes": m.cubes, "upgrades": list(m.upg_list),
                          "in_zone": m.in_zone, "weapons": weapons, "stats": stats},
                 "mechs": ms, "projectiles": projs,
                 "crates": [{"id": cid, "pos": [round(v, 2) for v in c["pos"]], "hp": round(c["hp"], 1)}
                            for cid, c in self.crates.items() if cid in vi],
                 "cubes": [{"id": cid, "pos": [round(v, 2) for v in p]} for cid, p in self.cubes.items() if cid in vi],
                 "pickups": [{"id": pid, "kind": p["kind"], "pos": [round(v, 2) for v in p["pos"]]}
                             for pid, p in self.pickups.items() if pid in vi],
                 "sounds": heard.get(m.id, []), "beams": bs, "events": m.events, "drops": drops,
                 "zone": zone, "offer": m.offer}
            m.events = []
            out[m.id] = o
        return out
