"""Minimum Risk Movement + Utility Scoring bot for mechfight/1 (practice-squad calibration).

Every tick the bot samples a bounded set of reachable destinations P_t (three rings of 12
directions over a 0.4-1.4 s horizon, plus tactical anchors toward cover, the zone and loot),
vetoes illegal or immediately lethal ones with -1,000,000 and scores the rest with

    U(p) = - w_zone f_zone - w_cross a(h) f_cross - w_aoe f_aoe - w_trap f_trap - w_kin f_kin
           + w_cover a(h) f_cover + w_dist f_dist + w_loot b(h) f_loot + w_space f_space

with every f in [0, 1] and phase weights summing to 100. The practice squad (15 house bots that
share vision, never fight each other and hunt us) gets the document's defensive variant: +5 on
w_cross and w_cover paid by w_loot and w_dist in the first two phases, every enemy treated as
aiming at us (psi = 1.5), and a(h) active from h < 0.50.

Targets are scored with S(e) = u_hit + u_finish (1 - h_e) + u_duel [aims at me] - u_obst and only
switched when another one beats the current by 10%.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from brain.aim import lead, wrap, yaw_pitch
from brain.nav import NavField, NavGrid
from brain.world import World

# Practice-squad variant of the document's weight table (sum of magnitudes = 100 per phase).
PHASE_W = {
    "early": dict(zone=14, cross=27, aoe=10, trap=12, kin=6, cover=19, dist=1, loot=5, space=6),
    "mid": dict(zone=18, cross=25, aoe=12, trap=11, kin=7, cover=19, dist=3, loot=1, space=4),
    "final": dict(zone=20, cross=14, aoe=14, trap=9, kin=12, cover=11, dist=16, loot=2, space=2),
}

DEFAULTS = {
    "budget_s": 0.022,       # internal compute budget per tick (document: 20-25 ms)
    "alpha_h": 0.50,         # a(h) activates below this HP fraction (practice variant)
    "gamma": 1.25,           # a(h) = 1 + gamma (1 - h)
    "psi": 1.5,              # every squad member is hunting us
    "d_ref": 100.0,          # f_cross distance scale (m)
    "eps": 0.5,
    "decay_s": 3.0,          # memory decay of last-known enemy positions
    "los_keep": 0.15,        # f_cross fraction kept when a building blocks the line
    "los_keep_mortar": 0.5,
    "d_loot": 40.0,
    "b_loot": 10.0,
    "u_hit": 1.0,
    "u_finish": 0.8,
    "u_duel": 1.0,
    "u_obst": 1.5,
    "hysteresis": 1.10,
    "fire_min": 0.35,        # minimum S_target to pull the trigger
    "energy_reserve": 30.0,
    "safe_frac": 0.72,
}

UPGRADE_ORDER = ["plating", "armor", "nanites", "servos", "sensors", "jets", "capacitor", "reactor",
                 "gyros", "lightning_efficiency", "lightning_range", "lightning_chain", "gun_mag",
                 "gun_rate", "gun_damage", "gun_velocity", "laser_charge", "laser_damage",
                 "laser_pierce", "mortar_reload", "mortar_velocity", "mortar_splash"]

COMBAT_SOUNDS = {"gun": 0.6, "laser": 1.0, "laser_charge": 1.0, "lightning": 0.8, "mortar": 0.7,
                 "explosion": 0.4, "npc_gun": 0.0}
MOVE_SOUNDS = {"step": 1.0, "boost": 0.8, "jet": 0.8, "reload": 0.8}
BAND_DIST = {"near": 35.0, "mid": 110.0, "far": 240.0}
EST_DIST = {"laser": 150.0, "gun": 110.0, "lightning": 30.0, "mortar": 150.0}
NDIR = 16
RING_DIRS = 12
RINGS = (5.0, 10.0, 16.0)
# r_opt(w) and sigma_w for f_dist; cooling down adds R_COOL to back out of the enemy's reach
R_OPT = {"lightning": (30.0, 10.0), "gun": (95.0, 35.0), "laser": (170.0, 45.0), "mortar": (150.0, 45.0)}
R_COOL = 40.0


def _params() -> dict:
    p = dict(DEFAULTS)
    raw = os.environ.get("AGENT_PARAMS")
    if raw:
        try:
            p.update(json.loads(raw))
        except ValueError:
            pass
    return p


class Threat:
    __slots__ = ("id", "pos", "vel", "aim", "weapons", "charging", "hp", "max_hp", "seen", "est", "conf")

    def __init__(self, tid: int):
        self.id = tid
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.aim: Optional[float] = None
        self.weapons: set = {"gun"}
        self.charging = False
        self.hp = 1400.0
        self.max_hp = 1400.0
        self.seen = -1e9
        self.est = True
        self.conf = 0.0


def weapon_danger(kinds, d: float) -> float:
    """threat(e): how dangerous the enemy's arsenal is at distance d, in [0, 1]."""
    if "npc" in kinds:
        return 0.3 if d <= 50 else 0.12 if d <= 95 else 0.0
    s = 0.0
    if "laser" in kinds and d <= 295:
        s = max(s, min(1.0, (300.0 - d) / 130.0))
    if "gun" in kinds and d <= 235:
        s = max(s, 0.65 * (1.15 - d / 235))
    if "lightning" in kinds and d <= 55:
        s = 1.0
    if "mortar" in kinds and 12 <= d <= 240:
        s = max(s, 0.6)
    return min(1.0, s)


class Agent:
    name = "survivor"

    def __init__(self, hello: dict):
        self.P = _params()
        self.cfg = hello.get("config", {})
        self.rng = random.Random(hello.get("seed", 0))
        mp = hello.get("map", {})
        self.world = World(mp.get("dir") or "/arena", mp.get("files"))
        self.mech = self.cfg.get("mech", {})
        self.nav = NavGrid(self.world, self.mech)
        self.escape = self._escape_map()
        self.field: Optional[NavField] = None
        self.field_key = None
        self.pending: Optional[Tuple[tuple, NavField]] = None
        self.cover: Optional[NavField] = None
        self.cover_key = None
        self.threats: Dict[int, Threat] = {}
        self.blobs: List[Tuple[float, float, float, str, float]] = []
        self.attackers: Dict[int, float] = {}
        self.last_hit_t = -1e9
        self.last_dmg_bearing: Optional[Tuple[float, float]] = None
        self.target_id: Optional[int] = None
        self.target_s = 0.0
        self.dest: Optional[Tuple[float, float]] = None
        self.heading: Optional[Tuple[float, float]] = None
        self.straight_s = 0.0
        self.still_s = 0.0
        self.last_pos: Optional[Tuple[float, float]] = None
        self.stuck = 0.0
        self.unstick_until = -1.0
        self.unstick_dir = (1.0, 0.0)
        self.scan_yaw: Optional[float] = None
        self.scan_t = 0.0
        self.look_age = np.zeros(NDIR)
        self.mode = "early"
        self.t = 0.0

    def _escape_map(self) -> np.ndarray:
        """Fraction of the 8 directions with at least two walkable steps (8 m) from each cell."""
        nav = self.nav
        from brain.nav import DIRS8, shift
        esc = np.zeros(nav.walk.shape, np.float32)
        for d, (dj, di, _) in enumerate(DIRS8):
            ok1 = np.isfinite(nav.cost_full[d])
            ok2 = shift(ok1, dj, di)
            esc += (ok1 & ok2).astype(np.float32)
        return esc / 8.0

    # ================================================================== main
    def act(self, obs: dict) -> dict:
        t0 = time.perf_counter()
        self.deadline = t0 + self.P["budget_s"]
        self.t = float(obs["t"])
        me = obs["self"]
        self.me = me
        self.pos = np.array(me["pos"], dtype=float)
        self.zone = obs["zone"]
        self.alive = int(obs.get("alive", 16))
        a: dict = {}
        if obs.get("offer"):
            a["pick"] = self.pick_upgrade(obs["offer"])
        self.update_zone(obs)
        self.update_threats(obs)
        self.mode = self.phase()
        locks = self.laser_locks(obs)
        incoming = self.incoming(obs)
        target = self.select_target()
        move, boost, jump = self.plan_move(obs, locks, incoming, target)
        aim, fire = self.plan_aim_fire(locks, target)
        a.update(self.plan_take(obs))
        gun = next((w for w in me["weapons"] if w and w["kind"] == "gun"), None)
        if target is None and gun is not None and gun.get("reload", 0) <= 0 \
                and gun.get("ammo", 12) < gun["stats"].get("mag", 12) // 2:
            a["reload"] = True
        a.update(move=[round(move[0], 4), round(move[1], 4)], boost=boost, jump=jump,
                 aim=[round(aim[0], 4), round(aim[1], 4)], fire=fire)
        return a

    def phase(self) -> str:
        """Phase from mechs alive; the squad rarely loses members, so a closing zone also
        advances the phase."""
        n, r = self.alive, float(self.zone.get("radius", 1e9))
        if n <= 4 or r <= 40:
            return "final"
        if n <= 10 or r <= 120:
            return "mid"
        return "early"

    # ================================================================== zone
    def safe_radius(self, nr: float) -> float:
        return max(4.0, nr * self.P["safe_frac"])

    def update_zone(self, obs: dict) -> None:
        z = obs["zone"]
        nc, nr = z["next_center"], z["next_radius"]
        key = (round(nc[0], 1), round(nc[1], 1), round(nr, 1))
        if self.field_key != key and (self.pending is None or self.pending[0] != key):
            goal = self.nav.circle_mask(nc[0], nc[1], self.safe_radius(nr))
            if not (goal & self.nav.walk).any():
                goal = self.nav.circle_mask(nc[0], nc[1], max(12.0, nr))
            self.pending = (key, NavField(self.nav, goal))
        if self.pending is not None:
            budget = 0.006 if self.field is not None else 0.010
            if self.pending[1].run(budget_s=budget):
                self.field_key, self.field = self.pending
                self.pending = None

    def _cells(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nav = self.nav
        I = np.clip(np.rint((xs - nav.w.minx) / nav.w.step - nav.k // 2) / nav.k, 0, nav.NX - 1).astype(np.intp)
        J = np.clip(np.rint((ys - nav.w.miny) / nav.w.step - nav.k // 2) / nav.k, 0, nav.NY - 1).astype(np.intp)
        return J, I

    def _field_dists(self, f: Optional[NavField], xs: np.ndarray, ys: np.ndarray, fallback) -> np.ndarray:
        if f is None:
            return np.broadcast_to(np.asarray(fallback, dtype=float), xs.shape).copy()
        J, I = self._cells(xs, ys)
        dj, di = J - f.j0, I - f.i0
        ok = (dj >= 0) & (dj < f.D.shape[0]) & (di >= 0) & (di < f.D.shape[1])
        out = np.full(len(xs), np.inf)
        out[ok] = f.D[dj[ok], di[ok]]
        cx, cy = self.nav.center_batch(J, I)
        return np.where(np.isfinite(out), out + np.hypot(cx - xs, cy - ys), fallback)

    def field_dist(self, x: float, y: float) -> float:
        z = self.zone
        fb = max(0.0, math.dist((x, y), z["next_center"]) - self.safe_radius(z["next_radius"]))
        return float(self._field_dists(self.field, np.array([x]), np.array([y]), fb)[0])

    def time_to_edge(self, x: float, y: float) -> float:
        """Seconds until the zone edge passes (x, y) in the current phase (inf if it will not)."""
        z = self.zone
        c, r = z["center"], z["radius"]
        nc, nr = z["next_center"], z["next_radius"]
        if math.dist((x, y), c) > r:
            return 0.0
        if math.dist((x, y), nc) <= nr:
            return math.inf
        t0 = max(self.t, z["shrink_start"])
        span = max(1e-3, z["shrink_end"] - t0)
        lo, hi = 0.0, 1.0
        for _ in range(14):
            f = (lo + hi) / 2
            cx, cy = c[0] + (nc[0] - c[0]) * f, c[1] + (nc[1] - c[1]) * f
            if math.hypot(x - cx, y - cy) <= r + (nr - r) * f:
                lo = f
            else:
                hi = f
        return (t0 - self.t) + lo * span

    def zone_urgency(self) -> float:
        x, y = self.pos[0], self.pos[1]
        z = self.zone
        if math.dist((x, y), z["center"]) > z["radius"]:
            return 1.0
        if math.dist((x, y), z["next_center"]) <= self.safe_radius(z["next_radius"]):
            return 0.0
        need = self.field_dist(x, y) / (self.mech.get("walk_speed", 11.7) * 0.8) + 10.0
        left = self.time_to_edge(x, y)
        if left == math.inf:
            return 0.15 if z["shrink_start"] - self.t > need + 20 else 0.5
        slack = left - need
        if slack <= 0:
            return 1.0
        return max(0.15, min(1.0, 1.0 - slack / 30.0))

    # ================================================================== threats
    def update_threats(self, obs: dict) -> None:
        t = self.t
        vis = {}
        for m in obs["mechs"]:
            npc = m.get("kind") != "mech"
            th = self.threats.get(m["id"]) or Threat(m["id"])
            th.pos = np.array(m["pos"], dtype=float)
            th.vel = np.array(m.get("vel", (0, 0, 0)), dtype=float)
            th.aim = m["aim"][0] if m.get("aim") else None
            th.weapons = {"npc"} if npc else set(m.get("weapons") or ["gun"])
            th.charging = bool(m.get("charging"))
            th.hp = float(m.get("hp", th.hp))
            th.max_hp = float(m.get("max_hp", th.max_hp))
            th.seen, th.est, th.conf = t, False, 1.0
            self.threats[m["id"]] = th
            vis[m["id"]] = th
        for b in obs.get("beams", []):
            by = b.get("by")
            if by is None or by in vis or by >= 100:
                continue
            th = self.threats.get(by) or Threat(by)
            p0 = b["path"][0]
            th.pos = np.array([p0[0], p0[1], p0[2] - self.mech.get("muzzle_height", 7.0)])
            th.weapons.add(b["kind"])
            th.seen, th.est, th.conf, th.charging = t, True, 0.8, False
            self.threats[by] = th
        for e in obs.get("events", []):
            if e.get("type") == "damaged" and e.get("weapon") != "zone":
                self.last_hit_t = t
                if e.get("bearing") is not None:
                    self.last_dmg_bearing = (e["bearing"], t)
                by = e.get("by")
                if by is not None:
                    self.attackers[by] = t
                if by is not None and by not in vis and e.get("bearing") is not None:
                    th = self.threats.get(by)
                    if th is None or t - th.seen > 1.0:
                        th = th or Threat(by)
                        d = 30.0 if by >= 100 else EST_DIST.get(e.get("weapon"), 100.0)
                        th.pos = self.pos + np.array([d * math.cos(e["bearing"]), d * math.sin(e["bearing"]), 0.0])
                        th.est, th.conf, th.seen = True, 0.6, t
                        th.aim = wrap(e["bearing"] + math.pi)
                    th.weapons.add("npc" if by >= 100 else e.get("weapon", "gun"))
                    self.threats[by] = th
        for th in list(self.threats.values()):
            age = t - th.seen
            if age > (6.0 if not th.est else 8.0):
                del self.threats[th.id]
            elif th.id not in vis:
                th.charging = False
        for d in obs.get("drops", []):
            h = d.get("holder")
            if h is not None and h != self.me["id"] and h not in vis:
                th = self.threats.get(h) or Threat(h)
                th.pos = np.array(d["pos"], dtype=float)
                th.seen, th.est, th.conf = t, True, 0.7
                self.threats[h] = th
        self.blobs = [b for b in self.blobs if t - b[2] < 6.0]
        for s in obs.get("sounds", []):
            k = s.get("kind")
            w = COMBAT_SOUNDS.get(k, MOVE_SOUNDS.get(k, 0.0))
            if w <= 0 or s.get("bearing") is None:
                continue
            d = 25.0 if k in MOVE_SOUNDS else BAND_DIST.get(s.get("dist"), 200.0)
            x = self.pos[0] + d * math.cos(s["bearing"])
            y = self.pos[1] + d * math.sin(s["bearing"])
            self.blobs.append((x, y, t - s.get("age", 0.0), k, w))

    def enemies(self) -> List[Threat]:
        """E_t: squad mechs we know of, plus NPC drones that are shooting at us."""
        out = []
        for th in self.threats.values():
            if th.id < 0:
                continue
            if th.id >= 100 and self.t - self.attackers.get(th.id, -1e9) > 3.0:
                continue
            out.append(th)
        return out

    def aims_at_me(self, th: Threat, memory: float = 1.5) -> bool:
        if self.t - self.attackers.get(th.id, -1e9) < memory:
            return True
        if th.est or th.aim is None or self.t - th.seen > 0.3:
            return False
        d = math.dist(th.pos[:2], self.pos[:2])
        b = math.atan2(self.pos[1] - th.pos[1], self.pos[0] - th.pos[0])
        return abs(wrap(th.aim - b)) < math.atan2(6.0, max(d, 1.0)) + 0.06

    def laser_locks(self, obs: dict) -> List[Threat]:
        out = [th for th in self.threats.values() if th.charging and not th.est and self.aims_at_me(th)]
        for s in obs.get("sounds", []):
            if s.get("kind") == "laser_charge" and s.get("dist") in ("near", "mid") and s.get("bearing") is not None:
                th = Threat(-1)
                d = BAND_DIST[s["dist"]]
                th.pos = self.pos + np.array([d * math.cos(s["bearing"]), d * math.sin(s["bearing"]), 0.0])
                th.weapons, th.charging, th.est, th.conf = {"laser"}, True, True, 0.5
                out.append(th)
        return out

    def incoming(self, obs: dict) -> List[Tuple[str, np.ndarray, np.ndarray, float]]:
        """Enemy projectiles that might reach us: (kind, pos, vel, time)."""
        out = []
        g = (self.cfg.get("weapons") or {}).get("mortar", {}).get("gravity", 22.0)
        for p in obs.get("projectiles", []):
            pos = np.array(p["pos"], dtype=float)
            vel = np.array(p["vel"], dtype=float)
            rel = self.pos - pos
            if p.get("kind") == "mortar":
                q, v = pos.copy(), vel.copy()
                for k in range(1, 80):
                    v[2] -= g * 0.1
                    q = q + v * 0.1
                    if q[2] <= self.world.roof_top(q[0], q[1]) + 0.5:
                        if math.dist(q[:2], self.pos[:2]) < 40:
                            out.append(("mortar", q, vel, k * 0.1))
                        break
            else:
                sp = float(np.linalg.norm(vel[:2])) or 1.0
                along = float(rel[:2] @ vel[:2]) / sp
                if along < -5 or along > sp * 0.8:
                    continue
                perp = abs(float(rel[0] * vel[1] - rel[1] * vel[0])) / sp
                if perp < 12:
                    out.append((p.get("kind", "gun"), pos, vel, along / sp))
        return out

    def update_cover(self, srcs: List[Threat]) -> None:
        """Tactical anchor: local field (~70 m) of walking distance to cells shielded from srcs."""
        if not srcs:
            self.cover = None
            return
        x, y = float(self.pos[0]), float(self.pos[1])
        p = srcs[0].pos
        if self.cover is not None and self.cover_key is not None:
            kx, ky, px, py, kt = self.cover_key
            if self.t - kt < 0.5 and math.hypot(x - kx, y - ky) < 4 and math.hypot(p[0] - px, p[1] - py) < 8:
                return
        nav = self.nav
        J, I = nav.cell(x, y)
        R = 18
        j0, j1 = max(0, J - R), min(nav.NY, J + R + 1)
        i0, i1 = max(0, I - R), min(nav.NX, I + R + 1)
        X = nav.cx[i0:i1][None, :]
        Y = nav.cy[j0:j1][:, None]
        covered = nav.walk[j0:j1, i0:i1].copy()
        rf = nav.ray_free[:, j0:j1, i0:i1]
        for th in srcs:
            dx, dy = th.pos[0] - X, th.pos[1] - Y
            idx = np.rint(np.arctan2(dy, dx) * (NDIR / (2 * math.pi))).astype(np.int64) % NDIR
            free = np.take_along_axis(rf, idx[None], 0)[0]
            covered &= free < np.hypot(dx, dy) - nav.res
        f = NavField(nav, covered, window=(j0, j1, i0, i1))
        f.run(max_iters=40)
        self.cover = f
        self.cover_key = (x, y, float(p[0]), float(p[1]), self.t)

    # ================================================================== target selection
    def _hit_utility(self, d: float, kinds: List[Optional[str]]) -> float:
        me = self.me
        u = 0.0
        if "lightning" in kinds and d <= 44 and me["energy"] >= 3:
            u = max(u, 1.0)
        if "gun" in kinds:
            u = max(u, max(0.0, (230.0 - d) / 230.0))
        if "laser" in kinds and d <= 280:
            u = max(u, 0.9)
        if "mortar" in kinds and 20 <= d <= 235:
            u = max(u, 0.5)
        return u

    def _shot_clear(self, muzzle: np.ndarray, th: Threat) -> bool:
        """u_obst test: a shot from our muzzle reaches the target's feet, torso or head."""
        hs = (0.8, 1.75, 2.6) if th.id >= 100 else (2.0, 4.5, 7.5)
        B = np.array([th.pos + np.array([0.0, 0.0, h]) for h in hs])
        A = np.repeat(muzzle[None], len(hs), axis=0)
        return bool(self.world.los_batch(A, B, step=2.0).any())

    def select_target(self) -> Optional[Tuple[Threat, float, bool]]:
        """S(e) = u_hit + u_finish (1 - h_e) + u_duel [aims at me] - u_obst, 10% hysteresis.
        Against the squad a member that hit us in the last 8 s is still hunting us."""
        P = self.P
        kinds = [w["kind"] if w else None for w in self.me["weapons"]]
        mh = self.me["stats"].get("muzzle_height", self.mech.get("muzzle_height", 7.0))
        muzzle = self.pos + np.array([0.0, 0.0, mh])
        scored: Dict[int, Tuple[float, Threat, bool]] = {}
        for th in self.enemies():
            if th.est or self.t - th.seen > 0.3:
                continue
            d = math.dist(th.pos[:2], self.pos[:2])
            hit = self._hit_utility(d, kinds)
            if hit <= 0:
                continue
            clear = self._shot_clear(muzzle, th)
            s = P["u_hit"] * hit + P["u_finish"] * (1.0 - th.hp / max(1.0, th.max_hp)) \
                + P["u_duel"] * self.aims_at_me(th, memory=8.0) - (0.0 if clear else P["u_obst"])
            scored[th.id] = (s, th, clear)
        if not scored:
            self.target_id = None
            return None
        best_id = max(scored, key=lambda k: scored[k][0])
        cur = scored.get(self.target_id) if self.target_id is not None else None
        if cur is not None and scored[best_id][0] <= P["hysteresis"] * cur[0]:
            best_id = self.target_id
        self.target_id = best_id
        s, th, clear = scored[best_id]
        self.target_s = s
        return th, s, clear

    # ================================================================== movement
    def candidates(self) -> List[Tuple[float, float]]:
        x, y = float(self.pos[0]), float(self.pos[1])
        pts = [(x, y)]
        off = self.rng.uniform(0, 2 * math.pi / RING_DIRS)
        for r in RINGS:
            for k in range(RING_DIRS):
                a = off + 2 * math.pi * k / RING_DIRS
                pts.append((x + r * math.cos(a), y + r * math.sin(a)))
        for f in (self.field, self.cover):
            if f is not None:
                d = f.steer(x, y)
                if d is not None:
                    pts.append((x + d[0] * 12.0, y + d[1] * 12.0))
        if self.dest is not None and 1.0 < math.dist(self.dest, (x, y)) < 20.0:
            pts.append(self.dest)
        return pts

    def f_cross_at(self, X: np.ndarray, Y: np.ndarray, srcs, los: Optional[np.ndarray]):
        """Returns (f_cross, per-source pressure matrix)."""
        P = self.P
        n = len(X)
        press = np.zeros((len(srcs), n))
        for s, th in enumerate(srcs):
            d = np.hypot(X - th.pos[0], Y - th.pos[1])
            thr = np.array([weapon_danger(th.weapons, float(v)) for v in d])
            psi = P["psi"] if th.id < 100 else 1.0
            decay = math.exp(-max(0.0, self.t - th.seen) / P["decay_s"]) * th.conf
            v = thr * psi * decay / ((d / P["d_ref"]) ** 2 + P["eps"])
            if los is not None:
                keep = P["los_keep_mortar"] if "mortar" in th.weapons else P["los_keep"]
                v = v * np.where(los[s], 1.0, keep)
            press[s] = v
        return np.clip(press.sum(0), 0.0, 1.0), press

    def plan_move(self, obs: dict, locks, incoming, target):
        P = self.P
        me = self.me
        x, y, z = self.pos
        hp = me["hp"] / max(1.0, me["max_hp"])
        W = PHASE_W[self.mode]
        alpha = 1.0 + P["gamma"] * (1.0 - hp) if hp < P["alpha_h"] else 1.0
        beta = hp
        on_roof = z > self.world.ground_at(x, y) + 2.5

        # stuck detection
        if self.last_pos is not None and self.dest is not None and math.dist(self.dest, (x, y)) > 2.0 \
                and math.dist(self.last_pos, (x, y)) < 0.25 and me.get("grounded", True):
            self.stuck += 0.1
        else:
            self.stuck = max(0.0, self.stuck - 0.05)
        moved = self.last_pos is not None and math.dist(self.last_pos, (x, y)) > 0.3
        self.still_s = 0.0 if moved else self.still_s + 0.1
        self.last_pos = (x, y)
        if self.stuck > 1.2:
            a = self.rng.uniform(0, 2 * math.pi)
            self.unstick_dir = (math.cos(a), math.sin(a))
            self.unstick_until = self.t + 0.8
            self.stuck = 0.0
        if self.t < self.unstick_until:
            return self.unstick_dir, me["energy"] > 20, bool(me.get("grounded"))

        enemies = self.enemies()
        dmeE = {th.id: math.dist(th.pos[:2], (x, y)) for th in enemies}
        danger_srcs = sorted(enemies, key=lambda th: -weapon_danger(th.weapons, dmeE[th.id]) * th.conf
                             / ((dmeE[th.id] / P["d_ref"]) ** 2 + P["eps"]))
        srcs = [th for th in danger_srcs if weapon_danger(th.weapons, dmeE[th.id]) > 0.02][:4]
        if time.perf_counter() < self.deadline:
            self.update_cover([th for th in srcs if th.conf >= 0.5][:2])

        pts = self.candidates()
        n = len(pts)
        C = np.array(pts)
        X, Y = C[:, 0], C[:, 1]
        dist = np.hypot(X - x, Y - y)
        ux = np.where(dist > 0.1, (X - x) / np.maximum(dist, 1e-6), 0.0)
        uy = np.where(dist > 0.1, (Y - y) / np.maximum(dist, 1e-6), 0.0)
        Jc, Ic = self._cells(X, Y)

        # ---------------------------------------------------------- hard filter
        veto = np.zeros(n, dtype=bool)
        if not on_roof:
            for k in range(1, n):
                veto[k] = not self.nav.clear_line(x, y, X[k], Y[k])
        zn = self.zone
        zc, zr = zn["center"], float(zn["radius"])
        nc, nr = zn["next_center"], float(zn["next_radius"])
        dc = np.hypot(X - zc[0], Y - zc[1])
        dn = np.hypot(X - nc[0], Y - nc[1])
        sr = self.safe_radius(nr)
        fd = self._field_dists(self.field, X, Y, np.maximum(0.0, dn - sr))
        urg = self.zone_urgency()
        inside = dc <= zr
        if (inside & ~veto).any():
            veto |= ~inside
        if not me.get("in_zone", True) or urg >= 1.0:
            better = fd < fd[0] - 1.0
            if (better & ~veto).any():
                veto |= ~better

        # ---------------------------------------------------------- f_zone
        m = min(40.0, max(8.0, 0.2 * zr))
        fz_cur = np.minimum(1.0, np.maximum(0.0, (dc - (zr - m)) / m) ** 2)
        fz_next = 0.6 * np.minimum(1.0, (fd / 40.0) ** 2) + 0.4 * np.minimum(1.0, fd / 200.0)
        f_zone = np.maximum(fz_cur, urg * fz_next)

        # ---------------------------------------------------------- f_cross and f_cover
        los = None
        if srcs and time.perf_counter() < self.deadline:
            gz = np.full(n, z) if on_roof else np.array([self.world.ground_at(px, py) for px, py in pts])
            eh = self.mech.get("eye_height", 8.0)
            A = np.repeat([th.pos + np.array([0.0, 0.0, eh]) for th in srcs], n, axis=0)
            B = np.stack([np.tile(X, len(srcs)), np.tile(Y, len(srcs)), np.tile(gz + 5.5, len(srcs))], axis=1)
            los = self.world.los_batch(A, B, step=2.0).reshape(len(srcs), n)
        f_cross, press = self.f_cross_at(X, Y, srcs, los)
        canopy = self.nav.canopy[Jc, Ic]
        f_cross = f_cross * (1.0 - 0.3 * canopy)
        static_cover = 0.6 * (1.0 - self.nav.openness[Jc, Ic]) + 0.4 * canopy
        if srcs and los is not None:
            wts = press.max(1) + 1e-6
            blocked = (~los).astype(float)
            f_cover = 0.8 * (wts @ blocked) / wts.sum() + 0.2 * static_cover
        else:
            f_cover = 0.5 * static_cover
        kinds = [w["kind"] if w else None for w in me["weapons"]]
        if ("mortar" in kinds or "laser" in kinds) and not on_roof:
            g0 = self.world.ground_at(x, y)
            gz2 = np.array([self.world.ground_at(px, py) for px, py in pts])
            f_cover = np.clip(f_cover + 0.15 * np.clip((gz2 - g0) / 5.0, 0.0, 1.0), 0.0, 1.0)

        # ---------------------------------------------------------- f_aoe
        f_aoe = np.zeros(n)
        spd = self.mech.get("boost_speed", 19.5) if me["energy"] > 10 else self.mech.get("walk_speed", 11.7)
        r = me["stats"].get("radius", 2.5)
        for kind, ppos, pvel, tt in incoming:
            if kind == "mortar":
                reach_d = np.minimum(dist, spd * max(0.0, min(tt, 3.0) - 0.15))
                dd = np.hypot(x + ux * reach_d - ppos[0], y + uy * reach_d - ppos[1])
                R = (self.cfg.get("weapons") or {}).get("mortar", {}).get("splash_radius", 7.0) + r + 2.0
                f_aoe = np.maximum(f_aoe, np.clip(2.0 - dd / R, 0.0, 1.0))
            else:
                best = np.full(n, 1e9)
                for tau in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
                    bx, by = ppos[0] + pvel[0] * tau, ppos[1] + pvel[1] * tau
                    mv = np.minimum(dist, spd * max(0.0, tau - 0.1))
                    best = np.minimum(best, np.hypot(x + ux * mv - bx, y + uy * mv - by))
                f_aoe = np.maximum(f_aoe, (1.0 if kind == "gun" else 0.5) * np.clip(1.0 - (best - r) / 2.0, 0.0, 1.0))
        for th in locks:
            idx = next((s for s, sth in enumerate(srcs) if sth is th), None)
            exposed = los[idx].astype(float) if (los is not None and idx is not None) else np.ones(n)
            bx, by = x - th.pos[0], y - th.pos[1]
            L = math.hypot(bx, by) or 1.0
            lat = np.abs((X - th.pos[0]) * by - (Y - th.pos[1]) * bx) / L
            f_aoe = np.maximum(f_aoe, exposed * (0.5 + 0.5 * np.clip(1.0 - lat / 8.0, 0.0, 1.0)))
        for bx, by, bt, k, w in self.blobs:
            if k in COMBAT_SOUNDS and self.t - bt < 3.0:
                db = np.hypot(X - bx, Y - by)
                f_aoe = np.maximum(f_aoe, 0.5 * w * np.clip(1.0 - db / 40.0, 0.0, 1.0))

        # ---------------------------------------------------------- f_trap
        near = any(d < 90.0 for d in dmeE.values())
        f_trap = (1.0 - self.escape[Jc, Ic]) * (1.0 if near else 0.3)

        # ---------------------------------------------------------- f_kin
        vel = np.array(me.get("vel", (0.0, 0.0, 0.0))[:2], dtype=float)
        vs = float(np.hypot(*vel))
        shooter_near = any(self.aims_at_me(th) and dmeE[th.id] < 150 for th in enemies) or bool(locks)
        turn = np.zeros(n)
        if vs > 2.0 and shooter_near:
            cosang = (ux * vel[0] + uy * vel[1]) / vs
            turn = np.clip((-cosang) / 1.0, 0.0, 1.0) * (dist > 0.1)
        still = np.zeros(n)
        if shooter_near or self.t - self.last_hit_t < 3.0:
            still[dist < 1.5] = 1.0
        elif self.still_s > 3.0:
            still[dist < 1.5] = min(1.0, (self.still_s - 3.0) / 5.0)
        repeat = np.zeros(n)
        if self.heading is not None and self.straight_s > 1.5 and shooter_near:
            repeat = np.clip((ux * self.heading[0] + uy * self.heading[1] - 0.9) / 0.1, 0.0, 1.0)
        f_kin = np.clip(0.45 * turn + 0.4 * still + 0.15 * repeat, 0.0, 1.0)

        # ---------------------------------------------------------- f_dist
        f_dist = np.zeros(n)
        if target is not None:
            th = target[0]
            ready = [w for w in me["weapons"] if w]
            best_k = None
            for k in ("lightning", "gun", "laser", "mortar"):
                if k in kinds and (k != "lightning" or me["energy"] >= 10):
                    best_k = k
                    break
            if best_k is not None:
                ropt, sig = R_OPT[best_k]
                w = next(w for w in ready if w["kind"] == best_k)
                if w.get("reload", 0) > 0 or w.get("cooldown", 0) > 0.5:
                    ropt += R_COOL
                dt = np.hypot(X - th.pos[0], Y - th.pos[1])
                f_dist = np.exp(-((dt - ropt) ** 2) / (2 * sig * sig))

        # ---------------------------------------------------------- f_loot
        f_loot = np.zeros(n)
        objs = self.loot_objects(obs, kinds)
        if objs:
            O = np.array([(o[0], o[1]) for o in objs])
            oc, _ = self.f_cross_at(O[:, 0], O[:, 1], srcs, None)
            for (ox, oy, need), c in zip(objs, oc):
                do = np.hypot(X - ox, Y - oy)
                v = need * (0.5 + np.arctan((P["d_loot"] - do) / P["b_loot"]) / math.pi) * (1.0 - c)
                f_loot = np.maximum(f_loot, v)

        # ---------------------------------------------------------- f_space
        mechs = [th for th in enemies if th.id < 100]
        if mechs:
            M = np.array([th.pos[:2] for th in mechs])
            dmin = np.min(np.hypot(X[:, None] - M[None, :, 0], Y[:, None] - M[None, :, 1]), axis=1)
            f_vor = np.clip(dmin / 150.0, 0.0, 1.0)
        else:
            f_vor = np.full(n, 0.5)
        f_per = np.where((self.alive > 10) & (dn <= sr), np.clip(dn / max(1.0, 0.6 * nr), 0.0, 1.0), 0.0)
        ang = (np.rint(np.arctan2(uy, ux) * (NDIR / (2 * math.pi))).astype(np.int64)) % NDIR
        stale = np.minimum(self.t - self.look_age[ang], 8.0) / 8.0 * (dist > 0.1)
        f_space = 0.5 * f_vor + 0.3 * f_per + 0.2 * stale

        U = (- W["zone"] * f_zone - W["cross"] * alpha * f_cross - W["aoe"] * f_aoe
             - W["trap"] * f_trap - W["kin"] * f_kin + W["cover"] * alpha * f_cover
             + W["dist"] * f_dist + W["loot"] * beta * f_loot + W["space"] * f_space)
        U[veto] = -1_000_000.0

        best = int(np.argmax(U))
        tx, ty = pts[best]
        self.dest = (tx, ty)
        if on_roof and best == 0:
            c = self.nav.nearest_walk(x, y)
            if c is not None:
                tx, ty = self.nav.center(*c)
        dx, dy = tx - x, ty - y
        L = math.hypot(dx, dy)
        if L < 0.8:
            self.straight_s = 0.0
            return (0.0, 0.0), False, False
        h = (dx / L, dy / L)
        if self.heading is not None and h[0] * self.heading[0] + h[1] * self.heading[1] > 0.95:
            self.straight_s += 0.1
        else:
            self.straight_s = 0.0
        self.heading = h
        mag = 1.0 if L >= 3.0 else max(0.3, L / 3.0)
        threatened = bool(locks or incoming or shooter_near)
        boost = me["energy"] > P["energy_reserve"] and (
            threatened or urg >= 1.0 or not me.get("in_zone", True))
        return (h[0] * mag, h[1] * mag), boost, False

    # ================================================================== loot
    def loot_objects(self, obs: dict, kinds) -> List[Tuple[float, float, float]]:
        """O_t with need(o, inventory) in [0, 1]."""
        x, y, z = self.pos
        out = []
        cubes = self.me.get("cubes", 0)
        for c in obs.get("cubes", []):
            p = c["pos"]
            if math.dist(p[:2], (x, y)) < 120 and abs(p[2] - z) < 3.5:
                out.append((p[0], p[1], 0.9 if cubes < 10 else 0.15))
        if None in kinds:
            for p in obs.get("pickups", []):
                if p["kind"] in kinds or p["kind"] in ("mortar", "gun"):
                    continue
                q = p["pos"]
                if math.dist(q[:2], (x, y)) < 120 and abs(q[2] - z) < 3.5:
                    out.append((q[0], q[1], 1.0 if p["kind"] == "lightning" else 0.85))
        for d in obs.get("drops", []):
            h = d.get("holder")
            if (h is not None and h != self.me["id"]) or d.get("lands_in", 99.0) > 6.0:
                continue
            q = d["pos"]
            if math.dist(q[:2], (x, y)) < 120 and self.time_to_edge(q[0], q[1]) > 14.0:
                out.append((q[0], q[1], 1.0))
        return out

    def plan_take(self, obs: dict) -> dict:
        x, y, z = self.pos
        weapons = self.me["weapons"]
        kinds = [w["kind"] if w else None for w in weapons]
        for slot, k in enumerate(kinds):
            if k == "mortar" and not (weapons[slot].get("charge", 0) > 0):
                return {"drop": slot}
        if None not in kinds:
            return {}
        reach = (self.cfg.get("loot") or {}).get("pickup_radius", 4.0) - 0.3
        best = None
        for p in obs.get("pickups", []):
            if p["kind"] in kinds or p["kind"] in ("mortar", "gun"):
                continue
            d = math.dist(p["pos"][:2], (x, y))
            if d <= reach and abs(p["pos"][2] - z) < 3.8:
                pr = 0 if p["kind"] == "lightning" else 1
                if best is None or pr < best[0]:
                    best = (pr, p["id"])
        return {"take": best[1]} if best else {}

    # ================================================================== aim / fire
    def plan_aim_fire(self, locks, target):
        me = self.me
        x, y, z = self.pos
        t = self.t
        eye = (x, y, z + me["stats"].get("muzzle_height", self.mech.get("muzzle_height", 7.0)))
        weapons = me["weapons"]
        kinds = [w["kind"] if w else None for w in weapons]
        if target is not None:
            th, s, clear = target
            d = math.dist(th.pos[:2], (x, y))
            short = th.charging and "lightning" in kinds and d <= 44
            if (s >= self.P["fire_min"] and clear) or short:
                return self.weapon_choice(th, weapons, kinds, eye, d, short)
            return list(yaw_pitch(eye, (th.pos[0], th.pos[1], th.pos[2] + 4.5))), [False, False]
        # no firing target: keep the front arc toward the most dangerous known enemy
        enemies = self.enemies()
        if locks:
            th = min(locks, key=lambda l: math.dist(l.pos[:2], self.pos[:2]))
            return list(yaw_pitch(eye, (th.pos[0], th.pos[1], th.pos[2] + 4.5))), [False, False]
        if enemies:
            th = max(enemies, key=lambda th: th.conf * (weapon_danger(th.weapons, math.dist(th.pos[:2], (x, y))) + 0.05)
                     * math.exp(-(t - th.seen) / self.P["decay_s"]))
            if t - th.seen < 4.0:
                return list(yaw_pitch(eye, (th.pos[0], th.pos[1], th.pos[2] + 4.5))), [False, False]
        if self.last_dmg_bearing and t - self.last_dmg_bearing[1] < 3.0:
            return [self.last_dmg_bearing[0], 0.0], [False, False]
        if self.blobs:
            b = max(self.blobs, key=lambda b: b[4] / (1.0 + t - b[2]))
            if t - b[2] < 4.0:
                return [math.atan2(b[1] - y, b[0] - x), 0.0], [False, False]
        return self.scan(me), [False, False]

    def weapon_choice(self, th: Threat, weapons, kinds, eye, d: float, short: bool):
        fire = [False, False]
        centre = (th.pos[0], th.pos[1], th.pos[2] + (1.75 if th.id >= 100 else 4.5))
        me = self.me
        aim = list(yaw_pitch(eye, centre))
        if "lightning" in kinds and d <= 44 and me["energy"] >= 3:
            s = kinds.index("lightning")
            cone = math.radians(weapons[s]["stats"].get("cone_deg", 30)) / 2
            fire[s] = abs(wrap(aim[0] - me["aim"][0])) < cone * 0.8
            return aim, fire
        if short:
            return aim, fire
        if "laser" in kinds and d <= 280:
            s = kinds.index("laser")
            w = weapons[s]
            err = abs(wrap(aim[0] - me["aim"][0])) + abs(aim[1] - me["aim"][1])
            full = w.get("charge", 0) >= w["stats"].get("charge", 0.75) * 0.97
            if w.get("cooldown", 0) <= 0.05 or w.get("charge", 0) > 0:
                fire[s] = not (full and err < math.atan2(2.0, max(d, 1.0)))
                return aim, fire
        if "gun" in kinds:
            s = kinds.index("gun")
            w = weapons[s]
            p = lead(eye, centre, th.vel, w["stats"].get("speed", 252.0), 1.0)
            aim = list(yaw_pitch(eye, p))
            err = abs(wrap(aim[0] - me["aim"][0])) + abs(aim[1] - me["aim"][1])
            fire[s] = err < max(0.01, math.atan2(2.2, max(d, 1.0))) and w.get("ammo", 1) > 0 \
                and d <= w["stats"].get("range", 230.0)
        return aim, fire

    def scan(self, me: dict) -> List[float]:
        t = self.t
        if self.scan_yaw is None or t > self.scan_t:
            J, I = self.nav.cell(self.pos[0], self.pos[1])
            free = self.nav.ray_free[:, J, I]
            stale = np.minimum(t - self.look_age, 8.0)
            k = int(np.argmax(free * (0.3 + stale)))
            self.scan_yaw = wrap(2 * math.pi * k / NDIR)
            self.scan_t = t + 1.1
        yaw = me["aim"][0]
        for k in range(NDIR):
            if abs(wrap(2 * math.pi * k / NDIR - yaw)) < math.radians(55):
                self.look_age[k] = t
        return [self.scan_yaw, 0.0]

    # ================================================================== upgrades
    def pick_upgrade(self, offer: dict) -> int:
        ids = [c["id"] for c in offer["choices"]]
        rank = {u: n for n, u in enumerate(UPGRADE_ORDER)}
        return min(range(len(ids)), key=lambda i: rank.get(ids[i], 99))
