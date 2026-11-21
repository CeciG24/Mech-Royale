"""Passive survival bot for mechfight/1.

Survival beats kills (score = 16 - place), and once the zone reaches radius 0 everyone burns at
the last phase's dps, so the bot spends the match staying unseen, out of lines of fire, inside
the zone, and at full HP. Each decision scores a ring of candidate points around the mech:

    H(p) = - w_zone*zone(p) - w_threat*danger(p) - w_proj*incoming(p) + w_cover*cover(p)
           + w_loot*loot(p) + jink(p) + stay(p) - w_move*|p - me|

and walks toward the best one. The torso faces the most dangerous known enemy (front arc, so
hits are not x1.5) or scans. It only fires to defend itself, at the very end, or to short a laser.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from brain.aim import lead, mortar_pitch, wrap, yaw_pitch
from brain.nav import NavField, NavGrid
from brain.world import World

DEFAULTS = {
    "safe_frac": 0.72,       # aim to stand inside next circle * this
    "ring_lo": 0.35,         # mild preference to stay off the next circle's centre (where bots converge)
    "w_ring": 2.0,
    "w_zone": 0.35,          # per metre of path still to walk to the safe disk, times urgency
    "w_out": 30.0,           # per metre outside (or near the edge of) the current circle
    "w_threat": 40.0,
    "w_proj": 60.0,
    "w_cover": 4.0,
    "w_loot": 6.0,
    "w_move": 0.12,
    "w_stay": 3.0,
    "w_jink": 14.0,
    "hide_hp": 0.7,          # below this fraction of max HP, sit in cover and regenerate
    "fight_close": 38.0,     # always shoot back at visible mechs closer than this
    "sneak_alive": 7,        # with this many mechs alive or fewer, move silently when unseen
    "energy_reserve": 30.0,
    "flip_s": 0.35,
    "loot_radius": 35.0,
    "zone_margin": 10.0,
    "w_cpath": 0.9,          # per metre of walk to the nearest cell shielded from the attackers
    "cover_far": 20.0,       # shoot back when cover is farther than this
    "farm_npc": False,       # attack idle NPC drones for cubes
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
    __slots__ = ("id", "pos", "vel", "aim", "weapons", "charging", "hp", "seen", "est", "conf")

    def __init__(self, tid: int):
        self.id = tid
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.aim: Optional[float] = None
        self.weapons: set = {"gun"}
        self.charging = False
        self.hp = 1400.0
        self.seen = -1e9
        self.est = True
        self.conf = 0.0


def weapon_danger(kinds, d: float) -> float:
    s = 0.0
    if "npc" in kinds:
        return 0.3 if d <= 50 else 0.12 if d <= 95 else 0.0
    if "laser" in kinds and d <= 295:
        s = max(s, min(1.0, (300.0 - d) / 130.0))
    if "gun" in kinds and d <= 235:
        s = max(s, 0.65 * (1.15 - d / 235))
    if "lightning" in kinds and d <= 55:
        s = max(s, 1.1)
    if "mortar" in kinds and 12 <= d <= 240:
        s = max(s, 0.6)
    return s


class Agent:
    name = "survivor"

    def __init__(self, hello: dict):
        self.P = _params()
        self.cfg = hello["config"]
        self.rng = random.Random(hello.get("seed", 0))
        mp = hello["map"]
        self.world = World(mp["dir"], mp.get("files"))
        self.nav = NavGrid(self.world, self.cfg["mech"])
        self.field: Optional[NavField] = None
        self.field_key = None
        self.pending: Optional[Tuple[tuple, NavField]] = None
        self.threats: Dict[int, Threat] = {}
        self.blobs: List[Tuple[float, float, float, str, float]] = []
        self.last_hit_t = -1e9
        self.last_dmg_bearing: Optional[Tuple[float, float]] = None
        self.target: Optional[Tuple[float, float]] = None
        self.flip = 1.0
        self.flip_t = 0.0
        self.scan_yaw: Optional[float] = None
        self.scan_t = 0.0
        self.look_age = np.zeros(NDIR)
        self.last_pos: Optional[Tuple[float, float]] = None
        self.stuck = 0.0
        self.unstick_until = -1.0
        self.unstick_dir = (1.0, 0.0)
        self.mode = "idle"
        self.mech = self.cfg["mech"]
        self.attackers: Dict[int, float] = {}
        self.cover: Optional[NavField] = None
        self.cover_key: Optional[Tuple[float, float, float, float, float]] = None

    # ================================================================== main
    def act(self, obs: dict) -> dict:
        t0 = time.perf_counter()
        self.t = float(obs["t"])
        me = obs["self"]
        self.me = me
        self.pos = np.array(me["pos"], dtype=float)
        self.zone = obs["zone"]
        a: dict = {}
        if obs.get("offer"):
            a["pick"] = self.pick_upgrade(obs["offer"])
        self.update_zone(obs, t0)
        self.update_threats(obs)
        locks = self.laser_locks(obs)
        incoming = self.incoming(obs)
        self.choose_mode(obs, locks, incoming)
        move, boost, jump = self.plan_move(obs, locks, incoming, t0)
        aim, fire = self.plan_aim_fire(obs, locks)
        loot = self.plan_take(obs)
        a.update(loot)
        a.update(move=[round(move[0], 4), round(move[1], 4)], boost=boost, jump=jump,
                 aim=[round(aim[0], 4), round(aim[1], 4)], fire=fire)
        return a

    # ================================================================== zone
    def safe_radius(self, nr: float) -> float:
        return max(4.0, nr * self.P["safe_frac"])

    def update_zone(self, obs: dict, t0: float) -> None:
        z = obs["zone"]
        nc, nr = z["next_center"], z["next_radius"]
        key = (round(nc[0], 1), round(nc[1], 1), round(nr, 1))
        if self.field_key != key and (self.pending is None or self.pending[0] != key):
            goal = self.nav.circle_mask(nc[0], nc[1], self.safe_radius(nr))
            if not (goal & self.nav.walk).any():
                goal = self.nav.circle_mask(nc[0], nc[1], max(12.0, nr))
            self.pending = (key, NavField(self.nav, goal))
        if self.pending is not None:
            budget = 0.012 if self.field is not None else 0.025
            if self.pending[1].run(budget_s=budget):
                self.field_key, self.field = self.pending
                self.pending = None

    def field_dist(self, x: float, y: float) -> float:
        if self.field is None:
            z = self.zone
            return max(0.0, math.dist((x, y), z["next_center"]) - self.safe_radius(z["next_radius"]))
        J, I = self.nav.cell(x, y)
        d = self.field.at(J, I)
        if math.isfinite(d):
            cx, cy = self.nav.center(J, I)
            return d + math.hypot(cx - x, cy - y)
        return self.field.dist(x, y)

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
            th.seen, th.est, th.conf = t, False, 1.0
            self.threats[m["id"]] = th
            vis[m["id"]] = th
        for b in obs.get("beams", []):
            by = b.get("by")
            if by is None or by in vis or by >= 100:
                continue
            th = self.threats.get(by) or Threat(by)
            p0 = b["path"][0]
            th.pos = np.array([p0[0], p0[1], p0[2] - self.mech["muzzle_height"]])
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
                        d = 30.0 if by >= 100 else EST_DIST.get(e["weapon"], 100.0)
                        th.pos = self.pos + np.array([d * math.cos(e["bearing"]), d * math.sin(e["bearing"]), 0.0])
                        th.est, th.conf, th.seen = True, 0.6, t
                        th.aim = wrap(e["bearing"] + math.pi)
                    th.weapons.add("npc" if by >= 100 else e["weapon"])
                    self.threats[by] = th
        for th in list(self.threats.values()):
            age = t - th.seen
            if age > (6.0 if not th.est else 8.0):
                del self.threats[th.id]
            elif th.id not in vis and not th.est:
                th.conf = max(0.3, 0.9 - 0.1 * age)
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
            k = s["kind"]
            w = COMBAT_SOUNDS.get(k, MOVE_SOUNDS.get(k, 0.0))
            if w <= 0:
                continue
            d = BAND_DIST.get(s["dist"], 200.0)
            if k in MOVE_SOUNDS:
                d = 25.0
            x = self.pos[0] + d * math.cos(s["bearing"])
            y = self.pos[1] + d * math.sin(s["bearing"])
            self.blobs.append((x, y, t - s.get("age", 0.0), k, w))

    def top_threats(self, k: int = 3) -> List[Threat]:
        p = self.pos
        scored = []
        for th in self.threats.values():
            d = math.dist(th.pos[:2], p[:2])
            s = th.conf * (weapon_danger(th.weapons, d) + 0.05) * (1.0 + 30.0 / max(d, 15.0))
            if th.charging:
                s *= 2.0
            scored.append((s, th))
        scored.sort(key=lambda x: -x[0])
        return [th for _, th in scored[:k]]

    def shooters(self) -> List[Threat]:
        """Mechs firing at us now: they hit us in the last 1.5 s, or we see them aiming at us."""
        out = []
        for th in self.threats.values():
            if th.id < 0 or th.id >= 100:
                continue
            d = math.dist(th.pos[:2], self.pos[:2])
            hit = self.t - self.attackers.get(th.id, -1e9) < 1.5
            aimed = False
            if not th.est and th.aim is not None and self.t - th.seen < 0.15 and d < 235:
                b = math.atan2(self.pos[1] - th.pos[1], self.pos[0] - th.pos[0])
                aimed = abs(wrap(th.aim - b)) < math.atan2(5.0, max(d, 1.0)) + 0.04
            if hit or aimed:
                out.append(th)
        return out

    def update_cover(self, srcs: List[Threat]) -> None:
        """Local field (≈70 m around us): walking distance to cells an eye-height building
        shields from every source."""
        if not srcs:
            self.cover = None
            return
        x, y = float(self.pos[0]), float(self.pos[1])
        p = srcs[0].pos
        key = (x, y, float(p[0]), float(p[1]), self.t)
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
            dist = np.hypot(dx, dy)
            idx = np.rint(np.arctan2(dy, dx) * (NDIR / (2 * math.pi))).astype(np.int64) % NDIR
            free = np.take_along_axis(rf, idx[None], 0)[0]
            covered &= free < dist - nav.res
        f = NavField(nav, covered, window=(j0, j1, i0, i1))
        f.run(max_iters=40)
        self.cover = f
        self.cover_key = key

    def cover_dist(self, x: float, y: float) -> float:
        if self.cover is None:
            return math.inf
        J, I = self.nav.cell(x, y)
        return self.cover.at(J, I)

    def laser_locks(self, obs: dict) -> List[Threat]:
        out = []
        for th in self.threats.values():
            if not th.charging or th.est or th.aim is None:
                continue
            d = math.dist(th.pos[:2], self.pos[:2])
            bearing = math.atan2(self.pos[1] - th.pos[1], self.pos[0] - th.pos[0])
            if abs(wrap(th.aim - bearing)) < math.atan2(6.0, max(d, 1.0)) + 0.05:
                out.append(th)
        for s in obs.get("sounds", []):
            if s["kind"] == "laser_charge" and s["dist"] != "far":
                th = Threat(-1)
                d = BAND_DIST[s["dist"]]
                th.pos = self.pos + np.array([d * math.cos(s["bearing"]), d * math.sin(s["bearing"]), 0.0])
                th.weapons, th.charging, th.est, th.conf = {"laser"}, True, True, 0.5
                out.append(th)
        return out

    def incoming(self, obs: dict) -> List[Tuple[str, np.ndarray, np.ndarray, float]]:
        """Enemy projectiles that might reach us: (kind, pos, vel, landing time for shells)."""
        out = []
        g = self.cfg["weapons"]["mortar"]["gravity"]
        for p in obs.get("projectiles", []):
            pos = np.array(p["pos"], dtype=float)
            vel = np.array(p["vel"], dtype=float)
            rel = self.pos - pos
            if p["kind"] == "mortar":
                land_t = None
                q, v = pos.copy(), vel.copy()
                for k in range(1, 80):
                    v[2] -= g * 0.1
                    q = q + v * 0.1
                    if q[2] <= self.world.roof_top(q[0], q[1]) + 0.5:
                        land_t = k * 0.1
                        break
                if land_t is not None and math.dist(q[:2], self.pos[:2]) < 40:
                    out.append(("mortar", q, vel, land_t))
            else:
                sp = float(np.linalg.norm(vel[:2])) or 1.0
                along = float(rel[:2] @ vel[:2]) / sp
                if along < -5 or along > sp * 0.8:
                    continue
                perp = abs(float(rel[0] * vel[1] - rel[1] * vel[0])) / sp
                if perp < 12:
                    out.append((p["kind"], pos, vel, along / sp))
        return out

    # ================================================================== modes
    def choose_mode(self, obs: dict, locks, incoming) -> None:
        me = self.me
        hpf = me["hp"] / max(1.0, me["max_hp"])
        recent_hit = self.t - self.last_hit_t < 2.5
        visible = [th for th in self.threats.values() if not th.est and self.t - th.seen < 0.15 and th.id < 100]
        urg = self.zone_urgency()
        if not me.get("in_zone", True) or urg >= 1.0:
            self.mode = "rotate"
        elif locks or incoming or recent_hit:
            self.mode = "evade"
        elif hpf < self.P["hide_hp"] and not visible:
            self.mode = "hide"
        elif visible:
            self.mode = "avoid"
        else:
            self.mode = "idle"

    def zone_urgency(self) -> float:
        x, y = self.pos[0], self.pos[1]
        z = self.zone
        if math.dist((x, y), z["center"]) > z["radius"]:
            return 1.0
        if math.dist((x, y), z["next_center"]) <= self.safe_radius(z["next_radius"]):
            return 0.0
        need = self.field_dist(x, y) / (self.mech["walk_speed"] * 0.8) + self.P["zone_margin"]
        left = self.time_to_edge(x, y)
        if left == math.inf:
            to_shrink = z["shrink_start"] - self.t
            return 0.15 if to_shrink > need + 20 else 0.5
        slack = left - need
        if slack <= 0:
            return 1.0
        return max(0.15, min(1.0, 1.0 - slack / 30.0))

    # ================================================================== movement
    def candidates(self) -> List[Tuple[float, float]]:
        x, y = self.pos[0], self.pos[1]
        pts = [(x, y)]
        for r in (7.0, 15.0):
            for k in range(NDIR):
                a = 2 * math.pi * k / NDIR
                pts.append((x + r * math.cos(a), y + r * math.sin(a)))
        for f in (self.field, self.cover):
            if f is not None:
                d = f.steer(x, y)
                if d is not None:
                    pts.append((x + d[0] * 10.0, y + d[1] * 10.0))
        if self.target is not None and math.dist(self.target, (x, y)) > 1.0:
            pts.append(self.target)
        return pts

    def plan_move(self, obs: dict, locks, incoming, t0: float):
        P = self.P
        me = self.me
        x, y, z = self.pos
        on_roof = z > self.world.ground_at(x, y) + 2.5
        # stuck detection
        if self.last_pos is not None and self.target is not None and math.dist(self.target, (x, y)) > 2.0 \
                and math.dist(self.last_pos, (x, y)) < 0.25 and me.get("grounded", True):
            self.stuck += 0.1
        else:
            self.stuck = max(0.0, self.stuck - 0.05)
        self.last_pos = (x, y)
        if self.stuck > 1.2:
            a = self.rng.uniform(0, 2 * math.pi)
            self.unstick_dir = (math.cos(a), math.sin(a))
            self.unstick_until = self.t + 0.8
            self.stuck = 0.0
        if self.t < self.unstick_until:
            return self.unstick_dir, me["energy"] > 20, bool(me.get("grounded"))

        shooters = self.shooters()
        cover_srcs = []
        if self.mode in ("evade", "avoid", "hide"):
            for th in self.top_threats(4):
                if th.id < 0 or th.conf < 0.5:
                    continue
                if th.id >= 100 and self.t - self.attackers.get(th.id, -1e9) > 3.0:
                    continue
                if weapon_danger(th.weapons, math.dist(th.pos[:2], self.pos[:2])) > 0.1:
                    cover_srcs.append(th)
            cover_srcs = cover_srcs[:2]
        self.update_cover(cover_srcs)

        pts = self.candidates()
        n = len(pts)
        C = np.array(pts)
        reach = np.ones(n, dtype=bool)
        if not on_roof:
            for k in range(1, n):
                reach[k] = self.nav.clear_line(x, y, C[k, 0], C[k, 1])
        dist = np.hypot(C[:, 0] - x, C[:, 1] - y)
        score = -P["w_move"] * dist
        score[~reach] -= 1e6

        # zone
        urg = self.zone_urgency()
        zn = self.zone
        nc, nr = zn["next_center"], zn["next_radius"]
        sr = self.safe_radius(nr)
        zc, zr = zn["center"], zn["radius"]
        fd = np.array([self.field_dist(px, py) for px, py in pts])
        dn = np.hypot(C[:, 0] - nc[0], C[:, 1] - nc[1])
        dc = np.hypot(C[:, 0] - zc[0], C[:, 1] - zc[1])
        zcost = np.where(dn > sr, fd, 0.0) * (0.15 + urg * 3.0)
        edge = np.maximum(0.0, dc - (zr - 8.0))
        zcost += P["w_out"] / P["w_zone"] * edge * (1.0 + zn.get("dps", 20) / 30.0)
        if nr > 30:
            zcost += P["w_ring"] / P["w_zone"] * np.maximum(0.0, P["ring_lo"] - dn / max(nr, 1.0)) * 10.0
        score -= P["w_zone"] * zcost

        # threats: exposure along lines of fire
        tops = self.top_threats(3)
        blob = None
        if self.blobs:
            bl = max(self.blobs, key=lambda b: b[4] / (1.0 + self.t - b[2]))
            blob = Threat(-2)
            blob.pos = np.array([bl[0], bl[1], z])
            blob.weapons = {"laser", "gun"} if bl[3] not in ("lightning",) else {"lightning", "gun"}
            blob.conf = 0.25 * bl[4]
            blob.est = True
        srcs = tops + ([blob] if blob is not None else []) + [l for l in locks if l.id < 0]
        danger = np.zeros(n)
        if srcs:
            gz = np.array([self.world.ground_at(px, py) for px, py in pts]) if not on_roof else np.full(n, z)
            A, B, meta = [], [], []
            for th in srcs:
                eye = th.pos + np.array([0.0, 0.0, self.mech["eye_height"]])
                for k in range(n):
                    A.append(eye)
                    B.append((C[k, 0], C[k, 1], gz[k] + 5.5))
            los = self.world.los_batch(A, B, step=2.0).reshape(len(srcs), n)
            for s, th in enumerate(srcs):
                dx = C[:, 0] - th.pos[0]
                dy = C[:, 1] - th.pos[1]
                d = np.hypot(dx, dy)
                wd = np.array([weapon_danger(th.weapons, float(v)) for v in d])
                exp = los[s].astype(float)
                if "mortar" in th.weapons:
                    exp = np.maximum(exp, 0.5 * (d < 240))
                seen = np.ones(n)
                if th.aim is not None and not th.est:
                    off = np.abs((np.arctan2(dy, dx) - th.aim + np.pi) % (2 * np.pi) - np.pi)
                    seen = np.where((d <= 45) | (off <= math.radians(70)), 1.0, 0.45)
                prox = 1.0 + 40.0 / np.maximum(d, 15.0)
                lock = 2.5 if th.charging else 1.0
                danger += th.conf * wd * exp * seen * prox * lock
            canopy = np.array([self.nav.canopy[self.nav.cell(px, py)] for px, py in pts])
            danger *= 1.0 - 0.4 * canopy
            score -= P["w_threat"] * danger

        # incoming projectiles
        if incoming:
            spd = self.mech["boost_speed"] if me["energy"] > 10 else self.mech["walk_speed"]
            ux = np.where(dist > 0.1, (C[:, 0] - x) / np.maximum(dist, 1e-6), 0.0)
            uy = np.where(dist > 0.1, (C[:, 1] - y) / np.maximum(dist, 1e-6), 0.0)
            risk = np.zeros(n)
            r = me["stats"].get("radius", 2.5)
            for kind, ppos, pvel, tt in incoming:
                if kind == "mortar":
                    tl = min(tt, 3.0)
                    reach_d = np.minimum(dist, spd * max(0.0, tl - 0.15))
                    fx, fy = x + ux * reach_d, y + uy * reach_d
                    dd = np.hypot(fx - ppos[0], fy - ppos[1])
                    R = self.cfg["weapons"]["mortar"]["splash_radius"] + r + 2.0
                    risk += 3.0 * np.clip(1.0 - dd / R, 0.0, 1.0)
                else:
                    best = np.full(n, 1e9)
                    for tau in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
                        bx, by = ppos[0] + pvel[0] * tau, ppos[1] + pvel[1] * tau
                        mv = np.minimum(dist, spd * max(0.0, tau - 0.1))
                        best = np.minimum(best, np.hypot(x + ux * mv - bx, y + uy * mv - by))
                    risk += (1.0 if kind == "gun" else 0.5) * (best < r + 1.0)
            score -= P["w_proj"] * risk

        # walking distance to cover from the current attackers
        if self.cover is not None:
            cd = np.array([self.cover_dist(px, py) for px, py in pts])
            score -= P["w_cpath"] * np.minimum(cd, 80.0)

        # laser lock or incoming fire: break the line or jink across it
        jinkers = locks or shooters
        if jinkers:
            if self.t > self.flip_t:
                self.flip = -self.flip
                self.flip_t = self.t + P["flip_s"] * self.rng.uniform(0.8, 1.3)
            th = min(jinkers, key=lambda l: math.dist(l.pos[:2], self.pos[:2]))
            bx, by = x - th.pos[0], y - th.pos[1]
            L = math.hypot(bx, by) or 1.0
            px_, py_ = -by / L * self.flip, bx / L * self.flip
            ux = (C[:, 0] - x) / np.maximum(dist, 1e-6)
            uy = (C[:, 1] - y) / np.maximum(dist, 1e-6)
            wj = P["w_jink"] * (1.0 if locks else 0.6)
            score += wj * np.where(dist > 0.1, np.maximum(0.0, ux * px_ + uy * py_), -0.5)

        # cover (static) and loot
        cov = np.array([1.0 - self.nav.openness[self.nav.cell(px, py)] for px, py in pts])
        wcov = P["w_cover"] * (2.0 if self.mode == "hide" else 1.0)
        score += wcov * cov
        loot = self.loot_targets(obs)
        if loot:
            for lx, ly, val in loot:
                d0 = math.hypot(lx - x, ly - y)
                dd = np.hypot(C[:, 0] - lx, C[:, 1] - ly)
                score += P["w_loot"] * val * np.clip((d0 - dd) / max(d0, 1.0), -1.0, 1.0)
        if self.target is not None:
            score += P["w_stay"] * (np.hypot(C[:, 0] - self.target[0], C[:, 1] - self.target[1]) < 3.0)
        if self.mode == "hide":
            score[0] += P["w_stay"]

        best = int(np.argmax(score))
        tx, ty = pts[best]
        self.target = (tx, ty)
        boost = False
        if on_roof and best == 0:
            c = self.nav.nearest_walk(x, y)
            if c is not None:
                tx, ty = self.nav.center(*c)
        if self.mode == "rotate" and self.field is not None and urg >= 1.0:
            d = self.field.steer(x, y)
            if d is not None and score[best] < -1e5 + 1:
                tx, ty = x + d[0] * 10, y + d[1] * 10
        dx, dy = tx - x, ty - y
        L = math.hypot(dx, dy)
        if L < 0.8:
            return (0.0, 0.0), False, False
        mag = 1.0
        if self.mode in ("idle", "hide") and self.sneaky(obs):
            mag = 0.4
        if L < 3.0:
            mag = min(mag, max(0.3, L / 3.0))
        energy = me["energy"]
        if self.mode in ("evade", "rotate") and energy > P["energy_reserve"]:
            boost = bool(locks) or bool(incoming) or bool(shooters) or urg >= 1.0 or not me.get("in_zone", True)
        if self.mode == "avoid" and energy > 70:
            boost = True
        return (dx / L * mag, dy / L * mag), boost, False

    def sneaky(self, obs: dict) -> bool:
        """Walk under step_speed (silent) when someone may be close but has not found us."""
        if obs["alive"] <= self.P["sneak_alive"]:
            return True
        return any(b[3] in MOVE_SOUNDS and self.t - b[2] < 4.0 for b in self.blobs)

    # ================================================================== loot
    def loot_targets(self, obs: dict) -> List[Tuple[float, float, float]]:
        if self.mode not in ("idle", "hide"):
            return []
        if self.t - self.last_hit_t < 5.0:
            return []
        x, y, z = self.pos
        out = []
        cubes = self.me.get("cubes", 0)
        for c in obs.get("cubes", []):
            d = math.dist(c["pos"][:2], (x, y))
            if d < self.P["loot_radius"] and abs(c["pos"][2] - z) < 3.5 and self.nav.clear_line(x, y, c["pos"][0], c["pos"][1]):
                out.append((c["pos"][0], c["pos"][1], 1.0 if cubes < 10 else 0.2))
        weapons = self.me["weapons"]
        held = {w["kind"] for w in weapons if w}
        if None in weapons:
            for p in obs.get("pickups", []):
                if p["kind"] in held or p["kind"] in ("mortar", "gun"):
                    continue
                d = math.dist(p["pos"][:2], (x, y))
                if d < self.P["loot_radius"] + 10 and abs(p["pos"][2] - z) < 3.5 and self.nav.clear_line(x, y, p["pos"][0], p["pos"][1]):
                    out.append((p["pos"][0], p["pos"][1], 1.5 if p["kind"] == "lightning" else 1.0))
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
        reach = self.cfg["loot"]["pickup_radius"] - 0.3
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
    def plan_aim_fire(self, obs: dict, locks):
        me = self.me
        x, y, z = self.pos
        t = self.t
        eye = (x, y, z + me["stats"].get("muzzle_height", self.mech["muzzle_height"]))
        fire = [False, False]
        tops = self.top_threats(1)
        primary = tops[0] if tops else None
        if locks:
            lk = [l for l in locks if l.id >= 0] or locks
            primary = min(lk, key=lambda l: math.dist(l.pos[:2], self.pos[:2]))
        aim = None
        if primary is not None and t - primary.seen < 4.0:
            c = (primary.pos[0], primary.pos[1], primary.pos[2] + 4.5)
            aim = list(yaw_pitch(eye, c))
        elif self.last_dmg_bearing and t - self.last_dmg_bearing[1] < 3.0:
            aim = [self.last_dmg_bearing[0], 0.0]
        elif self.blobs:
            b = max(self.blobs, key=lambda b: b[4] / (1.0 + t - b[2]))
            if t - b[2] < 4.0:
                aim = [math.atan2(b[1] - y, b[0] - x), 0.0]
        if aim is None:
            aim = self.scan(me)
        # defensive fire
        weapons = me["weapons"]
        kinds = [w["kind"] if w else None for w in weapons]
        vis_mechs = [th for th in self.threats.values()
                     if 0 <= th.id < 100 and not th.est and t - th.seen < 0.15]
        target = None
        if vis_mechs:
            th = min(vis_mechs, key=lambda th: math.dist(th.pos[:2], (x, y)))
            if primary is not None and primary in vis_mechs:
                th = primary
            d = math.dist(th.pos[:2], (x, y))
            zone = self.zone
            endgame = zone["radius"] <= 30 or obs["alive"] <= 2 or zone["dps"] >= 160
            under_fire = th in self.shooters()
            far_cover = self.cover_dist(x, y) > self.P["cover_far"]
            shoot = d <= self.P["fight_close"] or endgame or (th.hp < 260 and d < 120) \
                or (under_fire and far_cover and d < 200)
            short = th.charging and "lightning" in kinds and d <= 44
            if shoot or short:
                target = (th, d, short)
        else:
            npcs = [th for th in self.threats.values() if th.id >= 100 and not th.est and t - th.seen < 0.15]
            if npcs:
                th = min(npcs, key=lambda th: math.dist(th.pos[:2], (x, y)))
                d = math.dist(th.pos[:2], (x, y))
                attacking = t - self.attackers.get(th.id, -1e9) < 3.0
                if (attacking or (self.P["farm_npc"] and d < 60)) and d < 120:
                    target = (th, d, False)
        if target is not None:
            th, d, short = target
            fire, aim = self.weapon_choice(th, weapons, kinds, eye, d, short, aim)
        return aim, fire

    def weapon_choice(self, th: Threat, weapons, kinds, eye, d: float, short: bool, aim):
        fire = [False, False]
        centre = (th.pos[0], th.pos[1], th.pos[2] + (1.75 if th.id >= 100 else 4.5))
        me = self.me
        if "lightning" in kinds and d <= 44 and me["energy"] >= 3:
            s = kinds.index("lightning")
            aim = list(yaw_pitch(eye, centre))
            cone = math.radians(weapons[s]["stats"].get("cone_deg", 30)) / 2
            fire[s] = abs(wrap(aim[0] - me["aim"][0])) < cone * 0.8
            return fire, aim
        if short:
            return fire, aim
        if "gun" in kinds:
            s = kinds.index("gun")
            w = weapons[s]
            p = lead(eye, centre, th.vel, w["stats"]["speed"], 1.0)
            aim = list(yaw_pitch(eye, p))
            err = abs(wrap(aim[0] - me["aim"][0])) + abs(aim[1] - me["aim"][1])
            fire[s] = err < max(0.01, math.atan2(2.2, max(d, 1.0))) and w.get("ammo", 1) > 0
            return fire, aim
        if "laser" in kinds:
            s = kinds.index("laser")
            w = weapons[s]
            aim = list(yaw_pitch(eye, centre))
            err = abs(wrap(aim[0] - me["aim"][0])) + abs(aim[1] - me["aim"][1])
            full = w.get("charge", 0) >= w["stats"]["charge"] * 0.97
            fire[s] = not (full and err < math.atan2(2.0, max(d, 1.0)))
            return fire, aim
        return fire, aim

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
            a = 2 * math.pi * k / NDIR
            if abs(wrap(a - yaw)) < math.radians(55):
                self.look_age[k] = t
        return [self.scan_yaw, 0.0]

    # ================================================================== upgrades
    def pick_upgrade(self, offer: dict) -> int:
        ids = [c["id"] for c in offer["choices"]]
        rank = {u: n for n, u in enumerate(UPGRADE_ORDER)}
        return min(range(len(ids)), key=lambda i: rank.get(ids[i], 99))
