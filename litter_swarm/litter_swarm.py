"""
TEB2103 Modelling and Simulation
Agent-Based Swarm for Litter Collection
Authors: Dashita Vadivel (22010193), Eve Dellyienna Wahidi (24000632), Jashbindar

Parts implemented
  A  Environment: 2D park, base station, obstacles, clustered + scattered litter
  B  Robot searching and collection (limited sensing radius, imperfect detection)
  C  Swarm signaling (temporary broadcast that pulls nearby robots in)
  D  Return to base (unload litter, battery management and recharge)

Run:
    python litter_swarm.py (for live animation)
    python litter_swarm.py compare (signaling ON vs OFF experiment (plot))
Requires: numpy, matplotlib
"""
import math
import random
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


# ---------------------------------------------------#
# Configuration: every tunable parameter lives here  #
# ---------------------------------------------------#
@dataclass
class Config:
    # Part A: environment
    width: float = 100.0
    height: float = 100.0
    n_obstacles: int = 10
    n_clusters: int = 3
    cluster_size: int = 12          # litter items per cluster
    cluster_spread: float = 4.0     # std-dev of a cluster (map units)
    n_scattered: int = 20           # scattered litter items
    base_pos: tuple = (50.0, 50.0)
    base_radius: float = 4.0

    # Part B: robots
    n_robots: int = 8
    speed: float = 1.5
    sense_radius: float = 7.0       # limited sensing range
    detect_prob: float = 0.85       # imperfect sensor: chance to spot litter per step
    pickup_radius: float = 1.5
    robot_radius: float = 0.8
    avoid_range: float = 5.0        # start steering away from obstacles/robots

    # Part C: signaling
    signaling: bool = True
    comm_range: float = 30.0        # robots within this range hear the signal
    signal_ttl: int = 80            # signal lifetime (steps)

    # Part D: battery
    battery_max: float = 500.0
    recharge_rate: float = 10.0     # battery gained per step at base
    battery_margin: float = 1.4     # safety factor on the return trip estimate

    seed: int = 1


# ----------------------------------#
# Passive things in the environment #
# ----------------------------------#
class Obstacle:
    def __init__(self, pos, radius):
        self.pos = np.array(pos, dtype=float)
        self.radius = radius


class Litter:
    def __init__(self, uid, pos, clustered):
        self.uid = uid
        self.pos = np.array(pos, dtype=float)
        self.clustered = clustered
        self.collected = False


class Signal:
    """Temporary broadcast left where litter was found (like a pheromone)."""
    _next_id = 0

    def __init__(self, pos, ttl):
        self.uid = Signal._next_id
        Signal._next_id += 1
        self.pos = np.array(pos, dtype=float)
        self.ttl = ttl
        self.max_ttl = ttl


# ----------------------------------------------------------------------------
# The agent (robots & helpers)
# ----------------------------------------------------------------------------
class Robot:
    SEARCH, TO_LITTER, TO_SIGNAL, RETURN, CHARGING, STRANDED = (
        "SEARCH", "TO_LITTER", "TO_SIGNAL", "RETURN", "CHARGING", "STRANDED")

    def __init__(self, uid, model):
        self.uid = uid
        self.model = model
        cfg = model.cfg
        # start near the base
        ang = random.uniform(0, 2 * math.pi)
        self.pos = np.array(cfg.base_pos) + 2.0 * np.array([math.cos(ang), math.sin(ang)])
        self.heading = random.uniform(0, 2 * math.pi)
        self.battery = cfg.battery_max
        self.state = Robot.SEARCH
        self.carrying = 0
        self.target_litter = None
        self.target_signal = None
        self.ignored_signals = set()    # signals already checked (so no endless loops)
        self.delivered = 0

    # ---- helpers ---------------------------------------------------------
    def dist_to(self, p):
        return float(np.linalg.norm(self.pos - p))

    def needs_recharge(self):
        cfg = self.model.cfg
        steps_home = self.dist_to(np.array(cfg.base_pos)) / cfg.speed
        return self.battery < steps_home * cfg.battery_margin + 20

    def blocked(self, p):
        cfg = self.model.cfg
        if not (0 <= p[0] <= cfg.width and 0 <= p[1] <= cfg.height):
            return True
        for ob in self.model.obstacles:
            if np.linalg.norm(p - ob.pos) < ob.radius + cfg.robot_radius:
                return True
        return False

    def repulsion(self):
        """Steering vector pushing away from nearby obstacles and robots."""
        cfg = self.model.cfg
        rep = np.zeros(2)
        for ob in self.model.obstacles:
            v = self.pos - ob.pos
            gap = np.linalg.norm(v) - ob.radius
            if 1e-9 < gap < cfg.avoid_range:
                rep += v / np.linalg.norm(v) * (1 - gap / cfg.avoid_range) * 2.0
        for other in self.model.robots:
            if other is self:
                continue
            v = self.pos - other.pos
            d = np.linalg.norm(v)
            if 1e-9 < d < cfg.avoid_range:
                rep += v / d * (1 - d / cfg.avoid_range) * 1.5
        return rep

    def move(self, direction, max_step=None):
        """Move along `direction` with obstacle/robot avoidance. Tries side-steps if blocked."""
        cfg = self.model.cfg
        step = cfg.speed if max_step is None else min(cfg.speed, max_step)
        d = direction + self.repulsion()
        n = np.linalg.norm(d)
        if n < 1e-9:
            return
        base_angle = math.atan2(d[1], d[0])
        for offset in (0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6):
            a = base_angle + offset
            new = self.pos + step * np.array([math.cos(a), math.sin(a)])
            if not self.blocked(new):
                self.pos = new
                self.heading = a
                return
        self.heading += random.uniform(-math.pi, math.pi)   # stuck: turn randomly

    def move_toward(self, target):
        v = target - self.pos
        dist = np.linalg.norm(v)
        if dist > 1e-9:
            self.move(v / dist, max_step=dist)

    def wander(self):
        self.heading += random.gauss(0, 0.35)
        self.move(np.array([math.cos(self.heading), math.sin(self.heading)]))

    # ---- Part B: sensing -------------------------------------------------
    def detect_litter(self):
        """Return nearest litter in sensing radius (each item seen with prob detect_prob)."""
        cfg = self.model.cfg
        best, best_d = None, 1e9
        for lt in self.model.litter:
            if lt.collected:
                continue
            d = self.dist_to(lt.pos)
            if d <= cfg.sense_radius and d < best_d and random.random() < cfg.detect_prob:
                best, best_d = lt, d
        return best

    # ---- Part C: listening for signals ------------------------------------
    def hear_signal(self):
        cfg = self.model.cfg
        if not cfg.signaling:
            return None
        best, best_d = None, 1e9
        for s in self.model.signals:
            if s.uid in self.ignored_signals:
                continue
            d = self.dist_to(s.pos)
            if d <= cfg.comm_range and d < best_d:
                best, best_d = s, d
        return best

    # ---- main loop: sense -> plan -> act ----------------------------------
    def step(self):
        cfg = self.model.cfg

        if self.state == Robot.STRANDED:
            return

        # Part D: charging at base
        if self.state == Robot.CHARGING:
            self.battery = min(cfg.battery_max, self.battery + cfg.recharge_rate)
            if self.battery >= cfg.battery_max:
                self.state = Robot.SEARCH
            return

        self.battery -= 1
        if self.battery <= 0:
            self.state = Robot.STRANDED
            return

        # Part D: battery check overrides everything except already heading home
        if self.state != Robot.RETURN and self.needs_recharge():
            self.state = Robot.RETURN

        if self.state == Robot.SEARCH:
            lt = self.detect_litter()
            if lt:
                self.target_litter, self.state = lt, Robot.TO_LITTER
            else:
                sig = self.hear_signal()
                if sig:
                    self.target_signal, self.state = sig, Robot.TO_SIGNAL
                else:
                    self.wander()

        elif self.state == Robot.TO_SIGNAL:
            sig = self.target_signal
            lt = self.detect_litter()
            if lt:                                   # found something on the way
                self.target_litter, self.state = lt, Robot.TO_LITTER
            elif sig.ttl <= 0 or self.dist_to(sig.pos) < 2.0:
                self.ignored_signals.add(sig.uid)    # nothing here, give up on this signal
                self.state = Robot.SEARCH
            else:
                self.move_toward(sig.pos)

        elif self.state == Robot.TO_LITTER:
            lt = self.target_litter
            if lt.collected:                         # another robot got it first
                self.state = Robot.SEARCH
            elif self.dist_to(lt.pos) <= cfg.pickup_radius:
                lt.collected = True                  # Part B: remove from environment
                self.carrying += 1
                self.model.emit_signal(lt.pos)       # Part C: broadcast
                self.state = Robot.RETURN            # Part D: carrying -> go home
            else:
                self.move_toward(lt.pos)

        elif self.state == Robot.RETURN:
            base = np.array(cfg.base_pos)
            if self.dist_to(base) <= cfg.base_radius:
                self.model.delivered += self.carrying
                self.delivered += self.carrying
                self.carrying = 0
                self.state = Robot.CHARGING
            else:
                self.move_toward(base)


# ----------------------------------------------------------------------------
# The model (environment + scheduler)
# ----------------------------------------------------------------------------
class LitterModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.t = 0
        self.delivered = 0
        self.obstacles, self.litter, self.signals = [], [], []
        self._build_environment()
        self.robots = [Robot(i, self) for i in range(cfg.n_robots)]
        self.history = []   # litter remaining per step

    # ---- Part A ----------------------------------------------------------
    def _free(self, p, margin=1.5):
        """True if point is inside the map, outside obstacles and away from the base."""
        c = self.cfg
        if not (1 <= p[0] <= c.width - 1 and 1 <= p[1] <= c.height - 1):
            return False
        if np.linalg.norm(p - np.array(c.base_pos)) < c.base_radius + 3:
            return False
        return all(np.linalg.norm(p - ob.pos) > ob.radius + margin for ob in self.obstacles)

    def _build_environment(self):
        c = self.cfg
        # obstacles (trees / benches) kept away from the base
        while len(self.obstacles) < c.n_obstacles:
            pos = np.array([random.uniform(5, c.width - 5), random.uniform(5, c.height - 5)])
            if np.linalg.norm(pos - np.array(c.base_pos)) > 15:
                self.obstacles.append(Obstacle(pos, random.uniform(2.0, 4.0)))

        uid = 0
        # clustered litter: gaussian blobs around random "bin/bench" centres
        for _ in range(c.n_clusters):
            while True:
                centre = np.array([random.uniform(15, c.width - 15), random.uniform(15, c.height - 15)])
                if self._free(centre, margin=6):
                    break
            placed = 0
            while placed < c.cluster_size:
                p = centre + np.random.normal(0, c.cluster_spread, 2)
                if self._free(p):
                    self.litter.append(Litter(uid, p, clustered=True))
                    uid += 1
                    placed += 1
        # scattered litter: uniform random
        placed = 0
        while placed < c.n_scattered:
            p = np.array([random.uniform(1, c.width - 1), random.uniform(1, c.height - 1)])
            if self._free(p):
                self.litter.append(Litter(uid, p, clustered=False))
                uid += 1
                placed += 1

    # ---- Part C ----------------------------------------------------------
    def emit_signal(self, pos):
        if self.cfg.signaling:
            self.signals.append(Signal(pos, self.cfg.signal_ttl))

    def remaining(self):
        return sum(1 for l in self.litter if not l.collected)

    def step(self):
        for s in self.signals:
            s.ttl -= 1
        self.signals = [s for s in self.signals if s.ttl > 0]
        order = self.robots[:]
        random.shuffle(order)               # random activation order
        for r in order:
            r.step()
        self.t += 1
        self.history.append(self.remaining())

    def done(self):
        return self.remaining() == 0 and all(r.carrying == 0 for r in self.robots)


# ----------------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------------
STATE_COLOURS = {
    Robot.SEARCH: "tab:blue", Robot.TO_LITTER: "tab:green", Robot.TO_SIGNAL: "tab:orange",
    Robot.RETURN: "tab:purple", Robot.CHARGING: "gold", Robot.STRANDED: "black",
}


def animate(cfg: Config, steps=1500, interval=40):
    model = LitterModel(cfg)
    fig, ax = plt.subplots(figsize=(8, 8))

    def draw(_frame):
        if not model.done() and model.t < steps:
            model.step()
        ax.clear()
        ax.set_xlim(0, cfg.width)
        ax.set_ylim(0, cfg.height)
        ax.set_aspect("equal")
        for ob in model.obstacles:
            ax.add_patch(Circle(ob.pos, ob.radius, color="forestgreen", alpha=0.6))
        for s in model.signals:                       # fading signal rings
            ax.add_patch(Circle(s.pos, 6, fill=False, ec="red", alpha=0.15 + 0.6 * s.ttl / s.max_ttl))
        left = [l for l in model.litter if not l.collected]
        if left:
            ax.scatter([l.pos[0] for l in left], [l.pos[1] for l in left],
                       c="saddlebrown", marker="s", s=18, label="litter")
        ax.plot(*cfg.base_pos, marker="*", color="red", markersize=22, label="base")
        ax.add_patch(Circle(cfg.base_pos, cfg.base_radius, fill=False, ec="red", ls="--"))
        for r in model.robots:
            ax.scatter(*r.pos, c=STATE_COLOURS[r.state], s=60, edgecolors="k", zorder=5)
            if r.carrying:
                ax.scatter(*r.pos, c="saddlebrown", marker="s", s=16, zorder=6)
        ax.set_title(f"t={model.t}   remaining={model.remaining()}   delivered={model.delivered}   "
                     f"signaling={'ON' if cfg.signaling else 'OFF'}")
        handles = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec="k", label=s)
                   for s, c in STATE_COLOURS.items()]
        ax.legend(handles=handles, loc="upper right", fontsize=7)

    _anim = FuncAnimation(fig, draw, interval=interval, cache_frame_data=False)
    plt.show()


# ----------------------------------------------------------------------------
# Experiment: does swarm signaling actually help? (Part C evidence)
# ----------------------------------------------------------------------------
def run_headless(cfg: Config, max_steps=3000):
    model = LitterModel(cfg)
    while not model.done() and model.t < max_steps:
        model.step()
    return model


def compare(n_runs=10, max_steps=3000):
    results = {}
    for flag in (True, False):
        curves, finish = [], []
        for seed in range(n_runs):
            m = run_headless(Config(signaling=flag, seed=seed), max_steps)
            h = m.history + [m.history[-1]] * (max_steps - len(m.history))
            curves.append(h)
            finish.append(m.t)
        results[flag] = (np.mean(curves, axis=0), np.mean(finish))
        print(f"signaling={'ON ' if flag else 'OFF'}: mean steps to clear all litter = {np.mean(finish):.0f}")

    plt.figure(figsize=(8, 5))
    plt.plot(results[True][0], label="Signaling ON")
    plt.plot(results[False][0], label="Signaling OFF")
    plt.xlabel("Time step")
    plt.ylabel("Litter remaining (mean of runs)")
    plt.title(f"Effect of swarm signaling ({n_runs} runs each)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("signaling_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare()
    else:
        animate(Config())