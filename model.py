# model.py
import os
import math
from typing import List, Tuple, Optional
from enum import IntEnum

import numpy as np
from mpi4py import MPI

from repast4py import parameters
from repast4py.context import SharedContext
from repast4py.schedule import SharedScheduleRunner
from repast4py.space import SharedGrid, DiscretePoint, BorderType, OccupancyType
from repast4py.geometry import BoundingBox
from repast4py.logging import TabularLogger, DataSource
from repast4py import random as rrandom

from agents import Firefighter, TYPE_FIREFIGHTER  # Fire is now an environment grid

class FireState(IntEnum):
    SAFE = 0
    BURNING = 1
    BURNT = 2
    EXTINGUISHED = 3


class RadioBoard:
    def __init__(self):
        self.reports: List[Tuple[int, int]] = []

    def post(self, x: int, y: int):
        self.reports.append((x, y))

    def clear(self):
        self.reports.clear()


class ScalarGetter(DataSource):
    def __init__(self, name, getter):
        self._name = name
        self._getter = getter

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return float

    def get(self):
        return float(self._getter())


class WildfireModel:

    def __init__(self, params):
        self.params = params
        self.done = False
        self.width = int(params['width'])
        self.height = int(params['height'])
        self.torus = bool(params['torus'])
        self.ignitions = int(params['ignitions'])
        self.base_spread_p = float(params['base_spread_p'])

        self.ff_count = int(params['firefighters'])
        self.perception_r = int(params['perception_r'])
        self.comm_r = int(params['comm_r'])
        self.max_water = int(params['max_water'])
        self.comm_freq = int(params['comm_freq'])

        self.max_ticks = int(params['max_ticks'])
        self.seed = int(params['random_seed'])
        self.log_dir = str(params.get('log_dir', 'output'))

        # ---- MPI, RNG, context, grid ----
        self.comm = MPI.COMM_WORLD
        rrandom.init(self.seed)

        self.context = SharedContext(self.comm)

        self.grid = SharedGrid(
            "grid",
            bounds=BoundingBox(0, self.width, 0, self.height),
            borders=BorderType.Torus if self.torus else BorderType.Sticky,
            occupancy=OccupancyType.Multiple,
            buffer_size=1,
            comm=self.comm
        )
        self.context.add_projection(self.grid)

        # ENVIRONMENT: fire as numpy arrays
        # state: FireState enum int; intensity: float 0..1 (only relevant when BURNING)
        self.fire_state = np.full((self.width, self.height), int(FireState.SAFE), dtype=np.int8)
        self.fire_intensity = np.zeros((self.width, self.height), dtype=np.float32)

        # random ignitions
        for _ in range(self.ignitions):
            x = int(rrandom.default_rng.integers(0, self.width))
            y = int(rrandom.default_rng.integers(0, self.height))
            self._ignite_cell(x, y)
        print(f"[init] initial burning = {self.count_burning()} (expected 3)")

        # Firefighters
        aid = 1
        for _ in range(self.ff_count):
            ff = Firefighter(
                aid=aid, rank=0,
                max_water=self.max_water,
                perception_r=self.perception_r,
                comm_r=self.comm_r,
                comm_freq=self.comm_freq
            )
            self.context.add(ff)
            rx = int(rrandom.default_rng.integers(0, self.width))
            ry = int(rrandom.default_rng.integers(0, self.height))
            self.grid.move(ff, DiscretePoint(rx, ry))
            aid += 1

        # Radio + metrics / logging
        self.radio = RadioBoard()
        self.tick = 0
        self.messages = 0
        self.contained_at = math.nan

        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = TabularLogger(
          self.comm,
          os.path.join(self.log_dir, "metrics.csv"),
          ["tick","burning","burnt","extinguished","messages","firefighters","contained_at"]
        )

        # Scheduler
        self.runner = SharedScheduleRunner(self.comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        
    def step(self):
        self.burndown_step()
        self.spread_step()
        self.firefighters_step()
        self.radio_clear_step()
        self.log_step()

    # Environment helpers
    def _ignite_cell(self, x: int, y: int):
        if self.fire_state[x, y] == int(FireState.SAFE):
            self.fire_state[x, y] = int(FireState.BURNING)
            self.fire_intensity[x, y] = 1.0

    def _extinguish_cell(self, x: int, y: int):
        if self.fire_state[x, y] == int(FireState.BURNING):
            self.fire_state[x, y] = int(FireState.EXTINGUISHED)
            self.fire_intensity[x, y] = 0.0

    def _neighbors(self, x: int, y: int):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.torus:
                    nx %= self.width
                    ny %= self.height
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    yield nx, ny

    def count_burning(self) -> int:
        return int(np.count_nonzero(self.fire_state == int(FireState.BURNING)))

    def count_burnt(self) -> int:
        return int(np.count_nonzero(self.fire_state == int(FireState.BURNT)))

    def count_extinguished(self) -> int:
        return int(np.count_nonzero(self.fire_state == int(FireState.EXTINGUISHED)))

    def burndown_step(self):
        burning_mask = (self.fire_state == int(FireState.BURNING))
        self.fire_intensity[burning_mask] -= 0.25
        finished_mask = burning_mask & (self.fire_intensity <= 0.0)
        self.fire_state[finished_mask] = int(FireState.BURNT)
        self.fire_intensity[finished_mask] = 0.0

    def spread_step(self):
        # probabilistic spread from burning cells to SAFE neighbors
        to_ignite = []
        burning_coords = np.argwhere(self.fire_state == int(FireState.BURNING))
        for x, y in burning_coords:
            # spread chance scales with intensity
            p = self.base_spread_p * (0.5 + 0.5 * float(self.fire_intensity[x, y]))
            if rrandom.default_rng.random() < p:
                for nx, ny in self._neighbors(int(x), int(y)):
                    if self.fire_state[nx, ny] == int(FireState.SAFE):
                        to_ignite.append((nx, ny))
        for (ix, iy) in to_ignite:
            self._ignite_cell(ix, iy)

    def firefighters_step(self):
        # build burning set for quick lookups
        burning_coords = np.argwhere(self.fire_state == int(FireState.BURNING))
        burning_set = { (int(x), int(y)) for x, y in burning_coords }

        # 1) broadcasting
        for ff in self.context.agents(TYPE_FIREFIGHTER):
            ff.step_tick()
            if ff.ready_to_broadcast():
                x, y = self._loc(ff)
                r = ff.perception_r
                posted = 0
                x0, x1 = max(0, x - r), min(self.width, x + r + 1)
                y0, y1 = max(0, y - r), min(self.height, y + r + 1)
                for ix in range(x0, x1):
                    for iy in range(y0, y1):
                        if (ix, iy) in burning_set:
                            self.radio.post(ix, iy)
                            posted += 1
                            if posted >= 8:
                                break
                    if posted >= 8:
                        break
                ff.messages_sent += posted
                self.messages += posted

        # 2) move + extinguish
        for ff in self.context.agents(TYPE_FIREFIGHTER):
            x, y = self._loc(ff)

            target = self._nearest_burning_in_vision(x, y, ff.perception_r, burning_set)
            if target is None:
                target = self._nearest_radio(x, y, ff.comm_r)

            if target is not None:
                tx, ty = target
                dx = int(np.sign(tx - x))
                dy = int(np.sign(ty - y))
                self.grid.move(ff, DiscretePoint(self._clip(x + dx, 0, self.width - 1),
                                                 self._clip(y + dy, 0, self.height - 1)))
            else:
                # small random patrol
                rx = self._clip(x + int(rrandom.default_rng.integers(-1, 2)), 0, self.width - 1)
                ry = self._clip(y + int(rrandom.default_rng.integers(-1, 2)), 0, self.height - 1)
                self.grid.move(ff, DiscretePoint(rx, ry))

            # extinguish if standing on burning cell and has water
            nx, ny = self._loc(ff)
            if ff.water > 0 and self.fire_state[nx, ny] == int(FireState.BURNING):
                self._extinguish_cell(nx, ny)
                ff.water -= 1
            else:
                # instant refill on border (simple placeholder rule)
                if ff.water == 0 and (nx == 0 or ny == 0 or nx == self.width - 1 or ny == self.height - 1):
                    ff.water = ff.max_water

    def _nearest_burning_in_vision(self, x, y, r, burning_set) -> Optional[Tuple[int, int]]:
        best = None
        best_d = 10**9
        x0, x1 = max(0, x - r), min(self.width, x + r + 1)
        y0, y1 = max(0, y - r), min(self.height, y + r + 1)
        for ix in range(x0, x1):
            for iy in range(y0, y1):
                if (ix, iy) in burning_set:
                    d = abs(ix - x) + abs(iy - y)
                    if d < best_d:
                        best_d = d
                        best = (ix, iy)
        return best

    def _nearest_radio(self, x, y, comm_r) -> Optional[Tuple[int, int]]:
        if comm_r <= 0 or not self.radio.reports:
            return None
        best = None
        best_d = 10**9
        for (ix, iy) in self.radio.reports:
            d = abs(ix - x) + abs(iy - y)
            if comm_r >= 9999 or d <= comm_r:
                if d < best_d:
                    best_d = d
                    best = (ix, iy)
        return best

    def radio_clear_step(self):
        self.radio.clear()

    def log_step(self):
        self.tick += 1
        if math.isnan(self.contained_at) and self.count_burning() == 0:
            self.contained_at = self.tick
        self.logger.log_row(
            self.tick,
            self.count_burning(),
            self.count_burnt(),
            self.count_extinguished(),
            self.messages,
            self.ff_count,
            (-1 if math.isnan(self.contained_at) else self.contained_at),
        )
        if self.tick % 5 == 0:
          print(f"[tick {self.tick}] burning={self.count_burning()} "
             f"burnt={self.count_burnt()} ext={self.count_extinguished()} msgs={self.messages}")
        if self.count_burning() == 0 or self.tick >= self.max_ticks:
          self._done = True
          self.runner.stop()

    def _loc(self, agent) -> Tuple[int, int]:
        pt: DiscretePoint = self.grid.get_location(agent)
        return int(pt.x), int(pt.y)

    def _clip(self, v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    def run(self):
      self._done = False
      while not self._done:
        self.runner.execute()
      self.logger.write()
      self.logger.close()
      print(f"[DONE] tick={self.tick} burning={self.count_burning()} "
            f"burnt={self.count_burnt()} extinguished={self.count_extinguished()} "
            f"messages={self.messages} contained_at={self.contained_at}")


if __name__ == "__main__":
    p = parameters.init_params("params.yaml", "")
    model = WildfireModel(p)
    model.run()
