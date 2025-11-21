from enum import IntEnum
from typing import Tuple
from repast4py.core import Agent

TYPE_FIREFIGHTER = 1

class FFMode(IntEnum):
    ATTACK = 0
    REFILL = 1
    SCOUT = 2


class Firefighter(Agent):

    def __init__(self, aid: int, rank: int, max_water: int, perception_r: int, comm_r: int, comm_freq: int):
        super().__init__(aid, TYPE_FIREFIGHTER, rank)
        self.max_water = int(max_water)
        self.water = int(max_water)
        self.perception_r = int(perception_r)
        self.comm_r = int(comm_r)
        self.comm_freq = int(comm_freq)

        self.messages_sent = 0
        self._tick = 0
        self.mode = FFMode.ATTACK

    def save(self) -> Tuple[int, int, int, int, int, int, int]:
        return (
            self.max_water,
            self.water,
            self.perception_r,
            self.comm_r,
            self.comm_freq,
            self.messages_sent,
            self._tick,
            int(self.mode),
        )

    def update(self, data: Tuple[int, int, int, int, int, int, int]):
        (
            self.max_water,
            self.water,
            self.perception_r,
            self.comm_r,
            self.comm_freq,
            self.messages_sent,
            self._tick,
        ) = map(int, data)
        self.mode = FFMode(mode_val)

    def step_tick(self):
        self._tick += 1

    def ready_to_broadcast(self) -> bool:
        return self.comm_freq > 0 and (self._tick % self.comm_freq == 0)
