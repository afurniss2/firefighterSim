from typing import Tuple
from repast4py.core import Agent

TYPE_FIREFIGHTER = 1


class Firefighter(Agent):
    """
    Firefighter agent with local perception and lightweight radio comms.
    The fire grid / spread lives in the environment (value layer), not as agents.
    """

    def __init__(self, aid: int, rank: int, max_water: int, perception_r: int, comm_r: int, comm_freq: int):
        super().__init__(aid, TYPE_FIREFIGHTER, rank)
        self.max_water = int(max_water)
        self.water = int(max_water)
        self.perception_r = int(perception_r)
        self.comm_r = int(comm_r)
        self.comm_freq = int(comm_freq)

        self.messages_sent = 0
        self._tick = 0

    def save(self) -> Tuple[int, int, int, int, int, int, int]:
        return (
            self.max_water,
            self.water,
            self.perception_r,
            self.comm_r,
            self.comm_freq,
            self.messages_sent,
            self._tick,
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

    def step_tick(self):
        self._tick += 1

    def ready_to_broadcast(self) -> bool:
        return self.comm_freq > 0 and (self._tick % self.comm_freq == 0)
