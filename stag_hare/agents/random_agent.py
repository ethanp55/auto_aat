from stag_hare.agents.agent import Agent
from stag_hare.environment.state import State
from typing import Tuple


class Random(Agent):
    def __init__(self, name: str) -> None:
        Agent.__init__(self, name)

    def act(self, state: State, reward: float, round_num: int) -> Tuple[int, int]:
        return self.random_action(state)
