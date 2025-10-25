from abc import ABC, abstractmethod
from typing import Any

class Reward(ABC):
    @abstractmethod
    def compute(self) -> float:
        pass

    def reset(self):
        pass
