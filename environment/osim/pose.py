from typing import Dict, Optional, Tuple, Iterator
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class CoordState:
    q: Optional[float] = None  # position (angle or translation)
    u: Optional[float] = None  # speed (angular or linear velocity)

@dataclass(frozen=True, slots=True)
class Pose:
    _coord: Dict[str, CoordState] = field(default_factory=dict)

    def set(self, name: str, q: Optional[float]=None, u: Optional[float]=None):
        self._coord[name] = CoordState(q=q, u=u)

    def __iter__(self) -> Iterator[Tuple[str, CoordState]]:
        return iter(self._coord.items())

    def __getitem__(self, name: str) -> CoordState:
        coord = self._coord.get(name)
        if coord is None:
            raise KeyError(f"Coordinate '{name}' not found in Pose.")

        return coord
    
    def __contains__(self, name: str) -> bool:
        return name in self._coord
