import math
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass(frozen=True, slots=True)
class CoordState:
    q: Optional[float] = None  # position (angle or translation)
    u: Optional[float] = None  # speed (angular or linear velocity)


@dataclass(slots=True)
class Pose:
    name: str
    _coord: dict[str, CoordState] = field(default_factory=dict)

    def set(self, name: str, q: Optional[float] = None, u: Optional[float] = None):
        self._coord[name] = CoordState(q=q, u=u)

    def __iter__(self) -> Iterator[tuple[str, CoordState]]:
        return iter(self._coord.items())

    def __getitem__(self, name: str) -> CoordState:
        coord = self._coord.get(name)
        if coord is None:
            raise KeyError(f"Coordinate '{name}' not found in Pose.")

        return coord

    def __contains__(self, name: str) -> bool:
        return name in self._coord


def get_default_pose() -> Pose:
    pose = Pose("default")

    return pose


def get_tilted_pose() -> Pose:
    pose = get_default_pose()
    pose.name = "tilted"
    pose.set("pelvis_tilt", q=math.radians(-12))
    pose.set("hip_flexion_r", q=math.radians(12))
    pose.set("hip_flexion_l", q=math.radians(12))

    return pose


def get_bent_pose() -> Pose:
    pose = get_tilted_pose()
    pose.name = "bent"
    pose.set("pelvis_ty", q=0.920)
    pose.set("knee_angle_r", q=math.radians(-12))
    pose.set("knee_angle_l", q=math.radians(-12))
    pose.set("ankle_angle_r", q=math.radians(12))
    pose.set("ankle_angle_l", q=math.radians(12))

    return pose


def get_forward_pose() -> Pose:
    pose = get_bent_pose()
    pose.name = "forward"
    pose.set("hip_flexion_r", q=math.radians(20))
    pose.set("hip_adduction_r", q=math.radians(-3))
    pose.set("hip_adduction_l", q=math.radians(-3))
    return pose


POSE_REGISTRY = {
    "default": get_default_pose,
    "tilted": get_tilted_pose,
    "bent": get_bent_pose,
    "forward": get_forward_pose,
}
