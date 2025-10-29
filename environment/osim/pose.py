from typing import Dict, Optional, Tuple, Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CoordState:
    q: Optional[float] = None  # position (angle or translation)
    u: Optional[float] = None  # speed (angular or linear velocity)


@dataclass(slots=True)
class Pose:
    _coord: Dict[str, CoordState] = field(default_factory=dict)

    def set(self, name: str, q: Optional[float] = None, u: Optional[float] = None):
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


pose = Pose()

# speeds (QQDot)
pose.set("pelvis_tx", u=0.0)  # forward speed (QQDot[3])
pose.set("pelvis_tz", u=0.0)  # rightward speed (QQDot[5])

# pelvis orientations/translations set to zero in reset
pose.set("pelvis_list", q=0.0)  # QQ[1]
pose.set("pelvis_rotation", q=0.0)  # QQ[2]
pose.set("pelvis_ty", q=0.94)  # pelvis height (QQ[4])
pose.set("pelvis_tilt", q=-0.0)  # trunk lean (+ backward) (QQ[0])

# right leg
pose.set("hip_adduction_r", q=-0.0)  # abduction is negative of adduction (QQ[7])
pose.set("hip_flexion_r", q=-0.0)  # flexion is negative in this model (QQ[6])
pose.set("knee_angle_r", q=0.0)  # extension positive (QQ[13])
pose.set("ankle_angle_r", q=-0.0)  # “flex” sign per model (QQ[15])

# left leg
pose.set("hip_adduction_l", q=-0.0)  # (QQ[10])
pose.set("hip_flexion_l", q=-0.0)  # (QQ[9])
pose.set("knee_angle_l", q=0.0)  # (QQ[14])
pose.set("ankle_angle_l", q=-0.0)  # (QQ[16])

osim_rl_pose = pose
