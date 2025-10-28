from dataclasses import dataclass
from typing import Tuple
import math

import opensim


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __truediv__(self, s: float) -> "Vec3":
        return Vec3(self.x / s, self.y / s, self.z / s)

    def magnitude(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def norm(self) -> "Vec3":
        magnitude = self.magnitude()
        if magnitude == 0:
            return Vec3(0.0, 0.0, 0.0)
        return self / magnitude

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def from_Vec3(cls, vec: opensim.Vec3) -> "Vec3":
        return Vec3(vec[0], vec[1], vec[2])

    @classmethod
    def from_SpatialVec(cls, vec: opensim.SpatialVec) -> Tuple["Vec3", "Vec3"]:
        ang = vec.get(0)
        lin = vec.get(1)
        return (cls.from_Vec3(ang), cls.from_Vec3(lin))

    @classmethod
    def from_Transform(cls, transform: opensim.Transform) -> Tuple["Vec3", "Vec3"]:
        p = transform.p()
        r = transform.R().convertRotationToBodyFixedXYZ()
        return (cls.from_Vec3(p), cls.from_Vec3(r))
