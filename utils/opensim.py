from typing import Tuple

import opensim

Vec3 = Tuple[float, float, float]

def to_Vec3(vec: opensim.Vec3) -> Vec3:
    return (vec[0], vec[1], vec[2])

def SpatialVec_to_Vec3(vec: opensim.SpatialVec) -> Tuple[Vec3, Vec3]:
    ang = vec.get(0)
    lin = vec.get(1)
    return (to_Vec3(ang), to_Vec3(lin))

def Transform_to_Vec3(transform: opensim.Transform) -> Tuple[Vec3, Vec3]:
    p = transform.p()
    r = transform.R().convertRotationToBodyFixedXYZ()
    return (to_Vec3(p), to_Vec3(r))
