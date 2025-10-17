
from typing import List, Tuple, Dict, Iterable, Self, ClassVar
from dataclasses import dataclass

import opensim
import numpy as np
from numpy.typing import NDArray
import torch

from utils.opensim import SpatialVec_to_Vec3, Transform_to_Vec3, Vec3, to_Vec3

@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    L: float
    T: float
    mass: float
    g: float
    joint_ranges: Dict[str, Tuple[np.ndarray, np.ndarray]]
    muscle_ranges: Dict[str, Dict[str, float]]
    force_label_role: Dict[Tuple[str, str], str]

    @classmethod
    def build_normalizer(cls, model: opensim.Model, state: opensim.State) -> Self:
        T = 1.0
        mass = float(model.getTotalMass(state))
        g = model.getGravity().get(1)

        # length scale L = height/2 (~ pelvis height)
        body_set = model.getBodySet()
        pelvis = body_set.get("pelvis")
        p = pelvis.getTransformInGround(state).p()
        L = p.get(1)

        # joint
        joint_ranges: Dict[str, Tuple[Tuple[float, float], ...]] = {}
        joint_set = model.getJointSet()
        for i in range(joint_set.getSize()):
            j = joint_set.get(i)
            name = j.getName()
            rng: List[Tuple[float, float]] = []

            n = j.numCoordinates()
            for k in range(n):
                c = j.get_coordinates(k)
                lo = c.getRangeMin()
                hi = c.getRangeMax()
                
                rng.append((lo, hi))
            joint_ranges[name] = tuple(rng)

        # muscle
        muscle_set = model.getMuscles()
        for i in range(muscle_set.getSize()):
            m = muscle_set.get(i)
            name = m.getName()

            lopt = m.getOptimalFiberLength()
            vmax = m.getMaxContractionVelocity()
            fmax = m.getMaxIsometricForce()

            muscle_set[name] = {
                "fiber_length": lopt,
                "fiber_velocity": vmax,
                "fiber_force": fmax,
            }

        # force
        force_set = model.getForceSet()
        role: Dict[Tuple[str, str], str] = {}
        for i in range(force_set.getSize()):
            f = force_set.get(i)
            fname = f.getName()
            labels = f.getRecordLabels()
            for j in range(labels.getSize()):
                lbl = labels.get(j)
                LBL = lbl.lower()
                if "moment" in LBL or LBL.startswith("m_") or "_mx" in LBL or "_my" in LBL or "_mz" in LBL:
                    role[(fname, lbl)] = "M"
                elif "cop" in LBL or "pos" in LBL:
                    role[(fname, lbl)] = "P"
                else:
                    role[(fname, lbl)] = "F"

        # spec = NormalizationSpec(
        #     L=L, T=T, mass=mass, g=g, pelvis_name=pelvis_name,
        #     joint_ranges=joint_ranges,
        #     muscle_lopt=muscle_lopt,
        #     muscle_vmax_lopt_per_s=muscle_vmax,
        #     muscle_fmax=muscle_fmax,
        #     force_label_role=role,
        # )
        # cls.Normalizer = spec
        # return spec

@dataclass(frozen=True, slots=True)
class JointState:
    dof: int
    ang: Tuple[float, ...] # angle
    ang_vel: Tuple[float, ...] # angular velocity
    ang_acc: Tuple[float, ...] # angular acceleration

    def __post_init__(self):
        # check attributes have length of self.dof
        for name in ("ang", "ang_vel", "ang_acc"):
            val = getattr(self, name)
            if len(val) == self.dof:
                continue

            raise ValueError(
                f"{name} length {len(val)} does not match dof={self.dof}"
            )

@dataclass(frozen=True, slots=True)
class BodyState:
    pos: Vec3 # position
    vel: Vec3 # velocity
    acc: Vec3 # acceleration
    ang: Vec3 # angle
    ang_vel: Vec3 # angular velocity
    ang_acc: Vec3 # angular acceleration

@dataclass(frozen=True, slots=True)
class MuscleState:
    activation: float
    fiber_length: float
    fiber_velocity: float
    fiber_force: float

@dataclass(frozen=True, slots=True)
class ComponentState:
    force: Vec3
    torque: Vec3

@dataclass(frozen=True, slots=True)
class FootState:
    ground: ComponentState
    calcn: ComponentState
    toes: ComponentState

@dataclass(frozen=True, slots=True)
class MarkerState:
    pos: Vec3
    vel: Vec3
    acc: Vec3

class Observation:
    __slots__ = ("joint", "body", "muscle", "foot", "force", "marker", "mass_center")

    joint_index: ClassVar[Dict[str, int]] = {}
    body_index: ClassVar[Dict[str, int]] = {}
    muscle_index: ClassVar[Dict[str, int]] = {}
    force_index: ClassVar[Dict[str, int]] = {}
    force_label_index: ClassVar[Dict[Tuple[str, str], int]] = {}
    marker_index: ClassVar[Dict[str, int]] = {}

    norm_spec: ClassVar[NormalizationSpec | None] = None

    def __init__(
        self,
        joint: Dict[str, JointState],
        body: Dict[str, BodyState],
        muscle: Dict[str, MuscleState],
        foot: Dict[str, FootState],
        force: Dict[str, Dict[str, float]],
        marker: Dict[str, Dict[str, Tuple[float, ...]]],
        mass_center: Dict[str, Tuple[float, ...]],
    ):
        self.joint = joint
        self.body = body
        self.muscle = muscle
        self.foot = foot
        self.force = force
        self.marker = marker
        self.mass_center = mass_center

    @classmethod
    def from_opensim(
        cls,
        model: opensim.Model,
        state: opensim.State,
    ) -> "Observation":
        if not cls.joint_index:
            cls._init_indices(model)

        joint_set = model.getJointSet()
        joint: Dict[str, JointState] = {}
        for i in range(joint_set.getSize()):
            jnt = joint_set.get(i)
            name = jnt.getName()
            dof = jnt.numCoordinates()
            ang = tuple(jnt.get_coordinates(k).getValue(state) for k in range(dof))
            ang_vel = tuple(jnt.get_coordinates(k).getSpeedValue(state) for k in range(dof))
            ang_acc = tuple(jnt.get_coordinates(k).getAccelerationValue(state) for k in range(dof))

            joint[name] = JointState(
                dof = dof,
                ang = ang,
                ang_vel = ang_vel,
                ang_acc = ang_acc,
            )

        body_set = model.getBodySet()
        body: Dict[str, BodyState] = {}
        for i in range(body_set.getSize()):
            b = body_set.get(i)
            name = b.getName()

            pos_G = b.getTransformInGround(state)
            vel_G = b.getVelocityInGround(state)
            acc_G = b.getAccelerationInGround(state)
            
            pos, ang = Transform_to_Vec3(pos_G)
            vel, ang_vel = SpatialVec_to_Vec3(vel_G)
            acc, ang_acc = SpatialVec_to_Vec3(acc_G)

            body[name] = BodyState(
                pos = pos,
                vel = vel,
                acc = acc,
                ang = ang,
                ang_vel = ang_vel,
                ang_acc = ang_acc,
            )

        muscle_set = model.getMuscles()
        muscle: Dict[str, MuscleState] = {}
        for i in range(muscle_set.getSize()):
            m = muscle_set.get(i)
            name = m.getName()
            muscle[name] = MuscleState(
                activation = m.getActivation(state),
                fiber_length = m.getFiberLength(state),
                fiber_velocity = m.getFiberVelocity(state),
                fiber_force = m.getFiberForce(state),
            )

        def _get_triplet(data: Dict[str, float], base: str) -> Vec3:
            # Handles missing components by defaulting to 0.0
            return (
                float(data.get(f"{base}.X", 0.0)),
                float(data.get(f"{base}.Y", 0.0)),
                float(data.get(f"{base}.Z", 0.0)),
            )

        force_set = model.getForceSet()
        foot: Dict[str, FootState] = {}
        for i in range(force_set.getSize()):
            f = force_set.get(i)
            name = f.getName()  # e.g., "foot_r", "foot_l"
            if not name.startswith("foot_"):
                continue

            labels = f.getRecordLabels()
            values = f.getRecordValues(state)

            label_list = [labels.get(k) for k in range(labels.getSize())]
            value_list = [values.get(k) for k in range(values.size())]
            data = dict(zip(label_list, value_list))

            suffix = name[-2:]  # "_r" or "_l"
            def partner_key(p: str) -> str:
                return p if p == "ground" else f"{p}{suffix}"

            def read_component(partner: str) -> ComponentState:
                base_force  = f"{name}.{partner_key(partner)}.force"
                base_torque = f"{name}.{partner_key(partner)}.torque"
                Fx, Fy, Fz = _get_triplet(data, base_force)
                Tx, Ty, Tz = _get_triplet(data, base_torque)
                return ComponentState(force=(Fx, Fy, Fz), torque=(Tx, Ty, Tz))

            foot[name] = FootState(
                ground=read_component("ground"),
                calcn=read_component("calcn"),
                toes=read_component("toes"),
            )

        force: Dict[str, Dict[str, float]] = {}
        for i in range(force_set.getSize()):
            f = force_set.get(i)
            name = f.getName()
            if name in muscle or name in foot:
                continue

            labels = f.getRecordLabels()
            values = f.getRecordValues(state)
            label_list = [labels.get(k) for k in range(labels.getSize())]
            value_list = [values.get(k) for k in range(values.size())]

            force[name] = dict(zip(label_list, value_list))

        marker_set = model.getMarkerSet()
        marker: Dict[str, MarkerState] = {}
        for i in range(marker_set.getSize()):
            mk = marker_set.get(i)
            name = mk.getName()
            pos = to_Vec3(mk.getLocationInGround(state))
            vel = to_Vec3(mk.getVelocityInGround(state))
            acc = to_Vec3(mk.getAccelerationInGround(state))
            marker[name] = MarkerState(pos=pos, vel=vel, acc=acc)

        # Center of Mass
        mass_center: Dict[str, Tuple[float, ...]] = {
            "pos": to_Vec3(model.calcMassCenterPosition(state)),
            "vel": to_Vec3(model.calcMassCenterVelocity(state)),
            "acc": to_Vec3(model.calcMassCenterAcceleration(state)),
        }

        return cls(joint=joint, body=body, muscle=muscle, foot=foot, force=force, marker=marker, mass_center=mass_center)

    @classmethod
    def _init_indices(
        cls,
        model: opensim.Model,
    ) -> None:
        joint_set = model.getJointSet()
        body_set = model.getBodySet()
        muscle_set = model.getMuscles()
        force_set = model.getForceSet()
        marker_set = model.getMarkerSet()

        cls.joint_index = {joint_set.get(i).getName(): i for i in range(joint_set.getSize())}
        cls.body_index = {body_set.get(i).getName(): i for i in range(body_set.getSize())}
        cls.muscle_index = {muscle_set.get(i).getName(): i for i in range(muscle_set.getSize())}
        cls.force_index = {force_set.get(i).getName(): i for i in range(force_set.getSize())}
        cls.marker_index = {marker_set.get(i).getName(): i for i in range(marker_set.getSize())}

        for i in range(force_set.getSize()):
            f = force_set.get(i)
            name = f.getName()
            labels = f.getRecordLabels()
            for j in range(labels.getSize()):
                cls.force_label_index[(name, labels.get(j))] = j

    def flatten(self) -> np.ndarray:
        parts: list[float] = []

        for jname in sorted(self.joint.keys()):
            j = self.joint[jname]
            parts.extend(j["pos"])
            parts.extend(j["vel"])
            parts.extend(j["acc"])

        for bname in sorted(self.body.keys()):
            b = self.body[bname]
            parts.extend(b["pos"])
            parts.extend(b["vel"])
            parts.extend(b["acc"])
            parts.extend(b["pos_rot"])
            parts.extend(b["vel_rot"])
            parts.extend(b["acc_rot"])

        for mname in sorted(self.muscle.keys()):
            m = self.muscle[mname]
            parts.extend([
                m["activation"],
                m["fiber_length"],
                m["fiber_velocity"],
                m["fiber_force"],
            ])

        for fname in sorted(self.force.keys()):
            f = self.force[fname]
            # use force_label_index to preserve OpenSim’s record order
            labels = sorted(f.keys(), key=lambda lbl: self.force_label_index.get((fname, lbl), 1e9))
            for lbl in labels:
                parts.append(f[lbl])

        for mkname in sorted(self.marker.keys()):
            mk = self.marker[mkname]
            parts.extend(mk["pos"])
            parts.extend(mk["vel"])
            parts.extend(mk["acc"])

        parts.extend(self.mass_center["pos"])
        parts.extend(self.mass_center["vel"])
        parts.extend(self.mass_center["acc"])

        return np.asarray(parts, dtype=np.float32)

    def normalize(self):
        if self.norm_spec is None:
            raise RuntimeError
