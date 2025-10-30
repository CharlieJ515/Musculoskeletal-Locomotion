from typing import List, Tuple, Dict, Self, ClassVar, Any
from dataclasses import dataclass
from math import pi

import opensim
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .index import body_index
from utils.vec3 import Vec3


Range = Tuple[float, float]


@dataclass(frozen=True, slots=True)
class NormSpec:
    L: float
    T: float
    mass: float
    g: float
    joint_ranges: Dict[str, Tuple[Range, ...]]
    muscle_ranges: Dict[str, Dict[str, float]]

    @classmethod
    def build(cls, T: float, model: opensim.Model, state: opensim.State) -> Self:
        mass = float(model.getTotalMass(state))
        g = abs(model.getGravity().get(1))

        # length scale L = height/2 (~ pelvis height)
        body_set = model.getBodySet()
        pelvis = body_set.get("pelvis")
        p = pelvis.getTransformInGround(state).p()
        L = p[1]

        # joint
        joint_set = model.getJointSet()
        joint_ranges: Dict[str, Tuple[Range, ...]] = {}
        for i in range(joint_set.getSize()):
            j = joint_set.get(i)
            name = j.getName()
            rng: List[Range] = []

            n = j.numCoordinates()
            for k in range(n):
                c = j.get_coordinates(k)
                lo = c.getRangeMin()
                hi = c.getRangeMax()
                rng.append((lo, hi))
            joint_ranges[name] = tuple(rng)

        # muscle
        muscle_ranges: Dict[str, Dict[str, float]] = {}
        muscle_set = model.getMuscles()
        for i in range(muscle_set.getSize()):
            m = muscle_set.get(i)
            name = m.getName()

            lopt = m.getOptimalFiberLength()
            vmax = m.getMaxContractionVelocity()
            fmax = m.getMaxIsometricForce()

            muscle_ranges[name] = {
                "fiber_length": lopt,
                "fiber_velocity": vmax,
                "fiber_force": fmax,
            }

        spec = cls(
            T=T,
            L=L,
            mass=mass,
            g=g,
            joint_ranges=joint_ranges,
            muscle_ranges=muscle_ranges,
        )
        return spec


@dataclass(frozen=True, slots=True)
class JointState:
    dof: int
    ang: Tuple[float, ...]  # angle
    ang_vel: Tuple[float, ...]  # angular velocity
    ang_acc: Tuple[float, ...]  # angular acceleration

    normalized: bool = False

    def __post_init__(self):
        # check attributes have length of self.dof
        for name in ("ang", "ang_vel", "ang_acc"):
            val = getattr(self, name)
            if len(val) == self.dof:
                continue

            raise ValueError(f"{name} length {len(val)} does not match dof={self.dof}")

    def norm(self, rngs: Tuple[Range, ...], T: float) -> "JointState":
        if len(rngs) != self.dof:
            raise RuntimeError

        ang = []
        ang_vel = []
        ang_acc = []
        for i in range(self.dof):
            lo, hi = rngs[i]
            half = 0.5 * (hi - lo)
            mid = 0.5 * (hi + lo)

            ang.append((self.ang[i] - mid) / half)
            ang_vel.append(self.ang_vel[i] / (half / T))
            ang_acc.append(self.ang_acc[i] / (half / (T * T)))

        return JointState(
            dof=self.dof,
            ang=tuple(ang),
            ang_vel=tuple(ang_vel),
            ang_acc=tuple(ang_acc),
            normalized=True,
        )

    @classmethod
    def from_Joint(cls, joint: opensim.Joint, state: opensim.State) -> "JointState":
        dof = joint.numCoordinates()
        ang = tuple(joint.get_coordinates(k).getValue(state) for k in range(dof))
        ang_vel = tuple(
            joint.get_coordinates(k).getSpeedValue(state) for k in range(dof)
        )
        ang_acc = tuple(
            joint.get_coordinates(k).getAccelerationValue(state) for k in range(dof)
        )

        return JointState(
            dof=dof,
            ang=ang,
            ang_vel=ang_vel,
            ang_acc=ang_acc,
        )


def rot_to_np(R: opensim.Rotation) -> np.ndarray:
    return np.array(
        [
            [R.get(0, 0), R.get(0, 1), R.get(0, 2)],
            [R.get(1, 0), R.get(1, 1), R.get(1, 2)],
            [R.get(2, 0), R.get(2, 1), R.get(2, 2)],
        ],
        dtype=float,
    )


def vec3_to_np(v: opensim.Vec3) -> np.ndarray:
    return np.array([v.get(0), v.get(1), v.get(2)], dtype=float)


def spatial_to_np(v: opensim.SpatialVec):
    return vec3_to_np(v.get(0)), vec3_to_np(v.get(1))


def find_relative_velocity(
    X_GA: opensim.Transform,
    V_GA: opensim.SpatialVec,
    X_GB: opensim.Transform,
    V_GB: opensim.SpatialVec,
) -> tuple[Vec3, Vec3]:
    R_GA = rot_to_np(X_GA.R())
    R_AG = R_GA.T
    p_GA = vec3_to_np(X_GA.p())
    p_GB = vec3_to_np(X_GB.p())

    w_GA, v_GA = spatial_to_np(V_GA)
    w_GB, v_GB = spatial_to_np(V_GB)

    r = p_GB - p_GA

    # ω_AB in A
    w_AB = R_AG @ (w_GB - w_GA)
    # v_AB in A
    v_AB = R_AG @ (v_GB - v_GA - np.cross(w_GA, r))

    return Vec3.from_numpy(w_AB), Vec3.from_numpy(v_AB)


def find_relative_acceleration(
    X_GA: opensim.Transform,
    V_GA: opensim.SpatialVec,
    A_GA: opensim.SpatialVec,
    X_GB: opensim.Transform,
    V_GB: opensim.SpatialVec,
    A_GB: opensim.SpatialVec,
) -> tuple[Vec3, Vec3]:
    R_GA = rot_to_np(X_GA.R())
    R_AG = R_GA.T
    p_GA = vec3_to_np(X_GA.p())
    p_GB = vec3_to_np(X_GB.p())

    w_GA, v_GA = spatial_to_np(V_GA)
    w_GB, v_GB = spatial_to_np(V_GB)
    alpha_GA, a_GA = spatial_to_np(A_GA)
    alpha_GB, a_GB = spatial_to_np(A_GB)

    r = p_GB - p_GA

    # α_AB in A
    alpha_AB = R_AG @ (alpha_GB - alpha_GA)

    # Relative linear velocity in G of B-origin w.r.t. A-origin:
    v_rel_G = (v_GB - v_GA) - np.cross(w_GA, r)

    # a_AB in A
    a_AB = R_AG @ (
        (a_GB - a_GA)
        - np.cross(alpha_GA, r)
        - np.cross(w_GA, np.cross(w_GA, r))
        - 2.0 * np.cross(w_GA, v_rel_G)
    )

    return Vec3.from_numpy(alpha_AB), Vec3.from_numpy(a_AB)


@dataclass(frozen=True, slots=True)
class BodyState:
    pos: Vec3  # position
    vel: Vec3  # velocity
    acc: Vec3  # acceleration
    ang: Vec3  # angle
    ang_vel: Vec3  # angular velocity
    ang_acc: Vec3  # angular acceleration

    ref_frame: opensim.Frame
    normalized: bool = False

    def norm(self, ref: "BodyState", L: float, T: float) -> "BodyState":
        return BodyState(
            pos=(self.pos - ref.pos) / L,
            vel=(self.vel - ref.vel) / (L / T),
            acc=(self.acc - ref.acc) / (L / (T * T)),
            ang=self.ang / pi,
            ang_vel=self.ang_vel * T,
            ang_acc=self.ang_acc * (T * T),
            ref_frame=self.ref_frame,
            normalized=True,
        )

    @classmethod
    def from_Body(
        cls, body: opensim.Body, ref: opensim.Frame, state: opensim.State
    ) -> "BodyState":
        X_GB = body.getTransformInGround(state)
        V_GB = body.getVelocityInGround(state)
        A_GB = body.getAccelerationInGround(state)

        X_GA = ref.getTransformInGround(state)
        V_GA = ref.getVelocityInGround(state)
        A_GA = ref.getAccelerationInGround(state)

        X_AB = body.findTransformBetween(state, ref)

        ang, pos = Vec3.from_Transform(X_AB)
        ang_vel, vel = find_relative_velocity(X_GA, V_GA, X_GB, V_GB)
        ang_acc, acc = find_relative_acceleration(X_GA, V_GA, A_GA, X_GB, V_GB, A_GB)

        return cls(
            pos=pos,
            vel=vel,
            acc=acc,
            ang=ang,
            ang_vel=ang_vel,
            ang_acc=ang_acc,
            ref_frame=ref,
        )


@dataclass(frozen=True, slots=True)
class PointState:
    pos: Vec3
    vel: Vec3
    acc: Vec3

    normalized: bool = False

    def norm(self, ref: "BodyState|PointState", L: float, T: float) -> "PointState":
        return PointState(
            pos=(self.pos - ref.pos) / L,
            vel=(self.vel - ref.vel) / (L / T),
            acc=(self.acc - ref.acc) / (L / (T * T)),
            normalized=True,
        )

    @classmethod
    def from_Marker(cls, marker: opensim.Marker, state: opensim.State) -> "PointState":
        pos = Vec3.from_Vec3(marker.getLocationInGround(state))
        vel = Vec3.from_Vec3(marker.getVelocityInGround(state))
        acc = Vec3.from_Vec3(marker.getAccelerationInGround(state))

        return PointState(pos=pos, vel=vel, acc=acc)


@dataclass(frozen=True, slots=True)
class MuscleState:
    activation: float
    fiber_length: float
    fiber_velocity: float
    fiber_force: float

    normalized: bool = False

    def norm(self, muscle_range: Dict[str, float]) -> "MuscleState":
        return MuscleState(
            activation=self.activation,
            fiber_length=self.fiber_length / muscle_range["fiber_length"],
            fiber_velocity=self.fiber_velocity / muscle_range["fiber_velocity"],
            fiber_force=self.fiber_force / muscle_range["fiber_force"],
            normalized=True,
        )

    @classmethod
    def from_Muscle(cls, muscle: opensim.Muscle, state: opensim.State) -> "MuscleState":
        return MuscleState(
            activation=muscle.getActivation(state),
            fiber_length=muscle.getFiberLength(state),
            fiber_velocity=muscle.getFiberVelocity(state),
            fiber_force=muscle.getFiberForce(state),
        )


@dataclass(frozen=True, slots=True)
class ComponentState:
    force: Vec3
    torque: Vec3

    normalized: bool = False

    def norm(self, mg: float, L: float) -> "ComponentState":
        return ComponentState(
            force=self.force / mg,
            torque=self.torque / (mg * L),
            normalized=True,
        )


@dataclass(frozen=True, slots=True)
class FootState:
    ground: ComponentState
    calcn: ComponentState
    toes: ComponentState

    normalized: bool = False

    def norm(self, mg: float, L: float) -> "FootState":
        return FootState(
            ground=self.ground.norm(mg, L),
            calcn=self.calcn.norm(mg, L),
            toes=self.toes.norm(mg, L),
            normalized=True,
        )

    @classmethod
    def from_Force(cls, force: opensim.Force, state: opensim.State) -> "FootState":
        name = force.getName()
        suffix = name[-2:]  # "_r" or "_l"
        labels = force.getRecordLabels()
        values = force.getRecordValues(state)

        label_list = [labels.get(k) for k in range(labels.getSize())]
        value_list = [values.get(k) for k in range(values.size())]
        data = dict(zip(label_list, value_list))

        def _get_triplet(data: Dict[str, float], base: str) -> Vec3:
            # Handles missing components by defaulting to 0.0
            return Vec3(
                float(data.get(f"{base}.X", 0.0)),
                float(data.get(f"{base}.Y", 0.0)),
                float(data.get(f"{base}.Z", 0.0)),
            )

        def partner_key(p: str) -> str:
            return p if p == "ground" else f"{p}{suffix}"

        def read_component(partner: str) -> ComponentState:
            base_force = f"{name}.{partner_key(partner)}.force"
            base_torque = f"{name}.{partner_key(partner)}.torque"
            force = _get_triplet(data, base_force)
            torque = _get_triplet(data, base_torque)
            return ComponentState(force=force, torque=torque)

        return FootState(
            ground=read_component("ground"),
            calcn=read_component("calcn"),
            toes=read_component("toes"),
        )


@dataclass(frozen=True, slots=True)
class Observation:
    joint: Dict[str, JointState]
    body: Dict[str, BodyState]
    pelvis: BodyState
    muscle: Dict[str, MuscleState]
    foot: Dict[str, FootState]
    force: Dict[str, Dict[str, float]]
    marker: Dict[str, PointState]
    mass_center: PointState
    target_velocity: Vec3

    normalized: bool = False

    norm_spec: ClassVar[NormSpec | None] = None

    @classmethod
    def build(
        cls,
        model: opensim.Model,
        state: opensim.State,
        target_velocity: Vec3,
    ) -> "Observation":
        # joint
        joint_set = model.getJointSet()
        joint: Dict[str, JointState] = {}
        for i in range(joint_set.getSize()):
            j = joint_set.get(i)
            name = j.getName()
            joint[name] = JointState.from_Joint(j, state)

        # body
        body_set = model.getBodySet()
        pelvis_idx = body_index("pelvis")
        pelvis = body_set.get(pelvis_idx)
        body: Dict[str, BodyState] = {}
        for i in range(body_set.getSize()):
            b = body_set.get(i)
            name = b.getName()
            body[name] = BodyState.from_Body(b, pelvis, state)

        ground = model.getGround()
        pelvis_state = BodyState.from_Body(pelvis, ground, state)

        # muscle
        muscle_set = model.getMuscles()
        muscle: Dict[str, MuscleState] = {}
        for i in range(muscle_set.getSize()):
            m = muscle_set.get(i)
            name = m.getName()
            muscle[name] = MuscleState.from_Muscle(m, state)

        # foot
        force_set = model.getForceSet()
        foot: Dict[str, FootState] = {}
        for i in range(force_set.getSize()):
            f = force_set.get(i)
            name = f.getName()  # e.g., "foot_r", "foot_l"
            if not name.startswith("foot_"):
                continue
            foot[name] = FootState.from_Force(f, state)

        # force
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

        # marker
        marker_set = model.getMarkerSet()
        marker: Dict[str, PointState] = {}
        for i in range(marker_set.getSize()):
            mk = marker_set.get(i)
            name = mk.getName()
            marker[name] = PointState.from_Marker(mk, state)

        # center of mass
        pos = Vec3.from_Vec3(model.calcMassCenterPosition(state))
        vel = Vec3.from_Vec3(model.calcMassCenterVelocity(state))
        acc = Vec3.from_Vec3(model.calcMassCenterAcceleration(state))
        mass_center = PointState(pos=pos, vel=vel, acc=acc)

        return Observation(
            joint=joint,
            body=body,
            pelvis=pelvis_state,
            muscle=muscle,
            foot=foot,
            force=force,
            marker=marker,
            mass_center=mass_center,
            target_velocity=target_velocity,
        )

    def norm(self) -> "Observation":
        if self.norm_spec is None:
            raise RuntimeError(
                "Observation.norm_spec is not set. Call NormSpec.build(...) before normalize()."
            )

        L = self.norm_spec.L
        T = self.norm_spec.T
        mg = self.norm_spec.g * self.norm_spec.mass

        ref = self.body.get("pelvis")
        if ref is None:
            raise RuntimeError("pelvis body not found in Observation.body.")

        joint: Dict[str, JointState] = {}
        for name, j in self.joint.items():
            rngs = self.norm_spec.joint_ranges.get(name)
            if rngs is None:
                raise RuntimeError(f"missing joint ranges for '{name}' in norm_spec.")
            joint[name] = j.norm(rngs, T)

        body: Dict[str, BodyState] = {}
        for name, b in self.body.items():
            body[name] = b.norm(ref, L, T)

        muscle: Dict[str, MuscleState] = {}
        for name, m in self.muscle.items():
            rng = self.norm_spec.muscle_ranges.get(name)
            if rng is None:
                raise RuntimeError(f"missing muscle range for '{name}' in norm_spec.")
            muscle[name] = m.norm(rng)

        foot: Dict[str, FootState] = {}
        for name, f in self.foot.items():
            foot[name] = f.norm(mg, L)

        force: Dict[str, Dict[str, float]] = self.force

        marker: Dict[str, PointState] = {}
        for name, m in self.marker.items():
            marker[name] = m.norm(ref, L, T)

        mass_center: PointState = self.mass_center.norm(ref, L, T)

        return Observation(
            joint=joint,
            body=body,
            pelvis=self.pelvis,
            muscle=muscle,
            foot=foot,
            force=force,
            marker=marker,
            mass_center=mass_center,
            normalized=True,
            target_velocity=self.target_velocity,
        )

    def to_L2M(self) -> Dict[str, Any]:
        obs_dict = {}

        pelvis_body = self.body["pelvis"]
        pelvis_joint = self.joint["ground_pelvis"]
        pelvis = {
            "height": pelvis_body.pos.y,
            "pitch": -pelvis_joint.ang[0],
            "roll": pelvis_joint.ang[1],
            "vel": [
                pelvis_body.vel.x,
                -pelvis_body.vel.y,
                pelvis_body.vel.z,
                -pelvis_joint.ang_vel[0],
                pelvis_joint.ang_vel[1],
                pelvis_joint.ang_vel[2],
            ],
        }
        obs_dict["pelvis"] = pelvis

        for side in ["r", "l"]:
            leg = {}

            foot = self.foot[f"foot_{side}"]
            ground = foot.ground
            leg["ground_reaction_forces"] = [
                ground.force.x,
                ground.force.y,
                ground.force.z,
            ]

            leg["joint"] = {
                "hip_abd": self.joint[f"hip_{side}"].ang[1],
                "hip": self.joint[f"hip_{side}"].ang[0],
                "knee": self.joint[f"knee_{side}"].ang[0],
                "ankle": self.joint[f"ankle_{side}"].ang[0],
            }
            leg["d_joint"] = {
                "hip_abd": self.joint[f"hip_{side}"].ang_vel[1],
                "hip": self.joint[f"hip_{side}"].ang_vel[0],
                "knee": self.joint[f"knee_{side}"].ang_vel[0],
                "ankle": self.joint[f"ankle_{side}"].ang_vel[0],
            }

            muscle = {name: m.norm(self.norm_spec.muscle_ranges[name]) for (name, m) in self.muscle.items()}  # type: ignore[reportOptionalMemberAccess]
            muscle_name_map = {
                "abd": "HAB",
                "add": "HAD",
                "iliopsoas": "HFL",
                "glut_max": "GLU",
                "hamstrings": "HAM",
                "rect_fem": "RF",
                "vasti": "VAS",
                "bifemsh": "BFSH",
                "gastroc": "GAS",
                "soleus": "SOL",
                "tib_ant": "TA",
            }
            for name, m in muscle.items():
                key = muscle_name_map[name]
                leg[key] = {
                    "f": m.fiber_force,
                    "l": m.fiber_length,
                    "v": m.fiber_velocity,
                }

            obs_dict[f"{side}_leg"] = leg

        return obs_dict
