from pathlib import Path
from typing import Dict
import warnings

import opensim

from utils.vec3 import Vec3

from .pose import Pose
from .action import Action
from .observation import Observation, NormSpec
from .index import build_index_bundle
from utils import require_reset


class OsimModel:
    def __init__(
        self,
        model_path: Path,
        visualize: bool,
        integrator_accuracy: float,
        stepsize: float,
    ):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model_path = model_path
        self._model = opensim.Model(str(self.model_path))
        self._model.buildSystem()
        build_index_bundle(self._model)

        self._build_brain()
        self._model.addController(self._brain)

        # Enable the visualizer
        self.visualize = visualize
        self._model.setUseVisualizer(self.visualize)

        self._init_state = self._model.initSystem()

        self.integrator_accuracy = integrator_accuracy
        self.stepsize = stepsize
        # indicator to prevent running other methods before reset
        self._was_reset = False

    @require_reset
    def actuate(self, action: Action):
        # storing constant function (actuator) in python dictionary fails to work properly
        # it seems control function can change as model progresses
        # thereby instead of storing function in python, access them by index at each actuation
        function_set: opensim.FunctionSet = self._brain.get_ControlFunctions()
        for name, activation in action:
            idx = self._actuator_idx[name]
            func: opensim.Function = function_set.get(idx)
            conse_func: opensim.Constant = opensim.Constant.safeDownCast(func)
            conse_func.setValue(activation)

    @require_reset
    def get_obs(self):
        self.model.realizeAcceleration(self._state)
        target_vec = Vec3(0, 0, 0)
        return Observation.build(self._model, self._state, target_vec)

    def _build_brain(self):
        # Add actuators as constant functions. Then, during simulations
        # we will change levels of constants.
        # One actuartor per each muscle
        self._brain = opensim.PrescribedController()
        self._actuator_idx: Dict[str, int] = {}
        muscle_set = self._model.getMuscles()
        for i in range(muscle_set.getSize()):
            muscle = muscle_set.get(i)
            name = muscle.getName()

            func = opensim.Constant(0.0)
            func.setName(name)
            self._brain.addActuator(muscle)
            self._brain.prescribeControlForActuator(name, func)

            self._actuator_idx[name] = i

    @property
    def state(self) -> opensim.State:
        return self._state

    @property
    def model(self) -> opensim.Model:
        return self._model

    def reset(self, pose: Pose) -> Observation:
        self._state = self._init_state
        self._state.setTime(0)
        self._reset_pose(pose)
        self._model.equilibrateMuscles(self._state)
        self._reset_manager()

        if Observation.norm_spec is None:
            norm_spec = NormSpec.build(self.stepsize, self._model, self._state)
            Observation.norm_spec = norm_spec

        self.step = 0
        self._was_reset = True

        obs = self.get_obs()
        return obs

    def _reset_pose(self, pose: Pose):
        coord_set = self._model.getCoordinateSet()
        for i in range(coord_set.getSize()):
            coord = coord_set.get(i)
            name = coord.getName()

            if name not in pose:
                continue
            if coord.getLocked(self._state):
                warnings.warn(
                    f"Coordinate '{name}' is locked — skipping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            coord_state = pose[name]
            q, u = coord_state.q, coord_state.u
            if q is not None:
                lo, hi = coord.getRangeMin(), coord.getRangeMax()
                if not (lo <= q <= hi):
                    raise ValueError(
                        f"Coordinate '{name}' out of range: {q:.4f} not in [{lo:.4f}, {hi:.4f}]."
                    )
                coord.setValue(self._state, q)

            if u is not None:
                coord.setSpeedValue(self._state, u)

    def _reset_manager(self):
        self.manager = opensim.Manager(self._model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self._state)

    @require_reset
    def integrate(self):
        # Define the new endtime of the simulation
        self.step = self.step + 1

        # Integrate till the new endtime
        self._state = self.manager.integrate(self.stepsize * self.step)

    def list_elements(self):
        print("JOINTS")
        jointSet = self._model.getJointSet()
        for i in range(jointSet.getSize()):
            print(i, jointSet.get(i).getName())

        print("\nBODIES")
        bodySet = self._model.getBodySet()
        for i in range(bodySet.getSize()):
            print(i, bodySet.get(i).getName())

        print("\nMUSCLES")
        muscleSet = self._model.getMuscles()
        for i in range(muscleSet.getSize()):
            muscle = muscleSet.get(i)
            print(i, muscle.getName(), muscle.getMaxIsometricForce())

        print("\nFORCES")
        forceSet = self._model.getForceSet()
        for i in range(forceSet.getSize()):
            print(i, forceSet.get(i).getName())

        print("\nMARKERS")
        markerSet = self._model.getMarkerSet()
        for i in range(markerSet.getSize()):
            print(i, markerSet.get(i).getName())
