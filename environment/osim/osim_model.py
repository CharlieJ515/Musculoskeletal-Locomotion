from pathlib import Path
from typing import Dict
import warnings

import opensim

from .pose import Pose
from .action import Action
from .observation import Observation, NormSpec
from utils import require_reset

class RangeError(ValueError): ...

class OsimModel:
    def __init__(self, model_path: Path, visualize: bool, integrator_accuracy: float, stepsize: float):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model_path = model_path
        self._model = opensim.Model(str(self.model_path))
        self._model.buildSystem()

        # Add actuators as constant functions. Then, during simulations
        # we will change levels of constants.
        # One actuartor per each muscle
        self._brain = opensim.PrescribedController()
        self._actuators: Dict[str, opensim.Constant] = {}
        muscle_set = self._model.getMuscles()
        for i in range(muscle_set.getSize()):
            muscle = muscle_set.get(i)
            name = muscle.getName()

            func = opensim.Constant(1.0)
            self._brain.addActuator(muscle_set.get(i))
            self._brain.prescribeControlForActuator(i, func)
            self._actuators[name] = func
        self._model.addController(self._brain)

        # Enable the visualizer
        self.visualize = visualize
        self._model.setUseVisualizer(self.visualize)

        self.integrator_accuracy = integrator_accuracy
        self.stepsize = stepsize
        # indicator to prevent running other methods before reset
        self._was_reset = False

        norm_spec = NormSpec.build(self.stepsize, self._model, self._state)
        Observation.norm_spec = norm_spec

    @require_reset
    def actuate(self, action: Action):
        for name, activation in action:
            actuator = self._actuators[name]
            actuator.setValue(activation)

    @require_reset
    def get_obs(self):
        self.model.realizeAcceleration(self._state)
        return Observation.from_opensim(self._model, self._state)

    @property
    def state(self) -> opensim.State:
        return self._state

    @property
    def model(self) -> opensim.Model:
        return self._model

    def reset(self, pose: Pose) -> Observation:
        self._state = self._model.initSystem()
        self._state.setTime(0)
        self._reset_pose(pose)
        self._model.equilibrateMuscles(self._state)
        self._reset_manager()

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
                    raise RangeError(f"Coordinate '{name}' out of range: {q:.4f} not in [{lo:.4f}, {hi:.4f}].")
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
            print(i,jointSet.get(i).getName())

        print("\nBODIES")
        bodySet = self._model.getBodySet()
        for i in range(bodySet.getSize()):
            print(i,bodySet.get(i).getName())

        print("\nMUSCLES")
        muscleSet = self._model.getMuscles()
        for i in range(muscleSet.getSize()):
            muscle = muscleSet.get(i) 
            print(i,muscle.getName(),muscle.getMaxIsometricForce())

        print("\nFORCES")
        forceSet = self._model.getForceSet()
        for i in range(forceSet.getSize()):
            print(i,forceSet.get(i).getName())

        print("\nMARKERS")
        markerSet = self._model.getMarkerSet()
        for i in range(markerSet.getSize()):
            print(i,markerSet.get(i).getName())

