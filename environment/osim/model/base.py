# revised version of L2M2019
# https://github.com/stanfordnmbl/osim-rl/blob/master/osim/env/osim.py

from pathlib import Path
from typing import Tuple, Dict, Iterable, Self, ClassVar
import math

import opensim
import numpy as np
from numpy.typing import NDArray
import torch

from .action import Action
from .observation import Observation
from utils import require_reset

class OsimModel:
    def __init__(self, model_path: Path, visualize: bool, integrator_accuracy: float=5e-5, stepsize: float=0.01):
        if Action.muscle_order == ():
            raise RuntimeError

        self.model_path = model_path
        self._model = opensim.Model(str(self.model_path))
        self._model.buildSystem()

        # Add actuators as constant functions. Then, during simulations
        # we will change levels of constants.
        # One actuartor per each muscle
        self._brain = opensim.PrescribedController()
        muscleSet = self._model.getMuscles()
        for j in range(muscleSet.getSize()):
            func = opensim.Constant(1.0)
            self._brain.addActuator(muscleSet.get(j))
            self._brain.prescribeControlForActuator(j, func)
        self._model.addController(self._brain)

        # Enable the visualizer
        self.visualize = visualize
        self._model.setUseVisualizer(self.visualize)

        self.integrator_accuracy = integrator_accuracy
        self.stepsize = stepsize
        # indicator to prevent running other methods before reset
        self._was_reset = False

    # @require_reset
    # def actuate(self, action: Action):
    #     brain = opensim.PrescribedController.safeDownCast(self.model.getControllerSet().get(0))
    #     functionSet = brain.get_ControlFunctions()
    #
    #     for j in range(functionSet.getSize()):
    #         func = opensim.Constant.safeDownCast(functionSet.get(j))
    #         func.setValue( float(action[j]) )
    #     raise NotImplementedError

    @require_reset
    def get_obs(self):
        jointSet:opensim.JointSet = self._model.getJointSet()
        joint_dict = {}
        for i in range(jointSet.getSize()):
            joint:opensim.Joint = jointSet.get(i)
            name:str = joint.getName()
            num_coord:int = joint.numCoordinates()

            pos = (joint.get_coordinates(j).getValue(self._state) for j in range(num_coord))
            vel = (joint.get_coordinates(j).getSpeedValue(self._state) for j in range(num_coord))
            acc = (joint.get_coordinates(j).getAccelerationValue(self._state) for j in range(num_coord))

            joint_dict[name] = {"pos":pos, "vel":vel, "acc":acc}

        bodySet:opensim.BodySet = self._model.getBodySet()
        body_dict = {}
        for i in range(bodySet.getSize()):
            body:opensim.Body = bodySet.get(i)
            name:str = body.getName()
            num_coord:int = 3

            pos = (body.getTrans
            pos = (body.get_coordinates(j).getValue(self._state) for j in range(num_coord))
            vel = (body.get_coordinates(j).getSpeedValue(self._state) for j in range(num_coord))
            acc = (body.get_coordinates(j).getAccelerationValue(self._state) for j in range(num_coord))

            joint_dict[name] = {"pos":pos, "vel":vel, "acc":acc}


    def get_state(self) -> opensim.State:
        return self._state

    def get_model(self) -> opensim.Model:
        return self._model

    def reset(self) -> Observation:
        self._state = self._model.initSystem()
        self._model.equilibrateMuscles(self._state)
        self._state.setTime(0)
        self._reset_manager()
        self.step = 0

        self._was_reset = True

        obs = self.get_obs()
        return obs

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

