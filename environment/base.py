# from __future__ import annotations
# from abc import ABC, abstractmethod
# from typing import Any, Tuple, List, Dict
# import torch
#
# from utils import TransitionBatch
#
# from abc import ABC, abstractmethod
# from collections.abc import Iterable, Sequence
# from typing import Any, Optional, Union
#
# import numpy as np
#
# class VecEnv(ABC):
#     """
#     An abstract asynchronous, vectorized environment.
#
#     :param num_envs: Number of environments
#     :param observation_space: Observation space
#     :param action_space: Action space
#     """
#
#     def step(self, actions: np.ndarray) -> TransitionBatch:
#         """
#         Step the environments with the given action
#
#         :param actions: the action
#         :return: observation, reward, done, information
#         """
#         self.step_async(actions)
#         return self.step_wait()
#
#     @abstractmethod
#     def step_async(self, actions: np.ndarray) -> None:
#         """
#         Tell all the environments to start taking a step
#         with the given actions.
#         Call step_wait() to get the results of the step.
#
#         You should not call this if a step_async run is
#         already pending.
#         """
#         raise NotImplementedError()
#
#     @abstractmethod
#     def step_wait(self) -> TransitionBatch:
#         """
#         Wait for the step taken with step_async().
#
#         :return: observation, reward, done, information
#         """
#         raise NotImplementedError()
#
#     @abstractmethod
#     def reset(self) -> torch.Tensor:
#         """
#         Reset all the environments and return an array of
#         observations, or a tuple of observation arrays.
#
#         If step_async is still doing work, that work will
#         be cancelled and step_wait() should not be called
#         until step_async() is invoked again.
#
#         :return: observation
#         """
#
#     def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
#         """
#         Gym environment rendering
#
#         :param mode: the rendering type
#         """
#
#     @abstractmethod
#     def close(self) -> None:
#         """
#         Clean up the environment's resources.
#         """
#
#     def seed(self, seed: Optional[int] = None) -> Sequence[Union[None, int]]:
#         """
#         Sets the random seeds for all environments, based on a given seed.
#         Each individual environment will still get its own seed, by incrementing the given seed.
#         WARNING: since gym 0.26, those seeds will only be passed to the environment
#         at the next reset.
#
#         :param seed: The random seed. May be None for completely random seeding.
#         :return: Returns a list containing the seeds for each individual env.
#             Note that all list elements may be None, if the env does not return anything when being seeded.
#         """
#         if seed is None:
#             # To ensure that subprocesses have different seeds,
#             # we still populate the seed variable when no argument is passed
#             seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
#
#         self._seeds = [seed + idx for idx in range(self.num_envs)]
#         return self._seeds
#
#     @property
#     def action_space(self) -> torch.Size:
#         return

