from dataclasses import replace
from typing import Tuple, Union

import numpy as np
import torch

from analysis import MlflowWriter
from configs import PERConfig
from rl.replay_buffer.replay_buffer import ReplayBuffer
from rl.transition import Transition, TransitionBatch


class SumTree:
    def __init__(self, capacity: int):
        self.depth = int(np.ceil(np.log2(capacity)))
        self.tree_capacity = 2**self.depth

        self.tree = np.zeros(2 * self.tree_capacity, dtype=np.float64)
        self._size = 0

    def add(self, priority: float, buffer_idx: int):
        tree_idx = buffer_idx + self.tree_capacity
        self.update(tree_idx, priority)
        self._size += 1

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate
        while tree_idx != 1:
            tree_idx //= 2
            self.tree[tree_idx] += change

    def get_leaf(self, v: float) -> Tuple[int, float]:
        parent_idx = 1

        while parent_idx < self.tree_capacity:
            left_child = 2 * parent_idx
            right_child = left_child + 1

            if self.tree[left_child] >= v:
                parent_idx = left_child
            else:
                v -= self.tree[left_child]
                parent_idx = right_child

        return parent_idx, self.tree[parent_idx]

    def find(self, v_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = len(v_batch)
        parent_idx = np.ones(batch_size, dtype=np.int64)

        for _ in range(self.depth):
            left_child = parent_idx * 2
            right_child = left_child + 1

            left_vals = self.tree[left_child]
            mask = v_batch <= left_vals

            parent_idx = np.where(mask, left_child, right_child)
            v_batch = np.where(mask, v_batch, v_batch - left_vals)

        return parent_idx, self.tree[parent_idx]

    @property
    def total_priority(self) -> float:
        return self.tree[1]


class PER(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        reward_shape: tuple[int, ...] = (1,),
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-5,
        *,
        device: torch.device = torch.device("cpu"),
        obs_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
        name: str = "PrioritizedReplayBuffer/",
    ) -> None:
        super().__init__(
            capacity=capacity,
            obs_shape=obs_shape,
            action_shape=action_shape,
            reward_shape=reward_shape,
            device=device,
            obs_dtype=obs_dtype,
            action_dtype=action_dtype,
            reward_dtype=reward_dtype,
            name=name,
        )

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1

        self.sum_tree = SumTree(capacity)
        self.max_priority = 1.0

    def add(self, transition: Union[Transition, TransitionBatch]) -> None:
        batch = (
            transition.to_batch() if isinstance(transition, Transition) else transition
        )

        for i in range(len(batch)):
            idx = (self._ptr + i) % self.capacity
            self.sum_tree.add(self.max_priority**self.alpha, idx)

        super().add(batch)

    def sample(self, batch_size: int, *, pin_memory: bool = True) -> TransitionBatch:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        beta = self.beta_start + (1.0 - self.beta_start) * min(
            1.0, self.frame / self.beta_frames
        )
        self.frame += 1

        total_p = self.sum_tree.total_priority
        segment = total_p / batch_size
        v = np.arange(batch_size) * segment + np.random.uniform(0, segment, batch_size)

        tree_indices, priorities = self.sum_tree.find(v)

        sampling_probs = priorities / total_p
        weights = (self._size * sampling_probs) ** (-beta)
        weights = weights / weights.max()
        weights = torch.as_tensor(
            weights,
            dtype=torch.float32,
            device=self._device,
        ).unsqueeze(1)

        batch_indices = tree_indices - self.sum_tree.tree_capacity
        batch_indices = batch_indices % self.capacity

        batch = self.sample_index(batch_indices, pin_memory=pin_memory)
        batch = replace(batch, weights=weights)

        return batch

    def update_priorities(
        self, batch_indices: torch.Tensor, td_errors: torch.Tensor
    ) -> None:
        priorities = td_errors.detach().cpu().abs().sum(dim=1).numpy() + self.epsilon
        tree_indices = (
            batch_indices.detach().cpu().numpy() + self.sum_tree.tree_capacity
        )

        for tree_idx, priority in zip(tree_indices, priorities):
            self.sum_tree.update(tree_idx, priority**self.alpha)

        self.max_priority = max(self.max_priority, priorities.max())

    def clear(self) -> None:
        super().clear()
        self.sum_tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self.frame = 1

    def log_params(
        self, mlflow_writer: MlflowWriter, *, prefix: str = "buffer/"
    ) -> None:
        super().log_params(mlflow_writer, prefix=prefix)

        p = prefix
        per_params_dict = {
            f"{p}alpha": self.alpha,
            f"{p}beta_start": self.beta_start,
            f"{p}beta_frames": self.beta_frames,
            f"{p}epsilon": self.epsilon,
        }

        mlflow_writer.log_params(per_params_dict)

    @classmethod
    def from_config(cls, cfg: PERConfig) -> "PER":  # type: ignore
        return cls(
            capacity=cfg.capacity,
            obs_shape=cfg.obs_shape,
            action_shape=cfg.action_shape,
            reward_shape=cfg.reward_shape,
            alpha=cfg.alpha,
            beta_start=cfg.beta_start,
            beta_frames=cfg.beta_frames,
            epsilon=cfg.epsilon,
            device=cfg.device,
            obs_dtype=cfg.obs_dtype,
            action_dtype=cfg.action_dtype,
            reward_dtype=cfg.reward_dtype,
            name=cfg.name,
        )
