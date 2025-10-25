from typing import Iterator, Tuple, Dict, ClassVar
import math

import numpy as np
from numpy.typing import NDArray
import torch


class Action:
    __slots__ = ("_activation",)
    muscle_order: ClassVar[Tuple[str, ...]]=()

    def __init__(self, activation: Dict[str, float], check_integrity: bool=True):
        self._activation = activation

        if check_integrity:
            self._check_key()
            self._check_range()

    def _check_key(self):
        keys = set(self.muscle_order)
        have = set(self._activation.keys())
        missing = [k for k in self.muscle_order if k not in have]
        extra = [k for k in have if k not in keys]

        if missing or extra:
            parts = []
            if missing: parts.append(f"missing={missing}")
            if extra:   parts.append(f"extra={extra}")
            raise ValueError("Invalid muscles mapping: " + "; ".join(parts))

    def _check_range(self):
        for k, v in self._activation.items():
            if math.isnan(v):
                raise ValueError(f"NaN activation for {k}")
            if not math.isfinite(v):
                raise ValueError(f"Inf activation for {k}: {v}")
            if v < 0.0 or v > 1.0:
                raise ValueError(f"Activation out of [0,1] for {k}: {v}")

    def __iter__(self) -> Iterator[Tuple[str, float]]:
        return iter(self._activation.items())

    def __getitem__(self, name: str) -> float:
        act = self._activation.get(name)
        if act is None:
            raise KeyError(f"Muscle activation '{name}' not found in Action.")

        return act

    def to_numpy(self) -> NDArray[np.float32]:
        return np.asarray([self._activation[name] for name in self.muscle_order], dtype=np.float32)

    def to_torch(self) -> torch.Tensor:
        return torch.as_tensor(self.to_numpy())

    @classmethod
    def from_numpy(cls, arr: NDArray[np.float32]) -> "Action":
        expected_shape = (len(cls.muscle_order), )
        if arr.shape != expected_shape:
            raise ValueError(f"Expected array of length {expected_shape}, got {arr.shape[0]}")

        if np.any(np.isnan(arr)):
            raise ValueError("NaN values detected in activation array.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Inf or non-finite values detected in activation array.")
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError("Activation values must be within [0, 1].")

        mapping: Dict[str, float] = {name: float(val) for name, val in zip(cls.muscle_order, arr)}
        return cls(mapping, check_integrity=False)
    
    @classmethod
    def from_torch(cls, tensor: torch.Tensor) -> "Action":
        if tensor.dtype != torch.float32:
            raise TypeError(f"from_torch expects dtype=torch.float32, got {tensor.dtype}.")

        arr = tensor.detach().cpu().numpy()
        return cls.from_numpy(arr)

