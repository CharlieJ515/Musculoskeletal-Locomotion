from __future__ import annotations
from pathlib import Path

model_root: Path = Path(__file__).resolve().parent
gait14dof22_path: Path = model_root / "gait14dof22musc.osim"

__all__ = [
    "gait14dof22_path"
]

