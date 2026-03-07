from pathlib import Path
import mlflow
from rl.base import BaseRL
from utils.tmp_dir import get_tmp


def save_ckpt(
    agent: BaseRL,
    ckpt_name: str,
    log_artifact: bool = True,
    artifact_path: str = "checkpoints",
) -> Path:
    tmpdir = get_tmp()
    if not ckpt_name.endswith(".pt"):
        ckpt_name = f"{ckpt_name}.pt"
    ckpt_path = tmpdir / ckpt_name
    agent.save(ckpt_path)

    if not log_artifact:
        return ckpt_path

    mlflow.log_artifact(str(ckpt_path), artifact_path=artifact_path)
    return ckpt_path
