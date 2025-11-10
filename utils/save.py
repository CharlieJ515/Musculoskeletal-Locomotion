from pathlib import Path
from rl.base import BaseRL
from utils.tmp_dir import get_tmp


def save_ckpt(agent: BaseRL, ckpt_name: str) -> Path:
    tmpdir = get_tmp()
    if not ckpt_name.endswith(".pt"):
        ckpt_name = f"{ckpt_name}.pt"
    ckpt_path = tmpdir / ckpt_name
    agent.save(ckpt_path)
    return ckpt_path
