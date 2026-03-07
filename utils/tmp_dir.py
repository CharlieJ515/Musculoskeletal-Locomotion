import tempfile
import shutil
from pathlib import Path

_tmpdir_str: str = tempfile.mkdtemp(prefix="opensim_rl")
_tmpdir: Path = Path(_tmpdir_str)


def get_tmp() -> Path:
    """
    Retrieve the global temporary directory path.

    This function returns the path to the temporary directory created
    at module import time. It can be used by other parts of the code
    to store intermediate files or artifacts.

    :return: Path to the global temporary directory.
    :rtype: Path
    """
    return _tmpdir


def clear_tmp():
    """
    Remove the temporary directory created for intermediate artifacts.

    This function deletes the global temporary directory created at
    module import time. It should typically be called at the end of a
    training or evaluation run to clean up disk space.

    :return: None
    :rtype: None
    """
    tmp = get_tmp()
    shutil.rmtree(tmp)
