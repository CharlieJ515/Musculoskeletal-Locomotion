import tempfile
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

_tmpdir_str:str = tempfile.mkdtemp(prefix="opensim_rl")
_tmpdir:Path = Path(_tmpdir_str)

def get_tmp()->Path:
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

def next_attempt(experiment_name: str, run_name: str) -> int:
    """
    Determine the next attempt number for a given run name.

    This function queries MLflow for all runs in the given experiment
    that have the specified ``run_name`` tag, inspects their existing
    ``attempt`` tags, and returns one greater than the highest attempt
    value found.  
    If the experiment does not exist or no attempts have been logged,
    the function returns 1.

    :param experiment_name: Name of the MLflow experiment.
    :type experiment_name: str
    :param run_name: Name of the run whose attempts are tracked.
    :type run_name: str
    :return: The next attempt number to use.
    :rtype: int
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return 1

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
    )

    max_attempt = 0
    for r in runs:
        if "attempt" in r.data.tags:
            try:
                max_attempt = max(max_attempt, int(r.data.tags["attempt"]))
            except ValueError:
                pass

    return max_attempt + 1

def tag_attempt(experiment_name: str, run_name: str) -> int:
    """
    Set the ``attempt`` tag for the active MLflow run.

    This function computes the next attempt number by calling
    :func:`next_attempt` and then sets it as the ``attempt`` tag on
    the currently active MLflow run.  
    It raises an error if no MLflow run is active.

    :param experiment_name: Name of the MLflow experiment.
    :type experiment_name: str
    :param run_name: Name of the run whose attempt is being tagged.
    :type run_name: str
    :raises RuntimeError: If no active MLflow run is found.
    :return: The attempt number that was assigned and tagged.
    :rtype: int
    """
    if mlflow.active_run() is None:
        raise RuntimeError("Start an MLflow run before calling tag_attempt().")

    attempt = next_attempt(experiment_name, run_name)
    mlflow.set_tag("attempt", attempt)

    return attempt

