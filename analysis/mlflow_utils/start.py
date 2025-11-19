import mlflow
import gymnasium as gym
import opensim
import torch
import numpy as np

from .attempt import tag_attempt


def start_mlflow(uri: str, experiment_name: str, run_name: str):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)
    tag_attempt(experiment_name, run_name)

    mlflow.log_params(
        {
            "gym_version": gym.__version__,
            "opensim_version": opensim.__version__,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }
    )
