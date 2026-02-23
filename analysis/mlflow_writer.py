import contextlib
import os
from typing import Any

import gymnasium as gym
import mlflow
import numpy as np
import opensim
import torch
from mlflow.tracking import MlflowClient


class MlflowWriter:
    def __init__(self):
        self.is_enabled = os.getenv("ENABLE_MLFLOW", "true").lower() == "true"
        self.client = MlflowClient() if self.is_enabled else None

        self.log_params_flag = True
        self.log_metrics_flag = True
        self.log_artifacts_flag = True
        self.log_figures_flag = True

    def set_enabled(self, state: bool) -> None:
        self.is_enabled = state
        if self.is_enabled and self.client is None:
            self.client = MlflowClient()

    def set_logging(
        self,
        *,
        params: bool | None = None,
        metrics: bool | None = None,
        artifacts: bool | None = None,
        figures: bool | None = None,
    ) -> None:
        if params is not None:
            self.log_params_flag = params
        if metrics is not None:
            self.log_metrics_flag = metrics
        if artifacts is not None:
            self.log_artifacts_flag = artifacts
        if figures is not None:
            self.log_figures_flag = figures

    def _next_attempt(self, experiment_name: str, run_name: str) -> int:
        if not self.is_enabled or self.client is None:
            return 1

        exp = self.client.get_experiment_by_name(experiment_name)
        if exp is None:
            return 1

        runs = self.client.search_runs(
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

    def _tag_attempt(self, experiment_name: str, run_name: str) -> int:
        if not self.is_enabled:
            return 1

        if mlflow.active_run() is None:
            raise RuntimeError("Start an MLflow run before calling tag_attempt().")

        attempt = self._next_attempt(experiment_name, run_name)
        mlflow.set_tag("attempt", attempt)
        return attempt

    def start_main_run(self, uri: str, experiment_name: str, run_name: str) -> None:
        if not self.is_enabled:
            return

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)

        self._tag_attempt(experiment_name, run_name)

        mlflow.log_params(
            {
                "gym_version": gym.__version__,
                "opensim_version": opensim.__version__,
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
            }
        )

    def end_main_run(self) -> None:
        if self.is_enabled:
            mlflow.end_run()

    @contextlib.contextmanager
    def nested_run(self, run_name: str):
        if self.is_enabled:
            with mlflow.start_run(run_name=run_name, nested=True) as run:
                yield run
        else:
            yield None

    def log_params(self, params: dict[str, Any]) -> None:
        if self.is_enabled and self.log_params_flag:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.is_enabled and self.log_metrics_flag:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        if self.is_enabled and self.log_artifacts_flag:
            mlflow.log_artifact(local_path, artifact_path)

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        if self.is_enabled and self.log_figures_flag:
            mlflow.log_figure(figure, artifact_file)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self.is_enabled:
            mlflow.set_tags(tags)

    def active_run_name(self, fallback: str = "test_run") -> str:
        if not self.is_enabled:
            return fallback

        active_run = mlflow.active_run()
        if active_run:
            return active_run.data.tags.get("mlflow.runName", fallback)
        return fallback
