"""Nodes for AutoML training and evaluation."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from autogluon.tabular import TabularPredictor


def _labels_to_1d(y: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
    """Coerce target labels to a 1D int array."""
    return np.asarray(y, dtype=np.int64).ravel()


def train_autogluon(
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    params: dict[str, Any],
) -> str:
    """Train an AutoGluon predictor on the prepared training set."""

    label_column = params["target_column"]
    model_path = Path(params["model_path"])

    if params.get("clean_model_dir", True) and model_path.exists():
        shutil.rmtree(model_path)

    train_df = X_train.copy()
    train_df[label_column] = _labels_to_1d(y_train)

    predictor = TabularPredictor(
        label=label_column,
        path=str(model_path),
        problem_type="binary",
        eval_metric=params["eval_metric"],
    )
    predictor.fit(
        train_data=train_df,
        presets=params["presets"],
        time_limit=params["time_limit"],
        verbosity=params.get("verbosity", 2),
    )

    return str(model_path)


def evaluate_automl_model(
    model_path: str,
    X_test: pd.DataFrame,
    y_test: pd.Series | pd.DataFrame,
    automl_params: dict[str, Any],
    split_params: dict[str, Any],
) -> dict[str, float]:
    """Evaluate an AutoGluon predictor and log results to Weights & Biases."""
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(model_path)
    y_true = _labels_to_1d(y_test)
    y_pred = predictor.predict(X_test).to_numpy(dtype=np.int64)

    y_proba_raw = predictor.predict_proba(X_test)
    if isinstance(y_proba_raw, pd.DataFrame):
        if 1 in y_proba_raw.columns:
            y_proba = y_proba_raw[1].to_numpy(dtype=float)
        else:
            y_proba = y_proba_raw.iloc[:, -1].to_numpy(dtype=float)
    else:
        y_proba = np.asarray(y_proba_raw, dtype=float)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        name=f"AutoGluon-{datetime.now().strftime('%H:%M:%S')}-{automl_params['presets']}",
        group=os.getenv("WANDB_GROUP", "model-comparison"),
        job_type="evaluation",
        config={
            "model_type": "AutoGluonTabular",
            "presets": automl_params["presets"],
            "time_limit": automl_params["time_limit"],
            "eval_metric": automl_params["eval_metric"],
            "random_state": split_params["random_state"],
            "test_size": split_params["test_size"],
        },
        tags=["automl", "autogluon", "tabular"],
    )

    try:
        wandb.log(metrics)

        leaderboard = predictor.leaderboard(
            pd.concat(
                [X_test.reset_index(drop=True), pd.Series(y_true, name=automl_params["target_column"])],
                axis=1,
            ),
            silent=True,
        )
        wandb.log({"leaderboard": wandb.Table(dataframe=leaderboard)})

        artifact = wandb.Artifact(
            name="autogluon-model",
            type="model",
            description=f"AutoGluon preset={automl_params['presets']}",
        )
        artifact.add_dir(model_path)
        wandb.log_artifact(artifact)
    finally:
        if run is not None:
            wandb.finish()

    return metrics
