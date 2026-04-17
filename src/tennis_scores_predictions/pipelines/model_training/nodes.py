"""Nodes for model training and evaluation."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def _labels_to_1d(y: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
    """Coerce target labels to a 1D int array (CSV reload yields a single-column DataFrame)."""
    return np.asarray(y, dtype=np.int64).ravel()


def split_features_and_target(
    df: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    target_col = params["target_column"]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    params: dict[str, Any],
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, _labels_to_1d(y_train))
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series | pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, float]:
    y_true = _labels_to_1d(y_test)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
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
        name=f"RF-{datetime.now().strftime('%H:%M:%S')}-n{params['n_estimators']}",
        config={
            "model_type": "RandomForest",
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "random_state": params["random_state"],
            "test_size": params["test_size"],
        },
        tags=["baseline", "sklearn"],
    )

    try:
        wandb.log(metrics)

        if hasattr(model, "feature_importances_"):
            wandb.sklearn.plot_feature_importances(
                model, feature_names=list(X_test.columns)
            )

        artifact = wandb.Artifact(
            name="trained-model",
            type="model",
            description=f"RandomForest n={params['n_estimators']}",
        )
        artifact_path = "data/06_models/trained_model.pkl"
        if os.path.exists(artifact_path):
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
    finally:
        if run is not None:
            wandb.finish()

    return metrics
