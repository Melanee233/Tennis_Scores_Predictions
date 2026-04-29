from kedro.pipeline import Pipeline, node

from tennis_scores_predictions.pipelines.model_training.nodes import (
    split_features_and_target,
)

from .nodes import evaluate_automl_model, train_autogluon


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                split_features_and_target,
                inputs=dict(df="matches_pre_encoded", params="params:model_training"),
                outputs=[
                    "X_train_automl",
                    "X_test_automl",
                    "y_train_automl",
                    "y_test_automl",
                ],
            ),
            node(
                train_autogluon,
                inputs=dict(
                    X_train="X_train_automl",
                    y_train="y_train_automl",
                    params="params:automl_training",
                ),
                outputs="automl_model_path",
            ),
            node(
                evaluate_automl_model,
                inputs=dict(
                    model_path="automl_model_path",
                    X_test="X_test_automl",
                    y_test="y_test_automl",
                    automl_params="params:automl_training",
                    split_params="params:model_training",
                ),
                outputs="automl_metrics",
            ),
        ]
    )
