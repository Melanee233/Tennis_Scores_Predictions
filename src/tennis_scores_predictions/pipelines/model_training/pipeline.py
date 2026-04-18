from kedro.pipeline import Pipeline, node

from .nodes import evaluate_model, split_features_and_target, train_random_forest


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                split_features_and_target,
                inputs=dict(df="matches_features", params="params:model_training"),
                outputs=["X_train", "X_test", "y_train", "y_test"],
            ),
            node(
                train_random_forest,
                inputs=dict(
                    X_train="X_train",
                    y_train="y_train",
                    params="params:model_training",
                ),
                outputs="trained_model",
            ),
            node(
                evaluate_model,
                inputs=dict(
                    model="trained_model",
                    X_test="X_test",
                    y_test="y_test",
                    params="params:model_training",
                ),
                outputs="metrics",
            ),
        ]
    )
