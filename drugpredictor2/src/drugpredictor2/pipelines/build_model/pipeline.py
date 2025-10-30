# Refined and correct pipeline definition
from kedro.pipeline import Pipeline, node
from typing import Dict

# Only the main orchestration function needs to be imported here
from .nodes import train_model_on_partitions

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_model_on_partitions,
                inputs=[
                    "featurized_data",
                    "params:train_params",
                    "params:split_params"
                ],
                outputs=[
                    "lipinski_model",
                    "lipinski_training_history",
                    "lipinski_train_predictions",
                    "lipinski_train_report",
                    "lipinski_val_predictions",
                    "lipinski_val_report"
                ],
                name="train_model_on_partitions_node"
            )
        ]
    )