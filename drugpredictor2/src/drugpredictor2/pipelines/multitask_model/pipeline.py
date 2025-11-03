from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_multitask_model, evaluate_multitask_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_multitask_model,
            inputs=[
                "lipinski_model",
                "drug_X",
                "drug_y_drug",
                "drug_y_atc",
                # Train/val split needs to be handled in process_drug_data node: process_drug_dataset
                "X_val",
                "y_drug_val",
                "y_atc_val",
                "params:n_atc_classes"
            ],
            outputs=["multitask_model", "multitask_training_history"],
            name="train_multitask_model_node"
        ),
        node(
            func=evaluate_multitask_model,
            inputs=["multitask_model", "X_train", "y_drug_train", "y_atc_train"],
            outputs=[
                "multitask_train_predictions",
                "multitask_train_drug_report",
                "multitask_train_atc_report"
            ],
            name="evaluate_multitask_train_node"
        ),
        node(
            func=evaluate_multitask_model,
            inputs=["multitask_model", "X_val", "y_drug_val", "y_atc_val"],
            outputs=[
                "multitask_val_predictions",
                "multitask_val_drug_report",
                "multitask_val_atc_report"
            ],
            name="evaluate_multitask_val_node"
        ),
    ])

