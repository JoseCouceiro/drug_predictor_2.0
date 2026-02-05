from kedro.pipeline import Pipeline, node, pipeline
from .nodes import process_drug_dataset
from .nodes import split_drug_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=process_drug_dataset,
            inputs="drug_raw",
            outputs=["drug_X", "drug_y_drug", "drug_y_atc", "atc_mapping"],
            name="process_drug_dataset_node"
        ),
        node(
            func=split_drug_data,
            inputs=["drug_X", "drug_y_drug", "drug_y_atc"],
            outputs={
                "X_train": "X_train",
                "X_val": "X_val",
                "y_drug_train": "y_drug_train",
                "y_drug_val": "y_drug_val",
                "y_atc_train": "y_atc_train",
                "y_atc_val": "y_atc_val",
                "n_atc_classes": "n_atc_classes"
            },
            name="split_drug_data_node"
        ),
    ])
