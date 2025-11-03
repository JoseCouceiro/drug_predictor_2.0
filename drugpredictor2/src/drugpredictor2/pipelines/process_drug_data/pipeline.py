from kedro.pipeline import Pipeline, node, pipeline
from .nodes import process_drug_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=process_drug_dataset,
            inputs="drug_raw",
            outputs=["drug_X", "drug_y_drug", "drug_y_atc", "atc_mapping"],
            name="process_drug_dataset_node"
        ),
    ])
