from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    process_drug_dataset,
    process_drug_dataset_action,
    process_drug_dataset_organ,
    split_drug_data,
)


def _taxonomy_branch(process_func, prefix: str, node_suffix: str) -> Pipeline:
    """Build a process+split branch for one fixed ATC taxonomy subset,
    producing dataset names prefixed with e.g. 'action_' or 'organ_' so both
    taxonomies can be trained and compared in the same pipeline run."""
    return pipeline([
        node(
            func=process_func,
            inputs="drug_raw",
            outputs=[f"{prefix}drug_X", f"{prefix}drug_y_drug", f"{prefix}drug_y_atc", f"{prefix}atc_mapping"],
            name=f"process_drug_dataset_{node_suffix}_node"
        ),
        node(
            func=split_drug_data,
            inputs=[f"{prefix}drug_X", f"{prefix}drug_y_drug", f"{prefix}drug_y_atc"],
            outputs={
                "X_train": f"{prefix}X_train",
                "X_val": f"{prefix}X_val",
                "y_drug_train": f"{prefix}y_drug_train",
                "y_drug_val": f"{prefix}y_drug_val",
                "y_atc_train": f"{prefix}y_atc_train",
                "y_atc_val": f"{prefix}y_atc_val",
                "n_atc_classes": f"{prefix}n_atc_classes"
            },
            name=f"split_drug_data_{node_suffix}_node"
        ),
    ])


def create_pipeline(**kwargs) -> Pipeline:
    # Default flow, driven by params:atc_subset (null / action / organ / etc.)
    default_pipeline = pipeline([
        node(
            func=process_drug_dataset,
            inputs=["drug_raw", "params:atc_subset"],
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

    # Fixed action-based and organ-based branches, so both taxonomies are
    # processed in a single `kedro run` for direct side-by-side comparison.
    action_pipeline = _taxonomy_branch(process_drug_dataset_action, "action_", "action")
    organ_pipeline = _taxonomy_branch(process_drug_dataset_organ, "organ_", "organ")

    return default_pipeline + action_pipeline + organ_pipeline
