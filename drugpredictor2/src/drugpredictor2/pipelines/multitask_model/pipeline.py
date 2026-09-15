from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    select_backbone,
    filter_drugs_only,
    get_num_atc_classes_drugs_only,
    create_atc_mapping_drugs_only,
    train_and_evaluate_drug_classifier,
    train_and_evaluate_atc_classifier,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create two hierarchical classification pipelines:
    1. Drug classifier: binary drug vs non-drug (uses all data)
    2. ATC classifier: multiclass ATC classification (uses drug data only)
    """

    # Pick which pretrained Lipinski backbone to transfer from
    # (params:backbone_source = "single" or "multitask")
    backbone_pipeline = pipeline([
        node(
            func=select_backbone,
            inputs=["lipinski_model", "lipinski_multitask_model", "params:backbone_source"],
            outputs="selected_lipinski_backbone",
            name="select_backbone_node"
        ),
    ])

    # Data preparation: Filter drugs for ATC classifier
    data_prep_pipeline = pipeline([
        node(
            func=filter_drugs_only,
            inputs=["X_train", "y_drug_train", "y_atc_train"],
            outputs=["X_train_drugs_only", "y_atc_train_drugs_only"],
            name="filter_train_drugs_node"
        ),
        node(
            func=filter_drugs_only,
            inputs=["X_val", "y_drug_val", "y_atc_val"],
            outputs=["X_val_drugs_only", "y_atc_val_drugs_only"],
            name="filter_val_drugs_node"
        ),
        node(
            func=get_num_atc_classes_drugs_only,
            inputs=["y_atc_train_drugs_only"],
            outputs="n_atc_classes_drugs_only",
            name="get_num_atc_classes_node"
        ),
        node(
            func=create_atc_mapping_drugs_only,
            inputs=["atc_mapping"],
            outputs="atc_mapping_drugs_only",
            name="create_atc_mapping_drugs_only_node"
        ),
    ])
    
    # Pipeline 1: Binary Drug Classifier — train + evaluate in one node
    drug_pipeline = pipeline([
        node(
            func=train_and_evaluate_drug_classifier,
            inputs=[
                "selected_lipinski_backbone",
                "X_train",
                "y_drug_train",
                "X_val",
                "y_drug_val",
            ],
            outputs=[
                "drug_classifier_model",
                "drug_classifier_history",
                "drug_classifier_train_predictions",
                "drug_classifier_train_report",
                "drug_classifier_val_predictions",
                "drug_classifier_val_report",
            ],
            name="train_and_evaluate_drug_classifier_node"
        ),
    ])
    
    # Pipeline 2: ATC Classifier — train + evaluate in one node (drugs only)
    atc_pipeline = pipeline([
        node(
            func=train_and_evaluate_atc_classifier,
            inputs=[
                "selected_lipinski_backbone",
                "X_train_drugs_only",
                "y_atc_train_drugs_only",
                "X_val_drugs_only",
                "y_atc_val_drugs_only",
                "n_atc_classes_drugs_only",
            ],
            outputs=[
                "atc_classifier_model",
                "atc_classifier_history",
                "atc_classifier_train_predictions",
                "atc_classifier_train_report",
                "atc_classifier_val_predictions",
                "atc_classifier_val_report",
            ],
            name="train_and_evaluate_atc_classifier_node"
        ),
    ])

    # Action-based and organ-based ATC branches — both taxonomies trained and
    # evaluated in the same run (against the same selected_lipinski_backbone)
    # for direct side-by-side comparison, reusing process_drug_data's
    # action_/organ_ prefixed datasets.
    def _taxonomy_atc_branch(prefix: str) -> Pipeline:
        return pipeline([
            node(
                func=filter_drugs_only,
                inputs=[f"{prefix}X_train", f"{prefix}y_drug_train", f"{prefix}y_atc_train"],
                outputs=[f"{prefix}X_train_drugs_only", f"{prefix}y_atc_train_drugs_only"],
                name=f"filter_train_drugs_{prefix}node"
            ),
            node(
                func=filter_drugs_only,
                inputs=[f"{prefix}X_val", f"{prefix}y_drug_val", f"{prefix}y_atc_val"],
                outputs=[f"{prefix}X_val_drugs_only", f"{prefix}y_atc_val_drugs_only"],
                name=f"filter_val_drugs_{prefix}node"
            ),
            node(
                func=get_num_atc_classes_drugs_only,
                inputs=[f"{prefix}y_atc_train_drugs_only"],
                outputs=f"{prefix}n_atc_classes_drugs_only",
                name=f"get_num_atc_classes_{prefix}node"
            ),
            node(
                func=create_atc_mapping_drugs_only,
                inputs=[f"{prefix}atc_mapping"],
                outputs=f"{prefix}atc_mapping_drugs_only",
                name=f"create_atc_mapping_drugs_only_{prefix}node"
            ),
            node(
                func=train_and_evaluate_atc_classifier,
                inputs=[
                    "selected_lipinski_backbone",
                    f"{prefix}X_train_drugs_only",
                    f"{prefix}y_atc_train_drugs_only",
                    f"{prefix}X_val_drugs_only",
                    f"{prefix}y_atc_val_drugs_only",
                    f"{prefix}n_atc_classes_drugs_only",
                ],
                outputs=[
                    f"{prefix}atc_classifier_model",
                    f"{prefix}atc_classifier_history",
                    f"{prefix}atc_classifier_train_predictions",
                    f"{prefix}atc_classifier_train_report",
                    f"{prefix}atc_classifier_val_predictions",
                    f"{prefix}atc_classifier_val_report",
                ],
                name=f"train_and_evaluate_atc_classifier_{prefix}node"
            ),
        ])

    action_atc_pipeline = _taxonomy_atc_branch("action_")
    organ_atc_pipeline = _taxonomy_atc_branch("organ_")

    # Combine all pipelines
    return (
        backbone_pipeline
        + data_prep_pipeline
        + drug_pipeline
        + atc_pipeline
        + action_atc_pipeline
        + organ_atc_pipeline
    )