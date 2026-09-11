from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
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
                "lipinski_model",
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
                "lipinski_model",
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
    
    # Combine all pipelines
    return data_prep_pipeline + drug_pipeline + atc_pipeline