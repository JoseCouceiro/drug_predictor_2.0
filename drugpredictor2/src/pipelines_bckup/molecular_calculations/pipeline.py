from kedro.pipeline import Pipeline, node, pipeline
from .nodes import process_batch_to_fingerprints, create

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
      node(
        func=get_model_input,
        inputs=[
                "combined_csv",
                "presplit_model_input",
                "params:size_validation_data",
                "fp_tracker"
                ],
        outputs=[
                "presplit_model_input_updated",
                "validation_dataset",
                "model_input",
                "fp_tracker_updated"
                ],
        name="get_model_input_node",
        )
      ]
    )
