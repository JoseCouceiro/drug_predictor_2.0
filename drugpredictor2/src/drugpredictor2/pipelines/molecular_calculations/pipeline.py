from kedro.pipeline import Pipeline, node, pipeline
from .nodes import get_model_input

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
      node(
        func=get_model_input,
        inputs=[
                "combined_csv_mock",
                "presplit_model_input_mock",
                "params:size_validation_data",
                "fp_tracker"
                ],
        outputs=['presplit_model_input_mock_updated',
                 'validation_dataset_mock',
                 'model_input_mock'],
        name="get_model_input_node",
        )
      ]
    )
