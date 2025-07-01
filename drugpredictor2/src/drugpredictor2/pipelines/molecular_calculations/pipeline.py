from kedro.pipeline import Pipeline, node, pipeline
from .nodes import get_model_input

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
      node(
        func=get_model_input,
        inputs=["combined_csv", "params:size_validation_data"],
        outputs=['validation_dataset', 'model_input'],
        name="get_model_input_node",
        )
      ]
    )
