from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_test_split_column

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
        node(
            func=train_test_split_column,
            inputs=["model_input",
                    "params:columns",
                    "params:columns"],
            outputs=None,
            name="train_test_split_node",
        )
      ]
    )
