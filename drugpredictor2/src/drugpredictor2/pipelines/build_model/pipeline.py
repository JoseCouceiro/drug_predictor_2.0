from kedro.pipeline import Pipeline, node, pipeline
from .nodes import obtain_trained_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
        node(
            func=obtain_trained_model,
            inputs=["model_input",
                    "params:X_y_split",
                    "params:tune_model"],
            outputs=["def_model", "history"],
            name="obtain_trained_model",
        )
      ]
    )
