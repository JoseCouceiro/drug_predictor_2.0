from kedro.pipeline import Pipeline, node, pipeline
from .nodes import obtain_trained_model, visualize_training

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
        node(
          func=obtain_trained_model,
          inputs=["model_input",
                  "validation_dataset",
                  "params:X_y_split",
                  "params:tune_model"],
          outputs=["def_model",
                   "history",
                   "train_predictions",
                   "train_classification_report",
                   "validation_predictions",
                   "validation_classification_report"],
          name="obtain_trained_model_node",   
        ),
        node(
          func=visualize_training,
          inputs='history',
          outputs='training_fig',
          name='visualization_node'
        )
      ]
    )
