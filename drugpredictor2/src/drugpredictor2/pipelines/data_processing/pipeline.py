from kedro.pipeline import Pipeline, node, pipeline
from .nodes import update_dataframe


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
            [
                node(
                    func=update_dataframe,
                    inputs=["combined_csv",
                            "sdf_folder",
                            "tracker"],
                    outputs=["combined_csv_updated", "tracker_updated"],
                    name="update_dataframe_node"
                )  
            ]
        )

