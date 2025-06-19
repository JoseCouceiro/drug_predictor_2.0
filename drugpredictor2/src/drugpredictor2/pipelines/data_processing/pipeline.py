"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import sdf_to_csv, concat_dataframes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
            [
                node(
                    func=sdf_to_csv,
                    inputs=["sdf_folder",
                            "csv_folder",
                            'tracker'],
                    outputs="tracker_updated",
                    name="sdf_to_csv_node",
                ),
                node(
                    func=concat_dataframes,
                    inputs="csv_folder",
                    outputs="combined_csv",
                    name="concat_dataframes_node",
                )            
            ]
        )

