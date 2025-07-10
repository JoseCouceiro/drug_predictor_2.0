from kedro.pipeline import Pipeline, node, pipeline
from .nodes import process_inputs, apply_lipinski_and_prepare_for_saving


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=process_inputs,
                inputs=["sdf_folder", "tracker"],
                outputs=["processed_sdf_dataframes", "tracker_updated"],
                name="process_new_sdf_files_node"
            ),
            node(
                func=apply_lipinski_and_prepare_for_saving,
                inputs="processed_sdf_dataframes",
                outputs="processed_and_lipinski_dataframes",
                name="apply_lipinski_and_prepare_node"
            )
        ]
    )

