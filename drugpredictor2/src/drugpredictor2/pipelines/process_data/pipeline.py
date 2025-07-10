from kedro.pipeline import Pipeline, node, pipeline
from .nodes import process_single_raw_sdf_to_fingerprints

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=process_single_raw_sdf_to_fingerprints,
            inputs="raw_sdf_files",
            outputs="featurized_data",
            name="process_sdf_node",
        )
    ])