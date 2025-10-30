from kedro.pipeline import Pipeline, node, pipeline
# Import both new node functions
from .nodes import get_existing_fnames, process_new_partitions

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_existing_fnames,
            inputs="params:featurized_data_path", 
            outputs="existing_partitions_set",
            name="get_existing_partitions_node",
        ),
        node(
            func=process_new_partitions,
            inputs=["raw_csv_files", "existing_partitions_set"],
            outputs="featurized_data",
            name="process_new_partitions_node",
        )
    ])