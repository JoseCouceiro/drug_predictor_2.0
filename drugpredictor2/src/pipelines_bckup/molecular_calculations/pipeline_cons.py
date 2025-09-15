from kedro.pipeline import Pipeline, node, pipeline
from .nodes_cons import process_single_lipinski_batch_to_fingerprints, create_empty_fingerprint_dataframe

# Import the new fingerprinting node
from .nodes import (
    process_single_lipinski_batch_to_fingerprints,
    create_empty_fingerprint_dataframe # For initial empty Parquet
)

def create_pipeline(**kwargs) -> Pipeline:
    # Define the first part of the pipeline (raw SDF -> Lipinski batches)
    # This part should be from your previous successful pipeline modification.


    # Define the second part of the pipeline (Lipinski batches -> Fingerprint batches)
    fingerprinting_pipeline = pipeline([
        # Initialize an empty Parquet for training fingerprints (if it doesn't exist)
        node(
            func=create_empty_fingerprint_dataframe,
            inputs=None,
            outputs="training_fingerprints_accumulated@partitioned_fingerprints", # Use a named dataset
            name="create_initial_train_fp_file",
        ),
        # Initialize an empty Parquet for validation fingerprints (if it doesn't exist)
        node(
            func=create_empty_fingerprint_dataframe,
            inputs=None,
            outputs="validation_fingerprints_accumulated@partitioned_fingerprints",
            name="create_initial_val_fp_file",
        ),

        # Node to iterate over each Lipinski batch and process its fingerprints
        # IMPORTANT: Kedro's PartitionedDataSet handles iterating through its partitions
        # when it's consumed by a node. If you want to process *all* batches,
        # you need a node that consumes the *entire PartitionedDataSet*.
        #
        # However, to save them individually, a single node cannot return a dynamically
        # changing number of outputs. We need to create a "mapper" or a way to
        # trigger individual runs for each file.

        # A common pattern for this in Kedro for many-to-many is to define a "template"
        # node that runs for each partition.

        # Let's create a "template" node for fingerprinting one file:
        node(
            func=process_single_lipinski_batch_to_fingerprints,
            inputs={
                "lipinski_batch_df": "processed_and_lipinski_dataframes.raw_data_input", # This is the template for each partition
                "batch_name": "processed_and_lipinski_dataframes.id", # Kedro injects the partition ID
            },
            outputs="fingerprinted_batches.fingerprinted_data_output", # This is the output template
            name="process_fingerprints_for_single_batch",
        ),

        # Now, how do we perform the train/validation split with these individual files?
        # This is where the 'split before fingerprinting' strategy is key.
        # If your initial `combined_csv` split into `training_set_raw` and `validation_set_raw`
        # (from previous discussion), then you'd apply the lipinski_processing_pipeline
        # and fingerprinting_pipeline *separately* for train and validation.

        # Example of how that would look (assuming initial split of raw data happened):
        # raw_train_files_to_lipinski_batches = pipeline([...], namespace="train_raw_to_lipinski")
        # raw_val_files_to_lipinski_batches = pipeline([...], namespace="val_raw_to_lipinski")

        # lipinski_train_batches_to_fingerprints = pipeline([
        #     node(
        #         func=process_single_lipinski_batch_to_fingerprints,
        #         inputs={
        #             "lipinski_batch_df": "lipinski_batches_creation.processed_and_lipinski_dataframes_train.raw_data_input",
        #             "batch_name": "lipinski_batches_creation.processed_and_lipinski_dataframes_train.id",
        #         },
        #         outputs="training_fingerprints_accumulated@partitioned_fingerprints", # This will save to the train partition
        #         name="process_fingerprints_for_training_batch",
        #     ),
        # ], namespace="train_fingerprinting")

        # lipinski_val_batches_to_fingerprints = pipeline([
        #     node(
        #         func=process_single_lipinski_batch_to_fingerprints,
        #         inputs={
        #             "lipinski_batch_df": "lipinski_batches_creation.processed_and_lipinski_dataframes_val.raw_data_input",
        #             "batch_name": "lipinski_batches_creation.processed_and_lipinski_dataframes_val.id",
        #         },
        #         outputs="validation_fingerprints_accumulated@partitioned_fingerprints", # This will save to the val partition
        #         name="process_fingerprints_for_validation_batch",
        #     ),
        # ], namespace="val_fingerprinting")
        
        # --- Let's stick to the current scenario where we fingerprint all, then deal with split ---
        # The output `fingerprinted_batches` will be a PartitionedDataSet containing ALL fingerprinted files.

    ], namespace="fingerprinting_processing") # Namespace for fingerprinting part


    # Combine all parts of the pipeline
    return lipinski_processing_pipeline + fingerprinting_pipeline

    # If you had the initial raw data splitting:
    # return (
    #     raw_data_splitter_pipeline + # A pipeline that splits raw_data into raw_train, raw_val
    #     lipinski_processing_pipeline_train + # Processes raw_train into lipinski_train_batches
    #     lipinski_processing_pipeline_val + # Processes raw_val into lipinski_val_batches
    #     fingerprinting_pipeline_train + # Processes lipinski_train_batches into training_fingerprints_accumulated
    #     fingerprinting_pipeline_val # Processes lipinski_val_batches into validation_fingerprints_accumulated
    # )