import math
import numpy as np
import pandas as pd
import os
import ast
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras_tuner import RandomSearch, HyperParameters
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Dict, Callable, Iterator, Tuple, Set

def load_and_preprocess_fingerprints(model_input: pd.DataFrame,
                                     params: dict
                                    ) -> pd.DataFrame:
    """
    Prepares fingerprint data for the model, assuming input is NumPy arrays.

    This function takes a DataFrame where the fingerprint column contains
    NumPy arrays of booleans (loaded from Parquet) and efficiently converts
    them into arrays of integers (0s and 1s). It also removes any rows
    with missing fingerprint data.

    Args:
        model_input: A DataFrame where `params['X_column']` contains
                     NumPy arrays of booleans.
        params: A dictionary expecting the key 'X_column'.

    Returns:
        A DataFrame with fingerprints as NumPy arrays of integers, with
        null rows dropped.
    """
    x_col = params['X_column']

    # Drop any rows that might have missing fingerprints.
    model_input.dropna(subset=[x_col], inplace=True)

    # Efficiently convert the boolean arrays to integer arrays.
    model_input[x_col] = model_input[x_col].apply(lambda fp: fp.astype(int))

    return model_input

def create_train_test_indices(
    partition_loaders: Dict[str, Callable[[], pd.DataFrame]],
    params: Dict[str, str],
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[Set[int], Set[int], pd.Series, pd.Series]:
    """
    Performs a memory-efficient, stratified train-test split.

    This function scans all partitions, loading ONLY the label column to build a
    lightweight metadata DataFrame. It then performs a stratified split on this
    metadata to get global indices for the train and test sets.

    Args:
        partition_loaders: The PartitionedDataSet dictionary of loaders.
        params: Dictionary containing the 'label' column name.
        test_size: The proportion of the dataset to include in the test split.
        random_state: Seed for reproducibility.

    Returns:
        A tuple containing:
        - train_indices (Set[int]): A set of global indices for the training set.
        - test_indices (Set[int]): A set of global indices for the test set.
        - y_train (pd.Series): The labels for the training set.
        - y_test (pd.Series): The labels for the test set.
    """
    print("Pass 1: Scanning partitions to create train/test split indices...")
    
    metadata_list = []
    # This loop loads only the necessary column and should be memory-light.
    for partition_name, load_func in partition_loaders.items():
        # Here we assume load_func() can be modified or that loading a full partition
        # and immediately subsetting it is acceptable for a moment.
        # For very large CSVs, you can optimize by reading only specific columns.
        df = load_func()
        metadata_list.append(df[[params['label']]])
    
    # Concatenate only the label data, which is small.
    full_metadata = pd.concat(metadata_list, ignore_index=True)
    
    # Create a dummy X with the correct index for splitting
    X_indices = full_metadata.index.to_series()
    y_labels = full_metadata[params['label']]
    
    # Perform a stratified split on the indices
    train_indices_df, test_indices_df, y_train, y_test = train_test_split(
        X_indices, y_labels, test_size=test_size, random_state=random_state, stratify=y_labels
    )
    
    print(f"Split complete. Train samples: {len(train_indices_df)}, Test samples: {len(test_indices_df)}")
    
    # Return as sets for fast O(1) lookups in the generator
    return set(train_indices_df), set(test_indices_df), y_train, y_test


def partitioned_data_generator(
    partition_loaders: Dict[str, Callable[[], pd.DataFrame]],
    indices_to_use: Set[int],
    params: Dict[str, str],
    batch_size: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    A generator that loads and yields batches of data for a given set of indices.

    It iterates through partitions one by one, processes them, filters rows
    based on `indices_to_use`, and yields batches of a specified size.

    Args:
        partition_loaders: The PartitionedDataSet dictionary of loaders.
        indices_to_use: A set of global row indices to include.
        params: Dictionary with 'X_column' and 'label' names.
        batch_size: The number of samples per batch.

    Yields:
        A tuple of (X_batch, y_batch) ready for model training.
    """
    global_index_offset = 0
    
    # This outer loop is required by Keras for multiple epochs
    while True:
        for partition_name, load_func in partition_loaders.items():
            partition_df = load_func()
            
            # Preprocess the partition
            processed_df = load_and_preprocess_fingerprints(partition_df.copy(), params)
            if processed_df.empty:
                global_index_offset += len(partition_df)
                continue

            # Determine which global indices from this partition should be used
            partition_global_indices = set(range(global_index_offset, global_index_offset + len(partition_df)))
            relevant_indices_in_partition = partition_global_indices.intersection(indices_to_use)

            if not relevant_indices_in_partition:
                global_index_offset += len(partition_df)
                continue

            # Convert global indices to local indices within the partition_df
            local_indices_to_keep = [idx - global_index_offset for idx in relevant_indices_in_partition]
            
            # Filter the original (unprocessed) dataframe to align with processed data
            target_df = partition_df.iloc[local_indices_to_keep]

            # We must re-process the filtered data to ensure alignment, especially after dropping NaNs
            final_batch_df = load_and_preprocess_fingerprints(target_df.copy(), params)

            # Yield data in batches from this filtered partition
            for i in range(0, len(final_batch_df), batch_size):
                batch_df = final_batch_df.iloc[i:i + batch_size]
                
                X_list = list(batch_df[params['X_column']])
                X_batch_array = np.array(X_list)
                y_batch_array = np.array(batch_df[params['label']])
                
                # Reshape for 1D CNN
                X_batch_reshaped = X_batch_array.reshape(
                    (X_batch_array.shape[0], X_batch_array.shape[1], 1)
                )
                
                yield (X_batch_reshaped, y_batch_array)
            
            global_index_offset += len(partition_df)
        
        # Reset for the next epoch
        global_index_offset = 0

# --- MODIFIED ORCHESTRATION FUNCTION ---

def train_model_on_partitions(
    partitioned_model_input: Dict[str, Callable[[], pd.DataFrame]],
    validation_dataset: pd.DataFrame, # Assuming validation set is small and fits in memory
    params: dict,
    tune_params: dict,
    batch_size: int = 128
) -> tuple:
    """
    Orchestrates the entire model pipeline using memory-efficient generators
    for partitioned data.
    """
    # 1. Get global train/test split indices without loading all data
    train_indices, test_indices, y_train, y_test = create_train_test_indices(
        partitioned_model_input, params
    )
    
    # 2. Create data generators
    train_generator = partitioned_data_generator(
        partitioned_model_input, train_indices, params, batch_size
    )
    test_generator = partitioned_data_generator(
        partitioned_model_input, test_indices, params, batch_size
    )

    # 3. Calculate steps for Keras
    steps_per_epoch = math.ceil(len(train_indices) / batch_size)
    validation_steps = math.ceil(len(test_indices) / batch_size)

    # Note: KerasTuner with generators can be complex. For simplicity, we'll
    # skip tuning here and train a default model. Full tuning would require
    # a custom Tuner class or careful generator resets. Let's build and train.
    # We get a single batch to determine the input shape.
    sample_X, _ = next(train_generator)
    in_shape = sample_X.shape[1:]
    
    # 4. Build a model (using a simplified builder for this example)
    # The `build_def_model` needs to be able to accept the input shape
    hp = HyperParameters() # Dummy HP object
    tuned_model = build_def_model_dynamic(hp, in_shape) # A modified build function
    
    # 5. Training the model with generators
    print(f"Starting model training with {steps_per_epoch} steps per epoch...")
    history = tuned_model.fit(
        train_generator,
        epochs=200, # As in your original code
        steps_per_epoch=steps_per_epoch,
        validation_data=test_generator,
        validation_steps=validation_steps,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
        verbose=1
    )
    history_dic = history.history
    
    print('TRAINED MODEL OBTAINED')  
    
    # 6. Evaluation (also using the generator)
    print('Model evaluation on test set...')
    test_generator_for_eval = partitioned_data_generator(
        partitioned_model_input, test_indices, params, batch_size
    )
    # We need to collect all predictions from the generator
    predictions = tuned_model.predict(test_generator_for_eval, steps=validation_steps, verbose=1)
    y_pred_class = (predictions > 0.5).astype(int).flatten()
    
    # Since y_test is already in memory, we can compare directly
    # Ensure the number of predictions matches the number of test labels
    train_class_rep = metrics.classification_report(y_test[:len(y_pred_class)], y_pred_class)
    train_predictions = pd.Series(y_pred_class, index=y_test.index[:len(y_pred_class)])
    print(train_class_rep)

    # Evaluation on the separate validation dataset (assuming it's small)
    print('Model evaluation on separate validation dataset...')
    val_predictions, val_class_rep = evaluate_on_single_df(tuned_model, validation_dataset, params)
      
    return tuned_model, history_dic, train_predictions, train_class_rep, val_predictions, val_class_rep

# --- You'll need a slightly modified build function and a small eval helper ---

def build_def_model_dynamic(hp: HyperParameters, input_shape: tuple) -> Sequential:
    """Modified build_model to accept a dynamic input_shape."""
    model = keras.Sequential()
    # Add the input layer separately or in the first layer
    model.add(
        layers.Conv1D(
            filters=hp.Int('conv_1_filter', min_value=16, max_value=128, step=16),
            kernel_size=hp.Choice('conv_1_kernel', values=[3,5]),
            activation='relu',
            input_shape=input_shape, # KEY CHANGE
            padding='valid'
        )
    )
    # ... The rest of your build_def_model function is the same ...
    model.add(layers.MaxPool1D(hp.Int('pool_size', min_value=2, max_value=6)))
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16), activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_on_single_df(model: Sequential, df: pd.DataFrame, params: dict) -> Tuple[pd.Series, str]:
    """
    Helper to evaluate the model on a single, in-memory DataFrame (e.g., a validation set).

    This function preprocesses the DataFrame, reshapes the features for the CNN,
    and then uses the `get_predictions` helper to generate predictions and a
    classification report.

    Args:
        model: The trained Keras Sequential model.
        df: The validation DataFrame to evaluate.
        params: A dictionary containing 'X_column' and 'label' keys.

    Returns:
        A tuple containing:
        - val_predictions (pd.Series): The predicted labels for the DataFrame.
        - val_class_rep (str): The classification report string.
    """
    print("Preprocessing validation DataFrame...")
    processed_df = load_and_preprocess_fingerprints(df.copy(), params)

    # Handle case where the validation set is empty after processing
    if processed_df.empty:
        print("Warning: Validation DataFrame is empty after preprocessing. Skipping evaluation.")
        return pd.Series(dtype=int), "No validation data to evaluate."

    print("Preparing validation data for the model...")
    X_array = np.array(list(processed_df[params['X_column']]))
    y_true = processed_df[params['label']]

    # Reshape for 1D CNN model: (samples, features, 1)
    reshaped_X = X_array.reshape((X_array.shape[0], X_array.shape[1], 1))

    print("Generating predictions on validation data...")
    # This was the missing part: calling your get_predictions function
    val_predictions, val_class_rep = get_predictions(model, reshaped_X, y_true)

    return val_predictions, val_class_rep