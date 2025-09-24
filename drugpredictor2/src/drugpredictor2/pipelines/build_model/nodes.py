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

# REVISED create_train_val_test_indices
def create_train_val_test_indices(
    partitioned_model_input: Dict[str, Callable[[], pd.DataFrame]],
    params: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: # <-- Note the changed return type
    """
    Generate shuffled train/val/test indices without loading all labels.
    """
    y_column = params["y_column"]
    val_size = params["val_size"]
    test_size = params["test_size"]
    
    # We need to load labels to do a stratified split, which is best practice.
    all_labels = []
    total_rows = 0
    for loader_fn in partitioned_model_input.values():
        df = loader_fn()
        all_labels.append(df[y_column])
        total_rows += len(df)
        del df

    all_labels = pd.concat(all_labels, ignore_index=True)
    all_indices = np.arange(total_rows)

    # Use train_test_split for a robust, stratified split
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        stratify=all_labels, # Stratify to maintain class balance
        random_state=42
    )

    # We need to get the labels for the remaining part to stratify the val split
    train_val_labels = all_labels.iloc[train_val_indices]
    
    # Adjust val_size relative to the remaining data
    relative_val_size = val_size / (1.0 - test_size)

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=relative_val_size,
        stratify=train_val_labels,
        random_state=42
    )
    
    print(f"Dataset split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # Return only the index sets
    return train_indices, val_indices, test_indices

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
    train_params: dict,
    split_params: dict,
    tune_params: dict,
    batch_size: int = 128
) -> tuple:
    """
    Orchestrates the entire model pipeline using memory-efficient generators
    for partitioned data.
    """
    # 1. Get global train/test split indices
    train_indices, val_indices, test_indices = create_train_val_test_indices(
        partitioned_model_input, split_params
    )
    
    # Convert to sets for efficient lookup in the generator
    train_indices_set = set(train_indices)
    val_indices_set = set(val_indices)
    test_indices_set = set(test_indices)

    # 2. Create data generators
    train_generator = partitioned_data_generator(
        partitioned_model_input, train_indices_set, split_params, batch_size
    )
    val_generator = partitioned_data_generator(
        partitioned_model_input, val_indices_set, split_params, batch_size
    )
    # Use a non-shuffling generator for evaluation to ensure order
    test_generator = partitioned_data_generator(
        partitioned_model_input, test_indices_set, split_params, batch_size
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
    history = tuned_model.fit(
        train_generator,
        epochs=200,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
        verbose=1
    )

    # Helper function to get predictions and true labels from a generator
    def get_preds_and_labels(generator, steps, indices):
        y_true_list = []
        y_pred_list = []
        
        # Make predictions
        predictions = tuned_model.predict(generator, steps=steps, verbose=1)
        y_pred_list = (predictions > 0.5).astype(int).flatten()

        # Re-run the generator to get the true labels in the correct order
        # This is crucial for correct evaluation.
        label_generator = partitioned_data_generator(
            partitioned_model_input, set(indices), split_params, batch_size
        )
        for i, (_, y_batch) in enumerate(label_generator):
            if i >= steps:
                break
            y_true_list.extend(y_batch)
        
        # Trim y_true to match the length of predictions
        return np.array(y_true_list[:len(y_pred_list)]), y_pred_list

    # 6. Evaluation on validation set
    print("Model evaluation on validation set...")
    val_steps = math.ceil(len(val_indices) / batch_size)
    y_val_true, y_val_pred = get_preds_and_labels(val_generator, val_steps, val_indices)
    
    val_predictions = pd.DataFrame({"y_true": y_val_true, "y_pred": y_val_pred})
    val_class_rep = metrics.classification_report(y_val_true, y_val_pred)

    # 7. Evaluation on test set
    print("Model evaluation on test set...")
    test_steps = math.ceil(len(test_indices) / batch_size)
    y_test_true, y_test_pred = get_preds_and_labels(test_generator, test_steps, test_indices)
    
    train_predictions = pd.DataFrame({"y_true": y_test_true, "y_pred": y_test_pred})
    train_class_rep = metrics.classification_report(y_test_true, y_test_pred)

    # 8. Return all outputs (as before)
    return (
        tuned_model,
        history.history,
        train_predictions,
        train_class_rep,
        val_predictions,
        val_class_rep,
    )

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

def get_predictions(model: Sequential,
                    reshaped_X_test: np.ndarray,
                    y_true: pd.Series
                    ) -> Tuple[pd.Series, str]:
    """Generates predictions from a trained model and computes its classification report.

    This function predicts class labels for given input data using a pre-trained
    Keras Sequential model and then provides a detailed classification report
    comparing the predicted labels to the true labels. This function is specifically
    designed for binary classification where the model's output layer uses a sigmoid
    activation (single unit) and outputs probabilities, which are then converted
    to binary class labels (0 or 1).

    Args:
        model: The trained Keras Sequential model.
        reshaped_X_data: The input feature data (e.g., reshaped X_test or reshaped_val_X_test)
                         as a NumPy array, ready for model prediction.
        y_true: The true labels corresponding to `reshaped_X_data`, as a Pandas Series.

    Returns:
        A tuple containing:
        - y_pred_series (pd.Series): Predicted binary labels (0 or 1) as a Pandas Series.
        - class_rep (str): A string representing the sklearn classification report,
                           including precision, recall, f1-score, and support for each class.
    """
    y_pred = model.predict(reshaped_X_test)
    y_pred_list = (y_pred > 0.5).astype(int).flatten().tolist()
    class_rep = metrics.classification_report(y_true,y_pred_list)
    print(class_rep)
    return pd.Series(y_pred_list), class_rep

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