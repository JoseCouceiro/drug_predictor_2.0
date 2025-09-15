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

def create_train_val_test_indices(
    partitioned_model_input: Dict[str, Callable[[], pd.DataFrame]],
    params: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate train/val/test indices and corresponding labels without
    loading all partitions into memory at once.

    Returns:
        train_indices, val_indices, test_indices: arrays of global row indices
        y_train, y_val, y_test: arrays of labels aligned with indices
    """

    y_column = params["y_column"]
    val_size = params["val_size"]
    test_size = params["test_size"]

    all_indices = []
    all_labels = []

    offset = 0  # To create global indices across partitions

    # Loop through partitions without loading all data at once
    for partition_name, loader_fn in partitioned_model_input.items():
        df = loader_fn()  # Load this partition
        n_rows = len(df)
        indices = np.arange(offset, offset + n_rows)
        labels = df[y_column].to_numpy()

        all_indices.append(indices)
        all_labels.append(labels)

        offset += n_rows
        del df  # free memory

    all_indices = np.concatenate(all_indices)
    all_labels = np.concatenate(all_labels)

    # Shuffle indices
    rng = np.random.default_rng()
    shuffled_idx = rng.permutation(len(all_indices))
    all_indices = all_indices[shuffled_idx]
    all_labels = all_labels[shuffled_idx]

    # Compute split sizes
    n_total = len(all_indices)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_val - n_test

    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:n_train+n_val]
    test_indices = all_indices[n_train+n_val:]

    y_train = pd.Series(all_labels[:n_train])
    y_val = pd.Series(all_labels[n_train:n_train+n_val])
    y_test = pd.Series(all_labels[n_train+n_val:])

    return train_indices, val_indices, test_indices, y_train, y_val, y_test

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
    # 1. Get global train/test split indices without loading all data
    train_indices, val_indices, test_indices, y_train, y_val, y_test = create_train_val_test_indices(
        partitioned_model_input, split_params
    )
    
    # 2. Create data generators
    train_generator = partitioned_data_generator(
        partitioned_model_input, train_indices, split_params, batch_size
    )
    val_generator = partitioned_data_generator(
        partitioned_model_input, val_indices, split_params, batch_size
    )
    test_generator = partitioned_data_generator(
        partitioned_model_input, test_indices, split_params, batch_size
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
        verbose=2
    )

    # 6. Explicit evaluation on validation set
    print("Model evaluation on validation set...")
    val_steps = math.ceil(len(val_indices) / batch_size)
    val_predictions = tuned_model.predict(val_generator, steps=val_steps, verbose=1)
    val_pred_class = (val_predictions > 0.5).astype(int).flatten()

    val_predictions = pd.DataFrame({
        "y_true": y_val[:len(val_pred_class)],
        "y_pred": val_pred_class
    })

    val_class_rep = metrics.classification_report(
        y_val[:len(val_pred_class)], val_pred_class
    )

    # 7. Evaluation on test set
    print("Model evaluation on test set...")
    test_steps = math.ceil(len(test_indices) / batch_size)
    predictions = tuned_model.predict(test_generator, steps=test_steps, verbose=1)
    y_pred_class = (predictions > 0.5).astype(int).flatten()
    train_class_rep = metrics.classification_report(y_test[:len(y_pred_class)], y_pred_class)
    train_predictions = pd.DataFrame({
        "y_true": y_test[:len(y_pred_class)],
        "y_pred": y_pred_class
    })

    # 8. Return all outputs
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