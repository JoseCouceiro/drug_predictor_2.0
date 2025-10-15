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
    model_input[x_col] = model_input[x_col].apply(lambda fp: fp.astype(np.float32))

    return model_input

def _discover_valid_rows(partition_loaders: Dict[str, Callable[[], pd.DataFrame]],
                         params: dict):
    """Return (valid_global_indices_sorted, valid_labels_series) after applying the same
    'row keep' criterion used in preprocessing (i.e., non-null fingerprints)."""
    x_col = params["X_column"]
    y_col = params["label"]

    labels = []
    keep_masks = []
    total = 0
    for load in partition_loaders.values():
        df = load()
        labels.append(df[y_col])
        keep_masks.append(df[x_col].notna())
        total += len(df)

    labels = pd.concat(labels, ignore_index=True)
    keep = pd.concat(keep_masks, ignore_index=True)

    valid_idx = np.flatnonzero(keep.values)
    valid_labels = labels.iloc[valid_idx].reset_index(drop=True)
    return valid_idx, valid_labels

def create_train_val_test_indices(
    partitioned_model_input: Dict[str, Callable[[], pd.DataFrame]],
    params: dict
) -> tuple[list[int], list[int], list[int]]:
    """
    Build stratified train/val/test splits **on rows that will actually be used**
    (i.e., after preprocessing drops). Returns sorted lists of global indices.
    Required keys in params: 'label', 'val_size', 'test_size'.
    """
    val_size = params["val_size"]
    test_size = params["test_size"]

    valid_idx, valid_labels = _discover_valid_rows(partitioned_model_input, params)

    train_val_idx, test_idx = train_test_split(
        valid_idx,
        test_size=test_size,
        stratify=valid_labels,
        random_state=42,
    )

    # labels for the remaining portion to stratify val
    # Map train_val_idx to positions within valid_idx:
    pos_in_valid = np.searchsorted(valid_idx, np.sort(train_val_idx))
    remain_labels = valid_labels.iloc[pos_in_valid]

    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=rel_val,
        stratify=remain_labels,
        random_state=42,
    )

    train_idx = sorted(train_idx.tolist())
    val_idx   = sorted(val_idx.tolist())
    test_idx  = sorted(test_idx.tolist())

    print(f"Dataset split (valid rows only): Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return train_idx, val_idx, test_idx


# --- Deterministic generators: accept ordered lists, keep order stable ---

def partitioned_data_generator(
    partition_loaders: Dict[str, Callable[[], pd.DataFrame]],
    ordered_indices: list[int],
    params: Dict[str, str],
    batch_size: int,
    shuffle_each_epoch: bool = False,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Infinite generator for training. Keeps a deterministic order unless shuffle_each_epoch=True.
    """
    x_col, y_col = params["X_column"], params["label"]

    # Precompute spans for each partition in dict insertion order (stable in Py3.7+)
    spans = []
    total = 0
    for name, load in partition_loaders.items():
        n = len(load())
        spans.append((name, total, total + n))
        total += n

    base_order = np.array(ordered_indices, dtype=int)

    while True:
        order = base_order.copy()
        if shuffle_each_epoch:
            rng = np.random.default_rng(42)
            rng.shuffle(order)

        for (name, start, end) in spans:
            # pick the part of 'order' inside this partition
            mask = (order >= start) & (order < end)
            part_global = order[mask]
            if part_global.size == 0:
                continue

            local = np.sort(part_global - start)

            df = partition_loaders[name]()
            df = df.iloc[local]
            df = load_and_preprocess_fingerprints(df.copy(), params)
            if df.empty:
                continue

            # verify all fingerprints are same length; stack for numeric array
            X = np.stack(df[x_col].to_numpy())
            y = df[y_col].to_numpy()
            X = X.reshape((X.shape[0], X.shape[1], 1))

            for i in range(0, len(df), batch_size):
                yield X[i:i+batch_size], y[i:i+batch_size]


def finite_data_generator(
    partition_loaders: Dict[str, Callable[[], pd.DataFrame]],
    ordered_indices: list[int],
    params: Dict[str, str],
    batch_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Single-pass generator for validation/test with deterministic, stable order.
    """
    x_col, y_col = params["X_column"], params["label"]

    spans = []
    total = 0
    for name, load in partition_loaders.items():
        n = len(load())
        spans.append((name, total, total + n))
        total += n

    order = np.array(ordered_indices, dtype=int)

    for (name, start, end) in spans:
        mask = (order >= start) & (order < end)
        part_global = order[mask]
        if part_global.size == 0:
            continue

        local = np.sort(part_global - start)

        df = partition_loaders[name]()
        df = df.iloc[local]
        df = load_and_preprocess_fingerprints(df.copy(), params)
        if df.empty:
            continue

        X = np.stack(df[x_col].to_numpy())
        y = df[y_col].to_numpy()
        X = X.reshape((X.shape[0], X.shape[1], 1))

        for i in range(0, len(df), batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]


# --- Orchestration (key fixes: splits, steps, evaluation alignment) ---

def train_model_on_partitions(
    partitioned_model_input: Dict[str, Callable[[], pd.DataFrame]],
    train_params: dict,
    split_params: dict,
    tune_params: dict,
    batch_size: int = 128
) -> tuple:
    """
    Orchestrates training & evaluation with memory-efficient generators on partitioned data.
    Expects split_params to contain: 'X_column', 'label', 'val_size', 'test_size'.
    """
    # 1) Proper split on valid rows only
    train_indices, val_indices, test_indices = create_train_val_test_indices(
        partitioned_model_input, split_params
    )

    # 2) Generators
    train_gen = partitioned_data_generator(
        partitioned_model_input, train_indices, split_params, batch_size, shuffle_each_epoch=True
    )
    val_gen = finite_data_generator(
        partitioned_model_input, val_indices, split_params, batch_size
    )

    # 3) Steps
    steps_per_epoch  = math.ceil(len(train_indices) / batch_size)
    validation_steps = math.ceil(len(val_indices)   / batch_size)   # <-- FIXED

    # 4) Build model (use an explicit Input layer to silence Keras warning)
    sample_X, _ = next(train_gen)
    in_shape = sample_X.shape[1:]

    hp = HyperParameters()
    model = build_def_model_dynamic(hp, in_shape)

    # 5) Train
    history = model.fit(
        train_gen,
        epochs=train_params.get("epochs", 200),
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        verbose=1
    )

    # --- Evaluation helpers: materialize once for perfect alignment ---

    def _materialize(generator: Iterator[Tuple[np.ndarray, np.ndarray]], steps: int):
        Xb, yb = [], []
        for _ in range(steps):
            try:
                X, y = next(generator)
            except StopIteration:
                break
            Xb.append(X); yb.append(y)
        Xall = np.concatenate(Xb, axis=0) if Xb else np.empty((0,)+in_shape)
        yall = np.concatenate(yb, axis=0) if yb else np.empty((0,))
        return Xall, yall

    # Validation eval
    val_gen_eval = finite_data_generator(
        partitioned_model_input, val_indices, split_params, batch_size
    )
    Xv, yv = _materialize(val_gen_eval, validation_steps)
    yv_pred = (model.predict(Xv, verbose=0) > 0.5).astype(int).ravel()
    n = min(len(yv), len(yv_pred))
    val_predictions = pd.DataFrame({"y_true": yv[:n], "y_pred": yv_pred[:n]})
    val_report = metrics.classification_report(yv[:n], yv_pred[:n])

    # Test eval
    test_steps = math.ceil(len(test_indices) / batch_size)
    test_gen_eval = finite_data_generator(
        partitioned_model_input, test_indices, split_params, batch_size
    )
    Xt, yt = _materialize(test_gen_eval, test_steps)
    yt_pred = (model.predict(Xt, verbose=0) > 0.5).astype(int).ravel()
    n2 = min(len(yt), len(yt_pred))
    test_predictions = pd.DataFrame({"y_true": yt[:n2], "y_pred": yt_pred[:n2]})
    test_report = metrics.classification_report(yt[:n2], yt_pred[:n2])

    return (
        model,
        history.history,
        test_predictions,   # name them test_* to avoid confusion
        test_report,
        val_predictions,
        val_report,
    )


# --- Model builder (optional tidy-up): use explicit Input to avoid warning ---

def build_def_model_dynamic(hp: HyperParameters, input_shape: tuple) -> Sequential:
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv1D(
            filters=hp.Int('conv_1_filter', min_value=16, max_value=128, step=16),
            kernel_size=hp.Choice('conv_1_kernel', values=[3,5]),
            activation='relu',
            padding='valid'
        ),
        layers.MaxPool1D(hp.Int('pool_size', min_value=2, max_value=6)),
        layers.Dropout(0.25) if hp.Boolean("dropout") else layers.Layer(),
        layers.Flatten(),
        layers.Dense(
            units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
            activation='relu',
            kernel_initializer='he_uniform'
        ),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
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