import math
import numpy as np
import pandas as pd
from typing import Dict, Callable, Iterator, Optional, Tuple, List

from sklearn import metrics
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import layers, models, optimizers, regularizers
from keras_tuner import HyperParameters


# --------------------------
# Preprocessing
# --------------------------

def load_and_preprocess_fingerprints(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Keep rows with non-null fingerprints; cast to float32 for Keras Conv1D.
    """
    x_col = params["X_column"]
    # drop rows where the fingerprint is missing
    df = df.dropna(subset=[x_col]).copy()
    # ensure numpy arrays of dtype float32 (from bool/ints)
    df[x_col] = df[x_col].apply(lambda fp: np.asarray(fp, dtype=np.float32))
    return df


# --------------------------
# Index discovery & splits
# --------------------------

def _discover_valid_rows(partition_loaders: Dict[str, Callable[[], pd.DataFrame]], params: dict):
    """
    Return (valid_global_indices_sorted, valid_labels_series) where 'valid'
    means rows that survive the same preprocessing keep/drop rule.
    """
    x_col, y_col = params["X_column"], params["label"]

    labels = []
    keep_masks = []
    total = 0
    # dict preserves insertion order in py>=3.7
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
    params: dict,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified split on rows that will actually be used.
    Returns sorted lists of global indices.
    """
    val_size = params["val_size"]
    test_size = params["test_size"]

    valid_idx, valid_labels = _discover_valid_rows(partitioned_model_input, params)
    print(valid_labels.value_counts())

    train_val_idx, test_idx = train_test_split(
        valid_idx,
        test_size=test_size,
        stratify=valid_labels,
        random_state=25,
    )

    # labels for remaining (train+val)
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


# --------------------------
# Generators (deterministic)
# --------------------------

def _partition_spans(partition_loaders: Dict[str, Callable[[], pd.DataFrame]]):
    """Return list of (name, start, end) global spans per partition, in insertion order."""
    spans, total = [], 0
    for name, load in partition_loaders.items():
        n = len(load())
        spans.append((name, total, total + n))
        total += n
    return spans


def infinite_train_generator(
    partition_loaders: Dict[str, Callable[[], pd.DataFrame]],
    ordered_indices: List[int],
    params: Dict[str, str],
    batch_size: int,
    shuffle_each_epoch: bool = False,
    seed: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Infinite generator for training. Deterministic unless shuffling is enabled.
    If shuffle_each_epoch=True and seed is None, uses a fresh RNG each epoch.
    """
    x_col, y_col = params["X_column"], params["label"]
    spans = _partition_spans(partition_loaders)
    base_order = np.array(ordered_indices, dtype=int)

    # Create RNG once (don’t re-seed inside the loop)
    rng = np.random.default_rng(seed) if shuffle_each_epoch else None

    while True:
        order = base_order.copy()
        if rng is not None:
            rng.shuffle(order)

        for (name, start, end) in spans:
            mask = (order >= start) & (order < end)
            part_global = order[mask]
            if part_global.size == 0:
                continue

            local = np.sort(part_global - start)
            df = partition_loaders[name]()
            df = df.iloc[local]
            df = load_and_preprocess_fingerprints(df, params)
            if df.empty:
                continue

            X = np.stack(df[x_col].to_numpy()).astype(np.float32)
            y = df[y_col].to_numpy().astype(np.float32)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            for i in range(0, len(df), batch_size):
                yield X[i:i+batch_size], y[i:i+batch_size]


def repeating_eval_generator(
    partition_loaders: Dict[str, Callable[[], pd.DataFrame]],
    ordered_indices: List[int],
    params: Dict[str, str],
    batch_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Infinite (repeating) generator for validation/test. Deterministic order each cycle.
    Safe to pass to Keras with `validation_steps`.
    """
    x_col, y_col = params["X_column"], params["label"]
    spans = _partition_spans(partition_loaders)
    order = np.array(ordered_indices, dtype=int)

    while True:
        for (name, start, end) in spans:
            mask = (order >= start) & (order < end)
            part_global = order[mask]
            if part_global.size == 0:
                continue

            local = np.sort(part_global - start)
            df = partition_loaders[name]()
            df = df.iloc[local]
            df = load_and_preprocess_fingerprints(df, params)
            if df.empty:
                continue

            X = np.stack(df[x_col].to_numpy()).astype(np.float32)
            y = df[y_col].to_numpy().astype(np.float32)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            for i in range(0, len(df), batch_size):
                yield X[i:i+batch_size], y[i:i+batch_size]


# --------------------------
# Model
# --------------------------

def build_def_model_dynamic_Dense(hp, input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape, dtype="float32"),
        layers.Flatten(),
        layers.Dense(
            hp.Int('dense_1_units', 256, 512, step=64),
            activation=None,
            kernel_initializer='he_uniform'
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(hp.Float('dropout_1', 0.2, 0.4, step=0.1)),

        layers.Dense(
            hp.Int('dense_2_units', 128, 256, step=64),
            activation='leaky_relu',
            kernel_initializer='he_uniform',
            kernel_regularizer=regularizers.l2(1e-5)
        ),
        layers.Dropout(0.5),

        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# This is the model used in the final experiments
def build_def_model_dynamic(hp: HyperParameters, input_shape: tuple) -> keras.Sequential:
    model = models.Sequential([
        layers.Input(shape=input_shape, dtype="float32"),

        # Single Conv1D layer
        layers.Conv1D(
            filters=hp.Int('conv_filters', 32, 128, step=16),
            kernel_size=hp.Choice('conv_kernel', [3,5]),
            activation='relu',
            padding='valid'
        ),

        layers.Flatten(),

        # Dense layers
        layers.Dense(
            units=hp.Int('dense_1_units', 256, 512, step=64),
            activation='relu',
            kernel_initializer='he_uniform'
        ),
        layers.Dropout(hp.Float('dropout_1', 0.3, 0.5, step=0.05)),

        layers.Dense(
            units=hp.Int('dense_2_units', 128, 256, step=32),
            activation='relu',
            kernel_initializer='he_uniform'
        ),
        layers.Dropout(hp.Float('dropout_2', 0.3, 0.5, step=0.05)),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [1e-3, 5e-4])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# --------------------------
# Orchestration
# --------------------------

def train_model_on_partitions(
    partitioned_model_input: Dict[str, Callable[[], pd.DataFrame]],
    train_params: dict,
    split_params: dict,
):
    """
    Train + evaluate on partitioned data with correct splits and stable ordering.
    Expects split_params: {X_column, label, val_size, test_size}
            train_params: {epochs (int), batch_size (optional)}
    """
    batch_size = train_params['batch_size']

    # 1) Proper split on valid rows only (after keep/drop)
    train_idx, val_idx, test_idx = create_train_val_test_indices(
        partitioned_model_input, split_params
    )

    # 2) Generators
    train_gen = infinite_train_generator(
    partitioned_model_input, train_idx, split_params, batch_size, shuffle_each_epoch=True
    )
    val_gen = repeating_eval_generator(
    partitioned_model_input, val_idx, split_params, batch_size
    )

    # 3) Steps
    steps_per_epoch  = math.ceil(len(train_idx) / batch_size)
    validation_steps = math.ceil(len(val_idx)   / batch_size)

    # 4) Build model
    sample_X, _ = next(train_gen)
    in_shape = sample_X.shape[1:]
    hp = HyperParameters()
    model = build_def_model_dynamic(hp, in_shape)

    # 5) Train
    history = model.fit(
        train_gen,
        epochs=train_params['epochs'],
        steps_per_epoch=math.ceil(len(train_idx) / batch_size),
        validation_data=val_gen,
        validation_steps=math.ceil(len(val_idx) / batch_size),
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        verbose=1,
    )

    # 6) Evaluation (materialize once for perfect alignment)
    def _materialize(gen: Iterator[Tuple[np.ndarray, np.ndarray]], steps: int):
        Xb, yb = [], []
        for _ in range(steps):
            try:
                X, y = next(gen)
            except StopIteration:
                break
            Xb.append(X); yb.append(y)
        Xall = np.concatenate(Xb, axis=0) if Xb else np.empty((0,) + in_shape, dtype=np.float32)
        yall = np.concatenate(yb, axis=0) if yb else np.empty((0,), dtype=np.float32)
        return Xall, yall

    # Val
    val_gen_eval = repeating_eval_generator(
        partitioned_model_input, val_idx, split_params, batch_size
    )
    Xv, yv = _materialize(val_gen_eval, validation_steps)
    yv_pred = (model.predict(Xv, verbose=0) > 0.5).astype(int).ravel()
    n = min(len(yv), len(yv_pred))
    val_predictions = pd.DataFrame({"y_true": yv[:n], "y_pred": yv_pred[:n]})
    val_report = metrics.classification_report(yv[:n], yv_pred[:n])

    # Test
    test_steps = math.ceil(len(test_idx) / batch_size)
    test_gen_eval = repeating_eval_generator(
        partitioned_model_input, test_idx, split_params, batch_size
    )
    Xt, yt = _materialize(test_gen_eval, test_steps)
    yt_pred = (model.predict(Xt, verbose=0) > 0.5).astype(int).ravel()
    n2 = min(len(yt), len(yt_pred))
    test_predictions = pd.DataFrame({"y_true": yt[:n2], "y_pred": yt_pred[:n2]})
    test_report = metrics.classification_report(yt[:n2], yt_pred[:n2])

    return (
        model,
        history.history,
        test_predictions,
        test_report,
        val_predictions,
        val_report,
    )
