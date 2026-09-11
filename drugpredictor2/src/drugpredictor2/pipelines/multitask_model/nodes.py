import gc
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


# ====================================================
# GPU Memory Growth — prevent TF from pre-allocating the entire GPU
# ====================================================
_gpus = tf.config.list_physical_devices("GPU")
for _gpu in _gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass  # must be set before GPUs are initialised


# ====================================================
# Transfer Learning Helpers
# ====================================================
def _clone_conv1d_backbone(pretrained_model, input_dim):
    """Clone ONLY the Conv1D + Flatten layers from the pretrained lipinski model.
    
    The Conv1D layer is a general "fingerprint reader" trained on hundreds of
    thousands of molecules — it learned to parse molecular fingerprints, which
    is a transferable skill.  The Dense layers above it are lipinski-specific
    (charge / size / hydrophobicity thresholds) and should NOT be transferred.
    
    The Conv1D is frozen; each downstream task builds its own Dense stack.
    
    Args:
        pretrained_model: Trained lipinski Keras model (Sequential Conv1D).
        input_dim: Number of flat input features (e.g. 6288).
    
    Returns:
        (input_layer, x) — Keras tensors ready for a new Dense head.
    """
    input_layer = layers.Input(shape=(input_dim,), name="tl_input")
    x = layers.Reshape((input_dim, 1), name="tl_reshape")(input_layer)

    # Find the Conv1D and Flatten layers only
    source_layers = [
        l for l in pretrained_model.layers
        if not isinstance(l, layers.InputLayer)
    ]

    for src_layer in source_layers:
        is_conv = isinstance(src_layer, layers.Conv1D)
        is_flatten = isinstance(src_layer, layers.Flatten)
        if not (is_conv or is_flatten):
            continue  # skip Dense, Dropout, BN — those are lipinski-specific

        config = src_layer.get_config()
        config["name"] = f"tl_{config['name']}"
        new_layer = src_layer.__class__.from_config(config)
        x = new_layer(x)

        if src_layer.get_weights():
            new_layer.set_weights(src_layer.get_weights())

        # Freeze the Conv1D — it's already a good fingerprint reader
        new_layer.trainable = False

        if is_flatten:
            break  # stop after Flatten — everything above is task-specific

    return input_layer, x


# ====================================================
# Data Preparation: Filter drugs for ATC classifier
# ====================================================
def filter_drugs_only(X, y_drug, y_atc):
    """Filter dataset to keep only drug samples (is_drug=1) for ATC training.

    Also drops the 'ND' (non-drug) one-hot column at index 1, but ONLY if it
    is actually empty after row filtering. When `atc_subset` already excludes
    ND upstream (in process_drug_dataset), index 1 is a real class and must
    be kept — hardcoding its removal silently drops a real class.

    Returns:
        X_drugs: Features for drug samples only
        y_atc_drugs: ATC labels for drug samples only (ND column dropped if empty)
    """
    # Find indices where is_drug == 1
    drug_indices = np.where(y_drug.flatten() == 1)[0]
    
    X_drugs = X[drug_indices]
    y_atc_temp = y_atc[drug_indices]
    
    print(f"Filtered {len(drug_indices)} drug samples from {len(X)} total samples")
    print(f"Original y_atc shape: {y_atc_temp.shape}")

    # Only remove column 1 if it is genuinely unused (the ND placeholder),
    # not when it's a real class (e.g. atc_subset runs with ND already excluded).
    if y_atc_temp.shape[1] > 1 and y_atc_temp[:, 1].sum() == 0:
        y_atc_drugs = np.delete(y_atc_temp, 1, axis=1)
        print(f"Removed empty ND column (index 1): {y_atc_drugs.shape}")
    else:
        y_atc_drugs = y_atc_temp
        print("Column 1 has real samples — keeping all columns (no ND to remove).")

    print(f"Class distribution after filtering:")
    class_counts = np.sum(y_atc_drugs, axis=0)
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {int(count)} samples")
    
    return X_drugs, y_atc_drugs


def get_num_atc_classes_drugs_only(y_atc_train_drugs_only):
    """Get number of ATC classes after removing ND."""
    return y_atc_train_drugs_only.shape[1]


def create_atc_mapping_drugs_only(atc_mapping):
    """Create ATC mapping matching the columns produced by filter_drugs_only.

    Drops the 'ND' row only if present (full-dataset runs); for atc_subset
    runs ND is already absent upstream, so the mapping passes through
    unchanged. Encoded_Label is always renumbered 0..n-1 in the same sorted
    order as the original labels, matching the one-hot column order from
    to_categorical (and the column dropped by filter_drugs_only).

    Args:
        atc_mapping: Original ATC mapping DataFrame with columns [ATC_Code, Encoded_Label]

    Returns:
        DataFrame with updated mapping excluding ND (if present).
    """
    atc_mapping_drugs = atc_mapping[atc_mapping['ATC_Code'] != 'ND'].copy()
    atc_mapping_drugs = atc_mapping_drugs.sort_values('Encoded_Label').reset_index(drop=True)
    atc_mapping_drugs['Encoded_Label'] = range(len(atc_mapping_drugs))

    return atc_mapping_drugs[['ATC_Code', 'Encoded_Label']]


# ====================================================
# MODEL 1: Drug vs Non-Drug Binary Classifier (Transfer Learning)
# ====================================================
def build_drug_classifier(pretrained_model, input_dim: int):
    """Build binary drug classifier using transfer learning from lipinski model.

    Wider head than before; backbone starts frozen for phase-1 warmup and is
    unfrozen in phase 2 inside train_and_evaluate_drug_classifier.
    """
    input_layer, x = _clone_conv1d_backbone(pretrained_model, input_dim)

    # First Dense is the bottleneck on the large flattened Conv1D output; keep
    # it at 512 to avoid OOM (402k-unit flatten × 1024 ≈ 1.6 GB of weights).
    x = layers.Dense(512, activation="relu",
                     name="drug_dense_1")(x)
    x = layers.BatchNormalization(name="drug_bn_1")(x)
    x = layers.Dropout(0.5, name="drug_dropout_1")(x)

    x = layers.Dense(256, activation="relu",
                     name="drug_dense_2")(x)
    x = layers.BatchNormalization(name="drug_bn_2")(x)
    x = layers.Dropout(0.4, name="drug_dropout_2")(x)

    x = layers.Dense(128, activation="relu",
                     name="drug_dense_3")(x)
    x = layers.BatchNormalization(name="drug_bn_3")(x)
    x = layers.Dropout(0.4, name="drug_dropout_3")(x)

    x = layers.Dense(64, activation="relu",
                     name="drug_dense_4")(x)
    x = layers.Dropout(0.3, name="drug_dropout_4")(x)

    output = layers.Dense(1, activation="sigmoid", name="drug_output")(x)

    model = models.Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=keras.losses.BinaryFocalCrossentropy(gamma=3.0),
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall(),
                 keras.metrics.AUC(name="auroc")],
    )

    return model


def _evaluate_drug_classifier(model, X, y_true, threshold=0.5):
    """Evaluate drug classifier and generate classification report (internal helper)."""
    y_pred_prob = model.predict(X, batch_size=32)
    y_pred = (y_pred_prob > threshold).astype(int)

    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    pred_df = pd.DataFrame({
        "true": y_true.flatten(),
        "pred_prob": y_pred_prob.flatten(),
        "pred": y_pred.flatten(),
    })

    return pred_df, report


def train_and_evaluate_drug_classifier(pretrained_model, X_train, y_train, X_val, y_val):
    """Train the binary drug classifier and evaluate on train+val sets.

    Single-phase training with focal loss and AUC-guided early stopping.
    A second model.compile() causes OOM because Adam allocates duplicate
    momentum tensors while the first set is still live in GPU memory.
    Prediction threshold is optimised on the validation set to maximise macro F1.
    """
    keras.backend.clear_session()
    gc.collect()

    input_dim = X_train.shape[1]
    model = build_drug_classifier(pretrained_model, input_dim)
    del pretrained_model
    gc.collect()

    print("="*60)
    print("TRAINING DRUG CLASSIFIER (transfer learning from lipinski)")
    print("="*60)
    total = model.count_params()
    trainable = sum(l.count_params() for l in model.layers if l.trainable)
    print(f"Total params: {total:,} | Trainable: {trainable:,} "
          f"({100*trainable/total:.1f}%) | Frozen: {total-trainable:,}")
    print(f"Drug class distribution - Train: {np.bincount(y_train.flatten())}")
    print(f"Drug class distribution - Val: {np.bincount(y_val.flatten())}")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train.flatten()
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"Drug class weights: {class_weight_dict}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        epochs=200,
        batch_size=64,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_auroc', patience=12, mode='max',
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1
            ),
        ],
        verbose=1,
    )

    # ---------- Threshold optimisation on validation ----------
    val_probs = model.predict(X_val, batch_size=32).flatten()
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.90, 0.02):
        preds = (val_probs > t).astype(int)
        f1 = classification_report(
            y_val.flatten(), preds, output_dict=True, zero_division=0
        )['macro avg']['f1-score']
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(t)
    print(f"\nOptimal threshold: {best_thresh:.2f} (val macro F1={best_f1:.4f})")

    print("\n" + "="*60)
    print("EVALUATING DRUG CLASSIFIER")
    print("="*60)
    train_pred_df, train_report = _evaluate_drug_classifier(
        model, X_train, y_train, threshold=best_thresh
    )
    print("Train report:\n", train_report)
    val_pred_df, val_report = _evaluate_drug_classifier(
        model, X_val, y_val, threshold=best_thresh
    )
    print("Val report:\n", val_report)
    val_report = f"Optimal threshold: {best_thresh:.2f}\n{val_report}"

    return (
        model,
        history.history,
        train_pred_df,
        train_report,
        val_pred_df,
        val_report,
    )


# ====================================================
# MODEL 2: ATC Multi-class Classifier (for drugs only)
# ====================================================
def build_atc_classifier(pretrained_model, n_atc_classes: int, input_dim: int):
    """Build ATC classifier using transfer learning from pretrained lipinski model.
    
    Only the Conv1D fingerprint-reader is transferred (frozen).  Fresh Dense
    layers are trained from scratch with label smoothing to handle the
    imbalanced 16-class ATC problem.
    
    Note: This model should ONLY be trained on drug samples (excluding ND/non-drugs).
    """
    input_layer, x = _clone_conv1d_backbone(pretrained_model, input_dim)

    # Smaller head: further reduced to 128→64 to cut overparameterisation.
    x = layers.Dense(128, activation="relu", name="atc_dense_1")(x)
    x = layers.BatchNormalization(name="atc_bn_1")(x)
    x = layers.Dropout(0.6, name="atc_dropout_1")(x)

    x = layers.Dense(64, activation="relu", name="atc_dense_2")(x)
    x = layers.BatchNormalization(name="atc_bn_2")(x)
    x = layers.Dropout(0.5, name="atc_dropout_2")(x)

    output = layers.Dense(n_atc_classes, activation="softmax", name="atc_output")(x)

    model = models.Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss=keras.losses.CategoricalFocalCrossentropy(gamma=1.5, label_smoothing=0.15),
        metrics=["accuracy"],
    )

    return model


def _evaluate_atc_classifier(model, X, y_true):
    """Evaluate ATC classifier and generate classification report (internal helper)."""
    y_pred_prob = model.predict(X, batch_size=32)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)

    report = classification_report(y_true_classes, y_pred, digits=4)

    pred_df = pd.DataFrame({
        "true": y_true_classes,
        "pred": y_pred,
    })
    for i in range(y_pred_prob.shape[1]):
        pred_df[f"prob_class_{i}"] = y_pred_prob[:, i]

    return pred_df, report


def train_and_evaluate_atc_classifier(
    pretrained_model, 
    X_train, 
    y_train, 
    X_val, 
    y_val, 
    n_atc_classes: int
):
    """Train the ATC classifier on drug samples only and evaluate on train+val.

    Training and evaluation happen in a single node to avoid pickle round-trip
    OOM issues.
    """
    # Free GPU memory from drug classifier before building ATC model
    keras.backend.clear_session()
    gc.collect()

    input_dim = X_train.shape[1]
    model = build_atc_classifier(pretrained_model, n_atc_classes, input_dim)

    # Free the pretrained model from GPU — backbone weights are already copied
    del pretrained_model
    gc.collect()

    print("="*60)
    print("TRAINING ATC CLASSIFIER (transfer learning from lipinski)")
    print("="*60)
    trainable = sum(l.count_params() for l in model.layers if l.trainable)
    total = model.count_params()
    print(f"Total params: {total:,} | Trainable: {trainable:,} "
          f"({100*trainable/total:.1f}%) | Frozen: {total-trainable:,}")

    # Report class distribution
    y_train_classes = np.argmax(y_train, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    print(f"ATC class distribution - Train: {np.bincount(y_train_classes)}")
    print(f"ATC class distribution - Val: {np.bincount(y_val_classes)}")
    
    # Compute class weights for balanced training
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_classes),
        y=y_train_classes
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights_array)}
    print(f"ATC class weights: {class_weight_dict}")

    # Callbacks for proper training control
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        epochs=200,
        batch_size=32,  # restored: ATC runs in isolation so drug classifier memory is not present
        callbacks=callbacks,
        verbose=1,
    )

    # --- Evaluate in the same node (model is still in GPU memory) ---
    print("\n" + "="*60)
    print("EVALUATING ATC CLASSIFIER")
    print("="*60)
    train_pred_df, train_report = _evaluate_atc_classifier(model, X_train, y_train)
    print("Train report:\n", train_report)

    val_pred_df, val_report = _evaluate_atc_classifier(model, X_val, y_val)
    print("Val report:\n", val_report)

    return (
        model,
        history.history,
        train_pred_df,
        train_report,
        val_pred_df,
        val_report,
    )