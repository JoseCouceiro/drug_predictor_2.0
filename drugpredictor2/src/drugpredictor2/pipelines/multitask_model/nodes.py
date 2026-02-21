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
    
    Removes all non-drug samples (ND class) from the dataset.
    Also removes the ND column (class 1) from the one-hot encoded y_atc.
    
    Returns:
        X_drugs: Features for drug samples only
        y_atc_drugs: ATC labels for drug samples only (ND column removed, 16 classes instead of 17)
    """
    # Find indices where is_drug == 1
    drug_indices = np.where(y_drug.flatten() == 1)[0]
    
    X_drugs = X[drug_indices]
    y_atc_temp = y_atc[drug_indices]
    
    print(f"Filtered {len(drug_indices)} drug samples from {len(X)} total samples")
    print(f"Original y_atc shape: {y_atc_temp.shape}")
    
    # Remove ND column (index 1) from one-hot encoded y_atc
    # ND is encoded as class 1, so we remove column 1
    y_atc_drugs = np.delete(y_atc_temp, 1, axis=1)
    
    print(f"After removing ND class: {y_atc_drugs.shape}")
    print(f"Class distribution after filtering:")
    class_counts = np.sum(y_atc_drugs, axis=0)
    for i, count in enumerate(class_counts):
        # Adjust index for display (since we removed index 1)
        original_idx = i if i < 1 else i + 1
        print(f"  Class {original_idx}: {int(count)} samples")
    
    return X_drugs, y_atc_drugs


def get_num_atc_classes_drugs_only(y_atc_train_drugs_only):
    """Get number of ATC classes after removing ND."""
    return y_atc_train_drugs_only.shape[1]


def create_atc_mapping_drugs_only(atc_mapping):
    """Create updated ATC mapping after removing ND class.
    
    Args:
        atc_mapping: Original ATC mapping DataFrame with columns [ATC_Code, Encoded_Label]
    
    Returns:
        DataFrame with updated mapping excluding ND (clean encoding 0-15)
    """
    # Remove ND from mapping
    atc_mapping_drugs = atc_mapping[atc_mapping['ATC_Code'] != 'ND'].copy()
    
    # Create new encoding that reflects the removal of ND (index 1)
    new_labels = []
    for old_label in atc_mapping_drugs['Encoded_Label']:
        if old_label < 1:
            # N (0) stays at 0
            new_labels.append(old_label)
        else:
            # Everything after ND shifts down by 1
            new_labels.append(old_label - 1)
    
    atc_mapping_drugs['Encoded_Label'] = new_labels
    
    return atc_mapping_drugs[['ATC_Code', 'Encoded_Label']]


# ====================================================
# MODEL 1: Drug vs Non-Drug Binary Classifier (Transfer Learning)
# ====================================================
def build_drug_classifier(pretrained_model, input_dim: int):
    """Build binary drug classifier using transfer learning from lipinski model.
    
    Only the Conv1D fingerprint-reader is transferred (frozen).  All Dense
    layers are trained from scratch — the drug/non-drug decision surface is
    very different from the lipinski compliance surface.
    """
    input_layer, x = _clone_conv1d_backbone(pretrained_model, input_dim)

    # Fresh Dense stack trained from scratch
    x = layers.Dense(512, activation="relu", name="drug_dense_1")(x)
    x = layers.BatchNormalization(name="drug_bn_1")(x)
    x = layers.Dropout(0.4, name="drug_dropout_1")(x)

    x = layers.Dense(256, activation="relu", name="drug_dense_2")(x)
    x = layers.BatchNormalization(name="drug_bn_2")(x)
    x = layers.Dropout(0.3, name="drug_dropout_2")(x)

    x = layers.Dense(128, activation="relu", name="drug_dense_3")(x)
    x = layers.Dropout(0.3, name="drug_dropout_3")(x)

    output = layers.Dense(1, activation="sigmoid", name="drug_output")(x)

    model = models.Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
    )

    return model


def train_drug_classifier(pretrained_model, X_train, y_train, X_val, y_val):
    """Train the binary drug classifier with transfer learning from lipinski."""
    # Free any leftover GPU memory from previous runs
    keras.backend.clear_session()
    gc.collect()

    input_dim = X_train.shape[1]
    model = build_drug_classifier(pretrained_model, input_dim)
    
    print("="*60)
    print("TRAINING DRUG CLASSIFIER (transfer learning from lipinski)")
    print("="*60)
    # Summary showing frozen vs trainable layers
    trainable = sum(l.count_params() for l in model.layers if l.trainable)
    total = model.count_params()
    print(f"Total params: {total:,} | Trainable: {trainable:,} "
          f"({100*trainable/total:.1f}%) | Frozen: {total-trainable:,}")

    # Report class distribution
    print(f"Drug class distribution - Train: {np.bincount(y_train.flatten())}")
    print(f"Drug class distribution - Val: {np.bincount(y_val.flatten())}")
    
    # Compute class weights for balanced training
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train.flatten()
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Drug class weights: {class_weight_dict}")

    # Add callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
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
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history.history


def evaluate_drug_classifier(model, X, y_true):
    """Evaluate drug classifier and generate classification report."""
    y_pred_prob = model.predict(X, batch_size=256)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Classification report
    report = classification_report(y_true, y_pred, digits=4)

    # Predictions dataframe
    pred_df = pd.DataFrame({
        "true": y_true.flatten(),
        "pred_prob": y_pred_prob.flatten(),
        "pred": y_pred.flatten(),
    })

    return pred_df, report


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

    # Fresh Dense stack trained from scratch
    x = layers.Dense(512, activation="relu", name="atc_dense_1")(x)
    x = layers.BatchNormalization(name="atc_bn_1")(x)
    x = layers.Dropout(0.4, name="atc_dropout_1")(x)

    x = layers.Dense(256, activation="relu", name="atc_dense_2")(x)
    x = layers.BatchNormalization(name="atc_bn_2")(x)
    x = layers.Dropout(0.3, name="atc_dropout_2")(x)

    x = layers.Dense(128, activation="relu", name="atc_dense_3")(x)
    x = layers.Dropout(0.3, name="atc_dropout_3")(x)

    output = layers.Dense(n_atc_classes, activation="softmax", name="atc_output")(x)

    model = models.Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    return model


def train_atc_classifier(
    pretrained_model, 
    X_train, 
    y_train, 
    X_val, 
    y_val, 
    n_atc_classes: int
):
    """Train the ATC classifier on drug samples only (no ND/non-drugs).
    
    Uses transfer learning from the lipinski pretrained backbone.
    """
    # Free GPU memory from drug classifier before building ATC model
    keras.backend.clear_session()
    gc.collect()

    input_dim = X_train.shape[1]
    model = build_atc_classifier(pretrained_model, n_atc_classes, input_dim)

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
            monitor='val_accuracy',
            patience=20,
            mode='max',
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
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history.history


def evaluate_atc_classifier(model, X, y_true):
    """Evaluate ATC classifier and generate classification report."""
    y_pred_prob = model.predict(X, batch_size=256)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)

    # Classification report
    report = classification_report(y_true_classes, y_pred, digits=4)

    # Predictions dataframe
    pred_df = pd.DataFrame({
        "true": y_true_classes,
        "pred": y_pred,
    })
    
    # Add prediction probabilities for each class
    for i in range(y_pred_prob.shape[1]):
        pred_df[f"prob_class_{i}"] = y_pred_prob[:, i]

    return pred_df, report