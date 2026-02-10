import json
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


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
# MODEL 1: Drug vs Non-Drug Binary Classifier
# ====================================================
def build_drug_classifier(pretrained_model, input_dim: int):
    """Build binary drug classifier using transfer learning from pretrained lipinski model."""
    
    # Create input layer for flat features
    input_layer = layers.Input(shape=(input_dim,))
    
    # Reshape to (input_dim, 1) for Conv1D (matching lipinski model format)
    x = layers.Reshape((input_dim, 1))(input_layer)
    
    # Apply pretrained layers: UNFREEZE ALL for full fine-tuning
    # Skip final output layer ([-1])
    for layer in pretrained_model.layers[:-1]:
        layer.trainable = True  # Unfreeze all layers for better learning
        x = layer(x)
    
    # Add BIGGER classification head - more capacity to learn
    x = layers.Dense(512, activation="relu", name="drug_dense_1")(x)
    x = layers.BatchNormalization(name="drug_bn_1")(x)
    x = layers.Dense(256, activation="relu", name="drug_dense_2")(x)
    x = layers.BatchNormalization(name="drug_bn_2")(x)
    x = layers.Dense(128, activation="relu", name="drug_dense_3")(x)
    x = layers.BatchNormalization(name="drug_bn_3")(x)
    
    # Output: binary classification (drug vs non-drug)
    output = layers.Dense(1, activation="sigmoid", name="drug_output")(x)

    model = models.Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),  # Higher LR for faster learning
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
    )

    return model


def train_drug_classifier(pretrained_model, X_train, y_train, X_val, y_val):
    """Train the binary drug classifier."""
    input_dim = X_train.shape[1]
    model = build_drug_classifier(pretrained_model, input_dim)

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
            monitor='val_accuracy',  # Monitor accuracy instead of loss
            patience=20,  # More patience
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        epochs=200,  # More epochs - let it learn!
        batch_size=64,  # Smaller batches for better gradients
        callbacks=callbacks,
        verbose=1,
    )

    return model, history.history


def evaluate_drug_classifier(model, X, y_true):
    """Evaluate drug classifier and generate classification report."""
    y_pred_prob = model.predict(X)
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
    
    Note: This model should ONLY be trained on drug samples (excluding ND/non-drugs).
    """
    
    # Create input layer for flat features
    input_layer = layers.Input(shape=(input_dim,))
    
    # Reshape to (input_dim, 1) for Conv1D (matching lipinski model format)
    x = layers.Reshape((input_dim, 1))(input_layer)
    
    # Apply pretrained layers: Unfreeze last 2 dense layers for fine-tuning
    # Skip final output layer ([-1])
    num_layers = len(pretrained_model.layers) - 1
    for i, layer in enumerate(pretrained_model.layers[:-1]):
        # Freeze Conv1D and first dense layers, unfreeze last 2 dense layers
        if i < num_layers - 2:
            layer.trainable = False
        else:
            layer.trainable = True  # Fine-tune last 2 layers
        x = layer(x)
    
    # Add classification head for ATC
    x = layers.Dense(512, activation="relu", name="atc_dense_1")(x)
    x = layers.Dropout(0.3, name="atc_dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="atc_dense_2")(x)
    x = layers.Dropout(0.2, name="atc_dropout_2")(x)
    
    # Output: multiclass classification (ATC categories)
    output = layers.Dense(n_atc_classes, activation="softmax", name="atc_output")(x)

    model = models.Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
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
    """Train the ATC classifier on drug samples only (no ND/non-drugs)."""
    input_dim = X_train.shape[1]
    model = build_atc_classifier(pretrained_model, n_atc_classes, input_dim)

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

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        epochs=50,
        batch_size=128,
        verbose=1,
    )

    return model, history.history


def evaluate_atc_classifier(model, X, y_true):
    """Evaluate ATC classifier and generate classification report."""
    y_pred_prob = model.predict(X)
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