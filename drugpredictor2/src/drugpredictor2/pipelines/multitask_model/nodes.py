import json
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report


# ====================================================
# Load pretrained Lipinski model and build multitask model
# ====================================================
def build_multitask_model(pretrained_model, n_atc_classes: int, input_dim: int):
    """Build multitask model using transfer learning from pretrained lipinski model."""
    
    # Create input layer for flat features
    input_layer = layers.Input(shape=(input_dim,))
    
    # Reshape to (input_dim, 1) for Conv1D (matching lipinski model format)
    x = layers.Reshape((input_dim, 1))(input_layer)
    
    # Apply pretrained Conv1D and intermediate layers (skip only final output Dense[-1])
    # Freeze pretrained layers for transfer learning
    for layer in pretrained_model.layers[:-1]:
        layer.trainable = False
        x = layer(x)
    
    # Add shared layer with unique names
    shared = layers.Dense(256, activation="relu", name="shared_dense")(x)
    shared = layers.Dropout(0.3, name="shared_dropout")(shared)

    # Head 1: drug vs non-drug
    drug_output = layers.Dense(1, activation="sigmoid", name="drug_output")(shared)

    # Head 2: ATC classification
    atc_output = layers.Dense(n_atc_classes, activation="softmax", name="atc_output")(shared)

    multitask_model = models.Model(
        inputs=input_layer, outputs=[drug_output, atc_output]
    )

    multitask_model.compile(
        optimizer="adam",
        loss={
            "drug_output": "binary_crossentropy",
            "atc_output": "categorical_crossentropy",
        },
        loss_weights={
            "drug_output": 1.0,
            "atc_output": 0.5,
        },
        metrics={
            "drug_output": ["accuracy"],
            "atc_output": ["accuracy"],
        },
    )

    return multitask_model


# ====================================================
# Train multitask model
# ====================================================
def train_multitask_model(
    pretrained_model,
    X_train,
    y_drug_train,
    y_atc_train,
    X_val,
    y_drug_val,
    y_atc_val,
    n_atc_classes: int
):
    """Train the multitask model using transfer learning."""
    input_dim = X_train.shape[1]  # Get feature dimension from data
    model = build_multitask_model(pretrained_model, n_atc_classes, input_dim)

    history = model.fit(
        X_train,
        {"drug_output": y_drug_train, "atc_output": y_atc_train},
        validation_data=(X_val, {"drug_output": y_drug_val, "atc_output": y_atc_val}),
        epochs=25,
        batch_size=256,
        verbose=1,
    )

    return model, history.history


# ====================================================
# Generate predictions and evaluation
# ====================================================
def evaluate_multitask_model(model, X, y_drug_true, y_atc_true):
    """Predict and generate classification reports."""
    drug_pred, atc_pred = model.predict(X)
    drug_pred_classes = (drug_pred > 0.5).astype(int)
    atc_pred_classes = np.argmax(atc_pred, axis=1)
    atc_true_classes = np.argmax(y_atc_true, axis=1)

    # Reports
    drug_report = classification_report(y_drug_true, drug_pred_classes, digits=4)
    atc_report = classification_report(atc_true_classes, atc_pred_classes, digits=4)

    # Predictions dataframe
    pred_df = pd.DataFrame({
        "drug_true": y_drug_true.flatten(),
        "drug_pred": drug_pred.flatten(),
        "atc_true": atc_true_classes,
        "atc_pred": atc_pred_classes,
    })

    return pred_df, drug_report, atc_report