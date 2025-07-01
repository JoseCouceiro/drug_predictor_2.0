import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras_tuner import RandomSearch, HyperParameters
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Dict, Any

# Functions to prepare data input"
def train_test_split_column(model_input: pd.DataFrame,
                            params: dict
                            )-> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits a specified column of a DataFrame into training and testing sets.

    This function takes a Pandas DataFrame and extracts a feature column (X_column)
    and a label column (label) based on provided parameters. It then performs a
    stratified train-test split on these selected columns. The 'X_column' is
    expected to contain list-like data (e.g., fingerprints), which are expanded
    into a 2D DataFrame for the split.

    Args:
        model_input: The input Pandas DataFrame containing the features and labels.
        params: A dictionary of parameters, expected to contain:
            - 'X_column' (str): The name of the column to be used as features (X values).
                               This column should contain list-like data (e.g., molecular fingerprints).
            - 'label' (str): The name of the column to be used as the target (y values).

    Returns:
        A tuple containing four elements:
        - X_train (pd.DataFrame): Training features.
        - X_test (pd.DataFrame): Testing features.
        - y_train (pd.Series): Training labels.
        - y_test (pd.Series): Testing labels.

    Raises:
        KeyError: If 'X_column' or 'label' are not found in the `params` dictionary
                  or `model_input` DataFrame.
    """
    X = model_input[params['X_column']]
    y = model_input[params['label']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def reshape_input(X_train: pd.DataFrame,
                  X_test: pd.DataFrame,
                  y_train: pd.Series
                  ) -> tuple[np.ndarray, np.ndarray, Tuple[int,int], int]:
    """Reshapes train/test feature data and determines class information for CNN input.

    This function prepares feature data (X_train, X_test) for use with 1D Convolutional
    Neural Networks by converting them to NumPy arrays and adding a channel dimension.
    It also calculates the number of unique classes present in the training labels (y_train)
    and determines the appropriate input shape for the CNN model.

    Args:
        X_train: Training features, expected as a Pandas DataFrame where each row
                 represents a sample and columns represent features (e.g., fingerprint bits).
        X_test: Testing features, expected as a Pandas DataFrame.
        y_train: Training labels, expected as a Pandas Series. Used to determine
                 the number of unique classes.

    Returns:
        A tuple containing four elements:
        - reshaped_X_train (np.ndarray): Reshaped training features.
        - reshaped_X_test (np.ndarray): Reshaped testing features.
        - in_shape (Tuple[int, int]): The expected input shape for the 1D CNN model.
        - n_classes (int): The number of unique classes found in `y_train`.
    """
    X_train_array = np.array(list(X_train))
    X_test_array = np.array(list(X_test))

    n_classes = len(np.unique(y_train))
    
    # Reshape for 1D CNN model: (samples, features, 1)
    reshaped_X_train= X_train_array.reshape((X_train_array.shape[0], X_train_array.shape[1], 1))
    reshaped_X_test= X_test_array.reshape((X_test_array.shape[0], X_test_array.shape[1], 1))

    # Determine in_shape for the CNN input layer (excluding batch size)
    in_shape = reshaped_X_train.shape[1:]
    
    return reshaped_X_train, reshaped_X_test, in_shape, n_classes

def build_def_model(hp: HyperParameters) -> Sequential:
    """Builds a 1D Convolutional Neural Network model using KerasTuner's HyperParameters
    for binary classification.

    This function defines the architecture of a 1D CNN model, allowing for
    hyperparameter tuning through the provided `hp` (HyperParameters) object.
    It constructs a sequential model with configurable convolutional layers,
    max pooling, dropout, and dense layers, specifically tailored for
    binary classification tasks.

    Args:
        hp: An instance of `keras_tuner.HyperParameters` which allows for defining
            hyperparameter search spaces (e.g., number of layers, filter sizes,
            kernel sizes, learning rate, dropout rates).

    Returns:
        A `tensorflow.keras.models.Sequential` model compiled with Adam optimizer,
        binary crossentropy loss, and accuracy metrics, optimized for two classes.

    Note:
        The input shape for the Conv1D layer is hardcoded to (2048, 1) as per the
        current implementation. The output layer is configured for binary
        classification (2 classes) using a single unit with a sigmoid activation
        and `binary_crossentropy` loss. Ensure these match your data characteristics.
    """
    # Create model object
    model = keras.Sequential()
    # Choose number of layers
    for i in range(hp.Int("num_layers", 1, 5)):
        model.add(
            layers.Conv1D(
        filters=hp.Int('conv_1_filter',
                       min_value=16,
                       max_value=128,
                       step=16),
        kernel_size=hp.Choice('conv_1_kernel',
                              values = [3,5]),
        activation='relu',
        input_shape=(2048, 1),
        padding='valid')) #no padding
        model.add(layers.MaxPool1D(
                hp.Int('pool_size',
                       min_value=2,
                       max_value=6)
                       ))
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('dense_1_units',
                     min_value=32,
                     max_value=128,
                     step=16),
        activation='relu',
        kernel_initializer = 'he_uniform'
        ))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    # Compilation of model
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
        loss='binary_crossentropy',
        metrics=['accuracy']
        )
    return model


def tune_hp_def_model(model_data: tuple[np.ndarray, np.ndarray, Tuple[int,int], int],
                      y_train: pd.Series,
                      y_test: pd.Series,
                      tune_params: dict
                      ) -> Sequential:
    """Tunes the hyperparameters of a 1D CNN model using KerasTuner's RandomSearch.

    This function sets up and runs a hyperparameter optimization search for the
    `build_def_model` using a RandomSearch algorithm. It trains and validates
    different model configurations based on the specified tuning parameters
    and returns the best-performing model found.

    Args:
        model_data: A tuple containing the reshaped data for training and testing:
            - reshaped_X_train (np.ndarray): Training features.
            - reshaped_X_test (np.ndarray): Testing features.
            - in_shape (Tuple[int, ...]): Input shape for the CNN (not directly used here but part of the tuple).
            - n_classes (int): Number of unique classes (not directly used here but part of the tuple).
        y_train: Training labels, a Pandas Series.
        y_test: Testing labels, a Pandas Series.
        tune_params: A dictionary of tuning parameters for KerasTuner, expected to contain:
            - 'objective' (str or `kerastuner.Objective`): The objective to optimize (e.g., 'val_accuracy' or 'val_loss').
            - 'max_trials' (int): The total number of trials (model configurations) to test during the search.
            - 'date' (str): A string representing the current date, used to create a unique directory for tuner logs.

    Returns:
        A `tensorflow.keras.models.Sequential` object representing the best model
        found during the hyperparameter tuning process.

    Raises:
        ValueError: If the `tune_params` dictionary is missing required keys.
    """
    reshaped_X_train, reshaped_X_test, _, _, = model_data
    
    tuner = RandomSearch(
        build_def_model,
        objective=tune_params['objective'],
        max_trials =tune_params['max_trials'],
        directory = os.path.join('temp', 'tuner', tune_params['date']))
    
    tuner.search(reshaped_X_train,y_train,epochs=3,validation_data=(reshaped_X_test,y_test))
    
    tuned_model = tuner.get_best_models(num_models=1)[0]
    tuned_model.summary() 
    
    return tuned_model

def train_def_model(tuned_model: Sequential,
                    model_data: tuple[np.ndarray, np.ndarray, Tuple[int,int], int],
                    y_train: pd.Series,
                    y_test: pd.Series
                    ) -> tuple[Sequential, Dict[str,Any]]:
    """Trains the best-tuned 1D CNN model and evaluates its performance.

    This function takes a previously tuned Keras Sequential model and trains it
    using the provided training data. It incorporates EarlyStopping and ModelCheckpoint
    callbacks for robust training and saving of the best model weights. After training,
    it evaluates the model on the test set.

    Args:
        tuned_model: The Keras `Sequential` model obtained from hyperparameter tuning.
        model_data: A tuple containing the reshaped data:
            - reshaped_X_train (np.ndarray): Training features.
            - reshaped_X_test (np.ndarray): Testing features.
            - Other elements (in_shape, n_classes) are present but not directly used here.
        y_train: Training labels, a Pandas Series.
        y_test: Testing labels, a Pandas Series.

    Returns:
        A tuple containing:
        - def_model (Sequential): The trained Keras Sequential model.
        - history_dic (Dict[str, Any]): A dictionary containing the training history (loss, accuracy, etc.).

    Note:
        Model checkpoints are saved to `temp/compiled_model/checkpoints/`.
        Ensure this directory exists or is created before running.
    """
    reshaped_X_train, reshaped_X_test, _, _ = model_data

    # Configure early stopping
    es = EarlyStopping(monitor='val_loss', patience=10)
    mc = ModelCheckpoint(
        filepath = os.path.join('temp', 'compiled_model', 'checkpoints', '{epoch:02d}-{val_loss:.3f}.hdf5'),
        monitor = 'val_loss',
        save_best_only = True)
    
    # Fit the model
    tuned_model.fit(reshaped_X_train,
                    y_train,
                    epochs=200,
                    batch_size=128,
                    verbose=1,
                    validation_split = 0.3,
                    callbacks = [es,mc])
    history_dic = tuned_model.history.history
    
    # Evaluate the model
    tuned_model.evaluate(reshaped_X_test, y_test, verbose=1)
    return tuned_model, history_dic

def obtain_trained_model(model_input: pd.DataFrame,
                         validation_dataset: pd.DataFrame,
                         params: dict,
                         tune_params: dict
                         ) -> tuple[Sequential, Dict[str, Any], pd.Series, str, pd.Series, str]:
    """Orchestrates the entire model building, tuning, training, and evaluation process for a CNN.

    This function serves as a high-level pipeline node, combining multiple steps:
    1. Splitting the raw input data into training and testing sets.
    2. Reshaping the feature data for compatibility with a 1D CNN.
    3. Tuning the CNN model's hyperparameters using KerasTuner.
    4. Training the best-found model on the prepared data.
    5. Evaluating the trained model on both the training/test split and a separate validation dataset.

    Args:
        model_input: The raw input Pandas DataFrame containing features and labels for training and initial testing.
        validation_dataset: A separate Pandas DataFrame used for final model validation, containing
                            features and labels in the same format as `model_input`.
        params: A dictionary of parameters, expected to contain:
                - 'X_column' (str): The name of the column in `model_input` and `validation_dataset`
                                    that contains the input features (fingerprints).
                - 'label' (str): The name of the column in `model_input` and `validation_dataset`
                                 that contains the target labels.
        tune_params: A dictionary of parameters for the `tune_hp_def_model` function.

    Returns:
        A tuple containing:
        - def_model (Sequential): The final trained Keras Sequential model.
        - history (Dict[str, Any]): A dictionary containing the training history
                                    (loss, accuracy, val_loss, val_accuracy per epoch)
                                    of the final trained model.
        - train_predictions (pd.Series): Predicted labels for the test set from the `model_input` split.
        - train_class_rep (str): Classification report string for the test set from the `model_input` split.
        - val_predictions (pd.Series): Predicted labels for the `validation_dataset`.
        - val_class_rep (str): Classification report string for the `validation_dataset`.
    """
    # 1. Splitting the raw input data
    X_train, X_test, y_train, y_test = train_test_split_column(model_input, params)
    # 2. Reshaping the feature data for CNN
    model_data = reshape_input(X_train, X_test, y_train)
    # 3. Tuning the CNN model's hyperparameters
    tuned_model = tune_hp_def_model(model_data, y_train, y_test, tune_params)
    # 4. Training the best-found model
    def_model, history = train_def_model(tuned_model, model_data, y_train, y_test)
    # 5. Evaluation
    train_predictions, train_class_rep, val_predictions, val_class_rep = \
        evaluate_model(def_model, model_data, y_test, validation_dataset, params)
    
    return def_model, history, train_predictions, train_class_rep, val_predictions, val_class_rep

# Model evaluation

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

def evaluate_model(model: Sequential,
                   model_data: Tuple[np.ndarray, np.ndarray, Tuple[int,int], int],
                   y_test: pd.Series,
                   validation_dataset: pd.DataFrame,
                   params: dict
                   ) -> Tuple [pd.Series, str, pd.Series, str]:
    """Evaluates a trained model on both the initial test set and a separate validation dataset.

    This function takes a trained Keras model and assesses its performance on two distinct
    datasets: the test set derived from the initial `model_input` (used during tuning
    and training evaluation) and an entirely separate `validation_dataset`. It uses
    the `get_predictions` helper function to obtain predictions and classification reports
    for each.

    Args:
        model: The trained Keras Sequential model.
        model_data: A tuple containing the reshaped data from the initial split,
                    expected to contain: `(_, reshaped_X_test, _, _)`.
        y_test: The true labels for the test set derived from `model_input`, as a Pandas Series.
        validation_dataset: A separate Pandas DataFrame containing features and labels for validation.
        params: A dictionary of parameters, expected to contain:
                - 'X_column' (str): The name of the column in `validation_dataset`
                                    that contains the input features.
                - 'label' (str): The name of the column in `validation_dataset`
                                 that contains the target labels.

    Returns:
        A tuple containing:
        - train_predictions (pd.Series): Predicted labels for the test set (from `model_input` split).
        - train_class_rep (str): Classification report string for the test set.
        - val_predictions (pd.Series): Predicted labels for the `validation_dataset`.
        - val_class_rep (str): Classification report string for the `validation_dataset`.
    """
    # Extracting training values
    _, reshaped_X_test, _, _ = model_data

    # Prepare validation dataset
    val_X_test = validation_dataset[params['X_column']]
    val_X_test_array = np.array(list(val_X_test))
    val_y_test = validation_dataset[params['label']]
    # Reshape the validation features for a 1D CNN (batch_size, sequence_length, features)
    reshaped_val_X_test = val_X_test_array.reshape((val_X_test_array.shape[0], val_X_test_array.shape[1], 1))
    
    # Get predictions and reports for the initial test set
    train_predictions, train_class_rep = get_predictions(model, reshaped_X_test, y_test)
    
    # Get predictions and reports for the separate validation dataset
    val_predictions, val_class_rep = get_predictions(model, reshaped_val_X_test, val_y_test)
    
    return train_predictions, train_class_rep, val_predictions, val_class_rep

def visualize_training(history_dic):
    """Visualizes the training and validation loss and accuracy over epochs.

    This function takes a Keras training history dictionary (as returned by `model.fit()`)
    and plots the 'loss', 'val_loss', 'accuracy', and 'val_accuracy' metrics
    against the training epochs. It uses a 'ggplot' style for aesthetics.

    Args:
        history_dic: A dictionary containing the training history. Expected keys include
                     'loss', 'val_loss', 'accuracy', and 'val_accuracy'.

    Returns:
        A `matplotlib.figure.Figure` object containing the plot. This can be used
        to save the figure or display it.
    """
    fig,ax = plt.subplots()
    plt.style.use('ggplot')
    
    # Determine the number of epochs from the length of the 'loss' history
    epochs = len(history_dic['loss'])
    epoch_values = list(range(epochs))

    # Plotting the metrics
    ax.plot(epoch_values, history_dic['loss'], label='Training loss')
    ax.plot(epoch_values, history_dic['val_loss'], label='Validation loss')
    ax.plot(epoch_values, history_dic['accuracy'], label='Training accuracy')
    ax.plot(epoch_values, history_dic['val_accuracy'], label='Validation accuracy')

    # Setting plot title and labels
    ax.set_title('Training loss and accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss/Accuracy')
    ax.legend()
    
    return fig