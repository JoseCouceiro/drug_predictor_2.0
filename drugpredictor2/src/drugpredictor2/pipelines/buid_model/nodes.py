import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras_tuner import RandomSearch
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Functions to prepare data input
""" ESTAMOS VALIDANDO ESTA"""
def train_test_split_column(model_input: pd.DataFrame, columns, label: dict) -> tuple:
    """
    Function that makes a train/test split of a selected column of a pandas dataframe. Y values are taken from the label column.
    Args:
      input_pickle: input dataframe.
      column: name of the column selected as X values.
      label: name of the column selected as y values (target).
    Output: train/test tuple.
    """
    X = model_input[columns]
    y = model_input[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test

def reshape_input(tuple_split: tuple) -> tuple:
    """
    Function that reshapes the input data (X_train, X_test) to fit a 1D CNN model and determines both the number of classes \
        from the label data and the shape of the reshaped data.
    Input: train/test split tuple.
    Output: tuple with the data necessary to run the CNN model.
    """
    # Transform arrays into lists
    X_train = np.array(list(tuple_split[0]))
    X_test = np.array(list(tuple_split[1]))
    # Determine number of classes
    n_classes = len(np.unique(tuple_split[2]))
    # Reshape
    reshaped_X_train= X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    reshaped_X_test= X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    # Determine in_shape
    in_shape = reshaped_X_train.shape[1:]
    return reshaped_X_train, reshaped_X_test, in_shape, n_classes

# Functions to build the definitive model

def tune_hp_def_model(model_data: tuple, y_train: pd.Series, y_test: pd.Series, tune_params: dict) -> Sequential:
    """
    Funtion that uses RandomSearch from keras.tuner to obtain the best hyperparameters for a CNN model.
    Args:
      model_data: tuple containing pandas series for X_train, X_test and the values for the in_shape and number of classes.
      y_train: pandas series containing y_train data.
      y_test: pandas series containing y_test.
      tune_params: a set of optional parameters.
    Output: a tuned CNN model.
    """
    tuner = RandomSearch(build_def_model,
                    objective=tune_params['objective'],
                    max_trials =tune_params['max_trials'],
                    directory = os.path.join('temp', 'tuner', tune_params['date']))
    tuner.search(model_data[0],y_train,epochs=3,validation_data=(model_data[1],y_test))
    tuned_model = tuner.get_best_models(num_models=1)[0]
    tuned_model.summary() 
    return tuned_model

def build_def_model(hp) -> Sequential:
    """
    Function that builds a CNN model with sets of hyperparameters that will be chosen by the function "tune_hp_def_model".
    Input: a set of hyperparamneters.
    Output: a cnn model.
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
    model.add(layers.Dense(16, activation='softmax'))
    # Compilation of model
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )
    return model

def train_def_model(tuned_model: Sequential, model_data: tuple, y_train: pd.Series, y_test: pd.Series) -> tuple:
    """
    Function that trains a 1D CNN model.
    Args: 
      tuned_mode: a 1D CNN model.
      model_data: tuple containing pandas series for X_train, X_test and the values for the in_shape and number of classes.
      y_train: pandas series containing y_train data.
      y_test: pandas series containing y_test.
    Output: a 1D CNN model, the training history.
    """
    # Configure early stopping
    es = EarlyStopping(monitor='val_loss', patience=10)
    mc = ModelCheckpoint(filepath = os.path.join('temp', 'compiled_model', 'checkpoints', '{epoch:02d}-{val_loss:.3f}.hdf5'),
                         monitor = 'val_loss',
                         save_best_only = True)
    # Fit the model
    tuned_model.fit(model_data[0],
                    y_train,
                    epochs=200,
                    batch_size=128,
                    verbose=1,
                    validation_split = 0.3,
                    callbacks = [es,mc])
    history_dic = tuned_model.history.history
    # Evaluate the model
    tuned_model.evaluate(model_data[1], y_test, verbose=1)
    
    return tuned_model, history_dic

def obtain_model(split_col: tuple, model_data: tuple, tune_params: dict) -> tuple:
    """
    Function that builds a CNN model, chooses the best hyperparameters, trains it and return the tuned model and the predictions \
        obtained during the training.
    Args: 
      split_col: train/test tuple.
      model_data: tuple containing pandas series for reshaped X_train, X_test and the values for the in_shape and number of classes.
    Output:
      def_model: a tuned and trained CNN_model.
      history: the training history with values of accuracy and loss for each epoch.
    """
    tuned_model = tune_hp_def_model(model_data, split_col[2], split_col[3], tune_params)
    def_model, history = train_def_model(tuned_model, model_data, split_col[2], split_col[3])
    return def_model, history 