import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import numpy as np

# model libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from scipy.stats import randint
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout, InputLayer
from tensorflow.keras.layers import MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import keras_tuner as kt
from datetime import datetime
import os

# evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

def train_and_evaluate_rf(X_train, y_train, X_test, y_test, rf_params=None, n_runs=10):
    """
    Trains and evaluates a RandomForestRegressor over multiple runs and logs RMSE, MAE, and R².

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        rf_params (dict): Hyperparameters for the Random Forest model.
        n_runs (int): Number of repetitions for evaluation.

    Returns:
        tuple: (Last trained model, list of predictions from all runs).
    """
    all_preds = []
    for i in range(n_runs):
        print(f"\nRandom Forest Run {i+1}/{n_runs}")
        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_preds.append(y_pred)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    return model, all_preds

def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test, xgb_params=None, n_runs=10):
    """
    Trains and evaluates an XGBoost regressor multiple times and reports key metrics.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        xgb_params (dict): Parameters for XGBRegressor.
        n_runs (int): Repeated training-evaluation cycles.

    Returns:
        tuple: (Last trained XGB model, list of predictions).
    """
    all_preds = []
    for i in range(n_runs):
        print(f"\XGBoost Run {i+1}/{n_runs}")
        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_preds.append(y_pred)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    return model, all_preds

def train_and_evaluate_mlp(X_train, y_train, X_test, y_test, mlp_params=None, n_runs=10):
    """
    Trains and evaluates a Scikit-learn MLPRegressor across multiple runs.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        mlp_params (dict): MLP hyperparameters.
        n_runs (int): Number of model runs.

    Returns:
        tuple: (Final model, all predictions).
    """
    all_preds = []
    for i in range(n_runs):
        print(f"\nMLP Run {i+1}/{n_runs}")
        model = MLPRegressor(**mlp_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_preds.append(y_pred)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"✅ RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    return model, all_preds

def train_and_evaluate_svm(X_train, y_train, X_test, y_test, svm_params=None, n_runs=10):
    """
    Trains and evaluates an SVR model for regression over multiple runs.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        svm_params (dict): SVR parameters.
        n_runs (int): Number of evaluation runs.

    Returns:
        tuple: (Final model, predictions from each run).
    """
    all_preds = []
    for i in range(n_runs):
        print(f"\nSVM Run {i+1}/{n_runs}")
        model = SVR(**svm_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_preds.append(y_pred)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    return model, all_preds

def tune_model(model_name, X, y, n_iter=40, cv=5, motor_name="FD", iteration="I"):
    """
    Tunes classical regression models (RF, XGB, MLP, SVM) using BayesSearchCV or RandomizedSearchCV.

    Args:
        model_name (str): One of "rf", "xgb", "mlp", "svm".
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        n_iter (int): Number of tuning iterations.
        cv (int): Cross-validation folds.
        motor_name (str): Identifier for the dataset or engine.
        iteration (str): Custom label for experiment run.

    Returns:
        search object: Trained hyperparameter search result.
    """
    if model_name == "rf":
        estimator = RandomForestRegressor(n_jobs=-1, random_state=42)
        param_space = {
            'n_estimators': Integer(100, 300),
            'max_depth': Integer(5, 25),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5),
            'max_features': Categorical(['sqrt', 'log2', 0.5])
        }

    elif model_name == "xgb":
        estimator = XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)
        param_space = {
            'n_estimators': Integer(100, 300),
            'max_depth': Integer(3, 10),
            'learning_rate': Real(0.001, 0.2, prior='log-uniform'),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'gamma': Real(0, 5),
            'reg_alpha': Real(0, 1.0),
            'reg_lambda': Real(1, 5)
        }

    elif model_name == "mlp":
        estimator = MLPRegressor(
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10
        )
        
        param_dist = {
            'hidden_layer_sizes': [(64,), (128, 64), (256, 128)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            scoring='neg_root_mean_squared_error',
            n_iter=n_iter,
            cv=cv,
            verbose=2,
            n_jobs=-1
        )
        
        search.fit(X, y)

        print(f"Best score: {-search.best_score_:.2f}")
        print(f"Best parameters:\n{search.best_params_}")

        joblib.dump(search, f"search_results/search_{model_name}_{motor_name}_{iteration}.pkl")
        
        return search

    elif model_name == "svm":
        estimator = SVR()
        param_space = {
            'C': Real(0.1, 100, prior='log-uniform'),
            'epsilon': Real(0.01, 0.5),
            'kernel': Categorical(['rbf']),
            'gamma': Categorical(['scale', 'auto'])
        }

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    print(f"Starting hyperparameter tuning for: {model_name.upper()}")

    search = BayesSearchCV(
        estimator=estimator,
        search_spaces=param_space,
        n_iter=n_iter,
        scoring='neg_root_mean_squared_error',
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X, y)

    print(f"Best score: {-search.best_score_:.2f}")
    print(f"Best parameters:\n{search.best_params_}")

    joblib.dump(search, f"search_results/search_{model_name}_{motor_name}_{iteration}.pkl")

    return search

#-----NEURAL NETWORKS-------------------

# LSTM
def build_lstm_model(hp, input_shape):
    """
    Builds an LSTM model using KerasTuner hyperparameters.

    Args:
        hp (kt.HyperParameters): KerasTuner search space.
        input_shape (tuple): Shape of input sequence (time_steps, features).

    Returns:
        keras.Model: Compiled LSTM model.
    """
    model = Sequential([
        InputLayer(shape=input_shape),
        LSTM(
            units=hp.Int("units", min_value=32, max_value=128, step=32, default=64),
            return_sequences=False,
            recurrent_dropout=hp.Choice("recurrent_dropout", values=[0.0, 0.1, 0.2], default=0.1)
        ),
        Dropout(rate=hp.Choice("dropout", values=[0.2, 0.3, 0.4], default=0.3)),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice("lr", values=[0.001, 0.0005], default=0.001)),
        loss="mse"
    )
    return model

def train_and_evaluate_lstm(X_train, y_train, X_test, y_test, lstm_params, build_lstm_model, epochs=100, batch_size=64, n_runs=10):
    """
    Trains and evaluates an LSTM model multiple times and prints aggregated metrics.

    Args:
        X_train, y_train: Training dataset.
        X_test, y_test: Test dataset.
        lstm_params: Best hyperparameters for LSTM.
        build_lstm_model: Function to build the LSTM model.
        epochs (int): Training epochs.
        batch_size (int): Batch size for training.
        n_runs (int): Repetition count for training.

    Returns:
        tuple: (Trained model, list of predictions).
    """
    input_shape = X_train.shape[1:]
    all_preds = []
    all_rmse = []
    all_mae = []
    all_r2 = []

    for i in range(n_runs):
        print(f"\nLSTM Run {i+1}/{n_runs}")
        model = build_lstm_model(lstm_params, input_shape)

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=1
        )

        y_pred = model.predict(X_test).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

        all_preds.append(y_pred)
        all_rmse.append(rmse)
        all_mae.append(mae)
        all_r2.append(r2)

    print("\nLSTM Accuracy Summary Across Runs:")
    print(f"RMSE  - Mean: {np.mean(all_rmse):.4f}, Std: {np.std(all_rmse):.4f}")
    print(f"MAE   - Mean: {np.mean(all_mae):.4f}, Std: {np.std(all_mae):.4f}")
    print(f"R²    - Mean: {np.mean(all_r2):.4f}, Std: {np.std(all_r2):.4f}")

    return model, all_preds

def tune_lstm(X, y, max_trials=10, executions_per_trial=1, epochs=20, motor_name="", iteration=""):
    """
    Tunes an LSTM model using Bayesian optimization with KerasTuner.

    Args:
        X (np.ndarray): Input sequence data.
        y (np.ndarray): Target RUL values.
        max_trials (int): Number of tuning trials.
        executions_per_trial (int): Training repetitions per trial.
        epochs (int): Training duration per trial.
        motor_name (str): Label for engine.
        iteration (str): Experiment version label.

    Returns:
        tuner (kt.BayesianOptimization): Fitted tuner object.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuner_path = f"tuner_results/lstm_bayes_tuner_{motor_name}_{iteration}_{timestamp}"

    input_shape = (X.shape[1], X.shape[2])

    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: build_lstm_model(hp, input_shape),
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        overwrite=True,
        directory=tuner_path,
        project_name="rul_lstm_bayes"
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    tuner.search(
        X, y,
        epochs=epochs,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=2
    )

    return tuner

# CNN
def build_cnn_model(hp, input_shape):
    """
    Builds a simple 1D CNN regression model using KerasTuner hyperparameters.

    Args:
        hp (kt.HyperParameters): Search space.
        input_shape (tuple): Input data shape.

    Returns:
        keras.Model: Compiled CNN model.
    """
    model = Sequential([
        InputLayer(shape=input_shape),
        Conv1D(
            filters=hp.Choice("filters", values=[32, 64], default=64),
            kernel_size=hp.Choice("kernel_size", values=[2, 3, 5], default=3),
            activation='relu'
        ),
        Dropout(rate=hp.Choice("dropout", values=[0.2, 0.3], default=0.3)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice("lr", values=[0.001, 0.0005], default=0.001)),
        loss="mse"
    )
    return model

def train_and_evaluate_cnn(X_train, y_train, X_test, y_test, cnn_params, build_cnn_model, epochs=100, batch_size=64, n_runs=10):
    """
    Trains and evaluates a CNN model over multiple runs.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        cnn_params: Best parameters for CNN.
        build_cnn_model: Function to build the model.
        epochs (int): Epoch count per training.
        batch_size (int): Mini-batch size.
        n_runs (int): Number of full training runs.

    Returns:
        tuple: (Final model, predictions from all runs).
    """
    input_shape = X_train.shape[1:]
    all_preds = []
    all_rmse = []
    all_mae = []
    all_r2 = []

    for i in range(n_runs):
        print(f"\nCNN Run {i+1}/{n_runs}")
        model = build_cnn_model(cnn_params, input_shape)

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=1
        )

        y_pred = model.predict(X_test).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

        all_preds.append(y_pred)
        all_rmse.append(rmse)
        all_mae.append(mae)
        all_r2.append(r2)

    print("\nCNN Accuracy Summary Across Runs:")
    print(f"RMSE  - Mean: {np.mean(all_rmse):.4f}, Std: {np.std(all_rmse):.4f}")
    print(f"MAE   - Mean: {np.mean(all_mae):.4f}, Std: {np.std(all_mae):.4f}")
    print(f"R²    - Mean: {np.mean(all_r2):.4f}, Std: {np.std(all_r2):.4f}")

    return model, all_preds

def tune_cnn(X, y, max_trials=10, executions_per_trial=1, epochs=20, motor_name="", iteration=""):
    """
    Tunes a 1D CNN model using KerasTuner Bayesian Optimization.

    Args:
        X (np.ndarray): Input sequences.
        y (np.ndarray): Target values.
        max_trials (int): Number of tuning rounds.
        executions_per_trial (int): Number of executions per config.
        epochs (int): Epochs per trial.
        motor_name (str): Experiment label.
        iteration (str): Run version identifier.

    Returns:
        tuner (kt.BayesianOptimization): Trained KerasTuner object.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuner_path = f"tuner_results/cnn_bayes_tuner_{motor_name}_{iteration}_{timestamp}"

    input_shape = (X.shape[1], X.shape[2])
    
    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: build_cnn_model(hp, input_shape),
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        overwrite=True,
        directory=tuner_path,
        project_name="rul_cnn_bayes"
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    tuner.search(
        X, y,
        epochs=epochs,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=2
    )

    return tuner

# DEEP CNN
def build_deep_cnn(hp, input_shape, num_conv_layers=3):
    """
    Builds a deeper CNN with multiple Conv1D layers for RUL regression.

    Args:
        hp (kt.HyperParameters): Hyperparameter space.
        input_shape (tuple): Shape of input sequences.
        num_conv_layers (int): Number of convolution blocks.

    Returns:
        keras.Model: Compiled deep CNN model.
    """
    model = Sequential()
    model.add(InputLayer(shape=input_shape))

    # Add convolutional blocks
    for i in range(1, num_conv_layers + 1):
        filters = hp.Choice(f'filters_{i}', [32, 64, 128])
        dropout = hp.Choice(f'dropout_{i}', [0.2, 0.3, 0.4])

        model.add(Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout))

    # Dense layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(hp.Choice('dense_dropout', [0.2, 0.3, 0.4])))
    model.add(Dense(1))

    # Compile model
    lr = hp.Choice('lr', [0.001, 0.0005])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    return model

def train_and_evaluate_deep_cnn(X_train, y_train, X_test, y_test, cnn_params, build_deep_cnn, epochs=100, batch_size=64, n_runs=10):
    """
    Trains and evaluates a deep CNN over multiple runs and reports performance statistics.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        cnn_params: Tuned parameters for CNN.
        build_deep_cnn: CNN model constructor.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        n_runs (int): Number of evaluation runs.

    Returns:
        tuple: (Trained CNN model, list of prediction arrays).
    """
    input_shape = X_train.shape[1:]

    all_preds = []
    all_rmse = []
    all_mae = []
    all_r2 = []

    for i in range(n_runs):
        print(f"\nDeep CNN Run {i+1}/{n_runs}")
        model = build_deep_cnn(cnn_params, input_shape)

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=1
        )

        y_pred = model.predict(X_test).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

        all_preds.append(y_pred)
        all_rmse.append(rmse)
        all_mae.append(mae)
        all_r2.append(r2)

    print("\nDeep CNN Performance Summary Across Runs:")
    print(f"RMSE  - Mean: {np.mean(all_rmse):.4f}, Std: {np.std(all_rmse):.4f}")
    print(f"MAE   - Mean: {np.mean(all_mae):.4f}, Std: {np.std(all_mae):.4f}")
    print(f"R²    - Mean: {np.mean(all_r2):.4f}, Std: {np.std(all_r2):.4f}")

    return model, all_preds

def tune_deep_cnn(X, y, max_trials=10, executions_per_trial=1, epochs=20, motor_name="", iteration=""):
    """
    Tunes a deep CNN with multiple convolutional layers using Bayesian search.

    Args:
        X (np.ndarray): Input tensor for training.
        y (np.ndarray): Target values.
        max_trials (int): Number of hyperparameter trials.
        executions_per_trial (int): Repeats per trial.
        epochs (int): Epochs per trial.
        motor_name (str): Engine/dataset name.
        iteration (str): Experiment iteration name.

    Returns:
        tuner (kt.BayesianOptimization): Fitted tuner.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuner_path = f"tuner_results/deep_cnn_bayes_tuner_{motor_name}_{iteration}_{timestamp}"

    input_shape = (X.shape[1], X.shape[2])
    
    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: build_deep_cnn(hp, input_shape),
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        overwrite=True,
        directory=tuner_path,
        project_name="rul_cnn_bayes"
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    tuner.search(
        X, y,
        epochs=epochs,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=2
    )

    return tuner

#----------HELPER FUNCTIONS---------------
def load_best_hyperparameters(tuner_dir, project_name):
    """
    Loads the best hyperparameters from a saved KerasTuner project.

    Args:
        tuner_dir (str): Directory where the tuner is stored.
        project_name (str): Project name used during tuning.

    Returns:
        kt.HyperParameters: Best set of hyperparameters.
    """
    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: None,  # Placeholder, not used when loading
        objective='val_accuracy',
        max_trials=1,  # ignored when loading
        directory=tuner_dir,
        project_name=project_name
    )
    tuner.reload()
    best_hp = tuner.get_best_hyperparameters(1)[0]
    print(f"Loaded best hyperparameters from '{project_name}' in '{tuner_dir}':")
    print(best_hp.values)
    return best_hp

def load_tuned_model(file_name, path="search_results"):
    """
    Loads a saved BayesSearchCV or RandomizedSearchCV model from disk.

    Args:
        file_name (str): Model file name (without extension).
        path (str): Directory path to the .pkl file.

    Returns:
        search object: Loaded model object.
    """
    # Create full file path
    if os.path.isdir(path):
        file_path = os.path.join(path, f"{file_name}.pkl")
    else:
        file_path = path
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    print(f"Loading tuned model from: {file_path}")
    search = joblib.load(file_path)
    return search

import os
import pickle

def save_predictions_dict(predictions_dict, save_path="saved_predictions/predictions.pkl"):
    """
    Saves the dictionary of model predictions to a .pkl file.

    Args:
        predictions_dict (dict): Predictions from various models.
        save_path (str): Output file path.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(predictions_dict, f)
    print(f"Saved predictions dictionary to: {save_path}")

def load_predictions_dict(load_path):
    """
    Loads a saved predictions dictionary from a .pkl file.

    Args:
        load_path (str): File path to the saved predictions.

    Returns:
        dict: Loaded predictions dictionary.
    """
    with open(load_path, "rb") as f:
        predictions_dict = pickle.load(f)
    print(f"Loaded predictions dictionary from: {load_path}")
    return predictions_dict