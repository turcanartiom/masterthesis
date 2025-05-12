import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import BayesianOptimization

from tcn import TCN
from tensorflow.keras.optimizers import Adam

def tune_model(model_name, X, y, n_iter=40, cv=5):
    """
    Performs hyperparameter tuning for the specified classical model (RF, XGB, MLP, SVM) using Bayesian Search.

    Args:
        model_name (str): Model identifier ("rf", "xgb", "mlp", or "svm").
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label vector.
        n_iter (int): Number of iterations for the search.
        cv (int): Number of cross-validation folds.

    Returns:
        search object: Trained BayesSearchCV or RandomizedSearchCV model with best parameters.
    """
    if model_name == "rf":
        estimator = RandomForestClassifier(n_jobs=-1, random_state=42)
        param_space = {
            'n_estimators': Integer(100, 300),
            'max_depth': Integer(5, 25),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5),
            'max_features': Categorical(['sqrt', 'log2', 0.5])
        }

    elif model_name == "xgb":
        estimator = XGBClassifier(
            objective='multi:softprob',
            n_jobs=-1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
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
        estimator = MLPClassifier(
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10
        )
        param_dist = {
            'hidden_layer_sizes': [
                (64,), (128, 64), (256, 128), (128, 64, 32), (256, 128, 64)
            ],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01],
            'learning_rate': ['constant', 'adaptive', 'invscaling']
        }

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            scoring='accuracy',
            n_iter=n_iter,
            cv=cv,
            verbose=2,
            n_jobs=-1,
            random_state=42
        )

        search.fit(X, y)

        print(f"\nBest score: {search.best_score_:.4f}")
        print(f"Best parameters:\n{search.best_params_}")

        joblib.dump(search, f"search_{model_name}.pkl")
        return search

    elif model_name == "svm":
        estimator = SVC(probability=True)
        param_space = {
            'C': Real(0.1, 100, prior='log-uniform'),
            'kernel': Categorical(['rbf']),
            'gamma': Categorical(['scale', 'auto'])
        }

    else:
        raise ValueError(f"❌ Unsupported model: {model_name}")

    print(f"\nStarting hyperparameter tuning for: {model_name.upper()}")

    search = BayesSearchCV(
        estimator=estimator,
        search_spaces=param_space,
        n_iter=n_iter,
        scoring='accuracy',
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X, y)

    print(f"\nBest score: {search.best_score_:.4f}")
    print(f"Best parameters:\n{search.best_params_}")

    joblib.dump(search, f"search_results/search_{model_name}.pkl")
    return search

def train_and_evaluate_rf(X_train, y_train, X_test, y_test, rf_params=None, n_repeats=10):
    """
    Trains and evaluates a RandomForestClassifier multiple times to report average performance.

    Args:
        X_train, X_test (np.ndarray): Train/test features.
        y_train, y_test (np.ndarray): Train/test labels.
        rf_params (dict): Optional custom parameters for RandomForest.
        n_repeats (int): Number of training repetitions.

    Returns:
        list: List of predictions from each repeat.
    """
    if rf_params is None:
        rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
    rf_params['n_jobs'] = -1

    all_preds = []
    accuracies = []

    print("\nTraining and Evaluating RandomForestClassifier...")
    for i in range(n_repeats):
        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Run {i+1}/{n_repeats} - Accuracy: {acc:.4f}")
        all_preds.append(y_pred)
        accuracies.append(acc)

    print("\nAccuracy Summary (Random Forest):")
    print(f"Mean: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}, Var: {np.var(accuracies):.4f}")
    return all_preds

def train_and_evaluate_xgb(X_train, y_train, X_test, y_test, xgb_params=None, n_repeats=10):
    """
    Trains and evaluates an XGBoost classifier multiple times and prints accuracy statistics.

    Args:
        X_train, X_test (np.ndarray): Train/test features.
        y_train, y_test (np.ndarray): Train/test labels.
        xgb_params (dict): Parameters for XGBClassifier.
        n_repeats (int): Number of repetitions for evaluation.

    Returns:
        list: Predictions from each repetition.
    """
    if xgb_params is None:
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'multi:softprob',
            'random_state': 42,
            'n_jobs': -1,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss'
        }
    xgb_params['n_jobs'] = -1

    all_preds = []
    accuracies = []

    print("\nTraining and Evaluating XGBClassifier...")
    for i in range(n_repeats):
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Run {i+1}/{n_repeats} - Accuracy: {acc:.4f}")
        all_preds.append(y_pred)
        accuracies.append(acc)

    print("\nAccuracy Summary (XGBoost):")
    print(f"Mean: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}, Var: {np.var(accuracies):.4f}")
    return all_preds

def train_and_evaluate_svm(X_train, y_train, X_test, y_test, svm_params=None, n_repeats=10):
    """
    Trains and evaluates an SVM classifier (SVC) multiple times and reports metrics.

    Args:
        X_train, X_test (np.ndarray): Feature matrices.
        y_train, y_test (np.ndarray): Label arrays.
        svm_params (dict): SVC parameters.
        n_repeats (int): Number of repetitions.

    Returns:
        list: Predictions from each evaluation.
    """
    if svm_params is None:
        svm_params = {
            'kernel': 'rbf',
            'C': 10.0,
            'gamma': 'scale'
        }

    all_preds = []
    accuracies = []

    print("\nTraining and Evaluating SVC...")
    for i in range(n_repeats):
        model = SVC(**svm_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Run {i+1}/{n_repeats} - Accuracy: {acc:.4f}")
        all_preds.append(y_pred)
        accuracies.append(acc)

    print("\nAccuracy Summary (SVM):")
    print(f"Mean: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}, Var: {np.var(accuracies):.4f}")
    return all_preds

def train_and_evaluate_mlp(X_train, y_train, X_test, y_test, mlp_params=None, n_repeats=10):
    """
    Trains and evaluates an MLPClassifier multiple times to measure accuracy consistency.

    Args:
        X_train, X_test (np.ndarray): Feature data.
        y_train, y_test (np.ndarray): Target labels.
        mlp_params (dict): Hyperparameters for MLPClassifier.
        n_repeats (int): Number of training iterations.

    Returns:
        list: Predictions for each run.
    """
    if mlp_params is None:
        mlp_params = {
            'hidden_layer_sizes': (128, 64),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 200,
            'random_state': 42
        }

    all_preds = []
    accuracies = []

    print("\nTraining and Evaluating MLPClassifier...")
    for i in range(n_repeats):
        model = MLPClassifier(**mlp_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Run {i+1}/{n_repeats} - Accuracy: {acc:.4f}")
        all_preds.append(y_pred)
        accuracies.append(acc)

    print("\nAccuracy Summary (MLP):")
    print(f"Mean: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}, Var: {np.var(accuracies):.4f}")
    return all_preds

def build_lstm(hp, input_shape, num_classes):
    """
    Builds a simple LSTM model using KerasTuner hyperparameters.

    Args:
        hp: HyperParameters object.
        input_shape (tuple): Shape of the input data (time_steps, features).
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(LSTM(
        units=hp.Int('lstm_units', 32, 128, step=32),
        dropout=hp.Float('lstm_dropout', 0.1, 0.5, step=0.1),
        return_sequences=False
    ))

    model.add(Dense(
        hp.Int('dense_units', 32, 128, step=32),
        activation='relu'
    ))

    model.add(Dropout(hp.Float('dense_dropout', 0.1, 0.5, step=0.1)))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('lr', [1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def tune_lstm(X_lstm, y_categorical, max_trials=10, executions_per_trial=1, epochs=20):
    """
    Tunes hyperparameters for a simple LSTM model using Bayesian optimization.

    Args:
        X_lstm (np.ndarray): Input features (shape: samples, time_steps, features).
        y_categorical (np.ndarray): One-hot encoded labels.
        max_trials (int): Maximum number of hyperparameter trials.
        executions_per_trial (int): Number of executions per trial for robustness.
        epochs (int): Number of training epochs.

    Returns:
        tuple: KerasTuner tuner object and best hyperparameters.
    """
    input_shape = X_lstm.shape[1:]  # (time_steps, features)
    num_classes = y_categorical.shape[1]

    tuner = BayesianOptimization(
        hypermodel=lambda hp: build_lstm(hp, input_shape, num_classes),
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='lstm_tuning',
        project_name='fault_classification_bayes'
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    tuner.search(
        X_lstm, y_categorical,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    return tuner, best_hp

def train_and_evaluate_lstm(X_lstm, y_categorical, best_hp, test_size=0.2):
    """
    Trains and evaluates the simple LSTM model with the best hyperparameters.

    Args:
        X_lstm (np.ndarray): Input features with shape (samples, time_steps, features).
        y_categorical (np.ndarray): One-hot encoded targets.
        best_hp: Best hyperparameter object from KerasTuner.
        test_size (float): Fraction of the dataset to be used as the test set.

    Returns:
        tuple: Trained Keras model and predicted class labels on the test set.
    """
    input_shape = X_lstm.shape[1:]  # (time_steps, features)
    num_classes = y_categorical.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y_categorical, test_size=test_size, random_state=42
    )

    model = build_lstm(best_hp, input_shape, num_classes)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nLSTM Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    return model, y_pred

def load_best_hyperparameters(tuner_dir, project_name):
    """
    Loads the best KerasTuner hyperparameters from a saved tuner project.

    Args:
        tuner_dir (str): Directory where tuner was saved.
        project_name (str): Name of the tuning project.

    Returns:
        HyperParameters: Best found hyperparameters.
    """
    tuner = BayesianOptimization(
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

def build_cnn(hp, input_shape, num_classes):
    """
    Builds a 1D CNN model using hyperparameters from KerasTuner.

    Args:
        hp: HyperParameters instance.
        input_shape (tuple): Shape of input data.
        num_classes (int): Number of classes for classification.

    Returns:
        keras.Model: Compiled CNN model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv1D(
        filters=hp.Int("filters_1", 32, 128, step=32),
        kernel_size=hp.Choice("kernel_size_1", [3, 5, 7]),
        activation='relu'
    ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(hp.Float("dropout_1", 0.2, 0.5, step=0.1)))

    model.add(Conv1D(
        filters=hp.Int("filters_2", 32, 128, step=32),
        kernel_size=hp.Choice("kernel_size_2", [3, 5]),
        activation='relu'
    ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(hp.Float("dropout_2", 0.2, 0.5, step=0.1)))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(
        hp.Int("dense_units", 64, 256, step=64),
        activation='relu'
    ))
    model.add(Dropout(hp.Float("dropout_3", 0.2, 0.5, step=0.1)))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def tune_cnn(X_cnn, y_categorical, max_trials=10, executions_per_trial=1, epochs=20):
    """
    Tunes CNN model using Bayesian optimization on 1D signal data.

    Args:
        X_cnn (np.ndarray): Input features (samples, time, 1).
        y_categorical (np.ndarray): One-hot encoded labels.
        max_trials (int): Trials for tuning.
        executions_per_trial (int): Repeats per trial.
        epochs (int): Training epochs.

    Returns:
        tuple: Tuner and best hyperparameters.
    """
    input_shape = (X_cnn.shape[1], 1)
    num_classes = y_categorical.shape[1]

    tuner = BayesianOptimization(
        hypermodel=lambda hp: build_cnn(hp, input_shape, num_classes),
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='cnn_tuning',
        project_name='cnn_fault_classification'
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    tuner.search(X_cnn, y_categorical, epochs=epochs, validation_split=0.2, verbose=1, callbacks=[early_stop])
    best_hp = tuner.get_best_hyperparameters(1)[0]
    return tuner, best_hp

def train_and_evaluate_cnn(X_train, y_train_cat, X_test, y_test_cat, best_hp, n_repeats=10):
    """
    Trains and evaluates a CNN classifier using provided hyperparameters.

    Args:
        X_train, X_test (np.ndarray): Input data.
        y_train_cat, y_test_cat (np.ndarray): One-hot encoded targets.
        best_hp: Best KerasTuner hyperparameters.
        n_repeats (int): Number of repetitions.

    Returns:
        list: List of predicted labels per run.
    """
    input_shape = (X_train.shape[1], 1)  # 1D FFT input (samples, freq_bins, 1)
    num_classes = y_train_cat.shape[1]

    all_preds = []
    accuracies = []

    print("\nTraining and Evaluating CNNClassifier...")

    for i in range(n_repeats):
        model = build_cnn(best_hp, input_shape, num_classes)

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(
            X_train, y_train_cat,
            validation_split=0.1,
            epochs=50,
            batch_size=64,
            callbacks=[early_stop],
            verbose=0  # You can set to 1 for full logs
        )

        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test_cat, axis=1)

        acc = accuracy_score(y_true, y_pred)
        print(f"Run {i+1}/{n_repeats} - Accuracy: {acc:.4f}")

        all_preds.append(y_pred)
        accuracies.append(acc)

    print("\nAccuracy Summary (CNN):")
    print(f"Mean: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}, Var: {np.var(accuracies):.4f}")

    return all_preds

def build_cnn2d(hp, input_shape, num_classes):
    """
    Builds a 2D CNN model with tunable parameters using KerasTuner.

    Args:
        hp: Hyperparameter object.
        input_shape (tuple): Shape of the input (time, freq, 1).
        num_classes (int): Number of classification categories.

    Returns:
        keras.Model: Compiled CNN2D model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))  # input_shape = (time_steps, freq_bins, 1)

    # Map kernel size choice
    kernel_size_choice = hp.Choice("kernel_size_index", [0, 1])
    kernel_sizes = [(3, 3), (5, 5)]
    kernel_size = kernel_sizes[kernel_size_choice]

    model.add(Conv2D(
        filters=hp.Int("filters", 32, 128, step=32),
        kernel_size=kernel_size,
        activation='relu',
        padding='same'
    ))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1)))

    model.add(Flatten())

    model.add(Dense(
        units=hp.Int("dense_units", 64, 256, step=64),
        activation='relu'
    ))
    model.add(Dropout(hp.Float("dense_dropout", 0.2, 0.5, step=0.1)))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def tune_cnn2d(X_cnn, y_categorical, max_trials=10, executions_per_trial=1, epochs=20):
    """
    Tunes a 2D CNN model using Bayesian optimization.

    Args:
        X_cnn (np.ndarray): Input feature array.
        y_categorical (np.ndarray): One-hot target array.
        max_trials (int): Maximum number of tuning trials.
        executions_per_trial (int): Repeats per trial.
        epochs (int): Number of training epochs.

    Returns:
        tuple: Tuner object and best hyperparameters.
    """
    input_shape = X_cnn.shape[1:]  # (time, freq, 1)
    num_classes = y_categorical.shape[1]

    tuner = BayesianOptimization(
        hypermodel=lambda hp: build_cnn2d(hp, input_shape, num_classes),
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='cnn2d_tuning',
        project_name='cnn2d_fault_classification'
    )

    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    tuner.search(
        X_cnn,
        y_categorical,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    return tuner, best_hp

def train_and_evaluate_cnn2d(X_train, y_train, X_test, y_test, best_hp, build_fn, epochs=20, batch_size=64):
    """
    Trains and evaluates a 2D CNN model with best hyperparameters.

    Args:
        X_train, X_test (np.ndarray): Training and testing feature arrays.
        y_train, y_test (np.ndarray): One-hot label arrays.
        best_hp: KerasTuner best hyperparameters.
        build_fn: Function to build the CNN2D model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        tuple: Trained model and predicted labels.
    """
    input_shape = X_train.shape[1:]  # (time_steps, freq_bins, 1)
    num_classes = y_train.shape[1]

    model = build_fn(best_hp, input_shape, num_classes)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred_probs = model.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nCNN2D Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred))

    return model, y_pred

def build_lstm_stft(hp, input_shape, num_classes):
    """
    Builds an LSTM model for STFT features using tunable hyperparameters.

    Args:
        hp: KerasTuner hyperparameter object.
        input_shape (tuple): Input shape (time_steps, features).
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))  # shape = (time_steps, features)

    model.add(LSTM(
        units=hp.Int('units', 64, 128, step=32),
        dropout=hp.Float('dropout', 0.1, 0.3, step=0.1)
    ))

    model.add(Dense(
        hp.Int('dense_units', 64, 128, step=32),
        activation='relu'
    ))
    model.add(Dropout(hp.Float('dense_dropout', 0.2, 0.4, step=0.1)))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def tune_lstm_stft(X_lstm, y_categorical, max_trials=10, executions_per_trial=1, epochs=20):
    """
    Tunes LSTM model designed for STFT input using Bayesian search.

    Args:
        X_lstm (np.ndarray): Input sequence data.
        y_categorical (np.ndarray): Categorical targets.
        max_trials (int): Number of hyperparameter trials.
        executions_per_trial (int): Executions per configuration.
        epochs (int): Training epochs.

    Returns:
        tuple: Tuner and best hyperparameters.
    """
    input_shape = X_lstm.shape[1:]  # (time_steps, features)
    num_classes = y_categorical.shape[1]

    tuner = BayesianOptimization(
        hypermodel=lambda hp: build_lstm_stft(hp, input_shape, num_classes),
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='lstm_stft_tuning',
        project_name='lstm_fault_classification'
    )

    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    tuner.search(
        X_lstm,
        y_categorical,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    return tuner, best_hp

def train_and_evaluate_lstm_stft(X_train, y_train, X_test, y_test, best_hp, build_fn, epochs=20, batch_size=64):
    """
    Trains and evaluates LSTM model on STFT-transformed data.

    Args:
        X_train, X_test (np.ndarray): Input arrays.
        y_train, y_test (np.ndarray): One-hot encoded labels.
        best_hp: Best hyperparameters object.
        build_fn: LSTM model building function.
        epochs (int): Epochs for training.
        batch_size (int): Size of mini-batches.

    Returns:
        tuple: Trained model and predicted classes.
    """
    input_shape = X_train.shape[1:]  # (time_steps, freq_bins)
    num_classes = y_train.shape[1]

    model = build_fn(best_hp, input_shape, num_classes)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred_probs = model.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nLSTM Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred))

    return model, y_pred

#-----------------HELPER FUNCTIONS-----------------------------------------------------

def load_tuned_model(model_name, path="search_results"):
    """
    Loads a previously saved tuned model search result from disk.

    Args:
        model_name (str): Name of the model ("rf", "xgb", etc.).
        path (str): Directory or full path to the model pickle file.

    Returns:
        object: Loaded search object from joblib.
    """
    # Create full file path
    if os.path.isdir(path):
        file_path = os.path.join(path, f"search_{model_name}.pkl")
    else:
        file_path = path
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    print(f"Loading tuned model from: {file_path}")
    search = joblib.load(file_path)
    return search