# import libraries
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

class rul_predictor:
    def __init__(self) -> None:
        # define column names
        self.index_names = ['unit_no', 'time_cycle']
        self.setting_names = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_names = [f's_{i}' for i in range(1, 22)]
        self.col_names = self.index_names + self.setting_names + self.sensor_names
    
        # read data from files
        # self.read_data()

    def read_data(self,
                  filename_train='CMaps/train_FD001.txt',
                  filename_test="CMaps/test_FD001.txt",
                  filename_y="CMaps/RUL_FD001.txt"):

        # read train, test data
        self.train = pd.read_csv(filename_train,
                                 sep='\s+', 
                                 header=None, 
                                 names=self.col_names)
        self.test = pd.read_csv(filename_test,
                                sep='\s+',
                                header=None, 
                                names=self.col_names)
        self.y_test = pd.read_csv(filename_y,
                                  sep='\s+', 
                                  header=None, 
                                  names=['RUL'])

    def add_RUL(self, df):
        # Get the total number of cycles for each unit
        grouped_by_unit = df.groupby(by="unit_no")
        max_cycle = grouped_by_unit["time_cycle"].max()
    
        # Merge the max cycle back into the original frame
        result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), 
                                left_on='unit_no', 
                                right_index=True)
    
        # Calculate remaining useful life for each row
        remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycle"]
    
        result_frame["RUL"] = remaining_useful_life
    
        # Drop max_cycle as it's no longer needed
        result_frame = result_frame.drop("max_cycle", axis=1)
    
        return result_frame

    def cap_rul(self, df, cap):
        if isinstance(cap, int) and cap > 0:
            df["RUL"] = df["RUL"].clip(upper=cap)
        return df

    def variance_threshold_analysis(self, df, threshold=0.01, export_path=None):
        """
        Apply VarianceThreshold to sensor columns and plot variance values.
    
        Args:
            df (pd.DataFrame): DataFrame containing the full turbofan dataset.
            threshold (float): Minimum variance required to keep a feature.
            export_path (str, optional): If provided, saves the plot to this file path.
        """
        print(f"🔍 Running Variance Threshold Analysis (threshold = {threshold})...")
    
        # Identify sensor columns: usually start with 's_' or 's1', 's2', etc.
        sensor_cols = [col for col in df.columns if col.startswith('s_') or (col.startswith('s') and col[1:].isdigit())]
    
        # Drop non-sensor columns
        sensor_data = df[sensor_cols]
    
        # Apply VarianceThreshold (fit only to compute mask)
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(sensor_data)
    
        # Get variance values
        variances = sensor_data.var()
        variances_sorted = variances.sort_values()
    
        # Plot
        plt.figure(figsize=(12, 6))
        variances_sorted.plot(kind='bar', color='skyblue')
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
        plt.title('Sensor Variance Analysis')
        plt.xlabel('Sensor')
        plt.ylabel('Variance')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
    
        # Save or show plot
        if export_path:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            plt.savefig(export_path)
            print(f"📸 Variance plot saved to: {export_path}")
        else:
            plt.show()
    
        # Print sensors below threshold
        low_variance_sensors = variances_sorted[variances_sorted < threshold]
        print("📉 Low-variance sensors to consider dropping:")
        print(low_variance_sensors)
    
        return low_variance_sensors.index.tolist()

    def remove_low_variance_features(self, df, features, threshold=0.01):
        """Remove features with variance below the specified threshold and drop uninformative ones from the DataFrame."""
        print(f"🔍 Removing low variance features (threshold={threshold})...")
        
        # Ensure we're only analyzing valid sensor features
        feature_cols = [col for col in features if col in df]
        
        # Apply variance threshold
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df[feature_cols])
        
        retained_features = list(np.array(feature_cols)[selector.get_support()])
        print(f"✅ Retained {len(retained_features)} out of {len(feature_cols)} sensor features.")
        print(f"✅ Retained features: {retained_features}")
        
        # Drop unretained features from original dataframe
        columns_to_keep = [col for col in df.columns if col not in feature_cols or col in retained_features]
        return df[columns_to_keep], retained_features
    
    def apply_signal_smoothing(self, df, window=5):
        """Apply moving average smoothing to sensor columns."""
        print(f"📉 Applying rolling mean smoothing (window={window}) to sensor signals...")
        smoothed_df = df.copy()
        sensor_cols = [col for col in df.columns if col.startswith('s_')]
        for col in sensor_cols:
            smoothed_df[col] = df[col].rolling(window=window, min_periods=1).mean()
        print("✅ Smoothing complete.")
        return smoothed_df

    def normalize(self, train_df, test_df, features):
        feature_cols = [col for col in features if col in train_df]

        scaler = MinMaxScaler()

        # 1. Fit the scaler only on training data
        scaler = MinMaxScaler()
        scaler.fit(train_df[features])
        
        # 2. Transform training and test data separately
        train_df[features] = scaler.transform(train_df[features])
        test_df[features] = scaler.transform(test_df[features])

    def add_rolling_features(self, df, features, window=5):
        for feature in features:
            # Rolling mean
            df[f'{feature}_rolling_mean_{window}'] = (
                df.groupby('unit_no')[feature]
                  .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
    
            # Rolling std
            df[f'{feature}_rolling_std_{window}'] = (
                df.groupby('unit_no')[feature]
                  .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
            )
        return df
    
    def generate_window(self, df, seq_length, features):
        sequences = []
        labels = []
        
        for unit in df["unit_no"].unique():
            unit_data = df[df["unit_no"] == unit]
            for i in range(len(unit_data) - seq_length):
                sequence = unit_data.iloc[i : i + seq_length][features].values
                label = unit_data.iloc[ i + (seq_length - 1)]["RUL"]
                sequences.append(sequence)
                labels.append(label)
        
        return np.array(sequences), np.array(labels)

    def prepare_test_data(self, test_df, rul_truth, features, seq_length):
        X_test = []

        for unit in test_df["unit_no"].unique():
            unit_df = test_df[test_df["unit_no"] == unit].reset_index(drop=True)
            seq = unit_df[features].values
    
            if len(seq) < seq_length:
                # Pad with zeros at the beginning
                padding = np.zeros((seq_length - len(seq), len(features)))
                seq = np.vstack([padding, seq])
            else:
                # Use last `seq_length` cycles
                seq = seq[-seq_length:]
    
            X_test.append(seq)
    
        X_test = np.array(X_test)
        y_test = rul_truth
    
        return X_test, y_test

    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test, rf_params=None):
        if rf_params is None:
            rf_params = {
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42,
                'n_jobs': -1
            }
            
        rf_params['n_jobs'] = -1
        rf_params['verbose'] = 1
        
        print("Training RandomForestRegressor with params:", rf_params)
        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)
        
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ RMSE: {rmse:.2f}")
        print(f"✅ MAE:  {mae:.2f}")
        print(f"✅ R2:  {r2:.2f}")
        
        return model, y_pred

    def train_and_evaluate_xgboost(self, X_train, y_train, X_test, y_test, xgb_params=None):
        if xgb_params is None:
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'objective': 'reg:squarederror',
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 1
            }
    
        print("Training XGBRegressor with params:", xgb_params)
        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)
    
        print("Evaluating model...")
        y_pred = model.predict(X_test)
    
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        print(f"✅ RMSE: {rmse:.2f}")
        print(f"✅ MAE:  {mae:.2f}")
        print(f"✅ R2:  {r2:.2f}")
    
        return model, y_pred

    def train_and_evaluate_mlp(self, X_train, y_train, X_test, y_test, mlp_params=None):
        if mlp_params is None:
            mlp_params = {
                'hidden_layer_sizes': (128, 64),
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 200,
                'random_state': 42
            }
    
        print("Training MLPRegressor with params:", mlp_params)
        model = MLPRegressor(**mlp_params)
        model.fit(X_train, y_train)
    
        print("Evaluating model...")
        y_pred = model.predict(X_test)
    
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        print(f"✅ RMSE: {rmse:.2f}")
        print(f"✅ MAE:  {mae:.2f}")
        print(f"✅ R2:  {r2:.2f}")
    
        return model, y_pred

    def train_and_evaluate_svm(self, X_train, y_train, X_test, y_test, svm_params=None):
        if svm_params is None:
            svm_params = {
                'kernel': 'rbf',
                'C': 10.0,
                'epsilon': 0.2
            }
    
        print("Training SVR with params:", svm_params)
        model = SVR(**svm_params)
        model.fit(X_train, y_train)
    
        print("Evaluating model...")
        y_pred = model.predict(X_test)
    
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        print(f"✅ RMSE: {rmse:.2f}")
        print(f"✅ MAE:  {mae:.2f}")
        print(f"✅ R2:  {r2:.2f}")
    
        return model, y_pred

    def train_and_evaluate_lstm(self, X_train, y_train, X_test, y_test, lstm_params=None):
        if lstm_params is None:
            lstm_params = {
                'units': 64,
                'dropout': 0.2,
                'batch_size': 64,
                'epochs': 50,
                'learning_rate': 0.001
            }
    
        model = Sequential([
            InputLayer(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(lstm_params['units'], return_sequences=False),
            Dropout(lstm_params['dropout']),
            Dense(1)
        ])
    
        model.compile(optimizer=Adam(learning_rate=lstm_params['learning_rate']), loss='mse')
    
        print("Training LSTM...")
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=lstm_params['batch_size'],
            epochs=lstm_params['epochs'],
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1
        )
    
        y_pred = model.predict(X_test).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        print(f"✅ RMSE: {rmse:.2f}")
        print(f"✅ MAE:  {mae:.2f}")
        print(f"✅ R2:   {r2:.2f}")
    
        return model, y_pred

    def train_and_evaluate_cnn(self, X_train, y_train, X_test, y_test, cnn_params=None):
        if cnn_params is None:
            cnn_params = {
                'filters': 64,
                'kernel_size': 3,
                'dropout': 0.3,
                'batch_size': 64,
                'epochs': 50,
                'learning_rate': 0.001
            }
    
        model = Sequential([
            InputLayer(shape=(X_train.shape[1], X_train.shape[2])),
            Conv1D(filters=cnn_params['filters'], kernel_size=cnn_params['kernel_size'], activation='relu'),
            Dropout(cnn_params['dropout']),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1)
        ])
    
        model.compile(optimizer=Adam(learning_rate=cnn_params['learning_rate']), loss='mse')
    
        print("Training CNN...")
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=cnn_params['batch_size'],
            epochs=cnn_params['epochs'],
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1
        )
    
        y_pred = model.predict(X_test).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        print(f"✅ RMSE: {rmse:.2f}")
        print(f"✅ MAE:  {mae:.2f}")
        print(f"✅ R2:   {r2:.2f}")
    
        return model, y_pred
    
    def train_and_evaluate_ann(self, X_train, y_train, X_test, y_test, ann_params=None):
        if ann_params is None:
            ann_params = {
                'hidden_units': [128, 64],
                'dropout': 0.2,
                'batch_size': 64,
                'epochs': 50,
                'learning_rate': 0.001
            }

        # Flatten input
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        model = Sequential()
        model.add(InputLayer(shape=(X_train_flat.shape[1],)))
        for units in ann_params['hidden_units']:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(ann_params['dropout']))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=ann_params['learning_rate']), loss='mse')

        print("Training ANN...")
        model.fit(
            X_train_flat, y_train,
            validation_data=(X_test_flat, y_test),
            batch_size=ann_params['batch_size'],
            epochs=ann_params['epochs'],
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1
        )

        y_pred = model.predict(X_test_flat).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"✅ RMSE: {rmse:.2f}")
        print(f"✅ MAE:  {mae:.2f}")
        print(f"✅ R2:   {r2:.2f}")

        return model, y_pred

    def train_and_evaluate_deep_cnn(self, X_train, y_train, X_test, y_test, cnn_params=None):
        """
        Train and evaluate a 3-layer deep CNN model for RUL prediction.

        Args:
            X_train (np.array): 3D input for training, shape (samples, seq_len, num_features)
            y_train (np.array): Training labels (RUL values)
            X_test (np.array): 3D input for testing
            y_test (np.array): Test labels
            cnn_params (dict): Hyperparameters like batch_size, epochs, learning_rate

        Returns:
            model: Trained Keras model
            y_pred: Predictions on test set
        """
        if cnn_params is None:
            cnn_params = {
                'batch_size': 64,
                'epochs': 50,
                'learning_rate': 0.001
            }

        # Build the model
        input_shape = X_train.shape[1:]  # (seq_len, num_features)
        model = self.build_deep_cnn(input_shape=input_shape, learning_rate=cnn_params['learning_rate'])

        print("🚀 Training Deep CNN...")
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=cnn_params['batch_size'],
            epochs=cnn_params['epochs'],
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1
        )

        # Predict and evaluate
        y_pred = model.predict(X_test).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"✅ RMSE: {rmse:.2f}")
        print(f"✅ MAE:  {mae:.2f}")
        print(f"✅ R²:   {r2:.2f}")

        return model, y_pred
    
    def train_and_evaluate_deeper_cnn(self, X_train, y_train, X_test, y_test, cnn_params=None):
        """
        Train and evaluate a deeper CNN model.

        Parameters:
        - X_train, y_train: training data
        - X_test, y_test: test data
        - cnn_params: dictionary of hyperparameters (optional). If None, default values are used.

        Returns:
        - model: trained Keras model
        - y_pred: predictions on the test set
        """
        if cnn_params is None:
            cnn_params = {
                'filters_1': 64,
                'filters_2': 64,
                'filters_3': 128,
                'filters_4': 128,
                'filters_5': 128,
                'dropout_1': 0.2,
                'dropout_2': 0.2,
                'dropout_3': 0.3,
                'dropout_4': 0.3,
                'dropout_5': 0.3,
                'dense_units': 64,
                'dense_dropout': 0.3,
                'lr': 0.001,
                'batch_size': 64,
                'epochs': 50
            }

        # Input shape for CNN
        input_shape = X_train.shape[1:]

        # Wrap into a dummy hp object for reuse
        from keras_tuner import HyperParameters
        hp = HyperParameters()
        for key, val in cnn_params.items():
            hp.Fixed(key, val)

        # Build and compile the model
        model = self.build_deeper_cnn(hp=hp, input_shape=input_shape)

        print("🚀 Training Deep CNN (5 layers)...")
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=cnn_params['epochs'],
            batch_size=cnn_params['batch_size'],
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1
        )

        # Predict and evaluate
        y_pred = model.predict(X_test).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"✅ RMSE: {rmse:.2f}")
        print(f"✅ MAE:  {mae:.2f}")
        print(f"✅ R2:   {r2:.2f}")

        return model, y_pred

    def perform_random_search(self, X_train, y_train, n_iter=20, cv=3): # n_iter=20, cv=3
        param_dist = {
            'n_estimators': randint(50, 300), #50, 200
            'max_depth': randint(5, 30), 
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5),
            'max_features': ['sqrt', 'log2', None]
        }
    
        rf = RandomForestRegressor(random_state=42)
    
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='neg_root_mean_squared_error',
            cv=cv,
            verbose= 3,
            n_jobs=-1,
            random_state=42
        )
    
        random_search.fit(X_train, y_train)
    
        print("✅ Best Parameters:", random_search.best_params_)
        print(f"✅ Best RMSE: {-random_search.best_score_:.2f}")
        
        return random_search

    def save_search_results(self, search_obj, prefix="search"):
        """Save RandomizedSearchCV object with a timestamped filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure output folder exists
        save_dir = "search_results"
        os.makedirs(save_dir, exist_ok=True)
    
        # Final path with timestamp
        filename = os.path.join(save_dir, f"{prefix}_object_{timestamp}.pkl")
    
        joblib.dump(search_obj, filename)
        print(f"✅ Saved: {filename}")

    def load_search_results(self, name):
        """Load saved RandomizedSearchCV object and cv_results_ CSV."""
        # Load full object
        search_obj = joblib.load(f"{name}")
    
        print(f"✅ Loaded: {name}")
        return search_obj

    def plot_model_performance(self, y_test, predictions_dict, save_path="model_performance_plots"):
        """
        Parameters:
        - y_test: true labels, shape (n_samples,)
        - predictions_dict: {'model_name': y_pred, ...}
        - save_path: directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(n_models, 1, figsize=(10, 4 * n_models))
    
        if n_models == 1:
            axes = [axes]
    
        # Store performance metrics
        performance = {
            'Model': [],
            'RMSE': [],
            'MAE': [],
            'R²': []
        }
    
        # Plot prediction curves and collect metrics
        for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
            ax = axes[i]
            ax.plot(y_test, label="True RUL", alpha=0.7)
            ax.plot(y_pred, label="Predicted RUL", alpha=0.7)
            ax.set_title(f"RUL Prediction - {model_name.upper()}")
            ax.set_xlabel("Sample")
            ax.set_ylabel("RUL")
            ax.legend()
            ax.grid(True)
    
            # Compute metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
    
            performance['Model'].append(model_name.upper())
            performance['RMSE'].append(rmse)
            performance['MAE'].append(mae)
            performance['R²'].append(r2)
    
        plt.tight_layout()
        pred_plot_path = os.path.join(save_path, "model_prediction_curves.png")
        fig.savefig(pred_plot_path)
        print(f"📸 Saved prediction curves to {pred_plot_path}")
        plt.show()
    
        # --- Side-by-side bar plots for RMSE, MAE, R² ---
        metrics = ['RMSE', 'MAE', 'R²']
        colors = ['steelblue', 'orange', 'green']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[i]
            values = performance[metric]
            ax.bar(performance['Model'], values, color=color)
            ax.set_title(f"Model Comparison - {metric}")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric)
            ax.grid(True, linestyle="--", alpha=0.5)
    
            # Auto-scale Y-axis around actual range
            min_val, max_val = min(values), max(values)
            buffer = 0.05 * (max_val - min_val) if max_val != min_val else 0.1
            ax.set_ylim(min_val - buffer, max_val + buffer)
    
        plt.tight_layout()
        summary_plot_path = os.path.join(save_path, "metric_comparison_barplots.png")
        fig.savefig(summary_plot_path)
        print(f"📸 Saved metric comparison to {summary_plot_path}")
        plt.show()

    # === Model builders ===
    def build_lstm_model(self, hp, input_shape):
        model = Sequential([
            InputLayer(shape=input_shape),
            LSTM(units=hp.Choice("units", [32, 64, 128]), return_sequences=False,
                 recurrent_dropout=hp.Choice("recurrent_dropout", [0.0, 0.1, 0.2])),
            Dropout(rate=hp.Choice("dropout", [0.2, 0.3, 0.4])),
            Dense(1)
        ])
        model.compile(
            optimizer=Adam(learning_rate=hp.Choice("lr", [0.001, 0.0005])),
            loss="mse"
        )
        return model
    
    def build_cnn_model(self, hp, input_shape):
        model = Sequential([
            InputLayer(shape=input_shape),
            Conv1D(filters=hp.Choice("filters", [32, 64]),
                   kernel_size=hp.Choice("kernel_size", [2, 3, 5]),
                   activation='relu'),
            Dropout(hp.Choice("dropout", [0.2, 0.3])),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=hp.Choice("lr", [0.001, 0.0005])), loss="mse")
        return model
    
    def build_ann_model(self, hp, input_dim):
        model = Sequential([
            InputLayer(shape=(input_dim,)),
            Dense(hp.Choice("units_1", [64, 128, 256]), activation='relu'),
            Dropout(hp.Choice("dropout", [0.2, 0.3])),
            Dense(hp.Choice("units_2", [32, 64]), activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=hp.Choice("lr", [0.001, 0.0005])), loss="mse")
        return model
    
    def build_deep_cnn(self, input_shape, learning_rate=0.001):
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))  # Explicit input shape

        # First convolutional block
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))

        # Second convolutional block
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))

        # Third convolutional block
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.4))

        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1))  # Output: Predicted RUL

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model

    def build_deeper_cnn(self, hp, input_shape, max_layers=5):
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))

        # Declare hyperparameters
        for i in range(max_layers):
            hp.Choice(f'filters_{i+1}', [32, 64, 128])
            hp.Choice(f'dropout_{i+1}', [0.2, 0.3, 0.4])

        hp.Choice("dense_units", [64, 128])
        hp.Choice("final_dropout", [0.2, 0.3])
        hp.Choice("lr", [0.001, 0.0005])

        # Add convolutional layers using declared hyperparameters
        for i in range(max_layers):
            filters = hp.get(f'filters_{i+1}')
            dropout = hp.get(f'dropout_{i+1}')
            model.add(Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(units=hp.get("dense_units"), activation='relu'))
        model.add(Dropout(hp.get("final_dropout")))
        model.add(Dense(1))

        learning_rate = hp.get("lr")
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

        return model

    # === Unified tuning method ===
    def tune_model(self, model_name, X, y, n_iter=40, cv=5):
        
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

            print(f"✅ Best score: {-search.best_score_:.2f}")
            print(f"✅ Best parameters:\n{search.best_params_}")
    
            self.save_search_results(search, "search_" + model_name)
    
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
            raise ValueError(f"❌ Unsupported model: {model_name}")

        print(f"🔍 Starting hyperparameter tuning for: {model_name.upper()}")

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

        print(f"✅ Best score: {-search.best_score_:.2f}")
        print(f"✅ Best parameters:\n{search.best_params_}")

        self.save_search_results(search, "search_" + model_name)

        return search

    def tune_lstm(self, X, y, max_trials=10):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuner_path = f"tuner_results/lstm_bayes_tuner_{timestamp}"

        input_shape = (X.shape[1], X.shape[2])
    
        tuner = kt.BayesianOptimization(
            hypermodel=lambda hp: self.build_lstm_model(hp, input_shape),
            objective="val_loss",
            max_trials=max_trials,
            overwrite=True,
            directory=tuner_path,
            project_name="rul_lstm_bayes"
        )
    
        tuner.search(
            X, y,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=2
        )
    
        return tuner

    def tune_cnn(self, X, y, max_trials=10):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuner_path = f"tuner_results/cnn_bayes_tuner_{timestamp}"

        input_shape = (X.shape[1], X.shape[2])
        
        tuner = kt.BayesianOptimization(
            hypermodel=lambda hp: self.build_cnn_model(hp, input_shape),
            objective="val_loss",
            max_trials=max_trials,
            overwrite=True,
            directory=tuner_path,
            project_name="rul_cnn_bayes"
        )
    
        tuner.search(
            X, y,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=2
        )
    
        return tuner

    def tune_ann(self, X, y, max_trials=10):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuner_path = f"tuner_results/ann_bayes_tuner_{timestamp}"
    
        tuner = kt.BayesianOptimization(
            hypermodel=lambda hp: self.build_ann_model(hp, X.shape[1]),
            objective="val_loss",
            max_trials=max_trials,
            overwrite=True,
            directory=tuner_path,
            project_name="rul_ann_bayes"
        )
    
        tuner.search(
            X, y,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=2
        )
    
        return tuner

    def tune_deep_cnn(self, X, y, max_trials=15):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuner_path = f"tuner_results/deep_cnn_bayes_tuner_{timestamp}"

        input_shape = (X.shape[1], X.shape[2])
        tuner = kt.BayesianOptimization(
            hypermodel=lambda hp: self.build_deeper_cnn(hp, input_shape, 3),
            objective="val_loss",
            max_trials=max_trials,
            overwrite=True,
            directory=tuner_path,
            project_name="rul_deep_cnn"
        )

        tuner.search(
            X, y,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=2
        )

        return tuner

    def save_tuner(self, tuner, model_name):
        """Save a Keras Tuner object to disk using a model-specific path and timestamp."""
        os.makedirs("tuner_results", exist_ok=True)
    
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuner_path = f"tuner_results/{model_name}_tuner_{timestamp}"
    
        tuner.save(tuner_path)
        print(f"✅ Tuner saved to {tuner_path}")

    # === Final Performance Evaluation Methods ===
    def plot_rul_prediction_scatter(self, y_test, y_pred, model_name="Model"):
        """Scatter plot: Predicted vs. True RUL"""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("True RUL")
        plt.ylabel("Predicted RUL")
        plt.title(f"{model_name} - Predicted vs. True RUL")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_rul_over_time(self, test_df, y_pred, model_name="Model", unit_col='unit_no', time_col='time_cycle'):
        """Line plot of predicted RUL over time for random engines"""
        test_df = test_df.copy()
        test_df['predicted_RUL'] = y_pred

        selected_units = test_df[unit_col].drop_duplicates().sample(3, random_state=42)
        plt.figure(figsize=(10, 6))

        for unit in selected_units:
            engine_df = test_df[test_df[unit_col] == unit]
            plt.plot(engine_df[time_col], engine_df['predicted_RUL'], label=f'Unit {unit}')

        plt.xlabel("Time Cycle")
        plt.ylabel("Predicted RUL")
        plt.title(f"{model_name} - Predicted RUL over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_residual_distribution(self, y_test, y_pred, model_name="Model"):
        """Histogram of residuals (prediction error)"""
        residuals = y_pred - np.ravel(y_test)
        plt.figure(figsize=(8, 5))
        sns.histplot(residuals, kde=True, bins=50, color='gray')
        plt.axvline(0, color='red', linestyle='--')
        plt.title(f"{model_name} - Prediction Error Distribution")
        plt.xlabel("Prediction Error (y_pred - y_true)")
        plt.tight_layout()
        plt.show()


    def plot_feature_importance(self, model, feature_names, model_name="Model"):
        """Horizontal bar chart of feature importances (for RF, XGB, etc.)"""
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️ {model_name} does not support feature importances.")
            return

        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        sorted_names = np.array(feature_names)[sorted_idx]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[sorted_idx], y=sorted_names)
        plt.title(f"{model_name} - Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()


    def plot_time_to_failure_heatmap(self, y_pred, units, max_length=150):
        """Heatmap of predicted RUL per time step per unit"""
        heatmap_data = {}
        for i, unit in enumerate(np.unique(units)):
            unit_mask = units == unit
            preds = y_pred[unit_mask]
            padded = np.pad(preds, (0, max_length - len(preds)), mode='constant', constant_values=np.nan)
            heatmap_data[unit] = padded

        df_heatmap = pd.DataFrame(heatmap_data).T
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_heatmap, cmap="coolwarm", cbar_kws={'label': 'Predicted RUL'}, mask=df_heatmap.isna())
        plt.xlabel("Time Step")
        plt.ylabel("Unit")
        plt.title("🔬 Time-to-Failure Heatmap (Predicted RUL)")
        plt.tight_layout()
        plt.show()

    def plot_all_evaluation_graphs(self, model, y_test, y_pred, model_name, test_df=None, feature_names=None, unit_col='unit_no', time_col='time_cycle'):
        """
        Generates all relevant plots to evaluate RUL prediction model performance.
        
        Parameters:
        - model: Trained model
        - y_test: True RUL values
        - y_pred: Predicted RUL values
        - model_name: String name for the model (e.g., "LSTM", "RF")
        - test_df: Optional DataFrame with test data (used for time-based plots)
        - feature_names: List of feature names (for feature importance)
        - unit_col: Column name for unit ID in test_df
        - time_col: Column name for time/cycle in test_df
        """

        self.plot_rul_prediction_scatter(y_test, y_pred, model_name=model_name)
        self.plot_residual_distribution(y_test, y_pred, model_name=model_name)

        # if test_df is not None:
            # self.plot_rul_over_time(test_df.copy(), y_pred, model_name=model_name, unit_col=unit_col, time_col=time_col)
            # self.plot_time_to_failure_heatmap(y_pred, test_df[unit_col].values)

        if feature_names and hasattr(model, "feature_importances_"):
            self.plot_feature_importance(model, feature_names, model_name=model_name)