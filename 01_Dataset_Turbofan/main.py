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
    """
    A class to handle preprocessing, feature engineering, and test preparation 
    for Remaining Useful Life (RUL) prediction tasks using CMAPSS turbofan engine data.
    """
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
        """
        Loads training, test, and RUL label data from text files.

        Args:
            filename_train (str): Path to the training dataset.
            filename_test (str): Path to the test dataset.
            filename_y (str): Path to the file with actual RUL values for the test set.

        Returns:
            None
        """
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
        """
        Computes Remaining Useful Life (RUL) for each row in the DataFrame.

        Args:
            df (pd.DataFrame): Dataset with 'unit_no' and 'time_cycle' columns.

        Returns:
            pd.DataFrame: Updated DataFrame with an added 'RUL' column.
        """
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
        """
        Caps the RUL values at a maximum threshold.

        Args:
            df (pd.DataFrame): DataFrame with an 'RUL' column.
            cap (int): Maximum allowed RUL value.

        Returns:
            pd.DataFrame: DataFrame with capped RUL values.
        """
        if isinstance(cap, int) and cap > 0:
            df["RUL"] = df["RUL"].clip(upper=cap)
        return df

    def variance_threshold_analysis(self, df, threshold=0.01, export_path=None):
        """
        Applies variance threshold filtering to sensor columns and plots the variance.

        Args:
            df (pd.DataFrame): Dataset containing sensor data.
            threshold (float): Minimum variance for a sensor to be kept.
            export_path (str or None): If given, saves the plot to this path.

        Returns:
            list: List of sensor names with variance below the threshold.
        """
        print(f"Running Variance Threshold Analysis (threshold = {threshold})...")
    
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
            print(f"Variance plot saved to: {export_path}")
        else:
            plt.show()
    
        # Print sensors below threshold
        low_variance_sensors = variances_sorted[variances_sorted < threshold]
        print("Low-variance sensors to consider dropping:")
        print(low_variance_sensors)
    
        return low_variance_sensors.index.tolist()

    def remove_low_variance_features(self, df, features, threshold=0.01):
        """
        Removes features with variance below a threshold.

        Args:
            df (pd.DataFrame): DataFrame containing sensor features.
            features (list): List of feature (sensor) column names to check.
            threshold (float): Minimum variance threshold to retain a feature.

        Returns:
            tuple: (filtered DataFrame, list of retained feature names)
        """
        print(f"Removing low variance features (threshold={threshold})...")
        
        # Ensure we're only analyzing valid sensor features
        feature_cols = [col for col in features if col in df]
        
        # Apply variance threshold
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df[feature_cols])
        
        retained_features = list(np.array(feature_cols)[selector.get_support()])
        print(f"Retained {len(retained_features)} out of {len(feature_cols)} sensor features.")
        print(f"Retained features: {retained_features}")
        
        # Drop unretained features from original dataframe
        columns_to_keep = [col for col in df.columns if col not in feature_cols or col in retained_features]
        return df[columns_to_keep], retained_features
    
    def apply_signal_smoothing(self, df, window=5):
        """
        Applies rolling mean smoothing to all sensor signal columns.

        Args:
            df (pd.DataFrame): Input DataFrame with sensor readings.
            window (int): Window size for the moving average.

        Returns:
            pd.DataFrame: Smoothed DataFrame.
        """
        print(f"Applying rolling mean smoothing (window={window}) to sensor signals...")
        smoothed_df = df.copy()
        sensor_cols = [col for col in df.columns if col.startswith('s_')]
        for col in sensor_cols:
            smoothed_df[col] = df[col].rolling(window=window, min_periods=1).mean()
        print("Smoothing complete.")
        return smoothed_df

    def normalize(self, train_df, test_df, features):
        """
        Normalizes feature columns in training and test sets using MinMax scaling.

        Args:
            train_df (pd.DataFrame): Training dataset.
            test_df (pd.DataFrame): Test dataset.
            features (list): List of feature column names to normalize.

        Returns:
            None
        """
        feature_cols = [col for col in features if col in train_df]

        scaler = MinMaxScaler()

        # 1. Fit the scaler only on training data
        scaler = MinMaxScaler()
        scaler.fit(train_df[features])
        
        # 2. Transform training and test data separately
        train_df[features] = scaler.transform(train_df[features])
        test_df[features] = scaler.transform(test_df[features])

    def add_rolling_features(self, df, features, window=5):
        """
        Adds rolling mean and standard deviation as new features for each unit.

        Args:
            df (pd.DataFrame): Original dataset with sensor readings.
            features (list): List of feature columns to apply rolling stats.
            window (int): Size of the rolling window.

        Returns:
            pd.DataFrame: DataFrame with added rolling mean and std features.
        """
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
        """
        Generates fixed-length sequences (sliding windows) for model training.

        Args:
            df (pd.DataFrame): Input data with RUL labels.
            seq_length (int): Length of each input sequence.
            features (list): Columns to include in each input window.

        Returns:
            tuple: (X, y) where X is an array of shape (samples, seq_length, features),
                   and y is the corresponding RUL values.
        """
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
        """
        Prepares test data sequences of fixed length for each unit.

        Args:
            test_df (pd.DataFrame): Raw test data with features.
            rul_truth (np.ndarray): Ground truth RUL values for each test unit.
            features (list): Feature columns to include.
            seq_length (int): Desired sequence length per test sample.

        Returns:
            tuple: (X_test, y_test) — arrays ready for model evaluation.
        """
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