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

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import math

def plot_sensor_values(df, FEATURES, unit_no, n_cols=4, figsize=(20, 6 * 6), save_path=None):
    """
    Plots time series of sensor and setting values for a specified engine unit.

    Args:
        df (pd.DataFrame): Full dataset containing sensor readings.
        FEATURES (list): List of sensor or setting columns to plot.
        unit_no (int): Unit (engine) number to visualize.
        n_cols (int): Number of columns in the subplot grid.
        figsize (tuple): Size of the figure.
        save_path (str or None): If given, saves the plot to this path.

    Returns:
        None
    """
    df_unit = df[df['unit_no'] == unit_no]

    n_features = len(FEATURES)
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, feature in enumerate(FEATURES):
        ax = axes[idx]
        ax.plot(df_unit[feature].values)
        ax.set_title(f"{feature}")
        ax.set_xlabel("Time Cycle")
        ax.set_ylabel("Value")
        ax.grid(True)

    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Sensor & Setting Values for Unit {unit_no}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path)
        print(f"Saved sensor plots for unit {unit_no} to: {save_path}")
    else:
        plt.show()

def variance_threshold_analysis( df, threshold=0.01, export_path=None):
    """
    Analyzes and visualizes sensor variances, returning low-variance sensors.

    Args:
        df (pd.DataFrame): Dataset with sensor columns.
        threshold (float): Minimum variance threshold.
        export_path (str or None): If provided, saves the plot to this path.

    Returns:
        list: Names of low-variance sensor columns.
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

def remove_low_variance_features( df, features, threshold=0.01):
    """
    Removes features from a DataFrame with variance below a specified threshold.

    Args:
        df (pd.DataFrame): Input dataset.
        features (list): List of columns to evaluate for variance.
        threshold (float): Minimum variance threshold to retain a feature.

    Returns:
        tuple: (Filtered DataFrame, list of retained feature names)
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

def apply_signal_smoothing(df, window=5):
    """
    Applies rolling mean smoothing to sensor columns.

    Args:
        df (pd.DataFrame): DataFrame with time-series sensor data.
        window (int): Rolling window size.

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

def compute_snr_for_sensors(df, features, window=5, polyorder=2):
    """
    Computes Signal-to-Noise Ratio (SNR) for specified features using Savitzky-Golay filtering.

    Args:
        df (pd.DataFrame): Input DataFrame with raw signals.
        features (list): Columns to compute SNR on.
        window (int): Window size for Savitzky-Golay filter.
        polyorder (int): Polynomial order for smoothing.

    Returns:
        dict: Mapping from feature names to their computed SNR.
    """
    snr_results = {}
    for col in features:
        if col not in df:
            continue

        signal = savgol_filter(df[col], window_length=window, polyorder=polyorder)
        noise = df[col] - signal

        signal_var = np.var(signal)
        noise_var = np.var(noise)

        snr = np.inf if noise_var == 0 else signal_var / noise_var
        snr_results[col] = snr
        print(f"SNR for {col}: {snr:.2f}")
    
    return snr_results

def apply_savgol_filter(df, features, window_length=7, polyorder=2):
    """
    Applies Savitzky-Golay filter to smooth each selected feature in the dataset.

    Args:
        df (pd.DataFrame): Raw data with time-series signals.
        features (list): Sensor columns to filter.
        window_length (int): Size of the smoothing window (must be odd).
        polyorder (int): Polynomial order to use in the filter.

    Returns:
        pd.DataFrame: Smoothed DataFrame.
    """
    df_filtered = df.copy()
    sensor_cols = [col for col in features if col in df]
    
    for col in sensor_cols:
        try:
            df_filtered[col] = savgol_filter(df[col], window_length=window_length, polyorder=polyorder)
        except Exception as e:
            print(f"Could not filter {col}: {e}")

    print(f"Applied Savitzky-Golay filter to the passed features.")
    
    return df_filtered

def plot_snr_improvement(snr_before, snr_after, title="SNR Improvement After Denoising"):
    """
    Plots a bar chart showing SNR before and after denoising for each sensor.

    Args:
        snr_before (dict): Sensor-to-SNR mapping before filtering.
        snr_after (dict): Sensor-to-SNR mapping after filtering.
        title (str): Title of the plot.

    Returns:
        None
    """
    sensors = list(snr_before.keys())
    snr_vals_before = [snr_before[sensor] for sensor in sensors]
    snr_vals_after = [snr_after[sensor] for sensor in sensors]
    improvements = np.array(snr_vals_after) - np.array(snr_vals_before)

    x = np.arange(len(sensors))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, snr_vals_before, width, label='Before Filtering', color='skyblue')
    plt.bar(x + width/2, snr_vals_after, width, label='After Filtering', color='limegreen')

    # Annotate improvement
    for i, val in enumerate(improvements):
        plt.text(i, max(snr_vals_before[i], snr_vals_after[i]) + 0.5, f"+{val:.2f}", 
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

    plt.xticks(x, sensors, rotation=45, ha='right')
    plt.ylabel('SNR (Signal-to-Noise Ratio)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_snr_distributions(snr_before, snr_after, title="SNR Distributions Before and After Filtering"):
    """
    Displays histograms of SNR values before and after signal filtering.

    Args:
        snr_before (dict): SNR values before denoising.
        snr_after (dict): SNR values after denoising.
        title (str): Overall title for the figure.

    Returns:
        None
    """
    snr_vals_before = list(snr_before.values())
    snr_vals_after = list(snr_after.values())

    plt.figure(figsize=(12, 5))

    # --- Plot SNR before filtering ---
    plt.subplot(1, 2, 1)
    sns.histplot(snr_vals_before, bins=10, kde=True, color="skyblue")
    plt.axvline(np.mean(snr_vals_before), color='blue', linestyle='--', label=f"Mean: {np.mean(snr_vals_before):.2f}")
    plt.title("SNR Before Filtering")
    plt.xlabel("SNR")
    plt.ylabel("Count")
    plt.legend()

    # --- Plot SNR after filtering ---
    plt.subplot(1, 2, 2)
    sns.histplot(snr_vals_after, bins=10, kde=True, color="limegreen")
    plt.axvline(np.mean(snr_vals_after), color='green', linestyle='--', label=f"Mean: {np.mean(snr_vals_after):.2f}")
    plt.title("SNR After Filtering")
    plt.xlabel("SNR")
    plt.ylabel("Count")
    plt.legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def compute_snr_stats(snr_before, snr_after):
    """
    Computes and prints summary statistics (mean and std) for SNR before and after filtering.

    Args:
        snr_before (dict): SNR values before processing.
        snr_after (dict): SNR values after processing.

    Returns:
        dict: Summary statistics with keys 'mean_before', 'std_before', 'mean_after', 'std_after'.
    """
    snr_before_vals = list(snr_before.values())
    snr_after_vals = list(snr_after.values())

    summary = {
        "mean_before": np.mean(snr_before_vals),
        "std_before": np.std(snr_before_vals),
        "mean_after": np.mean(snr_after_vals),
        "std_after": np.std(snr_after_vals)
    }

    print(f"Mean SNR Before: {summary['mean_before']:.2f}, Std: {summary['std_before']:.2f}")
    print(f"Mean SNR After:  {summary['mean_after']:.2f}, Std: {summary['std_after']:.2f}")

    return summary

#-----FREQUENCY DOMAIN----------------------------------
def apply_fft_to_sliding_windows(X):
    """
    Applies real FFT to time-series sliding windows along the time axis.

    Args:
        X (np.ndarray): Input array with shape (samples, time_steps, features).

    Returns:
        np.ndarray: Magnitude spectrum of the FFT, normalized globally.
    """
    # Apply rfft along axis=1 (time steps), output shape will be (samples, freq_bins, sensors)
    X_fft = np.fft.rfft(X, axis=1)  # shape: (48559, 11, 24) for rfft of 20 points
    
    # Compute magnitude (real-valued features)
    X_fft_mag = np.abs(X_fft)
    
    # Normalize (optional)
    X_fft_mag /= np.max(X_fft_mag)

    return X_fft_mag