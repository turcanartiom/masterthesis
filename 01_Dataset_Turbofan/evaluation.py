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

def plot_model_performance(y_test, predictions_dict, save_path="Plots/model_performance_plots"):
    """
    Plots RMSE, MAE, and R² bar charts with standard deviation across model predictions.

    Args:
        y_test (np.ndarray): Ground truth labels for regression.
        predictions_dict (dict): Dictionary where keys are model names and values are lists of predicted arrays.
        save_path (str): Directory to save the performance plot.

    Returns:
        None
    """
    os.makedirs(save_path, exist_ok=True)

    performance = {
        'Model': [],
        'RMSE_mean': [], 'RMSE_std': [],
        'MAE_mean': [], 'MAE_std': [],
        'R²_mean': [], 'R²_std': []
    }

    for model_name, y_preds in predictions_dict.items():
        y_preds = np.array(y_preds)  # (n_runs, n_samples)

        rmses, maes, r2s = [], [], []
        for y_pred in y_preds:
            rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            maes.append(mean_absolute_error(y_test, y_pred))
            r2s.append(r2_score(y_test, y_pred))

        performance['Model'].append(model_name.upper())
        performance['RMSE_mean'].append(np.mean(rmses))
        performance['RMSE_std'].append(np.std(rmses))
        performance['MAE_mean'].append(np.mean(maes))
        performance['MAE_std'].append(np.std(maes))
        performance['R²_mean'].append(np.mean(r2s))
        performance['R²_std'].append(np.std(r2s))

    # --- Summary Bar Plots with Variance ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['RMSE', 'MAE', 'R²']
    colors = ['steelblue', 'darkorange', 'green']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[i]
        mean_vals = np.array(performance[f"{metric}_mean"])
        std_vals = np.array(performance[f"{metric}_std"])
        models = performance['Model']

        ax.bar(models, mean_vals, yerr=std_vals, capsize=5, color=color)
        ax.set_title(f"{metric} Comparison")
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Y-axis limit with 10% buffer
        min_val = (mean_vals - std_vals).min()
        max_val = (mean_vals + std_vals).max()
        buffer = 0.1 * (max_val - min_val) if max_val != min_val else 0.1
        ax.set_ylim(min_val - buffer, max_val + buffer)

    plt.tight_layout()
    summary_path = os.path.join(save_path, "performance_summary.png")
    fig.savefig(summary_path)
    print(f"Saved performance summary to: {summary_path}")
    plt.show()

# === Final Performance Evaluation Methods ===
def plot_rul_prediction_scatter(y_test, y_pred, model_name="Model"):
    """
    Creates a scatter plot comparing predicted RUL values to true RUL values.

    Args:
        y_test (np.ndarray): Ground truth RUL values.
        y_pred (np.ndarray): Predicted RUL values.
        model_name (str): Name of the model used for predictions.

    Returns:
        None
    """
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

def plot_rul_over_time(test_df, y_pred, model_name="Model", unit_col='unit_no', time_col='time_cycle'):
    """
    Plots predicted RUL over time for three randomly selected units.

    Args:
        test_df (pd.DataFrame): DataFrame containing test metadata (e.g., unit_no, time_cycle).
        y_pred (np.ndarray): Predicted RUL values.
        model_name (str): Model identifier for the plot title.
        unit_col (str): Column name representing the unit/engine identifier.
        time_col (str): Column name representing the time step or cycle.

    Returns:
        None
    """
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

def plot_residual_distribution(y_test, y_pred, model_name="Model"):
    """
    Plots a histogram of residuals (prediction error) with a KDE overlay.

    Args:
        y_test (np.ndarray): Ground truth RUL values.
        y_pred (np.ndarray): Predicted RUL values.
        model_name (str): Name of the model used for predictions.

    Returns:
        None
    """
    residuals = y_pred - np.ravel(y_test)
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=50, color='gray')
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"{model_name} - Prediction Error Distribution")
    plt.xlabel("Prediction Error (y_pred - y_true)")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, model_name="Model"):
    """
    Displays a horizontal bar chart of feature importances from a model (e.g., RF or XGB).

    Args:
        model: A trained model with a feature_importances_ attribute.
        feature_names (list): List of feature names corresponding to the model input.
        model_name (str): Name of the model for the plot title.

    Returns:
        None
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"{model_name} does not support feature importances.")
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

def plot_time_to_failure_heatmap(y_pred, units, max_length=150):
    """
    Visualizes predicted RUL over time for each unit using a heatmap.

    Args:
        y_pred (np.ndarray): Predicted RUL values.
        units (np.ndarray): Unit identifiers matching the predictions.
        max_length (int): Max sequence length for heatmap padding.

    Returns:
        None
    """
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
    plt.title("Time-to-Failure Heatmap (Predicted RUL)")
    plt.tight_layout()
    plt.show()

def plot_all_evaluation_graphs(model, y_test, y_pred, model_name, test_df=None, feature_names=None, unit_col='unit_no', time_col='time_cycle'):
    """
    Plots a comprehensive set of evaluation graphs: scatter plot, residuals, feature importance,
    and optionally RUL over time.

    Args:
        model: Trained regression model.
        y_test (np.ndarray): Ground truth RUL values.
        y_pred (np.ndarray): Predicted RUL values.
        model_name (str): Name to display in plots.
        test_df (pd.DataFrame or None): Test data with metadata for RUL over time plot.
        feature_names (list or None): List of input feature names (used in feature importance plot).
        unit_col (str): Column name for unit IDs in test_df.
        time_col (str): Column name for time steps in test_df.

    Returns:
        None
    """
    plot_rul_prediction_scatter(y_test, y_pred, model_name=model_name)
    plot_residual_distribution(y_test, y_pred, model_name=model_name)

    if feature_names and hasattr(model, "feature_importances_"):
            plot_feature_importance(model, feature_names, model_name=model_name)