import os
import scipy.io
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def load_mat_time_signals_to_df(folder_path):
    """
    Loads time-domain signals from all .mat files in a given folder and returns a DataFrame
    where each column represents one signal.

    Args:
        folder_path (str): Path to the folder containing .mat files.

    Returns:
        pd.DataFrame: A DataFrame where each column is a named signal from a .mat file.
    """
    all_data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.mat'):
            file_path = os.path.join(folder_path, filename)
            mat_contents = scipy.io.loadmat(file_path)

            for key, value in mat_contents.items():
                if "time" in key.lower() and isinstance(value, (np.ndarray, list)):
                    # Flatten to 1D if necessary
                    signal = np.squeeze(value)
                    col_name = f"{os.path.splitext(filename)[0]}_{key}"
                    all_data[col_name] = signal

                    # Print column name and number of samples
                    print(f"Loaded: {col_name}, shape[0] = {signal.shape[0]}")

    # Combine into DataFrame (columns = each signal)
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_data.items()]))

    return df

def generate_windows_from_signals(df, window_size=2048, step=1024):
    """
    Splits each signal column in the DataFrame into overlapping sliding windows and appends
    a label extracted from the column name.

    Args:
        df (pd.DataFrame): DataFrame with time-series signal columns.
        window_size (int): Number of samples per window.
        step (int): Number of samples to shift for each new window.

    Returns:
        pd.DataFrame: A DataFrame with rows as flattened windows and an associated label.
    """
    windowed_data = []
    for col in df.columns:
        signal = df[col].dropna().values
        label_match = re.match(r"([A-Za-z0-9@]+)_", col)
        label = label_match.group(1) if label_match else col  # fallback to full column name

        for i in range(0, len(signal) - window_size + 1, step):
            window = signal[i:i + window_size]
            windowed_data.append(np.append(window, label))

    # Create column names: a1 to a2048, plus 'label'
    feature_cols = [f"a{i}" for i in range(1, window_size + 1)] + ["label"]
    return pd.DataFrame(windowed_data, columns=feature_cols)

def balance_windows(df):
    """
    Balances the dataset so that all classes have the same number of samples 
    by undersampling to the minimum class size.

    Args:
        df (pd.DataFrame): DataFrame containing a 'label' column.

    Returns:
        pd.DataFrame: Balanced DataFrame with equal number of samples per class.
    """
    min_class_size = df['label'].value_counts().min()
    balanced_df = pd.concat([
        df[df['label'] == lbl].sample(n=min_class_size, random_state=42)
        for lbl in df['label'].unique()
    ])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

def apply_fft(X):
    """
    Applies real FFT to each sample along the time axis and normalizes the magnitude.

    Args:
        X (np.ndarray): Time-domain signals with shape (samples, time_steps).

    Returns:
        np.ndarray: Frequency-domain features with shape (samples, freq_bins).
    """
    X_fft = np.fft.rfft(X, axis=1)
    X_mag = np.abs(X_fft)
    X_norm = X_mag / np.max(X_mag)  # normalize globally
    return X_norm

def preprocess_fft_pipeline(df, window_size=1600, step=800, test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline for time-series data including windowing, scaling,
    FFT transformation, and formatting for ML and DL models.

    Args:
        df (pd.DataFrame): Input DataFrame with signal columns.
        window_size (int): Length of each sliding window.
        step (int): Overlap between sliding windows.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: A dictionary with preprocessed datasets and encoders for training and testing.
    """    
    # 1. Generate sliding windows
    window_df = generate_windows_from_signals(df, window_size=window_size, step=step)

    # 2. Extract features and labels
    X = window_df.drop(columns=["label"]).astype(np.float32).values
    y = window_df["label"].values

    # 3. Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. Train-test split
    X_train_raw, X_test_raw, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # 5. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 6. FFT transformation
    X_train_fft = apply_fft(X_train_scaled)
    X_test_fft = apply_fft(X_test_scaled)

    # 7. Prepare data for different models
    X_train_flat = X_train_fft
    X_test_flat = X_test_fft

    X_train_cnn = X_train_fft[..., np.newaxis]
    X_test_cnn = X_test_fft[..., np.newaxis]

    X_train_lstm = X_train_cnn  # Same shape
    X_test_lstm = X_test_cnn

    # 8. One-hot encoding
    y_train_cat = to_categorical(y_train_encoded)
    y_test_cat = to_categorical(y_test_encoded)

    return {
        "X_train_flat": X_train_flat,
        "X_test_flat": X_test_flat,
        "X_train_cnn": X_train_cnn,
        "X_test_cnn": X_test_cnn,
        "X_train_lstm": X_train_lstm,
        "X_test_lstm": X_test_lstm,
        "y_train_encoded": y_train_encoded,
        "y_test_encoded": y_test_encoded,
        "y_train_cat": y_train_cat,
        "y_test_cat": y_test_cat,
        "label_encoder": le
    }

def preprocess_time_pipeline(df, window_size=1600, step=800, test_size=0.2, random_state=42):
    """
    Preprocessing pipeline that keeps the data in the time domain, including
    window generation, label encoding, scaling, and format preparation.

    Args:
        df (pd.DataFrame): DataFrame with raw signals.
        window_size (int): Number of samples in each window.
        step (int): Overlap step between windows.
        test_size (float): Test set split ratio.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Dictionary with training and testing sets ready for ML/DL models.
    """    
    # 1. Generate windows
    window_df = generate_windows_from_signals(df, window_size=window_size, step=step)

    # 2. Extract features and labels
    X = window_df.drop(columns=["label"]).astype(np.float32).values
    y = window_df["label"].values

    # 3. Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. Stratified train-test split
    X_train_raw, X_test_raw, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # 5. Scale only training data, transform test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 6. Format for different model types
    X_train_flat = X_train_scaled
    X_test_flat = X_test_scaled

    X_train_cnn = X_train_scaled[..., np.newaxis]  # (samples, time_steps, 1)
    X_test_cnn = X_test_scaled[..., np.newaxis]

    X_train_lstm = X_train_cnn
    X_test_lstm = X_test_cnn

    # 7. One-hot encode labels for neural networks
    y_train_cat = to_categorical(y_train_encoded)
    y_test_cat = to_categorical(y_test_encoded)

    return {
        "X_train_flat": X_train_flat,
        "X_test_flat": X_test_flat,
        "X_train_cnn": X_train_cnn,
        "X_test_cnn": X_test_cnn,
        "X_train_lstm": X_train_lstm,
        "X_test_lstm": X_test_lstm,
        "y_train_encoded": y_train_encoded,
        "y_test_encoded": y_test_encoded,
        "y_train_cat": y_train_cat,
        "y_test_cat": y_test_cat,
        "label_encoder": le
    }

#------PLOTS------------------------------------------------------

def plot_tsne(X, y_encoded, label_encoder, title="t-SNE: Class Separability", save_path=None):
    """
    Plots a 2D t-SNE visualization of feature vectors with class labels.

    Args:
        X (np.ndarray): Feature matrix.
        y_encoded (np.ndarray): Encoded class labels.
        label_encoder (LabelEncoder): Fitted encoder to decode labels.
        title (str): Title of the plot.
        save_path (str or None): Path to save the plot image. If None, shows the plot.

    Returns:
        None
    """    
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    class_names = label_encoder.inverse_transform(sorted(set(y_encoded)))
    y_labels = label_encoder.inverse_transform(y_encoded)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_labels, palette="tab10", s=60, alpha=0.8)
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Class")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"t-SNE plot saved to {save_path}")
    else:
        plt.show()