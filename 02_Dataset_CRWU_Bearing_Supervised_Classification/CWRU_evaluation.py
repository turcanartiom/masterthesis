import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import os

def plot_prediction_accuracies(predictions_dict, y_true, save_path=None, filename="model_accuracy_plot.png"):
    """
    Plots the mean accuracy with standard deviation for each model across multiple predictions.

    Args:
        predictions_dict (dict): Dictionary where keys are model names and values are lists of predictions.
        y_true (np.ndarray): Ground truth labels for evaluation.
        save_path (str or None): If provided, saves the plot to the given directory.
        filename (str): Filename to use if saving the plot.

    Returns:
        None
    """
    model_names = []
    mean_accuracies = []
    std_devs = []
    all_accuracies = []

    for model_name, preds in predictions_dict.items():
        accs = [accuracy_score(y_true, y_pred) for y_pred in preds]
        model_names.append(model_name)
        mean_accuracies.append(np.mean(accs))
        std_devs.append(np.std(accs))
        all_accuracies.extend(accs)

    # Calculate dynamic y-axis limits
    min_acc = min(all_accuracies)
    max_acc = max(all_accuracies)
    padding = 0.05
    y_min = max(0, min_acc - padding)
    y_max = min(1.0, max_acc + padding)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, mean_accuracies, yerr=std_devs, capsize=8, alpha=0.8)
    plt.ylabel("Accuracy")
    # plt.title("Model Accuracy with Deviation over Multiple Evaluations")
    plt.ylim(y_min, y_max)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate bars
    for i, (mean, std) in enumerate(zip(mean_accuracies, std_devs)):
        plt.text(i, mean + std + 0.005, f"{mean:.3f} ± {std:.3f}", ha='center', va='bottom')

    plt.tight_layout()

    # Save the plot if path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path)
        print(f"\nPlot saved to: {full_path}")

    plt.show()

def get_best_model_and_prediction(predictions_dict, y_true):
    """
    Identifies the model and prediction with the highest accuracy from multiple predictions.

    Args:
        predictions_dict (dict): Dictionary mapping model names to lists of predicted label arrays.
        y_true (np.ndarray): True labels for accuracy comparison.

    Returns:
        tuple: (best_model_name, best_y_pred) — model name with highest accuracy and its prediction.
    """
    best_score = 0
    best_model_name = None
    best_y_pred = None

    for model_name, preds in predictions_dict.items():
        for i, y_pred in enumerate(preds):
            acc = accuracy_score(y_true, y_pred)
            if acc > best_score:
                best_score = acc
                best_model_name = model_name
                best_y_pred = y_pred

    print(f"\nBest model: {best_model_name} with accuracy: {best_score:.4f}")
    return best_model_name, best_y_pred

def plot_confusion_matrix_for_model(y_true, y_pred, model_name="Model",hp = "0HP", save_path=None, label_map=None, filename=None):
    """
    Plots and optionally saves the confusion matrix for a specific model's predictions.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels by the model.
        model_name (str): Name of the model for plot title and filename.
        hp (str): Hyperparameter configuration name or identifier.
        save_path (str or None): Directory path to save the plot (if provided).
        label_map (dict or None): Optional mapping from class indices to readable labels.
        filename (str or None): Custom filename to save the confusion matrix image.

    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred)

    if label_map:
        sorted_keys = sorted(label_map.keys())
        target_names = [label_map[k] for k in sorted_keys]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    else:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f"Confusion Matrix - {model_name} - Load: {hp}")
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)

        # Use custom filename if provided
        if filename:
            file_path = os.path.join(save_path, filename)
        else:
            # Fallback to auto-generated filename
            auto_name = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            file_path = os.path.join(save_path, auto_name)

        plt.savefig(file_path)
        print(f"Confusion matrix saved to: {file_path}")

    plt.show()