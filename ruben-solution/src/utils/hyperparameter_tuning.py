from typing import Any, Callable, List, Tuple

import numpy as np
import pandas as pd
from models.tscnn import SlidingWindowGenerator
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from utils.data_preprocessing import create_generators_or_datasets


def evaluate_cnn_performance(
    model: Model, data_generator: SlidingWindowGenerator, true_labels: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    """
    Evaluate the performance of a CNN model on a given dataset using a SlidingWindowGenerator.

    Parameters:
    - model (Model): The trained Keras model to be evaluated.
    - data_generator (SlidingWindowGenerator): A generator that generates batches of data using a sliding window approach.
    - true_labels (np.ndarray): The true labels for the dataset.

    Returns:
    - Tuple[float, float, float, np.ndarray]: Returns the loss, accuracy, AUC, and the predictions as a tuple.
    """
    predictions = model.predict(data_generator, verbose=0)
    num_windows_per_sample = len(predictions) // len(true_labels)
    aggregated_predictions = predictions.reshape(-1, num_windows_per_sample).mean(axis=1)
    predicted_classes = np.round(aggregated_predictions)

    accuracy = accuracy_score(true_labels, predicted_classes)
    auc = roc_auc_score(true_labels, aggregated_predictions)

    loss = model.evaluate(data_generator, verbose=0)
    if isinstance(loss, (list, tuple)):
        loss = loss[0]

    return loss, accuracy, auc, aggregated_predictions


def evaluate_resnet_performance(model: Model, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """
    Evaluate the performance of a ResNet model on a given dataset.

    Parameters:
    - model (Model): The trained Keras model to be evaluated.
    - X (np.ndarray): Input data of shape (num_samples, timesteps, channels).
    - y (np.ndarray): True labels corresponding to X.

    Returns:
    - Tuple[float, float, float, np.ndarray]: Returns the loss, accuracy, AUC, and the predictions as a tuple.
    """
    predictions = model.predict(X, verbose=0)
    predicted_classes = np.round(predictions)

    accuracy = accuracy_score(y, predicted_classes)
    auc = roc_auc_score(y, predictions)

    loss = model.evaluate(X, y, verbose=0)
    if isinstance(loss, (list, tuple)):
        loss = loss[0]

    return loss, accuracy, auc, predictions


def objective(
    trial: Any,
    objective_metric: str,
    model_class: Callable,
    input_shape: Tuple[int, int],
    Y: pd.DataFrame,
    sampling_rate: int,
    sample_size: float,
    data_path: str,
    batch_size: int,
    num_classes: int,
    epochs: int,
    filters: List[int],
) -> float:
    """
    Objective function for Optuna to optimize hyperparameters for a CNN model.

    Parameters:
    - trial (Any): A trial object used to suggest values for the hyperparameters.
    - objective_metric (str): The metric to optimize, e.g., 'val_loss'.
    - model_class (Callable): The CNN model class to instantiate.
    - input_shape (Tuple[int, int]): Shape of the input data (timesteps, channels).
    - Y (pd.DataFrame): DataFrame containing the labels and other metadata.
    - sampling_rate (int): The sampling rate for the ECG data.
    - sample_size (float): The fraction of the data to sample.
    - data_path (str): The path to the raw data directory.
    - batch_size (int): Number of windows per batch.
    - num_classes (int): Number of classes for the output layer.
    - epochs (int): Number of training epochs.
    - filters (List[int]): List of filter sizes for each convolutional layer.

    Returns:
    - float: The objective_metric score for the current set of hyperparameters.
    """
    window_size = trial.suggest_int("window_size", 50, 200)
    step_size = trial.suggest_int("step_size", 10, window_size)
    l2_reg = trial.suggest_float("l2_reg", 1e-12, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    train_generator, val_generator, test_generator = create_generators_or_datasets(
        y=Y,
        window_size=window_size,
        step_size=step_size,
        batch_size=batch_size,
        sample_size=sample_size,
        sampling_rate=sampling_rate,
        data_path=data_path,
        return_generators=True,
    )

    cnn = model_class(
        input_shape=input_shape,
        window_size=window_size,
        num_classes=num_classes,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        filters=filters,
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = cnn.train(train_generator, val_generator, epochs=epochs, verbose=0, callbacks=[early_stopping])

    return history.history[objective_metric][-1]
