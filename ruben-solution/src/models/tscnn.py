from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, metrics, models, regularizers
from tensorflow.keras.utils import Sequence


class TimeSeriesCNN:
    def __init__(
        self,
        input_shape: Tuple[int],
        window_size: int,
        dropout_rate: float,
        l2_reg: float,
        filters: List[int],
        num_classes: int = 1,
    ):
        """
        Initialize the CNN model for time series data with a sliding window approach.

        Parameters:
        - input_shape (tuple): Shape of the input data (ecg_length, channels).
        - window_size (int): Size of the sliding window (number of timesteps).
        - num_classes (int): Number of classes for classification. Defaults to 1 for binary classification.
        - dropout_rate (float): Dropout rate for regularization.
        - l2_reg (float): L2 regularization factor.
        - filters (list): Number of filters for each Conv1D layer.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.window_size = window_size
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.filters = filters
        self.model = self.build_model()

    def build_model(self) -> models.Model:
        """
        Build and compile the CNN model.

        Returns:
        - tf.keras.Model: Compiled Keras model ready for training.
        """
        model = models.Sequential()
        model.add(layers.Input(shape=(self.window_size, self.input_shape[1])))

        for i, filter_count in enumerate(self.filters):
            model.add(
                layers.Conv1D(
                    filter_count,
                    kernel_size=5 if i < 2 else 3,
                    activation="relu",
                    padding="same",
                    kernel_regularizer=regularizers.l2(self.l2_reg),
                )
            )
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.Dropout(self.dropout_rate))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(self.l2_reg)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation="sigmoid"))

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", metrics.AUC(name="auc")])
        return model

    def train(
        self,
        train_generator: Sequence,
        val_generator: Sequence,
        callbacks: List[tf.keras.callbacks.Callback],
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> tf.keras.callbacks.History:
        """
        Train the CNN model.

        Parameters:
        - train_generator (Sequence): Data generator for training data.
        - val_generator (Sequence): Data generator for validation data.
        - callbacks (list[tf.keras.callbacks.Callback]): List of Keras callbacks to apply during training.
        - epochs (int): Number of training epochs. Defaults to 10.
        - batch_size (int): Number of samples per gradient update. Defaults to 32.
        - verbose (int): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

        Returns:
        - tf.keras.callbacks.History: Training history object containing training and validation metrics.
        """
        return self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            batch_size=batch_size,
        )

    def evaluate(self, test_generator: Sequence, verbose: int) -> Tuple[float, float, float]:
        """
        Evaluate the CNN model.

        Parameters:
        - test_generator (Sequence): Data generator for test data.
        - verbose (int): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Defaults to 1.

        Returns:
        - tuple: Test loss, test accuracy, and any other metrics specified in the model.
        """
        return self.model.evaluate(test_generator, verbose=verbose)

    def predict(self, test_generator: Sequence, verbose: int) -> np.ndarray:
        """
        Make predictions using the trained model.

        Parameters:
        - test_generator (Sequence): Data generator for test data.
        - verbose (int): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Defaults to 1.

        Returns:
        - np.ndarray: Predicted probabilities for each class.
        """
        return self.model.predict(test_generator, verbose=verbose)


class SlidingWindowGenerator(Sequence):
    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int, step_size: int, batch_size: int):
        """
        Initialize the sliding window generator.

        Parameters:
        - X (np.ndarray): Input data of shape (n_rows, ecg_length, channels).
        - y (np.ndarray): Binary labels corresponding to X.
        - window_size (int): Size of the sliding window.
        - step_size (int): Step size for the sliding window.
        - batch_size (int): Number of windows per batch.
        """
        super().__init__()
        self.X = X
        self.y = y
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.n_rows, self.ecg_length, self.channels = X.shape
        self.indices = np.arange(0, self.ecg_length - window_size + 1, step_size)
        self.on_epoch_end()

    def __len__(self) -> int:
        """
        Calculate the total number of batches.

        Returns:
        - int: Total number of batches.
        """
        num_windows_per_row = (self.ecg_length - self.window_size) // self.step_size + 1
        total_windows = self.n_rows * num_windows_per_row
        return int(np.ceil(total_windows / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data.

        Parameters:
        - index (int): Index of the batch.

        Returns:
        - tuple: Batch of data (X_batch, y_batch).
        """
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.indices) * self.n_rows)

        batch_indices = [(i // len(self.indices), i % len(self.indices)) for i in range(start_idx, end_idx)]

        return self.__data_generation(batch_indices)

    def on_epoch_end(self):
        """Shuffle the indices after each epoch."""
        np.random.shuffle(self.indices)

    def __data_generation(self, batch_indices: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for one batch.

        Parameters:
        - batch_indices (list): List of tuples containing row and window indices.

        Returns:
        - tuple: Arrays of shape (batch_size, window_size, channels) and (batch_size,).
        """
        X_batch = np.array(
            [
                self.X[row, self.indices[window_idx] : self.indices[window_idx] + self.window_size, :]
                for row, window_idx in batch_indices
            ]
        )
        y_batch = np.array([self.y[row] for row, window_idx in batch_indices])

        return X_batch, y_batch
