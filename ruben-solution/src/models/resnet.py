from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import layers, metrics, models, regularizers


class ECGResNet:
    def __init__(self, input_shape: Tuple[int], l2_reg: float, dropout_rate: float, num_classes: int = 1):
        """
        Initialize ResNet model for ECG time series data.

        Parameters:
        - input_shape (tuple): Shape of the input data (timesteps, channels).
        - num_classes (int): Number of classes for classification. Defaults to 1 for binary classification.
        - l2_reg (float): L2 regularization factor. Defaults to 0.001.
        - dropout_rate (float): Dropout rate. Defaults to 0.5.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def residual_block(self, x: tf.Tensor, filters: int, kernel_size: int = 3) -> tf.keras.models:
        """
        Create a residual block with two Conv1D layers and a skip connection.

        Parameters:
        - x (tf.Tensor): Input tensor.
        - filters (int): Number of filters for the convolutional layers.
        - kernel_size (int): Size of the convolutional kernel. Defaults to 3.

        Returns:
        - tf.keras.models: A model containing the residual block.
        """
        shortcut = x
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(self.l2_reg),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation=None,
            kernel_regularizer=regularizers.l2(self.l2_reg),
        )(x)
        x = layers.BatchNormalization()(x)

        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters=filters, kernel_size=1, padding="same")(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    def build_model(self) -> tf.keras.Model:
        """
        Build and compile the ResNet model.

        Returns:
        - tf.keras.Model: Compiled Keras model ready for training.
        """
        inputs = layers.Input(shape=self.input_shape)

        x = layers.Conv1D(64, kernel_size=7, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = self.residual_block(x, filters=64)
        x = self.residual_block(x, filters=128)
        x = self.residual_block(x, filters=256)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.Dropout(self.dropout_rate)(x)

        outputs = layers.Dense(self.num_classes, activation="sigmoid")(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="binary_crossentropy",
            metrics=["accuracy", metrics.AUC(name="auc")],
        )
        return model

    def train(
        self,
        X_train: tf.Tensor,
        y_train: tf.Tensor,
        X_val: tf.Tensor,
        y_val: tf.Tensor,
        callbacks: List[tf.keras.callbacks.Callback] = [],
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> tf.keras.callbacks.History:
        """
        Train the ResNet model.

        Parameters:
        - X_train (tf.Tensor): Training data of shape (num_samples, timesteps, channels).
        - y_train (tf.Tensor): Training labels of shape (num_samples, 1).
        - X_val (tf.Tensor): Validation data of shape (num_samples, timesteps, channels).
        - y_val (tf.Tensor): Validation labels of shape (num_samples, 1).
        - callbacks (list): List of Keras callbacks to apply during training. Defaults to an empty list.
        - epochs (int): Number of training epochs. Defaults to 10.
        - batch_size (int): Number of samples per gradient update. Defaults to 32.
        - verbose (int): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Defaults to 1.

        Returns:
        - tf.keras.callbacks.History: Training history object containing training and validation metrics.
        """
        return self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
        )

    def evaluate(self, X_test: tf.Tensor, y_test: tf.Tensor, verbose: int = 0) -> Tuple[float, float]:
        """
        Evaluate the ResNet model.

        Parameters:
        - X_test (tf.Tensor): Test data of shape (num_samples, timesteps, channels).
        - y_test (tf.Tensor): Test labels of shape (num_samples, 1).
        - verbose (int): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Defaults to 1.

        Returns:
        - tuple: A tuple containing the test loss and test AUC score.
        """
        return self.model.evaluate(X_test, y_test, verbose=verbose)

    def predict(self, X: tf.Tensor, verbose: int = 0) -> tf.Tensor:
        """
        Make predictions using the trained model.

        Parameters:
        - X (tf.Tensor): Input data of shape (num_samples, timesteps, channels).
        - verbose (int): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Defaults to 1.

        Returns:
        - tf.Tensor: Predicted probabilities for each sample.
        """
        return self.model.predict(X, verbose=verbose)
