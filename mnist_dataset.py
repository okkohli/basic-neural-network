from tensorflow.keras.datasets import mnist
import numpy as np


class Dataset:
    def __init__(self, n_train=1000, n_test=1000):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Shuffle training data:
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Shuffle test data:
        indices = np.random.permutation(len(X_test))
        X_test = X_test[indices]
        y_test = y_test[indices]

        self._input_dim = X_train.shape[1] * X_train.shape[2]                   # Flattened size
        self._n_classes = np.maximum(np.max(y_train) + 1, np.max(y_test) + 1)   # Number of classes

        # Select subset, normalize and flatten:
        self._X_train = (X_train[:n_train].astype("float32") / 255.0).reshape(n_train, self._input_dim).T

        self._X_test = (X_test[:n_test].astype("float32") / 255.0).reshape(n_test, self._input_dim).T

        # Select labels and one-hot encode:
        self._y_train = self._one_hot_encode(y_train[:n_train])
        self._y_test = self._one_hot_encode(y_test[:n_test])

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def n_classes(self):
        return self._n_classes

    def _one_hot_encode(self, y):
        one_hot = np.zeros((y.shape[0], self._n_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot.T

    def get_mnist_data(self):
        return (
            self._X_train,
            self._y_train,
            self._X_test,
            self._y_test
        )


if __name__ == "__main__":
    dataset = Dataset(n_train=1000, n_test=100)
    X_train, y_train, X_test, y_test = dataset.get_mnist_data()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")