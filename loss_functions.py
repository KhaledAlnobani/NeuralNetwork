import numpy as np
<<<<<<< HEAD
class LossFunction:
    def __init__(self, loss="mse"):
        self.loss_type = loss.lower()
        if self.loss_type not in ["mse", "binary_cross_entropy", "categorical_cross_entropy"]:
            raise ValueError("Unsupported loss type. Use 'mse', 'binary_cross_entropy', or 'categorical_cross_entropy'.")
=======
def binary_cross_entropy(y_pred, y_true,epsilon = 1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
>>>>>>> 7de3706 (final commit)

    def binary_cross_entropy(self, y_pred, y_true, epsilon=1e-12):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

<<<<<<< HEAD
    def binary_cross_entropy_prime(self, y_pred, y_true, epsilon=1e-12):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_pred.shape[0])
=======
def binary_cross_entropy_prime(y_pred, y_true,epsilon = 1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true / y_pred + (1 - y_true) / (1 - y_pred)
>>>>>>> 7de3706 (final commit)

    def categorical_cross_entropy(self, y_pred, y_true, epsilon=1e-12):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

    def categorical_cross_entropy_prime(self, y_pred, y_true):
        return y_pred - y_true

    def mse(self, y_pred, y_true):
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_prime(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / np.size(y_true)
    
    def compute(self, y_pred, y_true):
        if self.loss_type == "mse":
            return self.mse(y_pred, y_true)
        elif self.loss_type == "binary_cross_entropy":
            return self.binary_cross_entropy(y_pred, y_true)
        elif self.loss_type == "categorical_cross_entropy":
            return self.categorical_cross_entropy(y_pred, y_true)

    def gradient(self, y_pred, y_true):
        if self.loss_type == "mse":
            return self.mse_prime(y_pred, y_true)
        elif self.loss_type == "binary_cross_entropy":
            return self.binary_cross_entropy_prime(y_pred, y_true)
        elif self.loss_type == "categorical_cross_entropy":
            return self.categorical_cross_entropy_prime(y_pred, y_true)

