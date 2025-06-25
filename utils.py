import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt



def load_mnist_full():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Convert labels to one-hot encoding
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    
    # Normalize and reshape
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    return X_train, y_train, X_test, y_test



def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, X_test, y_test, num_samples=5):
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}")
        plt.axis('off')
    plt.show()

    
def scale(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_scaled = (X - mean) / std
    return X_scaled


def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true) * 100.0


def train_test_split(X, y, random_state=None, test_size=0.2):
    if random_state is not None:
        np.random.seed(random_state)
    X = np.array(X)
    y = np.array(y)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * test_size)
    X = X[indices, :]
    y = y[indices, :]
    X_train, y_train = X[split:], y[split:]
    X_test, y_test = X[:split], y[:split]
    return X_train, y_train, X_test, y_test


