# Neural Network from Scratch

This project demonstrates a modular, educational implementation of a Convolutional Neural Network (CNN) in pure Python and NumPy, with no deep learning frameworks required. It includes custom implementations of Conv2D, Pooling, Flatten, and Dense layers, as well as optimizers and loss functions. The code is designed for learning and experimentation, not for production or high performance.

---

## Features

- **Custom Layers:** Conv2D, Pooling (max/average), Flatten, Dense
- **Activation Functions:** ReLU, Sigmoid, Tanh, Softmax, Linear
- **Loss Functions:** MSE, Binary Cross-Entropy, Categorical Cross-Entropy
- **Optimizers:** Adam, RMSprop, SGD
- **Gradient Checking:** For debugging backpropagation
- **Training Utilities:** Training/validation split, accuracy, scaling, plotting
- **Educational:** Explicit loops for clarity, not optimized for speed

---

## File Structure

- `main.py` — Example usage and training pipeline on MNIST
- `layer/conv.py` — Convolutional layer (Conv2D)
- `layer/pooling.py` — Pooling layer (max/average)
- `layer/flatten.py` — Flatten layer
- `layer/dense.py` — Dense (fully connected) layer
- `activations.py` — Activation functions and derivatives
- `loss_functions.py` — Loss functions and derivatives
- `optimizers.py` — Adam, RMSprop, and SGD optimizers
- `model.py` — Sequential model class (forward, backward, fit, predict)
- `utils.py` — Data loading, preprocessing, plotting, accuracy, etc.

---

## Notes

- **Performance:** This code is for demonstration and learning. Training is slow due to explicit Python loops in Conv2D and Pooling.
- **Accuracy:** The goal is not state-of-the-art results, but to show how deep learning layers and training pipelines work.
- **Weight Initialization:** He initialization for ReLU, Xavier for others:
  ```python
  scale = np.sqrt(2. / input_size) if activation.__name__ == "relu" else np.sqrt(1. / input_size)
  ```

