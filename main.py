import numpy as np
import matplotlib.pyplot as plt
from layer.dense import Dense
from layer.conv import Conv2D
from layer.pooling import Pooling
from model import Sequential
from optimizers import Adam, RMSprop, SGD
from utils import *
from layer.flatten import Flatten


# Only a small subset of data is used because the goal is to implement and test 
# the model structure and training pipeline — not to achieve high accuracy.

X_train, y_train, X_test, y_test = load_mnist_full()

X_train, y_train, X_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train, y_train = X_train[:500], y_train[:500]
X_val, y_val = X_val[:200], y_val[:200]
X_test, y_test = X_test[:50], y_test[:50]

model = Sequential([
    Conv2D(num_filters=5, filter_shape=(3,3), padding="same", activation="relu"),
    Pooling(mode="max", pool_size=2, stride=2),
    Conv2D(num_filters=5, filter_shape=(3,3), padding="valid", activation="relu"),
    Pooling(mode="max", pool_size=2, stride=2),
    Flatten(),
    # Dense( 128, activation="relu"), 
    Dense( 64, activation="relu"),  
 
    Dense( 10, activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.01),
                loss='categorical_cross_entropy',)

history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=15, 
                    batch_size=32)

plot_training_history(history)

print("\nSample predictions:")
visualize_predictions(model, X_test[:5], y_test[:5], num_samples=5)


#The goal of this project is **not** to achieve state-of-the-art accuracy or to train a production-level model,
# but rather to ensure correct and modular implementation of deep learning layers and training pipeline logic.

