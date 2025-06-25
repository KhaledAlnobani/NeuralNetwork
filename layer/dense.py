import numpy as np
from activations import Activation

class Dense:
    def __init__(self, num_neurons, activation="linear"):
        # Initialize the Dense layer with the given number of neurons and activation function
        self.activation = Activation(activation)
        self.built = False  # Flag to check if weights are initialized
        self.input = None   # Store input for backward pass
        self.z = None       # Store pre-activation output
        self.num_neurons = num_neurons
    
    def build(self, input_size):
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, self.num_neurons)
        self.bias = np.zeros((1, self.num_neurons))

    def forward(self, x):
        # Forward pass: compute output of the layer
        if not self.built:
            self.build(x.shape[1])
            self.built = True
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias
        return self.activation.forward(self.z)
    
    def backward(self, d_out, batch_size):
        # Backward pass: compute gradients
        dz = self.activation.backward(self.z) * d_out  # Gradient w.r.t. pre-activation
        dw = np.dot(self.input.T, dz) / batch_size     # Gradient w.r.t. weights
        db = np.sum(dz, axis=0, keepdims=True) / batch_size  # Gradient w.r.t. bias
        dx = np.dot(dz, self.weights.T)                # Gradient w.r.t. input
        
        return {'dw': dw, 'db': db, 'dx': dx}