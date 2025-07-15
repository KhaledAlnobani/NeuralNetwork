import numpy as np

class Activation:
    def __init__(self, activation_type=None):
        if activation_type is None:
            self.activation_type = 'linear'
        else:
            self.activation_type = activation_type.lower()

    def linear(self, x):
        return x

    def d_linear(self, x):
        return np.ones(x.shape)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def d_sigmoid(self, x):
        return self.sigmoid(x) * (1-self.sigmoid(x))
     
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def d_tanh(self, x):
        return 1-(self.tanh(x))**2

    def ReLU(self, x):
        return x * (x > 0)

    def d_ReLU(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def d_softmax(self, x):
        # The derivative of softmax is more complex and usually handled with cross-entropy
        # For practical purposes, we return 1 since the gradient is handled in the loss function
        return np.ones(x.shape)

    def get_activation(self, x):
        if self.activation_type == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_type == 'tanh':
            return self.tanh(x)
        elif self.activation_type == 'relu':
            return self.ReLU(x)
        elif self.activation_type == 'linear':
            return self.linear(x)
        elif self.activation_type == 'softmax':
            return self.softmax(x)
        else:
            raise ValueError("Valid Activations are 'sigmoid', 'linear', 'tanh', 'relu', and 'softmax'")

    def get_d_activation(self, x):
        if self.activation_type == 'sigmoid':
            return self.d_sigmoid(x)
        elif self.activation_type == 'tanh':
            return self.d_tanh(x)
        elif self.activation_type == 'relu':
            return self.d_ReLU(x)
        elif self.activation_type == 'linear':
            return self.d_linear(x)
        elif self.activation_type == 'softmax':
            return self.d_softmax(x)
        else:
            raise ValueError("Valid Activations are 'sigmoid', 'linear', 'tanh', 'relu', and 'softmax'")

    def forward(self, X):
        return self.get_activation(X)
    
    def backward(self, z):
        return self.get_d_activation(z)
