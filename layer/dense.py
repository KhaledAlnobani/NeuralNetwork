import numpy as np
from activations import Activation

class Dense:
    def __init__(self, num_neurons, activation="linear"):
        self.activation = Activation(activation)
        self.built = False  
        self.input = None  
        self.z = None       
        self.num_neurons = num_neurons
    
    def build(self, input_size):
        
        self.weights = np.random.randn(input_size, self.num_neurons)
        self.bias = np.zeros((1, self.num_neurons))

    def forward(self, x):
      
        if not self.built:
            self.build(x.shape[1])
            self.built = True
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias
        return self.activation.forward(self.z)
    
    def backward(self, d_out, batch_size):
        
        dz = self.activation.backward(self.z) * d_out  
        dw = np.dot(self.input.T, dz) / batch_size    
        db = np.sum(dz, axis=0, keepdims=True) / batch_size  
        dx = np.dot(dz, self.weights.T)               
        
        return {'dw': dw, 'db': db, 'dx': dx}