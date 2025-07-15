from loss_functions import *
import numpy as np
from typing import Dict

class Sequential:
    def __init__(self, layers: list = None):
        self.layers = layers if layers else []
        self.optimizer = None
        self.loss_function = None
        self.history = {'loss': [], 'val_loss': []}
        

    
    def add(self, layer):
        self.layers.append(layer)
        
    def compile(self, optimizer=None, loss=None):
        if optimizer:
            self.optimizer = optimizer
        if loss:
            self.loss_function = LossFunction(loss.lower())
        
    def _forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    
    def _backward(self, y_pred, y_true, batch_size):
        d_out = self.loss_function.gradient(y_pred, y_true)
        grads = {}
        
        for i, layer in enumerate(reversed(self.layers)):
            layer_index = len(self.layers) - 1 - i
            layer_grads = layer.backward(d_out, batch_size)
            
            if hasattr(layer, 'weights'):
                grads[f'layer_{layer_index}_w'] = layer_grads.get('dw', 0)
            if hasattr(layer, 'bias'):
                grads[f'layer_{layer_index}_b'] = layer_grads.get('db', 0)
            
            d_out = layer_grads['dx']
        
        return grads
    
    def _update_params(self, grads):
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                params[f'layer_{i}_w'] = layer.weights
            if hasattr(layer, 'bias'):
                params[f'layer_{i}_b'] = layer.bias
        
        self.optimizer.update(params, grads)
        
       

    def _gradient_check(self, X, y, epsilon=1e-7):
        y_pred = self._forward(X)

        analytical_grads = self._backward(y_pred, y, batch_size=X.shape[0])
        analytical = []
        for i, layer in enumerate(self.layers):
            analytical.extend(analytical_grads[f'layer_{i}_w'].flatten())
            analytical.extend(analytical_grads[f'layer_{i}_b'].flatten())
        analytical = np.array(analytical)

        gradapprox = []

        for layer_index, layer in enumerate(self.layers):
            for idx in np.ndindex(layer.weights.shape):
                layer.weights[idx] += epsilon
                J_plus = self.loss_function.compute(self._forward(X), y)

                layer.weights[idx] -= 2 * epsilon
                J_minus = self.loss_function.compute(self._forward(X), y)

                layer.weights[idx] += epsilon

                grad = (J_plus - J_minus) / (2 * epsilon)
                gradapprox.append(grad)

            for idx in np.ndindex(layer.bias.shape):
                layer.bias[idx] += epsilon
                J_plus = self.loss_function.compute(self._forward(X), y)

                layer.bias[idx] -= 2 * epsilon
                J_minus = self.loss_function.compute(self._forward(X), y)

                layer.bias[idx] += epsilon

                grad = (J_plus - J_minus) / (2 * epsilon)
                gradapprox.append(grad)


        gradapprox = np.array(gradapprox)

        numerator = np.linalg.norm(analytical - gradapprox)
        denominator = np.linalg.norm(analytical) + np.linalg.norm(gradapprox) + 1e-10
        difference = numerator / denominator

        print("Gradient Check Difference:", difference)
        if difference < 1e-7:
            print(f"Gradient check passed! Difference: {difference:.2e}")
        else:
            print(f"Gradient check failed! Difference: {difference:.2e}")


    def fit(self, X, y, epochs, batch_size=None, validation_data=None):
        m = X.shape[0]
        batch_size = batch_size if batch_size else m
        
        self.history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            epoch_losses = []
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = self._forward(X_batch)
                grads = self._backward(output, y_batch, batch_size)

                self._update_params(grads)

                loss = self.loss_function.compute(output, y_batch)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            self.history['loss'].append(avg_loss)
            
            if validation_data:
                X_val, y_val = validation_data
                val_output = self._forward(X_val)
                val_loss = self.loss_function.compute(val_output, y_val)
                self.history['val_loss'].append(val_loss)
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
                
        return self.history

                                

    def predict(self, X):
        return self._forward(X)
