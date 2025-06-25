import numpy as np 
from activations import Activation

class Conv2D:
    def __init__(self, num_filters=1, filter_shape=(3,3), stride=1, padding="valid", activation="linear"):
        self.num_filters = num_filters
        self.filter_height, self.filter_width = filter_shape
        self.stride = stride
        self.activation = Activation(activation)
        self.built = False
        
        if padding == "valid":
            self.padding = 0
        elif padding == "same":
            self.padding = (self.filter_height - 1) // 2
        else:
            raise ValueError("Padding must be 'valid' or 'same'")
    def build(self, input_shape):
        _, _, _, in_channels = input_shape
        scale = np.sqrt(2. / (in_channels * self.filter_height * self.filter_width))
        self.weights = np.random.randn(self.filter_height, self.filter_width, in_channels, self.num_filters) * scale
        self.bias = np.zeros((1, 1, 1, self.num_filters))
        self.built = True

    # Note: This implementation prioritizes clarity and educational value over performance.
    # For production or large-scale use, consider optimizing with vectorized operations or libraries like CuDNN.
        
    def forward(self, x):  
        if not self.built:
            self.build(x.shape)
            self.built = True
            
        self.input = x
        batch_size, input_height, input_width, input_channels = x.shape
        
        output_height = (input_height - self.filter_height + 2*self.padding) // self.stride + 1
        output_width = (input_width - self.filter_width + 2*self.padding) // self.stride + 1
        
        x_pad = np.pad(x, ((0,0), (self.padding, self.padding), 
                      (self.padding, self.padding), (0,0), 
                      ), mode='constant', constant_values=0)
        z = np.zeros((batch_size, output_height, output_width, self.num_filters))
        
        for i in range(batch_size):
            for h in range(output_height):
                for w in range(output_width):
                    for f in range(self.num_filters):
                        v_start = h * self.stride
                        v_end = v_start + self.filter_height
                        h_start = w * self.stride
                        h_end = h_start + self.filter_width
                        
                        x_slice = x_pad[i, v_start:v_end, h_start:h_end, :]
                        z[i, h, w, f] = np.sum(
                            x_slice * self.weights[:, :, :, f]
                        ) + self.bias[0, 0, 0, f]
        
        self.z = z
        return self.activation.forward(z)
    
    def backward(self, d_out, batch_size=None):
        d_out = self.activation.backward(self.z) * d_out
        m, h_in, w_in, c_in = self.input.shape
        self.filter_height, self.filter_width, _, _ = self.weights.shape
        
        dx = np.zeros_like(self.input)
        dw = np.zeros_like(self.weights)
        db = np.zeros_like(self.bias)
        
        x_pad = np.pad(self.input, ((0,0), (self.padding, self.padding), 
                      (self.padding, self.padding), (0,0)))
        dx_pad = np.pad(dx, ((0,0), (self.padding, self.padding), 
                           (self.padding, self.padding), (0,0)))
        
        for i in range(m):
            for h in range(d_out.shape[1]):
                for w in range(d_out.shape[2]):
                    for f in range(self.num_filters):
                        v_start = h * self.stride
                        v_end = v_start + self.filter_height
                        h_start = w * self.stride
                        h_end = h_start + self.filter_width
                        
                        x_slice = x_pad[i, v_start:v_end, h_start:h_end, :]
                        
                        dw[:, :, :, f] += x_slice * d_out[i, h, w, f]
                        
                        dx_pad[i, v_start:v_end, h_start:h_end, :] += (
                            self.weights[:, :, :, f] * d_out[i, h, w, f]
                        )
                        
                        db[0, 0, 0, f] += d_out[i, h, w, f]
        
        if self.padding > 0:
            dx = dx_pad[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dx = dx_pad
            
        return {'dx': dx, 'dw': dw, 'db': db}