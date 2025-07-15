import numpy as np


class Pooling:
    def __init__(self, mode="max", pool_size=2, stride=2):
        self.mode = mode.lower()
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.mask = None  


    def forward(self, x):
        """
        Forward pass of pooling layer (educational implementation)
        
        Note: This implementation uses explicit loops for clarity in demonstrating the
        pooling operation mechanics.
        """
        self.input = x
        batch_size, input_height, input_width, input_channels = x.shape
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, output_height, output_width, input_channels))
        self.mask = np.zeros_like(x) if self.mode == "max" else None
        
        for i in range(batch_size):
            for h in range(output_height):
                for w in range(output_width):
                    for c in range(input_channels):
                        v_start = h * self.stride
                        v_end = v_start + self.pool_size
                        h_start = w * self.stride
                        h_end = h_start + self.pool_size
                        
                        x_slice = x[i, v_start:v_end, h_start:h_end, c]
                        
                        if self.mode == "max":
                            output[i, h, w, c] = np.max(x_slice)
                            if self.mask is not None:
                                mask_slice = (x_slice == np.max(x_slice))
                                self.mask[i, v_start:v_end, h_start:h_end, c] = mask_slice
                        elif self.mode == "average":
                            output[i, h, w, c] = np.mean(x_slice)
        
        return output
    

    def backward(self, d_out, batch_size=None):

        """ 
        Backward pass of pooling layer (educational implementation)

        Note: This implementation uses explicit loops for clarity in demonstrating the
        pooling operation mechanics.
        """
        dx = np.zeros_like(self.input)
        pool_area = self.pool_size * self.pool_size
        
        num_samples,output_height,output_width,output_channels = d_out.shape
        for i in range(num_samples):
            for h in range(output_height):
                for w in range(output_width):
                    for c in range(output_channels):
                        v_start = h * self.stride
                        v_end = v_start + self.pool_size
                        h_start = w * self.stride
                        h_end = h_start + self.pool_size
                        
                        if self.mode == "max":
                            dx[i, v_start:v_end, h_start:h_end, c] += (
                                self.mask[i, v_start:v_end, h_start:h_end, c] * 
                                d_out[i, h, w, c]
                            )
                        elif self.mode == "average":
                            dx[i, v_start:v_end, h_start:h_end, c] += (
                                d_out[i, h, w, c] / pool_area
                            )
        
        return {'dx': dx}  