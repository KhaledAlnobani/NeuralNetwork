import numpy as np


class Pooling:
    def __init__(self, mode="max", pool_size=2, stride=2):
        self.mode = mode.lower()
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.mask = None  
        
    def forward(self, x):
        self.input = x
        m, h_prev, w_prev, c_prev = x.shape
        h_out = (h_prev - self.pool_size) // self.stride + 1
        w_out = (w_prev - self.pool_size) // self.stride + 1
        
        output = np.zeros((m, h_out, w_out, c_prev))
        self.mask = np.zeros_like(x) if self.mode == "max" else None
        
        for i in range(m):
            for h in range(h_out):
                for w in range(w_out):
                    for c in range(c_prev):
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
        if self.mask is None and self.mode == "max":
            raise ValueError("Max pooling requires mask from forward pass")
        
        d_input = np.zeros_like(self.input)
        pool_area = self.pool_size * self.pool_size
        
        m,h_out,w_out,c_out = d_out.shape
        for i in range(m):
            for h in range(h_out):
                for w in range(w_out):
                    for c in range(c_out):
                        v_start = h * self.stride
                        v_end = v_start + self.pool_size
                        h_start = w * self.stride
                        h_end = h_start + self.pool_size
                        
                        if self.mode == "max":
                            d_input[i, v_start:v_end, h_start:h_end, c] += (
                                self.mask[i, v_start:v_end, h_start:h_end, c] * 
                                d_out[i, h, w, c]
                            )
                        elif self.mode == "average":
                            d_input[i, v_start:v_end, h_start:h_end, c] += (
                                d_out[i, h, w, c] / pool_area
                            )
        
        return {'dx': d_input}  