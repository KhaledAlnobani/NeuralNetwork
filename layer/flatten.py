class Flatten:
    def __init__(self):
        self.input_shape = None
        
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, d_out, batch_size=None): 
        return {'dx': d_out.reshape(self.input_shape)} 