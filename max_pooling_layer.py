class MaxPoolingLayer(Layer):
    def __init__(self, pool_size=(2, 2)):
        self.pool_size = pool_size
        
    def forward_propagation(self, input_data):
        self.input = input_data
        input_height, input_width, depth = input_data.shape
        
        output_height = input_height // self.pool_size[0]
        output_width = input_width // self.pool_size[1]
        self.output = np.zeros((output_height, output_width, depth))
        
        for d in range(depth):  # Apply max pooling to each depth slice
            for i in range(0, input_height, self.pool_size[0]):
                for j in range(0, input_width, self.pool_size[1]):
                    self.output[i//2, j//2, d] = np.max(input_data[i:i+2, j:j+2, d])
        
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        return None  # Not implementing backprop for pooling in this example
