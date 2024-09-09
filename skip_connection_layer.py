from layer import Layer

class SkipConnection(Layer):
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2
        
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.layer1.forward_propagation(input_data) + self.layer2.forward_propagation(input_data)
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        error1 = self.layer1.backward_propagation(output_error, learning_rate)
        error2 = self.layer2.backward_propagation(output_error, learning_rate)
        return error1 + error2
