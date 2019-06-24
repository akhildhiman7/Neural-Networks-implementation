import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        self.activation_function = lambda x : 1/(1 + np.exp(-x))
    
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            ### Forward pass ###
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)
            hidden_outputs = self.activation_function(hidden_inputs)
            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
            # since the last layer just passes on its value, we don't have to apply the sigmoid here.
            final_outputs = final_inputs
            
            ### Backward pass ###
            error = y - final_outputs
            # The derivative of the activation function y=x is 1
            output_error_term = error * 1.0
            hidden_error = np.dot(self.weights_hidden_to_output, error) 
            # Backpropagated error terms
            hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
            # Weight step (input to hidden)
            delta_weights_i_h += hidden_error_term * X[:,None]
            # Weight step (hidden to output)
            delta_weights_h_o += output_error_term * hidden_outputs[:,None]
        # Weights update
        self.weights_hidden_to_output += self.lr*delta_weights_h_o/n_records
        self.weights_input_to_hidden += self.lr*delta_weights_i_h/n_records
        
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        # Forward pass
        hidden_inputs =  np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs 
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 3000
learning_rate = 1.1
hidden_nodes = 15
output_nodes = 1
