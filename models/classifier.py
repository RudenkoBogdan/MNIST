import numpy as np
from tqdm import tqdm

class Classifier:
    def __init__(self, 
                 input_size=784, 
                 hidden_size=[128, 64],
                 output_size=10,
                 learning_rate=0.01,
                 epochs=10,
                 batch_size=64,
                 activation_func='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation_func = activation_func

        self.weights = []
        self.biases = []
        self._init_weights()
        self.activation = self._set_activation(activation_func)

    def _set_activation(self, name):
        activation_map = {
            'relu': lambda x: np.maximum(0, x)
            'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
            'tanh': lambda x: np.tanh(x),
            'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
            'elu': lambda x: np.where(x > 0, x, np.exp(x) - 1),
            'swish': lambda x: x * (1 / (1 + np.exp(-x))),
            'gelu': lambda x: x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))),
            'linear': lambda x: x }
        return activation_map.get(name)
        
    
    def _init_weights(self):
        return 0
    
    def forward(self, X):
        return 0
    
    def bachward(self, y_true):
        return 0
    
    def _update_parameters(self):
        return 0
    
    def fit(self, X_train, y_train):
        return 0
    
    def predict(self, X):
        return 0 
    
    def predict_proba(self, X):
        return 0
    
    def evaulate(self, X, y):
        return 0
    
    def visualize(self):
        return 0
    
