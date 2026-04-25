import numpy as np
from tqdm import tqdm

class Classifier:
    def __init__(self, 
                 input_size=784, 
                 hidden_size=[128, 64],
                 output_size=10,
                 learning_rate=0.01,
                 epochs=10,
                 batch_size=64):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.weights = []
        self.biases = []
        self._init_weights()

    def _set_activation(self):
        return 0
    
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
    
