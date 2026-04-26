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
        sizes = [self.input_size] + self.hidden_size + [self.output_size]

        for i in range(len(sizes) - 1):
            scale = np.sqrt(2 / sizes[i])
            w = np.random.randn(sizes[i], sizes[i+1]) * scale
            b = np.zeros((1, sizes[i+1]))

            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        self.activations = [X]
        self.z_arr = []

        tmp = X 
        for i in range(len(self.weights) - 1):
            z_arr = tmp @ self.weights[i] + self.biases[i]
            self.z_arr.append(z_arr)
            a = np.maximum(0, z_arr)
            self.activations.append(a)

            tmp = a

        z_last = tmp @ self.weights[-1] + self.biases[-1]
        self.z_arr.append(z_last)

        output = self.softmax(z_last)
        self.activations.append(output)

        return output
    
    def backward(self, y_true):
        m = y_true.shape[0]

        self.d_weights = [np.zeros_like(w) for w in self.weights]
        self.d_biases = [np.zeros_like(b) for b in self.biases]

        dz = self.activations[-1] - y_true

        for i in range(len(self.weights));
            self.d_weights[i] = (self.activations[i].T @ dz) / m 
            self.d_biases[i] = np.sum(dz, axis=0, keepdims=True) / m 

            if i > 0:
                da_prev = dz @ self.weights[i].T 
                dz = da_prev * self.d_activations(self.z_arr[i - 1])
    
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
    
