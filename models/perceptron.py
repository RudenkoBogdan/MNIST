import numpy as np

class Perceptron:
    def __init__(
            self,
            input_size = 784,
            n_classes = 10,
            learning_rate = 1e-3,
            epochs = 30,
            batch_size = 64,
            random_seed = 42):
        
        self.input_size = input_size
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.weights = None

        self.rng = np.random.default_rng(random_seed)

    def _init_weights(self):
        self.weights = self.rng.normal(0.0, 2/self.input_size, size=(self.input_size, self.n_classes))
        self.bias = self.rng.normal(0.0, 2/self.n_classes, size=self.n_classes)

    def forward(self, x):
        logits = x @ self.weights + self.bias[None, :]
        self.logits = logits         
        return logits

    def _softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def _cross_entropy_loss():
        return 0

    def backward():
        return 0 

    def fit():
        return 0

    def predict():
        return 0

    def predict_proba():
        return 0 
    
    def evaluate():
        return 0