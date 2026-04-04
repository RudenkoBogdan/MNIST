import numpy as np

class Perceptron:
    def __init__(
            self,
            input_size = 784,
            n_classes = 10,
            learning_rate = 1e-3,
            epochs = 30,
            batch_size = 64,
            random_seed =42):
        
        self.input_size = input_size
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.rng = np.random.default_rng(random_seed)

    def _init_weights():

    def forward():

    def _softmax():

    def _cross_entropy_loss():

    def backward():

    def fit():

    def predict():

    def predict_proba():
    
    def evaluate():
