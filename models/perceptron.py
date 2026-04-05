import numpy as np
from tqdm import tqdm

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
    
    def _cross_entropy_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
        loss_per_sample = -np.log(y_pred[np.arange(len(y_true)), y_true])
        return np.mean(loss_per_sample)

    def backward(self, X, y_true):
        probs = self._softmax(self.logits)

        y_true = np.eye(self.n_classes)[y_true]

        dL_dout = probs - y_true

        N = self.logits.shape[0]
        self.dw = X.T @ dL_dout / N
        self.db = np.sum(dL_dout, axis=0) / N
        self.dX = dL_dout @ self.weights.T

        return self.dX

    def fit(self, X, y):
        self._init_weights
        self.history = []

        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            batch_loss = 0

            for i in tqdm(range(0, len(X), self.batch_size), desc="Batches", leave=False):
                X_batch = X[i : i + self.batch_size]
                y_batch = y[i : i + self.batch_size]

                logits = self.forward(X_batch)
                probs = self._softmax(logits)

                self.backward(X_batch, y_batch)

                self.weights -= self.learning_rate * self.dw 
                self.bias -= self.learning_rate * self.db 

                batch_loss += self._cross_entropy_loss(y_batch, probs)
            
            loss = batch_loss / (len(X) // self.batch_size)
            self.history.append(loss)

    def predict():
        return 0

    def predict_proba():
        return 0 
    
    def evaluate():
        return 0