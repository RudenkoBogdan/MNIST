from torchvision import datasets, transforms
import numpy as np

class DataLoader:
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        self.X_train = np.array([img.numpy().flatten() for img, _ in train_dataset])
        self.y_train = np.array([label for _, label in train_dataset])
        self.X_test = np.array([img.numpy().flatten() for img, _ in test_dataset])
        self.y_test = np.array([label for _, label in test_dataset])

    def get_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def batch(self, size=64, train=True):
        data, labels = (self.X_train, self.y_train) if train else (self.X_test, self.y_test)
        for i in range(0, len(data), size):
            yield data[i : i + size], labels[i : i + size]