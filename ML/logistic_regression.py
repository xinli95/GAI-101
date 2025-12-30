
import numpy as np

class LogisticRegressionGD():
    def __init__(self, lr, epochs):
        self.lr = lr 
        self.epochs = epochs
        self.w = None
        self.b = None
        
    def sigmoid(self, z):
        return 1/ (1 + np.exp(-z))

    
    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = self.sigmoid(z)

            dw = X.T @ (p-y) / n
            db = np.sum(p-y) / n

            self.w -= self.lr * dw
            self.b -= self.lr * db
        
    def predict_probas(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_probas(X) >= threshold).astype(int)
