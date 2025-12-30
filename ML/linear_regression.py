
import numpy as np

class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):

        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0

        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b

            dw = (2/n) * X.T @ (y_pred - y)
            db = (2/n) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db
    
    def predict(self, X):
        return X @ self.w + self.b