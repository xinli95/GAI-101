import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict_one(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        idx = np.argsort(distances)[:self.k]
        labels = self.y_train[idx]
        return Counter(labels).most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])
    