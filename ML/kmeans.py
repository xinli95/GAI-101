
import numpy as np

class Kmeans:

    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
    
    def fit(self, X):
        n, d = X.shape
        idx = np.random.choice(n, self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels==i].mean(axis=0) for i in range(self.k)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return labels