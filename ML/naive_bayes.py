import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-9
            self.prior[c] = len(X_c) / len(X)
    
    def _log_likelihood(self, x, c):
        mean = self.mean[c]
        var = self.var[c]
        return -0.5 * np.sum(
            np.log(2 * np.pi * var)
            + ((x - mean) ** 2) / var
        )
    
    def predict_one(self, x):
        posteriors = []

        for c in self.classes:
            log_prior = np.log(self.prior[c])
            log_likelihood = self._log_likelihood(x, c)
            posteriors.append(log_prior + log_likelihood)
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])
    