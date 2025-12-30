import numpy as np
from decision_trees import DecisionTree

class GradientBoostingRegressor:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=1):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.init_pred = None
    
    def fit(self, X, y):
        self.init_pred = y.mean()
        y_pred = np.full_like(y, self.init_pred, dtype=float)

        for _ in range(self.n_estimators):
            # Compute residuals 
            residuals = y - y_pred

            # Fit weak learner
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Update predictions
            y_pred += self.lr * tree.predict(X)

            self.models.append(tree)
    
    def predict(self, X):
        y_pred = np.full(X.shape[0],self.init_pred)
        for tree in self.models:
            y_pred += self.lr * tree.predict(X)
        return y_pred