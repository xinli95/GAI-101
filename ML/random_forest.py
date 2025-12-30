
from decision_trees import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
    
    def fit(self, X, y):
        n = len(X)

        for _ in range(self.n_trees):

            # Bootstrap sample
            idx = np.random.choice(n, n, replace=True)
            X_sample = X[idx]
            y_sample = y[idx]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        preds = []

        for i in range(len(X)):
            preds.append(Counter(tree_preds[:,i]).most_common(1)[0][0])
        return np.array(preds)
    