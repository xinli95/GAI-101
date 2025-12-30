import numpy as np
from collections import Counter

def entropy(y):
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum(p * np.log2(p) for p in probs if p > 0)

def information_gain(y, y_left, y_right):
    n = len(y)
    return (
        entropy(y)
        - (len(y_left) / n) * entropy(y_left)
        - (len(y_right) / n) * entropy(y_right)
    )

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue
                gain =information_gain(y, y[left_idx], y[right_idx])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth):
        num_samples = len(y)
        num_labels = len(set(y))

        # Stopping conditions
        if (
            depth >= self.max_depth
            or num_labels == 1
            or num_samples <= self.min_samples_split
        ):
            leaf_value = Counter(y).most_common(1)[0][0]
            return TreeNode(value=leaf_value)
        
        feature, threshold = self._best_split(X, y)

        if feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return TreeNode(value=leaf_value)
        
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        left_child = self._build_tree(X[left_idx], y[left_idx], depth+1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth+1)

        return TreeNode(
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child
        )
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])
    