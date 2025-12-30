# Machine Learning Interview Preparation — Core Models

This document summarizes the **core ML models frequently tested in interviews**, focusing on **intuition, assumptions, objectives, and trade-offs** rather than code implementation.

---

## 1. Linear Regression

### What it models

- Models the conditional mean of a continuous target as a linear function of features
- Assumes:  
  $$ y = Xw + b + \varepsilon $$

### Noise / data assumption

- Additive **Gaussian noise**: $$ \varepsilon \sim \mathcal{N}(0, \sigma^2) $$
- Errors are symmetric, zero-mean, and homoscedastic

### Loss function

- Mean Squared Error (MSE)
- Equivalent to **negative log-likelihood under Gaussian noise**

### Optimization view

- Convex objective → global optimum
- Can be solved via normal equation or gradient descent

### Bias–variance profile

- High bias if relationship is non-linear
- Low variance

### When it works well

- Linear relationships
- Well-behaved noise

---

## 2. Logistic Regression

### What it models

- Models **probability** of a binary outcome
- Linear model on the **log-odds**:
  $$ \log \frac{p}{1-p} = Xw + b $$

### Noise / data assumption

- Target follows a **Bernoulli distribution**

### Loss function

- Log loss / binary cross-entropy
- Negative log-likelihood of Bernoulli model

### Key intuition

- Penalizes **confident wrong predictions heavily**
- Produces calibrated probabilities

### Optimization view

- Convex objective
- Gradient has clean form: $$ X^T (p - y) $$

### Bias–variance profile

- Lower bias than linear regression for classification
- Still relatively low variance

---

## 3. k-Means Clustering

### What it models

- Partitions data into $k$ clusters by minimizing within-cluster variance

### Objective

- Minimize sum of squared Euclidean distances to cluster centroids

### Algorithmic structure

- Alternating minimization:
  - Assignment step (nearest centroid)
  - Update step (mean of assigned points)

### Assumptions

- Clusters are **spherical** and similar in size
- Euclidean distance is meaningful

### Optimization characteristics

- Non-convex objective
- Converges to local minimum

### Failure modes

- Non-spherical clusters
- Different densities
- Sensitive to initialization

---

## 4. k-Nearest Neighbors (kNN)

### What it models

- Instance-based, non-parametric method
- No explicit training phase

### Prediction rule

- Predict based on labels of the $k$ closest points

### Key assumptions

- Similar points have similar labels
- Distance metric captures similarity

### Bias–variance tradeoff

- Small $k$: low bias, high variance
- Large $k$: high bias, low variance

### Practical considerations

- Highly sensitive to feature scaling
- Inference cost scales with dataset size

---

## 5. Decision Trees

### What it models

- Hierarchical partitioning of feature space
- Learns non-linear decision boundaries

### Split criteria

- Entropy (information theory)
- Gini impurity (misclassification probability)

### Optimization view

- Greedy, top-down local optimization
- Maximizes information gain (entropy reduction)

### Key properties

- Low bias, **high variance**
- No need for feature scaling
- Handles mixed data types

### Failure modes

- Overfitting
- Unstable to small data changes

---

## 6. Naive Bayes

### What it models

- Generative classifier: models $$ P(x, y) $$

### Core assumption

- Features are conditionally independent given the class

### Decision rule

- Choose class with maximum posterior probability
- Computed in log space for numerical stability

### Bias–variance profile

- **High bias**, very low variance
- Strong regularization via independence assumption

### When it works well

- Small datasets
- High-dimensional sparse data (e.g., text)

---

## 7. Random Forests

### What it models

- Ensemble of decision trees trained independently

### Two sources of randomness

- Bootstrap sampling (bagging)
- Feature subsampling at each split

### Key intuition

- Averaging **decorrelated trees** reduces variance

### Bias–variance profile

- Low bias
- Much lower variance than single trees

### Out-of-Bag (OOB) estimation

- Uses unused bootstrap samples as validation data
- Efficient alternative to cross-validation

---

## 8. Gradient Boosting

### What it models

- Additive model built sequentially
- Each stage improves previous model

### Core idea

- Perform **gradient descent in function space**
- Each new model fits the **negative gradient of the loss**

### Key distinction

- Residual fitting is a special case (squared loss)
- General framework works for any differentiable loss

### Bias–variance profile

- Strong bias reduction
- Can overfit without regularization

### Regularization mechanisms

- Learning rate
- Shallow trees
- Early stopping

---

## 9. High-Level Comparison

| Model             | Bias     | Variance | Key Strength     |
| ----------------- | -------- | -------- | ---------------- |
| Linear / Logistic | High     | Low      | Interpretability |
| kNN               | Low      | High     | Local modeling   |
| Naive Bayes       | High     | Low      | Small data       |
| Decision Tree     | Low      | High     | Non-linearity    |
| Random Forest     | Low      | Low      | Robust default   |
| Gradient Boosting | Very Low | Medium   | Accuracy         |
