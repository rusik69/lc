package machinelearning

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMachineLearningModules([]problems.CourseModule{
		{
			ID:          2519,
			Title:       "Classical ML Algorithms Deep Dive",
			Description: "Master decision trees, ensemble methods, SVMs, clustering algorithms, dimensionality reduction, and their implementations from scratch.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "Trees Ensembles SVMs Clustering and Dimensionality Reduction",
					Content: `Classical machine learning algorithms remain foundational. Understanding their internals is crucial for selecting the right algorithm and tuning for optimal performance.

**Decision Trees:**

Algorithm (CART — Classification and Regression Trees):
  1. For each feature, find best split point
  2. Select feature/split that maximizes information gain
  3. Split dataset into two subsets
  4. Recursively repeat for each subset
  5. Stop when max depth, min samples, or pure node reached

Split criteria (Classification):
  Gini Impurity: 1 - sum(p_i^2)
    Probability of misclassifying a random element
    Range: 0 (pure) to 0.5 (binary, equal split)
    
  Entropy: -sum(p_i * log2(p_i))
    Information content of the distribution
    Range: 0 (pure) to 1 (binary, equal split)
    
  Information Gain: parent_entropy - weighted_avg(children_entropy)
    Higher = better split

Split criteria (Regression):
  MSE: mean((y - y_mean)^2) for each subset
  MAE: mean(|y - y_median|) for each subset
  Variance reduction: parent_variance - weighted_avg(children_variance)

Pruning:
  Pre-pruning (early stopping):
    max_depth: Maximum tree depth
    min_samples_split: Minimum samples to split
    min_samples_leaf: Minimum samples in leaf
    max_features: Random subset for split consideration
    
  Post-pruning (cost-complexity pruning):
    Build full tree, then remove branches
    Cost-complexity: R(T) + alpha * |T|
    Higher alpha → more pruning

Pros: Interpretable, no feature scaling, handles mixed types
Cons: Overfitting, unstable (small data changes → different tree), axis-aligned splits

**Random Forest:**

Algorithm:
  1. Bootstrap N samples from training data (bagging)
  2. For each tree, at each split, consider random sqrt(features)
  3. Build full tree (no pruning)
  4. Repeat for n_trees (100-1000)
  5. Prediction: majority vote (classification) or mean (regression)

Key hyperparameters:
  n_estimators: Number of trees (100-1000)
  max_features: Features per split (sqrt(n) classification, n/3 regression)
  max_depth: Tree depth (None for full trees is common)
  min_samples_leaf: Minimum samples in leaf

Feature Importance:
  Mean decrease in impurity (MDI): Sum of impurity reductions at each split
  Permutation importance: Decrease in score when feature values shuffled
  
  MDI is biased toward high-cardinality features
  Permutation importance is more reliable

Out-of-Bag (OOB) error:
  ~37% of samples not used in each tree (bootstrap)
  Use OOB samples for validation
  No separate validation set needed

Pros: Low variance, robust, parallel training, feature importance
Cons: Slow inference (many trees), less interpretable, memory intensive

**Gradient Boosted Trees:**

Algorithm (Gradient Boosting):
  1. Initialize with constant prediction (mean for regression)
  2. For each iteration:
     a. Compute residuals (negative gradient of loss)
     b. Fit tree to residuals
     c. Find optimal leaf values (line search)
     d. Update prediction: F_m = F_{m-1} + learning_rate * tree_m

Key concepts:
  Residuals: What the current ensemble gets wrong
  Learning rate (shrinkage): Scale each tree's contribution (0.01-0.3)
  Number of trees: More trees with smaller learning rate
  Tree depth: Usually 3-8 (interaction depth)
  
  Regularization:
    Learning rate (shrinkage): Slow learning
    Subsampling: Use fraction of data per tree (0.5-0.8)
    Column subsampling: Random features per tree/split/level
    Max depth: Limit tree complexity
    Min samples: Minimum samples per leaf
    L1/L2 regularization on leaf values

XGBoost:
  Regularized objective: Loss + L1 + L2 on leaf weights
  Second-order gradient (Newton's method): Uses Hessian
  Histogram-based: Bin continuous features for faster splits
  Sparsity-aware: Native missing value handling
  Parallel tree construction: Split finding parallelized
  
  Score gain for split:
    Gain = 0.5 * (G_L^2/H_L + G_R^2/H_R - (G_L+G_R)^2/(H_L+H_R)) - gamma
    G = sum of gradients, H = sum of hessians
    gamma = minimum gain to split

LightGBM:
  Leaf-wise tree growth (vs level-wise)
  GOSS: Gradient-based One-Side Sampling
  EFB: Exclusive Feature Bundling (sparse features)
  Faster than XGBoost on large datasets
  
CatBoost:
  Ordered target encoding for categorical features
  Ordered boosting to prevent target leakage
  Symmetric trees (same split at each depth level)

**Support Vector Machines (SVM):**

Linear SVM:
  Find hyperplane maximizing margin between classes
  margin = 2 / ||w||
  
  Hard margin: No misclassifications allowed (separable data)
  Soft margin: Allow misclassifications with penalty C
  
  Objective: min 0.5*||w||^2 + C*sum(xi_i)
  Subject to: y_i(w·x_i + b) >= 1 - xi_i
  
  C: Regularization parameter
    Small C: Wider margin, more errors (more regularization)
    Large C: Narrow margin, fewer errors (less regularization)

Kernel Trick:
  Map data to higher dimensions without explicit computation
  K(x_i, x_j) = phi(x_i) · phi(x_j)
  
  Linear: K(x,y) = x·y
  Polynomial: K(x,y) = (gamma*x·y + r)^d
  RBF (Gaussian): K(x,y) = exp(-gamma*||x-y||^2)
  Sigmoid: K(x,y) = tanh(gamma*x·y + r)
  
  RBF gamma:
    Small gamma: Smooth decision boundary (underfitting)
    Large gamma: Complex boundary following data (overfitting)

SVM for regression (SVR):
  Epsilon-insensitive loss
  Points within epsilon tube contribute no loss
  Support vectors are points at or outside tube boundary

Pros: Effective in high dimensions, memory efficient (only support vectors)
Cons: Slow for large datasets O(n^2-n^3), sensitive to feature scaling, kernel choice

**Clustering Algorithms:**

K-Means:
  1. Initialize K centroids (randomly or K-means++)
  2. Assign each point to nearest centroid
  3. Update centroids as mean of assigned points
  4. Repeat until convergence
  
  K-means++ initialization:
    Choose first centroid randomly
    Choose subsequent centroids proportional to D(x)^2
    Leads to better convergence
  
  Choosing K:
    Elbow method: Plot inertia vs K
    Silhouette score: Measure cohesion vs separation
    Gap statistic: Compare with null reference distribution
  
  Limitations:
    Assumes spherical, equal-size clusters
    Sensitive to initialization
    Must specify K
    Only convex clusters

DBSCAN:
  Density-based, finds arbitrary-shaped clusters
  Parameters:
    eps: Maximum distance between neighbors
    min_samples: Minimum points to form dense region
    
  Point types:
    Core: >= min_samples within eps
    Border: Within eps of core point
    Noise: Neither core nor border
    
  Doesn't require specifying K
  Handles noise (outliers)

Hierarchical Clustering:
  Agglomerative (bottom-up):
    Start: Each point is a cluster
    Merge closest clusters until one remains
    
  Linkage methods:
    Single: Min distance between clusters (chaining problem)
    Complete: Max distance (compact clusters)
    Average: Mean distance
    Ward: Minimize total variance increase
    
  Dendrogram: Tree visualization of merges
  Cut at desired level for K clusters

Gaussian Mixture Model (GMM):
  Probabilistic clustering
  Each cluster is a Gaussian distribution
  Soft assignment (probability of belonging to each cluster)
  EM algorithm: E-step (assign) → M-step (update parameters)
  
  Parameters: Mean, covariance, weight per component
  BIC/AIC for model selection (number of components)

**Dimensionality Reduction:**

PCA (Principal Component Analysis):
  Find directions of maximum variance
  Project data onto top K principal components
  
  Algorithm:
    1. Center data (subtract mean)
    2. Compute covariance matrix
    3. Eigendecomposition (or SVD)
    4. Select top K eigenvectors
    5. Project: X_reduced = X · V_k
    
  Explained variance ratio: How much variance each component captures
  Choose K: Cumulative explained variance > 95%
  
  Pros: Fast, well-understood, optimal for linear reduction
  Cons: Linear only, assumes directions of max variance are meaningful

t-SNE:
  Non-linear, for visualization (2D/3D)
  Preserves local neighborhood structure
  Perplexity: Effective number of neighbors (5-50)
  
  Not suitable for:
    New data projection (no transform for unseen data)
    Preserving global structure
    Clustering (distances in output are not meaningful)

UMAP:
  Non-linear, faster than t-SNE
  Better preserves global structure
  Parameters:
    n_neighbors: Local vs global structure
    min_dist: How tightly points are packed
  Can project new data (unlike t-SNE)

**Model Selection and Evaluation:**

Cross-Validation:
  K-fold: Split data into K folds, train on K-1, test on 1
  Stratified K-fold: Preserve class distribution
  Leave-one-out: K = N (expensive, low bias)
  Time series: Forward-chaining (no future data leakage)

Metrics (Classification):
  Accuracy: (TP + TN) / Total
  Precision: TP / (TP + FP) — of predicted positive, how many correct
  Recall (Sensitivity): TP / (TP + FN) — of actual positive, how many found
  F1 Score: 2 * Precision * Recall / (Precision + Recall)
  AUC-ROC: Area under ROC curve (TPR vs FPR)
  Log Loss: Negative log-likelihood of predictions
  
  Confusion Matrix: 2×2 table of TP, FP, FN, TN

Metrics (Regression):
  MSE: Mean Squared Error
  RMSE: Root Mean Squared Error (same units as target)
  MAE: Mean Absolute Error (robust to outliers)
  R² (coefficient of determination): 1 - SS_res/SS_tot
  MAPE: Mean Absolute Percentage Error

Hyperparameter Tuning:
  Grid search: Exhaustive, all combinations
  Random search: Sample random configurations (often better than grid)
  Bayesian optimization: Surrogate model guides search (Optuna, Hyperopt)
  Early stopping: Stop unpromising configurations early (Hyperband)

Bias-Variance Tradeoff:
  Error = Bias² + Variance + Irreducible noise
  
  High bias (underfitting): Model too simple
    Signs: High training error, high test error
    Fix: More features, more complex model, less regularization
    
  High variance (overfitting): Model too complex
    Signs: Low training error, high test error
    Fix: More data, simpler model, more regularization, ensemble`,
					CodeExamples: `# Classical ML Algorithms Implementation

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict

# ============================================================
# Decision Tree
# ============================================================

@dataclass
class TreeNode:
    feature_index: int = -1
    threshold: float = 0.0
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    value: Any = None  # Leaf value
    num_samples: int = 0
    impurity: float = 0.0


class DecisionTree:
    """CART decision tree for classification."""
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, criterion: str = "gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root: Optional[TreeNode] = None
    
    def fit(self, X: List[List[float]], y: List[int]):
        self.n_classes = len(set(y))
        self.n_features = len(X[0])
        self.root = self._build_tree(X, y, depth=0)
    
    def predict(self, X: List[List[float]]) -> List[int]:
        return [self._predict_one(x, self.root) for x in X]
    
    def _gini(self, y: List[int]) -> float:
        counter = Counter(y)
        n = len(y)
        return 1.0 - sum((count / n) ** 2 for count in counter.values())
    
    def _entropy(self, y: List[int]) -> float:
        counter = Counter(y)
        n = len(y)
        return -sum((count / n) * math.log2(count / n)
                    for count in counter.values() if count > 0)
    
    def _impurity(self, y: List[int]) -> float:
        if self.criterion == "gini":
            return self._gini(y)
        return self._entropy(y)
    
    def _best_split(self, X: List[List[float]], y: List[int]) -> Tuple[
            int, float, float]:
        best_gain = -1
        best_feature = -1
        best_threshold = 0.0
        
        parent_impurity = self._impurity(y)
        n = len(y)
        
        for feature in range(self.n_features):
            values = sorted(set(x[feature] for x in X))
            thresholds = [(values[i] + values[i + 1]) / 2
                         for i in range(len(values) - 1)]
            
            for threshold in thresholds:
                left_y = [y[i] for i in range(n) if X[i][feature] <= threshold]
                right_y = [y[i] for i in range(n) if X[i][feature] > threshold]
                
                if (len(left_y) < self.min_samples_leaf or
                        len(right_y) < self.min_samples_leaf):
                    continue
                
                left_impurity = self._impurity(left_y)
                right_impurity = self._impurity(right_y)
                
                gain = parent_impurity - (
                    len(left_y) / n * left_impurity +
                    len(right_y) / n * right_impurity)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: List[List[float]], y: List[int],
                    depth: int) -> TreeNode:
        n = len(y)
        
        # Check stopping criteria
        if (depth >= self.max_depth or n < self.min_samples_split or
                len(set(y)) == 1):
            return TreeNode(
                value=Counter(y).most_common(1)[0][0],
                num_samples=n,
                impurity=self._impurity(y) if len(set(y)) > 1 else 0)
        
        feature, threshold, gain = self._best_split(X, y)
        
        if gain <= 0:
            return TreeNode(
                value=Counter(y).most_common(1)[0][0],
                num_samples=n)
        
        left_idx = [i for i in range(n) if X[i][feature] <= threshold]
        right_idx = [i for i in range(n) if X[i][feature] > threshold]
        
        left_X = [X[i] for i in left_idx]
        left_y = [y[i] for i in left_idx]
        right_X = [X[i] for i in right_idx]
        right_y = [y[i] for i in right_idx]
        
        return TreeNode(
            feature_index=feature,
            threshold=threshold,
            left=self._build_tree(left_X, left_y, depth + 1),
            right=self._build_tree(right_X, right_y, depth + 1),
            num_samples=n,
            impurity=self._impurity(y))
    
    def _predict_one(self, x: List[float], node: TreeNode) -> int:
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)


# ============================================================
# Random Forest
# ============================================================

class RandomForest:
    """Random forest classifier."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 max_features: str = "sqrt",
                 min_samples_split: int = 2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.trees: List[Tuple[DecisionTree, List[int]]] = []
    
    def _bootstrap_sample(self, X: List[List[float]],
                          y: List[int]) -> Tuple[List[List[float]], List[int], Set[int]]:
        n = len(X)
        indices = [random.randint(0, n - 1) for _ in range(n)]
        oob = set(range(n)) - set(indices)
        return [X[i] for i in indices], [y[i] for i in indices], oob
    
    def _get_max_features(self, n_features: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(math.log2(n_features)))
        return n_features
    
    def fit(self, X: List[List[float]], y: List[int]):
        n_features = len(X[0])
        max_feat = self._get_max_features(n_features)
        
        for _ in range(self.n_estimators):
            X_boot, y_boot, _ = self._bootstrap_sample(X, y)
            
            # Select random features
            feature_indices = sorted(random.sample(
                range(n_features), min(max_feat, n_features)))
            
            # Filter features
            X_sub = [[x[f] for f in feature_indices] for x in X_boot]
            
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split)
            tree.fit(X_sub, y_boot)
            
            self.trees.append((tree, feature_indices))
    
    def predict(self, X: List[List[float]]) -> List[int]:
        predictions = []
        for x in X:
            votes = []
            for tree, feature_indices in self.trees:
                x_sub = [x[f] for f in feature_indices]
                votes.append(tree._predict_one(x_sub, tree.root))
            predictions.append(Counter(votes).most_common(1)[0][0])
        return predictions
    
    def feature_importances(self) -> Dict[int, float]:
        importances = defaultdict(float)
        for tree, feature_indices in self.trees:
            self._accumulate_importance(tree.root, feature_indices, importances)
        
        total = sum(importances.values())
        if total > 0:
            return {k: v / total for k, v in sorted(importances.items())}
        return dict(importances)
    
    def _accumulate_importance(self, node: TreeNode,
                               feature_indices: List[int],
                               importances: Dict[int, float]):
        if node is None or node.value is not None:
            return
        
        original_feature = feature_indices[node.feature_index]
        importances[original_feature] += (
            node.num_samples * node.impurity)
        
        self._accumulate_importance(node.left, feature_indices, importances)
        self._accumulate_importance(node.right, feature_indices, importances)


# ============================================================
# Gradient Boosted Trees (Simplified)
# ============================================================

class GradientBoostedRegressor:
    """Gradient boosted regression trees."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, subsample: float = 1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.trees: List[Any] = []
        self.initial_prediction = 0.0
    
    def fit(self, X: List[List[float]], y: List[float]):
        self.initial_prediction = sum(y) / len(y)
        predictions = [self.initial_prediction] * len(y)
        
        for _ in range(self.n_estimators):
            # Compute residuals
            residuals = [y[i] - predictions[i] for i in range(len(y))]
            
            # Subsample
            n = len(X)
            sample_size = max(1, int(n * self.subsample))
            indices = random.sample(range(n), sample_size)
            X_sub = [X[i] for i in indices]
            r_sub = [residuals[i] for i in indices]
            
            # Fit tree to residuals
            tree = RegressionTree(max_depth=self.max_depth)
            tree.fit(X_sub, r_sub)
            self.trees.append(tree)
            
            # Update predictions
            for i in range(len(X)):
                predictions[i] += self.learning_rate * tree.predict_one(X[i])
    
    def predict(self, X: List[List[float]]) -> List[float]:
        predictions = [self.initial_prediction] * len(X)
        for tree in self.trees:
            for i, x in enumerate(X):
                predictions[i] += self.learning_rate * tree.predict_one(x)
        return predictions


class RegressionTree:
    """Simple regression tree."""
    
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.root: Optional[TreeNode] = None
    
    def fit(self, X: List[List[float]], y: List[float]):
        self.root = self._build(X, y, 0)
    
    def predict_one(self, x: List[float]) -> float:
        node = self.root
        while node.value is None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
    
    def _build(self, X: List[List[float]], y: List[float],
               depth: int) -> TreeNode:
        if depth >= self.max_depth or len(y) <= 2:
            return TreeNode(value=sum(y) / max(len(y), 1), num_samples=len(y))
        
        best_feature, best_threshold, best_gain = -1, 0.0, 0.0
        parent_var = self._variance(y)
        n = len(y)
        
        for f in range(len(X[0])):
            values = sorted(set(x[f] for x in X))
            for i in range(len(values) - 1):
                t = (values[i] + values[i + 1]) / 2
                left_y = [y[j] for j in range(n) if X[j][f] <= t]
                right_y = [y[j] for j in range(n) if X[j][f] > t]
                
                if not left_y or not right_y:
                    continue
                
                gain = parent_var - (
                    len(left_y) / n * self._variance(left_y) +
                    len(right_y) / n * self._variance(right_y))
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = t
        
        if best_gain <= 0:
            return TreeNode(value=sum(y) / len(y), num_samples=len(y))
        
        left_idx = [i for i in range(n) if X[i][best_feature] <= best_threshold]
        right_idx = [i for i in range(n) if X[i][best_feature] > best_threshold]
        
        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=self._build([X[i] for i in left_idx], [y[i] for i in left_idx], depth + 1),
            right=self._build([X[i] for i in right_idx], [y[i] for i in right_idx], depth + 1),
            num_samples=n)
    
    def _variance(self, y: List[float]) -> float:
        if not y:
            return 0.0
        mean = sum(y) / len(y)
        return sum((yi - mean) ** 2 for yi in y) / len(y)


# ============================================================
# K-Means Clustering
# ============================================================

class KMeans:
    """K-Means clustering with K-means++ initialization."""
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 300,
                 tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids: List[List[float]] = []
        self.labels: List[int] = []
        self.inertia: float = 0.0
    
    def _distance(self, a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
    
    def _init_centroids_pp(self, X: List[List[float]]):
        """K-means++ initialization."""
        centroids = [X[random.randint(0, len(X) - 1)]]
        
        for _ in range(1, self.n_clusters):
            distances = [min(self._distance(x, c) ** 2 for c in centroids)
                        for x in X]
            total = sum(distances)
            probs = [d / total for d in distances]
            
            r = random.random()
            cumsum = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    centroids.append(X[i])
                    break
        
        self.centroids = centroids
    
    def fit(self, X: List[List[float]]):
        self._init_centroids_pp(X)
        n = len(X)
        d = len(X[0])
        
        for iteration in range(self.max_iter):
            # Assign clusters
            self.labels = [
                min(range(self.n_clusters),
                    key=lambda k: self._distance(x, self.centroids[k]))
                for x in X
            ]
            
            # Update centroids
            new_centroids = []
            for k in range(self.n_clusters):
                cluster = [X[i] for i in range(n) if self.labels[i] == k]
                if cluster:
                    new_centroids.append([
                        sum(x[j] for x in cluster) / len(cluster)
                        for j in range(d)])
                else:
                    new_centroids.append(self.centroids[k])
            
            # Check convergence
            shift = max(self._distance(old, new)
                       for old, new in zip(self.centroids, new_centroids))
            self.centroids = new_centroids
            
            if shift < self.tol:
                break
        
        self.inertia = sum(
            self._distance(X[i], self.centroids[self.labels[i]]) ** 2
            for i in range(n))
    
    def predict(self, X: List[List[float]]) -> List[int]:
        return [
            min(range(self.n_clusters),
                key=lambda k: self._distance(x, self.centroids[k]))
            for x in X
        ]
    
    def silhouette_score(self, X: List[List[float]]) -> float:
        n = len(X)
        scores = []
        
        for i in range(n):
            # a(i): mean distance to same cluster
            same_cluster = [j for j in range(n)
                          if self.labels[j] == self.labels[i] and j != i]
            if not same_cluster:
                scores.append(0)
                continue
            a = sum(self._distance(X[i], X[j]) for j in same_cluster) / len(same_cluster)
            
            # b(i): min mean distance to other clusters
            b = float('inf')
            for k in range(self.n_clusters):
                if k == self.labels[i]:
                    continue
                other = [j for j in range(n) if self.labels[j] == k]
                if other:
                    mean_dist = sum(self._distance(X[i], X[j]) for j in other) / len(other)
                    b = min(b, mean_dist)
            
            scores.append((b - a) / max(a, b))
        
        return sum(scores) / len(scores)


# ============================================================
# PCA (Principal Component Analysis)
# ============================================================

class PCA:
    """Principal Component Analysis."""
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components: List[List[float]] = []
        self.mean: List[float] = []
        self.explained_variance: List[float] = []
    
    def fit(self, X: List[List[float]]):
        n = len(X)
        d = len(X[0])
        
        # Center data
        self.mean = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
        X_centered = [[X[i][j] - self.mean[j] for j in range(d)]
                      for i in range(n)]
        
        # Compute covariance matrix
        cov = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(i, d):
                val = sum(X_centered[k][i] * X_centered[k][j]
                         for k in range(n)) / (n - 1)
                cov[i][j] = val
                cov[j][i] = val
        
        # Power iteration for top eigenvectors
        self.components = []
        self.explained_variance = []
        
        for _ in range(min(self.n_components, d)):
            eigenvalue, eigenvector = self._power_iteration(cov, 100)
            self.components.append(eigenvector)
            self.explained_variance.append(eigenvalue)
            
            # Deflate
            for i in range(d):
                for j in range(d):
                    cov[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j]
    
    def _power_iteration(self, matrix: List[List[float]],
                         max_iter: int) -> Tuple[float, List[float]]:
        d = len(matrix)
        v = [random.gauss(0, 1) for _ in range(d)]
        norm = math.sqrt(sum(x * x for x in v))
        v = [x / norm for x in v]
        
        for _ in range(max_iter):
            # Matrix-vector multiplication
            new_v = [sum(matrix[i][j] * v[j] for j in range(d))
                     for i in range(d)]
            
            # Eigenvalue estimate
            eigenvalue = sum(new_v[i] * v[i] for i in range(d))
            
            # Normalize
            norm = math.sqrt(sum(x * x for x in new_v))
            if norm == 0:
                break
            v = [x / norm for x in new_v]
        
        return eigenvalue, v
    
    def transform(self, X: List[List[float]]) -> List[List[float]]:
        return [[sum((X[i][j] - self.mean[j]) * self.components[k][j]
                     for j in range(len(self.mean)))
                 for k in range(self.n_components)]
                for i in range(len(X))]
    
    @property
    def explained_variance_ratio(self) -> List[float]:
        total = sum(self.explained_variance)
        if total == 0:
            return [0.0] * len(self.explained_variance)
        return [v / total for v in self.explained_variance]`,
				},
			},
		},
	})
}
