package machinelearning

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMachineLearningModules([]problems.CourseModule{
		{
			ID:          90,
			Title:       "Ensemble Methods",
			Description: "Learn how combining multiple models improves performance: Random Forests, Gradient Boosting, and XGBoost.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to Ensemble Methods",
					Content: `Ensemble methods combine multiple models to achieve better performance than any single model.

**Core Idea:**
- Multiple weak learners → One strong learner
- Diversity in models → Better generalization
- Reduces overfitting and variance

**Why Ensembles Work:**
- **Bias Reduction**: Different models capture different patterns
- **Variance Reduction**: Averaging reduces impact of individual errors
- **Robustness**: Less sensitive to noise and outliers

**Types of Ensemble Methods:**

**1. Bagging (Bootstrap Aggregating)**
- Train multiple models on different subsets of data
- Average predictions (regression) or vote (classification)
- Examples: Random Forest

**2. Boosting**
- Train models sequentially, each correcting previous errors
- Weighted combination of models
- Examples: AdaBoost, Gradient Boosting, XGBoost

**3. Stacking**
- Train multiple different models
- Use meta-learner to combine predictions
- More complex but often best performance

**Key Principles:**
- **Diversity**: Models should make different errors
- **Accuracy**: Each model should be reasonably good
- **Independence**: Models should be trained independently (bagging) or sequentially (boosting)`,
					CodeExamples: `from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Single Decision Tree
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
dt_score = accuracy_score(y_test, dt.predict(X_test))
print(f"Single Decision Tree: {dt_score:.3f}")

# Random Forest (Bagging)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
rf_score = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest: {rf_score:.3f}")

# Voting Classifier (Stacking)
voting = VotingClassifier(estimators=[
    ('dt', DecisionTreeClassifier(max_depth=5)),
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier(n_estimators=50))
], voting='hard')
voting.fit(X_train, y_train)
voting_score = accuracy_score(y_test, voting.predict(X_test))
print(f"Voting Classifier: {voting_score:.3f}")`,
				},
				{
					Title: "Random Forests",
					Content: `Random Forest is a bagging ensemble method using decision trees with additional randomness.

**How Random Forest Works:**
1. **Bootstrap Sampling**: Create multiple datasets by sampling with replacement
2. **Feature Randomness**: At each split, consider only random subset of features
3. **Train Trees**: Each tree trained on different data and features
4. **Aggregate**: Average (regression) or vote (classification) predictions

**Key Features:**
- **Bootstrap Aggregation**: Each tree sees different subset of data
- **Feature Subsampling**: Reduces correlation between trees
- **No Pruning**: Trees grown to full depth (or max_depth)
- **Parallel Training**: Trees trained independently

**Hyperparameters:**
- **n_estimators**: Number of trees (more = better, but slower)
- **max_depth**: Maximum tree depth
- **max_features**: Number of features to consider per split
- **min_samples_split**: Minimum samples to split node
- **min_samples_leaf**: Minimum samples in leaf

**Advantages:**
- Handles overfitting well
- Feature importance available
- Works with missing values
- No feature scaling needed
- Handles non-linear relationships

**Limitations:**
- Less interpretable than single tree
- Can be memory intensive
- Slower prediction than single model`,
					CodeExamples: `from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import numpy as np

# Classification
X_clf, y_clf = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                max_features='sqrt', random_state=42)
rf_clf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_clf.predict(X_test)):.3f}")

# Feature Importance
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]
print(f"\nTop 5 Important Features:")
for i in range(5):
    print(f"  Feature {indices[i]}: {importances[indices[i]]:.3f}")

# Regression
X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
print(f"\nRandom Forest Regression R²: {rf_reg.score(X_test_reg, y_test_reg):.3f}")

# Tuning n_estimators
n_trees = [10, 50, 100, 200, 500]
for n in n_trees:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    score = accuracy_score(y_test, rf.predict(X_test))
    print(f"n_estimators={n}: Accuracy = {score:.3f}")`,
				},
				{
					Title: "Gradient Boosting",
					Content: `Gradient Boosting builds models sequentially, with each new model correcting errors of previous models.

**How Gradient Boosting Works:**
1. **Start**: Train initial model (usually simple, e.g., mean)
2. **Calculate Residuals**: Errors of current ensemble
3. **Fit Next Model**: Train model to predict residuals
4. **Add to Ensemble**: Weighted combination with previous models
5. **Repeat**: Continue until stopping criterion

**Key Concepts:**
- **Loss Function**: Measures prediction error (MSE, log loss, etc.)
- **Gradient**: Direction to reduce loss
- **Learning Rate**: Controls contribution of each model
- **Shrinkage**: Prevents overfitting

**Gradient Boosting Algorithm:**
1. Initialize: F₀(x) = argmin Σ L(yᵢ, γ)
2. For m = 1 to M:
   - Compute residuals: rᵢ = -∂L/∂F for each sample
   - Fit hₘ(x) to residuals
   - Update: Fₘ(x) = Fₘ₋₁(x) + α·hₘ(x)

**Advantages:**
- Often best performance
- Handles non-linear relationships
- Feature importance available
- Works with various loss functions

**Limitations:**
- Sequential training (slow)
- Prone to overfitting
- Requires careful tuning
- Less interpretable`,
					CodeExamples: `from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import numpy as np

# Classification
X_clf, y_clf = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                    max_depth=3, random_state=42)
gb_clf.fit(X_train, y_train)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_clf.predict(X_test)):.3f}")

# Staged predictions (see improvement over iterations)
test_scores = []
for y_pred in gb_clf.staged_predict(X_test):
    test_scores.append(accuracy_score(y_test, y_pred))

print(f"\nAccuracy by iteration:")
for i in [0, 25, 50, 75, 99]:
    print(f"  Iteration {i+1}: {test_scores[i]:.3f}")

# Regression
X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                   max_depth=3, random_state=42)
gb_reg.fit(X_train_reg, y_train_reg)
print(f"\nGradient Boosting Regression R²: {gb_reg.score(X_test_reg, y_test_reg):.3f}")

# Tuning learning rate
learning_rates = [0.01, 0.1, 0.3, 0.5]
for lr in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, random_state=42)
    gb.fit(X_train, y_train)
    score = accuracy_score(y_test, gb.predict(X_test))
    print(f"Learning Rate {lr}: Accuracy = {score:.3f}")`,
				},
				{
					Title: "XGBoost",
					Content: `XGBoost (Extreme Gradient Boosting) is an optimized implementation of gradient boosting with additional features.

**XGBoost Improvements:**
- **Regularization**: L1 and L2 regularization terms
- **Parallel Processing**: Approximate parallel tree construction
- **Tree Pruning**: More efficient tree building
- **Handling Missing Values**: Built-in missing value handling
- **Cross-Validation**: Built-in CV during training

**Key Features:**
- **Regularized Objective**: Prevents overfitting
- **Approximate Algorithm**: Faster tree construction
- **Sparsity Awareness**: Handles sparse data efficiently
- **Cache Awareness**: Optimized memory access
- **Out-of-Core Computing**: Handles large datasets

**Hyperparameters:**
- **n_estimators**: Number of boosting rounds
- **learning_rate**: Step size shrinkage
- **max_depth**: Maximum tree depth
- **min_child_weight**: Minimum sum of instance weight
- **subsample**: Fraction of samples for training
- **colsample_bytree**: Fraction of features per tree
- **reg_alpha**: L1 regularization
- **reg_lambda**: L2 regularization

**When to Use:**
- Structured/tabular data
- Need best possible performance
- Have computational resources
- Want feature importance

**Advantages:**
- State-of-the-art performance
- Fast training and prediction
- Handles missing values
- Feature importance
- Regularization built-in

**Limitations:**
- Requires parameter tuning
- Less interpretable
- Can overfit with small data`,
					CodeExamples: `try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Install with: pip install xgboost")

if XGBOOST_AVAILABLE:
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score
    
    # Classification
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)
    
    xgb_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, 
                                max_depth=3, random_state=42)
    xgb_clf.fit(X_train, y_train)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_clf.predict(X_test)):.3f}")
    
    # Feature Importance
    importances = xgb_clf.feature_importances_
    print(f"\nTop 5 Important Features:")
    indices = np.argsort(importances)[::-1][:5]
    for idx in indices:
        print(f"  Feature {idx}: {importances[idx]:.3f}")
    
    # Regression
    X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )
    
    xgb_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, 
                                max_depth=3, random_state=42)
    xgb_reg.fit(X_train_reg, y_train_reg)
    print(f"\nXGBoost Regression R²: {xgb_reg.score(X_test_reg, y_test_reg):.3f}")
    
    # Early Stopping
    xgb_early = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.1, 
                                  early_stopping_rounds=10, random_state=42)
    xgb_early.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"\nEarly Stopping - Best Iteration: {xgb_early.best_iteration}")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          91,
			Title:       "Unsupervised Learning",
			Description: "Discover patterns in unlabeled data: clustering, dimensionality reduction, and anomaly detection.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "Clustering Fundamentals",
					Content: `Clustering groups similar data points together without labeled examples.

**What is Clustering?**
- Partition data into groups (clusters)
- Points in same cluster are similar
- Points in different clusters are dissimilar
- No labels needed (unsupervised)

**Applications:**
- Customer segmentation
- Image segmentation
- Anomaly detection
- Data compression
- Pattern discovery

**Clustering Algorithms:**

**1. K-Means**
- Partition into K clusters
- Minimizes within-cluster variance
- Fast and scalable
- Assumes spherical clusters

**2. Hierarchical Clustering**
- Builds tree of clusters
- Agglomerative (bottom-up) or Divisive (top-down)
- No need to specify K
- Creates dendrogram

**3. DBSCAN**
- Density-based clustering
- Finds arbitrary-shaped clusters
- Handles outliers (noise points)
- No need to specify K

**Evaluation:**
- **Silhouette Score**: Measures cluster quality
- **Inertia**: Within-cluster sum of squares (K-Means)
- **Elbow Method**: Find optimal K`,
					CodeExamples: `from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import numpy as np

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)
labels_kmeans = kmeans.labels_
print(f"K-Means Silhouette Score: {silhouette_score(X, labels_kmeans):.3f}")

# Finding optimal K (Elbow Method)
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

print(f"\nInertia by K:")
for k, inertia in zip(K_range, inertias):
    print(f"  K={k}: {inertia:.2f}")

# Hierarchical Clustering
agg = AgglomerativeClustering(n_clusters=4)
labels_agg = agg.fit_predict(X)
print(f"\nHierarchical Clustering Silhouette Score: {silhouette_score(X, labels_agg):.3f}")

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)
n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)
print(f"\nDBSCAN:")
print(f"  Number of clusters: {n_clusters}")
print(f"  Number of noise points: {n_noise}")`,
				},
				{
					Title: "Dimensionality Reduction",
					Content: `Dimensionality reduction reduces number of features while preserving important information.

**Why Reduce Dimensions?**
- **Curse of Dimensionality**: Performance degrades with many features
- **Visualization**: Reduce to 2D/3D for plotting
- **Noise Reduction**: Remove irrelevant features
- **Storage**: Reduce memory requirements
- **Speed**: Faster training and prediction

**Principal Component Analysis (PCA):**
- Finds directions of maximum variance
- Projects data onto principal components
- Linear transformation
- Preserves variance

**How PCA Works:**
1. Standardize data
2. Compute covariance matrix
3. Find eigenvectors (principal components)
4. Project data onto top K components

**Other Methods:**
- **t-SNE**: Non-linear, good for visualization
- **UMAP**: Non-linear, preserves local structure
- **ICA**: Independent Component Analysis
- **Factor Analysis**: Similar to PCA

**When to Use:**
- High-dimensional data
- Need visualization
- Remove redundancy
- Speed up algorithms`,
					CodeExamples: `from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import numpy as np

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {X_pca.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")

# Finding optimal number of components
pca_full = PCA()
pca_full.fit(X)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

print(f"\nCumulative Variance Explained:")
for i, var in enumerate(cumulative_variance):
    print(f"  {i+1} components: {var:.3f}")

# Components needed for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nComponents for 95% variance: {n_components_95}")

# PCA for visualization
import matplotlib.pyplot as plt
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Visualization')
plt.show()`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          92,
			Title:       "Neural Networks Fundamentals",
			Description: "Learn perceptrons, multi-layer perceptrons, activation functions, backpropagation, and gradient descent.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Perceptrons and Multi-Layer Perceptrons",
					Content: `Neural networks are inspired by biological neurons and form the foundation of deep learning.

**Perceptron:**
- Simplest neural network unit
- Takes weighted sum of inputs
- Applies activation function
- Output: f(Σwᵢxᵢ + b)

**Multi-Layer Perceptron (MLP):**
- Multiple layers of perceptrons
- Input layer → Hidden layers → Output layer
- Can learn non-linear patterns
- Universal function approximator

**Forward Propagation:**
1. Input → Hidden Layer 1
2. Hidden Layer 1 → Hidden Layer 2
3. ... → Output Layer
4. Each layer: z = W·x + b, a = activation(z)

**Key Components:**
- **Weights**: Learned parameters
- **Biases**: Offset terms
- **Activation Functions**: Non-linearity
- **Layers**: Organization of neurons`,
					CodeExamples: `import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Multi-Layer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu',
                   solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
print(f"MLP Accuracy: {accuracy_score(y_test, mlp.predict(X_test)):.3f}")

# Manual Perceptron (conceptual)
def perceptron(x, weights, bias, activation='sigmoid'):
    z = np.dot(weights, x) + bias
    if activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    elif activation == 'relu':
        return max(0, z)
    return z`,
				},
				{
					Title: "Activation Functions",
					Content: `Activation functions introduce non-linearity, enabling neural networks to learn complex patterns.

**Common Activation Functions:**

**Sigmoid**: σ(x) = 1/(1+e^(-x))
- Range: (0, 1)
- Smooth gradient
- Problem: Vanishing gradient

**Tanh**: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
- Range: (-1, 1)
- Zero-centered
- Better than sigmoid

**ReLU**: f(x) = max(0, x)
- Most common
- Fast computation
- Problem: Dying ReLU

**Leaky ReLU**: f(x) = max(αx, x)
- Fixes dying ReLU
- Small gradient for negative values

**Softmax**: f(xᵢ) = e^(xᵢ) / Σe^(xⱼ)
- For output layer (multi-class)
- Outputs probabilities`,
					CodeExamples: `import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

x = np.linspace(-5, 5, 100)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.legend()
plt.title('Activation Functions')
plt.show()`,
				},
				{
					Title: "Backpropagation and Gradient Descent",
					Content: `Backpropagation computes gradients, and gradient descent updates weights to minimize loss.

**Gradient Descent:**
- Minimize loss function
- Update weights: w = w - α·∇L
- α: Learning rate
- Repeat until convergence

**Backpropagation:**
- Computes gradients efficiently
- Uses chain rule
- Propagates errors backward

**Process:**
1. Forward pass: Compute predictions
2. Compute loss
3. Backward pass: Compute gradients
4. Update weights`,
					CodeExamples: `# Conceptual implementation
def gradient_descent_step(weights, gradients, learning_rate):
    return weights - learning_rate * gradients

# Backpropagation computes gradients using chain rule
# This is handled automatically by frameworks like PyTorch, TensorFlow`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          93,
			Title:       "Deep Learning Introduction",
			Description: "Explore deep neural networks, vanishing gradients, regularization, and optimizers.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "Deep Neural Networks",
					Content: `Deep neural networks have multiple hidden layers, enabling them to learn hierarchical representations.

**Why Deep?**
- Each layer learns different abstraction levels
- Lower layers: Simple features (edges)
- Higher layers: Complex features (objects)
- More layers = More capacity

**Challenges:**
- Vanishing gradients
- Overfitting
- Training instability
- Computational cost`,
					CodeExamples: `import torch
import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)`,
				},
				{
					Title: "Regularization Techniques",
					Content: `Regularization prevents overfitting in deep networks.

**Dropout:**
- Randomly set neurons to zero during training
- Prevents co-adaptation
- Test time: Use all neurons with scaled weights

**Batch Normalization:**
- Normalize activations
- Stabilizes training
- Allows higher learning rates

**L1/L2 Regularization:**
- Penalize large weights
- Add to loss function`,
					CodeExamples: `import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          94,
			Title:       "Convolutional Neural Networks (CNNs)",
			Description: "Learn convolution operations, pooling, CNN architectures, and transfer learning.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Convolution Operations",
					Content: `Convolution extracts local features using filters/kernels that slide across input.

**Key Concepts:**
- **Filter/Kernel**: Small matrix that detects features
- **Feature Map**: Output after convolution
- **Stride**: Step size of filter
- **Padding**: Add zeros around input

**Why Convolution?**
- Parameter sharing (same filter everywhere)
- Translation invariance
- Local connectivity`,
					CodeExamples: `import torch
import torch.nn as nn

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 14 * 14, 10)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = self.fc(x)
        return x`,
				},
				{
					Title: "CNN Architectures",
					Content: `Famous CNN architectures: LeNet, AlexNet, VGG, ResNet, etc.

**Transfer Learning:**
- Use pre-trained models
- Fine-tune on your data
- Saves training time`,
					CodeExamples: `import torchvision.models as models

# Pre-trained ResNet
resnet = models.resnet18(pretrained=True)
# Freeze early layers, fine-tune later layers`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
