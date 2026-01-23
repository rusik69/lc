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

**Step-by-Step Algorithm:**
1. **Bootstrap Sampling**: Create multiple datasets by sampling with replacement
   - Each bootstrap sample has same size as original (n samples)
   - On average, ~63% of original samples appear in each bootstrap
   - ~37% are "out-of-bag" (OOB) - not used for training that tree
2. **Feature Randomness**: At each split, consider only random subset of features
   - Default: √p features for classification, p/3 for regression (p = total features)
   - Reduces correlation between trees
3. **Train Trees**: Each tree trained on different data and features
   - Trees grown to full depth (or max_depth)
   - No pruning (unlike single decision tree)
4. **Aggregate**: Average (regression) or vote (classification) predictions

**Bootstrap Sampling - Detailed:**

**What is Bootstrap Sampling?**
- Sample n samples with replacement from n original samples
- Some samples appear multiple times
- Some samples don't appear at all (OOB samples)

**Example:**
Original: [A, B, C, D, E]
Bootstrap 1: [A, A, C, D, E] (B missing, A appears twice)
Bootstrap 2: [A, B, B, C, E] (D missing, B appears twice)
Bootstrap 3: [B, C, D, D, E] (A missing, D appears twice)

**Why Bootstrap?**
- Creates diversity: Each tree sees different data
- Reduces overfitting: No single tree sees all data
- Enables OOB error estimation: Evaluate on samples not in training set

**Out-of-Bag (OOB) Error:**
- For each sample, predict using trees that didn't see it
- Average OOB predictions across all samples
- Provides unbiased error estimate without separate validation set
- Similar to cross-validation but built-in!

**Feature Randomness - Why It Matters:**

**The Problem:**
If all trees consider all features, they become highly correlated.
- Strong features always chosen first
- Trees become similar
- Ensemble doesn't help much

**The Solution:**
Randomly select subset of features at each split.
- Different trees consider different features
- Reduces correlation between trees
- Increases diversity

**Example:**
- Tree 1: Might consider features [1, 3, 5] at root
- Tree 2: Might consider features [2, 4, 6] at root
- Different trees, different perspectives!

**Why Random Forests Work:**

**1. Bias-Variance Decomposition:**
Error = Bias² + Variance + Irreducible Error

**Single Tree:**
- Low bias (can fit complex patterns)
- High variance (unstable, sensitive to data)

**Random Forest:**
- Low bias (trees still flexible)
- Low variance (averaging reduces variance)
- **Key**: Trees must be diverse (low correlation)

**2. Law of Large Numbers:**
- As number of trees increases, prediction converges
- Variance of ensemble decreases
- More trees = more stable predictions

**3. Diversity:**
- Different data (bootstrap)
- Different features (random subset)
- Different trees (randomness in splits)
- Diversity reduces correlation, improves ensemble

**Voting/Averaging Mechanism:**

**Classification (Majority Vote):**
- Each tree votes for a class
- Final prediction: Most common class
- Can also use probability: Average probabilities from all trees

**Regression (Averaging):**
- Each tree predicts a value
- Final prediction: Average of all tree predictions
- Reduces variance (noise cancels out)

**Why Averaging Works:**
- If errors are uncorrelated, averaging reduces variance
- Variance of average = Variance / n (for independent errors)
- More trees = lower variance

**Feature Importance Calculation:**

**Method 1: Gini Importance (Default):**
- Sum impurity reduction over all nodes using feature
- Normalized across all features
- Higher = feature used more and creates better splits

**Method 2: Permutation Importance:**
- Shuffle feature values
- Measure performance drop
- Larger drop = more important feature
- More reliable but computationally expensive

**Hyperparameters:**

**n_estimators:**
- Number of trees in forest
- More trees = better performance (diminishing returns)
- Typical: 100-500
- **Rule**: Use enough trees for stable predictions

**max_features:**
- Number of features to consider per split
- 'sqrt': √p (default for classification)
- 'log2': log₂(p)
- Integer: Exact number
- Smaller = more diversity, but may miss important features

**max_depth:**
- Maximum depth of trees
- None: Grow to full depth
- Smaller: Simpler trees, faster, less overfitting

**min_samples_split:**
- Minimum samples required to split node
- Larger = simpler trees
- Default: 2

**min_samples_leaf:**
- Minimum samples in leaf node
- Larger = smoother predictions
- Default: 1

**Advantages:**
- Handles overfitting well (ensemble reduces variance)
- Feature importance available (identifies important features)
- Works with missing values (can handle during training)
- No feature scaling needed (trees are scale-invariant)
- Handles non-linear relationships (trees capture non-linearity)
- Parallelizable (trees independent)
- OOB error estimate (built-in validation)
- Robust to outliers (averaging effect)
- Handles high-dimensional data well

**Limitations:**
- Less interpretable than single tree (many trees)
- Can be memory intensive (stores all trees)
- Slower prediction than single model (must query all trees)
- May overfit with noisy data (if trees too deep)
- Biased toward features with more levels (like single trees)
- Doesn't extrapolate well (predictions bounded by training range)`,
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

**Core Idea:**
Instead of training models independently (like bagging), train models sequentially where each new model corrects the mistakes of the previous ensemble.

**Step-by-Step Algorithm:**

**Step 1: Initialize**
F₀(x) = argmin_γ Σ L(yᵢ, γ)
- Start with constant prediction that minimizes loss
- For MSE: F₀(x) = mean(y) (average of target)
- For log loss: F₀(x) = log(odds) of positive class

**Step 2: For m = 1 to M (number of boosting rounds):**

**a) Compute Pseudo-Residuals (Negative Gradients):**
rᵢₘ = -[∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)] evaluated at F = Fₘ₋₁

**Interpretation:**
- Residuals point in direction that reduces loss
- For MSE: rᵢₘ = yᵢ - Fₘ₋₁(xᵢ) (actual prediction errors)
- For other losses: Negative gradient of loss function

**b) Fit Weak Learner to Residuals:**
hₘ(x) = argmin_h Σ [rᵢₘ - h(xᵢ)]²
- Train model (usually decision tree) to predict residuals
- This model learns to correct errors of previous ensemble

**c) Find Optimal Step Size:**
γₘ = argmin_γ Σ L(yᵢ, Fₘ₋₁(xᵢ) + γ·hₘ(xᵢ))
- Find best weight for new model
- Can use line search or fixed learning rate

**d) Update Ensemble:**
Fₘ(x) = Fₘ₋₁(x) + α·γₘ·hₘ(x)
- Add new model to ensemble
- α: Learning rate (shrinkage factor, typically 0.01-0.1)
- Smaller α = slower learning, more models needed, less overfitting

**Step 3: Final Prediction**
F_M(x) = F₀(x) + α·Σₘ₌₁ᴹ γₘ·hₘ(x)
- Sum of initial prediction and all corrections

**Worked Example - Regression:**

**Data:** x = [1, 2, 3, 4, 5], y = [2, 4, 6, 8, 10] (perfect linear relationship)

**Initialization (m=0):**
F₀(x) = mean(y) = 6 (constant prediction)
Predictions: [6, 6, 6, 6, 6]
Residuals: [-4, -2, 0, 2, 4]

**Iteration 1 (m=1):**
- Fit tree to residuals: h₁(x) ≈ 2x - 6 (learns linear pattern)
- Learning rate α = 0.1
- Update: F₁(x) = 6 + 0.1·h₁(x)
- New predictions: [5.6, 5.8, 6.0, 6.2, 6.4]
- New residuals: [-3.6, -1.8, 0, 1.8, 3.6] (smaller!)

**Iteration 2 (m=2):**
- Fit tree to new residuals
- Update: F₂(x) = F₁(x) + 0.1·h₂(x)
- Residuals decrease further
- Continue until residuals are small

**Why Fit to Residuals?**

**Gradient Descent Connection:**
- Gradient descent: θ = θ - α·∇L(θ)
- Gradient boosting: F = F - α·h where h approximates gradient
- Both move in direction that reduces loss!

**Residuals = Negative Gradient:**
- For MSE: Residual = y - ŷ = -∂L/∂ŷ
- Fitting to residuals = following gradient direction
- Each model moves ensemble toward minimum loss

**Learning Rate (Shrinkage):**

**Purpose:**
- Controls contribution of each model
- Prevents overfitting
- Slows down learning

**Effect:**
- **Large α (e.g., 0.5)**: Fast learning, fewer models needed, risk of overfitting
- **Small α (e.g., 0.01)**: Slow learning, more models needed, better generalization

**Typical Values:**
- α = 0.1: Common default
- α = 0.01-0.05: For better generalization
- α = 0.3-0.5: Faster convergence, more risk

**Early Stopping:**

**Purpose:**
Stop training when validation performance stops improving.

**Process:**
1. Monitor validation loss after each iteration
2. If validation loss increases for N consecutive iterations, stop
3. Use model from best iteration

**Why Important:**
- Prevents overfitting
- Saves computation
- Improves generalization

**Loss Functions:**

**Regression:**
- **MSE**: L(y, ŷ) = (y - ŷ)²
  - Residuals = y - ŷ (prediction errors)
- **MAE**: L(y, ŷ) = |y - ŷ|
  - More robust to outliers
- **Huber Loss**: Combines MSE and MAE

**Classification:**
- **Log Loss**: L(y, p) = -[y·log(p) + (1-y)·log(1-p)]
  - Residuals = y - p (probability errors)
- **Exponential Loss**: Used in AdaBoost

**Advantages:**
- Often best performance (state-of-the-art on many problems)
- Handles non-linear relationships (trees capture complexity)
- Feature importance available (from trees)
- Works with various loss functions (flexible)
- Can handle missing values (trees handle it)
- No feature scaling needed

**Limitations:**
- Sequential training (can't parallelize, slow)
- Prone to overfitting (need careful tuning)
- Requires careful tuning (many hyperparameters)
- Less interpretable (many sequential models)
- Sensitive to outliers (especially with MSE loss)
- Memory intensive (stores all models)`,
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

**XGBoost Improvements Over Standard Gradient Boosting:**

**1. Regularized Objective Function:**

**Standard Gradient Boosting:**
Objective = Loss + (no regularization)

**XGBoost:**
Objective = Loss + L1_regularization + L2_regularization + Tree_complexity

**Regularization Terms:**
- **L1 (reg_alpha)**: λ₁·Σ|w| - Lasso regularization, can set weights to zero
- **L2 (reg_lambda)**: λ₂·Σw² - Ridge regularization, shrinks weights
- **Tree complexity**: γ·T where T = number of leaves
  - Penalizes complex trees
  - Encourages simpler models

**Why Regularization Helps:**
- Prevents overfitting
- Improves generalization
- Allows using deeper trees without overfitting

**2. Approximate Algorithm for Tree Construction:**

**Standard Approach:**
- Consider all possible split points
- Computationally expensive: O(n×m) where n=samples, m=features

**XGBoost Approach:**
- Use quantiles (percentiles) instead of all values
- Create histogram of feature values
- Consider only quantile boundaries as split candidates
- **Speedup**: 10-100x faster!

**Example:**
- Instead of checking 1000 values, check 10 quantiles
- Approximate but very close to optimal
- Configurable: tree_method='hist' or 'approx'

**3. Parallel Tree Construction:**

**Standard Gradient Boosting:**
- Sequential: Must build tree 1 before tree 2
- Can't parallelize across trees

**XGBoost:**
- Parallelize within tree building
- Different features can be processed in parallel
- Different quantiles can be computed in parallel
- **Speedup**: Utilizes all CPU cores

**4. Tree Pruning:**

**Standard Approach:**
- Build tree, then prune (post-pruning)

**XGBoost Approach:**
- **Pre-pruning**: Stop splitting if gain < threshold
- **Gain calculation**: Gain = Loss_reduction - Complexity_penalty
- If gain < γ (min_split_loss), don't split
- More efficient: Don't build unnecessary branches

**5. Handling Missing Values:**

**Standard Approach:**
- Need to impute missing values before training

**XGBoost Approach:**
- Automatically learns best direction for missing values
- For each split, tries sending missing values left and right
- Chooses direction that minimizes loss
- **No preprocessing needed!**

**6. Second-Order Gradient Information:**

**Standard Gradient Boosting:**
- Uses only first derivatives (gradients)

**XGBoost:**
- Uses both first and second derivatives (Hessian)
- More accurate step size calculation
- Faster convergence
- Better optimization

**7. Column (Feature) Subsampling:**

**XGBoost adds:**
- colsample_bytree: Fraction of features per tree
- colsample_bylevel: Fraction of features per level
- colsample_bynode: Fraction of features per node
- Increases diversity, reduces overfitting

**8. Row (Sample) Subsampling:**

**XGBoost adds:**
- subsample: Fraction of samples per tree
- Similar to Random Forest bootstrap
- Further reduces overfitting
- Faster training

**XGBoost Objective Function:**

**Mathematical Form:**
Obj = Σ L(yᵢ, ŷᵢ) + Σₜ [λ₁·Σ|wₜ| + λ₂·Σwₜ² + γ·Tₜ]

Where:
- L: Loss function (MSE, log loss, etc.)
- wₜ: Weights in tree t
- Tₜ: Number of leaves in tree t
- λ₁, λ₂: L1 and L2 regularization coefficients
- γ: Complexity penalty

**Tree Building Algorithm:**

**Gain Calculation:**
Gain = (1/2) × [GL²/(HL+λ) + GR²/(HR+λ) - (GL+GR)²/(HL+HR+λ)] - γ

Where:
- GL, GR: Sum of gradients in left/right child
- HL, HR: Sum of Hessians in left/right child
- λ: L2 regularization
- γ: Minimum gain threshold

**Split if**: Gain > 0 (and > γ)

**Key Optimizations:**

**1. Cache Awareness:**
- Pre-sort data for efficient access
- Cache frequently accessed gradients/hessians
- Reduces memory access time

**2. Sparsity Awareness:**
- Efficient handling of sparse features
- Special algorithms for sparse data
- Important for text/one-hot encoded data

**3. Out-of-Core Computing:**
- Can train on data larger than RAM
- Uses disk when memory is full
- Enables training on huge datasets

**4. Block Structure:**
- Organizes data into blocks
- Enables parallel processing
- Efficient memory layout

**Hyperparameters:**

**Core Parameters:**
- **n_estimators**: Number of boosting rounds (trees)
- **learning_rate**: Step size (typically 0.01-0.3)
- **max_depth**: Maximum tree depth (typically 3-10)

**Regularization:**
- **reg_alpha**: L1 regularization (default: 0)
- **reg_lambda**: L2 regularization (default: 1)
- **gamma**: Minimum loss reduction for split (default: 0)

**Sampling:**
- **subsample**: Row sampling (default: 1.0)
- **colsample_bytree**: Feature sampling per tree (default: 1.0)
- **colsample_bylevel**: Feature sampling per level (default: 1.0)

**Tree Construction:**
- **min_child_weight**: Minimum sum of instance weight in child (default: 1)
- **max_delta_step**: Maximum delta step (for imbalanced classes)

**When to Use:**
- Structured/tabular data (not images/text directly)
- Need best possible performance
- Have computational resources
- Want feature importance
- Large datasets (XGBoost handles scale well)

**Advantages:**
- State-of-the-art performance (often best on tabular data)
- Fast training and prediction (optimized implementation)
- Handles missing values (automatic handling)
- Feature importance (multiple types available)
- Regularization built-in (prevents overfitting)
- Highly optimized (parallel, approximate algorithms)
- Scalable (handles large datasets)
- Flexible (many hyperparameters to tune)

**Limitations:**
- Requires parameter tuning (many hyperparameters)
- Less interpretable (complex ensemble)
- Can overfit with small data (need regularization)
- Memory intensive (stores all trees)
- Sequential training (can't fully parallelize across trees)
- Sensitive to hyperparameters (need careful tuning)`,
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

**K-Means Algorithm - Detailed Walkthrough:**

**Objective:**
Minimize within-cluster sum of squares (WCSS):
WCSS = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²
Where:
- Cₖ: Cluster k
- μₖ: Centroid (mean) of cluster k
- xᵢ: Data point i

**Algorithm Steps:**

**1. Initialize Centroids:**
- Random initialization: Randomly select K points as centroids
- K-means++: Smart initialization (spread centroids apart)
  - First centroid: Random point
  - Subsequent: Choose point farthest from existing centroids
  - Better than random (faster convergence, better results)

**2. Assignment Step:**
For each data point xᵢ:
- Calculate distance to all K centroids
- Assign to nearest centroid: cᵢ = argminₖ ||xᵢ - μₖ||²
- Creates K clusters

**3. Update Step:**
For each cluster k:
- Recalculate centroid: μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ
- New centroid is mean of all points in cluster

**4. Repeat:**
- Steps 2-3 until convergence
- Convergence: Centroids don't change (or change < threshold)
- Or: Maximum iterations reached

**Example Walkthrough:**

**Initial Data:** Points: [(1,1), (1,2), (2,1), (8,8), (9,8), (9,9)], K=2

**Iteration 1:**
- Initialize: μ₁ = (1,1), μ₂ = (8,8)
- Assign: Cluster 1: [(1,1), (1,2), (2,1)], Cluster 2: [(8,8), (9,8), (9,9)]
- Update: μ₁ = (1.33, 1.33), μ₂ = (8.67, 8.33)

**Iteration 2:**
- Assign: Same clusters (converged!)
- Final centroids: μ₁ = (1.33, 1.33), μ₂ = (8.67, 8.33)

**Why K-Means Works:**
- **Convergence**: WCSS decreases each iteration (guaranteed)
- **Local Minimum**: May converge to local (not global) optimum
- **Multiple Runs**: Run with different initializations, pick best

**Limitations:**
- Assumes spherical clusters (Euclidean distance)
- Sensitive to initialization (local minima)
- Need to specify K (number of clusters)
- Sensitive to outliers (centroids pulled toward outliers)
- Assumes clusters of similar size

**Elbow Method for Choosing K:**

**Process:**
1. Run K-means for K = 1 to K = max_K
2. Calculate WCSS (inertia) for each K
3. Plot K vs WCSS
4. Find "elbow" - point where decrease slows down

**Interpretation:**
- Before elbow: Large decrease (adding clusters helps)
- After elbow: Small decrease (diminishing returns)
- Elbow point: Optimal K

**Silhouette Score:**
Measures how well points fit their clusters.

**Formula:**
s(i) = (b(i) - a(i)) / max(a(i), b(i))
Where:
- a(i): Average distance to points in same cluster
- b(i): Average distance to points in nearest other cluster

**Range:** [-1, 1]
- **+1**: Perfect clustering
- **0**: Overlapping clusters
- **-1**: Wrong clusters

**Average silhouette**: Mean over all points

**Hierarchical Clustering:**

**Agglomerative (Bottom-Up):**
1. Start: Each point is its own cluster
2. Merge: Find two closest clusters, merge them
3. Repeat: Until K clusters remain (or single cluster)

**Linkage Methods:**
- **Single**: Minimum distance between clusters
- **Complete**: Maximum distance between clusters
- **Average**: Average distance between clusters
- **Ward**: Minimizes within-cluster variance

**Dendrogram:**
- Tree diagram showing cluster merges
- Height = distance when clusters merged
- Cut dendrogram at desired height to get K clusters

**DBSCAN (Density-Based):**

**Key Concepts:**
- **Core Point**: Has ≥ min_samples neighbors within ε radius
- **Border Point**: Within ε of core point but not core itself
- **Noise Point**: Neither core nor border

**Algorithm:**
1. Find all core points
2. Form clusters: Core points within ε of each other
3. Add border points to nearest cluster
4. Mark remaining as noise

**Advantages:**
- Finds arbitrary-shaped clusters
- Handles outliers (noise points)
- No need to specify K
- Density-based (finds dense regions)

**Parameters:**
- **ε (eps)**: Maximum distance for neighbors
- **min_samples**: Minimum points to form core

**Evaluation Metrics:**

**1. Inertia (WCSS):**
- Within-cluster sum of squares
- Lower is better
- Used in elbow method

**2. Silhouette Score:**
- Measures cluster quality
- Higher is better (max = 1)

**3. Adjusted Rand Index (ARI):**
- Compares to ground truth labels
- Range: [-1, 1], higher is better

**4. Normalized Mutual Information (NMI):**
- Information-theoretic measure
- Range: [0, 1], higher is better`,
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

**Principal Component Analysis (PCA) - Detailed Derivation:**

**Goal:**
Find directions (principal components) that capture maximum variance in data.

**Mathematical Formulation:**

**Step 1: Standardize Data**
Center data: X_centered = X - mean(X)
(Optional: Also scale to unit variance)

**Step 2: Covariance Matrix**
C = (1/n) X_centered^T × X_centered
- Measures how features vary together
- Diagonal: Variance of each feature
- Off-diagonal: Covariance between features

**Step 3: Eigendecomposition**
C × v = λ × v
- Find eigenvectors v (principal components)
- Find eigenvalues λ (variance along each component)
- Sort by eigenvalues (descending)

**Step 4: Projection**
X_reduced = X_centered × V_k
- V_k: Top K eigenvectors (columns)
- Projects data onto K-dimensional space

**Why Variance Maximization?**

**Intuition:**
- Directions with high variance contain more information
- Directions with low variance are noise or redundant
- Preserving variance ≈ preserving information

**Mathematical Justification:**
- First PC: Direction of maximum variance
- Second PC: Direction of maximum variance orthogonal to first
- And so on...

**Geometric Interpretation:**

**2D Example:**
- Data points form ellipse
- First PC: Long axis (most variance)
- Second PC: Short axis (orthogonal to first)
- Project onto first PC: 1D representation preserving most information

**Eigenvalues and Explained Variance:**

**Eigenvalue λᵢ:**
- Variance along principal component i
- Larger λᵢ = more variance = more important

**Explained Variance Ratio:**
λᵢ / Σⱼ λⱼ
- Proportion of total variance explained by component i
- Sum of top K ratios: Total variance preserved

**Example:**
If eigenvalues = [5, 2, 1, 0.5]:
- PC1 explains: 5/(5+2+1+0.5) = 59%
- PC2 explains: 2/8.5 = 24%
- PC1+PC2: 83% of variance

**Choosing Number of Components:**

**Method 1: Explained Variance Threshold**
- Keep components until cumulative variance ≥ threshold (e.g., 95%)
- Common: 80-95% variance

**Method 2: Scree Plot**
- Plot eigenvalues vs component number
- Find "elbow" where eigenvalues level off
- Keep components before elbow

**Method 3: Kaiser Criterion**
- Keep components with eigenvalue > 1
- (For standardized data)

**PCA Properties:**

**1. Orthogonality:**
- Principal components are orthogonal (perpendicular)
- No correlation between components

**2. Uncorrelated Features:**
- Transformed features are uncorrelated
- Diagonal covariance matrix

**3. Optimal Reconstruction:**
- Minimizes reconstruction error
- Best linear approximation in least squares sense

**4. Dimensionality Reduction:**
- Reduces from d dimensions to k dimensions (k < d)
- Preserves maximum variance

**Reconstruction:**

**Forward Transform:**
X_reduced = X × V_k

**Inverse Transform (Reconstruction):**
X_reconstructed = X_reduced × V_k^T

**Reconstruction Error:**
||X - X_reconstructed||²
- Measures information loss
- Decreases as k increases

**Limitations of PCA:**

**1. Linear Transformation:**
- Only captures linear relationships
- Can't handle non-linear patterns
- Solution: Kernel PCA (non-linear)

**2. Assumes Variance = Information:**
- May not hold if low-variance features are important
- Standardization helps but doesn't solve completely

**3. Interpretability:**
- Principal components are linear combinations
- Hard to interpret original features
- May not correspond to meaningful concepts

**4. Sensitive to Scaling:**
- Features with larger scales dominate
- Always standardize before PCA!

**Other Dimensionality Reduction Methods:**

**1. t-SNE (t-Distributed Stochastic Neighbor Embedding):**
- Non-linear method
- Excellent for visualization
- Preserves local structure (nearby points stay nearby)
- **Limitation**: Can't use for new data (no transform function)

**2. UMAP (Uniform Manifold Approximation and Projection):**
- Non-linear method
- Preserves both local and global structure
- Faster than t-SNE
- Can transform new data

**3. ICA (Independent Component Analysis):**
- Finds statistically independent components
- Assumes non-Gaussian sources
- Used for signal separation

**4. Factor Analysis:**
- Similar to PCA but with error term
- Assumes underlying factors cause observed features
- More interpretable than PCA

**When to Use PCA:**
- High-dimensional data (many features)
- Need visualization (reduce to 2D/3D)
- Remove redundancy (correlated features)
- Speed up algorithms (fewer dimensions)
- Noise reduction (remove low-variance components)
- Feature extraction (create new features)
- Before other ML algorithms (preprocessing)`,
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

**Forward Propagation - Detailed:**

**Step-by-Step Process:**

**Layer 1 (Input to Hidden):**
z¹ = W¹·x + b¹  (linear transformation)
a¹ = σ(z¹)      (activation)

**Layer 2 (Hidden to Hidden):**
z² = W²·a¹ + b²
a² = σ(z²)

**Output Layer:**
z³ = W³·a² + b³
ŷ = σ(z³)  (or softmax for multi-class)

**Matrix Form:**
- Input: x (n_features × 1)
- Weights: W (n_neurons × n_inputs)
- Bias: b (n_neurons × 1)
- Output: a (n_neurons × 1)

**Example Calculation:**
Input: x = [1, 2]
Weights: W = [[0.5, 0.3], [0.2, 0.8]]
Bias: b = [0.1, -0.2]

z = W·x + b = [[0.5, 0.3], [0.2, 0.8]]·[1, 2] + [0.1, -0.2]
  = [0.5×1 + 0.3×2 + 0.1, 0.2×1 + 0.8×2 - 0.2]
  = [1.2, 1.6]

a = ReLU(z) = [max(0, 1.2), max(0, 1.6)] = [1.2, 1.6]

**Key Components:**
- **Weights**: Learned parameters (determine connection strength)
- **Biases**: Offset terms (shift activation function)
- **Activation Functions**: Non-linearity (enables complex patterns)
- **Layers**: Organization of neurons (hierarchical feature learning)`,
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

**Why Activation Functions?**
Without activation functions, neural network is just linear transformation:
- Multiple layers = single linear layer
- Can't learn non-linear patterns
- Activation adds non-linearity

**Common Activation Functions:**

**1. Sigmoid**: σ(x) = 1/(1+e^(-x))
- **Range**: (0, 1)
- **Derivative**: σ'(x) = σ(x)(1-σ(x))
- **Properties**: Smooth, differentiable everywhere
- **Problem**: Vanishing gradient (derivative → 0 for large |x|)
- **Use**: Output layer for binary classification

**2. Tanh**: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
- **Range**: (-1, 1)
- **Derivative**: tanh'(x) = 1 - tanh²(x)
- **Properties**: Zero-centered (better than sigmoid)
- **Problem**: Still has vanishing gradient
- **Use**: Hidden layers (better than sigmoid)

**3. ReLU**: f(x) = max(0, x)
- **Range**: [0, ∞)
- **Derivative**: f'(x) = 1 if x > 0, else 0
- **Properties**: Fast computation, no saturation for positive values
- **Problem**: Dying ReLU (neurons output 0, gradient = 0, never update)
- **Use**: Most common for hidden layers

**4. Leaky ReLU**: f(x) = max(αx, x) where α ≈ 0.01
- **Range**: (-∞, ∞)
- **Derivative**: f'(x) = 1 if x > 0, else α
- **Properties**: Fixes dying ReLU (small gradient for negative values)
- **Use**: Alternative to ReLU

**5. Softmax**: f(xᵢ) = e^(xᵢ) / Σⱼe^(xⱼ)
- **Range**: (0, 1), sums to 1
- **Properties**: Converts logits to probabilities
- **Use**: Output layer for multi-class classification

**Activation Function Derivatives:**

**Why Derivatives Matter:**
- Needed for backpropagation
- Determine gradient flow
- Affect learning speed

**Sigmoid Derivative:**
σ'(x) = σ(x)(1-σ(x))
- Maximum at x=0: σ'(0) = 0.25
- Approaches 0 as |x| → ∞ (vanishing gradient)

**ReLU Derivative:**
ReLU'(x) = 1 if x > 0, else 0
- Constant gradient for positive values
- Zero gradient for negative values (problem!)

**Weight Initialization:**

**Why It Matters:**
- Poor initialization → vanishing/exploding gradients
- Good initialization → faster convergence

**Xavier/Glorot Initialization:**
- For sigmoid/tanh: W ~ N(0, 1/n_in)
- Variance: Var(W) = 1/n_in
- Keeps activations in reasonable range

**He Initialization:**
- For ReLU: W ~ N(0, 2/n_in)
- Variance: Var(W) = 2/n_in
- Accounts for ReLU killing half of activations`,
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

**Backpropagation - Detailed Explanation:**

**The Chain Rule:**
If y = f(g(x)), then dy/dx = (dy/dg) × (dg/dx)

**Applied to Neural Networks:**
Loss depends on output, output depends on hidden layers, hidden layers depend on weights.
- Need to chain derivatives backward through network

**Step-by-Step Backpropagation:**

**1. Forward Pass:**
Compute activations for all layers:
- z¹ = W¹·x + b¹, a¹ = σ(z¹)
- z² = W²·a¹ + b², a² = σ(z²)
- z³ = W³·a² + b³, ŷ = σ(z³)

**2. Compute Loss:**
L = (1/2)(y - ŷ)²  (for MSE)

**3. Backward Pass (Output Layer):**
∂L/∂z³ = ∂L/∂ŷ × ∂ŷ/∂z³ = (ŷ - y) × σ'(z³)
∂L/∂W³ = ∂L/∂z³ × ∂z³/∂W³ = (∂L/∂z³) × a²
∂L/∂b³ = ∂L/∂z³

**4. Backward Pass (Hidden Layer 2):**
∂L/∂z² = ∂L/∂z³ × ∂z³/∂a² × ∂a²/∂z²
       = (∂L/∂z³) × W³ × σ'(z²)
∂L/∂W² = (∂L/∂z²) × a¹
∂L/∂b² = ∂L/∂z²

**5. Backward Pass (Hidden Layer 1):**
∂L/∂z¹ = ∂L/∂z² × ∂z²/∂a¹ × ∂a¹/∂z¹
       = (∂L/∂z²) × W² × σ'(z¹)
∂L/∂W¹ = (∂L/∂z¹) × x
∂L/∂b¹ = ∂L/∂z¹

**6. Update Weights:**
W = W - α × (∂L/∂W)
b = b - α × (∂L/∂b)

**Gradient Flow:**

**Vanishing Gradient Problem:**
- In deep networks, gradients can become very small
- Each layer multiplies by σ'(z) < 1
- After many layers: gradient ≈ 0
- Early layers don't update (learn slowly)

**Exploding Gradient Problem:**
- Gradients can become very large
- Weights update too much
- Training becomes unstable

**Solutions:**
- Proper initialization (Xavier/He)
- Batch normalization
- Residual connections
- Gradient clipping

**Process Summary:**
1. Forward pass: Compute predictions (store activations)
2. Compute loss: Compare predictions to targets
3. Backward pass: Compute gradients using chain rule
4. Update weights: Move in direction that reduces loss`,
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
		{
			ID:          113,
			Title:       "Natural Language Processing Basics",
			Description: "Learn NLP fundamentals: text preprocessing, word embeddings, text classification, sentiment analysis, and NER.",
			Order:       23,
			Lessons: []problems.Lesson{
				{
					Title: "Text Preprocessing",
					Content: `Text preprocessing prepares raw text for machine learning models.

**Common Steps:**
- **Tokenization**: Split text into tokens (words/subwords)
- **Lowercasing**: Convert to lowercase
- **Removing Punctuation**: Clean special characters
- **Stop Word Removal**: Remove common words (the, a, an)
- **Stemming**: Reduce words to root (running → run)
- **Lemmatization**: More sophisticated than stemming (better → good)

**Tokenization Methods:**
- **Whitespace**: Split on spaces
- **Word Tokenization**: Handle punctuation properly
- **Subword Tokenization**: BPE, WordPiece, SentencePiece

**Stemming vs Lemmatization:**
- **Stemming**: Fast, rule-based, may produce invalid words
- **Lemmatization**: Slower, dictionary-based, produces valid words

**Why Preprocess:**
- Reduces vocabulary size
- Normalizes variations
- Removes noise
- Improves model performance

**Considerations:**
- **Domain-specific**: Medical text needs different preprocessing
- **Language**: Different rules for different languages
- **Task-dependent**: Some tasks need punctuation

**Common Libraries:**
- **NLTK**: Natural Language Toolkit
- **spaCy**: Industrial-strength NLP
- **TextBlob**: Simple API`,
					CodeExamples: `import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Download required data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

text = "The quick brown foxes are jumping over the lazy dogs. They're running fast!"

# Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

sentences = sent_tokenize(text)
print("Sentences:", sentences)

# Lowercasing
tokens_lower = [token.lower() for token in tokens]
print("Lowercased:", tokens_lower)

# Remove punctuation
tokens_clean = [re.sub(r'[^\\w\\s]', '', token) for token in tokens_lower]
tokens_clean = [token for token in tokens_clean if token]
print("No punctuation:", tokens_clean)

# Stop word removal
stop_words = set(stopwords.words('english'))
tokens_no_stop = [token for token in tokens_clean if token not in stop_words]
print("No stop words:", tokens_no_stop)

# Stemming
stemmer = PorterStemmer()
tokens_stemmed = [stemmer.stem(token) for token in tokens_no_stop]
print("Stemmed:", tokens_stemmed)

# Lemmatization
lemmatizer = WordNetLemmatizer()
tokens_lemmatized = [lemmatizer.lemmatize(token) for token in tokens_no_stop]
print("Lemmatized:", tokens_lemmatized)

# Using spaCy
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens_spacy = [token.text for token in doc]
    lemmas_spacy = [token.lemma_ for token in doc]
    print("spaCy tokens:", tokens_spacy[:10])
    print("spaCy lemmas:", lemmas_spacy[:10])
except ImportError:
    print("Install spaCy: pip install spacy && python -m spacy download en_core_web_sm")`,
				},
				{
					Title: "Word Embeddings",
					Content: `Word embeddings represent words as dense vectors capturing semantic meaning.

**Why Embeddings:**
- **Dense Representation**: Fixed-size vectors
- **Semantic Similarity**: Similar words have similar vectors
- **Contextual**: Can capture word relationships

**Word2Vec:**
- **Skip-gram**: Predict context from word
- **CBOW**: Predict word from context
- **Negative Sampling**: Efficient training

**GloVe (Global Vectors):**
- Combines global statistics with local context
- Uses co-occurrence matrix
- Often better than Word2Vec

**FastText:**
- Extends Word2Vec with subword information
- Handles out-of-vocabulary words
- Good for morphologically rich languages

**Embedding Properties:**
- **Similarity**: Cosine similarity between vectors
- **Analogies**: king - man + woman ≈ queen
- **Clustering**: Related words cluster together

**Pre-trained Embeddings:**
- **Word2Vec**: Google News vectors
- **GloVe**: Stanford vectors (Common Crawl)
- **FastText**: Facebook vectors (Wikipedia)

**Contextual Embeddings:**
- **ELMo**: Context-dependent embeddings
- **BERT**: Bidirectional context
- **GPT**: Unidirectional context

**Using Embeddings:**
- Initialize embedding layer
- Fine-tune or freeze
- Transfer learning`,
					CodeExamples: `import numpy as np
from gensim.models import Word2Vec, FastText
from gensim.downloader import load

# Train Word2Vec
sentences = [
    ['the', 'quick', 'brown', 'fox'],
    ['jumps', 'over', 'the', 'lazy', 'dog'],
    ['the', 'dog', 'is', 'lazy'],
    ['the', 'fox', 'is', 'quick']
]

# Skip-gram model
model = Word2Vec(sentences, vector_size=100, window=5, 
                min_count=1, sg=1)  # sg=1 for skip-gram

# Get word vector
fox_vector = model.wv['fox']
print(f"Fox vector shape: {fox_vector.shape}")

# Find similar words
similar = model.wv.most_similar('fox', topn=3)
print("Similar to 'fox':", similar)

# Word analogies
analogy = model.wv.most_similar(positive=['king', 'woman'], 
                                negative=['man'], topn=1)
print("king - man + woman =", analogy)

# Load pre-trained GloVe
try:
    glove_vectors = load('glove-wiki-gigaword-100')
    print("Loaded GloVe vectors")
    
    # Get vector
    vector = glove_vectors['computer']
    print(f"Computer vector shape: {vector.shape}")
    
    # Similar words
    similar = glove_vectors.most_similar('computer', topn=5)
    print("Similar to 'computer':", similar)
except:
    print("Downloading GloVe vectors...")

# Using embeddings in PyTorch
import torch
import torch.nn as nn

# Create embedding layer
vocab_size = 10000
embedding_dim = 300
embedding = nn.Embedding(vocab_size, embedding_dim)

# Load pre-trained weights (example)
# pretrained_weights = load_glove_vectors()
# embedding.weight.data.copy_(pretrained_weights)

# Use in model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        # Average pooling
        pooled = embedded.mean(dim=1)
        return self.fc(pooled)`,
				},
				{
					Title: "Text Classification",
					Content: `Text classification assigns categories or labels to text documents.

**Applications:**
- Spam detection
- Sentiment analysis
- Topic classification
- Language detection
- Intent classification

**Approaches:**
- **Bag of Words**: Count word frequencies
- **TF-IDF**: Term frequency-inverse document frequency
- **Word Embeddings**: Dense vector representations
- **Deep Learning**: CNNs, RNNs, Transformers

**Naive Bayes:**
- Probabilistic classifier
- Assumes feature independence
- Fast and interpretable
- Good baseline

**Logistic Regression:**
- Linear classifier
- Works with TF-IDF features
- Interpretable coefficients
- Fast training

**Support Vector Machines:**
- Can use kernel trick
- Good with high-dimensional features
- Robust to overfitting

**Deep Learning:**
- **CNN**: 1D convolutions over sequences
- **LSTM/GRU**: Sequential processing
- **BERT**: Pre-trained transformers

**Evaluation:**
- Accuracy, Precision, Recall, F1
- Confusion matrix
- Per-class metrics`,
					CodeExamples: `from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn

# Sample data
texts = [
    "I love this product!",
    "This is terrible.",
    "Great quality, highly recommend.",
    "Waste of money.",
    "Amazing experience!"
]
labels = [1, 0, 1, 0, 1]  # 1: positive, 0: negative

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X, labels)
nb_pred = nb_model.predict(X)
print("Naive Bayes:", nb_pred)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X, labels)
lr_pred = lr_model.predict(X)
print("Logistic Regression:", lr_pred)

# CNN for Text Classification
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, 
                 filter_sizes, num_classes, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # (batch, num_filters, seq_len')
            pooled = torch.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        return self.fc(x)

# LSTM for Text Classification
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 num_layers, num_classes, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use last hidden state
        last_hidden = hidden[-1]
        output = self.dropout(last_hidden)
        return self.fc(output)`,
				},
				{
					Title: "Sentiment Analysis",
					Content: `Sentiment analysis determines emotional tone or opinion in text.

**Types:**
- **Binary**: Positive/Negative
- **Multi-class**: Positive/Neutral/Negative
- **Fine-grained**: 5-star ratings
- **Aspect-based**: Sentiment for specific aspects

**Approaches:**
- **Lexicon-based**: Use sentiment dictionaries
- **Machine Learning**: Train classifier
- **Deep Learning**: Neural networks
- **Transformer Models**: BERT, RoBERTa

**Lexicon Methods:**
- **VADER**: Valence Aware Dictionary
- **TextBlob**: Simple sentiment scores
- **AFINN**: Word-sentiment scores

**Challenges:**
- **Sarcasm**: Hard to detect
- **Context**: Same word different sentiment
- **Negation**: "not good" vs "good"
- **Emojis**: Important for social media

**Pre-trained Models:**
- **TextBlob**: Simple API
- **VADER**: Social media focused
- **BERT-based**: State-of-the-art

**Evaluation:**
- Accuracy, F1-score
- Confusion matrix
- Per-class metrics`,
					CodeExamples: `from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
from transformers import pipeline

# TextBlob
text = "I love this movie! It's amazing."
blob = TextBlob(text)
print(f"TextBlob Polarity: {blob.sentiment.polarity}")  # -1 to 1
print(f"TextBlob Subjectivity: {blob.sentiment.subjectivity}")

# VADER (for social media)
analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores(text)
print(f"VADER Scores: {scores}")
# {'neg': 0.0, 'neu': 0.254, 'pos': 0.746, 'compound': 0.8316}

# Using BERT for sentiment
try:
    sentiment_pipeline = pipeline("sentiment-analysis", 
                                  model="nlptown/bert-base-multilingual-uncased-sentiment")
    result = sentiment_pipeline(text)
    print(f"BERT Sentiment: {result}")
except:
    print("Install transformers: pip install transformers")

# Custom sentiment classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training data
train_texts = [
    "I love this product!",
    "This is terrible.",
    "Amazing quality!",
    "Waste of money.",
    "Highly recommend!",
    "Not worth it."
]
train_labels = [1, 0, 1, 0, 1, 0]

# Train model
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
model = LogisticRegression()
model.fit(X_train, train_labels)

# Predict
test_text = "This product exceeded my expectations!"
X_test = vectorizer.transform([test_text])
prediction = model.predict(X_test)[0]
probability = model.predict_proba(X_test)[0]
print(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
print(f"Probabilities: {probability}")`,
				},
				{
					Title: "Named Entity Recognition (NER)",
					Content: `NER identifies and classifies named entities in text (persons, organizations, locations, etc.).

**Entity Types:**
- **PERSON**: People names
- **ORG**: Organizations
- **LOC**: Locations
- **GPE**: Geopolitical entities
- **DATE**: Dates
- **MONEY**: Monetary values

**Approaches:**
- **Rule-based**: Patterns and dictionaries
- **Machine Learning**: CRF, SVM
- **Deep Learning**: BiLSTM-CRF, BERT

**CRF (Conditional Random Fields):**
- Sequence labeling model
- Considers label dependencies
- Good baseline for NER

**BiLSTM-CRF:**
- Bidirectional LSTM for context
- CRF layer for label dependencies
- State-of-the-art before transformers

**BERT-based NER:**
- Fine-tune BERT for NER
- Token classification task
- Current state-of-the-art

**Evaluation:**
- **Precision**: Correct entities / Predicted entities
- **Recall**: Correct entities / Actual entities
- **F1-score**: Harmonic mean

**Applications:**
- Information extraction
- Question answering
- Knowledge graph construction
- Document understanding`,
					CodeExamples: `import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Using spaCy
try:
    nlp = spacy.load("en_core_web_sm")
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
    doc = nlp(text)
    
    print("Named Entities:")
    for ent in doc.ents:
        print(f"{ent.text}: {ent.label_} ({spacy.explain(ent.label_)})")
except:
    print("Install: python -m spacy download en_core_web_sm")

# Using BERT-based NER
try:
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, 
                           aggregation_strategy="simple")
    
    text = "My name is John Smith and I work at Google in Mountain View."
    entities = ner_pipeline(text)
    print("BERT NER:")
    for entity in entities:
        print(f"{entity['word']}: {entity['entity_group']} (confidence: {entity['score']:.2f})")
except:
    print("Install transformers: pip install transformers")

# Custom NER with BiLSTM-CRF (conceptual)
import torch
import torch.nn as nn

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                           num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        # CRF layer would be added here
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        tag_scores = self.hidden2tag(lstm_out)
        return tag_scores`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          114,
			Title:       "Recommendation Systems",
			Description: "Learn collaborative filtering, matrix factorization, content-based filtering, and hybrid recommendation approaches.",
			Order:       24,
			Lessons: []problems.Lesson{
				{
					Title: "Collaborative Filtering",
					Content: `Collaborative filtering recommends items based on user behavior patterns and similarities.

**Core Idea:**
Users who liked similar items in the past will like similar items in the future.

**User-Based Collaborative Filtering:**
- Find users similar to target user
- Recommend items liked by similar users
- Similarity: Cosine, Pearson correlation

**Item-Based Collaborative Filtering:**
- Find items similar to items user liked
- Recommend similar items
- More stable than user-based

**Similarity Metrics:**
- **Cosine Similarity**: cos(θ) = (A·B) / (||A|| ||B||)
- **Pearson Correlation**: Measures linear correlation
- **Jaccard Similarity**: For binary data

**Advantages:**
- No need for item features
- Discovers complex patterns
- Works well with implicit feedback

**Limitations:**
- **Cold Start**: New users/items have no data
- **Sparsity**: Most user-item pairs missing
- **Scalability**: Computationally expensive
- **Popularity Bias**: Popular items dominate

**Matrix Representation:**
- Rows: Users
- Columns: Items
- Values: Ratings/interactions
- Sparse matrix (most entries missing)`,
					CodeExamples: `import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# User-item rating matrix
ratings = np.array([
    [5, 4, 0, 0, 1],  # User 1
    [4, 0, 0, 1, 5],  # User 2
    [0, 5, 4, 0, 0],  # User 3
    [5, 0, 5, 4, 0],  # User 4
    [0, 4, 0, 5, 4]   # User 5
])

# User-based collaborative filtering
def user_based_cf(ratings, user_id, item_id, k=2):
    # Calculate user similarities
    user_similarities = cosine_similarity(ratings)
    
    # Get k most similar users
    similar_users = np.argsort(user_similarities[user_id])[-k-1:-1][::-1]
    
    # Predict rating
    numerator = sum(user_similarities[user_id, u] * ratings[u, item_id] 
                   for u in similar_users if ratings[u, item_id] > 0)
    denominator = sum(abs(user_similarities[user_id, u]) 
                      for u in similar_users if ratings[u, item_id] > 0)
    
    if denominator == 0:
        return 0
    
    return numerator / denominator

# Item-based collaborative filtering
def item_based_cf(ratings, user_id, item_id, k=2):
    # Transpose to get item-user matrix
    item_ratings = ratings.T
    
    # Calculate item similarities
    item_similarities = cosine_similarity(item_ratings)
    
    # Get user's ratings
    user_ratings = ratings[user_id]
    
    # Find items user has rated
    rated_items = np.where(user_ratings > 0)[0]
    
    if len(rated_items) == 0:
        return 0
    
    # Get k most similar items
    similar_items = []
    for rated_item in rated_items:
        similar = np.argsort(item_similarities[rated_item])[-k-1:-1][::-1]
        similar_items.extend([(item, item_similarities[rated_item, item], 
                              user_ratings[rated_item]) 
                             for item in similar if item != rated_item])
    
    # Predict rating
    if len(similar_items) == 0:
        return 0
    
    numerator = sum(sim * rating for _, sim, rating in similar_items[:k])
    denominator = sum(abs(sim) for _, sim, _ in similar_items[:k])
    
    return numerator / denominator if denominator > 0 else 0

# Example
prediction_user = user_based_cf(ratings, user_id=0, item_id=2)
prediction_item = item_based_cf(ratings, user_id=0, item_id=2)
print(f"User-based prediction: {prediction_user:.2f}")
print(f"Item-based prediction: {prediction_item:.2f}")`,
				},
				{
					Title: "Matrix Factorization",
					Content: `Matrix factorization decomposes user-item matrix into lower-dimensional representations.

**SVD (Singular Value Decomposition):**
R ≈ U × Σ × V^T
- U: User factors (latent features)
- V: Item factors (latent features)
- Σ: Singular values

**Matrix Factorization Model:**
R ≈ P × Q^T
- P: User embedding matrix (n_users × k)
- Q: Item embedding matrix (n_items × k)
- k: Number of latent factors

**Prediction:**
r̂_ui = p_u · q_i
- Dot product of user and item vectors

**Learning:**
Minimize: Σ (r_ui - p_u · q_i)² + λ(||p_u||² + ||q_i||²)
- First term: Reconstruction error
- Second term: Regularization

**Stochastic Gradient Descent:**
For each observed rating (u, i, r):
- p_u ← p_u + α[(r - p_u·q_i)q_i - λp_u]
- q_i ← q_i + α[(r - p_u·q_i)p_u - λq_i]

**Advantages:**
- Handles sparsity well
- Captures latent factors
- Scalable
- Good accuracy

**Extensions:**
- **Bias Terms**: Account for user/item biases
- **Non-negative**: NMF for interpretability
- **Deep Learning**: Neural matrix factorization`,
					CodeExamples: `import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix

# User-item matrix
ratings = np.array([
    [5, 4, 0, 0, 1],
    [4, 0, 0, 1, 5],
    [0, 5, 4, 0, 0],
    [5, 0, 5, 4, 0],
    [0, 4, 0, 5, 4]
])

# Matrix Factorization with SGD
class MatrixFactorization:
    def __init__(self, n_factors=2, learning_rate=0.01, reg=0.01, n_epochs=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
    
    def fit(self, ratings):
        n_users, n_items = ratings.shape
        
        # Initialize matrices
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Get non-zero ratings
        rows, cols = np.nonzero(ratings)
        
        # Training
        for epoch in range(self.n_epochs):
            for u, i in zip(rows, cols):
                r_ui = ratings[u, i]
                pred = np.dot(self.P[u], self.Q[i])
                error = r_ui - pred
                
                # Update
                P_u_old = self.P[u].copy()
                self.P[u] += self.learning_rate * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.learning_rate * (error * P_u_old - self.reg * self.Q[i])
    
    def predict(self, user_id, item_id):
        return np.dot(self.P[user_id], self.Q[item_id])
    
    def predict_all(self):
        return np.dot(self.P, self.Q.T)

# Train model
mf = MatrixFactorization(n_factors=2, n_epochs=100)
mf.fit(ratings)

# Predictions
predictions = mf.predict_all()
print("Predictions:")
print(predictions)

# Using NMF (Non-negative Matrix Factorization)
nmf = NMF(n_components=2, random_state=42)
W = nmf.fit_transform(ratings)  # User factors
H = nmf.components_  # Item factors
nmf_predictions = np.dot(W, H)
print("\nNMF Predictions:")
print(nmf_predictions)`,
				},
				{
					Title: "Content-Based Filtering",
					Content: `Content-based filtering recommends items similar to items user liked, based on item features.

**Core Idea:**
If user liked item with certain features, recommend items with similar features.

**Process:**
1. Extract item features
2. Build user profile from liked items
3. Find items similar to user profile
4. Recommend top items

**Item Features:**
- **Text**: TF-IDF, embeddings
- **Categories**: One-hot encoding
- **Metadata**: Genre, director, etc.
- **Images**: Visual features

**User Profile:**
- Weighted average of liked items' features
- Weights: ratings or binary preferences

**Similarity:**
- Cosine similarity between user profile and items
- Higher similarity = better recommendation

**Advantages:**
- No cold start for items (have features)
- Interpretable (can explain why)
- No popularity bias
- Works for niche items

**Limitations:**
- Requires item features
- Limited diversity
- Cold start for new users
- Feature engineering needed`,
					CodeExamples: `from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Sample data: movies with genres
movies = pd.DataFrame({
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['Action Movie', 'Comedy Film', 'Action Thriller', 'Romantic Comedy', 'Sci-Fi Action'],
    'genres': ['action thriller', 'comedy', 'action thriller', 'romance comedy', 'sci-fi action']
})

# User ratings
user_ratings = {
    1: 5,  # Liked action movie
    3: 4   # Liked action thriller
}

# Extract features using TF-IDF
vectorizer = TfidfVectorizer()
item_features = vectorizer.fit_transform(movies['genres'])

# Build user profile
user_profile = np.zeros(item_features.shape[1])
for movie_id, rating in user_ratings.items():
    idx = movies[movies['movie_id'] == movie_id].index[0]
    user_profile += rating * item_features[idx].toarray().flatten()

# Normalize
user_profile = user_profile / sum(user_ratings.values())

# Calculate similarities
similarities = cosine_similarity(user_profile.reshape(1, -1), item_features).flatten()

# Get recommendations
recommendations = pd.DataFrame({
    'movie_id': movies['movie_id'],
    'title': movies['title'],
    'similarity': similarities
}).sort_values('similarity', ascending=False)

print("Recommendations:")
print(recommendations)

# Content-based with multiple features
def content_based_recommend(user_id, item_features, user_ratings, top_n=5):
    # Build user profile
    user_profile = np.zeros(item_features.shape[1])
    rated_items = [item for item, rating in user_ratings.items() if rating > 0]
    
    for item_id in rated_items:
        item_idx = item_id - 1  # Assuming 1-indexed
        rating = user_ratings[item_id]
        user_profile += rating * item_features[item_idx].toarray().flatten()
    
    if len(rated_items) > 0:
        user_profile /= len(rated_items)
    
    # Calculate similarities
    similarities = cosine_similarity(user_profile.reshape(1, -1), item_features).flatten()
    
    # Get top N (excluding already rated)
    top_indices = np.argsort(similarities)[::-1]
    recommendations = [idx + 1 for idx in top_indices 
                     if (idx + 1) not in rated_items][:top_n]
    
    return recommendations

recs = content_based_recommend(1, item_features, user_ratings)
print(f"\nTop recommendations: {recs}")`,
				},
				{
					Title: "Hybrid Approaches",
					Content: `Hybrid recommendation systems combine multiple approaches for better performance.

**Hybrid Strategies:**
- **Weighted**: Combine predictions with weights
- **Switching**: Use different methods in different contexts
- **Cascading**: Refine recommendations from one method with another
- **Feature Combination**: Merge features from different sources
- **Meta-Learning**: Learn to combine methods

**Weighted Hybrid:**
r̂_ui = α × r̂_collab + (1-α) × r̂_content
- α: Weight parameter
- Tune α on validation set

**Switching:**
- Use collaborative when enough data
- Use content-based for cold start
- Context-dependent switching

**Cascading:**
- First: Content-based filtering
- Then: Collaborative filtering on results
- Refines recommendations

**Feature Combination:**
- Combine user-item interactions with item features
- Neural networks can learn interactions
- More expressive

**Advantages:**
- Better accuracy
- Handles limitations of individual methods
- More robust
- Better coverage

**Challenges:**
- More complex
- Requires tuning
- Higher computational cost`,
					CodeExamples: `import numpy as np

# Hybrid: Weighted combination
def hybrid_recommend(user_id, item_id, collab_pred, content_pred, alpha=0.7):
    """
    Combine collaborative and content-based predictions
    alpha: weight for collaborative filtering
    """
    hybrid_pred = alpha * collab_pred + (1 - alpha) * content_pred
    return hybrid_pred

# Example
collab_pred = 4.2
content_pred = 3.8
hybrid_pred = hybrid_recommend(1, 1, collab_pred, content_pred, alpha=0.6)
print(f"Hybrid prediction: {hybrid_pred:.2f}")

# Switching hybrid
def switching_hybrid(user_id, item_id, user_ratings_count, 
                    collab_pred, content_pred, threshold=5):
    """
    Use collaborative if user has enough ratings, else content-based
    """
    if user_ratings_count >= threshold:
        return collab_pred
    else:
        return content_pred

# Cascading hybrid
def cascading_hybrid(user_id, item_features, user_ratings, 
                    collab_similarities, top_n=10):
    """
    First filter by content, then rank by collaborative
    """
    # Content-based filtering
    user_profile = build_user_profile(user_ratings, item_features)
    content_similarities = cosine_similarity(
        user_profile.reshape(1, -1), item_features
    ).flatten()
    
    # Get top items by content
    content_top = np.argsort(content_similarities)[-top_n*2:][::-1]
    
    # Re-rank by collaborative filtering
    hybrid_scores = []
    for item_idx in content_top:
        score = 0.5 * content_similarities[item_idx] + 0.5 * collab_similarities[item_idx]
        hybrid_scores.append((item_idx, score))
    
    # Sort and return top N
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in hybrid_scores[:top_n]]

# Neural collaborative filtering (conceptual)
import torch
import torch.nn as nn

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=50):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        concat = torch.cat([user_emb, item_emb], dim=1)
        return self.fc_layers(concat)`,
				},
				{
					Title: "Evaluation Metrics",
					Content: `Evaluating recommendation systems requires appropriate metrics.

**Rating Prediction Metrics:**
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- Lower is better

**Ranking Metrics:**
- **Precision@K**: Relevant items in top K / K
- **Recall@K**: Relevant items in top K / Total relevant
- **NDCG**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision

**Coverage:**
- **Catalog Coverage**: % of items recommended
- **Diversity**: How different recommendations are
- **Novelty**: How surprising recommendations are

**Business Metrics:**
- **Click-through Rate**: % of recommendations clicked
- **Conversion Rate**: % leading to purchase
- **Revenue**: Total revenue from recommendations

**Offline Evaluation:**
- Train/test split
- Cross-validation
- Hold-out validation

**Online Evaluation:**
- A/B testing
- Real user feedback
- Business metrics`,
					CodeExamples: `import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Rating prediction metrics
def evaluate_ratings(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'MAE': mae, 'RMSE': rmse}

# Ranking metrics
def precision_at_k(relevant_items, recommended_items, k):
    recommended_k = recommended_items[:k]
    relevant_recommended = len(set(recommended_k) & set(relevant_items))
    return relevant_recommended / k if k > 0 else 0

def recall_at_k(relevant_items, recommended_items, k):
    recommended_k = recommended_items[:k]
    relevant_recommended = len(set(recommended_k) & set(relevant_items))
    return relevant_recommended / len(relevant_items) if len(relevant_items) > 0 else 0

def ndcg_at_k(relevant_items, recommended_items, k):
    recommended_k = recommended_items[:k]
    dcg = sum((1 if item in relevant_items else 0) / np.log2(idx + 2) 
             for idx, item in enumerate(recommended_k))
    
    ideal_relevant = sorted(relevant_items, reverse=True)[:k]
    idcg = sum(1 / np.log2(idx + 2) for idx in range(len(ideal_relevant)))
    
    return dcg / idcg if idcg > 0 else 0

# Example
relevant = [1, 3, 5, 7, 9]
recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(f"Precision@5: {precision_at_k(relevant, recommended, 5):.2f}")
print(f"Recall@5: {recall_at_k(relevant, recommended, 5):.2f}")
print(f"NDCG@5: {ndcg_at_k(relevant, recommended, 5):.2f}")

# Coverage
def catalog_coverage(all_items, recommended_items):
    unique_recommended = len(set(recommended_items))
    return unique_recommended / len(all_items) if len(all_items) > 0 else 0

# Diversity (pairwise similarity)
def diversity(recommended_items, item_features):
    if len(recommended_items) < 2:
        return 0
    
    similarities = []
    for i in range(len(recommended_items)):
        for j in range(i+1, len(recommended_items)):
            sim = cosine_similarity(
                item_features[recommended_items[i]].reshape(1, -1),
                item_features[recommended_items[j]].reshape(1, -1)
            )[0, 0]
            similarities.append(sim)
    
    return 1 - np.mean(similarities)  # Higher diversity = lower similarity`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          115,
			Title:       "Anomaly Detection",
			Description: "Learn anomaly detection techniques: statistical methods, isolation forest, one-class SVM, and deep learning approaches.",
			Order:       25,
			Lessons: []problems.Lesson{
				{
					Title: "Anomaly Detection Fundamentals",
					Content: `Anomaly detection identifies rare items, events, or observations that differ significantly from the majority.

**Types of Anomalies:**
- **Point Anomalies**: Individual data points
- **Contextual Anomalies**: Anomalous in specific context
- **Collective Anomalies**: Collection of related instances

**Applications:**
- Fraud detection
- Intrusion detection
- Medical diagnosis
- Manufacturing quality control
- Network monitoring

**Challenges:**
- **Imbalanced Data**: Very few anomalies
- **Label Scarcity**: Often no labels
- **Definition**: What is "normal"?
- **Evolving Patterns**: Normal changes over time

**Approaches:**
- **Statistical**: Z-score, IQR
- **Distance-based**: KNN, LOF
- **Density-based**: Isolation Forest
- **Machine Learning**: One-Class SVM, Autoencoders
- **Deep Learning**: LSTM, GANs

**Evaluation:**
- **Precision/Recall**: If labels available
- **ROC-AUC**: For binary classification
- **Visualization**: Manual inspection
- **Business Metrics**: False positive rate`,
					CodeExamples: `import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate sample data with anomalies
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 2))
anomalies = np.random.uniform(-5, 5, (50, 2))
data = np.vstack([normal_data, anomalies])
labels = np.hstack([np.zeros(1000), np.ones(50)])

# Visualize
plt.scatter(data[labels==0, 0], data[labels==0, 1], 
           alpha=0.5, label='Normal')
plt.scatter(data[labels==1, 0], data[labels==1, 1], 
           c='red', label='Anomaly')
plt.legend()
plt.title('Anomaly Detection Dataset')
plt.show()

# Statistical methods
def z_score_anomaly_detection(data, threshold=3):
    z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    anomalies = np.any(z_scores > threshold, axis=1)
    return anomalies

def iqr_anomaly_detection(data):
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies = np.any((data < lower_bound) | (data > upper_bound), axis=1)
    return anomalies

# Detect anomalies
z_anomalies = z_score_anomaly_detection(data)
iqr_anomalies = iqr_anomaly_detection(data)

print(f"Z-score anomalies: {np.sum(z_anomalies)}")
print(f"IQR anomalies: {np.sum(iqr_anomalies)}")`,
				},
				{
					Title: "Isolation Forest",
					Content: `Isolation Forest isolates anomalies instead of profiling normal data.

**Core Idea:**
Anomalies are easier to isolate than normal points.

**Algorithm:**
1. Randomly select feature and split value
2. Recursively partition data
3. Anomalies isolated in fewer splits
4. Average path length indicates anomaly score

**Isolation Process:**
- Normal points: Need many splits to isolate
- Anomalies: Isolated in few splits
- Path length inversely related to anomaly score

**Anomaly Score:**
s(x, n) = 2^(-E(h(x))/c(n))
- h(x): Path length
- E(h(x)): Average path length
- c(n): Normalization constant
- Score close to 1: Anomaly
- Score close to 0: Normal

**Advantages:**
- Handles high-dimensional data
- No need for normal data distribution
- Fast training and prediction
- Works with mixed data types

**Parameters:**
- **n_estimators**: Number of trees
- **max_samples**: Sample size per tree
- **contamination**: Expected proportion of anomalies`,
					CodeExamples: `from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, 
                            random_state=42)
anomaly_labels = iso_forest.fit_predict(data)

# Convert to binary (1 = normal, -1 = anomaly)
anomaly_labels_binary = (anomaly_labels == -1).astype(int)

# Anomaly scores
anomaly_scores = iso_forest.score_samples(data)

# Evaluate
print("Classification Report:")
print(classification_report(labels, anomaly_labels_binary, 
                           target_names=['Normal', 'Anomaly']))

# Visualize
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=anomaly_scores, cmap='coolwarm')
plt.colorbar(label='Anomaly Score')
plt.title('Isolation Forest Anomaly Scores')

plt.subplot(1, 2, 2)
plt.scatter(data[anomaly_labels == 1, 0], 
           data[anomaly_labels == 1, 1], 
           alpha=0.5, label='Normal')
plt.scatter(data[anomaly_labels == -1, 0], 
           data[anomaly_labels == -1, 1], 
           c='red', label='Anomaly')
plt.legend()
plt.title('Detected Anomalies')
plt.show()

# Tuning contamination parameter
contaminations = [0.01, 0.05, 0.1, 0.2]
for cont in contaminations:
    iso = IsolationForest(contamination=cont, random_state=42)
    pred = iso.fit_predict(data)
    n_anomalies = np.sum(pred == -1)
    print(f"Contamination {cont}: {n_anomalies} anomalies detected")`,
				},
				{
					Title: "One-Class SVM",
					Content: `One-Class SVM learns a decision boundary that separates normal data from outliers.

**Core Idea:**
Find hyperplane that separates normal data from origin in feature space.

**Mathematical Formulation:**
Minimize: (1/2)||w||² + (1/νn)Σξᵢ - ρ
Subject to: w·φ(xᵢ) ≥ ρ - ξᵢ, ξᵢ ≥ 0

- w: Weight vector
- φ: Kernel function
- ξ: Slack variables
- ρ: Offset
- ν: Controls fraction of outliers

**Kernel Trick:**
- Maps data to higher dimension
- Common kernels: RBF, polynomial, linear
- RBF most common for non-linear boundaries

**Decision Function:**
f(x) = sign(w·φ(x) - ρ)
- f(x) > 0: Normal
- f(x) < 0: Anomaly

**Advantages:**
- Works well with non-linear boundaries
- Handles high-dimensional data
- Probabilistic outputs possible

**Limitations:**
- Sensitive to kernel parameters
- Can be slow for large datasets
- Requires tuning`,
					CodeExamples: `from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report

# One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
oc_svm.fit(data)

# Predictions
oc_predictions = oc_svm.predict(data)
oc_scores = oc_svm.score_samples(data)

# Convert to binary
oc_binary = (oc_predictions == -1).astype(int)

# Evaluate
print("One-Class SVM Results:")
print(classification_report(labels, oc_binary, 
                           target_names=['Normal', 'Anomaly']))

# Visualize decision boundary
def plot_decision_boundary(model, data, labels):
    h = 0.1
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(data[labels==0, 0], data[labels==0, 1], 
               alpha=0.5, label='Normal')
    plt.scatter(data[labels==1, 0], data[labels==1, 1], 
               c='red', label='Anomaly')
    plt.legend()
    plt.title('One-Class SVM Decision Boundary')

plot_decision_boundary(oc_svm, data, labels)
plt.show()

# Tuning parameters
nu_values = [0.01, 0.05, 0.1, 0.2]
gamma_values = ['scale', 'auto', 0.1, 1.0]

for nu in nu_values:
    for gamma in gamma_values:
        oc = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        oc.fit(data)
        pred = oc.predict(data)
        n_anomalies = np.sum(pred == -1)
        print(f"nu={nu}, gamma={gamma}: {n_anomalies} anomalies")`,
				},
				{
					Title: "Autoencoders for Anomaly Detection",
					Content: `Autoencoders learn to reconstruct normal data; anomalies have high reconstruction error.

**Autoencoder Architecture:**
- **Encoder**: Compresses input to latent representation
- **Decoder**: Reconstructs input from latent code
- **Bottleneck**: Lower-dimensional representation

**Anomaly Detection:**
- Train on normal data only
- Anomalies have high reconstruction error
- Threshold reconstruction error to detect anomalies

**Why It Works:**
- Autoencoder learns normal patterns
- Can't reconstruct unseen anomalies well
- Reconstruction error indicates anomaly

**Types:**
- **Vanilla Autoencoder**: Basic encoder-decoder
- **Variational Autoencoder**: Probabilistic latent space
- **LSTM Autoencoder**: For sequential data

**Advantages:**
- Learns complex normal patterns
- Handles high-dimensional data
- No feature engineering needed
- Can use pre-trained models

**Limitations:**
- Requires training data
- May overfit to training distribution
- Threshold selection critical`,
					CodeExamples: `import torch
import torch.nn as nn
import torch.optim as optim

# Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Prepare data (normalize)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
normal_data = data_scaled[labels == 0]

# Convert to tensors
normal_tensor = torch.FloatTensor(normal_data)
all_tensor = torch.FloatTensor(data_scaled)

# Initialize model
input_dim = data_scaled.shape[1]
encoding_dim = 10
model = Autoencoder(input_dim, encoding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train on normal data only
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    reconstructed = model(normal_tensor)
    loss = criterion(reconstructed, normal_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Detect anomalies
model.eval()
with torch.no_grad():
    reconstructed_all = model(all_tensor)
    reconstruction_errors = torch.mean((all_tensor - reconstructed_all) ** 2, dim=1).numpy()

# Set threshold (e.g., 95th percentile)
threshold = np.percentile(reconstruction_errors[labels == 0], 95)
anomaly_predictions = (reconstruction_errors > threshold).astype(int)

print(f"Threshold: {threshold:.4f}")
print(f"Detected anomalies: {np.sum(anomaly_predictions)}")

# Visualize reconstruction errors
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(reconstruction_errors[labels == 0], bins=50, alpha=0.7, label='Normal')
plt.hist(reconstruction_errors[labels == 1], bins=50, alpha=0.7, label='Anomaly')
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.title('Reconstruction Error Distribution')

plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=reconstruction_errors, cmap='coolwarm')
plt.colorbar(label='Reconstruction Error')
plt.title('Reconstruction Errors')
plt.show()`,
				},
				{
					Title: "LSTM for Anomaly Detection",
					Content: `LSTM networks can detect anomalies in sequential/time series data.

**Approach:**
- Train LSTM to predict next value in sequence
- High prediction error indicates anomaly
- Can model temporal dependencies

**Architecture:**
- LSTM layers for sequence modeling
- Predict next time step
- Compare prediction to actual
- Flag high errors as anomalies

**Advantages:**
- Captures temporal patterns
- Handles long sequences
- Can detect contextual anomalies

**Applications:**
- Network intrusion detection
- Sensor anomaly detection
- Financial fraud detection
- Equipment failure prediction`,
					CodeExamples: `import torch
import torch.nn as nn

# LSTM Autoencoder for sequences
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, 
                               batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, 
                               batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # Encode
        encoded, (hidden, cell) = self.encoder(x)
        
        # Decode (use last encoded state)
        decoder_input = encoded[:, -1:, :]
        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        output = self.fc(decoded)
        
        return output

# LSTM Predictor
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Predict next time step
        prediction = self.fc(lstm_out[:, -1, :])
        return prediction

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Example usage
seq_length = 10
X, y = create_sequences(data_scaled, seq_length)

# Train on normal sequences only
normal_mask = labels[:-seq_length] == 0
X_normal = X[normal_mask]
y_normal = y[normal_mask]

X_tensor = torch.FloatTensor(X_normal)
y_tensor = torch.FloatTensor(y_normal)

model = LSTMPredictor(input_dim=X.shape[2], hidden_dim=50)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)
    loss.backward()
    optimizer.step()

# Detect anomalies
model.eval()
with torch.no_grad():
    all_X = torch.FloatTensor(X)
    predictions_all = model(all_X)
    errors = torch.mean((torch.FloatTensor(y) - predictions_all) ** 2, dim=1).numpy()

threshold = np.percentile(errors[normal_mask], 95)
anomaly_predictions = (errors > threshold).astype(int)
print(f"LSTM detected {np.sum(anomaly_predictions)} anomalies")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          116,
			Title:       "Optimization Algorithms",
			Description: "Learn gradient descent variants, learning rate scheduling, convex optimization, and optimization techniques for deep learning.",
			Order:       26,
			Lessons: []problems.Lesson{
				{
					Title: "Gradient Descent Variants",
					Content: `Gradient descent and its variants are fundamental optimization algorithms for machine learning.

**Gradient Descent:**
θ ← θ - α∇_θ L(θ)
- α: Learning rate
- ∇_θ L(θ): Gradient of loss

**Batch Gradient Descent:**
- Uses all training data
- Computes true gradient
- Stable but slow
- Memory intensive

**Stochastic Gradient Descent (SGD):**
- Uses single sample per update
- Noisy but fast
- Can escape local minima
- High variance

**Mini-Batch Gradient Descent:**
- Uses small batch (32-256 samples)
- Balance between batch and SGD
- Most common in practice
- Parallelizable

**Momentum:**
- Accumulates gradient history
- v_t = βv_{t-1} + α∇L
- θ ← θ - v_t
- Helps escape local minima
- Faster convergence

**Nesterov Accelerated Gradient (NAG):**
- Lookahead momentum
- v_t = βv_{t-1} + α∇L(θ - βv_{t-1})
- Better than standard momentum

**AdaGrad:**
- Adaptive learning rate per parameter
- Accumulates squared gradients
- Learning rate decreases over time
- Good for sparse gradients

**RMSprop:**
- Fixes AdaGrad's decaying learning rate
- Exponential moving average of squared gradients
- Better for non-stationary objectives

**Adam (Adaptive Moment Estimation):**
- Combines momentum and RMSprop
- Maintains per-parameter learning rates
- Bias correction for initial estimates
- Most popular optimizer

**AdamW:**
- Adam with decoupled weight decay
- Better generalization than Adam
- Separates weight decay from gradient update`,
					CodeExamples: `import numpy as np
import matplotlib.pyplot as plt

# Simple gradient descent
def gradient_descent(f, grad_f, x0, learning_rate=0.01, n_iterations=100):
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(n_iterations):
        gradient = grad_f(x)
        x = x - learning_rate * gradient
        history.append(x.copy())
    
    return x, history

# Example: Minimize f(x) = x²
f = lambda x: x**2
grad_f = lambda x: 2*x

x_opt, history = gradient_descent(f, grad_f, x0=5.0, learning_rate=0.1)
print(f"Optimum: {x_opt:.4f} (true: 0.0)")

# Momentum
class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v = None
    
    def update(self, gradient):
        if self.v is None:
            self.v = np.zeros_like(gradient)
        
        self.v = self.momentum * self.v + self.lr * gradient
        return -self.v

# Adam optimizer (simplified)
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, gradient):
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)
        
        self.t += 1
        
        # Update biased first moment
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        
        # Update biased second moment
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update
        update = -self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return update

# Using PyTorch optimizers
import torch
import torch.optim as optim

# Create model
model = torch.nn.Linear(10, 1)

# Different optimizers
optimizers = {
    'SGD': optim.SGD(model.parameters(), lr=0.01),
    'SGD Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'Adam': optim.Adam(model.parameters(), lr=0.001),
    'AdamW': optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.01)
}

# Training loop example
for name, optimizer in optimizers.items():
    print(f"{name}: {optimizer}")`,
				},
				{
					Title: "Learning Rate Scheduling",
					Content: `Learning rate scheduling adapts learning rate during training for better convergence.

**Why Schedule Learning Rate:**
- Start with larger LR for fast learning
- Decrease LR for fine-tuning
- Escape local minima early
- Converge to better minima

**Fixed Learning Rate:**
- Constant throughout training
- Simple but may not converge well
- May overshoot optimum

**Step Decay:**
- Reduce LR by factor every N epochs
- α(t) = α₀ × γ^floor(t/N)
- Common: Reduce by 0.1 every 10 epochs

**Exponential Decay:**
- Continuous decay
- α(t) = α₀ × e^(-kt)
- Smooth decrease

**Cosine Annealing:**
- α(t) = α_min + (α_max - α_min)(1 + cos(πt/T))/2
- Smooth decrease to minimum
- Can restart (SGDR)

**Reduce on Plateau:**
- Reduce LR when loss plateaus
- Monitor validation loss
- Patient: wait N epochs without improvement

**Warmup:**
- Gradually increase LR at start
- Helps with training stability
- Common in transformers

**Cyclical Learning Rates:**
- Cycle between min and max LR
- Helps escape local minima
- Can improve generalization`,
					CodeExamples: `import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Step decay
optimizer = optim.SGD([torch.randn(2, 2, requires_grad=True)], lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

lrs_step = []
for epoch in range(50):
    lrs_step.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

# Exponential decay
optimizer2 = optim.SGD([torch.randn(2, 2, requires_grad=True)], lr=0.1)
scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.95)

lrs_exp = []
for epoch in range(50):
    lrs_exp.append(optimizer2.param_groups[0]['lr'])
    scheduler2.step()

# Cosine annealing
optimizer3 = optim.SGD([torch.randn(2, 2, requires_grad=True)], lr=0.1)
scheduler3 = optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=50)

lrs_cosine = []
for epoch in range(50):
    lrs_cosine.append(optimizer3.param_groups[0]['lr'])
    scheduler3.step()

# Reduce on plateau
optimizer4 = optim.SGD([torch.randn(2, 2, requires_grad=True)], lr=0.1)
scheduler4 = optim.lr_scheduler.ReduceLROnPlateau(optimizer4, mode='min', 
                                                 factor=0.5, patience=5)

lrs_plateau = []
losses = [1.0] * 10 + [0.5] * 10 + [0.3] * 10 + [0.2] * 20  # Simulated loss
for loss in losses:
    lrs_plateau.append(optimizer4.param_groups[0]['lr'])
    scheduler4.step(loss)

# Plot
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(lrs_step)
plt.title('Step Decay')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.subplot(2, 2, 2)
plt.plot(lrs_exp)
plt.title('Exponential Decay')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.subplot(2, 2, 3)
plt.plot(lrs_cosine)
plt.title('Cosine Annealing')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.subplot(2, 2, 4)
plt.plot(lrs_plateau)
plt.title('Reduce on Plateau')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.tight_layout()
plt.show()

# Custom learning rate schedule
def custom_lr_schedule(epoch):
    if epoch < 10:
        return 0.01
    elif epoch < 30:
        return 0.001
    else:
        return 0.0001

scheduler_custom = optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=custom_lr_schedule
)`,
				},
				{
					Title: "Convex Optimization Basics",
					Content: `Convex optimization deals with minimizing convex functions over convex sets.

**Convex Function:**
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
- Graph lies below line segment
- Any local minimum is global minimum
- Easier to optimize

**Convex Sets:**
- Line segment between any two points in set is in set
- Examples: Hyperplanes, half-spaces, balls

**Convex Optimization Problem:**
minimize f(x)
subject to g_i(x) ≤ 0, h_j(x) = 0
- f, g_i convex
- h_j affine

**Why Convex:**
- Global optimum guaranteed
- Efficient algorithms
- Strong duality
- KKT conditions characterize optimum

**Examples:**
- Linear programming
- Quadratic programming
- Least squares
- Logistic regression (convex)

**Non-Convex:**
- Neural networks
- Clustering
- Many ML problems

**Gradient Descent for Convex:**
- Converges to global minimum
- Convergence rate: O(1/t) for smooth
- O(1/√t) for non-smooth`,
					CodeExamples: `import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Convex function: f(x) = x²
def convex_function(x):
    return x**2

# Non-convex function: f(x) = x⁴ - 2x²
def non_convex_function(x):
    return x**4 - 2*x**2

# Plot
x = np.linspace(-2, 2, 100)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, convex_function(x))
plt.title('Convex Function: f(x) = x²')
plt.xlabel('x')
plt.ylabel('f(x)')

plt.subplot(1, 2, 2)
plt.plot(x, non_convex_function(x))
plt.title('Non-Convex Function: f(x) = x⁴ - 2x²')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

# Convex optimization: Least squares
# minimize ||Ax - b||²
A = np.random.randn(20, 10)
b = np.random.randn(20)
x_true = np.random.randn(10)

# Using scipy
def objective(x):
    return np.sum((A @ x - b)**2)

result = minimize(objective, x0=np.zeros(10), method='BFGS')
x_opt = result.x

# Analytical solution
x_analytical = np.linalg.lstsq(A, b, rcond=None)[0]

print(f"Optimized: {np.linalg.norm(x_opt - x_analytical):.6f}")

# Quadratic programming (convex)
from cvxpy import Variable, Minimize, Problem, quad_form

try:
    # minimize x^T P x + q^T x
    # subject to Gx <= h, Ax = b
    x = Variable(2)
    P = np.array([[2, 0], [0, 2]])
    q = np.array([-1, -1])
    
    objective = Minimize(quad_form(x, P) + q.T @ x)
    constraints = [x[0] + x[1] <= 1, x >= 0]
    prob = Problem(objective, constraints)
    prob.solve()
    
    print(f"Optimal value: {prob.value:.4f}")
    print(f"Optimal x: {x.value}")
except ImportError:
    print("Install cvxpy: pip install cvxpy")`,
				},
				{
					Title: "Second-Order Methods",
					Content: `Second-order methods use second derivatives (Hessian) for faster convergence.

**Newton's Method:**
θ ← θ - H⁻¹∇L
- H: Hessian matrix
- Uses curvature information
- Quadratic convergence (very fast)

**Advantages:**
- Faster convergence than gradient descent
- Fewer iterations needed
- Better for well-conditioned problems

**Disadvantages:**
- Computationally expensive (O(n³) for inversion)
- Requires computing Hessian
- May not converge if not convex
- Memory intensive for large problems

**Quasi-Newton Methods:**
- Approximate Hessian instead of computing
- **BFGS**: Broyden-Fletcher-Goldfarb-Shanno
- **L-BFGS**: Limited memory BFGS
- More practical than Newton's

**L-BFGS:**
- Limited memory version
- Stores only recent gradients
- Good for large problems
- Used in scikit-learn, PyTorch

**When to Use:**
- Small to medium problems
- Smooth, well-conditioned
- When gradient descent is slow
- Can afford computation`,
					CodeExamples: `import numpy as np
from scipy.optimize import minimize, fmin_l_bfgs_b
import torch
import torch.optim as optim

# Newton's method (for small problems)
def newtons_method(f, grad_f, hess_f, x0, n_iterations=10):
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(n_iterations):
        gradient = grad_f(x)
        hessian = hess_f(x)
        try:
            hessian_inv = np.linalg.inv(hessian)
            x = x - hessian_inv @ gradient
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            hessian_inv = np.linalg.pinv(hessian)
            x = x - hessian_inv @ gradient
        history.append(x.copy())
    
    return x, history

# Example: Minimize f(x) = x₁² + 2x₂²
f = lambda x: x[0]**2 + 2*x[1]**2
grad_f = lambda x: np.array([2*x[0], 4*x[1]])
hess_f = lambda x: np.array([[2, 0], [0, 4]])

x_opt, history = newtons_method(f, grad_f, hess_f, x0=np.array([3.0, 3.0]))
print(f"Newton's method optimum: {x_opt}")

# Using scipy's BFGS (quasi-Newton)
result = minimize(f, x0=np.array([3.0, 3.0]), method='BFGS', jac=grad_f)
print(f"BFGS optimum: {result.x}")

# L-BFGS
result_lbfgs = minimize(f, x0=np.array([3.0, 3.0]), method='L-BFGS-B', jac=grad_f)
print(f"L-BFGS optimum: {result_lbfgs.x}")

# Using PyTorch L-BFGS
model = torch.nn.Linear(10, 1)
optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=20)

# L-BFGS requires closure
def closure():
    optimizer.zero_grad()
    output = model(torch.randn(5, 10))
    loss = output.mean()
    loss.backward()
    return loss

for epoch in range(10):
    optimizer.step(closure)

print("L-BFGS training completed")`,
				},
				{
					Title: "Optimization for Deep Learning",
					Content: `Deep learning optimization has unique challenges requiring specialized techniques.

**Challenges:**
- **Non-convex**: Many local minima
- **High-dimensional**: Millions of parameters
- **Large datasets**: Can't use full gradient
- **Ill-conditioned**: Different scales

**Techniques:**
- **Batch Normalization**: Normalizes activations
- **Gradient Clipping**: Prevents exploding gradients
- **Warmup**: Gradual learning rate increase
- **Mixed Precision**: FP16/BF16 for speed

**Gradient Clipping:**
- Clip gradients to prevent explosion
- Methods: By value, by norm
- Common in RNNs, transformers

**Learning Rate Warmup:**
- Start with small LR
- Gradually increase
- Helps training stability
- Common in transformers

**Adaptive Optimizers:**
- Adam, AdamW most common
- Per-parameter learning rates
- Good default choice

**Second-Order Methods:**
- Usually too expensive
- L-BFGS sometimes used
- K-FAC: Kronecker-factored approximation

**Best Practices:**
- Use Adam/AdamW as default
- Learning rate scheduling
- Gradient clipping for RNNs
- Batch normalization
- Warmup for large models`,
					CodeExamples: `import torch
import torch.nn as nn
import torch.optim as optim

# Gradient clipping
def train_with_clipping(model, dataloader, max_norm=1.0):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch['input'])
        loss = criterion(output, batch['target'])
        loss.backward()
        
        # Gradient clipping by norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        optimizer.step()

# Learning rate warmup
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            lr = self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Example training with warmup
model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = WarmupScheduler(optimizer, warmup_epochs=5, base_lr=0.001)

for epoch in range(20):
    scheduler.step()
    print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}")

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(batch['input'])
        loss = criterion(output, batch['target'])
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Lookahead optimizer (wraps any optimizer)
class Lookahead:
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.slow_weights = {name: param.clone() 
                           for name, param in optimizer.param_groups[0]['params'][0].named_parameters()}
    
    def step(self):
        self.optimizer.step()
        self.step_count += 1
        
        if self.step_count % self.k == 0:
            # Update slow weights
            for name, param in self.optimizer.param_groups[0]['params'][0].named_parameters():
                self.slow_weights[name] += self.alpha * (param - self.slow_weights[name])
                param.data.copy_(self.slow_weights[name])`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          117,
			Title:       "Bayesian Methods",
			Description: "Learn Bayesian inference, MAP estimation, Bayesian neural networks, Gaussian processes, and MCMC methods.",
			Order:       27,
			Lessons: []problems.Lesson{
				{
					Title: "Bayesian Inference Fundamentals",
					Content: `Bayesian methods provide probabilistic framework for learning from data with uncertainty quantification.

**Bayes' Theorem:**
P(θ|D) = P(D|θ) × P(θ) / P(D)
- **Posterior**: P(θ|D) - Updated belief after seeing data
- **Likelihood**: P(D|θ) - Probability of data given parameters
- **Prior**: P(θ) - Belief before seeing data
- **Evidence**: P(D) - Normalizing constant

**Bayesian vs Frequentist:**
- **Frequentist**: Parameters are fixed, data is random
- **Bayesian**: Parameters are random, have distributions
- **Bayesian**: Provides uncertainty estimates

**Prior Distributions:**
- **Conjugate Priors**: Lead to closed-form posteriors
- **Non-informative**: Weak beliefs (uniform, Jeffreys)
- **Informative**: Strong beliefs from domain knowledge

**Posterior Distribution:**
- Combines prior knowledge with data
- More data → posterior closer to likelihood
- Quantifies uncertainty in parameters

**Maximum A Posteriori (MAP):**
- Point estimate: mode of posterior
- Balances likelihood and prior
- Regularization effect

**Advantages:**
- Quantifies uncertainty
- Incorporates prior knowledge
- Handles small datasets well
- Natural for sequential learning`,
					CodeExamples: `import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Bayesian inference example: Coin flip
# Prior: Beta distribution (conjugate for binomial)
alpha_prior = 2  # Prior belief
beta_prior = 2

# Data: 7 heads out of 10 flips
n_flips = 10
n_heads = 7

# Posterior: Beta(alpha + n_heads, beta + n_flips - n_heads)
alpha_posterior = alpha_prior + n_heads
beta_posterior = beta_prior + n_flips - n_heads

# Posterior distribution
theta = np.linspace(0, 1, 100)
prior_pdf = stats.beta.pdf(theta, alpha_prior, beta_prior)
posterior_pdf = stats.beta.pdf(theta, alpha_posterior, beta_posterior)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(theta, prior_pdf, label='Prior', linestyle='--')
plt.plot(theta, posterior_pdf, label='Posterior')
plt.axvline(n_heads/n_flips, color='red', linestyle=':', label='MLE')
plt.xlabel('Probability of Heads (θ)')
plt.ylabel('Density')
plt.legend()
plt.title('Bayesian Inference: Coin Flip')
plt.show()

# MAP estimate
map_estimate = (alpha_posterior - 1) / (alpha_posterior + beta_posterior - 2)
print(f"MAP estimate: {map_estimate:.3f}")

# Credible interval (95%)
lower = stats.beta.ppf(0.025, alpha_posterior, beta_posterior)
upper = stats.beta.ppf(0.975, alpha_posterior, beta_posterior)
print(f"95% Credible Interval: [{lower:.3f}, {upper:.3f}]")

# Bayesian linear regression
class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha  # Precision of prior
        self.beta = beta    # Precision of noise
    
    def fit(self, X, y):
        # Prior: N(0, alpha^-1 I)
        # Posterior: N(m_N, S_N)
        N = X.shape[0]
        S_N_inv = self.alpha * np.eye(X.shape[1]) + self.beta * X.T @ X
        self.S_N = np.linalg.inv(S_N_inv)
        self.m_N = self.beta * self.S_N @ X.T @ y
        return self
    
    def predict(self, X, return_std=False):
        y_pred = X @ self.m_N
        if return_std:
            y_std = np.sqrt(np.diag(X @ self.S_N @ X.T) + 1/self.beta)
            return y_pred, y_std
        return y_pred

# Example
X = np.random.randn(20, 1)
y = 2 * X.flatten() + 1 + np.random.randn(20) * 0.5

blr = BayesianLinearRegression(alpha=1.0, beta=2.0)
blr.fit(X, y)

X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred, y_std = blr.predict(X_test, return_std=True)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='Data')
plt.plot(X_test, y_pred, 'b-', label='Mean Prediction')
plt.fill_between(X_test.flatten(), y_pred - 2*y_std, y_pred + 2*y_std, 
                 alpha=0.3, label='95% Credible Interval')
plt.legend()
plt.title('Bayesian Linear Regression')
plt.show()`,
				},
				{
					Title: "Maximum A Posteriori (MAP) Estimation",
					Content: `MAP estimation finds the mode of the posterior distribution.

**MAP vs MLE:**
- **MLE**: Maximizes P(D|θ)
- **MAP**: Maximizes P(θ|D) = P(D|θ) × P(θ)

**MAP Objective:**
θ_MAP = argmax_θ [log P(D|θ) + log P(θ)]
- Log-likelihood + log-prior
- Prior acts as regularization

**Common Priors:**
- **Gaussian Prior**: L2 regularization (Ridge)
- **Laplace Prior**: L1 regularization (Lasso)
- **Uniform Prior**: Equivalent to MLE

**Gaussian Prior → Ridge Regression:**
- Prior: θ ~ N(0, λ⁻¹I)
- MAP: Minimizes ||y - Xθ||² + λ||θ||²

**Laplace Prior → Lasso:**
- Prior: θ ~ Laplace(0, λ⁻¹)
- MAP: Minimizes ||y - Xθ||² + λ||θ||₁

**Advantages:**
- Incorporates prior knowledge
- Regularization effect
- Single point estimate (simpler than full posterior)

**Limitations:**
- Doesn't capture uncertainty
- Mode may not be representative
- Requires choosing prior`,
					CodeExamples: `import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, Lasso

# MAP with Gaussian prior (Ridge)
def map_ridge(X, y, alpha=1.0):
    """
    MAP estimation with Gaussian prior
    Equivalent to Ridge regression
    """
    n_samples, n_features = X.shape
    
    def objective(theta):
        # Negative log-posterior
        likelihood = 0.5 * np.sum((y - X @ theta) ** 2)
        prior = 0.5 * alpha * np.sum(theta ** 2)
        return likelihood + prior
    
    theta0 = np.zeros(n_features)
    result = minimize(objective, theta0, method='BFGS')
    return result.x

# MAP with Laplace prior (Lasso)
def map_lasso(X, y, alpha=1.0):
    """
    MAP estimation with Laplace prior
    Equivalent to Lasso regression
    """
    n_samples, n_features = X.shape
    
    def objective(theta):
        # Negative log-posterior
        likelihood = 0.5 * np.sum((y - X @ theta) ** 2)
        prior = alpha * np.sum(np.abs(theta))
        return likelihood + prior
    
    theta0 = np.zeros(n_features)
    result = minimize(objective, theta0, method='L-BFGS-B')
    return result.x

# Generate data
np.random.seed(42)
X = np.random.randn(50, 10)
true_theta = np.array([1, 2, 0, 0, 3, 0, 0, 0, 0, 0])  # Sparse
y = X @ true_theta + np.random.randn(50) * 0.5

# Compare MLE, MAP Ridge, MAP Lasso
mle_theta = np.linalg.lstsq(X, y, rcond=None)[0]
map_ridge_theta = map_ridge(X, y, alpha=1.0)
map_lasso_theta = map_lasso(X, y, alpha=0.1)

# Using sklearn
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
ridge_theta = ridge.coef_

lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
lasso_theta = lasso.coef_

print("True theta:", true_theta)
print("MLE theta:", mle_theta)
print("MAP Ridge theta:", map_ridge_theta)
print("MAP Lasso theta:", map_lasso_theta)
print("Lasso (sklearn):", lasso_theta)

# Lasso recovers sparsity better`,
				},
				{
					Title: "Bayesian Neural Networks",
					Content: `Bayesian neural networks treat weights as distributions, providing uncertainty estimates.

**Key Difference:**
- **Standard NN**: Point estimates for weights
- **Bayesian NN**: Distributions over weights

**Posterior over Weights:**
P(w|D) ∝ P(D|w) × P(w)
- Prior: P(w) - distribution over weights
- Likelihood: P(D|w) - data likelihood
- Posterior: P(w|D) - updated distribution

**Inference:**
- **Variational Inference**: Approximate posterior
- **MCMC**: Sample from posterior
- **Dropout as VI**: Dropout approximates Bayesian inference

**Variational Inference:**
- Approximate posterior: q(w|θ) ≈ P(w|D)
- Minimize KL divergence: KL(q||p)
- Tractable optimization

**Dropout as Bayesian Approximation:**
- Dropout during training ≈ sampling from posterior
- Monte Carlo dropout for uncertainty
- Simple and effective

**Uncertainty Types:**
- **Epistemic**: Model uncertainty (reduced with more data)
- **Aleatoric**: Data uncertainty (inherent noise)

**Advantages:**
- Quantifies prediction uncertainty
- Better with small datasets
- Robust to overfitting

**Challenges:**
- More complex inference
- Slower training
- Approximations needed`,
					CodeExamples: `import torch
import torch.nn as nn
import torch.nn.functional as F

# Bayesian Neural Network with Variational Inference
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Variational parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.randn(out_features))
        self.bias_logvar = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        # Sample weights from variational posterior
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight_eps = torch.randn_like(weight_std)
        weight = self.weight_mu + weight_eps * weight_std
        
        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias_eps = torch.randn_like(bias_std)
        bias = self.bias_mu + bias_eps * bias_std
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        # KL divergence between q(w) and prior N(0,1)
        kl_weight = -0.5 * torch.sum(1 + self.weight_logvar - 
                                    self.weight_mu**2 - 
                                    torch.exp(self.weight_logvar))
        kl_bias = -0.5 * torch.sum(1 + self.bias_logvar - 
                                  self.bias_mu**2 - 
                                  torch.exp(self.bias_logvar))
        return kl_weight + kl_bias

class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Monte Carlo Dropout (simpler approximation)
class MCDropoutNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, training=True):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """Monte Carlo sampling for uncertainty"""
        self.train()  # Enable dropout
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, training=True)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        return mean, std

# Example usage
model = MCDropoutNN(input_dim=10, hidden_dim=50, output_dim=1)
x_test = torch.randn(1, 10)

# Get predictions with uncertainty
mean_pred, std_pred = model.predict_with_uncertainty(x_test, n_samples=100)
print(f"Prediction: {mean_pred.item():.3f} ± {std_pred.item():.3f}")`,
				},
				{
					Title: "Gaussian Processes",
					Content: `Gaussian processes provide Bayesian non-parametric approach to regression and classification.

**Gaussian Process:**
- Collection of random variables, any finite subset is Gaussian
- Defined by mean function m(x) and covariance function k(x, x')
- f(x) ~ GP(m(x), k(x, x'))

**Covariance Functions (Kernels):**
- **RBF**: k(x, x') = σ² exp(-||x - x'||² / (2l²))
- **Matern**: More flexible than RBF
- **Linear**: k(x, x') = x^T x'

**GP Regression:**
- Prior: f ~ GP(0, k)
- Likelihood: y = f(x) + ε, ε ~ N(0, σ²)
- Posterior: f|y ~ GP(μ*, k*)

**Prediction:**
- Mean: μ* = k(x*, X)[k(X, X) + σ²I]⁻¹y
- Variance: σ*² = k(x*, x*) - k(x*, X)[k(X, X) + σ²I]⁻¹k(X, x*)

**Advantages:**
- Provides uncertainty estimates
- Non-parametric (flexible)
- Handles small datasets well
- Interpretable kernels

**Limitations:**
- O(n³) complexity (cubic in data size)
- Requires choosing kernel
- Less scalable than neural networks`,
					CodeExamples: `import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
X = np.linspace(0, 10, 20).reshape(-1, 1)
y = np.sin(X).flatten() + np.random.randn(20) * 0.1

# Define kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

# GP Regression
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X, y)

# Predictions
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred, sigma = gp.predict(X_test, return_std=True)

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(X, y, c='red', label='Training Data')
plt.plot(X_test, y_pred, 'b-', label='Mean Prediction')
plt.fill_between(X_test.flatten(), y_pred - 2*sigma, y_pred + 2*sigma, 
                 alpha=0.3, label='95% Confidence Interval')
plt.legend()
plt.title('Gaussian Process Regression')
plt.show()

# Different kernels
kernels = {
    'RBF': C(1.0) * RBF(1.0),
    'Matern 3/2': C(1.0) * Matern(length_scale=1.0, nu=1.5),
    'Matern 5/2': C(1.0) * Matern(length_scale=1.0, nu=2.5)
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (name, kernel) in enumerate(kernels.items()):
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(X, y)
    y_pred, sigma = gp.predict(X_test, return_std=True)
    
    axes[idx].scatter(X, y, c='red')
    axes[idx].plot(X_test, y_pred, 'b-')
    axes[idx].fill_between(X_test.flatten(), y_pred - 2*sigma, 
                          y_pred + 2*sigma, alpha=0.3)
    axes[idx].set_title(name)
plt.tight_layout()
plt.show()`,
				},
				{
					Title: "MCMC Methods",
					Content: `Markov Chain Monte Carlo (MCMC) methods sample from posterior distributions.

**Why MCMC:**
- Posterior often intractable (no closed form)
- Need samples from posterior
- Approximate integrals via Monte Carlo

**Metropolis-Hastings Algorithm:**
1. Start with initial θ₀
2. Propose new θ' from proposal q(θ'|θ)
3. Accept with probability: min(1, P(θ'|D)q(θ|θ') / P(θ|D)q(θ'|θ))
4. Repeat

**Gibbs Sampling:**
- Special case of Metropolis-Hastings
- Sample each parameter conditional on others
- Always accepts (efficient)

**Hamiltonian Monte Carlo (HMC):**
- Uses gradient information
- More efficient exploration
- Requires gradients

**Convergence:**
- **Burn-in**: Initial samples discarded
- **Thinning**: Keep every k-th sample
- **Diagnostics**: Trace plots, R-hat statistic

**Applications:**
- Bayesian inference
- Posterior sampling
- Model comparison
- Uncertainty quantification`,
					CodeExamples: `import numpy as np
import matplotlib.pyplot as plt

# Simple Metropolis-Hastings
def metropolis_hastings(log_target, proposal, n_samples, initial):
    """
    log_target: log of target distribution
    proposal: function that proposes new sample
    """
    samples = [initial]
    current = initial
    
    for _ in range(n_samples):
        # Propose new sample
        proposed = proposal(current)
        
        # Acceptance probability
        log_alpha = log_target(proposed) - log_target(current)
        alpha = min(1, np.exp(log_alpha))
        
        # Accept or reject
        if np.random.rand() < alpha:
            current = proposed
        
        samples.append(current)
    
    return np.array(samples)

# Example: Sample from posterior
# Target: N(2, 1) (posterior)
def log_target(x):
    return -0.5 * (x - 2)**2

# Proposal: Random walk
def proposal(current):
    return current + np.random.normal(0, 0.5)

# Sample
samples = metropolis_hastings(log_target, proposal, n_samples=10000, initial=0.0)

# Discard burn-in
burn_in = 1000
samples = samples[burn_in:]

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(samples[:500])
plt.title('Trace Plot (First 500 samples)')
plt.xlabel('Iteration')
plt.ylabel('Sample Value')

plt.subplot(1, 2, 2)
plt.hist(samples, bins=50, density=True, alpha=0.7)
x_true = np.linspace(-2, 6, 100)
plt.plot(x_true, np.exp(log_target(x_true)) / np.sqrt(2*np.pi), 
         'r-', label='True Posterior')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.title('Posterior Samples')
plt.show()

print(f"Mean: {np.mean(samples):.3f} (true: 2.0)")
print(f"Std: {np.std(samples):.3f} (true: 1.0)")

# Using PyMC (if available)
try:
    import pymc3 as pm
    import theano.tensor as tt
    
    # Bayesian linear regression with MCMC
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=X.shape[1])
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Likelihood
        mu = alpha + tt.dot(X, beta)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        # Sample
        trace = pm.sample(2000, return_inferencedata=False)
    
    print("MCMC sampling completed")
    pm.traceplot(trace)
    plt.show()
except ImportError:
    print("Install PyMC3: pip install pymc3")`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
