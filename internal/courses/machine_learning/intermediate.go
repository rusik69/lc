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
	})
}
