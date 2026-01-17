package machinelearning

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMachineLearningModules([]problems.CourseModule{
		{
			ID:          80,
			Title:       "Introduction to Machine Learning",
			Description: "Learn what machine learning is, its types, applications, and how it differs from traditional programming.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "What is Machine Learning?",
					Content: `Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.

**Core Concept:**
Traditional programming: Input + Rules → Output
Machine Learning: Input + Output → Rules (learned from data)

**Key Characteristics:**
- **Data-Driven**: Learns patterns from data rather than explicit instructions
- **Adaptive**: Improves performance with more data and experience
- **Generalizable**: Can make predictions on new, unseen data
- **Automated**: Reduces need for manual feature engineering (in deep learning)

**Why Machine Learning Matters:**
- **Complex Problems**: Solves problems too complex for traditional algorithms
- **Pattern Recognition**: Identifies patterns humans might miss
- **Scalability**: Handles large-scale data processing
- **Automation**: Automates decision-making processes
- **Personalization**: Enables personalized experiences (recommendations, search)

**Real-World Applications:**
- **Healthcare**: Medical diagnosis, drug discovery, personalized treatment
- **Finance**: Fraud detection, algorithmic trading, credit scoring
- **Technology**: Search engines, recommendation systems, voice assistants
- **Transportation**: Self-driving cars, route optimization, traffic prediction
- **E-commerce**: Product recommendations, price optimization, inventory management
- **Entertainment**: Content recommendation (Netflix, Spotify), game AI

**ML vs Traditional Programming:**

Traditional Programming:
- Explicit rules defined by programmer
- Deterministic output
- Requires domain expertise to code rules
- Difficult to adapt to new scenarios

Machine Learning:
- Rules learned from data
- Probabilistic output
- Learns patterns automatically
- Adapts to new data`,
					CodeExamples: `# Traditional Programming: Rule-based email spam detection
def is_spam_traditional(email):
    spam_keywords = ['free', 'winner', 'click here', 'limited time']
    if any(keyword in email.lower() for keyword in spam_keywords):
        return True
    if '!!!' in email:
        return True
    return False

# Machine Learning: Learn from examples
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Training data
emails = [
    "Congratulations! You've won a free prize! Click here now!!!",
    "Meeting scheduled for tomorrow at 3pm",
    "Limited time offer! Buy now!!!",
    "Project update: The report is ready for review"
]
labels = [1, 0, 1, 0]  # 1 = spam, 0 = ham

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
model = MultinomialNB()
model.fit(X, labels)

# Predict on new email
new_email = "Free money! Click here!!!"
X_new = vectorizer.transform([new_email])
prediction = model.predict(X_new)
print(f"Spam: {prediction[0] == 1}")  # True`,
				},
				{
					Title: "Types of Machine Learning",
					Content: `Machine learning can be categorized into three main types based on how the learning process works:

**1. Supervised Learning**
- **Definition**: Learning with labeled training data (input-output pairs)
- **Goal**: Learn a mapping from inputs to outputs
- **Examples**: 
  - Classification: Email spam detection, image recognition
  - Regression: House price prediction, stock price forecasting
- **Key Characteristics**:
  - Requires labeled training data
  - Clear objective (predict labels/values)
  - Can evaluate performance on test data
- **Use Cases**: When you have historical data with known outcomes

**2. Unsupervised Learning**
- **Definition**: Learning patterns from unlabeled data
- **Goal**: Discover hidden patterns or structure in data
- **Examples**:
  - Clustering: Customer segmentation, anomaly detection
  - Dimensionality Reduction: Feature extraction, visualization
  - Association: Market basket analysis, recommendation systems
- **Key Characteristics**:
  - No labeled data required
  - Discovers patterns automatically
  - Harder to evaluate (no ground truth)
- **Use Cases**: When you want to explore data or don't have labels

**3. Reinforcement Learning**
- **Definition**: Learning through interaction with environment, receiving rewards/penalties
- **Goal**: Learn optimal actions to maximize cumulative reward
- **Examples**: 
  - Game playing (Chess, Go, video games)
  - Robotics (navigation, manipulation)
  - Autonomous vehicles
  - Recommendation systems (optimizing user engagement)
- **Key Characteristics**:
  - Agent learns from trial and error
  - Delayed rewards (actions affect future outcomes)
  - Exploration vs exploitation trade-off
- **Use Cases**: Sequential decision-making problems

**Other Categories:**

**Semi-Supervised Learning**: Combines labeled and unlabeled data
- Useful when labeling is expensive
- Uses small labeled dataset + large unlabeled dataset

**Self-Supervised Learning**: Creates labels from data itself
- Common in NLP and computer vision
- Example: Predict next word in sentence (label is the next word)

**Transfer Learning**: Apply knowledge from one task to another
- Pre-train on large dataset, fine-tune on specific task
- Common in deep learning (ImageNet → specific image classification)`,
					CodeExamples: `# Supervised Learning: Classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load labeled dataset
iris = load_iris()
X, y = iris.data, iris.target  # Features and labels

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict on test data
predictions = clf.predict(X_test)
print(f"Accuracy: {clf.score(X_test, y_test)}")

# Unsupervised Learning: Clustering
from sklearn.cluster import KMeans
import numpy as np

# Generate unlabeled data
X = np.random.rand(100, 2)

# Discover clusters
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X)

print(f"Found {len(np.unique(clusters))} clusters")

# Reinforcement Learning: Simple example concept
# Agent learns to maximize reward through actions
class SimpleRLAgent:
    def __init__(self):
        self.q_table = {}  # State-action values
    
    def choose_action(self, state, epsilon=0.1):
        # Epsilon-greedy: explore with probability epsilon
        if np.random.random() < epsilon:
            return np.random.choice(['action1', 'action2'])
        else:
            # Exploit: choose best known action
            return max(self.q_table.get(state, {}), 
                      key=self.q_table.get(state, {}).get)
    
    def update(self, state, action, reward, next_state):
        # Update Q-value based on reward
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        
        # Q-learning update
        self.q_table[state][action] += 0.1 * (
            reward + 0.9 * max(self.q_table.get(next_state, {}).values(), default=0) 
            - self.q_table[state][action]
        )`,
				},
				{
					Title: "ML vs Traditional Programming",
					Content: `Understanding when to use ML vs traditional programming is crucial for effective problem-solving.

**When to Use Traditional Programming:**
- **Clear Rules**: Problem has well-defined, deterministic rules
- **Small Problem Space**: Limited number of cases to handle
- **Exact Solutions**: Need precise, predictable outputs
- **Interpretability**: Need to understand exactly how solution works
- **Examples**: 
  - Calculator operations
  - Sorting algorithms
  - Database queries
  - Business logic (if-else rules)

**When to Use Machine Learning:**
- **Complex Patterns**: Patterns too complex to code explicitly
- **Large Problem Space**: Too many cases to handle manually
- **Adaptive Behavior**: Need system to adapt to new data
- **Pattern Recognition**: Identifying patterns in data
- **Examples**:
  - Image recognition (millions of pixel combinations)
  - Natural language understanding (infinite sentence variations)
  - Recommendation systems (complex user preferences)
  - Fraud detection (evolving fraud patterns)

**Hybrid Approaches:**
Many real-world systems combine both:
- ML for pattern recognition
- Traditional programming for business rules and logic
- Example: E-commerce site uses ML for recommendations but traditional code for checkout process

**Trade-offs:**

**Traditional Programming:**
- ✅ Predictable and deterministic
- ✅ Interpretable and debuggable
- ✅ No training data needed
- ❌ Requires domain expertise
- ❌ Hard to adapt to new scenarios
- ❌ Limited by programmer's knowledge

**Machine Learning:**
- ✅ Learns complex patterns automatically
- ✅ Adapts to new data
- ✅ Can improve over time
- ❌ Requires training data
- ❌ Less interpretable (black box)
- ❌ Can make mistakes
- ❌ Needs careful validation`,
					CodeExamples: `# Traditional Programming: Clear rules
def calculate_tax(income, brackets):
    """Calculate tax using explicit rules."""
    tax = 0
    for bracket in brackets:
        if income > bracket['min']:
            taxable = min(income - bracket['min'], 
                         bracket['max'] - bracket['min'])
            tax += taxable * bracket['rate']
    return tax

# Machine Learning: Learn from data
from sklearn.linear_model import LinearRegression
import pandas as pd

# Historical data: features → house price
data = pd.DataFrame({
    'size': [1000, 1500, 2000, 1200, 1800],
    'bedrooms': [2, 3, 4, 2, 3],
    'age': [5, 10, 2, 15, 8],
    'price': [200000, 300000, 450000, 180000, 350000]
})

# Learn relationship from data
X = data[['size', 'bedrooms', 'age']]
y = data['price']
model = LinearRegression()
model.fit(X, y)

# Predict on new house
new_house = [[1600, 3, 7]]
predicted_price = model.predict(new_house)
print(f"Predicted price: ${predicted_price[0]:,.0f}")

# Hybrid: ML + Rules
def recommend_product(user_id, products, ml_model, business_rules):
    # ML: Get personalized recommendations
    ml_scores = ml_model.predict(user_id, products)
    
    # Rules: Apply business constraints
    filtered = []
    for product, score in zip(products, ml_scores):
        # Rule: Don't recommend out-of-stock items
        if product['in_stock']:
            # Rule: Boost premium products
            if product['premium']:
                score *= 1.2
            filtered.append((product, score))
    
    # Sort by final score
    return sorted(filtered, key=lambda x: x[1], reverse=True)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          81,
			Title:       "Data Preprocessing & Feature Engineering",
			Description: "Master data cleaning, handling missing values, feature scaling, encoding, and feature selection techniques.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Data Cleaning Fundamentals",
					Content: `Data cleaning is the process of detecting and correcting (or removing) corrupt, inaccurate, or irrelevant data. It's often the most time-consuming part of a machine learning project but critical for model performance.

**Common Data Quality Issues:**
- **Missing Values**: Null, NaN, empty strings
- **Duplicates**: Identical or near-duplicate records
- **Outliers**: Extreme values that may be errors
- **Inconsistencies**: Different formats, units, or encodings
- **Errors**: Typos, invalid entries, wrong data types

**Data Cleaning Steps:**

1. **Inspect Data**
   - Load and examine first few rows
   - Check data types
   - Identify missing values
   - Look for obvious errors

2. **Handle Missing Values**
   - Remove rows/columns (if appropriate)
   - Impute with mean/median/mode
   - Use forward/backward fill for time series
   - Create "missing" indicator feature

3. **Remove Duplicates**
   - Identify exact duplicates
   - Handle near-duplicates (fuzzy matching)

4. **Fix Data Types**
   - Convert strings to numbers
   - Parse dates correctly
   - Ensure categorical variables are strings

5. **Standardize Formats**
   - Consistent date formats
   - Consistent text casing
   - Standardize units (kg vs lbs)

**Why Data Quality Matters:**
- Garbage in, garbage out (GIGO)
- Poor data leads to poor models
- Clean data improves model accuracy
- Reduces training time
- Makes models more interpretable`,
					CodeExamples: `import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# 1. Inspect data
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 2. Handle missing values
# Option A: Remove rows with missing values
df_clean = df.dropna()

# Option B: Fill with mean (for numerical)
df['age'].fillna(df['age'].mean(), inplace=True)

# Option C: Fill with mode (for categorical)
df['category'].fillna(df['category'].mode()[0], inplace=True)

# Option D: Forward fill (for time series)
df['value'].fillna(method='ffill', inplace=True)

# Option E: Create missing indicator
df['age_missing'] = df['age'].isnull().astype(int)
df['age'].fillna(df['age'].mean(), inplace=True)

# 3. Remove duplicates
df = df.drop_duplicates()

# 4. Fix data types
df['date'] = pd.to_datetime(df['date'])
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# 5. Standardize formats
df['name'] = df['name'].str.title()  # Title case
df['email'] = df['email'].str.lower()  # Lowercase

# Remove leading/trailing whitespace
df['text'] = df['text'].str.strip()

# Handle outliers (using IQR method)
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]`,
				},
				{
					Title: "Feature Scaling",
					Content: `Feature scaling is crucial when features have different scales. Many ML algorithms are sensitive to feature scale.

**Why Scaling Matters:**
- **Distance-based algorithms** (KNN, K-Means) are dominated by features with larger scales
- **Gradient descent** converges faster with scaled features
- **Regularization** (L1/L2) penalizes features differently based on scale
- **Some algorithms** (SVM, Neural Networks) require scaling

**Common Scaling Methods:**

**1. Standardization (Z-score normalization)**
- Formula: (x - μ) / σ
- Mean = 0, Std = 1
- Preserves outliers
- Best for: Most ML algorithms

**2. Min-Max Scaling**
- Formula: (x - min) / (max - min)
- Range: [0, 1]
- Sensitive to outliers
- Best for: Neural networks, when you need bounded range

**3. Robust Scaling**
- Uses median and IQR instead of mean/std
- Less sensitive to outliers
- Best for: Data with outliers

**4. Normalization (L2)**
- Scales each sample to unit norm
- Best for: Text classification, when sample norm matters

**When to Scale:**
- ✅ Linear Regression, Logistic Regression
- ✅ SVM, KNN, K-Means
- ✅ Neural Networks
- ✅ PCA, LDA

**When NOT to Scale:**
- ❌ Tree-based algorithms (Decision Trees, Random Forest, XGBoost)
- ❌ Algorithms that use feature splits (scale doesn't matter)`,
					CodeExamples: `from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

# Sample data with different scales
X = np.array([
    [1000, 2.5, 30],
    [2000, 3.0, 25],
    [1500, 2.8, 28]
])

# 1. Standardization (Z-score)
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)
print("Standardized:")
print(X_std)
print(f"Mean: {X_std.mean(axis=0)}")
print(f"Std: {X_std.std(axis=0)}")

# 2. Min-Max Scaling
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)
print("\nMin-Max Scaled:")
print(X_minmax)
print(f"Min: {X_minmax.min(axis=0)}")
print(f"Max: {X_minmax.max(axis=0)}")

# 3. Robust Scaling (uses median and IQR)
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X)
print("\nRobust Scaled:")
print(X_robust)

# Real-world example with pandas
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    'income': [50000, 75000, 100000, 60000],
    'age': [25, 30, 35, 28],
    'score': [0.8, 0.9, 0.7, 0.85]
})

# Split data
X_train, X_test = train_test_split(df, test_size=0.25, random_state=42)

# Fit scaler on training data only!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler

# Important: Always fit on train, transform on test
# Prevents data leakage`,
				},
				{
					Title: "Encoding Categorical Variables",
					Content: `Most ML algorithms require numerical input, so categorical variables must be encoded.

**Types of Categorical Variables:**

**1. Nominal**: No order (colors, countries, categories)
- Examples: Red/Blue/Green, USA/UK/France
- Encoding: One-Hot Encoding, Target Encoding

**2. Ordinal**: Has order (ratings, sizes, education levels)
- Examples: Small/Medium/Large, Low/Medium/High
- Encoding: Ordinal Encoding, Label Encoding

**Encoding Methods:**

**1. One-Hot Encoding**
- Creates binary columns for each category
- Best for: Nominal variables with few categories
- Pros: No ordinal relationship assumed
- Cons: Creates many columns (curse of dimensionality)

**2. Label Encoding**
- Assigns integer to each category
- Best for: Ordinal variables or tree-based algorithms
- Pros: Keeps dimensionality low
- Cons: Assumes ordinal relationship

**3. Target Encoding (Mean Encoding)**
- Replaces category with mean target value
- Best for: High cardinality categorical variables
- Pros: Captures relationship with target
- Cons: Can cause overfitting (use cross-validation)

**4. Frequency Encoding**
- Replaces category with its frequency
- Best for: High cardinality variables
- Pros: Simple, captures category importance
- Cons: Loses category identity

**Best Practices:**
- Handle high cardinality (many categories) carefully
- Use cross-validation for target encoding
- Consider embedding for very high cardinality (deep learning)`,
					CodeExamples: `import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce

# Sample data
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['S', 'M', 'L', 'M', 'L'],
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
    'target': [1, 0, 1, 0, 1]
})

# 1. One-Hot Encoding (for nominal)
df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')
print("One-Hot Encoded:")
print(df_encoded[['color_blue', 'color_green', 'color_red']])

# Using sklearn
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False, drop='first')  # drop first to avoid multicollinearity
color_encoded = ohe.fit_transform(df[['color']])
print("\nOne-Hot (sklearn):")
print(color_encoded)

# 2. Label Encoding (for ordinal)
size_mapping = {'S': 0, 'M': 1, 'L': 2}
df['size_encoded'] = df['size'].map(size_mapping)

# Or using sklearn
le = LabelEncoder()
df['size_label'] = le.fit_transform(df['size'])

# 3. Target Encoding (mean encoding)
target_encoder = ce.TargetEncoder(cols=['city'])
df_target_encoded = target_encoder.fit_transform(df[['city']], df['target'])
print("\nTarget Encoded:")
print(df_target_encoded)

# 4. Frequency Encoding
city_freq = df['city'].value_counts().to_dict()
df['city_freq'] = df['city'].map(city_freq)
print("\nFrequency Encoded:")
print(df[['city', 'city_freq']])

# Handling high cardinality
# Example: 1000+ unique cities
# Option 1: Group rare categories
city_counts = df['city'].value_counts()
rare_cities = city_counts[city_counts < 2].index
df['city'] = df['city'].replace(rare_cities, 'Other')

# Option 2: Use target encoding with regularization
target_encoder_reg = ce.TargetEncoder(cols=['city'], smoothing=1.0)
df_reg = target_encoder_reg.fit_transform(df[['city']], df['target'])`,
				},
				{
					Title: "Feature Selection",
					Content: `Feature selection improves model performance by removing irrelevant or redundant features.

**Benefits:**
- **Reduces Overfitting**: Fewer features = simpler model
- **Faster Training**: Less data to process
- **Better Interpretability**: Focus on important features
- **Lower Storage**: Less data to store and process

**Feature Selection Methods:**

**1. Filter Methods**
- Select features based on statistical measures
- Independent of ML algorithm
- Fast but may miss feature interactions
- Examples: Correlation, Chi-square, Mutual Information

**2. Wrapper Methods**
- Use ML algorithm to evaluate feature subsets
- More accurate but computationally expensive
- Examples: Forward Selection, Backward Elimination, Recursive Feature Elimination

**3. Embedded Methods**
- Feature selection built into algorithm
- Examples: Lasso regularization, Tree-based feature importance

**Common Techniques:**

**Correlation-based:**
- Remove highly correlated features (redundancy)
- Keep features correlated with target

**Variance-based:**
- Remove low-variance features (little information)

**Univariate Selection:**
- Select K best features based on statistical test

**Recursive Feature Elimination (RFE):**
- Recursively remove worst features
- Uses model to evaluate feature importance`,
					CodeExamples: `import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV

# Sample data
X = pd.DataFrame(np.random.randn(100, 20))
y = np.random.randint(0, 2, 100)

# 1. Univariate Selection (Filter Method)
selector_kbest = SelectKBest(score_func=f_classif, k=10)
X_selected = selector_kbest.fit_transform(X, y)
selected_features = selector_kbest.get_support(indices=True)
print(f"Selected {len(selected_features)} features: {selected_features}")

# 2. Mutual Information (Filter Method)
selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
X_mi = selector_mi.fit_transform(X, y)

# 3. Recursive Feature Elimination (Wrapper Method)
model = RandomForestClassifier(n_estimators=100)
rfe = RFE(estimator=model, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)
print(f"RFE selected features: {rfe.get_support(indices=True)}")

# 4. Lasso-based Selection (Embedded Method)
lasso = LassoCV()
lasso.fit(X, y)
selector_lasso = SelectFromModel(lasso, prefit=True, threshold='median')
X_lasso = selector_lasso.transform(X)
print(f"Lasso selected {X_lasso.shape[1]} features")

# 5. Tree-based Feature Importance (Embedded Method)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]
print(f"Top 10 features by importance: {indices}")

# Remove highly correlated features
correlation_matrix = X.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
high_corr_features = [col for col in upper_triangle.columns 
                      if any(upper_triangle[col] > 0.95)]
X_low_corr = X.drop(columns=high_corr_features)
print(f"Removed {len(high_corr_features)} highly correlated features")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          82,
			Title:       "Supervised Learning Fundamentals",
			Description: "Learn linear regression, logistic regression, decision trees, and basic model evaluation.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "Linear Regression",
					Content: `Linear Regression is the foundation of supervised learning, predicting a continuous target variable using a linear relationship with features.

**Simple Linear Regression:**
- One feature (X) → one target (y)
- Formula: y = β₀ + β₁x + ε
- β₀: intercept (y when x=0)
- β₁: slope (change in y per unit change in x)
- ε: error term

**Multiple Linear Regression:**
- Multiple features → one target
- Formula: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
- Each feature has its own coefficient

**How It Works:**
1. **Training**: Find coefficients that minimize prediction error
2. **Cost Function**: Mean Squared Error (MSE)
   - MSE = (1/n) Σ(yᵢ - ŷᵢ)²
3. **Optimization**: Gradient Descent or Normal Equation
4. **Prediction**: Use learned coefficients to predict new values

**Assumptions:**
- Linear relationship between features and target
- Features are independent (no multicollinearity)
- Errors are normally distributed
- Homoscedasticity (constant variance)

**Advantages:**
- Simple and interpretable
- Fast training and prediction
- No hyperparameters to tune
- Works well when relationship is linear

**Limitations:**
- Assumes linear relationship
- Sensitive to outliers
- Can't capture non-linear patterns
- Requires feature scaling`,
					CodeExamples: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.flatten() + 1.0 + np.random.randn(100) * 2

# Train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Multiple Linear Regression
from sklearn.datasets import make_regression

X_multi, y_multi = make_regression(n_samples=100, n_features=3, noise=10)
model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

print(f"\nMultiple Regression Coefficients: {model_multi.coef_}")
print(f"Intercept: {model_multi.intercept_:.2f}")

# Predict on new data
X_new = np.array([[5.0]])
prediction = model.predict(X_new)
print(f"\nPrediction for X=5.0: {prediction[0]:.2f}")

# Manual implementation (for understanding)
def simple_linear_regression(X, y):
    """Simple linear regression using normal equation."""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    return coefficients[0], coefficients[1:]  # intercept, slope

intercept, slope = simple_linear_regression(X.flatten(), y)
print(f"\nManual calculation - Intercept: {intercept:.2f}, Slope: {slope[0]:.2f}")`,
				},
				{
					Title: "Logistic Regression",
					Content: `Logistic Regression is used for binary classification, predicting probabilities using a logistic (sigmoid) function.

**Key Concept:**
- Outputs probability (0 to 1) instead of continuous value
- Uses sigmoid function to map linear combination to probability
- Formula: P(y=1) = 1 / (1 + e^(-z)) where z = β₀ + β₁x₁ + ...

**Sigmoid Function:**
- S-shaped curve mapping any value to [0, 1]
- Smooth, differentiable (important for optimization)
- At z=0, probability = 0.5 (decision boundary)

**How It Works:**
1. **Linear Combination**: z = β₀ + β₁x₁ + ...
2. **Sigmoid Transform**: p = sigmoid(z) = 1/(1+e^(-z))
3. **Prediction**: Class 1 if p > 0.5, else Class 0
4. **Training**: Maximize likelihood (or minimize log loss)

**Cost Function:**
- Log Loss (Binary Cross-Entropy)
- Penalizes confident wrong predictions heavily
- Formula: -[y·log(p) + (1-y)·log(1-p)]

**Multiclass Extension:**
- **One-vs-Rest (OvR)**: Train one classifier per class
- **Multinomial**: Single model with softmax activation
- **Softmax**: Converts logits to probabilities summing to 1

**Advantages:**
- Probabilistic output (not just class)
- Fast and interpretable
- No hyperparameters (basic version)
- Works well for linearly separable data

**Limitations:**
- Assumes linear decision boundary
- Requires feature scaling
- Can't capture complex non-linear patterns`,
					CodeExamples: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification

# Generate binary classification data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=42)

# Train logistic regression
model = LogisticRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)
y_proba = model.predict_proba(X)

print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
print(f"\nClassification Report:")
print(classification_report(y, y_pred))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

# Probability predictions
print(f"\nFirst 5 probability predictions:")
print(y_proba[:5])

# Decision boundary visualization
def plot_decision_boundary(X, y, model):
    import matplotlib.pyplot as plt
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Logistic Regression Decision Boundary")
    plt.show()

# Multiclass Logistic Regression
from sklearn.datasets import load_iris

iris = load_iris()
X_multi, y_multi = iris.data, iris.target

model_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model_multi.fit(X_multi, y_multi)

y_pred_multi = model_multi.predict(X_multi)
print(f"\nMulticlass Accuracy: {accuracy_score(y_multi, y_pred_multi):.2f}")

# Manual sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_values = np.linspace(-10, 10, 100)
probabilities = sigmoid(z_values)
print(f"\nSigmoid at z=0: {sigmoid(0):.2f}")  # Should be 0.5
print(f"Sigmoid at z=5: {sigmoid(5):.2f}")  # Close to 1`,
				},
				{
					Title: "Decision Trees",
					Content: `Decision Trees are intuitive, tree-based models that make decisions by asking a series of yes/no questions.

**How Decision Trees Work:**
1. **Root Node**: Start with all data
2. **Split**: Choose feature and threshold that best separates classes
3. **Recurse**: Repeat for each subset
4. **Leaf Nodes**: Final predictions (class or value)

**Splitting Criteria:**

**For Classification:**
- **Gini Impurity**: Measures probability of misclassification
  - Gini = 1 - Σ(pᵢ)² where pᵢ is proportion of class i
  - Lower is better (0 = pure node)
- **Entropy**: Measures disorder/uncertainty
  - Entropy = -Σ(pᵢ·log₂(pᵢ))
  - Lower is better (0 = pure node)
- **Information Gain**: Reduction in entropy after split
  - Choose split with highest information gain

**For Regression:**
- **MSE (Mean Squared Error)**: Minimize variance in leaf nodes
- **MAE (Mean Absolute Error)**: Alternative splitting criterion

**Tree Construction:**
1. Start with root (all data)
2. For each feature, find best split threshold
3. Choose feature/threshold with best criterion score
4. Split data and recurse on subsets
5. Stop when:
   - Node is pure (all same class)
   - Maximum depth reached
   - Minimum samples per leaf reached
   - No improvement possible

**Advantages:**
- Highly interpretable (visual tree structure)
- No feature scaling needed
- Handles non-linear relationships
- Handles mixed data types
- Feature importance available

**Limitations:**
- Prone to overfitting
- Unstable (small data changes → different tree)
- Greedy algorithm (may miss global optimum)
- Biased toward features with more levels`,
					CodeExamples: `from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Classification Tree
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X_train, y_train)

# Predictions
y_pred = tree_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Feature importance
print(f"\nFeature Importances:")
for i, importance in enumerate(tree_clf.feature_importances_):
    print(f"{iris.feature_names[i]}: {importance:.3f}")

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(tree_clf, feature_names=iris.feature_names, 
          class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Regression Tree
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
tree_reg = DecisionTreeRegressor(max_depth=5)
tree_reg.fit(X_reg, y_reg)

y_pred_reg = tree_reg.predict(X_reg)
print(f"\nRegression Tree R²: {tree_reg.score(X_reg, y_reg):.2f}")

# Understanding splits
print(f"\nTree Depth: {tree_clf.get_depth()}")
print(f"Number of Leaves: {tree_clf.get_n_leaves()}")

# Manual Gini calculation
def gini_impurity(y):
    """Calculate Gini impurity for a node."""
    if len(y) == 0:
        return 0
    counts = np.bincount(y)
    proportions = counts / len(y)
    return 1 - np.sum(proportions ** 2)

# Example: Pure node
pure_node = np.array([0, 0, 0, 0])
print(f"Gini (pure): {gini_impurity(pure_node):.2f}")  # 0.0

# Example: Mixed node
mixed_node = np.array([0, 0, 1, 1])
print(f"Gini (mixed): {gini_impurity(mixed_node):.2f}")  # 0.5`,
				},
				{
					Title: "Model Evaluation Basics",
					Content: `Evaluating model performance is crucial to understand how well your model will work on new, unseen data.

**Train-Test Split:**
- **Training Set**: Used to train the model (typically 70-80%)
- **Test Set**: Used to evaluate final performance (typically 20-30%)
- **Important**: Never evaluate on training data (overfitting risk)

**Cross-Validation:**
- K-Fold CV: Split data into K folds, train on K-1, test on 1, repeat K times
- More robust than single train-test split
- Reduces variance in performance estimates
- Common: 5-fold or 10-fold CV

**Classification Metrics:**

**Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- Overall correctness
- Can be misleading with imbalanced classes

**Precision**: TP / (TP + FP)
- Of positive predictions, how many are correct?
- Important when false positives are costly

**Recall (Sensitivity)**: TP / (TP + FN)
- Of actual positives, how many did we find?
- Important when false negatives are costly

**F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean of precision and recall
- Balances both metrics

**Confusion Matrix:**
- Visual representation of predictions vs actual
- Shows TP, TN, FP, FN
- Helps understand error types

**Regression Metrics:**

**MSE (Mean Squared Error)**: Average squared differences
- Penalizes large errors more
- Same units as target squared

**RMSE (Root Mean Squared Error)**: √MSE
- Same units as target
- More interpretable than MSE

**MAE (Mean Absolute Error)**: Average absolute differences
- Less sensitive to outliers than MSE
- Same units as target

**R² Score**: Proportion of variance explained
- Range: (-∞, 1], 1 = perfect, 0 = baseline
- Can be negative (worse than baseline)`,
					CodeExamples: `from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_regression
import numpy as np

# Classification Evaluation
X_clf, y_clf = make_classification(n_samples=1000, n_features=20, 
                                   n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Basic metrics
print("Classification Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)
print(f"TP: {cm[1,1]}, TN: {cm[0,0]}, FP: {cm[0,1]}, FN: {cm[1,0]}")

# Detailed report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-Validation
cv_scores = cross_val_score(model, X_clf, y_clf, cv=5, scoring='accuracy')
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Regression Evaluation
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)

print(f"\nRegression Metrics:")
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_reg):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.2f}")
print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.2f}")
print(f"R² Score: {r2_score(y_test_reg, y_pred_reg):.3f}")

# Cross-Validation for Regression
cv_r2 = cross_val_score(reg_model, X_reg, y_reg, cv=5, scoring='r2')
print(f"\n5-Fold CV R²: {cv_r2.mean():.3f} (+/- {cv_r2.std() * 2:.3f})")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          83,
			Title:       "Classification Algorithms",
			Description: "Explore K-Nearest Neighbors, Naive Bayes, Support Vector Machines, and classification evaluation metrics.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "K-Nearest Neighbors (KNN)",
					Content: `K-Nearest Neighbors is a simple, instance-based learning algorithm that makes predictions based on similarity to training examples.

**How KNN Works:**
1. **Training**: Store all training examples (lazy learning - no explicit training)
2. **Prediction**: 
   - Find K nearest neighbors (by distance)
   - For classification: Majority vote of K neighbors
   - For regression: Average of K neighbors

**Distance Metrics:**
- **Euclidean**: √Σ(xᵢ - yᵢ)² (straight-line distance)
- **Manhattan**: Σ|xᵢ - yᵢ| (city-block distance)
- **Hamming**: For categorical data (number of differing positions)
- **Cosine**: For text/data with high dimensionality

**Choosing K:**
- **Small K (K=1)**: 
  - Very sensitive to noise
  - Complex decision boundary
  - High variance, low bias
- **Large K**: 
  - Smoother decision boundary
  - Less sensitive to noise
  - Lower variance, higher bias
- **Rule of thumb**: K = √n (n = number of samples)
- **Odd K**: Prevents ties in binary classification

**Advantages:**
- Simple and intuitive
- No assumptions about data distribution
- Works for both classification and regression
- Can learn complex decision boundaries
- No training time (lazy learning)

**Limitations:**
- Slow prediction (must compute distances to all points)
- Sensitive to irrelevant features
- Requires feature scaling
- Memory intensive (stores all training data)
- Sensitive to curse of dimensionality`,
					CodeExamples: `from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Classification with KNN
X, y = make_classification(n_samples=200, n_features=2, n_classes=3, 
                         n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Important: Scale features for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different K values
for k in [1, 3, 5, 10, 20]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"K={k}: Accuracy = {acc:.3f}")

# Best K (using cross-validation)
from sklearn.model_selection import cross_val_score
k_values = range(1, 21)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
print(f"\nBest K (by CV): {best_k}")

# Final model with best K
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
print(f"Final Accuracy: {accuracy_score(y_test, knn_best.predict(X_test_scaled)):.3f}")

# Regression with KNN
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_reg, y_reg)
print(f"\nKNN Regression R²: {knn_reg.score(X_reg, y_reg):.3f}")

# Distance metrics
knn_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_euclidean.fit(X_train_scaled, y_train)
knn_manhattan.fit(X_train_scaled, y_train)
print(f"\nEuclidean Accuracy: {accuracy_score(y_test, knn_euclidean.predict(X_test_scaled)):.3f}")
print(f"Manhattan Accuracy: {accuracy_score(y_test, knn_manhattan.predict(X_test_scaled)):.3f}")`,
				},
				{
					Title: "Naive Bayes",
					Content: `Naive Bayes is a probabilistic classifier based on Bayes' theorem with a "naive" assumption of feature independence.

**Bayes' Theorem:**
P(y|X) = P(X|y) × P(y) / P(X)
- P(y|X): Posterior probability (class given features)
- P(X|y): Likelihood (features given class)
- P(y): Prior probability (class probability)
- P(X): Evidence (normalizing constant)

**Naive Assumption:**
- Features are conditionally independent given the class
- P(x₁, x₂, ..., xₙ|y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
- Simplifies calculation but rarely true in practice
- Still works surprisingly well!

**Types of Naive Bayes:**

**1. Gaussian Naive Bayes:**
- Assumes features follow Gaussian (normal) distribution
- For continuous/numerical features
- Estimates mean and variance for each class

**2. Multinomial Naive Bayes:**
- For count data (word counts, frequencies)
- Uses multinomial distribution
- Common in text classification

**3. Bernoulli Naive Bayes:**
- For binary features (present/absent)
- Uses Bernoulli distribution
- Also common in text (word presence)

**Advantages:**
- Fast training and prediction
- Works well with small datasets
- Handles multiple classes naturally
- Not sensitive to irrelevant features
- Probabilistic output

**Limitations:**
- Naive independence assumption (often violated)
- Requires feature independence
- Can be outperformed by more complex models`,
					CodeExamples: `from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Gaussian Naive Bayes (for continuous features)
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(f"Gaussian NB Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Probability predictions
y_proba = gnb.predict_proba(X_test)
print(f"\nFirst prediction probabilities: {y_proba[0]}")

# Multinomial Naive Bayes (for text/counts)
# Example: Text classification
documents = [
    "I love machine learning",
    "Python is great for data science",
    "Machine learning is fascinating",
    "Data science requires Python skills",
    "I hate this boring lecture",
    "This is terrible and awful"
]
labels = [1, 1, 1, 1, 0, 0]  # 1 = positive, 0 = negative

# Convert text to count vectors
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(documents)

mnb = MultinomialNB()
mnb.fit(X_text, labels)

# Predict on new document
new_doc = ["I love Python and machine learning"]
X_new = vectorizer.transform(new_doc)
prediction = mnb.predict(X_new)
probabilities = mnb.predict_proba(X_new)
print(f"\nText: '{new_doc[0]}'")
print(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
print(f"Probabilities: {probabilities[0]}")

# Bernoulli Naive Bayes (for binary features)
# Example: Binary word presence
X_binary = (X_text > 0).astype(int)  # Convert to binary

bnb = BernoulliNB()
bnb.fit(X_binary, labels)
y_pred_bnb = bnb.predict(X_binary)
print(f"\nBernoulli NB Accuracy: {accuracy_score(labels, y_pred_bnb):.3f}")

# Understanding the algorithm
print(f"\nGaussian NB Parameters:")
for i, class_name in enumerate(iris.target_names):
    print(f"Class {class_name}:")
    print(f"  Mean: {gnb.theta_[i]}")
    print(f"  Variance: {gnb.sigma_[i]}")`,
				},
				{
					Title: "Support Vector Machines (SVM)",
					Content: `Support Vector Machines find the optimal hyperplane that best separates classes with maximum margin.

**Key Concepts:**

**Hyperplane:**
- Decision boundary separating classes
- In 2D: a line, in 3D: a plane, in nD: hyperplane
- Formula: w·x + b = 0

**Support Vectors:**
- Data points closest to the hyperplane
- Define the margin
- Only these points matter (sparse solution)

**Margin:**
- Distance between hyperplane and nearest points
- Larger margin = better generalization
- SVM maximizes this margin

**How SVM Works:**
1. Find hyperplane with maximum margin
2. Support vectors are the "hard" examples
3. All other points don't affect the solution
4. Prediction: Which side of hyperplane is point on?

**Kernel Trick:**
- Maps data to higher-dimensional space
- Finds linear separation in that space
- Common kernels:
  - **Linear**: No transformation
  - **Polynomial**: (γ·xᵢ·xⱼ + r)ᵈ
  - **RBF (Radial Basis Function)**: exp(-γ||xᵢ - xⱼ||²)
  - **Sigmoid**: tanh(γ·xᵢ·xⱼ + r)

**Soft Margin (C parameter):**
- Allows some misclassification
- C: Controls trade-off between margin and errors
- Large C: Hard margin (fewer errors, smaller margin)
- Small C: Soft margin (more errors, larger margin)

**Advantages:**
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors only)
- Versatile (different kernels)
- Works well with clear margin of separation

**Limitations:**
- Doesn't perform well with large datasets
- Sensitive to feature scaling
- Doesn't provide probability estimates (need calibration)
- Black box (hard to interpret)`,
					CodeExamples: `from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Generate non-linearly separable data
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear SVM (won't work well for circles)
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)
print(f"Linear SVM Accuracy: {accuracy_score(y_test, svm_linear.predict(X_test_scaled)):.3f}")

# RBF Kernel (handles non-linear data)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)
print(f"RBF SVM Accuracy: {accuracy_score(y_test, svm_rbf.predict(X_test_scaled)):.3f}")

# Polynomial Kernel
svm_poly = SVC(kernel='poly', degree=3, C=1.0)
svm_poly.fit(X_train_scaled, y_train)
print(f"Polynomial SVM Accuracy: {accuracy_score(y_test, svm_poly.predict(X_test_scaled)):.3f}")

# Tuning C parameter
C_values = [0.1, 1, 10, 100]
for C in C_values:
    svm = SVC(kernel='rbf', C=C, gamma='scale')
    svm.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, svm.predict(X_test_scaled))
    print(f"C={C}: Accuracy = {acc:.3f}")

# Support vectors
print(f"\nNumber of support vectors: {len(svm_rbf.support_vectors_)}")
print(f"Support vector indices: {svm_rbf.support_[:10]}")  # First 10

# SVM for Regression
from sklearn.datasets import make_regression
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
svr = SVR(kernel='rbf', C=100, epsilon=0.1)
svr.fit(X_reg, y_reg)
print(f"\nSVM Regression R²: {svr.score(X_reg, y_reg):.3f}")

# Visualizing decision boundary (concept)
def plot_svm_boundary(X, y, svm_model):
    """Conceptual visualization of SVM decision boundary."""
    # Create grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on grid
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot (would use matplotlib in real code)
    print("Decision boundary plotted (conceptual)")`,
				},
				{
					Title: "Classification Evaluation Metrics",
					Content: `Understanding different evaluation metrics helps choose the right metric for your problem.

**Confusion Matrix Components:**
- **TP (True Positive)**: Correctly predicted positive
- **TN (True Negative)**: Correctly predicted negative
- **FP (False Positive)**: Incorrectly predicted positive (Type I error)
- **FN (False Negative)**: Incorrectly predicted negative (Type II error)

**Key Metrics:**

**Accuracy**: (TP + TN) / Total
- Overall correctness
- **Use when**: Balanced classes, equal cost of errors
- **Problem**: Misleading with imbalanced data

**Precision**: TP / (TP + FP)
- Of positive predictions, how many are correct?
- **Use when**: False positives are costly
- **Example**: Spam detection (don't want to mark real emails as spam)

**Recall (Sensitivity)**: TP / (TP + FN)
- Of actual positives, how many did we catch?
- **Use when**: False negatives are costly
- **Example**: Disease detection (don't want to miss sick patients)

**Specificity**: TN / (TN + FP)
- Of actual negatives, how many did we correctly identify?
- Complement of false positive rate

**F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean of precision and recall
- **Use when**: Need balance between precision and recall
- Range: [0, 1], higher is better

**Fβ Score**: (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
- Weighted F1 score
- β > 1: Emphasize recall
- β < 1: Emphasize precision

**ROC Curve & AUC:**
- **ROC Curve**: Plots TPR vs FPR at different thresholds
- **AUC (Area Under Curve)**: Overall performance measure
- **Use when**: Need to compare models, threshold-independent
- Range: [0, 1], 0.5 = random, 1.0 = perfect

**Precision-Recall Curve:**
- Plots precision vs recall at different thresholds
- **Use when**: Imbalanced classes (better than ROC)
- **AUC-PR**: Area under precision-recall curve`,
					CodeExamples: `from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2,
                          weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Basic metrics
print("Classification Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

# Specificity
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
print(f"Specificity: {specificity:.3f}")

# Detailed report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nROC AUC: {roc_auc:.3f}")

# Precision-Recall Curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
print(f"PR AUC: {pr_auc:.3f}")

# Finding optimal threshold (F1 maximization)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = pr_thresholds[optimal_idx]
print(f"\nOptimal Threshold (F1): {optimal_threshold:.3f}")
print(f"Optimal F1: {f1_scores[optimal_idx]:.3f}")

# Predictions with optimal threshold
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
print(f"\nWith Optimal Threshold:")
print(f"Precision: {precision_score(y_test, y_pred_optimal):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_optimal):.3f}")
print(f"F1: {f1_score(y_test, y_pred_optimal):.3f}")

# Per-class metrics (multiclass)
from sklearn.datasets import load_iris
iris = load_iris()
X_multi, y_multi = iris.data, iris.target
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42
)

model_multi = LogisticRegression(multi_class='multinomial')
model_multi.fit(X_train_multi, y_train_multi)
y_pred_multi = model_multi.predict(X_test_multi)

print(f"\nMulticlass Metrics:")
print(classification_report(y_test_multi, y_pred_multi, 
                          target_names=iris.target_names))`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          84,
			Title:       "Regression & Model Evaluation",
			Description: "Learn polynomial regression, regularization (Ridge & Lasso), cross-validation, and overfitting concepts.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Polynomial Regression",
					Content: `Polynomial Regression extends linear regression to capture non-linear relationships by adding polynomial features.

**Concept:**
- Transform features to polynomial features
- Linear regression on transformed features
- Can model curves, not just straight lines

**Polynomial Features:**
- Original: x
- Degree 2: x, x²
- Degree 3: x, x², x³
- Multiple features: x₁, x₂, x₁², x₂², x₁x₂

**Formula:**
y = β₀ + β₁x + β₂x² + ... + βₙxⁿ + ε

**When to Use:**
- Relationship is non-linear but smooth
- You have domain knowledge suggesting polynomial relationship
- Need more flexibility than linear regression

**Advantages:**
- Captures non-linear patterns
- Still uses linear regression techniques
- Interpretable (can see which polynomial terms matter)

**Limitations:**
- Prone to overfitting (especially high degree)
- Can create unrealistic curves outside training range
- Requires careful degree selection
- Feature scaling becomes more important

**Degree Selection:**
- Too low: Underfitting (can't capture pattern)
- Too high: Overfitting (fits noise)
- Use cross-validation to find optimal degree`,
					CodeExamples: `import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate non-linear data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 0.5 * X.flatten()**2 - 2 * X.flatten() + 3 + np.random.randn(100) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear Regression (won't work well)
lr_linear = LinearRegression()
lr_linear.fit(X_train, y_train)
y_pred_linear = lr_linear.predict(X_test)
print(f"Linear R²: {r2_score(y_test, y_pred_linear):.3f}")

# Polynomial Regression (degree 2)
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)
y_pred_poly = lr_poly.predict(X_test_poly)
print(f"Polynomial (deg=2) R²: {r2_score(y_test, y_pred_poly):.3f}")

# Try different degrees
degrees = [1, 2, 3, 5, 10, 20]
train_scores = []
test_scores = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_p = poly.fit_transform(X_train)
    X_test_p = poly.transform(X_test)
    
    lr = LinearRegression()
    lr.fit(X_train_p, y_train)
    
    train_score = r2_score(y_train, lr.predict(X_train_p))
    test_score = r2_score(y_test, lr.predict(X_test_p))
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f"Degree {degree}: Train R²={train_score:.3f}, Test R²={test_score:.3f}")

# Multiple features polynomial
X_multi = np.random.rand(100, 2)
y_multi = X_multi[:, 0]**2 + X_multi[:, 1]**2 + np.random.randn(100) * 0.1

poly_multi = PolynomialFeatures(degree=2, include_bias=False)
X_multi_poly = poly_multi.fit_transform(X_multi)
print(f"\nOriginal features: {X_multi.shape[1]}")
print(f"Polynomial features: {X_multi_poly.shape[1]}")
print(f"Feature names: {poly_multi.get_feature_names_out(['x1', 'x2'])}")`,
				},
				{
					Title: "Ridge & Lasso Regression",
					Content: `Regularization techniques prevent overfitting by adding penalty terms to the cost function.

**Overfitting Problem:**
- Model learns training data too well
- Captures noise instead of signal
- Poor generalization to new data
- High variance, low bias

**Regularization:**
- Adds penalty for large coefficients
- Encourages simpler models
- Reduces overfitting
- Improves generalization

**Ridge Regression (L2 Regularization):**
- Penalty: λ × Σ(βᵢ)² (sum of squared coefficients)
- Shrinks coefficients toward zero
- Doesn't eliminate features (coefficients → 0 but not exactly 0)
- Good when: Many features, multicollinearity

**Lasso Regression (L1 Regularization):**
- Penalty: λ × Σ|βᵢ| (sum of absolute coefficients)
- Can eliminate features (coefficients → exactly 0)
- Performs feature selection automatically
- Good when: Feature selection needed, sparse solutions

**Elastic Net:**
- Combines L1 and L2 penalties
- Balance between Ridge and Lasso
- Good when: Many correlated features

**Lambda (α) Parameter:**
- Controls strength of regularization
- Large λ: More regularization (simpler model)
- Small λ: Less regularization (more complex model)
- Choose via cross-validation

**Key Differences:**

**Ridge:**
- All features remain in model
- Coefficients shrink smoothly
- Better for correlated features

**Lasso:**
- Can remove features entirely
- Sparse solution
- Better for feature selection`,
					CodeExamples: `from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate data with many features
X, y = make_regression(n_samples=100, n_features=20, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features (important for regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression (baseline)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
print(f"Linear Regression R²: {r2_score(y_test, lr.predict(X_test_scaled)):.3f}")
print(f"Number of non-zero coefficients: {np.sum(lr.coef_ != 0)}")

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
print(f"\nRidge Regression R²: {r2_score(y_test, ridge.predict(X_test_scaled)):.3f}")
print(f"Number of non-zero coefficients: {np.sum(ridge.coef_ != 0)}")
print(f"Sum of squared coefficients: {np.sum(ridge.coef_**2):.3f}")

# Lasso Regression
lasso = Lasso(alpha=1.0)
lasso.fit(X_train_scaled, y_train)
print(f"\nLasso Regression R²: {r2_score(y_test, lasso.predict(X_test_scaled)):.3f}")
print(f"Number of non-zero coefficients: {np.sum(lasso.coef_ != 0)}")
print(f"Sum of absolute coefficients: {np.sum(np.abs(lasso.coef_)):.3f}")

# Tuning alpha parameter
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
print(f"\nAlpha Tuning:")

for alpha in alphas:
    ridge_cv = Ridge(alpha=alpha)
    scores = cross_val_score(ridge_cv, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"Ridge α={alpha}: CV R² = {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

for alpha in alphas:
    lasso_cv = Lasso(alpha=alpha)
    scores = cross_val_score(lasso_cv, X_train_scaled, y_train, cv=5, scoring='r2')
    n_features = np.sum(lasso_cv.fit(X_train_scaled, y_train).coef_ != 0)
    print(f"Lasso α={alpha}: CV R² = {scores.mean():.3f}, Features = {n_features}")

# Elastic Net
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio: 0=Ridge, 1=Lasso
elastic.fit(X_train_scaled, y_train)
print(f"\nElastic Net R²: {r2_score(y_test, elastic.predict(X_test_scaled)):.3f}")
print(f"Number of non-zero coefficients: {np.sum(elastic.coef_ != 0)}")

# Feature selection with Lasso
lasso_strong = Lasso(alpha=10.0)
lasso_strong.fit(X_train_scaled, y_train)
selected_features = np.where(lasso_strong.coef_ != 0)[0]
print(f"\nLasso selected {len(selected_features)} features: {selected_features}")`,
				},
				{
					Title: "Cross-Validation",
					Content: `Cross-validation is a robust technique for model evaluation and hyperparameter tuning.

**Why Cross-Validation?**
- Single train-test split can be unreliable (depends on split)
- Uses all data for both training and validation
- More reliable performance estimate
- Reduces variance in performance estimates

**K-Fold Cross-Validation:**
1. Split data into K folds (typically K=5 or K=10)
2. For each fold:
   - Use fold as validation set
   - Use remaining K-1 folds as training set
   - Train and evaluate model
3. Average performance across K folds

**Stratified K-Fold:**
- Maintains class distribution in each fold
- Important for imbalanced datasets
- Ensures each fold represents population

**Leave-One-Out CV (LOOCV):**
- K = n (number of samples)
- Each sample is validation set once
- Most thorough but computationally expensive

**Time Series Cross-Validation:**
- Respects temporal order
- Train on past, validate on future
- Prevents data leakage

**Nested Cross-Validation:**
- Outer loop: Model evaluation
- Inner loop: Hyperparameter tuning
- Prevents overfitting to validation set

**Use Cases:**
- Model evaluation (more reliable than single split)
- Hyperparameter tuning (find best parameters)
- Feature selection (evaluate feature sets)
- Model comparison (compare different algorithms)`,
					CodeExamples: `from sklearn.model_selection import (
    cross_val_score, KFold, StratifiedKFold,
    LeaveOneOut, TimeSeriesSplit, GridSearchCV
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score
import numpy as np

# Classification example
X_clf, y_clf = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
model_clf = LogisticRegression()

# Standard K-Fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kfold = cross_val_score(model_clf, X_clf, y_clf, cv=kfold, scoring='accuracy')
print(f"5-Fold CV Accuracy: {scores_kfold.mean():.3f} (+/- {scores_kfold.std() * 2:.3f})")
print(f"Individual fold scores: {scores_kfold}")

# Stratified K-Fold (for imbalanced data)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_stratified = cross_val_score(model_clf, X_clf, y_clf, cv=skfold, scoring='accuracy')
print(f"\nStratified 5-Fold CV Accuracy: {scores_stratified.mean():.3f}")

# Leave-One-Out CV (computationally expensive)
# loo = LeaveOneOut()
# scores_loo = cross_val_score(model_clf, X_clf, y_clf, cv=loo, scoring='accuracy')
# print(f"LOOCV Accuracy: {scores_loo.mean():.3f}")  # Commented out - slow!

# Regression example
X_reg, y_reg = make_regression(n_samples=100, n_features=10, noise=10, random_state=42)
model_reg = Ridge(alpha=1.0)

scores_reg = cross_val_score(model_reg, X_reg, y_reg, cv=5, scoring='r2')
print(f"\nRegression 5-Fold CV R²: {scores_reg.mean():.3f} (+/- {scores_reg.std() * 2:.3f})")

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
# For time series data, this ensures train on past, test on future

# Nested Cross-Validation (for hyperparameter tuning)
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge = Ridge()

# Outer CV: Model evaluation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_scores = []

for train_idx, test_idx in outer_cv.split(X_reg):
    X_train_outer, X_test_outer = X_reg[train_idx], X_reg[test_idx]
    y_train_outer, y_test_outer = y_reg[train_idx], y_reg[test_idx]
    
    # Inner CV: Hyperparameter tuning
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(ridge, param_grid, cv=inner_cv, scoring='r2')
    grid_search.fit(X_train_outer, y_train_outer)
    
    # Evaluate best model on outer test set
    best_model = grid_search.best_estimator_
    score = best_model.score(X_test_outer, y_test_outer)
    outer_scores.append(score)
    print(f"\nOuter fold - Best α: {grid_search.best_params_['alpha']}, R²: {score:.3f}")

print(f"\nNested CV R²: {np.mean(outer_scores):.3f} (+/- {np.std(outer_scores) * 2:.3f})")

# Manual CV implementation (for understanding)
def manual_kfold_cv(X, y, model, k=5):
    """Manual K-fold cross-validation."""
    n_samples = len(X)
    fold_size = n_samples // k
    scores = []
    
    for i in range(k):
        # Define validation indices
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n_samples
        
        val_idx = np.arange(val_start, val_end)
        train_idx = np.setdiff1d(np.arange(n_samples), val_idx)
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train and evaluate
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

mean_score, std_score = manual_kfold_cv(X_reg, y_reg, Ridge(alpha=1.0), k=5)
print(f"\nManual K-Fold CV R²: {mean_score:.3f} (+/- {std_score * 2:.3f})")`,
				},
				{
					Title: "Overfitting & Underfitting",
					Content: `Understanding overfitting and underfitting is crucial for building good ML models.

**Bias-Variance Trade-off:**

**Bias:**
- Error from overly simplistic assumptions
- High bias = underfitting
- Model can't capture underlying pattern
- Example: Linear model for non-linear data

**Variance:**
- Error from sensitivity to small fluctuations
- High variance = overfitting
- Model captures noise instead of signal
- Example: Complex model fitting training noise

**Underfitting:**
- Model too simple for data
- High bias, low variance
- Poor performance on both train and test
- **Signs:**
  - Low training accuracy
  - Low test accuracy
  - Training ≈ Test (both low)
- **Solutions:**
  - Increase model complexity
  - Add more features
  - Reduce regularization
  - Train longer

**Overfitting:**
- Model too complex for data
- Low bias, high variance
- Good on training, poor on test
- **Signs:**
  - High training accuracy
  - Low test accuracy
  - Large gap: Train >> Test
- **Solutions:**
  - Reduce model complexity
  - Add more training data
  - Increase regularization
  - Feature selection
  - Early stopping

**Learning Curves:**
- Plot training/validation score vs training set size
- Helps diagnose bias/variance issues
- **High Bias**: Both curves converge to low score
- **High Variance**: Large gap between curves

**Regularization:**
- Technique to prevent overfitting
- Adds penalty for complexity
- Examples: Ridge, Lasso, Dropout (neural networks)

**Early Stopping:**
- Stop training when validation performance stops improving
- Prevents overfitting in iterative algorithms
- Common in neural networks, gradient boosting`,
					CodeExamples: `from sklearn.model_selection import learning_curve, validation_curve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Example 1: Underfitting (too simple model)
simple_model = LogisticRegression(C=1000)  # High regularization = simple
simple_model.fit(X_train, y_train)
train_score_simple = simple_model.score(X_train, y_train)
test_score_simple = simple_model.score(X_test, y_test)
print(f"Underfitting Model:")
print(f"  Train Score: {train_score_simple:.3f}")
print(f"  Test Score: {test_score_simple:.3f}")
print(f"  Gap: {abs(train_score_simple - test_score_simple):.3f}")

# Example 2: Overfitting (too complex model)
complex_model = DecisionTreeClassifier(max_depth=20, min_samples_split=2)
complex_model.fit(X_train, y_train)
train_score_complex = complex_model.score(X_train, y_train)
test_score_complex = complex_model.score(X_test, y_test)
print(f"\nOverfitting Model:")
print(f"  Train Score: {train_score_complex:.3f}")
print(f"  Test Score: {test_score_complex:.3f}")
print(f"  Gap: {abs(train_score_complex - test_score_complex):.3f}")

# Example 3: Good fit (balanced model)
balanced_model = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
balanced_model.fit(X_train, y_train)
train_score_balanced = balanced_model.score(X_train, y_train)
test_score_balanced = balanced_model.score(X_test, y_test)
print(f"\nBalanced Model:")
print(f"  Train Score: {train_score_balanced:.3f}")
print(f"  Test Score: {test_score_balanced:.3f}")
print(f"  Gap: {abs(train_score_balanced - test_score_balanced):.3f}")

# Learning Curves (diagnose bias/variance)
train_sizes, train_scores, val_scores = learning_curve(
    balanced_model, X_train, y_train, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

print(f"\nLearning Curve Analysis:")
print(f"Final Train Score: {train_mean[-1]:.3f}")
print(f"Final Val Score: {val_mean[-1]:.3f}")
print(f"Gap: {abs(train_mean[-1] - val_mean[-1]):.3f}")

# Validation Curve (tune hyperparameters)
param_range = [1, 5, 10, 20, 50]
train_scores_vc, val_scores_vc = validation_curve(
    DecisionTreeClassifier(), X_train, y_train,
    param_name='max_depth', param_range=param_range,
    cv=5, scoring='accuracy'
)

train_mean_vc = train_scores_vc.mean(axis=1)
val_mean_vc = val_scores_vc.mean(axis=1)

print(f"\nValidation Curve (max_depth):")
for depth, train_sc, val_sc in zip(param_range, train_mean_vc, val_mean_vc):
    print(f"  Depth {depth}: Train={train_sc:.3f}, Val={val_sc:.3f}, Gap={abs(train_sc-val_sc):.3f}")

# Polynomial regression overfitting example
from sklearn.linear_model import LinearRegression
X_poly = np.linspace(0, 10, 20).reshape(-1, 1)
y_poly = 2 * X_poly.flatten() + 1 + np.random.randn(20) * 0.5

degrees = [1, 3, 15]
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly_feat = poly.fit_transform(X_poly)
    lr = LinearRegression()
    lr.fit(X_poly_feat, y_poly)
    train_r2 = lr.score(X_poly_feat, y_poly)
    print(f"\nPolynomial Degree {degree}: Train R² = {train_r2:.3f}")`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
