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

**Formal Definition:**
A computer program is said to learn from experience E with respect to some task T and performance measure P, if its performance on T, as measured by P, improves with experience E (Tom Mitchell, 1997).

**Key Characteristics:**
- **Data-Driven**: Learns patterns from data rather than explicit instructions
- **Adaptive**: Improves performance with more data and experience
- **Generalizable**: Can make predictions on new, unseen data
- **Automated**: Reduces need for manual feature engineering (in deep learning)
- **Probabilistic**: Provides uncertainty estimates, not just predictions
- **Scalable**: Handles large datasets efficiently

**Why Machine Learning Matters:**
- **Complex Problems**: Solves problems too complex for traditional algorithms (e.g., image recognition with millions of pixel combinations)
- **Pattern Recognition**: Identifies subtle patterns humans might miss (e.g., fraud detection)
- **Scalability**: Handles large-scale data processing (billions of records)
- **Automation**: Automates decision-making processes (e.g., loan approvals)
- **Personalization**: Enables personalized experiences (recommendations, search, ads)
- **Continuous Improvement**: Models improve with more data over time
- **Cost Efficiency**: Reduces need for manual rule creation and maintenance

**Real-World Applications:**

**Healthcare:**
- Medical diagnosis (skin cancer detection, radiology)
- Drug discovery (molecular property prediction)
- Personalized treatment plans
- Predictive analytics for patient outcomes
- Medical image analysis

**Finance:**
- Fraud detection (credit card, insurance)
- Algorithmic trading (high-frequency trading)
- Credit scoring and risk assessment
- Loan approval automation
- Market prediction and analysis

**Technology:**
- Search engines (Google's ranking algorithms)
- Recommendation systems (Netflix, Amazon)
- Voice assistants (Siri, Alexa)
- Email filtering (spam detection)
- Content moderation (social media)

**Transportation:**
- Self-driving cars (perception, planning, control)
- Route optimization (Uber, delivery services)
- Traffic prediction and management
- Predictive maintenance for vehicles
- Demand forecasting for ride-sharing

**E-commerce:**
- Product recommendations (collaborative filtering)
- Price optimization (dynamic pricing)
- Inventory management (demand forecasting)
- Customer segmentation
- Search ranking

**Entertainment:**
- Content recommendation (Netflix, Spotify)
- Game AI (NPCs, procedural generation)
- Music generation (AI composers)
- Video game difficulty adjustment

**ML vs Traditional Programming:**

**Traditional Programming:**
- Explicit rules defined by programmer
- Deterministic output (same input → same output)
- Requires domain expertise to code rules
- Difficult to adapt to new scenarios
- Limited by programmer's knowledge
- Good for: Well-defined problems, exact solutions needed

**Machine Learning:**
- Rules learned from data
- Probabilistic output (provides confidence scores)
- Learns patterns automatically
- Adapts to new data (retraining)
- Can discover unexpected patterns
- Good for: Complex patterns, large datasets, adaptive systems

**When to Use ML:**
- Problem has patterns but rules are hard to define
- Large amounts of data available
- Problem requires adaptation to new data
- Traditional algorithms perform poorly
- Need to automate decision-making`,
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
- **No Training Data**: Don't have labeled data or historical examples
- **Real-time Constraints**: Need guaranteed response times
- **Examples**: 
  - Calculator operations (arithmetic is deterministic)
  - Sorting algorithms (well-defined comparison rules)
  - Database queries (SQL is explicit)
  - Business logic (if-else rules for discounts, taxes)
  - Authentication (password verification)
  - Payment processing (transaction validation)

**When to Use Machine Learning:**
- **Complex Patterns**: Patterns too complex to code explicitly
- **Large Problem Space**: Too many cases to handle manually
- **Adaptive Behavior**: Need system to adapt to new data
- **Pattern Recognition**: Identifying patterns in data
- **Historical Data Available**: Have examples of inputs and outputs
- **Examples**:
  - Image recognition (millions of pixel combinations)
  - Natural language understanding (infinite sentence variations)
  - Recommendation systems (complex user preferences)
  - Fraud detection (evolving fraud patterns)
  - Speech recognition (accent variations)
  - Sentiment analysis (subjective interpretation)
  - Medical diagnosis (complex symptom patterns)

**Decision Framework:**

Use Traditional Programming if:
1. Rules are clear and can be explicitly coded
2. Problem space is small and well-defined
3. Need 100% deterministic behavior
4. No historical data available
5. Interpretability is critical

Use Machine Learning if:
1. Patterns exist but are hard to define explicitly
2. Large amounts of data available
3. Problem requires adaptation
4. Performance improves with more data
5. Some uncertainty/error is acceptable

**Hybrid Approaches:**
Many real-world systems combine both:
- **ML for Pattern Recognition**: Use ML to identify patterns
- **Traditional Programming for Rules**: Apply business logic, validation, constraints
- **Example 1**: E-commerce site
  - ML: Product recommendations, search ranking
  - Traditional: Checkout process, payment validation, inventory updates
- **Example 2**: Autonomous vehicles
  - ML: Object detection, path planning
  - Traditional: Safety rules, emergency braking logic
- **Example 3**: Fraud detection
  - ML: Anomaly detection, risk scoring
  - Traditional: Business rules (max transaction limits), compliance checks

**Trade-offs:**

**Traditional Programming:**
- ✅ Predictable and deterministic
- ✅ Interpretable and debuggable
- ✅ No training data needed
- ✅ Fast development for simple problems
- ✅ Guaranteed correctness (if implemented correctly)
- ❌ Requires domain expertise
- ❌ Hard to adapt to new scenarios
- ❌ Limited by programmer's knowledge
- ❌ Doesn't improve with more data
- ❌ Brittle (breaks with edge cases)

**Machine Learning:**
- ✅ Learns complex patterns automatically
- ✅ Adapts to new data (retraining)
- ✅ Can improve over time
- ✅ Handles high-dimensional data
- ✅ Discovers unexpected patterns
- ❌ Requires training data (often large amounts)
- ❌ Less interpretable (black box problem)
- ❌ Can make mistakes (no 100% guarantee)
- ❌ Needs careful validation and testing
- ❌ Computationally expensive (training)
- ❌ Can perpetuate biases in data

**Cost-Benefit Analysis:**

**Development Cost:**
- Traditional: Lower initial cost, but maintenance can be high
- ML: Higher initial cost (data collection, training), but scales better

**Maintenance:**
- Traditional: Manual updates needed for rule changes
- ML: Retrain with new data, but can adapt automatically

**Scalability:**
- Traditional: Limited by rule complexity
- ML: Improves with more data

**Best Practice:**
Start with traditional programming if rules are clear. Consider ML when:
- Rules become too complex
- Data shows patterns you can't code
- System needs to adapt frequently
- Performance requirements exceed traditional approaches`,
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
				{
					Title: "Mathematical Foundations for ML",
					Content: `Understanding the mathematical foundations is crucial for mastering machine learning algorithms and concepts.

**Linear Algebra:**
Essential for understanding how ML algorithms work internally.

**Key Concepts:**
- **Vectors**: Represent data points, features, or model parameters
  - Dot product: Measures similarity between vectors
  - Norm: Measures vector magnitude (L1, L2 norms)
- **Matrices**: Represent datasets, transformations, and model weights
  - Matrix multiplication: Core operation in neural networks
  - Transpose: Used in gradient computations
  - Inverse: Solving linear systems
- **Eigenvalues/Eigenvectors**: Used in PCA, dimensionality reduction
- **Matrix Decompositions**: 
  - SVD (Singular Value Decomposition): Used in PCA, recommendation systems
  - Eigendecomposition: Understanding transformations

**Why Important:**
- Neural networks: Matrix multiplications between layers
- PCA: Uses eigendecomposition for dimensionality reduction
- Regularization: L1/L2 norms used in loss functions
- Optimization: Gradients are vectors, Hessians are matrices

**Calculus:**
Necessary for understanding optimization and learning algorithms.

**Key Concepts:**
- **Derivatives**: Rate of change, used in gradient descent
- **Partial Derivatives**: How loss changes with respect to each parameter
- **Gradient**: Vector of partial derivatives, points to steepest ascent
- **Chain Rule**: Essential for backpropagation in neural networks
- **Optimization**: Finding minima/maxima of functions
- **Multivariate Calculus**: Functions with multiple variables (most ML problems)

**Why Important:**
- **Gradient Descent**: Uses derivatives to find optimal parameters
- **Backpropagation**: Chain rule computes gradients through network
- **Loss Functions**: Minimizing loss requires calculus
- **Regularization**: Understanding how penalties affect optimization

**Probability & Statistics:**
Fundamental for understanding uncertainty, evaluation, and many algorithms.

**Key Concepts:**
- **Probability Distributions**: 
  - Normal/Gaussian: Common assumption in many algorithms
  - Bernoulli/Binomial: Binary classification
  - Multinomial: Multi-class classification
- **Bayes' Theorem**: Foundation of Naive Bayes, Bayesian methods
- **Expectation & Variance**: Understanding model outputs and uncertainty
- **Maximum Likelihood Estimation (MLE)**: How models learn from data
- **Hypothesis Testing**: Evaluating model significance
- **Confidence Intervals**: Quantifying uncertainty in predictions

**Why Important:**
- **Probabilistic Models**: Naive Bayes, Gaussian processes
- **Evaluation**: Statistical significance of results
- **Uncertainty Quantification**: Understanding model confidence
- **Bayesian Methods**: Prior knowledge incorporation
- **A/B Testing**: Statistical validation of model improvements

**Information Theory:**
Useful for understanding decision trees, feature selection, and model evaluation.

**Key Concepts:**
- **Entropy**: Measure of uncertainty/randomness
  - H(X) = -Σ P(x) log₂ P(x)
  - Used in decision trees (information gain)
- **Mutual Information**: Measures dependency between variables
- **KL Divergence**: Measures difference between distributions
- **Cross-Entropy**: Loss function for classification

**Why Important:**
- **Decision Trees**: Use entropy for splitting criteria
- **Feature Selection**: Mutual information identifies relevant features
- **Loss Functions**: Cross-entropy for classification
- **Model Comparison**: KL divergence compares distributions

**Optimization Theory:**
Understanding how algorithms find optimal solutions.

**Key Concepts:**
- **Convex vs Non-Convex**: 
  - Convex: Single global minimum (linear regression, SVM)
  - Non-Convex: Multiple local minima (neural networks)
- **Gradient Descent Variants**:
  - Batch GD: Uses all data
  - Stochastic GD: Uses one sample
  - Mini-batch GD: Uses small batches
- **Learning Rate**: Step size in optimization
- **Momentum**: Helps escape local minima
- **Second-Order Methods**: Newton's method, uses Hessian

**Why Important:**
- **Training Models**: All ML models use optimization
- **Convergence**: Understanding when algorithms converge
- **Hyperparameter Tuning**: Learning rate, batch size selection
- **Regularization**: How penalties affect optimization landscape

**Practical Tips:**
- **Don't need to master everything upfront**: Learn as you encounter algorithms
- **Focus on intuition**: Understanding concepts more important than proofs
- **Use libraries**: NumPy, SciPy handle most math operations
- **Visualize**: Graphs help understand gradients, distributions
- **Practice**: Implement algorithms from scratch to deepen understanding`,
					CodeExamples: `import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Linear Algebra: Vectors and Matrices
# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32
print(f"Dot product: {dot_product}")

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
matrix_product = np.dot(A, B)  # Matrix multiplication
print(f"Matrix product:\n{matrix_product}")

# L2 norm (Euclidean distance)
l2_norm = np.linalg.norm(v1)
print(f"L2 norm of v1: {l2_norm:.2f}")

# Calculus: Gradient computation
def loss_function(x):
    return x**2 + 2*x + 1

def gradient(x):
    return 2*x + 2  # Derivative of loss_function

# Gradient descent
x = 5.0
learning_rate = 0.1
for i in range(10):
    grad = gradient(x)
    x = x - learning_rate * grad
    print(f"Iteration {i+1}: x = {x:.3f}, loss = {loss_function(x):.3f}")

# Probability: Distributions
# Normal distribution
mu, sigma = 0, 1
x = np.linspace(-4, 4, 100)
pdf = stats.norm.pdf(x, mu, sigma)
plt.plot(x, pdf, label='Normal Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Normal Distribution')
plt.legend()
plt.show()

# Bayes' Theorem example
# P(A|B) = P(B|A) * P(A) / P(B)
# Example: Spam detection
P_spam = 0.3  # Prior: 30% of emails are spam
P_word_given_spam = 0.8  # 80% of spam contains "free"
P_word_given_ham = 0.1  # 10% of ham contains "free"
P_word = P_word_given_spam * P_spam + P_word_given_ham * (1 - P_spam)

# Posterior: P(spam|word)
P_spam_given_word = (P_word_given_spam * P_spam) / P_word
print(f"P(spam|'free'): {P_spam_given_word:.3f}")

# Information Theory: Entropy
def entropy(probs):
    """Calculate entropy of a probability distribution."""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Remove zeros
    return -np.sum(probs * np.log2(probs))

# Example: Coin flip
fair_coin = [0.5, 0.5]
biased_coin = [0.9, 0.1]
print(f"Fair coin entropy: {entropy(fair_coin):.3f}")  # Maximum uncertainty
print(f"Biased coin entropy: {entropy(biased_coin):.3f}")  # Less uncertainty

# Optimization: Gradient descent visualization
def f(x):
    return x**2 + 2*x + 1

def df(x):
    return 2*x + 2

x_vals = np.linspace(-5, 3, 100)
y_vals = f(x_vals)

x_current = 2.0
learning_rate = 0.2
trajectory = [x_current]

for _ in range(10):
    gradient = df(x_current)
    x_current = x_current - learning_rate * gradient
    trajectory.append(x_current)

plt.plot(x_vals, y_vals, label='f(x)')
plt.plot(trajectory, [f(x) for x in trajectory], 'ro-', label='Gradient Descent')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Optimization')
plt.legend()
plt.grid(True)
plt.show()`,
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

**Why Feature Selection Matters:**
- **Curse of Dimensionality**: Performance degrades with too many features
- **Noise Reduction**: Irrelevant features add noise
- **Computational Efficiency**: Fewer features = faster training
- **Model Interpretability**: Easier to understand simpler models
- **Generalization**: Reduces overfitting risk

**Benefits:**
- **Reduces Overfitting**: Fewer features = simpler model
- **Faster Training**: Less data to process
- **Better Interpretability**: Focus on important features
- **Lower Storage**: Less data to store and process
- **Improved Accuracy**: Removes noise and irrelevant information
- **Cost Reduction**: Fewer features to collect and maintain

**Feature Selection Methods:**

**1. Filter Methods**
- Select features based on statistical measures
- Independent of ML algorithm
- Fast but may miss feature interactions
- Computed before model training
- Examples: 
  - Correlation coefficient (linear relationships)
  - Chi-square test (categorical features)
  - Mutual Information (non-linear relationships)
  - ANOVA F-test (variance analysis)
- **Pros**: Fast, scalable, model-agnostic
- **Cons**: Ignores feature interactions, may miss optimal subset

**2. Wrapper Methods**
- Use ML algorithm to evaluate feature subsets
- More accurate but computationally expensive
- Train model for each feature subset
- Examples: 
  - Forward Selection: Start empty, add best features
  - Backward Elimination: Start with all, remove worst
  - Recursive Feature Elimination (RFE): Recursively remove worst
  - Exhaustive Search: Try all combinations (very expensive)
- **Pros**: Considers feature interactions, finds good subsets
- **Cons**: Slow, computationally expensive, can overfit

**3. Embedded Methods**
- Feature selection built into algorithm
- Performed during model training
- Examples: 
  - Lasso regularization (L1): Sets coefficients to zero
  - Ridge regularization (L2): Shrinks coefficients
  - Tree-based importance: Random Forest, XGBoost feature importance
  - Elastic Net: Combines L1 and L2
- **Pros**: Efficient, considers interactions, model-specific
- **Cons**: Tied to specific algorithm, may not generalize

**Common Techniques:**

**Correlation-based:**
- Remove highly correlated features (redundancy)
  - Threshold: Remove if |correlation| > 0.8-0.9
- Keep features correlated with target
- **Use case**: Remove redundant features (e.g., height in cm and inches)

**Variance-based:**
- Remove low-variance features (little information)
  - Features with variance < threshold
- **Use case**: Remove constant or near-constant features
- **Caution**: May remove important binary features

**Univariate Selection:**
- Select K best features based on statistical test
- Independent evaluation of each feature
- **Use case**: Quick initial feature selection
- **Limitation**: Doesn't consider feature interactions

**Recursive Feature Elimination (RFE):**
- Recursively remove worst features
- Uses model to evaluate feature importance
- **Use case**: When you have a good model and want to optimize features
- **Process**: 
  1. Train model with all features
  2. Rank features by importance
  3. Remove least important feature
  4. Repeat until desired number of features

**Best Practices:**
- **Start with filter methods**: Quick initial reduction
- **Use domain knowledge**: Understand which features matter
- **Cross-validate**: Feature selection on training set only
- **Avoid data leakage**: Don't use test set for feature selection
- **Consider interactions**: Some methods miss feature interactions
- **Iterate**: Feature selection is iterative process
- **Monitor performance**: Track how selection affects model performance`,
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
print(f"Removed {len(high_corr_features)} highly correlated features")

# Variance-based selection
from sklearn.feature_selection import VarianceThreshold
selector_var = VarianceThreshold(threshold=0.01)  # Remove low variance
X_var_selected = selector_var.fit_transform(X)
print(f"Features after variance selection: {X_var_selected.shape[1]}")

# Correlation with target (for regression)
if hasattr(y, 'corr'):
    correlations = X.corrwith(y).abs()
    top_features = correlations.nlargest(10).index
    print(f"\nTop 10 features by correlation with target:")
    print(correlations[top_features])`,
				},
				{
					Title: "Advanced Feature Engineering",
					Content: `Advanced feature engineering creates new features that help models learn better patterns.

**Feature Engineering Principles:**
- **Domain Knowledge**: Use expertise to create meaningful features
- **Transformations**: Apply mathematical transformations
- **Interactions**: Combine features to capture relationships
- **Temporal Features**: Extract time-based patterns
- **Aggregations**: Summarize information across groups

**Common Advanced Techniques:**

**1. Polynomial Features:**
- Create interaction terms (x₁ × x₂)
- Higher-order terms (x², x³)
- Captures non-linear relationships
- **Use case**: When relationships are non-linear
- **Caution**: Can lead to overfitting, increases dimensionality

**2. Binning/Discretization:**
- Convert continuous to categorical
- Groups similar values together
- **Methods**: 
  - Equal-width bins
  - Equal-frequency bins
  - Domain-based bins (e.g., age groups)
- **Use case**: Non-linear relationships, reducing noise

**3. Log Transformations:**
- Apply log to skewed distributions
- Makes data more normally distributed
- **Use case**: Right-skewed data (prices, counts)
- **Formula**: x' = log(x + 1)  # +1 to handle zeros

**4. Feature Interactions:**
- Multiply features (x₁ × x₂)
- Add features (x₁ + x₂)
- Ratio features (x₁ / x₂)
- **Use case**: Capture feature relationships
- **Example**: Price per square foot = price / area

**5. Temporal Features:**
- Extract from dates/times:
  - Day of week, month, quarter
  - Hour of day
  - Time since event
  - Is weekend? Is holiday?
- **Use case**: Time series, seasonal patterns

**6. Aggregation Features:**
- Group by category and aggregate:
  - Mean, median, std, min, max
  - Count, sum
- **Use case**: Capture group-level patterns
- **Example**: Average purchase by customer category

**7. Text Features:**
- Length of text
- Word count
- Character count
- Presence of keywords
- Sentiment score
- **Use case**: NLP tasks, text classification

**8. Target Encoding:**
- Replace category with mean target value
- Captures relationship with target
- **Use case**: High cardinality categorical features
- **Caution**: Can cause overfitting, use cross-validation

**Best Practices:**
- **Start Simple**: Begin with basic features
- **Domain Knowledge**: Use expertise to guide engineering
- **Visualize**: Plot features to understand distributions
- **Validate**: Test engineered features improve performance
- **Avoid Leakage**: Don't use future information
- **Document**: Keep track of feature creation logic
- **Iterate**: Feature engineering is iterative process`,
					CodeExamples: `import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import KBinsDiscretizer

# Sample data
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50],
    'income': [50000, 60000, 70000, 80000, 90000, 100000],
    'price': [100000, 150000, 200000, 250000, 300000, 350000]
})

# 1. Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['age', 'income']])
print("Polynomial features shape:", X_poly.shape)
print("Feature names:", poly.get_feature_names_out(['age', 'income']))

# 2. Binning/Discretization
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
age_binned = discretizer.fit_transform(df[['age']])
df['age_group'] = age_binned.astype(int)
print("\nAge groups:")
print(df[['age', 'age_group']])

# 3. Log Transformations
df['log_income'] = np.log1p(df['income'])  # log(1+x) handles zeros
df['log_price'] = np.log1p(df['price'])
print("\nLog transformed features:")
print(df[['income', 'log_income', 'price', 'log_price']].head())

# 4. Feature Interactions
df['income_per_age'] = df['income'] / df['age']
df['price_per_income'] = df['price'] / df['income']
df['age_income_interaction'] = df['age'] * df['income']
print("\nInteraction features:")
print(df[['age', 'income', 'income_per_age', 'age_income_interaction']].head())

# 5. Temporal Features (example with dates)
dates = pd.date_range('2024-01-01', periods=6, freq='D')
df['date'] = dates
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['month'] = df['date'].dt.month
print("\nTemporal features:")
print(df[['date', 'day_of_week', 'is_weekend', 'month']])

# 6. Aggregation Features (example)
df['category'] = ['A', 'A', 'B', 'B', 'C', 'C']
category_stats = df.groupby('category')['income'].agg(['mean', 'std', 'count'])
df = df.merge(category_stats, left_on='category', right_index=True, suffixes=('', '_category'))
print("\nAggregation features:")
print(df[['category', 'income', 'mean_category', 'std_category']])

# 7. Text Features (example)
texts = ["Hello world", "Machine learning is great", "Python programming"]
df_text = pd.DataFrame({'text': texts})
df_text['text_length'] = df_text['text'].str.len()
df_text['word_count'] = df_text['text'].str.split().str.len()
df_text['has_learning'] = df_text['text'].str.contains('learning', case=False).astype(int)
print("\nText features:")
print(df_text)

# 8. Target Encoding (with cross-validation)
from sklearn.model_selection import KFold

# Simulated target
df['target'] = [1, 0, 1, 0, 1, 0]
df['category_encoded'] = 0

kf = KFold(n_splits=3, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(df):
    train_mean = df.loc[train_idx].groupby('category')['target'].mean()
    df.loc[val_idx, 'category_encoded'] = df.loc[val_idx, 'category'].map(train_mean)

print("\nTarget encoding:")
print(df[['category', 'target', 'category_encoded']])`,
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
- ε: error term (residual, difference between predicted and actual)

**Multiple Linear Regression:**
- Multiple features → one target
- Formula: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
- Each feature has its own coefficient
- Matrix form: y = Xβ + ε where X is feature matrix, β is coefficient vector

**Cost Function - Mean Squared Error (MSE):**
The goal is to minimize prediction error. MSE measures average squared difference between predictions and actual values.

MSE = (1/n) Σ(yᵢ - ŷᵢ)²

Where:
- n: number of samples
- yᵢ: actual value for sample i
- ŷᵢ: predicted value for sample i
- (yᵢ - ŷᵢ): residual (error) for sample i

**Why squared errors?**
- Penalizes large errors more than small ones
- Differentiable everywhere (smooth optimization)
- Mathematically convenient (leads to closed-form solution)

**Optimization Methods:**

**1. Normal Equation (Closed-Form Solution):**
Derived by setting derivative of MSE to zero and solving for coefficients.

For simple linear regression:
- β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
- β₀ = ȳ - β₁x̄

For multiple regression (matrix form):
- β = (XᵀX)⁻¹Xᵀy

**When to use Normal Equation:**
- Small datasets (n < 10,000)
- When XᵀX is invertible (no perfect multicollinearity)
- Need exact solution quickly
- **Limitation**: O(n³) complexity, doesn't scale well

**2. Gradient Descent (Iterative Solution):**
Iteratively updates coefficients to minimize cost function.

**Algorithm:**
1. Initialize coefficients (β₀, β₁, ...) randomly or to zero
2. Calculate predictions: ŷ = Xβ
3. Calculate cost: MSE = (1/n)Σ(y - ŷ)²
4. Calculate gradients (partial derivatives):
   - ∂MSE/∂β₀ = -(2/n)Σ(yᵢ - ŷᵢ)
   - ∂MSE/∂βⱼ = -(2/n)Σ(yᵢ - ŷᵢ)xᵢⱼ
5. Update coefficients:
   - βⱼ = βⱼ - α × (∂MSE/∂βⱼ)
   - α: learning rate (step size)
6. Repeat steps 2-5 until convergence

**Gradient Descent Variants:**
- **Batch GD**: Uses all samples for each update (slow but stable)
- **Stochastic GD**: Uses one sample per update (fast but noisy)
- **Mini-batch GD**: Uses small batches (balance of speed and stability)

**Learning Rate Selection:**
- Too small: Slow convergence, many iterations needed
- Too large: May overshoot minimum, diverge
- Adaptive: Learning rate schedules (decay over time)

**When to use Gradient Descent:**
- Large datasets (n > 10,000)
- When XᵀX is not invertible
- Need approximate solution quickly
- **Advantage**: O(n) per iteration, scales well

**Assumptions and Violations:**

**1. Linear Relationship:**
- **Assumption**: Relationship between features and target is linear
- **Check**: Plot residuals vs predicted values (should be random scatter)
- **Violation**: Curved pattern in residual plot
- **Remedy**: Transform features (log, polynomial), use non-linear models

**2. Independence of Features (No Multicollinearity):**
- **Assumption**: Features are not highly correlated with each other
- **Check**: Correlation matrix, Variance Inflation Factor (VIF)
  - VIF > 10 indicates multicollinearity
  - VIF = 1 / (1 - R²) where R² is from regressing one feature on others
- **Violation**: High correlation between features
- **Remedy**: Remove redundant features, use regularization (Ridge/Lasso), PCA

**3. Normality of Errors:**
- **Assumption**: Residuals are normally distributed
- **Check**: Q-Q plot, Shapiro-Wilk test, histogram of residuals
- **Violation**: Skewed or heavy-tailed distribution
- **Remedy**: Transform target variable (log, Box-Cox), robust regression methods
- **Note**: Less critical for large samples (Central Limit Theorem)

**4. Homoscedasticity (Constant Variance):**
- **Assumption**: Variance of errors is constant across all values of X
- **Check**: Plot residuals vs predicted values (should have constant spread)
- **Violation**: Funnel shape (heteroscedasticity)
- **Remedy**: Transform target (log), weighted least squares, robust standard errors

**5. Independence of Observations:**
- **Assumption**: Observations are independent (not autocorrelated)
- **Check**: Durbin-Watson test for time series
- **Violation**: Time series data, clustered data
- **Remedy**: Time series models, mixed-effects models

**Coefficient Interpretation:**

**Simple Linear Regression:**
- β₀ (Intercept): Expected value of y when x = 0
  - May not be meaningful if x=0 is outside data range
- β₁ (Slope): Expected change in y for one-unit increase in x
  - Example: β₁ = 2.5 means y increases by 2.5 units per unit increase in x

**Multiple Linear Regression:**
- βⱼ: Expected change in y for one-unit increase in xⱼ, holding all other features constant
- **Partial effect**: Effect of one feature controlling for others
- **Example**: β₁ = 0.5 means y increases by 0.5 units per unit increase in x₁, assuming x₂, x₃, ... remain constant

**Statistical Significance:**
- t-test: H₀: βⱼ = 0 vs H₁: βⱼ ≠ 0
- p-value < 0.05: Feature is statistically significant
- Confidence intervals: Range of plausible coefficient values

**Residual Analysis:**
Residuals (errors) = y - ŷ provide insights into model quality.

**Diagnostic Plots:**
1. **Residuals vs Predicted**: Should show random scatter (no patterns)
   - Pattern indicates non-linearity or heteroscedasticity
2. **Q-Q Plot**: Residuals should follow normal distribution line
   - Deviations indicate non-normal errors
3. **Scale-Location Plot**: Spread of residuals vs predicted values
   - Should be horizontal (constant variance)
4. **Residuals vs Leverage**: Identify influential points
   - Points far from others may have high leverage

**Common Issues:**
- **Outliers**: Points with large residuals
  - Check: Standardized residuals > 3 or < -3
  - Remedy: Remove outliers, use robust regression
- **Influential Points**: Points that significantly affect coefficients
  - Check: Cook's distance > 4/n
  - Remedy: Investigate, consider removing if data error

**Multicollinearity Detection and Handling:**

**Detection Methods:**
1. **Correlation Matrix**: |r| > 0.8 suggests multicollinearity
2. **Variance Inflation Factor (VIF)**:
   - VIF = 1 / (1 - R²ⱼ) where R²ⱼ is from regressing feature j on other features
   - VIF > 10: High multicollinearity
   - VIF > 5: Moderate multicollinearity
3. **Eigenvalues**: Small eigenvalues of XᵀX indicate multicollinearity

**Effects of Multicollinearity:**
- Unstable coefficients (small data changes → large coefficient changes)
- Large standard errors (wide confidence intervals)
- Counterintuitive coefficient signs
- Difficulty interpreting individual feature effects

**Remedies:**
1. **Remove redundant features**: Drop highly correlated features
2. **Feature selection**: Use techniques to select most important features
3. **Regularization**: Ridge regression adds penalty for large coefficients
4. **PCA**: Transform to orthogonal components
5. **Domain knowledge**: Combine related features into single feature

**Advantages:**
- Simple and interpretable
- Fast training and prediction
- No hyperparameters to tune (basic version)
- Works well when relationship is linear
- Provides statistical inference (confidence intervals, p-values)
- Baseline for comparison with complex models

**Limitations:**
- Assumes linear relationship (may not capture non-linear patterns)
- Sensitive to outliers (can skew line)
- Can't capture non-linear patterns without transformations
- Requires feature scaling for gradient descent (not normal equation)
- Assumes independence and homoscedasticity (may not hold in practice)
- Multicollinearity can cause interpretation issues`,
					CodeExamples: `import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.flatten() + 1.0 + np.random.randn(100) * 2

# Method 1: Using sklearn (Normal Equation internally)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Sklearn - Coefficient: {model.coef_[0]:.2f}, Intercept: {model.intercept_:.2f}")
print(f"MSE: {mse:.2f}, R²: {r2:.2f}")

# Method 2: Normal Equation (closed-form solution)
def normal_equation(X, y):
    """Solve linear regression using normal equation: β = (XᵀX)⁻¹Xᵀy"""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    XTX = X_with_intercept.T @ X_with_intercept
    XTy = X_with_intercept.T @ y
    coefficients = np.linalg.inv(XTX) @ XTy
    return coefficients[0], coefficients[1:]  # intercept, slope

intercept_ne, slope_ne = normal_equation(X.flatten(), y)
print(f"\nNormal Equation - Intercept: {intercept_ne:.2f}, Slope: {slope_ne[0]:.2f}")

# Method 3: Gradient Descent (iterative solution)
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Gradient descent for linear regression.
    Updates coefficients iteratively to minimize MSE.
    """
    m = len(y)  # number of samples
    # Initialize coefficients
    beta_0 = 0.0  # intercept
    beta_1 = 0.0  # slope
    
    # Store cost history
    cost_history = []
    
    for i in range(iterations):
        # Predictions
        y_pred = beta_0 + beta_1 * X
        
        # Calculate cost (MSE)
        cost = (1/(2*m)) * np.sum((y_pred - y)**2)
        cost_history.append(cost)
        
        # Calculate gradients (partial derivatives)
        d_beta_0 = (1/m) * np.sum(y_pred - y)
        d_beta_1 = (1/m) * np.sum((y_pred - y) * X)
        
        # Update coefficients
        beta_0 = beta_0 - learning_rate * d_beta_0
        beta_1 = beta_1 - learning_rate * d_beta_1
        
        # Check convergence
        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < 1e-6:
            print(f"Converged after {i+1} iterations")
            break
    
    return beta_0, beta_1, cost_history

beta_0_gd, beta_1_gd, costs = gradient_descent(X.flatten(), y, learning_rate=0.01, iterations=1000)
print(f"\nGradient Descent - Intercept: {beta_0_gd:.2f}, Slope: {beta_1_gd:.2f}")
print(f"Final cost: {costs[-1]:.4f}")

# Plot cost convergence
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Gradient Descent Convergence')
plt.grid(True)

# Multiple Linear Regression
from sklearn.datasets import make_regression
X_multi, y_multi = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)

# Scale features for better gradient descent performance
scaler = StandardScaler()
X_multi_scaled = scaler.fit_transform(X_multi)

model_multi = LinearRegression()
model_multi.fit(X_multi_scaled, y_multi)
print(f"\nMultiple Regression Coefficients: {model_multi.coef_}")
print(f"Intercept: {model_multi.intercept_:.2f}")

# Residual Analysis - Diagnostic Plots
residuals = y - y_pred
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()

# Check assumptions
print("\n=== Assumption Checks ===")

# 1. Normality of residuals
from scipy.stats import shapiro
stat, p_value = shapiro(residuals)
print(f"Shapiro-Wilk test for normality: p-value = {p_value:.4f}")
if p_value > 0.05:
    print("  Residuals appear normally distributed")
else:
    print("  Residuals may not be normally distributed")

# 2. Homoscedasticity (constant variance)
# Check if variance of residuals is constant across predicted values
residuals_by_pred = [residuals[y_pred < np.median(y_pred)], 
                     residuals[y_pred >= np.median(y_pred)]]
var1, var2 = np.var(residuals_by_pred[0]), np.var(residuals_by_pred[1])
print(f"\nVariance check (homoscedasticity):")
print(f"  Variance for low predictions: {var1:.2f}")
print(f"  Variance for high predictions: {var2:.2f}")
if abs(var1 - var2) / max(var1, var2) < 0.5:
    print("  Variance appears constant (homoscedastic)")
else:
    print("  Variance may not be constant (heteroscedastic)")

# 3. Multicollinearity (for multiple regression)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_with_const = add_constant(X_multi_scaled)
vif_data = pd.DataFrame()
vif_data["Feature"] = ['Intercept'] + [f'X{i+1}' for i in range(X_multi_scaled.shape[1])]
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                   for i in range(X_with_const.shape[1])]
print(f"\nVariance Inflation Factors (VIF):")
print(vif_data)
print("  VIF > 10 indicates multicollinearity")

# Coefficient Interpretation Example
print("\n=== Coefficient Interpretation ===")
print(f"For each unit increase in X, y increases by {model.coef_[0]:.2f} units")
print(f"When X = 0, the expected value of y is {model.intercept_:.2f}")

# Predict on new data
X_new = np.array([[5.0]])
prediction = model.predict(X_new)
print(f"\nPrediction for X=5.0: {prediction[0]:.2f}")

# Compare methods
print("\n=== Method Comparison ===")
print(f"Normal Equation:    β₀={intercept_ne:.4f}, β₁={slope_ne[0]:.4f}")
print(f"Gradient Descent:  β₀={beta_0_gd:.4f}, β₁={beta_1_gd:.4f}")
print(f"Sklearn:           β₀={model.intercept_:.4f}, β₁={model.coef_[0]:.4f}")
print("All methods should give similar results!")`,
				},
				{
					Title: "Logistic Regression",
					Content: `Logistic Regression is used for binary classification, predicting probabilities using a logistic (sigmoid) function.

**Key Concept:**
- Outputs probability (0 to 1) instead of continuous value
- Uses sigmoid function to map linear combination to probability
- Formula: P(y=1) = 1 / (1 + e^(-z)) where z = β₀ + β₁x₁ + ...
- z is called the "logit" or "log-odds"

**Why Sigmoid Function? Derivation from Odds Ratio:**

The sigmoid function naturally arises from modeling the odds ratio.

**Step 1: Define Odds**
- Odds = P(y=1) / P(y=0) = P / (1-P)
- Odds > 1: More likely class 1
- Odds < 1: More likely class 0

**Step 2: Log-Odds (Logit)**
- Log-odds = log(P / (1-P)) = logit(P)
- This can be modeled linearly: logit(P) = β₀ + β₁x₁ + ...

**Step 3: Solve for Probability**
- log(P / (1-P)) = z
- P / (1-P) = e^z
- P = e^z / (1 + e^z)
- P = 1 / (1 + e^(-z))  ← Sigmoid function!

**Sigmoid Function Properties:**
- **Range**: (0, 1) - perfect for probabilities
- **S-shaped curve**: Steepest at z=0, flattens at extremes
- **Smooth and differentiable**: Essential for gradient-based optimization
- **Symmetric**: sigmoid(-z) = 1 - sigmoid(z)
- **At z=0**: probability = 0.5 (decision boundary)
- **Derivative**: σ'(z) = σ(z)(1 - σ(z)) - convenient for backpropagation

**How Logistic Regression Works:**

**Forward Pass:**
1. **Linear Combination**: z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
   - This is the "logit" or "log-odds"
2. **Sigmoid Transform**: p = σ(z) = 1/(1+e^(-z))
   - Maps logit to probability [0, 1]
3. **Prediction**: 
   - Class 1 if p > 0.5 (or z > 0)
   - Class 0 if p ≤ 0.5 (or z ≤ 0)
   - Threshold can be adjusted (not always 0.5)

**Decision Boundary:**
- Decision boundary is where p = 0.5, which occurs when z = 0
- For 2D: β₀ + β₁x₁ + β₂x₂ = 0
- This is a **linear** decision boundary (straight line in 2D, hyperplane in higher dimensions)
- All points on one side → class 1, other side → class 0

**Maximum Likelihood Estimation (MLE):**

Logistic regression uses MLE to find optimal coefficients.

**Likelihood Function:**
For each sample i:
- If yᵢ = 1: probability is pᵢ
- If yᵢ = 0: probability is (1 - pᵢ)

Combined: L(β) = Π pᵢ^yᵢ × (1-pᵢ)^(1-yᵢ)

**Log-Likelihood (easier to optimize):**
ℓ(β) = Σ [yᵢ·log(pᵢ) + (1-yᵢ)·log(1-pᵢ)]

**MLE Process:**
1. Start with initial coefficients β
2. Calculate probabilities pᵢ for all samples
3. Calculate log-likelihood
4. Update β to maximize log-likelihood (using gradient ascent or equivalent)
5. Repeat until convergence

**Cost Function - Log Loss (Binary Cross-Entropy):**

Instead of maximizing likelihood, we minimize negative log-likelihood (equivalent):

J(β) = -(1/n) Σ [yᵢ·log(pᵢ) + (1-yᵢ)·log(1-pᵢ)]

**Why This Cost Function?**
- **Penalizes confident wrong predictions heavily**: 
  - If y=1 but p≈0: log(p) → -∞, cost → ∞
  - If y=0 but p≈1: log(1-p) → -∞, cost → ∞
- **Rewards correct predictions**: 
  - If y=1 and p≈1: log(p) ≈ 0, cost ≈ 0
- **Convex**: Single global minimum (guaranteed convergence)
- **Differentiable**: Enables gradient-based optimization

**Gradient Descent for Logistic Regression:**

**Gradient Calculation:**
The gradient of log loss with respect to coefficient βⱼ:

∂J/∂βⱼ = (1/n) Σ (pᵢ - yᵢ) · xᵢⱼ

**Key Insight**: The gradient is the average of (prediction error) × (feature value)

**Gradient Descent Algorithm:**
1. Initialize coefficients β (usually zeros)
2. For each iteration:
   a. Calculate predictions: pᵢ = σ(β₀ + β₁xᵢ₁ + ...)
   b. Calculate cost: J(β)
   c. Calculate gradients: ∂J/∂βⱼ for all j
   d. Update coefficients: βⱼ = βⱼ - α · (∂J/∂βⱼ)
     - α: learning rate
3. Repeat until convergence (gradients ≈ 0)

**Note**: Unlike linear regression, there's no closed-form solution, so iterative optimization is necessary.

**Coefficient Interpretation:**

**Log-Odds Interpretation:**
- βⱼ: Change in log-odds for one-unit increase in xⱼ
- Example: β₁ = 0.5 means log-odds increases by 0.5 per unit increase in x₁

**Odds Ratio Interpretation:**
- e^(βⱼ): Multiplicative change in odds
- Example: e^0.5 ≈ 1.65 means odds increase by 65% per unit increase in x₁

**Probability Change:**
- Effect on probability depends on current probability level
- At p=0.5: Effect is largest
- At p near 0 or 1: Effect is smaller (sigmoid is flatter)

**Multiclass Extension:**

**1. One-vs-Rest (OvR):**
- Train K binary classifiers (one per class)
- Each classifier: "this class" vs "all others"
- Prediction: Class with highest probability
- **Pros**: Simple, works well when classes are balanced
- **Cons**: May have overlapping decision regions

**2. Multinomial (Softmax):**
- Single model with softmax activation
- **Softmax**: Converts K logits to K probabilities summing to 1
  - softmax(zⱼ) = e^(zⱼ) / Σᵢ e^(zᵢ)
- **Cost**: Cross-entropy loss (generalization of log loss)
- **Pros**: Single coherent model, better for imbalanced classes
- **Cons**: More complex, requires more data

**Regularization in Logistic Regression:**

**L1 Regularization (Lasso):**
- Adds penalty: λ · Σ|βⱼ|
- Can set coefficients to exactly zero (feature selection)
- Useful when many irrelevant features

**L2 Regularization (Ridge):**
- Adds penalty: λ · Σβⱼ²
- Shrinks coefficients toward zero
- Prevents overfitting, handles multicollinearity

**Elastic Net:**
- Combines L1 and L2: λ₁·Σ|βⱼ| + λ₂·Σβⱼ²
- Balance between feature selection and coefficient shrinkage

**Advantages:**
- Probabilistic output (not just class) - enables risk assessment
- Fast training and prediction
- Interpretable coefficients (odds ratios, log-odds)
- No hyperparameters (basic version, but regularization adds them)
- Works well for linearly separable data
- Provides confidence scores for predictions
- Handles both numerical and categorical features (with encoding)

**Limitations:**
- Assumes linear decision boundary (in log-odds space)
- Requires feature scaling for gradient descent (not for solvers like L-BFGS)
- Can't capture complex non-linear patterns without feature engineering
- Sensitive to outliers (though less than linear regression)
- May struggle with highly imbalanced classes (need class weights or sampling)
- Assumes independence of observations`,
					CodeExamples: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate binary classification data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_informative=2, n_clusters_per_class=1, random_state=42)

# Scale features (important for gradient descent)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Method 1: Using sklearn
model = LogisticRegression()
model.fit(X_scaled, y)

y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

print(f"Sklearn - Accuracy: {accuracy_score(y, y_pred):.2f}")
print(f"Log Loss: {log_loss(y, y_proba):.4f}")
print(f"Coefficients: {model.coef_[0]}, Intercept: {model.intercept_[0]:.2f}")

# Manual sigmoid function
def sigmoid(z):
    """Sigmoid function: σ(z) = 1/(1+e^(-z))"""
    # Clip z to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Visualize sigmoid function
z_values = np.linspace(-10, 10, 100)
probabilities = sigmoid(z_values)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(z_values, probabilities)
plt.axvline(x=0, color='r', linestyle='--', label='Decision boundary (z=0)')
plt.axhline(y=0.5, color='r', linestyle='--')
plt.xlabel('z (logit)')
plt.ylabel('Probability P(y=1)')
plt.title('Sigmoid Function')
plt.grid(True)
plt.legend()

# Method 2: Manual Gradient Descent Implementation
def logistic_regression_gd(X, y, learning_rate=0.1, iterations=1000):
    """
    Logistic regression using gradient descent.
    Implements the algorithm step-by-step.
    """
    m, n = X.shape
    # Add intercept term
    X_with_intercept = np.column_stack([np.ones(m), X])
    
    # Initialize coefficients
    beta = np.zeros(n + 1)
    cost_history = []
    
    for i in range(iterations):
        # Forward pass: Calculate predictions
        z = X_with_intercept @ beta  # Linear combination
        p = sigmoid(z)  # Probabilities
        
        # Calculate cost (log loss)
        cost = -(1/m) * np.sum(y * np.log(p + 1e-15) + (1-y) * np.log(1-p + 1e-15))
        cost_history.append(cost)
        
        # Backward pass: Calculate gradients
        # Gradient: (1/m) * X^T * (predictions - actual)
        gradient = (1/m) * X_with_intercept.T @ (p - y)
        
        # Update coefficients
        beta = beta - learning_rate * gradient
        
        # Check convergence
        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < 1e-6:
            print(f"Converged after {i+1} iterations")
            break
    
    return beta, cost_history

beta_gd, costs = logistic_regression_gd(X_scaled, y, learning_rate=0.1, iterations=1000)
print(f"\nGradient Descent - Intercept: {beta_gd[0]:.2f}, Coefficients: {beta_gd[1:]}")
print(f"Final cost: {costs[-1]:.4f}")

# Plot cost convergence
plt.subplot(1, 3, 2)
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost (Log Loss)')
plt.title('Gradient Descent Convergence')
plt.grid(True)

# Decision Boundary Visualization
def plot_decision_boundary_detailed(X, y, model, beta=None):
    """Plot decision boundary with probability contours."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get predictions for grid
    if beta is not None:
        # Manual prediction
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_scaled = scaler.transform(grid_points)
        grid_with_intercept = np.column_stack([np.ones(len(grid_scaled)), grid_scaled])
        z_grid = grid_with_intercept @ beta
        Z_proba = sigmoid(z_grid).reshape(xx.shape)
        Z_pred = (Z_proba > 0.5).astype(int)
    else:
        # Sklearn prediction
        Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        Z_pred = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.subplot(1, 3, 3)
    # Plot probability contours
    contour = plt.contourf(xx, yy, Z_proba, levels=20, cmap='RdYlBu', alpha=0.6)
    plt.colorbar(contour, label='Probability P(y=1)')
    
    # Plot decision boundary (where probability = 0.5)
    plt.contour(xx, yy, Z_proba, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary & Probability Contours')
    plt.legend(['Decision Boundary (p=0.5)'])
    plt.tight_layout()
    plt.show()

plot_decision_boundary_detailed(X, y, model, beta_gd)

# Coefficient Interpretation
print("\n=== Coefficient Interpretation ===")
print(f"Intercept (β₀): {beta_gd[0]:.2f}")
print(f"  When all features = 0, log-odds = {beta_gd[0]:.2f}")
print(f"  Odds ratio = e^{beta_gd[0]:.2f} = {np.exp(beta_gd[0]):.2f}")
print(f"  Probability = {sigmoid(beta_gd[0]):.2f}")

for i, coef in enumerate(beta_gd[1:]):
    print(f"\nCoefficient β{i+1}: {coef:.2f}")
    print(f"  One-unit increase in feature {i+1} changes log-odds by {coef:.2f}")
    print(f"  Odds multiply by e^{coef:.2f} = {np.exp(coef):.2f}")
    print(f"  Odds increase by {(np.exp(coef) - 1) * 100:.1f}%")

# Maximum Likelihood Estimation demonstration
print("\n=== Maximum Likelihood Estimation ===")
def log_likelihood(X, y, beta):
    """Calculate log-likelihood for given coefficients."""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    z = X_with_intercept @ beta
    p = sigmoid(z)
    # Log-likelihood: sum of log probabilities
    ll = np.sum(y * np.log(p + 1e-15) + (1-y) * np.log(1-p + 1e-15))
    return ll

ll_optimal = log_likelihood(X_scaled, y, beta_gd)
print(f"Log-likelihood with optimal coefficients: {ll_optimal:.2f}")

# Try different coefficients to show MLE finds maximum
beta_random = np.random.randn(3) * 0.1
ll_random = log_likelihood(X_scaled, y, beta_random)
print(f"Log-likelihood with random coefficients: {ll_random:.2f}")
print(f"MLE finds coefficients that maximize log-likelihood!")

# Multiclass Logistic Regression
from sklearn.datasets import load_iris
iris = load_iris()
X_multi, y_multi = iris.data, iris.target

# Scale features
scaler_multi = StandardScaler()
X_multi_scaled = scaler_multi.fit_transform(X_multi)

# One-vs-Rest
model_ovr = LogisticRegression(multi_class='ovr', solver='lbfgs')
model_ovr.fit(X_multi_scaled, y_multi)
y_pred_ovr = model_ovr.predict(X_multi_scaled)
print(f"\nOne-vs-Rest Accuracy: {accuracy_score(y_multi, y_pred_ovr):.2f}")

# Multinomial (Softmax)
model_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model_multi.fit(X_multi_scaled, y_multi)
y_pred_multi = model_multi.predict(X_multi_scaled)
print(f"Multinomial (Softmax) Accuracy: {accuracy_score(y_multi, y_pred_multi):.2f}")

# Regularization example
print("\n=== Regularization ===")
model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
model_l1.fit(X_scaled, y)
print(f"L1 Regularization - Coefficients: {model_l1.coef_[0]}")
print(f"  Some coefficients may be exactly zero (feature selection)")

model_l2 = LogisticRegression(penalty='l2', C=1.0)
model_l2.fit(X_scaled, y)
print(f"L2 Regularization - Coefficients: {model_l2.coef_[0]}")
print(f"  Coefficients shrunk toward zero (but not exactly zero)")`,
				},
				{
					Title: "Decision Trees",
					Content: `Decision Trees are intuitive, tree-based models that make decisions by asking a series of yes/no questions.

**How Decision Trees Work:**
1. **Root Node**: Start with all data
2. **Split**: Choose feature and threshold that best separates classes
3. **Recurse**: Repeat for each subset
4. **Leaf Nodes**: Final predictions (class or value)

**Detailed Splitting Algorithm:**

**Step-by-Step Process:**
1. **For each feature**:
   a. Sort feature values
   b. Consider all possible split points (midpoints between consecutive values)
   c. For each split point:
      - Split data: left (≤ threshold), right (> threshold)
      - Calculate impurity for left and right nodes
      - Calculate weighted average impurity
      - Calculate information gain (or Gini gain)
   d. Find best split point for this feature
2. **Compare all features**: Choose feature with best information gain
3. **Split node**: Create left and right child nodes
4. **Recurse**: Repeat for each child node

**Example Walkthrough:**
Suppose we have data with feature "age" and target "buys_product" (yes/no):
- Age: [20, 25, 30, 35, 40, 45], Labels: [0, 0, 1, 1, 1, 1]
- Consider split at age=27.5 (midpoint between 25 and 30):
  - Left: ages ≤ 27.5 → [20, 25] → labels [0, 0] → pure (Gini=0)
  - Right: ages > 27.5 → [30, 35, 40, 45] → labels [1, 1, 1, 1] → pure (Gini=0)
  - Weighted Gini: (2/6)×0 + (4/6)×0 = 0
  - Information Gain = Gini(parent) - Weighted Gini = high gain!

**Splitting Criteria:**

**For Classification:**

**1. Gini Impurity:**
Measures probability of misclassification if we randomly assign labels according to class distribution.

**Formula**: Gini = 1 - Σ(pᵢ)² where pᵢ is proportion of class i

**Properties:**
- Range: [0, 0.5] for binary classification
- Gini = 0: Pure node (all same class) - perfect!
- Gini = 0.5: Maximum impurity (50/50 split)
- Gini = 1 - (p² + (1-p)²) = 2p(1-p) for binary classification

**Example Calculation:**
- Node with [3 class A, 2 class B]: p_A = 0.6, p_B = 0.4
- Gini = 1 - (0.6² + 0.4²) = 1 - (0.36 + 0.16) = 0.48
- Pure node [5 class A]: Gini = 1 - 1² = 0

**2. Entropy:**
Measures disorder/uncertainty in the node.

**Formula**: Entropy = -Σ(pᵢ·log₂(pᵢ))

**Properties:**
- Range: [0, log₂(K)] where K is number of classes
- Entropy = 0: Pure node (no uncertainty)
- Entropy = log₂(K): Maximum uncertainty (uniform distribution)
- For binary: max entropy = log₂(2) = 1

**Example Calculation:**
- Node with [3 class A, 2 class B]: p_A = 0.6, p_B = 0.4
- Entropy = -(0.6×log₂(0.6) + 0.4×log₂(0.4))
- Entropy = -(0.6×(-0.737) + 0.4×(-1.322)) = 0.971

**3. Information Gain:**
Measures reduction in entropy (or impurity) after split.

**Formula**: IG = Entropy(parent) - Σ(nᵢ/n) × Entropy(childᵢ)

Where:
- n: samples in parent node
- nᵢ: samples in child node i

**Interpretation:**
- High information gain: Split creates purer child nodes
- Choose split with maximum information gain
- Information gain ≥ 0 (entropy never increases with more information)

**Worked Example - Information Gain:**
Parent node: [3 class A, 3 class B] → Entropy = 1.0 (maximum)

Split 1: Feature X ≤ 5
- Left: [3 A, 1 B] → Entropy = -(0.75×log₂(0.75) + 0.25×log₂(0.25)) = 0.811
- Right: [0 A, 2 B] → Entropy = 0 (pure)
- Weighted: (4/6)×0.811 + (2/6)×0 = 0.541
- Information Gain: 1.0 - 0.541 = 0.459

Split 2: Feature Y ≤ 3
- Left: [2 A, 2 B] → Entropy = 1.0
- Right: [1 A, 1 B] → Entropy = 1.0
- Weighted: (4/6)×1.0 + (2/6)×1.0 = 1.0
- Information Gain: 1.0 - 1.0 = 0

**Choose Split 1** (higher information gain)!

**For Regression:**
- **MSE (Mean Squared Error)**: Minimize variance in leaf nodes
  - Split that minimizes weighted MSE of child nodes
  - MSE = (1/n)Σ(yᵢ - ȳ)² where ȳ is mean of node
- **MAE (Mean Absolute Error)**: Alternative splitting criterion
  - More robust to outliers than MSE

**Tree Construction Algorithm:**

**Recursive Algorithm:**
    function build_tree(data, depth, max_depth, min_samples):
        if stopping_criterion(data, depth, max_depth, min_samples):
            return leaf_node(predict(data))
        
        best_split = find_best_split(data)
        left_data, right_data = split(data, best_split)
        
        left_child = build_tree(left_data, depth+1, max_depth, min_samples)
        right_child = build_tree(right_data, depth+1, max_depth, min_samples)
        
        return internal_node(best_split, left_child, right_child)

**Stopping Criteria:**
1. **Node is pure**: All samples have same class (Gini=0, Entropy=0)
2. **Maximum depth reached**: Prevents overfitting
3. **Minimum samples per leaf**: Ensures statistical significance
4. **Minimum samples to split**: Prevents splits on too few samples
5. **No improvement**: Information gain below threshold
6. **All features used**: No more features to split on

**Why Greedy Algorithm Works:**
- **Greedy**: Makes locally optimal choice at each step
- **Not globally optimal**: May miss better tree structure
- **Why it works**: Information gain is "submodular" - greedy is near-optimal
- **Efficient**: O(n·m·log(n)) where n=samples, m=features
- **Alternative**: Exhaustive search is exponential - computationally infeasible

**Pruning Methods:**

**1. Pre-Pruning (Early Stopping):**
Stop tree growth before it becomes too complex.

**Methods:**
- Maximum depth: Limit tree depth
- Minimum samples per leaf: Require minimum samples in leaf
- Minimum samples to split: Require minimum samples to consider split
- Maximum features: Consider only subset of features per split
- Minimum information gain: Only split if gain exceeds threshold

**Pros**: Fast, simple
**Cons**: May stop too early, miss important splits

**2. Post-Pruning (Cost-Complexity Pruning):**
Grow full tree, then prune branches that don't improve validation performance.

**Process:**
1. Grow tree to maximum depth
2. Calculate cost-complexity: Cost = Error + α × Complexity
   - α: Complexity parameter (tuning hyperparameter)
   - Complexity: Number of leaf nodes
3. Prune branches that don't reduce cost-complexity
4. Use validation set to choose best α

**Pruning Algorithm:**
- Start from leaves, work upward
- For each internal node:
  - Calculate error if we keep subtree
  - Calculate error if we replace subtree with leaf
  - Prune if leaf has lower cost-complexity

**Pros**: Better generalization, finds optimal tree size
**Cons**: More computationally expensive, needs validation set

**Feature Importance Calculation:**

**Method 1: Gini Importance (Scikit-learn default):**
Importance of feature = Σ (n_t / N) × (Gini_t - Gini_left - Gini_right)

Where:
- n_t: samples in node t
- N: total samples
- Gini_t: Gini impurity of node t
- Gini_left, Gini_right: Gini of child nodes

**Interpretation:**
- Sum over all nodes that use this feature
- Higher importance = feature used more often and creates better splits
- Normalized to sum to 1.0

**Method 2: Permutation Importance:**
1. Train model and get baseline performance
2. For each feature:
   - Shuffle feature values (breaks relationship with target)
   - Measure performance drop
   - Importance = performance drop
3. Features with large drops are more important

**Handling Missing Values:**

**Method 1: Surrogate Splits:**
- Find best split using available features
- For samples with missing values, use surrogate split (next best split)
- Can use multiple surrogates

**Method 2: Default Direction:**
- Send missing values to most common child node
- Or to child with most samples

**Method 3: Imputation:**
- Fill missing values before training
- Use mean/median for numerical, mode for categorical

**Handling Categorical Variables:**

**Method 1: Binary Encoding:**
- For binary categories: 0/1 encoding works directly

**Method 2: One-Hot Encoding:**
- Create binary features for each category
- Can lead to many features

**Method 3: Ordinal Encoding:**
- If categories have order, encode as integers
- Tree can find optimal thresholds

**Method 4: Target Encoding:**
- Encode category as mean target value
- Can be very effective but risk of overfitting

**Advantages:**
- Highly interpretable (visual tree structure, easy to explain)
- No feature scaling needed (splits are threshold-based)
- Handles non-linear relationships (can capture complex patterns)
- Handles mixed data types (numerical and categorical)
- Feature importance available (identifies important features)
- Can handle missing values (with surrogate splits)
- Fast prediction (O(depth) time complexity)
- No assumptions about data distribution

**Limitations:**
- Prone to overfitting (can memorize training data)
- Unstable (small data changes → different tree structure)
- Greedy algorithm (may miss global optimum)
- Biased toward features with more levels (more split opportunities)
- Can create biased trees (if classes are imbalanced)
- Doesn't extrapolate well (predictions bounded by training data range)
- Can be memory intensive (stores entire tree structure)`,
					CodeExamples: `import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
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

y_pred = tree_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Manual Gini Impurity Calculation
def gini_impurity(y):
    """Calculate Gini impurity for a node."""
    if len(y) == 0:
        return 0
    counts = np.bincount(y)
    proportions = counts / len(y)
    return 1 - np.sum(proportions ** 2)

def entropy(y):
    """Calculate entropy for a node."""
    if len(y) == 0:
        return 0
    counts = np.bincount(y)
    proportions = counts[counts > 0] / len(y)
    return -np.sum(proportions * np.log2(proportions))

# Example calculations
pure_node = np.array([0, 0, 0, 0])
mixed_node = np.array([0, 0, 1, 1])
balanced_node = np.array([0, 0, 0, 1, 1, 1])

print("\n=== Impurity Measures ===")
print(f"Pure node [0,0,0,0]:")
print(f"  Gini: {gini_impurity(pure_node):.3f}, Entropy: {entropy(pure_node):.3f}")

print(f"\nMixed node [0,0,1,1]:")
print(f"  Gini: {gini_impurity(mixed_node):.3f}, Entropy: {entropy(mixed_node):.3f}")

print(f"\nBalanced node [0,0,0,1,1,1]:")
print(f"  Gini: {gini_impurity(balanced_node):.3f}, Entropy: {entropy(balanced_node):.3f}")

# Information Gain Calculation Example
def information_gain(parent, left, right):
    """Calculate information gain from a split."""
    n_parent = len(parent)
    n_left = len(left)
    n_right = len(right)
    
    entropy_parent = entropy(parent)
    entropy_left = entropy(left)
    entropy_right = entropy(right)
    
    # Weighted average entropy of children
    weighted_entropy = (n_left / n_parent) * entropy_left + (n_right / n_parent) * entropy_right
    
    # Information gain
    ig = entropy_parent - weighted_entropy
    return ig

# Example: Calculate information gain for a split
parent_labels = np.array([0, 0, 0, 1, 1, 1])  # 3 of each class
left_labels = np.array([0, 0, 1])  # Split creates this distribution
right_labels = np.array([0, 1, 1])  # And this distribution

ig = information_gain(parent_labels, left_labels, right_labels)
print(f"\n=== Information Gain Example ===")
print(f"Parent entropy: {entropy(parent_labels):.3f}")
print(f"Left entropy: {entropy(left_labels):.3f}")
print(f"Right entropy: {entropy(right_labels):.3f}")
print(f"Information Gain: {ig:.3f}")

# Better split example
left_better = np.array([0, 0, 0])  # Pure node
right_better = np.array([1, 1, 1])  # Pure node
ig_better = information_gain(parent_labels, left_better, right_better)
print(f"\nBetter split (creates pure nodes):")
print(f"  Information Gain: {ig_better:.3f} (much higher!)")

# Feature Importance Explanation
print(f"\n=== Feature Importances ===")
print("Feature importance is calculated as:")
print("  Sum over all nodes using feature: (n_samples_in_node / n_samples_total) × impurity_reduction")
for i, importance in enumerate(tree_clf.feature_importances_):
    print(f"{iris.feature_names[i]}: {importance:.3f}")

# Pruning Demonstration
print(f"\n=== Pruning Comparison ===")
# Unpruned (deep) tree
tree_deep = DecisionTreeClassifier(max_depth=20, min_samples_split=2, random_state=42)
tree_deep.fit(X_train, y_train)
train_acc_deep = accuracy_score(y_train, tree_deep.predict(X_train))
test_acc_deep = accuracy_score(y_test, tree_deep.predict(X_test))

# Pruned (shallow) tree
tree_pruned = DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)
tree_pruned.fit(X_train, y_train)
train_acc_pruned = accuracy_score(y_train, tree_pruned.predict(X_train))
test_acc_pruned = accuracy_score(y_test, tree_pruned.predict(X_test))

print(f"Deep tree (max_depth=20):")
print(f"  Train accuracy: {train_acc_deep:.3f}, Test accuracy: {test_acc_deep:.3f}")
print(f"  Depth: {tree_deep.get_depth()}, Leaves: {tree_deep.get_n_leaves()}")
print(f"  Gap: {train_acc_deep - test_acc_deep:.3f} (overfitting!)")

print(f"\nPruned tree (max_depth=3, min_samples_split=10):")
print(f"  Train accuracy: {train_acc_pruned:.3f}, Test accuracy: {test_acc_pruned:.3f}")
print(f"  Depth: {tree_pruned.get_depth()}, Leaves: {tree_pruned.get_n_leaves()}")
print(f"  Gap: {train_acc_pruned - test_acc_pruned:.3f} (better generalization)")

# Cost-Complexity Pruning (Post-pruning)
from sklearn.tree import DecisionTreeClassifier
tree_full = DecisionTreeClassifier(random_state=42)
tree_full.fit(X_train, y_train)

# Get cost-complexity pruning path
path = tree_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

print(f"\n=== Cost-Complexity Pruning Path ===")
print(f"Number of alphas: {len(ccp_alphas)}")
print(f"Alpha range: [{ccp_alphas.min():.4f}, {ccp_alphas.max():.4f}]")

# Try different alpha values
alphas_to_try = [ccp_alphas[0], ccp_alphas[len(ccp_alphas)//2], ccp_alphas[-1]]
for alpha in alphas_to_try:
    tree_alpha = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    tree_alpha.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, tree_alpha.predict(X_test))
    print(f"Alpha={alpha:.4f}: Depth={tree_alpha.get_depth()}, Leaves={tree_alpha.get_n_leaves()}, Test Acc={test_acc:.3f}")

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(tree_clf, feature_names=iris.feature_names, 
          class_names=iris.target_names, filled=True, fontsize=10)
plt.title("Decision Tree Visualization (max_depth=3)")
plt.show()

# Regression Tree
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
tree_reg = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
tree_reg.fit(X_reg, y_reg)

y_pred_reg = tree_reg.predict(X_reg)
print(f"\n=== Regression Tree ===")
print(f"R² Score: {tree_reg.score(X_reg, y_reg):.3f}")
print(f"Depth: {tree_reg.get_depth()}, Leaves: {tree_reg.get_n_leaves()}")

# Tree Statistics
print(f"\n=== Tree Statistics ===")
print(f"Classification Tree:")
print(f"  Depth: {tree_clf.get_depth()}")
print(f"  Number of Leaves: {tree_clf.get_n_leaves()}")
print(f"  Number of Nodes: {tree_clf.tree_.node_count}")

# Understanding how tree makes predictions
print(f"\n=== Prediction Process ===")
sample = X_test[0:1]
prediction = tree_clf.predict(sample)[0]
proba = tree_clf.predict_proba(sample)[0]

print(f"Sample: {sample[0]}")
print(f"Predicted class: {prediction} ({iris.target_names[prediction]})")
print(f"Class probabilities: {proba}")
print(f"\nTree decision path:")
decision_path = tree_clf.decision_path(sample)
node_indicator = decision_path.toarray()[0]
leaf_id = tree_clf.apply(sample)[0]

print(f"  Nodes visited: {np.where(node_indicator == 1)[0]}")
print(f"  Leaf node: {leaf_id}")

# Feature importance calculation details
print(f"\n=== Feature Importance Details ===")
tree_structure = tree_clf.tree_
feature_importance = np.zeros(X_train.shape[1])

for node_id in range(tree_structure.node_count):
    if tree_structure.children_left[node_id] != tree_structure.children_right[node_id]:
        # Internal node
        feature = tree_structure.feature[node_id]
        n_samples = tree_structure.n_node_samples[node_id]
        impurity = tree_structure.impurity[node_id]
        
        left_child = tree_structure.children_left[node_id]
        right_child = tree_structure.children_right[node_id]
        n_left = tree_structure.n_node_samples[left_child]
        n_right = tree_structure.n_node_samples[right_child]
        impurity_left = tree_structure.impurity[left_child]
        impurity_right = tree_structure.impurity[right_child]
        
        # Weighted impurity reduction
        impurity_reduction = impurity - (n_left/n_samples)*impurity_left - (n_right/n_samples)*impurity_right
        feature_importance[feature] += (n_samples / len(X_train)) * impurity_reduction

print("Manual calculation of feature importance:")
for i, imp in enumerate(feature_importance):
    print(f"  {iris.feature_names[i]}: {imp:.4f}")
print("\nSklearn feature importance:")
for i, imp in enumerate(tree_clf.feature_importances_):
    print(f"  {iris.feature_names[i]}: {imp:.4f}")`,
				},
				{
					Title: "Model Evaluation Basics",
					Content: `Evaluating model performance is crucial to understand how well your model will work on new, unseen data.

**Why Evaluation Matters:**
- **Detect Overfitting**: High train accuracy but low test accuracy
- **Compare Models**: Which algorithm performs best?
- **Tune Hyperparameters**: Find optimal settings
- **Estimate Generalization**: How well will model work on new data?
- **Debug Models**: Identify failure modes and improve

**Train-Test Split:**

**Purpose:**
Separate data into training (to learn) and testing (to evaluate) sets.

**Typical Split:**
- **Training Set**: 70-80% of data - used to train the model
- **Test Set**: 20-30% of data - used for final evaluation only
- **Validation Set**: Sometimes split training further (60/20/20)

**Why Separate?**
- **Overfitting Detection**: Model may memorize training data
- **Unbiased Evaluation**: Test set simulates real-world performance
- **Model Selection**: Compare different models fairly

**Critical Rule:**
- **Never** look at test set during training
- **Never** tune hyperparameters on test set
- **Never** use test set for feature selection
- Test set should only be used for final evaluation

**Stratified Split:**
For classification, maintain class distribution in train/test splits.
- Prevents one class from being underrepresented in test set
- Important for imbalanced datasets

**Cross-Validation - Detailed Mechanics:**

**K-Fold Cross-Validation Process:**

**Step-by-Step:**
1. **Split data into K folds** (typically K=5 or K=10)
   - Each fold has approximately n/K samples
   - Folds should be non-overlapping
2. **For each fold i (i = 1 to K):**
   a. Use fold i as validation set
   b. Use remaining K-1 folds as training set
   c. Train model on training folds
   d. Evaluate on validation fold
   e. Record performance score
3. **Average scores** across all K folds
4. **Report**: Mean ± Standard Deviation

**Visual Example (5-Fold CV):**
    Fold 1: [Train: F2,F3,F4,F5] [Val: F1] -> Score1
    Fold 2: [Train: F1,F3,F4,F5] [Val: F2] -> Score2
    Fold 3: [Train: F1,F2,F4,F5] [Val: F3] -> Score3
    Fold 4: [Train: F1,F2,F3,F5] [Val: F4] -> Score4
    Fold 5: [Train: F1,F2,F3,F4] [Val: F5] -> Score5

    Final Score = (Score1 + Score2 + Score3 + Score4 + Score5) / 5

**Why Cross-Validation?**
- **More Robust**: Uses all data for both training and validation
- **Reduces Variance**: Multiple evaluations reduce randomness
- **Better Estimate**: More reliable than single train-test split
- **Data Efficiency**: Especially important for small datasets

**Stratified K-Fold:**
- Maintains class distribution in each fold
- Important for imbalanced classes
- Ensures each fold represents population

**Leave-One-Out CV (LOOCV):**
- K = n (number of samples)
- Each sample is validation set once
- Most thorough but computationally expensive
- Use for very small datasets

**Confusion Matrix - Detailed Guide:**

**Structure:**
                    Predicted
                  Negative  Positive
    Actual Negative   TN      FP
           Positive   FN      TP

**Components:**
- **TP (True Positive)**: Correctly predicted positive
  - Example: Predicted spam, actually spam
- **TN (True Negative)**: Correctly predicted negative
  - Example: Predicted not spam, actually not spam
- **FP (False Positive)**: Incorrectly predicted positive (Type I error)
  - Example: Predicted spam, actually not spam
  - **Cost**: Unnecessary action taken
- **FN (False Negative)**: Incorrectly predicted negative (Type II error)
  - Example: Predicted not spam, actually spam
  - **Cost**: Missed important case

**Interpretation Guide:**
- **Diagonal (TP, TN)**: Correct predictions - want these high
- **Off-diagonal (FP, FN)**: Errors - want these low
- **Row sums**: Actual class distribution
- **Column sums**: Predicted class distribution

**Classification Metrics - Detailed:**

**1. Accuracy:**
**Formula**: Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Interpretation:**
- Overall proportion of correct predictions
- Range: [0, 1], higher is better
- **When to use**: Balanced classes, equal cost of errors

**Limitations:**
- **Misleading with imbalanced classes**:
  - Example: 95% class A, 5% class B
  - Naive classifier predicting always A gets 95% accuracy!
  - But fails to identify any class B samples

**2. Precision:**
**Formula**: Precision = TP / (TP + FP)

**Interpretation:**
- Of all positive predictions, how many are correct?
- "When I say positive, how often am I right?"
- Range: [0, 1], higher is better

**When to prioritize:**
- **False positives are costly**:
  - Spam detection: Don't want to mark real emails as spam
  - Medical diagnosis: Don't want false alarms
  - Fraud detection: Don't want to block legitimate transactions

**3. Recall (Sensitivity, True Positive Rate):**
**Formula**: Recall = TP / (TP + FN)

**Interpretation:**
- Of all actual positives, how many did we catch?
- "Of all positives, how many did I find?"
- Range: [0, 1], higher is better

**When to prioritize:**
- **False negatives are costly**:
  - Disease detection: Don't want to miss sick patients
  - Security: Don't want to miss threats
  - Search: Don't want to miss relevant results

**4. Specificity (True Negative Rate):**
**Formula**: Specificity = TN / (TN + FP)

**Interpretation:**
- Of all actual negatives, how many did we correctly identify?
- Complement of False Positive Rate
- Range: [0, 1], higher is better

**5. F1 Score:**
**Formula**: F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Interpretation:**
- Harmonic mean of precision and recall
- Balances both metrics equally
- Range: [0, 1], higher is better
- **When to use**: Need balance between precision and recall

**Why Harmonic Mean?**
- Harmonic mean penalizes extreme values
- If precision=1, recall=0 → F1=0 (not good!)
- Arithmetic mean would give 0.5 (misleading)

**6. Fβ Score:**
**Formula**: Fβ = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)

**Interpretation:**
- Weighted F1 score
- β > 1: Emphasize recall (find more positives)
- β < 1: Emphasize precision (fewer false positives)
- β = 1: Standard F1 score

**ROC Curve - Step-by-Step Construction:**

**ROC (Receiver Operating Characteristic) Curve:**
Plots True Positive Rate (TPR) vs False Positive Rate (FPR) at different classification thresholds.

**Construction Process:**
1. **Get probability predictions** (not just class predictions)
2. **Vary threshold** from 0 to 1:
   - Threshold = 0: Predict all as positive (TPR=1, FPR=1)
   - Threshold = 1: Predict all as negative (TPR=0, FPR=0)
   - Intermediate thresholds: Different TPR/FPR pairs
3. **For each threshold**:
   - Calculate TPR = TP / (TP + FN)
   - Calculate FPR = FP / (FP + TN)
   - Plot point (FPR, TPR)
4. **Connect points** to form ROC curve

**AUC (Area Under Curve):**
- Area under ROC curve
- Range: [0, 1]
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: Random classifier (diagonal line)
- **AUC < 0.5**: Worse than random (flip predictions!)

**Interpretation:**
- **AUC = 0.9**: 90% chance model ranks random positive higher than random negative
- **Threshold-independent**: Doesn't depend on chosen threshold
- **Useful for**: Comparing models, threshold selection

**Precision-Recall Curve:**
- Plots Precision vs Recall at different thresholds
- **Better than ROC for imbalanced classes**
- **AUC-PR**: Area under precision-recall curve
- Focuses on positive class performance

**Regression Metrics - Detailed:**

**1. MSE (Mean Squared Error):**
**Formula**: MSE = (1/n) Σ(yᵢ - ŷᵢ)²

**Properties:**
- Penalizes large errors more (squared term)
- Units: Same as target squared (harder to interpret)
- Sensitive to outliers
- Differentiable (good for optimization)

**2. RMSE (Root Mean Squared Error):**
**Formula**: RMSE = √MSE = √[(1/n) Σ(yᵢ - ŷᵢ)²]

**Properties:**
- Same units as target (more interpretable)
- Still sensitive to outliers
- Common default metric for regression

**Interpretation:**
- "On average, predictions are off by RMSE units"
- Example: RMSE = 5 means average error is 5 units

**3. MAE (Mean Absolute Error):**
**Formula**: MAE = (1/n) Σ|yᵢ - ŷᵢ|

**Properties:**
- Less sensitive to outliers than MSE/RMSE
- Same units as target
- Robust to outliers
- Not differentiable at zero (harder to optimize)

**When to use:**
- Outliers are problematic
- Want equal weight to all errors
- Need robust metric

**4. R² Score (Coefficient of Determination):**
**Formula**: R² = 1 - (SS_res / SS_tot)
- SS_res = Σ(yᵢ - ŷᵢ)² (sum of squared residuals)
- SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)

**Interpretation:**
- Proportion of variance in target explained by model
- Range: (-∞, 1]
- **R² = 1**: Perfect predictions (all variance explained)
- **R² = 0**: Model performs as well as predicting mean
- **R² < 0**: Model worse than baseline (mean predictor)

**Adjusted R²:**
- Adjusts for number of features
- Penalizes adding unnecessary features
- Formula: R²_adj = 1 - [(1-R²)(n-1)/(n-p-1)] where p = features`,
					CodeExamples: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_regression
import seaborn as sns

# Classification Evaluation
X_clf, y_clf = make_classification(n_samples=1000, n_features=20, 
                                   n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class

# Basic metrics
print("=== Classification Metrics ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Detailed Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n=== Confusion Matrix ===")
print("                Predicted")
print("              Negative  Positive")
print(f"Actual Negative   {tn:4d}    {fp:4d}")
print(f"       Positive   {fn:4d}    {tp:4d}")

print(f"\nBreakdown:")
print(f"  True Positives (TP): {tp} - Correctly predicted positive")
print(f"  True Negatives (TN): {tn} - Correctly predicted negative")
print(f"  False Positives (FP): {fp} - Incorrectly predicted positive")
print(f"  False Negatives (FN): {fn} - Incorrectly predicted negative")

# Calculate metrics manually for understanding
accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity_manual = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nManual Calculations:")
print(f"  Accuracy = (TP+TN)/Total = ({tp}+{tn})/{len(y_test)} = {accuracy_manual:.3f}")
print(f"  Precision = TP/(TP+FP) = {tp}/({tp}+{fp}) = {precision_manual:.3f}")
print(f"  Recall = TP/(TP+FN) = {tp}/({tp}+{fn}) = {recall_manual:.3f}")
print(f"  Specificity = TN/(TN+FP) = {tn}/({tn}+{fp}) = {specificity_manual:.3f}")

# Visualize Confusion Matrix
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')

# Detailed classification report
print(f"\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# ROC Curve Construction
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n=== ROC Curve ===")
print(f"AUC (Area Under Curve): {roc_auc:.3f}")
print(f"  AUC = 1.0: Perfect classifier")
print(f"  AUC = 0.5: Random classifier")
print(f"  AUC > 0.5: Better than random")

# Plot ROC Curve
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity/Recall)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# Precision-Recall Curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

print(f"\n=== Precision-Recall Curve ===")
print(f"PR AUC: {pr_auc:.3f}")
print("  Better metric for imbalanced classes")

# Find optimal threshold (maximize F1)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else 0.5

print(f"\nOptimal Threshold (F1 maximization): {optimal_threshold:.3f}")
print(f"  Optimal F1: {f1_scores[optimal_idx]:.3f}")

# Predictions with optimal threshold
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
print(f"\nWith Optimal Threshold:")
print(f"  Precision: {precision_score(y_test, y_pred_optimal):.3f}")
print(f"  Recall: {recall_score(y_test, y_pred_optimal):.3f}")
print(f"  F1: {f1_score(y_test, y_pred_optimal):.3f}")

# Cross-Validation - Detailed Mechanics
print(f"\n=== Cross-Validation Mechanics ===")

# Standard K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_clf, y_clf, cv=kfold, scoring='accuracy')
print(f"5-Fold CV Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std: {cv_scores.std():.3f}")
print(f"95% CI: {cv_scores.mean():.3f} ± {cv_scores.std() * 1.96:.3f}")

# Stratified K-Fold (for imbalanced classes)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_strat = cross_val_score(model, X_clf, y_clf, cv=skfold, scoring='accuracy')
print(f"\nStratified 5-Fold CV:")
print(f"  Mean: {cv_scores_strat.mean():.3f}, Std: {cv_scores_strat.std():.3f}")

# Manual K-Fold implementation (for understanding)
def manual_kfold_cv(X, y, model, k=5):
    """Manual K-fold cross-validation to understand mechanics."""
    n_samples = len(X)
    fold_size = n_samples // k
    scores = []
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for i in range(k):
        # Define validation indices
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n_samples
        val_idx = indices[val_start:val_end]
        train_idx = np.setdiff1d(indices, val_idx)
        
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train and evaluate
        model.fit(X_train_fold, y_train_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
        print(f"  Fold {i+1}: Train size={len(train_idx)}, Val size={len(val_idx)}, Score={score:.3f}")
    
    return np.array(scores)

print(f"\nManual K-Fold CV:")
manual_scores = manual_kfold_cv(X_clf, y_clf, LogisticRegression(), k=5)
print(f"  Mean: {manual_scores.mean():.3f}, Std: {manual_scores.std():.3f}")

# Regression Evaluation
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)

print(f"\n=== Regression Metrics ===")
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"MSE: {mse:.2f} (units²)")
print(f"RMSE: {rmse:.2f} (units) - Average prediction error")
print(f"MAE: {mae:.2f} (units) - Robust to outliers")
print(f"R² Score: {r2:.3f}")

# R² interpretation
y_mean = np.mean(y_test_reg)
ss_tot = np.sum((y_test_reg - y_mean)**2)
ss_res = np.sum((y_test_reg - y_pred_reg)**2)
r2_manual = 1 - (ss_res / ss_tot)

print(f"\nR² Manual Calculation:")
print(f"  SS_res (sum squared residuals): {ss_res:.2f}")
print(f"  SS_tot (total sum squares): {ss_tot:.2f}")
print(f"  R² = 1 - (SS_res/SS_tot) = {r2_manual:.3f}")
print(f"  Model explains {r2_manual*100:.1f}% of variance")

# Adjusted R²
n = len(y_test_reg)
p = X_test_reg.shape[1]
r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print(f"  Adjusted R²: {r2_adj:.3f} (penalizes {p} features)")

# Cross-Validation for Regression
cv_r2 = cross_val_score(reg_model, X_reg, y_reg, cv=5, scoring='r2')
print(f"\n5-Fold CV R²: {cv_r2.mean():.3f} (+/- {cv_r2.std() * 2:.3f})")

# Compare train vs test performance (overfitting check)
train_r2 = reg_model.score(X_train_reg, y_train_reg)
test_r2 = reg_model.score(X_test_reg, y_test_reg)
print(f"\n=== Overfitting Check ===")
print(f"Train R²: {train_r2:.3f}")
print(f"Test R²: {test_r2:.3f}")
print(f"Gap: {train_r2 - test_r2:.3f}")
if train_r2 - test_r2 > 0.1:
    print("  Warning: Large gap suggests overfitting!")
else:
    print("  Good: Model generalizes well")`,
				},
				{
					Title: "Hyperparameter Tuning & Model Selection",
					Content: `Hyperparameter tuning is crucial for achieving optimal model performance. Unlike model parameters (learned from data), hyperparameters are set before training.

**What are Hyperparameters?**
- Settings that control the learning process
- Not learned from data
- Must be set before training
- Examples: Learning rate, tree depth, regularization strength, number of neighbors

**Why Hyperparameter Tuning Matters:**
- **Performance**: Can significantly improve model accuracy
- **Overfitting Prevention**: Proper regularization prevents overfitting
- **Efficiency**: Optimal settings reduce training time
- **Generalization**: Better hyperparameters improve test performance

**Hyperparameter Tuning Methods:**

**1. Grid Search:**
- Exhaustively search over specified parameter grid
- Try all combinations of hyperparameters
- **Pros**: Guaranteed to find best in grid, simple
- **Cons**: Computationally expensive, limited to grid values
- **Use case**: Small parameter spaces, when compute is available

**2. Random Search:**
- Randomly sample from parameter distributions
- More efficient than grid search
- **Pros**: Faster, can explore wider ranges, often finds better solutions
- **Cons**: May miss optimal values, not exhaustive
- **Use case**: Large parameter spaces, limited compute

**3. Bayesian Optimization:**
- Uses probabilistic model to guide search
- Learns from previous evaluations
- **Pros**: Efficient, learns optimal regions
- **Cons**: More complex, requires tuning itself
- **Use case**: Expensive evaluations, limited budget

**4. Automated ML (AutoML):**
- Automated hyperparameter optimization
- Tools: Optuna, Hyperopt, Auto-sklearn
- **Pros**: Hands-off, often finds good solutions
- **Cons**: Less control, can be slow

**Best Practices:**

**1. Use Cross-Validation:**
- Never tune on test set
- Use nested CV for unbiased evaluation
- Outer CV: Model evaluation
- Inner CV: Hyperparameter tuning

**2. Start with Defaults:**
- Many algorithms have good defaults
- Only tune if performance is insufficient

**3. Focus on Important Hyperparameters:**
- Not all hyperparameters matter equally
- Focus on those with biggest impact
- Example: Learning rate > batch size for neural networks

**4. Use Appropriate Search Space:**
- Start wide, then narrow down
- Use log scale for learning rates (0.001, 0.01, 0.1)
- Use integer ranges for counts (n_estimators: 50-500)

**5. Early Stopping:**
- Stop if no improvement after N iterations
- Saves computation time
- Prevents overfitting to validation set

**6. Track Experiments:**
- Log all hyperparameter combinations
- Track performance metrics
- Use tools: MLflow, Weights & Biases

**Common Hyperparameters by Algorithm:**

**Decision Trees:**
- max_depth: Maximum tree depth
- min_samples_split: Minimum samples to split
- min_samples_leaf: Minimum samples in leaf
- max_features: Features to consider per split

**Random Forest:**
- n_estimators: Number of trees
- max_depth: Tree depth
- max_features: Features per split
- min_samples_split: Minimum samples to split

**Neural Networks:**
- learning_rate: Step size
- batch_size: Samples per update
- hidden_layer_sizes: Architecture
- dropout: Regularization strength
- epochs: Training iterations

**SVM:**
- C: Regularization parameter
- gamma: Kernel coefficient (RBF)
- kernel: Kernel type

**KNN:**
- n_neighbors: Number of neighbors
- weights: Distance weighting
- metric: Distance metric

**Model Selection:**
Choosing between different algorithms or model types.

**Approaches:**
1. **Try Multiple Algorithms**: Compare different model types
2. **Ensemble Methods**: Combine multiple models
3. **Cross-Validation**: Evaluate on multiple folds
4. **Hold-out Test Set**: Final evaluation on unseen data

**Evaluation Strategy:**
1. Split data: Train (60%), Validation (20%), Test (20%)
2. Train multiple models on training set
3. Tune hyperparameters on validation set
4. Evaluate final models on test set
5. Select best model based on test performance`,
					CodeExamples: `from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import numpy as np

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, 
    param_grid, 
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all CPUs
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print(f"Test score: {grid_search.score(X_test, y_test):.3f}")

# 2. Random Search (more efficient)
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20),
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    rf,
    param_distributions,
    n_iter=50,  # Number of random combinations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print(f"\nRandom Search - Best parameters: {random_search.best_params_}")
print(f"Random Search - Best CV score: {random_search.best_score_:.3f}")

# 3. Manual Hyperparameter Tuning
best_score = 0
best_params = None

for n_est in [50, 100, 200]:
    for max_dep in [5, 10, 20]:
        rf_temp = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_dep,
            random_state=42
        )
        scores = cross_val_score(rf_temp, X_train, y_train, cv=5)
        mean_score = scores.mean()
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'n_estimators': n_est, 'max_depth': max_dep}

print(f"\nManual Tuning - Best params: {best_params}")
print(f"Manual Tuning - Best CV score: {best_score:.3f}")

# 4. Learning Curves for Hyperparameter Selection
from sklearn.model_selection import validation_curve

param_range = [5, 10, 15, 20, 25, 30]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, y_train,
    param_name='max_depth',
    param_range=param_range,
    cv=5,
    scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

print(f"\nValidation Curve Results:")
for depth, train_sc, val_sc in zip(param_range, train_mean, val_mean):
    print(f"Depth {depth}: Train={train_sc:.3f}, Val={val_sc:.3f}, Gap={train_sc-val_sc:.3f}")

# Optimal depth: best validation score
optimal_depth_idx = np.argmax(val_mean)
optimal_depth = param_range[optimal_depth_idx]
print(f"\nOptimal max_depth: {optimal_depth}")

# 5. Model Selection: Compare Different Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

print("\nModel Comparison:")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# 6. Nested Cross-Validation (unbiased evaluation)
from sklearn.model_selection import cross_val_score, KFold

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

nested_scores = []
for train_idx, test_idx in outer_cv.split(X):
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    # Inner CV for hyperparameter tuning
    grid_search_inner = GridSearchCV(
        rf, param_grid, cv=inner_cv, scoring='accuracy'
    )
    grid_search_inner.fit(X_train_outer, y_train_outer)
    
    # Evaluate on outer test set
    score = grid_search_inner.score(X_test_outer, y_test_outer)
    nested_scores.append(score)

print(f"\nNested CV Score: {np.mean(nested_scores):.3f} (+/- {np.std(nested_scores) * 2:.3f})")`,
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

**Training Phase (Lazy Learning):**
- **No explicit training**: Simply stores all training examples
- **No model building**: No parameters to learn
- **Memory-based**: All computation deferred until prediction

**Prediction Phase:**
1. **Calculate distances**: Compute distance from query point to all training points
2. **Find K nearest**: Select K training points with smallest distances
3. **Aggregate neighbors**:
   - **Classification**: Majority vote (most common class among K neighbors)
   - **Regression**: Average (mean value of K neighbors)
   - **Weighted**: Can weight by inverse distance (closer neighbors matter more)

**Example Walkthrough:**
Query point: (3, 4)
Training points: [(1,1)→Class A, (2,2)→Class A, (5,5)→Class B, (6,6)→Class B, (3,3)→Class A]

Distances:
- (1,1): √((3-1)² + (4-1)²) = √13 ≈ 3.6
- (2,2): √((3-2)² + (4-2)²) = √5 ≈ 2.2
- (5,5): √((3-5)² + (4-5)²) = √5 ≈ 2.2
- (6,6): √((3-6)² + (4-6)²) = √13 ≈ 3.6
- (3,3): √((3-3)² + (4-3)²) = 1.0

For K=3: Nearest neighbors are (3,3), (2,2), (5,5)
- Classes: [A, A, B]
- Majority vote: Class A

**Distance Metrics - Detailed:**

**1. Euclidean Distance (L2):**
**Formula**: d(x,y) = √Σ(xᵢ - yᵢ)²

**Properties:**
- Straight-line distance in Euclidean space
- Most common default
- Sensitive to feature scale (requires scaling)
- Works well when features are on similar scales

**Geometric Intuition:**
- In 2D: Straight line between two points
- In nD: Generalization of Pythagorean theorem
- Forms hyperspheres around query point

**2. Manhattan Distance (L1, City-Block):**
**Formula**: d(x,y) = Σ|xᵢ - yᵢ|

**Properties:**
- Sum of absolute differences
- Like walking city blocks (can't cut diagonally)
- Less sensitive to outliers than Euclidean
- More robust to irrelevant features

**When to use:**
- Features have different scales
- Want L1 regularization effect
- High-dimensional sparse data

**3. Minkowski Distance (Generalization):**
**Formula**: d(x,y) = (Σ|xᵢ - yᵢ|^p)^(1/p)
- p=1: Manhattan
- p=2: Euclidean
- p=∞: Chebyshev (max difference)

**4. Cosine Similarity:**
**Formula**: cos(θ) = (x·y) / (||x|| × ||y||)

**Properties:**
- Measures angle between vectors, not magnitude
- Range: [-1, 1] (typically [0, 1] for non-negative features)
- Good for high-dimensional data (text, images)
- Normalized (scale-invariant)

**When to use:**
- Text classification (TF-IDF vectors)
- High-dimensional sparse data
- Want to ignore magnitude, focus on direction

**5. Hamming Distance:**
**Formula**: Number of positions where values differ

**Properties:**
- For categorical/binary data
- Count of mismatches
- Example: "cat" vs "bat" → Hamming = 1

**Why Scaling Matters:**

**Problem Example:**
Feature 1: Age (20-80 years)
Feature 2: Income ($20,000-$200,000)

Without scaling:
- Age difference: 10 years → distance contribution: 10
- Income difference: $10,000 → distance contribution: 10,000
- Income dominates! Age is ignored.

With scaling:
- Both features contribute equally
- StandardScaler: (x - mean) / std
- MinMaxScaler: (x - min) / (max - min)

**Curse of Dimensionality:**

**The Problem:**
As dimensions increase, all points become equidistant!

**Why:**
- In high dimensions, volume concentrates in corners
- Distance between any two points becomes similar
- Nearest neighbor becomes meaningless

**Example:**
- 1D: Points spread along line
- 2D: Points spread in plane
- 100D: All points roughly same distance from origin
- Nearest neighbor ≈ random point

**Solutions:**
- Feature selection (reduce dimensions)
- Dimensionality reduction (PCA)
- Use cosine similarity (angle-based, not distance-based)
- Domain-specific distance metrics

**Choosing K - Detailed:**

**Small K (K=1):**
- **Voronoi diagram**: Each training point gets its own region
- **Very flexible**: Can learn any decision boundary
- **High variance**: Sensitive to noise and outliers
- **Low bias**: Can fit training data perfectly
- **Risk**: Overfitting (memorizes training data)

**Large K:**
- **Smoother boundaries**: More robust to noise
- **Lower variance**: More stable predictions
- **Higher bias**: May underfit (too simple)
- **Risk**: Underfitting (misses local patterns)

**Optimal K Selection:**
1. **Cross-validation**: Try K=1 to K=√n, pick best
2. **Elbow method**: Plot accuracy vs K, find elbow
3. **Domain knowledge**: Consider problem characteristics
4. **Odd K**: For binary classification, prevents ties

**Weighted KNN:**
Instead of simple majority vote, weight neighbors by inverse distance:
- Weight = 1 / distance
- Closer neighbors have more influence
- Smooths decision boundary
- Can improve performance

**Computational Complexity:**

**Naive Implementation:**
- Training: O(1) - just store data
- Prediction: O(n×d) where n=samples, d=features
  - Must compute distance to all n points
  - Each distance: O(d) operations

**Optimization Techniques:**

**1. KD-Trees:**
- Spatial data structure for fast nearest neighbor search
- Builds tree partitioning space
- Query time: O(log n) average case
- **Limitation**: Degrades in high dimensions (>20D)

**2. Ball Trees:**
- Similar to KD-trees but uses hyperspheres
- Better for high-dimensional data
- Query time: O(log n) average case

**3. Locality-Sensitive Hashing (LSH):**
- Approximate nearest neighbor search
- Hash similar points to same buckets
- Fast but approximate

**4. Approximate Methods:**
- Sample subset of training data
- Use only nearby region (radius-based)
- Trade accuracy for speed

**Advantages:**
- Simple and intuitive (easy to understand and explain)
- No assumptions about data distribution (non-parametric)
- Works for both classification and regression
- Can learn complex decision boundaries (non-linear)
- No training time (lazy learning - instant setup)
- Naturally handles multi-class problems
- Can provide confidence scores (proportion of neighbors)

**Limitations:**
- Slow prediction (must compute distances to all points - O(n×d))
- Sensitive to irrelevant features (adds noise to distances)
- Requires feature scaling (distances depend on scale)
- Memory intensive (stores all training data)
- Sensitive to curse of dimensionality (performance degrades with dimensions)
- Sensitive to imbalanced classes (may favor majority class)
- No feature importance (can't identify important features)`,
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

**Bayes' Theorem - Derivation:**

Bayes' theorem comes from the definition of conditional probability.

**Step 1: Conditional Probability**
P(A|B) = P(A ∩ B) / P(B)
- Probability of A given B occurred

**Step 2: Joint Probability**
P(A ∩ B) = P(A|B) × P(B) = P(B|A) × P(A)

**Step 3: Rearrange**
P(A|B) = P(B|A) × P(A) / P(B)

**Step 4: Apply to Classification**
P(y|X) = P(X|y) × P(y) / P(X)

**Components:**
- **P(y|X)**: Posterior probability - probability of class y given features X
  - What we want to compute
  - "Given these features, what's the probability of each class?"
- **P(X|y)**: Likelihood - probability of features X given class y
  - "If class is y, how likely are these features?"
  - Learned from training data
- **P(y)**: Prior probability - probability of class y (before seeing features)
  - "How common is class y in general?"
  - Estimated from training data: P(y) = count(y) / total_samples
- **P(X)**: Evidence - probability of features X (normalizing constant)
  - "How common are these features?"
  - Same for all classes, so can be ignored when comparing classes

**Why "Naive"?**

**The Problem:**
Computing P(X|y) = P(x₁, x₂, ..., xₙ|y) requires joint probability distribution.
- For n features, need 2ⁿ probabilities (for binary features)
- Exponential complexity!
- Requires huge amounts of data

**The Naive Assumption:**
Features are conditionally independent given the class:
P(x₁, x₂, ..., xₙ|y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)

**Why This Helps:**
- Instead of 2ⁿ probabilities, need only n×c probabilities (n features, c classes)
- Linear complexity!
- Much less data needed

**When Assumption Breaks:**
- Features are correlated (e.g., height and weight)
- Still works surprisingly well in practice!
- Independence assumption is often "good enough"

**Naive Bayes Classification:**

**Prediction Rule:**
Choose class with highest posterior probability:
ŷ = argmax_y P(y|X) = argmax_y [P(X|y) × P(y) / P(X)]

Since P(X) is same for all classes, we can ignore it:
ŷ = argmax_y [P(X|y) × P(y)]

**Using Naive Assumption:**
ŷ = argmax_y [P(y) × Πᵢ P(xᵢ|y)]

**In Log Space (Numerical Stability):**
ŷ = argmax_y [log P(y) + Σᵢ log P(xᵢ|y)]
- Prevents underflow (multiplying many small probabilities)
- Faster computation (addition vs multiplication)

**Types of Naive Bayes:**

**1. Gaussian Naive Bayes:**
Assumes each feature follows Gaussian (normal) distribution per class.

**Probability Density Function:**
P(xᵢ|y) = (1/√(2πσ²ᵧ)) × exp(-(xᵢ - μᵧ)² / (2σ²ᵧ))

**Parameters to Estimate:**
- μᵧ: Mean of feature i for class y
  - μᵧ = (1/nᵧ) Σ xᵢ where samples belong to class y
- σ²ᵧ: Variance of feature i for class y
  - σ²ᵧ = (1/nᵧ) Σ (xᵢ - μᵧ)²

**When to Use:**
- Continuous/numerical features
- Features approximately normally distributed
- Each feature can be modeled as Gaussian

**2. Multinomial Naive Bayes:**
For count data (word counts, frequencies).

**Probability:**
P(xᵢ|y) = (count(xᵢ, y) + α) / (Σⱼ count(xⱼ, y) + α×|vocabulary|)

**Parameters:**
- Count of feature i in class y
- Total count of all features in class y
- α: Smoothing parameter (Laplace smoothing)

**When to Use:**
- Text classification (word counts)
- Document classification
- Categorical count data

**3. Bernoulli Naive Bayes:**
For binary features (present/absent).

**Probability:**
P(xᵢ=1|y) = (count(xᵢ=1, y) + α) / (nᵧ + 2α)
P(xᵢ=0|y) = 1 - P(xᵢ=1|y)

**When to Use:**
- Binary features (0/1)
- Text classification (word presence, not counts)
- Boolean features

**Laplace Smoothing (Additive Smoothing):**

**The Problem:**
If a feature value never appears in training for a class:
- P(xᵢ|y) = 0
- Entire product becomes 0 (even if other features suggest the class)
- "Zero probability problem"

**The Solution:**
Add small constant α to all counts:
- P(xᵢ|y) = (count(xᵢ, y) + α) / (total_count(y) + α×|values|)
- α = 1: Laplace smoothing
- α < 1: Lidstone smoothing

**Why It Works:**
- Prevents zero probabilities
- Acts as prior (assumes each value occurs α times)
- Smooths probability estimates

**Worked Example:**

**Training Data:**
- Email 1: ["free", "money"] → Spam
- Email 2: ["meeting", "tomorrow"] → Ham
- Email 3: ["free", "click"] → Spam

**Test Email:**
["free", "meeting"]

**Step 1: Calculate Priors**
P(Spam) = 2/3, P(Ham) = 1/3

**Step 2: Calculate Likelihoods (with smoothing α=1)**
P("free"|Spam) = (2+1)/(2+1×2) = 3/4
P("meeting"|Spam) = (0+1)/(2+1×2) = 1/4
P("free"|Ham) = (0+1)/(1+1×2) = 1/3
P("meeting"|Ham) = (1+1)/(1+1×2) = 2/3

**Step 3: Calculate Posteriors**
P(Spam|["free","meeting"]) ∝ P(Spam) × P("free"|Spam) × P("meeting"|Spam)
= (2/3) × (3/4) × (1/4) = 6/48 = 1/8

P(Ham|["free","meeting"]) ∝ P(Ham) × P("free"|Ham) × P("meeting"|Ham)
= (1/3) × (1/3) × (2/3) = 2/27 ≈ 0.074

**Step 4: Predict**
P(Spam) > P(Ham) → Predict Spam

**Advantages:**
- Fast training and prediction (O(n×d) where n=samples, d=features)
- Works well with small datasets (efficient parameter estimation)
- Handles multiple classes naturally (just compute for each class)
- Not sensitive to irrelevant features (independence assumption helps)
- Probabilistic output (provides confidence scores)
- Interpretable (can see which features contribute to prediction)
- Good baseline (often surprisingly effective)

**Limitations:**
- Naive independence assumption (often violated in practice)
- Requires feature independence (may not hold)
- Can be outperformed by more complex models (when data is large)
- Sensitive to feature representation (encoding matters)
- May give overconfident predictions (probabilities not well-calibrated)`,
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
- **Formula**: w·x + b = 0
  - w: weight vector (normal to hyperplane)
  - b: bias term (offset from origin)
  - x: feature vector

**Distance from Point to Hyperplane:**
Distance = |w·x + b| / ||w||

**Support Vectors:**
- Data points closest to the hyperplane
- Define the margin boundaries
- Only these points matter (sparse solution)
- Typically small fraction of training data
- "Hard" examples that are difficult to classify

**Margin:**
- Distance between hyperplane and nearest points
- **Margin = 2 / ||w||** (for linearly separable case)
- Larger margin = better generalization (more robust)
- SVM maximizes this margin

**Why Maximize Margin?**
- **Better Generalization**: Larger margin means more tolerance to noise
- **Robustness**: Small changes in data less likely to change decision boundary
- **Theoretical Guarantee**: Maximum margin minimizes VC dimension (complexity)

**Hard Margin SVM - Mathematical Derivation:**

**Objective:**
Maximize margin = 2 / ||w||
Equivalently: Minimize ||w||² / 2

**Constraints:**
For all training points (xᵢ, yᵢ) where yᵢ ∈ {-1, +1}:
- yᵢ(w·xᵢ + b) ≥ 1
- Ensures all points are on correct side
- Points on margin boundary: yᵢ(w·xᵢ + b) = 1

**Optimization Problem:**
Minimize: (1/2)||w||²
Subject to: yᵢ(w·xᵢ + b) ≥ 1 for all i

**Solution:**
This is a quadratic programming problem.
- Convex optimization (unique global minimum)
- Can be solved using Lagrange multipliers
- Solution depends only on support vectors

**Dual Form (Key Insight):**
The solution can be written as:
w = Σ αᵢyᵢxᵢ (sum over support vectors)
- αᵢ: Lagrange multipliers (non-zero only for support vectors)
- Only support vectors contribute to solution!

**Soft Margin SVM - Handling Non-Separable Data:**

**The Problem:**
Hard margin requires perfect separation (may not exist or overfits).

**The Solution:**
Allow some misclassification with slack variables ξᵢ.

**Modified Objective:**
Minimize: (1/2)||w||² + C·Σξᵢ
Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

**C Parameter:**
- **C**: Regularization parameter (trade-off)
- **Large C**: Penalize errors heavily → smaller margin, fewer errors
  - "Hard margin" behavior
  - Risk of overfitting
- **Small C**: Allow more errors → larger margin, more errors
  - More robust to outliers
  - Better generalization

**Interpretation:**
- C → ∞: Hard margin (no errors allowed)
- C → 0: Maximize margin regardless of errors

**Kernel Trick - Detailed Explanation:**

**The Problem:**
Data may not be linearly separable in original space.

**The Idea:**
Map data to higher-dimensional space where it becomes linearly separable.

**Example:**
- Original space: x = (x₁, x₂)
- Feature space: φ(x) = (x₁², √2·x₁·x₂, x₂²)
- In feature space, data becomes linearly separable!

**The Kernel Trick:**
Instead of explicitly computing φ(x), use kernel function:
K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ)

**Why This Works:**
- SVM only needs dot products: w·x = Σ αᵢyᵢK(xᵢ, x)
- Never need to compute φ(x) explicitly!
- Can work in infinite-dimensional spaces efficiently

**Common Kernels:**

**1. Linear Kernel:**
K(xᵢ, xⱼ) = xᵢ·xⱼ
- No transformation (works in original space)
- Fast, interpretable
- Use when data is linearly separable

**2. Polynomial Kernel:**
K(xᵢ, xⱼ) = (γ·xᵢ·xⱼ + r)ᵈ
- **γ**: Scale parameter
- **r**: Coefficient
- **d**: Degree
- Captures polynomial relationships
- Higher degree = more complex boundaries

**3. RBF (Radial Basis Function) / Gaussian Kernel:**
K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)
- **γ**: Controls influence radius (1/(2σ²))
- Large γ: Narrow influence (complex boundary)
- Small γ: Wide influence (smooth boundary)
- Most popular kernel
- Can approximate any function (universal kernel)

**4. Sigmoid Kernel:**
K(xᵢ, xⱼ) = tanh(γ·xᵢ·xⱼ + r)
- Similar to neural network activation
- Less commonly used

**Kernel Selection:**
- **Linear**: Start here, if works use it (fastest)
- **RBF**: Default choice, works well for most problems
- **Polynomial**: When you know relationship is polynomial
- **Custom**: Domain-specific kernels

**How SVM Works - Step by Step:**

**Training:**
1. **Input**: Training data (xᵢ, yᵢ)
2. **Solve optimization**: Find w, b that maximize margin
3. **Identify support vectors**: Points with αᵢ > 0
4. **Store**: Support vectors, αᵢ, and kernel parameters

**Prediction:**
1. **For new point x**: Compute f(x) = Σ αᵢyᵢK(xᵢ, x) + b
2. **Sign**: Predict class = sign(f(x))
3. **Distance**: |f(x)| gives confidence (distance from hyperplane)

**Support Vector Importance:**
- Only support vectors affect prediction
- All other training points can be removed without changing model
- Makes SVM memory efficient
- Typically 1-10% of training data are support vectors

**Advantages:**
- Effective in high-dimensional spaces (even when d > n)
- Memory efficient (uses support vectors only - sparse solution)
- Versatile (different kernels for different problems)
- Works well with clear margin of separation
- Strong theoretical foundation (statistical learning theory)
- Robust to outliers (with appropriate C)
- Can handle non-linear boundaries (with kernels)

**Limitations:**
- Doesn't perform well with large datasets (training is O(n²) to O(n³))
- Sensitive to feature scaling (distances matter)
- Doesn't provide probability estimates directly (need calibration)
- Black box (hard to interpret, especially with kernels)
- Requires careful tuning (C and kernel parameters)
- Slow training for large datasets
- Memory intensive for large number of support vectors`,
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
