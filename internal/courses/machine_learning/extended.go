package machinelearning

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMachineLearningModules([]problems.CourseModule{
		{
			ID:          2515,
			Title:       "ML in Production (MLOps)",
			Description: "Learn how to deploy, monitor, and maintain machine learning models in production environments, covering ML pipelines, model serving, monitoring, and MLOps best practices.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "ML Pipeline Architecture",
					Content: `Moving ML models from notebook to production is where most ML projects fail. MLOps bridges this gap with engineering practices for the ML lifecycle.

**The ML Lifecycle:**

` + "```" + `
Data Collection → Data Processing → Feature Engineering → Model Training
       ↑                                                         ↓
    Monitoring ← Model Serving ← Model Validation ← Experiment Tracking
` + "```" + `

**Why ML in Production is Hard:**
1. **Data drift**: Input data distribution changes over time
2. **Model decay**: Model accuracy degrades as patterns change
3. **Reproducibility**: Same code + same data should give same results
4. **Scale**: Training on a laptop ≠ training on a cluster
5. **Serving latency**: Inference must be fast (< 100ms for real-time)

**ML Pipeline Components:**

**1. Data Pipeline:**
- Collect, clean, validate, and version data
- Feature stores for reusable features
- Data validation (schema checks, distribution checks)
- Tools: Apache Airflow, Prefect, dbt, Great Expectations

**2. Training Pipeline:**
- Experiment tracking (parameters, metrics, artifacts)
- Hyperparameter tuning (grid search, Bayesian optimization)
- Distributed training for large models
- Tools: MLflow, Weights & Biases, Optuna, Ray

**3. Model Registry:**
- Version control for models (like Git for code)
- Model metadata (who trained it, on what data, performance metrics)
- Approval workflow (staging → production)
- Tools: MLflow Model Registry, Amazon SageMaker, Vertex AI

**4. Serving Infrastructure:**
- Real-time inference (REST/gRPC API)
- Batch inference (process large datasets)
- Edge inference (on-device)
- Tools: TensorFlow Serving, TorchServe, Triton, BentoML, Ray Serve

**5. Monitoring:**
- Model performance metrics (accuracy, latency, throughput)
- Data drift detection (input distribution changes)
- Concept drift detection (relationship between features and target changes)
- Tools: Evidently AI, WhyLabs, Amazon SageMaker Monitor

**MLOps Maturity Levels:**

| Level | Description | Automation |
|-------|-------------|-----------|
| 0 | Manual: Notebooks, manual deployment | None |
| 1 | ML Pipeline: Automated training, manual deployment | Training |
| 2 | CI/CD: Automated training AND deployment | Full pipeline |
| 3 | Continuous Training: Auto-retrain on new data or drift | Everything |

**Feature Stores:**
Centralized repository for ML features with:
- **Online store**: Low-latency serving (Redis, DynamoDB)
- **Offline store**: Historical data for training (S3, BigQuery)
- **Feature computation**: Transform raw data into features
- **Feature sharing**: Teams reuse features across models
- Tools: Feast, Tecton, Amazon SageMaker Feature Store`,
					CodeExamples: `# MLflow: Experiment Tracking + Model Registry
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Start experiment
mlflow.set_experiment("customer_churn")

with mlflow.start_run():
    # Log parameters
    params = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5}
    mlflow.log_params(params)

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Log metrics
    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # Log model to registry
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="churn_predictor"
    )

# Model Serving with FastAPI
from fastapi import FastAPI
import mlflow
import numpy as np

app = FastAPI()
model = mlflow.sklearn.load_model("models:/churn_predictor/Production")

@app.post("/predict")
async def predict(features: list[float]):
    prediction = model.predict(np.array([features]))
    probability = model.predict_proba(np.array([features]))
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0].max())
    }

# Data Validation with Great Expectations
import great_expectations as gx

context = gx.get_context()
validator = context.sources.pandas_default.read_csv("data.csv")

# Define expectations
validator.expect_column_values_to_not_be_null("customer_id")
validator.expect_column_values_to_be_between("age", 18, 120)
validator.expect_column_values_to_be_in_set("status", ["active", "churned"])
validator.expect_column_mean_to_be_between("monthly_spend", 10, 500)

# Validate
results = validator.validate()
if not results.success:
    raise ValueError("Data validation failed!")`,
				},
				{
					Title: "Model Evaluation and Selection",
					Content: `Choosing the right model and properly evaluating it is more important than tuning hyperparameters. Poor evaluation leads to models that look good on paper but fail in production.

**The Evaluation Framework:**

**1. Train/Validation/Test Split:**
` + "```" + `
Full Dataset (100%)
├── Training Set (70-80%) — Used to train the model
├── Validation Set (10-15%) — Used to tune hyperparameters
└── Test Set (10-15%) — Used ONCE for final evaluation
` + "```" + `

**CRITICAL RULE:** Never use test data for model selection or tuning. It's your final exam — you only look at it once.

**2. Cross-Validation:**
For small datasets, k-fold cross-validation gives more reliable estimates:
- Split data into k folds (typically k=5 or k=10)
- Train on k-1 folds, validate on the remaining fold
- Repeat k times, average the results

**Classification Metrics:**

| Metric | Formula | Use When |
|--------|---------|----------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
| Precision | TP/(TP+FP) | Cost of false positives is high (spam filter) |
| Recall | TP/(TP+FN) | Cost of false negatives is high (cancer detection) |
| F1 Score | 2×(P×R)/(P+R) | Imbalanced classes, need balance of P and R |
| AUC-ROC | Area under ROC curve | Overall model quality, threshold-independent |

**Confusion Matrix:**
` + "```" + `
                Predicted
              Pos    Neg
Actual Pos | TP  |  FN  |  ← Recall = TP/(TP+FN)
Actual Neg | FP  |  TN  |
              ↑
        Precision = TP/(TP+FP)
` + "```" + `

**Regression Metrics:**
- **MSE**: Mean Squared Error (penalizes large errors)
- **RMSE**: Root MSE (same unit as target)
- **MAE**: Mean Absolute Error (robust to outliers)
- **R²**: Coefficient of determination (closer to 1 is better)
- **MAPE**: Mean Absolute Percentage Error (percentage-based)

**Common Evaluation Mistakes:**

1. **Data Leakage**: Training data contains information from the future or test set
   - Example: Using the target variable during feature engineering
   - Fix: All preprocessing must be fit on training data only

2. **Overfitting to Validation Set**: Tuning too many hyperparameters
   - Fix: Use nested cross-validation or separate holdout

3. **Wrong Metric**: Using accuracy on imbalanced data
   - 99% accuracy with 99% negative class = useless model
   - Fix: Use F1, AUC-ROC, or precision/recall

4. **Ignoring Business Context**: Model might be accurate but not useful
   - A model that predicts 0.01% more accurately but takes 10x longer to serve
   - Consider: latency, interpretability, fairness, cost

**Model Selection Guide:**

| Problem Type | Models to Try First | When to Use |
|-------------|-------------------|-------------|
| Tabular data | XGBoost, LightGBM, CatBoost | Most common, structured data |
| Images | CNNs (ResNet, EfficientNet) | Computer vision |
| Text | Transformers (BERT, GPT) | NLP tasks |
| Time series | Prophet, LSTM, XGBoost | Forecasting |
| Small data | Linear/Logistic Regression, SVM | < 1000 samples |
| Anomaly detection | Isolation Forest, Autoencoders | Fraud, defects |`,
					CodeExamples: `import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, mean_squared_error
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Cross-validation comparison
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"{name}: F1 = {scores.mean():.3f} (+/- {scores.std():.3f})")

# Output:
# Logistic Regression: F1 = 0.821 (+/- 0.023)
# Random Forest:       F1 = 0.856 (+/- 0.018)
# Gradient Boosting:   F1 = 0.872 (+/- 0.015)  ← Best

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1, 0.3],
}

grid_search = GridSearchCV(
    GradientBoostingClassifier(),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best F1: {grid_search.best_score_:.3f}")

# Final evaluation on test set (ONLY ONCE!)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nFinal Test Results:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")`,
				},
				{
					Title: "Deep Learning Fundamentals",
					Content: `Deep learning uses neural networks with multiple layers to learn hierarchical representations from data. Understanding the fundamentals helps you know when (and when NOT) to use deep learning.

**When to Use Deep Learning:**
- Large amounts of data (> 10K-100K samples)
- Unstructured data (images, text, audio, video)
- Complex patterns that domain experts can't easily specify
- Available compute resources (GPUs/TPUs)

**When NOT to Use Deep Learning:**
- Small datasets (< 1K samples) — XGBoost usually wins
- Tabular data — gradient boosting usually matches or beats neural nets
- Need interpretability — neural nets are black boxes
- Limited compute — training can take hours/days/weeks

**Neural Network Building Blocks:**

**1. Neuron (Perceptron):**
` + "```" + `
Inputs: x1, x2, ..., xn
Weights: w1, w2, ..., wn
Bias: b

Output = activation(w1*x1 + w2*x2 + ... + wn*xn + b)
       = activation(dot(w, x) + b)
` + "```" + `

**2. Activation Functions:**
- **ReLU**: max(0, x) — most common, fast, avoids vanishing gradients
- **Sigmoid**: 1/(1+e^(-x)) — output between 0 and 1 (probability)
- **Softmax**: Normalized exponentials — output probability distribution (classification)
- **Tanh**: (e^x - e^(-x))/(e^x + e^(-x)) — output between -1 and 1

**3. Loss Functions:**
- **Cross-Entropy**: Classification (how wrong are the probabilities?)
- **MSE**: Regression (how far are predictions from actuals?)
- **Binary Cross-Entropy**: Binary classification

**4. Optimization (Backpropagation + Gradient Descent):**
1. Forward pass: Compute predictions
2. Compute loss: How wrong are we?
3. Backward pass: Compute gradients of loss w.r.t. each weight
4. Update weights: weights = weights - learning_rate * gradients

**Optimizers:**
- **SGD**: Simple, requires manual learning rate tuning
- **Adam**: Adaptive learning rate (recommended default)
- **AdamW**: Adam with weight decay (regularization)

**Common Architectures:**

**Feedforward (Dense/MLP):**
- Fully connected layers
- Good for: Tabular data, simple classification
- Not good for: Large inputs (images, text)

**CNN (Convolutional Neural Network):**
- Convolutional layers extract spatial features
- Good for: Images, spatial data
- Key: Feature maps, pooling, stride

**RNN/LSTM (Recurrent Neural Network):**
- Process sequential data
- Good for: Time series (though Transformers often better now)
- Problem: Vanishing gradients for long sequences

**Transformer:**
- Attention mechanism (no recurrence)
- Good for: NLP, vision (ViT), multimodal
- Key innovation: Self-attention allows processing all positions in parallel
- Foundation of: GPT, BERT, T5, ChatGPT

**Practical Tips:**
1. **Start simple**: Logistic regression → small network → larger network
2. **Use pretrained models**: Transfer learning saves massive compute
3. **Regularize**: Dropout, weight decay, early stopping
4. **Batch normalization**: Stabilizes training, allows higher learning rates
5. **Data augmentation**: Artificially expand training data (especially for images)
6. **Learning rate scheduling**: Reduce LR as training progresses`,
					CodeExamples: `# PyTorch: Simple neural network for classification
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define model
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)

# Training loop
model = SimpleNet(input_dim=20, hidden_dim=64, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(100):
    model.train()
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        accuracy = (val_outputs.argmax(1) == y_val).float().mean()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={val_loss:.4f}, acc={accuracy:.4f}")

# Transfer Learning with pretrained model (image classification)
from torchvision import models, transforms

# Load pretrained ResNet (trained on ImageNet)
model = models.resnet50(pretrained=True)

# Freeze all layers except final
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for our task (10 classes)
model.fc = nn.Linear(model.fc.in_features, 10)

# Only train the new final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Fine-tune: After training final layer, unfreeze and train all
for param in model.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR`,
				},
			},
		},
	})
}
