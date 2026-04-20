package machinelearning

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMachineLearningModules([]problems.CourseModule{
		{
			ID:          2520,
			Title:       "Feature Engineering and ML System Design",
			Description: "Master feature engineering techniques, data preprocessing, feature stores, model monitoring, A/B testing, and end-to-end ML system architecture.",
			Order:       20,
			Lessons: []problems.Lesson{
				{
					Title: "Feature Engineering Preprocessing Monitoring and ML Systems",
					Content: `Feature engineering is the process of transforming raw data into features that better represent the underlying problem, improving model performance. Combined with proper system design, it forms the backbone of production ML.

**Numerical Features:**

Scaling:
  StandardScaler (Z-score): x' = (x - mean) / std
    Zero mean, unit variance
    Sensitive to outliers
    Use: Linear models, SVMs, neural networks
    
  MinMaxScaler: x' = (x - min) / (max - min)
    Scale to [0, 1]
    Preserves zero entries (sparse data)
    Use: Neural networks with bounded activations
    
  RobustScaler: x' = (x - median) / IQR
    Uses median and interquartile range
    Robust to outliers
    
  MaxAbsScaler: x' = x / max(|x|)
    Scale to [-1, 1]
    Preserves sparsity
    
  Log transform: x' = log(x + 1)
    Right-skewed distributions → more normal
    Revenue, counts, prices
    
  Box-Cox: Power transform to normality
    Requires positive values
    Automatically finds optimal power parameter
    
  Yeo-Johnson: Like Box-Cox but handles negatives

Binning/Discretization:
  Equal-width: Fixed bin size
  Equal-frequency: Same number of samples per bin
  Custom: Domain-specific thresholds
  Quantile: Based on percentiles
  
  When to bin:
    Non-linear relationships
    Reduce noise
    Handle outliers
    Create interaction with other features

Polynomial Features:
  x1, x2 → x1, x2, x1², x2², x1*x2
  Captures non-linear relationships
  Curse of dimensionality (feature explosion)
  Usually degree 2-3 max

**Categorical Features:**

Label Encoding:
  Map categories to integers: cat→0, dog→1, bird→2
  Use: Ordinal features (small, medium, large)
  Problem: Implies ordering for non-ordinal features

One-Hot Encoding:
  Binary column per category
  [1,0,0], [0,1,0], [0,0,1]
  Use: Non-ordinal, low cardinality
  Problem: High dimensional for many categories

Target Encoding:
  Replace category with mean of target variable
  Smoothing: blend with global mean using regularization
  Use: High cardinality
  Risk: Data leakage (use cross-validation encoding)

Frequency Encoding:
  Replace category with frequency count
  Simple but informative

Binary Encoding:
  Encode integer label as binary
  3 categories: 0→00, 1→01, 2→10
  Fewer columns than one-hot

Embedding:
  Learned dense vector per category
  Use: Very high cardinality (user IDs, product IDs)
  Train with neural network
  Transfer to other models

**Text Features:**

Bag of Words (BoW):
  Count of each word in document
  Sparse, high dimensional
  Ignores order and context

TF-IDF:
  TF: Term frequency in document
  IDF: Log(N / doc_freq) — penalize common words
  TF-IDF: TF × IDF
  Better than raw counts

N-grams:
  Unigrams: Single words
  Bigrams: Two-word sequences
  Trigrams: Three-word sequences
  Captures local context

Word Embeddings:
  Word2Vec: Skip-gram or CBOW
  GloVe: Global co-occurrence statistics
  FastText: Subword information
  
  Document embedding:
    Average of word embeddings
    Weighted average (TF-IDF weights)
    Doc2Vec

Transformer Embeddings:
  BERT: Bidirectional context
  Sentence-BERT: Sentence-level embeddings
  Fine-tuned for specific domains

**Time Series Features:**

Lag features:
  Value at previous time steps: y(t-1), y(t-2), ...
  Most important features for time series

Rolling statistics:
  Rolling mean, std, min, max over window
  Captures trends and volatility

Calendar features:
  Hour, day of week, month, quarter
  Is holiday, is weekend, day of year
  Encode cyclically: sin/cos for periodic features

Difference features:
  y(t) - y(t-1): First difference (removes trend)
  y(t) - y(t-7): Seasonal difference (removes weekly pattern)

Exponential moving average:
  More weight to recent observations
  Smoothing parameter alpha

**Missing Data Handling:**

Detection:
  MCAR: Missing Completely At Random
  MAR: Missing At Random (depends on other features)
  MNAR: Missing Not At Random (depends on missing value)

Imputation:
  Mean/Median/Mode: Simple, baseline
  K-NN imputation: Find similar samples, use their values
  MICE: Multiple imputation by chained equations
  Model-based: Predict missing values with a model
  
  Indicator: Add binary column "is_missing"
  Useful when missingness itself is informative

Deletion:
  Listwise: Remove entire row (if small fraction)
  Feature: Remove feature (if > 50% missing)

**Feature Selection:**

Filter methods (before model):
  Correlation: Remove highly correlated features (r > 0.95)
  Variance threshold: Remove low-variance features
  Mutual information: Measure dependency with target
  Chi-squared test: For categorical features

Wrapper methods (use model):
  Forward selection: Add features one by one
  Backward elimination: Remove features one by one
  Recursive Feature Elimination (RFE): Remove least important

Embedded methods (during training):
  L1 regularization (Lasso): Drives coefficients to zero
  Tree-based importance: Feature importance from trees
  Attention weights: From neural network attention

**Data Preprocessing Pipeline:**

Typical order:
  1. Handle missing values
  2. Remove duplicates
  3. Handle outliers
  4. Encode categorical features
  5. Scale numerical features
  6. Feature selection
  7. Split train/validation/test

Class Imbalance:
  Undersampling: Remove majority class samples
  Oversampling: Duplicate minority class samples
  SMOTE: Synthesize minority class samples
  Class weights: Weight loss function inversely to class frequency
  Focal loss: Down-weight easy examples

Data Leakage Prevention:
  Never fit scalers/encoders on test data
  Time-based splits for temporal data
  Target encoding with cross-validation
  No future information in features

**Feature Stores:**

Architecture:
  Offline store: Historical features for training (batch)
  Online store: Low-latency features for inference (real-time)
  Feature registry: Metadata, lineage, documentation
  
  Examples: Feast, Tecton, Hopsworks, Vertex AI Feature Store

Benefits:
  Feature reuse across teams and models
  Consistent features between training and serving
  Point-in-time correct features (avoid leakage)
  Feature versioning and lineage

**Model Monitoring:**

Data drift:
  Feature distributions change over time
  Detection: KL divergence, KS test, PSI (Population Stability Index)
  
  PSI = Sum((actual% - expected%) × ln(actual% / expected%))
    PSI < 0.1: No drift
    PSI 0.1-0.2: Moderate drift
    PSI > 0.2: Significant drift

Concept drift:
  Relationship between features and target changes
  Types:
    Sudden: Abrupt change (new policy, pandemic)
    Gradual: Slow transition
    Incremental: Continuous small changes
    Recurring: Seasonal patterns
  
  Detection: Monitor prediction quality metrics

Model performance:
  Track accuracy, precision, recall, F1 over time
  Alert when metrics degrade below threshold
  Sliding window evaluation

Prediction monitoring:
  Distribution of predictions
  Confidence scores
  Latency and throughput
  Error rate

**A/B Testing for ML:**

Setup:
  Control: Current model (baseline)
  Treatment: New model
  Random assignment of users
  
  Metrics:
    Primary: Business metric (conversion, revenue)
    Secondary: Model metrics (accuracy, latency)
    Guardrail: Safety metrics (error rate, latency p99)

Statistical significance:
  p-value < 0.05 (95% confidence)
  Effect size: Practical significance
  Power analysis: Determine sample size needed
  
  Duration: Run long enough for:
    Statistical significance
    Day-of-week effects
    User learning/novelty effects

Progressive rollout:
  1% → 5% → 25% → 50% → 100%
  Monitor metrics at each stage
  Automatic rollback on metric degradation

Shadow mode:
  Run new model alongside production
  Compare predictions without serving
  Validate before switching

**End-to-End ML System Architecture:**

Training pipeline:
  Data ingestion → Validation → Preprocessing → Feature engineering →
  Model training → Evaluation → Model registry → Deployment

Serving patterns:
  Batch prediction: Run periodically, store results
  Online prediction: Real-time API
  Streaming: Process events as they arrive
  Embedded: Model in client (mobile, edge)

Model registry:
  Version models with metadata
  Track lineage (data, code, hyperparameters)
  Promote through stages (dev → staging → production)
  Examples: MLflow, W&B, DVC

Infrastructure:
  Training: GPU clusters (Kubernetes, SageMaker)
  Serving: Model servers (TorchServe, Triton, TF Serving)
  Feature store: Online/offline features
  Monitoring: Prometheus, Grafana, Evidently
  Orchestration: Airflow, Kubeflow, Prefect`,
					CodeExamples: `# Feature Engineering and ML System Design Examples

import math
import random
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from abc import ABC, abstractmethod

# ============================================================
# Feature Transformers
# ============================================================

class StandardScaler:
    """Z-score normalization."""
    
    def __init__(self):
        self.mean: List[float] = []
        self.std: List[float] = []
    
    def fit(self, X: List[List[float]]):
        n = len(X)
        d = len(X[0])
        self.mean = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
        self.std = [
            math.sqrt(sum((X[i][j] - self.mean[j]) ** 2 for i in range(n)) / n)
            for j in range(d)]
        # Avoid division by zero
        self.std = [s if s > 0 else 1.0 for s in self.std]
    
    def transform(self, X: List[List[float]]) -> List[List[float]]:
        return [[(x[j] - self.mean[j]) / self.std[j]
                 for j in range(len(x))]
                for x in X]
    
    def fit_transform(self, X: List[List[float]]) -> List[List[float]]:
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: List[List[float]]) -> List[List[float]]:
        return [[x[j] * self.std[j] + self.mean[j]
                 for j in range(len(x))]
                for x in X]


class MinMaxScaler:
    """Min-max normalization to [0, 1]."""
    
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.min_vals: List[float] = []
        self.max_vals: List[float] = []
    
    def fit(self, X: List[List[float]]):
        d = len(X[0])
        self.min_vals = [min(x[j] for x in X) for j in range(d)]
        self.max_vals = [max(x[j] for x in X) for j in range(d)]
    
    def transform(self, X: List[List[float]]) -> List[List[float]]:
        lo, hi = self.feature_range
        result = []
        for x in X:
            row = []
            for j in range(len(x)):
                range_val = self.max_vals[j] - self.min_vals[j]
                if range_val == 0:
                    row.append(0.0)
                else:
                    scaled = (x[j] - self.min_vals[j]) / range_val
                    row.append(lo + scaled * (hi - lo))
            result.append(row)
        return result
    
    def fit_transform(self, X: List[List[float]]) -> List[List[float]]:
        self.fit(X)
        return self.transform(X)


class RobustScaler:
    """Scale using median and interquartile range."""
    
    def __init__(self):
        self.median: List[float] = []
        self.iqr: List[float] = []
    
    def fit(self, X: List[List[float]]):
        d = len(X[0])
        for j in range(d):
            values = sorted(x[j] for x in X)
            n = len(values)
            self.median.append(values[n // 2])
            q1 = values[n // 4]
            q3 = values[3 * n // 4]
            iqr = q3 - q1
            self.iqr.append(iqr if iqr > 0 else 1.0)
    
    def transform(self, X: List[List[float]]) -> List[List[float]]:
        return [[(x[j] - self.median[j]) / self.iqr[j]
                 for j in range(len(x))]
                for x in X]


# ============================================================
# Categorical Encoders
# ============================================================

class OneHotEncoder:
    """One-hot encoding for categorical features."""
    
    def __init__(self):
        self.categories: Dict[int, List[Any]] = {}
    
    def fit(self, X: List[List[Any]], categorical_cols: List[int]):
        for col in categorical_cols:
            self.categories[col] = sorted(set(x[col] for x in X))
    
    def transform(self, X: List[List[Any]],
                  categorical_cols: List[int]) -> List[List[float]]:
        result = []
        for x in X:
            row = []
            for j in range(len(x)):
                if j in categorical_cols:
                    cats = self.categories.get(j, [])
                    one_hot = [1.0 if x[j] == cat else 0.0 for cat in cats]
                    row.extend(one_hot)
                else:
                    row.append(float(x[j]))
            result.append(row)
        return result


class TargetEncoder:
    """Target encoding with smoothing."""
    
    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.encodings: Dict[int, Dict[Any, float]] = {}
        self.global_mean: float = 0.0
    
    def fit(self, X: List[List[Any]], y: List[float],
            categorical_cols: List[int]):
        self.global_mean = sum(y) / len(y)
        
        for col in categorical_cols:
            category_stats: Dict[Any, Tuple[float, int]] = {}
            for i in range(len(X)):
                cat = X[i][col]
                if cat not in category_stats:
                    category_stats[cat] = (0.0, 0)
                total, count = category_stats[cat]
                category_stats[cat] = (total + y[i], count + 1)
            
            self.encodings[col] = {}
            for cat, (total, count) in category_stats.items():
                cat_mean = total / count
                # Smoothing: blend with global mean
                weight = count / (count + self.smoothing)
                self.encodings[col][cat] = (
                    weight * cat_mean + (1 - weight) * self.global_mean)
    
    def transform(self, X: List[List[Any]],
                  categorical_cols: List[int]) -> List[List[float]]:
        result = []
        for x in X:
            row = []
            for j in range(len(x)):
                if j in categorical_cols:
                    encoded = self.encodings.get(j, {}).get(
                        x[j], self.global_mean)
                    row.append(encoded)
                else:
                    row.append(float(x[j]))
            result.append(row)
        return result


class FrequencyEncoder:
    """Encode categories by their frequency."""
    
    def __init__(self):
        self.frequencies: Dict[int, Dict[Any, float]] = {}
    
    def fit(self, X: List[List[Any]], categorical_cols: List[int]):
        n = len(X)
        for col in categorical_cols:
            counter = Counter(x[col] for x in X)
            self.frequencies[col] = {k: v / n for k, v in counter.items()}
    
    def transform(self, X: List[List[Any]],
                  categorical_cols: List[int]) -> List[List[float]]:
        result = []
        for x in X:
            row = []
            for j in range(len(x)):
                if j in categorical_cols:
                    row.append(self.frequencies.get(j, {}).get(x[j], 0.0))
                else:
                    row.append(float(x[j]))
            result.append(row)
        return result


# ============================================================
# Missing Value Imputer
# ============================================================

class SimpleImputer:
    """Impute missing values."""
    
    def __init__(self, strategy: str = "mean", fill_value: float = 0.0,
                 add_indicator: bool = False):
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        self.fill_values: List[float] = []
    
    def fit(self, X: List[List[Optional[float]]]):
        d = len(X[0])
        self.fill_values = []
        
        for j in range(d):
            values = [x[j] for x in X if x[j] is not None]
            
            if self.strategy == "mean":
                self.fill_values.append(
                    sum(values) / len(values) if values else 0.0)
            elif self.strategy == "median":
                sorted_vals = sorted(values)
                mid = len(sorted_vals) // 2
                self.fill_values.append(
                    sorted_vals[mid] if values else 0.0)
            elif self.strategy == "most_frequent":
                self.fill_values.append(
                    Counter(values).most_common(1)[0][0] if values else 0.0)
            else:
                self.fill_values.append(self.fill_value)
    
    def transform(self, X: List[List[Optional[float]]]) -> List[List[float]]:
        result = []
        for x in X:
            row = []
            indicator = []
            for j in range(len(x)):
                if x[j] is None:
                    row.append(self.fill_values[j] if j < len(self.fill_values) else 0.0)
                    indicator.append(1.0)
                else:
                    row.append(float(x[j]))
                    indicator.append(0.0)
            if self.add_indicator:
                row.extend(indicator)
            result.append(row)
        return result


# ============================================================
# Feature Selection
# ============================================================

class VarianceThreshold:
    """Remove low-variance features."""
    
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.variances: List[float] = []
        self.selected_indices: List[int] = []
    
    def fit(self, X: List[List[float]]):
        n = len(X)
        d = len(X[0])
        
        for j in range(d):
            mean = sum(X[i][j] for i in range(n)) / n
            var = sum((X[i][j] - mean) ** 2 for i in range(n)) / n
            self.variances.append(var)
        
        self.selected_indices = [
            j for j, v in enumerate(self.variances) if v > self.threshold]
    
    def transform(self, X: List[List[float]]) -> List[List[float]]:
        return [[x[j] for j in self.selected_indices] for x in X]


class CorrelationFilter:
    """Remove highly correlated features."""
    
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.selected_indices: List[int] = []
    
    def fit(self, X: List[List[float]]):
        d = len(X[0])
        n = len(X)
        
        # Compute correlation matrix
        means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
        stds = [math.sqrt(sum((X[i][j] - means[j]) ** 2
                              for i in range(n)) / n)
                for j in range(d)]
        
        to_remove: Set[int] = set()
        
        for i in range(d):
            if i in to_remove or stds[i] == 0:
                continue
            for j in range(i + 1, d):
                if j in to_remove or stds[j] == 0:
                    continue
                
                corr = sum((X[k][i] - means[i]) * (X[k][j] - means[j])
                          for k in range(n)) / (n * stds[i] * stds[j])
                
                if abs(corr) > self.threshold:
                    to_remove.add(j)
        
        self.selected_indices = [j for j in range(d) if j not in to_remove]
    
    def transform(self, X: List[List[float]]) -> List[List[float]]:
        return [[x[j] for j in self.selected_indices] for x in X]


# ============================================================
# Data Drift Detection
# ============================================================

class PSICalculator:
    """Population Stability Index for drift detection."""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_edges: List[float] = []
        self.expected_percents: List[float] = []
    
    def fit(self, reference: List[float]):
        sorted_ref = sorted(reference)
        n = len(sorted_ref)
        
        self.bin_edges = [sorted_ref[0] - 1e-6]
        for i in range(1, self.n_bins):
            idx = int(i * n / self.n_bins)
            self.bin_edges.append(sorted_ref[min(idx, n - 1)])
        self.bin_edges.append(sorted_ref[-1] + 1e-6)
        
        counts = self._bin_counts(reference)
        total = sum(counts)
        self.expected_percents = [max(c / total, 1e-6) for c in counts]
    
    def calculate(self, actual: List[float]) -> float:
        counts = self._bin_counts(actual)
        total = sum(counts)
        actual_percents = [max(c / total, 1e-6) for c in counts]
        
        psi = 0.0
        for a, e in zip(actual_percents, self.expected_percents):
            psi += (a - e) * math.log(a / e)
        
        return psi
    
    def _bin_counts(self, data: List[float]) -> List[int]:
        counts = [0] * self.n_bins
        for value in data:
            for i in range(self.n_bins):
                if value <= self.bin_edges[i + 1]:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1
        return counts
    
    @staticmethod
    def interpret(psi: float) -> str:
        if psi < 0.1:
            return "No significant drift"
        elif psi < 0.2:
            return "Moderate drift — monitor closely"
        else:
            return "Significant drift — action needed"


# ============================================================
# Model Registry
# ============================================================

@dataclass
class ModelVersion:
    model_id: str
    version: int
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    stage: str = "development"
    created_at: float = 0
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


class ModelRegistry:
    """Simple model registry."""
    
    def __init__(self):
        self._models: Dict[str, List[ModelVersion]] = defaultdict(list)
    
    def register(self, model_id: str, metrics: Dict[str, float],
                 parameters: Dict[str, Any],
                 description: str = "",
                 tags: Dict[str, str] = None) -> ModelVersion:
        versions = self._models[model_id]
        version = len(versions) + 1
        
        mv = ModelVersion(
            model_id=model_id,
            version=version,
            metrics=metrics,
            parameters=parameters,
            description=description,
            tags=tags or {},
            created_at=random.random() * 1000,
        )
        versions.append(mv)
        return mv
    
    def promote(self, model_id: str, version: int, stage: str):
        mv = self.get_version(model_id, version)
        if mv:
            # Demote current version in target stage
            for v in self._models.get(model_id, []):
                if v.stage == stage:
                    v.stage = "archived"
            mv.stage = stage
    
    def get_version(self, model_id: str,
                    version: int) -> Optional[ModelVersion]:
        versions = self._models.get(model_id, [])
        for v in versions:
            if v.version == version:
                return v
        return None
    
    def get_latest(self, model_id: str,
                   stage: str = None) -> Optional[ModelVersion]:
        versions = self._models.get(model_id, [])
        if stage:
            versions = [v for v in versions if v.stage == stage]
        return versions[-1] if versions else None
    
    def list_versions(self, model_id: str) -> List[Dict]:
        return [
            {
                "version": v.version,
                "stage": v.stage,
                "metrics": v.metrics,
                "description": v.description,
            }
            for v in self._models.get(model_id, [])
        ]
    
    def compare(self, model_id: str, v1: int,
                v2: int) -> Dict[str, Any]:
        mv1 = self.get_version(model_id, v1)
        mv2 = self.get_version(model_id, v2)
        
        if not mv1 or not mv2:
            return {}
        
        metric_diff = {}
        all_metrics = set(mv1.metrics.keys()) | set(mv2.metrics.keys())
        for m in all_metrics:
            val1 = mv1.metrics.get(m, 0)
            val2 = mv2.metrics.get(m, 0)
            metric_diff[m] = {
                f"v{v1}": val1,
                f"v{v2}": val2,
                "diff": val2 - val1,
                "pct_change": ((val2 - val1) / abs(val1) * 100
                              if val1 != 0 else 0),
            }
        
        return {"model_id": model_id, "metrics": metric_diff}


# ============================================================
# ML Pipeline
# ============================================================

class PipelineStep(ABC):
    """Abstract pipeline step."""
    
    @abstractmethod
    def fit(self, X: Any, y: Any = None) -> Any:
        pass
    
    @abstractmethod
    def transform(self, X: Any) -> Any:
        pass
    
    def fit_transform(self, X: Any, y: Any = None) -> Any:
        self.fit(X, y)
        return self.transform(X)


class Pipeline:
    """ML preprocessing pipeline."""
    
    def __init__(self, steps: List[Tuple[str, PipelineStep]]):
        self.steps = steps
    
    def fit(self, X: Any, y: Any = None):
        current = X
        for name, step in self.steps:
            current = step.fit_transform(current, y)
        return self
    
    def transform(self, X: Any) -> Any:
        current = X
        for name, step in self.steps:
            current = step.transform(current)
        return current
    
    def fit_transform(self, X: Any, y: Any = None) -> Any:
        self.fit(X, y)
        return self.transform(X)


# ============================================================
# Cross-Validation
# ============================================================

def k_fold_split(n: int, k: int = 5,
                 shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """Generate K-fold cross-validation splits."""
    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)
    
    fold_size = n // k
    folds = []
    
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        test_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, test_idx))
    
    return folds


def stratified_k_fold(y: List[int], k: int = 5) -> List[
        Tuple[List[int], List[int]]]:
    """Stratified K-fold preserving class distribution."""
    class_indices: Dict[int, List[int]] = defaultdict(list)
    for i, label in enumerate(y):
        class_indices[label].append(i)
    
    for indices in class_indices.values():
        random.shuffle(indices)
    
    folds = [[] for _ in range(k)]
    for label, indices in class_indices.items():
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)
    
    result = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = [idx for j in range(k) if j != i for idx in folds[j]]
        result.append((train_idx, test_idx))
    
    return result


# ============================================================
# Metrics
# ============================================================

def confusion_matrix(y_true: List[int],
                     y_pred: List[int]) -> Dict[str, int]:
    """Binary confusion matrix."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def classification_report(y_true: List[int],
                          y_pred: List[int]) -> Dict[str, float]:
    """Classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": len(y_true),
    }


def regression_metrics(y_true: List[float],
                       y_pred: List[float]) -> Dict[str, float]:
    """Regression metrics."""
    n = len(y_true)
    
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)
    
    mean_true = sum(y_true) / n
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}`,
				},
			},
		},
	})
}
