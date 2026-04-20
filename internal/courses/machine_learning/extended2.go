package machinelearning

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMachineLearningModules([]problems.CourseModule{
		{
			ID:          2516,
			Title:       "Deep Learning Foundations",
			Description: "Master neural network architectures, backpropagation, optimization algorithms, regularization techniques, and training strategies for deep learning models.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "Neural Networks Backpropagation and Optimization",
					Content: `Deep learning extends classical machine learning with multi-layered neural networks capable of learning hierarchical representations from data.

**Neural Network Fundamentals:**

Neuron (Perceptron):
  output = activation(sum(weights * inputs) + bias)
  
  Components:
    Inputs: x1, x2, ..., xn
    Weights: w1, w2, ..., wn (learnable parameters)
    Bias: b (learnable threshold)
    Activation: Non-linear function
    Output: Single scalar value

Layers:
  Input layer: Receives raw features
  Hidden layers: Learn intermediate representations
  Output layer: Produces final prediction
  
  Dense (Fully Connected):
    Every neuron connects to all neurons in next layer
    Parameters: weights matrix W (n_in × n_out) + bias vector b
    
  Convolutional:
    Shared weights across spatial dimensions
    Local receptive fields
    Translation invariant
    
  Recurrent:
    Hidden state carries information across time steps
    Weight sharing across time
    
  Attention/Transformer:
    Self-attention mechanism
    Parallel processing of sequences
    Position encoding

**Activation Functions:**

ReLU (Rectified Linear Unit):
  f(x) = max(0, x)
  Pros: Fast, no vanishing gradient for positive values
  Cons: Dead neurons (output 0 for all inputs)
  Use: Default for hidden layers

Leaky ReLU:
  f(x) = x if x > 0, else alpha * x (alpha = 0.01)
  Fixes dead neuron problem
  
ELU (Exponential Linear Unit):
  f(x) = x if x > 0, else alpha * (exp(x) - 1)
  Smooth, pushes mean activations toward zero

GELU (Gaussian Error Linear Unit):
  f(x) = x * Phi(x) where Phi is standard Gaussian CDF
  Used in Transformers (BERT, GPT)
  Smooth approximation of ReLU

Sigmoid:
  f(x) = 1 / (1 + exp(-x))
  Range: (0, 1)
  Use: Binary classification output, gates in LSTM
  Problem: Vanishing gradient for extreme values

Tanh:
  f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
  Range: (-1, 1)
  Zero-centered
  Use: Hidden layers (less common now), LSTM

Softmax:
  f(xi) = exp(xi) / sum(exp(xj))
  Outputs sum to 1 (probability distribution)
  Use: Multi-class classification output

Swish:
  f(x) = x * sigmoid(x)
  Self-gated
  Often better than ReLU

**Backpropagation:**

Forward Pass:
  Input → Layer 1 → ... → Layer N → Output → Loss
  
  At each layer:
    z = W·x + b (linear transformation)
    a = activation(z) (non-linear activation)
    Cache z, a for backward pass

Backward Pass (Chain Rule):
  Compute gradient of loss w.r.t. each parameter
  
  Output layer:
    dL/da_N = derivative of loss w.r.t. output
    
  For each layer (N to 1):
    dL/dz = dL/da * da/dz (activation gradient)
    dL/dW = dL/dz * x^T (weight gradient)
    dL/db = dL/dz (bias gradient)
    dL/dx = W^T * dL/dz (propagate to previous layer)
    
  Update: W = W - learning_rate * dL/dW

Computational Graph:
  DAG of operations
  Forward: Compute values node by node
  Backward: Compute gradients in reverse topological order
  Automatic differentiation (autograd in PyTorch)

Gradient Issues:
  Vanishing gradients:
    Gradients shrink exponentially through layers
    Cause: Sigmoid/tanh saturation, deep networks
    Solutions: ReLU, residual connections, LSTM, normalization
    
  Exploding gradients:  
    Gradients grow exponentially
    Cause: Large weights, long sequences
    Solutions: Gradient clipping, weight initialization, normalization

**Loss Functions:**

Regression:
  MSE (Mean Squared Error): L = mean((y - y_hat)^2)
    Penalizes large errors heavily
    Sensitive to outliers
    
  MAE (Mean Absolute Error): L = mean(|y - y_hat|)
    Robust to outliers
    Not differentiable at 0
    
  Huber Loss: MSE for small errors, MAE for large errors
    Smooth, robust to outliers
    
  Log-Cosh: L = mean(log(cosh(y - y_hat)))
    Similar to Huber but smoother

Classification:
  Binary Cross-Entropy: L = -(y*log(p) + (1-y)*log(1-p))
    For binary classification
    Used with sigmoid output
    
  Categorical Cross-Entropy: L = -sum(y_i * log(p_i))
    For multi-class classification
    Used with softmax output
    
  Focal Loss: L = -alpha * (1-p)^gamma * log(p)
    For class imbalance
    Down-weights easy examples
    gamma=0 reduces to standard cross-entropy

**Optimization Algorithms:**

SGD (Stochastic Gradient Descent):
  w = w - lr * gradient
  Simple but can be slow
  Learning rate is critical
  
SGD with Momentum:
  v = beta * v + gradient
  w = w - lr * v
  Accelerates in consistent gradient direction
  Reduces oscillation
  beta typically 0.9

Nesterov Momentum:
  Look ahead: compute gradient at (w - lr * beta * v)
  Faster convergence than standard momentum

RMSProp:
  s = beta * s + (1-beta) * gradient^2
  w = w - lr * gradient / sqrt(s + epsilon)
  Adapts learning rate per parameter
  Good for non-stationary problems

Adam (Adaptive Moment Estimation):
  m = beta1 * m + (1-beta1) * gradient (first moment)
  v = beta2 * v + (1-beta2) * gradient^2 (second moment)
  m_hat = m / (1 - beta1^t) (bias correction)
  v_hat = v / (1 - beta2^t) (bias correction)
  w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
  
  Default: beta1=0.9, beta2=0.999, epsilon=1e-8
  Most popular optimizer

AdamW (Adam with Weight Decay):
  Decouples weight decay from gradient update
  w = w - lr * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * w)
  Better generalization than Adam
  Default in modern deep learning

LAMB (Layer-wise Adaptive Moments):
  Scales Adam update by ratio of weight norm to update norm
  Enables large batch training
  Used for BERT pretraining

Learning Rate Schedules:
  Step decay: Reduce by factor every N epochs
  Cosine annealing: lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*t/T))
  Warmup + decay: Linear warmup then cosine/linear decay
  One-cycle: Increase then decrease in one cycle
  Reduce on plateau: Reduce when metric stops improving
  
  Warmup:
    Start with small lr, increase linearly
    Prevents early large updates
    Critical for Transformers

**Regularization:**

L1 Regularization (Lasso):
  Loss + lambda * sum(|w|)
  Produces sparse weights (feature selection)
  
L2 Regularization (Ridge / Weight Decay):
  Loss + lambda * sum(w^2)
  Prevents large weights
  Smooth penalty

Dropout:
  Randomly zero out activations during training
  Rate: Fraction of neurons dropped (0.1-0.5)
  Ensemble effect
  Must scale during inference (or inverted dropout)
  
  Variants:
    Spatial dropout: Drop entire feature maps (CNNs)
    DropBlock: Drop contiguous regions
    DropPath: Drop entire residual branches (Transformers)

Batch Normalization:
  Normalize activations to zero mean, unit variance
  Per mini-batch statistics during training
  Running statistics during inference
  Learnable scale (gamma) and shift (beta)
  
  Benefits: Faster training, higher learning rates, regularization
  Placement: After linear, before activation (debated)

Layer Normalization:
  Normalize across features (not batch)
  Independent of batch size
  Used in Transformers, RNNs
  
Group Normalization:
  Divide channels into groups, normalize within each
  Works with small batch sizes
  
Instance Normalization:
  Normalize each sample, each channel independently
  Used in style transfer

Data Augmentation:
  Images: Flip, rotate, crop, color jitter, cutout, mixup
  Text: Synonym replacement, back-translation, random insertion
  Audio: Time stretch, pitch shift, noise injection

Early Stopping:
  Monitor validation loss
  Stop when no improvement for N epochs (patience)
  Restore best weights

Label Smoothing:
  Replace hard labels (0/1) with soft (0.1/0.9)
  Prevents overconfident predictions
  Regularization effect

**Weight Initialization:**

Xavier/Glorot:
  W ~ N(0, 2/(n_in + n_out)) or U(-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out)))
  For sigmoid/tanh activations
  
He/Kaiming:
  W ~ N(0, 2/n_in) 
  For ReLU activations
  Accounts for ReLU zeroing half the values

Orthogonal:
  Initialize with orthogonal matrix
  Preserves gradient norm through layers
  Good for RNNs

**Training Strategies:**

Mixed Precision Training:
  Use FP16 for forward/backward, FP32 for weight updates
  2x memory reduction, faster on modern GPUs
  Loss scaling to prevent underflow

Gradient Accumulation:
  Accumulate gradients over multiple mini-batches
  Simulate larger batch size with limited memory
  Update weights every N steps

Gradient Clipping:
  Clip by value: Clamp gradients to [-threshold, threshold]
  Clip by norm: Scale gradients if norm > threshold
  Prevents exploding gradients

Transfer Learning:
  Use pretrained model as starting point
  Fine-tune: Train all layers with small lr
  Feature extraction: Freeze backbone, train head only
  Progressive unfreezing: Gradually unfreeze layers`,
					CodeExamples: `# Deep Learning Foundations Implementation

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# ============================================================
# Tensor (Simplified)
# ============================================================

class Tensor:
    """Simple tensor with automatic differentiation."""
    
    def __init__(self, data: List[List[float]], requires_grad: bool = False):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0
        self.requires_grad = requires_grad
        self.grad: Optional['Tensor'] = None
        self._backward: Callable = lambda: None
        self._prev: set = set()
    
    @staticmethod
    def zeros(rows: int, cols: int, requires_grad: bool = False) -> 'Tensor':
        return Tensor([[0.0] * cols for _ in range(rows)], requires_grad)
    
    @staticmethod
    def random(rows: int, cols: int, requires_grad: bool = False,
               scale: float = 1.0) -> 'Tensor':
        data = [[random.gauss(0, scale) for _ in range(cols)]
                for _ in range(rows)]
        return Tensor(data, requires_grad)
    
    @staticmethod
    def he_init(rows: int, cols: int) -> 'Tensor':
        scale = math.sqrt(2.0 / rows)
        return Tensor.random(rows, cols, requires_grad=True, scale=scale)
    
    @staticmethod
    def xavier_init(rows: int, cols: int) -> 'Tensor':
        scale = math.sqrt(2.0 / (rows + cols))
        return Tensor.random(rows, cols, requires_grad=True, scale=scale)
    
    def __add__(self, other: 'Tensor') -> 'Tensor':
        result_data = [[self.data[i][j] + other.data[i][j]
                        for j in range(self.cols)]
                       for i in range(self.rows)]
        result = Tensor(result_data)
        result._prev = {self, other}
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor.zeros(self.rows, self.cols)
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.grad.data[i][j] += result.grad.data[i][j]
            if other.requires_grad:
                if other.grad is None:
                    other.grad = Tensor.zeros(other.rows, other.cols)
                for i in range(other.rows):
                    for j in range(other.cols):
                        other.grad.data[i][j] += result.grad.data[i][j]
        
        result._backward = _backward
        return result
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        assert self.cols == other.rows
        result_data = [[sum(self.data[i][k] * other.data[k][j]
                           for k in range(self.cols))
                        for j in range(other.cols)]
                       for i in range(self.rows)]
        result = Tensor(result_data)
        result._prev = {self, other}
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor.zeros(self.rows, self.cols)
                for i in range(self.rows):
                    for j in range(self.cols):
                        for k in range(other.cols):
                            self.grad.data[i][j] += (
                                result.grad.data[i][k] * other.data[j][k])
            if other.requires_grad:
                if other.grad is None:
                    other.grad = Tensor.zeros(other.rows, other.cols)
                for i in range(other.rows):
                    for j in range(other.cols):
                        for k in range(self.rows):
                            other.grad.data[i][j] += (
                                self.data[k][i] * result.grad.data[k][j])
        
        result._backward = _backward
        return result
    
    def relu(self) -> 'Tensor':
        result_data = [[max(0, self.data[i][j])
                        for j in range(self.cols)]
                       for i in range(self.rows)]
        result = Tensor(result_data)
        result._prev = {self}
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor.zeros(self.rows, self.cols)
                for i in range(self.rows):
                    for j in range(self.cols):
                        if self.data[i][j] > 0:
                            self.grad.data[i][j] += result.grad.data[i][j]
        
        result._backward = _backward
        return result
    
    def sigmoid(self) -> 'Tensor':
        result_data = [[1.0 / (1.0 + math.exp(-min(max(self.data[i][j], -500), 500)))
                        for j in range(self.cols)]
                       for i in range(self.rows)]
        result = Tensor(result_data)
        result._prev = {self}
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor.zeros(self.rows, self.cols)
                for i in range(self.rows):
                    for j in range(self.cols):
                        s = result.data[i][j]
                        self.grad.data[i][j] += result.grad.data[i][j] * s * (1 - s)
        
        result._backward = _backward
        return result
    
    def mse_loss(self, target: 'Tensor') -> 'Tensor':
        total = 0.0
        n = self.rows * self.cols
        for i in range(self.rows):
            for j in range(self.cols):
                total += (self.data[i][j] - target.data[i][j]) ** 2
        loss = Tensor([[total / n]])
        loss._prev = {self}
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor.zeros(self.rows, self.cols)
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.grad.data[i][j] += (
                            2.0 * (self.data[i][j] - target.data[i][j]) / n)
        
        loss._backward = _backward
        loss.grad = Tensor([[1.0]])
        return loss
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        if self.grad is None:
            self.grad = Tensor([[1.0]])
        
        for v in reversed(topo):
            v._backward()
    
    def zero_grad(self):
        self.grad = None


# ============================================================
# Neural Network Layers
# ============================================================

class Linear:
    """Fully connected layer."""
    
    def __init__(self, in_features: int, out_features: int,
                 init: str = "he"):
        if init == "he":
            self.weight = Tensor.he_init(in_features, out_features)
        else:
            self.weight = Tensor.xavier_init(in_features, out_features)
        self.bias = Tensor.zeros(1, out_features, requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight)
        # Broadcast add bias
        result_data = [[out.data[i][j] + self.bias.data[0][j]
                        for j in range(out.cols)]
                       for i in range(out.rows)]
        result = Tensor(result_data)
        result._prev = {out, self.bias}
        
        def _backward():
            if out.grad is None:
                out.grad = Tensor.zeros(out.rows, out.cols)
            for i in range(out.rows):
                for j in range(out.cols):
                    out.grad.data[i][j] += result.grad.data[i][j]
            if self.bias.grad is None:
                self.bias.grad = Tensor.zeros(1, self.bias.cols)
            for i in range(result.rows):
                for j in range(result.cols):
                    self.bias.grad.data[0][j] += result.grad.data[i][j]
            out._backward()
        
        result._backward = _backward
        return result
    
    @property
    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]


class MLP:
    """Multi-layer perceptron."""
    
    def __init__(self, layer_sizes: List[int], activation: str = "relu"):
        self.layers = []
        self.activation = activation
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
    
    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                if self.activation == "relu":
                    x = x.relu()
                elif self.activation == "sigmoid":
                    x = x.sigmoid()
        return x
    
    @property
    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters)
        return params


# ============================================================
# Optimizers
# ============================================================

class SGD:
    """Stochastic Gradient Descent with momentum."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [Tensor.zeros(p.rows, p.cols) for p in parameters]
    
    def step(self):
        for param, vel in zip(self.parameters, self.velocities):
            if param.grad is None:
                continue
            for i in range(param.rows):
                for j in range(param.cols):
                    grad = param.grad.data[i][j]
                    if self.weight_decay > 0:
                        grad += self.weight_decay * param.data[i][j]
                    vel.data[i][j] = self.momentum * vel.data[i][j] + grad
                    param.data[i][j] -= self.lr * vel.data[i][j]
    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()


class Adam:
    """Adam optimizer."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, weight_decay: float = 0.0):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m = [Tensor.zeros(p.rows, p.cols) for p in parameters]
        self.v = [Tensor.zeros(p.rows, p.cols) for p in parameters]
    
    def step(self):
        self.t += 1
        
        for idx, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            for i in range(param.rows):
                for j in range(param.cols):
                    grad = param.grad.data[i][j]
                    
                    # Weight decay (AdamW style)
                    if self.weight_decay > 0:
                        param.data[i][j] -= self.lr * self.weight_decay * param.data[i][j]
                    
                    # Update moments
                    self.m[idx].data[i][j] = (
                        self.beta1 * self.m[idx].data[i][j] +
                        (1 - self.beta1) * grad)
                    self.v[idx].data[i][j] = (
                        self.beta2 * self.v[idx].data[i][j] +
                        (1 - self.beta2) * grad * grad)
                    
                    # Bias correction
                    m_hat = self.m[idx].data[i][j] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[idx].data[i][j] / (1 - self.beta2 ** self.t)
                    
                    param.data[i][j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()


# ============================================================
# Learning Rate Schedulers
# ============================================================

class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler."""
    
    def __init__(self, optimizer, T_max: int, eta_min: float = 0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        self.optimizer.lr = self.eta_min + 0.5 * (
            self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.step_count / self.T_max))


class WarmupScheduler:
    """Linear warmup followed by cosine decay."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 eta_min: float = 0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            self.optimizer.lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (
                self.total_steps - self.warmup_steps)
            self.optimizer.lr = self.eta_min + 0.5 * (
                self.base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))


# ============================================================
# Regularization
# ============================================================

class Dropout:
    """Dropout regularization."""
    
    def __init__(self, rate: float = 0.5):
        self.rate = rate
        self.training = True
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        
        mask = [[1.0 if random.random() > self.rate else 0.0
                 for _ in range(x.cols)]
                for _ in range(x.rows)]
        
        scale = 1.0 / (1.0 - self.rate)
        result_data = [[x.data[i][j] * mask[i][j] * scale
                        for j in range(x.cols)]
                       for i in range(x.rows)]
        return Tensor(result_data)


class BatchNorm:
    """Batch normalization."""
    
    def __init__(self, num_features: int, epsilon: float = 1e-5,
                 momentum: float = 0.1):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum_val = momentum
        self.training = True
        
        self.gamma = Tensor([[1.0] * num_features], requires_grad=True)
        self.beta = Tensor([[0.0] * num_features], requires_grad=True)
        
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
    
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Compute batch statistics
            mean = [0.0] * x.cols
            for i in range(x.rows):
                for j in range(x.cols):
                    mean[j] += x.data[i][j]
            mean = [m / x.rows for m in mean]
            
            var = [0.0] * x.cols
            for i in range(x.rows):
                for j in range(x.cols):
                    var[j] += (x.data[i][j] - mean[j]) ** 2
            var = [v / x.rows for v in var]
            
            # Update running stats
            for j in range(x.cols):
                self.running_mean[j] = (
                    (1 - self.momentum_val) * self.running_mean[j] +
                    self.momentum_val * mean[j])
                self.running_var[j] = (
                    (1 - self.momentum_val) * self.running_var[j] +
                    self.momentum_val * var[j])
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        result_data = []
        for i in range(x.rows):
            row = []
            for j in range(x.cols):
                normalized = (x.data[i][j] - mean[j]) / math.sqrt(
                    var[j] + self.epsilon)
                row.append(
                    self.gamma.data[0][j] * normalized + self.beta.data[0][j])
            result_data.append(row)
        
        return Tensor(result_data)


# ============================================================
# Training Loop
# ============================================================

class Trainer:
    """Neural network trainer."""
    
    def __init__(self, model: MLP, optimizer, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": []}
    
    def train_epoch(self, X: List[Tensor], Y: List[Tensor]) -> float:
        total_loss = 0.0
        
        for x_batch, y_batch in zip(X, Y):
            self.optimizer.zero_grad()
            
            output = self.model.forward(x_batch)
            loss = output.mse_loss(y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.data[0][0]
        
        avg_loss = total_loss / len(X)
        self.history["train_loss"].append(avg_loss)
        
        if self.scheduler:
            self.scheduler.step()
        
        return avg_loss
    
    def evaluate(self, X: List[Tensor], Y: List[Tensor]) -> float:
        total_loss = 0.0
        
        for x_batch, y_batch in zip(X, Y):
            output = self.model.forward(x_batch)
            loss = output.mse_loss(y_batch)
            total_loss += loss.data[0][0]
        
        avg_loss = total_loss / len(X)
        self.history["val_loss"].append(avg_loss)
        return avg_loss
    
    def fit(self, train_X: List[Tensor], train_Y: List[Tensor],
            val_X: List[Tensor] = None, val_Y: List[Tensor] = None,
            epochs: int = 100, patience: int = 10) -> Dict:
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_X, train_Y)
            
            if val_X and val_Y:
                val_loss = self.evaluate(val_X, val_Y)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
        
        return self.history`,
				},
			},
		},
	})
}
