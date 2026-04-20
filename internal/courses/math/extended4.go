package math

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMathModules([]problems.CourseModule{
		{
			ID:          2718,
			Title:       "Optimization and Calculus for ML",
			Description: "Master optimization algorithms, multivariable calculus, and numerical methods that power machine learning: gradient descent variants, convex optimization, automatic differentiation, and constrained optimization.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "Optimization Algorithms for Machine Learning",
					Content: `Optimization is at the core of machine learning — training a model means finding parameters that minimize a loss function. Understanding optimization theory is essential for ML practitioners.

**Gradient Descent:**

The fundamental optimization algorithm:
  θ_{t+1} = θ_t - α × ∇f(θ_t)
  
  α: Learning rate (step size)
  ∇f: Gradient of objective function

Batch Gradient Descent:
  Use entire dataset to compute gradient
  Stable convergence, expensive per step
  Good for convex problems

Stochastic Gradient Descent (SGD):
  Use single sample to estimate gradient
  Noisy but fast updates
  Can escape local minima
  θ_{t+1} = θ_t - α × ∇f_i(θ_t)

Mini-batch SGD:
  Use batch of B samples
  Balance between batch and stochastic
  Typical B = 32, 64, 128, 256

**Learning Rate Schedules:**

Constant: α_t = α₀
  Simple but requires tuning

Step decay: α_t = α₀ × γ^(floor(t/s))
  Reduce by factor γ every s steps

Exponential: α_t = α₀ × e^(-kt)

Cosine annealing: α_t = α_min + (α_max - α_min)/2 × (1 + cos(πt/T))
  Smooth decay with warm restarts

Warmup: Linear increase from 0 to α₀ over first few steps
  Stabilizes training early on
  Combined with cosine or linear decay

**Momentum:**

SGD with momentum:
  v_t = β × v_{t-1} + ∇f(θ_t)
  θ_{t+1} = θ_t - α × v_t
  
  β: Momentum coefficient (typically 0.9)
  Accelerates in consistent gradient direction
  Dampens oscillations

Nesterov Momentum:
  Look ahead before computing gradient
  v_t = β × v_{t-1} + ∇f(θ_t - α × β × v_{t-1})
  θ_{t+1} = θ_t - α × v_t
  Often converges faster than standard momentum

**Adaptive Learning Rate Methods:**

AdaGrad:
  g_t = ∇f(θ_t)
  G_t = G_{t-1} + g_t²  (element-wise)
  θ_{t+1} = θ_t - α × g_t / (√G_t + ε)
  
  Adapts learning rate per parameter
  Good for sparse gradients
  Problem: Learning rate monotonically decreases

RMSProp:
  G_t = γ × G_{t-1} + (1-γ) × g_t²  (exponential average)
  θ_{t+1} = θ_t - α × g_t / (√G_t + ε)
  
  Fixes AdaGrad's diminishing learning rate
  γ typically 0.9

Adam (Adaptive Moment Estimation):
  m_t = β₁m_{t-1} + (1-β₁)g_t       (first moment)
  v_t = β₂v_{t-1} + (1-β₂)g_t²      (second moment)
  m̂_t = m_t / (1-β₁ᵗ)               (bias correction)
  v̂_t = v_t / (1-β₂ᵗ)               (bias correction)
  θ_{t+1} = θ_t - α × m̂_t / (√v̂_t + ε)
  
  Default: β₁=0.9, β₂=0.999, ε=1e-8
  Combines momentum + adaptive learning rate
  Most popular optimizer for deep learning

AdamW:
  Adam with decoupled weight decay
  θ_{t+1} = θ_t - α × (m̂_t / (√v̂_t + ε) + λθ_t)
  Better generalization than Adam with L2

**Convex Optimization:**

Convex function:
  f(αx + (1-α)y) ≤ αf(x) + (1-α)f(y) for α ∈ [0,1]
  Local minimum = global minimum
  Gradient is zero only at minimum

Strictly convex: Unique minimum
  f(αx + (1-α)y) < αf(x) + (1-α)f(y) for x ≠ y

Convexity conditions:
  f''(x) ≥ 0 (1D) → convex
  Hessian H(f) positive semi-definite → convex

Examples:
  Convex: Linear functions, quadratic (positive definite), norms, log-sum-exp
  Non-convex: Neural network loss surfaces, mixture models

**Newton's Method:**

Uses second-order information (Hessian):
  θ_{t+1} = θ_t - H⁻¹∇f(θ_t)
  
  H: Hessian matrix (∂²f/∂θᵢ∂θⱼ)
  
  Quadratic convergence near optimum
  Expensive: O(n³) for n parameters
  May not converge for non-convex functions

Quasi-Newton (L-BFGS):
  Approximate Hessian inverse
  Use limited memory (store last m updates)
  Good for medium-scale smooth problems

**Constrained Optimization:**

Lagrange multipliers:
  Minimize f(x) subject to g(x) = 0
  L(x, λ) = f(x) + λg(x)
  ∇L = 0 gives necessary conditions

KKT conditions (inequality constraints):
  Minimize f(x) subject to g(x) ≤ 0
  Stationarity: ∇f + Σλᵢ∇gᵢ = 0
  Primal feasibility: gᵢ(x) ≤ 0
  Dual feasibility: λᵢ ≥ 0
  Complementary slackness: λᵢgᵢ(x) = 0

**Gradient-Free Optimization:**

When gradient is unavailable or noisy:

Random search: Sample random points
Grid search: Evaluate on grid
Bayesian optimization: Build surrogate model (Gaussian process)
  Acquisition function: UCB, Expected Improvement
  Good for expensive function evaluations (hyperparameter tuning)

Evolutionary algorithms:
  Genetic: Crossover + mutation + selection
  CMA-ES: Covariance Matrix Adaptation
  Good for non-differentiable, multi-modal problems

**Automatic Differentiation:**

Forward mode:
  Compute derivative along with function evaluation
  Dual numbers: (x, dx) where operations propagate derivatives
  Good when: Few inputs, many outputs

Reverse mode (backpropagation):
  Forward pass: Compute values, build computation graph
  Backward pass: Propagate gradients from output to inputs
  Good when: Many inputs, few outputs
  Used in deep learning: Many parameters, one loss

Chain rule:
  df/dx = (df/dg) × (dg/dx)
  Applied recursively through computation graph

Numerical differentiation:
  f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
  Central difference: O(h²) accurate
  Problem: Numerical instability for small h

**Multivariable Calculus Review:**

Gradient:
  ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
  Direction of steepest ascent
  Perpendicular to level curves

Jacobian (vector function f: ℝⁿ → ℝᵐ):
  Jᵢⱼ = ∂fᵢ/∂xⱼ
  J ∈ ℝᵐˣⁿ

Hessian (scalar function f: ℝⁿ → ℝ):
  Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ
  Symmetric matrix
  Eigenvalues determine curvature:
    All positive → local minimum
    All negative → local maximum
    Mixed → saddle point

Directional derivative:
  D_u f = ∇f · u (in direction of unit vector u)
  Maximum rate of change = ||∇f|| (gradient direction)

Taylor expansion (multivariate):
  f(x + δ) ≈ f(x) + ∇f(x)ᵀδ + ½δᵀH(x)δ + ...
  First order: Gradient descent approximation
  Second order: Newton's method approximation`,
					CodeExamples: `# Optimization Algorithms Implementation

import math
import random
from typing import Callable, Dict, List, Optional, Tuple

# ============================================================
# Gradient Descent Variants
# ============================================================

class GradientDescent:
    """Batch gradient descent."""
    
    def __init__(self, lr: float = 0.01, max_iters: int = 1000,
                 tol: float = 1e-8):
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol
        self.history: List[float] = []
    
    def minimize(self, f: Callable, grad_f: Callable,
                 x0: List[float]) -> List[float]:
        x = x0[:]
        
        for _ in range(self.max_iters):
            g = grad_f(x)
            value = f(x)
            self.history.append(value)
            
            # Check convergence
            grad_norm = math.sqrt(sum(gi**2 for gi in g))
            if grad_norm < self.tol:
                break
            
            # Update
            x = [x[i] - self.lr * g[i] for i in range(len(x))]
        
        return x


class SGD:
    """Stochastic gradient descent with momentum."""
    
    def __init__(self, lr: float = 0.01, momentum: float = 0.0,
                 nesterov: bool = False):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity: List[float] = []
    
    def step(self, params: List[float],
             grads: List[float]) -> List[float]:
        if not self.velocity:
            self.velocity = [0.0] * len(params)
        
        new_params = []
        for i in range(len(params)):
            self.velocity[i] = (self.momentum * self.velocity[i] +
                               grads[i])
            
            if self.nesterov:
                update = self.momentum * self.velocity[i] + grads[i]
            else:
                update = self.velocity[i]
            
            new_params.append(params[i] - self.lr * update)
        
        return new_params


class Adam:
    """Adam optimizer."""
    
    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m: List[float] = []
        self.v: List[float] = []
        self.t: int = 0
    
    def step(self, params: List[float],
             grads: List[float]) -> List[float]:
        if not self.m:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)
        
        self.t += 1
        new_params = []
        
        for i in range(len(params)):
            # Update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = (self.beta2 * self.v[i] +
                        (1 - self.beta2) * grads[i] ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update
            update = m_hat / (math.sqrt(v_hat) + self.epsilon)
            
            # Weight decay (AdamW)
            if self.weight_decay > 0:
                update += self.weight_decay * params[i]
            
            new_params.append(params[i] - self.lr * update)
        
        return new_params


class RMSProp:
    """RMSProp optimizer."""
    
    def __init__(self, lr: float = 0.001, gamma: float = 0.9,
                 epsilon: float = 1e-8):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.v: List[float] = []
    
    def step(self, params: List[float],
             grads: List[float]) -> List[float]:
        if not self.v:
            self.v = [0.0] * len(params)
        
        new_params = []
        for i in range(len(params)):
            self.v[i] = (self.gamma * self.v[i] +
                        (1 - self.gamma) * grads[i] ** 2)
            update = grads[i] / (math.sqrt(self.v[i]) + self.epsilon)
            new_params.append(params[i] - self.lr * update)
        
        return new_params


class AdaGrad:
    """AdaGrad optimizer."""
    
    def __init__(self, lr: float = 0.01, epsilon: float = 1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.G: List[float] = []
    
    def step(self, params: List[float],
             grads: List[float]) -> List[float]:
        if not self.G:
            self.G = [0.0] * len(params)
        
        new_params = []
        for i in range(len(params)):
            self.G[i] += grads[i] ** 2
            update = grads[i] / (math.sqrt(self.G[i]) + self.epsilon)
            new_params.append(params[i] - self.lr * update)
        
        return new_params


# ============================================================
# Learning Rate Schedulers
# ============================================================

class StepLR:
    """Step learning rate schedule."""
    
    def __init__(self, initial_lr: float, step_size: int,
                 gamma: float = 0.1):
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self, epoch: int) -> float:
        return self.initial_lr * (self.gamma ** (epoch // self.step_size))


class CosineAnnealingLR:
    """Cosine annealing learning rate schedule."""
    
    def __init__(self, initial_lr: float, T_max: int,
                 eta_min: float = 0.0):
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self, epoch: int) -> float:
        return (self.eta_min +
                (self.initial_lr - self.eta_min) *
                (1 + math.cos(math.pi * epoch / self.T_max)) / 2)


class WarmupCosineScheduler:
    """Warmup + cosine annealing."""
    
    def __init__(self, initial_lr: float, warmup_steps: int,
                 total_steps: int, min_lr: float = 0.0):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.initial_lr * step / max(self.warmup_steps, 1)
        
        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1)
        return (self.min_lr +
                (self.initial_lr - self.min_lr) *
                (1 + math.cos(math.pi * progress)) / 2)


# ============================================================
# Automatic Differentiation (Forward Mode)
# ============================================================

class DualNumber:
    """Dual number for forward-mode automatic differentiation."""
    
    def __init__(self, value: float, derivative: float = 0.0):
        self.value = value
        self.derivative = derivative
    
    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.value + other.value,
                            self.derivative + other.derivative)
        return DualNumber(self.value + other, self.derivative)
    
    def __radd__(self, other):
        return DualNumber(other + self.value, self.derivative)
    
    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.value - other.value,
                            self.derivative - other.derivative)
        return DualNumber(self.value - other, self.derivative)
    
    def __rsub__(self, other):
        return DualNumber(other - self.value, -self.derivative)
    
    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(
                self.value * other.value,
                self.value * other.derivative + self.derivative * other.value)
        return DualNumber(self.value * other, self.derivative * other)
    
    def __rmul__(self, other):
        return DualNumber(other * self.value, other * self.derivative)
    
    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(
                self.value / other.value,
                (self.derivative * other.value -
                 self.value * other.derivative) / (other.value ** 2))
        return DualNumber(self.value / other, self.derivative / other)
    
    def __pow__(self, n):
        if isinstance(n, DualNumber):
            # x^y = exp(y * ln(x))
            val = self.value ** n.value
            deriv = val * (n.derivative * math.log(max(abs(self.value), 1e-10)) +
                          n.value * self.derivative / max(abs(self.value), 1e-10))
            return DualNumber(val, deriv)
        return DualNumber(
            self.value ** n,
            n * (self.value ** (n - 1)) * self.derivative)
    
    def __neg__(self):
        return DualNumber(-self.value, -self.derivative)
    
    def __repr__(self):
        return f"DualNumber({self.value}, {self.derivative})"


def dual_sin(x: DualNumber) -> DualNumber:
    return DualNumber(math.sin(x.value),
                     math.cos(x.value) * x.derivative)

def dual_cos(x: DualNumber) -> DualNumber:
    return DualNumber(math.cos(x.value),
                     -math.sin(x.value) * x.derivative)

def dual_exp(x: DualNumber) -> DualNumber:
    val = math.exp(x.value)
    return DualNumber(val, val * x.derivative)

def dual_log(x: DualNumber) -> DualNumber:
    return DualNumber(math.log(max(x.value, 1e-10)),
                     x.derivative / max(x.value, 1e-10))


def compute_gradient(f: Callable, x: List[float]) -> List[float]:
    """Compute gradient using forward-mode AD."""
    gradient = []
    for i in range(len(x)):
        dual_x = [DualNumber(x[j], 1.0 if j == i else 0.0)
                   for j in range(len(x))]
        result = f(dual_x)
        gradient.append(result.derivative)
    return gradient


# ============================================================
# Reverse-Mode AD (Backpropagation)
# ============================================================

class Var:
    """Variable for reverse-mode automatic differentiation."""
    _counter = 0
    
    def __init__(self, value: float, children: List[Tuple['Var', float]] = None):
        self.value = value
        self.children = children or []
        self.grad = 0.0
        Var._counter += 1
        self.id = Var._counter
    
    def __add__(self, other):
        if not isinstance(other, Var):
            other = Var(other)
        result = Var(self.value + other.value)
        result.children = [(self, 1.0), (other, 1.0)]
        return result
    
    def __radd__(self, other):
        return Var(other).__add__(self)
    
    def __mul__(self, other):
        if not isinstance(other, Var):
            other = Var(other)
        result = Var(self.value * other.value)
        result.children = [(self, other.value), (other, self.value)]
        return result
    
    def __rmul__(self, other):
        return Var(other).__mul__(self)
    
    def __sub__(self, other):
        if not isinstance(other, Var):
            other = Var(other)
        result = Var(self.value - other.value)
        result.children = [(self, 1.0), (other, -1.0)]
        return result
    
    def __neg__(self):
        result = Var(-self.value)
        result.children = [(self, -1.0)]
        return result
    
    def __pow__(self, n: float):
        result = Var(self.value ** n)
        result.children = [(self, n * self.value ** (n - 1))]
        return result
    
    def backward(self):
        """Compute gradients via reverse-mode AD."""
        topo_order = []
        visited = set()
        
        def build_topo(v):
            if v.id not in visited:
                visited.add(v.id)
                for child, _ in v.children:
                    build_topo(child)
                topo_order.append(v)
        
        build_topo(self)
        
        self.grad = 1.0
        for v in reversed(topo_order):
            for child, local_grad in v.children:
                child.grad += v.grad * local_grad


# ============================================================
# Numerical Differentiation
# ============================================================

def numerical_gradient(f: Callable, x: List[float],
                       h: float = 1e-5) -> List[float]:
    """Central difference gradient."""
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad


def numerical_hessian(f: Callable, x: List[float],
                      h: float = 1e-5) -> List[List[float]]:
    """Numerical Hessian via finite differences."""
    n = len(x)
    H = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i, n):
            x_pp = x[:]
            x_pm = x[:]
            x_mp = x[:]
            x_mm = x[:]
            
            x_pp[i] += h; x_pp[j] += h
            x_pm[i] += h; x_pm[j] -= h
            x_mp[i] -= h; x_mp[j] += h
            x_mm[i] -= h; x_mm[j] -= h
            
            H[i][j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4*h*h)
            H[j][i] = H[i][j]
    
    return H


# ============================================================
# Newton's Method for Optimization
# ============================================================

def newtons_method(f: Callable, grad_f: Callable,
                   hess_f: Callable, x0: List[float],
                   max_iters: int = 100,
                   tol: float = 1e-8) -> List[float]:
    """Newton's method for optimization."""
    x = x0[:]
    n = len(x)
    
    for _ in range(max_iters):
        g = grad_f(x)
        H = hess_f(x)
        
        grad_norm = math.sqrt(sum(gi**2 for gi in g))
        if grad_norm < tol:
            break
        
        # Solve H @ d = -g
        from typing import List as L
        aug = [H[i][:] + [-g[i]] for i in range(n)]
        
        for col in range(n):
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]
            
            if abs(aug[col][col]) < 1e-12:
                break
            
            for row in range(col + 1, n):
                factor = aug[row][col] / aug[col][col]
                for j in range(n + 1):
                    aug[row][j] -= factor * aug[col][j]
        
        d = [0.0] * n
        for i in range(n - 1, -1, -1):
            d[i] = aug[i][n]
            for j in range(i + 1, n):
                d[i] -= aug[i][j] * d[j]
            if abs(aug[i][i]) > 1e-12:
                d[i] /= aug[i][i]
        
        x = [x[i] + d[i] for i in range(n)]
    
    return x


# ============================================================
# Bayesian Optimization
# ============================================================

class BayesianOptimizer:
    """Simple Bayesian optimization with random forest surrogate."""
    
    def __init__(self, bounds: List[Tuple[float, float]],
                 n_initial: int = 5):
        self.bounds = bounds
        self.n_initial = n_initial
        self.X: List[List[float]] = []
        self.y: List[float] = []
    
    def suggest(self) -> List[float]:
        """Suggest next point to evaluate."""
        if len(self.X) < self.n_initial:
            return [random.uniform(lo, hi) for lo, hi in self.bounds]
        
        # Simple acquisition: random sampling + predicted best
        best_x = None
        best_score = float('-inf')
        
        for _ in range(100):
            x = [random.uniform(lo, hi) for lo, hi in self.bounds]
            mean, std = self._predict(x)
            
            # UCB acquisition
            score = mean + 2.0 * std
            
            if score > best_score:
                best_score = score
                best_x = x
        
        return best_x
    
    def observe(self, x: List[float], y: float):
        self.X.append(x[:])
        self.y.append(y)
    
    def _predict(self, x: List[float]) -> Tuple[float, float]:
        """Simple nearest-neighbor prediction."""
        distances = []
        for i, xi in enumerate(self.X):
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(x, xi)))
            distances.append((dist, self.y[i]))
        
        distances.sort()
        k = min(3, len(distances))
        nearest = distances[:k]
        
        if not nearest:
            return 0.0, 1.0
        
        weights = [1.0 / max(d, 1e-6) for d, _ in nearest]
        total_w = sum(weights)
        
        mean = sum(w * y for (d, y), w in zip(nearest, weights)) / total_w
        var = sum(w * (y - mean)**2
                 for (d, y), w in zip(nearest, weights)) / total_w
        
        return mean, math.sqrt(max(var, 1e-6))
    
    def get_best(self) -> Tuple[List[float], float]:
        if not self.y:
            return [], float('-inf')
        best_idx = self.y.index(max(self.y))
        return self.X[best_idx], self.y[best_idx]


# ============================================================
# Constrained Optimization (Lagrangian)
# ============================================================

def augmented_lagrangian(
        f: Callable, grad_f: Callable,
        constraints: List[Callable],
        constraint_grads: List[Callable],
        x0: List[float],
        mu: float = 1.0, rho: float = 10.0,
        max_outer: int = 50,
        max_inner: int = 100,
        tol: float = 1e-6) -> List[float]:
    """Augmented Lagrangian method for constrained optimization."""
    x = x0[:]
    n = len(x)
    m = len(constraints)
    lambdas = [0.0] * m
    
    for _ in range(max_outer):
        # Inner minimization of augmented Lagrangian
        for _ in range(max_inner):
            grad = grad_f(x)[:]
            
            for j in range(m):
                c = constraints[j](x)
                cg = constraint_grads[j](x)
                
                for i in range(n):
                    grad[i] += lambdas[j] * cg[i] + mu * c * cg[i]
            
            grad_norm = math.sqrt(sum(g**2 for g in grad))
            if grad_norm < tol:
                break
            
            lr = 0.001
            x = [x[i] - lr * grad[i] for i in range(n)]
        
        # Update multipliers
        max_violation = 0.0
        for j in range(m):
            c = constraints[j](x)
            lambdas[j] += mu * c
            max_violation = max(max_violation, abs(c))
        
        if max_violation < tol:
            break
        
        mu *= rho
    
    return x`,
				},
			},
		},
	})
}
