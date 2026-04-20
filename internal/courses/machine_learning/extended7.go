package machinelearning

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMachineLearningModules([]problems.CourseModule{
		{
			ID:          2521,
			Title:       "Reinforcement Learning and Model Interpretability",
			Description: "Understand reinforcement learning fundamentals including MDPs, Q-learning, policy gradients, and deep RL. Master model interpretability with SHAP, LIME, and explainability techniques.",
			Order:       21,
			Lessons: []problems.Lesson{
				{
					Title: "Reinforcement Learning Fundamentals",
					Content: `Reinforcement learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions and learns to maximize cumulative reward.

**Core Concepts:**

Agent: The learner and decision maker
Environment: What the agent interacts with
State (s): Current situation of the agent
Action (a): What the agent can do
Reward (r): Feedback from environment after action
Policy (π): Strategy mapping states to actions
Value function V(s): Expected cumulative future reward from state s
Action-value function Q(s,a): Expected cumulative reward from state s taking action a

Episode: Sequence of states, actions, rewards from start to terminal state
Return (G): Total discounted reward from time t
  G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
  γ (gamma): Discount factor (0 to 1) — importance of future rewards

**Markov Decision Process (MDP):**

Formal framework for RL:
  S: Set of states
  A: Set of actions
  P(s'|s,a): Transition probability
  R(s,a,s'): Reward function
  γ: Discount factor

Markov property: Future depends only on current state, not history
  P(s_{t+1}|s_t, a_t) = P(s_{t+1}|s_1,...,s_t, a_1,...,a_t)

Bellman Equation (Value):
  V(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γ*V(s')]

Bellman Equation (Action-Value):
  Q(s,a) = Σ P(s'|s,a)[R(s,a,s') + γ*max_a' Q(s',a')]

**Dynamic Programming:**

Policy Evaluation:
  Given policy π, compute V^π(s) iteratively
  V_{k+1}(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R + γ*V_k(s')]
  Iterate until convergence

Policy Improvement:
  Given V^π, compute greedy policy
  π'(s) = argmax_a Σ_{s'} P(s'|s,a)[R + γ*V^π(s')]

Policy Iteration:
  1. Initialize random policy
  2. Policy Evaluation: Compute V^π
  3. Policy Improvement: Get greedy π'
  4. If π' = π, stop; else π = π', go to 2

Value Iteration:
  Combine evaluation and improvement
  V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a)[R + γ*V_k(s')]
  Directly computes optimal value function

**Temporal Difference (TD) Learning:**

TD(0):
  V(s) ← V(s) + α[r + γ*V(s') - V(s)]
  Bootstrap: Update using estimate of next state
  Online: Update after each step (no need for full episode)

TD error: δ = r + γ*V(s') - V(s)
  The "surprise" of what actually happened vs prediction

**Q-Learning (Off-Policy TD):**

Q(s,a) ← Q(s,a) + α[r + γ*max_a' Q(s',a') - Q(s,a)]

Off-policy: Learns optimal Q regardless of behavior policy
  Behavior policy: How to explore (e.g., ε-greedy)
  Target policy: Optimal greedy policy

ε-greedy exploration:
  With probability ε: Random action (explore)
  With probability 1-ε: Best action (exploit)
  Typically ε decays over time

**SARSA (On-Policy TD):**

Q(s,a) ← Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]

On-policy: a' chosen from same policy being learned
  More conservative than Q-learning
  Safer in environments with penalties

**Deep Q-Network (DQN):**

Use neural network to approximate Q(s,a;θ)
  Input: State
  Output: Q-value for each action
  Training: Minimize (r + γ*max_a' Q(s',a';θ⁻) - Q(s,a;θ))²

Key innovations:
  Experience replay: Store (s,a,r,s') in buffer
    Sample random minibatch for training
    Breaks correlation between consecutive samples
    
  Target network: Separate network θ⁻ for target
    Copy θ → θ⁻ periodically (or soft update)
    Stabilizes training
    
  Extensions:
    Double DQN: Decouple action selection and evaluation
      a* = argmax_a Q(s',a;θ)  [select with online]
      Target: r + γ*Q(s',a*;θ⁻) [evaluate with target]
    
    Dueling DQN: Separate value and advantage streams
      Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    
    Prioritized replay: Sample transitions with high TD error
    
    Noisy Nets: Parametric noise for exploration

**Policy Gradient Methods:**

Instead of learning Q-values, learn policy directly
  π(a|s;θ): Parameterized policy (neural network)

REINFORCE:
  ∇J(θ) = E[Σ_t ∇log π(a_t|s_t;θ) * G_t]
  Monte Carlo: Use full episode return G_t
  High variance — use baseline to reduce

Actor-Critic:
  Actor: Policy π(a|s;θ) — decides actions
  Critic: Value V(s;w) — evaluates states
  
  Advantage: A(s,a) = Q(s,a) - V(s) ≈ r + γ*V(s') - V(s)
  Update actor: ∇J(θ) = E[∇log π(a|s;θ) * A(s,a)]
  Update critic: Minimize (V(s;w) - G_t)²

**PPO (Proximal Policy Optimization):**

State-of-the-art policy gradient method
  Clip objective to prevent too-large policy updates
  
  L^CLIP(θ) = E[min(r_t(θ)*A_t, clip(r_t(θ), 1-ε, 1+ε)*A_t)]
  where r_t(θ) = π(a_t|s_t;θ) / π(a_t|s_t;θ_old)
  
  Simple, stable, widely used (ChatGPT's RLHF uses PPO)

**Multi-Armed Bandits:**

Simplified RL — no state transitions, just actions and rewards
  K arms (actions), each with unknown reward distribution
  Goal: Maximize cumulative reward

Exploration strategies:
  ε-greedy: Random exploration with probability ε
  UCB (Upper Confidence Bound):
    a = argmax_a [Q(a) + c*sqrt(ln(t)/N(a))]
    Exploration bonus: Less-tried arms get higher bonus
  
  Thompson Sampling:
    Sample from posterior distribution of each arm
    Select arm with highest sample
    Bayesian approach — naturally balances explore/exploit

Applications:
  Ad placement, news article recommendation
  Clinical trials, A/B testing
  Hyperparameter tuning`,
					CodeExamples: `# Reinforcement Learning Examples

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

# ============================================================
# Gridworld Environment
# ============================================================

class GridWorld:
    """Simple gridworld MDP."""
    
    def __init__(self, rows: int = 4, cols: int = 4,
                 terminal_states: List[Tuple[int, int]] = None,
                 walls: List[Tuple[int, int]] = None,
                 rewards: Dict[Tuple[int, int], float] = None):
        self.rows = rows
        self.cols = cols
        self.terminal_states = terminal_states or [(0, 0), (rows-1, cols-1)]
        self.walls = set(walls or [])
        self.rewards = rewards or {(rows-1, cols-1): 1.0, (0, 0): -1.0}
        self.default_reward = -0.04
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U
        self.action_names = ["right", "left", "down", "up"]
        self.state = (0, 0)
    
    def reset(self, start: Tuple[int, int] = None) -> Tuple[int, int]:
        if start:
            self.state = start
        else:
            while True:
                s = (random.randint(0, self.rows-1),
                     random.randint(0, self.cols-1))
                if s not in self.terminal_states and s not in self.walls:
                    self.state = s
                    break
        return self.state
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        if self.state in self.terminal_states:
            return self.state, 0.0, True
        
        dr, dc = self.actions[action]
        new_r = max(0, min(self.rows - 1, self.state[0] + dr))
        new_c = max(0, min(self.cols - 1, self.state[1] + dc))
        
        new_state = (new_r, new_c)
        if new_state in self.walls:
            new_state = self.state
        
        self.state = new_state
        reward = self.rewards.get(new_state, self.default_reward)
        done = new_state in self.terminal_states
        
        return new_state, reward, done
    
    def get_states(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.rows)
                for c in range(self.cols) if (r, c) not in self.walls]


# ============================================================
# Q-Learning Agent
# ============================================================

class QLearningAgent:
    """Tabular Q-Learning."""
    
    def __init__(self, n_actions: int, alpha: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[Any, List[float]] = defaultdict(
            lambda: [0.0] * n_actions)
    
    def select_action(self, state: Any) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_values = self.q_table[state]
        max_q = max(q_values)
        return q_values.index(max_q)
    
    def update(self, state: Any, action: int, reward: float,
               next_state: Any, done: bool):
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_state])
        
        self.q_table[state][action] = (
            current_q + self.alpha * (target - current_q))
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                          self.epsilon * self.epsilon_decay)
    
    def get_policy(self) -> Dict[Any, int]:
        policy = {}
        for state, q_values in self.q_table.items():
            policy[state] = q_values.index(max(q_values))
        return policy


# ============================================================
# SARSA Agent
# ============================================================

class SARSAAgent:
    """On-policy SARSA."""
    
    def __init__(self, n_actions: int, alpha: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[Any, List[float]] = defaultdict(
            lambda: [0.0] * n_actions)
    
    def select_action(self, state: Any) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_values = self.q_table[state]
        return q_values.index(max(q_values))
    
    def update(self, state: Any, action: int, reward: float,
               next_state: Any, next_action: int, done: bool):
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state][next_action]
        
        self.q_table[state][action] = (
            current_q + self.alpha * (target - current_q))
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                          self.epsilon * self.epsilon_decay)


# ============================================================
# Value Iteration
# ============================================================

def value_iteration(env: GridWorld, gamma: float = 0.99,
                    theta: float = 1e-6,
                    max_iters: int = 1000) -> Tuple[
                        Dict[Tuple, float], Dict[Tuple, int]]:
    """Value iteration for GridWorld."""
    states = env.get_states()
    V: Dict[Tuple, float] = {s: 0.0 for s in states}
    
    for _ in range(max_iters):
        delta = 0.0
        
        for s in states:
            if s in env.terminal_states:
                continue
            
            old_v = V[s]
            action_values = []
            
            for a in range(len(env.actions)):
                env.state = s
                next_s, reward, done = env.step(a)
                env.state = s  # Reset state
                
                if done:
                    value = reward
                else:
                    value = reward + gamma * V.get(next_s, 0.0)
                action_values.append(value)
            
            V[s] = max(action_values) if action_values else 0.0
            delta = max(delta, abs(old_v - V[s]))
        
        if delta < theta:
            break
    
    # Extract policy
    policy: Dict[Tuple, int] = {}
    for s in states:
        if s in env.terminal_states:
            continue
        
        best_a = 0
        best_v = float('-inf')
        
        for a in range(len(env.actions)):
            env.state = s
            next_s, reward, done = env.step(a)
            env.state = s
            
            value = reward + (0 if done else gamma * V.get(next_s, 0.0))
            if value > best_v:
                best_v = value
                best_a = a
        
        policy[s] = best_a
    
    return V, policy


# ============================================================
# Policy Gradient (REINFORCE)
# ============================================================

class SimplePolicy:
    """Softmax policy with linear features."""
    
    def __init__(self, n_features: int, n_actions: int):
        self.weights = [[random.gauss(0, 0.1) for _ in range(n_features)]
                        for _ in range(n_actions)]
    
    def _logits(self, state_features: List[float]) -> List[float]:
        logits = []
        for a in range(len(self.weights)):
            logit = sum(w * f for w, f in
                       zip(self.weights[a], state_features))
            logits.append(logit)
        return logits
    
    def action_probs(self, state_features: List[float]) -> List[float]:
        logits = self._logits(state_features)
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_logits)
        return [e / total for e in exp_logits]
    
    def select_action(self, state_features: List[float]) -> int:
        probs = self.action_probs(state_features)
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return i
        return len(probs) - 1
    
    def update(self, trajectory: List[Tuple[List[float], int, float]],
               gamma: float = 0.99, lr: float = 0.01):
        """REINFORCE update."""
        T = len(trajectory)
        
        for t in range(T):
            features, action, _ = trajectory[t]
            
            # Compute return G_t
            G = 0.0
            for k in range(t, T):
                G += (gamma ** (k - t)) * trajectory[k][2]
            
            # Compute gradient
            probs = self.action_probs(features)
            
            for a in range(len(self.weights)):
                indicator = 1.0 if a == action else 0.0
                grad_log_pi = indicator - probs[a]
                
                for j in range(len(features)):
                    self.weights[a][j] += lr * G * grad_log_pi * features[j]


# ============================================================
# Experience Replay Buffer
# ============================================================

@dataclass
class Transition:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position: int = 0
    
    def push(self, state: Any, action: int, reward: float,
             next_state: Any, done: bool):
        transition = Transition(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized experience replay."""
    
    def __init__(self, capacity: int = 10000,
                 alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer: List[Transition] = []
        self.priorities: List[float] = []
        self.position: int = 0
        self.max_priority: float = 1.0
    
    def push(self, state: Any, action: int, reward: float,
             next_state: Any, done: bool):
        transition = Transition(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(self.max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Transition],
                                                List[int],
                                                List[float]]:
        n = len(self.buffer)
        priorities = [p ** self.alpha for p in self.priorities[:n]]
        total = sum(priorities)
        probs = [p / total for p in priorities]
        
        indices = random.choices(range(n), weights=probs, k=batch_size)
        
        max_weight = (n * min(probs)) ** (-self.beta)
        weights = [(n * probs[i]) ** (-self.beta) / max_weight
                   for i in indices]
        
        transitions = [self.buffer[i] for i in indices]
        return transitions, indices, weights
    
    def update_priorities(self, indices: List[int],
                         td_errors: List[float]):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
            self.max_priority = max(self.max_priority,
                                   self.priorities[idx])


# ============================================================
# Multi-Armed Bandit
# ============================================================

class EpsilonGreedyBandit:
    """Epsilon-greedy multi-armed bandit."""
    
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
    
    def select_arm(self) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_arms - 1)
        return self.values.index(max(self.values))
    
    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class UCBBandit:
    """Upper Confidence Bound bandit."""
    
    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_count = 0
    
    def select_arm(self) -> int:
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i
        
        ucb_values = []
        for i in range(self.n_arms):
            bonus = self.c * math.sqrt(
                math.log(self.total_count) / self.counts[i])
            ucb_values.append(self.values[i] + bonus)
        
        return ucb_values.index(max(ucb_values))
    
    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.total_count += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class ThompsonSamplingBandit:
    """Thompson Sampling for Bernoulli bandits."""
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.alpha = [1.0] * n_arms  # successes + 1
        self.beta = [1.0] * n_arms   # failures + 1
    
    def select_arm(self) -> int:
        samples = [self._sample_beta(self.alpha[i], self.beta[i])
                   for i in range(self.n_arms)]
        return samples.index(max(samples))
    
    def update(self, arm: int, reward: float):
        if reward > 0.5:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
    
    @staticmethod
    def _sample_beta(a: float, b: float) -> float:
        """Approximate Beta sampling using Gamma."""
        x = sum(-math.log(max(random.random(), 1e-10)) for _ in range(int(a)))
        y = sum(-math.log(max(random.random(), 1e-10)) for _ in range(int(b)))
        return x / max(x + y, 1e-10)`,
				},
				{
					Title: "Model Interpretability and Explainability",
					Content: `Model interpretability is crucial for trust, debugging, regulatory compliance, and understanding model behavior. Different techniques provide different levels of explanation.

**Intrinsic Interpretability:**

Linear models:
  Coefficient = feature importance and direction
  Positive coefficient → positive impact on prediction
  Standardize features for comparable coefficients
  
Decision trees:
  Natural feature importance: Gini importance
  Visual explanation: Follow decision path
  Limited depth for interpretability

Rule-based models:
  IF-THEN rules
  Easy to understand and audit
  Can be extracted from complex models

GAMs (Generalized Additive Models):
  y = f₁(x₁) + f₂(x₂) + ... + fₖ(xₖ)
  Each function captures non-linear effect of one feature
  Interactions can be added: f_{ij}(xᵢ, xⱼ)
  Interpretable: Plot each shape function

**Post-hoc Explanations:**

Feature Importance:
  Permutation importance:
    1. Train model, compute baseline score
    2. Shuffle one feature, compute new score
    3. Importance = decrease in score
    Model-agnostic, reliable
    
  Drop-column importance:
    Retrain without each feature
    More expensive but more accurate
    
  SHAP-based importance: Mean |SHAP values|

**SHAP (SHapley Additive exPlanations):**

Based on game theory — Shapley values
  Fair allocation of "credit" among features
  
Properties:
  Efficiency: SHAP values sum to prediction - base value
  Symmetry: Equal contributions → equal SHAP values
  Dummy: Features that don't contribute get 0
  Additivity: For ensemble, SHAP of sum = sum of SHAPs

SHAP value for feature i:
  ϕᵢ = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)!/|N|!] *
        [f(S∪{i}) - f(S)]
  
  Sum over all subsets S not containing feature i
  Compute marginal contribution of adding feature i

Variants:
  KernelSHAP: Model-agnostic, uses weighted linear regression
  TreeSHAP: Exact computation for tree-based models (fast)
  DeepSHAP: Approximate for neural networks
  LinearSHAP: Exact for linear models

Visualizations:
  Force plot: Individual prediction explanation
  Summary plot: Feature importance across dataset
  Dependence plot: Feature value vs SHAP value
  Interaction plot: Pairwise feature interactions

**LIME (Local Interpretable Model-agnostic Explanations):**

Explains individual predictions by locally fitting simple model

Algorithm:
  1. Generate perturbed samples around instance
  2. Get predictions from complex model for perturbations
  3. Weight samples by proximity to original instance
  4. Fit interpretable model (linear, decision tree) on weighted data
  5. Return interpretable model as explanation

Key concepts:
  Locality: Explanation valid only near the instance
  Fidelity: How well local model approximates complex model
  Interpretability: Simple model is human-understandable

For tabular data:
  Perturb by sampling from training distribution
  Use superpixels for image data
  Use word removal for text data

Limitations:
  Sensitive to perturbation strategy
  Locality is hard to define
  Explanations can be unstable
  Does not satisfy Shapley axioms

**Partial Dependence Plots (PDP):**

Show marginal effect of feature on prediction
  PDP(xₛ) = (1/n) Σᵢ f(xₛ, xᵢ_c)
  Average prediction while varying feature s
  Keep complement features at their observed values

Individual Conditional Expectation (ICE):
  Like PDP but show individual curves
  Reveals heterogeneous effects hidden by averaging

Limitations:
  Assumes feature independence
  Can be misleading with correlated features
  Only 1D or 2D visualizations practical

**Counterfactual Explanations:**

"What would need to change for a different prediction?"
  Find minimal change to input that flips prediction
  
  Minimize: distance(x, x') 
  Subject to: f(x') = desired_class

Properties:
  Actionable: Tell user what to change
  Human-friendly: "If your income were $5K higher..."
  Sparse: Change as few features as possible
  
  DiCE: Diverse Counterfactual Explanations
    Generate multiple diverse counterfactuals
    Provide alternative paths to desired outcome

**Attention Visualization:**

For transformer models:
  Attention heatmaps: Which tokens attend to which
  Layer-by-layer visualization
  Head-specific patterns

Limitations:
  Attention ≠ explanation (debate in literature)
  May not reflect true feature importance
  Useful as debugging tool, not definitive explanation

**Gradient-based Methods (Neural Networks):**

Saliency maps:
  |∂f/∂x|: Gradient of output w.r.t. input
  Highlights important input regions

Integrated Gradients:
  IG(x) = (x - x') × ∫₀¹ (∂f/∂x)(x' + α(x-x')) dα
  Satisfies completeness: sum of attributions = f(x) - f(x')
  Baseline x': Usually zero input

GradCAM:
  For CNNs: Gradient w.r.t. feature map activations
  Coarse localization of important regions
  GradCAM++: Improved version with pixel-wise weighting

**Fairness and Bias:**

Types of bias:
  Historical bias: Training data reflects past discrimination
  Representation bias: Underrepresentation of groups
  Measurement bias: Features measured differently across groups
  Aggregation bias: One model for heterogeneous populations
  Evaluation bias: Benchmark doesn't match deployment

Fairness metrics:
  Demographic parity: P(positive|A=0) = P(positive|A=1)
  Equalized odds: Same TPR and FPR across groups
  Calibration: Same accuracy across groups
  Individual fairness: Similar individuals treated similarly

Mitigation:
  Pre-processing: Reweighting, data augmentation
  In-processing: Adversarial debiasing, constrained optimization
  Post-processing: Threshold adjustment per group

**Model Cards and Documentation:**

Model card template:
  Model details: Architecture, training procedure
  Intended use: Primary use cases, out-of-scope uses
  Factors: Demographics, environment conditions
  Metrics: Performance across subgroups
  Ethical considerations: Risks, mitigations
  Limitations: Known failure modes`,
					CodeExamples: `# Model Interpretability Examples

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from collections import defaultdict
from itertools import combinations

# ============================================================
# Permutation Feature Importance
# ============================================================

class PermutationImportance:
    """Model-agnostic feature importance via permutation."""
    
    def __init__(self, model_predict: Callable,
                 score_fn: Callable, n_repeats: int = 10):
        self.model_predict = model_predict
        self.score_fn = score_fn
        self.n_repeats = n_repeats
    
    def compute(self, X: List[List[float]],
                y: List[float]) -> Dict[int, Dict[str, float]]:
        baseline = self.score_fn(y, self.model_predict(X))
        n_features = len(X[0])
        importances: Dict[int, Dict[str, float]] = {}
        
        for j in range(n_features):
            scores = []
            for _ in range(self.n_repeats):
                X_perm = [row[:] for row in X]
                # Shuffle column j
                col_vals = [X_perm[i][j] for i in range(len(X_perm))]
                random.shuffle(col_vals)
                for i in range(len(X_perm)):
                    X_perm[i][j] = col_vals[i]
                
                perm_score = self.score_fn(y, self.model_predict(X_perm))
                scores.append(baseline - perm_score)
            
            mean_imp = sum(scores) / len(scores)
            std_imp = math.sqrt(
                sum((s - mean_imp) ** 2 for s in scores) / len(scores))
            
            importances[j] = {
                "mean": mean_imp,
                "std": std_imp,
                "scores": scores,
            }
        
        return importances


# ============================================================
# SHAP Values (Exact Shapley - exponential)
# ============================================================

class ExactSHAP:
    """Exact Shapley values (for small feature sets)."""
    
    def __init__(self, model_predict: Callable,
                 background_data: List[List[float]]):
        self.model_predict = model_predict
        self.background = background_data
    
    def explain(self, instance: List[float]) -> List[float]:
        n = len(instance)
        shap_values = [0.0] * n
        
        for i in range(n):
            # Sum over all subsets not containing feature i
            other_features = [j for j in range(n) if j != i]
            
            for size in range(len(other_features) + 1):
                for subset in combinations(other_features, size):
                    subset_set = set(subset)
                    
                    # Compute f(S ∪ {i}) - f(S)
                    with_i = self._marginal_expectation(
                        instance, subset_set | {i})
                    without_i = self._marginal_expectation(
                        instance, subset_set)
                    
                    marginal = with_i - without_i
                    
                    # Shapley weight
                    weight = (math.factorial(size) *
                             math.factorial(n - size - 1) /
                             math.factorial(n))
                    
                    shap_values[i] += weight * marginal
        
        return shap_values
    
    def _marginal_expectation(self, instance: List[float],
                              features: Set[int]) -> float:
        """Compute E[f(x)] using features from instance, rest from background."""
        predictions = []
        for bg in self.background:
            x = bg[:]
            for j in features:
                x[j] = instance[j]
            pred = self.model_predict([x])
            predictions.append(pred[0] if isinstance(pred, list) else pred)
        
        return sum(predictions) / len(predictions)


# ============================================================
# KernelSHAP (Approximation)
# ============================================================

class KernelSHAP:
    """Approximate SHAP values using weighted linear regression."""
    
    def __init__(self, model_predict: Callable,
                 background_data: List[List[float]],
                 n_samples: int = 2048):
        self.model_predict = model_predict
        self.background = background_data
        self.n_samples = n_samples
    
    def explain(self, instance: List[float]) -> List[float]:
        n = len(instance)
        
        # Generate coalition samples
        coalitions = []
        predictions = []
        weights = []
        
        # Always include all-zeros and all-ones
        coalitions.append([0] * n)
        coalitions.append([1] * n)
        
        for _ in range(self.n_samples - 2):
            coalition = [random.randint(0, 1) for _ in range(n)]
            coalitions.append(coalition)
        
        for coalition in coalitions:
            # Create input: features in coalition from instance, others from background
            bg = random.choice(self.background)
            x = [instance[j] if coalition[j] else bg[j] for j in range(n)]
            pred = self.model_predict([x])
            predictions.append(pred[0] if isinstance(pred, list) else pred)
            
            # Kernel SHAP weight
            s = sum(coalition)
            if 0 < s < n:
                w = (n - 1) / (math.comb(n, s) * s * (n - s))
            else:
                w = 1e6  # Large weight for full/empty coalitions
            weights.append(w)
        
        # Weighted linear regression
        shap_values = self._weighted_regression(
            coalitions, predictions, weights, n)
        
        return shap_values
    
    def _weighted_regression(self, X: List[List[int]],
                            y: List[float], w: List[float],
                            n: int) -> List[float]:
        """Solve weighted least squares."""
        # Simplified: compute weighted mean contribution
        contrib = [0.0] * n
        counts = [0.0] * n
        
        base = y[0]  # All-zeros prediction
        
        for i in range(len(X)):
            for j in range(n):
                if X[i][j] == 1:
                    contrib[j] += w[i] * (y[i] - base)
                    counts[j] += w[i]
        
        shap_values = [contrib[j] / max(counts[j], 1e-10) for j in range(n)]
        
        # Normalize to sum to prediction - base
        full_pred = y[1]
        current_sum = sum(shap_values)
        target_sum = full_pred - base
        
        if abs(current_sum) > 1e-10:
            scale = target_sum / current_sum
            shap_values = [v * scale for v in shap_values]
        
        return shap_values


# ============================================================
# LIME
# ============================================================

class LIME:
    """Local Interpretable Model-agnostic Explanations."""
    
    def __init__(self, model_predict: Callable,
                 n_samples: int = 500,
                 kernel_width: float = 0.25):
        self.model_predict = model_predict
        self.n_samples = n_samples
        self.kernel_width = kernel_width
    
    def explain(self, instance: List[float],
                feature_means: List[float],
                feature_stds: List[float]) -> Dict[int, float]:
        n = len(instance)
        
        # Generate perturbed samples
        perturbed = []
        for _ in range(self.n_samples):
            sample = [
                instance[j] + random.gauss(0, feature_stds[j])
                for j in range(n)]
            perturbed.append(sample)
        
        # Get predictions for perturbed samples
        predictions = self.model_predict(perturbed)
        if not isinstance(predictions, list):
            predictions = list(predictions)
        
        # Compute distances and weights
        weights = []
        for sample in perturbed:
            dist_sq = sum((s - i) ** 2 for s, i in zip(sample, instance))
            w = math.exp(-dist_sq / (2 * self.kernel_width ** 2))
            weights.append(w)
        
        # Weighted linear regression
        coefficients = self._weighted_linear_regression(
            perturbed, predictions, weights)
        
        return {j: coefficients[j] for j in range(n)}
    
    def _weighted_linear_regression(self, X: List[List[float]],
                                    y: List[float],
                                    w: List[float]) -> List[float]:
        """Simple weighted linear regression."""
        n_features = len(X[0])
        # Gradient descent for weighted least squares
        coeffs = [0.0] * n_features
        bias = sum(w[i] * y[i] for i in range(len(y))) / max(sum(w), 1e-10)
        
        lr = 0.001
        for _ in range(100):
            for i in range(len(X)):
                pred = bias + sum(coeffs[j] * X[i][j]
                                for j in range(n_features))
                error = pred - y[i]
                
                for j in range(n_features):
                    coeffs[j] -= lr * w[i] * error * X[i][j]
                bias -= lr * w[i] * error
        
        return coeffs


# ============================================================
# Partial Dependence
# ============================================================

class PartialDependence:
    """Partial dependence plots."""
    
    def __init__(self, model_predict: Callable):
        self.model_predict = model_predict
    
    def compute_1d(self, X: List[List[float]], feature_idx: int,
                   grid_points: int = 50) -> Tuple[List[float], List[float]]:
        """1D partial dependence for a single feature."""
        values = sorted(set(x[feature_idx] for x in X))
        
        if len(values) > grid_points:
            step = max(1, len(values) // grid_points)
            values = values[::step]
        
        pdp_values = []
        for val in values:
            # Set feature to val for all samples
            X_modified = [row[:] for row in X]
            for row in X_modified:
                row[feature_idx] = val
            
            preds = self.model_predict(X_modified)
            if isinstance(preds, list):
                mean_pred = sum(preds) / len(preds)
            else:
                mean_pred = preds
            pdp_values.append(mean_pred)
        
        return values, pdp_values
    
    def compute_ice(self, X: List[List[float]], feature_idx: int,
                    grid_points: int = 50,
                    n_instances: int = 50) -> Tuple[
                        List[float], List[List[float]]]:
        """Individual Conditional Expectation curves."""
        values = sorted(set(x[feature_idx] for x in X))
        
        if len(values) > grid_points:
            step = max(1, len(values) // grid_points)
            values = values[::step]
        
        sample = random.sample(X, min(n_instances, len(X)))
        
        ice_curves = []
        for instance in sample:
            curve = []
            for val in values:
                x_mod = instance[:]
                x_mod[feature_idx] = val
                pred = self.model_predict([x_mod])
                curve.append(pred[0] if isinstance(pred, list) else pred)
            ice_curves.append(curve)
        
        return values, ice_curves


# ============================================================
# Counterfactual Explanations
# ============================================================

class CounterfactualExplainer:
    """Find counterfactual explanations."""
    
    def __init__(self, model_predict: Callable,
                 feature_ranges: List[Tuple[float, float]],
                 immutable_features: List[int] = None):
        self.model_predict = model_predict
        self.feature_ranges = feature_ranges
        self.immutable = set(immutable_features or [])
    
    def explain(self, instance: List[float],
                desired_class: int,
                n_counterfactuals: int = 5,
                max_iters: int = 1000,
                lr: float = 0.01,
                lambda_dist: float = 0.1) -> List[Dict]:
        """Generate counterfactual explanations."""
        counterfactuals = []
        
        for _ in range(n_counterfactuals):
            cf = instance[:]
            
            # Add random perturbation
            for j in range(len(cf)):
                if j not in self.immutable:
                    lo, hi = self.feature_ranges[j]
                    cf[j] += random.gauss(0, (hi - lo) * 0.1)
                    cf[j] = max(lo, min(hi, cf[j]))
            
            for _ in range(max_iters):
                pred = self.model_predict([cf])
                pred_val = pred[0] if isinstance(pred, list) else pred
                
                if (desired_class == 1 and pred_val > 0.5) or \
                   (desired_class == 0 and pred_val < 0.5):
                    break
                
                # Gradient-free optimization: random perturbation
                for j in range(len(cf)):
                    if j not in self.immutable:
                        lo, hi = self.feature_ranges[j]
                        direction = (1 if desired_class == 1 else -1)
                        cf[j] += direction * random.gauss(0, lr * (hi - lo))
                        cf[j] = max(lo, min(hi, cf[j]))
            
            pred = self.model_predict([cf])
            pred_val = pred[0] if isinstance(pred, list) else pred
            
            distance = math.sqrt(sum((a - b) ** 2
                                    for a, b in zip(instance, cf)))
            n_changed = sum(1 for a, b in zip(instance, cf)
                          if abs(a - b) > 1e-6)
            
            changes = {}
            for j in range(len(instance)):
                if abs(instance[j] - cf[j]) > 1e-6:
                    changes[j] = {
                        "from": instance[j],
                        "to": cf[j],
                        "delta": cf[j] - instance[j],
                    }
            
            counterfactuals.append({
                "counterfactual": cf,
                "prediction": pred_val,
                "distance": distance,
                "n_changed": n_changed,
                "changes": changes,
            })
        
        counterfactuals.sort(key=lambda x: x["distance"])
        return counterfactuals


# ============================================================
# Fairness Metrics
# ============================================================

class FairnessAnalyzer:
    """Compute fairness metrics across groups."""
    
    def __init__(self, y_true: List[int], y_pred: List[int],
                 sensitive_attr: List[int]):
        self.y_true = y_true
        self.y_pred = y_pred
        self.sensitive = sensitive_attr
        self.groups = sorted(set(sensitive_attr))
    
    def demographic_parity(self) -> Dict[int, float]:
        """P(Y_hat=1 | A=a) for each group."""
        rates = {}
        for g in self.groups:
            mask = [i for i, a in enumerate(self.sensitive) if a == g]
            if mask:
                rates[g] = sum(self.y_pred[i] for i in mask) / len(mask)
            else:
                rates[g] = 0.0
        return rates
    
    def equalized_odds(self) -> Dict[int, Dict[str, float]]:
        """TPR and FPR for each group."""
        metrics = {}
        for g in self.groups:
            mask = [i for i, a in enumerate(self.sensitive) if a == g]
            
            tp = sum(1 for i in mask
                    if self.y_true[i] == 1 and self.y_pred[i] == 1)
            fn = sum(1 for i in mask
                    if self.y_true[i] == 1 and self.y_pred[i] == 0)
            fp = sum(1 for i in mask
                    if self.y_true[i] == 0 and self.y_pred[i] == 1)
            tn = sum(1 for i in mask
                    if self.y_true[i] == 0 and self.y_pred[i] == 0)
            
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            
            metrics[g] = {"tpr": tpr, "fpr": fpr}
        
        return metrics
    
    def disparate_impact(self) -> float:
        """Ratio of positive rates between groups."""
        rates = self.demographic_parity()
        if len(rates) < 2:
            return 1.0
        
        vals = list(rates.values())
        return min(vals) / max(max(vals), 1e-10)
    
    def summary(self) -> Dict:
        dp = self.demographic_parity()
        eo = self.equalized_odds()
        di = self.disparate_impact()
        
        dp_diff = max(dp.values()) - min(dp.values())
        tpr_diff = max(eo[g]["tpr"] for g in self.groups) - \
                   min(eo[g]["tpr"] for g in self.groups)
        
        return {
            "demographic_parity": dp,
            "dp_difference": dp_diff,
            "equalized_odds": eo,
            "tpr_difference": tpr_diff,
            "disparate_impact": di,
            "fair_by_dp": dp_diff < 0.1,
            "fair_by_di": di > 0.8,
        }`,
				},
			},
		},
	})
}
