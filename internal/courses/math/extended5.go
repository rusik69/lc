package math

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMathModules([]problems.CourseModule{
		{
			ID:          2719,
			Title:       "Information Theory and Combinatorics",
			Description: "Master information theory concepts including entropy, KL divergence, mutual information, and coding theory, along with advanced combinatorics for algorithm design.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "Information Theory",
					Content: `Information theory, founded by Claude Shannon, quantifies information content and provides the mathematical basis for data compression, communication, and machine learning loss functions.

**Entropy:**

Shannon entropy measures uncertainty/information content:
  H(X) = -Σ P(x) log₂ P(x)

Properties:
  H(X) ≥ 0 (non-negative)
  H(X) = 0 iff X is deterministic
  H(X) ≤ log₂(n) where n = number of outcomes
  Maximum when uniform: H = log₂(n)

Example:
  Fair coin: H = -0.5 log₂(0.5) - 0.5 log₂(0.5) = 1 bit
  Biased coin (p=0.9): H = -0.9 log₂(0.9) - 0.1 log₂(0.1) ≈ 0.47 bits
  Certain event: H = 0 bits

Binary entropy function:
  H(p) = -p log₂(p) - (1-p) log₂(1-p)
  Maximum at p = 0.5

Joint entropy:
  H(X,Y) = -Σ P(x,y) log₂ P(x,y)
  H(X,Y) ≤ H(X) + H(Y) (equality iff independent)

Conditional entropy:
  H(Y|X) = -Σ P(x,y) log₂ P(y|x)
  H(Y|X) = H(X,Y) - H(X)
  H(Y|X) ≤ H(Y) (conditioning reduces entropy)

Chain rule:
  H(X,Y) = H(X) + H(Y|X)
  H(X₁,...,Xₙ) = Σ H(Xᵢ|X₁,...,Xᵢ₋₁)

**Cross-Entropy:**

H(P,Q) = -Σ P(x) log Q(x)

Properties:
  H(P,Q) ≥ H(P) (always ≥ true entropy)
  H(P,Q) = H(P) iff P = Q
  Used as loss function in classification

Binary cross-entropy:
  H(y, ŷ) = -[y log(ŷ) + (1-y) log(1-ŷ)]
  Standard loss for binary classification

Categorical cross-entropy:
  H(y, ŷ) = -Σ yᵢ log(ŷᵢ)
  Standard loss for multi-class classification

**KL Divergence:**

Kullback-Leibler divergence (relative entropy):
  D_KL(P || Q) = Σ P(x) log(P(x)/Q(x))
  = H(P,Q) - H(P)

Properties:
  D_KL ≥ 0 (Gibbs' inequality)
  D_KL = 0 iff P = Q
  Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
  Not a true metric (no triangle inequality)

Forward KL (D_KL(P||Q)): Mode-covering
  Q tries to cover all of P's support
  Penalizes Q(x) = 0 where P(x) > 0

Reverse KL (D_KL(Q||P)): Mode-seeking
  Q focuses on peak of P
  Used in variational inference

**Mutual Information:**

I(X;Y) = H(X) + H(Y) - H(X,Y)
       = H(X) - H(X|Y)
       = D_KL(P(X,Y) || P(X)P(Y))

Properties:
  I(X;Y) ≥ 0
  I(X;Y) = 0 iff X and Y are independent
  I(X;Y) = I(Y;X) (symmetric)
  I(X;X) = H(X) (self-information = entropy)

Applications:
  Feature selection: Features with high MI with target
  Decision trees: Information gain = MI
  Representation learning: Maximize MI between input and representation

**Jensen-Shannon Divergence:**

JSD(P || Q) = ½ D_KL(P || M) + ½ D_KL(Q || M)
  where M = ½(P + Q)

Properties:
  Symmetric: JSD(P||Q) = JSD(Q||P)
  Bounded: 0 ≤ JSD ≤ log(2) (using log base 2)
  √JSD is a proper metric
  Used in GAN training (original formulation)

**Source Coding Theorem:**

Shannon's source coding theorem:
  Minimum average code length = H(X)
  Cannot compress below entropy without loss

Huffman coding: Optimal prefix-free code
  Achieves rate close to H(X)
  Prefix-free: No codeword is prefix of another

Arithmetic coding: Can achieve rate even closer to H(X)
  Encode entire message as single number in [0,1)

**Channel Coding:**

Noisy channel: Input X → Channel → Output Y
  Channel capacity: C = max_P(X) I(X;Y)
  
Shannon's channel coding theorem:
  Can communicate at rate R < C with arbitrarily small error
  Cannot communicate reliably at R > C

Binary symmetric channel:
  C = 1 - H(p) where p = error probability
  
Binary erasure channel:
  C = 1 - ε where ε = erasure probability

**Rate-Distortion Theory:**

Lossy compression: How much distortion for given compression rate?
  R(D) = min_{P(x̂|x): E[d(x,x̂)]≤D} I(X;X̂)
  
  Trade-off between rate (bits) and distortion
  Foundation for image/video compression

**Information Theory in ML:**

Loss functions:
  Cross-entropy = -log-likelihood (for categorical distributions)
  KL divergence in VAEs: regularization toward prior
  Mutual information in contrastive learning (InfoNCE)

Decision trees:
  Information gain = H(Y) - H(Y|X)
  Choose split that maximizes information gain
  Equivalent to maximizing mutual information

Variational inference:
  ELBO = E_q[log p(x|z)] - D_KL(q(z|x) || p(z))
  Maximize ELBO ≈ minimize KL between approximate and true posterior`,
					CodeExamples: `# Information Theory Implementations

import math
import random
from typing import Dict, List, Optional, Tuple
from collections import Counter
import heapq

# ============================================================
# Entropy Calculations
# ============================================================

class InformationTheory:
    """Information theory utilities."""
    
    @staticmethod
    def entropy(probs: List[float], base: float = 2) -> float:
        """Shannon entropy."""
        H = 0.0
        log_base = math.log(base)
        for p in probs:
            if p > 0:
                H -= p * math.log(p) / log_base
        return H
    
    @staticmethod
    def cross_entropy(p: List[float], q: List[float],
                      base: float = 2) -> float:
        """Cross-entropy H(P, Q)."""
        H = 0.0
        log_base = math.log(base)
        for pi, qi in zip(p, q):
            if pi > 0:
                H -= pi * math.log(max(qi, 1e-15)) / log_base
        return H
    
    @staticmethod
    def kl_divergence(p: List[float], q: List[float],
                      base: float = 2) -> float:
        """KL divergence D_KL(P || Q)."""
        D = 0.0
        log_base = math.log(base)
        for pi, qi in zip(p, q):
            if pi > 0:
                D += pi * math.log(pi / max(qi, 1e-15)) / log_base
        return D
    
    @staticmethod
    def js_divergence(p: List[float], q: List[float],
                      base: float = 2) -> float:
        """Jensen-Shannon divergence."""
        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
        return 0.5 * (InformationTheory.kl_divergence(p, m, base) +
                      InformationTheory.kl_divergence(q, m, base))
    
    @staticmethod
    def mutual_information(joint: List[List[float]],
                          base: float = 2) -> float:
        """Mutual information from joint distribution."""
        rows = len(joint)
        cols = len(joint[0])
        
        # Marginals
        px = [sum(joint[i]) for i in range(rows)]
        py = [sum(joint[i][j] for i in range(rows)) for j in range(cols)]
        
        MI = 0.0
        log_base = math.log(base)
        for i in range(rows):
            for j in range(cols):
                if joint[i][j] > 0 and px[i] > 0 and py[j] > 0:
                    MI += joint[i][j] * math.log(
                        joint[i][j] / (px[i] * py[j])) / log_base
        
        return MI
    
    @staticmethod
    def conditional_entropy(joint: List[List[float]],
                           base: float = 2) -> float:
        """H(Y|X) from joint distribution P(X,Y)."""
        H_XY = InformationTheory.entropy(
            [joint[i][j] for i in range(len(joint))
             for j in range(len(joint[0]))], base)
        
        px = [sum(row) for row in joint]
        H_X = InformationTheory.entropy(px, base)
        
        return H_XY - H_X
    
    @staticmethod
    def information_gain(y: List[int],
                         x: List[int]) -> float:
        """Information gain (mutual information from data)."""
        n = len(y)
        
        # H(Y)
        y_counts = Counter(y)
        H_Y = InformationTheory.entropy(
            [c / n for c in y_counts.values()])
        
        # H(Y|X)
        x_values = set(x)
        H_Y_X = 0.0
        for xv in x_values:
            indices = [i for i in range(n) if x[i] == xv]
            weight = len(indices) / n
            y_sub = [y[i] for i in indices]
            y_sub_counts = Counter(y_sub)
            H_Y_X += weight * InformationTheory.entropy(
                [c / len(y_sub) for c in y_sub_counts.values()])
        
        return H_Y - H_Y_X


# ============================================================
# Huffman Coding
# ============================================================

class HuffmanNode:
    def __init__(self, char: Optional[str], freq: float):
        self.char = char
        self.freq = freq
        self.left: Optional[HuffmanNode] = None
        self.right: Optional[HuffmanNode] = None
    
    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanCoding:
    """Huffman coding for data compression."""
    
    def __init__(self):
        self.codes: Dict[str, str] = {}
        self.reverse_codes: Dict[str, str] = {}
    
    def build_tree(self, text: str) -> HuffmanNode:
        freq = Counter(text)
        heap = [HuffmanNode(char, f) for char, f in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = HuffmanNode(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            
            heapq.heappush(heap, merged)
        
        root = heap[0] if heap else HuffmanNode(None, 0)
        self._build_codes(root, "")
        self.reverse_codes = {v: k for k, v in self.codes.items()}
        
        return root
    
    def _build_codes(self, node: Optional[HuffmanNode], code: str):
        if node is None:
            return
        
        if node.char is not None:
            self.codes[node.char] = code if code else "0"
            return
        
        self._build_codes(node.left, code + "0")
        self._build_codes(node.right, code + "1")
    
    def encode(self, text: str) -> str:
        return "".join(self.codes[c] for c in text)
    
    def decode(self, encoded: str, root: HuffmanNode) -> str:
        result = []
        node = root
        
        for bit in encoded:
            if bit == "0":
                node = node.left
            else:
                node = node.right
            
            if node and node.char is not None:
                result.append(node.char)
                node = root
        
        return "".join(result)
    
    def compression_ratio(self, text: str) -> Dict[str, float]:
        original_bits = len(text) * 8
        encoded = self.encode(text)
        compressed_bits = len(encoded)
        
        freq = Counter(text)
        n = len(text)
        entropy = InformationTheory.entropy(
            [c / n for c in freq.values()])
        
        avg_code_length = sum(
            len(self.codes[c]) * freq[c] / n
            for c in self.codes)
        
        return {
            "original_bits": original_bits,
            "compressed_bits": compressed_bits,
            "ratio": compressed_bits / max(original_bits, 1),
            "entropy": entropy,
            "avg_code_length": avg_code_length,
        }


# ============================================================
# Combinatorics
# ============================================================

class Combinatorics:
    """Combinatorial computations."""
    
    @staticmethod
    def factorial(n: int) -> int:
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    
    @staticmethod
    def permutations(n: int, r: int) -> int:
        """P(n, r) = n! / (n-r)!"""
        if r > n:
            return 0
        result = 1
        for i in range(n, n - r, -1):
            result *= i
        return result
    
    @staticmethod
    def combinations(n: int, r: int) -> int:
        """C(n, r) = n! / (r! * (n-r)!)"""
        if r > n:
            return 0
        r = min(r, n - r)
        result = 1
        for i in range(r):
            result = result * (n - i) // (i + 1)
        return result
    
    @staticmethod
    def catalan(n: int) -> int:
        """n-th Catalan number: C(2n,n)/(n+1)"""
        return Combinatorics.combinations(2 * n, n) // (n + 1)
    
    @staticmethod
    def stirling_second(n: int, k: int) -> int:
        """Stirling numbers of the second kind: partitions of n into k subsets."""
        if n == 0 and k == 0:
            return 1
        if n == 0 or k == 0:
            return 0
        
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                dp[i][j] = j * dp[i-1][j] + dp[i-1][j-1]
        
        return dp[n][k]
    
    @staticmethod
    def bell_number(n: int) -> int:
        """Bell number: total partitions of n elements."""
        return sum(Combinatorics.stirling_second(n, k) for k in range(n + 1))
    
    @staticmethod
    def derangements(n: int) -> int:
        """Number of derangements (permutations with no fixed points)."""
        if n == 0:
            return 1
        if n == 1:
            return 0
        
        d = [0] * (n + 1)
        d[0] = 1
        d[1] = 0
        
        for i in range(2, n + 1):
            d[i] = (i - 1) * (d[i-1] + d[i-2])
        
        return d[n]
    
    @staticmethod
    def multinomial(n: int, groups: List[int]) -> int:
        """Multinomial coefficient: n! / (k1! * k2! * ... * km!)"""
        result = Combinatorics.factorial(n)
        for k in groups:
            result //= Combinatorics.factorial(k)
        return result
    
    @staticmethod
    def stars_and_bars(n: int, k: int) -> int:
        """Distribute n identical items into k distinct bins."""
        return Combinatorics.combinations(n + k - 1, k - 1)
    
    @staticmethod
    def fibonacci(n: int) -> int:
        """n-th Fibonacci number using matrix exponentiation."""
        if n <= 0:
            return 0
        if n == 1:
            return 1
        
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    @staticmethod
    def generate_permutations(items: List) -> List[List]:
        """Generate all permutations."""
        if len(items) <= 1:
            return [items[:]]
        
        result = []
        for i in range(len(items)):
            first = items[i]
            rest = items[:i] + items[i+1:]
            for perm in Combinatorics.generate_permutations(rest):
                result.append([first] + perm)
        
        return result
    
    @staticmethod
    def generate_combinations(items: List, r: int) -> List[List]:
        """Generate all r-combinations."""
        if r == 0:
            return [[]]
        if len(items) < r:
            return []
        
        # Include first item
        with_first = [[items[0]] + c
                      for c in Combinatorics.generate_combinations(items[1:], r-1)]
        # Exclude first item
        without_first = Combinatorics.generate_combinations(items[1:], r)
        
        return with_first + without_first
    
    @staticmethod
    def inclusion_exclusion(sets: List[set]) -> int:
        """Size of union using inclusion-exclusion principle."""
        n = len(sets)
        total = 0
        
        for size in range(1, n + 1):
            for combo in Combinatorics.generate_combinations(
                    list(range(n)), size):
                intersection = sets[combo[0]]
                for idx in combo[1:]:
                    intersection = intersection & sets[idx]
                
                if size % 2 == 1:
                    total += len(intersection)
                else:
                    total -= len(intersection)
        
        return total`,
				},
			},
		},
	})
}
