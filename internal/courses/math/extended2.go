package math

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMathModules([]problems.CourseModule{
		{
			ID:          2716,
			Title:       "Probability and Statistics",
			Description: "Master probability theory, statistical distributions, hypothesis testing, Bayesian inference, and statistical methods essential for data science and machine learning.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "Probability Theory Foundations",
					Content: `Probability theory provides the mathematical foundation for reasoning about uncertainty. It is essential for machine learning, data science, algorithms, and decision making.

**Basic Probability:**

Sample space (Ω): Set of all possible outcomes
Event (A): Subset of sample space
Probability P(A): Measure of likelihood, 0 ≤ P(A) ≤ 1

Axioms of probability:
  1. P(A) ≥ 0 for all events A
  2. P(Ω) = 1
  3. If A and B are mutually exclusive: P(A ∪ B) = P(A) + P(B)

Complement: P(A') = 1 - P(A)
Union: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
Inclusion-Exclusion: P(A∪B∪C) = P(A)+P(B)+P(C)-P(A∩B)-P(A∩C)-P(B∩C)+P(A∩B∩C)

**Conditional Probability:**

P(A|B) = P(A ∩ B) / P(B)

"Probability of A given B has occurred"
  P(A|B) × P(B) = P(A ∩ B)

Independence:
  A and B independent iff P(A ∩ B) = P(A) × P(B)
  Equivalently: P(A|B) = P(A)

Conditional independence:
  A ⊥ B | C iff P(A ∩ B | C) = P(A|C) × P(B|C)

**Bayes' Theorem:**

P(A|B) = P(B|A) × P(A) / P(B)

Components:
  P(A|B): Posterior — updated belief after evidence
  P(B|A): Likelihood — probability of evidence given hypothesis
  P(A): Prior — initial belief before evidence
  P(B): Evidence — total probability of observation

Law of Total Probability:
  P(B) = Σᵢ P(B|Aᵢ) × P(Aᵢ)
  where {Aᵢ} is a partition of sample space

Applications:
  Medical testing: P(disease | positive test)
  Spam filtering: P(spam | word appears)
  Machine learning: Bayesian inference

**Random Variables:**

Discrete random variable:
  Takes countable values
  PMF: P(X = x) for each value
  Σ P(X = x) = 1

Continuous random variable:
  Takes values in interval
  PDF: f(x) such that P(a ≤ X ≤ b) = ∫ₐᵇ f(x)dx
  f(x) ≥ 0 and ∫ f(x)dx = 1

CDF (Cumulative Distribution Function):
  F(x) = P(X ≤ x)
  F is non-decreasing, F(-∞) = 0, F(∞) = 1
  P(a < X ≤ b) = F(b) - F(a)

**Expectation and Variance:**

Expected value (mean):
  Discrete: E[X] = Σ x × P(X = x)
  Continuous: E[X] = ∫ x × f(x) dx

Properties:
  E[aX + b] = aE[X] + b (linearity)
  E[X + Y] = E[X] + E[Y] (always)
  E[XY] = E[X]E[Y] (if independent)

Variance:
  Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
  Standard deviation: σ = √Var(X)

Properties:
  Var(aX + b) = a²Var(X)
  Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)
  If independent: Var(X + Y) = Var(X) + Var(Y)

Covariance:
  Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] = E[XY] - E[X]E[Y]
  Cov(X,X) = Var(X)
  
Correlation:
  ρ(X,Y) = Cov(X,Y) / (σₓ × σᵧ)
  -1 ≤ ρ ≤ 1
  ρ = 0: uncorrelated (not necessarily independent)

**Common Discrete Distributions:**

Bernoulli(p):
  X ∈ {0, 1}
  P(X=1) = p, P(X=0) = 1-p
  E[X] = p, Var(X) = p(1-p)
  Example: Coin flip

Binomial(n, p):
  X = number of successes in n independent trials
  P(X=k) = C(n,k) × p^k × (1-p)^(n-k)
  E[X] = np, Var(X) = np(1-p)
  Example: Number of heads in 10 coin flips

Geometric(p):
  X = number of trials until first success
  P(X=k) = (1-p)^(k-1) × p
  E[X] = 1/p, Var(X) = (1-p)/p²
  Memoryless property: P(X > s+t | X > s) = P(X > t)

Poisson(λ):
  X = number of events in fixed interval
  P(X=k) = e^(-λ) × λ^k / k!
  E[X] = λ, Var(X) = λ
  Approximates Binomial when n large, p small, λ = np

Negative Binomial(r, p):
  X = number of trials until r-th success
  Generalizes Geometric (r=1)

Hypergeometric(N, K, n):
  Sampling without replacement
  N items, K successes, n draws

**Common Continuous Distributions:**

Uniform(a, b):
  f(x) = 1/(b-a) for a ≤ x ≤ b
  E[X] = (a+b)/2, Var(X) = (b-a)²/12

Normal (Gaussian) N(μ, σ²):
  f(x) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))
  E[X] = μ, Var(X) = σ²
  68-95-99.7 rule: within 1σ, 2σ, 3σ of mean
  Standard normal: Z = (X-μ)/σ ~ N(0,1)
  Sum of normals is normal: X+Y ~ N(μ₁+μ₂, σ₁²+σ₂²)

Exponential(λ):
  f(x) = λe^(-λx) for x ≥ 0
  E[X] = 1/λ, Var(X) = 1/λ²
  Memoryless: P(X > s+t | X > s) = P(X > t)
  Time between Poisson events

Beta(α, β):
  f(x) ∝ x^(α-1)(1-x)^(β-1) for x ∈ [0,1]
  E[X] = α/(α+β)
  Conjugate prior for Bernoulli/Binomial

Gamma(α, β):
  Generalizes exponential
  Sum of α exponential RVs

Chi-squared(k):
  Sum of k squared standard normals
  Used in hypothesis testing

Student's t(ν):
  Heavier tails than normal
  Used for small samples
  Approaches normal as ν → ∞

**Moment Generating Functions:**

M(t) = E[e^(tX)]
  E[X] = M'(0)
  E[X²] = M''(0)
  Uniquely determines distribution
  
  Normal: M(t) = exp(μt + σ²t²/2)
  Poisson: M(t) = exp(λ(e^t - 1))

**Inequalities:**

Markov: P(X ≥ a) ≤ E[X]/a (X ≥ 0)
Chebyshev: P(|X-μ| ≥ kσ) ≤ 1/k²
Hoeffding: P(|X̄-μ| ≥ t) ≤ 2exp(-2nt²/(b-a)²)
Chernoff: P(X ≥ a) ≤ min_t e^(-ta)M(t)

**Limit Theorems:**

Law of Large Numbers:
  X̄ₙ → μ as n → ∞ (almost surely)
  Sample mean converges to population mean

Central Limit Theorem:
  √n(X̄ₙ - μ)/σ → N(0,1) as n → ∞
  Sum of many independent RVs is approximately normal
  Regardless of original distribution
  Foundation of statistical inference`,
					CodeExamples: `# Probability and Statistics Implementations

import math
import random
from typing import Callable, Dict, List, Optional, Tuple
from collections import Counter

# ============================================================
# Probability Distributions
# ============================================================

class BernoulliDistribution:
    """Bernoulli distribution."""
    
    def __init__(self, p: float):
        self.p = p
    
    def pmf(self, k: int) -> float:
        if k == 1:
            return self.p
        elif k == 0:
            return 1 - self.p
        return 0.0
    
    def sample(self) -> int:
        return 1 if random.random() < self.p else 0
    
    def mean(self) -> float:
        return self.p
    
    def variance(self) -> float:
        return self.p * (1 - self.p)


class BinomialDistribution:
    """Binomial distribution."""
    
    def __init__(self, n: int, p: float):
        self.n = n
        self.p = p
    
    def pmf(self, k: int) -> float:
        if k < 0 or k > self.n:
            return 0.0
        coeff = math.comb(self.n, k)
        return coeff * (self.p ** k) * ((1 - self.p) ** (self.n - k))
    
    def cdf(self, k: int) -> float:
        return sum(self.pmf(i) for i in range(k + 1))
    
    def sample(self) -> int:
        return sum(1 for _ in range(self.n) if random.random() < self.p)
    
    def mean(self) -> float:
        return self.n * self.p
    
    def variance(self) -> float:
        return self.n * self.p * (1 - self.p)


class PoissonDistribution:
    """Poisson distribution."""
    
    def __init__(self, lam: float):
        self.lam = lam
    
    def pmf(self, k: int) -> float:
        if k < 0:
            return 0.0
        return math.exp(-self.lam) * (self.lam ** k) / math.factorial(k)
    
    def cdf(self, k: int) -> float:
        return sum(self.pmf(i) for i in range(k + 1))
    
    def sample(self) -> int:
        """Knuth's algorithm."""
        L = math.exp(-self.lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1
    
    def mean(self) -> float:
        return self.lam
    
    def variance(self) -> float:
        return self.lam


class GeometricDistribution:
    """Geometric distribution (trials until first success)."""
    
    def __init__(self, p: float):
        self.p = p
    
    def pmf(self, k: int) -> float:
        if k < 1:
            return 0.0
        return ((1 - self.p) ** (k - 1)) * self.p
    
    def cdf(self, k: int) -> float:
        if k < 1:
            return 0.0
        return 1 - (1 - self.p) ** k
    
    def sample(self) -> int:
        k = 1
        while random.random() >= self.p:
            k += 1
        return k
    
    def mean(self) -> float:
        return 1.0 / self.p
    
    def variance(self) -> float:
        return (1 - self.p) / (self.p ** 2)


class NormalDistribution:
    """Normal (Gaussian) distribution."""
    
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma
    
    def pdf(self, x: float) -> float:
        z = (x - self.mu) / self.sigma
        return math.exp(-0.5 * z * z) / (self.sigma * math.sqrt(2 * math.pi))
    
    def cdf(self, x: float) -> float:
        """Approximate CDF using error function."""
        z = (x - self.mu) / (self.sigma * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))
    
    def sample(self) -> float:
        """Box-Muller transform."""
        u1 = max(random.random(), 1e-10)
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return self.mu + self.sigma * z
    
    def sample_n(self, n: int) -> List[float]:
        return [self.sample() for _ in range(n)]
    
    def mean(self) -> float:
        return self.mu
    
    def variance(self) -> float:
        return self.sigma ** 2
    
    def z_score(self, x: float) -> float:
        return (x - self.mu) / self.sigma
    
    def quantile(self, p: float) -> float:
        """Approximate inverse CDF (quantile function)."""
        # Rational approximation for standard normal
        if p <= 0:
            return float('-inf')
        if p >= 1:
            return float('inf')
        
        if p < 0.5:
            t = math.sqrt(-2 * math.log(p))
        else:
            t = math.sqrt(-2 * math.log(1 - p))
        
        # Abramowitz and Stegun approximation
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        
        z = t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)
        
        if p < 0.5:
            z = -z
        
        return self.mu + self.sigma * z


class ExponentialDistribution:
    """Exponential distribution."""
    
    def __init__(self, lam: float):
        self.lam = lam
    
    def pdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return self.lam * math.exp(-self.lam * x)
    
    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return 1 - math.exp(-self.lam * x)
    
    def sample(self) -> float:
        return -math.log(max(random.random(), 1e-10)) / self.lam
    
    def mean(self) -> float:
        return 1.0 / self.lam
    
    def variance(self) -> float:
        return 1.0 / (self.lam ** 2)


class BetaDistribution:
    """Beta distribution."""
    
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
    
    def pdf(self, x: float) -> float:
        if x <= 0 or x >= 1:
            return 0.0
        B = (math.gamma(self.alpha) * math.gamma(self.beta) /
             math.gamma(self.alpha + self.beta))
        return (x ** (self.alpha - 1)) * ((1 - x) ** (self.beta - 1)) / B
    
    def sample(self) -> float:
        """Approximate using gamma samples."""
        x = sum(-math.log(max(random.random(), 1e-10))
                for _ in range(max(1, int(self.alpha))))
        y = sum(-math.log(max(random.random(), 1e-10))
                for _ in range(max(1, int(self.beta))))
        return x / max(x + y, 1e-10)
    
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    def variance(self) -> float:
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1))


# ============================================================
# Bayesian Inference
# ============================================================

class BayesianBernoulli:
    """Bayesian inference for Bernoulli parameter with Beta prior."""
    
    def __init__(self, prior_alpha: float = 1.0,
                 prior_beta: float = 1.0):
        self.alpha = prior_alpha
        self.beta = prior_beta
    
    def update(self, data: List[int]):
        """Update posterior with observed data."""
        successes = sum(data)
        failures = len(data) - successes
        self.alpha += successes
        self.beta += failures
    
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    def posterior_mode(self) -> float:
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return self.posterior_mean()
    
    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Approximate credible interval."""
        mean = self.posterior_mean()
        var = self.posterior_variance()
        std = math.sqrt(var)
        z = NormalDistribution().quantile(0.5 + level / 2)
        return (max(0, mean - z * std), min(1, mean + z * std))
    
    def posterior_variance(self) -> float:
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1))
    
    def predictive_probability(self) -> float:
        """Probability next observation is 1."""
        return self.posterior_mean()


# ============================================================
# Descriptive Statistics
# ============================================================

class DescriptiveStats:
    """Compute descriptive statistics."""
    
    @staticmethod
    def mean(data: List[float]) -> float:
        return sum(data) / len(data)
    
    @staticmethod
    def median(data: List[float]) -> float:
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 1:
            return sorted_data[n // 2]
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    
    @staticmethod
    def mode(data: List[float]) -> float:
        counter = Counter(data)
        return counter.most_common(1)[0][0]
    
    @staticmethod
    def variance(data: List[float], ddof: int = 0) -> float:
        m = sum(data) / len(data)
        return sum((x - m) ** 2 for x in data) / (len(data) - ddof)
    
    @staticmethod
    def std(data: List[float], ddof: int = 0) -> float:
        return math.sqrt(DescriptiveStats.variance(data, ddof))
    
    @staticmethod
    def percentile(data: List[float], p: float) -> float:
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
    
    @staticmethod
    def iqr(data: List[float]) -> float:
        q1 = DescriptiveStats.percentile(data, 25)
        q3 = DescriptiveStats.percentile(data, 75)
        return q3 - q1
    
    @staticmethod
    def skewness(data: List[float]) -> float:
        n = len(data)
        m = sum(data) / n
        s = math.sqrt(sum((x - m) ** 2 for x in data) / n)
        if s == 0:
            return 0.0
        return sum(((x - m) / s) ** 3 for x in data) / n
    
    @staticmethod
    def kurtosis(data: List[float]) -> float:
        n = len(data)
        m = sum(data) / n
        s = math.sqrt(sum((x - m) ** 2 for x in data) / n)
        if s == 0:
            return 0.0
        return sum(((x - m) / s) ** 4 for x in data) / n - 3
    
    @staticmethod
    def covariance(x: List[float], y: List[float]) -> float:
        n = len(x)
        mx = sum(x) / n
        my = sum(y) / n
        return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n
    
    @staticmethod
    def correlation(x: List[float], y: List[float]) -> float:
        cov = DescriptiveStats.covariance(x, y)
        sx = DescriptiveStats.std(x)
        sy = DescriptiveStats.std(y)
        if sx * sy == 0:
            return 0.0
        return cov / (sx * sy)`,
				},
				{
					Title: "Hypothesis Testing and Statistical Inference",
					Content: `Hypothesis testing provides a framework for making decisions based on data. It is fundamental to A/B testing, scientific research, and validating ML model improvements.

**Hypothesis Testing Framework:**

Null hypothesis (H₀): Default assumption (no effect, no difference)
Alternative hypothesis (H₁): What we want to show (there IS an effect)

Types:
  One-sided: H₁: μ > μ₀  or  H₁: μ < μ₀
  Two-sided: H₁: μ ≠ μ₀

Test statistic: Summary of data relevant to hypotheses
p-value: P(data as extreme or more | H₀ is true)
Significance level (α): Threshold for rejecting H₀ (typically 0.05)

Decision rule:
  p-value < α → Reject H₀ (statistically significant)
  p-value ≥ α → Fail to reject H₀ (not significant)

**Type I and Type II Errors:**

Type I error (false positive):
  Reject H₀ when H₀ is true
  P(Type I) = α (significance level)

Type II error (false negative):
  Fail to reject H₀ when H₁ is true
  P(Type II) = β

Power = 1 - β
  Probability of correctly rejecting H₀ when false
  Depends on: effect size, sample size, α

**Z-Test:**

For large samples or known variance:
  Z = (X̄ - μ₀) / (σ/√n)
  Under H₀: Z ~ N(0,1)

Two-sample Z-test:
  Z = (X̄₁ - X̄₂) / √(σ₁²/n₁ + σ₂²/n₂)

**T-Test:**

For small samples, unknown variance:
  t = (X̄ - μ₀) / (s/√n)
  Under H₀: t ~ t-distribution with n-1 degrees of freedom

Two-sample t-test (equal variance):
  t = (X̄₁ - X̄₂) / (s_p × √(1/n₁ + 1/n₂))
  s_p = √(((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2))

Welch's t-test (unequal variance):
  t = (X̄₁ - X̄₂) / √(s₁²/n₁ + s₂²/n₂)
  Degrees of freedom by Welch-Satterthwaite

**Chi-Squared Tests:**

Goodness of fit:
  χ² = Σ (observed - expected)² / expected
  Tests if distribution matches expected

Test of independence:
  χ² = Σᵢⱼ (Oᵢⱼ - Eᵢⱼ)² / Eᵢⱼ
  Tests if two categorical variables are independent
  df = (rows - 1) × (cols - 1)

**Confidence Intervals:**

Point estimate ± margin of error
  CI = X̄ ± z_{α/2} × (σ/√n)
  
  95% CI: z = 1.96
  99% CI: z = 2.576

For proportions:
  CI = p̂ ± z_{α/2} × √(p̂(1-p̂)/n)

Interpretation:
  If we repeat experiment many times, 95% of CIs will contain true parameter
  NOT: "95% probability that true value is in this interval"

**Multiple Testing:**

Testing many hypotheses increases false positive rate
  Family-wise error rate (FWER): P(at least one false positive)
  If m independent tests at α: FWER ≈ 1-(1-α)^m

Corrections:
  Bonferroni: α_adjusted = α/m (conservative)
  Benjamini-Hochberg: Controls False Discovery Rate (FDR)
    Sort p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
    Reject all pᵢ ≤ (i/m) × α

**Effect Size:**

Cohen's d = (X̄₁ - X̄₂) / s_pooled
  Small: d = 0.2
  Medium: d = 0.5
  Large: d = 0.8

Statistical significance ≠ practical significance
  Large n can make tiny effects "significant"
  Always report effect size alongside p-value

**Power Analysis:**

Sample size needed:
  n = (z_{α/2} + z_β)² × 2σ² / δ²
  where δ = minimum detectable effect

Factors:
  Larger effect → smaller n needed
  Smaller α → larger n needed
  Higher power → larger n needed
  Less variance → smaller n needed

**Bootstrap:**

Resampling method for estimating distribution of statistic
  1. Sample n observations with replacement from data
  2. Compute statistic on bootstrap sample
  3. Repeat B times (B = 1000-10000)
  4. Use distribution of bootstrap statistics

Bootstrap confidence interval:
  Percentile method: (2.5th percentile, 97.5th percentile)
  BCa: Bias-corrected and accelerated

Advantages:
  No distributional assumptions
  Works for any statistic
  Simple to implement`,
					CodeExamples: `# Hypothesis Testing and Statistical Inference

import math
import random
from typing import Dict, List, Optional, Tuple

# ============================================================
# Hypothesis Tests
# ============================================================

class ZTest:
    """One-sample and two-sample Z-tests."""
    
    @staticmethod
    def one_sample(data: List[float], mu_0: float,
                   sigma: float,
                   alternative: str = "two-sided") -> Dict[str, float]:
        n = len(data)
        x_bar = sum(data) / n
        z = (x_bar - mu_0) / (sigma / math.sqrt(n))
        
        normal = NormalDistribution()
        
        if alternative == "two-sided":
            p_value = 2 * (1 - normal.cdf(abs(z)))
        elif alternative == "greater":
            p_value = 1 - normal.cdf(z)
        else:  # less
            p_value = normal.cdf(z)
        
        return {
            "z_statistic": z,
            "p_value": p_value,
            "sample_mean": x_bar,
            "n": n,
        }
    
    @staticmethod
    def two_sample(data1: List[float], data2: List[float],
                   sigma1: float, sigma2: float,
                   alternative: str = "two-sided") -> Dict[str, float]:
        n1, n2 = len(data1), len(data2)
        x1 = sum(data1) / n1
        x2 = sum(data2) / n2
        
        se = math.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
        z = (x1 - x2) / se
        
        normal = NormalDistribution()
        
        if alternative == "two-sided":
            p_value = 2 * (1 - normal.cdf(abs(z)))
        elif alternative == "greater":
            p_value = 1 - normal.cdf(z)
        else:
            p_value = normal.cdf(z)
        
        return {
            "z_statistic": z,
            "p_value": p_value,
            "mean_diff": x1 - x2,
        }
    
    @staticmethod
    def proportion_test(successes1: int, n1: int,
                        successes2: int, n2: int,
                        alternative: str = "two-sided") -> Dict[str, float]:
        p1 = successes1 / n1
        p2 = successes2 / n2
        p_pool = (successes1 + successes2) / (n1 + n2)
        
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z = (p1 - p2) / max(se, 1e-10)
        
        normal = NormalDistribution()
        
        if alternative == "two-sided":
            p_value = 2 * (1 - normal.cdf(abs(z)))
        elif alternative == "greater":
            p_value = 1 - normal.cdf(z)
        else:
            p_value = normal.cdf(z)
        
        return {
            "z_statistic": z,
            "p_value": p_value,
            "p1": p1,
            "p2": p2,
            "diff": p1 - p2,
        }


class TTest:
    """Student's t-tests."""
    
    @staticmethod
    def one_sample(data: List[float], mu_0: float,
                   alternative: str = "two-sided") -> Dict[str, float]:
        n = len(data)
        x_bar = sum(data) / n
        s = math.sqrt(sum((x - x_bar)**2 for x in data) / (n - 1))
        
        t = (x_bar - mu_0) / (s / math.sqrt(n))
        df = n - 1
        p_value = TTest._t_pvalue(t, df, alternative)
        
        return {
            "t_statistic": t,
            "p_value": p_value,
            "df": df,
            "sample_mean": x_bar,
            "sample_std": s,
        }
    
    @staticmethod
    def two_sample(data1: List[float], data2: List[float],
                   equal_var: bool = True,
                   alternative: str = "two-sided") -> Dict[str, float]:
        n1, n2 = len(data1), len(data2)
        x1 = sum(data1) / n1
        x2 = sum(data2) / n2
        s1_sq = sum((x - x1)**2 for x in data1) / (n1 - 1)
        s2_sq = sum((x - x2)**2 for x in data2) / (n2 - 1)
        
        if equal_var:
            sp_sq = ((n1-1)*s1_sq + (n2-1)*s2_sq) / (n1 + n2 - 2)
            se = math.sqrt(sp_sq * (1/n1 + 1/n2))
            df = n1 + n2 - 2
        else:
            se = math.sqrt(s1_sq/n1 + s2_sq/n2)
            num = (s1_sq/n1 + s2_sq/n2)**2
            den = (s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1)
            df = num / max(den, 1e-10)
        
        t = (x1 - x2) / max(se, 1e-10)
        p_value = TTest._t_pvalue(t, df, alternative)
        
        return {
            "t_statistic": t,
            "p_value": p_value,
            "df": df,
            "mean_diff": x1 - x2,
        }
    
    @staticmethod
    def _t_pvalue(t: float, df: float,
                  alternative: str) -> float:
        """Approximate t-distribution p-value."""
        # For large df, approximate with normal
        if df > 30:
            normal = NormalDistribution()
            if alternative == "two-sided":
                return 2 * (1 - normal.cdf(abs(t)))
            elif alternative == "greater":
                return 1 - normal.cdf(t)
            else:
                return normal.cdf(t)
        
        # Simple approximation using normal with correction
        z = t * (1 - 1/(4*df)) / math.sqrt(1 + t*t/(2*df))
        normal = NormalDistribution()
        
        if alternative == "two-sided":
            return 2 * (1 - normal.cdf(abs(z)))
        elif alternative == "greater":
            return 1 - normal.cdf(z)
        else:
            return normal.cdf(z)


class ChiSquaredTest:
    """Chi-squared tests."""
    
    @staticmethod
    def goodness_of_fit(observed: List[int],
                        expected: List[float]) -> Dict[str, float]:
        chi_sq = sum((o - e)**2 / e
                    for o, e in zip(observed, expected) if e > 0)
        df = len(observed) - 1
        p_value = ChiSquaredTest._chi2_pvalue(chi_sq, df)
        
        return {
            "chi_squared": chi_sq,
            "p_value": p_value,
            "df": df,
        }
    
    @staticmethod
    def independence(contingency_table: List[List[int]]) -> Dict[str, float]:
        rows = len(contingency_table)
        cols = len(contingency_table[0])
        
        row_totals = [sum(row) for row in contingency_table]
        col_totals = [sum(contingency_table[r][c] for r in range(rows))
                     for c in range(cols)]
        total = sum(row_totals)
        
        chi_sq = 0.0
        for r in range(rows):
            for c in range(cols):
                expected = row_totals[r] * col_totals[c] / total
                if expected > 0:
                    chi_sq += (contingency_table[r][c] - expected)**2 / expected
        
        df = (rows - 1) * (cols - 1)
        p_value = ChiSquaredTest._chi2_pvalue(chi_sq, df)
        
        cramers_v = math.sqrt(chi_sq / (total * min(rows-1, cols-1)))
        
        return {
            "chi_squared": chi_sq,
            "p_value": p_value,
            "df": df,
            "cramers_v": cramers_v,
        }
    
    @staticmethod
    def _chi2_pvalue(chi_sq: float, df: int) -> float:
        """Approximate chi-squared p-value."""
        if df <= 0 or chi_sq <= 0:
            return 1.0
        
        z = ((chi_sq / df) ** (1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df))
        normal = NormalDistribution()
        return 1 - normal.cdf(z)


# ============================================================
# Confidence Intervals
# ============================================================

class ConfidenceInterval:
    """Confidence interval calculations."""
    
    @staticmethod
    def for_mean(data: List[float],
                 confidence: float = 0.95) -> Tuple[float, float]:
        n = len(data)
        x_bar = sum(data) / n
        s = math.sqrt(sum((x - x_bar)**2 for x in data) / (n - 1))
        
        z = NormalDistribution().quantile(0.5 + confidence / 2)
        margin = z * s / math.sqrt(n)
        
        return (x_bar - margin, x_bar + margin)
    
    @staticmethod
    def for_proportion(successes: int, n: int,
                       confidence: float = 0.95) -> Tuple[float, float]:
        p_hat = successes / n
        z = NormalDistribution().quantile(0.5 + confidence / 2)
        margin = z * math.sqrt(p_hat * (1 - p_hat) / n)
        
        return (max(0, p_hat - margin), min(1, p_hat + margin))
    
    @staticmethod
    def for_difference(data1: List[float], data2: List[float],
                       confidence: float = 0.95) -> Tuple[float, float]:
        n1, n2 = len(data1), len(data2)
        x1 = sum(data1) / n1
        x2 = sum(data2) / n2
        s1 = math.sqrt(sum((x - x1)**2 for x in data1) / (n1 - 1))
        s2 = math.sqrt(sum((x - x2)**2 for x in data2) / (n2 - 1))
        
        se = math.sqrt(s1**2/n1 + s2**2/n2)
        z = NormalDistribution().quantile(0.5 + confidence / 2)
        diff = x1 - x2
        margin = z * se
        
        return (diff - margin, diff + margin)


# ============================================================
# Bootstrap
# ============================================================

class Bootstrap:
    """Bootstrap resampling methods."""
    
    @staticmethod
    def resample(data: List[float]) -> List[float]:
        n = len(data)
        return [data[random.randint(0, n-1)] for _ in range(n)]
    
    @staticmethod
    def confidence_interval(
            data: List[float],
            statistic: Callable,
            n_bootstrap: int = 1000,
            confidence: float = 0.95) -> Tuple[float, float, List[float]]:
        
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = Bootstrap.resample(data)
            bootstrap_stats.append(statistic(sample))
        
        bootstrap_stats.sort()
        alpha = 1 - confidence
        lower_idx = int(alpha / 2 * n_bootstrap)
        upper_idx = int((1 - alpha / 2) * n_bootstrap)
        
        return (bootstrap_stats[lower_idx],
                bootstrap_stats[min(upper_idx, len(bootstrap_stats)-1)],
                bootstrap_stats)
    
    @staticmethod
    def hypothesis_test(
            data1: List[float], data2: List[float],
            statistic: Callable,
            n_bootstrap: int = 1000) -> Dict[str, float]:
        
        observed = statistic(data1) - statistic(data2)
        
        combined = data1 + data2
        n1 = len(data1)
        
        count = 0
        for _ in range(n_bootstrap):
            random.shuffle(combined)
            perm_stat = statistic(combined[:n1]) - statistic(combined[n1:])
            if abs(perm_stat) >= abs(observed):
                count += 1
        
        return {
            "observed_diff": observed,
            "p_value": count / n_bootstrap,
        }


# ============================================================
# Power Analysis
# ============================================================

class PowerAnalysis:
    """Statistical power calculations."""
    
    @staticmethod
    def sample_size_for_mean(effect_size: float,
                             sigma: float,
                             alpha: float = 0.05,
                             power: float = 0.8) -> int:
        z_alpha = NormalDistribution().quantile(1 - alpha / 2)
        z_beta = NormalDistribution().quantile(power)
        
        n = ((z_alpha + z_beta) * sigma / effect_size) ** 2
        return math.ceil(n)
    
    @staticmethod
    def sample_size_for_proportion(p1: float, p2: float,
                                   alpha: float = 0.05,
                                   power: float = 0.8) -> int:
        z_alpha = NormalDistribution().quantile(1 - alpha / 2)
        z_beta = NormalDistribution().quantile(power)
        
        p_avg = (p1 + p2) / 2
        n = ((z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) +
              z_beta * math.sqrt(p1*(1-p1) + p2*(1-p2))) /
             (p1 - p2)) ** 2
        
        return math.ceil(n)
    
    @staticmethod
    def compute_power(n: int, effect_size: float,
                      sigma: float,
                      alpha: float = 0.05) -> float:
        z_alpha = NormalDistribution().quantile(1 - alpha / 2)
        se = sigma / math.sqrt(n)
        
        z_power = effect_size / se - z_alpha
        
        return NormalDistribution().cdf(z_power)`,
				},
			},
		},
	})
}
