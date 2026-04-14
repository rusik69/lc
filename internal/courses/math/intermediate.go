package math

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMathModules([]problems.CourseModule{
		{
			ID:          2007,
			Title:       "Calculus: Limits and Derivatives",
			Description: "Master the foundations of calculus: limits, continuity, derivatives, and their applications.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Limits and Continuity",
					Content: `Limits are the foundation of calculus. A limit describes the value a function approaches as its input approaches a particular point.

**Formal Definition (ε-δ):**
For every ε > 0, there exists a δ > 0 such that if 0 < |x - a| < δ, then |f(x) - L| < ε. This means f(x) can be made arbitrarily close to L by making x sufficiently close to a.

**Key Limit Laws:**
- **Sum Rule**: lim[f(x) + g(x)] = lim f(x) + lim g(x)
- **Product Rule**: lim[f(x) · g(x)] = lim f(x) · lim g(x)
- **Quotient Rule**: lim[f(x)/g(x)] = lim f(x) / lim g(x), provided lim g(x) ≠ 0
- **Squeeze Theorem**: If g(x) ≤ f(x) ≤ h(x) and lim g(x) = lim h(x) = L, then lim f(x) = L

**Important Limits:**
- lim(x→0) sin(x)/x = 1
- lim(x→∞) (1 + 1/n)^n = e ≈ 2.71828
- lim(x→0) (eˣ - 1)/x = 1

**Continuity:**
A function f is continuous at point a if:
1. f(a) is defined
2. lim(x→a) f(x) exists
3. lim(x→a) f(x) = f(a)

**Types of Discontinuity:**
- **Removable**: Limit exists but f(a) is undefined or doesn't match (hole in graph)
- **Jump**: Left and right limits exist but differ
- **Infinite**: Function approaches ±∞ (vertical asymptote)

**Why Limits Matter in CS:**
- Asymptotic analysis (Big-O notation is fundamentally about limits)
- Numerical stability (floating-point operations as precision limits)
- Convergence of iterative algorithms (gradient descent, Newton's method)
- Machine learning loss function behavior near optima`,
					CodeExamples: `import math

def limit_estimate(f, a, h_values=None):
    """Estimate limit of f(x) as x approaches a."""
    if h_values is None:
        h_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    print(f"Estimating lim f(x) as x -> {a}:")
    for h in h_values:
        left = f(a - h)
        right = f(a + h)
        avg = (left + right) / 2
        print(f"  h={h:.5f}: f({a}-h)={left:.6f}, f({a}+h)={right:.6f}, avg={avg:.6f}")

# Classic limit: sin(x)/x as x -> 0
limit_estimate(lambda x: math.sin(x)/x if x != 0 else 1, 0)

# (1 + 1/n)^n as n -> infinity
print("\nApproaching e:")
for n in [10, 100, 1000, 10000, 100000]:
    approx = (1 + 1/n)**n
    print(f"  n={n:>6}: (1+1/n)^n = {approx:.8f}")
print(f"  Actual e = {math.e:.8f}")`,
				},
				{
					Title: "Derivatives: Rules and Computation",
					Content: `The derivative measures the instantaneous rate of change of a function. Geometrically, it gives the slope of the tangent line at any point.

**Definition:**
f'(x) = lim(h→0) [f(x+h) - f(x)] / h

**Fundamental Differentiation Rules:**

**Power Rule:** d/dx[xⁿ] = n·xⁿ⁻¹
**Constant Multiple:** d/dx[c·f(x)] = c·f'(x)
**Sum/Difference:** d/dx[f(x) ± g(x)] = f'(x) ± g'(x)

**Product Rule:** d/dx[f(x)·g(x)] = f'(x)·g(x) + f(x)·g'(x)
**Quotient Rule:** d/dx[f(x)/g(x)] = [f'(x)·g(x) - f(x)·g'(x)] / [g(x)]²

**Chain Rule:** d/dx[f(g(x))] = f'(g(x))·g'(x)

**Common Derivatives:**
- d/dx[sin(x)] = cos(x)
- d/dx[cos(x)] = -sin(x)
- d/dx[eˣ] = eˣ
- d/dx[ln(x)] = 1/x
- d/dx[tan(x)] = sec²(x)

**Applications in CS:**
- **Gradient Descent**: Uses derivatives to minimize loss functions in ML
- **Backpropagation**: Chain rule applied through neural network layers
- **Optimization**: Finding minima/maxima of objective functions
- **Physics Simulations**: Velocity is derivative of position, acceleration is derivative of velocity`,
					CodeExamples: `def numerical_derivative(f, x, h=1e-7):
    """Compute derivative using central difference."""
    return (f(x + h) - f(x - h)) / (2 * h)

import math

# Verify: d/dx[x^2] = 2x at x=3 should be 6
print(f"d/dx[x^2] at x=3: {numerical_derivative(lambda x: x**2, 3):.6f} (expected 6)")

# Verify: d/dx[sin(x)] = cos(x) at x=pi/4
x = math.pi / 4
print(f"d/dx[sin(x)] at x=pi/4: {numerical_derivative(math.sin, x):.6f}")
print(f"cos(pi/4) = {math.cos(x):.6f}")

# Gradient descent to minimize f(x) = (x-3)^2 + 1
def gradient_descent(f, df, x0, lr=0.1, steps=20):
    x = x0
    for i in range(steps):
        grad = df(x)
        x = x - lr * grad
        if i % 5 == 0:
            print(f"  Step {i}: x={x:.4f}, f(x)={f(x):.4f}, grad={grad:.4f}")
    return x

f = lambda x: (x - 3)**2 + 1
df = lambda x: 2 * (x - 3)
print("\nGradient descent on f(x) = (x-3)^2 + 1:")
result = gradient_descent(f, df, x0=0.0)
print(f"  Minimum at x={result:.4f}, f(x)={f(result):.4f}")`,
				},
				{
					Title: "Applications of Derivatives",
					Content: `Derivatives have powerful applications in optimization, curve analysis, and approximation.

**Critical Points and Optimization:**
A critical point occurs where f'(x) = 0 or f'(x) is undefined.

**First Derivative Test:**
- If f' changes from + to - at c: local maximum
- If f' changes from - to + at c: local minimum
- If f' doesn't change sign: neither (inflection point)

**Second Derivative Test:**
- If f''(c) > 0: local minimum (concave up)
- If f''(c) < 0: local maximum (concave down)
- If f''(c) = 0: test is inconclusive

**L'Hôpital's Rule:**
For indeterminate forms 0/0 or ∞/∞:
lim[f(x)/g(x)] = lim[f'(x)/g'(x)]

**Taylor Series Approximation:**
f(x) ≈ f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + f'''(a)(x-a)³/3! + ...

The Maclaurin series is the Taylor series centered at a = 0:
- eˣ = 1 + x + x²/2! + x³/3! + ...
- sin(x) = x - x³/3! + x⁵/5! - ...
- cos(x) = 1 - x²/2! + x⁴/4! - ...

**Why This Matters:**
- Taylor approximations are used in numerical computing to evaluate transcendental functions
- Newton's method uses derivative information for fast root-finding
- Optimization algorithms (Adam, RMSprop) use first and second derivative estimates`,
					CodeExamples: `import math

def find_critical_points(f, df, x_range, step=0.01):
    """Find approximate critical points where derivative is near zero."""
    points = []
    x = x_range[0]
    while x < x_range[1]:
        if abs(df(x)) < step:
            points.append((x, f(x)))
        x += step
    return points

# Find critical points of f(x) = x^3 - 3x + 1
f = lambda x: x**3 - 3*x + 1
df = lambda x: 3*x**2 - 3
points = find_critical_points(f, df, (-3, 3))
print("Critical points of x^3 - 3x + 1:")
for x, y in points:
    print(f"  x={x:.2f}, f(x)={y:.2f}")

# Taylor series approximation of e^x
def taylor_exp(x, terms=10):
    result = 0
    for n in range(terms):
        result += x**n / math.factorial(n)
    return result

print(f"\nTaylor approx of e^1:")
for n in [1, 3, 5, 10, 15]:
    approx = taylor_exp(1.0, n)
    error = abs(approx - math.e)
    print(f"  {n} terms: {approx:.10f} (error: {error:.2e})")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          2008,
			Title:       "Integral Calculus",
			Description: "Learn integration techniques, the fundamental theorem of calculus, and applications of integrals.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "Definite and Indefinite Integrals",
					Content: `Integration is the reverse process of differentiation. It computes accumulated quantities: areas, volumes, total change.

**Indefinite Integral (Antiderivative):**
∫f(x)dx = F(x) + C, where F'(x) = f(x)

**Definite Integral:**
∫ₐᵇ f(x)dx = F(b) - F(a)

**Fundamental Theorem of Calculus:**
Part 1: If F(x) = ∫ₐˣ f(t)dt, then F'(x) = f(x)
Part 2: ∫ₐᵇ f(x)dx = F(b) - F(a), where F is any antiderivative of f

**Basic Integration Rules:**
- ∫xⁿ dx = xⁿ⁺¹/(n+1) + C (n ≠ -1)
- ∫1/x dx = ln|x| + C
- ∫eˣ dx = eˣ + C
- ∫sin(x) dx = -cos(x) + C
- ∫cos(x) dx = sin(x) + C

**Integration Techniques:**

**Substitution (u-substitution):**
If ∫f(g(x))·g'(x)dx, let u = g(x), du = g'(x)dx → ∫f(u)du

**Integration by Parts:**
∫u·dv = u·v - ∫v·du (LIATE rule for choosing u: Log, Inverse trig, Algebraic, Trig, Exponential)

**Partial Fractions:**
Decompose rational functions into simpler fractions before integrating.

**Applications in CS:**
- Computing cumulative distribution functions in statistics
- Signal processing (Fourier transforms are integrals)
- Probability theory (expected value E[X] = ∫x·f(x)dx)
- Physics simulations (work = ∫F·dx)`,
					CodeExamples: `import math

def trapezoidal_rule(f, a, b, n=1000):
    """Numerical integration using trapezoidal rule."""
    h = (b - a) / n
    total = (f(a) + f(b)) / 2
    for i in range(1, n):
        total += f(a + i * h)
    return total * h

def simpsons_rule(f, a, b, n=1000):
    """Numerical integration using Simpson's rule (higher accuracy)."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n, 2):
        total += 4 * f(a + i * h)
    for i in range(2, n, 2):
        total += 2 * f(a + i * h)
    return total * h / 3

# Compute integral of x^2 from 0 to 3 (exact: 9)
result = trapezoidal_rule(lambda x: x**2, 0, 3)
print(f"Integral of x^2 from 0 to 3: {result:.6f} (exact: 9)")

# Compute integral of sin(x) from 0 to pi (exact: 2)
result = simpsons_rule(math.sin, 0, math.pi)
print(f"Integral of sin(x) from 0 to pi: {result:.6f} (exact: 2)")

# Monte Carlo integration: estimate pi
import random
random.seed(42)
n = 100000
inside = sum(1 for _ in range(n) if random.random()**2 + random.random()**2 <= 1)
pi_est = 4 * inside / n
print(f"\nMonte Carlo pi estimate ({n} samples): {pi_est:.4f} (actual: {math.pi:.4f})")`,
				},
				{
					Title: "Multivariable Calculus Essentials",
					Content: `Multivariable calculus extends derivatives and integrals to functions of multiple variables, essential for machine learning and optimization.

**Partial Derivatives:**
For f(x, y), the partial derivative ∂f/∂x treats y as constant and differentiates with respect to x.

**The Gradient:**
∇f = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)
The gradient points in the direction of steepest ascent. Its magnitude is the rate of steepest increase.

**The Hessian Matrix:**
The matrix of second partial derivatives. For f(x, y):
H = [[∂²f/∂x², ∂²f/∂x∂y], [∂²f/∂y∂x, ∂²f/∂y²]]
- If H is positive definite at a critical point: local minimum
- If H is negative definite: local maximum
- If H is indefinite: saddle point

**Multiple Integrals:**
- Double integral: ∫∫ f(x,y) dA computes volume under a surface
- Triple integral: ∫∫∫ f(x,y,z) dV computes hypervolume

**Vector Calculus Operations:**
- **Divergence**: ∇·F = ∂F₁/∂x + ∂F₂/∂y + ∂F₃/∂z (scalar result, measures "source strength")
- **Curl**: ∇×F (vector result, measures rotation)
- **Laplacian**: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²

**CS Applications:**
- Gradient descent in high-dimensional spaces (deep learning)
- Hessian for second-order optimization (Newton's method, L-BFGS)
- Jacobian matrices for coordinate transformations in computer graphics
- Divergence and curl in fluid simulations`,
					CodeExamples: `import math

def gradient_2d(f, x, y, h=1e-7):
    """Compute gradient of f(x, y) numerically."""
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return (df_dx, df_dy)

def gradient_descent_2d(f, x0, y0, lr=0.1, steps=50):
    """2D gradient descent."""
    x, y = x0, y0
    for i in range(steps):
        gx, gy = gradient_2d(f, x, y)
        x -= lr * gx
        y -= lr * gy
        if i % 10 == 0:
            print(f"  Step {i}: ({x:.4f}, {y:.4f}), f={f(x,y):.4f}")
    return x, y

# Minimize f(x,y) = (x-2)^2 + (y-3)^2
f = lambda x, y: (x - 2)**2 + (y - 3)**2
print("Gradient descent on (x-2)^2 + (y-3)^2:")
xm, ym = gradient_descent_2d(f, 0.0, 0.0)
print(f"  Minimum at ({xm:.4f}, {ym:.4f})")

# Compute gradient at a point
g = gradient_2d(f, 1.0, 1.0)
print(f"\nGradient at (1,1): ({g[0]:.4f}, {g[1]:.4f})")
print(f"  Points toward minimum at (2, 3)")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          2009,
			Title:       "Linear Algebra",
			Description: "Master vectors, matrices, eigenvalues, and linear transformations fundamental to data science and graphics.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Vectors and Vector Spaces",
					Content: `Vectors are ordered lists of numbers that represent magnitude and direction. They are the building blocks of linear algebra.

**Vector Operations:**
- **Addition**: (a₁, a₂) + (b₁, b₂) = (a₁+b₁, a₂+b₂)
- **Scalar Multiplication**: c·(a₁, a₂) = (c·a₁, c·a₂)
- **Dot Product**: a·b = a₁b₁ + a₂b₂ + ... + aₙbₙ (scalar result)
- **Cross Product**: a×b (3D only, vector result, perpendicular to both)
- **Norm (magnitude)**: ||a|| = √(a₁² + a₂² + ... + aₙ²)

**Vector Space Properties (must satisfy all):**
1. Closure under addition and scalar multiplication
2. Associativity and commutativity of addition
3. Existence of zero vector and additive inverses
4. Distributivity of scalar multiplication

**Linear Independence:**
Vectors v₁, v₂, ..., vₙ are linearly independent if the only solution to c₁v₁ + c₂v₂ + ... + cₙvₙ = 0 is c₁ = c₂ = ... = cₙ = 0.

**Basis and Dimension:**
A basis is a minimal set of linearly independent vectors that spans the entire space. The dimension is the number of vectors in a basis.

**CS Applications:**
- Word embeddings in NLP (word2vec, GloVe) represent words as vectors
- Feature vectors in machine learning
- 3D graphics: positions, normals, directions
- Recommendation systems: user/item vectors in collaborative filtering`,
					CodeExamples: `import math

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def vector_norm(v):
    return math.sqrt(sum(x**2 for x in v))

def cosine_similarity(a, b):
    return dot_product(a, b) / (vector_norm(a) * vector_norm(b))

def vector_add(a, b):
    return tuple(x + y for x, y in zip(a, b))

def scalar_mult(c, v):
    return tuple(c * x for x in v)

# Basic operations
a = (3, 4)
b = (1, 2)
print(f"a = {a}, b = {b}")
print(f"a + b = {vector_add(a, b)}")
print(f"2 * a = {scalar_mult(2, a)}")
print(f"a · b = {dot_product(a, b)}")
print(f"||a|| = {vector_norm(a):.4f}")
print(f"cosine similarity = {cosine_similarity(a, b):.4f}")

# Word embedding similarity (simplified)
cat = (0.7, 0.2, 0.9, 0.1)
dog = (0.8, 0.3, 0.85, 0.15)
car = (0.1, 0.9, 0.2, 0.8)
print(f"\nWord similarity:")
print(f"  cat-dog: {cosine_similarity(cat, dog):.4f}")
print(f"  cat-car: {cosine_similarity(cat, car):.4f}")`,
				},
				{
					Title: "Matrices and Linear Transformations",
					Content: `A matrix is a rectangular array of numbers. Matrices represent linear transformations and systems of equations.

**Matrix Operations:**
- **Addition**: Element-wise, same dimensions required
- **Scalar Multiplication**: Multiply every element
- **Matrix Multiplication**: (AB)ᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ (rows of A × columns of B)

**Special Matrices:**
- **Identity Matrix (I)**: AI = IA = A
- **Transpose (Aᵀ)**: Rows become columns
- **Inverse (A⁻¹)**: AA⁻¹ = A⁻¹A = I (exists only for square, non-singular matrices)
- **Symmetric**: A = Aᵀ
- **Orthogonal**: AᵀA = AAᵀ = I

**Determinant:**
- det(A) = 0 means A is singular (no inverse)
- |det(A)| represents the scaling factor of the transformation
- For 2×2: det([[a,b],[c,d]]) = ad - bc

**Rank:**
The number of linearly independent rows (or columns). A matrix is full rank when rank equals min(rows, cols).

**Systems of Equations (Ax = b):**
- Gaussian elimination: O(n³) row reduction
- If det(A) ≠ 0: unique solution x = A⁻¹b
- If rank(A) < n: infinitely many or no solutions

**CS Applications:**
- Computer graphics: rotation, scaling, projection matrices
- PageRank: transition matrix and eigenvectors
- Image processing: convolution matrices
- Neural networks: weight matrices multiply input vectors`,
					CodeExamples: `def mat_mult(A, B):
    """Multiply two matrices."""
    rows_a, cols_a = len(A), len(A[0])
    rows_b, cols_b = len(B), len(B[0])
    assert cols_a == rows_b
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += A[i][k] * B[k][j]
    return result

def mat_transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def det_2x2(A):
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]

# 2D rotation matrix
import math
def rotation_matrix(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

# Rotate point (1, 0) by 90 degrees
R = rotation_matrix(math.pi / 2)
point = [[1], [0]]
rotated = mat_mult(R, point)
print(f"(1,0) rotated 90°: ({rotated[0][0]:.4f}, {rotated[1][0]:.4f})")

# Matrix multiplication
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = mat_mult(A, B)
print(f"\nA*B = {C}")
print(f"det(A) = {det_2x2(A)}")`,
				},
				{
					Title: "Eigenvalues and Eigenvectors",
					Content: `Eigenvalues and eigenvectors reveal the fundamental behavior of linear transformations.

**Definition:**
For a square matrix A, if Av = λv (where v ≠ 0), then:
- λ is an eigenvalue
- v is the corresponding eigenvector

The eigenvector's direction is unchanged by the transformation; it is only scaled by λ.

**Finding Eigenvalues:**
Solve det(A - λI) = 0 (the characteristic equation).

**Key Properties:**
- trace(A) = sum of eigenvalues
- det(A) = product of eigenvalues
- Symmetric matrices have real eigenvalues and orthogonal eigenvectors
- A matrix is positive definite iff all eigenvalues > 0

**Eigendecomposition:**
A = PDP⁻¹, where P contains eigenvectors and D is diagonal with eigenvalues.

**Singular Value Decomposition (SVD):**
A = UΣVᵀ, works for any matrix (not just square).
- U: left singular vectors (m×m orthogonal)
- Σ: diagonal matrix of singular values
- V: right singular vectors (n×n orthogonal)

**Principal Component Analysis (PCA):**
1. Center the data (subtract mean)
2. Compute covariance matrix
3. Find eigenvectors (principal components)
4. Project data onto top k components

**CS Applications:**
- PCA for dimensionality reduction
- SVD for recommendation systems (Netflix Prize)
- Google PageRank is an eigenvector problem
- Image compression via low-rank approximation
- Stability analysis of dynamical systems`,
					CodeExamples: `import math

def power_iteration(A, num_iterations=100):
    """Find dominant eigenvalue/eigenvector using power iteration."""
    n = len(A)
    b = [1.0 / math.sqrt(n)] * n

    for _ in range(num_iterations):
        # Multiply A * b
        Ab = [sum(A[i][j] * b[j] for j in range(n)) for i in range(n)]
        # Normalize
        norm = math.sqrt(sum(x**2 for x in Ab))
        b = [x / norm for x in Ab]

    # Compute eigenvalue (Rayleigh quotient)
    Ab = [sum(A[i][j] * b[j] for j in range(n)) for i in range(n)]
    eigenvalue = sum(Ab[i] * b[i] for i in range(n))
    return eigenvalue, b

# Find dominant eigenvalue of a 2x2 matrix
A = [[4, 1], [2, 3]]
eigenvalue, eigenvector = power_iteration(A)
print(f"Matrix: {A}")
print(f"Dominant eigenvalue: {eigenvalue:.4f}")
print(f"Eigenvector: [{eigenvector[0]:.4f}, {eigenvector[1]:.4f}]")

# Verify: A*v should equal lambda*v
Av = [sum(A[i][j] * eigenvector[j] for j in range(2)) for i in range(2)]
lv = [eigenvalue * eigenvector[i] for i in range(2)]
print(f"A*v = [{Av[0]:.4f}, {Av[1]:.4f}]")
print(f"λ*v = [{lv[0]:.4f}, {lv[1]:.4f}]")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          2010,
			Title:       "Probability and Statistics",
			Description: "Master probability distributions, statistical inference, hypothesis testing, and Bayesian reasoning.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "Probability Foundations",
					Content: `Probability quantifies uncertainty. It assigns a number between 0 and 1 to events, where 0 means impossible and 1 means certain.

**Axioms of Probability (Kolmogorov):**
1. P(A) ≥ 0 for any event A
2. P(sample space) = 1
3. For mutually exclusive events: P(A ∪ B) = P(A) + P(B)

**Key Rules:**
- **Complement**: P(Aᶜ) = 1 - P(A)
- **Union**: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
- **Conditional**: P(A|B) = P(A ∩ B) / P(B)
- **Independence**: P(A ∩ B) = P(A) · P(B) iff A and B are independent

**Bayes' Theorem:**
P(A|B) = P(B|A) · P(A) / P(B)

This is the foundation of Bayesian inference: update beliefs (prior) with evidence (likelihood) to get updated beliefs (posterior).

**Random Variables:**
- **Discrete**: Takes countable values (coin flips, dice rolls)
- **Continuous**: Takes values in an interval (height, temperature)

**Expected Value and Variance:**
- E[X] = Σ xᵢ · P(xᵢ) (discrete) or ∫ x · f(x)dx (continuous)
- Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
- Standard deviation: σ = √Var(X)

**Important Distributions:**
- **Bernoulli**: Single trial, P(success) = p
- **Binomial**: n independent Bernoulli trials
- **Poisson**: Rare events over interval (λ = expected count)
- **Normal (Gaussian)**: Bell curve, characterized by μ and σ
- **Exponential**: Time between Poisson events`,
					CodeExamples: `import random
import math

random.seed(42)

def simulate_coin_flips(n, p=0.5):
    """Simulate n coin flips with probability p of heads."""
    flips = [1 if random.random() < p else 0 for _ in range(n)]
    return sum(flips) / n

# Law of large numbers: frequency converges to probability
print("Coin flip convergence:")
for n in [10, 100, 1000, 10000, 100000]:
    freq = simulate_coin_flips(n)
    print(f"  n={n:>6}: P(heads) = {freq:.4f}")

# Bayes' theorem: disease testing
# P(disease) = 0.01, P(pos|disease) = 0.95, P(pos|healthy) = 0.05
p_disease = 0.01
p_pos_disease = 0.95
p_pos_healthy = 0.05
p_pos = p_pos_disease * p_disease + p_pos_healthy * (1 - p_disease)
p_disease_pos = p_pos_disease * p_disease / p_pos
print(f"\nBayes' theorem (disease test):")
print(f"  P(disease|positive test) = {p_disease_pos:.4f}")
print(f"  Even with 95% accurate test, only {p_disease_pos*100:.1f}% chance of disease!")

# Central limit theorem demonstration
print(f"\nCentral limit theorem (means of uniform[0,1] samples):")
for sample_size in [1, 5, 30, 100]:
    means = [sum(random.random() for _ in range(sample_size)) / sample_size for _ in range(10000)]
    avg = sum(means) / len(means)
    std = math.sqrt(sum((m - avg)**2 for m in means) / len(means))
    print(f"  n={sample_size:>3}: mean={avg:.4f}, std={std:.4f}")`,
				},
				{
					Title: "Statistical Inference and Hypothesis Testing",
					Content: `Statistical inference draws conclusions about populations from sample data.

**Descriptive Statistics:**
- **Mean**: Average value (sensitive to outliers)
- **Median**: Middle value (robust to outliers)
- **Mode**: Most frequent value
- **Standard Deviation**: Spread around the mean
- **Percentiles**: Values below which a percentage of data falls

**Confidence Intervals:**
A 95% confidence interval means: if we repeated sampling many times, 95% of intervals would contain the true parameter.
For large samples: x̄ ± z · (σ/√n), where z = 1.96 for 95%

**Hypothesis Testing:**
1. State null (H₀) and alternative (H₁) hypotheses
2. Choose significance level (α, typically 0.05)
3. Compute test statistic and p-value
4. If p-value < α, reject H₀

**Common Tests:**
- **z-test**: Compare sample mean to known population mean (large n)
- **t-test**: Compare means with unknown population variance (small n)
- **Chi-square test**: Test independence of categorical variables
- **ANOVA**: Compare means across multiple groups
- **F-test**: Compare variances

**p-value Interpretation:**
The p-value is the probability of observing data at least as extreme as what was observed, assuming H₀ is true. It is NOT the probability that H₀ is true.

**Common Pitfalls:**
- Multiple testing problem (Bonferroni correction)
- Confusing statistical significance with practical significance
- Assuming correlation implies causation
- Survivorship bias in data collection`,
					CodeExamples: `import math
import random

random.seed(42)

def mean(data):
    return sum(data) / len(data)

def std_dev(data):
    m = mean(data)
    return math.sqrt(sum((x - m)**2 for x in data) / (len(data) - 1))

def z_test(sample, mu0):
    """One-sample z-test against hypothesized mean."""
    n = len(sample)
    x_bar = mean(sample)
    s = std_dev(sample)
    z = (x_bar - mu0) / (s / math.sqrt(n))
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p_value

# Test: Are these exam scores significantly different from 75?
scores = [random.gauss(78, 10) for _ in range(50)]
z, p = z_test(scores, 75)
print(f"H0: mean = 75")
print(f"Sample mean: {mean(scores):.2f}")
print(f"z-statistic: {z:.4f}")
print(f"p-value: {p:.4f}")
print(f"Reject H0 at α=0.05: {p < 0.05}")

# Confidence interval
m = mean(scores)
s = std_dev(scores)
n = len(scores)
ci_low = m - 1.96 * s / math.sqrt(n)
ci_high = m + 1.96 * s / math.sqrt(n)
print(f"\n95% CI: ({ci_low:.2f}, {ci_high:.2f})")`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
