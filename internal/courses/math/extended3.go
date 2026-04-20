package math

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMathModules([]problems.CourseModule{
		{
			ID:          2717,
			Title:       "Linear Algebra for Computing",
			Description: "Deep dive into linear algebra concepts essential for machine learning, computer graphics, and scientific computing: matrix operations, eigenvalues, SVD, and applications.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Vectors Matrices and Transformations",
					Content: `Linear algebra is the mathematics of vectors, matrices, and linear transformations. It underpins machine learning, computer graphics, signal processing, and scientific computing.

**Vectors:**

A vector is an ordered list of numbers:
  v = [v₁, v₂, ..., vₙ] ∈ ℝⁿ

Operations:
  Addition: u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]
  Scalar multiplication: cv = [cv₁, cv₂, ..., cvₙ]
  Dot product: u·v = Σ uᵢvᵢ = u₁v₁ + u₂v₂ + ... + uₙvₙ
  
  Norm (length): ||v|| = √(v·v) = √(Σ vᵢ²)
  L1 norm: ||v||₁ = Σ |vᵢ|
  L∞ norm: ||v||∞ = max |vᵢ|
  Lp norm: ||v||ₚ = (Σ |vᵢ|ᵖ)^(1/p)

  Unit vector: v̂ = v / ||v||
  
  Angle between vectors:
    cos(θ) = (u·v) / (||u|| × ||v||)
    
  Orthogonal: u·v = 0

Cross product (3D):
  u × v = [u₂v₃-u₃v₂, u₃v₁-u₁v₃, u₁v₂-u₂v₁]
  Result is perpendicular to both u and v
  ||u × v|| = ||u|| × ||v|| × sin(θ)

Projection:
  proj_u(v) = (v·u / u·u) × u
  Component of v along u

**Matrices:**

Matrix A ∈ ℝᵐˣⁿ: m rows, n columns
  Aᵢⱼ: Element at row i, column j

Operations:
  Addition: (A+B)ᵢⱼ = Aᵢⱼ + Bᵢⱼ (same dimensions)
  Scalar multiplication: (cA)ᵢⱼ = c × Aᵢⱼ
  
  Matrix multiplication: C = AB where A ∈ ℝᵐˣⁿ, B ∈ ℝⁿˣᵖ
    Cᵢⱼ = Σₖ AᵢₖBₖⱼ
    Result: C ∈ ℝᵐˣᵖ
    Not commutative: AB ≠ BA in general
    Associative: (AB)C = A(BC)
    Distributive: A(B+C) = AB + AC

  Transpose: (Aᵀ)ᵢⱼ = Aⱼᵢ
    (AB)ᵀ = BᵀAᵀ
    Symmetric: A = Aᵀ

Special matrices:
  Identity I: Iᵢᵢ = 1, Iᵢⱼ = 0 for i≠j; AI = IA = A
  Zero matrix: All elements 0
  Diagonal: Non-zero only on diagonal
  Triangular: Upper (Aᵢⱼ = 0 for i > j) or Lower
  Orthogonal: AᵀA = AAᵀ = I ⟹ A⁻¹ = Aᵀ

**Linear Transformations:**

Every matrix A defines a linear transformation T(x) = Ax
  T(αu + βv) = αT(u) + βT(v)

2D transformations:
  Rotation by θ:
    R = [[cos θ, -sin θ], [sin θ, cos θ]]
  
  Scaling by (sₓ, sᵧ):
    S = [[sₓ, 0], [0, sᵧ]]
  
  Reflection across x-axis:
    M = [[1, 0], [0, -1]]
  
  Shear:
    H = [[1, k], [0, 1]]

**Systems of Linear Equations:**

Ax = b where A ∈ ℝᵐˣⁿ, x ∈ ℝⁿ, b ∈ ℝᵐ

Gaussian elimination:
  Convert to row echelon form using row operations:
    1. Swap rows
    2. Multiply row by nonzero scalar
    3. Add multiple of one row to another
  
  Back substitution to solve

Gauss-Jordan: Reduce to reduced row echelon form (RREF)
  Leading 1 in each row, zeros above and below

Solutions:
  Unique: When rank(A) = n (number of unknowns)
  Infinite: When rank(A) < n
  None: When rank(A) < rank([A|b]) (inconsistent)

**Matrix Inverse:**

A⁻¹ exists iff det(A) ≠ 0 (A is invertible/nonsingular)
  A × A⁻¹ = A⁻¹ × A = I

Computing:
  2×2: A⁻¹ = (1/det(A)) × [[d, -b], [-c, a]] for A = [[a,b],[c,d]]
  General: Row reduce [A|I] → [I|A⁻¹]

Properties:
  (AB)⁻¹ = B⁻¹A⁻¹
  (Aᵀ)⁻¹ = (A⁻¹)ᵀ

**Determinant:**

2×2: det([[a,b],[c,d]]) = ad - bc
3×3: Cofactor expansion along any row/column

Properties:
  det(AB) = det(A) × det(B)
  det(Aᵀ) = det(A)
  det(cA) = cⁿdet(A) for n×n matrix
  det(A⁻¹) = 1/det(A)
  Row swap changes sign of det
  Row of zeros ⟹ det = 0
  
Geometric interpretation:
  |det(A)| = volume scaling factor of transformation
  det < 0: Orientation reversed

**Vector Spaces:**

A vector space V over ℝ satisfies:
  Closed under addition and scalar multiplication
  Has zero vector
  Associative, commutative addition
  Distributive scalar multiplication

Subspace: Subset that is also a vector space
  Must contain zero vector
  Closed under addition and scalar multiplication

Linear independence:
  v₁,...,vₖ are independent if c₁v₁+...+cₖvₖ = 0 implies all cᵢ = 0

Basis:
  Maximal linearly independent set
  Minimal spanning set
  Every vector has unique representation in basis

Dimension: Number of vectors in any basis
  ℝⁿ has dimension n

Column space: span of columns of A = {Ax : x ∈ ℝⁿ}
Row space: span of rows of A
Null space: {x : Ax = 0}
Rank: dim(column space) = dim(row space)
Rank-nullity theorem: rank(A) + nullity(A) = n`,
					CodeExamples: `# Linear Algebra Implementations

import math
import random
from typing import List, Optional, Tuple

# ============================================================
# Vector Operations
# ============================================================

class Vector:
    """N-dimensional vector."""
    
    def __init__(self, data: List[float]):
        self.data = data[:]
        self.n = len(data)
    
    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector([a + b for a, b in zip(self.data, other.data)])
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector([a - b for a, b in zip(self.data, other.data)])
    
    def __mul__(self, scalar: float) -> 'Vector':
        return Vector([x * scalar for x in self.data])
    
    def __rmul__(self, scalar: float) -> 'Vector':
        return self.__mul__(scalar)
    
    def dot(self, other: 'Vector') -> float:
        return sum(a * b for a, b in zip(self.data, other.data))
    
    def norm(self, p: float = 2) -> float:
        if p == float('inf'):
            return max(abs(x) for x in self.data)
        return sum(abs(x) ** p for x in self.data) ** (1 / p)
    
    def normalize(self) -> 'Vector':
        n = self.norm()
        if n == 0:
            return Vector([0.0] * self.n)
        return Vector([x / n for x in self.data])
    
    def angle(self, other: 'Vector') -> float:
        cos_theta = self.dot(other) / (self.norm() * other.norm())
        cos_theta = max(-1, min(1, cos_theta))
        return math.acos(cos_theta)
    
    def project_onto(self, other: 'Vector') -> 'Vector':
        scalar = self.dot(other) / other.dot(other)
        return other * scalar
    
    @staticmethod
    def cross(u: 'Vector', v: 'Vector') -> 'Vector':
        """3D cross product."""
        a, b = u.data, v.data
        return Vector([
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0],
        ])
    
    def __repr__(self) -> str:
        return f"Vector({self.data})"


# ============================================================
# Matrix Operations
# ============================================================

class Matrix:
    """Matrix with basic operations."""
    
    def __init__(self, data: List[List[float]]):
        self.data = [row[:] for row in data]
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0
    
    @staticmethod
    def identity(n: int) -> 'Matrix':
        return Matrix([[1.0 if i == j else 0.0 for j in range(n)]
                       for i in range(n)])
    
    @staticmethod
    def zeros(rows: int, cols: int) -> 'Matrix':
        return Matrix([[0.0] * cols for _ in range(rows)])
    
    @staticmethod
    def diagonal(values: List[float]) -> 'Matrix':
        n = len(values)
        return Matrix([[values[i] if i == j else 0.0
                       for j in range(n)] for i in range(n)])
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        return Matrix([[self.data[i][j] + other.data[i][j]
                       for j in range(self.cols)]
                      for i in range(self.rows)])
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        return Matrix([[self.data[i][j] - other.data[i][j]
                       for j in range(self.cols)]
                      for i in range(self.rows)])
    
    def scalar_mul(self, c: float) -> 'Matrix':
        return Matrix([[self.data[i][j] * c for j in range(self.cols)]
                      for i in range(self.rows)])
    
    def matmul(self, other: 'Matrix') -> 'Matrix':
        result = [[0.0] * other.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return Matrix(result)
    
    def matvec(self, v: Vector) -> Vector:
        return Vector([sum(self.data[i][j] * v.data[j]
                          for j in range(self.cols))
                      for i in range(self.rows)])
    
    def transpose(self) -> 'Matrix':
        return Matrix([[self.data[j][i] for j in range(self.rows)]
                      for i in range(self.cols)])
    
    def trace(self) -> float:
        return sum(self.data[i][i] for i in range(min(self.rows, self.cols)))
    
    def determinant(self) -> float:
        """Compute determinant using LU decomposition."""
        if self.rows != self.cols:
            raise ValueError("Not square")
        n = self.rows
        
        if n == 1:
            return self.data[0][0]
        if n == 2:
            return (self.data[0][0] * self.data[1][1] -
                    self.data[0][1] * self.data[1][0])
        
        # Cofactor expansion along first row
        det = 0.0
        for j in range(n):
            minor = Matrix([
                [self.data[i][k] for k in range(n) if k != j]
                for i in range(1, n)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det
    
    def inverse(self) -> Optional['Matrix']:
        """Compute inverse using Gauss-Jordan elimination."""
        n = self.rows
        if n != self.cols:
            return None
        
        # Augment with identity
        aug = [self.data[i][:] + [1.0 if j == i else 0.0
                                   for j in range(n)]
               for i in range(n)]
        
        # Forward elimination
        for col in range(n):
            # Find pivot
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]
            
            if abs(aug[col][col]) < 1e-12:
                return None  # Singular
            
            # Scale pivot row
            pivot = aug[col][col]
            for j in range(2 * n):
                aug[col][j] /= pivot
            
            # Eliminate column
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] -= factor * aug[col][j]
        
        return Matrix([aug[i][n:] for i in range(n)])
    
    def rank(self) -> int:
        """Compute rank via row echelon form."""
        rref = [row[:] for row in self.data]
        m, n = self.rows, self.cols
        r = 0
        
        for col in range(n):
            # Find pivot
            pivot = None
            for row in range(r, m):
                if abs(rref[row][col]) > 1e-12:
                    pivot = row
                    break
            
            if pivot is None:
                continue
            
            rref[r], rref[pivot] = rref[pivot], rref[r]
            
            factor = rref[r][col]
            for j in range(n):
                rref[r][j] /= factor
            
            for row in range(m):
                if row != r and abs(rref[row][col]) > 1e-12:
                    f = rref[row][col]
                    for j in range(n):
                        rref[row][j] -= f * rref[r][j]
            
            r += 1
        
        return r
    
    def solve(self, b: Vector) -> Optional[Vector]:
        """Solve Ax = b using Gaussian elimination."""
        n = self.rows
        aug = [self.data[i][:] + [b.data[i]] for i in range(n)]
        
        for col in range(n):
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]
            
            if abs(aug[col][col]) < 1e-12:
                return None
            
            for row in range(col + 1, n):
                factor = aug[row][col] / aug[col][col]
                for j in range(n + 1):
                    aug[row][j] -= factor * aug[col][j]
        
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            x[i] /= aug[i][i]
        
        return Vector(x)
    
    def __repr__(self) -> str:
        rows = [f"  {row}" for row in self.data]
        return "Matrix([\n" + "\n".join(rows) + "\n])"


# ============================================================
# Eigenvalue Computation
# ============================================================

def power_iteration(A: Matrix, max_iters: int = 1000,
                    tol: float = 1e-10) -> Tuple[float, Vector]:
    """Find dominant eigenvalue and eigenvector."""
    n = A.rows
    v = Vector([random.gauss(0, 1) for _ in range(n)])
    v = v.normalize()
    
    eigenvalue = 0.0
    
    for _ in range(max_iters):
        Av = A.matvec(v)
        new_eigenvalue = v.dot(Av)
        v_new = Av.normalize()
        
        if abs(new_eigenvalue - eigenvalue) < tol:
            return new_eigenvalue, v_new
        
        eigenvalue = new_eigenvalue
        v = v_new
    
    return eigenvalue, v


def qr_algorithm(A: Matrix, max_iters: int = 100,
                 tol: float = 1e-10) -> List[float]:
    """Find all eigenvalues using QR algorithm."""
    n = A.rows
    M = Matrix([row[:] for row in A.data])
    
    for _ in range(max_iters):
        Q, R = qr_decomposition(M)
        M = R.matmul(Q)
        
        # Check convergence (off-diagonal elements)
        off_diag = sum(abs(M.data[i][j])
                      for i in range(n) for j in range(n) if i != j)
        if off_diag < tol:
            break
    
    return [M.data[i][i] for i in range(n)]


def qr_decomposition(A: Matrix) -> Tuple[Matrix, Matrix]:
    """QR decomposition using Gram-Schmidt."""
    m, n = A.rows, A.cols
    
    Q_cols: List[Vector] = []
    R = Matrix.zeros(n, n)
    
    for j in range(n):
        v = Vector([A.data[i][j] for i in range(m)])
        
        for k in range(len(Q_cols)):
            R.data[k][j] = Q_cols[k].dot(v)
            v = v - Q_cols[k] * R.data[k][j]
        
        R.data[j][j] = v.norm()
        
        if R.data[j][j] > 1e-12:
            Q_cols.append(v.normalize())
        else:
            Q_cols.append(Vector([0.0] * m))
    
    Q = Matrix([[Q_cols[j].data[i] for j in range(n)]
               for i in range(m)])
    
    return Q, R


# ============================================================
# SVD (Simplified)
# ============================================================

def svd_2x2(A: Matrix) -> Tuple[Matrix, List[float], Matrix]:
    """SVD for 2x2 matrix."""
    ATA = A.transpose().matmul(A)
    
    eigenvalues = qr_algorithm(ATA)
    eigenvalues.sort(reverse=True)
    
    sigma = [math.sqrt(max(0, ev)) for ev in eigenvalues]
    
    return Matrix.identity(2), sigma, Matrix.identity(2)


# ============================================================
# Gram-Schmidt Orthogonalization
# ============================================================

def gram_schmidt(vectors: List[Vector]) -> List[Vector]:
    """Orthonormalize a set of vectors."""
    orthonormal = []
    
    for v in vectors:
        u = Vector(v.data[:])
        
        for q in orthonormal:
            proj = q * u.dot(q)
            u = u - proj
        
        norm = u.norm()
        if norm > 1e-12:
            orthonormal.append(u.normalize())
    
    return orthonormal


# ============================================================
# Least Squares
# ============================================================

def least_squares(A: Matrix, b: Vector) -> Vector:
    """Solve least squares: min ||Ax - b||² via normal equations."""
    AT = A.transpose()
    ATA = AT.matmul(A)
    ATb = AT.matvec(b)
    return ATA.solve(ATb)


def linear_regression(X: List[List[float]],
                      y: List[float]) -> Tuple[List[float], float]:
    """Fit linear regression y = Xw + b."""
    n = len(X)
    d = len(X[0])
    
    # Add bias column
    X_aug = [row[:] + [1.0] for row in X]
    
    A = Matrix(X_aug)
    b = Vector(y)
    
    w = least_squares(A, b)
    
    weights = w.data[:d]
    bias = w.data[d]
    
    return weights, bias`,
				},
				{
					Title: "Advanced Linear Algebra: Eigenvalues SVD and Applications",
					Content: `Eigenvalues, singular value decomposition, and related concepts power dimensionality reduction, recommender systems, image compression, and spectral methods in graph analysis.

**Eigenvalues and Eigenvectors:**

For square matrix A, if Av = λv (v ≠ 0):
  λ is an eigenvalue
  v is the corresponding eigenvector
  
Characteristic equation:
  det(A - λI) = 0
  Degree n polynomial in λ
  n×n matrix has n eigenvalues (counting multiplicity, possibly complex)

Properties:
  Σ λᵢ = trace(A)
  Π λᵢ = det(A)
  Eigenvalues of A⁻¹ are 1/λᵢ
  Eigenvalues of Aᵏ are λᵢᵏ

Diagonalization:
  A = PDP⁻¹ where D = diag(λ₁,...,λₙ), P = [v₁|...|vₙ]
  Only possible when A has n linearly independent eigenvectors
  Symmetric matrices are always diagonalizable: A = QDQᵀ

Spectral theorem (symmetric matrices):
  All eigenvalues are real
  Eigenvectors of distinct eigenvalues are orthogonal
  A = Σ λᵢvᵢvᵢᵀ (spectral decomposition)

**Positive Definite Matrices:**

A is positive definite if xᵀAx > 0 for all x ≠ 0
  Equivalent conditions:
    All eigenvalues > 0
    All pivots > 0
    All leading principal minors > 0
    A = BᵀB for some B with independent columns
  
  Positive semi-definite: xᵀAx ≥ 0 (eigenvalues ≥ 0)
  
  Importance:
    Covariance matrices are positive semi-definite
    Positive definite ⟹ unique minimum of quadratic form
    Hessians: Positive definite → local minimum

**Singular Value Decomposition (SVD):**

Any matrix A ∈ ℝᵐˣⁿ can be decomposed:
  A = UΣVᵀ
  where:
    U ∈ ℝᵐˣᵐ: Left singular vectors (orthogonal)
    Σ ∈ ℝᵐˣⁿ: Diagonal matrix of singular values σ₁ ≥ σ₂ ≥ ... ≥ 0
    V ∈ ℝⁿˣⁿ: Right singular vectors (orthogonal)

Relationship to eigenvalues:
  σᵢ = √λᵢ(AᵀA)
  Columns of V = eigenvectors of AᵀA
  Columns of U = eigenvectors of AAᵀ

Compact SVD:
  A = U_rΣ_rV_rᵀ where r = rank(A)
  Only keep non-zero singular values

Low-rank approximation (Eckart-Young theorem):
  Best rank-k approximation: A_k = Σᵢ₌₁ᵏ σᵢuᵢvᵢᵀ
  Minimizes ||A - A_k||_F (Frobenius norm)

Applications:
  Image compression: Store only top k singular values/vectors
  Latent Semantic Analysis: Topic modeling for documents
  Recommender systems: Matrix factorization
  Noise reduction: Remove small singular values
  Pseudoinverse: A⁺ = VΣ⁺Uᵀ (for least squares)
  PCA: Eigendecomposition of covariance matrix (or SVD of data)

**Principal Component Analysis (PCA):**

Goal: Find directions of maximum variance

Algorithm:
  1. Center data: X̃ = X - mean(X)
  2. Compute covariance: C = X̃ᵀX̃/(n-1)
  3. Eigendecomposition: C = VDVᵀ
  4. Project: Z = X̃V_k (top k eigenvectors)

Via SVD:
  X̃ = UΣVᵀ
  Principal components: Z = UΣ = X̃V
  Explained variance ratio: σᵢ² / Σσⱼ²

Choosing k:
  Scree plot: Elbow in explained variance
  Threshold: Keep enough for 95% variance
  Cross-validation

**Matrix Norms:**

Frobenius norm: ||A||_F = √(Σᵢⱼ Aᵢⱼ²) = √(trace(AᵀA)) = √(Σσᵢ²)
Spectral norm: ||A||₂ = σ₁ (largest singular value)
Nuclear norm: ||A||* = Σσᵢ (sum of singular values)

**Applications in ML:**

Covariance matrix:
  C = (1/n)XᵀX (centered data)
  Symmetric positive semi-definite
  Eigenvalues = variances along principal components

Kernel methods:
  Kernel matrix K: Kᵢⱼ = k(xᵢ, xⱼ)
  Must be positive semi-definite
  Eigendecomposition for spectral clustering

Matrix factorization:
  A ≈ WH (non-negative matrix factorization)
  Recommender systems: Rating matrix ≈ User × Item
  Topic modeling: Document-term matrix

Graph Laplacian:
  L = D - A (degree matrix - adjacency matrix)
  Positive semi-definite
  Eigenvalues for spectral clustering
  Second smallest eigenvalue: algebraic connectivity`,
					CodeExamples: `# Advanced Linear Algebra Applications

import math
import random
from typing import List, Tuple

# ============================================================
# PCA Implementation
# ============================================================

class PCA:
    """Principal Component Analysis."""
    
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components: List[List[float]] = []
        self.explained_variance: List[float] = []
        self.mean: List[float] = []
    
    def fit(self, X: List[List[float]]):
        n = len(X)
        d = len(X[0])
        
        # Center data
        self.mean = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
        X_centered = [[X[i][j] - self.mean[j] for j in range(d)]
                      for i in range(n)]
        
        # Compute covariance matrix
        cov = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(i, d):
                val = sum(X_centered[k][i] * X_centered[k][j]
                         for k in range(n)) / (n - 1)
                cov[i][j] = val
                cov[j][i] = val
        
        # Power iteration for top eigenvalues
        cov_matrix = Matrix(cov)
        
        self.components = []
        self.explained_variance = []
        
        M = Matrix([row[:] for row in cov])
        
        for _ in range(self.n_components):
            eigenvalue, eigenvector = power_iteration(M)
            self.explained_variance.append(eigenvalue)
            self.components.append(eigenvector.data[:])
            
            # Deflate: M = M - λvvᵀ
            for i in range(d):
                for j in range(d):
                    M.data[i][j] -= (eigenvalue *
                                     eigenvector.data[i] *
                                     eigenvector.data[j])
    
    def transform(self, X: List[List[float]]) -> List[List[float]]:
        result = []
        for x in X:
            centered = [x[j] - self.mean[j] for j in range(len(x))]
            projected = [sum(centered[j] * self.components[k][j]
                            for j in range(len(centered)))
                        for k in range(self.n_components)]
            result.append(projected)
        return result
    
    def inverse_transform(self, Z: List[List[float]]) -> List[List[float]]:
        result = []
        d = len(self.mean)
        for z in Z:
            reconstructed = [self.mean[j] for j in range(d)]
            for k in range(len(z)):
                for j in range(d):
                    reconstructed[j] += z[k] * self.components[k][j]
            result.append(reconstructed)
        return result
    
    def explained_variance_ratio(self) -> List[float]:
        total = sum(self.explained_variance)
        if total == 0:
            return [0.0] * len(self.explained_variance)
        return [v / total for v in self.explained_variance]


# ============================================================
# Matrix Factorization (NMF-like)
# ============================================================

class MatrixFactorization:
    """Simple matrix factorization for recommender systems."""
    
    def __init__(self, n_factors: int = 10, lr: float = 0.01,
                 reg: float = 0.01, n_epochs: int = 100):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.user_factors: List[List[float]] = []
        self.item_factors: List[List[float]] = []
        self.user_bias: List[float] = []
        self.item_bias: List[float] = []
        self.global_bias: float = 0.0
    
    def fit(self, ratings: List[Tuple[int, int, float]],
            n_users: int, n_items: int):
        # Initialize
        self.user_factors = [
            [random.gauss(0, 0.1) for _ in range(self.n_factors)]
            for _ in range(n_users)]
        self.item_factors = [
            [random.gauss(0, 0.1) for _ in range(self.n_factors)]
            for _ in range(n_items)]
        self.user_bias = [0.0] * n_users
        self.item_bias = [0.0] * n_items
        self.global_bias = sum(r for _, _, r in ratings) / len(ratings)
        
        for epoch in range(self.n_epochs):
            random.shuffle(ratings)
            total_error = 0.0
            
            for user, item, rating in ratings:
                pred = self.predict(user, item)
                error = rating - pred
                total_error += error ** 2
                
                # Update biases
                self.user_bias[user] += self.lr * (
                    error - self.reg * self.user_bias[user])
                self.item_bias[item] += self.lr * (
                    error - self.reg * self.item_bias[item])
                
                # Update factors
                for f in range(self.n_factors):
                    uf = self.user_factors[user][f]
                    vf = self.item_factors[item][f]
                    
                    self.user_factors[user][f] += self.lr * (
                        error * vf - self.reg * uf)
                    self.item_factors[item][f] += self.lr * (
                        error * uf - self.reg * vf)
    
    def predict(self, user: int, item: int) -> float:
        pred = self.global_bias + self.user_bias[user] + self.item_bias[item]
        pred += sum(self.user_factors[user][f] * self.item_factors[item][f]
                   for f in range(self.n_factors))
        return pred
    
    def recommend(self, user: int, n_items: int,
                  exclude: List[int] = None) -> List[Tuple[int, float]]:
        exclude_set = set(exclude or [])
        scores = []
        
        for item in range(len(self.item_factors)):
            if item not in exclude_set:
                scores.append((item, self.predict(user, item)))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:n_items]


# ============================================================
# Spectral Clustering Helper
# ============================================================

def graph_laplacian(adjacency: Matrix) -> Matrix:
    """Compute graph Laplacian L = D - A."""
    n = adjacency.rows
    L = Matrix([row[:] for row in adjacency.data])
    
    for i in range(n):
        degree = sum(adjacency.data[i])
        L.data[i][i] = degree - adjacency.data[i][i]
        for j in range(n):
            if i != j:
                L.data[i][j] = -adjacency.data[i][j]
    
    return L


def normalized_laplacian(adjacency: Matrix) -> Matrix:
    """Compute normalized Laplacian: I - D^{-1/2} A D^{-1/2}."""
    n = adjacency.rows
    degrees = [sum(adjacency.data[i]) for i in range(n)]
    
    L_norm = Matrix.identity(n)
    
    for i in range(n):
        for j in range(n):
            if degrees[i] > 0 and degrees[j] > 0:
                L_norm.data[i][j] -= (adjacency.data[i][j] /
                                      math.sqrt(degrees[i] * degrees[j]))
    
    return L_norm


# ============================================================
# LU Decomposition
# ============================================================

def lu_decomposition(A: Matrix) -> Tuple[Matrix, Matrix]:
    """LU decomposition (without pivoting)."""
    n = A.rows
    L = Matrix.identity(n)
    U = Matrix([row[:] for row in A.data])
    
    for j in range(n):
        for i in range(j + 1, n):
            if abs(U.data[j][j]) < 1e-12:
                continue
            factor = U.data[i][j] / U.data[j][j]
            L.data[i][j] = factor
            for k in range(j, n):
                U.data[i][k] -= factor * U.data[j][k]
    
    return L, U


# ============================================================
# Cholesky Decomposition
# ============================================================

def cholesky(A: Matrix) -> Optional[Matrix]:
    """Cholesky decomposition: A = LLᵀ (A must be positive definite)."""
    n = A.rows
    L = Matrix.zeros(n, n)
    
    for i in range(n):
        for j in range(i + 1):
            s = sum(L.data[i][k] * L.data[j][k] for k in range(j))
            
            if i == j:
                val = A.data[i][i] - s
                if val <= 0:
                    return None  # Not positive definite
                L.data[i][j] = math.sqrt(val)
            else:
                L.data[i][j] = (A.data[i][j] - s) / L.data[j][j]
    
    return L`,
				},
			},
		},
	})
}
