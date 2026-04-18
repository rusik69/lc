package math

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMathModules([]problems.CourseModule{
		{
			ID:          2715,
			Title:       "Math for Programming Interviews",
			Description: "Learn essential mathematical concepts that appear frequently in programming interviews and competitive programming: number theory, combinatorics, probability, and bit manipulation.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "Number Theory for Programmers",
					Content: `Number theory concepts appear frequently in coding interviews and competitive programming. These aren't abstract math — they're practical tools for solving problems.

**GCD (Greatest Common Divisor):**
The largest number that divides both a and b. Uses Euclidean algorithm.

` + "```" + `
gcd(48, 18):
  48 = 18 * 2 + 12
  18 = 12 * 1 + 6
  12 = 6 * 2 + 0    ← remainder is 0, so gcd = 6
` + "```" + `

**LCM (Least Common Multiple):**
LCM(a, b) = (a * b) / GCD(a, b)

**Modular Arithmetic:**
- (a + b) mod m = ((a mod m) + (b mod m)) mod m
- (a * b) mod m = ((a mod m) * (b mod m)) mod m
- Used extensively in competitive programming to prevent overflow
- Common modulus: 10^9 + 7 (prime, fits in 32-bit int)

**Why 10^9 + 7?**
- It's prime (needed for modular inverse)
- Large enough to avoid collisions
- Small enough that (10^9+7)^2 fits in 64-bit integer

**Fast Exponentiation (Binary Exponentiation):**
Compute a^n mod m in O(log n) instead of O(n).

**Prime Numbers:**
- Sieve of Eratosthenes: Find all primes up to N in O(N log log N)
- Primality test: Check divisibility up to sqrt(N)
- Fundamental theorem of arithmetic: Every integer > 1 has a unique prime factorization

**Applications in Programming:**
- **GCD/LCM**: Simplifying fractions, scheduling problems, clock alignment
- **Modular arithmetic**: Hash functions, cryptography, large number calculations
- **Primes**: Hash table sizes, cryptographic keys, random number generation
- **Fast exponentiation**: RSA encryption, competitive programming`,
					CodeExamples: `# Python: Essential number theory functions

def gcd(a, b):
    """Euclidean algorithm — O(log(min(a,b)))"""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """LCM using GCD"""
    return a * b // gcd(a, b)

def fast_pow(base, exp, mod):
    """Binary exponentiation — O(log exp)"""
    result = 1
    base %= mod
    while exp > 0:
        if exp % 2 == 1:
            result = result * base % mod
        exp //= 2
        base = base * base % mod
    return result

# Example: 2^100 mod (10^9 + 7)
MOD = 10**9 + 7
print(fast_pow(2, 100, MOD))  # 976371285

def sieve_of_eratosthenes(n):
    """Find all primes up to n — O(n log log n)"""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(n + 1) if is_prime[i]]

# Primes up to 100
print(sieve_of_eratosthenes(100))
# [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, ...]

def prime_factorization(n):
    """Find prime factors — O(sqrt(n))"""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

print(prime_factorization(360))  # {2: 3, 3: 2, 5: 1} → 2³ × 3² × 5

// Go: GCD and fast exponentiation
func gcd(a, b int) int {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}

func fastPow(base, exp, mod int) int {
    result := 1
    base %= mod
    for exp > 0 {
        if exp%2 == 1 {
            result = result * base % mod
        }
        exp /= 2
        base = base * base % mod
    }
    return result
}`,
				},
				{
					Title: "Combinatorics and Probability",
					Content: `Combinatorics and probability questions are common in interviews, especially for problems involving counting, arrangements, and expected values.

**Fundamental Counting Principles:**

**Multiplication Principle:** If task A can be done in m ways and task B in n ways, then both tasks can be done in m × n ways.

**Addition Principle:** If task A can be done in m ways and task B in n ways (mutually exclusive), then either task can be done in m + n ways.

**Permutations:**
Ordered arrangements of n items:
- All: n! = n × (n-1) × ... × 1
- Choose r from n: P(n,r) = n!/(n-r)!
- With repetitions: n^r

**Combinations:**
Unordered selections of r items from n:
- C(n,r) = n!/(r! × (n-r)!)
- Also written as "n choose r" or (n r)
- C(n,r) = C(n, n-r) (symmetry)

**Pascal's Triangle:**
C(n,r) = C(n-1,r-1) + C(n-1,r) — useful for DP!

**Probability Basics:**
- P(A) = favorable outcomes / total outcomes (0 ≤ P ≤ 1)
- P(A or B) = P(A) + P(B) - P(A and B)
- P(A and B) = P(A) × P(B|A)
- Independent events: P(A and B) = P(A) × P(B)

**Expected Value:**
E[X] = Σ(value × probability)

**Interview Applications:**

1. **Password combinations**: How many 8-char passwords with uppercase, lowercase, digits?
   - 62^8 ≈ 2.18 × 10^14

2. **Card probability**: Probability of getting a flush in poker?
   - C(13,5) × 4 / C(52,5) ≈ 0.198%

3. **Birthday problem**: With 23 people, probability of shared birthday > 50%
   - P(no match) = 365/365 × 364/365 × ... × 343/365

4. **Subsets**: How many subsets of a set of size n?
   - 2^n (each element is either in or out)

5. **Paths in a grid**: Number of paths from (0,0) to (m,n) moving only right/down?
   - C(m+n, m) = (m+n)! / (m! × n!)

**Catalan Numbers:**
Appear in many combinatorial problems:
C_n = C(2n, n) / (n+1)

Applications:
- Number of valid parentheses strings of length 2n
- Number of BSTs with n nodes
- Number of paths that don't cross the diagonal
- Number of triangulations of a polygon

First few: 1, 1, 2, 5, 14, 42, 132, 429, ...`,
					CodeExamples: `# Python: Combinatorics functions

from math import factorial, comb
from functools import lru_cache

def permutations(n, r):
    """P(n,r) = n!/(n-r)!"""
    return factorial(n) // factorial(n - r)

def combinations(n, r):
    """C(n,r) = n!/(r!(n-r)!)"""
    return comb(n, r)  # Built-in since Python 3.8

# Grid paths: from (0,0) to (m,n)
def grid_paths(m, n):
    """Number of paths moving only right or down."""
    return comb(m + n, m)

print(grid_paths(3, 3))  # 20 paths in a 3x3 grid

# Catalan number
def catalan(n):
    """C_n = C(2n,n)/(n+1)"""
    return comb(2 * n, n) // (n + 1)

# Valid parentheses with n pairs
print(catalan(3))  # 5: ((())), (()()), (())(), ()(()), ()()()
print(catalan(4))  # 14

# Birthday problem
def birthday_probability(n_people):
    """Probability at least 2 people share a birthday."""
    p_no_match = 1.0
    for i in range(n_people):
        p_no_match *= (365 - i) / 365
    return 1 - p_no_match

print(f"23 people: {birthday_probability(23):.1%}")  # 50.7%
print(f"50 people: {birthday_probability(50):.1%}")  # 97.0%
print(f"70 people: {birthday_probability(70):.1%}")  # 99.9%

# Expected value: Dice rolling
def expected_dice_rolls_to_get_6():
    """E[rolls to get first 6] = 1/p = 6"""
    return 1 / (1/6)

# Combinations with modular arithmetic (for large n)
MOD = 10**9 + 7

@lru_cache(maxsize=None)
def mod_comb(n, r):
    """C(n,r) mod p using Pascal's triangle with memoization."""
    if r == 0 or r == n:
        return 1
    return (mod_comb(n-1, r-1) + mod_comb(n-1, r)) % MOD

# Or use Fermat's little theorem for modular inverse
def mod_inverse(a, p):
    """a^(-1) mod p using Fermat's little theorem (p must be prime)."""
    return pow(a, p - 2, p)

def nCr_mod(n, r, p=MOD):
    """C(n,r) mod p."""
    if r > n:
        return 0
    num = 1
    den = 1
    for i in range(r):
        num = num * (n - i) % p
        den = den * (i + 1) % p
    return num * mod_inverse(den, p) % p`,
				},
				{
					Title: "Bit Manipulation",
					Content: `Bit manipulation is a powerful technique that operates directly on binary representations of numbers. Many interview problems have elegant bit manipulation solutions.

**Basic Bitwise Operations:**

| Operation | Symbol | Example | Result |
|-----------|--------|---------|--------|
| AND | & | 1010 & 1100 | 1000 |
| OR | \| | 1010 \| 1100 | 1110 |
| XOR | ^ | 1010 ^ 1100 | 0110 |
| NOT | ~ | ~1010 | 0101 |
| Left Shift | << | 0001 << 2 | 0100 |
| Right Shift | >> | 1000 >> 2 | 0010 |

**Key Properties:**

**XOR (^):**
- a ^ 0 = a (identity)
- a ^ a = 0 (self-inverse)
- a ^ b ^ a = b (used to find single missing/duplicate element)
- XOR is commutative and associative

**AND (&):**
- a & 0 = 0
- a & a = a
- a & (a-1) = removes lowest set bit (Brian Kernighan's trick)
- a & (-a) = isolates lowest set bit

**Shifts:**
- a << n = a × 2^n
- a >> n = a ÷ 2^n (integer division)

**Common Bit Tricks:**

1. **Check if power of 2**: n & (n-1) == 0 (and n > 0)
2. **Count set bits**: Repeatedly apply n & (n-1)
3. **Get ith bit**: (n >> i) & 1
4. **Set ith bit**: n | (1 << i)
5. **Clear ith bit**: n & ~(1 << i)
6. **Toggle ith bit**: n ^ (1 << i)
7. **Check odd/even**: n & 1 (0 = even, 1 = odd)
8. **Swap without temp**: a ^= b; b ^= a; a ^= b

**Interview Problems Solved with Bits:**

1. **Single Number**: All numbers appear twice except one → XOR all
2. **Missing Number**: 0 to n, one missing → XOR range with array
3. **Power of Two**: n & (n-1) == 0
4. **Counting Bits**: Count 1s in binary representation
5. **Subsets**: Iterate 0 to 2^n-1, each bit represents include/exclude
6. **Bitmask DP**: Use integers as sets for state compression`,
					CodeExamples: `# Python: Bit manipulation techniques

# 1. Single Number (all appear twice except one)
def single_number(nums):
    result = 0
    for n in nums:
        result ^= n  # XOR cancels duplicates
    return result

print(single_number([4, 1, 2, 1, 2]))  # 4

# 2. Count set bits (Brian Kernighan's algorithm)
def count_bits(n):
    count = 0
    while n:
        n &= n - 1  # Remove lowest set bit
        count += 1
    return count

print(count_bits(0b11010110))  # 5

# 3. Power of two check
def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

print(is_power_of_two(16))  # True (10000)
print(is_power_of_two(18))  # False (10010)

# 4. Generate all subsets using bitmask
def subsets(nums):
    n = len(nums)
    result = []
    for mask in range(1 << n):  # 0 to 2^n - 1
        subset = []
        for i in range(n):
            if mask & (1 << i):  # Check if ith bit is set
                subset.append(nums[i])
        result.append(subset)
    return result

print(subsets([1, 2, 3]))
# [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]

# 5. XOR from 1 to n (pattern-based, O(1))
def xor_1_to_n(n):
    """XOR of all numbers from 1 to n."""
    # Pattern repeats every 4:
    # n%4==0: n, n%4==1: 1, n%4==2: n+1, n%4==3: 0
    r = n % 4
    if r == 0: return n
    if r == 1: return 1
    if r == 2: return n + 1
    return 0

# 6. Missing number using XOR
def missing_number(nums):
    """Find missing number in 0..n"""
    n = len(nums)
    result = n
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result

print(missing_number([3, 0, 1]))  # 2

// Go: Bit manipulation
func countBits(n int) int {
    count := 0
    for n > 0 {
        n &= n - 1
        count++
    }
    return count
}

func isPowerOfTwo(n int) bool {
    return n > 0 && n&(n-1) == 0
}

func singleNumber(nums []int) int {
    result := 0
    for _, n := range nums {
        result ^= n
    }
    return result
}`,
				},
			},
		},
	})
}
