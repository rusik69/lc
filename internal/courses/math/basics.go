package math

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMathModules([]problems.CourseModule{
		{
			ID:          500,
			Title:       "Introduction to Mathematics",
			Description: "Learn fundamental mathematical concepts including arithmetic, algebra, and number theory.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "What is Mathematics?",
					Content: `**Introduction: The Universal Language**

Mathematics is far more than just numbers and calculations - it's the universal language of patterns, relationships, and logical structure that underlies everything in our universe. From the spirals of galaxies to the algorithms running on your computer, mathematics provides the framework for understanding and describing reality.

**What Makes Mathematics Unique?**

Mathematics is unique among disciplines because it deals with abstract concepts that exist independently of the physical world. A mathematical truth like "2 + 2 = 4" is true regardless of whether you're counting apples, electrons, or galaxies. This universality makes mathematics the foundation of all quantitative sciences.

**Historical Development:**

Mathematics has evolved over thousands of years, from ancient civilizations using arithmetic for trade and construction, to modern mathematicians exploring abstract structures. Key milestones include:
- Ancient Egypt and Babylon: Practical arithmetic and geometry (3000-500 BCE)
- Ancient Greece: Formal proofs and logical reasoning (600-300 BCE)
- Islamic Golden Age: Algebra and number theory (800-1200 CE)
- European Renaissance: Analytical geometry and calculus (1500-1700 CE)
- Modern Era: Abstract algebra, topology, and computational mathematics (1800-present)

**Branches of Mathematics:**

**1. Arithmetic: The Foundation**

Arithmetic is the most fundamental branch, dealing with basic operations on numbers. It's where all mathematical learning begins and forms the foundation for everything else.

**Core Operations:**
- **Addition**: Combining quantities (3 + 5 = 8)
- **Subtraction**: Finding differences (8 - 3 = 5)
- **Multiplication**: Repeated addition (3 × 4 = 3 + 3 + 3 + 3 = 12)
- **Division**: Splitting into equal parts (12 ÷ 3 = 4)

**Why Arithmetic Matters:**
Every algorithm, from the simplest loop counter to complex machine learning models, relies on arithmetic operations. Understanding arithmetic deeply helps you:
- Analyze algorithm efficiency (counting operations)
- Understand data structures (array indexing uses arithmetic)
- Debug numerical code (off-by-one errors are arithmetic mistakes)
- Optimize performance (knowing when multiplication is faster than repeated addition)

**Common Pitfall:** Many students think arithmetic is "too simple" and skip mastering it, but arithmetic errors are the #1 cause of bugs in numerical code. A misplaced operator or incorrect order of operations can break entire systems.

**2. Algebra: The Language of Relationships**

Algebra introduces the powerful concept of variables - symbols that can represent unknown or changing values. This abstraction allows us to solve problems without knowing specific numbers.

**Key Concepts:**
- **Variables**: Symbols (like x, y) representing unknown values
- **Equations**: Mathematical statements expressing equality
- **Expressions**: Combinations of variables, numbers, and operations
- **Functions**: Rules mapping inputs to outputs

**Why Algebra Matters in CS:**
- **Algorithm Analysis**: We use algebra to express time complexity (O(n²), O(n log n))
- **Data Structures**: Array indices, tree heights, hash functions all use algebraic expressions
- **Optimization**: Finding maximum/minimum values requires solving equations
- **Recurrence Relations**: Analyzing recursive algorithms uses algebraic techniques

**Real Example:** When analyzing binary search, we set up the equation: "How many times can we divide n by 2 before reaching 1?" This leads to the algebraic solution: log₂(n), which tells us binary search is O(log n).

**3. Geometry: The Mathematics of Space**

Geometry studies shapes, sizes, positions, and properties of space. While it might seem abstract, geometry is everywhere in computer science.

**Fundamental Concepts:**
- **Points**: Locations in space (coordinates)
- **Lines**: Straight paths extending infinitely
- **Angles**: Measures of rotation or intersection
- **Shapes**: Circles, triangles, polygons, and their properties

**Why Geometry Matters:**
- **Computer Graphics**: Every pixel, every 3D model, every animation uses geometric calculations
- **Game Development**: Collision detection, pathfinding, camera positioning all rely on geometry
- **Data Visualization**: Charts, graphs, and plots are geometric representations
- **Spatial Data Structures**: Quadtrees, octrees, and spatial hashing use geometric partitioning
- **Image Processing**: Rotation, scaling, and transformation are geometric operations

**Practical Example:** When implementing a 2D game, you need to check if a point (player position) is inside a circle (enemy detection radius). This uses the distance formula from geometry: distance = √[(x₂-x₁)² + (y₂-y₁)²]. If distance < radius, the player is detected.

**4. Number Theory: The Mathematics of Integers**

Number theory studies properties of integers - whole numbers and their relationships. While it might seem theoretical, number theory is the foundation of modern cryptography and security.

**Key Topics:**
- **Prime Numbers**: Numbers divisible only by 1 and themselves
- **Divisibility**: When one number divides another evenly
- **Modular Arithmetic**: Arithmetic with "wrap-around" (clock arithmetic)
- **GCD and LCM**: Greatest common divisor and least common multiple

**Why Number Theory Matters:**
- **Cryptography**: RSA encryption, used in HTTPS and secure communications, relies on the difficulty of factoring large numbers
- **Hash Functions**: Many hash algorithms use modular arithmetic
- **Random Number Generation**: Pseudorandom generators use number theory
- **Error Detection**: Checksums and error-correcting codes use divisibility properties
- **Algorithm Design**: GCD algorithm (Euclidean algorithm) is one of the oldest and most efficient algorithms

**Real-World Impact:** Every time you make a secure online purchase, number theory is protecting your credit card information. The security relies on mathematical problems that are easy to solve in one direction (multiplying primes) but extremely difficult in reverse (factoring the product).

**5. Calculus: The Mathematics of Change**

Calculus studies rates of change and accumulation. While you might not use calculus directly in every programming task, understanding it helps with:
- **Optimization**: Finding maximum/minimum values (gradient descent in machine learning)
- **Rate of Change**: Understanding how algorithms scale
- **Integration**: Accumulating values over time or space
- **Limits**: Understanding asymptotic behavior (Big O notation is based on limits)

**Why Mathematics Matters in Computer Science:**

**1. Algorithm Design and Analysis:**
Every algorithm is fundamentally a mathematical process. Understanding mathematics helps you:
- Design efficient algorithms (knowing when O(n log n) is better than O(n²))
- Analyze complexity (using limits and asymptotic analysis)
- Optimize performance (understanding trade-offs)

**2. Data Structures:**
Data structures are mathematical structures implemented in code:
- Arrays: Mathematical sequences
- Trees: Mathematical graphs
- Hash tables: Mathematical functions (hash functions)
- Heaps: Mathematical trees with ordering properties

**3. Cryptography and Security:**
Modern security relies entirely on mathematics:
- Public-key cryptography uses number theory
- Hash functions use modular arithmetic
- Digital signatures use mathematical proofs

**4. Machine Learning and AI:**
Machine learning is applied mathematics:
- Linear algebra for data transformations
- Calculus for optimization (gradient descent)
- Probability and statistics for making predictions
- Information theory for feature selection

**5. Graphics and Visualization:**
Computer graphics is applied geometry and linear algebra:
- 3D transformations use matrix mathematics
- Ray tracing uses geometric calculations
- Shading and lighting use calculus concepts

**Problem-Solving Skills Developed:**

**Logical Reasoning:**
Mathematics trains you to think logically and sequentially. When debugging code, you use the same logical reasoning: "If A is true and B follows from A, then B must be true."

**Pattern Recognition:**
Mathematics is all about recognizing patterns. In programming, this translates to:
- Recognizing when a problem matches a known algorithm pattern
- Identifying code smells (patterns that indicate problems)
- Seeing optimization opportunities

**Abstract Thinking:**
Mathematics teaches abstraction - working with concepts rather than concrete objects. This is essential in programming:
- Creating reusable functions (abstraction)
- Designing APIs (abstracting complexity)
- Working with data structures (abstracting data organization)

**Breaking Complex Problems into Simpler Parts:**
Mathematical problem-solving involves breaking complex problems into manageable steps. This is exactly what programming requires:
- Decomposing large functions into smaller ones
- Solving subproblems independently
- Combining solutions

**Common Misconceptions:**

**"I don't need math for programming"**
While you can write simple programs without deep math, advanced programming requires mathematical thinking. Even "simple" tasks like sorting use mathematical concepts (comparison operations, ordering relations).

**"Math is just memorizing formulas"**
Mathematics is about understanding relationships and reasoning logically. Memorization helps, but deep understanding comes from seeing why formulas work and when to apply them.

**"I'm bad at math, so I can't be a good programmer"**
Many successful programmers initially struggled with math. The key is persistence and understanding that mathematical thinking is a skill that improves with practice, just like programming.

**Real-World Applications:**

**Finance:**
- Interest calculations use exponential functions
- Risk analysis uses probability and statistics
- Portfolio optimization uses linear algebra

**Engineering:**
- Structural analysis uses calculus and linear algebra
- Signal processing uses Fourier transforms (advanced calculus)
- Control systems use differential equations

**Science:**
- Data analysis uses statistics
- Modeling uses differential equations
- Predictions use probability theory

**Technology:**
- Algorithm design uses discrete mathematics
- Performance optimization uses complexity analysis
- System design uses mathematical modeling

**How to Approach Learning Mathematics:**

1. **Start with Intuition**: Understand the "why" before memorizing the "what"
2. **Practice Actively**: Work through problems, don't just read solutions
3. **Connect to Programming**: See how mathematical concepts appear in code
4. **Build Gradually**: Master fundamentals before moving to advanced topics
5. **Use Multiple Resources**: Different explanations help solidify understanding

**Next Steps:**

As we progress through this course, you'll see how each mathematical concept connects directly to programming and computer science. We'll start with fundamentals and build up to more advanced topics, always showing the practical applications.`,
					CodeExamples: `# Python examples of mathematical operations

# Basic arithmetic
a = 10
b = 3
print(f"Addition: {a} + {b} = {a + b}")
print(f"Subtraction: {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")
print(f"Floor Division: {a} // {b} = {a // b}")
print(f"Modulo: {a} % {b} = {a % b}")
print(f"Exponentiation: {a} ** {b} = {a ** b}")

# Working with fractions
from fractions import Fraction
f1 = Fraction(1, 3)
f2 = Fraction(1, 2)
print(f"Fraction addition: {f1} + {f2} = {f1 + f2}")

# Mathematical functions
import math
print(f"Square root of 16: {math.sqrt(16)}")
print(f"Pi: {math.pi}")
print(f"Euler's number: {math.e}")
print(f"Logarithm base 10 of 100: {math.log10(100)}")
print(f"Natural logarithm of e: {math.log(math.e)}")`,
				},
				{
					Title: "Number Systems and Operations",
					Content: `**Introduction: Understanding Number Systems**

Number systems form the foundation of all mathematics and computing. Understanding how numbers are organized helps you work with different data types in programming, understand algorithm complexity, and reason about numerical computations.

**The Hierarchy of Number Systems:**

Numbers are organized in a hierarchy, with each system building upon the previous one. This hierarchy reflects both historical development and logical structure.

**1. Natural Numbers (N): The Counting Numbers**

Natural numbers are the most basic number system: 1, 2, 3, 4, 5, ...

**Properties:**
- Start at 1 (some definitions include 0, but we'll use the traditional definition)
- Used for counting discrete, indivisible objects
- No zero, no negative numbers
- Infinite but countable

**Why Natural Numbers Matter:**
- **Array Indices**: In most programming languages, arrays start at index 0 or 1 (natural numbers)
- **Loop Counters**: for i in range(1, n+1) uses natural numbers
- **Counting Operations**: When analyzing algorithms, we count operations using natural numbers
- **Discrete Mathematics**: Many CS problems deal with discrete quantities (nodes, edges, items)

**Common Mistake:** Confusing natural numbers with whole numbers. Natural numbers don't include zero, which matters when indexing arrays or counting items.

**2. Whole Numbers: Adding Zero**

Whole numbers extend natural numbers by including zero: 0, 1, 2, 3, 4, ...

**The Importance of Zero:**
Zero is a profound mathematical concept that took centuries to develop. It serves multiple purposes:
- **Placeholder**: Allows positional notation (10 vs 1)
- **Identity Element**: For addition (a + 0 = a)
- **Null/Empty**: Represents absence or nothing
- **Reference Point**: Starting point for measurements

**Why Zero Matters in Programming:**
- **Array Indexing**: Many languages use 0-based indexing
- **Initialization**: Variables often start at 0
- **Null/None Values**: Zero often represents "no value" or "empty"
- **Boolean Logic**: 0 represents false in many languages

**3. Integers (Z): Including Negatives**

Integers extend whole numbers to include negatives: ..., -3, -2, -1, 0, 1, 2, 3, ...

**Why Integers Exist:**
Negative numbers were initially controversial but are essential for:
- Representing debts or losses
- Measuring temperatures below zero
- Describing directions (forward/backward)
- Solving equations (x + 5 = 2 requires x = -3)

**Integer Representation in Computers:**
Computers represent integers using binary:
- **Signed Integers**: Use two's complement representation
- **Range**: For n bits, signed integers range from -2^(n-1) to 2^(n-1)-1
- **Overflow**: Adding large integers can cause overflow (wraps around)

**Example:** In 8-bit signed integers:
- Maximum: 127 (01111111)
- Minimum: -128 (10000000)
- Adding 127 + 1 = -128 (overflow!)

**Why This Matters:** Understanding integer limits helps prevent bugs. Many security vulnerabilities come from integer overflow.

**4. Rational Numbers (Q): Fractions**

Rational numbers can be expressed as fractions: a/b where a and b are integers, b ≠ 0.

**Key Properties:**
- Includes all integers (any integer n = n/1)
- Can be terminating decimals (1/4 = 0.25)
- Can be repeating decimals (1/3 = 0.333...)
- Dense: Between any two rationals, there's another rational

**Decimal Representations:**
- **Terminating**: 1/2 = 0.5, 1/4 = 0.25, 1/5 = 0.2
- **Repeating**: 1/3 = 0.333..., 1/6 = 0.1666..., 1/7 = 0.142857142857...

**Why Rational Numbers Matter:**
- **Exact Arithmetic**: Fractions allow exact representation (1/3 exactly, not 0.333...)
- **Financial Calculations**: Money calculations need exact fractions
- **Ratios**: Many algorithms use ratios (aspect ratios, probabilities)

**Common Mistake:** Assuming all decimals are exact. 0.1 + 0.2 ≠ 0.3 in floating-point because these are approximations of fractions.

**5. Irrational Numbers: Non-Repeating, Non-Terminating**

Irrational numbers cannot be expressed as fractions. Their decimal expansion never repeats or terminates.

**Famous Examples:**
- **√2**: The diagonal of a unit square (≈1.41421356...)
- **π**: Ratio of circle's circumference to diameter (≈3.14159265...)
- **e**: Base of natural logarithms (≈2.71828182...)
- **φ**: Golden ratio (≈1.61803398...)

**Why Irrational Numbers Matter:**
- **Geometric Calculations**: Circles, triangles, and other shapes involve irrational numbers
- **Exponential Growth**: e appears in compound interest, population growth
- **Trigonometry**: Angles and trigonometric functions involve π

**In Programming:** We approximate irrationals with floating-point numbers, which introduces rounding errors. Understanding this helps debug numerical issues.

**6. Real Numbers (R): The Complete Number Line**

Real numbers include all rational and irrational numbers - everything on the number line.

**Properties:**
- Complete: No "gaps" in the number line
- Continuous: Between any two reals, there are infinitely many reals
- Used for measurements (length, weight, time)

**Real Numbers in Computing:**
- **Floating-Point**: Approximations of real numbers
- **Precision Limits**: Cannot represent all reals exactly
- **Rounding Errors**: Accumulate in calculations

**Basic Operations: Understanding the Properties**

**Addition: Combining Quantities**

Addition is the most fundamental operation. Understanding its properties helps in algorithm design.

**Commutative Property: a + b = b + a**
- Order doesn't matter: 3 + 5 = 5 + 3 = 8
- **Why It Matters:** Allows reordering operations for optimization
- **In Programming:** Can reorder array operations if order doesn't matter

**Associative Property: (a + b) + c = a + (b + c)**
- Grouping doesn't matter: (2 + 3) + 4 = 2 + (3 + 4) = 9
- **Why It Matters:** Allows parallel computation
- **In Programming:** Can compute (a + b) and (c + d) simultaneously, then add results

**Identity Element: a + 0 = a**
- Zero doesn't change the value
- **Why It Matters:** Initialization in loops and accumulators
- **In Programming:** sum = 0 before a loop, then sum += value

**Inverse Element: a + (-a) = 0**
- Every number has an opposite
- **Why It Matters:** Allows subtraction (a - b = a + (-b))

**Multiplication: Repeated Addition**

Multiplication is repeated addition: 3 × 4 = 3 + 3 + 3 + 3 = 12

**Commutative Property: a × b = b × a**
- Order doesn't matter: 3 × 5 = 5 × 3 = 15
- **Why It Matters:** Matrix multiplication is NOT commutative - this matters in linear algebra

**Associative Property: (a × b) × c = a × (b × c)**
- Grouping doesn't matter: (2 × 3) × 4 = 2 × (3 × 4) = 24

**Identity Element: a × 1 = a**
- One doesn't change the value
- **Why It Matters:** Initialization for products

**Distributive Property: a × (b + c) = a × b + a × c**
- Multiplication distributes over addition
- **Why It Matters:** 
  - Factoring expressions: 3x + 6 = 3(x + 2)
  - Algorithm optimization: Can compute a×b and a×c separately, then add
  - **Example:** Computing 7 × 13 = 7 × (10 + 3) = 7×10 + 7×3 = 70 + 21 = 91

**Order of Operations (PEMDAS): Avoiding Ambiguity**

Mathematical expressions can be ambiguous without rules. PEMDAS (Parentheses, Exponents, Multiplication/Division, Addition/Subtraction) resolves this.

**The Rules:**
1. **Parentheses**: Evaluate innermost first
2. **Exponents**: Evaluate powers
3. **Multiplication and Division**: Left to right (equal precedence)
4. **Addition and Subtraction**: Left to right (equal precedence)

**Step-by-Step Example: 2 + 3 × 4² - (5 + 3) ÷ 2**

1. Parentheses: (5 + 3) = 8 → 2 + 3 × 4² - 8 ÷ 2
2. Exponents: 4² = 16 → 2 + 3 × 16 - 8 ÷ 2
3. Multiplication/Division (left to right):
   - 3 × 16 = 48 → 2 + 48 - 8 ÷ 2
   - 8 ÷ 2 = 4 → 2 + 48 - 4
4. Addition/Subtraction (left to right):
   - 2 + 48 = 50 → 50 - 4 = 46

**Common Mistakes:**

**Mistake 1: Doing operations left to right without precedence**
- Wrong: 2 + 3 × 4 = 5 × 4 = 20
- Correct: 2 + 3 × 4 = 2 + 12 = 14

**Mistake 2: Forgetting that multiplication and division have equal precedence**
- Expression: 8 ÷ 4 × 2
- Wrong: 8 ÷ (4 × 2) = 8 ÷ 8 = 1 (if you think × comes first)
- Correct: (8 ÷ 4) × 2 = 2 × 2 = 4 (left to right)

**Mistake 3: Incorrectly applying distributive property**
- Wrong: 2 + 3 × 4 = (2 + 3) × 4 = 20
- Correct: 2 + 3 × 4 = 2 + 12 = 14
- Note: Distributive property is a × (b + c), not (a + b) × c

**Why Order of Operations Matters in Programming:**

Programming languages follow similar rules, but with some differences:
- **Python:** Follows mathematical precedence
- **Parentheses:** Always evaluated first (use them for clarity!)
- **Operator Precedence:** *, / have higher precedence than +, -
- **Left to Right:** For operators of equal precedence

**Example in Code:**

    result = 2 + 3 * 4  # = 14, not 20
    result = (2 + 3) * 4  # = 20 (parentheses change order)

**Best Practice:** Use parentheses liberally in code for clarity, even when not strictly necessary. It makes your intent clear and prevents bugs.

**Properties in Algorithm Design:**

Understanding these properties helps optimize algorithms:

**Commutativity:** If operations commute, you can reorder them:
- Useful for parallel processing
- Example: Summing array elements can be done in any order

**Associativity:** If operations associate, you can regroup them:
- Useful for divide-and-conquer algorithms
- Example: (a + b) + (c + d) can be computed in parallel

**Distributivity:** Allows factoring and optimization:
- Example: n × (a + b) can be computed as n×a + n×b
- Useful when n×a and n×b are easier to compute separately

**Next Steps:**

Mastering number systems and operations provides the foundation for all mathematical thinking. In the next lessons, we'll see how these concepts apply to more complex problems and connect directly to programming challenges.`,
					CodeExamples: `# Number system examples in Python

# Integers
positive_int = 42
negative_int = -17
zero = 0

# Floating point (real numbers)
pi = 3.14159
e = 2.71828

# Complex numbers
complex_num = 3 + 4j
print(f"Complex number: {complex_num}")
print(f"Real part: {complex_num.real}")
print(f"Imaginary part: {complex_num.imag}")

# Order of operations
result1 = 2 + 3 * 4  # = 14 (multiplication first)
result2 = (2 + 3) * 4  # = 20 (parentheses first)
print(f"2 + 3 * 4 = {result1}")
print(f"(2 + 3) * 4 = {result2}")

# Properties demonstration
a, b, c = 5, 3, 2

# Commutative property
print(f"Addition commutative: {a} + {b} = {b} + {a} → {a + b} = {b + a}")
print(f"Multiplication commutative: {a} * {b} = {b} * {a} → {a * b} = {b * a}")

# Associative property
print(f"Addition associative: ({a} + {b}) + {c} = {a} + ({b} + {c}) → {(a + b) + c} = {a + (b + c)}")

# Distributive property
print(f"Distributive: {a} * ({b} + {c}) = {a} * {b} + {a} * {c} → {a * (b + c)} = {a * b + a * c}")`,
				},
				{
					Title: "Prime Numbers and Divisibility",
					Content: `**Introduction: The Building Blocks of Numbers**

Prime numbers are often called the "atoms" of mathematics - just as atoms are the fundamental building blocks of matter, prime numbers are the fundamental building blocks of all integers. Understanding primes is crucial for cryptography, algorithm design, and many areas of computer science.

**What Are Prime Numbers?**

A prime number is a natural number greater than 1 that has exactly two distinct positive divisors: 1 and itself. In other words, a prime number cannot be formed by multiplying two smaller natural numbers.

**Formal Definition:**
A number p > 1 is prime if and only if its only divisors are 1 and p.

**Examples of Prime Numbers:**
- First few primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, ...
- Notice: They become less frequent as numbers get larger
- There are infinitely many primes (proven by Euclid around 300 BCE)

**Special Cases:**

**2 is Special:**
- 2 is the only even prime number
- Why? Any other even number is divisible by 2, so it can't be prime
- This makes 2 unique among all primes

**1 is NOT Prime:**
- By definition, 1 is excluded from primes
- Why? If 1 were prime, prime factorization wouldn't be unique
- Example: 6 = 2 × 3, but also 6 = 1 × 2 × 3, and 6 = 1 × 1 × 2 × 3, etc.
- Excluding 1 ensures unique factorization (Fundamental Theorem of Arithmetic)

**Why Primes Matter: Fundamental Theorem of Arithmetic**

Every integer greater than 1 can be uniquely expressed as a product of prime numbers (up to the order of factors). This is one of the most important theorems in mathematics.

**Example:** 60 = 2² × 3 × 5
- This is the ONLY way to write 60 as a product of primes (ignoring order)
- You can't write 60 = 2 × 2 × 3 × 5 × 1 (1 doesn't count)
- You can't write 60 = 4 × 3 × 5 (4 is not prime)

**Why This Uniqueness Matters:**
- Ensures consistent mathematical results
- Allows efficient algorithms for GCD, LCM, and factorization
- Forms the basis for cryptographic security

**Composite Numbers: Non-Primes**

Composite numbers are natural numbers greater than 1 that are not prime. They can be factored into smaller positive integers.

**Examples:**
- 4 = 2 × 2 (smallest composite)
- 6 = 2 × 3
- 8 = 2 × 2 × 2 = 2³
- 9 = 3 × 3 = 3²
- 10 = 2 × 5
- 12 = 2 × 2 × 3 = 2² × 3

**Key Insight:** Every composite number has at least one prime factor less than or equal to its square root. This is why we only need to check up to √n when testing for primality.

**Why Prime Numbers Are Critical in Computer Science:**

**1. Cryptography - RSA Encryption:**
RSA encryption, used in HTTPS, SSL/TLS, and secure communications, relies on the difficulty of factoring large composite numbers into their prime factors.

**How RSA Works (Simplified):**
- Choose two large primes p and q (typically 300+ digits)
- Calculate n = p × q (public key component)
- Encryption is easy: multiply p and q
- Decryption requires factoring n, which is computationally infeasible for large primes
- Security: Factoring a 600-digit number would take billions of years with current computers

**2. Hash Functions:**
Many hash functions use prime numbers to ensure good distribution:
- Prime table sizes reduce collisions
- Prime multipliers in hash functions improve randomness
- Example: Java's String.hashCode() uses prime 31

**3. Random Number Generation:**
Pseudorandom number generators often use prime numbers:
- Linear congruential generators use prime moduli
- Mersenne primes are used in high-quality generators

**4. Algorithm Optimization:**
- Prime numbers help optimize certain algorithms
- Example: Using prime-sized hash tables reduces clustering

**Testing for Primality: Trial Division Method**

The simplest method to test if a number n is prime is trial division: try dividing n by all integers from 2 to √n.

**Why Only Up to √n?**
If n has a factor greater than √n, it must also have a factor less than √n. Here's why:
- Suppose n = a × b where both a and b > √n
- Then a × b > √n × √n = n (contradiction!)
- So at least one factor must be ≤ √n

**Step-by-Step Example: Is 17 prime?**

1. Check if 17 < 2: No, proceed
2. Check if 17 = 2: No, proceed
3. Check if 17 is even: No (would be divisible by 2)
4. Check divisibility by odd numbers up to √17 ≈ 4.12:
   - 17 ÷ 3 = 5.67... (not integer)
   - 17 ÷ 5 = 3.4 (not integer, but we only need to check up to 4)
5. Since no divisors found, 17 is prime

**Optimization:** We only need to check odd numbers (and 2), since even numbers > 2 are composite.

**Divisibility Rules: Quick Tests**

Divisibility rules let you quickly determine if a number is divisible by another without performing full division. These are based on modular arithmetic properties.

**Divisible by 2:**
- **Rule:** Last digit is even (0, 2, 4, 6, 8)
- **Why:** In base 10, only the ones place determines even/odd
- **Example:** 124 → last digit is 4 (even) → divisible by 2
- **Counter-example:** 123 → last digit is 3 (odd) → NOT divisible by 2

**Divisible by 3:**
- **Rule:** Sum of digits is divisible by 3
- **Why:** 10 ≡ 1 (mod 3), so 10ⁿ ≡ 1 (mod 3) for all n
- **Example:** 123 → 1+2+3 = 6, and 6 is divisible by 3 → 123 is divisible by 3
- **Step-by-step:** 
  - 123 = 1×100 + 2×10 + 3×1
  - ≡ 1×1 + 2×1 + 3×1 (mod 3) since 10 ≡ 1
  - ≡ 1 + 2 + 3 = 6 ≡ 0 (mod 3)

**Divisible by 4:**
- **Rule:** Last two digits form a number divisible by 4
- **Why:** 100 = 4 × 25, so any number = (hundreds part) + (last two digits)
- **Example:** 124 → last two digits are 24, and 24 ÷ 4 = 6 → divisible by 4
- **Another example:** 1236 → last two digits are 36, and 36 ÷ 4 = 9 → divisible by 4

**Divisible by 5:**
- **Rule:** Last digit is 0 or 5
- **Why:** In base 10, only the ones place matters for divisibility by 5
- **Example:** 125 → last digit is 5 → divisible by 5
- **Example:** 120 → last digit is 0 → divisible by 5

**Divisible by 6:**
- **Rule:** Divisible by both 2 AND 3
- **Why:** 6 = 2 × 3, and 2 and 3 are relatively prime
- **Example:** 126 → 
  - Even? Yes (ends in 6)
  - Sum of digits = 1+2+6 = 9, divisible by 3? Yes
  - Therefore divisible by 6

**Divisible by 9:**
- **Rule:** Sum of digits is divisible by 9
- **Why:** Similar to rule for 3, but 10 ≡ 1 (mod 9)
- **Example:** 126 → 1+2+6 = 9, and 9 is divisible by 9 → 126 is divisible by 9
- **Note:** This is why the "casting out nines" method works for checking arithmetic

**Divisible by 11:**
- **Rule:** Alternating sum of digits is divisible by 11
- **How:** Start from right, alternate signs: (ones) - (tens) + (hundreds) - (thousands) + ...
- **Why:** 10 ≡ -1 (mod 11), so 10ⁿ ≡ (-1)ⁿ (mod 11)
- **Example:** 121 → 
  - From right: 1 - 2 + 1 = 0
  - 0 is divisible by 11 → 121 is divisible by 11
- **Another example:** 1331 →
  - From right: 1 - 3 + 3 - 1 = 0
  - Divisible by 11

**Common Mistake:** Forgetting that divisibility by 6 requires BOTH conditions (divisible by 2 AND 3), not just one.

**Greatest Common Divisor (GCD): Finding Shared Factors**

The GCD of two numbers is the largest positive integer that divides both numbers without remainder.

**Notation:** GCD(a, b) or gcd(a, b)

**Examples:**
- GCD(12, 18) = 6 (largest number dividing both)
- GCD(7, 13) = 1 (they're relatively prime)
- GCD(100, 25) = 25

**Finding GCD: Method 1 - Listing Factors**

**Step-by-step for GCD(12, 18):**
1. List factors of 12: 1, 2, 3, 4, 6, 12
2. List factors of 18: 1, 2, 3, 6, 9, 18
3. Find common factors: 1, 2, 3, 6
4. Largest common factor: 6
5. Therefore, GCD(12, 18) = 6

**Why This Method is Inefficient:**
For large numbers, listing all factors is time-consuming. The Euclidean algorithm is much faster (O(log min(a, b))).

**Finding GCD: Method 2 - Prime Factorization**

**Step-by-step for GCD(12, 18) using prime factorization:**
1. Factor 12: 12 = 2² × 3
2. Factor 18: 18 = 2 × 3²
3. Take lowest power of each prime:
   - Prime 2: min(2, 1) = 1 → 2¹
   - Prime 3: min(1, 2) = 1 → 3¹
4. Multiply: 2¹ × 3¹ = 6
5. Therefore, GCD(12, 18) = 6

**Why GCD Matters:**
- **Simplifying Fractions:** 12/18 = (12÷6)/(18÷6) = 2/3
- **Cryptography:** Extended Euclidean algorithm finds modular inverses
- **Algorithm Design:** GCD appears in many algorithms
- **Number Theory:** Fundamental concept in mathematics

**Least Common Multiple (LCM): Finding Common Multiples**

The LCM of two numbers is the smallest positive integer that is a multiple of both numbers.

**Notation:** LCM(a, b) or lcm(a, b)

**Examples:**
- LCM(4, 6) = 12 (smallest number divisible by both)
- LCM(7, 13) = 91 (7 × 13, since they're relatively prime)
- LCM(12, 18) = 36

**Finding LCM: Method 1 - Listing Multiples**

**Step-by-step for LCM(4, 6):**
1. Multiples of 4: 4, 8, 12, 16, 20, 24, ...
2. Multiples of 6: 6, 12, 18, 24, 30, ...
3. Common multiples: 12, 24, 36, ...
4. Smallest: 12
5. Therefore, LCM(4, 6) = 12

**Finding LCM: Method 2 - Using GCD**

**Formula:** LCM(a, b) = |a × b| / GCD(a, b)

**Step-by-step for LCM(4, 6):**
1. Find GCD(4, 6) = 2
2. Calculate: LCM(4, 6) = (4 × 6) / 2 = 24 / 2 = 12

**Why This Formula Works:**
- a × b contains all prime factors from both numbers
- GCD contains the common factors
- Dividing removes the "double counting" of common factors
- Result is the product of all unique prime factors at their highest powers

**Finding LCM: Method 3 - Prime Factorization**

**Step-by-step for LCM(12, 18):**
1. Factor 12: 12 = 2² × 3
2. Factor 18: 18 = 2 × 3²
3. Take highest power of each prime:
   - Prime 2: max(2, 1) = 2 → 2²
   - Prime 3: max(1, 2) = 2 → 3²
4. Multiply: 2² × 3² = 4 × 9 = 36
5. Therefore, LCM(12, 18) = 36

**Why LCM Matters:**
- **Adding Fractions:** Need common denominator, which is the LCM
- **Synchronization:** Finding when two periodic events align
- **Algorithm Design:** Some algorithms need to process data in batches of size LCM

**Prime Factorization: The Unique Representation**

Prime factorization expresses a number as a product of prime numbers. The Fundamental Theorem of Arithmetic guarantees this representation is unique (up to order).

**Step-by-Step Factorization: Factor 60**

**Method 1: Trial Division**
1. Start with smallest prime: 2
   - 60 ÷ 2 = 30, so 2 is a factor
   - Continue: 30 ÷ 2 = 15, so 2² is a factor
   - 15 is not divisible by 2, move to next prime
2. Try 3:
   - 15 ÷ 3 = 5, so 3 is a factor
   - 5 is not divisible by 3, move to next prime
3. Try 5:
   - 5 ÷ 5 = 1, so 5 is a factor
   - Result is 1, we're done
4. Result: 60 = 2² × 3 × 5

**Method 2: Factor Tree**

        60
       /  \\
      6    10
     / \\   / \\
    2   3 2   5
Reading the leaves: 2, 2, 3, 5 → 60 = 2² × 3 × 5

**Why Prime Factorization is Powerful:**

**Finding All Divisors:**
If n = p₁^a₁ × p₂^a₂ × ... × pₖ^aₖ, then the number of divisors is (a₁+1)(a₂+1)...(aₖ+1)

**Example:** 60 = 2² × 3¹ × 5¹
- Number of divisors = (2+1)(1+1)(1+1) = 3 × 2 × 2 = 12
- Divisors: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60

**Finding GCD Efficiently:**
- Factor both numbers
- Take minimum power of each prime
- Multiply

**Finding LCM Efficiently:**
- Factor both numbers
- Take maximum power of each prime
- Multiply

**Common Mistakes and Pitfalls:**

**Mistake 1: Thinking 1 is prime**
- 1 is NOT prime by definition
- Including 1 would break unique factorization

**Mistake 2: Not checking up to √n**
- Checking all numbers up to n is inefficient
- Only need to check up to √n

**Mistake 3: Forgetting that 2 is prime**
- 2 is the only even prime
- Many students incorrectly exclude it

**Mistake 4: Confusing GCD and LCM**
- GCD finds what divides both (smaller)
- LCM finds what both divide (larger)
- Remember: GCD ≤ min(a, b), LCM ≥ max(a, b)

**Mistake 5: Incorrectly applying divisibility rules**
- Rule for 6 requires BOTH 2 AND 3
- Rule for 11 uses alternating sum, not regular sum

**Applications in Programming:**

**1. Optimizing Hash Tables:**
Using prime-sized hash tables reduces collisions. Many hash table implementations use prime numbers for table size.

**2. Generating Random Numbers:**
Prime moduli in linear congruential generators produce better pseudorandom sequences.

**3. Cryptography:**
RSA encryption relies on the difficulty of factoring large numbers. Understanding primes helps understand security.

**4. Algorithm Analysis:**
Many algorithms have complexity related to prime factors. Understanding factorization helps analyze performance.

**Next Steps:**

Mastering prime numbers and divisibility provides essential tools for number theory, cryptography, and algorithm design. In later modules, we'll see how these concepts extend to modular arithmetic and advanced cryptographic systems.`,
					CodeExamples: `# Prime number and divisibility examples

def is_prime(n):
    """Check if a number is prime using trial division."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    # Check up to square root of n
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Test prime numbers
print("Prime numbers up to 20:")
for num in range(2, 21):
    if is_prime(num):
        print(f"{num} is prime")

# Sieve of Eratosthenes - efficient algorithm for finding primes
def sieve_of_eratosthenes(limit):
    """Find all primes up to limit."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    return [i for i in range(limit + 1) if is_prime[i]]

primes = sieve_of_eratosthenes(50)
print(f"\nPrimes up to 50: {primes}")

# GCD using Euclidean algorithm (efficient)
def gcd(a, b):
    """Calculate Greatest Common Divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return abs(a)

# LCM
def lcm(a, b):
    """Calculate Least Common Multiple."""
    return abs(a * b) // gcd(a, b)

print(f"\nGCD(12, 18) = {gcd(12, 18)}")
print(f"LCM(4, 6) = {lcm(4, 6)}")

# Prime factorization
def prime_factors(n):
    """Find prime factors of a number."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

print(f"\nPrime factors of 60: {prime_factors(60)}")
print(f"60 = {' × '.join(map(str, prime_factors(60)))}")

# Count divisors using prime factorization
def count_divisors(n):
    """Count number of divisors of n."""
    factors = prime_factors(n)
    from collections import Counter
    factor_count = Counter(factors)
    result = 1
    for count in factor_count.values():
        result *= (count + 1)
    return result

print(f"\nNumber of divisors of 60: {count_divisors(60)}")`,
				},
				{
					Title: "Working with Fractions and Decimals",
					Content: `**Introduction: Precise Representation of Quantities**

Fractions and decimals are two ways to represent numbers that aren't whole. Understanding both forms and how to convert between them is essential for precise calculations, especially in financial applications, scientific computing, and when exact representation matters.

**Understanding Fractions: Parts of a Whole**

A fraction represents a part of a whole, written as a/b where:
- **a** is the numerator (top number) - tells how many parts we have
- **b** is the denominator (bottom number) - tells how many equal parts the whole is divided into
- **Critical:** b ≠ 0 (division by zero is undefined)

**Visual Representation:**
Think of a pizza cut into 8 slices. If you eat 3 slices:
- Numerator: 3 (slices eaten)
- Denominator: 8 (total slices)
- Fraction: 3/8 of the pizza

**Types of Fractions:**

**1. Proper Fractions: Less Than One Whole**

**Definition:** Numerator < Denominator
- Value is always less than 1
- Examples: 1/2, 3/4, 2/5, 7/8

**Why They're Called "Proper":**
They represent a "proper" part of a whole - not exceeding the whole.

**Visual Example:** 3/4 means 3 out of 4 equal parts, which is less than the whole (4/4 = 1).

**2. Improper Fractions: One or More Wholes**

**Definition:** Numerator ≥ Denominator
- Value is greater than or equal to 1
- Examples: 5/4 (1¼), 7/3 (2⅓), 11/2 (5½), 8/8 (exactly 1)

**Why They're Useful:**
Improper fractions are often easier to work with in calculations than mixed numbers, especially in programming where you want a single rational number representation.

**Converting to Mixed Number:**
- Divide numerator by denominator
- Whole number part = quotient
- Fractional part = remainder/denominator
- Example: 7/3 = 2 remainder 1, so 7/3 = 2 1/3

**3. Mixed Numbers: Whole + Fraction**

**Definition:** Combination of a whole number and a proper fraction
- Examples: 1 1/2, 2 3/4, 5 1/3, 10 7/8

**When to Use:**
Mixed numbers are intuitive for humans (we think "2 and a half cups") but computers typically work with improper fractions or decimals.

**Converting Mixed to Improper:**
- Multiply whole number by denominator
- Add numerator
- Keep same denominator
- Example: 2 3/4 = (2×4 + 3)/4 = 11/4

**Operations with Fractions: Step-by-Step Methods**

**1. Addition and Subtraction: Finding Common Ground**

To add or subtract fractions, they must have the same denominator. This is like trying to add apples and oranges - you need to convert to a common unit first.

**Method: Finding the Least Common Denominator (LCD)**

**Step-by-Step: Add 1/3 + 1/2**

1. **Find LCD:** LCM of denominators 3 and 2
   - Multiples of 3: 3, 6, 9, 12, ...
   - Multiples of 2: 2, 4, 6, 8, ...
   - LCD = 6

2. **Convert to equivalent fractions:**
   - 1/3 = ?/6: Multiply numerator and denominator by 2 → 2/6
   - 1/2 = ?/6: Multiply numerator and denominator by 3 → 3/6

3. **Add numerators, keep denominator:**
   - 2/6 + 3/6 = (2 + 3)/6 = 5/6

4. **Simplify if needed:** 5/6 is already in simplest form

**Why LCD (not just any common denominator)?**
Using the LCD keeps numbers smaller and makes simplification easier. You could use 12, 18, or any common multiple, but 6 is most efficient.

**Common Mistake:** Adding numerators and denominators separately:
- Wrong: 1/3 + 1/2 = (1+1)/(3+2) = 2/5
- Correct: 1/3 + 1/2 = 2/6 + 3/6 = 5/6

**2. Multiplication: Straightforward but Watch for Simplification**

Multiplying fractions is simpler than addition - no common denominator needed!

**Rule:** Multiply numerators, multiply denominators

**Step-by-Step: Multiply (2/3) × (3/4)**

1. **Multiply numerators:** 2 × 3 = 6
2. **Multiply denominators:** 3 × 4 = 12
3. **Result:** 6/12
4. **Simplify:** Find GCD(6, 12) = 6
5. **Divide:** (6÷6)/(12÷6) = 1/2

**Shortcut - Cancel Before Multiplying:**
You can simplify before multiplying:
- (2/3) × (3/4) = (2/~~3~~) × (~~3~~/4) = 2/4 = 1/2
- This is more efficient and reduces chance of errors

**Why This Works:**
Multiplying fractions is like saying "take 2/3 of 3/4":
- First, divide into 4 parts (denominator 4)
- Take 3 of those parts (3/4)
- Then take 2/3 of that result
- Final: (2/3) × (3/4) = 6/12 = 1/2

**3. Division: Multiply by the Reciprocal**

Division by a fraction is the same as multiplication by its reciprocal (flipped fraction).

**Why This Works:**
Dividing by a number is the same as multiplying by its inverse. For fractions, the inverse is the reciprocal.

**Step-by-Step: Divide (2/3) ÷ (3/4)**

1. **Find reciprocal of divisor:** Reciprocal of 3/4 is 4/3
2. **Change division to multiplication:** (2/3) ÷ (3/4) = (2/3) × (4/3)
3. **Multiply:** (2 × 4)/(3 × 3) = 8/9
4. **Check if simplification needed:** 8/9 is already simplest

**Visual Understanding:**
"How many times does 3/4 fit into 2/3?"
- 3/4 is larger than 2/3, so answer < 1
- 8/9 ≈ 0.89, which makes sense

**Common Mistake:** Forgetting to flip the second fraction:
- Wrong: (2/3) ÷ (3/4) = (2/3) × (3/4) = 6/12 = 1/2
- Correct: (2/3) ÷ (3/4) = (2/3) × (4/3) = 8/9

**Simplifying Fractions: Reducing to Lowest Terms**

A fraction is in simplest form when the numerator and denominator have no common factors other than 1.

**Method: Divide by GCD**

**Step-by-Step: Simplify 12/18**

1. **Find GCD(12, 18):**
   - Factors of 12: 1, 2, 3, 4, 6, 12
   - Factors of 18: 1, 2, 3, 6, 9, 18
   - GCD = 6

2. **Divide both by GCD:**
   - 12 ÷ 6 = 2
   - 18 ÷ 6 = 3
   - Result: 2/3

3. **Verify:** GCD(2, 3) = 1, so 2/3 is in simplest form

**Why Simplification Matters:**
- Easier to work with smaller numbers
- Standard form for answers
- Easier to compare fractions
- Reduces computational errors

**Decimals: Another Way to Represent Fractions**

Decimals are a convenient way to write fractions with denominators that are powers of 10 (10, 100, 1000, ...).

**Understanding Decimal Place Value:**

In the number 123.456:
- 1 is in the hundreds place (1 × 100)
- 2 is in the tens place (2 × 10)
- 3 is in the ones place (3 × 1)
- 4 is in the tenths place (4 × 1/10 = 4/10)
- 5 is in the hundredths place (5 × 1/100 = 5/100)
- 6 is in the thousandths place (6 × 1/1000 = 6/1000)

**Types of Decimals:**

**1. Terminating Decimals: Finite Representation**

Terminating decimals end after a finite number of digits.

**Examples:**
- 0.5 = 5/10 = 1/2
- 0.25 = 25/100 = 1/4
- 0.75 = 75/100 = 3/4
- 0.125 = 125/1000 = 1/8

**When Do Fractions Produce Terminating Decimals?**
A fraction produces a terminating decimal if and only if the denominator (in lowest terms) has no prime factors other than 2 and/or 5.

**Why?** Because 10 = 2 × 5, so any fraction with denominator having only factors 2 and 5 can be written with denominator 10ⁿ.

**Examples:**
- 1/2 = 0.5 (denominator is 2)
- 1/4 = 0.25 (denominator is 2²)
- 1/5 = 0.2 (denominator is 5)
- 1/8 = 0.125 (denominator is 2³)

**2. Repeating Decimals: Infinite but Predictable**

Repeating decimals have a pattern that repeats infinitely.

**Notation:**
- 0.333... = 0.3̄ (bar over repeating digit)
- 0.1666... = 0.16̄ (bar over repeating part)
- 0.142857142857... = 0.142857̄ (bar over repeating block)

**Examples:**
- 1/3 = 0.333...
- 1/6 = 0.1666...
- 1/7 = 0.142857142857...
- 2/3 = 0.666...

**When Do Fractions Produce Repeating Decimals?**
Any fraction that doesn't produce a terminating decimal will produce a repeating decimal. The repeating block length is at most (denominator - 1).

**Converting Repeating Decimals to Fractions:**

**Method: Algebraic Approach**

**Example: Convert 0.333... to fraction**

1. Let x = 0.333...
2. Multiply by 10: 10x = 3.333...
3. Subtract original: 10x - x = 3.333... - 0.333...
4. Simplify: 9x = 3
5. Solve: x = 3/9 = 1/3

**More Complex Example: Convert 0.142857142857...**

1. Let x = 0.142857142857...
2. Multiply by 1,000,000 (length of repeating block is 6):
   1,000,000x = 142857.142857142857...
3. Subtract: 1,000,000x - x = 142857
4. Simplify: 999,999x = 142857
5. Solve: x = 142857/999999
6. Simplify: Find GCD and reduce (this equals 1/7)

**Converting Between Forms:**

**Fraction to Decimal: Long Division**

**Step-by-Step: Convert 3/8 to decimal**

1. Set up: 3 ÷ 8
2. 8 doesn't go into 3, so add decimal: 3.0 ÷ 8
3. 8 goes into 30 three times: 0.3
4. Remainder: 30 - 24 = 6
5. Bring down 0: 60 ÷ 8 = 7 remainder 4
6. Continue: 0.37...
7. Final: 3/8 = 0.375

**Decimal to Fraction: Terminating**

**Step-by-Step: Convert 0.75 to fraction**

1. Write as fraction with power of 10: 75/100
2. Simplify: Find GCD(75, 100) = 25
3. Divide: (75÷25)/(100÷25) = 3/4
4. Verify: 3/4 = 0.75 ✓

**Decimal to Fraction: Repeating**

Use the algebraic method shown above for repeating decimals.

**Why Fractions vs Decimals Matter in Programming:**

**Floating-Point Precision Issues:**

Computers represent decimals using floating-point, which has limited precision. This causes rounding errors:

**Example in Python:**

    0.1 + 0.2  # Results in 0.30000000000000004, not exactly 0.3!

**Why This Happens:**
- 0.1 in binary is a repeating decimal: 0.0001100110011...
- Computers can't store infinite digits
- Rounding errors accumulate

**Solution: Use Fractions for Exact Arithmetic**

Python's fractions.Fraction module provides exact arithmetic:

    from fractions import Fraction
    Fraction(1, 10) + Fraction(2, 10)  # Exactly Fraction(3, 10)

**When to Use Each:**

**Use Fractions When:**
- Exact representation is critical (money, measurements)
- You need to avoid rounding errors
- Working with ratios or probabilities

**Use Decimals When:**
- Displaying to users (more intuitive)
- Working with measurements that are inherently approximate
- Performance is critical (decimals are faster)

**Common Mistakes:**

**Mistake 1: Adding fractions without common denominator**
- Wrong: 1/3 + 1/2 = 2/5
- Correct: 1/3 + 1/2 = 2/6 + 3/6 = 5/6

**Mistake 2: Dividing fractions without flipping**
- Wrong: (2/3) ÷ (3/4) = (2/3) × (3/4) = 6/12
- Correct: (2/3) ÷ (3/4) = (2/3) × (4/3) = 8/9

**Mistake 3: Not simplifying final answers**
- Always reduce fractions to lowest terms
- Check if numerator and denominator have common factors

**Mistake 4: Confusing terminating and repeating decimals**
- 1/3 = 0.333... (repeating), not 0.3 (terminating)
- Check denominator's prime factors to determine type

**Mistake 5: Floating-point equality comparisons**
- Never use == with floats: 0.1 + 0.2 == 0.3 is False!
- Use tolerance: abs(0.1 + 0.2 - 0.3) < 1e-10
- Or use fractions for exact comparisons

**Applications in Computer Science:**

**1. Financial Calculations:**
- Interest rates, percentages, currency conversions
- Must use exact fractions or Decimal type to avoid rounding errors
- Example: Calculating compound interest requires precise fractions

**2. Probability and Statistics:**
- Probabilities are fractions (number of favorable outcomes / total outcomes)
- Ratios and proportions use fractions
- Example: P(event) = favorable/total

**3. Graphics and Scaling:**
- Aspect ratios are fractions (width/height)
- Scaling transformations use fractional multipliers
- Example: 16:9 aspect ratio = 16/9

**4. Algorithm Design:**
- Some algorithms work with ratios
- Performance metrics often expressed as fractions
- Example: Cache hit rate = hits/(hits + misses)

**Next Steps:**

Mastering fractions and decimals provides the foundation for working with rational numbers, percentages, and precise calculations. In the next lesson, we'll explore exponents and roots, which extend our number system further.`,
					CodeExamples: `# Working with fractions and decimals in Python

from fractions import Fraction
import decimal

# Creating fractions
f1 = Fraction(1, 3)
f2 = Fraction(1, 2)
print(f"Fraction 1: {f1}")
print(f"Fraction 2: {f2}")

# Fraction operations
print(f"\n{f1} + {f2} = {f1 + f2}")
print(f"{f1} - {f2} = {f1 - f2}")
print(f"{f1} * {f2} = {f1 * f2}")
print(f"{f1} / {f2} = {f1 / f2}")

# Converting to decimal
print(f"\n{f1} as decimal: {float(f1)}")
print(f"{f2} as decimal: {float(f2)}")

# Converting decimal to fraction
decimal_val = 0.75
fraction_from_decimal = Fraction(decimal_val).limit_denominator()
print(f"\n{decimal_val} as fraction: {fraction_from_decimal}")

# Simplifying fractions
f3 = Fraction(12, 18)
print(f"\n{f3} simplified: {f3}")  # Automatically simplified to 2/3

# Mixed numbers
def to_mixed_number(fraction):
    """Convert improper fraction to mixed number."""
    whole = fraction.numerator // fraction.denominator
    remainder = fraction.numerator % fraction.denominator
    if whole == 0:
        return str(fraction)
    if remainder == 0:
        return str(whole)
    return f"{whole} {Fraction(remainder, fraction.denominator)}"

print(f"\n{Fraction(7, 3)} as mixed number: {to_mixed_number(Fraction(7, 3))}")

# Comparing fractions
f4 = Fraction(2, 3)
f5 = Fraction(3, 4)
print(f"\n{f4} < {f5}: {f4 < f5}")
print(f"{f4} > {f5}: {f4 > f5}")

# Decimal precision
decimal.getcontext().prec = 10
d1 = decimal.Decimal('0.1')
d2 = decimal.Decimal('0.2')
print(f"\nUsing Decimal for precision:")
print(f"{d1} + {d2} = {d1 + d2}")  # Exact: 0.3
print(f"Float: 0.1 + 0.2 = {0.1 + 0.2}")  # May have rounding error`,
				},
				{
					Title: "Exponents and Roots",
					Content: `**Introduction: Repeated Multiplication and Its Inverse**

Exponents represent repeated multiplication, while roots are the inverse operation. Together, they form a powerful system for expressing large numbers, analyzing algorithm complexity, and working with exponential growth. Understanding exponents and roots is crucial for computer science, from analyzing algorithm performance to cryptographic operations.

**Understanding Exponents: Repeated Multiplication**

An exponent tells us how many times to multiply a number (the base) by itself.

**Notation:** aⁿ where:
- **a** is the base (the number being multiplied)
- **n** is the exponent or power (how many times to multiply)
- Read as "a to the power of n" or "a raised to the nth power"

**Basic Examples:**
- 2³ = 2 × 2 × 2 = 8 (multiply 2 by itself 3 times)
- 5² = 5 × 5 = 25 (multiply 5 by itself 2 times)
- 10⁴ = 10 × 10 × 10 × 10 = 10,000 (multiply 10 by itself 4 times)
- 3⁵ = 3 × 3 × 3 × 3 × 3 = 243

**Visual Understanding:**
Think of exponents as "how many layers deep" the multiplication goes:
- 2¹ = 2 (one layer: just 2)
- 2² = 2 × 2 = 4 (two layers)
- 2³ = 2 × 2 × 2 = 8 (three layers)
- 2⁴ = 2 × 2 × 2 × 2 = 16 (four layers)

**Why Exponents Matter: Exponential Growth**

Exponential growth is one of the most important concepts in computer science:
- **Algorithm Complexity:** Some algorithms have exponential time complexity O(2ⁿ)
- **Data Structures:** Binary trees have 2ⁿ possible paths
- **Cryptography:** Security relies on exponential difficulty
- **Binary Representation:** n bits can represent 2ⁿ different values

**The Power of Exponential Growth:**
- 2¹⁰ = 1,024 (about 1K)
- 2²⁰ = 1,048,576 (about 1M)
- 2³⁰ = 1,073,741,824 (about 1B)
- 2⁶⁴ = 18,446,744,073,709,551,616 (huge!)

This is why exponential algorithms become impractical quickly!

**Exponent Rules: The Laws of Exponents**

These rules allow us to manipulate exponential expressions efficiently.

**1. Product Rule: Adding Exponents**

**Rule:** aᵐ × aⁿ = aᵐ⁺ⁿ

**Why This Works:**
When multiplying powers with the same base, we're combining the multiplications:
- aᵐ = a × a × ... × a (m times)
- aⁿ = a × a × ... × a (n times)
- aᵐ × aⁿ = (a × a × ... × a) × (a × a × ... × a) = a × a × ... × a (m+n times) = aᵐ⁺ⁿ

**Step-by-Step Example: 2³ × 2²**

1. Expand: 2³ = 2 × 2 × 2, 2² = 2 × 2
2. Multiply: (2 × 2 × 2) × (2 × 2) = 2 × 2 × 2 × 2 × 2 = 2⁵
3. Using rule: 2³ × 2² = 2³⁺² = 2⁵ = 32
4. Verify: 8 × 4 = 32 ✓

**Common Mistake:** Only works when bases are the same:
- Correct: 2³ × 2² = 2⁵
- Wrong: 2³ × 3² ≠ 5⁵ (bases are different!)

**2. Quotient Rule: Subtracting Exponents**

**Rule:** aᵐ ÷ aⁿ = aᵐ⁻ⁿ (for a ≠ 0)

**Why This Works:**
When dividing powers with the same base, we cancel common factors:
- aᵐ ÷ aⁿ = (a × a × ... × a) / (a × a × ... × a)
- Cancel n factors from numerator and denominator
- Remaining: a × a × ... × a (m-n times) = aᵐ⁻ⁿ

**Step-by-Step Example: 2⁵ ÷ 2²**

1. Expand: 2⁵ = 2 × 2 × 2 × 2 × 2, 2² = 2 × 2
2. Divide: (2 × 2 × 2 × 2 × 2) / (2 × 2) = 2 × 2 × 2 = 2³
3. Using rule: 2⁵ ÷ 2² = 2⁵⁻² = 2³ = 8
4. Verify: 32 ÷ 4 = 8 ✓

**Special Case: When m = n**
- aⁿ ÷ aⁿ = aⁿ⁻ⁿ = a⁰ = 1
- This gives us the zero exponent rule!

**3. Power Rule: Multiplying Exponents**

**Rule:** (aᵐ)ⁿ = aᵐⁿ

**Why This Works:**
Raising a power to a power means multiplying the base m times, then doing that n times:
- (aᵐ)ⁿ = aᵐ × aᵐ × ... × aᵐ (n times)
- = (a × a × ... × a) × (a × a × ... × a) × ... (n groups of m a's)
- = a × a × ... × a (m×n times) = aᵐⁿ

**Step-by-Step Example: (2³)²**

1. Expand inner: 2³ = 2 × 2 × 2 = 8
2. Raise to power: (2³)² = 8² = 8 × 8 = 64
3. Using rule: (2³)² = 2³ˣ² = 2⁶ = 64
4. Verify: 2⁶ = 64 ✓

**Common Mistake:** Don't confuse with product rule:
- (2³)² = 2⁶ (power rule: multiply exponents)
- 2³ × 2² = 2⁵ (product rule: add exponents)

**4. Zero Exponent: The Identity**

**Rule:** a⁰ = 1 (for a ≠ 0)

**Why This Makes Sense:**
Using the quotient rule: aⁿ ÷ aⁿ = aⁿ⁻ⁿ = a⁰
But we also know: aⁿ ÷ aⁿ = 1 (any number divided by itself is 1)
Therefore: a⁰ = 1

**Examples:**
- 5⁰ = 1
- 100⁰ = 1
- (-3)⁰ = 1
- **Exception:** 0⁰ is undefined (leads to contradictions)

**Why This Matters:**
- Base cases in recursive algorithms often use a⁰ = 1
- Polynomials: x⁰ = 1 represents the constant term
- Binary: 2⁰ = 1 represents the rightmost bit

**5. Negative Exponent: Reciprocal**

**Rule:** a⁻ⁿ = 1/aⁿ (for a ≠ 0)

**Why This Makes Sense:**
Using the quotient rule: a⁰ ÷ aⁿ = a⁰⁻ⁿ = a⁻ⁿ
But a⁰ = 1, so: 1 ÷ aⁿ = a⁻ⁿ
And 1 ÷ aⁿ = 1/aⁿ
Therefore: a⁻ⁿ = 1/aⁿ

**Step-by-Step Example: 2⁻³**

1. Apply rule: 2⁻³ = 1/2³
2. Calculate: 1/2³ = 1/(2 × 2 × 2) = 1/8
3. Verify: 2⁻³ × 2³ = 2⁻³⁺³ = 2⁰ = 1 ✓

**Why Negative Exponents Matter:**
- Representing very small numbers (scientific notation)
- Probability: P(not event) = 1 - P(event) uses reciprocal concepts
- Algorithm analysis: Some algorithms have complexity like O(n⁻¹) = O(1/n)

**6. Fractional Exponent: Roots**

**Rule:** a^(1/n) = ⁿ√a (the nth root of a)

**Why This Makes Sense:**
If (a^(1/n))ⁿ = a^(1/n × n) = a¹ = a
Then a^(1/n) must be the number that, when raised to the nth power, equals a
This is exactly the definition of the nth root!

**Examples:**
- 8^(1/3) = ∛8 = 2 (because 2³ = 8)
- 16^(1/4) = ⁴√16 = 2 (because 2⁴ = 16)
- 25^(1/2) = √25 = 5 (because 5² = 25)

**General Form: a^(m/n) = (ⁿ√a)ᵐ = ⁿ√(aᵐ)**
- Example: 8^(2/3) = (∛8)² = 2² = 4
- Or: 8^(2/3) = ∛(8²) = ∛64 = 4

**Understanding Roots: The Inverse of Exponents**

Roots answer the question: "What number, when raised to a certain power, gives me this value?"

**Square Root (√): Second Root**

**Definition:** √a is the number that, when squared, equals a
- √16 = 4 because 4² = 16
- √25 = 5 because 5² = 25
- √9 = 3 because 3² = 9

**Important Notes:**
- √a is always non-negative (by convention)
- For positive a, there are two square roots: +√a and -√a
- Example: Both 4 and -4 satisfy x² = 16, but √16 = 4 (positive root)

**Perfect Squares:**
Numbers whose square roots are integers:
- 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, ...
- These are 1², 2², 3², 4², 5², ...

**Cube Root (∛): Third Root**

**Definition:** ∛a is the number that, when cubed, equals a
- ∛8 = 2 because 2³ = 8
- ∛27 = 3 because 3³ = 27
- ∛64 = 4 because 4³ = 64

**Difference from Square Root:**
- Cube roots can be negative: ∛(-8) = -2 because (-2)³ = -8
- This is because cubing preserves the sign

**nth Root: General Case**

**Definition:** ⁿ√a is the number that, when raised to the nth power, equals a
- ⁴√16 = 2 because 2⁴ = 16
- ⁵√32 = 2 because 2⁵ = 32

**Properties of Roots:**

**1. Product Property:** √(a × b) = √a × √b

**Why This Works:**
If √a × √b = x, then x² = (√a × √b)² = (√a)² × (√b)² = a × b
So x = √(a × b)

**Example:** √(4 × 9) = √36 = 6
And: √4 × √9 = 2 × 3 = 6 ✓

**2. Quotient Property:** √(a/b) = √a / √b (for b ≠ 0)

**Example:** √(16/4) = √4 = 2
And: √16 / √4 = 4 / 2 = 2 ✓

**3. Power Property:** (ⁿ√a)ⁿ = a

This is the definition of nth root - raising it to the nth power gives back the original number.

**Simplifying Radicals:**

**Method: Factor Out Perfect Powers**

**Step-by-Step: Simplify √72**

1. Factor 72: 72 = 36 × 2 = 6² × 2
2. Apply product property: √72 = √(6² × 2) = √6² × √2 = 6√2
3. Check: (6√2)² = 36 × 2 = 72 ✓

**Why This Matters:**
- Simplified radicals are easier to work with
- Standard form for answers
- Makes addition/subtraction of radicals possible

**Common Mistakes:**

**Mistake 1: Confusing product and sum rules**
- Wrong: √(a + b) = √a + √b
- Correct: √(a × b) = √a × √b
- Example: √(9 + 16) = √25 = 5, but √9 + √16 = 3 + 4 = 7 (not equal!)

**Mistake 2: Forgetting that √a² = |a|, not a**
- √((-3)²) = √9 = 3, not -3
- The square root function always returns the non-negative value

**Mistake 3: Incorrectly applying power rule**
- Wrong: (a + b)ⁿ = aⁿ + bⁿ
- Correct: (a × b)ⁿ = aⁿ × bⁿ
- Example: (2 + 3)² = 5² = 25, but 2² + 3² = 4 + 9 = 13 (not equal!)

**Mistake 4: Confusing negative exponents with negative bases**
- (-2)³ = -8 (negative base, odd exponent = negative)
- -2³ = -(2³) = -8 (negative of positive power)
- But: (-2)² = 4, while -2² = -4 (different!)

**Applications in Computer Science:**

**1. Algorithm Complexity Analysis:**

**Time Complexities:**
- O(2ⁿ): Exponential (very slow, avoid if possible)
- O(n²): Quadratic (acceptable for small n)
- O(n log n): Linearithmic (efficient sorting)
- O(√n): Sublinear (very efficient, e.g., checking primes up to √n)

**Example:** Checking if n is prime:
- Naive: Check all numbers 2 to n-1 → O(n)
- Optimized: Check up to √n → O(√n) (much faster!)

**2. Binary Representation:**

**n bits can represent 2ⁿ values:**
- 1 bit: 2¹ = 2 values (0, 1)
- 8 bits: 2⁸ = 256 values (0-255)
- 32 bits: 2³² = 4,294,967,296 values
- 64 bits: 2⁶⁴ values (huge!)

**3. Cryptography - Modular Exponentiation:**

**RSA Encryption uses:**
- c = mᵉ mod n (encryption)
- m = cᵈ mod n (decryption)
- Security relies on difficulty of computing d from e and n

**Efficient Computation:**
- Naive: Multiply m by itself e times → O(e) multiplications
- Fast exponentiation: Use binary representation of e → O(log e) multiplications
- Critical for practical cryptography!

**4. Data Structures:**

**Binary Trees:**
- Height h tree has at most 2ʰ nodes
- Full binary tree with n nodes has height ≈ log₂(n)
- This logarithmic relationship is fundamental to tree operations

**Hash Tables:**
- Table size often chosen as prime or power of 2
- Power of 2 allows efficient modulo: x mod 2ⁿ = x & (2ⁿ - 1) (bitwise AND)

**5. Scientific Notation:**

**Representing Large/Small Numbers:**
- 6.022 × 10²³ (Avogadro's number)
- 1.602 × 10⁻¹⁹ (electron charge)
- Exponents make huge/tiny numbers manageable

**Next Steps:**

Mastering exponents and roots provides essential tools for understanding algorithm complexity, working with binary systems, and analyzing exponential growth. In the next lesson, we'll explore scientific notation and significant figures, which use exponents to represent precise measurements.`,
					CodeExamples: `# Exponents and roots in Python

import math

# Basic exponentiation
print("Basic exponents:")
print(f"2^3 = {2**3}")
print(f"5^2 = {5**2}")
print(f"10^4 = {10**4}")

# Using pow() function
print(f"\nUsing pow():")
print(f"pow(2, 3) = {pow(2, 3)}")
print(f"pow(2, 3, 5) = {pow(2, 3, 5)}")  # Modular exponentiation: 2^3 mod 5

# Square roots
print(f"\nSquare roots:")
print(f"sqrt(16) = {math.sqrt(16)}")
print(f"sqrt(25) = {math.sqrt(25)}")
print(f"16^(1/2) = {16**(1/2)}")

# Cube roots
print(f"\nCube roots:")
print(f"8^(1/3) = {8**(1/3)}")
print(f"27^(1/3) = {27**(1/3)}")

# nth root
def nth_root(x, n):
    """Calculate nth root of x."""
    return x**(1/n)

print(f"\nnth roots:")
print(f"16^(1/4) = {nth_root(16, 4)}")
print(f"32^(1/5) = {nth_root(32, 5)}")

# Exponent rules demonstration
a, m, n = 2, 3, 2
print(f"\nExponent rules (a={a}, m={m}, n={n}):")
print(f"Product: {a}^{m} × {a}^{n} = {a**(m+n)}")
print(f"Quotient: {a}^{m+n} ÷ {a}^{n} = {a**m}")
print(f"Power: ({a}^{m})^{n} = {a**(m*n)}")

# Negative exponents
print(f"\nNegative exponents:")
print(f"2^(-3) = {2**(-3)}")
print(f"5^(-2) = {5**(-2)}")

# Zero exponent
print(f"\nZero exponent:")
print(f"5^0 = {5**0}")
print(f"100^0 = {100**0}")

# Large exponents (scientific notation)
print(f"\nLarge exponents:")
print(f"2^10 = {2**10}")
print(f"2^20 = {2**20}")
print(f"2^30 = {2**30}")  # About 1 billion`,
				},
				{
					Title: "Scientific Notation and Significant Figures",
					Content: `**Introduction: Managing Extreme Numbers**

Scientific notation is a powerful way to express very large or very small numbers using powers of 10. Combined with significant figures, it allows us to communicate both magnitude and precision. This is essential in science, engineering, and computer science, where we deal with numbers ranging from subatomic scales to astronomical distances.

**Understanding Scientific Notation: A Compact Representation**

Scientific notation expresses numbers as a product of two parts:
- A coefficient (between 1 and 10, including 1 but excluding 10)
- A power of 10

**Format:** a × 10ⁿ where:
- **a** is the coefficient (1 ≤ a < 10)
- **n** is an integer exponent
- The × symbol is often omitted in scientific contexts: a × 10ⁿ or aE+n

**Why This Format?**
- Standardized: Always one digit before the decimal point
- Easy to compare: Compare exponents first, then coefficients
- Efficient: Compact representation of huge/tiny numbers
- Precise: Maintains all significant digits

**Examples:**
- 3,000,000 = 3 × 10⁶ (3 million)
- 0.000004 = 4 × 10⁻⁶ (4 millionths)
- 1,234,567 = 1.234567 × 10⁶ (preserves all digits)
- 6.022 × 10²³ (Avogadro's number - atoms per mole)
- 1.602 × 10⁻¹⁹ (electron charge in coulombs)

**Converting to Scientific Notation: Step-by-Step**

**Method for Large Numbers (≥ 1):**

**Step-by-Step: Convert 5,200 to scientific notation**

1. **Identify the number:** 5,200
2. **Place decimal:** 5,200.0 (implicit decimal at end)
3. **Move decimal left** until coefficient is between 1 and 10:
   - 5,200.0 → 520.0 → 52.0 → 5.2
   - Moved 3 places left
4. **Determine exponent:** Positive (moved left) = +3
5. **Write result:** 5.2 × 10³
6. **Verify:** 5.2 × 10³ = 5.2 × 1,000 = 5,200 ✓

**Method for Small Numbers (< 1):**

**Step-by-Step: Convert 0.00052 to scientific notation**

1. **Identify the number:** 0.00052
2. **Move decimal right** until coefficient is between 1 and 10:
   - 0.00052 → 0.0052 → 0.052 → 0.52 → 5.2
   - Moved 4 places right
3. **Determine exponent:** Negative (moved right) = -4
4. **Write result:** 5.2 × 10⁻⁴
5. **Verify:** 5.2 × 10⁻⁴ = 5.2 × 0.0001 = 0.00052 ✓

**Memory Aid:**
- **Left = Large:** Moving decimal left means large number → positive exponent
- **Right = Really small:** Moving decimal right means small number → negative exponent

**Operations with Scientific Notation: Efficient Calculations**

**1. Multiplication: Combine Coefficients and Exponents**

**Rule:** (a × 10ᵐ) × (b × 10ⁿ) = (a × b) × 10ᵐ⁺ⁿ

**Why This Works:**
- Multiply coefficients: a × b
- Add exponents: 10ᵐ × 10ⁿ = 10ᵐ⁺ⁿ
- Combine: (a × b) × 10ᵐ⁺ⁿ

**Step-by-Step: (2 × 10³) × (3 × 10⁴)**

1. Multiply coefficients: 2 × 3 = 6
2. Add exponents: 3 + 4 = 7
3. Combine: 6 × 10⁷
4. Verify: 2,000 × 30,000 = 60,000,000 = 6 × 10⁷ ✓

**Adjusting the Result:**
Sometimes the coefficient product exceeds 10, requiring adjustment:

**Example: (4 × 10²) × (3 × 10³)**
1. Multiply: 4 × 3 = 12 (exceeds 10!)
2. Adjust: 12 = 1.2 × 10¹
3. Add exponents: 2 + 3 + 1 = 6
4. Result: 1.2 × 10⁶

**2. Division: Divide Coefficients, Subtract Exponents**

**Rule:** (a × 10ᵐ) ÷ (b × 10ⁿ) = (a ÷ b) × 10ᵐ⁻ⁿ

**Step-by-Step: (6 × 10⁷) ÷ (2 × 10³)**

1. Divide coefficients: 6 ÷ 2 = 3
2. Subtract exponents: 7 - 3 = 4
3. Result: 3 × 10⁴
4. Verify: 60,000,000 ÷ 2,000 = 30,000 = 3 × 10⁴ ✓

**3. Addition and Subtraction: Align Exponents First**

**Rule:** Convert to same exponent, then add/subtract coefficients

**Step-by-Step: (3 × 10⁴) + (2 × 10³)**

1. Align exponents (use larger): Convert 2 × 10³ to 0.2 × 10⁴
2. Add coefficients: 3 + 0.2 = 3.2
3. Result: 3.2 × 10⁴
4. Verify: 30,000 + 2,000 = 32,000 = 3.2 × 10⁴ ✓

**Why Scientific Notation Matters in Programming:**

**1. Floating-Point Representation:**
Computers use scientific notation internally:
- IEEE 754 format: sign × mantissa × 2^exponent
- Similar concept, but base 2 instead of 10
- Understanding scientific notation helps understand floating-point precision

**2. Large Number Handling:**
- Representing very large integers (BigInt libraries)
- Avoiding overflow in calculations
- Comparing magnitudes efficiently

**3. Performance Metrics:**
- Algorithm complexity: O(10ⁿ) operations
- Data sizes: 10⁶ bytes = 1 MB, 10⁹ bytes = 1 GB
- Network speeds: 10⁶ bits/sec = 1 Mbps

**Significant Figures: Communicating Precision**

Significant figures indicate how precisely a number is known. They tell us which digits are reliable and which are uncertain.

**Why Significant Figures Matter:**
- Measurements have inherent uncertainty
- Calculations can't create precision that wasn't there
- Important for scientific communication and error analysis

**Rules for Counting Significant Figures:**

**1. All Non-Zero Digits Are Significant:**
- 123 has 3 significant figures (all digits matter)
- 4567 has 4 significant figures
- 9 has 1 significant figure

**2. Zeros Between Non-Zero Digits Are Significant:**
- 1005 has 4 significant figures (zeros are "trapped")
- 2003 has 4 significant figures
- 50.07 has 4 significant figures

**3. Leading Zeros Are NOT Significant:**
- 0.005 has 1 significant figure (the 5)
- 0.00045 has 2 significant figures (the 4 and 5)
- 0.00123 has 3 significant figures (the 1, 2, 3)

**Why?** Leading zeros are just placeholders, not measured values.

**4. Trailing Zeros After Decimal ARE Significant:**
- 5.00 has 3 significant figures (all zeros matter)
- 12.50 has 4 significant figures
- 0.100 has 3 significant figures

**Why?** These zeros indicate precision - the measurement was made to that decimal place.

**5. Trailing Zeros Before Decimal Are Ambiguous:**
- 100 could be 1, 2, or 3 significant figures
- Context matters: Is it exactly 100 or approximately 100?
- Use scientific notation to clarify: 1.0 × 10² (2 sig figs) or 1.00 × 10² (3 sig figs)

**Examples with Explanations:**

- **123.45:** 5 significant figures (all digits are non-zero)
- **0.0045:** 2 significant figures (leading zeros don't count, only 4 and 5)
- **100.0:** 4 significant figures (trailing zero after decimal counts)
- **100:** Ambiguous (could be 1, 2, or 3 significant figures)
- **1.00 × 10²:** 3 significant figures (scientific notation clarifies)
- **0.0500:** 3 significant figures (leading zero doesn't count, trailing zeros after decimal do)

**Operations with Significant Figures:**

**Multiplication and Division:**
Result has same number of significant figures as the factor with fewest significant figures.

**Example: 2.34 × 1.2**
- 2.34 has 3 sig figs, 1.2 has 2 sig figs
- Calculate: 2.34 × 1.2 = 2.808
- Round to 2 sig figs: 2.8
- Result: 2.8 (2 significant figures)

**Addition and Subtraction:**
Result is rounded to the least precise decimal place.

**Example: 12.34 + 0.5**
- 12.34 is precise to hundredths, 0.5 is precise to tenths
- Calculate: 12.34 + 0.5 = 12.84
- Round to least precise (tenths): 12.8
- Result: 12.8

**Common Mistakes:**

**Mistake 1: Counting all zeros as significant**
- Wrong: 100 has 3 significant figures (not always!)
- Correct: Context-dependent, use scientific notation to clarify

**Mistake 2: Forgetting trailing zeros after decimal**
- Wrong: 5.00 has 1 significant figure
- Correct: 5.00 has 3 significant figures (zeros indicate precision)

**Mistake 3: Not adjusting coefficient after multiplication**
- Wrong: (4 × 10²) × (3 × 10³) = 12 × 10⁵
- Correct: (4 × 10²) × (3 × 10³) = 1.2 × 10⁶ (coefficient must be < 10)

**Mistake 4: Incorrectly aligning exponents in addition**
- Wrong: (3 × 10⁴) + (2 × 10³) = 5 × 10⁷
- Correct: Convert first: (3 × 10⁴) + (0.2 × 10⁴) = 3.2 × 10⁴

**Applications in Computer Science:**

**1. Floating-Point Precision:**
- Understanding IEEE 754 helps avoid precision errors
- Significant figures concept applies to mantissa precision
- Example: float has ~7 decimal digits of precision

**2. Error Analysis:**
- Rounding errors accumulate in calculations
- Significant figures help estimate error bounds
- Critical for numerical algorithms

**3. Data Representation:**
- File sizes: 1.5 × 10⁶ bytes = 1.5 MB
- Memory addresses: 2³² possible addresses
- Network speeds: 1.0 × 10⁹ bits/sec = 1 Gbps

**4. Algorithm Analysis:**
- Expressing large input sizes: n = 10⁶ elements
- Comparing complexities: O(10ⁿ) vs O(n²)
- Performance metrics: 10³ operations per second

**Next Steps:**

Mastering scientific notation and significant figures provides essential tools for working with extreme numbers and understanding precision. In Module 2, we'll apply these concepts to algebra, where we'll work with variables and solve equations systematically.`,
					CodeExamples: `# Scientific notation and precision in Python

# Scientific notation
print("Scientific notation:")
print(f"3,000,000 = {3e6}")
print(f"0.000004 = {4e-6}")
print(f"1,234,567 = {1.234567e6}")

# Formatting in scientific notation
num = 1234567
print(f"\n{num} in scientific notation: {num:.2e}")

# Operations with scientific notation
a = 2e3  # 2 × 10^3
b = 3e4  # 3 × 10^4
print(f"\n{a} × {b} = {a * b}")
print(f"{a * b:.2e}")  # Formatted

# Floating point precision
print(f"\nFloating point precision:")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3: {0.1 + 0.2 == 0.3}")

# Using decimal for exact precision
from decimal import Decimal
d1 = Decimal('0.1')
d2 = Decimal('0.2')
print(f"\nUsing Decimal:")
print(f"{d1} + {d2} = {d1 + d2}")
print(f"{d1} + {d2} == Decimal('0.3'): {d1 + d2 == Decimal('0.3')}")

# Significant figures approximation
def round_to_sig_figs(num, sig_figs):
    """Round number to specified significant figures."""
    if num == 0:
        return 0
    magnitude = math.floor(math.log10(abs(num)))
    return round(num, sig_figs - magnitude - 1)

print(f"\nRounding to significant figures:")
print(f"123.456 to 3 sig figs: {round_to_sig_figs(123.456, 3)}")
print(f"0.004567 to 2 sig figs: {round_to_sig_figs(0.004567, 2)}")

# Large number handling
large_num = 10**20
print(f"\nLarge numbers:")
print(f"10^20 = {large_num}")
print(f"10^20 in scientific: {large_num:.2e}")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          501,
			Title:       "Algebra Fundamentals",
			Description: "Master variables, equations, and algebraic expressions.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Variables and Expressions",
					Content: `**Introduction: The Language of Algebra**

Variables and expressions form the foundation of algebra, allowing us to work with unknown quantities and create general formulas. Understanding how to work with variables is essential for solving equations, writing functions in programming, and modeling real-world situations mathematically.

**Understanding Variables: Symbols for Unknowns**

A variable is a symbol (usually a letter) that represents an unknown or changing value. Think of it as a placeholder that can take on different values.

**Why Variables Matter:**
- **Generalization:** Express patterns that work for many different values
- **Problem Solving:** Represent unknown quantities we're trying to find
- **Modeling:** Describe relationships between quantities
- **Programming:** Variables in code work similarly - they store values that can change

**Common Variable Names:**
- **x, y, z:** Most common in mathematics
- **a, b, c:** Often used for constants or parameters
- **i, j, k:** Frequently used for indices (especially in programming)
- **n, m:** Often represent counts or sizes

**Key Properties:**
- Variables can represent any number (unless restricted)
- Value can change depending on context
- Same variable name in same context represents same value
- Different variables can have different values

**Variables vs Constants:**
- **Variable:** Value can change (x, y, temperature)
- **Constant:** Value stays the same (π, e, speed of light)
- **Parameter:** Constant in one context, variable in another (like 'a' in ax + b)

**Algebraic Expressions: Building Mathematical Phrases**

An algebraic expression is a combination of variables, numbers, and operations. Unlike equations, expressions don't have an equals sign - they're mathematical phrases, not sentences.

**Components of Expressions:**
- **Terms:** Parts separated by + or - (e.g., in 3x + 5, terms are 3x and 5)
- **Coefficients:** Numbers multiplying variables (e.g., 3 is coefficient of x in 3x)
- **Constants:** Numbers without variables (e.g., 5 in 3x + 5)
- **Variables:** Letters representing unknown values

**Types of Expressions:**

**1. Linear Expressions:**
- Highest power of variable is 1
- Examples: 3x + 5, 2x - 7, -x + 3
- Graph as straight lines
- General form: ax + b

**2. Quadratic Expressions:**
- Highest power of variable is 2
- Examples: x² + 2x + 1, 3x² - 5x + 2
- Graph as parabolas
- General form: ax² + bx + c

**3. Polynomial Expressions:**
- Sum of terms with non-negative integer powers
- Examples: x³ + 2x² - x + 5, 4x⁴ - 3x² + 7
- Can have any degree (highest power)

**4. Expressions with Multiple Variables:**
- Contain more than one variable
- Examples: 2x + 3y - 7, x² + xy + y²
- Represent relationships between quantities

**Evaluating Expressions: Substituting Values**

To evaluate an expression means to substitute specific values for variables and calculate the result.

**Step-by-Step Process:**

**Example: Evaluate 3x + 5 when x = 4**

1. **Identify the expression:** 3x + 5
2. **Substitute the value:** Replace x with 4 → 3(4) + 5
3. **Follow order of operations:**
   - Multiplication first: 3 × 4 = 12
   - Then addition: 12 + 5 = 17
4. **Result:** 17

**More Complex Example: Evaluate x² + 2x + 1 when x = 3**

1. **Substitute:** (3)² + 2(3) + 1
2. **Exponents first:** 9 + 2(3) + 1
3. **Multiplication:** 9 + 6 + 1
4. **Addition:** 16
5. **Result:** 16

**Multiple Variables Example: Evaluate 2x + 3y - 7 when x = 5, y = 3**

1. **Substitute both values:** 2(5) + 3(3) - 7
2. **Multiply:** 10 + 9 - 7
3. **Add/subtract left to right:** 19 - 7 = 12
4. **Result:** 12

**Common Mistake:** Forgetting to use parentheses when substituting:
- Wrong: 3x + 5, x = -2 → 3-2 + 5 (confusing!)
- Correct: 3(-2) + 5 = -6 + 5 = -1

**Simplifying Expressions: Making Them Cleaner**

Simplifying means rewriting an expression in an equivalent but simpler form. This involves combining like terms and using properties of operations.

**Like Terms: The Key to Simplification**

Like terms are terms that have the same variable(s) raised to the same power(s). Only like terms can be combined.

**Identifying Like Terms:**

**Like Terms (can combine):**
- 3x and 5x (both have x to power 1)
- 2x² and -7x² (both have x²)
- 3xy and 5xy (both have xy)
- 4 and 7 (both are constants)

**NOT Like Terms (cannot combine):**
- 3x and 3x² (different powers)
- 2x and 2y (different variables)
- 3xy and 3x (different variable combinations)
- 5x and 5 (variable vs constant)

**Why This Matters:**
You can only combine terms that represent the same "type" of quantity. Adding 3 apples and 5 apples makes sense (8 apples), but adding 3 apples and 5 oranges doesn't give you a meaningful result without more context.

**Combining Like Terms: Step-by-Step**

**Example: Simplify 3x + 5x - 2x**

1. **Identify like terms:** All terms have x to power 1
2. **Group coefficients:** (3 + 5 - 2)x
3. **Calculate:** 6x
4. **Result:** 6x

**More Complex Example: Simplify 2x² + 3x + x² - x + 5**

1. **Group like terms:**
   - x² terms: 2x² + x² = (2 + 1)x² = 3x²
   - x terms: 3x - x = (3 - 1)x = 2x
   - Constants: 5
2. **Combine:** 3x² + 2x + 5
3. **Result:** 3x² + 2x + 5

**Using Distributive Property:**

**Example: Simplify 3(x + 2) - 2(x - 1)**

1. **Distribute:** 3x + 6 - 2x + 2
   - Note: -2(x - 1) = -2x + 2 (distribute the negative!)
2. **Combine like terms:**
   - x terms: 3x - 2x = x
   - Constants: 6 + 2 = 8
3. **Result:** x + 8

**Common Mistakes:**

**Mistake 1: Combining unlike terms**
- Wrong: 3x + 2x² = 5x³
- Correct: 3x + 2x² cannot be simplified further (unlike terms)

**Mistake 2: Incorrectly distributing negatives**
- Wrong: -(x - 3) = -x - 3
- Correct: -(x - 3) = -x + 3 (distribute the negative to both terms)

**Mistake 3: Forgetting coefficients of 1**
- Wrong: x + 3x = 3x
- Correct: x + 3x = 1x + 3x = 4x (x has coefficient 1)

**Mistake 4: Not following order of operations**
- Wrong: 2x + 3 × x = 5x × x = 5x²
- Correct: 2x + 3x = 5x (multiplication before addition, but here we're combining like terms)

**Applications in Programming:**

**1. Function Definitions:**
Expressions become functions:

    def linear_function(x):
        return 3*x + 5  # Expression 3x + 5

**2. Variable Assignment:**
Variables store values:

    x = 4
    result = 3*x + 5  # Evaluating expression

**3. Expression Evaluation:**
Expressions are evaluated with specific values:

    def evaluate_quadratic(x):
        return x**2 + 2*x + 1  # x² + 2x + 1

**4. Simplification in Code:**
Some systems can simplify expressions symbolically (like SymPy in Python), but often we evaluate directly.

**Why This Matters for Problem Solving:**

**1. General Solutions:**
Expressions let us write formulas that work for any value:
- Area of rectangle: A = lw (works for any length and width)
- Distance formula: d = rt (works for any rate and time)

**2. Pattern Recognition:**
Expressions help us see patterns:
- 2, 4, 6, 8, ... can be written as 2n (where n = 1, 2, 3, ...)
- This pattern recognition is crucial in algorithm design

**3. Modeling Relationships:**
Expressions model how quantities relate:
- Cost = 5x + 10 (x items cost $5 each, plus $10 fee)
- This helps in optimization problems

**Next Steps:**

Mastering variables and expressions provides the foundation for solving equations. In the next lesson, we'll learn how to solve linear equations, which involves manipulating expressions to find the value of variables that make equations true.`,
					CodeExamples: `# Working with variables and expressions in Python

# Variables in programming
x = 5
y = 3

# Evaluating expressions
expression1 = 3 * x + 5
print(f"If x = {x}, then 3x + 5 = {expression1}")

expression2 = x**2 + 2*x + 1
print(f"If x = {x}, then x² + 2x + 1 = {expression2}")

# Expression with multiple variables
expression3 = 2*x + 3*y - 7
print(f"If x = {x} and y = {y}, then 2x + 3y - 7 = {expression3}")

# Function to evaluate expression for different values
def evaluate_expression(x_val):
    """Evaluate 3x + 5 for given x."""
    return 3 * x_val + 5

# Test with different values
for x_val in [0, 1, 2, 3, 4, 5]:
    result = evaluate_expression(x_val)
    print(f"3({x_val}) + 5 = {result}")

# Simplifying expressions (combining like terms)
# 3x + 5x - 2x = 6x
coefficient = 3 + 5 - 2
print(f"\n3x + 5x - 2x = {coefficient}x")`,
				},
				{
					Title: "Solving Linear Equations",
					Content: `**Introduction: Finding the Unknown**

Solving linear equations is one of the most fundamental skills in algebra. A linear equation is a statement that two expressions are equal, where the highest power of the variable is 1. Learning to solve these equations systematically is essential for problem-solving in mathematics, science, and computer science.

**Understanding Linear Equations**

A linear equation is an equation where the highest power of the variable is 1. When graphed, these equations form straight lines, hence the name "linear."

**General Form:** ax + b = c
- **a** is the coefficient of x (a ≠ 0)
- **b** is a constant term
- **c** is another constant
- **x** is the variable we're solving for

**Other Forms:**
- Standard form: ax + b = 0
- Slope-intercept form: y = mx + b (when graphed)
- Point-slope form: y - y₁ = m(x - x₁)

**What Makes an Equation "Linear":**
- No variables raised to powers other than 1
- No products of variables (like xy)
- No variables in denominators (like 1/x)
- Graph is a straight line

**Examples of Linear Equations:**
- 3x + 5 = 17 ✓
- 2x - 7 = 0 ✓
- x = 4 ✓
- x² + 3 = 7 ✗ (not linear - has x²)
- 1/x = 5 ✗ (not linear - variable in denominator)

**The Goal: Isolate the Variable**

To solve a linear equation means to find the value(s) of the variable that make the equation true. We do this by isolating the variable on one side of the equation.

**Fundamental Principle: Whatever you do to one side, do to the other**

This maintains equality - if two things are equal, and you do the same thing to both, they remain equal.

**Solving Steps: A Systematic Approach**

**Step 1: Simplify Both Sides**
- Clear parentheses using distributive property
- Combine like terms on each side
- Clear fractions (if any)

**Step 2: Move Variable Terms to One Side**
- Add or subtract terms to get all variable terms on one side
- Add or subtract terms to get all constants on the other side

**Step 3: Isolate the Variable**
- Multiply or divide to get the variable alone
- The coefficient should become 1

**Step 4: Check Your Answer**
- Substitute the solution back into the original equation
- Verify both sides are equal

**Detailed Example 1: Simple Case**

**Solve: 3x + 5 = 17**

**Step-by-Step Solution:**

1. **Identify the equation:** 3x + 5 = 17
   - Variable term: 3x
   - Constant on left: 5
   - Constant on right: 17

2. **Isolate variable term (move constant):**
   - Subtract 5 from both sides: 3x + 5 - 5 = 17 - 5
   - Simplify: 3x = 12
   - Why this works: We're undoing the addition of 5

3. **Isolate variable (remove coefficient):**
   - Divide both sides by 3: 3x ÷ 3 = 12 ÷ 3
   - Simplify: x = 4
   - Why this works: We're undoing the multiplication by 3

4. **Check the solution:**
   - Substitute x = 4: 3(4) + 5 = 12 + 5 = 17
   - Left side equals right side ✓
   - Solution is correct!

**Detailed Example 2: Variables on Both Sides**

**Solve: 2x - 7 = 3x + 1**

**Step-by-Step Solution:**

1. **Identify the equation:** 2x - 7 = 3x + 1
   - Variable terms on both sides!
   - Need to move all variables to one side

2. **Move variable terms to one side:**
   - Subtract 2x from both sides: 2x - 7 - 2x = 3x + 1 - 2x
   - Simplify: -7 = x + 1
   - Why subtract 2x? To eliminate it from the left side

3. **Move constants to the other side:**
   - Subtract 1 from both sides: -7 - 1 = x + 1 - 1
   - Simplify: -8 = x
   - Or equivalently: x = -8

4. **Check the solution:**
   - Substitute x = -8:
     - Left: 2(-8) - 7 = -16 - 7 = -23
     - Right: 3(-8) + 1 = -24 + 1 = -23
   - Both sides equal -23 ✓
   - Solution is correct!

**Why This Method Works: Inverse Operations**

We use inverse operations to "undo" what's been done to the variable:
- **Addition ↔ Subtraction:** If something is added, subtract it
- **Multiplication ↔ Division:** If something is multiplied, divide by it
- **Power ↔ Root:** If something is raised to a power, take the root

**Word Problems: Translating Language to Equations**

Word problems require translating real-world situations into mathematical equations. This is a crucial skill for modeling and problem-solving.

**Strategy for Word Problems:**

1. **Read carefully:** Understand what's being asked
2. **Identify unknowns:** What are you trying to find?
3. **Choose variables:** Let a variable represent the unknown
4. **Translate words to math:**
   - "is" or "equals" → =
   - "increased by" or "more than" → +
   - "decreased by" or "less than" → -
   - "times" or "of" → ×
   - "divided by" → ÷
5. **Write the equation**
6. **Solve the equation**
7. **Check:** Does the answer make sense in context?

**Example: "A number increased by 5 equals 12"**

1. **Identify unknown:** "a number" → let x = the number
2. **Translate:**
   - "increased by 5" → x + 5
   - "equals 12" → = 12
3. **Write equation:** x + 5 = 12
4. **Solve:** x = 7
5. **Check:** 7 + 5 = 12 ✓
6. **Answer:** The number is 7

**More Complex Word Problem Example:**

**Problem:** "Three times a number decreased by 7 equals 14. Find the number."

1. **Let x = the number**
2. **Translate:**
   - "Three times a number" → 3x
   - "decreased by 7" → 3x - 7
   - "equals 14" → = 14
3. **Equation:** 3x - 7 = 14
4. **Solve:**
   - Add 7: 3x = 21
   - Divide by 3: x = 7
5. **Check:** 3(7) - 7 = 21 - 7 = 14 ✓
6. **Answer:** The number is 7

**Checking Solutions: Verifying Your Work**

Always check your solution by substituting it back into the original equation. This catches errors and builds confidence.

**Why Checking Matters:**
- Catches arithmetic mistakes
- Verifies the solution is correct
- Helps identify extraneous solutions (in more advanced problems)
- Builds good mathematical habits

**Checking Process:**

1. **Write the original equation**
2. **Substitute your solution** for the variable
3. **Simplify both sides**
4. **Verify equality:** Both sides should be equal

**Example: Check x = 4 for 3x + 5 = 17**

1. Original: 3x + 5 = 17
2. Substitute: 3(4) + 5 = 17
3. Simplify: 12 + 5 = 17
4. Result: 17 = 17 ✓
5. Solution is correct!

**Equations with Fractions: Clearing Denominators**

When equations contain fractions, it's often easier to clear them first by multiplying both sides by the least common denominator (LCD).

**Why Clear Fractions?**
- Eliminates fractions, making arithmetic easier
- Reduces chance of errors
- Standard approach in algebra

**Step-by-Step: Solve (x/3) + 2 = 5**

1. **Identify LCD:** Denominators are 3 and 1, so LCD = 3
2. **Multiply both sides by 3:**
   - 3[(x/3) + 2] = 3(5)
   - Distribute: 3(x/3) + 3(2) = 15
   - Simplify: x + 6 = 15
3. **Solve:** x = 9
4. **Check:** (9/3) + 2 = 3 + 2 = 5 ✓

**More Complex Fraction Example: Solve (x/2) - (x/3) = 1**

1. **Find LCD:** Denominators are 2 and 3, so LCD = 6
2. **Multiply by 6:**
   - 6[(x/2) - (x/3)] = 6(1)
   - Distribute: 6(x/2) - 6(x/3) = 6
   - Simplify: 3x - 2x = 6
   - Combine: x = 6
3. **Check:** (6/2) - (6/3) = 3 - 2 = 1 ✓

**Common Mistakes:**

**Mistake 1: Not performing the same operation on both sides**
- Wrong: 3x + 5 = 17, so 3x = 12 (forgot to subtract 5 from right side)
- Correct: 3x + 5 = 17, so 3x + 5 - 5 = 17 - 5, giving 3x = 12

**Mistake 2: Incorrectly handling negative numbers**
- Wrong: -x = 5, so x = 5
- Correct: -x = 5, multiply by -1: x = -5

**Mistake 3: Not checking the solution**
- Always verify! Many errors can be caught by checking.

**Mistake 4: Forgetting to distribute when clearing fractions**
- Wrong: 3(x/3 + 2) = 3(5), so x/3 + 2 = 5 (didn't distribute)
- Correct: 3(x/3 + 2) = 3(5), so x + 6 = 15 (distributed correctly)

**Mistake 5: Dividing by zero**
- Never divide by zero! If the coefficient becomes 0, check if the equation has no solution or infinitely many solutions.

**Special Cases:**

**No Solution:**
- Example: 3x + 5 = 3x + 7
- Subtract 3x: 5 = 7 (false statement)
- This equation has no solution

**Infinitely Many Solutions:**
- Example: 3x + 5 = 3x + 5
- Subtract 3x: 5 = 5 (always true)
- This equation is true for all values of x

**Applications in Computer Science:**

**1. Algorithm Analysis:**
- Setting up equations to find break-even points
- Example: When is O(n log n) faster than O(n²)? Solve: n log n = n²

**2. Optimization Problems:**
- Finding maximum/minimum values
- Cost-benefit analysis
- Resource allocation

**3. System Design:**
- Capacity planning equations
- Performance modeling
- Load balancing calculations

**4. Data Structures:**
- Hash table collision resolution
- Tree balancing equations
- Array indexing formulas

**Why This Matters:**

Linear equations are everywhere:
- **Cost calculations:** Fixed costs + variable costs per unit
- **Distance/rate/time:** d = rt
- **Optimization:** Finding optimal values
- **Modeling:** Representing real-world relationships

**Next Steps:**

Mastering linear equations provides the foundation for solving more complex equations (quadratic, systems) and understanding linear relationships in data. In the next lesson, we'll explore inequalities, which extend these concepts to comparisons rather than equalities.`,
					CodeExamples: `# Solving linear equations in Python

def solve_linear_equation(a, b, c):
    """
    Solve ax + b = c
    Returns x = (c - b) / a
    """
    if a == 0:
        if b == c:
            return "Infinite solutions"
        else:
            return "No solution"
    return (c - b) / a

# Example 1: 3x + 5 = 17
x1 = solve_linear_equation(3, 5, 17)
print(f"3x + 5 = 17 → x = {x1}")

# Verify solution
verify1 = 3 * x1 + 5
print(f"Check: 3({x1}) + 5 = {verify1}")

# Example 2: 2x - 7 = 3x + 1
x2 = solve_linear_equation(-1, -8, 0)
print(f"\n2x - 7 = 3x + 1 → x = {x2}")

# Word problem: "A number increased by 5 equals 12"
x3 = solve_linear_equation(1, 5, 12)
print(f"\nWord problem: x + 5 = 12 → x = {x3}")

# General solver for equations in form: ax + b = cx + d
def solve_equation(a, b, c, d):
    """
    Solve ax + b = cx + d
    Rearranges to: (a - c)x = d - b
    """
    coeff = a - c
    const = d - b
    if coeff == 0:
        if const == 0:
            return "Infinite solutions"
        else:
            return "No solution"
    return const / coeff

# Test: 2x - 7 = 3x + 1
x4 = solve_equation(2, -7, 3, 1)
print(f"\n2x - 7 = 3x + 1 → x = {x4}")`,
				},
				{
					Title: "Inequalities",
					Content: `**Introduction: Comparing Quantities**

Inequalities extend the concept of equations to comparisons. Instead of asking "when are two things equal?" we ask "when is one thing greater or less than another?" This is fundamental in optimization, algorithm analysis, and constraint satisfaction problems.

**Understanding Inequality Symbols**

Inequalities use symbols to compare two expressions:

- **<** less than: The left side is smaller than the right side
- **>** greater than: The left side is larger than the right side
- **≤** less than or equal to: The left side is smaller than or equal to the right side
- **≥** greater than or equal to: The left side is larger than or equal to the right side
- **≠** not equal to: The two sides are different (not an inequality in the strict sense, but a comparison)

**Key Difference from Equations:**
- Equations: Two expressions are equal (x = 5 means x is exactly 5)
- Inequalities: One expression is compared to another (x < 5 means x can be any number less than 5)

**Why This Matters:**
Inequalities describe ranges and constraints, which are essential in:
- Optimization: Finding maximum/minimum values subject to constraints
- Algorithm analysis: Describing complexity bounds (O(n²) means "at most" n² operations)
- Range queries: Finding all values in a certain range
- Boundary conditions: Checking if values are within acceptable limits

**Solving Linear Inequalities: The Critical Rule**

Solving inequalities is similar to solving equations, with one crucial difference: **when you multiply or divide by a negative number, you must flip the inequality sign.**

**Why Flip the Sign?**

Consider: 2 < 5 (true)
- Multiply by -1: -2 < -5 (false!)
- Correct: -2 > -5 (true)

When we multiply by a negative number, we're reflecting across zero on the number line, which reverses the order. The number that was smaller becomes larger, and vice versa.

**Visual Understanding:**
- Positive side: 2 < 5
- Multiply by -1: Reflect across zero
- Negative side: -2 > -5 (order is reversed)

**Step-by-Step: Solve 3x + 5 < 17**

1. **Isolate variable term:**
   - Subtract 5 from both sides: 3x + 5 - 5 < 17 - 5
   - Simplify: 3x < 12

2. **Isolate variable:**
   - Divide by 3 (positive number, no flip): 3x ÷ 3 < 12 ÷ 3
   - Simplify: x < 4

3. **Solution:** All numbers less than 4 satisfy the inequality

4. **Check:** Test a value less than 4 (e.g., x = 3):
   - 3(3) + 5 = 9 + 5 = 14 < 17 ✓
   - Test a value greater than 4 (e.g., x = 5):
   - 3(5) + 5 = 15 + 5 = 20 < 17 ✗ (as expected)

**Step-by-Step: Solve -2x > 8**

1. **Isolate variable:**
   - Divide by -2 (negative number, must flip!): -2x ÷ (-2) < 8 ÷ (-2)
   - Simplify: x < -4
   - Note: > became < because we divided by negative

2. **Solution:** All numbers less than -4 satisfy the inequality

3. **Check:** Test x = -5 (less than -4):
   - -2(-5) = 10 > 8 ✓
   - Test x = -3 (greater than -4):
   - -2(-3) = 6 > 8 ✗ (as expected)

**Step-by-Step: Solve 2x - 7 ≤ 3x + 1**

1. **Move variable terms to one side:**
   - Subtract 2x from both sides: 2x - 7 - 2x ≤ 3x + 1 - 2x
   - Simplify: -7 ≤ x + 1

2. **Move constants to the other side:**
   - Subtract 1 from both sides: -7 - 1 ≤ x + 1 - 1
   - Simplify: -8 ≤ x
   - Or equivalently: x ≥ -8

3. **Solution:** All numbers greater than or equal to -8 satisfy the inequality

4. **Check:** Test x = -8 (boundary):
   - Left: 2(-8) - 7 = -16 - 7 = -23
   - Right: 3(-8) + 1 = -24 + 1 = -23
   - -23 ≤ -23 ✓
   - Test x = -7 (greater than -8):
   - Left: 2(-7) - 7 = -21
   - Right: 3(-7) + 1 = -20
   - -21 ≤ -20 ✓

**Common Mistakes:**

**Mistake 1: Forgetting to flip when multiplying/dividing by negative**
- Wrong: -2x > 8, so x > -4
- Correct: -2x > 8, so x < -4 (flipped because divided by negative)

**Mistake 2: Flipping when adding/subtracting**
- Wrong: x + 5 < 10, so x > 5 (no flip needed for addition!)
- Correct: x + 5 < 10, so x < 5 (no flip for addition/subtraction)

**Mistake 3: Confusing < and ≤**
- < means strictly less than (not equal)
- ≤ means less than or equal to (can be equal)
- This matters for boundary values!

**Graphing Inequalities: Visual Representation**

Graphing helps visualize the solution set (all values that satisfy the inequality).

**On a Number Line:**

**Open Circle (○):** Used for < or >
- Indicates the endpoint is NOT included
- Example: x < 4 → open circle at 4, shade left

**Closed Circle (●):** Used for ≤ or ≥
- Indicates the endpoint IS included
- Example: x ≥ -8 → closed circle at -8, shade right

**Shading:**
- Shade the region containing all solutions
- For x < 4: Shade everything to the left of 4
- For x ≥ -8: Shade everything to the right of -8 (including -8)

**Example: Graph x < 4**
Number line: ... -2 -1  0  1  2  3  4  5  6 ...
                        (open circle, shade left)

**Example: Graph x ≥ -8**
Number line: ... -10 -9 -8 -7 -6 -5 ...
                    (closed circle, shade right)

**Compound Inequalities: Multiple Conditions**

Compound inequalities combine two or more inequalities using "and" or "or."

**"And" (Intersection): Both Must Be True**

When two inequalities are joined by "and," both conditions must be satisfied simultaneously. The solution is the intersection (overlap) of the individual solution sets.

**Example: -2 < x < 5**

This means: x > -2 AND x < 5
- x must be greater than -2
- x must be less than 5
- Solution: All numbers between -2 and 5 (not including endpoints)

**Graph:**

Number line: ... -3 -2 -1  0  1  2  3  4  5  6 ...
                    (open circles, shade between)

**Step-by-Step: Solve -3 ≤ 2x + 1 < 7**

1. **Split into two inequalities:**
   - -3 ≤ 2x + 1 AND 2x + 1 < 7

2. **Solve first inequality:**
   - -3 ≤ 2x + 1
   - Subtract 1: -4 ≤ 2x
   - Divide by 2: -2 ≤ x

3. **Solve second inequality:**
   - 2x + 1 < 7
   - Subtract 1: 2x < 6
   - Divide by 2: x < 3

4. **Combine:** -2 ≤ x < 3
   - x is between -2 (inclusive) and 3 (exclusive)

**"Or" (Union): At Least One Must Be True**

When two inequalities are joined by "or," at least one condition must be satisfied. The solution is the union (combination) of the individual solution sets.

**Example: x < -2 OR x > 5**

This means: Either x is less than -2, OR x is greater than 5 (or both, though both can't happen simultaneously)

**Graph:**

Number line: ... -4 -3 -2 -1  0  1  2  3  4  5  6  7 ...
                (shade left of -2 and right of 5)

**Step-by-Step: Solve x + 3 < 1 OR 2x - 5 > 7**

1. **Solve first inequality:**
   - x + 3 < 1
   - Subtract 3: x < -2

2. **Solve second inequality:**
   - 2x - 5 > 7
   - Add 5: 2x > 12
   - Divide by 2: x > 6

3. **Combine with OR:** x < -2 OR x > 6
   - Solution includes all numbers less than -2 and all numbers greater than 6

**Applications in Computer Science:**

**1. Algorithm Complexity Analysis:**
- Big O notation uses inequalities: O(n²) means "at most" cn² operations for some constant c
- Example: If algorithm takes ≤ 3n² + 5n operations, we say it's O(n²)

**2. Optimization Problems:**
- Constraints are often inequalities
- Example: Maximize profit subject to: x ≥ 0, y ≥ 0, x + y ≤ 100
- Linear programming uses systems of inequalities

**3. Range Queries:**
- Finding all values in a range: a ≤ x ≤ b
- Used in database queries, search algorithms
- Example: Find all users with age between 18 and 65

**4. Boundary Conditions:**
- Checking if values are within acceptable limits
- Input validation: 0 < x < 100
- Array bounds checking: 0 ≤ index < length

**5. Conditional Logic:**
- Programming conditions often use inequalities
- Example: if (x < 10 && x > 0) { ... }
- Understanding inequalities helps write correct conditions

**Why This Matters:**

Inequalities are everywhere in computer science:
- **Algorithm analysis:** Describing performance bounds
- **Optimization:** Constraint satisfaction
- **Data structures:** Range queries, bounds checking
- **System design:** Capacity limits, resource constraints

**Next Steps:**

Mastering inequalities provides essential tools for optimization, algorithm analysis, and constraint problems. In the next lesson, we'll explore systems of linear equations, which involve finding values that satisfy multiple equations simultaneously.`,
					CodeExamples: `# Working with inequalities in Python

def solve_inequality(a, b, c, strict=True):
    """
    Solve ax + b < c (or <= if strict=False)
    Returns the solution and whether to flip sign
    """
    if a == 0:
        return "No solution" if b >= c else "All real numbers"
    
    if a < 0:
        # Need to flip inequality
        solution = (c - b) / a
        return (solution, True)  # True means flip
    else:
        solution = (c - b) / a
        return (solution, False)

# Example 1: 3x + 5 < 17
result1 = solve_inequality(3, 5, 17)
print(f"3x + 5 < 17 → x < {result1[0]}")

# Example 2: -2x > 8 (flip sign)
result2 = solve_inequality(-2, 0, 8)
print(f"-2x > 8 → x < {result2[0]}")

# Checking if a value satisfies inequality
def check_inequality(x, a, b, c, strict=True):
    """Check if x satisfies ax + b < c"""
    left_side = a * x + b
    if strict:
        return left_side < c
    else:
        return left_side <= c

# Test values
test_values = [-5, -4, -3, 0, 3, 4, 5]
print(f"\nChecking 3x + 5 < 17:")
for val in test_values:
    satisfies = check_inequality(val, 3, 5, 17)
    print(f"x = {val}: {val} < 4? {satisfies}")

# Compound inequality: -2 < x < 5
def in_range(x, lower, upper, include_lower=True, include_upper=True):
    """Check if x is in range [lower, upper] or (lower, upper)"""
    if include_lower and include_upper:
        return lower <= x <= upper
    elif include_lower:
        return lower <= x < upper
    elif include_upper:
        return lower < x <= upper
    else:
        return lower < x < upper

print(f"\nCompound inequality -2 < x < 5:")
for val in [-3, -2, 0, 4, 5, 6]:
    result = in_range(val, -2, 5, include_lower=False, include_upper=False)
    print(f"x = {val}: {result}")`,
				},
				{
					Title: "Systems of Linear Equations",
					Content: `**Introduction: Multiple Constraints, One Solution**

A system of linear equations consists of two or more linear equations with the same variables. The solution to a system is the set of values that satisfy ALL equations simultaneously. This is fundamental in modeling real-world problems with multiple constraints, optimization, and many areas of computer science.

**Understanding Systems**

A system of equations represents multiple conditions that must all be true at the same time. Each equation is a constraint, and the solution is the point (or points) where all constraints are satisfied.

**Example System:**
- Equation 1: 2x + y = 8
- Equation 2: x - y = 1

**Geometric Interpretation:**
- Each equation represents a line in 2D space
- The solution is the point where the lines intersect
- In 3D, equations represent planes, and solutions are intersection points/lines

**Solution:** The ordered pair (x, y) that makes both equations true simultaneously.

**Verifying a Solution:**
To check if (3, 2) is a solution:
- Equation 1: 2(3) + 2 = 6 + 2 = 8 ✓
- Equation 2: 3 - 2 = 1 ✓
- Both are true, so (3, 2) is the solution!

**Methods for Solving Systems**

There are several methods, each with advantages in different situations.

**Method 1: Substitution - Express and Replace**

The substitution method works by solving one equation for one variable, then substituting that expression into the other equation.

**When to Use:**
- One equation is easily solved for one variable
- Coefficients are 1 or -1
- Avoids dealing with fractions

**Step-by-Step: Solve the system:**
- 2x + y = 8
- x - y = 1

**Step 1: Solve one equation for one variable**
- From equation 2: x - y = 1
- Add y to both sides: x = y + 1
- Now x is expressed in terms of y

**Step 2: Substitute into the other equation**
- Replace x in equation 1: 2(y + 1) + y = 8
- Distribute: 2y + 2 + y = 8
- Combine like terms: 3y + 2 = 8

**Step 3: Solve for the remaining variable**
- Subtract 2: 3y = 6
- Divide by 3: y = 2

**Step 4: Back-substitute to find the first variable**
- Use y = 2 in x = y + 1
- x = 2 + 1 = 3

**Step 5: Verify the solution**
- Check (3, 2) in both equations:
  - 2(3) + 2 = 8 ✓
  - 3 - 2 = 1 ✓

**Solution: (3, 2)**

**Method 2: Elimination - Add to Cancel**

The elimination method works by adding or subtracting equations to eliminate one variable, then solving for the remaining variable.

**When to Use:**
- Coefficients are already similar
- Avoids fractions in substitution
- Works well with larger systems

**Step-by-Step: Solve the system:**
- 2x + y = 8
- x - y = 1

**Step 1: Align equations**

    2x + y = 8
     x - y = 1

**Step 2: Add equations to eliminate a variable**
- Notice: y and -y will cancel when added
- Add: (2x + y) + (x - y) = 8 + 1
- Simplify: 3x = 9
- Solve: x = 3

**Step 3: Substitute back to find the other variable**
- Use x = 3 in equation 2: 3 - y = 1
- Solve: -y = -2, so y = 2

**Step 4: Verify**
- Check (3, 2) in both equations ✓

**Solution: (3, 2)**

**When Coefficients Don't Match: Multiply First**

**Example: Solve the system:**
- 3x + 2y = 7
- 2x - y = 1

**Step 1: Make coefficients match**
- Multiply equation 2 by 2: 4x - 2y = 2
- Now we have: 3x + 2y = 7 and 4x - 2y = 2

**Step 2: Add to eliminate y**
- (3x + 2y) + (4x - 2y) = 7 + 2
- 7x = 9
- x = 9/7

**Step 3: Substitute back**
- Use x = 9/7 in 2x - y = 1
- 2(9/7) - y = 1
- 18/7 - y = 1
- -y = 1 - 18/7 = -11/7
- y = 11/7

**Solution: (9/7, 11/7)**

**Types of Solutions**

Systems can have three types of solutions, depending on how the lines (or planes) relate to each other.

**1. One Solution: Unique Intersection**

The lines intersect at exactly one point. This is the most common case.

**Characteristics:**
- Lines have different slopes
- Exactly one ordered pair satisfies both equations
- System is called "consistent and independent"

**Example:**
- x + y = 5
- x - y = 1
- Solution: (3, 2) - unique point of intersection

**2. No Solution: Parallel Lines**

The lines are parallel and never intersect. No ordered pair satisfies both equations.

**Characteristics:**
- Lines have the same slope but different y-intercepts
- System is called "inconsistent"
- When solving, you get a false statement (like 0 = 5)

**Example:**
- x + y = 1
- x + y = 2
- These are parallel lines (both have slope -1)
- If we subtract: (x + y) - (x + y) = 1 - 2
- Result: 0 = -1 (false statement)
- No solution exists!

**3. Infinite Solutions: Coincident Lines**

The lines are the same (one equation is a multiple of the other). Every point on the line satisfies both equations.

**Characteristics:**
- Lines have the same slope AND same y-intercept
- System is called "consistent and dependent"
- When solving, you get an identity (like 0 = 0)

**Example:**
- x + y = 1
- 2x + 2y = 2 (this is 2 times the first equation)
- If we subtract 2 times equation 1 from equation 2:
  - (2x + 2y) - 2(x + y) = 2 - 2(1)
  - 0 = 0 (always true)
- Infinite solutions: Any point (x, 1-x) works!

**Applications in Computer Science:**

**1. Resource Allocation:**
- Allocating limited resources to multiple tasks
- Example: Allocate CPU and memory to processes subject to constraints

**2. Network Flow:**
- Modeling flow through networks
- Finding maximum flow, shortest paths
- Used in routing algorithms

**3. Linear Programming:**
- Optimization problems with linear constraints
- Finding maximum/minimum of linear objective function
- Used in operations research, scheduling

**4. Computer Graphics:**
- Solving systems to find intersection points
- Ray tracing: Finding where rays intersect surfaces
- Collision detection: Finding intersection of objects

**5. Machine Learning:**
- Linear regression: Finding best-fit line through data points
- Solving normal equations in least squares
- Gradient descent (iterative method for systems)

**6. Cryptography:**
- Some cryptographic systems use linear algebra
- Solving systems modulo a prime (modular arithmetic)

**Why This Matters:**

Systems of equations are everywhere:
- **Optimization:** Finding best solutions under constraints
- **Modeling:** Representing complex relationships
- **Algorithms:** Many algorithms solve systems implicitly
- **Graphics:** Computing intersections and transformations

**Next Steps:**

Mastering systems of linear equations provides essential tools for optimization, modeling, and solving complex problems. In the next lesson, we'll explore polynomials and factoring, which extend our algebraic toolkit to higher-degree expressions.`,
					CodeExamples: `# Solving systems of linear equations

import numpy as np

# Method 1: Using substitution
def solve_by_substitution(eq1, eq2):
    """
    Solve system: a1*x + b1*y = c1, a2*x + b2*y = c2
    Returns (x, y) or None if no solution
    """
    a1, b1, c1 = eq1
    a2, b2, c2 = eq2
    
    # Solve equation 2 for x: x = (c2 - b2*y) / a2
    # But we need to handle cases where a2 = 0
    if a2 == 0:
        # Solve for y first
        if b2 == 0:
            return None  # Invalid equation
        y = c2 / b2
        if a1 == 0:
            return None
        x = (c1 - b1 * y) / a1
        return (x, y)
    
    # Solve equation 2 for x
    # x = (c2 - b2*y) / a2
    # Substitute into equation 1
    # a1*(c2 - b2*y)/a2 + b1*y = c1
    # Multiply by a2: a1*(c2 - b2*y) + b1*a2*y = c1*a2
    # a1*c2 - a1*b2*y + b1*a2*y = c1*a2
    # y*(b1*a2 - a1*b2) = c1*a2 - a1*c2
    
    denominator = b1 * a2 - a1 * b2
    if denominator == 0:
        return None  # No unique solution
    
    y = (c1 * a2 - a1 * c2) / denominator
    x = (c2 - b2 * y) / a2
    return (x, y)

# Example: 2x + y = 8, x - y = 1
eq1 = (2, 1, 8)  # 2x + y = 8
eq2 = (1, -1, 1)  # x - y = 1
solution = solve_by_substitution(eq1, eq2)
print(f"System: 2x + y = 8, x - y = 1")
print(f"Solution: x = {solution[0]}, y = {solution[1]}")

# Verify
x, y = solution
print(f"Check: 2({x}) + {y} = {2*x + y}")
print(f"Check: {x} - {y} = {x - y}")

# Method 2: Using matrix (Gaussian elimination)
def solve_by_matrix(eq1, eq2):
    """Solve using matrix operations."""
    a1, b1, c1 = eq1
    a2, b2, c2 = eq2
    
    # Create coefficient matrix and constant vector
    A = np.array([[a1, b1], [a2, b2]])
    b = np.array([c1, c2])
    
    try:
        solution = np.linalg.solve(A, b)
        return tuple(solution)
    except np.linalg.LinAlgError:
        return None  # No unique solution

solution2 = solve_by_matrix(eq1, eq2)
print(f"\nMatrix method solution: {solution2}")

# System with no solution (parallel lines)
eq3 = (1, 1, 1)  # x + y = 1
eq4 = (1, 1, 2)  # x + y = 2
solution3 = solve_by_substitution(eq3, eq4)
print(f"\nSystem: x + y = 1, x + y = 2")
print(f"Solution: {solution3}")  # None - parallel lines`,
				},
				{
					Title: "Polynomials and Factoring",
					Content: `**Polynomials:**

Expressions with variables raised to non-negative integer powers.

**General Form:** aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀

**Terms:**
- **Coefficient:** The number multiplying the variable (aₙ, aₙ₋₁, etc.)
- **Degree:** Highest power of the variable (n)
- **Constant term:** Term with no variable (a₀)

**Examples:**
- 3x² + 2x + 1 (degree 2, quadratic)
- 5x³ - 2x + 7 (degree 3, cubic)
- 2x + 3 (degree 1, linear)

**Operations:**

**Addition/Subtraction:** Combine like terms
- (3x² + 2x + 1) + (x² - x + 5) = 4x² + x + 6

**Multiplication:** Use distributive property
- (x + 2)(x + 3) = x² + 3x + 2x + 6 = x² + 5x + 6

**Factoring:**

Expressing a polynomial as a product of simpler polynomials.

**Common Factoring Patterns:**

**1. Greatest Common Factor (GCF):**
- 6x² + 9x = 3x(2x + 3)

**2. Difference of Squares:** a² - b² = (a + b)(a - b)
- x² - 9 = (x + 3)(x - 3)

**3. Perfect Square Trinomials:**
- a² + 2ab + b² = (a + b)²
- a² - 2ab + b² = (a - b)²
- Example: x² + 6x + 9 = (x + 3)²

**4. Factoring Quadratics:** x² + bx + c = (x + p)(x + q) where p + q = b and pq = c
- x² + 5x + 6 = (x + 2)(x + 3) because 2 + 3 = 5 and 2×3 = 6

**Why This Matters:**
Polynomials appear in:
- Algorithm complexity (polynomial time algorithms)
- Curve fitting and interpolation
- Error correction codes
- Cryptographic protocols`,
					CodeExamples: `# Working with polynomials in Python

# Representing polynomials as lists [coefficient for x^0, x^1, x^2, ...]
# Example: 3x² + 2x + 1 = [1, 2, 3]

def add_polynomials(p1, p2):
    """Add two polynomials."""
    max_len = max(len(p1), len(p2))
    result = [0] * max_len
    for i in range(len(p1)):
        result[i] += p1[i]
    for i in range(len(p2)):
        result[i] += p2[i]
    return result

def multiply_polynomials(p1, p2):
    """Multiply two polynomials."""
    result = [0] * (len(p1) + len(p2) - 1)
    for i in range(len(p1)):
        for j in range(len(p2)):
            result[i + j] += p1[i] * p2[j]
    return result

# Example: (3x² + 2x + 1) + (x² - x + 5)
p1 = [1, 2, 3]  # 3x² + 2x + 1
p2 = [5, -1, 1]  # x² - x + 5
sum_poly = add_polynomials(p1, p2)
print(f"({p1}) + ({p2}) = {sum_poly}")

# Example: (x + 2)(x + 3) = x² + 5x + 6
p3 = [2, 1]  # x + 2
p4 = [3, 1]  # x + 3
product = multiply_polynomials(p3, p4)
print(f"({p3}) × ({p4}) = {product}")

# Factoring: finding factors of x² + 5x + 6
def factor_quadratic(a, b, c):
    """Factor quadratic ax² + bx + c."""
    # Find two numbers that multiply to ac and add to b
    product = a * c
    for p in range(-abs(product), abs(product) + 1):
        if p == 0:
            continue
        q = product // p if p != 0 else 0
        if p + q == b and p * q == product:
            # Factor out a if needed
            return f"(x + {p//a})(x + {q//a})" if a == 1 else "Complex"
    return "Cannot factor easily"

# Factor x² + 5x + 6
factors = factor_quadratic(1, 5, 6)
print(f"\nFactor x² + 5x + 6: {factors}")

# Evaluate polynomial at a point
def evaluate_polynomial(poly, x):
    """Evaluate polynomial at x."""
    result = 0
    for i, coeff in enumerate(poly):
        result += coeff * (x ** i)
    return result

# Evaluate 3x² + 2x + 1 at x = 2
value = evaluate_polynomial([1, 2, 3], 2)
print(f"\n3x² + 2x + 1 at x=2: {value}")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          502,
			Title:       "Geometry Basics",
			Description: "Learn fundamental geometric concepts including shapes, angles, and formulas.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "Points, Lines, and Planes",
					Content: `**Introduction: The Building Blocks of Geometry**

Points, lines, and planes are the fundamental undefined terms in geometry - we can describe them but not define them in terms of simpler concepts. Understanding these basics is essential for all geometric reasoning, computer graphics, and spatial algorithms.

**Basic Geometric Concepts: The Foundation**

**Point: A Location in Space**

A point represents a specific location in space. It has no size, no dimension - just position.

**Properties:**
- No length, width, or height
- Represented by a dot (·) in diagrams
- Named with capital letters: Point A, Point B, Point P
- In coordinate systems: Represented by coordinates (x, y) in 2D, (x, y, z) in 3D

**Why Points Matter:**
- **Coordinates:** Every point can be represented numerically
- **Vertices:** Points define corners of shapes
- **Locations:** Represent positions in space (GPS coordinates, pixel positions)
- **Data Structures:** Points are fundamental in computational geometry

**Line: An Infinite Path**

A line is a straight path that extends infinitely in both directions. It has length but no width.

**Properties:**
- Extends infinitely in both directions
- Contains infinitely many points
- Has no endpoints
- Named by two points on it: Line AB, Line ↔AB, or simply "line l"

**Equation of a Line:**
In coordinate geometry, a line can be represented as:
- Slope-intercept form: y = mx + b
- Point-slope form: y - y₁ = m(x - x₁)
- General form: Ax + By + C = 0

**Why Lines Matter:**
- **Paths:** Represent trajectories, routes, boundaries
- **Graphs:** Linear functions graph as lines
- **Algorithms:** Line intersection, line-of-sight calculations
- **Graphics:** Rendering lines, edges of polygons

**Line Segment: A Finite Portion**

A line segment is part of a line with two distinct endpoints. Unlike a line, it has a definite length.

**Properties:**
- Has two endpoints
- Has measurable length
- Named by its endpoints: Segment AB, Segment —AB, or AB with a bar

**Length Calculation:**
In coordinate geometry, if A = (x₁, y₁) and B = (x₂, y₂):
- Distance = √[(x₂ - x₁)² + (y₂ - y₁)²]
- This is the distance formula (Pythagorean theorem)

**Why Line Segments Matter:**
- **Edges:** Form edges of polygons and polyhedra
- **Paths:** Represent finite paths between points
- **Measurements:** Calculate distances between locations
- **Graphics:** Draw line segments to create shapes

**Ray: A Half-Line**

A ray is part of a line with one endpoint extending infinitely in one direction.

**Properties:**
- Has one endpoint (the starting point)
- Extends infinitely in one direction
- Named by endpoint and direction: Ray AB, Ray →AB
- The endpoint is always listed first

**Why Rays Matter:**
- **Directions:** Represent directions from a point
- **Angles:** Form sides of angles
- **Ray Tracing:** Used in computer graphics for rendering
- **Line-of-Sight:** Check visibility from a point

**Plane: A Flat Surface**

A plane is a flat surface that extends infinitely in all directions. It has length and width but no thickness.

**Properties:**
- Two-dimensional (2D)
- Extends infinitely in all directions
- Named by three non-collinear points: Plane ABC
- Or by a single letter: Plane P

**Equation of a Plane:**
In 3D coordinate geometry:
- General form: Ax + By + Cz + D = 0
- Point-normal form: n · (r - r₀) = 0
- Where n is the normal vector, r is a point on the plane

**Why Planes Matter:**
- **Surfaces:** Represent flat surfaces in 3D space
- **Clipping:** Used in 3D graphics for view frustum culling
- **Collision Detection:** Check if objects intersect planes
- **Coordinate Systems:** Define reference frames

**Relationships: How These Objects Interact**

**Parallel Lines: Never Meeting**

Two lines are parallel if they lie in the same plane and never intersect, no matter how far extended.

**Properties:**
- Symbol: ∥ (parallel symbol)
- Same direction (slope in coordinate geometry)
- Never intersect
- Distance between parallel lines is constant

**In Coordinate Geometry:**
- Two lines are parallel if they have the same slope
- Example: y = 2x + 3 and y = 2x - 5 are parallel (both have slope 2)

**Why Parallel Lines Matter:**
- **Graphics:** Parallel projection, isometric views
- **Algorithms:** Parallel processing, parallel data structures
- **Optimization:** Some problems are easier when lines are parallel

**Perpendicular Lines: Right Angles**

Two lines are perpendicular if they intersect at a 90° (right) angle.

**Properties:**
- Symbol: ⊥ (perpendicular symbol)
- Form four 90° angles at intersection
- In coordinate geometry: Slopes are negative reciprocals
- If line 1 has slope m₁ and line 2 has slope m₂, then m₁ × m₂ = -1

**Example:**
- Line 1: y = 2x + 3 (slope = 2)
- Line 2: y = -½x + 1 (slope = -½)
- Check: 2 × (-½) = -1 ✓ (perpendicular!)

**Why Perpendicular Lines Matter:**
- **Coordinate Systems:** Define axes (x-axis ⊥ y-axis)
- **Graphics:** Orthogonal projections, grid systems
- **Optimization:** Perpendicular directions for gradient descent
- **Construction:** Building structures at right angles

**Collinear Points: On the Same Line**

Points are collinear if they all lie on the same straight line.

**Properties:**
- Three or more points can be collinear
- Two points are always collinear (they define a line)
- In coordinate geometry: Check if slopes between points are equal

**Checking Collinearity:**
For points A(x₁, y₁), B(x₂, y₂), C(x₃, y₃):
- Calculate slopes: m_AB = (y₂ - y₁)/(x₂ - x₁), m_BC = (y₃ - y₂)/(x₃ - x₂)
- If m_AB = m_BC, points are collinear
- Or use area: If area of triangle ABC = 0, points are collinear

**Why Collinearity Matters:**
- **Simplification:** Collinear points can be simplified in algorithms
- **Convex Hull:** Collinear points on boundary need special handling
- **Path Planning:** Check if waypoints are on a straight path
- **Graphics:** Simplify polygon edges

**Coplanar Points: In the Same Plane**

Points are coplanar if they all lie in the same plane.

**Properties:**
- Four or more points can be coplanar
- Three points are always coplanar (they define a plane)
- In 3D: Check if all points satisfy the same plane equation

**Checking Coplanarity:**
For four points in 3D, calculate the volume of the tetrahedron they form:
- If volume = 0, points are coplanar
- Or check if all points satisfy the same plane equation

**Why Coplanarity Matters:**
- **3D Graphics:** Many algorithms assume coplanar polygons
- **Simplification:** Reduce 3D problems to 2D when points are coplanar
- **Collision Detection:** Check if objects lie in the same plane
- **Optimization:** Planar problems are often easier to solve

**Applications in Computer Science:**

**1. Computer Graphics:**
- **2D Rendering:** Points define vertices, lines form edges, planes define surfaces
- **3D Modeling:** Represent 3D objects using points, lines, and planes
- **Ray Tracing:** Cast rays (lines) to determine visibility and lighting
- **Clipping:** Use planes to determine what's visible in the view frustum

**2. Game Development:**
- **Collision Detection:** Check if points, lines, or shapes intersect
- **Spatial Relationships:** Determine if objects are parallel, perpendicular, collinear
- **Pathfinding:** Use line segments to represent paths
- **Line-of-Sight:** Cast rays to check visibility

**3. Geographic Information Systems (GIS):**
- **Coordinates:** Points represent locations (latitude, longitude)
- **Paths:** Line segments represent roads, boundaries
- **Regions:** Planes (or polygons) represent areas
- **Distance Calculations:** Use line segments to measure distances

**4. Pathfinding Algorithms:**
- **Graph Representation:** Points are nodes, line segments are edges
- **Shortest Path:** Find shortest path along line segments
- **Visibility Graphs:** Use lines to determine visible paths
- **Navigation:** Plan routes using geometric relationships

**5. Computational Geometry:**
- **Convex Hull:** Find the smallest convex shape containing points
- **Line Intersection:** Find where lines intersect
- **Point-in-Polygon:** Determine if a point is inside a polygon
- **Voronoi Diagrams:** Partition space based on points

**Why This Matters:**

Geometric concepts are fundamental to:
- **Spatial Reasoning:** Understanding relationships in space
- **Algorithm Design:** Many algorithms work with geometric objects
- **Problem Solving:** Many real-world problems have geometric components
- **Visualization:** Representing data and relationships visually

**Next Steps:**

Mastering points, lines, and planes provides the foundation for understanding angles, shapes, and more complex geometric relationships. In the next lesson, we'll explore angles and their properties, which build on these fundamental concepts.`,
					CodeExamples: `# Representing geometric objects in Python

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    
    def length(self):
        """Calculate length of line segment."""
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        return (dx**2 + dy**2)**0.5
    
    def slope(self):
        """Calculate slope of line."""
        dx = self.p2.x - self.p1.x
        if dx == 0:
            return float('inf')  # Vertical line
        return (self.p2.y - self.p1.y) / dx

# Create points
A = Point(0, 0)
B = Point(3, 4)
C = Point(6, 8)

# Create lines
line_AB = Line(A, B)
line_BC = Line(B, C)

print(f"Line AB length: {line_AB.length()}")
print(f"Line AB slope: {line_AB.slope()}")

# Check if lines are parallel (same slope)
def are_parallel(line1, line2):
    """Check if two lines are parallel."""
    return abs(line1.slope() - line2.slope()) < 1e-10

print(f"\nLines AB and BC parallel: {are_parallel(line_AB, line_BC)}")

# Check if lines are perpendicular (slopes multiply to -1)
def are_perpendicular(line1, line2):
    """Check if two lines are perpendicular."""
    s1, s2 = line1.slope(), line2.slope()
    if s1 == float('inf') or s2 == float('inf'):
        return False
    return abs(s1 * s2 + 1) < 1e-10`,
				},
				{
					Title: "Angles and Their Properties",
					Content: `**Introduction: Measuring Rotation and Direction**

Angles are fundamental geometric concepts that measure rotation, direction, and the relationship between lines. Understanding angles is essential for trigonometry, computer graphics, navigation, and many algorithms that work with spatial relationships.

**Understanding Angles: Rotation and Measurement**

An angle is formed by two rays (or line segments) sharing a common endpoint called the vertex. The angle measures the amount of rotation needed to rotate one ray to coincide with the other.

**Components of an Angle:**
- **Vertex:** The common endpoint where the two rays meet
- **Sides:** The two rays that form the angle
- **Measure:** The amount of rotation (in degrees or radians)

**Notation:**
- ∠ABC or ∠B (angle with vertex B)
- Greek letters: α (alpha), β (beta), θ (theta)
- Measured in degrees (°) or radians (rad)

**Types of Angles: Classification by Measure**

**1. Acute Angle: Sharp and Less Than 90°**

An acute angle measures less than 90 degrees. The name comes from Latin meaning "sharp."

**Properties:**
- Measure: 0° < angle < 90°
- Appears "sharp" or "pointed"
- Common examples: 30°, 45°, 60°

**Examples:**
- 30°: One-third of a right angle
- 45°: Half of a right angle (very common in geometry)
- 60°: One-third of a straight angle (equilateral triangle angles)

**Why Acute Angles Matter:**
- **Triangles:** All angles in an equilateral triangle are 60° (acute)
- **Trigonometry:** Sine and cosine functions are positive for acute angles
- **Graphics:** Many rotations use acute angles
- **Navigation:** Small course corrections are often acute angles

**2. Right Angle: The Perfect 90°**

A right angle measures exactly 90 degrees. It's called "right" because it's the standard, correct angle for many constructions.

**Properties:**
- Measure: Exactly 90°
- Symbol: ∟ or a small square in diagrams
- Forms an "L" shape
- Perpendicular lines form right angles

**Examples:**
- Corner of a square or rectangle
- Intersection of x-axis and y-axis in coordinate system
- Corner of a room (ideally!)

**Why Right Angles Matter:**
- **Coordinate Systems:** Define perpendicular axes
- **Pythagorean Theorem:** Works with right triangles
- **Graphics:** Screen coordinates use right angles
- **Construction:** Building structures rely on right angles

**3. Obtuse Angle: Between 90° and 180°**

An obtuse angle measures more than 90° but less than 180°. The name comes from Latin meaning "blunt" or "dull."

**Properties:**
- Measure: 90° < angle < 180°
- Appears "wide" or "open"
- Common examples: 120°, 135°, 150°

**Examples:**
- 120°: Angle in a regular hexagon
- 135°: Supplement of 45° (common in octagons)
- 150°: Supplement of 30°

**Why Obtuse Angles Matter:**
- **Polygons:** Many polygons have obtuse angles
- **Trigonometry:** Cosine is negative for obtuse angles
- **Navigation:** Wide turns use obtuse angles
- **Graphics:** Some transformations use obtuse rotations

**4. Straight Angle: The 180° Line**

A straight angle measures exactly 180 degrees. The two rays form a straight line.

**Properties:**
- Measure: Exactly 180°
- Forms a straight line
- Half of a full rotation (360°)

**Examples:**
- A line segment extended in both directions
- Diameter of a circle (subtends 180°)
- Supplementary angles sum to 180°

**Why Straight Angles Matter:**
- **Lines:** Define straight paths
- **Supplementary Angles:** Two angles that sum to 180°
- **Half-Turns:** 180° rotation reverses direction
- **Geometry:** Many theorems involve straight angles

**5. Reflex Angle: Beyond 180°**

A reflex angle measures more than 180° but less than 360°. It's the "other" part of an angle when you go the long way around.

**Properties:**
- Measure: 180° < angle < 360°
- The larger angle formed by two rays
- Less commonly used than other types

**Examples:**
- 270°: Three-quarters of a full rotation
- 225°: 180° + 45°
- Any angle > 180° and < 360°

**Why Reflex Angles Matter:**
- **Complete Description:** Sometimes we need the larger angle
- **Navigation:** Some paths require reflex angles
- **Rotations:** Full rotations can exceed 180°
- **Geometry:** Complete angle descriptions

**Angle Relationships: How Angles Interact**

Understanding relationships between angles helps solve geometric problems and understand spatial configurations.

**1. Complementary Angles: Sum to 90°**

Two angles are complementary if their measures sum to exactly 90 degrees.

**Properties:**
- ∠A + ∠B = 90°
- If one angle is known, the other is 90° - (known angle)
- Often appear in right triangles

**Examples:**
- 30° and 60°: 30° + 60° = 90°
- 45° and 45°: 45° + 45° = 90° (equal complements)
- 20° and 70°: 20° + 70° = 90°

**Why Complementary Angles Matter:**
- **Right Triangles:** Two acute angles in a right triangle are complementary
- **Trigonometry:** sin(θ) = cos(90° - θ) for complementary angles
- **Problem Solving:** Finding unknown angles in geometric figures

**2. Supplementary Angles: Sum to 180°**

Two angles are supplementary if their measures sum to exactly 180 degrees.

**Properties:**
- ∠A + ∠B = 180°
- If one angle is known, the other is 180° - (known angle)
- Form a straight line when placed adjacent

**Examples:**
- 120° and 60°: 120° + 60° = 180°
- 90° and 90°: 90° + 90° = 180° (equal supplements)
- 45° and 135°: 45° + 135° = 180°

**Why Supplementary Angles Matter:**
- **Straight Lines:** Angles on a straight line are supplementary
- **Parallel Lines:** Same-side interior angles are supplementary
- **Polygons:** Interior and exterior angles are supplementary
- **Problem Solving:** Finding unknown angles

**3. Vertical Angles: Opposite and Equal**

Vertical angles are formed when two lines intersect. They are the angles opposite each other at the intersection point.

**Properties:**
- Vertical angles are always equal
- Formed by two intersecting lines
- Come in pairs (two pairs per intersection)

**Visual Example:**

     a
     |
  c--+--d
     |
     b
Angles a and b are vertical (equal), and angles c and d are vertical (equal).

**Why Vertical Angles Matter:**
- **Problem Solving:** If one vertical angle is known, the opposite is equal
- **Proofs:** Many geometric proofs use vertical angle equality
- **Graphics:** Intersecting lines create vertical angles
- **Navigation:** Intersecting paths form vertical angles

**4. Adjacent Angles: Sharing a Side**

Adjacent angles share a common side and vertex but don't overlap.

**Properties:**
- Share a common side (ray)
- Share a common vertex
- Don't overlap (no interior points in common)
- Can be complementary, supplementary, or neither

**Example:**
If ∠ABC and ∠CBD share side BC and vertex B, they are adjacent.

**Why Adjacent Angles Matter:**
- **Angle Addition:** Can add adjacent angles to find larger angles
- **Problem Solving:** Breaking complex angles into adjacent parts
- **Geometry:** Many theorems involve adjacent angles

**5. Corresponding Angles: Matching Positions**

When a transversal (a line crossing two other lines) cuts through two lines, corresponding angles are in matching positions.

**Properties:**
- Same relative position at each intersection
- If lines are parallel, corresponding angles are equal
- Used to prove lines are parallel

**Visual Example:**

    1    2
    |    |
----+----+----  (line 1)
    |    |
    3    4
    |    |
----+----+----  (line 2)
    |    |
    5    6
Angles 1 and 5 are corresponding, angles 2 and 6 are corresponding, etc.

**Why Corresponding Angles Matter:**
- **Parallel Lines:** Equal corresponding angles prove lines are parallel
- **Problem Solving:** Finding unknown angles in parallel line configurations
- **Proofs:** Fundamental to many geometric proofs
- **Graphics:** Parallel projections use corresponding angles

**Angle Measurement: Degrees vs Radians**

**Degrees:**
- Full circle = 360°
- Right angle = 90°
- Straight angle = 180°
- More intuitive for most people

**Radians:**
- Full circle = 2π radians
- Right angle = π/2 radians
- Straight angle = π radians
- More natural for mathematics and programming
- Conversion: degrees × π/180 = radians

**Why Both Matter:**
- **Degrees:** Human-friendly, used in navigation, construction
- **Radians:** Mathematically natural, used in calculus, programming
- **Conversion:** Essential skill for working with angles in code

**Applications in Computer Science:**

**1. Rotation Transformations:**
- Rotating objects in 2D/3D graphics
- Sprite rotation in games
- UI element rotations
- Example: Rotate image by 45° (π/4 radians)

**2. Navigation and Compass Directions:**
- Bearing calculations (angles from north)
- GPS navigation
- Path planning with angle constraints
- Example: Turn 90° right, then 45° left

**3. Trigonometry Calculations:**
- Sine, cosine, tangent functions use angles
- Calculating distances and heights
- Projection calculations
- Example: Calculate height using angle of elevation

**4. 3D Modeling and Rendering:**
- Camera angles and field of view
- Light direction angles
- Surface normal calculations
- Example: Calculate lighting based on angle of incidence

**5. Collision Detection:**
- Angle between velocity vectors
- Reflection angles
- Bounce calculations
- Example: Calculate reflection angle when ball hits wall

**6. Pathfinding:**
- Direction changes (angles)
- Smooth path generation
- Obstacle avoidance with angle constraints
- Example: Find path with maximum turn angle of 45°

**Common Mistakes:**

**Mistake 1: Confusing complementary and supplementary**
- Complementary = 90°, supplementary = 180°
- Memory aid: "C" comes before "S", 90° < 180°

**Mistake 2: Assuming all adjacent angles are supplementary**
- Only true if they form a straight line
- Adjacent angles can have any relationship

**Mistake 3: Forgetting vertical angles are equal**
- Always remember: vertical angles = equal
- Useful for solving many problems

**Mistake 4: Mixing degrees and radians**
- Always check units!
- Most programming languages use radians by default
- Convert explicitly: radians = degrees × π/180

**Next Steps:**

Mastering angles provides the foundation for trigonometry, which extends these concepts to calculate distances, heights, and relationships in triangles. In the next lesson, we'll explore triangles, which are fundamental shapes built from angles and line segments.`,
					CodeExamples: `# Working with angles in Python

import math

def degrees_to_radians(degrees):
    """Convert degrees to radians."""
    return degrees * math.pi / 180

def radians_to_degrees(radians):
    """Convert radians to degrees."""
    return radians * 180 / math.pi

# Angle types
def classify_angle(degrees):
    """Classify an angle by its measure."""
    if 0 < degrees < 90:
        return "Acute"
    elif degrees == 90:
        return "Right"
    elif 90 < degrees < 180:
        return "Obtuse"
    elif degrees == 180:
        return "Straight"
    elif 180 < degrees < 360:
        return "Reflex"
    else:
        return "Invalid"

# Test angles
angles = [30, 45, 90, 120, 180, 270]
for angle in angles:
    print(f"{angle}° is {classify_angle(angle)}")

# Complementary angles
def find_complement(angle):
    """Find complement of an angle."""
    return 90 - angle if angle < 90 else None

# Supplementary angles
def find_supplement(angle):
    """Find supplement of an angle."""
    return 180 - angle if angle < 180 else None

print(f"\nComplement of 30°: {find_complement(30)}°")
print(f"Supplement of 120°: {find_supplement(120)}°")

# Angle conversion
angle_deg = 45
angle_rad = degrees_to_radians(angle_deg)
print(f"\n{angle_deg}° = {angle_rad:.4f} radians")
print(f"{angle_rad:.4f} radians = {radians_to_degrees(angle_rad):.2f}°")`,
				},
				{
					Title: "Triangles: Types, Properties, and Theorems",
					Content: `**Introduction: The Simplest Polygon**

Triangles are the simplest polygons - just three sides and three angles. Yet they're incredibly powerful and fundamental to geometry, computer graphics, and computational geometry. Every polygon can be decomposed into triangles, making them the building blocks of complex shapes.

**What is a Triangle?**

A triangle is a polygon with exactly three sides and three angles. The three vertices (corner points) are connected by three edges (sides).

**Key Properties:**
- **Three vertices:** Points A, B, C
- **Three sides:** Segments AB, BC, CA
- **Three angles:** ∠A, ∠B, ∠C
- **Closed figure:** Forms a closed loop
- **Planar:** All points lie in the same plane

**Types by Sides: Classification by Side Lengths**

**1. Equilateral Triangle: Perfect Symmetry**

**Definition:** All three sides are equal in length.

**Properties:**
- All sides: a = b = c
- All angles: 60° each (since sum = 180°)
- Maximum symmetry: Three lines of symmetry
- Special case: Regular triangle

**Example:**
- Sides: 5, 5, 5
- Angles: 60°, 60°, 60°
- Perimeter: 15
- Area: (√3/4) × 5² ≈ 10.83

**Why Equilateral Matters:**
- **Graphics:** Used in tessellations and patterns
- **Algorithms:** Simplest case for many geometric algorithms
- **Optimization:** Often optimal shape (minimum perimeter for given area)

**2. Isosceles Triangle: Two Equal Sides**

**Definition:** Exactly two sides are equal in length.

**Properties:**
- Two equal sides: a = b (or a = c, or b = c)
- Two equal angles: Angles opposite equal sides are equal
- One line of symmetry: Through the vertex and midpoint of base

**Example:**
- Sides: 5, 5, 3
- Angles: ~53.1°, ~53.1°, ~73.8° (approximately)
- The two 5-unit sides are equal
- Angles opposite these sides are equal

**Why Isosceles Matters:**
- **Common shape:** Appears frequently in real-world objects
- **Symmetry:** One axis of symmetry simplifies calculations
- **Algorithms:** Symmetry can be exploited for optimization

**3. Scalene Triangle: All Sides Different**

**Definition:** All three sides have different lengths.

**Properties:**
- All sides different: a ≠ b ≠ c
- All angles different: ∠A ≠ ∠B ≠ ∠C
- No lines of symmetry
- Most general case

**Example:**
- Sides: 3, 4, 5 (Pythagorean triple!)
- Angles: ~36.9°, ~53.1°, 90° (right triangle)
- No equal sides or angles

**Why Scalene Matters:**
- **General case:** Most triangles are scalene
- **Algorithms:** Must handle general case
- **Real-world:** Most real-world triangles are scalene

**Types by Angles: Classification by Angle Measures**

**1. Acute Triangle: All Angles Sharp**

**Definition:** All three angles are less than 90°.

**Properties:**
- All angles < 90°
- All sides can be different lengths
- Can be equilateral, isosceles, or scalene

**Example:**
- Angles: 50°, 60°, 70° (sum = 180°)
- All angles are acute
- No right or obtuse angles

**Why Acute Matters:**
- **Stability:** Acute triangles are stable structures
- **Graphics:** Used in mesh generation
- **Algorithms:** Some algorithms assume acute triangles

**2. Right Triangle: One Right Angle**

**Definition:** Exactly one angle equals 90°.

**Properties:**
- One angle = 90°
- Longest side is the hypotenuse (opposite the right angle)
- Two shorter sides are the legs
- Pythagorean theorem applies: a² + b² = c²

**Example:**
- Sides: 3, 4, 5
- Angles: ~36.9°, ~53.1°, 90°
- Hypotenuse: 5 (longest side)
- Check: 3² + 4² = 9 + 16 = 25 = 5² ✓

**Why Right Triangles Matter:**
- **Pythagorean theorem:** Fundamental relationship
- **Trigonometry:** Basis for trigonometric functions
- **Graphics:** Used extensively in coordinate systems
- **Algorithms:** Many algorithms use right triangles

**3. Obtuse Triangle: One Obtuse Angle**

**Definition:** Exactly one angle is greater than 90°.

**Properties:**
- One angle > 90°
- The side opposite the obtuse angle is the longest
- Two acute angles

**Example:**
- Angles: 30°, 40°, 110°
- The 110° angle is obtuse
- Side opposite 110° is longest

**Why Obtuse Matters:**
- **Algorithms:** Some algorithms have special cases for obtuse triangles
- **Graphics:** Can cause issues in some rendering algorithms
- **Stability:** Less stable than acute triangles

**Important Theorems: Fundamental Triangle Properties**

**1. Triangle Sum Theorem: The 180° Rule**

**Statement:** The sum of the three interior angles of any triangle equals 180°.

**Formula:** ∠A + ∠B + ∠C = 180°

**Why It Works:**
- Can be proven using parallel lines and alternate interior angles
- Fundamental property of Euclidean geometry
- Works for all triangles (acute, right, obtuse)

**Applications:**
- **Find missing angle:** If two angles known, third = 180° - (sum of two)
- **Verify triangle:** If angles don't sum to 180°, not a valid triangle
- **Proofs:** Used in many geometric proofs

**Example:**
- Given: ∠A = 50°, ∠B = 60°
- Find: ∠C = 180° - (50° + 60°) = 180° - 110° = 70°

**2. Pythagorean Theorem: Right Triangle Relationship**

**Statement:** For any right triangle, the square of the hypotenuse equals the sum of squares of the legs.

**Formula:** a² + b² = c² (where c is hypotenuse)

**Proof Concept:**
- Visual proof using squares on each side
- Algebraic proof using similar triangles
- Many different proofs exist

**Applications:**
- **Find missing side:** Given two sides, find the third
- **Verify right triangle:** If a² + b² = c², triangle is right
- **Distance formula:** Basis for calculating distances

**Example:**
- Legs: a = 3, b = 4
- Hypotenuse: c = √(3² + 4²) = √(9 + 16) = √25 = 5

**3. Triangle Inequality: When Three Sides Form a Triangle**

**Statement:** For any triangle, the sum of any two sides must be greater than the third side.

**Formulas:**
- a + b > c
- a + c > b
- b + c > a

**Why It Works:**
- If sum equals third side, points are collinear (degenerate triangle)
- If sum is less, sides can't connect to form triangle
- Only when sum is greater can triangle exist

**Applications:**
- **Validate triangle:** Check if three lengths can form a triangle
- **Algorithms:** Used in geometric algorithms to check validity
- **Optimization:** Constraint in optimization problems

**Example:**
- Sides: 3, 4, 5
- Check: 3 + 4 = 7 > 5 ✓
- Check: 3 + 5 = 8 > 4 ✓
- Check: 4 + 5 = 9 > 3 ✓
- Valid triangle!

**Counter-example:**
- Sides: 1, 2, 4
- Check: 1 + 2 = 3 < 4 ✗
- Cannot form a triangle!

**Area Formulas: Multiple Ways to Calculate**

**1. General Formula: Base and Height**

**Formula:** Area = (1/2) × base × height

**How It Works:**
- Base: Any side of the triangle
- Height: Perpendicular distance from base to opposite vertex
- Works for any triangle

**Example:**
- Base: 6 units
- Height: 4 units
- Area: (1/2) × 6 × 4 = 12 square units

**Why This Matters:**
- **Most intuitive:** Easy to understand and visualize
- **Graphics:** Used in rendering (rasterization)
- **Algorithms:** Basis for many area calculations

**2. Heron's Formula: When You Know All Sides**

**Formula:**
- s = (a + b + c) / 2 (semi-perimeter)
- Area = √[s(s-a)(s-b)(s-c)]

**When to Use:**
- Know all three side lengths
- Don't know height
- Need precise area calculation

**Example:**
- Sides: a = 3, b = 4, c = 5
- s = (3 + 4 + 5) / 2 = 6
- Area = √[6(6-3)(6-4)(6-5)] = √[6 × 3 × 2 × 1] = √36 = 6

**Why Heron's Formula Matters:**
- **No height needed:** Only requires side lengths
- **Algorithms:** Used in computational geometry
- **Precision:** Numerically stable for most cases

**3. Coordinate Formula: Using Vertices**

**Formula (Shoelace Formula):**
- Given vertices (x₁, y₁), (x₂, y₂), (x₃, y₃)
- Area = |(x₁(y₂ - y₃) + x₂(y₃ - y₁) + x₃(y₁ - y₂)) / 2|

**When to Use:**
- Have coordinates of vertices
- Working in coordinate system
- Need signed area (for orientation)

**Example:**
- Vertices: (0, 0), (3, 0), (0, 4)
- Area = |(0(0-4) + 3(4-0) + 0(0-0)) / 2| = |12 / 2| = 6

**Why Coordinate Formula Matters:**
- **Graphics:** Directly applicable to pixel coordinates
- **Algorithms:** Used in many computational geometry algorithms
- **Orientation:** Can determine triangle orientation (clockwise/counterclockwise)

**Applications in Computer Science:**

**1. 3D Graphics: Triangulation**

**Triangulation:**
- All 3D surfaces are decomposed into triangles
- Triangles are planar (easy to render)
- GPU hardware optimized for triangles
- Every polygon → triangles

**Why Triangles:**
- **Planar:** Three points always define a plane
- **Simple:** Simplest polygon to render
- **Hardware:** GPUs optimized for triangle rendering
- **Flexible:** Can approximate any surface

**2. Computational Geometry:**

**Delaunay Triangulation:**
- Triangulation that maximizes minimum angle
- Avoids skinny triangles
- Used in mesh generation, finite element analysis

**Convex Hull:**
- Smallest convex polygon containing points
- Often computed using triangulation
- Used in collision detection, path planning

**3. Mesh Generation:**

**Finite Element Analysis:**
- Domain divided into triangular elements
- Each triangle analyzed separately
- Results combined for overall solution

**Surface Meshing:**
- 3D surfaces represented as triangle meshes
- Quality depends on triangle shape
- Equilateral triangles preferred (but not always possible)

**4. Collision Detection:**

**Bounding Triangles:**
- Objects approximated by triangles
- Check triangle-triangle intersection
- Faster than exact collision detection

**Spatial Partitioning:**
- Space divided into triangular regions
- Efficient spatial queries
- Used in game engines, physics simulations

**5. Pathfinding and Navigation:**

**Navigation Meshes:**
- Walkable areas represented as triangles
- A* pathfinding on triangle mesh
- Used in game AI, robotics

**Why This Matters:**

Triangles are fundamental to:
- **3D Graphics:** All rendering uses triangles
- **Computational Geometry:** Basis for many algorithms
- **Mesh Generation:** Creating computational meshes
- **Collision Detection:** Efficient collision algorithms
- **Game Development:** Navigation, rendering, physics
- **Scientific Computing:** Finite element methods

**Next Steps:**

Understanding triangles provides the foundation for more complex geometric concepts. In the next lesson, we'll explore circles, which have their own rich set of properties and applications.`,
					CodeExamples: `# Working with triangles in Python

import math

class Triangle:
    def __init__(self, a, b, c):
        """Create triangle with sides a, b, c."""
        if not self.is_valid(a, b, c):
            raise ValueError("Invalid triangle: violates triangle inequality")
        self.a = a
        self.b = b
        self.c = c
    
    def is_valid(self, a, b, c):
        """Check triangle inequality."""
        return (a + b > c) and (a + c > b) and (b + c > a)
    
    def perimeter(self):
        """Calculate perimeter."""
        return self.a + self.b + self.c
    
    def area_heron(self):
        """Calculate area using Heron's formula."""
        s = self.perimeter() / 2
        return math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))
    
    def is_right_triangle(self):
        """Check if triangle is right-angled."""
        sides = sorted([self.a, self.b, self.c])
        return abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < 1e-10
    
    def classify_by_sides(self):
        """Classify triangle by side lengths."""
        if self.a == self.b == self.c:
            return "Equilateral"
        elif self.a == self.b or self.a == self.c or self.b == self.c:
            return "Isosceles"
        else:
            return "Scalene"

# Example triangles
tri1 = Triangle(3, 4, 5)
print(f"Triangle(3, 4, 5):")
print(f"  Perimeter: {tri1.perimeter()}")
print(f"  Area: {tri1.area_heron():.2f}")
print(f"  Right triangle: {tri1.is_right_triangle()}")
print(f"  Type: {tri1.classify_by_sides()}")

tri2 = Triangle(5, 5, 5)
print(f"\nTriangle(5, 5, 5):")
print(f"  Type: {tri2.classify_by_sides()}")
print(f"  Area: {tri2.area_heron():.2f}")

# Pythagorean theorem
def pythagorean(a, b, find='c'):
    """Find missing side of right triangle."""
    if find == 'c':
        return math.sqrt(a**2 + b**2)
    elif find == 'a':
        return math.sqrt(b**2 - a**2) if b > a else None
    else:  # find == 'b'
        return math.sqrt(a**2 - b**2) if a > b else None

print(f"\nPythagorean theorem:")
print(f"If a=3, b=4, then c={pythagorean(3, 4, 'c'):.2f}")`,
				},
				{
					Title: "Circles: Properties and Formulas",
					Content: `**Introduction: Perfect Symmetry**

Circles are among the most fundamental and beautiful shapes in mathematics. Defined by perfect symmetry around a center point, circles appear everywhere - from wheels and clocks to computer graphics and game development. Understanding circles is essential for working with curves, rotations, and circular motion.

**What is a Circle?**

A circle is the set of all points in a plane that are equidistant from a fixed point called the center.

**Definition:**
- **Locus definition:** All points at distance r from center O
- **Perfect symmetry:** Every point on circle is equivalent
- **2D shape:** Lies in a plane
- **Closed curve:** Continuous, no endpoints

**Key Terms: Understanding Circle Components**

**1. Center (O): The Fixed Point**

**Definition:** The fixed point from which all points on the circle are equidistant.

**Properties:**
- Usually denoted as O or (h, k) in coordinates
- Unique: Every circle has exactly one center
- Symmetry point: All rotations around center map circle to itself

**Why Center Matters:**
- **Coordinates:** Defines position of circle in space
- **Transformations:** Rotations and scaling centered at this point
- **Algorithms:** Many algorithms use center as reference point

**2. Radius (r): The Constant Distance**

**Definition:** The distance from the center to any point on the circle.

**Properties:**
- Constant: Same for all points on circle
- Positive: r > 0 (r = 0 is degenerate point)
- Determines size: Larger r = larger circle

**Why Radius Matters:**
- **Size:** Determines how big the circle is
- **Formulas:** Appears in all circle formulas
- **Scaling:** Doubling radius quadruples area

**3. Diameter (d): Through the Center**

**Definition:** A line segment passing through the center with endpoints on the circle.

**Properties:**
- Length: d = 2r (twice the radius)
- Longest chord: Diameter is the longest possible chord
- Two diameters: Perpendicular diameters divide circle into four equal parts

**Why Diameter Matters:**
- **Alternative formula:** Sometimes easier to work with diameter
- **Symmetry:** Defines axes of symmetry
- **Measurements:** Often measured directly (e.g., wheel diameter)

**4. Circumference (C): The Perimeter**

**Definition:** The distance around the circle - the perimeter.

**Properties:**
- Closed curve: Complete loop around circle
- Related to π: C = 2πr = πd
- π ≈ 3.14159... (irrational number)

**Why Circumference Matters:**
- **Perimeter:** Like perimeter for polygons
- **Real-world:** Wheel rotations, measuring distances
- **π relationship:** Historical importance in calculating π

**5. Chord: Connecting Two Points**

**Definition:** A line segment with both endpoints on the circle.

**Properties:**
- Any two points: Can draw chord between any two points
- Length varies: From 0 (point) to d (diameter)
- Perpendicular bisector: Passes through center for diameter

**Why Chords Matter:**
- **Segments:** Define circular segments
- **Algorithms:** Used in circle intersection algorithms
- **Graphics:** Drawing arcs and segments

**6. Arc: Part of the Circumference**

**Definition:** A portion of the circle's circumference.

**Types:**
- **Minor arc:** Less than 180° (smaller arc)
- **Major arc:** More than 180° (larger arc)
- **Semicircle:** Exactly 180°

**Properties:**
- Measured by angle: Arc length proportional to central angle
- Two arcs: Two points define two arcs (minor and major)

**Why Arcs Matter:**
- **Curves:** Represent curved paths
- **Graphics:** Drawing partial circles
- **Path planning:** Circular motion paths

**7. Sector: Pie Slice**

**Definition:** Region bounded by two radii and the arc between them.

**Properties:**
- Like pizza slice: Triangular region with curved edge
- Area: Proportional to central angle
- Angle: Measured in degrees or radians

**Why Sectors Matter:**
- **Area calculations:** Finding area of circular regions
- **Graphics:** Drawing pie charts, circular progress bars
- **Real-world:** Slices of pie, wheel sectors

**8. Segment: Chord and Arc**

**Definition:** Region bounded by a chord and the arc it subtends.

**Properties:**
- Two types: Minor segment (smaller) and major segment (larger)
- Area: Can calculate using sector minus triangle
- Common shape: Appears in many applications

**Why Segments Matter:**
- **Area calculations:** Finding area of circular regions
- **Graphics:** Drawing circular segments
- **Real-world:** Window shapes, architectural elements

**Formulas: Calculating Circle Properties**

**1. Circumference: Distance Around**

**Formula:** C = 2πr = πd

**Derivation:**
- π is defined as ratio of circumference to diameter
- C/d = π, so C = πd
- Since d = 2r, C = 2πr

**Example:**
- Radius: r = 5
- Circumference: C = 2π(5) = 10π ≈ 31.42 units

**Why This Formula Matters:**
- **Fundamental:** One of the most important formulas
- **π definition:** Historically how π was discovered
- **Real-world:** Wheel rotations, measuring circular objects

**2. Area: Space Inside**

**Formula:** A = πr²

**Derivation:**
- Can derive using calculus (integration)
- Or by dividing circle into many sectors and rearranging
- Shows why area grows as r² (quadratic growth)

**Example:**
- Radius: r = 5
- Area: A = π(5)² = 25π ≈ 78.54 square units

**Why This Formula Matters:**
- **Quadratic growth:** Doubling radius quadruples area
- **Optimization:** Maximize area for given perimeter
- **Real-world:** Material calculations, coverage area

**3. Arc Length: Distance Along Curve**

**Formula (Radians):** s = rθ
**Formula (Degrees):** s = (θ/360) × 2πr = (θ/180) × πr

**Why Two Formulas:**
- Radians: Natural unit (s = rθ is simpler)
- Degrees: More intuitive for many people
- Conversion: 180° = π radians

**Example:**
- Radius: r = 5, Angle: θ = 60° = π/3 radians
- Arc length: s = 5 × (π/3) = 5π/3 ≈ 5.24 units
- Or: s = (60/360) × 2π(5) = (1/6) × 10π = 5π/3 ✓

**Why Arc Length Matters:**
- **Curved paths:** Calculate distance along curves
- **Graphics:** Drawing arcs and curves
- **Motion:** Circular motion calculations

**4. Sector Area: Area of Pie Slice**

**Formula (Radians):** A = (1/2)r²θ
**Formula (Degrees):** A = (θ/360) × πr²

**Derivation:**
- Proportional to angle: (θ/2π) × total area
- Total area: πr²
- Sector area: (θ/2π) × πr² = (1/2)r²θ

**Example:**
- Radius: r = 5, Angle: θ = 60° = π/3 radians
- Sector area: A = (1/2)(5)²(π/3) = (25π)/6 ≈ 13.09 square units
- Or: A = (60/360) × π(5)² = (1/6) × 25π = 25π/6 ✓

**Why Sector Area Matters:**
- **Pie charts:** Calculate area of each slice
- **Graphics:** Drawing circular progress indicators
- **Real-world:** Slicing circular objects

**5. Equation of a Circle: Coordinate Representation**

**Standard Form:** (x - h)² + (y - k)² = r²

**Components:**
- (h, k): Center coordinates
- r: Radius
- (x, y): Any point on circle

**Derivation:**
- Distance from (x, y) to (h, k) equals r
- Distance formula: √[(x-h)² + (y-k)²] = r
- Square both sides: (x-h)² + (y-k)² = r²

**Example:**
- Center: (3, 4), Radius: 5
- Equation: (x - 3)² + (y - 4)² = 25
- Check: Point (0, 0): (0-3)² + (0-4)² = 9 + 16 = 25 ✓

**Why Circle Equation Matters:**
- **Graphics:** Determine if point is inside/on/outside circle
- **Collision detection:** Check circle-point, circle-circle intersection
- **Algorithms:** Many geometric algorithms use this form

**Applications in Computer Science:**

**1. Computer Graphics: Drawing Curves**

**Circle Rendering:**
- Rasterization: Convert circle equation to pixels
- Bresenham's algorithm: Efficient circle drawing
- Anti-aliasing: Smooth circle edges

**Why Circles:**
- **Curves:** Basis for many curved shapes
- **Sprites:** Circular objects in games
- **UI elements:** Buttons, progress indicators

**2. Game Development: Circular Boundaries**

**Circular Hitboxes:**
- Simple collision detection: Check distance to center
- Efficient: O(1) point-circle collision check
- Common: Many game objects approximated as circles

**Orbital Motion:**
- Planets, satellites follow circular/elliptical paths
- Calculate positions using angle and radius
- Smooth, predictable motion

**3. Path Planning: Circular Paths**

**Circular Routes:**
- Vehicles following circular paths
- Robots navigating circular obstacles
- Smooth turns and curves

**Why Circular Paths:**
- **Smooth:** Continuous curvature
- **Predictable:** Easy to calculate positions
- **Natural:** Many real-world paths are circular

**4. Collision Detection: Efficient Checks**

**Point-Circle Collision:**
- Check: distance(point, center) ≤ radius
- Very fast: O(1) operation
- Used extensively in games

**Circle-Circle Collision:**
- Check: distance(center1, center2) ≤ (r1 + r2)
- Simple distance calculation
- Much faster than polygon collision

**5. User Interface: Circular Elements**

**Progress Indicators:**
- Circular progress bars
- Loading spinners
- Percentage displays

**Buttons and Icons:**
- Circular buttons
- Icon containers
- Modern UI design

**Why This Matters:**

Circles appear in:
- **Computer Graphics:** Drawing curves, sprites, UI elements
- **Game Development:** Circular boundaries, orbits, hitboxes
- **Path Planning:** Circular paths, smooth navigation
- **Collision Detection:** Efficient circular collision checks
- **User Interface:** Progress indicators, buttons, icons
- **Real-world Modeling:** Wheels, orbits, circular objects

**Next Steps:**

Understanding circles provides the foundation for working with curves and circular motion. In the next lesson, we'll explore area and perimeter formulas for various shapes, building on our understanding of circles and triangles.`,
					CodeExamples: `# Working with circles in Python

import math

class Circle:
    def __init__(self, radius, center=(0, 0)):
        """Create circle with radius and center."""
        self.radius = radius
        self.center = center
    
    def diameter(self):
        """Calculate diameter."""
        return 2 * self.radius
    
    def circumference(self):
        """Calculate circumference."""
        return 2 * math.pi * self.radius
    
    def area(self):
        """Calculate area."""
        return math.pi * self.radius ** 2
    
    def arc_length(self, angle_degrees):
        """Calculate arc length for given angle in degrees."""
        return (angle_degrees / 360) * self.circumference()
    
    def sector_area(self, angle_degrees):
        """Calculate sector area for given angle in degrees."""
        return (angle_degrees / 360) * self.area()
    
    def point_on_circle(self, angle_degrees):
        """Get point on circle at given angle."""
        angle_rad = math.radians(angle_degrees)
        x = self.center[0] + self.radius * math.cos(angle_rad)
        y = self.center[1] + self.radius * math.sin(angle_rad)
        return (x, y)
    
    def contains_point(self, point):
        """Check if point is inside or on circle."""
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        distance = math.sqrt(dx**2 + dy**2)
        return distance <= self.radius

# Example circle
circle = Circle(5)
print(f"Circle with radius 5:")
print(f"  Diameter: {circle.diameter()}")
print(f"  Circumference: {circle.circumference():.2f}")
print(f"  Area: {circle.area():.2f}")

# Arc and sector
angle = 90
print(f"\nFor {angle}° arc:")
print(f"  Arc length: {circle.arc_length(angle):.2f}")
print(f"  Sector area: {circle.sector_area(angle):.2f}")

# Points on circle
for angle in [0, 90, 180, 270]:
    point = circle.point_on_circle(angle)
    print(f"  Point at {angle}°: ({point[0]:.2f}, {point[1]:.2f})")

# Check if point is in circle
test_point = (3, 4)
print(f"\nPoint {test_point} in circle: {circle.contains_point(test_point)}")`,
				},
				{
					Title: "Area and Perimeter Formulas",
					Content: `**Introduction: Measuring Space and Boundaries**

Area and perimeter are fundamental concepts in geometry. Area measures the space inside a shape, while perimeter measures the distance around it. These calculations are essential in computer graphics, game development, UI design, and many other applications.

**Understanding Area and Perimeter**

**Area:** The amount of space inside a 2D shape, measured in square units (e.g., square meters, square pixels).

**Perimeter:** The distance around the boundary of a 2D shape, measured in linear units (e.g., meters, pixels).

**Key Difference:**
- **Area:** 2D measurement (length × width)
- **Perimeter:** 1D measurement (sum of side lengths)

**Rectangle: The Fundamental Shape**

**Definition:** A quadrilateral with four right angles and opposite sides equal.

**Perimeter Formula:** P = 2(l + w)
- Where l = length, w = width
- Add all four sides: l + w + l + w = 2(l + w)

**Area Formula:** A = l × w
- Multiply length by width
- Counts unit squares that fit inside

**Example:**
- Length: l = 5, Width: w = 3
- Perimeter: P = 2(5 + 3) = 2(8) = 16 units
- Area: A = 5 × 3 = 15 square units

**Why Rectangles Matter:**
- **Most common shape:** Appears everywhere in computing
- **Screen coordinates:** Pixels form rectangular grid
- **Bounding boxes:** Used to represent object boundaries
- **UI elements:** Buttons, windows, panels are rectangular

**Square: Perfect Rectangle**

**Definition:** A rectangle with all sides equal (special case of rectangle).

**Perimeter Formula:** P = 4s
- Where s = side length
- All four sides equal, so P = s + s + s + s = 4s

**Area Formula:** A = s²
- Side squared
- Special case: l = w = s, so A = s × s = s²

**Example:**
- Side: s = 4
- Perimeter: P = 4(4) = 16 units
- Area: A = 4² = 16 square units

**Why Squares Matter:**
- **Optimal shape:** Maximum area for given perimeter
- **Grid systems:** Basis for pixel grids, tile maps
- **Simplification:** Simplest shape for many algorithms
- **Symmetry:** Maximum symmetry simplifies calculations

**Parallelogram: Slanted Rectangle**

**Definition:** A quadrilateral with opposite sides parallel.

**Properties:**
- Opposite sides equal and parallel
- Opposite angles equal
- Diagonals bisect each other

**Perimeter Formula:** P = 2(a + b)
- Where a, b are lengths of adjacent sides
- Opposite sides equal, so P = 2a + 2b = 2(a + b)

**Area Formula:** A = base × height
- Base: length of one side
- Height: perpendicular distance from base to opposite side
- NOT side × side (that's only for rectangles!)

**Example:**
- Base: 6, Height: 4
- Area: A = 6 × 4 = 24 square units
- Note: Side lengths don't matter, only base and height!

**Why Parallelograms Matter:**
- **Generalization:** Rectangles are special parallelograms
- **Transformations:** Parallelograms result from shearing rectangles
- **Graphics:** Used in 2D transformations and projections

**Trapezoid: One Pair of Parallel Sides**

**Definition:** A quadrilateral with exactly one pair of parallel sides.

**Properties:**
- Two bases: parallel sides (b₁ and b₂)
- Two legs: non-parallel sides
- Height: perpendicular distance between bases

**Perimeter Formula:** P = sum of all four sides
- Add: b₁ + b₂ + leg₁ + leg₂
- No simple formula (depends on all sides)

**Area Formula:** A = (1/2)(b₁ + b₂)h
- Average of bases times height
- Can think of it as "average width × height"

**Derivation:**
- Can split into rectangle and two triangles
- Or use formula: average of parallel sides × height

**Example:**
- Base 1: b₁ = 5, Base 2: b₂ = 7, Height: h = 4
- Area: A = (1/2)(5 + 7)(4) = (1/2)(12)(4) = 24 square units

**Why Trapezoids Matter:**
- **Common shape:** Appears in many real-world objects
- **Approximation:** Can approximate curves with trapezoids
- **Numerical integration:** Trapezoidal rule for integration

**Circle: Perfect Symmetry**

**Definition:** Set of all points equidistant from center.

**Circumference Formula:** C = 2πr = πd
- Where r = radius, d = diameter
- Distance around the circle
- π ≈ 3.14159...

**Area Formula:** A = πr²
- Radius squared times π
- Can derive using calculus or geometric methods

**Example:**
- Radius: r = 3
- Circumference: C = 2π(3) = 6π ≈ 18.85 units
- Area: A = π(3)² = 9π ≈ 28.27 square units

**Why Circles Matter:**
- **Curves:** Basis for many curved shapes
- **Collision detection:** Circular hitboxes are efficient
- **Graphics:** Used extensively in rendering

**Triangle: Simplest Polygon**

**Definition:** Polygon with three sides and three angles.

**Perimeter Formula:** P = a + b + c
- Sum of all three sides
- Simple addition

**Area Formulas:**

**1. Base and Height:** A = (1/2) × base × height
- Works for any triangle
- Height must be perpendicular to base

**2. Heron's Formula:** A = √[s(s-a)(s-b)(s-c)]
- Where s = (a + b + c)/2 (semi-perimeter)
- Use when you know all three sides but not height

**Example (Base and Height):**
- Base: 6, Height: 4
- Area: A = (1/2) × 6 × 4 = 12 square units

**Example (Heron's Formula):**
- Sides: a = 3, b = 4, c = 5
- Semi-perimeter: s = (3 + 4 + 5)/2 = 6
- Area: A = √[6(6-3)(6-4)(6-5)] = √[6 × 3 × 2 × 1] = √36 = 6

**Why Triangles Matter:**
- **Fundamental:** Simplest polygon
- **Graphics:** All polygons decomposed into triangles
- **Meshes:** 3D surfaces represented as triangle meshes

**Applications in Computer Science:**

**1. Graphics Rendering: Filling Shapes**

**Rasterization:**
- Convert shapes to pixels
- Calculate which pixels are inside shape (area)
- Fill those pixels with color

**Bounding Boxes:**
- Calculate rectangular bounds around shape
- Used for culling (skip rendering if outside view)
- Fast rejection test

**Why This Matters:**
- **Performance:** Efficient rendering requires area calculations
- **Optimization:** Bounding boxes speed up rendering
- **Clipping:** Determine what's visible on screen

**2. Game Development: Hitboxes and Collision**

**Hitbox Calculation:**
- Define collision boundaries using shapes
- Calculate area for damage zones
- Check if objects overlap (area intersection)

**Collision Detection:**
- Check if bounding boxes overlap
- More precise: check if actual shapes overlap
- Area calculations determine overlap amount

**Why This Matters:**
- **Gameplay:** Accurate collision detection essential
- **Performance:** Simple shapes (rectangles, circles) are fast
- **Realism:** More complex shapes for more realistic collisions

**3. UI Design: Layout Calculations**

**Layout Systems:**
- Calculate available space (area)
- Distribute space among UI elements
- Ensure elements fit (perimeter/area constraints)

**Responsive Design:**
- Calculate areas for different screen sizes
- Maintain aspect ratios
- Scale elements proportionally

**Why This Matters:**
- **User Experience:** Proper layout essential
- **Adaptability:** Works on different screen sizes
- **Efficiency:** Optimize space usage

**4. Image Processing: Region Analysis**

**Region Properties:**
- Calculate area of regions (blobs)
- Measure perimeter of boundaries
- Analyze shape properties

**Feature Extraction:**
- Area: size of objects
- Perimeter: boundary complexity
- Area/perimeter ratio: compactness measure

**Why This Matters:**
- **Computer Vision:** Identify and classify objects
- **Medical Imaging:** Analyze tissue regions
- **Quality Control:** Detect defects in manufacturing

**5. Optimization Problems**

**Maximum Area Problems:**
- Given fixed perimeter, maximize area
- Example: Rectangle with P = 20, maximize area
- Solution: Square (s = 5, A = 25)

**Minimum Perimeter Problems:**
- Given fixed area, minimize perimeter
- Example: Rectangle with A = 16, minimize perimeter
- Solution: Square (s = 4, P = 16)

**Why This Matters:**
- **Resource Optimization:** Minimize material (perimeter) for given area
- **Algorithm Design:** Optimize space/time trade-offs
- **Real-world:** Fencing, packaging, layout design

**Common Mistakes:**

**Mistake 1: Confusing Area and Perimeter**
- Area: space inside (square units)
- Perimeter: distance around (linear units)
- Don't mix them up!

**Mistake 2: Using Wrong Height**
- For parallelogram/trapezoid, height must be perpendicular
- Not the slanted side length!
- Always use perpendicular distance

**Mistake 3: Forgetting Units**
- Area: square units (m², cm², px²)
- Perimeter: linear units (m, cm, px)
- Keep units consistent!

**Why This Matters:**

Area and perimeter calculations are used in:
- **Graphics Rendering:** Filling shapes, calculating bounds, culling
- **Game Development:** Hitboxes, collision detection, damage zones
- **UI Design:** Layout calculations, responsive design, space distribution
- **Image Processing:** Region analysis, feature extraction, object detection
- **Optimization:** Maximizing/minimizing area or perimeter
- **Real-world Applications:** Architecture, manufacturing, design

**Next Steps:**

Mastering area and perimeter formulas provides the foundation for working with 2D shapes in computer graphics and game development. In the next lesson, we'll explore the Pythagorean theorem, which connects geometry with distance calculations.`,
					CodeExamples: `# Area and perimeter calculations

import math

def rectangle_perimeter(length, width):
    return 2 * (length + width)

def rectangle_area(length, width):
    return length * width

def square_perimeter(side):
    return 4 * side

def square_area(side):
    return side ** 2

def parallelogram_area(base, height):
    return base * height

def trapezoid_area(base1, base2, height):
    return 0.5 * (base1 + base2) * height

def circle_circumference(radius):
    return 2 * math.pi * radius

def circle_area(radius):
    return math.pi * radius ** 2

def triangle_area_heron(a, b, c):
    """Calculate triangle area using Heron's formula."""
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))

# Examples
print("Rectangle (5 × 3):")
print(f"  Perimeter: {rectangle_perimeter(5, 3)}")
print(f"  Area: {rectangle_area(5, 3)}")

print("\nSquare (side = 4):")
print(f"  Perimeter: {square_perimeter(4)}")
print(f"  Area: {square_area(4)}")

print("\nCircle (radius = 3):")
print(f"  Circumference: {circle_circumference(3):.2f}")
print(f"  Area: {circle_area(3):.2f}")

print("\nTriangle (sides 3, 4, 5):")
print(f"  Area: {triangle_area_heron(3, 4, 5):.2f}")

print("\nTrapezoid (bases 5, 7, height 4):")
print(f"  Area: {trapezoid_area(5, 7, 4)}")`,
				},
				{
					Title: "Pythagorean Theorem and Applications",
					Content: `**Introduction: The Most Famous Theorem**

The Pythagorean theorem is one of the most famous and useful theorems in mathematics. It relates the three sides of a right triangle in a simple yet powerful way. This theorem is fundamental to distance calculations, which are essential in computer graphics, game development, and many algorithms.

**Understanding the Pythagorean Theorem**

**Statement:** For any right triangle, the square of the length of the hypotenuse equals the sum of squares of the lengths of the two legs.

**Formula:** a² + b² = c²

**Components:**
- **a, b:** Legs (the two shorter sides)
- **c:** Hypotenuse (the longest side, opposite the right angle)
- **Right angle:** The 90° angle that makes the triangle right

**Visual Understanding:**
- Draw squares on each side of the right triangle
- Area of square on hypotenuse = sum of areas of squares on legs
- This geometric relationship is the basis of the theorem

**Why It Works: Proof Concept**

**Geometric Proof:**
- Construct squares on each side of the right triangle
- Show that area of large square (on hypotenuse) equals sum of two smaller squares
- Can be done by rearranging pieces or using similar triangles

**Algebraic Understanding:**
- The theorem is a relationship between side lengths
- Works for any right triangle, regardless of size
- Fundamental property of Euclidean geometry

**Finding Missing Sides: Practical Applications**

**1. Finding the Hypotenuse**

**Formula:** c = √(a² + b²)

**Steps:**
1. Square both legs: a² and b²
2. Add them: a² + b²
3. Take square root: √(a² + b²)

**Example:**
- Legs: a = 3, b = 4
- Calculate: a² = 9, b² = 16
- Sum: 9 + 16 = 25
- Hypotenuse: c = √25 = 5

**Verification:** 3² + 4² = 9 + 16 = 25 = 5² ✓

**2. Finding a Leg**

**Formula:** a = √(c² - b²)

**Steps:**
1. Square hypotenuse and known leg: c² and b²
2. Subtract: c² - b²
3. Take square root: √(c² - b²)

**Example:**
- Hypotenuse: c = 5, Leg: b = 4
- Calculate: c² = 25, b² = 16
- Difference: 25 - 16 = 9
- Missing leg: a = √9 = 3

**Verification:** 3² + 4² = 9 + 16 = 25 = 5² ✓

**Important Note:**
- Make sure c > b (hypotenuse is longest side)
- If c² - b² is negative, triangle is not valid
- Always check: a² + b² = c²

**Pythagorean Triples: Integer Solutions**

**Definition:** Sets of three positive integers (a, b, c) that satisfy a² + b² = c².

**Why They Matter:**
- Exact integer solutions (no decimals)
- Useful in many applications
- Common in geometry problems

**Common Triples:**

**1. (3, 4, 5) - The Most Famous**
- Check: 3² + 4² = 9 + 16 = 25 = 5² ✓
- Multiples: (6, 8, 10), (9, 12, 15), (12, 16, 20), etc.
- All multiples are also Pythagorean triples!

**2. (5, 12, 13)**
- Check: 5² + 12² = 25 + 144 = 169 = 13² ✓

**3. (8, 15, 17)**
- Check: 8² + 15² = 64 + 225 = 289 = 17² ✓

**4. (7, 24, 25)**
- Check: 7² + 24² = 49 + 576 = 625 = 25² ✓

**Generating More Triples:**
- Use formulas: a = m² - n², b = 2mn, c = m² + n²
- Where m > n > 0 are integers
- Example: m = 2, n = 1 → (3, 4, 5)

**Distance Formula: 2D Coordinates**

**Formula:** d = √[(x₂ - x₁)² + (y₂ - y₁)²]

**Derivation from Pythagorean Theorem:**
- Two points: (x₁, y₁) and (x₂, y₂)
- Horizontal distance: Δx = x₂ - x₁
- Vertical distance: Δy = y₂ - y₁
- These form legs of right triangle
- Hypotenuse is the distance: d = √(Δx² + Δy²)

**Step-by-Step Process:**

**Example:** Find distance from (0, 0) to (3, 4)

1. **Identify coordinates:**
   - Point 1: (x₁, y₁) = (0, 0)
   - Point 2: (x₂, y₂) = (3, 4)

2. **Calculate differences:**
   - Δx = 3 - 0 = 3
   - Δy = 4 - 0 = 4

3. **Square and add:**
   - Δx² = 9
   - Δy² = 16
   - Sum = 25

4. **Take square root:**
   - d = √25 = 5

**Result:** Distance = 5 units

**Why Distance Formula Matters:**
- **Fundamental:** Used everywhere in coordinate geometry
- **Graphics:** Calculate distances between objects
- **Games:** Measure distances for gameplay mechanics
- **Algorithms:** Many algorithms use distance calculations

**3D Distance Formula: Extending to Three Dimensions**

**Formula:** d = √[(x₂ - x₁)² + (y₂ - y₁)² + (z₂ - z₁)²]

**Derivation:**
- Extend 2D formula to include z-coordinate
- Same principle: sum of squares of differences
- Works in any number of dimensions!

**Example:**
- Point 1: (0, 0, 0)
- Point 2: (3, 4, 12)
- Distance: d = √[(3-0)² + (4-0)² + (12-0)²] = √[9 + 16 + 144] = √169 = 13

**Applications in Computer Science:**

**1. Distance Calculations in Games and Graphics**

**Object Positioning:**
- Calculate distance between game objects
- Determine if objects are close enough to interact
- Measure range for weapons, spells, etc.

**Example:**
- Player at (10, 20), Enemy at (15, 25)
- Distance: √[(15-10)² + (25-20)²] = √[25 + 25] = √50 ≈ 7.07
- If weapon range = 10, enemy is in range!

**Why This Matters:**
- **Gameplay:** Essential for many game mechanics
- **Performance:** Fast calculation (O(1))
- **Accuracy:** Precise distance measurements

**2. Collision Detection: Distance Between Objects**

**Point-Circle Collision:**
- Check if point is inside circle
- Calculate distance from point to center
- If distance ≤ radius, collision!

**Circle-Circle Collision:**
- Calculate distance between centers
- If distance ≤ (r₁ + r₂), circles overlap
- Very efficient collision check

**Example:**
- Circle 1: center (0, 0), radius 5
- Circle 2: center (8, 0), radius 4
- Distance: √[(8-0)² + (0-0)²] = 8
- Sum of radii: 5 + 4 = 9
- Since 8 < 9, circles overlap!

**Why This Matters:**
- **Performance:** Much faster than complex collision detection
- **Simplicity:** Easy to implement and understand
- **Common:** Used in many games and simulations

**3. Pathfinding Algorithms: Distance Heuristics**

**A* Algorithm:**
- Uses distance as heuristic function
- Estimates distance to goal
- Helps find optimal path

**Euclidean Distance:**
- Straight-line distance (as the crow flies)
- Good heuristic for pathfinding
- Admissible (never overestimates)

**Example:**
- Current position: (5, 5)
- Goal: (10, 15)
- Heuristic: √[(10-5)² + (15-5)²] = √[25 + 100] = √125 ≈ 11.18
- Guides search toward goal

**Why This Matters:**
- **Efficiency:** Helps find paths quickly
- **Optimality:** Finds shortest paths
- **Games:** Used in NPC pathfinding, navigation

**4. 3D Graphics: Calculating Distances in 3D Space**

**Camera Distance:**
- Calculate distance from camera to objects
- Determine rendering order (depth sorting)
- Cull objects too far away

**Lighting Calculations:**
- Distance affects light intensity
- Inverse square law: intensity ∝ 1/distance²
- Realistic lighting effects

**Example:**
- Light source at (0, 0, 0)
- Object at (3, 4, 12)
- Distance: √[9 + 16 + 144] = 13
- Light intensity: 1/13² = 1/169 ≈ 0.006

**Why This Matters:**
- **Realism:** Accurate 3D rendering
- **Performance:** Cull distant objects
- **Visual Quality:** Proper lighting and shadows

**5. Spatial Data Structures**

**KD-Trees:**
- Partition space using distances
- Efficient nearest neighbor search
- Used in many algorithms

**Quadtrees/Octrees:**
- Divide space based on distances
- Efficient spatial queries
- Used in collision detection, rendering

**Why This Matters:**
- **Performance:** Efficient spatial queries
- **Scalability:** Handle large numbers of objects
- **Algorithms:** Foundation for many spatial algorithms

**Common Mistakes:**

**Mistake 1: Using Wrong Sides**
- Make sure c is the hypotenuse (longest side)
- Hypotenuse is always opposite the right angle
- Don't confuse legs with hypotenuse!

**Mistake 2: Forgetting Square Root**
- a² + b² gives c², not c
- Must take square root to get c
- Common error: forgetting the √

**Mistake 3: Wrong Order in Distance Formula**
- (x₂ - x₁)², not (x₁ - x₂)² (doesn't matter, but be consistent)
- Make sure to square the differences
- Then add, then take square root

**Why This Matters:**

Pythagorean theorem is fundamental to:
- **Distance Calculations:** Used everywhere in games and graphics
- **Collision Detection:** Efficient distance-based collision checks
- **Pathfinding Algorithms:** A* and other algorithms use distance heuristics
- **3D Graphics:** Calculating distances in 3D space for rendering and lighting
- **Spatial Algorithms:** Foundation for many spatial data structures and queries
- **Real-world Applications:** GPS, navigation, robotics, computer vision

**Next Steps:**

Mastering the Pythagorean theorem provides the foundation for working with distances and coordinates. This theorem connects geometry with algebra and is essential for many computer science applications.`,
					CodeExamples: `# Pythagorean theorem applications

import math

def pythagorean_hypotenuse(a, b):
    """Find hypotenuse given two legs."""
    return math.sqrt(a**2 + b**2)

def pythagorean_leg(hypotenuse, other_leg):
    """Find leg given hypotenuse and other leg."""
    if hypotenuse <= other_leg:
        return None  # Invalid triangle
    return math.sqrt(hypotenuse**2 - other_leg**2)

def distance_2d(p1, p2):
    """Calculate distance between two 2D points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx**2 + dy**2)

def distance_3d(p1, p2):
    """Calculate distance between two 3D points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)

def is_pythagorean_triple(a, b, c):
    """Check if (a, b, c) is a Pythagorean triple."""
    sides = sorted([a, b, c])
    return abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < 1e-10

# Examples
print("Pythagorean theorem:")
print(f"Hypotenuse of triangle with legs 3, 4: {pythagorean_hypotenuse(3, 4)}")
print(f"Leg of triangle with hypotenuse 5, leg 4: {pythagorean_leg(5, 4)}")

print("\nDistance calculations:")
p1 = (0, 0)
p2 = (3, 4)
print(f"Distance from {p1} to {p2}: {distance_2d(p1, p2)}")

p3 = (0, 0, 0)
p4 = (3, 4, 5)
print(f"3D distance from {p3} to {p4}: {distance_3d(p3, p4):.2f}")

print("\nPythagorean triples:")
triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]
for triple in triples:
    print(f"{triple}: {is_pythagorean_triple(*triple)}")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          503,
			Title:       "Number Theory for CS",
			Description: "Advanced number theory concepts essential for computer science and cryptography.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "Modular Arithmetic",
					Content: `**Introduction: Clock Arithmetic and Beyond**

Modular arithmetic, also called "clock arithmetic," is a system where numbers wrap around after reaching a certain value called the modulus. It's fundamental to cryptography, computer science, and many algorithms. Understanding modular arithmetic is essential for secure communications, efficient computations, and working with cyclic structures.

**Understanding Modular Arithmetic: The Wrap-Around System**

Think of a clock: after 12 comes 1 again. This is modular arithmetic with modulus 12. In modular arithmetic, we only care about the remainder when dividing by the modulus.

**Notation:** a ≡ b (mod n)
- Read as: "a is congruent to b modulo n"
- Means: a and b have the same remainder when divided by n
- Equivalently: (a - b) is divisible by n

**Key Insight:** In mod n arithmetic, we work with numbers 0, 1, 2, ..., n-1. Any number outside this range is equivalent to its remainder when divided by n.

**Examples:**

**Example 1: 17 ≡ 2 (mod 5)**
- 17 ÷ 5 = 3 remainder 2
- So 17 ≡ 2 (mod 5)
- Also: 17 - 2 = 15, and 15 is divisible by 5 ✓

**Example 2: 23 ≡ 3 (mod 10)**
- 23 ÷ 10 = 2 remainder 3
- So 23 ≡ 3 (mod 10)
- The last digit determines equivalence mod 10

**Example 3: 15 ≡ 0 (mod 5)**
- 15 ÷ 5 = 3 remainder 0
- So 15 ≡ 0 (mod 5)
- Multiples of the modulus are equivalent to 0

**Visual Understanding: The Number Circle**

Imagine numbers arranged in a circle:
- Mod 5: 0, 1, 2, 3, 4, then back to 0
- Adding moves clockwise
- Subtracting moves counterclockwise
- This is why it's called "clock arithmetic"

**Properties of Modular Arithmetic**

**1. Addition: Sum and Then Reduce**

**Property:** (a + b) mod n = [(a mod n) + (b mod n)] mod n

**Why This Works:**
We can reduce numbers before or after operations. This is crucial for avoiding overflow in computations.

**Step-by-Step Example: (17 + 23) mod 5**

**Method 1: Add then reduce**
- 17 + 23 = 40
- 40 mod 5 = 0 (since 40 ÷ 5 = 8 remainder 0)

**Method 2: Reduce then add (more efficient)**
- 17 mod 5 = 2
- 23 mod 5 = 3
- (2 + 3) mod 5 = 5 mod 5 = 0
- Same result, but worked with smaller numbers!

**Why Method 2 Matters:**
- Prevents integer overflow
- More efficient for large numbers
- Essential in cryptography with huge numbers

**2. Multiplication: Multiply and Then Reduce**

**Property:** (a × b) mod n = [(a mod n) × (b mod n)] mod n

**Step-by-Step Example: (17 × 23) mod 5**

**Method 1: Multiply then reduce**
- 17 × 23 = 391
- 391 mod 5 = 1 (since 391 ÷ 5 = 78 remainder 1)

**Method 2: Reduce then multiply**
- 17 mod 5 = 2
- 23 mod 5 = 3
- (2 × 3) mod 5 = 6 mod 5 = 1
- Same result, avoided large multiplication!

**Why This Matters:**
- Cryptography uses numbers with hundreds of digits
- Can't compute 1000-digit × 1000-digit directly
- Reduce first, then multiply - much more efficient!

**3. Exponentiation: Fast Modular Exponentiation**

Computing aᵇ mod n efficiently is crucial for cryptography. The naive method (multiply a by itself b times) is too slow.

**Fast Exponentiation Algorithm (Binary Method):**

**Idea:** Use binary representation of exponent
- Write exponent in binary
- Square repeatedly, multiply when bit is 1
- Time complexity: O(log b) instead of O(b)

**Example: Compute 2¹³ mod 7**

**Step 1: Write 13 in binary**
- 13 = 1101₂ = 8 + 4 + 1

**Step 2: Build up powers**
- 2¹ mod 7 = 2
- 2² mod 7 = 4
- 2⁴ mod 7 = (2²)² mod 7 = 4² mod 7 = 16 mod 7 = 2
- 2⁸ mod 7 = (2⁴)² mod 7 = 2² mod 7 = 4

**Step 3: Combine based on binary**
- 2¹³ = 2⁸ × 2⁴ × 2¹
- 2¹³ mod 7 = (4 × 2 × 2) mod 7 = 16 mod 7 = 2

**Why This is Critical:**
- RSA encryption: Need to compute mᵉ mod n where e might be 65537
- Naive: 65537 multiplications (impossible!)
- Fast method: ~16 multiplications (feasible!)

**Modular Inverse: Division in Modular Arithmetic**

Division in modular arithmetic is tricky because we can't just divide. Instead, we multiply by the modular inverse.

**Definition:** The modular inverse of a (mod n) is a number b such that:
- a × b ≡ 1 (mod n)
- Denoted: a⁻¹ (mod n) or sometimes a^(-1) mod n

**Key Requirement:** The modular inverse exists if and only if gcd(a, n) = 1 (a and n are relatively prime).

**Why?** If gcd(a, n) > 1, then a shares factors with n, and no multiple of a can be ≡ 1 (mod n).

**Example: Find inverse of 3 (mod 10)**

**Method: Try values until we find one**
- 3 × 1 = 3 mod 10 = 3 ✗
- 3 × 2 = 6 mod 10 = 6 ✗
- 3 × 3 = 9 mod 10 = 9 ✗
- 3 × 4 = 12 mod 10 = 2 ✗
- 3 × 5 = 15 mod 10 = 5 ✗
- 3 × 6 = 18 mod 10 = 8 ✗
- 3 × 7 = 21 mod 10 = 1 ✓
- So 7 is the inverse of 3 (mod 10)

**Efficient Method: Extended Euclidean Algorithm**

For large numbers, use the Extended Euclidean Algorithm to find inverses efficiently. This is essential for cryptography.

**Applications in Computer Science:**

**1. Cryptography - RSA Encryption:**

RSA encryption relies entirely on modular arithmetic:
- **Key Generation:** Choose two large primes p and q
- **Public Key:** n = p × q, e (usually 65537)
- **Encryption:** c = mᵉ mod n
- **Decryption:** m = cᵈ mod n (where d is the private key)
- **Security:** Based on difficulty of factoring n

**Why Modular Arithmetic?**
- Keeps numbers bounded (prevents overflow)
- Provides mathematical structure for security
- Enables efficient computation

**2. Hash Functions:**

Many hash functions use modular arithmetic:
- **Division Method:** h(key) = key mod m (table size)
- **Multiplication Method:** Uses fractional parts and mod
- **Universal Hashing:** Uses mod with random parameters

**Example:** Hash table with size 11
- Key 23 → 23 mod 11 = 1 (slot 1)
- Key 45 → 45 mod 11 = 1 (collision!)
- Need collision resolution

**3. Random Number Generation:**

Pseudorandom number generators use modular arithmetic:
- **Linear Congruential Generator:** xₙ₊₁ = (a × xₙ + c) mod m
- **Choice of modulus:** Usually a large prime or power of 2
- **Period:** Maximum period is m (if parameters chosen correctly)

**Example:** Simple LCG
- xₙ₊₁ = (7 × xₙ + 3) mod 10
- Starting with x₀ = 0: 0, 3, 4, 1, 0, 3, ... (period 4)

**4. Cyclic Data Structures:**

Modular arithmetic is perfect for circular structures:
- **Circular Buffers:** index = (index + 1) mod size
- **Array Wrapping:** Access array elements cyclically
- **Ring Buffers:** Used in audio processing, networking

**Example: Circular Array**

    arr = [0, 1, 2, 3, 4]  # size 5
    index = 4
    next_index = (index + 1) % 5  # wraps to 0

**5. Checksums and Error Detection:**

Modular arithmetic detects errors:
- **ISBN Checksum:** Uses mod 11
- **Credit Card Numbers:** Luhn algorithm uses mod 10
- **CRC:** Cyclic Redundancy Check uses polynomial mod arithmetic

**6. Clock Arithmetic:**

The original motivation:
- **12-hour clock:** 13:00 ≡ 1:00 (mod 12)
- **24-hour clock:** 25:00 ≡ 1:00 (mod 24)
- **Days of week:** Day 8 ≡ Day 1 (mod 7)

**Common Mistakes:**

**Mistake 1: Forgetting to reduce mod n at the end**
- Wrong: (a + b) mod n = a + b (if a + b < n, but what if it's not?)
- Correct: Always reduce: (a + b) mod n

**Mistake 2: Assuming division works like normal**
- Wrong: (a / b) mod n = (a mod n) / (b mod n)
- Correct: Multiply by modular inverse: (a × b⁻¹) mod n

**Mistake 3: Not checking if inverse exists**
- Wrong: Assuming every number has an inverse mod n
- Correct: Check gcd(a, n) = 1 first

**Mistake 4: Confusing mod with remainder for negative numbers**
- In some languages: -5 mod 3 might be -2 or 1
- Standard: -5 mod 3 = 1 (always non-negative)
- Check your language's behavior!

**Why This Matters:**

Modular arithmetic is everywhere in computer science:
- **Security:** Cryptography depends on it
- **Efficiency:** Enables fast computations with large numbers
- **Data Structures:** Hash tables, circular buffers
- **Algorithms:** Many algorithms use mod for indexing and wrapping

**Next Steps:**

Mastering modular arithmetic provides the foundation for understanding cryptography, hash functions, and efficient algorithms. In the next lesson, we'll explore the Euclidean Algorithm, which efficiently computes GCDs and is essential for finding modular inverses.`,
					CodeExamples: `# Modular arithmetic in Python

def mod(a, n):
    """Calculate a mod n."""
    return a % n

def mod_add(a, b, n):
    """Calculate (a + b) mod n."""
    return (a + b) % n

def mod_multiply(a, b, n):
    """Calculate (a × b) mod n."""
    return (a * b) % n

def mod_power(base, exp, n):
    """Calculate base^exp mod n efficiently."""
    result = 1
    base = base % n
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % n
        exp = exp >> 1
        base = (base * base) % n
    return result

def mod_inverse(a, n):
    """Find modular inverse of a mod n using extended Euclidean algorithm."""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    gcd, x, _ = extended_gcd(a, n)
    if gcd != 1:
        return None  # Inverse doesn't exist
    return (x % n + n) % n

# Examples
print("Modular arithmetic:")
print(f"17 mod 5 = {mod(17, 5)}")
print(f"(17 + 23) mod 5 = {mod_add(17, 23, 5)}")
print(f"(17 × 23) mod 5 = {mod_multiply(17, 23, 5)}")
print(f"2^10 mod 7 = {mod_power(2, 10, 7)}")

# Modular inverse
inv = mod_inverse(3, 10)
print(f"\nModular inverse of 3 mod 10: {inv}")
if inv:
    print(f"Check: 3 × {inv} mod 10 = {mod_multiply(3, inv, 10)}")`,
				},
				{
					Title: "Euclidean Algorithm (Extended)",
					Content: `**Introduction: Finding the Greatest Common Divisor Efficiently**

The Euclidean algorithm is one of the oldest and most efficient algorithms in mathematics. It finds the greatest common divisor (GCD) of two numbers quickly, and its extended version solves even more problems - from modular inverses to RSA decryption. Understanding this algorithm is essential for cryptography and many computer science applications.

**What is GCD?**

**Greatest Common Divisor (GCD):** The largest positive integer that divides both numbers without remainder.

**Examples:**
- GCD(12, 18) = 6 (6 divides both 12 and 18)
- GCD(17, 13) = 1 (they're coprime - share no common factors)
- GCD(48, 18) = 6

**Why GCD Matters:**
- **Simplifying fractions:** GCD helps reduce fractions to lowest terms
- **Modular arithmetic:** GCD determines if numbers have modular inverses
- **Cryptography:** Essential for RSA and other cryptographic systems

**Basic Euclidean Algorithm: The Efficient Method**

**Key Insight:** GCD(a, b) = GCD(b, a mod b)

This is the fundamental property that makes the algorithm work.

**Why This Works:**
- If d divides both a and b, then d divides (a mod b)
- If d divides both b and (a mod b), then d divides a
- So the set of common divisors is the same!

**Algorithm Steps:**

1. **Given:** Two numbers a and b (assume a ≥ b)
2. **While b ≠ 0:**
   - Calculate r = a mod b
   - Set a = b, b = r
3. **Result:** GCD is the final value of a

**Step-by-Step Example: GCD(48, 18)**

**Iteration 1:**
- a = 48, b = 18
- 48 = 18 × 2 + 12 (remainder 12)
- So: GCD(48, 18) = GCD(18, 12)
- New: a = 18, b = 12

**Iteration 2:**
- a = 18, b = 12
- 18 = 12 × 1 + 6 (remainder 6)
- So: GCD(18, 12) = GCD(12, 6)
- New: a = 12, b = 6

**Iteration 3:**
- a = 12, b = 6
- 12 = 6 × 2 + 0 (remainder 0)
- So: GCD(12, 6) = GCD(6, 0) = 6
- Since b = 0, we stop

**Result:** GCD(48, 18) = 6 ✓

**Verification:**
- 48 = 6 × 8
- 18 = 6 × 3
- 6 is the largest number dividing both

**Time Complexity: Why It's So Fast**

**Complexity:** O(log min(a, b))

**Why So Efficient:**
- Each iteration roughly halves the numbers
- Fibonacci numbers give worst case (slowest reduction)
- Even worst case is logarithmic!

**Example:**
- GCD(1000000, 999999) takes only ~20 iterations
- Much faster than checking all divisors!

**Extended Euclidean Algorithm: Finding Coefficients**

**What It Does:** Finds GCD and also integers x, y such that:
- ax + by = gcd(a, b)

**Why This Matters:**
- **Modular inverses:** Find x such that ax ≡ 1 (mod b)
- **Linear Diophantine equations:** Solve ax + by = c
- **RSA decryption:** Essential for finding private keys

**How It Works:**

**Backward Substitution:**
- Work backwards through Euclidean algorithm steps
- Express each remainder in terms of original a and b
- Build up coefficients x and y

**Step-by-Step Example: Extended GCD(48, 18)**

**Forward Pass (Euclidean Algorithm):**
- 48 = 18 × 2 + 12  →  12 = 48 - 18 × 2
- 18 = 12 × 1 + 6   →  6 = 18 - 12 × 1
- 12 = 6 × 2 + 0    →  Stop (GCD = 6)

**Backward Pass (Find Coefficients):**

**Step 1:** Express 6 in terms of 18 and 12
- From: 6 = 18 - 12 × 1
- We have: 6 = 18 - 12 × 1

**Step 2:** Substitute 12 = 48 - 18 × 2
- 6 = 18 - (48 - 18 × 2) × 1
- 6 = 18 - 48 + 18 × 2
- 6 = 18 × 3 - 48 × 1
- 6 = 48 × (-1) + 18 × 3

**Result:** x = -1, y = 3
- Check: 48 × (-1) + 18 × 3 = -48 + 54 = 6 = GCD(48, 18) ✓

**Applications: Solving Real Problems**

**1. Modular Inverse: Finding Multiplicative Inverse**

**Problem:** Find x such that ax ≡ 1 (mod n)

**When It Exists:** Only if gcd(a, n) = 1 (a and n are coprime)

**Solution Using Extended Euclidean:**
- Find x, y such that ax + ny = 1
- Then: ax ≡ 1 (mod n)
- So x is the modular inverse!

**Example:** Find inverse of 7 mod 26
- gcd(7, 26) = 1 (they're coprime) ✓
- Extended Euclidean gives: 7 × 15 + 26 × (-4) = 1
- So: 7 × 15 ≡ 1 (mod 26)
- Inverse: 15

**Verification:** 7 × 15 = 105 ≡ 1 (mod 26) ✓

**Why This Matters:**
- **RSA decryption:** Need modular inverses for private keys
- **Cryptography:** Essential for many cryptographic protocols
- **Algorithms:** Used in many number-theoretic algorithms

**2. Linear Diophantine Equations: Integer Solutions**

**Problem:** Solve ax + by = c for integers x, y

**When Solutions Exist:** Only if gcd(a, b) divides c

**Solution Process:**
1. Find gcd(a, b) using Euclidean algorithm
2. Check if gcd(a, b) divides c
3. If yes, use extended Euclidean to find one solution
4. Generate all solutions using the general formula

**Example:** Solve 48x + 18y = 12
- gcd(48, 18) = 6
- Check: Does 6 divide 12? Yes! ✓
- Divide by 6: 8x + 3y = 2
- Extended Euclidean: 8 × (-1) + 3 × 3 = 1
- Multiply by 2: 8 × (-2) + 3 × 6 = 2
- One solution: x = -2, y = 6

**Why This Matters:**
- **Number theory:** Fundamental problem in number theory
- **Algorithms:** Used in various algorithms
- **Cryptography:** Related to solving modular equations

**3. RSA Key Generation: Essential for Cryptography**

**RSA Private Key:**
- Need to find d such that e × d ≡ 1 (mod φ(n))
- This is finding modular inverse!
- Extended Euclidean algorithm solves this

**Example:**
- e = 7, φ(n) = 26
- Need: 7d ≡ 1 (mod 26)
- Extended Euclidean: 7 × 15 + 26 × (-4) = 1
- So: d = 15

**Why This Matters:**
- **Security:** RSA security depends on this
- **Cryptography:** Used in HTTPS, SSL/TLS
- **Real-world:** Protects billions of transactions daily

**4. Rational Number Arithmetic: Simplifying Fractions**

**Problem:** Simplify fraction a/b to lowest terms

**Solution:**
- Find gcd(a, b) = d
- Simplified fraction: (a/d) / (b/d)

**Example:** Simplify 48/18
- gcd(48, 18) = 6
- Simplified: 48/18 = (48/6) / (18/6) = 8/3

**Why This Matters:**
- **Exact arithmetic:** Avoid floating-point errors
- **Algorithms:** Many algorithms use rational numbers
- **Mathematics:** Fundamental operation

**Common Mistakes:**

**Mistake 1: Not Handling Order**
- Algorithm works regardless of order
- But assuming a ≥ b makes it clearer
- Can swap if needed: GCD(a, b) = GCD(b, a)

**Mistake 2: Confusing Basic and Extended**
- Basic: Only finds GCD
- Extended: Finds GCD and coefficients
- Need extended for modular inverses!

**Mistake 3: Wrong Sign in Coefficients**
- Coefficients can be negative
- That's fine! ax + by = gcd works with negative values
- Example: 48 × (-1) + 18 × 3 = 6 is correct

**Why This Matters:**

Euclidean algorithm is crucial for:
- **Cryptography:** RSA key generation, modular inverses
- **Rational Number Arithmetic:** Simplifying fractions, exact computation
- **Simplifying Fractions:** Reducing to lowest terms
- **Solving Modular Equations:** Finding solutions to ax ≡ b (mod n)
- **Number Theory:** Fundamental tool in number theory
- **Algorithms:** Used in many computer science algorithms

**Next Steps:**

Mastering the Euclidean algorithm provides powerful tools for working with integers, modular arithmetic, and cryptography. The extended version opens up even more applications, from finding modular inverses to solving Diophantine equations.`,
					CodeExamples: `# Euclidean algorithm implementations

def gcd_basic(a, b):
    """Basic Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return abs(a)

def gcd_recursive(a, b):
    """Recursive Euclidean algorithm."""
    if b == 0:
        return abs(a)
    return gcd_recursive(b, a % b)

def extended_gcd(a, b):
    """
    Extended Euclidean algorithm.
    Returns (gcd, x, y) where ax + by = gcd(a, b)
    """
    if a == 0:
        return abs(b), 0, 1 if b >= 0 else -1
    
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    return gcd, x, y

def solve_diophantine(a, b, c):
    """
    Solve ax + by = c for integer solutions.
    Returns (x, y) if solution exists, None otherwise.
    """
    gcd, x0, y0 = extended_gcd(a, b)
    
    if c % gcd != 0:
        return None  # No solution
    
    # Scale solution
    x = x0 * (c // gcd)
    y = y0 * (c // gcd)
    
    return (x, y)

# Examples
print("GCD calculations:")
print(f"GCD(48, 18) = {gcd_basic(48, 18)}")
print(f"GCD(17, 5) = {gcd_basic(17, 5)}")
print(f"GCD(100, 25) = {gcd_basic(100, 25)}")

print("\nExtended Euclidean:")
gcd, x, y = extended_gcd(48, 18)
print(f"GCD(48, 18) = {gcd}")
print(f"48({x}) + 18({y}) = {48*x + 18*y}")

gcd2, x2, y2 = extended_gcd(17, 5)
print(f"\nGCD(17, 5) = {gcd2}")
print(f"17({x2}) + 5({y2}) = {17*x2 + 5*y2}")

# Diophantine equation
print("\nDiophantine equation 48x + 18y = 6:")
solution = solve_diophantine(48, 18, 6)
if solution:
    x, y = solution
    print(f"Solution: x = {x}, y = {y}")
    print(f"Check: 48({x}) + 18({y}) = {48*x + 18*y}")`,
				},
				{
					Title: "Prime Number Theorems",
					Content: `**Introduction: The Deep Structure of Primes**

Prime number theorems reveal profound truths about the distribution and properties of prime numbers. These theorems are not just mathematical curiosities - they form the foundation of modern cryptography and are essential for understanding the security of encryption systems.

**Fundamental Theorem of Arithmetic: The Unique Factorization**

This is one of the most important theorems in number theory, stating that every integer has a unique prime factorization.

**Statement:** Every integer greater than 1 can be uniquely expressed as a product of prime numbers (up to the order of factors).

**What "Unique" Means:**
- The same primes appear in every factorization
- Only the order can differ
- Example: 60 = 2² × 3 × 5 = 3 × 2² × 5 (same primes, different order)

**Why This Matters:**
- Ensures consistent mathematical results
- Allows efficient algorithms for GCD, LCM
- Forms basis for many number theory proofs
- Critical for cryptography

**Example: 60 = 2² × 3 × 5**
- This is the ONLY way to write 60 as product of primes (ignoring order)
- You can't write 60 = 2 × 2 × 3 × 5 × 1 (1 doesn't count)
- You can't write 60 = 4 × 3 × 5 (4 is not prime)

**Euclid's Theorem: Infinitely Many Primes**

One of the oldest and most elegant proofs in mathematics, dating back to around 300 BCE.

**Statement:** There are infinitely many prime numbers.

**Euclid's Proof (by Contradiction):**

**Step 1: Assume the opposite**
- Assume there are only finitely many primes
- List them: p₁, p₂, p₃, ..., pₖ

**Step 2: Construct a new number**
- Multiply all primes: N = p₁ × p₂ × ... × pₖ
- Add 1: M = N + 1

**Step 3: Analyze M**
- M is greater than any prime in our list
- M divided by any pᵢ gives remainder 1 (since M = p₁×...×pₖ + 1)
- So M is not divisible by any prime in our list

**Step 4: Reach contradiction**
- Case 1: M is prime → Contradiction! (M is prime but not in our list)
- Case 2: M is composite → M has a prime factor q
- But q cannot be any pᵢ (they all leave remainder 1)
- So q is a prime not in our list → Contradiction!

**Conclusion:** Our assumption was wrong. There must be infinitely many primes.

**Why This Proof is Beautiful:**
- Simple and elegant
- Uses only basic arithmetic
- One of the first proofs by contradiction
- Still taught 2300+ years later!

**Prime Number Theorem: The Distribution of Primes**

This theorem describes how primes are distributed among integers - one of the most important results in analytic number theory.

**Statement:** As n approaches infinity, the number of primes ≤ n is approximately n / ln(n)

**Notation:** π(n) ≈ n / ln(n)
- π(n) is the prime-counting function (number of primes ≤ n)
- ln(n) is the natural logarithm

**What This Means:**
- Primes become less frequent as numbers get larger
- But they never run out (infinitely many)
- The density of primes decreases like 1/ln(n)

**Step-by-Step Example: π(100)**

**Using the formula:**
- π(100) ≈ 100 / ln(100)
- ln(100) ≈ 4.605
- π(100) ≈ 100 / 4.605 ≈ 21.7

**Actual value:** π(100) = 25
- Primes ≤ 100: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
- Count: 25

**Accuracy:** Formula gives 21.7, actual is 25
- Error: ~13% (reasonable for n=100)
- Gets more accurate as n increases

**Why This Matters:**
- Estimates how many primes exist up to a given number
- Helps analyze prime generation algorithms
- Used in cryptography to estimate key space size
- Shows primes are "common enough" for practical use

**Twin Prime Conjecture: Pairs of Primes**

**Statement:** There are infinitely many pairs of primes that differ by 2.

**Examples:**
- (3, 5): difference is 2
- (11, 13): difference is 2
- (17, 19): difference is 2
- (29, 31): difference is 2

**Status:** Conjecture (not yet proven!)
- Believed to be true
- Very difficult to prove
- Recent progress: Infinitely many pairs differing by at most 246 (proven in 2014)
- Still open: Can we get it down to 2?

**Why It's Hard:**
- Requires understanding prime distribution very deeply
- Related to many other unsolved problems
- Would be a major breakthrough if proven

**Goldbach's Conjecture: Even Numbers as Sums of Primes**

**Statement:** Every even integer greater than 2 can be expressed as the sum of two primes.

**Examples:**
- 4 = 2 + 2
- 6 = 3 + 3
- 8 = 3 + 5
- 10 = 3 + 7 = 5 + 5
- 12 = 5 + 7
- 100 = 3 + 97 = 11 + 89 = 17 + 83 = ... (many ways!)

**Status:** Conjecture (not yet proven!)
- Verified for all even numbers up to 4 × 10¹⁸
- Believed to be true
- Very difficult to prove
- One of the oldest unsolved problems (1742)

**Why It's Hard:**
- Requires understanding how primes can combine
- Related to additive number theory
- Would prove something deep about prime structure

**Fermat's Little Theorem: A Powerful Tool**

A fundamental theorem in number theory with many applications.

**Statement:** If p is prime and gcd(a, p) = 1, then a^(p-1) ≡ 1 (mod p)

**What This Means:**
- For any number a not divisible by prime p
- Raising a to the (p-1) power gives remainder 1 when divided by p
- This is always true for primes!

**Step-by-Step Example: 3^6 mod 7**

**Given:** p = 7 (prime), a = 3, gcd(3, 7) = 1 ✓

**Calculate:**
- 3^6 = 729
- 729 mod 7 = 729 - 7×104 = 729 - 728 = 1 ✓

**Using the theorem:**
- 3^(7-1) = 3^6 ≡ 1 (mod 7) ✓
- Much faster than computing 3^6!

**Applications:**

**1. Modular Inverses:**
- If a^(p-1) ≡ 1 (mod p), then a × a^(p-2) ≡ 1 (mod p)
- So a^(p-2) is the inverse of a (mod p)
- Example: Inverse of 3 mod 7 is 3^5 = 243 ≡ 5 mod 7
- Check: 3 × 5 = 15 ≡ 1 mod 7 ✓

**2. Primality Testing:**
- If a^(n-1) ≢ 1 (mod n) for some a, then n is composite
- Basis for Fermat primality test
- Used in cryptography

**3. Fast Exponentiation:**
- Can reduce exponents modulo (p-1)
- a^k mod p = a^(k mod (p-1)) mod p (if gcd(a,p)=1)
- Speeds up modular exponentiation

**Miller-Rabin Primality Test: Practical Primality Testing**

A probabilistic algorithm for testing if a number is prime, based on Fermat's Little Theorem.

**How It Works:**

**Basic Idea:**
- If n is prime and gcd(a, n) = 1, then a^(n-1) ≡ 1 (mod n)
- If this fails for some a, n is definitely composite
- If it passes for many random a, n is probably prime

**Algorithm Steps:**

1. Write n-1 = 2^r × d (where d is odd)
2. Choose random a with 1 < a < n
3. Compute a^d mod n
4. If a^d ≡ ±1 (mod n), n might be prime
5. Square repeatedly: a^(2d), a^(4d), ..., a^((n-1)/2)
6. If we reach -1 before n-1, n might be prime
7. If we reach n-1 without seeing -1, n is composite

**Why It's Probabilistic:**
- Some composite numbers pass the test (called "pseudoprimes")
- But probability of error is very low
- Running multiple rounds reduces error probability exponentially

**Advantages:**
- Much faster than trial division for large numbers
- O(k log³ n) where k is number of rounds
- Practical for generating large primes (hundreds of digits)

**Applications in Cryptography:**
- RSA key generation needs large random primes
- Miller-Rabin can test 300-digit numbers quickly
- Used in OpenSSL, GnuPG, and other cryptographic libraries

**Why This Matters:**

Prime theorems are essential for:
- **Cryptography:** RSA encryption requires large primes
- **Security:** Understanding what makes encryption secure
- **Hash Functions:** Many use prime moduli
- **Random Number Generation:** Prime moduli in generators
- **Algorithm Design:** Many algorithms rely on prime properties

**Next Steps:**

Mastering prime theorems provides the foundation for understanding cryptographic security and number theory algorithms. In the next lesson, we'll explore factorization methods, which are the inverse operation and crucial for understanding why factoring is hard (which makes RSA secure).`,
					CodeExamples: `# Prime number theorems and applications

import math

def prime_counting_function(n):
    """Approximate number of primes ≤ n using Prime Number Theorem."""
    if n < 2:
        return 0
    return int(n / math.log(n))

def count_primes_actual(n):
    """Actually count primes ≤ n using Sieve of Eratosthenes."""
    if n < 2:
        return 0
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return sum(is_prime)

def fermats_little_theorem(a, p):
    """Verify Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p)."""
    if not is_prime_simple(p):
        return False
    if math.gcd(a, p) != 1:
        return False
    
    result = pow(a, p - 1, p)
    return result == 1

def is_prime_simple(n):
    """Simple primality test."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

# Prime Number Theorem
print("Prime Number Theorem:")
for n in [10, 100, 1000, 10000]:
    approx = prime_counting_function(n)
    actual = count_primes_actual(n)
    print(f"π({n}) ≈ {approx}, actual = {actual}")

# Fermat's Little Theorem
print("\nFermat's Little Theorem:")
test_cases = [(3, 7), (5, 11), (2, 13)]
for a, p in test_cases:
    result = fermats_little_theorem(a, p)
    power_result = pow(a, p - 1, p)
    print(f"{a}^({p}-1) mod {p} = {power_result}, theorem holds: {result}")

# Twin primes
def find_twin_primes(limit):
    """Find twin prime pairs up to limit."""
    primes = []
    for i in range(2, limit + 1):
        if is_prime_simple(i):
            primes.append(i)
    
    twin_pairs = []
    for i in range(len(primes) - 1):
        if primes[i+1] - primes[i] == 2:
            twin_pairs.append((primes[i], primes[i+1]))
    return twin_pairs

print("\nTwin primes up to 50:")
twins = find_twin_primes(50)
for pair in twins:
    print(f"{pair[0]} and {pair[1]}")`,
				},
				{
					Title: "Factorization Methods",
					Content: `**Introduction: The Reverse of Multiplication**

Factorization is the process of breaking a number into its prime factors. While multiplication is easy (multiply two large primes), factorization is computationally difficult - and this difficulty is what makes RSA encryption secure. Understanding factorization methods helps us appreciate why some problems are hard and why cryptography works.

**Understanding Factorization: Breaking Down Numbers**

Factorization means expressing a number as a product of smaller numbers, ideally prime numbers.

**Goal:** Given n, find primes p₁, p₂, ..., pₖ such that n = p₁ × p₂ × ... × pₖ

**Fundamental Theorem of Arithmetic guarantees:** This factorization is unique (up to order).

**Example:** 60 = 2² × 3 × 5
- This is the only way to factor 60 into primes
- Order doesn't matter: 60 = 3 × 2² × 5 is the same factorization

**Method 1: Trial Division - The Straightforward Approach**

The simplest factorization method: try dividing by each prime number.

**Algorithm:**

1. Start with smallest prime (2)
2. While current number is divisible by prime:
   - Divide by prime
   - Count how many times
   - Add prime to factors with that exponent
3. Move to next prime
4. Stop when current number becomes 1 or prime > √n

**Step-by-Step Example: Factor 60**

**Step 1: Try 2**
- 60 ÷ 2 = 30 (divisible, factor: 2¹)
- 30 ÷ 2 = 15 (divisible again, factor: 2²)
- 15 ÷ 2 = 7.5 (not divisible, move on)

**Step 2: Try 3**
- 15 ÷ 3 = 5 (divisible, factor: 3¹)
- 5 ÷ 3 = 1.67 (not divisible, move on)

**Step 3: Try 5**
- 5 ÷ 5 = 1 (divisible, factor: 5¹)
- Result is 1, we're done!

**Result:** 60 = 2² × 3 × 5 ✓

**Why Stop at √n?**

**Key Insight:** If n has a factor > √n, it must also have a factor < √n.

**Proof:**
- Suppose n = a × b where both a, b > √n
- Then a × b > √n × √n = n (contradiction!)
- So at least one factor must be ≤ √n

**Time Complexity:** O(√n)
- Need to check up to √n primes
- For each prime, may divide multiple times
- Overall: O(√n) operations

**Why This is Slow:**
- For n = 10¹², need to check up to 10⁶ primes
- For n = 10²⁴, need to check up to 10¹² primes
- Exponential growth in number of checks!

**Method 2: Pollard's Rho Algorithm - Probabilistic Speedup**

A more sophisticated algorithm that uses randomness and cycle detection.

**Key Idea:** Use a function f(x) = (x² + c) mod n to generate a sequence
- This sequence will eventually cycle (pigeonhole principle)
- The cycle reveals information about factors

**How It Works:**

1. **Choose random starting values:** x₀, c
2. **Generate sequence:** xᵢ₊₁ = (xᵢ² + c) mod n
3. **Detect cycle:** Use Floyd's cycle-finding algorithm
4. **Find factor:** When cycle detected, compute gcd(|xᵢ - xⱼ|, n)
5. **If gcd > 1:** We found a factor!

**Why It Works:**
- The sequence mod n cycles
- But mod p (where p is a factor) also cycles, possibly with different period
- When cycles align differently, we can detect p via gcd

**Time Complexity:** O(√p) where p is smallest prime factor
- Much better than O(√n) when p is small
- Example: n = p × q where p ≈ 10⁶, q ≈ 10¹⁸
- Trial division: O(10⁹) operations
- Pollard's Rho: O(10³) operations (much faster!)

**Method 3: Fermat's Factorization - For Special Cases**

Works well when factors are close together.

**Key Idea:** Express n as difference of two squares: n = a² - b² = (a - b)(a + b)

**Algorithm:**

1. Start with a = ⌈√n⌉ (smallest integer ≥ √n)
2. Check if a² - n is a perfect square
3. If yes: n = (a - b)(a + b) where b = √(a² - n)
4. If no: increment a and try again

**Step-by-Step Example: Factor 15**

**Step 1: Find starting point**
- √15 ≈ 3.87, so a = 4

**Step 2: Check a = 4**
- a² - n = 16 - 15 = 1
- Is 1 a perfect square? Yes! (1 = 1²)
- So b = 1

**Step 3: Factor**
- n = (a - b)(a + b) = (4 - 1)(4 + 1) = 3 × 5 ✓

**Why Factorization is Hard: The Security Foundation**

**The Asymmetry:**

**Easy Direction (Multiplication):**
- Given: p = 17, q = 19
- Compute: n = p × q = 323
- Time: O(1) - instant!

**Hard Direction (Factorization):**
- Given: n = 323
- Find: p and q such that p × q = 323
- Time: O(√n) = O(√323) ≈ O(18) for small numbers
- But for 300-digit numbers: O(10¹⁵⁰) - impossible!

**Why This Matters for RSA:**

**RSA Encryption:**
- Public key: n = p × q (product of two large primes)
- Private key: p and q (the factors)
- Encryption: Easy (just multiply p and q)
- Decryption: Hard (need to factor n)

**Security:** If factoring were easy, RSA would be broken!

**Best Known Algorithms: General Number Field Sieve (GNFS)**

**Current State:**
- Fastest known algorithm for large numbers
- Still sub-exponential but very slow
- Time complexity: O(exp((64/9)^(1/3) × (ln n)^(1/3) × (ln ln n)^(2/3)))

**What This Means:**
- For 100-digit numbers: Feasible (hours/days)
- For 200-digit numbers: Very difficult (years)
- For 300-digit numbers: Practically impossible (centuries)
- For 600-digit numbers: Impossible with current technology

**Why RSA Uses Large Numbers:**
- 1024-bit RSA: n is ~300 digits
- 2048-bit RSA: n is ~600 digits
- 4096-bit RSA: n is ~1200 digits
- Larger = more secure (but slower operations)

**Applications in Computer Science:**

**1. Cryptography:**
- **Breaking RSA:** Requires factoring the public key
- **Understanding Security:** Know what key sizes are safe
- **Post-Quantum Cryptography:** Need new methods when quantum computers can factor

**2. Primality Testing:**
- Some tests use partial factorization
- Understanding factors helps test primality

**3. Algorithm Design:**
- Some algorithms use prime factors
- Example: Fast Fourier Transform uses prime factorization

**4. Number Theory:**
- Many theorems require factorization
- Understanding factors helps solve number theory problems

**Common Mistakes:**

**Mistake 1: Not checking if number is already prime**
- Before factoring, check if n is prime
- If prime, factorization is just n itself
- Saves time!

**Mistake 2: Continuing past √n**
- Only need to check up to √n
- If no factor found by then, number is prime
- Don't waste time checking larger numbers!

**Mistake 3: Forgetting to check if factor is prime**
- After finding a factor, check if it's prime
- If not prime, factor it further
- Recursive factorization needed

**Why This Matters:**

Factorization is crucial for:
- **Cryptography:** Understanding why RSA is secure
- **Security Analysis:** Evaluating encryption strength
- **Algorithm Design:** Some algorithms use factorization
- **Number Theory:** Fundamental operation in mathematics

**Next Steps:**

Mastering factorization methods provides insight into computational complexity and cryptographic security. In the next lesson, we'll explore the Chinese Remainder Theorem, which uses factorization concepts to solve systems of congruences efficiently.`,
					CodeExamples: `# Factorization methods

import math
import random

def trial_division(n):
    """Factor n using trial division."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def pollard_rho(n):
    """Pollard's Rho algorithm for factorization."""
    if n == 1:
        return []
    if is_prime_simple(n):
        return [n]
    
    def f(x):
        return (x * x + 1) % n
    
    x = random.randint(2, n - 1)
    y = x
    d = 1
    
    while d == 1:
        x = f(x)
        y = f(f(y))
        d = math.gcd(abs(x - y), n)
    
    if d == n:
        return [n]  # Failed, try again
    
    return pollard_rho(d) + pollard_rho(n // d)

def fermats_factorization(n):
    """Fermat's factorization method."""
    if n % 2 == 0:
        return [2] + fermats_factorization(n // 2)
    
    a = math.ceil(math.sqrt(n))
    while a < n:
        b_squared = a * a - n
        b = int(math.sqrt(b_squared))
        if b * b == b_squared:
            factor1 = a - b
            factor2 = a + b
            if factor1 == 1:
                return [n]  # Prime
            return fermats_factorization(factor1) + fermats_factorization(factor2)
        a += 1
    return [n]  # Prime

def is_prime_simple(n):
    """Simple primality test."""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Examples
print("Trial Division:")
n = 60
factors = trial_division(n)
print(f"Factors of {n}: {factors}")

n2 = 323
factors2 = trial_division(n2)
print(f"Factors of {n2}: {factors2}")

print("\nFermat's Factorization:")
n3 = 15
factors3 = fermats_factorization(n3)
print(f"Factors of {n3}: {factors3}")

# Compare methods
print("\nComparing methods for n = 1001:")
print(f"Trial division: {trial_division(1001)}")
print(f"Fermat's method: {fermats_factorization(1001)}")`,
				},
				{
					Title: "Chinese Remainder Theorem",
					Content: `**Introduction: Solving Systems Efficiently**

The Chinese Remainder Theorem (CRT) is a powerful tool for solving systems of simultaneous congruences. It's named after ancient Chinese mathematical texts, but its applications are thoroughly modern - from cryptography to parallel computation. Understanding CRT helps us work efficiently with large numbers and solve complex modular systems.

**Understanding the Chinese Remainder Theorem**

**Statement:** If we have a system of congruences with pairwise coprime moduli:
- x ≡ a₁ (mod n₁)
- x ≡ a₂ (mod n₂)
- ...
- x ≡ aₖ (mod nₖ)

And gcd(nᵢ, nⱼ) = 1 for all i ≠ j (moduli are pairwise coprime), then there exists a unique solution modulo N = n₁ × n₂ × ... × nₖ.

**What "Pairwise Coprime" Means:**
- Every pair of moduli has gcd = 1
- Example: {3, 5, 7} are pairwise coprime
- Not pairwise coprime: {4, 6} (gcd(4,6) = 2 ≠ 1)

**What "Unique Modulo N" Means:**
- There is exactly one solution in the range 0 to N-1
- All other solutions differ by multiples of N
- Example: If x = 23 is a solution mod 105, then 23, 128, 233, ... are all solutions

**Step-by-Step Solution Process:**

**Example: Solve the system:**
- x ≡ 2 (mod 3)
- x ≡ 3 (mod 5)
- x ≡ 2 (mod 7)

**Step 1: Verify Pairwise Coprimality**
- gcd(3, 5) = 1 ✓
- gcd(3, 7) = 1 ✓
- gcd(5, 7) = 1 ✓
- All pairwise coprime, CRT applies!

**Step 2: Calculate N (Product of All Moduli)**
- N = 3 × 5 × 7 = 105
- Solution will be unique mod 105

**Step 3: Calculate Nᵢ for Each Modulus**
- N₁ = N / n₁ = 105 / 3 = 35
- N₂ = N / n₂ = 105 / 5 = 21
- N₃ = N / n₃ = 105 / 7 = 15

**Why Nᵢ?** Nᵢ is divisible by all moduli except nᵢ, which helps isolate each congruence.

**Step 4: Find Modular Inverses**

**For each i, find inverse of Nᵢ mod nᵢ:**

**For i = 1 (modulus 3):**
- Need inverse of 35 mod 3
- 35 mod 3 = 2
- Need: 2 × ? ≡ 1 (mod 3)
- Try: 2 × 2 = 4 ≡ 1 (mod 3) ✓
- So inverse of 35 mod 3 is 2

**For i = 2 (modulus 5):**
- Need inverse of 21 mod 5
- 21 mod 5 = 1
- 1 × 1 = 1 ≡ 1 (mod 5) ✓
- So inverse of 21 mod 5 is 1

**For i = 3 (modulus 7):**
- Need inverse of 15 mod 7
- 15 mod 7 = 1
- 1 × 1 = 1 ≡ 1 (mod 7) ✓
- So inverse of 15 mod 7 is 1

**Step 5: Construct the Solution**

**Formula:** x = Σ(aᵢ × Nᵢ × inverse(Nᵢ)) mod N

**Calculate each term:**

**Term 1:** a₁ × N₁ × inv₁ = 2 × 35 × 2 = 140
**Term 2:** a₂ × N₂ × inv₂ = 3 × 21 × 1 = 63
**Term 3:** a₃ × N₃ × inv₃ = 2 × 15 × 1 = 30

**Sum:** 140 + 63 + 30 = 233

**Reduce mod N:** 233 mod 105 = 233 - 2×105 = 233 - 210 = 23

**Solution:** x ≡ 23 (mod 105)

**Step 6: Verification**

**Check each congruence:**
- 23 mod 3 = 23 - 7×3 = 23 - 21 = 2 ✓
- 23 mod 5 = 23 - 4×5 = 23 - 20 = 3 ✓
- 23 mod 7 = 23 - 3×7 = 23 - 21 = 2 ✓

All congruences satisfied! ✓

**Why This Formula Works:**

**Key Insight:** Each term aᵢ × Nᵢ × inv(Nᵢ) is:
- ≡ aᵢ (mod nᵢ) (satisfies the ith congruence)
- ≡ 0 (mod nⱼ) for j ≠ i (doesn't affect other congruences)

**Why?**
- Nᵢ is divisible by all nⱼ (j ≠ i), so Nᵢ ≡ 0 (mod nⱼ)
- Nᵢ × inv(Nᵢ) ≡ 1 (mod nᵢ), so aᵢ × Nᵢ × inv(Nᵢ) ≡ aᵢ (mod nᵢ)

**Applications in Computer Science:**

**1. Large Number Representation:**

**Problem:** Working with very large numbers (hundreds of digits) is slow
**Solution:** Represent number as remainders modulo smaller primes

**How It Works:**
- Instead of storing 300-digit number directly
- Store remainders mod several smaller primes (e.g., mod 2³¹-1, mod 2⁶¹-1, etc.)
- Perform operations on remainders (much faster!)
- Reconstruct original number using CRT when needed

**Example:**
- Large number: 12345678901234567890
- Represent as: (mod 97, mod 101, mod 103)
- Operations on remainders are much faster
- Reconstruct using CRT when needed

**2. RSA Decryption Optimization:**

**RSA Decryption:** m = cᵈ mod n
- Where n = p × q (product of two large primes)
- d is large, so cᵈ mod n is expensive

**CRT Optimization:**
- Instead: Compute cᵈ mod p and cᵈ mod q separately
- Both are faster (smaller moduli)
- Combine results using CRT
- Speedup: ~4x faster decryption!

**Why Faster:**
- Computing mod p: O(log³ p) vs mod n: O(log³ n)
- Since n = p × q and p ≈ q ≈ √n
- log p ≈ (1/2) log n
- (1/2)³ = 1/8, but we do two operations, so ~4x speedup

**3. Parallel Computation:**

**Divide Work:**
- Split large computation into smaller pieces
- Each processor works mod different prime
- Combine results using CRT
- Natural parallelization!

**Example:**
- Compute large product: a × b × c × ... × z
- Processor 1: Compute mod 97
- Processor 2: Compute mod 101
- Processor 3: Compute mod 103
- Combine using CRT

**4. Error Correction:**

**Redundant Representation:**
- Store number mod several moduli
- If one remainder is corrupted, can recover using others
- Used in some error-correcting codes

**Example:**
- Number mod (3, 5, 7)
- If mod 3 result is corrupted, can still recover from mod 5 and mod 7
- Provides fault tolerance

**5. Algorithm Optimization:**

**Modular Arithmetic:**
- Some algorithms work mod different primes
- Combine results using CRT
- Avoids working with huge numbers until final step

**Common Mistakes:**

**Mistake 1: Not checking pairwise coprimality**
- CRT only works if moduli are pairwise coprime
- If gcd(nᵢ, nⱼ) > 1 for some i ≠ j, CRT doesn't apply
- Must check this first!

**Mistake 2: Forgetting to reduce mod N**
- Solution formula gives a number, but need to reduce mod N
- Example: Got 233, but answer is 233 mod 105 = 23
- Always reduce to get canonical solution!

**Mistake 3: Incorrectly calculating inverses**
- Must find inverse of Nᵢ mod nᵢ, not mod N
- Check: Nᵢ × inv(Nᵢ) ≡ 1 (mod nᵢ)
- Verify your inverses!

**Why This Matters:**

CRT is used in:
- **Cryptography:** Speeding up RSA and other protocols
- **Fast Arithmetic:** Working with huge numbers efficiently
- **Parallel Computing:** Distributing computations
- **Error Correction:** Providing redundancy and fault tolerance
- **Algorithm Design:** Optimizing modular computations

**Next Steps:**

Mastering CRT provides powerful tools for efficient computation with large numbers and modular systems. In the next lesson, we'll explore applications in cryptography, showing how number theory concepts combine to create secure systems.`,
					CodeExamples: `# Chinese Remainder Theorem implementation

def mod_inverse(a, n):
    """Find modular inverse using extended Euclidean algorithm."""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    gcd, x, _ = extended_gcd(a, n)
    if gcd != 1:
        return None
    return (x % n + n) % n

def chinese_remainder_theorem(congruences):
    """
    Solve system of congruences using CRT.
    congruences: list of tuples (a_i, n_i) representing x ≡ a_i (mod n_i)
    """
    # Calculate N = product of all moduli
    N = 1
    for _, n in congruences:
        N *= n
    
    result = 0
    for a, n in congruences:
        N_i = N // n
        inv = mod_inverse(N_i, n)
        if inv is None:
            return None  # Moduli not pairwise coprime
        result += a * N_i * inv
    
    return result % N

# Example: x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
congruences = [(2, 3), (3, 5), (2, 7)]
solution = chinese_remainder_theorem(congruences)
print(f"System: x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)")
print(f"Solution: x ≡ {solution} (mod 105)")

# Verify
print("\nVerification:")
for a, n in congruences:
    print(f"  {solution} mod {n} = {solution % n} (should be {a})")

# Another example
print("\nAnother example:")
congruences2 = [(1, 3), (2, 5), (3, 7)]
solution2 = chinese_remainder_theorem(congruences2)
print(f"System: x ≡ 1 (mod 3), x ≡ 2 (mod 5), x ≡ 3 (mod 7)")
print(f"Solution: x ≡ {solution2} (mod 105)")

# Verify
for a, n in congruences2:
    print(f"  {solution2} mod {n} = {solution2 % n} (should be {a})")`,
				},
				{
					Title: "Applications in Cryptography",
					Content: `**Introduction: Mathematics Securing the Digital World**

Cryptography is the art and science of secure communication. Modern cryptography relies heavily on number theory - the mathematical properties of integers, primes, and modular arithmetic. Understanding these connections helps us appreciate why encryption works and how secure systems are designed.

**The Foundation: Why Number Theory?**

Number theory provides mathematical problems that are:
- **Easy in one direction:** Multiplication (easy)
- **Hard in reverse:** Factorization (hard)
- **Trapdoor functions:** Easy with secret, hard without

This asymmetry is the foundation of modern cryptography.

**RSA Encryption: The Most Widely Used Public-Key System**

RSA (Rivest-Shamir-Adleman) is the most common public-key encryption system, used in HTTPS, SSL/TLS, and secure communications worldwide.

**How RSA Works: Step-by-Step**

**Key Generation:**

**Step 1: Choose Two Large Primes**
- Select p and q (typically 300+ digits for security)
- These must be kept secret!
- Use probabilistic tests (Miller-Rabin) to find primes

**Step 2: Calculate Modulus and Totient**
- n = p × q (this becomes public)
- φ(n) = (p-1)(q-1) (Euler's totient function)
- φ(n) counts numbers < n relatively prime to n
- For n = p × q: φ(n) = (p-1)(q-1)

**Step 3: Choose Public Exponent**
- Select e such that 1 < e < φ(n) and gcd(e, φ(n)) = 1
- Common choice: e = 65537 (2¹⁶ + 1)
- Why 65537? It's prime, has only two 1-bits in binary (efficient exponentiation)

**Step 4: Calculate Private Exponent**
- Find d such that e × d ≡ 1 (mod φ(n))
- This is the modular inverse of e mod φ(n)
- Use Extended Euclidean Algorithm
- d is the private key (must be kept secret!)

**Step 5: Key Pair**
- **Public Key:** (n, e) - can be shared with everyone
- **Private Key:** (n, d) - must be kept secret

**Encryption Process:**

**Given:** Message m (as a number), Public key (n, e)

**Compute:** c = mᵉ mod n
- This is modular exponentiation
- Use fast exponentiation algorithm
- Result c is the ciphertext

**Example (Small Numbers for Illustration):**
- p = 17, q = 19, so n = 323
- φ(n) = 16 × 18 = 288
- Choose e = 5 (gcd(5, 288) = 1 ✓)
- Find d: 5 × d ≡ 1 (mod 288) → d = 173
- Encrypt m = 10: c = 10⁵ mod 323 = 100000 mod 323 = 40

**Decryption Process:**

**Given:** Ciphertext c, Private key (n, d)

**Compute:** m = cᵈ mod n
- This recovers the original message
- Works because: (mᵉ)ᵈ = m^(ed) ≡ m (mod n) by Euler's theorem

**Example:**
- Decrypt c = 40: m = 40¹⁷³ mod 323 = 10 ✓

**Why RSA Works: Mathematical Foundation**

**Euler's Theorem:** If gcd(a, n) = 1, then a^φ(n) ≡ 1 (mod n)

**For RSA:**
- We have: e × d ≡ 1 (mod φ(n))
- So: e × d = 1 + k × φ(n) for some k
- Then: (mᵉ)ᵈ = m^(ed) = m^(1 + kφ(n)) = m × (m^φ(n))^k
- If gcd(m, n) = 1: (m^φ(n))^k ≡ 1^k = 1 (mod n)
- Therefore: (mᵉ)ᵈ ≡ m (mod n) ✓

**Security: Why Factoring is Hard**

**Breaking RSA:** To decrypt without private key, need to find d
- To find d, need φ(n) = (p-1)(q-1)
- To find φ(n), need to factor n = p × q
- Factoring large numbers is computationally infeasible!

**Current Security:**
- 1024-bit RSA: n is ~300 digits, considered secure until ~2030
- 2048-bit RSA: n is ~600 digits, recommended for long-term security
- 4096-bit RSA: n is ~1200 digits, maximum security

**Diffie-Hellman Key Exchange: Secure Key Establishment**

Allows two parties to establish a shared secret over an insecure channel.

**How It Works:**

**Setup:** Agree on public parameters (g, p) where p is prime, g is generator

**Alice:**
- Chooses secret a, computes A = g^a mod p
- Sends A to Bob

**Bob:**
- Chooses secret b, computes B = g^b mod p
- Sends B to Alice

**Shared Secret:**
- Alice computes: s = B^a mod p = (g^b)^a mod p = g^(ab) mod p
- Bob computes: s = A^b mod p = (g^a)^b mod p = g^(ab) mod p
- Both get the same secret s!

**Security:** Based on Discrete Logarithm Problem
- Given g, p, and g^a mod p, finding a is hard
- Attacker sees A = g^a and B = g^b, but can't compute g^(ab)

**Elliptic Curve Cryptography (ECC): More Efficient**

Uses elliptic curves over finite fields instead of modular arithmetic.

**Advantages:**
- Smaller keys for same security level
- 256-bit ECC ≈ 3072-bit RSA security
- Faster operations, less memory
- Used in mobile devices, IoT

**Example:**
- RSA-2048: 2048-bit keys
- ECC equivalent: 256-bit keys
- 8x smaller keys!

**Hash Functions: One-Way Functions**

Hash functions use modular arithmetic to ensure uniform distribution.

**Properties:**
- **Deterministic:** Same input → same output
- **Fast:** Easy to compute
- **One-way:** Hard to reverse (find input from output)
- **Avalanche effect:** Small input change → large output change

**Examples:**
- **MD5:** 128-bit output (deprecated, insecure)
- **SHA-256:** 256-bit output (widely used)
- **SHA-512:** 512-bit output (higher security)

**Applications:**
- **Digital Signatures:** Sign hash of message, not message itself
- **Password Storage:** Store hash, not password
- **Blockchain:** Hash blocks to create chain
- **Data Integrity:** Detect tampering

**Random Number Generation: Pseudorandom Sequences**

**Linear Congruential Generator (LCG):**
- Formula: xₙ₊₁ = (a × xₙ + c) mod m
- Parameters: multiplier a, increment c, modulus m
- Seed: x₀ (starting value)

**Properties:**
- Period: At most m (if parameters chosen well)
- Fast: O(1) per number
- Not cryptographically secure (predictable)

**Cryptographically Secure PRNGs:**
- Use cryptographic primitives
- Unpredictable even with partial output
- Used for keys, nonces, salts

**Why Security Matters: Real-World Impact**

**Prime Size Determines Security:**

**1024-bit RSA:**
- Primes: ~300 digits each
- n: ~600 digits
- Current status: Deprecated, use only for legacy systems
- Vulnerable to advances in factoring

**2048-bit RSA:**
- Primes: ~600 digits each
- n: ~1200 digits
- Current status: Recommended standard
- Estimated secure until ~2030

**4096-bit RSA:**
- Primes: ~1200 digits each
- n: ~2400 digits
- Current status: Maximum security
- Slower but most secure

**Quantum Computing Threat:**
- Shor's algorithm can factor efficiently on quantum computers
- Would break RSA and many other systems
- Post-quantum cryptography needed for future

**Real-World Applications:**

**1. HTTPS/SSL/TLS:**
- Every secure website uses RSA or ECC
- Protects credit card numbers, passwords, personal data
- Millions of transactions daily rely on this

**2. Digital Signatures:**
- Verify authenticity of documents
- Used in software distribution, legal documents
- Based on RSA or ECC

**3. Blockchain and Cryptocurrencies:**
- Bitcoin uses ECC for addresses and signatures
- Ethereum uses ECC
- Entire cryptocurrency ecosystem depends on cryptography

**4. Secure Authentication:**
- Two-factor authentication
- Certificate-based authentication
- All rely on number theory

**5. Encrypted Communication:**
- Signal, WhatsApp use end-to-end encryption
- Based on number theory primitives
- Protects privacy of billions of users

**Why This Matters:**

Number theory is essential for:
- **Secure Communication:** Protecting data in transit
- **Data Privacy:** Encrypting sensitive information
- **Digital Identity:** Verifying who you are
- **Financial Security:** Protecting transactions
- **National Security:** Classified communications

**The Future: Post-Quantum Cryptography**

**Challenge:** Quantum computers threaten current systems
**Solution:** New algorithms resistant to quantum attacks
- Lattice-based cryptography
- Code-based cryptography
- Hash-based signatures
- All still use advanced number theory!

**Next Steps:**

Understanding cryptographic applications shows how abstract mathematics protects real-world systems. The security of the internet, financial systems, and personal privacy all depend on the number theory concepts we've explored.`,
					CodeExamples: `# Cryptographic applications (simplified examples)

import random

def generate_rsa_keys_simple(p, q):
    """Simplified RSA key generation (for educational purposes)."""
    n = p * q
    phi = (p - 1) * (q - 1)
    
    # Find e (usually 65537)
    e = 65537
    if math.gcd(e, phi) != 1:
        e = 3  # Fallback
    
    # Find d (modular inverse of e mod phi)
    def mod_inverse(a, m):
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        gcd, x, _ = extended_gcd(a, m)
        if gcd != 1:
            return None
        return (x % m + m) % m
    
    d = mod_inverse(e, phi)
    return (n, e), (n, d)

def rsa_encrypt(message, public_key):
    """RSA encryption."""
    n, e = public_key
    return pow(message, e, n)

def rsa_decrypt(ciphertext, private_key):
    """RSA decryption."""
    n, d = private_key
    return pow(ciphertext, d, n)

# Example RSA (small primes for demonstration)
p, q = 61, 53  # Small primes for demo
public, private = generate_rsa_keys_simple(p, q)
print(f"RSA Keys (p={p}, q={q}):")
print(f"Public key (n, e): {public}")
print(f"Private key (n, d): {private}")

message = 42
ciphertext = rsa_encrypt(message, public)
decrypted = rsa_decrypt(ciphertext, private)
print(f"\nMessage: {message}")
print(f"Encrypted: {ciphertext}")
print(f"Decrypted: {decrypted}")

# Linear Congruential Generator
class LCG:
    def __init__(self, seed, a=1664525, c=1013904223, m=2**32):
        self.state = seed
        self.a = a
        self.c = c
        self.m = m
    
    def next(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

print("\nLinear Congruential Generator:")
lcg = LCG(12345)
for i in range(10):
    print(f"Random {i+1}: {lcg.next()}")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          504,
			Title:       "Combinatorics and Probability",
			Description: "Learn counting principles, permutations, combinations, and probability fundamentals.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Permutations and Combinations",
					Content: `**Introduction: Counting Arrangements and Selections**

Permutations and combinations are fundamental counting techniques in combinatorics. They help us answer questions like "How many ways can we arrange objects?" and "How many ways can we choose objects?" Understanding when to use each is crucial for probability, algorithm analysis, and optimization problems.

**Permutations: Arrangements Where Order Matters**

A permutation is an arrangement of objects where the order is important. Changing the order creates a different permutation.

**Key Question:** "How many ways can we arrange r objects from n objects?"

**Formula:** P(n, r) = n! / (n - r)! = n × (n-1) × ... × (n-r+1)

**Why This Formula Works:**

**Step-by-Step Reasoning:**

**Example: Arrange 3 books from 5 books on a shelf**

1. **First position:** 5 choices (any of the 5 books)
2. **Second position:** 4 choices (one book already used)
3. **Third position:** 3 choices (two books already used)

**Total:** 5 × 4 × 3 = 60 arrangements

**General Pattern:**
- First: n choices
- Second: (n-1) choices
- Third: (n-2) choices
- ...
- r-th: (n-r+1) choices

**Product:** n × (n-1) × ... × (n-r+1) = n! / (n-r)!

**Using Factorial Notation:**
- P(n, r) = n! / (n-r)!
- Numerator: n! (arrangements of all n objects)
- Denominator: (n-r)! (we ignore arrangements of unused objects)

**Step-by-Step Example: P(5, 3)**

**Method 1: Direct multiplication**
- P(5, 3) = 5 × 4 × 3 = 60

**Method 2: Using factorial formula**
- P(5, 3) = 5! / (5-3)! = 5! / 2! = 120 / 2 = 60 ✓

**Special Cases:**

**1. Permutations of All Objects: P(n, n)**
- P(n, n) = n! / (n-n)! = n! / 0! = n! / 1 = n!
- Example: P(5, 5) = 5! = 120 (arrange all 5 books)

**2. Permutations of Zero Objects: P(n, 0)**
- P(n, 0) = n! / n! = 1
- There's exactly one way to arrange nothing (the empty arrangement)

**3. Permutations of One Object: P(n, 1)**
- P(n, 1) = n! / (n-1)! = n
- n ways to choose and arrange one object

**Real-World Examples:**

**Example 1: Password Arrangements**
- How many 4-digit PINs using digits 0-9?
- P(10, 4) = 10 × 9 × 8 × 7 = 5,040
- Order matters: 1234 ≠ 4321

**Example 2: Race Finishing Orders**
- 8 runners, how many ways for top 3?
- P(8, 3) = 8 × 7 × 6 = 336
- Order matters: 1st-2nd-3rd is different from 3rd-2nd-1st

**Combinations: Selections Where Order Doesn't Matter**

A combination is a selection of objects where the order is not important. Only which objects are selected matters, not their arrangement.

**Key Question:** "How many ways can we choose r objects from n objects?"

**Formula:** C(n, r) = n! / [r!(n-r)!] = P(n, r) / r!

**Why This Formula Works:**

**Key Insight:** Combinations are permutations divided by the number of ways to arrange the selected objects.

**Step-by-Step Reasoning:**

**Example: Choose 3 books from 5 books (order doesn't matter)**

1. **First, count permutations:** P(5, 3) = 60
2. **But each combination appears multiple times:**
   - Combination {A, B, C} appears as: ABC, ACB, BAC, BCA, CAB, CBA
   - That's 3! = 6 arrangements
3. **Divide by arrangements:** 60 / 6 = 10 combinations

**General Formula:**
- C(n, r) = P(n, r) / r! = [n! / (n-r)!] / r! = n! / [r!(n-r)!]

**Step-by-Step Example: C(5, 3)**

**Method 1: Using combination formula**
- C(5, 3) = 5! / [3!(5-3)!] = 5! / (3! × 2!) = 120 / (6 × 2) = 120 / 12 = 10

**Method 2: Using permutation then divide**
- P(5, 3) = 60
- C(5, 3) = 60 / 3! = 60 / 6 = 10 ✓

**Notation:**
- C(n, r) or (n choose r) or nCr or C(n, r)
- Read as "n choose r"

**Key Difference: Order Matters vs. Doesn't Matter**

**Permutations (Order Matters):**
- ABC ≠ ACB (different arrangements)
- "How many ways to arrange?"
- Example: Arranging books on a shelf

**Combinations (Order Doesn't Matter):**
- {A, B, C} = {C, B, A} (same selection)
- "How many ways to choose?"
- Example: Choosing a committee

**Visual Example:**

**Permutations of {A, B, C}:**
- ABC, ACB, BAC, BCA, CAB, CBA (6 permutations)

**Combinations of 3 from {A, B, C, D, E}:**
- {A, B, C}, {A, B, D}, {A, B, E}, {A, C, D}, {A, C, E}, {A, D, E}, {B, C, D}, {B, C, E}, {B, D, E}, {C, D, E} (10 combinations)

**Special Properties:**

**1. Symmetry: C(n, r) = C(n, n-r)**
- C(5, 3) = C(5, 2) = 10
- Choosing 3 is same as excluding 2
- Why: n! / [r!(n-r)!] = n! / [(n-r)!r!]

**2. Boundary Cases:**
- C(n, 0) = 1 (one way to choose nothing)
- C(n, n) = 1 (one way to choose everything)
- C(n, 1) = n (n ways to choose one)

**3. Pascal's Identity: C(n, r) = C(n-1, r-1) + C(n-1, r)**
- Forms Pascal's Triangle
- Useful for recursive computation

**Pascal's Triangle: A Visual Pattern**

Pascal's Triangle is a triangular array where each number is the sum of the two numbers directly above it.

**Construction:**

Row 0:          1
Row 1:        1   1
Row 2:      1   2   1
Row 3:    1   3   3   1
Row 4:  1   4   6   4   1

**Properties:**
- Row n gives C(n, 0), C(n, 1), ..., C(n, n)
- Each number is sum of two above: C(n, r) = C(n-1, r-1) + C(n-1, r)
- Row n sums to 2ⁿ
- Symmetric: left side mirrors right side

**Applications:**

**1. Binomial Expansion:**
- (x + y)ⁿ = Σ C(n, r) × x^(n-r) × y^r
- Coefficients come from Pascal's Triangle
- Example: (x + y)³ = x³ + 3x²y + 3xy² + y³

**2. Probability:**
- C(n, r) ways to get r successes in n trials
- Used in binomial probability: P(r successes) = C(n, r) × p^r × (1-p)^(n-r)

**Applications in Computer Science:**

**1. Algorithm Analysis:**
- **Search Space:** Count possible solutions
- **Brute Force:** Calculate total arrangements to check
- **Optimization:** Determine if exhaustive search is feasible

**Example:** Password cracking
- 4-digit PIN: P(10, 4) = 5,040 possibilities
- If checking 1 per second: ~84 minutes worst case

**2. Probability Calculations:**
- **Random Sampling:** Probability of specific combinations
- **Monte Carlo:** Count favorable outcomes
- **Statistical Testing:** Calculate test statistics

**Example:** Probability of getting 3 heads in 5 coin flips
- Total outcomes: 2⁵ = 32
- Favorable: C(5, 3) = 10 ways to get 3 heads
- Probability: 10/32 = 5/16

**3. Optimization Problems:**
- **Subset Selection:** Choose optimal subset
- **Resource Allocation:** Ways to allocate resources
- **Scheduling:** Count possible schedules

**Example:** Choose 5 servers from 20 for load balancing
- C(20, 5) = 15,504 possible configurations
- Need to evaluate which is optimal

**4. Search Space Analysis:**
- **Backtracking:** Count nodes in search tree
- **Branching Factor:** Calculate tree size
- **Pruning:** Determine if search is feasible

**Example:** 8-queens problem
- Naive: P(64, 8) = huge number (infeasible)
- With constraints: Much smaller, but still exponential

**5. Data Structures:**
- **Hash Tables:** Count possible hash collisions
- **Graph Theory:** Count paths, cycles, subgraphs
- **Set Operations:** Count subsets, intersections

**Common Mistakes:**

**Mistake 1: Using permutation when combination is needed**
- Wrong: "Choose 3 from 5" using P(5, 3) = 60
- Correct: Use C(5, 3) = 10 (order doesn't matter)

**Mistake 2: Using combination when permutation is needed**
- Wrong: "Arrange 3 books" using C(5, 3) = 10
- Correct: Use P(5, 3) = 60 (order matters)

**Mistake 3: Forgetting to divide by r! in combinations**
- Wrong: C(n, r) = P(n, r) (missing division)
- Correct: C(n, r) = P(n, r) / r!

**Mistake 4: Confusing n and r**
- n = total objects available
- r = objects to choose/arrange
- Keep track of which is which!

**Why This Matters:**

Combinatorics is essential for:
- **Algorithm Design:** Understanding problem complexity
- **Probability:** Calculating likelihoods
- **Optimization:** Evaluating solution spaces
- **Cryptography:** Key space analysis
- **Machine Learning:** Feature selection, model complexity

**Next Steps:**

Mastering permutations and combinations provides the foundation for probability theory, where we'll use these counting techniques to calculate probabilities of events. In the next lesson, we'll explore probability fundamentals, building on these combinatorial concepts.`,
					CodeExamples: `# Permutations and combinations in Python

import math

def factorial(n):
    """Calculate n!"""
    if n < 0:
        return None
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def permutation(n, r):
    """Calculate P(n, r) = n! / (n-r)!"""
    if r > n or r < 0:
        return 0
    return factorial(n) // factorial(n - r)

def combination(n, r):
    """Calculate C(n, r) = n! / [r!(n-r)!]"""
    if r > n or r < 0:
        return 0
    return factorial(n) // (factorial(r) * factorial(n - r))

# Examples
print("Permutations:")
print(f"P(5, 3) = {permutation(5, 3)}")
print(f"P(10, 2) = {permutation(10, 2)}")

print("\nCombinations:")
print(f"C(5, 3) = {combination(5, 3)}")
print(f"C(10, 2) = {combination(10, 2)}")

# Generate Pascal's triangle
def pascal_triangle(n):
    """Generate first n rows of Pascal's triangle."""
    triangle = []
    for i in range(n):
        row = []
        for j in range(i + 1):
            row.append(combination(i, j))
        triangle.append(row)
    return triangle

print("\nPascal's Triangle (first 7 rows):")
pascal = pascal_triangle(7)
for row in pascal:
    print(" ".join(f"{num:3}" for num in row))

# Compare: permutations vs combinations
print("\nDifference:")
items = ['A', 'B', 'C']
print(f"Permutations of {items} (order matters):")
from itertools import permutations
for p in permutations(items, 2):
    print(f"  {p}")

print(f"\nCombinations of {items} (order doesn't matter):")
from itertools import combinations
for c in combinations(items, 2):
    print(f"  {c}")`,
				},
				{
					Title: "Basic Counting Principles",
					Content: `**Introduction: Systematic Counting**

Counting principles provide systematic methods for determining the number of ways events can occur. These principles are fundamental to combinatorics, probability, and algorithm analysis. Mastering them helps us count efficiently without listing every possibility.

**Fundamental Counting Principle: The Multiplication Rule**

The most basic counting principle: when events occur in sequence, multiply the number of ways.

**Statement:** If there are m ways to do one thing and n ways to do another (independent), there are m × n ways to do both.

**Key Word:** "AND" - both events must occur

**Step-by-Step Example: Choosing an Outfit**

**Problem:** 3 shirts AND 4 pants = how many outfits?

**Reasoning:**
- For each of 3 shirts, can pair with any of 4 pants
- Shirt 1: 4 pant choices
- Shirt 2: 4 pant choices
- Shirt 3: 4 pant choices
- Total: 3 × 4 = 12 outfits

**Visual Representation:**

Shirt 1 -> Pant 1, 2, 3, 4 (4 choices)
Shirt 2 -> Pant 1, 2, 3, 4 (4 choices)
Shirt 3 -> Pant 1, 2, 3, 4 (4 choices)
Total: 3 * 4 = 12

**Generalization:** For k independent choices with n₁, n₂, ..., nₖ options:
- Total ways = n₁ × n₂ × ... × nₖ

**Example:** 3 shirts × 4 pants × 2 shoes = 24 outfits

**Addition Principle: The "OR" Rule**

When events are mutually exclusive (can't happen together), add the number of ways.

**Statement:** If there are m ways to do one thing OR n ways to do another (mutually exclusive), there are m + n total ways.

**Key Word:** "OR" - only one event occurs

**Step-by-Step Example: Choosing a Meal**

**Problem:** Choose pizza (5 types) OR pasta (3 types) = how many choices?

**Reasoning:**
- Can't have both pizza and pasta (mutually exclusive)
- Pizza options: 5
- Pasta options: 3
- Total: 5 + 3 = 8 choices

**When to Use:**
- Events are mutually exclusive
- Only one can happen at a time
- Example: Choosing one item from different categories

**Multiplication vs Addition: Key Distinction**

**Multiplication (AND):**
- Events happen together
- "Choose shirt AND pants"
- Multiply: m × n

**Addition (OR):**
- Events are alternatives
- "Choose pizza OR pasta"
- Add: m + n

**Common Mistake:** Using addition when multiplication is needed
- Wrong: "3 shirts + 4 pants = 7 outfits"
- Correct: "3 shirts × 4 pants = 12 outfits"

**Inclusion-Exclusion Principle: Handling Overlaps**

When counting unions of sets, we must account for overlaps.

**Formula:** |A ∪ B| = |A| + |B| - |A ∩ B|

**Why Subtract Intersection?**
- |A| + |B| counts A ∩ B twice (once in A, once in B)
- Must subtract one copy to count correctly

**Step-by-Step Example: Students Taking Courses**

**Problem:** Students taking math (30) or science (25), with 10 taking both
- Find: Total students taking at least one course

**Method 1: Direct counting**
- Math only: 30 - 10 = 20
- Science only: 25 - 10 = 15
- Both: 10
- Total: 20 + 15 + 10 = 45

**Method 2: Inclusion-Exclusion**
- |Math ∪ Science| = |Math| + |Science| - |Math ∩ Science|
- = 30 + 25 - 10 = 45 ✓

**Generalization (3 Sets):**
- |A ∪ B ∪ C| = |A| + |B| + |C| - |A ∩ B| - |A ∩ C| - |B ∩ C| + |A ∩ B ∩ C|
- Add singles, subtract pairs, add triple

**Pigeonhole Principle: Guaranteed Collisions**

A simple but powerful principle about distribution.

**Statement:** If n+1 objects are placed into n boxes, at least one box must contain 2 or more objects.

**Why This Works:**
- n boxes can hold at most n objects (1 per box)
- n+1 objects means at least one box gets 2+
- This is guaranteed, not just probable!

**Step-by-Step Example: Birth Months**

**Problem:** 13 people → prove at least 2 share a birth month

**Reasoning:**
- 12 months (boxes)
- 13 people (objects)
- 13 > 12, so by pigeonhole principle, at least one month has 2+ people ✓

**Generalized Form:**
- If kn+1 objects in n boxes, at least one box has k+1 objects
- Example: 25 people, 12 months → at least ⌈25/12⌉ = 3 share a month

**Applications in Computer Science:**

**1. Algorithm Complexity: Counting Operations**

**Nested Loops:**
- Outer loop: n iterations
- Inner loop: m iterations per outer
- Total operations: n × m (multiplication principle)

**Example:** Matrix multiplication
- n × n matrices
- For each of n² result positions
- Compute dot product (n multiplications)
- Total: n² × n = n³ operations

**2. Database Queries: Counting Results**

**SQL Queries:**
- SELECT COUNT(*) uses counting principles
- JOIN operations: multiplication principle
- UNION operations: addition principle (if disjoint)

**Example:** Users × Products = User-Product pairs
- If 1000 users and 100 products
- Possible pairs: 1000 × 100 = 100,000

**3. Game Theory: Counting Strategies**

**Game Trees:**
- Each move has multiple options
- Total strategies: product of options at each level
- Helps analyze game complexity

**Example: Tic-tac-toe**
- First move: 9 choices
- Second move: 8 choices (one square taken)
- Total game states: exponential in moves
- But many are equivalent (symmetries reduce count)

**4. Network Analysis: Counting Paths**

**Graph Theory:**
- Number of paths from A to B
- Use multiplication (sequential steps)
- Use addition (alternative routes)

**Example: Routing**
- Path 1: 3 hops
- Path 2: 4 hops
- Path 3: 3 hops
- Total paths: 3 (addition - alternative routes)

**5. Password/Key Space Analysis:**

**Password Strength:**
- Character set size: c
- Password length: L
- Total passwords: c^L (multiplication - each position independent)
- Example: 26 letters, length 8 → 26⁸ ≈ 208 billion possibilities

**6. Hash Collisions:**

**Pigeonhole Principle Application:**
- Hash function maps large space to smaller space
- More inputs than outputs → collisions guaranteed
- Example: 2³² possible hash values, 2³³ inputs → collision guaranteed

**Common Mistakes:**

**Mistake 1: Adding when should multiply**
- Wrong: "3 ways AND 4 ways = 7 total"
- Correct: "3 ways AND 4 ways = 12 total"
- Remember: AND = multiply, OR = add

**Mistake 2: Multiplying when should add**
- Wrong: "5 pizzas OR 3 pastas = 15 choices"
- Correct: "5 pizzas OR 3 pastas = 8 choices"
- OR means alternatives, not combinations

**Mistake 3: Forgetting to subtract intersection**
- Wrong: |A ∪ B| = |A| + |B|
- Correct: |A ∪ B| = |A| + |B| - |A ∩ B|
- Must account for double-counting!

**Mistake 4: Misapplying pigeonhole principle**
- Need more objects than boxes
- 10 objects, 10 boxes → no guarantee (could be 1 per box)
- 11 objects, 10 boxes → guarantee at least 2 in one box

**Why This Matters:**

Counting principles are used in:
- **Algorithm Analysis:** Counting operations and complexity
- **Probability:** Calculating sample spaces and events
- **Database Design:** Understanding query complexity
- **Cryptography:** Analyzing key space and security
- **Optimization:** Counting solution spaces

**Next Steps:**

Mastering basic counting principles provides the foundation for permutations, combinations, and more advanced combinatorics. In the next lesson, we'll explore applications of these principles in algorithms and probability.`,
					CodeExamples: `# Basic counting principles

# Fundamental Counting Principle
def count_combinations(*options):
    """Count total combinations using multiplication principle."""
    result = 1
    for count in options:
        result *= count
    return result

# Example: 3 shirts × 4 pants × 2 shoes
outfits = count_combinations(3, 4, 2)
print(f"Outfits: 3 shirts × 4 pants × 2 shoes = {outfits}")

# Addition Principle
def count_alternatives(*options):
    """Count alternatives using addition principle."""
    return sum(options)

# Example: 5 pizzas OR 3 pastas
meals = count_alternatives(5, 3)
print(f"\nMeal choices: 5 pizzas + 3 pastas = {meals}")

# Inclusion-Exclusion Principle
def inclusion_exclusion(set_a_size, set_b_size, intersection_size):
    """Calculate union size using inclusion-exclusion."""
    return set_a_size + set_b_size - intersection_size

# Example: Math students (30), Science students (25), Both (10)
total_students = inclusion_exclusion(30, 25, 10)
print(f"\nTotal students: 30 + 25 - 10 = {total_students}")

# Pigeonhole Principle
def pigeonhole_principle(objects, boxes):
    """Determine minimum objects in a box."""
    return (objects - 1) // boxes + 1

# Example: 13 people, 12 months
min_same_month = pigeonhole_principle(13, 12)
print(f"\nPigeonhole: 13 people, 12 months → at least {min_same_month} share a month")

# Complex example: License plates
# Format: 3 letters × 3 digits
letters = 26
digits = 10
license_plates = count_combinations(letters, letters, letters, digits, digits, digits)
print(f"\nLicense plates (3 letters, 3 digits): {license_plates:,}")`,
				},
				{
					Title: "Probability Fundamentals",
					Content: `**Introduction: Quantifying Uncertainty**

Probability is the mathematical framework for dealing with uncertainty and randomness. It allows us to make predictions, analyze risks, and understand random processes. In computer science, probability is essential for randomized algorithms, machine learning, game development, and data analysis.

**Understanding Probability: The Language of Chance**

Probability measures how likely an event is to occur. It's a number between 0 and 1, where 0 means impossible and 1 means certain.

**Key Concepts:**

**Sample Space (S):** The set of all possible outcomes of an experiment
- Denoted by S
- Each outcome is equally likely (in many cases)
- Example: Rolling a die → S = {1, 2, 3, 4, 5, 6}

**Event (E):** A subset of the sample space
- Denoted by E
- Represents outcomes we're interested in
- Example: "Rolling an even number" → E = {2, 4, 6}

**Probability Formula:** P(E) = |E| / |S| = favorable outcomes / total outcomes

**Why This Formula Works:**
- If all outcomes are equally likely, probability = (number of favorable outcomes) / (total outcomes)
- This is the classical definition of probability
- Assumes uniform distribution (each outcome equally likely)

**Range of Probability: 0 ≤ P(E) ≤ 1**

**P(E) = 0: Impossible Event**
- Event cannot occur
- Example: Rolling a 7 on a standard die
- Empty set: E = ∅

**P(E) = 1: Certain Event**
- Event must occur
- Example: Rolling a number between 1 and 6 on a standard die
- Entire sample space: E = S

**P(E) = 0.5: Equally Likely**
- Event is as likely to occur as not occur
- Example: Flipping a fair coin (heads or tails)
- "Fifty-fifty chance"

**Detailed Example: Rolling a Die**

**Sample Space:** S = {1, 2, 3, 4, 5, 6}
- Total outcomes: |S| = 6
- Each outcome equally likely (assuming fair die)

**Example 1: P(rolling 3)**
- Favorable outcomes: {3}
- |E| = 1
- P(rolling 3) = 1/6 ≈ 0.167

**Example 2: P(rolling even)**
- Favorable outcomes: {2, 4, 6}
- |E| = 3
- P(rolling even) = 3/6 = 1/2 = 0.5

**Example 3: P(rolling > 4)**
- Favorable outcomes: {5, 6}
- |E| = 2
- P(rolling > 4) = 2/6 = 1/3 ≈ 0.333

**Probability Rules: Building Complex Probabilities**

**1. Complement Rule: The "Not" Rule**

**Rule:** P(not E) = 1 - P(E)
- Also written as: P(E') = 1 - P(E) or P(E^c) = 1 - P(E)
- The complement of E is all outcomes not in E

**Why This Works:**
- Every outcome is either in E or not in E
- P(E) + P(not E) = 1
- Therefore: P(not E) = 1 - P(E)

**Step-by-Step Example: P(not rolling 3)**
- P(rolling 3) = 1/6
- P(not rolling 3) = 1 - 1/6 = 5/6
- Verification: Favorable outcomes = {1, 2, 4, 5, 6}, so 5/6 ✓

**When to Use:**
- Sometimes easier to calculate P(not E) than P(E)
- Example: P(at least one success) = 1 - P(no successes)

**2. Addition Rule: The "Or" Rule**

**For Mutually Exclusive Events:** P(A or B) = P(A) + P(B)

**Mutually Exclusive:** Events that cannot both occur
- If A occurs, B cannot occur (and vice versa)
- A ∩ B = ∅ (empty intersection)

**Why This Works:**
- If A and B are mutually exclusive, |A ∪ B| = |A| + |B|
- P(A or B) = |A ∪ B| / |S| = (|A| + |B|) / |S| = P(A) + P(B)

**Step-by-Step Example: P(rolling 1 or 2)**
- A = {1}, B = {2}
- A and B are mutually exclusive (can't roll both 1 and 2)
- P(A) = 1/6, P(B) = 1/6
- P(A or B) = 1/6 + 1/6 = 2/6 = 1/3
- Verification: Favorable outcomes = {1, 2}, so 2/6 = 1/3 ✓

**General Addition Rule (Not Mutually Exclusive):**
- P(A or B) = P(A) + P(B) - P(A and B)
- Subtract P(A and B) because it's counted twice
- This is the inclusion-exclusion principle

**Example: P(rolling even or > 4)**
- A = {2, 4, 6} (even), B = {5, 6} (> 4)
- A and B are NOT mutually exclusive (6 is in both)
- P(A) = 3/6, P(B) = 2/6, P(A and B) = 1/6 (rolling 6)
- P(A or B) = 3/6 + 2/6 - 1/6 = 4/6 = 2/3
- Verification: Favorable = {2, 4, 5, 6}, so 4/6 = 2/3 ✓

**3. Multiplication Rule: The "And" Rule**

**For Independent Events:** P(A and B) = P(A) × P(B)

**Independent Events:** Events where occurrence of one doesn't affect the other
- P(B | A) = P(B) (B's probability unchanged by A)
- Example: Coin flips are independent

**Why This Works:**
- If A and B are independent, the probability of both = product of individual probabilities
- This extends to multiple independent events

**Step-by-Step Example: P(rolling 3 twice)**
- First roll: P(3) = 1/6
- Second roll: P(3) = 1/6 (independent of first roll)
- P(rolling 3 twice) = (1/6) × (1/6) = 1/36 ≈ 0.028

**General Multiplication Rule (Not Independent):**
- P(A and B) = P(A) × P(B | A)
- P(B | A) is conditional probability (probability of B given A)
- This accounts for dependence between events

**Example: Drawing cards without replacement**
- P(first card is ace) = 4/52
- P(second card is ace | first was ace) = 3/51 (one ace removed)
- P(both aces) = (4/52) × (3/51) = 12/2652 ≈ 0.0045

**Applications in Computer Science:**

**1. Randomized Algorithms:**
- **Monte Carlo Methods:** Use probability to approximate solutions
- **Las Vegas Algorithms:** Always correct, but runtime is probabilistic
- **Randomized Quicksort:** Average case O(n log n) with high probability

**Example:** Randomized selection algorithm
- Choose random pivot → expected O(n) time
- Probability analysis determines expected performance

**2. Machine Learning:**
- **Bayesian Methods:** Update probabilities based on evidence
- **Naive Bayes:** Uses multiplication rule (assumes independence)
- **Probabilistic Models:** Represent uncertainty in predictions

**Example:** Spam detection
- P(spam | words) calculated using Bayes' theorem
- Combines prior probability with evidence

**3. Game Development:**
- **Random Events:** Loot drops, damage calculations
- **Probability Distributions:** Normal, uniform, exponential
- **Monte Carlo Simulation:** Simulate game scenarios

**Example:** Loot drop system
- P(rare item) = 0.01 (1% chance)
- Use probability to balance game economy

**4. Risk Analysis:**
- **System Reliability:** Probability of system failure
- **Security:** Probability of successful attack
- **Performance:** Probability of meeting SLA

**Example:** System with 3 components
- Each has 0.9 reliability (90% chance of working)
- P(all work) = 0.9³ = 0.729 (if independent)
- P(system works) = 1 - P(all fail) = 1 - 0.1³ = 0.999

**5. Data Science:**
- **Statistical Inference:** Make conclusions from data
- **Hypothesis Testing:** Calculate p-values (probabilities)
- **A/B Testing:** Compare probabilities of success

**Example:** A/B test
- Version A: 100 conversions out of 1000 (10%)
- Version B: 120 conversions out of 1000 (12%)
- Use probability to determine if difference is significant

**Common Mistakes:**

**Mistake 1: Adding probabilities for non-mutually exclusive events**
- Wrong: P(A or B) = P(A) + P(B) (if A and B overlap)
- Correct: P(A or B) = P(A) + P(B) - P(A and B)

**Mistake 2: Multiplying probabilities for dependent events**
- Wrong: P(A and B) = P(A) × P(B) (if A and B are dependent)
- Correct: P(A and B) = P(A) × P(B | A)

**Mistake 3: Confusing "or" with "and"**
- "A or B" means at least one occurs
- "A and B" means both occur
- Be careful with language!

**Mistake 4: Assuming events are independent**
- Not all events are independent!
- Check if occurrence of one affects the other
- Example: Drawing cards without replacement → dependent

**Why This Matters:**

Probability is fundamental to:
- **Decision Making:** Quantify uncertainty
- **Algorithm Design:** Analyze randomized algorithms
- **Data Analysis:** Make statistical inferences
- **Risk Management:** Assess and mitigate risks
- **Machine Learning:** Model uncertainty and make predictions

**Next Steps:**

Mastering probability fundamentals provides the foundation for conditional probability, Bayes' theorem, and more advanced probabilistic reasoning. In the next lesson, we'll explore conditional probability, which extends these concepts to dependent events.`,
					CodeExamples: `# Probability calculations

import random

def probability(event_outcomes, total_outcomes):
    """Calculate probability of an event."""
    return event_outcomes / total_outcomes

# Example: Rolling a die
die_faces = 6
print("Die probabilities:")
print(f"P(rolling 3) = {probability(1, die_faces)}")
print(f"P(rolling even) = {probability(3, die_faces)}")
print(f"P(rolling > 4) = {probability(2, die_faces)}")

# Complement rule
def complement_probability(p):
    """Calculate probability of complement."""
    return 1 - p

p_3 = probability(1, 6)
p_not_3 = complement_probability(p_3)
print(f"\nComplement: P(not 3) = {p_not_3}")

# Addition rule (mutually exclusive)
def add_probabilities(p1, p2):
    """Add probabilities of mutually exclusive events."""
    return p1 + p2

p_1_or_2 = add_probabilities(probability(1, 6), probability(1, 6))
print(f"\nAddition: P(1 or 2) = {p_1_or_2}")

# Multiplication rule (independent)
def multiply_probabilities(p1, p2):
    """Multiply probabilities of independent events."""
    return p1 * p2

p_two_3s = multiply_probabilities(p_3, p_3)
print(f"\nMultiplication: P(two 3s) = {p_two_3s}")

# Empirical probability (simulation)
def empirical_probability(event_func, trials=10000):
    """Estimate probability through simulation."""
    successes = 0
    for _ in range(trials):
        if event_func():
            successes += 1
    return successes / trials

# Simulate rolling die
def roll_die():
    return random.randint(1, 6)

def is_even():
    return roll_die() % 2 == 0

empirical_p_even = empirical_probability(is_even)
print(f"\nEmpirical P(even) after 10000 trials: {empirical_p_even:.4f}")
print(f"Theoretical: {probability(3, 6):.4f}")`,
				},
				{
					Title: "Conditional Probability",
					Content: `**Introduction: Probability Updated by Information**

Conditional probability answers the question "What's the probability of A, given that B has already happened?" This is fundamental to updating beliefs with new information, which is central to machine learning, medical diagnosis, and decision-making under uncertainty.

**Understanding Conditional Probability: Updating with Information**

Conditional probability adjusts probabilities based on new information. It's how we update our beliefs when we learn something new.

**Notation:** P(A | B)
- Read as: "Probability of A given B" or "Probability of A conditional on B"
- The vertical bar | means "given" or "conditional on"
- B is the condition or information we know

**Key Insight:** Knowing B has occurred changes the sample space from S to B, and we only consider outcomes in B.

**Formula:** P(A | B) = P(A and B) / P(B)

**Why This Formula Works:**

**Intuitive Explanation:**
- We know B occurred, so our new sample space is B (not the original S)
- We want probability of A, but only considering outcomes where B occurred
- This is A ∩ B (A and B both occur)
- Probability = (outcomes in A ∩ B) / (outcomes in B)
- = P(A and B) / P(B)

**Visual Understanding:**
- Original sample space: S (all possible outcomes)
- After learning B: New sample space is B (only B outcomes)
- Probability of A in new space: (A ∩ B) / B

**Step-by-Step Example: Drawing Cards**

**Problem:** What's the probability of drawing a King, given that we drew a Face card?

**Given:**
- Standard deck: 52 cards
- Face cards: Jack, Queen, King (4 of each) = 12 total
- Kings: 4 total (all are face cards)

**Method 1: Using Formula**

1. **P(King and Face card):**
   - All kings are face cards, so P(King and Face card) = P(King) = 4/52

2. **P(Face card):**
   - 12 face cards out of 52, so P(Face card) = 12/52

3. **Calculate:**
   - P(King | Face card) = (4/52) / (12/52) = 4/12 = 1/3

**Method 2: Direct Reasoning**

- We know we have a face card (12 possibilities)
- Among face cards, 4 are kings
- Probability = 4/12 = 1/3 ✓

**Independent Events: No Information Gain**

Events A and B are independent if knowing B doesn't change the probability of A.

**Definition:** P(A | B) = P(A)

**Equivalent Definitions:**
- P(A | B) = P(A)
- P(B | A) = P(B)
- P(A and B) = P(A) × P(B)

**Why These Are Equivalent:**
- If P(A | B) = P(A), then P(A and B) / P(B) = P(A)
- Therefore: P(A and B) = P(A) × P(B)
- This is the multiplication rule for independent events

**Examples of Independent Events:**

**Coin Flips:**
- First flip: P(Heads) = 1/2
- Second flip: P(Heads | first was Heads) = 1/2
- Knowing first flip doesn't affect second → independent

**Rolling Dice:**
- First die: P(6) = 1/6
- Second die: P(6 | first was 6) = 1/6
- Dice don't affect each other → independent

**Dependent Events: Information Changes Probability**

Events A and B are dependent if knowing B changes the probability of A.

**Definition:** P(A | B) ≠ P(A)

**Key Insight:** The occurrence of B provides information about A.

**Examples of Dependent Events:**

**Drawing Cards Without Replacement:**
- First card: P(Ace) = 4/52
- Second card: P(Ace | first was Ace) = 3/51 (one ace removed)
- 3/51 ≠ 4/52 → dependent!

**Weather and Seasons:**
- P(Rain) might be 0.3 overall
- P(Rain | Winter) might be 0.5 (higher in winter)
- Different probabilities → dependent

**Bayes' Theorem: Reversing Conditional Probabilities**

Bayes' theorem allows us to reverse conditional probabilities, which is incredibly powerful for updating beliefs.

**Formula:** P(A | B) = [P(B | A) × P(A)] / P(B)

**Why This Matters:**
- Sometimes P(B | A) is easier to know than P(A | B)
- Example: P(test positive | disease) is known, but we want P(disease | test positive)
- Bayes' theorem lets us flip this!

**Derivation:**

1. **From conditional probability:** P(A | B) = P(A and B) / P(B)
2. **Also:** P(B | A) = P(A and B) / P(A)
3. **Solve for P(A and B):** P(A and B) = P(B | A) × P(A)
4. **Substitute:** P(A | B) = [P(B | A) × P(A)] / P(B) ✓

**Step-by-Step Example: Medical Test**

**Problem:** A disease affects 1% of population. Test is 95% accurate (95% true positive, 5% false positive). If someone tests positive, what's the probability they have the disease?

**Given:**
- P(disease) = 0.01 (prior probability)
- P(positive | disease) = 0.95 (test sensitivity)
- P(positive | no disease) = 0.05 (false positive rate)

**Find:** P(disease | positive)

**Step 1: Calculate P(positive) using Law of Total Probability**
- P(positive) = P(positive | disease) × P(disease) + P(positive | no disease) × P(no disease)
- P(positive) = 0.95 × 0.01 + 0.05 × 0.99
- P(positive) = 0.0095 + 0.0495 = 0.059

**Step 2: Apply Bayes' Theorem**
- P(disease | positive) = [P(positive | disease) × P(disease)] / P(positive)
- P(disease | positive) = (0.95 × 0.01) / 0.059
- P(disease | positive) = 0.0095 / 0.059 ≈ 0.161

**Result:** Only about 16% chance of having disease despite positive test!

**Why This Seems Counterintuitive:**
- Test is 95% accurate, but disease is rare (1%)
- Many false positives from the 99% healthy population
- This is why screening tests need high specificity for rare diseases

**Applications in Computer Science:**

**1. Machine Learning - Naive Bayes Classifier:**

**Spam Filtering:**
- P(spam | words) calculated using Bayes' theorem
- P(words | spam) learned from training data
- Updates probability as new emails arrive

**Example:** Classify email as spam
- P(spam) = prior probability
- P(words | spam) = learned from spam emails
- P(words | ham) = learned from legitimate emails
- Calculate P(spam | words) using Bayes' theorem

**2. Medical Diagnosis:**
- Update disease probability with test results
- Combine multiple tests using conditional probability
- Critical for decision support systems

**3. Recommendation Systems:**
- P(like item | user features) using Bayes' theorem
- Update recommendations based on user behavior
- Used by Netflix, Amazon, etc.

**4. Natural Language Processing:**
- P(word | context) for language modeling
- Machine translation uses conditional probabilities
- Speech recognition: P(word | audio features)

**5. Computer Vision:**
- P(object | image features) for object detection
- Face recognition: P(identity | facial features)
- Image classification uses conditional probabilities

**6. Anomaly Detection:**
- P(anomaly | system metrics)
- Update probability as metrics change
- Used in cybersecurity, fraud detection

**Common Mistakes:**

**Mistake 1: Confusing P(A | B) with P(B | A)**
- Wrong: Assuming P(disease | positive) = P(positive | disease)
- Correct: Use Bayes' theorem to convert
- These are very different! (0.95 vs 0.16 in medical example)

**Mistake 2: Forgetting to normalize**
- Wrong: P(A | B) = P(B | A) × P(A) (missing division by P(B))
- Correct: P(A | B) = [P(B | A) × P(A)] / P(B)
- Must divide by P(B) to get valid probability

**Mistake 3: Assuming independence when events are dependent**
- Wrong: P(A | B) = P(A) always
- Correct: Check if events are actually independent
- Many real-world events are dependent!

**Mistake 4: Not considering the base rate**
- Wrong: Ignoring P(A) in Bayes' theorem
- Correct: Prior probability P(A) is crucial
- Rare events need strong evidence to overcome low prior

**Why This Matters:**

Conditional probability is essential for:
- **Updating Beliefs:** Incorporating new information
- **Machine Learning:** Training classifiers and models
- **Decision Making:** Making informed choices under uncertainty
- **Risk Assessment:** Evaluating probabilities in context
- **Data Analysis:** Understanding relationships between events

**Next Steps:**

Mastering conditional probability and Bayes' theorem provides the foundation for Bayesian statistics, machine learning algorithms, and probabilistic reasoning. In the next lesson, we'll explore expected value, which combines probability with outcomes to calculate averages.`,
					CodeExamples: `# Conditional probability

def conditional_probability(p_a_and_b, p_b):
    """Calculate P(A | B) = P(A and B) / P(B)."""
    if p_b == 0:
        return None
    return p_a_and_b / p_b

# Example: Cards
# P(King and Face card) = 4/52 (all kings are face cards)
# P(Face card) = 12/52
p_king_and_face = 4 / 52
p_face = 12 / 52
p_king_given_face = conditional_probability(p_king_and_face, p_face)
print(f"P(King | Face card) = {p_king_given_face:.4f}")

# Independent events check
def are_independent(p_a, p_b, p_a_and_b):
    """Check if events A and B are independent."""
    return abs(p_a_and_b - p_a * p_b) < 1e-10

# Example: Coin flips
p_heads1 = 0.5
p_heads2 = 0.5
p_both_heads = 0.25
print(f"\nCoin flips independent: {are_independent(p_heads1, p_heads2, p_both_heads)}")

# Bayes' Theorem
def bayes_theorem(p_b_given_a, p_a, p_b):
    """Calculate P(A | B) using Bayes' theorem."""
    if p_b == 0:
        return None
    return (p_b_given_a * p_a) / p_b

# Medical test example
p_disease = 0.01
p_positive_given_disease = 0.95
p_positive_given_no_disease = 0.05

# Calculate P(positive) using law of total probability
p_positive = (p_positive_given_disease * p_disease + 
              p_positive_given_no_disease * (1 - p_disease))

p_disease_given_positive = bayes_theorem(
    p_positive_given_disease, p_disease, p_positive
)

print(f"\nMedical test (Bayes' theorem):")
print(f"P(disease) = {p_disease}")
print(f"P(positive | disease) = {p_positive_given_disease}")
print(f"P(disease | positive) = {p_disease_given_positive:.4f}")`,
				},
				{
					Title: "Expected Value",
					Content: `**Introduction: The Long-Run Average**

Expected value, also called the mean, is the average outcome we expect over many trials of a random process. It's a fundamental concept in probability, statistics, and algorithm analysis. Understanding expected value helps us predict average performance, make optimal decisions, and analyze randomized algorithms.

**Understanding Expected Value: Weighted Average**

Expected value is a weighted average, where each possible outcome is weighted by its probability.

**Formula:** E(X) = Σ [x × P(x)] for all possible values x

**Breaking Down the Formula:**
- **x:** A possible outcome value
- **P(x):** Probability of that outcome
- **x × P(x):** Weighted contribution of that outcome
- **Sum:** Add up all weighted contributions

**Key Insight:** Expected value is the "center of mass" of the probability distribution - where the outcomes tend to cluster on average.

**Step-by-Step Example: Rolling a Die**

**Possible Outcomes:** 1, 2, 3, 4, 5, 6
**Probability of Each:** 1/6 (fair die)

**Calculate Expected Value:**
- E(X) = 1×(1/6) + 2×(1/6) + 3×(1/6) + 4×(1/6) + 5×(1/6) + 6×(1/6)
- E(X) = (1 + 2 + 3 + 4 + 5 + 6) × (1/6)
- E(X) = 21 × (1/6) = 21/6 = 3.5

**Interpretation:**
- Over many rolls, the average value approaches 3.5
- Not that we'll roll 3.5 (impossible!), but the average will be 3.5
- Law of Large Numbers: As number of trials → ∞, average → expected value

**More Complex Example: Weighted Die**

**Outcomes and Probabilities:**
- 1: P(1) = 0.1
- 2: P(2) = 0.2
- 3: P(3) = 0.3
- 4: P(4) = 0.2
- 5: P(5) = 0.1
- 6: P(6) = 0.1

**Calculate Expected Value:**
- E(X) = 1×0.1 + 2×0.2 + 3×0.3 + 4×0.2 + 5×0.1 + 6×0.1
- E(X) = 0.1 + 0.4 + 0.9 + 0.8 + 0.5 + 0.6 = 3.3

**Note:** Expected value is 3.3, not 3.5, because probabilities are weighted toward lower numbers.

**Properties of Expected Value: Powerful Tools**

**1. Linearity: E(aX + b) = aE(X) + b**

**Why This Works:**
- Scaling all outcomes scales the average
- Adding constant shifts the average

**Example:**
- If E(X) = 3.5 (die), then E(2X + 10) = 2×3.5 + 10 = 17
- Verification: 2×1+10=12, 2×6+10=22, average = 17 ✓

**2. Addition: E(X + Y) = E(X) + E(Y)**

**Why This Works:**
- Expected value of sum = sum of expected values
- Works even if X and Y are dependent!

**Example:**
- E(die) = 3.5
- E(two dice) = E(die₁) + E(die₂) = 3.5 + 3.5 = 7
- This makes sense: average sum of two dice is 7

**3. Multiplication (Independent Only): E(XY) = E(X)E(Y)**

**Important:** Only true if X and Y are independent!

**Why This Works:**
- For independent variables, product of averages = average of products
- This is NOT true for dependent variables!

**Example:**
- Two independent dice: E(XY) = E(X) × E(Y) = 3.5 × 3.5 = 12.25
- But if Y = X (same die twice): E(XY) = E(X²) = 91/6 ≈ 15.17 ≠ 12.25

**Variance: Measuring Spread**

Variance measures how spread out the outcomes are around the expected value.

**Formula:** Var(X) = E(X²) - [E(X)]²

**Why This Formula Works:**
- E(X²) is the average of squared values
- [E(X)]² is the square of the average
- Their difference measures spread

**Alternative Formula:** Var(X) = E[(X - E(X))²]
- Average squared deviation from mean
- More intuitive but algebraically equivalent

**Step-by-Step Example: Coin Flip**

**Setup:** Heads = 1, Tails = 0, P(1) = P(0) = 0.5

**Calculate Expected Value:**
- E(X) = 1×0.5 + 0×0.5 = 0.5

**Calculate E(X²):**
- E(X²) = 1²×0.5 + 0²×0.5 = 1×0.5 + 0 = 0.5

**Calculate Variance:**
- Var(X) = E(X²) - [E(X)]² = 0.5 - (0.5)² = 0.5 - 0.25 = 0.25

**Standard Deviation:** σ = √Var(X) = √0.25 = 0.5

**Interpretation:**
- Outcomes are 0 or 1, average is 0.5
- Standard deviation 0.5 means outcomes are spread around the mean
- For binary outcomes, σ = √[p(1-p)] where p = probability of success

**Applications in Computer Science:**

**1. Algorithm Analysis: Average Case Complexity**

**Average Case Time:**
- Expected number of operations over all possible inputs
- E(T(n)) = Σ [T(input) × P(input)]
- Helps understand typical performance

**Example: Quick Sort Average Case**
- Best case: O(n log n)
- Worst case: O(n²)
- Average case: E(comparisons) = O(n log n)
- Expected value shows average is good, even though worst case is bad!

**Randomized Algorithms:**
- Algorithms that use randomness
- Performance is probabilistic
- Expected value gives average performance

**Example: Randomized Quick Sort**
- Choose random pivot
- Expected comparisons: O(n log n)
- High probability of good performance

**2. Game Theory: Optimal Strategies**

**Expected Payoff:**
- Calculate expected value of different strategies
- Choose strategy with highest expected value
- Used in AI, game design, economics

**Example: Game with Two Moves**
- Move A: 50% chance of +10, 50% chance of -5 → E(A) = 2.5
- Move B: 30% chance of +20, 70% chance of 0 → E(B) = 6
- Optimal: Choose B (higher expected value)

**3. Risk Assessment:**

**Expected Loss:**
- E(loss) = Σ [loss × P(loss)]
- Helps quantify risk
- Used in insurance, security, system design

**Example: System Failure**
- Failure probability: 0.01
- Cost if fails: $100,000
- Expected loss: 0.01 × $100,000 = $1,000
- Helps decide if prevention is worth cost

**4. Decision Making Under Uncertainty:**

**Expected Utility:**
- Combine outcomes with their probabilities
- Make decisions maximizing expected utility
- Used in AI, economics, operations research

**Example: Investment Decision**
- Option A: 60% chance of +$1000, 40% chance of -$500
  - E(A) = 0.6×1000 + 0.4×(-500) = 600 - 200 = $400
- Option B: 80% chance of +$300, 20% chance of +$100
  - E(B) = 0.8×300 + 0.2×100 = 240 + 20 = $260
- Choose A (higher expected value)

**5. Performance Prediction:**

**Expected Response Time:**
- Model system performance probabilistically
- E(response time) = Σ [time × P(time)]
- Helps predict and optimize performance

**Example: Web Server**
- Fast response (0.1s): 90% of requests
- Slow response (1.0s): 10% of requests
- E(response) = 0.9×0.1 + 0.1×1.0 = 0.19 seconds
- Helps set SLAs and capacity planning

**Common Mistakes:**

**Mistake 1: Confusing expected value with most likely value**
- Expected value is average, not mode
- Example: Die expected value is 3.5, but no single roll gives 3.5
- Most likely might be different from expected value

**Mistake 2: Assuming E(XY) = E(X)E(Y) always**
- Only true for independent variables!
- For dependent variables, need covariance term
- Check independence first!

**Mistake 3: Not weighting by probabilities**
- Wrong: E(X) = average of outcomes (only if equal probabilities)
- Correct: E(X) = Σ [x × P(x)] (weighted average)
- Must account for different probabilities!

**Mistake 4: Confusing variance with standard deviation**
- Variance = σ² (squared units)
- Standard deviation = σ (original units)
- Use standard deviation for interpretation (same units as data)

**Why This Matters:**

Expected value is essential for:
- **Algorithm Analysis:** Understanding average performance
- **Decision Making:** Choosing optimal strategies
- **Risk Management:** Quantifying and managing risks
- **Performance Prediction:** Forecasting system behavior
- **Optimization:** Making data-driven decisions

**Next Steps:**

Mastering expected value provides the foundation for understanding variance, standard deviation, and more advanced statistical concepts. In the next lesson, we'll explore applications in algorithms, showing how expected value helps analyze randomized algorithms and average-case complexity.`,
					CodeExamples: `# Expected value calculations

def expected_value(values, probabilities):
    """Calculate expected value E(X) = Σ x × P(x)."""
    if len(values) != len(probabilities):
        return None
    return sum(x * p for x, p in zip(values, probabilities))

# Example: Rolling a die
die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
die_expected = expected_value(die_values, die_probs)
print(f"Expected value of die: {die_expected}")

# Example: Coin flip (heads=1, tails=0)
coin_values = [0, 1]
coin_probs = [0.5, 0.5]
coin_expected = expected_value(coin_values, coin_probs)
print(f"Expected value of coin: {coin_expected}")

# Variance
def variance(values, probabilities):
    """Calculate variance Var(X) = E(X²) - [E(X)]²."""
    e_x = expected_value(values, probabilities)
    e_x_squared = expected_value([x**2 for x in values], probabilities)
    return e_x_squared - e_x**2

def standard_deviation(values, probabilities):
    """Calculate standard deviation σ = √Var(X)."""
    import math
    return math.sqrt(variance(values, probabilities))

die_var = variance(die_values, die_probs)
die_std = standard_deviation(die_values, die_probs)
print(f"\nDie variance: {die_var:.4f}")
print(f"Die standard deviation: {die_std:.4f}")

# Example: Game with payouts
game_outcomes = [-5, 0, 10, 20]  # Lose $5, break even, win $10, win $20
game_probs = [0.3, 0.4, 0.2, 0.1]
game_expected = expected_value(game_outcomes, game_probs)
print(f"\nGame expected value: ${game_expected:.2f}")
print("Interpretation: Average winnings per game")`,
				},
				{
					Title: "Applications in Algorithms",
					Content: `**Introduction: Mathematics Powers Algorithm Design**

Combinatorics and probability are not just abstract mathematics - they're essential tools for designing and analyzing algorithms. From counting operations to analyzing randomized algorithms, these mathematical concepts help us understand algorithm performance and make informed design decisions.

**Combinatorics in Algorithms: Counting and Enumerating**

**1. Counting Operations: Analyzing Algorithm Complexity**

Combinatorics helps us count how many operations an algorithm performs.

**Example: Sorting Analysis**

**Bubble Sort Comparisons:**
- Pass 1: Compare n-1 pairs
- Pass 2: Compare n-2 pairs
- ...
- Pass (n-1): Compare 1 pair
- Total: Σᵢ₌₁ⁿ⁻¹ i = (n-1)n/2 = O(n²)
- Uses sum of arithmetic sequence!

**Merge Sort Comparisons:**
- Divide array in half: log₂(n) levels
- Merge at each level: n comparisons
- Total: n × log₂(n) = O(n log n)
- Uses geometric sequence (halving) and multiplication!

**2. Search Space: Understanding Problem Complexity**

Counting possible solutions helps determine if exhaustive search is feasible.

**N-Queens Problem:**
- Place n queens on n×n board so none attack each other
- Naive approach: Try all placements
- Total placements: nⁿ (exponential!)
- With constraints: Much smaller, but still exponential
- Understanding search space helps choose algorithms

**Example: 8-Queens**
- Without constraints: 8⁸ = 16,777,216 possibilities
- With row constraint: 8! = 40,320 (one queen per row)
- With all constraints: ~92 solutions
- Backtracking reduces search space dramatically!

**3. Graph Algorithms: Counting Paths and Cycles**

**Shortest Path:**
- Count paths between nodes
- Use multiplication principle (sequential steps)
- Use addition principle (alternative routes)

**Example: Routes from A to B**
- Path 1: 3 steps
- Path 2: 4 steps  
- Path 3: 3 steps
- Total paths: 3 (addition - alternatives)

**Probability in Algorithms: Embracing Randomness**

**1. Randomized Algorithms: Better Average Performance**

Randomized algorithms use randomness to achieve better average-case performance.

**Quicksort with Random Pivot:**

**Deterministic Quicksort:**
- Worst case: O(n²) if pivot is always minimum/maximum
- Can happen with sorted/reverse-sorted input

**Randomized Quicksort:**
- Choose pivot randomly
- Worst case still O(n²), but probability is negligible
- Expected case: O(n log n)
- Much better in practice!

**Why Randomization Helps:**
- Adversary can't force worst case
- Random choice distributes bad cases
- Expected performance is good

**2. Monte Carlo Methods: Approximating Solutions**

Use random sampling to approximate solutions to difficult problems.

**Estimating π:**

**Method:**
- Generate random points in unit square [0,1] × [0,1]
- Count points inside quarter circle (radius 1)
- Ratio: (points in circle) / (total points) ≈ π/4
- Therefore: π ≈ 4 × (ratio)

**Why It Works:**
- Area of quarter circle = π/4
- Area of square = 1
- Ratio of areas = (π/4) / 1 = π/4
- Random sampling approximates this ratio

**Accuracy:**
- More samples → better approximation
- Error decreases as 1/√n (square root of samples)
- 1 million samples → ~0.1% error

**Applications:**
- Numerical integration
- Optimization problems
- Simulation and modeling

**3. Las Vegas Algorithms: Random Runtime, Guaranteed Correctness**

Las Vegas algorithms always produce correct results, but running time is random.

**Randomized Quicksort (Las Vegas variant):**
- Always produces sorted array (correct)
- Running time varies (random)
- Expected time: O(n log n)
- Worst case: O(n²) but very unlikely

**Comparison:**
- **Monte Carlo:** Fast but may be wrong
- **Las Vegas:** Always correct but time varies
- **Deterministic:** Predictable but may be slow

**Expected Running Time: Average Case Analysis**

Expected value helps analyze average performance.

**Formula:** E[T(n)] = Σ P(input) × T(input)

**Why Expected Value Matters:**
- Worst case may be rare
- Average case often more representative
- Helps choose between algorithms

**Example: Quicksort Analysis**

**Worst Case:** O(n²)
- Occurs when pivot is always minimum/maximum
- Probability: Very low for random input

**Average Case:** O(n log n)
- Most inputs perform well
- Expected comparisons: ~2n ln(n)

**Expected Case Analysis:**
- E[comparisons] = Σ P(input) × comparisons(input)
- For random input: E[comparisons] = O(n log n)
- Shows algorithm is good on average!

**Hash Functions: Probability-Based Design**

Hash functions use probability theory to analyze collision rates.

**Birthday Paradox:**
- With n items and m buckets
- P(no collision) decreases rapidly
- P(collision) ≈ 1 - e^(-n²/(2m)) for large m

**Collision Probability:**
- For n items, m buckets: P(collision) ≈ n²/(2m)
- Example: 100 items, 1000 buckets → ~5% collision chance
- Helps choose table size!

**Load Factor:**
- α = n/m (items per bucket)
- Expected chain length: α
- Want α < 1 for good performance
- Resize when α exceeds threshold (e.g., 0.75)

**Applications:**

**1. Algorithm Design:**
- **Backtracking:** Use combinatorics to prune search space
- **Dynamic Programming:** Count subproblems to optimize
- **Greedy Algorithms:** Use probability to analyze correctness

**2. Performance Optimization:**
- **Caching:** Use probability to predict cache hits
- **Load Balancing:** Distribute load probabilistically
- **Resource Allocation:** Optimize using expected values

**3. Randomized Data Structures:**
- **Skip Lists:** Probabilistic balanced trees
- **Treaps:** Randomized binary search trees
- Use randomness to maintain balance

**4. Probabilistic Data Structures:**
- **Bloom Filters:** Space-efficient membership testing
- **Count-Min Sketch:** Approximate frequency counting
- **HyperLogLog:** Estimate cardinality
- Trade accuracy for space/time

**Example: Bloom Filter**

**Problem:** Check if element is in set (very large set)
**Solution:** Probabilistic data structure
- **Space:** Much smaller than storing all elements
- **Time:** O(1) lookup
- **Trade-off:** May have false positives (never false negatives)

**How It Works:**
- Use k hash functions
- Set bits in bit array for each hash
- Check: If all k bits set → probably in set
- Probability analysis determines optimal k and array size

**Why This Matters:**

Combinatorics and probability are essential for:
- **Algorithm Design:** Creating efficient solutions
- **Performance Analysis:** Understanding algorithm behavior
- **Optimization:** Making informed trade-offs
- **Randomized Algorithms:** Leveraging randomness for better performance
- **Data Structures:** Designing space-efficient structures

**Next Steps:**

Mastering these applications shows how mathematical concepts directly translate to practical algorithm design. The counting and probability techniques you've learned are tools used daily by algorithm designers and computer scientists.`,
					CodeExamples: `# Applications in algorithms

import random
import math

# Monte Carlo method to estimate π
def estimate_pi(n_samples=1000000):
    """Estimate π using Monte Carlo method."""
    inside_circle = 0
    for _ in range(n_samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            inside_circle += 1
    return 4 * inside_circle / n_samples

pi_estimate = estimate_pi()
print(f"Monte Carlo π estimate: {pi_estimate:.6f}")
print(f"Actual π: {math.pi:.6f}")
print(f"Error: {abs(pi_estimate - math.pi):.6f}")

# Expected comparisons in linear search
def expected_linear_search_comparisons(n):
    """Calculate expected comparisons to find item in list of n items."""
    # Assuming item is equally likely to be at any position
    positions = list(range(1, n + 1))
    probabilities = [1/n] * n
    return sum(p * pos for p, pos in zip(probabilities, positions))

n = 100
expected_comps = expected_linear_search_comparisons(n)
print(f"\nExpected comparisons in linear search (n={n}): {expected_comps:.2f}")
print(f"Worst case: {n}")

# Hash collision probability
def hash_collision_probability(n_items, n_buckets):
    """Estimate probability of at least one collision."""
    # Approximation: P(collision) ≈ 1 - e^(-n²/(2m))
    return 1 - math.exp(-(n_items**2) / (2 * n_buckets))

n_items = 23
n_buckets = 365  # Birthday paradox
collision_prob = hash_collision_probability(n_items, n_buckets)
print(f"\nHash collision probability:")
print(f"{n_items} items in {n_buckets} buckets: {collision_prob:.4f}")
print("(This is the birthday paradox!)")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          505,
			Title:       "Logarithms and Exponentials",
			Description: "Master logarithmic and exponential functions essential for algorithm analysis.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "Understanding Logarithms",
					Content: `**Introduction: The Power of Inverse Operations**

Logarithms are one of the most important mathematical concepts in computer science. They answer the question "What power do I need to raise this base to get this number?" Understanding logarithms is essential for algorithm analysis, understanding exponential growth, and working with large numbers efficiently.

**What Are Logarithms: The Inverse of Exponentiation**

A logarithm is the inverse operation of exponentiation. Just as subtraction undoes addition and division undoes multiplication, logarithms undo exponentiation.

**Definition:** log_b(a) = c means b^c = a

**Breaking Down the Definition:**
- **b** is the base (the number being raised to a power)
- **a** is the argument (the result we're trying to achieve)
- **c** is the logarithm (the power we need)

**Read as:** "log base b of a equals c" or "the logarithm of a to the base b is c"

**Key Relationship:**
- If b^c = a, then log_b(a) = c
- If log_b(a) = c, then b^c = a
- These are equivalent statements!

**Why Logarithms Exist:**

**The Problem:** Exponentiation makes numbers grow very quickly
- 2¹⁰ = 1,024
- 2²⁰ = 1,048,576
- 2³⁰ = 1,073,741,824

**The Solution:** Logarithms compress this growth
- log₂(1,024) = 10
- log₂(1,048,576) = 20
- log₂(1,073,741,824) = 30

Logarithms transform multiplicative relationships into additive ones, making huge numbers manageable.

**Examples with Detailed Explanation:**

**Example 1: log₂(8) = 3**
- Question: "What power of 2 equals 8?"
- Answer: 2³ = 8, so log₂(8) = 3
- Verification: 2³ = 2 × 2 × 2 = 8 ✓

**Example 2: log₁₀(100) = 2**
- Question: "What power of 10 equals 100?"
- Answer: 10² = 100, so log₁₀(100) = 2
- Verification: 10² = 10 × 10 = 100 ✓

**Example 3: log₅(25) = 2**
- Question: "What power of 5 equals 25?"
- Answer: 5² = 25, so log₅(25) = 2
- Verification: 5² = 5 × 5 = 25 ✓

**Common Bases: Which One to Use?**

Different bases are used for different purposes. Understanding when to use each is crucial.

**1. Base 10 (Common Logarithm): log₁₀(x) or log(x)**

**Properties:**
- Most intuitive for humans (we use base 10 number system)
- Used in scientific notation
- Easy to estimate: log(1000) = 3, log(100) = 2, log(10) = 1

**Examples:**
- log(1000) = 3 (because 10³ = 1000)
- log(100) = 2 (because 10² = 100)
- log(10) = 1 (because 10¹ = 10)
- log(1) = 0 (because 10⁰ = 1)

**When to Use:**
- Scientific calculations
- Decibel calculations (logarithmic scale)
- pH calculations (negative log)
- Orders of magnitude comparisons

**2. Base e (Natural Logarithm): log_e(x) or ln(x)**

**Properties:**
- e ≈ 2.71828 (Euler's number, a fundamental constant)
- Most natural for calculus and continuous growth
- Derivative of ln(x) is 1/x (simplest possible!)
- Used in exponential growth/decay models

**Why e is Special:**
- e is the unique number where d/dx(e^x) = e^x
- Natural for continuous compounding
- Appears in many natural phenomena

**Examples:**
- ln(e) = 1 (by definition)
- ln(1) = 0 (because e⁰ = 1)
- ln(e²) = 2
- ln(10) ≈ 2.3026

**When to Use:**
- Calculus and mathematical analysis
- Continuous growth models (population, investments)
- Probability and statistics
- Machine learning (many algorithms use natural log)

**3. Base 2 (Binary Logarithm): log₂(x) or lg(x)**

**Properties:**
- Most important in computer science!
- Directly related to binary representation
- Answers "how many bits needed?"
- Used in algorithm complexity analysis

**Examples:**
- log₂(8) = 3 (because 2³ = 8, and 8 = 1000₂ needs 4 bits, but we count as 3+1)
- log₂(1024) = 10 (because 2¹⁰ = 1024)
- log₂(1) = 0 (because 2⁰ = 1)
- log₂(2) = 1 (because 2¹ = 2)

**When to Use:**
- Algorithm complexity (binary search, tree operations)
- Information theory (bits, entropy)
- Computer architecture (addressing, memory)
- Divide-and-conquer algorithms

**Why Base 2 Matters Most in CS:**

**1. Binary Representation:**
- n bits can represent 2ⁿ different values
- log₂(n) tells us how many bits we need
- Example: To represent 1000 values, need ⌈log₂(1000)⌉ = 10 bits

**2. Algorithm Complexity:**
- Binary search: O(log₂ n) - halve search space each step
- Balanced binary trees: Height is O(log₂ n)
- Divide-and-conquer: Often O(n log₂ n)

**3. Information Content:**
- log₂(n) bits needed to distinguish n possibilities
- Entropy calculations use base 2
- Data compression relies on logarithms

**Converting Between Bases:**

**Change of Base Formula:** log_b(a) = log_c(a) / log_c(b)

**Why This Works:**
If log_b(a) = x, then b^x = a
Taking log_c of both sides: log_c(b^x) = log_c(a)
Using power rule: x × log_c(b) = log_c(a)
Solving: x = log_c(a) / log_c(b)

**Examples:**

**Convert log₂(8) to base 10:**
- log₂(8) = log₁₀(8) / log₁₀(2) ≈ 0.903 / 0.301 ≈ 3 ✓

**Convert log₁₀(100) to base 2:**
- log₁₀(100) = log₂(100) / log₂(10) ≈ 6.644 / 3.322 ≈ 2 ✓

**Why This Matters:**
- Programming languages often provide only natural log or base 10
- Can compute any logarithm using change of base
- Example: log₂(x) = ln(x) / ln(2) = log₁₀(x) / log₁₀(2)

**Key Logarithm Properties:**

**1. Product Rule: log_b(xy) = log_b(x) + log_b(y)**
- Logarithms turn multiplication into addition
- Example: log₂(8 × 4) = log₂(8) + log₂(4) = 3 + 2 = 5
- Verification: log₂(32) = 5 ✓

**2. Quotient Rule: log_b(x/y) = log_b(x) - log_b(y)**
- Logarithms turn division into subtraction
- Example: log₂(8/2) = log₂(8) - log₂(2) = 3 - 1 = 2
- Verification: log₂(4) = 2 ✓

**3. Power Rule: log_b(x^y) = y × log_b(x)**
- Logarithms turn exponentiation into multiplication
- Example: log₂(8²) = 2 × log₂(8) = 2 × 3 = 6
- Verification: log₂(64) = 6 ✓

**4. Change of Base: log_b(a) = log_c(a) / log_c(b)**
- Allows conversion between bases
- Essential for computation

**Applications in Computer Science:**

**1. Algorithm Complexity Analysis:**

**Binary Search: O(log n)**
- Each step halves the search space
- log₂(n) steps to find element in sorted array of n elements
- Example: Finding item in 1,000,000 elements takes ~20 comparisons!

**Balanced Trees:**
- Height of balanced binary tree: O(log n)
- AVL trees, red-black trees maintain this property
- Enables O(log n) search, insert, delete operations

**Divide-and-Conquer:**
- Merge sort: O(n log n)
- Quick sort (average case): O(n log n)
- Many efficient algorithms use logarithmic complexity

**2. Information Theory:**

**Bits Needed:**
- To represent n distinct values: ⌈log₂(n)⌉ bits
- Example: 256 values need 8 bits (log₂(256) = 8)
- This is fundamental to data representation

**Entropy:**
- Information content: -Σ P(x) × log₂(P(x))
- Measures uncertainty/randomness
- Used in data compression, cryptography

**3. Exponential Growth Analysis:**

**Population Growth:**
- If population doubles every year: P(t) = P₀ × 2^t
- Time to reach certain size: t = log₂(P/P₀)
- Logarithms help analyze growth rates

**Algorithm Performance:**
- Some algorithms have exponential time complexity
- Logarithms help understand when they become impractical
- Example: O(2ⁿ) vs O(n²) - log helps compare

**4. Data Structures:**

**Tree Height:**
- Perfect binary tree with n nodes: height = ⌊log₂(n)⌋
- Understanding tree structure requires logarithms
- Helps optimize tree-based data structures

**Hash Tables:**
- Load factor analysis uses logarithms
- Understanding collision probability
- Optimizing table size

**5. Cryptography:**

**Discrete Logarithm Problem:**
- Security of many cryptographic systems
- Finding x where g^x ≡ h (mod p) is hard
- Basis for Diffie-Hellman, ElGamal

**Key Size Analysis:**
- Security often related to logarithm of key space
- Understanding why larger keys are more secure

**Common Mistakes:**

**Mistake 1: Confusing log with natural log**
- log(x) usually means log₁₀(x) in math
- ln(x) means log_e(x)
- In programming, log() often means natural log - check your language!

**Mistake 2: Forgetting log(1) = 0 for any base**
- b⁰ = 1 for any b, so log_b(1) = 0
- This is always true, regardless of base

**Mistake 3: Incorrectly applying power rule**
- Wrong: log(x^y) = log(x)^y
- Correct: log(x^y) = y × log(x)
- The exponent moves to the front as a multiplier!

**Mistake 4: Assuming log(x + y) = log(x) + log(y)**
- Wrong: Logarithms don't distribute over addition
- Correct: log(xy) = log(x) + log(y) (product rule)
- Only works for multiplication, not addition!

**Why This Matters:**

Logarithms are everywhere in computer science:
- **Algorithm Analysis:** Understanding O(log n) complexity
- **Information Theory:** Measuring information content
- **Data Structures:** Tree heights, hash table analysis
- **Cryptography:** Security of cryptographic systems
- **Performance:** Analyzing exponential vs polynomial growth

**Next Steps:**

Mastering logarithms provides the foundation for understanding algorithm complexity, information theory, and exponential processes. In the next lesson, we'll explore properties of logarithms in more detail, including how to solve logarithmic equations and apply them to real problems.`,
					CodeExamples: `# Working with logarithms in Python

import math

# Natural logarithm (base e)
print("Natural logarithm:")
print(f"ln(e) = {math.log(math.e)}")
print(f"ln(1) = {math.log(1)}")
print(f"ln(10) = {math.log(10):.4f}")

# Base 10 logarithm
print("\nBase 10 logarithm:")
print(f"log₁₀(100) = {math.log10(100)}")
print(f"log₁₀(1000) = {math.log10(1000)}")

# Base 2 logarithm (important for CS)
print("\nBase 2 logarithm:")
print(f"log₂(8) = {math.log2(8)}")
print(f"log₂(1024) = {math.log2(1024)}")
print(f"log₂(1) = {math.log2(1)}")

# Logarithm with arbitrary base
def log_base(x, base):
    """Calculate log_base(x)."""
    return math.log(x) / math.log(base)

print("\nArbitrary base:")
print(f"log₅(25) = {log_base(25, 5):.4f}")
print(f"log₃(27) = {log_base(27, 3):.4f}")

# Verify: log_b(a) = c means b^c = a
base, result = 2, 8
log_value = math.log2(result)
print(f"\nVerification: log₂({result}) = {log_value}")
print(f"Check: {base}^{log_value} = {base**log_value}")

# Bits needed to represent n values
def bits_needed(n):
    """Calculate bits needed to represent n distinct values."""
    return math.ceil(math.log2(n))

print(f"\nBits needed:")
for n in [8, 16, 100, 256]:
    print(f"  {n} values need {bits_needed(n)} bits")`,
				},
				{
					Title: "Properties of Logarithms",
					Content: `**Introduction: The Rules That Make Logarithms Powerful**

Logarithm properties transform complex calculations into simpler ones. They turn multiplication into addition, division into subtraction, and exponentiation into multiplication. These properties are essential for simplifying expressions, analyzing algorithms, and performing stable numerical computations.

**Understanding Logarithm Properties: The Transformation Rules**

Logarithm properties follow directly from exponent properties, since logarithms are the inverse of exponentiation. Each property has a corresponding exponent property.

**1. Product Rule: Multiplication Becomes Addition**

**Rule:** log_b(xy) = log_b(x) + log_b(y)

**Why This Works:**

**Derivation from Exponents:**
- Let log_b(x) = m, so b^m = x
- Let log_b(y) = n, so b^n = y
- Then xy = b^m × b^n = b^(m+n)
- Therefore: log_b(xy) = m + n = log_b(x) + log_b(y) ✓

**Step-by-Step Example: log₂(8 × 4)**

**Method 1: Using product rule**
- log₂(8 × 4) = log₂(8) + log₂(4)
- = 3 + 2 = 5

**Method 2: Direct calculation**
- log₂(8 × 4) = log₂(32) = 5 (since 2⁵ = 32) ✓

**Why This Matters:**
- **Avoid Overflow:** Instead of computing large product then log, compute logs then add
- **Numerical Stability:** Adding logs is more stable than multiplying then logging
- **Simplification:** Break complex products into manageable parts

**Example: log(1000 × 100 × 10)**
- Instead of: log(1,000,000) = 6
- Use: log(1000) + log(100) + log(10) = 3 + 2 + 1 = 6 ✓

**2. Quotient Rule: Division Becomes Subtraction**

**Rule:** log_b(x/y) = log_b(x) - log_b(y)

**Why This Works:**

**Derivation from Exponents:**
- Let log_b(x) = m, so b^m = x
- Let log_b(y) = n, so b^n = y
- Then x/y = b^m / b^n = b^(m-n)
- Therefore: log_b(x/y) = m - n = log_b(x) - log_b(y) ✓

**Step-by-Step Example: log₂(8/4)**

**Method 1: Using quotient rule**
- log₂(8/4) = log₂(8) - log₂(4)
- = 3 - 2 = 1

**Method 2: Direct calculation**
- log₂(8/4) = log₂(2) = 1 (since 2¹ = 2) ✓

**Why This Matters:**
- **Simplify Fractions:** Break complex fractions into difference of logs
- **Ratio Analysis:** Compare ratios using log differences
- **Stability:** More stable than dividing then logging

**3. Power Rule: Exponentiation Becomes Multiplication**

**Rule:** log_b(xⁿ) = n × log_b(x)

**Why This Works:**

**Derivation from Exponents:**
- Let log_b(x) = m, so b^m = x
- Then xⁿ = (b^m)ⁿ = b^(mn)
- Therefore: log_b(xⁿ) = mn = n × log_b(x) ✓

**Step-by-Step Example: log₂(8²)**

**Method 1: Using power rule**
- log₂(8²) = 2 × log₂(8)
- = 2 × 3 = 6

**Method 2: Direct calculation**
- log₂(8²) = log₂(64) = 6 (since 2⁶ = 64) ✓

**Why This Matters:**
- **Simplify Powers:** Bring exponents down as multipliers
- **Algorithm Analysis:** log(n²) = 2log(n) → still O(log n)
- **Roots:** log(√x) = log(x^(1/2)) = (1/2)log(x)

**4. Change of Base: Converting Between Bases**

**Rule:** log_b(x) = log_c(x) / log_c(b)

**Why This Works:**

**Derivation:**
- Let log_b(x) = y, so b^y = x
- Take log_c of both sides: log_c(b^y) = log_c(x)
- Using power rule: y × log_c(b) = log_c(x)
- Solve: y = log_c(x) / log_c(b)
- Therefore: log_b(x) = log_c(x) / log_c(b) ✓

**Step-by-Step Example: log₂(8) using base 10**

- log₂(8) = log₁₀(8) / log₁₀(2)
- = 0.903 / 0.301 ≈ 3 ✓

**Why This Matters:**
- **Computation:** Most calculators/languages provide only natural log or base 10
- **Flexibility:** Can compute any logarithm using available functions
- **Programming:** log₂(x) = log(x) / log(2) or ln(x) / ln(2)

**Special Values: Fundamental Identities**

**1. log_b(1) = 0**
- Because b⁰ = 1 for any base b
- The logarithm of 1 is always 0, regardless of base

**2. log_b(b) = 1**
- Because b¹ = b
- The logarithm of the base is always 1

**3. log_b(bⁿ) = n**
- Because bⁿ = bⁿ
- Direct application of power rule: n × log_b(b) = n × 1 = n

**4. b^(log_b(x)) = x**
- This is the inverse property
- Exponentiating and taking log (same base) cancel out

**Applications in Computer Science:**

**1. Algorithm Complexity Analysis:**

**Simplifying Complexity Expressions:**
- log(n²) = 2log(n) → still O(log n)
- log(n³) = 3log(n) → still O(log n)
- Powers don't change asymptotic complexity!

**Example: Binary search in sorted 2D array**
- Search space: n² elements
- Complexity: O(log(n²)) = O(2log(n)) = O(log n)
- Same as 1D binary search!

**2. Numerical Stability:**

**Avoiding Overflow:**
- Instead of: product = a × b × c × ... (might overflow)
- Use: log(product) = log(a) + log(b) + log(c) + ...
- Then: product = exp(sum of logs)
- More stable for large numbers!

**Example: Calculating large product**
- Product of 1000 numbers, each around 10
- Direct: 10^1000 (huge, might overflow)
- Using logs: 1000 × log(10) = 1000, then exp(1000)
- More manageable!

**3. Probability and Information Theory:**

**Log-Likelihood:**
- Instead of multiplying probabilities: P = p₁ × p₂ × ... × pₙ
- Use log-likelihood: log(P) = log(p₁) + log(p₂) + ... + log(pₙ)
- Avoids underflow with many small probabilities

**Entropy Calculations:**
- H = -Σ P(x) × log(P(x))
- Uses logarithm properties for efficient computation

**4. Machine Learning:**

**Loss Functions:**
- Many use logarithms (log loss, cross-entropy)
- Properties help simplify gradient calculations
- log(ab) = log(a) + log(b) simplifies derivatives

**Feature Scaling:**
- Log transformation: y = log(x)
- Properties help understand transformed data
- log(xy) = log(x) + log(y) shows multiplicative relationships become additive

**5. Data Compression:**

**Information Content:**
- I(x) = -log₂(P(x)) bits
- Properties help calculate total information
- Sum of logs = log of product

**Common Mistakes:**

**Mistake 1: Incorrectly applying product rule**
- Wrong: log(x + y) = log(x) + log(y)
- Correct: log(xy) = log(x) + log(y)
- Only works for multiplication, not addition!

**Mistake 2: Confusing power rule**
- Wrong: log(x^y) = log(x)^y
- Correct: log(x^y) = y × log(x)
- Exponent becomes multiplier, not exponent of log!

**Mistake 3: Forgetting special values**
- log(1) = 0 always (not undefined!)
- log(base) = 1 always
- These are fundamental identities

**Mistake 4: Incorrect change of base**
- Wrong: log_b(x) = log_c(x) × log_c(b)
- Correct: log_b(x) = log_c(x) / log_c(b)
- Division, not multiplication!

**Why This Matters:**

Logarithm properties are essential for:
- **Simplifying Expressions:** Making complex calculations manageable
- **Algorithm Analysis:** Understanding complexity relationships
- **Numerical Computation:** Stable and efficient calculations
- **Theoretical Work:** Proving theorems and deriving formulas

**Next Steps:**

Mastering logarithm properties provides the foundation for solving logarithmic equations, analyzing algorithms, and working with exponential models. In the next lesson, we'll explore exponential functions and how they relate to logarithms.`,
					CodeExamples: `# Logarithm properties

import math

# Product rule: log(xy) = log(x) + log(y)
x, y = 8, 4
product_log = math.log2(x * y)
sum_logs = math.log2(x) + math.log2(y)
print(f"Product rule:")
print(f"log₂({x} × {y}) = {product_log}")
print(f"log₂({x}) + log₂({y}) = {sum_logs}")
print(f"Equal: {abs(product_log - sum_logs) < 1e-10}")

# Quotient rule: log(x/y) = log(x) - log(y)
quotient_log = math.log2(x / y)
diff_logs = math.log2(x) - math.log2(y)
print(f"\nQuotient rule:")
print(f"log₂({x}/{y}) = {quotient_log}")
print(f"log₂({x}) - log₂({y}) = {diff_logs}")
print(f"Equal: {abs(quotient_log - diff_logs) < 1e-10}")

# Power rule: log(x^n) = n × log(x)
n = 3
power_log = math.log2(x ** n)
n_times_log = n * math.log2(x)
print(f"\nPower rule:")
print(f"log₂({x}^{n}) = {power_log}")
print(f"{n} × log₂({x}) = {n_times_log}")
print(f"Equal: {abs(power_log - n_times_log) < 1e-10}")

# Change of base
def change_of_base(x, old_base, new_base):
    """Change logarithm base."""
    return math.log(x, new_base) / math.log(old_base, new_base)

print(f"\nChange of base:")
log2_8 = math.log2(8)
log10_8_over_log10_2 = change_of_base(8, 2, 10)
print(f"log₂(8) = {log2_8}")
print(f"log₁₀(8) / log₁₀(2) = {log10_8_over_log10_2}")
print(f"Equal: {abs(log2_8 - log10_8_over_log10_2) < 1e-10}")

# Special values
print(f"\nSpecial values:")
print(f"log₂(1) = {math.log2(1)}")
print(f"log₂(2) = {math.log2(2)}")
print(f"log₂(2^5) = {math.log2(2**5)} = 5")

# Application: Simplify log(product)
def log_of_product(numbers):
    """Calculate log of product using sum of logs."""
    return sum(math.log2(n) for n in numbers)

numbers = [2, 4, 8, 16]
product = 1
for n in numbers:
    product *= n

log_product_direct = math.log2(product)
log_product_sum = log_of_product(numbers)
print(f"\nLog of product:")
print(f"log₂({product}) = {log_product_direct}")
print(f"sum of logs = {log_product_sum}")`,
				},
				{
					Title: "Exponential Functions",
					Content: `**Introduction: The Power of Exponential Change**

Exponential functions model rapid growth and decay, appearing everywhere from population dynamics to algorithm complexity. Understanding exponential functions is crucial for analyzing exponential algorithms, modeling growth processes, and understanding why some problems become intractable as size increases.

**Understanding Exponential Functions: Rapid Change**

An exponential function has the form f(x) = aˣ where:
- **a** is the base (a > 0, a ≠ 1)
- **x** is the exponent (can be any real number)
- The variable is in the exponent, not the base

**Key Insight:** In exponential functions, the rate of change is proportional to the current value. This creates explosive growth or rapid decay.

**Key Properties:**

**1. Growth Rate: The Base Determines Behavior**

**If a > 1: Exponential Growth**
- Function increases as x increases
- Gets steeper and steeper
- Example: f(x) = 2ˣ
  - f(0) = 1, f(1) = 2, f(2) = 4, f(3) = 8, f(4) = 16
  - Doubles each time x increases by 1

**If 0 < a < 1: Exponential Decay**
- Function decreases as x increases
- Approaches 0 but never reaches it
- Example: f(x) = (1/2)ˣ
  - f(0) = 1, f(1) = 0.5, f(2) = 0.25, f(3) = 0.125
  - Halves each time x increases by 1

**Why a ≠ 1?**
- If a = 1, then f(x) = 1ˣ = 1 (constant function, not exponential)
- If a ≤ 0, then aˣ is not defined for all real x

**2. Domain and Range:**

**Domain:** All real numbers (-∞, ∞)
- Can raise positive number to any real power
- Example: 2^(-3) = 1/8, 2^(1/2) = √2, 2^π ≈ 8.825

**Range:** (0, ∞) for a > 0
- aˣ is always positive (never zero or negative)
- Approaches 0 as x → -∞ (for a > 1)
- Approaches ∞ as x → ∞ (for a > 1)

**3. Key Point: Always Passes Through (0, 1)**

**Why?** Because a⁰ = 1 for any base a
- This is true for all exponential functions
- The y-intercept is always (0, 1)
- Useful for graphing and identifying exponential functions

**Natural Exponential: The Special Base e**

The natural exponential function uses e ≈ 2.71828 as the base.

**Function:** f(x) = eˣ

**Why e is Special:**

**1. Calculus Property:**
- The derivative of eˣ is eˣ itself!
- d/dx(eˣ) = eˣ
- This is unique - no other function has this property
- Makes calculus with exponentials elegant

**2. Natural Growth:**
- e appears naturally in continuous growth models
- When growth is continuous (not discrete), e emerges
- Many natural processes follow e-based patterns

**3. Logarithm Connection:**
- e is the base for natural logarithms
- ln(x) = log_e(x)
- e and ln are inverse functions: e^(ln(x)) = x, ln(eˣ) = x

**4. Mathematical Elegance:**
- e = lim(n→∞) (1 + 1/n)ⁿ
- e = Σ(n=0 to ∞) 1/n! = 1 + 1 + 1/2 + 1/6 + 1/24 + ...
- Appears in many mathematical contexts

**Compound Interest: Exponential Growth in Finance**

Compound interest is a practical application of exponential functions.

**Discrete Compounding:** A = P(1 + r/n)^(nt)
- **P:** Principal (initial amount)
- **r:** Annual interest rate (as decimal)
- **n:** Number of times compounded per year
- **t:** Time in years
- **A:** Final amount

**Step-by-Step Example:**
- Invest $1000 at 5% annual rate, compounded monthly for 2 years
- P = 1000, r = 0.05, n = 12, t = 2
- A = 1000(1 + 0.05/12)^(12×2)
- A = 1000(1.004167)^24 ≈ $1104.94

**Continuous Compounding:** A = Pe^(rt)
- When compounding happens continuously (infinitely often)
- More accurate for very frequent compounding
- Uses natural exponential

**Example:**
- Same investment with continuous compounding
- A = 1000e^(0.05×2) = 1000e^0.1 ≈ $1105.17
- Slightly more than monthly compounding

**Why Continuous?**
- As n → ∞, (1 + r/n)^(nt) → e^(rt)
- Continuous compounding gives maximum growth
- Used in theoretical finance and economics

**Exponential Growth and Decay Models:**

**Exponential Growth:** N(t) = N₀ × e^(rt) where r > 0
- **N₀:** Initial quantity
- **r:** Growth rate (positive)
- **t:** Time
- **N(t):** Quantity at time t

**Example: Population Growth**
- Initial population: 1000
- Growth rate: 2% per year (r = 0.02)
- After 10 years: N(10) = 1000 × e^(0.02×10) = 1000e^0.2 ≈ 1221

**Exponential Decay:** N(t) = N₀ × e^(-rt) where r > 0
- Same form, but negative exponent
- Quantity decreases over time
- Approaches 0 as t → ∞

**Example: Radioactive Decay**
- Initial amount: 100 grams
- Decay rate: 0.05 per year
- After 20 years: N(20) = 100 × e^(-0.05×20) = 100e^(-1) ≈ 36.8 grams

**Half-Life: Time to Halve**

The half-life is the time it takes for a quantity to reduce to half its initial value.

**Formula:** For decay N(t) = N₀e^(-rt), half-life is:
- t_half = ln(2) / r

**Derivation:**
- We want N(t_half) = N₀/2
- N₀e^(-r×t_half) = N₀/2
- e^(-r×t_half) = 1/2
- -r×t_half = ln(1/2) = -ln(2)
- t_half = ln(2) / r ✓

**Example:**
- Decay rate r = 0.05 per year
- Half-life = ln(2) / 0.05 ≈ 13.86 years
- After ~13.86 years, half the substance remains

**Applications in Computer Science:**

**1. Algorithm Complexity: Exponential Time**

**Exponential Algorithms:**
- Time complexity: O(2ⁿ), O(3ⁿ), etc.
- Become impractical very quickly
- Example: Brute force for some problems

**Why Exponential is Bad:**
- 2¹⁰ = 1,024 operations (manageable)
- 2²⁰ = 1,048,576 operations (slow)
- 2³⁰ = 1,073,741,824 operations (impractical!)
- 2⁴⁰ = over 1 trillion operations (impossible)

**Example: Traveling Salesman Problem**
- Naive solution: Try all permutations
- n cities → n! possible routes
- n! grows faster than 2ⁿ (super-exponential!)
- Need better algorithms (heuristics, approximation)

**2. Computer Virus Spread:**

**Exponential Growth Model:**
- Each infected computer infects r others per time unit
- N(t) = N₀ × (1 + r)^t
- Can spread very rapidly!

**Example:**
- Start with 1 infected computer
- Each infects 2 others per day (r = 2)
- After 10 days: N(10) = 1 × 3¹⁰ = 59,049 infected!
- Shows why early containment is crucial

**3. Population Growth:**

**Modeling User Growth:**
- Social networks, online platforms
- User base grows exponentially initially
- N(t) = N₀e^(rt)
- Helps predict server capacity needs

**4. Memory and Storage:**

**Exponential Growth in Data:**
- Data generation grows exponentially
- Storage needs: S(t) = S₀e^(rt)
- Planning for future capacity

**5. Recursive Algorithms:**

**Exponential Recursion:**
- Some recursive algorithms make exponential calls
- Example: Naive Fibonacci: F(n) = F(n-1) + F(n-2)
- Makes ~2ⁿ function calls
- Need memoization or iterative approach

**6. Cryptography:**

**Exponential Difficulty:**
- Security relies on exponential difficulty
- Breaking encryption: O(2^key_length)
- Doubling key length squares the work
- Makes brute force attacks impractical

**Common Mistakes:**

**Mistake 1: Confusing exponential with polynomial**
- Exponential: 2ⁿ (variable in exponent)
- Polynomial: n² (variable in base)
- Very different growth rates!

**Mistake 2: Forgetting domain restrictions**
- Base must be positive: a > 0
- Base cannot be 1: a ≠ 1
- Check these when working with exponentials

**Mistake 3: Incorrectly applying exponent rules**
- (aˣ)ʸ = a^(xy), not a^(x+y)
- aˣ × aʸ = a^(x+y)
- Don't confuse these!

**Mistake 4: Not recognizing exponential decay**
- Decay uses negative exponent: e^(-rt)
- Or base < 1: (1/2)ˣ
- Both represent decay

**Why This Matters:**

Exponential functions model:
- **Rapid Growth:** Populations, data, viruses
- **Rapid Decay:** Radioactivity, memory, signals
- **Algorithm Complexity:** Understanding exponential time
- **Financial Models:** Interest, investments
- **Natural Phenomena:** Many processes follow exponential patterns

**Next Steps:**

Mastering exponential functions provides the foundation for understanding exponential algorithms, growth models, and the relationship between exponentials and logarithms. In the next lesson, we'll explore solving exponential and logarithmic equations, which combines these concepts.`,
					CodeExamples: `# Exponential functions

import math

# Basic exponential
def exponential(base, exponent):
    """Calculate base^exponent."""
    return base ** exponent

print("Exponential functions:")
print(f"2^3 = {exponential(2, 3)}")
print(f"2^0 = {exponential(2, 0)}")
print(f"2^(-2) = {exponential(2, -2)}")

# Natural exponential e^x
print(f"\nNatural exponential:")
for x in [0, 1, 2, -1]:
    print(f"e^{x} = {math.exp(x):.4f}")

# Exponential growth
def exponential_growth(initial, rate, time):
    """Calculate exponential growth: N(t) = N₀ × e^(rt)."""
    return initial * math.exp(rate * time)

initial_pop = 1000
growth_rate = 0.05  # 5% per year
print(f"\nExponential growth (initial={initial_pop}, rate={growth_rate}):")
for t in [0, 1, 5, 10]:
    population = exponential_growth(initial_pop, growth_rate, t)
    print(f"  Year {t}: {population:.0f}")

# Exponential decay
def exponential_decay(initial, rate, time):
    """Calculate exponential decay: N(t) = N₀ × e^(-rt)."""
    return initial * math.exp(-rate * time)

def half_life(decay_rate):
    """Calculate half-life for exponential decay."""
    return math.log(2) / decay_rate

initial_amount = 100
decay_rate = 0.1  # 10% per time unit
half_life_time = half_life(decay_rate)
print(f"\nExponential decay (initial={initial_amount}, rate={decay_rate}):")
print(f"Half-life: {half_life_time:.2f} time units")
for t in [0, half_life_time, 2*half_life_time]:
    amount = exponential_decay(initial_amount, decay_rate, t)
    print(f"  Time {t:.2f}: {amount:.2f}")

# Compound interest
def compound_interest(principal, rate, time, compounding_freq=1):
    """Calculate compound interest."""
    return principal * (1 + rate/compounding_freq) ** (compounding_freq * time)

def continuous_compound(principal, rate, time):
    """Calculate continuous compound interest."""
    return principal * math.exp(rate * time)

p = 1000
r = 0.05
t = 10
print(f"\nCompound interest (P=${p}, r={r}, t={t} years):")
print(f"Annual: ${compound_interest(p, r, t):.2f}")
print(f"Monthly: ${compound_interest(p, r, t, 12):.2f}")
print(f"Continuous: ${continuous_compound(p, r, t):.2f}")`,
				},
				{
					Title: "Solving Exponential and Logarithmic Equations",
					Content: `**Introduction: Unlocking Exponents and Logarithms**

Solving exponential and logarithmic equations is essential for algorithm analysis, especially when analyzing recursive algorithms and solving recurrence relations. These equations appear when we need to find how many times an operation repeats or how long an algorithm takes.

**Solving Exponential Equations: Finding the Power**

Exponential equations have the variable in the exponent, making them trickier than linear equations.

**General Form:** aˣ = b (where a > 0, a ≠ 1)

**Method 1: Same Base - The Simplest Case**

**When to Use:** Both sides can be written with the same base

**Principle:** If aˣ = aʸ, then x = y (for a > 0, a ≠ 1)

**Why This Works:**
- Exponential functions are one-to-one (injective)
- If outputs are equal, inputs must be equal
- Only works when bases are the same!

**Step-by-Step Example: 2ˣ = 2³**

- Both sides have base 2
- Since bases are equal, exponents must be equal
- Therefore: x = 3 ✓

**More Complex Example: 4ˣ = 8**

**Step 1: Express both sides with same base**
- 4 = 2², so 4ˣ = (2²)ˣ = 2²ˣ
- 8 = 2³
- Equation becomes: 2²ˣ = 2³

**Step 2: Equate exponents**
- 2x = 3
- x = 3/2 = 1.5 ✓

**Method 2: Take Logarithm - The General Method**

**When to Use:** Can't easily express both sides with same base

**Principle:** If aˣ = b, then x = log_a(b)

**Why This Works:**
- Logarithms are inverse of exponentials
- log_a(aˣ) = x (by definition)
- Taking log of both sides isolates the exponent

**Step-by-Step Example: 2ˣ = 8**

**Step 1: Take logarithm of both sides**
- log₂(2ˣ) = log₂(8)

**Step 2: Simplify using log properties**
- Left: log₂(2ˣ) = x × log₂(2) = x × 1 = x
- Right: log₂(8) = log₂(2³) = 3
- Therefore: x = 3 ✓

**Using Natural Log: 2ˣ = 8**

**Alternative approach:**
- ln(2ˣ) = ln(8)
- x × ln(2) = ln(8)
- x = ln(8) / ln(2) = 2.079 / 0.693 ≈ 3 ✓
- Uses change of base formula

**Method 3: Isolate Exponent - Step-by-Step**

**When to Use:** Exponent is not alone (coefficient or other terms)

**Step-by-Step Example: 3 × 2ˣ = 24**

**Step 1: Isolate the exponential term**
- Divide both sides by 3: 2ˣ = 8

**Step 2: Solve using Method 1 or 2**
- 2ˣ = 8 = 2³
- Therefore: x = 3 ✓

**More Complex Example: 5 × 3ˣ - 2 = 43**

**Step 1: Isolate exponential term**
- Add 2: 5 × 3ˣ = 45
- Divide by 5: 3ˣ = 9

**Step 2: Solve**
- 3ˣ = 9 = 3²
- Therefore: x = 2 ✓

**Solving Logarithmic Equations: Finding the Argument**

Logarithmic equations have the variable inside a logarithm.

**General Form:** log_b(x) = c

**Method 1: Convert to Exponential - The Direct Method**

**Principle:** If log_b(x) = c, then x = b^c

**Why This Works:**
- This is the definition of logarithm!
- log_b(x) = c means b^c = x

**Step-by-Step Example: log₂(x) = 3**

**Convert to exponential:**
- log₂(x) = 3 means 2³ = x
- Therefore: x = 8 ✓

**Verification:**
- Check: log₂(8) = log₂(2³) = 3 ✓

**Method 2: Use Properties - Combining Logs**

**When to Use:** Multiple logarithms that can be combined

**Step-by-Step Example: log(x) + log(3) = log(12)**

**Step 1: Combine logarithms**
- Using product rule: log(x) + log(3) = log(3x)
- Equation becomes: log(3x) = log(12)

**Step 2: Equate arguments**
- Since logs are equal and base is same, arguments equal
- 3x = 12
- x = 4 ✓

**Step 3: Check domain**
- log(x) requires x > 0
- x = 4 > 0 ✓ (valid solution)

**More Complex Example: log₂(x + 1) + log₂(x - 1) = 3**

**Step 1: Combine logs**
- log₂[(x + 1)(x - 1)] = 3
- log₂(x² - 1) = 3

**Step 2: Convert to exponential**
- x² - 1 = 2³ = 8
- x² = 9
- x = ±3

**Step 3: Check domain**
- log₂(x + 1) requires x + 1 > 0, so x > -1
- log₂(x - 1) requires x - 1 > 0, so x > 1
- Combined: x > 1
- x = 3 > 1 ✓ (valid)
- x = -3 < 1 ✗ (invalid, reject)

**Solution:** x = 3

**Method 3: Check Domain - Critical Step!**

**Domain Restriction:** Logarithms only defined for positive arguments

**Always Check:**
- For log_b(x): x > 0
- For log_b(x - a): x - a > 0, so x > a
- For log_b(f(x)): f(x) > 0

**Why This Matters:**
- Invalid solutions often appear when solving
- Must verify solutions satisfy domain restrictions
- Example: log(x) = -2 → x = 10^(-2) = 0.01 > 0 ✓

**Common Mistakes:**

**Mistake 1: Forgetting domain restrictions**
- Wrong: Solve log(x) = -5, get x = 10^(-5), don't check
- Correct: Always verify x > 0
- In this case: 10^(-5) > 0 ✓ (actually valid, but always check!)

**Mistake 2: Incorrectly applying log properties**
- Wrong: log(x + y) = log(x) + log(y)
- Correct: log(xy) = log(x) + log(y)
- log(x + y) cannot be simplified!

**Mistake 3: Not checking solutions**
- Always substitute back into original equation
- Some operations (like squaring) can introduce extraneous solutions
- Domain restrictions can eliminate solutions

**Mistake 4: Confusing log bases**
- log(x) usually means log₁₀(x) in math
- ln(x) means log_e(x)
- In programming, log() often means natural log - check your language!

**Applications in Computer Science:**

**1. Algorithm Analysis: Solving Recurrence Relations**

**Example: Binary Search Recurrence**
- T(n) = T(n/2) + 1
- To find when T(n) = k: n/2^k = 1
- Taking log: log₂(n) - k = 0
- Therefore: k = log₂(n)
- Confirms binary search is O(log n)

**2. Time Complexity Calculations:**

**Example: When does 2ⁿ exceed n²?**
- Solve: 2ⁿ = n²
- Take log: n = 2log₂(n)
- For large n, 2ⁿ grows faster
- Crossover point around n = 4

**3. Growth Rate Analysis:**

**Example: Comparing algorithms**
- Algorithm A: T(n) = n²
- Algorithm B: T(n) = 2ⁿ
- When is A faster? Solve: n² < 2ⁿ
- Take log: 2log(n) < n
- For n > 4, A is faster (until exponential takes over)

**4. Logarithmic Scales:**

**Example: Decibel calculations**
- dB = 10 × log₁₀(P/P₀)
- To find power ratio: P/P₀ = 10^(dB/10)
- Solving exponential equations!

**5. Interest and Growth:**

**Example: How long to double investment?**
- A = P(1 + r)^t
- Want A = 2P: 2P = P(1 + r)^t
- 2 = (1 + r)^t
- t = log(2) / log(1 + r)
- Rule of 72: t ≈ 72 / (100r) for small r

**Why This Matters:**

Solving these equations is needed for:
- **Algorithm Analysis:** Finding complexity from recurrences
- **Performance Prediction:** Determining when algorithms become slow
- **Optimization:** Finding optimal parameters
- **Growth Analysis:** Understanding how functions scale

**Next Steps:**

Mastering exponential and logarithmic equations provides the tools needed for analyzing recursive algorithms and understanding growth rates. In the next lesson, we'll explore applications in algorithm analysis, showing how these solving techniques apply directly to computer science problems.`,
					CodeExamples: `# Solving exponential and logarithmic equations

import math

# Solving exponential equations
def solve_exponential_same_base(base, exponent1, exponent2):
    """Solve a^x1 = a^x2."""
    if exponent1 == exponent2:
        return "All real numbers"
    return None  # No solution

def solve_exponential(base, result):
    """Solve base^x = result."""
    if result <= 0:
        return None
    return math.log(result, base)

# Example: 2^x = 8
x1 = solve_exponential(2, 8)
print(f"Solve 2^x = 8:")
print(f"  x = {x1}")
print(f"  Check: 2^{x1} = {2**x1}")

# Example: 3 × 2^x = 24
# First isolate: 2^x = 8
x2 = solve_exponential(2, 8)
print(f"\nSolve 3 × 2^x = 24:")
print(f"  2^x = 8")
print(f"  x = {x2}")

# Solving logarithmic equations
def solve_logarithmic(base, result):
    """Solve log_base(x) = result."""
    return base ** result

# Example: log_2(x) = 3
x3 = solve_logarithmic(2, 3)
print(f"\nSolve log₂(x) = 3:")
print(f"  x = {x3}")
print(f"  Check: log₂({x3}) = {math.log2(x3)}")

# Example: log(x) + log(3) = log(12)
# Using property: log(3x) = log(12)
# So: 3x = 12
x4 = 12 / 3
print(f"\nSolve log(x) + log(3) = log(12):")
print(f"  log(3x) = log(12)")
print(f"  3x = 12")
print(f"  x = {x4}")
print(f"  Check: log({x4}) + log(3) = {math.log10(x4) + math.log10(3):.4f}")
print(f"         log(12) = {math.log10(12):.4f}")

# More complex example: 2^(x+1) = 16
# Take log2: x + 1 = log2(16) = 4
# So: x = 3
x5 = math.log2(16) - 1
print(f"\nSolve 2^(x+1) = 16:")
print(f"  x + 1 = log₂(16) = {math.log2(16)}")
print(f"  x = {x5}")
print(f"  Check: 2^({x5}+1) = {2**(x5+1)}")`,
				},
				{
					Title: "Applications in Algorithm Analysis",
					Content: `**Introduction: Mathematics Meets Performance**

Logarithms and exponentials are fundamental to understanding algorithm complexity. They appear everywhere - from binary search to divide-and-conquer algorithms. Mastering these concepts helps you analyze algorithms, predict performance, and make informed design choices.

**Logarithms in Algorithm Analysis: The Power of Halving**

**1. Binary Search: The Classic Logarithmic Algorithm**

Binary search demonstrates why O(log n) is so powerful.

**How It Works:**
- Start with sorted array of n elements
- Compare target with middle element
- Eliminate half the search space
- Repeat until found or space exhausted

**Complexity Analysis:**
- After 1 step: n/2 elements remain
- After 2 steps: n/4 elements remain
- After k steps: n/2ᵏ elements remain
- When n/2ᵏ = 1: k = log₂(n)
- Therefore: O(log n) steps

**Why This Matters:**
- log₂(1,000,000) ≈ 20 comparisons
- Linear search: up to 1,000,000 comparisons
- Binary search: ~20 comparisons
- 50,000x improvement!

**Example: Finding word in dictionary**
- 1 million words
- Linear search: up to 1 million checks
- Binary search: ~20 checks
- Massive difference!

**2. Balanced Binary Trees: Logarithmic Height**

Tree data structures achieve O(log n) operations through balanced height.

**Perfect Binary Tree:**
- Level 0: 1 node
- Level 1: 2 nodes
- Level k: 2ᵏ nodes
- Total nodes: n = 2^(h+1) - 1
- Height: h ≈ log₂(n)

**Operations:**
- **Search:** Follow path from root to leaf: O(height) = O(log n)
- **Insert:** Find position + rebalance: O(log n)
- **Delete:** Find + remove + rebalance: O(log n)

**Examples:**
- **AVL Trees:** Self-balancing, height difference ≤ 1
- **Red-Black Trees:** Self-balancing, used in many libraries
- **B-Trees:** Used in databases, optimized for disk access

**Why Balance Matters:**
- Unbalanced tree: O(n) worst case (like linked list)
- Balanced tree: O(log n) guaranteed
- Balance ensures logarithmic performance

**3. Divide and Conquer: The n log n Pattern**

Many efficient algorithms use divide-and-conquer with logarithmic depth.

**Merge Sort:**
- Divide: Split array in half: O(1)
- Conquer: Sort halves recursively: 2T(n/2)
- Combine: Merge sorted halves: O(n)
- Recurrence: T(n) = 2T(n/2) + O(n)

**Solving the Recurrence:**
- Depth: log₂(n) levels (divide in half each time)
- Work per level: O(n) (merging)
- Total: O(n log n) ✓

**Other O(n log n) Algorithms:**
- **Quick Sort (average):** O(n log n)
- **Heap Sort:** O(n log n)
- **FFT (Fast Fourier Transform):** O(n log n)

**Master Theorem: Solving Recurrences Efficiently**

A powerful tool for analyzing divide-and-conquer algorithms.

**Form:** T(n) = aT(n/b) + f(n)
- a: number of subproblems
- b: size reduction factor
- f(n): cost to combine solutions

**Three Cases:**

**Case 1: f(n) = O(n^c) where c < log_b(a)**
- Combining is cheap compared to subproblems
- T(n) = O(n^log_b(a))
- Example: T(n) = 4T(n/2) + n, then T(n) = O(n²)

**Case 2: f(n) = O(n^log_b(a))**
- Combining cost matches subproblem cost
- T(n) = O(n^log_b(a) × log n)
- Example: T(n) = 2T(n/2) + n, then T(n) = O(n log n)

**Case 3: f(n) = Ω(n^c) where c > log_b(a)**
- Combining dominates
- T(n) = O(f(n))
- Example: T(n) = T(n/2) + n², then T(n) = O(n²)

**Example: Merge Sort**
- T(n) = 2T(n/2) + n
- a = 2, b = 2, f(n) = n
- log_b(a) = log₂(2) = 1
- f(n) = n = O(n¹) = O(n^log_b(a))
- Case 2 applies: T(n) = O(n log n) ✓

**Exponential Algorithms: When Problems Become Intractable**

Some problems have exponential complexity, making them impractical for large inputs.

**Brute Force: Generating All Possibilities**

**Example: Generating All Subsets**
- Set of n elements
- Each element: include or exclude (2 choices)
- Total subsets: 2ⁿ
- Time to generate: O(2ⁿ)

**Why Exponential is Bad:**
- n = 20: 2²⁰ = 1,048,576 (manageable)
- n = 30: 2³⁰ = 1,073,741,824 (slow)
- n = 40: 2⁴⁰ ≈ 1 trillion (impractical!)
- n = 50: 2⁵⁰ ≈ 1 quadrillion (impossible!)

**Example: Traveling Salesman Problem (Naive)**
- Try all possible routes
- n cities: (n-1)! possible routes
- n! grows faster than 2ⁿ (super-exponential!)
- Need better algorithms (heuristics, approximation)

**Why Logarithms Matter: Making the Impossible Possible**

**Growth Comparison:**

**log₂(n) grows very slowly:**
- log₂(10) ≈ 3.3
- log₂(100) ≈ 6.6
- log₂(1,000) ≈ 10
- log₂(1,000,000) ≈ 20
- log₂(1,000,000,000) ≈ 30

**This means:**
- Algorithms with O(log n) complexity scale beautifully
- Can handle billions of elements efficiently
- Makes many problems tractable

**Complexity Hierarchy:**
- O(1): Constant - best possible
- O(log n): Logarithmic - excellent
- O(n): Linear - good
- O(n log n): Linearithmic - acceptable
- O(n²): Quadratic - acceptable for small n
- O(2ⁿ): Exponential - avoid if possible!

**Real-World Examples:**

**1. Database Indexing:**
- B-tree indexes: O(log n) lookups
- Enables fast queries on huge databases
- Without indexes: O(n) sequential scan
- Difference: milliseconds vs hours!

**2. File Systems:**
- Directory trees: O(log n) depth
- Finding files: O(log n) path traversal
- Enables organizing millions of files

**3. Network Routing:**
- Routing tables: O(log n) lookup
- Binary search in sorted routing table
- Critical for internet performance

**4. Memory Management:**
- Buddy allocator: O(log n) allocation
- Uses binary tree structure
- Efficient memory management

**5. Compiler Design:**
- Symbol tables: O(log n) lookup
- Binary search trees for identifiers
- Fast compilation of large programs

**Why Understanding Complexity Matters:**

**Algorithm Selection:**
- Choose O(log n) over O(n) when possible
- Avoid O(2ⁿ) unless necessary
- Understand trade-offs

**Performance Prediction:**
- Predict runtime for large inputs
- Plan system capacity
- Set realistic expectations

**Optimization:**
- Identify bottlenecks
- Focus optimization efforts
- Measure impact of improvements

**Next Steps:**

Mastering algorithm analysis provides the foundation for designing efficient systems. The mathematical tools - logarithms, exponentials, sequences, and series - directly translate to understanding and improving algorithm performance.`,
					CodeExamples: `# Applications in algorithm analysis

import math

# Binary search complexity
def binary_search_steps(n):
    """Calculate steps needed for binary search."""
    return math.ceil(math.log2(n))

print("Binary search steps:")
for n in [10, 100, 1000, 1000000]:
    steps = binary_search_steps(n)
    print(f"  n = {n:,}: {steps} steps")

# Tree height
def balanced_tree_height(n_nodes):
    """Calculate height of balanced binary tree."""
    return math.ceil(math.log2(n_nodes + 1))

print(f"\nBalanced tree height:")
for n in [7, 15, 31, 1023]:
    height = balanced_tree_height(n)
    print(f"  {n} nodes: height {height}")

# Compare growth rates
def compare_growth_rates():
    """Compare different growth rates."""
    n_values = [10, 100, 1000, 10000]
    print("\nGrowth rate comparison:")
    print(f"{'n':<8} {'log n':<10} {'n':<10} {'n log n':<12} {'n²':<12} {'2ⁿ':<15}")
    print("-" * 70)
    for n in n_values:
        log_n = math.log2(n)
        n_log_n = n * math.log2(n)
        n_squared = n ** 2
        two_n = 2 ** n if n <= 20 else float('inf')
        print(f"{n:<8} {log_n:<10.2f} {n:<10} {n_log_n:<12.2f} {n_squared:<12} {two_n:<15}")

compare_growth_rates()

# Master theorem examples
def master_theorem_category(a, b, f_n_complexity):
    """Determine Master theorem category."""
    log_b_a = math.log(a) / math.log(b)
    if f_n_complexity < log_b_a:
        return f"Case 1: O(n^{log_b_a:.2f})"
    elif abs(f_n_complexity - log_b_a) < 0.01:
        return f"Case 2: O(n^{log_b_a:.2f} × log n)"
    else:
        return f"Case 3: O(n^{f_n_complexity:.2f})"

print("\nMaster theorem examples:")
print(f"T(n) = 2T(n/2) + O(n): {master_theorem_category(2, 2, 1)}")
print(f"T(n) = 8T(n/2) + O(n²): {master_theorem_category(8, 2, 2)}")
print(f"T(n) = T(n/2) + O(1): {master_theorem_category(1, 2, 0)}")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          506,
			Title:       "Sequences and Series",
			Description: "Learn arithmetic and geometric sequences, summation formulas, and their applications.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "Arithmetic Sequences",
					Content: `**Introduction: Patterns of Constant Change**

An arithmetic sequence is a sequence where each term differs from the previous term by a constant amount called the common difference. These sequences model linear growth, appear frequently in algorithms, and are fundamental to understanding summation formulas and algorithm complexity.

**Understanding Arithmetic Sequences: Linear Patterns**

An arithmetic sequence follows a simple pattern: add (or subtract) the same value each time.

**Definition:**
A sequence a₁, a₂, a₃, ..., aₙ is arithmetic if:
- a₂ - a₁ = a₃ - a₂ = a₄ - a₃ = ... = d (constant)
- The difference between consecutive terms is always the same

**Key Components:**
- **a₁:** First term
- **d:** Common difference (constant amount added each time)
- **n:** Term number (position in sequence)
- **aₙ:** nth term

**General Form:** aₙ = a₁ + (n - 1)d

**Why This Formula Works:**

**Step-by-Step Derivation:**

1. **First term:** a₁
2. **Second term:** a₂ = a₁ + d (add d once)
3. **Third term:** a₃ = a₂ + d = (a₁ + d) + d = a₁ + 2d (add d twice)
4. **Fourth term:** a₄ = a₃ + d = (a₁ + 2d) + d = a₁ + 3d (add d three times)
5. **Pattern:** To get the nth term, we add d a total of (n-1) times
6. **Formula:** aₙ = a₁ + (n - 1)d

**Detailed Example: 2, 5, 8, 11, 14, ...**

**Identify Components:**
- First term: a₁ = 2
- Common difference: d = 5 - 2 = 3 (or 8 - 5 = 3, etc.)
- Verify: Each term is 3 more than the previous

**Find Specific Terms:**

**5th term (a₅):**
- a₅ = a₁ + (5 - 1)d = 2 + 4 × 3 = 2 + 12 = 14 ✓
- Verification: Sequence is 2, 5, 8, 11, 14 → a₅ = 14 ✓

**10th term (a₁₀):**
- a₁₀ = a₁ + (10 - 1)d = 2 + 9 × 3 = 2 + 27 = 29

**General term:**
- aₙ = 2 + (n - 1) × 3 = 2 + 3n - 3 = 3n - 1
- Check: n=1 → 3(1) - 1 = 2 ✓, n=2 → 3(2) - 1 = 5 ✓

**Finding Components from the Sequence:**

**Finding Common Difference:**
- d = aₙ - aₙ₋₁ (difference between any consecutive terms)
- Example: From 2, 5, 8, 11, 14
  - d = 5 - 2 = 3
  - d = 8 - 5 = 3
  - d = 11 - 8 = 3 ✓

**Finding First Term:**
- If you know a term and d, solve: aₙ = a₁ + (n - 1)d for a₁
- a₁ = aₙ - (n - 1)d
- Example: If a₅ = 14 and d = 3
  - a₁ = 14 - (5 - 1) × 3 = 14 - 12 = 2 ✓

**Finding Number of Terms:**
- Given a₁, aₙ, and d, solve: aₙ = a₁ + (n - 1)d for n
- aₙ - a₁ = (n - 1)d
- n - 1 = (aₙ - a₁) / d
- n = [(aₙ - a₁) / d] + 1
- Example: Sequence from 2 to 14 with d = 3
  - n = [(14 - 2) / 3] + 1 = [12 / 3] + 1 = 4 + 1 = 5 ✓

**Sum of Arithmetic Sequence: The Famous Formula**

The sum of the first n terms of an arithmetic sequence has a beautiful formula discovered by Gauss as a child.

**Formula:** Sₙ = n/2 × (a₁ + aₙ) = n/2 × [2a₁ + (n-1)d]

**Why This Formula Works: The Gauss Trick**

**Legend:** Young Gauss was asked to sum 1 + 2 + 3 + ... + 100
- He paired: (1 + 100) + (2 + 99) + ... + (50 + 51)
- Each pair sums to 101
- 50 pairs × 101 = 5050
- Generalizes to: n/2 × (first + last)

**Step-by-Step Derivation:**

**Method 1: Pairing Terms**
- Write sum forward: S = a₁ + a₂ + ... + aₙ₋₁ + aₙ
- Write sum backward: S = aₙ + aₙ₋₁ + ... + a₂ + a₁
- Add: 2S = (a₁ + aₙ) + (a₂ + aₙ₋₁) + ... + (aₙ + a₁)
- Each pair sums to (a₁ + aₙ) because a₂ + aₙ₋₁ = (a₁ + d) + (aₙ - d) = a₁ + aₙ
- There are n pairs, so: 2S = n(a₁ + aₙ)
- Therefore: S = n/2 × (a₁ + aₙ)

**Method 2: Using General Term**
- Sₙ = Σ [a₁ + (i - 1)d] from i=1 to n
- = Σ a₁ + d Σ (i - 1)
- = na₁ + d[0 + 1 + 2 + ... + (n-1)]
- = na₁ + d × (n-1)n/2 (sum of first n-1 integers)
- = n/2 × [2a₁ + (n-1)d] ✓

**Step-by-Step Example: Sum of 2, 5, 8, 11, 14**

**Given:** a₁ = 2, a₅ = 14, n = 5

**Method 1: Using first and last**
- S₅ = 5/2 × (2 + 14) = 5/2 × 16 = 5 × 8 = 40

**Method 2: Using formula with d**
- d = 3
- S₅ = 5/2 × [2(2) + (5-1)(3)] = 5/2 × [4 + 12] = 5/2 × 16 = 40 ✓

**Verification:** 2 + 5 + 8 + 11 + 14 = 40 ✓

**Applications in Computer Science:**

**1. Array Indexing:**
- Arrays use arithmetic sequences for indices
- Index sequence: 0, 1, 2, 3, ..., n-1
- a₁ = 0, d = 1, aₙ = n - 1

**Example:** Accessing array elements
- Indices form arithmetic sequence
- Loop: for i in range(n) → i = 0, 1, 2, ..., n-1

**2. Loop Iterations:**
- Many loops iterate with constant step
- for i in range(0, 100, 5) → 0, 5, 10, 15, ..., 95
- a₁ = 0, d = 5

**Example:** Processing every 5th element
- Sequence: 0, 5, 10, 15, ..., 95
- Number of iterations: n = [(95 - 0) / 5] + 1 = 20

**3. Time Complexity O(n):**
- Linear time algorithms often process elements in arithmetic sequence
- Each operation takes constant time
- Total time = constant × n (linear)

**Example:** Linear search
- Check indices 0, 1, 2, ..., n-1
- Arithmetic sequence of operations
- Time complexity: O(n)

**4. Memory Allocation:**
- Sequential memory addresses form arithmetic sequence
- Address = base + (index × element_size)
- a₁ = base_address, d = element_size

**Example:** Array in memory
- Base address: 1000
- Element size: 4 bytes
- Addresses: 1000, 1004, 1008, 1012, ...
- Arithmetic sequence with d = 4

**5. Algorithm Analysis:**
- Counting operations in loops
- Sum of iterations: Sₙ = n/2 × (first + last)
- Helps analyze nested loops

**Example:** Nested loop analysis
- Outer loop: n iterations
- Inner loop: arithmetic sequence of iterations
- Total operations = sum of arithmetic sequence

**6. Scheduling and Timing:**
- Periodic events form arithmetic sequences
- Event times: t, t+d, t+2d, t+3d, ...
- Used in scheduling algorithms

**Example:** Periodic task scheduling
- Task runs every d milliseconds
- Execution times: 0, d, 2d, 3d, ...
- Arithmetic sequence of time points

**Common Mistakes:**

**Mistake 1: Using wrong formula for nth term**
- Wrong: aₙ = a₁ + nd
- Correct: aₙ = a₁ + (n - 1)d
- Remember: (n - 1) multiplications, not n!

**Mistake 2: Confusing term number with term value**
- Term number (n) is position: 1st, 2nd, 3rd, ...
- Term value (aₙ) is the actual number
- Don't mix them up!

**Mistake 3: Incorrect sum formula**
- Wrong: Sₙ = n × (a₁ + aₙ) (missing division by 2)
- Correct: Sₙ = n/2 × (a₁ + aₙ)
- The division by 2 comes from pairing!

**Mistake 4: Not checking if sequence is arithmetic**
- Not all sequences are arithmetic!
- Check: Is difference between consecutive terms constant?
- Example: 2, 4, 8, 16 is NOT arithmetic (it's geometric!)

**Why This Matters:**

Arithmetic sequences model:
- **Linear Growth:** Constant rate of change
- **Algorithm Complexity:** O(n) operations
- **Data Structures:** Array indexing, memory layout
- **Real-World Patterns:** Many phenomena follow linear patterns

**Next Steps:**

Mastering arithmetic sequences provides the foundation for geometric sequences (exponential growth) and more complex patterns. In the next lesson, we'll explore geometric sequences, where each term is multiplied by a constant ratio rather than added to.`,
					CodeExamples: `# Arithmetic sequences

def arithmetic_term(first, difference, n):
    """Find nth term of arithmetic sequence."""
    return first + (n - 1) * difference

def arithmetic_sum(first, difference, n):
    """Calculate sum of first n terms."""
    last_term = arithmetic_term(first, difference, n)
    return n * (first + last_term) / 2

def arithmetic_sum_formula2(first, difference, n):
    """Alternative formula: n/2 × [2a₁ + (n-1)d]."""
    return n * (2 * first + (n - 1) * difference) / 2

# Example: 2, 5, 8, 11, 14, ...
first_term = 2
common_diff = 3
print("Arithmetic sequence: 2, 5, 8, 11, 14, ...")
print(f"First term: {first_term}, Common difference: {common_diff}")

print("\nFirst 10 terms:")
for n in range(1, 11):
    term = arithmetic_term(first_term, common_diff, n)
    print(f"  a_{n} = {term}")

# Sum of first 5 terms
n_terms = 5
sum_5 = arithmetic_sum(first_term, common_diff, n_terms)
print(f"\nSum of first {n_terms} terms: {sum_5}")
print(f"Verification: 2 + 5 + 8 + 11 + 14 = {2+5+8+11+14}")

# Find term number
def find_term_number(first, difference, target):
    """Find which term equals target value."""
    if (target - first) % difference != 0:
        return None  # Not in sequence
    return ((target - first) // difference) + 1

target = 14
term_num = find_term_number(first_term, common_diff, target)
print(f"\nTerm number for {target}: {term_num}")`,
				},
				{
					Title: "Geometric Sequences",
					Content: `**Introduction: Patterns of Exponential Change**

A geometric sequence is a sequence where each term is obtained by multiplying the previous term by a constant called the common ratio. These sequences model exponential growth and decay, appear in recursive algorithms, and are fundamental to understanding compound interest, population growth, and algorithm complexity.

**Understanding Geometric Sequences: Multiplicative Patterns**

A geometric sequence follows a pattern: multiply (or divide) by the same value each time.

**Definition:**
A sequence a₁, a₂, a₃, ..., aₙ is geometric if:
- a₂ / a₁ = a₃ / a₂ = a₄ / a₃ = ... = r (constant)
- The ratio between consecutive terms is always the same

**Key Components:**
- **a₁:** First term
- **r:** Common ratio (constant multiplier)
- **n:** Term number (position in sequence)
- **aₙ:** nth term

**General Form:** aₙ = a₁ × r^(n-1)

**Why This Formula Works:**

**Step-by-Step Derivation:**

1. **First term:** a₁
2. **Second term:** a₂ = a₁ × r (multiply by r once)
3. **Third term:** a₃ = a₂ × r = (a₁ × r) × r = a₁ × r² (multiply by r twice)
4. **Fourth term:** a₄ = a₃ × r = (a₁ × r²) × r = a₁ × r³ (multiply by r three times)
5. **Pattern:** To get the nth term, we multiply by r a total of (n-1) times
6. **Formula:** aₙ = a₁ × r^(n-1)

**Detailed Example: 2, 6, 18, 54, 162, ...**

**Identify Components:**
- First term: a₁ = 2
- Common ratio: r = 6 / 2 = 3 (or 18 / 6 = 3, etc.)
- Verify: Each term is 3 times the previous

**Find Specific Terms:**

**5th term (a₅):**
- a₅ = a₁ × r^(5-1) = 2 × 3⁴ = 2 × 81 = 162 ✓
- Verification: Sequence is 2, 6, 18, 54, 162 → a₅ = 162 ✓

**10th term (a₁₀):**
- a₁₀ = a₁ × r^(10-1) = 2 × 3⁹ = 2 × 19,683 = 39,366

**General term:**
- aₙ = 2 × 3^(n-1)
- Check: n=1 → 2 × 3⁰ = 2 ✓, n=2 → 2 × 3¹ = 6 ✓

**Finding Components from the Sequence:**

**Finding Common Ratio:**
- r = aₙ / aₙ₋₁ (ratio of any consecutive terms)
- Example: From 2, 6, 18, 54, 162
  - r = 6 / 2 = 3
  - r = 18 / 6 = 3
  - r = 54 / 18 = 3 ✓

**Finding First Term:**
- If you know a term and r, solve: aₙ = a₁ × r^(n-1) for a₁
- a₁ = aₙ / r^(n-1)
- Example: If a₅ = 162 and r = 3
  - a₁ = 162 / 3^(5-1) = 162 / 81 = 2 ✓

**Sum of Finite Geometric Sequence: The Powerful Formula**

The sum of the first n terms of a geometric sequence has a elegant closed-form formula.

**Formula:** Sₙ = a₁ × (1 - rⁿ) / (1 - r) for r ≠ 1

**Why This Formula Works:**

**Derivation:**

1. **Write the sum:** Sₙ = a₁ + a₁r + a₁r² + ... + a₁r^(n-1)
2. **Multiply by r:** rSₙ = a₁r + a₁r² + a₁r³ + ... + a₁rⁿ
3. **Subtract:** Sₙ - rSₙ = a₁ - a₁rⁿ
4. **Factor:** Sₙ(1 - r) = a₁(1 - rⁿ)
5. **Solve:** Sₙ = a₁(1 - rⁿ) / (1 - r) ✓

**Special Case: r = 1**
- If r = 1, all terms are equal: a₁, a₁, a₁, ...
- Sum: Sₙ = n × a₁ (just add n copies)

**Step-by-Step Example: Sum of 2, 6, 18, 54**

**Given:** a₁ = 2, r = 3, n = 4

**Method: Using formula**
- S₄ = 2 × (1 - 3⁴) / (1 - 3)
- S₄ = 2 × (1 - 81) / (-2)
- S₄ = 2 × (-80) / (-2)
- S₄ = 2 × 40 = 80

**Verification:** 2 + 6 + 18 + 54 = 80 ✓

**Sum of Infinite Geometric Series: When It Converges**

If the common ratio has absolute value less than 1, the infinite sum converges to a finite value.

**Formula:** If |r| < 1: S_∞ = a₁ / (1 - r)

**Why This Works:**
- As n → ∞, rⁿ → 0 (since |r| < 1)
- Sₙ = a₁(1 - rⁿ) / (1 - r) → a₁(1 - 0) / (1 - r) = a₁ / (1 - r)

**Step-by-Step Example: 1/2 + 1/4 + 1/8 + ...**

**Given:** a₁ = 1/2, r = 1/2

**Check convergence:** |r| = |1/2| = 0.5 < 1 ✓ (converges)

**Calculate sum:**
- S_∞ = (1/2) / (1 - 1/2)
- S_∞ = (1/2) / (1/2)
- S_∞ = 1

**Verification:** 1/2 + 1/4 + 1/8 + 1/16 + ... = 1
- This is the famous "Zeno's paradox" resolution
- The infinite sum equals 1!

**Applications in Computer Science:**

**1. Exponential Growth/Decay:**

**Population Growth:**
- If population doubles every year: P(t) = P₀ × 2^t
- Geometric sequence with r = 2
- Models exponential growth

**Radioactive Decay:**
- Half-life: amount halves every period
- Geometric sequence with r = 1/2
- Models exponential decay

**2. Binary Tree Node Counts:**

**Perfect Binary Tree:**
- Level 0: 1 node (2⁰)
- Level 1: 2 nodes (2¹)
- Level 2: 4 nodes (2²)
- Level k: 2ᵏ nodes
- Geometric sequence: aₙ = 2^(n-1)

**Total Nodes in Perfect Tree:**
- Sum of geometric sequence: Sₙ = (2ⁿ - 1)
- Height h tree has 2^(h+1) - 1 nodes

**3. Recursive Algorithm Analysis:**

**Divide-and-Conquer:**
- Problem size reduces by factor r each level
- Number of subproblems: geometric sequence
- Total work: sum of geometric sequence

**Example: Binary Search Tree Operations:**
- Each level has 2^level nodes
- Geometric sequence of nodes per level
- Helps analyze tree operations

**4. Compound Interest:**

**Formula:** A = P(1 + r)^t
- P = principal
- r = interest rate per period
- t = number of periods
- This is a geometric sequence!

**Example:** $1000 at 5% annual interest
- Year 1: $1000 × 1.05 = $1050
- Year 2: $1050 × 1.05 = $1102.50
- Year t: $1000 × 1.05^t
- Geometric sequence with r = 1.05

**5. Algorithm Complexity:**

**Exponential Algorithms:**
- Some algorithms have exponential time complexity
- T(n) = a × r^n (geometric sequence in n)
- Example: Brute force for some problems

**Recursive Calls:**
- Number of recursive calls can form geometric sequence
- Helps analyze recursion depth and complexity

**6. Data Structures:**

**Hash Table Growth:**
- When table fills, often double size
- Sizes: 8, 16, 32, 64, ...
- Geometric sequence with r = 2

**Memory Allocation:**
- Some allocators use geometric progression
- Block sizes: 1, 2, 4, 8, 16, ...
- Efficient for various allocation sizes

**Common Mistakes:**

**Mistake 1: Using wrong exponent**
- Wrong: aₙ = a₁ × rⁿ
- Correct: aₙ = a₁ × r^(n-1)
- Remember: (n-1) multiplications, not n!

**Mistake 2: Confusing geometric with arithmetic**
- Arithmetic: add constant (linear growth)
- Geometric: multiply constant (exponential growth)
- Very different behaviors!

**Mistake 3: Applying infinite sum formula incorrectly**
- Wrong: Using S_∞ when |r| ≥ 1
- Correct: Only works when |r| < 1
- If |r| ≥ 1, series diverges (no finite sum)

**Mistake 4: Sign errors in sum formula**
- Formula: Sₙ = a₁(1 - rⁿ) / (1 - r)
- Be careful with signs when r is negative!
- Example: If r = -2, (1 - r) = 3, not -1

**Why This Matters:**

Geometric sequences model:
- **Exponential Growth:** Populations, investments, data
- **Algorithm Analysis:** Recursive algorithms, tree structures
- **Data Structures:** Binary trees, hash tables
- **Real-World Patterns:** Many natural phenomena follow geometric patterns

**Next Steps:**

Mastering geometric sequences provides the foundation for understanding exponential growth, analyzing recursive algorithms, and working with infinite series. In the next lesson, we'll explore summation formulas and techniques for calculating sums efficiently.`,
					CodeExamples: `# Geometric sequences

import math

def geometric_term(first, ratio, n):
    """Find nth term of geometric sequence."""
    return first * (ratio ** (n - 1))

def geometric_sum_finite(first, ratio, n):
    """Calculate sum of first n terms."""
    if ratio == 1:
        return n * first
    return first * (1 - ratio**n) / (1 - ratio)

def geometric_sum_infinite(first, ratio):
    """Calculate sum of infinite series (if |r| < 1)."""
    if abs(ratio) >= 1:
        return None  # Diverges
    return first / (1 - ratio)

# Example: 2, 6, 18, 54, 162, ...
first_term = 2
common_ratio = 3
print("Geometric sequence: 2, 6, 18, 54, 162, ...")
print(f"First term: {first_term}, Common ratio: {common_ratio}")

print("\nFirst 10 terms:")
for n in range(1, 11):
    term = geometric_term(first_term, common_ratio, n)
    print(f"  a_{n} = {term}")

# Sum of first 4 terms
n_terms = 4
sum_4 = geometric_sum_finite(first_term, common_ratio, n_terms)
print(f"\nSum of first {n_terms} terms: {sum_4}")
print(f"Verification: 2 + 6 + 18 + 54 = {2+6+18+54}")

# Infinite series: 1/2 + 1/4 + 1/8 + ...
inf_first = 0.5
inf_ratio = 0.5
inf_sum = geometric_sum_infinite(inf_first, inf_ratio)
print(f"\nInfinite series: 1/2 + 1/4 + 1/8 + ...")
print(f"Sum: {inf_sum}")

# Binary tree: nodes at each level form geometric sequence
# Level 0: 1, Level 1: 2, Level 2: 4, Level 3: 8, ...
print(f"\nBinary tree levels (geometric sequence):")
for level in range(5):
    nodes = geometric_term(1, 2, level + 1)
    print(f"  Level {level}: {nodes} nodes")`,
				},
				{
					Title: "Summation Formulas",
					Content: `**Introduction: Efficiently Summing Sequences**

Summation formulas provide closed-form expressions for sums that would otherwise require adding many terms. These formulas are essential for algorithm analysis, where we need to count operations efficiently, and for mathematical proofs and numerical methods.

**Understanding Summation Notation: The Sigma Symbol**

The Greek letter Σ (sigma) represents summation - adding up a sequence of terms.

**Notation:** Σᵢ₌₁ⁿ aᵢ = a₁ + a₂ + a₃ + ... + aₙ

**Breaking Down the Notation:**
- **Σ:** Summation symbol (sigma)
- **i = 1:** Starting index (lower bound)
- **n:** Ending index (upper bound)
- **aᵢ:** General term (formula for the ith term)
- **Read as:** "Sum of aᵢ from i equals 1 to n"

**Examples:**
- Σᵢ₌₁⁵ i = 1 + 2 + 3 + 4 + 5 = 15
- Σᵢ₌₁³ i² = 1² + 2² + 3² = 1 + 4 + 9 = 14
- Σᵢ₌₁⁴ 2i = 2(1) + 2(2) + 2(3) + 2(4) = 2 + 4 + 6 + 8 = 20

**Properties of Summation:**

**1. Constant Multiple:** Σᵢ₌₁ⁿ (c × aᵢ) = c × Σᵢ₌₁ⁿ aᵢ
- Can factor out constants

**2. Sum of Sums:** Σᵢ₌₁ⁿ (aᵢ + bᵢ) = Σᵢ₌₁ⁿ aᵢ + Σᵢ₌₁ⁿ bᵢ
- Summation distributes over addition

**3. Splitting:** Σᵢ₌₁ⁿ aᵢ = Σᵢ₌₁ᵐ aᵢ + Σᵢ₌ₘ₊₁ⁿ aᵢ (for 1 ≤ m < n)
- Can split summation at any point

**Common Summation Formulas: Essential Tools**

**1. Sum of First n Natural Numbers: The Gauss Formula**

**Formula:** Σᵢ₌₁ⁿ i = 1 + 2 + ... + n = n(n+1)/2

**Famous Story:** Young Gauss was asked to sum 1 + 2 + ... + 100. He quickly realized:
- Pair terms: (1 + 100) + (2 + 99) + ... + (50 + 51)
- Each pair sums to 101
- 50 pairs × 101 = 5050
- General formula: n/2 × (first + last) = n/2 × (1 + n) = n(n+1)/2

**Derivation:**

**Method 1: Pairing (Gauss's method)**
- Write sum forward: S = 1 + 2 + ... + (n-1) + n
- Write sum backward: S = n + (n-1) + ... + 2 + 1
- Add: 2S = (1+n) + (2+(n-1)) + ... + ((n-1)+2) + (n+1)
- Each pair sums to (n+1), and there are n pairs
- 2S = n(n+1)
- Therefore: S = n(n+1)/2 ✓

**Method 2: Mathematical induction**
- Base case: n=1 → 1 = 1(2)/2 = 1 ✓
- Assume true for n=k: 1 + ... + k = k(k+1)/2
- Prove for n=k+1: 1 + ... + k + (k+1) = k(k+1)/2 + (k+1) = (k+1)(k+2)/2 ✓

**Step-by-Step Example: Sum 1 to 100**

- Σᵢ₌₁¹⁰⁰ i = 100 × 101 / 2 = 10,100 / 2 = 5,050
- Much faster than adding 100 numbers!

**2. Sum of Squares: A More Complex Pattern**

**Formula:** Σᵢ₌₁ⁿ i² = 1² + 2² + ... + n² = n(n+1)(2n+1)/6

**Derivation (using telescoping sum):**
- Key identity: (i+1)³ - i³ = 3i² + 3i + 1
- Sum both sides from i=1 to n
- Left: Telescoping sum → (n+1)³ - 1³ = (n+1)³ - 1
- Right: 3Σi² + 3Σi + Σ1 = 3Σi² + 3n(n+1)/2 + n
- Solve for Σi²: Σi² = [2(n+1)³ - 3n(n+1) - 2n] / 6
- Simplify: Σi² = n(n+1)(2n+1)/6 ✓

**Step-by-Step Example: Sum 1² + 2² + 3²**

- Σᵢ₌₁³ i² = 3 × 4 × 7 / 6 = 84 / 6 = 14
- Verification: 1 + 4 + 9 = 14 ✓

**3. Sum of Cubes: A Beautiful Pattern**

**Formula:** Σᵢ₌₁ⁿ i³ = 1³ + 2³ + ... + n³ = [n(n+1)/2]²

**Beautiful Observation:** Sum of cubes equals square of sum of first n numbers!

**Derivation:**
- Can be proven by induction or using binomial theorem
- The pattern is elegant: (1 + 2 + ... + n)² = 1³ + 2³ + ... + n³

**Step-by-Step Example: Sum 1³ + 2³ + 3³**

- Method 1: Direct formula
  - Σᵢ₌₁³ i³ = [3 × 4 / 2]² = 6² = 36

- Method 2: Using sum of first n numbers
  - Sum of first 3 = 1 + 2 + 3 = 6
  - Square: 6² = 36 ✓

- Verification: 1 + 8 + 27 = 36 ✓

**4. Telescoping Series: Terms That Cancel**

A telescoping series is one where most terms cancel out, leaving only a few terms.

**General Form:** Σᵢ₌₁ⁿ [f(i) - f(i+1)] = f(1) - f(n+1)

**Why It Works:**
- Write out terms: [f(1) - f(2)] + [f(2) - f(3)] + ... + [f(n) - f(n+1)]
- Most terms cancel: f(1) - f(n+1)

**Example: Σᵢ₌₁ⁿ [1/i - 1/(i+1)]**

- Terms: (1/1 - 1/2) + (1/2 - 1/3) + ... + (1/n - 1/(n+1))
- Cancel: 1 - 1/(n+1)
- Result: n/(n+1)

**Applications in Computer Science:**

**1. Algorithm Analysis: Counting Operations**

**Nested Loops:**
- Outer loop: n iterations
- Inner loop: i iterations (i from 1 to n)
- Total operations: Σᵢ₌₁ⁿ i = n(n+1)/2 = O(n²)

**Example:** Bubble sort comparisons
- Pass 1: n-1 comparisons
- Pass 2: n-2 comparisons
- ...
- Total: Σᵢ₌₁ⁿ⁻¹ i = (n-1)n/2 = O(n²)

**2. Time Complexity Calculations:**

**Quadratic Algorithms:**
- Many algorithms have O(n²) complexity
- Sum of squares formula helps analyze them
- Example: Matrix multiplication, nested loops

**Example:** Comparing all pairs in array
- For each of n elements, compare with remaining
- Comparisons: Σᵢ₌₁ⁿ⁻¹ i = n(n-1)/2 ≈ n²/2 = O(n²)

**3. Mathematical Proofs:**

**Induction Proofs:**
- Summation formulas often proven by induction
- Base case + inductive step
- Essential technique in algorithm correctness

**4. Numerical Methods:**

**Numerical Integration:**
- Approximating integrals using sums
- Riemann sums use summation formulas
- Error analysis uses summation properties

**Example:** Trapezoidal rule
- Approximates integral as sum of trapezoid areas
- Uses sum of function values
- Error analysis uses summation formulas

**5. Data Structure Analysis:**

**Tree Operations:**
- Sum of nodes at each level
- Height calculations
- Path length analysis

**Example:** Perfect binary tree
- Level i has 2^i nodes
- Total nodes: Σᵢ₌₀ʰ 2^i = 2^(h+1) - 1
- Uses geometric sum formula

**Common Mistakes:**

**Mistake 1: Incorrect bounds**
- Wrong: Σᵢ₌₁ⁿ i = n(n)/2 (missing +1)
- Correct: Σᵢ₌₁ⁿ i = n(n+1)/2
- Check with small n: n=2 → 1+2=3, formula gives 2×3/2=3 ✓

**Mistake 2: Confusing sum formulas**
- Sum of i: n(n+1)/2
- Sum of i²: n(n+1)(2n+1)/6
- Sum of i³: [n(n+1)/2]²
- Don't mix them up!

**Mistake 3: Off-by-one errors**
- Σᵢ₌₁ⁿ means sum from 1 to n (n terms)
- Σᵢ₌₀ⁿ means sum from 0 to n (n+1 terms)
- Be careful with starting index!

**Mistake 4: Not recognizing telescoping patterns**
- Look for terms that cancel
- Common pattern: 1/i - 1/(i+1)
- Simplifies dramatically!

**Why This Matters:**

Summation formulas are essential for:
- **Algorithm Analysis:** Counting operations efficiently
- **Complexity Theory:** Determining time/space complexity
- **Mathematical Proofs:** Proving algorithm correctness
- **Numerical Methods:** Approximating integrals and sums
- **Optimization:** Understanding algorithm performance

**Next Steps:**

Mastering summation formulas provides the foundation for analyzing algorithm complexity and understanding how operations scale with input size. In the next lesson, we'll explore infinite series and convergence, which extend these concepts to infinite sums.`,
					CodeExamples: `# Summation formulas

def sum_natural_numbers(n):
    """Sum of first n natural numbers: n(n+1)/2."""
    return n * (n + 1) // 2

def sum_squares(n):
    """Sum of first n squares: n(n+1)(2n+1)/6."""
    return n * (n + 1) * (2 * n + 1) // 6

def sum_cubes(n):
    """Sum of first n cubes: [n(n+1)/2]²."""
    return (n * (n + 1) // 2) ** 2

def sum_powers(n, power):
    """Sum of first n numbers raised to power."""
    if power == 1:
        return sum_natural_numbers(n)
    elif power == 2:
        return sum_squares(n)
    elif power == 3:
        return sum_cubes(n)
    else:
        # General case: sum manually
        return sum(i**power for i in range(1, n + 1))

# Examples
n = 10
print(f"Sum formulas for n = {n}:")
print(f"Sum of natural numbers: {sum_natural_numbers(n)}")
print(f"  Verification: {sum(range(1, n+1))}")

print(f"\nSum of squares: {sum_squares(n)}")
print(f"  Verification: {sum(i**2 for i in range(1, n+1))}")

print(f"\nSum of cubes: {sum_cubes(n)}")
print(f"  Verification: {sum(i**3 for i in range(1, n+1))}")

# Famous example: Gauss's sum
n_gauss = 100
gauss_sum = sum_natural_numbers(n_gauss)
print(f"\nGauss's sum (1 to {n_gauss}): {gauss_sum}")
print(f"Formula: {n_gauss}×{n_gauss+1}/2 = {gauss_sum}")

# Telescoping series
def telescoping_sum(n):
    """Sum of telescoping series: Σ[1/i - 1/(i+1)]."""
    return 1 - 1/(n + 1)

n_tel = 10
tel_sum = telescoping_sum(n_tel)
print(f"\nTelescoping series (n={n_tel}): {tel_sum:.4f}")
print(f"Limit as n→∞: 1")`,
				},
				{
					Title: "Infinite Series",
					Content: `**Introduction: Summing the Infinite**

An infinite series is the sum of infinitely many terms. Understanding when these sums converge (approach a finite limit) or diverge (grow without bound) is crucial for algorithm analysis, numerical methods, and understanding asymptotic behavior. Many algorithms and mathematical concepts involve infinite processes that we need to analyze.

**Understanding Infinite Series: Adding Forever**

An infinite series is written as: Σᵢ₌₁∞ aᵢ = a₁ + a₂ + a₃ + ...

**Key Question:** Does this infinite sum have a finite value, or does it grow without bound?

**Partial Sums:**
- S₁ = a₁
- S₂ = a₁ + a₂
- S₃ = a₁ + a₂ + a₃
- ...
- Sₙ = a₁ + a₂ + ... + aₙ

**Convergence:** The series converges if the sequence of partial sums {Sₙ} approaches a finite limit L as n → ∞.

**Convergence vs Divergence: The Fundamental Question**

**Convergent Series:**
- The infinite sum approaches a finite limit
- Partial sums get closer and closer to some number
- We can assign a meaningful value to the sum

**Example: 1/2 + 1/4 + 1/8 + ...**
- S₁ = 1/2 = 0.5
- S₂ = 1/2 + 1/4 = 0.75
- S₃ = 1/2 + 1/4 + 1/8 = 0.875
- S₄ = 0.9375
- S₅ = 0.96875
- Approaches 1 as n → ∞
- Series converges to 1

**Divergent Series:**
- The infinite sum does not approach a finite limit
- Either grows without bound or oscillates
- Cannot assign a finite value

**Example: 1 + 2 + 3 + 4 + ...**
- S₁ = 1
- S₂ = 3
- S₃ = 6
- S₄ = 10
- Sₙ = n(n+1)/2 → ∞ as n → ∞
- Series diverges to infinity

**Geometric Series: The Most Important Infinite Series**

A geometric series has the form: a₁ + a₁r + a₁r² + a₁r³ + ...

**Convergence Criterion:**

**Convergent if |r| < 1:**
- Sum: S = a₁ / (1 - r)
- This is the limit of the finite sum formula as n → ∞
- When |r| < 1, rⁿ → 0, so Sₙ → a₁/(1-r)

**Step-by-Step Example: 1/2 + 1/4 + 1/8 + ...**
- a₁ = 1/2, r = 1/2
- Check: |r| = |1/2| = 0.5 < 1 ✓ (converges)
- Sum: S = (1/2) / (1 - 1/2) = (1/2) / (1/2) = 1 ✓

**Divergent if |r| ≥ 1:**
- If r ≥ 1: Terms don't decrease, sum grows
- If r ≤ -1: Terms alternate and don't approach zero
- In both cases, series diverges

**Example: 2 + 4 + 8 + 16 + ...**
- a₁ = 2, r = 2
- Check: |r| = 2 ≥ 1 ✗ (diverges)
- Sum grows without bound

**Harmonic Series: The Slow Divergence**

The harmonic series is: Σᵢ₌₁∞ 1/i = 1 + 1/2 + 1/3 + 1/4 + ...

**Surprising Property:** This series diverges, even though terms approach zero!

**Why It Diverges (Proof Sketch):**

Group terms:
- 1
- 1/2
- 1/3 + 1/4 > 1/4 + 1/4 = 1/2
- 1/5 + 1/6 + 1/7 + 1/8 > 4 × (1/8) = 1/2
- 1/9 + ... + 1/16 > 8 × (1/16) = 1/2
- Each group sums to at least 1/2
- Infinitely many groups → sum diverges

**Partial Sums: Hₙ = 1 + 1/2 + ... + 1/n**

**Approximation:** Hₙ ≈ ln(n) + γ
- Where γ ≈ 0.5772156649 (Euler-Mascheroni constant)
- Very accurate for large n
- Shows harmonic series grows like logarithm (very slowly!)

**Example:**
- H₁₀₀ ≈ ln(100) + 0.577 ≈ 4.605 + 0.577 ≈ 5.18
- H₁₀₀₀ ≈ ln(1000) + 0.577 ≈ 6.908 + 0.577 ≈ 7.49
- Grows very slowly, but unbounded

**Why This Matters:**
- Many algorithms have harmonic series in complexity
- Example: Average case of quicksort involves harmonic numbers
- Understanding divergence helps analyze algorithms

**p-Series: Generalizing the Harmonic Series**

A p-series has the form: Σᵢ₌₁∞ 1/iᵖ = 1 + 1/2ᵖ + 1/3ᵖ + ...

**Convergence Criterion:**

**Convergent if p > 1:**
- Terms decrease fast enough
- Sum approaches finite limit
- Example: p = 2 → Σ1/i² = π²/6 ≈ 1.645

**Divergent if p ≤ 1:**
- Terms don't decrease fast enough
- Sum grows without bound
- p = 1 is the harmonic series (diverges)

**Intuitive Understanding:**
- p > 1: Terms decrease quickly (1/i², 1/i³, etc.)
- p = 1: Terms decrease slowly (1/i)
- p < 1: Terms decrease even slower (1/i^0.5, etc.)

**Examples:**

**p = 2 (Converges):**
- Σ1/i² = 1 + 1/4 + 1/9 + 1/16 + ...
- Sum = π²/6 ≈ 1.644934
- Famous result from Euler

**p = 1/2 (Diverges):**
- Σ1/√i = 1 + 1/√2 + 1/√3 + ...
- Terms decrease slower than 1/i
- Sum diverges

**Applications in Computer Science:**

**1. Algorithm Analysis: Asymptotic Behavior**

**Harmonic Numbers in Algorithms:**
- Average case quicksort: O(n log n) comparisons
- Involves harmonic numbers: Hₙ ≈ ln(n)
- Understanding divergence helps analyze average case

**Example: Randomized Algorithms:**
- Expected number of operations often involves series
- Need to determine if series converges
- Helps find expected complexity

**2. Numerical Methods: Approximations**

**Taylor Series:**
- Functions represented as infinite series
- eˣ = Σ(n=0 to ∞) xⁿ/n! = 1 + x + x²/2! + x³/3! + ...
- sin(x) = Σ(n=0 to ∞) (-1)ⁿx^(2n+1)/(2n+1)!
- Used for numerical approximations

**Example: Computing e:**
- e = 1 + 1 + 1/2! + 1/3! + 1/4! + ...
- Sum converges rapidly
- Can approximate e to high precision

**3. Mathematical Analysis:**

**Riemann Zeta Function:**
- ζ(s) = Σ(n=1 to ∞) 1/nˢ
- For s > 1, this is a p-series (converges)
- Important in number theory and analysis

**4. Computer Graphics: Infinite Precision**

**Fractals:**
- Some fractals defined by infinite series
- Convergence determines if fractal is bounded
- Used in rendering and generation

**5. Probability and Statistics:**

**Expected Values:**
- Some expected values involve infinite series
- Need to check convergence
- Example: Expected number of trials until success

**6. Signal Processing:**

**Fourier Series:**
- Represent periodic signals as infinite series
- Convergence determines signal properties
- Used in audio, image processing

**Common Mistakes:**

**Mistake 1: Assuming terms → 0 means convergence**
- Wrong: "Terms approach zero, so series converges"
- Counterexample: Harmonic series (terms → 0 but diverges)
- Correct: Terms → 0 is necessary but not sufficient

**Mistake 2: Confusing convergence with divergence**
- Convergent: Sum approaches finite limit
- Divergent: Sum doesn't approach finite limit
- Don't confuse "grows slowly" with "converges"

**Mistake 3: Incorrectly applying geometric series formula**
- Formula S = a₁/(1-r) only works when |r| < 1
- If |r| ≥ 1, series diverges (no finite sum)
- Always check convergence first!

**Mistake 4: Not recognizing series type**
- Geometric: terms multiplied by constant ratio
- Harmonic: terms are 1/n
- p-series: terms are 1/nᵖ
- Different convergence criteria!

**Why This Matters:**

Infinite series appear in:
- **Algorithm Analysis:** Understanding asymptotic behavior
- **Numerical Methods:** Approximating functions and values
- **Mathematical Theory:** Proving theorems and properties
- **Computer Graphics:** Rendering and generation
- **Signal Processing:** Analyzing and synthesizing signals

**Next Steps:**

Mastering infinite series provides the foundation for understanding algorithm complexity, numerical methods, and mathematical analysis. In the next lesson, we'll explore applications in algorithm complexity, showing how these mathematical concepts directly apply to analyzing computer algorithms.`,
					CodeExamples: `# Infinite series

import math

def geometric_series_infinite(first, ratio, max_terms=1000):
    """Approximate infinite geometric series."""
    if abs(ratio) >= 1:
        return None  # Diverges
    
    # Sum first max_terms terms
    partial_sum = 0
    for i in range(max_terms):
        partial_sum += first * (ratio ** i)
    
    # Compare with formula
    exact_sum = first / (1 - ratio)
    return partial_sum, exact_sum

# Example: 1/2 + 1/4 + 1/8 + ...
partial, exact = geometric_series_infinite(0.5, 0.5)
print("Geometric series: 1/2 + 1/4 + 1/8 + ...")
print(f"Partial sum (1000 terms): {partial:.10f}")
print(f"Exact sum: {exact}")
print(f"Difference: {abs(partial - exact):.10f}")

# Harmonic series
def harmonic_number(n):
    """Calculate nth harmonic number H_n."""
    return sum(1/i for i in range(1, n + 1))

def harmonic_approximation(n):
    """Approximate H_n using ln(n) + γ."""
    gamma = 0.5772156649  # Euler-Mascheroni constant
    return math.log(n) + gamma

print(f"\nHarmonic series H_n:")
for n in [10, 100, 1000]:
    H_n = harmonic_number(n)
    approx = harmonic_approximation(n)
    print(f"  H_{n} = {H_n:.4f}, Approximation: {approx:.4f}")

# p-series
def p_series_sum(n, p):
    """Calculate partial sum of p-series."""
    return sum(1 / (i**p) for i in range(1, n + 1))

print(f"\np-series (p=2):")
for n in [10, 100, 1000]:
    partial = p_series_sum(n, 2)
    print(f"  First {n} terms: {partial:.6f}")
print(f"  Limit (π²/6): {math.pi**2/6:.6f}")

# Alternating series
def alternating_series(n):
    """Calculate 1 - 1/2 + 1/3 - 1/4 + ..."""
    return sum((-1)**(i+1) / i for i in range(1, n + 1))

print(f"\nAlternating harmonic series:")
for n in [10, 100, 1000]:
    partial = alternating_series(n)
    print(f"  First {n} terms: {partial:.6f}")
print(f"  Limit (ln 2): {math.log(2):.6f}")`,
				},
				{
					Title: "Applications in Algorithm Complexity",
					Content: `**Introduction: Mathematics Meets Algorithms**

Sequences and series are fundamental tools for analyzing algorithm complexity. They help us count operations, understand growth patterns, and predict how algorithms scale with input size. Mastering these mathematical concepts is essential for becoming an effective algorithm designer and analyst.

**Sequences in Algorithm Analysis: Counting Operations**

**1. Arithmetic Sequences: Linear Growth Patterns**

Arithmetic sequences model operations that increase by a constant amount each iteration.

**Loop Iterations:**
- Simple loop: for i in range(n): operations
- Iterations: 1, 2, 3, ..., n
- This is an arithmetic sequence: a₁ = 1, d = 1, aₙ = n
- Total iterations: n (linear)

**Nested Loops:**
- Outer loop: n iterations
- Inner loop: i iterations (where i goes from 1 to n)
- Total operations: Σᵢ₌₁ⁿ i = n(n+1)/2 = O(n²)

**Step-by-Step Example: Bubble Sort Comparisons**

**Algorithm:** Compare adjacent elements, swap if needed
- Pass 1: Compare positions (n-1) times
- Pass 2: Compare positions (n-2) times
- ...
- Pass (n-1): Compare positions 1 time

**Total Comparisons:**
- Σᵢ₌₁ⁿ⁻¹ i = (n-1)n/2 = n²/2 - n/2 = O(n²)
- This is sum of arithmetic sequence
- Confirms bubble sort is O(n²)

**2. Geometric Sequences: Divide-and-Conquer Patterns**

Geometric sequences model operations in divide-and-conquer algorithms where problem size reduces by a constant factor.

**Binary Tree Levels:**
- Level 0: 1 node (root)
- Level 1: 2 nodes
- Level 2: 4 nodes
- Level k: 2ᵏ nodes
- This is geometric: a₁ = 1, r = 2, aₖ = 2ᵏ

**Complete Binary Tree:**
- Total nodes: Σᵢ₌₀ʰ 2ᵢ = 2^(h+1) - 1
- For n nodes: height h ≈ log₂(n)
- Operations per level often constant
- Total: O(n) for complete tree

**Step-by-Step Example: Merge Sort Recursion Tree**

**Algorithm:** Divide array in half, sort recursively, merge
- Level 0: 1 problem of size n
- Level 1: 2 problems of size n/2
- Level 2: 4 problems of size n/4
- Level k: 2ᵏ problems of size n/2ᵏ

**Operations per Level:**
- Merge at each level: O(n) total
- Number of levels: log₂(n)
- Total: O(n log n) ✓

**Recurrence Relations: Describing Algorithm Behavior**

Recurrence relations express algorithm complexity recursively.

**1. Arithmetic Recurrence: Linear Time**

**Form:** T(n) = T(n-1) + c

**Solution:** T(n) = O(n)
- Each step reduces problem by 1
- n steps total
- Linear time complexity

**Examples:**
- **Linear Search:** T(n) = T(n-1) + 1 → O(n)
- **Factorial:** T(n) = T(n-1) + 1 → O(n)
- **Printing Array:** T(n) = T(n-1) + 1 → O(n)

**Derivation:**
- T(n) = T(n-1) + c
- T(n-1) = T(n-2) + c
- ...
- T(2) = T(1) + c
- Sum: T(n) = T(1) + (n-1)c = O(n)

**2. Geometric Recurrence: Logarithmic Time**

**Form:** T(n) = T(n/2) + c

**Solution:** T(n) = O(log n)
- Each step halves problem size
- log₂(n) steps to reach base case
- Logarithmic time complexity

**Examples:**
- **Binary Search:** T(n) = T(n/2) + 1 → O(log n)
- **Finding in Sorted Array:** T(n) = T(n/2) + 1 → O(log n)

**Derivation:**
- T(n) = T(n/2) + c
- T(n/2) = T(n/4) + c
- ...
- T(1) = c (base case)
- Number of steps: log₂(n)
- Total: T(n) = c × log₂(n) = O(log n)

**3. Divide and Conquer: Linearithmic Time**

**Form:** T(n) = 2T(n/2) + O(n)

**Solution:** T(n) = O(n log n)
- Split into 2 subproblems of size n/2
- Combine results in O(n) time
- Classic divide-and-conquer pattern

**Examples:**
- **Merge Sort:** T(n) = 2T(n/2) + O(n) → O(n log n)
- **Quick Sort (average):** T(n) = 2T(n/2) + O(n) → O(n log n)
- **Binary Tree Traversal:** T(n) = 2T(n/2) + O(1) → O(n)

**Derivation (Master Theorem):**
- a = 2 (number of subproblems)
- b = 2 (size reduction factor)
- f(n) = O(n) (combine cost)
- Since f(n) = Θ(n^(log_b a)) = Θ(n), case 2 applies
- T(n) = Θ(n log n) = O(n log n)

**Summation in Complexity: Counting Total Operations**

**1. Nested Loops: Quadratic Complexity**

**Pattern:** Σᵢ₌₁ⁿ Σⱼ₌₁ⁱ 1

**Analysis:**
- Outer loop: i from 1 to n
- Inner loop: j from 1 to i
- Inner operations: i (for each i)
- Total: Σᵢ₌₁ⁿ i = n(n+1)/2 = O(n²)

**Example: Selection Sort**
- Find minimum in unsorted portion
- Operations: Σᵢ₌₁ⁿ (n-i) = n(n-1)/2 = O(n²)

**2. Tree Traversal: Linear Complexity**

**Pattern:** Sum of nodes at each level

**Complete Binary Tree:**
- Level i has 2ᵢ nodes
- Total nodes: Σᵢ₌₀ʰ 2ᵢ = 2^(h+1) - 1
- For n nodes: h ≈ log₂(n)
- Visiting all nodes: O(n)

**Example: Inorder Traversal**
- Visit left subtree, root, right subtree
- Each node visited once
- Total: O(n) operations

**3. Matrix Operations: Cubic Complexity**

**Pattern:** Triple nested loops

**Matrix Multiplication:**
- For each of n rows
- For each of n columns
- Compute dot product (n multiplications)
- Total: Σᵢ₌₁ⁿ Σⱼ₌₁ⁿ n = n³ = O(n³)

**Optimizations:**
- Strassen's algorithm: O(n^2.81)
- Best known: O(n^2.373)
- But still polynomial, not exponential!

**Applications: Real Algorithm Examples**

**1. Sorting Algorithms:**

**Bubble Sort:**
- Comparisons: Σᵢ₌₁ⁿ⁻¹ i = O(n²)
- Arithmetic sequence sum
- Confirms O(n²) complexity

**Merge Sort:**
- Recursion tree: geometric sequence of problem sizes
- Levels: log₂(n)
- Work per level: O(n)
- Total: O(n log n)

**2. Search Algorithms:**

**Linear Search:**
- Worst case: check all n elements
- Operations: n (arithmetic sequence with 1 term)
- Complexity: O(n)

**Binary Search:**
- Each step halves search space
- Steps needed: log₂(n)
- Geometric sequence (halving)
- Complexity: O(log n)

**3. Tree Operations:**

**Binary Search Tree Search:**
- Height: O(log n) for balanced tree
- Operations per level: O(1)
- Total: O(log n)
- Uses geometric sequence (tree levels)

**Tree Traversal:**
- Visit all n nodes once
- Operations: O(n)
- Sum of nodes at all levels

**4. Dynamic Programming:**

**Fibonacci (Naive):**
- T(n) = T(n-1) + T(n-2) + 1
- Makes ~2ⁿ calls (exponential!)
- Geometric sequence with r > 1 (diverges)
- Need memoization: O(n) time

**Fibonacci (Memoized):**
- Each value computed once
- Operations: n (arithmetic sequence)
- Complexity: O(n)

**Why This Matters:**

Understanding sequences helps:
- **Analyze Complexity:** Count operations accurately
- **Optimize Algorithms:** Identify bottlenecks
- **Choose Data Structures:** Match structure to problem
- **Predict Performance:** Estimate runtime for large inputs
- **Design Algorithms:** Create efficient solutions

**Common Patterns:**

**O(1):** Constant - no loops, direct access
**O(log n):** Logarithmic - binary search, balanced trees
**O(n):** Linear - single loop, tree traversal
**O(n log n):** Linearithmic - divide and conquer, efficient sorts
**O(n²):** Quadratic - nested loops, naive matrix multiplication
**O(2ⁿ):** Exponential - brute force, naive recursion

**Next Steps:**

Mastering these mathematical tools provides the foundation for analyzing any algorithm's complexity. The patterns you've learned - arithmetic sequences, geometric sequences, and summation formulas - are the building blocks of algorithm analysis.`,
					CodeExamples: `# Applications in algorithm complexity

# Nested loop complexity
def nested_loop_operations(n):
    """Count operations in nested loops."""
    operations = 0
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            operations += 1
    return operations

def nested_loop_formula(n):
    """Calculate using formula: n(n+1)/2."""
    return n * (n + 1) // 2

n = 10
actual_ops = nested_loop_operations(n)
formula_ops = nested_loop_formula(n)
print(f"Nested loop operations (n={n}):")
print(f"  Actual: {actual_ops}")
print(f"  Formula: {formula_ops}")
print(f"  Complexity: O(n²)")

# Binary tree node count (geometric sequence)
def binary_tree_nodes(levels):
    """Count nodes in complete binary tree."""
    # Levels: 1, 2, 4, 8, ... (geometric)
    return sum(2**i for i in range(levels))

def binary_tree_formula(levels):
    """Calculate using geometric sum formula."""
    return 2**levels - 1

levels = 5
actual_nodes = binary_tree_nodes(levels)
formula_nodes = binary_tree_formula(levels)
print(f"\nBinary tree nodes (levels={levels}):")
print(f"  Actual: {actual_nodes}")
print(f"  Formula: {formula_nodes}")
print(f"  Complexity: O(2^h) where h is height")

# Merge sort recurrence analysis
def merge_sort_complexity(n):
    """Analyze merge sort using geometric sequence."""
    # At each level: n operations
    # Number of levels: log₂(n)
    levels = math.ceil(math.log2(n))
    return n * levels

n_merge = 16
complexity = merge_sort_complexity(n_merge)
print(f"\nMerge sort complexity (n={n_merge}):")
print(f"  Operations: {complexity}")
print(f"  Complexity: O(n log n)")

# Fibonacci sequence (exponential growth)
def fibonacci_naive_complexity(n):
    """Complexity of naive Fibonacci: O(2^n)."""
    return 2**n

def fibonacci_terms(n):
    """Generate first n Fibonacci numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

print(f"\nFibonacci sequence (first 10 terms):")
fib_10 = fibonacci_terms(10)
print(f"  {fib_10}")
print(f"  Growth: exponential (golden ratio^n)")`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
