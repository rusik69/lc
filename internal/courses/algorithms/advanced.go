package algorithms

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAlgorithmsModules([]problems.CourseModule{
		{
			ID:          7,
			Title:       "Dynamic Programming",
			Description: "Master dynamic programming techniques to solve complex optimization problems efficiently.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to Dynamic Programming",
					Content: `Dynamic Programming (DP) is less about programming and more about smart recursion. It's an optimization technique that turns exponential-time algorithms into polynomial-time ones.

**The core idea:** "Those who cannot remember the past are condemned to repeat it."
In code, this means: "If we've solved this subproblem before, don't solve it again. Look it up."

**Two Pillars of DP:**

1.  **Overlapping Subproblems:**
    *   The problem can be broken down into smaller pieces.
    *   The SAME pieces are needed multiple times.
    *   *Example:* In Fibonacci, fib(5) asks for fib(3) and fib(4). fib(4) AGAIN asks for fib(3).

2.  **Optimal Substructure:**
    *   The optimal solution to a big problem can be built from optimal solutions to small problems.
    *   *Example:* Shortest path A to C is shortest A to B + shortest B to C.

**Top-Down vs Bottom-Up:**

*   **Top-Down (Memoization):**
    *   Start at the goal, recurse down.
    *   "I need fib(10). To get that, I need fib(9) and fib(8)..."
    *   Store results in a map/array as you go.
    *   *Pros:* Only solves what is needed. Easier to write from recursive logic.

*   **Bottom-Up (Tabulation):**
    *   Start at the base case, build up.
    *   "I compute fib(0), then fib(1), then fib(2)... up to fib(10)."
    *   Fill a table (array) iteratively.
    *   *Pros:* No recursion overhead. Often easier to optimize space.`,
					CodeExamples: `// Go: Fibonacci with memoization
func fib(n int) int {
    memo := make(map[int]int)
    return fibMemo(n, memo)
}

func fibMemo(n int, memo map[int]int) int {
    if n <= 1 {
        return n
    }
    if val, ok := memo[n]; ok {
        return val
    }
    memo[n] = fibMemo(n-1, memo) + fibMemo(n-2, memo)
    return memo[n]
}

# Python: Fibonacci with memoization
def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]`,
				},
				{
					Title: "DP Patterns",
					Content: `Recognizing the pattern is 90% of solving a DP problem. Here are the most common templates.

**1. 1D DP (Linear)**
*   **Concept:** The answer depends on previous steps in a single line.
*   **Recurrence:** 'dp[i] = dp[i-1] + dp[i-2]' (like Climbing Stairs)
*   **Space:** often O(n), can be optimized to O(1).

**2. 2D DP (Grid/Matrix)**
*   **Concept:** Moving through a grid or comparing two strings.
*   **Recurrence:** 'dp[i][j] = dp[i-1][j] + dp[i][j-1]' (like Unique Paths)
*   **Space:** O(n*m).

**3. Knapsack Pattern (0/1)**
*   **Concept:** You have a capacity and items with weights/values. Take it or leave it.
*   **Question:** Max value within weight limit?
*   **State:** 'dp[i][w]' = max value using first i items with weight limit w.

**4. Unbounded Knapsack**
*   **Concept:** Same as above, but you can use an item infinite times involved.
*   **Example:** Coin Change (fewest coins to make amount X).

**5. Longest Common Subsequence (LCS)**
*   **Concept:** Comparing two strings to find commonalities.
*   **State:** 'dp[i][j]' = LCS length of string A[:i] and string B[:j].
*   **Logic:** If chars match, '1 + dp[i-1][j-1]'. If not, 'max(dp[i-1][j], dp[i][j-1])'.

**6. Interval DP**
*   **Concept:** Solving for a range [i, j].
*   **Logic:** Combine valid sub-intervals [i, k] and [k+1, j].
*   **Example:** Matrix Chain Multiplication, Burst Balloons.

**How to Solve Any DP Problem:**
1.  **Define State:** What does 'dp[i]' mean? (e.g., "max profit at day i")
2.  **Find Recurrence:** How does 'dp[i]' relate to 'dp[i-1]'?
3.  **Base Cases:** What is 'dp[0]'?
4.  **Order of Computation:** Do we loop i from 0 to n?
5.  **Location of Answer:** Is it 'dp[n]'? Or 'max(dp)'?`,
					CodeExamples: `// Go: 1D DP - Climbing stairs
func climbStairs(n int) int {
    if n <= 2 {
        return n
    }
    dp := make([]int, n+1)
    dp[1] = 1
    dp[2] = 2
    for i := 3; i <= n; i++ {
        dp[i] = dp[i-1] + dp[i-2]
    }
    return dp[n]
}

# Python: 2D DP - Unique paths
def uniquePaths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]`,
				},
				{
					Title: "DP Optimization Techniques",
					Content: `Optimizing DP solutions can reduce space complexity and sometimes improve time complexity. Understanding optimization techniques is crucial for efficient DP solutions.

**Space Optimization:**

**1. Rolling Array:**
- For 1D DP where dp[i] depends on dp[i-1] and dp[i-2]
- Only need to keep last 2 values, not entire array
- Reduces space from O(n) to O(1)
- Example: Fibonacci, Climbing stairs

**2. 2D to 1D:**
- For 2D DP where dp[i][j] depends only on previous row
- Use 1D array and update in-place
- Reduces space from O(m×n) to O(n)
- Example: Unique paths, Edit distance

**3. State Compression:**
- For DP with multiple states
- Use bitmask to represent states
- Reduces space significantly
- Example: Traveling salesman problem

**Time Optimization:**

**1. Sliding Window:**
- For DP with window constraints
- Maintain window of previous values
- Example: Maximum sum subarray with constraints

**2. Monotonic Queue/Stack:**
- Optimize transitions in DP
- Maintain monotonic structure for efficient queries
- Example: Maximum in sliding window DP

**3. Convex Hull Trick:**
- Optimize DP with linear transitions
- Advanced technique for competitive programming
- Reduces time complexity significantly

**Common Optimizations:**
- **Fibonacci**: O(n) space → O(1) space
- **Unique Paths**: O(m×n) space → O(n) space
- **Edit Distance**: O(m×n) space → O(min(m,n)) space
- **Knapsack**: Can optimize space, but time stays same

**When to Optimize:**
- Memory constraints
- Large input sizes
- Need to improve space complexity
- Interview optimization questions`,
					CodeExamples: `// Go: Space-optimized Fibonacci (O(1) space)
func fibOptimized(n int) int {
    if n <= 1 {
        return n
    }
    
    prev2, prev1 := 0, 1
    for i := 2; i <= n; i++ {
        curr := prev1 + prev2
        prev2 = prev1
        prev1 = curr
    }
    return prev1
}

// Go: Space-optimized Unique Paths (O(n) instead of O(m×n))
func uniquePathsOptimized(m int, n int) int {
    dp := make([]int, n)
    for i := range dp {
        dp[i] = 1
    }
    
    for i := 1; i < m; i++ {
        for j := 1; j < n; j++ {
            dp[j] += dp[j-1]  // dp[j] = dp[j] (from above) + dp[j-1] (from left)
        }
    }
    
    return dp[n-1]
}

// Go: Space-optimized Edit Distance
func minDistanceOptimized(word1 string, word2 string) int {
    m, n := len(word1), len(word2)
    
    // Use shorter string for dp array
    if m < n {
        word1, word2 = word2, word1
        m, n = n, m
    }
    
    prev := make([]int, n+1)
    curr := make([]int, n+1)
    
    // Initialize first row
    for j := 0; j <= n; j++ {
        prev[j] = j
    }
    
    for i := 1; i <= m; i++ {
        curr[0] = i
        for j := 1; j <= n; j++ {
            if word1[i-1] == word2[j-1] {
                curr[j] = prev[j-1]
            } else {
                curr[j] = 1 + min(prev[j], min(curr[j-1], prev[j-1]))
            }
        }
        prev, curr = curr, prev
    }
    
    return prev[n]
}

# Python: Space-optimized Fibonacci
def fib_optimized(n):
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    
    return prev1

# Python: Space-optimized Unique Paths
def unique_paths_optimized(m, n):
    dp = [1] * n
    
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    
    return dp[n - 1]`,
				},
				{
					Title: "State Machine DP",
					Content: `State Machine DP models problems where we need to track different states and transitions between them. This is powerful for problems with restrictions or multiple conditions.

**Key Concept:**
- Define states that represent different conditions
- Track transitions between states
- DP value represents optimal value in each state

**Common State Machines:**

**1. Buy/Sell Stock:**
- States: Hold stock, Don't hold stock
- Transitions: Buy, Sell, Hold
- Track maximum profit in each state
- Can extend to multiple transactions, cooldown periods

**2. House Robber:**
- States: Rob current house, Don't rob current house
- Transitions: Rob (can't rob previous), Don't rob (can rob previous)
- Track maximum money in each state

**3. String Matching:**
- States: Characters matched so far
- Transitions: Match current character, Don't match
- Track number of ways or best match

**4. Game Theory:**
- States: Current player's turn, game state
- Transitions: Possible moves
- Track winning/losing states

**State Definition:**
- dp[i][state] = optimal value at position i in given state
- Or: dp[state] = optimal value in current state (if position implicit)

**Transition Rules:**
- Define how states transition
- Consider constraints (can't do certain actions in certain states)
- Update DP values based on transitions

**When to Use:**
- Problems with restrictions (can't do X after Y)
- Multiple conditions to track
- Sequential decisions with state dependencies
- Game theory problems`,
					CodeExamples: `// Go: Buy/Sell Stock with State Machine
func maxProfit(prices []int) int {
    n := len(prices)
    if n == 0 {
        return 0
    }
    
    // dp[i][0] = max profit at day i without stock
    // dp[i][1] = max profit at day i with stock
    dp := make([][]int, n)
    for i := range dp {
        dp[i] = make([]int, 2)
    }
    
    // Base cases
    dp[0][0] = 0           // Don't hold on day 0
    dp[0][1] = -prices[0]  // Buy on day 0
    
    for i := 1; i < n; i++ {
        // Don't hold: max of (keep not holding, sell today)
        dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
        
        // Hold: max of (keep holding, buy today)
        dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
    }
    
    return dp[n-1][0]  // Best to end without stock
}

// Go: House Robber with State Machine
func rob(nums []int) int {
    n := len(nums)
    if n == 0 {
        return 0
    }
    
    // dp[i][0] = max money up to i without robbing i
    // dp[i][1] = max money up to i with robbing i
    dp := make([][]int, n)
    for i := range dp {
        dp[i] = make([]int, 2)
    }
    
    dp[0][0] = 0
    dp[0][1] = nums[0]
    
    for i := 1; i < n; i++ {
        // Don't rob: can rob or not rob previous
        dp[i][0] = max(dp[i-1][0], dp[i-1][1])
        
        // Rob: must not rob previous
        dp[i][1] = dp[i-1][0] + nums[i]
    }
    
    return max(dp[n-1][0], dp[n-1][1])
}

# Python: Buy/Sell Stock with State Machine
def max_profit(prices):
    if not prices:
        return 0
    
    n = len(prices)
    # dp[i][0] = max profit without stock, dp[i][1] = with stock
    dp = [[0] * 2 for _ in range(n)]
    
    dp[0][0] = 0
    dp[0][1] = -prices[0]
    
    for i in range(1, n):
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
    
    return dp[n-1][0]

# Python: House Robber with State Machine
def rob(nums):
    if not nums:
        return 0
    
    n = len(nums)
    dp = [[0] * 2 for _ in range(n)]
    
    dp[0][0] = 0
    dp[0][1] = nums[0]
    
    for i in range(1, n):
        dp[i][0] = max(dp[i-1][0], dp[i-1][1])
        dp[i][1] = dp[i-1][0] + nums[i]
    
    return max(dp[n-1][0], dp[n-1][1])`,
				},
				{
					Title: "Interval DP",
					Content: `Interval DP solves problems on intervals (subarrays, substrings) by considering all possible ways to split the interval and combining optimal solutions.

**Key Pattern:**
- dp[i][j] = optimal value for interval from i to j
- Consider all possible splits: dp[i][j] = min/max over k of (dp[i][k] + dp[k+1][j] + cost)
- Build from smaller intervals to larger intervals

**Common Problems:**

**1. Matrix Chain Multiplication:**
- Find optimal way to multiply matrices
- Minimize number of scalar multiplications
- dp[i][j] = min cost to multiply matrices from i to j

**2. Palindrome Partitioning:**
- Partition string into palindromes
- Minimize number of partitions
- dp[i][j] = whether substring i..j is palindrome

**3. Burst Balloons:**
- Pop balloons to maximize coins
- Consider which balloon to pop last in interval
- dp[i][j] = max coins from popping balloons i to j

**4. Stone Game:**
- Two players take stones from ends
- Maximize score difference
- dp[i][j] = max score difference for stones i to j

**Algorithm Pattern:**
1. Define dp[i][j] for interval [i, j]
2. Base case: dp[i][i] for single elements
3. For length L from 2 to n:
   - For each starting position i:
     - j = i + L - 1
     - Try all splits k: dp[i][j] = optimize over dp[i][k] + dp[k+1][j]
4. Return dp[0][n-1]

**Time Complexity:** Usually O(n³) - three nested loops
**Space Complexity:** O(n²) - 2D DP table

**Key Insight:** Think about which element/operation happens last in the interval, then recursively solve subintervals.`,
					CodeExamples: `// Go: Matrix Chain Multiplication
func matrixChainOrder(p []int) int {
    n := len(p) - 1  // Number of matrices
    dp := make([][]int, n)
    for i := range dp {
        dp[i] = make([]int, n)
    }
    
    // L is chain length
    for L := 2; L <= n; L++ {
        for i := 0; i < n-L+1; i++ {
            j := i + L - 1
            dp[i][j] = int(^uint(0) >> 1)  // Max int
            
            // Try all splits
            for k := i; k < j; k++ {
                cost := dp[i][k] + dp[k+1][j] + p[i]*p[k+1]*p[j+1]
                if cost < dp[i][j] {
                    dp[i][j] = cost
                }
            }
        }
    }
    
    return dp[0][n-1]
}

// Go: Palindrome Partitioning
func minCut(s string) int {
    n := len(s)
    
    // Precompute palindromes
    isPal := make([][]bool, n)
    for i := range isPal {
        isPal[i] = make([]bool, n)
    }
    
    // Single characters are palindromes
    for i := 0; i < n; i++ {
        isPal[i][i] = true
    }
    
    // Check palindromes of length 2+
    for L := 2; L <= n; L++ {
        for i := 0; i < n-L+1; i++ {
            j := i + L - 1
            if L == 2 {
                isPal[i][j] = s[i] == s[j]
            } else {
                isPal[i][j] = s[i] == s[j] && isPal[i+1][j-1]
            }
        }
    }
    
    // DP: min cuts needed
    dp := make([]int, n)
    for i := 0; i < n; i++ {
        if isPal[0][i] {
            dp[i] = 0
        } else {
            dp[i] = i
            for j := 0; j < i; j++ {
                if isPal[j+1][i] && dp[j]+1 < dp[i] {
                    dp[i] = dp[j] + 1
                }
            }
        }
    }
    
    return dp[n-1]
}

# Python: Matrix Chain Multiplication
def matrix_chain_order(p):
    n = len(p) - 1
    dp = [[0] * n for _ in range(n)]
    
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            dp[i][j] = float('inf')
            
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + p[i] * p[k+1] * p[j+1]
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[0][n-1]

# Python: Palindrome Partitioning
def min_cut(s):
    n = len(s)
    
    # Precompute palindromes
    is_pal = [[False] * n for _ in range(n)]
    
    for i in range(n):
        is_pal[i][i] = True
    
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            if L == 2:
                is_pal[i][j] = s[i] == s[j]
            else:
                is_pal[i][j] = s[i] == s[j] and is_pal[i+1][j-1]
    
    # DP for min cuts
    dp = [0] * n
    for i in range(n):
        if is_pal[0][i]:
            dp[i] = 0
        else:
            dp[i] = i
            for j in range(i):
                if is_pal[j+1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)
    
    return dp[n-1]`,
				},
			},
			ProblemIDs: []int{51, 52, 53, 54, 55, 56, 57, 58, 59, 60},
		},
		{
			ID:          10,
			Title:       "Backtracking",
			Description: "Master backtracking algorithms to solve constraint satisfaction problems and generate all possible solutions.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to Backtracking",
					Content: `Backtracking is a systematic method for solving problems by trying partial solutions and abandoning them ("backtracking") if they cannot lead to a valid solution.

**Key Concept:**
- Build solution incrementally
- Abandon partial solution if it cannot be completed (pruning)
- Try next possibility when backtracking
- Explore all possibilities systematically

**When to Use:**
- Constraint satisfaction problems
- Generate all possible solutions
- Problems with multiple valid solutions
- Combinatorial problems
- Problems where you need to try all possibilities

**Backtracking Template:**
1. Choose: Make a choice
2. Explore: Recursively solve with that choice
3. Unchoose: Undo the choice (backtrack)
4. Base case: Check if solution is complete

**Common Patterns:**
- **N-Queens**: Place queens on chessboard
- **Sudoku Solver**: Fill sudoku grid
- **Permutations/Combinations**: Generate all arrangements
- **Subset Generation**: Generate all subsets
- **Path Finding**: Find all paths in maze

**Optimization Techniques:**
- **Pruning**: Eliminate invalid branches early
- **Memoization**: Cache results to avoid recomputation
- **Constraint Propagation**: Use constraints to reduce search space

**Time Complexity:** Usually exponential O(2^n) or O(n!) - explores all possibilities
**Space Complexity:** O(n) - recursion depth`,
					CodeExamples: `// Go: Backtracking Template
func backtrack(candidates []int, current []int, result *[][]int) {
    // Base case: solution found
    if isSolution(current) {
        *result = append(*result, append([]int{}, current...))
        return
    }
    
    // Try each candidate
    for _, candidate := range candidates {
        // Choose
        if isValid(current, candidate) {
            current = append(current, candidate)
            
            // Explore
            backtrack(candidates, current, result)
            
            // Unchoose (backtrack)
            current = current[:len(current)-1]
        }
    }
}

// Go: Generate Permutations
func permute(nums []int) [][]int {
    result := [][]int{}
    used := make([]bool, len(nums))
    
    var backtrack func([]int)
    backtrack = func(current []int) {
        if len(current) == len(nums) {
            result = append(result, append([]int{}, current...))
            return
        }
        
        for i := 0; i < len(nums); i++ {
            if !used[i] {
                used[i] = true
                current = append(current, nums[i])
                backtrack(current)
                current = current[:len(current)-1]
                used[i] = false
            }
        }
    }
    
    backtrack([]int{})
    return result
}

# Python: Backtracking Template
def backtrack(candidates, current, result):
    if is_solution(current):
        result.append(current[:])
        return
    
    for candidate in candidates:
        if is_valid(current, candidate):
            current.append(candidate)
            backtrack(candidates, current, result)
            current.pop()  # Backtrack

# Python: Generate Permutations
def permute(nums):
    result = []
    used = [False] * len(nums)
    
    def backtrack(current):
        if len(current) == len(nums):
            result.append(current[:])
            return
        
        for i in range(len(nums)):
            if not used[i]:
                used[i] = True
                current.append(nums[i])
                backtrack(current)
                current.pop()
                used[i] = False
    
    backtrack([])
    return result`,
				},
				{
					Title: "N-Queens Problem",
					Content: `The N-Queens problem is a classic backtracking problem: place N queens on an N×N chessboard such that no two queens attack each other.

**Problem Statement:**
Place N queens on an N×N board so that:
- No two queens are in the same row
- No two queens are in the same column
- No two queens are on the same diagonal

**Approach:**
- Place queens row by row (one per row)
- For each row, try each column
- Check if placement is valid (no conflicts)
- If valid, recurse to next row
- If no valid placement, backtrack

**Optimization:**
- Use arrays to track occupied columns and diagonals
- Check conflicts in O(1) time
- Prune invalid branches early

**Key Insight:**
- One queen per row (place row by row)
- Track columns and diagonals efficiently
- Backtrack when no valid placement found

**Variations:**
- Count number of solutions
- Return all solutions
- Return first solution
- Print board representation`,
					CodeExamples: `// Go: N-Queens - Find all solutions
func solveNQueens(n int) [][]string {
    result := [][]string{}
    board := make([]int, n)  // board[i] = column of queen in row i
    
    var backtrack func(int)
    backtrack = func(row int) {
        if row == n {
            // Convert to string representation
            solution := []string{}
            for i := 0; i < n; i++ {
                rowStr := ""
                for j := 0; j < n; j++ {
                    if board[i] == j {
                        rowStr += "Q"
                    } else {
                        rowStr += "."
                    }
                }
                solution = append(solution, rowStr)
            }
            result = append(result, solution)
            return
        }
        
        for col := 0; col < n; col++ {
            if isValid(board, row, col) {
                board[row] = col
                backtrack(row + 1)
                // No need to reset - overwritten in next iteration
            }
        }
    }
    
    backtrack(0)
    return result
}

func isValid(board []int, row, col int) bool {
    for i := 0; i < row; i++ {
        // Check column
        if board[i] == col {
            return false
        }
        // Check diagonals
        if abs(board[i]-col) == abs(i-row) {
            return false
        }
    }
    return true
}

func abs(x int) int {
    if x < 0 {
        return -x
    }
    return x
}

# Python: N-Queens
def solve_n_queens(n):
    result = []
    board = [-1] * n  # board[i] = column of queen in row i
    
    def is_valid(row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False
        return True
    
    def backtrack(row):
        if row == n:
            # Convert to string representation
            solution = []
            for i in range(n):
                row_str = ""
                for j in range(n):
                    row_str += "Q" if board[i] == j else "."
                solution.append(row_str)
            result.append(solution)
            return
        
        for col in range(n):
            if is_valid(row, col):
                board[row] = col
                backtrack(row + 1)
    
    backtrack(0)
    return result`,
				},
				{
					Title: "Sudoku Solver",
					Content: `Solving Sudoku using backtracking demonstrates how to handle constraint satisfaction problems with multiple constraints.

**Problem Statement:**
Fill a 9×9 grid with digits so that:
- Each row contains digits 1-9 exactly once
- Each column contains digits 1-9 exactly once
- Each 3×3 subgrid contains digits 1-9 exactly once

**Approach:**
1. Find empty cell
2. Try digits 1-9
3. Check if digit is valid (row, column, box constraints)
4. If valid, place digit and recurse
5. If no digit works, backtrack (return false)
6. If all cells filled, solution found

**Optimization:**
- Check constraints efficiently
- Find next empty cell quickly
- Use bit masks for faster constraint checking
- Prune early when constraint violated

**Key Techniques:**
- **Constraint Checking**: Verify row, column, and box constraints
- **Cell Selection**: Choose next empty cell to fill
- **Backtracking**: Undo placement when stuck
- **Early Termination**: Return immediately when solution found

**Time Complexity:** O(9^m) where m is number of empty cells (worst case)
**Space Complexity:** O(1) excluding recursion stack

**Variations:**
- Count number of solutions
- Check if solution exists
- Generate valid sudoku puzzles`,
					CodeExamples: `// Go: Sudoku Solver
func solveSudoku(board [][]byte) bool {
    for i := 0; i < 9; i++ {
        for j := 0; j < 9; j++ {
            if board[i][j] == '.' {
                for num := byte('1'); num <= '9'; num++ {
                    if isValidSudoku(board, i, j, num) {
                        board[i][j] = num
                        if solveSudoku(board) {
                            return true
                        }
                        board[i][j] = '.'  // Backtrack
                    }
                }
                return false  // No valid number
            }
        }
    }
    return true  // All cells filled
}

func isValidSudoku(board [][]byte, row, col int, num byte) bool {
    // Check row
    for j := 0; j < 9; j++ {
        if board[row][j] == num {
            return false
        }
    }
    
    // Check column
    for i := 0; i < 9; i++ {
        if board[i][col] == num {
            return false
        }
    }
    
    // Check 3x3 box
    boxRow, boxCol := (row/3)*3, (col/3)*3
    for i := boxRow; i < boxRow+3; i++ {
        for j := boxCol; j < boxCol+3; j++ {
            if board[i][j] == num {
                return false
            }
        }
    }
    
    return true
}

# Python: Sudoku Solver
def solve_sudoku(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == '.':
                for num in '123456789':
                    if is_valid(board, i, j, num):
                        board[i][j] = num
                        if solve_sudoku(board):
                            return True
                        board[i][j] = '.'  # Backtrack
                return False
    return True

def is_valid(board, row, col, num):
    # Check row
    for j in range(9):
        if board[row][j] == num:
            return False
    
    # Check column
    for i in range(9):
        if board[i][col] == num:
            return False
    
    # Check 3x3 box
    box_row, box_col = (row // 3) * 3, (col // 3) * 3
    for i in range(box_row, box_row + 3):
        for j in range(box_col, box_col + 3):
            if board[i][j] == num:
                return False
    
    return True`,
				},
				{
					Title: "Permutations and Combinations",
					Content: `Generating permutations and combinations is fundamental to many backtracking problems. Understanding these patterns is essential.

**Permutations:**
- Order matters: [1,2,3] ≠ [3,2,1]
- All arrangements of elements
- Number of permutations: n! for n distinct elements
- Example: Permutations of [1,2,3] = [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

**Combinations:**
- Order doesn't matter: [1,2] = [2,1]
- Selections of elements
- Number of combinations: C(n,k) = n!/(k!(n-k)!)
- Example: Combinations of [1,2,3] choose 2 = [[1,2], [1,3], [2,3]]

**Subsets:**
- All possible combinations (choose 0 to n elements)
- Number of subsets: 2^n
- Example: Subsets of [1,2] = [[], [1], [2], [1,2]]

**Backtracking Patterns:**

**Permutations:**
- Try each unused element at current position
- Mark as used, recurse, unmark (backtrack)

**Combinations:**
- Only consider elements after current position (avoid duplicates)
- Don't need to track used (order doesn't matter)

**Subsets:**
- For each element: include or exclude
- Two choices per element: 2^n possibilities

**Key Techniques:**
- **Used Array**: Track which elements already used (permutations)
- **Start Index**: Only consider elements from start index (combinations)
- **Include/Exclude**: Two branches per element (subsets)`,
					CodeExamples: `// Go: Generate Permutations
func permute(nums []int) [][]int {
    result := [][]int{}
    used := make([]bool, len(nums))
    
    var backtrack func([]int)
    backtrack = func(current []int) {
        if len(current) == len(nums) {
            result = append(result, append([]int{}, current...))
            return
        }
        
        for i := 0; i < len(nums); i++ {
            if !used[i] {
                used[i] = true
                current = append(current, nums[i])
                backtrack(current)
                current = current[:len(current)-1]
                used[i] = false
            }
        }
    }
    
    backtrack([]int{})
    return result
}

// Go: Generate Combinations (choose k from n)
func combine(n int, k int) [][]int {
    result := [][]int{}
    
    var backtrack func(int, []int)
    backtrack = func(start int, current []int) {
        if len(current) == k {
            result = append(result, append([]int{}, current...))
            return
        }
        
        for i := start; i <= n; i++ {
            current = append(current, i)
            backtrack(i+1, current)  // Next start is i+1
            current = current[:len(current)-1]
        }
    }
    
    backtrack(1, []int{})
    return result
}

// Go: Generate Subsets
func subsets(nums []int) [][]int {
    result := [][]int{}
    
    var backtrack func(int, []int)
    backtrack = func(start int, current []int) {
        result = append(result, append([]int{}, current...))
        
        for i := start; i < len(nums); i++ {
            current = append(current, nums[i])
            backtrack(i+1, current)
            current = current[:len(current)-1]
        }
    }
    
    backtrack(0, []int{})
    return result
}

# Python: Generate Permutations
def permute(nums):
    result = []
    used = [False] * len(nums)
    
    def backtrack(current):
        if len(current) == len(nums):
            result.append(current[:])
            return
        
        for i in range(len(nums)):
            if not used[i]:
                used[i] = True
                current.append(nums[i])
                backtrack(current)
                current.pop()
                used[i] = False
    
    backtrack([])
    return result

# Python: Generate Combinations
def combine(n, k):
    result = []
    
    def backtrack(start, current):
        if len(current) == k:
            result.append(current[:])
            return
        
        for i in range(start, n + 1):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(1, [])
    return result

# Python: Generate Subsets
def subsets(nums):
    result = []
    
    def backtrack(start, current):
        result.append(current[:])
        
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          11,
			Title:       "Greedy Algorithms",
			Description: "Learn greedy algorithms that make locally optimal choices to find globally optimal solutions.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to Greedy Algorithms",
					Content: `Greedy algorithms make the best choice at each step, hoping it leads to a globally optimal solution. They're simple, efficient, but only work for specific problem types.

**Key Principle:**
- Make locally optimal choice at each step
- Never reconsider previous choices
- Hope local optima lead to global optimum

**When Greedy Works:**
- **Greedy Choice Property**: Global optimum includes optimal local choice
- **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems
- **No need to explore all possibilities**: Can make decision immediately

**When Greedy Fails:**
- Local optimum doesn't lead to global optimum
- Need to consider future consequences
- Problem requires exploring all possibilities

**Common Greedy Problems:**
- Activity Selection: Choose maximum non-overlapping activities
- Fractional Knapsack: Take items with best value/weight ratio
- Huffman Coding: Build optimal prefix code
- Minimum Spanning Tree: Kruskal's and Prim's algorithms
- Shortest Path: Dijkstra's algorithm (greedy approach)

**Greedy vs Dynamic Programming:**
- **Greedy**: Make choice and never reconsider
- **DP**: Consider all possibilities, choose best
- Greedy is faster but only works for specific problems

**Proving Greedy Correctness:**
- Show greedy choice is part of optimal solution
- Show optimal substructure property
- Use exchange argument or induction`,
					CodeExamples: `// Go: Activity Selection (Greedy)
type Activity struct {
    Start, End int
}

func activitySelection(activities []Activity) []Activity {
    // Sort by end time
    sort.Slice(activities, func(i, j int) bool {
        return activities[i].End < activities[j].End
    })
    
    result := []Activity{activities[0]}
    lastEnd := activities[0].End
    
    for i := 1; i < len(activities); i++ {
        if activities[i].Start >= lastEnd {
            result = append(result, activities[i])
            lastEnd = activities[i].End
        }
    }
    
    return result
}

// Go: Fractional Knapsack
type Item struct {
    Value, Weight int
}

func fractionalKnapsack(items []Item, capacity int) float64 {
    // Sort by value/weight ratio (descending)
    sort.Slice(items, func(i, j int) bool {
        ratioI := float64(items[i].Value) / float64(items[i].Weight)
        ratioJ := float64(items[j].Value) / float64(items[j].Weight)
        return ratioI > ratioJ
    })
    
    totalValue := 0.0
    remaining := capacity
    
    for _, item := range items {
        if remaining >= item.Weight {
            totalValue += float64(item.Value)
            remaining -= item.Weight
        } else {
            fraction := float64(remaining) / float64(item.Weight)
            totalValue += float64(item.Value) * fraction
            break
        }
    }
    
    return totalValue
}

# Python: Activity Selection
def activity_selection(activities):
    # Sort by end time
    activities.sort(key=lambda x: x[1])
    
    result = [activities[0]]
    last_end = activities[0][1]
    
    for start, end in activities[1:]:
        if start >= last_end:
            result.append((start, end))
            last_end = end
    
    return result

# Python: Fractional Knapsack
def fractional_knapsack(items, capacity):
    # Sort by value/weight ratio
    items.sort(key=lambda x: x[0]/x[1], reverse=True)
    
    total_value = 0.0
    remaining = capacity
    
    for value, weight in items:
        if remaining >= weight:
            total_value += value
            remaining -= weight
        else:
            fraction = remaining / weight
            total_value += value * fraction
            break
    
    return total_value`,
				},
				{
					Title: "Activity Selection",
					Content: `The activity selection problem is a classic greedy problem: select the maximum number of non-overlapping activities.

**Problem Statement:**
Given n activities with start and end times, select maximum number of activities that don't overlap.

**Greedy Strategy:**
1. Sort activities by end time (earliest finish first)
2. Select first activity
3. For each remaining activity:
   - If it starts after last selected activity ends, select it
   - Otherwise, skip it

**Why This Works:**
- Always better to finish earlier (leaves more time for other activities)
- If activity A finishes before B, and we can choose only one, choose A
- This leaves maximum time for remaining activities

**Proof:**
- Assume optimal solution doesn't include earliest-finishing activity
- Replace first activity in optimal with earliest-finishing
- Still valid (doesn't conflict), same or better count
- Therefore, earliest-finishing must be in optimal solution

**Time Complexity:** O(n log n) - sorting dominates
**Space Complexity:** O(1) excluding input/output

**Variations:**
- Weighted activity selection (use DP, not greedy)
- Maximum non-overlapping intervals
- Meeting room scheduling`,
					CodeExamples: `// Go: Activity Selection
func maxActivities(activities [][]int) int {
    if len(activities) == 0 {
        return 0
    }
    
    // Sort by end time
    sort.Slice(activities, func(i, j int) bool {
        return activities[i][1] < activities[j][1]
    })
    
    count := 1
    lastEnd := activities[0][1]
    
    for i := 1; i < len(activities); i++ {
        if activities[i][0] >= lastEnd {
            count++
            lastEnd = activities[i][1]
        }
    }
    
    return count
}

# Python: Activity Selection
def max_activities(activities):
    if not activities:
        return 0
    
    activities.sort(key=lambda x: x[1])
    count = 1
    last_end = activities[0][1]
    
    for start, end in activities[1:]:
        if start >= last_end:
            count += 1
            last_end = end
    
    return count`,
				},
				{
					Title: "Fractional Knapsack",
					Content: `Fractional knapsack allows taking fractions of items, making it solvable with a greedy approach (unlike 0/1 knapsack which requires DP).

**Problem Statement:**
Given items with values and weights, and a knapsack capacity, maximize value. You can take fractions of items.

**Greedy Strategy:**
1. Calculate value/weight ratio for each item
2. Sort items by ratio (highest first)
3. Take items greedily:
   - If item fits completely, take it all
   - Otherwise, take fraction that fits
   - Stop when knapsack full

**Why This Works:**
- Always better to take item with higher value/weight ratio
- If we have capacity, better to use it for highest ratio item
- Taking fraction doesn't prevent taking other items

**Key Insight:**
- Since we can take fractions, always take highest ratio first
- This maximizes value per unit weight
- Greedy choice is always optimal

**Time Complexity:** O(n log n) - sorting dominates
**Space Complexity:** O(1) excluding input

**Note:** This is different from 0/1 knapsack (can't take fractions), which requires dynamic programming.`,
					CodeExamples: `// Go: Fractional Knapsack
func fractionalKnapsack(items []Item, capacity int) float64 {
    // Sort by value/weight ratio (descending)
    sort.Slice(items, func(i, j int) bool {
        ratioI := float64(items[i].Value) / float64(items[i].Weight)
        ratioJ := float64(items[j].Value) / float64(items[j].Weight)
        return ratioI > ratioJ
    })
    
    totalValue := 0.0
    remaining := capacity
    
    for _, item := range items {
        if remaining >= item.Weight {
            totalValue += float64(item.Value)
            remaining -= item.Weight
        } else {
            fraction := float64(remaining) / float64(item.Weight)
            totalValue += float64(item.Value) * fraction
            break
        }
    }
    
    return totalValue
}

# Python: Fractional Knapsack
def fractional_knapsack(items, capacity):
    # items: list of (value, weight) tuples
    items.sort(key=lambda x: x[0]/x[1], reverse=True)
    
    total_value = 0.0
    remaining = capacity
    
    for value, weight in items:
        if remaining >= weight:
            total_value += value
            remaining -= weight
        else:
            fraction = remaining / weight
            total_value += value * fraction
            break
    
    return total_value`,
				},
				{
					Title: "Huffman Coding",
					Content: `Huffman coding is a lossless data compression algorithm that assigns variable-length codes to characters based on frequency.

**Problem Statement:**
Given characters and their frequencies, assign binary codes to minimize total encoded length.

**Greedy Strategy:**
1. Create leaf node for each character with frequency
2. Build min-heap (priority queue) of nodes
3. Repeatedly:
   - Extract two nodes with lowest frequency
   - Create internal node with these as children
   - Frequency = sum of children frequencies
   - Insert back into heap
4. Continue until one node remains (root)
5. Traverse tree to assign codes (left=0, right=1)

**Why This Works:**
- Characters with higher frequency should have shorter codes
- Greedy: Always merge two least frequent nodes
- This minimizes total encoding length

**Properties:**
- Prefix code: No code is prefix of another
- Optimal: Minimizes expected code length
- Can be decoded uniquely

**Time Complexity:** O(n log n) - n insertions/extractions from heap
**Space Complexity:** O(n) - for tree and codes

**Applications:**
- Data compression (ZIP, JPEG)
- File compression utilities
- Network protocols
- Image compression`,
					CodeExamples: `// Go: Huffman Coding
import "container/heap"

type HuffmanNode struct {
    char     rune
    freq     int
    left     *HuffmanNode
    right    *HuffmanNode
}

type MinHeap []*HuffmanNode

func (h MinHeap) Len() int { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].freq < h[j].freq }
func (h MinHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x interface{}) { *h = append(*h, x.(*HuffmanNode)) }
func (h *MinHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0:n-1]
    return x
}

func buildHuffmanTree(freq map[rune]int) *HuffmanNode {
    h := &MinHeap{}
    heap.Init(h)
    
    // Create leaf nodes
    for char, frequency := range freq {
        heap.Push(h, &HuffmanNode{char: char, freq: frequency})
    }
    
    // Build tree
    for h.Len() > 1 {
        left := heap.Pop(h).(*HuffmanNode)
        right := heap.Pop(h).(*HuffmanNode)
        
        internal := &HuffmanNode{
            freq: left.freq + right.freq,
            left: left,
            right: right,
        }
        heap.Push(h, internal)
    }
    
    return heap.Pop(h).(*HuffmanNode)
}

func buildCodes(root *HuffmanNode) map[rune]string {
    codes := make(map[rune]string)
    
    var traverse func(*HuffmanNode, string)
    traverse = func(node *HuffmanNode, code string) {
        if node.left == nil && node.right == nil {
            codes[node.char] = code
            return
        }
        traverse(node.left, code+"0")
        traverse(node.right, code+"1")
    }
    
    traverse(root, "")
    return codes
}

# Python: Huffman Coding
import heapq

class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq):
    heap = []
    for char, frequency in freq.items():
        heapq.heappush(heap, HuffmanNode(char=char, freq=frequency))
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        internal = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, internal)
    
    return heapq.heappop(heap)

def build_codes(root):
    codes = {}
    
    def traverse(node, code):
        if node.left is None and node.right is None:
            codes[node.char] = code
            return
        traverse(node.left, code + "0")
        traverse(node.right, code + "1")
    
    traverse(root, "")
    return codes`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          12,
			Title:       "Bit Manipulation",
			Description: "Master bit manipulation techniques for efficient problem solving.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Bitwise Operations",
					Content: `Bit manipulation involves directly manipulating bits in binary representation of numbers. It's powerful for optimization and certain problem types.

**Basic Operations:**

**1. AND (&):**
- Result bit is 1 only if both bits are 1
- Use: Check if bit is set, clear bits
- Example: 5 & 3 = 1 (101 & 011 = 001)

**2. OR (|):**
- Result bit is 1 if either bit is 1
- Use: Set bits, combine flags
- Example: 5 | 3 = 7 (101 | 011 = 111)

**3. XOR (^):**
- Result bit is 1 if bits are different
- Use: Toggle bits, find unique element
- Example: 5 ^ 3 = 6 (101 ^ 011 = 110)
- Properties: a ^ a = 0, a ^ 0 = a

**4. NOT (~):**
- Flips all bits
- Use: Complement, create masks
- Example: ~5 = ...11111010 (depends on integer size)

**5. Left Shift (<<):**
- Shifts bits left, fills with 0
- Equivalent to multiplying by 2^n
- Example: 5 << 1 = 10 (101 << 1 = 1010)

**6. Right Shift (>>):**
- Shifts bits right
- Equivalent to dividing by 2^n (for unsigned)
- Example: 5 >> 1 = 2 (101 >> 1 = 10)

**Common Tricks:**
- Check if bit is set: (n & (1 << i)) != 0
- Set bit: n | (1 << i)
- Clear bit: n & ~(1 << i)
- Toggle bit: n ^ (1 << i)
- Get lowest set bit: n & -n
- Clear lowest set bit: n & (n-1)
- Check if power of 2: (n & (n-1)) == 0`,
					CodeExamples: `// Go: Bitwise Operations
func bitwiseExamples() {
    a, b := 5, 3  // 101, 011
    
    and := a & b   // 001 = 1
    or := a | b    // 111 = 7
    xor := a ^ b   // 110 = 6
    notA := ^a     // Complement
    leftShift := a << 1  // 1010 = 10
    rightShift := a >> 1 // 010 = 2
}

// Go: Check if bit is set
func isBitSet(n int, pos int) bool {
    return (n & (1 << pos)) != 0
}

// Go: Set bit
func setBit(n int, pos int) int {
    return n | (1 << pos)
}

// Go: Clear bit
func clearBit(n int, pos int) int {
    return n & ^(1 << pos)
}

// Go: Toggle bit
func toggleBit(n int, pos int) int {
    return n ^ (1 << pos)
}

// Go: Count set bits
func countSetBits(n int) int {
    count := 0
    for n > 0 {
        count++
        n &= n - 1  // Clear lowest set bit
    }
    return count
}

// Go: Check if power of 2
func isPowerOfTwo(n int) bool {
    return n > 0 && (n & (n-1)) == 0
}

# Python: Bitwise Operations
def bitwise_examples():
    a, b = 5, 3  # 101, 011
    
    and_op = a & b   # 1
    or_op = a | b    # 7
    xor_op = a ^ b   # 6
    not_a = ~a       # Complement
    left_shift = a << 1  # 10
    right_shift = a >> 1  # 2

# Python: Check if bit is set
def is_bit_set(n, pos):
    return (n & (1 << pos)) != 0

# Python: Count set bits
def count_set_bits(n):
    count = 0
    while n:
        count += 1
        n &= n - 1  # Clear lowest set bit
    return count

# Python: Check if power of 2
def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0`,
				},
				{
					Title: "Bit Tricks",
					Content: `Master common bit manipulation tricks that appear frequently in coding problems.

**Essential Bit Tricks:**

**1. Get/Set/Clear/Toggle Bit:**
- Get: (n >> i) & 1
- Set: n | (1 << i)
- Clear: n & ~(1 << i)
- Toggle: n ^ (1 << i)

**2. Check Power of 2:**
- (n & (n-1)) == 0 for n > 0
- Only one bit set in power of 2

**3. Get Lowest Set Bit:**
- n & -n (two's complement)
- Isolates rightmost 1 bit

**4. Clear Lowest Set Bit:**
- n & (n-1)
- Removes rightmost 1 bit
- Used in counting set bits

**5. Swap Without Temp:**
- a ^= b; b ^= a; a ^= b
- Uses XOR properties

**6. Find Missing Number:**
- XOR all numbers with 1..n
- Missing number remains

**7. Find Single Element:**
- XOR all elements
- Pairs cancel out, single element remains

**8. Reverse Bits:**
- Extract bits and rebuild in reverse
- Or use lookup table for speed

**Common Problems:**
- Single Number (XOR trick)
- Power of Two
- Number of 1 Bits
- Reverse Bits
- Missing Number
- Bitwise AND of Numbers Range`,
					CodeExamples: `// Go: Single Number (XOR trick)
func singleNumber(nums []int) int {
    result := 0
    for _, num := range nums {
        result ^= num  // XOR all numbers
    }
    return result  // Pairs cancel out
}

// Go: Reverse Bits
func reverseBits(n uint32) uint32 {
    result := uint32(0)
    for i := 0; i < 32; i++ {
        result <<= 1
        result |= n & 1
        n >>= 1
    }
    return result
}

// Go: Number of 1 Bits
func hammingWeight(n uint32) int {
    count := 0
    for n > 0 {
        count++
        n &= n - 1  // Clear lowest set bit
    }
    return count
}

// Go: Missing Number
func missingNumber(nums []int) int {
    n := len(nums)
    result := 0
    
    // XOR all numbers
    for _, num := range nums {
        result ^= num
    }
    
    // XOR with 0..n
    for i := 0; i <= n; i++ {
        result ^= i
    }
    
    return result
}

# Python: Single Number
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result

# Python: Reverse Bits
def reverse_bits(n):
    result = 0
    for i in range(32):
        result <<= 1
        result |= n & 1
        n >>= 1
    return result

# Python: Number of 1 Bits
def hamming_weight(n):
    count = 0
    while n:
        count += 1
        n &= n - 1
    return count`,
				},
				{
					Title: "Bit DP",
					Content: `Bit DP (Dynamic Programming with Bitmasks) uses bit manipulation to represent states in DP, enabling efficient solutions to subset and permutation problems.

**Key Concept:**
- Use integer bitmask to represent subset/state
- Each bit represents inclusion/exclusion of element
- dp[mask] = optimal value for state represented by mask
- Enables DP on subsets without storing arrays

**Bitmask Representation:**
- mask = 0b1011 means elements 0, 1, 3 are selected
- Set bit i: mask | (1 << i)
- Check bit i: (mask >> i) & 1
- Toggle bit i: mask ^ (1 << i)

**Common Problems:**
- Traveling Salesman Problem (TSP)
- Assignment Problem
- Subset Sum variations
- Graph problems with subset states

**TSP Example:**
- dp[mask][last] = minimum cost to visit cities in mask ending at last
- Try all unvisited cities as next
- Update dp[mask | (1 << next)][next]

**Time Complexity:** Usually O(2^n × n) or O(2^n × n²)
**Space Complexity:** O(2^n × n)

**When to Use:**
- Need to track subsets efficiently
- State can be represented as bitmask
- Problem size allows 2^n states (n typically ≤ 20)
- Need to avoid storing large arrays`,
					CodeExamples: `// Go: Traveling Salesman Problem (TSP) with Bit DP
func tsp(dist [][]int) int {
    n := len(dist)
    if n == 0 {
        return 0
    }
    
    // dp[mask][last] = min cost to visit cities in mask ending at last
    dp := make([][]int, 1<<n)
    for i := range dp {
        dp[i] = make([]int, n)
        for j := range dp[i] {
            dp[i][j] = int(^uint(0) >> 1)  // Max int
        }
    }
    
    // Base case: starting from city 0
    dp[1][0] = 0
    
    // Try all masks
    for mask := 1; mask < (1 << n); mask++ {
        for last := 0; last < n; last++ {
            if dp[mask][last] == int(^uint(0)>>1) {
                continue
            }
            
            // Try all unvisited cities
            for next := 0; next < n; next++ {
                if (mask & (1 << next)) == 0 {  // Not visited
                    newMask := mask | (1 << next)
                    newCost := dp[mask][last] + dist[last][next]
                    if newCost < dp[newMask][next] {
                        dp[newMask][next] = newCost
                    }
                }
            }
        }
    }
    
    // Find minimum cost to visit all cities and return to start
    result := int(^uint(0) >> 1)
    allVisited := (1 << n) - 1
    for last := 0; last < n; last++ {
        totalCost := dp[allVisited][last] + dist[last][0]
        if totalCost < result {
            result = totalCost
        }
    }
    
    return result
}

# Python: TSP with Bit DP
def tsp(dist):
    n = len(dist)
    if n == 0:
        return 0
    
    # dp[mask][last] = min cost
    dp = [[float('inf')] * n for _ in range(1 << n)]
    
    # Base case
    dp[1][0] = 0
    
    # Try all masks
    for mask in range(1, 1 << n):
        for last in range(n):
            if dp[mask][last] == float('inf'):
                continue
            
            # Try unvisited cities
            for next_city in range(n):
                if not (mask & (1 << next_city)):
                    new_mask = mask | (1 << next_city)
                    new_cost = dp[mask][last] + dist[last][next_city]
                    dp[new_mask][next_city] = min(dp[new_mask][next_city], new_cost)
    
    # Return to start
    result = float('inf')
    all_visited = (1 << n) - 1
    for last in range(n):
        result = min(result, dp[all_visited][last] + dist[last][0])
    
    return int(result)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          13,
			Title:       "Math & Number Theory",
			Description: "Master mathematical concepts and number theory algorithms essential for competitive programming and problem-solving.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "Prime Numbers",
					Content: `Prime numbers are fundamental in number theory and appear in many algorithmic problems. Understanding prime-related algorithms is essential.

**Prime Number Properties:**
- Number > 1 divisible only by 1 and itself
- 2 is only even prime
- All primes > 2 are odd
- Infinite primes exist

**Prime Checking:**

**1. Naive Approach:**
- Check divisibility from 2 to n-1
- Time: O(n)

**2. Optimized:**
- Check up to √n (if n = a×b, one factor ≤ √n)
- Time: O(√n)

**3. Sieve of Eratosthenes:**
- Mark multiples of primes as composite
- Find all primes up to n
- Time: O(n log log n)

**Prime Factorization:**
- Express number as product of primes
- Useful for: GCD, LCM, divisors
- Time: O(√n)

**Applications:**
- Cryptography (RSA)
- Hash functions
- Random number generation
- Mathematical problems`,
					CodeExamples: `// Go: Check if prime
func isPrime(n int) bool {
    if n < 2 {
        return false
    }
    if n == 2 {
        return true
    }
    if n%2 == 0 {
        return false
    }
    
    for i := 3; i*i <= n; i += 2 {
        if n%i == 0 {
            return false
        }
    }
    return true
}

// Go: Sieve of Eratosthenes
func sieve(n int) []bool {
    isPrime := make([]bool, n+1)
    for i := 2; i <= n; i++ {
        isPrime[i] = true
    }
    
    for i := 2; i*i <= n; i++ {
        if isPrime[i] {
            for j := i * i; j <= n; j += i {
                isPrime[j] = false
            }
        }
    }
    
    return isPrime
}

// Go: Prime Factorization
func primeFactors(n int) map[int]int {
    factors := make(map[int]int)
    
    for i := 2; i*i <= n; i++ {
        for n%i == 0 {
            factors[i]++
            n /= i
        }
    }
    
    if n > 1 {
        factors[n]++
    }
    
    return factors
}

# Python: Check if prime
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Python: Sieve of Eratosthenes
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    return is_prime

# Python: Prime Factorization
def prime_factors(n):
    factors = {}
    i = 2
    
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
        i += 1
    
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    
    return factors`,
				},
				{
					Title: "GCD and LCM",
					Content: `GCD (Greatest Common Divisor) and LCM (Least Common Multiple) are fundamental number theory concepts with many applications.

**GCD (Greatest Common Divisor):**
- Largest number that divides both numbers
- Example: GCD(12, 18) = 6
- Used in: Fraction simplification, modular arithmetic

**LCM (Least Common Multiple):**
- Smallest number divisible by both numbers
- Example: LCM(12, 18) = 36
- Relationship: LCM(a, b) = (a × b) / GCD(a, b)

**Euclidean Algorithm:**
- Efficient algorithm for computing GCD
- Based on: GCD(a, b) = GCD(b, a mod b)
- Time: O(log min(a, b))

**Extended Euclidean Algorithm:**
- Finds GCD and coefficients x, y such that: ax + by = GCD(a, b)
- Useful for: Modular inverse, solving linear Diophantine equations

**Applications:**
- Fraction arithmetic
- Modular arithmetic
- Cryptography
- Solving equations
- Optimizing algorithms`,
					CodeExamples: `// Go: GCD using Euclidean Algorithm
func gcd(a, b int) int {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}

// Go: GCD Recursive
func gcdRecursive(a, b int) int {
    if b == 0 {
        return a
    }
    return gcdRecursive(b, a%b)
}

// Go: LCM
func lcm(a, b int) int {
    return a * b / gcd(a, b)
}

// Go: Extended Euclidean Algorithm
func extendedGCD(a, b int) (int, int, int) {
    if b == 0 {
        return a, 1, 0
    }
    gcd, x1, y1 := extendedGCD(b, a%b)
    x := y1
    y := x1 - (a/b)*y1
    return gcd, x, y
}

// Go: Modular Inverse (if exists)
func modInverse(a, m int) int {
    gcd, x, _ := extendedGCD(a, m)
    if gcd != 1 {
        return -1  // Inverse doesn't exist
    }
    return (x%m + m) % m
}

# Python: GCD
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Python: LCM
def lcm(a, b):
    return a * b // gcd(a, b)

# Python: Extended Euclidean
def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    gcd, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return gcd, x, y

# Python: Modular Inverse
def mod_inverse(a, m):
    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        return None
    return (x % m + m) % m`,
				},
				{
					Title: "Modular Arithmetic",
					Content: `Modular arithmetic deals with remainders and is crucial for handling large numbers and cryptographic operations.

**Basic Operations:**
- a mod m = remainder when a divided by m
- Properties: (a + b) mod m = ((a mod m) + (b mod m)) mod m
- (a × b) mod m = ((a mod m) × (b mod m)) mod m

**Modular Exponentiation:**
- Compute a^b mod m efficiently
- Use binary exponentiation (fast exponentiation)
- Time: O(log b) instead of O(b)

**Modular Inverse:**
- Number x such that (a × x) mod m = 1
- Exists only if GCD(a, m) = 1
- Use Extended Euclidean Algorithm

**Fermat's Little Theorem:**
- If p is prime and GCD(a, p) = 1: a^(p-1) mod p = 1
- Corollary: a^(-1) mod p = a^(p-2) mod p (when p is prime)

**Chinese Remainder Theorem:**
- Solve system of congruences
- Find number satisfying multiple modulo conditions

**Applications:**
- Cryptography (RSA, Diffie-Hellman)
- Hash functions
- Random number generation
- Avoiding overflow with large numbers`,
					CodeExamples: `// Go: Modular Exponentiation (Fast Power)
func modPow(base, exp, mod int) int {
    result := 1
    base %= mod
    
    for exp > 0 {
        if exp%2 == 1 {
            result = (result * base) % mod
        }
        exp >>= 1
        base = (base * base) % mod
    }
    
    return result
}

// Go: Modular Inverse using Fermat's Little Theorem
func modInverseFermat(a, p int) int {
    // p must be prime
    return modPow(a, p-2, p)
}

// Go: Modular Arithmetic Operations
func modAdd(a, b, m int) int {
    return ((a % m) + (b % m)) % m
}

func modMul(a, b, m int) int {
    return ((a % m) * (b % m)) % m
}

func modSub(a, b, m int) int {
    return ((a%m - b%m) + m) % m
}

# Python: Modular Exponentiation
def mod_pow(base, exp, mod):
    result = 1
    base %= mod
    
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    
    return result

# Python: Modular Inverse (Fermat's)
def mod_inverse_fermat(a, p):
    return mod_pow(a, p - 2, p)

# Python: Modular Operations
def mod_add(a, b, m):
    return ((a % m) + (b % m)) % m

def mod_mul(a, b, m):
    return ((a % m) * (b % m)) % m`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
