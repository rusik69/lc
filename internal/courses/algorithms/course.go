package algorithms

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAlgorithmsModules([]problems.CourseModule{
		{
			ID:          0,
			Title:       "Introduction & Complexity Analysis",
			Description: "Master the fundamentals: what data structures are, how to analyze algorithms, and understand Big O notation.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction",
					Content: `Welcome to Data Structures and Algorithms! This course will teach you how to solve complex programming problems efficiently.

**What You'll Learn:**
- Fundamental data structures (arrays, linked lists, trees, graphs, etc.)
- Essential algorithms (sorting, searching, dynamic programming, etc.)
- Problem-solving techniques and patterns
- How to analyze time and space complexity

**Why This Matters:**
- Write efficient code that scales
- Solve technical interview problems
- Understand how software systems work
- Build better applications

**Course Structure:**
Each module contains lessons explaining concepts, followed by practice problems to reinforce your learning. Start with the fundamentals and progress to advanced topics.`,
					CodeExamples: `// Example: Understanding the problem-solving process
// Problem: Find the maximum number in an array

// Approach 1: Simple iteration
func findMax(nums []int) int {
    max := nums[0]
    for i := 1; i < len(nums); i++ {
        if nums[i] > max {
            max = nums[i]
        }
    }
    return max
}

# Python equivalent
def find_max(nums):
    max_val = nums[0]
    for num in nums[1:]:
        if num > max_val:
            max_val = num
    return max_val`,
				},
				{
					Title: "What Are Data Structures?",
					Content: `Data structures are ways of organizing and storing data in a computer so that it can be accessed and modified efficiently.

**Types of Data Structures:**

1. **Linear Structures:**
   - Arrays: Contiguous memory, indexed access
   - Linked Lists: Nodes connected by pointers
   - Stacks: LIFO (Last In, First Out)
   - Queues: FIFO (First In, First Out)

2. **Non-Linear Structures:**
   - Trees: Hierarchical relationships
   - Graphs: Network of connected nodes
   - Hash Tables: Key-value pairs

**Choosing the Right Data Structure:**
- **Arrays**: When you need random access by index
- **Linked Lists**: When you need frequent insertions/deletions
- **Hash Tables**: When you need fast lookups
- **Trees**: When you need hierarchical data or sorted access
- **Graphs**: When you need to model relationships

**Key Principle:** The right data structure can make an O(n²) algorithm O(n log n) or even O(n)!`,
					CodeExamples: `// Example: Same problem, different data structures

// Using array - O(n) search
func findInArray(arr []int, target int) bool {
    for _, val := range arr {
        if val == target {
            return true
        }
    }
    return false
}

// Using hash map - O(1) average lookup
func findInMap(m map[int]bool, target int) bool {
    return m[target]
}

# Python equivalent
def find_in_array(arr, target):
    return target in arr  # O(n)

def find_in_map(m, target):
    return target in m  # O(1) average`,
				},
				{
					Title: "Complexity Analysis",
					Content: `Complexity analysis helps us understand how algorithms perform as input size grows. It's crucial for writing efficient code.

**Why Analyze Complexity?**
- Predict performance on large inputs
- Compare different algorithms
- Optimize bottlenecks
- Make informed design decisions

**What We Analyze:**
1. **Time Complexity**: How long an algorithm takes
2. **Space Complexity**: How much memory an algorithm uses

**Common Scenarios:**
- **Best Case**: Minimum time/space needed
- **Average Case**: Expected time/space for typical input
- **Worst Case**: Maximum time/space needed (most important!)

**Key Insight:** We usually focus on worst-case complexity because we want guarantees about performance.`,
					CodeExamples: `// Example: Analyzing complexity

// O(1) - Constant time
func getFirst(arr []int) int {
    return arr[0]  // Always one operation
}

// O(n) - Linear time
func sumArray(arr []int) int {
    sum := 0
    for _, val := range arr {  // n operations
        sum += val
    }
    return sum
}

// O(n²) - Quadratic time
func findPairs(arr []int) int {
    count := 0
    for i := 0; i < len(arr); i++ {  // n iterations
        for j := i + 1; j < len(arr); j++ {  // n iterations
            count++  // Total: n * n = n²
        }
    }
    return count
}`,
				},
				{
					Title: "Memory",
					Content: `Understanding how memory works helps you write efficient code and avoid common pitfalls.

**Memory Basics:**
- **Stack**: Fast, limited size, stores local variables
- **Heap**: Larger, slower, stores dynamically allocated data
- **Cache**: Very fast, small, stores frequently accessed data

**Memory Considerations:**

1. **Stack vs Heap:**
   - Stack: Automatic allocation/deallocation, limited size
   - Heap: Manual management (in some languages), larger size

2. **Cache Locality:**
   - Arrays: Good cache locality (contiguous memory)
   - Linked Lists: Poor cache locality (scattered memory)
   - This affects real-world performance significantly!

3. **Space Complexity:**
   - **Auxiliary Space**: Extra space used by algorithm
   - **Input Space**: Space used by input itself
   - We usually focus on auxiliary space

**Memory Optimization Tips:**
- Prefer arrays over linked lists when possible
- Reuse data structures instead of creating new ones
- Be mindful of recursion depth (uses stack space)`,
					CodeExamples: `// Example: Memory considerations

// Good: Reusing array
func processInPlace(arr []int) {
    for i := range arr {
        arr[i] *= 2  // Modifies existing memory
    }
}

// Less efficient: Creating new array
func processNew(arr []int) []int {
    result := make([]int, len(arr))  // New memory allocation
    for i, val := range arr {
        result[i] = val * 2
    }
    return result
}

# Python: Memory considerations
# List comprehension creates new list
result = [x * 2 for x in arr]  # New memory

# In-place modification
for i in range(len(arr)):
    arr[i] *= 2  # Reuses existing memory`,
				},
				{
					Title: "Big O Notation",
					Content: `Big O notation describes how algorithm performance scales with input size. It's the language of algorithm analysis.

**What Big O Means:**
- Describes worst-case performance
- Focuses on growth rate, not exact numbers
- Ignores constants and lower-order terms

**Common Complexities (from fastest to slowest):**

1. **O(1)** - Constant: Always same time
   - Example: Array access, hash table lookup

2. **O(log n)** - Logarithmic: Grows slowly
   - Example: Binary search, balanced tree operations

3. **O(n)** - Linear: Grows proportionally
   - Example: Iterating through array, linear search

4. **O(n log n)** - Linearithmic: Common in efficient sorts
   - Example: Merge sort, heap sort

5. **O(n²)** - Quadratic: Grows quickly
   - Example: Nested loops, bubble sort

6. **O(2ⁿ)** - Exponential: Grows very quickly
   - Example: Recursive Fibonacci (naive), generating subsets

**Rules:**
- Drop constants: O(2n) → O(n)
- Drop lower terms: O(n² + n) → O(n²)
- Different terms for inputs: O(a + b) stays O(a + b)`,
					CodeExamples: `// O(1) - Constant
func get(arr []int, i int) int {
    return arr[i]  // One operation
}

// O(log n) - Logarithmic (binary search)
func binarySearch(arr []int, target int) int {
    left, right := 0, len(arr)-1
    for left <= right {
        mid := (left + right) / 2
        if arr[mid] == target {
            return mid
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    return -1
}

// O(n) - Linear
func findMax(arr []int) int {
    max := arr[0]
    for _, val := range arr {  // n operations
        if val > max {
            max = val
        }
    }
    return max
}

// O(n²) - Quadratic
func bubbleSort(arr []int) {
    for i := 0; i < len(arr); i++ {
        for j := 0; j < len(arr)-i-1; j++ {
            if arr[j] > arr[j+1] {
                arr[j], arr[j+1] = arr[j+1], arr[j]
            }
        }
    }
}`,
				},
				{
					Title: "Logarithm",
					Content: `Logarithms appear frequently in algorithm analysis, especially in divide-and-conquer algorithms.

**What is a Logarithm?**
- log₂(n) = "What power of 2 equals n?"
- log₂(8) = 3 because 2³ = 8
- log₂(16) = 4 because 2⁴ = 16

**Why Logarithms Matter:**
- Binary search: O(log n) - cuts problem in half each step
- Balanced trees: O(log n) operations
- Divide-and-conquer: Often O(n log n)

**Key Properties:**
- log(ab) = log(a) + log(b)
- log(a/b) = log(a) - log(b)
- log(aᵇ) = b × log(a)
- log base doesn't matter for Big O (all bases are equivalent)

**Visual Intuition:**
- log₂(1,000) ≈ 10
- log₂(1,000,000) ≈ 20
- log₂(1,000,000,000) ≈ 30

Notice how slowly logarithms grow! This is why O(log n) algorithms are so efficient.`,
					CodeExamples: `// Example: Why binary search is O(log n)

// Each iteration eliminates half the remaining elements
// After k iterations, we have n / 2ᵏ elements left
// When n / 2ᵏ = 1, we're done
// So k = log₂(n)

func binarySearch(arr []int, target int) bool {
    left, right := 0, len(arr)-1
    
    for left <= right {
        mid := (left + right) / 2  // Divide in half
        
        if arr[mid] == target {
            return true
        } else if arr[mid] < target {
            left = mid + 1  // Search right half
        } else {
            right = mid - 1  // Search left half
        }
        // Each iteration eliminates half the elements
        // Total iterations: log₂(n)
    }
    return false
}

# Python: Understanding logarithmic growth
import math

# How many times can we divide n by 2?
n = 1000000
divisions = 0
while n > 1:
    n //= 2
    divisions += 1

print(f"Can divide {1000000} by 2, {divisions} times")
print(f"log₂(1000000) ≈ {math.log2(1000000):.1f}")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1,
			Title:       "Arrays & Hash Tables",
			Description: "Learn fundamental data structures: arrays and hash tables. Master techniques for efficient lookups and manipulations.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to Arrays",
					Content: `Arrays are one of the most fundamental data structures in computer science. An array is a collection of elements stored in contiguous memory locations, each identified by an index.

**Key Properties:**
- Fixed size (in most languages) - though dynamic arrays exist
- Random access: O(1) time complexity - can jump to any index instantly
- Contiguous memory allocation - elements stored next to each other
- Zero-based indexing - first element at index 0

**Memory Layout:**
Arrays store elements sequentially in memory. For an array [10, 20, 30]:
- Index 0: Memory address = base + 0×size
- Index 1: Memory address = base + 1×size  
- Index 2: Memory address = base + 2×size

This contiguous layout provides excellent cache locality, making arrays very fast for sequential access.

**Common Operations:**
- **Access by index**: O(1) - direct memory calculation
- **Search**: O(n) - must check each element
- **Insertion at end**: O(1) amortized (dynamic arrays)
- **Insertion at position**: O(n) - requires shifting elements
- **Deletion**: O(n) - requires shifting elements

**Dynamic Arrays (Slices/Lists):**
Most modern languages provide dynamic arrays that grow automatically:
- When capacity is reached, allocate larger array (usually 2×)
- Copy existing elements to new array
- Amortized O(1) for append operations

**When to Use Arrays:**
- Need random access by index
- Fixed or predictable size
- Sequential processing
- Cache-friendly operations

**Common Pitfalls:**
- Index out of bounds errors
- Off-by-one errors (0-indexed vs 1-indexed thinking)
- Forgetting arrays are passed by reference in some languages`,
					CodeExamples: `// Go: Array operations
arr := []int{1, 2, 3, 4, 5}

// Access: O(1)
first := arr[0]  // 1
last := arr[len(arr)-1]  // 5

// Search: O(n)
func find(arr []int, target int) int {
    for i, val := range arr {
        if val == target {
            return i
        }
    }
    return -1
}

// Insertion at end: O(1) amortized
arr = append(arr, 6)  // [1, 2, 3, 4, 5, 6]

// Insertion at position: O(n)
func insert(arr []int, index int, value int) []int {
    arr = append(arr, 0)  // Make room
    copy(arr[index+1:], arr[index:])  // Shift right
    arr[index] = value
    return arr
}

# Python: List operations
arr = [1, 2, 3, 4, 5]

# Access: O(1)
first = arr[0]  # 1
last = arr[-1]  # 5 (negative indexing)

# Search: O(n)
def find(arr, target):
    try:
        return arr.index(target)
    except ValueError:
        return -1

# Insertion at end: O(1) amortized
arr.append(6)  # [1, 2, 3, 4, 5, 6]

# Insertion at position: O(n)
arr.insert(2, 99)  # [1, 2, 99, 3, 4, 5, 6]`,
				},
				{
					Title: "Hash Tables",
					Content: `Hash tables (also called hash maps or dictionaries) provide O(1) average-case time complexity for insertions, deletions, and lookups.

**How They Work:**
1. **Hash Function**: Converts keys into array indices
   - Input: key (string, int, etc.)
   - Output: index in underlying array
   - Should distribute keys uniformly

2. **Storage**: Values stored at calculated indices
   - Array of buckets/slots
   - Each bucket can hold one or more key-value pairs

3. **Collision Handling**: When two keys hash to same index
   - **Chaining**: Each bucket is a linked list
   - **Open Addressing**: Find next available slot (linear probing, quadratic probing)

**Hash Function Properties:**
- **Deterministic**: Same key always produces same hash
- **Uniform Distribution**: Keys spread evenly across buckets
- **Fast Computation**: Should be O(1) or very fast
- **Avalanche Effect**: Small change in input → large change in output

**Key Properties:**
- **Average Case**: O(1) for insert, delete, lookup
- **Worst Case**: O(n) if all keys hash to same bucket (rare with good hash function)
- **Load Factor**: ratio of elements to buckets (typically kept < 0.75)
- **No Guaranteed Order**: Keys unordered (unless using ordered hash map)

**Collision Resolution Strategies:**

1. **Separate Chaining**:
   - Each bucket is a linked list
   - Simple, handles many collisions well
   - Extra memory for pointers

2. **Open Addressing**:
   - Store directly in array
   - Linear probing: check next slot
   - Quadratic probing: check slots at increasing distances
   - Better cache performance, but clustering issues

**Common Use Cases:**
- **Frequency Counting**: Count occurrences of elements
- **Fast Lookups**: O(1) instead of O(n) search
- **Caching/Memoization**: Store computed results
- **Grouping**: Group elements by key
- **Deduplication**: Remove duplicates efficiently

**Real-World Applications:**
- Database indexing
- Caching systems (Redis, Memcached)
- Symbol tables in compilers
- Session management in web apps
- Counting word frequencies

**Best Practices:**
- Choose appropriate initial capacity
- Monitor load factor and resize when needed
- Use immutable keys when possible
- Consider thread-safety for concurrent access`,
					CodeExamples: `// Go: Hash table operations
m := make(map[int]int)

// Insert: O(1) average
m[1] = 100
m[2] = 200
m[3] = 300

// Lookup: O(1) average
val, exists := m[1]  // val=100, exists=true
val, exists = m[99]  // val=0, exists=false

// Delete: O(1) average
delete(m, 1)

// Iteration: O(n) - order not guaranteed
for key, value := range m {
    fmt.Printf("%d: %d\n", key, value)
}

// Frequency counting example
func countFreq(nums []int) map[int]int {
    freq := make(map[int]int)
    for _, num := range nums {
        freq[num]++  // O(1) per operation
    }
    return freq  // Total: O(n)
}

# Python: Dictionary operations
m = {}

# Insert: O(1) average
m[1] = 100
m[2] = 200
m[3] = 300

# Lookup: O(1) average
val = m.get(1)  # 100
val = m.get(99, 0)  # 0 (default if not found)

# Delete: O(1) average
del m[1]
# or: m.pop(1)

# Iteration: O(n) - order preserved in Python 3.7+
for key, value in m.items():
    print(f"{key}: {value}")

# Frequency counting example
def count_freq(nums):
    freq = {}
    for num in nums:
        freq[num] = freq.get(num, 0) + 1  # O(1) per operation
    return freq  # Total: O(n)`,
				},
				{
					Title: "Array Techniques",
					Content: `Master essential array manipulation techniques for efficient problem-solving.

**Prefix Sum:**
- Precompute cumulative sums for range queries
- Enables O(1) range sum queries after O(n) preprocessing
- Useful for: subarray sum problems, range queries

**Two-Pass Techniques:**
- First pass: collect information
- Second pass: use information to solve problem
- Example: Product except self, trapping rainwater

**In-Place Modifications:**
- Modify array without extra space
- Use array itself to store information
- Example: Move zeros, remove duplicates

**Array Rotation:**
- Rotate array left/right by k positions
- Techniques: Reverse method, cyclic replacement
- Time: O(n), Space: O(1)

**Common Patterns:**
- **Partitioning**: Separate elements based on condition (two pointers)
- **Cyclic Replacements**: Move elements in cycles
- **Reversal**: Reverse entire array or segments`,
					CodeExamples: `// Go: Prefix sum
func prefixSum(nums []int) []int {
    prefix := make([]int, len(nums)+1)
    for i := 0; i < len(nums); i++ {
        prefix[i+1] = prefix[i] + nums[i]
    }
    return prefix
}

// Range sum query: O(1)
func rangeSum(prefix []int, i, j int) int {
    return prefix[j+1] - prefix[i]
}

// Product except self (two passes)
func productExceptSelf(nums []int) []int {
    n := len(nums)
    result := make([]int, n)
    result[0] = 1
    // Left products
    for i := 1; i < n; i++ {
        result[i] = result[i-1] * nums[i-1]
    }
    // Right products
    right := 1
    for i := n - 1; i >= 0; i-- {
        result[i] *= right
        right *= nums[i]
    }
    return result
}

# Python: Prefix sum
def prefix_sum(nums):
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)
    return prefix

# Range sum query: O(1)
def range_sum(prefix, i, j):
    return prefix[j+1] - prefix[i]`,
				},
				{
					Title: "Hash Table Advanced Techniques",
					Content: `Advanced hash table techniques for complex problem-solving.

**Frequency Counting:**
- Count occurrences of elements
- Track character frequencies in strings
- Group elements by frequency

**Two-Sum Pattern:**
- Store complements (target - current) in map
- Check if complement exists before adding current
- Enables O(n) solution instead of O(n²)

**Grouping/Partitioning:**
- Group elements by some property
- Use property as key, list of elements as value
- Example: Group anagrams by sorted string

**Sliding Window with Hash Map:**
- Track window state using hash map
- Add/remove elements as window moves
- Efficiently check constraints

**Common Patterns:**
- **Character Frequency Array**: For limited alphabet (26 letters)
- **Index Mapping**: Store indices for quick lookup
- **Complement Tracking**: Store what we're looking for`,
					CodeExamples: `// Go: Group anagrams
func groupAnagrams(strs []string) [][]string {
    groups := make(map[[26]int][]string)
    for _, s := range strs {
        key := [26]int{}
        for _, c := range s {
            key[c-'a']++
        }
        groups[key] = append(groups[key], s)
    }
    result := make([][]string, 0, len(groups))
    for _, group := range groups {
        result = append(result, group)
    }
    return result
}

// Two-sum pattern
func twoSum(nums []int, target int) []int {
    m := make(map[int]int)
    for i, num := range nums {
        if idx, ok := m[target-num]; ok {
            return []int{idx, i}
        }
        m[num] = i
    }
    return nil
}

# Python: Group anagrams
def group_anagrams(strs):
    groups = {}
    for s in strs:
        key = [0] * 26
        for c in s:
            key[ord(c) - ord('a')] += 1
        key_tuple = tuple(key)
        if key_tuple not in groups:
            groups[key_tuple] = []
        groups[key_tuple].append(s)
    return list(groups.values())`,
				},
				{
					Title: "Sliding Window Maximum",
					Content: `Finding the maximum element in every window of size k is a classic problem that can be solved efficiently using a deque (double-ended queue).

**Problem Statement:**
Given an array and window size k, find the maximum element in every window of size k as it slides through the array.

**Naive Approach:**
- For each window, scan all k elements to find maximum
- Time: O(n×k), Space: O(1)
- Inefficient for large arrays

**Optimal Approach Using Deque:**
- Use deque to maintain indices of elements in decreasing order
- Front of deque always has index of maximum element in current window
- Time: O(n), Space: O(k)

**Algorithm:**
1. Process first k elements: remove indices of elements smaller than current
2. Add current index to back of deque
3. Front of deque is maximum of first window
4. For remaining elements:
   - Remove indices outside current window from front
   - Remove indices of elements smaller than current from back
   - Add current index to back
   - Front of deque is maximum of current window

**Key Insight:** We only need to keep track of elements that could be maximum in future windows. Elements smaller than current element can never be maximum, so we remove them.

**When to Use:**
- Finding maximum/minimum in sliding windows
- Range maximum queries
- Stock price problems with time windows`,
					CodeExamples: `// Go: Sliding window maximum using deque
func maxSlidingWindow(nums []int, k int) []int {
    if len(nums) == 0 {
        return []int{}
    }
    
    var deque []int  // Store indices
    result := make([]int, 0, len(nums)-k+1)
    
    // Process first window
    for i := 0; i < k; i++ {
        // Remove indices of elements smaller than current
        for len(deque) > 0 && nums[deque[len(deque)-1]] <= nums[i] {
            deque = deque[:len(deque)-1]
        }
        deque = append(deque, i)
    }
    result = append(result, nums[deque[0]])
    
    // Process remaining windows
    for i := k; i < len(nums); i++ {
        // Remove indices outside current window
        for len(deque) > 0 && deque[0] <= i-k {
            deque = deque[1:]
        }
        // Remove indices of elements smaller than current
        for len(deque) > 0 && nums[deque[len(deque)-1]] <= nums[i] {
            deque = deque[:len(deque)-1]
        }
        deque = append(deque, i)
        result = append(result, nums[deque[0]])
    }
    
    return result
}

# Python: Sliding window maximum using deque
from collections import deque

def max_sliding_window(nums, k):
    if not nums:
        return []
    
    dq = deque()  # Store indices
    result = []
    
    # Process first window
    for i in range(k):
        # Remove indices of elements smaller than current
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        dq.append(i)
    result.append(nums[dq[0]])
    
    # Process remaining windows
    for i in range(k, len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        # Remove indices of elements smaller than current
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        dq.append(i)
        result.append(nums[dq[0]])
    
    return result`,
				},
				{
					Title: "Prefix Sum Advanced",
					Content: `Prefix sums (also called cumulative sums) are a powerful technique for answering range queries efficiently. After O(n) preprocessing, you can answer range sum queries in O(1) time.

**Basic Prefix Sum:**
- Precompute cumulative sums: prefix[i] = sum of elements from index 0 to i-1
- Range sum query [i, j]: prefix[j+1] - prefix[i]
- Time: O(1) per query after O(n) preprocessing

**2D Prefix Sum:**
- Extend concept to 2D arrays (matrices)
- Precompute sum of all submatrices starting from (0,0)
- Answer sum of any rectangular region in O(1)
- Useful for: image processing, matrix range queries

**Prefix Sum Applications:**
1. **Range Sum Queries**: Sum of elements in range [i, j]
2. **Subarray Sum Problems**: Find subarrays with specific sum
3. **Equilibrium Point**: Find index where left sum equals right sum
4. **Maximum Subarray**: Kadane's algorithm uses prefix sum concept
5. **Difference Array**: Range updates in O(1), then reconstruct

**Difference Array Technique:**
- Instead of updating range [i, j] by adding value v to each element (O(n))
- Update difference array: diff[i] += v, diff[j+1] -= v (O(1))
- Reconstruct array using prefix sum of difference array
- Enables O(1) range updates, O(n) reconstruction

**When to Use:**
- Multiple range sum queries on static array
- Subarray problems with sum constraints
- Range update problems (use difference array)
- Matrix range queries`,
					CodeExamples: `// Go: 2D Prefix Sum
func build2DPrefixSum(matrix [][]int) [][]int {
    m, n := len(matrix), len(matrix[0])
    prefix := make([][]int, m+1)
    for i := range prefix {
        prefix[i] = make([]int, n+1)
    }
    
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            prefix[i][j] = matrix[i-1][j-1] + 
                          prefix[i-1][j] + 
                          prefix[i][j-1] - 
                          prefix[i-1][j-1]
        }
    }
    return prefix
}

// Query sum of rectangle from (r1, c1) to (r2, c2)
func query2D(prefix [][]int, r1, c1, r2, c2 int) int {
    return prefix[r2+1][c2+1] - 
           prefix[r1][c2+1] - 
           prefix[r2+1][c1] + 
           prefix[r1][c1]
}

// Difference Array for Range Updates
func rangeUpdate(nums []int, updates [][]int) []int {
    diff := make([]int, len(nums)+1)
    
    // Apply updates to difference array
    for _, update := range updates {
        start, end, val := update[0], update[1], update[2]
        diff[start] += val
        diff[end+1] -= val
    }
    
    // Reconstruct array using prefix sum
    result := make([]int, len(nums))
    sum := 0
    for i := 0; i < len(nums); i++ {
        sum += diff[i]
        result[i] = nums[i] + sum
    }
    return result
}

# Python: 2D Prefix Sum
def build_2d_prefix_sum(matrix):
    m, n = len(matrix), len(matrix[0])
    prefix = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            prefix[i][j] = (matrix[i-1][j-1] + 
                           prefix[i-1][j] + 
                           prefix[i][j-1] - 
                           prefix[i-1][j-1])
    return prefix

# Query sum of rectangle
def query_2d(prefix, r1, c1, r2, c2):
    return (prefix[r2+1][c2+1] - 
            prefix[r1][c2+1] - 
            prefix[r2+1][c1] + 
            prefix[r1][c1])

# Difference Array
def range_update(nums, updates):
    diff = [0] * (len(nums) + 1)
    
    for start, end, val in updates:
        diff[start] += val
        diff[end + 1] -= val
    
    result = []
    sum_val = 0
    for i in range(len(nums)):
        sum_val += diff[i]
        result.append(nums[i] + sum_val)
    return result`,
				},
				{
					Title: "Hash Table Collision Resolution",
					Content: `When two keys hash to the same index, we have a collision. How we handle collisions significantly impacts hash table performance.

**Collision Resolution Strategies:**

**1. Separate Chaining (Open Hashing):**
- Each bucket contains a linked list (or other data structure)
- Colliding elements are stored in the same bucket's list
- Simple to implement, handles many collisions well
- Extra memory overhead for pointers
- Worst case: O(n) if all keys hash to same bucket

**2. Open Addressing (Closed Hashing):**
- Store elements directly in the hash table array
- When collision occurs, find next available slot using probe sequence
- No extra memory for pointers
- Better cache performance (all data in one array)
- Clustering can be an issue

**Probe Sequences:**

**Linear Probing:**
- Check next slot: h(k, i) = (h(k) + i) mod m
- Simple but causes primary clustering
- Clusters form and grow, degrading performance

**Quadratic Probing:**
- Check slots at quadratic distances: h(k, i) = (h(k) + i²) mod m
- Reduces clustering but can fail to find empty slot even if table not full
- Requires table size to be prime and load factor < 0.5

**Double Hashing:**
- Use second hash function: h(k, i) = (h₁(k) + i×h₂(k)) mod m
- Best probe sequence, reduces clustering
- Requires careful choice of second hash function

**Load Factor:**
- α = n/m (number of elements / number of buckets)
- For chaining: α can be > 1, but performance degrades as α increases
- For open addressing: α must be < 1, typically kept < 0.75
- When load factor exceeds threshold, resize table (usually 2×) and rehash

**Rehashing:**
- When load factor too high, create larger table
- Recompute hash for all elements and insert into new table
- Expensive operation (O(n)) but amortized over many operations
- Typically doubles table size

**Choosing a Strategy:**
- **Chaining**: Better for unknown number of elements, handles clustering better
- **Open Addressing**: Better cache performance, less memory overhead
- **Double Hashing**: Best performance for open addressing

**Real-World Implementations:**
- Java HashMap: Uses chaining with tree conversion for long chains
- Python dict: Uses open addressing with pseudo-random probing
- Go map: Uses bucket chaining with overflow buckets`,
					CodeExamples: `// Go: Separate Chaining Implementation
type HashTable struct {
    buckets []*Node
    size    int
    capacity int
}

type Node struct {
    key   int
    value int
    next  *Node
}

func NewHashTable(capacity int) *HashTable {
    return &HashTable{
        buckets: make([]*Node, capacity),
        capacity: capacity,
    }
}

func (ht *HashTable) hash(key int) int {
    return key % ht.capacity
}

func (ht *HashTable) Put(key, value int) {
    index := ht.hash(key)
    node := ht.buckets[index]
    
    // Check if key already exists
    for node != nil {
        if node.key == key {
            node.value = value
            return
        }
        node = node.next
    }
    
    // Insert at head of chain
    newNode := &Node{key: key, value: value, next: ht.buckets[index]}
    ht.buckets[index] = newNode
    ht.size++
}

// Go: Linear Probing Implementation
type OpenHashTable struct {
    keys    []int
    values  []int
    deleted []bool
    size    int
    capacity int
}

func NewOpenHashTable(capacity int) *OpenHashTable {
    return &OpenHashTable{
        keys:    make([]int, capacity),
        values:  make([]int, capacity),
        deleted: make([]bool, capacity),
        capacity: capacity,
    }
}

func (ht *OpenHashTable) hash(key int) int {
    return key % ht.capacity
}

func (ht *OpenHashTable) Put(key, value int) {
    index := ht.hash(key)
    
    // Linear probing
    for ht.keys[index] != 0 && ht.keys[index] != key && !ht.deleted[index] {
        index = (index + 1) % ht.capacity
    }
    
    if ht.keys[index] == 0 || ht.deleted[index] {
        ht.size++
    }
    ht.keys[index] = key
    ht.values[index] = value
    ht.deleted[index] = false
}

# Python: Separate Chaining
class HashTable:
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.buckets = [[] for _ in range(capacity)]
        self.size = 0
    
    def _hash(self, key):
        return hash(key) % self.capacity
    
    def put(self, key, value):
        index = self._hash(key)
        bucket = self.buckets[index]
        
        # Check if key exists
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        # Add new key-value pair
        bucket.append((key, value))
        self.size += 1

# Python: Double Hashing
class OpenHashTable:
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.keys = [None] * capacity
        self.values = [None] * capacity
        self.size = 0
    
    def _hash1(self, key):
        return hash(key) % self.capacity
    
    def _hash2(self, key):
        return 7 - (hash(key) % 7)  # Must be coprime with capacity
    
    def put(self, key, value):
        index = self._hash1(key)
        step = self._hash2(key)
        
        # Double hashing probe
        while self.keys[index] is not None and self.keys[index] != key:
            index = (index + step) % self.capacity
        
        if self.keys[index] is None:
            self.size += 1
        self.keys[index] = key
        self.values[index] = value`,
				},
			},
			ProblemIDs: []int{1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
		},
		{
			ID:          2,
			Title:       "Two Pointers",
			Description: "Master the two-pointer technique for solving array and string problems efficiently.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "Two Pointers Technique",
					Content: `The two-pointer technique uses two pointers moving through a data structure to solve problems efficiently. It's one of the most powerful optimization techniques, often reducing O(n²) brute force solutions to O(n).

**Common Patterns:**

1. **Opposite Ends (Converging Pointers)**:
   - Start with pointers at both ends
   - Move them toward each other based on conditions
   - Best for: sorted arrays, palindromes, finding pairs

2. **Same Direction (Fast/Slow Pointers)**:
   - Both start at beginning
   - Move at different speeds
   - Best for: removing duplicates, cycle detection, partitioning

3. **Sliding Window**:
   - Maintain a window of elements
   - Expand and contract based on conditions
   - Best for: subarray/substring problems

**When to Use:**
- **Sorted arrays**: Can eliminate half the search space
- **Palindrome checking**: Compare from both ends
- **Finding pairs/triplets**: Avoid nested loops
- **Removing duplicates**: In-place modification
- **Merging sorted arrays**: Combine efficiently
- **Cycle detection**: Floyd's algorithm

**Key Insight:** Instead of checking all pairs O(n²), use pointers to eliminate possibilities systematically, achieving O(n).

**Common Mistakes:**
- Moving pointers incorrectly (infinite loops)
- Off-by-one errors with pointer positions
- Not handling edge cases (empty arrays, single element)
- Forgetting to update pointers in all branches`,
					CodeExamples: `// Go: Two pointers from opposite ends
func twoSumSorted(nums []int, target int) []int {
    left, right := 0, len(nums)-1
    for left < right {
        sum := nums[left] + nums[right]
        if sum == target {
            return []int{left, right}
        } else if sum < target {
            left++
        } else {
            right--
        }
    }
    return nil
}

# Python: Same direction (fast/slow)
def removeDuplicates(nums):
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1`,
				},
				{
					Title: "Fast and Slow Pointers",
					Content: `The fast/slow pointer technique uses two pointers moving at different speeds to solve problems efficiently.

**Common Applications:**

1. **Cycle Detection** (Floyd's Algorithm):
   - Fast pointer moves 2 steps, slow moves 1 step
   - If cycle exists, they will meet
   - Time: O(n), Space: O(1)

2. **Finding Middle Element**:
   - Fast pointer reaches end when slow is at middle
   - Useful for: splitting list, palindrome checking

3. **Finding kth Element from End**:
   - Move fast pointer k steps ahead
   - Then move both at same speed
   - When fast reaches end, slow is at kth from end

**Why It Works:**
- Fast pointer covers distance twice as fast
- When fast completes 2n steps, slow completes n steps
- They meet at specific positions based on problem

**Pattern:**
- Initialize both pointers at head
- Move fast pointer faster (usually 2x speed)
- Check conditions when pointers meet or fast reaches end`,
					CodeExamples: `// Go: Cycle detection
func hasCycle(head *ListNode) bool {
    if head == nil {
        return false
    }
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        if slow == fast {
            return true  // Cycle detected
        }
    }
    return false
}

// Find middle element
func findMiddle(head *ListNode) *ListNode {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    return slow  // Middle element
}

# Python: Cycle detection
def has_cycle(head):
    if not head:
        return False
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False`,
				},
				{
					Title: "Three Sum Pattern",
					Content: `The three sum problem extends the two sum pattern to find three elements that sum to a target value. This is a classic problem that demonstrates advanced two-pointer techniques.

**Problem Statement:**
Given an array of integers, find all unique triplets that sum to zero (or any target value).

**Approach:**
1. Sort the array (enables two-pointer technique)
2. Fix first element, then use two pointers for remaining two
3. Move pointers based on current sum compared to target
4. Skip duplicates to avoid duplicate triplets

**Algorithm:**
1. Sort array: O(n log n)
2. For each element at index i:
   - Set left = i+1, right = n-1
   - While left < right:
     - Calculate sum = nums[i] + nums[left] + nums[right]
     - If sum == target: add triplet, move both pointers, skip duplicates
     - If sum < target: move left pointer right (need larger sum)
     - If sum > target: move right pointer left (need smaller sum)
3. Skip duplicates for first element

**Time Complexity:** O(n²) - outer loop O(n), two pointers O(n)
**Space Complexity:** O(1) excluding output array

**Key Insights:**
- Sorting enables two-pointer technique
- Skipping duplicates is crucial for correctness
- Can be extended to k-sum problems using recursion

**Variations:**
- Three Sum Closest: Find triplet with sum closest to target
- Three Sum Smaller: Count triplets with sum less than target
- Four Sum: Extend to four elements`,
					CodeExamples: `// Go: Three Sum
func threeSum(nums []int) [][]int {
    sort.Ints(nums)
    result := [][]int{}
    n := len(nums)
    
    for i := 0; i < n-2; i++ {
        // Skip duplicates for first element
        if i > 0 && nums[i] == nums[i-1] {
            continue
        }
        
        left, right := i+1, n-1
        target := -nums[i]  // We want nums[left] + nums[right] = -nums[i]
        
        for left < right {
            sum := nums[left] + nums[right]
            
            if sum == target {
                result = append(result, []int{nums[i], nums[left], nums[right]})
                
                // Skip duplicates
                for left < right && nums[left] == nums[left+1] {
                    left++
                }
                for left < right && nums[right] == nums[right-1] {
                    right--
                }
                left++
                right--
            } else if sum < target {
                left++  // Need larger sum
            } else {
                right--  // Need smaller sum
            }
        }
    }
    
    return result
}

# Python: Three Sum
def three_sum(nums):
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, n - 1
        target = -nums[i]
        
        while left < right:
            current_sum = nums[left] + nums[right]
            
            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    
    return result`,
				},
				{
					Title: "Partitioning Arrays",
					Content: `Partitioning arrays involves rearranging elements so that elements satisfying a condition come before elements that don't. This is fundamental to algorithms like quicksort and many array problems.

**Partition Patterns:**

**1. Two-Way Partition:**
- Partition into two groups based on condition
- Example: Move all zeros to end, separate even/odd numbers
- Use two pointers: one for elements satisfying condition, one for scanning

**2. Three-Way Partition (Dutch National Flag):**
- Partition into three groups: < pivot, == pivot, > pivot
- Useful for: quicksort optimization, sorting with many duplicates
- Uses three pointers: left (less), current (scanning), right (greater)

**3. Lomuto Partition:**
- Classic partition scheme for quicksort
- Maintains invariant: elements before pivot are < pivot
- Simpler but less efficient than Hoare partition

**4. Hoare Partition:**
- More efficient partition scheme
- Uses two pointers converging from both ends
- Fewer swaps than Lomuto partition

**Common Partition Problems:**
- Move zeros to end
- Separate even and odd numbers
- Partition around pivot (quicksort)
- Sort colors (Dutch flag problem)
- Rearrange array by sign (positive/negative)

**Key Techniques:**
- Two pointers: one for "write" position, one for "read" position
- Swap elements to maintain partition invariant
- Handle edge cases: empty arrays, all elements same, etc.`,
					CodeExamples: `// Go: Two-way partition - Move zeros to end
func moveZeros(nums []int) {
    writePos := 0
    
    // Move all non-zero elements to front
    for i := 0; i < len(nums); i++ {
        if nums[i] != 0 {
            nums[writePos], nums[i] = nums[i], nums[writePos]
            writePos++
        }
    }
}

// Go: Three-way partition (Dutch National Flag)
func sortColors(nums []int) {
    left, current, right := 0, 0, len(nums)-1
    
    for current <= right {
        if nums[current] == 0 {
            nums[left], nums[current] = nums[current], nums[left]
            left++
            current++
        } else if nums[current] == 2 {
            nums[right], nums[current] = nums[current], nums[right]
            right--
            // Don't increment current - need to check swapped element
        } else {
            current++
        }
    }
}

// Go: Lomuto Partition (for quicksort)
func lomutoPartition(nums []int, low, high int) int {
    pivot := nums[high]
    i := low  // Index of smaller element
    
    for j := low; j < high; j++ {
        if nums[j] <= pivot {
            nums[i], nums[j] = nums[j], nums[i]
            i++
        }
    }
    nums[i], nums[high] = nums[high], nums[i]
    return i
}

# Python: Two-way partition
def move_zeros(nums):
    write_pos = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[write_pos], nums[i] = nums[i], nums[write_pos]
            write_pos += 1

# Python: Three-way partition
def sort_colors(nums):
    left = current = 0
    right = len(nums) - 1
    
    while current <= right:
        if nums[current] == 0:
            nums[left], nums[current] = nums[current], nums[left]
            left += 1
            current += 1
        elif nums[current] == 2:
            nums[right], nums[current] = nums[current], nums[right]
            right -= 1
        else:
            current += 1`,
				},
				{
					Title: "Merge Sorted Arrays",
					Content: `Merging sorted arrays is a fundamental operation in many algorithms, especially merge sort and when combining data from multiple sorted sources.

**Problem Types:**

**1. Merge Two Sorted Arrays:**
- Given two sorted arrays, merge into one sorted array
- Can merge in-place if one array has enough space
- Time: O(m + n), Space: O(1) if in-place, O(m + n) otherwise

**2. Merge K Sorted Arrays:**
- Merge k sorted arrays into one sorted array
- Approaches: divide and conquer, heap/priority queue
- Time: O(n log k) where n is total elements

**3. Merge Sorted Lists:**
- Merge two sorted linked lists
- Simpler than arrays (no space issues)
- Time: O(m + n), Space: O(1)

**Algorithm for Two Arrays:**
1. Use two pointers, one for each array
2. Compare elements at both pointers
3. Add smaller element to result, advance its pointer
4. Continue until one array is exhausted
5. Add remaining elements from other array

**In-Place Merging:**
- If one array has enough space at end
- Start from end (largest elements) to avoid overwriting
- Fill from right to left

**Key Insights:**
- Always compare current elements, not next elements
- Handle remaining elements after main loop
- For in-place: work backwards to avoid overwriting
- For k arrays: use heap to always get minimum element`,
					CodeExamples: `// Go: Merge two sorted arrays
func merge(nums1 []int, m int, nums2 []int, n int) {
    // Merge in-place from end
    i, j, k := m-1, n-1, m+n-1
    
    for i >= 0 && j >= 0 {
        if nums1[i] > nums2[j] {
            nums1[k] = nums1[i]
            i--
        } else {
            nums1[k] = nums2[j]
            j--
        }
        k--
    }
    
    // Copy remaining elements from nums2
    for j >= 0 {
        nums1[k] = nums2[j]
        j--
        k--
    }
}

// Go: Merge k sorted arrays using heap
import "container/heap"

type Item struct {
    val   int
    arrIdx int
    elemIdx int
}

type MinHeap []Item

func (h MinHeap) Len() int { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].val < h[j].val }
func (h MinHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x interface{}) { *h = append(*h, x.(Item)) }
func (h *MinHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0:n-1]
    return x
}

func mergeKSortedArrays(arrays [][]int) []int {
    h := &MinHeap{}
    heap.Init(h)
    result := []int{}
    
    // Add first element of each array to heap
    for i, arr := range arrays {
        if len(arr) > 0 {
            heap.Push(h, Item{val: arr[0], arrIdx: i, elemIdx: 0})
        }
    }
    
    // Extract min and add next element from same array
    for h.Len() > 0 {
        item := heap.Pop(h).(Item)
        result = append(result, item.val)
        
        if item.elemIdx+1 < len(arrays[item.arrIdx]) {
            heap.Push(h, Item{
                val: arrays[item.arrIdx][item.elemIdx+1],
                arrIdx: item.arrIdx,
                elemIdx: item.elemIdx + 1,
            })
        }
    }
    
    return result
}

# Python: Merge two sorted arrays
def merge(nums1, m, nums2, n):
    i, j, k = m - 1, n - 1, m + n - 1
    
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
    
    # Copy remaining from nums2
    while j >= 0:
        nums1[k] = nums2[j]
        j -= 1
        k -= 1

# Python: Merge k sorted arrays using heap
import heapq

def merge_k_sorted_arrays(arrays):
    heap = []
    result = []
    
    # Add first element of each array
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))
    
    # Extract min and add next
    while heap:
        val, arr_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        if elem_idx + 1 < len(arrays[arr_idx]):
            heapq.heappush(heap, (arrays[arr_idx][elem_idx + 1], arr_idx, elem_idx + 1))
    
    return result`,
				},
			},
			ProblemIDs: []int{2, 3, 4, 5, 21, 22, 23, 24, 25},
		},
		{
			ID:          3,
			Title:       "Sliding Window",
			Description: "Learn to solve problems using the sliding window technique for optimal subarray/substring solutions.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "Sliding Window Technique",
					Content: `The sliding window technique maintains a window of elements and slides it across the array/string to find optimal solutions. It's essential for substring/subarray problems.

**Types:**

1. **Fixed Window**:
   - Window size is constant (e.g., always k elements)
   - Slide window one position at a time
   - Example: Maximum sum of subarray of size k

2. **Variable Window**:
   - Window size changes based on conditions
   - Expand when condition not met, contract when met
   - Example: Longest substring with at most k distinct characters

**Window Maintenance:**
- **Expand**: Move right pointer forward
- **Contract**: Move left pointer forward
- **Track State**: Use hash map or variables to track window properties

**When to Use:**
- Subarray/substring problems
- Finding maximum/minimum in a window
- Problems asking for "longest" or "shortest" subarray
- Problems with constraints (sum, unique characters, frequency)
- String pattern matching

**Time Complexity:** O(n) - each element visited at most twice (once by left pointer, once by right pointer)

**Template Pattern:**
1. Initialize left = 0, right = 0
2. Expand window (right++)
3. While window is invalid, contract (left++)
4. Update answer when window is valid
5. Repeat until right reaches end

**Common Variations:**
- **At most k**: Use while loop to contract
- **Exactly k**: Use if statement to check
- **At least k**: Different contraction logic`,
					CodeExamples: `// Go: Fixed window - maximum sum of subarray of size k
func maxSumSubarray(nums []int, k int) int {
    windowSum := 0
    for i := 0; i < k; i++ {
        windowSum += nums[i]
    }
    maxSum := windowSum
    
    for i := k; i < len(nums); i++ {
        windowSum = windowSum - nums[i-k] + nums[i]
        if windowSum > maxSum {
            maxSum = windowSum
        }
    }
    return maxSum
}

# Python: Variable window - longest substring without repeating characters
def lengthOfLongestSubstring(s):
    charSet = set()
    left = 0
    maxLen = 0
    for right in range(len(s)):
        while s[right] in charSet:
            charSet.remove(s[left])
            left += 1
        charSet.add(s[right])
        maxLen = max(maxLen, right - left + 1)
    return maxLen`,
				},
				{
					Title: "Variable Window Patterns",
					Content: `Variable window sliding window problems require dynamically adjusting window size based on conditions. These are more complex than fixed window problems.

**Common Patterns:**

**1. Longest Substring/Subarray:**
- Expand window until condition violated
- Contract window until condition satisfied again
- Track maximum valid window size
- Example: Longest substring with at most k distinct characters

**2. Shortest Substring/Subarray:**
- Expand window until condition satisfied
- Contract window while condition still satisfied
- Track minimum valid window size
- Example: Minimum window substring

**3. Count Subarrays:**
- For each right position, count valid windows ending at right
- Use "at most k" trick: count(at most k) - count(at most k-1) = count(exactly k)
- Example: Count subarrays with exactly k distinct elements

**Key Techniques:**
- Use hash map to track window state (character frequencies, element counts)
- Maintain invariant: window is always valid when checking
- Update answer at appropriate points (during expansion or contraction)

**When to Contract:**
- **At most k**: Contract when count > k
- **At least k**: Contract when count < k (but this is less common)
- **Exactly k**: Use "at most" trick

**Optimization Tips:**
- Use array instead of map for limited alphabet (26 letters)
- Track window state incrementally (add/remove one element at a time)
- Avoid recalculating window properties from scratch`,
					CodeExamples: `// Go: Longest substring with at most k distinct characters
func lengthOfLongestSubstringKDistinct(s string, k int) int {
    if k == 0 {
        return 0
    }
    
    freq := make(map[byte]int)
    left, maxLen := 0, 0
    
    for right := 0; right < len(s); right++ {
        freq[s[right]]++
        
        // Contract window if more than k distinct characters
        for len(freq) > k {
            freq[s[left]]--
            if freq[s[left]] == 0 {
                delete(freq, s[left])
            }
            left++
        }
        
        maxLen = max(maxLen, right-left+1)
    }
    
    return maxLen
}

// Go: Count subarrays with exactly k distinct elements
func subarraysWithKDistinct(nums []int, k int) int {
    // Count(at most k) - Count(at most k-1) = Count(exactly k)
    return atMostKDistinct(nums, k) - atMostKDistinct(nums, k-1)
}

func atMostKDistinct(nums []int, k int) int {
    freq := make(map[int]int)
    left, count := 0, 0
    
    for right := 0; right < len(nums); right++ {
        freq[nums[right]]++
        
        for len(freq) > k {
            freq[nums[left]]--
            if freq[nums[left]] == 0 {
                delete(freq, nums[left])
            }
            left++
        }
        
        // All subarrays ending at right with at most k distinct
        count += right - left + 1
    }
    
    return count
}

# Python: Longest substring with at most k distinct
def length_of_longest_substring_k_distinct(s, k):
    if k == 0:
        return 0
    
    freq = {}
    left = max_len = 0
    
    for right in range(len(s)):
        freq[s[right]] = freq.get(s[right], 0) + 1
        
        while len(freq) > k:
            freq[s[left]] -= 1
            if freq[s[left]] == 0:
                del freq[s[left]]
            left += 1
        
        max_len = max(max_len, right - left + 1)
    
    return max_len`,
				},
				{
					Title: "Minimum Window Substring",
					Content: `The minimum window substring problem is a classic sliding window problem that finds the smallest substring containing all characters of a pattern.

**Problem Statement:**
Given a string S and a string T, find the minimum window in S that contains all characters of T (including duplicates).

**Approach:**
1. Use two pointers (left and right) to maintain a window
2. Expand window by moving right pointer
3. When window contains all required characters, try to contract from left
4. Track minimum valid window

**Algorithm:**
1. Count frequency of each character in pattern T
2. Use two pointers: left = 0, right = 0
3. Expand window (right++):
   - Add character at right to window
   - Update frequency count
   - Check if window is valid (contains all characters)
4. When window is valid:
   - Update minimum window if current is smaller
   - Try to contract from left (left++)
   - Remove characters until window becomes invalid
5. Repeat until right reaches end

**Key Insights:**
- Use hash map to track required characters and their frequencies
- Track how many characters are "satisfied" (have required frequency)
- Window is valid when all characters are satisfied
- Contract greedily to find minimum window

**Time Complexity:** O(|S| + |T|) - each character visited at most twice
**Space Complexity:** O(|S| + |T|) - for frequency maps

**Variations:**
- Minimum window subsequence (characters need not be contiguous)
- Longest substring with at most k distinct characters
- Substring with concatenation of all words`,
					CodeExamples: `// Go: Minimum window substring
func minWindow(s string, t string) string {
    if len(s) < len(t) {
        return ""
    }
    
    // Count frequency of characters in pattern
    need := make(map[byte]int)
    for i := 0; i < len(t); i++ {
        need[t[i]]++
    }
    
    window := make(map[byte]int)
    left, right := 0, 0
    valid := 0  // Number of characters with required frequency
    start, minLen := 0, len(s)+1
    
    for right < len(s) {
        c := s[right]
        right++
        
        // Add character to window
        if need[c] > 0 {
            window[c]++
            if window[c] == need[c] {
                valid++
            }
        }
        
        // Try to contract window
        for valid == len(need) {
            // Update minimum window
            if right-left < minLen {
                start = left
                minLen = right - left
            }
            
            d := s[left]
            left++
            
            // Remove character from window
            if need[d] > 0 {
                if window[d] == need[d] {
                    valid--
                }
                window[d]--
            }
        }
    }
    
    if minLen == len(s)+1 {
        return ""
    }
    return s[start : start+minLen]
}

# Python: Minimum window substring
def min_window(s, t):
    if len(s) < len(t):
        return ""
    
    # Count frequency of characters in pattern
    need = {}
    for char in t:
        need[char] = need.get(char, 0) + 1
    
    window = {}
    left = right = 0
    valid = 0  # Number of satisfied characters
    start = 0
    min_len = float('inf')
    
    while right < len(s):
        c = s[right]
        right += 1
        
        # Add to window
        if c in need:
            window[c] = window.get(c, 0) + 1
            if window[c] == need[c]:
                valid += 1
        
        # Try to contract
        while valid == len(need):
            # Update minimum
            if right - left < min_len:
                start = left
                min_len = right - left
            
            d = s[left]
            left += 1
            
            # Remove from window
            if d in need:
                if window[d] == need[d]:
                    valid -= 1
                window[d] -= 1
    
    return "" if min_len == float('inf') else s[start:start+min_len]`,
				},
				{
					Title: "Longest Substring with K Distinct",
					Content: `Finding the longest substring with exactly k distinct characters is a fundamental sliding window problem that demonstrates the "at most k" technique.

**Problem Statement:**
Find the length of the longest substring that contains exactly k distinct characters.

**Key Insight - "At Most K" Trick:**
Instead of finding "exactly k", we can use:
- Longest substring with at most k distinct characters
- Longest substring with at most (k-1) distinct characters
- The difference gives us substrings with exactly k

However, for "exactly k", we can also directly track:
- Expand window while distinct count ≤ k
- Contract window when distinct count > k
- Update answer when distinct count == k

**Algorithm:**
1. Use hash map to track character frequencies in window
2. Expand window (right++):
   - Add character, update frequency
   - If new character: increment distinct count
3. When distinct count > k:
   - Contract window (left++)
   - Remove characters until distinct count ≤ k
4. When distinct count == k:
   - Update maximum length
5. Continue until right reaches end

**Time Complexity:** O(n) - each character visited at most twice
**Space Complexity:** O(k) - hash map stores at most k+1 characters

**Common Mistakes:**
- Not handling k = 0 case
- Updating answer at wrong time
- Not properly tracking distinct count
- Forgetting to remove characters when frequency becomes 0`,
					CodeExamples: `// Go: Longest substring with exactly k distinct
func longestSubstringKDistinct(s string, k int) int {
    if k == 0 {
        return 0
    }
    
    freq := make(map[byte]int)
    left, distinct, maxLen := 0, 0, 0
    
    for right := 0; right < len(s); right++ {
        // Add character to window
        if freq[s[right]] == 0 {
            distinct++
        }
        freq[s[right]]++
        
        // Contract window if too many distinct characters
        for distinct > k {
            freq[s[left]]--
            if freq[s[left]] == 0 {
                distinct--
            }
            left++
        }
        
        // Update answer when exactly k distinct
        if distinct == k {
            maxLen = max(maxLen, right-left+1)
        }
    }
    
    return maxLen
}

// Alternative: Using "at most k" trick
func longestSubstringAtMostK(s string, k int) int {
    freq := make(map[byte]int)
    left, distinct, maxLen := 0, 0, 0
    
    for right := 0; right < len(s); right++ {
        if freq[s[right]] == 0 {
            distinct++
        }
        freq[s[right]]++
        
        for distinct > k {
            freq[s[left]]--
            if freq[s[left]] == 0 {
                distinct--
            }
            left++
        }
        
        maxLen = max(maxLen, right-left+1)
    }
    
    return maxLen
}

// Exactly k = at most k - at most (k-1)
func longestSubstringExactlyK(s string, k int) int {
    if k == 0 {
        return 0
    }
    return longestSubstringAtMostK(s, k) - longestSubstringAtMostK(s, k-1)
}

# Python: Longest substring with exactly k distinct
def longest_substring_k_distinct(s, k):
    if k == 0:
        return 0
    
    freq = {}
    left = distinct = max_len = 0
    
    for right in range(len(s)):
        # Add character
        if s[right] not in freq or freq[s[right]] == 0:
            distinct += 1
        freq[s[right]] = freq.get(s[right], 0) + 1
        
        # Contract if too many distinct
        while distinct > k:
            freq[s[left]] -= 1
            if freq[s[left]] == 0:
                distinct -= 1
            left += 1
        
        # Update if exactly k
        if distinct == k:
            max_len = max(max_len, right - left + 1)
    
    return max_len`,
				},
			},
			ProblemIDs: []int{26, 27, 28, 29, 30},
		},
		{
			ID:          4,
			Title:       "Stack & Queue",
			Description: "Understand stack and queue data structures and their applications in problem-solving.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Stack",
					Content: `A stack is a LIFO (Last In, First Out) data structure. Think of it like a stack of plates - you add to the top and remove from the top.

**Core Operations:**
- **Push**: Add element to top - O(1)
- **Pop**: Remove element from top - O(1)
- **Peek/Top**: View top element without removing - O(1)
- **IsEmpty**: Check if stack is empty - O(1)
- **Size**: Get number of elements - O(1)

**Implementation:**
- **Array-based**: Fixed or dynamic array, push/pop at end
- **Linked list-based**: Nodes with pointers, push/pop at head

**Common Use Cases:**

1. **Expression Evaluation**:
   - Infix to postfix conversion
   - Postfix expression evaluation
   - Handling operator precedence

2. **Parentheses Matching**:
   - Valid parentheses problem
   - Nested structure validation
   - HTML/XML tag matching

3. **Function Call Stack**:
   - Recursion implementation
   - Call stack management
   - Backtracking algorithms

4. **Undo/Redo Operations**:
   - Text editor undo
   - Browser back button
   - Command pattern implementation

5. **Monotonic Stack**:
   - Next greater/smaller element
   - Largest rectangle in histogram
   - Stock span problem

**Monotonic Stack Pattern:**
Maintain stack in sorted order (increasing or decreasing) to efficiently find next greater/smaller elements.

**Time Complexity:** All operations O(1) - constant time

**Space Complexity:** O(n) - stores n elements

**Real-World Applications:**
- Compiler syntax checking
- Browser navigation
- Text editor undo/redo
- Expression parsers
- Graph algorithms (DFS)`,
					CodeExamples: `// Go: Using slice as stack
stack := []int{}
stack = append(stack, 1)  // Push
top := stack[len(stack)-1]  // Peek
stack = stack[:len(stack)-1]  // Pop

# Python: Using list as stack
stack = []
stack.append(1)  # Push
top = stack[-1]  # Peek
stack.pop()  # Pop`,
				},
				{
					Title: "Queue",
					Content: `A queue is a FIFO (First In, First Out) data structure. Think of it like a line at a store - first person in is first person out.

**Core Operations:**
- **Enqueue**: Add element to rear (back) - O(1)
- **Dequeue**: Remove element from front - O(1)
- **Front/Peek**: View front element without removing - O(1)
- **IsEmpty**: Check if queue is empty - O(1)
- **Size**: Get number of elements - O(1)

**Implementation:**
- **Array-based**: Circular array to avoid shifting
- **Linked list-based**: Nodes with head and tail pointers
- **Deque (Double-ended queue)**: Can add/remove from both ends

**Common Use Cases:**

1. **BFS (Breadth-First Search)**:
   - Level-order tree traversal
   - Shortest path in unweighted graphs
   - Finding minimum steps

2. **Task Scheduling**:
   - Process scheduling in OS
   - Print queue management
   - Request handling in web servers

3. **Sliding Window Maximum**:
   - Using deque to maintain window
   - Efficiently find max in window

4. **Level-order Processing**:
   - Process tree nodes level by level
   - Generate binary numbers
   - Right side view of tree

**Queue Variants:**

1. **Priority Queue**:
   - Elements dequeued by priority
   - Implemented with heap
   - Used in Dijkstra's algorithm

2. **Circular Queue**:
   - Fixed size, wraps around
   - Efficient memory usage
   - Used in buffering

3. **Deque (Double-ended Queue)**:
   - Add/remove from both ends
   - More flexible than regular queue
   - Used in sliding window problems

**Time Complexity:** All operations O(1) with proper implementation

**Space Complexity:** O(n) - stores n elements

**Real-World Applications:**
- Operating system process scheduling
- Network packet routing
- Print queue management
- Web server request handling
- BFS graph traversal`,
					CodeExamples: `// Go: Using slice as queue
queue := []int{}
queue = append(queue, 1)  // Enqueue
front := queue[0]  // Front
queue = queue[1:]  // Dequeue

# Python: Using collections.deque
from collections import deque
queue = deque()
queue.append(1)  # Enqueue
front = queue[0]  # Front
queue.popleft()  # Dequeue`,
				},
				{
					Title: "Monotonic Stack Advanced",
					Content: `A monotonic stack maintains elements in sorted order (either increasing or decreasing). This enables efficient solutions to problems involving "next greater/smaller element" queries.

**Monotonic Stack Properties:**
- Elements are ordered (monotonically increasing or decreasing)
- When new element added, remove elements that violate order
- Enables O(n) solutions to problems that seem O(n²)

**Types:**

**1. Monotonically Increasing Stack:**
- Elements increase from bottom to top
- Used for: next smaller element, previous smaller element
- Remove elements larger than current before pushing

**2. Monotonically Decreasing Stack:**
- Elements decrease from bottom to top
- Used for: next greater element, previous greater element
- Remove elements smaller than current before pushing

**Common Problems:**
- Next Greater Element: Find first element to right that's greater
- Previous Greater Element: Find first element to left that's greater
- Largest Rectangle in Histogram: Find largest rectangle area
- Trapping Rain Water: Calculate trapped water
- Stock Span Problem: Calculate span for each day

**Key Pattern:**
1. Iterate through array
2. While stack not empty and current element violates monotonic property:
   - Pop from stack and process
   - This element is the "next greater/smaller" for popped element
3. Push current element to stack
4. After iteration, remaining elements have no next greater/smaller

**Time Complexity:** O(n) - each element pushed and popped at most once
**Space Complexity:** O(n) - stack stores at most n elements`,
					CodeExamples: `// Go: Next Greater Element
func nextGreaterElement(nums []int) []int {
    result := make([]int, len(nums))
    stack := []int{}  // Store indices
    
    for i := 0; i < len(nums); i++ {
        // While current element is greater than stack top
        for len(stack) > 0 && nums[stack[len(stack)-1]] < nums[i] {
            idx := stack[len(stack)-1]
            stack = stack[:len(stack)-1]
            result[idx] = nums[i]  // Next greater element
        }
        stack = append(stack, i)
    }
    
    // Remaining elements have no next greater
    for len(stack) > 0 {
        idx := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        result[idx] = -1
    }
    
    return result
}

// Go: Largest Rectangle in Histogram
func largestRectangleArea(heights []int) int {
    stack := []int{}  // Store indices
    maxArea := 0
    
    for i := 0; i <= len(heights); i++ {
        var h int
        if i < len(heights) {
            h = heights[i]
        } else {
            h = 0  // Process remaining bars
        }
        
        // While current bar is shorter than stack top
        for len(stack) > 0 && heights[stack[len(stack)-1]] > h {
            height := heights[stack[len(stack)-1]]
            stack = stack[:len(stack)-1]
            
            width := i
            if len(stack) > 0 {
                width = i - stack[len(stack)-1] - 1
            }
            
            maxArea = max(maxArea, height*width)
        }
        stack = append(stack, i)
    }
    
    return maxArea
}

# Python: Next Greater Element
def next_greater_element(nums):
    result = [-1] * len(nums)
    stack = []  # Store indices
    
    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    
    return result

# Python: Largest Rectangle in Histogram
def largest_rectangle_area(heights):
    stack = []
    max_area = 0
    
    for i in range(len(heights) + 1):
        h = heights[i] if i < len(heights) else 0
        
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        stack.append(i)
    
    return max_area`,
				},
				{
					Title: "Queue-based BFS",
					Content: `Breadth-First Search (BFS) uses a queue to explore graphs level by level. It's fundamental for finding shortest paths in unweighted graphs and level-order traversals.

**BFS Algorithm:**
1. Start from source node, mark as visited
2. Add source to queue
3. While queue not empty:
   - Dequeue node (process it)
   - Add all unvisited neighbors to queue
   - Mark neighbors as visited
4. Continue until queue is empty

**Key Properties:**
- **Level Order**: Visits nodes at distance 0, then 1, then 2, etc.
- **Shortest Path**: Finds shortest path in unweighted graphs (minimum edges)
- **Complete**: Visits all reachable nodes from source
- **Queue-based**: Uses FIFO queue to maintain level order

**BFS Applications:**

**1. Shortest Path (Unweighted Graphs):**
- Minimum steps to reach target
- Shortest path length
- Level of each node from source

**2. Level-order Tree Traversal:**
- Process tree nodes level by level
- Print tree by levels
- Find nodes at specific depth

**3. Connected Components:**
- Find all nodes reachable from source
- Count connected components
- Check if graph is connected

**4. Bipartite Checking:**
- Color nodes alternately (BFS coloring)
- Check if graph is bipartite
- Detect odd-length cycles

**5. Word Ladder:**
- Transform one word to another
- Each step changes one character
- Find minimum transformations

**Time Complexity:** O(V + E) - visit each vertex and edge once
**Space Complexity:** O(V) - queue and visited array

**Implementation Tips:**
- Use queue to maintain level order
- Mark nodes as visited when adding to queue (not when processing)
- Track distance/level by processing level by level
- Use separate data structure to reconstruct path if needed`,
					CodeExamples: `// Go: BFS Shortest Path
func bfsShortestPath(graph [][]int, start, target int) int {
    if start == target {
        return 0
    }
    
    visited := make([]bool, len(graph))
    queue := []int{start}
    visited[start] = true
    distance := 0
    
    for len(queue) > 0 {
        levelSize := len(queue)
        distance++
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            
            for _, neighbor := range graph[node] {
                if neighbor == target {
                    return distance
                }
                if !visited[neighbor] {
                    visited[neighbor] = true
                    queue = append(queue, neighbor)
                }
            }
        }
    }
    
    return -1  // Not reachable
}

// Go: Level-order Tree Traversal
func levelOrder(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{}
    }
    
    result := [][]int{}
    queue := []*TreeNode{root}
    
    for len(queue) > 0 {
        levelSize := len(queue)
        level := []int{}
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            level = append(level, node.Val)
            
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
        result = append(result, level)
    }
    
    return result
}

# Python: BFS Shortest Path
from collections import deque

def bfs_shortest_path(graph, start, target):
    if start == target:
        return 0
    
    visited = [False] * len(graph)
    queue = deque([start])
    visited[start] = True
    distance = 0
    
    while queue:
        level_size = len(queue)
        distance += 1
        
        for _ in range(level_size):
            node = queue.popleft()
            
            for neighbor in graph[node]:
                if neighbor == target:
                    return distance
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
    
    return -1

# Python: Level-order Traversal
def level_order(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result`,
				},
				{
					Title: "Deque Applications",
					Content: `A deque (double-ended queue) allows insertion and deletion from both ends. This makes it perfect for problems requiring access to both ends efficiently.

**Deque Operations:**
- **Push Front**: Add to beginning - O(1)
- **Push Back**: Add to end - O(1)
- **Pop Front**: Remove from beginning - O(1)
- **Pop Back**: Remove from end - O(1)
- **Front/Back**: Access elements at ends - O(1)

**Key Applications:**

**1. Sliding Window Maximum:**
- Maintain deque with indices in decreasing order
- Front always has index of maximum in current window
- Remove indices outside window from front
- Remove indices of smaller elements from back

**2. Palindrome Checking:**
- Add characters to both ends
- Check if front and back match
- Efficient for checking palindromes

**3. BFS with Priority:**
- Use deque for 0-1 BFS (edges have weight 0 or 1)
- Add weight-0 edges to front, weight-1 to back
- Ensures nodes are processed in correct order

**4. Maximum in All Subarrays:**
- Similar to sliding window maximum
- Find maximum in every window of size k
- Efficient O(n) solution using deque

**Advantages over Regular Queue:**
- Can access/modify both ends
- More flexible for certain algorithms
- Better for sliding window problems
- Enables 0-1 BFS optimization

**Implementation:**
- Can use doubly linked list
- Can use circular array with two pointers
- Most languages provide efficient deque implementations`,
					CodeExamples: `// Go: Sliding Window Maximum using Deque
func maxSlidingWindow(nums []int, k int) []int {
    if len(nums) == 0 {
        return []int{}
    }
    
    var deque []int  // Store indices
    result := make([]int, 0, len(nums)-k+1)
    
    for i := 0; i < len(nums); i++ {
        // Remove indices outside current window
        for len(deque) > 0 && deque[0] <= i-k {
            deque = deque[1:]  // Remove from front
        }
        
        // Remove indices of elements smaller than current
        for len(deque) > 0 && nums[deque[len(deque)-1]] <= nums[i] {
            deque = deque[:len(deque)-1]  // Remove from back
        }
        
        deque = append(deque, i)  // Add to back
        
        // Front of deque is maximum of current window
        if i >= k-1 {
            result = append(result, nums[deque[0]])
        }
    }
    
    return result
}

// Go: Check Palindrome using Deque
func isPalindromeDeque(s string) bool {
    var deque []rune
    for _, char := range s {
        if (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || (char >= '0' && char <= '9') {
            deque = append(deque, toLower(char))
        }
    }
    
    for len(deque) > 1 {
        front := deque[0]
        back := deque[len(deque)-1]
        deque = deque[1 : len(deque)-1]
        
        if front != back {
            return false
        }
    }
    
    return true
}

# Python: Sliding Window Maximum
from collections import deque

def max_sliding_window(nums, k):
    if not nums:
        return []
    
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove indices of smaller elements
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# Python: Palindrome Check
from collections import deque

def is_palindrome_deque(s):
    dq = deque()
    for char in s.lower():
        if char.isalnum():
            dq.append(char)
    
    while len(dq) > 1:
        if dq.popleft() != dq.pop():
            return False
    
    return True`,
				},
			},
			ProblemIDs: []int{31, 32, 33, 34, 35},
		},
		{
			ID:          5,
			Title:       "Linked Lists",
			Description: "Master linked list operations and common patterns for solving linked list problems.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "Linked Lists Basics",
					Content: `A linked list is a linear data structure where elements are stored in nodes, each containing data and a reference to the next node. Unlike arrays, elements are not stored in contiguous memory.

**Types:**

1. **Singly Linked List**:
   - Each node points only to next node
   - Can traverse only forward
   - Less memory overhead

2. **Doubly Linked List**:
   - Each node points to both next and previous
   - Can traverse both forward and backward
   - More memory but more flexibility

3. **Circular Linked List**:
   - Last node points back to first
   - No null pointer at end
   - Useful for round-robin scheduling

**Memory Structure:**
Array:    [10][20][30]  (contiguous)
          All in same memory block

Linked List:  [10]->[20]->[30]->nil
              Scattered in memory

**Advantages:**
- **Dynamic Size**: Grow/shrink as needed
- **Efficient Insertion/Deletion**: O(1) at known position (no shifting)
- **Memory Efficiency**: Only allocate what you need
- **Flexible**: Easy to rearrange elements

**Disadvantages:**
- **No Random Access**: Must traverse from head to reach element
- **Extra Memory**: Need to store pointers (overhead)
- **Cache Unfriendly**: Nodes scattered in memory
- **No Backward Traversal**: In singly linked lists

**Common Operations:**

- **Insert at Head**: O(1) - just update head pointer
- **Insert at Tail**: O(1) if tail pointer maintained, else O(n)
- **Insert at Position**: O(n) - must traverse to position
- **Delete at Head**: O(1) - update head pointer
- **Delete at Position**: O(n) - must find node first
- **Search**: O(n) - linear search required

**Common Patterns:**

1. **Dummy Node**: Use dummy head to simplify edge cases
2. **Two Pointers**: Fast/slow pointers for cycle detection, finding middle
3. **Reverse**: Iterative or recursive reversal
4. **Merge**: Combine two sorted lists efficiently

**When to Use:**
- Unknown or frequently changing size
- Frequent insertions/deletions in middle
- Don't need random access
- Implementing other data structures (stack, queue)

**Common Mistakes:**
- Losing reference to nodes (memory leaks)
- Not handling empty list case
- Off-by-one errors in traversal
- Forgetting to update pointers correctly`,
					CodeExamples: `// Go: ListNode structure
type ListNode struct {
    Val  int
    Next *ListNode
}

// Reverse linked list
func reverseList(head *ListNode) *ListNode {
    var prev *ListNode
    curr := head
    for curr != nil {
        next := curr.Next
        curr.Next = prev
        prev = curr
        curr = next
    }
    return prev
}

# Python: ListNode structure
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next`,
				},
				{
					Title: "Reverse Operations",
					Content: `Reversing linked lists is a fundamental operation that appears in many problems. There are multiple ways to reverse: iterative, recursive, and partial reversals.

**Types of Reversals:**

**1. Reverse Entire List:**
- Reverse all nodes from head to tail
- Iterative: Use three pointers (prev, curr, next)
- Recursive: Reverse rest, then connect head

**2. Reverse in Groups:**
- Reverse k nodes at a time
- Example: Reverse nodes in groups of 2: 1->2->3->4 becomes 2->1->4->3
- Requires careful pointer management

**3. Reverse Between Positions:**
- Reverse nodes from position m to n
- Keep nodes before m and after n unchanged
- Requires finding start position and reconnecting

**4. Reverse in Pairs:**
- Swap every two adjacent nodes
- Example: 1->2->3->4 becomes 2->1->4->3
- Can be done iteratively or recursively

**Key Techniques:**
- **Three Pointer Technique**: prev, curr, next for iterative reversal
- **Dummy Node**: Simplify edge cases (empty list, single node)
- **Recursive Approach**: Reverse rest, then connect current node
- **Group Reversal**: Reverse k nodes, then recursively reverse rest

**Common Patterns:**
- Save next pointer before modifying
- Update pointers in correct order
- Handle edge cases: empty list, single node, k > list length
- Return new head after reversal`,
					CodeExamples: `// Go: Reverse entire list (iterative)
func reverseList(head *ListNode) *ListNode {
    var prev *ListNode
    curr := head
    
    for curr != nil {
        next := curr.Next  // Save next
        curr.Next = prev   // Reverse link
        prev = curr        // Move prev forward
        curr = next        // Move curr forward
    }
    
    return prev  // New head
}

// Go: Reverse entire list (recursive)
func reverseListRecursive(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    
    newHead := reverseListRecursive(head.Next)
    head.Next.Next = head  // Reverse link
    head.Next = nil        // Set tail to nil
    
    return newHead
}

// Go: Reverse in groups of k
func reverseKGroup(head *ListNode, k int) *ListNode {
    curr := head
    count := 0
    
    // Check if k nodes exist
    for curr != nil && count < k {
        curr = curr.Next
        count++
    }
    
    if count == k {
        // Reverse remaining groups first
        curr = reverseKGroup(curr, k)
        
        // Reverse current k nodes
        for count > 0 {
            next := head.Next
            head.Next = curr
            curr = head
            head = next
            count--
        }
        head = curr
    }
    
    return head
}

# Python: Reverse entire list (iterative)
def reverse_list(head):
    prev = None
    curr = head
    
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    
    return prev

# Python: Reverse in groups of k
def reverse_k_group(head, k):
    curr = head
    count = 0
    
    while curr and count < k:
        curr = curr.next
        count += 1
    
    if count == k:
        curr = reverse_k_group(curr, k)
        
        while count > 0:
            next_node = head.next
            head.next = curr
            curr = head
            head = next_node
            count -= 1
        head = curr
    
    return head`,
				},
				{
					Title: "Merge K Sorted Lists",
					Content: `Merging k sorted linked lists is a classic problem that demonstrates divide-and-conquer and heap/priority queue techniques.

**Problem Statement:**
Given k sorted linked lists, merge them into one sorted linked list.

**Approaches:**

**1. Brute Force:**
- Merge lists one by one
- Time: O(k×n) where n is average list length
- Simple but inefficient

**2. Divide and Conquer:**
- Pair up lists and merge pairs
- Repeat until one list remains
- Time: O(n log k) - log k levels, each level processes n nodes
- Space: O(1) excluding recursion stack

**3. Priority Queue/Heap:**
- Add head of each list to min-heap
- Extract minimum, add to result
- Add next node from same list to heap
- Time: O(n log k) - n extractions, each O(log k)
- Space: O(k) - heap stores k nodes

**Best Approach:**
- **Divide and Conquer**: Best time complexity, good space efficiency
- **Priority Queue**: Clean code, good for interview, slightly more space

**Key Insights:**
- Divide and conquer reduces problem size logarithmically
- Priority queue always gives us minimum element efficiently
- Can merge two lists in O(m + n) time
- Handle edge cases: empty lists, single list, null inputs`,
					CodeExamples: `// Go: Merge K Lists using Divide and Conquer
func mergeKLists(lists []*ListNode) *ListNode {
    if len(lists) == 0 {
        return nil
    }
    if len(lists) == 1 {
        return lists[0]
    }
    
    mid := len(lists) / 2
    left := mergeKLists(lists[:mid])
    right := mergeKLists(lists[mid:])
    
    return mergeTwoLists(left, right)
}

func mergeTwoLists(l1, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    curr := dummy
    
    for l1 != nil && l2 != nil {
        if l1.Val <= l2.Val {
            curr.Next = l1
            l1 = l1.Next
        } else {
            curr.Next = l2
            l2 = l2.Next
        }
        curr = curr.Next
    }
    
    if l1 != nil {
        curr.Next = l1
    } else {
        curr.Next = l2
    }
    
    return dummy.Next
}

# Python: Merge K Lists using Divide and Conquer
def merge_k_lists(lists):
    if not lists:
        return None
    if len(lists) == 1:
        return lists[0]
    
    mid = len(lists) // 2
    left = merge_k_lists(lists[:mid])
    right = merge_k_lists(lists[mid:])
    
    return merge_two_lists(left, right)

def merge_two_lists(l1, l2):
    dummy = ListNode()
    curr = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    curr.next = l1 if l1 else l2
    return dummy.next`,
				},
				{
					Title: "Floyd's Cycle Detection Details",
					Content: `Floyd's cycle detection algorithm (also called the "tortoise and hare" algorithm) detects cycles in linked lists using two pointers moving at different speeds.

**How It Works:**
1. Use two pointers: slow (moves 1 step) and fast (moves 2 steps)
2. If cycle exists, fast pointer will eventually catch up to slow pointer
3. If no cycle, fast pointer reaches end (nil)

**Mathematical Proof:**
- Let distance from start to cycle start = m
- Let cycle length = n
- When slow enters cycle, fast is m steps ahead
- Fast catches up at rate of 1 step per iteration
- They meet after (n - m) iterations from cycle start
- Meeting point is (n - m) steps from cycle start

**Finding Cycle Start:**
After detecting cycle, to find where cycle starts:
1. Keep one pointer at meeting point
2. Move another pointer from head
3. Move both one step at a time
4. They meet at cycle start

**Why Finding Cycle Start Works:**
- Distance from head to cycle start = m
- Distance from meeting point to cycle start = n - (n - m) = m
- So both pointers travel same distance to cycle start

**Applications:**
- Detect cycles in linked lists
- Find duplicate numbers (array as linked list)
- Detect infinite loops
- Find cycle length
- Find cycle start position

**Time Complexity:** O(n) - linear time
**Space Complexity:** O(1) - constant space`,
					CodeExamples: `// Go: Detect cycle
func hasCycle(head *ListNode) bool {
    if head == nil {
        return false
    }
    
    slow, fast := head, head
    
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        
        if slow == fast {
            return true  // Cycle detected
        }
    }
    
    return false
}

// Go: Find cycle start
func detectCycle(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    
    slow, fast := head, head
    
    // Detect cycle
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        
        if slow == fast {
            break  // Cycle detected
        }
    }
    
    // No cycle
    if fast == nil || fast.Next == nil {
        return nil
    }
    
    // Find cycle start
    slow = head
    for slow != fast {
        slow = slow.Next
        fast = fast.Next
    }
    
    return slow  // Cycle start
}

# Python: Detect cycle
def has_cycle(head):
    if not head:
        return False
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False

# Python: Find cycle start
def detect_cycle(head):
    if not head:
        return None
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            break
    
    if not fast or not fast.next:
        return None
    
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow`,
				},
			},
			ProblemIDs: []int{36, 37, 38, 39, 40},
		},
		{
			ID:          6,
			Title:       "Trees",
			Description: "Learn tree data structures, traversal methods, and tree-based algorithms.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "Binary Trees",
					Content: `A binary tree is a tree data structure where each node has at most two children: left and right. Trees are fundamental for representing hierarchical relationships.

**Tree Terminology:**
- **Root**: Topmost node (no parent)
- **Leaf**: Node with no children (terminal node)
- **Internal Node**: Node with at least one child
- **Depth**: Distance from root to node (root depth = 0)
- **Height**: Maximum depth of tree (longest path from root to leaf)
- **Subtree**: Tree formed by a node and all its descendants
- **Ancestor**: Any node on path from root to current node
- **Descendant**: Any node in subtree of current node
- **Sibling**: Nodes with same parent

**Tree Properties:**
- **Full Binary Tree**: Every node has 0 or 2 children
- **Complete Binary Tree**: All levels filled except possibly last, filled left to right
- **Perfect Binary Tree**: All internal nodes have 2 children, all leaves at same level
- **Balanced Tree**: Height difference between left and right subtrees ≤ 1

**Traversal Methods:**

1. **Inorder (Left → Root → Right)**:
   - Visits nodes in sorted order for BST
   - Used for: printing sorted values, expression trees

2. **Preorder (Root → Left → Right)**:
   - Root visited before children
   - Used for: copying tree, prefix expressions

3. **Postorder (Left → Right → Root)**:
   - Root visited after children
   - Used for: deleting tree, postfix expressions

4. **Level-order (Breadth-First)**:
   - Visits nodes level by level
   - Uses queue for implementation
   - Used for: finding level-specific information

**Recursive vs Iterative:**
- **Recursive**: Natural for tree problems, uses call stack
- **Iterative**: Uses explicit stack/queue, more control

**Common Patterns:**
- **DFS (Depth-First)**: Preorder, Inorder, Postorder
- **BFS (Breadth-First)**: Level-order traversal
- **Path Problems**: Track path from root to node
- **Tree Construction**: Build tree from traversal sequences

**Time Complexity:** O(n) for all traversals - visit each node once

**Space Complexity:** 
- Recursive: O(h) where h is height (call stack)
- Iterative: O(n) worst case (explicit stack/queue)`,
					CodeExamples: `// Go: TreeNode structure
type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

// Inorder traversal
func inorderTraversal(root *TreeNode) []int {
    var result []int
    if root == nil {
        return result
    }
    result = append(result, inorderTraversal(root.Left)...)
    result = append(result, root.Val)
    result = append(result, inorderTraversal(root.Right)...)
    return result
}

# Python: TreeNode structure
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right`,
				},
				{
					Title: "Binary Search Trees",
					Content: `A Binary Search Tree (BST) is a binary tree with ordering property that enables efficient searching, insertion, and deletion.

**BST Property:**
- **Left subtree**: All nodes have values < root
- **Right subtree**: All nodes have values > root
- **Both subtrees**: Are also valid BSTs
- **No duplicates**: Typically (or handle with count/frequency)

**Key Properties:**
- **Inorder Traversal**: Always gives sorted order (ascending)
- **Search**: O(log n) average (balanced), O(n) worst (skewed)
- **Insert**: O(log n) average, O(n) worst
- **Delete**: O(log n) average, O(n) worst
- **Min/Max**: O(log n) - leftmost/rightmost node

**BST Operations:**

1. **Search**:
   - Compare target with root
   - Go left if smaller, right if larger
   - Repeat until found or null

2. **Insert**:
   - Find correct position (like search)
   - Insert new node as leaf
   - Maintain BST property

3. **Delete** (Three Cases):
   - **No children**: Simply remove node
   - **One child**: Replace with child
   - **Two children**: Replace with inorder successor (or predecessor)

**Balanced BSTs:**
- **AVL Tree**: Self-balancing, height difference ≤ 1
- **Red-Black Tree**: Self-balancing with color properties
- **B-Tree**: Multi-way tree for databases
- **Splay Tree**: Recently accessed nodes move to root

**When BST Degenerates:**
- Inserting sorted sequence creates linked list
- Height becomes O(n) instead of O(log n)
- Operations become O(n) instead of O(log n)

**Common Problems:**
- Validate BST (check ordering property)
- Find kth smallest/largest element
- Range queries (find all values in range)
- Lowest Common Ancestor (LCA)

**Real-World Applications:**
- Database indexing
- Symbol tables in compilers
- Priority queues (with modifications)
- Expression evaluation`,
					CodeExamples: `// Go: BST search
func searchBST(root *TreeNode, val int) *TreeNode {
    if root == nil || root.Val == val {
        return root
    }
    if val < root.Val {
        return searchBST(root.Left, val)
    }
    return searchBST(root.Right, val)
}

# Python: BST search
def searchBST(root, val):
    if not root or root.val == val:
        return root
    if val < root.val:
        return searchBST(root.left, val)
    return searchBST(root.right, val)`,
				},
				{
					Title: "Tree Traversal Techniques",
					Content: `Master different tree traversal methods and when to use each.

**Iterative Traversals:**

All recursive traversals can be converted to iterative using explicit stack:

1. **Preorder Iterative**:
   - Push root to stack
   - While stack not empty: pop, process, push right then left

2. **Inorder Iterative**:
   - Use stack to simulate recursion
   - Go left as far as possible, then process, then go right

3. **Postorder Iterative**:
   - Most complex - need to track if node processed
   - Use two stacks or modified preorder

**Level-order (BFS) Traversal:**
- Use queue instead of stack
- Process nodes level by level
- Natural for: finding level-specific info, shortest path

**Morris Traversal:**
- Inorder traversal with O(1) space (no stack)
- Uses threaded binary tree concept
- Advanced technique for space-constrained scenarios

**Common Tree Problems:**
- Maximum depth/height
- Symmetric tree checking
- Path sum problems
- Lowest Common Ancestor (LCA)
- Serialization/Deserialization`,
					CodeExamples: `// Go: Iterative preorder
func preorderIterative(root *TreeNode) []int {
    if root == nil {
        return []int{}
    }
    stack := []*TreeNode{root}
    result := []int{}
    
    for len(stack) > 0 {
        node := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        result = append(result, node.Val)
        
        if node.Right != nil {
            stack = append(stack, node.Right)
        }
        if node.Left != nil {
            stack = append(stack, node.Left)
        }
    }
    return result
}

// Level-order traversal
func levelOrder(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{}
    }
    queue := []*TreeNode{root}
    result := [][]int{}
    
    for len(queue) > 0 {
        levelSize := len(queue)
        level := []int{}
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            level = append(level, node.Val)
            
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
        result = append(result, level)
    }
    return result
}

# Python: Iterative preorder
def preorder_iterative(root):
    if not root:
        return []
    stack = [root]
    result = []
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result`,
				},
				{
					Title: "Tree Construction",
					Content: `Building trees from various inputs is a common problem type. Understanding how to construct trees from traversal sequences is essential.

**Common Construction Problems:**

**1. Build Tree from Preorder and Inorder:**
- Preorder gives root first, then left subtree, then right subtree
- Inorder gives left subtree, then root, then right subtree
- Use preorder to find root, use inorder to split into left/right subtrees
- Recursively build subtrees

**2. Build Tree from Postorder and Inorder:**
- Postorder gives left subtree, then right subtree, then root
- Similar approach: use postorder to find root (from end), use inorder to split
- Process postorder from right to left

**3. Build Tree from Preorder and Postorder:**
- More challenging - need to identify subtree boundaries
- Use preorder to find root, find next root in postorder to identify left subtree end

**4. Build Balanced BST from Sorted Array:**
- Middle element is root
- Recursively build left subtree from left half, right subtree from right half
- Results in balanced BST

**5. Build Tree from Level-order:**
- Use queue to process nodes level by level
- Connect children as you process each node

**Key Techniques:**
- Use hash map to quickly find positions in inorder array
- Recursively build left and right subtrees
- Handle edge cases: empty arrays, single element
- Track indices carefully to avoid out-of-bounds errors`,
					CodeExamples: `// Go: Build tree from preorder and inorder
func buildTree(preorder []int, inorder []int) *TreeNode {
    // Create map for O(1) lookup
    inMap := make(map[int]int)
    for i, val := range inorder {
        inMap[val] = i
    }
    
    preIndex := 0
    return buildHelper(preorder, &preIndex, inorder, 0, len(inorder)-1, inMap)
}

func buildHelper(preorder []int, preIndex *int, inorder []int, inStart, inEnd int, inMap map[int]int) *TreeNode {
    if inStart > inEnd {
        return nil
    }
    
    rootVal := preorder[*preIndex]
    *preIndex++
    root := &TreeNode{Val: rootVal}
    
    inRoot := inMap[rootVal]
    
    root.Left = buildHelper(preorder, preIndex, inorder, inStart, inRoot-1, inMap)
    root.Right = buildHelper(preorder, preIndex, inorder, inRoot+1, inEnd, inMap)
    
    return root
}

// Go: Build balanced BST from sorted array
func sortedArrayToBST(nums []int) *TreeNode {
    if len(nums) == 0 {
        return nil
    }
    
    mid := len(nums) / 2
    root := &TreeNode{Val: nums[mid]}
    root.Left = sortedArrayToBST(nums[:mid])
    root.Right = sortedArrayToBST(nums[mid+1:])
    
    return root
}

# Python: Build tree from preorder and inorder
def build_tree(preorder, inorder):
    in_map = {val: idx for idx, val in enumerate(inorder)}
    pre_idx = [0]  # Use list to modify in recursive calls
    
    def build(in_start, in_end):
        if in_start > in_end:
            return None
        
        root_val = preorder[pre_idx[0]]
        pre_idx[0] += 1
        root = TreeNode(root_val)
        
        in_root = in_map[root_val]
        
        root.left = build(in_start, in_root - 1)
        root.right = build(in_root + 1, in_end)
        
        return root
    
    return build(0, len(inorder) - 1)

# Python: Build balanced BST from sorted array
def sorted_array_to_bst(nums):
    if not nums:
        return None
    
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid+1:])
    
    return root`,
				},
				{
					Title: "Lowest Common Ancestor",
					Content: `The Lowest Common Ancestor (LCA) of two nodes in a tree is the deepest node that has both nodes as descendants. This is a fundamental tree problem with many applications.

**Problem Types:**

**1. LCA in Binary Tree:**
- No ordering property (not necessarily BST)
- Need to search both subtrees
- Return node if found in both subtrees, or return non-null result

**2. LCA in BST:**
- Can use BST property to navigate
- If both nodes < root, go left
- If both nodes > root, go right
- Otherwise, root is LCA

**3. LCA with Parent Pointers:**
- Each node has pointer to parent
- Find depth of both nodes
- Move deeper node up to same depth
- Move both up until they meet

**4. LCA of Deepest Leaves:**
- Find deepest leaves first
- Then find LCA of all deepest leaves
- Requires two passes: find depth, then find LCA

**Algorithm for Binary Tree:**
1. If root is null or equals p or q, return root
2. Recursively search left and right subtrees
3. If both subtrees return non-null, root is LCA
4. Otherwise, return non-null result (LCA is in that subtree)

**Time Complexity:** O(n) - visit each node once
**Space Complexity:** O(h) - recursion stack depth

**Applications:**
- Find distance between two nodes
- Determine relationship between nodes
- Tree queries and path problems
- Network routing algorithms`,
					CodeExamples: `// Go: LCA in Binary Tree
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
    if root == nil || root == p || root == q {
        return root
    }
    
    left := lowestCommonAncestor(root.Left, p, q)
    right := lowestCommonAncestor(root.Right, p, q)
    
    if left != nil && right != nil {
        return root  // p and q in different subtrees
    }
    
    if left != nil {
        return left  // Both in left subtree
    }
    return right  // Both in right subtree
}

// Go: LCA in BST (optimized)
func lowestCommonAncestorBST(root, p, q *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    
    // Both nodes in left subtree
    if p.Val < root.Val && q.Val < root.Val {
        return lowestCommonAncestorBST(root.Left, p, q)
    }
    
    // Both nodes in right subtree
    if p.Val > root.Val && q.Val > root.Val {
        return lowestCommonAncestorBST(root.Right, p, q)
    }
    
    // Nodes in different subtrees, root is LCA
    return root
}

// Go: LCA with parent pointers
type NodeWithParent struct {
    Val    int
    Left   *NodeWithParent
    Right  *NodeWithParent
    Parent *NodeWithParent
}

func lowestCommonAncestorWithParent(p, q *NodeWithParent) *NodeWithParent {
    // Find depth of both nodes
    depthP := getDepth(p)
    depthQ := getDepth(q)
    
    // Move deeper node up to same level
    if depthP > depthQ {
        for i := 0; i < depthP-depthQ; i++ {
            p = p.Parent
        }
    } else {
        for i := 0; i < depthQ-depthP; i++ {
            q = q.Parent
        }
    }
    
    // Move both up until they meet
    for p != q {
        p = p.Parent
        q = q.Parent
    }
    
    return p
}

func getDepth(node *NodeWithParent) int {
    depth := 0
    for node.Parent != nil {
        depth++
        node = node.Parent
    }
    return depth
}

# Python: LCA in Binary Tree
def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root
    
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    if left and right:
        return root
    
    return left if left else right

# Python: LCA in BST
def lowest_common_ancestor_bst(root, p, q):
    if not root:
        return None
    
    if p.val < root.val and q.val < root.val:
        return lowest_common_ancestor_bst(root.left, p, q)
    
    if p.val > root.val and q.val > root.val:
        return lowest_common_ancestor_bst(root.right, p, q)
    
    return root`,
				},
				{
					Title: "Tree Serialization",
					Content: `Tree serialization converts a tree into a string representation, and deserialization reconstructs the tree from that string. This is essential for storing trees or transmitting them over networks.

**Serialization Formats:**

**1. Preorder with Null Markers:**
- Use special marker (e.g., "#" or "null") for null nodes
- Preorder traversal: root, left, right
- Example: "1,2,#,#,3,4,#,#,5,#,#"

**2. Level-order Serialization:**
- Serialize level by level
- Include null markers for missing children
- Example: "1,2,3,#,#,4,5"

**3. JSON Format:**
- Use JSON structure with nested objects
- More verbose but human-readable
- Example: {"val": 1, "left": {"val": 2}, "right": {"val": 3}}

**Deserialization:**
- Parse string back into tree structure
- Reconstruct nodes in same order as serialization
- Handle null markers correctly

**Key Considerations:**
- **Uniqueness**: Serialization should uniquely represent tree
- **Efficiency**: Compact representation saves space
- **Robustness**: Handle edge cases (empty tree, single node)
- **Parsing**: Efficient string parsing is important

**Use Cases:**
- Storing trees in databases
- Transmitting trees over network
- Caching tree structures
- Debugging tree algorithms
- Version control for tree data

**Common Problems:**
- Serialize and deserialize binary tree
- Serialize and deserialize BST (can use more efficient format)
- Serialize N-ary tree
- Compare trees using serialization`,
					CodeExamples: `// Go: Serialize tree using preorder
import (
    "strconv"
    "strings"
)

func serialize(root *TreeNode) string {
    var result []string
    serializeHelper(root, &result)
    return strings.Join(result, ",")
}

func serializeHelper(root *TreeNode, result *[]string) {
    if root == nil {
        *result = append(*result, "#")
        return
    }
    
    *result = append(*result, strconv.Itoa(root.Val))
    serializeHelper(root.Left, result)
    serializeHelper(root.Right, result)
}

// Go: Deserialize tree
func deserialize(data string) *TreeNode {
    values := strings.Split(data, ",")
    index := 0
    return deserializeHelper(values, &index)
}

func deserializeHelper(values []string, index *int) *TreeNode {
    if *index >= len(values) || values[*index] == "#" {
        (*index)++
        return nil
    }
    
    val, _ := strconv.Atoi(values[*index])
    (*index)++
    
    root := &TreeNode{Val: val}
    root.Left = deserializeHelper(values, index)
    root.Right = deserializeHelper(values, index)
    
    return root
}

# Python: Serialize tree using preorder
def serialize(root):
    result = []
    
    def serialize_helper(node):
        if not node:
            result.append("#")
            return
        result.append(str(node.val))
        serialize_helper(node.left)
        serialize_helper(node.right)
    
    serialize_helper(root)
    return ",".join(result)

# Python: Deserialize tree
def deserialize(data):
    values = data.split(",")
    index = [0]
    
    def deserialize_helper():
        if index[0] >= len(values) or values[index[0]] == "#":
            index[0] += 1
            return None
        
        val = int(values[index[0]])
        index[0] += 1
        
        root = TreeNode(val)
        root.left = deserialize_helper()
        root.right = deserialize_helper()
        
        return root
    
    return deserialize_helper()`,
				},
			},
			ProblemIDs: []int{41, 42, 43, 44, 45, 46, 47, 48, 49, 50},
		},
		{
			ID:          8,
			Title:       "Strings",
			Description: "Master string manipulation techniques, pattern matching, and string algorithms.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "String Fundamentals",
					Content: `Strings are sequences of characters. Understanding string operations is crucial for many programming problems.

**Key Concepts:**
- Strings are immutable in many languages (Go, Python, Java)
- Character encoding (ASCII, Unicode)
- String vs character array
- Common operations: concatenation, slicing, searching

**Important Operations:**
- **Length**: O(1) in most languages
- **Access**: O(1) by index
- **Substring**: O(n) where n is substring length
- **Search**: O(n) for single character, O(n×m) for pattern matching

**Common Patterns:**
- Two pointers for palindromes
- Sliding window for substrings
- Character frequency counting
- String building (use StringBuilder in Java, list + join in Python)`,
					CodeExamples: `// Go: String operations
s := "hello"
length := len(s)  // O(1)
char := s[0]  // O(1) - byte access
substr := s[1:3]  // O(n) - creates new string

// String building (efficient)
var builder strings.Builder
for i := 0; i < 10; i++ {
    builder.WriteString("a")
}
result := builder.String()  // O(n) total

# Python: String operations
s = "hello"
length = len(s)  # O(1)
char = s[0]  # O(1)
substr = s[1:3]  # O(n) - creates new string

# String building (efficient)
result = ''.join(['a' for _ in range(10)])  # O(n) total`,
				},
				{
					Title: "String Pattern Matching",
					Content: `Pattern matching is finding occurrences of a pattern in a text string.

**Naive Approach:**
- Check every position: O(n×m) where n=text length, m=pattern length
- Simple but inefficient for long strings

**Efficient Algorithms:**
1. **KMP (Knuth-Morris-Pratt)**: O(n + m)
   - Uses failure function to avoid re-checking characters
   - Preprocesses pattern to build skip table

2. **Rabin-Karp**: O(n + m) average, O(n×m) worst
   - Uses rolling hash
   - Good for multiple pattern matching

3. **Boyer-Moore**: O(n/m) best case, O(n×m) worst
   - Skips characters using bad character rule
   - Efficient for long patterns

**When to Use:**
- Simple problems: Naive approach is fine
- Interview problems: Usually expect O(n) solution
- Production: Use built-in functions (strings.Contains, regex)`,
					CodeExamples: `// Go: Naive pattern matching
func containsPattern(text, pattern string) bool {
    for i := 0; i <= len(text)-len(pattern); i++ {
        match := true
        for j := 0; j < len(pattern); j++ {
            if text[i+j] != pattern[j] {
                match = false
                break
            }
        }
        if match {
            return true
        }
    }
    return false
}

# Python: Using built-in (highly optimized)
def contains_pattern(text, pattern):
    return pattern in text  # Uses optimized algorithm

# Manual implementation
def contains_pattern_manual(text, pattern):
    for i in range(len(text) - len(pattern) + 1):
        if text[i:i+len(pattern)] == pattern:
            return True
    return False`,
				},
				{
					Title: "String Manipulation Techniques",
					Content: `Common string manipulation techniques for solving problems:

**1. Character Frequency Counting:**
- Use hash map to count occurrences
- Useful for anagrams, character replacement

**2. Two Pointers:**
- Check palindromes
- Reverse strings
- Remove duplicates

**3. Sliding Window:**
- Longest substring without repeating characters
- Minimum window substring

**4. String Building:**
- Build strings efficiently (avoid concatenation in loops)
- Use StringBuilder or list + join

**5. Character Mapping:**
- Map characters to indices (a→0, b→1, etc.)
- Useful for frequency arrays

**Common Problems:**
- Valid anagrams
- Longest palindromic substring
- String compression
- String matching`,
					CodeExamples: `// Go: Character frequency
func countChars(s string) map[rune]int {
    freq := make(map[rune]int)
    for _, char := range s {
        freq[char]++
    }
    return freq
}

// Two pointers: Check palindrome
func isPalindrome(s string) bool {
    left, right := 0, len(s)-1
    for left < right {
        if s[left] != s[right] {
            return false
        }
        left++
        right--
    }
    return true
}

# Python: Character frequency
def count_chars(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq

# Two pointers: Check palindrome
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True`,
				},
				{
					Title: "String Matching Algorithms",
					Content: `Efficient string matching algorithms are crucial for text processing, search engines, and pattern recognition. Understanding these algorithms helps solve complex string problems.

**Algorithm Comparison:**

**1. Naive Algorithm:**
- Check pattern at every position
- Time: O(n×m) where n=text length, m=pattern length
- Simple but inefficient

**2. KMP (Knuth-Morris-Pratt):**
- Preprocesses pattern to build failure function (LPS array)
- Uses information from previous matches to skip positions
- Time: O(n + m) - linear time!
- Space: O(m) for LPS array

**3. Rabin-Karp:**
- Uses rolling hash to compare substrings
- Hash text substrings and compare with pattern hash
- Time: O(n + m) average, O(n×m) worst (hash collisions)
- Good for multiple pattern matching

**4. Boyer-Moore:**
- Scans pattern from right to left
- Uses bad character rule and good suffix rule to skip
- Time: O(n/m) best case, O(n×m) worst
- Very efficient for long patterns

**5. Z-Algorithm:**
- Builds Z-array: Z[i] = longest substring starting at i that matches prefix
- Used for pattern matching and string problems
- Time: O(n + m)

**When to Use:**
- **Naive**: Short strings, simple problems
- **KMP**: General pattern matching, guaranteed O(n+m)
- **Rabin-Karp**: Multiple patterns, plagiarism detection
- **Boyer-Moore**: Long patterns, text editors
- **Built-in**: Production code (highly optimized)`,
					CodeExamples: `// Go: KMP Algorithm
func kmpSearch(text, pattern string) []int {
    n, m := len(text), len(pattern)
    if m == 0 {
        return []int{}
    }
    
    // Build LPS (Longest Proper Prefix which is also Suffix) array
    lps := buildLPS(pattern)
    
    result := []int{}
    i, j := 0, 0  // i for text, j for pattern
    
    for i < n {
        if text[i] == pattern[j] {
            i++
            j++
        }
        
        if j == m {
            result = append(result, i-j)  // Pattern found
            j = lps[j-1]
        } else if i < n && text[i] != pattern[j] {
            if j != 0 {
                j = lps[j-1]
            } else {
                i++
            }
        }
    }
    
    return result
}

func buildLPS(pattern string) []int {
    m := len(pattern)
    lps := make([]int, m)
    length := 0
    i := 1
    
    for i < m {
        if pattern[i] == pattern[length] {
            length++
            lps[i] = length
            i++
        } else {
            if length != 0 {
                length = lps[length-1]
            } else {
                lps[i] = 0
                i++
            }
        }
    }
    
    return lps
}

// Go: Rabin-Karp Algorithm
func rabinKarp(text, pattern string) []int {
    n, m := len(text), len(pattern)
    if m == 0 || n < m {
        return []int{}
    }
    
    const base = 256
    const mod = 101  // Prime number
    
    // Calculate hash of pattern and first window
    patternHash := 0
    textHash := 0
    h := 1
    
    for i := 0; i < m-1; i++ {
        h = (h * base) % mod
    }
    
    for i := 0; i < m; i++ {
        patternHash = (base*patternHash + int(pattern[i])) % mod
        textHash = (base*textHash + int(text[i])) % mod
    }
    
    result := []int{}
    
    for i := 0; i <= n-m; i++ {
        if patternHash == textHash {
            // Verify match (hash collision check)
            match := true
            for j := 0; j < m; j++ {
                if text[i+j] != pattern[j] {
                    match = false
                    break
                }
            }
            if match {
                result = append(result, i)
            }
        }
        
        // Calculate hash for next window
        if i < n-m {
            textHash = (base*(textHash-int(text[i])*h) + int(text[i+m])) % mod
            if textHash < 0 {
                textHash += mod
            }
        }
    }
    
    return result
}

# Python: KMP Algorithm
def kmp_search(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0:
        return []
    
    lps = build_lps(pattern)
    result = []
    i = j = 0
    
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == m:
            result.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return result

def build_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    
    return lps`,
				},
				{
					Title: "Trie Data Structure",
					Content: `A Trie (prefix tree) is a tree-like data structure that stores strings efficiently and enables fast prefix-based searches. It's essential for autocomplete, spell checkers, and IP routing.

**Trie Properties:**
- Each node represents a character
- Path from root to node represents a string
- Common prefixes share nodes (space efficient)
- Fast prefix search: O(m) where m is prefix length

**Operations:**
- **Insert**: Add string to trie - O(m) where m is string length
- **Search**: Check if string exists - O(m)
- **StartsWith**: Check if prefix exists - O(m)
- **Delete**: Remove string - O(m)

**Node Structure:**
- Character value (or implicit from position)
- Children array/map (one per possible character)
- Boolean flag: isEndOfWord
- Optional: count/frequency for word frequency

**Advantages:**
- Fast prefix searches
- Space efficient for common prefixes
- Easy to implement autocomplete
- Good for dictionary/word problems

**Disadvantages:**
- Can use significant memory (especially with large alphabets)
- Slower than hash table for exact matches
- More complex than simple data structures

**Applications:**
- Autocomplete systems
- Spell checkers
- IP routing (longest prefix matching)
- Word games (Scrabble, Boggle)
- Search engines (prefix matching)
- Contact list search

**Variations:**
- **Compressed Trie**: Merge single-child nodes
- **Suffix Trie**: Store all suffixes (for suffix matching)
- **Ternary Search Trie**: More memory efficient`,
					CodeExamples: `// Go: Trie Implementation
type TrieNode struct {
    children map[rune]*TrieNode
    isEnd    bool
}

type Trie struct {
    root *TrieNode
}

func NewTrie() *Trie {
    return &Trie{root: &TrieNode{children: make(map[rune]*TrieNode)}}
}

func (t *Trie) Insert(word string) {
    node := t.root
    for _, char := range word {
        if node.children[char] == nil {
            node.children[char] = &TrieNode{children: make(map[rune]*TrieNode)}
        }
        node = node.children[char]
    }
    node.isEnd = true
}

func (t *Trie) Search(word string) bool {
    node := t.root
    for _, char := range word {
        if node.children[char] == nil {
            return false
        }
        node = node.children[char]
    }
    return node.isEnd
}

func (t *Trie) StartsWith(prefix string) bool {
    node := t.root
    for _, char := range prefix {
        if node.children[char] == nil {
            return false
        }
        node = node.children[char]
    }
    return true
}

// Go: Word Search using Trie
func findWords(board [][]byte, words []string) []string {
    trie := NewTrie()
    for _, word := range words {
        trie.Insert(word)
    }
    
    result := []string{}
    visited := make([][]bool, len(board))
    for i := range visited {
        visited[i] = make([]bool, len(board[0]))
    }
    
    var dfs func(int, int, *TrieNode, string)
    dfs = func(i, j int, node *TrieNode, path string) {
        if i < 0 || i >= len(board) || j < 0 || j >= len(board[0]) || visited[i][j] {
            return
        }
        
        char := rune(board[i][j])
        if node.children[char] == nil {
            return
        }
        
        node = node.children[char]
        path += string(char)
        
        if node.isEnd {
            result = append(result, path)
            node.isEnd = false  // Avoid duplicates
        }
        
        visited[i][j] = true
        dfs(i+1, j, node, path)
        dfs(i-1, j, node, path)
        dfs(i, j+1, node, path)
        dfs(i, j-1, node, path)
        visited[i][j] = false
    }
    
    for i := 0; i < len(board); i++ {
        for j := 0; j < len(board[0]); j++ {
            dfs(i, j, trie.root, "")
        }
    }
    
    return result
}

# Python: Trie Implementation
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True`,
				},
				{
					Title: "Suffix Arrays",
					Content: `Suffix arrays are powerful data structures for string processing that enable efficient substring searches and various string algorithms.

**What is a Suffix Array?**
- Array containing all suffixes of a string in sorted order
- Each element is starting index of a suffix
- Enables binary search for substring matching

**Construction:**
- Generate all suffixes: O(n) suffixes, each O(n) length
- Sort suffixes: O(n log n) comparisons, each O(n) time
- Total: O(n² log n) naive, O(n log n) optimized

**Applications:**

**1. Substring Search:**
- Binary search on suffix array
- Find all occurrences of pattern
- Time: O(m log n) where m=pattern length, n=text length

**2. Longest Common Substring:**
- Compare adjacent suffixes in sorted order
- Find longest common prefix between adjacent suffixes

**3. Longest Repeated Substring:**
- Find longest substring that appears at least twice
- Compare adjacent suffixes

**4. Suffix Array vs Suffix Tree:**
- Suffix Array: Simpler, less memory, slightly slower
- Suffix Tree: Faster queries, more memory, more complex

**Key Operations:**
- **Build**: Construct suffix array from string
- **Search**: Find pattern using binary search
- **LCP Array**: Longest Common Prefix array for optimizations

**LCP Array:**
- LCP[i] = length of longest common prefix between SA[i] and SA[i-1]
- Enables faster substring searches
- Can be computed in O(n) time`,
					CodeExamples: `// Go: Suffix Array Construction
func buildSuffixArray(s string) []int {
    n := len(s)
    suffixes := make([][2]int, n)  // [index, rank]
    
    // Initial ranks (characters)
    for i := 0; i < n; i++ {
        suffixes[i] = [2]int{i, int(s[i])}
    }
    
    // Sort by rank
    sort.Slice(suffixes, func(i, j int) bool {
        return suffixes[i][1] < suffixes[j][1]
    })
    
    // Assign new ranks
    rank := make([]int, n)
    rank[suffixes[0][0]] = 0
    for i := 1; i < n; i++ {
        if suffixes[i][1] == suffixes[i-1][1] {
            rank[suffixes[i][0]] = rank[suffixes[i-1][0]]
        } else {
            rank[suffixes[i][0]] = i
        }
    }
    
    // Build suffix array indices
    sa := make([]int, n)
    for i := 0; i < n; i++ {
        sa[i] = suffixes[i][0]
    }
    
    return sa
}

// Go: Search pattern using suffix array
func searchPattern(text string, sa []int, pattern string) []int {
    n := len(text)
    m := len(pattern)
    result := []int{}
    
    // Binary search for pattern
    left, right := 0, n-1
    start, end := -1, -1
    
    // Find left boundary
    for left <= right {
        mid := (left + right) / 2
        suffix := text[sa[mid]:]
        cmp := comparePrefix(suffix, pattern)
        
        if cmp == 0 {
            start = mid
            right = mid - 1
        } else if cmp < 0 {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    // Find right boundary
    left, right = 0, n-1
    for left <= right {
        mid := (left + right) / 2
        suffix := text[sa[mid]:]
        cmp := comparePrefix(suffix, pattern)
        
        if cmp == 0 {
            end = mid
            left = mid + 1
        } else if cmp < 0 {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    // Collect all occurrences
    if start != -1 && end != -1 {
        for i := start; i <= end; i++ {
            result = append(result, sa[i])
        }
    }
    
    return result
}

func comparePrefix(s, pattern string) int {
    minLen := len(s)
    if len(pattern) < minLen {
        minLen = len(pattern)
    }
    
    for i := 0; i < minLen; i++ {
        if s[i] < pattern[i] {
            return -1
        } else if s[i] > pattern[i] {
            return 1
        }
    }
    
    if len(s) < len(pattern) {
        return -1
    } else if len(s) > len(pattern) {
        return 1
    }
    return 0
}

# Python: Suffix Array Construction
def build_suffix_array(s):
    n = len(s)
    suffixes = [(i, s[i:]) for i in range(n)]
    suffixes.sort(key=lambda x: x[1])
    return [idx for idx, _ in suffixes]

# Python: Search pattern using suffix array
def search_pattern(text, sa, pattern):
    n = len(text)
    result = []
    
    # Binary search for left boundary
    left, right = 0, n - 1
    start = -1
    
    while left <= right:
        mid = (left + right) // 2
        suffix = text[sa[mid]:]
        if suffix.startswith(pattern):
            start = mid
            right = mid - 1
        elif suffix < pattern:
            left = mid + 1
        else:
            right = mid - 1
    
    # Binary search for right boundary
    left, right = 0, n - 1
    end = -1
    
    while left <= right:
        mid = (left + right) // 2
        suffix = text[sa[mid]:]
        if suffix.startswith(pattern):
            end = mid
            left = mid + 1
        elif suffix < pattern:
            left = mid + 1
        else:
            right = mid - 1
    
    if start != -1 and end != -1:
        return [sa[i] for i in range(start, end + 1)]
    return []`,
				},
			},
			ProblemIDs: []int{2, 4, 15, 27, 30},
		},
		{
			ID:          9,
			Title:       "Graphs",
			Description: "Learn graph data structures, traversal algorithms, and graph-based problem solving.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Graph Fundamentals",
					Content: `A graph is a collection of nodes (vertices) connected by edges. Graphs are the most general data structure, modeling any relationship or network.

**Graph Types:**

1. **Directed Graph (Digraph)**:
   - Edges have direction: A→B means edge from A to B
   - A→B ≠ B→A (unless both edges exist)
   - Examples: Web links, dependencies, social media follows

2. **Undirected Graph**:
   - Edges have no direction: A-B means bidirectional connection
   - A-B = B-A
   - Examples: Social networks (friends), road networks

3. **Weighted Graph**:
   - Edges have weights/costs
   - Can represent distance, cost, time, etc.
   - Examples: Maps with distances, network latency

4. **Unweighted Graph**:
   - All edges equal (weight = 1)
   - Simpler, used when only connectivity matters

**Graph Representations:**

1. **Adjacency List** (Most common):
   - List of lists: graph[i] = [neighbors of node i]
   - **Space**: O(V + E) - efficient for sparse graphs
   - **Edge Lookup**: O(degree) - check neighbors
   - **Add Edge**: O(1) - append to list
   - **Best for**: Sparse graphs, most algorithms

2. **Adjacency Matrix**:
   - 2D array: matrix[i][j] = 1 if edge exists, 0 otherwise
   - **Space**: O(V²) - can be wasteful for sparse graphs
   - **Edge Lookup**: O(1) - direct array access
   - **Add Edge**: O(1) - set matrix[i][j] = 1
   - **Best for**: Dense graphs, frequent edge queries

**Key Terminology:**
- **Vertex/Node**: A point in the graph
- **Edge**: Connection between vertices
- **Degree**: Number of edges connected to vertex
  - In-degree: incoming edges (directed)
  - Out-degree: outgoing edges (directed)
- **Path**: Sequence of connected vertices
- **Cycle**: Path that starts and ends at same vertex
- **Connected**: All vertices reachable from each other
- **Component**: Maximal connected subgraph
- **Tree**: Connected graph with no cycles (n-1 edges for n vertices)
- **Forest**: Collection of trees

**Graph Properties:**
- **Sparse**: Few edges relative to vertices (E ≈ V)
- **Dense**: Many edges (E ≈ V²)
- **Acyclic**: No cycles
- **Connected**: All vertices reachable
- **Bipartite**: Can partition vertices into two sets with no edges within sets`,
					CodeExamples: `// Go: Graph representation (adjacency list)
type Graph struct {
    vertices int
    adj      [][]int
}

func NewGraph(vertices int) *Graph {
    return &Graph{
        vertices: vertices,
        adj:      make([][]int, vertices),
    }
}

func (g *Graph) AddEdge(u, v int) {
    g.adj[u] = append(g.adj[u], v)
    // For undirected: g.adj[v] = append(g.adj[v], u)
}

# Python: Graph representation
class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj = [[] for _ in range(vertices)]
    
    def add_edge(self, u, v):
        self.adj[u].append(v)
        # For undirected: self.adj[v].append(u)`,
				},
				{
					Title: "Graph Traversal: BFS",
					Content: `Breadth-First Search (BFS) explores a graph level by level, visiting all neighbors before moving to next level. It's like ripples spreading from a stone dropped in water.

**BFS Algorithm:**
1. Start from source node, mark as visited
2. Add source to queue
3. While queue not empty:
   - Dequeue node
   - Process node
   - Add all unvisited neighbors to queue
   - Mark neighbors as visited

**Key Properties:**
- **Shortest Path**: Finds shortest path in unweighted graphs (minimum edges)
- **Level Order**: Visits nodes level by level (distance 0, 1, 2, ...)
- **Queue-based**: Uses FIFO queue to maintain level order
- **Complete**: Visits all reachable nodes from source
- **Time**: O(V + E) - visit each vertex and edge once
- **Space**: O(V) - queue and visited array

**BFS Applications:**

1. **Shortest Path** (Unweighted):
   - Minimum steps to reach target
   - Shortest path length
   - Level of each node from source

2. **Level-order Processing**:
   - Process nodes level by level
   - Find nodes at specific distance
   - Level-order tree traversal

3. **Connected Components**:
   - Find all nodes reachable from source
   - Count connected components
   - Check connectivity

4. **Bipartite Checking**:
   - Color nodes alternately
   - Check if graph is bipartite

**BFS vs DFS:**
- **BFS**: Finds shortest path, uses more memory (queue)
- **DFS**: Uses less memory (stack), doesn't guarantee shortest path`,
					CodeExamples: `// Go: BFS implementation
func BFS(graph [][]int, start int) []int {
    visited := make([]bool, len(graph))
    queue := []int{start}
    visited[start] = true
    result := []int{}
    
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        result = append(result, node)
        
        for _, neighbor := range graph[node] {
            if !visited[neighbor] {
                visited[neighbor] = true
                queue = append(queue, neighbor)
            }
        }
    }
    return result
}

# Python: BFS implementation
from collections import deque

def bfs(graph, start):
    visited = [False] * len(graph)
    queue = deque([start])
    visited[start] = True
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    return result`,
				},
				{
					Title: "Graph Traversal: DFS",
					Content: `Depth-First Search (DFS) explores as far as possible along each branch before backtracking. Like exploring a maze - go as deep as possible before trying another path.

**DFS Algorithm:**
1. Start from source node, mark as visited
2. For each unvisited neighbor:
   - Recursively visit neighbor
   - Or use explicit stack
3. Backtrack when no more unvisited neighbors

**Key Properties:**
- **Stack-based**: Uses LIFO stack (naturally recursive)
- **Deep Exploration**: Explores one path completely before others
- **Memory Efficient**: Only stores path from root to current node
- **Time**: O(V + E) - visit each vertex and edge once
- **Space**: O(V) for recursion stack (or explicit stack)

**DFS Variants:**

1. **Pre-order**: Process node before children
2. **In-order**: Process node between children (trees)
3. **Post-order**: Process node after children

**DFS Applications:**

1. **Path Finding**:
   - Find any path from source to target
   - Check if path exists
   - Enumerate all paths

2. **Cycle Detection**:
   - Detect cycles in directed/undirected graphs
   - Use color marking (white/gray/black)

3. **Topological Sort**:
   - Order nodes such that dependencies come first
   - Used in: build systems, course prerequisites

4. **Connected Components**:
   - Find all nodes in component
   - Count components
   - Check connectivity

5. **Strongly Connected Components**:
   - Kosaraju's or Tarjan's algorithm
   - Find SCCs in directed graphs

**DFS vs BFS:**
- **DFS**: Less memory, doesn't guarantee shortest path
- **BFS**: More memory, guarantees shortest path (unweighted)

**Common Patterns:**
- **Backtracking**: Use DFS with state restoration
- **Memoization**: Cache results during DFS
- **Path Tracking**: Maintain path during traversal`,
					CodeExamples: `// Go: DFS recursive
func DFS(graph [][]int, start int) []int {
    visited := make([]bool, len(graph))
    result := []int{}
    dfsHelper(graph, start, visited, &result)
    return result
}

func dfsHelper(graph [][]int, node int, visited []bool, result *[]int) {
    visited[node] = true
    *result = append(*result, node)
    
    for _, neighbor := range graph[node] {
        if !visited[neighbor] {
            dfsHelper(graph, neighbor, visited, result)
        }
    }
}

# Python: DFS recursive
def dfs(graph, start):
    visited = [False] * len(graph)
    result = []
    
    def dfs_helper(node):
        visited[node] = True
        result.append(node)
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs_helper(neighbor)
    
    dfs_helper(start)
    return result`,
				},
				{
					Title: "Common Graph Algorithms",
					Content: `Essential graph algorithms for problem-solving:

**1. Shortest Path:**
- **BFS**: Unweighted graphs, O(V + E)
- **Dijkstra**: Weighted graphs, non-negative weights, O((V + E) log V)
- **Bellman-Ford**: Handles negative weights, O(V × E)

**2. Minimum Spanning Tree:**
- **Kruskal**: Greedy, O(E log E)
- **Prim**: Greedy, O(E log V)

**3. Topological Sort:**
- Order nodes such that all dependencies come before dependents
- Uses DFS, O(V + E)

**4. Cycle Detection:**
- **Undirected**: DFS with parent tracking
- **Directed**: DFS with color marking (white/gray/black)

**5. Connected Components:**
- Use DFS/BFS to find all connected nodes
- Important for social networks, islands problems`,
					CodeExamples: `// Go: Topological sort
func topologicalSort(graph [][]int) []int {
    visited := make([]bool, len(graph))
    result := []int{}
    
    var dfs func(int)
    dfs = func(node int) {
        visited[node] = true
        for _, neighbor := range graph[node] {
            if !visited[neighbor] {
                dfs(neighbor)
            }
        }
        result = append([]int{node}, result...)  // Prepend
    }
    
    for i := 0; i < len(graph); i++ {
        if !visited[i] {
            dfs(i)
        }
    }
    return result
}

# Python: Cycle detection (undirected)
def has_cycle(graph):
    visited = [False] * len(graph)
    
    def dfs(node, parent):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True  # Back edge found
        return False
    
    for i in range(len(graph)):
        if not visited[i]:
            if dfs(i, -1):
                return True
    return False`,
				},
				{
					Title: "Shortest Path Algorithms",
					Content: `Finding shortest paths in graphs is fundamental to many applications. Different algorithms handle different graph types and constraints.

**Algorithm Types:**

**1. BFS (Unweighted Graphs):**
- Finds shortest path in terms of number of edges
- Time: O(V + E)
- Use when: All edges have same weight (or unweighted)

**2. Dijkstra's Algorithm (Non-negative Weights):**
- Finds shortest path from source to all nodes
- Greedy algorithm using priority queue
- Time: O((V + E) log V) with binary heap
- Use when: Non-negative edge weights

**3. Bellman-Ford Algorithm (Negative Weights):**
- Handles negative edge weights
- Can detect negative cycles
- Time: O(V × E)
- Use when: Negative weights possible, need cycle detection

**4. Floyd-Warshall (All Pairs):**
- Finds shortest paths between all pairs of nodes
- Dynamic programming approach
- Time: O(V³)
- Use when: Need all-pairs shortest paths

**5. A* Algorithm (Heuristic Search):**
- Uses heuristic to guide search
- More efficient than Dijkstra for specific targets
- Time: Depends on heuristic quality

**Key Concepts:**
- **Relaxation**: Update distance if shorter path found
- **Priority Queue**: Always process closest unvisited node
- **Negative Cycles**: Make shortest path undefined
- **Path Reconstruction**: Store parent pointers to reconstruct path`,
					CodeExamples: `// Go: Dijkstra's Algorithm
import "container/heap"

type Item struct {
    node     int
    distance int
}

type MinHeap []Item

func (h MinHeap) Len() int { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].distance < h[j].distance }
func (h MinHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x interface{}) { *h = append(*h, x.(Item)) }
func (h *MinHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0:n-1]
    return x
}

func dijkstra(graph [][]int, weights [][]int, start int) []int {
    n := len(graph)
    dist := make([]int, n)
    for i := range dist {
        dist[i] = int(^uint(0) >> 1)  // Max int
    }
    dist[start] = 0
    
    h := &MinHeap{Item{start, 0}}
    heap.Init(h)
    visited := make([]bool, n)
    
    for h.Len() > 0 {
        item := heap.Pop(h).(Item)
        u := item.node
        
        if visited[u] {
            continue
        }
        visited[u] = true
        
        for _, v := range graph[u] {
            if !visited[v] {
                newDist := dist[u] + weights[u][v]
                if newDist < dist[v] {
                    dist[v] = newDist
                    heap.Push(h, Item{v, dist[v]})
                }
            }
        }
    }
    
    return dist
}

// Go: Bellman-Ford Algorithm
func bellmanFord(edges [][]int, n int, start int) []int {
    dist := make([]int, n)
    for i := range dist {
        dist[i] = int(^uint(0) >> 1)
    }
    dist[start] = 0
    
    // Relax edges V-1 times
    for i := 0; i < n-1; i++ {
        for _, edge := range edges {
            u, v, w := edge[0], edge[1], edge[2]
            if dist[u] != int(^uint(0)>>1) && dist[u]+w < dist[v] {
                dist[v] = dist[u] + w
            }
        }
    }
    
    // Check for negative cycles
    for _, edge := range edges {
        u, v, w := edge[0], edge[1], edge[2]
        if dist[u] != int(^uint(0)>>1) && dist[u]+w < dist[v] {
            // Negative cycle detected
            return nil
        }
    }
    
    return dist
}

# Python: Dijkstra's Algorithm
import heapq

def dijkstra(graph, weights, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    
    heap = [(0, start)]
    visited = [False] * n
    
    while heap:
        d, u = heapq.heappop(heap)
        
        if visited[u]:
            continue
        visited[u] = True
        
        for v in graph[u]:
            if not visited[v]:
                new_dist = dist[u] + weights[u][v]
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(heap, (dist[v], v))
    
    return dist

# Python: Bellman-Ford
def bellman_ford(edges, n, start):
    dist = [float('inf')] * n
    dist[start] = 0
    
    # Relax V-1 times
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    
    # Check for negative cycles
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # Negative cycle
    
    return dist`,
				},
				{
					Title: "Union-Find",
					Content: `Union-Find (also called Disjoint Set Union, DSU) is a data structure that efficiently tracks disjoint sets and supports union and find operations.

**Operations:**
- **Find(x)**: Find representative (root) of set containing x
- **Union(x, y)**: Merge sets containing x and y
- **Connected(x, y)**: Check if x and y are in same set

**Data Structure:**
- Each element points to its parent
- Root element points to itself
- Path compression: Make all nodes point directly to root
- Union by rank: Attach smaller tree under larger tree

**Optimizations:**

**1. Path Compression:**
- When finding root, make all nodes on path point directly to root
- Amortizes to nearly O(1) per operation
- Makes tree flat

**2. Union by Rank:**
- Keep track of tree depth (rank)
- Always attach smaller tree to larger tree
- Prevents tree from becoming too deep

**Time Complexity:**
- Without optimizations: O(n) worst case
- With path compression: O(α(n)) amortized (inverse Ackermann, effectively constant)
- With union by rank: O(α(n)) amortized

**Applications:**
- **Connected Components**: Find all connected components in graph
- **Kruskal's Algorithm**: Minimum spanning tree
- **Network Connectivity**: Check if nodes are connected
- **Image Processing**: Connected component labeling
- **Equivalence Relations**: Partition elements into equivalence classes

**Common Problems:**
- Number of islands
- Friend circles
- Redundant connections
- Accounts merge
- Sentence similarity`,
					CodeExamples: `// Go: Union-Find with path compression and union by rank
type UnionFind struct {
    parent []int
    rank   []int
}

func NewUnionFind(n int) *UnionFind {
    parent := make([]int, n)
    rank := make([]int, n)
    for i := range parent {
        parent[i] = i
    }
    return &UnionFind{parent: parent, rank: rank}
}

func (uf *UnionFind) Find(x int) int {
    if uf.parent[x] != x {
        uf.parent[x] = uf.Find(uf.parent[x])  // Path compression
    }
    return uf.parent[x]
}

func (uf *UnionFind) Union(x, y int) {
    rootX := uf.Find(x)
    rootY := uf.Find(y)
    
    if rootX == rootY {
        return  // Already in same set
    }
    
    // Union by rank
    if uf.rank[rootX] < uf.rank[rootY] {
        uf.parent[rootX] = rootY
    } else if uf.rank[rootX] > uf.rank[rootY] {
        uf.parent[rootY] = rootX
    } else {
        uf.parent[rootY] = rootX
        uf.rank[rootX]++
    }
}

func (uf *UnionFind) Connected(x, y int) bool {
    return uf.Find(x) == uf.Find(y)
}

// Example: Number of connected components
func numConnectedComponents(n int, edges [][]int) int {
    uf := NewUnionFind(n)
    for _, edge := range edges {
        uf.Union(edge[0], edge[1])
    }
    
    components := make(map[int]bool)
    for i := 0; i < n; i++ {
        components[uf.Find(i)] = true
    }
    
    return len(components)
}

# Python: Union-Find
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)

# Example: Number of connected components
def num_connected_components(n, edges):
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    
    components = set()
    for i in range(n):
        components.add(uf.find(i))
    
    return len(components)`,
				},
				{
					Title: "Topological Sort Applications",
					Content: `Topological sort orders nodes in a directed acyclic graph (DAG) such that for every edge (u, v), u appears before v. This has many practical applications.

**Key Properties:**
- Only works on DAGs (Directed Acyclic Graphs)
- Multiple valid topological orders may exist
- If cycle exists, topological sort is impossible

**Algorithms:**

**1. DFS-based (Kahn's Algorithm):**
- Use DFS with post-order processing
- Add node to result after processing all descendants
- Time: O(V + E)

**2. BFS-based (Kahn's Algorithm):**
- Start with nodes having in-degree 0
- Remove edges, add nodes with new in-degree 0
- Time: O(V + E)

**Applications:**

**1. Course Prerequisites:**
- Order courses so prerequisites come first
- Detect circular dependencies

**2. Build Systems:**
- Determine build order for dependencies
- Compile files in correct order

**3. Task Scheduling:**
- Schedule tasks respecting dependencies
- Find valid execution order

**4. Event Ordering:**
- Order events based on dependencies
- Determine chronological order

**5. Package Management:**
- Install packages in dependency order
- Resolve dependency conflicts

**6. Compiler Design:**
- Order symbol definitions
- Resolve forward references

**Common Problems:**
- Course schedule
- Alien dictionary
- Reconstruct itinerary
- Sequence reconstruction
- Minimum height trees`,
					CodeExamples: `// Go: Topological Sort using DFS
func topologicalSortDFS(graph [][]int) []int {
    n := len(graph)
    visited := make([]bool, n)
    result := []int{}
    
    var dfs func(int)
    dfs = func(node int) {
        visited[node] = true
        for _, neighbor := range graph[node] {
            if !visited[neighbor] {
                dfs(neighbor)
            }
        }
        result = append([]int{node}, result...)  // Prepend
    }
    
    for i := 0; i < n; i++ {
        if !visited[i] {
            dfs(i)
        }
    }
    
    return result
}

// Go: Topological Sort using BFS (Kahn's Algorithm)
func topologicalSortBFS(graph [][]int) []int {
    n := len(graph)
    inDegree := make([]int, n)
    
    // Calculate in-degrees
    for i := 0; i < n; i++ {
        for _, neighbor := range graph[i] {
            inDegree[neighbor]++
        }
    }
    
    // Start with nodes having in-degree 0
    queue := []int{}
    for i := 0; i < n; i++ {
        if inDegree[i] == 0 {
            queue = append(queue, i)
        }
    }
    
    result := []int{}
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        result = append(result, node)
        
        // Remove edges and update in-degrees
        for _, neighbor := range graph[node] {
            inDegree[neighbor]--
            if inDegree[neighbor] == 0 {
                queue = append(queue, neighbor)
            }
        }
    }
    
    // Check for cycle
    if len(result) != n {
        return nil  // Cycle exists
    }
    
    return result
}

# Python: Topological Sort using DFS
def topological_sort_dfs(graph):
    n = len(graph)
    visited = [False] * n
    result = []
    
    def dfs(node):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor)
        result.insert(0, node)  # Prepend
    
    for i in range(n):
        if not visited[i]:
            dfs(i)
    
    return result

# Python: Topological Sort using BFS (Kahn's)
def topological_sort_bfs(graph):
    n = len(graph)
    in_degree = [0] * n
    
    # Calculate in-degrees
    for i in range(n):
        for neighbor in graph[i]:
            in_degree[neighbor] += 1
    
    # Start with nodes having in-degree 0
    queue = [i for i in range(n) if in_degree[i] == 0]
    result = []
    
    while queue:
        node = queue.pop(0)
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == n else None  # None if cycle`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          7,
			Title:       "Dynamic Programming",
			Description: "Master dynamic programming techniques to solve complex optimization problems efficiently.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to Dynamic Programming",
					Content: `Dynamic Programming (DP) is a powerful method for solving complex problems by breaking them down into simpler subproblems and storing the results to avoid recomputation.

**Core Concept:**
Instead of solving the same subproblem multiple times (like naive recursion), DP solves each subproblem once and stores the result for future use.

**Key Characteristics:**

1. **Overlapping Subproblems**:
   - Same subproblems appear multiple times
   - Example: Fibonacci - fib(5) needs fib(3) and fib(4), fib(4) needs fib(3) again
   - Without DP: Exponential time O(2^n)
   - With DP: Linear time O(n)

2. **Optimal Substructure**:
   - Optimal solution contains optimal solutions to subproblems
   - Example: Shortest path from A→C via B = shortest A→B + shortest B→C
   - Can combine optimal subproblem solutions

**Two Main Approaches:**

1. **Top-down (Memoization)**:
   - Start with full problem, break down recursively
   - Cache results as you compute them
   - More intuitive, natural recursion
   - Space: O(n) for recursion stack + memo

2. **Bottom-up (Tabulation)**:
   - Start with base cases, build up iteratively
   - Fill table/dp array systematically
   - More efficient (no recursion overhead)
   - Space: Can often optimize to O(1) with careful variable management

**When to Use DP:**
- **Optimization**: "maximum", "minimum", "longest", "shortest"
- **Counting**: "number of ways", "how many paths"
- **Decision**: "can we achieve X?", "is it possible?"
- **Overlapping Subproblems**: Recursive solution repeats work
- **Optimal Substructure**: Can build solution from subproblems

**DP Problem Identification:**
- Problem asks for optimal value (max/min)
- Can be broken into smaller similar problems
- Brute force would be exponential
- Greedy doesn't work (need to consider all possibilities)

**Common DP Patterns:**
- 1D DP: dp[i] = optimal value up to position i
- 2D DP: dp[i][j] = optimal value at position (i,j)
- Knapsack: Choosing items with constraints
- LCS: Longest Common Subsequence
- Edit Distance: Minimum operations to transform`,
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
					Content: `**Common DP Patterns:**

1. **1D DP**: dp[i] represents optimal value up to position i
   - Examples: Climbing stairs, House robber, Coin change
   - Recurrence: dp[i] = f(dp[i-1], dp[i-2], ...)

2. **2D DP**: dp[i][j] represents optimal value at position (i, j)
   - Examples: Unique paths, Edit distance, LCS
   - Recurrence: dp[i][j] = f(dp[i-1][j], dp[i][j-1], ...)

3. **Knapsack Problems**: Choosing items with weight/value constraints
   - 0/1 Knapsack: Each item used at most once
   - Unbounded Knapsack: Items can be reused
   - Fractional Knapsack: Greedy (not DP)

4. **Longest Common Subsequence (LCS)**: Finding longest common pattern
   - Compare two sequences character by character
   - Build solution incrementally

5. **Edit Distance**: Minimum operations to transform one string to another
   - Operations: Insert, Delete, Replace
   - Used in: Spell checkers, DNA alignment, diff algorithms

6. **Interval DP**: dp[i][j] = optimal value for interval from i to j
   - Examples: Matrix chain multiplication, Palindrome partitioning

7. **State Machine DP**: Track state transitions
   - Examples: Buy/sell stock with restrictions, State transitions

**Steps to Solve DP Problems:**

1. **Identify Subproblems**:
   - What smaller problems need to be solved?
   - How do they relate to the main problem?

2. **Define State**:
   - What does dp[i] or dp[i][j] represent?
   - What information do we need to track?

3. **Find Recurrence Relation**:
   - How do we compute dp[i] from previous states?
   - What are the transitions?

4. **Set Base Cases**:
   - What are the smallest subproblems?
   - What are their known values?

5. **Implement**:
   - Choose: Memoization (top-down) or Tabulation (bottom-up)
   - Consider space optimization

6. **Optimize** (if needed):
   - Reduce space complexity
   - Optimize time if possible

**Common Mistakes:**
- Forgetting base cases
- Incorrect recurrence relation
- Off-by-one errors in indices
- Not handling edge cases
- Confusing state definition`,
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
			Description: "Master bitwise operations and bit manipulation techniques for efficient problem-solving.",
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
