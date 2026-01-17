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
			},
			ProblemIDs: []int{51, 52, 53, 54, 55, 56, 57, 58, 59, 60},
		},
	})
}
