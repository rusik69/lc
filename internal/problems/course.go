package problems

func init() {
	allModules = []CourseModule{
		{
			ID:          0,
			Title:       "Introduction & Complexity Analysis",
			Description: "Master the fundamentals: what data structures are, how to analyze algorithms, and understand Big O notation.",
			Order:       0,
			Lessons: []Lesson{
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
			Lessons: []Lesson{
				{
					Title: "Introduction to Arrays",
					Content: `Arrays are one of the most fundamental data structures in computer science. An array is a collection of elements stored in contiguous memory locations, each identified by an index.

**Key Properties:**
- Fixed size (in most languages)
- Random access: O(1) time complexity
- Contiguous memory allocation
- Zero-based indexing

**Common Operations:**
- Access: O(1)
- Search: O(n)
- Insertion: O(n) (requires shifting)
- Deletion: O(n) (requires shifting)`,
					CodeExamples: `// Go example
arr := []int{1, 2, 3, 4, 5}
fmt.Println(arr[0])  // Access: O(1)
arr = append(arr, 6)  // Append: O(1) amortized

# Python example
arr = [1, 2, 3, 4, 5]
print(arr[0])  # Access: O(1)
arr.append(6)  # Append: O(1) amortized`,
				},
				{
					Title: "Hash Tables",
					Content: `Hash tables (also called hash maps or dictionaries) provide O(1) average-case time complexity for insertions, deletions, and lookups.

**How They Work:**
1. A hash function converts keys into array indices
2. Values are stored at those indices
3. Collisions are handled using techniques like chaining or open addressing

**Key Properties:**
- Average O(1) for insert, delete, lookup
- Worst case O(n) if all keys hash to same bucket
- No guaranteed order (unless using ordered hash map)

**Common Use Cases:**
- Counting frequencies
- Fast lookups
- Caching/memoization
- Grouping elements`,
					CodeExamples: `// Go example
m := make(map[int]int)
m[1] = 100  // Insert: O(1)
val := m[1]  // Lookup: O(1)
delete(m, 1)  // Delete: O(1)

# Python example
m = {}
m[1] = 100  # Insert: O(1)
val = m[1]  # Lookup: O(1)
del m[1]  # Delete: O(1)`,
				},
			},
			ProblemIDs: []int{1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
		},
		{
			ID:          2,
			Title:       "Two Pointers",
			Description: "Master the two-pointer technique for solving array and string problems efficiently.",
			Order:       2,
			Lessons: []Lesson{
				{
					Title: "Two Pointers Technique",
					Content: `The two-pointer technique uses two pointers moving through a data structure to solve problems efficiently.

**Common Patterns:**
1. **Opposite Ends**: Start with pointers at both ends, move them toward each other
2. **Same Direction**: Both pointers start at the beginning, move at different speeds
3. **Sliding Window**: Maintain a window of elements between two pointers

**When to Use:**
- Sorted arrays
- Palindrome checking
- Finding pairs/triplets
- Removing duplicates
- Merging sorted arrays

**Time Complexity:** Often reduces O(n²) to O(n)`,
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
			},
			ProblemIDs: []int{2, 3, 4, 5, 21, 22, 23, 24, 25},
		},
		{
			ID:          3,
			Title:       "Sliding Window",
			Description: "Learn to solve problems using the sliding window technique for optimal subarray/substring solutions.",
			Order:       3,
			Lessons: []Lesson{
				{
					Title: "Sliding Window Technique",
					Content: `The sliding window technique maintains a window of elements and slides it across the array/string to find optimal solutions.

**Types:**
1. **Fixed Window**: Window size is constant
2. **Variable Window**: Window size changes based on conditions

**When to Use:**
- Subarray/substring problems
- Finding maximum/minimum in a window
- Problems asking for "longest" or "shortest" subarray
- Problems with constraints (sum, unique characters, etc.)

**Time Complexity:** O(n) - each element visited at most twice`,
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
			Lessons: []Lesson{
				{
					Title: "Stack",
					Content: `A stack is a LIFO (Last In, First Out) data structure.

**Operations:**
- Push: Add element to top
- Pop: Remove element from top
- Peek/Top: View top element without removing
- IsEmpty: Check if stack is empty

**Common Use Cases:**
- Expression evaluation
- Parentheses matching
- Function call stack
- Undo operations
- Backtracking algorithms

**Time Complexity:** All operations O(1)`,
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
					Content: `A queue is a FIFO (First In, First Out) data structure.

**Operations:**
- Enqueue: Add element to rear
- Dequeue: Remove element from front
- Front: View front element
- IsEmpty: Check if queue is empty

**Common Use Cases:**
- BFS (Breadth-First Search)
- Task scheduling
- Print queue
- Request handling

**Time Complexity:** All operations O(1) with proper implementation`,
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
			Lessons: []Lesson{
				{
					Title: "Linked Lists Basics",
					Content: `A linked list is a linear data structure where elements are stored in nodes, each containing data and a reference to the next node.

**Types:**
- Singly Linked List: Each node points to next
- Doubly Linked List: Each node points to next and previous
- Circular Linked List: Last node points back to first

**Advantages:**
- Dynamic size
- Efficient insertion/deletion at any position

**Disadvantages:**
- No random access
- Extra memory for pointers
- Cache unfriendly

**Common Operations:**
- Insert: O(1) at head, O(n) at position
- Delete: O(1) at head, O(n) at position
- Search: O(n)`,
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
			Lessons: []Lesson{
				{
					Title: "Binary Trees",
					Content: `A binary tree is a tree data structure where each node has at most two children: left and right.

**Tree Terminology:**
- Root: Topmost node
- Leaf: Node with no children
- Depth: Distance from root
- Height: Maximum depth
- Subtree: Tree formed by a node and its descendants

**Traversal Methods:**
1. **Inorder**: Left → Root → Right
2. **Preorder**: Root → Left → Right
3. **Postorder**: Left → Right → Root
4. **Level-order**: Breadth-first, level by level

**Time Complexity:** O(n) for all traversals`,
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
					Content: `A Binary Search Tree (BST) is a binary tree where:
- Left subtree contains nodes with values < root
- Right subtree contains nodes with values > root
- Both subtrees are also BSTs

**Properties:**
- Inorder traversal gives sorted order
- Search: O(log n) average, O(n) worst
- Insert: O(log n) average, O(n) worst
- Delete: O(log n) average, O(n) worst`,
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
			},
			ProblemIDs: []int{41, 42, 43, 44, 45, 46, 47, 48, 49, 50},
		},
		{
			ID:          8,
			Title:       "Strings",
			Description: "Master string manipulation techniques, pattern matching, and string algorithms.",
			Order:       8,
			Lessons: []Lesson{
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
			Lessons: []Lesson{
				{
					Title: "Graph Fundamentals",
					Content: `A graph is a collection of nodes (vertices) connected by edges. Graphs model relationships and networks.

**Graph Types:**
1. **Directed Graph**: Edges have direction (A→B ≠ B→A)
2. **Undirected Graph**: Edges have no direction (A-B = B-A)
3. **Weighted Graph**: Edges have weights/costs
4. **Unweighted Graph**: All edges equal

**Graph Representations:**

1. **Adjacency List** (Most common):
   - List of lists: graph[i] = [neighbors of node i]
   - Space: O(V + E)
   - Good for sparse graphs

2. **Adjacency Matrix**:
   - 2D array: matrix[i][j] = edge exists?
   - Space: O(V²)
   - Good for dense graphs, fast edge lookup

**Key Terminology:**
- **Vertex/Node**: A point in the graph
- **Edge**: Connection between vertices
- **Path**: Sequence of connected vertices
- **Cycle**: Path that starts and ends at same vertex
- **Connected**: All vertices reachable from each other`,
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
					Content: `Breadth-First Search (BFS) explores a graph level by level, visiting all neighbors before moving to next level.

**BFS Algorithm:**
1. Start from source node
2. Visit all neighbors at current level
3. Move to next level
4. Use queue to track nodes to visit

**Key Properties:**
- Finds shortest path in unweighted graphs
- Uses queue (FIFO)
- Visits nodes in order of distance from source
- Time: O(V + E), Space: O(V)

**When to Use:**
- Shortest path in unweighted graph
- Level-order traversal
- Finding connected components
- Minimum steps problems`,
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
					Content: `Depth-First Search (DFS) explores as far as possible along each branch before backtracking.

**DFS Algorithm:**
1. Start from source node
2. Explore one path completely
3. Backtrack when no more unvisited neighbors
4. Use stack (recursion or explicit)

**Key Properties:**
- Uses stack (LIFO) - naturally recursive
- Explores deep before wide
- Time: O(V + E), Space: O(V) for recursion stack
- Can find cycles, connected components

**When to Use:**
- Finding paths
- Detecting cycles
- Topological sorting
- Maze solving
- Backtracking problems`,
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
			Lessons: []Lesson{
				{
					Title: "Introduction to Dynamic Programming",
					Content: `Dynamic Programming (DP) is a method for solving complex problems by breaking them down into simpler subproblems and storing the results.

**Key Characteristics:**
- Overlapping subproblems
- Optimal substructure
- Memoization or tabulation

**Approaches:**
1. **Top-down (Memoization)**: Recursive with caching
2. **Bottom-up (Tabulation)**: Iterative, building from base cases

**When to Use:**
- Optimization problems
- Counting problems
- Problems with overlapping subproblems
- Problems asking for "maximum", "minimum", "number of ways"`,
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

1. **1D DP**: dp[i] represents state at position i
2. **2D DP**: dp[i][j] represents state at position (i, j)
3. **Knapsack**: Choosing items with constraints
4. **Longest Common Subsequence**: Finding common patterns
5. **Edit Distance**: Minimum operations to transform

**Steps to Solve:**
1. Identify subproblems
2. Define state
3. Find recurrence relation
4. Set base cases
5. Implement (memoization or tabulation)`,
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
	}
}
