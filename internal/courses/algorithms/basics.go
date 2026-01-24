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
					Content: `Welcome to the world of Data Structures and Algorithms! This is one of the most critical topics in computer science and software engineering. Whether you're a student, a self-taught developer, or an experienced engineer looking to sharpen your skills, mastering these concepts will transform the way you write code.

**Why This Course?**

1.  **Write Better Code:** You'll learn to write code that is not just correct, but also efficient and scalable.
2.  **Solve Hard Problems:** You'll gain the toolkit to tackle complex algorithmic challenges with confidence.
3.  **Career Advancement:** These are the fundamental skills tested in technical interviews at top tech companies.
4.  **System Design:** Understanding data structures is the first step towards designing large-scale distributed systems.

**What We Will Cover:**

*   **Fundamental Data Structures:**
    *   **Arrays & Strings:** The bedrock of data storage.
    *   **Linked Lists:** Understanding pointers and dynamic memory.
    *   **Stacks & Queues:** managing data flow (LIFO/FIFO).
    *   **Trees & Graphs:** Modeling hierarchical and relational data.
    *   **Hash Tables:** The magic of O(1) lookups.

*   **Essential Algorithms:**
    *   **Sorting & Searching:** Organizing and retrieving data efficiently.
    *   **Recursion:** solving problems by breaking them down.
    *   **Dynamic Programming:** Optimizing recursive solutions.
    *   **Greedy Algorithms:** Making locally optimal choices.

**How to Succeed:**

*   **Active Practice:** Don't just read; write code. Implement these structures from scratch.
*   **Visualize:** Draw diagrams. deeply understanding the flow of data is key.
*   **Analyze:** Always ask yourself: "What is the time complexity? What is the space complexity?"

Let's begin this journey together!`,
					CodeExamples: `// Example: Why efficiency matters
// Problem: Find if a number exists in a list

// Naive Approach (Linear Search)
// Imagine searching for a name in an unsorted pile of papers.
// You have to look at every single paper.
func contains(list []int) bool {
    for _, num := range list {
        if num == target {
            return true
        }
    }
    return false
}
// Time Complexity: O(n) - Linear time. 
// If list has 1,000,000 items, we might do 1,000,000 checks.

// Efficient Approach (Binary Search)
// Imagine searching for a name in a sorted phone book.
// You open the middle, see if the name is before or after, and cut the search space in half.
func binarySearch(list []int, target int) bool {
    low, high := 0, len(list)-1
    for low <= high {
        mid := low + (high-low)/2
        if list[mid] == target {
            return true
        }
        if list[mid] < target {
            low = mid + 1
        } else {
            high = mid - 1
        }
    }
    return false
}
// Time Complexity: O(log n) - Logarithmic time.
// If list has 1,000,000 items, we only need ~20 checks!`,
				},
				{
					Title: "What Are Data Structures?",
					Content: `Data structures are fundamental ways of organizing and storing data in a computer so that it can be accessed, modified, and manipulated efficiently. Choosing the right data structure is often the difference between an efficient solution and an inefficient one.

**Understanding Data Structures:**

Data structures are not just containers for data - they define how data is organized in memory and what operations can be performed efficiently. Each data structure has trade-offs between:
- **Time Complexity**: How fast operations are
- **Space Complexity**: How much memory is used
- **Flexibility**: How easily data can be modified
- **Ordering**: Whether data maintains a specific order
- **Cache Locality**: How well data fits in CPU cache (affects real-world performance)

**Why Data Structure Choice Matters:**

**Performance Impact:**
- Wrong choice can make O(n) problem O(n²)
- Right choice can reduce memory usage by 10x
- Cache-friendly structures can be 10-100x faster
- Real-world performance often differs from theoretical due to cache effects

**Memory Efficiency:**
- Arrays: Most memory-efficient (no overhead per element)
- Linked Lists: Extra memory for pointers (8-16 bytes per node)
- Hash Tables: Overhead for buckets and collision handling
- Trees: Overhead for parent/child pointers

**Types of Data Structures:**

**1. Linear Structures (Sequential Access):**

**Arrays:**
- Contiguous memory allocation
- Indexed access (O(1))
- Fixed or dynamic size
- Excellent cache locality (adjacent elements in memory)
- Use when: You need random access, know size in advance, or need cache efficiency
- **Trade-offs**: Fast access but expensive insertions/deletions in middle
- **Memory**: Most efficient, no per-element overhead
- **Real-world**: Used in image processing, matrices, buffers

**Linked Lists:**
- Nodes connected by pointers/references
- Dynamic size, easy insertion/deletion
- Sequential access only
- Poor cache locality (nodes scattered in memory)
- Use when: Size unknown, frequent insertions/deletions at arbitrary positions
- **Trade-offs**: Flexible but slower access, more memory overhead
- **Memory**: Extra 8-16 bytes per node for pointers
- **Real-world**: Used in undo/redo systems, music playlists, browser history

**Stacks:**
- LIFO (Last In, First Out) principle
- Push and pop operations (both O(1))
- Used for: Function calls, expression evaluation, undo operations
- Use when: You need to reverse order or track nested structures
- **Implementation**: Array or linked list (array usually faster)
- **Real-world**: Call stacks, expression parsers, backtracking algorithms

**Queues:**
- FIFO (First In, First Out) principle
- Enqueue and dequeue operations (both O(1))
- Used for: Task scheduling, BFS algorithms, request handling
- Use when: You need to process items in order of arrival
- **Variants**: Circular queues, priority queues, dequeues
- **Real-world**: Print queues, message queues, breadth-first search

**2. Non-Linear Structures (Hierarchical/Network):**

**Trees:**
- Hierarchical relationships (parent-child)
- Binary trees, BSTs, AVL trees, B-trees
- Efficient search, insertion, deletion (O(log n) for balanced trees)
- Used for: File systems, database indexes, expression parsing
- Use when: You need hierarchical data or sorted access
- **Balance matters**: Unbalanced trees degrade to O(n) worst case
- **Real-world**: File systems (directory trees), database indexes (B-trees), compiler ASTs

**Graphs:**
- Network of connected nodes (vertices and edges)
- Directed or undirected, weighted or unweighted
- Used for: Social networks, maps, dependency resolution
- Use when: You need to model relationships and connections
- **Representations**: Adjacency list (sparse) or adjacency matrix (dense)
- **Real-world**: Social networks, GPS navigation, dependency graphs

**Hash Tables (Hash Maps/Dictionaries):**
- Key-value pairs with O(1) average access
- Hash function maps keys to array indices
- Collision resolution needed (chaining or open addressing)
- Used for: Fast lookups, caching, counting frequencies
- Use when: You need fast lookups and don't need ordering
- **Load factor**: Critical for performance (typically 0.75)
- **Real-world**: Database indexes, caches, symbol tables, counting

**3. Specialized Structures:**

**Heaps:**
- Complete binary tree with heap property
- Min-heap or max-heap
- O(log n) insert, O(1) find min/max, O(log n) extract min/max
- Used for: Priority queues, heap sort, finding k largest/smallest
- **Implementation**: Usually array-based (parent at i, children at 2i+1, 2i+2)
- **Real-world**: Task schedulers, event systems, Dijkstra's algorithm

**Tries (Prefix Trees):**
- Tree for storing strings
- Efficient prefix matching
- Used for: Autocomplete, spell checkers, IP routing
- **Space-time trade-off**: Can use lots of memory but very fast lookups
- **Real-world**: Search engines, autocomplete, IP routing tables

**Choosing the Right Data Structure:**

**Decision Framework:**

1.  **What operations do you need?**
    - Random access? → Array
    - Fast lookups? → Hash table
    - Maintain order? → Tree or sorted array
    - Insert/delete frequently? → Linked list or tree
    - Need min/max quickly? → Heap

2.  **What are your constraints?**
    - Memory limited? → Arrays (more compact)
    - Need fast access? → Hash table or array
    - Need ordering? → Tree or sorted array
    - Cache performance critical? → Arrays (better locality)

3.  **What's your access pattern?**
    - Sequential? → Array or linked list
    - Random? → Array or hash table
    - Hierarchical? → Tree
    - Need prefix matching? → Trie

4.  **What's your data size?**
    - Small (< 100 elements): Any structure works, choose for clarity
    - Medium (100-10K): Consider hash tables, balanced trees
    - Large (> 10K): Optimize for cache locality, consider specialized structures

**Real-World Examples:**

- **Database Indexing**: B-trees for range queries, hash indexes for exact lookups
- **Web Caching**: Hash tables for O(1) cache lookups (Redis, Memcached)
- **Compiler Symbol Tables**: Hash tables for variable lookups
- **File Systems**: Trees for directory structure (ext4, NTFS)
- **Social Networks**: Graphs for friend connections (Facebook, LinkedIn)
- **Task Scheduling**: Priority queues (heaps) for task prioritization (OS schedulers)
- **Search Engines**: Tries for autocomplete, inverted indexes for search
- **Game Development**: Spatial data structures (quadtrees, octrees) for collision detection

**Key Principle:** The right data structure can transform an O(n²) algorithm into O(n log n) or even O(n)! Understanding data structures is fundamental to writing efficient code.

**Common Pitfalls:**

**1. Using arrays when you need frequent insertions/deletions:**
- Inserting in middle of array: O(n) - must shift elements
- Better: Use linked list for O(1) insertion
- Exception: If insertions are rare, arrays might still be better due to cache

**2. Using linked lists when you need random access:**
- Accessing element at index i: O(i) - must traverse
- Better: Use array for O(1) access
- Exception: If you maintain iterator, linked list can be efficient

**3. Using hash tables when you need ordered data:**
- Hash tables don't maintain order
- Better: Use balanced tree (BST, AVL) or sorted array
- Exception: If ordering is only needed occasionally, sort on-demand

**4. Not considering cache locality:**
- Arrays have excellent cache locality (adjacent in memory)
- Linked lists have poor cache locality (nodes scattered)
- For performance-critical code, cache effects can dominate
- Solution: Prefer arrays when possible, or use cache-friendly structures

**5. Choosing based on familiarity rather than requirements:**
- Don't use hash table just because you know it
- Analyze your actual access patterns
- Consider all operations, not just the most common one
- Profile if performance is critical

**Best Practices:**

**1. Start with simplest structure that works:**
- Don't over-engineer
- Arrays are often sufficient
- Optimize only when needed

**2. Consider all operations:**
- Don't optimize for one operation at expense of others
- Analyze your complete workload
- Consider worst-case scenarios

**3. Profile before optimizing:**
- Measure actual performance
- Identify real bottlenecks
- Optimize what matters

**4. Understand your data:**
- Size matters (small vs large)
- Access patterns matter (sequential vs random)
- Update frequency matters (read-heavy vs write-heavy)

**5. Use language-specific optimizations:**
- Python: Use sets/dicts (highly optimized)
- Go: Slices are efficient, maps are fast
- Java: ArrayList vs LinkedList (usually ArrayList wins)
- C++: std::vector usually better than std::list`,
					CodeExamples: `// Example: Same problem, different data structures
// Problem: Check if a value exists in a collection

// Approach 1: Using array - O(n) search time
func findInArray(arr []int, target int) bool {
    for _, val := range arr {
        if val == target {
            return true
        }
    }
    return false
}

// Approach 2: Using sorted array with binary search - O(log n) search time
func findInSortedArray(arr []int, target int) bool {
    left, right := 0, len(arr)-1
    for left <= right {
        mid := (left + right) / 2
        if arr[mid] == target {
            return true
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    return false
}

// Approach 3: Using hash map - O(1) average lookup time
func findInMap(m map[int]bool, target int) bool {
    return m[target]  // O(1) average, O(n) worst case
}

// Approach 4: Using set (hash set) - O(1) average lookup
func findInSet(s map[int]struct{}, target int) bool {
    _, exists := s[target]
    return exists
}

// Example: Insertion performance comparison
// Problem: Insert element at arbitrary position

// Array insertion - O(n) worst case
func insertInArray(arr []int, index int, value int) []int {
    // Must shift all elements after index
    arr = append(arr, 0)  // Make room
    copy(arr[index+1:], arr[index:])  // Shift elements
    arr[index] = value
    return arr
}

// Linked list insertion - O(1) if you have node reference
type Node struct {
    Val  int
    Next *Node
}

func insertInLinkedList(prev *Node, value int) *Node {
    newNode := &Node{Val: value, Next: prev.Next}
    prev.Next = newNode
    return newNode
}

// Example: Cache-friendly vs cache-unfriendly access
// Array: Excellent cache locality
func sumArray(arr []int) int {
    sum := 0
    // Sequential access - CPU cache friendly
    for i := 0; i < len(arr); i++ {
        sum += arr[i]
    }
    return sum
}

// Linked list: Poor cache locality
func sumLinkedList(head *Node) int {
    sum := 0
    current := head
    // Random memory access - cache misses likely
    for current != nil {
        sum += current.Val
        current = current.Next
    }
    return sum
}

// Example: Choosing structure based on operations
// Scenario: Need fast lookups AND maintain insertion order

// Wrong: Hash table doesn't maintain order
type WrongOrderedSet struct {
    m map[int]bool
}

// Right: Use both hash table and array
type OrderedSet struct {
    m     map[int]int  // value -> index in array
    items []int        // maintains order
}

func (os *OrderedSet) Add(val int) {
    if _, exists := os.m[val]; !exists {
        os.m[val] = len(os.items)
        os.items = append(os.items, val)
    }
}

func (os *OrderedSet) Contains(val int) bool {
    _, exists := os.m[val]
    return exists  // O(1) lookup
}

func (os *OrderedSet) GetOrdered() []int {
    return os.items  // O(1) - already ordered
}

# Python equivalent - showing different approaches
def find_in_list(arr, target):
    """O(n) - must check every element."""
    return target in arr

def find_in_sorted_list(arr, target):
    """O(log n) - binary search on sorted list."""
    import bisect
    index = bisect.bisect_left(arr, target)
    return index < len(arr) and arr[index] == target

def find_in_set(s, target):
    """O(1) average - hash set lookup."""
    return target in s

def find_in_dict(d, target):
    """O(1) average - dictionary key lookup."""
    return target in d

# Performance comparison
import time

# Test with 1 million elements
large_list = list(range(1000000))
large_set = set(large_list)
target = 999999

# Array search: O(n)
start = time.time()
result1 = find_in_list(large_list, target)
time1 = time.time() - start

# Set search: O(1)
start = time.time()
result2 = find_in_set(large_set, target)
time2 = time.time() - start

print(f"Array search: {time1:.6f} seconds")
print(f"Set search: {time2:.6f} seconds")
print(f"Speedup: {time1/time2:.1f}x faster")

# Example: Real-world data structure choice
# Problem: Count word frequencies in text

# Approach 1: Using list (inefficient)
def count_words_list(text):
    words = text.split()
    unique_words = []
    counts = []
    for word in words:
        if word in unique_words:
            index = unique_words.index(word)  # O(n) search!
            counts[index] += 1
        else:
            unique_words.append(word)
            counts.append(1)
    return dict(zip(unique_words, counts))  # O(n²) overall

# Approach 2: Using dictionary (efficient)
def count_words_dict(text):
    words = text.split()
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1  # O(1) lookup
    return counts  # O(n) overall

# Performance difference is dramatic for large texts!

# Example: Choosing structure for specific use case
# Use case: Implement undo/redo functionality

# Stack is perfect for this
class UndoRedo:
    def __init__(self):
        self.undo_stack = []  # Stack for undo
        self.redo_stack = []  # Stack for redo
    
    def do_action(self, action):
        self.undo_stack.append(action)
        self.redo_stack.clear()  # Clear redo when new action
    
    def undo(self):
        if self.undo_stack:
            action = self.undo_stack.pop()
            self.redo_stack.append(action)
            return action
        return None
    
    def redo(self):
        if self.redo_stack:
            action = self.redo_stack.pop()
            self.undo_stack.append(action)
            return action
        return None

# Example: Memory usage comparison
import sys

# Array: Compact
arr = [1, 2, 3, 4, 5]
print(f"Array size: {sys.getsizeof(arr)} bytes")

# Linked list equivalent (Python list acts like dynamic array)
# But if we had actual linked list nodes:
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

# Each node has overhead: object header + val + next pointer
# Much more memory than array!

# Key takeaway: Data structure choice dramatically affects performance!
# Always consider: time complexity, space complexity, and real-world factors like cache locality`,
				},
				{
					Title: "Complexity Analysis",
					Content: `Complexity Analysis is the heartbeat of efficient software. It allows us to predict how our code will perform as the data grows from 10 users to 10 million.

**The Big O Notation**

Big O notation describes the **worst-case scenario** or the **upper bound** of an algorithm's growth rate. It helps us answer: "If I double my input data, how much longer will the program take?"

**Common Time Complexities:**

1.  **O(1) - Constant Time:** 
    *   The best possible complexity.
    *   The operation takes the same amount of time regardless of input size.
    *   *Example:* Accessing a specific index in an array arr[5].

2.  **O(log n) - Logarithmic Time:**
    *   Highly efficient. The time grows very slowly as input increases.
    *   Usually means we are cutting the problem in half at each step.
    *   *Example:* Binary Search.

3.  **O(n) - Linear Time:**
    *   Performance grows directly proportionally to the input.
    *   *Example:* Looping through a list of names.

4.  **O(n log n) - Linearithmic Time:**
    *   Slightly slower than linear, but standard for efficient sorting.
    *   *Example:* Merge Sort, Heap Sort.

5.  **O(n²) - Quadratic Time:**
    *   Performance degrades significantly with large inputs.
    *   Usually involves nested loops (a loop inside a loop).
    *   *Example:* Comparing every item with every other item (Bubble Sort).

6.  **O(2^n) - Exponential Time:**
    *   Very slow. The operation count doubles with each new input element.
    *   *Example:* Recursively calculating Fibonacci numbers (without caching).

**Space Complexity**

Just as we measure time, we must also measure memory.
*   **O(1) Space:** Algorithm uses a fixed amount of extra memory (a few variables).
*   **O(n) Space:** Algorithm needs to store a copy of the input or use a stack/recursion depth proportional to input.

**Why does this matter?**

Imagine you have a website with 1,000 users. An O(n²) algorithm takes 1,000 * 1,000 = 1,000,000 operations.
Now your site grows to 1,000,000 users. That same algorithm now takes 1,000,000,000,000 operations.
Your server crashes. Your users leave.
Understanding complexity is how you build systems that **scale**.`,
					CodeExamples: `// Example: Visualizing complexity
// Input: n (number of items)

// O(1) - Constant
func constant(n int) {
    print("Hello") // Always runs once
}

// O(n) - Linear
func linear(n int) {
    for i := 0; i < n; i++ {
        print(i) // Runs n times
    }
}

// O(n^2) - Quadratic
func quadratic(n int) {
    for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
            print(i, j) // Runs n * n times
        }
    }
}

// O(log n) - Logarithmic
func logarithmic(n int) {
    for i := n; i > 0; i /= 2 {
        print(i) // Runs log2(n) times
    }
    // If n = 8: 8, 4, 2, 1 (4 steps)
    // If n = 16: 16, 8, 4, 2, 1 (5 steps)
    // Doubling n only adds 1 step!
}

// O(n log n) - Linearithmic time
// Common in efficient sorting algorithms
func mergeSort(arr []int) []int {
    if len(arr) <= 1 {
        return arr
    }
    mid := len(arr) / 2
    // Recursive calls: O(log n) levels
    left := mergeSort(arr[:mid])   // Each level processes n elements
    right := mergeSort(arr[mid:])  // Merge: O(n) per level
    return merge(left, right)      // Total: O(n log n)
}

// Space complexity examples
func sumArraySpace(arr []int) int {
    // Space: O(1) - only using a constant amount of extra space
    sum := 0
    for _, val := range arr {
        sum += val
    }
    return sum
}

func copyArray(arr []int) []int {
    // Space: O(n) - creating a new array of size n
    result := make([]int, len(arr))
    copy(result, arr)
    return result
}

# Python: Complexity analysis examples
def constant_time_example(arr):
    """O(1) - constant time."""
    return arr[0] if arr else None

def linear_time_example(arr):
    """O(n) - linear time."""
    total = 0
    for num in arr:  # n iterations
        total += num
    return total

def quadratic_time_example(arr):
    """O(n²) - quadratic time."""
    pairs = []
    for i in range(len(arr)):      # n iterations
        for j in range(i + 1, len(arr)):  # n-i iterations
            pairs.append((arr[i], arr[j]))
    return pairs

def logarithmic_time_example(arr, target):
    """O(log n) - logarithmic time (binary search)."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Comparing complexities visually
import time
import math

def time_complexity_demo():
    """Demonstrate how different complexities scale."""
    sizes = [100, 1000, 10000, 100000]
    
    print("Complexity Comparison:")
    print(f"{'Size':<10} {'O(1)':<15} {'O(log n)':<15} {'O(n)':<15} {'O(n log n)':<15} {'O(n²)':<15}")
    print("-" * 85)
    
    for n in sizes:
        o1 = 1
        olog = math.log2(n)
        on = n
        onlog = n * math.log2(n)
        on2 = n * n
        
        print(f"{n:<10} {o1:<15.2f} {olog:<15.2f} {on:<15.2f} {onlog:<15.2f} {on2:<15.2f}")
    
    # Key insight: Notice how O(n²) grows much faster than others!`,
				},
				{
					Title: "Memory",
					Content: `Understanding how memory works is crucial for writing efficient code and avoiding common performance pitfalls. Memory management affects both time and space complexity, and understanding it helps you make better algorithmic choices.

**Memory Hierarchy:**

Modern computers have a memory hierarchy with different speeds and sizes:

1. **Registers**: Fastest, smallest, inside CPU
2. **Cache (L1, L2, L3)**: Very fast, small (KB to MB), stores frequently accessed data
3. **RAM (Main Memory)**: Fast, larger (GB), stores active programs and data
4. **Disk/SSD**: Slowest, largest (TB), persistent storage

**Memory Basics:**

**Stack:**
- Fast access, limited size (typically 1-8 MB)
- Stores local variables, function parameters, return addresses
- Automatic allocation/deallocation (LIFO)
- Fixed size at compile time
- Stack overflow occurs if recursion is too deep
- Used for: Function calls, local variables, temporary data

**Heap:**
- Larger size (limited by system RAM)
- Stores dynamically allocated data
- Manual management in some languages (C/C++), automatic in others (Go, Python, Java)
- Slower access than stack (requires pointer dereferencing)
- Used for: Objects, arrays with dynamic size, data structures

**Cache:**
- Very fast, very small (typically 32KB-32MB total)
- Stores recently accessed data
- Managed automatically by hardware
- Cache hits are much faster than cache misses
- Critical for performance: cache-friendly code can be 10-100x faster

**Memory Considerations:**

**1. Stack vs Heap:**

**Stack Advantages:**
- Very fast allocation/deallocation
- Automatic memory management
- Predictable memory layout
- No fragmentation

**Stack Disadvantages:**
- Limited size
- Fixed size at compile time
- Stack overflow if too deep recursion

**Heap Advantages:**
- Large size (limited by RAM)
- Dynamic allocation
- Flexible size

**Heap Disadvantages:**
- Slower allocation/deallocation
- May require manual management
- Memory fragmentation possible
- Garbage collection overhead (in managed languages)

**2. Cache Locality:**

**Spatial Locality:**
- Accessing nearby memory locations is faster
- Arrays: Excellent (contiguous memory)
- Linked Lists: Poor (scattered memory)
- This is why arrays often outperform linked lists in practice

**Temporal Locality:**
- Recently accessed data is likely to be accessed again
- Cache stores recently used data
- Algorithms that reuse data benefit from caching

**Real-World Impact:**
- Array traversal: ~1-2 cycles per element (cache hit)
- Linked list traversal: ~10-100 cycles per element (cache miss)
- Cache misses can dominate performance!

**3. Space Complexity:**

**Auxiliary Space:**
- Extra space used by algorithm beyond input
- Focuses on temporary variables, recursion stack, etc.
- Usually what we analyze in space complexity

**Input Space:**
- Space used by input itself
- Often not counted in space complexity
- But important for total memory usage

**Space-Time Tradeoffs:**
- Can often use more space to reduce time
- Example: Hash table uses O(n) space for O(1) lookups
- Can sometimes use more time to reduce space
- Example: Recomputing values instead of storing them

**Memory Optimization Tips:**

**1. Choose Data Structures Wisely:**
- Prefer arrays over linked lists when possible (better cache locality)
- Use hash tables for fast lookups (space-time tradeoff)
- Consider memory layout for performance-critical code

**2. Reuse Memory:**
- Reuse data structures instead of creating new ones
- Pre-allocate arrays when size is known
- Use object pools for frequently allocated objects

**3. Be Mindful of Recursion:**
- Recursion uses stack space (O(depth))
- Deep recursion can cause stack overflow
- Consider iterative solutions for deep recursion
- Tail recursion can be optimized (in some languages)

**4. Avoid Memory Leaks:**
- Free allocated memory (in manual management languages)
- Be careful with circular references (in garbage-collected languages)
- Use weak references when appropriate

**5. Profile Your Code:**
- Measure actual memory usage
- Identify memory hotspots
- Use profiling tools to find leaks

**Common Memory Pitfalls:**

- **Stack Overflow**: Too deep recursion
- **Memory Leaks**: Not freeing allocated memory
- **Cache Misses**: Poor data locality
- **Memory Fragmentation**: Many small allocations
- **Excessive Allocations**: Creating too many temporary objects

**Language-Specific Considerations:**

**Go:**
- Garbage collected (automatic memory management)
- Stack for local variables, heap for pointers
- Escape analysis determines stack vs heap allocation
- Generally efficient, but be mindful of allocations in hot paths

**Python:**
- Everything is an object (on heap)
- Reference counting + cycle detection for garbage collection
- Can be memory-intensive due to object overhead
- Use generators for memory-efficient iteration

**C/C++:**
- Manual memory management
- Stack for local variables, heap for dynamic allocation
- Must manually free heap memory
- More control but more responsibility`,
					CodeExamples: `// Example: Memory considerations in Go

// Good: Reusing array (in-place modification)
// Space: O(1) auxiliary space
func processInPlace(arr []int) {
    for i := range arr {
        arr[i] *= 2  // Modifies existing memory, no new allocation
    }
}

// Less efficient: Creating new array
// Space: O(n) auxiliary space
func processNew(arr []int) []int {
    result := make([]int, len(arr))  // New memory allocation on heap
    for i, val := range arr {
        result[i] = val * 2
    }
    return result
}

// Memory-efficient: Pre-allocating with known capacity
func buildLargeSlice(size int) []int {
    result := make([]int, 0, size)  // Pre-allocate capacity
    for i := 0; i < size; i++ {
        result = append(result, i*2)  // No reallocation needed
    }
    return result
}

// Cache-friendly: Sequential access pattern
func sumArraySequential(arr []int) int {
    sum := 0
    // Sequential access = good cache locality
    for _, val := range arr {
        sum += val  // Cache hit likely
    }
    return sum
}

// Cache-unfriendly: Random access pattern
func sumArrayRandom(arr []int, indices []int) int {
    sum := 0
    // Random access = poor cache locality
    for _, idx := range indices {
        sum += arr[idx]  // Cache miss likely
    }
    return sum
}

// Stack space: Recursive function
// Space complexity: O(n) for call stack
func factorialRecursive(n int) int {
    if n <= 1 {
        return 1
    }
    return n * factorialRecursive(n-1)  // Each call uses stack space
}

// Iterative version: O(1) space
func factorialIterative(n int) int {
    result := 1
    for i := 2; i <= n; i++ {
        result *= i  // No stack space needed
    }
    return result
}

# Python: Memory considerations

# In-place modification (memory efficient)
def process_in_place(arr):
    """Modifies array in place, O(1) auxiliary space."""
    for i in range(len(arr)):
        arr[i] *= 2

# Creating new list (uses more memory)
def process_new(arr):
    """Creates new list, O(n) auxiliary space."""
    return [x * 2 for x in arr]  # List comprehension creates new list

# Memory-efficient: Generator (lazy evaluation)
def process_generator(arr):
    """Generator uses O(1) space, computes on demand."""
    for x in arr:
        yield x * 2  # No list created, just yields values

# Memory-intensive: Creating many small objects
def create_many_objects(n):
    """Creates n objects, each with overhead."""
    return [{"value": i} for i in range(n)]  # Each dict has overhead

# Memory-efficient: Using simple data types
def create_simple_list(n):
    """Creates list of integers, less overhead."""
    return list(range(n))  # More memory efficient

# Example: Memory usage comparison
import sys

arr = list(range(1000))

# In-place modification
arr_copy = arr.copy()
process_in_place(arr_copy)
print(f"In-place: {sys.getsizeof(arr_copy)} bytes")

# New list creation
new_arr = process_new(arr)
print(f"New list: {sys.getsizeof(new_arr)} bytes")

# Generator (no list created)
gen = process_generator(arr)
print(f"Generator: {sys.getsizeof(gen)} bytes (just the generator object)")

# Key insight: In-place operations and generators save memory!`,
				},
				{
					Title: "Big O Notation",
					Content: `Big O notation is the mathematical language used to describe how algorithm performance scales with input size. It's the foundation of algorithm analysis and helps us compare algorithms objectively.

**What Big O Means:**

**Formal Definition:**
- Big O describes the upper bound of an algorithm's growth rate
- f(n) = O(g(n)) means f(n) grows no faster than g(n) (asymptotically)
- Focuses on worst-case performance
- Describes behavior as input size approaches infinity

**Key Characteristics:**
- **Asymptotic Analysis**: Focuses on large input sizes
- **Growth Rate**: Describes how time/space increases with input
- **Upper Bound**: Worst-case performance guarantee
- **Simplified**: Ignores constants and lower-order terms

**Why We Ignore Constants:**
- Constants vary by hardware, compiler, language
- Growth rate matters more than exact numbers for large inputs
- O(100n) and O(n) both grow linearly
- For large n, the growth rate dominates

**Common Complexities (from fastest to slowest):**

**1. O(1) - Constant Time:**
- Execution time doesn't depend on input size
- Always takes the same amount of time
- Examples: Array access, hash table lookup, arithmetic operations
- Ideal but often not achievable

**2. O(log n) - Logarithmic Time:**
- Grows very slowly with input size
- Each step eliminates a fraction of remaining work
- Examples: Binary search, balanced tree operations, exponentiation by squaring
- Excellent for large datasets

**3. O(n) - Linear Time:**
- Grows proportionally with input size
- Time increases linearly as input doubles
- Examples: Iterating through array, linear search, counting elements
- Acceptable for most use cases

**4. O(n log n) - Linearithmic Time:**
- Common in efficient sorting and divide-and-conquer algorithms
- Slightly worse than linear but much better than quadratic
- Examples: Merge sort, heap sort, quick sort (average case), FFT
- Often the best achievable for comparison-based sorting

**5. O(n²) - Quadratic Time:**
- Grows quickly, becomes slow for large inputs
- Time increases quadratically as input doubles
- Examples: Nested loops, bubble sort, insertion sort, matrix multiplication (naive)
- Often indicates need for optimization

**6. O(n³) - Cubic Time:**
- Grows even faster than quadratic
- Examples: Three nested loops, naive matrix multiplication
- Usually too slow for large inputs

**7. O(2ⁿ) - Exponential Time:**
- Grows extremely quickly, often impractical
- Examples: Naive recursive Fibonacci, generating all subsets, brute-force search
- Usually indicates need for better algorithm or dynamic programming

**8. O(n!) - Factorial Time:**
- Grows fastest of all, extremely impractical
- Examples: Generating all permutations, traveling salesman (brute force)
- Only feasible for very small inputs

**Big O Rules:**

**1. Drop Constants:**
- O(2n) → O(n)
- O(100n) → O(n)
- O(n/2) → O(n)
- Constants don't affect growth rate

**2. Drop Lower-Order Terms:**
- O(n² + n) → O(n²)
- O(n² + n log n + n) → O(n²)
- Lower-order terms become negligible for large n

**3. Different Terms for Different Inputs:**
- O(a + b) stays O(a + b) (can't simplify)
- O(a × b) stays O(a × b)
- Must consider each input separately

**4. Multiplication vs Addition:**
- Nested loops: O(n × m) or O(n²) if same size
- Sequential operations: O(n + m) or O(n) if one dominates

**Visual Growth Comparison:**

For n = 1,000,000:
- O(1): 1 operation
- O(log n): ~20 operations
- O(n): 1,000,000 operations
- O(n log n): ~20,000,000 operations
- O(n²): 1,000,000,000,000 operations
- O(2ⁿ): Unimaginably large!

**When Constants Matter:**
- For small inputs, constants can dominate
- O(100n) might be slower than O(n²) for n < 100
- Profile code to understand actual performance
- Big O is asymptotic - real performance depends on constants

**Best, Average, and Worst Case:**
- **Best Case**: Minimum time (often not useful)
- **Average Case**: Expected time (more realistic)
- **Worst Case**: Maximum time (what Big O usually describes)

**Common Misconceptions:**
- Big O doesn't describe exact time, only growth rate
- O(n) doesn't mean "n operations" - it means "proportional to n"
- Constants matter in practice, just not asymptotically
- Best case is rarely what we care about

**Practical Application:**
- Use Big O to compare algorithms
- Choose algorithm based on expected input size
- Understand trade-offs between time and space
- Identify bottlenecks in your code`,
					CodeExamples: `// O(1) - Constant time
// No matter how large the array, this takes the same time
func get(arr []int, i int) int {
    if i < 0 || i >= len(arr) {
        return -1  // Error handling
    }
    return arr[i]  // Single operation, always O(1)
}

// O(log n) - Logarithmic time (binary search)
// Each iteration eliminates half the remaining elements
func binarySearch(arr []int, target int) int {
    left, right := 0, len(arr)-1
    // After k iterations: n/2^k elements remain
    // When n/2^k = 1: k = log₂(n)
    for left <= right {
        mid := (left + right) / 2
        if arr[mid] == target {
            return mid
        } else if arr[mid] < target {
            left = mid + 1  // Search right half
        } else {
            right = mid - 1  // Search left half
        }
    }
    return -1  // Total: O(log n)
}

// O(n) - Linear time
// Time grows proportionally with array size
func findMax(arr []int) int {
    if len(arr) == 0 {
        return -1
    }
    max := arr[0]
    // This loop runs n times (once per element)
    for _, val := range arr {
        if val > max {
            max = val  // Each iteration: O(1)
        }
    }
    return max  // Total: n * O(1) = O(n)
}

// O(n log n) - Linearithmic time (merge sort)
func mergeSort(arr []int) []int {
    if len(arr) <= 1 {
        return arr
    }
    mid := len(arr) / 2
    // Recursive calls: O(log n) levels
    left := mergeSort(arr[:mid])   // Each level processes n elements
    right := mergeSort(arr[mid:])  // Merge: O(n) per level
    return merge(left, right)      // Total: O(n log n)
}

func merge(left, right []int) []int {
    result := make([]int, 0, len(left)+len(right))
    i, j := 0, 0
    // Merge two sorted arrays: O(n)
    for i < len(left) && j < len(right) {
        if left[i] <= right[j] {
            result = append(result, left[i])
            i++
        } else {
            result = append(result, right[j])
            j++
        }
    }
    result = append(result, left[i:]...)
    result = append(result, right[j:]...)
    return result
}

// O(n²) - Quadratic time (bubble sort)
func bubbleSort(arr []int) {
    n := len(arr)
    // Outer loop: n iterations
    for i := 0; i < n; i++ {
        // Inner loop: (n-1) + (n-2) + ... + 1 = n(n-1)/2 iterations
        for j := 0; j < n-i-1; j++ {
            if arr[j] > arr[j+1] {
                arr[j], arr[j+1] = arr[j+1], arr[j]  // Each: O(1)
            }
        }
    }
    // Total: O(n²)
}

// O(2ⁿ) - Exponential time (naive Fibonacci)
func fibonacciNaive(n int) int {
    if n <= 1 {
        return n
    }
    // Each call makes two recursive calls
    // Total calls: O(2ⁿ)
    return fibonacciNaive(n-1) + fibonacciNaive(n-2)
}

// O(n) - Linear time (memoized Fibonacci)
func fibonacciMemo(n int, memo map[int]int) int {
    if n <= 1 {
        return n
    }
    if val, exists := memo[n]; exists {
        return val  // O(1) lookup
    }
    memo[n] = fibonacciMemo(n-1, memo) + fibonacciMemo(n-2, memo)
    return memo[n]  // Each number computed once: O(n)
}

# Python: Big O examples

# O(1) - Constant
def get_item(arr, index):
    return arr[index] if 0 <= index < len(arr) else None

# O(log n) - Logarithmic
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# O(n) - Linear
def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for val in arr:  # n iterations
        if val > max_val:
            max_val = val
    return max_val

# O(n log n) - Linearithmic
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])   # O(log n) levels
    right = merge_sort(arr[mid:])  # O(n) work per level
    return merge(left, right)      # Total: O(n log n)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# O(n²) - Quadratic
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

# Visualizing growth rates
import math

def compare_growth_rates():
    """Compare how different complexities scale."""
    sizes = [10, 100, 1000, 10000, 100000]
    
    print(f"{'Size':<10} {'O(1)':<15} {'O(log n)':<15} {'O(n)':<15} {'O(n log n)':<15} {'O(n²)':<15}")
    print("-" * 85)
    
    for n in sizes:
        o1 = 1
        olog = math.log2(n) if n > 0 else 0
        on = n
        onlog = n * math.log2(n) if n > 0 else 0
        on2 = n * n
        
        print(f"{n:<10} {o1:<15.2f} {olog:<15.2f} {on:<15.2f} {onlog:<15.2f} {on2:<15.2f}")
    
    # Key insight: Notice how O(n²) grows much faster than others!

# compare_growth_rates()`,
				},
				{
					Title: "Logarithm",
					Content: `Logarithms are fundamental mathematical concepts that appear frequently in algorithm analysis, especially in divide-and-conquer algorithms, tree-based data structures, and efficient search algorithms. Understanding logarithms deeply is crucial for analyzing algorithm complexity and recognizing when logarithmic time complexity is achievable.

**What is a Logarithm?**

A logarithm answers the question: "What power must we raise the base to get a certain number?" Formally, if bˣ = n, then log_b(n) = x. The logarithm is the inverse operation of exponentiation.

**Mathematical Definition:**
- log₂(n) = "What power of 2 equals n?"
- log₂(8) = 3 because 2³ = 8
- log₂(16) = 4 because 2⁴ = 16
- log₁₀(100) = 2 because 10² = 100
- log_e(n) = ln(n) (natural logarithm, base e ≈ 2.718)

**Common Bases:**
- **Base 2 (log₂)**: Most common in computer science (binary systems, divide-by-two algorithms)
- **Base 10 (log₁₀)**: Common in general mathematics and scientific notation
- **Base e (ln)**: Natural logarithm, appears in calculus and growth models
- **Base doesn't matter for Big O**: All logarithmic bases are equivalent in Big O notation (differ by constant factor)

**Why Logarithms Matter in Algorithms:**

**1. Divide-and-Conquer Algorithms:**
- Many efficient algorithms work by repeatedly dividing the problem in half
- Each division reduces problem size by factor of 2
- After k divisions: problem size = n / 2ᵏ
- When n / 2ᵏ = 1: k = log₂(n) divisions needed
- Examples: Binary search, merge sort, quick sort (average case)

**2. Tree-Based Data Structures:**
- Balanced binary trees have height ≈ log₂(n)
- Each level doubles the number of nodes
- Height determines search/insert/delete complexity
- Examples: Binary search trees, AVL trees, red-black trees, B-trees

**3. Recursive Algorithms:**
- Recursive algorithms that reduce problem size by constant factor
- Depth of recursion = log(n)
- Examples: Binary search, tree traversals, divide-and-conquer

**4. Efficient Search:**
- Logarithmic search is incredibly efficient
- Can search billions of items in ~30 comparisons
- Much better than linear O(n) search
- Examples: Binary search, searching in sorted arrays

**Key Logarithmic Properties:**

**1. Product Rule:**
- log(ab) = log(a) + log(b)
- Example: log₂(8 × 16) = log₂(8) + log₂(16) = 3 + 4 = 7
- Useful for: Analyzing nested loops, multiplying complexities

**2. Quotient Rule:**
- log(a/b) = log(a) - log(b)
- Example: log₂(16/4) = log₂(16) - log₂(4) = 4 - 2 = 2
- Useful for: Analyzing divide operations

**3. Power Rule:**
- log(aᵇ) = b × log(a)
- Example: log₂(8²) = 2 × log₂(8) = 2 × 3 = 6
- Useful for: Analyzing exponential growth, nested exponentials

**4. Change of Base:**
- log_b(n) = log_c(n) / log_c(b)
- Example: log₂(n) = log₁₀(n) / log₁₀(2) ≈ log₁₀(n) / 0.301
- Useful for: Converting between bases, calculator computations

**5. Base Doesn't Matter for Big O:**
- log₂(n), log₁₀(n), ln(n) all differ by constant factors
- In Big O notation: O(log n) regardless of base
- Constant factors are ignored in asymptotic analysis

**Visual Intuition - How Slowly Logarithms Grow:**

**Real-World Scale Examples:**
- log₂(1,000) ≈ 10 (can search 1,000 items in ~10 steps)
- log₂(1,000,000) ≈ 20 (can search 1 million items in ~20 steps)
- log₂(1,000,000,000) ≈ 30 (can search 1 billion items in ~30 steps)
- log₂(1,000,000,000,000) ≈ 40 (can search 1 trillion items in ~40 steps)

**Key Insight:** Notice how slowly logarithms grow! Doubling the input size only adds 1 to the logarithm. This is why O(log n) algorithms are so incredibly efficient - they scale extremely well.

**Comparison with Other Growth Rates:**
- n = 1,000,000
- log₂(n) ≈ 20 (logarithmic - excellent!)
- √n ≈ 1,000 (square root - good)
- n = 1,000,000 (linear - acceptable)
- n log n ≈ 20,000,000 (linearithmic - good for sorting)
- n² = 1,000,000,000,000 (quadratic - poor for large n)

**Real-World Applications:**

**1. Binary Search:**
- Searching sorted array: O(log n)
- Each comparison eliminates half the remaining elements
- Can find item in sorted array of 1 billion elements in ~30 comparisons
- Used in: Search engines, databases, sorted data structures

**2. Database Indexing:**
- B-trees use logarithmic height for efficient lookups
- Database indexes allow O(log n) record retrieval
- Critical for database performance at scale
- Examples: MySQL indexes, PostgreSQL B-tree indexes

**3. Balanced Tree Operations:**
- AVL trees, red-black trees maintain O(log n) height
- All operations (insert, delete, search) are O(log n)
- Used in: Standard library implementations (Java TreeMap, C++ std::map)
- Examples: Maintaining sorted data with efficient operations

**4. Divide-and-Conquer Sorting:**
- Merge sort: O(n log n) time complexity
- Quick sort (average case): O(n log n)
- Heap sort: O(n log n)
- The log n factor comes from the divide step (tree height)

**5. Exponentiation:**
- Fast exponentiation: O(log n) multiplications
- Compute aⁿ by repeated squaring
- Much faster than naive O(n) multiplication
- Used in: Cryptography, modular arithmetic

**6. Number of Digits:**
- Number of digits in base-b representation: ⌊log_b(n)⌋ + 1
- Example: 1000 in base 10 has 4 digits, log₁₀(1000) = 3, so 3 + 1 = 4
- Useful for: String conversion, number formatting

**7. Binary Representation:**
- Number of bits needed: ⌈log₂(n)⌉
- Example: Represent 1000 requires 10 bits (2¹⁰ = 1024 > 1000)
- Useful for: Bit manipulation, memory allocation

**Common Logarithmic Patterns in Algorithms:**

**1. Binary Search Pattern:**
- Repeatedly divide search space in half
- Each iteration eliminates half the possibilities
- Total iterations: log₂(n)
- Time complexity: O(log n)

**2. Tree Traversal:**
- Balanced tree height: O(log n)
- Traversing from root to leaf: O(log n)
- All tree operations depend on height
- Time complexity: O(log n) for balanced trees

**3. Divide-and-Conquer:**
- Divide problem into smaller subproblems
- Solve recursively
- Combine solutions
- Depth of recursion: O(log n) if dividing by constant factor
- Time complexity: Often O(n log n)

**Best Practices:**

**1. Recognize Logarithmic Opportunities:**
- Sorted data → binary search (O(log n))
- Balanced trees → O(log n) operations
- Divide-by-two patterns → O(log n) depth
- Repeated halving → logarithmic complexity

**2. Understand When Logarithmic is Achievable:**
- Requires structure (sorted, hierarchical, balanced)
- Not always possible (unsorted data, arbitrary graphs)
- Often requires preprocessing (sorting, building tree)

**3. Appreciate the Efficiency:**
- O(log n) is incredibly efficient
- Scales extremely well to large inputs
- Often worth preprocessing to achieve logarithmic access

**Common Pitfalls:**

**1. Confusing log(n) with n:**
- log(n) grows MUCH slower than n
- O(log n) is excellent, O(n) is acceptable
- Don't underestimate logarithmic efficiency

**2. Assuming Logarithmic Without Structure:**
- Unsorted array: O(n) search, not O(log n)
- Unbalanced tree: O(n) worst case, not O(log n)
- Need sorted data or balanced structure for O(log n)

**3. Ignoring Base in Non-Big-O Context:**
- Base matters for actual performance (constants)
- log₂(n) vs log₁₀(n) differs by ~3.3x
- Only ignore base in Big O analysis

**4. Forgetting Preprocessing Cost:**
- Sorting for binary search: O(n log n) preprocessing
- Building balanced tree: O(n log n) construction
- Logarithmic operations assume structure exists

**Performance Characteristics:**

**Why Logarithmic is So Efficient:**
- Grows extremely slowly with input size
- Practical for very large datasets
- 30 comparisons can search 1 billion items
- 40 comparisons can search 1 trillion items
- Essentially constant for practical purposes

**When Logarithmic Complexity Matters:**
- Large datasets (millions to billions of items)
- Frequent search/access operations
- Real-time systems requiring fast lookups
- Systems where preprocessing cost is amortized

**Mathematical Deep Dive:**

**Relationship to Exponentials:**
- Logarithm is inverse of exponentiation
- If 2ˣ = n, then x = log₂(n)
- This inverse relationship is key to understanding logarithmic growth

**Relationship to Binary:**
- log₂(n) = number of bits needed to represent n
- Binary representation directly relates to base-2 logarithm
- Computer systems naturally use base-2 logarithms

**Relationship to Trees:**
- Complete binary tree with n nodes has height ⌊log₂(n)⌋
- Each level doubles the number of nodes
- Height determines path length and operation complexity

**Advanced Concepts:**

**Iterated Logarithm:**
- log*(n) = number of times log must be applied to get ≤ 1
- Grows even slower than log(n)
- Appears in: Union-find with path compression, some graph algorithms

**Polylogarithmic:**
- O(polylog(n)) = O((log n)ᵏ) for some constant k
- Still grows very slowly
- Examples: Parallel algorithms, some graph algorithms

**Logarithmic in Practice:**
- Most efficient search algorithms use logarithmic complexity
- Tree-based data structures achieve logarithmic operations
- Divide-and-conquer algorithms often have logarithmic depth
- Understanding logarithms is essential for algorithm analysis`,
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
					Content: `Arrays are one of the most fundamental data structures in computer science. An array is a collection of elements stored in contiguous memory locations, each identified by an index. Understanding arrays deeply is essential because they form the foundation for many other data structures and algorithms.

**What Are Arrays?**

Arrays are the simplest and most efficient way to store a collection of elements of the same type. They provide direct access to any element by its position (index), making them incredibly fast for random access operations.

**Key Properties:**

**1. Fixed vs Dynamic Size:**
- **Fixed-size arrays**: Size determined at creation, cannot change (C/C++ arrays)
- **Dynamic arrays**: Size can grow/shrink (Go slices, Python lists, Java ArrayList)
- Most modern languages use dynamic arrays by default

**2. Random Access:**
- O(1) time complexity to access any element by index
- Direct memory calculation: address = base + index × element_size
- No need to traverse from beginning like linked lists
- This is arrays' greatest advantage

**3. Contiguous Memory:**
- All elements stored next to each other in memory
- Predictable memory layout
- Excellent cache locality (spatial locality)
- Makes sequential access extremely fast

**4. Zero-Based Indexing:**
- First element at index 0 (most languages)
- Some languages use 1-based indexing (Lua, MATLAB)
- Zero-based is more common and often more efficient

**Memory Layout:**

Arrays store elements sequentially in memory. For an array [10, 20, 30] of integers (4 bytes each):

` + "```" + `
Memory Address    Index    Value
base + 0          0        10
base + 4          1        20
base + 8          2        30
` + "```" + `

**Why This Matters:**
- CPU can predict which memory to load next (prefetching)
- Cache lines load multiple elements at once
- Sequential access is 10-100x faster than random access in linked lists
- Modern CPUs are optimized for sequential memory access

**Common Operations and Their Complexities:**

**Access by Index: O(1)**
- Direct memory calculation
- Single operation regardless of array size
- Fastest possible operation

**Search (Linear): O(n)**
- Must check each element until found
- Average case: n/2 comparisons
- Worst case: n comparisons (element not found)

**Insertion:**
- **At end**: O(1) amortized for dynamic arrays
- **At position**: O(n) - requires shifting all subsequent elements
- **At beginning**: O(n) - worst case, shifts all elements

**Deletion:**
- **From end**: O(1) for dynamic arrays
- **From position**: O(n) - requires shifting elements
- **From beginning**: O(n) - shifts all remaining elements

**Update: O(1)**
- Direct access and modification
- No shifting required

**Dynamic Arrays (Slices/Lists) - How They Work:**

Most modern languages provide dynamic arrays that grow automatically:

**Growth Strategy:**
1. Start with initial capacity (e.g., 4 or 8 elements)
2. When full, allocate larger array (usually 2× current capacity)
3. Copy all existing elements to new array
4. Continue using new array

**Amortized Analysis:**
- Most appends: O(1) (no reallocation needed)
- Occasional append: O(n) (reallocation and copy)
- Average over many operations: O(1) per operation
- This is called "amortized O(1)"

**Example Growth:**
- Capacity 4: [1,2,3,4] → append 5 → capacity 8: [1,2,3,4,5]
- Capacity 8: [1,2,3,4,5,6,7,8] → append 9 → capacity 16: [1,2,3,4,5,6,7,8,9]

**When to Use Arrays:**

**Perfect For:**
- Random access by index
- Sequential processing/iteration
- Fixed or predictable size
- Cache-friendly operations
- Performance-critical code
- Building other data structures (stacks, queues, hash tables)

**Not Ideal For:**
- Frequent insertions/deletions in middle
- Unknown size with many insertions/deletions
- Need for ordered insertion without sorting

**Real-World Applications:**
- **Image Processing**: Pixel arrays
- **Game Development**: Entity arrays, position arrays
- **Scientific Computing**: Numerical arrays (NumPy)
- **Database Systems**: Row storage, indexes
- **Operating Systems**: Process tables, file descriptors

**Common Pitfalls and How to Avoid Them:**

**1. Index Out of Bounds:**
- Always check bounds before accessing
- Use length checks: if i >= 0 && i < len(arr)
- Be careful with negative indices (if supported)

**2. Off-by-One Errors:**
- Remember: valid indices are 0 to len(arr)-1
- Last element: arr[len(arr)-1], not arr[len(arr)]
- Loop bounds: for i := 0; i < len(arr); i++ (not <=)

**3. Array vs Slice Confusion (Go):**
- Arrays: fixed size [5]int
- Slices: dynamic []int
- Slices are references, arrays are values

**4. Memory Management:**
- Large arrays can cause memory issues
- Be mindful of copying vs referencing
- Understand when arrays are copied vs passed by reference

**5. Multidimensional Arrays:**
- Row-major vs column-major order matters for performance
- Cache-friendly access patterns are crucial
- Nested loops order affects performance significantly

**Performance Tips:**

1. **Pre-allocate capacity** when size is known
2. **Use sequential access** when possible (cache-friendly)
3. **Avoid frequent insertions/deletions** in middle
4. **Consider memory layout** for multidimensional arrays
5. **Profile your code** to understand actual performance`,
					CodeExamples: `// Go: Comprehensive array operations

// ===== CREATION =====
// Fixed-size array
var fixedArr [5]int  // [0, 0, 0, 0, 0]

// Dynamic array (slice)
arr := []int{1, 2, 3, 4, 5}  // Slice literal
arr2 := make([]int, 5)        // Make with length
arr3 := make([]int, 0, 10)    // Make with length 0, capacity 10

// ===== ACCESS =====
// O(1) - Direct access by index
first := arr[0]              // 1
last := arr[len(arr)-1]      // 5
middle := arr[len(arr)/2]    // 3

// Bounds checking (important!)
func safeAccess(arr []int, index int) (int, bool) {
    if index < 0 || index >= len(arr) {
        return 0, false  // Out of bounds
    }
    return arr[index], true
}

// ===== SEARCH =====
// O(n) - Linear search
func find(arr []int, target int) int {
    for i, val := range arr {
        if val == target {
            return i
        }
    }
    return -1  // Not found
}

// O(log n) - Binary search (requires sorted array)
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

// ===== INSERTION =====
// O(1) amortized - Append to end
arr = append(arr, 6)  // [1, 2, 3, 4, 5, 6]

// O(n) - Insert at position (requires shifting)
func insert(arr []int, index int, value int) []int {
    // Validate index
    if index < 0 || index > len(arr) {
        return arr
    }
    // Make room
    arr = append(arr, 0)
    // Shift elements right
    copy(arr[index+1:], arr[index:len(arr)-1])
    // Insert value
    arr[index] = value
    return arr
}

// Efficient: Pre-allocate when size is known
func buildArray(size int) []int {
    result := make([]int, 0, size)  // Pre-allocate capacity
    for i := 0; i < size; i++ {
        result = append(result, i*2)  // No reallocation needed
    }
    return result
}

// ===== DELETION =====
// O(1) - Remove from end
arr = arr[:len(arr)-1]  // Remove last element

// O(n) - Remove from position
func remove(arr []int, index int) []int {
    if index < 0 || index >= len(arr) {
        return arr
    }
    // Shift elements left
    copy(arr[index:], arr[index+1:])
    // Truncate
    return arr[:len(arr)-1]
}

// O(n) - Remove by value (removes first occurrence)
func removeValue(arr []int, value int) []int {
    for i, v := range arr {
        if v == value {
            return remove(arr, i)
        }
    }
    return arr
}

// ===== ITERATION =====
// Method 1: Index-based
for i := 0; i < len(arr); i++ {
    fmt.Println(arr[i])
}

// Method 2: Range with index
for i, val := range arr {
    fmt.Printf("Index %d: %d\n", i, val)
}

// Method 3: Range value only
for _, val := range arr {
    fmt.Println(val)
}

// ===== MULTIDIMENSIONAL ARRAYS =====
// 2D array (matrix)
matrix := [][]int{
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
}

// Access: O(1)
value := matrix[1][2]  // 6

// Iteration: O(n×m) for n×m matrix
for i := 0; i < len(matrix); i++ {
    for j := 0; j < len(matrix[i]); j++ {
        fmt.Printf("%d ", matrix[i][j])
    }
    fmt.Println()
}

// Cache-friendly: Row-major order (faster)
func sumMatrixRowMajor(matrix [][]int) int {
    sum := 0
    for i := 0; i < len(matrix); i++ {
        for j := 0; j < len(matrix[i]); j++ {
            sum += matrix[i][j]  // Sequential memory access
        }
    }
    return sum
}

// Cache-unfriendly: Column-major order (slower)
func sumMatrixColumnMajor(matrix [][]int) int {
    sum := 0
    for j := 0; j < len(matrix[0]); j++ {
        for i := 0; i < len(matrix); i++ {
            sum += matrix[i][j]  // Non-sequential memory access
        }
    }
    return sum
}

# Python: Comprehensive list operations

# ===== CREATION =====
arr = [1, 2, 3, 4, 5]           # List literal
arr2 = list(range(5))           # [0, 1, 2, 3, 4]
arr3 = [0] * 10                 # [0, 0, 0, ..., 0] (10 zeros)

# ===== ACCESS =====
# O(1) - Direct access
first = arr[0]                  # 1
last = arr[-1]                  # 5 (negative indexing)
middle = arr[len(arr)//2]       # 3

# Slicing: O(k) where k is slice size
subset = arr[1:4]               # [2, 3, 4]
first_three = arr[:3]           # [1, 2, 3]
last_two = arr[-2:]            # [4, 5]

# ===== SEARCH =====
# O(n) - Linear search
def find(arr, target):
    try:
        return arr.index(target)
    except ValueError:
        return -1

# O(n) - Count occurrences
count = arr.count(3)            # Count how many times 3 appears

# O(n) - Check membership
if 3 in arr:                    # True
    print("Found!")

# ===== INSERTION =====
# O(1) amortized - Append to end
arr.append(6)                   # [1, 2, 3, 4, 5, 6]

# O(k) - Extend with another list
arr.extend([7, 8, 9])          # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# O(n) - Insert at position
arr.insert(2, 99)              # Insert 99 at index 2

# ===== DELETION =====
# O(n) - Remove by value (removes first occurrence)
arr.remove(99)                 # Remove first 99 found

# O(1) - Remove from end
last = arr.pop()               # Remove and return last element

# O(n) - Remove from position
item = arr.pop(2)              # Remove and return element at index 2

# O(n) - Delete slice
del arr[1:3]                   # Delete elements from index 1 to 2

# ===== ITERATION =====
# Method 1: Direct iteration
for val in arr:
    print(val)

# Method 2: With index
for i, val in enumerate(arr):
    print(f"Index {i}: {val}")

# Method 3: List comprehension (creates new list)
squared = [x**2 for x in arr]  # [1, 4, 9, 16, 25, ...]

# ===== PERFORMANCE COMPARISON =====
import time

# Pre-allocation vs dynamic growth
def test_append_performance():
    # Without pre-allocation
    start = time.time()
    arr1 = []
    for i in range(1000000):
        arr1.append(i)
    time1 = time.time() - start
    
    # With pre-allocation (list comprehension)
    start = time.time()
    arr2 = [0] * 1000000
    for i in range(1000000):
        arr2[i] = i
    time2 = time.time() - start
    
    print(f"Dynamic: {time1:.4f}s")
    print(f"Pre-allocated: {time2:.4f}s")
    # Pre-allocation is usually faster!

# Key insights:
# 1. Append is O(1) amortized but occasional O(n) reallocation
# 2. Insert is always O(n) - avoid if possible
# 3. Pre-allocate when size is known
# 4. Use list comprehensions for creating new lists
# 5. Slicing creates new lists (O(k) time and space)`,
				},
				{
					Title: "Hash Tables",
					Content: `Hash tables (also called hash maps or dictionaries) are one of the most powerful and widely used data structures. They provide O(1) average-case time complexity for insertions, deletions, and lookups, making them essential for efficient algorithms.

**What Are Hash Tables?**

Hash tables store key-value pairs and allow fast retrieval of values by their keys. The "hash" refers to the hash function that converts keys into array indices, enabling direct access to values.

**How They Work:**

**1. Hash Function:**
The hash function is the heart of a hash table. It converts keys into array indices:
- **Input**: Key (can be string, int, custom type, etc.)
- **Output**: Index in underlying array (bucket index)
- **Goal**: Distribute keys uniformly across buckets

**Example Hash Functions:**
- **Integer keys**: hash(key) = key % bucket_count
- **String keys**: Sum of character codes, or polynomial rolling hash
- **Custom types**: Combine hash of component fields

**2. Storage Structure:**
- Underlying array of buckets/slots
- Each bucket can hold one or more key-value pairs
- Size typically chosen as prime number to reduce collisions

**3. Collision Handling:**
When two different keys hash to the same index (collision), we need a strategy:

**Separate Chaining:**
- Each bucket contains a linked list (or array) of key-value pairs
- When collision occurs, add to the list in that bucket
- Simple and handles many collisions well
- Extra memory overhead for pointers/links
- Lookup: O(1 + α) where α is average chain length

**Open Addressing:**
- Store key-value pairs directly in the array
- When collision occurs, find next available slot using probing
- **Linear Probing**: Check next slot: (hash(key) + i) % size
- **Quadratic Probing**: Check slots at increasing distances: (hash(key) + i^2) % size
- **Double Hashing**: Use second hash function: (hash1(key) + i * hash2(key)) % size
- Better cache performance (contiguous memory)
- Can suffer from clustering (clumps of filled slots)
- Requires careful deletion handling (tombstone markers)

**Hash Function Properties:**

**1. Deterministic:**
- Same key must always produce same hash value
- Critical for correctness - if hash changes, value becomes unfindable

**2. Uniform Distribution:**
- Keys should spread evenly across all buckets
- Reduces collisions and keeps performance near O(1)
- Poor distribution leads to clustering and O(n) worst case

**3. Fast Computation:**
- Should be O(1) or very fast (O(k) for string of length k)
- Called frequently, so speed matters
- Avoid complex calculations in hash function

**4. Avalanche Effect:**
- Small change in input should cause large change in output
- Prevents similar keys from clustering
- Important for security (in cryptographic hashing)

**Key Properties:**

**Time Complexity:**
- **Average Case**: O(1) for insert, delete, lookup
- **Best Case**: O(1) - no collisions
- **Worst Case**: O(n) - all keys hash to same bucket (extremely rare with good hash function)

**Space Complexity:**
- O(n) space for n key-value pairs
- Additional overhead for collision handling structures
- Load factor affects space efficiency

**Load Factor:**
- Ratio of elements to buckets: alpha = n / m where n = elements, m = buckets
- Typically kept below 0.75 (75% full)
- When exceeded, resize (rehash) to maintain performance
- Higher load factor = more collisions = slower performance

**Ordering:**
- **Unordered**: Most hash tables don't guarantee order (Go maps, Java HashMap)
- **Ordered**: Some maintain insertion order (Python 3.7+ dicts, Java LinkedHashMap)
- **Sorted**: TreeMap maintains sorted order (but O(log n) operations)

**Collision Resolution Strategies - Deep Dive:**

**Separate Chaining Advantages:**
- Simple to implement
- Handles many collisions gracefully
- Easy to delete (just remove from list)
- Can store more elements than bucket count

**Separate Chaining Disadvantages:**
- Extra memory for pointers/links
- Poor cache locality (scattered memory)
- Can degrade to O(n) if many collisions

**Open Addressing Advantages:**
- Better cache performance (contiguous memory)
- No extra memory for pointers
- Simpler memory management

**Open Addressing Disadvantages:**
- Clustering can occur (clumps of filled slots)
- Deletion is complex (need tombstone markers)
- Load factor must be kept lower (< 0.5 typically)
- Performance degrades as table fills

**Common Use Cases:**

**1. Frequency Counting:**
Count how many times each element appears:
- Word frequency in text
- Character frequency in string
- Element frequency in array

**2. Fast Lookups:**
Replace O(n) linear search with O(1) hash lookup:
- Database indexes
- Symbol tables in compilers
- Cache implementations

**3. Caching/Memoization:**
Store computed results to avoid recomputation:
- Function memoization
- Web page caching
- DNS caching

**4. Grouping:**
Group elements by a key:
- Group students by grade
- Group transactions by user
- Group words by length

**5. Deduplication:**
Remove duplicates efficiently:
- Unique elements in array
- Unique words in document
- Unique users in system

**6. Set Operations:**
Implement set data structure:
- Fast membership testing
- Set union, intersection, difference

**Real-World Applications:**

**Database Systems:**
- Hash indexes for exact lookups
- Faster than B-tree for equality queries
- Used in MySQL, PostgreSQL for specific queries

**Caching Systems:**
- Redis, Memcached use hash tables
- Fast O(1) cache lookups
- Critical for web performance

**Compilers:**
- Symbol tables for variable lookups
- Fast identifier resolution
- Used in every compiler/interpreter

**Web Applications:**
- Session management (session ID → user data)
- URL routing (path → handler function)
- Request caching

**Operating Systems:**
- File descriptor tables
- Process ID mappings
- Virtual memory page tables

**Best Practices:**

**1. Choose Initial Capacity:**
- Estimate expected number of elements
- Choose capacity slightly larger than expected
- Use prime numbers for better distribution
- Avoid frequent resizing

**2. Monitor Load Factor:**
- Keep load factor below threshold (0.75 typical)
- Resize when threshold exceeded
- Resize to approximately 2× current size
- Rehash all elements after resize

**3. Use Immutable Keys:**
- Keys shouldn't change after insertion
- Changing key breaks hash function
- Use immutable types (strings, integers) when possible
- For custom types, ensure hash doesn't change

**4. Thread Safety:**
- Standard hash tables are not thread-safe
- Use concurrent hash maps for multi-threaded code
- Or use synchronization (locks, mutexes)
- Consider read-write locks for read-heavy workloads

**5. Hash Function Selection:**
- Use language's built-in hash for standard types
- For custom types, combine hashes of fields
- Test hash distribution with your data
- Avoid predictable hash functions for security-sensitive applications

**Common Pitfalls:**

**1. Mutable Keys:**
- Changing key after insertion makes it unfindable
- Always use immutable keys or be very careful

**2. Poor Hash Function:**
- Can lead to many collisions
- Test with your actual data
- Use well-tested hash functions

**3. High Load Factor:**
- Performance degrades as table fills
- Monitor and resize proactively
- Don't wait until performance is terrible

**4. Not Handling Missing Keys:**
- Always check if key exists before using value
- Use safe access patterns
- Provide default values when appropriate`,
					CodeExamples: `// Go: Comprehensive hash table operations

// ===== CREATION =====
// Empty map
m := make(map[int]int)

// Map literal
m2 := map[string]int{
    "apple":  5,
    "banana": 3,
    "cherry": 8,
}

// With initial capacity hint (doesn't guarantee capacity)
m3 := make(map[int]string, 100)  // Hint: expect ~100 elements

// ===== INSERTION =====
// O(1) average - Insert key-value pairs
m[1] = 100
m[2] = 200
m[3] = 300

// Overwriting existing key
m[1] = 150  // Updates value, still O(1)

// ===== LOOKUP =====
// O(1) average - Safe lookup with existence check
val, exists := m[1]  // val=150, exists=true
val, exists = m[99]  // val=0 (zero value), exists=false

// Unsafe lookup (returns zero value if not found)
val = m[99]  // Returns 0, but we don't know if key exists

// Check existence only
_, exists = m[1]  // exists=true

// ===== DELETION =====
// O(1) average - Delete key-value pair
delete(m, 1)  // Removes key 1

// Safe deletion (check first)
if _, exists := m[2]; exists {
    delete(m, 2)
}

// ===== ITERATION =====
// O(n) - Iterate over all key-value pairs
// Order is randomized (not guaranteed)
for key, value := range m {
    fmt.Printf("Key: %d, Value: %d\n", key, value)
}

// Iterate keys only
for key := range m {
    fmt.Println(key)
}

// Iterate values only
for _, value := range m {
    fmt.Println(value)
}

// ===== COMMON PATTERNS =====

// Pattern 1: Frequency Counting
func countFreq(nums []int) map[int]int {
    freq := make(map[int]int)
    for _, num := range nums {
        freq[num]++  // O(1) per operation
    }
    return freq  // Total: O(n)
}

// Pattern 2: Grouping
func groupByLength(words []string) map[int][]string {
    groups := make(map[int][]string)
    for _, word := range words {
        length := len(word)
        groups[length] = append(groups[length], word)
    }
    return groups
}

// Pattern 3: Deduplication
func unique(nums []int) []int {
    seen := make(map[int]bool)
    result := []int{}
    for _, num := range nums {
        if !seen[num] {
            seen[num] = true
            result = append(result, num)
        }
    }
    return result
}

// Pattern 4: Two Sum (find two numbers that sum to target)
func twoSum(nums []int, target int) []int {
    seen := make(map[int]int)  // value -> index
    for i, num := range nums {
        complement := target - num
        if idx, exists := seen[complement]; exists {
            return []int{idx, i}
        }
        seen[num] = i
    }
    return nil  // No solution
}

// Pattern 5: Memoization (caching function results)
var fibMemo = make(map[int]int)

func fibonacciMemoized(n int) int {
    if n <= 1 {
        return n
    }
    if val, exists := fibMemo[n]; exists {
        return val  // Return cached result
    }
    fibMemo[n] = fibonacciMemoized(n-1) + fibonacciMemoized(n-2)
    return fibMemo[n]
}

// Pattern 6: Set Operations
type IntSet map[int]bool

func (s IntSet) Add(val int) {
    s[val] = true
}

func (s IntSet) Contains(val int) bool {
    return s[val]
}

func (s IntSet) Remove(val int) {
    delete(s, val)
}

func (s IntSet) Union(other IntSet) IntSet {
    result := make(IntSet)
    for val := range s {
        result.Add(val)
    }
    for val := range other {
        result.Add(val)
    }
    return result
}

// ===== PERFORMANCE CONSIDERATIONS =====
// Pre-allocate when size is known
func efficientMap(size int) map[int]int {
    return make(map[int]int, size)  // Hint for initial capacity
}

// Check size
func mapSize(m map[int]int) int {
    return len(m)  // O(1) operation
}

# Python: Comprehensive dictionary operations

# ===== CREATION =====
# Empty dictionary
m = {}

# Dictionary literal
m2 = {
    "apple": 5,
    "banana": 3,
    "cherry": 8,
}

# Dictionary comprehension
m3 = {i: i**2 for i in range(10)}  # {0: 0, 1: 1, 4: 4, 9: 9, ...}

# From two lists (keys and values)
keys = ["a", "b", "c"]
values = [1, 2, 3]
m4 = dict(zip(keys, values))  # {"a": 1, "b": 2, "c": 3}

# ===== INSERTION =====
# O(1) average - Insert key-value pairs
m[1] = 100
m[2] = 200
m[3] = 300

# Update multiple keys
m.update({4: 400, 5: 500})  # Add/update multiple at once

# Set default if key doesn't exist
m.setdefault(6, 600)  # Sets 6:600 if 6 doesn't exist

# ===== LOOKUP =====
# O(1) average - Safe lookup with default
val = m.get(1)        # 100, returns None if not found
val = m.get(99, 0)    # 0 (default value if not found)

# Direct access (raises KeyError if not found)
try:
    val = m[99]
except KeyError:
    val = None

# Check existence
if 1 in m:
    print("Key exists")

# ===== DELETION =====
# O(1) average - Delete key-value pair
del m[1]              # Raises KeyError if key doesn't exist
m.pop(2)              # Remove and return value
m.pop(99, None)       # Safe pop with default (doesn't raise error)
m.popitem()           # Remove and return arbitrary (key, value) pair

# Clear all
m.clear()             # Remove all items

# ===== ITERATION =====
# O(n) - Iterate over key-value pairs (order preserved in Python 3.7+)
for key, value in m.items():
    print(f"{key}: {value}")

# Iterate keys only
for key in m:
    print(key)

# Iterate values only
for value in m.values():
    print(value)

# ===== COMMON PATTERNS =====

# Pattern 1: Frequency Counting
def count_freq(nums):
    freq = {}
    for num in nums:
        freq[num] = freq.get(num, 0) + 1  # O(1) per operation
    return freq  # Total: O(n)

# Using Counter (more Pythonic)
from collections import Counter
freq = Counter([1, 2, 2, 3, 3, 3])  # {1: 1, 2: 2, 3: 3}

# Pattern 2: Grouping
def group_by_length(words):
    groups = {}
    for word in words:
        length = len(word)
        if length not in groups:
            groups[length] = []
        groups[length].append(word)
    return groups

# Using defaultdict (more Pythonic)
from collections import defaultdict
groups = defaultdict(list)
for word in words:
    groups[len(word)].append(word)

# Pattern 3: Deduplication
def unique(nums):
    return list(dict.fromkeys(nums))  # Preserves order

# Pattern 4: Two Sum
def two_sum(nums, target):
    seen = {}  # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return None

# Pattern 5: Memoization
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Manual memoization
fib_memo = {}
def fibonacci_manual(n):
    if n <= 1:
        return n
    if n in fib_memo:
        return fib_memo[n]
    fib_memo[n] = fibonacci_manual(n-1) + fibonacci_manual(n-2)
    return fib_memo[n]

# Pattern 6: Set Operations (using dict as set)
def set_operations():
    set1 = {1, 2, 3, 4, 5}
    set2 = {4, 5, 6, 7, 8}
    
    # Union
    union = set1 | set2  # {1, 2, 3, 4, 5, 6, 7, 8}
    
    # Intersection
    intersection = set1 & set2  # {4, 5}
    
    # Difference
    difference = set1 - set2  # {1, 2, 3}
    
    # Symmetric difference
    sym_diff = set1 ^ set2  # {1, 2, 3, 6, 7, 8}

# ===== PERFORMANCE TIPS =====
# Dictionary comprehension is faster than loop + assignment
# Good:
squares = {i: i**2 for i in range(1000)}

# Less efficient:
squares = {}
for i in range(1000):
    squares[i] = i**2

# Use dict.get() for safe access
# Good:
value = m.get(key, default)

# Less efficient:
value = m[key] if key in m else default

# Key insights:
# 1. Average O(1) operations make hash tables extremely fast
# 2. Use for fast lookups, frequency counting, grouping
# 3. Python dicts maintain insertion order (3.7+)
# 4. Prefer dict.get() over direct access for safety
# 5. Use defaultdict or Counter for common patterns`,
				},
				{
					Title: "Array Techniques",
					Content: `Mastering essential array manipulation techniques is crucial for solving many algorithm problems efficiently. These patterns appear frequently in coding interviews and real-world applications.

**1. Prefix Sum (Cumulative Sum):**

Prefix sum is a powerful technique for range queries. Instead of calculating sums repeatedly, we precompute cumulative sums once.

**How It Works:**
- Precompute: prefix[i] = sum of elements from 0 to i-1
- Range query: sum from i to j = prefix[j+1] - prefix[i]
- Preprocessing: O(n) time, O(n) space
- Query: O(1) time per query

**When to Use:**
- Multiple range sum queries on same array
- Subarray sum problems
- Range update queries (with difference array)
- 2D range queries (2D prefix sum)

**Example Applications:**
- Find subarray with given sum
- Maximum subarray sum (Kadane's algorithm variant)
- Range sum queries in databases
- Image processing (sum of pixel values in rectangle)

**2. Two-Pass Techniques:**

Many problems require two passes through the array - first to collect information, second to use it.

**Pattern:**
1. **First Pass**: Collect information (left products, maximums, etc.)
2. **Second Pass**: Use collected information to solve problem

**Common Examples:**
- **Product Except Self**: First pass calculates left products, second calculates right products
- **Trapping Rainwater**: First pass finds left max, second finds right max
- **Next Greater Element**: First pass builds stack, second processes elements

**Advantages:**
- Often O(n) time complexity
- Can solve problems that seem to require nested loops
- Clear separation of concerns

**3. In-Place Modifications:**

Modify array without using extra space (or minimal extra space). Use the array itself to store information.

**Techniques:**
- **Two Pointers**: One pointer for reading, one for writing
- **Swap Elements**: Exchange elements to rearrange
- **Mark and Sweep**: Use special values to mark positions
- **Negative Marking**: Use negative values to mark visited (if all positive)

**Common Problems:**
- Remove duplicates in sorted array
- Move zeros to end
- Remove element by value
- Sort array with two colors (Dutch National Flag)

**Key Insight:** When space is constrained, use the array itself as storage.

**4. Array Rotation:**

Rotate array left or right by k positions efficiently.

**Methods:**

**Reverse Method (Most Elegant):**
1. Reverse entire array
2. Reverse first k elements
3. Reverse remaining elements
- Time: O(n), Space: O(1)

**Cyclic Replacement:**
- Move elements in cycles
- Each element moves to its final position
- Handle cycles that don't cover all elements
- Time: O(n), Space: O(1)

**Juggling Algorithm:**
- Uses GCD to determine number of cycles
- More complex but efficient
- Time: O(n), Space: O(1)

**5. Common Patterns:**

**Partitioning:**
- Separate elements based on condition
- Use two pointers (or three for three-way partition)
- Examples: Partition around pivot, separate even/odd, Dutch National Flag

**Cyclic Replacements:**
- Move elements in cycles to their correct positions
- Useful for rotation, permutation problems
- Track visited elements to avoid infinite loops

**Reversal:**
- Reverse entire array or segments
- Useful for rotation, palindrome checking
- Can be done in-place with two pointers

**6. Sliding Window:**

Maintain a window of elements and slide it through the array.
- Fixed window: Find maximum/minimum in each window
- Variable window: Find subarray satisfying condition
- Time: O(n) for single pass

**7. Kadane's Algorithm:**

Find maximum subarray sum efficiently.
- Single pass through array
- Track maximum sum ending at current position
- Update global maximum
- Time: O(n), Space: O(1)

**8. Moore's Voting Algorithm:**

Find majority element (appears more than n/2 times).
- Single pass algorithm
- Cancel out pairs of different elements
- Remaining element is candidate
- Verify candidate in second pass

**Best Practices:**

1. **Think About Space-Time Tradeoffs:**
   - Can you use extra space to reduce time?
   - Can you reduce space by using array itself?

2. **Consider Multiple Passes:**
   - Don't always try to solve in one pass
   - Two passes might be clearer and still O(n)

3. **Use Two Pointers:**
   - Often more efficient than nested loops
   - Reduces time complexity significantly

4. **Precompute When Possible:**
   - Prefix sums, suffix arrays, etc.
   - One-time cost for many queries

5. **In-Place When Space Matters:**
   - Modify array directly when possible
   - Use array itself to store information`,
					CodeExamples: `// Go: Comprehensive array techniques

// ===== PREFIX SUM =====
// Precompute cumulative sums
func prefixSum(nums []int) []int {
    prefix := make([]int, len(nums)+1)
    for i := 0; i < len(nums); i++ {
        prefix[i+1] = prefix[i] + nums[i]  // Cumulative sum
    }
    return prefix
}

// O(1) range sum query
func rangeSum(prefix []int, i, j int) int {
    // Sum from index i to j (inclusive)
    return prefix[j+1] - prefix[i]
}

// Example: Find subarray with sum k
func subarraySum(nums []int, k int) int {
    prefix := prefixSum(nums)
    count := 0
    sumMap := make(map[int]int)
    sumMap[0] = 1  // Empty subarray has sum 0
    
    for i := 1; i < len(prefix); i++ {
        // If prefix[i] - k exists, we found a subarray
        if val, exists := sumMap[prefix[i]-k]; exists {
            count += val
        }
        sumMap[prefix[i]]++
    }
    return count
}

// ===== TWO-PASS TECHNIQUES =====
// Product except self (O(n) time, O(1) extra space)
func productExceptSelf(nums []int) []int {
    n := len(nums)
    result := make([]int, n)
    
    // First pass: Calculate left products
    result[0] = 1
    for i := 1; i < n; i++ {
        result[i] = result[i-1] * nums[i-1]
    }
    
    // Second pass: Multiply by right products
    right := 1
    for i := n - 1; i >= 0; i-- {
        result[i] *= right
        right *= nums[i]
    }
    return result
}

// Trapping rainwater (two passes)
func trap(height []int) int {
    n := len(height)
    if n == 0 {
        return 0
    }
    
    // First pass: Find left max for each position
    leftMax := make([]int, n)
    leftMax[0] = height[0]
    for i := 1; i < n; i++ {
        leftMax[i] = max(leftMax[i-1], height[i])
    }
    
    // Second pass: Find right max and calculate water
    rightMax := height[n-1]
    water := 0
    for i := n - 2; i >= 0; i-- {
        rightMax = max(rightMax, height[i])
        // Water trapped = min(leftMax, rightMax) - height
        water += min(leftMax[i], rightMax) - height[i]
    }
    return water
}

// ===== IN-PLACE MODIFICATIONS =====
// Remove duplicates in sorted array (in-place)
func removeDuplicates(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    writeIdx := 1
    for i := 1; i < len(nums); i++ {
        if nums[i] != nums[i-1] {
            nums[writeIdx] = nums[i]
            writeIdx++
        }
    }
    return writeIdx
}

// Move zeros to end (in-place)
func moveZeros(nums []int) {
    writeIdx := 0
    // Move all non-zeros to front
    for i := 0; i < len(nums); i++ {
        if nums[i] != 0 {
            nums[writeIdx] = nums[i]
            writeIdx++
        }
    }
    // Fill rest with zeros
    for i := writeIdx; i < len(nums); i++ {
        nums[i] = 0
    }
}

// ===== ARRAY ROTATION =====
// Rotate right by k (reverse method)
func rotateRight(nums []int, k int) {
    n := len(nums)
    k = k % n  // Handle k > n
    reverse(nums, 0, n-1)      // Reverse entire array
    reverse(nums, 0, k-1)      // Reverse first k elements
    reverse(nums, k, n-1)      // Reverse remaining elements
}

func reverse(nums []int, start, end int) {
    for start < end {
        nums[start], nums[end] = nums[end], nums[start]
        start++
        end--
    }
}

// Rotate using cyclic replacement
func rotateCyclic(nums []int, k int) {
    n := len(nums)
    k = k % n
    count := 0
    for start := 0; count < n; start++ {
        current := start
        prev := nums[start]
        for {
            next := (current + k) % n
            temp := nums[next]
            nums[next] = prev
            prev = temp
            current = next
            count++
            if start == current {
                break
            }
        }
    }
}

// ===== SLIDING WINDOW =====
// Maximum sum of subarray of size k
func maxSumSubarray(nums []int, k int) int {
    if len(nums) < k {
        return 0
    }
    windowSum := 0
    // Calculate sum of first window
    for i := 0; i < k; i++ {
        windowSum += nums[i]
    }
    maxSum := windowSum
    // Slide window
    for i := k; i < len(nums); i++ {
        windowSum = windowSum - nums[i-k] + nums[i]
        maxSum = max(maxSum, windowSum)
    }
    return maxSum
}

// ===== KADANE'S ALGORITHM =====
// Maximum subarray sum
func maxSubarraySum(nums []int) int {
    maxEndingHere := nums[0]
    maxSoFar := nums[0]
    for i := 1; i < len(nums); i++ {
        // Either extend previous subarray or start new
        maxEndingHere = max(nums[i], maxEndingHere+nums[i])
        maxSoFar = max(maxSoFar, maxEndingHere)
    }
    return maxSoFar
}

// ===== PARTITIONING =====
// Partition around pivot (two pointers)
func partition(nums []int, pivot int) int {
    left := 0
    right := len(nums) - 1
    for left <= right {
        if nums[left] <= pivot {
            left++
        } else if nums[right] > pivot {
            right--
        } else {
            nums[left], nums[right] = nums[right], nums[left]
            left++
            right--
        }
    }
    return left
}

// Helper functions
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

# Python: Comprehensive array techniques

# ===== PREFIX SUM =====
def prefix_sum(nums):
    """Precompute cumulative sums."""
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)
    return prefix

def range_sum(prefix, i, j):
    """O(1) range sum query."""
    return prefix[j+1] - prefix[i]

# Find subarray with sum k
def subarray_sum(nums, k):
    prefix = prefix_sum(nums)
    count = 0
    sum_map = {0: 1}  # Empty subarray has sum 0
    
    for i in range(1, len(prefix)):
        # If prefix[i] - k exists, we found a subarray
        if prefix[i] - k in sum_map:
            count += sum_map[prefix[i] - k]
        sum_map[prefix[i]] = sum_map.get(prefix[i], 0) + 1
    return count

# ===== TWO-PASS TECHNIQUES =====
# Product except self
def product_except_self(nums):
    n = len(nums)
    result = [1] * n
    
    # First pass: left products
    for i in range(1, n):
        result[i] = result[i-1] * nums[i-1]
    
    # Second pass: multiply by right products
    right = 1
    for i in range(n-1, -1, -1):
        result[i] *= right
        right *= nums[i]
    
    return result

# Trapping rainwater
def trap(height):
    n = len(height)
    if n == 0:
        return 0
    
    # First pass: left max
    left_max = [0] * n
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i-1], height[i])
    
    # Second pass: right max and calculate water
    right_max = height[n-1]
    water = 0
    for i in range(n-2, -1, -1):
        right_max = max(right_max, height[i])
        water += min(left_max[i], right_max) - height[i]
    
    return water

# ===== IN-PLACE MODIFICATIONS =====
# Remove duplicates
def remove_duplicates(nums):
    if not nums:
        return 0
    write_idx = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i-1]:
            nums[write_idx] = nums[i]
            write_idx += 1
    return write_idx

# Move zeros
def move_zeros(nums):
    write_idx = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[write_idx] = nums[i]
            write_idx += 1
    # Fill rest with zeros
    for i in range(write_idx, len(nums)):
        nums[i] = 0

# ===== ARRAY ROTATION =====
# Rotate right by k (reverse method)
def rotate_right(nums, k):
    n = len(nums)
    k = k % n
    nums.reverse()        # Reverse entire array
    nums[:k] = reversed(nums[:k])      # Reverse first k
    nums[k:] = reversed(nums[k:])      # Reverse rest

# ===== SLIDING WINDOW =====
# Maximum sum of subarray of size k
def max_sum_subarray(nums, k):
    if len(nums) < k:
        return 0
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i-k] + nums[i]
        max_sum = max(max_sum, window_sum)
    return max_sum

# ===== KADANE'S ALGORITHM =====
# Maximum subarray sum
def max_subarray_sum(nums):
    max_ending_here = max_so_far = nums[0]
    for i in range(1, len(nums)):
        max_ending_here = max(nums[i], max_ending_here + nums[i])
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

# ===== PARTITIONING =====
# Partition around pivot
def partition(nums, pivot):
    left = 0
    right = len(nums) - 1
    while left <= right:
        if nums[left] <= pivot:
            left += 1
        elif nums[right] > pivot:
            right -= 1
        else:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
    return left

# Key insights:
# 1. Prefix sum enables O(1) range queries after O(n) preprocessing
# 2. Two-pass techniques often solve problems that seem to need nested loops
# 3. In-place modifications save space but require careful indexing
# 4. Array rotation can be done efficiently with reverse or cyclic methods
# 5. Sliding window reduces O(n*n) to O(n) for subarray problems`,
				},
				{
					Title: "Hash Table Advanced Techniques",
					Content: `Advanced hash table techniques enable efficient solutions to complex problems. Mastering these patterns is essential for coding interviews and real-world applications.

**1. Frequency Counting:**

One of the most common hash table applications is counting frequencies of elements.

**Basic Pattern:**
- Iterate through collection
- For each element, increment count in hash map
- Time: O(n), Space: O(k) where k is number of unique elements

**Applications:**
- Count word frequencies in text
- Count character frequencies in strings
- Find most/least frequent elements
- Detect duplicates
- Validate character counts (anagram checking)

**Optimization:**
- For limited alphabet (26 letters, 256 ASCII), use array instead of hash map
- Array access is faster and uses less memory
- Example: freq := [26]int{} for lowercase letters

**2. Two-Sum Pattern:**

The two-sum pattern is fundamental and appears in many variations.

**Core Idea:**
- Instead of checking all pairs (O(n²)), store complements
- For each element, check if its complement (target - element) exists
- If found, we have a pair; if not, store current element for future checks

**Why It Works:**
- Hash map lookup is O(1) average
- Single pass through array: O(n) total
- Much better than nested loops: O(n²)

**Variations:**
- **Two Sum**: Find two numbers that sum to target
- **Three Sum**: Find three numbers that sum to target (use two-sum as subroutine)
- **Four Sum**: Extend pattern further
- **Subarray Sum**: Find subarray with given sum (use prefix sums)

**3. Grouping/Partitioning:**

Group elements by some property or key.

**Pattern:**
- Use property as hash map key
- Value is list/array of elements with that property
- Single pass groups all elements

**Common Examples:**
- **Group Anagrams**: Key = sorted string or character frequency array
- **Group by Length**: Key = string/array length
- **Group by Property**: Key = any computed property
- **Partition by Condition**: Key = boolean or category

**When to Use:**
- Need to organize data by category
- Want to process similar elements together
- Need to partition array based on property

**4. Sliding Window with Hash Map:**

Combine sliding window technique with hash map for efficient substring/subarray problems.

**Pattern:**
- Maintain window state in hash map
- Add elements as window expands
- Remove elements as window contracts
- Check constraints using hash map state

**Advantages:**
- O(1) updates to window state
- Efficient constraint checking
- Single pass algorithm: O(n)

**Common Problems:**
- Longest substring with K distinct characters
- Minimum window substring
- Substring with all characters of pattern
- Longest substring without repeating characters

**5. Index Mapping:**

Store indices of elements for quick lookup.

**Pattern:**
- Key = element value
- Value = index (or list of indices)
- Enables O(1) index lookup

**Use Cases:**
- Find indices of target values
- Track last occurrence of elements
- Build index for fast lookups
- Two-sum problems (need indices, not just values)

**6. Complement Tracking:**

Store what we're looking for instead of what we have.

**Pattern:**
- Instead of storing current element, store its complement
- Check if current element matches stored complement
- Useful for difference, sum, and other operations

**Example:**
- Two-sum: Store target - current instead of current
- When we see complement, we found the pair

**7. Character Frequency Array:**

For problems with limited alphabet, use array instead of hash map.

**When to Use:**
- Lowercase letters (26): freq := [26]int{}
- Uppercase letters (26): freq := [26]int{}
- ASCII characters (128): freq := [128]int{}
- Extended ASCII (256): freq := [256]int{}

**Advantages:**
- Faster access (direct indexing)
- Less memory overhead
- Simpler code
- Better cache performance

**8. Multi-Level Hash Maps:**

Use nested hash maps for complex grouping.

**Pattern:**
- First level: Group by primary property
- Second level: Further group by secondary property
- Example: map[string]map[string]int for hierarchical grouping

**Use Cases:**
- Group by multiple properties
- Hierarchical data organization
- Multi-dimensional indexing

**9. Hash Map as Cache/Memo:**

Store computed results to avoid recomputation.

**Pattern:**
- Before computing, check if result exists in map
- If found, return cached result
- If not, compute, store, and return

**Applications:**
- Dynamic programming memoization
- Function result caching
- Expensive computation caching

**10. Set Operations with Hash Maps:**

Implement set operations using hash maps.

**Operations:**
- **Union**: Combine keys from both maps
- **Intersection**: Keys present in both maps
- **Difference**: Keys in first but not second
- **Symmetric Difference**: Keys in exactly one map

**Best Practices:**

1. **Choose Right Key:**
   - Key should uniquely identify what you're grouping
   - Consider using sorted strings or frequency arrays for anagrams
   - Use tuples/structs for composite keys

2. **Optimize for Your Data:**
   - Use arrays for limited alphabets
   - Use hash maps for arbitrary keys
   - Consider space-time tradeoffs

3. **Handle Edge Cases:**
   - Empty collections
   - Single element
   - All elements same
   - No valid solution

4. **Think About Order:**
   - Do you need insertion order? (Use LinkedHashMap or track separately)
   - Do you need sorted order? (Consider TreeMap or sort keys)

5. **Memory Considerations:**
   - Hash maps use more memory than arrays
   - For large datasets, consider if array is sufficient
   - Be mindful of load factor and resizing`,
					CodeExamples: `// Go: Comprehensive hash table advanced techniques

// ===== FREQUENCY COUNTING =====
// Count element frequencies
func countFreq(nums []int) map[int]int {
    freq := make(map[int]int)
    for _, num := range nums {
        freq[num]++
    }
    return freq
}

// Character frequency (using array for efficiency)
func charFreq(s string) [26]int {
    var freq [26]int
    for _, c := range s {
        if c >= 'a' && c <= 'z' {
            freq[c-'a']++
        }
    }
    return freq
}

// Find most frequent element
func mostFrequent(nums []int) int {
    freq := countFreq(nums)
    maxFreq := 0
    result := 0
    for num, count := range freq {
        if count > maxFreq {
            maxFreq = count
            result = num
        }
    }
    return result
}

// ===== TWO-SUM PATTERN =====
// Classic two-sum
func twoSum(nums []int, target int) []int {
    m := make(map[int]int)  // value -> index
    for i, num := range nums {
        complement := target - num
        if idx, exists := m[complement]; exists {
            return []int{idx, i}
        }
        m[num] = i  // Store for future lookups
    }
    return nil
}

// Three-sum (using two-sum as subroutine)
func threeSum(nums []int, target int) [][]int {
    sort.Ints(nums)  // Sort for duplicate handling
    result := [][]int{}
    for i := 0; i < len(nums)-2; i++ {
        // Skip duplicates
        if i > 0 && nums[i] == nums[i-1] {
            continue
        }
        // Two-sum for remaining elements
        left, right := i+1, len(nums)-1
        for left < right {
            sum := nums[i] + nums[left] + nums[right]
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
                left++
            } else {
                right--
            }
        }
    }
    return result
}

// ===== GROUPING/PARTITIONING =====
// Group anagrams (using character frequency as key)
func groupAnagrams(strs []string) [][]string {
    groups := make(map[[26]int][]string)
    for _, s := range strs {
        key := [26]int{}
        for _, c := range s {
            if c >= 'a' && c <= 'z' {
                key[c-'a']++
            }
        }
        groups[key] = append(groups[key], s)
    }
    result := make([][]string, 0, len(groups))
    for _, group := range groups {
        result = append(result, group)
    }
    return result
}

// Group anagrams (using sorted string as key - alternative)
func groupAnagramsSorted(strs []string) [][]string {
    groups := make(map[string][]string)
    for _, s := range strs {
        runes := []rune(s)
        sort.Slice(runes, func(i, j int) bool {
            return runes[i] < runes[j]
        })
        key := string(runes)
        groups[key] = append(groups[key], s)
    }
    result := make([][]string, 0, len(groups))
    for _, group := range groups {
        result = append(result, group)
    }
    return result
}

// Group by length
func groupByLength(strs []string) map[int][]string {
    groups := make(map[int][]string)
    for _, s := range strs {
        length := len(s)
        groups[length] = append(groups[length], s)
    }
    return groups
}

// ===== SLIDING WINDOW WITH HASH MAP =====
// Longest substring with K distinct characters
func longestSubstringKDistinct(s string, k int) int {
    charCount := make(map[byte]int)
    left := 0
    maxLen := 0
    
    for right := 0; right < len(s); right++ {
        // Expand window
        charCount[s[right]]++
        
        // Shrink window if more than k distinct
        for len(charCount) > k {
            charCount[s[left]]--
            if charCount[s[left]] == 0 {
                delete(charCount, s[left])
            }
            left++
        }
        
        // Update max length
        maxLen = max(maxLen, right-left+1)
    }
    return maxLen
}

// Minimum window substring
func minWindow(s string, t string) string {
    need := make(map[byte]int)
    for _, c := range t {
        need[byte(c)]++
    }
    
    window := make(map[byte]int)
    left, right := 0, 0
    valid := 0
    start, minLen := 0, math.MaxInt32
    
    for right < len(s) {
        c := s[right]
        right++
        
        if need[c] > 0 {
            window[c]++
            if window[c] == need[c] {
                valid++
            }
        }
        
        // Shrink window
        for valid == len(need) {
            if right-left < minLen {
                start = left
                minLen = right - left
            }
            
            d := s[left]
            left++
            
            if need[d] > 0 {
                if window[d] == need[d] {
                    valid--
                }
                window[d]--
            }
        }
    }
    
    if minLen == math.MaxInt32 {
        return ""
    }
    return s[start : start+minLen]
}

// ===== INDEX MAPPING =====
// Store indices for quick lookup
func buildIndexMap(nums []int) map[int][]int {
    indexMap := make(map[int][]int)
    for i, num := range nums {
        indexMap[num] = append(indexMap[num], i)
    }
    return indexMap
}

// Find all indices of target
func findAllIndices(nums []int, target int) []int {
    indices := []int{}
    for i, num := range nums {
        if num == target {
            indices = append(indices, i)
        }
    }
    return indices
}

// ===== COMPLEMENT TRACKING =====
// Subarray sum equals k (using prefix sum + complement)
func subarraySum(nums []int, k int) int {
    prefixSum := 0
    count := 0
    sumMap := make(map[int]int)
    sumMap[0] = 1  // Empty subarray has sum 0
    
    for _, num := range nums {
        prefixSum += num
        complement := prefixSum - k
        if val, exists := sumMap[complement]; exists {
            count += val
        }
        sumMap[prefixSum]++
    }
    return count
}

// ===== CHARACTER FREQUENCY ARRAY =====
// Check if two strings are anagrams
func isAnagram(s1, s2 string) bool {
    if len(s1) != len(s2) {
        return false
    }
    var freq1, freq2 [26]int
    for i := 0; i < len(s1); i++ {
        freq1[s1[i]-'a']++
        freq2[s2[i]-'a']++
    }
    return freq1 == freq2
}

// Find all anagrams of pattern in string
func findAnagrams(s string, p string) []int {
    if len(s) < len(p) {
        return []int{}
    }
    
    var pFreq, windowFreq [26]int
    for i := 0; i < len(p); i++ {
        pFreq[p[i]-'a']++
        windowFreq[s[i]-'a']++
    }
    
    result := []int{}
    if pFreq == windowFreq {
        result = append(result, 0)
    }
    
    for i := len(p); i < len(s); i++ {
        windowFreq[s[i-len(p)]-'a']--  // Remove left
        windowFreq[s[i]-'a']++          // Add right
        if pFreq == windowFreq {
            result = append(result, i-len(p)+1)
        }
    }
    return result
}

// ===== HASH MAP AS CACHE/MEMO =====
// Fibonacci with memoization
var fibMemo = make(map[int]int)

func fibonacciMemo(n int) int {
    if n <= 1 {
        return n
    }
    if val, exists := fibMemo[n]; exists {
        return val
    }
    fibMemo[n] = fibonacciMemo(n-1) + fibonacciMemo(n-2)
    return fibMemo[n]
}

// ===== SET OPERATIONS =====
// Union of two sets (represented as maps)
func setUnion(set1, set2 map[int]bool) map[int]bool {
    result := make(map[int]bool)
    for k := range set1 {
        result[k] = true
    }
    for k := range set2 {
        result[k] = true
    }
    return result
}

// Intersection of two sets
func setIntersection(set1, set2 map[int]bool) map[int]bool {
    result := make(map[int]bool)
    for k := range set1 {
        if set2[k] {
            result[k] = true
        }
    }
    return result
}

// Helper function
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

# Python: Comprehensive hash table advanced techniques

# ===== FREQUENCY COUNTING =====
from collections import Counter, defaultdict

# Count frequencies
def count_freq(nums):
    return Counter(nums)

# Character frequency (using array)
def char_freq(s):
    freq = [0] * 26
    for c in s:
        if 'a' <= c <= 'z':
            freq[ord(c) - ord('a')] += 1
    return freq

# Most frequent element
def most_frequent(nums):
    freq = Counter(nums)
    return freq.most_common(1)[0][0]

# ===== TWO-SUM PATTERN =====
# Classic two-sum
def two_sum(nums, target):
    seen = {}  # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return None

# Three-sum
def three_sum(nums, target):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == target:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < target:
                left += 1
            else:
                right -= 1
    return result

# ===== GROUPING/PARTITIONING =====
# Group anagrams (using sorted string as key)
def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)
    return list(groups.values())

# Group anagrams (using character frequency)
def group_anagrams_freq(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple([0] * 26)  # Character frequency array
        for c in s:
            if 'a' <= c <= 'z':
                key[ord(c) - ord('a')] += 1
        groups[tuple(key)].append(s)
    return list(groups.values())

# Group by length
def group_by_length(strs):
    groups = defaultdict(list)
    for s in strs:
        groups[len(s)].append(s)
    return dict(groups)

# ===== SLIDING WINDOW WITH HASH MAP =====
# Longest substring with K distinct characters
def longest_substring_k_distinct(s, k):
    char_count = {}
    left = 0
    max_len = 0
    
    for right in range(len(s)):
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_len = max(max_len, right - left + 1)
    
    return max_len

# Minimum window substring
def min_window(s, t):
    need = Counter(t)
    window = {}
    left = right = 0
    valid = 0
    start = 0
    min_len = float('inf')
    
    while right < len(s):
        c = s[right]
        right += 1
        
        if c in need:
            window[c] = window.get(c, 0) + 1
            if window[c] == need[c]:
                valid += 1
        
        while valid == len(need):
            if right - left < min_len:
                start = left
                min_len = right - left
            
            d = s[left]
            left += 1
            
            if d in need:
                if window[d] == need[d]:
                    valid -= 1
                window[d] -= 1
    
    return "" if min_len == float('inf') else s[start:start+min_len]

# ===== INDEX MAPPING =====
# Build index map
def build_index_map(nums):
    index_map = defaultdict(list)
    for i, num in enumerate(nums):
        index_map[num].append(i)
    return dict(index_map)

# ===== COMPLEMENT TRACKING =====
# Subarray sum equals k
def subarray_sum(nums, k):
    prefix_sum = 0
    count = 0
    sum_map = {0: 1}
    
    for num in nums:
        prefix_sum += num
        complement = prefix_sum - k
        if complement in sum_map:
            count += sum_map[complement]
        sum_map[prefix_sum] = sum_map.get(prefix_sum, 0) + 1
    
    return count

# ===== CHARACTER FREQUENCY ARRAY =====
# Check if anagrams
def is_anagram(s1, s2):
    if len(s1) != len(s2):
        return False
    freq1 = [0] * 26
    freq2 = [0] * 26
    for c in s1:
        freq1[ord(c) - ord('a')] += 1
    for c in s2:
        freq2[ord(c) - ord('a')] += 1
    return freq1 == freq2

# Find all anagrams
def find_anagrams(s, p):
    if len(s) < len(p):
        return []
    
    p_freq = [0] * 26
    window_freq = [0] * 26
    
    for i in range(len(p)):
        p_freq[ord(p[i]) - ord('a')] += 1
        window_freq[ord(s[i]) - ord('a')] += 1
    
    result = []
    if p_freq == window_freq:
        result.append(0)
    
    for i in range(len(p), len(s)):
        window_freq[ord(s[i-len(p)]) - ord('a')] -= 1
        window_freq[ord(s[i]) - ord('a')] += 1
        if p_freq == window_freq:
            result.append(i - len(p) + 1)
    
    return result

# ===== HASH MAP AS CACHE =====
# Fibonacci with memoization
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Manual memoization
fib_memo = {}
def fibonacci_manual(n):
    if n <= 1:
        return n
    if n in fib_memo:
        return fib_memo[n]
    fib_memo[n] = fibonacci_manual(n-1) + fibonacci_manual(n-2)
    return fib_memo[n]

# ===== SET OPERATIONS =====
# Union
def set_union(set1, set2):
    return set1 | set2

# Intersection
def set_intersection(set1, set2):
    return set1 & set2

# Key insights:
# 1. Frequency counting is fundamental - use arrays for limited alphabets
# 2. Two-sum pattern reduces O(n²) to O(n) - store complements
# 3. Grouping enables efficient organization by properties
# 4. Sliding window + hash map solves substring problems efficiently
# 5. Index mapping enables fast lookups of positions
# 6. Complement tracking stores what we need, not what we have`,
				},
				{
					Title: "Sliding Window Maximum",
					Content: `Finding the maximum element in every window of size k is a classic problem that demonstrates the power of the deque (double-ended queue) data structure. This technique appears frequently in coding interviews and real-world applications.

**Problem Statement:**
Given an array of integers and a window size k, find the maximum element in every window of size k as it slides through the array from left to right.

**Example:**
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation:
- Window 1: [1,3,-1] → max = 3
- Window 2: [3,-1,-3] → max = 3
- Window 3: [-1,-3,5] → max = 5
- Window 4: [-3,5,3] → max = 5
- Window 5: [5,3,6] → max = 6
- Window 6: [3,6,7] → max = 7

**Naive Approach:**

**Algorithm:**
- For each window position i from 0 to n-k:
  - Scan all k elements in window [i, i+k-1]
  - Find maximum among these k elements
- Return array of maximums

**Complexity:**
- Time: O(n×k) - n windows, each takes O(k) to scan
- Space: O(1) - only storing result array

**Why It's Inefficient:**
- Recomputes maximum from scratch for each window
- Doesn't reuse information from previous windows
- For large arrays with large k, becomes very slow

**Optimal Approach Using Deque:**

**Key Insight:**
We only need to keep track of elements that could be maximum in future windows. Once we see a larger element, all smaller elements before it can never be maximum (they'll leave the window before the larger element does).

**Deque Structure:**
- Store indices (not values) in deque
- Maintain indices in decreasing order of their values
- Front of deque: index of maximum element in current window
- Back of deque: indices of elements that might become maximum later

**Algorithm:**

**Step 1: Process First Window (indices 0 to k-1)**
- For each element in first window:
  - Remove indices from back whose values are ≤ current element
  - Add current index to back of deque
- Front of deque now contains index of maximum in first window

**Step 2: Process Remaining Elements (indices k to n-1)**
For each new element:
1. **Remove Out-of-Window Indices:**
   - Remove indices from front that are outside current window (≤ i-k)
   - These elements have left the window

2. **Remove Smaller Elements:**
   - Remove indices from back whose values are ≤ current element
   - These can never be maximum (current is larger and will stay longer)

3. **Add Current Index:**
   - Add current index to back of deque
   - It might be maximum in future windows

4. **Record Maximum:**
   - Front of deque contains index of maximum in current window
   - Add nums[deque[0]] to result

**Complexity:**
- Time: O(n) - each element added/removed at most once
- Space: O(k) - deque stores at most k indices

**Why It's Optimal:**
- Each element added to deque exactly once
- Each element removed from deque at most once
- Amortized O(1) per element
- Total: O(n) time

**Visual Example:**

Array: [1,3,-1,-3,5,3,6,7], k = 3

Window 1 [1,3,-1]:
- Process 1: deque = [0] (index of 1)
- Process 3: Remove 0 (1 < 3), deque = [1] (index of 3)
- Process -1: deque = [1,2] (3 > -1, keep both)
- Max = nums[1] = 3

Window 2 [3,-1,-3]:
- Remove 0 (out of window): deque = [1,2]
- Process -3: deque = [1,2,3] (all larger than -3)
- Max = nums[1] = 3

Window 3 [-1,-3,5]:
- Remove 1 (out of window): deque = [2,3]
- Process 5: Remove 2,3 (-1,-3 < 5), deque = [4] (index of 5)
- Max = nums[4] = 5

**When to Use:**

**Sliding Window Problems:**
- Maximum/minimum in each window
- Range maximum/minimum queries
- Stock price problems (max profit in time window)

**Related Problems:**
- Minimum window substring
- Longest substring with K distinct characters
- Subarray with given sum
- Maximum average subarray

**Variations:**

**1. Minimum in Sliding Window:**
- Same algorithm, but maintain indices in increasing order
- Remove larger elements instead of smaller ones

**2. K Maximum Elements:**
- Maintain deque with k largest elements
- More complex but similar principle

**3. Range Queries:**
- Preprocess array with sliding window technique
- Answer range maximum queries quickly

**Best Practices:**

1. **Use Indices, Not Values:**
   - Store indices in deque to check if element is in window
   - Access values via nums[index] when needed

2. **Maintain Monotonicity:**
   - Keep deque monotonic (decreasing for max, increasing for min)
   - This ensures front always has optimal element

3. **Handle Edge Cases:**
   - Empty array
   - k = 1 (each element is its own window)
   - k = n (single window, entire array)
   - k > n (invalid input)

4. **Optimize Deque Operations:**
   - Use efficient deque implementation
   - Avoid unnecessary operations
   - Consider using array-based deque for better cache performance`,
					CodeExamples: `// Go: Comprehensive sliding window maximum

// Optimal solution using deque
func maxSlidingWindow(nums []int, k int) []int {
    if len(nums) == 0 || k == 0 {
        return []int{}
    }
    if k == 1 {
        return nums  // Each element is its own window
    }
    if k >= len(nums) {
        // Single window - find max of entire array
        maxVal := nums[0]
        for _, val := range nums {
            if val > maxVal {
                maxVal = val
            }
        }
        return []int{maxVal}
    }
    
    var deque []int  // Store indices (not values!)
    result := make([]int, 0, len(nums)-k+1)
    
    // Process first window [0, k-1]
    for i := 0; i < k; i++ {
        // Remove indices from back whose values are <= current
        // These can never be maximum (current is larger and stays longer)
        for len(deque) > 0 && nums[deque[len(deque)-1]] <= nums[i] {
            deque = deque[:len(deque)-1]
        }
        deque = append(deque, i)
    }
    // First window's maximum
    result = append(result, nums[deque[0]])
    
    // Process remaining elements [k, n-1]
    for i := k; i < len(nums); i++ {
        // Remove indices outside current window [i-k+1, i]
        // Elements at indices <= i-k have left the window
        for len(deque) > 0 && deque[0] <= i-k {
            deque = deque[1:]
        }
        
        // Remove indices from back whose values are <= current
        // Maintain decreasing order
        for len(deque) > 0 && nums[deque[len(deque)-1]] <= nums[i] {
            deque = deque[:len(deque)-1]
        }
        
        // Add current index
        deque = append(deque, i)
        
        // Front of deque is maximum in current window
        result = append(result, nums[deque[0]])
    }
    
    return result
}

// Naive approach for comparison
func maxSlidingWindowNaive(nums []int, k int) []int {
    if len(nums) == 0 {
        return []int{}
    }
    result := make([]int, 0, len(nums)-k+1)
    for i := 0; i <= len(nums)-k; i++ {
        maxVal := nums[i]
        for j := i + 1; j < i+k; j++ {
            if nums[j] > maxVal {
                maxVal = nums[j]
            }
        }
        result = append(result, maxVal)
    }
    return result
    // Time: O(n×k), Space: O(1)
}

// Minimum in sliding window (variation)
func minSlidingWindow(nums []int, k int) []int {
    if len(nums) == 0 {
        return []int{}
    }
    var deque []int
    result := make([]int, 0, len(nums)-k+1)
    
    // Process first window
    for i := 0; i < k; i++ {
        // Remove indices whose values are >= current (opposite of max)
        for len(deque) > 0 && nums[deque[len(deque)-1]] >= nums[i] {
            deque = deque[:len(deque)-1]
        }
        deque = append(deque, i)
    }
    result = append(result, nums[deque[0]])
    
    // Process remaining
    for i := k; i < len(nums); i++ {
        // Remove out-of-window indices
        for len(deque) > 0 && deque[0] <= i-k {
            deque = deque[1:]
        }
        // Remove larger elements (maintain increasing order)
        for len(deque) > 0 && nums[deque[len(deque)-1]] >= nums[i] {
            deque = deque[:len(deque)-1]
        }
        deque = append(deque, i)
        result = append(result, nums[deque[0]])
    }
    return result
}

// Using container/list for proper deque (alternative)
import "container/list"

func maxSlidingWindowList(nums []int, k int) []int {
    if len(nums) == 0 {
        return []int{}
    }
    dq := list.New()
    result := make([]int, 0, len(nums)-k+1)
    
    for i := 0; i < len(nums); i++ {
        // Remove out-of-window indices
        for dq.Len() > 0 && dq.Front().Value.(int) <= i-k {
            dq.Remove(dq.Front())
        }
        
        // Remove smaller elements
        for dq.Len() > 0 && nums[dq.Back().Value.(int)] <= nums[i] {
            dq.Remove(dq.Back())
        }
        
        dq.PushBack(i)
        
        // Record maximum when window is complete
        if i >= k-1 {
            result = append(result, nums[dq.Front().Value.(int)])
        }
    }
    return result
}

# Python: Comprehensive sliding window maximum

from collections import deque

# Optimal solution using deque
def max_sliding_window(nums, k):
    """Find maximum in each sliding window of size k."""
    if not nums or k == 0:
        return []
    if k == 1:
        return nums  # Each element is its own window
    if k >= len(nums):
        return [max(nums)]  # Single window
    
    dq = deque()  # Store indices
    result = []
    
    # Process first window [0, k-1]
    for i in range(k):
        # Remove indices whose values are <= current
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        dq.append(i)
    result.append(nums[dq[0]])
    
    # Process remaining elements [k, n-1]
    for i in range(k, len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove smaller elements
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        
        dq.append(i)
        result.append(nums[dq[0]])
    
    return result

# Naive approach for comparison
def max_sliding_window_naive(nums, k):
    """O(n×k) naive approach."""
    if not nums:
        return []
    result = []
    for i in range(len(nums) - k + 1):
        window = nums[i:i+k]
        result.append(max(window))
    return result

# Minimum in sliding window
def min_sliding_window(nums, k):
    """Find minimum in each sliding window."""
    if not nums:
        return []
    dq = deque()
    result = []
    
    for i in range(len(nums)):
        # Remove out-of-window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove larger elements (maintain increasing order)
        while dq and nums[dq[-1]] >= nums[i]:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# Example usage and complexity analysis
if __name__ == "__main__":
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    
    print(f"Array: {nums}")
    print(f"Window size: {k}")
    print(f"Maximums: {max_sliding_window(nums, k)}")
    print(f"Expected: [3, 3, 5, 5, 6, 7]")
    
    # Complexity comparison
    import time
    
    large_nums = list(range(100000))
    k = 1000
    
    # Optimal: O(n)
    start = time.time()
    result1 = max_sliding_window(large_nums, k)
    time1 = time.time() - start
    
    # Naive: O(n×k)
    start = time.time()
    result2 = max_sliding_window_naive(large_nums[:10000], k)  # Smaller for demo
    time2 = time.time() - start
    
    print(f"\nOptimal (100k elements): {time1:.4f}s")
    print(f"Naive (10k elements): {time2:.4f}s")
    print("Optimal is much faster for large inputs!")

# Key insights:
# 1. Deque maintains indices in monotonic order
# 2. Each element added/removed at most once → O(n) time
# 3. Store indices, not values, to check window boundaries
# 4. Front always has optimal element for current window
# 5. Remove elements that can never be optimal`,
				},
				{
					Title: "Prefix Sum Advanced",
					Content: `Prefix sums (also called cumulative sums) are a powerful technique for answering range queries efficiently. After O(n) preprocessing, you can answer range sum queries in O(1) time. This technique extends to 2D arrays and enables efficient range updates.

**Understanding Prefix Sums:**

**Basic Concept:**
- Prefix sum array: prefix[i] = sum of elements from index 0 to i-1
- Convention: prefix[0] = 0 (sum of empty prefix)
- This allows O(1) range sum queries

**Why It Works:**
- Sum from i to j = Sum from 0 to j - Sum from 0 to i-1
- rangeSum(i, j) = prefix[j+1] - prefix[i]
- Both prefix values are precomputed, so query is O(1)

**1D Prefix Sum:**

**Construction:**
- Start with prefix[0] = 0
- For each i: prefix[i+1] = prefix[i] + nums[i]
- Time: O(n), Space: O(n)

**Query:**
- Range [i, j] (inclusive): prefix[j+1] - prefix[i]
- Time: O(1) per query

**2D Prefix Sum (Matrix):**

Extend the concept to 2D arrays for rectangular region queries.

**Construction:**
- prefix[i][j] = sum of rectangle from (0,0) to (i-1,j-1)
- Formula: prefix[i][j] = matrix[i-1][j-1] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]
- Subtract prefix[i-1][j-1] because it's counted twice

**Query:**
- Rectangle from (r1, c1) to (r2, c2):
- sum = prefix[r2+1][c2+1] - prefix[r1][c2+1] - prefix[r2+1][c1] + prefix[r1][c1]
- Add back prefix[r1][c1] because it's subtracted twice

**Visual Explanation:**
 + "" + 
To get sum of rectangle [r1, c1] to [r2, c2]:
Total = prefix[r2+1][c2+1]  (sum from origin to bottom-right)
      - prefix[r1][c2+1]     (remove top part)
      - prefix[r2+1][c1]     (remove left part)
      + prefix[r1][c1]       (add back double-subtracted top-left)
 + "" + 

**Prefix Sum Applications:**

**1. Range Sum Queries:**
- Answer multiple queries on same array efficiently
- Preprocessing: O(n)
- Each query: O(1)
- Much better than O(n) per query

**2. Subarray Sum Problems:**
- Find subarray with sum k: Use prefix sum + hash map
- Count subarrays with sum k: Track prefix sum frequencies
- Maximum subarray sum: Kadane's algorithm (related concept)

**3. Equilibrium Point:**
- Find index where sum of left elements = sum of right elements
- Use prefix sum to calculate left sum
- Use suffix sum (or total - prefix) for right sum

**4. Difference Array Technique:**

Powerful technique for range updates.

**Problem:** Update range [i, j] by adding value v to each element.

**Naive:** O(n) per update - update each element individually

**Difference Array:**
- Create difference array: diff[i] = nums[i] - nums[i-1] (for i > 0)
- To update range [i, j] by v:
  - diff[i] += v (start of range)
  - diff[j+1] -= v (end of range + 1)
- Reconstruct: nums[i] = nums[0] + prefixSum(diff)[i]
- Update: O(1) per range update
- Reconstruction: O(n) after all updates

**Why It Works:**
- Adding v to diff[i] affects all elements from i onwards
- Subtracting v from diff[j+1] cancels the effect after j
- Net effect: only elements [i, j] are affected

**5. Maximum Subarray (Kadane's Algorithm):**
- Related to prefix sum concept
- Track minimum prefix sum seen so far
- Current max = current prefix sum - minimum prefix sum

**6. Subarray with Given Sum:**
- Use prefix sum + hash map
- If prefix[j] - prefix[i] = target, then subarray [i, j-1] has sum target
- Store prefix sums in hash map for O(1) lookup

**When to Use:**

**Perfect For:**
- Multiple range sum queries on static array
- Subarray problems with sum constraints
- Range update problems (difference array)
- Matrix/2D range queries
- Problems involving cumulative properties

**Not Ideal For:**
- Single query (overhead not worth it)
- Frequently changing array (need to rebuild prefix)
- Very sparse queries (may not need preprocessing)

**Best Practices:**

1. **Use 1-indexed prefix arrays:**
   - prefix[0] = 0 makes formulas cleaner
   - Avoids special cases for index 0

2. **Handle edge cases:**
   - Empty ranges
   - Single element ranges
   - Out-of-bounds indices

3. **For 2D, be careful with indices:**
   - Matrix is 0-indexed, prefix is 1-indexed
   - Adjust indices accordingly in formulas

4. **Difference array for range updates:**
   - Use when you have many range updates
   - Reconstruct once at the end
   - More efficient than updating each element`,
					CodeExamples: `// Go: Comprehensive prefix sum techniques

// ===== 1D PREFIX SUM =====
// Build prefix sum array
func buildPrefixSum(nums []int) []int {
    prefix := make([]int, len(nums)+1)
    for i := 0; i < len(nums); i++ {
        prefix[i+1] = prefix[i] + nums[i]
    }
    return prefix
}

// Range sum query: O(1)
func rangeSum(prefix []int, i, j int) int {
    // Sum from index i to j (inclusive)
    return prefix[j+1] - prefix[i]
}

// Find subarray with sum k
func subarraySum(nums []int, k int) int {
    prefix := buildPrefixSum(nums)
    sumMap := make(map[int]int)
    sumMap[0] = 1  // Empty subarray has sum 0
    count := 0
    
    for i := 1; i < len(prefix); i++ {
        // If prefix[i] - k exists, we found a subarray
        complement := prefix[i] - k
        if val, exists := sumMap[complement]; exists {
            count += val
        }
        sumMap[prefix[i]]++
    }
    return count
}

// Equilibrium point (left sum = right sum)
func equilibriumPoint(nums []int) int {
    total := 0
    for _, num := range nums {
        total += num
    }
    
    leftSum := 0
    for i, num := range nums {
        if leftSum == total-leftSum-num {
            return i
        }
        leftSum += num
    }
    return -1
}

// ===== 2D PREFIX SUM =====
// Build 2D prefix sum matrix
func build2DPrefixSum(matrix [][]int) [][]int {
    m, n := len(matrix), len(matrix[0])
    prefix := make([][]int, m+1)
    for i := range prefix {
        prefix[i] = make([]int, n+1)
    }
    
    // Build prefix sum
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            // Current cell + top + left - top-left (double counted)
            prefix[i][j] = matrix[i-1][j-1] + 
                          prefix[i-1][j] + 
                          prefix[i][j-1] - 
                          prefix[i-1][j-1]
        }
    }
    return prefix
}

// Query sum of rectangle from (r1, c1) to (r2, c2) inclusive
func query2D(prefix [][]int, r1, c1, r2, c2 int) int {
    // Total - top - left + top-left (double subtracted)
    return prefix[r2+1][c2+1] - 
           prefix[r1][c2+1] - 
           prefix[r2+1][c1] + 
           prefix[r1][c1]
}

// ===== DIFFERENCE ARRAY =====
// Range update using difference array
func rangeUpdate(nums []int, updates [][]int) []int {
    // updates: [start, end, value] - add value to range [start, end]
    diff := make([]int, len(nums)+1)
    
    // Apply updates to difference array: O(1) per update
    for _, update := range updates {
        start, end, val := update[0], update[1], update[2]
        diff[start] += val      // Start of range
        if end+1 < len(diff) {
            diff[end+1] -= val  // End of range + 1
        }
    }
    
    // Reconstruct array using prefix sum: O(n)
    result := make([]int, len(nums))
    sum := 0
    for i := 0; i < len(nums); i++ {
        sum += diff[i]
        result[i] = nums[i] + sum
    }
    return result
}

// Example: Multiple range updates
func multipleRangeUpdates(nums []int, updates [][]int) []int {
    diff := make([]int, len(nums)+1)
    
    // Apply all updates: O(m) where m = number of updates
    for _, update := range updates {
        start, end, val := update[0], update[1], update[2]
        diff[start] += val
        if end+1 < len(diff) {
            diff[end+1] -= val
        }
    }
    
    // Reconstruct: O(n)
    for i := 1; i < len(nums); i++ {
        diff[i] += diff[i-1]  // Prefix sum in-place
    }
    
    // Apply to original array
    for i := 0; i < len(nums); i++ {
        nums[i] += diff[i]
    }
    return nums
}

// ===== MAXIMUM SUBARRAY (Kadane's - related to prefix sum) =====
func maxSubarraySum(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    minPrefixSum := 0
    prefixSum := 0
    maxSum := nums[0]
    
    for _, num := range nums {
        prefixSum += num
        // Maximum ending at current = prefixSum - minPrefixSum
        maxSum = max(maxSum, prefixSum-minPrefixSum)
        minPrefixSum = min(minPrefixSum, prefixSum)
    }
    return maxSum
}

// Helper functions
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

# Python: Comprehensive prefix sum techniques

# ===== 1D PREFIX SUM =====
def build_prefix_sum(nums):
    """Build prefix sum array."""
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)
    return prefix

def range_sum(prefix, i, j):
    """O(1) range sum query."""
    return prefix[j+1] - prefix[i]

# Find subarray with sum k
def subarray_sum(nums, k):
    prefix = build_prefix_sum(nums)
    sum_map = {0: 1}  # Empty subarray
    count = 0
    
    for i in range(1, len(prefix)):
        complement = prefix[i] - k
        if complement in sum_map:
            count += sum_map[complement]
        sum_map[prefix[i]] = sum_map.get(prefix[i], 0) + 1
    
    return count

# Equilibrium point
def equilibrium_point(nums):
    total = sum(nums)
    left_sum = 0
    for i, num in enumerate(nums):
        if left_sum == total - left_sum - num:
            return i
        left_sum += num
    return -1

# ===== 2D PREFIX SUM =====
def build_2d_prefix_sum(matrix):
    """Build 2D prefix sum matrix."""
    m, n = len(matrix), len(matrix[0])
    prefix = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            prefix[i][j] = (matrix[i-1][j-1] + 
                           prefix[i-1][j] + 
                           prefix[i][j-1] - 
                           prefix[i-1][j-1])
    return prefix

def query_2d(prefix, r1, c1, r2, c2):
    """Query sum of rectangle [r1, c1] to [r2, c2]."""
    return (prefix[r2+1][c2+1] - 
            prefix[r1][c2+1] - 
            prefix[r2+1][c1] + 
            prefix[r1][c1])

# ===== DIFFERENCE ARRAY =====
def range_update(nums, updates):
    """Apply range updates using difference array."""
    # updates: [(start, end, value), ...]
    diff = [0] * (len(nums) + 1)
    
    # Apply updates: O(1) per update
    for start, end, val in updates:
        diff[start] += val
        if end + 1 < len(diff):
            diff[end + 1] -= val
    
    # Reconstruct: O(n)
    result = []
    sum_val = 0
    for i in range(len(nums)):
        sum_val += diff[i]
        result.append(nums[i] + sum_val)
    return result

# Example: Multiple range updates efficiently
def multiple_range_updates(nums, updates):
    """Apply many range updates efficiently."""
    diff = [0] * (len(nums) + 1)
    
    # All updates: O(m) where m = number of updates
    for start, end, val in updates:
        diff[start] += val
        if end + 1 < len(diff):
            diff[end + 1] -= val
    
    # Reconstruct in-place: O(n)
    for i in range(1, len(nums)):
        diff[i] += diff[i-1]
    
    # Apply to original array
    for i in range(len(nums)):
        nums[i] += diff[i]
    
    return nums

# ===== MAXIMUM SUBARRAY (Kadane's) =====
def max_subarray_sum(nums):
    """Maximum subarray sum using prefix sum concept."""
    if not nums:
        return 0
    
    min_prefix_sum = 0
    prefix_sum = 0
    max_sum = nums[0]
    
    for num in nums:
        prefix_sum += num
        max_sum = max(max_sum, prefix_sum - min_prefix_sum)
        min_prefix_sum = min(min_prefix_sum, prefix_sum)
    
    return max_sum

# Example usage
if __name__ == "__main__":
    # 1D prefix sum
    nums = [1, 2, 3, 4, 5]
    prefix = build_prefix_sum(nums)
    print(f"Range sum [1,3]: {range_sum(prefix, 1, 3)}")  # 2+3+4 = 9
    
    # 2D prefix sum
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    prefix_2d = build_2d_prefix_sum(matrix)
    print(f"Rectangle sum [0,0] to [1,1]: {query_2d(prefix_2d, 0, 0, 1, 1)}")  # 1+2+4+5 = 12
    
    # Difference array
    nums2 = [0, 0, 0, 0, 0]
    updates = [(1, 3, 5), (2, 4, 3)]
    result = range_update(nums2, updates)
    print(f"After updates: {result}")  # [0, 5, 8, 8, 3]

# Key insights:
# 1. Prefix sum enables O(1) range queries after O(n) preprocessing
# 2. 2D prefix sum extends to matrices for rectangular queries
# 3. Difference array enables O(1) range updates
# 4. Combine prefix sum with hash map for subarray problems
# 5. Kadane's algorithm uses prefix sum concept for maximum subarray`,
				},
				{
					Title: "Hash Table Collision Resolution",
					Content: `When two different keys hash to the same index, we have a collision. Collisions are inevitable with any hash function (except perfect hash functions for fixed key sets). How we handle collisions significantly impacts hash table performance and determines the practical efficiency of hash table operations.

**Understanding Collisions:**

**Why Collisions Occur:**
- Hash function maps larger key space to smaller index space
- Pigeonhole principle: More keys than buckets guarantees collisions
- Even with good hash function, collisions happen probabilistically

**Impact of Collisions:**
- **No collisions**: O(1) operations (ideal)
- **Few collisions**: Still near O(1) average (good hash function)
- **Many collisions**: Degrades to O(n) worst case (poor hash function or bad luck)

**Collision Resolution Strategies:**

**1. Separate Chaining (Open Hashing):**

**How It Works:**
- Each bucket contains a data structure (usually linked list, sometimes tree)
- Colliding elements are stored in the same bucket's structure
- Lookup: Hash to bucket, then search within that bucket's structure

**Advantages:**
- Simple to implement and understand
- Handles many collisions gracefully
- No limit on number of elements (can store more than bucket count)
- Easy to delete (just remove from list)
- Performance degrades gradually as load factor increases

**Disadvantages:**
- Extra memory overhead for pointers/links
- Poor cache locality (scattered memory)
- Worst case: O(n) if all keys hash to same bucket
- Additional pointer dereferencing overhead

**Time Complexity:**
- Average: O(1 + α) where α is load factor
- Best case: O(1) - no collisions
- Worst case: O(n) - all keys in one bucket

**When to Use:**
- Unknown number of elements
- Need to store more elements than buckets
- Want simple implementation
- Memory overhead acceptable

**2. Open Addressing (Closed Hashing):**

**How It Works:**
- Store key-value pairs directly in hash table array
- When collision occurs, use probe sequence to find next available slot
- No separate data structures - everything in one array

**Advantages:**
- Better cache performance (contiguous memory)
- No extra memory for pointers
- Simpler memory management
- Better for small elements

**Disadvantages:**
- Load factor must be < 1 (can't store more than bucket count)
- Deletion is complex (need tombstone markers or rehashing)
- Clustering can occur (clumps of filled slots)
- Performance degrades quickly as table fills

**Time Complexity:**
- Average: O(1 / (1 - α)) for successful search
- Best case: O(1) - no collisions
- Worst case: O(n) - need to probe entire table

**When to Use:**
- Known maximum number of elements
- Cache performance critical
- Memory overhead matters
- Willing to handle deletion complexity

**Probe Sequences:**

**Linear Probing:**
- Formula: h(k, i) = (h(k) + i) mod m
- Check next slot, then next, then next...
- Simple to implement
- **Primary Clustering**: Consecutive filled slots form clusters
- Clusters grow and slow down operations
- Performance degrades significantly as load factor increases

**Quadratic Probing:**
- Formula: h(k, i) = (h(k) + c1*i + c2*i*i) mod m (often simplified to i*i)
- Check slots at increasing distances: h(k), h(k)+1, h(k)+4, h(k)+9, ...
- Reduces primary clustering
- **Secondary Clustering**: Keys with same initial hash follow same probe sequence
- Can fail to find empty slot even if table not full
- Requires table size to be prime and load factor < 0.5

**Double Hashing:**
- Formula: h(k, i) = (h1(k) + i*h2(k)) mod m
- Uses second hash function to determine probe step
- Best probe sequence - reduces both primary and secondary clustering
- Requires careful choice of h₂ (should be relatively prime to m)
- Most complex but best performance for open addressing

**Load Factor:**

**Definition:**
- α = n/m where n = number of elements, m = number of buckets
- Measures how "full" the hash table is

**For Separate Chaining:**
- α can be > 1 (more elements than buckets)
- Performance degrades gradually as α increases
- Typically resize when α > 2 or 3
- Average chain length = α

**For Open Addressing:**
- α must be < 1 (can't exceed bucket count)
- Performance degrades quickly as α approaches 1
- Typically resize when α > 0.75
- Expected probes for successful search ≈ 1/(1-α)

**Rehashing:**

**When to Rehash:**
- Load factor exceeds threshold (typically 0.75 for open addressing, 2-3 for chaining)
- Performance degrades significantly
- Want to maintain O(1) average operations

**Rehashing Process:**
1. Create new table (typically 2× current size, often next prime number)
2. Recompute hash for all existing elements using new table size
3. Insert all elements into new table
4. Replace old table with new table

**Cost:**
- Time: O(n) - must process all elements
- Space: O(n) - temporary new table
- Amortized: O(1) per operation over many operations

**Why Double Size:**
- Keeps load factor around 0.5 after rehashing
- Reduces frequency of rehashing
- Amortized cost is acceptable

**Choosing a Strategy:**

**Use Separate Chaining When:**
- Number of elements unknown or highly variable
- Want simple implementation
- Memory overhead acceptable
- Need to handle many collisions gracefully

**Use Open Addressing When:**
- Maximum number of elements known
- Cache performance critical
- Want to minimize memory overhead
- Can handle deletion complexity

**Use Double Hashing When:**
- Using open addressing
- Want best performance
- Can implement second hash function
- Performance is critical

**Real-World Implementations:**

**Java HashMap:**
- Uses separate chaining with linked lists
- Converts to balanced tree (TreeMap) when chain length > 8
- Improves worst-case performance from O(n) to O(log n)
- Default load factor: 0.75

**Python Dictionary (3.7+):**
- Uses open addressing with pseudo-random probing
- Maintains insertion order
- Combines hash table with array for ordering
- Very efficient for typical use cases

**Go Map:**
- Uses bucket chaining (each bucket holds 8 key-value pairs)
- Overflow buckets for collisions
- Good balance between chaining and open addressing
- Efficient for Go's use cases

**C++ unordered_map:**
- Implementation-dependent
- Often uses separate chaining
- Some implementations use open addressing

**Best Practices:**

1. **Choose Good Hash Function:**
   - Distributes keys uniformly
   - Fast to compute
   - Avoids patterns in data

2. **Monitor Load Factor:**
   - Resize proactively
   - Don't wait until performance is terrible
   - Consider expected vs actual load

3. **Handle Edge Cases:**
   - Empty table
   - Single element
   - All elements hash to same bucket (test with your data)

4. **Consider Your Data:**
   - Test hash function with actual data
   - Understand collision patterns
   - Choose strategy based on data characteristics`,
					CodeExamples: `// Go: Comprehensive hash table collision resolution implementations

// ===== SEPARATE CHAINING =====
type HashTableChaining struct {
    buckets []*Node
    size    int
    capacity int
    loadFactorThreshold float64
}

type Node struct {
    key   int
    value int
    next  *Node
}

func NewHashTableChaining(capacity int) *HashTableChaining {
    return &HashTableChaining{
        buckets: make([]*Node, capacity),
        capacity: capacity,
        loadFactorThreshold: 2.0,  // Resize when load factor > 2
    }
}

func (ht *HashTableChaining) hash(key int) int {
    // Simple hash function (in practice, use better hash)
    return key % ht.capacity
}

func (ht *HashTableChaining) Put(key, value int) {
    // Check if resize needed
    if float64(ht.size)/float64(ht.capacity) > ht.loadFactorThreshold {
        ht.resize()
    }
    
    index := ht.hash(key)
    node := ht.buckets[index]
    
    // Check if key already exists
    for node != nil {
        if node.key == key {
            node.value = value  // Update existing
            return
        }
        node = node.next
    }
    
    // Insert at head of chain (O(1) insertion)
    newNode := &Node{key: key, value: value, next: ht.buckets[index]}
    ht.buckets[index] = newNode
    ht.size++
}

func (ht *HashTableChaining) Get(key int) (int, bool) {
    index := ht.hash(key)
    node := ht.buckets[index]
    
    for node != nil {
        if node.key == key {
            return node.value, true
        }
        node = node.next
    }
    return 0, false
}

func (ht *HashTableChaining) Delete(key int) bool {
    index := ht.hash(key)
    node := ht.buckets[index]
    
    // Handle head of chain
    if node != nil && node.key == key {
        ht.buckets[index] = node.next
        ht.size--
        return true
    }
    
    // Search in chain
    for node != nil && node.next != nil {
        if node.next.key == key {
            node.next = node.next.next
            ht.size--
            return true
        }
        node = node.next
    }
    return false
}

func (ht *HashTableChaining) resize() {
    oldBuckets := ht.buckets
    oldCapacity := ht.capacity
    
    // Double capacity
    ht.capacity *= 2
    ht.buckets = make([]*Node, ht.capacity)
    ht.size = 0
    
    // Rehash all elements
    for i := 0; i < oldCapacity; i++ {
        node := oldBuckets[i]
        for node != nil {
            ht.Put(node.key, node.value)
            node = node.next
        }
    }
}

// ===== OPEN ADDRESSING - LINEAR PROBING =====
type HashTableLinear struct {
    keys    []int
    values  []int
    deleted []bool  // Tombstone markers
    size    int
    capacity int
    loadFactorThreshold float64
}

const EMPTY = 0  // Assuming 0 is not a valid key

func NewHashTableLinear(capacity int) *HashTableLinear {
    return &HashTableLinear{
        keys:    make([]int, capacity),
        values:  make([]int, capacity),
        deleted: make([]bool, capacity),
        capacity: capacity,
        loadFactorThreshold: 0.75,
    }
}

func (ht *HashTableLinear) hash(key int) int {
    return key % ht.capacity
}

func (ht *HashTableLinear) Put(key, value int) {
    // Check if resize needed
    if float64(ht.size)/float64(ht.capacity) > ht.loadFactorThreshold {
        ht.resize()
    }
    
    index := ht.hash(key)
    
    // Linear probing: check next slot if collision
    for ht.keys[index] != EMPTY && ht.keys[index] != key && !ht.deleted[index] {
        index = (index + 1) % ht.capacity
    }
    
    // Insert or update
    if ht.keys[index] == EMPTY || ht.deleted[index] {
        ht.size++
    }
    ht.keys[index] = key
    ht.values[index] = value
    ht.deleted[index] = false
}

func (ht *HashTableLinear) Get(key int) (int, bool) {
    index := ht.hash(key)
    startIndex := index
    
    // Linear probe until found or empty
    for ht.keys[index] != EMPTY {
        if ht.keys[index] == key && !ht.deleted[index] {
            return ht.values[index], true
        }
        index = (index + 1) % ht.capacity
        if index == startIndex {  // Wrapped around
            break
        }
    }
    return 0, false
}

func (ht *HashTableLinear) Delete(key int) bool {
    index := ht.hash(key)
    startIndex := index
    
    // Find key
    for ht.keys[index] != EMPTY {
        if ht.keys[index] == key && !ht.deleted[index] {
            ht.deleted[index] = true  // Mark as deleted (tombstone)
            ht.size--
            return true
        }
        index = (index + 1) % ht.capacity
        if index == startIndex {
            break
        }
    }
    return false
}

func (ht *HashTableLinear) resize() {
    oldKeys := ht.keys
    oldValues := ht.values
    oldCapacity := ht.capacity
    
    // Double capacity (use next prime for better distribution)
    ht.capacity *= 2
    ht.keys = make([]int, ht.capacity)
    ht.values = make([]int, ht.capacity)
    ht.deleted = make([]bool, ht.capacity)
    ht.size = 0
    
    // Rehash all non-deleted elements
    for i := 0; i < oldCapacity; i++ {
        if oldKeys[i] != EMPTY && !ht.deleted[i] {
            ht.Put(oldKeys[i], oldValues[i])
        }
    }
}

// ===== OPEN ADDRESSING - QUADRATIC PROBING =====
type HashTableQuadratic struct {
    keys    []int
    values  []int
    size    int
    capacity int
}

func NewHashTableQuadratic(capacity int) *HashTableQuadratic {
    // Ensure capacity is prime for quadratic probing
    capacity = nextPrime(capacity)
    return &HashTableQuadratic{
        keys:    make([]int, capacity),
        values:  make([]int, capacity),
        capacity: capacity,
    }
}

func (ht *HashTableQuadratic) hash(key int) int {
    return key % ht.capacity
}

func (ht *HashTableQuadratic) Put(key, value int) {
    index := ht.hash(key)
    i := 0
    
    // Quadratic probing: h(k) + i*i
    for ht.keys[index] != EMPTY && ht.keys[index] != key {
        i++
        index = (ht.hash(key) + i*i) % ht.capacity
        if i >= ht.capacity {  // Table full
            panic("Hash table full")
        }
    }
    
    if ht.keys[index] == EMPTY {
        ht.size++
    }
    ht.keys[index] = key
    ht.values[index] = value
}

// ===== OPEN ADDRESSING - DOUBLE HASHING =====
type HashTableDouble struct {
    keys    []int
    values  []int
    size    int
    capacity int
}

func NewHashTableDouble(capacity int) *HashTableDouble {
    capacity = nextPrime(capacity)  // Prime for double hashing
    return &HashTableDouble{
        keys:    make([]int, capacity),
        values:  make([]int, capacity),
        capacity: capacity,
    }
}

func (ht *HashTableDouble) hash1(key int) int {
    return key % ht.capacity
}

func (ht *HashTableDouble) hash2(key int) int {
    // Second hash function (must be coprime with capacity)
    // Common: use prime smaller than capacity
    prime := 7
    return prime - (key % prime)
}

func (ht *HashTableDouble) Put(key, value int) {
    index := ht.hash1(key)
    step := ht.hash2(key)
    i := 0
    
    // Double hashing: h1(k) + i*h2(k)
    for ht.keys[index] != EMPTY && ht.keys[index] != key {
        i++
        index = (ht.hash1(key) + i*ht.hash2(key)) % ht.capacity
        if i >= ht.capacity {
            panic("Hash table full")
        }
    }
    
    if ht.keys[index] == EMPTY {
        ht.size++
    }
    ht.keys[index] = key
    ht.values[index] = value
}

// Helper: Find next prime number
func nextPrime(n int) int {
    if n <= 2 {
        return 2
    }
    for {
        if isPrime(n) {
            return n
        }
        n++
    }
}

func isPrime(n int) bool {
    if n < 2 {
        return false
    }
    for i := 2; i*i <= n; i++ {
        if n%i == 0 {
            return false
        }
    }
    return true
}

# Python: Comprehensive hash table collision resolution

# ===== SEPARATE CHAINING =====
class HashTableChaining:
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.buckets = [[] for _ in range(capacity)]
        self.size = 0
        self.load_factor_threshold = 2.0
    
    def _hash(self, key):
        return hash(key) % self.capacity
    
    def put(self, key, value):
        # Check if resize needed
        if self.size / self.capacity > self.load_factor_threshold:
            self._resize()
        
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
    
    def get(self, key):
        index = self._hash(key)
        bucket = self.buckets[index]
        for k, v in bucket:
            if k == key:
                return v
        return None
    
    def delete(self, key):
        index = self._hash(key)
        bucket = self.buckets[index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.size -= 1
                return True
        return False
    
    def _resize(self):
        old_buckets = self.buckets
        old_capacity = self.capacity
        
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        
        # Rehash all elements
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)

# ===== OPEN ADDRESSING - LINEAR PROBING =====
class HashTableLinear:
    EMPTY = None
    
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.keys = [self.EMPTY] * capacity
        self.values = [None] * capacity
        self.deleted = [False] * capacity
        self.size = 0
        self.load_factor_threshold = 0.75
    
    def _hash(self, key):
        return hash(key) % self.capacity
    
    def put(self, key, value):
        if self.size / self.capacity > self.load_factor_threshold:
            self._resize()
        
        index = self._hash(key)
        start_index = index
        
        # Linear probing
        while (self.keys[index] != self.EMPTY and 
               self.keys[index] != key and 
               not self.deleted[index]):
            index = (index + 1) % self.capacity
            if index == start_index:
                raise Exception("Hash table full")
        
        if self.keys[index] == self.EMPTY or self.deleted[index]:
            self.size += 1
        self.keys[index] = key
        self.values[index] = value
        self.deleted[index] = False
    
    def get(self, key):
        index = self._hash(key)
        start_index = index
        
        while self.keys[index] != self.EMPTY:
            if self.keys[index] == key and not self.deleted[index]:
                return self.values[index]
            index = (index + 1) % self.capacity
            if index == start_index:
                break
        return None
    
    def delete(self, key):
        index = self._hash(key)
        start_index = index
        
        while self.keys[index] != self.EMPTY:
            if self.keys[index] == key and not self.deleted[index]:
                self.deleted[index] = True  # Tombstone
                self.size -= 1
                return True
            index = (index + 1) % self.capacity
            if index == start_index:
                break
        return False
    
    def _resize(self):
        old_keys = self.keys
        old_values = self.values
        old_capacity = self.capacity
        
        self.capacity *= 2
        self.keys = [self.EMPTY] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity
        self.size = 0
        
        # Rehash non-deleted elements
        for i in range(old_capacity):
            if old_keys[i] != self.EMPTY and not self.deleted[i]:
                self.put(old_keys[i], old_values[i])

# ===== OPEN ADDRESSING - QUADRATIC PROBING =====
class HashTableQuadratic:
    EMPTY = None
    
    def __init__(self, capacity=16):
        self.capacity = self._next_prime(capacity)
        self.keys = [self.EMPTY] * self.capacity
        self.values = [None] * self.capacity
        self.size = 0
    
    def _hash(self, key):
        return hash(key) % self.capacity
    
    def put(self, key, value):
        index = self._hash(key)
        i = 0
        
        # Quadratic probing: h(k) + i*i
        while (self.keys[index] != self.EMPTY and 
               self.keys[index] != key):
            i += 1
            index = (self._hash(key) + i * i) % self.capacity
            if i >= self.capacity:
                raise Exception("Hash table full")
        
        if self.keys[index] == self.EMPTY:
            self.size += 1
        self.keys[index] = key
        self.values[index] = value
    
    def _next_prime(self, n):
        if n <= 2:
            return 2
        while not self._is_prime(n):
            n += 1
        return n
    
    def _is_prime(self, n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

# ===== OPEN ADDRESSING - DOUBLE HASHING =====
class HashTableDouble:
    EMPTY = None
    
    def __init__(self, capacity=16):
        self.capacity = self._next_prime(capacity)
        self.keys = [self.EMPTY] * self.capacity
        self.values = [None] * self.capacity
        self.size = 0
    
    def _hash1(self, key):
        return hash(key) % self.capacity
    
    def _hash2(self, key):
        # Second hash function (coprime with capacity)
        prime = 7
        return prime - (hash(key) % prime)
    
    def put(self, key, value):
        index = self._hash1(key)
        step = self._hash2(key)
        i = 0
        
        # Double hashing: h1(k) + i*h2(k)
        while self.keys[index] != self.EMPTY and self.keys[index] != key:
            i += 1
            index = (self._hash1(key) + i * step) % self.capacity
            if i >= self.capacity:
                raise Exception("Hash table full")
        
        if self.keys[index] == self.EMPTY:
            self.size += 1
        self.keys[index] = key
        self.values[index] = value
    
    def _next_prime(self, n):
        if n <= 2:
            return 2
        while not self._is_prime(n):
            n += 1
        return n
    
    def _is_prime(self, n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

# Performance comparison
if __name__ == "__main__":
    import time
    
    # Test separate chaining
    ht_chaining = HashTableChaining(100)
    start = time.time()
    for i in range(1000):
        ht_chaining.put(i, i*2)
    time_chaining = time.time() - start
    
    # Test linear probing
    ht_linear = HashTableLinear(100)
    start = time.time()
    for i in range(1000):
        ht_linear.put(i, i*2)
    time_linear = time.time() - start
    
    print(f"Chaining: {time_chaining:.6f}s")
    print(f"Linear: {time_linear:.6f}s")
    # Both should be fast, but actual performance depends on data

# Key insights:
# 1. Separate chaining: Simple, handles many collisions, uses extra memory
# 2. Linear probing: Better cache performance, but clustering issues
# 3. Quadratic probing: Reduces clustering, but can fail to find slot
# 4. Double hashing: Best probe sequence, requires prime table size
# 5. Load factor management is crucial for performance
# 6. Rehashing is expensive but amortized over many operations`,
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
					Content: `The two-pointer technique is one of the most powerful optimization techniques in algorithm design. It uses two pointers moving through a data structure to solve problems efficiently, often reducing O(n²) brute force solutions to O(n) or even O(n log n) to O(n).

**Why Two Pointers Work:**

**Elimination Principle:**
- Instead of checking all pairs (O(n²)), pointers eliminate possibilities systematically
- Each pointer movement eliminates a portion of the search space
- Combined movements eliminate even more, achieving O(n) efficiency

**Key Insight:** When data is sorted or has some structure, we can make intelligent decisions about which elements to skip, rather than checking everything.

**Common Patterns:**

**1. Opposite Ends (Converging Pointers):**

**Setup:**
- Left pointer starts at beginning (index 0)
- Right pointer starts at end (index n-1)
- Move them toward each other based on conditions

**When to Use:**
- Sorted arrays (can eliminate half the search space)
- Palindrome checking (compare from both ends)
- Finding pairs that sum to target
- Reversing arrays/strings
- Valid parentheses problems

**How It Works:**
- If condition met with current pair, we're done
- If sum too small, move left pointer right (increase sum)
- If sum too large, move right pointer left (decrease sum)
- Each movement eliminates many possibilities

**Example:** Two sum in sorted array
- If nums[left] + nums[right] < target: left++ (all pairs with current left are too small)
- If nums[left] + nums[right] > target: right-- (all pairs with current right are too large)

**2. Same Direction (Fast/Slow Pointers):**

**Setup:**
- Both pointers start at beginning
- Fast pointer moves ahead, slow pointer follows
- Different speeds or conditions determine movement

**When to Use:**
- Removing duplicates in sorted array
- Cycle detection (Floyd's algorithm)
- Finding middle element
- Partitioning arrays
- Removing elements by value

**How It Works:**
- Slow pointer: "write" position (where we'll place next valid element)
- Fast pointer: "read" position (scans through array)
- Only move slow when we find something to keep

**Example:** Remove duplicates
- Fast pointer scans all elements
- Slow pointer tracks position for next unique element
- When fast finds new unique value, copy to slow position

**3. Sliding Window:**

**Setup:**
- Two pointers define a window [left, right]
- Expand window by moving right pointer
- Contract window by moving left pointer
- Maintain some property of the window

**When to Use:**
- Subarray/substring problems
- Finding minimum/maximum in window
- Problems with "at most K" or "exactly K" constraints
- Longest/shortest subarray problems

**How It Works:**
- Expand window until constraint violated
- Contract window until constraint satisfied again
- Track optimal window during process

**When to Use Two Pointers:**

**Perfect For:**
- **Sorted arrays**: Can eliminate half the search space with each comparison
- **Palindrome checking**: Compare characters from both ends simultaneously
- **Finding pairs/triplets**: Avoid nested loops, use sorted property
- **Removing duplicates**: In-place modification without extra space
- **Merging sorted arrays**: Combine efficiently without extra space
- **Cycle detection**: Floyd's algorithm for linked lists
- **Partitioning**: Separate elements based on condition

**Not Ideal For:**
- Unsorted arrays (usually need to sort first)
- Problems requiring checking all pairs
- When order doesn't matter for elimination

**Key Insight:** Instead of checking all pairs O(n²), use pointers to eliminate possibilities systematically, achieving O(n). The power comes from making decisions that eliminate many possibilities at once.

**Common Mistakes:**

**1. Moving Pointers Incorrectly:**
- Moving both pointers in same direction when should move toward each other
- Not updating pointers in all code branches
- Infinite loops from incorrect termination conditions

**2. Off-by-One Errors:**
- Using <= instead of < in loop condition
- Not handling edge cases (empty array, single element)
- Incorrect pointer initialization

**3. Edge Cases:**
- Empty array
- Single element
- All elements same
- No valid solution

**4. Termination Conditions:**
- left < right vs left <= right
- When to stop (found solution vs exhausted search space)
- Handling case when pointers meet

**Best Practices:**

1. **Start Simple:**
   - Use brute force first to understand problem
   - Then optimize with two pointers
   - Verify correctness with test cases

2. **Handle Edge Cases:**
   - Empty arrays
   - Single element
   - All elements same
   - No solution exists

3. **Think About Pointer Movement:**
   - Which pointer to move and why?
   - What possibilities does this eliminate?
   - Will this eventually find solution or exhaust search?

4. **Use Sorted Data:**
   - Sort array first if needed
   - Sorted data enables elimination logic
   - Often worth O(n log n) sort to get O(n) solution`,
					CodeExamples: `// Go: Comprehensive two-pointer techniques

// ===== OPPOSITE ENDS (CONVERGING POINTERS) =====
// Two sum in sorted array
func twoSumSorted(nums []int, target int) []int {
    left, right := 0, len(nums)-1
    for left < right {
        sum := nums[left] + nums[right]
        if sum == target {
            return []int{left, right}
        } else if sum < target {
            left++  // Need larger sum, move left right
        } else {
            right--  // Need smaller sum, move right left
        }
    }
    return nil  // No solution
}

// Reverse array in-place
func reverseArray(nums []int) {
    left, right := 0, len(nums)-1
    for left < right {
        nums[left], nums[right] = nums[right], nums[left]
        left++
        right--
    }
}

// Check if string is palindrome
func isPalindrome(s string) bool {
    left, right := 0, len(s)-1
    for left < right {
        // Skip non-alphanumeric
        for left < right && !isAlphanumeric(s[left]) {
            left++
        }
        for left < right && !isAlphanumeric(s[right]) {
            right--
        }
        if toLower(s[left]) != toLower(s[right]) {
            return false
        }
        left++
        right--
    }
    return true
}

// Container with most water
func maxArea(height []int) int {
    left, right := 0, len(height)-1
    maxArea := 0
    
    while left < right {
        width := right - left
        area := width * min(height[left], height[right])
        maxArea = max(maxArea, area)
        
        // Move pointer with smaller height
        if height[left] < height[right] {
            left++
        } else {
            right--
        }
    }
    return maxArea
}

// ===== SAME DIRECTION (FAST/SLOW POINTERS) =====
// Remove duplicates from sorted array
func removeDuplicates(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    slow := 0
    for fast := 1; fast < len(nums); fast++ {
        if nums[fast] != nums[slow] {
            slow++
            nums[slow] = nums[fast]  // Place unique element
        }
    }
    return slow + 1  // Length of unique array
}

// Remove element by value
func removeElement(nums []int, val int) int {
    slow := 0
    for fast := 0; fast < len(nums); fast++ {
        if nums[fast] != val {
            nums[slow] = nums[fast]
            slow++
        }
    }
    return slow
}

// Move zeros to end
func moveZeros(nums []int) {
    slow := 0
    // Move all non-zeros to front
    for fast := 0; fast < len(nums); fast++ {
        if nums[fast] != 0 {
            nums[slow] = nums[fast]
            slow++
        }
    }
    // Fill rest with zeros
    for slow < len(nums) {
        nums[slow] = 0
        slow++
    }
}

// Partition array (Dutch National Flag)
func partitionColors(nums []int) {
    // Three pointers: left (0s), curr (processing), right (2s)
    left, curr, right := 0, 0, len(nums)-1
    
    for curr <= right {
        if nums[curr] == 0 {
            nums[left], nums[curr] = nums[curr], nums[left]
            left++
            curr++
        } else if nums[curr] == 2 {
            nums[curr], nums[right] = nums[right], nums[curr]
            right--  // Don't increment curr (need to check swapped element)
        } else {
            curr++  // nums[curr] == 1, leave it
        }
    }
}

// ===== SLIDING WINDOW =====
// Longest substring without repeating characters
func lengthOfLongestSubstring(s string) int {
    charMap := make(map[byte]int)
    left := 0
    maxLen := 0
    
    for right := 0; right < len(s); right++ {
        char := s[right]
        // If character seen, move left pointer
        if idx, exists := charMap[char]; exists && idx >= left {
            left = idx + 1
        }
        charMap[char] = right
        maxLen = max(maxLen, right-left+1)
    }
    return maxLen
}

// Minimum window substring
func minWindow(s string, t string) string {
    need := make(map[byte]int)
    for _, c := range t {
        need[byte(c)]++
    }
    
    window := make(map[byte]int)
    left, right := 0, 0
    valid := 0
    start, minLen := 0, math.MaxInt32
    
    for right < len(s) {
        c := s[right]
        right++
        
        if need[c] > 0 {
            window[c]++
            if window[c] == need[c] {
                valid++
            }
        }
        
        // Shrink window
        for valid == len(need) {
            if right-left < minLen {
                start = left
                minLen = right - left
            }
            
            d := s[left]
            left++
            
            if need[d] > 0 {
                if window[d] == need[d] {
                    valid--
                }
                window[d]--
            }
        }
    }
    
    if minLen == math.MaxInt32 {
        return ""
    }
    return s[start : start+minLen]
}

// Helper functions
func isAlphanumeric(c byte) bool {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
}

func toLower(c byte) byte {
    if c >= 'A' && c <= 'Z' {
        return c + 32
    }
    return c
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

# Python: Comprehensive two-pointer techniques

# ===== OPPOSITE ENDS (CONVERGING POINTERS) =====
# Two sum in sorted array
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        sum_val = nums[left] + nums[right]
        if sum_val == target:
            return [left, right]
        elif sum_val < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum
    return None

# Reverse array in-place
def reverse_array(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1

# Check if palindrome
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        # Skip non-alphanumeric
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True

# Container with most water
def max_area(height):
    left, right = 0, len(height) - 1
    max_area_val = 0
    
    while left < right:
        width = right - left
        area = width * min(height[left], height[right])
        max_area_val = max(max_area_val, area)
        
        # Move pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area_val

# ===== SAME DIRECTION (FAST/SLOW POINTERS) =====
# Remove duplicates
def remove_duplicates(nums):
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1

# Remove element by value
def remove_element(nums, val):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
    return slow

# Move zeros to end
def move_zeros(nums):
    slow = 0
    # Move non-zeros to front
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1
    # Fill rest with zeros
    for i in range(slow, len(nums)):
        nums[i] = 0

# Partition (Dutch National Flag)
def partition_colors(nums):
    left = curr = 0
    right = len(nums) - 1
    
    while curr <= right:
        if nums[curr] == 0:
            nums[left], nums[curr] = nums[curr], nums[left]
            left += 1
            curr += 1
        elif nums[curr] == 2:
            nums[curr], nums[right] = nums[right], nums[curr]
            right -= 1  # Don't increment curr
        else:
            curr += 1  # nums[curr] == 1

# ===== SLIDING WINDOW =====
# Longest substring without repeating characters
def length_of_longest_substring(s):
    char_map = {}
    left = 0
    max_len = 0
    
    for right in range(len(s)):
        char = s[right]
        # Move left if character seen
        if char in char_map and char_map[char] >= left:
            left = char_map[char] + 1
        char_map[char] = right
        max_len = max(max_len, right - left + 1)
    
    return max_len

# Minimum window substring
def min_window(s, t):
    from collections import Counter
    need = Counter(t)
    window = {}
    left = right = 0
    valid = 0
    start = 0
    min_len = float('inf')
    
    while right < len(s):
        c = s[right]
        right += 1
        
        if c in need:
            window[c] = window.get(c, 0) + 1
            if window[c] == need[c]:
                valid += 1
        
        while valid == len(need):
            if right - left < min_len:
                start = left
                min_len = right - left
            
            d = s[left]
            left += 1
            
            if d in need:
                if window[d] == need[d]:
                    valid -= 1
                window[d] -= 1
    
    return "" if min_len == float('inf') else s[start:start+min_len]

# Key insights:
# 1. Opposite ends: Eliminate possibilities by moving toward each other
# 2. Same direction: Use slow as write position, fast as read position
# 3. Sliding window: Expand/contract to maintain constraints
# 4. Two pointers often reduce O(n²) to O(n)
# 5. Works best with sorted data or structured problems`,
				},
				{
					Title: "Fast and Slow Pointers",
					Content: `The fast/slow pointer technique (also called Floyd's cycle detection algorithm or the "tortoise and hare" algorithm) uses two pointers moving at different speeds to solve problems efficiently. This technique is particularly powerful for linked list problems and can solve many problems in O(n) time with O(1) space.

**Why Fast/Slow Pointers Work:**

**Mathematical Foundation:**
- Fast pointer moves at speed 2, slow pointer moves at speed 1
- Relative speed: Fast gains 1 position per step on slow
- If cycle exists, fast will eventually catch up to slow
- If no cycle, fast reaches end first

**Key Insight:** The different speeds create a "race" where the fast pointer will either:
1. Reach the end (no cycle)
2. Catch up to slow pointer (cycle exists)

**Common Applications:**

**1. Cycle Detection (Floyd's Algorithm):**

**Problem:** Detect if a linked list has a cycle.

**Algorithm:**
- Initialize both pointers at head
- Fast moves 2 steps, slow moves 1 step per iteration
- If they meet, cycle exists
- If fast reaches null, no cycle

**Why It Works:**
- If cycle length is L and distance to cycle start is D:
- When slow enters cycle (after D steps), fast is D steps ahead
- Fast catches up at rate of 1 step per iteration
- They meet after at most L iterations

**Time Complexity:** O(n) - each pointer visits each node at most once
**Space Complexity:** O(1) - only two pointers

**2. Finding Cycle Start:**

**Problem:** If cycle exists, find the node where cycle starts.

**Algorithm:**
1. Use Floyd's algorithm to find meeting point
2. Move one pointer to head
3. Move both pointers one step at a time
4. They meet at cycle start

**Why It Works:**
- Mathematical proof shows meeting point is equidistant from head and cycle start
- Moving both one step ensures they meet at cycle start

**3. Finding Middle Element:**

**Problem:** Find the middle element of a linked list.

**Algorithm:**
- Fast moves 2 steps, slow moves 1 step
- When fast reaches end, slow is at middle
- For even length, slow is at second middle (or first, depending on implementation)

**Variations:**
- First middle: Stop when fast.Next == nil
- Second middle: Stop when fast == nil

**4. Finding kth Element from End:**

**Problem:** Find the kth element from the end of a linked list.

**Algorithm:**
1. Move fast pointer k steps ahead
2. Move both pointers one step at a time
3. When fast reaches end, slow is at kth from end

**Edge Cases:**
- k > list length
- k = 0 (last element)
- k = list length (first element)

**5. Palindrome Checking (Linked List):**

**Problem:** Check if linked list is palindrome.

**Algorithm:**
1. Find middle using fast/slow pointers
2. Reverse second half
3. Compare first half with reversed second half
4. Restore list (if needed)

**6. Removing Duplicates:**

**Problem:** Remove duplicates from sorted array/list.

**Algorithm:**
- Slow pointer: position to write next unique element
- Fast pointer: scans through array
- Only move slow when fast finds new unique value

**Pattern:**
- Initialize slow = 0, fast = 1
- For each fast position:
  - If nums[fast] != nums[slow]: move slow and copy value
  - Fast always increments

**When to Use:**

**Perfect For:**
- Linked list problems
- Cycle detection
- Finding middle/kth element
- Problems requiring O(1) space
- When you can't use hash map (space constraint)

**Not Ideal For:**
- Problems requiring random access
- When you need to check all pairs
- Problems better solved with hash map

**Best Practices:**

1. **Always Check for Null:**
   - Check fast != nil && fast.Next != nil before accessing
   - Prevents null pointer exceptions

2. **Handle Edge Cases:**
   - Empty list
   - Single element
   - Two elements
   - No cycle vs cycle

3. **Understand Meeting Point:**
   - In cycle detection, meeting point is inside cycle
   - Not necessarily at cycle start
   - Use second phase to find cycle start

4. **Choose Right Speed Ratio:**
   - Usually 2:1 (fast:slow)
   - Other ratios possible but less common
   - 2:1 ensures O(n) time complexity`,
					CodeExamples: `// Go: Comprehensive fast/slow pointer techniques

// ListNode definition
type ListNode struct {
    Val  int
    Next *ListNode
}

// ===== CYCLE DETECTION =====
// Detect if linked list has cycle (Floyd's algorithm)
func hasCycle(head *ListNode) bool {
    if head == nil || head.Next == nil {
        return false
    }
    slow, fast := head, head
    
    // Fast moves 2 steps, slow moves 1 step
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        
        if slow == fast {
            return true  // Cycle detected - they met
        }
    }
    return false  // Fast reached end - no cycle
}

// Find cycle start node
func detectCycleStart(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return nil
    }
    
    // Phase 1: Find meeting point
    slow, fast := head, head
    hasCycle := false
    
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        if slow == fast {
            hasCycle = true
            break
        }
    }
    
    if !hasCycle {
        return nil  // No cycle
    }
    
    // Phase 2: Find cycle start
    // Move one pointer to head, both move one step
    slow = head
    for slow != fast {
        slow = slow.Next
        fast = fast.Next
    }
    return slow  // Meeting point is cycle start
}

// ===== FINDING MIDDLE ELEMENT =====
// Find middle element (first middle for even length)
func findMiddle(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    slow, fast := head, head
    
    // Stop when fast.Next is nil (first middle)
    for fast.Next != nil && fast.Next.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    return slow
}

// Find second middle (for even length lists)
func findSecondMiddle(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    slow, fast := head, head
    
    // Stop when fast is nil (second middle)
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    return slow
}

// ===== KTH ELEMENT FROM END =====
// Find kth element from end (1-indexed)
func findKthFromEnd(head *ListNode, k int) *ListNode {
    if head == nil || k <= 0 {
        return nil
    }
    
    // Move fast k steps ahead
    fast := head
    for i := 0; i < k; i++ {
        if fast == nil {
            return nil  // k > list length
        }
        fast = fast.Next
    }
    
    // Move both one step at a time
    slow := head
    for fast != nil {
        slow = slow.Next
        fast = fast.Next
    }
    return slow  // Slow is kth from end
}

// ===== PALINDROME CHECKING =====
// Check if linked list is palindrome
func isPalindromeList(head *ListNode) bool {
    if head == nil || head.Next == nil {
        return true
    }
    
    // Step 1: Find middle
    slow, fast := head, head
    for fast.Next != nil && fast.Next.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    
    // Step 2: Reverse second half
    secondHalf := reverseList(slow.Next)
    slow.Next = nil  // Split list
    
    // Step 3: Compare halves
    firstHalf := head
    for secondHalf != nil {
        if firstHalf.Val != secondHalf.Val {
            return false
        }
        firstHalf = firstHalf.Next
        secondHalf = secondHalf.Next
    }
    return true
}

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

// ===== REMOVING DUPLICATES (ARRAY) =====
// Remove duplicates from sorted array
func removeDuplicatesFastSlow(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    slow := 0
    for fast := 1; fast < len(nums); fast++ {
        if nums[fast] != nums[slow] {
            slow++
            nums[slow] = nums[fast]
        }
    }
    return slow + 1
}

// ===== PARTITIONING =====
// Partition array around pivot
func partitionFastSlow(nums []int, pivot int) int {
    slow := 0
    for fast := 0; fast < len(nums); fast++ {
        if nums[fast] < pivot {
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow++
        }
    }
    return slow  // Position where elements >= pivot start
}

# Python: Comprehensive fast/slow pointer techniques

# ListNode definition
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# ===== CYCLE DETECTION =====
# Detect if linked list has cycle
def has_cycle(head):
    if not head or not head.next:
        return False
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True  # Cycle detected
    
    return False  # No cycle

# Find cycle start
def detect_cycle_start(head):
    if not head or not head.next:
        return None
    
    # Phase 1: Find meeting point
    slow = fast = head
    has_cycle_flag = False
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            has_cycle_flag = True
            break
    
    if not has_cycle_flag:
        return None
    
    # Phase 2: Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow

# ===== FINDING MIDDLE ELEMENT =====
# Find middle element
def find_middle(head):
    if not head:
        return None
    
    slow = fast = head
    
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow

# Find second middle
def find_second_middle(head):
    if not head:
        return None
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow

# ===== KTH ELEMENT FROM END =====
# Find kth element from end
def find_kth_from_end(head, k):
    if not head or k <= 0:
        return None
    
    # Move fast k steps ahead
    fast = head
    for _ in range(k):
        if not fast:
            return None  # k > list length
        fast = fast.next
    
    # Move both one step at a time
    slow = head
    while fast:
        slow = slow.next
        fast = fast.next
    
    return slow

# ===== PALINDROME CHECKING =====
# Check if linked list is palindrome
def is_palindrome_list(head):
    if not head or not head.next:
        return True
    
    # Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    second_half = reverse_list(slow.next)
    slow.next = None
    
    # Compare halves
    first_half = head
    while second_half:
        if first_half.val != second_half.val:
            return False
        first_half = first_half.next
        second_half = second_half.next
    
    return True

def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev

# ===== REMOVING DUPLICATES =====
# Remove duplicates from sorted array
def remove_duplicates_fast_slow(nums):
    if not nums:
        return 0
    
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1

# Key insights:
# 1. Fast/slow pointers solve many linked list problems in O(n) time, O(1) space
# 2. Cycle detection: Fast catches slow if cycle exists
# 3. Middle element: Fast reaches end when slow is at middle
# 4. kth from end: Maintain k distance between pointers
# 5. Always check for null before accessing next pointer`,
				},
				{
					Title: "Three Sum Pattern",
					Content: `The three sum problem is a classic extension of the two sum pattern. It demonstrates how to combine sorting with two-pointer techniques to solve problems efficiently. This pattern appears frequently in coding interviews and can be extended to k-sum problems.

**Problem Statement:**
Given an array of integers, find all unique triplets (i, j, k) such that nums[i] + nums[j] + nums[k] = target (often target = 0).

**Example:**
Input: nums = [-1, 0, 1, 2, -1, -4]
Output: [[-1, -1, 2], [-1, 0, 1]]

**Why Sorting Helps:**

**Without Sorting:**
- Need to check all triplets: O(n*n*n)
- Difficult to avoid duplicates
- No way to eliminate possibilities

**With Sorting:**
- Can use two-pointer technique: O(n*n)
- Easy to skip duplicates
- Can eliminate many possibilities at once

**Algorithm:**

**Step 1: Sort Array**
- Sort in ascending order: O(n log n)
- Enables two-pointer technique
- Makes duplicate detection easier

**Step 2: Fix First Element**
- For each element at index i (from 0 to n-3):
  - This becomes the first element of triplet
  - Skip duplicates: if nums[i] == nums[i-1], continue

**Step 3: Two-Pointer Search**
- Set left = i+1, right = n-1
- Target for two pointers: target - nums[i]
- While left < right:
  - Calculate sum = nums[left] + nums[right]
  - If sum == target: Found triplet! Add to result
  - If sum < target: left++ (need larger sum)
  - If sum > target: right-- (need smaller sum)

**Step 4: Handle Duplicates**
- After finding a triplet, skip all duplicates
- Move left until nums[left] != nums[left+1]
- Move right until nums[right] != nums[right-1]
- Prevents duplicate triplets in result

**Time Complexity:**
- Sorting: O(n log n)
- Outer loop: O(n) iterations
- Two pointers: O(n) per iteration
- Total: O(n*n) - dominates sorting for large n

**Space Complexity:**
- O(1) extra space (excluding output)
- O(k) for output where k = number of triplets

**Key Insights:**

**1. Sorting is Essential:**
- Enables two-pointer technique
- Makes duplicate handling straightforward
- Worth O(n log n) cost to get O(n*n) solution

**2. Duplicate Handling:**
- Must skip duplicates for first element
- Must skip duplicates after finding triplet
- Critical for correctness

**3. Pointer Movement:**
- Move left when sum too small (need larger values)
- Move right when sum too large (need smaller values)
- Move both when found triplet (after skipping duplicates)

**4. Extensibility:**
- Can extend to k-sum using recursion
- Fix k-2 elements, use two pointers for last two
- General pattern: O(n^(k-1)) for k-sum

**Variations:**

**1. Three Sum Closest:**
- Find triplet with sum closest to target
- Track minimum difference
- Similar algorithm, but track closest instead of exact match

**2. Three Sum Smaller:**
- Count triplets with sum less than target
- When sum < target, all pairs with current left are valid
- Count: right - left

**3. Four Sum:**
- Extend to four elements
- Fix first two elements, use two pointers for last two
- O(n*n*n) time complexity

**4. K-Sum:**
- Generalize to k elements
- Use recursion: fix k-2 elements, two pointers for last two
- O(n^(k-1)) time complexity

**Common Mistakes:**

**1. Forgetting to Sort:**
- Two-pointer technique requires sorted array
- Won't work correctly without sorting

**2. Not Handling Duplicates:**
- Will produce duplicate triplets
- Must skip duplicates for first element
- Must skip duplicates after finding triplet

**3. Incorrect Pointer Movement:**
- Moving wrong pointer
- Not moving both pointers after finding triplet
- Not skipping duplicates correctly

**4. Edge Cases:**
- Array length < 3
- All elements same
- No valid triplets

**Best Practices:**

1. **Always Sort First:**
   - Required for two-pointer technique
   - Makes problem much easier

2. **Handle Duplicates Carefully:**
   - Skip duplicates for first element
   - Skip duplicates after finding triplet
   - Test with arrays containing many duplicates

3. **Think About Pointer Movement:**
   - Which pointer to move and why?
   - What possibilities does this eliminate?

4. **Optimize When Possible:**
   - Early termination if nums[i] > 0 and target = 0 (all positive)
   - Skip if nums[i] + nums[i+1] + nums[i+2] > target (too large)`,
					CodeExamples: `// Go: Comprehensive three sum pattern

// ===== THREE SUM (EXACT MATCH) =====
// Find all unique triplets that sum to target
func threeSum(nums []int, target int) [][]int {
    sort.Ints(nums)  // Essential: sort first
    result := [][]int{}
    n := len(nums)
    
    for i := 0; i < n-2; i++ {
        // Skip duplicates for first element
        if i > 0 && nums[i] == nums[i-1] {
            continue
        }
        
        // Early termination optimization
        if nums[i] > 0 && target <= 0 {
            break  // All remaining elements positive
        }
        
        left, right := i+1, n-1
        twoSumTarget := target - nums[i]  // Target for two pointers
        
        for left < right {
            sum := nums[left] + nums[right]
            
            if sum == twoSumTarget {
                // Found triplet!
                result = append(result, []int{nums[i], nums[left], nums[right]})
                
                // Skip duplicates for left pointer
                for left < right && nums[left] == nums[left+1] {
                    left++
                }
                // Skip duplicates for right pointer
                for left < right && nums[right] == nums[right-1] {
                    right--
                }
                // Move both pointers
                left++
                right--
            } else if sum < twoSumTarget {
                left++  // Need larger sum, move left right
            } else {
                right--  // Need smaller sum, move right left
            }
        }
    }
    return result
}

// Three sum with target = 0 (most common)
func threeSumZero(nums []int) [][]int {
    return threeSum(nums, 0)
}

// ===== THREE SUM CLOSEST =====
// Find triplet with sum closest to target
func threeSumClosest(nums []int, target int) int {
    sort.Ints(nums)
    n := len(nums)
    closestSum := nums[0] + nums[1] + nums[2]
    minDiff := abs(closestSum - target)
    
    for i := 0; i < n-2; i++ {
        left, right := i+1, n-1
        
        for left < right {
            sum := nums[i] + nums[left] + nums[right]
            diff := abs(sum - target)
            
            if diff < minDiff {
                minDiff = diff
                closestSum = sum
            }
            
            if sum < target {
                left++  // Need larger sum
            } else if sum > target {
                right--  // Need smaller sum
            } else {
                return sum  // Exact match found
            }
        }
    }
    return closestSum
}

// ===== THREE SUM SMALLER =====
// Count triplets with sum less than target
func threeSumSmaller(nums []int, target int) int {
    sort.Ints(nums)
    n := len(nums)
    count := 0
    
    for i := 0; i < n-2; i++ {
        left, right := i+1, n-1
        
        for left < right {
            sum := nums[i] + nums[left] + nums[right]
            
            if sum < target {
                // All pairs with current left are valid
                // (left, left+1), (left, left+2), ..., (left, right)
                count += right - left
                left++
            } else {
                right--  // Sum too large, need smaller
            }
        }
    }
    return count
}

// ===== FOUR SUM =====
// Find all unique quadruplets that sum to target
func fourSum(nums []int, target int) [][]int {
    sort.Ints(nums)
    result := [][]int{}
    n := len(nums)
    
    for i := 0; i < n-3; i++ {
        // Skip duplicates for first element
        if i > 0 && nums[i] == nums[i-1] {
            continue
        }
        
        for j := i+1; j < n-2; j++ {
            // Skip duplicates for second element
            if j > i+1 && nums[j] == nums[j-1] {
                continue
            }
            
            left, right := j+1, n-1
            twoSumTarget := target - nums[i] - nums[j]
            
            for left < right {
                sum := nums[left] + nums[right]
                
                if sum == twoSumTarget {
                    result = append(result, []int{nums[i], nums[j], nums[left], nums[right]})
                    
                    // Skip duplicates
                    for left < right && nums[left] == nums[left+1] {
                        left++
                    }
                    for left < right && nums[right] == nums[right-1] {
                        right--
                    }
                    left++
                    right--
                } else if sum < twoSumTarget {
                    left++
                } else {
                    right--
                }
            }
        }
    }
    return result
}

// ===== K-SUM (GENERALIZED) =====
// General k-sum solution using recursion
func kSum(nums []int, target int, k int) [][]int {
    sort.Ints(nums)
    result := [][]int{}
    current := []int{}
    kSumHelper(nums, target, k, 0, current, &result)
    return result
}

func kSumHelper(nums []int, target int, k int, start int, current []int, result *[][]int) {
    n := len(nums)
    
    // Base case: two sum
    if k == 2 {
        left, right := start, n-1
        for left < right {
            sum := nums[left] + nums[right]
            if sum == target {
                // Found solution
                solution := make([]int, len(current))
                copy(solution, current)
                solution = append(solution, nums[left], nums[right])
                *result = append(*result, solution)
                
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
                left++
            } else {
                right--
            }
        }
        return
    }
    
    // Recursive case: fix one element, solve (k-1)-sum
    for i := start; i < n-k+1; i++ {
        // Skip duplicates
        if i > start && nums[i] == nums[i-1] {
            continue
        }
        
        // Early termination
        if nums[i]*k > target || nums[n-1]*k < target {
            break
        }
        
        current = append(current, nums[i])
        kSumHelper(nums, target-nums[i], k-1, i+1, current, result)
        current = current[:len(current)-1]  // Backtrack
    }
}

// Helper function
func abs(x int) int {
    if x < 0 {
        return -x
    }
    return x
}

# Python: Comprehensive three sum pattern

# ===== THREE SUM (EXACT MATCH) =====
def three_sum(nums, target):
    """Find all unique triplets that sum to target."""
    nums.sort()  # Essential: sort first
    result = []
    n = len(nums)
    
    for i in range(n - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        # Early termination
        if nums[i] > 0 and target <= 0:
            break
        
        left, right = i + 1, n - 1
        two_sum_target = target - nums[i]
        
        while left < right:
            sum_val = nums[left] + nums[right]
            
            if sum_val == two_sum_target:
                # Found triplet
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif sum_val < two_sum_target:
                left += 1  # Need larger sum
            else:
                right -= 1  # Need smaller sum
    
    return result

# Three sum with target = 0
def three_sum_zero(nums):
    return three_sum(nums, 0)

# ===== THREE SUM CLOSEST =====
def three_sum_closest(nums, target):
    """Find triplet with sum closest to target."""
    nums.sort()
    n = len(nums)
    closest_sum = nums[0] + nums[1] + nums[2]
    min_diff = abs(closest_sum - target)
    
    for i in range(n - 2):
        left, right = i + 1, n - 1
        
        while left < right:
            sum_val = nums[i] + nums[left] + nums[right]
            diff = abs(sum_val - target)
            
            if diff < min_diff:
                min_diff = diff
                closest_sum = sum_val
            
            if sum_val < target:
                left += 1
            elif sum_val > target:
                right -= 1
            else:
                return sum_val  # Exact match
    
    return closest_sum

# ===== THREE SUM SMALLER =====
def three_sum_smaller(nums, target):
    """Count triplets with sum less than target."""
    nums.sort()
    n = len(nums)
    count = 0
    
    for i in range(n - 2):
        left, right = i + 1, n - 1
        
        while left < right:
            sum_val = nums[i] + nums[left] + nums[right]
            
            if sum_val < target:
                # All pairs with current left are valid
                count += right - left
                left += 1
            else:
                right -= 1
    
    return count

# ===== FOUR SUM =====
def four_sum(nums, target):
    """Find all unique quadruplets that sum to target."""
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j-1]:
                continue
            
            left, right = j + 1, n - 1
            two_sum_target = target - nums[i] - nums[j]
            
            while left < right:
                sum_val = nums[left] + nums[right]
                
                if sum_val == two_sum_target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                elif sum_val < two_sum_target:
                    left += 1
                else:
                    right -= 1
    
    return result

# ===== K-SUM (GENERALIZED) =====
def k_sum(nums, target, k):
    """General k-sum solution using recursion."""
    nums.sort()
    result = []
    
    def k_sum_helper(start, k, target, current):
        n = len(nums)
        
        # Base case: two sum
        if k == 2:
            left, right = start, n - 1
            while left < right:
                sum_val = nums[left] + nums[right]
                if sum_val == target:
                    result.append(current + [nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif sum_val < target:
                    left += 1
                else:
                    right -= 1
            return
        
        # Recursive case
        for i in range(start, n - k + 1):
            if i > start and nums[i] == nums[i-1]:
                continue
            
            # Early termination
            if nums[i] * k > target or nums[-1] * k < target:
                break
            
            k_sum_helper(i + 1, k - 1, target - nums[i], current + [nums[i]])
    
    k_sum_helper(0, k, target, [])
    return result

# Example usage
if __name__ == "__main__":
    nums = [-1, 0, 1, 2, -1, -4]
    print(f"Three sum zero: {three_sum_zero(nums)}")
    # Output: [[-1, -1, 2], [-1, 0, 1]]
    
    nums2 = [-1, 2, 1, -4]
    print(f"Three sum closest to 1: {three_sum_closest(nums2, 1)}")
    # Output: 2 (closest sum)

# Key insights:
# 1. Sorting is essential for two-pointer technique
# 2. Skip duplicates carefully to avoid duplicate results
# 3. Can extend to k-sum using recursion
# 4. Time complexity: O(n^(k-1)) for k-sum
# 5. Early termination optimizations can improve performance`,
				},
				{
					Title: "Partitioning Arrays",
					Content: `Partitioning arrays is a fundamental technique that involves rearranging elements so that elements satisfying a condition come before elements that don't. This technique is essential for algorithms like quicksort and appears in many array manipulation problems.

**What is Partitioning?**

Partitioning divides an array into sections based on a condition. The goal is to rearrange elements so that:
- Elements satisfying the condition are in one section
- Elements not satisfying the condition are in another section
- The relative order within each section may or may not matter

**Partition Patterns:**

**1. Two-Way Partition:**

**Concept:**
- Divide array into two groups based on a boolean condition
- All elements satisfying condition come first
- All elements not satisfying condition come second

**Common Examples:**
- Move all zeros to end
- Separate even and odd numbers
- Move all negative numbers to beginning
- Partition around a value (less than vs greater than)

**Technique:**
- Use two pointers: writePos (where to place next valid element) and i (scanning through array)
- When condition satisfied, swap to writePos and increment writePos
- When condition not satisfied, just increment i

**Time Complexity:** O(n) - single pass
**Space Complexity:** O(1) - in-place

**2. Three-Way Partition (Dutch National Flag):**

**Concept:**
- Divide array into three groups: elements < pivot, == pivot, > pivot
- Named after Dutch flag (three colors: red, white, blue)

**Why It's Useful:**
- Optimizes quicksort when many duplicates exist
- Reduces worst-case time complexity
- Handles three-way conditions efficiently

**Technique:**
- Use three pointers:
  - left: boundary for elements < pivot
  - current: scanning pointer
  - right: boundary for elements > pivot
- Maintain invariant: [0...left) < pivot, [left...current) == pivot, (right...n-1] > pivot

**Key Insight:** When swapping with right, don't increment current because swapped element hasn't been checked yet.

**Time Complexity:** O(n) - single pass
**Space Complexity:** O(1) - in-place

**3. Lomuto Partition:**

**Concept:**
- Classic partition scheme used in quicksort
- Simpler to understand and implement
- Less efficient than Hoare partition (more swaps)

**How It Works:**
- Choose pivot (usually last element)
- Maintain invariant: elements [low...i] are <= pivot
- Scan with j from low to high-1
- When element <= pivot, swap with element at i and increment i
- Finally, swap pivot to position i

**Invariant:**
- After partition, pivot is at correct position
- All elements before pivot are <= pivot
- All elements after pivot are > pivot

**Time Complexity:** O(n)
**Space Complexity:** O(1)

**4. Hoare Partition:**

**Concept:**
- More efficient partition scheme
- Uses two pointers converging from both ends
- Fewer swaps than Lomuto partition

**How It Works:**
- Choose pivot (usually first element)
- Left pointer starts at low+1, right at high
- Move left right until element > pivot
- Move right left until element <= pivot
- Swap if left < right, then continue
- Finally, swap pivot with right

**Advantages:**
- Fewer swaps on average
- Better cache performance
- More efficient for quicksort

**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Common Partition Problems:**

**1. Move Zeros to End:**
- Keep all non-zeros, move zeros to end
- Use two-way partition with condition: nums[i] != 0

**2. Separate Even and Odd:**
- Even numbers first, odd numbers second
- Use two-way partition with condition: nums[i] % 2 == 0

**3. Partition Around Pivot:**
- Elements < pivot before, >= pivot after
- Use Lomuto or Hoare partition

**4. Sort Colors (Dutch Flag):**
- Three colors: 0, 1, 2
- Use three-way partition

**5. Rearrange by Sign:**
- Positive numbers first, negative numbers second
- Or maintain relative order: use two-way partition carefully

**Key Techniques:**

**1. Two-Pointer Approach:**
- One pointer: "write" position (where to place next valid element)
- One pointer: "read" position (scanning through array)
- Only move write pointer when condition satisfied

**2. Swap Strategy:**
- Swap elements to maintain partition invariant
- Can swap with write position or use converging pointers
- Be careful about which elements to swap

**3. Invariant Maintenance:**
- Clearly define what invariant you're maintaining
- Ensure invariant holds after each operation
- Use invariant to prove correctness

**4. Edge Case Handling:**
- Empty arrays
- Single element
- All elements same
- All elements satisfy condition
- No elements satisfy condition

**Best Practices:**

1. **Choose Right Partition Scheme:**
   - Two-way for simple boolean conditions
   - Three-way for three categories or many duplicates
   - Lomuto for simplicity
   - Hoare for efficiency

2. **Maintain Invariants:**
   - Clearly define what each section contains
   - Ensure invariant holds throughout algorithm
   - Use invariant to verify correctness

3. **Handle Edge Cases:**
   - Test with empty array
   - Test with single element
   - Test with all elements same
   - Test with extreme values

4. **Optimize When Possible:**
   - Use Hoare partition for better performance
   - Skip unnecessary swaps
   - Early termination when possible`,
					CodeExamples: `// Go: Comprehensive partitioning techniques

// ===== TWO-WAY PARTITION =====
// Move zeros to end
func moveZeros(nums []int) {
    writePos := 0
    
    // Move all non-zero elements to front
    for i := 0; i < len(nums); i++ {
        if nums[i] != 0 {
            nums[writePos], nums[i] = nums[i], nums[writePos]
            writePos++
        }
    }
    // Zeros automatically at end (writePos to len(nums)-1)
}

// Separate even and odd numbers
func separateEvenOdd(nums []int) {
    writePos := 0
    
    // Move even numbers to front
    for i := 0; i < len(nums); i++ {
        if nums[i]%2 == 0 {
            nums[writePos], nums[i] = nums[i], nums[writePos]
            writePos++
        }
    }
    // Odd numbers automatically at end
}

// Partition around value (elements < val before, >= val after)
func partitionAroundValue(nums []int, val int) int {
    writePos := 0
    
    for i := 0; i < len(nums); i++ {
        if nums[i] < val {
            nums[writePos], nums[i] = nums[i], nums[writePos]
            writePos++
        }
    }
    return writePos  // Position where elements >= val start
}

// ===== THREE-WAY PARTITION (DUTCH NATIONAL FLAG) =====
// Sort colors: 0 (red), 1 (white), 2 (blue)
func sortColors(nums []int) {
    left, current, right := 0, 0, len(nums)-1
    
    // Invariant:
    // [0...left): 0s (red)
    // [left...current): 1s (white)
    // [current...right+1): unknown
    // (right...n-1]: 2s (blue)
    
    for current <= right {
        if nums[current] == 0 {
            // Move to red section
            nums[left], nums[current] = nums[current], nums[left]
            left++
            current++  // Increment because we know swapped element is 0 or 1
        } else if nums[current] == 2 {
            // Move to blue section
            nums[right], nums[current] = nums[current], nums[right]
            right--
            // Don't increment current - need to check swapped element
        } else {
            // nums[current] == 1, leave in white section
            current++
        }
    }
}

// Three-way partition around pivot value
func threeWayPartition(nums []int, pivot int) {
    left, current, right := 0, 0, len(nums)-1
    
    for current <= right {
        if nums[current] < pivot {
            nums[left], nums[current] = nums[current], nums[left]
            left++
            current++
        } else if nums[current] > pivot {
            nums[right], nums[current] = nums[current], nums[right]
            right--
            // Don't increment current
        } else {
            current++
        }
    }
}

// ===== LOMUTO PARTITION =====
// Partition for quicksort (simpler but less efficient)
func lomutoPartition(nums []int, low, high int) int {
    pivot := nums[high]  // Choose last element as pivot
    i := low  // Index of smaller element (elements <= pivot)
    
    // Invariant: elements [low...i] are <= pivot
    for j := low; j < high; j++ {
        if nums[j] <= pivot {
            // Move element to left section
            nums[i], nums[j] = nums[j], nums[i]
            i++
        }
    }
    
    // Place pivot in correct position
    nums[i], nums[high] = nums[high], nums[i]
    return i  // Return pivot index
}

// Quicksort using Lomuto partition
func quickSortLomuto(nums []int, low, high int) {
    if low < high {
        pivotIdx := lomutoPartition(nums, low, high)
        quickSortLomuto(nums, low, pivotIdx-1)  // Sort left part
        quickSortLomuto(nums, pivotIdx+1, high) // Sort right part
    }
}

// ===== HOARE PARTITION =====
// More efficient partition scheme
func hoarePartition(nums []int, low, high int) int {
    pivot := nums[low]  // Choose first element as pivot
    left := low + 1
    right := high
    
    for {
        // Move left until element > pivot
        for left <= right && nums[left] <= pivot {
            left++
        }
        
        // Move right until element <= pivot
        for left <= right && nums[right] > pivot {
            right--
        }
        
        if left >= right {
            break
        }
        
        // Swap elements
        nums[left], nums[right] = nums[right], nums[left]
    }
    
    // Place pivot in correct position
    nums[low], nums[right] = nums[right], nums[low]
    return right
}

// Quicksort using Hoare partition
func quickSortHoare(nums []int, low, high int) {
    if low < high {
        pivotIdx := hoarePartition(nums, low, high)
        quickSortHoare(nums, low, pivotIdx)    // Sort left part (include pivot)
        quickSortHoare(nums, pivotIdx+1, high) // Sort right part
    }
}

// ===== ADVANCED PARTITION PROBLEMS =====
// Rearrange array by sign (positive first, negative second)
func rearrangeBySign(nums []int) {
    writePos := 0
    
    // Move positive numbers to front
    for i := 0; i < len(nums); i++ {
        if nums[i] > 0 {
            nums[writePos], nums[i] = nums[i], nums[writePos]
            writePos++
        }
    }
    // Negative numbers automatically at end
}

// Rearrange maintaining relative order (requires extra space)
func rearrangeBySignOrdered(nums []int) {
    positives := []int{}
    negatives := []int{}
    
    for _, num := range nums {
        if num > 0 {
            positives = append(positives, num)
        } else {
            negatives = append(negatives, num)
        }
    }
    
    // Copy back maintaining order
    i := 0
    for _, p := range positives {
        nums[i] = p
        i++
    }
    for _, n := range negatives {
        nums[i] = n
        i++
    }
}

# Python: Comprehensive partitioning techniques

# ===== TWO-WAY PARTITION =====
# Move zeros to end
def move_zeros(nums):
    write_pos = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[write_pos], nums[i] = nums[i], nums[write_pos]
            write_pos += 1

# Separate even and odd
def separate_even_odd(nums):
    write_pos = 0
    for i in range(len(nums)):
        if nums[i] % 2 == 0:
            nums[write_pos], nums[i] = nums[i], nums[write_pos]
            write_pos += 1

# Partition around value
def partition_around_value(nums, val):
    write_pos = 0
    for i in range(len(nums)):
        if nums[i] < val:
            nums[write_pos], nums[i] = nums[i], nums[write_pos]
            write_pos += 1
    return write_pos

# ===== THREE-WAY PARTITION (DUTCH NATIONAL FLAG) =====
# Sort colors
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
            # Don't increment current
        else:
            current += 1

# Three-way partition around pivot
def three_way_partition(nums, pivot):
    left = current = 0
    right = len(nums) - 1
    
    while current <= right:
        if nums[current] < pivot:
            nums[left], nums[current] = nums[current], nums[left]
            left += 1
            current += 1
        elif nums[current] > pivot:
            nums[right], nums[current] = nums[current], nums[right]
            right -= 1
        else:
            current += 1

# ===== LOMUTO PARTITION =====
def lomuto_partition(nums, low, high):
    pivot = nums[high]
    i = low
    
    for j in range(low, high):
        if nums[j] <= pivot:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
    
    nums[i], nums[high] = nums[high], nums[i]
    return i

# Quicksort using Lomuto partition
def quicksort_lomuto(nums, low, high):
    if low < high:
        pivot_idx = lomuto_partition(nums, low, high)
        quicksort_lomuto(nums, low, pivot_idx - 1)
        quicksort_lomuto(nums, pivot_idx + 1, high)

# ===== HOARE PARTITION =====
def hoare_partition(nums, low, high):
    pivot = nums[low]
    left = low + 1
    right = high
    
    while True:
        while left <= right and nums[left] <= pivot:
            left += 1
        while left <= right and nums[right] > pivot:
            right -= 1
        
        if left >= right:
            break
        
        nums[left], nums[right] = nums[right], nums[left]
    
    nums[low], nums[right] = nums[right], nums[low]
    return right

# Quicksort using Hoare partition
def quicksort_hoare(nums, low, high):
    if low < high:
        pivot_idx = hoare_partition(nums, low, high)
        quicksort_hoare(nums, low, pivot_idx)
        quicksort_hoare(nums, pivot_idx + 1, high)

# ===== ADVANCED PARTITION PROBLEMS =====
# Rearrange by sign
def rearrange_by_sign(nums):
    write_pos = 0
    for i in range(len(nums)):
        if nums[i] > 0:
            nums[write_pos], nums[i] = nums[i], nums[write_pos]
            write_pos += 1

# Rearrange maintaining order
def rearrange_by_sign_ordered(nums):
    positives = [x for x in nums if x > 0]
    negatives = [x for x in nums if x <= 0]
    nums[:] = positives + negatives

# Key insights:
# 1. Two-way partition: Use write pointer for elements satisfying condition
# 2. Three-way partition: Use three pointers for three categories
# 3. Lomuto: Simpler, pivot at end, more swaps
# 4. Hoare: More efficient, pivot at start, fewer swaps
# 5. Maintain invariants carefully for correctness`,
				},
				{
					Title: "Merge Sorted Arrays",
					Content: `Merging sorted arrays is one of the most fundamental operations in computer science, appearing in countless algorithms and real-world applications. This operation is the cornerstone of merge sort, external sorting algorithms, database merge operations, and distributed systems that combine sorted data from multiple sources. Understanding how to efficiently merge sorted arrays is essential for any serious programmer.

**What is Merging?**

Merging combines two or more sorted sequences into a single sorted sequence while maintaining the sorted order. The key insight is that when both input sequences are already sorted, we can efficiently combine them by comparing elements and selecting the smallest remaining element at each step.

**Why Merging Matters:**

**1. Fundamental Algorithm Building Block:**
- Core operation in merge sort (divide-and-conquer)
- Used in external sorting (sorting data too large for memory)
- Basis for many advanced algorithms
- Appears frequently in coding interviews

**2. Real-World Applications:**
- **Database Systems**: Merging results from multiple sorted indexes, combining sorted query results
- **Distributed Systems**: Merging sorted data from multiple servers/nodes
- **Time-Series Data**: Combining sorted time-series data from multiple sources
- **Log Processing**: Merging sorted log files from multiple servers
- **Search Engines**: Merging sorted document lists from multiple indexes
- **Social Media**: Merging sorted feeds from multiple users or sources
- **Financial Systems**: Merging sorted transaction records
- **Version Control**: Merging sorted change lists (like in Git)

**3. Performance Characteristics:**
- Optimal time complexity: O(m + n) for two arrays
- Can be done in-place with O(1) extra space
- Very cache-friendly (sequential access pattern)
- Highly parallelizable (can merge multiple pairs simultaneously)

**Problem Variations:**

**1. Merge Two Sorted Arrays (Basic):**
- **Input**: Two sorted arrays of sizes m and n
- **Output**: One sorted array containing all elements
- **Time Complexity**: O(m + n) - must examine every element
- **Space Complexity**: O(m + n) for new array, O(1) if in-place possible
- **Key Challenge**: Maintaining sorted order while combining

**2. Merge Two Sorted Arrays In-Place:**
- **Constraint**: One array has enough space at the end
- **Challenge**: Avoid overwriting unprocessed elements
- **Solution**: Work backwards from the end
- **Time Complexity**: O(m + n)
- **Space Complexity**: O(1) - no extra space needed
- **Real-World**: Common in memory-constrained systems

**3. Merge K Sorted Arrays:**
- **Input**: K sorted arrays (potentially different sizes)
- **Output**: One sorted array containing all elements
- **Approaches**: 
  - Divide and conquer: O(n log k) time, O(n) space
  - Heap/Priority Queue: O(n log k) time, O(k) space
  - Pairwise merging: O(nk) time (inefficient)
- **Best Approach**: Heap for optimal time complexity
- **Real-World**: Merging results from multiple database shards

**4. Merge Sorted Linked Lists:**
- **Advantage**: No space issues (can modify in-place easily)
- **Time Complexity**: O(m + n)
- **Space Complexity**: O(1) - just rearrange pointers
- **Easier**: No need to worry about overwriting elements
- **Common**: Used in merge sort for linked lists

**5. Merge with Duplicates:**
- **Variation**: Handle duplicate elements
- **Options**: Keep all duplicates, keep unique only, count occurrences
- **Complexity**: Same as basic merge, just different comparison logic

**Algorithm for Merging Two Arrays:**

**Standard Approach (Using Extra Space):**

**Step-by-Step Process:**
1. **Initialize**: Create result array of size m + n, initialize two pointers (i, j) at start of both arrays
2. **Compare**: Compare elements at current positions in both arrays
3. **Select**: Add the smaller element to result array
4. **Advance**: Move pointer of array from which element was taken
5. **Repeat**: Continue until one array is exhausted
6. **Append**: Add all remaining elements from non-exhausted array

**Key Observations:**
- Each element is examined exactly once
- Total comparisons: at most m + n - 1 (when one array finishes, rest are appended)
- Maintains stability if equal elements are handled consistently
- Very cache-friendly (sequential access)

**In-Place Merging Technique:**

**The Challenge:**
When merging in-place, we risk overwriting elements we haven't processed yet. The solution is elegant: work backwards from the end.

**Why Backwards Works:**
- Largest elements go at the end first
- End of array has space (by problem constraint)
- No risk of overwriting unprocessed elements
- Can fill positions from right to left safely

**Algorithm:**
1. **Initialize**: Start pointers at ends: i = m-1, j = n-1, k = m+n-1 (write position)
2. **Compare**: Compare elements at current positions
3. **Place**: Put larger element at position k
4. **Advance**: Move appropriate pointer and write pointer backwards
5. **Repeat**: Continue until one array is exhausted
6. **Append**: Copy remaining elements from non-exhausted array

**Key Insight**: By working backwards, we ensure that positions we're writing to are either empty or contain elements we've already processed.

**Merging K Sorted Arrays:**

**Approach 1: Divide and Conquer (Optimal Time, More Space)**
- Recursively merge pairs of arrays
- Merge first half, merge second half, then merge the two halves
- Time: O(n log k) - log k levels, n elements per level
- Space: O(n) - need space for intermediate results
- **Best for**: When memory is available, want optimal time

**Approach 2: Heap/Priority Queue (Optimal Time, Less Space)**
- Use min-heap to always get smallest element
- Initialize heap with first element from each array
- Extract min, add next element from same array
- Time: O(n log k) - n extractions, heap size k
- Space: O(k) - only heap space needed
- **Best for**: Memory-constrained, optimal time needed

**Approach 3: Pairwise Merging (Simple but Inefficient)**
- Merge arrays one by one: merge arr1+arr2, then result+arr3, etc.
- Time: O(nk) - each merge is O(n), k merges
- Space: O(n) - intermediate results
- **Avoid**: Only use if k is very small

**Real-World Use Cases:**

**1. Database Query Processing:**
- Multiple indexes return sorted results
- Need to merge for intersection/union operations
- Example: "Find users who liked post A AND post B" (intersection of sorted user IDs)
- Performance critical: databases handle billions of records

**2. Distributed Systems:**
- Multiple servers return sorted results
- Coordinator merges results for client
- Example: Search across multiple sharded databases
- Challenge: Handle different result sizes, network delays

**3. External Sorting:**
- Data too large to fit in memory
- Sort chunks, then merge sorted chunks
- Example: Sorting 1TB file with 10GB RAM
- Critical: Efficient merging reduces I/O operations

**4. Time-Series Data Aggregation:**
- Multiple sensors produce sorted time-series data
- Merge to create unified timeline
- Example: Combining stock prices from multiple exchanges
- Challenge: Handle timestamps, align data points

**5. Log Processing:**
- Multiple servers produce sorted log files
- Merge for centralized analysis
- Example: Combining web server logs chronologically
- Real-world: Used in log aggregation systems (ELK stack, Splunk)

**6. Social Media Feeds:**
- Multiple users' posts are sorted by time
- Merge to create unified feed
- Example: Facebook news feed algorithm
- Challenge: Real-time merging, ranking, filtering

**Best Practices:**

**1. Handle Edge Cases:**
- Empty arrays (one or both)
- Single element arrays
- Arrays with all elements smaller/larger than other array
- Duplicate elements (decide on handling strategy)

**2. Optimize for Your Use Case:**
- Memory-constrained: Use in-place merging
- Time-critical: Use heap approach for k arrays
- Simple case: Standard two-pointer approach
- Large k: Use divide-and-conquer

**3. Consider Stability:**
- Stable merge: Maintains relative order of equal elements
- Important for: Multi-key sorting, maintaining insertion order
- Implementation: When equal, prefer element from first array

**4. Cache Optimization:**
- Merging is cache-friendly (sequential access)
- Prefer arrays over linked lists for merging
- Consider block-based merging for very large arrays

**5. Parallelization Opportunities:**
- Can merge multiple pairs in parallel
- Divide k arrays into groups, merge groups in parallel
- Useful for: Multi-core systems, distributed merging

**Common Pitfalls:**

**1. Off-by-One Errors:**
- Incorrect pointer initialization
- Forgetting to handle remaining elements
- Index out of bounds when one array finishes
- **Solution**: Carefully trace through examples, test edge cases

**2. In-Place Merging Mistakes:**
- Working forwards instead of backwards
- Overwriting unprocessed elements
- Incorrect pointer management
- **Solution**: Always work backwards, verify no overwrites

**3. K-Array Merging Inefficiency:**
- Using pairwise merging (O(nk)) instead of heap (O(n log k))
- Not considering divide-and-conquer approach
- **Solution**: Understand time complexity, choose appropriate approach

**4. Memory Issues:**
- Not pre-allocating result array (causes reallocations)
- Creating unnecessary intermediate arrays
- **Solution**: Pre-allocate, reuse arrays when possible

**5. Stability Issues:**
- Not maintaining relative order of equal elements
- Inconsistent handling of duplicates
- **Solution**: Define comparison clearly, test with duplicates

**Performance Analysis:**

**Time Complexity:**
- Two arrays: O(m + n) - optimal, must examine every element
- K arrays (heap): O(n log k) - optimal for comparison-based merging
- K arrays (divide-conquer): O(n log k) - same complexity, different approach
- **Cannot do better**: Must compare elements to determine order

**Space Complexity:**
- Standard merge: O(m + n) for result array
- In-place merge: O(1) extra space (if space available)
- K arrays (heap): O(k) for heap
- K arrays (divide-conquer): O(n) for intermediate results

**Cache Performance:**
- Excellent: Sequential access pattern
- Both input arrays accessed sequentially
- Result array written sequentially
- Very cache-friendly, good real-world performance

**Advanced Techniques:**

**1. Adaptive Merging:**
- Detect when one array is much smaller
- Use binary search to find insertion points
- Can be faster when size difference is large
- Time: O(m log n) when m << n

**2. Block-Based Merging:**
- Process arrays in blocks for cache efficiency
- Useful for very large arrays that don't fit in cache
- Reduces cache misses

**3. Parallel Merging:**
- Divide arrays into segments
- Merge segments in parallel
- Combine parallel results
- Useful for multi-core systems

**4. Streaming Merge:**
- Merge arrays as elements arrive (streaming)
- Don't need full arrays in memory
- Useful for: Real-time systems, large datasets
- Challenge: Handle different arrival rates

**Comparison with Alternatives:**

**Merging vs. Concatenation + Sorting:**
- Concatenate: O(m + n) time
- Sort: O((m+n) log(m+n)) time
- **Merging wins**: O(m + n) vs O((m+n) log(m+n))
- **When merging is better**: When inputs are already sorted

**Merging vs. Heap for K Arrays:**
- Merging (divide-conquer): O(n log k) time, O(n) space
- Heap: O(n log k) time, O(k) space
- **Heap wins on space**: O(k) vs O(n)
- **When to use**: Memory-constrained environments

**Implementation Considerations:**

**Language-Specific Optimizations:**
- **Go**: Use slices efficiently, pre-allocate capacity
- **Python**: List comprehensions can be slower, prefer explicit loops
- **Java**: ArrayList vs arrays, consider memory overhead
- **C++**: Use std::merge, highly optimized

**Testing Strategy:**
- Test with empty arrays
- Test with single elements
- Test with all elements from one array smaller
- Test with duplicates
- Test with very large arrays
- Test in-place merging edge cases

**Debugging Tips:**
- Print array states at each iteration
- Verify pointers don't go out of bounds
- Check that all elements are included
- Verify sorted order is maintained
- For in-place: Check no overwrites occur`,
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
}`,
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
					Content: `The sliding window technique is a powerful optimization method that maintains a window of elements and slides it across an array or string to find optimal solutions. It is essential for solving substring and subarray problems efficiently, often reducing O(n*n) or O(n*n*n) brute force solutions to O(n).

**What is a Sliding Window?**

A sliding window is a range of elements in an array or string that we maintain and slide across the data structure. Instead of checking all possible subarrays/substrings (which would be O(n*n) or worse), we efficiently maintain a window and update it incrementally.

**Why Sliding Window Works:**

**Efficiency Through Incremental Updates:**
- Instead of recalculating window properties from scratch, update them incrementally
- Add new element: update window state
- Remove old element: update window state
- Each element added/removed at most once: O(n) total

**Key Insight:** When we slide the window, we only need to account for the element entering and the element leaving, not recalculate everything.

**Types of Sliding Windows:**

**1. Fixed Window:**

**Characteristics:**
- Window size is constant (e.g., always k elements)
- Window slides one position at a time
- Simple to implement

**Algorithm:**
1. Calculate initial window [0, k-1]
2. For each position i from k to n-1:
   - Remove element at i-k (leaving window)
   - Add element at i (entering window)
   - Update window state
   - Check/update answer

**Time Complexity:** O(n) - each element processed once
**Space Complexity:** O(1) or O(k) depending on what we track

**Common Problems:**
- Maximum/minimum sum of subarray of size k
- Average of all subarrays of size k
- Maximum element in each window of size k (requires deque)

**2. Variable Window:**

**Characteristics:**
- Window size changes based on conditions
- Expand window when condition not satisfied
- Contract window when condition satisfied
- More complex but more powerful

**Algorithm:**
1. Initialize left = 0, right = 0
2. Expand window (right++) until condition satisfied or violated
3. Contract window (left++) while maintaining condition
4. Update answer at appropriate points
5. Repeat until right reaches end

**Time Complexity:** O(n) - each element visited at most twice
**Space Complexity:** O(k) where k is number of distinct elements in window

**Common Problems:**
- Longest substring with at most k distinct characters
- Minimum window substring
- Longest substring without repeating characters
- Count subarrays with exactly k distinct elements

**Window Maintenance:**

**Expand Window:**
- Move right pointer forward
- Add new element to window
- Update window state (frequencies, counts, etc.)
- Check if condition satisfied

**Contract Window:**
- Move left pointer forward
- Remove element from window
- Update window state
- Check if condition still satisfied

**Track State:**
- Use hash map for character/element frequencies
- Use variables for counts, sums, etc.
- Maintain invariant about window state

**When to Use Sliding Window:**

**Perfect For:**
- Subarray/substring problems
- Finding maximum/minimum in a window
- Problems asking for "longest" or "shortest" subarray
- Problems with constraints (sum, unique characters, frequency)
- String pattern matching
- Problems with "at most k" or "exactly k" constraints

**Not Ideal For:**
- Problems requiring checking all pairs
- Problems without subarray/substring structure
- When window state is expensive to maintain

**Time Complexity Analysis:**

**Why O(n)?**
- Right pointer: moves from 0 to n-1: O(n) steps
- Left pointer: moves from 0 to n-1: O(n) steps
- Each element visited at most twice (once by each pointer)
- Total: O(n) time

**Template Pattern:**

**Fixed Window:**
1. Calculate initial window [0, k-1]
2. Initialize answer with initial window
3. For i from k to n-1:
   - Remove nums[i-k]
   - Add nums[i]
   - Update answer

**Variable Window:**
1. Initialize left = 0, right = 0
2. While right < n:
   - Expand: Add nums[right] to window
   - While window invalid:
     - Contract: Remove nums[left] from window
     - left++
   - Update answer (window is now valid)
   - right++

**Common Variations:**

**1. At Most K:**
- Window can have at most k distinct elements
- Contract when distinct count > k
- Use while loop for contraction

**2. Exactly K:**
- Window must have exactly k distinct elements
- Use "at most k" trick: count(at most k) - count(at most k-1)
- Or maintain window with exactly k

**3. At Least K:**
- Window must have at least k of something
- Different contraction logic
- Expand until condition met, then contract while maintaining

**Best Practices:**

1. **Identify Window Type:**
   - Fixed: constant size
   - Variable: size changes based on condition

2. **Maintain Window State Efficiently:**
   - Use hash map for frequencies
   - Update incrementally (don't recalculate)
   - Track what matters (counts, sums, etc.)

3. **Handle Edge Cases:**
   - Empty array/string
   - Window size > array size
   - No valid window exists

4. **Update Answer Correctly:**
   - For "longest": update when window valid
   - For "shortest": update when window valid, then try to contract
   - For "count": update appropriately`,
					CodeExamples: `// Go: Comprehensive sliding window techniques

// ===== FIXED WINDOW =====
// Maximum sum of subarray of size k
func maxSumSubarray(nums []int, k int) int {
    if len(nums) < k {
        return 0
    }
    
    // Calculate initial window sum
    windowSum := 0
    for i := 0; i < k; i++ {
        windowSum += nums[i]
    }
    maxSum := windowSum
    
    // Slide window: remove left, add right
    for i := k; i < len(nums); i++ {
        windowSum = windowSum - nums[i-k] + nums[i]  // Update incrementally
        if windowSum > maxSum {
            maxSum = windowSum
        }
    }
    return maxSum
}

// Average of all subarrays of size k
func averageSubarrays(nums []int, k int) []float64 {
    if len(nums) < k {
        return []float64{}
    }
    
    result := make([]float64, len(nums)-k+1)
    windowSum := 0
    
    // Initial window
    for i := 0; i < k; i++ {
        windowSum += nums[i]
    }
    result[0] = float64(windowSum) / float64(k)
    
    // Slide window
    for i := k; i < len(nums); i++ {
        windowSum = windowSum - nums[i-k] + nums[i]
        result[i-k+1] = float64(windowSum) / float64(k)
    }
    return result
}

// ===== VARIABLE WINDOW =====
// Longest substring without repeating characters
func lengthOfLongestSubstring(s string) int {
    charMap := make(map[byte]int)  // char -> last index
    left := 0
    maxLen := 0
    
    for right := 0; right < len(s); right++ {
        char := s[right]
        
        // If character seen and within window, move left
        if idx, exists := charMap[char]; exists && idx >= left {
            left = idx + 1
        }
        
        charMap[char] = right
        maxLen = max(maxLen, right-left+1)
    }
    return maxLen
}

// Longest substring with at most k distinct characters
func longestSubstringKDistinct(s string, k int) int {
    charCount := make(map[byte]int)
    left := 0
    maxLen := 0
    
    for right := 0; right < len(s); right++ {
        // Expand window
        charCount[s[right]]++
        
        // Contract window if too many distinct characters
        for len(charCount) > k {
            charCount[s[left]]--
            if charCount[s[left]] == 0 {
                delete(charCount, s[left])
            }
            left++
        }
        
        // Update answer (window is valid)
        maxLen = max(maxLen, right-left+1)
    }
    return maxLen
}

// Minimum window substring
func minWindow(s string, t string) string {
    need := make(map[byte]int)
    for _, c := range t {
        need[byte(c)]++
    }
    
    window := make(map[byte]int)
    left, right := 0, 0
    valid := 0  // Number of characters with satisfied frequency
    start := 0
    minLen := math.MaxInt32
    
    for right < len(s) {
        c := s[right]
        right++
        
        // Update window
        if need[c] > 0 {
            window[c]++
            if window[c] == need[c] {
                valid++
            }
        }
        
        // Contract window
        for valid == len(need) {
            // Update minimum window
            if right-left < minLen {
                start = left
                minLen = right - left
            }
            
            d := s[left]
            left++
            
            // Update window
            if need[d] > 0 {
                if window[d] == need[d] {
                    valid--
                }
                window[d]--
            }
        }
    }
    
    if minLen == math.MaxInt32 {
        return ""
    }
    return s[start : start+minLen]
}

// Count subarrays with exactly k distinct elements
func countSubarraysKDistinct(nums []int, k int) int {
    // Use "at most k" trick: count(at most k) - count(at most k-1) = count(exactly k)
    return countAtMostKDistinct(nums, k) - countAtMostKDistinct(nums, k-1)
}

func countAtMostKDistinct(nums []int, k int) int {
    if k < 0 {
        return 0
    }
    
    count := make(map[int]int)
    left := 0
    result := 0
    
    for right := 0; right < len(nums); right++ {
        count[nums[right]]++
        
        // Contract window if too many distinct
        for len(count) > k {
            count[nums[left]]--
            if count[nums[left]] == 0 {
                delete(count, nums[left])
            }
            left++
        }
        
        // All subarrays ending at right with at most k distinct
        result += right - left + 1
    }
    return result
}

// Helper function
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

# Python: Comprehensive sliding window techniques

# ===== FIXED WINDOW =====
# Maximum sum of subarray of size k
def max_sum_subarray(nums, k):
    if len(nums) < k:
        return 0
    
    # Initial window sum
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide window
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i-k] + nums[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Average of all subarrays of size k
def average_subarrays(nums, k):
    if len(nums) < k:
        return []
    
    result = []
    window_sum = sum(nums[:k])
    result.append(window_sum / k)
    
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i-k] + nums[i]
        result.append(window_sum / k)
    
    return result

# ===== VARIABLE WINDOW =====
# Longest substring without repeating characters
def length_of_longest_substring(s):
    char_map = {}  # char -> last index
    left = 0
    max_len = 0
    
    for right in range(len(s)):
        char = s[right]
        
        # Move left if character seen
        if char in char_map and char_map[char] >= left:
            left = char_map[char] + 1
        
        char_map[char] = right
        max_len = max(max_len, right - left + 1)
    
    return max_len

# Longest substring with at most k distinct characters
def longest_substring_k_distinct(s, k):
    char_count = {}
    left = 0
    max_len = 0
    
    for right in range(len(s)):
        # Expand window
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        # Contract if too many distinct
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        # Update answer
        max_len = max(max_len, right - left + 1)
    
    return max_len

# Minimum window substring
def min_window(s, t):
    from collections import Counter
    need = Counter(t)
    window = {}
    left = right = 0
    valid = 0
    start = 0
    min_len = float('inf')
    
    while right < len(s):
        c = s[right]
        right += 1
        
        # Update window
        if c in need:
            window[c] = window.get(c, 0) + 1
            if window[c] == need[c]:
                valid += 1
        
        # Contract window
        while valid == len(need):
            # Update minimum
            if right - left < min_len:
                start = left
                min_len = right - left
            
            d = s[left]
            left += 1
            
            # Update window
            if d in need:
                if window[d] == need[d]:
                    valid -= 1
                window[d] -= 1
    
    return "" if min_len == float('inf') else s[start:start+min_len]

# Count subarrays with exactly k distinct elements
def count_subarrays_k_distinct(nums, k):
    # Use "at most k" trick
    return count_at_most_k_distinct(nums, k) - count_at_most_k_distinct(nums, k-1)

def count_at_most_k_distinct(nums, k):
    if k < 0:
        return 0
    
    count = {}
    left = 0
    result = 0
    
    for right in range(len(nums)):
        count[nums[right]] = count.get(nums[right], 0) + 1
        
        # Contract if too many distinct
        while len(count) > k:
            count[nums[left]] -= 1
            if count[nums[left]] == 0:
                del count[nums[left]]
            left += 1
        
        # All subarrays ending at right with at most k distinct
        result += right - left + 1
    
    return result

# Key insights:
# 1. Fixed window: Update incrementally (remove left, add right)
# 2. Variable window: Expand until condition met, contract while maintaining
# 3. Use hash map to track window state efficiently
# 4. Each element visited at most twice: O(n) time complexity
# 5. Update answer at appropriate points (when window valid)`,
				},
				{
					Title: "Variable Window Patterns",
					Content: `Variable window sliding window problems require dynamically adjusting window size based on conditions. These are more complex than fixed window problems but enable solving a wider range of problems efficiently.

**Understanding Variable Windows:**

**Key Difference from Fixed Windows:**
- Fixed: Window size constant, slide one position
- Variable: Window size changes, expand/contract based on conditions
- Variable windows can solve "longest", "shortest", and "count" problems

**Why Variable Windows Work:**
- Expand when condition not satisfied (need more elements)
- Contract when condition satisfied (try to optimize)
- Each element added/removed at most once: O(n) time

**Common Patterns:**

**1. Longest Substring/Subarray:**

**Problem Type:**
- Find longest subarray/substring satisfying condition
- Usually "at most k" constraint

**Algorithm:**
1. Expand window (right++) until condition violated
2. Contract window (left++) until condition satisfied again
3. Update maximum length when window valid
4. Repeat until right reaches end

**Key Insight:** Window is valid between contractions, so update answer during this time.

**Example:** Longest substring with at most k distinct characters
- Expand: Add characters until distinct count > k
- Contract: Remove characters until distinct count <= k
- Update: Maximum length when distinct count <= k

**2. Shortest Substring/Subarray:**

**Problem Type:**
- Find shortest subarray/substring satisfying condition
- Usually "at least k" or "contains all" constraint

**Algorithm:**
1. Expand window (right++) until condition satisfied
2. Contract window (left++) while condition still satisfied
3. Update minimum length when window valid
4. Repeat until right reaches end

**Key Insight:** Once condition satisfied, try to contract to find shorter valid window.

**Example:** Minimum window substring
- Expand: Add characters until all required characters present
- Contract: Remove characters while all still present
- Update: Minimum length when all characters present

**3. Count Subarrays:**

**Problem Type:**
- Count number of subarrays satisfying condition
- Often "exactly k" constraint

**Algorithm:**
- Use "at most k" trick: count(exactly k) = count(at most k) - count(at most k-1)
- Or maintain window with exactly k and count valid windows

**Key Insight:** For each right position, count how many valid windows end at right.

**Example:** Count subarrays with exactly k distinct elements
- Count subarrays with at most k distinct
- Count subarrays with at most k-1 distinct
- Difference gives exactly k

**Key Techniques:**

**1. Hash Map for Window State:**
- Track character/element frequencies
- Update incrementally: add when expanding, remove when contracting
- Check conditions using map size or frequencies

**2. Maintain Invariants:**
- Window is always valid when checking (for longest problems)
- Window becomes valid after contraction (for shortest problems)
- Clearly define what "valid" means

**3. Update Answer Correctly:**
- **Longest**: Update when window valid (after contraction)
- **Shortest**: Update when window valid, then try to contract further
- **Count**: Update for each valid window ending at current right

**When to Contract:**

**For "At Most K" Problems:**
- Contract when constraint violated (e.g., distinct count > k)
- Use while loop: while constraint violated: contract

**For "At Least K" Problems:**
- Contract when constraint satisfied (try to optimize)
- Use while loop: while constraint still satisfied: contract

**For "Exactly K" Problems:**
- Can use "at most k" trick
- Or maintain exactly k: contract when > k, expand when < k

**Common Mistakes:**

1. **Updating Answer at Wrong Time:**
   - For longest: update after contraction (window valid)
   - For shortest: update when valid, then contract
   - For count: update for each valid window

2. **Not Maintaining Window State:**
   - Forgot to update frequencies when expanding/contracting
   - Not checking map size correctly
   - Off-by-one errors in counts

3. **Incorrect Contraction Logic:**
   - Contracting when shouldn't
   - Not contracting when should
   - Infinite loops from incorrect conditions

**Best Practices:**

1. **Template for Longest:**
   ` + "```" + `
   Expand until invalid
   Contract until valid
   Update answer (window now valid)
   ` + "```" + `

2. **Template for Shortest:**
   ` + "```" + `
   Expand until valid
   Update answer
   Contract while still valid
   Update answer again
   ` + "```" + `

3. **Template for Count:**
   ` + "```" + `
   Expand window
   Contract if needed
   Count valid windows ending at right
   ` + "```" + `

4. **Use Helper Functions:**
   - isValid(): Check if window satisfies condition
   - expand(): Add element to window
   - contract(): Remove element from window`,
					CodeExamples: `// Go: Comprehensive variable window patterns

// ===== LONGEST PATTERN =====
// Longest substring with at most k distinct characters
func lengthOfLongestSubstringKDistinct(s string, k int) int {
    if k == 0 || len(s) == 0 {
        return 0
    }
    
    freq := make(map[byte]int)
    left, maxLen := 0, 0
    
    for right := 0; right < len(s); right++ {
        // Expand window
        freq[s[right]]++
        
        // Contract window if more than k distinct characters
        for len(freq) > k {
            freq[s[left]]--
            if freq[s[left]] == 0 {
                delete(freq, s[left])
            }
            left++
        }
        
        // Update answer (window is now valid)
        maxLen = max(maxLen, right-left+1)
    }
    
    return maxLen
}

// Longest subarray with sum at most k
func longestSubarraySumAtMostK(nums []int, k int) int {
    left := 0
    windowSum := 0
    maxLen := 0
    
    for right := 0; right < len(nums); right++ {
        // Expand window
        windowSum += nums[right]
        
        // Contract window if sum > k
        for windowSum > k {
            windowSum -= nums[left]
            left++
        }
        
        // Update answer (window sum <= k)
        maxLen = max(maxLen, right-left+1)
    }
    
    return maxLen
}

// ===== SHORTEST PATTERN =====
// Shortest subarray with sum at least k
func shortestSubarraySumAtLeastK(nums []int, k int) int {
    left := 0
    windowSum := 0
    minLen := math.MaxInt32
    
    for right := 0; right < len(nums); right++ {
        // Expand window
        windowSum += nums[right]
        
        // Contract window while sum >= k (try to minimize)
        for windowSum >= k {
            // Update answer
            minLen = min(minLen, right-left+1)
            
            // Contract
            windowSum -= nums[left]
            left++
        }
    }
    
    if minLen == math.MaxInt32 {
        return -1
    }
    return minLen
}

// ===== COUNT PATTERN =====
// Count subarrays with exactly k distinct elements
func subarraysWithKDistinct(nums []int, k int) int {
    // Use "at most k" trick: count(exactly k) = count(at most k) - count(at most k-1)
    return atMostKDistinct(nums, k) - atMostKDistinct(nums, k-1)
}

func atMostKDistinct(nums []int, k int) int {
    if k < 0 {
        return 0
    }
    
    freq := make(map[int]int)
    left, count := 0, 0
    
    for right := 0; right < len(nums); right++ {
        // Expand window
        freq[nums[right]]++
        
        // Contract window if too many distinct
        for len(freq) > k {
            freq[nums[left]]--
            if freq[nums[left]] == 0 {
                delete(freq, nums[left])
            }
            left++
        }
        
        // Count all subarrays ending at right with at most k distinct
        // [left, right], [left+1, right], ..., [right, right]
        count += right - left + 1
    }
    
    return count
}

// Count subarrays with sum exactly k (using prefix sum)
func countSubarraySumK(nums []int, k int) int {
    prefixSum := 0
    count := 0
    sumMap := make(map[int]int)
    sumMap[0] = 1  // Empty subarray has sum 0
    
    for _, num := range nums {
        prefixSum += num
        complement := prefixSum - k
        
        if val, exists := sumMap[complement]; exists {
            count += val
        }
        
        sumMap[prefixSum]++
    }
    
    return count
}

// ===== ADVANCED PATTERNS =====
// Longest subarray with at most k odd numbers
func longestSubarrayAtMostKOdd(nums []int, k int) int {
    left := 0
    oddCount := 0
    maxLen := 0
    
    for right := 0; right < len(nums); right++ {
        // Expand window
        if nums[right]%2 == 1 {
            oddCount++
        }
        
        // Contract window if too many odd numbers
        for oddCount > k {
            if nums[left]%2 == 1 {
                oddCount--
            }
            left++
        }
        
        // Update answer
        maxLen = max(maxLen, right-left+1)
    }
    
    return maxLen
}

// Helper functions
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

# Python: Comprehensive variable window patterns

# ===== LONGEST PATTERN =====
# Longest substring with at most k distinct characters
def length_of_longest_substring_k_distinct(s, k):
    if k == 0 or not s:
        return 0
    
    freq = {}
    left = max_len = 0
    
    for right in range(len(s)):
        # Expand window
        freq[s[right]] = freq.get(s[right], 0) + 1
        
        # Contract window if too many distinct
        while len(freq) > k:
            freq[s[left]] -= 1
            if freq[s[left]] == 0:
                del freq[s[left]]
            left += 1
        
        # Update answer (window is valid)
        max_len = max(max_len, right - left + 1)
    
    return max_len

# Longest subarray with sum at most k
def longest_subarray_sum_at_most_k(nums, k):
    left = window_sum = max_len = 0
    
    for right in range(len(nums)):
        # Expand window
        window_sum += nums[right]
        
        # Contract window if sum > k
        while window_sum > k:
            window_sum -= nums[left]
            left += 1
        
        # Update answer
        max_len = max(max_len, right - left + 1)
    
    return max_len

# ===== SHORTEST PATTERN =====
# Shortest subarray with sum at least k
def shortest_subarray_sum_at_least_k(nums, k):
    left = window_sum = 0
    min_len = float('inf')
    
    for right in range(len(nums)):
        # Expand window
        window_sum += nums[right]
        
        # Contract window while sum >= k
        while window_sum >= k:
            # Update answer
            min_len = min(min_len, right - left + 1)
            
            # Contract
            window_sum -= nums[left]
            left += 1
    
    return -1 if min_len == float('inf') else min_len

# ===== COUNT PATTERN =====
# Count subarrays with exactly k distinct elements
def subarrays_with_k_distinct(nums, k):
    # Use "at most k" trick
    return at_most_k_distinct(nums, k) - at_most_k_distinct(nums, k-1)

def at_most_k_distinct(nums, k):
    if k < 0:
        return 0
    
    freq = {}
    left = count = 0
    
    for right in range(len(nums)):
        # Expand window
        freq[nums[right]] = freq.get(nums[right], 0) + 1
        
        # Contract window if too many distinct
        while len(freq) > k:
            freq[nums[left]] -= 1
            if freq[nums[left]] == 0:
                del freq[nums[left]]
            left += 1
        
        # Count all subarrays ending at right
        count += right - left + 1
    
    return count

# Count subarrays with sum exactly k
def count_subarray_sum_k(nums, k):
    prefix_sum = 0
    count = 0
    sum_map = {0: 1}
    
    for num in nums:
        prefix_sum += num
        complement = prefix_sum - k
        
        if complement in sum_map:
            count += sum_map[complement]
        
        sum_map[prefix_sum] = sum_map.get(prefix_sum, 0) + 1
    
    return count

# ===== ADVANCED PATTERNS =====
# Longest subarray with at most k odd numbers
def longest_subarray_at_most_k_odd(nums, k):
    left = odd_count = max_len = 0
    
    for right in range(len(nums)):
        # Expand window
        if nums[right] % 2 == 1:
            odd_count += 1
        
        # Contract window if too many odd numbers
        while odd_count > k:
            if nums[left] % 2 == 1:
                odd_count -= 1
            left += 1
        
        # Update answer
        max_len = max(max_len, right - left + 1)
    
    return max_len

# Key insights:
# 1. Longest: Expand until invalid, contract until valid, update answer
# 2. Shortest: Expand until valid, update, contract while valid, update again
# 3. Count: Use "at most k" trick for "exactly k" problems
# 4. Maintain window state incrementally for efficiency
# 5. Update answer at correct points in algorithm`,
				},
				{
					Title: "Minimum Window Substring",
					Content: `The minimum window substring problem is a classic sliding window problem that finds the smallest substring containing all characters of a pattern. This problem demonstrates the "shortest substring" pattern in variable window sliding window techniques.

**Problem Statement:**
Given a string S and a string T, find the minimum window (substring) in S that contains all characters of T (including duplicates). If no such window exists, return an empty string.

**Example:**
- S = "ADOBECODEBANC", T = "ABC"
- Answer: "BANC" (contains A, B, C)

**Why This Problem is Important:**
- Demonstrates shortest substring pattern
- Shows how to track multiple character frequencies
- Illustrates greedy contraction strategy
- Common in real-world applications (text search, pattern matching)

**Understanding the Problem:**

**Requirements:**
1. Window must contain all characters from T
2. Characters can appear in any order
3. Window must be contiguous substring
4. Find the shortest such window

**Key Challenge:**
- Need to track multiple character frequencies simultaneously
- Must know when window is valid (all characters satisfied)
- Must contract greedily to find minimum

**Approach:**

**1. Frequency Tracking:**
- Count frequency of each character in pattern T
- Track frequency of each character in current window
- Know when a character is "satisfied" (window frequency >= required frequency)

**2. Two-Pointer Technique:**
- Left pointer: start of window
- Right pointer: end of window
- Expand window: move right pointer forward
- Contract window: move left pointer forward

**3. Window Validity:**
- Window is valid when all characters from T are satisfied
- Track count of satisfied characters (valid counter)
- Window valid when valid == number of unique characters in T

**Algorithm Steps:**

**Step 1: Initialize**
1. Count frequency of each character in T (need map)
2. Initialize empty window map
3. Set left = 0, right = 0
4. Initialize valid = 0 (number of satisfied characters)
5. Initialize minLen = infinity, start = 0

**Step 2: Expand Window**
1. Move right pointer forward
2. Add character at right to window
3. If character is in T:
   - Increment window frequency
   - If window frequency == required frequency:
     - Increment valid counter
4. Check if window is valid (valid == len(need))

**Step 3: Contract Window (When Valid)**
1. If window is valid:
   - Update minimum window if current is smaller
   - Move left pointer forward
   - Remove character at left from window
   - If character is in T:
     - If window frequency == required frequency:
       - Decrement valid counter
     - Decrement window frequency
2. Repeat contraction while window still valid

**Step 4: Repeat**
- Continue until right reaches end of S
- Return minimum window substring

**Key Insights:**

**1. Valid Counter:**
- Tracks how many characters have satisfied frequency
- Window valid when valid == number of unique characters in T
- More efficient than checking all frequencies each time

**2. Greedy Contraction:**
- Once window is valid, contract as much as possible
- This finds the minimum valid window ending at current right
- Continue contracting while window remains valid

**3. Frequency Comparison:**
- Don't need exact match, just >= required frequency
- Once frequency reaches required, character is satisfied
- Can have more than required (still satisfied)

**Time Complexity Analysis:**

**Why O(|S| + |T|)?**
- Right pointer: moves from 0 to |S|-1: O(|S|) steps
- Left pointer: moves from 0 to |S|-1: O(|S|) steps
- Each character visited at most twice (once by each pointer)
- Building frequency map for T: O(|T|)
- Total: O(|S| + |T|)

**Space Complexity:**
- need map: O(|T|) - unique characters in T
- window map: O(|S|) worst case - all characters in S
- Total: O(|S| + |T|)

**Common Mistakes:**

1. **Not Tracking Valid Counter:**
   - Checking all frequencies each time: O(|T|) per check
   - Should use valid counter for O(1) validity check

2. **Incorrect Contraction:**
   - Contracting when window not valid
   - Not contracting enough (missing minimum)
   - Contracting too much (breaking validity)

3. **Off-by-One Errors:**
   - Window length: right - left (not right - left + 1 in some implementations)
   - Index management when expanding/contracting

4. **Edge Cases:**
   - Empty string S or T
   - No valid window exists
   - T longer than S

**Variations:**

**1. Minimum Window Subsequence:**
- Characters need not be contiguous
- Use dynamic programming approach
- More complex: O(|S|²) time

**2. Longest Substring with At Most K Distinct:**
- Similar structure but "longest" instead of "shortest"
- Simpler: only track distinct count

**3. Substring with Concatenation of All Words:**
- T is array of words instead of characters
- More complex: need to match words, not characters

**Best Practices:**

1. **Use Valid Counter:**
   - Efficient validity checking
   - O(1) instead of O(|T|)

2. **Clear Variable Names:**
   - need: required frequencies
   - window: current window frequencies
   - valid: satisfied character count

3. **Handle Edge Cases:**
   - Check if |S| < |T|
   - Return empty string if no valid window

4. **Test with Examples:**
   - Simple: S="a", T="a"
   - Complex: S="ADOBECODEBANC", T="ABC"
   - Edge: S="a", T="aa" (no valid window)`,
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
					Content: `Finding the longest substring with exactly k distinct characters is a fundamental sliding window problem that demonstrates advanced variable window techniques and the powerful "at most k" transformation trick. This problem appears frequently in coding interviews and has practical applications in text processing, data analysis, and string manipulation algorithms.

**Problem Statement:**

Given a string s and an integer k, find the length of the longest substring that contains exactly k distinct characters. If no such substring exists, return 0.

**Example:**
- s = "eceba", k = 2
- Answer: 3 ("ece" has exactly 2 distinct characters: 'e' and 'c')

**Why This Problem Matters:**

**1. Interview Frequency:**
- Common in technical interviews at top tech companies
- Tests understanding of sliding window technique
- Demonstrates ability to optimize from O(n²) to O(n)

**2. Real-World Applications:**
- Text analysis: Finding longest text segments with limited vocabulary
- Data compression: Identifying repetitive patterns
- Bioinformatics: Analyzing DNA sequences with limited character sets
- Log analysis: Finding longest log segments with specific event types

**3. Algorithmic Learning:**
- Demonstrates variable window sliding window technique
- Shows "at most k" transformation trick
- Illustrates hash map usage for window state tracking
- Teaches when to expand vs contract windows

**Understanding the Problem:**

**Key Challenge:**
- Need to find longest substring (not just any substring)
- Must have exactly k distinct characters (not at most, not at least)
- Window size is variable (not fixed)
- Must efficiently track distinct character count

**Brute Force Approach:**
- Check all possible substrings: O(n²) substrings
- For each substring, count distinct characters: O(n) per substring
- Total: O(n³) time complexity
- Clearly inefficient for large strings

**Sliding Window Optimization:**
- Maintain window with variable size
- Track distinct characters efficiently using hash map
- Expand and contract window based on distinct count
- Achieve O(n) time complexity

**Two Solution Approaches:**

**Approach 1: Direct Tracking (Exactly K)**

**Algorithm:**
1. Initialize hash map to track character frequencies
2. Initialize left = 0, distinct = 0, maxLen = 0
3. For right from 0 to n-1:
   - Add s[right] to window
   - If freq[s[right]] == 0: distinct++ (new character)
   - Increment freq[s[right]]
   - While distinct > k:
     - Remove s[left] from window
     - Decrement freq[s[left]]
     - If freq[s[left]] == 0: distinct-- (character removed)
     - left++
   - If distinct == k: update maxLen = max(maxLen, right - left + 1)
4. Return maxLen

**Key Points:**
- Expand window by moving right pointer
- Contract window when distinct count exceeds k
- Update answer only when distinct count equals k exactly
- Window is always valid (distinct ≤ k) when checking

**Approach 2: "At Most K" Transformation Trick**

**Key Insight:**
Instead of finding "exactly k", we can:
1. Find longest substring with at most k distinct characters
2. Find longest substring with at most (k-1) distinct characters
3. The difference gives us substrings with exactly k

**Why This Works:**
- Substrings with exactly k are included in "at most k"
- Substrings with at most (k-1) don't have exactly k
- Difference isolates substrings with exactly k distinct characters

**Algorithm:**
- longestAtMostK(s, k) - finds longest substring with ≤ k distinct
- Answer = longestAtMostK(s, k) - longestAtMostK(s, k-1)
- Or: Answer = longestAtMostK(s, k) if it has exactly k distinct

**When to Use Each Approach:**

**Use Direct Tracking When:**
- Need exactly k (not just at most)
- Want single pass solution
- k is small (hash map size manageable)

**Use Transformation Trick When:**
- Problem asks for "exactly k" but "at most k" is easier
- Want to reuse "at most k" solution
- k might be large (but difference trick still works)

**Detailed Algorithm Walkthrough:**

**Step-by-Step Example:**
s = "eceba", k = 2

**Initialization:**
- freq = {}, left = 0, distinct = 0, maxLen = 0

**right = 0 ('e'):**
- Add 'e': freq['e'] = 1, distinct = 1
- distinct (1) ≤ k (2): valid window
- distinct (1) ≠ k (2): don't update maxLen
- Window: "e", distinct = 1

**right = 1 ('c'):**
- Add 'c': freq['c'] = 1, distinct = 2
- distinct (2) ≤ k (2): valid window
- distinct (2) == k (2): update maxLen = 2
- Window: "ec", distinct = 2, maxLen = 2

**right = 2 ('e'):**
- Add 'e': freq['e'] = 2, distinct = 2
- distinct (2) ≤ k (2): valid window
- distinct (2) == k (2): update maxLen = 3
- Window: "ece", distinct = 2, maxLen = 3

**right = 3 ('b'):**
- Add 'b': freq['b'] = 1, distinct = 3
- distinct (3) > k (2): invalid, need to contract
- Remove 'e' (left=0): freq['e'] = 1, distinct = 3 (still > k)
- Remove 'c' (left=1): freq['c'] = 0, distinct = 2
- distinct (2) ≤ k (2): valid window
- distinct (2) == k (2): update maxLen = max(3, 2) = 3
- Window: "eb", distinct = 2, maxLen = 3

**right = 4 ('a'):**
- Add 'a': freq['a'] = 1, distinct = 3
- distinct (3) > k (2): invalid, need to contract
- Remove 'e' (left=2): freq['e'] = 0, distinct = 2
- distinct (2) ≤ k (2): valid window
- distinct (2) == k (2): update maxLen = max(3, 2) = 3
- Window: "ba", distinct = 2, maxLen = 3

**Result:** maxLen = 3 ("ece")

**Time Complexity Analysis:**

**Each Character Visited:**
- Right pointer: Each character added once: O(n)
- Left pointer: Each character removed at most once: O(n)
- Total: O(n) - linear time complexity

**Why O(n) and Not O(n²):**
- Even though we have nested loops (for and while), each element is processed at most twice
- Amortized analysis: total operations = 2n = O(n)
- This is the key insight of sliding window technique

**Space Complexity Analysis:**

**Hash Map Storage:**
- Stores at most k+1 distinct characters (when distinct > k, we contract)
- In practice, stores characters currently in window
- Worst case: k+1 characters
- Space: O(k) or O(min(k, |alphabet|))

**Real-World Optimizations:**

**1. Character Set Optimization:**
- If only lowercase letters: use array[26] instead of hash map
- Faster access: O(1) vs O(1) amortized
- Less memory overhead
- Example: freq := [26]int{}

**2. Early Termination:**
- If string length < k: return 0 immediately
- If all characters same and k > 1: return 0
- Can save unnecessary processing

**3. Frequency Tracking:**
- Track distinct count separately (don't recalculate)
- Update count only when frequency crosses 0 or 1
- More efficient than counting distinct each time

**Common Variations:**

**1. At Most K Distinct:**
- Remove "exactly" constraint
- Update answer whenever distinct ≤ k
- Simpler than exactly k

**2. At Least K Distinct:**
- Find longest substring with ≥ k distinct
- Different contraction logic
- Expand until ≥ k, then try to optimize

**3. Shortest Substring with K Distinct:**
- Find shortest (not longest) substring
- Different update logic
- Contract when condition satisfied

**4. Count Substrings with K Distinct:**
- Count all substrings (not just longest)
- Use "at most k" trick
- Count = atMostK(k) - atMostK(k-1)

**Best Practices:**

**1. Handle Edge Cases:**
- k = 0: return 0 (no characters possible)
- k > unique characters in string: return string length
- Empty string: return 0
- k < 0: invalid input

**2. Initialize Correctly:**
- Hash map: make(map[byte]int) or [26]int for lowercase
- Pointers: left = 0, right = 0
- Counters: distinct = 0, maxLen = 0

**3. Update Distinct Count Correctly:**
- Increment when frequency goes 0 → 1 (new character)
- Decrement when frequency goes 1 → 0 (character removed)
- Don't update when frequency changes but stays > 0

**4. Contract Window Properly:**
- Use while loop (not if) to contract fully
- Continue contracting until condition satisfied
- Don't stop after removing one character

**5. Update Answer at Right Time:**
- For "exactly k": update when distinct == k
- For "at most k": update when distinct ≤ k
- Update after contraction (window is valid)

**Common Pitfalls:**

**1. Not Handling k = 0:**
- Edge case that causes issues
- Should return 0 immediately
- Check at start of function

**2. Updating Answer at Wrong Time:**
- Updating when distinct > k (invalid window)
- Updating before contraction completes
- Solution: Update only when window is valid

**3. Incorrect Distinct Count Tracking:**
- Forgetting to increment when new character
- Forgetting to decrement when character removed
- Not checking frequency == 0 or == 1 correctly

**4. Incomplete Window Contraction:**
- Using if instead of while
- Stopping after removing one character
- Solution: Use while loop to contract fully

**5. Off-by-One Errors:**
- Window length: right - left + 1 (not right - left)
- Array indices: 0-based vs 1-based confusion
- Check with examples

**6. Not Removing Characters Properly:**
- Forgetting to decrement frequency
- Not checking if frequency becomes 0
- Not updating distinct count when frequency reaches 0

**Performance Considerations:**

**Hash Map vs Array:**
- Hash map: O(1) amortized, works for any character set
- Array: O(1) worst case, only for limited character set (e.g., [a-z])
- Choose based on problem constraints

**Memory Usage:**
- Hash map: O(k) space for distinct characters
- Array: O(|alphabet|) space (fixed size)
- For small k and large alphabet: hash map better
- For small alphabet: array might be better

**Cache Performance:**
- Sequential access (sliding window) is cache-friendly
- Hash map lookups might cause cache misses
- Array access is more cache-friendly

**Advanced Techniques:**

**1. Two-Pass Solution:**
- First pass: find longest with at most k
- Second pass: find longest with at most k-1
- Difference gives exactly k
- More code but sometimes clearer

**2. Single-Pass with State:**
- Track current longest with exactly k
- Also track longest with at most k-1
- Update both in single pass
- More complex but efficient

**3. Deque for Optimization:**
- For some variations, deque can help
- Maintain characters in order
- Faster removal of specific characters

**Real-World Applications:**

**1. Text Analysis:**
- Find longest text segment with limited vocabulary
- Analyze readability (limited distinct words)
- Identify repetitive patterns

**2. Data Compression:**
- Identify segments with limited character diversity
- Apply compression algorithms selectively
- Optimize compression ratios

**3. Bioinformatics:**
- Analyze DNA sequences
- Find longest segments with limited nucleotide types
- Identify genetic patterns

**4. Log Processing:**
- Find longest log segments with specific event types
- Analyze system behavior patterns
- Identify anomalies

**5. Network Analysis:**
- Analyze packet streams
- Identify patterns in network traffic
- Detect unusual activity

Understanding this problem deeply helps you master the sliding window technique and prepares you for more complex variable window problems. The key is maintaining window state efficiently and knowing when to expand versus contract.`,
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
					Content: `A stack is a fundamental LIFO (Last In, First Out) data structure. Think of it like a stack of plates - you add to the top and remove from the top. The last element added is the first one removed.

**Understanding Stacks:**

**LIFO Principle:**
- Last In, First Out: most recently added element is removed first
- Like a stack of plates: add to top, remove from top
- Opposite of queue (FIFO)

**Visual Representation:**
` + "```" + `
Push 1: [1]
Push 2: [1, 2]
Push 3: [1, 2, 3]  <- Top
Pop:    [1, 2]     (returns 3)
Pop:    [1]        (returns 2)
` + "```" + `

**Core Operations:**

**1. Push (Add Element):**
- Add element to top of stack
- Time: O(1) - constant time
- Space: O(1) - single element

**2. Pop (Remove Element):**
- Remove and return top element
- Time: O(1) - constant time
- Space: O(1) - single element
- Error if stack empty

**3. Peek/Top (View Element):**
- View top element without removing
- Time: O(1) - constant time
- Space: O(1) - no allocation
- Error if stack empty

**4. IsEmpty (Check Empty):**
- Check if stack has no elements
- Time: O(1) - constant time

**5. Size (Get Count):**
- Get number of elements in stack
- Time: O(1) - if size tracked, O(n) if counted

**Implementation Options:**

**1. Array-Based Stack:**
- Use dynamic array (slice in Go, list in Python)
- Push: append to end
- Pop: remove from end
- Advantages: Simple, cache-friendly, O(1) amortized
- Disadvantages: May need resizing (amortized cost)

**2. Linked List-Based Stack:**
- Use linked list with head pointer
- Push: add new node at head
- Pop: remove head node
- Advantages: No resizing needed, O(1) worst case
- Disadvantages: Extra memory for pointers, cache misses

**3. Fixed-Size Array:**
- Pre-allocated array with size limit
- Track top index
- Advantages: No resizing, predictable memory
- Disadvantages: Size limit, wasted space if underutilized

**Common Use Cases:**

**1. Expression Evaluation:**

**Infix to Postfix Conversion:**
- Convert human-readable expressions to postfix (RPN)
- Use stack to handle operator precedence
- Example: "3 + 4 * 2" → "3 4 2 * +"

**Postfix Evaluation:**
- Evaluate postfix expressions efficiently
- Use stack to store operands
- Example: "3 4 2 * +" → 11

**Operator Precedence:**
- Stack helps manage operator hierarchy
- Higher precedence operators processed first

**2. Parentheses Matching:**

**Valid Parentheses:**
- Check if parentheses are balanced
- Push opening brackets, pop when matching closing
- Valid: "()[]{}", Invalid: "([)]"

**Nested Structure Validation:**
- Validate nested structures (HTML, XML, JSON)
- Match opening and closing tags
- Ensure proper nesting

**3. Function Call Stack:**

**Recursion Implementation:**
- Programming languages use call stack for recursion
- Each recursive call pushes frame onto stack
- Return pops frame from stack

**Backtracking Algorithms:**
- DFS uses stack (implicit or explicit)
- Track path through search space
- Backtrack by popping from stack

**4. Undo/Redo Operations:**

**Text Editor Undo:**
- Store operations on stack
- Undo pops last operation
- Can use two stacks (undo/redo)

**Browser Back Button:**
- Store visited URLs on stack
- Back button pops from history stack
- Forward uses separate stack

**5. Monotonic Stack:**

**Next Greater/Smaller Element:**
- Maintain stack in sorted order
- Find next greater element efficiently
- O(n) instead of O(n²)

**Largest Rectangle in Histogram:**
- Use monotonic stack to find boundaries
- Calculate area efficiently
- O(n) solution

**Stock Span Problem:**
- Find span of stock prices
- Use stack to track previous prices
- Efficient O(n) solution

**Monotonic Stack Pattern:**

**What is a Monotonic Stack?**
- Stack that maintains sorted order (increasing or decreasing)
- Elements popped when they violate order
- Used to find next/previous greater/smaller elements

**Increasing Monotonic Stack:**
- Elements increase from bottom to top
- Used to find next smaller element
- Pop elements >= current element

**Decreasing Monotonic Stack:**
- Elements decrease from bottom to top
- Used to find next greater element
- Pop elements <= current element

**Algorithm Pattern:**
` + "```" + `
For each element:
  While stack not empty and condition violated:
    Pop from stack and process
  Push current element
` + "```" + `

**Time Complexity:**
- All basic operations: O(1) amortized
- Monotonic stack: O(n) total (each element pushed/popped once)

**Space Complexity:**
- O(n) - stores up to n elements
- Monotonic stack: O(n) worst case

**Real-World Applications:**

**1. Compiler Design:**
- Syntax checking
- Expression parsing
- Code generation

**2. Operating Systems:**
- Function call management
- Memory stack management
- Process scheduling

**3. Web Development:**
- Browser history
- Undo/redo in editors
- Route navigation

**4. Graph Algorithms:**
- Depth-First Search (DFS)
- Topological sorting
- Strongly connected components

**5. Algorithm Design:**
- Backtracking
- Tree/graph traversal
- Expression evaluation

**Best Practices:**

1. **Choose Right Implementation:**
   - Array-based: Most common, simple
   - Linked list: When size unknown or frequent resizing expensive

2. **Handle Edge Cases:**
   - Empty stack operations
   - Stack overflow (fixed-size)
   - Null/undefined values

3. **Use Monotonic Stack:**
   - When need next/previous greater/smaller
   - Pattern recognition: "next greater", "largest rectangle"

4. **Stack vs Recursion:**
   - Recursion uses implicit stack
   - Explicit stack gives more control
   - Use explicit stack for iterative DFS`,
					CodeExamples: `// Go: Comprehensive stack implementations and examples

// ===== BASIC STACK OPERATIONS =====
// Using slice as stack
func basicStack() {
    stack := []int{}
    
    // Push operations
    stack = append(stack, 1)
    stack = append(stack, 2)
    stack = append(stack, 3)
    // Stack: [1, 2, 3]
    
    // Peek (view top without removing)
    if len(stack) > 0 {
        top := stack[len(stack)-1]
        fmt.Println("Top:", top)  // 3
    }
    
    // Pop (remove and return top)
    if len(stack) > 0 {
        top := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        fmt.Println("Popped:", top)  // 3
    }
    
    // Check if empty
    isEmpty := len(stack) == 0
    fmt.Println("Is empty:", isEmpty)  // false
    
    // Get size
    size := len(stack)
    fmt.Println("Size:", size)  // 2
}

// ===== STACK STRUCT IMPLEMENTATION =====
type Stack struct {
    items []int
}

func NewStack() *Stack {
    return &Stack{items: []int{}}
}

func (s *Stack) Push(val int) {
    s.items = append(s.items, val)
}

func (s *Stack) Pop() (int, bool) {
    if len(s.items) == 0 {
        return 0, false
    }
    val := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return val, true
}

func (s *Stack) Peek() (int, bool) {
    if len(s.items) == 0 {
        return 0, false
    }
    return s.items[len(s.items)-1], true
}

func (s *Stack) IsEmpty() bool {
    return len(s.items) == 0
}

func (s *Stack) Size() int {
    return len(s.items)
}

// ===== VALID PARENTHESES =====
func isValidParentheses(s string) bool {
    stack := []rune{}
    pairs := map[rune]rune{
        ')': '(',
        '}': '{',
        ']': '[',
    }
    
    for _, char := range s {
        if char == '(' || char == '{' || char == '[' {
            // Push opening bracket
            stack = append(stack, char)
        } else if char == ')' || char == '}' || char == ']' {
            // Check if stack empty or mismatch
            if len(stack) == 0 || stack[len(stack)-1] != pairs[char] {
                return false
            }
            // Pop matching opening bracket
            stack = stack[:len(stack)-1]
        }
    }
    
    return len(stack) == 0
}

// ===== POSTFIX EVALUATION =====
func evaluatePostfix(tokens []string) int {
    stack := []int{}
    
    for _, token := range tokens {
        if token == "+" || token == "-" || token == "*" || token == "/" {
            // Pop two operands
            b, _ := stack[len(stack)-1], stack[len(stack)-2]
            stack = stack[:len(stack)-2]
            a := stack[len(stack)-1]
            stack = stack[:len(stack)-1]
            
            // Perform operation
            var result int
            switch token {
            case "+":
                result = a + b
            case "-":
                result = a - b
            case "*":
                result = a * b
            case "/":
                result = a / b
            }
            stack = append(stack, result)
        } else {
            // Push operand
            num, _ := strconv.Atoi(token)
            stack = append(stack, num)
        }
    }
    
    return stack[0]
}

// ===== MONOTONIC STACK: NEXT GREATER ELEMENT =====
func nextGreaterElement(nums []int) []int {
    result := make([]int, len(nums))
    stack := []int{}  // Decreasing monotonic stack
    
    for i := len(nums) - 1; i >= 0; i-- {
        // Pop elements smaller than current
        for len(stack) > 0 && stack[len(stack)-1] <= nums[i] {
            stack = stack[:len(stack)-1]
        }
        
        // Next greater is top of stack (or -1)
        if len(stack) > 0 {
            result[i] = stack[len(stack)-1]
        } else {
            result[i] = -1
        }
        
        // Push current element
        stack = append(stack, nums[i])
    }
    
    return result
}

// ===== LARGEST RECTANGLE IN HISTOGRAM =====
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
        
        // Pop bars taller than current
        for len(stack) > 0 && heights[stack[len(stack)-1]] > h {
            height := heights[stack[len(stack)-1]]
            stack = stack[:len(stack)-1]
            
            width := i
            if len(stack) > 0 {
                width = i - stack[len(stack)-1] - 1
            }
            
            area := height * width
            if area > maxArea {
                maxArea = area
            }
        }
        
        stack = append(stack, i)
    }
    
    return maxArea
}

# Python: Comprehensive stack implementations and examples

# ===== BASIC STACK OPERATIONS =====
# Using list as stack
def basic_stack():
    stack = []
    
    # Push operations
    stack.append(1)
    stack.append(2)
    stack.append(3)
    # Stack: [1, 2, 3]
    
    # Peek (view top without removing)
    if stack:
        top = stack[-1]
        print("Top:", top)  # 3
    
    # Pop (remove and return top)
    if stack:
        top = stack.pop()
        print("Popped:", top)  # 3
    
    # Check if empty
    is_empty = len(stack) == 0
    print("Is empty:", is_empty)  # False
    
    # Get size
    size = len(stack)
    print("Size:", size)  # 2

# ===== STACK CLASS IMPLEMENTATION =====
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, val):
        self.items.append(val)
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# ===== VALID PARENTHESES =====
def is_valid_parentheses(s):
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in pairs.values():
            # Push opening bracket
            stack.append(char)
        elif char in pairs:
            # Check if stack empty or mismatch
            if not stack or stack[-1] != pairs[char]:
                return False
            # Pop matching opening bracket
            stack.pop()
    
    return len(stack) == 0

# ===== POSTFIX EVALUATION =====
def evaluate_postfix(tokens):
    stack = []
    
    for token in tokens:
        if token in ['+', '-', '*', '/']:
            # Pop two operands
            b = stack.pop()
            a = stack.pop()
            
            # Perform operation
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            elif token == '/':
                result = a / b
            
            stack.append(result)
        else:
            # Push operand
            stack.append(int(token))
    
    return stack[0]

# ===== MONOTONIC STACK: NEXT GREATER ELEMENT =====
def next_greater_element(nums):
    result = [-1] * len(nums)
    stack = []  # Decreasing monotonic stack
    
    for i in range(len(nums) - 1, -1, -1):
        # Pop elements smaller than current
        while stack and stack[-1] <= nums[i]:
            stack.pop()
        
        # Next greater is top of stack (or -1)
        if stack:
            result[i] = stack[-1]
        
        # Push current element
        stack.append(nums[i])
    
    return result

# ===== LARGEST RECTANGLE IN HISTOGRAM =====
def largest_rectangle_area(heights):
    stack = []  # Store indices
    max_area = 0
    
    for i in range(len(heights) + 1):
        h = heights[i] if i < len(heights) else 0
        
        # Pop bars taller than current
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            area = height * width
            max_area = max(max_area, area)
        
        stack.append(i)
    
    return max_area

# Key insights:
# 1. Stack is LIFO: last in, first out
# 2. Use slice/list: append for push, pop for pop
# 3. Monotonic stack: maintain sorted order for efficiency
# 4. Parentheses matching: push opening, pop on closing
# 5. Expression evaluation: use stack for operands/operators`,
				},
				{
					Title: "Queue",
					Content: `A queue is a fundamental FIFO (First In, First Out) data structure. Think of it like a line at a store - the first person to join is the first person to leave. The first element added is the first element removed.

**Understanding Queues:**

**FIFO Principle:**
- First In, First Out: oldest element is removed first
- Like a line: join at back, leave from front
- Opposite of stack (LIFO)

**Visual Representation:**
` + "```" + `
Enqueue 1: [1]           <- Front, Rear
Enqueue 2: [1, 2]        <- Front    Rear
Enqueue 3: [1, 2, 3]     <- Front         Rear
Dequeue:   [2, 3]        (returns 1) <- Front    Rear
Dequeue:   [3]           (returns 2) <- Front, Rear
` + "```" + `

**Core Operations:**

**1. Enqueue (Add Element):**
- Add element to rear (back) of queue
- Time: O(1) - constant time
- Space: O(1) - single element
- Also called: push, add, offer

**2. Dequeue (Remove Element):**
- Remove and return front element
- Time: O(1) - constant time (with proper implementation)
- Space: O(1) - single element
- Error if queue empty
- Also called: remove, poll

**3. Front/Peek (View Element):**
- View front element without removing
- Time: O(1) - constant time
- Space: O(1) - no allocation
- Error if queue empty
- Also called: peek, element

**4. IsEmpty (Check Empty):**
- Check if queue has no elements
- Time: O(1) - constant time

**5. Size (Get Count):**
- Get number of elements in queue
- Time: O(1) - if size tracked, O(n) if counted

**Implementation Options:**

**1. Array-Based Queue (Naive):**
- Use array, enqueue at end, dequeue from start
- Problem: Dequeue requires shifting all elements: O(n)
- Not efficient for frequent dequeues

**2. Circular Array Queue:**
- Use circular array with front and rear pointers
- Wrap around when reaching end
- Enqueue: O(1), Dequeue: O(1)
- Advantages: Cache-friendly, O(1) operations
- Disadvantages: Fixed size (or need resizing)

**3. Linked List-Based Queue:**
- Use linked list with head and tail pointers
- Enqueue: add at tail
- Dequeue: remove from head
- Advantages: Dynamic size, O(1) operations
- Disadvantages: Extra memory for pointers, cache misses

**4. Deque (Double-Ended Queue):**
- Can add/remove from both ends
- More flexible than regular queue
- Implemented with doubly linked list or circular array
- Used in sliding window problems

**Common Use Cases:**

**1. BFS (Breadth-First Search):**

**Level-Order Tree Traversal:**
- Process nodes level by level
- Enqueue children, dequeue current
- Ensures level-by-level processing

**Shortest Path in Unweighted Graphs:**
- BFS finds shortest path (minimum edges)
- Queue ensures nodes processed in order of distance
- First path found is shortest

**Finding Minimum Steps:**
- Problems asking for minimum steps/moves
- BFS naturally finds minimum
- Queue maintains order

**2. Task Scheduling:**

**Process Scheduling in OS:**
- Round-robin scheduling uses queue
- Processes added to ready queue
- CPU processes front of queue

**Print Queue Management:**
- Print jobs added to queue
- Processed in order (FIFO)
- Fair resource allocation

**Request Handling in Web Servers:**
- HTTP requests queued
- Processed in order
- Load balancing uses queues

**3. Sliding Window Maximum:**

**Using Deque:**
- Maintain deque with indices
- Remove elements outside window
- Keep deque in decreasing order
- Front always has maximum

**Efficient Solution:**
- O(n) instead of O(nk) naive
- Each element added/removed once

**4. Level-Order Processing:**

**Process Tree Nodes Level by Level:**
- Binary tree level-order traversal
- Process all nodes at level i before level i+1
- Queue maintains level order

**Generate Binary Numbers:**
- Generate binary numbers 1 to n
- Use queue: enqueue "1", then generate children
- Efficient pattern generation

**Right Side View of Tree:**
- See rightmost node at each level
- BFS with queue, track last node at each level

**Queue Variants:**

**1. Priority Queue:**

**Characteristics:**
- Elements dequeued by priority (not insertion order)
- Highest priority element dequeued first
- Implemented with heap (binary heap, Fibonacci heap)

**Operations:**
- Insert: O(log n)
- Extract Max/Min: O(log n)
- Peek: O(1)

**Use Cases:**
- Dijkstra's algorithm (shortest path)
- A* search algorithm
- Task scheduling by priority
- Merge k sorted lists

**2. Circular Queue:**

**Characteristics:**
- Fixed size array that wraps around
- Front and rear pointers
- Efficient memory usage
- No shifting needed

**Operations:**
- Enqueue: O(1)
- Dequeue: O(1)
- Full check: (rear+1) % size == front

**Use Cases:**
- Buffering (network, I/O)
- Fixed-size queues
- Memory-efficient queues

**3. Deque (Double-Ended Queue):**

**Characteristics:**
- Add/remove from both ends
- More flexible than regular queue
- Can act as both stack and queue

**Operations:**
- Push front/back: O(1)
- Pop front/back: O(1)
- Peek front/back: O(1)

**Use Cases:**
- Sliding window maximum/minimum
- Palindrome checking
- Both stack and queue operations needed

**Time Complexity:**

**Basic Operations:**
- Enqueue: O(1) - add to rear
- Dequeue: O(1) - remove from front (with proper implementation)
- Peek: O(1) - view front
- IsEmpty: O(1) - check size

**Note:** Naive array implementation has O(n) dequeue due to shifting. Use circular array or linked list for O(1).

**Space Complexity:**
- O(n) - stores up to n elements
- Additional O(1) for pointers/indexes

**Real-World Applications:**

**1. Operating Systems:**
- Process scheduling (round-robin)
- I/O request queuing
- Interrupt handling

**2. Networking:**
- Packet routing
- Network buffers
- Message queues

**3. System Design:**
- Task queues (Celery, RabbitMQ)
- Event processing
- Request queuing

**4. Algorithms:**
- BFS graph traversal
- Level-order tree traversal
- Shortest path algorithms

**5. Software Development:**
- Print queue management
- Background job processing
- Event-driven programming

**Best Practices:**

1. **Choose Right Implementation:**
   - Circular array: Fixed size, cache-friendly
   - Linked list: Dynamic size, flexible
   - Deque: Need both-end operations

2. **Handle Edge Cases:**
   - Empty queue operations
   - Queue overflow (fixed-size)
   - Null/undefined values

3. **Use Queue for BFS:**
   - Level-order processing
   - Shortest path (unweighted)
   - Minimum steps problems

4. **Consider Priority Queue:**
   - When need priority-based ordering
   - Dijkstra's, A* algorithms
   - Top K problems

5. **Use Deque for Sliding Window:**
   - When need efficient max/min in window
   - Both-end operations needed`,
					CodeExamples: `// Go: Comprehensive queue implementations and examples

// ===== BASIC QUEUE OPERATIONS =====
// Using slice as queue (simple but inefficient dequeue)
func basicQueue() {
    queue := []int{}
    
    // Enqueue operations
    queue = append(queue, 1)
    queue = append(queue, 2)
    queue = append(queue, 3)
    // Queue: [1, 2, 3] <- Front         Rear
    
    // Front (peek without removing)
    if len(queue) > 0 {
        front := queue[0]
        fmt.Println("Front:", front)  // 1
    }
    
    // Dequeue (remove from front - O(n) due to shifting)
    if len(queue) > 0 {
        front := queue[0]
        queue = queue[1:]  // Inefficient: creates new slice
        fmt.Println("Dequeued:", front)  // 1
    }
    
    // Check if empty
    isEmpty := len(queue) == 0
    fmt.Println("Is empty:", isEmpty)  // false
    
    // Get size
    size := len(queue)
    fmt.Println("Size:", size)  // 2
}

// ===== CIRCULAR QUEUE IMPLEMENTATION =====
type CircularQueue struct {
    items []int
    front int
    rear  int
    size  int
    capacity int
}

func NewCircularQueue(capacity int) *CircularQueue {
    return &CircularQueue{
        items: make([]int, capacity),
        front: 0,
        rear:  -1,
        size:  0,
        capacity: capacity,
    }
}

func (q *CircularQueue) Enqueue(val int) bool {
    if q.IsFull() {
        return false  // Queue full
    }
    q.rear = (q.rear + 1) % q.capacity
    q.items[q.rear] = val
    q.size++
    return true
}

func (q *CircularQueue) Dequeue() (int, bool) {
    if q.IsEmpty() {
        return 0, false
    }
    val := q.items[q.front]
    q.front = (q.front + 1) % q.capacity
    q.size--
    return val, true
}

func (q *CircularQueue) Front() (int, bool) {
    if q.IsEmpty() {
        return 0, false
    }
    return q.items[q.front], true
}

func (q *CircularQueue) IsEmpty() bool {
    return q.size == 0
}

func (q *CircularQueue) IsFull() bool {
    return q.size == q.capacity
}

func (q *CircularQueue) Size() int {
    return q.size
}

// ===== LINKED LIST QUEUE IMPLEMENTATION =====
type QueueNode struct {
    val  int
    next *QueueNode
}

type LinkedListQueue struct {
    front *QueueNode
    rear  *QueueNode
    size  int
}

func NewLinkedListQueue() *LinkedListQueue {
    return &LinkedListQueue{front: nil, rear: nil, size: 0}
}

func (q *LinkedListQueue) Enqueue(val int) {
    newNode := &QueueNode{val: val, next: nil}
    if q.rear == nil {
        q.front = newNode
        q.rear = newNode
    } else {
        q.rear.next = newNode
        q.rear = newNode
    }
    q.size++
}

func (q *LinkedListQueue) Dequeue() (int, bool) {
    if q.front == nil {
        return 0, false
    }
    val := q.front.val
    q.front = q.front.next
    if q.front == nil {
        q.rear = nil
    }
    q.size--
    return val, true
}

func (q *LinkedListQueue) Front() (int, bool) {
    if q.front == nil {
        return 0, false
    }
    return q.front.val, true
}

func (q *LinkedListQueue) IsEmpty() bool {
    return q.front == nil
}

func (q *LinkedListQueue) Size() int {
    return q.size
}

// ===== BFS WITH QUEUE =====
type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

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

// ===== SHORTEST PATH BFS =====
func shortestPath(graph map[int][]int, start int, target int) int {
    queue := []int{start}
    visited := make(map[int]bool)
    visited[start] = true
    distance := 0
    
    for len(queue) > 0 {
        levelSize := len(queue)
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            
            if node == target {
                return distance
            }
            
            for _, neighbor := range graph[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true
                    queue = append(queue, neighbor)
                }
            }
        }
        
        distance++
    }
    
    return -1  // Not found
}

# Python: Comprehensive queue implementations and examples

# ===== BASIC QUEUE OPERATIONS =====
# Using collections.deque (recommended)
from collections import deque

def basic_queue():
    queue = deque()
    
    # Enqueue operations
    queue.append(1)
    queue.append(2)
    queue.append(3)
    # Queue: [1, 2, 3] <- Front         Rear
    
    # Front (peek without removing)
    if queue:
        front = queue[0]
        print("Front:", front)  # 1
    
    # Dequeue (remove from front - O(1))
    if queue:
        front = queue.popleft()
        print("Dequeued:", front)  # 1
    
    # Check if empty
    is_empty = len(queue) == 0
    print("Is empty:", is_empty)  # False
    
    # Get size
    size = len(queue)
    print("Size:", size)  # 2

# ===== CIRCULAR QUEUE IMPLEMENTATION =====
class CircularQueue:
    def __init__(self, capacity):
        self.items = [None] * capacity
        self.front = 0
        self.rear = -1
        self.size = 0
        self.capacity = capacity
    
    def enqueue(self, val):
        if self.is_full():
            return False  # Queue full
        self.rear = (self.rear + 1) % self.capacity
        self.items[self.rear] = val
        self.size += 1
        return True
    
    def dequeue(self):
        if self.is_empty():
            return None
        val = self.items[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return val
    
    def front_element(self):
        if self.is_empty():
            return None
        return self.items[self.front]
    
    def is_empty(self):
        return self.size == 0
    
    def is_full(self):
        return self.size == self.capacity
    
    def get_size(self):
        return self.size

# ===== LINKED LIST QUEUE IMPLEMENTATION =====
class QueueNode:
    def __init__(self, val):
        self.val = val
        self.next = None

class LinkedListQueue:
    def __init__(self):
        self.front = None
        self.rear = None
        self.size = 0
    
    def enqueue(self, val):
        newNode = QueueNode(val)
        if self.rear is None:
            self.front = newNode
            self.rear = newNode
        else:
            self.rear.next = newNode
            self.rear = newNode
        self.size += 1
    
    def dequeue(self):
        if self.front is None:
            return None
        val = self.front.val
        self.front = self.front.next
        if self.front is None:
            self.rear = None
        self.size -= 1
        return val
    
    def front_element(self):
        if self.front is None:
            return None
        return self.front.val
    
    def is_empty(self):
        return self.front is None
    
    def get_size(self):
        return self.size

# ===== BFS WITH QUEUE =====
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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
    
    return result

# ===== SHORTEST PATH BFS =====
def shortest_path(graph, start, target):
    from collections import deque
    queue = deque([start])
    visited = {start}
    distance = 0
    
    while queue:
        level_size = len(queue)
        
        for _ in range(level_size):
            node = queue.popleft()
            
            if node == target:
                return distance
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        distance += 1
    
    return -1  # Not found

# ===== PRIORITY QUEUE (using heapq) =====
import heapq

def priority_queue_example():
    # Min heap (smallest element first)
    pq = []
    heapq.heappush(pq, 3)
    heapq.heappush(pq, 1)
    heapq.heappush(pq, 2)
    
    # Pop smallest
    smallest = heapq.heappop(pq)  # 1
    print("Smallest:", smallest)
    
    # Max heap (negate values)
    max_pq = []
    heapq.heappush(max_pq, -3)
    heapq.heappush(max_pq, -1)
    heapq.heappush(max_pq, -2)
    
    largest = -heapq.heappop(max_pq)  # 3
    print("Largest:", largest)

# Key insights:
# 1. Queue is FIFO: first in, first out
# 2. Use deque for O(1) enqueue/dequeue
# 3. Circular queue: efficient fixed-size queue
# 4. BFS uses queue for level-order processing
# 5. Priority queue: elements by priority, not insertion order`,
				},
				{
					Title: "Monotonic Stack Advanced",
					Content: `A monotonic stack is a specialized stack data structure that maintains its elements in sorted order (either monotonically increasing or decreasing). This seemingly simple property enables elegant O(n) solutions to problems that would otherwise require O(n²) brute force approaches. Understanding monotonic stacks is crucial for solving a wide range of array and string problems efficiently.

**What is a Monotonic Stack?**

A monotonic stack is a stack where elements are maintained in sorted order from bottom to top. When a new element is added, any elements that would violate this ordering are removed first. This property allows us to efficiently answer queries about "next greater/smaller element" or "previous greater/smaller element" for each position in an array.

**Key Properties:**

**1. Maintained Order:**
- Elements are always in sorted order (increasing or decreasing)
- Order is preserved as elements are added
- Violating elements are removed before adding new ones

**2. Efficient Queries:**
- Can answer "next greater/smaller" queries in O(1) per element
- Total time: O(n) for all queries
- Much better than O(n²) brute force

**3. Element Processing:**
- Each element is pushed once
- Each element is popped at most once
- Total operations: O(n)

**Types of Monotonic Stacks:**

**1. Monotonically Increasing Stack:**

**Structure:**
- Elements increase from bottom to top
- Bottom element is smallest, top is largest
- New elements must be ≥ top element

**When to Use:**
- Finding next smaller element (to the right)
- Finding previous smaller element (to the left)
- Problems requiring "next element that's smaller"

**Maintenance:**
- Before pushing: Remove all elements ≥ current element
- Popped elements have current as their "next smaller"
- Push current element

**Example Use Cases:**
- Next Smaller Element: For each element, find first smaller element to right
- Previous Smaller Element: For each element, find first smaller element to left
- Largest Rectangle: Find boundaries where height decreases

**2. Monotonically Decreasing Stack:**

**Structure:**
- Elements decrease from bottom to top
- Bottom element is largest, top is smallest
- New elements must be ≤ top element

**When to Use:**
- Finding next greater element (to the right)
- Finding previous greater element (to the left)
- Problems requiring "next element that's greater"

**Maintenance:**
- Before pushing: Remove all elements ≤ current element
- Popped elements have current as their "next greater"
- Push current element

**Example Use Cases:**
- Next Greater Element: For each element, find first greater element to right
- Previous Greater Element: For each element, find first greater element to left
- Stock Span: Calculate how many consecutive days price was ≤ current

**The Core Algorithm Pattern:**

**Generic Monotonic Stack Algorithm:**

1. Initialize empty stack
2. Iterate through array with index i:
   a. While stack not empty AND current element violates monotonic property:
      - Pop element from stack
      - Process popped element (found its "next" element)
   b. Push current element (and optionally its index) to stack
3. After iteration, remaining elements have no "next" element (set to -1 or default)

**Key Insight:** When we pop an element, the current element is the answer for that popped element. This is why monotonic stacks work so efficiently.

**Time Complexity Analysis:**

**Why O(n) and Not O(n²)?**

**Each Element Lifecycle:**
- Pushed once: O(1)
- Popped at most once: O(1)
- Total operations per element: O(1)
- Total for n elements: O(n)

**Amortized Analysis:**
- Even though we have nested loops (for + while), total operations are bounded
- Each element can only be popped once
- Total pops ≤ n (one per element)
- Total pushes = n (one per element)
- Total: O(n)

**Common Problems Solved:**

**1. Next Greater Element:**

**Problem:** For each element, find the first element to its right that is greater than it.

**Algorithm:**
- Use decreasing stack (top is smallest)
- When current > stack top: current is "next greater" for stack top
- Pop and record answer, continue until stack empty or current ≤ top
- Push current to stack

**Example:**
- Array: [4, 5, 2, 10]
- Next greater: [5, 10, 10, -1]

**2. Previous Greater Element:**

**Problem:** For each element, find the first element to its left that is greater than it.

**Algorithm:**
- Similar to next greater, but process left to right
- Use decreasing stack
- When current > stack top: pop (current is greater than popped)
- Previous greater for current is new stack top (if exists)

**3. Largest Rectangle in Histogram:**

**Problem:** Find the largest rectangle area that can be formed from histogram bars.

**Key Insight:**
- For each bar, find how far it can extend left and right
- Left boundary: first bar to left that's shorter
- Right boundary: first bar to right that's shorter
- Use increasing stack to find boundaries efficiently

**Algorithm:**
- Use increasing stack (maintain bars in increasing height)
- When current bar < stack top: current is right boundary for stack top
- Pop and calculate area: height × (current_index - new_stack_top_index - 1)
- Left boundary is new stack top, right boundary is current index

**Time Complexity:** O(n) - each bar pushed/popped once

**4. Trapping Rain Water:**

**Problem:** Calculate how much rainwater can be trapped between bars.

**Key Insight:**
- Water trapped at position i = min(max_left, max_right) - height[i]
- Use stack to track decreasing heights
- When height increases, calculate trapped water

**Algorithm:**
- Use decreasing stack (track bars in decreasing order)
- When current > stack top: water can be trapped
- Pop and calculate: water += (min(current, new_top) - popped_height) × width
- Width = current_index - new_top_index - 1

**5. Stock Span Problem:**

**Problem:** For each day, calculate span (consecutive days where price ≤ current price).

**Key Insight:**
- Span = current_index - index_of_previous_greater_price
- Use decreasing stack to track previous greater prices

**Algorithm:**
- Use decreasing stack storing indices
- When current price > stack top price: pop (current is greater)
- Span = current_index - new_stack_top_index
- Push current index

**Real-World Applications:**

**1. Financial Analysis:**
- Stock price analysis (span problem)
- Finding support/resistance levels
- Identifying trend reversals

**2. Image Processing:**
- Finding largest rectangle in binary images
- Histogram equalization
- Image segmentation

**3. Data Structures:**
- Building efficient range query structures
- Optimizing segment trees
- Implementing priority queues

**4. Algorithm Optimization:**
- Converting O(n²) solutions to O(n)
- Reducing space complexity
- Improving cache performance

**Best Practices:**

**1. Choose Correct Stack Type:**
- Next/Previous Greater: Use decreasing stack
- Next/Previous Smaller: Use increasing stack
- Match problem requirement to stack type

**2. Store Indices vs Values:**
- Store indices when you need position information
- Store values when you only need the value
- Indices needed for: width calculations, distance calculations

**3. Handle Remaining Elements:**
- After iteration, stack may not be empty
- These elements have no "next" element
- Set their answers to -1 or appropriate default

**4. Boundary Handling:**
- Consider what happens at array boundaries
- First element: no previous element
- Last element: no next element
- Handle edge cases explicitly

**5. Process While Popping:**
- Do calculations when popping (not when pushing)
- Popped element has found its answer
- Current element is the answer for popped element

**Common Pitfalls:**

**1. Wrong Stack Type:**
- Using increasing stack for "greater" problems
- Using decreasing stack for "smaller" problems
- Solution: Match stack type to problem requirement

**2. Not Processing Popped Elements:**
- Forgetting to calculate answer when popping
- Only processing at end
- Solution: Process immediately when popping

**3. Index vs Value Confusion:**
- Storing values when indices needed
- Storing indices when values needed
- Solution: Understand what information you need

**4. Boundary Errors:**
- Off-by-one errors in index calculations
- Not handling empty stack case
- Solution: Test with small examples, check boundaries

**5. Incomplete Popping:**
- Not popping all violating elements
- Stopping too early
- Solution: Use while loop, not if statement

**Advanced Techniques:**

**1. Monotonic Deque:**
- Use deque instead of stack for some problems
- Allows operations on both ends
- Useful for sliding window maximum/minimum

**2. Two-Pass Approach:**
- Sometimes need both next and previous
- Do two passes: one for next, one for previous
- Combine results

**3. Circular Array:**
- For circular arrays, do two passes
- First pass: normal
- Second pass: handle wrap-around cases

**4. Multi-Dimensional:**
- Extend to 2D problems
- Process row by row or column by column
- Apply 1D solution to each dimension

**Performance Optimizations:**

**1. Pre-allocate Stack:**
- If maximum size known, pre-allocate
- Reduces memory allocations
- Improves performance

**2. Use Array Instead of Slice:**
- For fixed-size problems, use array
- Avoids slice overhead
- Better cache performance

**3. Early Termination:**
- If answer found early, can terminate
- Not always possible, but worth considering
- Reduces unnecessary processing

**Comparison with Alternatives:**

**Monotonic Stack vs Brute Force:**
- Brute Force: O(n²) - check all pairs
- Monotonic Stack: O(n) - process each element once
- Massive improvement for large inputs

**Monotonic Stack vs Segment Tree:**
- Segment Tree: O(n log n) build, O(log n) query
- Monotonic Stack: O(n) total, simpler code
- Stack is better for "next/previous" queries

**Monotonic Stack vs Two Pointers:**
- Two Pointers: Works for some problems
- Monotonic Stack: More general, handles more cases
- Choose based on problem structure

Understanding monotonic stacks deeply enables you to solve many array problems efficiently. The key is recognizing when a problem can benefit from maintaining sorted order and understanding which type of monotonic stack to use.`,
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
					Content: `Breadth-First Search (BFS) is one of the most fundamental graph traversal algorithms, using a queue to explore graphs level by level. It's essential for finding shortest paths in unweighted graphs, performing level-order tree traversals, and solving a wide variety of graph problems. Understanding BFS deeply is crucial for algorithm design and problem-solving.

**What is BFS?**

BFS is a graph traversal algorithm that explores a graph by visiting all nodes at the current depth level before moving to nodes at the next depth level. It uses a First-In-First-Out (FIFO) queue to ensure that nodes are processed in the order they were discovered, which naturally creates a level-by-level exploration pattern.

**Historical Context:**

BFS was first described in the 1950s as part of graph theory research. It's one of the oldest and most well-studied graph algorithms, forming the foundation for many advanced algorithms including Dijkstra's algorithm (which can be seen as a weighted version of BFS) and A* search.

**Core Algorithm:**

**Step-by-Step Process:**

1. **Initialization:**
   - Choose a starting node (source)
   - Mark source as visited
   - Create empty queue
   - Add source to queue
   - Initialize distance/level tracking (optional)

2. **Main Loop:**
   - While queue is not empty:
     a. Dequeue a node (this is the current node to process)
     b. Process the current node (print, check condition, etc.)
     c. For each unvisited neighbor of current node:
        - Mark neighbor as visited
        - Add neighbor to queue
        - Update distance/level if tracking
     d. Repeat until queue is empty

3. **Termination:**
   - Algorithm ends when queue is empty
   - All reachable nodes have been visited

**Key Properties:**

**1. Level Order Traversal:**
- Nodes at distance 0 (source) are visited first
- Nodes at distance 1 are visited next
- Nodes at distance 2 follow, and so on
- This level-by-level order is guaranteed by the queue

**2. Shortest Path Property:**
- In unweighted graphs, BFS finds shortest path (minimum number of edges)
- First time a node is reached, it's via shortest path
- This is why we mark as visited when adding to queue (not when processing)

**3. Completeness:**
- BFS visits all nodes reachable from source
- If graph is connected, all nodes are visited
- If disconnected, only nodes in source's component are visited

**4. Queue-Based:**
- Uses FIFO queue to maintain level order
- First node discovered at level i is first processed
- Ensures level-by-level exploration

**Why Queue is Essential:**

**FIFO Property:**
- Queue ensures first-in-first-out order
- Nodes discovered earlier are processed earlier
- This maintains level order automatically
- Without queue, we'd lose the level-by-level guarantee

**Alternative (Inefficient):**
- Using stack would give DFS (depth-first)
- Using priority queue would give different order
- Queue is the natural choice for level-order

**Time Complexity Analysis:**

**Visiting Each Node:**
- Each node is added to queue exactly once: O(V)
- Each node is removed from queue exactly once: O(V)
- Total queue operations: O(V)

**Exploring Edges:**
- Each edge is examined exactly once (when processing its source node)
- Total edge examinations: O(E)

**Total Time Complexity:** O(V + E)
- V = number of vertices
- E = number of edges
- Linear in the size of the graph

**Space Complexity Analysis:**

**Queue Storage:**
- In worst case, queue contains all nodes at one level
- Maximum level size: O(V) in worst case (star graph)
- Queue space: O(V)

**Visited Array:**
- Boolean array of size V: O(V)
- Or hash set: O(V) space

**Total Space Complexity:** O(V)
- Dominated by queue and visited tracking

**BFS Applications:**

**1. Shortest Path in Unweighted Graphs:**

**Problem:** Find shortest path from source to target (minimum edges).

**Why BFS Works:**
- First time target is reached, it's via shortest path
- All paths of length d are explored before paths of length d+1
- Guaranteed to find shortest path

**Algorithm:**
- Run BFS from source
- Track distance/level for each node
- When target is found, return its distance
- Can reconstruct path using parent pointers

**Real-World Examples:**
- Social networks: Find shortest connection path between users
- Game AI: Find minimum moves to reach goal
- Web crawling: Find shortest link path between pages
- Network routing: Find minimum hops between nodes

**2. Level-Order Tree Traversal:**

**Problem:** Process binary tree nodes level by level.

**Why BFS is Perfect:**
- Trees are acyclic graphs
- BFS naturally processes by level
- Each level is processed completely before next

**Applications:**
- Print tree structure level by level
- Find nodes at specific depth
- Calculate tree width (maximum level size)
- Serialize/deserialize trees level by level

**Real-World:**
- File system traversal (directory levels)
- Organizational hierarchy processing
- Decision tree evaluation

**3. Connected Components:**

**Problem:** Find all nodes reachable from a source, or count connected components.

**Algorithm:**
- Run BFS from source to find its component
- For all components: run BFS from unvisited nodes
- Count number of BFS runs needed

**Applications:**
- Social networks: Find friend groups
- Image processing: Find connected regions
- Network analysis: Find subnetworks
- Puzzle solving: Find connected pieces

**4. Bipartite Graph Checking:**

**Problem:** Check if graph can be colored with 2 colors such that no adjacent nodes have same color.

**BFS Approach:**
- Color source with color 1
- Color neighbors with color 2
- Color their neighbors with color 1
- If conflict found (adjacent nodes same color): not bipartite

**Why BFS Works:**
- Level-by-level coloring ensures consistency
- Odd-length cycles cause conflicts
- BFS detects these conflicts naturally

**Applications:**
- Scheduling problems (assign tasks to two groups)
- Resource allocation (two types of resources)
- Graph theory problems

**5. Word Ladder Problem:**

**Problem:** Transform word A to word B, changing one letter at a time, using only valid words.

**BFS Approach:**
- Each word is a node
- Edge exists if words differ by one letter
- BFS finds shortest transformation sequence

**Why BFS:**
- Finds minimum number of transformations
- Explores all one-step transformations before two-step
- Guaranteed shortest path

**Real-World:**
- Spell checkers: Find closest valid words
- DNA sequencing: Find mutation paths
- Puzzle games: Find solution paths

**6. Minimum Steps Problems:**

**Problem:** Find minimum steps to reach target state.

**Examples:**
- Knight moves on chessboard
- Sliding puzzle solutions
- State machine transitions
- Game level completion

**BFS Approach:**
- Each state is a node
- Transitions are edges
- BFS finds shortest path to target state

**7. Network Broadcasting:**

**Problem:** Broadcast message to all nodes in network.

**BFS Approach:**
- Start from source
- Each node receives message and forwards to neighbors
- Ensures all nodes receive message
- Minimum number of hops

**Implementation Details:**

**1. Visited Marking Strategy:**

**Mark When Adding to Queue (Recommended):**
- Mark node as visited immediately when adding to queue
- Prevents duplicate additions
- Ensures shortest path property
- More efficient

**Mark When Processing (Alternative):**
- Mark when dequeuing
- Allows duplicate additions (less efficient)
- Still works but slower

**2. Level Tracking:**

**Method 1: Process Level by Level**
- Track level size at start of each iteration
- Process exactly that many nodes
- Increment level after processing level

**Method 2: Store Level with Node**
- Store (node, level) pairs in queue
- Level increases by 1 for neighbors
- More memory but simpler logic

**3. Path Reconstruction:**

**Using Parent Pointers:**
- Store parent for each node
- When target found, backtrack using parents
- Reconstruct path in reverse

**Using Distance Array:**
- Store distance from source
- Can verify shortest path
- Useful for multiple queries

**Best Practices:**

**1. Initialize Properly:**
- Mark source as visited before adding to queue
- Initialize queue with source
- Set up distance/level tracking if needed

**2. Handle Edge Cases:**
- Empty graph: return immediately
- Source equals target: return 0 or source
- Disconnected graph: handle appropriately

**3. Efficient Queue Operations:**
- Use efficient queue implementation
- In Go: use slice with proper indexing
- In Python: use collections.deque
- Avoid inefficient operations

**4. Memory Management:**
- Clear visited array if running multiple BFS
- Reuse data structures when possible
- Consider space-time tradeoffs

**Common Pitfalls:**

**1. Not Marking Visited Early:**
- Marking when processing instead of when adding
- Allows duplicate queue additions
- Slower and uses more memory

**2. Forgetting to Check Visited:**
- Adding already-visited nodes to queue
- Causes infinite loops or incorrect results
- Always check before adding

**3. Incorrect Level Tracking:**
- Not tracking level size correctly
- Mixing levels in processing
- Solution: Process level by level explicitly

**4. Queue Implementation Issues:**
- Using inefficient queue (e.g., list.pop(0) in Python)
- Should use deque or proper queue
- Affects performance significantly

**5. Not Handling Disconnected Graphs:**
- Assuming graph is connected
- Missing nodes in disconnected components
- Solution: Run BFS from each unvisited node

**Advanced Variations:**

**1. Multi-Source BFS:**
- Start BFS from multiple sources simultaneously
- Useful for finding closest source
- Applications: Nearest facility, infection spread

**2. Bidirectional BFS:**
- Run BFS from source and target simultaneously
- Meet in the middle
- Faster for finding paths (reduces search space)

**3. 0-1 BFS:**
- Graph with edge weights 0 or 1
- Use deque instead of queue
- Weight-0 edges to front, weight-1 to back
- More efficient than Dijkstra for 0-1 weights

**4. BFS with Constraints:**
- Add constraints to node processing
- Skip nodes that don't meet criteria
- Applications: Maze with obstacles, restricted movements

**Performance Optimizations:**

**1. Early Termination:**
- Stop when target is found
- Don't continue exploring
- Saves time for shortest path problems

**2. Bidirectional Search:**
- Search from both ends
- Meet in the middle
- Reduces search space significantly

**3. Efficient Data Structures:**
- Use deque for queue (faster than list)
- Use array for visited (faster than set for dense graphs)
- Pre-allocate when size known

**4. Parallel BFS:**
- Process multiple levels in parallel
- Useful for large graphs
- Requires synchronization

**Comparison with DFS:**

**BFS Advantages:**
- Finds shortest path in unweighted graphs
- Level-order traversal
- Guaranteed to find solution if exists (complete)

**DFS Advantages:**
- Less memory (O(depth) vs O(width))
- Simpler implementation (recursion)
- Better for deep graphs

**When to Use BFS:**
- Need shortest path
- Need level-order processing
- Graph is wide (not too deep)
- Need to find closest node

**Real-World Performance:**

**Social Networks:**
- Finding friend connections (6 degrees of separation)
- BFS finds shortest connection path
- Used by LinkedIn, Facebook for connection suggestions

**Web Crawling:**
- Crawling web pages level by level
- BFS ensures systematic exploration
- Used by search engines

**Game AI:**
- Pathfinding in games
- BFS finds shortest path
- Used in strategy games, puzzles

**Network Routing:**
- Finding shortest path in network
- BFS finds minimum hops
- Used in routing protocols

Understanding BFS deeply enables you to solve many graph problems efficiently. The key insight is that the queue maintains level order, which gives BFS its powerful shortest-path property in unweighted graphs.`,
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
					Content: `A deque (double-ended queue) is a versatile data structure that allows efficient insertion and deletion from both ends. This bidirectional access makes deques perfect for solving problems that require manipulating elements at both the front and back of a sequence. Understanding deque applications is essential for optimizing many algorithms, particularly sliding window problems and specialized graph traversals.

**What is a Deque?**

A deque combines the capabilities of both stacks and queues:
- Like a stack: Can push/pop from one end
- Like a queue: Can enqueue/dequeue from one end
- Unique: Can do both operations on both ends simultaneously

**Deque Operations:**

**Front Operations:**
- **Push Front**: Add element to beginning - O(1)
- **Pop Front**: Remove element from beginning - O(1)
- **Front**: Access element at beginning - O(1)

**Back Operations:**
- **Push Back**: Add element to end - O(1)
- **Pop Back**: Remove element from end - O(1)
- **Back**: Access element at end - O(1)

**Why Deque is Powerful:**

**Bidirectional Access:**
- Can add/remove from either end
- Enables algorithms that need both ends
- More flexible than stack or queue alone

**Efficient Operations:**
- All operations are O(1) amortized
- Better than using two stacks or arrays
- Optimized implementations in most languages

**Key Applications:**

**1. Sliding Window Maximum:**

**Problem:** Find maximum element in every window of size k in an array.

**Brute Force:** O(nk) - check each window separately
**Deque Solution:** O(n) - process each element once

**Algorithm:**
- Maintain deque storing indices (not values)
- Deque stores indices in decreasing order of values
- Front of deque always has index of maximum in current window

**Steps:**
1. Remove indices outside current window from front
2. Remove indices of elements smaller than current from back
3. Add current index to back
4. Front of deque is maximum of current window

**Why It Works:**
- Elements smaller than current can never be maximum (current is larger and stays longer)
- Remove them to keep deque clean
- Front element is always maximum (largest value, still in window)

**Time Complexity:** O(n) - each element added/removed at most once
**Space Complexity:** O(k) - deque stores at most k indices

**Real-World Applications:**
- Stock price analysis: Maximum price in time windows
- Network monitoring: Peak traffic in time intervals
- Signal processing: Maximum values in sliding windows
- Image processing: Maximum filter operations

**2. Palindrome Checking:**

**Problem:** Check if string is palindrome (reads same forwards and backwards).

**Deque Approach:**
- Add all characters to deque
- Compare front and back characters
- Remove matching pairs
- If deque empty or one element: palindrome

**Why Deque is Perfect:**
- Natural bidirectional access
- Compare ends directly
- Remove from both ends efficiently

**Algorithm:**
1. Add all characters to deque (filter non-alphanumeric)
2. While deque has more than 1 element:
   - Pop front and back
   - Compare characters
   - If different: not palindrome
3. If loop completes: palindrome

**Time Complexity:** O(n) - process each character once
**Space Complexity:** O(n) - store characters in deque

**Variations:**
- Check if linked list is palindrome
- Find longest palindromic substring
- Count palindromic substrings

**3. 0-1 BFS (BFS with Binary Weights):**

**Problem:** Find shortest path in graph where edges have weight 0 or 1.

**Why Regular BFS Doesn't Work:**
- BFS assumes unweighted (all edges weight 1)
- With 0-1 weights, need to prioritize weight-0 edges
- Regular queue doesn't maintain priority

**Deque Solution:**
- Use deque instead of queue
- Weight-0 edges: add to front (high priority)
- Weight-1 edges: add to back (low priority)
- Process front first (weight-0 edges processed before weight-1)

**Why This Works:**
- Weight-0 edges don't increase distance
- Should be processed immediately (like same level)
- Weight-1 edges increase distance by 1
- Should be processed after current level

**Algorithm:**
1. Initialize deque with source
2. While deque not empty:
   - Pop from front (get node with minimum distance)
   - For each neighbor:
     - If edge weight 0: add to front
     - If edge weight 1: add to back
3. Continue until deque empty

**Time Complexity:** O(V + E) - same as BFS
**Space Complexity:** O(V) - deque and visited tracking

**Applications:**
- Maze solving with teleporters (teleport = weight 0)
- Network routing with free/paid links
- Game pathfinding with special moves
- State transitions with free/expensive operations

**4. Maximum in All Subarrays:**

**Problem:** Find maximum element in every subarray of size k.

**Similar to Sliding Window Maximum:**
- Same deque technique
- Maintain decreasing order
- Front is always maximum

**Variations:**
- Minimum in all subarrays (use increasing order)
- Kth largest in all subarrays (use different approach)
- Range queries in sliding windows

**5. Monotonic Deque:**

**Problem:** Maintain deque in sorted order for efficient queries.

**Technique:**
- Similar to monotonic stack
- But can access both ends
- Useful for range queries

**Applications:**
- Finding min/max in ranges
- Range minimum queries
- Optimizing dynamic programming

**6. Rotating Array Problems:**

**Problem:** Rotate array or process circular arrays.

**Deque Approach:**
- Add all elements to deque
- Rotate by popping from back and pushing to front
- Efficient for multiple rotations

**Applications:**
- Circular buffer implementation
- Rotating displays
- Round-robin scheduling

**Advantages over Regular Queue:**

**1. Bidirectional Access:**
- Can manipulate both ends
- More flexible algorithms
- Enables optimizations

**2. Better for Sliding Windows:**
- Remove from front (old elements)
- Add to back (new elements)
- Natural sliding window structure

**3. Enables 0-1 BFS:**
- Priority without heap overhead
- O(1) operations vs O(log n) for priority queue
- More efficient for 0-1 weights

**4. Palindrome Operations:**
- Natural for palindrome checking
- Compare ends directly
- More intuitive than alternatives

**Implementation Details:**

**1. Data Structure Choices:**

**Doubly Linked List:**
- Natural deque implementation
- O(1) operations on both ends
- More memory overhead (pointers)

**Circular Array:**
- Array with two pointers (front, back)
- More memory efficient
- Need to handle wrap-around
- Fixed size or dynamic resizing

**Language Implementations:**
- Python: collections.deque (optimized C implementation)
- C++: std::deque (efficient implementation)
- Java: ArrayDeque (array-based, efficient)
- Go: Need to implement or use slice carefully

**2. Index vs Value Storage:**

**For Sliding Window Maximum:**
- Store indices (not values)
- Need indices to check if outside window
- Can access values via array[index]

**For Other Problems:**
- May store values directly
- Simpler if don't need position info
- Choose based on problem needs

**3. Maintaining Order:**

**Decreasing Order (for maximum):**
- Front has largest value
- Remove smaller values from back
- Add new values to back

**Increasing Order (for minimum):**
- Front has smallest value
- Remove larger values from back
- Add new values to back

**Best Practices:**

**1. Choose Right End:**
- Understand which end to use for what
- Front for high priority, back for low priority
- Match problem requirements

**2. Maintain Invariants:**
- Keep deque in correct order
- Remove violating elements
- Check conditions carefully

**3. Handle Edge Cases:**
- Empty deque
- Single element
- All elements same value
- Window size larger than array

**4. Efficient Operations:**
- Use language's optimized deque
- Avoid inefficient implementations
- Consider memory vs speed tradeoffs

**Common Pitfalls:**

**1. Wrong Order Maintenance:**
- Not maintaining decreasing/increasing order
- Deque becomes unsorted
- Front no longer has correct element

**2. Index Confusion:**
- Storing values when indices needed
- Storing indices when values needed
- Off-by-one errors in window calculations

**3. Not Removing Old Elements:**
- Forgetting to remove elements outside window
- Deque contains stale data
- Incorrect results

**4. Inefficient Implementation:**
- Using list with pop(0) (O(n) operation)
- Should use proper deque (O(1) operations)
- Significant performance difference

**Performance Considerations:**

**Deque vs Queue:**
- Deque: O(1) operations on both ends
- Queue: O(1) on one end, O(n) on other (if using list)
- Deque is more flexible

**Deque vs Priority Queue:**
- Deque: O(1) operations, but limited ordering
- Priority Queue: O(log n) operations, full ordering
- Use deque when order is simple (0-1, decreasing, etc.)

**Memory Usage:**
- Deque: O(n) space typically
- Can be optimized for specific use cases
- Consider if space is constrained

**Real-World Examples:**

**1. Stock Trading:**
- Find maximum price in time windows
- Use deque for efficient sliding window maximum
- Real-time analysis of stock prices

**2. Network Monitoring:**
- Monitor peak bandwidth in time intervals
- Deque maintains maximum efficiently
- Alert when thresholds exceeded

**3. Game Development:**
- Pathfinding with special moves (0-1 BFS)
- Efficient shortest path calculation
- Used in many game engines

**4. Text Processing:**
- Palindrome checking in text editors
- Efficient character comparison
- Used in spell checkers, validators

Understanding deque applications enables you to solve many problems more efficiently. The key is recognizing when bidirectional access and efficient end operations can optimize your algorithm.`,
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
	})
}
