package algorithms

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAlgorithmsModules([]problems.CourseModule{
		{
			ID:          1011,
			Title:       "Common Interview Patterns",
			Description: "Master the most common algorithm patterns used in technical interviews: Two Pointers, Sliding Window, Fast & Slow Pointers, Merge Intervals, and Top K Elements.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "Two Pointers Pattern",
					Content: `The Two Pointers pattern uses two pointers to iterate through a data structure, typically from opposite ends or at different speeds. It converts O(n²) brute-force solutions into O(n) solutions.

**When to Use Two Pointers:**
- Sorted array/linked list problems
- Finding pairs with a target sum
- Removing duplicates
- Partitioning arrays
- Palindrome problems
- Container with most water / trapping rain water

**Types of Two Pointers:**

**1. Opposite Direction (converging):**
Start from both ends, move toward each other.

` + "```" + `
left = 0, right = n-1
[1, 2, 3, 4, 5, 6, 7]
 ↑                    ↑
left                right
` + "```" + `

Use for: Two sum (sorted), container with most water, valid palindrome

**2. Same Direction (fast/slow):**
Both start from the beginning, move at different speeds.

` + "```" + `
slow = 0, fast = 0
[1, 1, 2, 2, 3, 3, 4]
 ↑  ↑
slow fast
` + "```" + `

Use for: Remove duplicates, linked list cycle detection, finding middle

**3. Two Array Pointers:**
One pointer per array, process in parallel.

Use for: Merge sorted arrays, intersection of sorted arrays

**Key Insight:**
If brute force requires two nested loops (O(n²)) and the data is sorted or has some ordering, two pointers might reduce it to O(n).

**Common Mistakes:**
- Forgetting to handle duplicates (especially in 3Sum)
- Off-by-one errors at array boundaries
- Not considering empty or single-element inputs
- Moving the wrong pointer`,
					CodeExamples: `// Two Sum II (sorted array)
func twoSum(numbers []int, target int) []int {
    left, right := 0, len(numbers)-1
    for left < right {
        sum := numbers[left] + numbers[right]
        if sum == target {
            return []int{left + 1, right + 1}
        } else if sum < target {
            left++
        } else {
            right--
        }
    }
    return nil
}

// Three Sum (find all triplets summing to 0)
func threeSum(nums []int) [][]int {
    sort.Ints(nums)
    var result [][]int
    
    for i := 0; i < len(nums)-2; i++ {
        if i > 0 && nums[i] == nums[i-1] { continue } // Skip duplicates
        
        left, right := i+1, len(nums)-1
        for left < right {
            sum := nums[i] + nums[left] + nums[right]
            if sum == 0 {
                result = append(result, []int{nums[i], nums[left], nums[right]})
                for left < right && nums[left] == nums[left+1] { left++ }
                for left < right && nums[right] == nums[right-1] { right-- }
                left++
                right--
            } else if sum < 0 {
                left++
            } else {
                right--
            }
        }
    }
    return result
}

// Remove Duplicates from Sorted Array (in-place)
func removeDuplicates(nums []int) int {
    if len(nums) == 0 { return 0 }
    slow := 0
    for fast := 1; fast < len(nums); fast++ {
        if nums[fast] != nums[slow] {
            slow++
            nums[slow] = nums[fast]
        }
    }
    return slow + 1
}

# Python: Is Palindrome (two pointers converging)
def is_palindrome(s: str) -> bool:
    s = ''.join(c.lower() for c in s if c.isalnum())
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True`,
				},
				{
					Title: "Sliding Window Pattern",
					Content: `The Sliding Window pattern maintains a "window" of elements as you traverse a data structure. It's used for problems involving contiguous subarrays or substrings.

**When to Use:**
- Maximum/minimum sum subarray of size k
- Longest substring with at most k distinct characters
- Smallest subarray with sum ≥ target
- String anagrams / permutation in string
- Maximum of all subarrays of size k

**Two Types:**

**1. Fixed-Size Window:**
Window size is constant (k). Slide it one step at a time.

` + "```" + `
Array: [1, 3, 2, 6, -1, 4, 1, 8, 2]
Window size k=3:
Step 1: [1, 3, 2] = 6
Step 2: [3, 2, 6] = 11
Step 3: [2, 6, -1] = 7
...
` + "```" + `

**2. Variable-Size Window:**
Window grows or shrinks based on a condition. The key insight: expand right pointer, shrink left pointer when condition violated.

` + "```" + `
Template:
left = 0
for right in range(len(arr)):
    # Add arr[right] to window
    while window_condition_violated:
        # Remove arr[left] from window
        left += 1
    # Update result
` + "```" + `

**Pattern Recognition:**
- "Contiguous subarray" or "substring" → Likely sliding window
- "Maximum/minimum of size k" → Fixed window
- "Longest/shortest with constraint" → Variable window

**Time Complexity:**
Despite the while loop inside the for loop, each element is added and removed at most once, so the total complexity is O(n), not O(n²).

**Common Variations:**
- Window with HashMap (character frequency)
- Window with Deque (maximum in window)
- Window with Counter (at most k distinct)`,
					CodeExamples: `// Fixed-size window: Maximum sum subarray of size k
func maxSumSubarray(nums []int, k int) int {
    windowSum := 0
    for i := 0; i < k; i++ {
        windowSum += nums[i]
    }
    maxSum := windowSum
    
    for i := k; i < len(nums); i++ {
        windowSum += nums[i] - nums[i-k] // Slide: add right, remove left
        if windowSum > maxSum {
            maxSum = windowSum
        }
    }
    return maxSum
}

// Variable-size window: Minimum size subarray with sum >= target
func minSubArrayLen(target int, nums []int) int {
    left, sum, minLen := 0, 0, len(nums)+1
    
    for right := 0; right < len(nums); right++ {
        sum += nums[right]
        
        for sum >= target {
            if right-left+1 < minLen {
                minLen = right - left + 1
            }
            sum -= nums[left]
            left++
        }
    }
    
    if minLen == len(nums)+1 { return 0 }
    return minLen
}

# Python: Longest substring without repeating characters
def length_of_longest_substring(s: str) -> int:
    char_index = {}  # Character -> last seen index
    left = 0
    max_len = 0
    
    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        char_index[char] = right
        max_len = max(max_len, right - left + 1)
    
    return max_len

# Example: "abcabcbb" -> 3 ("abc")
# Example: "bbbbb" -> 1 ("b")

# Python: Longest substring with at most k distinct characters
def longest_k_distinct(s: str, k: int) -> int:
    from collections import defaultdict
    char_count = defaultdict(int)
    left = 0
    max_len = 0
    
    for right in range(len(s)):
        char_count[s[right]] += 1
        
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_len = max(max_len, right - left + 1)
    
    return max_len`,
				},
				{
					Title: "Binary Search Patterns",
					Content: `Binary search is more than just finding a target in a sorted array. It's a general technique for reducing a search space by half at each step.

**The General Template:**
Binary search works whenever you can define a condition that partitions the search space into two halves: one where the condition is true and one where it's false.

**Three Binary Search Templates:**

**Template 1: Standard (find exact match)**
` + "```" + `
lo, hi = 0, n-1
while lo <= hi:
    mid = lo + (hi-lo)//2
    if arr[mid] == target: return mid
    elif arr[mid] < target: lo = mid+1
    else: hi = mid-1
` + "```" + `

**Template 2: Left Boundary (first occurrence)**
` + "```" + `
lo, hi = 0, n-1
while lo < hi:
    mid = lo + (hi-lo)//2
    if arr[mid] < target: lo = mid+1
    else: hi = mid
return lo
` + "```" + `

**Template 3: Right Boundary (last occurrence)**
` + "```" + `
lo, hi = 0, n-1
while lo < hi:
    mid = lo + (hi-lo+1)//2  # Round UP
    if arr[mid] > target: hi = mid-1
    else: lo = mid
return lo
` + "```" + `

**Binary Search on Answer:**
When the answer is in a sorted range and you can check if a value is valid:
- Minimum capacity to ship packages in D days
- Koko eating bananas
- Split array largest sum
- Magnetic force between balls

**Pattern:**
` + "```" + `
lo, hi = min_possible, max_possible
while lo < hi:
    mid = (lo + hi) // 2
    if can_achieve(mid):
        hi = mid  # Try smaller
    else:
        lo = mid + 1  # Need bigger
return lo
` + "```" + `

**Common Mistakes:**
1. Off-by-one errors (lo <= hi vs lo < hi)
2. Integer overflow in mid calculation (use lo + (hi-lo)/2)
3. Infinite loops (not updating lo/hi correctly)
4. Forgetting to round up for right boundary

**Key Insight:**
If you can frame a problem as "find the minimum X such that condition(X) is true" and condition is monotonic, binary search works.`,
					CodeExamples: `// Binary Search on Answer: Minimum capacity to ship in D days
func shipWithinDays(weights []int, days int) int {
    // Binary search on capacity
    lo, hi := 0, 0
    for _, w := range weights {
        if w > lo { lo = w }  // Min capacity = heaviest package
        hi += w               // Max capacity = all packages in one day
    }
    
    for lo < hi {
        mid := lo + (hi-lo)/2
        if canShip(weights, days, mid) {
            hi = mid  // Try smaller capacity
        } else {
            lo = mid + 1
        }
    }
    return lo
}

func canShip(weights []int, days, capacity int) bool {
    daysNeeded, currentLoad := 1, 0
    for _, w := range weights {
        if currentLoad+w > capacity {
            daysNeeded++
            currentLoad = 0
        }
        currentLoad += w
    }
    return daysNeeded <= days
}

// Find first and last position of element
func searchRange(nums []int, target int) []int {
    return []int{findFirst(nums, target), findLast(nums, target)}
}

func findFirst(nums []int, target int) int {
    lo, hi := 0, len(nums)-1
    result := -1
    for lo <= hi {
        mid := lo + (hi-lo)/2
        if nums[mid] == target {
            result = mid
            hi = mid - 1  // Keep searching left
        } else if nums[mid] < target {
            lo = mid + 1
        } else {
            hi = mid - 1
        }
    }
    return result
}

func findLast(nums []int, target int) int {
    lo, hi := 0, len(nums)-1
    result := -1
    for lo <= hi {
        mid := lo + (hi-lo)/2
        if nums[mid] == target {
            result = mid
            lo = mid + 1  // Keep searching right
        } else if nums[mid] < target {
            lo = mid + 1
        } else {
            hi = mid - 1
        }
    }
    return result
}

# Python: Search in rotated sorted array
def search_rotated(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1
    
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        
        # Left half is sorted
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    
    return -1`,
				},
				{
					Title: "Monotonic Stack Pattern",
					Content: `A Monotonic Stack maintains elements in either increasing or decreasing order. It's the key pattern for "next greater/smaller element" problems.

**When to Use:**
- Next greater element (to the right)
- Next smaller element (to the right/left)
- Largest rectangle in histogram
- Stock span problem
- Daily temperatures
- Trapping rain water (one approach)

**How It Works:**
Process elements one by one. When you encounter an element that violates the stack's monotonic property, pop elements until the property is restored.

**Monotonic Decreasing Stack (for next greater element):**

` + "```" + `
Array: [2, 1, 2, 4, 3]
Process left to right, maintain decreasing stack:

Step 1: Push 2.           Stack: [2]
Step 2: Push 1 (1 < 2).   Stack: [2, 1]
Step 3: See 2:
  - Pop 1 → next greater of 1 is 2
  - 2 == 2, don't pop.     
  - Push 2.                Stack: [2, 2]
Step 4: See 4:
  - Pop 2 → next greater is 4
  - Pop 2 → next greater is 4
  - Push 4.                Stack: [4]
Step 5: Push 3 (3 < 4).   Stack: [4, 3]

Remaining: 4 and 3 have no next greater → -1
Result: [4, 2, 4, -1, -1]
` + "```" + `

**Monotonic Increasing Stack (for next smaller element):**
Same idea but maintain increasing order.

**Time Complexity:**
O(n) — each element is pushed and popped at most once.

**Space Complexity:**
O(n) — stack can hold up to n elements.

**Key Pattern:**
The stack stores indices (not values) so you can calculate distances. When you pop an element, the current element is the "answer" for the popped element.`,
					CodeExamples: `// Next Greater Element
func nextGreaterElement(nums []int) []int {
    n := len(nums)
    result := make([]int, n)
    for i := range result { result[i] = -1 }
    
    stack := []int{} // Store indices
    
    for i := 0; i < n; i++ {
        for len(stack) > 0 && nums[i] > nums[stack[len(stack)-1]] {
            top := stack[len(stack)-1]
            stack = stack[:len(stack)-1]
            result[top] = nums[i]
        }
        stack = append(stack, i)
    }
    return result
}

// Daily Temperatures: How many days until warmer temperature
func dailyTemperatures(temperatures []int) []int {
    n := len(temperatures)
    result := make([]int, n)
    stack := []int{}
    
    for i := 0; i < n; i++ {
        for len(stack) > 0 && temperatures[i] > temperatures[stack[len(stack)-1]] {
            top := stack[len(stack)-1]
            stack = stack[:len(stack)-1]
            result[top] = i - top // Days until warmer
        }
        stack = append(stack, i)
    }
    return result
}

// Largest Rectangle in Histogram
func largestRectangleArea(heights []int) int {
    stack := []int{}
    maxArea := 0
    
    for i := 0; i <= len(heights); i++ {
        h := 0
        if i < len(heights) { h = heights[i] }
        
        for len(stack) > 0 && h < heights[stack[len(stack)-1]] {
            top := stack[len(stack)-1]
            stack = stack[:len(stack)-1]
            
            width := i
            if len(stack) > 0 {
                width = i - stack[len(stack)-1] - 1
            }
            
            area := heights[top] * width
            if area > maxArea { maxArea = area }
        }
        stack = append(stack, i)
    }
    return maxArea
}

# Python: Stock Span (days since price was higher)
def stock_span(prices):
    n = len(prices)
    spans = [0] * n
    stack = []  # Indices of decreasing prices
    
    for i in range(n):
        while stack and prices[i] >= prices[stack[-1]]:
            stack.pop()
        
        spans[i] = i + 1 if not stack else i - stack[-1]
        stack.append(i)
    
    return spans`,
				},
			},
		},
	})
}
