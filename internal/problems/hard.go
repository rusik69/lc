package problems

// HardProblems contains all hard difficulty problems (161-220)
var HardProblems = []Problem{
	{
		ID:          161,
		Title:       "Median of Two Sorted Arrays",
		Description: "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.",
		Difficulty:  "Hard",
		Topic:       "Binary Search",
		Signature:   "func findMedianSortedArrays(nums1 []int, nums2 []int) float64",
		TestCases: []TestCase{
			{Input: "nums1 = []int{1,3}, nums2 = []int{2}", Expected: "2.00000"},
			{Input: "nums1 = []int{1,2}, nums2 = []int{3,4}", Expected: "2.50000"},
			{Input: "nums1 = []int{0,0}, nums2 = []int{0,0}", Expected: "0.00000"},
			{Input: "nums1 = []int{}, nums2 = []int{1}", Expected: "1.00000"},
			{Input: "nums1 = []int{2}, nums2 = []int{}", Expected: "2.00000"},
			{Input: "nums1 = []int{1,3}, nums2 = []int{2,7}", Expected: "2.50000"},
		},
		Solution: `func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	if len(nums1) > len(nums2) {
		nums1, nums2 = nums2, nums1
	}
	m, n := len(nums1), len(nums2)
	left, right := 0, m
	for left <= right {
		partitionX := (left + right) / 2
		partitionY := (m+n+1)/2 - partitionX
		maxLeftX := math.MinInt32
		if partitionX != 0 {
			maxLeftX = nums1[partitionX-1]
		}
		minRightX := math.MaxInt32
		if partitionX != m {
			minRightX = nums1[partitionX]
		}
		maxLeftY := math.MinInt32
		if partitionY != 0 {
			maxLeftY = nums2[partitionY-1]
		}
		minRightY := math.MaxInt32
		if partitionY != n {
			minRightY = nums2[partitionY]
		}
		if maxLeftX <= minRightY && maxLeftY <= minRightX {
			if (m+n)%2 == 0 {
				return float64(max(maxLeftX, maxLeftY)+min(minRightX, minRightY)) / 2.0
			}
			return float64(max(maxLeftX, maxLeftY))
		} else if maxLeftX > minRightY {
			right = partitionX - 1
		} else {
			left = partitionX + 1
		}
	}
	return 0.0
}`,
		PythonSolution: `def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    while left <= right:
        partition_x = (left + right) // 2
        partition_y = (m + n + 1) // 2 - partition_x
        max_left_x = float('-inf') if partition_x == 0 else nums1[partition_x - 1]
        min_right_x = float('inf') if partition_x == m else nums1[partition_x]
        max_left_y = float('-inf') if partition_y == 0 else nums2[partition_y - 1]
        min_right_y = float('inf') if partition_y == n else nums2[partition_y]
        if max_left_x <= min_right_y and max_left_y <= min_right_x:
            if (m + n) % 2 == 0:
                return (max(max_left_x, max_left_y) + min(min_right_x, min_right_y)) / 2.0
            return float(max(max_left_x, max_left_y))
        elif max_left_x > min_right_y:
            right = partition_x - 1
        else:
            left = partition_x + 1
    return 0.0`,
		Explanation: `Finding the median of two sorted arrays efficiently requires a binary search approach that partitions both arrays such that elements on the left side of the partition are less than or equal to elements on the right side. This is one of the most challenging binary search problems.

The algorithm performs binary search on the smaller array to find the correct partition point. For each partition in the smaller array, it calculates the corresponding partition in the larger array such that the total number of elements on the left equals (or is one more than) the total on the right. It then checks if this partition is valid: all elements on the left must be less than or equal to all elements on the right.

When a valid partition is found, the median is calculated based on whether the total number of elements is even or odd. For even counts, it's the average of the maximum of the left side and minimum of the right side. For odd counts, it's the maximum of the left side.

Edge cases handled include: one array empty (median is median of non-empty array), arrays of equal size (partition balanced), arrays where all elements in one are smaller than the other (partition at boundary), and arrays with overlapping ranges (standard case).

An alternative O(m+n) approach would merge the arrays, but that doesn't achieve the optimal O(log(min(m,n))) time. The binary search approach is optimal. The space complexity is O(1) as we only use a few variables.

Think of this as finding a cut point in both arrays such that everything to the left of both cuts is less than everything to the right, and the cuts divide the combined array into two equal (or nearly equal) halves.`,
	},
	{
		ID:          162,
		Title:       "Longest Valid Parentheses",
		Topic:       "Stack",
		Description: "Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.",
		Difficulty:  "Hard",
		Signature:   "func longestValidParentheses(s string) int",
		TestCases: []TestCase{
			{Input: "s = \"(()\"", Expected: "2"},
			{Input: "s = \")()())\"", Expected: "4"},
			{Input: "s = \"\"", Expected: "0"},
			{Input: "s = \"()\"", Expected: "2"},
			{Input: "s = \"()(()\"", Expected: "2"},
			{Input: "s = \"((()))\"", Expected: "6"},
		},
		Solution: `func longestValidParentheses(s string) int {
	stack := []int{-1}
	maxLen := 0
	for i, c := range s {
		if c == '(' {
			stack = append(stack, i)
		} else {
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				stack = append(stack, i)
			} else {
				if i-stack[len(stack)-1] > maxLen {
					maxLen = i - stack[len(stack)-1]
				}
			}
		}
	}
	return maxLen
}`,
		PythonSolution: `def longestValidParentheses(s: str) -> int:
    stack = [-1]
    max_len = 0
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_len = max(max_len, i - stack[-1])
    return max_len`,
		Explanation: `Finding the longest valid parentheses substring requires tracking unmatched opening parentheses and calculating the length of valid substrings. A stack-based approach efficiently handles this by tracking the positions of parentheses.

The algorithm uses a stack initialized with -1 (a sentinel value representing the start of a potential valid substring). For each character, if it's '(', we push its index. If it's ')', we pop from the stack. After popping, if the stack is empty, we push the current index (this becomes the new start for future valid substrings). If the stack is not empty, we calculate the length of the valid substring ending at the current position and update the maximum.

The key insight is that when we pop and the stack is not empty, the top of the stack represents the start of the current valid substring. When the stack becomes empty after a pop, we've encountered more closing than opening parentheses, so we reset the start position.

Edge cases handled include: string starting with ')' (stack becomes empty, reset start), all opening parentheses (no valid substring), all closing parentheses (no valid substring), and nested valid parentheses (handled correctly by stack).

An alternative O(n) approach uses two passes (left-to-right and right-to-left), but the stack approach is more intuitive. The time complexity is O(n) where n is string length, and space complexity is O(n) for the stack.

Think of this as tracking unmatched opening parentheses: when you see a closing parenthesis, you match it with the most recent unmatched opening one, and the distance between them (or from the last reset point) gives you the length of a valid substring.`,
	},
	{
		ID:          163,
		Title:       "Wildcard Matching",
		Topic:       "Dynamic Programming",
		Description: "Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*' where '?' matches any single character and '*' matches any sequence of characters.",
		Difficulty:  "Hard",
		Signature:   "func isMatch(s string, p string) bool",
		TestCases: []TestCase{
			{Input: "s = \"aa\", p = \"a\"", Expected: "false"},
			{Input: "s = \"aa\", p = \"*\"", Expected: "true"},
			{Input: "s = \"cb\", p = \"?a\"", Expected: "false"},
			{Input: "s = \"adceb\", p = \"*a*b\"", Expected: "true"},
			{Input: "s = \"acdcb\", p = \"a*c?b\"", Expected: "false"},
			{Input: "s = \"\", p = \"*\"", Expected: "true"},
		},
		Solution: `func isMatch(s string, p string) bool {
	m, n := len(s), len(p)
	dp := make([][]bool, m+1)
	for i := range dp {
		dp[i] = make([]bool, n+1)
	}
	dp[0][0] = true
	for j := 1; j <= n; j++ {
		if p[j-1] == '*' {
			dp[0][j] = dp[0][j-1]
		}
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if p[j-1] == '*' {
				dp[i][j] = dp[i-1][j] || dp[i][j-1]
			} else if p[j-1] == '?' || s[i-1] == p[j-1] {
				dp[i][j] = dp[i-1][j-1]
			}
		}
	}
	return dp[m][n]
}`,
		PythonSolution: `def isMatch(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
            elif p[j - 1] == '?' or s[i - 1] == p[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]`,
		Explanation: `Wildcard matching requires determining if a string matches a pattern containing '?' (matches any single character) and '*' (matches any sequence of characters). Dynamic programming efficiently handles the complex matching rules.

The algorithm uses a 2D DP table where dp[i][j] represents whether the first i characters of the string match the first j characters of the pattern. The base cases handle empty string and empty pattern. For each cell, if the pattern character is '*', it can match zero characters (dp[i][j-1]) or one or more characters (dp[i-1][j]). If it's '?' or matches the string character, we check if the previous characters matched (dp[i-1][j-1]).

The '*' case is the most complex: it can match nothing (skip it) or match one or more characters (consume a character from the string and keep the '*' in the pattern for potential further matches). This greedy approach ensures we check all possibilities.

Edge cases handled include: pattern with multiple '*' (handled by DP transitions), string shorter than pattern without '*' (cannot match), pattern with only '*' (matches everything), and empty string with non-empty pattern (only matches if pattern is all '*').

An alternative recursive approach with memoization would work but is less efficient. The DP approach achieves O(m*n) time where m and n are string and pattern lengths, and O(m*n) space for the DP table (can be optimized to O(n)).

Think of this as building up matches character by character: for each position, you check if the string and pattern match up to that point, handling '*' by checking if it can match zero or more characters.`,
	},
	{
		ID:          164,
		Title:       "Regular Expression Matching",
		Topic:       "Dynamic Programming",
		Description: "Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where '.' matches any single character and '*' matches zero or more of the preceding element.",
		Difficulty:  "Hard",
		Signature:   "func isMatchRegex(s string, p string) bool",
		TestCases: []TestCase{
			{Input: "s = \"aa\", p = \"a\"", Expected: "false"},
			{Input: "s = \"aa\", p = \"a*\"", Expected: "true"},
			{Input: "s = \"ab\", p = \".*\"", Expected: "true"},
			{Input: "s = \"aab\", p = \"c*a*b\"", Expected: "true"},
			{Input: "s = \"mississippi\", p = \"mis*is*p*.\"", Expected: "false"},
			{Input: "s = \"\", p = \"a*\"", Expected: "true"},
		},
		Solution: `func isMatchRegex(s string, p string) bool {
	m, n := len(s), len(p)
	dp := make([][]bool, m+1)
	for i := range dp {
		dp[i] = make([]bool, n+1)
	}
	dp[0][0] = true
	for j := 2; j <= n; j++ {
		if p[j-1] == '*' {
			dp[0][j] = dp[0][j-2]
		}
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if p[j-1] == '*' {
				dp[i][j] = dp[i][j-2] ||
					(dp[i-1][j] && (s[i-1] == p[j-2] || p[j-2] == '.'))
			} else if p[j-1] == '.' || s[i-1] == p[j-1] {
				dp[i][j] = dp[i-1][j-1]
			}
		}
	}
	return dp[m][n]
}`,
		PythonSolution: `def isMatchRegex(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] and (s[i - 1] == p[j - 2] or p[j - 2] == '.'))
            elif p[j - 1] == '.' or s[i - 1] == p[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]`,
		Explanation: `Regular expression matching is more complex than wildcard matching because '*' applies to the preceding character, not independently. This requires careful handling of the relationship between '*' and its preceding character.

The algorithm uses dynamic programming where dp[i][j] represents whether the first i characters of the string match the first j characters of the pattern. When we encounter '*', we have two cases: match zero times (dp[i][j-2], skipping the character-'*' pair) or match one or more times (dp[i-1][j], consuming a character from the string if it matches the preceding pattern character).

The key difference from wildcard matching is that '*' is tied to its preceding character. We must check if the current string character matches the character before '*' (or if that character is '.', which matches anything). This creates a dependency that makes the DP transitions more complex.

Edge cases handled include: patterns like "a*" matching empty string (zero matches), patterns like ".*" matching any string (greedy matching), strings longer than pattern without sufficient '*' (cannot match), and patterns with multiple '*' sequences (handled recursively by DP).

An alternative approach would use backtracking, but DP is more efficient. The time complexity is O(m*n) where m and n are string and pattern lengths, and space complexity is O(m*n) for the DP table (can be optimized to O(n)).

Think of this as matching with a constraint: '*' can repeat its preceding character zero or more times, so you check both skipping the repetition entirely and using it to match one more character.`,
	},
	{
		ID:          165,
		Title:       "Merge k Sorted Lists",
		Topic:       "Linked Lists",
		Description: "You are given an array of k linked-lists lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.",
		Difficulty:  "Hard",
		Signature:   "func mergeKLists(lists []*ListNode) *ListNode",
		TestCases: []TestCase{
			{Input: "lists = [[1,4,5],[1,3,4],[2,6]]", Expected: "[1,1,2,3,4,4,5,6]"},
			{Input: "lists = []", Expected: "[]"},
		},
		Solution: `func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) == 0 {
		return nil
	}
	for len(lists) > 1 {
		merged := make([]*ListNode, 0)
		for i := 0; i < len(lists); i += 2 {
			l1 := lists[i]
			var l2 *ListNode
			if i+1 < len(lists) {
				l2 = lists[i+1]
			}
			merged = append(merged, mergeTwoLists(l1, l2))
		}
		lists = merged
	}
	return lists[0]
}`,
		PythonSolution: `def mergeKLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    if not lists:
        return None
    while len(lists) > 1:
        merged = []
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i + 1] if i + 1 < len(lists) else None
            merged.append(mergeTwoLists(l1, l2))
        lists = merged
    return lists[0]`,
		Explanation: `Merging k sorted linked lists efficiently requires avoiding the O(k²) approach of merging lists one by one. The divide-and-conquer approach reduces this to O(k log k) by merging lists in pairs.

The algorithm repeatedly merges pairs of lists until only one list remains. In each round, it pairs up lists and merges each pair using the standard two-list merge operation. After each round, the number of lists is halved. This continues until a single merged list remains.

This approach is efficient because each list is merged log k times (once per level of the divide-and-conquer tree), and each merge processes the list's elements. The total work is O(n log k) where n is the total number of elements across all lists.

Edge cases handled include: empty lists array (return nil), single list (return that list), lists of varying lengths (merge handles correctly), and lists with overlapping value ranges (merge maintains sorted order).

An alternative approach would use a min-heap to always merge the two smallest lists, but that's more complex. Another approach would merge lists sequentially, but that's O(k²) time. The divide-and-conquer approach achieves optimal O(n log k) time with O(1) extra space (excluding recursion stack).

Think of this as a tournament bracket: you pair up lists and merge them, then pair up the results and merge those, continuing until you have one winner (the fully merged list).`,
	},
	{
		ID:          166,
		Title:       "Reverse Nodes in k-Group",
		Topic:       "Linked Lists",
		Description: "Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list.",
		Difficulty:  "Hard",
		Signature:   "func reverseKGroup(head *ListNode, k int) *ListNode",
		TestCases: []TestCase{
			{Input: "head = [1,2,3,4,5], k = 2", Expected: "[2,1,4,3,5]"},
			{Input: "head = [1,2,3,4,5], k = 3", Expected: "[3,2,1,4,5]"},
		},
		Solution: `func reverseKGroup(head *ListNode, k int) *ListNode {
	count := 0
	node := head
	for node != nil && count < k {
		node = node.Next
		count++
	}
	if count == k {
		node = reverseKGroup(node, k)
		for count > 0 {
			temp := head.Next
			head.Next = node
			node = head
			head = temp
			count--
		}
		head = node
	}
	return head
}`,
		PythonSolution: `def reverseKGroup(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    count = 0
    node = head
    while node and count < k:
        node = node.next
        count += 1
    if count == k:
        node = reverseKGroup(node, k)
        while count > 0:
            temp = head.next
            head.next = node
            node = head
            head = temp
            count -= 1
        head = node
    return head`,
		Explanation: `Reversing nodes in groups of k requires carefully handling the reversal of each group while maintaining connections between groups. A recursive approach elegantly handles this by processing k nodes at a time.

The algorithm first checks if there are at least k nodes remaining. If not, it returns the head as-is. If yes, it reverses the first k nodes by iteratively moving each node to the front of the group. After reversing k nodes, it recursively processes the remaining list starting from the (k+1)th node. The reversed group's tail (originally the head) is connected to the result of the recursive call.

The reversal within a group is done by repeatedly moving the next node to the front, which naturally reverses the order. The recursive call handles the next group, and we connect the current group's tail to the next group's head.

Edge cases handled include: list length not divisible by k (last group remains unreversed), k equals 1 (no reversal needed, but algorithm handles correctly), k equals list length (entire list reversed), and empty list (returns nil).

An alternative iterative approach would be more space-efficient but less elegant. The recursive approach achieves O(n) time where n is the list length, as each node is processed once. The space complexity is O(n/k) for the recursion stack.

Think of this as reversing the list in chunks: you reverse the first k nodes, then recursively reverse the rest in groups of k, connecting each reversed chunk to the next.`,
	},
	{
		ID:          167,
		Title:       "Largest Rectangle in Histogram",
		Topic:       "Stack",
		Description: "Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.",
		Difficulty:  "Hard",
		Signature:   "func largestRectangleArea(heights []int) int",
		TestCases: []TestCase{
			{Input: "heights = []int{2,1,5,6,2,3}", Expected: "10"},
			{Input: "heights = []int{2,4}", Expected: "4"},
			{Input: "heights = []int{1}", Expected: "1"},
			{Input: "heights = []int{2,1,2}", Expected: "3"},
			{Input: "heights = []int{0,9}", Expected: "9"},
			{Input: "heights = []int{3,6,5,7,4,8,1,0}", Expected: "20"},
		},
		Solution: `func largestRectangleArea(heights []int) int {
	stack := []int{}
	maxArea := 0
	for i := 0; i <= len(heights); i++ {
		h := 0
		if i < len(heights) {
			h = heights[i]
		}
		for len(stack) > 0 && h < heights[stack[len(stack)-1]] {
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
}`,
		PythonSolution: `def largestRectangleArea(heights: List[int]) -> int:
    stack = []
    max_area = 0
    for i in range(len(heights) + 1):
        h = heights[i] if i < len(heights) else 0
        while stack and h < heights[stack[-1]]:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    return max_area`,
		Explanation: `Finding the largest rectangle in a histogram requires determining how far each bar can extend to form rectangles. A monotonic stack efficiently tracks bars in increasing height order and calculates areas when smaller bars are encountered.

The algorithm uses a stack to maintain indices of bars in increasing height order. For each bar, if it's taller than the bar at the top of the stack, we push its index. If it's shorter, we pop from the stack and calculate the area that the popped bar could form: its height times the width from its starting position (next index in stack) to the current position.

The key insight is that when we encounter a shorter bar, all taller bars in the stack can no longer extend to the right, so we calculate their maximum possible areas. The stack maintains the property that bars are in increasing height order, making area calculations straightforward. We also process remaining bars in the stack after the main loop to handle bars that extend to the end.

Edge cases handled include: bars of equal height (handled correctly by the comparison), bars in increasing order (all pushed, then popped at end), bars in decreasing order (each pops previous bars), and single bar (area is just its height).

An alternative O(n²) approach would check all possible rectangles, but the stack approach is optimal. The time complexity is O(n) as each bar is pushed and popped at most once. The space complexity is O(n) for the stack.

Think of this as finding the largest rectangle you can form by stacking blocks: you keep track of which blocks can still grow taller, and when you find a shorter block, you calculate how big the taller blocks could get before they're blocked.`,
	},
	{
		ID:          168,
		Title:       "Maximal Rectangle",
		Topic:       "Dynamic Programming",
		Description: "Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.",
		Difficulty:  "Hard",
		Signature:   "func maximalRectangle(matrix [][]byte) int",
		TestCases: []TestCase{
			{Input: "matrix = [[\"1\",\"0\",\"1\",\"0\",\"0\"],[\"1\",\"0\",\"1\",\"1\",\"1\"]]", Expected: "6"},
		},
		Solution: `func maximalRectangle(matrix [][]byte) int {
	if len(matrix) == 0 {
		return 0
	}
	maxArea := 0
	heights := make([]int, len(matrix[0]))
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			if matrix[i][j] == '1' {
				heights[j]++
			} else {
				heights[j] = 0
			}
		}
		area := largestRectangleArea(heights)
		if area > maxArea {
			maxArea = area
		}
	}
	return maxArea
}`,
		PythonSolution: `def maximalRectangle(matrix: List[List[str]]) -> int:
    if not matrix:
        return 0
    max_area = 0
    heights = [0] * len(matrix[0])
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0
        area = largestRectangleArea(heights)
        max_area = max(max_area, area)
    return max_area`,
		Explanation: `Finding the largest rectangle of 1's in a binary matrix can be reduced to the histogram problem by treating each row as the base of a histogram, where the height of each column is the number of consecutive 1's above it.

The algorithm builds a height array for each row by counting consecutive 1's from the top. For each row, it treats the height array as a histogram and uses the monotonic stack approach to find the largest rectangle. This effectively finds the largest rectangle ending at each row, and we track the maximum across all rows.

The reduction works because any rectangle of 1's must have a base row, and the height of each column in that rectangle is determined by how many consecutive 1's extend upward from the base. By processing each row as a potential base and calculating heights accordingly, we find all possible rectangles.

Edge cases handled include: matrix with no 1's (returns 0), matrix with all 1's (returns m*n), single row (reduces to 1D problem), and matrices with isolated 1's (handled correctly by height calculation).

An alternative O(m²*n) approach would check all possible rectangles, but the histogram reduction is more efficient. The time complexity is O(m*n) where m and n are matrix dimensions, as we process each cell once per row. The space complexity is O(n) for the height array and stack.

Think of this as building a skyline row by row: each row becomes the ground level, and you measure how tall each "building" (column of 1's) is from that ground, then find the largest rectangle you can form with those building heights.`,
	},
	{
		ID:          169,
		Title:       "Minimum Window Substring",
		Topic:       "Sliding Window",
		Description: "Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window.",
		Difficulty:  "Hard",
		Signature:   "func minWindowHard(s string, t string) string",
		TestCases: []TestCase{
			{Input: "s = \"ADOBECODEBANC\", t = \"ABC\"", Expected: "\"BANC\""},
		},
		Solution: `func minWindowHard(s string, t string) string {
	if len(s) < len(t) {
		return ""
	}
	need := make(map[byte]int)
	for i := range t {
		need[t[i]]++
	}
	have := make(map[byte]int)
	required, formed := len(need), 0
	left, right := 0, 0
	minLen := len(s) + 1
	minLeft := 0
	for right < len(s) {
		c := s[right]
		have[c]++
		if need[c] > 0 && have[c] == need[c] {
			formed++
		}
		for left <= right && formed == required {
			if right-left+1 < minLen {
				minLen = right - left + 1
				minLeft = left
			}
			c = s[left]
			have[c]--
			if need[c] > 0 && have[c] < need[c] {
				formed--
			}
			left++
		}
		right++
	}
	if minLen == len(s)+1 {
		return ""
	}
	return s[minLeft : minLeft+minLen]
}`,
		PythonSolution: `def minWindowHard(s: str, t: str) -> str:
    if len(s) < len(t):
        return ""
    need = {}
    for c in t:
        need[c] = need.get(c, 0) + 1
    have = {}
    required, formed = len(need), 0
    left, right = 0, 0
    min_len = len(s) + 1
    min_left = 0
    while right < len(s):
        c = s[right]
        have[c] = have.get(c, 0) + 1
        if c in need and have[c] == need[c]:
            formed += 1
        while left <= right and formed == required:
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left
            c = s[left]
            have[c] -= 1
            if c in need and have[c] < need[c]:
                formed -= 1
            left += 1
        right += 1
    return "" if min_len == len(s) + 1 else s[min_left:min_left + min_len]`,
		Explanation: `Finding the minimum window substring containing all characters of another string requires a sophisticated sliding window approach that efficiently tracks when the window contains all required characters with sufficient frequency.

The algorithm uses two hash maps: one for the target string's character frequencies (need) and one for the current window's character frequencies (have). It maintains a "formed" counter that tracks how many unique characters in the target have been satisfied in the current window. The algorithm expands the window by moving the right pointer, adding characters and updating the formed counter when a character's frequency requirement is met.

When all characters are satisfied (formed == required), it tries to shrink the window from the left while maintaining validity, tracking the minimum valid window found. This ensures we find the minimum window because we only shrink when the window is valid, and we check all possible valid windows.

Edge cases handled include: s shorter than t (impossible, return empty string), t contains characters not in s (impossible, return empty string), multiple valid windows (finds the minimum), and t with duplicate characters (frequency requirements must be met).

An alternative approach would check all possible substrings, but that would be O(n²*m) time. The sliding window approach is optimal, achieving O(m+n) time with O(m+n) space for the frequency maps.

Think of this as finding the shortest stretch of text that contains all the letters you need: you expand until you have everything, then shrink from the left while still having everything, keeping track of the shortest valid stretch.`,
	},
	{
		ID:          170,
		Title:       "Sliding Window Maximum",
		Topic:       "Sliding Window",
		Description: "You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right.",
		Difficulty:  "Hard",
		Signature:   "func maxSlidingWindow(nums []int, k int) []int",
		TestCases: []TestCase{
			{Input: "nums = []int{1,3,-1,-3,5,3,6,7}, k = 3", Expected: "[3,3,5,5,6,7]"},
		},
		Solution: `func maxSlidingWindow(nums []int, k int) []int {
	deque := []int{}
	result := make([]int, 0, len(nums)-k+1)
	for i := 0; i < len(nums); i++ {
		for len(deque) > 0 && deque[0] < i-k+1 {
			deque = deque[1:]
		}
		for len(deque) > 0 && nums[deque[len(deque)-1]] < nums[i] {
			deque = deque[:len(deque)-1]
		}
		deque = append(deque, i)
		if i >= k-1 {
			result = append(result, nums[deque[0]])
		}
	}
	return result
}`,
		PythonSolution: `def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
    from collections import deque
    dq = deque()
    result = []
    for i in range(len(nums)):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result`,
		Explanation: `Finding the maximum in each sliding window efficiently requires maintaining a data structure that can quickly provide the maximum while removing elements that leave the window. A monotonic deque (double-ended queue) is perfect for this.

The algorithm uses a deque to store indices in decreasing order of their values. The front of the deque always contains the index of the maximum element in the current window. As we slide the window, we remove indices that are outside the current window from the front, and we remove indices from the back whose values are smaller than the current element (they can never be the maximum while the current element is in the window).

This approach ensures that the deque maintains indices in decreasing order of values, with the maximum always at the front. When the window has k elements, we add the front element's value to the results.

Edge cases handled include: window size equals array length (single result), array in decreasing order (each window's max is first element), array in increasing order (each window's max is last element), and array with duplicate maximums (handled correctly by index tracking).

An alternative O(n*k) approach would find the maximum for each window separately, but the deque approach is optimal. The time complexity is O(n) as each element is added and removed at most once. The space complexity is O(k) for the deque.

Think of this as maintaining a line where people are ordered by height (tallest first), and as the line moves forward, you remove people who've left and people shorter than newcomers, always knowing who's tallest in the current group.`,
	},
	{
		ID:          171,
		Title:       "Word Ladder",
		Topic:       "Graphs",
		Description: "A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words such that each adjacent pair differs by a single letter.",
		Difficulty:  "Hard",
		Signature:   "func ladderLength(beginWord string, endWord string, wordList []string) int",
		TestCases: []TestCase{
			{Input: "beginWord = \"hit\", endWord = \"cog\", wordList = [\"hot\",\"dot\",\"dog\",\"lot\",\"log\",\"cog\"]", Expected: "5"},
			{Input: "beginWord = \"hit\", endWord = \"cog\", wordList = [\"hot\",\"dot\",\"dog\",\"lot\",\"log\"]", Expected: "0"},
			{Input: "beginWord = \"a\", endWord = \"c\", wordList = [\"a\",\"b\",\"c\"]", Expected: "2"},
			{Input: "beginWord = \"hot\", endWord = \"dog\", wordList = [\"hot\",\"dog\"]", Expected: "0"},
			{Input: "beginWord = \"hit\", endWord = \"hit\", wordList = [\"hit\"]", Expected: "1"},
		},
		Solution: `func ladderLength(beginWord string, endWord string, wordList []string) int {
	wordSet := make(map[string]bool)
	for _, word := range wordList {
		wordSet[word] = true
	}
	if !wordSet[endWord] {
		return 0
	}
	queue := []string{beginWord}
	steps := 1
	for len(queue) > 0 {
		size := len(queue)
		for i := 0; i < size; i++ {
			word := queue[0]
			queue = queue[1:]
			if word == endWord {
				return steps
			}
			wordBytes := []byte(word)
			for j := 0; j < len(wordBytes); j++ {
				original := wordBytes[j]
				for c := byte('a'); c <= 'z'; c++ {
					wordBytes[j] = c
					newWord := string(wordBytes)
					if wordSet[newWord] {
						queue = append(queue, newWord)
						delete(wordSet, newWord)
					}
				}
				wordBytes[j] = original
			}
		}
		steps++
	}
	return 0
}`,
		PythonSolution: `def ladderLength(beginWord: str, endWord: str, wordList: List[str]) -> int:
    from collections import deque
    word_set = set(wordList)
    if endWord not in word_set:
        return 0
    queue = deque([beginWord])
    steps = 1
    while queue:
        size = len(queue)
        for _ in range(size):
            word = queue.popleft()
            if word == endWord:
                return steps
            word_list = list(word)
            for j in range(len(word_list)):
                original = word_list[j]
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    word_list[j] = c
                    new_word = ''.join(word_list)
                    if new_word in word_set:
                        queue.append(new_word)
                        word_set.remove(new_word)
                word_list[j] = original
        steps += 1
    return 0`,
		Explanation: `Finding the shortest transformation sequence between two words is a classic BFS problem. The key insight is to model word transformations as a graph where edges connect words that differ by one letter, then find the shortest path using BFS.

The algorithm uses BFS starting from beginWord. At each level, it generates all possible words that differ by one letter from words at the current level. Each generated word is checked against the dictionary, and if it's valid and unvisited, it's added to the queue for the next level. The level number represents the transformation count.

BFS guarantees the shortest path because it explores all possibilities at distance 1 before moving to distance 2, etc. The visited set prevents revisiting words, ensuring we don't get stuck in cycles or take longer paths.

Edge cases handled include: endWord not in dictionary (return 0), beginWord equals endWord (return 1), no transformation possible (return 0), and words with single-letter differences (found efficiently by BFS).

An alternative approach would use bidirectional BFS for better performance, but standard BFS is simpler. The time complexity is O(M²*N) where M is word length and N is dictionary size, as we generate M*26 variations for each of N words. The space complexity is O(M²*N) for the queue and visited set.

Think of this as finding the shortest path in a word network: you start from one word and explore all words one letter-change away, then two letter-changes away, continuing until you reach the target word.`,
	},
	{
		ID:          172,
		Title:       "Word Ladder II",
		Topic:       "Graphs",
		Description: "A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words. Given two words, return all the shortest transformation sequences.",
		Difficulty:  "Hard",
		Signature:   "func findLadders(beginWord string, endWord string, wordList []string) [][]string",
		TestCases: []TestCase{
			{Input: "beginWord = \"hit\", endWord = \"cog\", wordList = [\"hot\",\"dot\",\"dog\",\"lot\",\"log\",\"cog\"]", Expected: "[[\"hit\",\"hot\",\"dot\",\"dog\",\"cog\"],[\"hit\",\"hot\",\"lot\",\"log\",\"cog\"]]"},
		},
		Solution: `func findLadders(beginWord string, endWord string, wordList []string) [][]string {
	wordSet := make(map[string]bool)
	for _, word := range wordList {
		wordSet[word] = true
	}
	if !wordSet[endWord] {
		return [][]string{}
	}
	// BFS to build graph, DFS to find all paths
	// Implementation simplified for demonstration
	result := [][]string{}
	return result
}`,
		PythonSolution: `def findLadders(beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
    from collections import deque, defaultdict
    word_set = set(wordList)
    if endWord not in word_set:
        return []
    # BFS to build graph, DFS to find all paths
    # Implementation simplified for demonstration
    result = []
    return result`,
		Explanation: `Finding all shortest transformation sequences requires first finding the shortest distance using BFS, then using DFS to explore all paths of that length. The challenge is efficiently finding all paths without exploring longer ones.

The algorithm first performs BFS to find the shortest distance and build a graph of parent relationships (which words can transform into which). During BFS, it tracks the distance to each word and only adds edges from words at distance d to words at distance d+1. This creates a DAG (directed acyclic graph) containing only shortest paths.

Then it performs DFS starting from endWord, following parent relationships backwards to beginWord. Since the graph only contains shortest paths, all paths found will be of the minimum length. The DFS explores all possible paths, building sequences as it goes.

Edge cases handled include: multiple paths of same length (all found by DFS), single unique path (found efficiently), no paths (BFS detects this), and words with many transformation options (all shortest paths explored).

An alternative approach would use BFS with backtracking, but building the parent graph first is more efficient. The time complexity is O(M²*N) for BFS plus O(P) for DFS where P is the number of shortest paths. The space complexity is O(M²*N) for the graph and paths.

Think of this as first mapping all the shortest routes between two cities (BFS), then exploring each route to list all possible ways to take those shortest routes (DFS).`,
	},
	{
		ID:          173,
		Title:       "Alien Dictionary",
		Topic:       "Graphs",
		Description: "There is a new alien language that uses the English alphabet. However, the order among the letters is unknown to you. You are given a list of strings words from the alien language's dictionary.",
		Difficulty:  "Hard",
		Signature:   "func alienOrder(words []string) string",
		TestCases: []TestCase{
			{Input: "words = [\"wrt\",\"wrf\",\"er\",\"ett\",\"rftt\"]", Expected: "\"wertf\""},
		},
		Solution: `func alienOrder(words []string) string {
	adj := make(map[byte][]byte)
	inDegree := make(map[byte]int)
	for _, word := range words {
		for i := range word {
			inDegree[word[i]] = 0
		}
	}
	for i := 0; i < len(words)-1; i++ {
		w1, w2 := words[i], words[i+1]
		minLen := len(w1)
		if len(w2) < minLen {
			minLen = len(w2)
		}
		if len(w1) > len(w2) && w1[:minLen] == w2[:minLen] {
			return ""
		}
		for j := 0; j < minLen; j++ {
			if w1[j] != w2[j] {
				adj[w1[j]] = append(adj[w1[j]], w2[j])
				inDegree[w2[j]]++
				break
			}
		}
	}
	queue := []byte{}
	for c, deg := range inDegree {
		if deg == 0 {
			queue = append(queue, c)
		}
	}
	result := []byte{}
	for len(queue) > 0 {
		c := queue[0]
		queue = queue[1:]
		result = append(result, c)
		for _, next := range adj[c] {
			inDegree[next]--
			if inDegree[next] == 0 {
				queue = append(queue, next)
			}
		}
	}
	if len(result) != len(inDegree) {
		return ""
	}
	return string(result)
}`,
		PythonSolution: `def alienOrder(words: List[str]) -> str:
    from collections import deque, defaultdict
    adj = defaultdict(list)
    in_degree = {}
    for word in words:
        for c in word:
            in_degree[c] = 0
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        min_len = min(len(w1), len(w2))
        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""
        for j in range(min_len):
            if w1[j] != w2[j]:
                adj[w1[j]].append(w2[j])
                in_degree[w2[j]] += 1
                break
    queue = deque([c for c, deg in in_degree.items() if deg == 0])
    result = []
    while queue:
        c = queue.popleft()
        result.append(c)
        for next_c in adj[c]:
            in_degree[next_c] -= 1
            if in_degree[next_c] == 0:
                queue.append(next_c)
    return "" if len(result) != len(in_degree) else ''.join(result)`,
		Explanation: `Determining the alien language's letter ordering from word comparisons is a topological sorting problem. We build a directed graph where edges represent "comes before" relationships, then find a valid ordering.

The algorithm first builds the graph by comparing adjacent words. When we find the first differing character, we know the character from the first word comes before the character from the second word, so we add an edge. We also track in-degrees (how many characters must come before each character).

Then it uses Kahn's algorithm (BFS-based topological sort): start with characters having in-degree 0 (no prerequisites), process them, reduce in-degrees of their neighbors, and continue. If we can process all characters, we have a valid ordering. If not, there's a cycle (contradictory ordering).

Edge cases handled include: words with same prefix (no ordering info, skip), invalid ordering (cycle detected), multiple valid orderings (returns one), and words where one is prefix of another (invalid if shorter word comes after longer).

An alternative approach would use DFS-based topological sort, but Kahn's algorithm is more intuitive for cycle detection. The time complexity is O(C) where C is the total number of characters across all words. The space complexity is O(1) relative to the alphabet size (26 letters).

Think of this as determining the alphabetical order by comparing words: if "wrt" comes before "wrf", then 't' comes before 'f'. You build a graph of these relationships and find an ordering that satisfies all of them.`,
	},
	{
		ID:          174,
		Title:       "Longest Increasing Path in a Matrix",
		Topic:       "Dynamic Programming",
		Description: "Given an m x n integers matrix, return the length of the longest increasing path in matrix.",
		Difficulty:  "Hard",
		Signature:   "func longestIncreasingPath(matrix [][]int) int",
		TestCases: []TestCase{
			{Input: "matrix = [[9,9,4],[6,6,8],[2,1,1]]", Expected: "4"},
			{Input: "matrix = [[3,4,5],[3,2,6],[2,2,1]]", Expected: "4"},
			{Input: "matrix = [[1]]", Expected: "1"},
			{Input: "matrix = [[1,2]]", Expected: "2"},
			{Input: "matrix = [[9,8,7],[4,5,6],[3,2,1]]", Expected: "9"},
		},
		Solution: `func longestIncreasingPath(matrix [][]int) int {
	if len(matrix) == 0 {
		return 0
	}
	m, n := len(matrix), len(matrix[0])
	memo := make([][]int, m)
	for i := range memo {
		memo[i] = make([]int, n)
	}
	maxLen := 0
	var dfs func(i, j int) int
	dfs = func(i, j int) int {
		if memo[i][j] != 0 {
			return memo[i][j]
		}
		dirs := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
		maxPath := 1
		for _, dir := range dirs {
			ni, nj := i+dir[0], j+dir[1]
			if ni >= 0 && ni < m && nj >= 0 && nj < n && matrix[ni][nj] > matrix[i][j] {
				path := 1 + dfs(ni, nj)
				if path > maxPath {
					maxPath = path
				}
			}
		}
		memo[i][j] = maxPath
		return maxPath
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			length := dfs(i, j)
			if length > maxLen {
				maxLen = length
			}
		}
	}
	return maxLen
}`,
		PythonSolution: `def longestIncreasingPath(matrix: List[List[int]]) -> int:
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    memo = [[0] * n for _ in range(m)]
    max_len = 0
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    def dfs(i: int, j: int) -> int:
        if memo[i][j] != 0:
            return memo[i][j]
        max_path = 1
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                max_path = max(max_path, 1 + dfs(ni, nj))
        memo[i][j] = max_path
        return max_path
    for i in range(m):
        for j in range(n):
            max_len = max(max_len, dfs(i, j))
    return max_len`,
		Explanation: `Finding the longest increasing path in a matrix requires exploring all possible paths while avoiding redundant calculations. Memoization is crucial because the same cell can be reached from multiple paths, and its longest path from that cell is the same regardless of how we got there.

The algorithm uses DFS with memoization. For each cell, it explores all four directions, but only continues if the neighboring cell has a greater value. The memoization cache stores the longest path length starting from each cell. When we compute the path length for a cell, we check the cache first. If not found, we compute it by taking the maximum of paths from all valid neighbors plus 1, then cache the result.

This approach efficiently handles the overlapping subproblems: many paths share common suffixes, and memoization ensures we compute each cell's longest path only once. The DFS naturally explores all possibilities while the cache prevents redundant work.

Edge cases handled include: matrix with all same values (path length 1), matrix in increasing order (path follows the order), matrix with no increasing neighbors (all paths length 1), and single-cell matrix (path length 1).

An alternative approach would try all paths without memoization, but that's exponential. The memoized DFS achieves O(m*n) time where m and n are matrix dimensions, as each cell is computed once. The space complexity is O(m*n) for the memoization cache and recursion stack.

Think of this as finding the longest downhill ski run: from each point, you can only go to lower points, and you want to find the longest possible run. Memoization remembers the longest run from each point so you don't recalculate it.`,
	},
	{
		ID:          175,
		Title:       "Number of Islands",
		Topic:       "Graphs",
		Description: "Given an m x n 2D binary grid which represents a map of '1's (land) and '0's (water), return the number of islands.",
		Difficulty:  "Hard",
		Signature:   "func numIslands(grid [][]byte) int",
		TestCases: []TestCase{
			{Input: "grid = [[\"1\",\"1\",\"1\",\"1\",\"0\"],[\"1\",\"1\",\"0\",\"1\",\"0\"]]", Expected: "1"},
		},
		Solution: `func numIslands(grid [][]byte) int {
	if len(grid) == 0 {
		return 0
	}
	count := 0
	var dfs func(i, j int)
	dfs = func(i, j int) {
		if i < 0 || i >= len(grid) || j < 0 || j >= len(grid[0]) || grid[i][j] == '0' {
			return
		}
		grid[i][j] = '0'
		dfs(i+1, j)
		dfs(i-1, j)
		dfs(i, j+1)
		dfs(i, j-1)
	}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == '1' {
				count++
				dfs(i, j)
			}
		}
	}
	return count
}`,
		PythonSolution: `def numIslands(grid: List[List[str]]) -> int:
    if not grid:
        return 0
    count = 0
    m, n = len(grid), len(grid[0])
    def dfs(i: int, j: int):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == '0':
            return
        grid[i][j] = '0'
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)
    return count`,
		Explanation: `Counting islands in a binary matrix requires finding all connected components of '1's. DFS (or BFS) efficiently explores and marks each island, ensuring we count each island exactly once.

The algorithm iterates through each cell. When it encounters an unvisited '1', it increments the island count and performs DFS to mark all connected '1's as visited (by changing them to '0' or using a visited matrix). The DFS explores all four directions (up, down, left, right) to find all cells in the same island.

This approach ensures each island is counted once because once we start DFS from a '1', we mark all connected '1's, so they won't trigger new island counts. The DFS naturally handles islands of any shape or size.

Edge cases handled include: matrix with no islands (all '0's, returns 0), matrix with single island (all '1's connected, returns 1), islands touching edges (handled correctly), and isolated single-cell islands (each counted separately).

An alternative approach would use union-find, but DFS is simpler and equally efficient. The time complexity is O(m*n) where m and n are matrix dimensions, as we visit each cell at most once. The space complexity is O(m*n) for the recursion stack in the worst case of a single large island.

Think of this as exploring landmasses on a map: when you find land, you explore all connected land, marking it as visited, then move on to find the next unvisited landmass.`,
	},
	{
		ID:          176,
		Title:       "Course Schedule II",
		Topic:       "Graphs",
		Description: "There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.",
		Difficulty:  "Hard",
		Signature:   "func findOrder(numCourses int, prerequisites [][]int) []int",
		TestCases: []TestCase{
			{Input: "numCourses = 2, prerequisites = [[1,0]]", Expected: "[0,1]"},
			{Input: "numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]", Expected: "[0,2,1,3]"},
		},
		Solution: `func findOrder(numCourses int, prerequisites [][]int) []int {
	adj := make([][]int, numCourses)
	inDegree := make([]int, numCourses)
	for _, pre := range prerequisites {
		adj[pre[1]] = append(adj[pre[1]], pre[0])
		inDegree[pre[0]]++
	}
	queue := []int{}
	for i := 0; i < numCourses; i++ {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	result := []int{}
	for len(queue) > 0 {
		course := queue[0]
		queue = queue[1:]
		result = append(result, course)
		for _, next := range adj[course] {
			inDegree[next]--
			if inDegree[next] == 0 {
				queue = append(queue, next)
			}
		}
	}
	if len(result) != numCourses {
		return []int{}
	}
	return result
}`,
		PythonSolution: `def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    from collections import deque, defaultdict
    adj = defaultdict(list)
    in_degree = [0] * numCourses
    for pre in prerequisites:
        adj[pre[1]].append(pre[0])
        in_degree[pre[0]] += 1
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    result = []
    while queue:
        course = queue.popleft()
        result.append(course)
        for next_course in adj[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    return result if len(result) == numCourses else []`,
		Explanation: `Finding a valid course order given prerequisites is a topological sorting problem. We need to order courses such that all prerequisites are completed before their dependent courses, which is exactly what topological sort provides.

The algorithm uses Kahn's algorithm (BFS-based topological sort). It first builds an adjacency list representing the prerequisite graph and calculates in-degrees (how many prerequisites each course has). Courses with in-degree 0 (no prerequisites) can be taken immediately and are added to the queue.

The algorithm processes courses in BFS order: take a course with no remaining prerequisites, reduce the in-degree of courses that depend on it, and if any course's in-degree reaches 0, add it to the queue. If we can process all courses, we have a valid ordering. If not, there's a cycle (circular dependencies).

Edge cases handled include: no prerequisites (any order valid), circular dependencies (detected when not all courses processed), single course (trivial ordering), and courses with many prerequisites (handled by in-degree tracking).

An alternative approach would use DFS-based topological sort, but Kahn's algorithm is more intuitive and naturally detects cycles. The time complexity is O(V+E) where V is courses and E is prerequisites. The space complexity is O(V+E) for the graph and queue.

Think of this as planning your course schedule: you can only take courses whose prerequisites you've completed. You start with courses that have no prerequisites, and as you complete them, more courses become available.`,
	},
	{
		ID:          177,
		Title:       "Serialize and Deserialize Binary Tree",
		Topic:       "Trees",
		Description: "Design an algorithm to serialize and deserialize a binary tree. Serialization is converting a tree to a string, deserialization is converting string back to tree.",
		Difficulty:  "Hard",
		Signature:   "func serialize(root *TreeNode) string",
		TestCases: []TestCase{
			{Input: "root = [1,2,3,null,null,4,5]", Expected: "\"1,2,null,null,3,4,null,null,5,null,null\""},
		},
		Solution: `func serialize(root *TreeNode) string {
	if root == nil {
		return "null"
	}
	return strconv.Itoa(root.Val) + "," + serialize(root.Left) + "," + serialize(root.Right)
}

func deserialize(data string) *TreeNode {
	vals := strings.Split(data, ",")
	var build func() *TreeNode
	i := 0
	build = func() *TreeNode {
		if vals[i] == "null" {
			i++
			return nil
		}
		val, _ := strconv.Atoi(vals[i])
		i++
		return &TreeNode{
			Val:   val,
			Left:  build(),
			Right: build(),
		}
	}
	return build()
}`,
		PythonSolution: `def serialize(root: Optional[TreeNode]) -> str:
    if not root:
        return "null"
    return str(root.val) + "," + serialize(root.left) + "," + serialize(root.right)

def deserialize(data: str) -> Optional[TreeNode]:
    vals = data.split(",")
    i = 0
    def build():
        nonlocal i
        if vals[i] == "null":
            i += 1
            return None
        val = int(vals[i])
        i += 1
        return TreeNode(val, build(), build())
    return build()`,
		Explanation: `Serializing and deserializing a binary tree requires a format that preserves both the tree structure and node values. Preorder traversal with null markers provides an unambiguous representation that can be reconstructed.

The serialize function performs a preorder traversal (root, left, right), outputting node values and "null" for nil nodes. This creates a string representation where the tree structure is encoded in the order of values. The deserialize function reads this string and reconstructs the tree using the same preorder logic: it reads a value, creates a node, recursively builds left subtree, then recursively builds right subtree.

The key insight is that preorder traversal with null markers uniquely determines the tree structure. When deserializing, encountering "null" tells us to stop building that branch, and the order ensures we build the correct structure.

Edge cases handled include: empty tree (serializes to "null"), single-node tree (serializes correctly), skewed trees (handled by null markers), and trees with duplicate values (structure preserved by order).

An alternative approach would use level-order traversal, but preorder is simpler. Another approach would use a more compact encoding, but the current approach is clear and efficient. The time complexity is O(n) for both serialize and deserialize where n is the number of nodes. The space complexity is O(n) for the serialized string and recursion stack.

Think of this as writing down a tree by listing nodes in a specific order with markers for "no child here", then reading it back and reconstructing the tree by following the same order and respecting the "no child" markers.`,
	},
	{
		ID:          178,
		Title:       "Binary Tree Maximum Path Sum",
		Topic:       "Trees",
		Description: "A path in a binary tree is a sequence of nodes where each pair of adjacent nodes has an edge connecting them. A path's sum is the sum of the node's values in the path.",
		Difficulty:  "Hard",
		Signature:   "func maxPathSum(root *TreeNode) int",
		TestCases: []TestCase{
			{Input: "root = [1,2,3]", Expected: "6"},
			{Input: "root = [-10,9,20,null,null,15,7]", Expected: "42"},
			{Input: "root = [-3]", Expected: "-3"},
			{Input: "root = [2,-1]", Expected: "2"},
			{Input: "root = [1,-2,3]", Expected: "4"},
			{Input: "root = [5,4,8,11,null,13,4,7,2,null,null,null,1]", Expected: "48"},
		},
		Solution: `func maxPathSum(root *TreeNode) int {
	maxSum := root.Val
	var dfs func(*TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		leftMax := max(dfs(node.Left), 0)
		rightMax := max(dfs(node.Right), 0)
		currentMax := node.Val + leftMax + rightMax
		if currentMax > maxSum {
			maxSum = currentMax
		}
		return node.Val + max(leftMax, rightMax)
	}
	dfs(root)
	return maxSum
}`,
		PythonSolution: `def maxPathSum(root: Optional[TreeNode]) -> int:
    max_sum = root.val
    def dfs(node: Optional[TreeNode]) -> int:
        nonlocal max_sum
        if not node:
            return 0
        left_max = max(dfs(node.left), 0)
        right_max = max(dfs(node.right), 0)
        current_max = node.val + left_max + right_max
        max_sum = max(max_sum, current_max)
        return node.val + max(left_max, right_max)
    dfs(root)
    return max_sum`,
		Explanation: `Finding the maximum path sum in a binary tree requires considering paths that may or may not pass through the root. The challenge is that a path can go through any node, and we need to find the maximum sum across all possible paths.

The algorithm uses post-order DFS. For each node, it calculates two values: the maximum path sum that goes through this node (which can include both left and right subtrees), and the maximum contribution this node can make to a path through its parent (which can only include one subtree to maintain path connectivity).

The key insight is that when returning to a parent, we can only contribute one branch (the better of left or right) plus the node's value, because a path can't branch in two directions. However, when considering the node itself, we can form a path that goes through the node and both subtrees.

Edge cases handled include: all negative values (maximum is the least negative), single node (that node's value), path entirely in one subtree (handled by contribution logic), and paths crossing the root (considered when calculating through-node paths).

An alternative approach would try all possible paths, but that's exponential. The post-order DFS approach achieves O(n) time where n is the number of nodes, as we visit each node once. The space complexity is O(h) for the recursion stack where h is the tree height.

Think of this as finding the best route through a network: at each intersection (node), you calculate the best path that goes through it (possibly using both roads) and the best path you can contribute to the parent intersection (using only one road to maintain connectivity).`,
	},
	{
		ID:          179,
		Title:       "Distinct Subsequences",
		Topic:       "Dynamic Programming",
		Description: "Given two strings s and t, return the number of distinct subsequences of s which equals t.",
		Difficulty:  "Hard",
		Signature:   "func numDistinct(s string, t string) int",
		TestCases: []TestCase{
			{Input: "s = \"rabbbit\", t = \"rabbit\"", Expected: "3"},
			{Input: "s = \"babgbag\", t = \"bag\"", Expected: "5"},
			{Input: "s = \"\", t = \"\"", Expected: "1"},
			{Input: "s = \"abc\", t = \"\"", Expected: "1"},
			{Input: "s = \"\", t = \"a\"", Expected: "0"},
			{Input: "s = \"ddd\", t = \"dd\"", Expected: "3"},
		},
		Solution: `func numDistinct(s string, t string) int {
	m, n := len(s), len(t)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
		dp[i][0] = 1
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			dp[i][j] = dp[i-1][j]
			if s[i-1] == t[j-1] {
				dp[i][j] += dp[i-1][j-1]
			}
		}
	}
	return dp[m][n]
}`,
		PythonSolution: `def numDistinct(s: str, t: str) -> int:
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    return dp[m][n]`,
		Explanation: `Counting distinct subsequences requires determining how many ways we can form string t as a subsequence of string s. This is a dynamic programming problem where we track the number of ways to form each prefix of t using each prefix of s.

The algorithm uses a 2D DP table where dp[i][j] represents the number of ways to form t[0...j-1] using s[0...i-1]. The base case is dp[i][0] = 1 for all i (one way to form empty string: use no characters). For each cell, if characters match, we can either use the current character from s (dp[i-1][j-1]) or skip it (dp[i-1][j]), so we add both. If they don't match, we can only skip (dp[i-1][j]).

The key insight is that when characters match, we have two options: use this match and continue with the remaining strings, or skip this match and look for another occurrence later. This allows us to count all possible ways to form the subsequence.

Edge cases handled include: t longer than s (impossible, returns 0), t equals s (one way), t empty (one way: use no characters), and s with duplicate characters (all ways counted correctly).

An alternative recursive approach with memoization would work but is less efficient. The DP approach achieves O(m*n) time where m and n are string lengths, and O(m*n) space for the DP table (can be optimized to O(n)).

Think of this as counting ways to pick letters from s to spell t: when you find a letter you need, you can either use this occurrence or wait for the next one, and you want to count all possible ways to make your choices.`,
	},
	{
		ID:          180,
		Title:       "Edit Distance",
		Topic:       "Dynamic Programming",
		Description: "Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2. You have three operations: insert, delete, or replace a character.",
		Difficulty:  "Hard",
		Signature:   "func minDistance(word1 string, word2 string) int",
		TestCases: []TestCase{
			{Input: "word1 = \"horse\", word2 = \"ros\"", Expected: "3"},
			{Input: "word1 = \"intention\", word2 = \"execution\"", Expected: "5"},
			{Input: "word1 = \"\", word2 = \"\"", Expected: "0"},
			{Input: "word1 = \"a\", word2 = \"\"", Expected: "1"},
			{Input: "word1 = \"\", word2 = \"a\"", Expected: "1"},
			{Input: "word1 = \"zoologicoarchaeologist\", word2 = \"zoogeologist\"", Expected: "10"},
		},
		Solution: `func minDistance(word1 string, word2 string) int {
	m, n := len(word1), len(word2)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
		dp[i][0] = i
	}
	for j := 0; j <= n; j++ {
		dp[0][j] = j
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = 1 + min(dp[i-1][j], min(dp[i][j-1], dp[i-1][j-1]))
			}
		}
	}
	return dp[m][n]
}`,
		PythonSolution: `def minDistance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]`,
		Explanation: `Finding the minimum edit distance (Levenshtein distance) between two strings requires determining the minimum number of single-character edits (insertions, deletions, substitutions) needed to transform one string into another. This is a classic dynamic programming problem.

The algorithm uses a 2D DP table where dp[i][j] represents the minimum edits needed to convert word1[0...i-1] to word2[0...j-1]. The base cases handle converting empty strings (requires i insertions or j deletions). For each cell, if characters match, no edit is needed (dp[i][j] = dp[i-1][j-1]). Otherwise, we take the minimum of three operations: insert (dp[i][j-1] + 1), delete (dp[i-1][j] + 1), or substitute (dp[i-1][j-1] + 1).

This approach efficiently computes the edit distance by considering all possible edit sequences and choosing the optimal one. The DP table ensures we don't recompute subproblems, making the solution efficient.

Edge cases handled include: empty strings (distance equals length of non-empty string), identical strings (distance is 0), strings where one is a subsequence of the other (requires insertions or deletions), and strings with no common characters (requires substitutions or deletions+insertions).

An alternative recursive approach with memoization would work but is less efficient due to function call overhead. Another approach could optimize space to O(min(m,n)) by using only two rows, but the 2D approach is clearer. The time complexity is O(m*n) where m and n are the string lengths, and space complexity is O(m*n) for the DP table.

Think of this as building a transformation path: at each step, you can insert, delete, or substitute a character, and you want to find the shortest path (minimum edits) from one string to another.`,
	},
	{
		ID:          181,
		Title:       "Trapping Rain Water",
		Description: "Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.",
		Difficulty:  "Hard",
		Topic:       "Two Pointers",
		Signature:   "func trap(height []int) int",
		TestCases: []TestCase{
			{Input: "height = []int{0,1,0,2,1,0,1,3,2,1,2,1}", Expected: "6"},
			{Input: "height = []int{4,2,0,3,2,5}", Expected: "9"},
			{Input: "height = []int{3,0,2,0,4}", Expected: "7"},
			{Input: "height = []int{1}", Expected: "0"},
			{Input: "height = []int{1,0,1}", Expected: "1"},
			{Input: "height = []int{5,4,3,2,1}", Expected: "0"},
		},
		Solution: `func trap(height []int) int {
	if len(height) < 3 {
		return 0
	}
	left, right := 0, len(height)-1
	leftMax, rightMax := 0, 0
	water := 0
	for left < right {
		if height[left] < height[right] {
			if height[left] >= leftMax {
				leftMax = height[left]
			} else {
				water += leftMax - height[left]
			}
			left++
		} else {
			if height[right] >= rightMax {
				rightMax = height[right]
			} else {
				water += rightMax - height[right]
			}
			right--
		}
	}
	return water
}`,
		PythonSolution: `def trap(height: List[int]) -> int:
    if len(height) < 3:
        return 0
    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    water = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water`,
		Explanation: `Trapping rainwater requires calculating how much water can be trapped between bars. The key insight is that water trapped at a position depends on the minimum of the maximum heights on the left and right, minus the current height.

The algorithm uses two pointers starting from both ends. It maintains leftMax and rightMax to track the maximum heights encountered so far from each side. At each step, it moves the pointer with the smaller maximum height (because water level is limited by the smaller side). For the current position, if the height is less than the corresponding max, water can be trapped (max - height). Otherwise, we update the max.

This approach works because we only need to know the limiting height (the smaller of left and right maxes) to calculate trapped water. By moving the pointer with the smaller max, we ensure we've found the correct limiting height for that position.

Edge cases handled include: all bars same height (no water trapped), bars in increasing order (no water trapped), bars in decreasing order (no water trapped), and bars forming a basin (water trapped in the middle).

An alternative O(n) approach would precompute left and right maxes, but the two-pointer approach achieves the same with O(1) space. The time complexity is O(n) where n is the number of bars, and space complexity is O(1) as we only use a few variables.

Think of this as filling a container: the water level at any point is determined by the lower of the two "walls" on either side, and you calculate how much water fits by moving inward from both sides, always focusing on the side with the lower wall.`,
	},
	{
		ID:          182,
		Title:       "Merge k Sorted Lists",
		Description: "You are given an array of k linked-lists lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.",
		Difficulty:  "Hard",
		Topic:       "Linked Lists",
		Signature:   "func mergeKLists(lists []*ListNode) *ListNode",
		TestCases: []TestCase{
			{Input: "lists = [[1,4,5],[1,3,4],[2,6]]", Expected: "[1 1 2 3 4 4 5 6]"},
			{Input: "lists = []", Expected: "[]"},
			{Input: "lists = [[]]", Expected: "[]"},
			{Input: "lists = [[1],[2],[3]]", Expected: "[1 2 3]"},
			{Input: "lists = [[1,2,3],[4,5,6]]", Expected: "[1 2 3 4 5 6]"},
		},
		Solution: `func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) == 0 {
		return nil
	}
	for len(lists) > 1 {
		merged := []*ListNode{}
		for i := 0; i < len(lists); i += 2 {
			l1 := lists[i]
			var l2 *ListNode
			if i+1 < len(lists) {
				l2 = lists[i+1]
			}
			merged = append(merged, mergeTwoLists(l1, l2))
		}
		lists = merged
	}
	return lists[0]
}

func mergeTwoLists(l1, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
	current := dummy
	for l1 != nil && l2 != nil {
		if l1.Val < l2.Val {
			current.Next = l1
			l1 = l1.Next
		} else {
			current.Next = l2
			l2 = l2.Next
		}
		current = current.Next
	}
	if l1 != nil {
		current.Next = l1
	} else {
		current.Next = l2
	}
	return dummy.Next
}`,
		PythonSolution: `def mergeKLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    if not lists:
        return None
    while len(lists) > 1:
        merged = []
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i + 1] if i + 1 < len(lists) else None
            merged.append(merge_two_lists(l1, l2))
        lists = merged
    return lists[0]

def merge_two_lists(l1, l2):
    dummy = ListNode()
    current = dummy
    while l1 and l2:
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    current.next = l1 or l2
    return dummy.next`,
		Explanation: `Merging k sorted linked lists efficiently requires avoiding the O(k²) approach of merging lists one by one. The divide-and-conquer approach reduces this to O(k log k) by merging lists in pairs iteratively.

The algorithm repeatedly merges pairs of lists until only one list remains. In each iteration, it pairs up consecutive lists and merges each pair using the standard two-list merge operation. After each iteration, the number of lists is halved. This continues until a single merged list remains.

This iterative approach is space-efficient compared to recursive divide-and-conquer, as it doesn't use a recursion stack. Each list is merged log k times (once per iteration level), and each merge processes the list's elements.

Edge cases handled include: empty lists array (return nil), single list (return that list), lists of varying lengths (merge handles correctly), and lists with overlapping value ranges (merge maintains sorted order).

An alternative approach would use a min-heap to always merge the two smallest lists, but that's more complex. Another approach would merge lists sequentially, but that's O(k²) time. The iterative divide-and-conquer approach achieves optimal O(n log k) time where n is total nodes and k is number of lists, with O(1) extra space.

Think of this as a tournament bracket: you pair up lists and merge them, then pair up the results and merge those, continuing until you have one winner (the fully merged list).`,
	},
	{
		ID:          183,
		Title:       "Serialize and Deserialize Binary Tree",
		Description: "Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.",
		Difficulty:  "Hard",
		Topic:       "Trees",
		Signature:   "type Codec struct{}\nfunc Constructor() Codec\nfunc (this *Codec) serialize(root *TreeNode) string\nfunc (this *Codec) deserialize(data string) *TreeNode",
		TestCases: []TestCase{
			{Input: "root = [1,2,3,null,null,4,5]", Expected: "[1 2 3 null null 4 5]"},
			{Input: "root = []", Expected: "[]"},
			{Input: "root = [1]", Expected: "[1]"},
			{Input: "root = [1,2]", Expected: "[1 2]"},
		},
		Solution: `import (
	"strconv"
	"strings"
)

type Codec struct{}

func Constructor() Codec {
	return Codec{}
}

func (this *Codec) serialize(root *TreeNode) string {
	if root == nil {
		return "null"
	}
	return strconv.Itoa(root.Val) + "," + this.serialize(root.Left) + "," + this.serialize(root.Right)
}

func (this *Codec) deserialize(data string) *TreeNode {
	values := strings.Split(data, ",")
	index := 0
	return this.deserializeHelper(&index, values)
}

func (this *Codec) deserializeHelper(index *int, values []string) *TreeNode {
	if *index >= len(values) || values[*index] == "null" {
		(*index)++
		return nil
	}
	val, _ := strconv.Atoi(values[*index])
	(*index)++
	node := &TreeNode{Val: val}
	node.Left = this.deserializeHelper(index, values)
	node.Right = this.deserializeHelper(index, values)
	return node
}`,
		PythonSolution: `class Codec:
    def serialize(self, root):
        if not root:
            return "null"
        return str(root.val) + "," + self.serialize(root.left) + "," + self.serialize(root.right)
    
    def deserialize(self, data):
        values = data.split(",")
        self.index = 0
        return self.deserialize_helper(values)
    
    def deserialize_helper(self, values):
        if self.index >= len(values) or values[self.index] == "null":
            self.index += 1
            return None
        val = int(values[self.index])
        self.index += 1
        node = TreeNode(val)
        node.left = self.deserialize_helper(values)
        node.right = self.deserialize_helper(values)
        return node`,
		Explanation: `Serializing and deserializing a binary tree requires a format that preserves both the tree structure and node values. Preorder traversal with null markers provides an unambiguous representation that can be reconstructed.

The serialize function performs a preorder traversal (root, left, right), outputting node values as comma-separated values and "null" for nil nodes. This creates a string representation where the tree structure is encoded in the order of values. The deserialize function reads this string and reconstructs the tree using the same preorder logic: it reads a value, creates a node, recursively builds left subtree, then recursively builds right subtree.

The key insight is that preorder traversal with null markers uniquely determines the tree structure. When deserializing, encountering "null" tells us to stop building that branch, and the order ensures we build the correct structure. An index counter tracks the current position in the serialized string.

Edge cases handled include: empty tree (serializes to "null"), single-node tree (serializes correctly), skewed trees (handled by null markers), and trees with duplicate values (structure preserved by order).

An alternative approach would use level-order traversal, but preorder is simpler. Another approach would use a more compact encoding, but the current approach is clear and efficient. The time complexity is O(n) for both serialize and deserialize where n is the number of nodes. The space complexity is O(n) for the serialized string and recursion stack.

Think of this as writing down a tree by listing nodes in a specific order with markers for "no child here", then reading it back and reconstructing the tree by following the same order and respecting the "no child" markers.`,
	},
	{
		ID:          184,
		Title:       "Word Ladder",
		Description: "A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that: Every adjacent pair of words differs by a single letter, and every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList. Return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.",
		Difficulty:  "Hard",
		Topic:       "Graphs",
		Signature:   "func ladderLength(beginWord string, endWord string, wordList []string) int",
		TestCases: []TestCase{
			{Input: "beginWord = \"hit\", endWord = \"cog\", wordList = [\"hot\",\"dot\",\"dog\",\"lot\",\"log\",\"cog\"]", Expected: "5"},
			{Input: "beginWord = \"hit\", endWord = \"cog\", wordList = [\"hot\",\"dot\",\"dog\",\"lot\",\"log\"]", Expected: "0"},
			{Input: "beginWord = \"a\", endWord = \"c\", wordList = [\"a\",\"b\",\"c\"]", Expected: "2"},
			{Input: "beginWord = \"hot\", endWord = \"dog\", wordList = [\"hot\",\"dog\"]", Expected: "0"},
		},
		Solution: `func ladderLength(beginWord string, endWord string, wordList []string) int {
	wordSet := make(map[string]bool)
	for _, word := range wordList {
		wordSet[word] = true
	}
	if !wordSet[endWord] {
		return 0
	}
	queue := []string{beginWord}
	level := 1
	visited := make(map[string]bool)
	visited[beginWord] = true
	for len(queue) > 0 {
		size := len(queue)
		for i := 0; i < size; i++ {
			word := queue[0]
			queue = queue[1:]
			if word == endWord {
				return level
			}
			chars := []byte(word)
			for j := 0; j < len(chars); j++ {
				original := chars[j]
				for c := 'a'; c <= 'z'; c++ {
					if byte(c) == original {
						continue
					}
					chars[j] = byte(c)
					nextWord := string(chars)
					if wordSet[nextWord] && !visited[nextWord] {
						visited[nextWord] = true
						queue = append(queue, nextWord)
					}
				}
				chars[j] = original
			}
		}
		level++
	}
	return 0
}`,
		PythonSolution: `def ladderLength(beginWord: str, endWord: str, wordList: List[str]) -> int:
    word_set = set(wordList)
    if endWord not in word_set:
        return 0
    queue = [beginWord]
    level = 1
    visited = {beginWord}
    while queue:
        size = len(queue)
        for _ in range(size):
            word = queue.pop(0)
            if word == endWord:
                return level
            chars = list(word)
            for i in range(len(chars)):
                original = chars[i]
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == original:
                        continue
                    chars[i] = c
                    next_word = ''.join(chars)
                    if next_word in word_set and next_word not in visited:
                        visited.add(next_word)
                        queue.append(next_word)
                chars[i] = original
        level += 1
    return 0`,
		Explanation: `Finding the shortest transformation sequence between two words is a classic BFS problem. The key insight is to model word transformations as a graph where edges connect words that differ by one letter, then find the shortest path using BFS.

The algorithm uses BFS starting from beginWord. At each level, it generates all possible words that differ by one letter from words at the current level. Each generated word is checked against the dictionary, and if it's valid and unvisited, it's added to the queue for the next level. The level number represents the transformation count.

BFS guarantees the shortest path because it explores all possibilities at distance 1 before moving to distance 2, etc. The visited set prevents revisiting words, ensuring we don't get stuck in cycles or take longer paths.

Edge cases handled include: endWord not in dictionary (return 0), beginWord equals endWord (return 1), no transformation possible (return 0), and words with single-letter differences (found efficiently by BFS).

An alternative approach would use bidirectional BFS for better performance, but standard BFS is simpler. The time complexity is O(M²*N) where M is word length and N is dictionary size, as we generate M*26 variations for each of N words. The space complexity is O(M²*N) for the queue and visited set.

Think of this as finding the shortest path in a word network: you start from one word and explore all words one letter-change away, then two letter-changes away, continuing until you reach the target word.`,
	},
	{
		ID:          185,
		Title:       "Find Median from Data Stream",
		Description: "The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values. Implement the MedianFinder class: MedianFinder() initializes the MedianFinder object. void addNum(int num) adds the integer num from the data stream to the data structure. double findMedian() returns the median of all elements so far.",
		Difficulty:  "Hard",
		Topic:       "Heaps",
		Signature:   "type MedianFinder struct{}\nfunc Constructor() MedianFinder\nfunc (this *MedianFinder) AddNum(num int)\nfunc (this *MedianFinder) FindMedian() float64",
		TestCases: []TestCase{
			{Input: "MedianFinder, addNum(1), addNum(2), findMedian(), addNum(3), findMedian()", Expected: "1.5, 2.0"},
			{Input: "MedianFinder, addNum(1), addNum(2), addNum(3), addNum(4), findMedian()", Expected: "2.5"},
			{Input: "MedianFinder, addNum(5), findMedian()", Expected: "5.0"},
		},
		Solution: `import "container/heap"

type IntHeap []int

func (h IntHeap) Len() int           { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h IntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *IntHeap) Push(x interface{}) { *h = append(*h, x.(int)) }
func (h *IntHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

type MedianFinder struct {
	small *IntHeap // max heap (negate for min heap behavior)
	large *IntHeap // min heap
}

func Constructor() MedianFinder {
	small := &IntHeap{}
	large := &IntHeap{}
	heap.Init(small)
	heap.Init(large)
	return MedianFinder{small: small, large: large}
}

func (this *MedianFinder) AddNum(num int) {
	if this.small.Len() == 0 || num <= -(*this.small)[0] {
		heap.Push(this.small, -num)
	} else {
		heap.Push(this.large, num)
	}
	if this.small.Len() > this.large.Len()+1 {
		val := heap.Pop(this.small).(int)
		heap.Push(this.large, -val)
	} else if this.large.Len() > this.small.Len() {
		val := heap.Pop(this.large).(int)
		heap.Push(this.small, -val)
	}
}

func (this *MedianFinder) FindMedian() float64 {
	if this.small.Len() > this.large.Len() {
		return float64(-(*this.small)[0])
	}
	return float64(-(*this.small)[0]+(*this.large)[0]) / 2.0
}`,
		PythonSolution: `import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # max heap (negate values)
        self.large = []  # min heap
    
    def addNum(self, num: int) -> None:
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
        else:
            heapq.heappush(self.large, num)
        if len(self.small) > len(self.large) + 1:
            val = heapq.heappop(self.small)
            heapq.heappush(self.large, -val)
        elif len(self.large) > len(self.small):
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)
    
    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return float(-self.small[0])
        return (-self.small[0] + self.large[0]) / 2.0`,
		Explanation: `Finding the median from a data stream requires maintaining the data in a way that allows efficient median access. The two-heap approach elegantly solves this by keeping the smaller half in a max-heap and the larger half in a min-heap.

The algorithm maintains two heaps: a max-heap for the smaller half of numbers and a min-heap for the larger half. When adding a number, if the heaps are balanced, we add to the appropriate heap based on the number's value. If they become unbalanced (differ by more than 1), we rebalance by moving the top element from the larger heap to the smaller heap.

The median is the top of the max-heap if there are an odd number of elements, or the average of both heap tops if there are an even number. This approach ensures O(log n) insertion and O(1) median access.

Edge cases handled include: empty stream (handled by heap checks), single element (median is that element), all elements same (median is that value), and elements in sorted order (heaps maintain balance).

An alternative approach would maintain a sorted array, but that's O(n) insertion. Another approach would use a balanced BST, but heaps are simpler. The two-heap approach achieves optimal O(log n) insertion and O(1) median access with O(n) space.

Think of this as splitting numbers into two groups: the lower half and the upper half. You keep track of the maximum of the lower half and the minimum of the upper half, and the median is between (or at) these values.`,
	},
	{
		ID:          186,
		Title:       "N-Queens",
		Description: "The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other. Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.",
		Difficulty:  "Hard",
		Topic:       "Backtracking",
		Signature:   "func solveNQueens(n int) [][]string",
		TestCases: []TestCase{
			{Input: "n = 4", Expected: "[[\".Q..\",\"...Q\",\"Q...\",\"..Q.\"],[\"..Q.\",\"Q...\",\"...Q\",\".Q..\"]]"},
			{Input: "n = 1", Expected: "[[\"Q\"]]"},
			{Input: "n = 2", Expected: "[]"},
			{Input: "n = 3", Expected: "[]"},
		},
		Solution: `func solveNQueens(n int) [][]string {
	result := [][]string{}
	board := make([][]byte, n)
	for i := range board {
		board[i] = make([]byte, n)
		for j := range board[i] {
			board[i][j] = '.'
		}
	}
	cols := make(map[int]bool)
	diag1 := make(map[int]bool) // row - col
	diag2 := make(map[int]bool) // row + col
	var backtrack func(row int)
	backtrack = func(row int) {
		if row == n {
			solution := make([]string, n)
			for i := 0; i < n; i++ {
				solution[i] = string(board[i])
			}
			result = append(result, solution)
			return
		}
		for col := 0; col < n; col++ {
			if cols[col] || diag1[row-col] || diag2[row+col] {
				continue
			}
			board[row][col] = 'Q'
			cols[col] = true
			diag1[row-col] = true
			diag2[row+col] = true
			backtrack(row + 1)
			board[row][col] = '.'
			delete(cols, col)
			delete(diag1, row-col)
			delete(diag2, row+col)
		}
	}
	backtrack(0)
	return result
}`,
		PythonSolution: `def solveNQueens(n: int) -> List[List[str]]:
    result = []
    board = [['.'] * n for _ in range(n)]
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col
    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            backtrack(row + 1)
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
    backtrack(0)
    return result`,
		Explanation: `The N-Queens problem requires placing n queens on an n×n chessboard such that no two queens attack each other. Backtracking is the natural approach, as we try placing queens and undo placements when conflicts are detected.

The algorithm places queens row by row. For each row, it tries each column position. Before placing a queen, it checks if that position is safe by verifying no queen exists in the same column or on either diagonal. Two sets track occupied columns and diagonals: one for diagonals going top-left to bottom-right (row - col is constant) and one for diagonals going top-right to bottom-left (row + col is constant).

When a valid position is found, the queen is placed, the position is marked, and we recurse to the next row. If no valid position exists in a row, we backtrack by removing the previous queen and trying the next column. When all n queens are placed, we've found a solution.

Edge cases handled include: n=1 (trivial solution), n=2 or n=3 (no solutions exist), and larger boards (multiple solutions possible, all found by backtracking).

An alternative approach would use bit manipulation for faster conflict checking, but the set-based approach is clearer. The time complexity is O(n!) in the worst case as we explore all possible placements, though pruning makes it much faster in practice. The space complexity is O(n²) for storing the board and O(n) for tracking columns and diagonals.

Think of this as solving a puzzle: you place pieces one by one, and if placing a piece creates a conflict, you remove it and try the next position, continuing until you've placed all pieces successfully.`,
	},
	{
		ID:          187,
		Title:       "Sudoku Solver",
		Description: "Write a program to solve a Sudoku puzzle by filling the empty cells. A sudoku solution must satisfy all of the following rules: Each of the digits 1-9 must occur exactly once in each row, each column, and each of the nine 3x3 sub-boxes of the grid. The '.' character indicates empty cells.",
		Difficulty:  "Hard",
		Topic:       "Backtracking",
		Signature:   "func solveSudoku(board [][]byte)",
		TestCases: []TestCase{
			{Input: "board = [[\"5\",\"3\",\".\",\".\",\"7\",\".\",\".\",\".\",\".\"],[\"6\",\".\",\".\",\"1\",\"9\",\"5\",\".\",\".\",\".\"],[\".\",\"9\",\"8\",\".\",\".\",\".\",\".\",\"6\",\".\"],[\"8\",\".\",\".\",\".\",\"6\",\".\",\".\",\".\",\"3\"],[\"4\",\".\",\".\",\"8\",\".\",\"3\",\".\",\".\",\"1\"],[\"7\",\".\",\".\",\".\",\"2\",\".\",\".\",\".\",\"6\"],[\".\",\"6\",\".\",\".\",\".\",\".\",\"2\",\"8\",\".\"],[\".\",\".\",\".\",\"4\",\"1\",\"9\",\".\",\".\",\"5\"],[\".\",\".\",\".\",\".\",\"8\",\".\",\".\",\"7\",\"9\"]]", Expected: "Solved board"},
		},
		Solution: `func solveSudoku(board [][]byte) {
	solve(board)
}

func solve(board [][]byte) bool {
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if board[i][j] == '.' {
				for c := byte('1'); c <= '9'; c++ {
					if isValid(board, i, j, c) {
						board[i][j] = c
						if solve(board) {
							return true
						}
						board[i][j] = '.'
					}
				}
				return false
			}
		}
	}
	return true
}

func isValid(board [][]byte, row, col int, c byte) bool {
	for i := 0; i < 9; i++ {
		if board[row][i] == c {
			return false
		}
		if board[i][col] == c {
			return false
		}
		if board[3*(row/3)+i/3][3*(col/3)+i%3] == c {
			return false
		}
	}
	return true
}`,
		PythonSolution: `def solveSudoku(board: List[List[str]]) -> None:
    def solve():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for c in '123456789':
                        if is_valid(i, j, c):
                            board[i][j] = c
                            if solve():
                                return True
                            board[i][j] = '.'
                    return False
        return True
    
    def is_valid(row, col, c):
        for i in range(9):
            if board[row][i] == c:
                return False
            if board[i][col] == c:
                return False
            if board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == c:
                return False
        return True
    
    solve()`,
		Explanation: `Solving Sudoku requires systematically trying all possible digit placements and backtracking when constraints are violated. This is a classic backtracking problem that explores the solution space efficiently.

The algorithm processes cells row by row, left to right. For each empty cell, it tries digits 1-9. Before placing a digit, it checks three constraints: the digit must not appear in the current row, current column, or the 3x3 subgrid containing the cell. If a digit is valid, it's placed and the algorithm recurses to the next cell. If no digit works, the algorithm backtracks by removing the placed digit and trying the next option.

The backtracking ensures that when we reach an impossible state (no valid digit for a cell), we undo recent choices and try alternatives. This systematic exploration guarantees finding a solution if one exists, or determining that no solution exists.

Edge cases handled include: boards with no empty cells (already solved), boards with multiple solutions (finds one solution), boards with no solution (backtracks through all possibilities), and boards with many constraints (backtracking prunes invalid branches early).

An alternative approach would use constraint propagation (like arc consistency), but that's more complex. The backtracking approach is straightforward and works well for standard 9x9 Sudoku. The time complexity is O(9^m) in the worst case where m is the number of empty cells, though pruning makes it much faster in practice. The space complexity is O(1) extra space beyond the board itself.

Think of this as solving a puzzle by trying each piece: if a piece fits, you place it and move to the next. If no piece fits, you remove the last piece and try a different one, continuing until the puzzle is complete.`,
	},
	{
		ID:          188,
		Title:       "First Missing Positive",
		Description: "Given an unsorted integer array nums, return the smallest missing positive integer. You must implement an algorithm that runs in O(n) time and uses O(1) extra space.",
		Difficulty:  "Hard",
		Topic:       "Arrays",
		Signature:   "func firstMissingPositive(nums []int) int",
		TestCases: []TestCase{
			{Input: "nums = []int{1,2,0}", Expected: "3"},
			{Input: "nums = []int{3,4,-1,1}", Expected: "2"},
			{Input: "nums = []int{7,8,9,11,12}", Expected: "1"},
			{Input: "nums = []int{1}", Expected: "2"},
			{Input: "nums = []int{1,2,3}", Expected: "4"},
			{Input: "nums = []int{-1,-2,-3}", Expected: "1"},
		},
		Solution: `func firstMissingPositive(nums []int) int {
	n := len(nums)
	for i := 0; i < n; i++ {
		for nums[i] > 0 && nums[i] <= n && nums[nums[i]-1] != nums[i] {
			nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
		}
	}
	for i := 0; i < n; i++ {
		if nums[i] != i+1 {
			return i + 1
		}
	}
	return n + 1
}`,
		PythonSolution: `def firstMissingPositive(nums: List[int]) -> int:
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1`,
		Explanation: `Finding the first missing positive integer requires an approach that handles the constraint of O(1) space. Cyclic sort is perfect for this, as it places each number at its correct index in-place.

The algorithm uses cyclic sort: for each position i, if the number nums[i] is in the valid range [1, n] and not already at its correct position (index nums[i]-1), it swaps nums[i] with the number at its correct position. This continues until the number at position i is either out of range or already in the correct position.

After sorting, we iterate through the array. The first position where nums[i] != i+1 indicates that i+1 is missing. If all positions are correct, then n+1 is missing (all numbers 1 through n are present).

Edge cases handled include: array with all negative numbers (first missing is 1), array with numbers greater than n (ignored, don't affect answer), array with duplicates (handled by swap logic), and array already sorted (quickly identifies missing number).

An alternative approach would use a hash set, but that requires O(n) space. The cyclic sort approach achieves O(n) time with O(1) space. The time complexity is O(n) because each element is placed at most once, and the space complexity is O(1) as we only use a few variables.

Think of this as organizing a numbered list: you put each number in its correct position (number k goes at position k-1), and the first position that doesn't have the expected number tells you what's missing.`,
	},
	{
		ID:          189,
		Title:       "Wildcard Matching",
		Description: "Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*' where: '?' Matches any single character. '*' Matches any sequence of characters (including the empty sequence). The matching should cover the entire input string (not partial).",
		Difficulty:  "Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func isMatch(s string, p string) bool",
		TestCases: []TestCase{
			{Input: "s = \"aa\", p = \"a\"", Expected: "false"},
			{Input: "s = \"aa\", p = \"*\"", Expected: "true"},
			{Input: "s = \"cb\", p = \"?a\"", Expected: "false"},
			{Input: "s = \"adceb\", p = \"*a*b\"", Expected: "true"},
			{Input: "s = \"acdcb\", p = \"a*c?b\"", Expected: "false"},
			{Input: "s = \"\", p = \"*\"", Expected: "true"},
		},
		Solution: `func isMatch(s string, p string) bool {
	m, n := len(s), len(p)
	dp := make([][]bool, m+1)
	for i := range dp {
		dp[i] = make([]bool, n+1)
	}
	dp[0][0] = true
	for j := 1; j <= n; j++ {
		if p[j-1] == '*' {
			dp[0][j] = dp[0][j-1]
		}
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if p[j-1] == '*' {
				dp[i][j] = dp[i][j-1] || dp[i-1][j]
			} else if p[j-1] == '?' || s[i-1] == p[j-1] {
				dp[i][j] = dp[i-1][j-1]
			}
		}
	}
	return dp[m][n]
}`,
		PythonSolution: `def isMatch(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif p[j - 1] == '?' or s[i - 1] == p[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]`,
		Explanation: `Wildcard matching requires determining if a string matches a pattern containing '?' (matches any single character) and '*' (matches any sequence of characters). Dynamic programming efficiently handles the complex matching rules.

The algorithm uses a 2D DP table where dp[i][j] represents whether the first i characters of the string match the first j characters of the pattern. The base cases handle empty string and empty pattern. For each cell, if the pattern character is '*', it can match zero characters (dp[i][j-1]) or one or more characters (dp[i-1][j]). If it's '?' or matches the string character, we check if the previous characters matched (dp[i-1][j-1]).

The '*' case is the most complex: it can match nothing (skip it) or match one or more characters (consume a character from the string and keep the '*' in the pattern for potential further matches). This greedy approach ensures we check all possibilities.

Edge cases handled include: pattern with multiple '*' (handled by DP transitions), string shorter than pattern without '*' (cannot match), pattern with only '*' (matches everything), and empty string with non-empty pattern (only matches if pattern is all '*').

An alternative recursive approach with memoization would work but is less efficient. The DP approach achieves O(m*n) time where m and n are string and pattern lengths, and O(m*n) space for the DP table (can be optimized to O(n)).

Think of this as building up matches character by character: for each position, you check if the string and pattern match up to that point, handling '*' by checking if it can match zero or more characters.`,
	},
	{
		ID:          190,
		Title:       "Regular Expression Matching",
		Description: "Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where: '.' Matches any single character. '*' Matches zero or more of the preceding element. The matching should cover the entire input string (not partial).",
		Difficulty:  "Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func isMatch(s string, p string) bool",
		TestCases: []TestCase{
			{Input: "s = \"aa\", p = \"a\"", Expected: "false"},
			{Input: "s = \"aa\", p = \"a*\"", Expected: "true"},
			{Input: "s = \"ab\", p = \".*\"", Expected: "true"},
			{Input: "s = \"aab\", p = \"c*a*b\"", Expected: "true"},
			{Input: "s = \"mississippi\", p = \"mis*is*p*.\"", Expected: "false"},
			{Input: "s = \"\", p = \"a*\"", Expected: "true"},
		},
		Solution: `func isMatch(s string, p string) bool {
	m, n := len(s), len(p)
	dp := make([][]bool, m+1)
	for i := range dp {
		dp[i] = make([]bool, n+1)
	}
	dp[0][0] = true
	for j := 2; j <= n; j++ {
		if p[j-1] == '*' {
			dp[0][j] = dp[0][j-2]
		}
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if p[j-1] == '*' {
				dp[i][j] = dp[i][j-2] || (dp[i-1][j] && (s[i-1] == p[j-2] || p[j-2] == '.'))
			} else if p[j-1] == '.' || s[i-1] == p[j-1] {
				dp[i][j] = dp[i-1][j-1]
			}
		}
	}
	return dp[m][n]
}`,
		PythonSolution: `def isMatch(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] and (s[i - 1] == p[j - 2] or p[j - 2] == '.'))
            elif p[j - 1] == '.' or s[i - 1] == p[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]`,
		Explanation: `Regular expression matching is more complex than wildcard matching because '*' applies to the preceding character, not independently. This requires careful handling of the relationship between '*' and its preceding character.

The algorithm uses dynamic programming where dp[i][j] represents whether the first i characters of the string match the first j characters of the pattern. When we encounter '*', we have two cases: match zero times (dp[i][j-2], skipping the character-'*' pair) or match one or more times (dp[i-1][j], consuming a character from the string if it matches the preceding pattern character).

The key difference from wildcard matching is that '*' is tied to its preceding character. We must check if the current string character matches the character before '*' (or if that character is '.', which matches anything). This creates a dependency that makes the DP transitions more complex.

Edge cases handled include: patterns like "a*" matching empty string (zero matches), patterns like ".*" matching any string (greedy matching), strings longer than pattern without sufficient '*' (cannot match), and patterns with multiple '*' sequences (handled recursively by DP).

An alternative approach would use backtracking, but DP is more efficient. The time complexity is O(m*n) where m and n are string and pattern lengths, and space complexity is O(m*n) for the DP table (can be optimized to O(n)).

Think of this as matching with a constraint: '*' can repeat its preceding character zero or more times, so you check both skipping the repetition entirely and using it to match one more character.`,
	},
	{
		ID:          191,
		Title:       "Four Number Sum",
		Description: "Write a function that takes in a non-empty array of distinct integers and an integer representing a target sum. The function should find all quadruplets in the array that sum up to the target sum and return a two-dimensional array of all these quadruplets in no particular order. If no four numbers sum up to the target sum, the function should return an empty array.",
		Difficulty:  "Hard",
		Topic:       "Arrays",
		Signature:   "func fourNumberSum(array []int, targetSum int) [][]int",
		TestCases: []TestCase{
			{Input: "array = [7, 6, 4, -1, 1, 2], targetSum = 16", Expected: "[[7, 6, 4, -1], [7, 6, 1, 2]]"},
			{Input: "array = [1, 2, 3, 4, 5, 6, 7], targetSum = 10", Expected: "[[1, 2, 3, 4]]"},
		},
		Solution: `func fourNumberSum(array []int, targetSum int) [][]int {
	allPairSums := make(map[int][][]int)
	quadruplets := [][]int{}
	for i := 1; i < len(array)-1; i++ {
		for j := i + 1; j < len(array); j++ {
			currentSum := array[i] + array[j]
			difference := targetSum - currentSum
			if pairs, found := allPairSums[difference]; found {
				for _, pair := range pairs {
					quadruplets = append(quadruplets, append(pair, array[i], array[j]))
				}
			}
		}
		for k := 0; k < i; k++ {
			currentSum := array[i] + array[k]
			if _, found := allPairSums[currentSum]; !found {
				allPairSums[currentSum] = [][]int{}
			}
			allPairSums[currentSum] = append(allPairSums[currentSum], []int{array[k], array[i]})
		}
	}
	return quadruplets
}`,
		PythonSolution: `def fourNumberSum(array: List[int], targetSum: int) -> List[List[int]]:
    all_pair_sums = {}
    quadruplets = []
    for i in range(1, len(array) - 1):
        for j in range(i + 1, len(array)):
            current_sum = array[i] + array[j]
            difference = targetSum - current_sum
            if difference in all_pair_sums:
                for pair in all_pair_sums[difference]:
                    quadruplets.append(pair + [array[i], array[j]])
        for k in range(i):
            current_sum = array[i] + array[k]
            if current_sum not in all_pair_sums:
                all_pair_sums[current_sum] = []
            all_pair_sums[current_sum].append([array[k], array[i]])
    return quadruplets`,
		Explanation: `Finding four numbers that sum to a target requires an efficient approach to avoid the O(n⁴) brute force solution. The two-sum technique extended to four numbers provides an optimal solution.

The algorithm uses a hash map to store pair sums. It iterates through pairs of numbers, and for each pair, it checks if the complement (target - pairSum) exists in the hash map. If it does, it combines the current pair with pairs from the map to form quadruplets. Importantly, pairs are added to the map after checking, ensuring we don't use the same element twice.

The key insight is that we can reduce the four-sum problem to multiple two-sum problems by precomputing pair sums. By checking complements before adding pairs, we avoid duplicate quadruplets and ensure each element is used at most once per quadruplet.

Edge cases handled include: no valid quadruplets (returns empty list), multiple valid quadruplets (all found), quadruplets with duplicate values (handled correctly), and arrays with fewer than four elements (returns empty list).

An alternative O(n³) approach would fix two numbers and use two pointers for the remaining two, but the hash map approach is more flexible. The time complexity is O(n²) on average where n is array length, though worst case can be O(n³) if many pairs have the same sum. The space complexity is O(n²) for storing pair sums.

Think of this as finding two pairs that sum to the target: you precompute all pair sums, then for each new pair, you check if you've seen a complementary pair that would complete the quadruplet.`,
	},
	{
		ID:          192,
		Title:       "Subarray Sort",
		Description: "Write a function that takes in an array of at least two integers and returns an array of the starting and ending indices of the smallest subarray in the input array that needs to be sorted in place in order for the entire input array to be sorted (in ascending order). If the input array is already sorted, the function should return [-1, -1].",
		Difficulty:  "Hard",
		Topic:       "Arrays",
		Signature:   "func subarraySort(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = [1, 2, 4, 7, 10, 11, 7, 12, 6, 7, 16, 18, 19]", Expected: "[3 9]"},
			{Input: "array = [1, 2, 4, 7, 10, 11, 7, 12, 7, 7, 16, 18, 19]", Expected: "[4 9]"},
		},
		Solution: `func subarraySort(array []int) []int {
	minOutOfOrder := math.MaxInt32
	maxOutOfOrder := math.MinInt32
	for i := 0; i < len(array); i++ {
		num := array[i]
		if isOutOfOrder(i, num, array) {
			minOutOfOrder = min(minOutOfOrder, num)
			maxOutOfOrder = max(maxOutOfOrder, num)
		}
	}
	if minOutOfOrder == math.MaxInt32 {
		return []int{-1, -1}
	}
	subarrayLeftIdx := 0
	for minOutOfOrder >= array[subarrayLeftIdx] {
		subarrayLeftIdx++
	}
	subarrayRightIdx := len(array) - 1
	for maxOutOfOrder <= array[subarrayRightIdx] {
		subarrayRightIdx--
	}
	return []int{subarrayLeftIdx, subarrayRightIdx}
}

func isOutOfOrder(i int, num int, array []int) bool {
	if i == 0 {
		return num > array[i+1]
	}
	if i == len(array)-1 {
		return num < array[i-1]
	}
	return num > array[i+1] || num < array[i-1]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def subarraySort(array: List[int]) -> List[int]:
    min_out_of_order = float('inf')
    max_out_of_order = float('-inf')
    for i in range(len(array)):
        num = array[i]
        if is_out_of_order(i, num, array):
            min_out_of_order = min(min_out_of_order, num)
            max_out_of_order = max(max_out_of_order, num)
    if min_out_of_order == float('inf'):
        return [-1, -1]
    subarray_left_idx = 0
    while min_out_of_order >= array[subarray_left_idx]:
        subarray_left_idx += 1
    subarray_right_idx = len(array) - 1
    while max_out_of_order <= array[subarray_right_idx]:
        subarray_right_idx -= 1
    return [subarray_left_idx, subarray_right_idx]

def is_out_of_order(i, num, array):
    if i == 0:
        return num > array[i + 1]
    if i == len(array) - 1:
        return num < array[i - 1]
    return num > array[i + 1] or num < array[i - 1]`,
		Explanation: `Finding the smallest subarray that needs to be sorted requires identifying elements that are out of order and determining the range they should occupy. The key insight is that out-of-order elements affect the positions of other elements.

The algorithm first finds the minimum and maximum out-of-order elements by scanning the array. An element is out of order if it's smaller than its left neighbor or larger than its right neighbor. Then it finds the correct positions for these elements: the minimum out-of-order element should be placed where all elements to its left are smaller, and the maximum should be placed where all elements to its right are larger.

This approach works because if we sort the subarray from the leftmost incorrect position to the rightmost incorrect position, the entire array will be sorted. The out-of-order elements define the boundaries of the subarray that needs sorting.

Edge cases handled include: already sorted array (returns [0, 0] or similar sentinel), array in reverse order (entire array needs sorting), single out-of-order element (finds correct range), and multiple out-of-order regions (finds the encompassing range).

An alternative O(n log n) approach would sort a copy and compare, but the current approach is optimal. The time complexity is O(n) where n is array length, as we make two passes: one to find out-of-order elements and one to find their correct positions. The space complexity is O(1) as we only track a few variables.

Think of this as finding the boundaries of disorder: you identify elements that are in the wrong place, then find where they should be, and the range between these positions is what needs to be sorted.`,
	},
	{
		ID:          193,
		Title:       "Largest Range",
		Description: "Write a function that takes in an array of integers and returns an array of length 2 representing the largest range of integers contained in that array. The first number in the output array should be the first number in the range, while the second number should be the last number in the range. A range of numbers is defined as a set of numbers that come right after each other in the set of real integers. For instance, the output array [2, 6] represents the range [2, 3, 4, 5, 6], which is a range of length 5. Note that numbers don't need to be sorted or adjacent in the input array in order to form a range.",
		Difficulty:  "Hard",
		Topic:       "Arrays",
		Signature:   "func largestRange(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = [1, 11, 3, 0, 15, 5, 2, 4, 10, 7, 12, 6]", Expected: "[0 7]"},
			{Input: "array = [1]", Expected: "[1 1]"},
		},
		Solution: `func largestRange(array []int) []int {
	bestRange := []int{}
	longestLength := 0
	nums := make(map[int]bool)
	for _, num := range array {
		nums[num] = true
	}
	for _, num := range array {
		if !nums[num] {
			continue
		}
		nums[num] = false
		currentLength := 1
		left := num - 1
		right := num + 1
		for nums[left] {
			nums[left] = false
			currentLength++
			left--
		}
		for nums[right] {
			nums[right] = false
			currentLength++
			right++
		}
		if currentLength > longestLength {
			longestLength = currentLength
			bestRange = []int{left + 1, right - 1}
		}
	}
	return bestRange
}`,
		PythonSolution: `def largestRange(array: List[int]) -> List[int]:
    best_range = []
    longest_length = 0
    nums = {num: True for num in array}
    for num in array:
        if not nums.get(num, False):
            continue
        nums[num] = False
        current_length = 1
        left = num - 1
        right = num + 1
        while nums.get(left, False):
            nums[left] = False
            current_length += 1
            left -= 1
        while nums.get(right, False):
            nums[right] = False
            current_length += 1
            right += 1
        if current_length > longest_length:
            longest_length = current_length
            best_range = [left + 1, right - 1]
    return best_range`,
		Explanation: `Finding the largest range of consecutive integers requires efficiently identifying contiguous sequences. The hash set approach allows O(1) lookups while avoiding sorting.

The algorithm first stores all numbers in a hash set for O(1) lookups. Then, for each unvisited number, it expands left and right to find all consecutive numbers. As it expands, it marks numbers as visited to avoid processing the same range multiple times. It tracks the longest range found.

The key insight is that we only need to start expanding from numbers that haven't been visited yet, meaning they're the start of a new range. Once we've expanded from a number, all numbers in that range are marked as visited, so we won't process them again.

Edge cases handled include: array with no consecutive numbers (each number is its own range), array with all consecutive numbers (entire array is the range), array with duplicates (handled by set), and array with negative numbers (handled correctly).

An alternative O(n log n) approach would sort the array, but the hash set approach achieves O(n) time where n is array length, as we visit each number at most twice (once to check, once when expanding). The space complexity is O(n) for the hash set.

Think of this as finding the longest chain: you look for unvisited starting points, then expand in both directions to see how long the chain is, marking everything you've seen so you don't start from the middle of a chain.`,
	},
	{
		ID:          194,
		Title:       "Min Rewards",
		Description: "Imagine that you're a teacher who's just graded the final exam in a class. You have a list of student scores on the final exam in a particular order (not necessarily sorted), and you want to reward your students. You decide to do so fairly by giving them arbitrary rewards following two rules: All students must receive at least one reward. Any given student must receive strictly more rewards than an adjacent student (a student immediately to the left or to the right) with a lower score and must receive strictly fewer rewards than an adjacent student with a higher score. Write a function that takes in a list of scores and returns the minimum number of rewards that you must give out to all students to satisfy the two rules.",
		Difficulty:  "Hard",
		Topic:       "Arrays",
		Signature:   "func minRewards(scores []int) int",
		TestCases: []TestCase{
			{Input: "scores = [8, 4, 2, 1, 3, 6, 7, 9, 5]", Expected: "25"},
			{Input: "scores = [1]", Expected: "1"},
		},
		Solution: `func minRewards(scores []int) int {
	rewards := make([]int, len(scores))
	for i := range rewards {
		rewards[i] = 1
	}
	for i := 1; i < len(scores); i++ {
		if scores[i] > scores[i-1] {
			rewards[i] = rewards[i-1] + 1
		}
	}
	for i := len(scores) - 2; i >= 0; i-- {
		if scores[i] > scores[i+1] {
			rewards[i] = max(rewards[i], rewards[i+1]+1)
		}
	}
	sum := 0
	for _, reward := range rewards {
		sum += reward
	}
	return sum
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def minRewards(scores: List[int]) -> int:
    rewards = [1] * len(scores)
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1]:
            rewards[i] = rewards[i - 1] + 1
    for i in range(len(scores) - 2, -1, -1):
        if scores[i] > scores[i + 1]:
            rewards[i] = max(rewards[i], rewards[i + 1] + 1)
    return sum(rewards)`,
		Explanation: `Distributing minimum rewards to students based on their scores relative to neighbors requires a two-pass approach. The key insight is that rewards must satisfy constraints from both directions: students with higher scores than left neighbors need more rewards, and students with higher scores than right neighbors need more rewards.

The algorithm first initializes all rewards to 1 (minimum requirement). In the first pass (left to right), if a student's score is higher than the previous student, their reward is set to the previous student's reward plus 1. In the second pass (right to left), if a student's score is higher than the next student, their reward is set to the maximum of their current reward and the next student's reward plus 1.

This two-pass approach ensures all constraints are satisfied: the first pass handles left-to-right constraints, and the second pass handles right-to-left constraints while preserving left-to-right constraints by taking the maximum.

Edge cases handled include: scores in increasing order (rewards 1, 2, 3, ...), scores in decreasing order (rewards n, n-1, ..., 1), scores with local minima (handled by two passes), and all scores equal (all get 1 reward).

An alternative approach would use a more complex single-pass algorithm, but the two-pass approach is simpler and clearer. The time complexity is O(n) where n is the number of students, as we make two passes through the array. The space complexity is O(n) for storing rewards.

Think of this as giving out rewards fairly: you first ensure students with better scores than their left neighbor get more, then ensure students with better scores than their right neighbor get more, taking the maximum when both constraints apply.`,
	},
	{
		ID:          195,
		Title:       "Zigzag Traverse",
		Description: "Write a function that takes in an n x m two-dimensional array (that can be square-shaped when n == m) and returns a one-dimensional array of all the array's elements in zigzag order. Zigzag order starts at the top left corner of the two-dimensional array, goes down by one element, and proceeds in a zigzag pattern all the way to the bottom right corner.",
		Difficulty:  "Hard",
		Topic:       "Arrays",
		Signature:   "func zigzagTraverse(array [][]int) []int",
		TestCases: []TestCase{
			{Input: "array = [[1, 3, 4, 10], [2, 5, 9, 11], [6, 8, 12, 15], [7, 13, 14, 16]]", Expected: "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]"},
		},
		Solution: `func zigzagTraverse(array [][]int) []int {
	height := len(array) - 1
	width := len(array[0]) - 1
	result := []int{}
	row, col := 0, 0
	goingDown := true
	for !isOutOfBounds(row, col, height, width) {
		result = append(result, array[row][col])
		if goingDown {
			if col == 0 || row == height {
				goingDown = false
				if row == height {
					col++
				} else {
					row++
				}
			} else {
				row++
				col--
			}
		} else {
			if row == 0 || col == width {
				goingDown = true
				if col == width {
					row++
				} else {
					col++
				}
			} else {
				row--
				col++
			}
		}
	}
	return result
}

func isOutOfBounds(row int, col int, height int, width int) bool {
	return row < 0 || row > height || col < 0 || col > width
}`,
		PythonSolution: `def zigzagTraverse(array: List[List[int]]) -> List[int]:
    height = len(array) - 1
    width = len(array[0]) - 1
    result = []
    row, col = 0, 0
    going_down = True
    while not is_out_of_bounds(row, col, height, width):
        result.append(array[row][col])
        if going_down:
            if col == 0 or row == height:
                going_down = False
                if row == height:
                    col += 1
                else:
                    row += 1
            else:
                row += 1
                col -= 1
        else:
            if row == 0 or col == width:
                going_down = True
                if col == width:
                    row += 1
                else:
                    col += 1
            else:
                row -= 1
                col += 1
    return result

def is_out_of_bounds(row, col, height, width):
    return row < 0 or row > height or col < 0 or col > width`,
		Explanation: `Traversing a 2D array in a zigzag diagonal pattern requires carefully tracking the current direction and handling boundary conditions. The pattern alternates between moving down-left and up-right, changing direction when hitting boundaries.

The algorithm maintains a direction flag (goingDown) to track whether we're moving diagonally down-left or up-right. When moving down-left, we increment row and decrement col. When moving up-right, we decrement row and increment col. At boundaries (top, bottom, left, right edges), we change direction and adjust position accordingly.

The boundary handling is crucial: when hitting the top or right edge while going up-right, we move right or down respectively. When hitting the bottom or left edge while going down-left, we move down or right respectively. This ensures we cover all elements without going out of bounds.

Edge cases handled include: single-row arrays (move right, change direction at edges), single-column arrays (move down, change direction at edges), square arrays (standard zigzag pattern), and rectangular arrays (pattern adapts to dimensions).

An alternative approach would precompute all diagonal coordinates, but that's less efficient. The current approach processes elements on-the-fly. The time complexity is O(n*m) where n and m are array dimensions, as we visit each element once. The space complexity is O(n*m) for the result array.

Think of this as drawing a zigzag line through a grid: you move diagonally, and when you hit an edge, you "bounce" off and change direction, continuing until you've visited all cells.`,
	},
	{
		ID:          196,
		Title:       "Longest Subarray With Sum",
		Description: "Write a function that takes in a non-empty array of integers and an integer representing a target sum. The function should find the longest subarray where the values sum to the target sum. The function should return an array containing the starting and ending indices of this subarray, both inclusive. If there is no such subarray, return [-1, -1].",
		Difficulty:  "Hard",
		Topic:       "Arrays",
		Signature:   "func longestSubarrayWithSum(array []int, targetSum int) []int",
		TestCases: []TestCase{
			{Input: "array = [1, 2, 3, 4, 3, 3, 1, 2, 1, 2], targetSum = 10", Expected: "[4 8]"},
			{Input: "array = [1, 2, 3], targetSum = 6", Expected: "[0 2]"},
		},
		Solution: `func longestSubarrayWithSum(array []int, targetSum int) []int {
	indices := []int{-1, -1}
	longestLength := 0
	currentSum := 0
	sums := make(map[int]int)
	sums[0] = -1
	for i := 0; i < len(array); i++ {
		currentSum += array[i]
		if _, found := sums[currentSum-targetSum]; found {
			leftIdx := sums[currentSum-targetSum] + 1
			currentLength := i - leftIdx + 1
			if currentLength > longestLength {
				longestLength = currentLength
				indices = []int{leftIdx, i}
			}
		}
		if _, found := sums[currentSum]; !found {
			sums[currentSum] = i
		}
	}
	return indices
}`,
		PythonSolution: `def longestSubarrayWithSum(array: List[int], targetSum: int) -> List[int]:
    indices = [-1, -1]
    longest_length = 0
    current_sum = 0
    sums = {0: -1}
    for i in range(len(array)):
        current_sum += array[i]
        if current_sum - targetSum in sums:
            left_idx = sums[current_sum - targetSum] + 1
            current_length = i - left_idx + 1
            if current_length > longest_length:
                longest_length = current_length
                indices = [left_idx, i]
        if current_sum not in sums:
            sums[current_sum] = i
    return indices`,
		Explanation: `Finding the longest subarray with a target sum efficiently requires using prefix sums and a hash map. The key insight is that if the sum from index 0 to j equals currentSum and the sum from 0 to i equals currentSum - targetSum, then the sum from i+1 to j equals targetSum.

The algorithm maintains a running sum (prefix sum) and a hash map that stores the first occurrence of each prefix sum value. For each position, it calculates the current prefix sum and checks if currentSum - targetSum exists in the map. If it does, we've found a subarray ending at the current position that sums to targetSum. We only store the first occurrence of each sum to ensure we get the longest subarray (earliest start for a given sum).

This approach efficiently finds all valid subarrays in a single pass. By tracking only the first occurrence of each prefix sum, we naturally find the longest subarray for each ending position.

Edge cases handled include: no valid subarray (returns [-1, -1]), entire array sums to target (returns [0, n-1]), multiple valid subarrays (finds longest), and arrays with negative numbers (prefix sum approach handles correctly).

An alternative O(n²) approach would check all possible subarrays, but the prefix sum approach is optimal. The time complexity is O(n) where n is array length, as we make a single pass. The space complexity is O(n) for the hash map.

Think of this as tracking cumulative totals: you keep a running total, and when you see that you've had a total before that's exactly (current total - target), you know the subarray between those two points sums to the target.`,
	},
	{
		ID:          197,
		Title:       "Knight Connection",
		Description: "A knight in chess can move to any square on the standard 8x8 chess board from any other square on the board, given enough turns. Its basic move is two steps in one direction and one step in a perpendicular direction. It can move to the square that is two squares horizontally and one square vertically, or two squares vertically and one square horizontally, regardless of whether the square is occupied. Write a function that returns the minimum number of knight moves needed to move from the position [x1, y1] to [x2, y2] on an infinite chess board.",
		Difficulty:  "Hard",
		Topic:       "Graphs",
		Signature:   "func knightConnection(knightA []int, knightB []int) int",
		TestCases: []TestCase{
			{Input: "knightA = [0, 0], knightB = [4, 2]", Expected: "1"},
			{Input: "knightA = [0, 0], knightB = [1, 1]", Expected: "2"},
		},
		Solution: `func knightConnection(knightA []int, knightB []int) int {
	queue := [][]int{{knightA[0], knightA[1], 0}}
	visited := make(map[string]bool)
	visited[getPositionString(knightA[0], knightA[1])] = true
	directions := [][]int{{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		row, col, moves := current[0], current[1], current[2]
		if row == knightB[0] && col == knightB[1] {
			return moves
		}
		for _, direction := range directions {
			newRow := row + direction[0]
			newCol := col + direction[1]
			posString := getPositionString(newRow, newCol)
			if !visited[posString] {
				visited[posString] = true
				queue = append(queue, []int{newRow, newCol, moves + 1})
			}
		}
	}
	return -1
}

func getPositionString(row int, col int) string {
	return string(rune(row)) + "," + string(rune(col))
}`,
		PythonSolution: `def knightConnection(knightA: List[int], knightB: List[int]) -> int:
    from collections import deque
    queue = deque([(knightA[0], knightA[1], 0)])
    visited = {(knightA[0], knightA[1])}
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    while queue:
        row, col, moves = queue.popleft()
        if row == knightB[0] and col == knightB[1]:
            return moves
        for dr, dc in directions:
            new_row = row + dr
            new_col = col + dc
            if (new_row, new_col) not in visited:
                visited.add((new_row, new_col))
                queue.append((new_row, new_col, moves + 1))
    return -1`,
		Explanation: `Finding the minimum moves for a knight to reach a target position is a classic BFS problem on an infinite chessboard. The knight's unique L-shaped movement pattern requires exploring all possible moves systematically.

The algorithm uses BFS starting from the knight's initial position. At each level, it generates all 8 possible knight moves (the knight moves in an L-shape: 2 squares in one direction and 1 square perpendicular). Each new position is checked: if it's the target, we return the move count. Otherwise, if it's unvisited, it's added to the queue for the next level.

BFS guarantees the shortest path because it explores all positions at distance 1 before distance 2, etc. The visited set prevents revisiting positions, ensuring we don't get stuck in cycles or take longer paths.

Edge cases handled include: knight already at target (0 moves), unreachable target (returns -1 or sentinel value), positions far apart (BFS finds shortest path), and positions requiring many moves (BFS handles efficiently).

An alternative approach would use A* search with a heuristic, but BFS is simpler and optimal for unweighted graphs. The time complexity is O(n) where n is the number of positions explored before reaching the target (bounded by the distance). The space complexity is O(n) for the queue and visited set.

Think of this as finding the shortest path on a chessboard: you explore all positions the knight can reach in 1 move, then 2 moves, continuing until you reach the target, always taking the shortest route.`,
	},
	{
		ID:          198,
		Title:       "Count Squares",
		Description: "You're given two positive integers representing the width and height of a grid-shaped, rectangular graph. Write a function that returns the number of ways to reach the bottom right corner of the graph when starting at the top left corner. Each move you take must either go down or right. In addition, you can't visit the same square more than once.",
		Difficulty:  "Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func countSquares(width int, height int) int",
		TestCases: []TestCase{
			{Input: "width = 3, height = 3", Expected: "6"},
		},
		Solution: `func countSquares(width int, height int) int {
	if width == 1 || height == 1 {
		return 1
	}
	return countSquares(width-1, height) + countSquares(width, height-1)
}`,
		PythonSolution: `def countSquares(width: int, height: int) -> int:
    if width == 1 or height == 1:
        return 1
    return countSquares(width - 1, height) + countSquares(width, height - 1)`,
		Explanation: `Counting the number of ways to traverse a grid from top-left to bottom-right requires recognizing that each position can be reached from either above or to the left. This creates overlapping subproblems that benefit from dynamic programming.

The recursive approach calculates ways(w, h) = ways(w-1, h) + ways(w, h-1), meaning the number of ways to reach a w×h grid equals the sum of ways to reach a (w-1)×h grid (coming from left) and ways to reach a w×(h-1) grid (coming from above). The base case is when width or height is 1, in which case there's only one way (straight line).

This recursive formula works because at each step, you can only move right or down, so the number of paths to a cell is the sum of paths to the cell above and the cell to the left. However, the naive recursive approach has exponential time complexity due to repeated calculations.

Edge cases handled include: 1×1 grid (1 way), 1×n or n×1 grid (1 way), and larger grids (calculated recursively).

The recursive approach can be optimized with memoization or bottom-up DP to achieve O(w*h) time and O(w*h) space (or O(min(w,h)) space with optimization). The naive recursive approach has O(2^(w+h)) time and O(w+h) space for the recursion stack.

Think of this as counting paths: at each intersection, you can come from the left or from above, so the total paths to that intersection is the sum of paths from both directions.`,
	},
	{
		ID:          199,
		Title:       "Same BSTs",
		Description: "An array of integers is said to represent a Binary Search Tree (BST) obtained by inserting each integer in the array, from left to right, into a BST. Write a function that takes in two arrays of integers and determines whether these arrays represent the same BST. Note that you're not allowed to construct any BSTs in your code.",
		Difficulty:  "Hard",
		Topic:       "Binary Search Trees",
		Signature:   "func sameBsts(arrayOne []int, arrayTwo []int) bool",
		TestCases: []TestCase{
			{Input: "arrayOne = [10, 15, 8, 12, 94, 81, 5, 2, 11], arrayTwo = [10, 8, 5, 15, 2, 12, 11, 94, 81]", Expected: "true"},
		},
		Solution: `func sameBsts(arrayOne []int, arrayTwo []int) bool {
	if len(arrayOne) != len(arrayTwo) {
		return false
	}
	if len(arrayOne) == 0 && len(arrayTwo) == 0 {
		return true
	}
	if arrayOne[0] != arrayTwo[0] {
		return false
	}
	leftOne := getSmaller(arrayOne)
	leftTwo := getSmaller(arrayTwo)
	rightOne := getBiggerOrEqual(arrayOne)
	rightTwo := getBiggerOrEqual(arrayTwo)
	return sameBsts(leftOne, leftTwo) && sameBsts(rightOne, rightTwo)
}

func getSmaller(array []int) []int {
	smaller := []int{}
	for i := 1; i < len(array); i++ {
		if array[i] < array[0] {
			smaller = append(smaller, array[i])
		}
	}
	return smaller
}

func getBiggerOrEqual(array []int) []int {
	biggerOrEqual := []int{}
	for i := 1; i < len(array); i++ {
		if array[i] >= array[0] {
			biggerOrEqual = append(biggerOrEqual, array[i])
		}
	}
	return biggerOrEqual
}`,
		PythonSolution: `def sameBsts(arrayOne: List[int], arrayTwo: List[int]) -> bool:
    if len(arrayOne) != len(arrayTwo):
        return False
    if len(arrayOne) == 0 and len(arrayTwo) == 0:
        return True
    if arrayOne[0] != arrayTwo[0]:
        return False
    left_one = get_smaller(arrayOne)
    left_two = get_smaller(arrayTwo)
    right_one = get_bigger_or_equal(arrayOne)
    right_two = get_bigger_or_equal(arrayTwo)
    return sameBsts(left_one, left_two) and sameBsts(right_one, right_two)

def get_smaller(array):
    return [x for x in array[1:] if x < array[0]]

def get_bigger_or_equal(array):
    return [x for x in array[1:] if x >= array[0]]`,
		Explanation: `Determining if two arrays represent the same BST without constructing the trees requires comparing the arrays recursively based on BST properties. The key insight is that the root must be the same, and the left and right subtrees must also represent the same BSTs.

The algorithm compares the roots (first elements) of both arrays. If they don't match, the BSTs are different. If they match, it splits each array into elements smaller than the root (left subtree) and elements greater than or equal to the root (right subtree), then recursively compares these subarrays.

This approach works because the BST property ensures that elements smaller than the root form the left subtree and elements greater or equal form the right subtree, regardless of insertion order. By comparing these subtrees recursively, we verify that the tree structures are identical.

Edge cases handled include: empty arrays (represent empty trees, return true), arrays with single element (trivial comparison), arrays with different roots (return false immediately), and arrays with same elements in different orders (handled by recursive comparison).

An alternative approach would construct both trees and compare them, but that violates the constraint. The recursive comparison approach achieves the same result without construction. The time complexity is O(n²) in the worst case where we create many subarrays, and space complexity is O(n²) for storing subarrays in the recursion stack.

Think of this as comparing two recipes: you check if they start with the same ingredient (root), then compare the ingredients that go before it (left subtree) and after it (right subtree), continuing recursively until you've verified everything matches.`,
	},
	{
		ID:          200,
		Title:       "Validate Three Nodes",
		Description: "You're given three nodes that are contained in the same Binary Search Tree: nodeOne, nodeTwo, and nodeThree. Write a function that returns a boolean representing whether one of nodeOne or nodeThree is an ancestor of nodeTwo and the other node is a descendant of nodeTwo.",
		Difficulty:  "Hard",
		Topic:       "Binary Search Trees",
		Signature:   "func validateThreeNodes(nodeOne *TreeNode, nodeTwo *TreeNode, nodeThree *TreeNode) bool",
		TestCases: []TestCase{
			{Input: "Tree structure", Expected: "Boolean result"},
		},
		Solution: `func validateThreeNodes(nodeOne *TreeNode, nodeTwo *TreeNode, nodeThree *TreeNode) bool {
	if isDescendant(nodeTwo, nodeOne) {
		return isDescendant(nodeThree, nodeTwo)
	}
	if isDescendant(nodeTwo, nodeThree) {
		return isDescendant(nodeOne, nodeTwo)
	}
	return false
}

func isDescendant(node *TreeNode, target *TreeNode) bool {
	if node == nil {
		return false
	}
	if node == target {
		return true
	}
	if target.Val < node.Val {
		return isDescendant(node.Left, target)
	}
	return isDescendant(node.Right, target)
}`,
		PythonSolution: `def validateThreeNodes(nodeOne: Optional[TreeNode], nodeTwo: Optional[TreeNode], nodeThree: Optional[TreeNode]) -> bool:
    if is_descendant(nodeTwo, nodeOne):
        return is_descendant(nodeThree, nodeTwo)
    if is_descendant(nodeTwo, nodeThree):
        return is_descendant(nodeOne, nodeTwo)
    return False

def is_descendant(node, target):
    if node is None:
        return False
    if node == target:
        return True
    if target.val < node.val:
        return is_descendant(node.left, target)
    return is_descendant(node.right, target)`,
		Explanation: `Validating that three nodes form an ancestor-descendant chain requires checking if one node is an ancestor of the middle node and the other is a descendant. The BST property enables efficient traversal to check these relationships.

The algorithm checks two cases: either nodeOne is an ancestor of nodeTwo and nodeThree is a descendant of nodeTwo, or nodeThree is an ancestor of nodeTwo and nodeOne is a descendant of nodeTwo. The isDescendant function uses the BST property: if the target value is less than the current node, search left; if greater, search right; if equal, we've found the node.

This approach efficiently verifies the ancestor-descendant relationships by leveraging the BST structure. We don't need to traverse the entire tree - we can follow the BST property to find specific nodes.

Edge cases handled include: nodeOne equals nodeTwo (not valid ancestor-descendant), nodeTwo equals nodeThree (not valid), all three nodes same (not valid), and nodes in different branches (returns false).

An alternative approach would find paths from root to each node and compare, but that's less efficient. The BST property-based traversal achieves O(h) time where h is tree height, and O(h) space for the recursion stack.

Think of this as checking family relationships in a family tree: you verify that one person is an ancestor of the middle person, and another is a descendant of the middle person, using the tree structure to navigate efficiently.`,
	},
	{
		ID:          201,
		Title:       "Repair BST",
		Description: "A Binary Search Tree (BST) is said to be valid if every node in the tree has a value strictly greater than the values of all nodes in its left subtree and less than or equal to the values of all nodes in its right subtree. Write a function that takes in a potentially invalid BST and returns the root of the repaired BST. You're allowed to change the values of nodes in the BST, but you're not allowed to change the structure of the BST.",
		Difficulty:  "Hard",
		Topic:       "Binary Search Trees",
		Signature:   "func repairBst(tree *TreeNode) *TreeNode",
		TestCases: []TestCase{
			{Input: "tree = [5, 7, 9, 1, 2, 8, 10, 0, -1, 15, 12]", Expected: "Repaired BST"},
		},
		Solution: `func repairBst(tree *TreeNode) *TreeNode {
	nodeOne := &TreeNode{}
	nodeTwo := &TreeNode{}
	inOrderTraverse(tree, &nodeOne, &nodeTwo, nil)
	nodeOne.Val, nodeTwo.Val = nodeTwo.Val, nodeOne.Val
	return tree
}

func inOrderTraverse(node *TreeNode, nodeOne **TreeNode, nodeTwo **TreeNode, previous **TreeNode) {
	if node == nil {
		return
	}
	inOrderTraverse(node.Left, nodeOne, nodeTwo, previous)
	if *previous != nil && (*previous).Val > node.Val {
		if *nodeOne == nil {
			*nodeOne = *previous
		}
		*nodeTwo = node
	}
	*previous = node
	inOrderTraverse(node.Right, nodeOne, nodeTwo, previous)
}`,
		PythonSolution: `def repairBst(tree: Optional[TreeNode]) -> Optional[TreeNode]:
    node_one = [None]
    node_two = [None]
    previous = [None]
    in_order_traverse(tree, node_one, node_two, previous)
    node_one[0].val, node_two[0].val = node_two[0].val, node_one[0].val
    return tree

def in_order_traverse(node, node_one, node_two, previous):
    if node is None:
        return
    in_order_traverse(node.left, node_one, node_two, previous)
    if previous[0] is not None and previous[0].val > node.val:
        if node_one[0] is None:
            node_one[0] = previous[0]
        node_two[0] = node
    previous[0] = node
    in_order_traverse(node.right, node_one, node_two, previous)`,
		Explanation: `Repairing a BST where two nodes have been swapped requires identifying which nodes are out of order and swapping their values. In-order traversal of a valid BST produces a sorted sequence, so violations indicate swapped nodes.

The algorithm performs an in-order traversal, tracking the previous node visited. When it encounters a node with a value less than the previous node's value, it has found a violation. The first violation identifies the first swapped node (stored in nodeOne), and the second violation identifies the second swapped node (stored in nodeTwo). After traversal, it swaps their values.

The key insight is that in a valid BST, in-order traversal yields values in strictly increasing order. When two nodes are swapped, we get exactly two violations: one where a larger value appears before a smaller value, and one where a smaller value appears after a larger value.

Edge cases handled include: adjacent nodes swapped (both violations found in one traversal), non-adjacent nodes swapped (two violations found), root node swapped (handled correctly), and nodes in different subtrees swapped (both found by in-order traversal).

An alternative approach would collect all values, sort them, and reconstruct, but that violates the constraint of not changing structure. The in-order traversal approach identifies and fixes only the swapped nodes. The time complexity is O(n) where n is the number of nodes, and space complexity is O(h) for the recursion stack.

Think of this as proofreading a sorted list: you go through it in order, and when you find a number that's smaller than the previous one, you've found an error. You find both errors and swap them back.`,
	},
	{
		ID:          202,
		Title:       "Sum BSTs",
		Description: "Write a function that takes in a Binary Tree and returns the sum of all of the subtrees' nodes' depths.",
		Difficulty:  "Hard",
		Topic:       "Binary Trees",
		Signature:   "func allKindsOfNodeDepths(root *TreeNode) int",
		TestCases: []TestCase{
			{Input: "root = [1, 2, 3, 4, 5, 6, 7, 8, 9]", Expected: "26"},
		},
		Solution: `func allKindsOfNodeDepths(root *TreeNode) int {
	return getTreeInfo(root).sumOfAllDepths
}

type TreeInfo struct {
	numNodesInTree int
	sumOfDepths    int
	sumOfAllDepths int
}

func getTreeInfo(tree *TreeNode) TreeInfo {
	if tree == nil {
		return TreeInfo{0, 0, 0}
	}
	leftTreeInfo := getTreeInfo(tree.Left)
	rightTreeInfo := getTreeInfo(tree.Right)
	sumOfLeftDepths := leftTreeInfo.sumOfDepths + leftTreeInfo.numNodesInTree
	sumOfRightDepths := rightTreeInfo.sumOfDepths + rightTreeInfo.numNodesInTree
	numNodesInTree := 1 + leftTreeInfo.numNodesInTree + rightTreeInfo.numNodesInTree
	sumOfDepths := sumOfLeftDepths + sumOfRightDepths
	sumOfAllDepths := sumOfDepths + leftTreeInfo.sumOfAllDepths + rightTreeInfo.sumOfAllDepths
	return TreeInfo{numNodesInTree, sumOfDepths, sumOfAllDepths}
}`,
		PythonSolution: `def allKindsOfNodeDepths(root: Optional[TreeNode]) -> int:
    return get_tree_info(root).sum_of_all_depths

class TreeInfo:
    def __init__(self, num_nodes_in_tree, sum_of_depths, sum_of_all_depths):
        self.num_nodes_in_tree = num_nodes_in_tree
        self.sum_of_depths = sum_of_depths
        self.sum_of_all_depths = sum_of_all_depths

def get_tree_info(tree):
    if tree is None:
        return TreeInfo(0, 0, 0)
    left_tree_info = get_tree_info(tree.left)
    right_tree_info = get_tree_info(tree.right)
    sum_of_left_depths = left_tree_info.sum_of_depths + left_tree_info.num_nodes_in_tree
    sum_of_right_depths = right_tree_info.sum_of_depths + right_tree_info.num_nodes_in_tree
    num_nodes_in_tree = 1 + left_tree_info.num_nodes_in_tree + right_tree_info.num_nodes_in_tree
    sum_of_depths = sum_of_left_depths + sum_of_right_depths
    sum_of_all_depths = sum_of_depths + left_tree_info.sum_of_all_depths + right_tree_info.sum_of_all_depths
    return TreeInfo(num_nodes_in_tree, sum_of_depths, sum_of_all_depths)`,
		Explanation: `Calculating the sum of all node depths across all subtrees requires understanding that each node contributes to the depths of nodes in its subtrees. The recursive approach efficiently computes this by combining information from subtrees.

The algorithm uses a recursive function that returns three values for each subtree: the number of nodes, the sum of depths of nodes in that subtree (relative to the subtree root), and the sum of all depths across all subtrees within that subtree. For each node, it combines information from left and right subtrees: the sum of depths at the current level is calculated by adding the subtree's sum of depths plus the number of nodes (since each node is one level deeper from the current root).

The sum of all depths combines the sum of depths at the current level with the sum of all depths from subtrees. This recursive combination ensures we count depths correctly at all levels.

Edge cases handled include: empty tree (returns zeros), single-node tree (depth 0, sum 0), balanced trees (depths calculated correctly), and skewed trees (depths calculated correctly).

An alternative O(n²) approach would calculate depths for each subtree separately, but the recursive approach is optimal. The time complexity is O(n) where n is the number of nodes, as we visit each node once. The space complexity is O(h) for the recursion stack where h is tree height.

Think of this as calculating the total "distance" from each node to all nodes in its subtrees: you recursively calculate distances in subtrees, then add the distances at the current level (which depend on how many nodes are in each subtree).`,
	},
	{
		ID:          203,
		Title:       "Max Path Sum In Binary Tree",
		Description: "Write a function that takes in a Binary Tree and returns the maximum path sum. A path is a collection of connected nodes in a tree, where no node is connected to more than two other nodes. A path sum is the sum of the values of the nodes in a particular path.",
		Difficulty:  "Hard",
		Topic:       "Binary Trees",
		Signature:   "func maxPathSum(tree *TreeNode) int",
		TestCases: []TestCase{
			{Input: "tree = [1, 2, 3, 4, 5, 6, 7]", Expected: "18"},
		},
		Solution: `func maxPathSum(tree *TreeNode) int {
	_, maxSum := findMaxSum(tree)
	return maxSum
}

func findMaxSum(tree *TreeNode) (int, int) {
	if tree == nil {
		return 0, math.MinInt32
	}
	leftMaxSumAsBranch, leftMaxPathSum := findMaxSum(tree.Left)
	rightMaxSumAsBranch, rightMaxPathSum := findMaxSum(tree.Right)
	maxChildSumAsBranch := max(leftMaxSumAsBranch, rightMaxSumAsBranch)
	value := tree.Val
	maxSumAsBranch := max(maxChildSumAsBranch+value, value)
	maxSumAsRootNode := max(leftMaxSumAsBranch+value+rightMaxSumAsBranch, maxSumAsBranch)
	maxPathSum := max(leftMaxPathSum, max(rightMaxPathSum, maxSumAsRootNode))
	return maxSumAsBranch, maxPathSum
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def maxPathSum(tree: Optional[TreeNode]) -> int:
    _, max_sum = find_max_sum(tree)
    return max_sum

def find_max_sum(tree):
    if tree is None:
        return 0, float('-inf')
    left_max_sum_as_branch, left_max_path_sum = find_max_sum(tree.left)
    right_max_sum_as_branch, right_max_path_sum = find_max_sum(tree.right)
    max_child_sum_as_branch = max(left_max_sum_as_branch, right_max_sum_as_branch)
    value = tree.val
    max_sum_as_branch = max(max_child_sum_as_branch + value, value)
    max_sum_as_root_node = max(left_max_sum_as_branch + value + right_max_sum_as_branch, max_sum_as_branch)
    max_path_sum = max(left_max_path_sum, max(right_max_path_sum, max_sum_as_root_node))
    return max_sum_as_branch, max_path_sum`,
		Explanation: `Finding the maximum path sum in a binary tree requires distinguishing between paths that pass through a node (connecting left and right subtrees) and paths that are entirely within subtrees. The recursive approach calculates both for each node.

The algorithm returns two values for each node: the maximum sum of a path ending at this node (as a branch), and the maximum path sum found in the subtree rooted at this node. For the branch sum, we can either take just the node value or add it to the maximum child branch sum. For the path through the node, we combine the node value with both left and right branch sums. The overall maximum is the maximum of: the path through the current node, the maximum path in the left subtree, and the maximum path in the right subtree.

This approach efficiently computes the maximum path by considering all possible paths: those entirely within subtrees and those passing through the current node. The recursive structure naturally handles the tree traversal.

Edge cases handled include: negative node values (branch sum can be just the node value), all negative values (maximum path is the least negative single node), paths that don't include the root (handled by subtree maximums), and single-node trees (path is just that node's value).

An alternative approach would enumerate all paths, but that's exponential. The recursive approach achieves O(n) time by visiting each node once, and O(h) space for the recursion stack where h is the tree height.

Think of this as finding the best route through a network: at each junction (node), you consider the best path ending here, the best path passing through here, and the best paths in each branch, always tracking the overall best.`,
	},
	{
		ID:          204,
		Title:       "Find Nodes Distance K",
		Description: "You're given the root node of a Binary Tree, a target node within that tree, and a positive integer k. Write a function that returns the values of all the nodes that are exactly distance k from the target node. The distance between two nodes is defined as the number of edges that must be traversed to go from one node to the other.",
		Difficulty:  "Hard",
		Topic:       "Binary Trees",
		Signature:   "func findNodesDistanceK(tree *TreeNode, target int, k int) []int",
		TestCases: []TestCase{
			{Input: "tree = [1, 2, 3, 4, 5, 6, 7, 8, 9], target = 3, k = 2", Expected: "[2, 7, 8]"},
		},
		Solution: `func findNodesDistanceK(tree *TreeNode, target int, k int) []int {
	nodesDistanceK := []int{}
	findDistanceFromNodeToTarget(tree, target, k, &nodesDistanceK)
	return nodesDistanceK
}

func findDistanceFromNodeToTarget(node *TreeNode, target int, k int, nodesDistanceK *[]int) int {
	if node == nil {
		return -1
	}
	if node.Val == target {
		addSubtreeNodesAtDistanceK(node, 0, k, nodesDistanceK)
		return 1
	}
	leftDistance := findDistanceFromNodeToTarget(node.Left, target, k, nodesDistanceK)
	rightDistance := findDistanceFromNodeToTarget(node.Right, target, k, nodesDistanceK)
	if leftDistance == k || rightDistance == k {
		*nodesDistanceK = append(*nodesDistanceK, node.Val)
	}
	if leftDistance != -1 {
		addSubtreeNodesAtDistanceK(node.Right, leftDistance+1, k, nodesDistanceK)
		return leftDistance + 1
	}
	if rightDistance != -1 {
		addSubtreeNodesAtDistanceK(node.Left, rightDistance+1, k, nodesDistanceK)
		return rightDistance + 1
	}
	return -1
}

func addSubtreeNodesAtDistanceK(node *TreeNode, distance int, k int, nodesDistanceK *[]int) {
	if node == nil {
		return
	}
	if distance == k {
		*nodesDistanceK = append(*nodesDistanceK, node.Val)
	} else {
		addSubtreeNodesAtDistanceK(node.Left, distance+1, k, nodesDistanceK)
		addSubtreeNodesAtDistanceK(node.Right, distance+1, k, nodesDistanceK)
	}
}`,
		PythonSolution: `def findNodesDistanceK(tree: Optional[TreeNode], target: int, k: int) -> List[int]:
    nodes_distance_k = []
    find_distance_from_node_to_target(tree, target, k, nodes_distance_k)
    return nodes_distance_k

def find_distance_from_node_to_target(node, target, k, nodes_distance_k):
    if node is None:
        return -1
    if node.val == target:
        add_subtree_nodes_at_distance_k(node, 0, k, nodes_distance_k)
        return 1
    left_distance = find_distance_from_node_to_target(node.left, target, k, nodes_distance_k)
    right_distance = find_distance_from_node_to_target(node.right, target, k, nodes_distance_k)
    if left_distance == k or right_distance == k:
        nodes_distance_k.append(node.val)
    if left_distance != -1:
        add_subtree_nodes_at_distance_k(node.right, left_distance + 1, k, nodes_distance_k)
        return left_distance + 1
    if right_distance != -1:
        add_subtree_nodes_at_distance_k(node.left, right_distance + 1, k, nodes_distance_k)
        return right_distance + 1
    return -1

def add_subtree_nodes_at_distance_k(node, distance, k, nodes_distance_k):
    if node is None:
        return
    if distance == k:
        nodes_distance_k.append(node.val)
    else:
        add_subtree_nodes_at_distance_k(node.left, distance + 1, k, nodes_distance_k)
        add_subtree_nodes_at_distance_k(node.right, distance + 1, k, nodes_distance_k)`,
		Explanation: `Finding all nodes at distance k from a target node requires handling two cases: nodes in the target's subtree and nodes reachable through ancestors. The recursive approach elegantly handles both cases.

The algorithm first searches for the target node. When found, it searches k distance away in the target's subtree. For ancestors of the target, it calculates the distance from the ancestor to the target, then searches the opposite subtree (the one not containing the target) at distance k - (distance to target).

The key insight is that nodes at distance k can be in the target's subtree (found by DFS from target) or in other subtrees reachable through ancestors. When we find the target at distance d from an ancestor, nodes at distance k-d from that ancestor in the opposite subtree are at distance k from the target.

Edge cases handled include: target is root (only search subtree), target is leaf (no subtree to search), k equals 0 (target itself), k larger than tree depth (no nodes found), and multiple nodes at distance k (all found).

An alternative approach would use BFS from target, but that doesn't handle the ancestor case correctly. The recursive approach naturally handles both subtree and ancestor cases. The time complexity is O(n) where n is the number of nodes, as we visit each node at most once. The space complexity is O(h) for the recursion stack.

Think of this as finding all places exactly k steps away: you can go k steps in the target's subtree, or you can go up to an ancestor and then go the remaining steps in the opposite direction from that ancestor.`,
	},
	{
		ID:          205,
		Title:       "Max Sum Increasing Subsequence",
		Description: "Write a function that takes in a non-empty array of integers and returns the greatest sum that can be generated from a strictly-increasing subsequence in the array as well as an array of the numbers in that subsequence.",
		Difficulty:  "Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func maxSumIncreasingSubsequence(array []int) (int, []int)",
		TestCases: []TestCase{
			{Input: "array = [10, 70, 20, 30, 50, 11, 30]", Expected: "sum: 110, sequence: [10, 20, 30, 50]"},
		},
		Solution: `func maxSumIncreasingSubsequence(array []int) (int, []int) {
	sums := make([]int, len(array))
	sequences := make([]int, len(array))
	for i := range sequences {
		sequences[i] = -1
	}
	copy(sums, array)
	maxSumIdx := 0
	for i := 0; i < len(array); i++ {
		currentNum := array[i]
		for j := 0; j < i; j++ {
			otherNum := array[j]
			if otherNum < currentNum && sums[j]+currentNum >= sums[i] {
				sums[i] = sums[j] + currentNum
				sequences[i] = j
			}
		}
		if sums[i] >= sums[maxSumIdx] {
			maxSumIdx = i
		}
	}
	return sums[maxSumIdx], buildSequence(array, sequences, maxSumIdx)
}

func buildSequence(array []int, sequences []int, currentIdx int) []int {
	sequence := []int{}
	for currentIdx != -1 {
		sequence = append([]int{array[currentIdx]}, sequence...)
		currentIdx = sequences[currentIdx]
	}
	return sequence
}`,
		PythonSolution: `def maxSumIncreasingSubsequence(array: List[int]) -> Tuple[int, List[int]]:
    sums = array[:]
    sequences = [None] * len(array)
    max_sum_idx = 0
    for i in range(len(array)):
        current_num = array[i]
        for j in range(i):
            other_num = array[j]
            if other_num < current_num and sums[j] + current_num >= sums[i]:
                sums[i] = sums[j] + current_num
                sequences[i] = j
        if sums[i] >= sums[max_sum_idx]:
            max_sum_idx = i
    return sums[max_sum_idx], build_sequence(array, sequences, max_sum_idx)

def build_sequence(array, sequences, current_idx):
    sequence = []
    while current_idx is not None:
        sequence.insert(0, array[current_idx])
        current_idx = sequences[current_idx]
    return sequence`,
		Explanation: `Finding the maximum sum increasing subsequence requires tracking both the maximum sum and the actual sequence. Dynamic programming efficiently solves this by building up solutions for longer subsequences from shorter ones.

The algorithm maintains two arrays: sums[i] stores the maximum sum of an increasing subsequence ending at index i, and sequences[i] stores the index of the previous element in that subsequence. For each position i, it checks all previous positions j where array[j] < array[i]. If sums[j] + array[i] improves sums[i], it updates both the sum and the sequence pointer.

After processing all elements, it finds the index with the maximum sum and backtracks through the sequence pointers to reconstruct the actual subsequence. This backtracking ensures we return the correct sequence, not just the sum.

Edge cases handled include: array in decreasing order (each element is its own subsequence), array in increasing order (entire array is the subsequence), array with all negative numbers (maximum is least negative), and array with duplicates (handled by < comparison).

An alternative O(n log n) approach would use binary search with a different data structure, but the DP approach is more intuitive and handles sequence reconstruction naturally. The time complexity is O(n²) where n is array length, as we check all previous elements for each position. The space complexity is O(n) for the sums and sequences arrays.

Think of this as building the best increasing sequence ending at each position: you look at all previous positions that could extend to the current position, pick the one that gives the best sum, and remember how you got there so you can reconstruct the sequence later.`,
	},
	{
		ID:          206,
		Title:       "Longest Common Subsequence",
		Description: "Write a function that takes in two strings and returns the longest common subsequence. A subsequence of a string is a set of characters that aren't necessarily adjacent in the string but that are in the same order as they appear in the string.",
		Difficulty:  "Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func longestCommonSubsequence(str1 string, str2 string) []string",
		TestCases: []TestCase{
			{Input: "str1 = \"ZXVVYZW\", str2 = \"XKYKZPW\"", Expected: "[\"X\", \"Y\", \"Z\", \"W\"]"},
		},
		Solution: `func longestCommonSubsequence(str1 string, str2 string) []string {
	lcs := make([][][]string, len(str2)+1)
	for i := range lcs {
		lcs[i] = make([][]string, len(str1)+1)
	}
	for i := 1; i < len(str2)+1; i++ {
		for j := 1; j < len(str1)+1; j++ {
			if str2[i-1] == str1[j-1] {
				lcs[i][j] = append(lcs[i-1][j-1], string(str2[i-1]))
			} else {
				if len(lcs[i-1][j]) > len(lcs[i][j-1]) {
					lcs[i][j] = lcs[i-1][j]
				} else {
					lcs[i][j] = lcs[i][j-1]
				}
			}
		}
	}
	return lcs[len(str2)][len(str1)]
}`,
		PythonSolution: `def longestCommonSubsequence(str1: str, str2: str) -> List[str]:
    lcs = [[[] for _ in range(len(str1) + 1)] for _ in range(len(str2) + 1)]
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) +1):
            if str2[i - 1] == str1[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + [str2[i - 1]]
            else:
                if len(lcs[i - 1][j]) > len(lcs[i][j - 1]):
                    lcs[i][j] = lcs[i - 1][j]
                else:
                    lcs[i][j] = lcs[i][j - 1]
    return lcs[len(str2)][len(str1)]`,
		Explanation: `Finding the longest common subsequence (LCS) between two strings requires determining the longest sequence of characters that appear in both strings in the same order (not necessarily adjacent). Dynamic programming efficiently solves this by building up solutions for longer prefixes from shorter ones.

The algorithm uses a 2D DP table where lcs[i][j] stores the LCS of the first i characters of str2 and the first j characters of str1. The base cases handle empty strings (LCS is empty). For each cell, if the characters match, we extend the LCS from the diagonal (lcs[i-1][j-1] + current character). If they don't match, we take the longer of the LCS from above (lcs[i-1][j]) or from the left (lcs[i][j-1]).

This approach works because when characters match, we can extend the LCS found so far. When they don't match, we keep the best LCS found by either skipping the current character from str1 or str2. The DP table ensures we don't recompute subproblems.

Edge cases handled include: one string empty (LCS is empty), strings identical (LCS is the entire string), strings with no common characters (LCS is empty), and strings where one is a subsequence of the other (LCS is the shorter string).

An alternative recursive approach with memoization would work but is less efficient. The DP approach achieves O(n*m) time where n and m are string lengths, and O(n*m) space for the DP table (can be optimized to O(min(n,m)) if only length is needed, but we need the actual sequence here).

Think of this as finding the longest matching pattern: you build up matches character by character, extending when characters match and choosing the best option when they don't, keeping track of the actual sequence as you go.`,
	},
	{
		ID:          207,
		Title:       "Min Number Of Jumps",
		Description: "You're given a non-empty array of positive integers where each integer represents the maximum number of steps you can take forward in the array. For example, if the element at index 1 is 3, you can go from index 1 to index 2, 3, or 4. Write a function that returns the minimum number of jumps needed to reach the final index. Note that jumping from index i to index i + x always constitutes one jump, no matter how large x is.",
		Difficulty:  "Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func minNumberOfJumps(array []int) int",
		TestCases: []TestCase{
			{Input: "array = [3, 4, 2, 1, 2, 3, 7, 1, 1, 1, 3]", Expected: "4"},
			{Input: "array = [1]", Expected: "0"},
		},
		Solution: `func minNumberOfJumps(array []int) int {
	if len(array) == 1 {
		return 0
	}
	jumps := 0
	maxReach := array[0]
	steps := array[0]
	for i := 1; i < len(array)-1; i++ {
		maxReach = max(maxReach, i+array[i])
		steps--
		if steps == 0 {
			jumps++
			steps = maxReach - i
		}
	}
	return jumps + 1
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def minNumberOfJumps(array: List[int]) -> int:
    if len(array) == 1:
        return 0
    jumps = 0
    max_reach = array[0]
    steps = array[0]
    for i in range(1, len(array) - 1):
        max_reach = max(max_reach, i + array[i])
        steps -= 1
        if steps == 0:
            jumps += 1
            steps = max_reach - i
    return jumps + 1`,
		Explanation: `Finding the minimum jumps to reach the end requires a greedy approach that tracks how far we can reach and how many steps we have remaining from the current jump. The key insight is to jump only when necessary (when steps are exhausted) and always jump to the position that maximizes future reach.

The algorithm maintains two key variables: maxReach (the farthest position we can reach with current and future jumps) and steps (remaining steps from the current jump). As we iterate through the array, we update maxReach to include positions reachable from the current position. When steps reach 0, we must jump, so we increment the jump count and update steps to the remaining steps from the new position (maxReach - current position).

This greedy approach is optimal because we always delay jumping until necessary and always choose the position that gives us the maximum future reach. By tracking maxReach, we ensure we're always aware of the best position we can reach.

Edge cases handled include: array starting with 0 (cannot progress, but handled by algorithm), single-element array (0 jumps needed), array where we can reach end in one jump (returns 1), and arrays where some positions are unreachable (algorithm detects this).

An alternative dynamic programming approach would be O(n²), but the greedy approach achieves O(n) time by making a single pass. The space complexity is O(1) as we only track a few variables.

Think of this as crossing stepping stones: you look ahead to see how far you can reach from your current stone, and you only step to a new stone when you've used all steps from your current jump, always choosing the stone that lets you reach farthest ahead.`,
	},
	{
		ID:          208,
		Title:       "Water Area",
		Description: "You're given an array of non-negative integers where each non-zero integer represents the height of a pillar of width 1. Imagine water being poured over all of the pillars. Write a function that returns the surface area of the water trapped between the pillars viewed from the front. Note that spilled water should be ignored.",
		Difficulty:  "Hard",
		Topic:       "Arrays",
		Signature:   "func waterArea(heights []int) int",
		TestCases: []TestCase{
			{Input: "heights = [0, 8, 0, 0, 5, 0, 0, 10, 0, 0, 1, 1, 0, 3]", Expected: "48"},
		},
		Solution: `func waterArea(heights []int) int {
	if len(heights) == 0 {
		return 0
	}
	leftIdx := 0
	rightIdx := len(heights) - 1
	leftMax := heights[leftIdx]
	rightMax := heights[rightIdx]
	surfaceArea := 0
	for leftIdx < rightIdx {
		if heights[leftIdx] < heights[rightIdx] {
			leftIdx++
			leftMax = max(leftMax, heights[leftIdx])
			surfaceArea += leftMax - heights[leftIdx]
		} else {
			rightIdx--
			rightMax = max(rightMax, heights[rightIdx])
			surfaceArea += rightMax - heights[rightIdx]
		}
	}
	return surfaceArea
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def waterArea(heights: List[int]) -> int:
    if len(heights) == 0:
        return 0
    left_idx = 0
    right_idx = len(heights) - 1
    left_max = heights[left_idx]
    right_max = heights[right_idx]
    surface_area = 0
    while left_idx < right_idx:
        if heights[left_idx] < heights[right_idx]:
            left_idx += 1
            left_max = max(left_max, heights[left_idx])
            surface_area += left_max - heights[left_idx]
        else:
            right_idx -= 1
            right_max = max(right_max, heights[right_idx])
            surface_area += right_max - heights[right_idx]
    return surface_area`,
		Explanation: `Calculating water area (trapped rainwater) requires determining how much water can be trapped between bars. The key insight is that water trapped at a position depends on the minimum of the maximum heights on the left and right, minus the current height.

The algorithm uses two pointers starting from both ends. It maintains leftMax and rightMax to track the maximum heights encountered so far from each side. At each step, it moves the pointer with the smaller maximum height (because water level is limited by the smaller side). For the current position, if the height is less than the corresponding max, water can be trapped (max - height). Otherwise, we update the max.

This approach works because we only need to know the limiting height (the smaller of left and right maxes) to calculate trapped water. By moving the pointer with the smaller max, we ensure we've found the correct limiting height for that position.

Edge cases handled include: all bars same height (no water trapped), bars in increasing order (no water trapped), bars in decreasing order (no water trapped), and bars forming a basin (water trapped in the middle).

An alternative O(n) approach would precompute left and right maxes, but the two-pointer approach achieves the same with O(1) space. The time complexity is O(n) where n is the number of bars, and space complexity is O(1) as we only use a few variables.

Think of this as filling a container: the water level at any point is determined by the lower of the two "walls" on either side, and you calculate how much water fits by moving inward from both sides, always focusing on the side with the lower wall.`,
	},
	{
		ID:          209,
		Title:       "Knapsack Problem",
		Description: "You're given an array of arrays where each subarray holds two integer values and represents an item; the first integer is the item's value and the second integer is the item's weight. You're also given an integer representing the maximum capacity of a knapsack that you have. Your goal is to fit items in your knapsack without having the sum of their weights exceed the knapsack's capacity, all the while maximizing their combined value. Note that you only have one of each item at your disposal. Write a function that returns the maximized combined value of the items that you should pick as well as an array of the indices of each item picked.",
		Difficulty:  "Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func knapsackProblem(items [][]int, capacity int) (int, []int)",
		TestCases: []TestCase{
			{Input: "items = [[1, 2], [4, 3], [5, 6], [6, 7]], capacity = 10", Expected: "value: 10, indices: [1, 3]"},
		},
		Solution: `func knapsackProblem(items [][]int, capacity int) (int, []int) {
	knapsackValues := make([][]int, len(items)+1)
	for i := range knapsackValues {
		knapsackValues[i] = make([]int, capacity+1)
	}
	for i := 1; i < len(items)+1; i++ {
		currentWeight := items[i-1][1]
		currentValue := items[i-1][0]
		for c := 0; c < capacity+1; c++ {
			if currentWeight > c {
				knapsackValues[i][c] = knapsackValues[i-1][c]
			} else {
				knapsackValues[i][c] = max(knapsackValues[i-1][c], knapsackValues[i-1][c-currentWeight]+currentValue)
			}
		}
	}
	return knapsackValues[len(items)][capacity], getKnapsackItems(knapsackValues, items)
}

func getKnapsackItems(knapsackValues [][]int, items [][]int) []int {
	sequence := []int{}
	i := len(knapsackValues) - 1
	c := len(knapsackValues[0]) - 1
	for i > 0 {
		if knapsackValues[i][c] == knapsackValues[i-1][c] {
			i--
		} else {
			sequence = append([]int{i - 1}, sequence...)
			c -= items[i-1][1]
			i--
		}
		if c == 0 {
			break
		}
	}
	return sequence
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def knapsackProblem(items: List[List[int]], capacity: int) -> Tuple[int, List[int]]:
    knapsack_values = [[0 for _ in range(capacity + 1)] for _ in range(len(items) + 1)]
    for i in range(1, len(items) + 1):
        current_weight = items[i - 1][1]
        current_value = items[i - 1][0]
        for c in range(capacity + 1):
            if current_weight > c:
                knapsack_values[i][c] = knapsack_values[i - 1][c]
            else:
                knapsack_values[i][c] = max(knapsack_values[i - 1][c], knapsack_values[i - 1][c - current_weight] + current_value)
    return knapsack_values[len(items)][capacity], get_knapsack_items(knapsack_values, items)

def get_knapsack_items(knapsack_values, items):
    sequence = []
    i = len(knapsack_values) - 1
    c = len(knapsack_values[0]) - 1
    while i > 0:
        if knapsack_values[i][c] == knapsack_values[i - 1][c]:
            i -= 1
        else:
            sequence.insert(0, i - 1)
            c -= items[i - 1][1]
            i -= 1
        if c == 0:
            break
    return sequence`,
		Explanation: `The knapsack problem requires selecting items to maximize value without exceeding capacity. Dynamic programming efficiently solves this by building up solutions for larger capacities and more items from smaller subproblems.

The algorithm uses a 2D DP table where dp[i][c] represents the maximum value achievable using the first i items with capacity c. For each item and capacity, we have two choices: take the item (if it fits, value is item value + dp[i-1][c-weight]) or skip it (value is dp[i-1][c]). We choose the maximum of these two options.

After filling the DP table, we backtrack to find which items were selected. Starting from dp[n][capacity], if dp[i][c] equals dp[i-1][c], the item wasn't taken; otherwise, it was taken, and we move to dp[i-1][c-weight].

Edge cases handled include: capacity 0 (no items can be taken), all items too heavy (no items taken), all items fit (all items taken), and items with same weight/value ratios (DP finds optimal selection).

An alternative recursive approach with memoization would work but is less efficient. The DP approach achieves O(n*c) time where n is number of items and c is capacity, and O(n*c) space for the DP table (can be optimized to O(c) if only value is needed, but we need backtracking here).

Think of this as packing a suitcase: for each item, you decide whether to take it based on whether it improves your total value and fits in the remaining space, building up the best packing strategy item by item, then tracing back to see what you actually packed.`,
	},
	{
		ID:          210,
		Title:       "Disk Stacking",
		Description: "You're given a non-empty array of arrays where each subarray holds three integers and represents a disk. These integers denote each disk's width, depth, and height, respectively. Your goal is to stack up the disks and maximize the total height of the stack. A disk must have a strictly smaller width, depth, and height than any other disk below it. Write a function that returns an array of the indices of the disks in the final stack, starting with the top disk and ending with the bottom disk. Note that the indices will be the same as the order of the disks in the input array.",
		Difficulty:  "Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func diskStacking(disks [][]int) [][]int",
		TestCases: []TestCase{
			{Input: "disks = [[2, 1, 2], [3, 2, 3], [2, 2, 8], [2, 3, 4], [1, 2, 1], [4, 4, 5]]", Expected: "[[2, 1, 2], [3, 2, 3], [4, 4, 5]]"},
		},
		Solution: `import "sort"

func diskStacking(disks [][]int) [][]int {
	sort.Slice(disks, func(i, j int) bool {
		return disks[i][2] < disks[j][2]
	})
	heights := make([]int, len(disks))
	sequences := make([]int, len(disks))
	for i := range heights {
		heights[i] = disks[i][2]
		sequences[i] = -1
	}
	maxHeightIdx := 0
	for i := 1; i < len(disks); i++ {
		currentDisk := disks[i]
		for j := 0; j < i; j++ {
			otherDisk := disks[j]
			if areValidDimensions(otherDisk, currentDisk) {
				if heights[i] <= currentDisk[2]+heights[j] {
					heights[i] = currentDisk[2] + heights[j]
					sequences[i] = j
				}
			}
		}
		if heights[i] >= heights[maxHeightIdx] {
			maxHeightIdx = i
		}
	}
	return buildSequence(disks, sequences, maxHeightIdx)
}

func areValidDimensions(o []int, c []int) bool {
	return o[0] < c[0] && o[1] < c[1] && o[2] < c[2]
}

func buildSequence(array [][]int, sequences []int, currentIdx int) [][]int {
	sequence := [][]int{}
	for currentIdx != -1 {
		sequence = append([][]int{array[currentIdx]}, sequence...)
		currentIdx = sequences[currentIdx]
	}
	return sequence
}`,
		PythonSolution: `def diskStacking(disks: List[List[int]]) -> List[List[int]]:
    disks.sort(key=lambda x: x[2])
    heights = [disk[2] for disk in disks]
    sequences = [None] * len(disks)
    max_height_idx = 0
    for i in range(1, len(disks)):
        current_disk = disks[i]
        for j in range(i):
            other_disk = disks[j]
            if are_valid_dimensions(other_disk, current_disk):
                if heights[i] <= current_disk[2] + heights[j]:
                    heights[i] = current_disk[2] + heights[j]
                    sequences[i] = j
        if heights[i] >= heights[max_height_idx]:
            max_height_idx = i
    return build_sequence(disks, sequences, max_height_idx)

def are_valid_dimensions(o, c):
    return o[0] < c[0] and o[1] < c[1] and o[2] < c[2]

def build_sequence(array, sequences, current_idx):
    sequence = []
    while current_idx is not None:
        sequence.insert(0, array[current_idx])
        current_idx = sequences[current_idx]
    return sequence`,
		Explanation: `Finding the tallest stack of disks requires dynamic programming to determine the maximum height achievable ending at each disk. The key constraint is that a disk can only be placed on top of a strictly smaller disk.

The algorithm first sorts disks by height (and width as a secondary key for stability). Then it uses dynamic programming where heights[i] represents the maximum height of a stack ending with disk i. For each disk, it checks all smaller disks and chooses the one that gives the maximum height when this disk is placed on top. It also maintains a sequences array to track which disk comes before each disk in the optimal stack.

After computing all maximum heights, it finds the disk that gives the maximum overall height, then reconstructs the sequence by following the sequences array backwards from that disk.

Edge cases handled include: all disks same size (no valid stack, returns empty), disks in optimal order (all used in sequence), and disks where only one can be used (returns that disk).

An alternative approach would try all permutations, but that's factorial time. The DP approach achieves O(n²) time by checking each disk against all smaller disks. The space complexity is O(n) for the DP arrays and sequences.

Think of this as building a tower: you sort blocks by size, then for each block, you find the tallest tower you can build ending with that block by placing it on top of the best smaller block, always tracking which block comes before to reconstruct the final tower.`,
	},
	{
		ID:          211,
		Title:       "Numbers In Pi",
		Description: "Given a string representation of the first n digits of Pi and a list of your favorite numbers (all positive integers), write a function that returns the smallest number of spaces that need to be added to the n digits of Pi such that all resulting numbers are found in the list of favorite numbers. If no number of spaces to be added exists such that all the resulting numbers are found in the list of favorite numbers, the function should return -1.",
		Difficulty:  "Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func numbersInPi(pi string, numbers []string) int",
		TestCases: []TestCase{
			{Input: "pi = \"3141592\", numbers = [\"3141\", \"5\", \"31\", \"2\", \"4159\", \"9\", \"42\"]", Expected: "2"},
		},
		Solution: `func numbersInPi(pi string, numbers []string) int {
	numbersTable := make(map[string]bool)
	for _, number := range numbers {
		numbersTable[number] = true
	}
	cache := make(map[int]int)
	minSpaces := getMinSpaces(pi, numbersTable, cache, 0)
	if minSpaces == math.MaxInt32 {
		return -1
	}
	return minSpaces
}

func getMinSpaces(pi string, numbersTable map[string]bool, cache map[int]int, idx int) int {
	if idx == len(pi) {
		return -1
	}
	if val, found := cache[idx]; found {
		return val
	}
	minSpaces := math.MaxInt32
	for i := idx; i < len(pi); i++ {
		prefix := pi[idx : i+1]
		if numbersTable[prefix] {
			minSpacesInSuffix := getMinSpaces(pi, numbersTable, cache, i+1)
			minSpaces = min(minSpaces, minSpacesInSuffix+1)
		}
	}
	cache[idx] = minSpaces
	return minSpaces
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}`,
		PythonSolution: `def numbersInPi(pi: str, numbers: List[str]) -> int:
    numbers_table = {number: True for number in numbers}
    cache = {}
    min_spaces = get_min_spaces(pi, numbers_table, cache, 0)
    return min_spaces if min_spaces != float('inf') else -1

def get_min_spaces(pi, numbers_table, cache, idx):
    if idx == len(pi):
        return -1
    if idx in cache:
        return cache[idx]
    min_spaces = float('inf')
    for i in range(idx, len(pi)):
        prefix = pi[idx:i + 1]
        if prefix in numbers_table:
            min_spaces_in_suffix = get_min_spaces(pi, numbers_table, cache, i + 1)
            min_spaces = min(min_spaces, min_spaces_in_suffix + 1)
    cache[idx] = min_spaces
    return min_spaces`,
		Explanation: `Finding the minimum spaces needed to partition a Pi string such that all resulting numbers are in a favorite numbers list requires trying all possible partitions. Memoized recursion efficiently explores this space while avoiding redundant calculations.

The algorithm tries all possible prefixes starting from the current index. For each prefix that exists in the numbers table, it recursively finds the minimum spaces needed for the remaining suffix. The minimum spaces for the current position is the minimum of all valid prefix choices plus one space (for the current partition).

Memoization is crucial here because the same suffix positions may be computed multiple times through different prefix choices. By caching results for each starting index, we avoid recalculating the minimum spaces for the same subproblems.

Edge cases handled include: entire string is a single number (returns 0 spaces), no valid partition exists (returns -1), string where all characters form valid numbers (minimum spaces found), and strings with overlapping number possibilities (all tried, minimum chosen).

An alternative dynamic programming approach would build from right to left, but the recursive approach is more intuitive. The time complexity is O(n³ + m) where n is Pi string length and m is numbers list size: O(n²) for trying all prefixes at each position, O(n) positions, plus O(m) for building the numbers table. The space complexity is O(n + m) for the cache and numbers table.

Think of this as breaking a long number into valid segments: you try breaking it at each possible position, and if the prefix is valid, you recursively find the best way to break the rest, always tracking the minimum total breaks needed.`,
	},
	{
		ID:          212,
		Title:       "Maximum Sum Submatrix",
		Description: "Given a two-dimensional array of potentially unequal height and width that's filled with integers, as well as a positive integer size, write a function that returns the maximum sum that can be generated from a submatrix with dimensions size * size.",
		Difficulty:  "Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func maximumSumSubmatrix(matrix [][]int, size int) int",
		TestCases: []TestCase{
			{Input: "matrix = [[5, 3, -1, 5], [-7, 3, 7, 4], [12, 8, 0, 0], [1, -8, -8, 2]], size = 2", Expected: "18"},
		},
		Solution: `func maximumSumSubmatrix(matrix [][]int, size int) int {
	sums := createSumMatrix(matrix)
	maxSubMatrixSum := math.MinInt32
	for row := size - 1; row < len(matrix); row++ {
		for col := size - 1; col < len(matrix[0]); col++ {
			total := sums[row][col]
			touchesTopBorder := row-size < 0
			if !touchesTopBorder {
				total -= sums[row-size][col]
			}
			touchesLeftBorder := col-size < 0
			if !touchesLeftBorder {
				total -= sums[row][col-size]
			}
			touchesTopOrLeftBorder := touchesTopBorder || touchesLeftBorder
			if !touchesTopOrLeftBorder {
				total += sums[row-size][col-size]
			}
			maxSubMatrixSum = max(maxSubMatrixSum, total)
		}
	}
	return maxSubMatrixSum
}

func createSumMatrix(matrix [][]int) [][]int {
	sums := make([][]int, len(matrix))
	for i := range sums {
		sums[i] = make([]int, len(matrix[0]))
	}
	sums[0][0] = matrix[0][0]
	for idx := 1; idx < len(matrix[0]); idx++ {
		sums[0][idx] = sums[0][idx-1] + matrix[0][idx]
	}
	for idx := 1; idx < len(matrix); idx++ {
		sums[idx][0] = sums[idx-1][0] + matrix[idx][0]
	}
	for row := 1; row < len(matrix); row++ {
		for col := 1; col < len(matrix[0]); col++ {
			sums[row][col] = sums[row-1][col] + sums[row][col-1] - sums[row-1][col-1] + matrix[row][col]
		}
	}
	return sums
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def maximumSumSubmatrix(matrix: List[List[int]], size: int) -> int:
    sums = create_sum_matrix(matrix)
    max_sub_matrix_sum = float('-inf')
    for row in range(size - 1, len(matrix)):
        for col in range(size - 1, len(matrix[0])):
            total = sums[row][col]
            touches_top_border = row - size < 0
            if not touches_top_border:
                total -= sums[row - size][col]
            touches_left_border = col - size < 0
            if not touches_left_border:
                total -= sums[row][col - size]
            touches_top_or_left_border = touches_top_border or touches_left_border
            if not touches_top_or_left_border:
                total += sums[row - size][col - size]
            max_sub_matrix_sum = max(max_sub_matrix_sum, total)
    return max_sub_matrix_sum

def create_sum_matrix(matrix):
    sums = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    sums[0][0] = matrix[0][0]
    for idx in range(1, len(matrix[0])):
        sums[0][idx] = sums[0][idx - 1] + matrix[0][idx]
    for idx in range(1, len(matrix)):
        sums[idx][0] = sums[idx - 1][0] + matrix[idx][0]
    for row in range(1, len(matrix)):
        for col in range(1, len(matrix[0])):
            sums[row][col] = sums[row - 1][col] + sums[row][col - 1] - sums[row - 1][col - 1] + matrix[row][col]
    return sums`,
		Explanation: `Finding the maximum sum submatrix of a given size requires efficiently calculating sums of all possible submatrices. A prefix sum matrix (also called a 2D prefix sum or integral image) enables O(1) calculation of any submatrix sum.

The algorithm first builds a prefix sum matrix where sums[i][j] represents the sum of all elements from (0,0) to (i,j). This is built using the inclusion-exclusion principle: sums[i][j] = sums[i-1][j] + sums[i][j-1] - sums[i-1][j-1] + matrix[i][j]. The subtraction accounts for the double-counted region.

Once the prefix sum matrix is built, calculating any submatrix sum from (r1,c1) to (r2,c2) is O(1): sums[r2][c2] - sums[r1-1][c2] - sums[r2][c1-1] + sums[r1-1][c1-1]. The algorithm then tries all possible top-left positions for the size×size submatrix and finds the maximum sum.

Edge cases handled include: matrix smaller than size (no valid submatrix), all negative values (maximum is least negative submatrix), single-element submatrix (size=1), and matrix where maximum spans entire matrix (size equals matrix dimensions).

An alternative approach would calculate each submatrix sum directly, but that's O(size²) per submatrix. The prefix sum approach achieves O(w*h) time for building the prefix matrix and O(w*h) time for trying all positions, giving overall O(w*h) time. The space complexity is O(w*h) for the prefix sum matrix.

Think of this as creating a running total map: you precompute sums from the origin to each point, then use these to quickly calculate any rectangular region's sum by adding and subtracting the right corner values.`,
	},
	{
		ID:          213,
		Title:       "Maximize Expression",
		Description: "Write a function that takes in an array of integers and returns the largest possible value for the expression array[a] - array[b] + array[c] - array[d], where a, b, c, and d are indices of the array and a < b < c < d.",
		Difficulty:  "Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func maximizeExpression(array []int) int",
		TestCases: []TestCase{
			{Input: "array = [3, 6, 1, -3, 2, 7]", Expected: "4"},
		},
		Solution: `func maximizeExpression(array []int) int {
	if len(array) < 4 {
		return 0
	}
	maxOfA := make([]int, len(array))
	maxOfAMinusB := make([]int, len(array))
	maxOfAMinusBPlusC := make([]int, len(array))
	maxOfAMinusBPlusCMinusD := make([]int, len(array))
	maxOfA[0] = array[0]
	for i := 1; i < len(array); i++ {
		maxOfA[i] = max(maxOfA[i-1], array[i])
	}
	maxOfAMinusB[1] = maxOfA[0] - array[1]
	for i := 2; i < len(array); i++ {
		maxOfAMinusB[i] = max(maxOfAMinusB[i-1], maxOfA[i-1]-array[i])
	}
	maxOfAMinusBPlusC[2] = maxOfAMinusB[1] + array[2]
	for i := 3; i < len(array); i++ {
		maxOfAMinusBPlusC[i] = max(maxOfAMinusBPlusC[i-1], maxOfAMinusB[i-1]+array[i])
	}
	maxOfAMinusBPlusCMinusD[3] = maxOfAMinusBPlusC[2] - array[3]
	for i := 4; i < len(array); i++ {
		maxOfAMinusBPlusCMinusD[i] = max(maxOfAMinusBPlusCMinusD[i-1], maxOfAMinusBPlusC[i-1]-array[i])
	}
	return maxOfAMinusBPlusCMinusD[len(array)-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def maximizeExpression(array: List[int]) -> int:
    if len(array) < 4:
        return 0
    max_of_a = [array[0]] * len(array)
    max_of_a_minus_b = [float('-inf')] * len(array)
    max_of_a_minus_b_plus_c = [float('-inf')] * len(array)
    max_of_a_minus_b_plus_c_minus_d = [float('-inf')] * len(array)
    for i in range(1, len(array)):
        max_of_a[i] = max(max_of_a[i - 1], array[i])
    max_of_a_minus_b[1] = max_of_a[0] - array[1]
    for i in range(2, len(array)):
        max_of_a_minus_b[i] = max(max_of_a_minus_b[i - 1], max_of_a[i - 1] - array[i])
    max_of_a_minus_b_plus_c[2] = max_of_a_minus_b[1] + array[2]
    for i in range(3, len(array)):
        max_of_a_minus_b_plus_c[i] = max(max_of_a_minus_b_plus_c[i - 1], max_of_a_minus_b[i - 1] + array[i])
    max_of_a_minus_b_plus_c_minus_d[3] = max_of_a_minus_b_plus_c[2] - array[3]
    for i in range(4, len(array)):
        max_of_a_minus_b_plus_c_minus_d[i] = max(max_of_a_minus_b_plus_c_minus_d[i - 1], max_of_a_minus_b_plus_c[i - 1] - array[i])
    return max_of_a_minus_b_plus_c_minus_d[len(array) - 1]`,
		Explanation: `Finding the maximum value of the expression a - b + c - d requires tracking the maximum value achievable for each partial expression as we process the array. Dynamic programming efficiently handles this by maintaining state for each expression part.

The algorithm processes the array left to right, maintaining four DP arrays: maxOfA (maximum single element), maxOfAMinusB (maximum a-b), maxOfAMinusBPlusC (maximum a-b+c), and maxOfAMinusBPlusCMinusD (maximum a-b+c-d). For each position, it updates these arrays by either continuing the previous maximum or starting a new expression part at the current position.

The key insight is that to maximize a-b+c-d, we need to maximize a-b+c (for the first three parts) and then subtract d optimally. Similarly, to maximize a-b+c, we maximize a-b and add c. This cascading optimization naturally leads to the DP recurrence relations.

Edge cases handled include: arrays with 4 or fewer elements (some expressions undefined), all positive values (maximum uses all values), all negative values (minimum uses least negative), and arrays where optimal solution skips some elements (handled by max operations).

An alternative approach would try all combinations of choosing 4 elements, but that's O(n⁴). The DP approach achieves O(n) time by processing each element once and updating all expression states. The space complexity is O(n) for the DP arrays, though it can be optimized to O(1) by only keeping the previous values.

Think of this as building up an expression step by step: you track the best value you can achieve with 1 element (a), then 2 elements (a-b), then 3 (a-b+c), and finally 4 (a-b+c-d), always choosing the maximum at each step.`,
	},
	{
		ID:          214,
		Title:       "Dice Throws",
		Description: "You're given two positive integers representing the number of dice and the number of faces on each die. Write a function that returns the number of ways in which you can get the target sum by rolling all of the dice.",
		Difficulty:  "Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func diceThrows(numDice int, numFaces int, target int) int",
		TestCases: []TestCase{
			{Input: "numDice = 3, numFaces = 6, target = 7", Expected: "15"},
		},
		Solution: `func diceThrows(numDice int, numFaces int, target int) int {
	memoize := make(map[string]int)
	return diceThrowsHelper(numDice, numFaces, target, memoize)
}

func diceThrowsHelper(numDice int, numFaces int, target int, memoize map[string]int) int {
	if numDice == 0 {
		if target == 0 {
			return 1
		}
		return 0
	}
	if target < 0 {
		return 0
	}
	key := string(rune(numDice)) + "," + string(rune(target))
	if val, found := memoize[key]; found {
		return val
	}
	numWays := 0
	for face := 1; face < numFaces+1; face++ {
		numWays += diceThrowsHelper(numDice-1, numFaces, target-face, memoize)
	}
	memoize[key] = numWays
	return numWays
}`,
		PythonSolution: `def diceThrows(numDice: int, numFaces: int, target: int) -> int:
    memoize = {}
    return dice_throws_helper(numDice, numFaces, target, memoize)

def dice_throws_helper(num_dice, num_faces, target, memoize):
    if num_dice == 0:
        return 1 if target == 0 else 0
    if target < 0:
        return 0
    key = f"{num_dice},{target}"
    if key in memoize:
        return memoize[key]
    num_ways = 0
    for face in range(1, num_faces + 1):
        num_ways += dice_throws_helper(num_dice - 1, num_faces, target - face, memoize)
    memoize[key] = num_ways
    return num_ways`,
		Explanation: `Counting the number of ways to achieve a target sum with a given number of dice and faces requires trying all possible die outcomes and summing the ways to achieve the remaining target. Memoized recursion efficiently handles this by avoiding recomputation of subproblems.

The algorithm uses memoized recursion. For each die, it tries all possible face values (1 to numFaces). For each face value, it recursively calculates the number of ways to achieve the remaining target (target - face) with the remaining dice (numDice - 1). The base case is when numDice reaches 0: if target is also 0, we've found one way; otherwise, no way.

The key insight is that the number of ways to achieve target with numDice dice equals the sum of ways to achieve (target - face) with (numDice - 1) dice for all possible face values. Memoization ensures we don't recompute the same (numDice, target) combination multiple times.

Edge cases handled include: target too large (impossible, return 0), target too small (impossible, return 0), target exactly achievable (ways calculated), and single die (ways equals number of faces that don't exceed target).

An alternative iterative DP approach would work similarly. The memoized recursion approach achieves O(dice*target*faces) time where we consider dice*target states and faces choices for each, and O(dice*target) space for the memoization table.

Think of this as counting paths: for each die roll, you branch into all possible outcomes, and you count how many paths lead to exactly the target sum, remembering results you've already calculated to avoid recalculating.`,
	},
	{
		ID:          215,
		Title:       "Juice Bottling",
		Description: "You're given an array of integers, each of which represents the number of juice bottles that can be made from a single fruit. You're also given an integer representing the total number of fruits. Write a function that returns the maximum number of juice bottles you can make.",
		Difficulty:  "Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func juiceBottling(prices []int, totalFruits int) int",
		TestCases: []TestCase{
			{Input: "prices = [0, 2, 5, 6, 7], totalFruits = 4", Expected: "10"},
		},
		Solution: `func juiceBottling(prices []int, totalFruits int) int {
	maxProfit := make([]int, totalFruits+1)
	for i := 1; i <= totalFruits; i++ {
		for j := 1; j <= i && j < len(prices); j++ {
			maxProfit[i] = max(maxProfit[i], prices[j]+maxProfit[i-j])
		}
	}
	return maxProfit[totalFruits]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def juiceBottling(prices: List[int], totalFruits: int) -> int:
    max_profit = [0] * (totalFruits + 1)
    for i in range(1, totalFruits + 1):
        for j in range(1, min(i + 1, len(prices))):
            max_profit[i] = max(max_profit[i], prices[j] + max_profit[i - j])
    return max_profit[totalFruits]`,
		Explanation: `Maximizing profit from selling fruits in bottles requires finding the optimal way to divide fruits into bottles of various sizes. Dynamic programming efficiently solves this by building up optimal solutions for larger quantities from smaller ones.

The algorithm uses DP where maxProfit[i] represents the maximum profit achievable by selling i fruits. For each quantity i, it tries all possible bottle sizes j (from 1 to min(i, len(prices))). If using a bottle of size j gives profit prices[j] + maxProfit[i-j], and this exceeds the current maxProfit[i], it updates the value.

The key insight is that the optimal way to sell i fruits is to try all possible first bottle sizes and combine each with the optimal way to sell the remaining fruits. This creates overlapping subproblems that DP handles efficiently.

Edge cases handled include: totalFruits is 0 (profit is 0), totalFruits less than smallest bottle (no valid sale), prices array with zeros (handled correctly), and prices where larger bottles aren't always better (DP finds optimal combination).

An alternative greedy approach would always choose the most profitable bottle size, but that fails when combinations are better. The DP approach is guaranteed to find the optimal solution. The time complexity is O(n²) where n is totalFruits, as we check all bottle sizes for each quantity. The space complexity is O(n) for the maxProfit array.

Think of this as finding the best way to package items: for each quantity, you try all possible first package sizes and see which combination with the optimal packaging of the remainder gives you the best total profit.`,
	},
	{
		ID:          216,
		Title:       "Dijkstra's Algorithm",
		Description: "You're given an integer start and a list edges of pairs of integers representing the edges of a weighted, directed graph. Write a function that returns the shortest distance from the start vertex to all other vertices in the graph.",
		Difficulty:  "Hard",
		Topic:       "Graphs",
		Signature:   "func dijkstrasAlgorithm(start int, edges [][]int) []int",
		TestCases: []TestCase{
			{Input: "start = 0, edges = [[[1, 7]], [[2, 6], [3, 20], [4, 3]], [[3, 14]], [[4, 2]], [], []]", Expected: "[0, 7, 13, 27, 10, -1]"},
		},
		Solution: `func dijkstrasAlgorithm(start int, edges [][]int) []int {
	numberOfVertices := len(edges)
	minDistances := make([]int, numberOfVertices)
	for i := range minDistances {
		minDistances[i] = math.MaxInt32
	}
	minDistances[start] = 0
	visited := make(map[int]bool)
	for len(visited) != numberOfVertices {
		vertex, currentMinDistance := getVertexWithMinDistance(minDistances, visited)
		if currentMinDistance == math.MaxInt32 {
			break
		}
		visited[vertex] = true
		for _, edge := range edges[vertex] {
			destination, distanceToDestination := edge[0], edge[1]
			if visited[destination] {
				continue
			}
			newPathDistance := currentMinDistance + distanceToDestination
			currentDestinationDistance := minDistances[destination]
			if newPathDistance < currentDestinationDistance {
				minDistances[destination] = newPathDistance
			}
		}
	}
	for i := range minDistances {
		if minDistances[i] == math.MaxInt32 {
			minDistances[i] = -1
		}
	}
	return minDistances
}

func getVertexWithMinDistance(distances []int, visited map[int]bool) (int, int) {
	currentMinDistance := math.MaxInt32
	vertex := -1
	for vertexIdx, distance := range distances {
		if visited[vertexIdx] {
			continue
		}
		if distance <= currentMinDistance {
			vertex = vertexIdx
			currentMinDistance = distance
		}
	}
	return vertex, currentMinDistance
}`,
		PythonSolution: `def dijkstrasAlgorithm(start: int, edges: List[List[List[int]]]) -> List[int]:
    number_of_vertices = len(edges)
    min_distances = [float('inf')] * number_of_vertices
    min_distances[start] = 0
    visited = set()
    while len(visited) != number_of_vertices:
        vertex, current_min_distance = get_vertex_with_min_distance(min_distances, visited)
        if current_min_distance == float('inf'):
            break
        visited.add(vertex)
        for edge in edges[vertex]:
            destination, distance_to_destination = edge
            if destination in visited:
                continue
            new_path_distance = current_min_distance + distance_to_destination
            current_destination_distance = min_distances[destination]
            if new_path_distance < current_destination_distance:
                min_distances[destination] = new_path_distance
    return [-1 if d == float('inf') else d for d in min_distances]

def get_vertex_with_min_distance(distances, visited):
    current_min_distance = float('inf')
    vertex = -1
    for vertex_idx, distance in enumerate(distances):
        if vertex_idx in visited:
            continue
        if distance <= current_min_distance:
            vertex = vertex_idx
            current_min_distance = distance
    return vertex, current_min_distance`,
		Explanation: `Dijkstra's algorithm finds the shortest paths from a source vertex to all other vertices in a weighted graph. It uses a greedy approach, always processing the unvisited vertex with the minimum known distance first.

The algorithm maintains a distance array initialized to infinity (except the start vertex, which is 0) and a visited set. In each iteration, it selects the unvisited vertex with the minimum distance, marks it as visited, and relaxes all its outgoing edges. Relaxation means updating the distance to a neighbor if a shorter path through the current vertex is found.

The greedy choice is optimal because once a vertex is processed, its shortest distance is known and won't change. By always choosing the minimum distance vertex, we ensure we're processing vertices in order of their shortest distances from the source.

Edge cases handled include: unreachable vertices (distance remains infinity, returned as -1), graphs with negative edge weights (algorithm may fail, but handles gracefully), single vertex graphs (returns [0]), and graphs with cycles (handled correctly as long as no negative cycles).

An alternative approach using a min-heap would improve time complexity to O((v+e) log v), but the array-based approach is simpler for dense graphs. The time complexity is O(v² + e) where v is vertices and e is edges, as we select minimum v times (O(v²)) and relax e edges. The space complexity is O(v) for distance and visited arrays.

Think of this as exploring a city: you always visit the closest unvisited location first, and when you find a shorter route to a location, you update your map, continuing until you've found the shortest route to every location.`,
	},
	{
		ID:          217,
		Title:       "Topological Sort",
		Description: "You're given a list of arbitrary jobs that need to be completed. These jobs are represented by distinct integers. You're also given a list of dependencies, which is represented by a list of pairs, where the first job in the pair is a prerequisite of the second job. In other words, the second job depends on the first; it can only be completed once the first job is completed. Write a function that takes in a list of jobs and a list of dependencies and returns a list containing a valid order in which the given jobs can be completed, or an empty array if no such order exists.",
		Difficulty:  "Hard",
		Topic:       "Graphs",
		Signature:   "func topologicalSort(jobs []int, deps [][]int) []int",
		TestCases: []TestCase{
			{Input: "jobs = [1, 2, 3, 4], deps = [[1, 2], [1, 3], [3, 2], [4, 2], [4, 3]]", Expected: "[1, 4, 3, 2]"},
		},
		Solution: `func topologicalSort(jobs []int, deps [][]int) []int {
	jobGraph := createJobGraph(jobs, deps)
	return getOrderedJobs(jobGraph)
}

func createJobGraph(jobs []int, deps [][]int) *JobGraph {
	graph := NewJobGraph(jobs)
	for _, dep := range deps {
		prereq, job := dep[0], dep[1]
		graph.addPrereq(job, prereq)
	}
	return graph
}

func getOrderedJobs(graph *JobGraph) []int {
	orderedJobs := []int{}
	nodesWithNoPrereqs := []*JobNode{}
	for _, node := range graph.nodes {
		if node.numOfPrereqs == 0 {
			nodesWithNoPrereqs = append(nodesWithNoPrereqs, node)
		}
	}
	for len(nodesWithNoPrereqs) > 0 {
		node := nodesWithNoPrereqs[0]
		nodesWithNoPrereqs = nodesWithNoPrereqs[1:]
		orderedJobs = append(orderedJobs, node.job)
		removeDeps(node, &nodesWithNoPrereqs)
	}
	graphHasEdges := false
	for _, node := range graph.nodes {
		if node.numOfPrereqs > 0 {
			graphHasEdges = true
		}
	}
	if graphHasEdges {
		return []int{}
	}
	return orderedJobs
}

func removeDeps(node *JobNode, nodesWithNoPrereqs *[]*JobNode) {
	for len(node.deps) > 0 {
		dep := node.deps[0]
		node.deps = node.deps[1:]
		dep.numOfPrereqs--
		if dep.numOfPrereqs == 0 {
			*nodesWithNoPrereqs = append(*nodesWithNoPrereqs, dep)
		}
	}
}

type JobGraph struct {
	nodes map[int]*JobNode
}

type JobNode struct {
	job          int
	prereqs      []*JobNode
	deps         []*JobNode
	numOfPrereqs int
}

func NewJobGraph(jobs []int) *JobGraph {
	graph := &JobGraph{nodes: make(map[int]*JobNode)}
	for _, job := range jobs {
		graph.nodes[job] = &JobNode{job: job, prereqs: []*JobNode{}, deps: []*JobNode{}}
	}
	return graph
}

func (graph *JobGraph) addPrereq(job int, prereq int) {
	jobNode := graph.getNode(job)
	prereqNode := graph.getNode(prereq)
	jobNode.prereqs = append(jobNode.prereqs, prereqNode)
	jobNode.numOfPrereqs++
	prereqNode.deps = append(prereqNode.deps, jobNode)
}

func (graph *JobGraph) getNode(job int) *JobNode {
	if _, found := graph.nodes[job]; !found {
		graph.nodes[job] = &JobNode{job: job, prereqs: []*JobNode{}, deps: []*JobNode{}}
	}
	return graph.nodes[job]
}`,
		PythonSolution: `def topologicalSort(jobs: List[int], deps: List[List[int]]) -> List[int]:
    job_graph = create_job_graph(jobs, deps)
    return get_ordered_jobs(job_graph)

def create_job_graph(jobs, deps):
    graph = JobGraph(jobs)
    for dep in deps:
        prereq, job = dep
        graph.add_prereq(job, prereq)
    return graph

def get_ordered_jobs(graph):
    ordered_jobs = []
    nodes_with_no_prereqs = [node for node in graph.nodes.values() if node.num_of_prereqs == 0]
    while nodes_with_no_prereqs:
        node = nodes_with_no_prereqs.pop(0)
        ordered_jobs.append(node.job)
        remove_deps(node, nodes_with_no_prereqs)
    graph_has_edges = any(node.num_of_prereqs > 0 for node in graph.nodes.values())
    return [] if graph_has_edges else ordered_jobs

def remove_deps(node, nodes_with_no_prereqs):
    while node.deps:
        dep = node.deps.pop(0)
        dep.num_of_prereqs -= 1
        if dep.num_of_prereqs == 0:
            nodes_with_no_prereqs.append(dep)

class JobGraph:
    def __init__(self, jobs):
        self.nodes = {}
        for job in jobs:
            self.nodes[job] = JobNode(job)
    
    def add_prereq(self, job, prereq):
        job_node = self.get_node(job)
        prereq_node = self.get_node(prereq)
        job_node.prereqs.append(prereq_node)
        job_node.num_of_prereqs += 1
        prereq_node.deps.append(job_node)
    
    def get_node(self, job):
        if job not in self.nodes:
            self.nodes[job] = JobNode(job)
        return self.nodes[job]

class JobNode:
    def __init__(self, job):
        self.job = job
        self.prereqs = []
        self.deps = []
        self.num_of_prereqs = 0`,
		Explanation: `Topological sorting orders vertices in a directed acyclic graph (DAG) such that for every directed edge (u, v), u appears before v in the ordering. Kahn's algorithm efficiently computes this by repeatedly removing vertices with no incoming edges.

The algorithm builds a graph representation tracking prerequisites (incoming edges) and dependencies (outgoing edges) for each job. It starts with jobs that have no prerequisites (numOfPrereqs == 0) and adds them to the result. When a job is processed, it removes its outgoing edges by decrementing the prerequisite count of dependent jobs. If a dependent job's prerequisite count reaches zero, it becomes available and is added to the queue.

This process continues until all jobs are processed or a cycle is detected (jobs remain with prerequisites). If jobs remain with prerequisites, the graph contains a cycle and no valid topological order exists.

Edge cases handled include: graphs with no edges (all jobs independent, any order valid), graphs with cycles (detected when jobs remain with prerequisites), single job (returns that job), and disconnected components (all processed correctly).

An alternative DFS-based approach would also work, but Kahn's algorithm is more intuitive and naturally detects cycles. The time complexity is O(j+d) where j is jobs and d is dependencies, as we process each job and dependency once. The space complexity is O(j+d) for the graph representation.

Think of this as course prerequisites: you can only take a course after completing its prerequisites. You start with courses that have no prerequisites, and as you complete them, other courses become available, continuing until all courses are scheduled or you find an impossible requirement (cycle).`,
	},
	{
		ID:          218,
		Title:       "Kruskal's Algorithm",
		Description: "You're given a list of edges representing a weighted, undirected graph with at least one node. Write a function that returns the minimum spanning tree of the graph. The minimum spanning tree is the subset of edges that connects all nodes in the graph with the minimum total edge weight.",
		Difficulty:  "Hard",
		Topic:       "Graphs",
		Signature:   "func kruskalsAlgorithm(edges [][]int) [][]int",
		TestCases: []TestCase{
			{Input: "edges = [[1, 2, 3], [1, 3, 4], [4, 2, 6], [5, 2, 2], [2, 3, 5], [3, 5, 7], [5, 4, 5]]", Expected: "Minimum spanning tree edges"},
		},
		Solution: `import "sort"

func kruskalsAlgorithm(edges [][]int) [][]int {
	sort.Slice(edges, func(i, j int) bool {
		return edges[i][2] < edges[j][2]
	})
	parent := make(map[int]int)
	rank := make(map[int]int)
	mst := [][]int{}
	for _, edge := range edges {
		node1, node2 := edge[0], edge[1]
		if find(node1, parent) != find(node2, parent) {
			mst = append(mst, edge)
			union(node1, node2, parent, rank)
		}
	}
	return mst
}

func find(node int, parent map[int]int) int {
	if parent[node] != node {
		parent[node] = find(parent[node], parent)
	}
	if _, found := parent[node]; !found {
		parent[node] = node
	}
	return parent[node]
}

func union(node1 int, node2 int, parent map[int]int, rank map[int]int) {
	root1 := find(node1, parent)
	root2 := find(node2, parent)
	if root1 == root2 {
		return
	}
	if rank[root1] < rank[root2] {
		parent[root1] = root2
	} else if rank[root1] > rank[root2] {
		parent[root2] = root1
	} else {
		parent[root2] = root1
		rank[root1]++
	}
}`,
		PythonSolution: `def kruskalsAlgorithm(edges: List[List[int]]) -> List[List[int]]:
    edges.sort(key=lambda x: x[2])
    parent = {}
    rank = {}
    mst = []
    for edge in edges:
        node1, node2 = edge[0], edge[1]
        if find(node1, parent) != find(node2, parent):
            mst.append(edge)
            union(node1, node2, parent, rank)
    return mst

def find(node, parent):
    if node not in parent:
        parent[node] = node
    if parent[node] != node:
        parent[node] = find(parent[node], parent)
    return parent[node]

def union(node1, node2, parent, rank):
    root1 = find(node1, parent)
    root2 = find(node2, parent)
    if root1 == root2:
        return
    if rank.get(root1, 0) < rank.get(root2, 0):
        parent[root1] = root2
    elif rank.get(root1, 0) > rank.get(root2, 0):
        parent[root2] = root1
    else:
        parent[root2] = root1
        rank[root1] = rank.get(root1, 0) + 1`,
		Explanation: `Kruskal's algorithm finds the minimum spanning tree (MST) of a graph by greedily adding the smallest edges that don't create cycles. Union-Find (Disjoint Set Union) efficiently detects cycles, making this approach optimal.

The algorithm first sorts all edges by weight. Then it processes edges in ascending order, adding each edge to the MST if it doesn't form a cycle. Union-Find tracks which vertices are in the same connected component: if two vertices of an edge have the same root, adding the edge would create a cycle, so it's skipped.

The greedy choice is optimal because if an edge is the smallest available and doesn't create a cycle, it must be part of the MST. By processing edges in sorted order and using Union-Find for cycle detection, we efficiently build the MST.

Edge cases handled include: disconnected graphs (MST connects all reachable vertices from each component), graphs with duplicate edge weights (any valid MST), single-vertex graphs (empty MST), and graphs where all edges have same weight (any spanning tree works).

An alternative approach would use Prim's algorithm, which grows the MST from a starting vertex. Kruskal's is often simpler to implement. The time complexity is O(e log e) for sorting edges, plus O(e α(v)) for Union-Find operations, where α is the inverse Ackermann function (effectively constant). The space complexity is O(v) for Union-Find data structures.

Think of this as building the cheapest network: you sort all possible connections by cost, then add them one by one, skipping any that would create a redundant connection (cycle), until everything is connected.`,
	},
	{
		ID:          219,
		Title:       "Prim's Algorithm",
		Description: "You're given a list of edges representing a weighted, undirected graph with at least one node. Write a function that returns the minimum spanning tree of the graph. The minimum spanning tree is the subset of edges that connects all nodes in the graph with the minimum total edge weight.",
		Difficulty:  "Hard",
		Topic:       "Graphs",
		Signature:   "func primsAlgorithm(edges [][]int) [][]int",
		TestCases: []TestCase{
			{Input: "edges = [[1, 2, 3], [1, 3, 4], [4, 2, 6], [5, 2, 2], [2, 3, 5], [3, 5, 7], [5, 4, 5]]", Expected: "Minimum spanning tree edges"},
		},
		Solution: `import "container/heap"

type Edge struct {
	node1   int
	node2   int
	weight  int
}

type EdgeHeap []Edge

func (h EdgeHeap) Len() int           { return len(h) }
func (h EdgeHeap) Less(i, j int) bool { return h[i].weight < h[j].weight }
func (h EdgeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *EdgeHeap) Push(x interface{}) { *h = append(*h, x.(Edge)) }
func (h *EdgeHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func primsAlgorithm(edges [][]int) [][]int {
	adjList := make(map[int][][]int)
	allNodes := make(map[int]bool)
	for _, edge := range edges {
		node1, node2, weight := edge[0], edge[1], edge[2]
		adjList[node1] = append(adjList[node1], []int{node2, weight})
		adjList[node2] = append(adjList[node2], []int{node1, weight})
		allNodes[node1] = true
		allNodes[node2] = true
	}
	startNode := edges[0][0]
	visited := make(map[int]bool)
	mst := [][]int{}
	edgeHeap := &EdgeHeap{}
	heap.Init(edgeHeap)
	visited[startNode] = true
	for _, neighbor := range adjList[startNode] {
		heap.Push(edgeHeap, Edge{startNode, neighbor[0], neighbor[1]})
	}
	for len(visited) < len(allNodes) {
		edge := heap.Pop(edgeHeap).(Edge)
		if visited[edge.node2] {
			continue
		}
		mst = append(mst, []int{edge.node1, edge.node2, edge.weight})
		visited[edge.node2] = true
		for _, neighbor := range adjList[edge.node2] {
			if !visited[neighbor[0]] {
				heap.Push(edgeHeap, Edge{edge.node2, neighbor[0], neighbor[1]})
			}
		}
	}
	return mst
}`,
		PythonSolution: `import heapq

def primsAlgorithm(edges: List[List[int]]) -> List[List[int]]:
    adj_list = {}
    all_nodes = set()
    for edge in edges:
        node1, node2, weight = edge
        if node1 not in adj_list:
            adj_list[node1] = []
        if node2 not in adj_list:
            adj_list[node2] = []
        adj_list[node1].append([node2, weight])
        adj_list[node2].append([node1, weight])
        all_nodes.add(node1)
        all_nodes.add(node2)
    start_node = edges[0][0]
    visited = {start_node}
    mst = []
    edge_heap = []
    for neighbor in adj_list[start_node]:
        heapq.heappush(edge_heap, (neighbor[1], start_node, neighbor[0]))
    while len(visited) < len(all_nodes):
        weight, node1, node2 = heapq.heappop(edge_heap)
        if node2 in visited:
            continue
        mst.append([node1, node2, weight])
        visited.add(node2)
        for neighbor in adj_list[node2]:
            if neighbor[0] not in visited:
                heapq.heappush(edge_heap, (neighbor[1], node2, neighbor[0]))
    return mst`,
		Explanation: `Prim's algorithm finds the minimum spanning tree by growing it from an arbitrary starting vertex. It uses a min-heap to efficiently select the minimum-weight edge connecting the current MST to an unvisited vertex.

The algorithm starts from an arbitrary vertex and adds it to the visited set. All edges from this vertex are added to a min-heap. In each iteration, it extracts the minimum-weight edge from the heap. If the edge connects a visited vertex to an unvisited vertex, it adds the edge to the MST and the unvisited vertex to the visited set, then adds all edges from the new vertex to the heap.

This greedy approach ensures that at each step, we add the cheapest edge that expands the MST, guaranteeing optimality. The heap ensures we always process the minimum-weight available edge efficiently.

Edge cases handled include: disconnected graphs (MST for each component), graphs with duplicate edge weights (any valid MST), single-vertex graphs (empty MST), and graphs where MST is a single path (all edges added sequentially).

An alternative approach would use Kruskal's algorithm, which processes all edges sorted by weight. Prim's is often more efficient for dense graphs. The time complexity is O(e log v) where e is edges and v is vertices, as each edge is processed once and heap operations are O(log v). The space complexity is O(v+e) for the heap and adjacency list.

Think of this as building a network starting from one location: you always add the cheapest connection to a new location, expanding your network one location at a time until everything is connected.`,
	},
	{
		ID:          220,
		Title:       "Boggle Board",
		Description: "You're given a two-dimensional array of potentially unequal height and width containing letters; this matrix represents a boggle board. You're also given a list of words. Write a function that returns an array of all the words contained in the boggle board. A word is constructed in the boggle board by connecting adjacent (horizontally, vertically, or diagonally adjacent) letters, without using any single letter at a given position more than once; while a word can of course have repeated letters, those repeated letters must come from different positions in the boggle board.",
		Difficulty:  "Hard",
		Topic:       "Graphs",
		Signature:   "func boggleBoard(board [][]rune, words []string) []string",
		TestCases: []TestCase{
			{Input: "board = [[\"t\", \"h\", \"i\", \"s\", \"i\", \"s\", \"a\"], [\"s\", \"i\", \"m\", \"p\", \"l\", \"e\", \"x\"], [\"b\", \"x\", \"x\", \"x\", \"x\", \"e\", \"b\"], [\"x\", \"o\", \"g\", \"g\", \"l\", \"x\", \"o\"], [\"x\", \"x\", \"x\", \"D\", \"T\", \"r\", \"a\"], [\"R\", \"E\", \"P\", \"E\", \"A\", \"d\", \"x\"], [\"x\", \"x\", \"x\", \"x\", \"x\", \"x\", \"x\"], [\"N\", \"O\", \"T\", \"R\", \"E\", \"-\", \"P\"], [\"x\", \"x\", \"D\", \"E\", \"T\", \"A\", \"E\"]], words = [\"this\", \"is\", \"not\", \"a\", \"simple\", \"boggle\", \"board\", \"test\", \"REPEATED\", \"NOTRE-PEATED\"]", Expected: "[\"this\", \"is\", \"a\", \"simple\", \"boggle\", \"board\", \"NOTRE-PEATED\"]"},
		},
		Solution: `func boggleBoard(board [][]rune, words []string) []string {
	trie := NewTrie()
	for _, word := range words {
		trie.Add(word)
	}
	finalWords := make(map[string]bool)
	visited := make([][]bool, len(board))
	for i := range visited {
		visited[i] = make([]bool, len(board[0]))
	}
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			explore(i, j, board, trie.root, visited, finalWords)
		}
	}
	result := []string{}
	for word := range finalWords {
		result = append(result, word)
	}
	return result
}

func explore(i int, j int, board [][]rune, trieNode *TrieNode, visited [][]bool, finalWords map[string]bool) {
	if visited[i][j] {
		return
	}
	letter := board[i][j]
	if _, found := trieNode.children[letter]; !found {
		return
	}
	visited[i][j] = true
	trieNode = trieNode.children[letter]
	if trieNode.endWord != "" {
		finalWords[trieNode.endWord] = true
	}
	neighbors := getNeighbors(i, j, board)
	for _, neighbor := range neighbors {
		explore(neighbor[0], neighbor[1], board, trieNode, visited, finalWords)
	}
	visited[i][j] = false
}

func getNeighbors(i int, j int, board [][]rune) [][]int {
	neighbors := [][]int{}
	if i > 0 && j > 0 {
		neighbors = append(neighbors, []int{i - 1, j - 1})
	}
	if i > 0 {
		neighbors = append(neighbors, []int{i - 1, j})
	}
	if i > 0 && j < len(board[0])-1 {
		neighbors = append(neighbors, []int{i - 1, j + 1})
	}
	if j < len(board[0])-1 {
		neighbors = append(neighbors, []int{i, j + 1})
	}
	if i < len(board)-1 && j < len(board[0])-1 {
		neighbors = append(neighbors, []int{i + 1, j + 1})
	}
	if i < len(board)-1 {
		neighbors = append(neighbors, []int{i + 1, j})
	}
	if i < len(board)-1 && j > 0 {
		neighbors = append(neighbors, []int{i + 1, j - 1})
	}
	if j > 0 {
		neighbors = append(neighbors, []int{i, j - 1})
	}
	return neighbors
}

type TrieNode struct {
	children map[rune]*TrieNode
	endWord  string
}

type Trie struct {
	root *TrieNode
}

func NewTrie() *Trie {
	return &Trie{root: &TrieNode{children: make(map[rune]*TrieNode)}}
}

func (t *Trie) Add(word string) {
	node := t.root
	for _, char := range word {
		if _, found := node.children[char]; !found {
			node.children[char] = &TrieNode{children: make(map[rune]*TrieNode)}
		}
		node = node.children[char]
	}
	node.endWord = word
}`,
		PythonSolution: `def boggleBoard(board: List[List[str]], words: List[str]) -> List[str]:
    trie = Trie()
    for word in words:
        trie.add(word)
    final_words = {}
    visited = [[False for _ in range(len(board[0]))] for _ in range(len(board))]
    for i in range(len(board)):
        for j in range(len(board[0])):
            explore(i, j, board, trie.root, visited, final_words)
    return list(final_words.keys())

def explore(i, j, board, trie_node, visited, final_words):
    if visited[i][j]:
        return
    letter = board[i][j]
    if letter not in trie_node.children:
        return
    visited[i][j] = True
    trie_node = trie_node.children[letter]
    if trie_node.end_word:
        final_words[trie_node.end_word] = True
    neighbors = get_neighbors(i, j, board)
    for neighbor in neighbors:
        explore(neighbor[0], neighbor[1], board, trie_node, visited, final_words)
    visited[i][j] = False

def get_neighbors(i, j, board):
    neighbors = []
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < len(board) and 0 <= nj < len(board[0]):
            neighbors.append([ni, nj])
    return neighbors

class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_word = None

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def add(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.end_word = word`,
		Explanation: `Finding all words from a dictionary in a Boggle board requires checking all possible paths through the board. A trie data structure enables efficient prefix matching, allowing early termination when a path cannot form any word.

The algorithm first builds a trie from all words in the dictionary. Then, for each cell in the board, it performs a depth-first search (DFS) that traverses the trie simultaneously. As it explores paths through the board, it follows corresponding paths in the trie. When it reaches a node marked with a complete word (endWord set), it adds that word to the results.

Backtracking is crucial: after exploring a path, the algorithm unmarks the cell as visited, allowing it to be used in other paths. This ensures all possible word formations are found, including words that share board paths.

Edge cases handled include: words not in board (not found), words appearing multiple times (all occurrences found), words that are prefixes of other words (both found correctly), and boards with repeated letters (handled by backtracking).

An alternative approach would check each word against all possible board paths, but that's exponential. The trie approach prunes impossible paths early. The time complexity is O(w*h*8^L) in the worst case where w and h are board dimensions and L is maximum word length, though pruning makes it much faster in practice. The space complexity is O(n*s) for the trie where n is number of words and s is average word length.

Think of this as solving a word search puzzle: you build a dictionary tree, then for each starting position, you explore paths through the board while following the dictionary tree, collecting words as you find complete paths.`,
	},
	{
		ID:          221,
		Title:       "Apartment Hunting",
		Description: "You're looking to move into a new apartment, and you're given a list of blocks where each block contains an apartment that you could move into. In order to pick your apartment, you want to optimize its location. You also have a list of requirements: a list of buildings that are important to you. For instance, you might value having a school and a gym near your apartment. The list of blocks that you have contains information at every block about all of the buildings that are present at that block. For instance, for every block, you might know whether a school, a pool, an office, and a gym are present. In order to optimize your life, you want to minimize the farthest distance you'd have to walk from your apartment to reach any of your required buildings. Write a function that takes in a list of blocks and a list of your required buildings and returns the location (the index) of the block that's most optimal for you.",
		Difficulty:  "Very Hard",
		Topic:       "Arrays",
		Signature:   "func apartmentHunting(blocks []map[string]bool, reqs []string) int",
		TestCases: []TestCase{
			{Input: "blocks = [{\"gym\": false, \"school\": true, \"store\": false}, {\"gym\": true, \"school\": false, \"store\": false}, {\"gym\": true, \"school\": true, \"store\": false}, {\"gym\": false, \"school\": true, \"store\": false}, {\"gym\": false, \"school\": true, \"store\": true}], reqs = [\"gym\", \"school\", \"store\"]", Expected: "3"},
		},
		Solution: `func apartmentHunting(blocks []map[string]bool, reqs []string) int {
	minDistancesFromBlocks := [][]int{}
	for _, req := range reqs {
		minDistancesFromBlocks = append(minDistancesFromBlocks, getMinDistances(blocks, req))
	}
	maxDistancesAtBlocks := getMaxDistancesAtBlocks(blocks, minDistancesFromBlocks)
	return getIdxAtMinValue(maxDistancesAtBlocks)
}

func getMinDistances(blocks []map[string]bool, req string) []int {
	minDistances := make([]int, len(blocks))
	closestReqIdx := math.MaxInt32
	for i := 0; i < len(blocks); i++ {
		if blocks[i][req] {
			closestReqIdx = i
		}
		minDistances[i] = distanceBetween(i, closestReqIdx)
	}
	for i := len(blocks) - 1; i >= 0; i-- {
		if blocks[i][req] {
			closestReqIdx = i
		}
		minDistances[i] = min(minDistances[i], distanceBetween(i, closestReqIdx))
	}
	return minDistances
}

func getMaxDistancesAtBlocks(blocks []map[string]bool, minDistancesFromBlocks [][]int) []int {
	maxDistancesAtBlocks := make([]int, len(blocks))
	for i := 0; i < len(blocks); i++ {
		maxDistancesAtBlocks[i] = math.MinInt32
		for j := 0; j < len(minDistancesFromBlocks); j++ {
			maxDistancesAtBlocks[i] = max(maxDistancesAtBlocks[i], minDistancesFromBlocks[j][i])
		}
	}
	return maxDistancesAtBlocks
}

func getIdxAtMinValue(array []int) int {
	idxAtMinValue := 0
	minValue := array[0]
	for i := 1; i < len(array); i++ {
		currentValue := array[i]
		if currentValue < minValue {
			minValue = currentValue
			idxAtMinValue = i
		}
	}
	return idxAtMinValue
}

func distanceBetween(a int, b int) int {
	if a > b {
		return a - b
	}
	return b - a
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def apartmentHunting(blocks: List[dict], reqs: List[str]) -> int:
    min_distances_from_blocks = [get_min_distances(blocks, req) for req in reqs]
    max_distances_at_blocks = get_max_distances_at_blocks(blocks, min_distances_from_blocks)
    return get_idx_at_min_value(max_distances_at_blocks)

def get_min_distances(blocks, req):
    min_distances = [0] * len(blocks)
    closest_req_idx = float('inf')
    for i in range(len(blocks)):
        if blocks[i][req]:
            closest_req_idx = i
        min_distances[i] = distance_between(i, closest_req_idx)
    for i in range(len(blocks) - 1, -1, -1):
        if blocks[i][req]:
            closest_req_idx = i
        min_distances[i] = min(min_distances[i], distance_between(i, closest_req_idx))
    return min_distances

def get_max_distances_at_blocks(blocks, min_distances_from_blocks):
    max_distances_at_blocks = [0] * len(blocks)
    for i in range(len(blocks)):
        max_distances_at_blocks[i] = max([distances[i] for distances in min_distances_from_blocks])
    return max_distances_at_blocks

def get_idx_at_min_value(array):
    idx_at_min_value = 0
    min_value = array[0]
    for i in range(1, len(array)):
        current_value = array[i]
        if current_value < min_value:
            min_value = current_value
            idx_at_min_value = i
    return idx_at_min_value

def distance_between(a, b):
    return abs(a - b)`,
		Explanation: `Finding the optimal apartment location requires minimizing the maximum distance to any requirement. This is a minimax optimization problem where we want to minimize the worst-case distance.

The algorithm computes the minimum distance from each block to each requirement by scanning the blocks array twice (left to right and right to left) to find the closest requirement in each direction. For each block, it takes the maximum distance across all requirements (the worst-case distance). Finally, it selects the block with the minimum maximum distance.

This approach ensures that no matter which requirement you need, the chosen block minimizes the worst-case travel distance. The two-pass scanning efficiently computes closest requirements without needing to check all blocks for each requirement.

Edge cases handled include: blocks with all requirements (distance 0 for all), blocks with no requirements (distance infinity), single requirement (minimizes distance to that requirement), and requirements at block boundaries (handled correctly by two-pass scanning).

An alternative approach would compute distances for all block-requirement pairs, but that's O(b*r) which is the same. The current approach is more intuitive. The time complexity is O(b*r) where b is blocks and r is requirements, as we compute distances for each requirement at each block. The space complexity is O(b*r) for storing minimum distances.

Think of this as choosing a meeting location: you want a place where the farthest any attendee has to travel is minimized, so you check the distance from each candidate location to each attendee, find the worst-case distance for each location, and choose the location with the best worst case.`,
	},
	{
		ID:          222,
		Title:       "Calendar Matching",
		Description: "Imagine that you want to schedule a meeting of a certain duration with a coworker. You have access to your calendar and your coworker's calendar (both of which contain your respective meetings for the day, in the form of [startTime, endTime]), as well as both of your daily bounds (i.e., the earliest and latest times at which you're available for meetings every day, in the form of [earliestTime, latestTime]). Write a function that takes in your calendar, your daily bounds, your coworker's calendar, your coworker's daily bounds, and the duration of the meeting that you want to schedule, and returns a list of all the time blocks (in the form of [startTime, endTime]) during which you could schedule the meeting, ordered from earliest time block to latest.",
		Difficulty:  "Very Hard",
		Topic:       "Arrays",
		Signature:   "func calendarMatching(calendar1 [][]string, dailyBounds1 []string, calendar2 [][]string, dailyBounds2 []string, meetingDuration int) [][]string",
		TestCases: []TestCase{
			{Input: "calendar1 = [[\"9:00\", \"10:30\"], [\"12:00\", \"13:00\"], [\"16:00\", \"18:00\"]], dailyBounds1 = [\"9:00\", \"20:00\"], calendar2 = [[\"10:00\", \"11:30\"], [\"12:30\", \"14:30\"], [\"14:30\", \"15:00\"], [\"16:00\", \"17:00\"]], dailyBounds2 = [\"10:00\", \"18:30\"], meetingDuration = 30", Expected: "[[\"11:30\", \"12:00\"], [\"15:00\", \"16:00\"], [\"18:00\", \"18:30\"]]"},
		},
		Solution: `func calendarMatching(calendar1 [][]string, dailyBounds1 []string, calendar2 [][]string, dailyBounds2 []string, meetingDuration int) [][]string {
	updatedCalendar1 := updateCalendar(calendar1, dailyBounds1)
	updatedCalendar2 := updateCalendar(calendar2, dailyBounds2)
	mergedCalendar := mergeCalendars(updatedCalendar1, updatedCalendar2)
	flattenedCalendar := flattenCalendar(mergedCalendar)
	return getMatchingAvailabilities(flattenedCalendar, meetingDuration)
}

func updateCalendar(calendar [][]string, dailyBounds []string) [][]int {
	updated := [][]int{}
	updated = append(updated, []int{timeToMinutes(dailyBounds[0]), timeToMinutes(dailyBounds[0])})
	updated = append(updated, timeToMinutesArray(calendar)...)
	updated = append(updated, []int{timeToMinutes(dailyBounds[1]), timeToMinutes(dailyBounds[1])})
	return updated
}

func mergeCalendars(calendar1 [][]int, calendar2 [][]int) [][]int {
	merged := [][]int{}
	i, j := 0, 0
	for i < len(calendar1) && j < len(calendar2) {
		meeting1, meeting2 := calendar1[i], calendar2[j]
		if meeting1[0] < meeting2[0] {
			merged = append(merged, meeting1)
			i++
		} else {
			merged = append(merged, meeting2)
			j++
		}
	}
	for i < len(calendar1) {
		merged = append(merged, calendar1[i])
		i++
	}
	for j < len(calendar2) {
		merged = append(merged, calendar2[j])
		j++
	}
	return merged
}

func flattenCalendar(calendar [][]int) [][]int {
	flattened := [][]int{calendar[0]}
	for i := 1; i < len(calendar); i++ {
		currentMeeting := calendar[i]
		previousMeeting := flattened[len(flattened)-1]
		currentStart, currentEnd := currentMeeting[0], currentMeeting[1]
		previousStart, previousEnd := previousMeeting[0], previousMeeting[1]
		if previousEnd >= currentStart {
			newPreviousMeeting := []int{previousStart, max(previousEnd, currentEnd)}
			flattened[len(flattened)-1] = newPreviousMeeting
		} else {
			flattened = append(flattened, currentMeeting)
		}
	}
	return flattened
}

func getMatchingAvailabilities(calendar [][]int, meetingDuration int) [][]string {
	matchingAvailabilities := [][]string{}
	for i := 1; i < len(calendar); i++ {
		start := calendar[i-1][1]
		end := calendar[i][0]
		availabilityDuration := end - start
		if availabilityDuration >= meetingDuration {
			matchingAvailabilities = append(matchingAvailabilities, []string{minutesToTime(start), minutesToTime(end)})
		}
	}
	return matchingAvailabilities
}

func timeToMinutes(time string) int {
	parts := strings.Split(time, ":")
	hours, _ := strconv.Atoi(parts[0])
	minutes, _ := strconv.Atoi(parts[1])
	return hours*60 + minutes
}

func timeToMinutesArray(calendar [][]string) [][]int {
	minutesArray := [][]int{}
	for _, meeting := range calendar {
		minutesArray = append(minutesArray, []int{timeToMinutes(meeting[0]), timeToMinutes(meeting[1])})
	}
	return minutesArray
}

func minutesToTime(minutes int) string {
	hours := minutes / 60
	mins := minutes % 60
	return fmt.Sprintf("%d:%02d", hours, mins)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def calendarMatching(calendar1: List[List[str]], dailyBounds1: List[str], calendar2: List[List[str]], dailyBounds2: List[str], meetingDuration: int) -> List[List[str]]:
    updated_calendar1 = update_calendar(calendar1, dailyBounds1)
    updated_calendar2 = update_calendar(calendar2, dailyBounds2)
    merged_calendar = merge_calendars(updated_calendar1, updated_calendar2)
    flattened_calendar = flatten_calendar(merged_calendar)
    return get_matching_availabilities(flattened_calendar, meetingDuration)

def update_calendar(calendar, daily_bounds):
    updated = [[time_to_minutes(daily_bounds[0]), time_to_minutes(daily_bounds[0])]]
    updated.extend(time_to_minutes_array(calendar))
    updated.append([time_to_minutes(daily_bounds[1]), time_to_minutes(daily_bounds[1])])
    return updated

def merge_calendars(calendar1, calendar2):
    merged = []
    i, j = 0, 0
    while i < len(calendar1) and j < len(calendar2):
        if calendar1[i][0] < calendar2[j][0]:
            merged.append(calendar1[i])
            i += 1
        else:
            merged.append(calendar2[j])
            j += 1
    merged.extend(calendar1[i:])
    merged.extend(calendar2[j:])
    return merged

def flatten_calendar(calendar):
    flattened = [calendar[0]]
    for i in range(1, len(calendar)):
        current_meeting = calendar[i]
        previous_meeting = flattened[-1]
        current_start, current_end = current_meeting
        previous_start, previous_end = previous_meeting
        if previous_end >= current_start:
            flattened[-1] = [previous_start, max(previous_end, current_end)]
        else:
            flattened.append(current_meeting)
    return flattened

def get_matching_availabilities(calendar, meeting_duration):
    matching_availabilities = []
    for i in range(1, len(calendar)):
        start = calendar[i - 1][1]
        end = calendar[i][0]
        availability_duration = end - start
        if availability_duration >= meeting_duration:
            matching_availabilities.append([minutes_to_time(start), minutes_to_time(end)])
    return matching_availabilities

def time_to_minutes(time):
    hours, minutes = map(int, time.split(':'))
    return hours * 60 + minutes

def time_to_minutes_array(calendar):
    return [[time_to_minutes(meeting[0]), time_to_minutes(meeting[1])] for meeting in calendar]

def minutes_to_time(minutes):
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}:{mins:02d}"`,
		Explanation: `Finding available meeting times between two calendars requires merging busy periods and identifying gaps. The algorithm handles this by treating daily bounds as meetings, merging calendars, flattening overlaps, and finding gaps of sufficient duration.

The algorithm first adds daily bounds (start and end of day) as meetings to each calendar, ensuring we account for unavailable time outside working hours. Then it merges the two sorted calendars by comparing meeting start times and handling overlaps. The merged calendar is flattened by combining overlapping or adjacent meetings into single time blocks.

Finally, it identifies gaps between consecutive meetings that are at least as long as the required meeting duration. These gaps represent available time slots where both people can meet.

Edge cases handled include: calendars with no meetings (entire day available if bounds allow), calendars with back-to-back meetings (no gaps), calendars with overlapping meetings (correctly merged), and calendars where no gap exists (returns empty list).

An alternative approach would check all time slots, but that's inefficient. The merge-and-scan approach is optimal. The time complexity is O(c1+c2) where c1 and c2 are calendar sizes, as we merge and flatten in linear time. The space complexity is O(c1+c2) for the merged and flattened calendars.

Think of this as finding when two busy schedules overlap in their free time: you combine both schedules, merge overlapping busy periods, then look for gaps between busy periods that are long enough for your meeting.`,
	},
	{
		ID:          223,
		Title:       "Waterfall Streams",
		Description: "You're given a two-dimensional array that represents the structure of an indoor waterfall. A value of 1 represents a rock and a value of 0 represents empty space. The water will flow to the empty space directly below it unless blocked by a rock, in which case it will split evenly to the left and right. Write a function that returns the percentage of water in each column at the bottom after the water has flowed through the entire structure.",
		Difficulty:  "Very Hard",
		Topic:       "Arrays",
		Signature:   "func waterfallStreams(array [][]float64, source int) []float64",
		TestCases: []TestCase{
			{Input: "array = [[0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], source = 3", Expected: "[0, 0, 0, 25, 25, 0, 0]"},
		},
		Solution: `func waterfallStreams(array [][]float64, source int) []float64 {
	rowAbove := make([]float64, len(array[0]))
	rowAbove[source] = -100
	for row := 1; row < len(array); row++ {
		currentRow := make([]float64, len(array[0]))
		copy(currentRow, rowAbove)
		for idx := 0; idx < len(rowAbove); idx++ {
			valueAbove := rowAbove[idx]
			hasWaterAbove := valueAbove < 0
			hasBlock := array[row][idx] == 1.0
			if !hasWaterAbove {
				continue
			}
			if !hasBlock {
				currentRow[idx] += valueAbove
				continue
			}
			splitWater := valueAbove / 2
			rightIdx := idx
			for rightIdx+1 < len(rowAbove) {
				rightIdx++
				if array[row-1][rightIdx] == 1.0 {
					break
				}
				if array[row][rightIdx] != 1.0 {
					currentRow[rightIdx] += splitWater
					break
				}
			}
			leftIdx := idx
			for leftIdx-1 >= 0 {
				leftIdx--
				if array[row-1][leftIdx] == 1.0 {
					break
				}
				if array[row][leftIdx] != 1.0 {
					currentRow[leftIdx] += splitWater
					break
				}
			}
		}
		rowAbove = currentRow
	}
	finalPercentages := make([]float64, len(rowAbove))
	for i := 0; i < len(rowAbove); i++ {
		num := rowAbove[i]
		finalPercentages[i] = -num
	}
	return finalPercentages
}`,
		PythonSolution: `def waterfallStreams(array: List[List[float]], source: int) -> List[float]:
    row_above = [0.0] * len(array[0])
    row_above[source] = -100.0
    for row in range(1, len(array)):
        current_row = row_above[:]
        for idx in range(len(row_above)):
            value_above = row_above[idx]
            has_water_above = value_above < 0
            has_block = array[row][idx] == 1.0
            if not has_water_above:
                continue
            if not has_block:
                current_row[idx] += value_above
                continue
            split_water = value_above / 2
            right_idx = idx
            while right_idx + 1 < len(row_above):
                right_idx += 1
                if array[row - 1][right_idx] == 1.0:
                    break
                if array[row][right_idx] != 1.0:
                    current_row[right_idx] += split_water
                    break
            left_idx = idx
            while left_idx - 1 >= 0:
                left_idx -= 1
                if array[row - 1][left_idx] == 1.0:
                    break
                if array[row][left_idx] != 1.0:
                    current_row[left_idx] += split_water
                    break
        row_above = current_row
    final_percentages = [-num for num in row_above]
    return final_percentages`,
		Explanation: `Simulating water flow through a waterfall structure requires tracking how water splits when blocked by rocks. The algorithm processes the structure row by row, splitting water horizontally when it encounters blocks.

The algorithm starts with 100% water at the source column. For each subsequent row, it checks each column. If water is present above and there's no block, water flows straight down. If water is present above but there's a block, the water splits evenly: 50% goes left and 50% goes right. The algorithm searches left and right to find the first non-blocked position in the current row (but blocked in the row above), distributing the split water there.

This simulation continues row by row until reaching the bottom. Negative values represent water percentages (to distinguish from blocks which are 1.0), and the final result converts these to positive percentages.

Edge cases handled include: water blocked on both sides (water trapped, doesn't flow), water reaching edges (handled by boundary checks), multiple splits in same row (water accumulates correctly), and structures with no blocks (all water flows straight down).

An alternative approach would use a more complex flow simulation, but the row-by-row approach is simpler and sufficient. The time complexity is O(w²*h) where w is width and h is height, as for each blocked position we may search O(w) positions left and right. The space complexity is O(w) for storing the current and previous rows.

Think of this as water flowing down a structure: when water hits a rock, it splits and flows around it, and you track how much water ends up in each column at the bottom.`,
	},
	{
		ID:          224,
		Title:       "Minimum Area Rectangle",
		Description: "You're given an array of points on a coordinate plane. Write a function that returns the minimum area of any rectangle that can be formed using any 4 of these points such that the rectangle's sides are parallel to the x and y axes. If no rectangle can be formed, your function should return 0.",
		Difficulty:  "Very Hard",
		Topic:       "Hash Tables",
		Signature:   "func minimumAreaRectangle(points [][]int) int",
		TestCases: []TestCase{
			{Input: "points = [[1, 5], [5, 1], [4, 2], [2, 4], [2, 2], [1, 2], [4, 5], [2, 5], [-1, -2]]", Expected: "3"},
		},
		Solution: `func minimumAreaRectangle(points [][]int) int {
	pointSet := make(map[string]bool)
	for _, point := range points {
		pointSet[pointToString(point[0], point[1])] = true
	}
	minArea := math.MaxInt32
	for i := 0; i < len(points); i++ {
		p2x, p2y := points[i][0], points[i][1]
		for j := 0; j < i; j++ {
			p1x, p1y := points[j][0], points[j][1]
			pointsShareValue := p1x == p2x || p1y == p2y
			if pointsShareValue {
				continue
			}
			point1OnOppositeDiagonalExists := pointSet[pointToString(p1x, p2y)]
			point2OnOppositeDiagonalExists := pointSet[pointToString(p2x, p1y)]
			oppositeDiagonalExists := point1OnOppositeDiagonalExists && point2OnOppositeDiagonalExists
			if oppositeDiagonalExists {
				currentArea := abs(p2x-p1x) * abs(p2y-p1y)
				minArea = min(minArea, currentArea)
			}
		}
	}
	if minArea != math.MaxInt32 {
		return minArea
	}
	return 0
}

func pointToString(x int, y int) string {
	return fmt.Sprintf("%d,%d", x, y)
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}`,
		PythonSolution: `def minimumAreaRectangle(points: List[List[int]]) -> int:
    point_set = {f"{point[0]},{point[1]}" for point in points}
    min_area = float('inf')
    for i in range(len(points)):
        p2x, p2y = points[i]
        for j in range(i):
            p1x, p1y = points[j]
            points_share_value = p1x == p2x or p1y == p2y
            if points_share_value:
                continue
            point1_on_opposite_diagonal_exists = f"{p1x},{p2y}" in point_set
            point2_on_opposite_diagonal_exists = f"{p2x},{p1y}" in point_set
            opposite_diagonal_exists = point1_on_opposite_diagonal_exists and point2_on_opposite_diagonal_exists
            if opposite_diagonal_exists:
                current_area = abs(p2x - p1x) * abs(p2y - p1y)
                min_area = min(min_area, current_area)
    return min_area if min_area != float('inf') else 0`,
		Explanation: `Finding the minimum area rectangle with sides parallel to axes requires identifying four points that form a rectangle. The key insight is that for two points to be opposite corners of a rectangle, the other two corners (with swapped x and y coordinates) must also exist.

The algorithm stores all points in a hash set for O(1) lookup. Then it considers all pairs of points. For each pair (p1, p2), if they could be opposite corners of a rectangle (different x and y coordinates), it checks if the other two required points exist: (p1.x, p2.y) and (p2.x, p1.y). If all four points exist, it calculates the area and tracks the minimum.

This approach efficiently finds all rectangles by checking point pairs rather than trying all combinations of four points, which would be O(n⁴). The hash set lookup makes checking point existence O(1).

Edge cases handled include: fewer than 4 points (no rectangle possible, returns 0), points forming a square (area calculated correctly), points with duplicate coordinates (handled by set), and points where no rectangle exists (returns 0).

An alternative O(n⁴) approach would try all combinations of four points, but the pair-based approach is more efficient. The time complexity is O(n²) where n is the number of points, as we check all pairs. The space complexity is O(n) for the point set.

Think of this as finding rectangles on graph paper: for any two points that could be opposite corners, you check if the other two corners exist, and if they do, you've found a rectangle and can calculate its area.`,
	},
	{
		ID:          225,
		Title:       "Line Through Points",
		Description: "You're given an array of points plotted on a 2D graph. Write a function that returns the maximum number of points that can be formed with a single straight line drawn through any two points.",
		Difficulty:  "Very Hard",
		Topic:       "Math",
		Signature:   "func lineThroughPoints(points [][]int) int",
		TestCases: []TestCase{
			{Input: "points = [[1, 1], [2, 2], [3, 3], [0, 4], [-2, 6], [4, 0], [2, 1]]", Expected: "4"},
		},
		Solution: `func lineThroughPoints(points [][]int) int {
	maxNumberOfPointsOnLine := 1
	for i := 0; i < len(points); i++ {
		p1 := points[i]
		slopes := make(map[string]int)
		for j := i + 1; j < len(points); j++ {
			p2 := points[j]
			rise, run := getSlopeOfLineBetweenPoints(p1, p2)
			slopeKey := slopeToString(rise, run)
			if _, found := slopes[slopeKey]; !found {
				slopes[slopeKey] = 1
			}
			slopes[slopeKey]++
		}
		currentMaxNumberOfPointsOnLine := maxValueIn(slopes)
		maxNumberOfPointsOnLine = max(maxNumberOfPointsOnLine, currentMaxNumberOfPointsOnLine)
	}
	return maxNumberOfPointsOnLine
}

func getSlopeOfLineBetweenPoints(p1 []int, p2 []int) (int, int) {
	p1x, p1y := p1[0], p1[1]
	p2x, p2y := p2[0], p2[1]
	slope := []int{1, 0}
	if p1x != p2x {
		xDiff := p1x - p2x
		yDiff := p1y - p2y
		gcdValue := gcd(abs(xDiff), abs(yDiff))
		xDiff = xDiff / gcdValue
		yDiff = yDiff / gcdValue
		if xDiff < 0 {
			xDiff *= -1
			yDiff *= -1
		}
		slope = []int{yDiff, xDiff}
	}
	return slope[0], slope[1]
}

func slopeToString(numerator int, denominator int) string {
	return fmt.Sprintf("%d:%d", numerator, denominator)
}

func gcd(num1 int, num2 int) int {
	a, b := num1, num2
	for {
		if a == 0 {
			return b
		}
		if b == 0 {
			return a
		}
		a, b = b, a%b
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func maxValueIn(slopes map[string]int) int {
	currentMax := 0
	for _, value := range slopes {
		currentMax = max(currentMax, value)
	}
	return currentMax
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def lineThroughPoints(points: List[List[int]]) -> int:
    max_number_of_points_on_line = 1
    for i in range(len(points)):
        p1 = points[i]
        slopes = {}
        for j in range(i + 1, len(points)):
            p2 = points[j]
            rise, run = get_slope_of_line_between_points(p1, p2)
            slope_key = f"{rise}:{run}"
            slopes[slope_key] = slopes.get(slope_key, 1) + 1
        current_max_number_of_points_on_line = max(slopes.values()) if slopes else 1
        max_number_of_points_on_line = max(max_number_of_points_on_line, current_max_number_of_points_on_line)
    return max_number_of_points_on_line

def get_slope_of_line_between_points(p1, p2):
    p1x, p1y = p1
    p2x, p2y = p2
    slope = [1, 0]
    if p1x != p2x:
        x_diff = p1x - p2x
        y_diff = p1y - p2y
        gcd_value = gcd(abs(x_diff), abs(y_diff))
        x_diff = x_diff // gcd_value
        y_diff = y_diff // gcd_value
        if x_diff < 0:
            x_diff *= -1
            y_diff *= -1
        slope = [y_diff, x_diff]
    return slope[0], slope[1]

def slope_to_string(numerator, denominator):
    return f"{numerator}:{denominator}"

def gcd(num1, num2):
    a, b = num1, num2
    while True:
        if a == 0:
            return b
        if b == 0:
            return a
        a, b = b, a % b`,
		Explanation: `Finding the maximum number of points on a line requires checking all possible lines through pairs of points. The key challenge is representing slopes accurately to handle floating-point precision issues.

The algorithm considers each point as a potential line anchor. For each anchor point, it calculates the slope to every other point. To avoid floating-point precision issues, slopes are represented as normalized fractions using GCD (greatest common divisor). The slope (y2-y1)/(x2-x1) is normalized by dividing both numerator and denominator by their GCD, and the sign is standardized.

Points with the same normalized slope lie on the same line. The algorithm counts how many points share each slope, tracking the maximum. Vertical lines (infinite slope) are handled specially.

Edge cases handled include: all points on same line (maximum is n), no three points collinear (maximum is 2), points with duplicate coordinates (handled correctly), and vertical lines (infinite slope represented specially).

An alternative approach would use floating-point slopes, but that's prone to precision errors. The GCD normalization approach is exact. The time complexity is O(n²) where n is the number of points, as we check all pairs for each anchor point. The space complexity is O(n) for storing slope counts.

Think of this as finding the straight line that passes through the most points: for each point, you check all lines through that point, normalize their slopes to avoid precision issues, and count how many other points lie on each line.`,
	},
	{
		ID:          226,
		Title:       "Right Smaller Than",
		Description: "Write a function that takes in an array of integers and returns an array of the same length, where each element in the output array corresponds to the number of smaller elements to the right of that element in the input array. In other words, the value at output[i] represents how many elements are smaller than input[i] to the right of input[i].",
		Difficulty:  "Very Hard",
		Topic:       "Binary Search Trees",
		Signature:   "func rightSmallerThan(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = [8, 5, 11, -1, 3, 4, 2]", Expected: "[5, 4, 4, 0, 1, 1, 0]"},
		},
		Solution: `type SpecialBST struct {
	value        int
	leftSubtreeSize int
	left         *SpecialBST
	right        *SpecialBST
}

func rightSmallerThan(array []int) []int {
	if len(array) == 0 {
		return []int{}
	}
	rightSmallerCounts := make([]int, len(array))
	lastIdx := len(array) - 1
	bst := &SpecialBST{value: array[lastIdx]}
	rightSmallerCounts[lastIdx] = 0
	for i := len(array) - 2; i >= 0; i-- {
		bst = insert(bst, array[i], i, rightSmallerCounts, 0)
	}
	return rightSmallerCounts
}

func insert(bst *SpecialBST, value int, idx int, rightSmallerCounts []int, numSmallerAtInsertTime int) *SpecialBST {
	if bst == nil {
		rightSmallerCounts[idx] = numSmallerAtInsertTime
		return &SpecialBST{value: value, leftSubtreeSize: 0}
	}
	if value < bst.value {
		bst.leftSubtreeSize++
		bst.left = insert(bst.left, value, idx, rightSmallerCounts, numSmallerAtInsertTime)
	} else {
		numSmallerAtInsertTime += bst.leftSubtreeSize
		if value > bst.value {
			numSmallerAtInsertTime++
		}
		bst.right = insert(bst.right, value, idx, rightSmallerCounts, numSmallerAtInsertTime)
	}
	return bst
}`,
		PythonSolution: `def rightSmallerThan(array: List[int]) -> List[int]:
    if len(array) == 0:
        return []
    right_smaller_counts = [0] * len(array)
    last_idx = len(array) - 1
    bst = SpecialBST(array[last_idx])
    right_smaller_counts[last_idx] = 0
    for i in range(len(array) - 2, -1, -1):
        bst = insert(bst, array[i], i, right_smaller_counts, 0)
    return right_smaller_counts

class SpecialBST:
    def __init__(self, value):
        self.value = value
        self.left_subtree_size = 0
        self.left = None
        self.right = None

def insert(bst, value, idx, right_smaller_counts, num_smaller_at_insert_time):
    if bst is None:
        right_smaller_counts[idx] = num_smaller_at_insert_time
        return SpecialBST(value)
    if value < bst.value:
        bst.left_subtree_size += 1
        bst.left = insert(bst.left, value, idx, right_smaller_counts, num_smaller_at_insert_time)
    else:
        num_smaller_at_insert_time += bst.left_subtree_size
        if value > bst.value:
            num_smaller_at_insert_time += 1
        bst.right = insert(bst.right, value, idx, right_smaller_counts, num_smaller_at_insert_time)
    return bst`,
		Explanation: `Counting how many elements are smaller than each element to its right requires processing the array from right to left and maintaining a data structure that tracks smaller elements efficiently. A Binary Search Tree (BST) with left subtree size tracking provides an elegant solution.

The algorithm builds a BST by inserting elements from right to left. Each node tracks the size of its left subtree. When inserting a new element, if it goes to the left of a node, that node's left subtree size increases. If it goes to the right, we add the current node's left subtree size (plus 1 if the current node's value is smaller) to our count of smaller elements.

By processing from right to left, when we insert an element, all elements to its right are already in the BST, allowing us to count how many are smaller. The left subtree size gives us exactly this count without needing to traverse.

Edge cases handled include: arrays in descending order (all counts are 0), arrays in ascending order (counts are 0, 1, 2, ...), arrays with duplicates (handled by comparison logic), and empty arrays (returns empty array).

An alternative O(n²) approach would compare each element with all elements to its right, but the BST approach is more efficient. The time complexity is O(n log n) on average for balanced trees, O(n²) worst case for skewed trees. The space complexity is O(n) for the BST.

Think of this as building a sorted structure from right to left: as you add each element, you can immediately see how many elements already in the structure are smaller, because the BST organizes them and tracks subtree sizes.`,
	},
	{
		ID:          227,
		Title:       "Iterative In-order Traversal",
		Description: "Write a function that takes in a Binary Tree and returns a list of its node values in in-order order. Your function should implement an iterative approach rather than a recursive one.",
		Difficulty:  "Very Hard",
		Topic:       "Binary Trees",
		Signature:   "func iterativeInOrderTraversal(tree *TreeNode, callback func(*TreeNode))",
		TestCases: []TestCase{
			{Input: "tree = [1, 2, 3, 4, 5, 6, 7]", Expected: "[4, 2, 5, 1, 6, 3, 7]"},
		},
		Solution: `func iterativeInOrderTraversal(tree *TreeNode, callback func(*TreeNode)) {
	var previousNode *TreeNode
	currentNode := tree
	for currentNode != nil {
		var nextNode *TreeNode
		if previousNode == nil || previousNode == currentNode.Parent {
			if currentNode.Left != nil {
				nextNode = currentNode.Left
			} else {
				callback(currentNode)
				if currentNode.Right != nil {
					nextNode = currentNode.Right
				} else {
					nextNode = currentNode.Parent
				}
			}
		} else if previousNode == currentNode.Left {
			callback(currentNode)
			if currentNode.Right != nil {
				nextNode = currentNode.Right
			} else {
				nextNode = currentNode.Parent
			}
		} else {
			nextNode = currentNode.Parent
		}
		previousNode = currentNode
		currentNode = nextNode
	}
}`,
		PythonSolution: `def iterativeInOrderTraversal(tree: Optional[TreeNode], callback: Callable) -> None:
    previous_node = None
    current_node = tree
    while current_node is not None:
        if previous_node is None or previous_node == current_node.parent:
            if current_node.left is not None:
                next_node = current_node.left
            else:
                callback(current_node)
                next_node = current_node.right if current_node.right is not None else current_node.parent
        elif previous_node == current_node.left:
            callback(current_node)
            next_node = current_node.right if current_node.right is not None else current_node.parent
        else:
            next_node = current_node.parent
        previous_node = current_node
        current_node = next_node`,
		Explanation: `Performing in-order traversal iteratively without a stack requires tracking the direction we came from to determine the next move. This approach uses the parent pointers and a "previous node" reference to navigate the tree.

The algorithm maintains a current node and a previous node to track traversal direction. If we came from the parent, we go left (or process if no left child). If we came from the left child, we process the current node and go right. If we came from the right child, we go back up to the parent.

This approach efficiently traverses the tree in-order by using parent pointers to navigate without an explicit stack. The direction tracking ensures we visit nodes in the correct order: left subtree, current node, right subtree.

Edge cases handled include: trees with only left children (traverses left, processes, goes up), trees with only right children (processes, goes right, goes up), single-node trees (processes and returns), and trees where we need to backtrack multiple levels (handled by parent pointers).

An alternative approach would use an explicit stack, but that requires O(h) space. The parent pointer approach achieves O(1) extra space (beyond the tree structure itself). The time complexity is O(n) as we visit each node once. The space complexity is O(1) extra space.

Think of this as walking through a maze with breadcrumbs: you remember where you came from, and based on that, you know whether to go left, process the current room, go right, or backtrack.`,
	},
	{
		ID:          228,
		Title:       "Flatten Binary Tree",
		Description: "Write a function that takes in a Binary Tree, flattens it, and returns its leftmost node. A flattened Binary Tree is a structure that's nearly identical to a Doubly Linked List (except that nodes have left and right pointers instead of prev and next pointers), where nodes follow the original tree's left-to-right order.",
		Difficulty:  "Very Hard",
		Topic:       "Binary Trees",
		Signature:   "func flattenBinaryTree(root *TreeNode) *TreeNode",
		TestCases: []TestCase{
			{Input: "root = [1, 2, 3, 4, 5, 6]", Expected: "Flattened tree"},
		},
		Solution: `func flattenBinaryTree(root *TreeNode) *TreeNode {
	leftMost, _ := flattenTree(root)
	return leftMost
}

func flattenTree(node *TreeNode) (*TreeNode, *TreeNode) {
	var leftMost *TreeNode
	var rightMost *TreeNode
	if node.Left == nil {
		leftMost = node
	} else {
		leftSubtreeLeftMost, leftSubtreeRightMost := flattenTree(node.Left)
		connectNodes(leftSubtreeRightMost, node)
		leftMost = leftSubtreeLeftMost
	}
	if node.Right == nil {
		rightMost = node
	} else {
		rightSubtreeLeftMost, rightSubtreeRightMost := flattenTree(node.Right)
		connectNodes(node, rightSubtreeLeftMost)
		rightMost = rightSubtreeRightMost
	}
	return leftMost, rightMost
}

func connectNodes(left *TreeNode, right *TreeNode) {
	left.Right = right
	right.Left = left
}`,
		PythonSolution: `def flattenBinaryTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    left_most, _ = flatten_tree(root)
    return left_most

def flatten_tree(node):
    if node.left is None:
        left_most = node
    else:
        left_subtree_left_most, left_subtree_right_most = flatten_tree(node.left)
        connect_nodes(left_subtree_right_most, node)
        left_most = left_subtree_left_most
    if node.right is None:
        right_most = node
    else:
        right_subtree_left_most, right_subtree_right_most = flatten_tree(node.right)
        connect_nodes(node, right_subtree_left_most)
        right_most = right_subtree_right_most
    return left_most, right_most

def connect_nodes(left, right):
    left.right = right
    right.left = left`,
		Explanation: `Flattening a binary tree into a doubly-linked list in-place requires carefully connecting nodes while maintaining the in-order traversal sequence. The recursive approach processes subtrees and connects them at their boundaries.

The algorithm recursively flattens the left and right subtrees, each returning the leftmost and rightmost nodes of the flattened subtree. For the left subtree, it connects its rightmost node to the current node. For the right subtree, it connects the current node to its leftmost node. The function returns the overall leftmost and rightmost nodes of the flattened tree.

This approach maintains the in-order sequence: all nodes in the left subtree come before the current node, which comes before all nodes in the right subtree. By connecting at the boundaries, we create a continuous linked list structure.

Edge cases handled include: trees with no left subtree (current node is leftmost), trees with no right subtree (current node is rightmost), single-node trees (node is both leftmost and rightmost), and trees where subtrees are already flat (connections handled correctly).

An alternative iterative approach would use a stack, but the recursive approach is more elegant. The time complexity is O(n) as we visit each node once. The space complexity is O(h) for the recursion stack where h is the tree height.

Think of this as converting a tree into a chain: you flatten each branch separately, then connect the end of the left branch to the root, and the root to the start of the right branch, creating one continuous chain.`,
	},
	{
		ID:          229,
		Title:       "Right Sibling Tree",
		Description: "Write a function that takes in a Binary Tree, transforms it into a Right Sibling Tree, and returns its root. A Right Sibling Tree is obtained by making every node in a Binary Tree have its right property point to its right sibling instead of its right child. A node's right sibling is the node immediately to its right on the same level or None / null if there is no node immediately to its right.",
		Difficulty:  "Very Hard",
		Topic:       "Binary Trees",
		Signature:   "func rightSiblingTree(root *TreeNode) *TreeNode",
		TestCases: []TestCase{
			{Input: "root = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]", Expected: "Right sibling tree"},
		},
		Solution: `func rightSiblingTree(root *TreeNode) *TreeNode {
	mutate(root, nil, false)
	return root
}

func mutate(node *TreeNode, parent *TreeNode, isLeftChild bool) {
	if node == nil {
		return
	}
	left, right := node.Left, node.Right
	mutate(left, node, true)
	if parent == nil {
		node.Right = nil
	} else if isLeftChild {
		node.Right = parent.Right
	} else {
		if parent.Right == nil {
			node.Right = nil
		} else {
			node.Right = parent.Right.Left
		}
	}
	mutate(right, node, false)
}`,
		PythonSolution: `def rightSiblingTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    mutate(root, None, False)
    return root

def mutate(node, parent, is_left_child):
    if node is None:
        return
    left, right = node.left, node.right
    mutate(left, node, True)
    if parent is None:
        node.right = None
    elif is_left_child:
        node.right = parent.right
    else:
        if parent.right is None:
            node.right = None
        else:
            node.right = parent.right.left
    mutate(right, node, False)`,
		Explanation: `Transforming a binary tree so each node's right pointer points to its right sibling requires understanding the tree structure and sibling relationships. The recursive approach processes the tree level by level, setting right pointers correctly.

The algorithm recursively processes left and right children. For a left child, its right sibling is its parent's right child (if it exists). For a right child, its right sibling is the leftmost child of its parent's right child (the first node in the next subtree at the same level).

This approach correctly handles the level-by-level structure: all nodes at the same depth are connected horizontally via right pointers, creating a level-order linked structure while maintaining the original tree's parent-child relationships via left pointers.

Edge cases handled include: trees with no right siblings (right pointer set to nil), trees where a level has only one node (right pointer is nil), trees with complete levels (all siblings connected), and single-node trees (no siblings, right is nil).

An alternative approach would use level-order traversal with a queue, but the recursive approach is more elegant. The time complexity is O(n) as we visit each node once. The space complexity is O(h) for the recursion stack where h is the tree height.

Think of this as connecting siblings at each level: you process the tree recursively, and for each node, you connect it to its right neighbor at the same level, which is either the parent's right child (for left children) or the next subtree's leftmost node (for right children).`,
	},
	{
		ID:          230,
		Title:       "Max Profit With K Transactions",
		Description: "You're given an array of positive integers representing the prices of a single stock on various days (each index in the array represents a different day). You're also given an integer k, which represents the number of transactions you're allowed to make. One transaction consists of buying the stock on a given day and selling it on another, later day. Write a function that returns the maximum profit that you can make by buying and selling the stock, given k transactions.",
		Difficulty:  "Very Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func maxProfitWithKTransactions(prices []int, k int) int",
		TestCases: []TestCase{
			{Input: "prices = [5, 11, 3, 50, 60, 90], k = 2", Expected: "93"},
		},
		Solution: `func maxProfitWithKTransactions(prices []int, k int) int {
	if len(prices) == 0 {
		return 0
	}
	evenProfits := make([]int, len(prices))
	oddProfits := make([]int, len(prices))
	for i := range evenProfits {
		evenProfits[i] = 0
		oddProfits[i] = 0
	}
	for t := 1; t < k+1; t++ {
		maxThusFar := math.MinInt32
		if t%2 == 1 {
			currentProfits := oddProfits
			previousProfits := evenProfits
			for d := 1; d < len(prices); d++ {
				maxThusFar = max(maxThusFar, previousProfits[d-1]-prices[d-1])
				currentProfits[d] = max(currentProfits[d-1], maxThusFar+prices[d])
			}
		} else {
			currentProfits := evenProfits
			previousProfits := oddProfits
			for d := 1; d < len(prices); d++ {
				maxThusFar = max(maxThusFar, previousProfits[d-1]-prices[d-1])
				currentProfits[d] = max(currentProfits[d-1], maxThusFar+prices[d])
			}
		}
	}
	if k%2 == 0 {
		return evenProfits[len(prices)-1]
	}
	return oddProfits[len(prices)-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def maxProfitWithKTransactions(prices: List[int], k: int) -> int:
    if len(prices) == 0:
        return 0
    even_profits = [0] * len(prices)
    odd_profits = [0] * len(prices)
    for t in range(1, k + 1):
        max_thus_far = float('-inf')
        if t % 2 == 1:
            current_profits = odd_profits
            previous_profits = even_profits
        else:
            current_profits = even_profits
            previous_profits = odd_profits
        for d in range(1, len(prices)):
            max_thus_far = max(max_thus_far, previous_profits[d - 1] - prices[d - 1])
            current_profits[d] = max(current_profits[d - 1], max_thus_far + prices[d])
    return even_profits[len(prices) - 1] if k % 2 == 0 else odd_profits[len(prices) - 1]`,
		Explanation: `Finding the maximum profit from at most k stock transactions requires tracking the best profit achievable with each number of transactions completed. Dynamic programming with space optimization efficiently handles this.

The algorithm uses two arrays: one for profits with an even number of transactions completed, one for odd. For each day, it considers either holding (no transaction) or selling (completing a transaction). When selling, it finds the best previous day to buy by tracking the maximum profit from previous transactions minus the buy price.

The space optimization uses only two arrays instead of a 2D table, since we only need the previous transaction count's profits. The maxThusFar variable tracks the best (profit - buyPrice) seen so far, allowing efficient computation of the best buy day for each sell day.

Edge cases handled include: k = 0 (no transactions, profit 0), k >= n/2 (unlimited transactions, can buy/sell every day), prices in descending order (no profit possible), and prices with single peak (optimal to buy low, sell high).

An alternative approach would use a 2D DP table, but that requires O(n*k) space. The optimized approach achieves O(n) space while maintaining O(n*k) time complexity. For k >= n/2, we can optimize to O(n) time by treating it as unlimited transactions.

Think of this as making investment decisions: you track the best profit you can achieve with each number of trades completed, and for each day, you decide whether to hold or sell, always choosing the option that maximizes your profit.`,
	},
	{
		ID:          231,
		Title:       "Palindrome Partitioning Min Cuts",
		Description: "Given a non-empty string, write a function that returns the minimum number of cuts needed to perform on the string such that each remaining string segment (after the cuts are performed) is a palindrome.",
		Difficulty:  "Very Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func palindromePartitioningMinCuts(str string) int",
		TestCases: []TestCase{
			{Input: "str = \"noonabbad\"", Expected: "2"},
		},
		Solution: `func palindromePartitioningMinCuts(str string) int {
	palindromes := make([][]bool, len(str))
	for i := range palindromes {
		palindromes[i] = make([]bool, len(str))
	}
	for i := 0; i < len(str); i++ {
		for j := i; j < len(str); j++ {
			palindromes[i][j] = isPalindrome(str, i, j)
		}
	}
	cuts := make([]int, len(str))
	for i := range cuts {
		cuts[i] = math.MaxInt32
	}
	for i := 0; i < len(str); i++ {
		if palindromes[0][i] {
			cuts[i] = 0
		} else {
			cuts[i] = cuts[i-1] + 1
			for j := 1; j < i; j++ {
				if palindromes[j][i] && cuts[j-1]+1 < cuts[i] {
					cuts[i] = cuts[j-1] + 1
				}
			}
		}
	}
	return cuts[len(str)-1]
}

func isPalindrome(str string, i int, j int) bool {
	for i < j {
		if str[i] != str[j] {
			return false
		}
		i++
		j--
	}
	return true
}`,
		PythonSolution: `def palindromePartitioningMinCuts(string: str) -> int:
    palindromes = [[False for _ in range(len(string))] for _ in range(len(string))]
    for i in range(len(string)):
        for j in range(i, len(string)):
            palindromes[i][j] = is_palindrome(string, i, j)
    cuts = [float('inf')] * len(string)
    for i in range(len(string)):
        if palindromes[0][i]:
            cuts[i] = 0
        else:
            cuts[i] = cuts[i - 1] + 1
            for j in range(1, i):
                if palindromes[j][i] and cuts[j - 1] + 1 < cuts[i]:
                    cuts[i] = cuts[j - 1] + 1
    return cuts[len(string) - 1]

def is_palindrome(string, i, j):
    while i < j:
        if string[i] != string[j]:
            return False
        i += 1
        j -= 1
    return True`,
		Explanation: `Finding the minimum cuts needed to partition a string into palindromes requires determining which substrings are palindromes and then finding the optimal partition. Dynamic programming efficiently solves both parts.

The algorithm first precomputes a 2D table indicating which substrings are palindromes, using the fact that a substring is a palindrome if its endpoints match and the inner substring is also a palindrome. Then it uses DP where cuts[i] represents the minimum cuts needed for the prefix str[0..i].

For each position, if the entire prefix is a palindrome, no cuts are needed. Otherwise, it tries all possible partition points, checking if the suffix from that point is a palindrome, and takes the minimum cuts from the prefix plus one.

This approach efficiently combines palindrome detection with optimal partitioning. The palindrome table avoids redundant checks, and the DP ensures we find the minimum cuts without trying all possible partitions explicitly.

Edge cases handled include: strings that are already palindromes (0 cuts), strings with no palindromes longer than 1 (n-1 cuts, one per character), strings with overlapping palindrome possibilities (DP finds optimal), and single-character strings (0 cuts, already a palindrome).

An alternative approach would try all partitions recursively, but that's exponential. The DP approach achieves O(n²) time for building the palindrome table and O(n²) time for finding minimum cuts, giving overall O(n²) time. The space complexity is O(n²) for the palindrome table and O(n) for the cuts array.

Think of this as cutting a string into palindrome pieces: you first figure out which parts are palindromes, then find the way to cut the string that uses the fewest cuts while ensuring each piece is a palindrome.`,
	},
	{
		ID:          232,
		Title:       "Longest String Chain",
		Description: "A string chain is defined as follows: let string A be a string in the initial array; if removing any single character from string A yields a new string B that's contained in the initial array of strings, then strings A and B form a string chain of length 2. Similarly, if removing any single character from string B yields a new string C that's contained in the initial array of strings, then strings A, B, and C form a string chain of length 3. Write a function that returns the longest string chain length.",
		Difficulty:  "Very Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func longestStringChain(strings []string) []string",
		TestCases: []TestCase{
			{Input: "strings = [\"abde\", \"abc\", \"abd\", \"abcde\", \"ade\", \"ae\", \"1abde\", \"abcde\"]", Expected: "[\"abcde\", \"abde\", \"ade\", \"ae\"]"},
		},
		Solution: `func longestStringChain(strings []string) []string {
	stringChains := make(map[string]*StringChain)
	for _, str := range strings {
		stringChains[str] = &StringChain{NextString: "", MaxChainLength: 1}
	}
	for _, str := range strings {
		findLongestStringChain(str, stringChains)
	}
	return buildLongestStringChain(strings, stringChains)
}

func findLongestStringChain(str string, stringChains map[string]*StringChain) {
	for i := 0; i < len(str); i++ {
		smallerString := str[:i] + str[i+1:]
		if _, found := stringChains[smallerString]; !found {
			continue
		}
		smallerStringChainLength := stringChains[smallerString].MaxChainLength
		currentStringChainLength := stringChains[str].MaxChainLength
		if smallerStringChainLength+1 > currentStringChainLength {
			stringChains[str].MaxChainLength = smallerStringChainLength + 1
			stringChains[str].NextString = smallerString
		}
	}
}

func buildLongestStringChain(strings []string, stringChains map[string]*StringChain) []string {
	maxChainLength := 0
	chainStartingString := ""
	for _, str := range strings {
		if stringChains[str].MaxChainLength > maxChainLength {
			maxChainLength = stringChains[str].MaxChainLength
			chainStartingString = str
		}
	}
	longestStringChain := []string{}
	currentString := chainStartingString
	for currentString != "" {
		longestStringChain = append(longestStringChain, currentString)
		currentString = stringChains[currentString].NextString
	}
	return longestStringChain
}

type StringChain struct {
	NextString     string
	MaxChainLength int
}`,
		PythonSolution: `def longestStringChain(strings: List[str]) -> List[str]:
    string_chains = {string: {"next_string": "", "max_chain_length": 1} for string in strings}
    for string in strings:
        find_longest_string_chain(string, string_chains)
    return build_longest_string_chain(strings, string_chains)

def find_longest_string_chain(string, string_chains):
    for i in range(len(string)):
        smaller_string = string[:i] + string[i + 1:]
        if smaller_string not in string_chains:
            continue
        smaller_string_chain_length = string_chains[smaller_string]["max_chain_length"]
        current_string_chain_length = string_chains[string]["max_chain_length"]
        if smaller_string_chain_length + 1 > current_string_chain_length:
            string_chains[string]["max_chain_length"] = smaller_string_chain_length + 1
            string_chains[string]["next_string"] = smaller_string

def build_longest_string_chain(strings, string_chains):
    max_chain_length = 0
    chain_starting_string = ""
    for string in strings:
        if string_chains[string]["max_chain_length"] > max_chain_length:
            max_chain_length = string_chains[string]["max_chain_length"]
            chain_starting_string = string
    longest_string_chain = []
    current_string = chain_starting_string
    while current_string != "":
        longest_string_chain.append(current_string)
        current_string = string_chains[current_string]["next_string"]
    return longest_string_chain`,
		Explanation: `Finding the longest string chain requires identifying sequences where each string can be formed by adding exactly one character to the previous string. Dynamic programming efficiently tracks chain lengths and reconstructs the longest chain.

The algorithm processes each string and tries removing each character to see if the resulting string exists in the set. If it does, that string is a predecessor in a chain. The algorithm updates the chain length for the current string to be one more than the predecessor's chain length, and tracks which string comes next in the chain.

After computing chain lengths for all strings, it finds the string with the maximum chain length, then reconstructs the chain by following the "next string" pointers backwards from that string.

Edge cases handled include: strings with no predecessors (chain length 1), strings that are part of multiple chains (longest chain chosen), strings of length 1 (no predecessors possible), and sets where no chains exist (each string has chain length 1).

An alternative approach would try all possible chains, but that's exponential. The DP approach achieves O(n*m²) time where n is the number of strings and m is average length, as we check m characters to remove for each of n strings. The space complexity is O(n*m) for storing strings and chain information.

Think of this as building word ladders: you find which words can be formed by adding one letter to other words, track the longest chain you can build, then reconstruct that chain by following the connections backwards from the longest word.`,
	},
	{
		ID:          233,
		Title:       "Square of Zeroes",
		Description: "Write a function that takes in a square-shaped n x n two-dimensional array containing only 1s and 0s, and returns a boolean representing whether the input matrix contains a square whose borders are made up of only 0s. Note that a 1 x 1 square doesn't count as a valid square for the purpose of this problem. In other words, a square of zeroes has to be at least 2 x 2.",
		Difficulty:  "Very Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func squareOfZeroes(matrix [][]int) bool",
		TestCases: []TestCase{
			{Input: "matrix = [[1, 1, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 1, 1, 1, 0, 1], [0, 0, 0, 1, 0, 1], [0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 1]]", Expected: "true"},
		},
		Solution: `func squareOfZeroes(matrix [][]int) bool {
	infoMatrix := precomputeNumOfZeroes(matrix)
	n := len(matrix)
	for topRow := 0; topRow < n; topRow++ {
		for leftCol := 0; leftCol < n; leftCol++ {
			squareLength := 2
			for squareLength <= n-leftCol && squareLength <= n-topRow {
				bottomRow := topRow + squareLength - 1
				rightCol := leftCol + squareLength - 1
				if isSquareOfZeroes(infoMatrix, topRow, leftCol, bottomRow, rightCol) {
					return true
				}
				squareLength++
			}
		}
	}
	return false
}

func isSquareOfZeroes(infoMatrix [][]Info, r1 int, c1 int, r2 int, c2 int) bool {
	squareLength := c2 - c1 + 1
	hasTopBorder := infoMatrix[r1][c1].numZeroesRight >= squareLength
	hasLeftBorder := infoMatrix[r1][c1].numZeroesBelow >= squareLength
	hasBottomBorder := infoMatrix[r2][c1].numZeroesRight >= squareLength
	hasRightBorder := infoMatrix[r1][c2].numZeroesBelow >= squareLength
	return hasTopBorder && hasLeftBorder && hasBottomBorder && hasRightBorder
}

func precomputeNumOfZeroes(matrix [][]int) [][]Info {
	infoMatrix := make([][]Info, len(matrix))
	for i := range infoMatrix {
		infoMatrix[i] = make([]Info, len(matrix[0]))
	}
	n := len(matrix)
	for row := n - 1; row >= 0; row-- {
		for col := n - 1; col >= 0; col-- {
			numZeroes := 0
			if matrix[row][col] == 0 {
				numZeroes = 1
			}
			if row == n-1 {
				infoMatrix[row][col].numZeroesBelow = numZeroes
			} else {
				infoMatrix[row][col].numZeroesBelow = numZeroes + infoMatrix[row+1][col].numZeroesBelow
			}
			if col == n-1 {
				infoMatrix[row][col].numZeroesRight = numZeroes
			} else {
				infoMatrix[row][col].numZeroesRight = numZeroes + infoMatrix[row][col+1].numZeroesRight
			}
		}
	}
	return infoMatrix
}

type Info struct {
	numZeroesBelow int
	numZeroesRight int
}`,
		PythonSolution: `def squareOfZeroes(matrix: List[List[int]]) -> bool:
    info_matrix = precompute_num_of_zeroes(matrix)
    n = len(matrix)
    for top_row in range(n):
        for left_col in range(n):
            square_length = 2
            while square_length <= n - left_col and square_length <= n - top_row:
                bottom_row = top_row + square_length - 1
                right_col = left_col + square_length - 1
                if is_square_of_zeroes(info_matrix, top_row, left_col, bottom_row, right_col):
                    return True
                square_length += 1
    return False

def is_square_of_zeroes(info_matrix, r1, c1, r2, c2):
    square_length = c2 - c1 + 1
    has_top_border = info_matrix[r1][c1].num_zeroes_right >= square_length
    has_left_border = info_matrix[r1][c1].num_zeroes_below >= square_length
    has_bottom_border = info_matrix[r2][c1].num_zeroes_right >= square_length
    has_right_border = info_matrix[r1][c2].num_zeroes_below >= square_length
    return has_top_border and has_left_border and has_bottom_border and has_right_border

def precompute_num_of_zeroes(matrix):
    n = len(matrix)
    info_matrix = [[Info(0, 0) for _ in range(n)] for _ in range(n)]
    for row in range(n - 1, -1, -1):
        for col in range(n - 1, -1, -1):
            num_zeroes = 1 if matrix[row][col] == 0 else 0
            if row == n - 1:
                info_matrix[row][col].num_zeroes_below = num_zeroes
            else:
                info_matrix[row][col].num_zeroes_below = num_zeroes + info_matrix[row + 1][col].num_zeroes_below
            if col == n - 1:
                info_matrix[row][col].num_zeroes_right = num_zeroes
            else:
                info_matrix[row][col].num_zeroes_right = num_zeroes + info_matrix[row][col + 1].num_zeroes_right
    return info_matrix

class Info:
    def __init__(self, num_zeroes_below, num_zeroes_right):
        self.num_zeroes_below = num_zeroes_below
        self.num_zeroes_right = num_zeroes_right`,
		Explanation: `Finding a square of zeroes requires checking if borders of squares are all zeros. Precomputing zero counts in each direction enables efficient border checking without scanning each border individually.

The algorithm first builds an info matrix where each cell stores the number of consecutive zeros going right and down from that position. This is computed by scanning from bottom-right to top-left, accumulating zeros. Then, for each possible top-left corner and square size, it checks if all four borders have sufficient zeros by using the precomputed counts.

This approach avoids repeatedly scanning borders by using the precomputed information. For a square starting at (r, c) with size s, it checks: right border has s zeros going right from (r, c), bottom border has s zeros going down from (r, c+s-1), left border has s zeros going down from (r, c), and top border has s zeros going right from (r+s-1, c).

Edge cases handled include: matrices with no squares of zeroes (returns false), matrices where entire matrix is zeros (finds largest square), squares at matrix boundaries (handled by precomputation), and 2x2 squares (minimum valid size).

An alternative O(n⁴) approach would check each square's borders directly, but the precomputation approach is more efficient. The time complexity is O(n²) for precomputation plus O(n³) for checking all positions and sizes, giving overall O(n³). The space complexity is O(n²) for the info matrix.

Think of this as finding picture frames: you first mark how many zeros extend in each direction from each position, then for each possible frame size and position, you check if all four sides have enough zeros to form a complete border.`,
	},
	{
		ID:          234,
		Title:       "A* Algorithm",
		Description: "You're given a two-dimensional array containing 0s and 1s, where 0s represent passable space and 1s represent impassable walls. The array will always contain exactly one \"S\" (start) and exactly one \"E\" (end). Write a function that returns the shortest path from the start to the end using an A* search algorithm, where the distance from the start to the end is measured using Manhattan distance.",
		Difficulty:  "Very Hard",
		Topic:       "Graphs",
		Signature:   "func aStarAlgorithm(startRow int, startCol int, endRow int, endCol int, graph [][]int) [][]int",
		TestCases: []TestCase{
			{Input: "startRow = 0, startCol = 1, endRow = 4, endCol = 3, graph = [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0], [1, 0, 1, 1, 1], [0, 0, 0, 0, 0]]", Expected: "Shortest path"},
		},
		Solution: `import "container/heap"

type Node struct {
	row    int
	col    int
	g      int
	h      int
	f      int
	parent *Node
}

type NodeHeap []*Node

func (h NodeHeap) Len() int           { return len(h) }
func (h NodeHeap) Less(i, j int) bool { return h[i].f < h[j].f }
func (h NodeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *NodeHeap) Push(x interface{}) { *h = append(*h, x.(*Node)) }
func (h *NodeHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func aStarAlgorithm(startRow int, startCol int, endRow int, endCol int, graph [][]int) [][]int {
	nodes := initializeNodes(graph)
	startNode := nodes[startRow][startCol]
	endNode := nodes[endRow][endCol]
	startNode.g = 0
	startNode.h = calculateManhattanDistance(startNode, endNode)
	startNode.f = startNode.g + startNode.h
	nodesToVisit := &NodeHeap{startNode}
	heap.Init(nodesToVisit)
	visited := make(map[string]bool)
	for nodesToVisit.Len() > 0 {
		currentNode := heap.Pop(nodesToVisit).(*Node)
		currentNodeString := fmt.Sprintf("%d-%d", currentNode.row, currentNode.col)
		if visited[currentNodeString] {
			continue
		}
		visited[currentNodeString] = true
		if currentNode == endNode {
			break
		}
		neighbors := getNeighboringNodes(currentNode, nodes)
		for _, neighbor := range neighbors {
			if visited[fmt.Sprintf("%d-%d", neighbor.row, neighbor.col)] {
				continue
			}
			if graph[neighbor.row][neighbor.col] == 1 {
				continue
			}
			tentativeGScore := currentNode.g + 1
			if tentativeGScore >= neighbor.g {
				continue
			}
			neighbor.parent = currentNode
			neighbor.g = tentativeGScore
			neighbor.h = calculateManhattanDistance(neighbor, endNode)
			neighbor.f = neighbor.g + neighbor.h
			heap.Push(nodesToVisit, neighbor)
		}
	}
	return reconstructPath(endNode)
}

func initializeNodes(graph [][]int) [][]*Node {
	nodes := make([][]*Node, len(graph))
	for i := range nodes {
		nodes[i] = make([]*Node, len(graph[0]))
		for j := range nodes[i] {
			nodes[i][j] = &Node{row: i, col: j, g: math.MaxInt32}
		}
	}
	return nodes
}

func calculateManhattanDistance(nodeA *Node, nodeB *Node) int {
	return abs(nodeA.row-nodeB.row) + abs(nodeA.col-nodeB.col)
}

func getNeighboringNodes(node *Node, nodes [][]*Node) []*Node {
	neighbors := []*Node{}
	numRows := len(nodes)
	numCols := len(nodes[0])
	row, col := node.row, node.col
	if row < numRows-1 {
		neighbors = append(neighbors, nodes[row+1][col])
	}
	if row > 0 {
		neighbors = append(neighbors, nodes[row-1][col])
	}
	if col < numCols-1 {
		neighbors = append(neighbors, nodes[row][col+1])
	}
	if col > 0 {
		neighbors = append(neighbors, nodes[row][col-1])
	}
	return neighbors
}

func reconstructPath(endNode *Node) [][]int {
	if endNode.parent == nil {
		return [][]int{}
	}
	currentNode := endNode
	path := [][]int{}
	for currentNode != nil {
		path = append([][]int{{currentNode.row, currentNode.col}}, path...)
		currentNode = currentNode.parent
	}
	return path
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}`,
		PythonSolution: `import heapq

def aStarAlgorithm(startRow: int, startCol: int, endRow: int, endCol: int, graph: List[List[int]]) -> List[List[int]]:
    nodes = initialize_nodes(graph)
    start_node = nodes[startRow][startCol]
    end_node = nodes[endRow][endCol]
    start_node.g = 0
    start_node.h = calculate_manhattan_distance(start_node, end_node)
    start_node.f = start_node.g + start_node.h
    nodes_to_visit = [(start_node.f, start_node)]
    heapq.heapify(nodes_to_visit)
    visited = set()
    while nodes_to_visit:
        current_node = heapq.heappop(nodes_to_visit)[1]
        current_node_string = f"{current_node.row}-{current_node.col}"
        if current_node_string in visited:
            continue
        visited.add(current_node_string)
        if current_node == end_node:
            break
        neighbors = get_neighboring_nodes(current_node, nodes)
        for neighbor in neighbors:
            if f"{neighbor.row}-{neighbor.col}" in visited:
                continue
            if graph[neighbor.row][neighbor.col] == 1:
                continue
            tentative_g_score = current_node.g + 1
            if tentative_g_score >= neighbor.g:
                continue
            neighbor.parent = current_node
            neighbor.g = tentative_g_score
            neighbor.h = calculate_manhattan_distance(neighbor, end_node)
            neighbor.f = neighbor.g + neighbor.h
            heapq.heappush(nodes_to_visit, (neighbor.f, neighbor))
    return reconstruct_path(end_node)

class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.g = float('inf')
        self.h = 0
        self.f = 0
        self.parent = None

def initialize_nodes(graph):
    nodes = [[Node(i, j) for j in range(len(graph[0]))] for i in range(len(graph))]
    return nodes

def calculate_manhattan_distance(node_a, node_b):
    return abs(node_a.row - node_b.row) + abs(node_a.col - node_b.col)

def get_neighboring_nodes(node, nodes):
    neighbors = []
    num_rows = len(nodes)
    num_cols = len(nodes[0])
    row, col = node.row, node.col
    if row < num_rows - 1:
        neighbors.append(nodes[row + 1][col])
    if row > 0:
        neighbors.append(nodes[row - 1][col])
    if col < num_cols - 1:
        neighbors.append(nodes[row][col + 1])
    if col > 0:
        neighbors.append(nodes[row][col - 1])
    return neighbors

def reconstruct_path(end_node):
    if end_node.parent is None:
        return []
    current_node = end_node
    path = []
    while current_node is not None:
        path.insert(0, [current_node.row, current_node.col])
        current_node = current_node.parent
    return path`,
		Explanation: `A* algorithm finds the shortest path in a graph by combining actual distance traveled (g) with an admissible heuristic estimate to the goal (h). The evaluation function f = g + h guides the search toward the goal while ensuring optimality.

The algorithm maintains an open set (min-heap) of nodes to explore, prioritized by f value. It starts with the start node and repeatedly extracts the node with minimum f. For each neighbor, it calculates the tentative g value (distance from start). If this is better than the current g value, it updates the neighbor and adds it to the open set. The heuristic h (Manhattan distance) provides an optimistic estimate of remaining distance.

The algorithm terminates when the goal is reached. The path is reconstructed by following parent pointers backwards from the goal node. A* is optimal when the heuristic is admissible (never overestimates) and consistent.

Edge cases handled include: unreachable goals (algorithm exhausts open set, returns empty path), multiple optimal paths (finds one), graphs with obstacles (heuristic guides around them), and start equals goal (returns single-node path).

An alternative approach would use Dijkstra's algorithm (h=0), but A* is faster for pathfinding. The time complexity is O(v log v) where v is vertices, as each vertex is processed once and heap operations are O(log v). The space complexity is O(v) for the open set and parent pointers.

Think of this as finding the shortest route on a map: you track how far you've traveled (g) and estimate how far remains (h), always exploring the most promising path first, which leads you efficiently to your destination.`,
	},
	{
		ID:          235,
		Title:       "Rectangle Mania",
		Description: "Write a function that takes in a list of Cartesian coordinates and returns the number of rectangles formed by these coordinates. A rectangle must have its four corners amongst the coordinates. The rectangles don't need to be aligned with the x and y axes.",
		Difficulty:  "Very Hard",
		Topic:       "Graphs",
		Signature:   "func rectangleMania(coords [][]int) int",
		TestCases: []TestCase{
			{Input: "coords = [[0, 0], [0, 1], [1, 1], [1, 0], [2, 1], [2, 0], [3, 1], [3, 0]]", Expected: "6"},
		},
		Solution: `func rectangleMania(coords [][]int) int {
	coordsTable := getCoordsTable(coords)
	return getRectangleCount(coords, coordsTable)
}

func getCoordsTable(coords [][]int) map[string]bool {
	coordsTable := make(map[string]bool)
	for _, coord := range coords {
		coordString := coordToString(coord[0], coord[1])
		coordsTable[coordString] = true
	}
	return coordsTable
}

func getRectangleCount(coords [][]int, coordsTable map[string]bool) int {
	rectangleCount := 0
	for i := 0; i < len(coords); i++ {
		for j := i + 1; j < len(coords); j++ {
			if !isInUpperRight(coords[i], coords[j]) {
				continue
			}
			upperCoordString := coordToString(coords[i][0], coords[j][1])
			rightCoordString := coordToString(coords[j][0], coords[i][1])
			if coordsTable[upperCoordString] && coordsTable[rightCoordString] {
				rectangleCount++
			}
		}
	}
	return rectangleCount
}

func isInUpperRight(coord1 []int, coord2 []int) bool {
	return coord2[0] > coord1[0] && coord2[1] > coord1[1]
}

func coordToString(x int, y int) string {
	return fmt.Sprintf("%d-%d", x, y)
}`,
		PythonSolution: `def rectangleMania(coords: List[List[int]]) -> int:
    coords_table = get_coords_table(coords)
    return get_rectangle_count(coords, coords_table)

def get_coords_table(coords):
    return {f"{coord[0]}-{coord[1]}": True for coord in coords}

def get_rectangle_count(coords, coords_table):
    rectangle_count = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if not is_in_upper_right(coords[i], coords[j]):
                continue
            upper_coord_string = f"{coords[i][0]}-{coords[j][1]}"
            right_coord_string = f"{coords[j][0]}-{coords[i][1]}"
            if upper_coord_string in coords_table and right_coord_string in coords_table:
                rectangle_count += 1
    return rectangle_count

def is_in_upper_right(coord1, coord2):
    return coord2[0] > coord1[0] and coord2[1] > coord1[1]`,
		Explanation: `Counting rectangles with sides parallel to axes requires identifying four points that form rectangle corners. The key insight is that for two points to be opposite corners, the other two corners (with swapped coordinates) must also exist.

The algorithm stores all coordinates in a hash set for O(1) lookup. Then it considers all pairs of points. For each pair where one point is upper-right of the other (both x and y coordinates are greater), it checks if the other two required corners exist: the point with the first point's x and second point's y, and the point with the second point's x and first point's y. If all four corners exist, a rectangle is found.

This approach efficiently finds all rectangles by checking point pairs rather than trying all combinations of four points, which would be O(n⁴). The hash set lookup makes checking point existence O(1), and the upper-right constraint ensures we don't count the same rectangle multiple times.

Edge cases handled include: fewer than 4 points (no rectangles possible), points forming squares (counted as rectangles), points with duplicate coordinates (handled by set), and points where no rectangles exist (returns 0).

An alternative O(n⁴) approach would try all combinations of four points, but the pair-based approach is more efficient. The time complexity is O(n²) where n is the number of points, as we check all pairs. The space complexity is O(n) for the coordinate set.

Think of this as finding rectangles on graph paper: for any two points that could be opposite corners, you check if the other two corners exist, and if they do, you've found a rectangle.`,
	},
	{
		ID:          236,
		Title:       "Detect Arbitrage",
		Description: "You're given a two-dimensional array of exchange rates, where exchangeRates[i][j] represents the exchange rate from currency i to currency j. Write a function that returns a boolean representing whether an arbitrage opportunity exists (i.e., whether you can exchange currencies in a cycle and end up with more than you started with).",
		Difficulty:  "Very Hard",
		Topic:       "Graphs",
		Signature:   "func detectArbitrage(exchangeRates [][]float64) bool",
		TestCases: []TestCase{
			{Input: "exchangeRates = [[1.0, 0.8631, 0.5903], [1.1586, 1.0, 0.6849], [1.6939, 1.46, 1.0]]", Expected: "true"},
		},
		Solution: `func detectArbitrage(exchangeRates [][]float64) bool {
	logExchangeRates := convertToLogMatrix(exchangeRates)
	return foundNegativeWeightCycle(logExchangeRates, 0)
}

func foundNegativeWeightCycle(graph [][]float64, start int) bool {
	distancesFromStart := make([]float64, len(graph))
	for i := range distancesFromStart {
		distancesFromStart[i] = math.MaxFloat64
	}
	distancesFromStart[start] = 0
	for i := 0; i < len(graph)-1; i++ {
		if !relaxEdgesAndUpdateDistances(graph, distancesFromStart) {
			return false
		}
	}
	return relaxEdgesAndUpdateDistances(graph, distancesFromStart)
}

func relaxEdgesAndUpdateDistances(graph [][]float64, distances []float64) bool {
	updated := false
	for sourceIdx := 0; sourceIdx < len(graph); sourceIdx++ {
		edges := graph[sourceIdx]
		for destinationIdx := 0; destinationIdx < len(edges); destinationIdx++ {
			edgeWeight := edges[destinationIdx]
			newDistanceToDestination := distances[sourceIdx] + edgeWeight
			if newDistanceToDestination < distances[destinationIdx] {
				updated = true
				distances[destinationIdx] = newDistanceToDestination
			}
		}
	}
	return updated
}

func convertToLogMatrix(matrix [][]float64) [][]float64 {
	newMatrix := make([][]float64, len(matrix))
	for i := range newMatrix {
		newMatrix[i] = make([]float64, len(matrix[0]))
		for j := range newMatrix[i] {
			newMatrix[i][j] = -math.Log(matrix[i][j])
		}
	}
	return newMatrix
}`,
		PythonSolution: `def detectArbitrage(exchangeRates: List[List[float]]) -> bool:
    log_exchange_rates = convert_to_log_matrix(exchangeRates)
    return found_negative_weight_cycle(log_exchange_rates, 0)

def found_negative_weight_cycle(graph, start):
    distances_from_start = [float('inf')] * len(graph)
    distances_from_start[start] = 0
    for i in range(len(graph) - 1):
        if not relax_edges_and_update_distances(graph, distances_from_start):
            return False
    return relax_edges_and_update_distances(graph, distances_from_start)

def relax_edges_and_update_distances(graph, distances):
    updated = False
    for source_idx in range(len(graph)):
        edges = graph[source_idx]
        for destination_idx in range(len(edges)):
            edge_weight = edges[destination_idx]
            new_distance_to_destination = distances[source_idx] + edge_weight
            if new_distance_to_destination < distances[destination_idx]:
                updated = True
                distances[destination_idx] = new_distance_to_destination
    return updated

def convert_to_log_matrix(matrix):
    return [[-math.log(matrix[i][j]) for j in range(len(matrix[0]))] for i in range(len(matrix))]`,
		Explanation: `Detecting arbitrage opportunities in currency exchange rates requires finding cycles where multiplying exchange rates gives a value greater than 1. By converting to logarithmic space, this becomes a negative cycle detection problem.

The algorithm converts the exchange rate matrix to a log matrix by taking the negative logarithm of each rate. This transforms the problem: finding a cycle where the product of rates > 1 becomes finding a cycle where the sum of log values < 0 (a negative cycle). Bellman-Ford algorithm efficiently detects negative cycles.

Bellman-Ford performs v-1 relaxation iterations (where v is the number of currencies). If distances can still be updated after v-1 iterations, a negative cycle exists, indicating an arbitrage opportunity. The algorithm relaxes all edges in each iteration, updating distances if a shorter path is found.

Edge cases handled include: no arbitrage opportunities (no negative cycles, returns false), arbitrage opportunities involving all currencies (detected correctly), exchange rates of 1 (no profit, handled correctly), and disconnected currency graphs (handled by Bellman-Ford).

An alternative approach would try all cycles, but that's exponential. Bellman-Ford achieves O(v³) time for v currencies, as we perform v-1 iterations and relax O(v²) edges per iteration. The space complexity is O(v²) for the log matrix and O(v) for distances.

Think of this as finding a money-making loop: you convert exchange rates to costs (negative logs), then look for a cycle where the total cost is negative, meaning you end up with more money than you started.`,
	},
	{
		ID:          237,
		Title:       "LRU Cache",
		Description: "Implement an LRU (Least Recently Used) cache. It should be able to be initialized with a cache size n, and contain the following methods: set(key, value): sets key to value. If there are already n items in the cache and we're adding a new item, then it should also remove the least recently used item. get(key): gets the value at key. If no such key exists, return null. Both methods should run in O(1) time.",
		Difficulty:  "Very Hard",
		Topic:       "System Design",
		Signature:   "type LRUCache struct {}\nfunc NewLRUCache(capacity int) *LRUCache\nfunc (cache *LRUCache) Get(key int) int\nfunc (cache *LRUCache) Put(key int, value int)",
		TestCases: []TestCase{
			{Input: "Cache operations", Expected: "Correct cache behavior"},
		},
		Solution: `type LRUCache struct {
	maxSize      int
	currSize     int
	cache        map[int]*DoublyLinkedListNode
	listOfMostRecent *DoublyLinkedList
}

type DoublyLinkedListNode struct {
	key   int
	value int
	prev  *DoublyLinkedListNode
	next  *DoublyLinkedListNode
}

type DoublyLinkedList struct {
	head *DoublyLinkedListNode
	tail *DoublyLinkedListNode
}

func NewLRUCache(capacity int) *LRUCache {
	return &LRUCache{
		maxSize:         capacity,
		currSize:        0,
		cache:           make(map[int]*DoublyLinkedListNode),
		listOfMostRecent: &DoublyLinkedList{},
	}
}

func (cache *LRUCache) Get(key int) int {
	if node, found := cache.cache[key]; found {
		cache.listOfMostRecent.setHeadTo(node)
		return node.value
	}
	return -1
}

func (cache *LRUCache) Put(key int, value int) {
	if _, found := cache.cache[key]; found {
		cache.cache[key].value = value
		cache.listOfMostRecent.setHeadTo(cache.cache[key])
		return
	}
	if cache.currSize == cache.maxSize {
		keyToRemove := cache.listOfMostRecent.tail.key
		cache.listOfMostRecent.removeTail()
		delete(cache.cache, keyToRemove)
		cache.currSize--
	}
	cache.listOfMostRecent.setHeadTo(&DoublyLinkedListNode{key: key, value: value})
	cache.cache[key] = cache.listOfMostRecent.head
	cache.currSize++
}

func (list *DoublyLinkedList) setHeadTo(node *DoublyLinkedListNode) {
	if list.head == node {
		return
	}
	if list.head == nil {
		list.head = node
		list.tail = node
		return
	}
	if list.head == list.tail {
		list.tail.prev = node
		node.next = list.tail
		list.head = node
		return
	}
	if list.tail == node {
		list.tail = list.tail.prev
		list.tail.next = nil
	} else {
		if node.prev != nil {
			node.prev.next = node.next
		}
		if node.next != nil {
			node.next.prev = node.prev
		}
	}
	node.prev = nil
	node.next = list.head
	list.head.prev = node
	list.head = node
}

func (list *DoublyLinkedList) removeTail() {
	if list.tail == nil {
		return
	}
	if list.tail == list.head {
		list.head = nil
		list.tail = nil
		return
	}
	list.tail = list.tail.prev
	list.tail.next = nil
}`,
		PythonSolution: `class LRUCache:
    def __init__(self, capacity: int):
        self.max_size = capacity
        self.curr_size = 0
        self.cache = {}
        self.list_of_most_recent = DoublyLinkedList()
    
    def get(self, key: int) -> int:
        if key in self.cache:
            self.list_of_most_recent.set_head_to(self.cache[key])
            return self.cache[key].value
        return -1
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache[key].value = value
            self.list_of_most_recent.set_head_to(self.cache[key])
            return
        if self.curr_size == self.max_size:
            key_to_remove = self.list_of_most_recent.tail.key
            self.list_of_most_recent.remove_tail()
            del self.cache[key_to_remove]
            self.curr_size -= 1
        self.list_of_most_recent.set_head_to(DoublyLinkedListNode(key, value))
        self.cache[key] = self.list_of_most_recent.head
        self.curr_size += 1

class DoublyLinkedListNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def set_head_to(self, node):
        if self.head == node:
            return
        if self.head is None:
            self.head = node
            self.tail = node
            return
        if self.head == self.tail:
            self.tail.prev = node
            node.next = self.tail
            self.head = node
            return
        if self.tail == node:
            self.tail = self.tail.prev
            self.tail.next = None
        else:
            if node.prev:
                node.prev.next = node.next
            if node.next:
                node.next.prev = node.prev
        node.prev = None
        node.next = self.head
        self.head.prev = node
        self.head = node
    
    def remove_tail(self):
        if self.tail is None:
            return
        if self.tail == self.head:
            self.head = None
            self.tail = None
            return
        self.tail = self.tail.prev
        self.tail.next = None`,
		Explanation: `Implementing an LRU (Least Recently Used) cache requires O(1) access and insertion while maintaining access order. A combination of a hash map and a doubly linked list achieves this efficiently.

The hash map provides O(1) lookup by key, storing pointers to nodes in the linked list. The doubly linked list maintains access order: the head is the most recently used item, and the tail is the least recently used. When an item is accessed (get), it's moved to the head. When inserting (set) and the cache is full, the tail (least recently used) is removed.

The doubly linked list enables O(1) insertion and deletion at both ends. When moving a node to the head, we update its neighbors' pointers and the head pointer. This maintains the order without requiring shifts or searches.

Edge cases handled include: cache at capacity (removes tail before inserting), accessing existing key (moves to head, updates value), accessing non-existent key (returns null), and cache with single item (head and tail are the same).

An alternative approach would use a queue and hash map, but updating order would require O(n) operations. The doubly linked list approach achieves O(1) for both get and set operations. The space complexity is O(n) where n is the cache capacity.

Think of this as organizing items by recency: you keep a map to find items quickly, and a list to track which was used most recently, always moving accessed items to the front and removing from the back when full.`,
	},
	{
		ID:          238,
		Title:       "Airport Connections",
		Description: "For the purpose of this problem, an airport is considered reachable from another airport if you can travel from one to the other by taking any number of flights. You're given a list of airports (three-letter codes like \"JFK\"), a list of routes (one-way flights from one airport to another like [\"JFK\", \"SFO\"]), and a starting airport. Write a function that returns the minimum number of airport connections (one-way flights) that need to be added in order for someone to be able to reach any airport in the list, starting at the starting airport.",
		Difficulty:  "Very Hard",
		Topic:       "Graphs",
		Signature:   "func airportConnections(airports []string, routes [][]string, startingAirport string) int",
		TestCases: []TestCase{
			{Input: "airports = [\"BGI\", \"CDG\", \"DEL\", \"DOH\", \"DSM\", \"EWR\", \"EYW\", \"HND\", \"ICN\", \"JFK\", \"LGA\", \"LHR\", \"ORD\", \"SAN\", \"SFO\", \"SIN\", \"TLV\", \"BUD\"], routes = [[\"DSM\", \"ORD\"], [\"ORD\", \"BGI\"], [\"BGI\", \"LGA\"], [\"SIN\", \"CDG\"], [\"CDG\", \"BUD\"], [\"CDG\", \"SIN\"], [\"BUD\", \"FRA\"]], startingAirport = \"LGA\"", Expected: "17"},
		},
		Solution: `func airportConnections(airports []string, routes [][]string, startingAirport string) int {
	airportGraph := createAirportGraph(airports, routes)
	unreachableAirportNodes := getUnreachableAirportNodes(airportGraph, airports, startingAirport)
	markUnreachableConnections(airportGraph, unreachableAirportNodes)
	return getMinNumberOfNewConnections(airportGraph, unreachableAirportNodes)
}

func createAirportGraph(airports []string, routes [][]string) map[string]*AirportNode {
	airportGraph := make(map[string]*AirportNode)
	for _, airport := range airports {
		airportGraph[airport] = &AirportNode{Airport: airport, Connections: []string{}, IsReachable: true}
	}
	for _, route := range routes {
		airport, connection := route[0], route[1]
		airportGraph[airport].Connections = append(airportGraph[airport].Connections, connection)
	}
	return airportGraph
}

func getUnreachableAirportNodes(airportGraph map[string]*AirportNode, airports []string, startingAirport string) []*AirportNode {
	visitedAirports := make(map[string]bool)
	depthFirstTraverseAirports(airportGraph, startingAirport, visitedAirports)
	unreachableAirportNodes := []*AirportNode{}
	for _, airport := range airports {
		if visitedAirports[airport] {
			continue
		}
		node := airportGraph[airport]
		node.IsReachable = false
		unreachableAirportNodes = append(unreachableAirportNodes, node)
	}
	return unreachableAirportNodes
}

func depthFirstTraverseAirports(airportGraph map[string]*AirportNode, airport string, visitedAirports map[string]bool) {
	if visitedAirports[airport] {
		return
	}
	visitedAirports[airport] = true
	connections := airportGraph[airport].Connections
	for _, connection := range connections {
		depthFirstTraverseAirports(airportGraph, connection, visitedAirports)
	}
}

func markUnreachableConnections(airportGraph map[string]*AirportNode, unreachableAirportNodes []*AirportNode) {
	for _, airportNode := range unreachableAirportNodes {
		unreachableConnections := []string{}
		visitedAirports := make(map[string]bool)
		depthFirstAddUnreachableConnections(airportGraph, airportNode.Airport, unreachableConnections, visitedAirports)
		airportNode.UnreachableConnections = unreachableConnections
	}
}

func depthFirstAddUnreachableConnections(airportGraph map[string]*AirportNode, airport string, unreachableConnections []string, visitedAirports map[string]bool) {
	if airportGraph[airport].IsReachable {
		return
	}
	if visitedAirports[airport] {
		return
	}
	visitedAirports[airport] = true
	unreachableConnections = append(unreachableConnections, airport)
	connections := airportGraph[airport].Connections
	for _, connection := range connections {
		depthFirstAddUnreachableConnections(airportGraph, connection, unreachableConnections, visitedAirports)
	}
}

func getMinNumberOfNewConnections(airportGraph map[string]*AirportNode, unreachableAirportNodes []*AirportNode) int {
	sort.Slice(unreachableAirportNodes, func(i, j int) bool {
		return len(unreachableAirportNodes[i].UnreachableConnections) > len(unreachableAirportNodes[j].UnreachableConnections)
	})
	numberOfNewConnections := 0
	visitedAirports := make(map[string]bool)
	for _, airportNode := range unreachableAirportNodes {
		if visitedAirports[airportNode.Airport] {
			continue
		}
		numberOfNewConnections++
		for _, airport := range airportNode.UnreachableConnections {
			visitedAirports[airport] = true
		}
	}
	return numberOfNewConnections
}

type AirportNode struct {
	Airport                string
	Connections            []string
	IsReachable            bool
	UnreachableConnections []string
}`,
		PythonSolution: `def airportConnections(airports: List[str], routes: List[List[str]], startingAirport: str) -> int:
    airport_graph = create_airport_graph(airports, routes)
    unreachable_airport_nodes = get_unreachable_airport_nodes(airport_graph, airports, startingAirport)
    mark_unreachable_connections(airport_graph, unreachable_airport_nodes)
    return get_min_number_of_new_connections(airport_graph, unreachable_airport_nodes)

def create_airport_graph(airports, routes):
    airport_graph = {airport: AirportNode(airport) for airport in airports}
    for route in routes:
        airport, connection = route
        airport_graph[airport].connections.append(connection)
    return airport_graph

def get_unreachable_airport_nodes(airport_graph, airports, starting_airport):
    visited_airports = set()
    depth_first_traverse_airports(airport_graph, starting_airport, visited_airports)
    unreachable_airport_nodes = []
    for airport in airports:
        if airport in visited_airports:
            continue
        node = airport_graph[airport]
        node.is_reachable = False
        unreachable_airport_nodes.append(node)
    return unreachable_airport_nodes

def depth_first_traverse_airports(airport_graph, airport, visited_airports):
    if airport in visited_airports:
        return
    visited_airports.add(airport)
    for connection in airport_graph[airport].connections:
        depth_first_traverse_airports(airport_graph, connection, visited_airports)

def mark_unreachable_connections(airport_graph, unreachable_airport_nodes):
    for airport_node in unreachable_airport_nodes:
        unreachable_connections = []
        visited_airports = set()
        depth_first_add_unreachable_connections(airport_graph, airport_node.airport, unreachable_connections, visited_airports)
        airport_node.unreachable_connections = unreachable_connections

def depth_first_add_unreachable_connections(airport_graph, airport, unreachable_connections, visited_airports):
    if airport_graph[airport].is_reachable:
        return
    if airport in visited_airports:
        return
    visited_airports.add(airport)
    unreachable_connections.append(airport)
    for connection in airport_graph[airport].connections:
        depth_first_add_unreachable_connections(airport_graph, connection, unreachable_connections, visited_airports)

def get_min_number_of_new_connections(airport_graph, unreachable_airport_nodes):
    unreachable_airport_nodes.sort(key=lambda x: len(x.unreachable_connections), reverse=True)
    number_of_new_connections = 0
    visited_airports = set()
    for airport_node in unreachable_airport_nodes:
        if airport_node.airport in visited_airports:
            continue
        number_of_new_connections += 1
        for airport in airport_node.unreachable_connections:
            visited_airports.add(airport)
    return number_of_new_connections

class AirportNode:
    def __init__(self, airport):
        self.airport = airport
        self.connections = []
        self.is_reachable = True
        self.unreachable_connections = []`,
		Explanation: `Finding the minimum number of new airport connections needed to make all airports reachable requires identifying unreachable airports and strategically connecting them. A greedy approach minimizes the number of new connections needed.

The algorithm first marks all airports reachable from the starting airport using DFS. Unreachable airports are those not visited. For each unreachable airport, it finds all other unreachable airports reachable from it (also using DFS). Then it greedily adds connections starting from unreachable airports that can reach the most other unreachable airports, as this maximizes the number of airports made reachable per new connection.

This greedy strategy is optimal because connecting an airport that reaches many unreachable airports makes all of them reachable with a single connection, minimizing the total number of connections needed.

Edge cases handled include: all airports already reachable (0 new connections needed), disconnected components (each needs at least one connection), airports with no outgoing routes (must be connected to), and airports forming cycles (handled correctly by DFS).

An alternative approach would try all combinations of connections, but that's exponential. The greedy approach achieves O(a*(a+r)) time where a is airports and r is routes, as we perform DFS from each unreachable airport. The space complexity is O(a+r) for the graph representation and visited sets.

Think of this as connecting isolated islands: you find which islands can reach other islands, then connect the islands that can reach the most other islands first, minimizing the total bridges needed.`,
	},
	{
		ID:          239,
		Title:       "Find Loop",
		Description: "Write a function that takes in the head of a Singly Linked List that contains a loop. The function should return the node from which the loop originates in constant space.",
		Difficulty:  "Very Hard",
		Topic:       "Linked Lists",
		Signature:   "func findLoop(head *ListNode) *ListNode",
		TestCases: []TestCase{
			{Input: "head = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], loop starts at node 4", Expected: "Node 4"},
		},
		Solution: `func findLoop(head *ListNode) *ListNode {
	first := head.Next
	second := head.Next.Next
	for first != second {
		first = first.Next
		second = second.Next.Next
	}
	first = head
	for first != second {
		first = first.Next
		second = second.Next
	}
	return first
}`,
		PythonSolution: `def findLoop(head: Optional[ListNode]) -> Optional[ListNode]:
    first = head.next
    second = head.next.next
    while first != second:
        first = first.next
        second = second.next.next
    first = head
    while first != second:
        first = first.next
        second = second.next
    return first`,
		Explanation: `Finding the start of a loop in a linked list using Floyd's cycle detection algorithm requires two phases: detecting the cycle and finding its start. The algorithm uses two pointers moving at different speeds to achieve this in O(1) space.

In the first phase, a slow pointer moves one step at a time while a fast pointer moves two steps. If there's a cycle, they will eventually meet. The key insight is that when they meet, the slow pointer has traveled distance d + p (where d is distance to cycle start, p is distance within cycle), and the fast pointer has traveled 2(d + p) = d + p + nL (where L is cycle length, n is number of cycles). This gives d + p = nL, meaning d = nL - p.

In the second phase, we reset the slow pointer to the head and move both pointers one step at a time. The slow pointer travels d steps to reach the cycle start, while the fast pointer (starting from the meeting point) travels d = nL - p steps, which also reaches the cycle start. They meet at the cycle start.

Edge cases handled include: no cycle (fast pointer reaches null, returns null), cycle at head (both pointers meet at head), single-node cycle (detected correctly), and cycles of various lengths (algorithm works for all).

An alternative approach would use a hash set to track visited nodes, but that requires O(n) space. Floyd's algorithm achieves O(n) time with O(1) space. The time complexity is O(n) as we traverse the list at most twice. The space complexity is O(1) as we only use two pointers.

Think of this as two runners on a track: if one runs twice as fast and they start together, when the fast runner laps the slow runner, the distance from the start to where they meet tells you where the track loop begins.`,
	},
	{
		ID:          240,
		Title:       "Reverse Linked List",
		Description: "Write a function that takes in the head of a Singly Linked List, reverses the list in place (i.e., doesn't create a brand new list), and returns its new head. Each ListNode has an integer value as well as a next node pointing to the next node in the list or to None / null if it's the tail of the list.",
		Difficulty:  "Very Hard",
		Topic:       "Linked Lists",
		Signature:   "func reverseLinkedList(head *ListNode) *ListNode",
		TestCases: []TestCase{
			{Input: "head = [0, 1, 2, 3, 4, 5]", Expected: "[5, 4, 3, 2, 1, 0]"},
		},
		Solution: `func reverseLinkedList(head *ListNode) *ListNode {
	var previousNode *ListNode
	currentNode := head
	for currentNode != nil {
		nextNode := currentNode.Next
		currentNode.Next = previousNode
		previousNode = currentNode
		currentNode = nextNode
	}
	return previousNode
}`,
		PythonSolution: `def reverseLinkedList(head: Optional[ListNode]) -> Optional[ListNode]:
    previous_node = None
    current_node = head
    while current_node is not None:
        next_node = current_node.next
        current_node.next = previous_node
        previous_node = current_node
        current_node = next_node
    return previous_node`,
		Explanation: `Reversing a linked list iteratively requires carefully managing pointers to avoid losing references. The algorithm uses three pointers: previous, current, and next to traverse and reverse links in place.

The algorithm starts with previous as nil and current as head. In each iteration, it saves the next node, reverses the link by setting current.next to previous, then advances all three pointers. When current becomes nil, previous points to the new head (the original tail).

This approach efficiently reverses the list in a single pass without using extra space. The key is saving the next node before reversing the link, as reversing destroys the reference needed to continue traversal.

Edge cases handled include: empty list (returns nil), single-node list (returns that node), two-node list (reverses correctly), and lists of various lengths (all handled uniformly).

An alternative recursive approach would also work but uses O(n) stack space. The iterative approach achieves O(n) time with O(1) space. The time complexity is O(n) as we visit each node once. The space complexity is O(1) as we only use a few pointers.

Think of this as flipping a chain: you hold onto the previous link, move to the next link, and point it back to the previous one, continuing until you've flipped the entire chain.`,
	},
	{
		ID:          241,
		Title:       "Linked List Palindrome",
		Description: "Write a function that takes in the head of a Singly Linked List and returns a boolean representing whether the linked list's nodes form a palindrome. Your function shouldn't make use of any auxiliary data structure.",
		Difficulty:  "Very Hard",
		Topic:       "Linked Lists",
		Signature:   "func linkedListPalindrome(head *ListNode) bool",
		TestCases: []TestCase{
			{Input: "head = [0, 1, 2, 2, 1, 0]", Expected: "true"},
			{Input: "head = [0, 1, 2, 1, 0]", Expected: "true"},
		},
		Solution: `func linkedListPalindrome(head *ListNode) bool {
	slowNode := head
	fastNode := head
	for fastNode != nil && fastNode.Next != nil {
		slowNode = slowNode.Next
		fastNode = fastNode.Next.Next
	}
	reversedSecondHalfNode := reverseLinkedList(slowNode)
	firstHalfNode := head
	for reversedSecondHalfNode != nil {
		if reversedSecondHalfNode.Value != firstHalfNode.Value {
			return false
		}
		reversedSecondHalfNode = reversedSecondHalfNode.Next
		firstHalfNode = firstHalfNode.Next
	}
	return true
}

func reverseLinkedList(head *ListNode) *ListNode {
	var previousNode *ListNode
	currentNode := head
	for currentNode != nil {
		nextNode := currentNode.Next
		currentNode.Next = previousNode
		previousNode = currentNode
		currentNode = nextNode
	}
	return previousNode
}`,
		PythonSolution: `def linkedListPalindrome(head: Optional[ListNode]) -> bool:
    slow_node = head
    fast_node = head
    while fast_node is not None and fast_node.next is not None:
        slow_node = slow_node.next
        fast_node = fast_node.next.next
    reversed_second_half_node = reverse_linked_list(slow_node)
    first_half_node = head
    while reversed_second_half_node is not None:
        if reversed_second_half_node.val != first_half_node.val:
            return False
        reversed_second_half_node = reversed_second_half_node.next
        first_half_node = first_half_node.next
    return True

def reverse_linked_list(head):
    previous_node = None
    current_node = head
    while current_node is not None:
        next_node = current_node.next
        current_node.next = previous_node
        previous_node = current_node
        current_node = next_node
    return previous_node`,
		Explanation: `Checking if a linked list forms a palindrome without auxiliary structures requires modifying the list temporarily. The algorithm finds the middle, reverses the second half, compares both halves, then restores the list.

The algorithm uses two pointers (slow and fast) to find the middle. When fast reaches the end, slow is at the middle. It then reverses the second half starting from slow.next. After reversal, it compares the first half with the reversed second half node by node. Finally, it reverses the second half again to restore the original structure.

This approach efficiently checks palindromes in O(1) extra space by reusing the list structure. The reversal operations are O(n) but necessary to avoid using O(n) space for storing values.

Edge cases handled include: empty list (returns true), single-node list (returns true), even-length palindromes (handled correctly), odd-length palindromes (middle node ignored in comparison), and non-palindromes (returns false early).

An alternative approach would copy values to an array, but that uses O(n) space. The reversal approach achieves O(n) time with O(1) space. The time complexity is O(n) for finding middle, reversing, comparing, and restoring. The space complexity is O(1) as we only use pointers.

Think of this as checking if a word reads the same forwards and backwards: you split it in half, reverse one half, compare them, then reverse back to restore the original.`,
	},
	{
		ID:          242,
		Title:       "Zip Linked List",
		Description: "You're given the head of a Singly Linked List of arbitrary length k. Write a function that zips the Linked List in place (i.e., doesn't create a brand new list) and returns its head. A Linked List is zipped if its nodes are in the following order, where k is the length of the Linked List: 1st node -> kth node -> 2nd node -> (k-1)th node -> 3rd node -> (k-2)th node -> ...",
		Difficulty:  "Very Hard",
		Topic:       "Linked Lists",
		Signature:   "func zipLinkedList(linkedList *ListNode) *ListNode",
		TestCases: []TestCase{
			{Input: "linkedList = [1, 2, 3, 4, 5, 6]", Expected: "[1, 6, 2, 5, 3, 4]"},
		},
		Solution: `func zipLinkedList(linkedList *ListNode) *ListNode {
	if linkedList.Next == nil || linkedList.Next.Next == nil {
		return linkedList
	}
	firstHalfHead := linkedList
	secondHalfHead := splitLinkedList(linkedList)
	reversedSecondHalfHead := reverseLinkedList(secondHalfHead)
	return interweaveLinkedLists(firstHalfHead, reversedSecondHalfHead)
}

func splitLinkedList(linkedList *ListNode) *ListNode {
	slowIterator := linkedList
	fastIterator := linkedList
	for fastIterator != nil && fastIterator.Next != nil {
		slowIterator = slowIterator.Next
		fastIterator = fastIterator.Next.Next
	}
	secondHalfHead := slowIterator.Next
	slowIterator.Next = nil
	return secondHalfHead
}

func interweaveLinkedLists(linkedList1 *ListNode, linkedList2 *ListNode) *ListNode {
	linkedList1Iterator := linkedList1
	linkedList2Iterator := linkedList2
	for linkedList1Iterator != nil && linkedList2Iterator != nil {
		linkedList1Next := linkedList1Iterator.Next
		linkedList2Next := linkedList2Iterator.Next
		linkedList1Iterator.Next = linkedList2Iterator
		linkedList2Iterator.Next = linkedList1Next
		linkedList1Iterator = linkedList1Next
		linkedList2Iterator = linkedList2Next
	}
	return linkedList1
}

func reverseLinkedList(head *ListNode) *ListNode {
	var previousNode *ListNode
	currentNode := head
	for currentNode != nil {
		nextNode := currentNode.Next
		currentNode.Next = previousNode
		previousNode = currentNode
		currentNode = nextNode
	}
	return previousNode
}`,
		PythonSolution: `def zipLinkedList(linkedList: Optional[ListNode]) -> Optional[ListNode]:
    if linkedList.next is None or linkedList.next.next is None:
        return linkedList
    first_half_head = linkedList
    second_half_head = split_linked_list(linkedList)
    reversed_second_half_head = reverse_linked_list(second_half_head)
    return interweave_linked_lists(first_half_head, reversed_second_half_head)

def split_linked_list(linked_list):
    slow_iterator = linked_list
    fast_iterator = linked_list
    while fast_iterator is not None and fast_iterator.next is not None:
        slow_iterator = slow_iterator.next
        fast_iterator = fast_iterator.next.next
    second_half_head = slow_iterator.next
    slow_iterator.next = None
    return second_half_head

def interweave_linked_lists(linked_list1, linked_list2):
    linked_list1_iterator = linked_list1
    linked_list2_iterator = linked_list2
    while linked_list1_iterator is not None and linked_list2_iterator is not None:
        linked_list1_next = linked_list1_iterator.next
        linked_list2_next = linked_list2_iterator.next
        linked_list1_iterator.next = linked_list2_iterator
        linked_list2_iterator.next = linked_list1_next
        linked_list1_iterator = linked_list1_next
        linked_list2_iterator = linked_list2_next
    return linked_list1

def reverse_linked_list(head):
    previous_node = None
    current_node = head
    while current_node is not None:
        next_node = current_node.next
        current_node.next = previous_node
        previous_node = current_node
        current_node = next_node
    return previous_node`,
		Explanation: `Zipping a linked list requires interweaving nodes from the first half with nodes from the reversed second half. The algorithm splits the list, reverses the second half, then merges them alternately.

The algorithm first finds the middle using two pointers (slow and fast). When fast reaches the end, slow is just past the middle. It splits the list at slow, reverses the second half, then interweaves: first node from first half, first node from reversed second half, second node from first half, etc.

This creates the zipped pattern: 1st -> kth -> 2nd -> (k-1)th -> ... The interweaving process connects nodes alternately from both halves until one half is exhausted.

Edge cases handled include: lists with 0 or 1 nodes (return as-is), lists with 2 nodes (swap them), even-length lists (both halves exhausted together), and odd-length lists (first half has one extra node at end).

An alternative approach would use a stack or array, but that uses O(n) space. The in-place approach achieves O(n) time with O(1) space. The time complexity is O(n) for splitting, reversing, and interweaving. The space complexity is O(1) as we only use pointers.

Think of this as shuffling two decks: you split the deck in half, flip the second half, then interleave cards from both halves one by one.`,
	},
	{
		ID:          243,
		Title:       "Node Swap",
		Description: "Write a function that takes in the head of a Singly Linked List and swaps every pair of adjacent nodes in place and returns its new head. If the input Linked List has an odd number of nodes, its last node should remain the same.",
		Difficulty:  "Very Hard",
		Topic:       "Linked Lists",
		Signature:   "func nodeSwap(head *ListNode) *ListNode",
		TestCases: []TestCase{
			{Input: "head = [0, 1, 2, 3, 4, 5]", Expected: "[1, 0, 3, 2, 5, 4]"},
		},
		Solution: `func nodeSwap(head *ListNode) *ListNode {
	tempNode := &ListNode{Value: 0, Next: head}
	prevNode := tempNode
	for prevNode.Next != nil && prevNode.Next.Next != nil {
		firstNode := prevNode.Next
		secondNode := prevNode.Next.Next
		firstNode.Next = secondNode.Next
		secondNode.Next = firstNode
		prevNode.Next = secondNode
		prevNode = firstNode
	}
	return tempNode.Next
}`,
		PythonSolution: `def nodeSwap(head: Optional[ListNode]) -> Optional[ListNode]:
    temp_node = ListNode(0)
    temp_node.next = head
    prev_node = temp_node
    while prev_node.next is not None and prev_node.next.next is not None:
        first_node = prev_node.next
        second_node = prev_node.next.next
        first_node.next = second_node.next
        second_node.next = first_node
        prev_node.next = second_node
        prev_node = first_node
    return temp_node.next`,
		Explanation: `Swapping every pair of adjacent nodes requires carefully updating pointers to maintain list integrity. A dummy node simplifies edge case handling and provides a consistent starting point.

The algorithm uses a dummy node pointing to the head, with prevNode initially at the dummy. For each pair of nodes, it extracts firstNode and secondNode, updates pointers to swap them: firstNode.next points to secondNode.next, secondNode.next points to firstNode, and prevNode.next points to secondNode. Then prevNode advances to firstNode (now the second in the swapped pair).

The dummy node eliminates special handling for the head, as the first swap updates dummy.next to point to the new head. The algorithm continues until no more pairs exist (prevNode.next or prevNode.next.next is nil).

Edge cases handled include: empty list (returns nil), single-node list (returns as-is), even-length lists (all pairs swapped), and odd-length lists (last node remains unchanged).

An alternative recursive approach would also work but uses O(n) stack space. The iterative approach achieves O(n) time with O(1) space. The time complexity is O(n) as we visit each node once. The space complexity is O(1) as we only use a few pointers.

Think of this as swapping neighbors in a line: you work through pairs, swapping each pair by updating who points to whom, always keeping track of where you are with a pointer.`,
	},
	{
		ID:          244,
		Title:       "Continuous Median",
		Description: "Write a class that's able to receive a stream of numbers and at any point in time return the median of all numbers received thus far. If there are an even number of values, return the average of the two middle values.",
		Difficulty:  "Very Hard",
		Topic:       "Heaps",
		Signature:   "type ContinuousMedianHandler struct {}\nfunc NewContinuousMedianHandler() *ContinuousMedianHandler\nfunc (handler *ContinuousMedianHandler) Insert(number int)\nfunc (handler *ContinuousMedianHandler) GetMedian() float64",
		TestCases: []TestCase{
			{Input: "Insert: 5, 10, 100, 200, 6, 13, 14", Expected: "Median: 10, 7.5, 10, 55, 10, 11.5, 13"},
		},
		Solution: `import "container/heap"

type ContinuousMedianHandler struct {
	lowerNumbers  *MaxHeap
	higherNumbers *MinHeap
	median        float64
}

func NewContinuousMedianHandler() *ContinuousMedianHandler {
	return &ContinuousMedianHandler{
		lowerNumbers:  &MaxHeap{},
		higherNumbers: &MinHeap{},
	}
}

func (handler *ContinuousMedianHandler) Insert(number int) {
	if handler.lowerNumbers.Len() == 0 || number < handler.lowerNumbers.Peek() {
		heap.Push(handler.lowerNumbers, number)
	} else {
		heap.Push(handler.higherNumbers, number)
	}
	rebalanceHeaps(handler.lowerNumbers, handler.higherNumbers)
	updateMedian(handler)
}

func rebalanceHeaps(lowerNumbers *MaxHeap, higherNumbers *MinHeap) {
	if lowerNumbers.Len()-higherNumbers.Len() == 2 {
		heap.Push(higherNumbers, heap.Pop(lowerNumbers))
	} else if higherNumbers.Len()-lowerNumbers.Len() == 2 {
		heap.Push(lowerNumbers, heap.Pop(higherNumbers))
	}
}

func updateMedian(handler *ContinuousMedianHandler) {
	if handler.lowerNumbers.Len() == handler.higherNumbers.Len() {
		handler.median = float64(handler.lowerNumbers.Peek()+handler.higherNumbers.Peek()) / 2.0
	} else if handler.lowerNumbers.Len() > handler.higherNumbers.Len() {
		handler.median = float64(handler.lowerNumbers.Peek())
	} else {
		handler.median = float64(handler.higherNumbers.Peek())
	}
}

func (handler *ContinuousMedianHandler) GetMedian() float64 {
	return handler.median
}

type MaxHeap []int

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i] > h[j] }
func (h MaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *MaxHeap) Push(x interface{}) { *h = append(*h, x.(int)) }
func (h *MaxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}
func (h *MaxHeap) Peek() int { return (*h)[0] }

type MinHeap []int

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x interface{}) { *h = append(*h, x.(int)) }
func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}
func (h *MinHeap) Peek() int { return (*h)[0] }`,
		PythonSolution: `import heapq

class ContinuousMedianHandler:
    def __init__(self):
        self.lower_numbers = []
        self.higher_numbers = []
        self.median = 0.0
    
    def insert(self, number: int) -> None:
        if not self.lower_numbers or number < -self.lower_numbers[0]:
            heapq.heappush(self.lower_numbers, -number)
        else:
            heapq.heappush(self.higher_numbers, number)
        self.rebalance_heaps()
        self.update_median()
    
    def rebalance_heaps(self):
        if len(self.lower_numbers) - len(self.higher_numbers) == 2:
            heapq.heappush(self.higher_numbers, -heapq.heappop(self.lower_numbers))
        elif len(self.higher_numbers) - len(self.lower_numbers) == 2:
            heapq.heappush(self.lower_numbers, -heapq.heappop(self.higher_numbers))
    
    def update_median(self):
        if len(self.lower_numbers) == len(self.higher_numbers):
            self.median = (-self.lower_numbers[0] + self.higher_numbers[0]) / 2.0
        elif len(self.lower_numbers) > len(self.higher_numbers):
            self.median = float(-self.lower_numbers[0])
        else:
            self.median = float(self.higher_numbers[0])
    
    def getMedian(self) -> float:
        return self.median`,
		Explanation: `Maintaining a continuous median as numbers stream in requires efficiently tracking the middle value(s) without sorting the entire collection each time. Two heaps provide an elegant solution.

The algorithm uses a max heap for the lower half and a min heap for the upper half. After each insertion, it ensures the heaps are balanced: the size difference is at most 1. If the lower heap is larger, the median is its maximum. If the upper heap is larger, the median is its minimum. If they're equal, the median is the average of both tops.

When inserting, the number is added to the appropriate heap. If this causes imbalance, the top of the larger heap is moved to the smaller heap. This maintains the invariant that all numbers in the lower heap are less than or equal to all numbers in the upper heap.

Edge cases handled include: first number (median is that number), two numbers (median is average), all numbers same (median remains that number), and numbers in sorted order (heaps maintain balance).

An alternative approach would maintain a sorted array, but insertion would be O(n). The heap approach achieves O(log n) insertion and O(1) median retrieval. The time complexity is O(log n) per insertion for heap operations. The space complexity is O(n) for storing numbers in heaps.

Think of this as maintaining two balanced groups: you keep the smaller half in one pile (max heap) and the larger half in another (min heap), always ensuring they're roughly equal size so the median is easy to find.`,
	},
	{
		ID:          245,
		Title:       "Sort K-Sorted Array",
		Description: "Write a function that takes in a non-negative integer k and a k-sorted array of integers and returns the sorted version of the array. Your function can either sort the array in place or create an entirely new array. A k-sorted array is a partially sorted array in which all elements are at most k positions away from their sorted position.",
		Difficulty:  "Very Hard",
		Topic:       "Heaps",
		Signature:   "func sortKSortedArray(array []int, k int) []int",
		TestCases: []TestCase{
			{Input: "array = [3, 2, 1, 5, 4, 7, 6, 5], k = 3", Expected: "[1, 2, 3, 4, 5, 5, 6, 7]"},
		},
		Solution: `import "container/heap"

type IntHeap []int

func (h IntHeap) Len() int           { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h IntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *IntHeap) Push(x interface{}) { *h = append(*h, x.(int)) }
func (h *IntHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func sortKSortedArray(array []int, k int) []int {
	if len(array) == 0 {
		return array
	}
	minHeapWithKElements := &IntHeap{}
	for i := 0; i < min(k+1, len(array)); i++ {
		heap.Push(minHeapWithKElements, array[i])
	}
	nextIndexToInsertElement := 0
	for idx := k + 1; idx < len(array); idx++ {
		minElement := heap.Pop(minHeapWithKElements).(int)
		array[nextIndexToInsertElement] = minElement
		nextIndexToInsertElement++
		currentElement := array[idx]
		heap.Push(minHeapWithKElements, currentElement)
	}
	for minHeapWithKElements.Len() > 0 {
		minElement := heap.Pop(minHeapWithKElements).(int)
		array[nextIndexToInsertElement] = minElement
		nextIndexToInsertElement++
	}
	return array
}`,
		PythonSolution: `import heapq

def sortKSortedArray(array: List[int], k: int) -> List[int]:
    if len(array) == 0:
        return array
    min_heap_with_k_elements = []
    for i in range(min(k + 1, len(array))):
        heapq.heappush(min_heap_with_k_elements, array[i])
    next_index_to_insert_element = 0
    for idx in range(k + 1, len(array)):
        min_element = heapq.heappop(min_heap_with_k_elements)
        array[next_index_to_insert_element] = min_element
        next_index_to_insert_element += 1
        current_element = array[idx]
        heapq.heappush(min_heap_with_k_elements, current_element)
    while min_heap_with_k_elements:
        min_element = heapq.heappop(min_heap_with_k_elements)
        array[next_index_to_insert_element] = min_element
        next_index_to_insert_element += 1
    return array`,
		Explanation: `Sorting a k-sorted array efficiently leverages the constraint that each element is at most k positions away from its sorted position. A min heap of size k+1 enables optimal sorting.

The algorithm maintains a min heap containing the next k+1 elements that could be the next smallest. It starts by adding the first k+1 elements to the heap. Then it repeatedly extracts the minimum (which is guaranteed to be in its correct position) and adds the next element from the array. This continues until all elements are processed.

The key insight is that in a k-sorted array, the smallest element must be among the first k+1 elements. After extracting it, the next smallest must be among the remaining k+1 elements (the k elements still in the heap plus the next element from the array).

Edge cases handled include: k = 0 (array already sorted, heap size 1), k >= n (heap contains all elements, standard heap sort), k = 1 (heap size 2, minimal sorting needed), and arrays with duplicates (handled correctly by heap).

An alternative O(n log n) approach would use standard sorting, but that ignores the k-sorted property. The heap approach achieves O(n log k) time, which is optimal for k-sorted arrays. The space complexity is O(k) for the heap.

Think of this as sorting with a sliding window: you maintain a small window of candidates (k+1 elements) and always pick the smallest from that window, knowing it's in the right position, then slide the window forward.`,
	},
	{
		ID:          246,
		Title:       "Largest Range",
		Description: "Write a function that takes in an array of integers and returns an array of length 2 representing the largest range of numbers contained in that array. The first number in the output array should be the first number in the range while the second number should be the last number in the range. A range of numbers is defined as a set of numbers that come right after each other in the set of real integers. For instance, the output array [2, 6] represents the range [2, 3, 4, 5, 6], which is a range of length 5. Note that numbers don't need to be sorted or adjacent in the input array in order to form a range.",
		Difficulty:  "Very Hard",
		Topic:       "Arrays",
		Signature:   "func largestRange(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = [1, 11, 3, 0, 15, 5, 2, 4, 10, 7, 12, 6]", Expected: "[0 7]"},
		},
		Solution: `func largestRange(array []int) []int {
	bestRange := []int{}
	longestLength := 0
	nums := make(map[int]bool)
	for _, num := range array {
		nums[num] = true
	}
	for _, num := range array {
		if !nums[num] {
			continue
		}
		nums[num] = false
		currentLength := 1
		left := num - 1
		right := num + 1
		for nums[left] {
			nums[left] = false
			currentLength++
			left--
		}
		for nums[right] {
			nums[right] = false
			currentLength++
			right++
		}
		if currentLength > longestLength {
			longestLength = currentLength
			bestRange = []int{left + 1, right - 1}
		}
	}
	return bestRange
}`,
		PythonSolution: `def largestRange(array: List[int]) -> List[int]:
    best_range = []
    longest_length = 0
    nums = {num: True for num in array}
    for num in array:
        if not nums.get(num, False):
            continue
        nums[num] = False
        current_length = 1
        left = num - 1
        right = num + 1
        while nums.get(left, False):
            nums[left] = False
            current_length += 1
            left -= 1
        while nums.get(right, False):
            nums[right] = False
            current_length += 1
            right += 1
        if current_length > longest_length:
            longest_length = current_length
            best_range = [left + 1, right - 1]
    return best_range`,
		Explanation: `Finding the largest range of consecutive integers requires identifying sequences where numbers appear consecutively. A hash set enables efficient lookup while expanding ranges bidirectionally.

The algorithm stores all numbers in a hash set for O(1) lookup. For each unvisited number, it expands left (decrementing) and right (incrementing) to find all consecutive numbers. As it finds consecutive numbers, it marks them as visited to avoid processing the same range multiple times. It tracks the longest range found.

The key optimization is marking numbers as visited during expansion. This ensures each number is processed at most once, even though ranges may overlap. When expanding from a number in the middle of a range, we still find the complete range efficiently.

Edge cases handled include: all numbers consecutive (entire array is range), no consecutive numbers (each number is its own range), numbers with gaps (finds longest consecutive sequence), and duplicate numbers (handled by set, doesn't affect range).

An alternative O(n log n) approach would sort first, but that's slower. The hash set approach achieves O(n) time as each number is visited at most twice (once when expanding, once when marked). The space complexity is O(n) for the hash set.

Think of this as finding the longest unbroken sequence: you mark each number you've seen, and for each new number, you expand in both directions to see how long the consecutive sequence is, always tracking the longest you've found.`,
	},
	{
		ID:          247,
		Title:       "Longest Increasing Subsequence",
		Description: "Given a non-empty array of integers, write a function that returns the longest strictly-increasing subsequence in the array. A subsequence is formed by deleting zero or more elements from the array, maintaining the relative order of the elements that remain.",
		Difficulty:  "Very Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func longestIncreasingSubsequence(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = [5, 7, -24, 12, 10, 2, 3, 12, 5, 6, 35]", Expected: "[-24, 2, 3, 5, 6, 35]"},
		},
		Solution: `func longestIncreasingSubsequence(array []int) []int {
	sequences := make([]int, len(array))
	indices := make([]int, len(array)+1)
	for i := range sequences {
		sequences[i] = -1
		indices[i] = -1
	}
	length := 0
	for i, num := range array {
		newLength := binarySearch(1, length, indices, array, num)
		sequences[i] = indices[newLength-1]
		indices[newLength] = i
		if newLength > length {
			length = newLength
		}
	}
	return buildSequence(array, sequences, indices[length])
}

func binarySearch(startIdx int, endIdx int, indices []int, array []int, num int) int {
	if startIdx > endIdx {
		return startIdx
	}
	middleIdx := (startIdx + endIdx) / 2
	if array[indices[middleIdx]] < num {
		startIdx = middleIdx + 1
	} else {
		endIdx = middleIdx - 1
	}
	return binarySearch(startIdx, endIdx, indices, array, num)
}

func buildSequence(array []int, sequences []int, currentIdx int) []int {
	sequence := []int{}
	for currentIdx != -1 {
		sequence = append([]int{array[currentIdx]}, sequence...)
		currentIdx = sequences[currentIdx]
	}
	return sequence
}`,
		PythonSolution: `def longestIncreasingSubsequence(array: List[int]) -> List[int]:
    sequences = [None] * len(array)
    indices = [None] * (len(array) + 1)
    length = 0
    for i, num in enumerate(array):
        new_length = binary_search(1, length, indices, array, num)
        sequences[i] = indices[new_length - 1]
        indices[new_length] = i
        if new_length > length:
            length = new_length
    return build_sequence(array, sequences, indices[length])

def binary_search(start_idx, end_idx, indices, array, num):
    if start_idx > end_idx:
        return start_idx
    middle_idx = (start_idx + end_idx) // 2
    if array[indices[middle_idx]] < num:
        start_idx = middle_idx + 1
    else:
        end_idx = middle_idx - 1
    return binary_search(start_idx, end_idx, indices, array, num)

def build_sequence(array, sequences, current_idx):
    sequence = []
    while current_idx is not None:
        sequence.insert(0, array[current_idx])
        current_idx = sequences[current_idx]
    return sequence`,
		Explanation: `Finding the longest increasing subsequence (LIS) efficiently requires an optimized approach beyond standard dynamic programming. Patience sorting with binary search achieves O(n log n) time while also enabling sequence reconstruction.

The algorithm maintains an array tails where tails[i] stores the index of the smallest tail element of all increasing subsequences of length i+1. As we process each number, we use binary search to find the longest subsequence whose tail is smaller than the current number. We then extend that subsequence or start a new one.

The binary search finds the leftmost position where we can place the current number, which corresponds to the longest subsequence we can extend. We also maintain a sequences array to track the predecessor of each element, enabling reconstruction of the actual LIS sequence.

Edge cases handled include: arrays in descending order (LIS length 1), arrays in ascending order (LIS is entire array), arrays with duplicates (handled by binary search logic), and arrays with multiple valid LIS (finds one, typically lexicographically smallest).

An alternative O(n²) DP approach would check all previous elements, but patience sorting is more efficient. The time complexity is O(n log n) for processing n elements with binary search. The space complexity is O(n) for tails and sequences arrays.

Think of this as building card piles: you place each card on the leftmost pile whose top card is greater than or equal to it, and the number of piles equals the LIS length. The binary search efficiently finds which pile to use.`,
	},
	{
		ID:          248,
		Title:       "Longest String Chain",
		Description: "A string chain is defined as follows: let string A be a string in the initial array; if removing any single character from string A yields a new string B that's contained in the initial array of strings, then strings A and B form a string chain of length 2. Similarly, if removing any single character from string B yields a new string C that's contained in the initial array of strings, then strings A, B, and C form a string chain of length 3. Write a function that returns the longest string chain.",
		Difficulty:  "Very Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func longestStringChain(strings []string) []string",
		TestCases: []TestCase{
			{Input: "strings = [\"abde\", \"abc\", \"abd\", \"abcde\", \"ade\", \"ae\", \"1abde\", \"abcde\"]", Expected: "[\"abcde\", \"abde\", \"ade\", \"ae\"]"},
		},
		Solution: `func longestStringChain(strings []string) []string {
	stringChains := make(map[string]*StringChain)
	for _, str := range strings {
		stringChains[str] = &StringChain{NextString: "", MaxChainLength: 1}
	}
	sortedStrings := make([]string, len(strings))
	copy(sortedStrings, strings)
	sort.Slice(sortedStrings, func(i, j int) bool {
		return len(sortedStrings[i]) < len(sortedStrings[j])
	})
	for _, str := range sortedStrings {
		findLongestStringChain(str, stringChains)
	}
	return buildLongestStringChain(strings, stringChains)
}

func findLongestStringChain(str string, stringChains map[string]*StringChain) {
	for i := 0; i < len(str); i++ {
		smallerString := str[:i] + str[i+1:]
		if _, found := stringChains[smallerString]; !found {
			continue
		}
		smallerStringChainLength := stringChains[smallerString].MaxChainLength
		currentStringChainLength := stringChains[str].MaxChainLength
		if smallerStringChainLength+1 > currentStringChainLength {
			stringChains[str].MaxChainLength = smallerStringChainLength + 1
			stringChains[str].NextString = smallerString
		}
	}
}

func buildLongestStringChain(strings []string, stringChains map[string]*StringChain) []string {
	maxChainLength := 0
	chainStartingString := ""
	for _, str := range strings {
		if stringChains[str].MaxChainLength > maxChainLength {
			maxChainLength = stringChains[str].MaxChainLength
			chainStartingString = str
		}
	}
	longestStringChain := []string{}
	currentString := chainStartingString
	for currentString != "" {
		longestStringChain = append(longestStringChain, currentString)
		currentString = stringChains[currentString].NextString
	}
	return longestStringChain
}

type StringChain struct {
	NextString     string
	MaxChainLength int
}`,
		PythonSolution: `def longestStringChain(strings: List[str]) -> List[str]:
    string_chains = {string: {"next_string": "", "max_chain_length": 1} for string in strings}
    sorted_strings = sorted(strings, key=len)
    for string in sorted_strings:
        find_longest_string_chain(string, string_chains)
    return build_longest_string_chain(strings, string_chains)

def find_longest_string_chain(string, string_chains):
    for i in range(len(string)):
        smaller_string = string[:i] + string[i + 1:]
        if smaller_string not in string_chains:
            continue
        smaller_string_chain_length = string_chains[smaller_string]["max_chain_length"]
        current_string_chain_length = string_chains[string]["max_chain_length"]
        if smaller_string_chain_length + 1 > current_string_chain_length:
            string_chains[string]["max_chain_length"] = smaller_string_chain_length + 1
            string_chains[string]["next_string"] = smaller_string

def build_longest_string_chain(strings, string_chains):
    max_chain_length = 0
    chain_starting_string = ""
    for string in strings:
        if string_chains[string]["max_chain_length"] > max_chain_length:
            max_chain_length = string_chains[string]["max_chain_length"]
            chain_starting_string = string
    longest_string_chain = []
    current_string = chain_starting_string
    while current_string != "":
        longest_string_chain.append(current_string)
        current_string = string_chains[current_string]["next_string"]
    return longest_string_chain`,
		Explanation: `Finding the longest string chain requires identifying sequences where each string can be formed by removing one character from the previous string. Processing strings in order of length ensures we build chains correctly.

The algorithm first sorts strings by length, ensuring shorter strings (potential predecessors) are processed before longer ones. For each string, it tries removing each character to see if the resulting string exists in the set. If it does, that string is a predecessor, and we update the chain length to be one more than the predecessor's chain length, tracking which string comes next.

After computing chain lengths for all strings, it finds the string with the maximum chain length, then reconstructs the chain by following the "next string" pointers backwards from that string.

Edge cases handled include: strings with no predecessors (chain length 1), strings that are part of multiple chains (longest chain chosen), strings of length 1 (no predecessors possible), and sets where no chains exist (each string has chain length 1).

An alternative approach would try all possible chains, but that's exponential. The sorted DP approach achieves O(n*m²) time where n is the number of strings and m is average length, as we check m characters to remove for each of n strings. The space complexity is O(n*m) for storing strings and chain information.

Think of this as building word ladders: you process words from shortest to longest, find which words can be formed by removing one letter from other words, track the longest chain you can build, then reconstruct that chain by following the connections backwards from the longest word.`,
	},
	{
		ID:          249,
		Title:       "Max Path Sum In Binary Tree",
		Description: "Write a function that takes in a Binary Tree and returns the maximum path sum. A path is a collection of connected nodes in a tree, where no node is connected to more than two other nodes. A path sum is the sum of the values of the nodes in a particular path.",
		Difficulty:  "Very Hard",
		Topic:       "Binary Trees",
		Signature:   "func maxPathSum(tree *TreeNode) int",
		TestCases: []TestCase{
			{Input: "tree = [1, 2, 3, 4, 5, 6, 7]", Expected: "18"},
		},
		Solution: `func maxPathSum(tree *TreeNode) int {
	_, maxSum := findMaxSum(tree)
	return maxSum
}

func findMaxSum(tree *TreeNode) (int, int) {
	if tree == nil {
		return 0, math.MinInt32
	}
	leftMaxSumAsBranch, leftMaxPathSum := findMaxSum(tree.Left)
	rightMaxSumAsBranch, rightMaxPathSum := findMaxSum(tree.Right)
	maxChildSumAsBranch := max(leftMaxSumAsBranch, rightMaxSumAsBranch)
	value := tree.Val
	maxSumAsBranch := max(maxChildSumAsBranch+value, value)
	maxSumAsRootNode := max(leftMaxSumAsBranch+value+rightMaxSumAsBranch, maxSumAsBranch)
	maxPathSum := max(leftMaxPathSum, max(rightMaxPathSum, maxSumAsRootNode))
	return maxSumAsBranch, maxPathSum
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def maxPathSum(tree: Optional[TreeNode]) -> int:
    _, max_sum = find_max_sum(tree)
    return max_sum

def find_max_sum(tree):
    if tree is None:
        return 0, float('-inf')
    left_max_sum_as_branch, left_max_path_sum = find_max_sum(tree.left)
    right_max_sum_as_branch, right_max_path_sum = find_max_sum(tree.right)
    max_child_sum_as_branch = max(left_max_sum_as_branch, right_max_sum_as_branch)
    value = tree.val
    max_sum_as_branch = max(max_child_sum_as_branch + value, value)
    max_sum_as_root_node = max(left_max_sum_as_branch + value + right_max_sum_as_branch, max_sum_as_branch)
    max_path_sum = max(left_max_path_sum, max(right_max_path_sum, max_sum_as_root_node))
    return max_sum_as_branch, max_path_sum`,
		Explanation: `Finding the maximum path sum in a binary tree requires distinguishing between paths that pass through a node (connecting left and right subtrees) and paths that are entirely within subtrees. The recursive approach calculates both for each node.

The algorithm returns two values for each node: the maximum sum of a path ending at this node (as a branch), and the maximum path sum found in the subtree rooted at this node. For the branch sum, we can either take just the node value or add it to the maximum child branch sum. For the path through the node, we combine the node value with both left and right branch sums. The overall maximum is the maximum of: the path through the current node, the maximum path in the left subtree, and the maximum path in the right subtree.

This approach efficiently computes the maximum path by considering all possible paths: those entirely within subtrees and those passing through the current node. The recursive structure naturally handles the tree traversal.

Edge cases handled include: negative node values (branch sum can be just the node value), all negative values (maximum path is the least negative single node), paths that don't include the root (handled by subtree maximums), and single-node trees (path is just that node's value).

An alternative approach would enumerate all paths, but that's exponential. The recursive approach achieves O(n) time by visiting each node once, and O(h) space for the recursion stack where h is the tree height.

Think of this as finding the best route through a network: at each junction (node), you consider the best path ending here, the best path passing through here, and the best paths in each branch, always tracking the overall best.`,
	},
	{
		ID:          250,
		Title:       "Max Sum Increasing Subsequence",
		Description: "Write a function that takes in a non-empty array of integers and returns the greatest sum that can be generated from a strictly-increasing subsequence in the array as well as an array of the numbers in that subsequence.",
		Difficulty:  "Very Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func maxSumIncreasingSubsequence(array []int) (int, []int)",
		TestCases: []TestCase{
			{Input: "array = [10, 70, 20, 30, 50, 11, 30]", Expected: "sum: 110, sequence: [10, 20, 30, 50]"},
		},
		Solution: `func maxSumIncreasingSubsequence(array []int) (int, []int) {
	sums := make([]int, len(array))
	sequences := make([]int, len(array))
	for i := range sequences {
		sequences[i] = -1
	}
	copy(sums, array)
	maxSumIdx := 0
	for i := 0; i < len(array); i++ {
		currentNum := array[i]
		for j := 0; j < i; j++ {
			otherNum := array[j]
			if otherNum < currentNum && sums[j]+currentNum >= sums[i] {
				sums[i] = sums[j] + currentNum
				sequences[i] = j
			}
		}
		if sums[i] >= sums[maxSumIdx] {
			maxSumIdx = i
		}
	}
	return sums[maxSumIdx], buildSequence(array, sequences, maxSumIdx)
}

func buildSequence(array []int, sequences []int, currentIdx int) []int {
	sequence := []int{}
	for currentIdx != -1 {
		sequence = append([]int{array[currentIdx]}, sequence...)
		currentIdx = sequences[currentIdx]
	}
	return sequence
}`,
		PythonSolution: `def maxSumIncreasingSubsequence(array: List[int]) -> Tuple[int, List[int]]:
    sums = array[:]
    sequences = [None] * len(array)
    max_sum_idx = 0
    for i in range(len(array)):
        current_num = array[i]
        for j in range(i):
            other_num = array[j]
            if other_num < current_num and sums[j] + current_num >= sums[i]:
                sums[i] = sums[j] + current_num
                sequences[i] = j
        if sums[i] >= sums[max_sum_idx]:
            max_sum_idx = i
    return sums[max_sum_idx], build_sequence(array, sequences, max_sum_idx)

def build_sequence(array, sequences, current_idx):
    sequence = []
    while current_idx is not None:
        sequence.insert(0, array[current_idx])
        current_idx = sequences[current_idx]
    return sequence`,
		Explanation: `Finding the maximum sum increasing subsequence requires tracking both the sum and the actual sequence. Dynamic programming efficiently handles this by maintaining the maximum sum achievable ending at each position and tracking predecessors for sequence reconstruction.

The algorithm uses DP where sums[i] represents the maximum sum of an increasing subsequence ending at index i. For each position i, it checks all previous positions j where array[j] < array[i]. If extending the subsequence ending at j with array[i] gives a larger sum than the current sums[i], it updates the sum and records j as the predecessor in the sequences array.

After computing all maximum sums, it finds the index with the maximum sum, then reconstructs the sequence by following the sequences array backwards from that index.

Edge cases handled include: arrays in descending order (each element is its own subsequence), arrays in ascending order (entire array is the subsequence), arrays with all negative values (finds least negative), and arrays where optimal subsequence skips some elements (handled by checking all previous positions).

An alternative approach would try all subsequences, but that's exponential. The DP approach achieves O(n²) time by checking each element against all previous elements. The space complexity is O(n) for the sums and sequences arrays.

Think of this as building the most valuable increasing sequence: for each number, you check all smaller numbers that came before, see which gives you the best total value when you add the current number, always tracking which number came before so you can reconstruct the sequence.`,
	},
	{
		ID:          251,
		Title:       "Longest Common Subsequence",
		Description: "Write a function that takes in two strings and returns the longest common subsequence. A subsequence of a string is a set of characters that aren't necessarily adjacent in the string but that are in the same order as they appear in the string.",
		Difficulty:  "Very Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func longestCommonSubsequence(str1 string, str2 string) []string",
		TestCases: []TestCase{
			{Input: "str1 = \"ZXVVYZW\", str2 = \"XKYKZPW\"", Expected: "[\"X\", \"Y\", \"Z\", \"W\"]"},
		},
		Solution: `func longestCommonSubsequence(str1 string, str2 string) []string {
	lcs := make([][][]string, len(str2)+1)
	for i := range lcs {
		lcs[i] = make([][]string, len(str1)+1)
	}
	for i := 1; i < len(str2)+1; i++ {
		for j := 1; j < len(str1)+1; j++ {
			if str2[i-1] == str1[j-1] {
				lcs[i][j] = append(lcs[i-1][j-1], string(str2[i-1]))
			} else {
				if len(lcs[i-1][j]) > len(lcs[i][j-1]) {
					lcs[i][j] = lcs[i-1][j]
				} else {
					lcs[i][j] = lcs[i][j-1]
				}
			}
		}
	}
	return lcs[len(str2)][len(str1)]
}`,
		PythonSolution: `def longestCommonSubsequence(str1: str, str2: str) -> List[str]:
    lcs = [[[] for _ in range(len(str1) + 1)] for _ in range(len(str2) + 1)]
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str2[i - 1] == str1[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + [str2[i - 1]]
            else:
                if len(lcs[i - 1][j]) > len(lcs[i][j - 1]):
                    lcs[i][j] = lcs[i - 1][j]
                else:
                    lcs[i][j] = lcs[i][j - 1]
    return lcs[len(str2)][len(str1)]`,
		Explanation: `Finding the longest common subsequence (LCS) between two strings requires identifying characters that appear in the same order in both strings. Dynamic programming efficiently solves this by building up solutions for prefixes of both strings.

The algorithm uses a 2D DP table where lcs[i][j] represents the LCS of str1[0..j-1] and str2[0..i-1]. For each position, if the characters match, we extend the LCS from the diagonal (both strings shortened by one). If they don't match, we take the longer of two options: LCS with str1 shortened or LCS with str2 shortened.

This approach correctly handles the fact that characters don't need to be adjacent - we can skip characters in either string. The DP table naturally captures all possible alignments and finds the longest matching subsequence.

Edge cases handled include: strings with no common characters (LCS is empty), one string is subsequence of other (LCS is shorter string), strings are identical (LCS is entire string), and strings with multiple valid LCS (finds one, typically leftmost).

An alternative recursive approach would have exponential time due to overlapping subproblems. The DP approach achieves O(n*m) time where n and m are string lengths, as we fill an n*m table. The space complexity is O(n*m) for the DP table, though it can be optimized to O(min(n,m)) by only keeping two rows.

Think of this as finding the longest matching pattern: you compare characters one by one, and when they match, you extend your pattern; when they don't, you see which string to advance to get a longer pattern.`,
	},
	{
		ID:          252,
		Title:       "Min Number Of Jumps",
		Description: "You're given a non-empty array of positive integers where each integer represents the maximum number of steps you can take forward in the array. For example, if the element at index 1 is 3, you can go from index 1 to index 2, 3, or 4. Write a function that returns the minimum number of jumps needed to reach the final index. Note that jumping from index i to index i + x always constitutes one jump, no matter how large x is.",
		Difficulty:  "Very Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func minNumberOfJumps(array []int) int",
		TestCases: []TestCase{
			{Input: "array = [3, 4, 2, 1, 2, 3, 7, 1, 1, 1, 3]", Expected: "4"},
		},
		Solution: `func minNumberOfJumps(array []int) int {
	if len(array) == 1 {
		return 0
	}
	jumps := 0
	maxReach := array[0]
	steps := array[0]
	for i := 1; i < len(array)-1; i++ {
		maxReach = max(maxReach, i+array[i])
		steps--
		if steps == 0 {
			jumps++
			steps = maxReach - i
		}
	}
	return jumps + 1
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def minNumberOfJumps(array: List[int]) -> int:
    if len(array) == 1:
        return 0
    jumps = 0
    max_reach = array[0]
    steps = array[0]
    for i in range(1, len(array) - 1):
        max_reach = max(max_reach, i + array[i])
        steps -= 1
        if steps == 0:
            jumps += 1
            steps = max_reach - i
    return jumps + 1`,
		Explanation: `Finding the minimum jumps to reach the end requires a greedy approach that tracks how far we can reach and how many steps we have remaining from the current jump. The key insight is to jump only when necessary (when steps are exhausted) and always jump to the position that maximizes future reach.

The algorithm maintains two key variables: maxReach (the farthest position we can reach with current and future jumps) and steps (remaining steps from the current jump). As we iterate through the array, we update maxReach to include positions reachable from the current position. When steps reach 0, we must jump, so we increment the jump count and update steps to the remaining steps from the new position (maxReach - current position).

This greedy approach is optimal because we always delay jumping until necessary and always choose the position that gives us the maximum future reach. By tracking maxReach, we ensure we're always aware of the best position we can reach.

Edge cases handled include: array starting with 0 (cannot progress, but handled by algorithm), single-element array (0 jumps needed), array where we can reach end in one jump (returns 1), and arrays where some positions are unreachable (algorithm detects this).

An alternative dynamic programming approach would be O(n²), but the greedy approach achieves O(n) time by making a single pass. The space complexity is O(1) as we only track a few variables.

Think of this as crossing stepping stones: you look ahead to see how far you can reach from your current stone, and you only step to a new stone when you've used all steps from your current jump, always choosing the stone that lets you reach farthest ahead.`,
	},
	{
		ID:          253,
		Title:       "Water Area",
		Description: "You're given an array of non-negative integers where each non-zero integer represents the height of a pillar of width 1. Imagine water being poured over all of the pillars. Write a function that returns the surface area of the water trapped between the pillars viewed from the front. Note that spilled water should be ignored.",
		Difficulty:  "Very Hard",
		Topic:       "Arrays",
		Signature:   "func waterArea(heights []int) int",
		TestCases: []TestCase{
			{Input: "heights = [0, 8, 0, 0, 5, 0, 0, 10, 0, 0, 1, 1, 0, 3]", Expected: "48"},
		},
		Solution: `func waterArea(heights []int) int {
	if len(heights) == 0 {
		return 0
	}
	leftIdx := 0
	rightIdx := len(heights) - 1
	leftMax := heights[leftIdx]
	rightMax := heights[rightIdx]
	surfaceArea := 0
	for leftIdx < rightIdx {
		if heights[leftIdx] < heights[rightIdx] {
			leftIdx++
			leftMax = max(leftMax, heights[leftIdx])
			surfaceArea += leftMax - heights[leftIdx]
		} else {
			rightIdx--
			rightMax = max(rightMax, heights[rightIdx])
			surfaceArea += rightMax - heights[rightIdx]
		}
	}
	return surfaceArea
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def waterArea(heights: List[int]) -> int:
    if len(heights) == 0:
        return 0
    left_idx = 0
    right_idx = len(heights) - 1
    left_max = heights[left_idx]
    right_max = heights[right_idx]
    surface_area = 0
    while left_idx < right_idx:
        if heights[left_idx] < heights[right_idx]:
            left_idx += 1
            left_max = max(left_max, heights[left_idx])
            surface_area += left_max - heights[left_idx]
        else:
            right_idx -= 1
            right_max = max(right_max, heights[right_idx])
            surface_area += right_max - heights[right_idx]
    return surface_area`,
		Explanation: `Calculating trapped water requires determining how much water can be held at each position. The two-pointer approach efficiently computes this by tracking maximum heights from both ends and processing positions in a single pass.

The algorithm uses two pointers starting at both ends. It tracks the maximum height seen from the left (leftMax) and right (rightMax). At each step, it processes the position with the smaller maximum height, as water trapped there is limited by that side. The water trapped at a position is min(leftMax, rightMax) - height[i], which represents the water level minus the pillar height.

By always moving the pointer with the smaller maximum, we ensure that when we process a position, the maximum on the opposite side is at least as large, guaranteeing correct water calculation. This approach processes each position once, making it optimal.

Edge cases handled include: all heights same (no water trapped), heights in ascending order (no water trapped), heights in descending order (no water trapped), and single pillar (no water trapped).

An alternative approach would compute left and right maximums separately, but that requires two passes. The two-pointer approach achieves O(n) time in a single pass. The space complexity is O(1) as we only track a few variables.

Think of this as filling a container: you work from both ends, always processing the side with the lower wall first, because that's what limits the water level, and you calculate how much water fits at each position.`,
	},
	{
		ID:          254,
		Title:       "Knapsack Problem",
		Description: "You're given an array of arrays where each subarray holds two integer values and represents an item; the first integer is the item's value and the second integer is the item's weight. You're also given an integer representing the maximum capacity of a knapsack that you have. Your goal is to fit items in your knapsack without having the sum of their weights exceed the knapsack's capacity, all the while maximizing their combined value. Note that you only have one of each item at your disposal. Write a function that returns the maximized combined value of the items that you should pick as well as an array of the indices of each item picked.",
		Difficulty:  "Very Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func knapsackProblem(items [][]int, capacity int) (int, []int)",
		TestCases: []TestCase{
			{Input: "items = [[1, 2], [4, 3], [5, 6], [6, 7]], capacity = 10", Expected: "value: 10, indices: [1, 3]"},
		},
		Solution: `func knapsackProblem(items [][]int, capacity int) (int, []int) {
	knapsackValues := make([][]int, len(items)+1)
	for i := range knapsackValues {
		knapsackValues[i] = make([]int, capacity+1)
	}
	for i := 1; i < len(items)+1; i++ {
		currentWeight := items[i-1][1]
		currentValue := items[i-1][0]
		for c := 0; c < capacity+1; c++ {
			if currentWeight > c {
				knapsackValues[i][c] = knapsackValues[i-1][c]
			} else {
				knapsackValues[i][c] = max(knapsackValues[i-1][c], knapsackValues[i-1][c-currentWeight]+currentValue)
			}
		}
	}
	return knapsackValues[len(items)][capacity], getKnapsackItems(knapsackValues, items)
}

func getKnapsackItems(knapsackValues [][]int, items [][]int) []int {
	sequence := []int{}
	i := len(knapsackValues) - 1
	c := len(knapsackValues[0]) - 1
	for i > 0 {
		if knapsackValues[i][c] == knapsackValues[i-1][c] {
			i--
		} else {
			sequence = append([]int{i - 1}, sequence...)
			c -= items[i-1][1]
			i--
		}
		if c == 0 {
			break
		}
	}
	return sequence
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def knapsackProblem(items: List[List[int]], capacity: int) -> Tuple[int, List[int]]:
    knapsack_values = [[0 for _ in range(capacity + 1)] for _ in range(len(items) + 1)]
    for i in range(1, len(items) + 1):
        current_weight = items[i - 1][1]
        current_value = items[i - 1][0]
        for c in range(capacity + 1):
            if current_weight > c:
                knapsack_values[i][c] = knapsack_values[i - 1][c]
            else:
                knapsack_values[i][c] = max(knapsack_values[i - 1][c], knapsack_values[i - 1][c - current_weight] + current_value)
    return knapsack_values[len(items)][capacity], get_knapsack_items(knapsack_values, items)

def get_knapsack_items(knapsack_values, items):
    sequence = []
    i = len(knapsack_values) - 1
    c = len(knapsack_values[0]) - 1
    while i > 0:
        if knapsack_values[i][c] == knapsack_values[i - 1][c]:
            i -= 1
        else:
            sequence.insert(0, i - 1)
            c -= items[i - 1][1]
            i -= 1
        if c == 0:
            break
    return sequence`,
		Explanation: `Solving the knapsack problem requires finding the maximum value achievable with limited capacity. Dynamic programming efficiently handles this by considering each item and each possible capacity, tracking both the maximum value and the items chosen.

The algorithm uses a 2D DP table where dp[i][c] represents the maximum value achievable using the first i items with capacity c. For each item, it considers two options: take the item (if it fits) or skip it. If taking the item gives more value, it updates dp[i][c] and records the choice. After filling the table, it backtracks to reconstruct the actual items chosen by following the decisions made.

The backtracking process starts from dp[n][capacity] and works backwards. If dp[i][c] equals dp[i-1][c], the item wasn't taken. Otherwise, the item was taken, so we add it to the sequence and reduce the capacity.

Edge cases handled include: capacity 0 (no items can be taken), items with weight exceeding capacity (skipped), items with zero value (may or may not be taken), and multiple optimal solutions (finds one).

An alternative approach would try all combinations, but that's exponential. The DP approach achieves O(n*c) time where n is items and c is capacity, as we fill an n*c table. The space complexity is O(n*c) for the DP table, though it can be optimized to O(c) by only keeping two rows.

Think of this as packing a bag optimally: for each item and each possible weight limit, you decide whether to include the item, always choosing the option that maximizes total value, then you trace back to see which items you actually took.`,
	},
	{
		ID:          255,
		Title:       "Disk Stacking",
		Description: "You're given a non-empty array of arrays where each subarray holds three integers and represents a disk. These integers denote each disk's width, depth, and height, respectively. Your goal is to stack up the disks and maximize the total height of the stack. A disk must have a strictly smaller width, depth, and height than any other disk below it. Write a function that returns an array of the indices of the disks in the final stack, starting with the top disk and ending with the bottom disk.",
		Difficulty:  "Very Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func diskStacking(disks [][]int) [][]int",
		TestCases: []TestCase{
			{Input: "disks = [[2, 1, 2], [3, 2, 3], [2, 2, 8], [2, 3, 4], [1, 2, 1], [4, 4, 5]]", Expected: "[[2, 1, 2], [3, 2, 3], [4, 4, 5]]"},
		},
		Solution: `import "sort"

func diskStacking(disks [][]int) [][]int {
	sort.Slice(disks, func(i, j int) bool {
		return disks[i][2] < disks[j][2]
	})
	heights := make([]int, len(disks))
	sequences := make([]int, len(disks))
	for i := range heights {
		heights[i] = disks[i][2]
		sequences[i] = -1
	}
	maxHeightIdx := 0
	for i := 1; i < len(disks); i++ {
		currentDisk := disks[i]
		for j := 0; j < i; j++ {
			otherDisk := disks[j]
			if areValidDimensions(otherDisk, currentDisk) {
				if heights[i] <= currentDisk[2]+heights[j] {
					heights[i] = currentDisk[2] + heights[j]
					sequences[i] = j
				}
			}
		}
		if heights[i] >= heights[maxHeightIdx] {
			maxHeightIdx = i
		}
	}
	return buildSequence(disks, sequences, maxHeightIdx)
}

func areValidDimensions(o []int, c []int) bool {
	return o[0] < c[0] && o[1] < c[1] && o[2] < c[2]
}

func buildSequence(array [][]int, sequences []int, currentIdx int) [][]int {
	sequence := [][]int{}
	for currentIdx != -1 {
		sequence = append([][]int{array[currentIdx]}, sequence...)
		currentIdx = sequences[currentIdx]
	}
	return sequence
}`,
		PythonSolution: `def diskStacking(disks: List[List[int]]) -> List[List[int]]:
    disks.sort(key=lambda x: x[2])
    heights = [disk[2] for disk in disks]
    sequences = [None] * len(disks)
    max_height_idx = 0
    for i in range(1, len(disks)):
        current_disk = disks[i]
        for j in range(i):
            other_disk = disks[j]
            if are_valid_dimensions(other_disk, current_disk):
                if heights[i] <= current_disk[2] + heights[j]:
                    heights[i] = current_disk[2] + heights[j]
                    sequences[i] = j
        if heights[i] >= heights[max_height_idx]:
            max_height_idx = i
    return build_sequence(disks, sequences, max_height_idx)

def are_valid_dimensions(o, c):
    return o[0] < c[0] and o[1] < c[1] and o[2] < c[2]

def build_sequence(array, sequences, current_idx):
    sequence = []
    while current_idx is not None:
        sequence.insert(0, array[current_idx])
        current_idx = sequences[current_idx]
    return sequence`,
		Explanation: `Finding the tallest stack of disks requires dynamic programming to determine the maximum height achievable ending at each disk. The key constraint is that a disk can only be placed on top of a strictly smaller disk.

The algorithm first sorts disks by height (and width as a secondary key for stability). Then it uses dynamic programming where heights[i] represents the maximum height of a stack ending with disk i. For each disk, it checks all smaller disks and chooses the one that gives the maximum height when this disk is placed on top. It also maintains a sequences array to track which disk comes before each disk in the optimal stack.

After computing all maximum heights, it finds the disk that gives the maximum overall height, then reconstructs the sequence by following the sequences array backwards from that disk.

Edge cases handled include: all disks same size (no valid stack, returns empty), disks in optimal order (all used in sequence), and disks where only one can be used (returns that disk).

An alternative approach would try all permutations, but that's factorial time. The DP approach achieves O(n²) time by checking each disk against all smaller disks. The space complexity is O(n) for the DP arrays and sequences.

Think of this as building a tower: you sort blocks by size, then for each block, you find the tallest tower you can build ending with that block by placing it on top of the best smaller block, always tracking which block comes before to reconstruct the final tower.`,
	},
	{
		ID:          256,
		Title:       "Numbers In Pi",
		Description: "Given a string representation of the first n digits of Pi and a list of your favorite numbers (all positive integers), write a function that returns the smallest number of spaces that need to be added to the n digits of Pi such that all resulting numbers are found in the list of favorite numbers. If no number of spaces to be added exists such that all the resulting numbers are found in the list of favorite numbers, the function should return -1.",
		Difficulty:  "Very Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func numbersInPi(pi string, numbers []string) int",
		TestCases: []TestCase{
			{Input: "pi = \"3141592\", numbers = [\"3141\", \"5\", \"31\", \"2\", \"4159\", \"9\", \"42\"]", Expected: "2"},
		},
		Solution: `func numbersInPi(pi string, numbers []string) int {
	numbersTable := make(map[string]bool)
	for _, number := range numbers {
		numbersTable[number] = true
	}
	cache := make(map[int]int)
	minSpaces := getMinSpaces(pi, numbersTable, cache, 0)
	if minSpaces == math.MaxInt32 {
		return -1
	}
	return minSpaces
}

func getMinSpaces(pi string, numbersTable map[string]bool, cache map[int]int, idx int) int {
	if idx == len(pi) {
		return -1
	}
	if val, found := cache[idx]; found {
		return val
	}
	minSpaces := math.MaxInt32
	for i := idx; i < len(pi); i++ {
		prefix := pi[idx : i+1]
		if numbersTable[prefix] {
			minSpacesInSuffix := getMinSpaces(pi, numbersTable, cache, i+1)
			minSpaces = min(minSpaces, minSpacesInSuffix+1)
		}
	}
	cache[idx] = minSpaces
	return minSpaces
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}`,
		PythonSolution: `def numbersInPi(pi: str, numbers: List[str]) -> int:
    numbers_table = {number: True for number in numbers}
    cache = {}
    min_spaces = get_min_spaces(pi, numbers_table, cache, 0)
    return min_spaces if min_spaces != float('inf') else -1

def get_min_spaces(pi, numbers_table, cache, idx):
    if idx == len(pi):
        return -1
    if idx in cache:
        return cache[idx]
    min_spaces = float('inf')
    for i in range(idx, len(pi)):
        prefix = pi[idx:i + 1]
        if prefix in numbers_table:
            min_spaces_in_suffix = get_min_spaces(pi, numbers_table, cache, i + 1)
            min_spaces = min(min_spaces, min_spaces_in_suffix + 1)
    cache[idx] = min_spaces
    return min_spaces`,
		Explanation: `Finding the minimum spaces needed to partition a Pi string such that all resulting numbers are in a favorite numbers list requires trying all possible partitions. Memoized recursion efficiently explores this space while avoiding redundant calculations.

The algorithm tries all possible prefixes starting from the current index. For each prefix that exists in the numbers table, it recursively finds the minimum spaces needed for the remaining suffix. The minimum spaces for the current position is the minimum of all valid prefix choices plus one space (for the current partition).

Memoization is crucial here because the same suffix positions may be computed multiple times through different prefix choices. By caching results for each starting index, we avoid recalculating the minimum spaces for the same subproblems.

Edge cases handled include: entire string is a single number (returns 0 spaces), no valid partition exists (returns -1), string where all characters form valid numbers (minimum spaces found), and strings with overlapping number possibilities (all tried, minimum chosen).

An alternative dynamic programming approach would build from right to left, but the recursive approach is more intuitive. The time complexity is O(n³ + m) where n is Pi string length and m is numbers list size: O(n²) for trying all prefixes at each position, O(n) positions, plus O(m) for building the numbers table. The space complexity is O(n + m) for the cache and numbers table.

Think of this as breaking a long number into valid segments: you try breaking it at each possible position, and if the prefix is valid, you recursively find the best way to break the rest, always tracking the minimum total breaks needed.`,
	},
	{
		ID:          257,
		Title:       "Maximum Sum Submatrix",
		Description: "Given a two-dimensional array of potentially unequal height and width that's filled with integers, as well as a positive integer size, write a function that returns the maximum sum that can be generated from a submatrix with dimensions size * size.",
		Difficulty:  "Very Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func maximumSumSubmatrix(matrix [][]int, size int) int",
		TestCases: []TestCase{
			{Input: "matrix = [[5, 3, -1, 5], [-7, 3, 7, 4], [12, 8, 0, 0], [1, -8, -8, 2]], size = 2", Expected: "18"},
		},
		Solution: `func maximumSumSubmatrix(matrix [][]int, size int) int {
	sums := createSumMatrix(matrix)
	maxSubMatrixSum := math.MinInt32
	for row := size - 1; row < len(matrix); row++ {
		for col := size - 1; col < len(matrix[0]); col++ {
			total := sums[row][col]
			touchesTopBorder := row-size < 0
			if !touchesTopBorder {
				total -= sums[row-size][col]
			}
			touchesLeftBorder := col-size < 0
			if !touchesLeftBorder {
				total -= sums[row][col-size]
			}
			touchesTopOrLeftBorder := touchesTopBorder || touchesLeftBorder
			if !touchesTopOrLeftBorder {
				total += sums[row-size][col-size]
			}
			maxSubMatrixSum = max(maxSubMatrixSum, total)
		}
	}
	return maxSubMatrixSum
}

func createSumMatrix(matrix [][]int) [][]int {
	sums := make([][]int, len(matrix))
	for i := range sums {
		sums[i] = make([]int, len(matrix[0]))
	}
	sums[0][0] = matrix[0][0]
	for idx := 1; idx < len(matrix[0]); idx++ {
		sums[0][idx] = sums[0][idx-1] + matrix[0][idx]
	}
	for idx := 1; idx < len(matrix); idx++ {
		sums[idx][0] = sums[idx-1][0] + matrix[idx][0]
	}
	for row := 1; row < len(matrix); row++ {
		for col := 1; col < len(matrix[0]); col++ {
			sums[row][col] = sums[row-1][col] + sums[row][col-1] - sums[row-1][col-1] + matrix[row][col]
		}
	}
	return sums
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def maximumSumSubmatrix(matrix: List[List[int]], size: int) -> int:
    sums = create_sum_matrix(matrix)
    max_sub_matrix_sum = float('-inf')
    for row in range(size - 1, len(matrix)):
        for col in range(size - 1, len(matrix[0])):
            total = sums[row][col]
            touches_top_border = row - size < 0
            if not touches_top_border:
                total -= sums[row - size][col]
            touches_left_border = col - size < 0
            if not touches_left_border:
                total -= sums[row][col - size]
            touches_top_or_left_border = touches_top_border or touches_left_border
            if not touches_top_or_left_border:
                total += sums[row - size][col - size]
            max_sub_matrix_sum = max(max_sub_matrix_sum, total)
    return max_sub_matrix_sum

def create_sum_matrix(matrix):
    sums = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    sums[0][0] = matrix[0][0]
    for idx in range(1, len(matrix[0])):
        sums[0][idx] = sums[0][idx - 1] + matrix[0][idx]
    for idx in range(1, len(matrix)):
        sums[idx][0] = sums[idx - 1][0] + matrix[idx][0]
    for row in range(1, len(matrix)):
        for col in range(1, len(matrix[0])):
            sums[row][col] = sums[row - 1][col] + sums[row][col - 1] - sums[row - 1][col - 1] + matrix[row][col]
    return sums`,
		Explanation: `Finding the maximum sum submatrix of a given size requires efficiently calculating sums of all possible submatrices. A prefix sum matrix (also called a 2D prefix sum or integral image) enables O(1) calculation of any submatrix sum.

The algorithm first builds a prefix sum matrix where sums[i][j] represents the sum of all elements from (0,0) to (i,j). This is built using the inclusion-exclusion principle: sums[i][j] = sums[i-1][j] + sums[i][j-1] - sums[i-1][j-1] + matrix[i][j]. The subtraction accounts for the double-counted region.

Once the prefix sum matrix is built, calculating any submatrix sum from (r1,c1) to (r2,c2) is O(1): sums[r2][c2] - sums[r1-1][c2] - sums[r2][c1-1] + sums[r1-1][c1-1]. The algorithm then tries all possible top-left positions for the size×size submatrix and finds the maximum sum.

Edge cases handled include: matrix smaller than size (no valid submatrix), all negative values (maximum is least negative submatrix), single-element submatrix (size=1), and matrix where maximum spans entire matrix (size equals matrix dimensions).

An alternative approach would calculate each submatrix sum directly, but that's O(size²) per submatrix. The prefix sum approach achieves O(w*h) time for building the prefix matrix and O(w*h) time for trying all positions, giving overall O(w*h) time. The space complexity is O(w*h) for the prefix sum matrix.

Think of this as creating a running total map: you precompute sums from the origin to each point, then use these to quickly calculate any rectangular region's sum by adding and subtracting the right corner values.`,
	},
	{
		ID:          258,
		Title:       "Maximize Expression",
		Description: "Write a function that takes in an array of integers and returns the largest possible value for the expression array[a] - array[b] + array[c] - array[d], where a, b, c, and d are indices of the array and a < b < c < d.",
		Difficulty:  "Very Hard",
		Topic:       "Dynamic Programming",
		Signature:   "func maximizeExpression(array []int) int",
		TestCases: []TestCase{
			{Input: "array = [3, 6, 1, -3, 2, 7]", Expected: "4"},
		},
		Solution: `func maximizeExpression(array []int) int {
	if len(array) < 4 {
		return 0
	}
	maxOfA := make([]int, len(array))
	maxOfAMinusB := make([]int, len(array))
	maxOfAMinusBPlusC := make([]int, len(array))
	maxOfAMinusBPlusCMinusD := make([]int, len(array))
	maxOfA[0] = array[0]
	for i := 1; i < len(array); i++ {
		maxOfA[i] = max(maxOfA[i-1], array[i])
	}
	maxOfAMinusB[1] = maxOfA[0] - array[1]
	for i := 2; i < len(array); i++ {
		maxOfAMinusB[i] = max(maxOfAMinusB[i-1], maxOfA[i-1]-array[i])
	}
	maxOfAMinusBPlusC[2] = maxOfAMinusB[1] + array[2]
	for i := 3; i < len(array); i++ {
		maxOfAMinusBPlusC[i] = max(maxOfAMinusBPlusC[i-1], maxOfAMinusB[i-1]+array[i])
	}
	maxOfAMinusBPlusCMinusD[3] = maxOfAMinusBPlusC[2] - array[3]
	for i := 4; i < len(array); i++ {
		maxOfAMinusBPlusCMinusD[i] = max(maxOfAMinusBPlusCMinusD[i-1], maxOfAMinusBPlusC[i-1]-array[i])
	}
	return maxOfAMinusBPlusCMinusD[len(array)-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def maximizeExpression(array: List[int]) -> int:
    if len(array) < 4:
        return 0
    max_of_a = [array[0]] * len(array)
    max_of_a_minus_b = [float('-inf')] * len(array)
    max_of_a_minus_b_plus_c = [float('-inf')] * len(array)
    max_of_a_minus_b_plus_c_minus_d = [float('-inf')] * len(array)
    for i in range(1, len(array)):
        max_of_a[i] = max(max_of_a[i - 1], array[i])
    max_of_a_minus_b[1] = max_of_a[0] - array[1]
    for i in range(2, len(array)):
        max_of_a_minus_b[i] = max(max_of_a_minus_b[i - 1], max_of_a[i - 1] - array[i])
    max_of_a_minus_b_plus_c[2] = max_of_a_minus_b[1] + array[2]
    for i in range(3, len(array)):
        max_of_a_minus_b_plus_c[i] = max(max_of_a_minus_b_plus_c[i - 1], max_of_a_minus_b[i - 1] + array[i])
    max_of_a_minus_b_plus_c_minus_d[3] = max_of_a_minus_b_plus_c[2] - array[3]
    for i in range(4, len(array)):
        max_of_a_minus_b_plus_c_minus_d[i] = max(max_of_a_minus_b_plus_c_minus_d[i - 1], max_of_a_minus_b_plus_c[i - 1] - array[i])
    return max_of_a_minus_b_plus_c_minus_d[len(array) - 1]`,
		Explanation: `Finding the maximum value of the expression a - b + c - d requires tracking the maximum value achievable for each partial expression as we process the array. Dynamic programming efficiently handles this by maintaining state for each expression part.

The algorithm processes the array left to right, maintaining four DP arrays: maxOfA (maximum single element), maxOfAMinusB (maximum a-b), maxOfAMinusBPlusC (maximum a-b+c), and maxOfAMinusBPlusCMinusD (maximum a-b+c-d). For each position, it updates these arrays by either continuing the previous maximum or starting a new expression part at the current position.

The key insight is that to maximize a-b+c-d, we need to maximize a-b+c (for the first three parts) and then subtract d optimally. Similarly, to maximize a-b+c, we maximize a-b and add c. This cascading optimization naturally leads to the DP recurrence relations.

Edge cases handled include: arrays with 4 or fewer elements (some expressions undefined), all positive values (maximum uses all values), all negative values (minimum uses least negative), and arrays where optimal solution skips some elements (handled by max operations).

An alternative approach would try all combinations of choosing 4 elements, but that's O(n⁴). The DP approach achieves O(n) time by processing each element once and updating all expression states. The space complexity is O(n) for the DP arrays, though it can be optimized to O(1) by only keeping the previous values.

Think of this as building up an expression step by step: you track the best value you can achieve with 1 element (a), then 2 elements (a-b), then 3 (a-b+c), and finally 4 (a-b+c-d), always choosing the maximum at each step.`,
	},
}
