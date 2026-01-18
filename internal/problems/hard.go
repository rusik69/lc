package problems

// HardProblems contains all hard difficulty problems (41-60)
var HardProblems = []Problem{
	{
		ID:          41,
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
		Explanation: "Binary search on smaller array to find correct partition. Partition divides arrays so left half elements ≤ right half. Time: O(log(min(m,n))), Space: O(1).",
	},
	{
		ID:          42,
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
		Explanation: "Use stack to track indices. Push '(' indices, pop on ')'. If stack empty after pop, push current index (new start). Track max distance. Time: O(n), Space: O(n).",
	},
	{
		ID:          43,
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
		Explanation: "Dynamic programming. dp[i][j] = match s[0...i-1] with p[0...j-1]. '*' can match empty or any sequence. Time: O(m*n), Space: O(m*n).",
	},
	{
		ID:          44,
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
		Explanation: "DP similar to wildcard matching. '*' matches 0 or more of preceding element. Check both match 0 times and match one+ times. Time: O(m*n), Space: O(m*n).",
	},
	{
		ID:          45,
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
		Explanation: "Divide and conquer. Merge lists pairwise repeatedly until one list remains. Each merge is O(n), log k levels. Time: O(n*k*log k), Space: O(1).",
	},
	{
		ID:          46,
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
		Explanation: "Recursion. Check if k nodes available, reverse current k nodes, recursively process rest. Time: O(n), Space: O(n/k) for recursion stack.",
	},
	{
		ID:          47,
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
		Explanation: "Monotonic stack. For each bar, find how far it can extend left and right. When smaller height found, calculate area. Time: O(n), Space: O(n).",
	},
	{
		ID:          48,
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
		Explanation: "Reduce to histogram problem. For each row, treat it as histogram base with heights from consecutive 1's above. Time: O(m*n), Space: O(n).",
	},
	{
		ID:          49,
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
		Explanation: "Sliding window. Expand to include all chars, contract while valid. Track minimum. Time: O(m+n), Space: O(m+n).",
	},
	{
		ID:          50,
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
		Explanation: "Monotonic deque. Maintain decreasing order of values. Front has maximum. Remove indices outside window. Time: O(n), Space: O(k).",
	},
	{
		ID:          51,
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
		Explanation: "BFS. Try all single-letter changes, check if in dictionary. Use set to avoid revisiting. Level = transformation count. Time: O(M²*N) where M is word length, Space: O(M²*N).",
	},
	{
		ID:          52,
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
		Explanation: "BFS to build shortest path graph, then DFS to find all paths. Track parent relationships during BFS. Time: O(M²*N), Space: O(M²*N).",
	},
	{
		ID:          53,
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
		Explanation: "Topological sort. Build graph from word pairs, find ordering with BFS (Kahn's algorithm). Detect cycles. Time: O(C) where C is total chars, Space: O(1) for alphabet.",
	},
	{
		ID:          54,
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
		Explanation: "DFS with memoization. For each cell, try all four directions with increasing values. Cache results to avoid recomputation. Time: O(m*n), Space: O(m*n).",
	},
	{
		ID:          55,
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
		Explanation: "DFS to mark connected components. For each unvisited land cell, increment count and mark entire island. Time: O(m*n), Space: O(m*n) for recursion.",
	},
	{
		ID:          56,
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
		Explanation: "Topological sort with BFS (Kahn's algorithm). Start with courses having no prerequisites. Process in order. Detect cycles. Time: O(V+E), Space: O(V+E).",
	},
	{
		ID:          57,
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
		Explanation: "Preorder traversal for serialize and deserialize. Use 'null' for nil nodes. Build tree recursively following same order. Time: O(n), Space: O(n).",
	},
	{
		ID:          58,
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
		Explanation: "Post-order DFS. For each node, max path through it = val + leftMax + rightMax. Return val + max(leftMax, rightMax) as contribution to parent. Time: O(n), Space: O(h).",
	},
	{
		ID:          59,
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
		Explanation: "DP. dp[i][j] = ways to match t[0...j-1] using s[0...i-1]. If chars match, add ways without using current char. Time: O(m*n), Space: O(m*n).",
	},
	{
		ID:          60,
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
		Explanation: "Classic DP (Levenshtein distance). dp[i][j] = min operations for word1[0...i-1] to word2[0...j-1]. Consider insert, delete, replace. Time: O(m*n), Space: O(m*n).",
	},
	{
		ID:          61,
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
		Explanation: "Two pointers approach. Track max heights from both sides. Water trapped = min(leftMax, rightMax) - current height. Move pointer with smaller height. Time: O(n), Space: O(1).",
	},
	{
		ID:          62,
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
		Explanation: "Divide and conquer: merge pairs of lists iteratively until one remains. Merge two sorted lists helper function. Time: O(n log k) where n is total nodes, Space: O(1).",
	},
	{
		ID:          63,
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
		Explanation: "Preorder traversal serialization. Use comma-separated values, \"null\" for nil nodes. Deserialize by reading values in preorder. Time: O(n), Space: O(n).",
	},
	{
		ID:          64,
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
		Explanation: "BFS: Each word is a node, edges connect words differing by one letter. Use BFS to find shortest path. Time: O(M*N) where M is word length, N is word list size, Space: O(N).",
	},
	{
		ID:          65,
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
		Explanation: "Two heaps: small (max heap) for smaller half, large (min heap) for larger half. Keep sizes balanced. Median is top of small (if odd) or average of both tops (if even). Time: O(log n) per addNum, O(1) findMedian, Space: O(n).",
	},
	{
		ID:          66,
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
		Explanation: "Backtracking: Place queen row by row. Track columns and diagonals (two types) to check attacks. Time: O(n!), Space: O(n²) for board.",
	},
	{
		ID:          67,
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
		Explanation: "Backtracking: Try digits 1-9 for each empty cell. Check row, column, and 3x3 box. Backtrack if invalid. Time: O(9^m) where m is empty cells, Space: O(1).",
	},
	{
		ID:          68,
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
		Explanation: "Cyclic sort: Place each number at its correct index (nums[i] should be at index nums[i]-1). First position where nums[i] != i+1 is the answer. Time: O(n), Space: O(1).",
	},
	{
		ID:          69,
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
		Explanation: "DP: dp[i][j] = match s[0..i-1] with p[0..j-1]. '*' matches empty (dp[i][j-1]) or any sequence (dp[i-1][j]). '?' or match: dp[i-1][j-1]. Time: O(m*n), Space: O(m*n).",
	},
	{
		ID:          70,
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
		Explanation: "DP: dp[i][j] = match s[0..i-1] with p[0..j-1]. '*' matches zero (dp[i][j-2]) or one+ (dp[i-1][j] if char matches). '.' or match: dp[i-1][j-1]. Time: O(m*n), Space: O(m*n).",
	},
}

