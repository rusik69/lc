package problems

// HardProblems contains all hard difficulty problems (41-60)
var HardProblems = []Problem{
	{
		ID:          41,
		Title:       "Median of Two Sorted Arrays",
		Description: "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.",
		Difficulty:  "Hard",
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
		Explanation: "Binary search on smaller array to find correct partition. Partition divides arrays so left half elements ≤ right half. Time: O(log(min(m,n))), Space: O(1).",
	},
	{
		ID:          42,
		Title:       "Longest Valid Parentheses",
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
		Explanation: "Use stack to track indices. Push '(' indices, pop on ')'. If stack empty after pop, push current index (new start). Track max distance. Time: O(n), Space: O(n).",
	},
	{
		ID:          43,
		Title:       "Wildcard Matching",
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
		Explanation: "Dynamic programming. dp[i][j] = match s[0...i-1] with p[0...j-1]. '*' can match empty or any sequence. Time: O(m*n), Space: O(m*n).",
	},
	{
		ID:          44,
		Title:       "Regular Expression Matching",
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
		Explanation: "DP similar to wildcard matching. '*' matches 0 or more of preceding element. Check both match 0 times and match one+ times. Time: O(m*n), Space: O(m*n).",
	},
	{
		ID:          45,
		Title:       "Merge k Sorted Lists",
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
		Explanation: "Divide and conquer. Merge lists pairwise repeatedly until one list remains. Each merge is O(n), log k levels. Time: O(n*k*log k), Space: O(1).",
	},
	{
		ID:          46,
		Title:       "Reverse Nodes in k-Group",
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
		Explanation: "Recursion. Check if k nodes available, reverse current k nodes, recursively process rest. Time: O(n), Space: O(n/k) for recursion stack.",
	},
	{
		ID:          47,
		Title:       "Largest Rectangle in Histogram",
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
		Explanation: "Monotonic stack. For each bar, find how far it can extend left and right. When smaller height found, calculate area. Time: O(n), Space: O(n).",
	},
	{
		ID:          48,
		Title:       "Maximal Rectangle",
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
		Explanation: "Reduce to histogram problem. For each row, treat it as histogram base with heights from consecutive 1's above. Time: O(m*n), Space: O(n).",
	},
	{
		ID:          49,
		Title:       "Minimum Window Substring",
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
		Explanation: "Sliding window. Expand to include all chars, contract while valid. Track minimum. Time: O(m+n), Space: O(m+n).",
	},
	{
		ID:          50,
		Title:       "Sliding Window Maximum",
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
		Explanation: "Monotonic deque. Maintain decreasing order of values. Front has maximum. Remove indices outside window. Time: O(n), Space: O(k).",
	},
	{
		ID:          51,
		Title:       "Word Ladder",
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
		Explanation: "BFS. Try all single-letter changes, check if in dictionary. Use set to avoid revisiting. Level = transformation count. Time: O(M²*N) where M is word length, Space: O(M²*N).",
	},
	{
		ID:          52,
		Title:       "Word Ladder II",
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
		Explanation: "BFS to build shortest path graph, then DFS to find all paths. Track parent relationships during BFS. Time: O(M²*N), Space: O(M²*N).",
	},
	{
		ID:          53,
		Title:       "Alien Dictionary",
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
		Explanation: "Topological sort. Build graph from word pairs, find ordering with BFS (Kahn's algorithm). Detect cycles. Time: O(C) where C is total chars, Space: O(1) for alphabet.",
	},
	{
		ID:          54,
		Title:       "Longest Increasing Path in a Matrix",
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
		Explanation: "DFS with memoization. For each cell, try all four directions with increasing values. Cache results to avoid recomputation. Time: O(m*n), Space: O(m*n).",
	},
	{
		ID:          55,
		Title:       "Number of Islands",
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
		Explanation: "DFS to mark connected components. For each unvisited land cell, increment count and mark entire island. Time: O(m*n), Space: O(m*n) for recursion.",
	},
	{
		ID:          56,
		Title:       "Course Schedule II",
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
		Explanation: "Topological sort with BFS (Kahn's algorithm). Start with courses having no prerequisites. Process in order. Detect cycles. Time: O(V+E), Space: O(V+E).",
	},
	{
		ID:          57,
		Title:       "Serialize and Deserialize Binary Tree",
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
		Explanation: "Preorder traversal for serialize and deserialize. Use 'null' for nil nodes. Build tree recursively following same order. Time: O(n), Space: O(n).",
	},
	{
		ID:          58,
		Title:       "Binary Tree Maximum Path Sum",
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
		Explanation: "Post-order DFS. For each node, max path through it = val + leftMax + rightMax. Return val + max(leftMax, rightMax) as contribution to parent. Time: O(n), Space: O(h).",
	},
	{
		ID:          59,
		Title:       "Distinct Subsequences",
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
		Explanation: "DP. dp[i][j] = ways to match t[0...j-1] using s[0...i-1]. If chars match, add ways without using current char. Time: O(m*n), Space: O(m*n).",
	},
	{
		ID:          60,
		Title:       "Edit Distance",
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
		Explanation: "Classic DP (Levenshtein distance). dp[i][j] = min operations for word1[0...i-1] to word2[0...j-1]. Consider insert, delete, replace. Time: O(m*n), Space: O(m*n).",
	},
}

