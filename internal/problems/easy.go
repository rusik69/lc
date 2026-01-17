package problems

// EasyProblems contains all easy difficulty problems (1-20)
var EasyProblems = []Problem{
	{
		ID:          1,
		Title:       "Two Sum",
		Description: "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice.",
		Difficulty:  "Easy",
		Signature:   "func twoSum(nums []int, target int) []int",
		TestCases: []TestCase{
			{Input: "nums = []int{2, 7, 11, 15}, target = 9", Expected: "[0 1]"},
			{Input: "nums = []int{3, 2, 4}, target = 6", Expected: "[1 2]"},
			{Input: "nums = []int{3, 3}, target = 6", Expected: "[0 1]"},
			{Input: "nums = []int{1, 5, 3, 7}, target = 8", Expected: "[1 3]"},
			{Input: "nums = []int{-1, -2, -3, -4, -5}, target = -8", Expected: "[2 4]"},
			{Input: "nums = []int{0, 4, 3, 0}, target = 0", Expected: "[0 3]"},
		},
		Solution: `func twoSum(nums []int, target int) []int {
	m := make(map[int]int)
	for i, num := range nums {
		if idx, ok := m[target-num]; ok {
			return []int{idx, i}
		}
		m[num] = i
	}
	return nil
}`,
		Explanation: "Use a hash map to store each number and its index as we iterate. For each number, check if the complement (target - num) exists in the map. If found, return both indices. Time: O(n), Space: O(n).",
	},
	{
		ID:          2,
		Title:       "Reverse String",
		Description: "Write a function that reverses a string. The input string is given as an array of characters. You must do this by modifying the input array in-place.",
		Difficulty:  "Easy",
		Signature:   "func reverseString(s []byte)",
		TestCases: []TestCase{
			{Input: "s = []byte{'h','e','l','l','o'}", Expected: "olleh"},
			{Input: "s = []byte{'H','a','n','n','a','h'}", Expected: "hannaH"},
			{Input: "s = []byte{'a'}", Expected: "a"},
			{Input: "s = []byte{'a','b'}", Expected: "ba"},
			{Input: "s = []byte{'1','2','3'}", Expected: "321"},
			{Input: "s = []byte{}", Expected: ""},
		},
		Solution: `func reverseString(s []byte) {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
}`,
		Explanation: "Use two pointers starting from both ends. Swap characters and move pointers toward the center until they meet. Time: O(n), Space: O(1).",
	},
	{
		ID:          3,
		Title:       "FizzBuzz",
		Description: "Given an integer n, return a string array answer where: answer[i] == \"FizzBuzz\" if i is divisible by 3 and 5, answer[i] == \"Fizz\" if i is divisible by 3, answer[i] == \"Buzz\" if i is divisible by 5, answer[i] == i if none of the above conditions are true.",
		Difficulty:  "Easy",
		Signature:   "func fizzBuzz(n int) []string",
		TestCases: []TestCase{
			{Input: "n = 3", Expected: "[1 2 Fizz]"},
			{Input: "n = 5", Expected: "[1 2 Fizz 4 Buzz]"},
			{Input: "n = 15", Expected: "[1 2 Fizz 4 Buzz Fizz 7 8 Fizz Buzz 11 Fizz 13 14 FizzBuzz]"},
			{Input: "n = 1", Expected: "[1]"},
			{Input: "n = 30", Expected: "[1 2 Fizz 4 Buzz Fizz 7 8 Fizz Buzz 11 Fizz 13 14 FizzBuzz 16 17 Fizz 19 Buzz Fizz 22 23 Fizz Buzz 26 Fizz 28 29 FizzBuzz]"},
		},
		Solution: `func fizzBuzz(n int) []string {
	result := make([]string, n)
	for i := 1; i <= n; i++ {
		if i%15 == 0 {
			result[i-1] = "FizzBuzz"
		} else if i%3 == 0 {
			result[i-1] = "Fizz"
		} else if i%5 == 0 {
			result[i-1] = "Buzz"
		} else {
			result[i-1] = strconv.Itoa(i)
		}
	}
	return result
}`,
		Explanation: "Check divisibility in order: 15 (both 3 and 5), then 3, then 5. Convert numbers to strings using strconv.Itoa. Time: O(n), Space: O(n).",
	},
	{
		ID:          4,
		Title:       "Palindrome Check",
		Description: "Given a string s, return true if it is a palindrome, or false otherwise. A palindrome is a string that reads the same forward and backward.",
		Difficulty:  "Easy",
		Signature:   "func isPalindrome(s string) bool",
		TestCases: []TestCase{
			{Input: "s = \"racecar\"", Expected: "true"},
			{Input: "s = \"hello\"", Expected: "false"},
			{Input: "s = \"A man a plan a canal Panama\"", Expected: "true"},
			{Input: "s = \"a\"", Expected: "true"},
			{Input: "s = \"\"", Expected: "true"},
			{Input: "s = \"race a car\"", Expected: "false"},
			{Input: "s = \"Madam\"", Expected: "true"},
		},
		Solution: `func isPalindrome(s string) bool {
	cleaned := ""
	for _, c := range strings.ToLower(s) {
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') {
			cleaned += string(c)
		}
	}
	for i, j := 0, len(cleaned)-1; i < j; i, j = i+1, j-1 {
		if cleaned[i] != cleaned[j] {
			return false
		}
	}
	return true
}`,
		Explanation: "Normalize the string by converting to lowercase and removing spaces. Use two pointers to compare characters from both ends. Time: O(n), Space: O(1).",
	},
	{
		ID:          5,
		Title:       "Fibonacci",
		Description: "The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1. Given n, calculate F(n).",
		Difficulty:  "Easy",
		Signature:   "func fib(n int) int",
		TestCases: []TestCase{
			{Input: "n = 2", Expected: "1"},
			{Input: "n = 3", Expected: "2"},
			{Input: "n = 4", Expected: "3"},
			{Input: "n = 10", Expected: "55"},
			{Input: "n = 0", Expected: "0"},
			{Input: "n = 1", Expected: "1"},
			{Input: "n = 5", Expected: "5"},
			{Input: "n = 7", Expected: "13"},
		},
		Solution: `func fib(n int) int {
	if n < 2 {
		return n
	}
	a, b := 0, 1
	for i := 2; i <= n; i++ {
		a, b = b, a+b
	}
	return b
}`,
		Explanation: "Use iterative approach with two variables to track previous two Fibonacci numbers. Avoid recursion to save space. Time: O(n), Space: O(1).",
	},
	{
		ID:          6,
		Title:       "Contains Duplicate",
		Description: "Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.",
		Difficulty:  "Easy",
		Signature:   "func containsDuplicate(nums []int) bool",
		TestCases: []TestCase{
			{Input: "nums = []int{1,2,3,1}", Expected: "true"},
			{Input: "nums = []int{1,2,3,4}", Expected: "false"},
			{Input: "nums = []int{1,1,1,3,3,4,3,2,4,2}", Expected: "true"},
			{Input: "nums = []int{1}", Expected: "false"},
			{Input: "nums = []int{1,1}", Expected: "true"},
			{Input: "nums = []int{}", Expected: "false"},
		},
		Solution: `func containsDuplicate(nums []int) bool {
	seen := make(map[int]bool)
	for _, num := range nums {
		if seen[num] {
			return true
		}
		seen[num] = true
	}
	return false
}`,
		Explanation: "Use a hash set to track seen numbers. If we encounter a number already in the set, return true. Time: O(n), Space: O(n).",
	},
	{
		ID:          7,
		Title:       "Valid Anagram",
		Description: "Given two strings s and t, return true if t is an anagram of s, and false otherwise.",
		Difficulty:  "Easy",
		Signature:   "func isAnagram(s string, t string) bool",
		TestCases: []TestCase{
			{Input: "s = \"anagram\", t = \"nagaram\"", Expected: "true"},
			{Input: "s = \"rat\", t = \"car\"", Expected: "false"},
			{Input: "s = \"listen\", t = \"silent\"", Expected: "true"},
			{Input: "s = \"a\", t = \"a\"", Expected: "true"},
			{Input: "s = \"a\", t = \"ab\"", Expected: "false"},
			{Input: "s = \"ab\", t = \"a\"", Expected: "false"},
		},
		Solution: `func isAnagram(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}
	counts := make(map[rune]int)
	for _, c := range s {
		counts[c]++
	}
	for _, c := range t {
		counts[c]--
		if counts[c] < 0 {
			return false
		}
	}
	return true
}`,
		Explanation: "Count character frequencies in both strings using a hash map. Increment for s, decrement for t. Time: O(n), Space: O(1) since limited to 26 letters.",
	},
	{
		ID:          8,
		Title:       "Valid Parentheses",
		Description: "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
		Difficulty:  "Easy",
		Signature:   "func isValid(s string) bool",
		TestCases: []TestCase{
			{Input: "s = \"()\"", Expected: "true"},
			{Input: "s = \"()[]{}\"", Expected: "true"},
			{Input: "s = \"(]\"", Expected: "false"},
			{Input: "s = \"([)]\"", Expected: "false"},
			{Input: "s = \"{[]}\"", Expected: "true"},
			{Input: "s = \"\"", Expected: "true"},
			{Input: "s = \"(\"", Expected: "false"},
			{Input: "s = \")\"", Expected: "false"},
		},
		Solution: `func isValid(s string) bool {
	stack := []rune{}
	pairs := map[rune]rune{')': '(', '}': '{', ']': '['}
	for _, c := range s {
		if c == '(' || c == '{' || c == '[' {
			stack = append(stack, c)
		} else {
			if len(stack) == 0 || stack[len(stack)-1] != pairs[c] {
				return false
			}
			stack = stack[:len(stack)-1]
		}
	}
	return len(stack) == 0
}`,
		Explanation: "Use a stack to match opening and closing brackets. Push opening brackets, pop and verify for closing brackets. Time: O(n), Space: O(n).",
	},
	{
		ID:          9,
		Title:       "Merge Two Sorted Lists",
		Description: "Merge two sorted linked lists and return it as a sorted list. The list should be made by splicing together the nodes of the first two lists.",
		Difficulty:  "Easy",
		Signature:   "func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode",
		TestCases: []TestCase{
			{Input: "l1 = [1,2,4], l2 = [1,3,4]", Expected: "[1,1,2,3,4,4]"},
			{Input: "l1 = [], l2 = []", Expected: "[]"},
			{Input: "l1 = [], l2 = [0]", Expected: "[0]"},
			{Input: "l1 = [1], l2 = []", Expected: "[1]"},
			{Input: "l1 = [1,3,5], l2 = [2,4,6]", Expected: "[1,2,3,4,5,6]"},
		},
		Solution: `func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
	curr := dummy
	for l1 != nil && l2 != nil {
		if l1.Val < l2.Val {
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
}`,
		Explanation: "Use a dummy node and compare values from both lists one by one. Append remaining list when one is exhausted. Time: O(m+n), Space: O(1).",
	},
	{
		ID:          10,
		Title:       "Best Time to Buy and Sell Stock",
		Description: "You are given an array prices where prices[i] is the price of a given stock on the ith day. You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.",
		Difficulty:  "Easy",
		Signature:   "func maxProfit(prices []int) int",
		TestCases: []TestCase{
			{Input: "prices = []int{7,1,5,3,6,4}", Expected: "5"},
			{Input: "prices = []int{7,6,4,3,1}", Expected: "0"},
			{Input: "prices = []int{1,2}", Expected: "1"},
			{Input: "prices = []int{2,4,1}", Expected: "2"},
			{Input: "prices = []int{3,3,5,0,0,3,1,4}", Expected: "4"},
			{Input: "prices = []int{1}", Expected: "0"},
		},
		Solution: `func maxProfit(prices []int) int {
	minPrice := prices[0]
	maxProfit := 0
	for i := 1; i < len(prices); i++ {
		if prices[i] < minPrice {
			minPrice = prices[i]
		} else if prices[i]-minPrice > maxProfit {
			maxProfit = prices[i] - minPrice
		}
	}
	return maxProfit
}`,
		Explanation: "Track the minimum price seen so far and calculate profit for each day. Keep the maximum profit. Time: O(n), Space: O(1).",
	},
	{
		ID:          11,
		Title:       "Binary Search",
		Description: "Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.",
		Difficulty:  "Easy",
		Signature:   "func search(nums []int, target int) int",
		TestCases: []TestCase{
			{Input: "nums = []int{-1,0,3,5,9,12}, target = 9", Expected: "4"},
			{Input: "nums = []int{-1,0,3,5,9,12}, target = 2", Expected: "-1"},
			{Input: "nums = []int{5}, target = 5", Expected: "0"},
			{Input: "nums = []int{5}, target = -5", Expected: "-1"},
			{Input: "nums = []int{2,5}, target = 5", Expected: "1"},
			{Input: "nums = []int{-1,0,3,5,9,12}, target = 13", Expected: "-1"},
		},
		Solution: `func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}`,
		Explanation: "Classic binary search. Compare middle element with target and narrow search space by half each iteration. Time: O(log n), Space: O(1).",
	},
	{
		ID:          12,
		Title:       "First Bad Version",
		Description: "You are a product manager and currently leading a team to develop a new product. Since each version is developed based on the previous version, all the versions after a bad version are also bad. Find the first bad version.",
		Difficulty:  "Easy",
		Signature:   "func firstBadVersion(n int) int",
		TestCases: []TestCase{
			{Input: "n = 5, bad = 4", Expected: "4"},
			{Input: "n = 1, bad = 1", Expected: "1"},
			{Input: "n = 2, bad = 1", Expected: "1"},
			{Input: "n = 10, bad = 3", Expected: "3"},
			{Input: "n = 100, bad = 50", Expected: "50"},
		},
		Solution: `func firstBadVersion(n int) int {
	left, right := 1, n
	for left < right {
		mid := left + (right-left)/2
		if isBadVersion(mid) {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return left
}`,
		Explanation: "Binary search variant. When bad version found, search left half for earlier bad version. Time: O(log n), Space: O(1).",
	},
	{
		ID:          13,
		Title:       "Ransom Note",
		Description: "Given two strings ransomNote and magazine, return true if ransomNote can be constructed by using the letters from magazine and false otherwise.",
		Difficulty:  "Easy",
		Signature:   "func canConstruct(ransomNote string, magazine string) bool",
		TestCases: []TestCase{
			{Input: "ransomNote = \"a\", magazine = \"b\"", Expected: "false"},
			{Input: "ransomNote = \"aa\", magazine = \"aab\"", Expected: "true"},
			{Input: "ransomNote = \"aa\", magazine = \"ab\"", Expected: "false"},
			{Input: "ransomNote = \"\", magazine = \"abc\"", Expected: "true"},
			{Input: "ransomNote = \"a\", magazine = \"a\"", Expected: "true"},
			{Input: "ransomNote = \"abc\", magazine = \"cba\"", Expected: "true"},
		},
		Solution: `func canConstruct(ransomNote string, magazine string) bool {
	counts := make(map[rune]int)
	for _, c := range magazine {
		counts[c]++
	}
	for _, c := range ransomNote {
		if counts[c] == 0 {
			return false
		}
		counts[c]--
	}
	return true
}`,
		Explanation: "Count available letters in magazine, then check if ransom note can be formed by consuming letters. Time: O(m+n), Space: O(1).",
	},
	{
		ID:          14,
		Title:       "Climbing Stairs",
		Description: "You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?",
		Difficulty:  "Easy",
		Signature:   "func climbStairs(n int) int",
		TestCases: []TestCase{
			{Input: "n = 2", Expected: "2"},
			{Input: "n = 3", Expected: "3"},
			{Input: "n = 1", Expected: "1"},
			{Input: "n = 4", Expected: "5"},
			{Input: "n = 5", Expected: "8"},
			{Input: "n = 10", Expected: "89"},
		},
		Solution: `func climbStairs(n int) int {
	if n <= 2 {
		return n
	}
	a, b := 1, 2
	for i := 3; i <= n; i++ {
		a, b = b, a+b
	}
	return b
}`,
		Explanation: "This is Fibonacci sequence. Ways to reach step n = ways(n-1) + ways(n-2). Time: O(n), Space: O(1).",
	},
	{
		ID:          15,
		Title:       "Longest Common Prefix",
		Description: "Write a function to find the longest common prefix string amongst an array of strings. If there is no common prefix, return an empty string.",
		Difficulty:  "Easy",
		Signature:   "func longestCommonPrefix(strs []string) string",
		TestCases: []TestCase{
			{Input: "strs = []string{\"flower\",\"flow\",\"flight\"}", Expected: "\"fl\""},
			{Input: "strs = []string{\"dog\",\"racecar\",\"car\"}", Expected: "\"\""},
			{Input: "strs = []string{\"ab\",\"a\"}", Expected: "\"a\""},
			{Input: "strs = []string{\"a\"}", Expected: "\"a\""},
			{Input: "strs = []string{\"\",\"b\"}", Expected: "\"\""},
			{Input: "strs = []string{\"abab\",\"aba\",\"abc\"}", Expected: "\"ab\""},
		},
		Solution: `func longestCommonPrefix(strs []string) string {
	if len(strs) == 0 {
		return ""
	}
	prefix := strs[0]
	for i := 1; i < len(strs); i++ {
		for len(prefix) > 0 && !strings.HasPrefix(strs[i], prefix) {
			prefix = prefix[:len(prefix)-1]
		}
	}
	return prefix
}`,
		Explanation: "Start with first string as prefix, then shorten it until it matches all strings. Time: O(S) where S is sum of all characters, Space: O(1).",
	},
	{
		ID:          16,
		Title:       "Single Number",
		Description: "Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.",
		Difficulty:  "Easy",
		Signature:   "func singleNumber(nums []int) int",
		TestCases: []TestCase{
			{Input: "nums = []int{2,2,1}", Expected: "1"},
			{Input: "nums = []int{4,1,2,1,2}", Expected: "4"},
			{Input: "nums = []int{1}", Expected: "1"},
			{Input: "nums = []int{1,2,2}", Expected: "1"},
			{Input: "nums = []int{1,1,2,2,3}", Expected: "3"},
		},
		Solution: `func singleNumber(nums []int) int {
	result := 0
	for _, num := range nums {
		result ^= num
	}
	return result
}`,
		Explanation: "Use XOR operation. XOR of two same numbers is 0, and XOR with 0 returns the number itself. Time: O(n), Space: O(1).",
	},
	{
		ID:          17,
		Title:       "Majority Element",
		Description: "Given an array nums of size n, return the majority element. The majority element is the element that appears more than ⌊n / 2⌋ times.",
		Difficulty:  "Easy",
		Signature:   "func majorityElement(nums []int) int",
		TestCases: []TestCase{
			{Input: "nums = []int{3,2,3}", Expected: "3"},
			{Input: "nums = []int{2,2,1,1,1,2,2}", Expected: "2"},
			{Input: "nums = []int{1}", Expected: "1"},
			{Input: "nums = []int{1,1,1,2,2,2,2}", Expected: "2"},
			{Input: "nums = []int{6,5,5}", Expected: "5"},
		},
		Solution: `func majorityElement(nums []int) int {
	candidate, count := 0, 0
	for _, num := range nums {
		if count == 0 {
			candidate = num
		}
		if num == candidate {
			count++
		} else {
			count--
		}
	}
	return candidate
}`,
		Explanation: "Boyer-Moore Voting Algorithm. Maintain a candidate and count, cancel out different elements. Time: O(n), Space: O(1).",
	},
	{
		ID:          18,
		Title:       "Remove Duplicates from Sorted Array",
		Description: "Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once.",
		Difficulty:  "Easy",
		Signature:   "func removeDuplicates(nums []int) int",
		TestCases: []TestCase{
			{Input: "nums = []int{1,1,2}", Expected: "2"},
			{Input: "nums = []int{0,0,1,1,1,2,2,3,3,4}", Expected: "5"},
			{Input: "nums = []int{1,2,3}", Expected: "3"},
			{Input: "nums = []int{1,1,1}", Expected: "1"},
			{Input: "nums = []int{1,1,2,2,3,3}", Expected: "3"},
			{Input: "nums = []int{0}", Expected: "1"},
		},
		Solution: `func removeDuplicates(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	i := 0
	for j := 1; j < len(nums); j++ {
		if nums[j] != nums[i] {
			i++
			nums[i] = nums[j]
		}
	}
	return i + 1
}`,
		Explanation: "Use two pointers. Keep unique elements at the beginning of array. Time: O(n), Space: O(1).",
	},
	{
		ID:          19,
		Title:       "Move Zeroes",
		Description: "Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.",
		Difficulty:  "Easy",
		Signature:   "func moveZeroes(nums []int)",
		TestCases: []TestCase{
			{Input: "nums = []int{0,1,0,3,12}", Expected: "[1,3,12,0,0]"},
			{Input: "nums = []int{0}", Expected: "[0]"},
			{Input: "nums = []int{0,0,1}", Expected: "[1,0,0]"},
			{Input: "nums = []int{1,0,1,0,1}", Expected: "[1,1,1,0,0]"},
			{Input: "nums = []int{1,2,3}", Expected: "[1,2,3]"},
			{Input: "nums = []int{0,0,0}", Expected: "[0,0,0]"},
		},
		Solution: `func moveZeroes(nums []int) {
	left := 0
	for right := 0; right < len(nums); right++ {
		if nums[right] != 0 {
			nums[left], nums[right] = nums[right], nums[left]
			left++
		}
	}
}`,
		Explanation: "Two pointers approach. Left pointer tracks position for next non-zero element. Swap when non-zero found. Time: O(n), Space: O(1).",
	},
	{
		ID:          20,
		Title:       "Reverse Linked List",
		Description: "Given the head of a singly linked list, reverse the list, and return the reversed list.",
		Difficulty:  "Easy",
		Signature:   "func reverseList(head *ListNode) *ListNode",
		TestCases: []TestCase{
			{Input: "head = [1,2,3,4,5]", Expected: "[5,4,3,2,1]"},
			{Input: "head = []", Expected: "[]"},
			{Input: "head = [1]", Expected: "[1]"},
			{Input: "head = [1,2]", Expected: "[2,1]"},
			{Input: "head = [1,2,3]", Expected: "[3,2,1]"},
		},
		Solution: `func reverseList(head *ListNode) *ListNode {
	var prev *ListNode
	curr := head
	for curr != nil {
		next := curr.Next
		curr.Next = prev
		prev = curr
		curr = next
	}
	return prev
}`,
		Explanation: "Iterate through list and reverse pointers. Keep track of previous, current, and next nodes. Time: O(n), Space: O(1).",
	},
}
