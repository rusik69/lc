package problems

// MediumProblems contains all medium difficulty problems (21-40)
var MediumProblems = []Problem{
	{
		ID:          21,
		Title:       "Group Anagrams",
		Description: "Given an array of strings strs, group the anagrams together. You can return the answer in any order.",
		Difficulty:  "Medium",
		Signature:   "func groupAnagrams(strs []string) [][]string",
		TestCases: []TestCase{
			{Input: "strs = []string{\"eat\",\"tea\",\"tan\",\"ate\",\"nat\",\"bat\"}", Expected: "[[bat] [nat tan] [ate eat tea]]"},
			{Input: "strs = []string{\"\"}", Expected: "[[]]"},
			{Input: "strs = []string{\"a\"}", Expected: "[[a]]"},
			{Input: "strs = []string{\"abc\",\"bca\",\"cab\"}", Expected: "[[abc bca cab]]"},
			{Input: "strs = []string{\"listen\",\"silent\",\"enlist\"}", Expected: "[[listen silent enlist]]"},
		},
		Solution: `func groupAnagrams(strs []string) [][]string {
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
}`,
		PythonSolution: `def groupAnagrams(strs: List[str]) -> List[List[str]]:
    groups = {}
    for s in strs:
        key = tuple(sorted(s))
        if key not in groups:
            groups[key] = []
        groups[key].append(s)
    return list(groups.values())`,
		Explanation: "Use character count array as key for grouping anagrams. Each anagram will have same character frequencies. Time: O(n*k), Space: O(n*k).",
	},
	{
		ID:          22,
		Title:       "Top K Frequent Elements",
		Description: "Given an integer array nums and an integer k, return the k most frequent elements.",
		Difficulty:  "Medium",
		Signature:   "func topKFrequent(nums []int, k int) []int",
		TestCases: []TestCase{
			{Input: "nums = []int{1,1,1,2,2,3}, k = 2", Expected: "[1 2]"},
			{Input: "nums = []int{1}, k = 1", Expected: "[1]"},
			{Input: "nums = []int{1,2}, k = 2", Expected: "[1 2]"},
			{Input: "nums = []int{4,1,-1,2,-1,2,3}, k = 2", Expected: "[-1 2]"},
			{Input: "nums = []int{1,1,1,2,2,3,3,3}, k = 2", Expected: "[1 3]"},
		},
		Solution: `func topKFrequent(nums []int, k int) []int {
	freq := make(map[int]int)
	for _, num := range nums {
		freq[num]++
	}
	buckets := make([][]int, len(nums)+1)
	for num, count := range freq {
		buckets[count] = append(buckets[count], num)
	}
	result := make([]int, 0, k)
	for i := len(buckets) - 1; i >= 0 && len(result) < k; i-- {
		result = append(result, buckets[i]...)
	}
	return result[:k]
}`,
		PythonSolution: `def topKFrequent(nums: List[int], k: int) -> List[int]:
    freq = {}
    for num in nums:
        freq[num] = freq.get(num, 0) + 1
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, count in freq.items():
        buckets[count].append(num)
    result = []
    for i in range(len(buckets) - 1, -1, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            break
    return result[:k]`,
		Explanation: "Use bucket sort. Create buckets indexed by frequency, then collect from highest frequency buckets. Time: O(n), Space: O(n).",
	},
	{
		ID:          23,
		Title:       "Product of Array Except Self",
		Description: "Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].",
		Difficulty:  "Medium",
		Signature:   "func productExceptSelf(nums []int) []int",
		TestCases: []TestCase{
			{Input: "nums = []int{1,2,3,4}", Expected: "[24 12 8 6]"},
			{Input: "nums = []int{-1,1,0,-3,3}", Expected: "[0 0 9 0 0]"},
			{Input: "nums = []int{2,3,4,5}", Expected: "[60 40 30 24]"},
			{Input: "nums = []int{-1,-1,1,-1,-1}", Expected: "[-1 -1 -1 -1 -1]"},
			{Input: "nums = []int{1,0}", Expected: "[0 1]"},
		},
		Solution: `func productExceptSelf(nums []int) []int {
	n := len(nums)
	result := make([]int, n)
	result[0] = 1
	for i := 1; i < n; i++ {
		result[i] = result[i-1] * nums[i-1]
	}
	right := 1
	for i := n - 1; i >= 0; i-- {
		result[i] *= right
		right *= nums[i]
	}
	return result
}`,
		PythonSolution: `def productExceptSelf(nums: List[int]) -> List[int]:
    n = len(nums)
    result = [1] * n
    for i in range(1, n):
        result[i] = result[i-1] * nums[i-1]
    right = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right
        right *= nums[i]
    return result`,
		Explanation: "Two passes: first calculate prefix products, then multiply by suffix products. Avoids division. Time: O(n), Space: O(1).",
	},
	{
		ID:          24,
		Title:       "Longest Consecutive Sequence",
		Description: "Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.",
		Difficulty:  "Medium",
		Signature:   "func longestConsecutive(nums []int) int",
		TestCases: []TestCase{
			{Input: "nums = []int{100,4,200,1,3,2}", Expected: "4"},
			{Input: "nums = []int{0,3,7,2,5,8,4,6,0,1}", Expected: "9"},
			{Input: "nums = []int{}", Expected: "0"},
			{Input: "nums = []int{1}", Expected: "1"},
			{Input: "nums = []int{1,3,5,7,9}", Expected: "1"},
			{Input: "nums = []int{1,2,3,4,5}", Expected: "5"},
		},
		Solution: `func longestConsecutive(nums []int) int {
	numSet := make(map[int]bool)
	for _, num := range nums {
		numSet[num] = true
	}
	longest := 0
	for num := range numSet {
		if !numSet[num-1] {
			length := 1
			for numSet[num+length] {
				length++
			}
			if length > longest {
				longest = length
			}
		}
	}
	return longest
}`,
		PythonSolution: `def longestConsecutive(nums: List[int]) -> int:
    num_set = set(nums)
    longest = 0
    for num in num_set:
        if num - 1 not in num_set:
            length = 1
            while num + length in num_set:
                length += 1
            longest = max(longest, length)
    return longest`,
		Explanation: "Use hash set. Only start counting from sequence starts (no num-1 exists). Count consecutive numbers. Time: O(n), Space: O(n).",
	},
	{
		ID:          25,
		Title:       "Valid Sudoku",
		Description: "Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated.",
		Difficulty:  "Medium",
		Signature:   "func isValidSudoku(board [][]byte) bool",
		TestCases: []TestCase{
			{Input: "board = valid sudoku", Expected: "true"},
			{Input: "board = invalid sudoku", Expected: "false"},
		},
		Solution: `func isValidSudoku(board [][]byte) bool {
	rows := make([]map[byte]bool, 9)
	cols := make([]map[byte]bool, 9)
	boxes := make([]map[byte]bool, 9)
	for i := range rows {
		rows[i] = make(map[byte]bool)
		cols[i] = make(map[byte]bool)
		boxes[i] = make(map[byte]bool)
	}
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if board[i][j] == '.' {
				continue
			}
			num := board[i][j]
			boxIndex := (i/3)*3 + j/3
			if rows[i][num] || cols[j][num] || boxes[boxIndex][num] {
				return false
			}
			rows[i][num] = true
			cols[j][num] = true
			boxes[boxIndex][num] = true
		}
	}
	return true
}`,
		PythonSolution: `def isValidSudoku(board: List[List[str]]) -> bool:
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    for i in range(9):
        for j in range(9):
            if board[i][j] == '.':
                continue
            num = board[i][j]
            box_index = (i // 3) * 3 + j // 3
            if num in rows[i] or num in cols[j] or num in boxes[box_index]:
                return False
            rows[i].add(num)
            cols[j].add(num)
            boxes[box_index].add(num)
    return True`,
		Explanation: "Use hash sets for rows, columns, and 3x3 boxes to track seen digits. Check for duplicates in single pass. Time: O(1), Space: O(1).",
	},
	{
		ID:          26,
		Title:       "Encode and Decode Strings",
		Description: "Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.",
		Difficulty:  "Medium",
		Signature:   "func encode(strs []string) string",
		TestCases: []TestCase{
			{Input: "strs = []string{\"hello\",\"world\"}", Expected: "encoded string"},
		},
		Solution: `func encode(strs []string) string {
	var result strings.Builder
	for _, s := range strs {
		result.WriteString(strconv.Itoa(len(s)))
		result.WriteString("#")
		result.WriteString(s)
	}
	return result.String()
}

func decode(s string) []string {
	result := []string{}
	i := 0
	for i < len(s) {
		j := i
		for s[j] != '#' {
			j++
		}
		length, _ := strconv.Atoi(s[i:j])
		result = append(result, s[j+1:j+1+length])
		i = j + 1 + length
	}
	return result
}`,
		PythonSolution: `def encode(strs: List[str]) -> str:
    result = []
    for s in strs:
        result.append(str(len(s)) + '#' + s)
    return ''.join(result)

def decode(s: str) -> List[str]:
    result = []
    i = 0
    while i < len(s):
        j = i
        while s[j] != '#':
            j += 1
        length = int(s[i:j])
        result.append(s[j+1:j+1+length])
        i = j + 1 + length
    return result`,
		Explanation: "Prefix each string with its length and a delimiter. Decode by reading length, then extracting that many characters. Time: O(n), Space: O(1).",
	},
	{
		ID:          27,
		Title:       "Longest Substring Without Repeating Characters",
		Description: "Given a string s, find the length of the longest substring without repeating characters.",
		Difficulty:  "Medium",
		Signature:   "func lengthOfLongestSubstring(s string) int",
		TestCases: []TestCase{
			{Input: "s = \"abcabcbb\"", Expected: "3"},
			{Input: "s = \"bbbbb\"", Expected: "1"},
			{Input: "s = \"pwwkew\"", Expected: "3"},
			{Input: "s = \"\"", Expected: "0"},
			{Input: "s = \"a\"", Expected: "1"},
			{Input: "s = \"abcdef\"", Expected: "6"},
			{Input: "s = \"dvdf\"", Expected: "3"},
		},
		Solution: `func lengthOfLongestSubstring(s string) int {
	charIndex := make(map[byte]int)
	maxLen := 0
	start := 0
	for end := 0; end < len(s); end++ {
		if idx, found := charIndex[s[end]]; found && idx >= start {
			start = idx + 1
		}
		charIndex[s[end]] = end
		if end-start+1 > maxLen {
			maxLen = end - start + 1
		}
	}
	return maxLen
}`,
		PythonSolution: `def lengthOfLongestSubstring(s: str) -> int:
    char_index = {}
    max_len = 0
    start = 0
    for end in range(len(s)):
        if s[end] in char_index and char_index[s[end]] >= start:
            start = char_index[s[end]] + 1
        char_index[s[end]] = end
        max_len = max(max_len, end - start + 1)
    return max_len`,
		Explanation: "Sliding window with hash map. Track last seen index of each character. Move start when duplicate found. Time: O(n), Space: O(min(n,m)) where m is charset size.",
	},
	{
		ID:          28,
		Title:       "Longest Repeating Character Replacement",
		Description: "You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.",
		Difficulty:  "Medium",
		Signature:   "func characterReplacement(s string, k int) int",
		TestCases: []TestCase{
			{Input: "s = \"ABAB\", k = 2", Expected: "4"},
			{Input: "s = \"AABABBA\", k = 1", Expected: "4"},
			{Input: "s = \"A\", k = 0", Expected: "1"},
			{Input: "s = \"AAAA\", k = 2", Expected: "4"},
			{Input: "s = \"ABAA\", k = 0", Expected: "2"},
		},
		Solution: `func characterReplacement(s string, k int) int {
	count := make(map[byte]int)
	maxCount := 0
	maxLen := 0
	left := 0
	for right := 0; right < len(s); right++ {
		count[s[right]]++
		if count[s[right]] > maxCount {
			maxCount = count[s[right]]
		}
		for right-left+1-maxCount > k {
			count[s[left]]--
			left++
		}
		if right-left+1 > maxLen {
			maxLen = right - left + 1
		}
	}
	return maxLen
}`,
		PythonSolution: `def characterReplacement(s: str, k: int) -> int:
    count = {}
    max_count = 0
    max_len = 0
    left = 0
    for right in range(len(s)):
        count[s[right]] = count.get(s[right], 0) + 1
        max_count = max(max_count, count[s[right]])
        while right - left + 1 - max_count > k:
            count[s[left]] -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len`,
		Explanation: "Sliding window. Track most frequent character count. Window is valid if replacements needed ≤ k. Time: O(n), Space: O(1).",
	},
	{
		ID:          29,
		Title:       "Permutation in String",
		Description: "Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise.",
		Difficulty:  "Medium",
		Signature:   "func checkInclusion(s1 string, s2 string) bool",
		TestCases: []TestCase{
			{Input: "s1 = \"ab\", s2 = \"eidbaooo\"", Expected: "true"},
			{Input: "s1 = \"ab\", s2 = \"eidboaoo\"", Expected: "false"},
			{Input: "s1 = \"a\", s2 = \"ab\"", Expected: "true"},
			{Input: "s1 = \"hello\", s2 = \"ooolleoooleh\"", Expected: "false"},
			{Input: "s1 = \"adc\", s2 = \"dcda\"", Expected: "true"},
		},
		Solution: `func checkInclusion(s1 string, s2 string) bool {
	if len(s1) > len(s2) {
		return false
	}
	s1Count := [26]int{}
	s2Count := [26]int{}
	for i := range s1 {
		s1Count[s1[i]-'a']++
		s2Count[s2[i]-'a']++
	}
	matches := 0
	for i := 0; i < 26; i++ {
		if s1Count[i] == s2Count[i] {
			matches++
		}
	}
	for i := len(s1); i < len(s2); i++ {
		if matches == 26 {
			return true
		}
		index := s2[i] - 'a'
		s2Count[index]++
		if s1Count[index] == s2Count[index] {
			matches++
		} else if s1Count[index]+1 == s2Count[index] {
			matches--
		}
		index = s2[i-len(s1)] - 'a'
		s2Count[index]--
		if s1Count[index] == s2Count[index] {
			matches++
		} else if s1Count[index]-1 == s2Count[index] {
			matches--
		}
	}
	return matches == 26
}`,
		PythonSolution: `def checkInclusion(s1: str, s2: str) -> bool:
    if len(s1) > len(s2):
        return False
    s1_count = [0] * 26
    s2_count = [0] * 26
    for i in range(len(s1)):
        s1_count[ord(s1[i]) - ord('a')] += 1
        s2_count[ord(s2[i]) - ord('a')] += 1
    matches = sum(1 for i in range(26) if s1_count[i] == s2_count[i])
    for i in range(len(s1), len(s2)):
        if matches == 26:
            return True
        index = ord(s2[i]) - ord('a')
        s2_count[index] += 1
        if s1_count[index] == s2_count[index]:
            matches += 1
        elif s1_count[index] + 1 == s2_count[index]:
            matches -= 1
        index = ord(s2[i - len(s1)]) - ord('a')
        s2_count[index] -= 1
        if s1_count[index] == s2_count[index]:
            matches += 1
        elif s1_count[index] - 1 == s2_count[index]:
            matches -= 1
    return matches == 26`,
		Explanation: "Sliding window with character frequency matching. Maintain count of matching character frequencies. Time: O(n), Space: O(1).",
	},
	{
		ID:          30,
		Title:       "Minimum Window Substring",
		Description: "Given two strings s and t, return the minimum window substring of s such that every character in t (including duplicates) is included in the window.",
		Difficulty:  "Medium",
		Signature:   "func minWindow(s string, t string) string",
		TestCases: []TestCase{
			{Input: "s = \"ADOBECODEBANC\", t = \"ABC\"", Expected: "\"BANC\""},
			{Input: "s = \"a\", t = \"a\"", Expected: "\"a\""},
			{Input: "s = \"a\", t = \"aa\"", Expected: "\"\""},
			{Input: "s = \"ab\", t = \"b\"", Expected: "\"b\""},
			{Input: "s = \"bba\", t = \"ab\"", Expected: "\"ba\""},
		},
		Solution: `func minWindow(s string, t string) string {
	if len(s) < len(t) {
		return ""
	}
	tCount := make(map[byte]int)
	for i := range t {
		tCount[t[i]]++
	}
	required := len(tCount)
	formed := 0
	windowCounts := make(map[byte]int)
	left, right := 0, 0
	minLen := len(s) + 1
	minLeft := 0
	for right < len(s) {
		c := s[right]
		windowCounts[c]++
		if tCount[c] > 0 && windowCounts[c] == tCount[c] {
			formed++
		}
		for left <= right && formed == required {
			if right-left+1 < minLen {
				minLen = right - left + 1
				minLeft = left
			}
			c = s[left]
			windowCounts[c]--
			if tCount[c] > 0 && windowCounts[c] < tCount[c] {
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
		PythonSolution: `def minWindow(s: str, t: str) -> str:
    if len(s) < len(t):
        return ""
    t_count = {}
    for c in t:
        t_count[c] = t_count.get(c, 0) + 1
    required = len(t_count)
    formed = 0
    window_counts = {}
    left, right = 0, 0
    min_len = len(s) + 1
    min_left = 0
    while right < len(s):
        c = s[right]
        window_counts[c] = window_counts.get(c, 0) + 1
        if c in t_count and window_counts[c] == t_count[c]:
            formed += 1
        while left <= right and formed == required:
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left
            c = s[left]
            window_counts[c] -= 1
            if c in t_count and window_counts[c] < t_count[c]:
                formed -= 1
            left += 1
        right += 1
    if min_len == len(s) + 1:
        return ""
    return s[min_left:min_left + min_len]`,
		Explanation: "Sliding window. Expand right to include all characters, shrink left while valid. Track minimum window. Time: O(m+n), Space: O(m+n).",
	},
	{
		ID:          31,
		Title:       "Valid Palindrome II",
		Description: "Given a string s, return true if the s can be palindrome after deleting at most one character from it.",
		Difficulty:  "Medium",
		Signature:   "func validPalindrome(s string) bool",
		TestCases: []TestCase{
			{Input: "s = \"aba\"", Expected: "true"},
			{Input: "s = \"abca\"", Expected: "true"},
			{Input: "s = \"abc\"", Expected: "false"},
			{Input: "s = \"a\"", Expected: "true"},
			{Input: "s = \"ab\"", Expected: "true"},
			{Input: "s = \"eeccccbebaeeabebccceea\"", Expected: "false"},
		},
		Solution: `func validPalindrome(s string) bool {
	left, right := 0, len(s)-1
	for left < right {
		if s[left] != s[right] {
			return isPalindromeRange(s, left+1, right) || isPalindromeRange(s, left, right-1)
		}
		left++
		right--
	}
	return true
}

func isPalindromeRange(s string, i, j int) bool {
	for i < j {
		if s[i] != s[j] {
			return false
		}
		i++
		j--
	}
	return true
}`,
		PythonSolution: `def validPalindrome(s: str) -> bool:
    def is_palindrome_range(i: int, j: int) -> bool:
        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True
    
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return is_palindrome_range(left + 1, right) or is_palindrome_range(left, right - 1)
        left += 1
        right -= 1
    return True`,
		Explanation: "Two pointers. When mismatch found, try skipping left or right character. Time: O(n), Space: O(1).",
	},
	{
		ID:          32,
		Title:       "3Sum",
		Description: "Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.",
		Difficulty:  "Medium",
		Signature:   "func threeSum(nums []int) [][]int",
		TestCases: []TestCase{
			{Input: "nums = []int{-1,0,1,2,-1,-4}", Expected: "[[-1,-1,2],[-1,0,1]]"},
			{Input: "nums = []int{}", Expected: "[]"},
		},
		Solution: `func threeSum(nums []int) [][]int {
	sort.Ints(nums)
	result := [][]int{}
	for i := 0; i < len(nums)-2; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		left, right := i+1, len(nums)-1
		for left < right {
			sum := nums[i] + nums[left] + nums[right]
			if sum == 0 {
				result = append(result, []int{nums[i], nums[left], nums[right]})
				for left < right && nums[left] == nums[left+1] {
					left++
				}
				for left < right && nums[right] == nums[right-1] {
					right--
				}
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
}`,
		PythonSolution: `def threeSum(nums: List[int]) -> List[List[int]]:
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return result`,
		Explanation: "Sort array, fix one number, use two pointers for remaining two. Skip duplicates to avoid duplicate triplets. Time: O(n²), Space: O(1).",
	},
	{
		ID:          33,
		Title:       "Container With Most Water",
		Description: "You are given an integer array height of length n. There are n vertical lines. Find two lines that together with the x-axis form a container that contains the most water.",
		Difficulty:  "Medium",
		Signature:   "func maxArea(height []int) int",
		TestCases: []TestCase{
			{Input: "height = []int{1,8,6,2,5,4,8,3,7}", Expected: "49"},
			{Input: "height = []int{1,1}", Expected: "1"},
			{Input: "height = []int{1,2,1}", Expected: "2"},
			{Input: "height = []int{2,3,4,5,18,17,6}", Expected: "17"},
			{Input: "height = []int{1,3,2,5,25,24,5}", Expected: "24"},
		},
		Solution: `func maxArea(height []int) int {
	left, right := 0, len(height)-1
	maxArea := 0
	for left < right {
		width := right - left
		h := height[left]
		if height[right] < h {
			h = height[right]
		}
		area := width * h
		if area > maxArea {
			maxArea = area
		}
		if height[left] < height[right] {
			left++
		} else {
			right--
		}
	}
	return maxArea
}`,
		PythonSolution: `def maxArea(height: List[int]) -> int:
    left, right = 0, len(height) - 1
    max_area = 0
    while left < right:
        width = right - left
        h = min(height[left], height[right])
        area = width * h
        max_area = max(max_area, area)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_area`,
		Explanation: "Two pointers from both ends. Move pointer with smaller height inward (only way to potentially increase area). Time: O(n), Space: O(1).",
	},
	{
		ID:          34,
		Title:       "Trapping Rain Water",
		Description: "Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.",
		Difficulty:  "Medium",
		Signature:   "func trap(height []int) int",
		TestCases: []TestCase{
			{Input: "height = []int{0,1,0,2,1,0,1,3,2,1,2,1}", Expected: "6"},
			{Input: "height = []int{4,2,0,3,2,5}", Expected: "9"},
			{Input: "height = []int{3,0,2,0,4}", Expected: "7"},
			{Input: "height = []int{0,1,0}", Expected: "0"},
			{Input: "height = []int{2,0,2}", Expected: "2"},
			{Input: "height = []int{5,4,1,2}", Expected: "1"},
		},
		Solution: `func trap(height []int) int {
	if len(height) == 0 {
		return 0
	}
	left, right := 0, len(height)-1
	leftMax, rightMax := height[left], height[right]
	water := 0
	for left < right {
		if leftMax < rightMax {
			left++
			if height[left] > leftMax {
				leftMax = height[left]
			} else {
				water += leftMax - height[left]
			}
		} else {
			right--
			if height[right] > rightMax {
				rightMax = height[right]
			} else {
				water += rightMax - height[right]
			}
		}
	}
	return water
}`,
		PythonSolution: `def trap(height: List[int]) -> int:
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    water = 0
    while left < right:
        if left_max < right_max:
            left += 1
            if height[left] > left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
        else:
            right -= 1
            if height[right] > right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
    return water`,
		Explanation: "Two pointers. Track max height from both sides. Water at position is min(leftMax, rightMax) - height. Time: O(n), Space: O(1).",
	},
	{
		ID:          35,
		Title:       "Minimum Size Subarray Sum",
		Description: "Given an array of positive integers nums and a positive integer target, return the minimal length of a contiguous subarray whose sum is greater than or equal to target.",
		Difficulty:  "Medium",
		Signature:   "func minSubArrayLen(target int, nums []int) int",
		TestCases: []TestCase{
			{Input: "target = 7, nums = []int{2,3,1,2,4,3}", Expected: "2"},
			{Input: "target = 4, nums = []int{1,4,4}", Expected: "1"},
			{Input: "target = 11, nums = []int{1,1,1,1,1,1,1,1}", Expected: "0"},
			{Input: "target = 15, nums = []int{5,1,3,5,10,7,4,9,2,8}", Expected: "2"},
			{Input: "target = 6, nums = []int{10,2,3}", Expected: "1"},
		},
		Solution: `func minSubArrayLen(target int, nums []int) int {
	minLen := len(nums) + 1
	sum := 0
	left := 0
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
	if minLen == len(nums)+1 {
		return 0
	}
	return minLen
}`,
		PythonSolution: `def minSubArrayLen(target: int, nums: List[int]) -> int:
    min_len = len(nums) + 1
    sum_val = 0
    left = 0
    for right in range(len(nums)):
        sum_val += nums[right]
        while sum_val >= target:
            if right - left + 1 < min_len:
                min_len = right - left + 1
            sum_val -= nums[left]
            left += 1
    return 0 if min_len == len(nums) + 1 else min_len`,
		Explanation: "Sliding window. Expand right to increase sum, shrink left when sum ≥ target. Track minimum length. Time: O(n), Space: O(1).",
	},
	{
		ID:          36,
		Title:       "Maximum Subarray",
		Description: "Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.",
		Difficulty:  "Medium",
		Signature:   "func maxSubArray(nums []int) int",
		TestCases: []TestCase{
			{Input: "nums = []int{-2,1,-3,4,-1,2,1,-5,4}", Expected: "6"},
			{Input: "nums = []int{1}", Expected: "1"},
			{Input: "nums = []int{5,4,-1,7,8}", Expected: "23"},
			{Input: "nums = []int{-1}", Expected: "-1"},
			{Input: "nums = []int{-2,-1}", Expected: "-1"},
			{Input: "nums = []int{1,2,3,4}", Expected: "10"},
		},
		Solution: `func maxSubArray(nums []int) int {
	maxSum := nums[0]
	currentSum := nums[0]
	for i := 1; i < len(nums); i++ {
		if currentSum < 0 {
			currentSum = nums[i]
		} else {
			currentSum += nums[i]
		}
		if currentSum > maxSum {
			maxSum = currentSum
		}
	}
	return maxSum
}`,
		PythonSolution: `def maxSubArray(nums: List[int]) -> int:
    max_sum = nums[0]
    current_sum = nums[0]
    for i in range(1, len(nums)):
        if current_sum < 0:
            current_sum = nums[i]
        else:
            current_sum += nums[i]
        max_sum = max(max_sum, current_sum)
    return max_sum`,
		Explanation: "Kadane's algorithm. Track current subarray sum, reset when negative (start new subarray). Keep maximum. Time: O(n), Space: O(1).",
	},
	{
		ID:          37,
		Title:       "Jump Game",
		Description: "You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.",
		Difficulty:  "Medium",
		Signature:   "func canJump(nums []int) bool",
		TestCases: []TestCase{
			{Input: "nums = []int{2,3,1,1,4}", Expected: "true"},
			{Input: "nums = []int{3,2,1,0,4}", Expected: "false"},
			{Input: "nums = []int{0}", Expected: "true"},
			{Input: "nums = []int{2,0}", Expected: "true"},
			{Input: "nums = []int{1,0,1,0}", Expected: "false"},
			{Input: "nums = []int{5,9,3,2,1,0,2,3,3,1,0,0}", Expected: "true"},
		},
		Solution: `func canJump(nums []int) bool {
	maxReach := 0
	for i := 0; i < len(nums); i++ {
		if i > maxReach {
			return false
		}
		if i+nums[i] > maxReach {
			maxReach = i + nums[i]
		}
		if maxReach >= len(nums)-1 {
			return true
		}
	}
	return true
}`,
		PythonSolution: `def canJump(nums: List[int]) -> bool:
    max_reach = 0
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
        if max_reach >= len(nums) - 1:
            return True
    return True`,
		Explanation: "Greedy approach. Track maximum reachable index. If current index exceeds max reach, can't proceed. Time: O(n), Space: O(1).",
	},
	{
		ID:          38,
		Title:       "Jump Game II",
		Description: "Given an array of non-negative integers nums, you are initially positioned at the first index of the array. Your goal is to reach the last index in the minimum number of jumps.",
		Difficulty:  "Medium",
		Signature:   "func jump(nums []int) int",
		TestCases: []TestCase{
			{Input: "nums = []int{2,3,1,1,4}", Expected: "2"},
			{Input: "nums = []int{2,3,0,1,4}", Expected: "2"},
			{Input: "nums = []int{2,1}", Expected: "1"},
			{Input: "nums = []int{1,2,3}", Expected: "2"},
			{Input: "nums = []int{1,1,1,1}", Expected: "3"},
			{Input: "nums = []int{7,0,9,6,9,6,1,7,9,0,1,2,9,0,3}", Expected: "2"},
		},
		Solution: `func jump(nums []int) int {
	jumps := 0
	currentEnd := 0
	farthest := 0
	for i := 0; i < len(nums)-1; i++ {
		if i+nums[i] > farthest {
			farthest = i + nums[i]
		}
		if i == currentEnd {
			jumps++
			currentEnd = farthest
		}
	}
	return jumps
}`,
		PythonSolution: `def jump(nums: List[int]) -> int:
    jumps = 0
    current_end = 0
    farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    return jumps`,
		Explanation: "Greedy BFS approach. Track farthest reachable position in current jump. Increment jumps when reaching end of current range. Time: O(n), Space: O(1).",
	},
	{
		ID:          39,
		Title:       "Gas Station",
		Description: "There are n gas stations along a circular route. Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once, otherwise return -1.",
		Difficulty:  "Medium",
		Signature:   "func canCompleteCircuit(gas []int, cost []int) int",
		TestCases: []TestCase{
			{Input: "gas = []int{1,2,3,4,5}, cost = []int{3,4,5,1,2}", Expected: "3"},
			{Input: "gas = []int{2,3,4}, cost = []int{3,4,3}", Expected: "-1"},
			{Input: "gas = []int{5,1,2,3,4}, cost = []int{4,4,1,5,1}", Expected: "4"},
			{Input: "gas = []int{2,3,1}, cost = []int{3,1,2}", Expected: "1"},
			{Input: "gas = []int{5,8,2,8}, cost = []int{6,5,6,6}", Expected: "3"},
		},
		Solution: `func canCompleteCircuit(gas []int, cost []int) int {
	totalGas, totalCost := 0, 0
	tank, start := 0, 0
	for i := 0; i < len(gas); i++ {
		totalGas += gas[i]
		totalCost += cost[i]
		tank += gas[i] - cost[i]
		if tank < 0 {
			start = i + 1
			tank = 0
		}
	}
	if totalGas < totalCost {
		return -1
	}
	return start
}`,
		PythonSolution: `def canCompleteCircuit(gas: List[int], cost: List[int]) -> int:
    total_gas = total_cost = 0
    tank = start = 0
    for i in range(len(gas)):
        total_gas += gas[i]
        total_cost += cost[i]
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    if total_gas < total_cost:
        return -1
    return start`,
		Explanation: "Greedy approach. If total gas < total cost, impossible. Otherwise, find first position where we can maintain positive tank. Time: O(n), Space: O(1).",
	},
	{
		ID:          40,
		Title:       "Hand of Straights",
		Description: "Given an integer array hand where hand[i] is the value of the ith card and an integer groupSize, return true if she can rearrange the cards into groups of groupSize consecutive cards.",
		Difficulty:  "Medium",
		Signature:   "func isNStraightHand(hand []int, groupSize int) bool",
		TestCases: []TestCase{
			{Input: "hand = []int{1,2,3,6,2,3,4,7,8}, groupSize = 3", Expected: "true"},
			{Input: "hand = []int{1,2,3,4,5}, groupSize = 4", Expected: "false"},
			{Input: "hand = []int{1,2,3,4,5,6}, groupSize = 2", Expected: "true"},
			{Input: "hand = []int{1}, groupSize = 1", Expected: "true"},
			{Input: "hand = []int{1,1,2,2,3,3}, groupSize = 3", Expected: "true"},
			{Input: "hand = []int{1,2,3,4,5,6}, groupSize = 4", Expected: "false"},
		},
		Solution: `func isNStraightHand(hand []int, groupSize int) bool {
	if len(hand)%groupSize != 0 {
		return false
	}
	count := make(map[int]int)
	for _, card := range hand {
		count[card]++
	}
	sort.Ints(hand)
	for _, card := range hand {
		if count[card] == 0 {
			continue
		}
		for i := 0; i < groupSize; i++ {
			if count[card+i] == 0 {
				return false
			}
			count[card+i]--
		}
	}
	return true
}`,
		PythonSolution: `def isNStraightHand(hand: List[int], groupSize: int) -> bool:
    if len(hand) % groupSize != 0:
        return False
    count = {}
    for card in hand:
        count[card] = count.get(card, 0) + 1
    hand.sort()
    for card in hand:
        if count[card] == 0:
            continue
        for i in range(groupSize):
            if count.get(card + i, 0) == 0:
                return False
            count[card + i] -= 1
    return True`,
		Explanation: "Sort cards and use greedy approach. For each card, try to form a group starting from it. Use hash map to track counts. Time: O(n log n), Space: O(n).",
	},
}
