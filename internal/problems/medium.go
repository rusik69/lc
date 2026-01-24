package problems

// MediumProblems contains all medium difficulty problems (58-160)
var MediumProblems = []Problem{
	{
		ID:          58,
		Title:       "Group Anagrams",
		Description: "Given an array of strings strs, group the anagrams together. You can return the answer in any order.",
		Difficulty:  "Medium",
		Topic:       "Hash Tables",
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
        key = [0] * 26
        for c in s:
            key[ord(c) - ord('a')] += 1
        key_tuple = tuple(key)
        if key_tuple not in groups:
            groups[key_tuple] = []
        groups[key_tuple].append(s)
    return list(groups.values())`,
		Explanation: `Grouping anagrams requires identifying strings that have the same character frequencies. The key insight is that anagrams will produce the same character count array when we count occurrences of each letter.

The algorithm uses a hash map where the key is a character frequency array (26 integers representing counts of 'a' through 'z'). For each string, we create this frequency array by counting each character. Strings with identical frequency arrays are anagrams and are grouped together in the hash map.

This approach is efficient because comparing frequency arrays is O(1) when used as hash map keys (in Go, arrays can be used as map keys). After processing all strings, we simply collect all groups from the hash map values.

Edge cases handled include: empty strings (grouped together as they have all-zero frequency arrays), single-character strings, strings with different lengths (cannot be anagrams, so grouped separately), and strings with special characters (problem specifies lowercase letters only).

An alternative approach would be to sort each string and use the sorted string as a key, but that requires O(k log k) time per string for sorting, making overall time O(n*k*log k). The frequency array approach is O(n*k) since counting characters is linear in string length.

Think of this as creating a "fingerprint" for each string based on its letter composition. Strings with the same fingerprint (anagrams) are grouped together.`,
	},
	{
		ID:          59,
		Title:       "Top K Frequent Elements",
		Topic:       "Hash Tables",
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
		Explanation: `Finding the k most frequent elements can be efficiently solved using bucket sort, which takes advantage of the fact that frequencies are bounded by the array length. This avoids the O(n log n) time complexity of sorting.

The algorithm first counts the frequency of each element using a hash map. Then, it creates an array of buckets where each bucket at index i contains all elements that appear exactly i times. Since frequencies range from 1 to n (array length), we create n+1 buckets. Finally, we iterate through buckets from highest index to lowest, collecting elements until we have k elements.

This bucket sort approach is optimal because it achieves O(n) time complexity. The frequency counting takes O(n) time, bucket creation takes O(n) time (one pass through frequencies), and collecting k elements takes at most O(n) time. The space complexity is O(n) for the frequency map and buckets.

Edge cases handled include: k equal to number of unique elements (return all elements), all elements having the same frequency (return any k elements), elements with frequency 1 (appear in bucket 1), and arrays where k is 1 (return the most frequent element).

An alternative approach would be to use a max heap, but that requires O(n log k) time. Another approach would be to sort frequencies, but that's O(n log n). The bucket sort approach achieves the optimal O(n) time complexity.

Think of this as organizing items by how often they appear: you put each item in a box labeled with its frequency count, then take items from the highest-numbered boxes first until you have k items.`,
	},
	{
		ID:          60,
		Title:       "Product of Array Except Self",
		Topic:       "Arrays",
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
		Explanation: `This problem requires computing the product of all elements except the current element, without using division. The key insight is that we can use prefix and suffix products to achieve this in O(n) time with O(1) extra space.

The algorithm makes two passes through the array. In the first pass (left to right), it calculates prefix products: result[i] contains the product of all elements to the left of index i. In the second pass (right to left), it multiplies each prefix product by the corresponding suffix product, which is calculated on-the-fly using a running variable.

This approach avoids division, which is important because the problem might contain zeros (division by zero) or the constraint might prohibit division. By separating the calculation into prefix and suffix products, we can compute the result efficiently without additional space for storing suffix products.

Edge cases handled include: arrays with zeros (product becomes 0 for all positions except the zero position), arrays with negative numbers (signs are preserved correctly), arrays with a single zero (all other positions are 0, zero position gets product of others), and arrays with all positive numbers (standard case).

An alternative approach would be to calculate the total product and divide by each element, but that fails with zeros and may not be allowed. Another approach would use O(n) extra space to store suffix products, but the current approach achieves O(1) space by calculating suffixes on-the-fly.

Think of this as calculating "what would the product be if I removed this element": you multiply everything to the left with everything to the right, which gives you the product of all elements except the current one.`,
	},
	{
		ID:          61,
		Title:       "Longest Consecutive Sequence",
		Topic:       "Hash Tables",
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
		Explanation: `Finding the longest consecutive sequence in an unsorted array requires efficient lookup of numbers. The key insight is that we should only start counting sequences from numbers that are the start of a sequence (i.e., numbers where num-1 doesn't exist in the array).

The algorithm first stores all numbers in a hash set for O(1) lookup. Then, for each number in the set, it checks if that number is the start of a sequence by verifying that num-1 is not in the set. If it is a start, it counts how many consecutive numbers follow by checking num+1, num+2, etc., until the sequence breaks. It keeps track of the longest sequence found.

This approach is optimal because each number is visited at most twice: once when we check if it's a sequence start, and once when we're counting a sequence that started earlier. This gives us O(n) time complexity despite the nested loop structure.

Edge cases handled include: empty arrays (returns 0), arrays with no consecutive sequences (returns 1), arrays with duplicate numbers (hash set handles this), arrays with negative numbers (sequences work correctly across zero), and arrays with a single consecutive sequence spanning the entire range.

An alternative O(n log n) approach would be to sort the array first, but the hash set approach is optimal. Another approach would check all possible sequences, but that would be O(n²) or worse. The current approach achieves O(n) time with O(n) space.

Think of this as finding the longest chain: you only start counting from the beginning of a chain (where there's no previous link), then count forward until the chain breaks.`,
	},
	{
		ID:          62,
		Title:       "Valid Sudoku",
		Topic:       "Hash Tables",
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
		Explanation: `Validating a Sudoku board requires checking three constraints: each row must contain digits 1-9 without repetition, each column must contain digits 1-9 without repetition, and each 3x3 box must contain digits 1-9 without repetition. The key insight is that we can check all constraints in a single pass using hash sets.

The algorithm maintains three arrays of hash sets: one for rows, one for columns, and one for 3x3 boxes. As we iterate through each cell in the board, we check if the digit already exists in the corresponding row, column, or box set. If it does, the board is invalid. Otherwise, we add the digit to all three sets.

The box index calculation is crucial: for a cell at position (i, j), the box index is (i/3)*3 + j/3, which correctly maps the 9x9 grid into 9 boxes arranged in a 3x3 grid. This formula groups cells (0-2, 0-2) into box 0, (0-2, 3-5) into box 1, etc.

Edge cases handled include: empty cells (represented by '.', skipped in validation), boards with duplicate digits in rows/columns/boxes (detected immediately), partially filled boards (only validate filled cells), and completely empty boards (valid, as there are no violations).

An alternative approach would be to check rows, columns, and boxes separately in three passes, but the single-pass approach is more efficient. Another approach would use bit masks instead of sets, but sets are more readable and the space difference is negligible for 9x9 boards.

Think of this as having three separate checklists for each constraint: as you fill in each cell, you mark that digit as used in the row checklist, column checklist, and box checklist. If you try to mark a digit that's already marked, you've found a violation.`,
	},
	{
		ID:          63,
		Title:       "Encode and Decode Strings",
		Topic:       "Arrays",
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
		Explanation: `Encoding and decoding a list of strings requires a way to distinguish where one string ends and another begins. The key insight is to prefix each string with its length followed by a delimiter, which allows unambiguous decoding without special character escaping.

The encoding algorithm iterates through each string, prepending its length as a string, followed by a delimiter character ('#'). For example, "hello" becomes "5#hello". This format allows the decoder to know exactly how many characters to read for each string.

The decoding algorithm reads the encoded string character by character. It first reads digits until it encounters the delimiter, converts those digits to an integer representing the length, then reads exactly that many characters as the string content. It repeats this process until the entire encoded string is processed.

This approach handles strings containing the delimiter character correctly because we use the length prefix to know exactly how many characters to read, rather than relying on the delimiter to mark the end. The delimiter is only used to separate the length from the content.

Edge cases handled include: empty strings (encoded as "0#"), strings containing the delimiter character (length prefix ensures correct decoding), strings with numeric prefixes (length is explicitly specified, not inferred), and very long strings (length can be multi-digit).

An alternative approach would use escape sequences for special characters, but that's more complex. Another approach would use a different encoding scheme, but the length-prefix approach is simple and efficient. Both encoding and decoding are O(n) where n is the total length of all strings.

Think of this as writing a recipe: you first write how many ingredients you need, then list them. When reading, you first check the count, then read exactly that many ingredients.`,
	},
	{
		ID:          64,
		Title:       "Longest Substring Without Repeating Characters",
		Topic:       "Sliding Window",
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
		Explanation: `Finding the longest substring without repeating characters requires efficiently tracking which characters we've seen and where. The sliding window technique combined with a hash map is perfect for this problem.

The algorithm maintains a sliding window defined by start and end pointers, and a hash map that stores the last seen index of each character. As we expand the window by moving the end pointer, we check if the current character has been seen within the current window. If it has (and its last index is >= start), we move the start pointer to just after that character's last occurrence, effectively removing the duplicate and all characters before it from our window.

This approach is optimal because each character is visited at most twice (once by end pointer, once when start pointer moves past it), giving us O(n) time complexity. The space complexity is O(min(n, m)) where m is the size of the character set, as we only store characters we've encountered.

Edge cases handled include: strings with all unique characters (entire string is the answer), strings with all same characters (answer is 1), empty strings (returns 0), and strings where the longest substring appears at the end (window expands to include it).

An alternative O(n²) approach would check all possible substrings, but the sliding window approach is optimal. Another approach would use a set and shrink the window character by character, but tracking last seen index allows us to jump the start pointer, making it more efficient.

Think of this as maintaining a window of unique characters: when you try to add a character that's already in the window, you slide the left edge of the window past that character's previous occurrence, keeping only unique characters.`,
	},
	{
		ID:          65,
		Title:       "Longest Repeating Character Replacement",
		Topic:       "Sliding Window",
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
		Explanation: `This problem requires finding the longest substring where we can make at most k character replacements to make all characters the same. The key insight is that we want to maximize the window size where the number of characters that need replacement (window size - most frequent character count) is at most k.

The algorithm uses a sliding window approach with a hash map to track character frequencies. As we expand the window by moving the right pointer, we update the frequency count and track the maximum frequency of any character in the current window. The window is valid if the number of characters that need to be replaced (window size - maxCount) is at most k. When the window becomes invalid, we shrink it from the left.

The key optimization is that we don't need to update maxCount when we shrink the window - we only update it when expanding. This works because we're looking for the maximum window size, and if a smaller window had a higher maxCount, we would have already found a larger valid window. This allows the algorithm to achieve O(n) time complexity.

Edge cases handled include: k=0 (can only use existing character sequences), strings with all same characters (entire string is valid), strings where k is large enough to replace all characters (entire string becomes valid), and strings shorter than k+1 (entire string is valid).

An alternative approach would be to try all possible characters as the target and find the longest substring for each, but that would be less efficient. The sliding window approach with frequency tracking is optimal, achieving O(n) time with O(1) space (limited character set).

Think of this as finding the longest stretch where you can "fix" at most k characters to match the most common character in that stretch. You expand the window until you need more than k fixes, then shrink until it's valid again.`,
	},
	{
		ID:          66,
		Title:       "Permutation in String",
		Topic:       "Sliding Window",
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
		Explanation: `Checking if one string contains a permutation of another requires comparing character frequencies in a sliding window. The key insight is that a permutation has the same character frequencies as the original string, just in a different order.

The algorithm uses a sliding window of fixed size (length of s1) and maintains frequency counts for both s1 and the current window in s2. It tracks how many character frequencies match between the two. When all 26 character frequencies match (or all non-zero frequencies match), we've found a permutation.

The algorithm first initializes frequency counts for s1 and the first window of s2. Then, as it slides the window through s2, it updates the frequency counts by removing the leftmost character and adding the new rightmost character. It efficiently updates the match count by checking if frequencies become equal or unequal as characters are added/removed.

This approach is optimal because it only needs to check character frequencies, not actual character order. The sliding window ensures we check all possible substrings of the correct length, and the match counting avoids comparing all 26 frequencies each time.

Edge cases handled include: s1 longer than s2 (impossible, return false), s1 equals s2 (permutation found immediately), s2 contains multiple permutations (returns true on first found), and strings with repeated characters (frequency matching handles this correctly).

An alternative approach would be to sort each window and compare with sorted s1, but that would be O(n*k*log k) where k is the window size. The frequency-based approach achieves O(n) time with O(1) space (26 character frequencies).

Think of this as having a template (s1's character counts) and sliding it over s2, checking if any window has the exact same character composition, regardless of order.`,
	},
	{
		ID:          67,
		Title:       "Minimum Window Substring",
		Topic:       "Sliding Window",
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
		Explanation: `Finding the minimum window substring that contains all characters of another string requires a sophisticated sliding window approach. The key challenge is efficiently tracking when the window contains all required characters with sufficient frequency.

The algorithm uses two hash maps: one for the target string's character frequencies and one for the current window's character frequencies. It maintains a "formed" counter that tracks how many unique characters in the target have been satisfied in the current window (i.e., their frequency in the window equals or exceeds their frequency in the target).

The algorithm expands the window by moving the right pointer, adding characters and updating the formed counter when a character's frequency requirement is met. When all characters are satisfied (formed == required), it tries to shrink the window from the left while maintaining validity, tracking the minimum valid window found.

This approach ensures we find the minimum window because we only shrink when the window is valid, and we check all possible valid windows. The two-pointer technique ensures each character is visited at most twice, giving O(m+n) time complexity.

Edge cases handled include: s shorter than t (impossible, return empty string), t contains characters not in s (impossible, return empty string), multiple valid windows (finds the minimum), and t with duplicate characters (frequency requirements must be met).

An alternative approach would be to check all possible substrings, but that would be O(n²*m) time. The sliding window approach is optimal, achieving O(m+n) time with O(m+n) space for the frequency maps.

Think of this as finding the shortest stretch of text that contains all the letters you need: you expand until you have everything, then shrink from the left while still having everything, keeping track of the shortest valid stretch.`,
	},
	{
		ID:          68,
		Title:       "Valid Palindrome II",
		Topic:       "Two Pointers",
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
		Explanation: `This problem extends the palindrome check by allowing at most one character deletion. The key insight is that when we find a mismatch, we can try deleting either the left or right character and check if the remaining substring is a palindrome.

The algorithm uses two pointers starting from both ends. As it compares characters, if it finds a mismatch, it doesn't immediately return false. Instead, it tries two possibilities: skip the left character (check if substring from left+1 to right is a palindrome) or skip the right character (check if substring from left to right-1 is a palindrome). If either possibility results in a valid palindrome, the original string can be made a palindrome by deleting one character.

The helper function isPalindromeRange efficiently checks if a substring is a palindrome using the standard two-pointer technique. This allows us to avoid creating new strings and check palindromes in-place.

Edge cases handled include: strings that are already palindromes (return true without deletion), strings requiring deletion of the first character, strings requiring deletion of the last character, strings where deletion doesn't help (return false), and single-character strings (always palindromes).

An alternative approach would be to try deleting each character and check if the result is a palindrome, but that would be O(n²) time. The current approach is more efficient, achieving O(n) time in the best case and O(n) in the worst case when we need to check both possibilities.

Think of this as checking a word with a possible typo: when you find characters that don't match, you try removing one character from either side to see if the rest forms a valid palindrome.`,
	},
	{
		ID:          69,
		Title:       "3Sum",
		Topic:       "Two Pointers",
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
		Explanation: `Finding all unique triplets that sum to zero requires avoiding duplicates efficiently. The key insight is to sort the array first, then use a combination of fixing one number and two pointers for the remaining two numbers.

The algorithm first sorts the array, which allows us to use two pointers and skip duplicates efficiently. For each position i, it fixes nums[i] as the first number and uses two pointers (left and right) to find pairs that sum to -nums[i]. When a valid triplet is found, it's added to the result, and both pointers skip over duplicate values to avoid duplicate triplets.

The duplicate skipping is crucial: after fixing nums[i], if nums[i] equals nums[i-1], we skip it because we've already considered all triplets starting with that value. Similarly, when we find a valid triplet, we skip duplicate values for both left and right pointers before moving them.

This approach efficiently finds all unique triplets in O(n²) time. The sorting takes O(n log n) time, and for each of the n positions, the two-pointer search takes O(n) time, giving overall O(n²) time complexity.

Edge cases handled include: arrays with no valid triplets (returns empty array), arrays with all zeros (returns one triplet [0,0,0]), arrays with duplicate numbers (skipped correctly), and arrays with fewer than three elements (returns empty array).

An alternative O(n³) approach would check all triplets, but the two-pointer approach is more efficient. Another approach would use a hash map, but handling duplicates becomes more complex. The sorting + two-pointer approach is optimal for this problem.

Think of this as fixing the first number and then solving the two-sum problem for the remaining numbers, with careful handling to avoid counting the same triplet multiple times.`,
	},
	{
		ID:          70,
		Title:       "Container With Most Water",
		Topic:       "Two Pointers",
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
		Explanation: `Finding the container with the most water requires maximizing the area formed by two vertical lines. The area is calculated as width (distance between indices) times height (minimum of the two line heights). The key insight is that we should move the pointer with the smaller height inward, as keeping it would never yield a larger area.

The algorithm uses two pointers starting from both ends of the array. At each step, it calculates the area with the current two lines and updates the maximum area if needed. Then, it moves the pointer pointing to the smaller height inward. This is optimal because if we kept the smaller height and moved the larger height pointer, the width would decrease while the height could only stay the same or decrease (limited by the smaller height), so the area would definitely decrease.

This greedy approach works because we're always eliminating the possibility of a better solution by moving the smaller pointer. By the time the pointers meet, we've considered all possible pairs where one pointer could contribute to a larger area.

Edge cases handled include: arrays with all same heights (area decreases as width decreases), arrays with increasing heights (optimal pair is at the ends), arrays with decreasing heights (optimal pair is at the ends), and arrays with two elements (those two form the container).

An alternative O(n²) approach would check all pairs, but the two-pointer approach is optimal, achieving O(n) time with O(1) space. The key insight that moving the smaller pointer is always safe makes this possible.

Think of this as having two walls and trying to maximize the water between them: you always move the shorter wall inward because keeping it would only give you a smaller container (less width, same or smaller height).`,
	},
	{
		ID:          71,
		Title:       "Trapping Rain Water",
		Topic:       "Two Pointers",
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
		Explanation: `Trapping rain water requires calculating how much water can be held at each position, which depends on the maximum heights to the left and right. The key insight is that water trapped at a position equals the minimum of the left and right maximum heights minus the current height.

The algorithm uses two pointers starting from both ends, maintaining the maximum height seen from each side. At each step, it processes the side with the smaller maximum height, because the water level at that position is limited by the smaller maximum. It calculates the trapped water by subtracting the current height from the maximum height on that side, then updates the maximum if the current height is greater.

This two-pointer approach is optimal because it processes each position exactly once while maintaining the necessary maximum height information. By always processing the side with the smaller maximum, we ensure we're calculating water correctly without needing to know the maximum on the other side for positions we've already processed.

Edge cases handled include: empty arrays (returns 0), arrays with no valleys (returns 0), arrays with all same heights (returns 0), arrays where water can be trapped (correctly calculated), and arrays with increasing or decreasing heights (no water trapped).

An alternative approach would use two passes to calculate left and right maximums separately, then a third pass to calculate water, but that requires O(n) extra space. The two-pointer approach achieves O(1) space by calculating on-the-fly.

Think of this as filling a valley with water: the water level at any point is determined by the lower of the two "walls" on either side. You process from the side with the lower wall because that's what limits the water level.`,
	},
	{
		ID:          72,
		Title:       "Minimum Size Subarray Sum",
		Topic:       "Sliding Window",
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
		Explanation: `Finding the minimum length subarray with sum >= target requires efficiently tracking subarrays as we expand and shrink a window. The sliding window technique is perfect for this, as we can maintain a valid window and minimize its size.

The algorithm uses two pointers to maintain a sliding window. It expands the window by moving the right pointer, adding elements to the sum. When the sum becomes >= target, the current window is valid, so we try to shrink it from the left while maintaining validity. We track the minimum length of valid windows encountered.

This approach is optimal because each element is added once when the right pointer passes it and removed at most once when the left pointer passes it, giving O(n) time complexity. The space complexity is O(1) as we only maintain a few variables.

Edge cases handled include: arrays where no subarray sums to target (returns 0), arrays where a single element >= target (returns 1), arrays where the entire array is needed (returns array length), and arrays with all elements smaller than target individually but sum >= target when combined.

An alternative O(n²) approach would check all possible subarrays, but the sliding window approach is optimal. Another approach would use prefix sums and binary search, but that's more complex and still O(n log n). The sliding window achieves optimal O(n) time.

Think of this as finding the shortest stretch of numbers that add up to at least the target: you expand until you reach the target, then shrink from the left while still meeting the target, keeping track of the shortest valid stretch.`,
	},
	{
		ID:          73,
		Title:       "Maximum Subarray",
		Topic:       "Dynamic Programming",
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
		Explanation: `Kadane's algorithm is the optimal solution for finding the maximum sum of a contiguous subarray. The key insight is that if the current sum becomes negative, it's better to start a new subarray from the next element rather than carrying forward a negative sum.

The algorithm maintains two variables: the current sum of the subarray ending at the current position, and the maximum sum seen so far. As we iterate through the array, we add each element to the current sum. If the current sum becomes negative, we reset it to the current element (effectively starting a new subarray). At each step, we update the maximum sum if the current sum is greater.

This approach works because any subarray that starts with a negative sum can be improved by starting from the next element instead. By resetting to 0 (or the current element) when the sum becomes negative, we ensure we're always considering the best possible starting point for our current subarray.

Edge cases handled include: arrays with all negative numbers (maximum sum is the least negative number), arrays with all positive numbers (sum of entire array), arrays with a single element, and arrays with mixed positive and negative numbers where the maximum subarray is in the middle.

An alternative O(n²) approach would check all possible subarrays, but Kadane's algorithm is optimal. Another approach would use divide-and-conquer, but it's more complex and still O(n log n). Kadane's algorithm achieves O(n) time with O(1) space.

Think of this as walking along a path collecting coins: if your current collection becomes negative (you've lost more than gained), it's better to drop everything and start fresh from the next position rather than carrying forward losses.`,
	},
	{
		ID:          74,
		Title:       "Jump Game",
		Topic:       "Dynamic Programming",
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
		Explanation: `Determining if we can reach the end of an array where each element represents maximum jump length requires tracking how far we can reach from the current position. The greedy approach efficiently solves this by maintaining the maximum reachable index.

The algorithm iterates through the array, updating the maximum reachable index as it goes. At each position i, it calculates the farthest it can reach from that position (i + nums[i]) and updates the maximum if this is greater. The key check is whether the current index exceeds the maximum reachable index - if it does, we cannot proceed further and return false.

This greedy approach works because if we can reach position i, we can reach any position before i. Therefore, we only need to track the farthest position we can reach, not all reachable positions. If we can reach or exceed the last index, we return true.

Edge cases handled include: arrays with a single element (already at end, return true), arrays where the first element is 0 and length > 1 (cannot proceed, return false), arrays where we can jump directly to the end, and arrays where we need to use multiple jumps to reach the end.

An alternative dynamic programming approach would track reachability for each position, but that requires O(n) space. The greedy approach achieves O(n) time with O(1) space by only tracking the maximum reachable index.

Think of this as a race where each checkpoint tells you how far you can jump ahead: you track the farthest checkpoint you can reach, and if you ever find yourself at a checkpoint beyond your maximum reach, you know you can't finish the race.`,
	},
	{
		ID:          75,
		Title:       "Jump Game II",
		Topic:       "Dynamic Programming",
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
		Explanation: `Finding the minimum number of jumps requires a greedy BFS-like approach where we determine the farthest we can reach with each jump. The key insight is that we should jump as far as possible with each jump to minimize the total number of jumps.

The algorithm maintains two key variables: currentEnd represents the farthest position reachable with the current number of jumps, and farthest represents the farthest position we can reach from any position in the current jump range. As we iterate through positions within the current jump range, we update farthest to the maximum reachable position. When we reach currentEnd, we've exhausted all options for the current number of jumps, so we increment the jump count and set currentEnd to farthest.

This greedy approach is optimal because jumping to the farthest reachable position gives us the maximum options for the next jump, which minimizes the total number of jumps needed. It's similar to BFS where each "level" represents one jump, and we want to find the minimum number of levels to reach the end.

Edge cases handled include: arrays where we can reach the end in one jump (returns 1), arrays requiring multiple jumps, arrays where the first element allows reaching the end directly, and arrays where we must use every position optimally.

An alternative dynamic programming approach would calculate minimum jumps for each position, but that requires O(n²) time in the worst case. The greedy BFS approach achieves O(n) time with O(1) space by tracking only the current jump range and farthest reachable position.

Think of this as level-order traversal (BFS) of jump possibilities: each level represents one jump, and you want to find the minimum level where you can reach the end. You process all positions in the current level, find the farthest you can reach, then move to the next level.`,
	},
	{
		ID:          76,
		Title:       "Gas Station",
		Topic:       "Greedy",
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
		Explanation: `This problem requires finding a starting gas station in a circular route where we can complete a full circuit. The key insight is that if the total gas is less than the total cost, it's impossible. Otherwise, there exists a valid starting point, and we can find it by tracking where we can maintain a non-negative tank.

The algorithm maintains a running tank level as it simulates the journey. It also tracks the total gas and total cost to check feasibility. As it iterates through stations, if the tank becomes negative at any point, it means we cannot start from any previous station and reach the current station, so we reset the starting point to the next station and reset the tank.

The key observation is that if we start from station A and cannot reach station B, then starting from any station between A and B will also fail to reach B (because we would have even less gas). Therefore, we can skip all those stations and try starting from B.

Edge cases handled include: total gas < total cost (impossible, return -1), single station (check if gas >= cost), stations where we can start from the first position, and stations where we need to start from a later position.

An alternative approach would try starting from each station, but that would be O(n²) time. The greedy approach achieves O(n) time by identifying that if we fail to reach a station from a starting point, we can skip all intermediate starting points.

Think of this as driving around a circular track: if you run out of gas between two stations, you know you couldn't have started from any station before where you ran out, so you try starting from the next station.`,
	},
	{
		ID:          77,
		Title:       "Hand of Straights",
		Topic:       "Hash Tables",
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
		Explanation: `Determining if cards can be arranged into groups of consecutive cards requires checking if we have enough cards of each value to form complete groups. The key insight is to process cards in sorted order and greedily form groups starting from the smallest available card.

The algorithm first checks if the total number of cards is divisible by groupSize (otherwise it's impossible). It counts the frequency of each card value using a hash map. Then, it iterates through cards in sorted order. For each card that still has remaining count, it tries to form a group of groupSize consecutive cards starting from that card. If any required consecutive card is missing, it returns false.

The greedy approach works because by always starting groups from the smallest available card, we ensure we use cards optimally. If we skip a card and try to form a group starting from a larger card, we might leave the smaller card unused, which could prevent forming valid groups later.

Edge cases handled include: number of cards not divisible by groupSize (returns false immediately), cards with duplicate values (handled by frequency counting), groups requiring cards we don't have (returns false), and groups that can be formed in multiple ways (greedy approach finds one valid way).

An alternative approach would try all possible arrangements, but that would be exponential. The greedy sorting approach is efficient, achieving O(n log n) time for sorting and O(n) time for group formation, with O(n) space for the frequency map.

Think of this as organizing playing cards into straights: you always start with the lowest card you have, try to form a consecutive group, and if you can't complete a group, you know it's impossible.`,
	},
	{
		ID:          78,
		Title:       "Longest Substring Without Repeating Characters",
		Description: "Given a string s, find the length of the longest substring without repeating characters.",
		Difficulty:  "Medium",
		Topic:       "Sliding Window",
		Signature:   "func lengthOfLongestSubstring(s string) int",
		TestCases: []TestCase{
			{Input: "s = \"abcabcbb\"", Expected: "3"},
			{Input: "s = \"bbbbb\"", Expected: "1"},
			{Input: "s = \"pwwkew\"", Expected: "3"},
			{Input: "s = \"\"", Expected: "0"},
			{Input: "s = \"dvdf\"", Expected: "3"},
			{Input: "s = \"abcdef\"", Expected: "6"},
		},
		Solution: `func lengthOfLongestSubstring(s string) int {
	charMap := make(map[byte]int)
	maxLen := 0
	left := 0
	for right := 0; right < len(s); right++ {
		if idx, exists := charMap[s[right]]; exists && idx >= left {
			left = idx + 1
		}
		charMap[s[right]] = right
		if right-left+1 > maxLen {
			maxLen = right - left + 1
		}
	}
	return maxLen
}`,
		PythonSolution: `def lengthOfLongestSubstring(s: str) -> int:
    char_map = {}
    max_len = 0
    left = 0
    for right in range(len(s)):
        if s[right] in char_map and char_map[s[right]] >= left:
            left = char_map[s[right]] + 1
        char_map[s[right]] = right
        max_len = max(max_len, right - left + 1)
    return max_len`,
		Explanation: `This problem requires finding the longest substring without repeating characters. The sliding window technique combined with a hash map that tracks the last occurrence of each character provides an optimal solution.

The algorithm maintains a sliding window using two pointers (left and right) and a hash map that stores the last index where each character was seen. As the right pointer expands the window, if we encounter a character that's already in the current window (its last occurrence index >= left), we move the left pointer to just after that last occurrence. This efficiently removes the duplicate and all characters before it from the window.

This approach is optimal because each character is visited at most twice (once by the right pointer, once when the left pointer moves past it), giving O(n) time complexity. The space complexity is O(min(n, m)) where m is the size of the character set, as we only store characters we've encountered.

Edge cases handled include: strings with all unique characters (entire string is the answer), strings with all same characters (answer is 1), empty strings (returns 0), and strings where the longest substring appears at different positions (algorithm finds the maximum).

An alternative O(n²) approach would check all possible substrings, but the sliding window approach is optimal. Another approach would use a set and shrink the window character by character, but tracking last occurrence allows us to jump the left pointer, making it more efficient.

Think of this as maintaining a window of unique characters: when you try to add a character that's already in the window, you slide the left edge of the window past that character's previous occurrence, keeping only unique characters.`,
	},
	{
		ID:          79,
		Title:       "Container With Most Water",
		Description: "You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]). Find two lines that together with the x-axis form a container, such that the container contains the most water. Return the maximum amount of water a container can store.",
		Difficulty:  "Medium",
		Topic:       "Two Pointers",
		Signature:   "func maxArea(height []int) int",
		TestCases: []TestCase{
			{Input: "height = []int{1,8,6,2,5,4,8,3,7}", Expected: "49"},
			{Input: "height = []int{1,1}", Expected: "1"},
			{Input: "height = []int{1,2,1}", Expected: "2"},
			{Input: "height = []int{2,3,4,5,18,17,6}", Expected: "17"},
			{Input: "height = []int{1,3,2,5,25,24,5}", Expected: "24"},
			{Input: "height = []int{10,9,8,7,6,5,4,3,2,1}", Expected: "25"},
		},
		Solution: `func maxArea(height []int) int {
	left, right := 0, len(height)-1
	maxArea := 0
	for left < right {
		width := right - left
		area := width * min(height[left], height[right])
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
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}`,
		PythonSolution: `def maxArea(height: List[int]) -> int:
    left, right = 0, len(height) - 1
    max_area = 0
    while left < right:
        width = right - left
        area = width * min(height[left], height[right])
        max_area = max(max_area, area)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_area`,
		Explanation: `Finding the container with the most water requires maximizing the area formed by two vertical lines. The area is width (distance between indices) times height (minimum of the two line heights). The two-pointer technique optimally solves this by eliminating impossible candidates.

The algorithm uses two pointers starting from both ends. At each step, it calculates the area with the current two lines and updates the maximum. Then, it moves the pointer pointing to the smaller height inward. This is optimal because keeping the smaller height and moving the larger height pointer would decrease width while height could only stay the same or decrease (limited by the smaller height), so area would definitely decrease.

This greedy approach works because we're always eliminating the possibility of a better solution by moving the smaller pointer. By the time the pointers meet, we've considered all possible pairs where one pointer could contribute to a larger area.

Edge cases handled include: arrays with all same heights (area decreases as width decreases), arrays with increasing heights (optimal pair is at the ends), arrays with decreasing heights (optimal pair is at the ends), arrays with two elements (those two form the container), and arrays where the optimal pair is in the middle (algorithm finds it).

An alternative O(n²) approach would check all pairs, but the two-pointer approach is optimal, achieving O(n) time with O(1) space. The key insight that moving the smaller pointer is always safe makes this possible.

Think of this as having two walls and trying to maximize the water between them: you always move the shorter wall inward because keeping it would only give you a smaller container (less width, same or smaller height).`,
	},
	{
		ID:          80,
		Title:       "3Sum",
		Description: "Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0. Notice that the solution set must not contain duplicate triplets.",
		Difficulty:  "Medium",
		Topic:       "Two Pointers",
		Signature:   "func threeSum(nums []int) [][]int",
		TestCases: []TestCase{
			{Input: "nums = []int{-1,0,1,2,-1,-4}", Expected: "[[-1 -1 2] [-1 0 1]]"},
			{Input: "nums = []int{0,1,1}", Expected: "[]"},
			{Input: "nums = []int{0,0,0}", Expected: "[[0 0 0]]"},
			{Input: "nums = []int{-2,0,1,1,2}", Expected: "[[-2 0 2] [-2 1 1]]"},
			{Input: "nums = []int{-1,0,1}", Expected: "[[-1 0 1]]"},
			{Input: "nums = []int{}", Expected: "[]"},
		},
		Solution: `import "sort"

func threeSum(nums []int) [][]int {
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
		Explanation: `Finding all unique triplets that sum to zero requires efficiently avoiding duplicates. The key insight is to sort the array first, then use a combination of fixing one number and two pointers for the remaining two numbers, with careful duplicate skipping.

The algorithm first sorts the array, which allows us to use two pointers and skip duplicates efficiently. For each position i, it fixes nums[i] as the first number and uses two pointers (left and right) to find pairs that sum to -nums[i]. When a valid triplet is found, it's added to the result, and both pointers skip over duplicate values to avoid duplicate triplets.

The duplicate skipping is crucial: after fixing nums[i], if nums[i] equals nums[i-1], we skip it because we've already considered all triplets starting with that value. Similarly, when we find a valid triplet, we skip duplicate values for both left and right pointers before moving them.

This approach efficiently finds all unique triplets in O(n²) time. The sorting takes O(n log n) time, and for each of the n positions, the two-pointer search takes O(n) time, giving overall O(n²) time complexity.

Edge cases handled include: arrays with no valid triplets (returns empty array), arrays with all zeros (returns one triplet [0,0,0]), arrays with duplicate numbers (skipped correctly), arrays with fewer than three elements (returns empty array), and arrays where triplets appear at different positions.

An alternative O(n³) approach would check all triplets, but the two-pointer approach is more efficient. Another approach would use a hash map, but handling duplicates becomes more complex. The sorting + two-pointer approach is optimal for this problem.

Think of this as fixing the first number and then solving the two-sum problem for the remaining numbers, with careful handling to avoid counting the same triplet multiple times.`,
	},
	{
		ID:          81,
		Title:       "Set Matrix Zeroes",
		Description: "Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's. You must do it in place.",
		Difficulty:  "Medium",
		Topic:       "Arrays",
		Signature:   "func setZeroes(matrix [][]int)",
		TestCases: []TestCase{
			{Input: "matrix = [[1,1,1],[1,0,1],[1,1,1]]", Expected: "[[1 0 1] [0 0 0] [1 0 1]]"},
			{Input: "matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]", Expected: "[[0 0 0 0] [0 4 5 0] [0 3 1 0]]"},
			{Input: "matrix = [[1,0,1],[1,1,1],[1,1,1]]", Expected: "[[0 0 0] [1 0 1] [1 0 1]]"},
			{Input: "matrix = [[1]]", Expected: "[[1]]"},
			{Input: "matrix = [[0]]", Expected: "[[0]]"},
		},
		Solution: `func setZeroes(matrix [][]int) {
	m, n := len(matrix), len(matrix[0])
	firstRowZero := false
	firstColZero := false
	for j := 0; j < n; j++ {
		if matrix[0][j] == 0 {
			firstRowZero = true
			break
		}
	}
	for i := 0; i < m; i++ {
		if matrix[i][0] == 0 {
			firstColZero = true
			break
		}
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if matrix[i][j] == 0 {
				matrix[i][0] = 0
				matrix[0][j] = 0
			}
		}
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if matrix[i][0] == 0 || matrix[0][j] == 0 {
				matrix[i][j] = 0
			}
		}
	}
	if firstRowZero {
		for j := 0; j < n; j++ {
			matrix[0][j] = 0
		}
	}
	if firstColZero {
		for i := 0; i < m; i++ {
			matrix[i][0] = 0
		}
	}
}`,
		PythonSolution: `def setZeroes(matrix: List[List[int]]) -> None:
    m, n = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][j] == 0 for j in range(n))
    first_col_zero = any(matrix[i][0] == 0 for i in range(m))
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0`,
		Explanation: `Setting entire rows and columns to zero when a zero is found requires an in-place solution that doesn't use extra space for tracking. The key insight is to use the first row and first column as markers to indicate which rows and columns should be zeroed.

The algorithm first checks if the first row or first column originally contained any zeros (stored in boolean flags). Then, it uses the first row and first column as markers: for any zero found at position (i, j) where i > 0 and j > 0, it marks matrix[i][0] = 0 and matrix[0][j] = 0. After marking all zeros, it iterates through the matrix again (excluding first row/column) and sets cells to zero if their row or column marker is zero. Finally, it handles the first row and column separately based on the flags.

This approach achieves O(1) space complexity by reusing the matrix itself for storage, rather than using separate arrays to track which rows/columns should be zeroed. The time complexity is O(m*n) as we need to scan the matrix multiple times.

Edge cases handled include: matrices with zeros in the first row (handled by flag), matrices with zeros in the first column (handled by flag), matrices with zeros at (0,0) (both flags set), matrices with no zeros (no changes), and single-cell matrices (handled correctly).

An alternative approach would use O(m+n) extra space to store which rows and columns to zero, but the marker approach achieves O(1) space. Another approach would use a set to track zero positions, but that also requires extra space.

Think of this as using the borders of the matrix as a checklist: when you find a zero, you mark its row and column on the borders, then go back and zero out all marked rows and columns.`,
	},
	{
		ID:          82,
		Title:       "Spiral Matrix",
		Description: "Given an m x n matrix, return all elements of the matrix in spiral order.",
		Difficulty:  "Medium",
		Topic:       "Arrays",
		Signature:   "func spiralOrder(matrix [][]int) []int",
		TestCases: []TestCase{
			{Input: "matrix = [[1,2,3],[4,5,6],[7,8,9]]", Expected: "[1 2 3 6 9 8 7 4 5]"},
			{Input: "matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]", Expected: "[1 2 3 4 8 12 11 10 9 5 6 7]"},
			{Input: "matrix = [[1]]", Expected: "[1]"},
			{Input: "matrix = [[1,2],[3,4]]", Expected: "[1 2 4 3]"},
			{Input: "matrix = [[1,2,3]]", Expected: "[1 2 3]"},
			{Input: "matrix = [[1],[2],[3]]", Expected: "[1 2 3]"},
		},
		Solution: `func spiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 {
		return []int{}
	}
	m, n := len(matrix), len(matrix[0])
	result := make([]int, 0, m*n)
	top, bottom, left, right := 0, m-1, 0, n-1
	for top <= bottom && left <= right {
		for j := left; j <= right; j++ {
			result = append(result, matrix[top][j])
		}
		top++
		for i := top; i <= bottom; i++ {
			result = append(result, matrix[i][right])
		}
		right--
		if top <= bottom {
			for j := right; j >= left; j-- {
				result = append(result, matrix[bottom][j])
			}
			bottom--
		}
		if left <= right {
			for i := bottom; i >= top; i-- {
				result = append(result, matrix[i][left])
			}
			left++
		}
	}
	return result
}`,
		PythonSolution: `def spiralOrder(matrix: List[List[int]]) -> List[int]:
    if not matrix:
        return []
    m, n = len(matrix), len(matrix[0])
    result = []
    top, bottom, left, right = 0, m - 1, 0, n - 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
    return result`,
		Explanation: `Traversing a matrix in spiral order requires systematically moving in four directions: right, down, left, and up, while maintaining boundaries that shrink as we complete each "layer". The key insight is to use four boundary variables to track the current spiral layer.

The algorithm maintains four boundaries: top, bottom, left, and right, which define the current rectangular layer to traverse. It processes the matrix layer by layer, moving right along the top row, down along the right column, left along the bottom row, and up along the left column. After completing each direction, it adjusts the corresponding boundary inward, effectively moving to the next inner layer.

The boundary checks (top <= bottom and left <= right) ensure we don't process the same row or column twice, which is important for non-square matrices. After moving right and down, we check if there's still a row/column to process before moving left and up.

Edge cases handled include: single-row matrices (only right traversal), single-column matrices (only down traversal), square matrices (complete spiral), rectangular matrices (handled correctly by boundary checks), and single-cell matrices (returns that cell).

An alternative approach would use direction vectors and visited markers, but that requires O(m*n) extra space. The boundary approach achieves O(1) extra space (excluding output array) by using the boundaries to track progress.

Think of this as peeling an onion layer by layer: you go around the outer edge, then move inward and go around the next edge, continuing until you've processed all layers.`,
	},
	{
		ID:          83,
		Title:       "Rotate Image",
		Description: "You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise). You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.",
		Difficulty:  "Medium",
		Topic:       "Arrays",
		Signature:   "func rotate(matrix [][]int)",
		TestCases: []TestCase{
			{Input: "matrix = [[1,2,3],[4,5,6],[7,8,9]]", Expected: "[[7 4 1] [8 5 2] [9 6 3]]"},
			{Input: "matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]", Expected: "[[15 13 2 5] [14 3 4 1] [12 6 8 9] [16 7 10 11]]"},
			{Input: "matrix = [[1]]", Expected: "[[1]]"},
			{Input: "matrix = [[1,2],[3,4]]", Expected: "[[3 1] [4 2]]"},
		},
		Solution: `func rotate(matrix [][]int) {
	n := len(matrix)
	for i := 0; i < n/2; i++ {
		for j := i; j < n-i-1; j++ {
			temp := matrix[i][j]
			matrix[i][j] = matrix[n-1-j][i]
			matrix[n-1-j][i] = matrix[n-1-i][n-1-j]
			matrix[n-1-i][n-1-j] = matrix[j][n-1-i]
			matrix[j][n-1-i] = temp
		}
	}
}`,
		PythonSolution: `def rotate(matrix: List[List[int]]) -> None:
    n = len(matrix)
    for i in range(n // 2):
        for j in range(i, n - i - 1):
            temp = matrix[i][j]
            matrix[i][j] = matrix[n - 1 - j][i]
            matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j]
            matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i]
            matrix[j][n - 1 - i] = temp`,
		Explanation: `Rotating a matrix 90 degrees clockwise in-place requires careful element swapping. The key insight is that rotation can be done layer by layer, and within each layer, we rotate four elements in a cycle: top-left goes to top-right, top-right goes to bottom-right, bottom-right goes to bottom-left, and bottom-left goes to top-left.

The algorithm processes the matrix in concentric layers, starting from the outermost layer. For each layer, it rotates elements in groups of four. The rotation pattern for a position (i, j) in an n×n matrix is: (i, j) → (n-1-j, i) → (n-1-i, n-1-j) → (j, n-1-i) → (i, j). This cycle of four swaps completes the rotation for one element and its three corresponding positions.

The layer boundaries are controlled by the outer loop (i from 0 to n/2) and inner loop (j from i to n-i-1), ensuring we process each layer exactly once without overlapping. This approach achieves O(n²) time complexity as we visit each element once, and O(1) space complexity as we only use a temporary variable for swapping.

Edge cases handled include: 1×1 matrices (no rotation needed), 2×2 matrices (four elements rotated), odd-sized matrices (center element stays in place), and even-sized matrices (all elements rotated).

An alternative approach would be to transpose the matrix and reverse each row, but that requires understanding the mathematical relationship. The four-element swap approach is more intuitive and directly implements the rotation pattern.

Think of this as rotating a picture frame: you swap the four corners of each layer, moving each corner to the next position clockwise, then move inward to the next layer.`,
	},
	{
		ID:          84,
		Title:       "Word Search",
		Description: "Given an m x n grid of characters board and a string word, return true if word exists in the grid. The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.",
		Difficulty:  "Medium",
		Topic:       "Backtracking",
		Signature:   "func exist(board [][]byte, word string) bool",
		TestCases: []TestCase{
			{Input: "board = [[\"A\",\"B\",\"C\",\"E\"],[\"S\",\"F\",\"C\",\"S\"],[\"A\",\"D\",\"E\",\"E\"]], word = \"ABCCED\"", Expected: "true"},
			{Input: "board = [[\"A\",\"B\",\"C\",\"E\"],[\"S\",\"F\",\"C\",\"S\"],[\"A\",\"D\",\"E\",\"E\"]], word = \"SEE\"", Expected: "true"},
			{Input: "board = [[\"A\",\"B\",\"C\",\"E\"],[\"S\",\"F\",\"C\",\"S\"],[\"A\",\"D\",\"E\",\"E\"]], word = \"ABCB\"", Expected: "false"},
			{Input: "board = [[\"A\"]], word = \"A\"", Expected: "true"},
			{Input: "board = [[\"A\",\"B\"],[\"C\",\"D\"]], word = \"ABCD\"", Expected: "false"},
		},
		Solution: `func exist(board [][]byte, word string) bool {
	m, n := len(board), len(board[0])
	visited := make([][]bool, m)
	for i := range visited {
		visited[i] = make([]bool, n)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if dfs(board, word, i, j, 0, visited) {
				return true
			}
		}
	}
	return false
}

func dfs(board [][]byte, word string, i, j, idx int, visited [][]bool) bool {
	if idx == len(word) {
		return true
	}
	if i < 0 || i >= len(board) || j < 0 || j >= len(board[0]) || visited[i][j] || board[i][j] != word[idx] {
		return false
	}
	visited[i][j] = true
	directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	for _, dir := range directions {
		if dfs(board, word, i+dir[0], j+dir[1], idx+1, visited) {
			return true
		}
	}
	visited[i][j] = false
	return false
}`,
		PythonSolution: `def exist(board: List[List[str]], word: str) -> bool:
    m, n = len(board), len(board[0])
    visited = [[False] * n for _ in range(m)]
    def dfs(i, j, idx):
        if idx == len(word):
            return True
        if i < 0 or i >= m or j < 0 or j >= n or visited[i][j] or board[i][j] != word[idx]:
            return False
        visited[i][j] = True
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if dfs(i + di, j + dj, idx + 1):
                return True
        visited[i][j] = False
        return False
    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0):
                return True
    return False`,
		Explanation: `Finding if a word exists in a 2D grid requires exploring paths through adjacent cells. This is a classic backtracking problem where we use depth-first search to explore all possible paths, marking cells as visited and unmarking them when backtracking.

The algorithm tries starting from each cell in the grid. For each starting position, it uses DFS to explore all four directions (up, down, left, right). It maintains a visited matrix to track which cells are currently in the current path, preventing cycles and reusing cells. When a path successfully matches the entire word, it returns true immediately. If a path fails, it backtracks by unmarking the current cell and trying the next direction.

The backtracking is crucial: after exploring a path that doesn't lead to the solution, we unmark the cell so it can be used in a different path. This allows the algorithm to explore all possible paths through the grid.

Edge cases handled include: word longer than total cells (impossible, returns false), word with repeated characters (backtracking handles this), word that appears multiple times (returns true on first found), and grids where the word requires going back over previously visited cells in a different path (backtracking allows this).

An alternative approach would use BFS, but DFS with backtracking is more natural for this path-finding problem. The time complexity is O(m*n*4^L) in the worst case where L is the word length, as we might explore all paths. The space complexity is O(L) for the recursion stack.

Think of this as a word search puzzle: you start from a cell, try moving in each direction to match the next letter, mark cells you've used, and if you get stuck, you backtrack and try a different path.`,
	},
	{
		ID:          85,
		Title:       "House Robber",
		Description: "You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night. Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.",
		Difficulty:  "Medium",
		Topic:       "Dynamic Programming",
		Signature:   "func rob(nums []int) int",
		TestCases: []TestCase{
			{Input: "nums = []int{1,2,3,1}", Expected: "4"},
			{Input: "nums = []int{2,7,9,3,1}", Expected: "12"},
			{Input: "nums = []int{2,1,1,2}", Expected: "4"},
			{Input: "nums = []int{1}", Expected: "1"},
			{Input: "nums = []int{1,2}", Expected: "2"},
			{Input: "nums = []int{2,1}", Expected: "2"},
		},
		Solution: `func rob(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	if len(nums) == 1 {
		return nums[0]
	}
	prev2, prev1 := nums[0], max(nums[0], nums[1])
	for i := 2; i < len(nums); i++ {
		curr := max(prev1, prev2+nums[i])
		prev2, prev1 = prev1, curr
	}
	return prev1
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def rob(nums: List[int]) -> int:
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    prev2, prev1 = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, curr
    return prev1`,
		Explanation: `This is a classic dynamic programming problem where we need to maximize the sum while ensuring no two adjacent elements are selected. The key insight is that for each house, we decide whether to rob it or skip it based on the maximum profit achievable up to that point.

The recurrence relation is: rob[i] = max(rob[i-1], rob[i-2] + nums[i]). This means the maximum profit at house i is either the profit from robbing up to house i-1 (skipping house i) or robbing house i plus the profit from robbing up to house i-2 (skipping house i-1 to avoid adjacency).

The algorithm uses an iterative approach with two variables (prev2 and prev1) to track the maximum profit from two houses ago and one house ago, avoiding the need for a full DP array. This achieves O(1) space complexity while maintaining O(n) time complexity.

Edge cases handled include: empty arrays (returns 0), single-house arrays (returns that house's value), two-house arrays (returns the maximum), arrays where robbing every other house is optimal, and arrays where skipping some houses gives better results.

An alternative recursive approach with memoization would have O(n) time and O(n) space. The iterative two-variable approach is more space-efficient. Another approach would use a full DP array, but the two-variable optimization is preferred.

Think of this as making decisions at each house: you can either take the money from this house plus the best you could do two houses ago, or skip this house and keep what you had from the previous house. You always choose the better option.`,
	},
	{
		ID:          86,
		Title:       "Coin Change",
		Description: "You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money. Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1. You may assume that you have an infinite number of each kind of coin.",
		Difficulty:  "Medium",
		Topic:       "Dynamic Programming",
		Signature:   "func coinChange(coins []int, amount int) int",
		TestCases: []TestCase{
			{Input: "coins = []int{1,2,5}, amount = 11", Expected: "3"},
			{Input: "coins = []int{2}, amount = 3", Expected: "-1"},
			{Input: "coins = []int{1}, amount = 0", Expected: "0"},
			{Input: "coins = []int{1}, amount = 1", Expected: "1"},
			{Input: "coins = []int{1}, amount = 2", Expected: "2"},
			{Input: "coins = []int{2,5,10,1}, amount = 27", Expected: "4"},
		},
		Solution: `func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	dp[0] = 0
	for i := 1; i <= amount; i++ {
		dp[i] = amount + 1
	}
	for i := 1; i <= amount; i++ {
		for _, coin := range coins {
			if coin <= i {
				dp[i] = min(dp[i], dp[i-coin]+1)
			}
		}
	}
	if dp[amount] > amount {
		return -1
	}
	return dp[amount]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}`,
		PythonSolution: `def coinChange(coins: List[int], amount: int) -> int:
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] <= amount else -1`,
		Explanation: `Finding the minimum number of coins to make an amount is a classic dynamic programming problem. The key insight is to build up the solution from smaller amounts: for each amount from 1 to the target, we calculate the minimum coins needed by trying each available coin.

The algorithm uses a DP array where dp[i] represents the minimum number of coins needed to make amount i. It initializes dp[0] = 0 (zero coins for zero amount) and all other values to a large number (amount + 1) to represent "impossible". For each amount i, it tries each coin: if the coin value is <= i, it updates dp[i] to the minimum of its current value and dp[i-coin] + 1.

This approach works because we're building optimal solutions for smaller amounts first. When calculating dp[i], we've already calculated the optimal solutions for all smaller amounts, so we can use them to find the best way to make amount i.

Edge cases handled include: amount = 0 (returns 0, no coins needed), impossible amounts (returns -1 when dp[amount] > amount), single coin denominations, coins larger than amount (skipped), and amounts requiring all coins of one type.

An alternative recursive approach with memoization would have the same time complexity but higher space complexity due to the recursion stack. The iterative DP approach is more efficient. The time complexity is O(amount * len(coins)) and space is O(amount).

Think of this as building up change from the smallest amount: for each possible amount, you try each coin and see if using that coin gives you a better (fewer coins) solution than what you already have.`,
	},
	{
		ID:          87,
		Title:       "Longest Increasing Subsequence",
		Description: "Given an integer array nums, return the length of the longest strictly increasing subsequence.",
		Difficulty:  "Medium",
		Topic:       "Dynamic Programming",
		Signature:   "func lengthOfLIS(nums []int) int",
		TestCases: []TestCase{
			{Input: "nums = []int{10,9,2,5,3,7,101,18}", Expected: "4"},
			{Input: "nums = []int{0,1,0,3,2,3}", Expected: "4"},
			{Input: "nums = []int{7,7,7,7,7,7,7}", Expected: "1"},
			{Input: "nums = []int{1}", Expected: "1"},
			{Input: "nums = []int{1,3,6,7,9,4,10,5,6}", Expected: "6"},
			{Input: "nums = []int{4,10,4,3,8,9}", Expected: "3"},
		},
		Solution: `func lengthOfLIS(nums []int) int {
	tails := []int{}
	for _, num := range nums {
		pos := binarySearch(tails, num)
		if pos == len(tails) {
			tails = append(tails, num)
		} else {
			tails[pos] = num
		}
	}
	return len(tails)
}

func binarySearch(tails []int, target int) int {
	left, right := 0, len(tails)
	for left < right {
		mid := (left + right) / 2
		if tails[mid] < target {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return left
}`,
		PythonSolution: `import bisect

def lengthOfLIS(nums: List[int]) -> int:
    tails = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)`,
		Explanation: `Finding the longest increasing subsequence efficiently requires a clever approach using patience sorting. The key insight is to maintain an array tails where tails[i] represents the smallest tail element of all increasing subsequences of length i+1.

The algorithm processes each number in the array. For each number, it uses binary search to find the leftmost position in tails where this number can be placed (i.e., the first position where tails[pos] >= num). If the position is at the end of tails, it means we found a longer subsequence, so we append the number. Otherwise, we replace the element at that position with the current number, which maintains the "smallest tail" property and allows for potentially longer subsequences later.

This approach is optimal because by keeping the smallest possible tail for each subsequence length, we maximize the chances of extending subsequences. The binary search makes each insertion O(log n), giving overall O(n log n) time complexity.

Edge cases handled include: arrays with all decreasing elements (LIS length is 1), arrays with all same elements (LIS length is 1), arrays that are already sorted (LIS length equals array length), and arrays with mixed patterns where the LIS is not contiguous.

An alternative O(n²) DP approach would calculate LIS ending at each position, but the patience sorting approach is more efficient. The space complexity is O(n) for the tails array, which in the worst case stores all elements.

Think of this as building card piles: you place each card on the leftmost pile where the top card is >= your card, or start a new pile if all top cards are smaller. The number of piles at the end is the length of the longest increasing subsequence.`,
	},
	{
		ID:          88,
		Title:       "Three Number Sum",
		Description: "Write a function that takes in a non-empty array of distinct integers and an integer representing a target sum. The function should find all triplets in the array that sum up to the target sum and return a two-dimensional array of all these triplets. The numbers in each triplet should be ordered in ascending order, and the triplets themselves should be ordered in ascending order with respect to the numbers they hold.",
		Difficulty:  "Medium",
		Topic:       "Arrays",
		Signature:   "func threeNumberSum(array []int, targetSum int) [][]int",
		TestCases: []TestCase{
			{Input: "array = []int{12, 3, 1, 2, -6, 5, -8, 6}, targetSum = 0", Expected: "[[-8 2 6] [-8 3 5] [-6 1 5]]"},
			{Input: "array = []int{1, 2, 3}, targetSum = 6", Expected: "[[1 2 3]]"},
			{Input: "array = []int{1, 2, 3}, targetSum = 7", Expected: "[]"},
			{Input: "array = []int{8, 10, -2, 49, 14}, targetSum = 57", Expected: "[[-2 10 49]]"},
		},
		Solution: `import "sort"

func threeNumberSum(array []int, targetSum int) [][]int {
	sort.Ints(array)
	triplets := [][]int{}
	for i := 0; i < len(array)-2; i++ {
		left := i + 1
		right := len(array) - 1
		for left < right {
			currentSum := array[i] + array[left] + array[right]
			if currentSum == targetSum {
				triplets = append(triplets, []int{array[i], array[left], array[right]})
				left++
				right--
			} else if currentSum < targetSum {
				left++
			} else {
				right--
			}
		}
	}
	return triplets
}`,
		PythonSolution: `def threeNumberSum(array: List[int], targetSum: int) -> List[List[int]]:
    array.sort()
    triplets = []
    for i in range(len(array) - 2):
        left = i + 1
        right = len(array) - 1
        while left < right:
            current_sum = array[i] + array[left] + array[right]
            if current_sum == targetSum:
                triplets.append([array[i], array[left], array[right]])
                left += 1
                right -= 1
            elif current_sum < targetSum:
                left += 1
            else:
                right -= 1
    return triplets`,
		Explanation: `Finding all triplets that sum to a target requires efficiently exploring combinations without duplicates. The key insight is to sort the array first, then use a combination of fixing one number and two pointers for the remaining two numbers.

The algorithm first sorts the array to enable the two-pointer technique and ensure results are in ascending order. For each position i, it fixes array[i] as the first number and uses two pointers (left and right) starting from i+1 and the end of the array. It calculates the sum and adjusts the pointers: if the sum equals the target, it adds the triplet and moves both pointers inward; if the sum is less than the target, it moves the left pointer right (to increase the sum); if greater, it moves the right pointer left (to decrease the sum).

This approach efficiently finds all valid triplets in O(n²) time. The sorting takes O(n log n) time, and for each of the n positions, the two-pointer search takes O(n) time, giving overall O(n²) time complexity. Since the array contains distinct integers, we don't need to skip duplicates.

Edge cases handled include: arrays with no valid triplets (returns empty array), arrays with exactly one valid triplet, arrays where multiple triplets sum to the target, and arrays with fewer than three elements (returns empty array).

An alternative O(n³) approach would check all triplets, but the two-pointer approach is more efficient. Another approach would use a hash map, but handling the ordering requirement becomes more complex. The sorting + two-pointer approach is optimal for this problem.

Think of this as fixing the first number and then solving the two-sum problem for the remaining numbers with the adjusted target (targetSum - first number).`,
	},
	{
		ID:          89,
		Title:       "Spiral Traverse",
		Description: "Write a function that takes in an n x m two-dimensional array (that can be square-shaped when n == m) and returns a one-dimensional array of all the array's elements in spiral order. Spiral order starts at the top left corner of the two-dimensional array, goes to the right, and proceeds in a spiral pattern all the way until every element has been visited.",
		Difficulty:  "Medium",
		Topic:       "Arrays",
		Signature:   "func spiralTraverse(array [][]int) []int",
		TestCases: []TestCase{
			{Input: "array = [[1, 2, 3, 4], [12, 13, 14, 5], [11, 16, 15, 6], [10, 9, 8, 7]]", Expected: "[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16]"},
			{Input: "array = [[1]]", Expected: "[1]"},
			{Input: "array = [[1, 2], [4, 3]]", Expected: "[1 2 3 4]"},
			{Input: "array = [[1, 2, 3], [8, 9, 4], [7, 6, 5]]", Expected: "[1 2 3 4 5 6 7 8 9]"},
		},
		Solution: `func spiralTraverse(array [][]int) []int {
	if len(array) == 0 {
		return []int{}
	}
	result := []int{}
	startRow, endRow := 0, len(array)-1
	startCol, endCol := 0, len(array[0])-1
	for startRow <= endRow && startCol <= endCol {
		for col := startCol; col <= endCol; col++ {
			result = append(result, array[startRow][col])
		}
		for row := startRow + 1; row <= endRow; row++ {
			result = append(result, array[row][endCol])
		}
		for col := endCol - 1; col >= startCol; col-- {
			if startRow == endRow {
				break
			}
			result = append(result, array[endRow][col])
		}
		for row := endRow - 1; row > startRow; row-- {
			if startCol == endCol {
				break
			}
			result = append(result, array[row][startCol])
		}
		startRow++
		endRow--
		startCol++
		endCol--
	}
	return result
}`,
		PythonSolution: `def spiralTraverse(array: List[List[int]]) -> List[int]:
    if not array:
        return []
    result = []
    start_row, end_row = 0, len(array) - 1
    start_col, end_col = 0, len(array[0]) - 1
    while start_row <= end_row and start_col <= end_col:
        for col in range(start_col, end_col + 1):
            result.append(array[start_row][col])
        for row in range(start_row + 1, end_row + 1):
            result.append(array[row][end_col])
        for col in range(end_col - 1, start_col - 1, -1):
            if start_row == end_row:
                break
            result.append(array[end_row][col])
        for row in range(end_row - 1, start_row, -1):
            if start_col == end_col:
                break
            result.append(array[row][start_col])
        start_row += 1
        end_row -= 1
        start_col += 1
        end_col -= 1
    return result`,
		Explanation: `Traversing a 2D array in spiral order requires systematically moving in four directions while maintaining boundaries that define the current layer. The key challenge is handling edge cases where the remaining area is a single row or column.

The algorithm maintains four boundaries (startRow, endRow, startCol, endCol) that define the current rectangular layer. It processes each layer by moving right along the top row, down along the right column, left along the bottom row (with a check to avoid re-processing the top row if it's the same), and up along the left column (with a check to avoid re-processing the right column if it's the same). After completing a layer, it shrinks all boundaries inward.

The edge case checks (startRow == endRow and startCol == endCol) are crucial for avoiding duplicate processing when the remaining area is a single row or column. These checks prevent going back over already-processed elements.

Edge cases handled include: single-row arrays (only right traversal, skip left and up), single-column arrays (only down traversal, skip left and up), square arrays (complete spiral), rectangular arrays (handled correctly by boundary checks), and single-cell arrays (returns that cell).

An alternative approach would use direction vectors and visited markers, but that requires O(n*m) extra space. The boundary approach achieves O(1) extra space (excluding output array) by using boundaries to track progress. The time complexity is O(n*m) as we visit each cell exactly once.

Think of this as peeling an onion layer by layer: you go around the outer edge in four directions, then move inward and repeat, with special handling when only one row or column remains.`,
	},
	{
		ID:          90,
		Title:       "Merge Overlapping Intervals",
		Description: "Write a function that takes in a non-empty array of arbitrary intervals, merges any overlapping intervals, and returns the new intervals in no particular order. Each interval is an array of two integers, with interval[0] as the start of the interval and interval[1] as the end of the interval.",
		Difficulty:  "Medium",
		Topic:       "Arrays",
		Signature:   "func mergeOverlappingIntervals(intervals [][]int) [][]int",
		TestCases: []TestCase{
			{Input: "intervals = [[1, 2], [3, 5], [4, 7], [6, 8], [9, 10]]", Expected: "[[1 2] [3 8] [9 10]]"},
			{Input: "intervals = [[1, 3], [2, 8], [9, 10]]", Expected: "[[1 8] [9 10]]"},
			{Input: "intervals = [[1, 10], [10, 20], [20, 30], [30, 40], [40, 50], [1, 100]]", Expected: "[[1 100]]"},
			{Input: "intervals = [[1, 10], [11, 20], [21, 30]]", Expected: "[[1 10] [11 20] [21 30]]"},
		},
		Solution: `import "sort"

func mergeOverlappingIntervals(intervals [][]int) [][]int {
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	mergedIntervals := [][]int{}
	currentInterval := intervals[0]
	mergedIntervals = append(mergedIntervals, currentInterval)
	for _, nextInterval := range intervals[1:] {
		currentIntervalEnd := currentInterval[1]
		nextIntervalStart, nextIntervalEnd := nextInterval[0], nextInterval[1]
		if currentIntervalEnd >= nextIntervalStart {
			currentInterval[1] = max(currentIntervalEnd, nextIntervalEnd)
		} else {
			currentInterval = nextInterval
			mergedIntervals = append(mergedIntervals, currentInterval)
		}
	}
	return mergedIntervals
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def mergeOverlappingIntervals(intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x: x[0])
    merged_intervals = []
    current_interval = intervals[0]
    merged_intervals.append(current_interval)
    for next_interval in intervals[1:]:
        current_interval_end = current_interval[1]
        next_interval_start, next_interval_end = next_interval
        if current_interval_end >= next_interval_start:
            current_interval[1] = max(current_interval_end, next_interval_end)
        else:
            current_interval = next_interval
            merged_intervals.append(current_interval)
    return merged_intervals`,
		Explanation: `Merging overlapping intervals requires identifying which intervals can be combined. The key insight is that after sorting intervals by their start time, we only need to check if the current interval overlaps with the next one, as all previous intervals have already been processed.

The algorithm first sorts all intervals by their start time. Then, it iterates through the sorted intervals, maintaining a "current interval" that represents the merged interval being built. For each next interval, if it overlaps with the current interval (next start <= current end), we merge them by extending the current interval's end to the maximum of both ends. If there's no overlap, we finalize the current interval and start a new one with the next interval.

This approach works because sorting ensures that if an interval doesn't overlap with the current one, no later interval will overlap with the current one either (since they all start later). This allows us to process intervals in a single pass after sorting.

Edge cases handled include: intervals that are completely contained within others (merged correctly), intervals that touch at endpoints (considered overlapping if current end >= next start), intervals with no overlaps (each becomes its own merged interval), and single interval arrays (returns that interval).

An alternative approach would check all pairs of intervals for overlaps, but that would be O(n²) time. The sorting approach is more efficient, achieving O(n log n) time for sorting and O(n) time for merging, with O(n) space for the result array.

Think of this as consolidating meeting times: you sort them by start time, then merge any meetings that overlap, extending the end time to cover both.`,
	},
	{
		ID:          91,
		Title:       "Smallest Difference",
		Description: "Write a function that takes in two non-empty arrays of integers, finds the pair of numbers (one from each array) whose absolute difference is closest to zero, and returns an array containing these two numbers, with the number from the first array in the first position.",
		Difficulty:  "Medium",
		Topic:       "Arrays",
		Signature:   "func smallestDifference(arrayOne []int, arrayTwo []int) []int",
		TestCases: []TestCase{
			{Input: "arrayOne = []int{-1, 5, 10, 20, 28, 3}, arrayTwo = []int{26, 134, 135, 15, 17}", Expected: "[28 26]"},
			{Input: "arrayOne = []int{-1, 5, 10, 20, 3}, arrayTwo = []int{26, 134, 135, 15, 17}", Expected: "[20 17]"},
			{Input: "arrayOne = []int{10, 0, 20, 25}, arrayTwo = []int{1005, 1006, 1014, 1032, 1031}", Expected: "[25 1005]"},
			{Input: "arrayOne = []int{10, 0, 20, 25, 2200}, arrayTwo = []int{1005, 1006, 1014, 1032, 1031}", Expected: "[25 1005]"},
		},
		Solution: `import (
	"math"
	"sort"
)

func smallestDifference(arrayOne []int, arrayTwo []int) []int {
	sort.Ints(arrayOne)
	sort.Ints(arrayTwo)
	idxOne, idxTwo := 0, 0
	smallest := math.MaxInt32
	current := math.MaxInt32
	smallestPair := []int{}
	for idxOne < len(arrayOne) && idxTwo < len(arrayTwo) {
		firstNum := arrayOne[idxOne]
		secondNum := arrayTwo[idxTwo]
		if firstNum < secondNum {
			current = secondNum - firstNum
			idxOne++
		} else if secondNum < firstNum {
			current = firstNum - secondNum
			idxTwo++
		} else {
			return []int{firstNum, secondNum}
		}
		if smallest > current {
			smallest = current
			smallestPair = []int{firstNum, secondNum}
		}
	}
	return smallestPair
}`,
		PythonSolution: `def smallestDifference(arrayOne: List[int], arrayTwo: List[int]) -> List[int]:
    arrayOne.sort()
    arrayTwo.sort()
    idx_one = idx_two = 0
    smallest = float('inf')
    current = float('inf')
    smallest_pair = []
    while idx_one < len(arrayOne) and idx_two < len(arrayTwo):
        first_num = arrayOne[idx_one]
        second_num = arrayTwo[idx_two]
        if first_num < second_num:
            current = second_num - first_num
            idx_one += 1
        elif second_num < first_num:
            current = first_num - second_num
            idx_two += 1
        else:
            return [first_num, second_num]
        if smallest > current:
            smallest = current
            smallest_pair = [first_num, second_num]
    return smallest_pair`,
		Explanation: `Finding the pair of numbers from two arrays with the smallest absolute difference requires efficiently comparing elements from both arrays. The key insight is that after sorting both arrays, we can use two pointers to find the closest pair without checking all combinations.

The algorithm first sorts both arrays. Then, it uses two pointers, one for each array, starting at the beginning. At each step, it calculates the absolute difference between the elements at the two pointers. If the first array's element is smaller, moving its pointer right will decrease the difference (or potentially find a better pair), so we move that pointer. Similarly, if the second array's element is smaller, we move its pointer. If the elements are equal, we've found the optimal pair (difference is 0) and can return immediately.

This two-pointer approach works because after sorting, we can make greedy decisions about which pointer to move. Moving the pointer with the smaller value is the only way to potentially find a smaller difference, as moving the other pointer would only increase the difference.

Edge cases handled include: arrays with equal elements (returns immediately with difference 0), arrays where the smallest difference is at the beginning, arrays where the smallest difference is at the end, and arrays with negative numbers (absolute difference handles this correctly).

An alternative O(n*m) approach would check all pairs, but the two-pointer approach is more efficient. The time complexity is O(n log n + m log m) for sorting plus O(n + m) for the two-pointer traversal, giving overall O(n log n + m log m). The space complexity is O(1) excluding the input arrays.

Think of this as having two sorted lists and trying to find the closest pair: you compare the current elements, move the pointer pointing to the smaller value (to potentially get closer), and keep track of the best pair you've seen.`,
	},
	{
		ID:          92,
		Title:       "Move Element To End",
		Description: "You're given an array of integers and an integer. Write a function that moves all instances of that integer in the array to the end of the array and returns the array. The function should perform this in place (i.e., it should mutate the input array) and doesn't need to maintain the order of the other integers.",
		Difficulty:  "Medium",
		Topic:       "Arrays",
		Signature:   "func moveElementToEnd(array []int, toMove int) []int",
		TestCases: []TestCase{
			{Input: "array = []int{2, 1, 2, 2, 2, 3, 4, 2}, toMove = 2", Expected: "[1 3 4 2 2 2 2 2]"},
			{Input: "array = []int{1, 2, 4, 5, 6}, toMove = 3", Expected: "[1 2 4 5 6]"},
			{Input: "array = []int{3, 1, 2, 4, 5}, toMove = 3", Expected: "[1 2 4 5 3]"},
			{Input: "array = []int{1, 2, 3, 4, 5}, toMove = 3", Expected: "[1 2 4 5 3]"},
		},
		Solution: `func moveElementToEnd(array []int, toMove int) []int {
	leftIdx := 0
	rightIdx := len(array) - 1
	for leftIdx < rightIdx {
		for leftIdx < rightIdx && array[rightIdx] == toMove {
			rightIdx--
		}
		if array[leftIdx] == toMove {
			array[leftIdx], array[rightIdx] = array[rightIdx], array[leftIdx]
		}
		leftIdx++
	}
	return array
}`,
		PythonSolution: `def moveElementToEnd(array: List[int], toMove: int) -> List[int]:
    left_idx = 0
    right_idx = len(array) - 1
    while left_idx < right_idx:
        while left_idx < right_idx and array[right_idx] == toMove:
            right_idx -= 1
        if array[left_idx] == toMove:
            array[left_idx], array[right_idx] = array[right_idx], array[left_idx]
        left_idx += 1
    return array`,
		Explanation: `Moving all instances of a specific element to the end of an array in-place requires careful pointer management. The two-pointer technique from both ends efficiently handles this by swapping elements as needed.

The algorithm uses two pointers starting from opposite ends. The right pointer moves leftward, skipping over elements that are already the target value (since they're already at the end where we want them). The left pointer moves rightward. When the left pointer finds the target value, it swaps with whatever the right pointer is pointing to (which is guaranteed not to be the target value after skipping). This effectively moves target values to the end.

The key optimization is skipping target values at the right end: before checking the left pointer, we move the right pointer past any target values, ensuring we only swap when necessary. This prevents unnecessary swaps and ensures target values accumulate at the end.

Edge cases handled include: arrays with no target values (no swaps, array unchanged), arrays with all target values (all already at end, minimal swaps), arrays where target values are already at the end (right pointer skips them efficiently), and arrays with target values only at the beginning (all moved to end).

An alternative approach would use a write pointer to collect non-target elements at the front, then fill the rest with target values, but the two-pointer swap approach is more intuitive. Both approaches achieve O(n) time with O(1) space.

Think of this as organizing items: you have items you want at the end (target values) and items you want to keep (other values). You swap items from the front that should be at the end with items from the back that should stay, working your way inward until you meet.`,
	},
	{
		ID:          93,
		Title:       "Monotonic Array",
		Description: "Write a function that takes in an array of integers and returns a boolean representing whether the array is monotonic. An array is said to be monotonic if its elements, from left to right, are entirely non-increasing or entirely non-decreasing.",
		Difficulty:  "Medium",
		Topic:       "Arrays",
		Signature:   "func isMonotonic(array []int) bool",
		TestCases: []TestCase{
			{Input: "array = []int{-1, -5, -10, -1100, -1100, -1101, -1102, -9001}", Expected: "true"},
			{Input: "array = []int{1, 1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 7, 9, 10, 11}", Expected: "false"},
			{Input: "array = []int{1, 1, 1, 1, 1, 1, 1, 1}", Expected: "true"},
			{Input: "array = []int{1, 2}", Expected: "true"},
			{Input: "array = []int{2, 1}", Expected: "true"},
		},
		Solution: `func isMonotonic(array []int) bool {
	if len(array) <= 2 {
		return true
	}
	direction := array[1] - array[0]
	for i := 2; i < len(array); i++ {
		if direction == 0 {
			direction = array[i] - array[i-1]
			continue
		}
		if breaksDirection(direction, array[i-1], array[i]) {
			return false
		}
	}
	return true
}

func breaksDirection(direction int, previous int, current int) bool {
	difference := current - previous
	if direction > 0 {
		return difference < 0
	}
	return difference > 0
}`,
		PythonSolution: `def isMonotonic(array: List[int]) -> bool:
    if len(array) <= 2:
        return True
    direction = array[1] - array[0]
    for i in range(2, len(array)):
        if direction == 0:
            direction = array[i] - array[i - 1]
            continue
        if breaks_direction(direction, array[i - 1], array[i]):
            return False
    return True

def breaks_direction(direction, previous, current):
    difference = current - previous
    if direction > 0:
        return difference < 0
    return difference > 0`,
		Explanation: `Checking if an array is monotonic requires determining whether it's entirely non-increasing or entirely non-decreasing. The key insight is to determine the direction from the first two distinct elements, then verify that all subsequent elements maintain that direction.

The algorithm first handles edge cases (arrays with 0, 1, or 2 elements are always monotonic). Then, it determines the direction by comparing the first two elements. If they're equal, it continues until it finds two elements with a non-zero difference. Once the direction is established (positive for increasing, negative for decreasing), it checks each subsequent pair to ensure they don't break the direction.

The helper function breaksDirection checks if a pair of elements violates the established direction. For an increasing direction (direction > 0), any decrease breaks it. For a decreasing direction (direction < 0), any increase breaks it. Equal elements don't break either direction, which is why we handle them specially when establishing the initial direction.

Edge cases handled include: arrays with all equal elements (monotonic, both increasing and decreasing), arrays that start with equal elements (direction determined from first non-equal pair), arrays with exactly two elements (always monotonic), and arrays that change direction (not monotonic, returns false).

An alternative approach would check both increasing and decreasing conditions separately, but the single-pass direction-tracking approach is more efficient. The time complexity is O(n) as we visit each element once, and space complexity is O(1).

Think of this as checking if a sequence is consistently going in one direction: you determine the direction from the first meaningful comparison, then verify that all subsequent comparisons maintain that direction.`,
	},
	{
		ID:          94,
		Title:       "Longest Peak",
		Description: "Write a function that takes in an array of integers and returns the length of the longest peak in the array. A peak is defined as adjacent integers in the array that are strictly increasing until they reach a tip (the highest value in the peak), at which point they become strictly decreasing. At least three integers are required to form a peak.",
		Difficulty:  "Medium",
		Topic:       "Arrays",
		Signature:   "func longestPeak(array []int) int",
		TestCases: []TestCase{
			{Input: "array = []int{1, 2, 3, 3, 4, 0, 10, 6, 5, -1, -3, 2, 3}", Expected: "6"},
			{Input: "array = []int{1, 3, 2}", Expected: "3"},
			{Input: "array = []int{1, 2, 3, 4, 5, 1}", Expected: "6"},
			{Input: "array = []int{5, 4, 3, 2, 1, 2, 1}", Expected: "3"},
		},
		Solution: `func longestPeak(array []int) int {
	longestPeakLength := 0
	i := 1
	for i < len(array)-1 {
		isPeak := array[i-1] < array[i] && array[i] > array[i+1]
		if !isPeak {
			i++
			continue
		}
		leftIdx := i - 2
		for leftIdx >= 0 && array[leftIdx] < array[leftIdx+1] {
			leftIdx--
		}
		rightIdx := i + 2
		for rightIdx < len(array) && array[rightIdx] < array[rightIdx-1] {
			rightIdx++
		}
		currentPeakLength := rightIdx - leftIdx - 1
		if currentPeakLength > longestPeakLength {
			longestPeakLength = currentPeakLength
		}
		i = rightIdx
	}
	return longestPeakLength
}`,
		PythonSolution: `def longestPeak(array: List[int]) -> int:
    longest_peak_length = 0
    i = 1
    while i < len(array) - 1:
        is_peak = array[i - 1] < array[i] and array[i] > array[i + 1]
        if not is_peak:
            i += 1
            continue
        left_idx = i - 2
        while left_idx >= 0 and array[left_idx] < array[left_idx + 1]:
            left_idx -= 1
        right_idx = i + 2
        while right_idx < len(array) and array[right_idx] < array[right_idx - 1]:
            right_idx += 1
        current_peak_length = right_idx - left_idx - 1
        if current_peak_length > longest_peak_length:
            longest_peak_length = current_peak_length
        i = right_idx
    return longest_peak_length`,
		Explanation: `Finding the longest peak requires identifying local maxima (peaks) and then expanding left and right from each peak to determine the full peak length. A peak is defined as a value that is greater than both its neighbors, with strictly increasing elements to the left and strictly decreasing elements to the right.

The algorithm iterates through the array, checking each position (except the first and last) to see if it's a peak (array[i-1] < array[i] > array[i+1]). When a peak is found, it expands leftward to find where the increasing sequence starts, and rightward to find where the decreasing sequence ends. The peak length is the distance between these endpoints.

The key optimization is that after processing a peak, we can jump the iterator to the right end of that peak, since we've already processed all positions within that peak. This ensures each element is visited at most a constant number of times, giving O(n) time complexity overall.

Edge cases handled include: arrays with no peaks (returns 0), arrays with multiple peaks (finds the longest), peaks at the beginning or end of the array (handled by boundary checks), and arrays where peaks share elements (each peak processed independently).

An alternative approach would check all possible subarrays, but that would be O(n³) time. The peak-finding and expansion approach is optimal, achieving O(n) time with O(1) space.

Think of this as finding mountain peaks: you look for points that are higher than their neighbors, then measure how far the mountain extends on both sides before the terrain starts going up again or levels off.`,
	},
	{
		ID:          95,
		Title:       "Array Of Products",
		Description: "Write a function that takes in a non-empty array of integers and returns an array of the same length, where each element in the output array is equal to the product of every other number in the input array. In other words, the value at output[i] is equal to the product of every number in the input array other than input[i]. Note that you're expected to solve this problem without using division.",
		Difficulty:  "Medium",
		Topic:       "Arrays",
		Signature:   "func arrayOfProducts(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = []int{5, 1, 4, 2}", Expected: "[8 40 10 20]"},
			{Input: "array = []int{1, 8, 6, 2, 4}", Expected: "[384 48 64 192 96]"},
			{Input: "array = []int{-5, 2, -4, 14, -6}", Expected: "[672 -1680 840 -240 560]"},
			{Input: "array = []int{9, 3, 2, 1, 9, 5, 3, 2}", Expected: "[1620 4860 7290 14580 1620 2916 4860 7290]"},
		},
		Solution: `func arrayOfProducts(array []int) []int {
	products := make([]int, len(array))
	leftProducts := make([]int, len(array))
	rightProducts := make([]int, len(array))
	leftRunningProduct := 1
	for i := 0; i < len(array); i++ {
		leftProducts[i] = leftRunningProduct
		leftRunningProduct *= array[i]
	}
	rightRunningProduct := 1
	for i := len(array) - 1; i >= 0; i-- {
		rightProducts[i] = rightRunningProduct
		rightRunningProduct *= array[i]
	}
	for i := 0; i < len(array); i++ {
		products[i] = leftProducts[i] * rightProducts[i]
	}
	return products
}`,
		PythonSolution: `def arrayOfProducts(array: List[int]) -> List[int]:
    products = [0] * len(array)
    left_products = [0] * len(array)
    right_products = [0] * len(array)
    left_running_product = 1
    for i in range(len(array)):
        left_products[i] = left_running_product
        left_running_product *= array[i]
    right_running_product = 1
    for i in range(len(array) - 1, -1, -1):
        right_products[i] = right_running_product
        right_running_product *= array[i]
    for i in range(len(array)):
        products[i] = left_products[i] * right_products[i]
    return products`,
		Explanation: `Computing the product of all elements except the current element without using division requires calculating prefix and suffix products separately. The key insight is that the product for position i equals the product of all elements to the left of i multiplied by the product of all elements to the right of i.

The algorithm makes three passes through the array. First, it calculates left products: for each position i, leftProducts[i] contains the product of all elements from index 0 to i-1. Then, it calculates right products: for each position i, rightProducts[i] contains the product of all elements from index i+1 to the end. Finally, it multiplies corresponding left and right products to get the final result.

This approach avoids division, which is important because the array might contain zeros (division by zero) or the problem might prohibit division. By separating the calculation into left and right products, we can compute the result efficiently.

Edge cases handled include: arrays with zeros (product becomes 0 for all positions except the zero position), arrays with negative numbers (signs are preserved correctly), arrays with a single zero (all other positions are 0, zero position gets product of others), and arrays with all positive numbers (standard case).

An alternative approach would calculate the total product and divide by each element, but that fails with zeros and may not be allowed. Another approach would use O(1) space by calculating right products on-the-fly in the final pass, but the current approach with separate arrays is clearer and still efficient.

Think of this as calculating "what would the product be if I removed this element": you multiply everything to the left with everything to the right, which gives you the product of all elements except the current one.`,
	},
	{
		ID:          96,
		Title:       "First Duplicate Value",
		Description: "Given an array of integers between 1 and n, inclusive, where n is the length of the array, write a function that returns the first integer that appears more than once (when the array is read from left to right). In other words, out of all the integers that might occur more than once in the input array, your function should return the one whose first duplicate value has the minimum index. If no integer appears more than once, your function should return -1.",
		Difficulty:  "Medium",
		Topic:       "Arrays",
		Signature:   "func firstDuplicateValue(array []int) int",
		TestCases: []TestCase{
			{Input: "array = []int{2, 1, 5, 2, 3, 3, 4}", Expected: "2"},
			{Input: "array = []int{2, 1, 5, 3, 3, 2, 4}", Expected: "3"},
			{Input: "array = []int{1, 1, 2, 3, 3, 2, 2}", Expected: "1"},
			{Input: "array = []int{3, 1, 3, 1, 1, 4, 4}", Expected: "3"},
			{Input: "array = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}", Expected: "-1"},
		},
		Solution: `func firstDuplicateValue(array []int) int {
	for _, value := range array {
		absValue := abs(value)
		if array[absValue-1] < 0 {
			return absValue
		}
		array[absValue-1] *= -1
	}
	return -1
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}`,
		PythonSolution: `def firstDuplicateValue(array: List[int]) -> int:
    for value in array:
        abs_value = abs(value)
        if array[abs_value - 1] < 0:
            return abs_value
        array[abs_value - 1] *= -1
    return -1`,
		Explanation: `Finding the first duplicate value efficiently requires tracking which values we've seen. The key insight is that since values are between 1 and n (where n is array length), we can use the array itself as a hash set by marking visited positions as negative.

The algorithm iterates through the array. For each value, it calculates its absolute value (to handle cases where we've already marked it negative) and uses it as an index (value - 1, since values are 1-indexed but arrays are 0-indexed). If the element at that index is already negative, it means we've seen this value before, so we return it as the first duplicate. Otherwise, we mark that position as visited by making it negative.

This approach achieves O(1) space complexity by reusing the input array for storage, rather than using a separate hash set. The time complexity is O(n) as we visit each element once.

Edge cases handled include: arrays with no duplicates (returns -1), arrays where the first duplicate appears early, arrays where multiple values have duplicates (returns the one whose first duplicate appears earliest), and arrays where values are already negative (handled by taking absolute value).

An alternative approach would use a hash set to track seen values, but that requires O(n) extra space. The negative marking approach achieves O(1) extra space by using the array itself. However, this modifies the input array, which may or may not be acceptable depending on requirements.

Think of this as using the array as a checklist: when you see a number, you check off its position by making it negative. If you try to check off a position that's already checked (negative), you've found a duplicate.`,
	},
	{
		ID:          97,
		Title:       "Missing Numbers",
		Description: "You're given an unsorted array containing n numbers taken from the range 1 to n+1, where n is the length of the array. The array has exactly one missing number. Write a function that returns the missing number.",
		Difficulty:  "Medium",
		Topic:       "Arrays",
		Signature:   "func missingNumbers(nums []int) []int",
		TestCases: []TestCase{
			{Input: "nums = []int{4, 5, 1, 3}", Expected: "[2]"},
			{Input: "nums = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100}", Expected: "[]"},
		},
		Solution: `func missingNumbers(nums []int) []int {
	n := len(nums) + 2
	missing := []int{}
	present := make(map[int]bool)
	for _, num := range nums {
		present[num] = true
	}
	for i := 1; i <= n; i++ {
		if !present[i] {
			missing = append(missing, i)
		}
	}
	return missing
}`,
		PythonSolution: `def missingNumbers(nums: List[int]) -> List[int]:
    n = len(nums) + 2
    missing = []
    present = set(nums)
    for i in range(1, n + 1):
        if i not in present:
            missing.append(i)
    return missing`,
		Explanation: `Finding missing numbers in an array containing numbers from 1 to n+2 (where n is array length) requires efficiently identifying which numbers are absent. The hash set data structure provides O(1) lookup to check if a number is present.

The algorithm first creates a hash set containing all numbers present in the input array. Then, it iterates through all numbers from 1 to n+2 (the expected range), checking if each number exists in the set. Numbers that don't exist in the set are added to the result array.

This approach is straightforward and efficient. The hash set creation takes O(n) time, and checking each number in the range takes O(n+2) time, giving overall O(n) time complexity. The space complexity is O(n) for the hash set.

Edge cases handled include: arrays with no missing numbers (returns empty array), arrays with one missing number (returns that number), arrays with multiple missing numbers (returns all missing numbers), and arrays where the missing number is 1 or n+2 (handled correctly by checking the full range).

An alternative approach would sort the array and check for gaps, but that requires O(n log n) time. Another approach would use the sum formula, but that only works for finding a single missing number. The hash set approach works for any number of missing numbers and is more flexible.

Think of this as having a checklist of numbers from 1 to n+2: you mark off which numbers you have, then go through the checklist and note which ones aren't marked.`,
	},
	{
		ID:          98,
		Title:       "BST Construction",
		Description: "Write a BST class for a Binary Search Tree. The class should support: Inserting values with the insert method, Removing values with the remove method (should only remove the first instance of a given value), Searching for values with the contains method.",
		Difficulty:  "Medium",
		Topic:       "Binary Search Trees",
		Signature:   "type BST struct { Value int; Left *BST; Right *BST }\nfunc (tree *BST) Insert(value int) *BST\nfunc (tree *BST) Contains(value int) bool\nfunc (tree *BST) Remove(value int) *BST",
		TestCases: []TestCase{
			{Input: "BST operations", Expected: "Correct BST structure"},
		},
		Solution: `type BST struct {
	Value int
	Left  *BST
	Right *BST
}

func (tree *BST) Insert(value int) *BST {
	if value < tree.Value {
		if tree.Left == nil {
			tree.Left = &BST{Value: value}
		} else {
			tree.Left.Insert(value)
		}
	} else {
		if tree.Right == nil {
			tree.Right = &BST{Value: value}
		} else {
			tree.Right.Insert(value)
		}
	}
	return tree
}

func (tree *BST) Contains(value int) bool {
	if value < tree.Value {
		if tree.Left == nil {
			return false
		}
		return tree.Left.Contains(value)
	} else if value > tree.Value {
		if tree.Right == nil {
			return false
		}
		return tree.Right.Contains(value)
	}
	return true
}

func (tree *BST) Remove(value int) *BST {
	tree.remove(value, nil)
	return tree
}

func (tree *BST) remove(value int, parent *BST) {
	if value < tree.Value {
		if tree.Left != nil {
			tree.Left.remove(value, tree)
		}
	} else if value > tree.Value {
		if tree.Right != nil {
			tree.Right.remove(value, tree)
		}
	} else {
		if tree.Left != nil && tree.Right != nil {
			tree.Value = tree.Right.getMinValue()
			tree.Right.remove(tree.Value, tree)
		} else if parent == nil {
			if tree.Left != nil {
				tree.Value = tree.Left.Value
				tree.Right = tree.Left.Right
				tree.Left = tree.Left.Left
			} else if tree.Right != nil {
				tree.Value = tree.Right.Value
				tree.Left = tree.Right.Left
				tree.Right = tree.Right.Right
			}
		} else if parent.Left == tree {
			if tree.Left != nil {
				parent.Left = tree.Left
			} else {
				parent.Left = tree.Right
			}
		} else if parent.Right == tree {
			if tree.Left != nil {
				parent.Right = tree.Left
			} else {
				parent.Right = tree.Right
			}
		}
	}
}

func (tree *BST) getMinValue() int {
	if tree.Left == nil {
		return tree.Value
	}
	return tree.Left.getMinValue()
}`,
		PythonSolution: `class BST:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    
    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BST(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BST(value)
            else:
                self.right.insert(value)
        return self
    
    def contains(self, value):
        if value < self.value:
            if self.left is None:
                return False
            return self.left.contains(value)
        elif value > self.value:
            if self.right is None:
                return False
            return self.right.contains(value)
        return True
    
    def remove(self, value, parent=None):
        if value < self.value:
            if self.left is not None:
                self.left.remove(value, self)
        elif value > self.value:
            if self.right is not None:
                self.right.remove(value, self)
        else:
            if self.left is not None and self.right is not None:
                self.value = self.right.get_min_value()
                self.right.remove(self.value, self)
            elif parent is None:
                if self.left is not None:
                    self.value = self.left.value
                    self.right = self.left.right
                    self.left = self.left.left
                elif self.right is not None:
                    self.value = self.right.value
                    self.left = self.right.left
                    self.right = self.right.right
            elif parent.left == self:
                parent.left = self.left if self.left is not None else self.right
            elif parent.right == self:
                parent.right = self.left if self.left is not None else self.right
        return self
    
    def get_min_value(self):
        if self.left is None:
            return self.value
        return self.left.get_min_value()`,
		Explanation: `Implementing a Binary Search Tree requires three core operations: insertion, search, and deletion. Each operation leverages the BST property: values less than the current node go left, values greater than or equal go right.

For insertion, we traverse the tree following the BST property until we find an empty spot (nil child), then create a new node there. The Contains method performs a binary search: compare the target value with the current node, and recursively search the left subtree if smaller or right subtree if larger. If we reach nil, the value doesn't exist.

The Remove operation is the most complex. First, we find the node to remove. If it has two children, we replace its value with the minimum value from its right subtree (the inorder successor), then recursively remove that minimum node. If it has one or zero children, we replace it with its single child (or nil if no children). Special handling is needed when removing the root node without a parent.

The time complexity is O(log n) on average for balanced trees, but O(n) in the worst case (degenerate tree). Space complexity is O(1) for the operations themselves, but O(h) for the recursion stack where h is the tree height.

Edge cases handled include: inserting duplicate values (goes to right subtree per BST property), removing the root node (handled with parent == nil check), removing nodes with zero, one, or two children (each case handled appropriately), and searching in empty trees (returns false).

An alternative iterative approach would avoid recursion overhead, but the recursive approach is cleaner and easier to understand. For production use, self-balancing trees (AVL, Red-Black) would ensure O(log n) worst-case performance.

Think of a BST as an organized filing system: smaller items go left, larger items go right. To find something, you follow the path based on comparisons. To add something, you find the right spot and place it there. To remove something, you replace it with its logical successor or simply remove it if it's a leaf.`,
	},
	{
		ID:          99,
		Title:       "Validate BST",
		Description: "Write a function that takes in a potentially invalid Binary Search Tree (BST) and returns a boolean representing whether the BST is valid. Each BST node has an integer value, a left child node, and a right child node. A node is said to be a valid BST node if and only if it satisfies the BST property: its value is strictly greater than the values of every node to its left; its value is less than or equal to the values of every node to its right; and its children nodes are either valid BST nodes themselves or None / null.",
		Difficulty:  "Medium",
		Topic:       "Binary Search Trees",
		Signature:   "func validateBst(tree *TreeNode) bool",
		TestCases: []TestCase{
			{Input: "tree = [10, 5, 15, 2, 5, 13, 22, 1, null, null, null, null, 14]", Expected: "true"},
			{Input: "tree = [10, 5, 15, 2, 5, 13, 22, 1, null, null, null, null, 14, null, null]", Expected: "true"},
			{Input: "tree = [5, 1, 4, null, null, 3, 6]", Expected: "false"},
		},
		Solution: `func validateBst(tree *TreeNode) bool {
	return validateBstHelper(tree, math.MinInt32, math.MaxInt32)
}

func validateBstHelper(tree *TreeNode, minValue int, maxValue int) bool {
	if tree == nil {
		return true
	}
	if tree.Val < minValue || tree.Val >= maxValue {
		return false
	}
	leftIsValid := validateBstHelper(tree.Left, minValue, tree.Val)
	return leftIsValid && validateBstHelper(tree.Right, tree.Val, maxValue)
}`,
		PythonSolution: `def validateBst(tree: Optional[TreeNode]) -> bool:
    def validate_bst_helper(node, min_value, max_value):
        if node is None:
            return True
        if node.val < min_value or node.val >= max_value:
            return False
        left_is_valid = validate_bst_helper(node.left, min_value, node.val)
        return left_is_valid and validate_bst_helper(node.right, node.val, max_value)
    return validate_bst_helper(tree, float('-inf'), float('inf'))`,
		Explanation: `Validating a Binary Search Tree requires checking that every node satisfies the BST property: all nodes in the left subtree are strictly less than the current node, and all nodes in the right subtree are greater than or equal to the current node. The key insight is that we need to track valid ranges (min, max) for each node as we traverse.

The algorithm uses a recursive helper function that takes a node and valid min/max bounds. For each node, we check if its value is within the bounds. If not, the tree is invalid. Then, we recursively validate the left subtree with updated bounds (min stays the same, max becomes the current node's value) and the right subtree (min becomes the current node's value, max stays the same).

This approach ensures that not only does each node satisfy the BST property with its immediate children, but also that all descendants satisfy the global BST property. The bounds propagate down the tree, becoming tighter as we go deeper.

Edge cases handled include: empty trees (valid BST), single-node trees (valid BST), trees where a node violates bounds due to a distant ancestor (caught by bound checking), and trees with duplicate values (right subtree allows >=, left subtree requires <).

An alternative approach would do an in-order traversal and check if the sequence is strictly increasing, but the bounds-checking approach is more direct and doesn't require storing the entire traversal. The time complexity is O(n) as we visit each node once, and space complexity is O(h) for the recursion stack where h is the tree height.

Think of this as checking if each node is within an allowed range that gets narrower as you go deeper: the root can be any value, but as you go left, the maximum allowed value decreases, and as you go right, the minimum allowed value increases.`,
	},
	{
		ID:          100,
		Title:       "BST Traversal",
		Description: "Write three functions that take in a Binary Search Tree (BST) and an empty array, traverse the BST, add its nodes' values to the input array, and return that array. The three functions should traverse the BST using the in-order, pre-order, and post-order tree-traversal techniques, respectively.",
		Difficulty:  "Medium",
		Topic:       "Binary Search Trees",
		Signature:   "func inOrderTraverse(tree *TreeNode, array []int) []int\nfunc preOrderTraverse(tree *TreeNode, array []int) []int\nfunc postOrderTraverse(tree *TreeNode, array []int) []int",
		TestCases: []TestCase{
			{Input: "tree = [10, 5, 15, 2, 5, 13, 22, 1, null, null, null, null, 14]", Expected: "In-order: [1, 2, 5, 5, 10, 13, 14, 15, 22]"},
		},
		Solution: `func inOrderTraverse(tree *TreeNode, array []int) []int {
	if tree != nil {
		array = inOrderTraverse(tree.Left, array)
		array = append(array, tree.Val)
		array = inOrderTraverse(tree.Right, array)
	}
	return array
}

func preOrderTraverse(tree *TreeNode, array []int) []int {
	if tree != nil {
		array = append(array, tree.Val)
		array = preOrderTraverse(tree.Left, array)
		array = preOrderTraverse(tree.Right, array)
	}
	return array
}

func postOrderTraverse(tree *TreeNode, array []int) []int {
	if tree != nil {
		array = postOrderTraverse(tree.Left, array)
		array = postOrderTraverse(tree.Right, array)
		array = append(array, tree.Val)
	}
	return array
}`,
		PythonSolution: `def inOrderTraverse(tree: Optional[TreeNode], array: List[int]) -> List[int]:
    if tree is not None:
        array = inOrderTraverse(tree.left, array)
        array.append(tree.val)
        array = inOrderTraverse(tree.right, array)
    return array

def preOrderTraverse(tree: Optional[TreeNode], array: List[int]) -> List[int]:
    if tree is not None:
        array.append(tree.val)
        array = preOrderTraverse(tree.left, array)
        array = preOrderTraverse(tree.right, array)
    return array

def postOrderTraverse(tree: Optional[TreeNode], array: List[int]) -> List[int]:
    if tree is not None:
        array = postOrderTraverse(tree.left, array)
        array = postOrderTraverse(tree.right, array)
        array.append(tree.val)
    return array`,
		Explanation: `Tree traversal algorithms visit all nodes in a tree, but the order in which nodes are visited differs based on when we process the root node relative to its children. The three standard traversal orders are in-order, pre-order, and post-order, each with different use cases.

In-order traversal processes nodes in the order: left subtree, root, right subtree. For a BST, this produces values in sorted order, making it useful for printing sorted values or validating BST properties. Pre-order traversal processes: root, left subtree, right subtree. This order is useful for creating a copy of the tree or serializing it, as the root comes first. Post-order traversal processes: left subtree, right subtree, root. This is useful for deleting a tree or evaluating expressions, as children are processed before their parent.

The implementation for all three is straightforward recursive: we check if the node is nil (base case), and if not, we recursively traverse subtrees and append values to the array in the appropriate order. The key difference is the position of the append statement relative to the recursive calls.

Edge cases handled include: empty trees (returns empty array), single-node trees (returns array with one element), and trees with only left or right children (traversal handles nil children gracefully).

An alternative iterative approach using a stack would avoid recursion overhead, but the recursive approach is simpler and more intuitive. The time complexity is O(n) as we visit each node exactly once, and space complexity is O(h) for the recursion stack where h is the tree height.

Think of tree traversal as visiting a house: pre-order is like entering a room, then exploring left and right rooms; in-order is like exploring the left room, then entering the current room, then the right room; post-order is like exploring left and right rooms first, then entering the current room.`,
	},
	{
		ID:          101,
		Title:       "Min Height BST",
		Description: "Write a function that takes in a non-empty sorted array of distinct integers, constructs a BST from the integers, and returns the root of the BST. The function should minimize the height of the BST.",
		Difficulty:  "Medium",
		Topic:       "Binary Search Trees",
		Signature:   "func minHeightBst(array []int) *TreeNode",
		TestCases: []TestCase{
			{Input: "array = [1, 2, 5, 7, 10, 13, 14, 15, 22]", Expected: "BST with min height"},
		},
		Solution: `func minHeightBst(array []int) *TreeNode {
	return constructMinHeightBst(array, 0, len(array)-1)
}

func constructMinHeightBst(array []int, startIdx int, endIdx int) *TreeNode {
	if endIdx < startIdx {
		return nil
	}
	midIdx := (startIdx + endIdx) / 2
	bst := &TreeNode{Val: array[midIdx]}
	bst.Left = constructMinHeightBst(array, startIdx, midIdx-1)
	bst.Right = constructMinHeightBst(array, midIdx+1, endIdx)
	return bst
}`,
		PythonSolution: `def minHeightBst(array: List[int]) -> Optional[TreeNode]:
    return construct_min_height_bst(array, 0, len(array) - 1)

def construct_min_height_bst(array, start_idx, end_idx):
    if end_idx < start_idx:
        return None
    mid_idx = (start_idx + end_idx) // 2
    bst = TreeNode(array[mid_idx])
    bst.left = construct_min_height_bst(array, start_idx, mid_idx - 1)
    bst.right = construct_min_height_bst(array, mid_idx + 1, end_idx)
    return bst`,
		Explanation: `Constructing a minimum-height BST from a sorted array requires choosing the root strategically. The key insight is that to minimize height, we should choose the middle element as the root, which ensures roughly equal numbers of elements in the left and right subtrees.

The algorithm uses a recursive divide-and-conquer approach. For each subarray, we select the middle element as the root node. Then, we recursively construct the left subtree from the left half of the array and the right subtree from the right half of the array. This ensures that the tree is balanced, with the height being approximately log(n).

By always choosing the middle element, we ensure that each recursive call processes roughly half the elements, leading to a balanced tree structure. The base case is when the start index exceeds the end index, indicating an empty subarray, which returns nil.

Edge cases handled include: arrays with a single element (returns a single-node tree), arrays with two elements (creates a tree with one child), and empty arrays (handled by the base case, though the problem states non-empty).

An alternative approach would insert elements one by one, but that could result in an unbalanced tree with O(n) height in the worst case. The middle-element approach guarantees O(log n) height. The time complexity is O(n) as we visit each element once, and space complexity is O(n) for the tree structure plus O(log n) for the recursion stack.

Think of this as building a balanced tree by always splitting the sorted array in half: the middle becomes the root, and we recursively build balanced subtrees from the left and right halves.`,
	},
	{
		ID:          102,
		Title:       "Find Kth Largest Value In BST",
		Description: "Write a function that takes in a Binary Search Tree (BST) and a positive integer k and returns the kth largest integer contained in the BST. You can assume that there will only be integer values in the BST and that k is less than or equal to the number of nodes in the tree.",
		Difficulty:  "Medium",
		Topic:       "Binary Search Trees",
		Signature:   "func findKthLargestValueInBst(tree *TreeNode, k int) int",
		TestCases: []TestCase{
			{Input: "tree = [15, 5, 20, 2, 5, 17, 22, 1, 3], k = 3", Expected: "17"},
			{Input: "tree = [5], k = 1", Expected: "5"},
		},
		Solution: `type TreeInfo struct {
	NumberOfNodesVisited int
	LatestVisitedNodeValue int
}

func findKthLargestValueInBst(tree *TreeNode, k int) int {
	treeInfo := &TreeInfo{NumberOfNodesVisited: 0, LatestVisitedNodeValue: -1}
	reverseInOrderTraverse(tree, k, treeInfo)
	return treeInfo.LatestVisitedNodeValue
}

func reverseInOrderTraverse(node *TreeNode, k int, treeInfo *TreeInfo) {
	if node == nil || treeInfo.NumberOfNodesVisited >= k {
		return
	}
	reverseInOrderTraverse(node.Right, k, treeInfo)
	if treeInfo.NumberOfNodesVisited < k {
		treeInfo.NumberOfNodesVisited++
		treeInfo.LatestVisitedNodeValue = node.Val
		reverseInOrderTraverse(node.Left, k, treeInfo)
	}
}`,
		PythonSolution: `class TreeInfo:
    def __init__(self):
        self.number_of_nodes_visited = 0
        self.latest_visited_node_value = -1

def findKthLargestValueInBst(tree: Optional[TreeNode], k: int) -> int:
    tree_info = TreeInfo()
    reverse_in_order_traverse(tree, k, tree_info)
    return tree_info.latest_visited_node_value

def reverse_in_order_traverse(node, k, tree_info):
    if node is None or tree_info.number_of_nodes_visited >= k:
        return
    reverse_in_order_traverse(node.right, k, tree_info)
    if tree_info.number_of_nodes_visited < k:
        tree_info.number_of_nodes_visited += 1
        tree_info.latest_visited_node_value = node.val
        reverse_in_order_traverse(node.left, k, tree_info)`,
		Explanation: `Finding the kth largest value in a BST requires traversing nodes in descending order. The key insight is that a reverse in-order traversal (right subtree, root, left subtree) visits nodes in descending order, which is perfect for finding the kth largest element.

The algorithm uses a helper structure to track how many nodes have been visited and the value of the most recently visited node. It performs a reverse in-order traversal: first recursively visit the right subtree (larger values), then process the current node, then recursively visit the left subtree (smaller values). After visiting k nodes, we stop and return the value.

The optimization is that we can stop early once we've visited k nodes, avoiding unnecessary traversal of the entire tree. We check the count before and after processing each node to ensure we stop as soon as we've found the kth largest element.

Edge cases handled include: k equals 1 (returns the largest value), k equals the number of nodes (returns the smallest value), and trees with duplicate values (handled correctly as BST allows duplicates in right subtree).

An alternative approach would do a full in-order traversal and store all values, then return the (n-k+1)th element, but that requires O(n) space. Another approach would use an iterative method with a stack, but the recursive approach is cleaner. The time complexity is O(h+k) where h is the tree height, as we traverse down to the rightmost node (h steps) then visit k nodes. Space complexity is O(h) for the recursion stack.

Think of this as reading a sorted list backwards: you start from the rightmost (largest) element and count backwards until you reach the kth position.`,
	},
	{
		ID:          103,
		Title:       "Reconstruct BST",
		Description: "The pre-order traversal of a Binary Search Tree (BST) is a traversal technique that starts at the root node and uses pre-order traversal to recursively visit all nodes in the BST. Write a function that takes in a non-empty array of integers representing a pre-order traversal of a BST and returns the root node of the BST.",
		Difficulty:  "Medium",
		Topic:       "Binary Search Trees",
		Signature:   "func reconstructBst(preOrderTraversalValues []int) *TreeNode",
		TestCases: []TestCase{
			{Input: "preOrderTraversalValues = [10, 4, 2, 1, 5, 17, 19, 18]", Expected: "Valid BST"},
		},
		Solution: `type TreeInfo struct {
	RootIdx int
}

func reconstructBst(preOrderTraversalValues []int) *TreeNode {
	treeInfo := &TreeInfo{RootIdx: 0}
	return reconstructBstFromRange(preOrderTraversalValues, math.MinInt32, math.MaxInt32, treeInfo)
}

func reconstructBstFromRange(preOrderTraversalValues []int, lowerBound int, upperBound int, currentSubtreeInfo *TreeInfo) *TreeNode {
	if currentSubtreeInfo.RootIdx == len(preOrderTraversalValues) {
		return nil
	}
	rootValue := preOrderTraversalValues[currentSubtreeInfo.RootIdx]
	if rootValue < lowerBound || rootValue >= upperBound {
		return nil
	}
	currentSubtreeInfo.RootIdx++
	leftSubtree := reconstructBstFromRange(preOrderTraversalValues, lowerBound, rootValue, currentSubtreeInfo)
	rightSubtree := reconstructBstFromRange(preOrderTraversalValues, rootValue, upperBound, currentSubtreeInfo)
	return &TreeNode{Val: rootValue, Left: leftSubtree, Right: rightSubtree}
}`,
		PythonSolution: `class TreeInfo:
    def __init__(self):
        self.root_idx = 0

def reconstructBst(preOrderTraversalValues: List[int]) -> Optional[TreeNode]:
    tree_info = TreeInfo()
    return reconstruct_bst_from_range(preOrderTraversalValues, float('-inf'), float('inf'), tree_info)

def reconstruct_bst_from_range(pre_order_traversal_values, lower_bound, upper_bound, current_subtree_info):
    if current_subtree_info.root_idx == len(pre_order_traversal_values):
        return None
    root_value = pre_order_traversal_values[current_subtree_info.root_idx]
    if root_value < lower_bound or root_value >= upper_bound:
        return None
    current_subtree_info.root_idx += 1
    left_subtree = reconstruct_bst_from_range(pre_order_traversal_values, lower_bound, root_value, current_subtree_info)
    right_subtree = reconstruct_bst_from_range(pre_order_traversal_values, root_value, upper_bound, current_subtree_info)
    return TreeNode(root_value, left_subtree, right_subtree)`,
		Explanation: `Reconstructing a BST from a pre-order traversal requires using the BST property to determine which values belong in the left and right subtrees. The key insight is that we can use bounds (min and max values) to validate whether a value can be placed at the current position.

The algorithm processes the pre-order array sequentially, maintaining a pointer to track the current position. For each recursive call, it checks if the current value falls within the valid range defined by lower and upper bounds. The first value in the pre-order array is the root. Values less than the root belong in the left subtree (with bounds [lower, root]), and values greater than or equal to the root belong in the right subtree (with bounds [root, upper]).

This approach efficiently reconstructs the tree in a single pass by leveraging the BST property: at each node, we know exactly which range of values can appear in its subtrees. The bounds prevent us from incorrectly placing values that belong elsewhere in the tree.

Edge cases handled include: empty arrays (returns nil), single-node trees (that value is the root), trees where all values are in left subtree (right subtree is nil), trees where all values are in right subtree (left subtree is nil), and trees with duplicate values (handled by the >= comparison for right subtree).

An alternative approach would be to sort the pre-order array to get in-order, then use both to reconstruct, but that requires O(n log n) time for sorting. The bounds-based approach achieves O(n) time by processing each value exactly once. The space complexity is O(n) for the recursion stack in the worst case of a skewed tree.

Think of this as building a tree by reading values in pre-order: you know the first value is the root, and you use the BST property to determine which subsequent values go left (smaller) or right (larger or equal), with bounds ensuring you don't place values in the wrong subtree.`,
	},
	{
		ID:          104,
		Title:       "Invert Binary Tree",
		Description: "Write a function that takes in a Binary Tree and inverts it. In other words, the function should swap every left node in the tree for its corresponding right node.",
		Difficulty:  "Medium",
		Topic:       "Binary Trees",
		Signature:   "func invertBinaryTree(tree *TreeNode)",
		TestCases: []TestCase{
			{Input: "tree = [1, 2, 3, 4, 5, 6, 7, 8, 9]", Expected: "Inverted tree"},
		},
		Solution: `func invertBinaryTree(tree *TreeNode) {
	if tree == nil {
		return
	}
	tree.Left, tree.Right = tree.Right, tree.Left
	invertBinaryTree(tree.Left)
	invertBinaryTree(tree.Right)
}`,
		PythonSolution: `def invertBinaryTree(tree: Optional[TreeNode]) -> None:
    if tree is None:
        return
    tree.left, tree.right = tree.right, tree.left
    invertBinaryTree(tree.left)
    invertBinaryTree(tree.right)`,
		Explanation: `Inverting a binary tree means swapping every left child with its corresponding right child throughout the entire tree. The key insight is that this can be done recursively: for each node, swap its children, then recursively invert both subtrees.

The algorithm uses a simple recursive approach: if the node is nil, return (base case). Otherwise, swap the left and right children, then recursively invert the left subtree and the right subtree. Since we swap before recursing, the recursive calls operate on the already-swapped children, which is correct.

This approach works because inverting a tree is symmetric: inverting the left and right subtrees and then swapping them produces the same result as swapping them first and then inverting. The order doesn't matter, making the implementation straightforward.

Edge cases handled include: empty trees (returns immediately), single-node trees (no children to swap, returns), and trees with only left or right children (swap handles nil correctly).

An alternative iterative approach using a queue would process nodes level by level, but the recursive approach is simpler and more intuitive. The time complexity is O(n) as we visit each node exactly once, and space complexity is O(h) for the recursion stack where h is the tree height.

Think of inverting a tree as creating a mirror image: every left becomes right and every right becomes left, recursively throughout the entire tree structure.`,
	},
	{
		ID:          105,
		Title:       "Binary Tree Diameter",
		Description: "Write a function that takes in a Binary Tree and returns its diameter. The diameter of a binary tree is defined as the length of its longest path, even if that path doesn't pass through the root of the tree. A path is a collection of connected nodes in a tree, where no node is connected to more than two other nodes. The length of a path is the number of edges between the path's first node and its last node.",
		Difficulty:  "Medium",
		Topic:       "Binary Trees",
		Signature:   "func binaryTreeDiameter(tree *TreeNode) int",
		TestCases: []TestCase{
			{Input: "tree = [1, 3, 2, 7, 4, null, null, 8, null, null, 5, null, null, 9, null, null, 6]", Expected: "6"},
		},
		Solution: `func binaryTreeDiameter(tree *TreeNode) int {
	return getTreeInfo(tree).Diameter
}

type TreeInfo struct {
	Height   int
	Diameter int
}

func getTreeInfo(tree *TreeNode) TreeInfo {
	if tree == nil {
		return TreeInfo{Height: 0, Diameter: 0}
	}
	leftTreeInfo := getTreeInfo(tree.Left)
	rightTreeInfo := getTreeInfo(tree.Right)
	longestPathThroughRoot := leftTreeInfo.Height + rightTreeInfo.Height
	maxDiameterSoFar := max(leftTreeInfo.Diameter, rightTreeInfo.Diameter)
	currentDiameter := max(longestPathThroughRoot, maxDiameterSoFar)
	currentHeight := 1 + max(leftTreeInfo.Height, rightTreeInfo.Height)
	return TreeInfo{Height: currentHeight, Diameter: currentDiameter}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def binaryTreeDiameter(tree: Optional[TreeNode]) -> int:
    return get_tree_info(tree).diameter

class TreeInfo:
    def __init__(self, height, diameter):
        self.height = height
        self.diameter = diameter

def get_tree_info(tree):
    if tree is None:
        return TreeInfo(0, 0)
    left_tree_info = get_tree_info(tree.left)
    right_tree_info = get_tree_info(tree.right)
    longest_path_through_root = left_tree_info.height + right_tree_info.height
    max_diameter_so_far = max(left_tree_info.diameter, right_tree_info.diameter)
    current_diameter = max(longest_path_through_root, max_diameter_so_far)
    current_height = 1 + max(left_tree_info.height, right_tree_info.height)
    return TreeInfo(current_height, current_diameter)`,
		Explanation: `Finding the diameter of a binary tree requires understanding that the diameter is the longest path between any two nodes, which may or may not pass through the root. The key insight is that we need to calculate both the height of subtrees and the diameter that passes through each node.

The algorithm uses a recursive approach that returns both height and diameter information for each subtree. For each node, it calculates the height of left and right subtrees. The diameter passing through the current node is the sum of left and right subtree heights. The overall diameter at this node is the maximum of: the diameter through the current node, the maximum diameter found in the left subtree, and the maximum diameter found in the right subtree.

This approach efficiently computes the diameter in a single pass by combining height and diameter calculations. We need both pieces of information because the longest path might be entirely within a subtree (captured by subtree diameters) or might pass through the current node (calculated from subtree heights).

Edge cases handled include: empty trees (diameter is 0), single-node trees (diameter is 0, as there are no edges), trees where diameter passes through root (sum of left and right subtree heights), and trees where diameter is entirely in one subtree (handled by comparing subtree diameters).

An alternative O(n²) approach would be to find the longest path for each node by calculating heights repeatedly, but the current approach is optimal with O(n) time complexity. The space complexity is O(h) for the recursion stack, where h is the height of the tree.

Think of this as finding the longest "chain" in the tree: at each node, you check if the longest chain goes through you (using subtree heights) or is entirely within one of your subtrees (using subtree diameters), and you keep track of the maximum found so far.`,
	},
	{
		ID:          106,
		Title:       "Find Successor",
		Description: "Write a function that takes in a Binary Search Tree (BST) and a target integer value and returns the node in the BST that comes right after the provided target value. You can assume that there will only be one closest value.",
		Difficulty:  "Medium",
		Topic:       "Binary Search Trees",
		Signature:   "func findSuccessor(tree *TreeNode, target int) *TreeNode",
		TestCases: []TestCase{
			{Input: "tree = [10, 5, 15, 2, 5, 13, 22, 1, null, null, null, null, 14], target = 12", Expected: "13"},
		},
		Solution: `func findSuccessor(tree *TreeNode, target int) *TreeNode {
	if tree == nil {
		return nil
	}
	var successor *TreeNode
	current := tree
	for current != nil {
		if current.Val <= target {
			current = current.Right
		} else {
			successor = current
			current = current.Left
		}
	}
	return successor
}`,
		PythonSolution: `def findSuccessor(tree: Optional[TreeNode], target: int) -> Optional[TreeNode]:
    if tree is None:
        return None
    successor = None
    current = tree
    while current is not None:
        if current.val <= target:
            current = current.right
        else:
            successor = current
            current = current.left
    return successor`,
		Explanation: `Finding the successor (next larger value) in a BST requires understanding the BST property: all values in the left subtree are smaller, and all values in the right subtree are larger. The successor is the smallest value that is greater than the target.

The algorithm traverses the BST starting from the root. If the current node's value is less than or equal to the target, the successor must be in the right subtree (since current and everything to its left is too small), so we go right. If the current node's value is greater than the target, this node is a candidate for the successor (it's larger than target), so we update our successor candidate and go left to find a potentially smaller value that's still greater than the target.

This approach efficiently finds the successor in O(h) time by leveraging the BST property to eliminate half of the remaining search space at each step. We maintain a successor variable that gets updated whenever we find a node larger than the target, ensuring we find the smallest such value.

Edge cases handled include: target is the maximum value (no successor, returns nil), target doesn't exist in tree (finds next larger value), target is the root (successor is minimum of right subtree or nil), and single-node trees (no successor if target is that node).

An alternative approach would be to do an in-order traversal and find the next element after target, but that requires O(n) time. The current approach is optimal with O(h) time complexity, where h is the height of the tree. The space complexity is O(1) since we use iterative traversal.

Think of this as finding the next number in a sorted list: if the current number is too small, look to the right; if it's larger than target, it's a candidate, but check left to see if there's a smaller number that's still larger than target.`,
	},
	{
		ID:          107,
		Title:       "Height Balanced Binary Tree",
		Description: "You're given the root node of a Binary Tree. Write a function that returns true if this Binary Tree is height balanced and false if it isn't. A Binary Tree is height balanced if for each node in the tree, the difference between the height of its left subtree and the height of its right subtree is at most 1.",
		Difficulty:  "Medium",
		Topic:       "Binary Trees",
		Signature:   "func heightBalancedBinaryTree(tree *TreeNode) bool",
		TestCases: []TestCase{
			{Input: "tree = [1, 2, 3, 4, 5, null, 6, 7, 8]", Expected: "false"},
			{Input: "tree = [1, 2, 3, 4, 5]", Expected: "true"},
		},
		Solution: `type TreeInfo struct {
	IsBalanced bool
	Height     int
}

func heightBalancedBinaryTree(tree *TreeNode) bool {
	treeInfo := getTreeInfo(tree)
	return treeInfo.IsBalanced
}

func getTreeInfo(node *TreeNode) TreeInfo {
	if node == nil {
		return TreeInfo{IsBalanced: true, Height: -1}
	}
	leftSubtreeInfo := getTreeInfo(node.Left)
	rightSubtreeInfo := getTreeInfo(node.Right)
	isBalanced := leftSubtreeInfo.IsBalanced && rightSubtreeInfo.IsBalanced && abs(leftSubtreeInfo.Height-rightSubtreeInfo.Height) <= 1
	height := max(leftSubtreeInfo.Height, rightSubtreeInfo.Height) + 1
	return TreeInfo{IsBalanced: isBalanced, Height: height}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def heightBalancedBinaryTree(tree: Optional[TreeNode]) -> bool:
    tree_info = get_tree_info(tree)
    return tree_info.is_balanced

class TreeInfo:
    def __init__(self, is_balanced, height):
        self.is_balanced = is_balanced
        self.height = height

def get_tree_info(node):
    if node is None:
        return TreeInfo(True, -1)
    left_subtree_info = get_tree_info(node.left)
    right_subtree_info = get_tree_info(node.right)
    is_balanced = (left_subtree_info.is_balanced and right_subtree_info.is_balanced and
                   abs(left_subtree_info.height - right_subtree_info.height) <= 1)
    height = max(left_subtree_info.height, right_subtree_info.height) + 1
    return TreeInfo(is_balanced, height)`,
		Explanation: `A height-balanced binary tree requires that for every node, the heights of its left and right subtrees differ by at most 1. The challenge is to check this efficiently without recalculating heights multiple times.

The algorithm uses a recursive approach that returns both the balance status and height for each subtree. For each node, it recursively checks if both left and right subtrees are balanced, and also verifies that their height difference is at most 1. If all conditions are met, the current subtree is balanced. The height is calculated as 1 plus the maximum of the left and right subtree heights.

This approach is efficient because it calculates heights and checks balance in a single pass, avoiding redundant calculations. By returning both pieces of information from each recursive call, we can check balance and calculate height simultaneously.

Edge cases handled include: empty trees (balanced by definition, height -1), single-node trees (balanced, height 0), trees where one subtree is much deeper (detected by height difference check), and trees that are balanced at root but unbalanced in subtrees (detected recursively).

An alternative O(n²) approach would be to calculate height separately for balance checking, but that's inefficient. The current approach is optimal with O(n) time complexity, visiting each node exactly once. The space complexity is O(h) for the recursion stack, where h is the height of the tree.

Think of this as checking if a tree is "well-proportioned": at each node, you check if your left and right sides are roughly the same height (difference ≤ 1) and if both sides are themselves well-proportioned.`,
	},
	{
		ID:          108,
		Title:       "Max Subset Sum No Adjacent",
		Description: "Write a function that takes in an array of positive integers and returns the maximum sum of non-adjacent elements in the array. If the input array is empty, the function should return 0.",
		Difficulty:  "Medium",
		Topic:       "Dynamic Programming",
		Signature:   "func maxSubsetSumNoAdjacent(array []int) int",
		TestCases: []TestCase{
			{Input: "array = []int{75, 105, 120, 75, 90, 135}", Expected: "330"},
			{Input: "array = []int{}", Expected: "0"},
			{Input: "array = []int{1}", Expected: "1"},
			{Input: "array = []int{1, 2}", Expected: "2"},
		},
		Solution: `func maxSubsetSumNoAdjacent(array []int) int {
	if len(array) == 0 {
		return 0
	}
	if len(array) == 1 {
		return array[0]
	}
	second := array[0]
	first := max(array[0], array[1])
	for i := 2; i < len(array); i++ {
		current := max(first, second+array[i])
		second = first
		first = current
	}
	return first
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def maxSubsetSumNoAdjacent(array: List[int]) -> int:
    if len(array) == 0:
        return 0
    if len(array) == 1:
        return array[0]
    second = array[0]
    first = max(array[0], array[1])
    for i in range(2, len(array)):
        current = max(first, second + array[i])
        second = first
        first = current
    return first`,
		Explanation: `This is a classic dynamic programming problem that requires finding the maximum sum of non-adjacent elements. The key insight is that for each position, we must choose between including the current element (and taking the best sum ending two positions back) or excluding it (and taking the best sum ending one position back).

The algorithm uses two variables to track the maximum sums ending at the previous two positions, avoiding the need for a full array. For each element, we calculate the maximum sum ending at the current position as the maximum of: the sum ending at the previous position (excluding current) or the sum ending two positions back plus the current element (including current). We then update our tracking variables to move forward.

This approach efficiently solves the problem in O(n) time with O(1) space by only keeping track of the two most recent maximum sums. The dynamic programming recurrence relation is: maxSum[i] = max(maxSum[i-1], maxSum[i-2] + array[i]), which we implement using just two variables.

Edge cases handled include: empty arrays (returns 0), single-element arrays (returns that element), two-element arrays (returns the maximum of the two), and arrays where alternating elements give the maximum sum.

An alternative approach would use a full DP array, but that requires O(n) space. Another approach would be recursive with memoization, but that has O(n) space for the memoization table and O(n) time. The two-variable approach is optimal for space while maintaining O(n) time.

Think of this as making choices at each house: you can either skip the current house and keep your previous best, or take the current house and add it to your best from two houses ago, always choosing the option that gives you more money.`,
	},
	{
		ID:          109,
		Title:       "Number Of Ways To Make Change",
		Description: "Given an array of distinct positive integers representing coin denominations and a single non-negative integer n representing a target amount of money, write a function that returns the number of ways to make change for that target amount using the given coin denominations. Note that an unlimited amount of coins is at your disposal.",
		Difficulty:  "Medium",
		Topic:       "Dynamic Programming",
		Signature:   "func numberOfWaysToMakeChange(n int, denoms []int) int",
		TestCases: []TestCase{
			{Input: "n = 6, denoms = [1, 5]", Expected: "2"},
			{Input: "n = 0, denoms = [2, 3, 4, 7]", Expected: "1"},
			{Input: "n = 9, denoms = [5]", Expected: "0"},
		},
		Solution: `func numberOfWaysToMakeChange(n int, denoms []int) int {
	ways := make([]int, n+1)
	ways[0] = 1
	for _, denom := range denoms {
		for amount := 1; amount <= n; amount++ {
			if denom <= amount {
				ways[amount] += ways[amount-denom]
			}
		}
	}
	return ways[n]
}`,
		PythonSolution: `def numberOfWaysToMakeChange(n: int, denoms: List[int]) -> int:
    ways = [0] * (n + 1)
    ways[0] = 1
    for denom in denoms:
        for amount in range(1, n + 1):
            if denom <= amount:
                ways[amount] += ways[amount - denom]
    return ways[n]`,
		Explanation: `This problem requires counting the number of ways to make change for a target amount using given coin denominations. The key insight is that we can build up the solution using dynamic programming: the number of ways to make amount X equals the sum of ways to make (X - denom) for each denomination.

The algorithm uses a DP array where ways[amount] represents the number of ways to make that amount. We initialize ways[0] = 1 (one way to make 0: use no coins). Then, for each amount from 1 to n, we iterate through all denominations. For each denomination, if the denomination is less than or equal to the current amount, we add ways[amount - denom] to ways[amount]. This accumulates all possible ways to reach the current amount.

The order of iteration matters: we iterate through denominations in the outer loop and amounts in the inner loop. This ensures that when we consider a denomination, we've already computed all ways to make smaller amounts using previous denominations, allowing us to build up the solution correctly.

Edge cases handled include: target amount of 0 (one way: use no coins), impossible amounts (ways remains 0), single denomination that divides evenly (multiple ways possible), and denominations larger than target (skipped correctly).

An alternative recursive approach with memoization would work but is less efficient. Another approach would iterate amounts first then denominations, but the current order is more intuitive. The time complexity is O(nd) where n is the target amount and d is the number of denominations, and space complexity is O(n) for the DP array.

Think of this as building up possibilities: to make amount X, you can take any coin and see how many ways there are to make the remaining amount (X - coin value), then sum up all those possibilities.`,
	},
	{
		ID:          110,
		Title:       "Min Number Of Coins For Change",
		Description: "Given an array of positive integers representing coin denominations and a single non-negative integer n representing a target amount of money, write a function that returns the smallest number of coins needed to make change for (to sum up to) that target amount using the given coin denominations. Note that you have access to an unlimited amount of coins. In other words, if you're given coin denominations of [1, 5, 10], you have access to an unlimited amount of 1s, 5s, and 10s. If it's impossible to make change for the target amount, return -1.",
		Difficulty:  "Medium",
		Topic:       "Dynamic Programming",
		Signature:   "func minNumberOfCoinsForChange(n int, denoms []int) int",
		TestCases: []TestCase{
			{Input: "n = 7, denoms = [1, 5, 10]", Expected: "3"},
			{Input: "n = 0, denoms = [1, 2, 3]", Expected: "0"},
			{Input: "n = 3, denoms = [2]", Expected: "-1"},
		},
		Solution: `func minNumberOfCoinsForChange(n int, denoms []int) int {
	numOfCoins := make([]int, n+1)
	for i := range numOfCoins {
		numOfCoins[i] = n + 1
	}
	numOfCoins[0] = 0
	for _, denom := range denoms {
		for amount := 0; amount <= n; amount++ {
			if denom <= amount {
				numOfCoins[amount] = min(numOfCoins[amount], numOfCoins[amount-denom]+1)
			}
		}
	}
	if numOfCoins[n] != n+1 {
		return numOfCoins[n]
	}
	return -1
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}`,
		PythonSolution: `def minNumberOfCoinsForChange(n: int, denoms: List[int]) -> int:
    num_of_coins = [n + 1] * (n + 1)
    num_of_coins[0] = 0
    for denom in denoms:
        for amount in range(n + 1):
            if denom <= amount:
                num_of_coins[amount] = min(num_of_coins[amount], num_of_coins[amount - denom] + 1)
    return num_of_coins[n] if num_of_coins[n] != n + 1 else -1`,
		Explanation: `This problem requires finding the minimum number of coins needed to make change for a target amount. Unlike counting ways, here we want to minimize the number of coins, which requires a different dynamic programming approach.

The algorithm uses a DP array where numCoins[amount] represents the minimum number of coins needed to make that amount. We initialize all values to a large number (or infinity) except numCoins[0] = 0 (zero coins needed for amount 0). Then, for each amount from 1 to n, we iterate through all denominations. For each denomination that is less than or equal to the current amount, we update numCoins[amount] to be the minimum of its current value and numCoins[amount - denom] + 1.

The +1 represents using one coin of the current denomination, and we're trying to minimize the total number of coins. By checking all denominations for each amount, we ensure we find the optimal solution. If numCoins[n] remains at the initial large value after processing, it means the amount cannot be made, so we return -1.

Edge cases handled include: target amount of 0 (returns 0 coins), impossible amounts (numCoins[n] remains large, return -1), denominations that don't divide evenly (finds minimum coins through combination), and single denomination cases (amount must be multiple of denomination).

An alternative greedy approach would work for certain coin systems (like US coins) but fails for general cases. The DP approach is guaranteed to find the optimal solution for any coin system. The time complexity is O(nd) where n is the target amount and d is the number of denominations, and space complexity is O(n) for the DP array.

Think of this as finding the shortest path to an amount: at each amount, you try using each coin and see which gives you the minimum total coins, building up from smaller amounts to larger ones.`,
	},
	{
		ID:          111,
		Title:       "Levenshtein Distance",
		Description: "Write a function that takes in two strings and returns the minimum number of edit operations that need to be performed on the first string to obtain the second string. There are three edit operations: insertion of a character, deletion of a character, and substitution of a character for another.",
		Difficulty:  "Medium",
		Topic:       "Dynamic Programming",
		Signature:   "func levenshteinDistance(str1 string, str2 string) int",
		TestCases: []TestCase{
			{Input: "str1 = \"abc\", str2 = \"yabd\"", Expected: "2"},
			{Input: "str1 = \"\", str2 = \"\"", Expected: "0"},
			{Input: "str1 = \"\", str2 = \"abc\"", Expected: "3"},
		},
		Solution: `func levenshteinDistance(str1 string, str2 string) int {
	edits := make([][]int, len(str2)+1)
	for i := range edits {
		edits[i] = make([]int, len(str1)+1)
	}
	for i := 0; i < len(str2)+1; i++ {
		for j := 0; j < len(str1)+1; j++ {
			edits[i][j] = j
		}
		edits[i][0] = i
	}
	for i := 1; i < len(str2)+1; i++ {
		for j := 1; j < len(str1)+1; j++ {
			if str2[i-1] == str1[j-1] {
				edits[i][j] = edits[i-1][j-1]
			} else {
				edits[i][j] = 1 + min(edits[i-1][j-1], min(edits[i-1][j], edits[i][j-1]))
			}
		}
	}
	return edits[len(str2)][len(str1)]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}`,
		PythonSolution: `def levenshteinDistance(str1: str, str2: str) -> int:
    edits = [[j for j in range(len(str1) + 1)] for i in range(len(str2) + 1)]
    for i in range(len(str2) + 1):
        edits[i][0] = i
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str2[i - 1] == str1[j - 1]:
                edits[i][j] = edits[i - 1][j - 1]
            else:
                edits[i][j] = 1 + min(edits[i - 1][j - 1], edits[i - 1][j], edits[i][j - 1])
    return edits[len(str2)][len(str1)]`,
		Explanation: `The Levenshtein distance (edit distance) measures the minimum number of single-character edits needed to transform one string into another. This is a classic dynamic programming problem that requires building up solutions for subproblems.

The algorithm uses a 2D DP table where edits[i][j] represents the minimum edits needed to convert the first j characters of str1 to the first i characters of str2. We initialize the first row and column to represent converting empty strings (requires i insertions or j deletions). Then, for each cell, if the characters match, no edit is needed, so edits[i][j] = edits[i-1][j-1]. Otherwise, we take the minimum of three operations: insert (edits[i-1][j] + 1), delete (edits[i][j-1] + 1), or substitute (edits[i-1][j-1] + 1).

This approach efficiently computes the edit distance by considering all possible edit sequences and choosing the optimal one. The DP table ensures we don't recompute subproblems, making the solution efficient.

Edge cases handled include: empty strings (distance equals length of non-empty string), identical strings (distance is 0), strings where one is a subsequence of the other (requires insertions or deletions), and strings with no common characters (requires substitutions or deletions+insertions).

An alternative recursive approach with memoization would work but is less efficient due to function call overhead. Another approach could optimize space to O(min(n,m)) by using only two rows, but the 2D approach is clearer. The time complexity is O(n*m) where n and m are the string lengths, and space complexity is O(n*m) for the DP table.

Think of this as building a transformation path: at each step, you can insert, delete, or substitute a character, and you want to find the shortest path (minimum edits) from one string to another.`,
	},
	{
		ID:          112,
		Title:       "Number Of Ways To Traverse Graph",
		Description: "You're given two positive integers representing the width and height of a grid-shaped graph. Write a function that returns the number of ways to reach the bottom right corner of the graph when starting at the top left corner. Each move you take must either go down or right. In other words, you can never move up or left in the graph.",
		Difficulty:  "Medium",
		Topic:       "Dynamic Programming",
		Signature:   "func numberOfWaysToTraverseGraph(width int, height int) int",
		TestCases: []TestCase{
			{Input: "width = 4, height = 3", Expected: "10"},
			{Input: "width = 2, height = 3", Expected: "3"},
		},
		Solution: `func numberOfWaysToTraverseGraph(width int, height int) int {
	if width == 1 || height == 1 {
		return 1
	}
	return numberOfWaysToTraverseGraph(width-1, height) + numberOfWaysToTraverseGraph(width, height-1)
}`,
		PythonSolution: `def numberOfWaysToTraverseGraph(width: int, height: int) -> int:
    if width == 1 or height == 1:
        return 1
    return numberOfWaysToTraverseGraph(width - 1, height) + numberOfWaysToTraverseGraph(width, height - 1)`,
		Explanation: `This problem asks for the number of unique paths from the top-left to bottom-right corner of a grid, moving only right or down. This is a classic dynamic programming problem that can be solved using the combinatorial insight that each path requires exactly (width-1) right moves and (height-1) down moves.

The recursive approach recognizes that to reach cell (w, h), you must come from either (w-1, h) by moving right or (w, h-1) by moving down. Therefore, ways(w, h) = ways(w-1, h) + ways(w, h-1). The base cases are: ways(1, h) = 1 (only one way: all down moves) and ways(w, 1) = 1 (only one way: all right moves).

This recurrence relation can be optimized using dynamic programming or memoization to avoid recalculating the same subproblems. The DP approach builds up solutions from smaller grids to larger ones, storing results in a 2D table.

Edge cases handled include: 1x1 grid (1 way: already at destination), single-row grids (1 way: all right moves), single-column grids (1 way: all down moves), and larger grids where paths can be computed efficiently.

An alternative mathematical approach recognizes this as a combinations problem: C((w+h-2), (w-1)) = (w+h-2)! / ((w-1)! * (h-1)!), which gives the answer directly but requires handling large factorials. The DP approach is more practical and avoids overflow issues. The time complexity with DP is O(w*h), and space can be optimized to O(min(w,h)).

Think of this as counting paths: at each intersection, you can go right or down, and the number of ways to reach the destination is the sum of ways from the cell above and the cell to the left.`,
	},
	{
		ID:          113,
		Title:       "Single Cycle Check",
		Description: "You're given an array of integers where each integer represents a jump of its value in the array. For instance, the integer 2 represents a jump of two indices forward in the array; the integer -3 represents a jump of three indices backward in the array. If a jump spills past the array's bounds, it wraps over to the other side. For instance, a jump of -1 at index 0 brings us to the last index in the array, and a jump of 1 at the last index in the array brings us to index 0. Write a function that returns a boolean representing whether the jumps in the array form a single cycle. A single cycle occurs if, starting at any index in the array and following the jumps, every element in the array is visited exactly once before landing back on the starting index.",
		Difficulty:  "Medium",
		Topic:       "Graphs",
		Signature:   "func hasSingleCycle(array []int) bool",
		TestCases: []TestCase{
			{Input: "array = []int{2, 3, 1, -4, -4, 2}", Expected: "true"},
			{Input: "array = []int{2, 2, -1}", Expected: "true"},
			{Input: "array = []int{2, 2, 2}", Expected: "true"},
			{Input: "array = []int{1, 1, 1, 1, 1}", Expected: "true"},
		},
		Solution: `func hasSingleCycle(array []int) bool {
	numElementsVisited := 0
	currentIdx := 0
	for numElementsVisited < len(array) {
		if numElementsVisited > 0 && currentIdx == 0 {
			return false
		}
		numElementsVisited++
		currentIdx = getNextIdx(currentIdx, array)
	}
	return currentIdx == 0
}

func getNextIdx(currentIdx int, array []int) int {
	jump := array[currentIdx]
	nextIdx := (currentIdx + jump) % len(array)
	if nextIdx >= 0 {
		return nextIdx
	}
	return nextIdx + len(array)
}`,
		PythonSolution: `def hasSingleCycle(array: List[int]) -> bool:
    num_elements_visited = 0
    current_idx = 0
    while num_elements_visited < len(array):
        if num_elements_visited > 0 and current_idx == 0:
            return False
        num_elements_visited += 1
        current_idx = get_next_idx(current_idx, array)
    return current_idx == 0

def get_next_idx(current_idx, array):
    jump = array[current_idx]
    next_idx = (current_idx + jump) % len(array)
    return next_idx if next_idx >= 0 else next_idx + len(array)`,
		Explanation: `This problem checks if following jumps in an array forms a single cycle that visits every element exactly once before returning to the start. The key insight is that a valid single cycle must visit exactly n elements (where n is array length) and return to the starting index.

The algorithm starts at index 0 and follows jumps, counting how many elements have been visited. The critical checks are: if we return to index 0 before visiting all n elements, there must be multiple cycles (return false). If we visit all n elements and end up back at index 0, we have a single cycle (return true). If we visit all n elements but don't end at 0, the cycle doesn't include the starting point correctly.

The next index calculation handles wrap-around using modulo arithmetic. For positive jumps, we use (currentIdx + jump) % len(array). For negative jumps that result in negative indices, we add len(array) to wrap around to the correct position.

Edge cases handled include: arrays with all positive jumps (wraps forward), arrays with all negative jumps (wraps backward), arrays with mixed jumps, arrays that form multiple cycles (detected early), and arrays where jumps don't form a cycle (detected when we don't return to start after n visits).

An alternative approach would use a visited set to track which indices have been seen, but that requires O(n) space. The current approach uses O(1) space by only tracking the count and current position, making it more space-efficient.

Think of this as following a path through a circular array: you start at position 0, jump according to each value, and check if you visit every position exactly once before returning to where you started.`,
	},
	{
		ID:          114,
		Title:       "Breadth-first Search",
		Description: "You're given a Node class that has a name and an array of optional children nodes. When put together, nodes form an acyclic tree-like structure. Implement the breadthFirstSearch method on the Node class, which takes in an empty array, traverses the tree using the Breadth-first Search approach (specifically navigating the tree from left to right), stores all of the nodes' names in the input array, and returns it.",
		Difficulty:  "Medium",
		Topic:       "Graphs",
		Signature:   "func (n *Node) BreadthFirstSearch(array []string) []string",
		TestCases: []TestCase{
			{Input: "graph = A -> [B, C, D], B -> [E, F], D -> [G, H], F -> [I, J], G -> [K]", Expected: "[\"A\", \"B\", \"C\", \"D\", \"E\", \"F\", \"G\", \"H\", \"I\", \"J\", \"K\"]"},
		},
		Solution: `type Node struct {
	Name     string
	Children []*Node
}

func (n *Node) BreadthFirstSearch(array []string) []string {
	queue := []*Node{n}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		array = append(array, current.Name)
		queue = append(queue, current.Children...)
	}
	return array
}`,
		PythonSolution: `from collections import deque

class Node:
    def __init__(self, name):
        self.children = []
        self.name = name
    
    def breadthFirstSearch(self, array):
        queue = deque([self])
        while queue:
            current = queue.popleft()
            array.append(current.name)
            queue.extend(current.children)
        return array`,
		Explanation: `Breadth-first search (BFS) explores a graph level by level, visiting all nodes at the current depth before moving to nodes at the next depth. This is implemented using a queue data structure that follows the First-In-First-Out (FIFO) principle.

The algorithm starts by adding the root node to the queue. Then, while the queue is not empty, it dequeues a node, adds its name to the result array, and enqueues all of its children. This ensures that nodes are processed in the order they were discovered, which naturally creates a level-order traversal.

The queue-based approach is essential for BFS because it maintains the correct order: nodes discovered earlier (at shallower levels) are processed before nodes discovered later (at deeper levels). This is different from depth-first search, which uses a stack (LIFO) to go deep before going wide.

Edge cases handled include: graphs with a single node (that node is added to result), graphs with no children (only root in result), graphs with varying numbers of children per node (all handled correctly by the queue), and graphs where nodes have many children (queue grows accordingly).

An alternative recursive approach would be more complex for BFS since recursion naturally implements DFS. The iterative queue-based approach is the standard and most intuitive way to implement BFS. The time complexity is O(v+e) where v is vertices and e is edges, as we visit each node and edge once. The space complexity is O(v) for the queue in the worst case.

Think of this as exploring a building floor by floor: you visit all rooms on the first floor, then all rooms on the second floor, and so on, using a queue to remember which rooms to visit next.`,
	},
	{
		ID:          115,
		Title:       "River Sizes",
		Description: "You're given a two-dimensional array of potentially unequal height and width containing only 0s and 1s. Each 0 represents land, and each 1 represents part of a river. A river consists of any number of 1s that are either horizontally or vertically adjacent (but not diagonally adjacent). The number of adjacent 1s forming a river determine its size. Note that a river can twist. In other words, it doesn't have to be a straight vertical line or a straight horizontal line; it can be L-shaped, for example. Write a function that returns an array of the sizes of all rivers represented in the input matrix. The sizes don't need to be in any particular order.",
		Difficulty:  "Medium",
		Topic:       "Graphs",
		Signature:   "func riverSizes(matrix [][]int) []int",
		TestCases: []TestCase{
			{Input: "matrix = [[1, 0, 0, 1, 0], [1, 0, 1, 0, 0], [0, 0, 1, 0, 1], [1, 0, 1, 0, 1], [1, 0, 1, 1, 0]]", Expected: "[1, 2, 2, 2, 5]"},
		},
		Solution: `func riverSizes(matrix [][]int) []int {
	sizes := []int{}
	visited := make([][]bool, len(matrix))
	for i := range visited {
		visited[i] = make([]bool, len(matrix[0]))
	}
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			if visited[i][j] {
				continue
			}
			traverseNode(i, j, matrix, visited, &sizes)
		}
	}
	return sizes
}

func traverseNode(i int, j int, matrix [][]int, visited [][]bool, sizes *[]int) {
	currentRiverSize := 0
	nodesToExplore := [][]int{{i, j}}
	for len(nodesToExplore) > 0 {
		currentNode := nodesToExplore[0]
		nodesToExplore = nodesToExplore[1:]
		i, j := currentNode[0], currentNode[1]
		if visited[i][j] {
			continue
		}
		visited[i][j] = true
		if matrix[i][j] == 0 {
			continue
		}
		currentRiverSize++
		unvisitedNeighbors := getUnvisitedNeighbors(i, j, matrix, visited)
		nodesToExplore = append(nodesToExplore, unvisitedNeighbors...)
	}
	if currentRiverSize > 0 {
		*sizes = append(*sizes, currentRiverSize)
	}
}

func getUnvisitedNeighbors(i int, j int, matrix [][]int, visited [][]bool) [][]int {
	unvisitedNeighbors := [][]int{}
	if i > 0 && !visited[i-1][j] {
		unvisitedNeighbors = append(unvisitedNeighbors, []int{i - 1, j})
	}
	if i < len(matrix)-1 && !visited[i+1][j] {
		unvisitedNeighbors = append(unvisitedNeighbors, []int{i + 1, j})
	}
	if j > 0 && !visited[i][j-1] {
		unvisitedNeighbors = append(unvisitedNeighbors, []int{i, j - 1})
	}
	if j < len(matrix[0])-1 && !visited[i][j+1] {
		unvisitedNeighbors = append(unvisitedNeighbors, []int{i, j + 1})
	}
	return unvisitedNeighbors
}`,
		PythonSolution: `def riverSizes(matrix: List[List[int]]) -> List[int]:
    sizes = []
    visited = [[False for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if visited[i][j]:
                continue
            traverse_node(i, j, matrix, visited, sizes)
    return sizes

def traverse_node(i, j, matrix, visited, sizes):
    current_river_size = 0
    nodes_to_explore = [[i, j]]
    while nodes_to_explore:
        current_node = nodes_to_explore.pop(0)
        i, j = current_node
        if visited[i][j]:
            continue
        visited[i][j] = True
        if matrix[i][j] == 0:
            continue
        current_river_size += 1
        unvisited_neighbors = get_unvisited_neighbors(i, j, matrix, visited)
        nodes_to_explore.extend(unvisited_neighbors)
    if current_river_size > 0:
        sizes.append(current_river_size)

def get_unvisited_neighbors(i, j, matrix, visited):
    unvisited_neighbors = []
    if i > 0 and not visited[i - 1][j]:
        unvisited_neighbors.append([i - 1, j])
    if i < len(matrix) - 1 and not visited[i + 1][j]:
        unvisited_neighbors.append([i + 1, j])
    if j > 0 and not visited[i][j - 1]:
        unvisited_neighbors.append([i, j - 1])
    if j < len(matrix[0]) - 1 and not visited[i][j + 1]:
        unvisited_neighbors.append([i, j + 1])
    return unvisited_neighbors`,
		Explanation: `This problem requires finding all connected components of 1s in a 2D matrix, where connectivity is defined as horizontal or vertical adjacency. Each connected component represents a "river" whose size we need to calculate.

The algorithm uses BFS (or DFS) to explore each connected component. It maintains a visited matrix to avoid revisiting cells and to ensure each river is counted exactly once. For each unvisited cell containing a 1, it initiates a BFS/DFS traversal that explores all adjacent 1s (horizontally and vertically), counting them as it goes. When the traversal completes, the count represents the size of that river.

The key is to mark cells as visited as soon as we start exploring them, preventing double-counting. We iterate through all cells in the matrix, but only start a new traversal for unvisited cells containing 1s. This ensures we find all rivers without missing any or counting any twice.

Edge cases handled include: matrices with no rivers (all 0s, returns empty array), matrices with a single river (returns array with one size), matrices with rivers of size 1 (isolated 1s), and matrices where rivers touch edges (handled correctly by boundary checks).

An alternative approach would use union-find, but BFS/DFS is more straightforward for this problem. The time complexity is O(w*h) where w and h are width and height, as we visit each cell at most once. The space complexity is O(w*h) for the visited matrix and the BFS/DFS stack/queue in the worst case.

Think of this as finding islands of 1s in a sea of 0s: you explore each island completely (counting all connected 1s) before moving to the next unvisited 1, ensuring you find and measure every island.`,
	},
	{
		ID:          116,
		Title:       "Youngest Common Ancestor",
		Description: "You're given three inputs, all of which are instances of an AncestralTree class that have an ancestor property pointing to their youngest ancestor. The first input is the top ancestor in an ancestral tree (i.e., the only instance that has no ancestor--its ancestor property points to None / null), and the other two inputs are descendants in the ancestral tree. Write a function that returns the youngest common ancestor to the two descendants.",
		Difficulty:  "Medium",
		Topic:       "Graphs",
		Signature:   "func getYoungestCommonAncestor(topAncestor *AncestralTree, descendantOne *AncestralTree, descendantTwo *AncestralTree) *AncestralTree",
		TestCases: []TestCase{
			{Input: "Tree structure", Expected: "Youngest common ancestor"},
		},
		Solution: `type AncestralTree struct {
	Name     string
	Ancestor *AncestralTree
}

func getYoungestCommonAncestor(topAncestor *AncestralTree, descendantOne *AncestralTree, descendantTwo *AncestralTree) *AncestralTree {
	depthOne := getDescendantDepth(descendantOne, topAncestor)
	depthTwo := getDescendantDepth(descendantTwo, topAncestor)
	if depthOne > depthTwo {
		return backtrackAncestralTree(descendantOne, descendantTwo, depthOne-depthTwo)
	}
	return backtrackAncestralTree(descendantTwo, descendantOne, depthTwo-depthOne)
}

func getDescendantDepth(descendant *AncestralTree, topAncestor *AncestralTree) int {
	depth := 0
	for descendant != topAncestor {
		depth++
		descendant = descendant.Ancestor
	}
	return depth
}

func backtrackAncestralTree(lowerDescendant *AncestralTree, higherDescendant *AncestralTree, diff int) *AncestralTree {
	for diff > 0 {
		lowerDescendant = lowerDescendant.Ancestor
		diff--
	}
	for lowerDescendant != higherDescendant {
		lowerDescendant = lowerDescendant.Ancestor
		higherDescendant = higherDescendant.Ancestor
	}
	return lowerDescendant
}`,
		PythonSolution: `class AncestralTree:
    def __init__(self, name):
        self.name = name
        self.ancestor = None

def getYoungestCommonAncestor(topAncestor, descendantOne, descendantTwo):
    depth_one = get_descendant_depth(descendantOne, topAncestor)
    depth_two = get_descendant_depth(descendantTwo, topAncestor)
    if depth_one > depth_two:
        return backtrack_ancestral_tree(descendantOne, descendantTwo, depth_one - depth_two)
    return backtrack_ancestral_tree(descendantTwo, descendantOne, depth_two - depth_one)

def get_descendant_depth(descendant, top_ancestor):
    depth = 0
    while descendant != top_ancestor:
        depth += 1
        descendant = descendant.ancestor
    return depth

def backtrack_ancestral_tree(lower_descendant, higher_descendant, diff):
    while diff > 0:
        lower_descendant = lower_descendant.ancestor
        diff -= 1
    while lower_descendant != higher_descendant:
        lower_descendant = lower_descendant.ancestor
        higher_descendant = higher_descendant.ancestor
    return lower_descendant`,
		Explanation: `Finding the youngest common ancestor in a tree where each node only has a pointer to its parent requires bringing both nodes to the same depth first, then moving them up together until they meet. This is more efficient than storing paths or using extra space.

The algorithm first calculates the depth of both descendants by traversing up to the top ancestor. Then, it identifies which descendant is deeper and moves it up by the depth difference, bringing both to the same level. Once at the same level, both descendants are moved up simultaneously, one step at a time, until they meet at the common ancestor.

This approach is optimal because it only requires O(1) extra space (just a few variables) and visits each node at most once. The depth calculation takes O(d) time where d is the depth, and the backtracking also takes O(d) time, giving overall O(d) time complexity.

Edge cases handled include: one descendant is an ancestor of the other (the ancestor is the common ancestor), both descendants are at the same depth (no initial adjustment needed), one descendant is the top ancestor (top ancestor is the common ancestor), and descendants in different branches (algorithm finds the correct common ancestor).

An alternative approach would store the path from each descendant to the top, then find the last common node, but that requires O(d) space. Another approach would use a hash set to mark visited ancestors, but that also requires O(d) space. The depth-based approach achieves O(1) space while maintaining O(d) time.

Think of this as finding where two paths up a family tree meet: you first bring both people to the same generation level, then have them both go up one generation at a time until they reach a common ancestor.`,
	},
	{
		ID:          117,
		Title:       "Remove Islands",
		Description: "You're given a two-dimensional array of potentially unequal height and width containing only 0s and 1s. The matrix represents a two-toned image, where each 1 represents black and each 0 represents white. An island is defined as any number of 1s that are horizontally or vertically adjacent (but not diagonally adjacent) and that don't touch the border of the image. In other words, a group of horizontally or vertically adjacent 1s is an island if and only if it isn't connected to any 1s on the border of the image. Write a function that returns a modified version of the input matrix, where all of the islands are removed. You remove an island by replacing it with 0s. Naturally, you're allowed to mutate the input matrix.",
		Difficulty:  "Medium",
		Topic:       "Graphs",
		Signature:   "func removeIslands(matrix [][]int) [][]int",
		TestCases: []TestCase{
			{Input: "matrix = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 1], [0, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 1]]", Expected: "Matrix with islands removed"},
		},
		Solution: `func removeIslands(matrix [][]int) [][]int {
	for row := 0; row < len(matrix); row++ {
		for col := 0; col < len(matrix[row]); col++ {
			rowIsBorder := row == 0 || row == len(matrix)-1
			colIsBorder := col == 0 || col == len(matrix[row])-1
			isBorder := rowIsBorder || colIsBorder
			if !isBorder {
				continue
			}
			if matrix[row][col] != 1 {
				continue
			}
			changeOnesConnectedToBorderToTwos(matrix, row, col)
		}
	}
	for row := 0; row < len(matrix); row++ {
		for col := 0; col < len(matrix[row]); col++ {
			color := matrix[row][col]
			if color == 1 {
				matrix[row][col] = 0
			} else if color == 2 {
				matrix[row][col] = 1
			}
		}
	}
	return matrix
}

func changeOnesConnectedToBorderToTwos(matrix [][]int, startRow int, startCol int) {
	stack := [][]int{{startRow, startCol}}
	for len(stack) > 0 {
		currentPosition := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		currentRow, currentCol := currentPosition[0], currentPosition[1]
		matrix[currentRow][currentCol] = 2
		neighbors := getNeighbors(matrix, currentRow, currentCol)
		for _, neighbor := range neighbors {
			row, col := neighbor[0], neighbor[1]
			if matrix[row][col] != 1 {
				continue
			}
			stack = append(stack, neighbor)
		}
	}
}

func getNeighbors(matrix [][]int, row int, col int) [][]int {
	neighbors := [][]int{}
	numRows := len(matrix)
	numCols := len(matrix[row])
	if row-1 >= 0 {
		neighbors = append(neighbors, []int{row - 1, col})
	}
	if row+1 < numRows {
		neighbors = append(neighbors, []int{row + 1, col})
	}
	if col-1 >= 0 {
		neighbors = append(neighbors, []int{row, col - 1})
	}
	if col+1 < numCols {
		neighbors = append(neighbors, []int{row, col + 1})
	}
	return neighbors
}`,
		PythonSolution: `def removeIslands(matrix: List[List[int]]) -> List[List[int]]:
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            row_is_border = row == 0 or row == len(matrix) - 1
            col_is_border = col == 0 or col == len(matrix[row]) - 1
            is_border = row_is_border or col_is_border
            if not is_border:
                continue
            if matrix[row][col] != 1:
                continue
            change_ones_connected_to_border_to_twos(matrix, row, col)
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            color = matrix[row][col]
            if color == 1:
                matrix[row][col] = 0
            elif color == 2:
                matrix[row][col] = 1
    return matrix

def change_ones_connected_to_border_to_twos(matrix, start_row, start_col):
    stack = [[start_row, start_col]]
    while stack:
        current_position = stack.pop()
        current_row, current_col = current_position
        matrix[current_row][current_col] = 2
        neighbors = get_neighbors(matrix, current_row, current_col)
        for neighbor in neighbors:
            row, col = neighbor
            if matrix[row][col] != 1:
                continue
            stack.append(neighbor)

def get_neighbors(matrix, row, col):
    neighbors = []
    num_rows = len(matrix)
    num_cols = len(matrix[row])
    if row - 1 >= 0:
        neighbors.append([row - 1, col])
    if row + 1 < num_rows:
        neighbors.append([row + 1, col])
    if col - 1 >= 0:
        neighbors.append([row, col - 1])
    if col + 1 < num_cols:
        neighbors.append([row, col + 1])
    return neighbors`,
		Explanation: `This problem requires removing "islands" - groups of 1s that are not connected to the border. The key insight is that instead of finding and removing islands, we can find and mark all 1s connected to the border, then remove everything else.

The algorithm uses a two-phase approach. First, it performs DFS (or BFS) starting from all border cells containing 1s, marking all connected 1s as 2 (a temporary marker). This identifies all 1s that are part of the "mainland" (connected to border). In the second phase, it converts all remaining 1s (islands) to 0s, and converts all 2s back to 1s, restoring the mainland.

This approach is efficient because it only needs to traverse the matrix a few times: once to mark border-connected 1s, and once to clean up. By using a temporary marker (2), we can distinguish between mainland 1s and island 1s without needing additional data structures.

Edge cases handled include: matrices with no islands (all 1s connected to border, nothing removed), matrices with only islands (all 1s removed), matrices with multiple separate islands (all removed correctly), and matrices where border has no 1s (all 1s are islands, all removed).

An alternative approach would be to find each island and remove it, but that's more complex. Another approach would use union-find, but the DFS marking approach is simpler and equally efficient. The time complexity is O(w*h) where w and h are width and height, as we visit each cell at most a constant number of times. The space complexity is O(w*h) for the DFS stack in the worst case.

Think of this as marking all land connected to the coast, then removing all unmarked land (islands). The temporary marker (2) lets you distinguish between "keep" and "remove" without extra memory.`,
	},
	{
		ID:          118,
		Title:       "Cycle In Graph",
		Description: "You're given a list of edges representing an unweighted, directed graph with at least one node. Write a function that returns a boolean indicating whether or not the given graph contains a cycle.",
		Difficulty:  "Medium",
		Topic:       "Graphs",
		Signature:   "func cycleInGraph(edges [][]int) bool",
		TestCases: []TestCase{
			{Input: "edges = [[1, 3], [2, 3, 4], [0], [], [2, 5], []]", Expected: "true"},
			{Input: "edges = [[1, 2], [2], []]", Expected: "false"},
		},
		Solution: `func cycleInGraph(edges [][]int) bool {
	numberOfNodes := len(edges)
	visited := make([]bool, numberOfNodes)
	currentlyInStack := make([]bool, numberOfNodes)
	for node := 0; node < numberOfNodes; node++ {
		if visited[node] {
			continue
		}
		containsCycle := isNodeInCycle(node, edges, visited, currentlyInStack)
		if containsCycle {
			return true
		}
	}
	return false
}

func isNodeInCycle(node int, edges [][]int, visited []bool, currentlyInStack []bool) bool {
	visited[node] = true
	currentlyInStack[node] = true
	neighbors := edges[node]
	for _, neighbor := range neighbors {
		if !visited[neighbor] {
			containsCycle := isNodeInCycle(neighbor, edges, visited, currentlyInStack)
			if containsCycle {
				return true
			}
		} else if currentlyInStack[neighbor] {
			return true
		}
	}
	currentlyInStack[node] = false
	return false
}`,
		PythonSolution: `def cycleInGraph(edges: List[List[int]]) -> bool:
    number_of_nodes = len(edges)
    visited = [False] * number_of_nodes
    currently_in_stack = [False] * number_of_nodes
    for node in range(number_of_nodes):
        if visited[node]:
            continue
        contains_cycle = is_node_in_cycle(node, edges, visited, currently_in_stack)
        if contains_cycle:
            return True
    return False

def is_node_in_cycle(node, edges, visited, currently_in_stack):
    visited[node] = True
    currently_in_stack[node] = True
    neighbors = edges[node]
    for neighbor in neighbors:
        if not visited[neighbor]:
            contains_cycle = is_node_in_cycle(neighbor, edges, visited, currently_in_stack)
            if contains_cycle:
                return True
        elif currently_in_stack[neighbor]:
            return True
    currently_in_stack[node] = False
    return False`,
		Explanation: `Detecting cycles in a directed graph requires tracking not just visited nodes, but also nodes currently in the recursion stack. A cycle exists if we encounter a node that is currently being processed (in the recursion stack), indicating a back edge.

The algorithm uses DFS with two tracking mechanisms: a visited set to mark nodes we've completely processed, and a currently_in_stack set to track nodes currently in the recursion path. When we encounter a neighbor that is already visited, we check if it's in the recursion stack. If it is, we've found a back edge, indicating a cycle. When we finish processing a node (backtrack), we remove it from the recursion stack but keep it in visited.

This approach correctly distinguishes between cross edges (to already processed nodes, no cycle) and back edges (to nodes in current path, cycle). The recursion stack naturally tracks the current path, making cycle detection straightforward.

Edge cases handled include: graphs with no cycles (all nodes processed without finding back edges), graphs with self-loops (detected when node points to itself), graphs with multiple cycles (returns true on first cycle found), and disconnected graphs (checks each component).

An alternative approach would use topological sort (Kahn's algorithm), but that only works for DAGs and doesn't directly detect cycles. Another approach would use union-find, but that's for undirected graphs. The DFS with recursion stack tracking is the standard approach for cycle detection in directed graphs. The time complexity is O(v+e) where v is vertices and e is edges, and space complexity is O(v) for the recursion stack and tracking sets.

Think of this as walking through a maze: if you encounter a door you're currently holding open (node in recursion stack), you've found a cycle. If you encounter a door you've already closed (visited but not in stack), it's just a different path, not a cycle.`,
	},
	{
		ID:          119,
		Title:       "Minimum Passes Of Matrix",
		Description: "Write a function that takes in an integer matrix of potentially unequal height and width and returns the minimum number of passes required to convert all negative integers in the matrix to positive integers. A negative integer in the matrix can only be converted to a positive integer if one or more of its adjacent elements is positive. An adjacent element is an element that is to the left, to the right, above, or below the current element in the matrix. Converting a negative to a positive simply involves multiplying it by -1. Note that the 0 value is neither positive nor negative, meaning that a 0 can't help convert a negative number to a positive number. A single pass through the matrix involves converting all the negative integers that can be converted at a particular point in time.",
		Difficulty:  "Medium",
		Topic:       "Graphs",
		Signature:   "func minimumPassesOfMatrix(matrix [][]int) int",
		TestCases: []TestCase{
			{Input: "matrix = [[0, -1, -3, 2, 0], [1, -2, -5, -1, -3], [3, 0, 0, -4, -1]]", Expected: "3"},
		},
		Solution: `func minimumPassesOfMatrix(matrix [][]int) int {
	passes := convertNegatives(matrix)
	if !containsNegative(matrix) {
		return passes - 1
	}
	return -1
}

func convertNegatives(matrix [][]int) int {
	nextPassQueue := getAllPositivePositions(matrix)
	passes := 0
	for len(nextPassQueue) > 0 {
		currentPassQueue := nextPassQueue
		nextPassQueue = [][]int{}
		for len(currentPassQueue) > 0 {
			currentRow, currentCol := currentPassQueue[0][0], currentPassQueue[0][1]
			currentPassQueue = currentPassQueue[1:]
			adjacentPositions := getAdjacentPositions(currentRow, currentCol, matrix)
			for _, position := range adjacentPositions {
				row, col := position[0], position[1]
				value := matrix[row][col]
				if value < 0 {
					matrix[row][col] *= -1
					nextPassQueue = append(nextPassQueue, []int{row, col})
				}
			}
		}
		passes++
	}
	return passes
}

func getAllPositivePositions(matrix [][]int) [][]int {
	positivePositions := [][]int{}
	for row := 0; row < len(matrix); row++ {
		for col := 0; col < len(matrix[row]); col++ {
			value := matrix[row][col]
			if value > 0 {
				positivePositions = append(positivePositions, []int{row, col})
			}
		}
	}
	return positivePositions
}

func getAdjacentPositions(row int, col int, matrix [][]int) [][]int {
	adjacentPositions := [][]int{}
	if row > 0 {
		adjacentPositions = append(adjacentPositions, []int{row - 1, col})
	}
	if row < len(matrix)-1 {
		adjacentPositions = append(adjacentPositions, []int{row + 1, col})
	}
	if col > 0 {
		adjacentPositions = append(adjacentPositions, []int{row, col - 1})
	}
	if col < len(matrix[0])-1 {
		adjacentPositions = append(adjacentPositions, []int{row, col + 1})
	}
	return adjacentPositions
}

func containsNegative(matrix [][]int) bool {
	for _, row := range matrix {
		for _, value := range row {
			if value < 0 {
				return true
			}
		}
	}
	return false
}`,
		PythonSolution: `def minimumPassesOfMatrix(matrix: List[List[int]]) -> int:
    passes = convert_negatives(matrix)
    if not contains_negative(matrix):
        return passes - 1
    return -1

def convert_negatives(matrix):
    next_pass_queue = get_all_positive_positions(matrix)
    passes = 0
    while next_pass_queue:
        current_pass_queue = next_pass_queue
        next_pass_queue = []
        while current_pass_queue:
            current_row, current_col = current_pass_queue.pop(0)
            adjacent_positions = get_adjacent_positions(current_row, current_col, matrix)
            for position in adjacent_positions:
                row, col = position
                value = matrix[row][col]
                if value < 0:
                    matrix[row][col] *= -1
                    next_pass_queue.append([row, col])
        passes += 1
    return passes

def get_all_positive_positions(matrix):
    positive_positions = []
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            value = matrix[row][col]
            if value > 0:
                positive_positions.append([row, col])
    return positive_positions

def get_adjacent_positions(row, col, matrix):
    adjacent_positions = []
    if row > 0:
        adjacent_positions.append([row - 1, col])
    if row < len(matrix) - 1:
        adjacent_positions.append([row + 1, col])
    if col > 0:
        adjacent_positions.append([row, col - 1])
    if col < len(matrix[0]) - 1:
        adjacent_positions.append([row, col + 1])
    return adjacent_positions

def contains_negative(matrix):
    for row in matrix:
        for value in row:
            if value < 0:
                return True
    return False`,
		Explanation: `This problem simulates a propagation process where positive values "convert" adjacent negative values in each pass. The challenge is to find the minimum number of passes needed to convert all negatives, or determine if it's impossible.

The algorithm uses a multi-pass BFS approach. In the first pass, it identifies all positive positions and adds them to a queue. Then, in each subsequent pass, it processes all positions from the current queue, converting their adjacent negative values to positive and adding those newly converted positions to the next pass's queue. The process continues until no more negatives can be converted.

The key insight is using separate queues for each pass to ensure we count passes correctly. All positions converted in the same pass are processed together, then we move to the next pass. If after all passes there are still negative values, it means some negatives were unreachable from any positive value, so we return -1.

Edge cases handled include: matrices with no negatives (returns 0 passes), matrices with no positives (returns -1, impossible), matrices where all negatives are adjacent to positives (converted in one pass), and matrices with isolated negative regions (detected when negatives remain after all passes).

An alternative approach would use a single BFS with distance tracking, but the multi-pass approach is more intuitive for this problem. The time complexity is O(w*h) where w and h are width and height, as each cell is processed at most once per pass, and there are at most O(w*h) passes. The space complexity is O(w*h) for the queues.

Think of this as a wave of conversion spreading from positive cells: each wave (pass) converts all adjacent negatives, and the next wave starts from the newly converted cells, continuing until the wave can't reach any more negatives.`,
	},
	{
		ID:          120,
		Title:       "Two-Colorable",
		Description: "You're given a list of edges representing an unweighted, undirected graph. Write a function that returns a boolean indicating whether or not the given graph is two-colorable. A graph is two-colorable (also called bipartite) if all of the vertices can be colored with one of two colors such that no two adjacent vertices share the same color.",
		Difficulty:  "Medium",
		Topic:       "Graphs",
		Signature:   "func twoColorable(edges [][]int) bool",
		TestCases: []TestCase{
			{Input: "edges = [[1, 2], [0, 2], [0, 1]]", Expected: "false"},
			{Input: "edges = [[1], [0]]", Expected: "true"},
		},
		Solution: `func twoColorable(edges [][]int) bool {
	colors := make([]int, len(edges))
	for i := range colors {
		colors[i] = -1
	}
	for i := 0; i < len(edges); i++ {
		if colors[i] != -1 {
			continue
		}
		stack := []int{i}
		colors[i] = 0
		for len(stack) > 0 {
			node := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			for _, connection := range edges[node] {
				if colors[connection] == -1 {
					colors[connection] = 1 - colors[node]
					stack = append(stack, connection)
				} else if colors[connection] == colors[node] {
					return false
				}
			}
		}
	}
	return true
}`,
		PythonSolution: `def twoColorable(edges: List[List[int]]) -> bool:
    colors = [-1] * len(edges)
    for i in range(len(edges)):
        if colors[i] != -1:
            continue
        stack = [i]
        colors[i] = 0
        while stack:
            node = stack.pop()
            for connection in edges[node]:
                if colors[connection] == -1:
                    colors[connection] = 1 - colors[node]
                    stack.append(connection)
                elif colors[connection] == colors[node]:
                    return False
    return True`,
		Explanation: `A graph is two-colorable (bipartite) if we can color all nodes using only two colors such that no two adjacent nodes have the same color. This problem can be solved using BFS or DFS with a coloring approach.

The algorithm starts by assigning a color to the first node, then uses BFS/DFS to visit all connected nodes. For each node, it assigns the opposite color of its parent/neighbor. If we ever encounter a node that has already been colored with a color that conflicts with what we're trying to assign, the graph is not two-colorable and we return false.

This approach works because if a graph is bipartite, we can always color it by alternating colors along any path. If we encounter a conflict, it means there's an odd-length cycle, which prevents two-coloring. The algorithm processes each connected component, ensuring all nodes are checked.

Edge cases handled include: graphs with no edges (trivially two-colorable), graphs with a single node (two-colorable), disconnected graphs (each component checked separately), graphs with odd-length cycles (not two-colorable), and graphs with even-length cycles only (two-colorable).

An alternative approach would check for odd-length cycles directly, but the coloring approach is simpler and more intuitive. Another approach would use union-find, but that's more complex. The BFS/DFS coloring approach is the standard solution. The time complexity is O(v+e) where v is vertices and e is edges, as we visit each node and edge once. The space complexity is O(v) for the color tracking and BFS/DFS data structures.

Think of this as trying to color a map with two colors: if you can color it such that no adjacent regions share the same color, it's two-colorable. If you're forced to use the same color for adjacent regions, it's not.`,
	},
	{
		ID:          121,
		Title:       "Task Assignment",
		Description: "You're given an integer k representing a number of workers and an array of positive integers representing durations of tasks that must be completed by the workers. Specifically, each worker must complete two unique tasks and can only work on one task at a time. The number of tasks will always be equal to 2k such that each worker always has exactly two tasks to complete. All tasks are independent of one another and can be completed in any order. Workers will complete their assigned tasks in parallel, and the time taken to complete all tasks will be equal to the time taken to complete the longest pair of tasks. Write a function that returns the optimal assignment of tasks to each worker such that the tasks are completed as fast as possible.",
		Difficulty:  "Medium",
		Topic:       "Greedy",
		Signature:   "func taskAssignment(k int, tasks []int) [][]int",
		TestCases: []TestCase{
			{Input: "k = 3, tasks = [1, 3, 5, 3, 1, 4]", Expected: "[[4, 2], [0, 5], [3, 1]]"},
		},
		Solution: `import "sort"

type Task struct {
	Duration int
	Index    int
}

func taskAssignment(k int, tasks []int) [][]int {
	sortedTasks := make([]Task, len(tasks))
	for i, duration := range tasks {
		sortedTasks[i] = Task{Duration: duration, Index: i}
	}
	sort.Slice(sortedTasks, func(i, j int) bool {
		return sortedTasks[i].Duration < sortedTasks[j].Duration
	})
	pairedTasks := [][]int{}
	for i := 0; i < k; i++ {
		task1Idx := sortedTasks[i].Index
		task2Idx := sortedTasks[len(tasks)-1-i].Index
		pairedTasks = append(pairedTasks, []int{task1Idx, task2Idx})
	}
	return pairedTasks
}`,
		PythonSolution: `def taskAssignment(k: int, tasks: List[int]) -> List[List[int]]:
    sorted_tasks = sorted([(duration, idx) for idx, duration in enumerate(tasks)])
    paired_tasks = []
    for i in range(k):
        task1_idx = sorted_tasks[i][1]
        task2_idx = sorted_tasks[len(tasks) - 1 - i][1]
        paired_tasks.append([task1_idx, task2_idx])
    return paired_tasks`,
		Explanation: `This problem requires assigning tasks to workers such that the maximum pair sum (time for the slowest worker) is minimized. The key insight is that to minimize the maximum, we should balance the pairs by pairing the shortest tasks with the longest tasks.

The algorithm first sorts tasks while preserving their original indices (since we need to return indices, not values). Then, it pairs the shortest task with the longest task, the second shortest with the second longest, and so on. This greedy approach ensures that no pair has an unnecessarily large sum, as we're always pairing extremes together.

This pairing strategy works because if we paired two short tasks together, we'd waste a long task that would then need to be paired with another long task, creating a larger maximum. By pairing short with long, we keep all pair sums relatively balanced, minimizing the maximum.

Edge cases handled include: all tasks having the same duration (any pairing works, all pairs have same sum), tasks with very different durations (pairing extremes minimizes maximum), and cases where k=1 (only one pair, sum is fixed).

An alternative approach would try all possible pairings, but that would be exponential. The greedy sorting and pairing approach is optimal and achieves O(n log n) time for sorting and O(n) time for pairing, giving overall O(n log n) time complexity. The space complexity is O(n) for storing sorted tasks with indices.

Think of this as balancing weights on a scale: to minimize the maximum weight on any scale, you pair the lightest weight with the heaviest weight, ensuring all scales are as balanced as possible.`,
	},
	{
		ID:          122,
		Title:       "Valid Starting City",
		Description: "Imagine you have a set of cities that are laid out in a circle, connected by a circular road running clockwise. Each city has a gas station that provides gallons of fuel, and each city is some distance away from the next city. You have a car that can drive some number of miles per gallon of fuel, and your goal is to pick a starting city such that you can fill up your car with that city's fuel, drive to the next city, refill up your car with that city's fuel, drive to the next city, and so on and so forth until you return back to the starting city with 0 or more gallons of fuel left. A city is said to be a valid starting city if you're able to travel around the circuit once without running out of gas. Write a function that returns the index of the valid starting city. There will always be exactly one valid starting city.",
		Difficulty:  "Medium",
		Topic:       "Greedy",
		Signature:   "func validStartingCity(distances []int, fuel []int, mpg int) int",
		TestCases: []TestCase{
			{Input: "distances = [5, 25, 15, 10, 15], fuel = [1, 2, 1, 0, 3], mpg = 10", Expected: "4"},
		},
		Solution: `func validStartingCity(distances []int, fuel []int, mpg int) int {
	numberOfCities := len(distances)
	milesRemaining := 0
	indexOfStartingCityCandidate := 0
	milesRemainingAtStartingCityCandidate := 0
	for cityIdx := 1; cityIdx < numberOfCities; cityIdx++ {
		fuelFromPreviousCity := fuel[cityIdx-1]
		distanceFromPreviousCity := distances[cityIdx-1]
		milesRemaining += fuelFromPreviousCity*mpg - distanceFromPreviousCity
		if milesRemaining < milesRemainingAtStartingCityCandidate {
			milesRemainingAtStartingCityCandidate = milesRemaining
			indexOfStartingCityCandidate = cityIdx
		}
	}
	return indexOfStartingCityCandidate
}`,
		PythonSolution: `def validStartingCity(distances: List[int], fuel: List[int], mpg: int) -> int:
    number_of_cities = len(distances)
    miles_remaining = 0
    index_of_starting_city_candidate = 0
    miles_remaining_at_starting_city_candidate = 0
    for city_idx in range(1, number_of_cities):
        fuel_from_previous_city = fuel[city_idx - 1]
        distance_from_previous_city = distances[city_idx - 1]
        miles_remaining += fuel_from_previous_city * mpg - distance_from_previous_city
        if miles_remaining < miles_remaining_at_starting_city_candidate:
            miles_remaining_at_starting_city_candidate = miles_remaining
            index_of_starting_city_candidate = city_idx
    return index_of_starting_city_candidate`,
		Explanation: `This problem requires finding the starting city for a circular route where you have limited fuel and must visit all cities. The key insight is that the starting city must be the one where, if you start there, you never run out of fuel (miles remaining never goes negative).

The algorithm tracks the miles remaining as you travel from city to city. Starting from city 0, it calculates how many miles would remain at each city. The city where miles remaining is minimum is the optimal starting point, because starting from any city before that would result in negative miles (running out of fuel) at some point.

This approach works because if you start at the city with minimum miles remaining, you ensure you have the maximum "buffer" of fuel. Any city before the minimum would require more fuel than available at some point in the journey. The circular nature means we can start anywhere, but starting at the minimum ensures we never go negative.

Edge cases handled include: routes where one city has much more fuel than needed (that city is the start), routes where fuel is exactly enough (any city works, but minimum is optimal), routes with negative fuel differences (some cities consume more than they provide), and routes with a single city (that city is the start).

An alternative approach would try each city as a starting point and simulate the journey, but that's O(n²) time. The current approach achieves O(n) time by calculating miles remaining in a single pass and finding the minimum. The space complexity is O(1) as we only track the minimum and current miles.

Think of this as finding the lowest point in a circular track: if you start at the lowest point going uphill, you'll have the most fuel buffer. Starting anywhere else would mean going downhill first, potentially running out of fuel before reaching the low point.`,
	},
	{
		ID:          123,
		Title:       "Min Heap Construction",
		Description: "Implement a MinHeap class that supports: Building a Min Heap from an input array of integers, Inserting integers in the heap, Removing the heap's minimum / root value, Peeking at the heap's minimum / root value, Sifting integers up and down the heap as needed to maintain the heap property.",
		Difficulty:  "Medium",
		Topic:       "Heaps",
		Signature:   "type MinHeap struct { heap []int }\nfunc NewMinHeap(array []int) *MinHeap\nfunc (h *MinHeap) Insert(value int)\nfunc (h *MinHeap) Remove() int\nfunc (h *MinHeap) Peek() int",
		TestCases: []TestCase{
			{Input: "Heap operations", Expected: "Correct heap structure"},
		},
		Solution: `type MinHeap struct {
	heap []int
}

func NewMinHeap(array []int) *MinHeap {
	heap := &MinHeap{heap: array}
	heap.buildHeap(array)
	return heap
}

func (h *MinHeap) buildHeap(array []int) {
	firstParentIdx := (len(array) - 2) / 2
	for currentIdx := firstParentIdx; currentIdx >= 0; currentIdx-- {
		h.siftDown(currentIdx, len(array)-1)
	}
}

func (h *MinHeap) siftDown(currentIdx int, endIdx int) {
	childOneIdx := currentIdx*2 + 1
	for childOneIdx <= endIdx {
		childTwoIdx := -1
		if currentIdx*2+2 <= endIdx {
			childTwoIdx = currentIdx*2 + 2
		}
		idxToSwap := childOneIdx
		if childTwoIdx != -1 && h.heap[childTwoIdx] < h.heap[childOneIdx] {
			idxToSwap = childTwoIdx
		}
		if h.heap[idxToSwap] < h.heap[currentIdx] {
			h.swap(currentIdx, idxToSwap)
			currentIdx = idxToSwap
			childOneIdx = currentIdx*2 + 1
		} else {
			return
		}
	}
}

func (h *MinHeap) siftUp(currentIdx int) {
	parentIdx := (currentIdx - 1) / 2
	for currentIdx > 0 && h.heap[currentIdx] < h.heap[parentIdx] {
		h.swap(currentIdx, parentIdx)
		currentIdx = parentIdx
		parentIdx = (currentIdx - 1) / 2
	}
}

func (h *MinHeap) Peek() int {
	if len(h.heap) == 0 {
		return -1
	}
	return h.heap[0]
}

func (h *MinHeap) Remove() int {
	h.swap(0, len(h.heap)-1)
	valueToRemove := h.heap[len(h.heap)-1]
	h.heap = h.heap[:len(h.heap)-1]
	h.siftDown(0, len(h.heap)-1)
	return valueToRemove
}

func (h *MinHeap) Insert(value int) {
	h.heap = append(h.heap, value)
	h.siftUp(len(h.heap) - 1)
}

func (h *MinHeap) swap(i int, j int) {
	h.heap[i], h.heap[j] = h.heap[j], h.heap[i]
}`,
		PythonSolution: `class MinHeap:
    def __init__(self, array):
        self.heap = self.build_heap(array)
    
    def build_heap(self, array):
        first_parent_idx = (len(array) - 2) // 2
        for current_idx in range(first_parent_idx, -1, -1):
            self.sift_down(current_idx, len(array) - 1, array)
        return array
    
    def sift_down(self, current_idx, end_idx, heap):
        child_one_idx = current_idx * 2 + 1
        while child_one_idx <= end_idx:
            child_two_idx = current_idx * 2 + 2 if current_idx * 2 + 2 <= end_idx else -1
            if child_two_idx != -1 and heap[child_two_idx] < heap[child_one_idx]:
                idx_to_swap = child_two_idx
            else:
                idx_to_swap = child_one_idx
            if heap[idx_to_swap] < heap[current_idx]:
                self.swap(current_idx, idx_to_swap, heap)
                current_idx = idx_to_swap
                child_one_idx = current_idx * 2 + 1
            else:
                return
    
    def sift_up(self, current_idx, heap):
        parent_idx = (current_idx - 1) // 2
        while current_idx > 0 and heap[current_idx] < heap[parent_idx]:
            self.swap(current_idx, parent_idx, heap)
            current_idx = parent_idx
            parent_idx = (current_idx - 1) // 2
    
    def peek(self):
        return self.heap[0] if self.heap else -1
    
    def remove(self):
        self.swap(0, len(self.heap) - 1, self.heap)
        value_to_remove = self.heap.pop()
        self.sift_down(0, len(self.heap) - 1, self.heap)
        return value_to_remove
    
    def insert(self, value):
        self.heap.append(value)
        self.sift_up(len(self.heap) - 1, self.heap)
    
    def swap(self, i, j, heap):
        heap[i], heap[j] = heap[j], heap[i]`,
		Explanation: `A min-heap is a complete binary tree where each node is smaller than or equal to its children. Implementing a min-heap requires maintaining the heap property through sift-up and sift-down operations.

The algorithm builds the heap from an array using a bottom-up approach: starting from the last parent node (at index (n-2)/2), it sifts down each node to its correct position. This is more efficient than inserting elements one by one, as it achieves O(n) build time instead of O(n log n). For insertion, it adds the new element at the end and sifts it up until the heap property is restored. For removal, it swaps the root with the last element, removes it, then sifts down the new root to restore the heap property.

The sift-down operation compares a node with its children and swaps with the smaller child if the node is larger, continuing down the tree. The sift-up operation compares a node with its parent and swaps if the node is smaller, continuing up the tree. These operations maintain the min-heap property efficiently.

Edge cases handled include: empty heap (Peek and Remove return sentinel values), single-element heap (no sifting needed), heap with duplicate values (handled correctly by comparison), and heap operations on already-sorted arrays (sift operations handle correctly).

An alternative approach would build the heap by inserting elements one by one, but that takes O(n log n) time. The bottom-up build approach is optimal. The time complexity is O(n) for building, O(log n) for insert and remove, and O(1) for peek. The space complexity is O(1) extra space beyond the array itself.

Think of a min-heap as a tournament bracket where the winner (minimum) is at the top: when you remove the winner, you promote the next best competitor, and when you add a new competitor, you insert them at the bottom and let them compete their way up.`,
	},
	{
		ID:          124,
		Title:       "Linked List Construction",
		Description: "Write a DoublyLinkedList class that has a head and a tail, both of which point to either a linked list Node or None / null. The class should support: Setting the head and tail of the linked list, Inserting nodes before and after other nodes as well as at given positions, Removing given nodes and removing nodes with given values, Searching for nodes with given values.",
		Difficulty:  "Medium",
		Topic:       "Linked Lists",
		Signature:   "type Node struct { Value int; Prev *Node; Next *Node }\ntype DoublyLinkedList struct { Head *Node; Tail *Node }\nfunc (ll *DoublyLinkedList) SetHead(node *Node)\nfunc (ll *DoublyLinkedList) SetTail(node *Node)\nfunc (ll *DoublyLinkedList) InsertBefore(node *Node, nodeToInsert *Node)\nfunc (ll *DoublyLinkedList) InsertAfter(node *Node, nodeToInsert *Node)\nfunc (ll *DoublyLinkedList) InsertAtPosition(position int, nodeToInsert *Node)\nfunc (ll *DoublyLinkedList) RemoveNodesWithValue(value int)\nfunc (ll *DoublyLinkedList) Remove(node *Node)\nfunc (ll *DoublyLinkedList) ContainsNodeWithValue(value int) bool",
		TestCases: []TestCase{
			{Input: "Linked list operations", Expected: "Correct doubly linked list"},
		},
		Solution: `type Node struct {
	Value int
	Prev  *Node
	Next  *Node
}

type DoublyLinkedList struct {
	Head *Node
	Tail *Node
}

func (ll *DoublyLinkedList) SetHead(node *Node) {
	if ll.Head == nil {
		ll.Head = node
		ll.Tail = node
		return
	}
	ll.InsertBefore(ll.Head, node)
}

func (ll *DoublyLinkedList) SetTail(node *Node) {
	if ll.Tail == nil {
		ll.SetHead(node)
		return
	}
	ll.InsertAfter(ll.Tail, node)
}

func (ll *DoublyLinkedList) InsertBefore(node *Node, nodeToInsert *Node) {
	if nodeToInsert == ll.Head && nodeToInsert == ll.Tail {
		return
	}
	ll.Remove(nodeToInsert)
	nodeToInsert.Prev = node.Prev
	nodeToInsert.Next = node
	if node.Prev == nil {
		ll.Head = nodeToInsert
	} else {
		node.Prev.Next = nodeToInsert
	}
	node.Prev = nodeToInsert
}

func (ll *DoublyLinkedList) InsertAfter(node *Node, nodeToInsert *Node) {
	if nodeToInsert == ll.Head && nodeToInsert == ll.Tail {
		return
	}
	ll.Remove(nodeToInsert)
	nodeToInsert.Prev = node
	nodeToInsert.Next = node.Next
	if node.Next == nil {
		ll.Tail = nodeToInsert
	} else {
		node.Next.Prev = nodeToInsert
	}
	node.Next = nodeToInsert
}

func (ll *DoublyLinkedList) InsertAtPosition(position int, nodeToInsert *Node) {
	if position == 1 {
		ll.SetHead(nodeToInsert)
		return
	}
	node := ll.Head
	currentPosition := 1
	for node != nil && currentPosition != position {
		node = node.Next
		currentPosition++
	}
	if node != nil {
		ll.InsertBefore(node, nodeToInsert)
	} else {
		ll.SetTail(nodeToInsert)
	}
}

func (ll *DoublyLinkedList) RemoveNodesWithValue(value int) {
	node := ll.Head
	for node != nil {
		nodeToRemove := node
		node = node.Next
		if nodeToRemove.Value == value {
			ll.Remove(nodeToRemove)
		}
	}
}

func (ll *DoublyLinkedList) Remove(node *Node) {
	if node == ll.Head {
		ll.Head = ll.Head.Next
	}
	if node == ll.Tail {
		ll.Tail = ll.Tail.Prev
	}
	removeNodeBindings(node)
}

func removeNodeBindings(node *Node) {
	if node.Prev != nil {
		node.Prev.Next = node.Next
	}
	if node.Next != nil {
		node.Next.Prev = node.Prev
	}
	node.Prev = nil
	node.Next = nil
}

func (ll *DoublyLinkedList) ContainsNodeWithValue(value int) bool {
	node := ll.Head
	for node != nil && node.Value != value {
		node = node.Next
	}
	return node != nil
}`,
		PythonSolution: `class Node:
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def setHead(self, node):
        if self.head is None:
            self.head = node
            self.tail = node
            return
        self.insertBefore(self.head, node)
    
    def setTail(self, node):
        if self.tail is None:
            self.setHead(node)
            return
        self.insertAfter(self.tail, node)
    
    def insertBefore(self, node, nodeToInsert):
        if nodeToInsert == self.head and nodeToInsert == self.tail:
            return
        self.remove(nodeToInsert)
        nodeToInsert.prev = node.prev
        nodeToInsert.next = node
        if node.prev is None:
            self.head = nodeToInsert
        else:
            node.prev.next = nodeToInsert
        node.prev = nodeToInsert
    
    def insertAfter(self, node, nodeToInsert):
        if nodeToInsert == self.head and nodeToInsert == self.tail:
            return
        self.remove(nodeToInsert)
        nodeToInsert.prev = node
        nodeToInsert.next = node.next
        if node.next is None:
            self.tail = nodeToInsert
        else:
            node.next.prev = nodeToInsert
        node.next = nodeToInsert
    
    def insertAtPosition(self, position, nodeToInsert):
        if position == 1:
            self.setHead(nodeToInsert)
            return
        node = self.head
        current_position = 1
        while node is not None and current_position != position:
            node = node.next
            current_position += 1
        if node is not None:
            self.insertBefore(node, nodeToInsert)
        else:
            self.setTail(nodeToInsert)
    
    def removeNodesWithValue(self, value):
        node = self.head
        while node is not None:
            nodeToRemove = node
            node = node.next
            if nodeToRemove.value == value:
                self.remove(nodeToRemove)
    
    def remove(self, node):
        if node == self.head:
            self.head = self.head.next
        if node == self.tail:
            self.tail = self.tail.prev
        self.removeNodeBindings(node)
    
    def removeNodeBindings(self, node):
        if node.prev is not None:
            node.prev.next = node.next
        if node.next is not None:
            node.next.prev = node.prev
        node.prev = None
        node.next = None
    
    def containsNodeWithValue(self, value):
        node = self.head
        while node is not None and node.value != value:
            node = node.next
        return node is not None`,
		Explanation: `A doubly linked list allows traversal in both directions by maintaining pointers to both the next and previous nodes. Implementing operations efficiently requires careful pointer management and handling of edge cases.

The implementation maintains head and tail pointers for O(1) access to both ends. For insertion operations, we update four pointers: the new node's prev and next, the adjacent nodes' next and prev pointers. For removal operations, we update the prev and next pointers of adjacent nodes, then nullify the removed node's pointers. The key is ensuring all pointer updates happen in the correct order to avoid breaking the list structure.

Edge cases require special handling: inserting at head (update head pointer), inserting at tail (update tail pointer), removing head (update head to next node), removing tail (update tail to previous node), and operations on empty lists (check for nil pointers).

This data structure provides O(1) insertion and removal at both ends, which is more efficient than arrays for these operations. However, searching requires O(n) time as we must traverse the list. The space complexity is O(1) per operation, with O(n) total space for storing n nodes.

An alternative would be a singly linked list, but that only allows forward traversal. Another alternative would be arrays, but insertion/removal at ends requires O(n) time. The doubly linked list is optimal for operations requiring bidirectional traversal and efficient end operations.

Think of this as a two-way street: each house (node) has doors to both the previous and next houses, allowing you to walk in either direction efficiently.`,
	},
	{
		ID:          125,
		Title:       "Remove Kth Node From End",
		Description: "Write a function that takes in the head of a Singly Linked List and an integer k and removes the kth node from the end of the list. The removal should be done in place, meaning that the original data structure should be mutated (no new structure should be created). Furthermore, the input head of the linked list should remain the head of the linked list after the removal is done, even if the head is the node that's supposed to be removed. In other words, if the head is the node that's supposed to be removed, your function should simply mutate its value and next pointer.",
		Difficulty:  "Medium",
		Topic:       "Linked Lists",
		Signature:   "func removeKthNodeFromEnd(head *ListNode, k int)",
		TestCases: []TestCase{
			{Input: "head = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], k = 4", Expected: "[0, 1, 2, 3, 4, 5, 7, 8, 9]"},
		},
		Solution: `func removeKthNodeFromEnd(head *ListNode, k int) {
	counter := 1
	first := head
	second := head
	for counter <= k {
		second = second.Next
		counter++
	}
	if second == nil {
		head.Value = head.Next.Value
		head.Next = head.Next.Next
		return
	}
	for second.Next != nil {
		second = second.Next
		first = first.Next
	}
	first.Next = first.Next.Next
}`,
		PythonSolution: `def removeKthNodeFromEnd(head: Optional[ListNode], k: int) -> None:
    counter = 1
    first = head
    second = head
    while counter <= k:
        second = second.next
        counter += 1
    if second is None:
        head.val = head.next.val
        head.next = head.next.next
        return
    while second.next is not None:
        second = second.next
        first = first.next
    first.next = first.next.next`,
		Explanation: `Removing the kth node from the end requires finding that node efficiently. The two-pointer technique allows us to do this in a single pass without knowing the list length beforehand.

The algorithm uses two pointers that start at the head. First, it moves the second pointer k steps ahead. Then, it moves both pointers together until the second pointer reaches the end. At this point, the first pointer is positioned just before the node to be removed (the kth from the end). We then remove that node by updating the first pointer's next to skip the target node.

A special case occurs when k equals the list length: after moving the second pointer k steps, it becomes nil, indicating we need to remove the head. In this case, we can't use the normal removal method (since there's no node before the head), so we copy the value and next pointer from the second node to the head, effectively removing the head node.

Edge cases handled include: removing the head node (k equals list length, handled specially), removing the tail node (k equals 1, normal removal works), removing a middle node (standard case), and lists with a single node (k must be 1, head removal case).

An alternative approach would be to traverse the list twice: once to find the length, then again to find the (length-k)th node. However, the two-pointer approach is more efficient, achieving O(n) time in a single pass with O(1) space.

Think of this as having two people walking: one starts k steps ahead, and when the leader reaches the end, the follower is exactly k steps from the end, pointing to the node that needs to be removed.`,
	},
	{
		ID:          126,
		Title:       "Sum of Linked Lists",
		Description: "You're given two Linked Lists of potentially unequal length. Each Linked List represents a non-negative integer, where each node in the Linked List is a digit of that integer, and the first node in each Linked List always represents the least significant digit of the integer. Write a function that returns the head of a new Linked List that represents the sum of the integers represented by the two input Linked Lists.",
		Difficulty:  "Medium",
		Topic:       "Linked Lists",
		Signature:   "func sumOfLinkedLists(linkedListOne *ListNode, linkedListTwo *ListNode) *ListNode",
		TestCases: []TestCase{
			{Input: "linkedListOne = [2, 4, 7, 1], linkedListTwo = [9, 4, 5]", Expected: "[1, 9, 2, 2]"},
		},
		Solution: `func sumOfLinkedLists(linkedListOne *ListNode, linkedListTwo *ListNode) *ListNode {
	newLinkedListHeadPointer := &ListNode{Val: 0}
	currentNode := newLinkedListHeadPointer
	carry := 0
	nodeOne := linkedListOne
	nodeTwo := linkedListTwo
	for nodeOne != nil || nodeTwo != nil || carry != 0 {
		valueOne := 0
		if nodeOne != nil {
			valueOne = nodeOne.Val
		}
		valueTwo := 0
		if nodeTwo != nil {
			valueTwo = nodeTwo.Val
		}
		sumOfValues := valueOne + valueTwo + carry
		newValue := sumOfValues % 10
		carry = sumOfValues / 10
		newNode := &ListNode{Val: newValue}
		currentNode.Next = newNode
		currentNode = newNode
		if nodeOne != nil {
			nodeOne = nodeOne.Next
		}
		if nodeTwo != nil {
			nodeTwo = nodeTwo.Next
		}
	}
	return newLinkedListHeadPointer.Next
}`,
		PythonSolution: `def sumOfLinkedLists(linkedListOne: Optional[ListNode], linkedListTwo: Optional[ListNode]) -> Optional[ListNode]:
    new_linked_list_head_pointer = ListNode(0)
    current_node = new_linked_list_head_pointer
    carry = 0
    node_one = linkedListOne
    node_two = linkedListTwo
    while node_one is not None or node_two is not None or carry != 0:
        value_one = node_one.val if node_one is not None else 0
        value_two = node_two.val if node_two is not None else 0
        sum_of_values = value_one + value_two + carry
        new_value = sum_of_values % 10
        carry = sum_of_values // 10
        new_node = ListNode(new_value)
        current_node.next = new_node
        current_node = new_node
        node_one = node_one.next if node_one is not None else None
        node_two = node_two.next if node_two is not None else None
    return new_linked_list_head_pointer.next`,
		Explanation: `Adding two numbers represented as linked lists (with least significant digit first) requires simulating manual addition with carry propagation. The algorithm processes both lists digit by digit, handling cases where lists have different lengths.

The algorithm uses a dummy head node to simplify list construction, then iterates through both lists simultaneously. For each position, it adds the digits from both lists (using 0 if a list has been exhausted) plus any carry from the previous addition. The sum is split into a new digit (sum % 10) and a carry (sum / 10). A new node is created with the new digit and appended to the result list.

The loop continues until both lists are exhausted and there's no remaining carry. This ensures we handle cases where the final addition produces an extra digit (e.g., 5 + 5 = 10, requiring a new node for the 1).

Edge cases handled include: lists of different lengths (shorter list treated as having leading zeros), final carry producing an extra digit (loop continues until carry is 0), one or both lists being empty (handled by nil checks), and lists representing zero (returns list with single 0 node).

An alternative approach would convert lists to integers, add them, then convert back, but that fails for very large numbers due to integer overflow. The digit-by-digit approach handles arbitrarily large numbers. The time complexity is O(max(n,m)) where n and m are the list lengths, and space complexity is O(max(n,m)) for the result list.

Think of this as adding numbers on paper: you add digits column by column from right to left, carrying over when the sum exceeds 9, and you might need an extra column if the final sum has a carry.`,
	},
	{
		ID:          127,
		Title:       "Merging Linked Lists",
		Description: "Write a function that takes in the heads of two Singly Linked Lists that are in sorted order, respectively. The function should merge the lists in place (i.e., it shouldn't create a brand new list) and return the head of the merged list; the merged list should be in sorted order.",
		Difficulty:  "Medium",
		Topic:       "Linked Lists",
		Signature:   "func mergingLinkedLists(linkedListOne *ListNode, linkedListTwo *ListNode) *ListNode",
		TestCases: []TestCase{
			{Input: "linkedListOne = [2, 6, 7, 8], linkedListTwo = [1, 3, 4, 5, 9, 10]", Expected: "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"},
		},
		Solution: `func mergingLinkedLists(linkedListOne *ListNode, linkedListTwo *ListNode) *ListNode {
	recursiveMerge(linkedListOne, linkedListTwo, nil)
	if linkedListOne.Val < linkedListTwo.Val {
		return linkedListOne
	}
	return linkedListTwo
}

func recursiveMerge(p1 *ListNode, p2 *ListNode, p1Prev *ListNode) {
	if p1 == nil {
		p1Prev.Next = p2
		return
	}
	if p2 == nil {
		return
	}
	if p1.Val < p2.Val {
		recursiveMerge(p1.Next, p2, p1)
	} else {
		if p1Prev != nil {
			p1Prev.Next = p2
		}
		newP2 := p2.Next
		p2.Next = p1
		recursiveMerge(p1, newP2, p2)
	}
}`,
		PythonSolution: `def mergingLinkedLists(linkedListOne: Optional[ListNode], linkedListTwo: Optional[ListNode]) -> Optional[ListNode]:
    recursive_merge(linkedListOne, linkedListTwo, None)
    if linkedListOne.val < linkedListTwo.val:
        return linkedListOne
    return linkedListTwo

def recursive_merge(p1, p2, p1_prev):
    if p1 is None:
        p1_prev.next = p2
        return
    if p2 is None:
        return
    if p1.val < p2.val:
        recursive_merge(p1.next, p2, p1)
    else:
        if p1_prev is not None:
            p1_prev.next = p2
        new_p2 = p2.next
        p2.next = p1
        recursive_merge(p1, new_p2, p2)`,
		Explanation: `Merging two sorted linked lists requires maintaining the sorted order while combining nodes from both lists. The recursive approach elegantly handles this by comparing nodes and linking them in order.

The algorithm recursively compares the current nodes from both lists. The smaller node is kept in its position, and we recurse with the next node from that list and the same node from the other list. A previous pointer is maintained to correctly link nodes as we merge. When one list is exhausted, we link the remaining nodes from the other list.

This recursive approach naturally handles the merging process: at each step, we choose the smaller value and continue merging the rest. The base cases occur when one list becomes empty, at which point we simply attach the remaining nodes from the other list.

Edge cases handled include: one list being empty (return the other list), lists of different lengths (remaining nodes attached correctly), lists with duplicate values (either can be chosen first), and lists where all values in one list are smaller than the other (one list completely merged before the other).

An alternative iterative approach would use a dummy head and two pointers, which is more space-efficient (O(1) vs O(n+m) for recursion stack) but less elegant. The recursive approach is more intuitive and easier to understand. The time complexity is O(n+m) where n and m are the list lengths, as we visit each node once.

Think of this as merging two sorted decks of cards: you always take the smaller top card, place it in the merged deck, and continue with the remaining cards, naturally maintaining sorted order.`,
	},
	{
		ID:          128,
		Title:       "Shift Linked List",
		Description: "Write a function that takes in the head of a Singly Linked List and an integer k, shifts the list in place (i.e., doesn't create a brand new list) by k positions, and returns its new head. Shifting a Linked List means moving its nodes forward or backward and wrapping them around the list where appropriate. For example, shifting a Linked List forward by one position would make its tail become the new head of the linked list. Whether nodes are moved forward or backward is determined by whether k is positive or negative.",
		Difficulty:  "Medium",
		Topic:       "Linked Lists",
		Signature:   "func shiftLinkedList(head *ListNode, k int) *ListNode",
		TestCases: []TestCase{
			{Input: "head = [0, 1, 2, 3, 4, 5], k = 2", Expected: "[4, 5, 0, 1, 2, 3]"},
		},
		Solution: `func shiftLinkedList(head *ListNode, k int) *ListNode {
	listLength := 1
	listTail := head
	for listTail.Next != nil {
		listTail = listTail.Next
		listLength++
	}
	offset := abs(k) % listLength
	if offset == 0 {
		return head
	}
	newTailPosition := listLength - offset
	if k <= 0 {
		newTailPosition = offset
	}
	newTail := head
	for i := 1; i < newTailPosition; i++ {
		newTail = newTail.Next
	}
	newHead := newTail.Next
	newTail.Next = nil
	listTail.Next = head
	return newHead
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}`,
		PythonSolution: `def shiftLinkedList(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    list_length = 1
    list_tail = head
    while list_tail.next is not None:
        list_tail = list_tail.next
        list_length += 1
    offset = abs(k) % list_length
    if offset == 0:
        return head
    new_tail_position = list_length - offset
    if k <= 0:
        new_tail_position = offset
    new_tail = head
    for i in range(1, new_tail_position):
        new_tail = new_tail.next
    new_head = new_tail.next
    new_tail.next = None
    list_tail.next = head
    return new_head`,
		Explanation: `Shifting a linked list by k positions requires finding the new head and tail positions, then reorganizing the list connections. The key insight is that shifting by k is equivalent to moving the last k nodes to the front.

The algorithm first finds the list length and the tail node by traversing the list. Then it calculates the new tail position: since we're shifting right by k, the new tail will be at position (length - k) from the start. We traverse to this new tail position, break the list there (set newTail.Next to nil), and connect the old tail to the original head, making the node after newTail the new head.

This approach handles the circular nature of the shift: the last k nodes become the first k nodes, and the first (length-k) nodes become the last nodes. By breaking and reconnecting at the calculated position, we achieve the shift efficiently.

Edge cases handled include: k equals 0 (no shift needed, return original head), k equals list length (full rotation, returns original head), k greater than list length (use modulo to reduce to valid range), and lists with a single node (any shift returns the same list).

An alternative approach would create a new list by copying nodes, but that requires O(n) extra space. The current approach achieves the shift in-place with O(1) extra space. The time complexity is O(n) for finding length and new tail position, where n is the list length.

Think of this as rotating a circular list: you find where to "cut" the circle (at position length-k), then reconnect it starting from that cut point, making that point the new beginning.`,
	},
	{
		ID:          129,
		Title:       "Interweaving Strings",
		Description: "Write a function that takes in three strings and returns a boolean representing whether the third string can be formed by interweaving the first two strings. To interweave strings means to merge them by alternating their letters without any specific pattern. For instance, the strings \"abc\" and \"123\" can be interwoven as \"a1b2c3\", as \"abc123\", and as \"ab1c23\" (this list is nonexhaustive).",
		Difficulty:  "Medium",
		Topic:       "Dynamic Programming",
		Signature:   "func interweavingStrings(one string, two string, three string) bool",
		TestCases: []TestCase{
			{Input: "one = \"algoexpert\", two = \"your-dream-job\", three = \"your-algodream-expertjob\"", Expected: "false"},
			{Input: "one = \"aabcc\", two = \"dbbca\", three = \"aadbbcbcac\"", Expected: "true"},
		},
		Solution: `func interweavingStrings(one string, two string, three string) bool {
	if len(three) != len(one)+len(two) {
		return false
	}
	cache := make(map[string]bool)
	return areInterwoven(one, two, three, 0, 0, cache)
}

func areInterwoven(one string, two string, three string, i int, j int, cache map[string]bool) bool {
	key := string(rune(i)) + "," + string(rune(j))
	if val, found := cache[key]; found {
		return val
	}
	k := i + j
	if k == len(three) {
		return true
	}
	if i < len(one) && one[i] == three[k] {
		cache[key] = areInterwoven(one, two, three, i+1, j, cache)
		if cache[key] {
			return true
		}
	}
	if j < len(two) && two[j] == three[k] {
		cache[key] = areInterwoven(one, two, three, i, j+1, cache)
		return cache[key]
	}
	cache[key] = false
	return false
}`,
		PythonSolution: `def interweavingStrings(one: str, two: str, three: str) -> bool:
    if len(three) != len(one) + len(two):
        return False
    cache = {}
    return are_interwoven(one, two, three, 0, 0, cache)

def are_interwoven(one, two, three, i, j, cache):
    key = f"{i},{j}"
    if key in cache:
        return cache[key]
    k = i + j
    if k == len(three):
        return True
    if i < len(one) and one[i] == three[k]:
        cache[key] = are_interwoven(one, two, three, i + 1, j, cache)
        if cache[key]:
            return True
    if j < len(two) and two[j] == three[k]:
        cache[key] = are_interwoven(one, two, three, i, j + 1, cache)
        return cache[key]
    cache[key] = False
    return False`,
		Explanation: `Determining if a string can be formed by interweaving two other strings requires checking all possible ways to combine characters from both strings. This is a classic dynamic programming problem that can be solved with memoized recursion.

The algorithm uses memoized recursion to explore all possible interweavings. For each position in the target string, it tries matching from both source strings. If a character from source one matches, it recurses with the next position in source one. If a character from source two matches, it recurses with the next position in source two. The memoization cache stores results for (i, j) pairs, where i and j are positions in the two source strings, preventing redundant calculations.

This approach efficiently explores the solution space by caching subproblem results. The key insight is that if we've already computed whether strings one[i:] and two[j:] can form three[k:], we don't need to recompute it.

Edge cases handled include: one source string being empty (target must match the other source exactly), both source strings being empty (target must also be empty), target string longer than sum of sources (impossible, return false), and target string shorter than either source (impossible if characters don't match).

An alternative iterative DP approach would use a 2D table, but the recursive approach is more intuitive. The time complexity is O(n*m) where n and m are the lengths of the two source strings, as we compute each (i, j) pair once. The space complexity is O(n*m) for the memoization cache and recursion stack.

Think of this as trying to form a word by alternately taking letters from two words: at each step, you can take the next letter from either word, and you need to check if this leads to the target word.`,
	},
	{
		ID:          130,
		Title:       "Solve Sudoku",
		Description: "You're given a two-dimensional array that represents a 9x9 partially filled Sudoku board. Write a function that returns the solved Sudoku board. Sudoku is solved when all of the empty cells are filled such that: Each row contains digits 1-9 with no repeats, Each column contains digits 1-9 with no repeats, Each 3x3 subgrid contains digits 1-9 with no repeats.",
		Difficulty:  "Medium",
		Topic:       "Backtracking",
		Signature:   "func solveSudoku(board [][]int) [][]int",
		TestCases: []TestCase{
			{Input: "board = [[7, 8, 0, 4, 0, 0, 1, 2, 0], [6, 0, 0, 0, 7, 5, 0, 0, 9], [0, 0, 0, 6, 0, 1, 0, 7, 8], [0, 0, 7, 0, 4, 0, 2, 6, 0], [0, 0, 1, 0, 5, 0, 9, 3, 0], [9, 0, 4, 0, 6, 0, 0, 0, 5], [0, 7, 0, 3, 0, 0, 0, 1, 2], [1, 2, 0, 0, 0, 7, 4, 0, 0], [0, 4, 9, 2, 0, 6, 0, 0, 7]]", Expected: "Solved board"},
		},
		Solution: `func solveSudoku(board [][]int) [][]int {
	solvePartialSudoku(0, 0, board)
	return board
}

func solvePartialSudoku(row int, col int, board [][]int) bool {
	currentRow := row
	currentCol := col
	if currentCol == len(board[currentRow]) {
		currentRow++
		currentCol = 0
		if currentRow == len(board) {
			return true
		}
	}
	if board[currentRow][currentCol] == 0 {
		return tryDigitsAtPosition(currentRow, currentCol, board)
	}
	return solvePartialSudoku(currentRow, currentCol+1, board)
}

func tryDigitsAtPosition(row int, col int, board [][]int) bool {
	for digit := 1; digit < 10; digit++ {
		if isValidAtPosition(digit, row, col, board) {
			board[row][col] = digit
			if solvePartialSudoku(row, col+1, board) {
				return true
			}
		}
	}
	board[row][col] = 0
	return false
}

func isValidAtPosition(value int, row int, col int, board [][]int) bool {
	rowIsValid := !valueInRow(board, row, value)
	colIsValid := !valueInCol(board, col, value)
	subgridIsValid := !valueInSubgrid(board, row, col, value)
	return rowIsValid && colIsValid && subgridIsValid
}

func valueInRow(board [][]int, row int, value int) bool {
	for _, element := range board[row] {
		if element == value {
			return true
		}
	}
	return false
}

func valueInCol(board [][]int, col int, value int) bool {
	for _, row := range board {
		if row[col] == value {
			return true
		}
	}
	return false
}

func valueInSubgrid(board [][]int, row int, col int, value int) bool {
	subgridRowStart := (row / 3) * 3
	subgridColStart := (col / 3) * 3
	for rowIdx := 0; rowIdx < 3; rowIdx++ {
		for colIdx := 0; colIdx < 3; colIdx++ {
			rowToCheck := subgridRowStart + rowIdx
			colToCheck := subgridColStart + colIdx
			existingValue := board[rowToCheck][colToCheck]
			if existingValue == value {
				return true
			}
		}
	}
	return false
}`,
		PythonSolution: `def solveSudoku(board: List[List[int]]) -> List[List[int]]:
    solve_partial_sudoku(0, 0, board)
    return board

def solve_partial_sudoku(row, col, board):
    current_row = row
    current_col = col
    if current_col == len(board[current_row]):
        current_row += 1
        current_col = 0
        if current_row == len(board):
            return True
    if board[current_row][current_col] == 0:
        return try_digits_at_position(current_row, current_col, board)
    return solve_partial_sudoku(current_row, current_col + 1, board)

def try_digits_at_position(row, col, board):
    for digit in range(1, 10):
        if is_valid_at_position(digit, row, col, board):
            board[row][col] = digit
            if solve_partial_sudoku(row, col + 1, board):
                return True
    board[row][col] = 0
    return False

def is_valid_at_position(value, row, col, board):
    row_is_valid = not value_in_row(board, row, value)
    col_is_valid = not value_in_col(board, col, value)
    subgrid_is_valid = not value_in_subgrid(board, row, col, value)
    return row_is_valid and col_is_valid and subgrid_is_valid

def value_in_row(board, row, value):
    return value in board[row]

def value_in_col(board, col, value):
    for row in board:
        if row[col] == value:
            return True
    return False

def value_in_subgrid(board, row, col, value):
    subgrid_row_start = (row // 3) * 3
    subgrid_col_start = (col // 3) * 3
    for row_idx in range(3):
        for col_idx in range(3):
            row_to_check = subgrid_row_start + row_idx
            col_to_check = subgrid_col_start + col_idx
            if board[row_to_check][col_to_check] == value:
                return True
    return False`,
		Explanation: `Solving Sudoku requires systematically trying all possible digit placements and backtracking when constraints are violated. This is a classic backtracking problem that explores the solution space efficiently.

The algorithm processes cells row by row, left to right. For each empty cell, it tries digits 1-9. Before placing a digit, it checks three constraints: the digit must not appear in the current row, current column, or the 3x3 subgrid containing the cell. If a digit is valid, it's placed and the algorithm recurses to the next cell. If no digit works, the algorithm backtracks by removing the placed digit and trying the next option.

The backtracking ensures that when we reach an impossible state (no valid digit for a cell), we undo recent choices and try alternatives. This systematic exploration guarantees finding a solution if one exists, or determining that no solution exists.

Edge cases handled include: boards with no empty cells (already solved), boards with multiple solutions (finds one solution), boards with no solution (backtracks through all possibilities), and boards with many constraints (backtracking prunes invalid branches early).

An alternative approach would use constraint propagation (like arc consistency), but that's more complex. The backtracking approach is straightforward and works well for standard 9x9 Sudoku. The time complexity is O(9^m) in the worst case where m is the number of empty cells, though pruning makes it much faster in practice. The space complexity is O(1) extra space beyond the board itself.

Think of this as solving a puzzle by trying each piece: if a piece fits, you place it and move to the next. If no piece fits, you remove the last piece and try a different one, continuing until the puzzle is complete.`,
	},
	{
		ID:          131,
		Title:       "Generate Div Tags",
		Description: "Write a function that takes in a positive integer numberOfTags and returns a list of all the valid strings that you can generate with that number of matched <div></div> tags.",
		Difficulty:  "Medium",
		Topic:       "Recursion",
		Signature:   "func generateDivTags(numberOfTags int) []string",
		TestCases: []TestCase{
			{Input: "numberOfTags = 3", Expected: "[\"<div><div><div></div></div></div>\", \"<div><div></div><div></div></div>\", \"<div><div></div></div><div></div>\", \"<div></div><div><div></div></div>\", \"<div></div><div></div><div></div>\"]"},
		},
		Solution: `func generateDivTags(numberOfTags int) []string {
	matchedDivTags := []string{}
	generateDivTagsFromPrefix(numberOfTags, numberOfTags, "", &matchedDivTags)
	return matchedDivTags
}

func generateDivTagsFromPrefix(openingTagsNeeded int, closingTagsNeeded int, prefix string, result *[]string) {
	if openingTagsNeeded > 0 {
		newPrefix := prefix + "<div>"
		generateDivTagsFromPrefix(openingTagsNeeded-1, closingTagsNeeded, newPrefix, result)
	}
	if openingTagsNeeded < closingTagsNeeded {
		newPrefix := prefix + "</div>"
		generateDivTagsFromPrefix(openingTagsNeeded, closingTagsNeeded-1, newPrefix, result)
	}
	if closingTagsNeeded == 0 {
		*result = append(*result, prefix)
	}
}`,
		PythonSolution: `def generateDivTags(numberOfTags: int) -> List[str]:
    matched_div_tags = []
    generate_div_tags_from_prefix(numberOfTags, numberOfTags, "", matched_div_tags)
    return matched_div_tags

def generate_div_tags_from_prefix(opening_tags_needed, closing_tags_needed, prefix, result):
    if opening_tags_needed > 0:
        new_prefix = prefix + "<div>"
        generate_div_tags_from_prefix(opening_tags_needed - 1, closing_tags_needed, new_prefix, result)
    if opening_tags_needed < closing_tags_needed:
        new_prefix = prefix + "</div>"
        generate_div_tags_from_prefix(opening_tags_needed, closing_tags_needed - 1, new_prefix, result)
    if closing_tags_needed == 0:
        result.append(prefix)`,
		Explanation: `Generating all valid combinations of matched div tags is equivalent to generating all valid parentheses combinations. The recursive approach builds strings by adding opening and closing tags while maintaining validity.

The algorithm uses two counters: openingTagsNeeded and closingTagsNeeded. At each step, if opening tags are still needed, we can add an opening tag "<div>" and recurse. If there are more closing tags needed than opening tags (meaning we have unmatched opening tags), we can add a closing tag "</div>" and recurse. The base case occurs when closingTagsNeeded reaches 0, meaning all tags are matched.

This approach ensures validity because we only add closing tags when there are unmatched opening tags (openingTagsNeeded < closingTagsNeeded). This prevents invalid sequences like "</div><div>" and ensures all generated sequences are well-formed.

Edge cases handled include: numberOfTags = 0 (returns empty list), numberOfTags = 1 (returns single matched pair), and larger numbers generating exponentially many combinations (all valid).

An alternative iterative approach using a stack would be more complex. The recursive approach naturally handles the generation process. The time complexity is O(2^(2n)) in the worst case as we generate all valid combinations, where n is numberOfTags. The space complexity is O(2^(2n)) for storing all results, plus O(n) for the recursion stack.

Think of this as building nested structures: you can always open a new level (add opening tag), and you can close a level if you've opened more than you've closed (add closing tag), continuing until all levels are closed.`,
	},
	{
		ID:          132,
		Title:       "Ambiguous Measurements",
		Description: "This problem deals with measuring cups that are missing important measuring labels. Specifically, a measuring cup only has two measuring lines, a Low (L) line and a High (H) line. This means that you can't be sure how much liquid is actually in it, but you do know that the amount is between the L and H line. Given an array of measuring cups containing their low and high lines, as well as one low integer and one high integer representing a range for a target measurement, write a function that returns a boolean representing whether you can use the cups to accurately measure the target range.",
		Difficulty:  "Medium",
		Topic:       "Dynamic Programming",
		Signature:   "func ambiguousMeasurements(measuringCups [][]int, low int, high int) bool",
		TestCases: []TestCase{
			{Input: "measuringCups = [[200, 210], [450, 465], [800, 850]], low = 2100, high = 2300", Expected: "true"},
		},
		Solution: `func ambiguousMeasurements(measuringCups [][]int, low int, high int) bool {
	memoize := make(map[string]bool)
	return canMeasureInRange(measuringCups, low, high, memoize)
}

func canMeasureInRange(measuringCups [][]int, low int, high int, memoize map[string]bool) bool {
	memoizeKey := createHashableKey(low, high)
	if val, found := memoize[memoizeKey]; found {
		return val
	}
	if low <= 0 && high <= 0 {
		return false
	}
	canMeasure := false
	for _, cup := range measuringCups {
		cupLow, cupHigh := cup[0], cup[1]
		if low <= cupLow && cupHigh <= high {
			canMeasure = true
			break
		}
		newLow := max(0, low-cupLow)
		newHigh := max(0, high-cupHigh)
		canMeasure = canMeasureInRange(measuringCups, newLow, newHigh, memoize)
		if canMeasure {
			break
		}
	}
	memoize[memoizeKey] = canMeasure
	return canMeasure
}

func createHashableKey(low int, high int) string {
	return string(rune(low)) + ":" + string(rune(high))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def ambiguousMeasurements(measuringCups: List[List[int]], low: int, high: int) -> bool:
    memoize = {}
    return can_measure_in_range(measuringCups, low, high, memoize)

def can_measure_in_range(measuring_cups, low, high, memoize):
    memoize_key = f"{low}:{high}"
    if memoize_key in memoize:
        return memoize[memoize_key]
    if low <= 0 and high <= 0:
        return False
    can_measure = False
    for cup in measuring_cups:
        cup_low, cup_high = cup
        if low <= cup_low and cup_high <= high:
            can_measure = True
            break
        new_low = max(0, low - cup_low)
        new_high = max(0, high - cup_high)
        can_measure = can_measure_in_range(measuring_cups, new_low, new_high, memoize)
        if can_measure:
            break
    memoize[memoize_key] = can_measure
    return can_measure`,
		Explanation: `Determining if a target range can be measured using given measuring cups requires checking all possible combinations of cup uses. This is a dynamic programming problem that can be solved with memoized recursion.

The algorithm tries each measuring cup and checks if it can help measure the target range. For each cup, if the cup's range fits entirely within the target range (cup_low >= low and cup_high <= high), we've found a solution. Otherwise, we subtract the cup's range from the target range (taking max with 0 to avoid negatives) and recursively check if the reduced range can be measured.

Memoization is crucial here because the same (low, high) ranges may be computed multiple times through different cup combinations. By caching results, we avoid redundant calculations and improve efficiency significantly.

Edge cases handled include: target range where low <= 0 and high <= 0 (base case, return true as range is already measured), cups that don't fit (skip or reduce range), and ranges that cannot be measured (exhaust all possibilities, return false).

An alternative iterative DP approach would use a 2D table, but the recursive approach is more intuitive. The time complexity is O(low*high*n) where n is the number of cups, as we compute each (low, high) pair at most once. The space complexity is O(low*high) for the memoization cache and O(low+high) for the recursion stack.

Think of this as trying to measure a range using available measuring tools: you try each tool, and if it fits, you're done. If it's too large, you see if you can measure what's left after using it, building up the solution recursively.`,
	},
	{
		ID:          133,
		Title:       "Shifted Binary Search",
		Description: "Write a function that takes in a sorted array of distinct integers as well as a target integer. The caveat is that the integers in the array have been shifted by some amount; in other words, they've been moved to the left or to the right by one or more positions. For example, [1, 2, 3, 4] might have turned into [3, 4, 1, 2]. The function should use a variation of the Binary Search algorithm to determine if the target integer is contained in the array and should return its index if it is, otherwise -1.",
		Difficulty:  "Medium",
		Topic:       "Binary Search",
		Signature:   "func shiftedBinarySearch(array []int, target int) int",
		TestCases: []TestCase{
			{Input: "array = [45, 61, 71, 72, 73, 0, 1, 21, 33, 37], target = 33", Expected: "8"},
			{Input: "array = [5, 23, 111, 1], target = 111", Expected: "2"},
		},
		Solution: `func shiftedBinarySearch(array []int, target int) int {
	return shiftedBinarySearchHelper(array, target, 0, len(array)-1)
}

func shiftedBinarySearchHelper(array []int, target int, left int, right int) int {
	if left > right {
		return -1
	}
	middle := (left + right) / 2
	potentialMatch := array[middle]
	leftNum := array[left]
	rightNum := array[right]
	if target == potentialMatch {
		return middle
	} else if leftNum <= potentialMatch {
		if target < potentialMatch && target >= leftNum {
			return shiftedBinarySearchHelper(array, target, left, middle-1)
		}
		return shiftedBinarySearchHelper(array, target, middle+1, right)
	} else {
		if target > potentialMatch && target <= rightNum {
			return shiftedBinarySearchHelper(array, target, middle+1, right)
		}
		return shiftedBinarySearchHelper(array, target, left, middle-1)
	}
}`,
		PythonSolution: `def shiftedBinarySearch(array: List[int], target: int) -> int:
    return shifted_binary_search_helper(array, target, 0, len(array) - 1)

def shifted_binary_search_helper(array, target, left, right):
    if left > right:
        return -1
    middle = (left + right) // 2
    potential_match = array[middle]
    left_num = array[left]
    right_num = array[right]
    if target == potential_match:
        return middle
    elif left_num <= potential_match:
        if target < potential_match and target >= left_num:
            return shifted_binary_search_helper(array, target, left, middle - 1)
        return shifted_binary_search_helper(array, target, middle + 1, right)
    else:
        if target > potential_match and target <= right_num:
            return shifted_binary_search_helper(array, target, middle + 1, right)
        return shifted_binary_search_helper(array, target, left, middle - 1)`,
		Explanation: `Searching in a rotated sorted array requires determining which half of the array is sorted, then deciding which half to search based on where the target might be. This is a modified binary search that handles the rotation.

The algorithm compares the middle element with the left and right bounds to determine which half is sorted. If the left half is sorted (leftNum <= potentialMatch), we check if the target is within that sorted range. If so, we search the left half; otherwise, we search the right half. If the right half is sorted, we apply similar logic in reverse.

The key insight is that at least one half of a rotated sorted array is always sorted. By identifying which half is sorted and checking if the target falls within that sorted range, we can eliminate half of the search space at each step, maintaining O(log n) time complexity.

Edge cases handled include: array not rotated (standard binary search), target at rotation point, target not in array (returns -1), and arrays with duplicate values (handled by the comparison logic).

An alternative approach would be to find the rotation point first, then search in the appropriate half, but that requires two passes. The current approach finds the target in a single pass. The time complexity is O(log n) for the binary search, and space complexity is O(log n) for the recursion stack (or O(1) for iterative version).

Think of this as searching in a circular list: you determine which arc is sorted, check if your target is in that arc, and if not, search the other arc, continuing until you find the target or exhaust all possibilities.`,
	},
	{
		ID:          134,
		Title:       "Search For Range",
		Description: "Write a function that takes in a sorted array of integers as well as a target integer. The function should use a variation of the Binary Search algorithm to find a range of indices in between which the target number is contained in the array and should return this range in the form of an array. The first number in the output array should represent the first index at which the target number is located, while the second number should represent the last index at which the target number is located. The function should return [-1, -1] if the integer isn't contained in the array.",
		Difficulty:  "Medium",
		Topic:       "Binary Search",
		Signature:   "func searchForRange(array []int, target int) []int",
		TestCases: []TestCase{
			{Input: "array = [0, 1, 21, 33, 45, 45, 45, 45, 45, 45, 61, 71, 73], target = 45", Expected: "[4 9]"},
			{Input: "array = [5, 7, 7, 8, 8, 10], target = 5", Expected: "[0 0]"},
			{Input: "array = [5, 7, 7, 8, 8, 10], target = 9", Expected: "[-1 -1]"},
		},
		Solution: `func searchForRange(array []int, target int) []int {
	finalRange := []int{-1, -1}
	alteredBinarySearch(array, target, 0, len(array)-1, &finalRange, true)
	alteredBinarySearch(array, target, 0, len(array)-1, &finalRange, false)
	return finalRange
}

func alteredBinarySearch(array []int, target int, left int, right int, finalRange *[]int, goLeft bool) {
	if left > right {
		return
	}
	mid := (left + right) / 2
	if array[mid] < target {
		alteredBinarySearch(array, target, mid+1, right, finalRange, goLeft)
	} else if array[mid] > target {
		alteredBinarySearch(array, target, left, mid-1, finalRange, goLeft)
	} else {
		if goLeft {
			if mid == 0 || array[mid-1] != target {
				(*finalRange)[0] = mid
			} else {
				alteredBinarySearch(array, target, left, mid-1, finalRange, goLeft)
			}
		} else {
			if mid == len(array)-1 || array[mid+1] != target {
				(*finalRange)[1] = mid
			} else {
				alteredBinarySearch(array, target, mid+1, right, finalRange, goLeft)
			}
		}
	}
}`,
		PythonSolution: `def searchForRange(array: List[int], target: int) -> List[int]:
    final_range = [-1, -1]
    altered_binary_search(array, target, 0, len(array) - 1, final_range, True)
    altered_binary_search(array, target, 0, len(array) - 1, final_range, False)
    return final_range

def altered_binary_search(array, target, left, right, final_range, go_left):
    if left > right:
        return
    mid = (left + right) // 2
    if array[mid] < target:
        altered_binary_search(array, target, mid + 1, right, final_range, go_left)
    elif array[mid] > target:
        altered_binary_search(array, target, left, mid - 1, final_range, go_left)
    else:
        if go_left:
            if mid == 0 or array[mid - 1] != target:
                final_range[0] = mid
            else:
                altered_binary_search(array, target, left, mid - 1, final_range, go_left)
        else:
            if mid == len(array) - 1 or array[mid + 1] != target:
                final_range[1] = mid
            else:
                altered_binary_search(array, target, mid + 1, right, final_range, go_left)`,
		Explanation: `Finding the range of a target value in a sorted array requires finding both the leftmost and rightmost occurrences. This can be efficiently done with two modified binary searches.

The algorithm performs two binary searches: one to find the leftmost occurrence and one to find the rightmost occurrence. For the leftmost search, when we find the target, we check if the element before it is different (or if we're at index 0). If so, we've found the left boundary; otherwise, we continue searching left. Similarly, for the rightmost search, we check if the element after is different (or if we're at the last index).

This approach efficiently finds both boundaries in O(log n) time by leveraging the sorted property. Each binary search eliminates half of the remaining search space at each step, making it much faster than linear search.

Edge cases handled include: target not in array (returns [-1, -1]), target appears once (both boundaries are the same index), target appears multiple times (finds correct range), and target at array boundaries (handled correctly by index checks).

An alternative approach would use linear search, but that's O(n) time. Another approach would find one occurrence then expand linearly, but that's O(n) in the worst case. The two binary searches approach is optimal with O(log n) time complexity. The space complexity is O(log n) for the recursion stack (or O(1) for iterative version).

Think of this as finding the edges of a block of identical items: you search for where the block starts (left boundary) and where it ends (right boundary), using binary search to find each edge efficiently.`,
	},
	{
		ID:          135,
		Title:       "Quickselect",
		Description: "Write a function that takes in an array of distinct integers as well as an integer k and returns the kth smallest integer in that array. The function should use the Quickselect algorithm.",
		Difficulty:  "Medium",
		Topic:       "Sorting",
		Signature:   "func quickselect(array []int, k int) int",
		TestCases: []TestCase{
			{Input: "array = [8, 5, 2, 9, 7, 6, 3], k = 3", Expected: "5"},
			{Input: "array = [1], k = 1", Expected: "1"},
		},
		Solution: `func quickselect(array []int, k int) int {
	position := k - 1
	return quickselectHelper(array, 0, len(array)-1, position)
}

func quickselectHelper(array []int, startIdx int, endIdx int, position int) int {
	for {
		if startIdx > endIdx {
			panic("Your algorithm should never arrive here!")
		}
		pivotIdx := startIdx
		leftIdx := startIdx + 1
		rightIdx := endIdx
		for leftIdx <= rightIdx {
			if array[leftIdx] > array[pivotIdx] && array[rightIdx] < array[pivotIdx] {
				swap(leftIdx, rightIdx, array)
			}
			if array[leftIdx] <= array[pivotIdx] {
				leftIdx++
			}
			if array[rightIdx] >= array[pivotIdx] {
				rightIdx--
			}
		}
		swap(pivotIdx, rightIdx, array)
		if rightIdx == position {
			return array[rightIdx]
		} else if rightIdx < position {
			startIdx = rightIdx + 1
		} else {
			endIdx = rightIdx - 1
		}
	}
}

func swap(i int, j int, array []int) {
	array[i], array[j] = array[j], array[i]
}`,
		PythonSolution: `def quickselect(array: List[int], k: int) -> int:
    position = k - 1
    return quickselect_helper(array, 0, len(array) - 1, position)

def quickselect_helper(array, start_idx, end_idx, position):
    while True:
        if start_idx > end_idx:
            raise Exception("Your algorithm should never arrive here!")
        pivot_idx = start_idx
        left_idx = start_idx + 1
        right_idx = end_idx
        while left_idx <= right_idx:
            if array[left_idx] > array[pivot_idx] and array[right_idx] < array[pivot_idx]:
                swap(left_idx, right_idx, array)
            if array[left_idx] <= array[pivot_idx]:
                left_idx += 1
            if array[right_idx] >= array[pivot_idx]:
                right_idx -= 1
        swap(pivot_idx, right_idx, array)
        if right_idx == position:
            return array[right_idx]
        elif right_idx < position:
            start_idx = right_idx + 1
        else:
            end_idx = right_idx - 1

def swap(i, j, array):
    array[i], array[j] = array[j], array[i]`,
		Explanation: `Quickselect is an algorithm to find the kth smallest element in an unsorted array without fully sorting it. It's based on the partition step of quicksort, but only recurses on the side containing the desired element.

The algorithm uses the partition operation: it chooses a pivot, partitions the array so elements smaller than the pivot are on the left and larger on the right, then places the pivot in its correct sorted position. If the pivot ends up at position k-1 (0-indexed), it's the kth smallest element and we return it. Otherwise, we recurse on the left side if k-1 is less than the pivot position, or the right side if it's greater.

This approach is efficient because on average, each partition eliminates half of the remaining elements, giving O(n) average time complexity. However, worst-case performance is O(n²) if we consistently choose bad pivots, though this is rare with good pivot selection strategies.

Edge cases handled include: k = 1 (find minimum), k = array length (find maximum), k out of bounds (handled by validation), and arrays with duplicate values (partition handles correctly).

An alternative approach would be to sort the array and return the kth element, but that's O(n log n) time. Another approach would use a heap of size k, but that's O(n log k) time. Quickselect achieves O(n) average time with O(1) extra space (or O(log n) for recursion stack).

Think of this as finding the kth place finisher in a race: you partition runners by a pivot time, and if k runners finished before the pivot, you know the kth runner is in that group, so you only need to search there.`,
	},
	{
		ID:          136,
		Title:       "Index Equals Value",
		Description: "Write a function that takes in a sorted array of distinct integers and returns the first index in the array that is equal to the value at that index. In other words, your function should return the minimum index where index == array[index]. If there is no such index, your function should return -1.",
		Difficulty:  "Medium",
		Topic:       "Binary Search",
		Signature:   "func indexEqualsValue(array []int) int",
		TestCases: []TestCase{
			{Input: "array = [-5, -3, 0, 3, 4, 5, 9]", Expected: "3"},
			{Input: "array = [-12, 1, 2, 3, 12]", Expected: "1"},
		},
		Solution: `func indexEqualsValue(array []int) int {
	leftIdx := 0
	rightIdx := len(array) - 1
	answer := -1
	for leftIdx <= rightIdx {
		middleIdx := (leftIdx + rightIdx) / 2
		middleValue := array[middleIdx]
		if middleValue < middleIdx {
			leftIdx = middleIdx + 1
		} else if middleValue == middleIdx && middleIdx == 0 {
			return middleIdx
		} else if middleValue == middleIdx && array[middleIdx-1] < middleIdx-1 {
			return middleIdx
		} else {
			answer = middleIdx
			rightIdx = middleIdx - 1
		}
	}
	return answer
}`,
		PythonSolution: `def indexEqualsValue(array: List[int]) -> int:
    left_idx = 0
    right_idx = len(array) - 1
    answer = -1
    while left_idx <= right_idx:
        middle_idx = (left_idx + right_idx) // 2
        middle_value = array[middle_idx]
        if middle_value < middle_idx:
            left_idx = middle_idx + 1
        elif middle_value == middle_idx and middle_idx == 0:
            return middle_idx
        elif middle_value == middle_idx and array[middle_idx - 1] < middle_idx - 1:
            return middle_idx
        else:
            answer = middle_idx
            right_idx = middle_idx - 1
    return answer`,
		Explanation: `Finding the first index where index equals value in a sorted distinct array requires a modified binary search that leverages the sorted property and distinctness to eliminate search space efficiently.

The algorithm uses binary search with a key insight: if array[mid] < mid, then for all indices i <= mid, we have array[i] <= array[mid] < mid <= i (since values are distinct and sorted), so array[i] cannot equal i. Therefore, we must search the right half. Similarly, if array[mid] > mid, we search the left half. When array[mid] == mid, we've found a match, but we need to check if it's the first occurrence by checking if array[mid-1] == mid-1.

This approach efficiently narrows down the search space by using the relationship between indices and values. The distinctness property ensures that if array[mid] < mid, all elements to the left also have array[i] < i, allowing us to eliminate that half.

Edge cases handled include: no index equals value (returns -1), first index equals value (array[0] == 0), last index equals value (array[n-1] == n-1), and multiple indices equal their values (finds the first one).

An alternative linear search would be O(n) time. The binary search approach achieves O(log n) time by eliminating half the search space at each step. The space complexity is O(1) for the iterative version (or O(log n) for recursive).

Think of this as searching for where a sorted sequence of distinct integers crosses the line y=x: if the value is below the line at the midpoint, the crossing must be to the right; if above, to the left; if on the line, check if it's the first crossing.`,
	},
	{
		ID:          137,
		Title:       "Quick Sort",
		Description: "Write a function that takes in an array of integers and returns a sorted version of that array. Use the Quick Sort algorithm to sort the array.",
		Difficulty:  "Medium",
		Topic:       "Sorting",
		Signature:   "func quickSort(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = [8, 5, 2, 9, 5, 6, 3]", Expected: "[2, 3, 5, 5, 6, 8, 9]"},
			{Input: "array = [1]", Expected: "[1]"},
		},
		Solution: `func quickSort(array []int) []int {
	quickSortHelper(array, 0, len(array)-1)
	return array
}

func quickSortHelper(array []int, startIdx int, endIdx int) {
	if startIdx >= endIdx {
		return
	}
	pivotIdx := startIdx
	leftIdx := startIdx + 1
	rightIdx := endIdx
	for rightIdx >= leftIdx {
		if array[leftIdx] > array[pivotIdx] && array[rightIdx] < array[pivotIdx] {
			swap(leftIdx, rightIdx, array)
		}
		if array[leftIdx] <= array[pivotIdx] {
			leftIdx += 1
		}
		if array[rightIdx] >= array[pivotIdx] {
			rightIdx -= 1
		}
	}
	swap(pivotIdx, rightIdx, array)
	leftSubarrayIsSmaller := rightIdx-1-startIdx < endIdx-(rightIdx+1)
	if leftSubarrayIsSmaller {
		quickSortHelper(array, startIdx, rightIdx-1)
		quickSortHelper(array, rightIdx+1, endIdx)
	} else {
		quickSortHelper(array, rightIdx+1, endIdx)
		quickSortHelper(array, startIdx, rightIdx-1)
	}
}

func swap(i int, j int, array []int) {
	array[i], array[j] = array[j], array[i]
}`,
		PythonSolution: `def quickSort(array: List[int]) -> List[int]:
    quick_sort_helper(array, 0, len(array) - 1)
    return array

def quick_sort_helper(array, start_idx, end_idx):
    if start_idx >= end_idx:
        return
    pivot_idx = start_idx
    left_idx = start_idx + 1
    right_idx = end_idx
    while right_idx >= left_idx:
        if array[left_idx] > array[pivot_idx] and array[right_idx] < array[pivot_idx]:
            swap(left_idx, right_idx, array)
        if array[left_idx] <= array[pivot_idx]:
            left_idx += 1
        if array[right_idx] >= array[pivot_idx]:
            right_idx -= 1
    swap(pivot_idx, right_idx, array)
    left_subarray_is_smaller = right_idx - 1 - start_idx < end_idx - (right_idx + 1)
    if left_subarray_is_smaller:
        quick_sort_helper(array, start_idx, right_idx - 1)
        quick_sort_helper(array, rightIdx + 1, end_idx)
    else:
        quick_sort_helper(array, right_idx + 1, end_idx)
        quick_sort_helper(array, start_idx, right_idx - 1)

def swap(i, j, array):
    array[i], array[j] = array[j], array[i]`,
		Explanation: `Quicksort is a divide-and-conquer sorting algorithm that works by selecting a pivot element and partitioning the array around it. Elements smaller than the pivot go to the left, and larger elements go to the right. The algorithm then recursively sorts the left and right partitions.

The partition step uses two pointers (leftIdx and rightIdx) that move towards each other, swapping elements that are on the wrong side of the pivot. When the pointers cross, the pivot is swapped into its correct position. The algorithm then recursively sorts the subarrays on either side of the pivot.

An important optimization is to sort the smaller partition first, which minimizes the recursion stack depth. This ensures that the maximum depth is O(log n) even in the worst case, improving space efficiency.

Edge cases handled include: arrays with one element (already sorted, return immediately), arrays with duplicate values (partition handles correctly), already sorted arrays (performance depends on pivot choice), and reverse-sorted arrays (worst case if pivot is always minimum/maximum).

An alternative approach would use a different pivot selection strategy (like median-of-three) to improve worst-case performance. The time complexity is O(n log n) on average, but O(n²) in the worst case with poor pivot choices. The space complexity is O(log n) for the recursion stack.

Think of this as organizing a deck of cards: you pick a card (pivot), separate cards smaller and larger than it, then organize each group the same way, continuing until everything is sorted.`,
	},
	{
		ID:          138,
		Title:       "Heap Sort",
		Description: "Write a function that takes in an array of integers and returns a sorted version of that array. Use the Heap Sort algorithm to sort the array.",
		Difficulty:  "Medium",
		Topic:       "Sorting",
		Signature:   "func heapSort(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = [8, 5, 2, 9, 5, 6, 3]", Expected: "[2, 3, 5, 5, 6, 8, 9]"},
		},
		Solution: `func heapSort(array []int) []int {
	buildMaxHeap(array)
	for endIdx := len(array) - 1; endIdx > 0; endIdx-- {
		swap(0, endIdx, array)
		siftDown(0, endIdx-1, array)
	}
	return array
}

func buildMaxHeap(array []int) {
	firstParentIdx := (len(array) - 2) / 2
	for currentIdx := firstParentIdx; currentIdx >= 0; currentIdx-- {
		siftDown(currentIdx, len(array)-1, array)
	}
}

func siftDown(currentIdx int, endIdx int, heap []int) {
	childOneIdx := currentIdx*2 + 1
	for childOneIdx <= endIdx {
		childTwoIdx := -1
		if currentIdx*2+2 <= endIdx {
			childTwoIdx = currentIdx*2 + 2
		}
		idxToSwap := childOneIdx
		if childTwoIdx != -1 && heap[childTwoIdx] > heap[childOneIdx] {
			idxToSwap = childTwoIdx
		}
		if heap[idxToSwap] > heap[currentIdx] {
			swap(currentIdx, idxToSwap, heap)
			currentIdx = idxToSwap
			childOneIdx = currentIdx*2 + 1
		} else {
			return
		}
	}
}

func swap(i int, j int, array []int) {
	array[i], array[j] = array[j], array[i]
}`,
		PythonSolution: `def heapSort(array: List[int]) -> List[int]:
    build_max_heap(array)
    for end_idx in range(len(array) - 1, 0, -1):
        swap(0, end_idx, array)
        sift_down(0, end_idx - 1, array)
    return array

def build_max_heap(array):
    first_parent_idx = (len(array) - 2) // 2
    for current_idx in range(first_parent_idx, -1, -1):
        sift_down(current_idx, len(array) - 1, array)

def sift_down(current_idx, end_idx, heap):
    child_one_idx = current_idx * 2 + 1
    while child_one_idx <= end_idx:
        child_two_idx = current_idx * 2 + 2 if current_idx * 2 + 2 <= end_idx else -1
        if child_two_idx != -1 and heap[child_two_idx] > heap[child_one_idx]:
            idx_to_swap = child_two_idx
        else:
            idx_to_swap = child_one_idx
        if heap[idx_to_swap] > heap[current_idx]:
            swap(current_idx, idx_to_swap, heap)
            current_idx = idx_to_swap
            child_one_idx = current_idx * 2 + 1
        else:
            return

def swap(i, j, array):
    array[i], array[j] = array[j], array[i]`,
		Explanation: `Heapsort is an in-place sorting algorithm that uses a max-heap data structure. It works by first building a max-heap from the array, then repeatedly extracting the maximum element and placing it at the end of the sorted portion.

The algorithm first builds a max-heap by calling siftDown on all non-leaf nodes from bottom to top. This ensures the heap property is maintained: each parent is greater than or equal to its children. Once the heap is built, the algorithm repeatedly swaps the root (maximum element) with the last unsorted element, then sifts down the new root to restore the heap property.

This process continues until all elements are sorted. The key insight is that after each swap, the largest remaining element is at the root, ready to be placed in its correct sorted position.

Edge cases handled include: arrays with duplicate values (heap property handles correctly), already sorted arrays (still requires full heap sort), reverse-sorted arrays (heap building handles correctly), and single-element arrays (already sorted).

An alternative approach would use a min-heap and extract minimums, but max-heap is more intuitive for sorting in ascending order. The time complexity is O(n log n) for both building the heap (O(n)) and extracting elements (O(n log n)). The space complexity is O(1) as the sorting is done in-place.

Think of this as repeatedly finding the tallest person in a group, moving them to the back of a line, then finding the tallest among the remaining people, continuing until everyone is sorted by height.`,
	},
	{
		ID:          139,
		Title:       "Radix Sort",
		Description: "Write a function that takes in an array of non-negative integers and returns a sorted version of that array. Use the Radix Sort algorithm to sort the array.",
		Difficulty:  "Medium",
		Topic:       "Sorting",
		Signature:   "func radixSort(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = [8762, 654, 3008, 345, 87, 65, 234, 12, 2]", Expected: "[2, 12, 65, 87, 234, 345, 654, 3008, 8762]"},
		},
		Solution: `func radixSort(array []int) []int {
	if len(array) == 0 {
		return array
	}
	maxNumber := max(array)
	digit := 0
	for maxNumber/pow(10, digit) > 0 {
		countingSort(array, digit)
		digit++
	}
	return array
}

func countingSort(array []int, digit int) {
	sortedArray := make([]int, len(array))
	countArray := []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	digitColumn := pow(10, digit)
	for _, num := range array {
		countIdx := (num / digitColumn) % 10
		countArray[countIdx]++
	}
	for idx := 1; idx < 10; idx++ {
		countArray[idx] += countArray[idx-1]
	}
	for idx := len(array) - 1; idx >= 0; idx-- {
		countIdx := (array[idx] / digitColumn) % 10
		countArray[countIdx]--
		sortedIndex := countArray[countIdx]
		sortedArray[sortedIndex] = array[idx]
	}
	for idx := 0; idx < len(array); idx++ {
		array[idx] = sortedArray[idx]
	}
}

func pow(base int, exponent int) int {
	result := 1
	for i := 0; i < exponent; i++ {
		result *= base
	}
	return result
}

func max(array []int) int {
	maxVal := array[0]
	for _, val := range array {
		if val > maxVal {
			maxVal = val
		}
	}
	return maxVal
}`,
		PythonSolution: `def radixSort(array: List[int]) -> List[int]:
    if len(array) == 0:
        return array
    max_number = max(array)
    digit = 0
    while max_number / (10 ** digit) > 0:
        counting_sort(array, digit)
        digit += 1
    return array

def counting_sort(array, digit):
    sorted_array = [0] * len(array)
    count_array = [0] * 10
    digit_column = 10 ** digit
    for num in array:
        count_idx = (num // digit_column) % 10
        count_array[count_idx] += 1
    for idx in range(1, 10):
        count_array[idx] += count_array[idx - 1]
    for idx in range(len(array) - 1, -1, -1):
        count_idx = (array[idx] // digit_column) % 10
        count_array[count_idx] -= 1
        sorted_index = count_array[count_idx]
        sorted_array[sorted_index] = array[idx]
    for idx in range(len(array)):
        array[idx] = sorted_array[idx]`,
		Explanation: `Radix sort is a non-comparative sorting algorithm that sorts numbers by processing individual digits. It works by sorting numbers digit by digit, starting from the least significant digit and moving to the most significant, using a stable sorting algorithm (counting sort) for each digit position.

The algorithm first determines the maximum number to know how many digits to process. Then, for each digit position from least to most significant, it performs counting sort based on that digit. Counting sort groups numbers by their digit value, then reconstructs the array in sorted order for that digit position.

This approach works because counting sort is stable, meaning numbers with the same digit value maintain their relative order from previous sorts. By sorting from least to most significant digit, we ensure that the final array is sorted correctly.

Edge cases handled include: numbers with different numbers of digits (shorter numbers treated as having leading zeros), negative numbers (would require special handling, typically not in scope), and numbers with many digits (handled by the digit iteration).

An alternative approach would use comparison-based sorting, but that's O(n log n). Radix sort achieves O(d*(n+k)) where d is the number of digits, n is the number of elements, and k is the radix (10 for decimal). For numbers with a fixed number of digits, this becomes O(n). The space complexity is O(n+k) for the counting sort arrays.

Think of this as sorting a deck of cards by first sorting by suit, then by rank within each suit, ensuring stability so cards maintain their order within groups.`,
	},
	{
		ID:          140,
		Title:       "Shorten Path",
		Description: "Write a function that takes in a non-empty string representing a valid Unix-shell path and returns a shortened version of that path. A path can be shortened by resolving any \".\" and \"..\" in the path. \".\" means current directory, \"..\" means parent directory.",
		Difficulty:  "Medium",
		Topic:       "Strings",
		Signature:   "func shortenPath(path string) string",
		TestCases: []TestCase{
			{Input: "path = \"/foo/../test/../test/../foo//bar/./baz\"", Expected: "\"/foo/bar/baz\""},
			{Input: "path = \"/foo/bar/baz\"", Expected: "\"/foo/bar/baz\""},
		},
		Solution: `import "strings"

func shortenPath(path string) string {
	startsWithSlash := path[0] == '/'
	tokens := filter(strings.Split(path, "/"), func(s string) bool {
		return len(s) > 0 && s != "."
	})
	stack := []string{}
	if startsWithSlash {
		stack = append(stack, "")
	}
	for _, token := range tokens {
		if token == ".." {
			if len(stack) == 0 || stack[len(stack)-1] == ".." {
				stack = append(stack, token)
			} else if stack[len(stack)-1] != "" {
				stack = stack[:len(stack)-1]
			}
		} else {
			stack = append(stack, token)
		}
	}
	if len(stack) == 1 && stack[0] == "" {
		return "/"
	}
	return strings.Join(stack, "/")
}

func filter(slice []string, test func(string) bool) []string {
	result := []string{}
	for _, s := range slice {
		if test(s) {
			result = append(result, s)
		}
	}
	return result
}`,
		PythonSolution: `def shortenPath(path: str) -> str:
    starts_with_slash = path[0] == '/'
    tokens = [token for token in path.split('/') if len(token) > 0 and token != '.']
    stack = []
    if starts_with_slash:
        stack.append('')
    for token in tokens:
        if token == '..':
            if len(stack) == 0 or stack[-1] == '..':
                stack.append(token)
            elif stack[-1] != '':
                stack.pop()
        else:
            stack.append(token)
    if len(stack) == 1 and stack[0] == '':
        return '/'
    return '/'.join(stack)`,
		Explanation: `Simplifying a file path requires handling special directory entries like ".", "..", and multiple slashes. The algorithm uses a stack to build the simplified path by processing each directory component.

The algorithm first splits the path by "/" and filters out empty strings and "." entries (which don't change the directory). It maintains a stack to track the current directory structure. For each token, if it's "..", it pops from the stack (goes up one directory) unless we're at the root or the stack contains ".." (relative paths). Otherwise, it pushes the directory onto the stack.

The root case is handled specially: if the path starts with "/", we push an empty string onto the stack to represent the root. This ensures that ".." at the root doesn't pop anything, maintaining the root.

Edge cases handled include: paths starting with "/" (absolute paths, root preserved), paths with ".." at root (ignored), paths with multiple slashes (filtered out), paths ending with "/" (handled correctly), and relative paths with ".." (stacked correctly).

An alternative approach would use string manipulation, but that's more error-prone. The stack-based approach is cleaner and handles edge cases naturally. The time complexity is O(n) where n is the path length, and space complexity is O(n) for the stack and filtered tokens.

Think of this as navigating a file system: "." means stay here (ignore), ".." means go up (pop stack), and directory names mean go into that directory (push stack), building up the final path from the stack.`,
	},
	{
		ID:          141,
		Title:       "Largest Rectangle Under Skyline",
		Description: "Write a function that takes in an array of positive integers representing the heights of adjacent buildings and returns the area of the largest rectangle that can be created by any number of adjacent buildings, including just one building. All buildings have the same width of 1 unit.",
		Difficulty:  "Medium",
		Topic:       "Stack",
		Signature:   "func largestRectangleUnderSkyline(buildings []int) int",
		TestCases: []TestCase{
			{Input: "buildings = [1, 3, 3, 2, 4, 1, 5, 3, 2]", Expected: "9"},
		},
		Solution: `func largestRectangleUnderSkyline(buildings []int) int {
	pillarIndices := []int{}
	maxArea := 0
	for idx := 0; idx < len(buildings)+1; idx++ {
		height := 0
		if idx < len(buildings) {
			height = buildings[idx]
		}
		for len(pillarIndices) > 0 && buildings[pillarIndices[len(pillarIndices)-1]] >= height {
			pillarHeight := buildings[pillarIndices[len(pillarIndices)-1]]
			pillarIndices = pillarIndices[:len(pillarIndices)-1]
			width := idx
			if len(pillarIndices) > 0 {
				width = idx - pillarIndices[len(pillarIndices)-1] - 1
			}
			maxArea = max(maxArea, width*pillarHeight)
		}
		pillarIndices = append(pillarIndices, idx)
	}
	return maxArea
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def largestRectangleUnderSkyline(buildings: List[int]) -> int:
    pillar_indices = []
    max_area = 0
    for idx in range(len(buildings) + 1):
        height = buildings[idx] if idx < len(buildings) else 0
        while pillar_indices and buildings[pillar_indices[-1]] >= height:
            pillar_height = buildings[pillar_indices.pop()]
            width = idx if not pillar_indices else idx - pillar_indices[-1] - 1
            max_area = max(max_area, width * pillar_height)
        pillar_indices.append(idx)
    return max_area`,
		Explanation: `Finding the largest rectangle in a histogram requires determining the maximum area that can be formed using bars of different heights. The monotonic stack approach efficiently tracks which bars can extend to form rectangles.

The algorithm uses a stack to maintain indices of bars in increasing height order. For each bar, if it's taller than the bar at the top of the stack, we push its index. If it's shorter, we pop from the stack and calculate the area that the popped bar could form: its height times the width from its starting position (next index in stack) to the current position.

The key insight is that when we encounter a shorter bar, all taller bars in the stack can no longer extend to the right, so we calculate their maximum possible areas. The stack maintains the property that bars are in increasing height order, making area calculations straightforward.

Edge cases handled include: bars of equal height (handled correctly by the comparison), bars in increasing order (all pushed, then popped at end), bars in decreasing order (each pops previous bars), and single bar (area is just its height).

An alternative O(n²) approach would check all possible rectangles, but the stack approach is optimal. The time complexity is O(n) as each bar is pushed and popped at most once. The space complexity is O(n) for the stack.

Think of this as finding the largest rectangle you can form by stacking blocks: you keep track of which blocks can still grow taller, and when you find a shorter block, you calculate how big the taller blocks could get before they're blocked.`,
	},
	{
		ID:          142,
		Title:       "Longest Substring Without Duplication",
		Description: "Write a function that takes in a string and returns the length of the longest substring without duplicate characters.",
		Difficulty:  "Medium",
		Topic:       "Strings",
		Signature:   "func longestSubstringWithoutDuplication(str string) string",
		TestCases: []TestCase{
			{Input: "str = \"clementisacap\"", Expected: "\"mentisac\""},
			{Input: "str = \"abc\"", Expected: "\"abc\""},
		},
		Solution: `func longestSubstringWithoutDuplication(str string) string {
	lastSeen := make(map[rune]int)
	longest := []int{0, 1}
	startIdx := 0
	for i, char := range str {
		if seenIdx, found := lastSeen[char]; found {
			startIdx = max(startIdx, seenIdx+1)
		}
		if longest[1]-longest[0] < i+1-startIdx {
			longest = []int{startIdx, i + 1}
		}
		lastSeen[char] = i
	}
	return str[longest[0]:longest[1]]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def longestSubstringWithoutDuplication(string: str) -> str:
    last_seen = {}
    longest = [0, 1]
    start_idx = 0
    for i, char in enumerate(string):
        if char in last_seen:
            start_idx = max(start_idx, last_seen[char] + 1)
        if longest[1] - longest[0] < i + 1 - start_idx:
            longest = [start_idx, i + 1]
        last_seen[char] = i
    return string[longest[0]:longest[1]]`,
		Explanation: `Finding the longest substring without duplicate characters requires efficiently tracking which characters have been seen and where. The sliding window technique combined with a hash map provides an optimal solution.

The algorithm maintains a sliding window using start and end pointers, and a hash map that stores the last seen index of each character. As the end pointer expands the window, if we encounter a character that's already in the current window (its last index >= start), we move the start pointer to just after that last occurrence. This efficiently removes the duplicate and all characters before it from the window.

We track the longest substring found so far by comparing the current window length with the longest seen. The hash map allows O(1) lookup to find where a duplicate character last appeared, enabling efficient window adjustment.

Edge cases handled include: strings with all unique characters (entire string is the answer), strings with all same characters (answer is 1), empty strings (returns empty string), and strings where the longest substring appears at the end (window expands to include it).

An alternative O(n²) approach would check all possible substrings, but the sliding window approach is optimal. The time complexity is O(n) as each character is visited at most twice (once by end pointer, once when start pointer moves past it). The space complexity is O(min(n, m)) where m is the character set size, as we only store characters we've encountered.

Think of this as finding the longest stretch of unique items: you expand your window until you see a duplicate, then shrink from the left until the duplicate is removed, always tracking the longest unique stretch you've found.`,
	},
	{
		ID:          143,
		Title:       "Underscorify Substring",
		Description: "Write a function that takes in two strings, a main string and a potential substring of the main string, and returns a version of the main string with every instance of the substring in it wrapped between underscores. If two or more instances of the substring in the main string overlap each other or sit right next to each other, the underscores relevant to these instances should only appear on the far left of the leftmost instance and on the far right of the rightmost instance.",
		Difficulty:  "Medium",
		Topic:       "Strings",
		Signature:   "func underscorifySubstring(str string, substring string) string",
		TestCases: []TestCase{
			{Input: "str = \"testthis is a testtest to see if testestest it works\", substring = \"test\"", Expected: "\"_test_this is a _testtest_ to see if _testestest_ it works\""},
		},
		Solution: `func underscorifySubstring(str string, substring string) string {
	locations := collapse(getLocations(str, substring))
	return underscorify(str, locations)
}

func getLocations(str string, substring string) [][]int {
	locations := [][]int{}
	startIdx := 0
	for startIdx < len(str) {
		nextIdx := indexOf(str, substring, startIdx)
		if nextIdx != -1 {
			locations = append(locations, []int{nextIdx, nextIdx + len(substring)})
			startIdx = nextIdx + 1
		} else {
			break
		}
	}
	return locations
}

func indexOf(str string, substring string, startIdx int) int {
	for i := startIdx; i <= len(str)-len(substring); i++ {
		if str[i:i+len(substring)] == substring {
			return i
		}
	}
	return -1
}

func collapse(locations [][]int) [][]int {
	if len(locations) == 0 {
		return locations
	}
	newLocations := [][]int{locations[0]}
	previous := newLocations[0]
	for i := 1; i < len(locations); i++ {
		current := locations[i]
		if current[0] <= previous[1] {
			previous[1] = current[1]
		} else {
			newLocations = append(newLocations, current)
			previous = current
		}
	}
	return newLocations
}

func underscorify(str string, locations [][]int) string {
	locationsIdx := 0
	stringIdx := 0
	inBetweenUnderscores := false
	var finalChars []rune
	i := 0
	for i < len(str) && locationsIdx < len(locations) {
		if i == locations[locationsIdx][stringIdx] {
			finalChars = append(finalChars, '_')
			inBetweenUnderscores = !inBetweenUnderscores
			if !inBetweenUnderscores {
				locationsIdx++
			}
			stringIdx = 1 - stringIdx
		}
		finalChars = append(finalChars, rune(str[i]))
		i++
	}
	if locationsIdx < len(locations) {
		finalChars = append(finalChars, '_')
	} else if i < len(str) {
		finalChars = append(finalChars, []rune(str[i:])...)
	}
	return string(finalChars)
}`,
		PythonSolution: `def underscorifySubstring(string: str, substring: str) -> str:
    locations = collapse(get_locations(string, substring))
    return underscorify(string, locations)

def get_locations(string, substring):
    locations = []
    start_idx = 0
    while start_idx < len(string):
        next_idx = string.find(substring, start_idx)
        if next_idx != -1:
            locations.append([next_idx, next_idx + len(substring)])
            start_idx = next_idx + 1
        else:
            break
    return locations

def collapse(locations):
    if not locations:
        return locations
    new_locations = [locations[0]]
    previous = new_locations[0]
    for i in range(1, len(locations)):
        current = locations[i]
        if current[0] <= previous[1]:
            previous[1] = current[1]
        else:
            new_locations.append(current)
            previous = current
    return new_locations

def underscorify(string, locations):
    locations_idx = 0
    string_idx = 0
    in_between_underscores = False
    final_chars = []
    i = 0
    while i < len(string) and locations_idx < len(locations):
        if i == locations[locations_idx][string_idx]:
            final_chars.append('_')
            in_between_underscores = not in_between_underscores
            if not in_between_underscores:
                locations_idx += 1
            string_idx = 1 - string_idx
        final_chars.append(string[i])
        i += 1
    if locations_idx < len(locations):
        final_chars.append('_')
    elif i < len(string):
        final_chars.extend(string[i:])
    return ''.join(final_chars)`,
		Explanation: `Underscorifying a substring requires finding all occurrences of the substring, handling overlapping and adjacent occurrences, then inserting underscores at the appropriate positions. The challenge is collapsing overlapping or adjacent ranges into single underscore regions.

The algorithm first finds all locations where the substring appears in the main string. Then it collapses these locations: if two locations overlap (end of first >= start of second) or are adjacent (end of first == start of second), they're merged into a single range. Finally, it inserts underscores at the boundaries of these collapsed ranges.

The collapsing step is crucial: it ensures that overlapping substrings like "testtest" (where "test" appears at positions 0 and 4, overlapping) are treated as a single region with underscores only at the far left and far right, not between overlapping instances.

Edge cases handled include: no occurrences (return original string), non-overlapping occurrences (each gets underscores), overlapping occurrences (collapsed into one region), adjacent occurrences (collapsed into one region), and substring equals main string (entire string wrapped).

An alternative approach would insert underscores as we find occurrences, but that's more complex with overlapping cases. The two-phase approach (find then collapse) is cleaner. The time complexity is O(n*m) where n is main string length and m is substring length for finding all occurrences, plus O(n) for collapsing and inserting. The space complexity is O(n) for storing locations and the result string.

Think of this as highlighting text: you find all places where the text appears, merge any highlights that touch or overlap, then draw a single underline for each merged region.`,
	},
	{
		ID:          144,
		Title:       "Pattern Matcher",
		Description: "You're given two non-empty strings. The first one is a pattern consisting of only \"x\"s and / or \"y\"s; the other one is a normal string of alphanumeric characters. Write a function that checks whether the normal string matches the pattern. A string S0 is said to match a pattern if replacing all \"x\"s in the pattern with some non-empty substring S1 of S0 and replacing all \"y\"s in the pattern with some non-empty substring S2 of S0 yields the same string S0.",
		Difficulty:  "Medium",
		Topic:       "Strings",
		Signature:   "func patternMatcher(pattern string, str string) []string",
		TestCases: []TestCase{
			{Input: "pattern = \"xxyxxy\", str = \"gogopowerrangergogopowerranger\"", Expected: "[\"go\", \"powerranger\"]"},
		},
		Solution: `func patternMatcher(pattern string, str string) []string {
	if len(pattern) > len(str) {
		return []string{}
	}
	newPattern := getNewPattern(pattern)
	didSwitch := newPattern[0] != rune(pattern[0])
	counts := map[rune]int{'x': 0, 'y': 0}
	firstYPos := getCountsAndFirstYPos(newPattern, counts)
	if counts['y'] != 0 {
		for lenOfX := 1; lenOfX < len(str); lenOfX++ {
			lenOfY := (len(str) - lenOfX*counts['x']) / counts['y']
			if lenOfY <= 0 || lenOfY%1 != 0 {
				continue
			}
			yIdx := firstYPos * lenOfX
			x := str[:lenOfX]
			y := str[yIdx : yIdx+lenOfY]
			potentialMatch := buildPattern(newPattern, x, y)
			if str == potentialMatch {
				if !didSwitch {
					return []string{x, y}
				}
				return []string{y, x}
			}
		}
	} else {
		lenOfX := len(str) / counts['x']
		if lenOfX%1 == 0 {
			x := str[:lenOfX]
			potentialMatch := strings.Repeat(x, counts['x'])
			if str == potentialMatch {
				if !didSwitch {
					return []string{x, ""}
				}
				return []string{"", x}
			}
		}
	}
	return []string{}
}

func getNewPattern(pattern string) string {
	patternLetters := []rune(pattern)
	if patternLetters[0] == 'x' {
		return pattern
	}
	newPattern := make([]rune, len(patternLetters))
	for i := 0; i < len(patternLetters); i++ {
		if patternLetters[i] == 'x' {
			newPattern[i] = 'y'
		} else {
			newPattern[i] = 'x'
		}
	}
	return string(newPattern)
}

func getCountsAndFirstYPos(pattern string, counts map[rune]int) int {
	firstYPos := -1
	for i, char := range pattern {
		counts[char]++
		if char == 'y' && firstYPos == -1 {
			firstYPos = i
		}
	}
	return firstYPos
}

func buildPattern(pattern string, x string, y string) string {
	var result strings.Builder
	for _, char := range pattern {
		if char == 'x' {
			result.WriteString(x)
		} else {
			result.WriteString(y)
		}
	}
	return result.String()
}`,
		PythonSolution: `def patternMatcher(pattern: str, string: str) -> List[str]:
    if len(pattern) > len(string):
        return []
    new_pattern = get_new_pattern(pattern)
    did_switch = new_pattern[0] != pattern[0]
    counts = {'x': 0, 'y': 0}
    first_y_pos = get_counts_and_first_y_pos(new_pattern, counts)
    if counts['y'] != 0:
        for len_of_x in range(1, len(string)):
            len_of_y = (len(string) - len_of_x * counts['x']) / counts['y']
            if len_of_y <= 0 or len_of_y % 1 != 0:
                continue
            len_of_y = int(len_of_y)
            y_idx = first_y_pos * len_of_x
            x = string[:len_of_x]
            y = string[y_idx:y_idx + len_of_y]
            potential_match = build_pattern(new_pattern, x, y)
            if string == potential_match:
                if not did_switch:
                    return [x, y]
                return [y, x]
    else:
        len_of_x = len(string) / counts['x']
        if len_of_x % 1 == 0:
            len_of_x = int(len_of_x)
            x = string[:len_of_x]
            potential_match = x * counts['x']
            if string == potential_match:
                if not did_switch:
                    return [x, '']
                return ['', x]
    return []

def get_new_pattern(pattern):
    pattern_letters = list(pattern)
    if pattern_letters[0] == 'x':
        return pattern
    return ''.join('x' if char == 'y' else 'y' for char in pattern_letters)

def get_counts_and_first_y_pos(pattern, counts):
    first_y_pos = -1
    for i, char in enumerate(pattern):
        counts[char] += 1
        if char == 'y' and first_y_pos == -1:
            first_y_pos = i
    return first_y_pos

def build_pattern(pattern, x, y):
    result = []
    for char in pattern:
        if char == 'x':
            result.append(x)
        else:
            result.append(y)
    return ''.join(result)`,
		Explanation: `Matching a pattern string where 'x' and 'y' represent variable-length substrings requires trying different possible lengths for 'x' and calculating the corresponding length for 'y', then verifying if those lengths produce a valid match.

The algorithm first normalizes the pattern to always start with 'x' by swapping x and y if needed. Then it tries all possible lengths for 'x' from 1 to the string length. For each x length, it calculates what 'y' must be based on the pattern structure and the remaining string length. It then builds the pattern by replacing 'x' and 'y' with their corresponding substrings and verifies if it matches the original string.

The key insight is that once we know the length of 'x', we can determine the length of 'y' from the pattern structure and total string length. We then extract the actual 'x' and 'y' values from the string and verify that using these values throughout the pattern produces the original string.

Edge cases handled include: patterns starting with 'y' (normalized to start with 'x'), patterns with only 'x' or only 'y' (handled by length calculations), patterns where x and y must be the same (detected during verification), and patterns where no valid lengths exist (returns empty list).

An alternative approach would use backtracking, but trying all x lengths is more straightforward. The time complexity is O(n² + m) where n is string length and m is pattern length, as we try O(n) x lengths and for each we build and verify a pattern of length m. The space complexity is O(n + m) for storing the pattern and result.

Think of this as solving an equation: if you know the pattern structure and total length, you can solve for the lengths of x and y, then verify if those values work throughout the pattern.`,
	},
	{
		ID:          145,
		Title:       "Multi String Search",
		Description: "Write a function that takes in a big string and an array of small strings, all of which are smaller in length than the big string, and returns an array of booleans, where each boolean represents whether the small string at that index in the array of small strings is contained in the big string.",
		Difficulty:  "Medium",
		Topic:       "Strings",
		Signature:   "func multiStringSearch(bigString string, smallStrings []string) []bool",
		TestCases: []TestCase{
			{Input: "bigString = \"this is a big string\", smallStrings = [\"this\", \"yo\", \"is\", \"a\", \"bigger\", \"string\", \"kappa\"]", Expected: "[true, false, true, true, false, true, false]"},
		},
		Solution: `func multiStringSearch(bigString string, smallStrings []string) []bool {
	trie := NewTrie()
	for _, smallString := range smallStrings {
		trie.Insert(smallString)
	}
	containedStrings := make(map[string]bool)
	for i := 0; i < len(bigString); i++ {
		findSmallStringsIn(bigString, i, trie, containedStrings)
	}
	solution := make([]bool, len(smallStrings))
	for i, smallString := range smallStrings {
		solution[i] = containedStrings[smallString]
	}
	return solution
}

func findSmallStringsIn(str string, startIdx int, trie *Trie, containedStrings map[string]bool) {
	currentNode := trie.root
	for i := startIdx; i < len(str); i++ {
		char := str[i]
		if _, found := currentNode.children[char]; !found {
			break
		}
		currentNode = currentNode.children[char]
		if currentNode.endWord != "" {
			containedStrings[currentNode.endWord] = true
		}
	}
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

func (t *Trie) Insert(str string) {
	node := t.root
	for _, char := range str {
		if _, found := node.children[char]; !found {
			node.children[char] = &TrieNode{children: make(map[rune]*TrieNode)}
		}
		node = node.children[char]
	}
	node.endWord = str
}`,
		PythonSolution: `def multiStringSearch(bigString: str, smallStrings: List[str]) -> List[bool]:
    trie = Trie()
    for small_string in smallStrings:
        trie.insert(small_string)
    contained_strings = {}
    for i in range(len(bigString)):
        find_small_strings_in(bigString, i, trie, contained_strings)
    return [contained_strings.get(small_string, False) for small_string in smallStrings]

def find_small_strings_in(string, start_idx, trie, contained_strings):
    current_node = trie.root
    for i in range(start_idx, len(string)):
        char = string[i]
        if char not in current_node.children:
            break
        current_node = current_node.children[char]
        if current_node.end_word:
            contained_strings[current_node.end_word] = True

class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_word = None

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, string):
        node = self.root
        for char in string:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.end_word = string`,
		Explanation: `Finding all small strings within a big string efficiently requires a data structure that allows fast prefix matching. A trie (prefix tree) is perfect for this, as it organizes strings by their prefixes, enabling efficient traversal.

The algorithm first builds a trie from all small strings. Each node in the trie represents a character, and paths from root to nodes with end_word set represent complete strings. Then, for each starting position in the big string, it traverses the trie character by character. If it reaches a node with end_word set, it has found a match and adds it to the results.

The trie allows us to check multiple small strings simultaneously as we traverse the big string. When we encounter a character that doesn't exist in the trie at the current position, we know no small string starting from that position can match, so we can stop that traversal.

Edge cases handled include: small strings that are prefixes of other small strings (both found correctly), big string containing no small strings (returns empty list), small strings appearing multiple times (all occurrences found), and small strings that overlap in the big string (all found correctly).

An alternative approach would check each small string against each position in the big string, but that's O(b*ns*s) where s is average small string length. The trie approach is more efficient, achieving O(b² + ns) time where b is big string length and ns is total small strings length. The space complexity is O(ns) for the trie.

Think of this as having a dictionary organized by first letters: you start reading the big string, and for each position, you look up words in the dictionary that start with that character, following the trie structure to find complete matches.`,
	},
	{
		ID:          146,
		Title:       "Min Max Stack Construction",
		Description: "Write a MinMaxStack class for a Min Max Stack. The class should support pushing and popping values on and off the stack, peeking at values at the top of the stack, and getting both the minimum and maximum values in the stack at any given point in time. All class methods, when considered independently, should run in constant time and with constant space.",
		Difficulty:  "Medium",
		Topic:       "Stacks",
		Signature:   "type MinMaxStack struct{}\nfunc NewMinMaxStack() *MinMaxStack\nfunc (stack *MinMaxStack) Peek() int\nfunc (stack *MinMaxStack) Pop() int\nfunc (stack *MinMaxStack) Push(number int)\nfunc (stack *MinMaxStack) GetMin() int\nfunc (stack *MinMaxStack) GetMax() int",
		TestCases: []TestCase{
			{Input: "Stack operations", Expected: "Correct min/max values"},
		},
		Solution: `type MinMaxStack struct {
	stack       []int
	minMaxStack []map[string]int
}

func NewMinMaxStack() *MinMaxStack {
	return &MinMaxStack{
		stack:       []int{},
		minMaxStack: []map[string]int{},
	}
}

func (stack *MinMaxStack) Peek() int {
	return stack.stack[len(stack.stack)-1]
}

func (stack *MinMaxStack) Pop() int {
	stack.minMaxStack = stack.minMaxStack[:len(stack.minMaxStack)-1]
	value := stack.stack[len(stack.stack)-1]
	stack.stack = stack.stack[:len(stack.stack)-1]
	return value
}

func (stack *MinMaxStack) Push(number int) {
	newMinMax := map[string]int{"min": number, "max": number}
	if len(stack.minMaxStack) > 0 {
		lastMinMax := stack.minMaxStack[len(stack.minMaxStack)-1]
		newMinMax["min"] = min(lastMinMax["min"], number)
		newMinMax["max"] = max(lastMinMax["max"], number)
	}
	stack.minMaxStack = append(stack.minMaxStack, newMinMax)
	stack.stack = append(stack.stack, number)
}

func (stack *MinMaxStack) GetMin() int {
	return stack.minMaxStack[len(stack.minMaxStack)-1]["min"]
}

func (stack *MinMaxStack) GetMax() int {
	return stack.minMaxStack[len(stack.minMaxStack)-1]["max"]
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
		PythonSolution: `class MinMaxStack:
    def __init__(self):
        self.stack = []
        self.min_max_stack = []
    
    def peek(self):
        return self.stack[-1]
    
    def pop(self):
        self.min_max_stack.pop()
        return self.stack.pop()
    
    def push(self, number):
        new_min_max = {"min": number, "max": number}
        if self.min_max_stack:
            last_min_max = self.min_max_stack[-1]
            new_min_max["min"] = min(last_min_max["min"], number)
            new_min_max["max"] = max(last_min_max["max"], number)
        self.min_max_stack.append(new_min_max)
        self.stack.append(number)
    
    def getMin(self):
        return self.min_max_stack[-1]["min"]
    
    def getMax(self):
        return self.min_max_stack[-1]["max"]`,
		Explanation: `Implementing a stack with O(1) min and max operations requires maintaining additional information alongside the main stack. The key insight is to track the min and max values at each stack level, not just globally.

The algorithm maintains two stacks: one for the actual values and one for min/max pairs. When pushing a value, it calculates the new min and max by comparing the new value with the previous level's min/max, then pushes both the value and the min/max pair. When popping, it removes from both stacks. Getting min or max is simply returning the top of the min/max stack.

This approach ensures that each stack level "remembers" what the min and max were when that level was the top, allowing O(1) access even after other values are pushed and popped above it.

Edge cases handled include: empty stack (getMin/getMax return sentinel values or raise error), single element (min and max are the same), all increasing values (max updates, min stays same), all decreasing values (min updates, max stays same), and duplicate min/max values (handled correctly).

An alternative approach would recalculate min/max on each query, but that's O(n) time. Another approach would use a heap, but that doesn't maintain stack order. The parallel stack approach achieves O(1) for all operations with O(n) space for the additional min/max stack.

Think of this as keeping a running record: as you add items to a stack, you also keep track of the smallest and largest items you've seen so far at each level, so you can instantly recall them without searching.`,
	},
	{
		ID:          147,
		Title:       "Balanced Brackets",
		Description: "Write a function that takes in a string made up of brackets (\"(\", \"[\", \"{\", \")\", \"]\", \"}\") and other optional characters. The function should return a boolean representing whether the string is balanced with regards to brackets. A string is said to be balanced if it has as many opening brackets of a certain type as it has closing brackets of that type and if no bracket is unmatched. Note that an opening bracket can't match a corresponding closing bracket that comes before it, and similarly, a closing bracket can't match a corresponding opening bracket that comes after it. Also, brackets can't overlap each other as in \"[(])\".",
		Difficulty:  "Medium",
		Topic:       "Stack",
		Signature:   "func balancedBrackets(str string) bool",
		TestCases: []TestCase{
			{Input: "str = \"([])(){}(())()()\"", Expected: "true"},
			{Input: "str = \"()[]{}{\"", Expected: "false"},
		},
		Solution: `func balancedBrackets(str string) bool {
	openingBrackets := "([{"
	closingBrackets := ")]}"
	matchingBrackets := map[rune]rune{')': '(', ']': '[', '}': '{'}
	stack := []rune{}
	for _, char := range str {
		if strings.ContainsRune(openingBrackets, char) {
			stack = append(stack, char)
		} else if strings.ContainsRune(closingBrackets, char) {
			if len(stack) == 0 {
				return false
			}
			if stack[len(stack)-1] == matchingBrackets[char] {
				stack = stack[:len(stack)-1]
			} else {
				return false
			}
		}
	}
	return len(stack) == 0
}`,
		PythonSolution: `def balancedBrackets(string: str) -> bool:
    opening_brackets = "([{"
    closing_brackets = ")]}"
    matching_brackets = {')': '(', ']': '[', '}': '{'}
    stack = []
    for char in string:
        if char in opening_brackets:
            stack.append(char)
        elif char in closing_brackets:
            if len(stack) == 0:
                return False
            if stack[-1] == matching_brackets[char]:
                stack.pop()
            else:
                return False
    return len(stack) == 0`,
		Explanation: `Validating balanced brackets requires checking that every opening bracket has a matching closing bracket in the correct order. A stack is the perfect data structure for this, as it naturally handles nested structures.

The algorithm uses a stack to track opening brackets. As we iterate through the string, when we encounter an opening bracket ('(', '[', '{'), we push it onto the stack. When we encounter a closing bracket, we check if the stack is empty (unmatched closing bracket) or if the top of the stack matches the closing bracket. If it matches, we pop from the stack. If it doesn't match or the stack is empty, the string is invalid.

At the end, if the stack is empty, all brackets were matched correctly. If the stack is not empty, there are unmatched opening brackets.

Edge cases handled include: empty string (valid, returns true), string with only opening brackets (invalid, stack not empty), string with only closing brackets (invalid, stack empty when closing encountered), mismatched bracket types (invalid, detected when popping), and properly nested brackets (valid, handled correctly).

An alternative approach would use a counter for each bracket type, but that doesn't handle nesting correctly (e.g., "([)]" would pass with counters but fail with stack). The stack approach correctly handles nesting. The time complexity is O(n) where n is string length, and space complexity is O(n) for the stack in the worst case of all opening brackets.

Think of this as matching parentheses in math: you open a parenthesis, and it must be closed before any outer parenthesis is closed. The stack tracks which parentheses are still open, ensuring proper nesting.`,
	},
	{
		ID:          148,
		Title:       "Sunset Views",
		Description: "Given an array of buildings and a direction that all of the buildings face, return an array of the indices of the buildings that can see the sunset. A building can see the sunset if it's strictly taller than all of the buildings that come after it in the direction that it faces. The input array named buildings contains positive, non-zero integers representing the heights of the buildings. A building at index i thus has a height denoted by buildings[i]. All of the buildings face the same direction, and this direction is either \"EAST\" or \"WEST\". In relation to the input array, you can interpret these directions as right for east and left for west.",
		Difficulty:  "Medium",
		Topic:       "Stack",
		Signature:   "func sunsetViews(buildings []int, direction string) []int",
		TestCases: []TestCase{
			{Input: "buildings = [3, 5, 4, 4, 3, 1, 3, 2], direction = \"EAST\"", Expected: "[1, 3, 6, 7]"},
			{Input: "buildings = [3, 5, 4, 4, 3, 1, 3, 2], direction = \"WEST\"", Expected: "[0, 1]"},
		},
		Solution: `func sunsetViews(buildings []int, direction string) []int {
	candidateBuildings := []int{}
	startIdx := len(buildings) - 1
	step := -1
	if direction == "WEST" {
		startIdx = 0
		step = 1
	}
	idx := startIdx
	for idx >= 0 && idx < len(buildings) {
		buildingHeight := buildings[idx]
		for len(candidateBuildings) > 0 && buildings[candidateBuildings[len(candidateBuildings)-1]] <= buildingHeight {
			candidateBuildings = candidateBuildings[:len(candidateBuildings)-1]
		}
		candidateBuildings = append(candidateBuildings, idx)
		idx += step
	}
	if direction == "EAST" {
		reverse(candidateBuildings)
	}
	return candidateBuildings
}

func reverse(array []int) {
	for i, j := 0, len(array)-1; i < j; i, j = i+1, j-1 {
		array[i], array[j] = array[j], array[i]
	}
}`,
		PythonSolution: `def sunsetViews(buildings: List[int], direction: str) -> List[int]:
    candidate_buildings = []
    start_idx = len(buildings) - 1
    step = -1
    if direction == "WEST":
        start_idx = 0
        step = 1
    idx = start_idx
    while idx >= 0 and idx < len(buildings):
        building_height = buildings[idx]
        while candidate_buildings and buildings[candidate_buildings[-1]] <= building_height:
            candidate_buildings.pop()
        candidate_buildings.append(idx)
        idx += step
    if direction == "EAST":
        candidate_buildings.reverse()
    return candidate_buildings`,
		Explanation: `Finding buildings that can see the sunset requires determining which buildings are taller than all buildings in the direction they face. A stack-based approach efficiently tracks candidate buildings and removes those blocked by taller buildings.

The algorithm uses a stack to maintain candidate buildings. For EAST-facing buildings, we traverse from right to left (opposite direction), and for WEST-facing buildings, we traverse from left to right. As we encounter each building, we remove from the stack any buildings that are shorter than the current building (they're blocked). Then we add the current building to the stack if it's taller than the previous top.

The key insight is that a building can only see the sunset if it's taller than all buildings between it and the horizon in its facing direction. By processing buildings in the opposite direction and maintaining a stack of increasingly taller buildings, we efficiently identify which buildings have an unobstructed view.

Edge cases handled include: all buildings same height (only first building sees sunset), buildings in increasing order (all see sunset), buildings in decreasing order (only last sees sunset), and single building (always sees sunset).

An alternative approach would compare each building with all buildings in its direction, but that's O(n²) time. The stack approach achieves O(n) time as each building is pushed and popped at most once. The space complexity is O(n) for the stack.

Think of this as looking at a skyline: buildings block the view of shorter buildings behind them, so only buildings taller than all those in front can see the sunset.`,
	},
	{
		ID:          149,
		Title:       "Best Digits",
		Description: "Write a function that takes in a positive integer num and an integer numDigits. Remove numDigits digits from the number so that the number represented by the remaining digits is as large as possible. The order of the digits in num cannot be changed.",
		Difficulty:  "Medium",
		Topic:       "Stack",
		Signature:   "func bestDigits(number string, numDigits int) string",
		TestCases: []TestCase{
			{Input: "number = \"462839\", numDigits = 2", Expected: "\"6839\""},
			{Input: "number = \"1432219\", numDigits = 3", Expected: "\"3219\""},
		},
		Solution: `func bestDigits(number string, numDigits int) string {
	stack := []rune{}
	removeCount := numDigits
	for _, digit := range number {
		for len(stack) > 0 && removeCount > 0 && stack[len(stack)-1] < digit {
			stack = stack[:len(stack)-1]
			removeCount--
		}
		stack = append(stack, digit)
	}
	for removeCount > 0 {
		stack = stack[:len(stack)-1]
		removeCount--
	}
	return string(stack)
}`,
		PythonSolution: `def bestDigits(number: str, numDigits: int) -> str:
    stack = []
    remove_count = numDigits
    for digit in number:
        while stack and remove_count > 0 and stack[-1] < digit:
            stack.pop()
            remove_count -= 1
        stack.append(digit)
    while remove_count > 0:
        stack.pop()
        remove_count -= 1
    return ''.join(stack)`,
		Explanation: `Finding the largest number by removing k digits requires a greedy approach: remove smaller digits from the left when we encounter larger digits, as this maximizes the remaining number's value.

The algorithm uses a stack to build the result. As we process each digit, if the stack is not empty, the current digit is larger than the top of the stack, and we still have removals remaining, we pop from the stack (removing the smaller digit). This continues until we can't remove more or the stack is empty. Then we push the current digit. After processing all digits, if we still have removals remaining, we remove digits from the end (they're the smallest remaining).

The greedy strategy works because removing a smaller digit early (when a larger digit comes after) always results in a larger number than keeping the smaller digit. We prioritize removing from the left to maximize the most significant digits.

Edge cases handled include: removing all digits (returns empty string or "0"), digits in decreasing order (remove from end), digits in increasing order (remove from beginning), and numbers with leading zeros (handled correctly by string operations).

An alternative approach would try all combinations of k removals, but that's exponential. The greedy stack approach achieves O(n) time where n is the number length, as each digit is pushed and popped at most once. The space complexity is O(n) for the stack.

Think of this as editing a number to make it as large as possible: you remove smaller digits from the left when you see larger digits coming, always keeping the largest possible digits in the most significant positions.`,
	},
	{
		ID:          150,
		Title:       "Sort Stack",
		Description: "Write a function that takes in an array of integers representing a stack, recursively sorts the stack in place (i.e., doesn't create a brand new array), and returns it. The array must be treated as a stack, with the end of the array as the top of the stack. Therefore, you're only allowed to pop from the top of the stack by removing elements from the end of the array, and you're only allowed to push to the top of the stack by appending elements to the end of the array. You're not allowed to perform any other operations on the array, including accessing elements (except for the last one), moving elements, etc. You're also not allowed to use any other data structures, and the solution must be recursive.",
		Difficulty:  "Medium",
		Topic:       "Stack",
		Signature:   "func sortStack(stack []int) []int",
		TestCases: []TestCase{
			{Input: "stack = [-5, 2, -2, 4, 3, 1]", Expected: "[-5, -2, 1, 2, 3, 4]"},
		},
		Solution: `func sortStack(stack []int) []int {
	if len(stack) == 0 {
		return stack
	}
	top := stack[len(stack)-1]
	stack = stack[:len(stack)-1]
	sortStack(stack)
	insertInSortedOrder(&stack, top)
	return stack
}

func insertInSortedOrder(stack *[]int, value int) {
	if len(*stack) == 0 || (*stack)[len(*stack)-1] <= value {
		*stack = append(*stack, value)
		return
	}
	top := (*stack)[len(*stack)-1]
	*stack = (*stack)[:len(*stack)-1]
	insertInSortedOrder(stack, value)
	*stack = append(*stack, top)
}`,
		PythonSolution: `def sortStack(stack: List[int]) -> List[int]:
    if len(stack) == 0:
        return stack
    top = stack.pop()
    sortStack(stack)
    insert_in_sorted_order(stack, top)
    return stack

def insert_in_sorted_order(stack, value):
    if len(stack) == 0 or stack[-1] <= value:
        stack.append(value)
        return
    top = stack.pop()
    insert_in_sorted_order(stack, value)
    stack.append(top)`,
		Explanation: `Sorting a stack recursively without using additional data structures requires a clever approach: we sort the stack by recursively sorting all but the top element, then inserting the top element in its correct sorted position.

The algorithm works by popping the top element, recursively sorting the remaining stack, then inserting the popped element in the correct position within the now-sorted stack. The insert operation itself is recursive: if the stack is empty or the value to insert is smaller than the top, we push it. Otherwise, we pop the top, recursively insert the value, then push the popped top back.

This approach maintains the stack property (only accessing the top) while achieving sorting through recursion. The recursive nature naturally handles the sorting of sub-stacks, and the insert operation ensures the top element is placed correctly.

Edge cases handled include: empty stack (already sorted), single-element stack (already sorted), already sorted stack (insert operations maintain order), reverse-sorted stack (all elements need to be inserted), and stack with duplicates (handled correctly by comparison).

An alternative iterative approach would require an auxiliary stack, violating the constraint. The recursive approach achieves sorting using only the call stack. The time complexity is O(n²) in the worst case as each element may need to be inserted by popping all elements above it. The space complexity is O(n) for the recursion stack.

Think of this as sorting a deck of cards using only one hand: you take the top card, sort the rest of the deck, then insert the card you took in the right position by temporarily moving cards out of the way.`,
	},
	{
		ID:          151,
		Title:       "Next Greater Element",
		Description: "Write a function that takes in an array of integers and returns a new array containing, at each index, the next element in the input array that's greater than the element at that index in the input array. In other words, your function should return a new array where outputArray[i] is the next element in the input array that's greater than inputArray[i]. If there's no such next greater element for a particular index, the value at that index in the output array should be -1. For example, given array = [1, 2], your function should return [2, -1].",
		Difficulty:  "Medium",
		Topic:       "Stack",
		Signature:   "func nextGreaterElement(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = [2, 5, -3, -4, 6, 7, 2]", Expected: "[5, 6, 6, 6, 7, -1, 5]"},
		},
		Solution: `func nextGreaterElement(array []int) []int {
	result := make([]int, len(array))
	for i := range result {
		result[i] = -1
	}
	stack := []int{}
	for idx := 0; idx < 2*len(array); idx++ {
		circularIdx := idx % len(array)
		for len(stack) > 0 && array[stack[len(stack)-1]] < array[circularIdx] {
			top := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			result[top] = array[circularIdx]
		}
		stack = append(stack, circularIdx)
	}
	return result
}`,
		PythonSolution: `def nextGreaterElement(array: List[int]) -> List[int]:
    result = [-1] * len(array)
    stack = []
    for idx in range(2 * len(array)):
        circular_idx = idx % len(array)
        while stack and array[stack[-1]] < array[circular_idx]:
            top = stack.pop()
            result[top] = array[circular_idx]
        stack.append(circular_idx)
    return result`,
		Explanation: `Finding the next greater element for each position in a circular array requires checking both the remaining elements and wrapping around to the beginning. A monotonic stack approach efficiently handles this.

The algorithm uses a stack to store indices of elements for which we haven't found the next greater element yet. We traverse the array twice (to handle the circular nature): for each element, we check if it's greater than the element at the top of the stack. If so, we've found the next greater element for that stack top, so we pop it and record the result. Then we push the current index onto the stack.

The circular traversal is handled by iterating through indices 0 to 2n-1, using modulo to wrap around. This ensures we check all possible next greater elements, including those that wrap around from the end to the beginning.

Edge cases handled include: all elements same (all get -1), elements in decreasing order (each finds next in next cycle), elements in increasing order (each finds immediate next), and single element (returns -1).

An alternative O(n²) approach would check all elements for each position. The stack approach is optimal, achieving O(n) time as each element is pushed and popped at most once. The space complexity is O(n) for the stack.

Think of this as finding the next taller person in a circle: you go around the circle, and when you find someone taller than people waiting, you tell those waiting people who their next taller person is, then continue.`,
	},
	{
		ID:          152,
		Title:       "Reverse Polish Notation",
		Description: "Write a function that takes in an arithmetic expression in Reverse Polish Notation (RPN) and returns the evaluated result. The expression is given as an array of strings. Each string is either an integer or an operator (\"+\", \"-\", \"*\", \"/\").",
		Difficulty:  "Medium",
		Topic:       "Stack",
		Signature:   "func reversePolishNotation(tokens []string) int",
		TestCases: []TestCase{
			{Input: "tokens = [\"3\", \"2\", \"+\", \"7\", \"*\"]", Expected: "35"},
			{Input: "tokens = [\"3\", \"2\", \"+\", \"7\", \"*\", \"2\", \"-\"]", Expected: "33"},
		},
		Solution: `import "strconv"

func reversePolishNotation(tokens []string) int {
	stack := []int{}
	for _, token := range tokens {
		if token == "+" || token == "-" || token == "*" || token == "/" {
			second := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			first := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if token == "+" {
				stack = append(stack, first+second)
			} else if token == "-" {
				stack = append(stack, first-second)
			} else if token == "*" {
				stack = append(stack, first*second)
			} else {
				stack = append(stack, first/second)
			}
		} else {
			num, _ := strconv.Atoi(token)
			stack = append(stack, num)
		}
	}
	return stack[0]
}`,
		PythonSolution: `def reversePolishNotation(tokens: List[str]) -> int:
    stack = []
    for token in tokens:
        if token in ['+', '-', '*', '/']:
            second = stack.pop()
            first = stack.pop()
            if token == '+':
                stack.append(first + second)
            elif token == '-':
                stack.append(first - second)
            elif token == '*':
                stack.append(first * second)
            else:
                stack.append(first // second)
        else:
            stack.append(int(token))
    return stack[0]`,
		Explanation: `Evaluating Reverse Polish Notation (RPN) requires processing operators and operands in postfix order, where operators come after their operands. A stack is perfect for this, as it naturally handles the order of operations.

The algorithm processes tokens left to right. When encountering a number, it's pushed onto the stack. When encountering an operator, the algorithm pops the top two numbers (the operands), applies the operator, and pushes the result back. The order of popping matters: the second popped is the first operand, and the first popped is the second operand (for non-commutative operations like subtraction and division).

This approach correctly evaluates RPN because operators are applied as soon as both operands are available, which happens when we encounter the operator after its operands. The stack maintains the correct order for operations.

Edge cases handled include: single number (returns that number), division by zero (handled by language or requires error checking), negative numbers (handled correctly), and operations resulting in floats (integer division truncates in some languages).

An alternative approach would convert to infix notation first, but that's more complex. The stack-based evaluation is straightforward and efficient. The time complexity is O(n) where n is the number of tokens, as each token is processed once. The space complexity is O(n) for the stack in the worst case.

Think of this as a calculator that reads operations backwards: you see numbers and remember them, and when you see an operator, you take the last two numbers you remember, do the operation, and remember the result.`,
	},
	{
		ID:          153,
		Title:       "Colliding Asteroids",
		Description: "You're given an array of integers representing asteroids in a row. For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Each asteroid moves at the same speed. Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.",
		Difficulty:  "Medium",
		Topic:       "Stack",
		Signature:   "func collidingAsteroids(asteroids []int) []int",
		TestCases: []TestCase{
			{Input: "asteroids = [-3, 5, -8, 6, 7, -4, -7]", Expected: "[-3, -8, 6]"},
			{Input: "asteroids = [4, -1]", Expected: "[4]"},
		},
		Solution: `func collidingAsteroids(asteroids []int) []int {
	stack := []int{}
	for _, asteroid := range asteroids {
		if asteroid > 0 {
			stack = append(stack, asteroid)
		} else {
			for len(stack) > 0 && stack[len(stack)-1] > 0 && stack[len(stack)-1] < -asteroid {
				stack = stack[:len(stack)-1]
			}
			if len(stack) == 0 || stack[len(stack)-1] < 0 {
				stack = append(stack, asteroid)
			} else if stack[len(stack)-1] == -asteroid {
				stack = stack[:len(stack)-1]
			}
		}
	}
	return stack
}`,
		PythonSolution: `def collidingAsteroids(asteroids: List[int]) -> List[int]:
    stack = []
    for asteroid in asteroids:
        if asteroid > 0:
            stack.append(asteroid)
        else:
            while stack and stack[-1] > 0 and stack[-1] < -asteroid:
                stack.pop()
            if not stack or stack[-1] < 0:
                stack.append(asteroid)
            elif stack[-1] == -asteroid:
                stack.pop()
    return stack`,
		Explanation: `Simulating asteroid collisions requires tracking asteroids moving in opposite directions and handling collisions when they meet. A stack efficiently models this by representing asteroids that haven't collided yet.

The algorithm processes asteroids left to right. Right-moving asteroids (positive values) are always pushed onto the stack, as they move away from previously processed asteroids. Left-moving asteroids (negative values) collide with right-moving asteroids on the stack. We pop right-moving asteroids that are smaller than the current left-moving asteroid (they explode). If a right-moving asteroid is larger, the left-moving asteroid explodes and we stop. If they're equal, both explode.

The stack naturally represents the "surviving" asteroids that haven't been destroyed yet. Right-moving asteroids accumulate on the stack, and left-moving asteroids "collide" with them from the right, removing destroyed asteroids.

Edge cases handled include: all asteroids moving right (all survive), all asteroids moving left (all survive), asteroids of equal size colliding (both destroyed), and no collisions (all survive).

An alternative approach would simulate collisions iteratively, but that's less efficient. The stack approach processes each asteroid once, achieving O(n) time where n is the number of asteroids. The space complexity is O(n) for the stack.

Think of this as a line of people walking: people going right keep walking forward, but when someone going left meets them, the smaller person steps aside (is destroyed), and if they're the same size, both step aside.`,
	},
	{
		ID:          154,
		Title:       "Longest Palindromic Substring",
		Description: "Write a function that, given a string, returns its longest palindromic substring. A palindrome is defined as a string that's written the same forward and backward. Note that single-character strings are palindromes. You can assume that there will only be one longest palindromic substring.",
		Difficulty:  "Medium",
		Topic:       "Strings",
		Signature:   "func longestPalindromicSubstring(str string) string",
		TestCases: []TestCase{
			{Input: "str = \"abaxyzzyxf\"", Expected: "\"xyzzyx\""},
			{Input: "str = \"a\"", Expected: "\"a\""},
		},
		Solution: `func longestPalindromicSubstring(str string) string {
	longest := []int{0, 1}
	for i := 1; i < len(str); i++ {
		odd := getLongestPalindromeFrom(str, i-1, i+1)
		even := getLongestPalindromeFrom(str, i-1, i)
		longestOdd := odd[1] - odd[0]
		longestEven := even[1] - even[0]
		currentLongest := longest[1] - longest[0]
		if longestOdd > longestEven {
			if longestOdd > currentLongest {
				longest = odd
			}
		} else {
			if longestEven > currentLongest {
				longest = even
			}
		}
	}
	return str[longest[0]:longest[1]]
}

func getLongestPalindromeFrom(str string, leftIdx int, rightIdx int) []int {
	for leftIdx >= 0 && rightIdx < len(str) {
		if str[leftIdx] != str[rightIdx] {
			break
		}
		leftIdx--
		rightIdx++
	}
	return []int{leftIdx + 1, rightIdx}
}`,
		PythonSolution: `def longestPalindromicSubstring(string: str) -> str:
    longest = [0, 1]
    for i in range(1, len(string)):
        odd = get_longest_palindrome_from(string, i - 1, i + 1)
        even = get_longest_palindrome_from(string, i - 1, i)
        longest_odd = odd[1] - odd[0]
        longest_even = even[1] - even[0]
        current_longest = longest[1] - longest[0]
        if longest_odd > longest_even:
            if longest_odd > current_longest:
                longest = odd
        else:
            if longest_even > current_longest:
                longest = even
    return string[longest[0]:longest[1]]

def get_longest_palindrome_from(string, left_idx, right_idx):
    while left_idx >= 0 and right_idx < len(string):
        if string[left_idx] != string[right_idx]:
            break
        left_idx -= 1
        right_idx += 1
    return [left_idx + 1, right_idx]`,
		Explanation: `Finding the longest palindromic substring requires checking all possible palindromes efficiently. The expand-around-center approach checks each possible center position and expands outward to find the longest palindrome centered there.

The algorithm considers each character (and each gap between characters) as a potential center. For each center, it expands outward in both directions, comparing characters until they don't match. It tracks the longest palindrome found. We need to check both odd-length palindromes (centered at a character) and even-length palindromes (centered between characters).

This approach is efficient because it only checks palindromes that could potentially be the longest, avoiding redundant checks. By expanding from centers, we naturally find palindromes in order of increasing length.

Edge cases handled include: single character (longest is that character), entire string is palindrome (returns entire string), no palindromes longer than 1 (returns first character), and multiple palindromes of same length (returns first found).

An alternative O(n) approach uses Manacher's algorithm, but it's more complex. The expand-around-center approach is simpler and achieves O(n²) time where n is string length, as we check O(n) centers and expand O(n) times for each. The space complexity is O(1) as we only track indices.

Think of this as looking for symmetry: for each point in the string, you check how far you can extend equally to the left and right while maintaining symmetry, keeping track of the longest symmetric stretch you find.`,
	},
	{
		ID:          155,
		Title:       "Group Anagrams",
		Description: "Write a function that takes in an array of strings and groups anagrams together. Anagrams are strings made up of exactly the same letters, where order doesn't matter. For example, \"cinema\" and \"iceman\" are anagrams; similarly, \"foo\" and \"ofo\" are anagrams. Your function should return a list of anagram groups in no particular order.",
		Difficulty:  "Medium",
		Topic:       "Hash Tables",
		Signature:   "func groupAnagrams(words []string) [][]string",
		TestCases: []TestCase{
			{Input: "words = [\"yo\", \"act\", \"flop\", \"tac\", \"foo\", \"cat\", \"oy\", \"olfp\"]", Expected: "[[\"yo\", \"oy\"], [\"act\", \"tac\", \"cat\"], [\"flop\", \"olfp\"], [\"foo\"]]"},
		},
		Solution: `import "sort"

func groupAnagrams(words []string) [][]string {
	anagrams := make(map[string][]string)
	for _, word := range words {
		sortedWord := sortString(word)
		anagrams[sortedWord] = append(anagrams[sortedWord], word)
	}
	result := [][]string{}
	for _, group := range anagrams {
		result = append(result, group)
	}
	return result
}

func sortString(s string) string {
	runes := []rune(s)
	sort.Slice(runes, func(i, j int) bool {
		return runes[i] < runes[j]
	})
	return string(runes)
}`,
		PythonSolution: `def groupAnagrams(words: List[str]) -> List[List[str]]:
    anagrams = {}
    for word in words:
        sorted_word = ''.join(sorted(word))
        if sorted_word not in anagrams:
            anagrams[sorted_word] = []
        anagrams[sorted_word].append(word)
    return list(anagrams.values())`,
		Explanation: `Grouping anagrams requires identifying words that contain the same letters in different orders. The key insight is that sorting the letters of anagrams produces the same string, which can be used as a grouping key.

The algorithm creates a hash map where the key is the sorted version of each word's letters. Words with the same sorted key are anagrams and are grouped together in the hash map. After processing all words, we return the values of the hash map, which are lists of anagram groups.

This approach efficiently groups anagrams by leveraging the fact that anagrams have identical character compositions. Sorting provides a canonical representation that anagrams share, making grouping straightforward.

Edge cases handled include: no anagrams (each word in its own group), all words are anagrams (one large group), words with different lengths (cannot be anagrams, grouped separately), and words with special characters (handled by sorting).

An alternative approach would compare character frequencies for each pair of words, but that's O(w²*n) time. The sorting approach achieves O(w*n log n) time where w is the number of words and n is the average word length. The space complexity is O(w*n) for storing words and sorted keys.

Think of this as organizing words by their "fingerprint": you rearrange the letters alphabetically to create a unique signature, and words with the same signature are anagrams.`,
	},
	{
		ID:          156,
		Title:       "Valid IP Addresses",
		Description: "You're given a string of length 12 or smaller, containing only digits. Write a function that returns all the possible IP addresses that can be created by inserting three \".\"s in the string. An IP address is a sequence of four positive integers that are separated by \".\"s, where each individual integer is within the range 0-255, inclusive. An IP address isn't valid if any of the individual integers contains leading 0s. For example, \"192.168.0.1\" is a valid IP address, but \"192.168.00.1\" and \"192.168.0.01\" aren't, because they contain \"00\" and \"01\" respectively. Another example of a valid IP address is \"99.1.1.10\"; conversely, \"991.1.1.0\" isn't valid because \"991\" is greater than 255.",
		Difficulty:  "Medium",
		Topic:       "Strings",
		Signature:   "func validIPAddresses(str string) []string",
		TestCases: []TestCase{
			{Input: "str = \"1921680\"", Expected: "[\"1.9.2.1680\", \"1.9.21.680\", \"1.9.216.80\", \"1.92.1.680\", \"1.92.16.80\", \"1.92.168.0\", \"19.2.1.680\", \"19.2.16.80\", \"19.2.168.0\", \"19.21.6.80\", \"19.21.68.0\", \"19.216.8.0\", \"192.1.6.80\", \"192.1.68.0\", \"192.16.8.0\"]"},
		},
		Solution: `func validIPAddresses(str string) []string {
	ipAddressesFound := []string{}
	for i := 1; i < min(len(str)-2, 4); i++ {
		currentIPAddressParts := []string{"", "", "", ""}
		currentIPAddressParts[0] = str[:i]
		if !isValidPart(currentIPAddressParts[0]) {
			continue
		}
		for j := i + 1; j < min(len(str)-1, i+4); j++ {
			currentIPAddressParts[1] = str[i:j]
			if !isValidPart(currentIPAddressParts[1]) {
				continue
			}
			for k := j + 1; k < len(str); k++ {
				currentIPAddressParts[2] = str[j:k]
				currentIPAddressParts[3] = str[k:]
				if isValidPart(currentIPAddressParts[2]) && isValidPart(currentIPAddressParts[3]) {
					ipAddressesFound = append(ipAddressesFound, currentIPAddressParts[0]+"."+currentIPAddressParts[1]+"."+currentIPAddressParts[2]+"."+currentIPAddressParts[3])
				}
			}
		}
	}
	return ipAddressesFound
}

func isValidPart(str string) bool {
	if len(str) > 3 {
		return false
	}
	if len(str) > 1 && str[0] == '0' {
		return false
	}
	num := 0
	for _, char := range str {
		digit := int(char - '0')
		num = num*10 + digit
	}
	return num <= 255
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}`,
		PythonSolution: `def validIPAddresses(string: str) -> List[str]:
    ip_addresses_found = []
    for i in range(1, min(len(string) - 2, 4)):
        current_ip_address_parts = ['', '', '', '']
        current_ip_address_parts[0] = string[:i]
        if not is_valid_part(current_ip_address_parts[0]):
            continue
        for j in range(i + 1, min(len(string) - 1, i + 4)):
            current_ip_address_parts[1] = string[i:j]
            if not is_valid_part(current_ip_address_parts[1]):
                continue
            for k in range(j + 1, len(string)):
                current_ip_address_parts[2] = string[j:k]
                current_ip_address_parts[3] = string[k:]
                if is_valid_part(current_ip_address_parts[2]) and is_valid_part(current_ip_address_parts[3]):
                    ip_addresses_found.append('.'.join(current_ip_address_parts))
    return ip_addresses_found

def is_valid_part(string):
    if len(string) > 3:
        return False
    if len(string) > 1 and string[0] == '0':
        return False
    num = int(string)
    return num <= 255`,
		Explanation: `Generating valid IP addresses from a string requires inserting three dots in valid positions such that each resulting part is a valid IP address component (0-255, no leading zeros). This is a combinatorial problem with constraints.

The algorithm tries all possible positions for the three dots by using nested loops. For each combination of dot positions, it extracts the four parts and validates each: the part must be a number between 0 and 255, and if it has more than one digit, it cannot start with '0'. If all four parts are valid, the IP address is added to the results.

The constraint that the string length is at most 12 limits the search space, making brute force feasible. We need to ensure dot positions create exactly four parts and that each part is non-empty.

Edge cases handled include: string too short (less than 4 digits, no valid IPs), string with leading zeros (parts starting with 0 invalid unless single digit), numbers greater than 255 (invalid), and strings with exactly 12 digits (maximum valid length).

An alternative recursive approach would be more elegant but less efficient for small inputs. The nested loop approach is straightforward for the constrained input size. The time complexity is O(1) since we're limited to string length 12, giving at most C(11,3) = 165 combinations to check. The space complexity is O(1) for a constant number of results.

Think of this as placing three dividers in a number: you try all possible positions for the dividers, check if each resulting segment is a valid number between 0-255 without leading zeros, and collect all valid combinations.`,
	},
	{
		ID:          157,
		Title:       "Reverse Words In String",
		Description: "Write a function that takes in a string of words separated by one or more whitespaces and returns a string that has these words in reverse order. For example, given the string \"tim is great\", your function should return \"great is tim\". For this problem, a word can contain special characters, punctuation, and numbers. The words in the string will be separated by one or more whitespaces, and the reversed string must contain the same whitespaces as the original string. For example, given the string \"whitespaces    4\" you would be expected to return \"4    whitespaces\". Note that you're not allowed to use any built-in split or reverse methods or functions.",
		Difficulty:  "Medium",
		Topic:       "Strings",
		Signature:   "func reverseWordsInString(str string) string",
		TestCases: []TestCase{
			{Input: "str = \"AlgoExpert is the best!\"", Expected: "\"best! the is AlgoExpert\""},
			{Input: "str = \"whitespaces    4\"", Expected: "\"4    whitespaces\""},
		},
		Solution: `func reverseWordsInString(str string) string {
	characters := []rune(str)
	reverseListRange(characters, 0, len(characters)-1)
	startOfWord := 0
	for idx := 0; idx < len(characters); idx++ {
		if characters[idx] == ' ' {
			reverseListRange(characters, startOfWord, idx-1)
			startOfWord = idx + 1
		}
	}
	reverseListRange(characters, startOfWord, len(characters)-1)
	return string(characters)
}

func reverseListRange(list []rune, start int, end int) {
	for start < end {
		list[start], list[end] = list[end], list[start]
		start++
		end--
	}
}`,
		PythonSolution: `def reverseWordsInString(string: str) -> str:
    characters = list(string)
    reverse_list_range(characters, 0, len(characters) - 1)
    start_of_word = 0
    for idx in range(len(characters)):
        if characters[idx] == ' ':
            reverse_list_range(characters, start_of_word, idx - 1)
            start_of_word = idx + 1
    reverse_list_range(characters, start_of_word, len(characters) - 1)
    return ''.join(characters)

def reverse_list_range(list_chars, start, end):
    while start < end:
        list_chars[start], list_chars[end] = list_chars[end], list_chars[start]
        start += 1
        end -= 1`,
		Explanation: `Reversing words in a string while preserving whitespace requires a two-step approach: first reverse the entire string, then reverse each individual word. This elegantly handles the word order reversal.

The algorithm first reverses the entire string character by character. This puts the words in reverse order, but each word is also reversed. Then, it identifies word boundaries (transitions between non-whitespace and whitespace) and reverses each word individually, restoring the correct character order within each word while maintaining the reversed word order.

This approach preserves whitespace naturally because whitespace characters remain in their positions during both reversal steps. The word boundaries are identified by checking where non-whitespace characters transition to whitespace or vice versa.

Edge cases handled include: multiple spaces between words (preserved correctly), leading/trailing spaces (preserved), single word (reversed correctly), and strings with only spaces (returned as-is).

An alternative approach would extract words and spaces separately, then reconstruct, but that's more complex. The two-reversal approach is elegant and efficient. The time complexity is O(n) where n is the string length, as we traverse the string a constant number of times. The space complexity is O(n) for the character array (or O(1) if modifying in-place is allowed).

Think of this as flipping a sentence: you first flip the entire sentence backwards, which reverses both word order and letters, then you flip each word back to restore the letters while keeping words in reverse order.`,
	},
	{
		ID:          158,
		Title:       "Minimum Characters For Words",
		Description: "Write a function that takes in an array of words and returns the minimum number of characters needed to form all of the words. The input array of words will never be empty.",
		Difficulty:  "Medium",
		Topic:       "Hash Tables",
		Signature:   "func minimumCharactersForWords(words []string) []string",
		TestCases: []TestCase{
			{Input: "words = [\"this\", \"that\", \"did\", \"deed\", \"them!\", \"a\"]", Expected: "[\"t\", \"t\", \"h\", \"i\", \"s\", \"a\", \"d\", \"d\", \"e\", \"e\", \"m\", \"!\"]"},
		},
		Solution: `func minimumCharactersForWords(words []string) []string {
	maximumCharacterFrequencies := make(map[rune]int)
	for _, word := range words {
		characterFrequencies := countCharacterFrequencies(word)
		updateMaximumFrequencies(characterFrequencies, maximumCharacterFrequencies)
	}
	return makeArrayFromCharacterFrequencies(maximumCharacterFrequencies)
}

func countCharacterFrequencies(str string) map[rune]int {
	frequencies := make(map[rune]int)
	for _, char := range str {
		frequencies[char]++
	}
	return frequencies
}

func updateMaximumFrequencies(frequencies map[rune]int, maximumFrequencies map[rune]int) {
	for char, frequency := range frequencies {
		if frequency > maximumFrequencies[char] {
			maximumFrequencies[char] = frequency
		}
	}
}

func makeArrayFromCharacterFrequencies(characterFrequencies map[rune]int) []string {
	characters := []string{}
	for char, frequency := range characterFrequencies {
		for frequency > 0 {
			characters = append(characters, string(char))
			frequency--
		}
	}
	return characters
}`,
		PythonSolution: `def minimumCharactersForWords(words: List[str]) -> List[str]:
    maximum_character_frequencies = {}
    for word in words:
        character_frequencies = count_character_frequencies(word)
        update_maximum_frequencies(character_frequencies, maximum_character_frequencies)
    return make_array_from_character_frequencies(maximum_character_frequencies)

def count_character_frequencies(string):
    frequencies = {}
    for char in string:
        frequencies[char] = frequencies.get(char, 0) + 1
    return frequencies

def update_maximum_frequencies(frequencies, maximum_frequencies):
    for char, frequency in frequencies.items():
        if frequency > maximum_frequencies.get(char, 0):
            maximum_frequencies[char] = frequency

def make_array_from_character_frequencies(character_frequencies):
    characters = []
    for char, frequency in character_frequencies.items():
        for _ in range(frequency):
            characters.append(char)
    return characters`,
		Explanation: `Finding the minimum characters needed to form all words requires determining the maximum frequency of each character across all words. This ensures we have enough of each character to form any word.

The algorithm processes each word, counting the frequency of each character. For each character, it tracks the maximum frequency seen across all words. After processing all words, it builds a result array containing each character repeated according to its maximum frequency.

This approach works because to form all words, we need at least as many of each character as the maximum required by any single word. If word A needs 3 'a's and word B needs 2 'a's, we need 3 'a's total to form both words.

Edge cases handled include: all words requiring same characters (frequencies are the maximums), words with no common characters (result contains all unique characters), words with varying character frequencies (maximums tracked correctly), and single word (result equals that word's characters).

An alternative approach would find the union of all character sets, but that doesn't account for frequencies. The frequency-tracking approach correctly handles cases where multiple words need the same character. The time complexity is O(n*c) where n is total characters across all words and c is unique characters. The space complexity is O(c) for frequency tracking.

Think of this as preparing a letter set for word games: you count how many of each letter each word needs, then prepare enough letters to satisfy the word that needs the most of each letter type.`,
	},
	{
		ID:          159,
		Title:       "One Edit",
		Description: "You're given two strings stringOne and stringTwo. Write a function that determines if these two strings can be made equal using only one edit. There are 3 types of edits: Replace: One character in one string is swapped for a different character. Add: One character is added at any index in one string. Remove: One character is removed at any index in one string.",
		Difficulty:  "Medium",
		Topic:       "Strings",
		Signature:   "func oneEdit(stringOne string, stringTwo string) bool",
		TestCases: []TestCase{
			{Input: "stringOne = \"hello\", stringTwo = \"hallo\"", Expected: "true"},
			{Input: "stringOne = \"hello\", stringTwo = \"helo\"", Expected: "true"},
		},
		Solution: `func oneEdit(stringOne string, stringTwo string) bool {
	lenOne := len(stringOne)
	lenTwo := len(stringTwo)
	if abs(lenOne-lenTwo) > 1 {
		return false
	}
	madeEdit := false
	idxOne := 0
	idxTwo := 0
	for idxOne < lenOne && idxTwo < lenTwo {
		if stringOne[idxOne] != stringTwo[idxTwo] {
			if madeEdit {
				return false
			}
			madeEdit = true
			if lenOne > lenTwo {
				idxOne++
			} else if lenTwo > lenOne {
				idxTwo++
			} else {
				idxOne++
				idxTwo++
			}
		} else {
			idxOne++
			idxTwo++
		}
	}
	return true
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}`,
		PythonSolution: `def oneEdit(stringOne: str, stringTwo: str) -> bool:
    len_one = len(stringOne)
    len_two = len(stringTwo)
    if abs(len_one - len_two) > 1:
        return False
    made_edit = False
    idx_one = 0
    idx_two = 0
    while idx_one < len_one and idx_two < len_two:
        if stringOne[idx_one] != stringTwo[idx_two]:
            if made_edit:
                return False
            made_edit = True
            if len_one > len_two:
                idx_one += 1
            elif len_two > len_one:
                idx_two += 1
            else:
                idx_one += 1
                idx_two += 1
        else:
            idx_one += 1
            idx_two += 1
    return True`,
		Explanation: `Determining if two strings can be made equal with at most one edit requires checking if they differ in at most one position. The two-pointer technique efficiently handles insertions, deletions, and replacements.

The algorithm first checks if the length difference exceeds 1 (impossible with one edit). Then it uses two pointers to traverse both strings. When characters match, both pointers advance. When they don't match, we've used our one edit: if strings are same length, it's a replacement (both pointers advance); if one is longer, it's an insertion/deletion (only the longer string's pointer advances). After one mismatch, any further mismatch means more than one edit is needed.

This approach efficiently checks all three edit types: replacement (same length, one mismatch), insertion (one string longer by 1), and deletion (one string shorter by 1). The pointer movement naturally handles these cases.

Edge cases handled include: identical strings (zero edits, return true), strings differing by one character (one replacement), strings where one needs one character added (one insertion), strings where one needs one character removed (one deletion), and strings requiring more than one edit (return false).

An alternative approach would use dynamic programming (edit distance), but that's O(n*m) time. The two-pointer approach achieves O(n) time where n is the length of the longer string, as we traverse at most once. The space complexity is O(1).

Think of this as proofreading: you compare two texts character by character, and you're allowed one mistake. If you find a difference, you note it and continue. If you find a second difference, the texts need more than one edit to match.`,
	},
	{
		ID:          160,
		Title:       "Suffix Trie Construction",
		Description: "Write a SuffixTrie class for a Suffix-Trie-like data structure. The class should have a root property set to the root node of the trie and should support: Creating the trie from a string; this will be done by calling the populateSuffixTrieFrom method upon class instantiation, which should populate the root of the class. Searching for strings in the trie. Note that every string added to the trie should end with the special endSymbol character: \"*\".",
		Difficulty:  "Medium",
		Topic:       "Tries",
		Signature:   "type SuffixTrie struct { root map[rune]interface{}; endSymbol rune }\nfunc NewSuffixTrie(str string) *SuffixTrie\nfunc (trie *SuffixTrie) Contains(str string) bool",
		TestCases: []TestCase{
			{Input: "str = \"babc\"", Expected: "Trie containing all suffixes"},
		},
		Solution: `type SuffixTrie struct {
	root      map[rune]interface{}
	endSymbol rune
}

func NewSuffixTrie(str string) *SuffixTrie {
	trie := &SuffixTrie{
		root:      make(map[rune]interface{}),
		endSymbol: '*',
	}
	trie.populateSuffixTrieFrom(str)
	return trie
}

func (trie *SuffixTrie) populateSuffixTrieFrom(str string) {
	for i := 0; i < len(str); i++ {
		trie.insertSubstringStartingAt(i, str)
	}
}

func (trie *SuffixTrie) insertSubstringStartingAt(i int, str string) {
	node := trie.root
	for j := i; j < len(str); j++ {
		char := rune(str[j])
		if _, found := node[char]; !found {
			node[char] = make(map[rune]interface{})
		}
		node = node[char].(map[rune]interface{})
	}
	node[trie.endSymbol] = true
}

func (trie *SuffixTrie) Contains(str string) bool {
	node := trie.root
	for _, char := range str {
		if _, found := node[char]; !found {
			return false
		}
		node = node[char].(map[rune]interface{})
	}
	_, found := node[trie.endSymbol]
	return found
}`,
		PythonSolution: `class SuffixTrie:
    def __init__(self, string):
        self.root = {}
        self.endSymbol = "*"
        self.populateSuffixTrieFrom(string)
    
    def populateSuffixTrieFrom(self, string):
        for i in range(len(string)):
            self.insert_substring_starting_at(i, string)
    
    def insert_substring_starting_at(self, i, string):
        node = self.root
        for j in range(i, len(string)):
            char = string[j]
            if char not in node:
                node[char] = {}
            node = node[char]
        node[self.endSymbol] = True
    
    def contains(self, string):
        node = self.root
        for char in string:
            if char not in node:
                return False
            node = node[char]
        return self.endSymbol in node`,
		Explanation: `A suffix trie stores all suffixes of a string, enabling efficient substring searches. Each path from the root to a node marked with an end symbol represents a suffix (or substring) of the original string.

The algorithm builds the trie by inserting all suffixes of the input string. For a string of length n, there are n suffixes (starting at each position). Each suffix is inserted character by character, creating nodes as needed. The end symbol "*" marks where each suffix ends, allowing us to distinguish between suffixes and their prefixes.

To search for a substring, we traverse the trie following the characters of the search string. If we can follow the entire path and the end node contains the end symbol, the substring exists. The trie structure allows O(m) search time where m is the substring length, independent of the original string length.

Edge cases handled include: searching for empty string (check if end symbol at root), searching for string longer than original (cannot exist), searching for string that's a prefix of a suffix (found correctly), and building from empty string (trie with just root and end symbol).

An alternative approach would use a suffix array or suffix tree, but suffix tries are simpler to implement. The build time is O(n²) where n is string length, as we insert n suffixes of average length n/2. The space complexity is O(n²) for storing all suffix paths. Search time is O(m) where m is search string length.

Think of this as creating an index of all possible endings of a word: you store every way the word could end, and to check if a substring exists, you follow the path through this index.`,
	},
}
