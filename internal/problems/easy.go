package problems

// EasyProblems contains all easy difficulty problems (1-20)
var EasyProblems = []Problem{
	{
		ID:          1,
		Title:       "Two Sum",
		Description: "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice.",
		Difficulty:  "Easy",
		Topic:       "Hash Tables",
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
		PythonSolution: `def twoSum(nums: List[int], target: int) -> List[int]:
    m = {}
    for i, num in enumerate(nums):
        if target - num in m:
            return [m[target - num], i]
        m[num] = i
    return []`,
		Explanation: `This problem can be solved efficiently using a hash map (dictionary) to achieve O(n) time complexity. The key insight is that for each number in the array, we need to find its complement (target - num) in the remaining array.

The algorithm works by iterating through the array once. For each number, we first check if its complement already exists in our hash map. If it does, we've found our pair and can immediately return both indices. If not, we add the current number and its index to the hash map for future lookups.

This approach eliminates the need for nested loops, which would result in O(n²) time complexity. Instead, hash map lookups are O(1) on average, making the overall solution O(n). The space complexity is O(n) in the worst case when we need to store all elements before finding a match.

Edge cases handled include: arrays with duplicate numbers (the hash map stores the most recent index), negative numbers (complement calculation works correctly), and arrays with exactly two elements.

An alternative O(n²) approach would be to check all pairs, but the hash map solution is optimal for time complexity. The trade-off is using O(n) extra space, which is acceptable for the significant time improvement.

Intuitively, think of this as building a "wishlist" as we go: for each number we see, we remember what number we're looking for to complete the pair.`,
	},
	{
		ID:          2,
		Title:       "Reverse String",
		Description: "Write a function that reverses a string. The input string is given as an array of characters. You must do this by modifying the input array in-place.",
		Difficulty:  "Easy",
		Topic:       "Two Pointers",
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
		PythonSolution: `def reverseString(s: List[str]) -> None:
    i, j = 0, len(s) - 1
    while i < j:
        s[i], s[j] = s[j], s[i]
        i += 1
        j -= 1`,
		Explanation: `The two-pointer technique is perfect for this in-place reversal problem. We use two pointers starting from opposite ends of the array and swap characters as they converge toward the center.

The algorithm initializes two pointers: one at the start (i) and one at the end (j) of the array. In each iteration, we swap the characters at these positions, then move both pointers inward (i increments, j decrements). The process continues until the pointers meet or cross, at which point all characters have been swapped.

This approach is optimal because it requires only a single pass through half the array, giving us O(n) time complexity. The space complexity is O(1) since we only use a constant amount of extra space for the swap operation, modifying the input array in-place.

Edge cases handled include: single character strings (no swap needed), empty strings (loop doesn't execute), and even-length strings (pointers meet exactly at the middle).

An alternative approach would be to create a new array and copy characters in reverse order, but that would require O(n) extra space. The two-pointer approach maintains the O(1) space requirement while being equally efficient in time.

Think of this as two people walking toward each other from opposite ends of a hallway, exchanging items as they pass.`,
	},
	{
		ID:          3,
		Title:       "FizzBuzz",
		Description: "Given an integer n, return a string array answer where: answer[i] == \"FizzBuzz\" if i is divisible by 3 and 5, answer[i] == \"Fizz\" if i is divisible by 3, answer[i] == \"Buzz\" if i is divisible by 5, answer[i] == i if none of the above conditions are true.",
		Difficulty:  "Easy",
		Topic:       "Arrays",
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
		PythonSolution: `def fizzBuzz(n: int) -> List[str]:
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result`,
		Explanation: `This classic problem requires checking divisibility conditions in a specific order to handle the "FizzBuzz" case correctly. The key is to check for divisibility by 15 first, since numbers divisible by both 3 and 5 should output "FizzBuzz".

The algorithm iterates from 1 to n, checking each number against our divisibility rules. We first check if the number is divisible by 15 (using the modulo operator with 15), which handles the case where both conditions are true. If not, we check divisibility by 3, then by 5. If none of these conditions are met, we convert the number to its string representation.

The order of checks is crucial: checking 15 before 3 and 5 ensures we don't output "Fizz" or "Buzz" for numbers that should output "FizzBuzz". This is an example of checking more specific conditions before general ones.

Edge cases handled include: n=1 (outputs "1"), n=15 (outputs "FizzBuzz" at position 15), and very large values of n (the algorithm scales linearly).

An alternative approach would be to check divisibility by 3 and 5 separately and concatenate, but checking 15 first is more efficient and clearer. The modulo operation is O(1), making the overall time complexity O(n).

Think of this as a filtering system: each number passes through increasingly specific filters (15, then 3, then 5) until it matches a condition or falls through to the default case.`,
	},
	{
		ID:          4,
		Title:       "Palindrome Check",
		Description: "Given a string s, return true if it is a palindrome, or false otherwise. A palindrome is a string that reads the same forward and backward.",
		Difficulty:  "Easy",
		Topic:       "Two Pointers",
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
		PythonSolution: `def isPalindrome(s: str) -> bool:
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    i, j = 0, len(cleaned) - 1
    while i < j:
        if cleaned[i] != cleaned[j]:
            return False
        i += 1
        j -= 1
    return True`,
		Explanation: `This problem requires preprocessing the string to handle case-insensitivity and non-alphanumeric characters before checking if it's a palindrome. The two-pointer technique is ideal for the comparison phase.

The algorithm first normalizes the input string by converting it to lowercase and filtering out non-alphanumeric characters, creating a cleaned string. Then, we use two pointers starting from both ends of the cleaned string, comparing characters as they move toward the center. If any pair of characters doesn't match, we immediately return false.

The preprocessing step is necessary because palindromes are defined only by alphanumeric characters in a case-insensitive manner. For example, "A man, a plan, a canal: Panama" should be considered a palindrome after removing punctuation and normalizing case.

Edge cases handled include: empty strings (considered palindromes), strings with only non-alphanumeric characters (empty cleaned string is a palindrome), single characters, and strings with mixed case and punctuation.

An alternative approach would be to clean and reverse the string, then compare, but that requires O(n) extra space. The two-pointer approach uses O(n) space for the cleaned string but O(1) additional space for the comparison.

Intuitively, imagine reading the string from both ends simultaneously, skipping over any non-letter characters, and checking if you're reading the same sequence forward and backward.`,
	},
	{
		ID:          5,
		Title:       "Fibonacci",
		Description: "The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1. Given n, calculate F(n).",
		Difficulty:  "Easy",
		Topic:       "Dynamic Programming",
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
		PythonSolution: `def fib(n: int) -> int:
    if n < 2:
        return n
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b`,
		Explanation: `The Fibonacci sequence can be computed efficiently using an iterative approach with constant space, avoiding the exponential time complexity of naive recursion. The key insight is that we only need to track the last two values to compute the next one.

The algorithm uses two variables (a and b) to represent F(n-2) and F(n-1) respectively. Starting from the base cases (F(0)=0, F(1)=1), we iterate from 2 to n, updating these variables: the new value is the sum of the previous two, and we shift the variables forward. This iterative approach builds up the sequence one step at a time.

This method is optimal because it computes each Fibonacci number exactly once, resulting in O(n) time complexity. The space complexity is O(1) since we only maintain two variables regardless of the input size, compared to O(n) space for a recursive approach with memoization or O(n) for an array-based solution.

Edge cases handled include: n=0 (returns 0), n=1 (returns 1), and n=2 (returns 1). The algorithm correctly handles all base cases before entering the iterative loop.

A recursive solution would be more intuitive but would have O(2^n) time complexity without memoization, or O(n) time and O(n) space with memoization. The iterative approach achieves the same time complexity with better space efficiency.

Think of this as climbing stairs: to reach step n, you can come from step n-1 (one step) or step n-2 (two steps), so the number of ways equals the sum of ways to reach those previous steps.`,
	},
	{
		ID:          6,
		Title:       "Contains Duplicate",
		Description: "Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.",
		Difficulty:  "Easy",
		Topic:       "Hash Tables",
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
		PythonSolution: `def containsDuplicate(nums: List[int]) -> bool:
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False`,
		Explanation: `This problem can be solved efficiently using a hash set to track numbers we've already seen. The key insight is that we can detect duplicates in a single pass through the array by maintaining a set of previously encountered values.

The algorithm iterates through the array once, checking each number against our hash set. If a number is already in the set, we've found a duplicate and can immediately return true. Otherwise, we add the number to the set and continue. If we complete the iteration without finding any duplicates, we return false.

This approach leverages the O(1) average-case lookup and insertion time of hash sets, resulting in an overall O(n) time complexity. The space complexity is O(n) in the worst case when all elements are unique and we need to store them all.

Edge cases handled include: arrays with a single element (no duplicates possible), empty arrays (no duplicates), arrays with multiple duplicates (returns true on first duplicate found), and arrays where all elements are the same.

An alternative O(n log n) approach would be to sort the array first and then check adjacent elements, but this modifies the input and is slower. Another O(n²) approach would use nested loops, but the hash set solution is optimal for time complexity.

Think of this as checking guests at a party: as each person arrives, you check if you've seen them before. If yes, you've found a duplicate guest.`,
	},
	{
		ID:          7,
		Title:       "Valid Anagram",
		Topic:       "Hash Tables",
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
		PythonSolution: `def isAnagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    counts = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    for c in t:
        counts[c] = counts.get(c, 0) - 1
        if counts[c] < 0:
            return False
    return True`,
		Explanation: `An anagram is a word formed by rearranging the letters of another word. This problem can be solved efficiently by comparing character frequencies between the two strings using a hash map.

The algorithm first checks if both strings have the same length - if not, they cannot be anagrams. Then, it counts character frequencies in the first string by incrementing counts in a hash map. Next, it iterates through the second string, decrementing the count for each character. If any count becomes negative during this process, it means the second string has more occurrences of that character than the first, so they cannot be anagrams.

This approach is efficient because it requires only two passes through the strings, each taking O(n) time. The space complexity is O(1) because we're limited to 26 lowercase English letters (or a fixed character set), so the hash map size is bounded by a constant.

Edge cases handled include: strings of different lengths (early return false), identical strings (anagrams of themselves), strings with repeated characters, and empty strings (anagrams of each other).

An alternative approach would be to sort both strings and compare them, but that would take O(n log n) time. Another approach would use two separate frequency maps and compare them, but the single-map approach with decrementing is more space-efficient.

Intuitively, think of this as having a bag of letter tiles: you remove tiles for the first word, then try to remove the same tiles for the second word. If you can't find a tile or have tiles left over, they're not anagrams.`,
	},
	{
		ID:          8,
		Title:       "Valid Parentheses",
		Topic:       "Stack",
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
		PythonSolution: `def isValid(s: str) -> bool:
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    for c in s:
        if c in pairs:
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
        else:
            stack.append(c)
    return len(stack) == 0`,
		Explanation: `This problem is a classic application of the stack data structure. The key insight is that valid parentheses must have matching pairs, and the most recently opened bracket must be the first one closed (LIFO - Last In, First Out).

The algorithm uses a stack to track opening brackets. As we iterate through the string, we push opening brackets ('(', '{', '[') onto the stack. When we encounter a closing bracket, we check if the stack is empty (meaning there's no matching opening bracket) or if the top of the stack doesn't match the expected opening bracket. If either condition fails, the string is invalid. If the stack is empty after processing all characters, the string is valid.

The stack naturally handles nested brackets correctly because it maintains the order of opening brackets, ensuring that inner brackets are closed before outer ones. This is exactly what we need for valid parentheses.

Edge cases handled include: empty strings (valid - empty stack), single opening bracket (invalid - non-empty stack), single closing bracket (invalid - empty stack when closing encountered), and mismatched bracket types.

An alternative approach would use a counter for each bracket type, but that doesn't work for nested structures like "([)]" which should be invalid. The stack approach correctly handles nesting and ordering requirements.

Think of this as a stack of plates: you can only remove the top plate (most recent opening bracket), and you must close brackets in the reverse order they were opened.`,
	},
	{
		ID:          9,
		Title:       "Merge Two Sorted Lists",
		Topic:       "Linked Lists",
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
		PythonSolution: `def mergeTwoLists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    curr = dummy
    while l1 and l2:
        if l1.val < l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    curr.next = l1 if l1 else l2
    return dummy.next`,
		Explanation: `Merging two sorted linked lists efficiently requires maintaining pointers to both lists and building the result list incrementally. The key insight is to use a dummy node to simplify edge case handling and avoid special checks for the first node.

The algorithm creates a dummy node to serve as the head of the merged list, and a current pointer to track where we're building the result. We then iterate through both lists simultaneously, comparing the values at the current positions. We attach the smaller node to our result list and advance the corresponding pointer. When one list is exhausted, we simply attach the remainder of the other list to our result.

The dummy node technique eliminates the need to check if we're creating the first node, making the code cleaner and more maintainable. After building the merged list, we return dummy.Next to skip the dummy node and return the actual merged list.

Edge cases handled include: one or both lists being empty, lists of different lengths, and lists with duplicate values. The algorithm correctly handles all these cases by checking list exhaustion and appending remaining nodes.

An alternative approach would be to create a new list by copying nodes, but that would require O(m+n) extra space. The in-place merging approach uses O(1) extra space (just the dummy node and pointers).

Think of this as merging two sorted decks of cards: you always take the smaller top card from either deck and add it to your merged deck, continuing until one deck is empty, then adding all remaining cards from the other deck.`,
	},
	{
		ID:          10,
		Title:       "Best Time to Buy and Sell Stock",
		Topic:       "Arrays",
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
		PythonSolution: `def maxProfit(prices: List[int]) -> int:
    min_price = prices[0]
    max_profit = 0
    for price in prices[1:]:
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price
    return max_profit`,
		Explanation: `This problem requires finding the maximum profit from buying and selling a stock once. The key insight is that we need to buy at the lowest price before selling at the highest price, but we can only sell after buying.

The algorithm maintains two variables: the minimum price seen so far and the maximum profit achievable. As we iterate through the prices array, we update the minimum price whenever we encounter a lower price. For each day, we calculate the profit we would make if we sold on that day (current price minus minimum price seen so far) and update the maximum profit if this is better than our current best.

This approach is optimal because we only need a single pass through the array. We're essentially tracking the best buying opportunity (minimum price) as we go, and calculating the best selling opportunity (maximum profit) at each step.

Edge cases handled include: prices that only decrease (maximum profit is 0, meaning we shouldn't buy), prices that only increase (buy on first day, sell on last day), arrays with a single price (no transaction possible, profit is 0), and prices with multiple local minima and maxima.

An alternative O(n²) approach would check all pairs of buy and sell days, but the single-pass approach is optimal. Another approach would use dynamic programming, but it's unnecessary for this simpler problem.

Intuitively, think of this as tracking the lowest valley (best buy price) and calculating the highest peak after that valley (best sell price). We update our best buy price whenever we see a lower price, and update our best profit whenever selling at the current price would yield more profit.`,
	},
	{
		ID:          11,
		Title:       "Binary Search",
		Topic:       "Binary Search",
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
		PythonSolution: `def search(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1`,
		Explanation: `Binary search is the optimal algorithm for searching in a sorted array. The key insight is that because the array is sorted, we can eliminate half of the remaining search space with each comparison.

The algorithm maintains two pointers (left and right) that define the current search range. In each iteration, we calculate the middle index and compare the element at that position with the target. If they match, we've found our answer. If the middle element is less than the target, we know the target must be in the right half, so we update the left pointer. Otherwise, we update the right pointer to search the left half.

This divide-and-conquer approach reduces the search space by half in each iteration, resulting in O(log n) time complexity. The space complexity is O(1) since we only use a constant amount of extra space for the pointers.

Edge cases handled include: target not in array (returns -1), target at the first or last position, single-element arrays, and arrays where the target appears multiple times (returns any valid index).

An alternative O(n) approach would be linear search, but binary search is optimal for sorted arrays. The key is using the sorted property to eliminate large portions of the search space efficiently.

Think of this as the "guess the number" game: if you guess the middle of the range and are told "higher" or "lower", you can eliminate half the possibilities each time, leading to logarithmic time complexity.`,
	},
	{
		ID:          12,
		Title:       "First Bad Version",
		Topic:       "Binary Search",
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
		PythonSolution: `def firstBadVersion(n: int) -> int:
    left, right = 1, n
    while left < right:
        mid = left + (right - left) // 2
        if isBadVersion(mid):
            right = mid
        else:
            left = mid + 1
    return left`,
		Explanation: `This is a variant of binary search that finds the first occurrence of a "bad" version. The key difference from standard binary search is that we're looking for the leftmost bad version, not just any bad version.

The algorithm uses binary search but with a modified condition. When we find a bad version at the middle position, we know that all versions after it are also bad, but there might be bad versions before it. So instead of returning immediately, we search the left half (including the current middle) to find an earlier bad version. When we find a good version, we know all versions before it are also good, so we search the right half.

The loop condition uses left < right instead of left <= right because we want to find the exact boundary. When the loop terminates, left will point to the first bad version. This is a common pattern in binary search problems where we're looking for a boundary or first occurrence.

Edge cases handled include: first version is bad (returns 1), last version is bad (returns n), all versions are bad (returns 1), and all versions are good (shouldn't happen per problem constraints, but algorithm handles it).

An alternative O(n) approach would be linear search from the beginning, but binary search is optimal. The key insight is that the "bad" property creates a sorted boundary: all versions before the first bad version are good, and all versions from the first bad version onward are bad.

Think of this as finding the first broken item in a production line: you can test the middle item, and if it's broken, you know everything after is broken, but you need to check earlier items to find where the breakage started.`,
	},
	{
		ID:          13,
		Title:       "Ransom Note",
		Topic:       "Hash Tables",
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
		PythonSolution: `def canConstruct(ransomNote: str, magazine: str) -> bool:
    counts = {}
    for c in magazine:
        counts[c] = counts.get(c, 0) + 1
    for c in ransomNote:
        if counts.get(c, 0) == 0:
            return False
        counts[c] -= 1
    return True`,
		Explanation: `This problem is essentially checking if we have enough characters in the magazine to form the ransom note. The key insight is that we need to track the frequency of each character available in the magazine and ensure we have enough of each character needed for the ransom note.

The algorithm first counts all characters in the magazine, storing their frequencies in a hash map. Then, it iterates through each character in the ransom note, checking if that character is available in our count. If a character is not available (count is 0), we immediately return false. Otherwise, we decrement the count to "consume" that character.

This approach efficiently handles character frequency requirements. We can use characters from the magazine multiple times if they appear multiple times, but we must have at least as many occurrences of each character in the magazine as we need in the ransom note.

Edge cases handled include: empty ransom note (can always be formed), ransom note longer than magazine (cannot be formed), magazine with extra characters (still valid), and ransom note requiring characters not in magazine.

An alternative approach would be to sort both strings and compare, but that would take O(m log m + n log n) time. The hash map approach is more efficient and handles the frequency requirement naturally.

Think of this as having a box of letter tiles (magazine) and trying to spell out a word (ransom note). You need to check if you have enough of each required letter tile to form the word.`,
	},
	{
		ID:          14,
		Title:       "Climbing Stairs",
		Topic:       "Dynamic Programming",
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
		PythonSolution: `def climbStairs(n: int) -> int:
    if n <= 2:
        return n
    a, b = 1, 2
    for i in range(3, n + 1):
        a, b = b, a + b
    return b`,
		Explanation: `This problem is a classic example of the Fibonacci sequence disguised as a climbing stairs problem. The key insight is recognizing that the number of ways to reach step n equals the sum of ways to reach step (n-1) and step (n-2), because you can only climb 1 or 2 steps at a time.

The algorithm uses an iterative approach with two variables to track the previous two Fibonacci numbers. Starting from the base cases (1 step = 1 way, 2 steps = 2 ways), we build up the solution by computing each step as the sum of the previous two steps. This avoids the exponential time complexity of naive recursion while maintaining O(1) space complexity.

The recurrence relation works because to reach step n, you must have come from either step (n-1) by taking 1 step, or from step (n-2) by taking 2 steps. Since these are the only ways to reach step n, the total ways is the sum of ways to reach those previous steps.

Edge cases handled include: n=1 (1 way: take 1 step), n=2 (2 ways: two 1-steps or one 2-step), and n=0 (typically 1 way: already at the top, though this case may not be in the problem constraints).

An alternative recursive approach would have O(2^n) time complexity without memoization, or O(n) time and O(n) space with memoization. The iterative approach achieves O(n) time with O(1) space, making it optimal.

Think of this as building up possibilities: each step's number of ways depends on the previous two steps, creating the Fibonacci sequence where each number is the sum of the two preceding ones.`,
	},
	{
		ID:          15,
		Title:       "Longest Common Prefix",
		Topic:       "Arrays",
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
		PythonSolution: `def longestCommonPrefix(strs: List[str]) -> str:
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix`,
		Explanation: `Finding the longest common prefix requires comparing characters across all strings at each position. The key insight is that the common prefix cannot be longer than the shortest string, and we can iteratively refine our prefix candidate.

The algorithm starts with the first string as the initial prefix candidate. Then, for each subsequent string, it checks if the string starts with the current prefix. If not, it shortens the prefix by removing the last character and checks again, repeating until a match is found or the prefix becomes empty. This process continues for all strings, ensuring the final prefix is common to all.

This approach is efficient because we only compare characters that could potentially be in the common prefix. Once we've shortened the prefix to match all strings seen so far, we know it will work for remaining strings (or we'll shorten it further if needed).

Edge cases handled include: empty array (returns empty string), array with single string (returns that string), strings with no common prefix (returns empty string), and strings where one is a prefix of another (returns the shorter string).

An alternative approach would be to compare all strings character by character at each position, but the prefix-shortening approach is more intuitive and handles edge cases naturally. Both approaches have similar time complexity O(S) where S is the sum of all characters.

Think of this as finding the longest word that all strings start with: you start with a candidate and keep shortening it until it fits all strings, like trying different keys until one fits all locks.`,
	},
	{
		ID:          16,
		Title:       "Single Number",
		Topic:       "Hash Tables",
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
		PythonSolution: `def singleNumber(nums: List[int]) -> int:
    result = 0
    for num in nums:
        result ^= num
    return result`,
		Explanation: `This problem can be solved elegantly using the XOR (exclusive OR) bitwise operation. The key properties of XOR are: any number XORed with itself equals 0, any number XORed with 0 equals itself, and XOR is commutative and associative.

The algorithm initializes a result variable to 0, then XORs all numbers in the array together. Since every number except one appears twice, all pairs will cancel out (a XOR a = 0), leaving only the single number that appears once. This single number XORed with 0 equals itself, giving us our answer.

This approach is optimal because it requires only a single pass through the array with O(1) extra space. The XOR operation is very fast and doesn't require any additional data structures like hash maps or sets.

Edge cases handled include: arrays with a single element (returns that element), arrays where the single number is 0 (XOR works correctly), arrays with negative numbers (XOR works on two's complement representation), and arrays with the single number at any position.

An alternative O(n) approach would use a hash map to count frequencies, but that requires O(n) space. Another approach would sort the array and check adjacent elements, but that requires O(n log n) time. The XOR approach is optimal for both time and space.

Think of XOR as a toggle: pairing numbers cancel each other out, leaving only the unpaired number. It's like having light switches where flipping the same switch twice returns it to the original state, and only one switch is flipped once.`,
	},
	{
		ID:          17,
		Title:       "Majority Element",
		Topic:       "Hash Tables",
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
		PythonSolution: `def majorityElement(nums: List[int]) -> int:
    candidate, count = 0, 0
    for num in nums:
        if count == 0:
            candidate = num
        if num == candidate:
            count += 1
        else:
            count -= 1
    return candidate`,
		Explanation: `The Boyer-Moore Voting Algorithm is a clever O(n) time, O(1) space solution for finding the majority element. The key insight is that a majority element will "survive" even if we pair it against all other elements and cancel them out.

The algorithm maintains a candidate element and a count. As we iterate through the array, if the count is 0, we set the current element as the new candidate. If the current element matches the candidate, we increment the count; otherwise, we decrement it. This process effectively pairs different elements and cancels them out. Since the majority element appears more than n/2 times, it will be the last candidate standing.

The algorithm works because when we cancel out a majority element with a non-majority element, we're removing two elements from consideration. Even if we cancel out all pairs, the majority element will still have at least one occurrence remaining because it appears more than half the time.

Edge cases handled include: arrays with a single element (that element is the majority), arrays where the majority element appears exactly n/2 + 1 times, and arrays where the majority element appears at the beginning, middle, or end.

An alternative O(n) approach would use a hash map to count frequencies, but that requires O(n) space. Another approach would sort and check the middle element, but that requires O(n log n) time. The Boyer-Moore algorithm is optimal for both time and space.

Think of this as a voting process: each vote for the candidate increases their lead, votes against decrease it. The majority candidate will always have a positive count at the end because they have more than half the votes.`,
	},
	{
		ID:          18,
		Title:       "Remove Duplicates from Sorted Array",
		Topic:       "Two Pointers",
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
		PythonSolution: `def removeDuplicates(nums: List[int]) -> int:
    if not nums:
        return 0
    i = 0
    for j in range(1, len(nums)):
        if nums[j] != nums[i]:
            i += 1
            nums[i] = nums[j]
    return i + 1`,
		Explanation: `This problem requires removing duplicates in-place while maintaining the relative order of unique elements. The two-pointer technique is perfect for this: one pointer tracks where to place the next unique element, and another scans through the array.

The algorithm uses two pointers: i marks the position of the last unique element placed, and j scans through the array. Since the array is sorted, duplicates will be adjacent. When j encounters a new unique element (different from the element at i), we increment i and copy that element to position i. This effectively compacts all unique elements to the beginning of the array.

The key insight is that because the array is sorted, we only need to compare each element with the last unique element we've placed, not with all previous unique elements. This makes the algorithm efficient and straightforward.

Edge cases handled include: empty arrays (returns 0), arrays with no duplicates (all elements remain), arrays with all duplicates (only one element remains), and arrays with a single element (that element remains).

An alternative approach would create a new array with unique elements, but that requires O(n) extra space. The two-pointer in-place approach uses O(1) extra space while maintaining the same O(n) time complexity.

Think of this as packing items into boxes: you keep one of each unique item at the front, and skip duplicates as you scan through your inventory.`,
	},
	{
		ID:          19,
		Title:       "Move Zeroes",
		Topic:       "Two Pointers",
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
		PythonSolution: `def moveZeroes(nums: List[int]) -> None:
    left = 0
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1`,
		Explanation: `Moving zeros to the end while maintaining the relative order of non-zero elements requires careful in-place manipulation. The two-pointer technique allows us to do this efficiently in a single pass.

The algorithm uses two pointers: left tracks the position where the next non-zero element should be placed, and right scans through the array. When right encounters a non-zero element, we swap it with the element at left (which might be zero or already a non-zero element), then increment left. This ensures all non-zero elements are moved to the front while maintaining their relative order.

The key insight is that left always points to the first position that might contain a zero (or is ready for the next non-zero element). By swapping non-zero elements to this position, we naturally collect all non-zeros at the front, leaving zeros at the end.

Edge cases handled include: arrays with no zeros (elements remain in place), arrays with all zeros (no swaps needed, just scanning), arrays with zeros only at the end (already in correct position), and arrays with zeros only at the beginning (all non-zeros move forward).

An alternative approach would be to count zeros and overwrite, but that requires two passes. Another approach would create a new array, but that requires O(n) extra space. The two-pointer swap approach is optimal for both time and space.

Think of this as organizing items on a shelf: you want all the important items (non-zeros) at the front, so you swap each important item you find with whatever is at the front position, gradually pushing zeros to the back.`,
	},
	{
		ID:          20,
		Title:       "Reverse Linked List",
		Topic:       "Linked Lists",
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
		PythonSolution: `def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev`,
		Explanation: `Reversing a linked list requires changing the direction of all the next pointers. The iterative approach does this efficiently by maintaining three pointers: one for the previous node, one for the current node, and one for the next node.

The algorithm starts with prev set to nil and curr set to the head. In each iteration, we save the next node (since we'll be changing curr.Next), then reverse the pointer by setting curr.Next to prev. We then advance both prev and curr to the next positions. When curr becomes nil, we've processed all nodes, and prev points to the new head (the old tail).

This approach is optimal because it requires only a single pass through the list with O(1) extra space. Each node's pointer is reversed exactly once, and we maintain the necessary references to continue the process.

Edge cases handled include: empty lists (returns nil), single-node lists (returns that node with Next=nil), and two-node lists (pointers are correctly reversed).

An alternative recursive approach would also work and might be more intuitive, but it uses O(n) space for the recursion stack. The iterative approach achieves the same result with O(1) space, making it more space-efficient.

Think of this as turning a chain around: you hold onto the previous link (prev), work on the current link (curr), and always know where the next link (next) is so you can continue the process.`,
	},
	{
		ID:          21,
		Title:       "Valid Anagram",
		Description: "Given two strings s and t, return true if t is an anagram of s, and false otherwise. An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.",
		Difficulty:  "Easy",
		Topic:       "Hash Tables",
		Signature:   "func isAnagram(s string, t string) bool",
		TestCases: []TestCase{
			{Input: "s = \"anagram\", t = \"nagaram\"", Expected: "true"},
			{Input: "s = \"rat\", t = \"car\"", Expected: "false"},
			{Input: "s = \"listen\", t = \"silent\"", Expected: "true"},
			{Input: "s = \"a\", t = \"a\"", Expected: "true"},
			{Input: "s = \"ab\", t = \"ba\"", Expected: "true"},
			{Input: "s = \"ab\", t = \"a\"", Expected: "false"},
		},
		Solution: `func isAnagram(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}
	count := make(map[rune]int)
	for _, char := range s {
		count[char]++
	}
	for _, char := range t {
		count[char]--
		if count[char] < 0 {
			return false
		}
	}
	return true
}`,
		PythonSolution: `def isAnagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    count = {}
    for char in s:
        count[char] = count.get(char, 0) + 1
    for char in t:
        if char not in count:
            return False
        count[char] -= 1
        if count[char] < 0:
            return False
    return True`,
		Explanation: `This problem checks if two strings are anagrams by comparing character frequencies. The key insight is that anagrams must have the same character counts, so we can use a single hash map to track the difference in frequencies.

The algorithm first checks if the strings have the same length - if not, they cannot be anagrams. Then, it counts characters in the first string by incrementing counts in a hash map. Next, it iterates through the second string, decrementing counts. If any count becomes negative, it means the second string has more occurrences of that character than the first, so they cannot be anagrams.

This single-map approach is more space-efficient than using two separate frequency maps. By decrementing instead of maintaining separate counts, we can detect mismatches immediately when a count goes negative.

Edge cases handled include: strings of different lengths (early return), identical strings (anagrams of themselves), strings with repeated characters, and empty strings (anagrams of each other).

An alternative approach would use two frequency maps and compare them, but the single-map decrementing approach is more efficient. Another approach would sort both strings and compare, but that takes O(n log n) time versus O(n) for the frequency approach.

Think of this as having a balance scale: you add weights (characters from first string) on one side, then remove weights (characters from second string) from the other. If the scale goes negative, you know there's a mismatch.`,
	},
	{
		ID:          22,
		Title:       "Contains Duplicate",
		Description: "Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.",
		Difficulty:  "Easy",
		Topic:       "Hash Tables",
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
		PythonSolution: `def containsDuplicate(nums: List[int]) -> bool:
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False`,
		Explanation: `This problem can be solved efficiently using a hash set to detect duplicates in a single pass. The key insight is that we can check for duplicates as we iterate, without needing to compare every pair of elements.

The algorithm maintains a hash set to track numbers we've already encountered. As we iterate through the array, we check if the current number is already in the set. If it is, we've found a duplicate and can immediately return true. Otherwise, we add the number to the set and continue. If we complete the iteration without finding any duplicates, we return false.

This approach leverages the O(1) average-case lookup and insertion time of hash sets, resulting in an overall O(n) time complexity. The space complexity is O(n) in the worst case when all elements are unique and we need to store them all.

Edge cases handled include: arrays with a single element (no duplicates possible), empty arrays (no duplicates), arrays with multiple duplicates (returns true on first duplicate found), and arrays where all elements are the same (duplicate found immediately).

An alternative O(n log n) approach would be to sort the array first and then check adjacent elements, but this modifies the input and is slower. Another O(n²) approach would use nested loops, but the hash set solution is optimal for time complexity.

Think of this as checking guests at a party: as each person arrives, you check your guest list. If you've seen them before, you've found a duplicate guest.`,
	},
	{
		ID:          23,
		Title:       "Best Time to Buy and Sell Stock",
		Description: "You are given an array prices where prices[i] is the price of a given stock on the ith day. You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock. Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.",
		Difficulty:  "Easy",
		Topic:       "Arrays",
		Signature:   "func maxProfit(prices []int) int",
		TestCases: []TestCase{
			{Input: "prices = []int{7,1,5,3,6,4}", Expected: "5"},
			{Input: "prices = []int{7,6,4,3,1}", Expected: "0"},
			{Input: "prices = []int{1,2}", Expected: "1"},
			{Input: "prices = []int{2,4,1}", Expected: "2"},
			{Input: "prices = []int{3,2,6,5,0,3}", Expected: "4"},
			{Input: "prices = []int{1}", Expected: "0"},
		},
		Solution: `func maxProfit(prices []int) int {
	if len(prices) < 2 {
		return 0
	}
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
		PythonSolution: `def maxProfit(prices: List[int]) -> int:
    if len(prices) < 2:
        return 0
    min_price = prices[0]
    max_profit = 0
    for price in prices[1:]:
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price
    return max_profit`,
		Explanation: `This problem requires finding the maximum profit from a single buy-sell transaction. The key insight is that we need to buy at the lowest price before selling at the highest price, and we can track both the best buying opportunity and best profit in a single pass.

The algorithm maintains two variables: the minimum price seen so far (best buying opportunity) and the maximum profit achievable. As we iterate through the prices array, we update the minimum price whenever we encounter a lower price. For each day, we calculate the profit we would make if we sold on that day (current price minus minimum price) and update the maximum profit if this is better than our current best.

This approach is optimal because we only need a single pass through the array. We're essentially tracking the best buying opportunity (minimum price) as we go, and calculating the best selling opportunity (maximum profit) at each step.

Edge cases handled include: prices that only decrease (maximum profit is 0), prices that only increase (buy on first day, sell on last day), arrays with fewer than two prices (no transaction possible), and prices with multiple local minima and maxima.

An alternative O(n²) approach would check all pairs of buy and sell days, but the single-pass approach is optimal. Another approach would use dynamic programming, but it's unnecessary for this simpler problem.

Intuitively, think of this as tracking the lowest valley (best buy price) and calculating the highest peak after that valley (best sell price). We update our best buy price whenever we see a lower price, and update our best profit whenever selling at the current price would yield more profit.`,
	},
	{
		ID:          24,
		Title:       "Valid Palindrome",
		Description: "A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Given a string s, return true if it is a palindrome, or false otherwise.",
		Difficulty:  "Easy",
		Topic:       "Two Pointers",
		Signature:   "func isPalindrome(s string) bool",
		TestCases: []TestCase{
			{Input: "s = \"A man, a plan, a canal: Panama\"", Expected: "true"},
			{Input: "s = \"race a car\"", Expected: "false"},
			{Input: "s = \" \"", Expected: "true"},
			{Input: "s = \"a\"", Expected: "true"},
			{Input: "s = \"Madam\"", Expected: "true"},
			{Input: "s = \"No 'x' in Nixon\"", Expected: "true"},
		},
		Solution: `func isPalindrome(s string) bool {
	left, right := 0, len(s)-1
	for left < right {
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

func isAlphanumeric(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
}

func toLower(c byte) byte {
	if c >= 'A' && c <= 'Z' {
		return c + 32
	}
	return c
}`,
		PythonSolution: `def isPalindrome(s: str) -> bool:
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True`,
		Explanation: `This problem requires checking if a string is a palindrome after normalizing it (case-insensitive, alphanumeric only). The two-pointer technique is ideal for the comparison phase, but we need to handle skipping non-alphanumeric characters.

The algorithm uses two pointers starting from both ends of the string. As we move the pointers toward the center, we skip any non-alphanumeric characters by advancing the pointers until we find valid characters. Once both pointers point to alphanumeric characters, we compare them (case-insensitively). If they don't match, we return false. If the pointers meet or cross, the string is a palindrome.

This approach avoids creating a new cleaned string, which would require O(n) extra space. Instead, we process the original string in-place, skipping invalid characters as we go.

Edge cases handled include: empty strings (considered palindromes), strings with only non-alphanumeric characters (empty after cleaning, so palindrome), single characters, strings with mixed case and punctuation, and strings where all characters are the same.

An alternative approach would be to create a cleaned string first, then use two pointers on that, but that requires O(n) extra space. The in-place skipping approach uses O(1) additional space.

Think of this as two people reading a book from opposite ends, skipping over punctuation and spaces, and checking if they're reading the same sequence of letters.`,
	},
	{
		ID:          25,
		Title:       "Missing Number",
		Description: "Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.",
		Difficulty:  "Easy",
		Topic:       "Arrays",
		Signature:   "func missingNumber(nums []int) int",
		TestCases: []TestCase{
			{Input: "nums = []int{3,0,1}", Expected: "2"},
			{Input: "nums = []int{0,1}", Expected: "2"},
			{Input: "nums = []int{9,6,4,2,3,5,7,0,1}", Expected: "8"},
			{Input: "nums = []int{0}", Expected: "1"},
			{Input: "nums = []int{1}", Expected: "0"},
			{Input: "nums = []int{1,2}", Expected: "0"},
		},
		Solution: `func missingNumber(nums []int) int {
	n := len(nums)
	expectedSum := n * (n + 1) / 2
	actualSum := 0
	for _, num := range nums {
		actualSum += num
	}
	return expectedSum - actualSum
}`,
		PythonSolution: `def missingNumber(nums: List[int]) -> int:
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum`,
		Explanation: `This problem can be solved efficiently using the mathematical property that the sum of integers from 0 to n equals n*(n+1)/2. The missing number is simply the difference between the expected sum and the actual sum of the array.

The algorithm calculates the expected sum of all integers from 0 to n using the formula n*(n+1)/2, where n is the length of the array (since the array contains n distinct numbers in the range [0, n], meaning one number is missing). Then, it calculates the actual sum of all numbers in the array. The difference between these two sums is the missing number.

This approach is optimal because it requires only a single pass through the array to calculate the sum, resulting in O(n) time complexity. The space complexity is O(1) since we only use a constant amount of extra space for the sum variables.

Edge cases handled include: missing number is 0, missing number is n (the largest number), arrays with a single element, and arrays where numbers are not in order (the sum approach doesn't require sorting).

An alternative approach would be to use XOR: XOR all numbers from 0 to n with all numbers in the array, and the result is the missing number. This also has O(n) time and O(1) space. Another approach would sort the array and find the gap, but that takes O(n log n) time.

Think of this as having a complete set of numbered items from 0 to n, and one is missing. If you know what the total should be and what you actually have, the difference tells you what's missing.`,
	},
	{
		ID:          26,
		Title:       "Climbing Stairs",
		Description: "You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?",
		Difficulty:  "Easy",
		Topic:       "Dynamic Programming",
		Signature:   "func climbStairs(n int) int",
		TestCases: []TestCase{
			{Input: "n = 2", Expected: "2"},
			{Input: "n = 3", Expected: "3"},
			{Input: "n = 4", Expected: "5"},
			{Input: "n = 5", Expected: "8"},
			{Input: "n = 1", Expected: "1"},
			{Input: "n = 10", Expected: "89"},
		},
		Solution: `func climbStairs(n int) int {
	if n <= 2 {
		return n
	}
	first, second := 1, 2
	for i := 3; i <= n; i++ {
		first, second = second, first+second
	}
	return second
}`,
		PythonSolution: `def climbStairs(n: int) -> int:
    if n <= 2:
        return n
    first, second = 1, 2
    for i in range(3, n + 1):
        first, second = second, first + second
    return second`,
		Explanation: `This problem is another classic Fibonacci sequence application. The recurrence relation is the same: ways to reach step n equals the sum of ways to reach step (n-1) and step (n-2), since you can only take 1 or 2 steps at a time.

The algorithm uses an iterative approach with two variables to track the previous two Fibonacci numbers, avoiding the need for an array or recursion stack. Starting from the base cases, we build up the solution by computing each step as the sum of the previous two steps. This gives us O(n) time complexity with O(1) space complexity.

The key optimization is using only two variables instead of storing all previous values. We update these variables in each iteration: the new value becomes the sum of the two previous values, and we shift the variables forward. This is more space-efficient than maintaining an array of all values.

Edge cases handled include: n=1 (1 way: take 1 step), n=2 (2 ways: two 1-steps or one 2-step), and n=0 (typically 1 way, though this may not be in the problem constraints).

An alternative recursive approach would have O(2^n) time complexity without memoization, or O(n) time and O(n) space with memoization. The iterative two-variable approach achieves optimal time and space complexity.

Think of this as building possibilities step by step: each step's number of ways builds on the previous two steps, creating the Fibonacci sequence where each number is the sum of the two preceding ones.`,
	},
	{
		ID:          27,
		Title:       "Maximum Subarray",
		Description: "Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum. A subarray is a contiguous part of an array.",
		Difficulty:  "Easy",
		Topic:       "Dynamic Programming",
		Signature:   "func maxSubArray(nums []int) int",
		TestCases: []TestCase{
			{Input: "nums = []int{-2,1,-3,4,-1,2,1,-5,4}", Expected: "6"},
			{Input: "nums = []int{1}", Expected: "1"},
			{Input: "nums = []int{5,4,-1,7,8}", Expected: "23"},
			{Input: "nums = []int{-1}", Expected: "-1"},
			{Input: "nums = []int{-2,-1}", Expected: "-1"},
			{Input: "nums = []int{1,2,3,4,5}", Expected: "15"},
		},
		Solution: `func maxSubArray(nums []int) int {
	maxSum := nums[0]
	currentSum := nums[0]
	for i := 1; i < len(nums); i++ {
		if currentSum < 0 {
			currentSum = 0
		}
		currentSum += nums[i]
		if currentSum > maxSum {
			maxSum = currentSum
		}
	}
	return maxSum
}`,
		PythonSolution: `def maxSubArray(nums: List[int]) -> int:
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        if current_sum < 0:
            current_sum = 0
        current_sum += num
        max_sum = max(max_sum, current_sum)
    return max_sum`,
		Explanation: `Kadane's algorithm is the optimal solution for finding the maximum sum of a contiguous subarray. The key insight is that if the current sum becomes negative, it's better to start a new subarray from the next element rather than carrying forward a negative sum.

The algorithm maintains two variables: the current sum of the subarray ending at the current position, and the maximum sum seen so far. As we iterate through the array, we add each element to the current sum. If the current sum becomes negative, we reset it to 0, effectively starting a new subarray from the next position. At each step, we update the maximum sum if the current sum is greater.

This approach works because any subarray that starts with a negative sum can be improved by starting from the next element instead. By resetting to 0 when the sum becomes negative, we ensure we're always considering the best possible starting point for our current subarray.

Edge cases handled include: arrays with all negative numbers (maximum sum is the least negative number), arrays with all positive numbers (sum of entire array), arrays with a single element, and arrays with mixed positive and negative numbers.

An alternative O(n²) approach would check all possible subarrays, but Kadane's algorithm is optimal. Another approach would use divide-and-conquer, but it's more complex and still O(n log n) time. Kadane's algorithm achieves O(n) time with O(1) space.

Think of this as walking along a path collecting coins: if your current collection becomes negative (you've lost more than gained), it's better to drop everything and start fresh from the next position rather than carrying forward losses.`,
	},
	{
		ID:          28,
		Title:       "Single Number",
		Description: "Given a non-empty array of integers nums, every element appears twice except for one. Find that single one. You must implement a solution with a linear runtime complexity and use only constant extra space.",
		Difficulty:  "Easy",
		Topic:       "Bit Manipulation",
		Signature:   "func singleNumber(nums []int) int",
		TestCases: []TestCase{
			{Input: "nums = []int{2,2,1}", Expected: "1"},
			{Input: "nums = []int{4,1,2,1,2}", Expected: "4"},
			{Input: "nums = []int{1}", Expected: "1"},
			{Input: "nums = []int{1,2,3,2,1}", Expected: "3"},
			{Input: "nums = []int{-1,-2,-2,-1,5}", Expected: "5"},
			{Input: "nums = []int{7,3,5,3,7}", Expected: "5"},
		},
		Solution: `func singleNumber(nums []int) int {
	result := 0
	for _, num := range nums {
		result ^= num
	}
	return result
}`,
		PythonSolution: `def singleNumber(nums: List[int]) -> int:
    result = 0
    for num in nums:
        result ^= num
    return result`,
		Explanation: `This problem can be solved elegantly using the XOR bitwise operation, which has the perfect properties for this use case. The key properties are: any number XORed with itself equals 0, any number XORed with 0 equals itself, and XOR is commutative and associative.

The algorithm initializes a result variable to 0, then XORs all numbers in the array together. Since every number except one appears exactly twice, all pairs will cancel out (a XOR a = 0), leaving only the single number that appears once. This single number XORed with 0 equals itself, giving us our answer.

This approach is optimal because it requires only a single pass through the array with O(1) extra space. The XOR operation is very fast and doesn't require any additional data structures like hash maps or sets.

Edge cases handled include: arrays with a single element (returns that element), arrays where the single number is 0 (XOR works correctly), arrays with negative numbers (XOR works on two's complement representation), and arrays with the single number at any position (XOR is commutative).

An alternative O(n) approach would use a hash map to count frequencies, but that requires O(n) space. Another approach would sort the array and check adjacent elements, but that requires O(n log n) time. The XOR approach is optimal for both time and space.

Think of XOR as a toggle: pairing numbers cancel each other out, leaving only the unpaired number. It's like having light switches where flipping the same switch twice returns it to the original state, and only one switch is flipped once, leaving that one on.`,
	},
	{
		ID:          29,
		Title:       "Intersection of Two Arrays",
		Description: "Given two integer arrays nums1 and nums2, return an array of their intersection. Each element in the result must be unique and you may return the result in any order.",
		Difficulty:  "Easy",
		Topic:       "Hash Tables",
		Signature:   "func intersection(nums1 []int, nums2 []int) []int",
		TestCases: []TestCase{
			{Input: "nums1 = []int{1,2,2,1}, nums2 = []int{2,2}", Expected: "[2]"},
			{Input: "nums1 = []int{4,9,5}, nums2 = []int{9,4,9,8,4}", Expected: "[4 9]"},
			{Input: "nums1 = []int{1,2,3}, nums2 = []int{4,5,6}", Expected: "[]"},
			{Input: "nums1 = []int{1}, nums2 = []int{1}", Expected: "[1]"},
			{Input: "nums1 = []int{1,2}, nums2 = []int{2,3}", Expected: "[2]"},
			{Input: "nums1 = []int{}, nums2 = []int{1,2}", Expected: "[]"},
		},
		Solution: `func intersection(nums1 []int, nums2 []int) []int {
	set1 := make(map[int]bool)
	for _, num := range nums1 {
		set1[num] = true
	}
	result := []int{}
	seen := make(map[int]bool)
	for _, num := range nums2 {
		if set1[num] && !seen[num] {
			result = append(result, num)
			seen[num] = true
		}
	}
	return result
}`,
		PythonSolution: `def intersection(nums1: List[int], nums2: List[int]) -> List[int]:
    set1 = set(nums1)
    result = []
    seen = set()
    for num in nums2:
        if num in set1 and num not in seen:
            result.append(num)
            seen.add(num)
    return result`,
		Explanation: `Finding the intersection of two arrays requires identifying elements that appear in both arrays, with each element appearing only once in the result. The hash set data structure is perfect for efficient lookups.

The algorithm first converts the first array into a hash set, which provides O(1) average-case lookup time. Then, it iterates through the second array, checking if each element exists in the first set. If it does and we haven't added it to the result yet (tracked by a separate "seen" set to avoid duplicates), we add it to the result.

The use of two sets - one for the first array and one for tracking already-added elements - ensures we handle duplicates correctly. An element from the second array is only added to the result if it exists in the first array and hasn't been added before.

Edge cases handled include: arrays with no common elements (returns empty array), arrays where one is a subset of the other, arrays with duplicate values in the second array (each unique common element appears once), and empty arrays.

An alternative approach would be to sort both arrays and use two pointers, but that requires O(n log n + m log m) time. Another approach would use nested loops, but that's O(n*m) time. The hash set approach achieves O(n+m) time with O(n+m) space.

Think of this as finding common items between two shopping lists: you check each item from the second list against the first list, and keep track of which common items you've already added to avoid duplicates.`,
	},
	{
		ID:          30,
		Title:       "Move Zeroes",
		Description: "Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements. Note that you must do this in-place without making a copy of the array.",
		Difficulty:  "Easy",
		Topic:       "Two Pointers",
		Signature:   "func moveZeroes(nums []int)",
		TestCases: []TestCase{
			{Input: "nums = []int{0,1,0,3,12}", Expected: "[1 3 12 0 0]"},
			{Input: "nums = []int{0}", Expected: "[0]"},
			{Input: "nums = []int{1,2,3}", Expected: "[1 2 3]"},
			{Input: "nums = []int{0,0,1}", Expected: "[1 0 0]"},
			{Input: "nums = []int{1,0,1,0,1}", Expected: "[1 1 1 0 0]"},
			{Input: "nums = []int{0,0,0,1}", Expected: "[1 0 0 0]"},
		},
		Solution: `func moveZeroes(nums []int) {
	writeIndex := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[writeIndex], nums[i] = nums[i], nums[writeIndex]
			writeIndex++
		}
	}
}`,
		PythonSolution: `def moveZeroes(nums: List[int]) -> None:
    write_index = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[write_index], nums[i] = nums[i], nums[write_index]
            write_index += 1`,
		Explanation: `Moving all zeros to the end while maintaining the relative order of non-zero elements requires careful in-place manipulation. The two-pointer technique allows us to do this efficiently in a single pass without creating a new array.

The algorithm uses two pointers: writeIndex tracks the position where the next non-zero element should be placed, and i scans through the array. When i encounters a non-zero element, we swap it with the element at writeIndex (which might be zero or already a non-zero element), then increment writeIndex. This ensures all non-zero elements are moved to the front while maintaining their relative order.

The key insight is that writeIndex always points to the first position that might contain a zero (or is ready for the next non-zero element). By swapping non-zero elements to this position, we naturally collect all non-zeros at the front, leaving zeros at the end.

Edge cases handled include: arrays with no zeros (elements remain in place with minimal swaps), arrays with all zeros (no swaps needed, just scanning), arrays with zeros only at the end (already in correct position), and arrays with zeros only at the beginning (all non-zeros move forward).

An alternative approach would be to count zeros and overwrite, but that requires two passes. Another approach would create a new array, but that requires O(n) extra space. The two-pointer swap approach is optimal for both time and space.

Think of this as organizing items on a shelf: you want all the important items (non-zeros) at the front, so you swap each important item you find with whatever is at the front position, gradually pushing zeros to the back.`,
	},
	{
		ID:          31,
		Title:       "Two Number Sum",
		Description: "Write a function that takes in a non-empty array of distinct integers and an integer representing a target sum. If any two numbers in the input array sum up to the target sum, the function should return them in an array, in any order. If no two numbers sum up to the target sum, the function should return an empty array.",
		Difficulty:  "Easy",
		Topic:       "Hash Tables",
		Signature:   "func twoNumberSum(array []int, targetSum int) []int",
		TestCases: []TestCase{
			{Input: "array = []int{3, 5, -4, 8, 11, 1, -1, 6}, targetSum = 10", Expected: "[11 -1]"},
			{Input: "array = []int{4, 6}, targetSum = 10", Expected: "[4 6]"},
			{Input: "array = []int{4, 6, 1}, targetSum = 5", Expected: "[4 1]"},
			{Input: "array = []int{4, 6, 1, -3}, targetSum = 3", Expected: "[6 -3]"},
			{Input: "array = []int{1, 2, 3, 4, 5, 6, 7, 8, 9}, targetSum = 17", Expected: "[8 9]"},
			{Input: "array = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15}, targetSum = 18", Expected: "[3 15]"},
		},
		Solution: `func twoNumberSum(array []int, targetSum int) []int {
	nums := make(map[int]bool)
	for _, num := range array {
		potentialMatch := targetSum - num
		if nums[potentialMatch] {
			return []int{potentialMatch, num}
		}
		nums[num] = true
	}
	return []int{}
}`,
		PythonSolution: `def twoNumberSum(array: List[int], targetSum: int) -> List[int]:
    nums = {}
    for num in array:
        potential_match = targetSum - num
        if potential_match in nums:
            return [potential_match, num]
        nums[num] = True
    return []`,
		Explanation: `This problem is similar to Two Sum but returns the actual numbers instead of indices. The hash map approach works perfectly here, allowing us to find the complement efficiently.

The algorithm uses a hash map to store numbers we've seen as we iterate through the array. For each number, we calculate its complement (targetSum - num) and check if that complement already exists in our hash map. If it does, we've found our pair and can immediately return both numbers. If not, we add the current number to the hash map for future lookups.

This approach eliminates the need for nested loops, which would result in O(n²) time complexity. Instead, hash map lookups are O(1) on average, making the overall solution O(n). The space complexity is O(n) in the worst case when we need to store all elements before finding a match.

Edge cases handled include: arrays with duplicate numbers (the hash map stores boolean values, so duplicates don't cause issues), negative numbers (complement calculation works correctly), arrays with exactly two elements that sum to target, and arrays where no pair exists (returns empty array).

An alternative O(n²) approach would be to check all pairs, but the hash map solution is optimal for time complexity. The trade-off is using O(n) extra space, which is acceptable for the significant time improvement.

Think of this as building a "wishlist" as we go: for each number we see, we remember what number we're looking for to complete the pair. When we find that number, we immediately return both.`,
	},
	{
		ID:          32,
		Title:       "Validate Subsequence",
		Description: "Given two non-empty arrays of integers, write a function that determines whether the second array is a subsequence of the first one. A subsequence of an array is a set of numbers that aren't necessarily adjacent in the array but that are in the same order as they appear in the array.",
		Difficulty:  "Easy",
		Topic:       "Arrays",
		Signature:   "func isValidSubsequence(array []int, sequence []int) bool",
		TestCases: []TestCase{
			{Input: "array = []int{5, 1, 22, 25, 6, -1, 8, 10}, sequence = []int{1, 6, -1, 10}", Expected: "true"},
			{Input: "array = []int{5, 1, 22, 25, 6, -1, 8, 10}, sequence = []int{22, 25, 6}", Expected: "true"},
			{Input: "array = []int{5, 1, 22, 25, 6, -1, 8, 10}, sequence = []int{5, 1, 22, 25, 6, -1, 8, 10}", Expected: "true"},
			{Input: "array = []int{5, 1, 22, 25, 6, -1, 8, 10}, sequence = []int{5, 1, 22, 25, 6, -1, 8, 10, 12}", Expected: "false"},
			{Input: "array = []int{1, 1, 1, 1, 1}, sequence = []int{1, 1, 1}", Expected: "true"},
			{Input: "array = []int{5, 1, 22, 25, 6, -1, 8, 10}, sequence = []int{26}", Expected: "false"},
		},
		Solution: `func isValidSubsequence(array []int, sequence []int) bool {
	seqIdx := 0
	for _, value := range array {
		if seqIdx == len(sequence) {
			break
		}
		if value == sequence[seqIdx] {
			seqIdx++
		}
	}
	return seqIdx == len(sequence)
}`,
		PythonSolution: `def isValidSubsequence(array: List[int], sequence: List[int]) -> bool:
    seq_idx = 0
    for value in array:
        if seq_idx == len(sequence):
            break
        if value == sequence[seq_idx]:
            seq_idx += 1
    return seq_idx == len(sequence)`,
		Explanation: `This problem checks if one array is a subsequence of another. A subsequence means the elements appear in the same order, but not necessarily consecutively. The key insight is that we only need a single pointer to track our progress through the sequence.

The algorithm uses a single pointer (seqIdx) to track our position in the sequence array. As we iterate through the main array, whenever we find an element that matches the current element in the sequence (at seqIdx), we advance the sequence pointer. If we successfully match all elements in the sequence (seqIdx reaches the length of the sequence), the sequence is valid.

This approach is efficient because we only need a single pass through the main array. We don't need to backtrack or check multiple possibilities - we simply move forward through both arrays, matching elements as we encounter them in order.

Edge cases handled include: empty sequence (always valid, as it's a subsequence of any array), sequence longer than array (invalid), sequence with duplicate values (each occurrence must be matched in order), and sequence that appears at the end of the array.

An alternative approach would use two nested loops, but that's less efficient. Another approach would use dynamic programming, but it's unnecessary for this simpler problem. The single-pointer approach is optimal.

Think of this as checking if you can find all items on a shopping list in a store, in order. You walk through the store once, and as you find each item on your list, you check it off. If you check off all items, you've found the sequence.`,
	},
	{
		ID:          33,
		Title:       "Sorted Squared Array",
		Description: "Write a function that takes in a non-empty array of integers that are sorted in ascending order and returns a new array of the same length with the squares of the original integers also sorted in ascending order.",
		Difficulty:  "Easy",
		Topic:       "Arrays",
		Signature:   "func sortedSquaredArray(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = []int{1, 2, 3, 5, 6, 8, 9}", Expected: "[1 4 9 25 36 64 81]"},
			{Input: "array = []int{1}", Expected: "[1]"},
			{Input: "array = []int{1, 2}", Expected: "[1 4]"},
			{Input: "array = []int{1, 2, 3, 4, 5}", Expected: "[1 4 9 16 25]"},
			{Input: "array = []int{-10, -5, 0, 5, 10}", Expected: "[0 25 25 100 100]"},
			{Input: "array = []int{-7, -3, 2, 3, 11}", Expected: "[4 9 9 49 121]"},
		},
		Solution: `func sortedSquaredArray(array []int) []int {
	result := make([]int, len(array))
	left := 0
	right := len(array) - 1
	for i := len(array) - 1; i >= 0; i-- {
		leftVal := array[left]
		rightVal := array[right]
		if abs(leftVal) > abs(rightVal) {
			result[i] = leftVal * leftVal
			left++
		} else {
			result[i] = rightVal * rightVal
			right--
		}
	}
	return result
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}`,
		PythonSolution: `def sortedSquaredArray(array: List[int]) -> List[int]:
    result = [0] * len(array)
    left = 0
    right = len(array) - 1
    for i in range(len(array) - 1, -1, -1):
        left_val = array[left]
        right_val = array[right]
        if abs(left_val) > abs(right_val):
            result[i] = left_val * left_val
            left += 1
        else:
            result[i] = right_val * right_val
            right -= 1
    return result`,
		Explanation: `This problem requires squaring a sorted array and returning the result in sorted order. The challenge is that negative numbers become positive when squared, so their squares might be larger than squares of smaller positive numbers. The two-pointer technique handles this elegantly.

The algorithm uses two pointers starting from both ends of the array. Since the array is sorted, the largest absolute values are at the ends. We compare the absolute values at both ends, square the larger one, and place it at the end of the result array (filling from right to left). We then move the pointer that contributed the larger square inward and repeat.

This approach works because squares are always non-negative, and the largest squares come from the largest absolute values. By processing from both ends and filling the result from right to left, we ensure the result array is sorted in ascending order.

Edge cases handled include: arrays with all negative numbers (squares are in reverse order of absolute values), arrays with all positive numbers (squares maintain original order), arrays with both negative and positive numbers (largest squares come from ends), and arrays with zeros.

An alternative approach would be to square all elements and sort, but that requires O(n log n) time. The two-pointer approach achieves O(n) time by leveraging the sorted property of the input array.

Think of this as having two people at opposite ends of a sorted line, each holding a number. You compare which number has a larger square, put that square at the end of your result, and have that person step forward. Continue until they meet.`,
	},
	{
		ID:          34,
		Title:       "Tournament Winner",
		Description: "There's an algorithms tournament taking place in which teams of programmers compete against each other to solve algorithmic problems as fast as possible. Teams compete in a round robin, where each team faces off against all other teams. Only two teams compete against each other at a time, and for each competition, one team is designated the home team, while the other team is the away team. In each competition there's always one winner and one loser; there are no ties. A team receives 3 points if it wins and 0 points if it loses. The winner of the tournament is the team that receives the most amount of points. Given an array of pairs representing the teams that have competed against each other and an array containing the results of each competition, write a function that returns the winner of the tournament.",
		Difficulty:  "Easy",
		Topic:       "Hash Tables",
		Signature:   "func tournamentWinner(competitions [][]string, results []int) string",
		TestCases: []TestCase{
			{Input: "competitions = [[\"HTML\", \"C#\"], [\"C#\", \"Python\"], [\"Python\", \"HTML\"]], results = []int{0, 0, 1}", Expected: "\"Python\""},
			{Input: "competitions = [[\"HTML\", \"C#\"]], results = []int{0}", Expected: "\"C#\""},
			{Input: "competitions = [[\"A\", \"B\"]], results = []int{1}", Expected: "\"A\""},
			{Input: "competitions = [[\"HTML\", \"Java\"], [\"Java\", \"Python\"], [\"Python\", \"HTML\"]], results = []int{0, 1, 1}", Expected: "\"Java\""},
		},
		Solution: `func tournamentWinner(competitions [][]string, results []int) string {
	scores := make(map[string]int)
	bestTeam := ""
	bestScore := 0
	for i, competition := range competitions {
		homeTeam := competition[0]
		awayTeam := competition[1]
		winner := awayTeam
		if results[i] == 1 {
			winner = homeTeam
		}
		scores[winner] += 3
		if scores[winner] > bestScore {
			bestScore = scores[winner]
			bestTeam = winner
		}
	}
	return bestTeam
}`,
		PythonSolution: `def tournamentWinner(competitions: List[List[str]], results: List[int]) -> str:
    scores = {}
    best_team = ""
    best_score = 0
    for i, competition in enumerate(competitions):
        home_team, away_team = competition
        winner = away_team if results[i] == 0 else home_team
        scores[winner] = scores.get(winner, 0) + 3
        if scores[winner] > best_score:
            best_score = scores[winner]
            best_team = winner
    return best_team`,
		Explanation: `This problem requires tracking scores across multiple competitions and finding the team with the highest total score. The hash map data structure is perfect for maintaining team scores efficiently.

The algorithm maintains a hash map to store each team's total score. For each competition, we determine the winner based on the result: if result is 1, the home team wins; if result is 0, the away team wins. We add 3 points to the winner's score in the hash map. Simultaneously, we track the best team and best score seen so far, updating them whenever we find a team with a higher score.

This approach is efficient because we process each competition exactly once, and hash map operations are O(1) on average. We also avoid a second pass to find the maximum by tracking it during the first pass.

Edge cases handled include: single competition (winner is the best team), competitions where one team wins all matches, competitions with ties (not possible per problem statement), and competitions where multiple teams have the same score (returns the last one encountered with that score, or can be modified to return the first).

An alternative approach would be to calculate all scores first, then find the maximum in a second pass, but that's less efficient. The single-pass approach with tracking is optimal.

Think of this as a tournament scoreboard: as each match finishes, you update the scores and keep an eye on who's currently in the lead. At the end, you know the winner.`,
	},
	{
		ID:          35,
		Title:       "Non-Constructible Change",
		Description: "Given an array of positive integers representing the values of coins in your possession, write a function that returns the minimum amount of change (the minimum sum of money) that you cannot create. The given coins can have any positive integer value and aren't necessarily unique (i.e., you can have multiple coins of the same value).",
		Difficulty:  "Easy",
		Topic:       "Greedy",
		Signature:   "func nonConstructibleChange(coins []int) int",
		TestCases: []TestCase{
			{Input: "coins = []int{1, 2, 5}", Expected: "4"},
			{Input: "coins = []int{5, 7, 1, 1, 2, 3, 22}", Expected: "20"},
			{Input: "coins = []int{1, 1, 1, 1, 1}", Expected: "6"},
			{Input: "coins = []int{1, 5, 1, 1, 1, 10, 15, 20, 100}", Expected: "55"},
			{Input: "coins = []int{6, 4, 5, 1, 1, 8, 9}", Expected: "3"},
			{Input: "coins = []int{1}", Expected: "2"},
		},
		Solution: `import "sort"

func nonConstructibleChange(coins []int) int {
	sort.Ints(coins)
	changeCreated := 0
	for _, coin := range coins {
		if coin > changeCreated+1 {
			return changeCreated + 1
		}
		changeCreated += coin
	}
	return changeCreated + 1
}`,
		PythonSolution: `def nonConstructibleChange(coins: List[int]) -> int:
    coins.sort()
    change_created = 0
    for coin in coins:
        if coin > change_created + 1:
            return change_created + 1
        change_created += coin
    return change_created + 1`,
		Explanation: `This problem requires finding the smallest amount of change that cannot be created from the given coins. The key insight is that if we can create all amounts from 1 to changeCreated, and the next coin is greater than changeCreated + 1, then we've found a gap we cannot fill.

The algorithm first sorts the coins in ascending order. Then, it iterates through the coins, maintaining changeCreated which represents the maximum amount of change we can create with the coins seen so far. For each coin, if it's greater than changeCreated + 1, we've found our answer: changeCreated + 1 is the smallest amount we cannot create. Otherwise, we add the coin's value to changeCreated, extending the range of change we can create.

This greedy approach works because by processing coins in sorted order and always extending our range from the smallest possible gap, we ensure we're creating the maximum possible range of change amounts. If a coin is too large to fill the next gap, that gap is our answer.

Edge cases handled include: coins starting with 1 (can create 1, then check if we can create 2), coins with large gaps (first gap found is the answer), and coins that allow creating all amounts up to a certain point (answer is the next amount after that point).

An alternative approach would be to use dynamic programming to determine which amounts can be created, but that would be O(n * amount) time and space. The greedy approach is more efficient for this specific problem.

Think of this as building a staircase: if you have blocks of certain sizes and you've built steps up to height H, you can only build the next step if you have a block that's at most H+1. If the next block is larger, there's a gap you cannot fill.`,
	},
	{
		ID:          36,
		Title:       "Transpose Matrix",
		Description: "You're given a 2D array of integers matrix. Write a function that returns the transpose of the matrix. The transpose of a matrix is the matrix flipped over its main diagonal, switching the row and column indices of the matrix.",
		Difficulty:  "Easy",
		Topic:       "Arrays",
		Signature:   "func transposeMatrix(matrix [][]int) [][]int",
		TestCases: []TestCase{
			{Input: "matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]", Expected: "[[1 4 7] [2 5 8] [3 6 9]]"},
			{Input: "matrix = [[1, 2], [3, 4], [5, 6]]", Expected: "[[1 3 5] [2 4 6]]"},
			{Input: "matrix = [[1]]", Expected: "[[1]]"},
			{Input: "matrix = [[1, 2, 3, 4]]", Expected: "[[1] [2] [3] [4]]"},
		},
		Solution: `func transposeMatrix(matrix [][]int) [][]int {
	rows := len(matrix)
	cols := len(matrix[0])
	transposed := make([][]int, cols)
	for i := range transposed {
		transposed[i] = make([]int, rows)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			transposed[j][i] = matrix[i][j]
		}
	}
	return transposed
}`,
		PythonSolution: `def transposeMatrix(matrix: List[List[int]]) -> List[List[int]]:
    rows = len(matrix)
    cols = len(matrix[0])
    transposed = [[0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    return transposed`,
		Explanation: `Transposing a matrix means flipping it over its main diagonal, which swaps row and column indices. The key insight is that element at position [i][j] in the original matrix becomes element [j][i] in the transposed matrix.

The algorithm first creates a new matrix with swapped dimensions: if the original matrix has rows rows and cols columns, the transposed matrix will have cols rows and rows columns. Then, it iterates through the original matrix, copying each element at position [i][j] to position [j][i] in the transposed matrix.

This approach is straightforward and efficient. We need to create a new matrix because transposing in-place is only possible for square matrices. For rectangular matrices, we must create a new matrix with different dimensions.

Edge cases handled include: square matrices (dimensions swap but size remains same), rectangular matrices (rows and columns swap), single-row matrices (become single-column), single-column matrices (become single-row), and 1x1 matrices (remain unchanged).

An alternative approach for square matrices would be to transpose in-place by swapping elements, but for general matrices, creating a new matrix is necessary. The time and space complexity are both O(n*m) where n and m are the dimensions.

Think of this as rotating a grid 90 degrees: what was a row becomes a column, and what was a column becomes a row.`,
	},
	{
		ID:          37,
		Title:       "Find Closest Value In BST",
		Description: "Write a function that takes in a Binary Search Tree (BST) and a target integer value and returns the closest value to that target value contained in the BST. You can assume that there will only be one closest value.",
		Difficulty:  "Easy",
		Topic:       "Binary Search Trees",
		Signature:   "func findClosestValueInBst(tree *TreeNode, target int) int",
		TestCases: []TestCase{
			{Input: "tree = [10, 5, 15, 2, 5, 13, 22, 1, null, null, null, null, 14], target = 12", Expected: "13"},
			{Input: "tree = [10, 5, 15, 2, 5, 13, 22, 1, null, null, null, null, 14], target = 10", Expected: "10"},
			{Input: "tree = [10, 5, 15, 2, 5, 13, 22, 1, null, null, null, null, 14], target = 14", Expected: "14"},
			{Input: "tree = [10], target = 5", Expected: "10"},
		},
		Solution: `func findClosestValueInBst(tree *TreeNode, target int) int {
	return findClosestHelper(tree, target, tree.Val)
}

func findClosestHelper(tree *TreeNode, target int, closest int) int {
	if tree == nil {
		return closest
	}
	if abs(target-closest) > abs(target-tree.Val) {
		closest = tree.Val
	}
	if target < tree.Val {
		return findClosestHelper(tree.Left, target, closest)
	} else if target > tree.Val {
		return findClosestHelper(tree.Right, target, closest)
	}
	return closest
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}`,
		PythonSolution: `def findClosestValueInBst(tree: Optional[TreeNode], target: int) -> int:
    def find_closest_helper(node, target, closest):
        if node is None:
            return closest
        if abs(target - closest) > abs(target - node.val):
            closest = node.val
        if target < node.val:
            return find_closest_helper(node.left, target, closest)
        elif target > node.val:
            return find_closest_helper(node.right, target, closest)
        return closest
    return find_closest_helper(tree, target, tree.val)`,
		Explanation: `Finding the closest value in a BST leverages the tree's sorted property to eliminate half of the remaining search space at each step. The key insight is that we can use the BST property to guide our search, similar to binary search.

The algorithm uses recursive traversal, maintaining a closest value that tracks the best candidate found so far. At each node, we compare the absolute difference between the target and the current node's value with the absolute difference between the target and our current closest value. If the current node is closer, we update closest. Then, we use the BST property: if the target is less than the current node's value, we search the left subtree (all values there are smaller); if greater, we search the right subtree (all values there are larger).

This approach is efficient because we only explore one path from root to leaf, eliminating half the tree at each step (in a balanced BST). The BST property ensures we're always moving toward the target value.

Edge cases handled include: target equals a node value (that node is the closest), target is between two node values (we compare both), single-node trees, and unbalanced trees (still works but may take O(n) time in worst case).

An alternative iterative approach would use a while loop instead of recursion, which would have the same time complexity but O(1) space instead of O(h) space for the recursion stack. However, the recursive approach is more intuitive.

Think of this as navigating a sorted list: at each step, you compare your current position with the target and decide which direction to go, always keeping track of the closest value you've seen so far.`,
	},
	{
		ID:          38,
		Title:       "Branch Sums",
		Description: "Write a function that takes in a Binary Tree and returns a list of its branch sums ordered from leftmost branch sum to rightmost branch sum. A branch sum is the sum of all values in a Binary Tree branch. A Binary Tree branch is a path of nodes in a tree that starts at the root node and ends at any leaf node.",
		Difficulty:  "Easy",
		Topic:       "Binary Trees",
		Signature:   "func branchSums(root *TreeNode) []int",
		TestCases: []TestCase{
			{Input: "root = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]", Expected: "[15 16 18 10 11]"},
			{Input: "root = [1]", Expected: "[1]"},
			{Input: "root = [1, 2]", Expected: "[3]"},
			{Input: "root = [1, 2, 3]", Expected: "[3 4]"},
		},
		Solution: `func branchSums(root *TreeNode) []int {
	sums := []int{}
	calculateBranchSums(root, 0, &sums)
	return sums
}

func calculateBranchSums(node *TreeNode, runningSum int, sums *[]int) {
	if node == nil {
		return
	}
	newRunningSum := runningSum + node.Val
	if node.Left == nil && node.Right == nil {
		*sums = append(*sums, newRunningSum)
		return
	}
	calculateBranchSums(node.Left, newRunningSum, sums)
	calculateBranchSums(node.Right, newRunningSum, sums)
}`,
		PythonSolution: `def branchSums(root: Optional[TreeNode]) -> List[int]:
    sums = []
    def calculate_branch_sums(node, running_sum):
        if node is None:
            return
        new_running_sum = running_sum + node.val
        if node.left is None and node.right is None:
            sums.append(new_running_sum)
            return
        calculate_branch_sums(node.left, new_running_sum)
        calculate_branch_sums(node.right, new_running_sum)
    calculate_branch_sums(root, 0)
    return sums`,
		Explanation: `This problem requires calculating the sum of values along each path from root to leaf. Depth-first search (DFS) is the natural approach, as we need to explore each complete path before backtracking.

The algorithm uses recursive DFS, maintaining a running sum that accumulates node values as we traverse down each path. When we reach a leaf node (a node with no children), we add the current running sum to our result list. For internal nodes, we recursively explore both left and right subtrees, passing the updated running sum to each recursive call.

This approach ensures we visit every node exactly once and calculate the sum for each root-to-leaf path. The recursive nature of DFS naturally handles backtracking, allowing us to explore all paths without explicitly managing a stack.

Edge cases handled include: single-node trees (that node's value is the branch sum), trees where all nodes are on one side (only one branch sum), and trees with varying path lengths (each path contributes its own sum).

An alternative iterative approach would use an explicit stack to track nodes and running sums, but the recursive approach is more intuitive and has the same time complexity. Both approaches require O(n) time to visit all nodes and O(h) space for the recursion stack, where h is the height of the tree.

Think of this as walking down each path from the root to every leaf, keeping a running total of what you've collected, and writing down the total when you reach each leaf.`,
	},
	{
		ID:          39,
		Title:       "Node Depths",
		Description: "The distance between a node in a Binary Tree and the tree's root is called the node's depth. Write a function that takes in a Binary Tree and returns the sum of all of its subtrees' nodes' depths.",
		Difficulty:  "Easy",
		Topic:       "Binary Trees",
		Signature:   "func nodeDepths(root *TreeNode) int",
		TestCases: []TestCase{
			{Input: "root = [1, 2, 3, 4, 5, 6, 7, 8, 9]", Expected: "16"},
			{Input: "root = [1]", Expected: "0"},
			{Input: "root = [1, 2, 3, 4, 5]", Expected: "6"},
		},
		Solution: `func nodeDepths(root *TreeNode) int {
	return calculateNodeDepths(root, 0)
}

func calculateNodeDepths(node *TreeNode, depth int) int {
	if node == nil {
		return 0
	}
	return depth + calculateNodeDepths(node.Left, depth+1) + calculateNodeDepths(node.Right, depth+1)
}`,
		PythonSolution: `def nodeDepths(root: Optional[TreeNode]) -> int:
    def calculate_node_depths(node, depth):
        if node is None:
            return 0
        return depth + calculate_node_depths(node.left, depth + 1) + calculate_node_depths(node.right, depth + 1)
    return calculate_node_depths(root, 0)`,
		Explanation: `This problem requires summing the depths of all nodes in a binary tree. The depth of a node is its distance from the root. We need to visit every node and accumulate its depth.

The algorithm uses recursive DFS with a depth parameter that tracks the current depth as we traverse. At each node, we add the current depth to our total, then recursively calculate and add the depths of the left and right subtrees (with depth incremented by 1). The base case is when we reach a nil node, which contributes 0 to the sum.

This approach efficiently calculates the sum by leveraging the recursive structure of the tree. Each node's depth is calculated exactly once, and we accumulate the sum as we traverse.

Edge cases handled include: single-node trees (depth 0, sum is 0), trees where all nodes are on one side (depths increase linearly), and empty trees (sum is 0).

An alternative iterative approach would use level-order traversal (BFS) with a queue, tracking depth for each level, but the recursive DFS approach is more straightforward and has the same O(n) time complexity. The space complexity is O(h) for the recursion stack, where h is the height of the tree.

Think of this as measuring the distance from the root to each node and adding up all those distances. The recursive approach naturally handles this by passing depth information down the tree.`,
	},
	{
		ID:          40,
		Title:       "Evaluate Expression Tree",
		Description: "You're given a binary expression tree. Write a function to evaluate this tree mathematically and return a single resulting integer. All leaf nodes in the tree represent operands, which will always be positive integers. All of the other nodes represent operators. There are 4 operators supported: + (addition), - (subtraction), * (multiplication), and / (division).",
		Difficulty:  "Easy",
		Topic:       "Binary Trees",
		Signature:   "func evaluateExpressionTree(tree *TreeNode) int",
		TestCases: []TestCase{
			{Input: "tree = [-1, -2, -3, -4, 2, 8, 3, 2]", Expected: "6"},
			{Input: "tree = [-2, 3, 2]", Expected: "5"},
			{Input: "tree = [-4, 2, 3]", Expected: "6"},
			{Input: "tree = [2]", Expected: "2"},
		},
		Solution: `func evaluateExpressionTree(tree *TreeNode) int {
	if tree.Left == nil && tree.Right == nil {
		return tree.Val
	}
	leftVal := evaluateExpressionTree(tree.Left)
	rightVal := evaluateExpressionTree(tree.Right)
	switch tree.Val {
	case -1: // addition
		return leftVal + rightVal
	case -2: // subtraction
		return leftVal - rightVal
	case -3: // division
		return leftVal / rightVal
	case -4: // multiplication
		return leftVal * rightVal
	default:
		return tree.Val
	}
}`,
		PythonSolution: `def evaluateExpressionTree(tree: Optional[TreeNode]) -> int:
    if tree.left is None and tree.right is None:
        return tree.val
    left_val = evaluateExpressionTree(tree.left)
    right_val = evaluateExpressionTree(tree.right)
    if tree.val == -1:  # addition
        return left_val + right_val
    elif tree.val == -2:  # subtraction
        return left_val - right_val
    elif tree.val == -3:  # division
        return left_val // right_val
    elif tree.val == -4:  # multiplication
        return left_val * right_val
    return tree.val`,
		Explanation: `Evaluating an expression tree requires a post-order traversal because we need to evaluate the operands (children) before we can apply the operator (parent). This is the natural order for expression evaluation.

The algorithm checks if the current node is a leaf node (no children). If it is, it's an operand, so we return its value directly. Otherwise, it's an operator node, so we recursively evaluate the left and right subtrees to get their values, then apply the operator to those values. The operators are encoded as negative numbers: -1 for addition, -2 for subtraction, -3 for division, and -4 for multiplication.

This post-order approach ensures that when we evaluate an operator node, both of its children have already been evaluated to their final integer values. We can then apply the operator and return the result up the recursion stack.

Edge cases handled include: single-node trees (just return the operand value), trees representing simple expressions like "3 + 2", and trees representing complex nested expressions with multiple operators.

An alternative approach would use an iterative post-order traversal with an explicit stack, but the recursive approach is more intuitive and has the same time complexity. Both require O(n) time to visit all nodes and O(h) space for the recursion stack.

Think of this as evaluating a mathematical expression written as a tree: you must calculate the values of sub-expressions (children) before you can calculate the value of the parent expression.`,
	},
	{
		ID:          41,
		Title:       "Depth-first Search",
		Description: "You're given a Node class that has a name and an array of optional children nodes. When put together, nodes form an acyclic tree-like structure. Implement the depthFirstSearch method on the Node class, which takes in an empty array, traverses the tree using the Depth-first Search approach (specifically navigating the tree from left to right), stores all of the nodes' names in the input array, and returns it.",
		Difficulty:  "Easy",
		Topic:       "Graphs",
		Signature:   "func (n *Node) DepthFirstSearch(array []string) []string",
		TestCases: []TestCase{
			{Input: "graph = A -> [B, C, D], B -> [E, F], D -> [G, H], F -> [I, J], G -> [K]", Expected: "[\"A\", \"B\", \"E\", \"F\", \"I\", \"J\", \"C\", \"D\", \"G\", \"K\", \"H\"]"},
			{Input: "graph = A", Expected: "[\"A\"]"},
		},
		Solution: `type Node struct {
	Name     string
	Children []*Node
}

func (n *Node) DepthFirstSearch(array []string) []string {
	array = append(array, n.Name)
	for _, child := range n.Children {
		array = child.DepthFirstSearch(array)
	}
	return array
}`,
		PythonSolution: `class Node:
    def __init__(self, name):
        self.children = []
        self.name = name
    
    def depthFirstSearch(self, array):
        array.append(self.name)
        for child in self.children:
            child.depthFirstSearch(array)
        return array`,
		Explanation: `Depth-first search (DFS) on a graph (represented as a tree-like structure) requires visiting each node and exploring as deep as possible before backtracking. The recursive approach naturally implements this traversal pattern.

The algorithm adds the current node's name to the result array at the beginning of the recursive call, implementing a pre-order traversal pattern. Then, it recursively calls DFS on all children of the current node. The children are processed in order (left to right as specified), ensuring the traversal follows the specified direction.

This approach visits each node exactly once and explores each edge exactly once (in a tree structure, each edge is traversed once). The recursive nature handles backtracking automatically, allowing us to explore the entire graph structure.

Edge cases handled include: graphs with a single node (that node is added to result), graphs with no children (node is added, then recursion stops), and graphs with varying depths (DFS naturally handles all depths).

An alternative iterative approach would use an explicit stack to track nodes to visit, but the recursive approach is more intuitive and has the same time complexity O(v+e) where v is vertices and e is edges. The space complexity is O(v) for the recursion stack in the worst case of a linear graph.

Think of this as exploring a maze: you mark your current location, then try each path (child) one by one, going as deep as possible down each path before trying the next one.`,
	},
	{
		ID:          42,
		Title:       "Minimum Waiting Time",
		Description: "You're given a non-empty array of positive integers representing the amounts of time that specific queries take to execute. Only one query can be executed at a time, but the queries can be executed in any order. A query's waiting time is defined as the amount of time that it must wait before its execution starts. In other words, if a query is executed second, then its waiting time is the duration of the first query; if a query is executed third, then its waiting time is the duration of the first two queries. Write a function that returns the minimum amount of total waiting time for all of the queries.",
		Difficulty:  "Easy",
		Topic:       "Greedy",
		Signature:   "func minimumWaitingTime(queries []int) int",
		TestCases: []TestCase{
			{Input: "queries = []int{3, 2, 1, 2, 6}", Expected: "17"},
			{Input: "queries = []int{2, 1, 1, 1}", Expected: "6"},
			{Input: "queries = []int{1, 2, 4, 5, 2, 1}", Expected: "23"},
			{Input: "queries = []int{25, 30, 2, 1}", Expected: "32"},
			{Input: "queries = []int{1}", Expected: "0"},
		},
		Solution: `import "sort"

func minimumWaitingTime(queries []int) int {
	sort.Ints(queries)
	totalWaitingTime := 0
	for i, duration := range queries {
		queriesLeft := len(queries) - (i + 1)
		totalWaitingTime += duration * queriesLeft
	}
	return totalWaitingTime
}`,
		PythonSolution: `def minimumWaitingTime(queries: List[int]) -> int:
    queries.sort()
    total_waiting_time = 0
    for i, duration in enumerate(queries):
        queries_left = len(queries) - (i + 1)
        total_waiting_time += duration * queries_left
    return total_waiting_time`,
		Explanation: `This is a classic greedy algorithm problem. The key insight is that to minimize total waiting time, we should execute shorter queries first. This ensures that queries with longer wait times are minimized, as they wait for fewer preceding queries.

The algorithm first sorts the queries by their duration in ascending order. Then, it iterates through the sorted queries, calculating the waiting time contribution of each query. For a query at position i, it will wait for all queries executed before it. Since there are (n - i - 1) queries executed after it, the waiting time contributed by this query is its duration multiplied by the number of queries that will wait for it.

This greedy approach is optimal because if we were to execute a longer query before a shorter one, we would increase the waiting time for the shorter query (which affects more subsequent queries) more than we would decrease the waiting time for the longer query (which affects fewer subsequent queries).

Edge cases handled include: single query (waiting time is 0, as there are no queries waiting for it), queries with equal durations (order doesn't matter), and queries already sorted (sorting still needed for correctness).

An alternative approach would be to try all permutations, but that would be O(n!) time. The greedy sorting approach is optimal with O(n log n) time for sorting and O(n) time for calculation, giving overall O(n log n) time complexity.

Think of this as a queue at a service counter: if you serve the quickest customers first, the total waiting time for everyone is minimized because fewer people wait for longer periods.`,
	},
	{
		ID:          43,
		Title:       "Class Photos",
		Description: "It's photo day at the local school, and you're the photographer assigned to take class photos. The class that you're photographing has an even number of students, and all these students are wearing red or blue shirts. In fact, exactly half of the class is wearing red shirts, and the other half is wearing blue shirts. You're responsible for arranging the students in two rows before taking the photo. Each row should contain the same number of students and should adhere to the following guidelines: All students wearing red shirts must be in the same row. All students wearing blue shirts must be in the same row. Each student in the back row must be strictly taller than the student directly in front of them in the front row. Write a function that returns whether or not a class photo that follows the stated guidelines can be taken.",
		Difficulty:  "Easy",
		Topic:       "Greedy",
		Signature:   "func classPhotos(redShirtHeights []int, blueShirtHeights []int) bool",
		TestCases: []TestCase{
			{Input: "redShirtHeights = []int{5, 8, 1, 3, 4}, blueShirtHeights = []int{6, 9, 2, 4, 5}", Expected: "true"},
			{Input: "redShirtHeights = []int{6, 9, 2, 4, 5}, blueShirtHeights = []int{5, 8, 1, 3, 4}", Expected: "true"},
			{Input: "redShirtHeights = []int{5, 8, 1, 3, 4}, blueShirtHeights = []int{5, 8, 1, 3, 4}", Expected: "false"},
			{Input: "redShirtHeights = []int{6}, blueShirtHeights = []int{6}", Expected: "false"},
		},
		Solution: `import "sort"

func classPhotos(redShirtHeights []int, blueShirtHeights []int) bool {
	sort.Sort(sort.Reverse(sort.IntSlice(redShirtHeights)))
	sort.Sort(sort.Reverse(sort.IntSlice(blueShirtHeights)))
	shirtColorInFirstRow := "BLUE"
	if redShirtHeights[0] < blueShirtHeights[0] {
		shirtColorInFirstRow = "RED"
	}
	for i := 0; i < len(redShirtHeights); i++ {
		redHeight := redShirtHeights[i]
		blueHeight := blueShirtHeights[i]
		if shirtColorInFirstRow == "RED" {
			if redHeight >= blueHeight {
				return false
			}
		} else {
			if blueHeight >= redHeight {
				return false
			}
		}
	}
	return true
}`,
		PythonSolution: `def classPhotos(redShirtHeights: List[int], blueShirtHeights: List[int]) -> bool:
    redShirtHeights.sort(reverse=True)
    blueShirtHeights.sort(reverse=True)
    shirt_color_in_first_row = "BLUE"
    if redShirtHeights[0] < blueShirtHeights[0]:
        shirt_color_in_first_row = "RED"
    for i in range(len(redShirtHeights)):
        red_height = redShirtHeights[i]
        blue_height = blueShirtHeights[i]
        if shirt_color_in_first_row == "RED":
            if red_height >= blue_height:
                return False
        else:
            if blue_height >= red_height:
                return False
    return True`,
		Explanation: `This problem requires arranging students in two rows such that each student in the back row is taller than the student directly in front. The key insight is that we should pair the tallest students together, with the shorter group in front.

The algorithm first sorts both arrays in descending order. Then, it determines which color should be in the front row by comparing the tallest student from each group - the group with the shorter tallest student should be in front. This ensures we can potentially satisfy the constraint. Then, it checks if for each position, the student in the back row (from the taller group) is strictly taller than the student in the front row (from the shorter group) at the same position.

This greedy approach works because by sorting and pairing students at the same rank (tallest with tallest, second tallest with second tallest, etc.), we maximize the height differences, giving us the best chance of satisfying the constraint. If the tallest student from one group is shorter than or equal to the tallest from the other, we can't arrange them (one group must be entirely in front).

Edge cases handled include: students with equal heights (returns false, as back row must be strictly taller), single student per group (must have different heights), and groups where all students from one group are taller than all from the other (arrangement is possible).

An alternative approach would be to try all possible arrangements, but that would be exponential. The greedy sorting approach is optimal with O(n log n) time for sorting and O(n) time for checking, giving overall O(n log n) time complexity.

Think of this as lining up two teams by height: you want the back team to be taller at every position, so you pair the tallest from each team, second tallest from each team, etc., and check if the back team wins every comparison.`,
	},
	{
		ID:          44,
		Title:       "Tandem Bicycle",
		Description: "A tandem bicycle is a bicycle that's operated by two people: person A and person B. Both people pedal the bicycle, but the person that pedals faster dictates the speed of the bicycle. So if person A pedals at a speed of 5, and person B pedals at a speed of 4, the tandem bicycle moves at a speed of 5. You're given two lists of positive integers: one that contains the speeds of riders wearing red shirts and one that contains the speeds of riders wearing blue shirts. Each rider is represented by a single positive integer, which is the speed that they pedal a tandem bicycle at. Both lists have the same length, meaning that there are as many red-shirt riders as there are blue-shirt riders. Your goal is to pair every red-shirt rider with a blue-shirt rider to operate a tandem bicycle. Write a function that returns the maximum possible total speed or the minimum possible total speed of all of the tandem bicycles being ridden based on an input parameter, fastest. If fastest = true, your function should return the maximum possible total speed; otherwise it should return the minimum total speed.",
		Difficulty:  "Easy",
		Topic:       "Greedy",
		Signature:   "func tandemBicycle(redShirtSpeeds []int, blueShirtSpeeds []int, fastest bool) int",
		TestCases: []TestCase{
			{Input: "redShirtSpeeds = []int{5, 5, 3, 9, 2}, blueShirtSpeeds = []int{3, 6, 7, 2, 1}, fastest = true", Expected: "32"},
			{Input: "redShirtSpeeds = []int{5, 5, 3, 9, 2}, blueShirtSpeeds = []int{3, 6, 7, 2, 1}, fastest = false", Expected: "25"},
			{Input: "redShirtSpeeds = []int{1, 2, 1, 9, 12, 3}, blueShirtSpeeds = []int{3, 3, 4, 6, 1, 2}, fastest = false", Expected: "30"},
			{Input: "redShirtSpeeds = []int{1, 2, 1, 9, 12, 3}, blueShirtSpeeds = []int{3, 3, 4, 6, 1, 2}, fastest = true", Expected: "37"},
		},
		Solution: `import "sort"

func tandemBicycle(redShirtSpeeds []int, blueShirtSpeeds []int, fastest bool) int {
	sort.Ints(redShirtSpeeds)
	sort.Ints(blueShirtSpeeds)
	totalSpeed := 0
	if fastest {
		for i := 0; i < len(redShirtSpeeds); i++ {
			totalSpeed += max(redShirtSpeeds[i], blueShirtSpeeds[len(blueShirtSpeeds)-1-i])
		}
	} else {
		for i := 0; i < len(redShirtSpeeds); i++ {
			totalSpeed += max(redShirtSpeeds[i], blueShirtSpeeds[i])
		}
	}
	return totalSpeed
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}`,
		PythonSolution: `def tandemBicycle(redShirtSpeeds: List[int], blueShirtSpeeds: List[int], fastest: bool) -> int:
    redShirtSpeeds.sort()
    blueShirtSpeeds.sort()
    total_speed = 0
    if fastest:
        for i in range(len(redShirtSpeeds)):
            total_speed += max(redShirtSpeeds[i], blueShirtSpeeds[len(blueShirtSpeeds) - 1 - i])
    else:
        for i in range(len(redShirtSpeeds)):
            total_speed += max(redShirtSpeeds[i], blueShirtSpeeds[i])
    return total_speed`,
		Explanation: `This problem requires pairing riders to maximize or minimize total tandem bicycle speed. The key insight is that the speed of a tandem bicycle is determined by the faster rider, so we want to either maximize or minimize the sum of maximum speeds in each pair.

For maximum total speed, we want to create pairs with the largest possible speed differences - pairing the slowest red-shirt rider with the fastest blue-shirt rider, then the second slowest red with the second fastest blue, and so on. This ensures that each pair contributes a relatively high maximum speed. For minimum total speed, we pair riders with similar speeds - smallest red with smallest blue, second smallest red with second smallest blue, etc.

The algorithm sorts both arrays in ascending order. For maximum speed, it pairs elements from opposite ends (smallest red with largest blue). For minimum speed, it pairs elements from the same positions (smallest with smallest). The total speed is the sum of the maximum speed in each pair.

This greedy approach is optimal because it creates the most extreme pairings (for maximum) or most balanced pairings (for minimum), which directly affects the sum of maximum speeds in each pair.

Edge cases handled include: arrays with a single rider each (one tandem bicycle), arrays where all red riders are faster than all blue riders (pairings still work correctly), and arrays with equal speeds (total speed is the same regardless of pairing for minimum, but maximum benefits from extreme pairings).

An alternative approach would be to try all possible pairings, but that would be factorial time complexity. The greedy sorting approach is optimal with O(n log n) time for sorting and O(n) time for pairing, giving overall O(n log n) time complexity.

Think of this as matching partners for a race: to maximize total speed, pair the slowest with the fastest (so the fast person's speed counts); to minimize, pair similar speeds together (so you're not wasting fast riders' potential).`,
	},
	{
		ID:          45,
		Title:       "Optimal Freelancing",
		Description: "You recently started freelance software development and have been offered a variety of job opportunities. Each job has a deadline, meaning there is no value in completing the job after the deadline. Additionally, each job has an associated payment representing the profit for completing that job. Given this information, write a function that returns the maximum profit that can be obtained in a 7-day period. Each job will take 1 full day to complete, and the deadline will be given as the number of days left to complete the job. For example, if a job has a deadline of 1, then it can only be completed on day 1 of the week; if a job has a deadline of 2, then it can be completed on either day 1 or day 2 of the week.",
		Difficulty:  "Easy",
		Topic:       "Greedy",
		Signature:   "func optimalFreelancing(jobs []map[string]int) int",
		TestCases: []TestCase{
			{Input: "jobs = [{\"deadline\": 1, \"payment\": 1}, {\"deadline\": 2, \"payment\": 2}, {\"deadline\": 2, \"payment\": 1}]", Expected: "3"},
			{Input: "jobs = [{\"deadline\": 2, \"payment\": 1}, {\"deadline\": 2, \"payment\": 2}, {\"deadline\": 2, \"payment\": 3}]", Expected: "5"},
			{Input: "jobs = [{\"deadline\": 1, \"payment\": 1}, {\"deadline\": 2, \"payment\": 2}, {\"deadline\": 3, \"payment\": 3}, {\"deadline\": 4, \"payment\": 4}, {\"deadline\": 5, \"payment\": 5}, {\"deadline\": 6, \"payment\": 6}, {\"deadline\": 7, \"payment\": 7}, {\"deadline\": 1, \"payment\": 8}]", Expected: "33"},
		},
		Solution: `import "sort"

type Job struct {
	Deadline int
	Payment  int
}

func optimalFreelancing(jobs []map[string]int) int {
	jobList := make([]Job, len(jobs))
	for i, job := range jobs {
		jobList[i] = Job{Deadline: job["deadline"], Payment: job["payment"]}
	}
	sort.Slice(jobList, func(i, j int) bool {
		return jobList[i].Payment > jobList[j].Payment
	})
	timeline := make([]bool, 8)
	profit := 0
	for _, job := range jobList {
		time := job.Deadline
		if time > 7 {
			time = 7
		}
		for time > 0 {
			if !timeline[time] {
				timeline[time] = true
				profit += job.Payment
				break
			}
			time--
		}
	}
	return profit
}`,
		PythonSolution: `def optimalFreelancing(jobs: List[dict]) -> int:
    jobs.sort(key=lambda x: x["payment"], reverse=True)
    timeline = [False] * 8
    profit = 0
    for job in jobs:
        time = min(job["deadline"], 7)
        while time > 0:
            if not timeline[time]:
                timeline[time] = True
                profit += job["payment"]
                break
            time -= 1
    return profit`,
		Explanation: `This is a scheduling problem that can be solved using a greedy algorithm. The key insight is that we should schedule the highest-paying jobs as late as possible within their deadline, leaving earlier time slots available for other jobs.

The algorithm first sorts all jobs by payment in descending order, ensuring we consider the most valuable jobs first. Then, for each job, it tries to schedule it on the latest possible day within its deadline (or day 7, whichever is earlier). It uses a boolean array (timeline) to track which days are already scheduled. If the preferred day is taken, it tries the previous day, continuing backward until it finds an available day or runs out of days.

This greedy approach works because by scheduling high-paying jobs late, we preserve earlier time slots for jobs with earlier deadlines. If we can't schedule a job within its deadline, we skip it. This maximizes the total profit we can achieve.

Edge cases handled include: jobs with deadlines beyond 7 days (capped at day 7), jobs that cannot be scheduled within their deadline (skipped), multiple jobs wanting the same day (scheduled in payment order, with higher-paying jobs getting preference for later days), and jobs with deadline 1 (must be scheduled on day 1 or skipped).

An alternative approach would be to use dynamic programming to consider all possible schedules, but that would be exponential. The greedy approach is more efficient, with O(n log n) time for sorting and O(n*d) time for scheduling, where d is the number of days (7).

Think of this as filling a calendar: you want to put the most valuable appointments as late as possible, working backward from their deadlines, so you leave room for other appointments earlier.`,
	},
	{
		ID:          46,
		Title:       "Middle Node",
		Description: "You're given a Linked List with at least one node. Write a function that returns the middle node of the Linked List. If there are two middle nodes (i.e. an even length list), your function should return the second of these nodes.",
		Difficulty:  "Easy",
		Topic:       "Linked Lists",
		Signature:   "func middleNode(head *ListNode) *ListNode",
		TestCases: []TestCase{
			{Input: "head = [2, 7, 3, 5]", Expected: "[3 5]"},
			{Input: "head = [1, 2, 3, 4, 5]", Expected: "[3 4 5]"},
			{Input: "head = [1]", Expected: "[1]"},
			{Input: "head = [1, 2]", Expected: "[2]"},
		},
		Solution: `func middleNode(head *ListNode) *ListNode {
	slow := head
	fast := head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	return slow
}`,
		PythonSolution: `def middleNode(head: Optional[ListNode]) -> Optional[ListNode]:
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow`,
		Explanation: `Finding the middle node of a linked list can be done efficiently using the "tortoise and hare" two-pointer technique. The key insight is that if one pointer moves twice as fast as the other, when the fast pointer reaches the end, the slow pointer will be at the middle.

The algorithm uses two pointers, both starting at the head. The slow pointer moves one step at a time, while the fast pointer moves two steps at a time. When the fast pointer reaches the end of the list (becomes nil or its next is nil), the slow pointer will be at the middle node. For even-length lists, this finds the second middle node (as specified in the problem).

This approach is optimal because it requires only a single pass through the list with O(1) extra space. We don't need to know the length of the list beforehand or traverse it multiple times.

Edge cases handled include: single-node lists (returns that node), two-node lists (returns the second node), odd-length lists (returns the true middle), and even-length lists (returns the second of the two middle nodes).

An alternative approach would be to traverse the list once to count nodes, then traverse again to the middle position, but that requires two passes. Another approach would be to use an array to store all nodes, but that requires O(n) extra space. The two-pointer approach is optimal for both time and space.

Think of this as two people walking: one walks at normal speed, the other runs at double speed. When the runner finishes the race, the walker is exactly halfway through.`,
	},
	{
		ID:          47,
		Title:       "Product Sum",
		Description: "Write a function that takes in a \"special\" array and returns its product sum. A \"special\" array is a non-empty array that contains either integers or other \"special\" arrays. The product sum of a \"special\" array is the sum of its elements, where \"special\" arrays inside it are summed themselves and then multiplied by their level of depth.",
		Difficulty:  "Easy",
		Topic:       "Recursion",
		Signature:   "func productSum(array []interface{}, multiplier int) int",
		TestCases: []TestCase{
			{Input: "array = [5, 2, [7, -1], 3, [6, [-13, 8], 4]]", Expected: "12"},
			{Input: "array = [1, 2, 3, 4, 5]", Expected: "15"},
			{Input: "array = [1, 2, [3], 4, 5]", Expected: "18"},
			{Input: "array = [[1, 2], 3, [4, 5]]", Expected: "27"},
		},
		Solution: `func productSum(array []interface{}, multiplier int) int {
	sum := 0
	for _, element := range array {
		if num, ok := element.(int); ok {
			sum += num
		} else if subArray, ok := element.([]interface{}); ok {
			sum += productSum(subArray, multiplier+1) * (multiplier + 1)
		}
	}
	return sum * multiplier
}`,
		PythonSolution: `def productSum(array: List[Union[int, List]], multiplier: int = 1) -> int:
    sum_val = 0
    for element in array:
        if isinstance(element, int):
            sum_val += element
        else:
            sum_val += productSum(element, multiplier + 1) * (multiplier + 1)
    return sum_val * multiplier`,
		Explanation: `This problem requires calculating a weighted sum where nested arrays contribute more to the total. The "product sum" means that elements at deeper nesting levels are multiplied by their depth level before being added to the sum.

The algorithm uses recursion to handle the nested structure. For each element, if it's an integer, we add it to the current sum multiplied by the current multiplier (depth level). If it's a nested array, we recursively calculate its product sum with an increased multiplier (depth + 1), then multiply that result by the current multiplier and add it to the sum.

The multiplier represents the depth level: the root level has multiplier 1, the first nested level has multiplier 2, and so on. This means that elements at deeper levels contribute more to the final sum, which is the "product" aspect of the problem.

Edge cases handled include: arrays with no nesting (all integers at level 1), arrays with multiple levels of nesting, arrays with empty nested arrays (contribute 0), and single-element arrays (that element multiplied by its depth).

An alternative iterative approach would use an explicit stack to track depth levels, but the recursive approach is more intuitive and naturally handles the nested structure. Both approaches have O(n) time complexity where n is the total number of elements (including nested ones), and O(d) space complexity where d is the maximum depth.

Think of this as calculating a weighted sum where items in boxes (nested arrays) are worth more than items outside boxes, and items in boxes within boxes are worth even more.`,
	},
	{
		ID:          48,
		Title:       "Find Three Largest Numbers",
		Description: "Write a function that takes in an array of at least three integers and, without sorting the input array, returns a sorted array of the three largest integers in the input array.",
		Difficulty:  "Easy",
		Topic:       "Arrays",
		Signature:   "func findThreeLargestNumbers(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = []int{141, 1, 17, -7, -17, -27, 18, 541, 8, 7, 7}", Expected: "[18 141 541]"},
			{Input: "array = []int{55, 7, 8}", Expected: "[7 8 55]"},
			{Input: "array = []int{55, 43, 11, 3, -3, 10}", Expected: "[11 43 55]"},
			{Input: "array = []int{7, 8, 3, 11, 43, 55}", Expected: "[11 43 55]"},
		},
		Solution: `func findThreeLargestNumbers(array []int) []int {
	threeLargest := []int{math.MinInt32, math.MinInt32, math.MinInt32}
	for _, num := range array {
		updateLargest(threeLargest, num)
	}
	return threeLargest
}

func updateLargest(threeLargest []int, num int) {
	if num > threeLargest[2] {
		shiftAndUpdate(threeLargest, num, 2)
	} else if num > threeLargest[1] {
		shiftAndUpdate(threeLargest, num, 1)
	} else if num > threeLargest[0] {
		shiftAndUpdate(threeLargest, num, 0)
	}
}

func shiftAndUpdate(array []int, num int, idx int) {
	for i := 0; i <= idx; i++ {
		if i == idx {
			array[i] = num
		} else {
			array[i] = array[i+1]
		}
	}
}`,
		PythonSolution: `def findThreeLargestNumbers(array: List[int]) -> List[int]:
    three_largest = [float('-inf'), float('-inf'), float('-inf')]
    for num in array:
        update_largest(three_largest, num)
    return three_largest

def update_largest(three_largest, num):
    if num > three_largest[2]:
        shift_and_update(three_largest, num, 2)
    elif num > three_largest[1]:
        shift_and_update(three_largest, num, 1)
    elif num > three_largest[0]:
        shift_and_update(three_largest, num, 0)

def shift_and_update(array, num, idx):
    for i in range(idx + 1):
        if i == idx:
            array[i] = num
        else:
            array[i] = array[i + 1]`,
		Explanation: `Finding the three largest numbers without sorting the entire array requires maintaining a running list of the top three candidates. The key insight is that we only need to track three values and update them as we encounter larger numbers.

The algorithm maintains an array of three elements initialized to negative infinity (or minimum integer values). As we iterate through the input array, for each number, we check if it's larger than any of our three candidates. If it's larger than the largest, we shift all three values left and insert the new number at the end. If it's larger than the second largest, we shift the first two and insert. If it's only larger than the smallest, we replace the smallest.

The shift-and-update operation ensures we maintain the three largest values in sorted order. By checking from largest to smallest, we can efficiently determine where to insert a new value and shift the appropriate elements.

Edge cases handled include: arrays with exactly three elements (those are the three largest), arrays with duplicate values (handled correctly by comparison logic), arrays with negative numbers (initialized with minimum values), and arrays where the three largest appear at the beginning (will be replaced by even larger values if found).

An alternative approach would be to sort the entire array and take the last three elements, but that requires O(n log n) time. This approach achieves O(n) time with O(1) space, making it more efficient for large arrays when we only need the top three.

Think of this as maintaining a podium with three positions: as athletes finish, you check if their score beats any of the top three, and if so, you shift the rankings and add the new athlete to the appropriate position.`,
	},
	{
		ID:          49,
		Title:       "Bubble Sort",
		Description: "Write a function that takes in an array of integers and returns a sorted version of that array. Use the Bubble Sort algorithm to sort the array.",
		Difficulty:  "Easy",
		Topic:       "Sorting",
		Signature:   "func bubbleSort(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = []int{8, 5, 2, 9, 5, 6, 3}", Expected: "[2 3 5 5 6 8 9]"},
			{Input: "array = []int{1}", Expected: "[1]"},
			{Input: "array = []int{1, 2, 3}", Expected: "[1 2 3]"},
			{Input: "array = []int{3, 2, 1}", Expected: "[1 2 3]"},
		},
		Solution: `func bubbleSort(array []int) []int {
	isSorted := false
	counter := 0
	for !isSorted {
		isSorted = true
		for i := 0; i < len(array)-1-counter; i++ {
			if array[i] > array[i+1] {
				array[i], array[i+1] = array[i+1], array[i]
				isSorted = false
			}
		}
		counter++
	}
	return array
}`,
		PythonSolution: `def bubbleSort(array: List[int]) -> List[int]:
    is_sorted = False
    counter = 0
    while not is_sorted:
        is_sorted = True
        for i in range(len(array) - 1 - counter):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                is_sorted = False
        counter += 1
    return array`,
		Explanation: `Bubble sort is a simple sorting algorithm that works by repeatedly stepping through the array, comparing adjacent elements and swapping them if they're in the wrong order. The algorithm gets its name because smaller elements "bubble" to the top of the array.

The algorithm uses nested loops: the outer loop controls how many passes we make, and the inner loop compares adjacent elements. In each pass, we compare each pair of adjacent elements and swap them if they're in the wrong order. An optimization is to reduce the inner loop's range by the number of passes completed, since the largest unsorted element will have "bubbled" to its correct position after each pass.

The algorithm uses a flag (isSorted) to detect early termination: if no swaps occur during a pass, the array is already sorted, and we can stop early. This optimization improves the best-case time complexity to O(n) when the array is already sorted.

Edge cases handled include: already-sorted arrays (terminates early after one pass), reverse-sorted arrays (requires maximum number of passes), arrays with duplicate values (handled correctly by comparison), and single-element arrays (already sorted).

An alternative approach would be to use more efficient sorting algorithms like quicksort or mergesort, which achieve O(n log n) average time complexity. However, bubble sort is simple to understand and implement, and is useful for educational purposes or very small arrays.

Think of this as bubbles rising in a liquid: lighter (smaller) elements gradually move toward the top through a series of swaps with heavier (larger) elements.`,
	},
	{
		ID:          50,
		Title:       "Insertion Sort",
		Description: "Write a function that takes in an array of integers and returns a sorted version of that array. Use the Insertion Sort algorithm to sort the array.",
		Difficulty:  "Easy",
		Topic:       "Sorting",
		Signature:   "func insertionSort(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = []int{8, 5, 2, 9, 5, 6, 3}", Expected: "[2 3 5 5 6 8 9]"},
			{Input: "array = []int{1}", Expected: "[1]"},
			{Input: "array = []int{1, 2, 3}", Expected: "[1 2 3]"},
			{Input: "array = []int{3, 2, 1}", Expected: "[1 2 3]"},
		},
		Solution: `func insertionSort(array []int) []int {
	for i := 1; i < len(array); i++ {
		j := i
		for j > 0 && array[j] < array[j-1] {
			array[j], array[j-1] = array[j-1], array[j]
			j--
		}
	}
	return array
}`,
		PythonSolution: `def insertionSort(array: List[int]) -> List[int]:
    for i in range(1, len(array)):
        j = i
        while j > 0 and array[j] < array[j - 1]:
            array[j], array[j - 1] = array[j - 1], array[j]
            j -= 1
    return array`,
		Explanation: `Insertion sort builds a sorted array one element at a time by inserting each new element into its correct position within the already-sorted portion. It's similar to how you might sort playing cards in your hand.

The algorithm maintains two portions of the array: a sorted portion at the beginning and an unsorted portion. For each element in the unsorted portion, it finds the correct position in the sorted portion by comparing it with elements from right to left, shifting larger elements one position to the right to make room, then inserting the element in its correct position.

This approach is efficient for small arrays or arrays that are already mostly sorted, as it has O(n) best-case time complexity when the array is already sorted. The algorithm is stable (maintains the relative order of equal elements) and in-place (only uses O(1) extra space).

Edge cases handled include: already-sorted arrays (each element is already in the correct position, O(n) time), reverse-sorted arrays (each element must be moved to the beginning, O(n²) time), arrays with duplicate values (maintains stability), and single-element arrays (already sorted).

An alternative approach would be to use binary search to find the insertion position, reducing comparisons to O(n log n), but the shifting still requires O(n²) time overall. For small arrays, the simple insertion sort is often faster due to lower overhead.

Think of this as organizing a hand of cards: you take each card and insert it into the correct position among the cards you've already organized, shifting other cards as needed to make room.`,
	},
	{
		ID:          51,
		Title:       "Selection Sort",
		Description: "Write a function that takes in an array of integers and returns a sorted version of that array. Use the Selection Sort algorithm to sort the array.",
		Difficulty:  "Easy",
		Topic:       "Sorting",
		Signature:   "func selectionSort(array []int) []int",
		TestCases: []TestCase{
			{Input: "array = []int{8, 5, 2, 9, 5, 6, 3}", Expected: "[2 3 5 5 6 8 9]"},
			{Input: "array = []int{1}", Expected: "[1]"},
			{Input: "array = []int{1, 2, 3}", Expected: "[1 2 3]"},
			{Input: "array = []int{3, 2, 1}", Expected: "[1 2 3]"},
		},
		Solution: `func selectionSort(array []int) []int {
	for i := 0; i < len(array)-1; i++ {
		smallestIdx := i
		for j := i + 1; j < len(array); j++ {
			if array[j] < array[smallestIdx] {
				smallestIdx = j
			}
		}
		array[i], array[smallestIdx] = array[smallestIdx], array[i]
	}
	return array
}`,
		PythonSolution: `def selectionSort(array: List[int]) -> List[int]:
    for i in range(len(array) - 1):
        smallest_idx = i
        for j in range(i + 1, len(array)):
            if array[j] < array[smallest_idx]:
                smallest_idx = j
        array[i], array[smallest_idx] = array[smallest_idx], array[i]
    return array`,
		Explanation: `Selection sort works by repeatedly finding the minimum element from the unsorted portion of the array and placing it at the beginning of the unsorted portion. It maintains two subarrays: one that is sorted (at the beginning) and one that is unsorted (the remainder).

The algorithm uses nested loops: the outer loop tracks the boundary between sorted and unsorted portions, and the inner loop finds the minimum element in the unsorted portion. Once found, it swaps this minimum element with the first element of the unsorted portion, effectively extending the sorted portion by one element.

This approach is simple and intuitive but has O(n²) time complexity in all cases (best, average, and worst), as it always performs the same number of comparisons regardless of the initial order. However, it minimizes the number of swaps (at most n-1 swaps), which can be advantageous when writes are expensive.

Edge cases handled include: already-sorted arrays (still performs all comparisons but no swaps needed), reverse-sorted arrays (performs maximum swaps), arrays with duplicate values (selects the first occurrence of the minimum), and single-element arrays (already sorted).

An alternative approach would be to use more efficient sorting algorithms, but selection sort is useful for its simplicity and the fact that it performs a minimal number of swaps. It's also useful when memory writes are expensive compared to reads.

Think of this as finding the smallest item in a pile, putting it aside, then finding the next smallest in the remaining pile, and so on, until you've sorted everything.`,
	},
	{
		ID:          52,
		Title:       "Caesar Cipher Encryptor",
		Description: "Given a non-empty string of lowercase letters and a non-negative integer representing a key, write a function that returns a new string obtained by shifting every letter in the input string by k positions in the alphabet, where k is the key.",
		Difficulty:  "Easy",
		Topic:       "Strings",
		Signature:   "func caesarCipherEncryptor(str string, key int) string",
		TestCases: []TestCase{
			{Input: "str = \"xyz\", key = 2", Expected: "\"zab\""},
			{Input: "str = \"abc\", key = 0", Expected: "\"abc\""},
			{Input: "str = \"abc\", key = 3", Expected: "\"def\""},
			{Input: "str = \"xyz\", key = 5", Expected: "\"cde\""},
			{Input: "str = \"abc\", key = 26", Expected: "\"abc\""},
		},
		Solution: `func caesarCipherEncryptor(str string, key int) string {
	result := make([]byte, len(str))
	key = key % 26
	for i, char := range str {
		newLetterCode := int(char) + key
		if newLetterCode > int('z') {
			newLetterCode = int('a') + (newLetterCode - int('z') - 1)
		}
		result[i] = byte(newLetterCode)
	}
	return string(result)
}`,
		PythonSolution: `def caesarCipherEncryptor(string: str, key: int) -> str:
    result = []
    key = key % 26
    for char in string:
        new_letter_code = ord(char) + key
        if new_letter_code > ord('z'):
            new_letter_code = ord('a') + (new_letter_code - ord('z') - 1)
        result.append(chr(new_letter_code))
    return ''.join(result)`,
		Explanation: `Caesar cipher is a substitution cipher where each letter is shifted by a fixed number of positions in the alphabet. The key challenge is handling wrap-around when the shift goes beyond 'z' or before 'a'.

The algorithm first normalizes the key using modulo 26, since shifting by 26 positions (or multiples) results in the same letter. Then, for each character in the string, it calculates the new character code by adding the key. If the new code exceeds 'z', it wraps around to 'a' by calculating the offset from 'z' and adding it to 'a'.

This approach handles the circular nature of the alphabet efficiently. The modulo operation on the key ensures we only need to handle shifts in the range 0-25, and the wrap-around logic ensures that shifts beyond 'z' correctly wrap to the beginning of the alphabet.

Edge cases handled include: key of 0 (no shift, original string), key equal to 26 (same as key of 0 after modulo), key larger than 26 (normalized by modulo), shifts that wrap multiple times (handled by the wrap-around calculation), and strings with all 'z' characters (wrap to 'a' and beyond correctly).

An alternative approach would be to create a mapping array, but the direct calculation approach is more space-efficient. Another approach would handle uppercase and lowercase separately, but this problem specifies lowercase only.

Think of the alphabet as a circle: if you shift 'z' by 2 positions, you go past 'z' and continue from 'a', landing on 'b'.`,
	},
	{
		ID:          53,
		Title:       "Run-Length Encoding",
		Description: "Write a function that takes in a non-empty string and returns its run-length encoding. From Wikipedia, \"run-length encoding is a form of lossless data compression in which runs of data are stored as a single data value and count, rather than as the original run.\" For this problem, a run of data is any sequence of consecutive, identical characters. So the run \"AAA\" would be run-length-encoded as \"3A\".",
		Difficulty:  "Easy",
		Topic:       "Strings",
		Signature:   "func runLengthEncoding(str string) string",
		TestCases: []TestCase{
			{Input: "str = \"AAAAAAAAAAAAABBCCCCDD\"", Expected: "\"9A4A2B4C2D\""},
			{Input: "str = \"aA\"", Expected: "\"1a1A\""},
			{Input: "str = \"122333\"", Expected: "\"112233\""},
			{Input: "str = \"************^^^^^^^$$$$$$%%%%%%%!!!!!!AAAAAAAAAAAAAAAAAAAA\"", Expected: "\"9*3*7^6$7%6!9A9A2A\""},
		},
		Solution: `func runLengthEncoding(str string) string {
	encodedStringCharacters := []byte{}
	currentRunLength := 1
	for i := 1; i < len(str); i++ {
		currentCharacter := str[i]
		previousCharacter := str[i-1]
		if currentCharacter != previousCharacter || currentRunLength == 9 {
			encodedStringCharacters = append(encodedStringCharacters, byte('0'+currentRunLength))
			encodedStringCharacters = append(encodedStringCharacters, previousCharacter)
			currentRunLength = 0
		}
		currentRunLength++
	}
	encodedStringCharacters = append(encodedStringCharacters, byte('0'+currentRunLength))
	encodedStringCharacters = append(encodedStringCharacters, str[len(str)-1])
	return string(encodedStringCharacters)
}`,
		PythonSolution: `def runLengthEncoding(string: str) -> str:
    encoded_string_characters = []
    current_run_length = 1
    for i in range(1, len(string)):
        current_character = string[i]
        previous_character = string[i - 1]
        if current_character != previous_character or current_run_length == 9:
            encoded_string_characters.append(str(current_run_length))
            encoded_string_characters.append(previous_character)
            current_run_length = 0
        current_run_length += 1
    encoded_string_characters.append(str(current_run_length))
    encoded_string_characters.append(string[len(string) - 1])
    return ''.join(encoded_string_characters)`,
		Explanation: `Run-length encoding compresses consecutive identical characters by storing the count followed by the character. The key challenge is handling runs longer than 9 characters, as we need to split them into multiple encoded segments (since we use single digits for counts).

The algorithm tracks the current run length as it iterates through the string. When it encounters a character different from the previous one, it encodes the previous run by appending the count and character. Additionally, when the run length reaches 9, it encodes the run even if the character hasn't changed, then resets the counter. This ensures no run is encoded with a count greater than 9. After the loop, it handles the last run separately since the loop doesn't process it.

This approach efficiently handles the constraint that counts are single digits. By splitting long runs at count 9, we ensure the encoding is valid while still achieving compression for most cases.

Edge cases handled include: strings with runs longer than 9 characters (split into multiple segments like "9A1A" for 10 A's), strings with no consecutive characters (each character encoded as "1X"), strings with mixed case (case-sensitive encoding), and single-character strings (encoded as "1X").

An alternative approach would allow multi-digit counts, but this problem specifically uses single-digit counts. The current approach handles this constraint efficiently by splitting long runs.

Think of this as compressing a string by counting consecutive characters: "AAA" becomes "3A", but if you have 12 A's, you encode it as "9A3A" to keep counts as single digits.`,
	},
	{
		ID:          54,
		Title:       "Common Characters",
		Description: "Write a function that takes in a non-empty list of non-empty strings and returns a list of characters that are common to all strings in the list, ignoring multiplicity. Note that the strings are not guaranteed to only contain alphanumeric characters. The list you return can be in any order.",
		Difficulty:  "Easy",
		Topic:       "Strings",
		Signature:   "func commonCharacters(strings []string) []string",
		TestCases: []TestCase{
			{Input: "strings = [\"abc\", \"bcd\", \"cbad\"]", Expected: "[\"b\", \"c\"]"},
			{Input: "strings = [\"ab\", \"bc\"]", Expected: "[\"b\"]"},
			{Input: "strings = [\"abcde\", \"aa\", \"foo\", \"omg\", \"bar\"]", Expected: "[]"},
			{Input: "strings = [\"abc\", \"bcd\", \"cbaccd\"]", Expected: "[\"b\", \"c\"]"},
		},
		Solution: `func commonCharacters(strings []string) []string {
	characterCounts := make(map[rune]int)
	for _, str := range strings {
		uniqueStringCharacters := make(map[rune]bool)
		for _, char := range str {
			uniqueStringCharacters[char] = true
		}
		for char := range uniqueStringCharacters {
			characterCounts[char]++
		}
	}
	result := []string{}
	for char, count := range characterCounts {
		if count == len(strings) {
			result = append(result, string(char))
		}
	}
	return result
}`,
		PythonSolution: `def commonCharacters(strings: List[str]) -> List[str]:
    character_counts = {}
    for string in strings:
        unique_string_characters = set(string)
        for char in unique_string_characters:
            character_counts[char] = character_counts.get(char, 0) + 1
    result = []
    for char, count in character_counts.items():
        if count == len(strings):
            result.append(char)
    return result`,
		Explanation: `Finding common characters across multiple strings requires tracking which characters appear in each string. The key insight is that a character is common only if it appears in every string, and we should ignore multiplicity (a character appearing multiple times in one string counts the same as appearing once).

The algorithm uses a two-step process. First, for each string, it creates a set of unique characters (ignoring duplicates within the string). Then, it counts how many strings contain each unique character by incrementing a count in a hash map. After processing all strings, characters with a count equal to the number of strings are common to all strings and are added to the result.

This approach efficiently handles the "ignoring multiplicity" requirement by using sets to represent unique characters per string. A character only needs to appear once in a string to count toward the common characters.

Edge cases handled include: strings with no common characters (returns empty array), strings where one string is a subset of others (returns the subset's characters), strings with duplicate characters (each unique character counted once per string), and strings with special characters (handled correctly).

An alternative approach would be to find the intersection of all character sets, but the counting approach is more straightforward. Another approach would use arrays for fixed character sets (like lowercase letters), but the hash map approach handles any characters.

Think of this as finding items that appear on every shopping list: you count how many lists each item appears on, and items that appear on all lists are the common items.`,
	},
	{
		ID:          55,
		Title:       "Generate Document",
		Description: "You're given a string of available characters and a string representing a document that you need to generate. Write a function that determines if you can generate the document using the available characters. If you can generate the document, your function should return true; otherwise, it should return false. You're only able to generate the document if the frequency of unique characters in the characters string is greater than or equal to the frequency of the same unique characters in the document string.",
		Difficulty:  "Easy",
		Topic:       "Hash Tables",
		Signature:   "func generateDocument(characters string, document string) bool",
		TestCases: []TestCase{
			{Input: "characters = \"Bste!hetsi ogEAxpelrt x \", document = \"AlgoExpert is the Best!\"", Expected: "true"},
			{Input: "characters = \"A\", document = \"a\"", Expected: "false"},
			{Input: "characters = \"a\", document = \"a\"", Expected: "true"},
			{Input: "characters = \"a hsgalhsa sanbhagsva;sanhags; ajhgasjkhagsakjg sdkalgdkhags dkjhadgasjkgdaskj gadsjkhgadkhgadakhgadakhgadakhgsakjgahskjgaakjgahkjgahkjgahkjghakjga\", document = \"\"", Expected: "true"},
		},
		Solution: `func generateDocument(characters string, document string) bool {
	characterCounts := make(map[rune]int)
	for _, char := range characters {
		characterCounts[char]++
	}
	for _, char := range document {
		if characterCounts[char] == 0 {
			return false
		}
		characterCounts[char]--
	}
	return true
}`,
		PythonSolution: `def generateDocument(characters: str, document: str) -> bool:
    character_counts = {}
    for char in characters:
        character_counts[char] = character_counts.get(char, 0) + 1
    for char in document:
        if character_counts.get(char, 0) == 0:
            return False
        character_counts[char] -= 1
    return True`,
		Explanation: `This problem checks if we can form a document using characters from a given character string. The key requirement is that we must have enough occurrences of each character - having the character once isn't enough if the document needs it multiple times.

The algorithm first counts all available characters in the characters string, storing frequencies in a hash map. Then, it iterates through each character in the document, checking if that character is available (count > 0). If available, it decrements the count to "use" that character. If a character is not available (count is 0), it immediately returns false, as we cannot form the document.

This approach efficiently handles character frequency requirements. By decrementing counts as we "use" characters, we ensure we don't use more characters than available. The early return when a character is unavailable makes the algorithm efficient for cases where the document cannot be formed.

Edge cases handled include: empty document (can always be formed, returns true), document longer than characters string (cannot be formed), document requiring characters not in characters string (returns false immediately), and documents with repeated characters (each occurrence consumes one available character).

An alternative approach would be to count characters in both strings and compare frequencies, but the decrementing approach is more space-efficient and allows early termination. Another approach would sort both strings, but that's less efficient.

Think of this as having a box of letter tiles and trying to spell a word: you need enough of each letter tile. As you use each tile, you remove it from your available tiles. If you run out of a needed tile, you can't complete the word.`,
	},
	{
		ID:          56,
		Title:       "First Non-Repeating Character",
		Description: "Write a function that takes in a string of lowercase English-alphabet letters and returns the index of the string's first non-repeating character. The first non-repeating character is the first character in a string that occurs only once. If the input string doesn't have any non-repeating characters, your function should return -1.",
		Difficulty:  "Easy",
		Topic:       "Hash Tables",
		Signature:   "func firstNonRepeatingCharacter(str string) int",
		TestCases: []TestCase{
			{Input: "str = \"abcdcaf\"", Expected: "1"},
			{Input: "str = \"faadabcbbebdf\"", Expected: "6"},
			{Input: "str = \"a\"", Expected: "0"},
			{Input: "str = \"ababac\"", Expected: "5"},
			{Input: "str = \"ababacc\"", Expected: "-1"},
		},
		Solution: `func firstNonRepeatingCharacter(str string) int {
	characterFrequencies := make(map[rune]int)
	for _, char := range str {
		characterFrequencies[char]++
	}
	for i, char := range str {
		if characterFrequencies[char] == 1 {
			return i
		}
	}
	return -1
}`,
		PythonSolution: `def firstNonRepeatingCharacter(string: str) -> int:
    character_frequencies = {}
    for char in string:
        character_frequencies[char] = character_frequencies.get(char, 0) + 1
    for i, char in enumerate(string):
        if character_frequencies[char] == 1:
            return i
    return -1`,
		Explanation: `Finding the first non-repeating character requires two passes: one to count frequencies and one to find the first character with frequency 1. The key insight is that we need to maintain the order of first occurrence while tracking frequencies.

The algorithm first counts the frequency of each character in the string using a hash map. Then, it iterates through the string again in order, checking the frequency of each character. The first character with frequency 1 is the answer. This two-pass approach ensures we find the first non-repeating character according to the original order in the string.

This approach is efficient because hash map lookups are O(1), making the overall time complexity O(n) for both passes. The space complexity is O(1) in terms of the character set size (limited to 26 lowercase letters), though it's O(k) where k is the number of unique characters in general.

Edge cases handled include: strings where all characters repeat (returns -1), strings with a single character (returns index 0), strings where the first character is non-repeating (returns immediately), and strings where non-repeating character appears later (must check all previous characters first).

An alternative approach would use a linked hash map to maintain insertion order, but the two-pass approach is simpler and equally efficient. Another approach would track first occurrence indices, but the current approach is more straightforward.

Think of this as finding the first person in a line who appears only once: you first count how many times each person appears, then go through the line again to find the first person with a count of 1.`,
	},
	{
		ID:          57,
		Title:       "Semordnilap",
		Description: "Write a function that takes in a list of unique strings and returns a list of semordnilap pairs. A semordnilap pair is defined as a set of different strings where the reverse of one word is the same as the forward version of the other. For example the words \"diaper\" and \"repaid\" are a semordnilap pair, as are \"palindromes\" and \"semordnilap\".",
		Difficulty:  "Easy",
		Topic:       "Hash Tables",
		Signature:   "func semordnilap(words []string) [][]string",
		TestCases: []TestCase{
			{Input: "words = [\"diaper\", \"abc\", \"test\", \"cba\", \"repaid\"]", Expected: "[[\"diaper\", \"repaid\"], [\"abc\", \"cba\"]]"},
			{Input: "words = [\"dog\", \"hello\", \"god\"]", Expected: "[[\"dog\", \"god\"]]"},
			{Input: "words = [\"dog\", \"desserts\", \"god\", \"stressed\"]", Expected: "[[\"dog\", \"god\"], [\"desserts\", \"stressed\"]]"},
		},
		Solution: `func semordnilap(words []string) [][]string {
	wordsSet := make(map[string]bool)
	for _, word := range words {
		wordsSet[word] = true
	}
	semordnilapPairs := [][]string{}
	for _, word := range words {
		reverse := reverseString(word)
		if wordsSet[reverse] && reverse != word {
			semordnilapPairs = append(semordnilapPairs, []string{word, reverse})
			delete(wordsSet, word)
			delete(wordsSet, reverse)
		}
	}
	return semordnilapPairs
}

func reverseString(s string) string {
	runes := []rune(s)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	return string(runes)
}`,
		PythonSolution: `def semordnilap(words: List[str]) -> List[List[str]]:
    words_set = set(words)
    semordnilap_pairs = []
    for word in words:
        reverse = word[::-1]
        if reverse in words_set and reverse != word:
            semordnilap_pairs.append([word, reverse])
            words_set.discard(word)
            words_set.discard(reverse)
    return semordnilap_pairs`,
		Explanation: `A semordnilap is a word that spells another word when reversed (like "diaper" and "repaid"). Finding semordnilap pairs requires checking if the reverse of each word exists in the word list.

The algorithm first stores all words in a hash set for O(1) lookup. Then, it iterates through each word, calculates its reverse, and checks if the reverse exists in the set. If it does and the reverse is different from the original word (to avoid palindromes), it adds the pair to the result and removes both words from the set to avoid duplicate pairs and ensure each word appears in at most one pair.

This approach efficiently finds all semordnilap pairs by leveraging hash set lookups. Removing words after pairing ensures we don't create duplicate pairs or pair the same word with multiple partners.

Edge cases handled include: words that are palindromes (reverse equals original, so not semordnilaps), words with no reverse in the list (not paired), words that appear multiple times (each occurrence can potentially form a pair, but we remove after pairing), and empty word lists (returns empty result).

An alternative approach would be to generate all pairs and check if one is the reverse of the other, but that would be O(n²) time. The hash set approach is more efficient with O(n) time for set creation and O(n*m) time for reversing and checking, where m is average word length.

Think of this as finding word pairs that are mirror images: you check each word to see if its reverse exists in your dictionary, and when you find a match, you pair them up and remove them from consideration.`,
	},
}
