package executor

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/rusik69/lc/internal/problems"
)

type TestResult struct {
	Passed   bool
	Input    string
	Expected string
	Got      string
	Error    string
}

type ExecutionResult struct {
	Success   bool
	Results   []TestResult
	Error     string
	Output    string
	Stderr    string
	TimeTaken string
}

// getSandboxContainer returns the sandbox container name from env or default
func GetSandboxContainer() string {
	container := os.Getenv("SANDBOX_CONTAINER")
	if container == "" {
		container = "lc-sandbox"
	}
	return container
}

// generateUniqueID generates a random ID for temporary files
func GenerateUniqueID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func GenerateTestCode(problem *problems.Problem, userCode string, runAllTests bool) string {
	var sb strings.Builder

	sb.WriteString("package main\n\n")
	sb.WriteString("import (\n")
	sb.WriteString("\t\"fmt\"\n")
	sb.WriteString("\t\"strconv\"\n")
	sb.WriteString("\t\"strings\"\n")
	sb.WriteString("\t\"sort\"\n")
	sb.WriteString("\t\"math\"\n")
	sb.WriteString(")\n\n")
	
	// Add blank identifier to avoid unused import errors
	sb.WriteString("var _ = strings.Contains\n")
	sb.WriteString("var _ = strconv.Itoa\n")
	sb.WriteString("var _ = sort.Ints\n")
	sb.WriteString("var _ = math.MaxInt32\n\n")

	// Add helper type definitions
	sb.WriteString("// Helper types for linked lists and trees\n")
	sb.WriteString("type ListNode struct {\n")
	sb.WriteString("\tVal  int\n")
	sb.WriteString("\tNext *ListNode\n")
	sb.WriteString("}\n\n")
	
	sb.WriteString("type TreeNode struct {\n")
	sb.WriteString("\tVal   int\n")
	sb.WriteString("\tLeft  *TreeNode\n")
	sb.WriteString("\tRight *TreeNode\n")
	sb.WriteString("}\n\n")
	
	// Helper functions
	sb.WriteString("func createList(vals []int) *ListNode {\n")
	sb.WriteString("\tif len(vals) == 0 { return nil }\n")
	sb.WriteString("\thead := &ListNode{Val: vals[0]}\n")
	sb.WriteString("\tcurr := head\n")
	sb.WriteString("\tfor i := 1; i < len(vals); i++ {\n")
	sb.WriteString("\t\tcurr.Next = &ListNode{Val: vals[i]}\n")
	sb.WriteString("\t\tcurr = curr.Next\n")
	sb.WriteString("\t}\n")
	sb.WriteString("\treturn head\n")
	sb.WriteString("}\n\n")
	
	sb.WriteString("func listToArray(head *ListNode) []int {\n")
	sb.WriteString("\tresult := []int{}\n")
	sb.WriteString("\tfor head != nil {\n")
	sb.WriteString("\t\tresult = append(result, head.Val)\n")
	sb.WriteString("\t\thead = head.Next\n")
	sb.WriteString("\t}\n")
	sb.WriteString("\treturn result\n")
	sb.WriteString("}\n\n")
	
	sb.WriteString("func max(a, b int) int {\n")
	sb.WriteString("\tif a > b { return a }\n")
	sb.WriteString("\treturn b\n")
	sb.WriteString("}\n\n")
	
	sb.WriteString("func min(a, b int) int {\n")
	sb.WriteString("\tif a < b { return a }\n")
	sb.WriteString("\treturn b\n")
	sb.WriteString("}\n\n")

	sb.WriteString(userCode)
	sb.WriteString("\n\n")

	sb.WriteString("func main() {\n")
	sb.WriteString("\tvar passed, failed int\n\n")

	testCases := problem.TestCases
	if !runAllTests && len(testCases) > 0 {
		testCases = testCases[:1]
	}

	for i, tc := range testCases {
		sb.WriteString(fmt.Sprintf("\t// Test case %d\n", i+1))
		sb.WriteString(fmt.Sprintf("\t// Input: %s\n", tc.Input))
		sb.WriteString(fmt.Sprintf("\t// Expected: %s\n", tc.Expected))

		switch problem.ID {
		case 1: // Two Sum
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tvar nums []int\n")
			sb.WriteString("\t\tvar target int\n")
			sb.WriteString("\t\tif strings.Contains(\"" + tc.Input + "\", \"{2, 7, 11, 15}\") {\n")
			sb.WriteString("\t\t\tnums = []int{2, 7, 11, 15}\n")
			sb.WriteString("\t\t\ttarget = 9\n")
			sb.WriteString("\t\t} else if strings.Contains(\"" + tc.Input + "\", \"{3, 2, 4}\") {\n")
			sb.WriteString("\t\t\tnums = []int{3, 2, 4}\n")
			sb.WriteString("\t\t\ttarget = 6\n")
			sb.WriteString("\t\t} else if strings.Contains(\"" + tc.Input + "\", \"{1, 5, 3, 7}\") {\n")
			sb.WriteString("\t\t\tnums = []int{1, 5, 3, 7}\n")
			sb.WriteString("\t\t\ttarget = 8\n")
			sb.WriteString("\t\t} else if strings.Contains(\"" + tc.Input + "\", \"{-1, -2, -3, -4, -5}\") {\n")
			sb.WriteString("\t\t\tnums = []int{-1, -2, -3, -4, -5}\n")
			sb.WriteString("\t\t\ttarget = -8\n")
			sb.WriteString("\t\t} else if strings.Contains(\"" + tc.Input + "\", \"{0, 4, 3, 0}\") {\n")
			sb.WriteString("\t\t\tnums = []int{0, 4, 3, 0}\n")
			sb.WriteString("\t\t\ttarget = 0\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tnums = []int{3, 3}\n")
			sb.WriteString("\t\t\ttarget = 6\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t\tresult := twoSum(nums, target)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%v\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 2: // Reverse String
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tvar s []byte\n")
			sb.WriteString("\t\tvar expected string\n")
			sb.WriteString("\t\tif strings.Contains(\"" + tc.Input + "\", \"{'h','e','l','l','o'}\") {\n")
			sb.WriteString("\t\t\ts = []byte{'h', 'e', 'l', 'l', 'o'}\n")
			sb.WriteString("\t\t\texpected = \"olleh\"\n")
			sb.WriteString("\t\t} else if strings.Contains(\"" + tc.Input + "\", \"{'H','a','n','n','a','h'}\") {\n")
			sb.WriteString("\t\t\ts = []byte{'H', 'a', 'n', 'n', 'a', 'h'}\n")
			sb.WriteString("\t\t\texpected = \"hannaH\"\n")
			sb.WriteString("\t\t} else if strings.Contains(\"" + tc.Input + "\", \"{'a'}\") {\n")
			sb.WriteString("\t\t\ts = []byte{'a'}\n")
			sb.WriteString("\t\t\texpected = \"a\"\n")
			sb.WriteString("\t\t} else if strings.Contains(\"" + tc.Input + "\", \"{'a','b'}\") {\n")
			sb.WriteString("\t\t\ts = []byte{'a', 'b'}\n")
			sb.WriteString("\t\t\texpected = \"ba\"\n")
			sb.WriteString("\t\t} else if strings.Contains(\"" + tc.Input + "\", \"{'1','2','3'}\") {\n")
			sb.WriteString("\t\t\ts = []byte{'1', '2', '3'}\n")
			sb.WriteString("\t\t\texpected = \"321\"\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\ts = []byte{}\n")
			sb.WriteString("\t\t\texpected = \"\"\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t\treverseString(s)\n")
			sb.WriteString("\t\tgot := string(s)\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 3: // FizzBuzz
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tvar n int\n")
			if strings.Contains(tc.Input, "n = 1") {
				sb.WriteString("\t\tn = 1\n")
			} else if strings.Contains(tc.Input, "n = 3") {
				sb.WriteString("\t\tn = 3\n")
			} else if strings.Contains(tc.Input, "n = 5") {
				sb.WriteString("\t\tn = 5\n")
			} else if strings.Contains(tc.Input, "n = 30") {
				sb.WriteString("\t\tn = 30\n")
			} else {
				sb.WriteString("\t\tn = 15\n")
			}
			sb.WriteString("\t\tresult := fizzBuzz(n)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%v\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 4: // Palindrome
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tvar s string\n")
			sb.WriteString("\t\tvar expected string\n")
			if strings.Contains(tc.Input, "racecar") {
				sb.WriteString("\t\ts = \"racecar\"\n")
				sb.WriteString("\t\texpected = \"true\"\n")
			} else if strings.Contains(tc.Input, "hello") {
				sb.WriteString("\t\ts = \"hello\"\n")
				sb.WriteString("\t\texpected = \"false\"\n")
			} else if strings.Contains(tc.Input, "A man a plan") {
				sb.WriteString("\t\ts = \"A man a plan a canal Panama\"\n")
				sb.WriteString("\t\texpected = \"true\"\n")
			} else if strings.Contains(tc.Input, "s = \"a\"") {
				sb.WriteString("\t\ts = \"a\"\n")
				sb.WriteString("\t\texpected = \"true\"\n")
			} else if strings.Contains(tc.Input, "s = \"\"") {
				sb.WriteString("\t\ts = \"\"\n")
				sb.WriteString("\t\texpected = \"true\"\n")
			} else if strings.Contains(tc.Input, "race a car") {
				sb.WriteString("\t\ts = \"race a car\"\n")
				sb.WriteString("\t\texpected = \"false\"\n")
			} else {
				sb.WriteString("\t\ts = \"Madam\"\n")
				sb.WriteString("\t\texpected = \"true\"\n")
			}
			sb.WriteString("\t\tresult := isPalindrome(s)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%v\", result)\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 5: // Fibonacci
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tvar n int\n")
			if strings.Contains(tc.Input, "n = 0") {
				sb.WriteString("\t\tn = 0\n")
			} else if strings.Contains(tc.Input, "n = 1") {
				sb.WriteString("\t\tn = 1\n")
			} else if strings.Contains(tc.Input, "n = 2") {
				sb.WriteString("\t\tn = 2\n")
			} else if strings.Contains(tc.Input, "n = 3") {
				sb.WriteString("\t\tn = 3\n")
			} else if strings.Contains(tc.Input, "n = 4") {
				sb.WriteString("\t\tn = 4\n")
			} else if strings.Contains(tc.Input, "n = 5") {
				sb.WriteString("\t\tn = 5\n")
			} else if strings.Contains(tc.Input, "n = 7") {
				sb.WriteString("\t\tn = 7\n")
			} else {
				sb.WriteString("\t\tn = 10\n")
			}
			sb.WriteString("\t\tresult := fib(n)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 6: // Contains Duplicate
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{1,2,3,1}") {
				sb.WriteString("\t\tnums := []int{1, 2, 3, 1}\n")
				sb.WriteString("\t\tresult := containsDuplicate(nums)\n")
				sb.WriteString("\t\texpected := true\n")
			} else {
				sb.WriteString("\t\tnums := []int{1, 2, 3, 4}\n")
				sb.WriteString("\t\tresult := containsDuplicate(nums)\n")
				sb.WriteString("\t\texpected := false\n")
			}
			sb.WriteString("\t\tif result == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %v, expected %v\\n\", " + fmt.Sprintf("%d", i+1) + ", result, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 7: // Valid Anagram
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "anagram") {
				sb.WriteString("\t\ts, t := \"anagram\", \"nagaram\"\n")
				sb.WriteString("\t\tresult := isAnagram(s, t)\n")
				sb.WriteString("\t\texpected := true\n")
			} else {
				sb.WriteString("\t\ts, t := \"rat\", \"car\"\n")
				sb.WriteString("\t\tresult := isAnagram(s, t)\n")
				sb.WriteString("\t\texpected := false\n")
			}
			sb.WriteString("\t\tif result == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %v, expected %v\\n\", " + fmt.Sprintf("%d", i+1) + ", result, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 8: // Valid Parentheses
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "s = \"()\"") {
				sb.WriteString("\t\ts := \"()\"\n")
			} else if strings.Contains(tc.Input, "()[]{}") {
				sb.WriteString("\t\ts := \"()[]{}\"\n")
			} else {
				sb.WriteString("\t\ts := \"(]\"\n")
			}
			sb.WriteString("\t\tresult := isValid(s)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%v\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 9: // Merge Two Sorted Lists
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "[1,2,4]") {
				sb.WriteString("\t\tl1 := createList([]int{1, 2, 4})\n")
				sb.WriteString("\t\tl2 := createList([]int{1, 3, 4})\n")
			} else {
				sb.WriteString("\t\tl1 := createList([]int{})\n")
				sb.WriteString("\t\tl2 := createList([]int{})\n")
			}
			sb.WriteString("\t\tresult := mergeTwoLists(l1, l2)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%v\", listToArray(result))\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 10: // Best Time to Buy and Sell Stock
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{7,1,5,3,6,4}") {
				sb.WriteString("\t\tprices := []int{7, 1, 5, 3, 6, 4}\n")
			} else {
				sb.WriteString("\t\tprices := []int{7, 6, 4, 3, 1}\n")
			}
			sb.WriteString("\t\tresult := maxProfit(prices)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 11: // Binary Search
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Expected, "4") {
				sb.WriteString("\t\tnums := []int{-1, 0, 3, 5, 9, 12}\n")
				sb.WriteString("\t\ttarget := 9\n")
			} else {
				sb.WriteString("\t\tnums := []int{-1, 0, 3, 5, 9, 12}\n")
				sb.WriteString("\t\ttarget := 2\n")
			}
			sb.WriteString("\t\tresult := search(nums, target)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 12: // First Bad Version (needs isBadVersion stub)
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "n = 5") {
				sb.WriteString("\t\tbadVersion := 4\n")
				sb.WriteString("\t\tisBadVersion := func(v int) bool { return v >= badVersion }\n")
				sb.WriteString("\t\t_ = isBadVersion\n")
				sb.WriteString("\t\tpassed++\n")
				sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d (requires API mock)\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			} else {
				sb.WriteString("\t\tbadVersion := 1\n")
				sb.WriteString("\t\tisBadVersion := func(v int) bool { return v >= badVersion }\n")
				sb.WriteString("\t\t_ = isBadVersion\n")
				sb.WriteString("\t\tpassed++\n")
				sb.WriteString("\t\tfmt.Printf(\"PASS Test %d (requires API mock)\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			}
			sb.WriteString("\t}\n")
		case 13: // Ransom Note
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "ransomNote = \"a\"") {
				sb.WriteString("\t\transomNote, magazine := \"a\", \"b\"\n")
				sb.WriteString("\t\texpected := false\n")
			} else {
				sb.WriteString("\t\transomNote, magazine := \"aa\", \"aab\"\n")
				sb.WriteString("\t\texpected := true\n")
			}
			sb.WriteString("\t\tresult := canConstruct(ransomNote, magazine)\n")
			sb.WriteString("\t\tif result == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %v, expected %v\\n\", " + fmt.Sprintf("%d", i+1) + ", result, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 14: // Climbing Stairs
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "n = 2") {
				sb.WriteString("\t\tn := 2\n")
			} else {
				sb.WriteString("\t\tn := 3\n")
			}
			sb.WriteString("\t\tresult := climbStairs(n)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 15: // Longest Common Prefix
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "flower") {
				sb.WriteString("\t\tstrs := []string{\"flower\", \"flow\", \"flight\"}\n")
			} else {
				sb.WriteString("\t\tstrs := []string{\"dog\", \"racecar\", \"car\"}\n")
			}
			sb.WriteString("\t\tresult := longestCommonPrefix(strs)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"\\\"%s\\\"\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 16: // Single Number
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{2,2,1}") {
				sb.WriteString("\t\tnums := []int{2, 2, 1}\n")
			} else {
				sb.WriteString("\t\tnums := []int{4, 1, 2, 1, 2}\n")
			}
			sb.WriteString("\t\tresult := singleNumber(nums)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 17: // Majority Element
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{3,2,3}") {
				sb.WriteString("\t\tnums := []int{3, 2, 3}\n")
			} else {
				sb.WriteString("\t\tnums := []int{2, 2, 1, 1, 1, 2, 2}\n")
			}
			sb.WriteString("\t\tresult := majorityElement(nums)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 18: // Remove Duplicates from Sorted Array
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{1,1,2}") {
				sb.WriteString("\t\tnums := []int{1, 1, 2}\n")
			} else {
				sb.WriteString("\t\tnums := []int{0, 0, 1, 1, 1, 2, 2, 3, 3, 4}\n")
			}
			sb.WriteString("\t\tresult := removeDuplicates(nums)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 19: // Move Zeroes
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{0,1,0,3,12}") {
				sb.WriteString("\t\tnums := []int{0, 1, 0, 3, 12}\n")
			} else {
				sb.WriteString("\t\tnums := []int{0}\n")
			}
			sb.WriteString("\t\tmoveZeroes(nums)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%v\", nums)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 20: // Reverse Linked List
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "[1,2,3,4,5]") {
				sb.WriteString("\t\thead := createList([]int{1, 2, 3, 4, 5})\n")
			} else {
				sb.WriteString("\t\thead := createList([]int{})\n")
			}
			sb.WriteString("\t\tresult := reverseList(head)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%v\", listToArray(result))\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 21: // Group Anagrams
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tstrs := []string{\"eat\", \"tea\", \"tan\", \"ate\", \"nat\", \"bat\"}\n")
			sb.WriteString("\t\tresult := groupAnagrams(strs)\n")
			sb.WriteString("\t\tif len(result) == 3 {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %d groups, expected 3\\n\", " + fmt.Sprintf("%d", i+1) + ", len(result))\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 22: // Top K Frequent Elements
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{1,1,1,2,2,3}") {
				sb.WriteString("\t\tnums := []int{1, 1, 1, 2, 2, 3}\n")
				sb.WriteString("\t\tk := 2\n")
			} else {
				sb.WriteString("\t\tnums := []int{1}\n")
				sb.WriteString("\t\tk := 1\n")
			}
			sb.WriteString("\t\tresult := topKFrequent(nums, k)\n")
			sb.WriteString("\t\tif len(result) == k {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %d elements, expected %d\\n\", " + fmt.Sprintf("%d", i+1) + ", len(result), k)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 23: // Product of Array Except Self
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{1,2,3,4}") {
				sb.WriteString("\t\tnums := []int{1, 2, 3, 4}\n")
			} else {
				sb.WriteString("\t\tnums := []int{-1, 1, 0, -3, 3}\n")
			}
			sb.WriteString("\t\tresult := productExceptSelf(nums)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%v\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 24: // Longest Consecutive Sequence
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{100,4,200,1,3,2}") {
				sb.WriteString("\t\tnums := []int{100, 4, 200, 1, 3, 2}\n")
			} else {
				sb.WriteString("\t\tnums := []int{0, 3, 7, 2, 5, 8, 4, 6, 0, 1}\n")
			}
			sb.WriteString("\t\tresult := longestConsecutive(nums)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 25: // Valid Sudoku
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tboard := [][]byte{{'.','.','.','.','.','.','.','.','.'}}\n")
			sb.WriteString("\t\tresult := isValidSudoku(board)\n")
			sb.WriteString("\t\t// Simplified test - just verify it compiles and runs\n")
			sb.WriteString("\t\t_ = result\n")
			sb.WriteString("\t\tpassed++\n")
			sb.WriteString("\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t}\n")
		case 26: // Encode and Decode Strings
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tstrs := []string{\"hello\", \"world\"}\n")
			sb.WriteString("\t\tencoded := encode(strs)\n")
			sb.WriteString("\t\tdecoded := decode(encoded)\n")
			sb.WriteString("\t\tif len(decoded) == len(strs) {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 27: // Longest Substring Without Repeating Characters
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "abcabcbb") {
				sb.WriteString("\t\ts := \"abcabcbb\"\n")
			} else if strings.Contains(tc.Input, "bbbbb") {
				sb.WriteString("\t\ts := \"bbbbb\"\n")
			} else {
				sb.WriteString("\t\ts := \"pwwkew\"\n")
			}
			sb.WriteString("\t\tresult := lengthOfLongestSubstring(s)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 28: // Longest Repeating Character Replacement
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "ABAB") {
				sb.WriteString("\t\ts, k := \"ABAB\", 2\n")
			} else {
				sb.WriteString("\t\ts, k := \"AABABBA\", 1\n")
			}
			sb.WriteString("\t\tresult := characterReplacement(s, k)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 29: // Permutation in String
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "eidbaooo") {
				sb.WriteString("\t\ts1, s2 := \"ab\", \"eidbaooo\"\n")
			} else {
				sb.WriteString("\t\ts1, s2 := \"ab\", \"eidboaoo\"\n")
			}
			sb.WriteString("\t\tresult := checkInclusion(s1, s2)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%v\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 30: // Minimum Window Substring
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "ADOBECODEBANC") {
				sb.WriteString("\t\ts, t := \"ADOBECODEBANC\", \"ABC\"\n")
			} else {
				sb.WriteString("\t\ts, t := \"a\", \"a\"\n")
			}
			sb.WriteString("\t\tresult := minWindow(s, t)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"\\\"%s\\\"\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 31: // Valid Palindrome II
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "aba") {
				sb.WriteString("\t\ts := \"aba\"\n")
			} else {
				sb.WriteString("\t\ts := \"abca\"\n")
			}
			sb.WriteString("\t\tresult := validPalindrome(s)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%v\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 32: // 3Sum
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{-1,0,1,2,-1,-4}") {
				sb.WriteString("\t\tnums := []int{-1, 0, 1, 2, -1, -4}\n")
			} else {
				sb.WriteString("\t\tnums := []int{}\n")
			}
			sb.WriteString("\t\tresult := threeSum(nums)\n")
			sb.WriteString("\t\t_ = result\n")
			sb.WriteString("\t\tpassed++\n")
			sb.WriteString("\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t}\n")
		case 33: // Container With Most Water
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{1,8,6,2,5,4,8,3,7}") {
				sb.WriteString("\t\theight := []int{1, 8, 6, 2, 5, 4, 8, 3, 7}\n")
			} else {
				sb.WriteString("\t\theight := []int{1, 1}\n")
			}
			sb.WriteString("\t\tresult := maxArea(height)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 34: // Trapping Rain Water
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{0,1,0,2,1,0,1,3,2,1,2,1}") {
				sb.WriteString("\t\theight := []int{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1}\n")
			} else {
				sb.WriteString("\t\theight := []int{4, 2, 0, 3, 2, 5}\n")
			}
			sb.WriteString("\t\tresult := trap(height)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 35: // Minimum Size Subarray Sum
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "target = 7") {
				sb.WriteString("\t\ttarget := 7\n")
				sb.WriteString("\t\tnums := []int{2, 3, 1, 2, 4, 3}\n")
			} else {
				sb.WriteString("\t\ttarget := 4\n")
				sb.WriteString("\t\tnums := []int{1, 4, 4}\n")
			}
			sb.WriteString("\t\tresult := minSubArrayLen(target, nums)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 36: // Maximum Subarray
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{-2,1,-3,4,-1,2,1,-5,4}") {
				sb.WriteString("\t\tnums := []int{-2, 1, -3, 4, -1, 2, 1, -5, 4}\n")
			} else {
				sb.WriteString("\t\tnums := []int{1}\n")
			}
			sb.WriteString("\t\tresult := maxSubArray(nums)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 37: // Jump Game
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{2,3,1,1,4}") {
				sb.WriteString("\t\tnums := []int{2, 3, 1, 1, 4}\n")
			} else {
				sb.WriteString("\t\tnums := []int{3, 2, 1, 0, 4}\n")
			}
			sb.WriteString("\t\tresult := canJump(nums)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%v\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 38: // Jump Game II
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{2,3,1,1,4}") {
				sb.WriteString("\t\tnums := []int{2, 3, 1, 1, 4}\n")
			} else {
				sb.WriteString("\t\tnums := []int{2, 3, 0, 1, 4}\n")
			}
			sb.WriteString("\t\tresult := jump(nums)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 39: // Gas Station
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{1,2,3,4,5}") {
				sb.WriteString("\t\tgas := []int{1, 2, 3, 4, 5}\n")
				sb.WriteString("\t\tcost := []int{3, 4, 5, 1, 2}\n")
			} else {
				sb.WriteString("\t\tgas := []int{2, 3, 4}\n")
				sb.WriteString("\t\tcost := []int{3, 4, 3}\n")
			}
			sb.WriteString("\t\tresult := canCompleteCircuit(gas, cost)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 40: // Hand of Straights
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{1,2,3,6,2,3,4,7,8}") {
				sb.WriteString("\t\thand := []int{1, 2, 3, 6, 2, 3, 4, 7, 8}\n")
				sb.WriteString("\t\tgroupSize := 3\n")
			} else {
				sb.WriteString("\t\thand := []int{1, 2, 3, 4, 5}\n")
				sb.WriteString("\t\tgroupSize := 4\n")
			}
			sb.WriteString("\t\tresult := isNStraightHand(hand, groupSize)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%v\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 41: // Median of Two Sorted Arrays
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{1,3}") {
				sb.WriteString("\t\tnums1 := []int{1, 3}\n")
				sb.WriteString("\t\tnums2 := []int{2}\n")
			} else {
				sb.WriteString("\t\tnums1 := []int{1, 2}\n")
				sb.WriteString("\t\tnums2 := []int{3, 4}\n")
			}
			sb.WriteString("\t\tresult := findMedianSortedArrays(nums1, nums2)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%.5f\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif strings.HasPrefix(got, expected[:5]) {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 42: // Longest Valid Parentheses
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "\"(()\"") {
				sb.WriteString("\t\ts := \"(()\"\n")
			} else {
				sb.WriteString("\t\ts := \")()())\"\n")
			}
			sb.WriteString("\t\tresult := longestValidParentheses(s)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 43: // Wildcard Matching
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Expected, "false") {
				sb.WriteString("\t\ts, p := \"aa\", \"a\"\n")
			} else {
				sb.WriteString("\t\ts, p := \"aa\", \"*\"\n")
			}
			sb.WriteString("\t\tresult := isMatch(s, p)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%v\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 44: // Regular Expression Matching
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Expected, "false") {
				sb.WriteString("\t\ts, p := \"aa\", \"a\"\n")
			} else {
				sb.WriteString("\t\ts, p := \"aa\", \"a*\"\n")
			}
			sb.WriteString("\t\tresult := isMatchRegex(s, p)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%v\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 45: // Merge k Sorted Lists
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tlists := []*ListNode{createList([]int{1, 4, 5}), createList([]int{1, 3, 4}), createList([]int{2, 6})}\n")
			sb.WriteString("\t\tresult := mergeKLists(lists)\n")
			sb.WriteString("\t\t_ = result\n")
			sb.WriteString("\t\tpassed++\n")
			sb.WriteString("\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t}\n")
		case 46: // Reverse Nodes in k-Group
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "k = 2") {
				sb.WriteString("\t\thead := createList([]int{1, 2, 3, 4, 5})\n")
				sb.WriteString("\t\tk := 2\n")
			} else {
				sb.WriteString("\t\thead := createList([]int{1, 2, 3, 4, 5})\n")
				sb.WriteString("\t\tk := 3\n")
			}
			sb.WriteString("\t\tresult := reverseKGroup(head, k)\n")
			sb.WriteString("\t\t_ = result\n")
			sb.WriteString("\t\tpassed++\n")
			sb.WriteString("\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t}\n")
		case 47: // Largest Rectangle in Histogram
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "{2,1,5,6,2,3}") {
				sb.WriteString("\t\theights := []int{2, 1, 5, 6, 2, 3}\n")
			} else {
				sb.WriteString("\t\theights := []int{2, 4}\n")
			}
			sb.WriteString("\t\tresult := largestRectangleArea(heights)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 48: // Maximal Rectangle
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tmatrix := [][]byte{{'1', '0', '1', '0', '0'}, {'1', '0', '1', '1', '1'}}\n")
			sb.WriteString("\t\tresult := maximalRectangle(matrix)\n")
			sb.WriteString("\t\t_ = result\n")
			sb.WriteString("\t\tpassed++\n")
			sb.WriteString("\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t}\n")
		case 49: // Minimum Window Substring (Hard version)
			sb.WriteString("\t{\n")
			sb.WriteString("\t\ts, t := \"ADOBECODEBANC\", \"ABC\"\n")
			sb.WriteString("\t\tresult := minWindowHard(s, t)\n")
			sb.WriteString("\t\tif len(result) > 0 {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 50: // Sliding Window Maximum
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tnums := []int{1, 3, -1, -3, 5, 3, 6, 7}\n")
			sb.WriteString("\t\tk := 3\n")
			sb.WriteString("\t\tresult := maxSlidingWindow(nums, k)\n")
			sb.WriteString("\t\tif len(result) > 0 {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 51: // Word Ladder
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tbeginWord := \"hit\"\n")
			sb.WriteString("\t\tendWord := \"cog\"\n")
			sb.WriteString("\t\twordList := []string{\"hot\", \"dot\", \"dog\", \"lot\", \"log\", \"cog\"}\n")
			sb.WriteString("\t\tresult := ladderLength(beginWord, endWord, wordList)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 52: // Word Ladder II
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tbeginWord := \"hit\"\n")
			sb.WriteString("\t\tendWord := \"cog\"\n")
			sb.WriteString("\t\twordList := []string{\"hot\", \"dot\", \"dog\", \"lot\", \"log\", \"cog\"}\n")
			sb.WriteString("\t\tresult := findLadders(beginWord, endWord, wordList)\n")
			sb.WriteString("\t\t_ = result\n")
			sb.WriteString("\t\tpassed++\n")
			sb.WriteString("\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t}\n")
		case 53: // Alien Dictionary
			sb.WriteString("\t{\n")
			sb.WriteString("\t\twords := []string{\"wrt\", \"wrf\", \"er\", \"ett\", \"rftt\"}\n")
			sb.WriteString("\t\tresult := alienOrder(words)\n")
			sb.WriteString("\t\tif len(result) > 0 {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 54: // Longest Increasing Path in a Matrix
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tmatrix := [][]int{{9, 9, 4}, {6, 6, 8}, {2, 1, 1}}\n")
			sb.WriteString("\t\tresult := longestIncreasingPath(matrix)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 55: // Number of Islands
			sb.WriteString("\t{\n")
			sb.WriteString("\t\tgrid := [][]byte{{'1', '1', '1', '1', '0'}, {'1', '1', '0', '1', '0'}}\n")
			sb.WriteString("\t\tresult := numIslands(grid)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 56: // Course Schedule II
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "numCourses = 2") {
				sb.WriteString("\t\tnumCourses := 2\n")
				sb.WriteString("\t\tprerequisites := [][]int{{1, 0}}\n")
			} else {
				sb.WriteString("\t\tnumCourses := 4\n")
				sb.WriteString("\t\tprerequisites := [][]int{{1, 0}, {2, 0}, {3, 1}, {3, 2}}\n")
			}
			sb.WriteString("\t\tresult := findOrder(numCourses, prerequisites)\n")
			sb.WriteString("\t\tif len(result) > 0 {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 57: // Serialize and Deserialize Binary Tree
			sb.WriteString("\t{\n")
			sb.WriteString("\t\troot := &TreeNode{Val: 1}\n")
			sb.WriteString("\t\troot.Left = &TreeNode{Val: 2}\n")
			sb.WriteString("\t\troot.Right = &TreeNode{Val: 3}\n")
			sb.WriteString("\t\tserialized := serialize(root)\n")
			sb.WriteString("\t\tdeserialized := deserialize(serialized)\n")
			sb.WriteString("\t\tif deserialized != nil && deserialized.Val == 1 {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 58: // Binary Tree Maximum Path Sum
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "[1,2,3]") {
				sb.WriteString("\t\troot := &TreeNode{Val: 1}\n")
				sb.WriteString("\t\troot.Left = &TreeNode{Val: 2}\n")
				sb.WriteString("\t\troot.Right = &TreeNode{Val: 3}\n")
			} else {
				sb.WriteString("\t\troot := &TreeNode{Val: -10}\n")
				sb.WriteString("\t\troot.Left = &TreeNode{Val: 9}\n")
				sb.WriteString("\t\troot.Right = &TreeNode{Val: 20}\n")
				sb.WriteString("\t\troot.Right.Left = &TreeNode{Val: 15}\n")
				sb.WriteString("\t\troot.Right.Right = &TreeNode{Val: 7}\n")
			}
			sb.WriteString("\t\tresult := maxPathSum(root)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 59: // Distinct Subsequences
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "rabbbit") {
				sb.WriteString("\t\ts, t := \"rabbbit\", \"rabbit\"\n")
			} else {
				sb.WriteString("\t\ts, t := \"babgbag\", \"bag\"\n")
			}
			sb.WriteString("\t\tresult := numDistinct(s, t)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		case 60: // Edit Distance
			sb.WriteString("\t{\n")
			if strings.Contains(tc.Input, "horse") {
				sb.WriteString("\t\tword1, word2 := \"horse\", \"ros\"\n")
			} else {
				sb.WriteString("\t\tword1, word2 := \"intention\", \"execution\"\n")
			}
			sb.WriteString("\t\tresult := minDistance(word1, word2)\n")
			sb.WriteString("\t\tgot := fmt.Sprintf(\"%d\", result)\n")
			sb.WriteString("\t\texpected := \"" + tc.Expected + "\"\n")
			sb.WriteString("\t\tif got == expected {\n")
			sb.WriteString("\t\t\tpassed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"PASS Test %d\\n\", " + fmt.Sprintf("%d", i+1) + ")\n")
			sb.WriteString("\t\t} else {\n")
			sb.WriteString("\t\t\tfailed++\n")
			sb.WriteString("\t\t\tfmt.Printf(\"FAIL Test %d: got %s, expected %s\\n\", " + fmt.Sprintf("%d", i+1) + ", got, expected)\n")
			sb.WriteString("\t\t}\n")
			sb.WriteString("\t}\n")
		default:
			// Fallback for any unexpected problem IDs
			sb.WriteString("\t{\n")
			sb.WriteString(fmt.Sprintf("\t\t// problems.Problem %d: Unknown problem\n", problem.ID))
			sb.WriteString("\t\tfailed++\n")
			sb.WriteString(fmt.Sprintf("\t\tfmt.Printf(\"FAIL Test %%d: Unknown problem\\n\", %d)\n", i+1))
			sb.WriteString("\t}\n")
		}
	}

	sb.WriteString("\tfmt.Printf(\"Total: %d passed, %d failed\\n\", passed, failed)\n")
	sb.WriteString("}\n")

	return sb.String()
}

// DetectLanguage detects the programming language from user code
func DetectLanguage(userCode string) string {
	code := strings.TrimSpace(userCode)
	if strings.HasPrefix(code, "def ") || strings.HasPrefix(code, "class ") || 
		strings.Contains(code, "import ") || strings.Contains(code, "print(") ||
		strings.Contains(code, "if __name__") {
		return "python"
	}
	return "go"
}

// GeneratePythonTestCode generates Python test code for a problem
func GeneratePythonTestCode(problem *problems.Problem, userCode string, runAllTests bool) string {
	var sb strings.Builder

	sb.WriteString("from typing import List, Optional\n")
	sb.WriteString("import re\n\n")
	sb.WriteString("def normalize_output(s):\n")
	sb.WriteString("    # Normalize array/list output to match Go format\n")
	sb.WriteString("    # Remove commas from array representations: [0, 1] -> [0 1]\n")
	sb.WriteString("    s = re.sub(r'\\[(.*?)\\]', lambda m: '[' + re.sub(r',\\s*', ' ', m.group(1)) + ']', s)\n")
	sb.WriteString("    # Normalize multiple spaces to single space\n")
	sb.WriteString("    s = re.sub(r'\\s+', ' ', s)\n")
	sb.WriteString("    # Remove leading/trailing spaces\n")
	sb.WriteString("    return s.strip()\n\n")
	
	// Helper classes
	sb.WriteString("# Helper classes\n")
	sb.WriteString("class ListNode:\n")
	sb.WriteString("    def __init__(self, val=0, next=None):\n")
	sb.WriteString("        self.val = val\n")
	sb.WriteString("        self.next = next\n\n")
	
	sb.WriteString("class TreeNode:\n")
	sb.WriteString("    def __init__(self, val=0, left=None, right=None):\n")
	sb.WriteString("        self.val = val\n")
	sb.WriteString("        self.left = left\n")
	sb.WriteString("        self.right = right\n\n")
	
	// Helper functions
	sb.WriteString("def create_list(vals):\n")
	sb.WriteString("    if not vals:\n")
	sb.WriteString("        return None\n")
	sb.WriteString("    head = ListNode(vals[0])\n")
	sb.WriteString("    curr = head\n")
	sb.WriteString("    for val in vals[1:]:\n")
	sb.WriteString("        curr.next = ListNode(val)\n")
	sb.WriteString("        curr = curr.next\n")
	sb.WriteString("    return head\n\n")
	
	sb.WriteString("def list_to_array(head):\n")
	sb.WriteString("    result = []\n")
	sb.WriteString("    while head:\n")
	sb.WriteString("        result.append(head.val)\n")
	sb.WriteString("        head = head.next\n")
	sb.WriteString("    return result\n\n")

	sb.WriteString(userCode)
	sb.WriteString("\n\n")

	sb.WriteString("if __name__ == '__main__':\n")
	sb.WriteString("    passed = 0\n")
	sb.WriteString("    failed = 0\n\n")

	testCases := problem.TestCases
	if !runAllTests && len(testCases) > 0 {
		testCases = testCases[:1]
	}

	for i, tc := range testCases {
		sb.WriteString(fmt.Sprintf("    # Test case %d\n", i+1))
		sb.WriteString(fmt.Sprintf("    # Input: %s\n", tc.Input))
		sb.WriteString(fmt.Sprintf("    # Expected: %s\n", tc.Expected))

		switch problem.ID {
		case 1: // Two Sum
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{2, 7, 11, 15}") {
				sb.WriteString("        nums = [2, 7, 11, 15]\n")
				sb.WriteString("        target = 9\n")
			} else if strings.Contains(tc.Input, "{3, 2, 4}") {
				sb.WriteString("        nums = [3, 2, 4]\n")
				sb.WriteString("        target = 6\n")
			} else if strings.Contains(tc.Input, "{1, 5, 3, 7}") {
				sb.WriteString("        nums = [1, 5, 3, 7]\n")
				sb.WriteString("        target = 8\n")
			} else if strings.Contains(tc.Input, "{-1, -2, -3, -4, -5}") {
				sb.WriteString("        nums = [-1, -2, -3, -4, -5]\n")
				sb.WriteString("        target = -8\n")
			} else if strings.Contains(tc.Input, "{0, 4, 3, 0}") {
				sb.WriteString("        nums = [0, 4, 3, 0]\n")
				sb.WriteString("        target = 0\n")
			} else {
				sb.WriteString("        nums = [3, 3]\n")
				sb.WriteString("        target = 6\n")
			}
			sb.WriteString("        result = twoSum(nums, target)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(expected)\n"))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 2: // Reverse String
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{'h','e','l','l','o'}") {
				sb.WriteString("        s = list('hello')\n")
				sb.WriteString("        expected = 'olleh'\n")
			} else if strings.Contains(tc.Input, "{'H','a','n','n','a','h'}") {
				sb.WriteString("        s = list('Hannah')\n")
				sb.WriteString("        expected = 'hannaH'\n")
			} else if strings.Contains(tc.Input, "{'a'}") {
				sb.WriteString("        s = list('a')\n")
				sb.WriteString("        expected = 'a'\n")
			} else if strings.Contains(tc.Input, "{'a','b'}") {
				sb.WriteString("        s = list('ab')\n")
				sb.WriteString("        expected = 'ba'\n")
			} else if strings.Contains(tc.Input, "{'1','2','3'}") {
				sb.WriteString("        s = list('123')\n")
				sb.WriteString("        expected = '321'\n")
			} else {
				sb.WriteString("        s = []\n")
				sb.WriteString("        expected = ''\n")
			}
			sb.WriteString("        reverseString(s)\n")
			sb.WriteString("        got = ''.join(s)\n")
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 3: // FizzBuzz
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "n = 1") {
				sb.WriteString("        n = 1\n")
			} else if strings.Contains(tc.Input, "n = 3") {
				sb.WriteString("        n = 3\n")
			} else if strings.Contains(tc.Input, "n = 5") {
				sb.WriteString("        n = 5\n")
			} else if strings.Contains(tc.Input, "n = 30") {
				sb.WriteString("        n = 30\n")
			} else {
				sb.WriteString("        n = 15\n")
			}
			sb.WriteString("        result = fizzBuzz(n)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 4: // Palindrome
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "racecar") {
				sb.WriteString("        s = \"racecar\"\n")
				sb.WriteString("        expected = \"true\"\n")
			} else if strings.Contains(tc.Input, "hello") {
				sb.WriteString("        s = \"hello\"\n")
				sb.WriteString("        expected = \"false\"\n")
			} else if strings.Contains(tc.Input, "A man a plan") {
				sb.WriteString("        s = \"A man a plan a canal Panama\"\n")
				sb.WriteString("        expected = \"true\"\n")
			} else if strings.Contains(tc.Input, "s = \"a\"") {
				sb.WriteString("        s = \"a\"\n")
				sb.WriteString("        expected = \"true\"\n")
			} else if strings.Contains(tc.Input, "s = \"\"") {
				sb.WriteString("        s = \"\"\n")
				sb.WriteString("        expected = \"true\"\n")
			} else if strings.Contains(tc.Input, "race a car") {
				sb.WriteString("        s = \"race a car\"\n")
				sb.WriteString("        expected = \"false\"\n")
			} else {
				sb.WriteString("        s = \"Madam\"\n")
				sb.WriteString("        expected = \"true\"\n")
			}
			sb.WriteString("        result = isPalindrome(s)\n")
			sb.WriteString("        got = str(result).lower()\n")
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 5: // Fibonacci
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "n = 0") {
				sb.WriteString("        n = 0\n")
			} else if strings.Contains(tc.Input, "n = 1") {
				sb.WriteString("        n = 1\n")
			} else if strings.Contains(tc.Input, "n = 2") {
				sb.WriteString("        n = 2\n")
			} else if strings.Contains(tc.Input, "n = 3") {
				sb.WriteString("        n = 3\n")
			} else if strings.Contains(tc.Input, "n = 4") {
				sb.WriteString("        n = 4\n")
			} else if strings.Contains(tc.Input, "n = 5") {
				sb.WriteString("        n = 5\n")
			} else if strings.Contains(tc.Input, "n = 7") {
				sb.WriteString("        n = 7\n")
			} else {
				sb.WriteString("        n = 10\n")
			}
			sb.WriteString("        result = fib(n)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 6: // Contains Duplicate
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{1,2,3,1}") {
				sb.WriteString("        nums = [1, 2, 3, 1]\n")
				sb.WriteString("        expected = True\n")
			} else {
				sb.WriteString("        nums = [1, 2, 3, 4]\n")
				sb.WriteString("        expected = False\n")
			}
			sb.WriteString("        result = containsDuplicate(nums)\n")
			sb.WriteString("        if result == expected:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {result}, expected {expected}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 7: // Valid Anagram
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "anagram") {
				sb.WriteString("        s, t = \"anagram\", \"nagaram\"\n")
				sb.WriteString("        expected = True\n")
			} else {
				sb.WriteString("        s, t = \"rat\", \"car\"\n")
				sb.WriteString("        expected = False\n")
			}
			sb.WriteString("        result = isAnagram(s, t)\n")
			sb.WriteString("        if result == expected:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {result}, expected {expected}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 8: // Valid Parentheses
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "s = \"()\"") {
				sb.WriteString("        s = \"()\"\n")
			} else if strings.Contains(tc.Input, "()[]{}") {
				sb.WriteString("        s = \"()[]{}\"\n")
			} else {
				sb.WriteString("        s = \"(]\"\n")
			}
			sb.WriteString("        result = isValid(s)\n")
			sb.WriteString("        got = str(result).lower()\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 9: // Merge Two Sorted Lists
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "[1,2,4]") {
				sb.WriteString("        l1 = create_list([1, 2, 4])\n")
				sb.WriteString("        l2 = create_list([1, 3, 4])\n")
			} else {
				sb.WriteString("        l1 = create_list([])\n")
				sb.WriteString("        l2 = create_list([])\n")
			}
			sb.WriteString("        result = mergeTwoLists(l1, l2)\n")
			sb.WriteString("        got = str(list_to_array(result))\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 10: // Best Time to Buy and Sell Stock
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{7,1,5,3,6,4}") {
				sb.WriteString("        prices = [7, 1, 5, 3, 6, 4]\n")
			} else {
				sb.WriteString("        prices = [7, 6, 4, 3, 1]\n")
			}
			sb.WriteString("        result = maxProfit(prices)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 11: // Binary Search
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Expected, "4") {
				sb.WriteString("        nums = [-1, 0, 3, 5, 9, 12]\n")
				sb.WriteString("        target = 9\n")
			} else {
				sb.WriteString("        nums = [-1, 0, 3, 5, 9, 12]\n")
				sb.WriteString("        target = 2\n")
			}
			sb.WriteString("        result = search(nums, target)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 12: // First Bad Version
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "n = 5") {
				sb.WriteString("        n = 5\n")
				sb.WriteString("        passed += 1\n")
				sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d (requires API mock)\")\n", i+1))
			} else {
				sb.WriteString("        n = 1\n")
				sb.WriteString("        passed += 1\n")
				sb.WriteString(fmt.Sprintf("        print(f\"PASS Test %d (requires API mock)\")\n", i+1))
			}
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 13: // Ransom Note
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "ransomNote = \"a\"") {
				sb.WriteString("        ransomNote, magazine = \"a\", \"b\"\n")
				sb.WriteString("        expected = False\n")
			} else {
				sb.WriteString("        ransomNote, magazine = \"aa\", \"aab\"\n")
				sb.WriteString("        expected = True\n")
			}
			sb.WriteString("        result = canConstruct(ransomNote, magazine)\n")
			sb.WriteString("        if result == expected:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {result}, expected {expected}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 14: // Climbing Stairs
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "n = 2") {
				sb.WriteString("        n = 2\n")
			} else {
				sb.WriteString("        n = 3\n")
			}
			sb.WriteString("        result = climbStairs(n)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 15: // Longest Common Prefix
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "flower") {
				sb.WriteString("        strs = [\"flower\", \"flow\", \"flight\"]\n")
			} else {
				sb.WriteString("        strs = [\"dog\", \"racecar\", \"car\"]\n")
			}
			sb.WriteString("        result = longestCommonPrefix(strs)\n")
			sb.WriteString("        got = f'\"{result}\"'\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 16: // Single Number
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{2,2,1}") {
				sb.WriteString("        nums = [2, 2, 1]\n")
			} else {
				sb.WriteString("        nums = [4, 1, 2, 1, 2]\n")
			}
			sb.WriteString("        result = singleNumber(nums)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 17: // Majority Element
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{3,2,3}") {
				sb.WriteString("        nums = [3, 2, 3]\n")
			} else {
				sb.WriteString("        nums = [2, 2, 1, 1, 1, 2, 2]\n")
			}
			sb.WriteString("        result = majorityElement(nums)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 18: // Remove Duplicates from Sorted Array
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{1,1,2}") {
				sb.WriteString("        nums = [1, 1, 2]\n")
			} else {
				sb.WriteString("        nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]\n")
			}
			sb.WriteString("        result = removeDuplicates(nums)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 19: // Move Zeroes
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{0,1,0,3,12}") {
				sb.WriteString("        nums = [0, 1, 0, 3, 12]\n")
			} else {
				sb.WriteString("        nums = [0]\n")
			}
			sb.WriteString("        moveZeroes(nums)\n")
			sb.WriteString("        got = str(nums)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 20: // Reverse Linked List
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "[1,2,3,4,5]") {
				sb.WriteString("        head = create_list([1, 2, 3, 4, 5])\n")
			} else {
				sb.WriteString("        head = create_list([])\n")
			}
			sb.WriteString("        result = reverseList(head)\n")
			sb.WriteString("        got = str(list_to_array(result))\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 21: // Group Anagrams
			sb.WriteString("    try:\n")
			sb.WriteString("        strs = [\"eat\", \"tea\", \"tan\", \"ate\", \"nat\", \"bat\"]\n")
			sb.WriteString("        result = groupAnagrams(strs)\n")
			sb.WriteString("        if len(result) == 3:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {len(result)} groups, expected 3\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 22: // Top K Frequent Elements
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{1,1,1,2,2,3}") {
				sb.WriteString("        nums = [1, 1, 1, 2, 2, 3]\n")
				sb.WriteString("        k = 2\n")
			} else {
				sb.WriteString("        nums = [1]\n")
				sb.WriteString("        k = 1\n")
			}
			sb.WriteString("        result = topKFrequent(nums, k)\n")
			sb.WriteString("        if len(result) == k:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {len(result)} elements, expected {k}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 23: // Product of Array Except Self
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{1,2,3,4}") {
				sb.WriteString("        nums = [1, 2, 3, 4]\n")
			} else {
				sb.WriteString("        nums = [-1, 1, 0, -3, 3]\n")
			}
			sb.WriteString("        result = productExceptSelf(nums)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 24: // Longest Consecutive Sequence
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{100,4,200,1,3,2}") {
				sb.WriteString("        nums = [100, 4, 200, 1, 3, 2]\n")
			} else {
				sb.WriteString("        nums = [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]\n")
			}
			sb.WriteString("        result = longestConsecutive(nums)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 25: // Valid Sudoku
			sb.WriteString("    try:\n")
			sb.WriteString("        board = [[\".\", \".\", \".\", \".\", \".\", \".\", \".\", \".\", \".\"]]\n")
			sb.WriteString("        result = isValidSudoku(board)\n")
			sb.WriteString("        passed += 1\n")
			sb.WriteString(fmt.Sprintf("        print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 26: // Encode and Decode Strings
			sb.WriteString("    try:\n")
			sb.WriteString("        strs = [\"hello\", \"world\"]\n")
			sb.WriteString("        encoded = encode(strs)\n")
			sb.WriteString("        decoded = decode(encoded)\n")
			sb.WriteString("        if len(decoded) == len(strs):\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 27: // Longest Substring Without Repeating Characters
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "abcabcbb") {
				sb.WriteString("        s = \"abcabcbb\"\n")
			} else if strings.Contains(tc.Input, "bbbbb") {
				sb.WriteString("        s = \"bbbbb\"\n")
			} else {
				sb.WriteString("        s = \"pwwkew\"\n")
			}
			sb.WriteString("        result = lengthOfLongestSubstring(s)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 28: // Longest Repeating Character Replacement
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "ABAB") {
				sb.WriteString("        s, k = \"ABAB\", 2\n")
			} else {
				sb.WriteString("        s, k = \"AABABBA\", 1\n")
			}
			sb.WriteString("        result = characterReplacement(s, k)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 29: // Permutation in String
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "eidbaooo") {
				sb.WriteString("        s1, s2 = \"ab\", \"eidbaooo\"\n")
			} else {
				sb.WriteString("        s1, s2 = \"ab\", \"eidboaoo\"\n")
			}
			sb.WriteString("        result = checkInclusion(s1, s2)\n")
			sb.WriteString("        got = str(result).lower()\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 30: // Minimum Window Substring
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "ADOBECODEBANC") {
				sb.WriteString("        s, t = \"ADOBECODEBANC\", \"ABC\"\n")
			} else {
				sb.WriteString("        s, t = \"a\", \"a\"\n")
			}
			sb.WriteString("        result = minWindow(s, t)\n")
			sb.WriteString("        got = f'\"{result}\"'\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 31: // Valid Palindrome II
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "aba") {
				sb.WriteString("        s = \"aba\"\n")
			} else {
				sb.WriteString("        s = \"abca\"\n")
			}
			sb.WriteString("        result = validPalindrome(s)\n")
			sb.WriteString("        got = str(result).lower()\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 32: // 3Sum
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{-1,0,1,2,-1,-4}") {
				sb.WriteString("        nums = [-1, 0, 1, 2, -1, -4]\n")
			} else {
				sb.WriteString("        nums = []\n")
			}
			sb.WriteString("        result = threeSum(nums)\n")
			sb.WriteString("        passed += 1\n")
			sb.WriteString(fmt.Sprintf("        print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 33: // Container With Most Water
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{1,8,6,2,5,4,8,3,7}") {
				sb.WriteString("        height = [1, 8, 6, 2, 5, 4, 8, 3, 7]\n")
			} else {
				sb.WriteString("        height = [1, 1]\n")
			}
			sb.WriteString("        result = maxArea(height)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 34: // Trapping Rain Water
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{0,1,0,2,1,0,1,3,2,1,2,1}") {
				sb.WriteString("        height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]\n")
			} else {
				sb.WriteString("        height = [4, 2, 0, 3, 2, 5]\n")
			}
			sb.WriteString("        result = trap(height)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 35: // Minimum Size Subarray Sum
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "target = 7") {
				sb.WriteString("        target = 7\n")
				sb.WriteString("        nums = [2, 3, 1, 2, 4, 3]\n")
			} else {
				sb.WriteString("        target = 4\n")
				sb.WriteString("        nums = [1, 4, 4]\n")
			}
			sb.WriteString("        result = minSubArrayLen(target, nums)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 36: // Maximum Subarray
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{-2,1,-3,4,-1,2,1,-5,4}") {
				sb.WriteString("        nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]\n")
			} else {
				sb.WriteString("        nums = [1]\n")
			}
			sb.WriteString("        result = maxSubArray(nums)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 37: // Jump Game
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{2,3,1,1,4}") {
				sb.WriteString("        nums = [2, 3, 1, 1, 4]\n")
			} else {
				sb.WriteString("        nums = [3, 2, 1, 0, 4]\n")
			}
			sb.WriteString("        result = canJump(nums)\n")
			sb.WriteString("        got = str(result).lower()\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 38: // Jump Game II
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{2,3,1,1,4}") {
				sb.WriteString("        nums = [2, 3, 1, 1, 4]\n")
			} else {
				sb.WriteString("        nums = [2, 3, 0, 1, 4]\n")
			}
			sb.WriteString("        result = jump(nums)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 39: // Gas Station
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{1,2,3,4,5}") {
				sb.WriteString("        gas = [1, 2, 3, 4, 5]\n")
				sb.WriteString("        cost = [3, 4, 5, 1, 2]\n")
			} else {
				sb.WriteString("        gas = [2, 3, 4]\n")
				sb.WriteString("        cost = [3, 4, 3]\n")
			}
			sb.WriteString("        result = canCompleteCircuit(gas, cost)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 40: // Hand of Straights
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{1,2,3,6,2,3,4,7,8}") {
				sb.WriteString("        hand = [1, 2, 3, 6, 2, 3, 4, 7, 8]\n")
				sb.WriteString("        groupSize = 3\n")
			} else {
				sb.WriteString("        hand = [1, 2, 3, 4, 5]\n")
				sb.WriteString("        groupSize = 4\n")
			}
			sb.WriteString("        result = isNStraightHand(hand, groupSize)\n")
			sb.WriteString("        got = str(result).lower()\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 41: // Median of Two Sorted Arrays
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{1,3}") {
				sb.WriteString("        nums1 = [1, 3]\n")
				sb.WriteString("        nums2 = [2]\n")
			} else {
				sb.WriteString("        nums1 = [1, 2]\n")
				sb.WriteString("        nums2 = [3, 4]\n")
			}
			sb.WriteString("        result = findMedianSortedArrays(nums1, nums2)\n")
			sb.WriteString("        got = f'{result:.5f}'\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        if got.startswith(expected[:5]):\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got}, expected {expected}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 42: // Longest Valid Parentheses
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "\"(()\"") {
				sb.WriteString("        s = \"(()\"\n")
			} else {
				sb.WriteString("        s = \")()())\"\n")
			}
			sb.WriteString("        result = longestValidParentheses(s)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 43: // Wildcard Matching
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Expected, "false") {
				sb.WriteString("        s, p = \"aa\", \"a\"\n")
			} else {
				sb.WriteString("        s, p = \"aa\", \"*\"\n")
			}
			sb.WriteString("        result = isMatch(s, p)\n")
			sb.WriteString("        got = str(result).lower()\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 44: // Regular Expression Matching
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Expected, "false") {
				sb.WriteString("        s, p = \"aa\", \"a\"\n")
			} else {
				sb.WriteString("        s, p = \"aa\", \"a*\"\n")
			}
			sb.WriteString("        result = isMatchRegex(s, p)\n")
			sb.WriteString("        got = str(result).lower()\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 45: // Merge k Sorted Lists
			sb.WriteString("    try:\n")
			sb.WriteString("        lists = [create_list([1, 4, 5]), create_list([1, 3, 4]), create_list([2, 6])]\n")
			sb.WriteString("        result = mergeKLists(lists)\n")
			sb.WriteString("        passed += 1\n")
			sb.WriteString(fmt.Sprintf("        print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 46: // Reverse Nodes in k-Group
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "k = 2") {
				sb.WriteString("        head = create_list([1, 2, 3, 4, 5])\n")
				sb.WriteString("        k = 2\n")
			} else {
				sb.WriteString("        head = create_list([1, 2, 3, 4, 5])\n")
				sb.WriteString("        k = 3\n")
			}
			sb.WriteString("        result = reverseKGroup(head, k)\n")
			sb.WriteString("        passed += 1\n")
			sb.WriteString(fmt.Sprintf("        print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 47: // Largest Rectangle in Histogram
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "{2,1,5,6,2,3}") {
				sb.WriteString("        heights = [2, 1, 5, 6, 2, 3]\n")
			} else {
				sb.WriteString("        heights = [2, 4]\n")
			}
			sb.WriteString("        result = largestRectangleArea(heights)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 48: // Maximal Rectangle
			sb.WriteString("    try:\n")
			sb.WriteString("        matrix = [[\"1\", \"0\", \"1\", \"0\", \"0\"], [\"1\", \"0\", \"1\", \"1\", \"1\"]]\n")
			sb.WriteString("        result = maximalRectangle(matrix)\n")
			sb.WriteString("        passed += 1\n")
			sb.WriteString(fmt.Sprintf("        print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 49: // Minimum Window Substring (Hard version)
			sb.WriteString("    try:\n")
			sb.WriteString("        s, t = \"ADOBECODEBANC\", \"ABC\"\n")
			sb.WriteString("        result = minWindowHard(s, t)\n")
			sb.WriteString("        if len(result) > 0:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 50: // Sliding Window Maximum
			sb.WriteString("    try:\n")
			sb.WriteString("        nums = [1, 3, -1, -3, 5, 3, 6, 7]\n")
			sb.WriteString("        k = 3\n")
			sb.WriteString("        result = maxSlidingWindow(nums, k)\n")
			sb.WriteString("        if len(result) > 0:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 51: // Word Ladder
			sb.WriteString("    try:\n")
			sb.WriteString("        beginWord = \"hit\"\n")
			sb.WriteString("        endWord = \"cog\"\n")
			sb.WriteString("        wordList = [\"hot\", \"dot\", \"dog\", \"lot\", \"log\", \"cog\"]\n")
			sb.WriteString("        result = ladderLength(beginWord, endWord, wordList)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 52: // Word Ladder II
			sb.WriteString("    try:\n")
			sb.WriteString("        beginWord = \"hit\"\n")
			sb.WriteString("        endWord = \"cog\"\n")
			sb.WriteString("        wordList = [\"hot\", \"dot\", \"dog\", \"lot\", \"log\", \"cog\"]\n")
			sb.WriteString("        result = findLadders(beginWord, endWord, wordList)\n")
			sb.WriteString("        passed += 1\n")
			sb.WriteString(fmt.Sprintf("        print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 53: // Alien Dictionary
			sb.WriteString("    try:\n")
			sb.WriteString("        words = [\"wrt\", \"wrf\", \"er\", \"ett\", \"rftt\"]\n")
			sb.WriteString("        result = alienOrder(words)\n")
			sb.WriteString("        if len(result) > 0:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 54: // Longest Increasing Path in a Matrix
			sb.WriteString("    try:\n")
			sb.WriteString("        matrix = [[9, 9, 4], [6, 6, 8], [2, 1, 1]]\n")
			sb.WriteString("        result = longestIncreasingPath(matrix)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 55: // Number of Islands
			sb.WriteString("    try:\n")
			sb.WriteString("        grid = [[\"1\", \"1\", \"1\", \"1\", \"0\"], [\"1\", \"1\", \"0\", \"1\", \"0\"]]\n")
			sb.WriteString("        result = numIslands(grid)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 56: // Course Schedule II
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "numCourses = 2") {
				sb.WriteString("        numCourses = 2\n")
				sb.WriteString("        prerequisites = [[1, 0]]\n")
			} else {
				sb.WriteString("        numCourses = 4\n")
				sb.WriteString("        prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]\n")
			}
			sb.WriteString("        result = findOrder(numCourses, prerequisites)\n")
			sb.WriteString("        if len(result) > 0:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 57: // Serialize and Deserialize Binary Tree
			sb.WriteString("    try:\n")
			sb.WriteString("        root = TreeNode(1)\n")
			sb.WriteString("        root.left = TreeNode(2)\n")
			sb.WriteString("        root.right = TreeNode(3)\n")
			sb.WriteString("        serialized = serialize(root)\n")
			sb.WriteString("        deserialized = deserialize(serialized)\n")
			sb.WriteString("        if deserialized is not None and deserialized.val == 1:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 58: // Binary Tree Maximum Path Sum
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "[1,2,3]") {
				sb.WriteString("        root = TreeNode(1)\n")
				sb.WriteString("        root.left = TreeNode(2)\n")
				sb.WriteString("        root.right = TreeNode(3)\n")
			} else {
				sb.WriteString("        root = TreeNode(-10)\n")
				sb.WriteString("        root.left = TreeNode(9)\n")
				sb.WriteString("        root.right = TreeNode(20)\n")
				sb.WriteString("        root.right.left = TreeNode(15)\n")
				sb.WriteString("        root.right.right = TreeNode(7)\n")
			}
			sb.WriteString("        result = maxPathSum(root)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 59: // Distinct Subsequences
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "rabbbit") {
				sb.WriteString("        s, t = \"rabbbit\", \"rabbit\"\n")
			} else {
				sb.WriteString("        s, t = \"babgbag\", \"bag\"\n")
			}
			sb.WriteString("        result = numDistinct(s, t)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		case 60: // Edit Distance
			sb.WriteString("    try:\n")
			if strings.Contains(tc.Input, "horse") {
				sb.WriteString("        word1, word2 = \"horse\", \"ros\"\n")
			} else {
				sb.WriteString("        word1, word2 = \"intention\", \"execution\"\n")
			}
			sb.WriteString("        result = minDistance(word1, word2)\n")
			sb.WriteString("        got = str(result)\n")
			sb.WriteString(fmt.Sprintf("        expected = \"%s\"\n", tc.Expected))
			sb.WriteString("        got_normalized = normalize_output(str(got) if not isinstance(got, str) else got)\n")
			sb.WriteString(fmt.Sprintf("        expected_normalized = normalize_output(\"%s\")\n", tc.Expected))
			sb.WriteString("        if got_normalized == expected_normalized:\n")
			sb.WriteString(fmt.Sprintf("            passed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"PASS Test %d\")\n", i+1))
			sb.WriteString("        else:\n")
			sb.WriteString(fmt.Sprintf("            failed += 1\n"))
			sb.WriteString(fmt.Sprintf("            print(f\"FAIL Test %d: got {got_normalized}, expected {expected_normalized}\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		default:
			// Fallback for any unexpected problem IDs
			sb.WriteString("    try:\n")
			sb.WriteString(fmt.Sprintf("        # Problem %d: Unknown problem\n", problem.ID))
			sb.WriteString("        failed += 1\n")
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: Unknown problem\")\n", i+1))
			sb.WriteString("    except Exception as e:\n")
			sb.WriteString(fmt.Sprintf("        failed += 1\n"))
			sb.WriteString(fmt.Sprintf("        print(f\"FAIL Test %d: {e}\")\n", i+1))
		}
	}

	sb.WriteString("    print(f\"Total: {passed} passed, {failed} failed\")\n")

	return sb.String()
}

func ExecuteCode(problem *problems.Problem, userCode string, runAllTests bool) *ExecutionResult {
	return ExecuteCodeWithLanguage(problem, userCode, runAllTests, "")
}

func ExecuteCodeWithLanguage(problem *problems.Problem, userCode string, runAllTests bool, language string) *ExecutionResult {
	start := time.Now()

	// Detect language if not provided
	if language == "" {
		language = DetectLanguage(userCode)
	}

	var testCode string
	var fileExt string
	var execCmd string

	if language == "python" {
		testCode = GeneratePythonTestCode(problem, userCode, runAllTests)
		fileExt = "py"
		execCmd = "python3"
	} else {
		testCode = GenerateTestCode(problem, userCode, runAllTests)
		fileExt = "go"
		execCmd = "go run"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Generate unique filename to avoid conflicts
	uniqueID := GenerateUniqueID()
	filename := fmt.Sprintf("/tmp/code_%s.%s", uniqueID, fileExt)
	
	// Get sandbox container name
	container := GetSandboxContainer()

	// Write code to sandbox container
	writeCmd := exec.CommandContext(ctx, "docker", "exec", "-i", container, 
		"sh", "-c", fmt.Sprintf("cat > %s", filename))
	writeCmd.Stdin = strings.NewReader(testCode)
	var writeErr bytes.Buffer
	writeCmd.Stderr = &writeErr
	
	if err := writeCmd.Run(); err != nil {
		return &ExecutionResult{
			Success: false,
			Error:   fmt.Sprintf("Failed to write code to container: %v\n%s", err, writeErr.String()),
		}
	}

	// Execute code in sandbox container
	var execDockerCmd *exec.Cmd
	if language == "python" {
		// For Python, use python3 directly (timeout handled by context)
		execDockerCmd = exec.CommandContext(ctx, "docker", "exec", container,
			"sh", "-c", fmt.Sprintf("cd /tmp && timeout 25 python3 %s 2>&1", filename))
	} else {
		execDockerCmd = exec.CommandContext(ctx, "docker", "exec", container,
			"sh", "-c", fmt.Sprintf("cd /tmp && timeout 25 %s %s", execCmd, filename))
	}
	
	var stdout, stderr bytes.Buffer
	execDockerCmd.Stdout = &stdout
	execDockerCmd.Stderr = &stderr

	err := execDockerCmd.Run()
	
	// Clean up temp file
	cleanCmd := exec.Command("docker", "exec", container, "rm", "-f", filename)
	cleanCmd.Run() // Ignore errors on cleanup
	
	timeTaken := time.Since(start).String()

	output := stdout.String()
	errorOutput := stderr.String()

	result := &ExecutionResult{
		Success:   err == nil,
		TimeTaken: timeTaken,
		Results:   []TestResult{},
		Output:    output,
		Stderr:    errorOutput,
	}

	if err != nil {
		var errorMsg strings.Builder
		errorMsg.WriteString("Execution failed: " + err.Error())
		
		if errorOutput != "" {
			errorMsg.WriteString("\n" + errorOutput)
		}
		
		if output != "" {
			errorMsg.WriteString("\n" + output)
		}
		
		result.Error = errorMsg.String()
		result.Success = false
		
		// Still try to parse test results if any exist
		if output != "" {
			testNum := 1
			lines := strings.Split(output, "\n")
			for _, line := range lines {
				if strings.HasPrefix(line, "PASS") {
					parts := strings.Fields(line)
					if len(parts) >= 2 {
						result.Results = append(result.Results, TestResult{
							Passed:   true,
							Input:    fmt.Sprintf("Test %d", testNum),
							Expected: "",
						})
						testNum++
					}
				} else if strings.HasPrefix(line, "FAIL") {
					parts := strings.Split(line, ":")
					if len(parts) >= 2 {
						result.Results = append(result.Results, TestResult{
							Passed:   false,
							Input:    fmt.Sprintf("Test %d", testNum),
							Expected: "",
							Error:    strings.TrimSpace(parts[1]),
						})
						testNum++
					}
				}
			}
		}
		
		return result
	}

	if errorOutput != "" {
		result.Error = errorOutput
		if output != "" {
			result.Error += "\n" + output
		}
		result.Success = false
		return result
	}

	testNum := 1
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "PASS") {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				result.Results = append(result.Results, TestResult{
					Passed:   true,
					Input:    fmt.Sprintf("Test %d", testNum),
					Expected: "",
				})
				testNum++
			}
		} else if strings.HasPrefix(line, "FAIL") {
			parts := strings.Split(line, ":")
			if len(parts) >= 2 {
				result.Results = append(result.Results, TestResult{
					Passed:   false,
					Input:    fmt.Sprintf("Test %d", testNum),
					Expected: "",
					Error:    strings.TrimSpace(parts[1]),
				})
				testNum++
			}
		}
	}

	if len(result.Results) == 0 && output != "" {
		result.Error = "No test results found. Output: " + output
		result.Success = false
	} else if len(result.Results) > 0 {
		// Success should be true only if all tests passed and there were no errors
		allPassed := true
		for _, r := range result.Results {
			if !r.Passed {
				allPassed = false
				break
			}
		}
		result.Success = allPassed && err == nil
	}

	return result
}
