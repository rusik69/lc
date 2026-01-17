package problems

import (
	"strings"
	"sync"
)

// TestCase represents a single test case for a problem
type TestCase struct {
	Input    string
	Expected string
}

// Problem represents a coding problem with its test cases and solution
type Problem struct {
	ID            int
	Title         string
	Description   string
	Difficulty    string // "Easy", "Medium", "Hard"
	Signature     string
	TestCases     []TestCase
	Solution      string // Go solution
	PythonSolution string // Python solution
	Explanation   string
}

var (
	allProblems []Problem
	problemsMu  sync.RWMutex
)

func init() {
	allProblems = make([]Problem, 0, 60)
	allProblems = append(allProblems, EasyProblems...)
	allProblems = append(allProblems, MediumProblems...)
	allProblems = append(allProblems, HardProblems...)
}

// GetProblem returns a problem by its ID
func GetProblem(id int) *Problem {
	problemsMu.RLock()
	defer problemsMu.RUnlock()
	for i := range allProblems {
		if allProblems[i].ID == id {
			return &allProblems[i]
		}
	}
	return nil
}

// GetAllProblems returns all problems
func GetAllProblems() []Problem {
	problemsMu.RLock()
	defer problemsMu.RUnlock()
	result := make([]Problem, len(allProblems))
	copy(result, allProblems)
	return result
}

// GetProblemsByDifficulty returns all problems filtered by difficulty
func GetProblemsByDifficulty(difficulty string) []Problem {
	problemsMu.RLock()
	defer problemsMu.RUnlock()
	result := []Problem{}
	for i := range allProblems {
		if allProblems[i].Difficulty == difficulty {
			result = append(result, allProblems[i])
		}
	}
	return result
}

// GetEasyProblems returns all easy problems
func GetEasyProblems() []Problem {
	return GetProblemsByDifficulty("Easy")
}

// GetMediumProblems returns all medium problems
func GetMediumProblems() []Problem {
	return GetProblemsByDifficulty("Medium")
}

// GetHardProblems returns all hard problems
func GetHardProblems() []Problem {
	return GetProblemsByDifficulty("Hard")
}

// GetSolution returns the solution for a problem in the specified language
func (p *Problem) GetSolution(language string) string {
	if language == "python" && p.PythonSolution != "" {
		return p.PythonSolution
	}
	return p.Solution
}

// HasLanguageSolution checks if a problem has a solution in the specified language
func (p *Problem) HasLanguageSolution(language string) bool {
	if language == "python" {
		return p.PythonSolution != ""
	}
	return p.Solution != ""
}

// GetSignature returns the signature for a problem in the specified language
func (p *Problem) GetSignature(language string) string {
	if language == "python" {
		return p.goToPythonSignature(p.Signature)
	}
	return p.Signature
}

// goToPythonSignature converts a Go function signature to Python signature
func (p *Problem) goToPythonSignature(goSig string) string {
	// Remove 'func ' prefix
	sig := goSig
	if len(sig) >= 5 && sig[:5] == "func " {
		sig = sig[5:]
	}
	
	// Extract function name (everything before first '(')
	funcNameEnd := 0
	for i, r := range sig {
		if r == '(' {
			funcNameEnd = i
			break
		}
	}
	if funcNameEnd == 0 {
		return "def solution():\n    pass"
	}
	
	funcName := strings.TrimSpace(sig[:funcNameEnd])
	
	// Extract parameters
	paramsStart := funcNameEnd
	paramsEnd := strings.Index(sig[paramsStart:], ")")
	if paramsEnd == -1 {
		return "def " + funcName + "():\n    pass"
	}
	paramsEnd += paramsStart
	
	paramsStr := strings.TrimSpace(sig[paramsStart+1 : paramsEnd])
	var params []string
	
	if paramsStr != "" {
		paramParts := strings.Split(paramsStr, ",")
		for _, part := range paramParts {
			part = strings.TrimSpace(part)
			fields := strings.Fields(part)
			if len(fields) >= 2 {
				paramName := fields[0]
				paramType := fields[1]
				
				// Convert Go types to Python types (order matters - do complex types first)
				pyType := paramType
				pyType = strings.ReplaceAll(pyType, "[]*ListNode", "List[Optional[ListNode]]")
				pyType = strings.ReplaceAll(pyType, "[]*TreeNode", "List[Optional[TreeNode]]")
				pyType = strings.ReplaceAll(pyType, "[]int", "List[int]")
				pyType = strings.ReplaceAll(pyType, "[]string", "List[str]")
				pyType = strings.ReplaceAll(pyType, "[]byte", "List[str]")
				pyType = strings.ReplaceAll(pyType, "*ListNode", "Optional[ListNode]")
				pyType = strings.ReplaceAll(pyType, "*TreeNode", "Optional[TreeNode]")
				pyType = strings.ReplaceAll(pyType, "float64", "float")
				pyType = strings.ReplaceAll(pyType, "string", "str")
				pyType = strings.ReplaceAll(pyType, "bool", "bool")
				// Keep int as int (no change needed)
				
				params = append(params, paramName+": "+pyType)
			}
		}
	}
	
	// Extract return type
	returnType := "None"
	remaining := strings.TrimSpace(sig[paramsEnd+1:])
	if strings.Contains(remaining, "[]int") {
		returnType = "List[int]"
	} else if strings.Contains(remaining, "[]string") {
		returnType = "List[str]"
	} else if strings.Contains(remaining, "string") {
		returnType = "str"
	} else if strings.Contains(remaining, "int") {
		returnType = "int"
	} else if strings.Contains(remaining, "bool") {
		returnType = "bool"
	} else if strings.Contains(remaining, "float64") {
		returnType = "float"
	} else if strings.Contains(remaining, "*ListNode") {
		returnType = "Optional[ListNode]"
	} else if strings.Contains(remaining, "*TreeNode") {
		returnType = "Optional[TreeNode]"
	}
	
	return "def " + funcName + "(" + strings.Join(params, ", ") + ") -> " + returnType
}
