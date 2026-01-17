package problems

import "sync"

// TestCase represents a single test case for a problem
type TestCase struct {
	Input    string
	Expected string
}

// Problem represents a coding problem with its test cases and solution
type Problem struct {
	ID          int
	Title       string
	Description string
	Difficulty  string // "Easy", "Medium", "Hard"
	Signature   string
	TestCases   []TestCase
	Solution    string
	Explanation string
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
