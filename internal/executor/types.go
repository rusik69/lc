package executor

// TestResult represents the result of a single test case
type TestResult struct {
	Passed   bool
	Input    string
	Expected string
	Got      string
	Error    string
}

// ExecutionResult represents the overall result of code execution
type ExecutionResult struct {
	Success   bool
	Results   []TestResult
	Error     string
	Output    string
	Stderr    string
	TimeTaken string
}
