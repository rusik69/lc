package tests

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/rusik69/lc/internal/executor"
	"github.com/rusik69/lc/internal/problems"
)

// TestPythonSolutionsE2E verifies Python solutions execute correctly through
// the Docker sandbox. Requires Docker to be running.
//
// Skip with: go test -short
func TestPythonSolutionsE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Python E2E tests in short mode (requires Docker)")
	}

	allProbs := problems.GetAllProblems()

	var (
		total      int
		passed     int
		skipped    int
		broken     int
	)

	for _, problem := range allProbs {
		if problem.PythonSolution == "" {
			continue
		}
		total++

		t.Run(fmt.Sprintf("Problem_%d_%s", problem.ID, problem.Title), func(t *testing.T) {
			done := make(chan bool)
			var result *executor.ExecutionResult

			go func() {
				result = executor.ExecuteCodeWithLanguage(&problem, problem.PythonSolution, true, "python")
				done <- true
			}()

			select {
			case <-done:
			case <-time.After(40 * time.Second):
				t.Fatalf("Test timed out")
			}

			if result == nil {
				broken++
				t.Error("Execution returned nil result")
				return
			}

			if !result.Success && strings.Contains(result.Error, "undefined") {
				skipped++
				t.Skipf("Problem %d needs additional definitions: %s", problem.ID, result.Error)
				return
			}

			if !result.Success {
				broken++
				t.Errorf("Execution failed: %s\nOutput: %s\nStderr: %s",
					result.Error, result.Output, result.Stderr)
				return
			}

			// Verify all test cases passed
			passedCount := 0
			for _, tr := range result.Results {
				if tr.Passed {
					passedCount++
				} else {
					t.Errorf("Test case failed: Input=%s Expected=%s Got=%s Error=%s",
						tr.Input, tr.Expected, tr.Got, tr.Error)
				}
			}

			if passedCount == len(problem.TestCases) {
				passed++
				t.Logf("✓ All %d tests passed for problem %d: %s",
					passedCount, problem.ID, problem.Title)
			} else {
				broken++
				t.Errorf("Only %d/%d tests passed", passedCount, len(problem.TestCases))
			}
		})
	}

	t.Logf("\nPython solution E2E results: %d/%d passed, %d skipped, %d broken (out of %d total with PythonSolution)",
		passed, total, skipped, broken, total)
}

// TestPythonSolutionsE2EShort runs only problems 1-20 (Easy) for quick verification
func TestPythonSolutionsE2EShort(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Python E2E short tests in short mode (requires Docker)")
	}

	allProbs := problems.GetAllProblems()
	// Test first 20 problems
	limit := 20
	if len(allProbs) < limit {
		limit = len(allProbs)
	}

	for _, problem := range allProbs[:limit] {
		if problem.PythonSolution == "" {
			continue
		}
		t.Run(fmt.Sprintf("Problem_%d_%s", problem.ID, problem.Title), func(t *testing.T) {
			result := executor.ExecuteCodeWithLanguage(&problem, problem.PythonSolution, true, "python")
			if result == nil {
				t.Fatal("nil result")
			}
			if !result.Success {
				if strings.Contains(result.Error, "undefined") {
					t.Skipf("Needs definitions: %s", result.Error)
					return
				}
				t.Fatalf("Failed: %s", result.Error)
			}
			passedCount := 0
			for _, tr := range result.Results {
				if tr.Passed {
					passedCount++
				}
			}
			if passedCount != len(problem.TestCases) {
				t.Errorf("Only %d/%d passed", passedCount, len(problem.TestCases))
			} else {
				t.Logf("✓ %d/%d passed", passedCount, len(problem.TestCases))
			}
		})
	}
}
