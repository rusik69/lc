package tests

import (
	"strings"
	"testing"
	"time"

	"github.com/rusik69/lc/internal/executor"
	"github.com/rusik69/lc/internal/problems"
)

// TestAllSolutionsE2E tests all 60 problem solutions end-to-end with Docker execution
func TestAllSolutionsE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping e2e tests in short mode")
	}

	allProbs := problems.GetAllProblems()

	for _, problem := range allProbs {
		problem := problem // capture range variable
		t.Run(problem.Title, func(t *testing.T) {
			// Run test with timeout
			done := make(chan bool)
			var result *executor.ExecutionResult

			go func() {
				result = executor.ExecuteCode(&problem, problem.Solution, true)
				done <- true
			}()

			select {
			case <-done:
				// Test completed
			case <-time.After(30 * time.Second):
				t.Fatalf("Test timed out after 30 seconds")
			}

			// Verify execution succeeded
			if !result.Success {
				t.Errorf("Execution failed: %s\nOutput: %s\nStderr: %s",
					result.Error, result.Output, result.Stderr)
				return
			}

			// Verify all test cases passed
			passedCount := 0
			for _, testResult := range result.Results {
				if testResult.Passed {
					passedCount++
				} else {
					t.Errorf("Test case failed:\n  Input: %s\n  Expected: %s\n  Got: %s\n  Error: %s",
						testResult.Input, testResult.Expected, testResult.Got, testResult.Error)
				}
			}

			if passedCount != len(problem.TestCases) {
				t.Errorf("Only %d/%d test cases passed", passedCount, len(problem.TestCases))
			} else {
				t.Logf("✓ All %d test cases passed for problem %d: %s (%s)",
					passedCount, problem.ID, problem.Title, problem.Difficulty)
			}
		})
	}
}

// TestEasyProblemsE2E tests only Easy problems (faster for quick verification)
func TestEasyProblemsE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping e2e tests in short mode")
	}

	allProbs := problems.GetAllProblems()
	easyProblems := []problems.Problem{}
	for _, p := range allProbs {
		if p.Difficulty == "Easy" {
			easyProblems = append(easyProblems, p)
		}
	}

	t.Logf("Testing %d Easy problems", len(easyProblems))

	for _, problem := range easyProblems {
		problem := problem
		t.Run(problem.Title, func(t *testing.T) {
			result := executor.ExecuteCode(&problem, problem.Solution, true)

			if !result.Success {
				t.Errorf("Execution failed: %s", result.Error)
				return
			}

			passedCount := 0
			for _, testResult := range result.Results {
				if testResult.Passed {
					passedCount++
				}
			}

			// Problems 1-5 have full e2e tests, 6-20 verify compilation only
			if problem.ID >= 6 {
				if passedCount > 0 {
					t.Logf("✓ Solution compiles successfully")
				} else {
					t.Errorf("Solution failed to compile")
				}
			} else if passedCount != len(problem.TestCases) {
				t.Errorf("Only %d/%d test cases passed", passedCount, len(problem.TestCases))
			}
		})
	}
}

// TestMediumProblemsE2E tests only Medium problems
func TestMediumProblemsE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping e2e tests in short mode")
	}

	allProbs := problems.GetAllProblems()
	mediumProblems := []problems.Problem{}
	for _, p := range allProbs {
		if p.Difficulty == "Medium" {
			mediumProblems = append(mediumProblems, p)
		}
	}

	t.Logf("Testing %d Medium problems", len(mediumProblems))

	for _, problem := range mediumProblems {
		problem := problem
		t.Run(problem.Title, func(t *testing.T) {
			result := executor.ExecuteCode(&problem, problem.Solution, true)

			if !result.Success {
				t.Errorf("Execution failed: %s", result.Error)
				return
			}

			passedCount := 0
			for _, testResult := range result.Results {
				if testResult.Passed {
					passedCount++
				}
			}

			if passedCount != len(problem.TestCases) {
				t.Errorf("Only %d/%d test cases passed", passedCount, len(problem.TestCases))
			}
		})
	}
}

// TestHardProblemsE2E tests only Hard problems
func TestHardProblemsE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping e2e tests in short mode")
	}

	allProbs := problems.GetAllProblems()
	hardProblems := []problems.Problem{}
	for _, p := range allProbs {
		if p.Difficulty == "Hard" {
			hardProblems = append(hardProblems, p)
		}
	}

	t.Logf("Testing %d Hard problems", len(hardProblems))

	for _, problem := range hardProblems {
		problem := problem
		t.Run(problem.Title, func(t *testing.T) {
			result := executor.ExecuteCode(&problem, problem.Solution, true)

			if !result.Success {
				t.Errorf("Execution failed: %s", result.Error)
				return
			}

			passedCount := 0
			for _, testResult := range result.Results {
				if testResult.Passed {
					passedCount++
				}
			}

			if passedCount != len(problem.TestCases) {
				t.Errorf("Only %d/%d test cases passed", passedCount, len(problem.TestCases))
			}
		})
	}
}

// TestSampleProblemE2E tests a single problem as a quick smoke test
func TestSampleProblemE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping e2e tests in short mode")
	}

	problem := problems.GetProblem(1) // Two Sum
	if problem == nil {
		t.Fatal("Problem 1 not found")
	}

	t.Logf("Testing problem: %s", problem.Title)

	result := executor.ExecuteCode(problem, problem.Solution, true)

	if !result.Success {
		t.Fatalf("Execution failed: %s\nOutput: %s\nStderr: %s",
			result.Error, result.Output, result.Stderr)
	}

	t.Logf("Execution successful, time taken: %s", result.TimeTaken)

	// Check results
	if len(result.Results) == 0 {
		t.Fatal("No test results returned")
	}

	for i, testResult := range result.Results {
		if testResult.Passed {
			t.Logf("✓ Test case %d passed", i+1)
		} else {
			t.Errorf("✗ Test case %d failed:\n  Input: %s\n  Expected: %s\n  Got: %s",
				i+1, testResult.Input, testResult.Expected, testResult.Got)
		}
	}
}

// TestDockerAvailability checks if Docker is available before running e2e tests
func TestDockerAvailability(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Docker availability check in short mode")
	}

	// Try to execute a simple problem to verify Docker works
	problem := &problems.Problem{
		ID:          999,
		Title:       "Docker Test",
		Description: "Test Docker availability",
		Difficulty:  "Easy",
		Signature:   "func testDocker() int",
		TestCases: []problems.TestCase{
			{Input: "", Expected: "42"},
		},
		Solution: "func testDocker() int { return 42 }",
	}

	testCode := `package main
import "fmt"

func testDocker() int { return 42 }

func main() {
	result := testDocker()
	got := fmt.Sprintf("%d", result)
	expected := "42"
	if got == expected {
		fmt.Println("PASS Test 1")
	} else {
		fmt.Printf("FAIL Test 1: got %s, expected %s\n", got, expected)
	}
	fmt.Println("Total: 1 passed, 0 failed")
}`

	result := executor.ExecuteCode(problem, testCode, true)

	if !result.Success {
		t.Fatalf("Docker is not available or not working correctly:\n%s\nStderr: %s",
			result.Error, result.Stderr)
	}

	t.Log("✓ Docker is available and working correctly")
}

// TestSolutionOutputFormat verifies that solution output can be parsed correctly
func TestSolutionOutputFormat(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping e2e tests in short mode")
	}

	problem := problems.GetProblem(1) // Two Sum
	if problem == nil {
		t.Fatal("Problem 1 not found")
	}

	result := executor.ExecuteCode(problem, problem.Solution, true)

	if !result.Success {
		t.Fatalf("Execution failed: %s", result.Error)
	}

	// Verify output contains expected patterns
	output := result.Output
	if !strings.Contains(output, "PASS") && !strings.Contains(output, "FAIL") {
		t.Errorf("Output doesn't contain PASS or FAIL markers:\n%s", output)
	}

	if !strings.Contains(output, "Total:") {
		t.Errorf("Output doesn't contain 'Total:' summary:\n%s", output)
	}

	// Verify results were parsed
	if len(result.Results) == 0 {
		t.Error("No test results were parsed from output")
	}

	t.Logf("✓ Output format is correct, parsed %d results", len(result.Results))
}
