package tests

import (
	"strings"
	"testing"

	"github.com/rusik69/lc/internal/executor"
	"github.com/rusik69/lc/internal/problems"
)

func TestGenerateTestCode(t *testing.T) {
	problem := problems.GetProblem(1) // Two Sum
	if problem == nil {
		t.Fatal("problems.GetProblem(1) returned nil")
	}

	userCode := `func twoSum(nums []int, target int) []int {
		m := make(map[int]int)
		for i, num := range nums {
			if j, ok := m[target-num]; ok {
				return []int{j, i}
			}
			m[num] = i
		}
		return nil
	}`

	tests := []struct {
		name        string
		runAllTests bool
		wantContain []string
	}{
		{
			name:        "Single test case",
			runAllTests: false,
			wantContain: []string{
				"package main",
				"func main()",
				"twoSum",
			},
		},
		{
			name:        "All test cases",
			runAllTests: true,
			wantContain: []string{
				"package main",
				"func main()",
				"twoSum",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := executor.GenerateTestCode(problem, userCode, tt.runAllTests)

			for _, want := range tt.wantContain {
				if !strings.Contains(got, want) {
					t.Errorf("executor.GenerateTestCode() missing expected string: %q", want)
				}
			}

			// Verify it's valid Go code structure
			if !strings.HasPrefix(got, "package main") {
				t.Error("Generated code doesn't start with 'package main'")
			}
		})
	}
}

func TestGenerateTestCodeAllProblems(t *testing.T) {
	problems := problems.GetAllProblems()

	for _, problem := range problems {
		t.Run(problem.Title, func(t *testing.T) {
			// Use the problem's solution
			code := executor.GenerateTestCode(&problem, problem.Solution, true)

			// Verify basic structure
			if !strings.Contains(code, "package main") {
				t.Error("Generated code missing 'package main'")
			}
			if !strings.Contains(code, "func main()") {
				t.Error("Generated code missing 'func main()'")
			}
			if !strings.Contains(code, problem.Solution) {
				t.Error("Generated code missing user's solution")
			}
		})
	}
}

func TestContainsDangerousPatterns(t *testing.T) {
	tests := []struct {
		name string
		code string
		want bool
	}{
		{
			name: "Safe code",
			code: `func twoSum(nums []int, target int) []int {
				return []int{0, 1}
			}`,
			want: false,
		},
		{
			name: "Code with os/exec import",
			code: `import "os/exec"
			func test() {}`,
			want: true,
		},
		{
			name: "Code with os.Exec call",
			code: `func test() {
				os.Exec("ls")
			}`,
			want: true,
		},
		{
			name: "Code with syscall import",
			code: `import "syscall"
			func test() {}`,
			want: true,
		},
		{
			name: "Code with net/http import",
			code: `import "net/http"
			func test() {}`,
			want: true,
		},
		{
			name: "Code with unsafe import",
			code: `import "unsafe"
			func test() {}`,
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// containsDangerousPatterns is a private function - tested indirectly through handlers
			got := false // Placeholder - this test should be removed or moved to handlers tests
			if got != tt.want {
				t.Errorf("containsDangerousPatterns() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestExecutionResultStructure(t *testing.T) {
	result := &executor.ExecutionResult{
		Success:   true,
		TimeTaken: "100ms",
		Results: []executor.TestResult{
			{
				Passed:   true,
				Input:    "Test 1",
				Expected: "[0 1]",
			},
		},
	}

	if !result.Success {
		t.Error("Expected Success to be true")
	}
	if result.TimeTaken == "" {
		t.Error("Expected TimeTaken to be set")
	}
	if len(result.Results) != 1 {
		t.Errorf("Expected 1 test result, got %d", len(result.Results))
	}
}
