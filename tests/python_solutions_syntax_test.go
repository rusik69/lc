package tests

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/rusik69/lc/internal/executor"
	"github.com/rusik69/lc/internal/problems"
)

// TestPythonSolutionsSyntax verifies every PythonSolution in the codebase
// is syntactically valid Python by compiling the solution itself AND the
// generated test harness.
func TestPythonSolutionsSyntax(t *testing.T) {
	allProbs := problems.GetAllProblems()

	var (
		totalWithPython int
		solutionErrors  int
		harnessErrors   int
	)

	for _, problem := range allProbs {
		if problem.PythonSolution == "" {
			continue
		}
		totalWithPython++

		t.Run(fmt.Sprintf("Problem_%d_%s", problem.ID, problem.Title), func(t *testing.T) {
			// 1. Check the raw PythonSolution compiles
			solFile := filepath.Join(t.TempDir(), "solution.py")
			content := "from typing import List, Optional\n" + problem.PythonSolution + "\n"
			if err := os.WriteFile(solFile, []byte(content), 0644); err != nil {
				t.Fatalf("Failed to write temp file: %v", err)
			}
			cmd := exec.Command("python3", "-c",
				fmt.Sprintf("compile(open(%q).read(), %q, 'exec')", solFile, solFile))
			if output, err := cmd.CombinedOutput(); err != nil {
				solutionErrors++
				t.Errorf("Solution syntax error for problem %d (%s):\n%s",
					problem.ID, problem.Title, string(output))
				return
			}

			// 2. Check the generated test harness compiles
			testCode := executor.GeneratePythonTestCode(&problem, problem.PythonSolution, false)
			harnessFile := filepath.Join(t.TempDir(), "harness.py")
			if err := os.WriteFile(harnessFile, []byte(testCode), 0644); err != nil {
				t.Fatalf("Failed to write temp file: %v", err)
			}
			cmd = exec.Command("python3", "-c",
				fmt.Sprintf("compile(open(%q).read(), %q, 'exec')", harnessFile, harnessFile))
			if output, err := cmd.CombinedOutput(); err != nil {
				harnessErrors++
				t.Errorf("Test harness syntax error for problem %d (%s):\n%s",
					problem.ID, problem.Title, string(output))
			} else {
				t.Logf("✓ Problem %d (%s) - valid syntax", problem.ID, problem.Title)
			}
		})
	}

	t.Logf("\nPython solution coverage: %d out of %d problems have PythonSolution",
		totalWithPython, len(allProbs))
	if solutionErrors > 0 {
		t.Errorf("%d Python solutions have syntax errors", solutionErrors)
	}
	if harnessErrors > 0 {
		t.Errorf("%d generated test harnesses have syntax errors (likely executor bugs)", harnessErrors)
	}
}

// TestPythonSolutionsCompleteness reports which problems are missing PythonSolution.
func TestPythonSolutionsCompleteness(t *testing.T) {
	allProbs := problems.GetAllProblems()

	missing := make([]problems.Problem, 0)
	for _, p := range allProbs {
		if p.PythonSolution == "" {
			missing = append(missing, p)
		}
	}

	t.Logf("Total problems: %d", len(allProbs))
	t.Logf("With PythonSolution: %d", len(allProbs)-len(missing))
	t.Logf("Missing PythonSolution: %d", len(missing))

	if len(missing) > 0 {
		t.Logf("Problems missing Python solution:")
		for _, p := range missing {
			t.Logf("  ID %3d [%s] %s", p.ID, p.Difficulty, p.Title)
		}
	}

	if len(missing) > len(allProbs)/2 {
		t.Errorf("More than half of problems are missing PythonSolution (%d missing)", len(missing))
	}
}

// DumpPythonGeneratedCode writes the generated Python test harness for a specific problem
// to stdout so we can inspect it during development.
func DumpPythonGeneratedCode(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping dump test in short mode")
	}
	// Dump problem 34 for debugging
	problem := problems.GetProblem(34)
	if problem == nil {
		t.Fatal("Problem 34 not found")
	}
	code := executor.GeneratePythonTestCode(problem, problem.PythonSolution, false)
	t.Logf("Generated Python test code for problem %d (%s):\n%s", problem.ID, problem.Title, code)
}
