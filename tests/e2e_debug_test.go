package tests

import (
	"bytes"
	"context"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/rusik69/lc/internal/executor"
	"github.com/rusik69/lc/internal/problems"
)

// TestDebugDockerExecution helps debug why Docker execution is failing
func TestDebugDockerExecution(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping debug test in short mode")
	}

	problem := problems.GetProblem(1)
	if problem == nil {
		t.Fatal("Problem 1 not found")
	}

	// Generate test code
	testCode := executor.GenerateTestCode(problem, problem.Solution, true)
	
	t.Logf("=== Generated Test Code ===\n%s\n=== End Code ===\n", testCode)

	// Try to run it directly with Docker
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "docker", "run", "--rm",
		"--network", "none",
		"--memory", "256m",
		"--cpus", "1",
		"--pids-limit", "50",
		"--read-only",
		"--tmpfs", "/tmp:rw,nosuid,size=100m",
		"--tmpfs", "/go:rw,nosuid,size=150m",
		"--tmpfs", "/root/.cache:rw,nosuid,size=100m",
		"--security-opt", "no-new-privileges:true",
		"--cap-drop", "ALL",
		"-e", "GOCACHE=/tmp/go-build",
		"-e", "GOMODCACHE=/tmp/go-mod",
		"-i", "golang:alpine", "sh", "-c", "cat > /tmp/main.go && timeout 5 go run /tmp/main.go")

	cmd.Stdin = strings.NewReader(testCode)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	t.Logf("=== Execution Result ===")
	t.Logf("Error: %v", err)
	t.Logf("=== STDOUT ===\n%s\n=== END STDOUT ===", stdout.String())
	t.Logf("=== STDERR ===\n%s\n=== END STDERR ===", stderr.String())

	if err != nil {
		t.Fatalf("Docker execution failed: %v", err)
	}
}

// TestSimpleDockerTest tests Docker with minimal code
func TestSimpleDockerTest(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping simple Docker test in short mode")
	}

	simpleCode := `package main

import "fmt"

func main() {
	fmt.Println("Hello from Docker!")
	fmt.Println("PASS Test 1")
	fmt.Println("Total: 1 passed, 0 failed")
}
`

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// Minimal Docker command for testing
	cmd := exec.CommandContext(ctx, "docker", "run", "--rm",
		"-i", "golang:alpine", "sh", "-c", "cat > /tmp/main.go && go run /tmp/main.go")

	cmd.Stdin = strings.NewReader(simpleCode)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	t.Logf("STDOUT: %s", stdout.String())
	t.Logf("STDERR: %s", stderr.String())

	if err != nil {
		t.Fatalf("Simple Docker test failed: %v", err)
	}

	if !strings.Contains(stdout.String(), "Hello from Docker!") {
		t.Error("Expected output not found")
	}

	t.Log("✓ Simple Docker test passed")
}
