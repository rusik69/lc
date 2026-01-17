package executor

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strings"
	"time"

	"github.com/rusik69/lc/internal/problems"
)

func ExecuteCodeStream(problem *problems.Problem, userCode string, runAllTests bool, w io.Writer) *ExecutionResult {
	start := time.Now()

	testCode := GenerateTestCode(problem, userCode, runAllTests)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Generate unique filename
	uniqueID := GenerateUniqueID()
	filename := fmt.Sprintf("/tmp/code_%s.go", uniqueID)
	
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
	cmd := exec.CommandContext(ctx, "docker", "exec", container,
		"sh", "-c", fmt.Sprintf("cd /tmp && timeout 25 go run %s", filename))

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return &ExecutionResult{Success: false, Error: err.Error()}
	}

	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return &ExecutionResult{Success: false, Error: err.Error()}
	}

	if err := cmd.Start(); err != nil {
		return &ExecutionResult{Success: false, Error: err.Error()}
	}

	var output strings.Builder
	var errorOutput strings.Builder
	doneChan := make(chan bool, 2)

	// Stream stdout
	go func() {
		defer func() {
			doneChan <- true
		}()
		scanner := bufio.NewScanner(stdoutPipe)
		for scanner.Scan() {
			line := scanner.Text()
			output.WriteString(line + "\n")
			fmt.Fprintf(w, "data: %s\n\n", escapeSSE(line))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
		if err := scanner.Err(); err != nil {
			errorOutput.WriteString("Stdout scanner error: " + err.Error() + "\n")
		}
	}()

	// Stream stderr
	go func() {
		defer func() {
			doneChan <- true
		}()
		scanner := bufio.NewScanner(stderrPipe)
		for scanner.Scan() {
			line := scanner.Text()
			errorOutput.WriteString(line + "\n")
			fmt.Fprintf(w, "data: %s\n\n", escapeSSE("[STDERR] "+line))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
		if err := scanner.Err(); err != nil {
			errorOutput.WriteString("Stderr scanner error: " + err.Error() + "\n")
		}
	}()

	err = cmd.Wait()
	
	<-doneChan
	<-doneChan
	
	// Clean up temp file
	cleanCmd := exec.Command("docker", "exec", container, "rm", "-f", filename)
	cleanCmd.Run() // Ignore errors on cleanup
	
	timeTaken := time.Since(start).String()

	outputStr := output.String()
	errorStr := errorOutput.String()

	result := &ExecutionResult{
		Success:   err == nil,
		TimeTaken: timeTaken,
		Results:   []TestResult{},
		Output:    outputStr,
		Stderr:    errorStr,
	}

	// Parse test results
	testNum := 1
	lines := strings.Split(output.String(), "\n")
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

	if err != nil {
		var errorMsg strings.Builder
		errorMsg.WriteString("Execution failed: " + err.Error())
		
		if errorStr != "" {
			errorMsg.WriteString("\n" + errorStr)
		}

		if outputStr != "" {
			errorMsg.WriteString("\n" + outputStr)
		}

		result.Error = errorMsg.String()
		result.Success = false
	}

	if errorStr != "" && err == nil {
		result.Error = errorStr
		if outputStr != "" {
			result.Error += "\n" + outputStr
		}
		result.Success = false
	}

	return result
}

// escapeSSE escapes special characters for Server-Sent Events
func escapeSSE(s string) string {
	s = strings.ReplaceAll(s, "\n", "\\n")
	s = strings.ReplaceAll(s, "\r", "\\r")
	s = strings.ReplaceAll(s, "\\", "\\\\")
	return s
}
