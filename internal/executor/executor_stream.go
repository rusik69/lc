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
	return ExecuteCodeStreamWithLanguage(problem, userCode, runAllTests, w, "")
}

func ExecuteCodeStreamWithLanguage(problem *problems.Problem, userCode string, runAllTests bool, w io.Writer, language string) *ExecutionResult {
	start := time.Now()

	// Write initial message to establish stream connection
	if httpW, ok := w.(http.ResponseWriter); ok {
		fmt.Fprintf(httpW, "data: %s\n\n", escapeSSE("Preparing execution..."))
		if flusher, ok := httpW.(http.Flusher); ok {
			flusher.Flush()
		}
	}

	// Detect language if not provided
	if language == "" {
		language = DetectLanguage(userCode)
	}

	var testCode string
	var fileExt string
	var execCmd string
	var genErr error

	// Generate test code with error handling
	func() {
		defer func() {
			if r := recover(); r != nil {
				genErr = fmt.Errorf("panic during test code generation: %v", r)
				errorMsg := genErr.Error()
				if httpW, ok := w.(http.ResponseWriter); ok {
					fmt.Fprintf(httpW, "data: %s\n\n", escapeSSE("[ERROR] "+errorMsg))
					if flusher, ok := httpW.(http.Flusher); ok {
						flusher.Flush()
					}
				}
			}
		}()
		if language == "python" {
			testCode = GeneratePythonTestCode(problem, userCode, runAllTests)
			fileExt = "py"
			execCmd = "python3"
		} else {
			testCode = GenerateTestCode(problem, userCode, runAllTests)
			fileExt = "go"
			execCmd = "go run"
		}
	}()

	// If test code generation failed, return error
	if genErr != nil || testCode == "" {
		errorMsg := "Failed to generate test code"
		if genErr != nil {
			errorMsg = genErr.Error()
		}
		// Write error to stream (ignore if connection closed)
		safeWriteSSE(w, "[ERROR] "+errorMsg)
		return &ExecutionResult{
			Success: false,
			Error:   errorMsg,
			Results: []TestResult{},
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Generate unique filename
	uniqueID := GenerateUniqueID()
	filename := fmt.Sprintf("/tmp/code_%s.%s", uniqueID, fileExt)
	
	// Get sandbox container name
	container := GetSandboxContainer()

	// Check if container exists and is running
	// First try: docker ps with filter
	checkCmd := exec.CommandContext(ctx, "docker", "ps", "--format", "{{.Names}}", "--filter", "name=^"+container+"$")
	var checkErr bytes.Buffer
	checkCmd.Stderr = &checkErr
	checkOutput, err := checkCmd.Output()
	checkOutputStr := strings.TrimSpace(string(checkOutput))
	
	containerRunning := err == nil && checkOutputStr == container
	
	// Second try: docker inspect (more reliable)
	if !containerRunning {
		inspectCmd := exec.CommandContext(ctx, "docker", "inspect", "--format", "{{.State.Running}}", container)
		var inspectErrBuf bytes.Buffer
		inspectCmd.Stderr = &inspectErrBuf
		inspectOutput, inspectErr := inspectCmd.Output()
		if inspectErr == nil && strings.TrimSpace(string(inspectOutput)) == "true" {
			containerRunning = true
		} else {
			// Try to get more info about what containers are available
			listCmd := exec.CommandContext(ctx, "docker", "ps", "--format", "{{.Names}}")
			listOutput, _ := listCmd.Output()
			checkErr.WriteString(fmt.Sprintf("\nAvailable containers: %s", strings.TrimSpace(string(listOutput))))
			if inspectErrBuf.Len() > 0 {
				checkErr.WriteString(fmt.Sprintf("\nInspect error: %s", inspectErrBuf.String()))
			}
			if inspectErr != nil {
				checkErr.WriteString(fmt.Sprintf("\nInspect command error: %v", inspectErr))
			}
		}
	}
	
	if !containerRunning {
		errorDetails := checkErr.String()
		if errorDetails == "" {
			errorDetails = "Container not found"
		}
		errorMsg := fmt.Sprintf("Sandbox container '%s' is not accessible.\nDetails: %s\n\nTroubleshooting:\n1. Check if container exists: docker ps -a | grep %s\n2. Start container: docker compose up -d sandbox\n3. Verify Docker socket access from app container", container, errorDetails, container)
		// Write error to stream (ignore if connection closed)
		safeWriteSSE(w, "[ERROR] "+errorMsg)
		return &ExecutionResult{
			Success: false,
			Error:   errorMsg,
			Results: []TestResult{},
		}
	}

	// Test connection to sandbox container first
	testCmd := exec.CommandContext(ctx, "docker", "exec", container, "echo", "test")
	var testErr bytes.Buffer
	testCmd.Stderr = &testErr
	if err := testCmd.Run(); err != nil {
		errorMsg := fmt.Sprintf("Cannot connect to sandbox container '%s'. Test command failed: %v\n%s\n\nThis usually means:\n1. Container is not running\n2. App container cannot access Docker socket\n3. Container name mismatch\n\nTry: docker compose ps", container, err, testErr.String())
		safeWriteSSE(w, "[ERROR] "+errorMsg)
		return &ExecutionResult{
			Success: false,
			Error:   errorMsg,
			Results: []TestResult{},
		}
	}

	// Write code to sandbox container
	writeCmd := exec.CommandContext(ctx, "docker", "exec", "-i", container,
		"sh", "-c", fmt.Sprintf("cat > %s", filename))
	writeCmd.Stdin = strings.NewReader(testCode)
	var writeErr bytes.Buffer
	writeCmd.Stderr = &writeErr
	
	if err := writeCmd.Run(); err != nil {
		errorMsg := fmt.Sprintf("Failed to write code to container '%s': %v\n%s", container, err, writeErr.String())
		// Write error to stream (ignore if connection closed)
		safeWriteSSE(w, "[ERROR] "+errorMsg)
		return &ExecutionResult{
			Success: false,
			Error:   errorMsg,
			Results: []TestResult{},
		}
	}

	// Execute code in sandbox container
	var cmd *exec.Cmd
	if language == "python" {
		// For Python, use python3 directly (timeout handled by context)
		cmd = exec.CommandContext(ctx, "docker", "exec", container,
			"sh", "-c", fmt.Sprintf("cd /tmp && timeout 25 python3 %s 2>&1", filename))
	} else {
		cmd = exec.CommandContext(ctx, "docker", "exec", container,
			"sh", "-c", fmt.Sprintf("cd /tmp && timeout 25 %s %s", execCmd, filename))
	}

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		errorMsg := "Failed to create stdout pipe: " + err.Error()
		safeWriteSSE(w, "[ERROR] "+errorMsg)
		return &ExecutionResult{Success: false, Error: errorMsg}
	}

	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		errorMsg := "Failed to create stderr pipe: " + err.Error()
		safeWriteSSE(w, "[ERROR] "+errorMsg)
		return &ExecutionResult{Success: false, Error: errorMsg}
	}

	if err := cmd.Start(); err != nil {
		errorMsg := "Failed to start command: " + err.Error()
		safeWriteSSE(w, "[ERROR] "+errorMsg)
		return &ExecutionResult{Success: false, Error: errorMsg}
	}

	var output strings.Builder
	var errorOutput strings.Builder
	doneChan := make(chan bool, 2)

	// Stream stdout
	go func() {
		defer func() {
			if r := recover(); r != nil {
				errorOutput.WriteString(fmt.Sprintf("Stdout goroutine panic: %v\n", r))
			}
			doneChan <- true
		}()
		scanner := bufio.NewScanner(stdoutPipe)
		for scanner.Scan() {
			line := scanner.Text()
			output.WriteString(line + "\n")
			// Stop streaming if connection is closed
			if !safeWriteSSE(w, line) {
				break
			}
		}
		if err := scanner.Err(); err != nil {
			errorMsg := "Stdout scanner error: " + err.Error()
			errorOutput.WriteString(errorMsg + "\n")
			safeWriteSSE(w, "[ERROR] "+errorMsg)
		}
	}()

	// Stream stderr
	go func() {
		defer func() {
			if r := recover(); r != nil {
				errorOutput.WriteString(fmt.Sprintf("Stderr goroutine panic: %v\n", r))
			}
			doneChan <- true
		}()
		scanner := bufio.NewScanner(stderrPipe)
		for scanner.Scan() {
			line := scanner.Text()
			errorOutput.WriteString(line + "\n")
			// Stop streaming if connection is closed
			if !safeWriteSSE(w, "[STDERR] "+line) {
				break
			}
		}
		if err := scanner.Err(); err != nil {
			errorMsg := "Stderr scanner error: " + err.Error()
			errorOutput.WriteString(errorMsg + "\n")
			safeWriteSSE(w, "[ERROR] "+errorMsg)
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
		
		// Write error to stream (ignore if connection closed)
		safeWriteSSE(w, "[ERROR] "+result.Error)
	}

	if errorStr != "" && err == nil {
		result.Error = errorStr
		if outputStr != "" {
			result.Error += "\n" + outputStr
		}
		result.Success = false
		
		// Write error to stream (ignore if connection closed)
		safeWriteSSE(w, "[ERROR] "+result.Error)
	}

	// Success should be true only if all tests passed and there were no errors
	if len(result.Results) > 0 {
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

// escapeSSE escapes special characters for Server-Sent Events
func escapeSSE(s string) string {
	s = strings.ReplaceAll(s, "\n", "\\n")
	s = strings.ReplaceAll(s, "\r", "\\r")
	s = strings.ReplaceAll(s, "\\", "\\\\")
	return s
}

// safeWriteSSE safely writes SSE data to the response writer, handling closed connections
func safeWriteSSE(w io.Writer, message string) bool {
	if httpW, ok := w.(http.ResponseWriter); ok {
		_, err := fmt.Fprintf(httpW, "data: %s\n\n", escapeSSE(message))
		if err != nil {
			// Connection closed or timed out - don't try to write more
			return false
		}
		if flusher, ok := httpW.(http.Flusher); ok {
			flusher.Flush()
		}
		return true
	}
	return false
}
