package executor

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
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
	executionID := GenerateUniqueID()
	
	log.Printf("[EXEC-%s] Starting code execution - Problem ID: %d, Language: %s, RunAllTests: %v, CodeLength: %d",
		executionID, problem.ID, language, runAllTests, len(userCode))

	// Write initial message to establish stream connection
	if httpW, ok := w.(http.ResponseWriter); ok {
		log.Printf("[EXEC-%s] Writing initial SSE message", executionID)
		_, err := fmt.Fprintf(httpW, "data: %s\n\n", escapeSSE("Preparing execution..."))
		if err != nil {
			log.Printf("[EXEC-%s] ERROR: Failed to write initial SSE message: %v", executionID, err)
		} else {
			if flusher, ok := httpW.(http.Flusher); ok {
				flusher.Flush()
				log.Printf("[EXEC-%s] Initial SSE message flushed", executionID)
			}
		}
	}

	// Detect language if not provided
	if language == "" {
		language = DetectLanguage(userCode)
		log.Printf("[EXEC-%s] Auto-detected language: %s", executionID, language)
	} else {
		log.Printf("[EXEC-%s] Using provided language: %s", executionID, language)
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
				log.Printf("[EXEC-%s] ERROR: Panic during test code generation: %v", executionID, r)
				errorMsg := genErr.Error()
				if httpW, ok := w.(http.ResponseWriter); ok {
					fmt.Fprintf(httpW, "data: %s\n\n", escapeSSE("[ERROR] "+errorMsg))
					if flusher, ok := httpW.(http.Flusher); ok {
						flusher.Flush()
					}
				}
			}
		}()
		log.Printf("[EXEC-%s] Generating test code for language: %s", executionID, language)
		if language == "python" {
			testCode = GeneratePythonTestCode(problem, userCode, runAllTests)
			fileExt = "py"
			execCmd = "python3"
			log.Printf("[EXEC-%s] Generated Python test code, length: %d", executionID, len(testCode))
		} else {
			testCode = GenerateTestCode(problem, userCode, runAllTests)
			fileExt = "go"
			execCmd = "go run"
			log.Printf("[EXEC-%s] Generated Go test code, length: %d", executionID, len(testCode))
		}
	}()

	// If test code generation failed, return error
	if genErr != nil || testCode == "" {
		errorMsg := "Failed to generate test code"
		if genErr != nil {
			errorMsg = genErr.Error()
		}
		log.Printf("[EXEC-%s] ERROR: Test code generation failed: %s", executionID, errorMsg)
		// Write error to stream (ignore if connection closed)
		safeWriteSSE(w, "[ERROR] "+errorMsg)
		return &ExecutionResult{
			Success: false,
			Error:   errorMsg,
			Results: []TestResult{},
		}
	}

	// Context timeout: 40 seconds (allows 35s execution + 5s overhead)
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()

	// Generate unique filename
	uniqueID := GenerateUniqueID()
	filename := fmt.Sprintf("/tmp/code_%s.%s", uniqueID, fileExt)
	log.Printf("[EXEC-%s] Generated filename: %s", executionID, filename)
	
	// Get sandbox container name
	container := GetSandboxContainer()
	log.Printf("[EXEC-%s] Using sandbox container: %s", executionID, container)

	// Check if container exists and is running
	log.Printf("[EXEC-%s] Checking if container '%s' is running", executionID, container)
	// First try: docker ps with filter
	checkCmd := exec.CommandContext(ctx, "docker", "ps", "--format", "{{.Names}}", "--filter", "name=^"+container+"$")
	var checkErr bytes.Buffer
	checkCmd.Stderr = &checkErr
	checkOutput, err := checkCmd.Output()
	checkOutputStr := strings.TrimSpace(string(checkOutput))
	
	containerRunning := err == nil && checkOutputStr == container
	log.Printf("[EXEC-%s] Container check (docker ps): running=%v, output='%s', error=%v", 
		executionID, containerRunning, checkOutputStr, err)
	
	// Second try: docker inspect (more reliable)
	if !containerRunning {
		log.Printf("[EXEC-%s] Container not found via docker ps, trying docker inspect", executionID)
		inspectCmd := exec.CommandContext(ctx, "docker", "inspect", "--format", "{{.State.Running}}", container)
		var inspectErrBuf bytes.Buffer
		inspectCmd.Stderr = &inspectErrBuf
		inspectOutput, inspectErr := inspectCmd.Output()
		if inspectErr == nil && strings.TrimSpace(string(inspectOutput)) == "true" {
			containerRunning = true
			log.Printf("[EXEC-%s] Container is running (confirmed via docker inspect)", executionID)
		} else {
			log.Printf("[EXEC-%s] Container inspect failed: output='%s', error=%v", 
				executionID, strings.TrimSpace(string(inspectOutput)), inspectErr)
			// Try to get more info about what containers are available
			listCmd := exec.CommandContext(ctx, "docker", "ps", "--format", "{{.Names}}")
			listOutput, _ := listCmd.Output()
			checkErr.WriteString(fmt.Sprintf("\nAvailable containers: %s", strings.TrimSpace(string(listOutput))))
			log.Printf("[EXEC-%s] Available containers: %s", executionID, strings.TrimSpace(string(listOutput)))
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
		log.Printf("[EXEC-%s] ERROR: Container '%s' is not running. Details: %s", executionID, container, errorDetails)
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
	log.Printf("[EXEC-%s] Testing connection to container", executionID)
	testCmd := exec.CommandContext(ctx, "docker", "exec", container, "echo", "test")
	var testErr bytes.Buffer
	testCmd.Stderr = &testErr
	if err := testCmd.Run(); err != nil {
		log.Printf("[EXEC-%s] ERROR: Cannot connect to container: %v, stderr: %s", executionID, err, testErr.String())
		errorMsg := fmt.Sprintf("Cannot connect to sandbox container '%s'. Test command failed: %v\n%s\n\nThis usually means:\n1. Container is not running\n2. App container cannot access Docker socket\n3. Container name mismatch\n\nTry: docker compose ps", container, err, testErr.String())
		safeWriteSSE(w, "[ERROR] "+errorMsg)
		return &ExecutionResult{
			Success: false,
			Error:   errorMsg,
			Results: []TestResult{},
		}
	}
	log.Printf("[EXEC-%s] Container connection test successful", executionID)

	// Write code to sandbox container
	log.Printf("[EXEC-%s] Writing code to container file: %s (size: %d bytes)", executionID, filename, len(testCode))
	writeCmd := exec.CommandContext(ctx, "docker", "exec", "-i", container,
		"sh", "-c", fmt.Sprintf("cat > %s", filename))
	writeCmd.Stdin = strings.NewReader(testCode)
	var writeErr bytes.Buffer
	writeCmd.Stderr = &writeErr
	
	if err := writeCmd.Run(); err != nil {
		log.Printf("[EXEC-%s] ERROR: Failed to write code to container: %v, stderr: %s", executionID, err, writeErr.String())
		errorMsg := fmt.Sprintf("Failed to write code to container '%s': %v\n%s", container, err, writeErr.String())
		// Write error to stream (ignore if connection closed)
		safeWriteSSE(w, "[ERROR] "+errorMsg)
		return &ExecutionResult{
			Success: false,
			Error:   errorMsg,
			Results: []TestResult{},
		}
	}
	log.Printf("[EXEC-%s] Code written successfully to container", executionID)

	// Execute code in sandbox container
	// Timeout: 35 seconds (allows for compilation + execution + overhead)
	// Go compilation can take a few seconds, so we need extra time
	var cmd *exec.Cmd
	var execCommand string
	if language == "python" {
		// For Python, use python3 directly (timeout handled by context)
		execCommand = fmt.Sprintf("cd /tmp && timeout 35 python3 %s 2>&1", filename)
		cmd = exec.CommandContext(ctx, "docker", "exec", container, "sh", "-c", execCommand)
		log.Printf("[EXEC-%s] Starting Python execution: %s", executionID, execCommand)
	} else {
		execCommand = fmt.Sprintf("cd /tmp && timeout 35 %s %s", execCmd, filename)
		cmd = exec.CommandContext(ctx, "docker", "exec", container, "sh", "-c", execCommand)
		log.Printf("[EXEC-%s] Starting Go execution: %s", executionID, execCommand)
	}

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		log.Printf("[EXEC-%s] ERROR: Failed to create stdout pipe: %v", executionID, err)
		errorMsg := "Failed to create stdout pipe: " + err.Error()
		safeWriteSSE(w, "[ERROR] "+errorMsg)
		return &ExecutionResult{Success: false, Error: errorMsg}
	}

	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		log.Printf("[EXEC-%s] ERROR: Failed to create stderr pipe: %v", executionID, err)
		errorMsg := "Failed to create stderr pipe: " + err.Error()
		safeWriteSSE(w, "[ERROR] "+errorMsg)
		return &ExecutionResult{Success: false, Error: errorMsg}
	}

	log.Printf("[EXEC-%s] Starting command execution", executionID)
	if err := cmd.Start(); err != nil {
		log.Printf("[EXEC-%s] ERROR: Failed to start command: %v", executionID, err)
		errorMsg := "Failed to start command: " + err.Error()
		safeWriteSSE(w, "[ERROR] "+errorMsg)
		return &ExecutionResult{Success: false, Error: errorMsg}
	}
	log.Printf("[EXEC-%s] Command started successfully, streaming output", executionID)

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
	log.Printf("[EXEC-%s] Command execution completed, exit error: %v", executionID, err)
	
	<-doneChan
	<-doneChan
	log.Printf("[EXEC-%s] Output streaming goroutines completed", executionID)
	
	// Clean up temp file
	log.Printf("[EXEC-%s] Cleaning up temp file: %s", executionID, filename)
	cleanCmd := exec.Command("docker", "exec", container, "rm", "-f", filename)
	if cleanErr := cleanCmd.Run(); cleanErr != nil {
		log.Printf("[EXEC-%s] WARNING: Failed to cleanup temp file: %v", executionID, cleanErr)
	} else {
		log.Printf("[EXEC-%s] Temp file cleaned up successfully", executionID)
	}
	
	timeTaken := time.Since(start).String()
	log.Printf("[EXEC-%s] Total execution time: %s", executionID, timeTaken)

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
		
		// Check for timeout (exit status 124 from timeout command)
		if exitError, ok := err.(*exec.ExitError); ok && exitError.ExitCode() == 124 {
			errorMsg.WriteString("Execution timed out after 35 seconds. ")
			errorMsg.WriteString("Your code may be taking too long to execute or may have an infinite loop. ")
			errorMsg.WriteString("Please optimize your code or check for infinite loops.")
			log.Printf("[EXEC-%s] ERROR: Execution timed out (exit status 124)", executionID)
		} else {
			errorMsg.WriteString("Execution failed: " + err.Error())
		}
		
		if errorStr != "" {
			errorMsg.WriteString("\n\nStderr: " + errorStr)
		}

		if outputStr != "" {
			errorMsg.WriteString("\n\nOutput: " + outputStr)
		}

		result.Error = errorMsg.String()
		result.Success = false
		log.Printf("[EXEC-%s] ERROR: Execution failed: %s", executionID, result.Error)
		
		// Write error to stream (ignore if connection closed)
		safeWriteSSE(w, "[ERROR] "+result.Error)
	}

	if errorStr != "" && err == nil {
		result.Error = errorStr
		if outputStr != "" {
			result.Error += "\n" + outputStr
		}
		result.Success = false
		log.Printf("[EXEC-%s] ERROR: Execution produced stderr: %s", executionID, errorStr)
		
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
		log.Printf("[EXEC-%s] Test results: %d total, all passed: %v, success: %v", 
			executionID, len(result.Results), allPassed, result.Success)
	} else {
		log.Printf("[EXEC-%s] No test results parsed, success: %v", executionID, result.Success)
	}

	log.Printf("[EXEC-%s] Execution completed - Success: %v, Tests: %d, Output length: %d, Error length: %d",
		executionID, result.Success, len(result.Results), len(outputStr), len(errorStr))
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
			log.Printf("ERROR: Failed to write SSE message (connection closed?): %v, message length: %d", err, len(message))
			return false
		}
		if flusher, ok := httpW.(http.Flusher); ok {
			flusher.Flush()
		}
		return true
	}
	return false
}
