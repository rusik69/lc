package tests

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/rusik69/lc/internal/executor"
	"github.com/rusik69/lc/internal/handlers"
	"github.com/rusik69/lc/internal/problems"
)

func init() {
	// Initialize templates for tests
	if err := handlers.InitTemplates("../../web/templates"); err != nil {
		panic("Failed to initialize templates: " + err.Error())
	}
}

func TestHandleIndex(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()

	handlers.HandleIndex(w, req)

	resp := w.Result()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status 200, got %d", resp.StatusCode)
	}

	bodyStr := string(body)
	if !strings.Contains(bodyStr, "Two Sum") {
		t.Error("Response doesn't contain 'Two Sum' problem")
	}
	if !strings.Contains(bodyStr, "LeetCode Clone") {
		t.Error("Response doesn't contain page title")
	}
}

func TestHandleProblem(t *testing.T) {
	tests := []struct {
		name           string
		path           string
		expectedStatus int
		wantContain    []string
	}{
		{
			name:           "Valid problem 1",
			path:           "/problem/1",
			expectedStatus: http.StatusOK,
			wantContain:    []string{"Two Sum", "func twoSum"},
		},
		{
			name:           "Valid problem 2",
			path:           "/problem/2",
			expectedStatus: http.StatusOK,
			wantContain:    []string{"Reverse String", "func reverseString"},
		},
		{
			name:           "Invalid problem 999",
			path:           "/problem/999",
			expectedStatus: http.StatusNotFound,
			wantContain:    nil,
		},
		{
			name:           "Invalid problem path",
			path:           "/problem/abc",
			expectedStatus: http.StatusNotFound,
			wantContain:    nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, tt.path, nil)
			w := httptest.NewRecorder()

			handlers.HandleProblem(w, req)

			resp := w.Result()
			body, _ := io.ReadAll(resp.Body)

			if resp.StatusCode != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d", tt.expectedStatus, resp.StatusCode)
			}

			if tt.wantContain != nil {
				bodyStr := string(body)
				for _, want := range tt.wantContain {
					if !strings.Contains(bodyStr, want) {
						t.Errorf("Response doesn't contain expected string: %q", want)
					}
				}
			}
		})
	}
}

func TestHandleSolution(t *testing.T) {
	tests := []struct {
		name           string
		path           string
		expectedStatus int
		wantContain    []string
	}{
		{
			name:           "Valid solution 1",
			path:           "/solution/1",
			expectedStatus: http.StatusOK,
			wantContain:    []string{"func twoSum", "Solution"},
		},
		{
			name:           "Invalid solution 999",
			path:           "/solution/999",
			expectedStatus: http.StatusNotFound,
			wantContain:    nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, tt.path, nil)
			w := httptest.NewRecorder()

			handlers.HandleSolution(w, req)

			resp := w.Result()
			body, _ := io.ReadAll(resp.Body)

			if resp.StatusCode != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d", tt.expectedStatus, resp.StatusCode)
			}

			if tt.wantContain != nil {
				bodyStr := string(body)
				for _, want := range tt.wantContain {
					if !strings.Contains(bodyStr, want) {
						t.Errorf("Response doesn't contain expected string: %q", want)
					}
				}
			}
		})
	}
}

func TestHandleRunMethodValidation(t *testing.T) {
	tests := []struct {
		name           string
		method         string
		expectedStatus int
	}{
		{
			name:           "POST method allowed",
			method:         http.MethodPost,
			expectedStatus: http.StatusBadRequest, // Will fail on empty code, but method is OK
		},
		{
			name:           "GET method not allowed",
			method:         http.MethodGet,
			expectedStatus: http.StatusMethodNotAllowed,
		},
		{
			name:           "PUT method not allowed",
			method:         http.MethodPut,
			expectedStatus: http.StatusMethodNotAllowed,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, "/run/1", nil)
			w := httptest.NewRecorder()

			handlers.HandleRun(w, req)

			resp := w.Result()
			if resp.StatusCode != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d", tt.expectedStatus, resp.StatusCode)
			}
		})
	}
}

func TestHandleRunInvalidPaths(t *testing.T) {
	tests := []struct {
		name string
		path string
	}{
		{
			name: "Missing problem ID",
			path: "/run/",
		},
		{
			name: "Invalid problem ID",
			path: "/run/abc",
		},
		{
			name: "Non-existent problem ID",
			path: "/run/999",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			form := url.Values{}
			form.Add("code", "func test() {}")
			form.Add("action", "run")

			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(form.Encode()))
			req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
			w := httptest.NewRecorder()

			handlers.HandleRun(w, req)

			resp := w.Result()
			if resp.StatusCode != http.StatusNotFound {
				t.Errorf("Expected status 404, got %d", resp.StatusCode)
			}
		})
	}
}

func TestHandleRunEmptyCode(t *testing.T) {
	form := url.Values{}
	form.Add("code", "")
	form.Add("action", "run")

	req := httptest.NewRequest(http.MethodPost, "/run/1", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	w := httptest.NewRecorder()

	handlers.HandleRun(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", resp.StatusCode)
	}

	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "Code is required") {
		t.Error("Expected 'Code is required' error message")
	}
}

func TestHandleRunDangerousCode(t *testing.T) {
	dangerousCodes := []string{
		`import "os/exec"`,
		`import "syscall"`,
		`import "net/http"`,
		`func test() { os.Remove("/tmp/file") }`,
	}

	for i, code := range dangerousCodes {
		t.Run("Dangerous pattern "+string(rune('A'+i)), func(t *testing.T) {
			form := url.Values{}
			form.Add("code", code)
			form.Add("action", "run")

			req := httptest.NewRequest(http.MethodPost, "/run/1", strings.NewReader(form.Encode()))
			req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
			w := httptest.NewRecorder()

			handlers.HandleRun(w, req)

			resp := w.Result()
			if resp.StatusCode != http.StatusBadRequest {
				t.Errorf("Expected status 400, got %d", resp.StatusCode)
			}

			body, _ := io.ReadAll(resp.Body)
			if !strings.Contains(string(body), "prohibited patterns") {
				t.Error("Expected 'prohibited patterns' error message")
			}
		})
	}
}

func TestHandleRunCodeSizeLimit(t *testing.T) {
	// Create code larger than maxCodeSize (100KB)
	maxCodeSize := 100 * 1024
	largeCode := strings.Repeat("a", maxCodeSize+1)

	form := url.Values{}
	form.Add("code", largeCode)
	form.Add("action", "run")

	req := httptest.NewRequest(http.MethodPost, "/run/1", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	w := httptest.NewRecorder()

	handlers.HandleRun(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", resp.StatusCode)
	}

	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "Code too large") {
		t.Error("Expected 'Code too large' error message")
	}
}

// Note: Tests for private functions (getClientIP, rateLimiter, escapeSSE) 
// are removed as they're internal implementation details.
// These are tested indirectly through the public handler APIs.

func TestExecuteCodeStreamBasic(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Docker test in short mode")
	}

	problem := problems.GetProblem(1) // Two Sum
	if problem == nil {
		t.Fatal("GetProblem(1) returned nil")
	}

	userCode := problem.Solution

	var buf bytes.Buffer
	result := executor.ExecuteCodeStream(problem, userCode, false, &buf)

	if result == nil {
		t.Fatal("ExecuteCodeStream returned nil result")
	}

	// Check basic result structure
	if result.TimeTaken == "" {
		t.Error("TimeTaken is empty")
	}

	t.Logf("Execution completed in %s, success=%v", result.TimeTaken, result.Success)
}
