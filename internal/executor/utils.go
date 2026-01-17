package executor

import (
	"crypto/rand"
	"encoding/hex"
	"os"
	"strings"
)

// GetSandboxContainer returns the sandbox container name from env or default
func GetSandboxContainer() string {
	container := os.Getenv("SANDBOX_CONTAINER")
	if container == "" {
		container = "lc-sandbox"
	}
	return container
}

// GenerateUniqueID generates a random ID for temporary files
func GenerateUniqueID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// DetectLanguage detects the programming language from user code
func DetectLanguage(userCode string) string {
	code := strings.TrimSpace(userCode)
	if strings.HasPrefix(code, "def ") || strings.HasPrefix(code, "class ") ||
		strings.Contains(code, "import ") || strings.Contains(code, "print(") ||
		strings.Contains(code, "if __name__") {
		return "python"
	}
	return "go"
}
