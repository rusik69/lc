package auth

import (
	"bufio"
	"crypto/subtle"
	"fmt"
	"net/http"
	"os"
	"strings"
)

var adminPassword string

// InitAuth loads admin password from .prod.env file
// If file doesn't exist, auth is disabled (for local development)
func InitAuth() error {
	file, err := os.Open(".prod.env")
	if err != nil {
		if os.IsNotExist(err) {
			// .prod.env doesn't exist - auth disabled for local dev
			return nil
		}
		return fmt.Errorf("failed to open .prod.env: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "ADMIN_PASSWORD=") {
			adminPassword = strings.TrimPrefix(line, "ADMIN_PASSWORD=")
			adminPassword = strings.Trim(adminPassword, "\"'")
			return nil
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("failed to read .prod.env: %w", err)
	}

	return fmt.Errorf("ADMIN_PASSWORD not found in .prod.env")
}

// Middleware protects routes with basic authentication
func Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Allow login/logout pages without auth
		if r.URL.Path == "/login" || r.URL.Path == "/logout" {
			next.ServeHTTP(w, r)
			return
		}

		// If no password is set (local dev), allow all
		if adminPassword == "" {
			next.ServeHTTP(w, r)
			return
		}

		// All other routes require authentication
		// Check for valid session cookie
		cookie, err := r.Cookie("auth_token")
		if err != nil || !isValidToken(cookie.Value) {
			// Redirect to login
			http.Redirect(w, r, "/login", http.StatusSeeOther)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// HandleLogin handles login requests
func HandleLogin(w http.ResponseWriter, r *http.Request) {
	// If no password is set (local dev), redirect to home
	if adminPassword == "" {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}

	if r.Method == http.MethodGet {
		// Show login page
		fmt.Fprintf(w, `<!DOCTYPE html>
<html>
<head>
	<title>Login - LeetCode Clone</title>
	<style>
		body { font-family: Arial, sans-serif; max-width: 400px; margin: 100px auto; padding: 20px; }
		form { display: flex; flex-direction: column; gap: 15px; }
		input { padding: 10px; font-size: 16px; }
		button { padding: 10px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; }
		button:hover { background: #0056b3; }
		.error { color: red; margin-top: 10px; }
	</style>
</head>
<body>
	<h2>Admin Login</h2>
	<form method="POST">
		<input type="password" name="password" placeholder="Password" required autofocus>
		<button type="submit">Login</button>
	</form>
	%s
</body>
</html>`, "")
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	password := r.FormValue("password")
	if password == "" {
		http.Error(w, "Password required", http.StatusBadRequest)
		return
	}

	// Constant-time comparison to prevent timing attacks
	if subtle.ConstantTimeCompare([]byte(password), []byte(adminPassword)) != 1 {
		fmt.Fprintf(w, `<!DOCTYPE html>
<html>
<head>
	<title>Login - LeetCode Clone</title>
	<style>
		body { font-family: Arial, sans-serif; max-width: 400px; margin: 100px auto; padding: 20px; }
		form { display: flex; flex-direction: column; gap: 15px; }
		input { padding: 10px; font-size: 16px; }
		button { padding: 10px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; }
		button:hover { background: #0056b3; }
		.error { color: red; margin-top: 10px; }
	</style>
</head>
<body>
	<h2>Admin Login</h2>
	<form method="POST">
		<input type="password" name="password" placeholder="Password" required autofocus>
		<button type="submit">Login</button>
	</form>
	<div class="error">Invalid password</div>
</body>
</html>`)
		return
	}

	// Set auth cookie
	token := generateToken()
	http.SetCookie(w, &http.Cookie{
		Name:     "auth_token",
		Value:    token,
		Path:     "/",
		HttpOnly: true,
		Secure:   false, // Set to true in production with HTTPS
		SameSite: http.SameSiteStrictMode,
		MaxAge:   86400 * 7, // 7 days
	})

	// Redirect to home
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// HandleLogout handles logout requests
func HandleLogout(w http.ResponseWriter, r *http.Request) {
	http.SetCookie(w, &http.Cookie{
		Name:     "auth_token",
		Value:    "",
		Path:     "/",
		HttpOnly: true,
		MaxAge:   -1,
	})
	http.Redirect(w, r, "/login", http.StatusSeeOther)
}

// Simple token validation (in production, use JWT or session store)
var validTokens = make(map[string]bool)

func generateToken() string {
	token := fmt.Sprintf("token_%d", len(validTokens)+1)
	validTokens[token] = true
	return token
}

func isValidToken(token string) bool {
	return validTokens[token]
}
