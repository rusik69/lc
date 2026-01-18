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
		* { margin: 0; padding: 0; box-sizing: border-box; }
		body {
			font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
			background-color: #0f172a;
			color: #e0e0e0;
			min-height: 100vh;
			display: flex;
			align-items: center;
			justify-content: center;
			padding: 20px;
		}
		.login-container {
			background-color: #1a2332;
			border: 1px solid #334155;
			border-radius: 8px;
			padding: 40px;
			max-width: 400px;
			width: 100%;
			box-shadow: 0 4px 6px rgba(0,0,0,0.3);
		}
		h2 {
			color: #e0e0e0;
			margin-bottom: 30px;
			text-align: center;
			font-size: 1.75rem;
		}
		form {
			display: flex;
			flex-direction: column;
			gap: 20px;
		}
		input {
			padding: 12px;
			font-size: 16px;
			background-color: #1e293b;
			color: #e0e0e0;
			border: 1px solid #334155;
			border-radius: 4px;
			transition: border-color 0.2s;
		}
		input:focus {
			outline: none;
			border-color: #3b82f6;
		}
		input::placeholder {
			color: #999;
		}
		button {
			padding: 12px;
			font-size: 16px;
			background: #3b82f6;
			color: white;
			border: none;
			border-radius: 4px;
			cursor: pointer;
			font-weight: 500;
			transition: background 0.2s;
		}
		button:hover {
			background: #60a5fa;
		}
		button:active {
			background: #2563eb;
		}
		.error {
			color: #ff6b6b;
			margin-top: 15px;
			padding: 10px;
			background-color: #4d1e1e;
			border: 1px solid #5a2d2d;
			border-radius: 4px;
			text-align: center;
		}
	</style>
</head>
<body>
	<div class="login-container">
		<h2>Admin Login</h2>
		<form method="POST">
			<input type="password" name="password" placeholder="Password" required autofocus>
			<button type="submit">Login</button>
		</form>
	</div>
</body>
</html>`)
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
		* { margin: 0; padding: 0; box-sizing: border-box; }
		body {
			font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
			background-color: #0f172a;
			color: #e0e0e0;
			min-height: 100vh;
			display: flex;
			align-items: center;
			justify-content: center;
			padding: 20px;
		}
		.login-container {
			background-color: #1a2332;
			border: 1px solid #334155;
			border-radius: 8px;
			padding: 40px;
			max-width: 400px;
			width: 100%;
			box-shadow: 0 4px 6px rgba(0,0,0,0.3);
		}
		h2 {
			color: #e0e0e0;
			margin-bottom: 30px;
			text-align: center;
			font-size: 1.75rem;
		}
		form {
			display: flex;
			flex-direction: column;
			gap: 20px;
		}
		input {
			padding: 12px;
			font-size: 16px;
			background-color: #1e293b;
			color: #e0e0e0;
			border: 1px solid #334155;
			border-radius: 4px;
			transition: border-color 0.2s;
		}
		input:focus {
			outline: none;
			border-color: #3b82f6;
		}
		input::placeholder {
			color: #999;
		}
		button {
			padding: 12px;
			font-size: 16px;
			background: #3b82f6;
			color: white;
			border: none;
			border-radius: 4px;
			cursor: pointer;
			font-weight: 500;
			transition: background 0.2s;
		}
		button:hover {
			background: #60a5fa;
		}
		button:active {
			background: #2563eb;
		}
		.error {
			color: #ff6b6b;
			margin-top: 15px;
			padding: 10px;
			background-color: #4d1e1e;
			border: 1px solid #5a2d2d;
			border-radius: 4px;
			text-align: center;
		}
	</style>
</head>
<body>
	<div class="login-container">
		<h2>Admin Login</h2>
		<form method="POST">
			<input type="password" name="password" placeholder="Password" required autofocus>
			<button type="submit">Login</button>
		</form>
		<div class="error">Invalid password</div>
	</div>
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
