package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          1626,
			Title:       "Security in Go",
			Description: "Write secure Go applications: cryptography, TLS, input validation, authentication, and common vulnerability prevention.",
			Order:       26,
			Lessons: []problems.Lesson{
				{
					Title: "Cryptography and Hashing",
					Content: `Go's crypto packages provide production-ready implementations of common cryptographic algorithms. Understanding when and how to use them correctly is critical for application security.

**Hashing:**
` + "```" + `
Hashing: one-way function, fixed output, deterministic

Common hash functions:
  SHA-256: General purpose, 32 bytes output
  SHA-512: Higher security, 64 bytes output
  bcrypt:  Password hashing (slow by design)
  argon2:  Modern password hashing (recommended)
  
NEVER use for passwords: MD5, SHA-1, SHA-256 (too fast!)
  Attacker with GPU: billions of SHA-256/sec
  bcrypt: ~5-10 per second (by design!)

Password hashing with bcrypt:
  import "golang.org/x/crypto/bcrypt"
  
  // Hash password (during registration)
  hash, err := bcrypt.GenerateFromPassword(
      []byte(password), bcrypt.DefaultCost)
  // Store hash in database
  
  // Verify password (during login)
  err := bcrypt.CompareHashAndPassword(hash, []byte(password))
  if err != nil {
      // Invalid password (or corrupted hash)
  }
  
  Cost parameter:
    bcrypt.MinCost     = 4
    bcrypt.DefaultCost = 10  (~100ms on modern CPU)
    bcrypt.MaxCost     = 31
    
    Each +1 doubles the time
    Recommendation: choose cost that takes ~250ms on your server

SHA-256 for data integrity:
  h := sha256.New()
  h.Write(data)
  digest := h.Sum(nil) // []byte, 32 bytes
  hex := fmt.Sprintf("%x", digest) // Hex string
  
  // Or shorthand:
  digest := sha256.Sum256(data) // [32]byte

HMAC for message authentication:
  mac := hmac.New(sha256.New, secretKey)
  mac.Write(message)
  signature := mac.Sum(nil)
  
  // Verify (constant-time comparison!)
  expectedMAC := computeMAC(message, key)
  if !hmac.Equal(signature, expectedMAC) {
      // Tampered!
  }
  
  HMAC prevents: message tampering, forgery
  Use for: API signatures, webhook verification, session tokens
` + "```" + `

**Encryption:**
` + "```" + `
Symmetric encryption (same key encrypts and decrypts):
  AES-GCM (recommended):
    - Authenticated encryption (confidentiality + integrity)
    - Fast (hardware-accelerated on modern CPUs)
    - 256-bit key recommended

  block, _ := aes.NewCipher(key) // key: 16/24/32 bytes
  gcm, _ := cipher.NewGCM(block)
  
  // Encrypt:
  nonce := make([]byte, gcm.NonceSize())
  io.ReadFull(rand.Reader, nonce) // Random nonce!
  ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)
  
  // Decrypt:
  nonce, ciphertext = ciphertext[:gcm.NonceSize()], ciphertext[gcm.NonceSize():]
  plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
  if err != nil { /* tampered or wrong key */ }

Asymmetric encryption (public/private key pair):
  RSA:
    privateKey, _ := rsa.GenerateKey(rand.Reader, 4096)
    publicKey := &privateKey.PublicKey
    
    // Encrypt with public key
    ciphertext, _ := rsa.EncryptOAEP(sha256.New(), rand.Reader, publicKey, message, nil)
    
    // Decrypt with private key
    plaintext, _ := rsa.DecryptOAEP(sha256.New(), rand.Reader, privateKey, ciphertext, nil)
  
  Ed25519 (for digital signatures, recommended over RSA for signing):
    pub, priv, _ := ed25519.GenerateKey(rand.Reader)
    signature := ed25519.Sign(priv, message)
    valid := ed25519.Verify(pub, message, signature)

Random number generation:
  ALWAYS use crypto/rand for security purposes!
  ✗ math/rand → predictable! (same seed = same sequence)
  ✓ crypto/rand → cryptographically secure
  
  token := make([]byte, 32)
  rand.Read(token)
  tokenStr := base64.URLEncoding.EncodeToString(token)
` + "```" + ``,
					CodeExamples: `// Cryptography patterns in Go
package main

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/hmac"
    "crypto/rand"
    "crypto/sha256"
    "encoding/base64"
    "encoding/hex"
    "fmt"
    "io"
)

// SHA-256 hashing
func hashData(data []byte) string {
    h := sha256.Sum256(data)
    return hex.EncodeToString(h[:])
}

// HMAC signing
func signMessage(message, key []byte) []byte {
    mac := hmac.New(sha256.New, key)
    mac.Write(message)
    return mac.Sum(nil)
}

func verifySignature(message, signature, key []byte) bool {
    expected := signMessage(message, key)
    return hmac.Equal(signature, expected)
}

// AES-GCM encryption
func encrypt(plaintext, key []byte) ([]byte, error) {
    block, err := aes.NewCipher(key)
    if err != nil {
        return nil, err
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    nonce := make([]byte, gcm.NonceSize())
    if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
        return nil, err
    }
    
    return gcm.Seal(nonce, nonce, plaintext, nil), nil
}

func decrypt(ciphertext, key []byte) ([]byte, error) {
    block, err := aes.NewCipher(key)
    if err != nil {
        return nil, err
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    nonceSize := gcm.NonceSize()
    if len(ciphertext) < nonceSize {
        return nil, fmt.Errorf("ciphertext too short")
    }
    
    nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]
    return gcm.Open(nil, nonce, ciphertext, nil)
}

// Generate secure random token
func generateToken(length int) string {
    b := make([]byte, length)
    if _, err := rand.Read(b); err != nil {
        panic(err) // crypto/rand should never fail
    }
    return base64.URLEncoding.EncodeToString(b)
}

// Generate secure random key
func generateKey(bits int) []byte {
    key := make([]byte, bits/8)
    if _, err := rand.Read(key); err != nil {
        panic(err)
    }
    return key
}

func main() {
    // Hashing
    fmt.Println("=== SHA-256 Hashing ===")
    data := []byte("Hello, World!")
    hash := hashData(data)
    fmt.Printf("  Data: %s\n", data)
    fmt.Printf("  Hash: %s\n", hash)
    
    // Same input → same hash
    hash2 := hashData(data)
    fmt.Printf("  Deterministic: %v\n", hash == hash2)
    
    // Different input → different hash
    hash3 := hashData([]byte("Hello, World"))
    fmt.Printf("  Different input → different hash: %v\n", hash != hash3)
    
    // HMAC
    fmt.Println("\n=== HMAC Signing ===")
    key := generateKey(256)
    message := []byte("Transfer $100 to Alice")
    
    sig := signMessage(message, key)
    fmt.Printf("  Message:   %s\n", message)
    fmt.Printf("  Signature: %s\n", hex.EncodeToString(sig))
    
    valid := verifySignature(message, sig, key)
    fmt.Printf("  Valid: %v\n", valid)
    
    // Tampered message
    tampered := []byte("Transfer $10000 to Eve")
    valid = verifySignature(tampered, sig, key)
    fmt.Printf("  Tampered valid: %v\n", valid)
    
    // AES-GCM Encryption
    fmt.Println("\n=== AES-GCM Encryption ===")
    aesKey := generateKey(256)
    plaintext := []byte("Secret message: the password is 42")
    
    ciphertext, err := encrypt(plaintext, aesKey)
    if err != nil {
        fmt.Printf("  Encrypt error: %v\n", err)
        return
    }
    fmt.Printf("  Plaintext:  %s\n", plaintext)
    fmt.Printf("  Ciphertext: %s... (%d bytes)\n",
        hex.EncodeToString(ciphertext[:16]), len(ciphertext))
    
    decrypted, err := decrypt(ciphertext, aesKey)
    if err != nil {
        fmt.Printf("  Decrypt error: %v\n", err)
        return
    }
    fmt.Printf("  Decrypted:  %s\n", decrypted)
    
    // Wrong key fails
    wrongKey := generateKey(256)
    _, err = decrypt(ciphertext, wrongKey)
    fmt.Printf("  Wrong key: %v\n", err != nil)
    
    // Random tokens
    fmt.Println("\n=== Secure Tokens ===")
    for i := 0; i < 3; i++ {
        fmt.Printf("  Token %d: %s\n", i+1, generateToken(32))
    }
}`,
				},
				{
					Title: "TLS and Secure Communication",
					Content: `TLS (Transport Layer Security) encrypts communication between clients and servers. Go's crypto/tls package is production-ready and used by major projects.

**TLS Server:**
` + "```" + `
Basic HTTPS server:
  server := &http.Server{
      Addr:    ":443",
      Handler: mux,
      TLSConfig: &tls.Config{
          MinVersion: tls.VersionTLS12, // Minimum TLS 1.2
          CurvePreferences: []tls.CurveID{
              tls.X25519,    // Fastest, most secure
              tls.CurveP256, // Widely supported
          },
          CipherSuites: []uint16{
              tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
              tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
              tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
              tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
          },
      },
  }
  
  server.ListenAndServeTLS("cert.pem", "key.pem")

TLS 1.3 (Go 1.12+):
  Go automatically supports TLS 1.3
  Cipher suites are NOT configurable for TLS 1.3
  (Go chooses the best automatically)
  
  MinVersion: tls.VersionTLS13  // Require TLS 1.3

Mutual TLS (mTLS):
  Both client and server present certificates
  Used in: microservices, service mesh, zero-trust
  
  Server config:
  tlsConfig := &tls.Config{
      ClientAuth: tls.RequireAndVerifyClientCert,
      ClientCAs:  caCertPool,
      MinVersion: tls.VersionTLS12,
  }
  
  Client config:
  tlsConfig := &tls.Config{
      Certificates: []tls.Certificate{clientCert},
      RootCAs:      caCertPool,
  }

Certificate management:
  Development: use self-signed certs
    go run $(go env GOROOT)/src/crypto/tls/generate_cert.go \
        -host localhost -duration 8760h
        
  Production: use Let's Encrypt (autocert package)
    import "golang.org/x/crypto/acme/autocert"
    
    m := &autocert.Manager{
        Cache:      autocert.DirCache("certs"),
        Prompt:     autocert.AcceptTOS,
        HostPolicy: autocert.HostWhitelist("example.com"),
    }
    server.TLSConfig = m.TLSConfig()
` + "```" + `

**Secure HTTP Client:**
` + "```" + `
Custom TLS client:
  client := &http.Client{
      Transport: &http.Transport{
          TLSClientConfig: &tls.Config{
              MinVersion: tls.VersionTLS12,
              // Do NOT set InsecureSkipVerify in production!
          },
      },
  }

Certificate pinning (advanced):
  Verify the server's certificate matches expected fingerprint
  
  client := &http.Client{
      Transport: &http.Transport{
          TLSClientConfig: &tls.Config{
              VerifyPeerCertificate: func(rawCerts [][]byte, _ [][]*x509.Certificate) error {
                  for _, raw := range rawCerts {
                      fingerprint := sha256.Sum256(raw)
                      if fingerprint == expectedFingerprint {
                          return nil
                      }
                  }
                  return errors.New("certificate pinning failed")
              },
          },
      },
  }

Common mistakes:
  ✗ InsecureSkipVerify: true    → disables ALL certificate validation!
  ✗ MinVersion not set          → allows TLS 1.0 (legacy, insecure)
  ✗ Using self-signed in prod    → no chain of trust
  ✗ Not rotating certificates   → expired certs = downtime
` + "```" + ``,
					CodeExamples: `// TLS configuration patterns
package main

import (
    "crypto/tls"
    "crypto/x509"
    "fmt"
    "net/http"
    "time"
)

// TLS config for production server
func productionTLSConfig() *tls.Config {
    return &tls.Config{
        MinVersion: tls.VersionTLS12,
        CurvePreferences: []tls.CurveID{
            tls.X25519,
            tls.CurveP256,
        },
        // For TLS 1.2 (TLS 1.3 cipher suites are automatic)
        CipherSuites: []uint16{
            tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
            tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
            tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
            tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
            tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
            tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
        },
    }
}

// mTLS config for server
func mTLSServerConfig(caCert []byte) *tls.Config {
    caCertPool := x509.NewCertPool()
    caCertPool.AppendCertsFromPEM(caCert)
    
    return &tls.Config{
        ClientAuth: tls.RequireAndVerifyClientCert,
        ClientCAs:  caCertPool,
        MinVersion: tls.VersionTLS12,
    }
}

// Secure HTTP client
func secureHTTPClient() *http.Client {
    return &http.Client{
        Timeout: 30 * time.Second,
        Transport: &http.Transport{
            TLSClientConfig: &tls.Config{
                MinVersion: tls.VersionTLS12,
            },
            MaxIdleConns:        100,
            MaxIdleConnsPerHost: 10,
            IdleConnTimeout:     90 * time.Second,
        },
    }
}

// TLS version name helper
func tlsVersionName(version uint16) string {
    switch version {
    case tls.VersionTLS10:
        return "TLS 1.0"
    case tls.VersionTLS11:
        return "TLS 1.1"
    case tls.VersionTLS12:
        return "TLS 1.2"
    case tls.VersionTLS13:
        return "TLS 1.3"
    default:
        return fmt.Sprintf("unknown (0x%x)", version)
    }
}

func main() {
    fmt.Println("=== Production TLS Config ===")
    config := productionTLSConfig()
    fmt.Printf("  MinVersion: %s\n", tlsVersionName(config.MinVersion))
    fmt.Printf("  Curves: %v\n", config.CurvePreferences)
    fmt.Printf("  CipherSuites: %d configured\n", len(config.CipherSuites))
    
    for _, cs := range config.CipherSuites {
        name := tls.CipherSuiteName(cs)
        fmt.Printf("    - %s\n", name)
    }
    
    fmt.Println("\n=== Secure HTTP Client ===")
    client := secureHTTPClient()
    transport := client.Transport.(*http.Transport)
    fmt.Printf("  Timeout: %v\n", client.Timeout)
    fmt.Printf("  TLS MinVersion: %s\n", 
        tlsVersionName(transport.TLSClientConfig.MinVersion))
    fmt.Printf("  MaxIdleConns: %d\n", transport.MaxIdleConns)
    
    // Demonstrate checking supported cipher suites
    fmt.Println("\n=== Supported Cipher Suites ===")
    secure := tls.CipherSuites()
    fmt.Printf("  Secure suites: %d\n", len(secure))
    for _, s := range secure[:5] {
        fmt.Printf("    [%04x] %s\n", s.ID, s.Name)
    }
    fmt.Println("    ...")
    
    insecure := tls.InsecureCipherSuites()
    fmt.Printf("  Insecure suites: %d (should NEVER use)\n", len(insecure))
    for _, s := range insecure[:3] {
        fmt.Printf("    [%04x] %s\n", s.ID, s.Name)
    }
}`,
				},
				{
					Title: "Input Validation and OWASP Prevention",
					Content: `Proper input validation and security-aware coding prevents the most common web vulnerabilities. Go's type safety helps, but application-level validation is still required.

**SQL Injection Prevention:**
` + "```" + `
SQL injection: attacker controls SQL query through input

Vulnerable code:
  query := "SELECT * FROM users WHERE name = '" + name + "'"
  db.Query(query) // If name = "'; DROP TABLE users; --" → disaster!

Prevention: ALWAYS use parameterized queries
  db.QueryContext(ctx, "SELECT * FROM users WHERE name = $1", name)
  // Driver handles escaping. Input is NEVER part of the SQL string.

Other injection vectors:
  - Dynamic column names: validate against whitelist
  - ORDER BY: validate direction (ASC/DESC only)
  - LIKE patterns: escape % and _ in user input
  - IN clauses: use parameterized arrays
` + "```" + `

**XSS (Cross-Site Scripting) Prevention:**
` + "```" + `
XSS: attacker injects JavaScript into pages

Go's html/template package auto-escapes by default:
  tmpl := template.Must(template.New("page").Parse(
      "<p>Hello, {{.Name}}</p>"))
  // If Name = "<script>alert('xss')</script>"
  // Output: <p>Hello, &lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;</p>
  
  html/template knows context:
    {{.}} in <a href="{{.}}"> → URL-escapes
    {{.}} in <script>{{.}}</script> → JS-escapes
    {{.}} in regular HTML → HTML-escapes

Danger: text/template does NOT escape!
  ✗ import "text/template"    → XSS vulnerable!
  ✓ import "html/template"   → Auto-escapes!

API responses (JSON):
  json.Marshal automatically escapes HTML entities:
  < → \u003c, > → \u003e, & → \u0026
  
  Content-Type: application/json prevents browser executing HTML
  Always set: w.Header().Set("Content-Type", "application/json")
` + "```" + `

**CSRF (Cross-Site Request Forgery) Prevention:**
` + "```" + `
CSRF: attacker tricks user's browser into making unwanted requests

Prevention with tokens:
  1. Server generates random token per session
  2. Include token in forms/headers
  3. Server validates token on state-changing requests

Token generation:
  func generateCSRFToken() string {
      b := make([]byte, 32)
      rand.Read(b)
      return base64.URLEncoding.EncodeToString(b)
  }

Double-submit cookie pattern:
  1. Set CSRF token in cookie
  2. Client sends same token in header
  3. Server compares cookie value with header value
  
  Attacker can trigger cookie but can't read it
  → can't set the matching header

SameSite cookies (modern approach):
  http.SetCookie(w, &http.Cookie{
      Name:     "session",
      Value:    sessionID,
      HttpOnly: true,          // Not accessible via JavaScript
      Secure:   true,          // HTTPS only
      SameSite: http.SameSiteStrictMode, // No cross-site sending
      Path:     "/",
      MaxAge:   3600,
  })
` + "```" + `

**Security Headers:**
` + "```" + `
Essential HTTP security headers:

func SecurityHeaders(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Prevent MIME type sniffing
        w.Header().Set("X-Content-Type-Options", "nosniff")
        
        // Prevent clickjacking
        w.Header().Set("X-Frame-Options", "DENY")
        
        // XSS protection (legacy browsers)
        w.Header().Set("X-XSS-Protection", "1; mode=block")
        
        // Content Security Policy
        w.Header().Set("Content-Security-Policy",
            "default-src 'self'; script-src 'self'")
        
        // Strict Transport Security (HTTPS only)
        w.Header().Set("Strict-Transport-Security",
            "max-age=63072000; includeSubDomains")
        
        // Referrer Policy
        w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
        
        next.ServeHTTP(w, r)
    })
}

Rate limiting:
  Prevent brute force attacks on login
  Prevent DoS (resource exhaustion)
  
  Algorithms:
    Token bucket: steady rate with burst capacity
    Sliding window: count requests in moving time window
    Leaky bucket: smooth output rate regardless of input
    
  Implementation: use golang.org/x/time/rate
    limiter := rate.NewLimiter(rate.Every(time.Second), 10) // 10 req/s
    if !limiter.Allow() {
        http.Error(w, "rate limited", 429)
    }
` + "```" + ``,
					CodeExamples: `// Security patterns: validation, CSRF, secure cookies
package main

import (
    "crypto/rand"
    "encoding/base64"
    "encoding/json"
    "errors"
    "fmt"
    "net"
    "net/mail"
    "regexp"
    "strings"
    "unicode"
)

// Input validation library

type ValidationError struct {
    Field   string
    Message string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("%s: %s", e.Field, e.Message)
}

type Validator struct {
    errors []ValidationError
}

func (v *Validator) AddError(field, message string) {
    v.errors = append(v.errors, ValidationError{field, message})
}

func (v *Validator) HasErrors() bool {
    return len(v.errors) > 0
}

func (v *Validator) Errors() []ValidationError {
    return v.errors
}

// String validation
func (v *Validator) Required(field, value string) {
    if strings.TrimSpace(value) == "" {
        v.AddError(field, "is required")
    }
}

func (v *Validator) MinLength(field, value string, min int) {
    if len(value) < min {
        v.AddError(field, fmt.Sprintf("must be at least %d characters", min))
    }
}

func (v *Validator) MaxLength(field, value string, max int) {
    if len(value) > max {
        v.AddError(field, fmt.Sprintf("must be at most %d characters", max))
    }
}

func (v *Validator) Email(field, value string) {
    _, err := mail.ParseAddress(value)
    if err != nil {
        v.AddError(field, "must be a valid email address")
    }
}

var usernameRegex = regexp.MustCompile("^[a-zA-Z0-9_-]+$")

func (v *Validator) Username(field, value string) {
    if !usernameRegex.MatchString(value) {
        v.AddError(field, "must contain only letters, numbers, hyphens, and underscores")
    }
}

func (v *Validator) Password(field, value string) {
    var (
        hasUpper   bool
        hasLower   bool
        hasDigit   bool
        hasSpecial bool
    )
    for _, r := range value {
        switch {
        case unicode.IsUpper(r):
            hasUpper = true
        case unicode.IsLower(r):
            hasLower = true
        case unicode.IsDigit(r):
            hasDigit = true
        case unicode.IsPunct(r) || unicode.IsSymbol(r):
            hasSpecial = true
        }
    }
    
    if len(value) < 8 {
        v.AddError(field, "must be at least 8 characters")
    }
    if !hasUpper {
        v.AddError(field, "must contain an uppercase letter")
    }
    if !hasLower {
        v.AddError(field, "must contain a lowercase letter")
    }
    if !hasDigit {
        v.AddError(field, "must contain a digit")
    }
    if !hasSpecial {
        v.AddError(field, "must contain a special character")
    }
}

func (v *Validator) IP(field, value string) {
    if net.ParseIP(value) == nil {
        v.AddError(field, "must be a valid IP address")
    }
}

// CSRF token management
type CSRFManager struct {
    tokens map[string]bool
}

func NewCSRFManager() *CSRFManager {
    return &CSRFManager{tokens: make(map[string]bool)}
}

func (m *CSRFManager) Generate() string {
    b := make([]byte, 32)
    if _, err := rand.Read(b); err != nil {
        panic(err)
    }
    token := base64.URLEncoding.EncodeToString(b)
    m.tokens[token] = true
    return token
}

func (m *CSRFManager) Validate(token string) bool {
    if valid, ok := m.tokens[token]; ok && valid {
        delete(m.tokens, token) // One-time use
        return true
    }
    return false
}

// Secure JSON response
type APIResponse struct {
    Success bool        ` + "`" + `json:"success"` + "`" + `
    Data    any         ` + "`" + `json:"data,omitempty"` + "`" + `
    Error   string      ` + "`" + `json:"error,omitempty"` + "`" + `
    Errors  []string    ` + "`" + `json:"errors,omitempty"` + "`" + `
}

// Sanitize user input (prevent stored XSS)
func sanitizeInput(input string) string {
    // Remove null bytes
    input = strings.ReplaceAll(input, "\x00", "")
    // Trim whitespace
    input = strings.TrimSpace(input)
    // Limit length
    if len(input) > 10000 {
        input = input[:10000]
    }
    return input
}

// Validate request body
type CreateUserRequest struct {
    Username string ` + "`" + `json:"username"` + "`" + `
    Email    string ` + "`" + `json:"email"` + "`" + `
    Password string ` + "`" + `json:"password"` + "`" + `
}

func (r *CreateUserRequest) Validate() error {
    v := &Validator{}
    
    r.Username = sanitizeInput(r.Username)
    r.Email = sanitizeInput(r.Email)
    
    v.Required("username", r.Username)
    v.MinLength("username", r.Username, 3)
    v.MaxLength("username", r.Username, 30)
    v.Username("username", r.Username)
    
    v.Required("email", r.Email)
    v.Email("email", r.Email)
    
    v.Required("password", r.Password)
    v.Password("password", r.Password)
    
    if v.HasErrors() {
        messages := make([]string, len(v.Errors()))
        for i, e := range v.Errors() {
            messages[i] = e.Error()
        }
        return errors.New(strings.Join(messages, "; "))
    }
    return nil
}

func main() {
    // Validation
    fmt.Println("=== Input Validation ===")
    
    // Valid request
    req1 := CreateUserRequest{
        Username: "alice_42",
        Email:    "alice@example.com",
        Password: "SecureP@ss1",
    }
    if err := req1.Validate(); err != nil {
        fmt.Printf("  Valid request errors: %v\n", err)
    } else {
        fmt.Println("  Valid request: OK")
    }
    
    // Invalid request
    req2 := CreateUserRequest{
        Username: "a",
        Email:    "not-an-email",
        Password: "weak",
    }
    if err := req2.Validate(); err != nil {
        fmt.Printf("  Invalid request: %v\n", err)
    }
    
    // XSS attempt
    req3 := CreateUserRequest{
        Username: "<script>alert('xss')</script>",
        Email:    "evil@example.com",
        Password: "SecureP@ss1",
    }
    if err := req3.Validate(); err != nil {
        fmt.Printf("  XSS attempt: %v\n", err)
    }
    
    // CSRF tokens
    fmt.Println("\n=== CSRF Protection ===")
    csrf := NewCSRFManager()
    
    token := csrf.Generate()
    fmt.Printf("  Generated token: %s...\n", token[:20])
    fmt.Printf("  Valid (first use): %v\n", csrf.Validate(token))
    fmt.Printf("  Valid (reuse): %v\n", csrf.Validate(token)) // One-time use
    
    fmt.Printf("  Forged token: %v\n", csrf.Validate("fake-token"))
    
    // Secure JSON response
    fmt.Println("\n=== Secure API Responses ===")
    
    // Success response
    resp := APIResponse{
        Success: true,
        Data:    map[string]string{"id": "1", "name": "Alice"},
    }
    data, _ := json.MarshalIndent(resp, "  ", "  ")
    fmt.Printf("  Success:\n  %s\n", data)
    
    // Error response (no internal details leaked)
    errResp := APIResponse{
        Success: false,
        Error:   "invalid credentials",
        // NOT: "password hash mismatch for user alice in table users"
    }
    data, _ = json.MarshalIndent(errResp, "  ", "  ")
    fmt.Printf("  Error:\n  %s\n", data)
    
    // Validation error response
    valResp := APIResponse{
        Success: false,
        Errors:  []string{"username: too short", "email: invalid format"},
    }
    data, _ = json.MarshalIndent(valResp, "  ", "  ")
    fmt.Printf("  Validation:\n  %s\n", data)
}`,
				},
			},
		},
	})
}
