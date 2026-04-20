package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          2321,
			Title:       "Security Architecture and Threat Modeling",
			Description: "Design secure systems with defense in depth, zero trust architecture, OWASP top 10 mitigations, and comprehensive threat modeling approaches.",
			Order:       21,
			Lessons: []problems.Lesson{
				{
					Title: "Security Architecture Principles",
					Content: `Security must be designed into the architecture from the start, not bolted on later.

**Defense in Depth:**
  Multiple layers of security controls
  If one layer fails, others still protect the system

  Layers:
    Network: Firewalls, VPNs, network segmentation
    Infrastructure: OS hardening, patching, least privilege
    Application: Input validation, authentication, authorization
    Data: Encryption at rest and in transit, data classification
    Operations: Monitoring, incident response, auditing

**Zero Trust Architecture:**
  Never trust, always verify
  Assume the network is compromised

  Principles:
    Verify explicitly: Always authenticate and authorize
    Least privilege: Just-in-time and just-enough access
    Assume breach: Minimize blast radius, segment access

  Implementation:
    Identity-based access control (not network-based)
    Mutual TLS (mTLS) between services
    Short-lived credentials and tokens
    Continuous verification (not just at login)
    Microsegmentation of network

**Authentication Patterns:**

Token-Based (JWT):
  Stateless authentication
  Client gets token after login
  Token included in subsequent requests
  Token contains claims (user ID, roles, expiration)
  
  Advantages: Stateless, scalable
  Disadvantages: Cannot revoke individual tokens
  Mitigation: Short expiration + refresh tokens

  JWT Structure:
    Header: Algorithm and token type
    Payload: Claims (sub, exp, iat, custom)
    Signature: HMAC-SHA256 or RSA

OAuth 2.0 / OpenID Connect:
  Delegated authorization / authentication
  Flows:
    Authorization Code: Server-side apps (most secure)
    Authorization Code + PKCE: Mobile/SPA apps
    Client Credentials: Machine-to-machine
    
  Tokens:
    Access Token: Short-lived, grants API access
    Refresh Token: Long-lived, gets new access tokens
    ID Token: OIDC, contains user identity

API Key Authentication:
  Simple, suitable for server-to-server
  Include in header: X-API-Key
  Rate limit per key
  Rotate keys regularly

**Authorization Models:**

RBAC (Role-Based Access Control):
  Users -> Roles -> Permissions
  Simple, well-understood
  Challenge: Role explosion in complex systems
  
  Example:
    Admin: read, write, delete, manage-users
    Editor: read, write
    Viewer: read

ABAC (Attribute-Based Access Control):
  Rules based on attributes of subject, resource, action, environment
  More flexible than RBAC
  Policy: "Allow if user.department == resource.department AND action == read"
  
  Attributes:
    Subject: Role, department, clearance level
    Resource: Type, classification, owner
    Action: Read, write, delete
    Environment: Time, location, device

ReBAC (Relationship-Based Access Control):
  Authorization based on relationships between entities
  Google Zanzibar model
  "User can edit document if user is owner or member of group that has edit access"
  Graph-based authorization
  Tools: SpiceDB, Ory Keto, Authzed

**OWASP Top 10 Architectural Mitigations:**

1. Broken Access Control:
   Deny by default
   Enforce access control at the server side
   Centralize authorization logic
   Audit access control failures

2. Cryptographic Failures:
   Encrypt data at rest (AES-256)
   Encrypt data in transit (TLS 1.3)
   Use strong key management (AWS KMS, HashiCorp Vault)
   Hash passwords with bcrypt/argon2

3. Injection:
   Parameterized queries for SQL
   Input validation at system boundaries
   Output encoding for XSS
   Content Security Policy headers

4. Insecure Design:
   Threat modeling during design
   Secure design patterns
   Abuse case testing
   Principle of least privilege

5. Security Misconfiguration:
   Automated hardening processes
   Minimal installation (no unnecessary features)
   Regular security scanning
   Infrastructure as Code with security baselines

**Threat Modeling:**

STRIDE Model:
  Spoofing: Pretending to be someone else
    Mitigation: Strong authentication
  Tampering: Modifying data or code
    Mitigation: Integrity checks, signing
  Repudiation: Denying an action occurred
    Mitigation: Audit logging, non-repudiation
  Information Disclosure: Exposing data to unauthorized parties
    Mitigation: Encryption, access control
  Denial of Service: Making system unavailable
    Mitigation: Rate limiting, auto-scaling
  Elevation of Privilege: Gaining unauthorized access
    Mitigation: Least privilege, input validation

Process:
  1. Diagram the system (data flow diagrams)
  2. Identify threats using STRIDE per element
  3. Rank threats (DREAD or risk matrix)
  4. Mitigate threats
  5. Validate mitigations

**Secrets Management:**

  Never store secrets in code or config files
  Use secrets management tools:
    HashiCorp Vault
    AWS Secrets Manager
    Azure Key Vault
    GCP Secret Manager
  
  Rotate secrets automatically
  Use short-lived dynamic secrets when possible
  Audit secret access`,
					CodeExamples: `// Security architecture patterns

// Authentication middleware with JWT
type AuthMiddleware struct {
    jwtSecret []byte
    logger    *Logger
}

type Claims struct {
    UserID string   "json:\"user_id\""
    Roles  []string "json:\"roles\""
    jwt.RegisteredClaims
}

func (m *AuthMiddleware) Authenticate(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        token := extractBearerToken(r)
        if token == "" {
            http.Error(w, "unauthorized", http.StatusUnauthorized)
            return
        }
        
        claims, err := m.validateToken(token)
        if err != nil {
            m.logger.Warn("invalid token", "error", err, "remote_addr", r.RemoteAddr)
            http.Error(w, "unauthorized", http.StatusUnauthorized)
            return
        }
        
        ctx := context.WithValue(r.Context(), userClaimsKey, claims)
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

func (m *AuthMiddleware) validateToken(tokenStr string) (*Claims, error) {
    claims := &Claims{}
    token, err := jwt.ParseWithClaims(tokenStr, claims, func(t *jwt.Token) (interface{}, error) {
        if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
            return nil, fmt.Errorf("unexpected signing method: %v", t.Header["alg"])
        }
        return m.jwtSecret, nil
    })
    if err != nil || !token.Valid {
        return nil, fmt.Errorf("invalid token: %w", err)
    }
    return claims, nil
}

func extractBearerToken(r *http.Request) string {
    auth := r.Header.Get("Authorization")
    if !strings.HasPrefix(auth, "Bearer ") {
        return ""
    }
    return strings.TrimPrefix(auth, "Bearer ")
}

// RBAC Authorization
type Permission string
const (
    PermRead       Permission = "read"
    PermWrite      Permission = "write"
    PermDelete     Permission = "delete"
    PermManageUsers Permission = "manage_users"
)

type RBACAuthorizer struct {
    rolePermissions map[string][]Permission
}

func NewRBACAuthorizer() *RBACAuthorizer {
    return &RBACAuthorizer{
        rolePermissions: map[string][]Permission{
            "admin":  {PermRead, PermWrite, PermDelete, PermManageUsers},
            "editor": {PermRead, PermWrite},
            "viewer": {PermRead},
        },
    }
}

func (a *RBACAuthorizer) HasPermission(roles []string, required Permission) bool {
    for _, role := range roles {
        perms, ok := a.rolePermissions[role]
        if !ok {
            continue
        }
        for _, p := range perms {
            if p == required {
                return true
            }
        }
    }
    return false
}

func (a *RBACAuthorizer) Authorize(perm Permission) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            claims, ok := r.Context().Value(userClaimsKey).(*Claims)
            if !ok {
                http.Error(w, "unauthorized", http.StatusUnauthorized)
                return
            }
            if !a.HasPermission(claims.Roles, perm) {
                http.Error(w, "forbidden", http.StatusForbidden)
                return
            }
            next.ServeHTTP(w, r)
        })
    }
}

// Rate limiter (token bucket)
type RateLimiter struct {
    mu       sync.Mutex
    buckets  map[string]*tokenBucket
    rate     float64
    capacity int
}

type tokenBucket struct {
    tokens   float64
    lastTime time.Time
}

func NewRateLimiter(requestsPerSecond float64, burstSize int) *RateLimiter {
    return &RateLimiter{
        buckets:  make(map[string]*tokenBucket),
        rate:     requestsPerSecond,
        capacity: burstSize,
    }
}

func (rl *RateLimiter) Allow(key string) bool {
    rl.mu.Lock()
    defer rl.mu.Unlock()
    
    now := time.Now()
    bucket, exists := rl.buckets[key]
    if !exists {
        rl.buckets[key] = &tokenBucket{
            tokens:   float64(rl.capacity) - 1,
            lastTime: now,
        }
        return true
    }
    
    elapsed := now.Sub(bucket.lastTime).Seconds()
    bucket.tokens += elapsed * rl.rate
    if bucket.tokens > float64(rl.capacity) {
        bucket.tokens = float64(rl.capacity)
    }
    bucket.lastTime = now
    
    if bucket.tokens < 1 {
        return false
    }
    bucket.tokens--
    return true
}

func (rl *RateLimiter) Middleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        key := r.RemoteAddr
        if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
            key = strings.Split(forwarded, ",")[0]
        }
        
        if !rl.Allow(key) {
            w.Header().Set("Retry-After", "1")
            http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
            return
        }
        next.ServeHTTP(w, r)
    })
}

// Input validation
type CreateUserRequest struct {
    Email    string "json:\"email\""
    Name     string "json:\"name\""
    Password string "json:\"password\""
}

func (r *CreateUserRequest) Validate() []ValidationError {
    var errs []ValidationError
    
    if r.Email == "" {
        errs = append(errs, ValidationError{Field: "email", Message: "required"})
    } else if !isValidEmail(r.Email) {
        errs = append(errs, ValidationError{Field: "email", Message: "invalid format"})
    }
    
    if r.Name == "" {
        errs = append(errs, ValidationError{Field: "name", Message: "required"})
    } else if len(r.Name) > 255 {
        errs = append(errs, ValidationError{Field: "name", Message: "too long"})
    }
    
    if len(r.Password) < 12 {
        errs = append(errs, ValidationError{Field: "password", Message: "minimum 12 characters"})
    }
    
    return errs
}

// Audit logging
type AuditLogger struct {
    store AuditStore
}

type AuditEntry struct {
    Timestamp time.Time              "json:\"timestamp\""
    UserID    string                 "json:\"user_id\""
    Action    string                 "json:\"action\""
    Resource  string                 "json:\"resource\""
    Details   map[string]interface{} "json:\"details\""
    IP        string                 "json:\"ip\""
    Status    string                 "json:\"status\""
}

func (al *AuditLogger) Log(ctx context.Context, action, resource string, details map[string]interface{}) {
    claims, _ := ctx.Value(userClaimsKey).(*Claims)
    userID := "anonymous"
    if claims != nil {
        userID = claims.UserID
    }
    
    entry := AuditEntry{
        Timestamp: time.Now().UTC(),
        UserID:    userID,
        Action:    action,
        Resource:  resource,
        Details:   details,
        IP:        extractIP(ctx),
        Status:    "success",
    }
    
    al.store.Save(ctx, entry)
}`,
				},
			},
		},
		{
			ID:          2322,
			Title:       "Resilience and Fault Tolerance Patterns",
			Description: "Design resilient systems with circuit breakers, bulkheads, retry strategies, chaos engineering, and disaster recovery planning.",
			Order:       22,
			Lessons: []problems.Lesson{
				{
					Title: "Building Resilient Distributed Systems",
					Content: `Distributed systems will fail. Resilience patterns help systems survive and recover from failures gracefully.

**Circuit Breaker Pattern:**

States:
  Closed (normal):
    Requests flow through normally
    Track failure count
    If failures exceed threshold -> Open
    
  Open (failing):
    Requests immediately fail/fallback
    Timer starts
    After timeout -> Half-Open
    
  Half-Open (testing):
    Allow limited requests through
    If successful -> Closed
    If fails -> Open

  ┌──────────┐  failure threshold  ┌──────────┐
  │  Closed   │ ─────────────────> │   Open    │
  │  (normal) │                    │ (failing) │
  └──────────┘ <───────────────── └──────────┘
        ^        success in           │
        │        half-open            │ timeout
        │                            v
        │                     ┌────────────┐
        └──────────────────── │ Half-Open  │
           success            │ (testing)  │
                              └────────────┘

**Retry Patterns:**

Simple Retry:
  Fixed delay between attempts
  3 attempts with 1-second delay
  Risk: Thundering herd

Exponential Backoff:
  Increasing delay: 1s, 2s, 4s, 8s, 16s
  Add jitter (random component) to prevent synchronized retries
  Full jitter: sleep = random(0, min(cap, base * 2^attempt))
  Decorrelated jitter: sleep = min(cap, random(base, sleep * 3))

Retry with Circuit Breaker:
  Don't retry if circuit is open
  Retries feed into circuit breaker failure count
  Prevents retry storms during outages

When NOT to retry:
  400 Bad Request (client error, will always fail)
  401/403 (auth issues)
  404 Not Found
  Idempotency not guaranteed

When to retry:
  429 Too Many Requests (with Retry-After header)
  500, 502, 503, 504 (transient server errors)
  Network timeouts
  Connection refused

**Bulkhead Pattern:**
  Isolate failures to prevent cascading
  Like watertight compartments in a ship
  
  Implementation:
    Thread pool isolation: Separate pools per dependency
    Semaphore isolation: Limit concurrent calls
    Process isolation: Separate processes per service
    
  Example:
    Payment Service: max 10 concurrent calls
    Inventory Service: max 20 concurrent calls
    If Payment is slow, it only affects its own pool
    Inventory continues working normally

**Timeout Patterns:**

Connection Timeout:
  Max time to establish TCP connection
  Usually 1-5 seconds
  
Request Timeout:
  Max time for the complete request
  Should be based on SLO requirements
  
Cascading Timeouts:
  Outer service timeout > inner service timeout
  API Gateway: 30s > Order Service: 10s > Payment: 5s
  Each layer has less time to prevent pileup

**Fallback Strategies:**

Cache Fallback:
  Return cached data when service is unavailable
  Stale data is often better than no data
  Useful for read-heavy, eventually consistent

Default Value:
  Return a reasonable default
  "Recommendations unavailable" instead of error
  
Degraded Service:
  Provide reduced functionality
  Show cached product catalog without personalization
  Accept orders without real-time inventory check

Queue for Later:
  Accept the request, process later
  Use message queue for eventual processing
  Acknowledge receipt, process when service recovers

**Chaos Engineering:**

Principles:
  1. Define steady state (normal behavior metrics)
  2. Hypothesize that steady state continues during failure
  3. Introduce real-world events (server crash, network issues)
  4. Try to disprove the hypothesis
  5. Fix any weaknesses found

Types of Chaos:
  Infrastructure: Kill instances, fill disks, CPU stress
  Network: Latency injection, packet loss, DNS failure
  Application: Exception injection, memory leaks
  Dependencies: Block external services, slow responses

Tools:
  Chaos Monkey (Netflix): Random instance termination
  Litmus: Kubernetes chaos engineering
  Gremlin: Commercial chaos platform
  Toxiproxy: Network failure simulation

**Disaster Recovery:**

RPO (Recovery Point Objective):
  Maximum acceptable data loss
  How much data can you afford to lose?
  Determines backup frequency

RTO (Recovery Time Objective):
  Maximum acceptable downtime
  How quickly must you recover?
  Determines recovery strategy

Strategies (by RTO):
  Backup/Restore: RTO hours-days, cheapest
  Pilot Light: RTO 10min-hours, infrastructure ready but scaled down
  Warm Standby: RTO minutes, scaled-down copy running
  Multi-Site Active/Active: RTO near-zero, most expensive

Data Replication:
  Synchronous: Zero data loss, higher latency
  Asynchronous: Near-zero data loss, lower latency
  Semi-synchronous: One replica sync, rest async`,
					CodeExamples: `// Resilience patterns implementation

// Circuit Breaker
type CircuitBreaker struct {
    mu            sync.Mutex
    state         CircuitState
    failureCount  int
    successCount  int
    threshold     int
    timeout       time.Duration
    lastFailure   time.Time
    halfOpenMax   int
    onStateChange func(from, to CircuitState)
}

type CircuitState int
const (
    CircuitClosed CircuitState = iota
    CircuitOpen
    CircuitHalfOpen
)

func NewCircuitBreaker(threshold int, timeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        state:       CircuitClosed,
        threshold:   threshold,
        timeout:     timeout,
        halfOpenMax: 3,
    }
}

func (cb *CircuitBreaker) Execute(fn func() error) error {
    cb.mu.Lock()
    
    switch cb.state {
    case CircuitOpen:
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.setState(CircuitHalfOpen)
            cb.successCount = 0
        } else {
            cb.mu.Unlock()
            return ErrCircuitOpen
        }
    case CircuitHalfOpen:
        // Allow limited requests
    }
    
    cb.mu.Unlock()
    
    err := fn()
    
    cb.mu.Lock()
    defer cb.mu.Unlock()
    
    if err != nil {
        cb.failureCount++
        cb.lastFailure = time.Now()
        
        if cb.state == CircuitHalfOpen {
            cb.setState(CircuitOpen)
        } else if cb.failureCount >= cb.threshold {
            cb.setState(CircuitOpen)
        }
        return err
    }
    
    if cb.state == CircuitHalfOpen {
        cb.successCount++
        if cb.successCount >= cb.halfOpenMax {
            cb.setState(CircuitClosed)
            cb.failureCount = 0
        }
    } else {
        cb.failureCount = 0
    }
    
    return nil
}

func (cb *CircuitBreaker) setState(state CircuitState) {
    old := cb.state
    cb.state = state
    if cb.onStateChange != nil {
        cb.onStateChange(old, state)
    }
}

// Retry with exponential backoff and jitter
type RetryConfig struct {
    MaxAttempts int
    BaseDelay   time.Duration
    MaxDelay    time.Duration
    Retryable   func(error) bool
}

func RetryWithBackoff(ctx context.Context, cfg RetryConfig, fn func() error) error {
    var lastErr error
    
    for attempt := 0; attempt < cfg.MaxAttempts; attempt++ {
        lastErr = fn()
        if lastErr == nil {
            return nil
        }
        
        if cfg.Retryable != nil && !cfg.Retryable(lastErr) {
            return lastErr
        }
        
        if attempt == cfg.MaxAttempts-1 {
            break
        }
        
        delay := calculateBackoff(attempt, cfg.BaseDelay, cfg.MaxDelay)
        
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-time.After(delay):
        }
    }
    
    return fmt.Errorf("all %d attempts failed, last error: %w", cfg.MaxAttempts, lastErr)
}

func calculateBackoff(attempt int, base, max time.Duration) time.Duration {
    backoff := float64(base) * math.Pow(2, float64(attempt))
    if backoff > float64(max) {
        backoff = float64(max)
    }
    // Full jitter
    jitter := rand.Float64() * backoff
    return time.Duration(jitter)
}

// Bulkhead (semaphore-based)
type Bulkhead struct {
    name     string
    sem      chan struct{}
    timeout  time.Duration
    metrics  *BulkheadMetrics
}

type BulkheadMetrics struct {
    accepted  int64
    rejected  int64
    timedOut  int64
}

func NewBulkhead(name string, maxConcurrent int, timeout time.Duration) *Bulkhead {
    return &Bulkhead{
        name:    name,
        sem:     make(chan struct{}, maxConcurrent),
        timeout: timeout,
        metrics: &BulkheadMetrics{},
    }
}

func (b *Bulkhead) Execute(ctx context.Context, fn func() error) error {
    timer := time.NewTimer(b.timeout)
    defer timer.Stop()
    
    select {
    case b.sem <- struct{}{}:
        atomic.AddInt64(&b.metrics.accepted, 1)
        defer func() { <-b.sem }()
        return fn()
    case <-timer.C:
        atomic.AddInt64(&b.metrics.timedOut, 1)
        return fmt.Errorf("bulkhead %s: timeout waiting for slot", b.name)
    case <-ctx.Done():
        return ctx.Err()
    }
}

// Timeout wrapper
func WithTimeout(ctx context.Context, timeout time.Duration, fn func(context.Context) error) error {
    ctx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()
    
    done := make(chan error, 1)
    go func() {
        done <- fn(ctx)
    }()
    
    select {
    case err := <-done:
        return err
    case <-ctx.Done():
        return fmt.Errorf("operation timed out after %v", timeout)
    }
}

// Resilient HTTP client combining patterns
type ResilientClient struct {
    httpClient     *http.Client
    circuitBreaker *CircuitBreaker
    bulkhead       *Bulkhead
    retryConfig    RetryConfig
    logger         *Logger
}

func (c *ResilientClient) Do(ctx context.Context, req *http.Request) (*http.Response, error) {
    var resp *http.Response
    
    err := c.bulkhead.Execute(ctx, func() error {
        return c.circuitBreaker.Execute(func() error {
            return RetryWithBackoff(ctx, c.retryConfig, func() error {
                var err error
                resp, err = c.httpClient.Do(req.WithContext(ctx))
                if err != nil {
                    return err
                }
                if resp.StatusCode >= 500 {
                    resp.Body.Close()
                    return fmt.Errorf("server error: %d", resp.StatusCode)
                }
                return nil
            })
        })
    })
    
    return resp, err
}`,
				},
			},
		},
	})
}
