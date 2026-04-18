package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          1619,
			Title:       "Production Go Patterns",
			Description: "Master patterns used in production Go services: graceful shutdown, dependency injection, configuration management, structured logging, and health checks.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "Graceful Shutdown",
					Content: `Every production Go service must handle shutdown cleanly. A "hard kill" can corrupt data, drop requests, or leave resources in a bad state.

**Why Graceful Shutdown Matters:**
- In-flight HTTP requests need to complete (not get dropped mid-response)
- Database connections need to close properly (not leave open transactions)
- Background goroutines need to stop (not leak resources)
- Kubernetes sends SIGTERM before SIGKILL (you have ~30s by default)

**The Pattern:**

` + "```" + `
1. Catch OS signals (SIGTERM, SIGINT)
2. Stop accepting new work
3. Wait for in-flight work to finish (with timeout)
4. Close resources (DB, caches, message queues)
5. Exit cleanly
` + "```" + `

**Signal Handling:**
- SIGINT: Sent by Ctrl+C in terminal
- SIGTERM: Sent by Kubernetes, Docker, systemd (polite "please stop")
- SIGKILL: Cannot be caught (forced kill after grace period)

**Critical Mistakes:**
1. Not calling cancel() on context — leak goroutines
2. No timeout on shutdown — hang forever if a handler is stuck
3. Closing DB before handlers finish — handlers get "connection closed" errors
4. Using os.Exit() directly — deferred cleanups don't run`,
					CodeExamples: `package main

import (
    "context"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"
)

func main() {
    // 1. Create the server
    srv := &http.Server{
        Addr:         ":8080",
        ReadTimeout:  5 * time.Second,
        WriteTimeout: 10 * time.Second,
        IdleTimeout:  120 * time.Second,
    }

    // 2. Start serving in a goroutine
    go func() {
        log.Println("starting server on :8080")
        if err := srv.ListenAndServe(); err != http.ErrServerClosed {
            log.Fatalf("server error: %v", err)
        }
    }()

    // 3. Wait for shutdown signal
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    sig := <-quit
    log.Printf("received signal %s, shutting down...", sig)

    // 4. Create shutdown context with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    // 5. Gracefully shutdown (waits for in-flight requests)
    if err := srv.Shutdown(ctx); err != nil {
        log.Printf("forced shutdown: %v", err)
    }

    // 6. Close other resources
    // db.Close()
    // cache.Close()

    log.Println("server stopped cleanly")
}`,
				},
				{
					Title: "Dependency Injection in Go",
					Content: `Dependency Injection (DI) in Go is about passing dependencies (like a database or logger) into structs rather than creating them internally. Go achieves DI through constructor functions and interfaces — no framework needed.

**Why DI?**
1. **Testability**: Swap real dependencies for mocks in tests
2. **Flexibility**: Change implementations without modifying consumers
3. **Explicit dependencies**: Every struct's constructor shows exactly what it needs

**The Go Approach (No Framework):**
Go typically uses "constructor injection" — pass dependencies through a NewXxx() function:

` + "```" + `
// BAD: Hidden dependency (hard to test)
type UserService struct{}
func (s *UserService) GetUser(id string) (*User, error) {
    db := database.GetGlobalDB()  // Hidden! Can't replace in tests
    return db.FindUser(id)
}

// GOOD: Explicit dependency
type UserService struct {
    db UserStore  // Interface, not concrete type
}
func NewUserService(db UserStore) *UserService {
    return &UserService{db: db}
}
` + "```" + `

**The Interface Trick:**
Define interfaces where they're USED, not where they're implemented. This is the opposite of Java.

` + "```" + `
// In your service package (consumer):
type UserStore interface {
    FindUser(ctx context.Context, id string) (*User, error)
}

// The real DB implementation satisfies this implicitly.
// Your test mock also satisfies this.
` + "```" + `

**When to Use a DI Framework (wire, fx):**
Only for large applications with 50+ services where manual wiring becomes tedious. Most Go projects do fine without one.`,
					CodeExamples: `// Define interfaces at the consumer
type UserRepository interface {
    GetByID(ctx context.Context, id string) (*User, error)
    Create(ctx context.Context, u *User) error
}

type EmailSender interface {
    Send(ctx context.Context, to, subject, body string) error
}

// Service with injected dependencies
type UserService struct {
    repo  UserRepository
    email EmailSender
    log   *slog.Logger
}

func NewUserService(repo UserRepository, email EmailSender, log *slog.Logger) *UserService {
    return &UserService{repo: repo, email: email, log: log}
}

func (s *UserService) Register(ctx context.Context, u *User) error {
    if err := s.repo.Create(ctx, u); err != nil {
        return fmt.Errorf("create user: %w", err)
    }
    
    // Non-critical: send welcome email
    go func() {
        if err := s.email.Send(ctx, u.Email, "Welcome!", "..."); err != nil {
            s.log.Error("failed to send welcome email", "err", err)
        }
    }()
    
    return nil
}

// In tests: use mock implementations
type mockRepo struct {
    users map[string]*User
}

func (m *mockRepo) GetByID(_ context.Context, id string) (*User, error) {
    u, ok := m.users[id]
    if !ok {
        return nil, ErrNotFound
    }
    return u, nil
}

func (m *mockRepo) Create(_ context.Context, u *User) error {
    m.users[u.ID] = u
    return nil
}

func TestRegister(t *testing.T) {
    repo := &mockRepo{users: make(map[string]*User)}
    email := &mockEmailSender{}
    log := slog.New(slog.NewTextHandler(io.Discard, nil))
    
    svc := NewUserService(repo, email, log)
    err := svc.Register(context.Background(), &User{ID: "1", Name: "Alice"})
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
}`,
				},
				{
					Title: "Table-Driven Tests Deep Dive",
					Content: `Table-driven tests are THE Go testing pattern. They reduce duplication, make adding test cases trivial, and produce clear failure messages.

**The Standard Pattern:**

` + "```" + `
func TestXxx(t *testing.T) {
    tests := []struct {
        name     string    // Descriptive sub-test name
        input    InputType // Test input
        want     OutputType // Expected output
        wantErr  bool      // Whether an error is expected
    }{
        {name: "...", input: ..., want: ..., wantErr: false},
        {name: "...", input: ..., want: ..., wantErr: true},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got, err := FunctionUnderTest(tt.input)
            if (err != nil) != tt.wantErr {
                t.Errorf("error = %v, wantErr %v", err, tt.wantErr)
                return
            }
            if got != tt.want {
                t.Errorf("got %v, want %v", got, tt.want)
            }
        })
    }
}
` + "```" + `

**Best Practices:**
1. **Use t.Run()**: Creates named sub-tests. Run a single case with ` + "`" + `go test -run TestXxx/case_name` + "`" + `
2. **Descriptive names**: ` + "`" + `"empty input"` + "`" + `, ` + "`" + `"negative number"` + "`" + `, not ` + "`" + `"test1"` + "`" + `
3. **Include edge cases**: nil, empty, zero, max, negative, Unicode
4. **Parallel sub-tests**: Add t.Parallel() to run cases concurrently
5. **testdata/ directory**: Store fixtures in ` + "`" + `testdata/` + "`" + ` (ignored by go build)

**Golden File Testing:**
For complex outputs (JSON, HTML), compare against a "golden" file. Update with ` + "`" + `-update` + "`" + ` flag:

` + "```" + `
var update = flag.Bool("update", false, "update golden files")

func TestOutput(t *testing.T) {
    got := generateOutput()
    golden := filepath.Join("testdata", t.Name()+".golden")
    if *update {
        os.WriteFile(golden, got, 0644)
    }
    want, _ := os.ReadFile(golden)
    if !bytes.Equal(got, want) {
        t.Errorf("output mismatch, run with -update to update golden file")
    }
}
` + "```" + `

**Testing HTTP Handlers:**
Use httptest.NewServer() for integration tests and httptest.NewRecorder() for unit tests.`,
					CodeExamples: `func TestAdd(t *testing.T) {
    tests := []struct {
        name string
        a, b int
        want int
    }{
        {"positive numbers", 2, 3, 5},
        {"zero", 0, 0, 0},
        {"negative", -1, -2, -3},
        {"mixed signs", -5, 10, 5},
        {"large numbers", 1<<30, 1<<30, 1<<31},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            if got := Add(tt.a, tt.b); got != tt.want {
                t.Errorf("Add(%d, %d) = %d, want %d", tt.a, tt.b, got, tt.want)
            }
        })
    }
}

// Testing HTTP handlers with httptest
func TestHealthHandler(t *testing.T) {
    tests := []struct {
        name       string
        method     string
        wantStatus int
        wantBody   string
    }{
        {"GET returns 200", "GET", http.StatusOK, ` + "`" + `{"status":"ok"}` + "`" + `},
        {"POST returns 405", "POST", http.StatusMethodNotAllowed, ""},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            req := httptest.NewRequest(tt.method, "/health", nil)
            rec := httptest.NewRecorder()
            
            HealthHandler(rec, req)
            
            if rec.Code != tt.wantStatus {
                t.Errorf("status = %d, want %d", rec.Code, tt.wantStatus)
            }
            if tt.wantBody != "" && strings.TrimSpace(rec.Body.String()) != tt.wantBody {
                t.Errorf("body = %q, want %q", rec.Body.String(), tt.wantBody)
            }
        })
    }
}

// Parallel sub-tests for independent cases
func TestParseConfig(t *testing.T) {
    tests := []struct {
        name    string
        input   string
        wantErr bool
    }{
        {"valid config", ` + "`" + `{"port": 8080}` + "`" + `, false},
        {"empty input", "", true},
        {"invalid json", "{bad}", true},
    }

    for _, tt := range tests {
        tt := tt // Capture range variable (not needed in Go 1.22+)
        t.Run(tt.name, func(t *testing.T) {
            t.Parallel()
            _, err := ParseConfig([]byte(tt.input))
            if (err != nil) != tt.wantErr {
                t.Errorf("ParseConfig() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}`,
				},
				{
					Title: "Structured Logging with slog",
					Content: `Go 1.21 introduced ` + "`" + `log/slog` + "`" + ` — a structured logging package in the standard library. It replaces the old ` + "`" + `log` + "`" + ` package for production use.

**Why Structured Logging?**
Plain text logs like ` + "`" + `"user created: alice"` + "`" + ` are hard to parse, filter, and query. Structured logs output key-value pairs (often as JSON) that can be indexed by log aggregation systems (ELK, Datadog, Grafana Loki).

**slog Basics:**

` + "```" + `
// Plain text: hard to parse
log.Println("user created:", username)

// Structured: easy to query
slog.Info("user created", "username", username, "email", email)
// Output: {"time":"...","level":"INFO","msg":"user created","username":"alice","email":"alice@example.com"}
` + "```" + `

**Log Levels:**
- slog.Debug: Development details (disabled by default)
- slog.Info: Normal operations
- slog.Warn: Something unexpected but non-fatal
- slog.Error: Something failed

**Handlers:**
- ` + "`" + `slog.NewTextHandler(os.Stderr, opts)` + "`" + ` — Human-readable (dev)
- ` + "`" + `slog.NewJSONHandler(os.Stderr, opts)` + "`" + ` — Machine-readable (production)

**Best Practices:**
1. Use ` + "`" + `slog.With()` + "`" + ` to add context that applies to all subsequent logs
2. Pass logger via dependency injection (not global)
3. Include request IDs, user IDs, trace IDs for debugging
4. Don't log sensitive data (passwords, tokens, PII)
5. Use ` + "`" + `slog.Group()` + "`" + ` for nested attributes`,
					CodeExamples: `package main

import (
    "context"
    "log/slog"
    "net/http"
    "os"
)

func main() {
    // Production: JSON handler
    logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
        Level: slog.LevelInfo,
    }))
    slog.SetDefault(logger)

    // Basic usage
    slog.Info("server starting", "port", 8080, "env", "production")
    // {"time":"...","level":"INFO","msg":"server starting","port":8080,"env":"production"}

    // With context (add fields to all subsequent logs)
    userLogger := logger.With("user_id", "u123", "request_id", "req-abc")
    userLogger.Info("processing request")
    userLogger.Error("database query failed", "query", "SELECT ...", "err", err)

    // Groups for nested JSON
    slog.Info("request",
        slog.Group("http",
            slog.String("method", "GET"),
            slog.String("path", "/api/users"),
            slog.Int("status", 200),
        ),
        slog.Group("timing",
            slog.Duration("latency", elapsed),
        ),
    )
    // {"time":"...","msg":"request","http":{"method":"GET","path":"/api/users","status":200},"timing":{"latency":"1.234ms"}}
}

// Middleware that adds request context to logger
func loggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        logger := slog.With(
            "method", r.Method,
            "path", r.URL.Path,
            "remote_addr", r.RemoteAddr,
        )
        
        // Store logger in context for handlers to use
        ctx := context.WithValue(r.Context(), loggerKey, logger)
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}`,
				},
				{
					Title: "Error Wrapping and Sentinel Errors",
					Content: `Go 1.13 introduced error wrapping with ` + "`" + `%w` + "`" + ` and checking with ` + "`" + `errors.Is()` + "`" + ` / ` + "`" + `errors.As()` + "`" + `. This is the modern way to handle errors in Go.

**Three Error Strategies:**

**1. Sentinel Errors** (predefined error values):
` + "```" + `
var ErrNotFound = errors.New("not found")
var ErrUnauthorized = errors.New("unauthorized")
` + "```" + `
Check with ` + "`" + `errors.Is(err, ErrNotFound)` + "`" + `. Use for known, expected conditions.

**2. Custom Error Types** (errors with extra data):
` + "```" + `
type ValidationError struct {
    Field   string
    Message string
}
func (e *ValidationError) Error() string { ... }
` + "```" + `
Check with ` + "`" + `errors.As(err, &target)` + "`" + `. Use when callers need to inspect error details.

**3. Error Wrapping** (add context while preserving the original):
` + "```" + `
return fmt.Errorf("create user %s: %w", name, err)
` + "```" + `
The ` + "`" + `%w` + "`" + ` verb wraps the error. ` + "`" + `errors.Is()` + "`" + ` can unwrap and find the original.

**The Error Wrapping Chain:**
` + "```" + `
Handler: "handle request: create user alice: insert row: connection refused"
           ↓ errors.Is(err, ErrConnectionRefused) → true at any level!
` + "```" + `

**Best Practices:**
1. Always add context when returning errors: ` + "`" + `fmt.Errorf("operation: %w", err)` + "`" + `
2. Use ` + "`" + `%w` + "`" + ` (wrapping) when callers should check the underlying error
3. Use ` + "`" + `%v` + "`" + ` (formatting only) when you want to hide the original error
4. Don't wrap errors with the same message as the original
5. Log errors at the top level, not at every layer (avoids duplicate logs)`,
					CodeExamples: `// Sentinel errors
var (
    ErrNotFound     = errors.New("not found")
    ErrConflict     = errors.New("already exists")
    ErrUnauthorized = errors.New("unauthorized")
)

// Custom error type with details
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation failed on %s: %s", e.Field, e.Message)
}

// Repository layer: wrap with context
func (r *UserRepo) GetByID(ctx context.Context, id string) (*User, error) {
    row := r.db.QueryRowContext(ctx, "SELECT ... WHERE id = $1", id)
    var u User
    if err := row.Scan(&u.ID, &u.Name); err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return nil, ErrNotFound  // Convert DB error to domain error
        }
        return nil, fmt.Errorf("query user %s: %w", id, err)
    }
    return &u, nil
}

// Service layer: wrap again
func (s *UserService) GetUser(ctx context.Context, id string) (*User, error) {
    u, err := s.repo.GetByID(ctx, id)
    if err != nil {
        return nil, fmt.Errorf("get user: %w", err)
    }
    return u, nil
}

// Handler layer: check and respond
func (h *Handler) GetUser(w http.ResponseWriter, r *http.Request) {
    u, err := h.svc.GetUser(r.Context(), r.PathValue("id"))
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            http.Error(w, "user not found", http.StatusNotFound)
            return
        }
        slog.Error("get user failed", "err", err)
        http.Error(w, "internal error", http.StatusInternalServerError)
        return
    }
    json.NewEncoder(w).Encode(u)
}

// errors.As: extract custom error type
func handleError(err error) {
    var ve *ValidationError
    if errors.As(err, &ve) {
        fmt.Printf("Invalid field %s: %s\n", ve.Field, ve.Message)
        return
    }
    fmt.Println("Unknown error:", err)
}`,
				},
			},
		},
		{
			ID:          1620,
			Title:       "Go Performance & Profiling",
			Description: "Learn to profile Go applications with pprof, optimize memory allocations, understand escape analysis, and write efficient concurrent code.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Profiling with pprof",
					Content: `Go has world-class profiling built into the standard library. The ` + "`" + `pprof` + "`" + ` tool helps you find where your program spends CPU time, allocates memory, and blocks on synchronization.

**Types of Profiles:**
1. **CPU Profile**: Where does the program spend time?
2. **Memory (Heap) Profile**: Where does the program allocate memory?
3. **Goroutine Profile**: What are all goroutines doing? (Detect goroutine leaks)
4. **Block Profile**: Where do goroutines block waiting on sync primitives?
5. **Mutex Profile**: Where is there lock contention?

**Two Ways to Profile:**

**1. net/http/pprof (production-safe):**
Import ` + "`" + `_ "net/http/pprof"` + "`" + ` and run a debug HTTP server. Access profiles via browser or ` + "`" + `go tool pprof` + "`" + `.

**2. runtime/pprof (in tests/benchmarks):**
Use ` + "`" + `go test -cpuprofile cpu.prof -memprofile mem.prof -bench .` + "`" + `

**Reading pprof Output:**
` + "```" + `
go tool pprof http://localhost:6060/debug/pprof/heap
(pprof) top 10          # Top 10 memory-allocating functions
(pprof) list funcName   # Line-by-line profile for a function
(pprof) web             # Visual graph in browser
` + "```" + `

**The Golden Rule:**
Profile FIRST, optimize SECOND. Never guess where the bottleneck is. Most performance issues are in 1-2 functions, not spread across the codebase.

**Common Findings:**
- String concatenation in loops (use strings.Builder)
- Unnecessary allocations (pre-size slices, reuse buffers)
- JSON marshaling/unmarshaling (use streaming or code-gen)
- Lock contention (switch to sync.RWMutex or lock-free)`,
					CodeExamples: `// Enable pprof in production (separate port!)
package main

import (
    "log"
    "net/http"
    _ "net/http/pprof" // Side-effect import registers handlers
)

func main() {
    // Debug server on separate port (don't expose to internet!)
    go func() {
        log.Println(http.ListenAndServe("localhost:6060", nil))
    }()

    // Your main application server...
}

// Profile from command line:
// go tool pprof http://localhost:6060/debug/pprof/heap
// go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30
// go tool pprof http://localhost:6060/debug/pprof/goroutine

// Profile benchmarks:
// go test -bench=BenchmarkProcess -cpuprofile=cpu.prof -memprofile=mem.prof
// go tool pprof cpu.prof

// Check for goroutine leaks (should be small and stable):
// curl http://localhost:6060/debug/pprof/goroutine?debug=1 | head -5

// Memory optimization example
// BAD: allocates a new string on every iteration
func concatBad(items []string) string {
    result := ""
    for _, item := range items {
        result += item + ","  // O(n²) — copies entire string each time
    }
    return result
}

// GOOD: uses strings.Builder (single allocation)
func concatGood(items []string) string {
    var b strings.Builder
    b.Grow(len(items) * 10) // Pre-allocate estimated capacity
    for i, item := range items {
        if i > 0 {
            b.WriteByte(',')
        }
        b.WriteString(item)
    }
    return b.String()
}`,
				},
				{
					Title: "Escape Analysis and Stack vs Heap",
					Content: `Go's compiler decides whether each variable lives on the **stack** (fast, automatic) or the **heap** (slower, garbage-collected). Understanding this helps you write code that minimizes GC pressure.

**Stack vs Heap:**
- **Stack**: Fast allocation/deallocation (just move a pointer). Each goroutine has its own stack. Variables are freed when the function returns.
- **Heap**: Managed by the garbage collector. Slower to allocate. Requires GC to reclaim. Shared across goroutines.

**When Does a Variable Escape to the Heap?**
1. **Returned pointer**: ` + "`" + `func f() *int { x := 42; return &x }` + "`" + `
2. **Stored in interface**: ` + "`" + `var i interface{} = x` + "`" + ` (boxing)
3. **Captured by closure**: ` + "`" + `go func() { fmt.Println(x) }()` + "`" + `
4. **Too large for stack**: Very large arrays/structs
5. **Slice/map that grows**: ` + "`" + `append()` + "`" + ` may trigger reallocation

**How to Check Escape Analysis:**
` + "```" + `
go build -gcflags="-m" ./...
# Shows: "moved to heap: x" or "does not escape"
go build -gcflags="-m -m" ./...  # Verbose: shows WHY it escapes
` + "```" + `

**Optimization Strategies:**
1. Return values instead of pointers when struct is small
2. Pre-allocate slices with ` + "`" + `make([]T, 0, expectedSize)` + "`" + `
3. Use ` + "`" + `sync.Pool` + "`" + ` for frequently allocated/freed objects
4. Avoid ` + "`" + `interface{}` + "`" + ` in hot paths (causes boxing → heap allocation)
5. Use ` + "`" + `strings.Builder` + "`" + ` instead of ` + "`" + `+` + "`" + ` for string concatenation

**When NOT to Optimize:**
If your service handles 100 req/s and GC pauses are <1ms, heap allocations don't matter. Profile first!`,
					CodeExamples: `// Example: Escape analysis in action
// Check with: go build -gcflags="-m" escape_example.go

// Does NOT escape (stays on stack)
func sumLocal() int {
    x, y := 10, 20  // Both on stack
    return x + y
}

// ESCAPES to heap (pointer returned)
func newUser() *User {
    u := User{Name: "Alice"}  // Escapes: returned as pointer
    return &u
}

// ESCAPES (stored in interface)
func printAny(v interface{}) { fmt.Println(v) }
func escape1() { 
    x := 42
    printAny(x)  // x escapes: boxed into interface{}
}

// Using sync.Pool to reduce allocations
var bufPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

func processRequest(data []byte) string {
    buf := bufPool.Get().(*bytes.Buffer)
    defer func() {
        buf.Reset()
        bufPool.Put(buf)
    }()
    
    buf.Write(data)
    buf.WriteString(" processed")
    return buf.String()
}

// Benchmark to compare stack vs heap
func BenchmarkStackAlloc(b *testing.B) {
    for i := 0; i < b.N; i++ {
        x := [100]int{}  // Stack allocation
        _ = x
    }
}

func BenchmarkHeapAlloc(b *testing.B) {
    for i := 0; i < b.N; i++ {
        x := make([]int, 100)  // Heap allocation
        _ = x
    }
}`,
				},
				{
					Title: "Benchmarking Best Practices",
					Content: `Go's built-in benchmarking framework (` + "`" + `testing.B` + "`" + `) is excellent. Here's how to write meaningful benchmarks and avoid common pitfalls.

**Basic Benchmark:**
` + "```" + `
func BenchmarkXxx(b *testing.B) {
    for i := 0; i < b.N; i++ {
        // Code to benchmark
    }
}
` + "```" + `

**Running Benchmarks:**
` + "```" + `
go test -bench=.                      # Run all benchmarks
go test -bench=BenchmarkSort          # Run specific benchmark
go test -bench=. -benchmem            # Include memory stats
go test -bench=. -count=5             # Run 5 times for statistical significance
go test -bench=. -benchtime=5s        # Run for 5 seconds (more stable)
go test -bench=. -cpuprofile=cpu.prof # Generate CPU profile
` + "```" + `

**Reading Benchmark Output:**
` + "```" + `
BenchmarkSort-8    1000000    1050 ns/op    256 B/op    4 allocs/op
                   ↑          ↑             ↑           ↑
                   iterations  time/op       bytes/op    allocations/op
` + "```" + `

**Common Pitfalls:**
1. **Compiler optimization**: The compiler may eliminate dead code. Use ` + "`" + `b.StopTimer()` + "`" + `/` + "`" + `b.StartTimer()` + "`" + ` or a package-level sink variable.
2. **Not using b.ResetTimer()**: If setup is expensive, call ` + "`" + `b.ResetTimer()` + "`" + ` after setup.
3. **Not using b.ReportAllocs()**: Call this or use ` + "`" + `-benchmem` + "`" + ` to see memory statistics.
4. **Too short runs**: Use ` + "`" + `-benchtime=3s` + "`" + ` and ` + "`" + `-count=5` + "`" + ` for reliable results.

**Comparing Benchmarks:**
Use ` + "`" + `benchstat` + "`" + ` to compare before/after:
` + "```" + `
go test -bench=. -count=10 > old.txt
# Make your optimization
go test -bench=. -count=10 > new.txt
benchstat old.txt new.txt
` + "```" + ``,
					CodeExamples: `// Preventing dead-code elimination
var result int // Package-level sink

func BenchmarkFib(b *testing.B) {
    var r int
    for i := 0; i < b.N; i++ {
        r = Fib(20)
    }
    result = r // Store to prevent optimization
}

// Benchmark with setup
func BenchmarkSort(b *testing.B) {
    data := make([]int, 10000)
    for i := range data {
        data[i] = rand.Intn(10000)
    }

    b.ResetTimer() // Don't count setup time
    for i := 0; i < b.N; i++ {
        copied := make([]int, len(data))
        copy(copied, data)
        sort.Ints(copied)
    }
}

// Sub-benchmarks for different input sizes
func BenchmarkLookup(b *testing.B) {
    sizes := []int{100, 1000, 10000, 100000}
    for _, size := range sizes {
        b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
            m := make(map[int]bool, size)
            for i := 0; i < size; i++ {
                m[i] = true
            }

            b.ResetTimer()
            b.ReportAllocs()
            for i := 0; i < b.N; i++ {
                _ = m[rand.Intn(size)]
            }
        })
    }
}

// Memory benchmark: pre-allocation vs dynamic growth
func BenchmarkSliceGrow(b *testing.B) {
    b.Run("dynamic", func(b *testing.B) {
        for i := 0; i < b.N; i++ {
            s := []int{}
            for j := 0; j < 10000; j++ {
                s = append(s, j)
            }
        }
    })

    b.Run("preallocated", func(b *testing.B) {
        for i := 0; i < b.N; i++ {
            s := make([]int, 0, 10000)
            for j := 0; j < 10000; j++ {
                s = append(s, j)
            }
        }
    })
}`,
				},
			},
		},
	})
}
