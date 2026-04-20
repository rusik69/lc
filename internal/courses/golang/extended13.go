package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          1632,
			Title:       "Go Idioms and Best Practices",
			Description: "Master idiomatic Go: error handling patterns, package design, API design, and code quality guidelines.",
			Order:       32,
			Lessons: []problems.Lesson{
				{
					Title: "Error Handling Patterns",
					Content: `Go's error handling philosophy emphasizes explicit, local handling. While often criticized for verbosity, well-structured error handling produces maintainable, debuggable code.

**Error Wrapping and Unwrapping:**
` + "```" + `
fmt.Errorf with %w:
  func readConfig(path string) (*Config, error) {
      data, err := os.ReadFile(path)
      if err != nil {
          return nil, fmt.Errorf("readConfig %s: %w", path, err)
      }
      // ...
  }
  
  // Caller can inspect:
  _, err := readConfig("/etc/app.conf")
  if errors.Is(err, os.ErrNotExist) {
      // File doesn't exist — use defaults
  }

errors.Is vs errors.As:
  errors.Is(err, target)  → checks if err (or wrapped) matches target value
  errors.As(err, &target) → checks if err (or wrapped) matches target type
  
  // errors.Is — value check
  if errors.Is(err, sql.ErrNoRows) { ... }
  if errors.Is(err, context.Canceled) { ... }
  if errors.Is(err, io.EOF) { ... }
  
  // errors.As — type check + extraction
  var pathErr *os.PathError
  if errors.As(err, &pathErr) {
      fmt.Println("failed path:", pathErr.Path)
  }
  
  var validErr *ValidationError 
  if errors.As(err, &validErr) {
      fmt.Println("field:", validErr.Field)
  }

Custom error types:
  type NotFoundError struct {
      Resource string
      ID       string
  }
  
  func (e *NotFoundError) Error() string {
      return fmt.Sprintf("%s %s not found", e.Resource, e.ID)
  }
  
  func (e *NotFoundError) Is(target error) bool {
      _, ok := target.(*NotFoundError)
      return ok
  }

Sentinel errors:
  var (
      ErrNotFound     = errors.New("not found")
      ErrUnauthorized = errors.New("unauthorized")
      ErrConflict     = errors.New("conflict")
  )
  
  // Use in packages:
  func GetUser(id string) (*User, error) {
      user, ok := users[id]
      if !ok {
          return nil, fmt.Errorf("user %s: %w", id, ErrNotFound)
      }
      return user, nil
  }
` + "```" + `

**Error Handling Patterns:**
` + "```" + `
Pattern 1: Early return
  func process(data []byte) error {
      if len(data) == 0 {
          return errors.New("empty data")
      }
      parsed, err := parse(data)
      if err != nil {
          return fmt.Errorf("parse: %w", err)
      }
      validated, err := validate(parsed)
      if err != nil {
          return fmt.Errorf("validate: %w", err)
      }
      return store(validated)
  }

Pattern 2: Error variable for repeated operations
  var errs []error
  for _, item := range items {
      if err := process(item); err != nil {
          errs = append(errs, fmt.Errorf("item %s: %w", item.ID, err))
      }
  }
  return errors.Join(errs...)

Pattern 3: Cleanup with defer
  func writeFile(path string, data []byte) (retErr error) {
      f, err := os.Create(path)
      if err != nil {
          return err
      }
      defer func() {
          closeErr := f.Close()
          if retErr == nil {
              retErr = closeErr
          }
      }()
      _, err = f.Write(data)
      return err
  }

Pattern 4: Must functions (for init-time only)
  func must[T any](val T, err error) T {
      if err != nil {
          panic(err)
      }
      return val
  }
  
  var tmpl = must(template.ParseFiles("base.html"))
  var re = must(regexp.Compile("[a-z]+"))
  // ONLY use in package-level init, never in runtime code

Pattern 5: Error groups for concurrent operations
  g, ctx := errgroup.WithContext(ctx)
  for _, url := range urls {
      url := url
      g.Go(func() error {
          return fetch(ctx, url)
      })
  }
  if err := g.Wait(); err != nil {
      return err
  }

Anti-patterns to avoid:
  ✗ if err != nil { return err }     // No context added!
  ✓ if err != nil { return fmt.Errorf("doing X: %w", err) }
  
  ✗ panic(err) in library code
  ✗ _ = riskyFunction()              // Ignoring errors
  ✗ log.Fatal(err) deep in call stack // Hard to test
` + "```" + ``,
					CodeExamples: `// Error handling patterns
package main

import (
    "errors"
    "fmt"
    "strings"
)

// Domain errors
var (
    ErrNotFound     = errors.New("not found")
    ErrUnauthorized = errors.New("unauthorized")  
    ErrValidation   = errors.New("validation error")
    ErrConflict     = errors.New("conflict")
)

// Typed error with context
type FieldError struct {
    Field   string
    Message string
}

func (e *FieldError) Error() string {
    return fmt.Sprintf("field %s: %s", e.Field, e.Message)
}

func (e *FieldError) Unwrap() error {
    return ErrValidation
}

// Multi-error for validation
type ValidationErrors struct {
    Errors []FieldError
}

func (e *ValidationErrors) Error() string {
    msgs := make([]string, len(e.Errors))
    for i, fe := range e.Errors {
        msgs[i] = fe.Error()
    }
    return "validation failed: " + strings.Join(msgs, "; ")
}

func (e *ValidationErrors) Unwrap() error {
    return ErrValidation
}

// Domain types
type User struct {
    ID    string
    Name  string
    Email string
    Age   int
}

// In-memory store
var users = map[string]*User{
    "1": {ID: "1", Name: "Alice", Email: "alice@example.com", Age: 30},
    "2": {ID: "2", Name: "Bob", Email: "bob@example.com", Age: 25},
}

// Validation
func validateUser(u *User) error {
    var errs []FieldError
    
    if u.Name == "" {
        errs = append(errs, FieldError{Field: "name", Message: "required"})
    }
    if u.Email == "" {
        errs = append(errs, FieldError{Field: "email", Message: "required"})
    } else if !strings.Contains(u.Email, "@") {
        errs = append(errs, FieldError{Field: "email", Message: "invalid format"})
    }
    if u.Age < 0 || u.Age > 150 {
        errs = append(errs, FieldError{Field: "age", Message: "must be 0-150"})
    }
    
    if len(errs) > 0 {
        return &ValidationErrors{Errors: errs}
    }
    return nil
}

// Repository
func getUser(id string) (*User, error) {
    user, ok := users[id]
    if !ok {
        return nil, fmt.Errorf("user %s: %w", id, ErrNotFound)
    }
    return user, nil
}

func createUser(u *User) error {
    if err := validateUser(u); err != nil {
        return fmt.Errorf("createUser: %w", err)
    }
    if _, exists := users[u.ID]; exists {
        return fmt.Errorf("user %s: %w", u.ID, ErrConflict)
    }
    users[u.ID] = u
    return nil
}

func updateUser(id string, update func(*User)) error {
    user, err := getUser(id)
    if err != nil {
        return fmt.Errorf("updateUser: %w", err)
    }
    update(user)
    if err := validateUser(user); err != nil {
        return fmt.Errorf("updateUser validate: %w", err)
    }
    return nil
}

// Service layer with error handling
func processUserRegistration(name, email string, age int) error {
    user := &User{
        ID:    fmt.Sprintf("%d", len(users)+1),
        Name:  name,
        Email: email,
        Age:   age,
    }
    
    if err := createUser(user); err != nil {
        return fmt.Errorf("registration: %w", err)
    }
    
    return nil
}

// Batch processing with error collection
func processUserBatch(updates []struct{ id, email string }) error {
    var errs []error
    for _, u := range updates {
        err := updateUser(u.id, func(user *User) {
            user.Email = u.email
        })
        if err != nil {
            errs = append(errs, fmt.Errorf("update %s: %w", u.id, err))
        }
    }
    return errors.Join(errs...)
}

func main() {
    fmt.Println("=== Error Handling Patterns ===")
    
    // 1. Basic error wrapping and checking
    fmt.Println("\n--- Error wrapping ---")
    _, err := getUser("999")
    fmt.Printf("  Error: %v\n", err)
    fmt.Printf("  Is NotFound: %v\n", errors.Is(err, ErrNotFound))
    fmt.Printf("  Is Unauthorized: %v\n", errors.Is(err, ErrUnauthorized))
    
    // 2. Typed error extraction
    fmt.Println("\n--- Validation errors ---")
    err = createUser(&User{ID: "3", Name: "", Email: "invalid", Age: -5})
    fmt.Printf("  Error: %v\n", err)
    fmt.Printf("  Is Validation: %v\n", errors.Is(err, ErrValidation))
    
    var valErrs *ValidationErrors
    if errors.As(err, &valErrs) {
        fmt.Println("  Fields with errors:")
        for _, fe := range valErrs.Errors {
            fmt.Printf("    - %s: %s\n", fe.Field, fe.Message)
        }
    }
    
    // 3. Conflict error
    fmt.Println("\n--- Conflict error ---")
    err = createUser(&User{ID: "1", Name: "Duplicate", Email: "dup@example.com", Age: 25})
    fmt.Printf("  Error: %v\n", err)
    fmt.Printf("  Is Conflict: %v\n", errors.Is(err, ErrConflict))
    
    // 4. Successful operation
    fmt.Println("\n--- Successful registration ---")
    err = processUserRegistration("Charlie", "charlie@example.com", 35)
    fmt.Printf("  Error: %v\n", err)
    
    // 5. Batch processing  
    fmt.Println("\n--- Batch processing ---")
    err = processUserBatch([]struct{ id, email string }{
        {"1", "alice_new@example.com"},
        {"999", "nope@example.com"},
        {"2", "bob_new@example.com"},
    })
    if err != nil {
        fmt.Printf("  Batch errors:\n")
        for _, line := range strings.Split(err.Error(), "\n") {
            fmt.Printf("    %s\n", line)
        }
    }
    
    // 6. Error chain inspection
    fmt.Println("\n--- Error chain ---")
    err = updateUser("999", func(u *User) { u.Name = "X" })
    fmt.Printf("  Full error: %v\n", err)
    
    // Unwrap chain
    current := err
    depth := 0
    for current != nil {
        fmt.Printf("  [depth %d] %v\n", depth, current)
        current = errors.Unwrap(current)
        depth++
    }

    // 7. errors.Join (Go 1.20+)
    fmt.Println("\n--- errors.Join ---")
    combined := errors.Join(
        fmt.Errorf("error 1"),
        fmt.Errorf("error 2"),
        fmt.Errorf("error 3: %w", ErrNotFound),
    )
    fmt.Printf("  Combined: %v\n", combined)
    fmt.Printf("  Contains NotFound: %v\n", errors.Is(combined, ErrNotFound))
}`,
				},
				{
					Title: "Package Design and API Guidelines",
					Content: `Good Go code starts with good package design. Packages are Go's primary unit of abstraction, encapsulation, and reuse.

**Package Naming:**
` + "```" + `
Naming rules:
  ✓ Short, lowercase, single word: http, fmt, json, os
  ✓ Noun-like names for things: bytes, strings, errors
  ✓ No underscores, no mixedCaps: net/http not net/HTTP
  ✓ Avoid "util", "common", "base", "misc" — too vague
  
  // Package name is part of the identifier:
  http.Client    not httpClient
  bytes.Buffer   not bytesBuffer
  json.Decoder   not jsonDecoder
  
  Avoid stutter:
  ✗ http.HTTPClient → ✓ http.Client
  ✗ json.JSONEncoder → ✓ json.Encoder
  ✗ user.UserService → ✓ user.Service

Package size guidelines:
  - Small, focused packages > large, monolithic ones
  - One clear purpose per package
  - If you can't name it clearly, it might be too broad
  - Don't split prematurely — start with one package
  
  Good structure:
    myapp/
      cmd/myapp/main.go     // Entry point
      internal/              // Private packages
        user/                // User domain
          user.go            // Types + core logic
          store.go           // Storage interface + impl
          service.go         // Business logic
        order/               // Order domain
      pkg/                   // Public API packages (optional)
` + "```" + `

**Interface Design:**
` + "```" + `
Accept interfaces, return structs:
  // Good — accepts interface
  func ProcessData(r io.Reader) error { ... }
  
  // Good — returns concrete type
  func NewServer(addr string) *Server { ... }
  
  // Bad — returns interface (hides implementation)
  func NewServer(addr string) ServerInterface { ... }

Small interfaces:
  // Go style: 1-3 methods
  type Reader interface { Read(p []byte) (n int, err error) }
  type Writer interface { Write(p []byte) (n int, err error) }
  type Closer interface { Close() error }
  
  // Compose when needed:
  type ReadWriteCloser interface {
      Reader
      Writer
      Closer
  }

Interface naming:
  Single method → method name + "er":
    Read → Reader
    Write → Writer
    Close → Closer
    Format → Formatter
    Handle → Handler
  
  Multiple methods → descriptive noun:
    type Store interface { Get, Put, Delete }
    type Authenticator interface { Authenticate, Authorize }

Define interfaces where they're USED, not where they're implemented:
  // In package "handler" (consumer):
  type UserStore interface {
      GetUser(ctx context.Context, id string) (*User, error)
  }
  
  // In package "postgres" (implementor):
  type Store struct { db *sql.DB }
  func (s *Store) GetUser(ctx context.Context, id string) (*User, error) { ... }
  
  // The postgres package doesn't import or know about handler's interface
  // Implicit interface satisfaction — Go's structural typing
` + "```" + `

**Functional Options Pattern:**
` + "```" + `
// For configurable constructors:
type Option func(*Server)

func WithPort(port int) Option {
    return func(s *Server) { s.port = port }
}

func WithTimeout(d time.Duration) Option {
    return func(s *Server) { s.timeout = d }
}

func WithLogger(l *slog.Logger) Option {
    return func(s *Server) { s.logger = l }
}

func NewServer(opts ...Option) *Server {
    s := &Server{
        port:    8080,           // Defaults
        timeout: 30 * time.Second,
        logger:  slog.Default(),
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// Usage:
srv := NewServer(
    WithPort(9090),
    WithTimeout(time.Minute),
)
` + "```" + `

**Context Usage:**
` + "```" + `
context.Context rules:
  ✓ First parameter named ctx: func DoSomething(ctx context.Context, ...)
  ✓ Pass through entire call chain
  ✓ Don't store in structs (except when wrapping for lifecycle)
  ✓ Use for cancellation, deadlines, request-scoped values
  
  // Cancellation
  ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
  defer cancel()
  
  result, err := longOperation(ctx)
  if err != nil {
      if errors.Is(err, context.DeadlineExceeded) {
          // Timed out
      }
      if errors.Is(err, context.Canceled) {
          // Caller canceled
      }
  }
  
  // Values (use sparingly — only for request-scoped data)
  type contextKey string
  const userIDKey contextKey = "userID"
  
  ctx = context.WithValue(ctx, userIDKey, "user-123")
  userID, ok := ctx.Value(userIDKey).(string)
  
  // DON'T use context.Value for:
  ✗ Optional function parameters
  ✗ Passing dependencies (use DI instead)
  ✗ Anything that could be a function argument
` + "```" + `

**Code Organization Guidelines:**
` + "```" + `
File organization:
  1. Package clause
  2. Imports (stdlib, blank line, external)
  3. Constants
  4. Variables
  5. Types (most important first)
  6. Constructor functions (New*)
  7. Methods
  8. Helper functions (unexported)
  
  File naming:
    user.go        // Main types and logic
    user_test.go   // Tests
    store.go       // Storage layer
    http.go        // HTTP handlers
    mock_test.go   // Test mocks

Documentation:
  // Package user provides user management functionality.
  package user
  
  // User represents a registered user in the system.
  // Users are identified by a unique ID.
  type User struct { ... }
  
  // New creates a User with the given name and email.
  // It returns an error if name or email are empty.
  func New(name, email string) (*User, error) { ... }
  
  // Unexported functions don't need doc comments
  // but complex ones benefit from them

Testing conventions:
  func TestUserCreation(t *testing.T) { ... }
  func TestUserCreation_EmptyName(t *testing.T) { ... }
  func TestUserCreation_InvalidEmail(t *testing.T) { ... }
  
  // Table-driven:
  func TestValidateEmail(t *testing.T) {
      tests := []struct{
          name  string
          email string
          valid bool
      }{ ... }
  }
  
  // Testable examples:
  func ExampleNew() {
      u, _ := user.New("Alice", "alice@example.com")
      fmt.Println(u.Name)
      // Output: Alice
  }
` + "```" + ``,
					CodeExamples: `// Package design and API patterns
package main

import (
    "context"
    "errors"
    "fmt"
    "strings"
    "time"
)

// ============================================================
// Domain types
// ============================================================

type UserID string
type Email string

type User struct {
    ID        UserID
    Name      string
    Email     Email
    CreatedAt time.Time
    UpdatedAt time.Time
}

// ============================================================
// Interface design — small, focused interfaces
// ============================================================

// UserReader — read-only operations (query side)
type UserReader interface {
    GetUser(ctx context.Context, id UserID) (*User, error)
    ListUsers(ctx context.Context, limit, offset int) ([]*User, error)
}

// UserWriter — write operations (command side)
type UserWriter interface {
    CreateUser(ctx context.Context, u *User) error
    UpdateUser(ctx context.Context, u *User) error
    DeleteUser(ctx context.Context, id UserID) error
}

// UserStore — composed interface
type UserStore interface {
    UserReader
    UserWriter
}

// ============================================================
// In-memory implementation
// ============================================================

type InMemoryUserStore struct {
    users map[UserID]*User
}

func NewInMemoryUserStore() *InMemoryUserStore {
    return &InMemoryUserStore{users: make(map[UserID]*User)}
}

func (s *InMemoryUserStore) GetUser(_ context.Context, id UserID) (*User, error) {
    u, ok := s.users[id]
    if !ok {
        return nil, fmt.Errorf("user %s: %w", id, ErrNotFound)
    }
    return u, nil
}

func (s *InMemoryUserStore) ListUsers(_ context.Context, limit, offset int) ([]*User, error) {
    all := make([]*User, 0, len(s.users))
    for _, u := range s.users {
        all = append(all, u)
    }
    if offset >= len(all) {
        return nil, nil
    }
    end := offset + limit
    if end > len(all) {
        end = len(all)
    }
    return all[offset:end], nil
}

func (s *InMemoryUserStore) CreateUser(_ context.Context, u *User) error {
    if _, exists := s.users[u.ID]; exists {
        return fmt.Errorf("user %s: %w", u.ID, ErrConflict)
    }
    u.CreatedAt = time.Now()
    u.UpdatedAt = time.Now()
    s.users[u.ID] = u
    return nil
}

func (s *InMemoryUserStore) UpdateUser(_ context.Context, u *User) error {
    if _, exists := s.users[u.ID]; !exists {
        return fmt.Errorf("user %s: %w", u.ID, ErrNotFound)
    }
    u.UpdatedAt = time.Now()
    s.users[u.ID] = u
    return nil
}

func (s *InMemoryUserStore) DeleteUser(_ context.Context, id UserID) error {
    if _, exists := s.users[id]; !exists {
        return fmt.Errorf("user %s: %w", id, ErrNotFound)
    }
    delete(s.users, id)
    return nil
}

// ============================================================
// Sentinel errors
// ============================================================

var (
    ErrNotFound = errors.New("not found")
    ErrConflict = errors.New("conflict")
)

// ============================================================
// Functional options for service configuration
// ============================================================

type Logger interface {
    Info(msg string, args ...any)
}

type simpleLogger struct{}

func (l *simpleLogger) Info(msg string, args ...any) {
    fmt.Printf("[INFO] %s", msg)
    for i := 0; i < len(args)-1; i += 2 {
        fmt.Printf(" %v=%v", args[i], args[i+1])
    }
    fmt.Println()
}

type UserServiceOption func(*UserService)

func WithStore(store UserStore) UserServiceOption {
    return func(s *UserService) { s.store = store }
}

func WithServiceLogger(l Logger) UserServiceOption {
    return func(s *UserService) { s.logger = l }
}

func WithMaxUsers(n int) UserServiceOption {
    return func(s *UserService) { s.maxUsers = n }
}

// ============================================================
// Service with functional options
// ============================================================

type UserService struct {
    store    UserStore
    logger   Logger
    maxUsers int
}

func NewUserService(opts ...UserServiceOption) *UserService {
    s := &UserService{
        store:    NewInMemoryUserStore(),
        logger:   &simpleLogger{},
        maxUsers: 1000,
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

func (s *UserService) Register(ctx context.Context, id, name, email string) error {
    s.logger.Info("registering user", "id", id, "name", name)
    
    // Validate
    if name == "" {
        return fmt.Errorf("name required")
    }
    if !strings.Contains(email, "@") {
        return fmt.Errorf("invalid email: %s", email)
    }
    
    user := &User{
        ID:    UserID(id),
        Name:  name,
        Email: Email(email),
    }
    
    if err := s.store.CreateUser(ctx, user); err != nil {
        return fmt.Errorf("register %s: %w", id, err)
    }
    
    s.logger.Info("user registered", "id", id)
    return nil
}

func (s *UserService) Get(ctx context.Context, id string) (*User, error) {
    return s.store.GetUser(ctx, UserID(id))
}

func (s *UserService) List(ctx context.Context) ([]*User, error) {
    return s.store.ListUsers(ctx, s.maxUsers, 0)
}

// ============================================================
// Context usage patterns
// ============================================================

type contextKey string

const requestIDKey contextKey = "requestID"

func withRequestID(ctx context.Context, id string) context.Context {
    return context.WithValue(ctx, requestIDKey, id)
}

func requestIDFrom(ctx context.Context) string {
    if id, ok := ctx.Value(requestIDKey).(string); ok {
        return id
    }
    return "unknown"
}

func simulateSlowOperation(ctx context.Context, name string, duration time.Duration) error {
    fmt.Printf("  Starting %s (timeout simulation)...\n", name)
    
    select {
    case <-time.After(duration):
        fmt.Printf("  %s completed\n", name)
        return nil
    case <-ctx.Done():
        fmt.Printf("  %s canceled: %v\n", name, ctx.Err())
        return ctx.Err()
    }
}

func main() {
    ctx := context.Background()
    
    // ============================================
    // Functional Options
    // ============================================
    fmt.Println("=== Functional Options ===")
    
    svc := NewUserService(
        WithMaxUsers(100),
    )
    
    // Register users
    _ = svc.Register(ctx, "1", "Alice", "alice@example.com")
    _ = svc.Register(ctx, "2", "Bob", "bob@example.com")
    _ = svc.Register(ctx, "3", "Charlie", "charlie@example.com")
    
    // Get user
    user, err := svc.Get(ctx, "1")
    if err != nil {
        fmt.Printf("  Error: %v\n", err)
    } else {
        fmt.Printf("  Got user: %s <%s>\n", user.Name, user.Email)
    }
    
    // List users
    users, _ := svc.List(ctx)
    fmt.Printf("  Total users: %d\n", len(users))
    
    // Duplicate registration
    err = svc.Register(ctx, "1", "Alice2", "alice2@example.com")
    fmt.Printf("  Duplicate: %v\n", err)
    fmt.Printf("  Is conflict: %v\n", errors.Is(err, ErrConflict))
    
    // Not found
    _, err = svc.Get(ctx, "999")
    fmt.Printf("  Not found: %v\n", err)
    fmt.Printf("  Is not found: %v\n", errors.Is(err, ErrNotFound))
    
    // ============================================
    // Context patterns
    // ============================================
    fmt.Println("\n=== Context Patterns ===")
    
    // Request-scoped values
    reqCtx := withRequestID(ctx, "req-abc-123")
    fmt.Printf("  Request ID: %s\n", requestIDFrom(reqCtx))
    fmt.Printf("  Missing ID: %s\n", requestIDFrom(ctx))
    
    // Context with timeout
    fmt.Println("\n--- Timeout ---")
    timeoutCtx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
    defer cancel()
    
    err = simulateSlowOperation(timeoutCtx, "fast-op", 50*time.Millisecond)
    fmt.Printf("  Fast result: %v\n", err)
    
    err = simulateSlowOperation(timeoutCtx, "slow-op", 200*time.Millisecond)
    fmt.Printf("  Slow result: %v\n", err)
    fmt.Printf("  Is deadline exceeded: %v\n", errors.Is(err, context.DeadlineExceeded))
    
    // Context with cancel
    fmt.Println("\n--- Cancel ---")
    cancelCtx, cancelFn := context.WithCancel(ctx)
    
    go func() {
        time.Sleep(50 * time.Millisecond)
        cancelFn()
    }()
    
    err = simulateSlowOperation(cancelCtx, "cancelable-op", time.Second)
    fmt.Printf("  Cancel result: %v\n", err)
    fmt.Printf("  Is canceled: %v\n", errors.Is(err, context.Canceled))
    
    // ============================================
    // Interface satisfaction check (compile-time)
    // ============================================
    fmt.Println("\n=== Interface Compliance ===")
    var _ UserStore = (*InMemoryUserStore)(nil)
    var _ UserReader = (*InMemoryUserStore)(nil)
    var _ UserWriter = (*InMemoryUserStore)(nil)
    fmt.Println("  InMemoryUserStore satisfies UserStore, UserReader, UserWriter")
    
    // ============================================
    // Summary
    // ============================================
    fmt.Println("\n=== Go Idioms Summary ===")
    idioms := []string{
        "Accept interfaces, return structs",
        "Define interfaces at the consumer",
        "Keep interfaces small (1-3 methods)",
        "Use functional options for constructors",
        "Error wrapping with fmt.Errorf and %w",
        "Context as first parameter",
        "Early return for error handling",
        "Don't stutter (http.Client not http.HTTPClient)",
    }
    for i, idiom := range idioms {
        fmt.Printf("  %d. %s\n", i+1, idiom)
    }
}`,
				},
			},
		},
	})
}
