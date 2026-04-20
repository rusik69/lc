package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          1629,
			Title:       "Advanced Testing and Quality",
			Description: "Master advanced testing techniques: table-driven tests, fuzzing, property-based testing, mocking, integration testing, and test architecture.",
			Order:       29,
			Lessons: []problems.Lesson{
				{
					Title: "Table-Driven Tests and Subtests",
					Content: `Table-driven tests are the standard Go testing pattern. They make tests readable, maintainable, and easy to extend.

**Table-Driven Test Pattern:**
` + "```" + `
func TestAdd(t *testing.T) {
    tests := []struct {
        name     string
        a, b     int
        expected int
    }{
        {"positive", 2, 3, 5},
        {"negative", -1, -2, -3},
        {"zero", 0, 0, 0},
        {"mixed", -1, 5, 4},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := Add(tt.a, tt.b)
            if got != tt.expected {
                t.Errorf("Add(%d, %d) = %d, want %d",
                    tt.a, tt.b, got, tt.expected)
            }
        })
    }
}

Why table-driven:
  - Adding test cases = adding rows (data, not code)
  - Each case has a name (easy to identify failures)
  - Subtests can be run individually: go test -run TestAdd/positive
  - DRY: test logic written once

Naming convention for test cases:
  - Descriptive: "empty input", "invalid email"
  - Use slashes for categories: "valid/admin", "valid/user"
  - Keep short but meaningful

Parallel table tests:
  for _, tt := range tests {
      tt := tt // Capture range variable
      t.Run(tt.name, func(t *testing.T) {
          t.Parallel()
          // test code
      })
  }
  
  Go 1.22: range variable fix (no need to capture)
  for _, tt := range tests {
      t.Run(tt.name, func(t *testing.T) {
          t.Parallel()
          // tt is safe in Go 1.22+
      })
  }
` + "```" + `

**Subtests and Test Organization:**
` + "```" + `
Hierarchical tests:
  func TestUserService(t *testing.T) {
      t.Run("Create", func(t *testing.T) {
          t.Run("valid user", func(t *testing.T) { ... })
          t.Run("duplicate email", func(t *testing.T) { ... })
          t.Run("invalid input", func(t *testing.T) { ... })
      })
      t.Run("Delete", func(t *testing.T) {
          t.Run("existing user", func(t *testing.T) { ... })
          t.Run("non-existing user", func(t *testing.T) { ... })
      })
  }
  
  Run specific: go test -run TestUserService/Create/valid_user

Test helpers:
  func TestMain(m *testing.M) {
      // Setup (runs once before all tests)
      setup()
      
      code := m.Run()
      
      // Teardown (runs once after all tests)
      teardown()
      
      os.Exit(code)
  }

Helper functions:
  func setupTestDB(t *testing.T) *sql.DB {
      t.Helper() // ← Marks as helper (error location shows caller)
      db, err := sql.Open(...)
      if err != nil {
          t.Fatal(err) // Fatal: stop this test immediately
      }
      t.Cleanup(func() {
          db.Close() // Cleanup runs at end of test
      })
      return db
  }

T methods:
  t.Error(args...)   → Log failure, continue running
  t.Errorf(...)      → Formatted failure, continue
  t.Fatal(args...)   → Log failure, STOP test
  t.Fatalf(...)      → Formatted failure, STOP test
  t.Skip(reason)     → Skip test (e.g., "requires Docker")
  t.Parallel()       → Run in parallel with other parallel tests
  t.Helper()         → Mark as helper (adjust error reporting)
  t.Cleanup(func())  → Register cleanup function
  t.TempDir()        → Create temporary directory (auto-cleaned)
` + "```" + ``,
					CodeExamples: `// Table-driven testing patterns
package main

import (
    "errors"
    "fmt"
    "strings"
    "unicode"
)

// Functions to test
func ValidateEmail(email string) error {
    if email == "" {
        return errors.New("email is required")
    }
    if !strings.Contains(email, "@") {
        return errors.New("email must contain @")
    }
    parts := strings.Split(email, "@")
    if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
        return errors.New("invalid email format")
    }
    if !strings.Contains(parts[1], ".") {
        return errors.New("domain must contain .")
    }
    return nil
}

func ValidatePassword(password string) []string {
    var errs []string
    
    if len(password) < 8 {
        errs = append(errs, "must be at least 8 characters")
    }
    
    var hasUpper, hasLower, hasDigit bool
    for _, r := range password {
        switch {
        case unicode.IsUpper(r):
            hasUpper = true
        case unicode.IsLower(r):
            hasLower = true
        case unicode.IsDigit(r):
            hasDigit = true
        }
    }
    
    if !hasUpper {
        errs = append(errs, "must contain uppercase letter")
    }
    if !hasLower {
        errs = append(errs, "must contain lowercase letter")
    }
    if !hasDigit {
        errs = append(errs, "must contain digit")
    }
    
    return errs
}

func Slugify(s string) string {
    var result strings.Builder
    for _, r := range strings.ToLower(s) {
        if unicode.IsLetter(r) || unicode.IsDigit(r) {
            result.WriteRune(r)
        } else if r == ' ' || r == '-' || r == '_' {
            result.WriteByte('-')
        }
    }
    // Remove consecutive dashes
    slug := result.String()
    for strings.Contains(slug, "--") {
        slug = strings.ReplaceAll(slug, "--", "-")
    }
    return strings.Trim(slug, "-")
}

// Test simulation (since this is not a _test.go file)
type TestResult struct {
    Name    string
    Passed  bool
    Message string
}

func runTableTest(name string, run func() (bool, string)) TestResult {
    passed, msg := run()
    return TestResult{Name: name, Passed: passed, Message: msg}
}

func main() {
    fmt.Println("=== Table-Driven Tests: ValidateEmail ===")
    
    emailTests := []struct {
        name    string
        email   string
        wantErr bool
    }{
        {"valid email", "alice@example.com", false},
        {"empty email", "", true},
        {"no @", "alice.example.com", true},
        {"no domain", "alice@", true},
        {"no user", "@example.com", true},
        {"no dot in domain", "alice@example", true},
        {"valid with subdomain", "alice@mail.example.com", false},
    }
    
    for _, tt := range emailTests {
        result := runTableTest(tt.name, func() (bool, string) {
            err := ValidateEmail(tt.email)
            if (err != nil) != tt.wantErr {
                return false, fmt.Sprintf("ValidateEmail(%q) error=%v, wantErr=%v", tt.email, err, tt.wantErr)
            }
            return true, ""
        })
        status := "PASS"
        if !result.Passed {
            status = "FAIL"
        }
        fmt.Printf("  [%s] %s", status, result.Name)
        if result.Message != "" {
            fmt.Printf(": %s", result.Message)
        }
        fmt.Println()
    }
    
    fmt.Println("\n=== Table-Driven Tests: ValidatePassword ===")
    
    passwordTests := []struct {
        name      string
        password  string
        wantErrs  int
    }{
        {"strong password", "SecureP@ss1", 0},
        {"too short", "Ab1", 1},
        {"no uppercase", "password1", 1},
        {"no lowercase", "PASSWORD1", 1},
        {"no digit", "Password", 1},
        {"empty", "", 4},
    }
    
    for _, tt := range passwordTests {
        result := runTableTest(tt.name, func() (bool, string) {
            errs := ValidatePassword(tt.password)
            if len(errs) != tt.wantErrs {
                return false, fmt.Sprintf("got %d errors, want %d: %v", len(errs), tt.wantErrs, errs)
            }
            return true, ""
        })
        status := "PASS"
        if !result.Passed {
            status = "FAIL"
        }
        fmt.Printf("  [%s] %s", status, result.Name)
        if result.Message != "" {
            fmt.Printf(": %s", result.Message)
        }
        fmt.Println()
    }
    
    fmt.Println("\n=== Table-Driven Tests: Slugify ===")
    
    slugTests := []struct {
        name  string
        input string
        want  string
    }{
        {"simple", "Hello World", "hello-world"},
        {"special chars", "Hello, World!", "hello-world"},
        {"multiple spaces", "Hello   World", "hello-world"},
        {"dashes and underscores", "Hello-World_Foo", "hello-world-foo"},
        {"numbers", "Go 1.22 Release", "go-122-release"},
        {"empty", "", ""},
        {"only special", "!@#$%", ""},
    }
    
    for _, tt := range slugTests {
        result := runTableTest(tt.name, func() (bool, string) {
            got := Slugify(tt.input)
            if got != tt.want {
                return false, fmt.Sprintf("Slugify(%q) = %q, want %q", tt.input, got, tt.want)
            }
            return true, ""
        })
        status := "PASS"
        if !result.Passed {
            status = "FAIL"
        }
        fmt.Printf("  [%s] %s", status, result.Name)
        if result.Message != "" {
            fmt.Printf(": %s", result.Message)
        }
        fmt.Println()
    }
}`,
				},
				{
					Title: "Fuzzing and Property-Based Testing",
					Content: `Fuzzing discovers edge cases that manual tests miss. Go 1.18+ has built-in fuzzing support in the testing package.

**Go Fuzzing:**
` + "```" + `
func FuzzParseJSON(f *testing.F) {
    // Seed corpus (initial inputs)
    f.Add([]byte(` + "`" + `{"name":"Alice"}` + "`" + `))
    f.Add([]byte(` + "`" + `{"age":30}` + "`" + `))
    f.Add([]byte(` + "`" + `[]` + "`" + `))
    
    // Fuzz function (run with random mutations of seeds)
    f.Fuzz(func(t *testing.T, data []byte) {
        var v any
        err := json.Unmarshal(data, &v)
        if err != nil {
            return // Invalid JSON is fine, just don't panic
        }
        
        // Re-marshal and verify round-trip
        encoded, err := json.Marshal(v)
        if err != nil {
            t.Errorf("Marshal failed after Unmarshal: %v", err)
        }
        
        var v2 any
        if err := json.Unmarshal(encoded, &v2); err != nil {
            t.Errorf("Unmarshal failed on re-encoded: %v", err)
        }
    })
}

Run:
  go test -fuzz=FuzzParseJSON          → Fuzz until stopped (Ctrl+C)
  go test -fuzz=FuzzParseJSON -fuzztime=30s  → Fuzz for 30 seconds
  go test -fuzz=FuzzParseJSON -fuzztime=1000x → Run 1000 iterations

Corpus:
  Seeds go in testdata/fuzz/<FuncName>/ directory
  Crash inputs automatically saved there
  Committed to VCS for regression testing

What fuzzing finds:
  - Panics (nil pointer, index out of range)
  - Infinite loops
  - Buffer overflows (in CGO code)
  - Logic errors with unusual inputs
  - Unicode handling bugs
  - Goroutine leaks (with timeout)
  
Supported types for fuzz input:
  []byte, string, int, uint, float32, float64, bool,
  int8..int64, uint8..uint64, rune
  
  For complex types: fuzz []byte, deserialize in test
` + "```" + `

**Property-Based Testing:**
` + "```" + `
Instead of testing specific inputs, test PROPERTIES that must hold:

Properties of a sort function:
  1. Length preserved: len(sort(x)) == len(x)
  2. Ordered: sort(x)[i] <= sort(x)[i+1] for all i
  3. Same elements: sort(x) contains same elements as x
  4. Idempotent: sort(sort(x)) == sort(x)
  5. Stability: equal elements maintain relative order

Properties of an encoder/decoder:
  1. Round-trip: decode(encode(x)) == x
  2. Deterministic: encode(x) == encode(x)
  
Properties of a cache:
  1. Get after Set returns value: set(k,v) → get(k) == v
  2. Get before Set returns miss
  3. Delete removes entry
  4. Size limit respected

Testing properties with rapid:
  import "pgregory.net/rapid"
  
  func TestSortProperties(t *testing.T) {
      rapid.Check(t, func(t *rapid.T) {
          s := rapid.SliceOf(rapid.Int()).Draw(t, "input")
          sorted := SortInts(s)
          
          // Property 1: same length
          if len(sorted) != len(s) {
              t.Fatal("length changed")
          }
          
          // Property 2: ordered
          for i := 1; i < len(sorted); i++ {
              if sorted[i-1] > sorted[i] {
                  t.Fatalf("not sorted at index %d", i)
              }
          }
      })
  }

Stateful testing (test against a model):
  Test your implementation against a simple reference:
  
  real := NewCache(100)
  model := map[string]string{} // Simple reference implementation
  
  for each random operation:
      key := random_key()
      value := random_value()
      
      real.Set(key, value)
      model[key] = value
      
      assert real.Get(key) == model[key]
` + "```" + ``,
					CodeExamples: `// Fuzzing and property-based testing patterns
package main

import (
    "fmt"
    "math/rand"
    "sort"
    "strings"
)

// Functions to fuzz-test

// URL parser (simplified)
type URL struct {
    Scheme string
    Host   string
    Path   string
    Query  string
}

func ParseURL(raw string) (*URL, error) {
    u := &URL{}
    
    // Scheme
    if idx := strings.Index(raw, "://"); idx >= 0 {
        u.Scheme = raw[:idx]
        raw = raw[idx+3:]
    }
    
    // Query
    if idx := strings.Index(raw, "?"); idx >= 0 {
        u.Query = raw[idx+1:]
        raw = raw[:idx]
    }
    
    // Host and path
    if idx := strings.Index(raw, "/"); idx >= 0 {
        u.Host = raw[:idx]
        u.Path = raw[idx:]
    } else {
        u.Host = raw
        u.Path = "/"
    }
    
    return u, nil
}

func (u *URL) String() string {
    var b strings.Builder
    if u.Scheme != "" {
        b.WriteString(u.Scheme)
        b.WriteString("://")
    }
    b.WriteString(u.Host)
    b.WriteString(u.Path)
    if u.Query != "" {
        b.WriteByte('?')
        b.WriteString(u.Query)
    }
    return b.String()
}

// Sortable type for property testing
func SortInts(s []int) []int {
    result := make([]int, len(s))
    copy(result, s)
    sort.Ints(result)
    return result
}

// Stack implementation to property-test
type Stack struct {
    items []int
}

func (s *Stack) Push(item int) {
    s.items = append(s.items, item)
}

func (s *Stack) Pop() (int, bool) {
    if len(s.items) == 0 {
        return 0, false
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item, true
}

func (s *Stack) Peek() (int, bool) {
    if len(s.items) == 0 {
        return 0, false
    }
    return s.items[len(s.items)-1], true
}

func (s *Stack) Len() int {
    return len(s.items)
}

// Property testing helpers
type PropertyTest struct {
    name    string
    runs    int
    passed  int
    failed  int
    errors  []string
}

func NewPropertyTest(name string, runs int) *PropertyTest {
    return &PropertyTest{name: name, runs: runs}
}

func (pt *PropertyTest) Run(check func() error) {
    for i := 0; i < pt.runs; i++ {
        if err := check(); err != nil {
            pt.failed++
            if len(pt.errors) < 5 { // Keep first 5 errors
                pt.errors = append(pt.errors, err.Error())
            }
        } else {
            pt.passed++
        }
    }
}

func (pt *PropertyTest) Report() {
    status := "PASS"
    if pt.failed > 0 {
        status = "FAIL"
    }
    fmt.Printf("  [%s] %s: %d/%d passed\n", status, pt.name, pt.passed, pt.runs)
    for _, err := range pt.errors {
        fmt.Printf("    Error: %s\n", err)
    }
}

func main() {
    // Property 1: Sort preserves length
    fmt.Println("=== Property-Based Tests: Sort ===")
    
    pt1 := NewPropertyTest("sort preserves length", 1000)
    pt1.Run(func() error {
        n := rand.Intn(100)
        input := make([]int, n)
        for i := range input {
            input[i] = rand.Intn(1000) - 500
        }
        sorted := SortInts(input)
        if len(sorted) != len(input) {
            return fmt.Errorf("length changed: %d → %d", len(input), len(sorted))
        }
        return nil
    })
    pt1.Report()
    
    // Property 2: Sort result is ordered
    pt2 := NewPropertyTest("sort result is ordered", 1000)
    pt2.Run(func() error {
        n := rand.Intn(100) + 1
        input := make([]int, n)
        for i := range input {
            input[i] = rand.Intn(1000) - 500
        }
        sorted := SortInts(input)
        for i := 1; i < len(sorted); i++ {
            if sorted[i-1] > sorted[i] {
                return fmt.Errorf("not sorted at index %d: %d > %d", i, sorted[i-1], sorted[i])
            }
        }
        return nil
    })
    pt2.Report()
    
    // Property 3: Sort is idempotent
    pt3 := NewPropertyTest("sort is idempotent", 1000)
    pt3.Run(func() error {
        n := rand.Intn(50) + 1
        input := make([]int, n)
        for i := range input {
            input[i] = rand.Intn(100)
        }
        once := SortInts(input)
        twice := SortInts(once)
        for i := range once {
            if once[i] != twice[i] {
                return fmt.Errorf("sort(sort(x)) != sort(x) at index %d", i)
            }
        }
        return nil
    })
    pt3.Report()
    
    // Property 4: Sort preserves elements
    pt4 := NewPropertyTest("sort preserves elements", 1000)
    pt4.Run(func() error {
        n := rand.Intn(50) + 1
        input := make([]int, n)
        for i := range input {
            input[i] = rand.Intn(100)
        }
        counts := make(map[int]int)
        for _, v := range input {
            counts[v]++
        }
        sorted := SortInts(input)
        sortedCounts := make(map[int]int)
        for _, v := range sorted {
            sortedCounts[v]++
        }
        for k, v := range counts {
            if sortedCounts[k] != v {
                return fmt.Errorf("element %d: count %d → %d", k, v, sortedCounts[k])
            }
        }
        return nil
    })
    pt4.Report()
    
    // Property tests for Stack
    fmt.Println("\n=== Property-Based Tests: Stack ===")
    
    // Push then Pop returns same value
    pt5 := NewPropertyTest("push then pop returns same value", 1000)
    pt5.Run(func() error {
        s := &Stack{}
        val := rand.Intn(10000)
        s.Push(val)
        got, ok := s.Pop()
        if !ok || got != val {
            return fmt.Errorf("push(%d) then pop: got %d, ok=%v", val, got, ok)
        }
        return nil
    })
    pt5.Report()
    
    // LIFO order
    pt6 := NewPropertyTest("LIFO order maintained", 1000)
    pt6.Run(func() error {
        s := &Stack{}
        n := rand.Intn(20) + 1
        values := make([]int, n)
        for i := 0; i < n; i++ {
            values[i] = rand.Intn(1000)
            s.Push(values[i])
        }
        // Pop should give reverse order
        for i := n - 1; i >= 0; i-- {
            got, ok := s.Pop()
            if !ok || got != values[i] {
                return fmt.Errorf("expected %d, got %d (ok=%v)", values[i], got, ok)
            }
        }
        return nil
    })
    pt6.Report()
    
    // Size tracking
    pt7 := NewPropertyTest("size tracks correctly", 1000)
    pt7.Run(func() error {
        s := &Stack{}
        expected := 0
        ops := rand.Intn(50)
        for i := 0; i < ops; i++ {
            if rand.Float32() < 0.6 || expected == 0 {
                s.Push(rand.Intn(100))
                expected++
            } else {
                s.Pop()
                expected--
            }
            if s.Len() != expected {
                return fmt.Errorf("expected len %d, got %d", expected, s.Len())
            }
        }
        return nil
    })
    pt7.Report()
    
    // URL round-trip property
    fmt.Println("\n=== Fuzz-like Tests: URL Parser ===")
    
    pt8 := NewPropertyTest("URL parse round-trip", 100)
    pt8.Run(func() error {
        schemes := []string{"http", "https", "ftp"}
        hosts := []string{"example.com", "localhost", "192.168.1.1"}
        paths := []string{"/", "/api/v1", "/users/123"}
        queries := []string{"", "q=hello", "page=1&limit=10"}
        
        scheme := schemes[rand.Intn(len(schemes))]
        host := hosts[rand.Intn(len(hosts))]
        path := paths[rand.Intn(len(paths))]
        query := queries[rand.Intn(len(queries))]
        
        raw := scheme + "://" + host + path
        if query != "" {
            raw += "?" + query
        }
        
        parsed, err := ParseURL(raw)
        if err != nil {
            return fmt.Errorf("parse %q: %v", raw, err)
        }
        
        reconstructed := parsed.String()
        if reconstructed != raw {
            return fmt.Errorf("round-trip failed:\n  input: %q\n  output: %q", raw, reconstructed)
        }
        return nil
    })
    pt8.Report()
}`,
				},
				{
					Title: "Mocking and Integration Testing",
					Content: `Effective testing requires both isolated unit tests (with mocks) and integration tests that verify real interactions.

**Interface-Based Mocking:**
` + "```" + `
Go's implicit interfaces make mocking natural:

// Production interface
type UserStore interface {
    GetUser(ctx context.Context, id string) (*User, error)
    CreateUser(ctx context.Context, u *User) error
    DeleteUser(ctx context.Context, id string) error
}

// Mock implementation (hand-written)
type MockUserStore struct {
    GetUserFunc    func(ctx context.Context, id string) (*User, error)
    CreateUserFunc func(ctx context.Context, u *User) error
    DeleteUserFunc func(ctx context.Context, id string) error
    
    // Call tracking
    GetUserCalls    []string
    CreateUserCalls []*User
}

func (m *MockUserStore) GetUser(ctx context.Context, id string) (*User, error) {
    m.GetUserCalls = append(m.GetUserCalls, id)
    if m.GetUserFunc != nil {
        return m.GetUserFunc(ctx, id)
    }
    return nil, errors.New("not implemented")
}

Usage in tests:
  mock := &MockUserStore{
      GetUserFunc: func(ctx context.Context, id string) (*User, error) {
          if id == "1" {
              return &User{ID: "1", Name: "Alice"}, nil
          }
          return nil, ErrNotFound
      },
  }
  
  svc := NewUserService(mock)
  user, err := svc.GetUser(ctx, "1")
  assert.NoError(t, err)
  assert.Equal(t, "Alice", user.Name)
  assert.Equal(t, []string{"1"}, mock.GetUserCalls)

Code generation tools:
  mockgen (gomock):
    //go:generate mockgen -source=store.go -destination=mock_store.go
    
  moq:
    //go:generate moq -out store_mock.go . UserStore
` + "```" + `

**Test Fixtures and Helpers:**
` + "```" + `
Test fixtures:
  testdata/ directory (special in Go):
    - Ignored by go build
    - Accessible in tests via relative path
    - Store test files, golden files, fixtures

  func TestParseConfig(t *testing.T) {
      data, err := os.ReadFile("testdata/config.yaml")
      if err != nil { t.Fatal(err) }
      // ...
  }

Golden files:
  Expected output stored in testdata/
  Update with -update flag
  
  var update = flag.Bool("update", false, "update golden files")
  
  func TestOutput(t *testing.T) {
      got := GenerateOutput()
      golden := filepath.Join("testdata", t.Name()+".golden")
      
      if *update {
          os.WriteFile(golden, got, 0644)
          return
      }
      
      want, _ := os.ReadFile(golden)
      if !bytes.Equal(got, want) {
          t.Errorf("output mismatch (run with -update to update)")
      }
  }

testcontainers-go (for real databases):
  container, _ := postgres.RunContainer(ctx,
      testcontainers.WithImage("postgres:16"),
      postgres.WithDatabase("testdb"),
  )
  defer container.Terminate(ctx)
  
  dsn, _ := container.ConnectionString(ctx)
  db, _ := sql.Open("postgres", dsn)
  // Run tests against real PostgreSQL
` + "```" + `

**Integration Test Patterns:**
` + "```" + `
Build tag separation:
  //go:build integration
  
  func TestDatabaseIntegration(t *testing.T) {
      if testing.Short() {
          t.Skip("skipping integration test in short mode")
      }
      // Real database tests
  }
  
  Run: go test -tags=integration -count=1 ./...
  Skip: go test -short ./...

Test environment:
  func TestMain(m *testing.M) {
      // Start external dependencies
      pool, _ := dockertest.NewPool("")
      resource, _ := pool.Run("postgres", "16", []string{
          "POSTGRES_PASSWORD=test",
      })
      defer pool.Purge(resource)
      
      // Wait for ready
      pool.Retry(func() error {
          db, _ := sql.Open("postgres", dsn)
          return db.Ping()
      })
      
      os.Exit(m.Run())
  }

HTTP handler testing:
  func TestGetUser(t *testing.T) {
      // Create handler with mock dependencies
      handler := NewHandler(mockStore)
      
      // Create test request
      req := httptest.NewRequest("GET", "/users/1", nil)
      rec := httptest.NewRecorder()
      
      // Execute
      handler.ServeHTTP(rec, req)
      
      // Assert
      assert.Equal(t, 200, rec.Code)
      
      var user User
      json.NewDecoder(rec.Body).Decode(&user)
      assert.Equal(t, "Alice", user.Name)
  }
  
httptest.NewServer for full server tests:
  ts := httptest.NewServer(handler)
  defer ts.Close()
  
  resp, _ := http.Get(ts.URL + "/users/1")
  // ... assert response
` + "```" + ``,
					CodeExamples: `// Mocking and integration testing patterns
package main

import (
    "context"
    "encoding/json"
    "errors"
    "fmt"
    "strings"
    "sync"
)

// Domain types
type User struct {
    ID    string ` + "`" + `json:"id"` + "`" + `
    Name  string ` + "`" + `json:"name"` + "`" + `
    Email string ` + "`" + `json:"email"` + "`" + `
}

// Store interface
type UserStore interface {
    Get(ctx context.Context, id string) (*User, error)
    Create(ctx context.Context, user *User) error
    List(ctx context.Context) ([]*User, error)
    Delete(ctx context.Context, id string) error
}

var ErrNotFound = errors.New("not found")

// Mock store for testing
type MockStore struct {
    mu         sync.RWMutex
    users      map[string]*User
    getCalls   []string
    createCalls []*User
    
    // Optional: override behavior
    GetFunc    func(ctx context.Context, id string) (*User, error)
    CreateFunc func(ctx context.Context, user *User) error
    GetError   error
}

func NewMockStore() *MockStore {
    return &MockStore{
        users: make(map[string]*User),
    }
}

func (m *MockStore) Get(ctx context.Context, id string) (*User, error) {
    m.mu.Lock()
    m.getCalls = append(m.getCalls, id)
    m.mu.Unlock()
    
    if m.GetFunc != nil {
        return m.GetFunc(ctx, id)
    }
    
    if m.GetError != nil {
        return nil, m.GetError
    }
    
    m.mu.RLock()
    defer m.mu.RUnlock()
    u, ok := m.users[id]
    if !ok {
        return nil, ErrNotFound
    }
    return u, nil
}

func (m *MockStore) Create(ctx context.Context, user *User) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.createCalls = append(m.createCalls, user)
    
    if m.CreateFunc != nil {
        return m.CreateFunc(ctx, user)
    }
    
    m.users[user.ID] = user
    return nil
}

func (m *MockStore) List(ctx context.Context) ([]*User, error) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    users := make([]*User, 0, len(m.users))
    for _, u := range m.users {
        users = append(users, u)
    }
    return users, nil
}

func (m *MockStore) Delete(ctx context.Context, id string) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    if _, ok := m.users[id]; !ok {
        return ErrNotFound
    }
    delete(m.users, id)
    return nil
}

// Assertions
func (m *MockStore) GetCallCount() int {
    m.mu.RLock()
    defer m.mu.RUnlock()
    return len(m.getCalls)
}

func (m *MockStore) CreateCallCount() int {
    m.mu.RLock()
    defer m.mu.RUnlock()
    return len(m.createCalls)
}

// Service under test
type UserService struct {
    store UserStore
}

func NewUserService(store UserStore) *UserService {
    return &UserService{store: store}
}

func (s *UserService) GetUser(ctx context.Context, id string) (*User, error) {
    if id == "" {
        return nil, errors.New("id is required")
    }
    return s.store.Get(ctx, id)
}

func (s *UserService) CreateUser(ctx context.Context, name, email string) (*User, error) {
    if name == "" {
        return nil, errors.New("name is required")
    }
    if email == "" || !strings.Contains(email, "@") {
        return nil, errors.New("valid email is required")
    }
    
    user := &User{
        ID:    fmt.Sprintf("user-%d", len(name)), // Simplified ID generation
        Name:  name,
        Email: email,
    }
    
    if err := s.store.Create(ctx, user); err != nil {
        return nil, fmt.Errorf("create user: %w", err)
    }
    return user, nil
}

// HTTP Handler testing simulation
type Request struct {
    Method string
    Path   string
    Body   string
}

type Response struct {
    Status int
    Body   string
}

type Handler struct {
    svc *UserService
}

func NewHandler(svc *UserService) *Handler {
    return &Handler{svc: svc}
}

func (h *Handler) Handle(req Request) Response {
    ctx := context.Background()
    
    switch {
    case req.Method == "GET" && strings.HasPrefix(req.Path, "/users/"):
        id := strings.TrimPrefix(req.Path, "/users/")
        user, err := h.svc.GetUser(ctx, id)
        if err != nil {
            if errors.Is(err, ErrNotFound) {
                return Response{Status: 404, Body: "{\"error\":\"not found\"}"}
            }
            return Response{Status: 500, Body: fmt.Sprintf("{\"error\":\"%s\"}", err)}
        }
        body, _ := json.Marshal(user)
        return Response{Status: 200, Body: string(body)}
        
    case req.Method == "POST" && req.Path == "/users":
        var input struct {
            Name  string ` + "`" + `json:"name"` + "`" + `
            Email string ` + "`" + `json:"email"` + "`" + `
        }
        json.Unmarshal([]byte(req.Body), &input)
        user, err := h.svc.CreateUser(ctx, input.Name, input.Email)
        if err != nil {
            return Response{Status: 400, Body: fmt.Sprintf("{\"error\":\"%s\"}", err)}
        }
        body, _ := json.Marshal(user)
        return Response{Status: 201, Body: string(body)}
        
    default:
        return Response{Status: 404, Body: "{\"error\":\"not found\"}"}
    }
}

// Test runner
type TestCase struct {
    Name     string
    Setup    func(*MockStore)
    Request  Request
    WantCode int
    Check    func(Response) error
}

func runTests(cases []TestCase) {
    for _, tc := range cases {
        store := NewMockStore()
        if tc.Setup != nil {
            tc.Setup(store)
        }
        
        svc := NewUserService(store)
        handler := NewHandler(svc)
        resp := handler.Handle(tc.Request)
        
        status := "PASS"
        var errMsg string
        
        if resp.Status != tc.WantCode {
            status = "FAIL"
            errMsg = fmt.Sprintf("status: got %d, want %d", resp.Status, tc.WantCode)
        } else if tc.Check != nil {
            if err := tc.Check(resp); err != nil {
                status = "FAIL"
                errMsg = err.Error()
            }
        }
        
        fmt.Printf("  [%s] %s", status, tc.Name)
        if errMsg != "" {
            fmt.Printf(": %s", errMsg)
        }
        fmt.Println()
    }
}

func main() {
    fmt.Println("=== Unit Tests with Mocks ===")
    
    // Test GetUser
    ctx := context.Background()
    store := NewMockStore()
    store.Create(ctx, &User{ID: "1", Name: "Alice", Email: "alice@example.com"})
    
    svc := NewUserService(store)
    
    // Success case
    user, err := svc.GetUser(ctx, "1")
    if err != nil {
        fmt.Printf("  FAIL: GetUser(1): %v\n", err)
    } else {
        fmt.Printf("  PASS: GetUser(1) = %s (%s)\n", user.Name, user.Email)
    }
    
    // Not found case
    _, err = svc.GetUser(ctx, "999")
    if errors.Is(err, ErrNotFound) {
        fmt.Printf("  PASS: GetUser(999) returns ErrNotFound\n")
    } else {
        fmt.Printf("  FAIL: expected ErrNotFound, got %v\n", err)
    }
    
    // Empty ID
    _, err = svc.GetUser(ctx, "")
    if err != nil {
        fmt.Printf("  PASS: GetUser('') returns error: %v\n", err)
    }
    
    // Verify mock was called correctly
    fmt.Printf("  Store.Get called %d times\n", store.GetCallCount())
    
    // Integration-style HTTP handler tests
    fmt.Println("\n=== HTTP Handler Tests ===")
    
    runTests([]TestCase{
        {
            Name: "GET /users/1 - found",
            Setup: func(s *MockStore) {
                s.Create(context.Background(), &User{ID: "1", Name: "Alice", Email: "alice@example.com"})
            },
            Request:  Request{Method: "GET", Path: "/users/1"},
            WantCode: 200,
            Check: func(r Response) error {
                var u User
                json.Unmarshal([]byte(r.Body), &u)
                if u.Name != "Alice" {
                    return fmt.Errorf("name: got %q, want %q", u.Name, "Alice")
                }
                return nil
            },
        },
        {
            Name:     "GET /users/999 - not found",
            Request:  Request{Method: "GET", Path: "/users/999"},
            WantCode: 404,
        },
        {
            Name:     "POST /users - success",
            Request:  Request{Method: "POST", Path: "/users", Body: "{\"name\":\"Bob\",\"email\":\"bob@example.com\"}"},
            WantCode: 201,
            Check: func(r Response) error {
                var u User
                json.Unmarshal([]byte(r.Body), &u)
                if u.Name != "Bob" {
                    return fmt.Errorf("name: got %q, want %q", u.Name, "Bob")
                }
                return nil
            },
        },
        {
            Name:     "POST /users - missing name",
            Request:  Request{Method: "POST", Path: "/users", Body: "{\"email\":\"bob@example.com\"}"},
            WantCode: 400,
        },
        {
            Name:     "POST /users - invalid email",
            Request:  Request{Method: "POST", Path: "/users", Body: "{\"name\":\"Bob\",\"email\":\"invalid\"}"},
            WantCode: 400,
        },
    })
    
    // Mock with custom behavior
    fmt.Println("\n=== Mock with Custom Behavior ===")
    
    errorStore := NewMockStore()
    errorStore.GetError = errors.New("database connection refused")
    
    errorSvc := NewUserService(errorStore)
    _, err = errorSvc.GetUser(ctx, "1")
    fmt.Printf("  Database error propagated: %v\n", err)
    
    // Mock with function override
    countStore := NewMockStore()
    callCount := 0
    countStore.GetFunc = func(ctx context.Context, id string) (*User, error) {
        callCount++
        return &User{ID: id, Name: fmt.Sprintf("User-%d", callCount)}, nil
    }
    
    countSvc := NewUserService(countStore)
    for i := 0; i < 3; i++ {
        u, _ := countSvc.GetUser(ctx, "1")
        fmt.Printf("  Call %d: %s\n", i+1, u.Name)
    }
}`,
				},
			},
		},
	})
}
