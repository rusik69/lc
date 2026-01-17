package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          38,
			Title:       "Error Handling",
			Description: "Learn Go's error handling patterns, custom errors, and panic/recover.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "Error Type",
					Content: `**Errors in Go:**
- Error is an interface type
- Convention: return error as last value
- nil means no error
- Check errors explicitly (no exceptions)

**Error Interface:**
type error interface {
    Error() string
}`,
					CodeExamples: `package main

import (
    "errors"
    "fmt"
)

func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

func main() {
    result, err := divide(10, 2)
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    fmt.Println("Result:", result)
}`,
				},
				{
					Title: "Error Handling Patterns",
					Content: `**Common Patterns:**
- Check error immediately after function call
- Return early on error
- Wrap errors with context
- Use errors.Is() and errors.As() for error inspection

**Best Practices:**
- Always check errors
- Provide context with errors
- Don't ignore errors with _`,
					CodeExamples: `package main

import (
    "errors"
    "fmt"
)

func processFile(filename string) error {
    // Simulate file operation
    if filename == "" {
        return errors.New("filename cannot be empty")
    }
    return nil
}

func main() {
    err := processFile("")
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    
    // Success path
    fmt.Println("File processed")
}`,
				},
				{
					Title: "Custom Error Types",
					Content: `**Custom Errors:**
- Create struct implementing error interface
- Can include additional fields
- Useful for error categorization
- Enable type assertions for error handling`,
					CodeExamples: `package main

import "fmt"

type ValidationError struct {
    Field   string
    Message string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("Validation error in %s: %s", e.Field, e.Message)
}

type NotFoundError struct {
    Resource string
}

func (e NotFoundError) Error() string {
    return fmt.Sprintf("%s not found", e.Resource)
}

func validateUser(name string) error {
    if name == "" {
        return ValidationError{
            Field:   "name",
            Message: "cannot be empty",
        }
    }
    return nil
}

func main() {
    err := validateUser("")
    if err != nil {
        fmt.Println(err)
    }
}`,
				},
				{
					Title: "Error Wrapping",
					Content: `**Error Wrapping (Go 1.13+):**
- Add context to errors
- Preserve original error
- Use fmt.Errorf with %w verb
- Unwrap with errors.Unwrap()
- Check with errors.Is()
- Extract with errors.As()`,
					CodeExamples: `package main

import (
    "errors"
    "fmt"
)

var ErrNotFound = errors.New("not found")

func findUser(id int) error {
    if id < 0 {
        return fmt.Errorf("invalid id %d: %w", id, ErrNotFound)
    }
    return nil
}

func main() {
    err := findUser(-1)
    
    // Check if error wraps ErrNotFound
    if errors.Is(err, ErrNotFound) {
        fmt.Println("User not found")
    }
    
    // Extract wrapped error
    var notFoundErr error
    if errors.As(err, &notFoundErr) {
        fmt.Println("Extracted:", notFoundErr)
    }
}`,
				},
				{
					Title: "Panic and Recover",
					Content: `**Panic:**
- Stops normal execution
- Deferred functions still run
- Should be used for programmer errors, not normal errors
- Similar to exceptions in other languages

**Recover:**
- Recovers from panic
- Only useful inside deferred functions
- Returns value passed to panic
- Use sparingly - prefer error returns`,
					CodeExamples: `package main

import "fmt"

func riskyFunction() {
    defer func() {
        if r := recover(); r != nil {
            fmt.Println("Recovered from panic:", r)
        }
    }()
    
    panic("something went wrong")
    fmt.Println("This won't execute")
}

func main() {
    riskyFunction()
    fmt.Println("Program continues")
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          39,
			Title:       "Concurrency",
			Description: "Master goroutines, channels, select, and concurrent programming patterns.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Goroutines",
					Content: `**Goroutines:**
- Lightweight threads managed by Go runtime
- Created with go keyword
- Very cheap to create (thousands possible)
- Multiplexed onto OS threads

**Starting Goroutines:**
go functionCall()

**Key Points:**
- Main goroutine must wait for others
- Use sync.WaitGroup or channels to coordinate
- Shared memory access needs synchronization`,
					CodeExamples: `package main

import (
    "fmt"
    "time"
)

func sayHello(name string) {
    for i := 0; i < 3; i++ {
        fmt.Printf("Hello from %s\n", name)
        time.Sleep(100 * time.Millisecond)
    }
}

func main() {
    // Start goroutine
    go sayHello("Alice")
    go sayHello("Bob")
    
    // Wait for goroutines
    time.Sleep(1 * time.Second)
    fmt.Println("Done")
}`,
				},
				{
					Title: "Channels",
					Content: `**Channels:**
- Communication mechanism between goroutines
- Typed conduit for sending/receiving values
- Created with make(chan Type)
- Send: ch <- value
- Receive: value := <-ch

**Channel Types:**
- Unbuffered: Synchronous, blocks until receiver ready
- Buffered: Asynchronous, blocks only when full`,
					CodeExamples: `package main

import "fmt"

func main() {
    // Unbuffered channel
    ch := make(chan int)
    
    // Send in goroutine
    go func() {
        ch <- 42
    }()
    
    // Receive
    value := <-ch
    fmt.Println(value)
    
    // Buffered channel
    buffered := make(chan string, 2)
    buffered <- "first"
    buffered <- "second"
    fmt.Println(<-buffered)
    fmt.Println(<-buffered)
}`,
				},
				{
					Title: "Select Statement",
					Content: `**Select:**
- Like switch but for channels
- Waits on multiple channel operations
- Chooses ready case (non-deterministic if multiple ready)
- Default case makes it non-blocking

**Common Uses:**
- Multiplexing channels
- Timeouts
- Non-blocking operations`,
					CodeExamples: `package main

import (
    "fmt"
    "time"
)

func main() {
    ch1 := make(chan string)
    ch2 := make(chan string)
    
    go func() {
        time.Sleep(1 * time.Second)
        ch1 <- "from ch1"
    }()
    
    go func() {
        time.Sleep(2 * time.Second)
        ch2 <- "from ch2"
    }()
    
    select {
    case msg1 := <-ch1:
        fmt.Println(msg1)
    case msg2 := <-ch2:
        fmt.Println(msg2)
    case <-time.After(3 * time.Second):
        fmt.Println("timeout")
    }
}`,
				},
				{
					Title: "Channel Directions",
					Content: `**Channel Directions:**
- Send-only: chan<- Type
- Receive-only: <-chan Type
- Bidirectional: chan Type (default)

**Benefits:**
- Type safety
- Clear intent
- Compiler enforces usage`,
					CodeExamples: `package main

import "fmt"

// Send-only channel
func sendOnly(ch chan<- int) {
    ch <- 42
    // <-ch  // Compile error: cannot receive
}

// Receive-only channel
func receiveOnly(ch <-chan int) {
    value := <-ch
    fmt.Println(value)
    // ch <- 1  // Compile error: cannot send
}

func main() {
    ch := make(chan int)
    
    go sendOnly(ch)
    receiveOnly(ch)
}`,
				},
				{
					Title: "Worker Pools",
					Content: `**Worker Pool Pattern:**
- Fixed number of goroutines (workers)
- Process jobs from channel
- Efficient resource usage
- Common concurrent pattern`,
					CodeExamples: `package main

import (
    "fmt"
    "sync"
)

func worker(id int, jobs <-chan int, results chan<- int) {
    for job := range jobs {
        fmt.Printf("Worker %d processing job %d\n", id, job)
        results <- job * 2
    }
}

func main() {
    jobs := make(chan int, 100)
    results := make(chan int, 100)
    
    // Start workers
    for w := 1; w <= 3; w++ {
        go worker(w, jobs, results)
    }
    
    // Send jobs
    for j := 1; j <= 5; j++ {
        jobs <- j
    }
    close(jobs)
    
    // Collect results
    for r := 1; r <= 5; r++ {
        fmt.Println("Result:", <-results)
    }
}`,
				},
				{
					Title: "Context Package",
					Content: `**Context:**
- Carries deadlines, cancellation signals, values
- Used for cancellation and timeouts
- Passed through call chains
- Created with context.Background() or context.TODO()

**Common Functions:**
- context.WithCancel()
- context.WithTimeout()
- context.WithDeadline()
- context.WithValue()`,
					CodeExamples: `package main

import (
    "context"
    "fmt"
    "time"
)

func doWork(ctx context.Context) {
    for {
        select {
        case <-ctx.Done():
            fmt.Println("Cancelled")
            return
        default:
            fmt.Println("Working...")
            time.Sleep(500 * time.Millisecond)
        }
    }
}

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()
    
    go doWork(ctx)
    time.Sleep(3 * time.Second)
}`,
				},
				{
					Title: "Mutexes and Sync Package",
					Content: `**Mutex:**
- Mutual exclusion lock
- Protects shared resources
- sync.Mutex for exclusive access
- sync.RWMutex for read-write access

**Other Sync Primitives:**
- sync.WaitGroup: Wait for goroutines
- sync.Once: Execute once
- sync.Map: Concurrent map`,
					CodeExamples: `package main

import (
    "fmt"
    "sync"
)

type SafeCounter struct {
    mu    sync.Mutex
    value int
}

func (c *SafeCounter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.value++
}

func (c *SafeCounter) GetValue() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.value
}

func main() {
    counter := SafeCounter{}
    var wg sync.WaitGroup
    
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter.Increment()
        }()
    }
    
    wg.Wait()
    fmt.Println(counter.GetValue())
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          40,
			Title:       "Packages & Modules",
			Description: "Understand package organization, modules, and dependency management.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "Package Organization",
					Content: `**Packages:**
- Collection of related Go files
- One package per directory
- Package name matches directory name (except main)
- Files in same package share namespace

**Package Declaration:**
package packagename

**Import Paths:**
- Standard library: "fmt", "net/http"
- Local packages: "./mypackage"
- Remote packages: "github.com/user/repo"`,
					CodeExamples: `// In file: math/calculator.go
package math

func Add(a, b int) int {
    return a + b
}

// In file: main.go
package main

import (
    "fmt"
    "./math"  // Local package
)

func main() {
    result := math.Add(2, 3)
    fmt.Println(result)
}`,
				},
				{
					Title: "Exported vs Unexported",
					Content: `**Visibility Rules:**
- Exported (public): Starts with capital letter
- Unexported (private): Starts with lowercase letter
- Only exported identifiers accessible from other packages
- Package-level visibility, not file-level

**Naming Convention:**
- Exported: Add, Calculate, User
- Unexported: add, calculate, user`,
					CodeExamples: `package math

// Exported - can be used from other packages
func Add(a, b int) int {
    return a + b
}

// Unexported - only within this package
func add(a, b int) int {
    return a + b
}

type Calculator struct {
    // Exported field
    Result int
    // Unexported field
    history []int
}`,
				},
				{
					Title: "Go Modules",
					Content: `**Modules:**
- Collection of related packages
- Defined by go.mod file
- Introduced in Go 1.11
- Replaces GOPATH

**go.mod File:**
- Module path
- Go version
- Dependencies with versions

**Commands:**
- go mod init: Create new module
- go mod tidy: Add/remove dependencies
- go mod download: Download dependencies`,
					CodeExamples: `// go.mod
module example.com/myproject

go 1.21

require (
    github.com/example/package v1.2.3
)

// Usage
package main

import "github.com/example/package"

func main() {
    package.DoSomething()
}`,
				},
				{
					Title: "Module Versioning",
					Content: `**Versioning:**
- Semantic versioning (v1.2.3)
- Major, minor, patch versions
- Tags in git repository
- Latest version selected automatically

**Version Selection:**
- go.mod specifies minimum version
- go get updates to latest compatible version
- Use go get package@version for specific version`,
					CodeExamples: `// In go.mod
require (
    github.com/example/package v1.2.3
)

// Update to latest
// go get github.com/example/package

// Update to specific version
// go get github.com/example/package@v1.3.0

// Update all dependencies
// go get -u ./...`,
				},
				{
					Title: "Dependency Management",
					Content: `**Dependency Management:**
- go.mod: Module definition and dependencies
- go.sum: Checksums for dependencies
- Version resolution is automatic
- Minimal version selection algorithm

**Best Practices:**
- Commit go.mod and go.sum
- Use go mod tidy regularly
- Pin versions for production
- Review dependency updates`,
					CodeExamples: `// go.mod
module myproject

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
)

// go.sum contains checksums
// Automatically generated, commit to version control`,
				},
				{
					Title: "Internal Packages",
					Content: `**Internal Packages:**
- Packages with "internal" in path
- Only importable by packages in parent directory tree
- Enforces encapsulation
- Useful for private implementation details

**Structure:**
project/
  internal/
    database/
    cache/`,
					CodeExamples: `// Directory structure:
// myproject/
//   internal/
//     db/
//       connection.go

// In connection.go
package db

func Connect() {
    // Implementation
}

// Can be imported from myproject/* but not from outside`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          41,
			Title:       "Standard Library",
			Description: "Explore Go's rich standard library: fmt, strings, io, time, json, http, and more.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "fmt Package",
					Content: `**fmt Package:**
- Formatting and printing
- fmt.Print, fmt.Println, fmt.Printf
- fmt.Scan, fmt.Scanf for input
- fmt.Sprintf for formatted strings
- fmt.Errorf for error creation`,
					CodeExamples: `package main

import "fmt"

func main() {
    name := "Alice"
    age := 30
    
    fmt.Print("Hello")
    fmt.Println("World")
    fmt.Printf("Name: %s, Age: %d\n", name, age)
    
    str := fmt.Sprintf("User: %s (%d)", name, age)
    fmt.Println(str)
}`,
				},
				{
					Title: "strings Package",
					Content: `**strings Package:**
- String manipulation functions
- Contains, HasPrefix, HasSuffix
- Index, LastIndex
- Replace, ReplaceAll
- Split, Join
- ToUpper, ToLower, Trim`,
					CodeExamples: `package main

import (
    "fmt"
    "strings"
)

func main() {
    s := "Hello, World"
    
    fmt.Println(strings.Contains(s, "World"))
    fmt.Println(strings.HasPrefix(s, "Hello"))
    fmt.Println(strings.Index(s, "World"))
    fmt.Println(strings.ReplaceAll(s, "World", "Go"))
    fmt.Println(strings.Split(s, ", "))
    fmt.Println(strings.ToUpper(s))
    fmt.Println(strings.Trim("  Hello  ", " "))
}`,
				},
				{
					Title: "strconv Package",
					Content: `**strconv Package:**
- String conversions
- Atoi, Itoa (int ↔ string)
- ParseFloat, FormatFloat
- ParseBool, FormatBool
- ParseInt, FormatInt`,
					CodeExamples: `package main

import (
    "fmt"
    "strconv"
)

func main() {
    // String to int
    num, _ := strconv.Atoi("42")
    fmt.Println(num)
    
    // Int to string
    str := strconv.Itoa(42)
    fmt.Println(str)
    
    // Parse float
    f, _ := strconv.ParseFloat("3.14", 64)
    fmt.Println(f)
    
    // Format float
    s := strconv.FormatFloat(3.14, 'f', 2, 64)
    fmt.Println(s)
}`,
				},
				{
					Title: "os and io Packages",
					Content: `**os Package:**
- Operating system interface
- File operations
- Environment variables
- Process management

**io Package:**
- I/O primitives
- io.Reader, io.Writer interfaces
- io.Copy, io.ReadAll
- io.Pipe`,
					CodeExamples: `package main

import (
    "fmt"
    "io"
    "os"
)

func main() {
    // Read file
    data, _ := os.ReadFile("file.txt")
    fmt.Println(string(data))
    
    // Write file
    os.WriteFile("output.txt", []byte("Hello"), 0644)
    
    // Environment variables
    path := os.Getenv("PATH")
    fmt.Println(path)
    
    // Copy
    src, _ := os.Open("source.txt")
    dst, _ := os.Create("dest.txt")
    io.Copy(dst, src)
}`,
				},
				{
					Title: "time Package",
					Content: `**time Package:**
- Time representation and operations
- time.Now(), time.Date()
- time.Duration for durations
- time.Ticker, time.Timer
- Time formatting and parsing`,
					CodeExamples: `package main

import (
    "fmt"
    "time"
)

func main() {
    // Current time
    now := time.Now()
    fmt.Println(now)
    
    // Format time
    fmt.Println(now.Format("2006-01-02 15:04:05"))
    
    // Parse time
    t, _ := time.Parse("2006-01-02", "2023-01-01")
    fmt.Println(t)
    
    // Duration
    duration := 2 * time.Hour
    fmt.Println(duration)
    
    // Sleep
    time.Sleep(1 * time.Second)
    
    // Ticker
    ticker := time.NewTicker(1 * time.Second)
    go func() {
        for t := range ticker.C {
            fmt.Println("Tick at", t)
        }
    }()
}`,
				},
				{
					Title: "encoding/json Package",
					Content: `**encoding/json Package:**
- JSON encoding and decoding
- json.Marshal (Go → JSON)
- json.Unmarshal (JSON → Go)
- Struct tags for field mapping
- Custom marshaling/unmarshaling`,
					CodeExamples: `package main

import (
    "encoding/json"
    "fmt"
)

type User struct {
    ID       int    ` + "`" + `json:"id"` + "`" + `
    Username string ` + "`" + `json:"username"` + "`" + `
    Email    string ` + "`" + `json:"email,omitempty"` + "`" + `
}

func main() {
    // Marshal
    u := User{ID: 1, Username: "alice", Email: "alice@example.com"}
    data, _ := json.Marshal(u)
    fmt.Println(string(data))
    
    // Unmarshal
    jsonStr := ` + "`" + `{"id":2,"username":"bob"}` + "`" + `
    var u2 User
    json.Unmarshal([]byte(jsonStr), &u2)
    fmt.Println(u2)
}`,
				},
				{
					Title: "net/http Package",
					Content: `**net/http Package:**
- HTTP client and server
- http.Get, http.Post
- http.HandleFunc, http.ListenAndServe
- http.Request, http.Response
- Middleware support`,
					CodeExamples: `package main

import (
    "fmt"
    "net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello, %s!", r.URL.Path[1:])
}

func main() {
    http.HandleFunc("/", handler)
    http.ListenAndServe(":8080", nil)
    
    // Client example
    resp, _ := http.Get("https://example.com")
    defer resp.Body.Close()
}`,
				},
				{
					Title: "regexp Package",
					Content: `**regexp Package:**
- Regular expressions
- regexp.Compile, regexp.MustCompile
- Match, FindString, FindAllString
- ReplaceAllString
- Compile once, use many times`,
					CodeExamples: `package main

import (
    "fmt"
    "regexp"
)

func main() {
    re := regexp.MustCompile(` + "`" + `\d+` + "`" + `)
    
    // Match
    matched := re.MatchString("abc123def")
    fmt.Println(matched)
    
    // Find
    found := re.FindString("abc123def456")
    fmt.Println(found)
    
    // Find all
    all := re.FindAllString("abc123def456", -1)
    fmt.Println(all)
    
    // Replace
    replaced := re.ReplaceAllString("abc123def", "X")
    fmt.Println(replaced)
}`,
				},
				{
					Title: "sort Package",
					Content: `**sort Package:**
- Sorting slices
- sort.Ints, sort.Float64s, sort.Strings
- sort.Sort for custom types
- Implementing sort.Interface
- sort.Search for binary search`,
					CodeExamples: `package main

import (
    "fmt"
    "sort"
)

func main() {
    // Sort integers
    nums := []int{3, 1, 4, 1, 5, 9, 2, 6}
    sort.Ints(nums)
    fmt.Println(nums)
    
    // Sort strings
    words := []string{"banana", "apple", "cherry"}
    sort.Strings(words)
    fmt.Println(words)
    
    // Custom sort
    people := []struct {
        Name string
        Age  int
    }{
        {"Alice", 30},
        {"Bob", 25},
        {"Charlie", 35},
    }
    
    sort.Slice(people, func(i, j int) bool {
        return people[i].Age < people[j].Age
    })
    fmt.Println(people)
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          42,
			Title:       "Testing & Benchmarking",
			Description: "Learn to write tests, benchmarks, and measure code performance.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Writing Tests",
					Content: `**Testing in Go:**
- Tests in files ending with _test.go
- Use testing package
- Test functions: func TestXxx(*testing.T)
- Run with: go test

**Test Functions:**
- Must start with Test
- Take *testing.T parameter
- Use t.Error or t.Fatal for failures`,
					CodeExamples: `package math

// math.go
func Add(a, b int) int {
    return a + b
}

// math_test.go
package math

import "testing"

func TestAdd(t *testing.T) {
    result := Add(2, 3)
    if result != 5 {
        t.Errorf("Add(2, 3) = %d; want 5", result)
    }
}

func TestAddNegative(t *testing.T) {
    result := Add(-1, -2)
    if result != -3 {
        t.Fatalf("Add(-1, -2) = %d; want -3", result)
    }
}`,
				},
				{
					Title: "Table-Driven Tests",
					Content: `**Table-Driven Tests:**
- Common Go testing pattern
- Define test cases in slice
- Loop through test cases
- Reduces code duplication
- Easy to add new cases`,
					CodeExamples: `package math

import "testing"

func TestMultiply(t *testing.T) {
    tests := []struct {
        name     string
        a        int
        b        int
        expected int
    }{
        {"positive", 2, 3, 6},
        {"negative", -2, 3, -6},
        {"zero", 0, 5, 0},
        {"one", 1, 10, 10},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := Multiply(tt.a, tt.b)
            if result != tt.expected {
                t.Errorf("Multiply(%d, %d) = %d; want %d",
                    tt.a, tt.b, result, tt.expected)
            }
        })
    }
}`,
				},
				{
					Title: "Test Coverage",
					Content: `**Test Coverage:**
- Measure how much code is tested
- Run: go test -cover
- Generate coverage profile: go test -coverprofile=coverage.out
- View HTML: go tool cover -html=coverage.out

**Coverage Goals:**
- Aim for high coverage
- Focus on critical paths
- Don't obsess over 100%`,
					CodeExamples: `// Run with coverage
// go test -cover

// Generate coverage file
// go test -coverprofile=coverage.out

// View in browser
// go tool cover -html=coverage.out

// Coverage percentage shows how much code is executed by tests`,
				},
				{
					Title: "Benchmarks",
					Content: `**Benchmarks:**
- Measure function performance
- Function: func BenchmarkXxx(*testing.B)
- Run with: go test -bench=.
- b.N is number of iterations

**Benchmark Functions:**
- Must start with Benchmark
- Take *testing.B parameter
- Loop b.N times
- Use b.ResetTimer() if needed`,
					CodeExamples: `package math

import "testing"

func BenchmarkAdd(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Add(10, 20)
    }
}

func BenchmarkAddLarge(b *testing.B) {
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        Add(1000000, 2000000)
    }
}

// Run: go test -bench=.
// Output shows ns/op (nanoseconds per operation)`,
				},
				{
					Title: "Examples",
					Content: `**Example Functions:**
- Documentation and testing combined
- Function: func ExampleXxx()
- Must have Output comment
- Run with: go test -run Example

**Example Output:**
- Comment: // Output: expected output
- Validates example produces correct output`,
					CodeExamples: `package math

import "fmt"

func ExampleAdd() {
    result := Add(2, 3)
    fmt.Println(result)
    // Output: 5
}

func ExampleMultiply() {
    result := Multiply(4, 5)
    fmt.Printf("Result: %d\n", result)
    // Output: Result: 20
}

// Examples appear in godoc documentation
// Run: go test -run Example`,
				},
				{
					Title: "Testing Best Practices",
					Content: `**Best Practices:**
- Write tests alongside code
- Test behavior, not implementation
- Use table-driven tests
- Test edge cases and errors
- Keep tests simple and readable
- Use subtests for organization
- Mock external dependencies
- Test in isolation`,
					CodeExamples: `package math

import "testing"

// Good: Test behavior
func TestDivide(t *testing.T) {
    result, err := Divide(10, 2)
    if err != nil {
        t.Fatal(err)
    }
    if result != 5 {
        t.Errorf("want 5, got %d", result)
    }
}

// Good: Test error case
func TestDivideByZero(t *testing.T) {
    _, err := Divide(10, 0)
    if err == nil {
        t.Error("expected error for division by zero")
    }
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          43,
			Title:       "Advanced Topics",
			Description: "Explore reflection, generics, build tags, profiling, and Go best practices.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "Reflection",
					Content: `**Reflection (reflect package):**
- Examine types and values at runtime
- reflect.TypeOf(), reflect.ValueOf()
- Type inspection and manipulation
- Used by encoding packages (JSON, etc.)
- Use sparingly - prefer type assertions

**Common Uses:**
- Type checking
- Struct field iteration
- Dynamic function calls`,
					CodeExamples: `package main

import (
    "fmt"
    "reflect"
)

func inspect(v interface{}) {
    t := reflect.TypeOf(v)
    val := reflect.ValueOf(v)
    
    fmt.Printf("Type: %s\n", t)
    fmt.Printf("Value: %v\n", val)
    
    if t.Kind() == reflect.Struct {
        for i := 0; i < t.NumField(); i++ {
            field := t.Field(i)
            value := val.Field(i)
            fmt.Printf("  %s: %v\n", field.Name, value)
        }
    }
}

func main() {
    type Person struct {
        Name string
        Age  int
    }
    inspect(Person{"Alice", 30})
}`,
				},
				{
					Title: "Generics (Go 1.18+)",
					Content: `**Generics:**
- Write code that works with multiple types
- Type parameters: [T Type]
- Constraints: [T comparable]
- Type inference

**Syntax:**
func FunctionName[T TypeConstraint](param T) T {
    // body
}`,
					CodeExamples: `package main

import "fmt"

// Generic function
func Max[T comparable](a, b T) T {
    if a > b {
        return a
    }
    return b
}

// Generic with constraint
func Sum[T int | float64](nums []T) T {
    var sum T
    for _, num := range nums {
        sum += num
    }
    return sum
}

// Generic type
type Stack[T any] struct {
    items []T
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() T {
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item
}

func main() {
    fmt.Println(Max(3, 5))
    fmt.Println(Sum([]int{1, 2, 3}))
}`,
				},
				{
					Title: "Build Tags",
					Content: `**Build Tags:**
- Conditional compilation
- // +build tag
- //go:build tag (newer syntax)
- Files compiled only when tag matches
- Use for platform-specific code

**Common Tags:**
- Operating systems: linux, windows, darwin
- Architectures: amd64, arm64
- Custom: dev, test, production`,
					CodeExamples: `// +build linux

package main

// This file only compiles on Linux

// +build !windows

package main

// This file compiles on all platforms except Windows

//go:build linux || darwin

package main

// Newer syntax - compiles on Linux or macOS

// Usage: go build -tags=dev`,
				},
				{
					Title: "CGO Basics",
					Content: `**CGO:**
- Call C code from Go
- import "C"
- Requires C compiler
- Use sparingly - adds complexity

**When to Use:**
- Interfacing with C libraries
- Performance-critical sections
- Legacy code integration

**Trade-offs:**
- Breaks cross-compilation
- Slower build times
- More complex builds`,
					CodeExamples: `package main

/*
#include <stdio.h>
#include <stdlib.h>
void hello() {
    printf("Hello from C!\n");
}
*/
import "C"

func main() {
    C.hello()
}

// Requires C compiler
// go build`,
				},
				{
					Title: "Profiling and Performance",
					Content: `**Profiling:**
- Measure program performance
- CPU profiling: go test -cpuprofile
- Memory profiling: go test -memprofile
- Use pprof tool to analyze

**Performance Tips:**
- Profile before optimizing
- Use sync.Pool for object reuse
- Preallocate slices when size known
- Avoid unnecessary allocations
- Use buffered channels appropriately`,
					CodeExamples: `package main

import (
    "os"
    "runtime/pprof"
)

func main() {
    // CPU profiling
    f, _ := os.Create("cpu.prof")
    pprof.StartCPUProfile(f)
    defer pprof.StopCPUProfile()
    
    // Your code here
    
    // Memory profiling
    mf, _ := os.Create("mem.prof")
    pprof.WriteHeapProfile(mf)
    mf.Close()
    
    // Analyze: go tool pprof cpu.prof
}`,
				},
				{
					Title: "Best Practices and Idioms",
					Content: `**Go Idioms:**
- Error handling: if err != nil { return err }
- Defer for cleanup
- Use interfaces for abstraction
- Prefer composition over inheritance
- Keep functions small and focused
- Use meaningful names
- Document exported functions

**Code Organization:**
- One package per directory
- Clear package boundaries
- Minimize dependencies
- Keep it simple`,
					CodeExamples: `package main

import (
    "fmt"
    "os"
)

// Good: Clear error handling
func readFile(filename string) ([]byte, error) {
    data, err := os.ReadFile(filename)
    if err != nil {
        return nil, fmt.Errorf("failed to read file: %w", err)
    }
    return data, nil
}

// Good: Use defer for cleanup
func processFile(filename string) error {
    file, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer file.Close()  // Always closes
    
    // Process file
    return nil
}

// Good: Small, focused functions
func validateUser(name string) error {
    if name == "" {
        return fmt.Errorf("name cannot be empty")
    }
    return nil
}`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
