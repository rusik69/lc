package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          38,
			Title:       "Generics (Go 1.18+)",
			Description: "Learn Go generics: type parameters, constraints, and modern generic programming patterns.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to Generics",
					Content: "Added in Go 1.18, **Generics** allow you to write functions and data structures that work with any type while maintaining strict compile-time type safety.\n\n" +
						"**1. Type Parameters**: You define a generic by adding square brackets `[T Constraint]` after the function name or type name. `T` is a placeholder for the type, and `Constraint` specifies what that type is allowed to do.\n\n" +
						"**2. The 'any' Constraint**: If you don't need the type to have any specific methods, use `any`. This allows literally any type (int, struct, pointer, etc.) to be used.\n\n" +
						"**3. Performance**: Unlike Generics in some other languages (like Java's type erasure), Go's Generics are highly efficient and don't require runtime type assertions.",
					CodeExamples: `# GENERIC FUNCTION
func Reverse[T any](s []T) []T {
    res := make([]T, len(s))
    for i, v := range s {
        res[len(s)-1-i] = v
    }
    return res
}`,
				},

				{
					Title: "Type Constraints",
					Content: "Type constraints define the set of permitted types that can be used for a type parameter. They ensure that the generic function only uses operations that all permitted types support.\n\n" +
						"**1. Predefined Constraints**: \n" +
						"- **`any`**: Allows any type. Use this when the function doesn't perform any operations on the generic values (e.g., just storing them).\n" +
						"- **`comparable`**: Allows any type that supports `==` and `!=`. This is required for map keys and set-like logic.\n\n" +
						"**2. Custom Constraints**: You define constraints using interfaces. A type set can be defined using the union operator `|` (e.g., `int | float64`).\n\n" +
						"**3. The Approximation Operator (`~`)**: If you use `~int`, the constraint allows both `int` and any custom types defined with `type MyInt int`. Without `~`, only the exact type `int` is allowed.",
					CodeExamples: `# COMPARABLE CONSTRAINT
func Contains[T comparable](s []T, x T) bool {
    for _, v := range s {
        if v == x { return true }
    }
    return false
}

# TYPE SET CONSTRAINTS
type Numeric interface {
    ~int | ~float64 | ~complex128
}

func Add[T Numeric](a, b T) T {
    return a + b
}

# USAGE
type MyInt int
Add(MyInt(1), MyInt(2)) // OK because of ~int`,
				},

				{
					Title: "Generic Data Structures",
					Content: "Generics shine when building common data structures (Stacks, Queues, Trees) that should work for any type without sacrificing type safety or requiring multiple copies of the same logic.\n\n" +
						"**1. Type Definitions**: To define a generic struct, add the type parameter to the struct name: `type Container[T any] struct { data T }`.\n\n" +
						"**2. Methods on Generic Types**: Methods must also declare the type parameter: `func (c *Container[T]) Get() T { ... }`.\n\n" +
						"**3. Instantiation**: When you create an instance of a generic type, you typically specify the type: `c := Container[int]{}`. For functions, Go can often infer the type, but for structs, it's usually explicit.",
					CodeExamples: `# GENERIC STACK
type Stack[T any] []T

func (s *Stack[T]) Push(v T) {
    *s = append(*s, v)
}

func (s *Stack[T]) Pop() T {
    res := (*s)[len(*s)-1]
    *s = (*s)[:len(*s)-1]
    return res
}

# USAGE
var s Stack[string]
s.Push("Hello")
val := s.Pop() // val is inferred as string`,
				},

				{
					Title: "Generic Algorithms",
					Content: "Generics enable the creation of high-level utility functions like `Map`, `Filter`, and `Reduce` that work across different slice types while maintaining type safety.\n\n" +
						"**1. Functional Patterns**: You can now write a single `Filter` function that works for `[]int`, `[]string`, or complex structs. This promotes the 'Don't Repeat Yourself' (DRY) principle.\n\n" +
						"**2. Multiple Type Parameters**: Functions can have more than one type parameter. For example, a `Map` function needs two: one for the input type `T` and one for the output type `U`.\n\n" +
						"**3. Zero Value Pattern**: When working with generics, you might need to return a \"zero value\" for a type you don't know yet. A common pattern is `var zero T; return zero`.",
					CodeExamples: `# GENERIC MAP
func Map[T any, U any](s []T, f func(T) U) []U {
    res := make([]U, len(s))
    for i, v := range s {
        res[i] = f(v)
    }
    return res
}

# USAGE
nums := []int{1, 2, 3}
strs := Map(nums, func(n int) string {
    return fmt.Sprintf("Number %d", n)
})`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          39,
			Title:       "Error Handling",
			Description: "Learn Go's error handling patterns, custom errors, and panic/recover.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "Error Type",
					Content: "In Go, errors are not exceptions. Instead, an error is a normal value that implements the built-in `error` interface. This forces developers to handle failures explicitly, leading to more robust and predictable code.\n\n" +
						"**1. The Interface**: The `error` interface is extremely simple, requiring only a single method: `Error() string`. Any type with this method is an error.\n\n" +
						"**2. Returning Errors**: By convention, if a function can fail, its last return value should be of type `error`. If the function succeeds, it returns `nil` for the error.\n\n" +
						"**3. Sentinel Errors**: You can define constant-like error values using `errors.New(\"message\")` to represent specific failure conditions that callers can check for.",
					CodeExamples: `# CREATING ERRORS
err := errors.New("something went wrong")

# RETURNING ERRORS
func FetchData() (string, error) {
    if !connected {
        return "", errors.New("no connection")
    }
    return "data", nil
}

# BASIC HANDLING
data, err := FetchData()
if err != nil {
    log.Fatal(err)
}`,
				},

				{
					Title: "Error Handling Patterns",
					Content: "Go doesn't have try/catch blocks. Errors are values that must be inspected. Modern Go (1.13+) introduced powerful patterns for working with error chains.\n\n" +
						"**1. The 'Check' Rule**: Always check the error value before using any other returned data. The idiom is `if err != nil { ... }`.\n\n" +
						"**2. Error Wrapping**: Use `fmt.Errorf(\"... %w\", ..., err)` to add context to an error while preserving the original error. The `%w` verb is special; it \"wraps\" the error so it can be unpacked later.\n\n" +
						"**3. Inspection**: Use `errors.Is(err, target)` to check if an error (or any error in its chain) matches a specific value. Use `errors.As(err, &target)` to check if an error can be cast to a specific custom type.",
					CodeExamples: `# WRAPPING ERRORS
func ReadConfig() error {
    err := os.Open("config.json")
    if err != nil {
        return fmt.Errorf("failed to open config: %w", err)
    }
    return nil
}

# INSPECTING THE CHAIN
err := ReadConfig()
if errors.Is(err, os.ErrNotExist) {
    fmt.Println("File is missing!")
}

# TYPE ASSERTION (errors.As)
var pathErr *os.PathError
if errors.As(err, &pathErr) {
    fmt.Println("Path error on:", pathErr.Path)
}`,
				},

				{
					Title: "Custom Error Types",
					Content: "While `errors.New` is fine for simple messages, you often need to attach more data to an error (like a status code, a field name, or a timestamp). You do this by defining a **Custom Struct** that implements the `Error() string` method.\n\n" +
						"**1. Rich Context**: A custom error struct can have any number of fields. This allows the caller to extract specific information about the failure using type assertions or `errors.As`.\n\n" +
						"**2. Implementation**: You only need to provide the `Error() string` method. It is standard practice to return a human-readable string that describes the failure.\n\n" +
						"**3. Pointers vs Values**: It is generally recommended to use a **pointer receiver** for your `Error()` method if the struct is large or if you want to ensure the error type is unique (pointers to different instances aren't equal).",
					CodeExamples: `# CUSTOM ERROR STRUCT
type QueryError struct {
    Query  string
    Err    error
}

func (e *QueryError) Error() string {
    return fmt.Sprintf("query %q failed: %v", e.Query, e.Err)
}

# USAGE
func Search(q string) error {
    return &QueryError{Query: q, Err: errors.New("timeout")}
}

# INSPECTING CUSTOM DATA
err := Search("select *")
var qe *QueryError
if errors.As(err, &qe) {
    fmt.Println("Original query was:", qe.Query)
}`,
				},

				{
					Title: "Panic and Recover",
					Content: "Go provides a mechanism for handling \"exceptional\" conditions that cannot be managed by regular error handling. This is done through `panic` and `recover`.\n\n" +
						"**1. Panic**: Stops the normal execution of a goroutine. When a function calls `panic`, it immediately stops, and Go begins running any `defer`ed functions in that goroutine.\n\n" +
						"**2. Recover**: A built-in function that regains control of a panicking goroutine. It is **only useful inside deferred functions**. If called during normal execution, it does nothing and returns `nil`.\n\n" +
						"**3. When to Use**: Only use `panic` for truly unrecoverable errors (e.g., out of bounds, nil pointer dereference) or during program initialization. Never use it for normal control flow or expected errors (like a missing file).",
					CodeExamples: `# RECOVERING FROM PANIC
func Protect() {
    defer func() {
        if r := recover(); r != nil {
            fmt.Println("Recovered from:", r)
        }
    }()
    
    # TRIGGER PANIC
    var slice []int
    _ = slice[0] // Out of bounds panic!
}

# PROGRAM CONTINUES AFTER RECOVER
func main() {
    Protect()
    fmt.Println("I am still alive!")
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          40,
			Title:       "Concurrency",
			Description: "Master goroutines, channels, select, and concurrent programming patterns.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Goroutines",
					Content: "A **Goroutine** is a lightweight thread of execution managed by the Go runtime. They are the fundamental building blocks of Go's concurrency model, allowing you to run thousands of tasks simultaneously with very little overhead.\n\n" +
						"**1. M:N Scheduling**: Go uses an M:N scheduler, meaning it multiplexes M goroutines onto N OS threads. This is far more efficient than OS threads because context switching happens in user space, not kernel space.\n\n" +
						"**2. Dynamic Stacks**: OS threads usually have a fixed 2MB stack. Goroutines start with a tiny 2KB stack that grows and shrinks as needed, allowing you to run millions of them in a single program.\n\n" +
						"**3. Concurrency vs Parallelism**: Concurrency is about *dealing* with lots of things at once (structure). Parallelism is about *doing* lots of things at once (execution). Goroutines enable concurrency; if you have multiple CPU cores, they can run in parallel.",
					CodeExamples: `# STARTING A GOROUTINE
go doWork() // Runs doWork concurrently

# ANONYMOUS GOROUTINE
go func(msg string) {
    fmt.Println(msg)
}("Hello from goroutine")

# SYNCHRONIZATION (Wait)
var wg sync.WaitGroup
wg.Add(1)
go func() {
    defer wg.Done()
    process()
}()
wg.Wait()`,
				},

				{
					Title: "Channels",
					Content: "Channels are the pipes that connect goroutines. They allow you to send values from one goroutine to another, providing a powerful mechanism for both **communication** and **synchronization**.\n\n" +
						"**1. Unbuffered Channels**: Created with `make(chan int)`. They are synchronous. A send operation blocks until a receiver is ready, and a receive blocks until a sender is ready. This acts as a 'handshake' between routines.\n\n" +
						"**2. Buffered Channels**: Created with a capacity, e.g., `make(chan int, 10)`. They allow sending values without an immediate receiver, up to the buffer limit. They decouple the producer from the consumer.\n\n" +
						"**3. Closing Channels**: Use `close(ch)` to signal that no more values will be sent. It's important to remember that only the **sender** should close a channel, never the receiver.",
					CodeExamples: `# UNBUFFERED (Synchronization)
ch := make(chan string)
go func() { 
    ch <- "ping" // Blocks until main receives
}()
fmt.Println(<-ch)

# BUFFERED (Decoupling)
emails := make(chan string, 3)
emails <- "job1"
emails <- "job2" 
# Doesn't block yet!

# CHANNEL DIRECTIONS
func process(in <-chan int, out chan<- int) {
    for v := range in {
        out <- v * 2
    }
    close(out)
}`,
				},

				{
					Title: "Select Statement",
					Content: "The `select` statement lets a goroutine wait on multiple communication operations. It's like a `switch` but for channels. It blocks until one of its cases can run, then it executes that case.\n\n" +
						"**1. Non-Deterministic**: If multiple channels are ready at the same time, Go picks one at random. This prevents starvation of any single channel.\n\n" +
						"**2. Timeouts**: You can implement timeouts using `time.After(duration)`, which returns a channel that sends a value after the specified time.\n\n" +
						"**3. Non-Blocking Operations**: Adding a `default:` case makes the `select` non-blocking. If no channel is ready, the `default` case executes immediately.",
					CodeExamples: `# WAITING ON MULTIPLE CHANNELS
select {
case msg1 := <-ch1:
    fmt.Println("Received", msg1)
case ch2 <- "hi":
    fmt.Println("Sent hi to ch2")
case <-time.After(time.Second):
    fmt.Println("Timed out")
}

# NON-BLOCKING RECEIVE
select {
case msg := <-ch:
    fmt.Println("Got message:", msg)
default:
    fmt.Println("Nothing ready")
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
					Content: "A **Worker Pool** is a common pattern for managing a fixed number of goroutines that process a stream of tasks. This prevents your program from starting too many concurrent routines and overwhelming system resources (like CPU or file handles).\n\n" +
						"**1. Implementation**: You create a channel for the jobs and another for the results. A fixed set of workers (goroutines) read from the jobs channel, perform the work, and send the outcome to the results channel.\n\n" +
						"**2. Scalability**: You can easily scale your application by changing the number of workers in the pool. This provides predictable performance even as the volume of jobs increases.\n\n" +
						"**3. Lifecycle**: It's important to close the jobs channel once all jobs are sent so that the workers can exit gracefully after finishing their current task.",
					CodeExamples: `# WORKER DEFINITION
func worker(id int, jobs <-chan int, results chan<- int) {
    for j := range jobs {
        fmt.Printf("worker %d processing job %d\n", id, j)
        results <- j * 2
    }
}

# STARTING THE POOL
const numJobs = 5
jobs := make(chan int, numJobs)
results := make(chan int, numJobs)

for w := 1; w <= 3; w++ {
    go worker(w, jobs, results)
}

# SPREADING WORK
for j := 1; j <= numJobs; j++ {
    jobs <- j
}
close(jobs) // Workers loop until channel is empty`,
				},

				{
					Title: "Context Package",
					Content: "The `context` package is indispensable for managing the lifecycle, cancellation, and timeouts of concurrent operations. It allows you to propagate signals across API boundaries and goroutines.\n\n" +
						"**1. Cancellation**: You can create a context with a cancel function: `ctx, cancel := context.WithCancel(parent)`. Calling `cancel()` signals all goroutines watching that context to stop.\n\n" +
						"**2. Timeouts & Deadlines**: Use `context.WithTimeout` or `context.WithDeadline` to automatically signal cancellation after a certain duration or at a specific time. This is critical for preventing goroutines from hanging forever on slow network requests.\n\n" +
						"**3. Convention**: Always pass `context.Context` as the first argument to functions that perform I/O or long-running tasks. This is a Go standard.",
					CodeExamples: `# WITH TIMEOUT
ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
defer cancel() // Always cleanup!

# CHECKING FOR DONE
go func(ctx context.Context) {
    for {
        select {
        case <-ctx.Done():
            return // Exit if cancelled
        default:
            doWork()
        }
    }
}(ctx)`,
				},
				{
					Title: "Mutexes and Sync Package",
					Content: "When multiple goroutines access the same memory, you must synchronize that access to prevent **Data Races**. The `sync` package provides the tools to do this safely.\n\n" +
						"**1. Mutex**: A `sync.Mutex` (mutual exclusion lock) ensures only one goroutine can access a block of code at a time. Use `Lock()` to enter and `Unlock()` to leave.\n\n" +
						"**2. RWMutex**: A `sync.RWMutex` allows multiple simultaneous readers but only one writer. This is much faster for read-heavy workloads.\n\n" +
						"**3. WaitGroup**: Used to wait for a collection of goroutines to finish. You call `Add(delta)` before starting a routine, `Done()` when it finishes, and `Wait()` to block until all are done.",
					CodeExamples: `# SYNC.MUTEX
type AtomicInt struct {
    mu sync.Mutex
    n  int
}

func (a *AtomicInt) Inc() {
    a.mu.Lock()
    defer a.mu.Unlock()
    a.n++
}

# SYNC.WAITGROUP
var wg sync.WaitGroup
for i := 0; i < 10; i++ {
    wg.Add(1)
    go func() {
        defer wg.Done()
        doWork()
    }()
}
wg.Wait()`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          41,
			Title:       "Packages & Modules",
			Description: "Understand package organization, modules, and dependency management.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "Package Organization",
					Content: "Go Code is organized into **Packages**. A package is simply a directory containing one or more `.go` files that all declare the same package name at the top. This structure promotes modularity and reusability.\n\n" +
						"**1. Directories**: Every Go file must belong to a package. By convention, the package name is the same as the directory name. The entry point of a program must be in `package main`.\n\n" +
						"**2. Imports**: You access code from other packages using the `import` keyword. Go uses a full path (starting from the module name) to identify packages uniquely.\n\n" +
						"**3. Standard Library**: Go comes with a rich standard library (e.g., `fmt`, `net/http`, `os`) that you can import without any external dependencies.",
					CodeExamples: `# DIRECTORY STRUCTURE
/myproject
  /math
    adder.go
  main.go

# math/adder.go
package math
func Add(a, b int) int { return a + b }

# main.go
import "myproject/math"
func main() {
    math.Add(1, 2)
}`,
				},
				{
					Title: "Exported vs Unexported",
					Content: "Go uses a very simple but strict rule for **Visibility**: access control is determined by the first letter of an identifier (variables, functions, types, fields).\n\n" +
						"**1. Exported**: If an identifier starts with an **Upper Case** letter (e.g., `User`, `FetchData`), it is *exported* and visible to other packages that import yours.\n\n" +
						"**2. Unexported**: If it starts with a **Lower Case** letter (e.g., `user`, `dbConn`), it is *unexported* and private to its own package. It cannot be accessed from outside.\n\n" +
						"**3. Encapsulation**: This mechanism makes it easy to hide implementation details and provide a clean, public API for your modules.",
					CodeExamples: `# PUBLIC API
type Config struct {
    Port int // Exported field
    host string // Unexported field
}

func Connect() { ... } // Exported function

# PRIVATE LOGIC
func dial() { ... } // Unexported function`,
				},

				{
					Title: "Go Modules & Dependencies",
					Content: "Go Modules are the standard for dependency management. A module is a collection of Go packages that are versioned together as a single unit, defined by a `go.mod` file at the root.\n\n" +
						"**1. Initialization**: Use `go mod init <name>` to start a new module. This creates the `go.mod` file which tracks your dependencies and Go version.\n\n" +
						"**2. Tidy**: Running `go mod tidy` is a daily habit for Go developers. It adds missing dependencies used in your code and removes unused ones, ensuring your module files are clean.\n\n" +
						"**3. Versioning**: Go uses semantic versioning. The `go.sum` file captures cryptographic hashes of dependency versions to ensure that builds are reproducible and secure.",
					CodeExamples: `# INITIALIZE
go mod init github.com/user/project

# ADD DEPENDENCY
go get github.com/gin-gonic/gin@v1.9.0

# CLEANUP
go mod tidy

# REPRODUCIBLE BUILDS
# go.mod: lists requirements
# go.sum: lists content hashes`,
				},
				{
					Title: "Internal Packages",
					Content: "A directory named `internal` (and its subdirectories) has special meaning in Go. Packages within an `internal` folder can only be imported by packages rooted in the **parent** of that folder.\n\n" +
						"**1. Encapsulation**: This allows you to share code between packages in your project while strictly preventing external users from importing those private internal APIs.\n\n" +
						"**2. Example**: If you have `project/internal/secret`, only code inside `project/...` can import `secret`. Any other project trying to import it will fail at compile time.\n\n" +
						"**3. Structure**: This is commonly used for database logic, private utilities, or complex subsystem implementations that shouldn't be exposed as public API.",
					CodeExamples: `# DIRECTORY
/app
  /internal
    /db
      mysql.go
  main.go

# app/internal/db/mysql.go
package db
func Connect() { ... }

# app/main.go
import "app/internal/db" // WORKS

# github.com/other/app/main.go
import "app/internal/db" // COMPILE ERROR`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          42,
			Title:       "Standard Library",
			Description: "Explore Go's rich standard library: fmt, strings, io, time, json, http, and more.",
			Order:       11,
			Lessons: []problems.Lesson{

				{
					Title: "Common Utilities",
					Content: "Go's standard library provides a suite of essential packages for formatting, string manipulation, and type conversion that are used in almost every program.\n\n" +
						"**1. fmt**: Used for formatted I/O. Beyond printing, `fmt.Errorf` is the standard way to create errors with dynamic context.\n\n" +
						"**2. strings**: Contains high-performance functions for searching, splitting, and joined strings. It is optimized for UTF-8.\n\n" +
						"**3. strconv**: Handles conversions between strings and various basic types like integers, floats, and booleans with strict error checking.",
					CodeExamples: `# STRING FORMATTING
s := fmt.Sprintf("value: %.2f", 3.1415)

# STRING MANIPULATION
clean := strings.TrimSpace("  hello  ")
parts := strings.Split("a,b,c", ",")

# TYPE CONVERSION
n, err := strconv.Atoi("123")
b := strconv.FormatBool(true)`,
				},
				{
					Title: "I/O & Filesystem",
					Content: "Go provides powerful primitives for input and output operations through the `io` and `os` packages. Most I/O in Go is based on two fundamental interfaces: `Reader` and `Writer`.\n\n" +
						"**1. io.Reader & Writer**: These interfaces allow Go to handle data streams abstractly, whether they come from a file, a network connection, or an in-memory buffer.\n\n" +
						"**2. os Package**: Used for interacting with the operating system. It provides platform-independent functions for file creation, deletion, and environment variable access.\n\n" +
						"**3. bufio**: For high-performance I/O, the `bufio` package adds buffering to any `Reader` or `Writer`, reducing the number of expensive system calls.",
					CodeExamples: `# READING A FILE
data, err := os.ReadFile("config.json")
if err != nil { ... }

# WRITING WITH BUFFERING
f, _ := os.Create("log.txt")
w := bufio.NewWriter(f)
w.WriteString("Hello Go!")
w.Flush() // Don't forget to flush!

# STREAMING DATA
io.Copy(os.Stdout, f) // Pipe file to terminal`,
				},
				{
					Title: "Time & Scheduling",
					Content: "The `time` package is the standard for measuring and displaying time, as well as managing concurrent scheduling through timers and tickers.\n\n" +
						"**1. Time Points**: `time.Now()` gives the current local time. You can compare times using `Before()`, `After()`, and `Equal()` methods.\n\n" +
						"**2. Durations**: Represented by the `time.Duration` type (e.g., `time.Second * 5`). Always use durations for timeouts and sleep operations.\n\n" +
						"**3. Tickers**: A `time.Ticker` sends a value on a channel at a regular interval. It is the idiomatic way to perform periodic tasks in Go.",
					CodeExamples: `# MEASURING ELAPSED TIME
start := time.Now()
doWork()
fmt.Println("Took:", time.Since(start))

# PERIODIC TASK
ticker := time.NewTicker(100 * time.Millisecond)
defer ticker.Stop()

go func() {
    for t := range ticker.C {
        fmt.Println("Tick at", t)
    }
}()`,
				},
				{
					Title: "JSON & Data Serialization",
					Content: "Go's `encoding/json` package makes it easy to convert between Go structs and JSON strings. It uses **Struct Tags** to map between Go fields and JSON keys.\n\n" +
						"**1. Marshal**: Converts a Go value into a JSON byte slice. If a field is unexported (starts with lowercase), it will be ignored by the marshaler.\n\n" +
						"**2. Unmarshal**: Parses JSON data into a Go struct or map. You must pass a pointer to the destination variable.\n\n" +
						"**3. Customizing**: Use tags like `json:\"user_id\"` to change keys, or `json:\"-\"` to skip fields entirely.",
					CodeExamples: `# STRUCT WITH TAGS
type User struct {
    Name  string // Exported
    Email string // Exported
}

# ENCODE
u := User{Name: "Alice"}
data, _ := json.Marshal(u) 

# DECODE
var u2 User
json.Unmarshal(data, &u2)`,
				},
				{
					Title: "Networking & HTTP",
					Content: "The `net/http` package provides everything you need to build robust HTTP clients and production-grade web servers without relying on third-party frameworks.\n\n" +
						"**1. HTTP Server**: You define handlers (functions that satisfy the `HandlerFunc` signature) and map them to routes using `http.HandleFunc`.\n\n" +
						"**2. HTTP Client**: The `http.Get`, `http.Post`, and `http.Do` functions allow you to perform requests. Always remember to close the response body to prevent resource leaks.\n\n" +
						"**3. Context Aware**: Standard library HTTP operations support `context.Context`, allowing you to easily propagate timeouts and cancellation signals across the network.",
					CodeExamples: `# SIMPLE SERVER
http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprint(w, "Hello!")
})
log.Fatal(http.ListenAndServe(":8080", nil))

# HTTP GET CLIENT
resp, _ := http.Get("https://example.com")
defer resp.Body.Close() // MANDATORY
body, _ := io.ReadAll(resp.Body)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          43,
			Title:       "Testing & Benchmarking",
			Description: "Learn to write tests, benchmarks, and measure code performance.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced Testing Patterns",
					Content: "Go's testing philosophy emphasizes readability and robustness. Beyond simple assertions, Advanced Go developers use specific patterns to handle complex logic.\n\n" +
						"**1. Table-Driven Tests**: The most common pattern. By defining a slice of anonymous structs containing inputs and expected outputs, you can test dozens of cases with a single loop.\n\n" +
						"**2. Subtests**: Use `t.Run()` inside your loop to give each case a unique name. This allows you to run specific sub-cases from the command line using `-run=TestName/SubName`.\n\n" +
						"**3. Helper Functions**: Use `t.Helper()` in shared testing logic. This marks the function as a helper, so that when a test fails, the error message points to the caller's line number instead of the helper.",
					CodeExamples: `# TABLE-DRIVEN WITH SUBTESTS
func TestAdd(t *testing.T) {
    cases := []struct {
        name, a, b string
        want      int
    }{
        {"positive", 2, 2, 4},
        {"zero", 0, 0, 0},
    }

    for _, tc := range cases {
        t.Run(tc.name, func(t *testing.T) {
            got := Add(tc.a, tc.b)
            if got != tc.want {
                t.Errorf("got %d, want %d", got, tc.want)
            }
        })
    }
}`,
				},
				{
					Title: "Performance & Analysis",
					Content: "Go has built-in support for measuring performance and code quality through benchmarks and coverage analysis.\n\n" +
						"**1. Benchmarks**: Identify performance bottlenecks. A benchmark function: `func BenchmarkX(b *testing.B)` loops `b.N` times. Use `b.ResetTimer()` to exclude setup time from the measurement.\n\n" +
						"**2. Coverage**: Measure how much of your code is executed by tests using `go test -cover`. Use `-coverprofile` to generate a detailed report that highlights untested branches in HTML.\n\n" +
						"**3. Fuzzing**: A modern feature that generates random inputs to find edge cases that cause crashes. Use `func FuzzX(f *testing.F)` to provide a starting corpus and let Go explore the input space.",
					CodeExamples: `# BENCHMARKING
func BenchmarkSort(b *testing.B) {
    data := makeLargeSlice()
    b.ResetTimer() // Exclude setup
    for i := 0; i < b.N; i++ {
        sort.Ints(data)
    }
}

# COVERAGE COMMANDS
# go test -coverprofile=c.out
# go tool cover -html=c.out

# FUZZING
func FuzzParse(f *testing.F) {
    f.Add("initial corpus")
    f.Fuzz(func(t *testing.T, s string) {
        Parse(s) // Search for crashes
    })
}`,
				},
				{
					Title: "Documentation Examples",
					Content: "Examples in Go are a unique feature where your code examples also serve as living tests that validate the library's behavior and appear in the documentation.\n\n" +
						"**1. Function Prefix**: Must start with `Example`. To link to a function `Foo`, name it `ExampleFoo`.\n\n" +
						"**2. Output Validation**: Include a comment at the end of the function starting with `// Output:`. The testing tool captures standard output and compares it to this comment.\n\n" +
						"**3. Documentation**: These examples are automatically rendered by `pkgsite` or `godoc`, providing users with runnable, verified examples of how to use your API.",
					CodeExamples: `# RUNNABLE EXAMPLE
func ExampleReverse() {
    fmt.Println(Reverse("hello"))
    // Output: olleh
}

# RUN WITH: go test -v`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          44,
			Title:       "Advanced Topics",
			Description: "Explore reflection, generics, build tags, profiling, and Go best practices.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced Reflection",
					Content: "Reflection allows a Go program to inspect its own structure and values at runtime. It is a powerful but complex tool used by JSON encoders, ORMs, and dependency injection frameworks.\n\n" +
						"**1. Type & Value**: The two pillars of reflection are `reflect.Type` (what something is) and `reflect.Value` (what something contains). Use `reflect.TypeOf()` and `reflect.ValueOf()` to enter the reflection world.\n\n" +
						"**2. Kind**: While `Type` might be `main.User`, its `Kind` is `reflect.Struct`. Checking the Kind is essential for writing generic functions that handle different shapes of data.\n\n" +
						"**3. Performance**: Reflection is significantly slower than regular Go code and bypasses type safety at compile time. Rules of thumb: Use it only when necessary, and always validate `CanSet()` before modifying values.",
					CodeExamples: `# INSPECTING A VALUE
v := reflect.ValueOf(42)
fmt.Println("Type:", v.Type())
fmt.Println("Kind:", v.Kind())

# MODIFYING VIA POINTER
x := 10
ptr := reflect.ValueOf(&x).Elem()
if ptr.CanSet() {
    ptr.SetInt(20)
}

# STRUCT FIELD ITERATION
t := reflect.TypeOf(User{})
for i := 0; i < t.NumField(); i++ {
    fmt.Println(t.Field(i).Name)
}`,
				},
				{
					Title: "Conditional Compilation",
					Content: "Go provides a simple mechanism to include or exclude files during the build process based on environment variables, OS, or architecture.\n\n" +
						"**1. Syntax**: At the very top of a file, use `//go:build tag`. You can use boolean logic: `//go:build linux && !amd64`.\n\n" +
						"**2. Platform Support**: This is most commonly used for implementing OS-specific logic (e.g., file paths in Windows vs Linux) without cluttering the main logic with `if` statements.\n\n" +
						"**3. Suffixes**: Go also supports file naming suffixes like `file_linux.go` or `file_amd64.go`, which are automatically applied without explicit tags.",
					CodeExamples: `# RUN ONLY ON LINUX
//go:build linux

package main

# EXCLUDE WINDOWS
//go:build !windows

# USAGE IN TERMINAL
go build -tags=pro`,
				},
				{
					Title: "CGO & Native Code",
					Content: "CGO allows Go packages to call C code. It is the bridge between the managed Go runtime and native C libraries.\n\n" +
						"**1. Import \"C\"**: By importing the pseudo-package `C`, you gain access to C types and functions. Comments immediately preceding this import are treated as C header code.\n\n" +
						"**2. Cost**: Calling C from Go has significant overhead due to stack management and context switching. It also makes your binary non-portable and significantly complicates building.\n\n" +
						"**3. Safe Bridge**: Use `C.CString` to pass Go strings to C, but always remember to free them with `C.free` using `unsafe` to avoid memory leaks.",
					CodeExamples: `# C INTEGRATION
/*
#include <stdio.h>
void hello() { printf("Native C says hi!"); }
*/
import "C"

func main() {
    C.hello()
}`,
				},
				{
					Title: "Profiling & Optimization",
					Content: "Go has world-class support for profiling and tuning the performance of your applications.\n\n" +
						"**1. CPU Profiling**: Identifies where your program spends most of its time. Generate it during tests with `-cpuprofile` and analyze using `go tool pprof`.\n\n" +
						"**2. Memory Profiling**: Tracks heap allocations (`-memprofile`). This is crucial for identifying memory leaks and reducing the pressure on the Garbage Collector.\n\n" +
						"**3. Tracing**: The execution tracer provides a visualization of concurrent events in your program, helping you find issues with scheduling or high latency.",
					CodeExamples: `# GENERATE PROFILE
go test -cpuprofile cpu.out -memprofile mem.out

# ANALYZE PROFILE
go tool pprof -http=:8080 cpu.out

# EXECUTION TRACE
go test -trace trace.out
go tool trace trace.out`,
				},
			},

			ProblemIDs: []int{},
		},

		{
			ID:          48,
			Title:       "Database Operations",
			Description: "Connect to databases, execute queries, and work with database/sql package.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "SQL & Connection Pooling",
					Content: "Go's `database/sql` package followed by a driver (like `pgx` or `go-sql-driver/mysql`) provides a high-performance, thread-safe way to interact with relational databases.\n\n" +
						"**1. The *sql.DB Object**: It is not a single connection, but a **pool** of connections. You should create it once and share it across your entire application. It automatically manages opening and closing physical connections.\n\n" +
						"**2. Pool Tuning**: In production, always configure `SetMaxOpenConns` (limit total connections) and `SetConnMaxLifetime` (prevent stale connections). Proper tuning prevents your database from being overwhelmed during traffic spikes.\n\n" +
						"**3. SQL Injection**: Never use `fmt.Sprintf` to build queries. Always use placeholders (e.g., `?` or `$1`). The `database/sql` package handles parameter escaping automatically.",
					CodeExamples: `# CONNECTION POOL SETUP
db, _ := sql.Open("postgres", dsn)
db.SetMaxOpenConns(25)
db.SetMaxIdleConns(5)
db.SetConnMaxLifetime(5 * time.Minute)

# SAFE QUERY WITH PLACEHOLDERS
var name string
err := db.QueryRow("SELECT name FROM users WHERE id = $1", userID).Scan(&name)`,
				},
				{
					Title: "Robust Transactions",
					Content: "Transactions ensure that a series of database operations either all succeed or all fail, maintaining data integrity in complex workflows.\n\n" +
						"**1. ACID Compliance**: Transactions are started with `db.BeginTx()`. Using the `context` package allows you to automatically roll back transactions if a request is cancelled or times out.\n\n" +
						"**2. Atomic Operations**: Within a transaction (`*sql.Tx`), all operations must use the transaction object instead of the global `db` object to be part of the atomic unit.\n\n" +
						"**3. Defer Rollback**: A critical pattern is to `defer tx.Rollback()`. If `tx.Commit()` is called successfully, the deferred rollback does nothing; otherwise, it ensures the transaction is safely closed on any error or panic.",
					CodeExamples: `# TRANSACTION PATTERN
func transfer(db *sql.DB, ctx context.Context) error {
    tx, err := db.BeginTx(ctx, nil)
    if err != nil { return err }
    defer tx.Rollback() // Safety net

    _, err = tx.ExecContext(ctx, "UPDATE ...")
    if err != nil { return err }

    return tx.Commit() // Success!
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          49,
			Title:       "Advanced Concurrency Patterns",
			Description: "Deep dive into sophisticated concurrency patterns: Fan-out/Fan-in, Pipelines, Context-aware control, and robust Worker Pools.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "Pipelines & Fan-out/Fan-in",
					Content: `**Pipelines** are a series of stages connected by channels, where each stage is a group of goroutines running the same function.

**1. Pipeline Pattern:**
A pipeline stage takes values from an input channel, performs some work, and sends them to an output channel.
` + "```" + `
[ Source ] -> [ Stage A ] -> [ Stage B ] -> [ Sink ]
` + "```" + `

**2. Fan-out:**
Multiple goroutines read from the same channel until it's closed. This allows work to be distributed across multiple CPU cores.

**3. Fan-in:**
Consolidating multiple input channels into a single output channel. This is often used to collect results from a fan-out stage.`,
					CodeExamples: `// FAN-OUT / FAN-IN EXAMPLE
func producer(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        for _, n := range nums { out <- n }
        close(out)
    }()
    return out
}

func worker(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in { out <- n * n }
        close(out)
    }()
    return out
}

func merge(cs ...<-chan int) <-chan int {
    var wg sync.WaitGroup
    out := make(chan int)
    output := func(c <-chan int) {
        for n := range c { out <- n }
        wg.Done()
    }
    wg.Add(len(cs))
    for _, c := range cs { go output(c) }
    go func() { wg.Wait(); close(out) }()
    return out
}`,
				},
				{
					Title: "Context-aware Goroutine Control",
					Content: `Properly managing the lifecycle of goroutines is critical for preventing resource leaks (goroutine "leak").

**1. Done Channel Pattern:**
The most basic way to signal a goroutine to stop. Usually a 'chan struct{}' that is closed by the parent.

**2. Context Cancellation:**
Go's 'context' package is the standard for propagating cancellation signals and timeouts.

**3. Leaking Prevention:**
Always ensure every goroutine has a clearly defined exit condition. If a goroutine is blocked on a channel send/receive that will never happen, it is leaked memory.`,
					CodeExamples: `// PREVENTING LEAKS WITH CONTEXT
func longRunningWork(ctx context.Context, data <-chan string) {
    for {
        select {
        case <-ctx.Done():
            return // Clean exit
        case msg, ok := <-data:
            if !ok { return }
            process(msg)
        }
    }
}

// Timeout pattern
func operationWithDeadline() {
    ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
    defer cancel()
    
    go func() {
        // This will stop when context times out
        doWork(ctx)
    }()
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          50,
			Title:       "Idiomatic Design Patterns",
			Description: "Learn patterns that are specific to Go's philosophy: Functional Options, Decorators, Registries, and Middleware.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "Functional Options",
					Content: `Go doesn't support default parameter values or method overloading. The **Functional Options** pattern is the idiomatic solution for creating flexible and readable constructors.

**1. The Problem:**
Long constructors like 'NewServer(host, port, timeout, maxConn, tls, ...)' are hard to read and change.

**2. The Solution:**
Define an 'Option' type as a function: 'type Option func(*Server)'. Each option modifies the server instance.

**3. Benefits:**
- API is extensible without breaking changes.
- Default values are easy to manage.
- Code is extremely readable.`,
					CodeExamples: `// FUNCTIONAL OPTIONS PATTERN
type Server struct {
    port int
    timeout time.Duration
}

type Option func(*Server)

func WithTimeout(t time.Duration) Option {
    return func(s *Server) { s.timeout = t }
}

func NewServer(port int, opts ...Option) *Server {
    s := &Server{port: port, timeout: 30 * time.Second} // Defaults
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// Usage
srv := NewServer(8080, WithTimeout(10*time.Second))`,
				},
				{
					Title: "Decorator & Middleware",
					Content: `Due to Go's treatment of functions as first-class citizens and its focus on interfaces, the **Decorator** pattern is very powerful and widely used.

**1. Function Wrapping:**
Taking a function, adding logic (logging, timing, auth), and returning a function with the same signature.

**2. Middleware Pattern:**
The cornerstone of HTTP servers. It wraps a 'Handler' with common logic.

**3. Structural Benefits:**
Allows separation of concerns (keeping core logic clean of cross-cutting concerns like logging or monitoring).`,
					CodeExamples: `// HTTP MIDDLEWARE EXAMPLE
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        log.Printf("Request: %s %s", r.Method, r.URL.Path)
        next.ServeHTTP(w, r)
    })
}

// Generic Decorator
func TimedExecute(name string, f func()) {
    start := time.Now()
    f()
    log.Printf("%s took %v", name, time.Since(start))
}`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
