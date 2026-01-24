package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          37,
			Title:       "Generics (Go 1.18+)",
			Description: "Learn Go generics: type parameters, constraints, and modern generic programming patterns.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to Generics",
					Content: `**What are Generics?**

Generics were introduced in Go 1.18 (March 2022) to enable writing code that works with multiple types while maintaining type safety. Before generics, Go developers had to use interfaces, type assertions, or code generation to achieve similar functionality.

**Why Generics Matter:**

**1. Type Safety:**
- Compile-time type checking
- No runtime type assertions needed
- Prevents type-related bugs

**2. Code Reuse:**
- Write functions that work with multiple types
- Reduce code duplication
- Maintainable and DRY (Don't Repeat Yourself)

**3. Performance:**
- No runtime overhead (unlike interfaces with type assertions)
- Compiler generates specific code for each type
- Same performance as non-generic code

**4. Better APIs:**
- More expressive function signatures
- Clearer intent in code
- Better IDE support and autocomplete

**Key Concepts:**

**Type Parameters:**
- Syntax: func FunctionName[T TypeConstraint](param T) T
- T is a type parameter (can be any name)
- TypeConstraint defines what types T can be

**Type Constraints:**
- Define what types are allowed
- Can be interfaces or type sets
- Built-in constraints: comparable, any, constraints package

**Generic Functions:**
- Functions that work with type parameters
- Type inferred from arguments
- Can be explicitly specified

**Generic Types:**
- Types that can work with multiple underlying types
- Examples: Stack[T], Queue[T], Tree[T]

**Real-World Applications:**
- Collections: Generic stacks, queues, trees
- Algorithms: Generic sorting, searching, filtering
- Utilities: Generic min/max, map/filter/reduce
- Data structures: Generic linked lists, heaps, graphs`,
					CodeExamples: `package main

import "fmt"

// Generic function - works with any comparable type
func Max[T comparable](a, b T) T {
    // Note: This is simplified - real implementation needs constraints.Ordered
    return a  // Placeholder
}

// Generic function with constraints
func Find[T comparable](slice []T, value T) int {
    for i, v := range slice {
        if v == value {
            return i
        }
    }
    return -1
}

// Generic type
type Stack[T any] struct {
    items []T
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item, true
}

func main() {
    // Type inference
    index := Find([]int{1, 2, 3, 4, 5}, 3)
    fmt.Println(index)  // 2
    
    // Explicit type specification
    index2 := Find[string]([]string{"a", "b", "c"}, "b")
    fmt.Println(index2)  // 1
    
    // Generic type
    intStack := Stack[int]{}
    intStack.Push(1)
    intStack.Push(2)
    val, _ := intStack.Pop()
    fmt.Println(val)  // 2
    
    stringStack := Stack[string]{}
    stringStack.Push("hello")
    stringStack.Push("world")
    str, _ := stringStack.Pop()
    fmt.Println(str)  // world
}`,
				},
				{
					Title: "Type Constraints",
					Content: `**Understanding Type Constraints:**

Type constraints define what types can be used with a generic function or type. They ensure type safety and enable operations on generic types.

**Built-in Constraints:**

**1. any:**
- Alias for interface{}
- Any type allowed
- Most permissive constraint

**2. comparable:**
- Types that can be compared with == and !=
- Includes: numbers, strings, booleans, pointers, arrays, structs
- Excludes: slices, maps, functions

**3. constraints Package (Go 1.18+):**
- constraints.Ordered: Types that can be ordered (<, >, <=, >=)
- constraints.Signed: Signed integer types
- constraints.Unsigned: Unsigned integer types
- constraints.Integer: All integer types
- constraints.Float: All float types
- constraints.Complex: All complex types

**Custom Constraints:**

**Interface Constraints:**
- Use interfaces as constraints
- Type must implement interface methods
- Enables method calls on generic types

**Type Sets (Go 1.18+):**
- Define sets of types explicitly
- More flexible than interfaces
- Can specify exact types

**Union Constraints:**
- Combine multiple constraints
- Type must satisfy any constraint
- Syntax: T1 | T2 | T3

**Best Practices:**
- Use most specific constraint possible
- Prefer constraints.Ordered over any for comparisons
- Use interfaces for method requirements
- Document constraint requirements`,
					CodeExamples: `package main

import (
    "fmt"
    "golang.org/x/exp/constraints"
)

// Using constraints.Ordered for comparisons
func Min[T constraints.Ordered](a, b T) T {
    if a < b {
        return a
    }
    return b
}

// Using constraints.Integer for integer operations
func Sum[T constraints.Integer](numbers []T) T {
    var sum T
    for _, n := range numbers {
        sum += n
    }
    return sum
}

// Custom interface constraint
type Stringer interface {
    String() string
}

func Print[T Stringer](value T) {
    fmt.Println(value.String())
}

// Type set constraint
type Numeric interface {
    ~int | ~float64  // ~ means underlying type
}

func Double[T Numeric](value T) T {
    return value * 2
}

// Union constraint
type Number interface {
    int | float64 | string
}

func Process[T Number](value T) {
    fmt.Printf("Processing: %v\n", value)
}

func main() {
    // Using constraints.Ordered
    minInt := Min(5, 3)
    fmt.Println(minInt)  // 3
    
    minFloat := Min(3.14, 2.71)
    fmt.Println(minFloat)  // 2.71
    
    // Using constraints.Integer
    sum := Sum([]int{1, 2, 3, 4, 5})
    fmt.Println(sum)  // 15
    
    // Custom constraint
    type MyInt int
    doubled := Double(MyInt(5))
    fmt.Println(doubled)  // 10
}`,
				},
				{
					Title: "Generic Data Structures",
					Content: `**Generic Collections:**

Generics enable type-safe, reusable data structures without sacrificing performance or requiring code generation.

**Common Generic Data Structures:**

**1. Stack:**
- LIFO (Last In, First Out)
- Push and Pop operations
- Type-safe element storage

**2. Queue:**
- FIFO (First In, First Out)
- Enqueue and Dequeue operations
- Useful for task processing

**3. Linked List:**
- Dynamic size
- Efficient insertions/deletions
- Type-safe node values

**4. Binary Tree:**
- Hierarchical structure
- Generic node values
- Type-safe operations

**5. Heap/Priority Queue:**
- Maintains order property
- Generic element types
- Efficient min/max operations

**Benefits:**
- Type safety at compile time
- No type assertions needed
- Better performance than interface-based solutions
- Code reuse across types

**Real-World Use Cases:**
- Generic collections in libraries
- Type-safe algorithms
- Reusable data structures
- API design with generics`,
					CodeExamples: `package main

import "fmt"

// Generic Stack
type Stack[T any] struct {
    items []T
}

func NewStack[T any]() *Stack[T] {
    return &Stack[T]{items: make([]T, 0)}
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item, true
}

func (s *Stack[T]) Peek() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    return s.items[len(s.items)-1], true
}

// Generic Queue
type Queue[T any] struct {
    items []T
}

func NewQueue[T any]() *Queue[T] {
    return &Queue[T]{items: make([]T, 0)}
}

func (q *Queue[T]) Enqueue(item T) {
    q.items = append(q.items, item)
}

func (q *Queue[T]) Dequeue() (T, bool) {
    if len(q.items) == 0 {
        var zero T
        return zero, false
    }
    item := q.items[0]
    q.items = q.items[1:]
    return item, true
}

// Generic Binary Tree Node
type TreeNode[T comparable] struct {
    Value T
    Left  *TreeNode[T]
    Right *TreeNode[T]
}

func (n *TreeNode[T]) Insert(value T) {
    // Simplified insertion logic
    if n.Value < value {  // Requires constraints.Ordered
        if n.Right == nil {
            n.Right = &TreeNode[T]{Value: value}
        } else {
            n.Right.Insert(value)
        }
    } else {
        if n.Left == nil {
            n.Left = &TreeNode[T]{Value: value}
        } else {
            n.Left.Insert(value)
        }
    }
}

func main() {
    // Generic Stack with int
    intStack := NewStack[int]()
    intStack.Push(1)
    intStack.Push(2)
    intStack.Push(3)
    val, _ := intStack.Pop()
    fmt.Println(val)  // 3
    
    // Generic Stack with string
    stringStack := NewStack[string]()
    stringStack.Push("hello")
    stringStack.Push("world")
    str, _ := stringStack.Pop()
    fmt.Println(str)  // world
    
    // Generic Queue
    queue := NewQueue[int]()
    queue.Enqueue(1)
    queue.Enqueue(2)
    queue.Enqueue(3)
    item, _ := queue.Dequeue()
    fmt.Println(item)  // 1 (FIFO)
}`,
				},
				{
					Title: "Generic Algorithms",
					Content: `**Generic Algorithm Patterns:**

Generics enable writing algorithms that work with any type while maintaining type safety and performance.

**Common Generic Algorithms:**

**1. Map/Transform:**
- Apply function to each element
- Return new slice with transformed values
- Type-safe transformation

**2. Filter:**
- Select elements matching predicate
- Return filtered slice
- Generic predicate functions

**3. Reduce/Fold:**
- Combine elements into single value
- Generic accumulator type
- Flexible reduction operations

**4. Find/Search:**
- Find element matching condition
- Generic search criteria
- Type-safe comparisons

**5. Sort:**
- Sort slices of any comparable type
- Custom comparison functions
- Efficient sorting algorithms

**Benefits:**
- Reusable across types
- Type-safe operations
- No runtime overhead
- Better than interface-based solutions

**Best Practices:**
- Use appropriate constraints
- Keep algorithms generic and reusable
- Document type requirements
- Test with multiple types`,
					CodeExamples: `package main

import (
    "fmt"
    "golang.org/x/exp/constraints"
)

// Generic Map function
func Map[T, U any](slice []T, fn func(T) U) []U {
    result := make([]U, len(slice))
    for i, v := range slice {
        result[i] = fn(v)
    }
    return result
}

// Generic Filter function
func Filter[T any](slice []T, predicate func(T) bool) []T {
    var result []T
    for _, v := range slice {
        if predicate(v) {
            result = append(result, v)
        }
    }
    return result
}

// Generic Reduce function
func Reduce[T, U any](slice []T, initial U, fn func(U, T) U) U {
    result := initial
    for _, v := range slice {
        result = fn(result, v)
    }
    return result
}

// Generic Find function
func Find[T comparable](slice []T, value T) (int, bool) {
    for i, v := range slice {
        if v == value {
            return i, true
        }
    }
    return -1, false
}

// Generic Contains function
func Contains[T comparable](slice []T, value T) bool {
    _, found := Find(slice, value)
    return found
}

// Generic Min/Max
func Min[T constraints.Ordered](slice []T) (T, bool) {
    if len(slice) == 0 {
        var zero T
        return zero, false
    }
    min := slice[0]
    for _, v := range slice[1:] {
        if v < min {
            min = v
        }
    }
    return min, true
}

func main() {
    // Map: Square numbers
    numbers := []int{1, 2, 3, 4, 5}
    squared := Map(numbers, func(x int) int {
        return x * x
    })
    fmt.Println(squared)  // [1, 4, 9, 16, 25]
    
    // Filter: Even numbers
    evens := Filter(numbers, func(x int) bool {
        return x%2 == 0
    })
    fmt.Println(evens)  // [2, 4]
    
    // Reduce: Sum
    sum := Reduce(numbers, 0, func(acc, x int) int {
        return acc + x
    })
    fmt.Println(sum)  // 15
    
    // Find
    index, found := Find(numbers, 3)
    fmt.Println(index, found)  // 2 true
    
    // Contains
    hasFive := Contains(numbers, 5)
    fmt.Println(hasFive)  // true
    
    // Min
    min, ok := Min(numbers)
    fmt.Println(min, ok)  // 1 true
}`,
				},
			},
			ProblemIDs: []int{},
		},
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
				{
					Title: "Error Wrapping Advanced",
					Content: `Advanced error wrapping techniques enable better error handling and debugging in complex applications.

**Error Wrapping Patterns:**

**1. Multi-Level Wrapping:**
- Wrap errors at each level
- Preserve full error chain
- Enables precise error matching

**2. Error Annotations:**
- Add context at each level
- Include function names, parameters
- Helpful for debugging

**3. Error Inspection:**
- errors.Is(): Check for specific error in chain
- errors.As(): Extract error of specific type
- errors.Unwrap(): Get next error

**4. Error Chains:**
- Errors can wrap multiple levels
- Chain preserved through wrapping
- Can traverse entire chain

**Best Practices:**
- Wrap at appropriate levels
- Add meaningful context
- Use errors.Is() for sentinel errors
- Use errors.As() for error types
- Don't wrap unnecessarily`,
					CodeExamples: `package main

import (
    "errors"
    "fmt"
)

var (
    ErrNotFound     = errors.New("not found")
    ErrInvalidInput = errors.New("invalid input")
)

func findUser(id int) error {
    if id < 0 {
        return fmt.Errorf("invalid user id %d: %w", id, ErrInvalidInput)
    }
    if id > 1000 {
        return fmt.Errorf("user %d: %w", id, ErrNotFound)
    }
    return nil
}

func getUserProfile(id int) error {
    user, err := findUser(id)
    if err != nil {
        return fmt.Errorf("failed to get user profile: %w", err)
    }
    // Process user...
    return nil
}

func handleRequest(id int) error {
    profile, err := getUserProfile(id)
    if err != nil {
        return fmt.Errorf("request failed: %w", err)
    }
    // Handle profile...
    return nil
}

func main() {
    err := handleRequest(-1)
    
    // Check entire chain
    if errors.Is(err, ErrInvalidInput) {
        fmt.Println("Invalid input detected")
    }
    
    if errors.Is(err, ErrNotFound) {
        fmt.Println("Not found detected")
    }
    
    // Print full error chain
    fmt.Printf("Full error: %v\n", err)
    
    // Unwrap chain
    current := err
    for current != nil {
        fmt.Printf("  %v\n", current)
        current = errors.Unwrap(current)
    }
}`,
				},
				{
					Title: "Error Handling Best Practices",
					Content: `Following best practices ensures robust, maintainable error handling throughout your Go application.

**Core Principles:**
- Check errors immediately
- Return errors up the call stack
- Add context when wrapping
- Don't ignore errors
- Handle errors appropriately

**Error Handling Patterns:**

**1. Early Return:**
- Return immediately on error
- Reduces nesting
- Clear error paths
- Most common pattern

**2. Error Wrapping:**
- Add context at each level
- Use fmt.Errorf with %w
- Preserve original error
- Enable error inspection

**3. Error Checking:**
- Always check errors
- Don't use _ to ignore
- Handle or propagate
- Log appropriately

**4. Error Types:**
- Create custom error types
- Use errors.Is() and errors.As()
- Enable type-specific handling
- Export error variables

**Common Mistakes:**
- Ignoring errors with _
- Not checking errors
- Losing error context
- Panicking for normal errors
- Not wrapping errors

**Best Practices:**
- Check errors immediately
- Return early on error
- Wrap with context
- Use errors.Is() and errors.As()
- Document error behavior
- Create error types for categories
- Log errors appropriately
- Don't panic for normal errors`,
					CodeExamples: `package main

import (
    "errors"
    "fmt"
    "os"
)

// Good: Early return
func readConfig(filename string) (string, error) {
    data, err := os.ReadFile(filename)
    if err != nil {
        return "", fmt.Errorf("failed to read config %s: %w", filename, err)
    }
    return string(data), nil
}

// Good: Error wrapping
func processUser(id int) error {
    user, err := getUser(id)
    if err != nil {
        return fmt.Errorf("failed to get user %d: %w", id, err)
    }
    
    err = validateUser(user)
    if err != nil {
        return fmt.Errorf("validation failed for user %d: %w", id, err)
    }
    
    return nil
}

// Good: Error checking
func safeOperation() error {
    result, err := riskyOperation()
    if err != nil {
        return err  // Don't ignore
    }
    
    // Use result only if no error
    fmt.Println(result)
    return nil
}

// Bad: Ignoring error
func badExample() {
    data, _ := os.ReadFile("file.txt")  // Don't do this!
    fmt.Println(string(data))
}

// Good: Error type checking
func handleError(err error) {
    var validationErr ValidationError
    if errors.As(err, &validationErr) {
        fmt.Printf("Validation error: %s\n", validationErr.Message)
        return
    }
    
    var notFoundErr NotFoundError
    if errors.As(err, &notFoundErr) {
        fmt.Printf("Not found: %s\n", notFoundErr.Resource)
        return
    }
    
    fmt.Printf("Unknown error: %v\n", err)
}

func main() {
    config, err := readConfig("config.json")
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    fmt.Println("Config:", config)
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
- Multiplexed onto OS threads (M:N threading model)
- Stack starts at 2KB, grows as needed (up to 1GB)

**Starting Goroutines:**
go functionCall()

**Key Characteristics (2024-2025 Best Practices):**

**1. Lightweight:**
- 2KB initial stack (vs 1-2MB for OS threads)
- Can create millions of goroutines
- Efficient context switching
- Managed by Go runtime scheduler

**2. GOMAXPROCS:**
- Controls number of OS threads used
- Default: Number of CPU cores
- Can be set: runtime.GOMAXPROCS(n)
- Affects parallelism, not concurrency

**3. Scheduling:**
- Cooperative scheduling (goroutines yield control)
- Preemptive scheduling for blocking operations
- Work-stealing scheduler for load balancing
- Efficient for I/O-bound and CPU-bound workloads

**4. Coordination:**
- Main goroutine must wait for others
- Use sync.WaitGroup or channels to coordinate
- Context package for cancellation and timeouts
- Shared memory access needs synchronization

**Best Practices:**
- Don't create goroutines in loops without limits
- Use worker pools for controlled concurrency
- Always handle goroutine lifecycle (start, wait, cleanup)
- Use context.Context for cancellation
- Avoid goroutine leaks (ensure all goroutines can exit)`,
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
- "Don't communicate by sharing memory; share memory by communicating" (Go proverb)

**Channel Types:**

**1. Unbuffered Channels:**
- Synchronous communication
- Sender blocks until receiver ready
- Receiver blocks until sender ready
- Guarantees delivery (handshake)
- Most common pattern

**2. Buffered Channels:**
- Asynchronous communication
- Sender blocks only when buffer full
- Receiver blocks only when buffer empty
- Buffer size: make(chan Type, size)
- Use for: Producer-consumer patterns, rate limiting

**Channel Operations:**
- Send: ch <- value (blocks if unbuffered and no receiver)
- Receive: value := <-ch (blocks if no sender)
- Close: close(ch) (signals no more values)
- Range: for value := range ch (receives until closed)
- Select: Choose from multiple channels

**Best Practices (2024-2025):**
- Prefer unbuffered channels (simpler, safer)
- Use buffered channels only when needed (performance, decoupling)
- Always close channels when done (prevents goroutine leaks)
- Use select for non-blocking operations
- Don't send on closed channels (panic)
- Receiving from closed channel returns zero value immediately`,
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
				{
					Title: "Worker Pools",
					Content: `Worker pools use a fixed number of goroutines to process jobs from a channel, providing efficient resource usage and controlled concurrency.

**Worker Pool Pattern:**
- Fixed number of worker goroutines
- Jobs sent via channel
- Results collected via channel
- Efficient resource usage
- Controlled concurrency

**Benefits:**
- Limit resource usage
- Control concurrency level
- Efficient for I/O-bound tasks
- Easy to scale

**Implementation Steps:**
1. Create job and result channels
2. Start fixed number of workers
3. Send jobs to channel
4. Workers process jobs
5. Collect results
6. Close channels when done

**Use Cases:**
- Processing files
- API requests
- Database operations
- Image processing
- Any parallelizable task`,
					CodeExamples: `package main

import (
    "fmt"
    "sync"
    "time"
)

type Job struct {
    ID     int
    Data   string
}

type Result struct {
    JobID  int
    Output string
}

func worker(id int, jobs <-chan Job, results chan<- Result, wg *sync.WaitGroup) {
    defer wg.Done()
    for job := range jobs {
        // Simulate work
        time.Sleep(100 * time.Millisecond)
        results <- Result{
            JobID:  job.ID,
            Output: fmt.Sprintf("Processed: %s", job.Data),
        }
    }
}

func main() {
    numWorkers := 3
    numJobs := 10
    
    jobs := make(chan Job, numJobs)
    results := make(chan Result, numJobs)
    var wg sync.WaitGroup
    
    // Start workers
    for w := 1; w <= numWorkers; w++ {
        wg.Add(1)
        go worker(w, jobs, results, &wg)
    }
    
    // Send jobs
    for j := 1; j <= numJobs; j++ {
        jobs <- Job{ID: j, Data: fmt.Sprintf("job-%d", j)}
    }
    close(jobs)
    
    // Wait for workers to finish
    go func() {
        wg.Wait()
        close(results)
    }()
    
    // Collect results
    for result := range results {
        fmt.Println(result.Output)
    }
}`,
				},
				{
					Title: "Fan-In/Fan-Out",
					Content: `Fan-in and fan-out are powerful concurrency patterns for distributing and collecting work across multiple goroutines.

**Fan-Out:**
- Distribute work across multiple goroutines
- Split input channel to multiple workers
- Each worker processes subset of work
- Increases throughput

**Fan-In:**
- Combine results from multiple channels
- Merge multiple channels into one
- Collect results from workers
- Common in worker pool patterns

**Patterns:**

**1. Fan-Out:**
- One input channel
- Multiple worker goroutines
- Each worker reads from same channel
- Distributes load

**2. Fan-In:**
- Multiple input channels
- One output channel
- Use select to merge
- Collects results

**Use Cases:**
- Parallel processing
- Load distribution
- Result aggregation
- Pipeline processing

**Benefits:**
- Better resource utilization
- Increased throughput
- Scalable design
- Clean separation of concerns`,
					CodeExamples: `package main

import (
    "fmt"
    "sync"
)

// Fan-out: Distribute work
func fanOut(input <-chan int, outputs []chan int) {
    defer func() {
        for _, ch := range outputs {
            close(ch)
        }
    }()
    
    for val := range input {
        // Round-robin distribution
        for _, ch := range outputs {
            select {
            case ch <- val:
            default:
            }
        }
    }
}

// Fan-in: Combine results
func fanIn(inputs []<-chan int) <-chan int {
    output := make(chan int)
    var wg sync.WaitGroup
    
    for _, input := range inputs {
        wg.Add(1)
        go func(ch <-chan int) {
            defer wg.Done()
            for val := range ch {
                output <- val
            }
        }(input)
    }
    
    go func() {
        wg.Wait()
        close(output)
    }()
    
    return output
}

// Worker function
func worker(id int, jobs <-chan int, results chan<- int) {
    for job := range jobs {
        results <- job * 2  // Process job
    }
}

func main() {
    // Create channels
    jobs := make(chan int, 10)
    results1 := make(chan int, 5)
    results2 := make(chan int, 5)
    
    // Start workers (fan-out)
    go worker(1, jobs, results1)
    go worker(2, jobs, results2)
    
    // Send jobs
    for i := 1; i <= 10; i++ {
        jobs <- i
    }
    close(jobs)
    
    // Fan-in: Combine results
    merged := fanIn([]<-chan int{results1, results2})
    
    // Collect results
    for result := range merged {
        fmt.Println("Result:", result)
    }
}`,
				},
				{
					Title: "Context Package Deep Dive",
					Content: `The context package provides a powerful way to manage cancellation, timeouts, and request-scoped values in Go programs.

**Context Basics:**
- Carries cancellation signals
- Carries deadlines and timeouts
- Carries request-scoped values
- Immutable (create new context, don't modify)

**Context Creation:**
- context.Background(): Root context
- context.TODO(): Placeholder context
- context.WithCancel(): Cancellable context
- context.WithTimeout(): Context with timeout
- context.WithDeadline(): Context with deadline
- context.WithValue(): Context with values

**Context Propagation:**
- Pass context as first parameter
- Convention: ctx context.Context
- Propagate through call chain
- Check ctx.Done() in loops

**Common Patterns:**
- HTTP request handling
- Database queries with timeout
- Long-running operations
- Graceful shutdown
- Request tracing

**Best Practices:**
- Pass context as first parameter
- Check ctx.Done() in loops
- Use context for cancellation
- Don't store contexts in structs
- Use context.WithValue sparingly`,
					CodeExamples: `package main

import (
    "context"
    "fmt"
    "time"
)

// Context as first parameter
func processRequest(ctx context.Context, data string) error {
    // Check cancellation
    select {
    case <-ctx.Done():
        return ctx.Err()
    default:
    }
    
    // Simulate work
    time.Sleep(1 * time.Second)
    
    // Check again
    if ctx.Err() != nil {
        return ctx.Err()
    }
    
    fmt.Println("Processed:", data)
    return nil
}

// With timeout
func doWithTimeout() {
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()
    
    err := processRequest(ctx, "data")
    if err != nil {
        fmt.Println("Error:", err)
    }
}

// With cancellation
func doWithCancel() {
    ctx, cancel := context.WithCancel(context.Background())
    
    go func() {
        time.Sleep(1 * time.Second)
        cancel()  // Cancel after 1 second
    }()
    
    err := processRequest(ctx, "data")
    if err != nil {
        fmt.Println("Cancelled:", err)
    }
}

// With value
func doWithValue() {
    ctx := context.WithValue(context.Background(), "userID", 123)
    
    userID := ctx.Value("userID")
    fmt.Println("User ID:", userID)
}

func main() {
    doWithTimeout()
    doWithCancel()
    doWithValue()
}`,
				},
				{
					Title: "Channel Patterns",
					Content: `Common channel patterns solve specific concurrency problems elegantly.

**Channel Patterns:**

**1. Pipeline:**
- Chain of stages connected by channels
- Each stage processes and passes to next
- Enables streaming processing
- Example: Read → Process → Write

**2. Generator:**
- Function that returns channel
- Produces values on demand
- Closes channel when done
- Example: Fibonacci generator

**3. Multiplexing:**
- Combine multiple channels into one
- Use select statement
- Fan-in pattern
- Example: Merge multiple result channels

**4. Demultiplexing:**
- Split one channel to multiple
- Distribute work
- Fan-out pattern
- Example: Distribute jobs to workers

**5. Or-Done:**
- Wait for channel or done signal
- Prevents blocking forever
- Common in context usage
- Example: Select with ctx.Done()

**6. Tee:**
- Split channel to multiple outputs
- Broadcast to multiple receivers
- Useful for logging, monitoring
- Example: Process and log simultaneously

**Best Practices:**
- Close channels when done
- Use select for multiple channels
- Handle context cancellation
- Avoid channel leaks`,
					CodeExamples: `package main

import (
    "context"
    "fmt"
    "time"
)

// Generator pattern
func generate(ctx context.Context, max int) <-chan int {
    ch := make(chan int)
    go func() {
        defer close(ch)
        for i := 0; i < max; i++ {
            select {
            case <-ctx.Done():
                return
            case ch <- i:
            }
        }
    }()
    return ch
}

// Pipeline stage
func square(ctx context.Context, input <-chan int) <-chan int {
    output := make(chan int)
    go func() {
        defer close(output)
        for val := range input {
            select {
            case <-ctx.Done():
                return
            case output <- val * val:
            }
        }
    }()
    return output
}

// Or-done pattern
func orDone(ctx context.Context, input <-chan int) <-chan int {
    output := make(chan int)
    go func() {
        defer close(output)
        for {
            select {
            case <-ctx.Done():
                return
            case val, ok := <-input:
                if !ok {
                    return
                }
                select {
                case <-ctx.Done():
                    return
                case output <- val:
                }
            }
        }
    }()
    return output
}

// Tee pattern: Split channel
func tee(ctx context.Context, input <-chan int) (<-chan int, <-chan int) {
    out1 := make(chan int)
    out2 := make(chan int)
    
    go func() {
        defer close(out1)
        defer close(out2)
        for val := range input {
            var out1, out2 = out1, out2
            for i := 0; i < 2; i++ {
                select {
                case <-ctx.Done():
                    return
                case out1 <- val:
                    out1 = nil
                case out2 <- val:
                    out2 = nil
                }
            }
        }
    }()
    
    return out1, out2
}

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()
    
    // Pipeline: generate → square
    numbers := generate(ctx, 5)
    squared := square(ctx, numbers)
    
    for val := range squared {
        fmt.Println(val)
    }
}`,
				},
				{
					Title: "Select Statement",
					Content: `The select statement is crucial for coordinating multiple channel operations and implementing timeouts.

**Select Basics:**
- Waits on multiple channel operations
- Chooses ready case (non-deterministic if multiple ready)
- Blocks until at least one case ready
- Default case makes it non-blocking

**Select Patterns:**

**1. Multiplexing:**
- Wait on multiple channels
- Process first ready
- Common in fan-in

**2. Timeouts:**
- Use time.After() channel
- Prevent blocking forever
- Essential for production code

**3. Non-Blocking:**
- Default case
- Check if operation ready
- Don't block

**4. Cancellation:**
- Select on ctx.Done()
- Cancel long operations
- Graceful shutdown

**Best Practices:**
- Always include timeout or cancellation
- Handle default case appropriately
- Close channels properly
- Use select in loops for continuous operations`,
					CodeExamples: `package main

import (
    "context"
    "fmt"
    "time"
)

// Multiplexing
func multiplex(ch1, ch2 <-chan string) {
    for {
        select {
        case msg := <-ch1:
            fmt.Println("From ch1:", msg)
        case msg := <-ch2:
            fmt.Println("From ch2:", msg)
        }
    }
}

// With timeout
func withTimeout() {
    ch := make(chan string)
    
    go func() {
        time.Sleep(2 * time.Second)
        ch <- "result"
    }()
    
    select {
    case result := <-ch:
        fmt.Println("Got result:", result)
    case <-time.After(1 * time.Second):
        fmt.Println("Timeout!")
    }
}

// Non-blocking
func nonBlocking(ch chan int) {
    select {
    case val := <-ch:
        fmt.Println("Got value:", val)
    default:
        fmt.Println("No value ready")
    }
}

// With context
func withContext(ctx context.Context, ch <-chan int) {
    for {
        select {
        case <-ctx.Done():
            fmt.Println("Cancelled")
            return
        case val := <-ch:
            fmt.Println("Value:", val)
        }
    }
}

func main() {
    // Timeout example
    withTimeout()
    
    // Non-blocking
    ch := make(chan int)
    nonBlocking(ch)
    
    // With context
    ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
    defer cancel()
    withContext(ctx, ch)
}`,
				},
				{
					Title: "Channel Buffering",
					Content: `Channel buffering affects when sends and receives block, impacting performance and behavior.

**Unbuffered Channels:**
- Synchronous communication
- Send blocks until receiver ready
- Receive blocks until sender ready
- Zero capacity: make(chan Type)

**Buffered Channels:**
- Asynchronous communication
- Send blocks only when buffer full
- Receive blocks only when buffer empty
- Capacity > 0: make(chan Type, capacity)

**Buffer Sizing:**
- Small buffer (1-10): Reduces blocking
- Medium buffer (10-100): Balances memory and performance
- Large buffer (100+): For high-throughput scenarios
- Zero buffer: Synchronous, most common

**When to Use Buffered:**
- Producer faster than consumer
- Batch processing
- Reduce blocking
- Performance optimization

**When to Use Unbuffered:**
- Synchronization needed
- Backpressure desired
- Simple communication
- Most common case

**Best Practices:**
- Start with unbuffered
- Add buffer if needed for performance
- Don't use buffer as queue
- Consider backpressure
- Monitor channel capacity`,
					CodeExamples: `package main

import (
    "fmt"
    "time"
)

// Unbuffered: Synchronous
func unbufferedExample() {
    ch := make(chan int)  // Unbuffered
    
    go func() {
        fmt.Println("Sending...")
        ch <- 42  // Blocks until receiver ready
        fmt.Println("Sent")
    }()
    
    time.Sleep(100 * time.Millisecond)
    fmt.Println("Receiving...")
    val := <-ch  // Now sender can proceed
    fmt.Println("Received:", val)
}

// Buffered: Asynchronous
func bufferedExample() {
    ch := make(chan int, 3)  // Buffer size 3
    
    // Can send 3 values without blocking
    ch <- 1
    ch <- 2
    ch <- 3
    
    fmt.Println("Sent 3 values")
    
    // Now buffer is full, next send would block
    // But we can receive
    fmt.Println(<-ch)  // 1
    fmt.Println(<-ch)  // 2
    fmt.Println(<-ch)  // 3
}

// Producer-consumer with buffer
func producerConsumer() {
    ch := make(chan int, 10)  // Buffer allows async
    
    // Producer
    go func() {
        for i := 0; i < 20; i++ {
            ch <- i
            fmt.Printf("Produced: %d\n", i)
        }
        close(ch)
    }()
    
    // Consumer
    for val := range ch {
        time.Sleep(50 * time.Millisecond)  // Slower consumer
        fmt.Printf("Consumed: %d\n", val)
    }
}

func main() {
    unbufferedExample()
    bufferedExample()
    producerConsumer()
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
    jsonStr := "{\"id\":2,\"username\":\"bob\"}"
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
		{
			ID:          44,
			Title:       "File I/O",
			Description: "Master file operations, reading, writing, and working with the filesystem.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "Reading Files",
					Content: `Reading files in Go is straightforward with the os and io packages.

**Reading Methods:**

**1. os.ReadFile():**
- Reads entire file into memory
- Returns []byte and error
- Simple for small files
- Closes file automatically

**2. os.Open() + io.ReadAll():**
- More control over file handle
- Can read in chunks
- Must close file manually
- Better for large files

**3. bufio.Scanner:**
- Line-by-line reading
- Memory efficient
- Handles large files well
- Convenient for text files

**4. io.Copy():**
- Copy from reader to writer
- Efficient for large files
- Streams data
- No memory buffer needed

**Best Practices:**
- Use os.ReadFile() for small files
- Use bufio.Scanner for line-by-line
- Always handle errors
- Close files when done
- Use defer for cleanup`,
					CodeExamples: `package main

import (
    "bufio"
    "fmt"
    "io"
    "os"
)

// Method 1: Read entire file
func readEntireFile(filename string) error {
    data, err := os.ReadFile(filename)
    if err != nil {
        return err
    }
    fmt.Println(string(data))
    return nil
}

// Method 2: Open and read
func readWithOpen(filename string) error {
    file, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    data, err := io.ReadAll(file)
    if err != nil {
        return err
    }
    fmt.Println(string(data))
    return nil
}

// Method 3: Line by line
func readLineByLine(filename string) error {
    file, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        fmt.Println(scanner.Text())
    }
    return scanner.Err()
}

// Method 4: Read in chunks
func readInChunks(filename string) error {
    file, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    buf := make([]byte, 1024)  // 1KB chunks
    for {
        n, err := file.Read(buf)
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }
        fmt.Print(string(buf[:n]))
    }
    return nil
}

func main() {
    readEntireFile("file.txt")
    readWithOpen("file.txt")
    readLineByLine("file.txt")
    readInChunks("file.txt")
}`,
				},
				{
					Title: "Writing Files",
					Content: `Writing files in Go offers multiple approaches depending on your needs.

**Writing Methods:**

**1. os.WriteFile():**
- Writes entire []byte to file
- Creates file if doesn't exist
- Truncates if exists
- Simple for small data

**2. os.Create() + Write():**
- More control over file handle
- Can write incrementally
- Must close file manually
- Better for large writes

**3. bufio.Writer:**
- Buffered writing
- More efficient for multiple writes
- Flush when done
- Reduces system calls

**4. io.Copy():**
- Copy from reader to file
- Efficient for large data
- Streams data
- No memory buffer needed

**File Modes:**
- 0644: Read/write for owner, read for others
- 0755: Execute permissions
- 0600: Read/write for owner only

**Best Practices:**
- Use os.WriteFile() for small data
- Use bufio.Writer for multiple writes
- Set appropriate file permissions
- Always handle errors
- Close files when done`,
					CodeExamples: `package main

import (
    "bufio"
    "fmt"
    "io"
    "os"
)

// Method 1: Write entire file
func writeEntireFile(filename string, data []byte) error {
    return os.WriteFile(filename, data, 0644)
}

// Method 2: Create and write
func writeWithCreate(filename string, data []byte) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    _, err = file.Write(data)
    return err
}

// Method 3: Buffered writing
func writeBuffered(filename string, lines []string) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    writer := bufio.NewWriter(file)
    defer writer.Flush()
    
    for _, line := range lines {
        _, err := writer.WriteString(line + "\n")
        if err != nil {
            return err
        }
    }
    return nil
}

// Method 4: Append to file
func appendToFile(filename string, data []byte) error {
    file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
    if err != nil {
        return err
    }
    defer file.Close()
    
    _, err = file.Write(data)
    return err
}

// Method 5: Copy from reader
func copyToFile(filename string, reader io.Reader) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    _, err = io.Copy(file, reader)
    return err
}

func main() {
    data := []byte("Hello, World!")
    writeEntireFile("output.txt", data)
    
    lines := []string{"Line 1", "Line 2", "Line 3"}
    writeBuffered("lines.txt", lines)
    
    appendToFile("output.txt", []byte("\nAppended line"))
}`,
				},
				{
					Title: "File Operations",
					Content: `Go provides comprehensive file and directory operations through the os package.

**File Operations:**

**1. File Info:**
- os.Stat(): Get file information
- FileInfo interface: Size, Mode, ModTime, IsDir
- Check if file exists
- Get file permissions

**2. File Manipulation:**
- os.Rename(): Rename/move file
- os.Remove(): Delete file
- os.RemoveAll(): Delete directory recursively
- os.Chmod(): Change permissions

**3. Directory Operations:**
- os.Mkdir(): Create directory
- os.MkdirAll(): Create directory tree
- os.ReadDir(): List directory contents
- os.Chdir(): Change directory

**4. Path Operations:**
- filepath.Join(): Join path components
- filepath.Dir(): Get directory
- filepath.Base(): Get filename
- filepath.Ext(): Get extension
- filepath.Clean(): Clean path

**Best Practices:**
- Use filepath.Join() for paths
- Check errors from file operations
- Handle file permissions
- Use os.RemoveAll() carefully
- Check if file exists before operations`,
					CodeExamples: `package main

import (
    "fmt"
    "os"
    "path/filepath"
)

// File info
func fileInfo(filename string) error {
    info, err := os.Stat(filename)
    if err != nil {
        return err
    }
    
    fmt.Printf("Name: %s\n", info.Name())
    fmt.Printf("Size: %d bytes\n", info.Size())
    fmt.Printf("Mode: %v\n", info.Mode())
    fmt.Printf("IsDir: %v\n", info.IsDir())
    fmt.Printf("ModTime: %v\n", info.ModTime())
    
    return nil
}

// Check if file exists
func fileExists(filename string) bool {
    _, err := os.Stat(filename)
    return !os.IsNotExist(err)
}

// Create directory
func createDir(path string) error {
    return os.MkdirAll(path, 0755)
}

// List directory
func listDir(path string) error {
    entries, err := os.ReadDir(path)
    if err != nil {
        return err
    }
    
    for _, entry := range entries {
        info, _ := entry.Info()
        fmt.Printf("%s (%d bytes)\n", entry.Name(), info.Size())
    }
    return nil
}

// Path operations
func pathOperations() {
    path := filepath.Join("dir", "subdir", "file.txt")
    fmt.Println("Full path:", path)
    fmt.Println("Dir:", filepath.Dir(path))
    fmt.Println("Base:", filepath.Base(path))
    fmt.Println("Ext:", filepath.Ext(path))
    fmt.Println("Clean:", filepath.Clean("/a/../b/./c"))
}

// Walk directory tree
func walkDir(root string) error {
    return filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
        if err != nil {
            return err
        }
        fmt.Println(path)
        return nil
    })
}

func main() {
    fileInfo("file.txt")
    fmt.Println("Exists:", fileExists("file.txt"))
    createDir("mydir")
    listDir(".")
    pathOperations()
    walkDir(".")
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          45,
			Title:       "HTTP Client & Server",
			Description: "Build HTTP clients and servers using Go's net/http package.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "HTTP Client",
					Content: `Go's net/http package provides a powerful HTTP client for making requests.

**HTTP Methods:**
- http.Get(): GET request
- http.Post(): POST request
- http.PostForm(): POST with form data
- http.Head(): HEAD request
- http.NewRequest(): Custom request

**Request Options:**
- Set headers
- Add query parameters
- Set request body
- Set timeout
- Add authentication

**Response Handling:**
- Read response body
- Check status code
- Read headers
- Handle redirects
- Close response body

**Best Practices:**
- Always close response body
- Set timeouts
- Handle errors
- Check status codes
- Use context for cancellation`,
					CodeExamples: `package main

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

// Simple GET request
func simpleGet(url string) error {
    resp, err := http.Get(url)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return err
    }
    
    fmt.Println(string(body))
    return nil
}

// GET with headers
func getWithHeaders(url string) error {
    req, err := http.NewRequest("GET", url, nil)
    if err != nil {
        return err
    }
    
    req.Header.Set("Authorization", "Bearer token")
    req.Header.Set("Content-Type", "application/json")
    
    client := &http.Client{Timeout: 10 * time.Second}
    resp, err := client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    body, _ := io.ReadAll(resp.Body)
    fmt.Println(string(body))
    return nil
}

// POST with JSON
func postJSON(url string, data interface{}) error {
    jsonData, err := json.Marshal(data)
    if err != nil {
        return err
    }
    
    req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
    if err != nil {
        return err
    }
    
    req.Header.Set("Content-Type", "application/json")
    
    client := &http.Client{Timeout: 10 * time.Second}
    resp, err := client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    return nil
}

// POST with form data
func postForm(url string, formData map[string]string) error {
    values := make(map[string][]string)
    for k, v := range formData {
        values[k] = []string{v}
    }
    
    resp, err := http.PostForm(url, values)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    return nil
}

// With context
func getWithContext(ctx context.Context, url string) error {
    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return err
    }
    
    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    return nil
}

func main() {
    simpleGet("https://example.com")
    getWithHeaders("https://api.example.com/data")
    postJSON("https://api.example.com/users", map[string]string{"name": "Alice"})
}`,
				},
				{
					Title: "HTTP Server",
					Content: `Building HTTP servers in Go is straightforward with the net/http package.

**Server Basics:**
- http.HandleFunc(): Register handler function
- http.Handle(): Register handler interface
- http.ListenAndServe(): Start server
- http.Server: Advanced server configuration

**Handler Functions:**
- Signature: func(w http.ResponseWriter, r *http.Request)
- Write response with w.Write()
- Set headers with w.Header()
- Set status with w.WriteHeader()

**Handler Interface:**
- Implement ServeHTTP method
- More flexible than functions
- Can store state
- Reusable handlers

**Server Configuration:**
- Set read/write timeouts
- Set max header size
- Configure TLS
- Set address and port

**Best Practices:**
- Use http.HandleFunc() for simple handlers
- Use http.Handle() for reusable handlers
- Set appropriate timeouts
- Handle errors
- Use middleware for common functionality`,
					CodeExamples: `package main

import (
    "fmt"
    "net/http"
    "time"
)

// Handler function
func helloHandler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello, %s!", r.URL.Path[1:])
}

// Handler with method check
func apiHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    fmt.Fprintf(w, "{\"status\": \"ok\"}")
}

// Handler interface
type CounterHandler struct {
    count int
}

func (h *CounterHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    h.count++
    fmt.Fprintf(w, "Count: %d", h.count)
}

// Middleware
func loggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        fmt.Printf("%s %s %v\n", r.Method, r.URL.Path, time.Since(start))
    })
}

// Advanced server
func advancedServer() {
    mux := http.NewServeMux()
    mux.HandleFunc("/", helloHandler)
    mux.Handle("/counter", &CounterHandler{})
    
    server := &http.Server{
        Addr:         ":8080",
        Handler:      loggingMiddleware(mux),
        ReadTimeout:  15 * time.Second,
        WriteTimeout: 15 * time.Second,
        IdleTimeout:  60 * time.Second,
    }
    
    server.ListenAndServe()
}

// Simple server
func simpleServer() {
    http.HandleFunc("/", helloHandler)
    http.HandleFunc("/api", apiHandler)
    http.ListenAndServe(":8080", nil)
}

func main() {
    simpleServer()
    // advancedServer()
}`,
				},
				{
					Title: "HTTP Middleware",
					Content: `Middleware provides a way to add cross-cutting concerns to HTTP handlers.

**Middleware Pattern:**
- Wrap handlers with additional functionality
- Chain multiple middlewares
- Common uses: logging, auth, CORS, rate limiting

**Middleware Types:**

**1. Function Middleware:**
- Takes http.Handler, returns http.Handler
- Simple and flexible
- Easy to chain

**2. Method Middleware:**
- Methods on handler struct
- Can store state
- More object-oriented

**Common Middleware:**

**1. Logging:**
- Log requests and responses
- Track request time
- Log errors

**2. Authentication:**
- Check authentication
- Validate tokens
- Set user context

**3. CORS:**
- Set CORS headers
- Handle preflight requests
- Allow cross-origin requests

**4. Rate Limiting:**
- Limit request rate
- Prevent abuse
- Track by IP or user

**5. Recovery:**
- Recover from panics
- Log panics
- Return error response

**Best Practices:**
- Keep middleware focused
- Chain in correct order
- Handle errors properly
- Don't modify request after reading body
- Use context for values`,
					CodeExamples: `package main

import (
    "context"
    "fmt"
    "net/http"
    "time"
)

// Logging middleware
func loggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        fmt.Printf("%s %s %v\n", r.Method, r.URL.Path, time.Since(start))
    })
}

// Auth middleware
func authMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        token := r.Header.Get("Authorization")
        if token == "" {
            http.Error(w, "Unauthorized", http.StatusUnauthorized)
            return
        }
        // Validate token...
        next.ServeHTTP(w, r)
    })
}

// CORS middleware
func corsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
        
        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }
        
        next.ServeHTTP(w, r)
    })
}

// Recovery middleware
func recoveryMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                fmt.Printf("Panic: %v\n", err)
                http.Error(w, "Internal Server Error", http.StatusInternalServerError)
            }
        }()
        next.ServeHTTP(w, r)
    })
}

// Context middleware
func contextMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        ctx := context.WithValue(r.Context(), "userID", 123)
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

// Chain middlewares
func chainMiddlewares(handler http.Handler, middlewares ...func(http.Handler) http.Handler) http.Handler {
    for i := len(middlewares) - 1; i >= 0; i-- {
        handler = middlewares[i](handler)
    }
    return handler
}

func helloHandler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello!")
}

func main() {
    handler := http.HandlerFunc(helloHandler)
    
    handler = chainMiddlewares(
        handler,
        loggingMiddleware,
        corsMiddleware,
        recoveryMiddleware,
    )
    
    http.Handle("/", handler)
    http.ListenAndServe(":8080", nil)
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          46,
			Title:       "JSON & Encoding",
			Description: "Work with JSON encoding and decoding, and other encoding formats.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "JSON Encoding",
					Content: `Go's encoding/json package provides powerful JSON encoding and decoding capabilities.

**JSON Encoding:**

**1. json.Marshal():**
- Convert Go value to JSON []byte
- Handles structs, maps, slices
- Uses struct tags for field names
- Returns error on failure

**2. json.Encoder:**
- Stream encoding
- Write directly to io.Writer
- More efficient for large data
- Use Encode() method

**Struct Tags:**
- json:"field_name": Custom field name
- json:"-": Ignore field
- json:",omitempty": Omit if zero value
- json:",string": Encode as string

**Best Practices:**
- Use struct tags for field names
- Handle encoding errors
- Use omitempty for optional fields
- Use Encoder for streaming
- Set Content-Type header`,
					CodeExamples: `package main

import (
    "encoding/json"
    "fmt"
    "os"
)

type User struct {
    ID       int    ` + "`" + `json:"id"` + "`" + `
    Username string ` + "`" + `json:"username"` + "`" + `
    Email    string ` + "`" + `json:"email,omitempty"` + "`" + `
    Password string ` + "`" + `json:"-"` + "`" + `
    Active   bool   ` + "`" + `json:"active"` + "`" + `
}

// Marshal to JSON
func marshalExample() {
    user := User{
        ID:       1,
        Username: "alice",
        Email:    "alice@example.com",
        Password: "secret",
        Active:   true,
    }
    
    data, err := json.Marshal(user)
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    
    fmt.Println(string(data))
    // {"id":1,"username":"alice","email":"alice@example.com","active":true}
}

// Marshal with indentation
func marshalIndent() {
    user := User{ID: 1, Username: "bob"}
    data, _ := json.MarshalIndent(user, "", "  ")
    fmt.Println(string(data))
}

// Encoder for streaming
func encoderExample() {
    user := User{ID: 1, Username: "charlie"}
    encoder := json.NewEncoder(os.Stdout)
    encoder.SetIndent("", "  ")
    encoder.Encode(user)
}

// Custom marshaling
func (u User) MarshalJSON() ([]byte, error) {
    type Alias User
    return json.Marshal(&struct {
        *Alias
        Username string ` + "`" + `json:"user_name"` + "`" + `
    }{
        Alias:    (*Alias)(&u),
        Username: u.Username,
    })
}

func main() {
    marshalExample()
    marshalIndent()
    encoderExample()
}`,
				},
				{
					Title: "JSON Decoding",
					Content: `Decoding JSON in Go converts JSON data back into Go values.

**JSON Decoding:**

**1. json.Unmarshal():**
- Convert JSON []byte to Go value
- Handles structs, maps, slices
- Uses struct tags for field mapping
- Returns error on failure

**2. json.Decoder:**
- Stream decoding
- Read directly from io.Reader
- More efficient for large data
- Use Decode() method

**Decoding Behavior:**
- Ignores unknown fields
- Uses field names or tags
- Converts compatible types
- Returns error on type mismatch

**Best Practices:**
- Use struct tags for field mapping
- Handle decoding errors
- Check for required fields
- Use Decoder for streaming
- Validate decoded data`,
					CodeExamples: `package main

import (
    "encoding/json"
    "fmt"
    "strings"
)

type User struct {
    ID       int    ` + "`" + `json:"id"` + "`" + `
    Username string ` + "`" + `json:"username"` + "`" + `
    Email    string ` + "`" + `json:"email,omitempty"` + "`" + `
}

// Unmarshal from JSON
func unmarshalExample() {
    jsonStr := "{\"id\":1,\"username\":\"alice\",\"email\":\"alice@example.com\"}"
    
    var user User
    err := json.Unmarshal([]byte(jsonStr), &user)
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    
    fmt.Printf("User: %+v\n", user)
}

// Decoder for streaming
func decoderExample() {
    jsonStr := "{\"id\":2,\"username\":\"bob\"}\n{\"id\":3,\"username\":\"charlie\"}"
    
    decoder := json.NewDecoder(strings.NewReader(jsonStr))
    for {
        var user User
        if err := decoder.Decode(&user); err != nil {
            break
        }
        fmt.Printf("User: %+v\n", user)
    }
}

// Decode to map
func decodeToMap() {
    jsonStr := "{\"id\":1,\"username\":\"alice\",\"extra\":\"value\"}"
    
    var result map[string]interface{}
    json.Unmarshal([]byte(jsonStr), &result)
    
    fmt.Println(result)
    fmt.Println(result["id"])
    fmt.Println(result["username"])
}

// Custom unmarshaling
func (u *User) UnmarshalJSON(data []byte) error {
    type Alias User
    aux := &struct {
        *Alias
        Username string ` + "`" + `json:"user_name"` + "`" + `
    }{
        Alias: (*Alias)(u),
    }
    
    if err := json.Unmarshal(data, &aux); err != nil {
        return err
    }
    
    u.Username = aux.Username
    return nil
}

func main() {
    unmarshalExample()
    decoderExample()
    decodeToMap()
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          47,
			Title:       "Database Operations",
			Description: "Connect to databases, execute queries, and work with database/sql package.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Database Connection",
					Content: `Go's database/sql package provides a generic interface for SQL databases.

**Database Drivers:**
- Import database driver (blank import)
- Register driver automatically
- Use driver-specific connection string
- Common drivers: mysql, postgres, sqlite

**Connection:**
- sql.Open(): Open database connection
- Returns *sql.DB (connection pool)
- Doesn't actually connect until used
- Manages connection pool automatically

**Connection Pool:**
- SetMaxOpenConns(): Max open connections
- SetMaxIdleConns(): Max idle connections
- SetConnMaxLifetime(): Connection lifetime
- SetConnMaxIdleTime(): Idle timeout

**Best Practices:**
- Import driver with blank import
- Use connection pooling
- Set appropriate pool settings
- Close database when done
- Handle connection errors`,
					CodeExamples: `package main

import (
    "database/sql"
    "fmt"
    "time"
    
    _ "github.com/lib/pq"  // PostgreSQL driver
)

func connectDB() (*sql.DB, error) {
    // Connection string format depends on driver
    dsn := "host=localhost user=postgres password=secret dbname=mydb sslmode=disable"
    
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, err
    }
    
    // Configure connection pool
    db.SetMaxOpenConns(25)
    db.SetMaxIdleConns(5)
    db.SetConnMaxLifetime(5 * time.Minute)
    db.SetConnMaxIdleTime(10 * time.Minute)
    
    // Test connection
    if err := db.Ping(); err != nil {
        return nil, err
    }
    
    return db, nil
}

func main() {
    db, err := connectDB()
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    defer db.Close()
    
    fmt.Println("Connected to database")
}`,
				},
				{
					Title: "Executing Queries",
					Content: `Executing SQL queries in Go involves different methods depending on the operation.

**Query Methods:**

**1. db.Query():**
- SELECT queries
- Returns multiple rows
- Use rows.Next() to iterate
- Close rows when done

**2. db.QueryRow():**
- SELECT query expecting one row
- Returns *sql.Row
- Use Scan() to get values
- Returns sql.ErrNoRows if no result

**3. db.Exec():**
- INSERT, UPDATE, DELETE
- Returns Result with LastInsertId, RowsAffected
- Use for data modification

**4. Prepared Statements:**
- db.Prepare(): Prepare statement
- stmt.Query(), stmt.Exec(): Execute
- More efficient for repeated queries
- Prevents SQL injection

**Best Practices:**
- Always check errors
- Close rows when done
- Use prepared statements for repeated queries
- Handle sql.ErrNoRows
- Use transactions for multiple operations`,
					CodeExamples: `package main

import (
    "database/sql"
    "fmt"
    "log"
    
    _ "github.com/lib/pq"
)

type User struct {
    ID       int
    Username string
    Email    string
}

// Query multiple rows
func queryUsers(db *sql.DB) ([]User, error) {
    rows, err := db.Query("SELECT id, username, email FROM users")
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var users []User
    for rows.Next() {
        var u User
        if err := rows.Scan(&u.ID, &u.Username, &u.Email); err != nil {
            return nil, err
        }
        users = append(users, u)
    }
    
    return users, rows.Err()
}

// Query single row
func queryUser(db *sql.DB, id int) (*User, error) {
    var u User
    err := db.QueryRow("SELECT id, username, email FROM users WHERE id = $1", id).
        Scan(&u.ID, &u.Username, &u.Email)
    
    if err == sql.ErrNoRows {
        return nil, fmt.Errorf("user %d not found", id)
    }
    if err != nil {
        return nil, err
    }
    
    return &u, nil
}

// Execute INSERT
func insertUser(db *sql.DB, username, email string) (int, error) {
    var id int
    err := db.QueryRow(
        "INSERT INTO users (username, email) VALUES ($1, $2) RETURNING id",
        username, email,
    ).Scan(&id)
    
    return id, err
}

// Execute UPDATE
func updateUser(db *sql.DB, id int, email string) error {
    result, err := db.Exec(
        "UPDATE users SET email = $1 WHERE id = $2",
        email, id,
    )
    if err != nil {
        return err
    }
    
    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return err
    }
    
    if rowsAffected == 0 {
        return fmt.Errorf("user %d not found", id)
    }
    
    return nil
}

// Prepared statement
func preparedQuery(db *sql.DB) error {
    stmt, err := db.Prepare("SELECT id, username FROM users WHERE id = $1")
    if err != nil {
        return err
    }
    defer stmt.Close()
    
    var id, username string
    err = stmt.QueryRow(1).Scan(&id, &username)
    if err != nil {
        return err
    }
    
    fmt.Println(id, username)
    return nil
}

func main() {
    db, err := sql.Open("postgres", "connection string")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()
    
    users, _ := queryUsers(db)
    fmt.Println(users)
}`,
				},
				{
					Title: "Transactions",
					Content: `Transactions ensure multiple database operations succeed or fail together.

**Transaction Basics:**
- db.Begin(): Start transaction
- tx.Commit(): Commit transaction
- tx.Rollback(): Rollback transaction
- All operations on tx, not db

**Transaction Methods:**
- tx.Query(), tx.QueryRow(), tx.Exec()
- Same as db methods but within transaction
- All operations atomic
- Use defer tx.Rollback()

**Best Practices:**
- Always defer Rollback()
- Commit only if all operations succeed
- Handle errors properly
- Keep transactions short
- Don't nest transactions

**Use Cases:**
- Multiple related operations
- Data consistency
- Atomic updates
- Complex operations`,
					CodeExamples: `package main

import (
    "database/sql"
    "fmt"
    "log"
    
    _ "github.com/lib/pq"
)

// Simple transaction
func transferMoney(db *sql.DB, fromID, toID int, amount float64) error {
    tx, err := db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()  // Rollback if not committed
    
    // Deduct from sender
    _, err = tx.Exec(
        "UPDATE accounts SET balance = balance - $1 WHERE id = $2",
        amount, fromID,
    )
    if err != nil {
        return err
    }
    
    // Add to receiver
    _, err = tx.Exec(
        "UPDATE accounts SET balance = balance + $1 WHERE id = $2",
        amount, toID,
    )
    if err != nil {
        return err
    }
    
    // Commit transaction
    return tx.Commit()
}

// Transaction with checks
func createUserWithProfile(db *sql.DB, username, email string) (int, error) {
    tx, err := db.Begin()
    if err != nil {
        return 0, err
    }
    defer tx.Rollback()
    
    // Insert user
    var userID int
    err = tx.QueryRow(
        "INSERT INTO users (username, email) VALUES ($1, $2) RETURNING id",
        username, email,
    ).Scan(&userID)
    if err != nil {
        return 0, err
    }
    
    // Insert profile
    _, err = tx.Exec(
        "INSERT INTO profiles (user_id) VALUES ($1)",
        userID,
    )
    if err != nil {
        return 0, err
    }
    
    // Commit
    if err := tx.Commit(); err != nil {
        return 0, err
    }
    
    return userID, nil
}

func main() {
    db, err := sql.Open("postgres", "connection string")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()
    
    err = transferMoney(db, 1, 2, 100.0)
    if err != nil {
        log.Fatal(err)
    }
}`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
