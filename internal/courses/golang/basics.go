package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          30,
			Title:       "Introduction to Go",
			Description: "Learn the fundamentals of Go: history, installation, and your first program.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "What is Go?",
					Content: `Go (also known as Golang) is an open-source programming language developed by Google in 2007 and released in 2009. It was designed by Robert Griesemer, Rob Pike, and Ken Thompson.

**Design Philosophy:**
- **Simplicity**: Clean, readable syntax without unnecessary complexity
- **Efficiency**: Fast compilation and execution
- **Concurrency**: Built-in support for concurrent programming
- **Safety**: Strong typing and garbage collection
- **Productivity**: Fast development cycle

**Key Features:**
- Statically typed with type inference
- Garbage collected
- Built-in concurrency primitives (goroutines, channels)
- Fast compilation
- Simple dependency management
- Cross-platform compilation
- Rich standard library

**Why Use Go?**
- Excellent for backend services and APIs
- Great for microservices architecture
- Strong performance characteristics
- Growing ecosystem and community
- Used by companies like Google, Docker, Kubernetes, Uber, Dropbox`,
					CodeExamples: `// Hello World in Go
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}`,
				},
				{
					Title: "Installation and Setup",
					Content: `**Installing Go:**

1. Download from https://go.dev/dl/
2. Install the package for your OS
3. Verify installation: go version

**Go Workspace:**
- GOPATH: Legacy workspace directory (deprecated)
- Go Modules: Modern dependency management (Go 1.11+)

**Environment Variables:**
- GOROOT: Go installation directory
- GOPATH: Workspace directory (legacy)
- GO111MODULE: Enable/disable modules (auto/on/off)

**Creating a New Project:**
go mod init example.com/myproject

This creates a go.mod file for dependency management.`,
					CodeExamples: `// go.mod file
module example.com/myproject

go 1.21

require (
    github.com/example/package v1.2.3
)`,
				},
				{
					Title: "Basic Program Structure",
					Content: `**Package Declaration:**
Every Go file starts with a package declaration. The main package is special - it creates an executable.

**Import Statement:**
Import packages you need. Standard library packages don't need a path prefix.

**Main Function:**
The main() function is the entry point for executable programs.

**File Organization:**
- One package per directory
- Multiple files can belong to the same package
- Package name matches directory name (except main)`,
					CodeExamples: `package main

import (
    "fmt"
    "math"
)

func main() {
    fmt.Println("Square root of 16:", math.Sqrt(16))
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          31,
			Title:       "Basic Syntax & Types",
			Description: "Master Go's basic types, variables, constants, and type system.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Variables",
					Content: `**Variable Declaration:**
Go has several ways to declare variables:

1. var name type = value
2. var name = value (type inference)
3. name := value (short declaration, only inside functions)

**Zero Values:**
Variables declared without initialization get zero values:
- Numbers: 0
- Strings: ""
- Booleans: false
- Pointers, slices, maps, channels: nil

**Multiple Variables:**
You can declare multiple variables at once.`,
					CodeExamples: `package main

import "fmt"

func main() {
    // Explicit type
    var age int = 25
    
    // Type inference
    var name = "Alice"
    
    // Short declaration (only in functions)
    city := "New York"
    
    // Multiple variables
    var x, y int = 1, 2
    a, b := 3, 4
    
    // Zero values
    var count int
    var message string
    
    fmt.Println(age, name, city, x, y, a, b, count, message)
}`,
				},
				{
					Title: "Basic Types",
					Content: `**Numeric Types:**
- Integers: int, int8, int16, int32, int64
- Unsigned: uint, uint8, uint16, uint32, uint64, uintptr
- Floating point: float32, float64
- Complex: complex64, complex128
- byte (alias for uint8)
- rune (alias for int32, represents Unicode code point)

**String Type:**
- Immutable sequence of bytes
- UTF-8 encoded by default
- Can use backticks for raw strings

**Boolean Type:**
- true or false
- No implicit conversion from numbers`,
					CodeExamples: `package main

import "fmt"

func main() {
    // Integers
    var i int = 42
    var i8 int8 = 127
    var i64 int64 = 9223372036854775807
    
    // Floating point
    var f32 float32 = 3.14
    var f64 float64 = 3.141592653589793
    
    // String
    var s1 string = "Hello"
    var s2 = "World"
    s3 := ` + "`" + `Raw
    String` + "`" + `
    
    // Boolean
    var b bool = true
    
    // Rune (character)
    var r rune = 'A'
    
    fmt.Println(i, i8, i64, f32, f64, s1, s2, s3, b, r)
}`,
				},
				{
					Title: "Constants",
					Content: `**Constants:**
- Declared with const keyword
- Must be compile-time values
- Can be typed or untyped
- Can use iota for sequential constants

**Iota:**
- Predeclared identifier for constant generation
- Resets to 0 in each const block
- Increments for each constant in the block`,
					CodeExamples: `package main

import "fmt"

const Pi = 3.14159
const (
    StatusOK = 200
    StatusNotFound = 404
    StatusError = 500
)

const (
    Sunday = iota  // 0
    Monday          // 1
    Tuesday         // 2
    Wednesday       // 3
    Thursday        // 4
    Friday          // 5
    Saturday        // 6
)

func main() {
    fmt.Println(Pi, StatusOK, Sunday, Monday)
}`,
				},
				{
					Title: "Type Conversion",
					Content: `**Type Conversion:**
Go requires explicit type conversion. There's no implicit conversion between types.

Syntax: Type(value)

**Common Conversions:**
- Between numeric types
- String to/from byte slices
- String to/from rune slices
- Interface type assertions`,
					CodeExamples: `package main

import (
    "fmt"
    "strconv"
)

func main() {
    // Numeric conversions
    var i int = 42
    var f float64 = float64(i)
    var u uint = uint(f)
    
    // String conversions
    var s string = strconv.Itoa(i)  // int to string
    num, _ := strconv.Atoi("42")     // string to int
    
    // Byte and rune
    var b byte = byte('A')
    var r rune = rune(b)
    
    fmt.Println(i, f, u, s, num, b, r)
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          32,
			Title:       "Control Structures",
			Description: "Learn if/else, switch, and loops in Go.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "If/Else Statements",
					Content: `**If Statement:**
- No parentheses around condition
- Braces are required
- Can have initialization statement before condition

**If-Else:**
- Standard if-else structure
- Can chain multiple conditions

**If with Initialization:**
Common pattern for checking errors or initializing variables.`,
					CodeExamples: `package main

import "fmt"

func main() {
    x := 10
    
    // Simple if
    if x > 5 {
        fmt.Println("x is greater than 5")
    }
    
    // If-else
    if x > 20 {
        fmt.Println("x is large")
    } else {
        fmt.Println("x is small")
    }
    
    // If with initialization
    if y := 15; y > 10 {
        fmt.Println("y is greater than 10")
    }
    
    // Multiple conditions
    if x > 0 && x < 100 {
        fmt.Println("x is in valid range")
    }
}`,
				},
				{
					Title: "Switch Statements",
					Content: `**Switch:**
- More flexible than other languages
- No fall-through by default (unlike C)
- Can switch on values or types
- Can have initialization statement
- Default case is optional

**Switch Variations:**
- Expression switch: switch value { case ... }
- Type switch: switch v := x.(type) { case ... }
- No expression switch: switch { case condition: }`,
					CodeExamples: `package main

import "fmt"

func main() {
    day := "Monday"
    
    // Expression switch
    switch day {
    case "Monday":
        fmt.Println("Start of week")
    case "Friday":
        fmt.Println("End of week")
    default:
        fmt.Println("Midweek")
    }
    
    // Switch with initialization
    switch x := 5; x {
    case 1, 2, 3:
        fmt.Println("Small")
    case 4, 5, 6:
        fmt.Println("Medium")
    }
    
    // No expression switch (like if-else chain)
    x := 10
    switch {
    case x < 0:
        fmt.Println("Negative")
    case x == 0:
        fmt.Println("Zero")
    default:
        fmt.Println("Positive")
    }
}`,
				},
				{
					Title: "For Loops",
					Content: `**For Loop:**
Go has only one loop construct: for. It can be used in three ways:

1. Traditional for loop: for init; condition; post { }
2. While-style: for condition { }
3. Infinite loop: for { }
4. Range loop: for index, value := range collection { }

**Range:**
- Iterates over arrays, slices, strings, maps, channels
- Returns index and value (or key and value for maps)
- Can omit index with _`,
					CodeExamples: `package main

import "fmt"

func main() {
    // Traditional for loop
    for i := 0; i < 5; i++ {
        fmt.Println(i)
    }
    
    // While-style loop
    i := 0
    for i < 5 {
        fmt.Println(i)
        i++
    }
    
    // Infinite loop
    count := 0
    for {
        if count >= 3 {
            break
        }
        fmt.Println(count)
        count++
    }
    
    // Range over slice
    nums := []int{1, 2, 3, 4, 5}
    for index, value := range nums {
        fmt.Printf("Index: %d, Value: %d\n", index, value)
    }
    
    // Range over string
    for i, char := range "Hello" {
        fmt.Printf("Index: %d, Char: %c\n", i, char)
    }
}`,
				},
				{
					Title: "Break and Continue",
					Content: `**Break:**
- Exits the innermost loop
- Can use labels to break outer loops

**Continue:**
- Skips to next iteration
- Can use labels to continue outer loops

**Labels:**
- Used with break and continue
- Allow control flow to outer loops`,
					CodeExamples: `package main

import "fmt"

func main() {
    // Break example
    for i := 0; i < 10; i++ {
        if i == 5 {
            break
        }
        fmt.Println(i)
    }
    
    // Continue example
    for i := 0; i < 10; i++ {
        if i%2 == 0 {
            continue
        }
        fmt.Println(i)
    }
    
    // Labeled break
    OuterLoop:
    for i := 0; i < 3; i++ {
        for j := 0; j < 3; j++ {
            if i == 1 && j == 1 {
                break OuterLoop
            }
            fmt.Printf("i=%d, j=%d\n", i, j)
        }
    }
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          33,
			Title:       "Functions",
			Description: "Master function declaration, multiple returns, variadic functions, and defer.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "Function Declaration",
					Content: `**Function Syntax:**
func functionName(parameters) returnType {
    // body
}

**Key Points:**
- Functions are first-class citizens
- Can return multiple values
- Can have named return values
- Functions can be assigned to variables
- Functions can be passed as arguments`,
					CodeExamples: `package main

import "fmt"

// Simple function
func greet(name string) {
    fmt.Println("Hello,", name)
}

// Function with return value
func add(a, b int) int {
    return a + b
}

// Multiple return values
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}

func main() {
    greet("Alice")
    sum := add(3, 4)
    result, err := divide(10, 2)
    fmt.Println(sum, result, err)
}`,
				},
				{
					Title: "Named Return Values",
					Content: `**Named Returns:**
- Return values can be named
- Act as variables in the function
- Naked return uses named values
- Improves readability for complex functions

**Best Practice:**
Use named returns for documentation, but prefer explicit returns for clarity.`,
					CodeExamples: `package main

import "fmt"

// Named return values
func calculate(x, y int) (sum int, product int) {
    sum = x + y
    product = x * y
    return  // Naked return
}

// Explicit return (preferred)
func calculate2(x, y int) (int, int) {
    sum := x + y
    product := x * y
    return sum, product
}

func main() {
    s, p := calculate(3, 4)
    fmt.Println(s, p)
}`,
				},
				{
					Title: "Variadic Functions",
					Content: `**Variadic Functions:**
- Accept variable number of arguments
- Use ... before type in parameter list
- Arguments become a slice inside function
- Can have at most one variadic parameter
- Must be last parameter`,
					CodeExamples: `package main

import "fmt"

// Variadic function
func sum(numbers ...int) int {
    total := 0
    for _, num := range numbers {
        total += num
    }
    return total
}

// Variadic with other parameters
func printInfo(name string, scores ...int) {
    fmt.Printf("Name: %s\n", name)
    fmt.Printf("Scores: %v\n", scores)
}

func main() {
    fmt.Println(sum(1, 2, 3))
    fmt.Println(sum(1, 2, 3, 4, 5))
    
    printInfo("Alice", 85, 90, 95)
}`,
				},
				{
					Title: "Defer Statement",
					Content: `**Defer:**
- Defers execution until surrounding function returns
- Arguments evaluated immediately, execution deferred
- Multiple defers execute in LIFO order (last in, first out)
- Common use: cleanup, closing files, unlocking mutexes

**Defer with Return Values:**
- Deferred functions can modify named return values
- Useful for error handling and cleanup`,
					CodeExamples: `package main

import "fmt"

func main() {
    defer fmt.Println("World")
    fmt.Println("Hello")
    // Output: Hello\nWorld
    
    // Multiple defers (LIFO)
    defer fmt.Println("First")
    defer fmt.Println("Second")
    defer fmt.Println("Third")
    // Output: Third\nSecond\nFirst
}

func example() (result int) {
    defer func() {
        result++  // Modifies named return
    }()
    return 0  // Returns 1, not 0
}`,
				},
				{
					Title: "Function Types and Closures",
					Content: `**Function Types:**
- Functions have types
- Can assign functions to variables
- Can pass functions as arguments
- Can return functions

**Closures:**
- Functions that reference variables from outer scope
- Can capture and modify outer variables
- Useful for callbacks, iterators, and stateful functions`,
					CodeExamples: `package main

import "fmt"

// Function type
type Operation func(int, int) int

func add(a, b int) int {
    return a + b
}

func multiply(a, b int) int {
    return a * b
}

func apply(op Operation, a, b int) int {
    return op(a, b)
}

// Closure example
func counter() func() int {
    count := 0
    return func() int {
        count++
        return count
    }
}

func main() {
    result := apply(add, 3, 4)
    fmt.Println(result)
    
    c := counter()
    fmt.Println(c())  // 1
    fmt.Println(c())  // 2
    fmt.Println(c())  // 3
}`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
