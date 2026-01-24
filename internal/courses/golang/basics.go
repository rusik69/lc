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
					Content: `Go (Golang) is an open-source programming language created at Google in 2007 and released in 2009. It was designed to address the challenges of modern software development: multicore processors, networked systems, and massive codebases.

**The Philosophy of Go:**
Go was created by **Robert Griesemer, Rob Pike, and Ken Thompson**. Their goal was to combine the efficiency of a compiled language (like C++) with the readability and usability of a dynamic language (like Python).
- **Pragmatism over Theory**: Go avoids complex features like inheritance and operator overloading in favor of simplicity.
- **Fast Build Times**: One of the primary motivations for Go was to reduce the time developers spent waiting for code to compile.

**Why Go Dominates the Cloud:**
If you look at the "Cloud Native" landscape, Go is everywhere. **Docker, Kubernetes, Prometheus, and Terraform** are all written in Go.
1. **Static Binaries**: Go compiles everything into a single file with no external dependencies (like a JVM or Python runtime). This makes it perfect for Docker containers.
2. **Concurrency First**: Unlike other languages where concurrency was an afterthought, Go was built from day one with **Goroutines** and **Channels**.
3. **Memory Safety**: Go is garbage-collected, which prevents many common memory-related bugs found in C or C++.

**Key Language Features:**
- **Static Typing & Type Inference**: You get the safety of static types without the verbosity, thanks to the ` + "`" + `:=` + "`" + ` operator.
- **Composition over Inheritance**: Go uses Interfaces and Structs to achieve polymorphism, avoiding the "fragile base class" problem.
- **Strict Compiler**: The Go compiler does not allow unused variables or unused imports, keeping codebases clean.`,
					CodeExamples: `// A simple, concurrent web server in Go
package main

import (
    "fmt"
    "net/http"
)

func main() {
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Hello from a Go-powered Cloud!")
    })

    fmt.Println("Server starting on :8080...")
    http.ListenAndServe(":8080", nil)
}`,
				},

				{
					Title: "Installation and Setup",
					Content: `Setting up a productive Go environment involves more than just installing a compiler; it's about understanding the toolchain and the module system that powers modern Go development.

**1. Critical Environment Variables:**
While Go is increasingly "zero-config," these variables control how it behaves:
- **GOPROXY**: Controls where Go downloads modules from. The default ` + "`" + `https://proxy.golang.org` + "`" + ` is fast and secure.
- **GOPRIVATE**: Essential for corporate work. Tells Go *not* to use the public proxy for internal repositories (e.g., ` + "`" + `GOPRIVATE=github.com/mycompany/*` + "`" + `).
- **GOBIN**: Where ` + "`" + `go install` + "`" + ` puts finished binaries. Add this directory to your system PATH to run your Go tools from anywhere.

**2. The Transition to Modules (go.mod):**
Forget the legacy ` + "`" + `GOPATH` + "`" + ` workspace. Modern Go uses **Modules**.
- Every project should start with ` + "`" + `go mod init <name>` + "`" + `.
- **go.sum**: This file is a security record. It contains specific cryptographic checksums of your dependencies, ensuring your builds are reproducible and haven't been tampered with.

**3. Tooling for Success:**
Go's real power is its built-in tooling:
- **gofmt**: Automatically formats your code. Never argue about style again.
- **go vet**: Catches common mistakes that the compiler might allow.
- **go test**: A world-class testing framework built directly into the toolchain.`,
					CodeExamples: `# INITIALIZE A NEW MODULE
mkdir my-app && cd my-app
go mod init github.com/username/my-app

# INSTALLING A TOOL (e.g., golangci-lint)
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# WORKING WITH PRIVATE REPOS
# 1. Set GOPRIVATE
export GOPRIVATE=github.com/my-org/*
# 2. Tell Git to use SSH instead of HTTPS for these repos
git config --global url."git@github.com:".insteadOf "https://github.com/"

# CLEANING UP UNUSED DEPENDENCIES
go mod tidy`,
				},

				{
					Title: "Go Toolchain",
					Content: `Go treats its toolchain as a first-class feature. There's no need for external build systems like Make or Maven for most projects.

**Essential Commands:**

*   **go run main.go**: Compiles and runs your code. Great for development.
*   **go build**: Compiles your code into a binary executable.
*   **go test ./...**: Runs all tests in your project.
*   **go fmt ./...**: Formats your code. RUN THIS OFTEN.
*   **go mod tidy**: Adds missing dependencies and removes unused ones from 'go.mod'.
*   **go get [package]**: Downloads and installs a dependency.

**Project Structure (Standard Layout):**
` + "```" + `
myproject/
├── go.mod          // Dependencies and version
├── go.sum          // Checksums for security
├── main.go         // Entry point
├── internal/       // Private library code (can't be imported by others)
│   └── auth/       // Auth package
├── pkg/            // Public library code (can be imported)
│   └── utils/
└── cmd/            // If you have multiple binaries
    └── server/
        └── main.go
` + "```" + `

**Best Practices:**
1.  **Use Go Modules:** Always run 'go mod init' when starting.
2.  **Format:** Use 'go fmt' or configure your editor to run it on save.
3.  **Check:** Use 'go vet' to catch common bugs before committing.`,
					CodeExamples: `// Common Workflow
// 1. Initialize
// go mod init github.com/myuser/myproject

// 2. Add dependency
// go get github.com/gin-gonic/gin

// 3. Write code...

// 4. Tidy up modules
// go mod tidy

// 5. Test
// go test ./...

// 6. Build
// go build -o myapp`,
				},
				{
					Title: "Go Workspace Setup",
					Content: `Setting up a proper Go workspace is crucial for effective development.

**Modern Workspace (Go Modules):**
- No need for GOPATH
- Each project is independent module
- go.mod defines module
- Dependencies in go.mod and go.sum

**Project Structure:**
- myproject/
  - go.mod              # Module definition
  - go.sum              # Dependency checksums
  - cmd/                # Application entry points
    - app1/
      - main.go
    - app2/
      - main.go
  - internal/           # Private application code
    - handlers/
    - services/
    - models/
  - pkg/                # Public library code
    - utils/
  - api/                # API definitions
  - web/                # Web assets
  - configs/            # Configuration files
  - scripts/            # Build scripts
  - tests/              # Integration tests

**Directory Conventions:**
- **cmd/**: Main applications (each subdirectory is an executable)
- **internal/**: Private code (not importable from outside)
- **pkg/**: Public library code (importable)
- **api/**: API contracts, protobuf definitions
- **web/**: Web frontend assets
- **configs/**: Configuration files
- **scripts/**: Build and deployment scripts

**Module Path:**
- Use reverse domain notation
- Example: github.com/username/project
- Or: company.com/project
- Should match repository location

**Environment Setup:**
- GOROOT: Go installation (usually auto-detected)
- GOPATH: Legacy (not needed with modules)
- GOBIN: Where go install puts binaries
- PATH: Include $GOROOT/bin and $GOBIN

**IDE Setup:**
- VS Code: Install Go extension
- GoLand: JetBrains IDE for Go
- Vim/Neovim: Use vim-go plugin
- Emacs: Use go-mode

**Best Practices:**
- One module per repository
- Use semantic versioning for releases
- Keep dependencies up to date
- Use go mod tidy regularly
- Commit go.mod and go.sum`,
					CodeExamples: `// go.mod
module github.com/username/myproject

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
)

// Project structure example
myproject/
  cmd/
    server/
      main.go          # HTTP server
    cli/
      main.go          # CLI tool
  internal/
    handlers/
      user.go
    services/
      user.go
    models/
      user.go
  pkg/
    utils/
      string.go        # Public utilities
  go.mod
  go.sum
  README.md

// Build commands
go build ./cmd/server    # Build server
go build ./cmd/cli       # Build CLI tool
go build ./...           # Build everything`,
				},
				{
					Title: "Go Modules Deep Dive",
					Content: `Go modules revolutionized dependency management in Go, replacing the legacy GOPATH system.

**Module Basics:**
- Module: Collection of related Go packages
- Defined by go.mod file in root directory
- Module path: Unique identifier (usually repository URL)
- Version: Semantic versioning (v1.2.3)

**go.mod File Structure:**
- module module-path
- go version
- require section lists dependencies
- exclude excludes specific versions
- replace replaces dependencies
- retract retracts versions

**Module Commands:**

**go mod init:**
- Initialize new module
- Creates go.mod file
- Example: go mod init github.com/user/project

**go mod tidy:**
- Add missing dependencies
- Remove unused dependencies
- Update go.mod and go.sum
- Run regularly

**go mod download:**
- Download dependencies
- Populates module cache
- Doesn't modify go.mod

**go mod vendor:**
- Create vendor directory
- Copies dependencies locally
- Enables offline builds
- Use: go build -mod=vendor

**go mod verify:**
- Verify dependencies haven't been modified
- Checks go.sum checksums
- Ensures security

**go mod graph:**
- Print module dependency graph
- Useful for debugging
- Shows all dependencies

**go mod why:**
- Explain why dependency is needed
- Shows dependency chain
- Example: go mod why github.com/pkg/errors

**Version Selection:**
- Minimal version selection (MVS)
- Chooses lowest compatible version
- Ensures reproducible builds
- go get updates to latest compatible

**Indirect Dependencies:**
- Dependencies of direct dependencies
- Marked with // indirect comment
- Automatically managed

**go.sum File:**
- Contains checksums for dependencies
- Ensures integrity
- Must be committed to version control
- Generated automatically

**Best Practices:**
- Commit go.mod and go.sum
- Use semantic versioning
- Pin versions for production
- Run go mod tidy before commits
- Review dependency updates
- Use go mod verify in CI`,
					CodeExamples: `// Initialize module
go mod init github.com/user/myproject

// Add dependency
go get github.com/gin-gonic/gin

// Update dependency
go get -u github.com/gin-gonic/gin

// Update all dependencies
go get -u ./...

// Remove unused dependencies
go mod tidy

// Vendor dependencies
go mod vendor

// Verify dependencies
go mod verify

// Why is dependency needed?
go mod why github.com/pkg/errors

// View dependency graph
go mod graph

// go.mod example
module github.com/user/myproject

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
)

require (
    github.com/bytedance/sonic v1.8.0 // indirect
    github.com/chenzhuoyu/base64x v0.0.0-20221115062448-fe3a3abad311 // indirect
)

// go.sum contains checksums
// Automatically generated, commit to version control`,
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
					Content: `Variables are the storage for data in your program. Go is a statically typed language, meaning every variable has a type that is determined at compile time.

**1. Ways to Declare Variables:**
- **Standard (` + "`" + `var` + "`" + `)**: ` + "`" + `var x int = 10` + "`" + `. Can be used anywhere (package or function level).
- **Short Declaration (` + "`" + `:=` + "`" + `)**: ` + "`" + `y := 20` + "`" + `. Go infers the type automatically. **Restriction**: This can ONLY be used inside function bodies.
- **Multiple Declaration**: ` + "`" + `var x, y, z int` + "`" + ` or ` + "`" + `a, b := 1, "hello"` + "`" + `.

**2. Zero Values:**
Variables declared without an explicit initial value are given their **Zero Value**:
- ` + "`" + `0` + "`" + ` for numeric types.
- ` + "`" + `false` + "`" + ` for booleans.
- ` + "`" + `""` + "`" + ` (empty string) for strings.
- ` + "`" + `nil` + "`" + ` for pointers, slices, maps, channels, and interfaces.

**3. Variable Shadowing:**
Shadowing occurs when you declare a variable with the same name as one in an outer scope. In Go, this is very common with error handling (` + "`" + `err` + "`" + `). Be careful as you can accidentally "hide" a variable you intended to use.`,
					CodeExamples: `# THE SHORT DECLARATION LIMIT
var GlobalVar = 100 // OK
// GlobalVar2 := 200 // ERROR! Cannot use := at package level

func main() {
    # ZERO VALUES IN ACTION
    var i int
    var s string
    var p *int
    fmt.Printf("%v, %q, %v\n", i, s, p) // 0, "", <nil>

    # SHADOWING TRAP
    x := 10
    if true {
        x := 5 // This is a different 'x'!
        fmt.Println("Inner:", x) // 5
    }
    fmt.Println("Outer:", x) // 10
}`,
				},

				{
					Title: "Basic Types",
					Content: `Go provides a rich set of built-in types that are both efficient and safe.

**1. Integers and Floats:**
- **Architecture-Dependent**: ` + "`" + `int` + "`" + ` and ` + "`" + `uint` + "`" + ` are typically 64 bits on modern systems, matching the CPU word size.
- **Explicit Sizes**: Use ` + "`" + `int8` + "`" + `, ` + "`" + `int64` + "`" + `, etc., when you need a fixed size regardless of the architecture.
- **Complex Numbers**: Go has native support for complex numbers (` + "`" + `complex64` + "`" + ` and ` + "`" + `complex128` + "`" + `).

**2. The Rune and Byte:**
- **byte**: An alias for ` + "`" + `uint8` + "`" + `. Used for raw data.
- **rune**: An alias for ` + "`" + `int32` + "`" + `. It represents a Unicode code point. Since Go strings are UTF-8, a single character might take multiple bytes, but a ` + "`" + `rune` + "`" + ` always represents one full character.

**3. Strings:**
- **Immutable**: Once created, a string's content cannot be changed.
- **Structure**: A string is a read-only slice of bytes.
- **Raw Strings**: Created with backticks ` + "`" + ` ` + "`" + `, these preserve newlines and special characters without escaping.`,
					CodeExamples: `// RUNES vs BYTES
s := "Gopher"
fmt.Println(len(s)) // 6 bytes

s2 := "世界" // "World" in Chinese
fmt.Println(len(s2)) // 6 bytes (each char is 3 bytes in UTF-8)
fmt.Println(utf8.RuneCountInString(s2)) // 2 runes

// ARCHITECTURE CHECK
fmt.Println(strconv.IntSize) // 64 or 32

// COMPLEX NUMBERS
c := complex(5, 7)
fmt.Println(real(c), imag(c)) // 5, 7`,
				},

				{
					Title: "Constants",
					Content: `Constants are values that are known at compile time and cannot be changed during execution. They are declared with the ` + "`" + `const` + "`" + ` keyword.

**1. Typed vs. Untyped Constants:**
This is a unique feature of Go.
- **Untyped Constants**: Constants like ` + "`" + `const Pi = 3.14` + "`" + ` are untyped. They have very high precision and can be used in expressions with different types without explicit conversion (as long as the value fits).
- **Typed Constants**: Constants like ` + "`" + `const Pi float64 = 3.14` + "`" + ` are treated exactly like variables of that type.

**2. Constant Groups and Iota:**
Go provides ` + "`" + `iota` + "`" + ` as a way to create incrementing values easily. It resets to 0 whenever a ` + "`" + `const` + "`" + ` keyword appears in the source and increments after each line in a constant block.`,
					CodeExamples: `# THE POWER OF UNTYPED CONSTANTS
const Big = 1 << 100 // A massive number
const Small = Big >> 99
fmt.Println(Small) // 2

# IOTA MAGIC
const (
    _ = iota // Skip 0
    KB = 1 << (10 * iota) // 1 << (10*1) = 1024
    MB = 1 << (10 * iota) // 1 << (10*2) = 1048576
    GB = 1 << (10 * iota) // 1 << (10*3)
)

# ENUMS PATTERN
const (
    StatusRunning = iota
    StatusStopped
    StatusError
)
fmt.Println(StatusRunning, StatusStopped) // 0, 1`,
				},

				{
					Title: "Type Conversion",
					Content: `Type conversion in Go is explicit and strict - there are no implicit conversions between types, even between compatible numeric types. This design choice prevents many subtle bugs that plague languages with implicit conversions. Understanding type conversion is essential for working with Go's type system effectively.

**Why Explicit Conversion?**

Go's designers chose explicit conversion to:
- Prevent accidental type mismatches
- Make code more readable (you see conversions)
- Avoid performance surprises (implicit conversions can hide costs)
- Catch bugs at compile time, not runtime

**Basic Syntax:**

The syntax for type conversion is: Type(value)

This creates a new value of the specified type from the given value. The conversion must be valid - Go will not perform unsafe conversions automatically.

**Numeric Type Conversions:**

**Integer Conversions:**
- All integer types can be converted to each other
- May truncate or extend bits depending on sizes
- No automatic sign extension (be careful with signed/unsigned)
- Overflow behavior is implementation-defined

**Integer to Integer:**
- int8, int16, int32, int64 conversions
- uint8, uint16, uint32, uint64 conversions
- int and uint conversions
- byte (uint8) and rune (int32) conversions

**Integer to Float:**
- All integer types can convert to float32 or float64
- Preserves numeric value
- May lose precision for very large integers

**Float Conversions:**
- float32 ↔ float64 conversions
- Float to integer truncates (doesn't round)
- May lose precision converting float64 → float32
- NaN and Inf values convert but may behave differently

**Common Conversion Patterns:**

**1. Numeric Conversions:**
- int to float64: float64(42)
- float64 to int: int(3.14) → 3 (truncates)
- int32 to int64: int64(x)
- uint to int: int(x) (may overflow if x > MaxInt)

**2. String Conversions:**
- String to []byte: []byte("hello")
- []byte to string: string([]byte{'h', 'e', 'l', 'l', 'o'})
- String to []rune: []rune("hello")
- []rune to string: string([]rune{'h', 'e', 'l', 'l', 'o'})
- Rune to string: string('A')
- Byte to string: string(byte('A'))

**3. Using strconv Package:**
- int to string: strconv.Itoa(42) or strconv.FormatInt(42, 10)
- string to int: strconv.Atoi("42") or strconv.ParseInt("42", 10, 64)
- float to string: strconv.FormatFloat(3.14, 'f', 2, 64)
- string to float: strconv.ParseFloat("3.14", 64)

**Important Conversion Rules:**

**1. No Implicit Conversions:**
- Cannot mix types in expressions: var x int = 42; var y int64 = x (error!)
- Must explicitly convert: var y int64 = int64(x)
- Even compatible types require explicit conversion

**2. Truncation vs Rounding:**
- Float to int truncates: int(3.9) → 3 (not 4)
- Use math.Round() if rounding needed: int(math.Round(3.9)) → 4
- Integer division also truncates: 5 / 2 → 2

**3. Overflow Behavior:**
- Converting larger to smaller type may overflow
- Overflow is implementation-defined (wraps around)
- Use math package constants to check bounds
- Example: int8(256) wraps to 0 (on most systems)

**4. String Conversions:**
- String ↔ []byte: Copies data (not a view)
- String ↔ []rune: Converts UTF-8 encoding
- Invalid UTF-8 in []byte → string: Replaced with replacement char
- Rune/byte to string: Single character string

**Real-World Examples:**

**1. API Response Parsing:**
- JSON numbers come as float64
- Need to convert to int: int(response["id"].(float64))
- Or use json.Number type

**2. String Processing:**
- Convert string to []byte for manipulation
- Process bytes, convert back to string
- More efficient than string concatenation

**3. Numeric Calculations:**
- Mixing int and float requires conversion
- Example: float64(x) * 3.14
- Be explicit about precision needs

**4. Type Assertions vs Conversions:**
- Conversions: Change type of value
- Assertions: Extract concrete type from interface
- Different use cases, similar syntax

**Best Practices:**

**1. Be Explicit:**
- Always show conversions clearly
- Don't rely on implicit behavior
- Makes code more readable

**2. Handle Errors:**
- strconv functions return errors
- Always check errors from Parse functions
- Atoi can fail, use ParseInt for better control

**3. Understand Precision:**
- float32 has ~7 decimal digits precision
- float64 has ~15 decimal digits precision
- Large integers may lose precision as floats

**4. Watch for Overflow:**
- Check bounds before converting
- Use math.MaxInt32, math.MaxInt64 constants
- Consider using larger types if needed

**5. String Conversions:**
- []byte(string) creates a copy (not a view)
- For performance-critical code, consider alternatives
- Use strings.Builder for efficient string building

**Common Pitfalls:**

**1. Assuming Rounding:**
- int(3.9) is 3, not 4
- Use math.Round() if rounding needed
- Integer division also truncates

**2. Overflow Issues:**
- Converting large int64 to int32 may overflow
- Check bounds before conversion
- Use appropriate types from start

**3. String/Byte Confusion:**
- String is immutable, []byte is mutable
- Conversion creates copy (performance consideration)
- Understand UTF-8 encoding implications

**4. Type Assertion vs Conversion:**
- Type assertion: value.(Type) - extracts from interface
- Type conversion: Type(value) - changes type
- Don't confuse the two

**5. Ignoring Errors:**
- strconv.ParseInt returns error
- Always check errors
- Invalid input causes runtime panic if ignored

**Performance Considerations:**

**1. String Conversions:**
- []byte(string) allocates new memory
- For hot paths, avoid repeated conversions
- Consider working with []byte directly

**2. Numeric Conversions:**
- Usually very fast (just bit manipulation)
- Float conversions may be slower
- Profile if performance is critical

**3. strconv Functions:**
- More overhead than direct conversions
- Necessary for string ↔ number
- Use appropriate function for your needs

**Advanced Topics:**

**1. Custom Type Conversions:**
- Define methods for custom conversions
- Example: func (t MyType) String() string
- Implements fmt.Stringer interface

**2. Type Aliases:**
- type MyInt int creates new type
- Requires explicit conversion: int(MyInt(42))
- type MyInt = int creates alias (no conversion needed)

**3. Unsafe Conversions:**
- unsafe package allows unsafe conversions
- Use only when absolutely necessary
- Breaks type safety guarantees

Understanding type conversion in Go helps you write safer, more explicit code. The explicit conversion requirement, while sometimes verbose, prevents many bugs and makes code more maintainable.`,
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
				{
					Title: "Type Assertions",
					Content: `Type assertions allow you to extract the concrete type from an interface value.

**Syntax:**
- value, ok := interface.(Type) - Safe assertion with ok check
- value := interface.(Type) - Unsafe assertion (panics if wrong type)

**When to Use:**
- Extract concrete type from interface{}
- Check if value implements interface
- Type switches for multiple types

**Type Switches:**
- Switch on type of interface value
- More convenient than multiple assertions
- Syntax: switch v := x.(type) { case Type: }

**Common Patterns:**
- Check if error is specific type
- Extract concrete type from interface{}
- Handle different types in generic code`,
					CodeExamples: `package main

import "fmt"

func processValue(v interface{}) {
    // Type assertion with ok check
    if str, ok := v.(string); ok {
        fmt.Printf("String: %s\n", str)
    } else if num, ok := v.(int); ok {
        fmt.Printf("Number: %d\n", num)
    }
    
    // Type switch (preferred for multiple types)
    switch val := v.(type) {
    case string:
        fmt.Printf("String: %s\n", val)
    case int:
        fmt.Printf("Int: %d\n", val)
    case float64:
        fmt.Printf("Float: %f\n", val)
    case bool:
        fmt.Printf("Boolean: %v\n", val)
    default:
        fmt.Printf("Unknown type: %v\n", val)
    }
}

// Error type assertion
func handleError(err error) {
    if validationErr, ok := err.(ValidationError); ok {
        fmt.Printf("Validation error in %s: %s\n", 
            validationErr.Field, validationErr.Message)
    } else if notFoundErr, ok := err.(NotFoundError); ok {
        fmt.Printf("%s not found\n", notFoundErr.Resource)
    } else {
        fmt.Printf("Unknown error: %v\n", err)
    }
}

func main() {
    processValue("hello")
    processValue(42)
    processValue(3.14)
    processValue(true)
}`,
				},
				{
					Title: "Type Switches",
					Content: `Type switches are a powerful way to handle multiple types in a clean, readable manner.

**Syntax:**
switch v := x.(type) {
case Type1:
    // v is Type1
case Type2:
    // v is Type2
default:
    // v is other type
}

**Key Points:**
- v is the value with asserted type in each case
- Cases are type names, not values
- Default case handles unmatched types
- More readable than multiple type assertions

**Use Cases:**
- Handle multiple types from interface{}
- Process different error types
- Generic-like behavior (before Go 1.18)
- JSON unmarshaling with multiple types

**Best Practices:**
- Use type switches for 3+ types
- Use type assertion for 1-2 types
- Always handle default case
- Document expected types`,
					CodeExamples: `package main

import "fmt"

func printType(v interface{}) {
    switch val := v.(type) {
    case int:
        fmt.Printf("Integer: %d\n", val)
    case string:
        fmt.Printf("String: %s\n", val)
    case bool:
        fmt.Printf("Boolean: %v\n", val)
    case []int:
        fmt.Printf("Slice of ints: %v\n", val)
    case map[string]int:
        fmt.Printf("Map: %v\n", val)
    default:
        fmt.Printf("Unknown type: %T\n", val)
    }
}

// Type switch with methods
type Stringer interface {
    String() string
}

func printStringer(v interface{}) {
    switch val := v.(type) {
    case Stringer:
        fmt.Println(val.String())
    case string:
        fmt.Println(val)
    case int:
        fmt.Printf("%d\n", val)
    default:
        fmt.Printf("%v\n", val)
    }
}

func main() {
    printType(42)
    printType("hello")
    printType(true)
    printType([]int{1, 2, 3})
    printType(map[string]int{"a": 1})
}`,
				},
				{
					Title: "Interface{} Usage",
					Content: `The empty interface (interface{}) can hold values of any type, making it useful for generic-like behavior.

**Empty Interface:**
- interface{} is alias for any (Go 1.18+)
- Can hold any type
- Used before generics were available
- Common in standard library

**Common Uses:**
- Accept any type as parameter
- Store heterogeneous collections
- JSON unmarshaling
- Reflection operations
- Function parameters (fmt.Printf, etc.)

**Limitations:**
- Need type assertions to use
- No type safety
- Prefer generics (Go 1.18+) when possible

**Best Practices:**
- Use sparingly (prefer generics)
- Document expected types
- Use type assertions/switches
- Consider generics for new code`,
					CodeExamples: `package main

import "fmt"

// Accept any type
func printAny(v interface{}) {
    fmt.Printf("Value: %v, Type: %T\n", v, v)
}

// Store different types
func storeMixed() {
    var values []interface{}
    values = append(values, 42)
    values = append(values, "hello")
    values = append(values, true)
    values = append(values, 3.14)
    
    for _, v := range values {
        printAny(v)
    }
}

// JSON-like structure
type JSONValue struct {
    Type  string
    Value interface{}
}

func createJSONValue(t string, v interface{}) JSONValue {
    return JSONValue{Type: t, Value: v}
}

func main() {
    printAny(42)
    printAny("hello")
    printAny([]int{1, 2, 3})
    
    storeMixed()
    
    jsonVal := createJSONValue("string", "hello")
    fmt.Println(jsonVal)
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
					Content: "The `if` statement in Go is similar to other languages, but it has some key differences and features that encourage cleaner code.\n\n" +
						"**1. No Parentheses, Mandatory Braces:**\n" +
						"Go removes the need for parentheses around conditions, making the code cleaner. However, braces `{}` are mandatory even for single-line bodies. This prevents the \"hanging else\" bug common in C.\n\n" +
						"**2. The Initialization Expression:**\n" +
						"Go allows you to execute a short statement before the condition. This is most commonly used for error handling or computing a temporary value only needed for the condition.\n" +
						"- **Scope**: Variables declared in the initialization statement are only available inside the `if` block (and any associated `else` blocks).\n\n" +
						"**3. The \"Early Return\" Pattern:**\n" +
						"In Go, it's idiomatic to check for errors or \"bad\" conditions first and return early. This reduces indentation and keeps the \"happy path\" of the code at the left margin.",
					CodeExamples: `# INITIALIZATION PATTERN
if err := computeValue(); err != nil {
    return err // 'err' is only valid here
}

# EARLY RETURN (Idiomatic)
func process(x int) error {
    if x < 0 {
        return errors.New("negative value")
    }
    // Happy path continues here without being nested in an 'else'
    fmt.Println("Processing", x)
    return nil
}`,
				},

				{
					Title: "Switch Statements",
					Content: "The `switch` statement in Go is a cleaner way to write multiple condition checks. Unlike many other C-family languages, Go's `switch` has several unique safety and power features.\n\n" +
						"**1. No Implicit Fallthrough:**\n" +
						"Go only runs the matching `case`. You don't need a `break` at the end of every case. If you *want* the execution to continue into the next case, you must explicitly use the `fallthrough` keyword.\n\n" +
						"**2. Multiple Values in a Single Case:**\n" +
						"You can comma-separate multiple values in one `case` statement (e.g., `case \"Saturday\", \"Sunday\":`).\n\n" +
						"**3. The No-Expression Switch (Switch True):**\n" +
						"If you omit the expression after `switch`, Go assumes it is `switch true`. This is a very powerful alternative to a long `if-else if-else` chain.\n\n" +
						"**4. Variable Initialization:**\n" +
						"Just like `if`, you can initialize a variable that is scoped to the switch: `switch day := getDay(); day { ... }`.",
					CodeExamples: `# SWITCH TRUE PATTERN
switch {
case x < 0:
    fmt.Println("Negative")
case x == 0:
    fmt.Println("Zero")
default:
    fmt.Println("Positive")
}

# MULTIPLE VALUES & FALLTHROUGH
switch day {
case "Saturday", "Sunday":
    fmt.Println("Weekend!")
case "Monday":
    fmt.Println("The grind begins")
    fallthrough // Moves into the next case regardless of match
default:
    fmt.Println("Standard weekday")
}`,
				},

				{
					Title: "For Loops",
					Content: "In Go, there is only one looping construct: the `for` loop. However, its versatile design allows it to behave like a traditional C-style for loop, a `while` loop, or even an infinite loop.\n\n" +
						"**1. The Three Faces of 'for':**\n" +
						"- **The Standard Loop**: `for i := 0; i < 10; i++ { ... }`. Classic initialization, condition, and post-iteration.\n" +
						"- **The While Loop**: `for condition { ... }`. Go omits the `while` keyword; if you only provide a condition, it behaves like a `while` loop.\n" +
						"- **The Infinite Loop**: `for { ... }`. Without any condition, the loop runs forever until it hits a `break` or `return`.\n\n" +
						"**2. Iterating with 'range':**\n" +
						"The `range` keyword is the most common way to iterate over slices, maps, or strings.\n" +
						"- **Slice/Array**: Returns index and value.\n" +
						"- **Map**: Returns key and value.\n" +
						"- **String**: Returns byte index and the `rune` at that position.",
					CodeExamples: `# THE WHILE-STYLE LOOP
i := 1
for i <= 3 {
    fmt.Println(i)
    i = i + 1
}

# INFINITE LOOP WITH BREAK
for {
    if workDone() {
        break
    }
    doWork()
}

# THE POWER OF RANGE
colors := []string{"Red", "Green", "Blue"}
for i, color := range colors {
    fmt.Printf("Index %d is %s\n", i, color)
}

# CLEARING A MAP (Range)
for k := range myMap {
    delete(myMap, k)
}`,
				},

				{
					Title: "Break and Continue",
					Content: `Go provides break and continue statements for controlling loop execution, similar to other languages but with a unique feature: labeled break and continue statements that allow you to control nested loops precisely. Understanding these control flow statements is essential for writing efficient and readable Go code.

**Break Statement:**

**Basic Break:**
The break statement immediately terminates the innermost for, switch, or select statement and transfers control to the statement immediately following it.

**When to Use Break:**

**1. Early Loop Termination:**
- Exit loop when condition is met
- Avoid unnecessary iterations
- Improve performance by early exit
- Common in search operations

**2. Switch Statement:**
- Exit switch block (though fallthrough is default)
- Break is implicit in switch (unlike C)
- Can break out of switch in loop

**3. Select Statement:**
- Not typically used (select doesn't loop)
- But can be used in for-select patterns

**Break Examples:**

**1. Finding First Match:**
- Search for first matching element
- Exit immediately when found
- Avoids processing remaining elements

**2. Input Validation:**
- Exit loop when valid input received
- Common in interactive programs
- Retry until valid input

**3. Error Conditions:**
- Exit when error detected
- Prevent further processing
- Stop on invalid data

**Continue Statement:**

**Basic Continue:**
The continue statement skips the remaining statements in the current loop iteration and immediately starts the next iteration of the innermost for loop.

**When to Use Continue:**

**1. Filtering Data:**
- Skip items that don't meet criteria
- Process only valid items
- Cleaner than nested if statements

**2. Error Handling:**
- Skip items that cause errors
- Continue processing remaining items
- Log errors but don't stop

**3. Early Validation:**
- Skip items that fail validation
- Avoid deep nesting
- Improve readability

**Labeled Break and Continue:**

**What Are Labels?**
Labels are identifiers followed by a colon, placed before a statement. They allow break and continue to target specific loops in nested structures.

**Syntax:**
LabelName: for/switch/select { ... }

**Why Labels Matter:**
- Go doesn't have goto (except for breaking out of nested loops)
- Labels provide controlled way to break/continue outer loops
- More explicit than flags or other workarounds
- Cleaner than deeply nested conditionals

**Labeled Break:**

**Breaking Out of Nested Loops:**
- Break innermost loop: break (default)
- Break specific loop: break LabelName
- Useful when searching in nested structures
- Cleaner than using flags

**Labeled Continue:**

**Continuing Outer Loop:**
- Continue innermost loop: continue (default)
- Continue specific loop: continue LabelName
- Skip to next iteration of labeled loop
- Useful for nested iteration patterns

**Common Patterns:**

**1. Search in Nested Structure:**
- Break outer loop when found
- Avoids flag variables
- More readable code

**2. Skip Outer Iteration:**
- Continue outer loop from inner loop
- Useful for validation patterns
- Cleaner than nested conditionals

**3. Switch in Loop:**
- Break out of switch and loop
- Use labeled break for clarity
- Handle different cases

**Best Practices:**

**1. Use Labels Sparingly:**
- Only when necessary
- Prefer refactoring if overused
- Makes code more complex

**2. Clear Label Names:**
- Use descriptive names
- Indicate what loop you're targeting
- Example: OuterLoop, SearchLoop

**3. Consider Alternatives:**
- Extract to function (return early)
- Use flags if simpler
- Refactor nested loops

**4. Document Complex Logic:**
- Comment why label is needed
- Explain control flow
- Help future maintainers

**Real-World Examples:**

**1. Matrix Search:**
- Search 2D array for value
- Break outer loop when found
- Labeled break is clean solution

**2. Nested Validation:**
- Validate nested data structures
- Continue outer loop on inner failure
- Process remaining items

**3. Early Exit Patterns:**
- Exit nested loops on condition
- Labeled break provides clean exit
- Better than multiple flags

**Common Pitfalls:**

**1. Forgetting Labels:**
- Break only exits innermost loop
- Need label for outer loops
- Common mistake in nested structures

**2. Overusing Labels:**
- Too many labels make code complex
- Consider refactoring instead
- Extract to functions

**3. Confusing Break/Continue:**
- Break: exit loop completely
- Continue: skip to next iteration
- Different behaviors, choose correctly

**4. Label Scope:**
- Labels are block-scoped
- Must be in same function
- Cannot jump across functions

**Performance Considerations:**

**1. Early Exit:**
- Break improves performance
- Avoids unnecessary iterations
- Use when possible

**2. Continue Efficiency:**
- Skips remaining loop body
- More efficient than nested if
- Reduces indentation

**3. Label Overhead:**
- Labels have no runtime cost
- Purely compile-time feature
- No performance penalty

Understanding break and continue, especially labeled versions, gives you precise control over loop execution in Go. Use them judiciously to write clearer, more efficient code.`,
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
				{
					Title: "Error Handling Patterns",
					Content: `Go's error handling is explicit and requires checking errors after each operation that can fail.

**Error Handling Pattern:**
result, err := function()
if err != nil {
    return err  // or handle error
}
// Use result

**Common Patterns:**

**1. Early Return:**
- Return immediately on error
- Keeps code readable
- Reduces nesting

**2. Error Wrapping:**
- Add context to errors
- Use fmt.Errorf with %w verb
- Preserves original error

**3. Error Checking:**
- Always check errors
- Don't ignore with _
- Handle appropriately

**4. Error Propagation:**
- Return errors up the call stack
- Let caller decide how to handle
- Add context at each level

**Best Practices:**
- Check errors immediately
- Return early on error
- Wrap errors with context
- Use errors.Is() and errors.As()
- Don't panic for normal errors`,
					CodeExamples: `package main

import (
    "errors"
    "fmt"
    "os"
)

// Early return pattern
func readConfig(filename string) (string, error) {
    data, err := os.ReadFile(filename)
    if err != nil {
        return "", fmt.Errorf("failed to read config: %w", err)
    }
    
    // Process data only if no error
    return string(data), nil
}

// Error wrapping
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

// Error checking
func safeDivide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

func main() {
    result, err := safeDivide(10, 2)
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    fmt.Println("Result:", result)
}`,
				},
				{
					Title: "Panic and Recover",
					Content: `Panic and recover provide exception-like behavior in Go, but should be used sparingly.

**Panic:**
- Stops normal execution
- Deferred functions still run
- Should be used for programmer errors, not normal errors
- Similar to exceptions in other languages
- Panics propagate up call stack

**Recover:**
- Recovers from panic
- Only useful inside deferred functions
- Returns value passed to panic
- Use sparingly - prefer error returns

**When to Use Panic:**
- Programming errors (nil pointer, index out of bounds)
- Impossible conditions
- Initialization failures
- Not for normal error conditions

**When to Use Recover:**
- Top-level error handling
- Graceful shutdown
- Logging panics
- Converting panics to errors

**Best Practices:**
- Prefer error returns over panic
- Use panic for unrecoverable errors
- Recover at appropriate boundaries
- Log panics before recovering
- Don't recover and continue silently`,
					CodeExamples: `package main

import "fmt"

// Panic example
func riskyFunction() {
    defer func() {
        if r := recover(); r != nil {
            fmt.Println("Recovered from panic:", r)
        }
    }()
    
    panic("something went wrong")
    fmt.Println("This won't execute")
}

// Recover pattern
func safeCall(fn func()) (err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic recovered: %v", r)
        }
    }()
    fn()
    return nil
}

// Converting panic to error
func mustDoSomething() error {
    return safeCall(func() {
        // Code that might panic
        panic("error occurred")
    })
}

func main() {
    riskyFunction()
    fmt.Println("Program continues")
    
    err := mustDoSomething()
    if err != nil {
        fmt.Println("Error:", err)
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
					Content: "Functions are the building blocks of Go programs. They are \"first-class citizens,\" meaning they can be assigned to variables, passed as arguments, and returned from other functions.\n\n" +
						"**1. Parameter and Return Syntax:**\n" +
						"- **Type Grouping**: If consecutive parameters share a type, you only need to write the type once at the end (e.g., `func add(a, b int)`).\n" +
						"- **Multiple Returns**: This is a powerful Go feature. Functions can return any number of values, which is the idiomatic way to return both a result and an `error`.\n\n" +
						"**2. The Scope of Variables:**\n" +
						"Variables declared inside a function are local to that function. Go does not have a global `this` or complex context rules like JavaScript; everything is explicit.",
					CodeExamples: `# MULTIPLE RETURN VALUES
func swap(x, y string) (string, string) {
    return y, x
}

# PARAMETER TYPE GROUPING
func multiply(x, y, z int) int {
    return x * y * z
}

# ASSIGNING TO A VARIABLE
myFunc := func(x int) int { return x * x }
fmt.Println(myFunc(5)) // 25`,
				},

				{
					Title: "Named Return Values",
					Content: "In Go, you can give names to the values your function returns. These names act like variables defined at the top of the function and provide built-in documentation for the caller.\n\n" +
						"**1. Documentation**: Named returns (e.g., `(sum, diff int)`) clarify the *meaning* of the values, which is especially useful for functions returning multiple values of the same type.\n\n" +
						"**2. Naked Returns**: A `return` statement without arguments returns the current values of the named return variables. While this saves typing, it can make long functions very hard to read because the returned values are \"hidden\" at the top.\n\n" +
						"**3. Zero Values**: Named return variables are initialized to their zero value when the function starts.",
					CodeExamples: `# NAMED RETURNS
func split(sum int) (x, y int) {
    x = sum * 4 / 9
    y = sum - x
    return // Returns the current values of x and y
}

# PREFERRED: EXPLICIT RETURNS (for clarity)
func calculate(a, b int) (result int, err error) {
    if b == 0 {
        return 0, errors.New("cannot divide by zero")
    }
    return a / b, nil // Explicit is often better
}`,
				},

				{
					Title: "Variadic Functions",
					Content: "A variadic function is a function that can accept any number of trailing arguments. This is incredibly common in standard library functions like `fmt.Println`.\n\n" +
						"**1. The Ellipsis Syntax**: You define a variadic parameter by placing `...` before the type (e.g., `nums ...int`). Inside the function, this parameter is treated as a **slice** of that type.\n\n" +
						"**2. Passing a Slice**: If you already have a slice of the required type, you can \"spread\" it into a variadic function by using the `...` suffix (e.g., `myFunc(mySlice...)`).\n\n" +
						"**3. Restrictions**: A function can have only **one** variadic parameter, and it must be the **last** parameter in the list.",
					CodeExamples: `# THE VARIADIC PATTERN
func sum(nums ...int) int {
    total := 0
    for _, n := range nums {
        total += n
    }
    return total
}

# SPREADING A SLICE
primes := []int{2, 3, 5, 7}
fmt.Println(sum(primes...))

# MIXED PARAMETERS
func log(prefix string, values ...any) {
    fmt.Print(prefix)
    fmt.Println(values...)
}`,
				},

				{
					Title: "Defer Statement",
					Content: "The `defer` statement is Go's idiomatic way to ensure resources are cleaned up (files closed, mutexes unlocked) regardless of which path a function takes to return.\n\n" +
						"**1. LIFO Execution**: If you have multiple `defer` calls, they are executed in **Last-In, First-Out** order. Think of it like a stack: the last thing you defer is the first thing that runs when the function ends.\n\n" +
						"**2. Immediate Evaluation**: When you call `defer foo(x)`, the value of `x` is evaluated **at that moment**, not when the function actually runs later.\n\n" +
						"**3. Interaction with Named Returns**: A deferred function can read and even modify the values of named return variables before the function actually returns to its caller.",
					CodeExamples: `# ENSURING SYSTEM SAFETY
func writeToFile(filename string) error {
    f, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer f.Close() // Guaranteed to run even if write fails

    return f.Write("Hello Go")
}

# LIFO ORDER
# defer print(1)
# defer print(2)
# Prints: 2 then 1

# MODIFYING RETURN VALUES
func triple() (result int) {
    defer func() { result = result * 3 }()
    return 10 // Returns 30!
}`,
				},

				{
					Title: "Function Types and Closures",
					Content: "Go treats functions as first-class citizens, meaning they can be assigned to variables, passed to other functions, and returned from other functions. This allows for powerful functional programming patterns like **Closures**.\n\n" +
						"**1. Function Types**: You can define a new type for a function signature (e.g., `type MathOp func(int, int) int`). This makes your code more readable when passing around higher-order functions.\n\n" +
						"**2. Closures**: A closure is a function value that references variables from outside its own body. The function may access and assign to these referenced variables; in this sense, the function is \"bound\" to the variables.\n\n" +
						"**3. Stateful Functions**: Closures are the primary way to create functions that maintain internal state without using global variables or complex objects.",
					CodeExamples: `# FUNCTION TYPES (Typedef)
type FilterFunc func(int) bool

func filter(nums []int, f FilterFunc) []int {
    var res []int
    for _, n := range nums {
        if f(n) {
            res = append(res, n)
        }
    }
    return res
}

# THE CLOSURE PATTERN
func makeMultiplier(factor int) func(int) int {
    return func(x int) int {
        return x * factor // 'factor' is captured from the outer scope
    }
}

double := makeMultiplier(2)
fmt.Println(double(10)) // 20`,
				},

				{
					Title: "Method Sets",
					Content: `Method sets determine which methods are available on a type and how it can satisfy interfaces.

**Method Set Rules:**

**Value Receiver Methods:**
- Available on both value and pointer
- T has methods with receiver T
- *T has methods with receiver T and *T

**Pointer Receiver Methods:**
- Available only on pointer
- *T has methods with receiver *T
- T does NOT have methods with receiver *T

**Interface Satisfaction:**
- Type T satisfies interface if T has all methods
- Type *T satisfies interface if *T has all methods
- Value receiver: both T and *T satisfy
- Pointer receiver: only *T satisfies

**Practical Implications:**
- Use pointer receivers when method modifies struct
- Use pointer receivers for large structs (efficiency)
- Use value receivers for small, immutable structs
- Pointer receivers enable nil checks

**Best Practices:**
- Be consistent: use same receiver type for all methods
- Pointer receivers: when modifying or struct is large
- Value receivers: when not modifying and struct is small`,
					CodeExamples: `package main

import "fmt"

type Counter struct {
    value int
}

// Pointer receiver (can modify)
func (c *Counter) Increment() {
    c.value++
}

// Value receiver (cannot modify)
func (c Counter) GetValue() int {
    return c.value
}

// Interface
type Incrementer interface {
    Increment()
    GetValue() int
}

func main() {
    c := Counter{value: 0}
    
    // Both work (Go automatically converts)
    c.Increment()      // Go converts to (&c).Increment()
    (&c).Increment()  // Explicit pointer
    
    fmt.Println(c.GetValue())  // 2
    
    // Interface satisfaction
    var inc Incrementer = &c  // *Counter satisfies
    // var inc2 Incrementer = c  // Error: Counter doesn't satisfy
    
    inc.Increment()
    fmt.Println(inc.GetValue())  // 3
}`,
				},
				{
					Title: "Function Values",
					Content: `Functions are first-class values in Go - they can be assigned, passed, and returned like any other value.

**Function Values:**
- Functions have types
- Can assign to variables
- Can pass as arguments
- Can return from functions
- Can compare with nil

**Function Types:**
- Syntax: func(parameters) returnType
- Can be used as type annotations
- Enable higher-order functions

**Use Cases:**
- Callbacks
- Strategy pattern
- Middleware
- Event handlers
- Higher-order functions

**Function Comparison:**
- Functions can only be compared to nil
- Cannot compare function values directly
- Use function types for type safety`,
					CodeExamples: `package main

import "fmt"

// Function type
type BinaryOp func(int, int) int

// Assign function to variable
var add BinaryOp = func(a, b int) int {
    return a + b
}

// Pass function as argument
func calculate(op BinaryOp, a, b int) int {
    return op(a, b)
}

// Return function
func getOperation(op string) BinaryOp {
    switch op {
    case "add":
        return func(a, b int) int { return a + b }
    case "multiply":
        return func(a, b int) int { return a * b }
    default:
        return nil
    }
}

// Higher-order function
func makeMultiplier(factor int) func(int) int {
    return func(x int) int {
        return x * factor
    }
}

func main() {
    // Use function value
    result := add(3, 4)
    fmt.Println(result)
    
    // Pass function
    result2 := calculate(add, 5, 6)
    fmt.Println(result2)
    
    // Return function
    op := getOperation("multiply")
    if op != nil {
        fmt.Println(op(3, 4))
    }
    
    // Closure
    double := makeMultiplier(2)
    triple := makeMultiplier(3)
    fmt.Println(double(5))  // 10
    fmt.Println(triple(5))  // 15
}`,
				},
				{
					Title: "Higher-Order Functions",
					Content: `Higher-order functions take functions as arguments or return functions, enabling powerful abstractions.

**Common Patterns:**

**1. Map:**
- Apply function to each element
- Transform slice elements
- Return new slice

**2. Filter:**
- Select elements matching condition
- Return subset of slice
- Predicate function determines inclusion

**3. Reduce/Fold:**
- Combine elements into single value
- Accumulate result
- Common for sums, products, etc.

**4. ForEach:**
- Execute function for each element
- Side effects only
- No return value

**Benefits:**
- Code reuse
- Separation of concerns
- Functional programming style
- Composable operations

**Go Approach:**
- No built-in map/filter/reduce
- Easy to implement
- Can use generics (Go 1.18+) for type safety`,
					CodeExamples: `package main

import "fmt"

// Map: Transform each element
func mapInts(slice []int, fn func(int) int) []int {
    result := make([]int, len(slice))
    for i, v := range slice {
        result[i] = fn(v)
    }
    return result
}

// Filter: Select matching elements
func filterInts(slice []int, fn func(int) bool) []int {
    var result []int
    for _, v := range slice {
        if fn(v) {
            result = append(result, v)
        }
    }
    return result
}

// Reduce: Combine elements
func reduceInts(slice []int, fn func(int, int) int, initial int) int {
    result := initial
    for _, v := range slice {
        result = fn(result, v)
    }
    return result
}

// ForEach: Execute for each element
func forEachInts(slice []int, fn func(int)) {
    for _, v := range slice {
        fn(v)
    }
}

func main() {
    numbers := []int{1, 2, 3, 4, 5}
    
    // Map: Square each number
    squared := mapInts(numbers, func(x int) int {
        return x * x
    })
    fmt.Println(squared)  // [1, 4, 9, 16, 25]
    
    // Filter: Even numbers
    evens := filterInts(numbers, func(x int) bool {
        return x%2 == 0
    })
    fmt.Println(evens)  // [2, 4]
    
    // Reduce: Sum
    sum := reduceInts(numbers, func(acc, x int) int {
        return acc + x
    }, 0)
    fmt.Println(sum)  // 15
    
    // ForEach: Print
    forEachInts(numbers, func(x int) {
        fmt.Printf("%d ", x)
    })
}`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
