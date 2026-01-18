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

**Official Installation:**
1. Download from https://go.dev/dl/
2. Choose version (latest stable recommended)
3. Install the package for your OS
4. Verify installation: go version

**Platform-Specific Instructions:**

**Linux:**
- Download .tar.gz archive
- Extract to /usr/local: sudo tar -C /usr/local -xzf go1.21.x.linux-amd64.tar.gz
- Add to PATH: export PATH=$PATH:/usr/local/go/bin
- Add to ~/.bashrc or ~/.zshrc for persistence

**macOS:**
- Download .pkg installer (recommended)
- Or use Homebrew: brew install go
- Verify: go version

**Windows:**
- Download .msi installer
- Run installer (adds to PATH automatically)
- Or use Chocolatey: choco install golang
- Verify in PowerShell: go version

**Version Management:**
- Use g (golang.org/x/website/cmd/g) for version switching
- Or use gvm (Go Version Manager)
- Or manually manage multiple versions

**Go Workspace:**

**Legacy GOPATH (Deprecated):**
- GOPATH: Workspace directory (deprecated in Go 1.11+)
- Required all Go code under $GOPATH/src
- Made dependency management difficult
- Still works but not recommended for new projects

**Modern Go Modules (Recommended):**
- Introduced in Go 1.11 (2018)
- Each project is independent module
- No GOPATH requirement
- Better dependency management
- Standard for all new projects

**Environment Variables:**

**GOROOT:**
- Go installation directory
- Usually /usr/local/go (Linux/macOS) or C:\Program Files\Go (Windows)
- Rarely needs to be set manually
- Used by Go toolchain to find standard library

**GOPATH (Legacy):**
- Workspace directory (deprecated)
- Was required for Go < 1.11
- Not needed with Go modules
- Still used by some tools (golangci-lint, etc.)

**GO111MODULE:**
- Controls module behavior (auto/on/off)
- auto: Use modules if go.mod exists (default)
- on: Always use modules
- off: Never use modules (legacy mode)
- Usually don't need to set (auto is fine)

**GOBIN:**
- Where go install puts binaries
- Defaults to $GOPATH/bin (if GOPATH set) or $HOME/go/bin
- Add to PATH to use installed tools

**GOPROXY:**
- Proxy for module downloads
- Default: https://proxy.golang.org,direct
- Can use private proxies: GOPROXY=https://proxy.company.com,direct
- direct: Bypass proxy, fetch directly

**GOSUMDB:**
- Checksum database for module verification
- Default: sum.golang.org
- Can disable: GOSUMDB=off (not recommended)

**GOPRIVATE:**
- Modules that should not be fetched from public proxy
- Example: GOPRIVATE=*.company.com,github.com/company/*
- Used for private repositories

**Creating a New Project:**

**Basic Module:**
go mod init example.com/myproject

**With Git Repository:**
go mod init github.com/username/project

**Module Path Best Practices:**
- Use reverse domain notation
- Match repository location (GitHub, GitLab, etc.)
- Use descriptive names
- Avoid generic names (utils, common, etc.)

**go.mod File:**
- Defines module name and Go version
- Lists direct dependencies
- Can specify minimum Go version
- Can exclude problematic versions

**go.sum File:**
- Contains checksums for dependencies
- Ensures reproducible builds
- Should be committed to version control
- Prevents dependency tampering

**Common Pitfalls:**

**1. Wrong Module Path:**
- Using incorrect path causes import issues
- Should match repository location
- Hard to fix after project starts
- Solution: Set correct path from start

**2. Not Committing go.sum:**
- Missing checksums cause build issues
- Breaks reproducible builds
- Solution: Always commit go.sum

**3. Mixing GOPATH and Modules:**
- Confusion about which to use
- Can cause import errors
- Solution: Use modules for all new projects

**4. Wrong Go Version:**
- Using outdated Go version
- Missing new features
- Security issues
- Solution: Use latest stable version

**5. Not Setting GOPRIVATE:**
- Private modules fetched from public proxy
- Causes authentication errors
- Solution: Set GOPRIVATE for private repos

**Best Practices:**

**1. Use Latest Stable Go:**
- Get latest features
- Security fixes
- Performance improvements
- Check: https://go.dev/doc/devel/release

**2. Use Go Modules:**
- Standard for all projects
- Better dependency management
- Easier to work with
- No GOPATH needed

**3. Set Up IDE:**
- Install Go extension/plugin
- Enable format on save
- Enable goimports on save
- Set up debugging

**4. Configure Git:**
- Add go.sum to repository
- Don't commit vendor/ (unless needed)
- Use .gitignore for build artifacts

**5. Use goimports:**
- Automatically adds/removes imports
- Better than go fmt for imports
- Install: go install golang.org/x/tools/cmd/goimports@latest

**Verification Steps:**

1. Check Go version: go version
2. Check GOROOT: go env GOROOT
3. Check GOPATH: go env GOPATH
4. Check module support: go env GO111MODULE
5. Test compilation: go build ./...
6. Test installation: go install example.com/cmd/tool`,
					CodeExamples: `// Installation verification
// Check Go version
$ go version
go version go1.21.5 linux/amd64

// Check environment
$ go env
GOARCH="amd64"
GOOS="linux"
GOROOT="/usr/local/go"
GOPATH="/home/user/go"
GO111MODULE="auto"
GOPROXY="https://proxy.golang.org,direct"
GOSUMDB="sum.golang.org"

// Create new project
$ mkdir myproject && cd myproject
$ go mod init github.com/username/myproject
go: creating new go.mod: module github.com/username/myproject

// go.mod file created
module github.com/username/myproject

go 1.21

// Add dependency
$ go get github.com/gin-gonic/gin
go: downloading github.com/gin-gonic/gin v1.9.1
go: added github.com/gin-gonic/gin v1.9.1

// go.mod updated
module github.com/username/myproject

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
)

// go.sum created automatically with checksums

// Example: Setting up private repository
// Set GOPRIVATE
$ export GOPRIVATE=github.com/company/*

// Or in .bashrc/.zshrc
export GOPRIVATE=github.com/company/*,*.company.com

// Configure Git for private repos
$ git config --global url."git@github.com:company/".insteadOf "https://github.com/company/"

// Example: Project structure
myproject/
├── go.mod
├── go.sum
├── cmd/
│   └── app/
│       └── main.go
├── internal/
│   ├── handlers/
│   │   └── handlers.go
│   └── services/
│       └── service.go
├── pkg/
│   └── utils/
│       └── utils.go
└── README.md

// Example: Multiple binaries
// cmd/server/main.go
package main
func main() { /* server code */ }

// cmd/cli/main.go
package main
func main() { /* CLI code */ }

// Build specific binary
$ go build ./cmd/server
$ go build ./cmd/cli

// Example: Using goimports
// Before (missing import)
package main
func main() {
    fmt.Println("Hello")
}

// After goimports (auto-added)
package main

import "fmt"

func main() {
    fmt.Println("Hello")
}

// Install goimports
$ go install golang.org/x/tools/cmd/goimports@latest

// Use in editor or manually
$ goimports -w .

// Example: Troubleshooting common issues

// Issue: Cannot find package
// Solution: Run go mod tidy
$ go mod tidy

// Issue: Checksum mismatch
// Solution: Clear module cache and re-download
$ go clean -modcache
$ go mod download

// Issue: Private repo authentication
// Solution: Set up Git credentials or GOPRIVATE
$ export GOPRIVATE=github.com/company/*
$ git config --global url."git@github.com:".insteadOf "https://github.com/"

// Issue: Wrong Go version
// Solution: Update Go or adjust go.mod
// In go.mod: go 1.21 (minimum version)

// Example: IDE setup (VS Code)
// Install Go extension
// Settings:
// - "go.formatTool": "goimports"
// - "go.lintTool": "golangci-lint"
// - "go.useLanguageServer": true

// Example: Docker setup
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o app ./cmd/server

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/app .
CMD ["./app"]`,
				},
				{
					Title: "Go Toolchain",
					Content: `The Go toolchain provides essential commands for building, testing, and managing Go code.

**Essential Commands:**

**1. go build:**
- Compiles packages and dependencies
- Creates executable binary
- Example: go build ./cmd/myapp
- Output: Executable file (or .exe on Windows)

**2. go run:**
- Compiles and runs Go program
- Convenient for quick testing
- Example: go run main.go
- No binary left behind

**3. go test:**
- Runs tests in package
- Example: go test ./...
- Flags: -v (verbose), -cover (coverage), -bench (benchmarks)

**4. go mod:**
- Module management commands
- go mod init: Create new module
- go mod tidy: Add/remove dependencies
- go mod download: Download dependencies
- go mod vendor: Create vendor directory

**5. go fmt:**
- Formats Go code
- Enforces standard formatting
- Example: go fmt ./...
- Usually run automatically by editors

**6. go vet:**
- Reports suspicious code
- Catches common mistakes
- Example: go vet ./...
- Run before committing

**7. go get:**
- Add/update dependencies
- Example: go get github.com/gin-gonic/gin
- Updates go.mod and go.sum

**8. go install:**
- Compile and install binary
- Installs to $GOPATH/bin or $GOBIN
- Example: go install example.com/cmd/tool

**9. go doc:**
- Show documentation
- Example: go doc fmt.Println
- Web server: godoc -http=:6060

**10. go generate:**
- Run code generators
- Uses //go:generate comments
- Example: go generate ./...

**Development Workflow:**
1. go mod init (create project)
2. Write code
3. go fmt (format)
4. go vet (check)
5. go test (test)
6. go build (build)
7. go run (run)`,
					CodeExamples: `// Project structure
myproject/
  go.mod
  go.sum
  cmd/
    myapp/
      main.go
  internal/
    handlers/
      handlers.go
  pkg/
    utils/
      utils.go
  go.mod

// Build
go build ./cmd/myapp

// Run
go run ./cmd/myapp/main.go

// Test
go test ./...
go test -v ./internal/handlers
go test -cover ./...

// Format
go fmt ./...

// Vet
go vet ./...

// Get dependency
go get github.com/gin-gonic/gin@v1.9.1

// Install tool
go install golang.org/x/tools/cmd/goimports@latest`,
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
