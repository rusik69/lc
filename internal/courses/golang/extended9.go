package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          1628,
			Title:       "Go Toolchain and Build System",
			Description: "Master the Go toolchain: modules, build tags, code generation, cross-compilation, vendoring, and build optimization.",
			Order:       28,
			Lessons: []problems.Lesson{
				{
					Title: "Go Modules Deep Dive",
					Content: `Go modules are the standard dependency management system since Go 1.16. Understanding their mechanics is essential for managing complex projects.

**go.mod File:**
` + "```" + `
module github.com/myorg/myapp

go 1.22

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
    golang.org/x/sync v0.5.0
)

require (
    // Indirect dependencies (transitive)
    github.com/bytedance/sonic v1.10.2 // indirect
    github.com/gabriel-vasile/mimetype v1.4.3 // indirect
)

Fields:
  module: Module path (import path prefix)
  go:     Minimum Go version (affects language features)
  require: Direct and indirect dependencies
  replace: Override module paths (local dev, forks)
  exclude: Ignore specific versions
  retract: Mark versions of own module as bad

Versioning:
  v1.2.3
  │ │ │
  │ │ └─ Patch: bug fixes (backward compatible)
  │ └─── Minor: new features (backward compatible)
  └───── Major: breaking changes
  
  v0.x.y: No compatibility guarantee
  v1.x.y: Must be backward compatible
  v2+: Must change module path! (v2/v3/...)
    module github.com/myorg/myapp/v2
` + "```" + `

**Module Commands:**
` + "```" + `
go mod init github.com/myorg/myapp  → Create new module
go mod tidy                          → Add missing, remove unused deps
go mod download                      → Download all dependencies
go mod verify                        → Verify checksums
go mod graph                         → Print dependency graph
go mod why -m <module>               → Why is this module needed?
go mod edit -replace=old=new         → Add replace directive
go mod vendor                        → Copy deps to vendor/

go get github.com/lib/pq@v1.10.9   → Add/update dependency
go get github.com/lib/pq@latest     → Get latest version
go get -u ./...                      → Update all direct deps
go get -u -t ./...                   → Including test deps

go.sum file:
  Checksum database for ALL dependencies
  h1:<hash> → SHA-256 of module zip
  Ensures reproducible builds
  COMMIT to version control!
  
  go.sum has MORE entries than go.mod:
  It includes checksums for all transitive dependencies

GOPROXY (module proxy):
  Default: https://proxy.golang.org,direct
  Caches modules, ensures availability
  Also: Athens, JFrog Artifactory (private proxy)
  
  GONOSUMDB/GONOSUMCHECK: Skip checksum for private modules
  GOPRIVATE=github.com/myorg/*  → Don't use proxy for private repos
  GOFLAGS=-mod=vendor            → Always use vendor directory
` + "```" + `

**Minimum Version Selection (MVS):**
` + "```" + `
Go uses MVS (not SAT solving like npm/pip):

Given:
  A requires B ≥ 1.2 and C ≥ 1.4
  B requires C ≥ 1.3 and D ≥ 1.1
  C requires D ≥ 1.2

MVS picks: B=1.2, C=1.4, D=1.2

Why minimum (not maximum/latest)?
  - Reproducible: same go.mod → same versions
  - No dependency hell: no conflicting constraints
  - Fast: no NP-hard constraint solving
  - Predictable: upgrade only when explicitly requested
  
  npm/pip: may pick latest, causing unexpected breakage
  Go: picks minimum that satisfies all constraints

Upgrading:
  go get github.com/lib/pq@v1.11.0
  → Only updates lib/pq (and its NEW requirements)
  → Does NOT update other deps

go get -u ./...
  → Updates ALL dependencies to latest minor/patch
  → May introduce breaking changes from transitive deps!
  → Use carefully, test thoroughly
` + "```" + `

**Workspaces (go.work, Go 1.18+):**
` + "```" + `
For developing multiple modules simultaneously:

go work init ./myapp ./mylib

go.work:
  go 1.22
  use (
      ./myapp
      ./mylib
  )

Now: changes to mylib are immediately available in myapp
Without workspace: need to push mylib, update go.mod in myapp

Commands:
  go work init ./mod1 ./mod2  → Create workspace
  go work use ./mod3          → Add module to workspace
  go work sync                → Sync go.mod files

Rules:
  - go.work should NOT be committed (development only)
  - Each module still works independently
  - CI/CD doesn't use go.work
  - Replaces the old "replace directive" workflow for local dev
` + "```" + ``,
					CodeExamples: `// Module management patterns and dependency injection
package main

import (
    "fmt"
    "strings"
)

// Dependency injection patterns for modular code

// Interface-based dependency injection
type Logger interface {
    Info(msg string, args ...any)
    Error(msg string, args ...any)
}

type Database interface {
    Query(query string, args ...any) ([]map[string]any, error)
    Exec(query string, args ...any) error
}

type Cache interface {
    Get(key string) (any, bool)
    Set(key string, value any)
}

// Production implementations
type stdLogger struct{}

func (l *stdLogger) Info(msg string, args ...any) {
    fmt.Printf("[INFO] "+msg+"\n", args...)
}

func (l *stdLogger) Error(msg string, args ...any) {
    fmt.Printf("[ERROR] "+msg+"\n", args...)
}

type memDB struct {
    data []map[string]any
}

func (db *memDB) Query(query string, args ...any) ([]map[string]any, error) {
    return db.data, nil
}

func (db *memDB) Exec(query string, args ...any) error {
    fmt.Printf("  Executed: %s\n", query)
    return nil
}

type memCache struct {
    store map[string]any
}

func (c *memCache) Get(key string) (any, bool) {
    v, ok := c.store[key]
    return v, ok
}

func (c *memCache) Set(key string, value any) {
    c.store[key] = value
}

// Service using dependency injection
type UserService struct {
    db     Database
    cache  Cache
    logger Logger
}

func NewUserService(db Database, cache Cache, logger Logger) *UserService {
    return &UserService{db: db, cache: cache, logger: logger}
}

func (s *UserService) GetUser(id string) (map[string]any, error) {
    // Try cache
    if cached, ok := s.cache.Get("user:" + id); ok {
        s.logger.Info("cache hit for user %s", id)
        return cached.(map[string]any), nil
    }
    
    s.logger.Info("cache miss for user %s, querying DB", id)
    
    // Query database
    results, err := s.db.Query("SELECT * FROM users WHERE id = ?", id)
    if err != nil {
        s.logger.Error("failed to query user %s: %v", id, err)
        return nil, err
    }
    
    if len(results) == 0 {
        return nil, fmt.Errorf("user %s not found", id)
    }
    
    user := results[0]
    s.cache.Set("user:"+id, user)
    return user, nil
}

// Build info embedding
type BuildInfo struct {
    Version   string
    Commit    string
    BuildTime string
    GoVersion string
}

// These would be set via ldflags:
// go build -ldflags "-X main.version=1.0.0 -X main.commit=abc123"
var (
    version   = "dev"
    commit    = "unknown"
    buildTime = "unknown"
)

func GetBuildInfo() BuildInfo {
    return BuildInfo{
        Version:   version,
        Commit:    commit,
        BuildTime: buildTime,
        GoVersion: "go1.22",
    }
}

// Module path validation
func ValidateModulePath(path string) error {
    if path == "" {
        return fmt.Errorf("module path cannot be empty")
    }
    
    parts := strings.Split(path, "/")
    if len(parts) < 2 {
        return fmt.Errorf("module path should have at least 2 parts: %s", path)
    }
    
    // Check for valid domain
    domain := parts[0]
    if !strings.Contains(domain, ".") {
        return fmt.Errorf("first element must be a domain: %s", domain)
    }
    
    // Check for uppercase (Go convention: lowercase only)
    if path != strings.ToLower(path) {
        return fmt.Errorf("module path must be lowercase: %s", path)
    }
    
    return nil
}

// Semantic version comparison
type SemVer struct {
    Major, Minor, Patch int
    Pre                 string
}

func ParseSemVer(s string) (SemVer, error) {
    s = strings.TrimPrefix(s, "v")
    
    var v SemVer
    preParts := strings.SplitN(s, "-", 2)
    if len(preParts) == 2 {
        v.Pre = preParts[1]
        s = preParts[0]
    }
    
    _, err := fmt.Sscanf(s, "%d.%d.%d", &v.Major, &v.Minor, &v.Patch)
    if err != nil {
        return v, fmt.Errorf("invalid semver: %s", s)
    }
    return v, nil
}

func (v SemVer) String() string {
    s := fmt.Sprintf("v%d.%d.%d", v.Major, v.Minor, v.Patch)
    if v.Pre != "" {
        s += "-" + v.Pre
    }
    return s
}

func (v SemVer) Compatible(other SemVer) bool {
    if v.Major == 0 || other.Major == 0 {
        return v.Major == other.Major && v.Minor == other.Minor
    }
    return v.Major == other.Major
}

func main() {
    // Dependency injection
    fmt.Println("=== Dependency Injection ===")
    
    db := &memDB{
        data: []map[string]any{
            {"id": "1", "name": "Alice", "email": "alice@example.com"},
        },
    }
    cache := &memCache{store: make(map[string]any)}
    logger := &stdLogger{}
    
    svc := NewUserService(db, cache, logger)
    
    // First call: cache miss
    user, err := svc.GetUser("1")
    if err != nil {
        fmt.Printf("  Error: %v\n", err)
    } else {
        fmt.Printf("  User: %v\n", user)
    }
    
    // Second call: cache hit
    user, err = svc.GetUser("1")
    if err != nil {
        fmt.Printf("  Error: %v\n", err)
    } else {
        fmt.Printf("  User: %v\n", user)
    }
    
    // Build info
    fmt.Println("\n=== Build Info ===")
    info := GetBuildInfo()
    fmt.Printf("  Version:   %s\n", info.Version)
    fmt.Printf("  Commit:    %s\n", info.Commit)
    fmt.Printf("  BuildTime: %s\n", info.BuildTime)
    fmt.Printf("  Go:        %s\n", info.GoVersion)
    
    // Module path validation
    fmt.Println("\n=== Module Path Validation ===")
    paths := []string{
        "github.com/myorg/myapp",
        "github.com/myorg/MyApp",
        "myapp",
        "golang.org/x/sync",
    }
    for _, p := range paths {
        err := ValidateModulePath(p)
        if err != nil {
            fmt.Printf("  ✗ %s: %v\n", p, err)
        } else {
            fmt.Printf("  ✓ %s\n", p)
        }
    }
    
    // Semantic versioning
    fmt.Println("\n=== Semantic Versioning ===")
    versions := []string{"v1.2.3", "v1.3.0", "v2.0.0", "v0.1.0", "v1.2.3-beta.1"}
    for _, vs := range versions {
        v, err := ParseSemVer(vs)
        if err != nil {
            fmt.Printf("  Error: %v\n", err)
            continue
        }
        fmt.Printf("  %s: major=%d minor=%d patch=%d pre=%q\n",
            v, v.Major, v.Minor, v.Patch, v.Pre)
    }
    
    // Compatibility check
    v1, _ := ParseSemVer("v1.2.3")
    v2, _ := ParseSemVer("v1.5.0")
    v3, _ := ParseSemVer("v2.0.0")
    fmt.Printf("\n  %s compatible with %s: %v\n", v1, v2, v1.Compatible(v2))
    fmt.Printf("  %s compatible with %s: %v\n", v1, v3, v1.Compatible(v3))
}`,
				},
				{
					Title: "Build Tags and Cross-Compilation",
					Content: `Build tags control which files are compiled. Cross-compilation builds for different OS/architecture combinations. Both are essential for writing portable Go code.

**Build Tags (Build Constraints):**
` + "```" + `
Go 1.17+ syntax (//go:build):
  //go:build linux
  //go:build !windows
  //go:build linux && amd64
  //go:build linux || darwin
  //go:build integration
  //go:build !integration

Old syntax (still recognized):
  // +build linux

Build tag placement:
  Must be FIRST line (before package declaration)
  Blank line between tag and package

Common built-in tags:
  OS:       linux, darwin, windows, freebsd, ...
  Arch:     amd64, arm64, 386, wasm, ...
  Compiler: gc, gccgo
  CGO:      cgo (enabled when CGO_ENABLED=1)
  Go version: go1.22 (Go 1.22+)

File naming convention (implicit build tags):
  file_linux.go          → only compiled on Linux
  file_windows_amd64.go  → only on Windows AMD64
  file_test.go           → only during testing
  
  Pattern: *_GOOS.go, *_GOARCH.go, *_GOOS_GOARCH.go

Custom build tags:
  //go:build integration
  
  Run: go test -tags=integration ./...
  
  Use for:
    - Integration tests vs unit tests
    - Feature flags (enable experimental code)
    - Database backends (postgres vs sqlite)
    - Log verbosity levels
` + "```" + `

**Cross-Compilation:**
` + "```" + `
Go can cross-compile to any supported platform:

  GOOS=linux GOARCH=amd64 go build -o myapp-linux
  GOOS=darwin GOARCH=arm64 go build -o myapp-darwin
  GOOS=windows GOARCH=amd64 go build -o myapp.exe

Supported combinations (go tool dist list):
  linux/amd64, linux/arm64, linux/386
  darwin/amd64, darwin/arm64
  windows/amd64, windows/386
  freebsd/amd64
  js/wasm (WebAssembly)
  wasip1/wasm (WASI)
  ... 30+ combinations

CGO and cross-compilation:
  CGO_ENABLED=0: Pure Go (easiest to cross-compile)
  CGO_ENABLED=1: Needs C compiler for target platform
  
  Rule: Disable CGO for cross-compilation unless needed
  GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build
  
  Libraries that need CGO:
    - mattn/go-sqlite3 (use modernc.org/sqlite instead)
    - Some crypto operations (Go has pure Go fallbacks)

Static binary (no dynamic linking):
  CGO_ENABLED=0 go build -ldflags="-w -s" -o myapp
  
  -w: Strip DWARF debug info
  -s: Strip symbol table
  Result: smaller binary, harder to debug
` + "```" + `

**Build Optimization:**
` + "```" + `
ldflags (linker flags):
  go build -ldflags="-X main.version=1.0.0 -X main.commit=$(git rev-parse HEAD)"
  
  Sets string variables at link time (no code change!)
  Use for: version, commit, build time

Trimpath:
  go build -trimpath
  Removes file system paths from binary
  Important for reproducible builds
  Security: doesn't leak local paths in stack traces

Build cache:
  go build caches compiled packages
  go clean -cache → clear build cache
  GOCACHE=... → custom cache directory
  
  CI optimization: cache $GOPATH/pkg/mod and $GOCACHE

Profile-guided optimization (PGO, Go 1.21+):
  1. Run with CPU profile: go test -cpuprofile=default.pgo
  2. Build with PGO: go build -pgo=default.pgo
  3. ~2-7% performance improvement
  
  Or: place default.pgo in module root (auto-detected)

Binary size reduction:
  1. -ldflags="-w -s"          → ~20-30% smaller
  2. CGO_ENABLED=0              → Removes libc dependency
  3. UPX compression            → ~60-70% smaller (slower startup)
     upx --best myapp
  4. -trimpath                  → Removes paths
  5. Use -gcflags="-B"          → Disable bounds checking (risky!)
` + "```" + ``,
					CodeExamples: `// Build system patterns: embedding, code generation, Makefile
package main

import (
    "fmt"
    "os"
    "runtime"
    "runtime/debug"
    "strings"
)

// Build info (set via ldflags)
var (
    appVersion = "dev"
    appCommit  = "unknown"
    appDate    = "unknown"
)

// Read build info from runtime (Go 1.18+)
func readBuildInfo() map[string]string {
    info := make(map[string]string)
    
    bi, ok := debug.ReadBuildInfo()
    if !ok {
        return info
    }
    
    info["go_version"] = bi.GoVersion
    info["module"] = bi.Path
    
    for _, setting := range bi.Settings {
        switch setting.Key {
        case "vcs.revision":
            info["vcs_revision"] = setting.Value
        case "vcs.time":
            info["vcs_time"] = setting.Value
        case "vcs.modified":
            info["vcs_modified"] = setting.Value
        case "GOOS":
            info["goos"] = setting.Value
        case "GOARCH":
            info["goarch"] = setting.Value
        case "CGO_ENABLED":
            info["cgo"] = setting.Value
        }
    }
    
    return info
}

// Platform detection
func platformInfo() {
    fmt.Println("=== Platform Info ===")
    fmt.Printf("  OS:          %s\n", runtime.GOOS)
    fmt.Printf("  Arch:        %s\n", runtime.GOARCH)
    fmt.Printf("  CPUs:        %d\n", runtime.NumCPU())
    fmt.Printf("  GOMAXPROCS:  %d\n", runtime.GOMAXPROCS(0))
    fmt.Printf("  Go version:  %s\n", runtime.Version())
    fmt.Printf("  Compiler:    %s\n", runtime.Compiler)
}

// Build configuration generation
type BuildConfig struct {
    AppName    string
    Version    string
    Platforms  []Platform
    LDFlags    []string
    BuildTags  []string
    CGOEnabled bool
}

type Platform struct {
    OS   string
    Arch string
}

func (p Platform) String() string {
    return p.OS + "/" + p.Arch
}

func (p Platform) BinaryName(appName string) string {
    name := fmt.Sprintf("%s-%s-%s", appName, p.OS, p.Arch)
    if p.OS == "windows" {
        name += ".exe"
    }
    return name
}

func DefaultBuildConfig(appName, version string) BuildConfig {
    return BuildConfig{
        AppName: appName,
        Version: version,
        Platforms: []Platform{
            {"linux", "amd64"},
            {"linux", "arm64"},
            {"darwin", "amd64"},
            {"darwin", "arm64"},
            {"windows", "amd64"},
        },
        LDFlags: []string{
            fmt.Sprintf("-X main.appVersion=%s", version),
            "-w", "-s",
        },
        CGOEnabled: false,
    }
}

func (c BuildConfig) BuildCommand(p Platform) string {
    parts := []string{
        fmt.Sprintf("GOOS=%s", p.OS),
        fmt.Sprintf("GOARCH=%s", p.Arch),
        fmt.Sprintf("CGO_ENABLED=%d", boolToInt(c.CGOEnabled)),
        "go", "build",
        "-trimpath",
    }
    
    if len(c.LDFlags) > 0 {
        parts = append(parts, fmt.Sprintf("-ldflags=%q", strings.Join(c.LDFlags, " ")))
    }
    
    if len(c.BuildTags) > 0 {
        parts = append(parts, fmt.Sprintf("-tags=%s", strings.Join(c.BuildTags, ",")))
    }
    
    parts = append(parts, "-o", fmt.Sprintf("dist/%s", p.BinaryName(c.AppName)))
    parts = append(parts, ".")
    
    return strings.Join(parts, " ")
}

func boolToInt(b bool) int {
    if b {
        return 1
    }
    return 0
}

// Makefile generator
func generateMakefile(config BuildConfig) string {
    var b strings.Builder
    
    b.WriteString(fmt.Sprintf("APP_NAME := %s\n", config.AppName))
    b.WriteString(fmt.Sprintf("VERSION := %s\n", config.Version))
    b.WriteString("COMMIT := $(shell git rev-parse --short HEAD)\n")
    b.WriteString("DATE := $(shell date -u '+%Y-%m-%dT%H:%M:%SZ')\n")
    b.WriteString(fmt.Sprintf("LDFLAGS := %s\n", strings.Join(config.LDFlags, " ")))
    b.WriteString("\n")
    
    b.WriteString(".PHONY: build test clean lint all\n\n")
    
    // Default target
    b.WriteString("all: lint test build\n\n")
    
    // Build
    b.WriteString("build:\n")
    b.WriteString("\t@echo \"Building $(APP_NAME) $(VERSION)...\"\n")
    b.WriteString("\tgo build -trimpath -ldflags=\"$(LDFLAGS)\" -o bin/$(APP_NAME) .\n\n")
    
    // Test
    b.WriteString("test:\n")
    b.WriteString("\tgo test -race -coverprofile=coverage.out ./...\n")
    b.WriteString("\tgo tool cover -func=coverage.out\n\n")
    
    // Lint
    b.WriteString("lint:\n")
    b.WriteString("\tgolangci-lint run ./...\n\n")
    
    // Cross compile
    b.WriteString("release:\n")
    b.WriteString("\t@mkdir -p dist\n")
    for _, p := range config.Platforms {
        b.WriteString(fmt.Sprintf("\t%s\n", config.BuildCommand(p)))
    }
    b.WriteString("\n")
    
    // Clean
    b.WriteString("clean:\n")
    b.WriteString("\trm -rf bin/ dist/ coverage.out\n")
    
    return b.String()
}

// Dockerfile generator
func generateDockerfile(config BuildConfig) string {
    return fmt.Sprintf("# Build stage\n"+
        "FROM golang:1.22-alpine AS builder\n\n"+
        "WORKDIR /app\n"+
        "COPY go.mod go.sum ./\n"+
        "RUN go mod download\n\n"+
        "COPY . .\n"+
        "RUN CGO_ENABLED=0 go build -trimpath -ldflags=\"%s\" -o /app/bin/%s .\n\n"+
        "# Runtime stage\n"+
        "FROM alpine:3.19\n\n"+
        "RUN apk --no-cache add ca-certificates\n"+
        "COPY --from=builder /app/bin/%s /usr/local/bin/\n\n"+
        "USER nobody:nobody\n"+
        "ENTRYPOINT [\"%s\"]\n",
        strings.Join(config.LDFlags, " "), config.AppName, config.AppName, config.AppName)
}

func main() {
    platformInfo()
    
    // Build info from ldflags
    fmt.Println("\n=== Build Info (ldflags) ===")
    fmt.Printf("  Version: %s\n", appVersion)
    fmt.Printf("  Commit:  %s\n", appCommit)
    fmt.Printf("  Date:    %s\n", appDate)
    
    // Build info from runtime
    fmt.Println("\n=== Build Info (runtime) ===")
    bi := readBuildInfo()
    for k, v := range bi {
        fmt.Printf("  %s: %s\n", k, v)
    }
    
    // Generate build configuration
    fmt.Println("\n=== Cross-Compilation Commands ===")
    config := DefaultBuildConfig("myapp", "1.0.0")
    for _, p := range config.Platforms {
        fmt.Printf("  %s:\n    %s\n\n", p, config.BuildCommand(p))
    }
    
    // Generate Makefile
    fmt.Println("=== Generated Makefile ===")
    makefile := generateMakefile(config)
    fmt.Println(makefile)
    
    // Generate Dockerfile
    fmt.Println("=== Generated Dockerfile ===")
    dockerfile := generateDockerfile(config)
    fmt.Println(dockerfile)
    
    // Detect if running in container
    fmt.Println("=== Environment Detection ===")
    inContainer := false
    if _, err := os.Stat("/.dockerenv"); err == nil {
        inContainer = true
    }
    fmt.Printf("  Running in container: %v\n", inContainer)
    fmt.Printf("  PID: %d\n", os.Getpid())
    fmt.Printf("  UID: %d\n", os.Getuid())
}`,
				},
				{
					Title: "Code Generation and go generate",
					Content: `Code generation reduces boilerplate and ensures consistency. Go provides go generate as a standard mechanism to invoke code generators.

**go generate:**
` + "```" + `
Syntax: place in .go file:
  //go:generate <command> <args>

Examples:
  //go:generate stringer -type=Color
  //go:generate mockgen -source=interfaces.go -destination=mocks/mock_store.go
  //go:generate protoc --go_out=. --go-grpc_out=. api.proto
  //go:generate go run gen.go

Run:
  go generate ./...          → Run all generators in all packages
  go generate ./pkg/models/  → Run generators in specific package

Rules:
  - go generate does NOT run during go build
  - Must be run explicitly
  - Generated files should be committed to VCS
  - Convention: add "Code generated ... DO NOT EDIT." header
  - Use //go:generate only in non-generated files

Common generators:
  stringer:    Enum String() method
  mockgen:     Interface mocks for testing
  protoc:      Protocol buffer code
  wire:        Compile-time dependency injection
  ent:         Database ORM
  sqlc:        Type-safe SQL
  oapi-codegen: OpenAPI → Go types + server/client
` + "```" + `

**stringer (Enum Strings):**
` + "```" + `
type Color int

const (
    Red Color = iota
    Green
    Blue
)

//go:generate stringer -type=Color

Generates color_string.go:
  func (c Color) String() string {
      // Returns "Red", "Green", or "Blue"
  }

Without stringer: fmt.Println(Red) → "0"
With stringer:    fmt.Println(Red) → "Red"

Install: go install golang.org/x/tools/cmd/stringer@latest
` + "```" + `

**sqlc (Type-safe SQL):**
` + "```" + `
Write SQL, generate Go code:

-- query.sql
-- name: GetUser :one
SELECT id, name, email FROM users WHERE id = $1;

-- name: ListUsers :many
SELECT id, name, email FROM users ORDER BY name;

-- name: CreateUser :one
INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *;

Generates:
  type User struct {
      ID    int64
      Name  string
      Email string
  }
  
  func (q *Queries) GetUser(ctx context.Context, id int64) (User, error)
  func (q *Queries) ListUsers(ctx context.Context) ([]User, error)
  func (q *Queries) CreateUser(ctx context.Context, arg CreateUserParams) (User, error)

Benefits:
  - Type-safe: catch SQL errors at compile time
  - No ORM overhead: raw SQL with Go types
  - Schema-aware: validates against database schema
  - Fast: no reflection, no runtime overhead
` + "```" + `

**Text Templates for Code Generation:**
` + "```" + `
Go's text/template can generate code:

  tmpl := template.Must(template.New("").Parse(` + "`" + `
  // Code generated by gen.go. DO NOT EDIT.
  package {{.Package}}
  
  type {{.Name}}Repository interface {
      Create(ctx context.Context, item *{{.Name}}) error
      GetByID(ctx context.Context, id {{.IDType}}) (*{{.Name}}, error)
      List(ctx context.Context, limit, offset int) ([]*{{.Name}}, error)
      Update(ctx context.Context, item *{{.Name}}) error
      Delete(ctx context.Context, id {{.IDType}}) error
  }
  ` + "`" + `))

Generate:
  var buf bytes.Buffer
  tmpl.Execute(&buf, map[string]string{
      "Package": "models",
      "Name":    "User",
      "IDType":  "int64",
  })
  
  // Format generated code
  formatted, _ := format.Source(buf.Bytes())
  os.WriteFile("user_repo_gen.go", formatted, 0644)

Always format generated Go code with go/format!
  import "go/format"
  formatted, err := format.Source(rawBytes)
` + "```" + ``,
					CodeExamples: `// Code generation patterns
package main

import (
    "bytes"
    "fmt"
    "strings"
    "text/template"
)

// Code generator for CRUD repository

type ModelDef struct {
    Package string
    Name    string
    Table   string
    IDType  string
    Fields  []FieldDef
}

type FieldDef struct {
    Name     string
    Type     string
    Column   string
    JSONTag  string
    Required bool
}

const repoTemplate = ` + "`" + `// Code generated by gen.go. DO NOT EDIT.
package {{.Package}}

import (
    "context"
    "database/sql"
    "fmt"
    "time"
)

// {{.Name}} represents a row in the {{.Table}} table.
type {{.Name}} struct {
{{- range .Fields}}
    {{.Name}} {{.Type}} ` + "`" + `json:"{{.JSONTag}}" db:"{{.Column}}"` + "`" + `
{{- end}}
    CreatedAt time.Time ` + "`" + `json:"created_at" db:"created_at"` + "`" + `
    UpdatedAt time.Time ` + "`" + `json:"updated_at" db:"updated_at"` + "`" + `
}

// {{.Name}}Repository provides CRUD operations for {{.Name}}.
type {{.Name}}Repository struct {
    db *sql.DB
}

// New{{.Name}}Repository creates a new repository.
func New{{.Name}}Repository(db *sql.DB) *{{.Name}}Repository {
    return &{{.Name}}Repository{db: db}
}

// Create inserts a new {{.Name}}.
func (r *{{.Name}}Repository) Create(ctx context.Context, item *{{.Name}}) error {
    columns := []string{ {{- range .Fields}}"{{.Column}}", {{end}}"created_at", "updated_at"}
    placeholders := make([]string, len(columns))
    for i := range placeholders {
        placeholders[i] = fmt.Sprintf("$%d", i+1)
    }
    
    query := fmt.Sprintf("INSERT INTO {{.Table}} (%s) VALUES (%s) RETURNING id",
        joinStrings(columns, ", "), joinStrings(placeholders, ", "))
    
    now := time.Now()
    return r.db.QueryRowContext(ctx, query,
        {{- range .Fields}}
        item.{{.Name}},
        {{- end}}
        now, now,
    ).Scan(&item.ID)
}

// GetByID retrieves a {{.Name}} by ID.
func (r *{{.Name}}Repository) GetByID(ctx context.Context, id {{.IDType}}) (*{{.Name}}, error) {
    var item {{.Name}}
    err := r.db.QueryRowContext(ctx,
        "SELECT {{range $i, $f := .Fields}}{{if $i}}, {{end}}{{$f.Column}}{{end}}, created_at, updated_at FROM {{.Table}} WHERE id = $1",
        id,
    ).Scan(
        {{- range .Fields}}
        &item.{{.Name}},
        {{- end}}
        &item.CreatedAt, &item.UpdatedAt,
    )
    if err != nil {
        return nil, fmt.Errorf("get {{.Name | toLower}} %v: %w", id, err)
    }
    return &item, nil
}

func joinStrings(s []string, sep string) string {
    result := ""
    for i, v := range s {
        if i > 0 { result += sep }
        result += v
    }
    return result
}
` + "`" + `

// Enum stringer generator
const stringerTemplate = ` + "`" + `// Code generated by gen.go. DO NOT EDIT.
package {{.Package}}

import "fmt"

func (v {{.TypeName}}) String() string {
    switch v {
    {{- range .Values}}
    case {{.Name}}:
        return "{{.Name}}"
    {{- end}}
    default:
        return fmt.Sprintf("{{.TypeName}}(%d)", int(v))
    }
}

func Parse{{.TypeName}}(s string) ({{.TypeName}}, error) {
    switch s {
    {{- range .Values}}
    case "{{.Name}}":
        return {{.Name}}, nil
    {{- end}}
    default:
        return 0, fmt.Errorf("unknown {{.TypeName}}: %s", s)
    }
}

func {{.TypeName}}Values() []{{.TypeName}} {
    return []{{.TypeName}}{
        {{- range .Values}}
        {{.Name}},
        {{- end}}
    }
}
` + "`" + `

type EnumDef struct {
    Package  string
    TypeName string
    Values   []EnumValue
}

type EnumValue struct {
    Name  string
    Value int
}

func generateCode(tmplStr string, data any) (string, error) {
    funcMap := template.FuncMap{
        "toLower": strings.ToLower,
    }
    
    tmpl, err := template.New("gen").Funcs(funcMap).Parse(tmplStr)
    if err != nil {
        return "", fmt.Errorf("parse template: %w", err)
    }
    
    var buf bytes.Buffer
    if err := tmpl.Execute(&buf, data); err != nil {
        return "", fmt.Errorf("execute template: %w", err)
    }
    
    return buf.String(), nil
}

func main() {
    // Generate CRUD repository
    fmt.Println("=== Generated Repository ===")
    
    model := ModelDef{
        Package: "models",
        Name:    "User",
        Table:   "users",
        IDType:  "int64",
        Fields: []FieldDef{
            {Name: "ID", Type: "int64", Column: "id", JSONTag: "id"},
            {Name: "Name", Type: "string", Column: "name", JSONTag: "name", Required: true},
            {Name: "Email", Type: "string", Column: "email", JSONTag: "email", Required: true},
            {Name: "Age", Type: "int", Column: "age", JSONTag: "age"},
        },
    }
    
    code, err := generateCode(repoTemplate, model)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    fmt.Println(code[:500] + "\n  ...")
    
    // Generate enum stringer
    fmt.Println("\n=== Generated Stringer ===")
    
    enum := EnumDef{
        Package:  "models",
        TypeName: "OrderStatus",
        Values: []EnumValue{
            {Name: "OrderPending", Value: 0},
            {Name: "OrderConfirmed", Value: 1},
            {Name: "OrderShipped", Value: 2},
            {Name: "OrderDelivered", Value: 3},
            {Name: "OrderCancelled", Value: 4},
        },
    }
    
    code, err = generateCode(stringerTemplate, enum)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    fmt.Println(code)
}`,
				},
			},
		},
	})
}
