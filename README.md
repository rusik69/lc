# LeetCode Clone

A secure, local LeetCode-style coding challenge platform built with Go. Features 60 curated problems (20 Easy, 20 Medium, 20 Hard) with sandboxed code execution.

## Features

- **60 Problems** - 20 Easy, 20 Medium, 20 Hard problems
- **Secure Execution** - Docker-based sandboxing with persistent container
- **Real-time Testing** - Run individual or all test cases
- **Solutions Included** - Each problem has optimized solutions with complexity analysis
- **Rate Limiting** - 10 requests per minute per IP
- **Modern UI** - Responsive design with syntax highlighting

## Quick Start

### Prerequisites
- Go 1.23+
- Docker & Docker Compose

### Docker Deployment (Recommended)

```bash
# Clean rebuild and start (recommended)
make dev

# Or just start if already built
make docker-up
```

The server will start on `http://localhost:8080`

**Other Docker commands:**
```bash
make docker-down         # Stop containers
make docker-logs         # View logs
make docker-restart      # Restart containers (without rebuild)
make docker-rebuild      # Fast rebuild (uses cache)
make docker-rebuild-clean # Clean rebuild (no cache)
```

### Local Development

```bash
# Run locally with sandbox container
make dev-local

# Or build and run manually
go mod download
make build
make run
```

### Using Makefile

```bash
make build            # Build the application
make run              # Build and run
make dev              # Full Docker clean rebuild and restart
make dev-local        # Run locally with sandbox container
make test             # Run all tests
make test-unit        # Run unit tests only (fast)
make test-e2e         # Run e2e tests (requires Docker)
make test-coverage    # Run tests with coverage report
make clean            # Clean build artifacts
make lint             # Run golangci-lint
make fmt              # Format code
make vet              # Run go vet
make check            # Run all checks (fmt, vet, test)
```

## Project Structure

This project follows the [Standard Go Project Layout](https://github.com/golang-standards/project-layout).

```
lc/
├── cmd/                          # Main applications
│   └── lc/                       # Main application entrypoint
│       └── main.go               # Server initialization
│
├── internal/                     # Private application code
│   ├── handlers/                 # HTTP handlers
│   │   └── handlers.go           # Web request handlers
│   ├── executor/                 # Code execution engine
│   │   ├── executor.go           # Main executor logic
│   │   └── executor_stream.go    # Streaming executor
│   └── problems/                 # Problem definitions
│       ├── types.go              # Problem and TestCase types
│       ├── easy.go               # Easy problems (1-20)
│       ├── medium.go             # Medium problems (21-40)
│       └── hard.go               # Hard problems (41-60)
│
├── tests/                        # Test files
│   ├── handlers_test.go          # Handler tests
│   ├── problems_test.go          # Problem tests
│   ├── executor_test.go         # Executor tests
│   ├── e2e_solutions_test.go    # End-to-end tests
│   └── solutions_verification_test.go
│
├── web/                          # Web assets
│   ├── templates/                # HTML templates
│   │   ├── layout.html           # Base layout
│   │   ├── index.html            # Problem list
│   │   ├── problem.html          # Problem detail
│   │   └── result.html           # Test results
│   └── static/                   # Static files
│       └── style.css             # Styles
│
├── bin/                          # Compiled binaries (gitignored)
│   └── lc                        # Built executable
│
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker services configuration
├── Makefile                      # Build automation
├── go.mod                        # Go module definition
└── README.md                     # This file
```

### Package Organization

**cmd/lc**
- Application entrypoint
- Server initialization, routing setup, template loading

**internal/handlers**
- HTTP request handling
- Request parsing, response rendering, rate limiting
- Exports: `HandleIndex`, `HandleProblem`, `HandleRun`, `HandleSolution`, `InitTemplates`

**internal/executor**
- Code execution in Docker sandbox
- Test code generation, Docker container management
- Exports: `ExecuteCode`, `ExecuteCodeStream`, `GenerateTestCode`

**internal/problems**
- Problem definitions and data access
- Files: `types.go`, `easy.go`, `medium.go`, `hard.go`
- Exports: `Problem`, `TestCase`, `GetProblem`, `GetAllProblems`

**web/**
- HTML templates and static files (CSS, JS)

## Docker Architecture

This application uses Docker Compose with a persistent sandbox container for code execution.

### Architecture

- **App Container**: Runs the web application
- **Sandbox Container**: Persistent container where user code is executed
  - Resource limits: 512MB RAM, 1.5 CPUs, 100 processes max
  - Security: no-new-privileges, all capabilities dropped
  - Isolated with tmpfs for temporary storage

### How It Works

1. The main application runs in the `app` container
2. A persistent `sandbox` container stays running (with `sleep infinity`)
3. When code needs to be executed:
   - The app writes code to the sandbox container using `docker exec`
   - Code is executed inside the sandbox with resource limits
   - Output is captured and sent back to the user
   - Temporary files are cleaned up

### Benefits

- **Fast**: No container startup overhead for each execution (previously ~20 seconds, now < 1 second)
- **Efficient**: Single sandbox container reused for all executions
- **Secure**: Sandbox has resource limits and restricted capabilities
- **Isolated**: Each execution uses unique temporary files

## Problem Categories

### Easy (1-20)
Two Sum, Reverse String, FizzBuzz, Palindrome, Fibonacci, Contains Duplicate, Valid Anagram, Valid Parentheses, Merge Lists, Stock Profit, Binary Search, First Bad Version, Ransom Note, Climbing Stairs, Common Prefix, Single Number, Majority Element, Remove Duplicates, Move Zeroes, Reverse List

### Medium (21-40)
Group Anagrams, Top K Frequent, Product Except Self, Longest Consecutive, Valid Sudoku, Encode/Decode, Longest Substring, Character Replacement, Permutation String, Minimum Window, Valid Palindrome II, 3Sum, Container Water, Trapping Water, Min Subarray, Max Subarray, Jump Game I & II, Gas Station, Hand of Straights

### Hard (41-60)
Median Arrays, Valid Parentheses, Wildcard Matching, Regex Matching, Merge k Lists, Reverse k-Group, Largest Rectangle, Maximal Rectangle, Sliding Window Max, Word Ladder I & II, Alien Dictionary, Longest Path Matrix, Number Islands, Course Schedule, Serialize Tree, Max Path Sum, Subsequences, Edit Distance

## Usage

1. **Browse Problems** - Visit `http://localhost:8080` to see all problems
2. **Solve Problem** - Click on a problem to see description and test cases
3. **Run Code** - Click "Run" to test with first test case (streams output in real-time)
4. **Submit** - Click "Submit" to test all cases
5. **View Solution** - Click "Show Solution" to see optimized solution

### Example Solution

```go
func twoSum(nums []int, target int) []int {
    m := make(map[int]int)
    for i, num := range nums {
        if idx, ok := m[target-num]; ok {
            return []int{idx, i}
        }
        m[num] = i
    }
    return nil
}
```

## Security

### Code Execution Sandbox
- **Persistent Container** - Single sandbox container for all executions (fast & efficient)
- **Memory Limit** - 512MB for sandbox container
- **CPU Limit** - 1.5 cores per container
- **Process Limit** - 100 processes maximum
- **Time Limit** - 25 second execution timeout
- **Isolated Execution** - Each execution uses unique temporary files
- **No Capabilities** - All Linux capabilities dropped
- **No New Privileges** - Prevents privilege escalation
- **Automatic Cleanup** - Temporary files deleted after each execution

### Input Validation
- **Rate Limiting** - 10 requests/minute per IP
- **Code Size Limit** - 100KB maximum
- **Request Size Limit** - 150KB maximum
- **Pattern Detection** - Blocks dangerous imports (`os/exec`, `syscall`, `unsafe`, etc.)

## Testing

### Run Tests
```bash
# All tests
make test

# Unit tests only (fast)
make test-unit

# E2E tests (requires Docker)
make test-e2e

# With coverage
make test-coverage

# Test specific package
go test ./tests/... -v
go test ./internal/problems/...
go test ./internal/handlers/...
go test ./internal/executor/...
```

### Test Coverage
- Tests verify all 60 problems generate valid code
- Security features fully tested
- End-to-end tests with Docker execution

## Development

### Adding New Problems

1. Add problem definition to appropriate file:
   - Easy: `internal/problems/easy.go`
   - Medium: `internal/problems/medium.go`
   - Hard: `internal/problems/hard.go`

2. Update `EasyProblems`, `MediumProblems`, or `HardProblems` slice

3. Add test case logic in `internal/executor/executor.go` if needed

4. Run tests to verify: `make test`

### Code Style
- Run `make fmt` to format code
- Run `make vet` to check for issues
- Run `make lint` for comprehensive linting
- Run `make check` to run all checks

### Docker Development

```bash
# Terminal 1: Start containers
make docker-up

# Terminal 2: Watch logs
make docker-logs

# After code changes
make docker-restart        # Quick restart
make docker-rebuild        # Fast rebuild
make dev                   # Clean rebuild
```

## Troubleshooting

### Docker Issues

**Container name conflict:**
```bash
# Clean up conflicting containers
make docker-down
docker rm -f lc-sandbox 2>/dev/null || true
make docker-up
```

**Sandbox container not running:**
```bash
docker ps | grep lc-sandbox
# If not running:
make docker-restart
```

**View sandbox container logs:**
```bash
docker logs lc-sandbox
```

**Clean start:**
```bash
make docker-down
docker system prune -f
make docker-up
```

**Check sandbox container health:**
```bash
docker exec lc-sandbox go version
```

**Docker build fails:**
```bash
# Ensure Docker is running
docker info

# Pull latest base images
docker pull golang:1.23-alpine

# Clean rebuild
make docker-build-no-cache
```

### Port Already in Use
```bash
# Find process using port 8080
lsof -i :8080

# Change port in cmd/lc/main.go if needed
Addr: ":3000",  // Instead of ":8080"
```

### Build Issues
```bash
# Clean and rebuild
make clean
make build

# Update dependencies
go mod tidy
go mod download
```

## CI/CD

GitHub Actions workflow runs on every push:
- Go 1.23 setup
- Dependency caching
- Code formatting check
- go vet
- Unit tests
- Build verification
- Coverage reporting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make check`
5. Submit a pull request

## License

MIT License - See LICENSE file for details
