package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          45,
			Title:       "Strings, Runes & Text Processing",
			Description: "Master Go's string internals, Unicode/UTF-8 handling, rune iteration, string building, and the strings/bytes packages.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "String Internals and UTF-8",
					Content: `In Go, a string is an immutable sequence of bytes, not characters. Understanding this distinction is essential for working correctly with Unicode text.

**1. String Header:**
*   A string is a two-word struct internally: a pointer to the data and a length (in bytes).
*   Strings are immutable -- you cannot modify individual bytes. Any "modification" creates a new string.
*   The ` + "`" + `len()` + "`" + ` function returns the number of **bytes**, not the number of characters.

**2. UTF-8 Encoding:**
*   Go source files are UTF-8 encoded.
*   ASCII characters use 1 byte. Most European characters use 2 bytes. CJK characters use 3 bytes. Emoji use 4 bytes.
*   "Hello" = 5 bytes. "Привет" (Russian) = 12 bytes (6 characters x 2 bytes). "日本語" (Japanese) = 9 bytes (3 characters x 3 bytes).

**3. Runes:**
*   A ` + "`" + `rune` + "`" + ` is an alias for ` + "`" + `int32` + "`" + `. It represents a single Unicode code point.
*   Use ` + "`" + `[]rune(s)` + "`" + ` to convert a string to a slice of code points for character-level operations.
*   ` + "`" + `range` + "`" + ` over a string iterates by rune (not by byte), automatically decoding UTF-8.

**4. Common Pitfalls:**
*   ` + "`" + `s[i]` + "`" + ` returns a byte, not a character. For multi-byte characters, this gives a meaningless byte value.
*   ` + "`" + `len(s)` + "`" + ` counts bytes. Use ` + "`" + `utf8.RuneCountInString(s)` + "`" + ` for character count.
*   String comparison is byte-wise. Two strings with the same characters in different Unicode normal forms may compare as unequal.`,
					CodeExamples: `package main

import (
    "fmt"
    "unicode/utf8"
)

func main() {
    s := "Hello, 世界!"

    // len() returns BYTES, not characters
    fmt.Println(len(s))                    // 13 (not 9)
    fmt.Println(utf8.RuneCountInString(s)) // 9

    // Indexing returns a BYTE
    fmt.Printf("%x\n", s[7]) // e4 (first byte of '世')

    // Range iterates by RUNE (correct!)
    for i, r := range s {
        fmt.Printf("byte %d: %c (U+%04X)\n", i, r, r)
    }

    // Convert to runes for character-level operations
    runes := []rune(s)
    fmt.Println(string(runes[7])) // 界
    fmt.Println(len(runes))       // 9 characters
}`,
				},
				{
					Title: "The strings Package",
					Content: `The ` + "`" + `strings` + "`" + ` package provides essential string manipulation functions. Since strings are immutable, these functions always return new strings.

**1. Searching:**
*   ` + "`" + `strings.Contains(s, substr)` + "`" + ` -- check if substring exists.
*   ` + "`" + `strings.HasPrefix(s, prefix)` + "`" + ` / ` + "`" + `HasSuffix` + "`" + ` -- check start/end.
*   ` + "`" + `strings.Index(s, substr)` + "`" + ` -- find position of first occurrence (-1 if not found).
*   ` + "`" + `strings.Count(s, substr)` + "`" + ` -- count non-overlapping occurrences.

**2. Transforming:**
*   ` + "`" + `strings.ToUpper(s)` + "`" + ` / ` + "`" + `ToLower(s)` + "`" + ` -- case conversion.
*   ` + "`" + `strings.TrimSpace(s)` + "`" + ` -- remove leading/trailing whitespace.
*   ` + "`" + `strings.Trim(s, cutset)` + "`" + ` -- remove specific characters from both ends.
*   ` + "`" + `strings.Replace(s, old, new, n)` + "`" + ` -- replace first n occurrences (-1 for all).
*   ` + "`" + `strings.ReplaceAll(s, old, new)` + "`" + ` -- replace all occurrences.
*   ` + "`" + `strings.Map(f, s)` + "`" + ` -- apply function to each rune.

**3. Splitting and Joining:**
*   ` + "`" + `strings.Split(s, sep)` + "`" + ` -- split into slice.
*   ` + "`" + `strings.Fields(s)` + "`" + ` -- split on any whitespace (handles multiple spaces).
*   ` + "`" + `strings.Join(slice, sep)` + "`" + ` -- join slice into string.

**4. Repeating and Padding:**
*   ` + "`" + `strings.Repeat(s, count)` + "`" + ` -- repeat a string.
*   ` + "`" + `fmt.Sprintf("%-20s", s)` + "`" + ` -- left-align with padding.

**5. strings.NewReplacer:**
*   Create a reusable replacer for multiple substitutions.
*   More efficient than chaining ` + "`" + `Replace` + "`" + ` calls.`,
					CodeExamples: `package main

import (
    "fmt"
    "strings"
)

func main() {
    s := "  Hello, Go World!  "

    // Searching
    fmt.Println(strings.Contains(s, "Go"))     // true
    fmt.Println(strings.HasPrefix(s, "  He"))  // true
    fmt.Println(strings.Count(s, "l"))         // 2

    // Transforming
    fmt.Println(strings.TrimSpace(s))          // "Hello, Go World!"
    fmt.Println(strings.ToUpper("hello"))      // "HELLO"

    // Splitting and Joining
    csv := "a,b,c,d"
    parts := strings.Split(csv, ",")
    fmt.Println(parts) // [a b c d]
    fmt.Println(strings.Join(parts, " | ")) // "a | b | c | d"

    // Fields handles multiple spaces
    messy := "  one   two  three "
    words := strings.Fields(messy)
    fmt.Println(words) // [one two three]

    // Replacer (efficient multi-replacement)
    r := strings.NewReplacer(
        "<", "&lt;",
        ">", "&gt;",
        "&", "&amp;",
    )
    safe := r.Replace("<script>alert('xss')&</script>")
    fmt.Println(safe) // &lt;script&gt;alert('xss')&amp;&lt;/script&gt;
}`,
				},
				{
					Title: "String Building and Performance",
					Content: `String concatenation with ` + "`" + `+` + "`" + ` creates a new string each time, copying all existing bytes. For building strings in loops, this is O(n^2). Go provides efficient alternatives.

**1. strings.Builder (Preferred):**
*   Pre-allocates a buffer and appends to it without copying.
*   Call ` + "`" + `builder.WriteString()` + "`" + `, ` + "`" + `WriteByte()` + "`" + `, or ` + "`" + `WriteRune()` + "`" + `.
*   Call ` + "`" + `builder.String()` + "`" + ` at the end. This is O(1) -- it does not copy the buffer.
*   Use ` + "`" + `builder.Grow(n)` + "`" + ` to pre-allocate if you know the final size.

**2. bytes.Buffer:**
*   Similar to strings.Builder but more versatile (implements io.Reader, io.Writer).
*   Use when you need to both write and read from the buffer, or need the result as ` + "`" + `[]byte` + "`" + `.
*   Slightly less efficient than strings.Builder for pure string building.

**3. fmt.Sprintf:**
*   Convenient for formatting but slower than Builder for simple concatenation.
*   Uses reflection internally to handle the ` + "`" + `%v` + "`" + ` verb.

**4. Performance Comparison (building a 10,000-character string):**
*   ` + "`" + `+` + "`" + ` concatenation: ~50 ms (creates 10,000 intermediate strings).
*   ` + "`" + `strings.Builder` + "`" + `: ~0.01 ms (one allocation).
*   ` + "`" + `bytes.Buffer` + "`" + `: ~0.02 ms.
*   ` + "`" + `strings.Join` + "`" + `: ~0.01 ms (if you already have a slice).

**5. []byte vs string Conversion:**
*   Converting ` + "`" + `string` + "`" + ` to ` + "`" + `[]byte` + "`" + ` and back requires copying (because strings are immutable).
*   The compiler optimizes some cases (e.g., ` + "`" + `string([]byte)` + "`" + ` in map lookups).
*   For high-performance code, minimize conversions.`,
					CodeExamples: `package main

import (
    "bytes"
    "fmt"
    "strings"
)

// BAD: O(n^2) -- creates N intermediate strings
func buildSlow(n int) string {
    s := ""
    for i := 0; i < n; i++ {
        s += "x" // Each += copies the entire string!
    }
    return s
}

// GOOD: O(n) -- single allocation
func buildFast(n int) string {
    var b strings.Builder
    b.Grow(n) // Pre-allocate
    for i := 0; i < n; i++ {
        b.WriteByte('x')
    }
    return b.String()
}

// bytes.Buffer (when you need io.Writer)
func buildWithBuffer() string {
    var buf bytes.Buffer
    fmt.Fprintf(&buf, "Name: %s, Age: %d", "Alice", 30)
    return buf.String()
}

// strings.Join (when building from a slice)
func buildFromSlice(parts []string) string {
    return strings.Join(parts, ", ")
}`,
				},
				{
					Title: "Regular Expressions",
					Content: `Go's ` + "`" + `regexp` + "`" + ` package implements RE2 syntax, which guarantees linear-time matching. This means no catastrophic backtracking -- Go regexes are always safe for untrusted input.

**1. Compiling Patterns:**
*   ` + "`" + `regexp.Compile(pattern)` + "`" + ` -- returns a *Regexp and error.
*   ` + "`" + `regexp.MustCompile(pattern)` + "`" + ` -- panics on invalid pattern. Use for package-level constants.
*   Compile once, use many times. Compilation is expensive; matching is fast.

**2. Common Methods:**
*   ` + "`" + `re.MatchString(s)` + "`" + ` -- does the string match?
*   ` + "`" + `re.FindString(s)` + "`" + ` -- first match.
*   ` + "`" + `re.FindAllString(s, n)` + "`" + ` -- all matches (n=-1 for all).
*   ` + "`" + `re.FindStringSubmatch(s)` + "`" + ` -- match with capture groups.
*   ` + "`" + `re.ReplaceAllString(s, repl)` + "`" + ` -- replace matches.

**3. Named Groups:**
*   Use ` + "`" + `(?P<name>pattern)` + "`" + ` for named capture groups.
*   Access via ` + "`" + `re.SubexpNames()` + "`" + ` and ` + "`" + `re.FindStringSubmatch()` + "`" + `.

**4. Limitations (RE2 vs PCRE):**
*   No lookahead or lookbehind.
*   No backreferences.
*   No atomic groups or possessive quantifiers.
*   These are deliberate design choices to guarantee O(n) matching.`,
					CodeExamples: `package main

import (
    "fmt"
    "regexp"
)

// Compile once at package level
var emailRe = regexp.MustCompile(
    ` + "`" + `^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$` + "`" + `,
)

func main() {
    // Simple matching
    fmt.Println(emailRe.MatchString("user@example.com"))  // true
    fmt.Println(emailRe.MatchString("not-an-email"))       // false

    // Find all matches
    re := regexp.MustCompile(` + "`" + `\d+` + "`" + `)
    nums := re.FindAllString("port 8080 and port 3000", -1)
    fmt.Println(nums) // [8080 3000]

    // Named capture groups
    logRe := regexp.MustCompile(
        ` + "`" + `(?P<level>\w+): (?P<msg>.+)` + "`" + `,
    )
    match := logRe.FindStringSubmatch("ERROR: disk full")
    for i, name := range logRe.SubexpNames() {
        if name != "" {
            fmt.Printf("%s: %s\n", name, match[i])
        }
    }
    // level: ERROR
    // msg: disk full

    // Replace
    cleaned := re.ReplaceAllString("v1.2.3", "X")
    fmt.Println(cleaned) // vX.X.X
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          46,
			Title:       "HTTP Servers & Web Development",
			Description: "Build production-quality HTTP servers in Go: routing, middleware, request handling, templates, REST APIs, and the new ServeMux in Go 1.22+.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "HTTP Server Fundamentals",
					Content: `Go's ` + "`" + `net/http` + "`" + ` package provides a production-ready HTTP server out of the box -- no frameworks needed. It powers some of the highest-traffic services on the internet.

**1. The Handler Interface:**
*   Everything revolves around ` + "`" + `http.Handler` + "`" + `: ` + "`" + `ServeHTTP(w http.ResponseWriter, r *http.Request)` + "`" + `.
*   ` + "`" + `http.HandlerFunc` + "`" + ` is an adapter that lets you use a regular function as a Handler.
*   The ResponseWriter is used to write the response body and headers.
*   The Request contains everything about the incoming request.

**2. Starting a Server:**
*   ` + "`" + `http.ListenAndServe(":8080", handler)` + "`" + ` -- starts an HTTP server.
*   Pass ` + "`" + `nil` + "`" + ` as handler to use the DefaultServeMux.
*   For production: use ` + "`" + `http.Server{}` + "`" + ` struct for timeouts, TLS, and graceful shutdown.

**3. The New ServeMux (Go 1.22+):**
*   Supports method-based routing: ` + "`" + `"GET /users/{id}"` + "`" + `.
*   Path parameters: ` + "`" + `r.PathValue("id")` + "`" + `.
*   Wildcard matching: ` + "`" + `"/files/{path...}"` + "`" + `.
*   This largely eliminates the need for third-party routers like chi or gorilla/mux.

**4. Request Lifecycle:**
1. Client sends HTTP request.
2. Go's HTTP server reads the request and creates an ` + "`" + `*http.Request` + "`" + `.
3. Server calls the matched handler's ` + "`" + `ServeHTTP` + "`" + ` method.
4. Handler writes response via ` + "`" + `http.ResponseWriter` + "`" + `.
5. ` + "`" + `WriteHeader()` + "`" + ` must be called before ` + "`" + `Write()` + "`" + `. If not called explicitly, ` + "`" + `200 OK` + "`" + ` is sent on first ` + "`" + `Write()` + "`" + `.`,
					CodeExamples: `package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
)

func main() {
    mux := http.NewServeMux()

    // Simple handler
    mux.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintln(w, "Hello, World!")
    })

    // Path parameters (Go 1.22+)
    mux.HandleFunc("GET /users/{id}", func(w http.ResponseWriter, r *http.Request) {
        id := r.PathValue("id")
        fmt.Fprintf(w, "User ID: %s\n", id)
    })

    // JSON response
    mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
    })

    // Production-ready server with timeouts
    srv := &http.Server{
        Addr:         ":8080",
        Handler:      mux,
        ReadTimeout:  5 * time.Second,
        WriteTimeout: 10 * time.Second,
        IdleTimeout:  120 * time.Second,
    }
    log.Fatal(srv.ListenAndServe())
}`,
				},
				{
					Title: "Middleware Pattern",
					Content: `Middleware is a function that wraps an HTTP handler to add cross-cutting concerns like logging, authentication, CORS, rate limiting, and panic recovery.

**1. The Pattern:**
*   A middleware is a function that takes an ` + "`" + `http.Handler` + "`" + ` and returns a new ` + "`" + `http.Handler` + "`" + `.
*   Signature: ` + "`" + `func(next http.Handler) http.Handler` + "`" + `.
*   The returned handler does its work, then calls ` + "`" + `next.ServeHTTP(w, r)` + "`" + `.

**2. Chaining:**
*   Middleware can be chained: ` + "`" + `logging(auth(rateLimiter(handler)))` + "`" + `.
*   The outermost middleware runs first on the way in, last on the way out.
*   This is a Russian-nesting-doll pattern.

**3. Common Middleware:**
*   **Logging:** Log request method, path, duration, status code.
*   **Recovery:** Catch panics in handlers, return 500 instead of crashing the server.
*   **CORS:** Set Access-Control-Allow-Origin headers.
*   **Authentication:** Check JWT tokens or session cookies.
*   **Rate Limiting:** Limit requests per IP using a token bucket.

**4. Request Context:**
*   Use ` + "`" + `r.Context()` + "`" + ` and ` + "`" + `context.WithValue` + "`" + ` to pass data from middleware to handlers.
*   Example: Auth middleware extracts user ID from JWT, stores in context.
*   Handler retrieves: ` + "`" + `userID := r.Context().Value("userID").(string)` + "`" + `.`,
					CodeExamples: `package main

import (
    "log"
    "net/http"
    "time"
)

// Middleware signature: func(http.Handler) http.Handler

func logging(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
    })
}

func recovery(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                log.Printf("panic: %v", err)
                http.Error(w, "Internal Server Error", 500)
            }
        }()
        next.ServeHTTP(w, r)
    })
}

func cors(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE")
        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }
        next.ServeHTTP(w, r)
    })
}

func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("GET /", homeHandler)

    // Chain middleware: recovery -> cors -> logging -> mux
    handler := recovery(cors(logging(mux)))
    http.ListenAndServe(":8080", handler)
}`,
				},
				{
					Title: "REST API Patterns",
					Content: `Building a REST API in Go follows a clean, idiomatic pattern. Go's standard library provides everything needed without heavy frameworks.

**1. Request Parsing:**
*   URL path parameters: ` + "`" + `r.PathValue("id")` + "`" + ` (Go 1.22+) or extract from ` + "`" + `r.URL.Path` + "`" + `.
*   Query parameters: ` + "`" + `r.URL.Query().Get("page")` + "`" + `.
*   JSON body: ` + "`" + `json.NewDecoder(r.Body).Decode(&data)` + "`" + `.
*   Form data: ` + "`" + `r.ParseForm()` + "`" + ` then ` + "`" + `r.FormValue("key")` + "`" + `.
*   Headers: ` + "`" + `r.Header.Get("Authorization")` + "`" + `.

**2. Response Writing:**
*   Set status: ` + "`" + `w.WriteHeader(http.StatusCreated)` + "`" + `.
*   Set content type: ` + "`" + `w.Header().Set("Content-Type", "application/json")` + "`" + `.
*   Write JSON: ` + "`" + `json.NewEncoder(w).Encode(data)` + "`" + `.
*   Write error: ` + "`" + `http.Error(w, "not found", http.StatusNotFound)` + "`" + `.

**3. Common Response Helper:**
*   Create a helper function that sets the content type, status code, and encodes JSON. Reduces boilerplate.

**4. Input Validation:**
*   Always validate input before processing.
*   Check required fields, value ranges, string lengths.
*   Return ` + "`" + `400 Bad Request` + "`" + ` with a JSON error body describing the issue.

**5. Error Handling Convention:**
*   Use consistent error response format: ` + "`" + `{"error": "message"}` + "`" + `.
*   Map internal errors to HTTP status codes.
*   Never expose internal error details to clients in production.`,
					CodeExamples: `package main

import (
    "encoding/json"
    "net/http"
    "sync"
)

type User struct {
    ID    string ` + "`" + `json:"id"` + "`" + `
    Name  string ` + "`" + `json:"name"` + "`" + `
    Email string ` + "`" + `json:"email"` + "`" + `
}

// In-memory store (use a database in production)
var (
    users = make(map[string]User)
    mu    sync.RWMutex
)

// JSON response helper
func jsonResponse(w http.ResponseWriter, status int, data any) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(data)
}

func jsonError(w http.ResponseWriter, status int, msg string) {
    jsonResponse(w, status, map[string]string{"error": msg})
}

// GET /api/users/{id}
func getUser(w http.ResponseWriter, r *http.Request) {
    id := r.PathValue("id")
    mu.RLock()
    user, ok := users[id]
    mu.RUnlock()
    if !ok {
        jsonError(w, http.StatusNotFound, "user not found")
        return
    }
    jsonResponse(w, http.StatusOK, user)
}

// POST /api/users
func createUser(w http.ResponseWriter, r *http.Request) {
    var user User
    if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
        jsonError(w, http.StatusBadRequest, "invalid JSON")
        return
    }
    if user.Name == "" || user.Email == "" {
        jsonError(w, http.StatusBadRequest, "name and email required")
        return
    }
    mu.Lock()
    users[user.ID] = user
    mu.Unlock()
    jsonResponse(w, http.StatusCreated, user)
}

func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("GET /api/users/{id}", getUser)
    mux.HandleFunc("POST /api/users", createUser)
    http.ListenAndServe(":8080", mux)
}`,
				},
				{
					Title: "Graceful Shutdown and Production Practices",
					Content: `A production HTTP server must handle shutdown gracefully, manage connections properly, and serve TLS.

**1. Graceful Shutdown:**
*   Catch OS signals (SIGINT, SIGTERM) and call ` + "`" + `srv.Shutdown(ctx)` + "`" + `.
*   ` + "`" + `Shutdown` + "`" + ` stops accepting new connections and waits for in-flight requests to complete.
*   Use a deadline context to limit how long to wait.

**2. Timeouts (Mandatory):**
*   ` + "`" + `ReadTimeout` + "`" + `: Maximum time to read the entire request (headers + body). Prevents Slowloris attacks.
*   ` + "`" + `WriteTimeout` + "`" + `: Maximum time to write the entire response. Prevents slow clients from holding connections.
*   ` + "`" + `IdleTimeout` + "`" + `: Maximum time to wait for the next request on a keep-alive connection.
*   ` + "`" + `ReadHeaderTimeout` + "`" + `: Maximum time to read request headers.

**3. TLS (HTTPS):**
*   ` + "`" + `srv.ListenAndServeTLS(certFile, keyFile)` + "`" + ` for HTTPS.
*   Use ` + "`" + `http.Redirect` + "`" + ` to redirect HTTP to HTTPS.
*   ` + "`" + `autocert` + "`" + ` package can automatically obtain Let's Encrypt certificates.

**4. Connection Limits:**
*   Use ` + "`" + `http.Server.MaxHeaderBytes` + "`" + ` to limit header size.
*   Limit request body size: ` + "`" + `http.MaxBytesReader(w, r.Body, maxSize)` + "`" + `.
*   Use connection limit middleware in front of the server for DDoS protection.

**5. Health Checks:**
*   Implement ` + "`" + `/healthz` + "`" + ` (liveness) and ` + "`" + `/readyz` + "`" + ` (readiness) endpoints.
*   Liveness: server is running. Readiness: server is ready to serve traffic (database connected, caches warm).`,
					CodeExamples: `package main

import (
    "context"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"
)

func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("ok"))
    })

    srv := &http.Server{
        Addr:              ":8080",
        Handler:           mux,
        ReadTimeout:       5 * time.Second,
        ReadHeaderTimeout: 2 * time.Second,
        WriteTimeout:      10 * time.Second,
        IdleTimeout:       120 * time.Second,
        MaxHeaderBytes:    1 << 20, // 1 MB
    }

    // Start server in goroutine
    go func() {
        log.Printf("Server starting on %s", srv.Addr)
        if err := srv.ListenAndServe(); err != http.ErrServerClosed {
            log.Fatalf("Server error: %v", err)
        }
    }()

    // Wait for interrupt signal
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    sig := <-quit
    log.Printf("Received signal %v, shutting down...", sig)

    // Graceful shutdown with 30-second deadline
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    if err := srv.Shutdown(ctx); err != nil {
        log.Fatalf("Forced shutdown: %v", err)
    }
    log.Println("Server stopped gracefully")
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          47,
			Title:       "CLI Tools & Project Structure",
			Description: "Build command-line tools with flag parsing, environment variables, configuration management, and organize Go projects using best practices.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Command-Line Argument Parsing",
					Content: `Go provides the ` + "`" + `flag` + "`" + ` package for parsing command-line arguments. For more complex CLIs, the ` + "`" + `os.Args` + "`" + ` slice gives raw access.

**1. The flag Package:**
*   ` + "`" + `flag.String("name", "default", "description")` + "`" + ` -- returns a *string.
*   ` + "`" + `flag.Int("port", 8080, "description")` + "`" + ` -- returns a *int.
*   ` + "`" + `flag.Bool("verbose", false, "description")` + "`" + ` -- returns a *bool.
*   Call ` + "`" + `flag.Parse()` + "`" + ` to parse. Remaining args available via ` + "`" + `flag.Args()` + "`" + `.

**2. flag.Var (Custom Types):**
*   Implement the ` + "`" + `flag.Value` + "`" + ` interface (` + "`" + `String()` + "`" + ` and ` + "`" + `Set(string) error` + "`" + `) for custom flag types.
*   Useful for comma-separated lists, durations, enums, etc.

**3. Subcommands:**
*   Use ` + "`" + `flag.NewFlagSet("name", flag.ExitOnError)` + "`" + ` for subcommand-specific flags.
*   Check ` + "`" + `os.Args[1]` + "`" + ` to determine the subcommand, then parse the appropriate FlagSet.

**4. Environment Variables:**
*   Use ` + "`" + `os.Getenv("KEY")` + "`" + ` for environment variables.
*   Common pattern: flags override env vars, which override defaults.
*   ` + "`" + `os.LookupEnv("KEY")` + "`" + ` returns a second bool indicating if the var is set.

**5. os.Args:**
*   ` + "`" + `os.Args[0]` + "`" + ` is the program name.
*   ` + "`" + `os.Args[1:]` + "`" + ` are the arguments.
*   Use for simple tools where flag is overkill.`,
					CodeExamples: `package main

import (
    "flag"
    "fmt"
    "os"
)

func main() {
    // Simple flags
    port := flag.Int("port", 8080, "server port")
    host := flag.String("host", "localhost", "server host")
    verbose := flag.Bool("v", false, "enable verbose logging")
    flag.Parse()

    fmt.Printf("Starting on %s:%d (verbose=%v)\n", *host, *port, *verbose)

    // Remaining positional args
    args := flag.Args()
    fmt.Println("Extra args:", args)
}

// Usage:
// go run main.go -port 3000 -v file1.txt file2.txt
// Starting on localhost:3000 (verbose=true)
// Extra args: [file1.txt file2.txt]

// Subcommand pattern:
func subcommands() {
    if len(os.Args) < 2 {
        fmt.Println("Usage: app <command> [flags]")
        os.Exit(1)
    }

    serveCmd := flag.NewFlagSet("serve", flag.ExitOnError)
    servePort := serveCmd.Int("port", 8080, "port")

    buildCmd := flag.NewFlagSet("build", flag.ExitOnError)
    buildOutput := buildCmd.String("o", "out", "output file")

    switch os.Args[1] {
    case "serve":
        serveCmd.Parse(os.Args[2:])
        fmt.Printf("Serving on :%d\n", *servePort)
    case "build":
        buildCmd.Parse(os.Args[2:])
        fmt.Printf("Building to %s\n", *buildOutput)
    default:
        fmt.Printf("Unknown command: %s\n", os.Args[1])
        os.Exit(1)
    }
}`,
				},
				{
					Title: "Configuration Management",
					Content: `Production Go applications need structured configuration from multiple sources: defaults, config files, environment variables, and flags.

**1. Configuration Struct Pattern:**
*   Define a ` + "`" + `Config` + "`" + ` struct with all settings.
*   Load defaults first, then overlay from file, env, and flags.
*   Use struct tags for JSON/YAML mapping.

**2. Config File Formats:**
*   **JSON:** Built-in support. Use ` + "`" + `json.NewDecoder` + "`" + ` to read.
*   **YAML:** Use ` + "`" + `gopkg.in/yaml.v3` + "`" + `. Most human-friendly format.
*   **TOML:** Use ` + "`" + `github.com/BurntSushi/toml` + "`" + `. Good for configuration.
*   **Environment variables:** Use ` + "`" + `os.Getenv` + "`" + ` or ` + "`" + `github.com/kelseyhightower/envconfig` + "`" + `.

**3. The 12-Factor App Approach:**
*   Store config in environment variables (portable, container-friendly).
*   Never commit secrets to source control.
*   Use a secret manager (Vault, AWS Secrets Manager) for sensitive values.

**4. Validation:**
*   Validate config at startup, fail fast with clear error messages.
*   Check required fields, valid ranges, reachable endpoints.
*   Log the loaded configuration (with secrets redacted).

**5. Hot Reloading:**
*   Watch config files with ` + "`" + `fsnotify` + "`" + ` for changes.
*   Use ` + "`" + `sync/atomic` + "`" + ` or ` + "`" + `sync.RWMutex` + "`" + ` to swap config without restart.`,
					CodeExamples: `package config

import (
    "encoding/json"
    "fmt"
    "os"
)

type Config struct {
    Server   ServerConfig   ` + "`" + `json:"server"` + "`" + `
    Database DatabaseConfig ` + "`" + `json:"database"` + "`" + `
    LogLevel string         ` + "`" + `json:"log_level"` + "`" + `
}

type ServerConfig struct {
    Port         int    ` + "`" + `json:"port"` + "`" + `
    Host         string ` + "`" + `json:"host"` + "`" + `
    ReadTimeout  int    ` + "`" + `json:"read_timeout_ms"` + "`" + `
    WriteTimeout int    ` + "`" + `json:"write_timeout_ms"` + "`" + `
}

type DatabaseConfig struct {
    DSN          string ` + "`" + `json:"dsn"` + "`" + `
    MaxOpenConns int    ` + "`" + `json:"max_open_conns"` + "`" + `
    MaxIdleConns int    ` + "`" + `json:"max_idle_conns"` + "`" + `
}

func Load(path string) (*Config, error) {
    // Start with defaults
    cfg := &Config{
        Server:   ServerConfig{Port: 8080, Host: "0.0.0.0", ReadTimeout: 5000, WriteTimeout: 10000},
        Database: DatabaseConfig{MaxOpenConns: 25, MaxIdleConns: 5},
        LogLevel: "info",
    }

    // Overlay from config file
    if path != "" {
        f, err := os.Open(path)
        if err != nil {
            return nil, fmt.Errorf("open config: %w", err)
        }
        defer f.Close()
        if err := json.NewDecoder(f).Decode(cfg); err != nil {
            return nil, fmt.Errorf("parse config: %w", err)
        }
    }

    // Overlay from environment variables
    if port := os.Getenv("PORT"); port != "" {
        fmt.Sscanf(port, "%d", &cfg.Server.Port)
    }
    if dsn := os.Getenv("DATABASE_URL"); dsn != "" {
        cfg.Database.DSN = dsn
    }

    // Validate
    if cfg.Database.DSN == "" {
        return nil, fmt.Errorf("database DSN is required")
    }
    return cfg, nil
}`,
				},
				{
					Title: "Go Project Structure",
					Content: `A well-organized Go project is easy to navigate, test, and maintain. While Go does not enforce a directory layout, the community has settled on conventions.

**1. The Standard Layout:**
*   ` + "`" + `cmd/` + "`" + ` -- Main applications. Each subdirectory is a binary (e.g., ` + "`" + `cmd/server/main.go` + "`" + `, ` + "`" + `cmd/cli/main.go` + "`" + `).
*   ` + "`" + `internal/` + "`" + ` -- Private packages. Go enforces that code outside this module cannot import from ` + "`" + `internal/` + "`" + `. This is your implementation detail.
*   ` + "`" + `pkg/` + "`" + ` -- Public library code intended for use by other projects. Only use if you're building a reusable library.
*   ` + "`" + `api/` + "`" + ` -- API definitions (OpenAPI specs, protobuf files).
*   ` + "`" + `web/` + "`" + ` -- Web assets (templates, static files).
*   ` + "`" + `configs/` + "`" + ` -- Configuration file templates.
*   ` + "`" + `scripts/` + "`" + ` -- Build, CI, and operational scripts.
*   ` + "`" + `docs/` + "`" + ` -- Design documents and user documentation.

**2. Internal Package Organization:**
*   Group by domain/feature, not by technical layer.
*   **Good:** ` + "`" + `internal/user/` + "`" + `, ` + "`" + `internal/order/` + "`" + `, ` + "`" + `internal/auth/` + "`" + `.
*   **Bad:** ` + "`" + `internal/models/` + "`" + `, ` + "`" + `internal/handlers/` + "`" + `, ` + "`" + `internal/repositories/` + "`" + ` (this leads to import cycles).

**3. Small Projects:**
*   For small tools, a flat structure is fine: ` + "`" + `main.go` + "`" + `, ` + "`" + `handler.go` + "`" + `, ` + "`" + `config.go` + "`" + `.
*   Don't over-engineer directories for a 500-line tool.

**4. Dependency Injection:**
*   Prefer constructor functions: ` + "`" + `func NewUserService(db *sql.DB, log *slog.Logger) *UserService` + "`" + `.
*   Wire dependencies in ` + "`" + `main()` + "`" + ` and pass them through constructors.
*   Avoid global variables and ` + "`" + `init()` + "`" + ` for dependencies.

**5. The main Function Pattern:**
*   Keep ` + "`" + `main()` + "`" + ` thin: parse config, create dependencies, start server.
*   Use a ` + "`" + `run()` + "`" + ` function that returns an error for testability.`,
					CodeExamples: `// Project layout:
// myproject/
// ├── cmd/
// │   └── server/
// │       └── main.go        <- Entry point
// ├── internal/
// │   ├── user/
// │   │   ├── handler.go     <- HTTP handlers
// │   │   ├── service.go     <- Business logic
// │   │   ├── repository.go  <- Database access
// │   │   └── model.go       <- Domain types
// │   ├── auth/
// │   │   └── middleware.go
// │   └── config/
// │       └── config.go
// ├── go.mod
// ├── go.sum
// ├── Makefile
// └── README.md

// cmd/server/main.go -- thin main function
package main

import (
    "log"
    "os"
    "myproject/internal/config"
    "myproject/internal/user"
)

func main() {
    if err := run(); err != nil {
        log.Fatal(err)
    }
}

func run() error {
    cfg, err := config.Load(os.Getenv("CONFIG_PATH"))
    if err != nil {
        return fmt.Errorf("load config: %w", err)
    }

    db, err := sql.Open("postgres", cfg.Database.DSN)
    if err != nil {
        return fmt.Errorf("open db: %w", err)
    }
    defer db.Close()

    userRepo := user.NewRepository(db)
    userSvc := user.NewService(userRepo)
    userHandler := user.NewHandler(userSvc)

    mux := http.NewServeMux()
    userHandler.RegisterRoutes(mux)

    srv := &http.Server{Addr: cfg.Server.Addr(), Handler: mux}
    return srv.ListenAndServe()
}`,
				},
				{
					Title: "Structured Logging with slog",
					Content: `Go 1.21 introduced ` + "`" + `log/slog` + "`" + ` as the standard structured logging package, replacing the basic ` + "`" + `log` + "`" + ` package for production use.

**1. Why Structured Logging?**
*   Plain text logs are hard to search, filter, and aggregate.
*   Structured logs emit key-value pairs (JSON or logfmt) that tools like ELK, Loki, or Datadog can parse.
*   Example: ` + "`" + `{"time":"...","level":"INFO","msg":"request","method":"GET","path":"/api","duration":"12ms"}` + "`" + `.

**2. slog Basics:**
*   ` + "`" + `slog.Info("message", "key", value, "key2", value2)` + "`" + ` -- log at INFO level with attributes.
*   Levels: ` + "`" + `Debug` + "`" + `, ` + "`" + `Info` + "`" + `, ` + "`" + `Warn` + "`" + `, ` + "`" + `Error` + "`" + `.
*   ` + "`" + `slog.With("key", value)` + "`" + ` -- create a child logger with pre-set attributes.

**3. Handlers:**
*   ` + "`" + `slog.NewTextHandler(w, opts)` + "`" + ` -- logfmt output (human-readable).
*   ` + "`" + `slog.NewJSONHandler(w, opts)` + "`" + ` -- JSON output (machine-readable).
*   Custom handlers can output to any format or destination.

**4. Logger Groups:**
*   ` + "`" + `logger.WithGroup("http")` + "`" + ` -- prefix all keys with ` + "`" + `http.` + "`" + `.
*   Useful for organizing attributes: ` + "`" + `http.method` + "`" + `, ` + "`" + `http.path` + "`" + `, ` + "`" + `http.status` + "`" + `.

**5. Context Integration:**
*   Pass logger via context for request-scoped attributes (request ID, user ID).
*   Middleware adds attributes; handlers use them automatically.`,
					CodeExamples: `package main

import (
    "log/slog"
    "net/http"
    "os"
    "time"
)

func main() {
    // JSON handler for production
    logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
        Level: slog.LevelInfo,
    }))
    slog.SetDefault(logger)

    // Basic logging
    slog.Info("server starting", "port", 8080)
    // {"time":"...","level":"INFO","msg":"server starting","port":8080}

    // With pre-set attributes
    dbLogger := logger.With("component", "database")
    dbLogger.Info("connected", "host", "localhost", "db", "myapp")
    // {"time":"...","level":"INFO","msg":"connected","component":"database","host":"localhost","db":"myapp"}

    // With groups
    httpLogger := logger.WithGroup("http")
    httpLogger.Info("request", "method", "GET", "path", "/api")
    // {"time":"...","level":"INFO","msg":"request","http":{"method":"GET","path":"/api"}}

    // Error with details
    slog.Error("query failed",
        "err", err,
        "query", "SELECT * FROM users",
        "duration", 150*time.Millisecond,
    )
}

// Logging middleware
func loggingMiddleware(logger *slog.Logger, next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        logger.Info("request",
            "method", r.Method,
            "path", r.URL.Path,
            "remote", r.RemoteAddr,
            "duration", time.Since(start),
        )
    })
}`,
				},
				{
					Title: "Embedding Files and Cross-Compilation",
					Content: `Go provides built-in features for embedding static files into binaries and cross-compiling for any platform.

**1. The embed Package (Go 1.16+):**
*   Embed files directly into the Go binary at compile time.
*   No external files needed at runtime -- everything is self-contained.
*   Use ` + "`" + `//go:embed` + "`" + ` directive to embed files or directories.
*   Supports ` + "`" + `string` + "`" + `, ` + "`" + `[]byte` + "`" + `, or ` + "`" + `embed.FS` + "`" + ` (for directories).

**2. Common Use Cases:**
*   HTML templates for web servers.
*   Static assets (CSS, JS, images).
*   SQL migration files.
*   Configuration defaults.
*   Version strings from git.

**3. Cross-Compilation:**
*   Go compiles to any supported OS/architecture from any host.
*   Set ` + "`" + `GOOS` + "`" + ` and ` + "`" + `GOARCH` + "`" + ` environment variables.
*   Common targets: ` + "`" + `linux/amd64` + "`" + `, ` + "`" + `linux/arm64` + "`" + `, ` + "`" + `darwin/arm64` + "`" + `, ` + "`" + `windows/amd64` + "`" + `.
*   The result is a static binary -- no runtime dependencies.

**4. Build Tags (Conditional Compilation):**
*   ` + "`" + `//go:build linux` + "`" + ` -- only compile this file on Linux.
*   ` + "`" + `//go:build !windows` + "`" + ` -- compile on everything except Windows.
*   ` + "`" + `//go:build integration` + "`" + ` -- only compile when ` + "`" + `-tags integration` + "`" + ` is passed.
*   Use for platform-specific code, test fixtures, and feature flags.

**5. Build-Time Variables:**
*   Use ` + "`" + `-ldflags "-X main.Version=1.2.3"` + "`" + ` to inject values at build time.
*   Common: version, commit hash, build date.`,
					CodeExamples: `package main

import (
    "embed"
    "fmt"
    "io/fs"
    "net/http"
)

// Embed a single file as string
//go:embed version.txt
var version string

// Embed a single file as bytes
//go:embed config/defaults.json
var defaultConfig []byte

// Embed an entire directory
//go:embed static/*
var staticFiles embed.FS

// Embed multiple patterns
//go:embed templates/*.html templates/*.tmpl
var templates embed.FS

func main() {
    fmt.Println("Version:", version)

    // Serve embedded static files
    staticFS, _ := fs.Sub(staticFiles, "static")
    http.Handle("/static/", http.StripPrefix("/static/",
        http.FileServer(http.FS(staticFS))))

    http.ListenAndServe(":8080", nil)
}

// Cross-compilation commands:
// GOOS=linux GOARCH=amd64 go build -o myapp-linux-amd64 ./cmd/server
// GOOS=darwin GOARCH=arm64 go build -o myapp-darwin-arm64 ./cmd/server
// GOOS=windows GOARCH=amd64 go build -o myapp.exe ./cmd/server

// Build with version injection:
// go build -ldflags "-X main.Version=$(git describe --tags)" ./cmd/server

// Build tags:
// //go:build linux && amd64
// //go:build integration

// Run with tags:
// go test -tags integration ./...`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          51,
			Title:       "Security, Race Detection & Performance",
			Description: "Write secure, thread-safe, and high-performance Go code: race detection, sync/atomic, crypto, secure coding, and profiling techniques.",
			Order:       21,
			Lessons: []problems.Lesson{
				{
					Title: "Race Detection and Data Races",
					Content: `Data races are one of the most common and dangerous bugs in concurrent Go programs. A data race occurs when two goroutines access the same variable concurrently, and at least one is writing.

**1. What is a Data Race?**
*   Two goroutines access the same memory location.
*   At least one access is a write.
*   No synchronization between the accesses.
*   Result: Undefined behavior. The program may crash, produce wrong results, or appear to work fine and fail later.

**2. The Race Detector:**
*   Built into Go: ` + "`" + `go run -race main.go` + "`" + `, ` + "`" + `go test -race ./...` + "`" + `, ` + "`" + `go build -race` + "`" + `.
*   Instruments all memory accesses at runtime.
*   Reports exact goroutine stacks where the race occurs.
*   10-20x slower and uses more memory -- use in testing, not production.
*   **Always run your CI tests with ` + "`" + `-race` + "`" + `.**

**3. Common Race Patterns:**
*   Shared counter without mutex or atomic.
*   Map accessed concurrently (maps are NOT safe for concurrent use).
*   Goroutine capturing loop variable by reference.
*   Reading struct fields that another goroutine is writing.

**4. Fixing Races:**
*   ` + "`" + `sync.Mutex` + "`" + ` / ` + "`" + `sync.RWMutex` + "`" + ` -- lock access to shared state.
*   ` + "`" + `sync/atomic` + "`" + ` -- lock-free atomic operations for counters and flags.
*   Channels -- communicate by sharing memory, don't share memory to communicate.
*   ` + "`" + `sync.Map` + "`" + ` -- concurrent-safe map (for specific patterns).
*   Redesign -- often the best fix is to avoid shared state entirely.`,
					CodeExamples: `package main

import (
    "fmt"
    "sync"
    "sync/atomic"
)

// BAD: Data race!
func raceExample() {
    counter := 0
    var wg sync.WaitGroup
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter++ // RACE! Multiple goroutines read-modify-write
        }()
    }
    wg.Wait()
    fmt.Println(counter) // Unpredictable: could be 980, 995, or 1000
}

// FIX 1: Mutex
func mutexExample() {
    var mu sync.Mutex
    counter := 0
    var wg sync.WaitGroup
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            mu.Lock()
            counter++
            mu.Unlock()
        }()
    }
    wg.Wait()
    fmt.Println(counter) // Always 1000
}

// FIX 2: Atomic (faster for simple counters)
func atomicExample() {
    var counter atomic.Int64
    var wg sync.WaitGroup
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter.Add(1) // Atomic, no lock needed
        }()
    }
    wg.Wait()
    fmt.Println(counter.Load()) // Always 1000
}

// BAD: Loop variable capture race
func loopRace() {
    for i := 0; i < 5; i++ {
        go func() {
            fmt.Println(i) // RACE! All goroutines see the same i
        }()
    }
    // In Go 1.22+, loop variables are per-iteration (this is fixed)
}`,
				},
				{
					Title: "Cryptography and Secure Coding",
					Content: `Go's ` + "`" + `crypto` + "`" + ` package provides a comprehensive set of cryptographic primitives. Writing secure Go code requires understanding both the crypto tools and the common pitfalls.

**1. Hashing:**
*   ` + "`" + `crypto/sha256` + "`" + ` -- SHA-256 (checksums, data integrity).
*   ` + "`" + `crypto/sha512` + "`" + ` -- SHA-512 (longer hash, some use cases).
*   **Password hashing:** NEVER use SHA for passwords. Use ` + "`" + `golang.org/x/crypto/bcrypt` + "`" + ` or ` + "`" + `argon2` + "`" + `.
*   ` + "`" + `crypto/hmac` + "`" + ` -- HMAC for message authentication.

**2. Encryption:**
*   ` + "`" + `crypto/aes` + "`" + ` -- AES block cipher. Always use with a mode (GCM recommended).
*   **AES-GCM:** Provides both encryption and authentication. Use ` + "`" + `cipher.NewGCM()` + "`" + `.
*   ` + "`" + `crypto/rand` + "`" + ` -- Cryptographically secure random number generator. Use for nonces, keys, tokens.
*   **NEVER use ` + "`" + `math/rand` + "`" + ` for security-sensitive values.**

**3. TLS:**
*   ` + "`" + `crypto/tls` + "`" + ` -- TLS client and server configuration.
*   ` + "`" + `crypto/x509` + "`" + ` -- Certificate parsing and validation.
*   Always use ` + "`" + `tls.Config{MinVersion: tls.VersionTLS13}` + "`" + ` for new services.

**4. Common Security Mistakes:**
*   Using ` + "`" + `==` + "`" + ` for comparing hashes or tokens (timing attack). Use ` + "`" + `crypto/subtle.ConstantTimeCompare` + "`" + `.
*   Hardcoding secrets in source code. Use environment variables or secret managers.
*   Not validating TLS certificates (` + "`" + `InsecureSkipVerify: true` + "`" + `). Only use in development.
*   SQL injection from string concatenation. Always use parameterized queries.
*   Not sanitizing user input for HTML (XSS). Use ` + "`" + `html/template` + "`" + ` which auto-escapes.`,
					CodeExamples: `package main

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "crypto/sha256"
    "crypto/subtle"
    "encoding/hex"
    "fmt"
    "io"

    "golang.org/x/crypto/bcrypt"
)

// SHA-256 hashing
func hashData(data []byte) string {
    h := sha256.Sum256(data)
    return hex.EncodeToString(h[:])
}

// Password hashing with bcrypt (CORRECT for passwords)
func hashPassword(password string) (string, error) {
    hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
    return string(hash), err
}

func checkPassword(password, hash string) bool {
    return bcrypt.CompareHashAndPassword([]byte(hash), []byte(password)) == nil
}

// Constant-time comparison (prevents timing attacks)
func secureCompare(a, b []byte) bool {
    return subtle.ConstantTimeCompare(a, b) == 1
}

// AES-GCM encryption
func encrypt(plaintext, key []byte) ([]byte, error) {
    block, err := aes.NewCipher(key)
    if err != nil {
        return nil, err
    }
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    nonce := make([]byte, gcm.NonceSize())
    if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
        return nil, err
    }
    return gcm.Seal(nonce, nonce, plaintext, nil), nil
}

// Secure random token generation
func generateToken(length int) (string, error) {
    b := make([]byte, length)
    if _, err := rand.Read(b); err != nil {
        return "", err
    }
    return hex.EncodeToString(b), nil
}`,
				},
				{
					Title: "Profiling and Performance Optimization",
					Content: `Go has world-class profiling tools built in. Before optimizing, measure. The go tool pprof and benchmarks tell you exactly where time is spent.

**1. Benchmarks:**
*   Write benchmark functions: ` + "`" + `func BenchmarkXxx(b *testing.B)` + "`" + `.
*   Run: ` + "`" + `go test -bench=. -benchmem ./...` + "`" + `.
*   ` + "`" + `-benchmem` + "`" + ` shows allocations per operation.
*   Use ` + "`" + `b.ResetTimer()` + "`" + ` after setup code.
*   Use ` + "`" + `b.RunParallel()` + "`" + ` for concurrent benchmarks.

**2. CPU Profiling:**
*   ` + "`" + `go test -cpuprofile=cpu.prof -bench=.` + "`" + ` -- generate CPU profile.
*   ` + "`" + `go tool pprof cpu.prof` + "`" + ` -- interactive analysis.
*   ` + "`" + `top` + "`" + ` -- show functions consuming most CPU.
*   ` + "`" + `list FuncName` + "`" + ` -- show line-by-line breakdown.
*   ` + "`" + `web` + "`" + ` -- visualize in browser (needs graphviz).

**3. Memory Profiling:**
*   ` + "`" + `go test -memprofile=mem.prof -bench=.` + "`" + ` -- generate heap profile.
*   Shows where allocations happen and how much memory is used.
*   ` + "`" + `-alloc_objects` + "`" + ` -- count of allocations (good for finding GC pressure).
*   ` + "`" + `-inuse_space` + "`" + ` -- current memory usage.

**4. Runtime Profiling (HTTP endpoint):**
*   Import ` + "`" + `_ "net/http/pprof"` + "`" + ` to add profiling endpoints to your HTTP server.
*   ` + "`" + `go tool pprof http://localhost:8080/debug/pprof/profile?seconds=30` + "`" + `.

**5. Common Optimizations:**
*   Pre-allocate slices with ` + "`" + `make([]T, 0, n)` + "`" + `.
*   Use ` + "`" + `strings.Builder` + "`" + ` instead of ` + "`" + `+` + "`" + ` concatenation.
*   Use ` + "`" + `sync.Pool` + "`" + ` for frequently allocated temporary objects.
*   Reduce pointer indirection (use value types when possible).
*   Use ` + "`" + `bufio.Reader` + "`" + `/` + "`" + `bufio.Writer` + "`" + ` for I/O.

**6. Escape Analysis:**
*   ` + "`" + `go build -gcflags="-m" ./...` + "`" + ` -- shows what escapes to the heap.
*   Heap allocations are 10-100x slower than stack allocations.
*   A variable escapes if: returned as pointer, stored in interface, captured by goroutine, or too large for stack.`,
					CodeExamples: `package main

import (
    "strings"
    "sync"
    "testing"
)

// Benchmark example
func BenchmarkConcat(b *testing.B) {
    for i := 0; i < b.N; i++ {
        s := ""
        for j := 0; j < 100; j++ {
            s += "x"
        }
    }
}

func BenchmarkBuilder(b *testing.B) {
    for i := 0; i < b.N; i++ {
        var sb strings.Builder
        sb.Grow(100)
        for j := 0; j < 100; j++ {
            sb.WriteByte('x')
        }
        _ = sb.String()
    }
}
// BenchmarkConcat:   ~5000 ns/op, 5 allocs/op
// BenchmarkBuilder:  ~100 ns/op,  1 allocs/op (50x faster!)

// sync.Pool for object reuse
var bufPool = sync.Pool{
    New: func() any {
        return new(strings.Builder)
    },
}

func getBuilder() *strings.Builder {
    b := bufPool.Get().(*strings.Builder)
    b.Reset()
    return b
}

func putBuilder(b *strings.Builder) {
    bufPool.Put(b)
}

// Run profiling:
// go test -cpuprofile=cpu.prof -memprofile=mem.prof -bench=. ./...
// go tool pprof -http=:9090 cpu.prof  (opens web UI)

// Escape analysis:
// go build -gcflags="-m" ./...
// Output: "./main.go:10:6: x escapes to heap"`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
