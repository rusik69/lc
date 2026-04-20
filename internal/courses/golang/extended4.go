package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          1623,
			Title:       "Networking and HTTP in Go",
			Description: "Build production-grade network services: HTTP servers and clients, gRPC, WebSockets, TCP/UDP, and middleware patterns.",
			Order:       23,
			Lessons: []problems.Lesson{
				{
					Title: "Production HTTP Servers",
					Content: `Go's net/http package is production-ready out of the box. Understanding its architecture helps you build scalable, secure web services.

**http.Server Configuration:**
` + "```" + `
server := &http.Server{
    Addr:              ":8080",
    Handler:           mux,
    ReadTimeout:       5 * time.Second,   // Time to read request headers
    ReadHeaderTimeout: 2 * time.Second,   // Time to read just headers
    WriteTimeout:      10 * time.Second,  // Time to write response
    IdleTimeout:       120 * time.Second, // Keep-alive timeout
    MaxHeaderBytes:    1 << 20,           // 1 MB max header size
}

WHY timeouts matter:
  Without ReadTimeout → Slowloris attack (slow header sends → hold connections)
  Without WriteTimeout → Hung clients hold server goroutines forever
  Without IdleTimeout → Keep-alive connections accumulate
  
  Each connection = one goroutine
  Without limits → OOM from too many goroutines

Connection lifecycle:
  1. Accept TCP connection
  2. TLS handshake (if HTTPS)
  3. Read request (subject to ReadTimeout)
  4. Route to handler
  5. Handler writes response (subject to WriteTimeout)
  6. If keep-alive → wait for next request (IdleTimeout)
  7. Close connection
` + "```" + `

**Routing with Go 1.22+ Enhanced Patterns:**
` + "```" + `
Go 1.22 added method-based routing and path parameters to the default mux:

mux := http.NewServeMux()

// Method-based routing
mux.HandleFunc("GET /api/users", listUsers)
mux.HandleFunc("POST /api/users", createUser)
mux.HandleFunc("GET /api/users/{id}", getUser)
mux.HandleFunc("PUT /api/users/{id}", updateUser)
mux.HandleFunc("DELETE /api/users/{id}", deleteUser)

// Path parameters
func getUser(w http.ResponseWriter, r *http.Request) {
    id := r.PathValue("id") // New in Go 1.22
    // ...
}

// Wildcard: match remaining path
mux.HandleFunc("GET /files/{path...}", serveFile)
func serveFile(w http.ResponseWriter, r *http.Request) {
    path := r.PathValue("path") // "foo/bar/baz.txt"
}

// Exact match vs prefix match
mux.HandleFunc("GET /api/", apiHandler)    // Matches /api/* (prefix)
mux.HandleFunc("GET /api/v2", v2Handler)   // Matches exactly /api/v2

Before Go 1.22, you needed third-party routers (chi, gorilla/mux)
for method routing and path params. Now stdlib handles most cases.
` + "```" + `

**Middleware Pattern:**
` + "```" + `
Middleware wraps handlers to add cross-cutting concerns:

  type Middleware func(http.Handler) http.Handler

  func Logging(next http.Handler) http.Handler {
      return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
          start := time.Now()
          // Wrap ResponseWriter to capture status code
          wr := &wrappedWriter{ResponseWriter: w, statusCode: 200}
          next.ServeHTTP(wr, r)
          log.Printf("%s %s %d %v", r.Method, r.URL.Path, 
              wr.statusCode, time.Since(start))
      })
  }

Chaining middleware:
  handler := Recovery(Logging(Auth(CORS(router))))
  
  Or with a chain helper:
  func Chain(h http.Handler, middlewares ...Middleware) http.Handler {
      for i := len(middlewares) - 1; i >= 0; i-- {
          h = middlewares[i](h)
      }
      return h
  }
  handler := Chain(router, CORS, Auth, Logging, Recovery)

Common middleware:
  - Logging (request/response logging)
  - Recovery (panic → 500 instead of crash)
  - CORS (Cross-Origin Resource Sharing headers)
  - Auth (JWT/session validation)
  - RateLimit (token bucket / sliding window)
  - RequestID (inject X-Request-ID header)
  - Compress (gzip response bodies)
  - Timeout (context deadline on requests)
` + "```" + `

**HTTP Client Best Practices:**
` + "```" + `
NEVER use http.DefaultClient in production:
  - No timeouts (can block forever)
  - Shares transport across entire program

Create a custom client:
  client := &http.Client{
      Timeout: 30 * time.Second, // Overall timeout
      Transport: &http.Transport{
          MaxIdleConns:        100,
          MaxIdleConnsPerHost: 10,
          IdleConnTimeout:     90 * time.Second,
          TLSHandshakeTimeout: 10 * time.Second,
          
          // Connection pooling (IMPORTANT for performance)
          // Reuses TCP connections for keep-alive
          // Without this: new TCP + TLS handshake per request
          
          // For high-throughput:
          MaxConnsPerHost:     50,
          DisableCompression:  false,
      },
  }

ALWAYS close response body:
  resp, err := client.Get(url)
  if err != nil { return err }
  defer resp.Body.Close() // ← MUST do this!
  
  // Even if you don't read the body!
  // Unclosed bodies leak connections from the pool

  // Read body with limit (prevent OOM from malicious server):
  body, err := io.ReadAll(io.LimitReader(resp.Body, 10<<20)) // 10 MB max

Context for cancellation:
  ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
  defer cancel()
  
  req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
  resp, err := client.Do(req)
` + "```" + ``,
					CodeExamples: `// Production HTTP server patterns
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"
)

// Response writer wrapper to capture status code
type wrappedWriter struct {
    http.ResponseWriter
    statusCode int
    written    bool
}

func (w *wrappedWriter) WriteHeader(code int) {
    if !w.written {
        w.statusCode = code
        w.written = true
    }
    w.ResponseWriter.WriteHeader(code)
}

// Middleware: Logging
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        wr := &wrappedWriter{ResponseWriter: w, statusCode: 200}
        next.ServeHTTP(wr, r)
        log.Printf("[%s] %s %s %d %v",
            r.Method, r.RemoteAddr, r.URL.Path,
            wr.statusCode, time.Since(start))
    })
}

// Middleware: Recovery from panics
func RecoveryMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                log.Printf("PANIC: %v", err)
                http.Error(w, "Internal Server Error", http.StatusInternalServerError)
            }
        }()
        next.ServeHTTP(w, r)
    })
}

// Middleware: Request timeout
func TimeoutMiddleware(timeout time.Duration) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            ctx, cancel := context.WithTimeout(r.Context(), timeout)
            defer cancel()
            next.ServeHTTP(w, r.WithContext(ctx))
        })
    }
}

// Middleware: Rate limiting
type RateLimiter struct {
    mu       sync.Mutex
    visitors map[string]*visitor
    rate     int
    window   time.Duration
}

type visitor struct {
    count    int
    lastSeen time.Time
}

func NewRateLimiter(rate int, window time.Duration) *RateLimiter {
    return &RateLimiter{
        visitors: make(map[string]*visitor),
        rate:     rate,
        window:   window,
    }
}

func (rl *RateLimiter) Middleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        rl.mu.Lock()
        ip := r.RemoteAddr
        v, exists := rl.visitors[ip]
        if !exists || time.Since(v.lastSeen) > rl.window {
            rl.visitors[ip] = &visitor{count: 1, lastSeen: time.Now()}
            rl.mu.Unlock()
            next.ServeHTTP(w, r)
            return
        }
        v.count++
        v.lastSeen = time.Now()
        if v.count > rl.rate {
            rl.mu.Unlock()
            http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
            return
        }
        rl.mu.Unlock()
        next.ServeHTTP(w, r)
    })
}

// Chain middleware
func Chain(h http.Handler, middlewares ...func(http.Handler) http.Handler) http.Handler {
    for i := len(middlewares) - 1; i >= 0; i-- {
        h = middlewares[i](h)
    }
    return h
}

// API types
type User struct {
    ID    string ` + "`" + `json:"id"` + "`" + `
    Name  string ` + "`" + `json:"name"` + "`" + `
    Email string ` + "`" + `json:"email"` + "`" + `
}

type UserStore struct {
    mu    sync.RWMutex
    users map[string]User
}

func NewUserStore() *UserStore {
    return &UserStore{users: make(map[string]User)}
}

// JSON response helper
func jsonResponse(w http.ResponseWriter, status int, data any) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(data)
}

// Handlers
func (s *UserStore) ListUsers(w http.ResponseWriter, r *http.Request) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    users := make([]User, 0, len(s.users))
    for _, u := range s.users {
        users = append(users, u)
    }
    jsonResponse(w, http.StatusOK, users)
}

func (s *UserStore) CreateUser(w http.ResponseWriter, r *http.Request) {
    var user User
    if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
        http.Error(w, "invalid JSON", http.StatusBadRequest)
        return
    }
    if user.ID == "" || user.Name == "" {
        http.Error(w, "id and name required", http.StatusBadRequest)
        return
    }
    
    s.mu.Lock()
    s.users[user.ID] = user
    s.mu.Unlock()
    
    jsonResponse(w, http.StatusCreated, user)
}

func (s *UserStore) GetUser(w http.ResponseWriter, r *http.Request) {
    id := r.PathValue("id")
    
    s.mu.RLock()
    user, ok := s.users[id]
    s.mu.RUnlock()
    
    if !ok {
        http.Error(w, "not found", http.StatusNotFound)
        return
    }
    jsonResponse(w, http.StatusOK, user)
}

func healthCheck(w http.ResponseWriter, r *http.Request) {
    jsonResponse(w, http.StatusOK, map[string]string{
        "status": "ok",
        "time":   time.Now().Format(time.RFC3339),
    })
}

func main() {
    store := NewUserStore()
    limiter := NewRateLimiter(100, time.Minute)
    
    mux := http.NewServeMux()
    mux.HandleFunc("GET /health", healthCheck)
    mux.HandleFunc("GET /api/users", store.ListUsers)
    mux.HandleFunc("POST /api/users", store.CreateUser)
    mux.HandleFunc("GET /api/users/{id}", store.GetUser)
    
    handler := Chain(mux,
        RecoveryMiddleware,
        LoggingMiddleware,
        TimeoutMiddleware(30*time.Second),
        limiter.Middleware,
    )
    
    server := &http.Server{
        Addr:              ":8080",
        Handler:           handler,
        ReadTimeout:       5 * time.Second,
        ReadHeaderTimeout: 2 * time.Second,
        WriteTimeout:      10 * time.Second,
        IdleTimeout:       120 * time.Second,
        MaxHeaderBytes:    1 << 20,
    }
    
    fmt.Println("Server starting on :8080")
    log.Fatal(server.ListenAndServe())
}`,
				},
				{
					Title: "gRPC and Protocol Buffers",
					Content: `gRPC is Google's high-performance RPC framework built on HTTP/2. It's widely used for microservice communication due to its strong typing, code generation, and efficiency.

**Why gRPC over REST/JSON:**
` + "```" + `
Feature          │ REST/JSON     │ gRPC
Serialization    │ JSON (text)   │ Protobuf (binary)
Transport        │ HTTP/1.1      │ HTTP/2
Schema           │ OpenAPI (opt) │ .proto (required)
Code generation  │ Optional      │ Built-in
Streaming        │ SSE/WebSocket │ Native bidirectional
Browser support  │ Native        │ grpc-web (proxy)
Performance      │ ~1x           │ ~3-10x faster
Message size     │ ~1x           │ ~0.3-0.5x (smaller)

When to use gRPC:
  ✓ Service-to-service communication
  ✓ High-throughput, low-latency requirements
  ✓ Streaming (server → client, client → server, bidirectional)
  ✓ Polyglot APIs (Go, Java, Python, C++, Node.js)
  ✗ Public-facing APIs (browsers prefer REST)
  ✗ Simple CRUD (gRPC is overkill)
` + "```" + `

**Protocol Buffer Definition:**
` + "```" + `
// user.proto
syntax = "proto3";

package user.v1;
option go_package = "myapp/gen/user/v1;userv1";

message User {
    string id = 1;
    string name = 2;
    string email = 3;
    int32 age = 4;
    repeated string tags = 5;       // Repeated = slice
    optional string bio = 6;        // Optional field
    google.protobuf.Timestamp created_at = 7;
}

message ListUsersRequest {
    int32 page_size = 1;
    string page_token = 2;
}

message ListUsersResponse {
    repeated User users = 1;
    string next_page_token = 2;
}

message GetUserRequest {
    string id = 1;
}

service UserService {
    // Unary RPC
    rpc GetUser(GetUserRequest) returns (User);
    rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);
    
    // Server streaming (server sends multiple messages)
    rpc WatchUser(GetUserRequest) returns (stream User);
    
    // Client streaming (client sends multiple messages)
    rpc BatchCreateUsers(stream User) returns (ListUsersResponse);
    
    // Bidirectional streaming
    rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}

Field numbers: NEVER change or reuse!
  1-15: Encoded in 1 byte (use for frequent fields)
  16-2047: Encoded in 2 bytes
  19000-19999: Reserved by protobuf implementation
` + "```" + `

**gRPC Patterns:**
` + "```" + `
Interceptors (middleware for gRPC):

Unary interceptor:
  func loggingInterceptor(
      ctx context.Context,
      req any,
      info *grpc.UnaryServerInfo,
      handler grpc.UnaryHandler,
  ) (any, error) {
      start := time.Now()
      resp, err := handler(ctx, req)
      log.Printf("method=%s duration=%v err=%v",
          info.FullMethod, time.Since(start), err)
      return resp, err
  }
  
  server := grpc.NewServer(
      grpc.UnaryInterceptor(loggingInterceptor),
  )

Stream interceptor:
  Similar but wraps the stream instead of a single call

Common interceptors:
  - Logging (request/response/duration)
  - Authentication (validate tokens)
  - Rate limiting
  - Retry (client-side, with backoff)
  - Circuit breaker
  - Metrics (Prometheus counters)
  - Tracing (OpenTelemetry spans)

Error handling:
  Use status codes (not HTTP codes):
  codes.OK, codes.NotFound, codes.InvalidArgument,
  codes.Internal, codes.Unavailable, codes.DeadlineExceeded
  
  return nil, status.Errorf(codes.NotFound, "user %s not found", id)

Health checking:
  gRPC has a standard health check protocol:
  grpc.health.v1.Health service
  Returns SERVING, NOT_SERVING, UNKNOWN
  Used by load balancers and Kubernetes
` + "```" + ``,
					CodeExamples: `// gRPC server and client concept (without actual protobuf dependency)
package main

import (
    "context"
    "errors"
    "fmt"
    "sync"
    "time"
)

// Simulated gRPC-like service (without actual gRPC dependency)

// "Proto" message types
type UserProto struct {
    ID    string
    Name  string
    Email string
}

type ListUsersReq struct {
    PageSize  int
    PageToken string
}

type ListUsersResp struct {
    Users         []UserProto
    NextPageToken string
}

// gRPC-like status codes
type Code int
const (
    OK Code = iota
    NotFound
    InvalidArgument
    Internal
    Unavailable
)

type Status struct {
    Code    Code
    Message string
}

func (s Status) Error() string { return fmt.Sprintf("rpc error: code=%d msg=%s", s.Code, s.Message) }

// Service interface (would be generated by protoc)
type UserServiceServer interface {
    GetUser(ctx context.Context, req *UserProto) (*UserProto, error)
    ListUsers(ctx context.Context, req *ListUsersReq) (*ListUsersResp, error)
}

// Implementation
type userServiceImpl struct {
    mu    sync.RWMutex
    users map[string]UserProto
}

func (s *userServiceImpl) GetUser(ctx context.Context, req *UserProto) (*UserProto, error) {
    if req.ID == "" {
        return nil, Status{InvalidArgument, "id is required"}
    }
    
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    user, ok := s.users[req.ID]
    if !ok {
        return nil, Status{NotFound, fmt.Sprintf("user %s not found", req.ID)}
    }
    return &user, nil
}

func (s *userServiceImpl) ListUsers(ctx context.Context, req *ListUsersReq) (*ListUsersResp, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    pageSize := req.PageSize
    if pageSize <= 0 { pageSize = 10 }
    
    users := make([]UserProto, 0, pageSize)
    for _, u := range s.users {
        users = append(users, u)
        if len(users) >= pageSize { break }
    }
    return &ListUsersResp{Users: users}, nil
}

// Interceptor pattern
type UnaryInterceptor func(ctx context.Context, method string, req any, handler func(context.Context, any) (any, error)) (any, error)

func loggingInterceptor(ctx context.Context, method string, req any, handler func(context.Context, any) (any, error)) (any, error) {
    start := time.Now()
    resp, err := handler(ctx, req)
    duration := time.Since(start)
    status := "OK"
    if err != nil { status = err.Error() }
    fmt.Printf("  [gRPC] %s duration=%v status=%s\n", method, duration, status)
    return resp, err
}

func recoveryInterceptor(ctx context.Context, method string, req any, handler func(context.Context, any) (any, error)) (any, error) {
    defer func() {
        if r := recover(); r != nil {
            fmt.Printf("  [PANIC] %s: %v\n", method, r)
        }
    }()
    return handler(ctx, req)
}

// Apply interceptors
func chainInterceptors(interceptors ...UnaryInterceptor) UnaryInterceptor {
    return func(ctx context.Context, method string, req any, handler func(context.Context, any) (any, error)) (any, error) {
        chain := handler
        for i := len(interceptors) - 1; i >= 0; i-- {
            ic := interceptors[i]
            prev := chain
            chain = func(ctx context.Context, req any) (any, error) {
                return ic(ctx, method, req, prev)
            }
        }
        return chain(ctx, req)
    }
}

// Retry with exponential backoff
func retryCall(ctx context.Context, maxRetries int, fn func() error) error {
    var lastErr error
    for i := 0; i <= maxRetries; i++ {
        if err := fn(); err != nil {
            lastErr = err
            var s Status
            if errors.As(err, &s) && s.Code != Unavailable {
                return err // Only retry Unavailable
            }
            
            backoff := time.Duration(1<<i) * 100 * time.Millisecond
            if i < maxRetries {
                fmt.Printf("  Retry %d/%d after %v: %v\n", i+1, maxRetries, backoff, err)
                select {
                case <-time.After(backoff):
                case <-ctx.Done():
                    return ctx.Err()
                }
            }
            continue
        }
        return nil
    }
    return lastErr
}

func main() {
    // Create service
    svc := &userServiceImpl{
        users: map[string]UserProto{
            "1": {ID: "1", Name: "Alice", Email: "alice@example.com"},
            "2": {ID: "2", Name: "Bob", Email: "bob@example.com"},
            "3": {ID: "3", Name: "Charlie", Email: "charlie@example.com"},
        },
    }
    
    interceptor := chainInterceptors(recoveryInterceptor, loggingInterceptor)
    
    // Simulate gRPC calls through interceptor chain
    fmt.Println("=== Unary RPCs ===")
    
    // GetUser - success
    _, _ = interceptor(context.Background(), "/user.v1.UserService/GetUser",
        &UserProto{ID: "1"},
        func(ctx context.Context, req any) (any, error) {
            return svc.GetUser(ctx, req.(*UserProto))
        })
    
    // GetUser - not found
    _, _ = interceptor(context.Background(), "/user.v1.UserService/GetUser",
        &UserProto{ID: "999"},
        func(ctx context.Context, req any) (any, error) {
            return svc.GetUser(ctx, req.(*UserProto))
        })
    
    // ListUsers
    resp, _ := interceptor(context.Background(), "/user.v1.UserService/ListUsers",
        &ListUsersReq{PageSize: 10},
        func(ctx context.Context, req any) (any, error) {
            return svc.ListUsers(ctx, req.(*ListUsersReq))
        })
    if r, ok := resp.(*ListUsersResp); ok {
        fmt.Printf("\n  Listed %d users\n", len(r.Users))
        for _, u := range r.Users {
            fmt.Printf("    - %s (%s)\n", u.Name, u.Email)
        }
    }
    
    // Retry demo
    fmt.Println("\n=== Retry with Backoff ===")
    attempts := 0
    err := retryCall(context.Background(), 3, func() error {
        attempts++
        if attempts < 3 {
            return Status{Unavailable, "service temporarily unavailable"}
        }
        return nil // Success on 3rd attempt
    })
    if err == nil {
        fmt.Printf("  Success after %d attempts\n", attempts)
    }
}`,
				},
				{
					Title: "WebSockets and TCP",
					Content: `Go's standard library provides excellent low-level networking primitives. Understanding net.Conn and WebSockets enables building real-time applications and custom protocols.

**TCP Server/Client:**
` + "```" + `
TCP Server:
  ln, err := net.Listen("tcp", ":9090")
  for {
      conn, err := ln.Accept()
      go handleConn(conn) // One goroutine per connection
  }

  func handleConn(conn net.Conn) {
      defer conn.Close()
      
      // Set deadline (prevent hung connections)
      conn.SetReadDeadline(time.Now().Add(30 * time.Second))
      
      scanner := bufio.NewScanner(conn)
      for scanner.Scan() {
          msg := scanner.Text()
          conn.Write([]byte("echo: " + msg + "\n"))
      }
  }

TCP Client:
  conn, err := net.DialTimeout("tcp", "server:9090", 5*time.Second)
  defer conn.Close()
  
  fmt.Fprintln(conn, "hello")
  response, _ := bufio.NewReader(conn).ReadString('\n')

Key points:
  - net.Conn implements io.ReadWriteCloser
  - Every blocked Read/Write parks the goroutine (no thread block)
  - Set deadlines to prevent resource leaks
  - Use bufio for efficient reading (reduces syscalls)
` + "```" + `

**Connection Pool Pattern:**
` + "```" + `
type ConnPool struct {
    mu      sync.Mutex
    conns   chan net.Conn
    factory func() (net.Conn, error)
    maxSize int
}

func NewConnPool(max int, factory func() (net.Conn, error)) *ConnPool {
    return &ConnPool{
        conns:   make(chan net.Conn, max),
        factory: factory,
        maxSize: max,
    }
}

func (p *ConnPool) Get() (net.Conn, error) {
    select {
    case conn := <-p.conns:
        return conn, nil // Reuse existing
    default:
        return p.factory() // Create new
    }
}

func (p *ConnPool) Put(conn net.Conn) {
    select {
    case p.conns <- conn: // Return to pool
    default:
        conn.Close() // Pool full, close
    }
}

Why connection pools:
  TCP handshake: ~1ms (same datacenter), ~50-100ms (cross-region)
  TLS handshake: add ~10-50ms
  With pooling: amortize handshake cost across many requests
  Rule: pools size ≈ expected concurrent requests
` + "```" + `

**WebSocket in Go:**
` + "```" + `
Using gorilla/websocket (most popular Go WebSocket library):

Server:
  var upgrader = websocket.Upgrader{
      ReadBufferSize:  1024,
      WriteBufferSize: 1024,
      CheckOrigin: func(r *http.Request) bool {
          return true // In production: validate origin!
      },
  }
  
  func wsHandler(w http.ResponseWriter, r *http.Request) {
      conn, err := upgrader.Upgrade(w, r, nil)
      if err != nil { return }
      defer conn.Close()
      
      for {
          messageType, message, err := conn.ReadMessage()
          if err != nil { break }
          
          // Echo back
          err = conn.WriteMessage(messageType, message)
          if err != nil { break }
      }
  }

WebSocket patterns:
  Read pump (one goroutine reads from WebSocket):
    Centralizes read logic, handles ping/pong
    
  Write pump (one goroutine writes to WebSocket):
    Serializes writes (WebSocket is NOT concurrent-safe for writes!)
    Uses a channel as message queue
    
  Hub pattern (broadcast to all connected clients):
    Hub goroutine manages set of connections
    Clients register/unregister through channels
    Messages broadcast to all registered clients

Important rules:
  - Only ONE goroutine can write at a time
  - Only ONE goroutine can read at a time
  - Use ping/pong for connection health checking
  - Set read/write deadlines
  - Handle reconnection on client side
` + "```" + ``,
					CodeExamples: `// TCP and WebSocket patterns
package main

import (
    "bufio"
    "context"
    "fmt"
    "net"
    "sync"
    "time"
)

// Simple TCP echo server
type TCPServer struct {
    listener net.Listener
    clients  sync.Map
    done     chan struct{}
}

func NewTCPServer(addr string) (*TCPServer, error) {
    ln, err := net.Listen("tcp", addr)
    if err != nil {
        return nil, err
    }
    return &TCPServer{
        listener: ln,
        done:     make(chan struct{}),
    }, nil
}

func (s *TCPServer) Start() {
    fmt.Printf("TCP server listening on %s\n", s.listener.Addr())
    for {
        conn, err := s.listener.Accept()
        if err != nil {
            select {
            case <-s.done:
                return
            default:
                fmt.Printf("Accept error: %v\n", err)
                continue
            }
        }
        go s.handleClient(conn)
    }
}

func (s *TCPServer) handleClient(conn net.Conn) {
    defer conn.Close()
    addr := conn.RemoteAddr().String()
    s.clients.Store(addr, conn)
    defer s.clients.Delete(addr)
    
    fmt.Printf("Client connected: %s\n", addr)
    
    conn.SetReadDeadline(time.Now().Add(5 * time.Minute))
    scanner := bufio.NewScanner(conn)
    
    for scanner.Scan() {
        msg := scanner.Text()
        conn.SetReadDeadline(time.Now().Add(5 * time.Minute))
        
        if msg == "quit" {
            fmt.Fprintf(conn, "goodbye\n")
            return
        }
        
        // Broadcast to all clients
        s.broadcast(addr, msg)
    }
    fmt.Printf("Client disconnected: %s\n", addr)
}

func (s *TCPServer) broadcast(sender, msg string) {
    s.clients.Range(func(key, value any) bool {
        if key.(string) != sender {
            conn := value.(net.Conn)
            fmt.Fprintf(conn, "[%s]: %s\n", sender, msg)
        }
        return true
    })
}

func (s *TCPServer) Stop() {
    close(s.done)
    s.listener.Close()
}

// WebSocket Hub pattern (simulated without websocket dependency)
type Client struct {
    id    string
    send  chan []byte
    close chan struct{}
}

type Hub struct {
    clients    map[string]*Client
    register   chan *Client
    unregister chan *Client
    broadcast  chan []byte
    mu         sync.RWMutex
    done       chan struct{}
}

func NewHub() *Hub {
    return &Hub{
        clients:    make(map[string]*Client),
        register:   make(chan *Client),
        unregister: make(chan *Client),
        broadcast:  make(chan []byte, 256),
        done:       make(chan struct{}),
    }
}

func (h *Hub) Run(ctx context.Context) {
    for {
        select {
        case client := <-h.register:
            h.mu.Lock()
            h.clients[client.id] = client
            h.mu.Unlock()
            fmt.Printf("  Hub: client %s registered (%d total)\n", client.id, len(h.clients))
            
        case client := <-h.unregister:
            h.mu.Lock()
            if _, ok := h.clients[client.id]; ok {
                delete(h.clients, client.id)
                close(client.send)
            }
            h.mu.Unlock()
            fmt.Printf("  Hub: client %s unregistered (%d total)\n", client.id, len(h.clients))
            
        case message := <-h.broadcast:
            h.mu.RLock()
            for id, client := range h.clients {
                select {
                case client.send <- message:
                default:
                    // Client's send buffer is full, remove them
                    close(client.send)
                    delete(h.clients, id)
                }
            }
            h.mu.RUnlock()
            
        case <-ctx.Done():
            return
        }
    }
}

// Connection pool
type ConnPool struct {
    conns   chan net.Conn
    factory func() (net.Conn, error)
    mu      sync.Mutex
    active  int
    maxSize int
}

func NewConnPool(max int, factory func() (net.Conn, error)) *ConnPool {
    return &ConnPool{
        conns:   make(chan net.Conn, max),
        factory: factory,
        maxSize: max,
    }
}

func (p *ConnPool) Get() (net.Conn, error) {
    select {
    case conn := <-p.conns:
        return conn, nil
    default:
        p.mu.Lock()
        defer p.mu.Unlock()
        if p.active >= p.maxSize {
            // Wait for a connection to be returned
            conn := <-p.conns
            return conn, nil
        }
        conn, err := p.factory()
        if err != nil {
            return nil, err
        }
        p.active++
        return conn, nil
    }
}

func (p *ConnPool) Put(conn net.Conn) {
    select {
    case p.conns <- conn:
    default:
        conn.Close()
        p.mu.Lock()
        p.active--
        p.mu.Unlock()
    }
}

func (p *ConnPool) Stats() (active int, idle int) {
    p.mu.Lock()
    defer p.mu.Unlock()
    return p.active, len(p.conns)
}

func main() {
    // Hub pattern demo
    fmt.Println("=== WebSocket Hub Pattern ===")
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()
    
    hub := NewHub()
    go hub.Run(ctx)
    
    // Simulate clients connecting
    clients := make([]*Client, 5)
    for i := 0; i < 5; i++ {
        clients[i] = &Client{
            id:   fmt.Sprintf("user-%d", i),
            send: make(chan []byte, 10),
        }
        hub.register <- clients[i]
    }
    
    time.Sleep(50 * time.Millisecond)
    
    // Broadcast a message
    hub.broadcast <- []byte("Hello, everyone!")
    
    time.Sleep(50 * time.Millisecond)
    
    // Check what each client received
    for _, c := range clients {
        select {
        case msg := <-c.send:
            fmt.Printf("  %s received: %s\n", c.id, string(msg))
        default:
            fmt.Printf("  %s: no message\n", c.id)
        }
    }
    
    // Unregister a client
    hub.unregister <- clients[2]
    
    time.Sleep(50 * time.Millisecond)
    
    // Connection pool demo
    fmt.Println("\n=== Connection Pool ===")
    pool := NewConnPool(3, func() (net.Conn, error) {
        fmt.Println("  Creating new connection")
        // In real code: return net.Dial("tcp", "server:9090")
        return nil, nil // Simulated
    })
    
    // Simulate getting and returning connections
    for i := 0; i < 5; i++ {
        conn, _ := pool.Get()
        active, idle := pool.Stats()
        fmt.Printf("  Get #%d: active=%d, idle=%d\n", i, active, idle)
        if conn != nil {
            pool.Put(conn)
        }
    }
}`,
				},
			},
		},
	})
}
