package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          1630,
			Title:       "Design Patterns in Go",
			Description: "Implement classic and Go-idiomatic design patterns: creational, structural, behavioral, and concurrency patterns.",
			Order:       30,
			Lessons: []problems.Lesson{
				{
					Title: "Creational and Structural Patterns",
					Content: `Go's approach to design patterns differs from traditional OOP languages. Go favors composition over inheritance, interfaces over abstract classes, and simplicity over complexity.

**Functional Options Pattern:**
` + "```" + `
The most idiomatic Go creational pattern for configurable types:

type Server struct {
    host    string
    port    int
    timeout time.Duration
    maxConn int
    tls     bool
}

type Option func(*Server)

func WithPort(port int) Option {
    return func(s *Server) { s.port = port }
}

func WithTimeout(d time.Duration) Option {
    return func(s *Server) { s.timeout = d }
}

func WithMaxConnections(n int) Option {
    return func(s *Server) { s.maxConn = n }
}

func WithTLS() Option {
    return func(s *Server) { s.tls = true }
}

func NewServer(host string, opts ...Option) *Server {
    s := &Server{
        host:    host,
        port:    8080,           // Defaults
        timeout: 30 * time.Second,
        maxConn: 100,
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// Usage:
s := NewServer("localhost",
    WithPort(9090),
    WithTimeout(60 * time.Second),
    WithTLS(),
)

Why functional options:
  - Self-documenting (WithPort is clearer than positional args)
  - Backward compatible (adding new options doesn't break callers)
  - Sensible defaults (no zero-value surprises)
  - No config struct explosion
  - Used by: gRPC, Zap logger, many stdlib proposals
` + "```" + `

**Builder Pattern:**
` + "```" + `
For complex object construction with validation:

type QueryBuilder struct {
    table      string
    columns    []string
    conditions []string
    orderBy    string
    limit      int
    err        error
}

func NewQuery(table string) *QueryBuilder {
    return &QueryBuilder{table: table, columns: []string{"*"}}
}

func (qb *QueryBuilder) Select(cols ...string) *QueryBuilder {
    qb.columns = cols
    return qb
}

func (qb *QueryBuilder) Where(condition string) *QueryBuilder {
    qb.conditions = append(qb.conditions, condition)
    return qb
}

func (qb *QueryBuilder) OrderBy(col string) *QueryBuilder {
    qb.orderBy = col
    return qb
}

func (qb *QueryBuilder) Limit(n int) *QueryBuilder {
    if n <= 0 { qb.err = errors.New("limit must be positive") }
    qb.limit = n
    return qb
}

func (qb *QueryBuilder) Build() (string, error) {
    if qb.err != nil { return "", qb.err }
    // Build SQL string
    ...
}

// Usage (fluent API):
query, err := NewQuery("users").
    Select("id", "name", "email").
    Where("active = true").
    OrderBy("name").
    Limit(10).
    Build()
` + "```" + `

**Decorator Pattern (Middleware):**
` + "```" + `
Wrap functionality around existing types:

type RoundTripper interface {
    RoundTrip(*http.Request) (*http.Response, error)
}

// Logging decorator
type loggingTransport struct {
    next http.RoundTripper
}

func (t *loggingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
    start := time.Now()
    resp, err := t.next.RoundTrip(req)
    log.Printf("%s %s %v", req.Method, req.URL, time.Since(start))
    return resp, err
}

// Auth decorator
type authTransport struct {
    next  http.RoundTripper
    token string
}

func (t *authTransport) RoundTrip(req *http.Request) (*http.Response, error) {
    req.Header.Set("Authorization", "Bearer "+t.token)
    return t.next.RoundTrip(req)
}

// Stack decorators:
client := &http.Client{
    Transport: &loggingTransport{
        next: &authTransport{
            next:  http.DefaultTransport,
            token: "my-token",
        },
    },
}

This is the SAME pattern as HTTP middleware but applied to any interface.
Go's implicit interfaces make this extremely powerful.
` + "```" + `

**Adapter Pattern:**
` + "```" + `
Convert one interface to another:

// Third-party logger
type ExternalLogger interface {
    LogMessage(level int, message string)
}

// Your application's logger interface
type Logger interface {
    Info(msg string)
    Error(msg string)
}

// Adapter
type loggerAdapter struct {
    external ExternalLogger
}

func (a *loggerAdapter) Info(msg string)  { a.external.LogMessage(0, msg) }
func (a *loggerAdapter) Error(msg string) { a.external.LogMessage(2, msg) }

func NewLoggerAdapter(ext ExternalLogger) Logger {
    return &loggerAdapter{external: ext}
}

// http.HandlerFunc is also an adapter:
type HandlerFunc func(ResponseWriter, *Request)
func (f HandlerFunc) ServeHTTP(w ResponseWriter, r *Request) { f(w, r) }
// Adapts a function to the Handler interface!
` + "```" + ``,
					CodeExamples: `// Creational and structural patterns in Go
package main

import (
    "fmt"
    "strings"
    "time"
)

// Functional Options Pattern
type Server struct {
    host       string
    port       int
    timeout    time.Duration
    maxConn    int
    tls        bool
    middleware []string
}

type ServerOption func(*Server)

func WithPort(port int) ServerOption {
    return func(s *Server) { s.port = port }
}

func WithTimeout(d time.Duration) ServerOption {
    return func(s *Server) { s.timeout = d }
}

func WithMaxConnections(n int) ServerOption {
    return func(s *Server) { s.maxConn = n }
}

func WithTLS() ServerOption {
    return func(s *Server) { s.tls = true }
}

func WithMiddleware(names ...string) ServerOption {
    return func(s *Server) { s.middleware = append(s.middleware, names...) }
}

func NewServer(host string, opts ...ServerOption) *Server {
    s := &Server{
        host:    host,
        port:    8080,
        timeout: 30 * time.Second,
        maxConn: 100,
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

func (s *Server) String() string {
    scheme := "http"
    if s.tls {
        scheme = "https"
    }
    return fmt.Sprintf("%s://%s:%d (timeout=%v, maxConn=%d, middleware=%v)",
        scheme, s.host, s.port, s.timeout, s.maxConn, s.middleware)
}

// Builder Pattern - SQL Query Builder
type QueryBuilder struct {
    table      string
    columns    []string
    conditions []string
    orderBy    string
    limit      int
    offset     int
    err        error
}

func NewQuery(table string) *QueryBuilder {
    return &QueryBuilder{table: table, columns: []string{"*"}}
}

func (qb *QueryBuilder) Select(cols ...string) *QueryBuilder {
    if qb.err != nil {
        return qb
    }
    qb.columns = cols
    return qb
}

func (qb *QueryBuilder) Where(condition string) *QueryBuilder {
    if qb.err != nil {
        return qb
    }
    qb.conditions = append(qb.conditions, condition)
    return qb
}

func (qb *QueryBuilder) OrderBy(col string) *QueryBuilder {
    if qb.err != nil {
        return qb
    }
    qb.orderBy = col
    return qb
}

func (qb *QueryBuilder) Limit(n int) *QueryBuilder {
    if qb.err != nil {
        return qb
    }
    if n <= 0 {
        qb.err = fmt.Errorf("limit must be positive, got %d", n)
        return qb
    }
    qb.limit = n
    return qb
}

func (qb *QueryBuilder) Offset(n int) *QueryBuilder {
    if qb.err != nil {
        return qb
    }
    qb.offset = n
    return qb
}

func (qb *QueryBuilder) Build() (string, error) {
    if qb.err != nil {
        return "", qb.err
    }
    
    var parts []string
    parts = append(parts, "SELECT "+strings.Join(qb.columns, ", "))
    parts = append(parts, "FROM "+qb.table)
    
    if len(qb.conditions) > 0 {
        parts = append(parts, "WHERE "+strings.Join(qb.conditions, " AND "))
    }
    if qb.orderBy != "" {
        parts = append(parts, "ORDER BY "+qb.orderBy)
    }
    if qb.limit > 0 {
        parts = append(parts, fmt.Sprintf("LIMIT %d", qb.limit))
    }
    if qb.offset > 0 {
        parts = append(parts, fmt.Sprintf("OFFSET %d", qb.offset))
    }
    
    return strings.Join(parts, " "), nil
}

// Decorator Pattern
type DataProcessor interface {
    Process(data []byte) ([]byte, error)
}

// Base processor
type jsonProcessor struct{}

func (p *jsonProcessor) Process(data []byte) ([]byte, error) {
    return data, nil // Simplified
}

// Logging decorator
type loggingProcessor struct {
    next DataProcessor
    name string
}

func WithLogging(name string, next DataProcessor) DataProcessor {
    return &loggingProcessor{next: next, name: name}
}

func (p *loggingProcessor) Process(data []byte) ([]byte, error) {
    fmt.Printf("  [%s] Processing %d bytes\n", p.name, len(data))
    start := time.Now()
    result, err := p.next.Process(data)
    fmt.Printf("  [%s] Done in %v (output: %d bytes)\n", p.name, time.Since(start), len(result))
    return result, err
}

// Validation decorator
type validatingProcessor struct {
    next    DataProcessor
    maxSize int
}

func WithValidation(maxSize int, next DataProcessor) DataProcessor {
    return &validatingProcessor{next: next, maxSize: maxSize}
}

func (p *validatingProcessor) Process(data []byte) ([]byte, error) {
    if len(data) > p.maxSize {
        return nil, fmt.Errorf("data too large: %d > %d", len(data), p.maxSize)
    }
    return p.next.Process(data)
}

// Compression decorator
type compressingProcessor struct {
    next DataProcessor
}

func WithCompression(next DataProcessor) DataProcessor {
    return &compressingProcessor{next: next}
}

func (p *compressingProcessor) Process(data []byte) ([]byte, error) {
    result, err := p.next.Process(data)
    if err != nil {
        return nil, err
    }
    // Simulate compression
    return result, nil
}

// Adapter Pattern
type OldLogger struct{}

func (l *OldLogger) LogMessage(level int, msg string) {
    levels := []string{"INFO", "WARN", "ERROR"}
    name := "UNKNOWN"
    if level >= 0 && level < len(levels) {
        name = levels[level]
    }
    fmt.Printf("  [OldLogger] [%s] %s\n", name, msg)
}

type AppLogger interface {
    Info(msg string)
    Warn(msg string)
    Error(msg string)
}

type loggerAdapter struct {
    old *OldLogger
}

func AdaptLogger(old *OldLogger) AppLogger {
    return &loggerAdapter{old: old}
}

func (a *loggerAdapter) Info(msg string)  { a.old.LogMessage(0, msg) }
func (a *loggerAdapter) Warn(msg string)  { a.old.LogMessage(1, msg) }
func (a *loggerAdapter) Error(msg string) { a.old.LogMessage(2, msg) }

func main() {
    // Functional Options
    fmt.Println("=== Functional Options ===")
    
    s1 := NewServer("localhost")
    fmt.Printf("  Default: %s\n", s1)
    
    s2 := NewServer("api.example.com",
        WithPort(443),
        WithTLS(),
        WithTimeout(60*time.Second),
        WithMaxConnections(1000),
        WithMiddleware("auth", "logging", "cors"),
    )
    fmt.Printf("  Custom:  %s\n", s2)
    
    // Builder Pattern
    fmt.Println("\n=== Builder Pattern ===")
    
    q1, err := NewQuery("users").
        Select("id", "name", "email").
        Where("active = true").
        Where("age > 18").
        OrderBy("name ASC").
        Limit(10).
        Build()
    if err != nil {
        fmt.Printf("  Error: %v\n", err)
    } else {
        fmt.Printf("  Query: %s\n", q1)
    }
    
    q2, err := NewQuery("orders").
        Select("id", "total", "status").
        Where("user_id = $1").
        Where("status = 'pending'").
        OrderBy("created_at DESC").
        Limit(20).
        Offset(40).
        Build()
    if err != nil {
        fmt.Printf("  Error: %v\n", err)
    } else {
        fmt.Printf("  Query: %s\n", q2)
    }
    
    // Error handling in builder
    _, err = NewQuery("users").Limit(-1).Build()
    fmt.Printf("  Invalid: %v\n", err)
    
    // Decorator Pattern
    fmt.Println("\n=== Decorator Pattern ===")
    
    processor := WithLogging("pipeline",
        WithValidation(1024,
            WithCompression(
                &jsonProcessor{},
            ),
        ),
    )
    
    data := []byte("{\"name\":\"Alice\",\"age\":30}")
    result, err := processor.Process(data)
    if err != nil {
        fmt.Printf("  Error: %v\n", err)
    } else {
        fmt.Printf("  Result: %d bytes\n", len(result))
    }
    
    // Adapter Pattern
    fmt.Println("\n=== Adapter Pattern ===")
    
    oldLog := &OldLogger{}
    logger := AdaptLogger(oldLog)
    
    logger.Info("server started")
    logger.Warn("high memory usage")
    logger.Error("connection failed")
}`,
				},
				{
					Title: "Behavioral and Concurrency Patterns",
					Content: `Behavioral patterns define communication between objects. Go's channels and goroutines enable elegant concurrency patterns that would be complex in other languages.

**Strategy Pattern:**
` + "```" + `
In Go, strategy is typically just an interface or function type:

  // Function type as strategy
  type SortStrategy func([]int) []int
  
  func BubbleSort(data []int) []int { ... }
  func QuickSort(data []int) []int { ... }
  func MergeSort(data []int) []int { ... }
  
  type Sorter struct {
      strategy SortStrategy
  }
  
  func (s *Sorter) Sort(data []int) []int {
      return s.strategy(data)
  }
  
  // Swap strategy at runtime:
  sorter := &Sorter{strategy: QuickSort}
  if len(data) < 10 {
      sorter.strategy = BubbleSort // Use simpler for small data
  }

  // Or just pass the function directly:
  func ProcessData(data []int, sort func([]int) []int) {
      sorted := sort(data)
      ...
  }
` + "```" + `

**Observer Pattern (Event Emitter):**
` + "```" + `
Go's channels naturally implement observer:

  type EventType string
  
  type Emitter struct {
      listeners map[EventType][]chan any
      mu        sync.RWMutex
  }
  
  func (e *Emitter) On(event EventType) <-chan any {
      ch := make(chan any, 1)
      e.mu.Lock()
      e.listeners[event] = append(e.listeners[event], ch)
      e.mu.Unlock()
      return ch
  }
  
  func (e *Emitter) Emit(event EventType, data any) {
      e.mu.RLock()
      for _, ch := range e.listeners[event] {
          select {
          case ch <- data:
          default: // Don't block if listener is slow
          }
      }
      e.mu.RUnlock()
  }
` + "```" + `

**Pipeline Pattern:**
` + "```" + `
Chain processing stages with channels:

  func generate(nums ...int) <-chan int {
      out := make(chan int)
      go func() {
          for _, n := range nums {
              out <- n
          }
          close(out)
      }()
      return out
  }
  
  func square(in <-chan int) <-chan int {
      out := make(chan int)
      go func() {
          for n := range in {
              out <- n * n
          }
          close(out)
      }()
      return out
  }
  
  func filter(in <-chan int, pred func(int) bool) <-chan int {
      out := make(chan int)
      go func() {
          for n := range in {
              if pred(n) { out <- n }
          }
          close(out)
      }()
      return out
  }
  
  // Compose pipeline:
  nums := generate(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  squares := square(nums)
  evens := filter(squares, func(n int) bool { return n%2 == 0 })
  for v := range evens { fmt.Println(v) }

Pipeline benefits:
  - Each stage is independent (can run on different cores)
  - Backpressure is natural (channel buffering)
  - Easy to add/remove stages
  - Memory efficient (streaming, not buffering all data)
` + "```" + `

**Fan-Out / Fan-In:**
` + "```" + `
Fan-out: distribute work across multiple goroutines
Fan-in: collect results from multiple goroutines

  func fanOut(in <-chan int, workers int) []<-chan int {
      channels := make([]<-chan int, workers)
      for i := 0; i < workers; i++ {
          channels[i] = worker(in) // Each reads from same input
      }
      return channels
  }
  
  func fanIn(channels ...<-chan int) <-chan int {
      var wg sync.WaitGroup
      out := make(chan int)
      
      for _, ch := range channels {
          wg.Add(1)
          go func(c <-chan int) {
              defer wg.Done()
              for v := range c { out <- v }
          }(ch)
      }
      
      go func() {
          wg.Wait()
          close(out)
      }()
      return out
  }
  
  // Usage:
  input := generate(1, 2, 3, ..., 1000)
  workers := fanOut(input, runtime.NumCPU())
  results := fanIn(workers...)
  for r := range results { process(r) }

When to use:
  Fan-out: CPU-bound work, each item independent
  Fan-in: Aggregate results from parallel computations
  
  Real examples:
    - Web scraper: fan-out URLs to workers
    - Map-reduce: fan-out map, fan-in reduce
    - Load testing: fan-out requests, fan-in metrics
` + "```" + `

**Context Pattern:**
` + "```" + `
Context for cancellation, deadlines, and request-scoped values:

  func processOrder(ctx context.Context, orderID string) error {
      // Check if already cancelled
      select {
      case <-ctx.Done():
          return ctx.Err()
      default:
      }
      
      // Pass context to downstream calls
      user, err := getUserFromDB(ctx, orderID)
      if err != nil { return err }
      
      // Long operation with cancellation check
      for _, item := range order.Items {
          select {
          case <-ctx.Done():
              return ctx.Err() // Cancelled or deadline exceeded
          default:
              processItem(ctx, item)
          }
      }
      return nil
  }

Context rules:
  1. First parameter, named ctx
  2. Never store in a struct
  3. Pass from request handler to all downstream calls
  4. Cancel contexts when done (defer cancel())
  5. Don't pass nil context (use context.TODO())
` + "```" + ``,
					CodeExamples: `// Behavioral and concurrency patterns
package main

import (
    "context"
    "fmt"
    "math/rand"
    "sort"
    "sync"
    "time"
)

// Strategy Pattern
type CompressStrategy func([]byte) []byte

func NoCompression(data []byte) []byte { return data }

func SimpleCompress(data []byte) []byte {
    // Simulated: just return half
    if len(data) > 1 {
        return data[:len(data)/2]
    }
    return data
}

type FileWriter struct {
    compress CompressStrategy
}

func (fw *FileWriter) Write(data []byte) []byte {
    return fw.compress(data)
}

// Observer Pattern
type EventType string

const (
    UserCreated  EventType = "user.created"
    UserDeleted  EventType = "user.deleted"
    OrderPlaced  EventType = "order.placed"
)

type EventData struct {
    Type    EventType
    Payload map[string]string
}

type EventEmitter struct {
    mu        sync.RWMutex
    listeners map[EventType][]func(EventData)
}

func NewEventEmitter() *EventEmitter {
    return &EventEmitter{
        listeners: make(map[EventType][]func(EventData)),
    }
}

func (e *EventEmitter) On(eventType EventType, handler func(EventData)) {
    e.mu.Lock()
    defer e.mu.Unlock()
    e.listeners[eventType] = append(e.listeners[eventType], handler)
}

func (e *EventEmitter) Emit(event EventData) {
    e.mu.RLock()
    handlers := e.listeners[event.Type]
    e.mu.RUnlock()
    
    for _, h := range handlers {
        h(event)
    }
}

// Pipeline Pattern
func generate(ctx context.Context, nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for _, n := range nums {
            select {
            case out <- n:
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}

func square(ctx context.Context, in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            select {
            case out <- n * n:
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}

func filterEven(ctx context.Context, in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            if n%2 == 0 {
                select {
                case out <- n:
                case <-ctx.Done():
                    return
                }
            }
        }
    }()
    return out
}

// Fan-Out / Fan-In
func fanOut(ctx context.Context, in <-chan int, workers int) []<-chan int {
    channels := make([]<-chan int, workers)
    for i := 0; i < workers; i++ {
        channels[i] = worker(ctx, in, i)
    }
    return channels
}

func worker(ctx context.Context, in <-chan int, id int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            // Simulate work
            result := n * n
            select {
            case out <- result:
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}

func fanIn(ctx context.Context, channels ...<-chan int) <-chan int {
    var wg sync.WaitGroup
    out := make(chan int)
    
    for _, ch := range channels {
        wg.Add(1)
        go func(c <-chan int) {
            defer wg.Done()
            for v := range c {
                select {
                case out <- v:
                case <-ctx.Done():
                    return
                }
            }
        }(ch)
    }
    
    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

// Semaphore pattern (bounded concurrency)
type Semaphore struct {
    ch chan struct{}
}

func NewSemaphore(max int) *Semaphore {
    return &Semaphore{ch: make(chan struct{}, max)}
}

func (s *Semaphore) Acquire() { s.ch <- struct{}{} }
func (s *Semaphore) Release() { <-s.ch }

// Worker Pool pattern
type Job struct {
    ID   int
    Data int
}

type Result struct {
    JobID  int
    Output int
}

func workerPool(ctx context.Context, jobs <-chan Job, results chan<- Result, numWorkers int) {
    var wg sync.WaitGroup
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()
            for job := range jobs {
                // Process job
                output := job.Data * job.Data
                select {
                case results <- Result{JobID: job.ID, Output: output}:
                case <-ctx.Done():
                    return
                }
            }
        }(i)
    }
    go func() {
        wg.Wait()
        close(results)
    }()
}

func main() {
    // Strategy Pattern
    fmt.Println("=== Strategy Pattern ===")
    
    data := []byte("Hello, World! This is some data to compress.")
    
    writer := &FileWriter{compress: NoCompression}
    result := writer.Write(data)
    fmt.Printf("  No compression: %d bytes → %d bytes\n", len(data), len(result))
    
    writer.compress = SimpleCompress
    result = writer.Write(data)
    fmt.Printf("  Simple compress: %d bytes → %d bytes\n", len(data), len(result))
    
    // Sort strategies
    numbers := []int{5, 3, 8, 1, 9, 2, 7, 4, 6}
    
    strategies := map[string]func([]int){
        "ascending":  func(s []int) { sort.Ints(s) },
        "descending": func(s []int) { sort.Sort(sort.Reverse(sort.IntSlice(s))) },
    }
    
    for name, strategy := range strategies {
        nums := make([]int, len(numbers))
        copy(nums, numbers)
        strategy(nums)
        fmt.Printf("  %s: %v\n", name, nums)
    }
    
    // Observer Pattern
    fmt.Println("\n=== Observer Pattern ===")
    
    emitter := NewEventEmitter()
    
    emitter.On(UserCreated, func(e EventData) {
        fmt.Printf("  [Email] Welcome email to %s\n", e.Payload["name"])
    })
    emitter.On(UserCreated, func(e EventData) {
        fmt.Printf("  [Analytics] New user: %s\n", e.Payload["name"])
    })
    emitter.On(OrderPlaced, func(e EventData) {
        fmt.Printf("  [Inventory] Reserve items for order %s\n", e.Payload["order_id"])
    })
    
    emitter.Emit(EventData{
        Type:    UserCreated,
        Payload: map[string]string{"name": "Alice", "email": "alice@example.com"},
    })
    emitter.Emit(EventData{
        Type:    OrderPlaced,
        Payload: map[string]string{"order_id": "ord-123", "total": "99.99"},
    })
    
    // Pipeline Pattern
    fmt.Println("\n=== Pipeline Pattern ===")
    
    ctx := context.Background()
    nums := generate(ctx, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    squares := square(ctx, nums)
    evens := filterEven(ctx, squares)
    
    fmt.Print("  Even squares: ")
    for v := range evens {
        fmt.Printf("%d ", v)
    }
    fmt.Println()
    
    // Fan-Out / Fan-In
    fmt.Println("\n=== Fan-Out / Fan-In ===")
    
    input := make(chan int, 20)
    go func() {
        for i := 1; i <= 20; i++ {
            input <- i
        }
        close(input)
    }()
    
    workers := fanOut(ctx, input, 4)
    results2 := fanIn(ctx, workers...)
    
    var allResults []int
    for r := range results2 {
        allResults = append(allResults, r)
    }
    sort.Ints(allResults)
    fmt.Printf("  Results (%d items): %v\n", len(allResults), allResults)
    
    // Worker Pool
    fmt.Println("\n=== Worker Pool ===")
    
    jobs := make(chan Job, 10)
    poolResults := make(chan Result, 10)
    
    workerPool(ctx, jobs, poolResults, 3)
    
    // Submit jobs
    go func() {
        for i := 1; i <= 10; i++ {
            jobs <- Job{ID: i, Data: rand.Intn(100)}
        }
        close(jobs)
    }()
    
    // Collect results
    for r := range poolResults {
        fmt.Printf("  Job %d: result=%d\n", r.JobID, r.Output)
    }
    
    // Semaphore (bounded concurrency)
    fmt.Println("\n=== Semaphore (max 3 concurrent) ===")
    
    sem := NewSemaphore(3)
    var wg sync.WaitGroup
    
    for i := 0; i < 8; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            sem.Acquire()
            defer sem.Release()
            fmt.Printf("  Worker %d: processing\n", id)
            time.Sleep(10 * time.Millisecond)
        }(i)
    }
    wg.Wait()
    
    // Context cancellation
    fmt.Println("\n=== Context Cancellation ===")
    
    ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
    defer cancel()
    
    ch := generate(ctx, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    count := 0
    for range square(ctx, ch) {
        count++
        time.Sleep(10 * time.Millisecond)
    }
    fmt.Printf("  Processed %d items before timeout\n", count)
}`,
				},
			},
		},
	})
}
