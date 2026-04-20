package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          2327,
			Title:       "Cloud-Native Architecture Patterns",
			Description: "Design cloud-native applications with the twelve-factor app methodology, serverless architecture, service mesh, and cloud design patterns.",
			Order:       27,
			Lessons: []problems.Lesson{
				{
					Title: "Cloud-Native Design Principles",
					Content: `Cloud-native architecture leverages cloud computing advantages: elasticity, distributed computing, and managed services.

**Twelve-Factor App Methodology:**

1. Codebase:
   One codebase tracked in version control, many deploys
   One repo per app, shared code extracted into libraries
   Same codebase for dev, staging, production

2. Dependencies:
   Explicitly declare and isolate dependencies
   Never rely on system-wide packages
   Use dependency managers (go.mod, package.json, requirements.txt)
   Vendoring for reproducibility

3. Config:
   Store config in the environment
   Strict separation of config from code
   Config varies between deploys, code does not
   Use environment variables (DATABASE_URL, API_KEY)

4. Backing Services:
   Treat backing services as attached resources
   Database, cache, queue, email are all attached resources
   Swap local PostgreSQL for Amazon RDS without code changes
   URL/credentials in config, not code

5. Build, Release, Run:
   Strictly separate build and run stages
   Build: Convert code into executable bundle
   Release: Combine build with config
   Run: Execute the app in the environment
   Every release has a unique ID (timestamp, version)

6. Processes:
   Execute the app as one or more stateless processes
   Any persistent data in stateful backing services
   No sticky sessions (use external session store)
   Share-nothing architecture

7. Port Binding:
   Export services via port binding
   App is self-contained, does not rely on runtime injection
   HTTP server built into the app
   One app can become another app's backing service

8. Concurrency:
   Scale out via the process model
   Different process types for different workloads
   Web processes for HTTP requests
   Worker processes for background jobs
   Horizontal scaling, not vertical

9. Disposability:
   Maximize robustness with fast startup and graceful shutdown
   Processes can be started or stopped at any time
   Graceful shutdown: stop accepting new work, finish current
   Crash-only design: assume processes will be killed

10. Dev/Prod Parity:
    Keep development, staging, and production as similar as possible
    Same backing services in all environments
    Use containers for environment consistency
    Deploy frequently to minimize drift

11. Logs:
    Treat logs as event streams
    App never concerns itself with routing or storing logs
    Write to stdout, let the environment handle aggregation
    Log aggregation: ELK, Loki, CloudWatch

12. Admin Processes:
    Run admin/management tasks as one-off processes
    Database migrations, console REPL, one-time scripts
    Same codebase and config as the app
    Run in identical environment

**Serverless Architecture:**

Function-as-a-Service (FaaS):
  Write functions, cloud handles infrastructure
  Pay only for execution time
  Auto-scales from zero to thousands
  Services: AWS Lambda, Google Cloud Functions, Azure Functions
  
  Best for:
    Event-driven processing
    API backends with variable traffic
    Data transformation pipelines
    Scheduled tasks
    Webhooks and integrations

  Challenges:
    Cold starts (latency on first invocation)
    Execution time limits (15 min for Lambda)
    State management (external state required)
    Debugging and testing complexity
    Vendor lock-in

  Patterns:
    API Gateway + Lambda: REST/GraphQL APIs
    Event Processing: S3 event -> Lambda -> DynamoDB
    Fan-out: SNS -> Multiple Lambdas
    Choreography: EventBridge -> Step Functions

Backend-as-a-Service (BaaS):
  Use managed services instead of custom backends
  Authentication: Auth0, Firebase Auth, Cognito
  Database: DynamoDB, Firestore, Supabase
  Storage: S3, Cloud Storage
  API: AppSync, Hasura

**Service Mesh:**

  Infrastructure layer for service-to-service communication
  Handles: Load balancing, encryption, observability, retry
  Sidecar proxy pattern (Envoy)
  
  Architecture:
    Data Plane: Sidecar proxies (handle actual traffic)
    Control Plane: Configuration and policy management
    
    [Service A] <-> [Sidecar Proxy] <-mTLS-> [Sidecar Proxy] <-> [Service B]
                           |                        |
                           v                        v
                    [Control Plane (Istiod)]
  
  Features:
    Mutual TLS (mTLS): Automatic encryption between services
    Traffic Management: Canary deployments, A/B testing
    Observability: Automatic metrics, traces, logs
    Resilience: Retries, circuit breaking, timeouts
    Authorization: Service-to-service access policies
  
  Tools: Istio, Linkerd, Consul Connect

**Cloud Design Patterns:**

Ambassador Pattern:
  Helper service that sends network requests on behalf of a consumer
  Handle cross-cutting concerns: retries, logging, routing
  Deployed as a sidecar container

Sidecar Pattern:
  Deploy helper components alongside the main application
  Logging agent, monitoring agent, configuration watcher
  Independent lifecycle from main application
  Shares same lifecycle as parent application

Gateway Aggregation:
  Aggregate multiple service calls into a single request
  Reduce client-to-server round trips
  Backend for Frontend (BFF) pattern
  API Gateway composes responses

Queue-Based Load Leveling:
  Use a queue between task producer and consumer
  Absorb traffic spikes
  Process at a steady rate
  Prevent overwhelming downstream services
  
  [Client] -> [Queue (SQS)] -> [Worker Pool]
  
  Benefits:
    Handles burst traffic gracefully
    Decouples producer from consumer
    Workers process at their own pace
    Automatic retry for failed messages

Competing Consumers:
  Multiple consumers read from the same queue
  Scale consumers independently based on queue depth
  Ensures messages are processed exactly once (with dedup)

**Immutable Infrastructure:**

  Infrastructure never modified after deployment
  Instead, replace with new version
  
  Mutable (bad):
    Deploy server -> SSH -> Install updates -> Configure
    Configuration drift over time
    "Works on my machine" problems
    
  Immutable (good):
    Build image (AMI, container) -> Deploy new instances -> Destroy old
    Every deployment is identical
    Reproducible environments
    Easy rollback (deploy previous image)
  
  Implementation:
    Containers: Docker images, never modify running containers
    VMs: Golden images (AMI), rebuild for changes
    Infrastructure as Code: Terraform, Pulumi
    GitOps: Git as source of truth for infrastructure`,
					CodeExamples: `// Cloud-native architecture patterns

// Twelve-factor config from environment
type Config struct {
    Port        int    "envconfig:\"PORT\" default:\"8080\""
    DatabaseURL string "envconfig:\"DATABASE_URL\" required:\"true\""
    RedisURL    string "envconfig:\"REDIS_URL\" required:\"true\""
    LogLevel    string "envconfig:\"LOG_LEVEL\" default:\"info\""
    Environment string "envconfig:\"ENVIRONMENT\" default:\"development\""
    JWTSecret   string "envconfig:\"JWT_SECRET\" required:\"true\""
}

func LoadConfig() (*Config, error) {
    cfg := &Config{}
    
    cfg.Port = getEnvInt("PORT", 8080)
    cfg.DatabaseURL = mustGetEnv("DATABASE_URL")
    cfg.RedisURL = mustGetEnv("REDIS_URL")
    cfg.LogLevel = getEnv("LOG_LEVEL", "info")
    cfg.Environment = getEnv("ENVIRONMENT", "development")
    cfg.JWTSecret = mustGetEnv("JWT_SECRET")
    
    return cfg, nil
}

func getEnv(key, defaultVal string) string {
    if val := os.Getenv(key); val != "" {
        return val
    }
    return defaultVal
}

func getEnvInt(key string, defaultVal int) int {
    val := os.Getenv(key)
    if val == "" {
        return defaultVal
    }
    n, err := strconv.Atoi(val)
    if err != nil {
        return defaultVal
    }
    return n
}

func mustGetEnv(key string) string {
    val := os.Getenv(key)
    if val == "" {
        panic(fmt.Sprintf("required environment variable %s is not set", key))
    }
    return val
}

// Graceful shutdown (Factor 9: Disposability)
type Server struct {
    httpServer *http.Server
    db         *sql.DB
    redis      *redis.Client
    logger     *Logger
}

func (s *Server) Start(ctx context.Context) error {
    // Start HTTP server
    go func() {
        s.logger.Info("server starting", "port", s.httpServer.Addr)
        if err := s.httpServer.ListenAndServe(); err != http.ErrServerClosed {
            s.logger.Error("server error", "error", err)
        }
    }()
    
    // Wait for shutdown signal
    <-ctx.Done()
    
    s.logger.Info("shutdown signal received")
    
    // Graceful shutdown with timeout
    shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    // Stop accepting new requests
    if err := s.httpServer.Shutdown(shutdownCtx); err != nil {
        s.logger.Error("http shutdown error", "error", err)
    }
    
    // Close database connections
    if err := s.db.Close(); err != nil {
        s.logger.Error("database close error", "error", err)
    }
    
    // Close Redis
    if err := s.redis.Close(); err != nil {
        s.logger.Error("redis close error", "error", err)
    }
    
    s.logger.Info("server stopped gracefully")
    return nil
}

func main() {
    ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
    defer stop()
    
    cfg, err := LoadConfig()
    if err != nil {
        log.Fatal(err)
    }
    
    server := NewServer(cfg)
    if err := server.Start(ctx); err != nil {
        log.Fatal(err)
    }
}

// Health and readiness probes (Kubernetes)
type ProbeHandler struct {
    checks []ReadinessCheck
}

type ReadinessCheck struct {
    Name  string
    Check func(ctx context.Context) error
}

func (h *ProbeHandler) Liveness(w http.ResponseWriter, r *http.Request) {
    w.WriteHeader(http.StatusOK)
    w.Write([]byte("ok"))
}

func (h *ProbeHandler) Readiness(w http.ResponseWriter, r *http.Request) {
    ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
    defer cancel()
    
    results := make(map[string]string)
    allHealthy := true
    
    for _, check := range h.checks {
        if err := check.Check(ctx); err != nil {
            results[check.Name] = err.Error()
            allHealthy = false
        } else {
            results[check.Name] = "ok"
        }
    }
    
    response, _ := json.Marshal(results)
    
    if allHealthy {
        w.WriteHeader(http.StatusOK)
    } else {
        w.WriteHeader(http.StatusServiceUnavailable)
    }
    w.Header().Set("Content-Type", "application/json")
    w.Write(response)
}

// Queue-based load leveling
type QueueProcessor struct {
    queue       MessageQueue
    handler     MessageHandler
    concurrency int
    logger      *Logger
}

type MessageQueue interface {
    Receive(ctx context.Context, maxMessages int) ([]Message, error)
    Delete(ctx context.Context, receiptHandle string) error
}

type Message struct {
    ID            string
    Body          string
    ReceiptHandle string
    Attributes    map[string]string
}

type MessageHandler interface {
    Handle(ctx context.Context, msg Message) error
}

func NewQueueProcessor(queue MessageQueue, handler MessageHandler, concurrency int) *QueueProcessor {
    return &QueueProcessor{
        queue:       queue,
        handler:     handler,
        concurrency: concurrency,
    }
}

func (p *QueueProcessor) Start(ctx context.Context) error {
    sem := make(chan struct{}, p.concurrency)
    
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
        }
        
        messages, err := p.queue.Receive(ctx, 10)
        if err != nil {
            p.logger.Error("receive error", "error", err)
            continue
        }
        
        for _, msg := range messages {
            msg := msg
            sem <- struct{}{} // Acquire semaphore
            
            go func() {
                defer func() { <-sem }() // Release semaphore
                
                if err := p.handler.Handle(ctx, msg); err != nil {
                    p.logger.Error("handler error",
                        "message_id", msg.ID,
                        "error", err,
                    )
                    return
                }
                
                if err := p.queue.Delete(ctx, msg.ReceiptHandle); err != nil {
                    p.logger.Error("delete error",
                        "message_id", msg.ID,
                        "error", err,
                    )
                }
            }()
        }
    }
}

// Gateway aggregation pattern
type APIGateway struct {
    userService    UserServiceClient
    orderService   OrderServiceClient
    productService ProductServiceClient
    logger         *Logger
}

type UserProfile struct {
    User         *UserDTO           "json:\"user\""
    RecentOrders []*OrderSummaryDTO "json:\"recent_orders\""
    Wishlist     []*ProductDTO      "json:\"wishlist\""
}

func (g *APIGateway) GetUserProfile(ctx context.Context, userID string) (*UserProfile, error) {
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()
    
    // Parallel calls to multiple services
    var (
        user   *UserDTO
        orders []*OrderSummaryDTO
        wishlist []*ProductDTO
        errs   = make(chan error, 3)
    )
    
    go func() {
        var err error
        user, err = g.userService.GetUser(ctx, userID)
        errs <- err
    }()
    
    go func() {
        var err error
        orders, err = g.orderService.GetRecentOrders(ctx, userID, 5)
        errs <- err
    }()
    
    go func() {
        var err error
        wishlist, err = g.productService.GetWishlist(ctx, userID)
        errs <- err
    }()
    
    // Collect results
    for i := 0; i < 3; i++ {
        if err := <-errs; err != nil {
            g.logger.Warn("partial failure in profile aggregation",
                "user_id", userID,
                "error", err,
            )
            // Continue with partial data (graceful degradation)
        }
    }
    
    return &UserProfile{
        User:         user,
        RecentOrders: orders,
        Wishlist:     wishlist,
    }, nil
}`,
				},
			},
		},
		{
			ID:          2328,
			Title:       "Performance Architecture and Optimization",
			Description: "Design high-performance systems with capacity planning, performance modeling, load testing strategies, and optimization techniques at the architecture level.",
			Order:       28,
			Lessons: []problems.Lesson{
				{
					Title: "Architecting for Performance",
					Content: `Performance is an architectural concern that must be designed in, not optimized later.

**Performance Requirements:**

Response Time:
  Interactive (web): < 200ms (p50), < 1s (p99)
  API: < 100ms (p50), < 500ms (p99)  
  Batch: Depends on SLA (minutes to hours)
  Real-time: < 10ms

Throughput:
  Requests per second the system can handle
  Peak vs sustained throughput
  Growth projections

Resource Utilization:
  CPU: Target 60-70% sustained
  Memory: Target 70-80% with headroom
  Network: Monitor bandwidth and latency
  Disk I/O: SSD vs HDD, IOPS requirements

**Capacity Planning:**

  Determine resource requirements for target load
  
  Steps:
    1. Define performance objectives (SLOs)
    2. Measure current performance baseline
    3. Project future growth (users, data, traffic)
    4. Model resource requirements
    5. Plan for peak load (2-3x average)
    6. Add headroom (20-30%)
    
  Little's Law: L = lambda * W
    L = average number of items in system
    lambda = average arrival rate
    W = average time in system
    
  Example:
    If 1000 requests/second arrive and each takes 100ms:
    L = 1000 * 0.1 = 100 concurrent requests
    Need enough instances to handle 100+ concurrent requests

  Universal Scalability Law (USL):
    Throughput = N / (1 + sigma*(N-1) + kappa*N*(N-1))
    sigma = contention penalty
    kappa = coherency penalty
    N = number of processors/nodes
    
    Contention: Serialization points (locks, queues)
    Coherency: Coordination overhead (cache invalidation)

**Caching Architecture:**

Cache Hierarchy:
  L1: In-process cache (HashMap, LRU)
    Latency: < 1 microsecond
    Size: MBs
    Invalidation: Simple, per-process
    
  L2: Distributed cache (Redis, Memcached)
    Latency: < 1 millisecond
    Size: GBs
    Invalidation: Shared, more complex
    
  L3: CDN (CloudFront, Cloudflare)
    Latency: < 50ms (depends on edge location)
    Size: TBs
    Invalidation: TTL-based, purge APIs

Cache Strategies:
  Cache-Aside (Lazy Loading):
    App checks cache first
    On miss, load from database and populate cache
    Most common pattern
    
  Read-Through:
    Cache loads from database automatically on miss
    App only talks to cache
    
  Write-Through:
    Write to cache and database synchronously
    Ensures cache is always up-to-date
    Higher write latency
    
  Write-Behind (Write-Back):
    Write to cache, asynchronously write to database
    Lower write latency
    Risk of data loss if cache fails
    
  Refresh-Ahead:
    Proactively refresh cache before expiration
    Reduces cache miss latency
    Wastes resources on unused entries

Cache Invalidation Strategies:
  TTL (Time-to-Live): Simple, eventual consistency
  Event-based: Invalidate on data change events
  Version-based: Cache key includes version number

**Connection Pooling:**

  Reuse expensive connections (database, HTTP)
  
  Database Pool:
    Min connections: Keep warm for baseline load
    Max connections: Limit to prevent overload
    Idle timeout: Close unused connections
    Max lifetime: Prevent stale connections
    
  HTTP Client Pool:
    Keep-alive connections
    Max connections per host
    Idle connection timeout
    
  Sizing:
    Pool size = Peak concurrent requests / Avg request time
    Example: 200 req/s * 50ms/req = 10 connections
    Add headroom: 15-20 connections

**Async Processing:**

  Move non-critical work off the request path
  
  Patterns:
    Fire-and-forget: Send to queue, respond immediately
    Request-reply: Send to queue, poll for result
    Webhook callback: Send to queue, callback when done
    
  Examples:
    Email sending
    Report generation
    Image processing
    Analytics events
    Notification delivery

**Database Performance:**

  Indexing Strategy:
    Index frequently queried columns
    Composite indexes for multi-column queries
    Cover indexes to avoid table lookups
    Avoid over-indexing (slows writes)
    
  Query Optimization:
    Use EXPLAIN ANALYZE to understand query plans
    Avoid N+1 queries (use JOINs or batch loading)
    Pagination with cursors instead of OFFSET
    Denormalize for read-heavy workloads
    
  Read Replicas:
    Route reads to replicas
    Route writes to primary
    Replication lag awareness
    Eventual consistency trade-off

  Partitioning:
    Horizontal: Distribute rows across partitions
    By range (date), hash (user ID), or list
    Reduces query scope
    Enables parallel processing

**Load Testing:**

Types:
  Load Test: Expected load for sustained period
  Stress Test: Beyond expected load to find limits
  Spike Test: Sudden increase in traffic
  Soak Test: Sustained load for extended period (memory leaks)
  Breakpoint Test: Gradually increase until failure

Metrics to Monitor:
  Response time (p50, p95, p99)
  Error rate
  Throughput (requests/second)
  CPU, memory, network, disk usage
  Database connection pool utilization
  Queue depth and processing rate

Tools:
  k6: Modern load testing (JavaScript scripts)
  Locust: Python-based, distributed
  Gatling: Scala-based, detailed reports
  Apache JMeter: GUI-based, protocol support
  wrk/hey: Simple HTTP benchmarking`,
					CodeExamples: `// Performance architecture patterns

// In-process LRU cache
type LRUCache struct {
    mu       sync.Mutex
    capacity int
    items    map[string]*cacheItem
    order    *list.List
}

type cacheItem struct {
    key       string
    value     interface{}
    element   *list.Element
    expiresAt time.Time
}

func NewLRUCache(capacity int) *LRUCache {
    return &LRUCache{
        capacity: capacity,
        items:    make(map[string]*cacheItem),
        order:    list.New(),
    }
}

func (c *LRUCache) Get(key string) (interface{}, bool) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    item, ok := c.items[key]
    if !ok {
        return nil, false
    }
    
    if time.Now().After(item.expiresAt) {
        c.removeItem(item)
        return nil, false
    }
    
    c.order.MoveToFront(item.element)
    return item.value, true
}

func (c *LRUCache) Set(key string, value interface{}, ttl time.Duration) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    if existing, ok := c.items[key]; ok {
        existing.value = value
        existing.expiresAt = time.Now().Add(ttl)
        c.order.MoveToFront(existing.element)
        return
    }
    
    if c.order.Len() >= c.capacity {
        oldest := c.order.Back()
        if oldest != nil {
            c.removeItem(oldest.Value.(*cacheItem))
        }
    }
    
    item := &cacheItem{
        key:       key,
        value:     value,
        expiresAt: time.Now().Add(ttl),
    }
    item.element = c.order.PushFront(item)
    c.items[key] = item
}

func (c *LRUCache) removeItem(item *cacheItem) {
    c.order.Remove(item.element)
    delete(c.items, item.key)
}

// Connection pool
type ConnectionPool struct {
    mu          sync.Mutex
    connections chan *Connection
    factory     func() (*Connection, error)
    maxSize     int
    activeCount int
    maxIdleTime time.Duration
}

type Connection struct {
    conn      net.Conn
    createdAt time.Time
    lastUsed  time.Time
    pool      *ConnectionPool
}

func NewConnectionPool(factory func() (*Connection, error), maxSize int) *ConnectionPool {
    return &ConnectionPool{
        connections: make(chan *Connection, maxSize),
        factory:     factory,
        maxSize:     maxSize,
        maxIdleTime: 5 * time.Minute,
    }
}

func (p *ConnectionPool) Acquire(ctx context.Context) (*Connection, error) {
    // Try to get from pool
    select {
    case conn := <-p.connections:
        if time.Since(conn.lastUsed) > p.maxIdleTime {
            conn.conn.Close()
            p.mu.Lock()
            p.activeCount--
            p.mu.Unlock()
            return p.createNew(ctx)
        }
        return conn, nil
    default:
    }
    
    return p.createNew(ctx)
}

func (p *ConnectionPool) createNew(ctx context.Context) (*Connection, error) {
    p.mu.Lock()
    if p.activeCount >= p.maxSize {
        p.mu.Unlock()
        // Wait for available connection
        select {
        case conn := <-p.connections:
            return conn, nil
        case <-ctx.Done():
            return nil, ctx.Err()
        }
    }
    p.activeCount++
    p.mu.Unlock()
    
    conn, err := p.factory()
    if err != nil {
        p.mu.Lock()
        p.activeCount--
        p.mu.Unlock()
        return nil, err
    }
    conn.pool = p
    return conn, nil
}

func (p *ConnectionPool) Release(conn *Connection) {
    conn.lastUsed = time.Now()
    select {
    case p.connections <- conn:
    default:
        conn.conn.Close()
        p.mu.Lock()
        p.activeCount--
        p.mu.Unlock()
    }
}

// Async task processor
type TaskProcessor struct {
    queue    chan Task
    workers  int
    logger   *Logger
    metrics  *TaskMetrics
    wg       sync.WaitGroup
}

type Task struct {
    ID      string
    Type    string
    Payload []byte
    Handler func(context.Context, []byte) error
}

type TaskMetrics struct {
    processed int64
    failed    int64
    duration  *prometheus.HistogramVec
}

func NewTaskProcessor(workers, queueSize int) *TaskProcessor {
    return &TaskProcessor{
        queue:   make(chan Task, queueSize),
        workers: workers,
    }
}

func (p *TaskProcessor) Start(ctx context.Context) {
    for i := 0; i < p.workers; i++ {
        p.wg.Add(1)
        go func(workerID int) {
            defer p.wg.Done()
            
            for {
                select {
                case <-ctx.Done():
                    return
                case task, ok := <-p.queue:
                    if !ok {
                        return
                    }
                    p.processTask(ctx, task)
                }
            }
        }(i)
    }
}

func (p *TaskProcessor) processTask(ctx context.Context, task Task) {
    start := time.Now()
    
    err := task.Handler(ctx, task.Payload)
    
    duration := time.Since(start)
    
    if err != nil {
        atomic.AddInt64(&p.metrics.failed, 1)
        p.logger.Error("task failed",
            "task_id", task.ID,
            "type", task.Type,
            "duration", duration,
            "error", err,
        )
        return
    }
    
    atomic.AddInt64(&p.metrics.processed, 1)
    p.logger.Info("task completed",
        "task_id", task.ID,
        "type", task.Type,
        "duration", duration,
    )
}

func (p *TaskProcessor) Submit(task Task) error {
    select {
    case p.queue <- task:
        return nil
    default:
        return fmt.Errorf("task queue full")
    }
}

func (p *TaskProcessor) Stop() {
    close(p.queue)
    p.wg.Wait()
}

// Cursor-based pagination
type CursorPaginator struct {
    db        *sql.DB
    pageSize  int
}

type Page struct {
    Items      []interface{} "json:\"items\""
    NextCursor string        "json:\"next_cursor,omitempty\""
    HasMore    bool          "json:\"has_more\""
}

func (p *CursorPaginator) GetPage(ctx context.Context, cursor string, pageSize int) (*Page, error) {
    if pageSize <= 0 || pageSize > 100 {
        pageSize = 20
    }
    
    query := "SELECT id, name, created_at FROM items"
    args := []interface{}{}
    
    if cursor != "" {
        decodedCursor, err := decodeCursor(cursor)
        if err != nil {
            return nil, fmt.Errorf("invalid cursor: %w", err)
        }
        query += " WHERE (created_at, id) < ($1, $2)"
        args = append(args, decodedCursor.CreatedAt, decodedCursor.ID)
    }
    
    query += " ORDER BY created_at DESC, id DESC LIMIT $" + strconv.Itoa(len(args)+1)
    args = append(args, pageSize+1) // Fetch one extra to check hasMore
    
    rows, err := p.db.QueryContext(ctx, query, args...)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var items []interface{}
    for rows.Next() {
        var item struct {
            ID        string
            Name      string
            CreatedAt time.Time
        }
        if err := rows.Scan(&item.ID, &item.Name, &item.CreatedAt); err != nil {
            return nil, err
        }
        items = append(items, item)
    }
    
    page := &Page{}
    if len(items) > pageSize {
        page.HasMore = true
        items = items[:pageSize]
        lastItem := items[len(items)-1]
        page.NextCursor = encodeCursor(lastItem)
    }
    page.Items = items
    
    return page, nil
}`,
				},
			},
		},
	})
}
