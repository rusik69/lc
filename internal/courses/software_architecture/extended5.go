package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          2317,
			Title:       "Scalability and Performance Architecture",
			Description: "Design scalable systems with caching strategies, database scaling, load balancing, CDN architecture, and performance optimization patterns.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Caching Architecture and Strategies",
					Content: `Caching is the most impactful technique for improving system performance and reducing load on backends.

**Cache Levels:**

Client-Side Cache:
  Browser cache (HTTP cache headers)
  Service worker cache (PWA)
  Application state cache (Redux, local storage)
  DNS cache
  
  HTTP Cache Headers:
    Cache-Control: max-age=3600, public
    Cache-Control: no-cache (revalidate every time)
    Cache-Control: no-store (never cache)
    Cache-Control: private (browser only, not CDN)
    Cache-Control: stale-while-revalidate=86400
    ETag: "abc123" (content hash for conditional requests)
    Last-Modified: Fri, 01 Jan 2024 00:00:00 GMT

CDN Cache:
  Geographic distribution of cached content
  Static assets (JS, CSS, images, fonts)
  Dynamic page caching with short TTLs
  Edge computing for personalization
  
  CDN Invalidation:
    Path-based: Purge /api/products/*
    Tag-based: Purge all content tagged "products"
    Surrogate keys: Fine-grained invalidation

Application Cache:
  In-process cache (HashMap, LRU cache)
  Distributed cache (Redis, Memcached)
  Query result cache
  Computed value cache
  Session cache

Database Cache:
  Query cache (MySQL query cache)
  Buffer pool (InnoDB buffer pool)
  Connection pool cache
  Materialized views

**Caching Strategies:**

Cache-Aside (Lazy Loading):
  Application manages cache explicitly
  Read: Check cache -> miss -> read DB -> populate cache
  Write: Update DB -> invalidate cache
  
  Pros: Only cache what's needed, cache failure doesn't break reads
  Cons: Initial request slow (cold cache), possible stale data

Read-Through:
  Cache sits between app and database
  Cache loads data on miss automatically
  App only talks to cache

Write-Through:
  Write to cache and database synchronously
  Ensures cache and DB are always consistent
  Higher write latency

Write-Behind (Write-Back):
  Write to cache, asynchronously write to database
  Lower write latency
  Risk of data loss if cache fails before DB write

Refresh-Ahead:
  Proactively refresh cache before expiration
  Based on access patterns prediction
  Reduces cache miss latency for popular items

**Cache Invalidation (The Hard Problem):**

Time-based (TTL):
  Simple, predictable
  Trade-off: shorter TTL = more freshness, more load
  Good for: News feeds, product catalogs, analytics

Event-based:
  Invalidate when data changes
  Use domain events or CDC (Change Data Capture)
  Good for: User profiles, inventory, prices

Version-based:
  Include version in cache key
  Changing version effectively invalidates
  Good for: Configuration, static assets

Pattern-based:
  Invalidate by key pattern
  Good for: Related data (all user's orders)

**Cache Consistency Patterns:**

Cache stampede prevention:
  Multiple requests hit cold cache simultaneously
  All request the same expensive data from DB
  Solutions:
    Mutex/lock: Only one request fetches, others wait
    Probabilistic early expiration: Refresh before TTL
    Pre-warming: Populate cache before traffic arrives

Thundering herd:
  Popular cache key expires
  All waiting requests hit backend simultaneously
  Solutions:
    Staggered TTLs (add random jitter)
    Background refresh before expiration
    Circuit breaker on backend

Hot key problem:
  Single cache key receives disproportionate traffic
  Solutions:
    Shard hot keys across multiple cache nodes
    Local (in-process) cache for hottest keys
    Read replicas for hot keys`,
					CodeExamples: `// Caching implementation patterns

// Multi-level cache with type safety
type Cache[T any] struct {
    local      *LRUCache[string, T]
    distributed DistributedCache
    loader     func(ctx context.Context, key string) (T, error)
    ttl        time.Duration
    mu         sync.Map // per-key mutex for stampede prevention
}

func NewCache[T any](
    localSize int,
    distributed DistributedCache,
    ttl time.Duration,
    loader func(ctx context.Context, key string) (T, error),
) *Cache[T] {
    return &Cache[T]{
        local:       NewLRUCache[string, T](localSize),
        distributed: distributed,
        loader:      loader,
        ttl:         ttl,
    }
}

func (c *Cache[T]) Get(ctx context.Context, key string) (T, error) {
    // Level 1: Local (in-process) cache
    if val, ok := c.local.Get(key); ok {
        return val, nil
    }

    // Level 2: Distributed cache (Redis)
    if c.distributed != nil {
        var val T
        err := c.distributed.Get(ctx, key, &val)
        if err == nil {
            c.local.Set(key, val)
            return val, nil
        }
    }

    // Cache miss: Load from source with stampede prevention
    return c.loadWithLock(ctx, key)
}

func (c *Cache[T]) loadWithLock(ctx context.Context, key string) (T, error) {
    // Get or create per-key lock
    lockI, _ := c.mu.LoadOrStore(key, &sync.Mutex{})
    lock := lockI.(*sync.Mutex)
    lock.Lock()
    defer lock.Unlock()

    // Double-check: another goroutine may have loaded it
    if val, ok := c.local.Get(key); ok {
        return val, nil
    }

    // Load from source
    val, err := c.loader(ctx, key)
    if err != nil {
        var zero T
        return zero, err
    }

    // Populate caches
    c.local.Set(key, val)
    if c.distributed != nil {
        // Add jitter to prevent thundering herd
        jitter := time.Duration(rand.Int63n(int64(c.ttl / 10)))
        c.distributed.Set(ctx, key, val, c.ttl+jitter)
    }

    return val, nil
}

func (c *Cache[T]) Invalidate(ctx context.Context, key string) error {
    c.local.Delete(key)
    if c.distributed != nil {
        return c.distributed.Delete(ctx, key)
    }
    return nil
}

func (c *Cache[T]) InvalidatePattern(ctx context.Context, pattern string) error {
    c.local.Clear()
    if c.distributed != nil {
        return c.distributed.DeletePattern(ctx, pattern)
    }
    return nil
}

// LRU Cache implementation
type LRUCache[K comparable, V any] struct {
    capacity int
    items    map[K]*list.Element
    order    *list.List
    mu       sync.RWMutex
}

type lruEntry[K comparable, V any] struct {
    key   K
    value V
}

func NewLRUCache[K comparable, V any](capacity int) *LRUCache[K, V] {
    return &LRUCache[K, V]{
        capacity: capacity,
        items:    make(map[K]*list.Element),
        order:    list.New(),
    }
}

func (c *LRUCache[K, V]) Get(key K) (V, bool) {
    c.mu.RLock()
    elem, ok := c.items[key]
    c.mu.RUnlock()
    if !ok {
        var zero V
        return zero, false
    }
    c.mu.Lock()
    c.order.MoveToFront(elem)
    c.mu.Unlock()
    return elem.Value.(*lruEntry[K, V]).value, true
}

func (c *LRUCache[K, V]) Set(key K, value V) {
    c.mu.Lock()
    defer c.mu.Unlock()

    if elem, ok := c.items[key]; ok {
        c.order.MoveToFront(elem)
        elem.Value.(*lruEntry[K, V]).value = value
        return
    }

    if c.order.Len() >= c.capacity {
        oldest := c.order.Back()
        if oldest != nil {
            c.order.Remove(oldest)
            delete(c.items, oldest.Value.(*lruEntry[K, V]).key)
        }
    }

    entry := &lruEntry[K, V]{key: key, value: value}
    elem := c.order.PushFront(entry)
    c.items[key] = elem
}

func (c *LRUCache[K, V]) Delete(key K) {
    c.mu.Lock()
    defer c.mu.Unlock()
    if elem, ok := c.items[key]; ok {
        c.order.Remove(elem)
        delete(c.items, key)
    }
}

func (c *LRUCache[K, V]) Clear() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.items = make(map[K]*list.Element)
    c.order.Init()
}

// Usage example
// productCache := NewCache[Product](
//     1000,
//     redisCache,
//     15 * time.Minute,
//     func(ctx context.Context, key string) (Product, error) {
//         return productRepo.FindByID(ctx, key)
//     },
// )`,
				},
				{
					Title: "Database Scaling and Data Architecture",
					Content: `Database scaling strategies enable systems to handle growing data volumes and query loads.

**Vertical Scaling (Scale Up):**
  Bigger hardware: more CPU, RAM, faster disks
  Simpler to implement
  Has physical limits
  Cost increases non-linearly

**Horizontal Scaling (Scale Out):**

Read Replicas:
  Primary handles writes, replicas handle reads
  Asynchronous replication (eventual consistency)
  Read-after-write concerns: read from primary after write
  Useful when read/write ratio is high (80%+ reads)
  
  Topology:
    Primary -> Replica 1
    Primary -> Replica 2
    Primary -> Replica 3

Sharding (Horizontal Partitioning):
  Split data across multiple database instances
  Each shard holds a subset of the data
  
  Sharding Strategies:
    Range-based: shard by ID ranges (1-1M, 1M-2M, ...)
      Pros: Simple, range queries within shard
      Cons: Hot spots, uneven distribution

    Hash-based: shard by hash(key) % num_shards
      Pros: Even distribution
      Cons: Range queries across shards, resharding is complex

    Directory-based: lookup table maps key to shard
      Pros: Flexible, easy to rebalance
      Cons: Directory is SPOF, additional lookups

    Geographic: shard by region/location
      Pros: Data locality, compliance
      Cons: Cross-region queries

  Challenges:
    Cross-shard joins: Avoid or denormalize
    Cross-shard transactions: Use sagas
    Rebalancing: Moving data between shards
    Schema changes: Must apply to all shards
    ID generation: Must be globally unique

Consistent Hashing:
  Minimizes data movement when adding/removing shards
  Hash ring with virtual nodes
  When shard added: only neighbors' data moves
  When shard removed: data distributed to neighbors

**Database Patterns:**

CQRS with Separate Databases:
  Write database: Normalized, optimized for writes
  Read database: Denormalized, optimized for queries
  Sync via events or CDC

Polyglot Persistence:
  Use different databases for different access patterns
  Relational: Transactional data, complex queries
  Document: Semi-structured data, flexible schema
  Key-Value: Caching, sessions, configuration
  Time-series: Metrics, logs, IoT data
  Graph: Relationships, social networks, recommendations
  Search: Full-text search, analytics

Connection Pooling:
  Reuse database connections
  Limit concurrent connections
  Track connection health
  Configuration:
    Min connections: Handle baseline load
    Max connections: Prevent resource exhaustion
    Idle timeout: Release unused connections
    Max lifetime: Prevent stale connections

Query Optimization:
  Indexing strategy:
    B-tree: Range queries, equality, sorting
    Hash: Equality only
    Composite: Multi-column queries
    Partial: Filter conditions (WHERE active = true)
    Covering: Include all query columns
  
  Query patterns:
    Avoid SELECT * (select only needed columns)
    Use pagination (LIMIT/OFFSET or cursor-based)
    Batch operations (bulk insert/update)
    Prepared statements (avoid SQL injection, improve perf)
    Explain/analyze queries for optimization

**Data Partitioning:**

Vertical Partitioning:
  Split table columns across tables
  Frequently accessed columns in one table
  Large/infrequently accessed columns in another
  Example: User table -> user_core + user_profile

Horizontal Partitioning:
  Split table rows across tables or databases
  Based on partition key
  PostgreSQL native partitioning by range, list, or hash
  Example: Orders partitioned by year

Archive Pattern:
  Move old data to cheaper storage
  Keep recent data in hot storage
  Archive query through separate path
  Example: Orders > 1 year old -> archive table`,
					CodeExamples: `// Database scaling patterns

// Connection pool with health checks
type DBPool struct {
    master   *sql.DB
    replicas []*sql.DB
    current  uint64
}

func NewDBPool(masterDSN string, replicaDSNs []string, maxConns int) (*DBPool, error) {
    master, err := sql.Open("postgres", masterDSN)
    if err != nil {
        return nil, err
    }
    master.SetMaxOpenConns(maxConns)
    master.SetMaxIdleConns(maxConns / 2)
    master.SetConnMaxLifetime(30 * time.Minute)
    master.SetConnMaxIdleTime(5 * time.Minute)

    replicas := make([]*sql.DB, len(replicaDSNs))
    for i, dsn := range replicaDSNs {
        replica, err := sql.Open("postgres", dsn)
        if err != nil {
            return nil, err
        }
        replica.SetMaxOpenConns(maxConns)
        replica.SetMaxIdleConns(maxConns / 2)
        replica.SetConnMaxLifetime(30 * time.Minute)
        replicas[i] = replica
    }

    return &DBPool{master: master, replicas: replicas}, nil
}

func (p *DBPool) Master() *sql.DB {
    return p.master
}

func (p *DBPool) Replica() *sql.DB {
    if len(p.replicas) == 0 {
        return p.master
    }
    // Round-robin selection
    idx := atomic.AddUint64(&p.current, 1) % uint64(len(p.replicas))
    return p.replicas[idx]
}

// Read-after-write consistency helper
type ReadWriteProxy struct {
    pool        *DBPool
    recentWrite sync.Map // key -> expiry time
    writeTTL    time.Duration
}

func (p *ReadWriteProxy) Write(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
    result, err := p.pool.Master().ExecContext(ctx, query, args...)
    return result, err
}

func (p *ReadWriteProxy) Read(ctx context.Context, key string, query string, args ...interface{}) (*sql.Rows, error) {
    // Check if recent write for this key exists
    if expiry, ok := p.recentWrite.Load(key); ok {
        if time.Now().Before(expiry.(time.Time)) {
            // Read from master for consistency
            return p.pool.Master().QueryContext(ctx, query, args...)
        }
        p.recentWrite.Delete(key)
    }
    // Read from replica
    return p.pool.Replica().QueryContext(ctx, query, args...)
}

func (p *ReadWriteProxy) MarkWritten(key string) {
    p.recentWrite.Store(key, time.Now().Add(p.writeTTL))
}

// Sharding implementation
type ShardRouter struct {
    shards    []*sql.DB
    numShards int
}

func NewShardRouter(dsns []string) (*ShardRouter, error) {
    shards := make([]*sql.DB, len(dsns))
    for i, dsn := range dsns {
        db, err := sql.Open("postgres", dsn)
        if err != nil {
            return nil, err
        }
        shards[i] = db
    }
    return &ShardRouter{shards: shards, numShards: len(dsns)}, nil
}

func (r *ShardRouter) GetShard(key string) *sql.DB {
    h := fnv.New32a()
    h.Write([]byte(key))
    idx := int(h.Sum32()) % r.numShards
    return r.shards[idx]
}

func (r *ShardRouter) QueryAll(ctx context.Context, query string, args ...interface{}) ([]map[string]interface{}, error) {
    type result struct {
        rows []map[string]interface{}
        err  error
    }
    
    results := make(chan result, r.numShards)
    for _, shard := range r.shards {
        go func(db *sql.DB) {
            rows, err := db.QueryContext(ctx, query, args...)
            if err != nil {
                results <- result{err: err}
                return
            }
            defer rows.Close()
            
            var data []map[string]interface{}
            cols, _ := rows.Columns()
            for rows.Next() {
                row := make(map[string]interface{})
                vals := make([]interface{}, len(cols))
                ptrs := make([]interface{}, len(cols))
                for i := range vals {
                    ptrs[i] = &vals[i]
                }
                rows.Scan(ptrs...)
                for i, col := range cols {
                    row[col] = vals[i]
                }
                data = append(data, row)
            }
            results <- result{rows: data}
        }(shard)
    }

    var allRows []map[string]interface{}
    for i := 0; i < r.numShards; i++ {
        r := <-results
        if r.err != nil {
            return nil, r.err
        }
        allRows = append(allRows, r.rows...)
    }
    return allRows, nil
}

// Cursor-based pagination (more efficient than OFFSET)
type Cursor struct {
    ID        string
    CreatedAt time.Time
}

func PaginateWithCursor(ctx context.Context, db *sql.DB, cursor *Cursor, limit int) ([]Order, *Cursor, error) {
    query := "SELECT id, customer_id, total, created_at FROM orders"
    args := []interface{}{}
    
    if cursor != nil {
        query += " WHERE (created_at, id) < ($1, $2)"
        args = append(args, cursor.CreatedAt, cursor.ID)
    }
    
    query += " ORDER BY created_at DESC, id DESC LIMIT $" + fmt.Sprintf("%d", len(args)+1)
    args = append(args, limit+1) // Fetch one extra to check if more exist
    
    rows, err := db.QueryContext(ctx, query, args...)
    if err != nil {
        return nil, nil, err
    }
    defer rows.Close()
    
    var orders []Order
    for rows.Next() {
        var o Order
        if err := rows.Scan(&o.ID, &o.CustomerID, &o.Total, &o.CreatedAt); err != nil {
            return nil, nil, err
        }
        orders = append(orders, o)
    }
    
    var nextCursor *Cursor
    if len(orders) > limit {
        last := orders[limit-1]
        nextCursor = &Cursor{ID: last.ID, CreatedAt: last.CreatedAt}
        orders = orders[:limit]
    }
    
    return orders, nextCursor, nil
}`,
				},
			},
		},
		{
			ID:          2318,
			Title:       "API Design and Versioning",
			Description: "Design robust APIs with RESTful principles, GraphQL, gRPC, versioning strategies, backward compatibility, and API governance.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "RESTful API Design Principles",
					Content: `Well-designed APIs are consistent, predictable, and evolve gracefully over time.

**REST API Design Guidelines:**

Resource Naming:
  Use nouns, not verbs: /orders not /getOrders
  Use plural: /users not /user
  Nested resources for relationships: /users/123/orders
  Use kebab-case: /order-items not /orderItems
  Avoid deep nesting: /users/123/orders is OK, /users/123/orders/456/items/789/details is too deep

HTTP Methods:
  GET /orders         - List orders (with pagination)
  GET /orders/123     - Get specific order
  POST /orders        - Create new order
  PUT /orders/123     - Replace entire order
  PATCH /orders/123   - Partial update
  DELETE /orders/123  - Delete order

  HEAD /orders/123    - Check if exists (no body)
  OPTIONS /orders     - Discover supported methods

Status Codes:
  2xx Success:
    200 OK - Successful GET, PUT, PATCH, DELETE
    201 Created - Successful POST (include Location header)
    202 Accepted - Request accepted for async processing
    204 No Content - Successful DELETE or PUT with no body

  3xx Redirection:
    301 Moved Permanently - Resource moved
    304 Not Modified - Cache still valid (conditional GET)

  4xx Client Error:
    400 Bad Request - Invalid input
    401 Unauthorized - Authentication required
    403 Forbidden - Authenticated but not authorized
    404 Not Found - Resource doesn't exist
    409 Conflict - State conflict (duplicate, version mismatch)
    422 Unprocessable Entity - Valid syntax but semantic errors
    429 Too Many Requests - Rate limited

  5xx Server Error:
    500 Internal Server Error - Unexpected failure
    502 Bad Gateway - Upstream service error
    503 Service Unavailable - Temporarily unavailable
    504 Gateway Timeout - Upstream timeout

Pagination:
  Offset-based: ?offset=20&limit=10
    Simple, allows jumping to any page
    Poor performance on large datasets (OFFSET is O(n))
    Inconsistent with concurrent writes
  
  Cursor-based: ?cursor=eyJpZCI6MTIzfQ&limit=10
    Consistent results with concurrent writes
    Better performance (keyset pagination)
    Cannot jump to arbitrary page
    Return Link headers or next_cursor in response

  Response format:
    {
      "data": [...],
      "pagination": {
        "total": 1000,
        "limit": 10,
        "next_cursor": "abc123",
        "has_more": true
      }
    }

Filtering, Sorting, and Field Selection:
  Filtering: ?status=active&min_total=100
  Sorting: ?sort=created_at:desc,total:asc
  Field selection: ?fields=id,name,total
  Search: ?q=searchterm

Error Response Format:
  {
    "error": {
      "code": "VALIDATION_ERROR",
      "message": "Invalid input data",
      "details": [
        {
          "field": "email",
          "message": "Must be a valid email address",
          "code": "INVALID_FORMAT"
        }
      ],
      "request_id": "req-abc-123"
    }
  }

**API Versioning Strategies:**

URL Path Versioning: /api/v1/orders
  Most visible and explicit
  Easy to route and test
  Breaking changes = new version
  Commonly used (GitHub, Stripe, Twilio)

Header Versioning: Accept: application/vnd.api+json; version=2
  Cleaner URLs
  Content negotiation
  Harder to test manually

Query Parameter: /api/orders?version=2
  Easy to add
  Easy to test
  Pollutes query string

**Backward Compatibility Rules:**
  SAFE changes (non-breaking):
    Add new fields to response
    Add new optional request fields
    Add new endpoints
    Add new HTTP methods to existing resources
    Add new enum values (with careful handling)

  BREAKING changes (require new version):
    Remove fields from response
    Rename fields
    Change field types
    Make optional field required
    Change URL structure
    Change authentication method
    Change error format

**API Evolution Strategies:**
  Deprecation process:
    1. Mark deprecated fields with Sunset header
    2. Log usage of deprecated features
    3. Notify consumers
    4. Grace period (6-12 months)
    5. Remove in new version

  Expand-Contract pattern:
    Phase 1 (Expand): Add new field alongside old
    Phase 2: Migrate consumers to new field
    Phase 3 (Contract): Remove old field`,
					CodeExamples: `// API design implementation

// Consistent error handling
type APIError struct {
    Code      string            "json:\"code\""
    Message   string            "json:\"message\""
    Details   []ValidationError "json:\"details,omitempty\""
    RequestID string            "json:\"request_id\""
}

type ValidationError struct {
    Field   string "json:\"field\""
    Message string "json:\"message\""
    Code    string "json:\"code\""
}

func WriteError(w http.ResponseWriter, status int, code, message string) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(map[string]APIError{
        "error": {Code: code, Message: message, RequestID: getRequestID(w)},
    })
}

func WriteValidationError(w http.ResponseWriter, details []ValidationError) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusUnprocessableEntity)
    json.NewEncoder(w).Encode(map[string]APIError{
        "error": {
            Code:      "VALIDATION_ERROR",
            Message:   "Invalid input data",
            Details:   details,
            RequestID: getRequestID(w),
        },
    })
}

// API resource representation
type OrderResponse struct {
    ID          string              "json:\"id\""
    CustomerID  string              "json:\"customer_id\""
    Status      string              "json:\"status\""
    Items       []OrderItemResponse "json:\"items\""
    Total       MoneyResponse       "json:\"total\""
    CreatedAt   time.Time           "json:\"created_at\""
    UpdatedAt   time.Time           "json:\"updated_at\""
    Links       map[string]string   "json:\"_links\""
}

func toOrderResponse(order *Order, baseURL string) OrderResponse {
    resp := OrderResponse{
        ID:         order.ID,
        CustomerID: order.CustomerID,
        Status:     string(order.Status),
        Total:      MoneyResponse{Amount: order.Total.Amount, Currency: order.Total.Currency},
        CreatedAt:  order.CreatedAt,
        UpdatedAt:  order.UpdatedAt,
        Links: map[string]string{
            "self":     baseURL + "/orders/" + order.ID,
            "customer": baseURL + "/customers/" + order.CustomerID,
            "cancel":   baseURL + "/orders/" + order.ID + "/cancel",
        },
    }
    for _, item := range order.Items {
        resp.Items = append(resp.Items, OrderItemResponse{
            ProductID: item.ProductID,
            Name:      item.Name,
            Quantity:  item.Quantity,
            Price:     MoneyResponse{Amount: item.Price.Amount, Currency: item.Price.Currency},
        })
    }
    return resp
}

// Pagination helper
type PaginatedResponse[T any] struct {
    Data       []T              "json:\"data\""
    Pagination PaginationMeta   "json:\"pagination\""
}

type PaginationMeta struct {
    Total      int64   "json:\"total\""
    Limit      int     "json:\"limit\""
    NextCursor string  "json:\"next_cursor,omitempty\""
    HasMore    bool    "json:\"has_more\""
}

// Rate limiting middleware
type RateLimiter struct {
    store    map[string]*TokenBucket
    mu       sync.Mutex
    rate     int  // tokens per second
    capacity int  // max burst
}

type TokenBucket struct {
    tokens    float64
    capacity  float64
    rate      float64
    lastCheck time.Time
}

func (tb *TokenBucket) Allow() bool {
    now := time.Now()
    elapsed := now.Sub(tb.lastCheck).Seconds()
    tb.tokens = math.Min(tb.capacity, tb.tokens+elapsed*tb.rate)
    tb.lastCheck = now
    
    if tb.tokens >= 1 {
        tb.tokens--
        return true
    }
    return false
}

func (rl *RateLimiter) Middleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        key := r.Header.Get("X-API-Key")
        if key == "" {
            key = r.RemoteAddr
        }
        
        rl.mu.Lock()
        bucket, ok := rl.store[key]
        if !ok {
            bucket = &TokenBucket{
                tokens:    float64(rl.capacity),
                capacity:  float64(rl.capacity),
                rate:      float64(rl.rate),
                lastCheck: time.Now(),
            }
            rl.store[key] = bucket
        }
        allowed := bucket.Allow()
        rl.mu.Unlock()
        
        if !allowed {
            w.Header().Set("Retry-After", "1")
            WriteError(w, http.StatusTooManyRequests, "RATE_LIMITED", "Too many requests")
            return
        }
        
        w.Header().Set("X-RateLimit-Limit", fmt.Sprintf("%d", rl.rate))
        w.Header().Set("X-RateLimit-Remaining", fmt.Sprintf("%d", int(bucket.tokens)))
        next.ServeHTTP(w, r)
    })
}

// API versioning via URL path
type Router struct {
    v1 *mux.Router
    v2 *mux.Router
}

func SetupRouter() *mux.Router {
    r := mux.NewRouter()
    
    // V1 routes
    v1 := r.PathPrefix("/api/v1").Subrouter()
    v1.HandleFunc("/orders", listOrdersV1).Methods("GET")
    v1.HandleFunc("/orders", createOrderV1).Methods("POST")
    v1.HandleFunc("/orders/{id}", getOrderV1).Methods("GET")
    
    // V2 routes (new response format, new features)
    v2 := r.PathPrefix("/api/v2").Subrouter()
    v2.HandleFunc("/orders", listOrdersV2).Methods("GET")
    v2.HandleFunc("/orders", createOrderV2).Methods("POST")
    v2.HandleFunc("/orders/{id}", getOrderV2).Methods("GET")
    
    // Deprecation header for V1
    v1.Use(func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            w.Header().Set("Sunset", "Sat, 01 Jan 2025 00:00:00 GMT")
            w.Header().Set("Deprecation", "true")
            w.Header().Set("Link", "</api/v2>; rel=\"successor-version\"")
            next.ServeHTTP(w, r)
        })
    })
    
    return r
}`,
				},
			},
		},
	})
}
