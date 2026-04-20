package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          2319,
			Title:       "Testing Architecture and Strategies",
			Description: "Design comprehensive testing strategies with the testing pyramid, contract testing, property-based testing, mutation testing, and testing in distributed systems.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "Testing Pyramid and Test Design",
					Content: `A well-designed testing strategy provides fast feedback, high confidence, and maintainable tests.

**Testing Pyramid:**

  E2E Tests (Few, Slow, Expensive)
  ━━━━━━━━━━━
  Integration Tests (Some)
  ━━━━━━━━━━━━━━━━━━━━━
  Unit Tests (Many, Fast, Cheap)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Unit Tests (70%):
  Test individual functions/methods in isolation
  Mock external dependencies
  Fast execution (milliseconds)
  Test edge cases and error paths
  
  What to test:
    Business logic and calculations
    Data transformations
    Validation rules
    State machines
    Algorithm correctness

Integration Tests (20%):
  Test interaction between components
  Real dependencies (database, cache, file system)
  Moderate execution time (seconds)
  Test data flow and integration points
  
  What to test:
    Database queries and transactions
    API endpoint behavior
    Message queue producers/consumers
    Cache invalidation
    External service integration

End-to-End Tests (10%):
  Test complete user flows
  Real infrastructure
  Slow execution (minutes)
  Test critical business paths
  
  What to test:
    User registration and login flow
    Purchase/checkout flow
    Critical business workflows
    Cross-service interactions

**Test Design Principles:**

FIRST Principles:
  Fast: Tests run quickly
  Isolated: Tests don't depend on each other
  Repeatable: Same result every time
  Self-validating: Pass/fail without manual inspection
  Timely: Written before or with the code

Arrange-Act-Assert (AAA):
  Arrange: Set up test data and preconditions
  Act: Execute the behavior being tested
  Assert: Verify the expected outcome

Given-When-Then (BDD style):
  Given: Initial context
  When: Event or action occurs
  Then: Expected outcome

**Test Doubles:**

Dummy: Placeholder, never actually used
  Example: Required parameter that isn't relevant to the test

Stub: Returns predetermined responses
  Example: HTTP client that always returns success

Spy: Records calls for later verification
  Example: Logger that records messages for assertion

Mock: Pre-programmed expectations verified after test
  Example: Repository that expects specific method calls

Fake: Working implementation with shortcuts
  Example: In-memory database instead of PostgreSQL

**Property-Based Testing:**
  Test properties that should always hold true
  Generate random inputs automatically
  Find edge cases humans wouldn't think of
  
  Example properties:
    Sorting: Output is ordered, same elements as input
    Serialization: Decode(Encode(x)) == x
    Parsing: Parse(Format(x)) == x
    Math: a + b == b + a (commutativity)

**Contract Testing:**
  Consumer-Driven Contracts: Consumer defines expected API behavior
  Provider verifies it meets all consumer contracts
  Prevents breaking changes between services
  Tools: Pact, Spring Cloud Contract

**Mutation Testing:**
  Modify code (mutants) and verify tests catch the change
  If tests still pass with mutated code, tests are weak
  Mutations: Change operators, remove statements, alter constants
  Metric: Mutation score = killed mutants / total mutants

**Testing in Distributed Systems:**

Chaos Engineering:
  Inject failures to test system resilience
  Network partitions, service crashes, latency injection
  Verify graceful degradation
  Tools: Chaos Monkey, Litmus, Gremlin

Consumer-Driven Contract Testing:
  Each consumer defines expected provider behavior
  Provider runs all consumer tests
  Catch incompatibilities before deployment

Integration Test Patterns:
  Test containers: Spin up real dependencies in Docker
  Service virtualization: Mock external services
  Sandbox environments: Isolated test environments`,
					CodeExamples: `// Testing patterns in Go

// Table-driven tests (Go idiomatic)
func TestCalculateDiscount(t *testing.T) {
    tests := []struct {
        name     string
        total    int64
        tier     string
        expected int64
        wantErr  bool
    }{
        {"no discount for small order", 5000, "basic", 0, false},
        {"10% for gold tier", 10000, "gold", 1000, false},
        {"20% for platinum tier", 10000, "platinum", 2000, false},
        {"max discount cap", 100000, "platinum", 15000, false},
        {"invalid tier", 10000, "invalid", 0, true},
        {"zero total", 0, "gold", 0, false},
        {"negative total", -100, "gold", 0, true},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            discount, err := CalculateDiscount(tt.total, tt.tier)
            if (err != nil) != tt.wantErr {
                t.Errorf("error = %v, wantErr %v", err, tt.wantErr)
                return
            }
            if discount != tt.expected {
                t.Errorf("got %d, want %d", discount, tt.expected)
            }
        })
    }
}

// Test with dependency injection (mocks)
type MockOrderRepo struct {
    orders map[string]*Order
    saveCalls []Order
}

func (m *MockOrderRepo) FindByID(ctx context.Context, id string) (*Order, error) {
    order, ok := m.orders[id]
    if !ok {
        return nil, ErrNotFound
    }
    return order, nil
}

func (m *MockOrderRepo) Save(ctx context.Context, order *Order) error {
    m.saveCalls = append(m.saveCalls, *order)
    m.orders[order.ID] = order
    return nil
}

func TestPlaceOrder(t *testing.T) {
    // Arrange
    repo := &MockOrderRepo{orders: make(map[string]*Order)}
    catalog := &MockCatalog{
        products: map[string]*Product{
            "prod-1": {ID: "prod-1", Name: "Widget", Price: Money{Amount: 1000, Currency: "USD"}},
        },
        available: true,
    }
    events := &MockEventPublisher{}
    
    svc := NewPlaceOrderUseCase(repo, catalog, events)
    
    // Act
    result, err := svc.Execute(context.Background(), PlaceOrderInput{
        CustomerID: "cust-1",
        Items: []PlaceOrderItem{
            {ProductID: "prod-1", Quantity: 2},
        },
    })
    
    // Assert
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    if result.OrderID == "" {
        t.Error("expected order ID")
    }
    if result.Total.Amount != 2000 {
        t.Errorf("expected total 2000, got %d", result.Total.Amount)
    }
    if len(repo.saveCalls) != 1 {
        t.Errorf("expected 1 save call, got %d", len(repo.saveCalls))
    }
    if len(events.publishedEvents) != 1 {
        t.Errorf("expected 1 event, got %d", len(events.publishedEvents))
    }
}

// Integration test with test containers
func TestOrderRepository_Integration(t *testing.T) {
    if testing.Short() {
        t.Skip("skipping integration test")
    }

    // Start PostgreSQL container
    ctx := context.Background()
    container, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
        ContainerRequest: testcontainers.ContainerRequest{
            Image:        "postgres:15-alpine",
            ExposedPorts: []string{"5432/tcp"},
            Env: map[string]string{
                "POSTGRES_DB":       "testdb",
                "POSTGRES_PASSWORD": "test",
            },
            WaitingFor: wait.ForLog("ready to accept connections").WithOccurrence(2),
        },
        Started: true,
    })
    if err != nil {
        t.Fatalf("failed to start container: %v", err)
    }
    defer container.Terminate(ctx)

    // Get connection string
    host, _ := container.Host(ctx)
    port, _ := container.MappedPort(ctx, "5432")
    dsn := fmt.Sprintf("postgres://postgres:test@%s:%s/testdb?sslmode=disable", host, port.Port())

    // Run migrations
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        t.Fatalf("failed to connect: %v", err)
    }
    runMigrations(db)

    // Create repository
    repo := NewPostgresOrderRepo(db)

    // Test Save and FindByID
    order := &Order{
        ID:         "test-order-1",
        CustomerID: "cust-1",
        Status:     OrderStatusPlaced,
        Total:      Money{Amount: 5000, Currency: "USD"},
        CreatedAt:  time.Now(),
        UpdatedAt:  time.Now(),
    }

    err = repo.Save(ctx, order)
    if err != nil {
        t.Fatalf("failed to save: %v", err)
    }

    loaded, err := repo.FindByID(ctx, "test-order-1")
    if err != nil {
        t.Fatalf("failed to find: %v", err)
    }
    if loaded.CustomerID != "cust-1" {
        t.Errorf("expected customer cust-1, got %s", loaded.CustomerID)
    }
    if loaded.Total.Amount != 5000 {
        t.Errorf("expected total 5000, got %d", loaded.Total.Amount)
    }
}

// Test fixtures and helpers
type TestFixture struct {
    DB       *sql.DB
    Repo     OrderRepository
    Catalog  ProductCatalog
    Events   EventPublisher
    Cleanup  func()
}

func SetupTestFixture(t *testing.T) *TestFixture {
    t.Helper()
    
    db := setupTestDB(t)
    repo := NewPostgresOrderRepo(db)
    catalog := &MockCatalog{available: true}
    events := &MockEventPublisher{}
    
    return &TestFixture{
        DB:      db,
        Repo:    repo,
        Catalog: catalog,
        Events:  events,
        Cleanup: func() {
            db.Exec("TRUNCATE orders, order_items CASCADE")
        },
    }
}

// Behavior Driven Tests with subtests
func TestOrderLifecycle(t *testing.T) {
    t.Run("when order is created", func(t *testing.T) {
        order := NewOrder(CustomerID("cust-1"), Address{City: "NYC"})
        
        t.Run("it should be in draft status", func(t *testing.T) {
            if order.status != OrderDraft {
                t.Errorf("expected draft, got %v", order.status)
            }
        })
        
        t.Run("it should have no items", func(t *testing.T) {
            if len(order.items) != 0 {
                t.Errorf("expected 0 items, got %d", len(order.items))
            }
        })
    })
    
    t.Run("when items are added", func(t *testing.T) {
        order := NewOrder(CustomerID("cust-1"), Address{City: "NYC"})
        err := order.AddItem(ProductID("p1"), "Widget", Money{Amount: 1000, Currency: "USD"}, 2)
        
        if err != nil {
            t.Fatalf("unexpected error: %v", err)
        }
        
        t.Run("it should calculate correct total", func(t *testing.T) {
            total := order.Total()
            if total.Amount != 2000 {
                t.Errorf("expected 2000, got %d", total.Amount)
            }
        })
    })
}`,
				},
			},
		},
		{
			ID:          2320,
			Title:       "Observability and Monitoring Architecture",
			Description: "Design observability systems with structured logging, distributed tracing, metrics collection, alerting strategies, and SLOs/SLIs.",
			Order:       20,
			Lessons: []problems.Lesson{
				{
					Title: "Three Pillars of Observability",
					Content: `Observability enables understanding of system behavior through external outputs without modifying the system.

**Logging:**

Structured Logging:
  Use JSON or key-value format instead of unstructured text
  Include context: request ID, user ID, trace ID
  Log levels: DEBUG, INFO, WARN, ERROR, FATAL
  
  Bad: "User 123 created order for $50.00"
  Good: {"level":"info","msg":"order created","user_id":"123","order_id":"ord-456","amount":5000,"currency":"USD","trace_id":"abc"}

Log Aggregation Pipeline:
  Application -> Log Shipper (Fluentd/Filebeat)
    -> Message Queue (Kafka)
    -> Log Processor (Logstash)
    -> Storage (Elasticsearch/Loki)
    -> Visualization (Kibana/Grafana)

Best Practices:
  Log at service boundaries (incoming/outgoing requests)
  Include correlation IDs for request tracing
  Don't log sensitive data (passwords, tokens, PII)
  Use sampling for high-volume endpoints
  Set appropriate retention policies
  Separate access logs from application logs

**Metrics:**

Types:
  Counter: Monotonically increasing (total requests, errors)
  Gauge: Current value (active connections, queue depth)
  Histogram: Distribution of values (request latency)
  Summary: Pre-calculated percentiles

RED Method (for request-driven services):
  Rate: Requests per second
  Errors: Number of failed requests
  Duration: Distribution of request latency

USE Method (for resources):
  Utilization: Percentage of resource used
  Saturation: Amount of work queued
  Errors: Number of error events

Golden Signals (Google SRE):
  Latency: Time to serve a request
  Traffic: Demand on the system
  Errors: Rate of failed requests
  Saturation: How full the system is

Metrics Pipeline:
  Application -> Prometheus scrape / push
    -> Prometheus TSDB
    -> Grafana dashboards
    -> Alertmanager -> PagerDuty/Slack

**Distributed Tracing:**

Trace Structure:
  Trace: End-to-end request journey
  Span: Individual operation within a trace
  
  Trace abc-123:
    ├── Span: API Gateway (20ms)
    │   ├── Span: Auth Service (5ms)
    │   └── Span: Order Service (15ms)
    │       ├── Span: Database Query (3ms)
    │       ├── Span: Payment Service (8ms)
    │       │   └── Span: Stripe API (6ms)
    │       └── Span: Send Event to Kafka (2ms)

Context Propagation:
  W3C Trace Context standard
  Headers: traceparent, tracestate
  Propagate across HTTP, gRPC, message queues

Sampling Strategies:
  Head-based: Decide at trace start (simple, may miss errors)
  Tail-based: Decide after trace complete (capture all errors, resource intensive)
  Adaptive: Adjust rate based on traffic

**SLIs, SLOs, and SLAs:**

SLI (Service Level Indicator):
  Quantitative measure of service behavior
  Examples:
    Request latency (p50, p99)
    Error rate
    Availability (successful requests / total)
    Throughput

SLO (Service Level Objective):
  Target value for an SLI
  Examples:
    99.9% of requests complete within 200ms
    Error rate below 0.1%
    99.95% availability per month

Error Budget:
  100% - SLO = Error Budget
  99.9% SLO = 0.1% error budget = 43.2 minutes/month downtime
  99.95% SLO = 0.05% error budget = 21.6 minutes/month
  99.99% SLO = 0.01% error budget = 4.32 minutes/month
  
  Error budget policy:
    Budget remaining: Deploy freely, take risks
    Budget exhausted: Freeze deployments, focus on reliability
    Budget exceeded: Incident review, invest in reliability

SLA (Service Level Agreement):
  Contract with customers including consequences
  Usually less aggressive than internal SLOs
  Financial penalties for violations

**Alerting Strategy:**

Severity Levels:
  P1 (Critical): Page on-call immediately
    Service completely down, data loss risk
  P2 (High): Page during business hours
    Degraded service, approaching SLO violation
  P3 (Medium): Ticket, response within hours
    Non-critical issues, trending toward problems
  P4 (Low): Ticket, response within days
    Minor issues, tech debt

Alert on symptoms, not causes:
  Good: "Error rate > 1% for 5 minutes"
  Bad: "CPU > 80%" (might be fine under load)

Avoid alert fatigue:
  Every alert should be actionable
  Regular alert review and pruning
  Route to the right team
  Include runbook links in alerts`,
					CodeExamples: `// Observability implementation

// Structured logger
type Logger struct {
    fields map[string]interface{}
    output io.Writer
    level  LogLevel
}

type LogLevel int
const (
    LevelDebug LogLevel = iota
    LevelInfo
    LevelWarn
    LevelError
)

func NewLogger(output io.Writer, level LogLevel) *Logger {
    return &Logger{
        fields: make(map[string]interface{}),
        output: output,
        level:  level,
    }
}

func (l *Logger) With(key string, value interface{}) *Logger {
    newFields := make(map[string]interface{}, len(l.fields)+1)
    for k, v := range l.fields {
        newFields[k] = v
    }
    newFields[key] = value
    return &Logger{fields: newFields, output: l.output, level: l.level}
}

func (l *Logger) WithContext(ctx context.Context) *Logger {
    logger := l
    if traceID, ok := ctx.Value(traceIDKey).(string); ok {
        logger = logger.With("trace_id", traceID)
    }
    if requestID, ok := ctx.Value(requestIDKey).(string); ok {
        logger = logger.With("request_id", requestID)
    }
    return logger
}

func (l *Logger) Info(msg string, keysAndValues ...interface{}) {
    l.log(LevelInfo, msg, keysAndValues...)
}

func (l *Logger) Error(msg string, keysAndValues ...interface{}) {
    l.log(LevelError, msg, keysAndValues...)
}

func (l *Logger) log(level LogLevel, msg string, keysAndValues ...interface{}) {
    if level < l.level {
        return
    }
    entry := make(map[string]interface{}, len(l.fields)+len(keysAndValues)/2+3)
    entry["level"] = levelString(level)
    entry["msg"] = msg
    entry["timestamp"] = time.Now().UTC().Format(time.RFC3339Nano)
    
    for k, v := range l.fields {
        entry[k] = v
    }
    for i := 0; i < len(keysAndValues)-1; i += 2 {
        entry[keysAndValues[i].(string)] = keysAndValues[i+1]
    }
    
    data, _ := json.Marshal(entry)
    l.output.Write(append(data, '\n'))
}

// Metrics middleware
type MetricsMiddleware struct {
    requestsTotal    *prometheus.CounterVec
    requestDuration  *prometheus.HistogramVec
    requestsInFlight prometheus.Gauge
}

func NewMetricsMiddleware(reg prometheus.Registerer) *MetricsMiddleware {
    m := &MetricsMiddleware{
        requestsTotal: prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "http_requests_total",
                Help: "Total number of HTTP requests",
            },
            []string{"method", "path", "status"},
        ),
        requestDuration: prometheus.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "http_request_duration_seconds",
                Help:    "HTTP request latency",
                Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
            },
            []string{"method", "path"},
        ),
        requestsInFlight: prometheus.NewGauge(
            prometheus.GaugeOpts{
                Name: "http_requests_in_flight",
                Help: "Current number of HTTP requests being processed",
            },
        ),
    }
    reg.MustRegister(m.requestsTotal, m.requestDuration, m.requestsInFlight)
    return m
}

func (m *MetricsMiddleware) Handler(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        m.requestsInFlight.Inc()
        defer m.requestsInFlight.Dec()
        
        start := time.Now()
        wrapped := &responseWriter{ResponseWriter: w, statusCode: 200}
        
        next.ServeHTTP(wrapped, r)
        
        duration := time.Since(start).Seconds()
        path := normalizePath(r.URL.Path)
        status := fmt.Sprintf("%d", wrapped.statusCode)
        
        m.requestsTotal.WithLabelValues(r.Method, path, status).Inc()
        m.requestDuration.WithLabelValues(r.Method, path).Observe(duration)
    })
}

type responseWriter struct {
    http.ResponseWriter
    statusCode int
}

func (w *responseWriter) WriteHeader(code int) {
    w.statusCode = code
    w.ResponseWriter.WriteHeader(code)
}

// Health check endpoint
type HealthChecker struct {
    checks map[string]HealthCheck
}

type HealthCheck func(ctx context.Context) error

type HealthStatus struct {
    Status string                    "json:\"status\""
    Checks map[string]CheckResult   "json:\"checks\""
}

type CheckResult struct {
    Status  string "json:\"status\""
    Message string "json:\"message,omitempty\""
}

func (h *HealthChecker) Check(ctx context.Context) HealthStatus {
    status := HealthStatus{
        Status: "healthy",
        Checks: make(map[string]CheckResult),
    }
    
    for name, check := range h.checks {
        if err := check(ctx); err != nil {
            status.Status = "unhealthy"
            status.Checks[name] = CheckResult{Status: "unhealthy", Message: err.Error()}
        } else {
            status.Checks[name] = CheckResult{Status: "healthy"}
        }
    }
    
    return status
}

func DatabaseHealthCheck(db *sql.DB) HealthCheck {
    return func(ctx context.Context) error {
        ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
        defer cancel()
        return db.PingContext(ctx)
    }
}

func RedisHealthCheck(client *redis.Client) HealthCheck {
    return func(ctx context.Context) error {
        return client.Ping(ctx).Err()
    }
}`,
				},
			},
		},
	})
}
