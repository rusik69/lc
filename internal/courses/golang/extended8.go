package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          1627,
			Title:       "Distributed Systems in Go",
			Description: "Build distributed systems: consensus, service discovery, circuit breakers, distributed tracing, and event-driven architectures.",
			Order:       27,
			Lessons: []problems.Lesson{
				{
					Title: "Distributed Systems Patterns",
					Content: `Go is the dominant language for distributed systems infrastructure (Kubernetes, etcd, CockroachDB, etc.). Understanding these patterns is essential for building reliable distributed applications.

**Circuit Breaker:**
` + "```" + `
Circuit breaker prevents cascading failures in distributed systems.

States:
  CLOSED (normal):
    Requests pass through
    Count failures
    If failure threshold reached → switch to OPEN
    
  OPEN (blocking):
    All requests immediately rejected
    Save resources, don't overwhelm failing service
    After timeout → switch to HALF-OPEN
    
  HALF-OPEN (testing):
    Allow ONE request through
    If succeeds → switch to CLOSED
    If fails → switch back to OPEN

State transitions:
  CLOSED ──failure threshold──→ OPEN
  OPEN ──timeout──→ HALF-OPEN
  HALF-OPEN ──success──→ CLOSED
  HALF-OPEN ──failure──→ OPEN

Parameters:
  - Failure threshold: how many failures before opening (e.g., 5)
  - Timeout: how long to stay open before testing (e.g., 30s)
  - Success threshold: how many successes in half-open to close (e.g., 1)

Why it matters:
  Without circuit breaker:
    Service A → Service B (down)
    A keeps retrying → goroutine/connection exhaustion
    A becomes slow → A's callers time out → cascade!
    
  With circuit breaker:
    Service A → Service B (down)
    After 5 failures → circuit opens
    A immediately returns error → callers handle gracefully
    After 30s → A tests B → B recovered? → close circuit
` + "```" + `

**Retry with Backoff:**
` + "```" + `
Retry policies for transient failures:

Simple retry:
  ✗ Just retry immediately
  Problem: "thundering herd" → all retries hit at once

Exponential backoff:
  Wait 100ms, 200ms, 400ms, 800ms, 1600ms, ...
  wait = baseDelay * 2^attempt
  Spreads load over time

Exponential backoff with jitter:
  wait = random(0, baseDelay * 2^attempt)
  Prevents retry storms when many clients fail simultaneously
  
  Types of jitter:
    Full jitter:  random(0, baseDelay * 2^attempt)
    Equal jitter: baseDelay * 2^attempt / 2 + random(0, baseDelay * 2^attempt / 2)
    Decorrelated: random(baseDelay, lastDelay * 3)

When to retry:
  ✓ Network timeouts (might be transient)
  ✓ HTTP 429 (rate limited, respect Retry-After header)
  ✓ HTTP 503 (service unavailable)
  ✓ Connection refused (service restarting)
  ✗ HTTP 400 (bad request, won't fix itself)
  ✗ HTTP 401/403 (auth error, retry won't help)
  ✗ HTTP 404 (not found, won't appear)
  
  Key: only retry idempotent operations!
  Non-idempotent: POST /orders (might create duplicate orders)
  Idempotent: GET, PUT (same result if repeated)
` + "```" + `

**Service Discovery:**
` + "```" + `
How services find each other in a distributed system:

1. DNS-based:
   service-a.namespace.svc.cluster.local
   Simple, built into Kubernetes
   Drawback: DNS caching can serve stale data
   
2. Service registry (consul, etcd):
   Services register themselves on startup
   Clients query registry to find services
   Health checks remove unhealthy instances
   
3. Client-side load balancing:
   Client gets list of instances from registry
   Client chooses which instance to call
   Algorithms: round-robin, least-connections, random
   
4. Server-side load balancing:
   Load balancer in front of service instances
   Client talks to load balancer
   Simpler for client, single point of failure

Leader election:
  Only one instance does certain work (cron jobs, migrations)
  
  Using etcd/consul:
    Acquire lock with TTL
    Renew lock periodically
    If lock lost → stop being leader
    
  Raft consensus:
    Used by etcd, CockroachDB
    Majority of nodes agree on leader
    Handles network partitions correctly
` + "```" + ``,
					CodeExamples: `// Distributed systems patterns in Go
package main

import (
    "context"
    "errors"
    "fmt"
    "math"
    "math/rand"
    "sync"
    "time"
)

// Circuit Breaker
type CircuitState int

const (
    StateClosed CircuitState = iota
    StateOpen
    StateHalfOpen
)

func (s CircuitState) String() string {
    switch s {
    case StateClosed:
        return "CLOSED"
    case StateOpen:
        return "OPEN"
    case StateHalfOpen:
        return "HALF-OPEN"
    default:
        return "UNKNOWN"
    }
}

type CircuitBreaker struct {
    mu               sync.Mutex
    state            CircuitState
    failures         int
    successes        int
    failureThreshold int
    successThreshold int
    timeout          time.Duration
    lastFailure      time.Time
    onStateChange    func(from, to CircuitState)
}

func NewCircuitBreaker(failureThreshold, successThreshold int, timeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        state:            StateClosed,
        failureThreshold: failureThreshold,
        successThreshold: successThreshold,
        timeout:          timeout,
    }
}

var ErrCircuitOpen = errors.New("circuit breaker is open")

func (cb *CircuitBreaker) Execute(fn func() error) error {
    cb.mu.Lock()
    
    switch cb.state {
    case StateOpen:
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.setState(StateHalfOpen)
        } else {
            cb.mu.Unlock()
            return ErrCircuitOpen
        }
    }
    
    cb.mu.Unlock()
    
    err := fn()
    
    cb.mu.Lock()
    defer cb.mu.Unlock()
    
    if err != nil {
        cb.failures++
        cb.successes = 0
        cb.lastFailure = time.Now()
        
        if cb.failures >= cb.failureThreshold {
            cb.setState(StateOpen)
        }
        return err
    }
    
    cb.successes++
    cb.failures = 0
    
    if cb.state == StateHalfOpen && cb.successes >= cb.successThreshold {
        cb.setState(StateClosed)
    }
    
    return nil
}

func (cb *CircuitBreaker) setState(state CircuitState) {
    if cb.state != state {
        from := cb.state
        cb.state = state
        cb.failures = 0
        cb.successes = 0
        if cb.onStateChange != nil {
            cb.onStateChange(from, state)
        }
    }
}

func (cb *CircuitBreaker) State() CircuitState {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    return cb.state
}

// Retry with exponential backoff and jitter
type RetryConfig struct {
    MaxRetries  int
    BaseDelay   time.Duration
    MaxDelay    time.Duration
    Retryable   func(error) bool
}

func DefaultRetryConfig() RetryConfig {
    return RetryConfig{
        MaxRetries: 5,
        BaseDelay:  100 * time.Millisecond,
        MaxDelay:   30 * time.Second,
        Retryable:  func(error) bool { return true },
    }
}

func RetryWithBackoff(ctx context.Context, config RetryConfig, fn func() error) error {
    var lastErr error
    
    for attempt := 0; attempt <= config.MaxRetries; attempt++ {
        err := fn()
        if err == nil {
            return nil
        }
        lastErr = err
        
        if !config.Retryable(err) {
            return err
        }
        
        if attempt == config.MaxRetries {
            break
        }
        
        // Exponential backoff with full jitter
        delay := float64(config.BaseDelay) * math.Pow(2, float64(attempt))
        if delay > float64(config.MaxDelay) {
            delay = float64(config.MaxDelay)
        }
        jitter := time.Duration(rand.Float64() * delay)
        
        fmt.Printf("    Retry %d/%d after %v (error: %v)\n",
            attempt+1, config.MaxRetries, jitter, err)
        
        select {
        case <-time.After(jitter):
        case <-ctx.Done():
            return ctx.Err()
        }
    }
    
    return fmt.Errorf("max retries exceeded: %w", lastErr)
}

// Service Registry (in-memory simulation)
type ServiceInstance struct {
    ID      string
    Address string
    Port    int
    Healthy bool
    Meta    map[string]string
}

type ServiceRegistry struct {
    mu       sync.RWMutex
    services map[string][]ServiceInstance
}

func NewServiceRegistry() *ServiceRegistry {
    return &ServiceRegistry{
        services: make(map[string][]ServiceInstance),
    }
}

func (r *ServiceRegistry) Register(name string, instance ServiceInstance) {
    r.mu.Lock()
    defer r.mu.Unlock()
    r.services[name] = append(r.services[name], instance)
    fmt.Printf("  Registered %s: %s at %s:%d\n", name, instance.ID, instance.Address, instance.Port)
}

func (r *ServiceRegistry) Deregister(name, instanceID string) {
    r.mu.Lock()
    defer r.mu.Unlock()
    instances := r.services[name]
    for i, inst := range instances {
        if inst.ID == instanceID {
            r.services[name] = append(instances[:i], instances[i+1:]...)
            fmt.Printf("  Deregistered %s: %s\n", name, instanceID)
            return
        }
    }
}

func (r *ServiceRegistry) Discover(name string) []ServiceInstance {
    r.mu.RLock()
    defer r.mu.RUnlock()
    instances := r.services[name]
    healthy := make([]ServiceInstance, 0, len(instances))
    for _, inst := range instances {
        if inst.Healthy {
            healthy = append(healthy, inst)
        }
    }
    return healthy
}

// Round-robin load balancer
type RoundRobinLB struct {
    mu      sync.Mutex
    current int
}

func (lb *RoundRobinLB) Pick(instances []ServiceInstance) *ServiceInstance {
    if len(instances) == 0 {
        return nil
    }
    lb.mu.Lock()
    defer lb.mu.Unlock()
    inst := &instances[lb.current%len(instances)]
    lb.current++
    return inst
}

func main() {
    // Circuit Breaker demo
    fmt.Println("=== Circuit Breaker ===")
    
    cb := NewCircuitBreaker(3, 1, 500*time.Millisecond)
    cb.onStateChange = func(from, to CircuitState) {
        fmt.Printf("  State: %s → %s\n", from, to)
    }
    
    // Simulate failing service
    callCount := 0
    call := func() error {
        callCount++
        if callCount <= 5 {
            return errors.New("service unavailable")
        }
        return nil // Recovers after 5 calls
    }
    
    // Normal calls → failures → circuit opens
    for i := 0; i < 5; i++ {
        err := cb.Execute(call)
        fmt.Printf("  Call %d: state=%s err=%v\n", i+1, cb.State(), err)
    }
    
    // Wait for timeout
    fmt.Println("  Waiting for timeout...")
    time.Sleep(600 * time.Millisecond)
    
    // Half-open → test → close
    err := cb.Execute(call)
    fmt.Printf("  Recovery call: state=%s err=%v\n", cb.State(), err)
    
    // Retry with backoff demo
    fmt.Println("\n=== Retry with Exponential Backoff ===")
    
    attempt := 0
    err = RetryWithBackoff(context.Background(), RetryConfig{
        MaxRetries: 5,
        BaseDelay:  10 * time.Millisecond,
        MaxDelay:   1 * time.Second,
        Retryable:  func(error) bool { return true },
    }, func() error {
        attempt++
        if attempt < 4 {
            return errors.New("temporary error")
        }
        return nil
    })
    if err != nil {
        fmt.Printf("  Final error: %v\n", err)
    } else {
        fmt.Printf("  Success after %d attempts\n", attempt)
    }
    
    // Service Discovery demo
    fmt.Println("\n=== Service Discovery ===")
    
    registry := NewServiceRegistry()
    
    // Register instances
    for i := 1; i <= 3; i++ {
        registry.Register("user-service", ServiceInstance{
            ID:      fmt.Sprintf("user-%d", i),
            Address: fmt.Sprintf("10.0.0.%d", i),
            Port:    8080,
            Healthy: true,
        })
    }
    
    // Discover healthy instances
    instances := registry.Discover("user-service")
    fmt.Printf("\n  Discovered %d healthy instances\n", len(instances))
    
    // Round-robin load balancing
    lb := &RoundRobinLB{}
    fmt.Println("\n  Round-robin calls:")
    for i := 0; i < 6; i++ {
        inst := lb.Pick(instances)
        fmt.Printf("    Request %d → %s (%s:%d)\n", i+1, inst.ID, inst.Address, inst.Port)
    }
    
    // Deregister one instance
    registry.Deregister("user-service", "user-2")
    instances = registry.Discover("user-service")
    fmt.Printf("\n  After deregister: %d instances\n", len(instances))
}`,
				},
				{
					Title: "Event-Driven Architecture",
					Content: `Event-driven architecture decouples services through asynchronous message passing. Go's concurrency primitives make it a natural fit for building event-driven systems.

**Event Bus Pattern:**
` + "```" + `
Event bus: publish-subscribe within a process

  type Event struct {
      Type    string
      Payload any
      Time    time.Time
  }
  
  type EventBus struct {
      subscribers map[string][]chan Event
      mu          sync.RWMutex
  }
  
  Subscribe: register a channel for event type
  Publish: send event to all subscribers of that type
  
  Use cases:
    - Decouple components within a monolith
    - Testing (subscribe to events, verify behavior)
    - Audit logging (subscribe to all events)
    - Metrics (count events by type)

Fan-out pattern:
  One event → multiple consumers process independently
  Each consumer has its own channel
  If one consumer is slow, others are not affected
  
Fan-in pattern:
  Multiple sources → one consumer aggregates
  Use select or a merged channel
` + "```" + `

**Message Queue Patterns:**
` + "```" + `
External message queues (Kafka, RabbitMQ, NATS):

Producer → Queue → Consumer

Key patterns:
  
  At-most-once delivery:
    Send and forget. Message may be lost.
    Fast, no overhead.
    Use for: metrics, logs (where loss is acceptable)
  
  At-least-once delivery:
    Retry until acknowledged.
    Message may be delivered multiple times!
    Consumer must be idempotent.
    Use for: most business events
  
  Exactly-once delivery:
    Extremely difficult to achieve.
    Usually: at-least-once + idempotent consumer.
    Or: transactional outbox pattern.

Transactional outbox:
  Problem: How to atomically update DB AND publish event?
  
  ✗ BEGIN; UPDATE orders; COMMIT; publish(event)
    // If publish fails → inconsistency!
    
  ✗ BEGIN; UPDATE orders; publish(event); COMMIT
    // If COMMIT fails → event published but data not saved!
    
  ✓ Outbox pattern:
    BEGIN
      UPDATE orders SET status = 'paid'
      INSERT INTO outbox (event_type, payload) VALUES (...)
    COMMIT
    
    Separate process reads outbox → publishes to queue → marks as sent
    Guaranteed consistency between DB and events!

Saga pattern:
  Distributed transaction across multiple services:
  
  1. Order Service: Create order (PENDING)
  2. Payment Service: Charge customer
  3. Inventory Service: Reserve items
  4. Shipping Service: Schedule delivery
  5. Order Service: Mark completed
  
  If step 3 fails:
    Compensating actions:
    - Payment Service: Refund customer
    - Order Service: Mark cancelled
    
  Two types:
    Choreography: each service publishes events, others react
    Orchestration: central coordinator manages the flow
` + "```" + `

**CQRS (Command Query Responsibility Segregation):**
` + "```" + `
Separate read and write models:

  Command side (writes):
    - Receives commands (CreateOrder, UpdateUser)
    - Validates business rules
    - Writes to primary database
    - Publishes events
    
  Query side (reads):
    - Receives queries (GetOrder, ListUsers)
    - Reads from optimized read store
    - May use different database/schema
    - Eventually consistent with write side

Why CQRS:
  - Read and write have different scaling needs
  - Read: 90%+ of traffic, optimize for fast queries
  - Write: complex validation, needs strong consistency
  - Can use different databases (PostgreSQL for writes, Elasticsearch for reads)

Event Sourcing (often paired with CQRS):
  Instead of storing current state, store all events:
  
  Events: [OrderCreated, ItemAdded, ItemRemoved, OrderPaid, OrderShipped]
  
  Current state = replay all events
  
  Benefits:
    - Complete audit trail
    - Can reconstruct state at any point in time
    - Can add new read models by replaying events
    - Natural fit for event-driven architecture
    
  Drawbacks:
    - Complexity (snapshots needed for fast rebuild)
    - Eventually consistent reads
    - Schema evolution of events is challenging
` + "```" + ``,
					CodeExamples: `// Event-driven architecture patterns
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Event system
type Event struct {
    Type      string
    Payload   any
    Timestamp time.Time
    ID        string
}

type EventHandler func(Event)

type EventBus struct {
    mu          sync.RWMutex
    handlers    map[string][]EventHandler
    middlewares []func(Event, EventHandler) EventHandler
}

func NewEventBus() *EventBus {
    return &EventBus{
        handlers: make(map[string][]EventHandler),
    }
}

func (b *EventBus) Subscribe(eventType string, handler EventHandler) {
    b.mu.Lock()
    defer b.mu.Unlock()
    b.handlers[eventType] = append(b.handlers[eventType], handler)
}

func (b *EventBus) Publish(event Event) {
    event.Timestamp = time.Now()
    
    b.mu.RLock()
    handlers := b.handlers[event.Type]
    allHandlers := b.handlers["*"] // Wildcard subscribers
    b.mu.RUnlock()
    
    for _, h := range handlers {
        h(event)
    }
    for _, h := range allHandlers {
        h(event)
    }
}

// Event Sourcing
type EventStore struct {
    mu     sync.RWMutex
    events []Event
}

func NewEventStore() *EventStore {
    return &EventStore{events: make([]Event, 0)}
}

func (s *EventStore) Append(event Event) {
    s.mu.Lock()
    defer s.mu.Unlock()
    event.Timestamp = time.Now()
    event.ID = fmt.Sprintf("evt-%d", len(s.events)+1)
    s.events = append(s.events, event)
}

func (s *EventStore) GetEvents(entityType string) []Event {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    var result []Event
    for _, e := range s.events {
        if e.Type == entityType || entityType == "" {
            result = append(result, e)
        }
    }
    return result
}

// Order aggregate (event-sourced)
type OrderStatus string

const (
    OrderPending   OrderStatus = "PENDING"
    OrderPaid       OrderStatus = "PAID"
    OrderShipped    OrderStatus = "SHIPPED"
    OrderCancelled  OrderStatus = "CANCELLED"
)

type Order struct {
    ID     string
    Status OrderStatus
    Items  []OrderItem
    Total  float64
}

type OrderItem struct {
    ProductID string
    Quantity  int
    Price     float64
}

// Rebuild order state from events
func RebuildOrder(events []Event) *Order {
    order := &Order{}
    for _, e := range events {
        switch e.Type {
        case "OrderCreated":
            data := e.Payload.(map[string]string)
            order.ID = data["id"]
            order.Status = OrderPending
        case "ItemAdded":
            item := e.Payload.(OrderItem)
            order.Items = append(order.Items, item)
            order.Total += item.Price * float64(item.Quantity)
        case "OrderPaid":
            order.Status = OrderPaid
        case "OrderShipped":
            order.Status = OrderShipped
        case "OrderCancelled":
            order.Status = OrderCancelled
        }
    }
    return order
}

// Saga pattern (orchestrator)
type SagaStep struct {
    Name       string
    Execute    func(ctx context.Context) error
    Compensate func(ctx context.Context) error
}

type Saga struct {
    steps     []SagaStep
    completed []int
}

func NewSaga(steps ...SagaStep) *Saga {
    return &Saga{steps: steps}
}

func (s *Saga) Run(ctx context.Context) error {
    for i, step := range s.steps {
        fmt.Printf("  Executing: %s\n", step.Name)
        if err := step.Execute(ctx); err != nil {
            fmt.Printf("  Failed: %s (%v)\n", step.Name, err)
            // Compensate in reverse order
            s.compensate(ctx)
            return fmt.Errorf("saga failed at step %s: %w", step.Name, err)
        }
        s.completed = append(s.completed, i)
    }
    fmt.Println("  Saga completed successfully")
    return nil
}

func (s *Saga) compensate(ctx context.Context) {
    fmt.Println("  Starting compensation...")
    for i := len(s.completed) - 1; i >= 0; i-- {
        step := s.steps[s.completed[i]]
        if step.Compensate != nil {
            fmt.Printf("  Compensating: %s\n", step.Name)
            if err := step.Compensate(ctx); err != nil {
                fmt.Printf("  Compensation failed: %s (%v)\n", step.Name, err)
            }
        }
    }
}

// Transactional Outbox (simplified)
type OutboxMessage struct {
    ID        int
    EventType string
    Payload   string
    Sent      bool
    CreatedAt time.Time
}

type Outbox struct {
    mu       sync.Mutex
    messages []OutboxMessage
    nextID   int
}

func NewOutbox() *Outbox {
    return &Outbox{messages: make([]OutboxMessage, 0), nextID: 1}
}

func (o *Outbox) Add(eventType, payload string) {
    o.mu.Lock()
    defer o.mu.Unlock()
    o.messages = append(o.messages, OutboxMessage{
        ID:        o.nextID,
        EventType: eventType,
        Payload:   payload,
        Sent:      false,
        CreatedAt: time.Now(),
    })
    o.nextID++
}

func (o *Outbox) GetUnsent() []OutboxMessage {
    o.mu.Lock()
    defer o.mu.Unlock()
    var unsent []OutboxMessage
    for _, m := range o.messages {
        if !m.Sent {
            unsent = append(unsent, m)
        }
    }
    return unsent
}

func (o *Outbox) MarkSent(id int) {
    o.mu.Lock()
    defer o.mu.Unlock()
    for i := range o.messages {
        if o.messages[i].ID == id {
            o.messages[i].Sent = true
            return
        }
    }
}

func main() {
    // Event Bus
    fmt.Println("=== Event Bus ===")
    bus := NewEventBus()
    
    // Subscribe to events
    bus.Subscribe("user.created", func(e Event) {
        fmt.Printf("  [EmailService] Send welcome email: %v\n", e.Payload)
    })
    bus.Subscribe("user.created", func(e Event) {
        fmt.Printf("  [Analytics] Track signup: %v\n", e.Payload)
    })
    bus.Subscribe("*", func(e Event) {
        fmt.Printf("  [AuditLog] %s: %v\n", e.Type, e.Payload)
    })
    
    bus.Publish(Event{Type: "user.created", Payload: map[string]string{"name": "Alice"}})
    bus.Publish(Event{Type: "order.placed", Payload: map[string]string{"id": "ord-1"}})
    
    // Event Sourcing
    fmt.Println("\n=== Event Sourcing ===")
    store := NewEventStore()
    
    store.Append(Event{Type: "OrderCreated", Payload: map[string]string{"id": "ord-1"}})
    store.Append(Event{Type: "ItemAdded", Payload: OrderItem{ProductID: "prod-1", Quantity: 2, Price: 29.99}})
    store.Append(Event{Type: "ItemAdded", Payload: OrderItem{ProductID: "prod-2", Quantity: 1, Price: 49.99}})
    store.Append(Event{Type: "OrderPaid", Payload: nil})
    
    events := store.GetEvents("")
    order := RebuildOrder(events)
    fmt.Printf("  Order: ID=%s Status=%s Items=%d Total=$%.2f\n",
        order.ID, order.Status, len(order.Items), order.Total)
    
    // Saga pattern
    fmt.Println("\n=== Saga Pattern (Success) ===")
    successSaga := NewSaga(
        SagaStep{
            Name:       "CreateOrder",
            Execute:    func(ctx context.Context) error { return nil },
            Compensate: func(ctx context.Context) error { fmt.Println("  → Cancel order"); return nil },
        },
        SagaStep{
            Name:       "ChargePayment",
            Execute:    func(ctx context.Context) error { return nil },
            Compensate: func(ctx context.Context) error { fmt.Println("  → Refund payment"); return nil },
        },
        SagaStep{
            Name:       "ReserveInventory",
            Execute:    func(ctx context.Context) error { return nil },
            Compensate: func(ctx context.Context) error { fmt.Println("  → Release inventory"); return nil },
        },
    )
    _ = successSaga.Run(context.Background())
    
    fmt.Println("\n=== Saga Pattern (Failure with Compensation) ===")
    failSaga := NewSaga(
        SagaStep{
            Name:       "CreateOrder",
            Execute:    func(ctx context.Context) error { return nil },
            Compensate: func(ctx context.Context) error { fmt.Println("  → Cancel order"); return nil },
        },
        SagaStep{
            Name:       "ChargePayment",
            Execute:    func(ctx context.Context) error { return nil },
            Compensate: func(ctx context.Context) error { fmt.Println("  → Refund payment"); return nil },
        },
        SagaStep{
            Name:       "ReserveInventory",
            Execute:    func(ctx context.Context) error { return fmt.Errorf("out of stock") },
            Compensate: func(ctx context.Context) error { return nil },
        },
    )
    if err := failSaga.Run(context.Background()); err != nil {
        fmt.Printf("  Saga error: %v\n", err)
    }
    
    // Outbox pattern
    fmt.Println("\n=== Transactional Outbox ===")
    outbox := NewOutbox()
    
    // Simulate: within DB transaction, write to outbox
    outbox.Add("order.created", "{\"id\":\"ord-1\",\"total\":109.97}")
    outbox.Add("payment.charged", "{\"order_id\":\"ord-1\",\"amount\":109.97}")
    
    // Outbox relay (separate process/goroutine)
    unsent := outbox.GetUnsent()
    fmt.Printf("  Unsent messages: %d\n", len(unsent))
    for _, msg := range unsent {
        fmt.Printf("  Publishing: [%s] %s\n", msg.EventType, msg.Payload)
        outbox.MarkSent(msg.ID)
    }
    
    unsent = outbox.GetUnsent()
    fmt.Printf("  Remaining unsent: %d\n", len(unsent))
}`,
				},
				{
					Title: "Distributed Tracing and Observability",
					Content: `Observability is critical for understanding distributed systems. Go has excellent support for metrics, logging, and tracing through OpenTelemetry.

**The Three Pillars of Observability:**
` + "```" + `
1. Metrics: Numerical measurements over time
   - Counters: monotonically increasing (requests_total)
   - Gauges: current value (goroutines_count)
   - Histograms: distribution (request_duration_seconds)
   
2. Logs: Discrete events with context
   - Structured logging (JSON)
   - Include trace_id for correlation
   - Levels: debug, info, warn, error
   
3. Traces: Request flow across services
   - Span: single operation with timing
   - Trace: collection of spans forming a DAG
   - Context propagation: pass trace_id across services

OpenTelemetry: Unified observability framework
  Replaces: OpenTracing + OpenCensus
  Supports: traces, metrics, logs
  Vendors: Jaeger, Zipkin, Prometheus, Datadog, etc.
` + "```" + `

**Structured Logging:**
` + "```" + `
slog (Go 1.21+ standard library):
  
  logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
      Level: slog.LevelInfo,
  }))
  
  logger.Info("user logged in",
      "user_id", "123",
      "ip", "192.168.1.1",
      "trace_id", traceID,
  )
  
  Output:
  {"time":"2024-01-15T10:30:00Z","level":"INFO","msg":"user logged in",
   "user_id":"123","ip":"192.168.1.1","trace_id":"abc123"}

Best practices:
  - ALWAYS use structured logging (key=value pairs)
  - Include request_id/trace_id in every log
  - Log at appropriate levels
  - Don't log sensitive data (passwords, tokens, PII)
  - Use context-aware loggers
  
  // Context-aware:
  logger = logger.With("service", "user-api", "version", "1.0.0")
  
  // Per-request:
  reqLogger := logger.With("request_id", requestID, "user_id", userID)
  reqLogger.Info("processing order", "order_id", orderID)

Logging levels:
  DEBUG: Detailed diagnostic (disabled in production)
  INFO:  Normal operations (user logged in, order placed)
  WARN:  Potential issues (slow query, retry needed)
  ERROR: Failures requiring attention (DB down, external API error)
  
  ✗ logger.Error("user not found") → INFO (expected business case)
  ✓ logger.Info("user not found", "user_id", id)
  ✗ logger.Info("database connection failed") → ERROR
  ✓ logger.Error("database connection failed", "error", err)
` + "```" + `

**Metrics with Prometheus:**
` + "```" + `
Prometheus client for Go:

  import "github.com/prometheus/client_golang/prometheus"
  
  // Counter (monotonically increasing)
  requestsTotal := prometheus.NewCounterVec(
      prometheus.CounterOpts{
          Name: "http_requests_total",
          Help: "Total HTTP requests",
      },
      []string{"method", "path", "status"},
  )
  
  // Histogram (distribution)
  requestDuration := prometheus.NewHistogramVec(
      prometheus.HistogramOpts{
          Name:    "http_request_duration_seconds",
          Help:    "HTTP request duration",
          Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
      },
      []string{"method", "path"},
  )
  
  // Gauge (current value)
  activeConnections := prometheus.NewGauge(
      prometheus.GaugeOpts{
          Name: "active_connections",
          Help: "Number of active connections",
      },
  )
  
  // Usage in handler:
  func handler(w http.ResponseWriter, r *http.Request) {
      start := time.Now()
      // ... handle request
      duration := time.Since(start).Seconds()
      
      requestsTotal.WithLabelValues(r.Method, r.URL.Path, "200").Inc()
      requestDuration.WithLabelValues(r.Method, r.URL.Path).Observe(duration)
  }

RED method (for services):
  Rate:     requests per second
  Errors:   error rate
  Duration: response time distribution

USE method (for resources):
  Utilization: % time busy
  Saturation:  queue length
  Errors:      error count
` + "```" + ``,
					CodeExamples: `// Observability patterns: metrics, logging, tracing
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "math/rand"
    "os"
    "sort"
    "sync"
    "time"
)

// Structured Logger (simplified slog-like)
type LogLevel int

const (
    LevelDebug LogLevel = iota
    LevelInfo
    LevelWarn
    LevelError
)

func (l LogLevel) String() string {
    switch l {
    case LevelDebug: return "DEBUG"
    case LevelInfo:  return "INFO"
    case LevelWarn:  return "WARN"
    case LevelError: return "ERROR"
    default: return "UNKNOWN"
    }
}

type Logger struct {
    level  LogLevel
    fields map[string]any
    mu     sync.Mutex
}

func NewLogger(level LogLevel) *Logger {
    return &Logger{
        level:  level,
        fields: make(map[string]any),
    }
}

func (l *Logger) With(keyvals ...any) *Logger {
    fields := make(map[string]any)
    for k, v := range l.fields {
        fields[k] = v
    }
    for i := 0; i < len(keyvals)-1; i += 2 {
        fields[fmt.Sprint(keyvals[i])] = keyvals[i+1]
    }
    return &Logger{level: l.level, fields: fields}
}

func (l *Logger) log(level LogLevel, msg string, keyvals ...any) {
    if level < l.level {
        return
    }
    
    entry := map[string]any{
        "time":  time.Now().Format(time.RFC3339),
        "level": level.String(),
        "msg":   msg,
    }
    for k, v := range l.fields {
        entry[k] = v
    }
    for i := 0; i < len(keyvals)-1; i += 2 {
        entry[fmt.Sprint(keyvals[i])] = keyvals[i+1]
    }
    
    l.mu.Lock()
    data, _ := json.Marshal(entry)
    fmt.Fprintf(os.Stdout, "  %s\n", data)
    l.mu.Unlock()
}

func (l *Logger) Debug(msg string, keyvals ...any) { l.log(LevelDebug, msg, keyvals...) }
func (l *Logger) Info(msg string, keyvals ...any)  { l.log(LevelInfo, msg, keyvals...) }
func (l *Logger) Warn(msg string, keyvals ...any)  { l.log(LevelWarn, msg, keyvals...) }
func (l *Logger) Error(msg string, keyvals ...any) { l.log(LevelError, msg, keyvals...) }

// Metrics system
type MetricType int
const (
    CounterType MetricType = iota
    GaugeType
    HistogramType
)

type Counter struct {
    mu    sync.Mutex
    name  string
    value float64
    labels map[string]string
}

func (c *Counter) Inc() {
    c.mu.Lock()
    c.value++
    c.mu.Unlock()
}

func (c *Counter) Add(v float64) {
    c.mu.Lock()
    c.value += v
    c.mu.Unlock()
}

func (c *Counter) Value() float64 {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.value
}

type Gauge struct {
    mu    sync.Mutex
    name  string
    value float64
}

func (g *Gauge) Set(v float64) {
    g.mu.Lock()
    g.value = v
    g.mu.Unlock()
}

func (g *Gauge) Inc() {
    g.mu.Lock()
    g.value++
    g.mu.Unlock()
}

func (g *Gauge) Dec() {
    g.mu.Lock()
    g.value--
    g.mu.Unlock()
}

func (g *Gauge) Value() float64 {
    g.mu.Lock()
    defer g.mu.Unlock()
    return g.value
}

type Histogram struct {
    mu      sync.Mutex
    name    string
    buckets []float64
    counts  []int
    sum     float64
    count   int
}

func NewHistogram(name string, buckets []float64) *Histogram {
    sort.Float64s(buckets)
    return &Histogram{
        name:    name,
        buckets: buckets,
        counts:  make([]int, len(buckets)+1), // +1 for +Inf
    }
}

func (h *Histogram) Observe(v float64) {
    h.mu.Lock()
    defer h.mu.Unlock()
    h.sum += v
    h.count++
    for i, b := range h.buckets {
        if v <= b {
            h.counts[i]++
            return
        }
    }
    h.counts[len(h.counts)-1]++ // +Inf bucket
}

func (h *Histogram) String() string {
    h.mu.Lock()
    defer h.mu.Unlock()
    
    result := fmt.Sprintf("  %s (count=%d, sum=%.3f, avg=%.3f)\n", h.name, h.count, h.sum, h.sum/float64(h.count))
    cumulative := 0
    for i, b := range h.buckets {
        cumulative += h.counts[i]
        result += fmt.Sprintf("    le=%.3f: %d (%.1f%%)\n", b, cumulative, float64(cumulative)/float64(h.count)*100)
    }
    cumulative += h.counts[len(h.counts)-1]
    result += fmt.Sprintf("    le=+Inf: %d (100%%)\n", cumulative)
    return result
}

// Distributed Tracing (simplified)
type Span struct {
    TraceID    string
    SpanID     string
    ParentID   string
    Name       string
    Service    string
    StartTime  time.Time
    EndTime    time.Time
    Status     string
    Attributes map[string]string
}

func (s *Span) Duration() time.Duration {
    return s.EndTime.Sub(s.StartTime)
}

type Tracer struct {
    mu    sync.Mutex
    spans []Span
}

type traceCtxKey struct{}

func NewTracer() *Tracer {
    return &Tracer{spans: make([]Span, 0)}
}

func (t *Tracer) StartSpan(ctx context.Context, name, service string) (context.Context, *Span) {
    span := &Span{
        TraceID:    fmt.Sprintf("trace-%d", rand.Intn(10000)),
        SpanID:     fmt.Sprintf("span-%d", rand.Intn(10000)),
        Name:       name,
        Service:    service,
        StartTime:  time.Now(),
        Attributes: make(map[string]string),
    }
    
    // Inherit trace ID from parent
    if parent, ok := ctx.Value(traceCtxKey{}).(*Span); ok {
        span.TraceID = parent.TraceID
        span.ParentID = parent.SpanID
    }
    
    return context.WithValue(ctx, traceCtxKey{}, span), span
}

func (t *Tracer) EndSpan(span *Span, status string) {
    span.EndTime = time.Now()
    span.Status = status
    
    t.mu.Lock()
    t.spans = append(t.spans, *span)
    t.mu.Unlock()
}

func (t *Tracer) PrintTrace(traceID string) {
    t.mu.Lock()
    defer t.mu.Unlock()
    
    fmt.Printf("\n  Trace: %s\n", traceID)
    for _, s := range t.spans {
        if s.TraceID == traceID {
            indent := "  "
            if s.ParentID != "" {
                indent = "    "
            }
            fmt.Printf("%s[%s] %s.%s %v (%s)\n",
                indent, s.SpanID, s.Service, s.Name, s.Duration(), s.Status)
        }
    }
}

func main() {
    // Structured Logging
    fmt.Println("=== Structured Logging ===")
    logger := NewLogger(LevelInfo)
    svcLogger := logger.With("service", "user-api", "version", "1.0")
    
    svcLogger.Info("server started", "port", 8080)
    
    reqLogger := svcLogger.With("request_id", "req-abc123", "user_id", "user-42")
    reqLogger.Info("handling request", "method", "GET", "path", "/api/users")
    reqLogger.Warn("slow query", "duration_ms", 250, "query", "SELECT * FROM users")
    reqLogger.Error("external service failed", "service", "payment-api", "status", 503)
    
    // Metrics
    fmt.Println("\n=== Metrics ===")
    
    // Counter
    requestsTotal := &Counter{name: "http_requests_total"}
    for i := 0; i < 100; i++ {
        requestsTotal.Inc()
    }
    fmt.Printf("  Requests total: %.0f\n", requestsTotal.Value())
    
    // Gauge
    activeConns := &Gauge{name: "active_connections"}
    activeConns.Set(42)
    activeConns.Inc()
    activeConns.Inc()
    activeConns.Dec()
    fmt.Printf("  Active connections: %.0f\n", activeConns.Value())
    
    // Histogram
    duration := NewHistogram("request_duration_seconds",
        []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0})
    
    for i := 0; i < 1000; i++ {
        d := rand.ExpFloat64() * 0.05
        duration.Observe(d)
    }
    fmt.Print(duration.String())
    
    // Distributed Tracing
    fmt.Println("\n=== Distributed Tracing ===")
    tracer := NewTracer()
    
    // Simulate a request flowing through services
    ctx := context.Background()
    
    // API Gateway
    ctx, gatewaySpan := tracer.StartSpan(ctx, "handle_request", "api-gateway")
    traceID := gatewaySpan.TraceID
    time.Sleep(1 * time.Millisecond)
    
    // User Service
    ctx2, userSpan := tracer.StartSpan(ctx, "get_user", "user-service")
    time.Sleep(2 * time.Millisecond)
    
    // Database
    _, dbSpan := tracer.StartSpan(ctx2, "SELECT", "postgres")
    time.Sleep(1 * time.Millisecond)
    tracer.EndSpan(dbSpan, "OK")
    
    tracer.EndSpan(userSpan, "OK")
    
    // Order Service
    ctx3, orderSpan := tracer.StartSpan(ctx, "list_orders", "order-service")
    time.Sleep(3 * time.Millisecond)
    
    // Cache
    _, cacheSpan := tracer.StartSpan(ctx3, "GET", "redis")
    time.Sleep(500 * time.Microsecond)
    tracer.EndSpan(cacheSpan, "MISS")
    
    // Database
    _, dbSpan2 := tracer.StartSpan(ctx3, "SELECT", "postgres")
    time.Sleep(2 * time.Millisecond)
    tracer.EndSpan(dbSpan2, "OK")
    
    tracer.EndSpan(orderSpan, "OK")
    
    time.Sleep(1 * time.Millisecond)
    tracer.EndSpan(gatewaySpan, "OK")
    
    tracer.PrintTrace(traceID)
}`,
				},
			},
		},
	})
}
