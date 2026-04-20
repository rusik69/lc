package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          2314,
			Title:       "Event-Driven Architecture",
			Description: "Design event-driven systems with event sourcing, CQRS, message brokers, saga patterns, and eventual consistency for scalable distributed applications.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "Event Sourcing and CQRS",
					Content: `Event Sourcing stores the state of an entity as a sequence of state-changing events rather than the current state.

**Event Sourcing Fundamentals:**

Traditional State Storage:
  Store current state: Account { balance: 500 }
  History is lost - we only know the final state

Event Sourcing:
  Store all events that led to current state:
    AccountCreated { id: "acc-1", owner: "Alice" }
    MoneyDeposited { amount: 1000 }
    MoneyWithdrawn { amount: 300 }
    MoneyDeposited { amount: 200 }
    MoneyWithdrawn { amount: 400 }
  
  Current state: Replay all events -> balance = 1000 - 300 + 200 - 400 = 500
  Complete audit trail preserved

Benefits of Event Sourcing:
  Complete audit trail - every change recorded
  Temporal queries - state at any point in time
  Event replay - rebuild state or create new projections
  Debugging - replay events to reproduce bugs
  Domain events as first-class citizens
  Supports CQRS naturally

Challenges:
  Event schema evolution (versioning events)
  Eventual consistency in read models
  Event store performance (snapshots needed)
  Complexity - steeper learning curve
  Not suitable for all domains

Event Store Structure:
  Stream: Sequence of events for an aggregate
  Stream ID: Usually aggregate type + aggregate ID
  Event: Immutable record with type, data, metadata
  Position: Global ordering of events
  Version: Per-stream ordering

  events table:
    stream_id    | version | event_type      | data              | metadata          | timestamp
    order-123    | 1       | OrderCreated    | {customer: "abc"} | {user: "admin"}   | 2024-01-01T10:00:00
    order-123    | 2       | ItemAdded       | {product: "xyz"}  | {user: "admin"}   | 2024-01-01T10:01:00
    order-123    | 3       | OrderPlaced     | {total: 5000}     | {user: "admin"}   | 2024-01-01T10:02:00

Snapshots:
  Periodically store materialized state
  Replay only events after last snapshot
  Improves read performance for aggregates with many events
  
  Snapshot strategy:
    Every N events (e.g., every 100 events)
    Time-based (e.g., daily)
    On-demand when aggregate is loaded

Event Versioning:
  Events are immutable - never modify published events
  Schema changes require versioning strategy:
    Weak schema: Use flexible format (JSON with optional fields)
    Upcasting: Transform old events to new format on read
    New event types: Introduce new events, keep old ones
    Copy-and-transform: Migrate event store to new schema

**CQRS (Command Query Responsibility Segregation):**

Separates read and write models:

  Command Side (Write):
    Receives commands (intent to change state)
    Validates business rules
    Produces domain events
    Optimized for writes (normalized, consistent)
    One model for the domain

  Query Side (Read):
    Subscribes to domain events
    Builds projections (read models)
    Optimized for specific queries (denormalized)
    Multiple models for different read needs
    Eventually consistent with write side

CQRS + Event Sourcing Flow:
  1. Client sends Command
  2. Command Handler validates and executes
  3. Aggregate produces Domain Events
  4. Events stored in Event Store
  5. Event Handlers update Read Models (projections)
  6. Client queries Read Models for data

When to use CQRS:
  Different read and write patterns
  High-performance read requirements
  Complex domain with many aggregates
  Collaborative domains with many concurrent users
  Event-driven architecture already in use

When NOT to use CQRS:
  Simple CRUD applications
  Small teams with limited experience
  Strong consistency requirements everywhere
  Low read/write ratio differences`,
					CodeExamples: `// Event Sourcing implementation

// Event types
type Event interface {
    EventType() string
    AggregateID() string
}

type BaseEvent struct {
    ID          string
    Type        string
    AggregateId string
    Version     int
    Timestamp   time.Time
    Data        interface{}
}

func (e BaseEvent) EventType() string   { return e.Type }
func (e BaseEvent) AggregateID() string { return e.AggregateId }

// Domain Events
type AccountCreated struct {
    AccountID string
    OwnerName string
    Currency  string
}

type MoneyDeposited struct {
    AccountID string
    Amount    int64
    Reference string
}

type MoneyWithdrawn struct {
    AccountID string
    Amount    int64
    Reference string
}

// Aggregate with Event Sourcing
type BankAccount struct {
    id       string
    owner    string
    balance  int64
    currency string
    version  int
    changes  []Event // uncommitted events
}

func NewBankAccountFromEvents(events []Event) *BankAccount {
    account := &BankAccount{}
    for _, event := range events {
        account.apply(event, false)
    }
    return account
}

func (a *BankAccount) apply(event Event, isNew bool) {
    switch e := event.(type) {
    case AccountCreated:
        a.id = e.AccountID
        a.owner = e.OwnerName
        a.currency = e.Currency
        a.balance = 0
    case MoneyDeposited:
        a.balance += e.Amount
    case MoneyWithdrawn:
        a.balance -= e.Amount
    }
    a.version++
    if isNew {
        a.changes = append(a.changes, event)
    }
}

func CreateBankAccount(id, owner, currency string) *BankAccount {
    account := &BankAccount{}
    account.apply(AccountCreated{
        AccountID: id,
        OwnerName: owner,
        Currency:  currency,
    }, true)
    return account
}

func (a *BankAccount) Deposit(amount int64, reference string) error {
    if amount <= 0 {
        return errors.New("deposit amount must be positive")
    }
    a.apply(MoneyDeposited{
        AccountID: a.id,
        Amount:    amount,
        Reference: reference,
    }, true)
    return nil
}

func (a *BankAccount) Withdraw(amount int64, reference string) error {
    if amount <= 0 {
        return errors.New("withdrawal amount must be positive")
    }
    if a.balance < amount {
        return errors.New("insufficient funds")
    }
    a.apply(MoneyWithdrawn{
        AccountID: a.id,
        Amount:    amount,
        Reference: reference,
    }, true)
    return nil
}

func (a *BankAccount) UncommittedChanges() []Event {
    return a.changes
}

func (a *BankAccount) ClearChanges() {
    a.changes = nil
}

// Event Store interface
type EventStore interface {
    AppendEvents(ctx context.Context, streamID string, expectedVersion int, events []Event) error
    LoadEvents(ctx context.Context, streamID string) ([]Event, error)
    LoadEventsFrom(ctx context.Context, streamID string, fromVersion int) ([]Event, error)
    Subscribe(ctx context.Context, handler func(Event)) error
}

// CQRS - Command Handler
type DepositCommand struct {
    AccountID string
    Amount    int64
    Reference string
}

type DepositHandler struct {
    eventStore EventStore
}

func (h *DepositHandler) Handle(ctx context.Context, cmd DepositCommand) error {
    events, err := h.eventStore.LoadEvents(ctx, "account-"+cmd.AccountID)
    if err != nil {
        return err
    }

    account := NewBankAccountFromEvents(events)
    if err := account.Deposit(cmd.Amount, cmd.Reference); err != nil {
        return err
    }

    return h.eventStore.AppendEvents(ctx,
        "account-"+cmd.AccountID,
        account.version-len(account.UncommittedChanges()),
        account.UncommittedChanges(),
    )
}

// CQRS - Read Model (Projection)
type AccountSummary struct {
    AccountID    string
    OwnerName    string
    Balance      int64
    Currency     string
    TotalDeposits  int
    TotalWithdrawals int
    LastActivity time.Time
}

type AccountProjection struct {
    store map[string]*AccountSummary
    mu    sync.RWMutex
}

func (p *AccountProjection) HandleEvent(event Event) {
    p.mu.Lock()
    defer p.mu.Unlock()

    switch e := event.(type) {
    case AccountCreated:
        p.store[e.AccountID] = &AccountSummary{
            AccountID: e.AccountID,
            OwnerName: e.OwnerName,
            Currency:  e.Currency,
        }
    case MoneyDeposited:
        if summary, ok := p.store[e.AccountID]; ok {
            summary.Balance += e.Amount
            summary.TotalDeposits++
            summary.LastActivity = time.Now()
        }
    case MoneyWithdrawn:
        if summary, ok := p.store[e.AccountID]; ok {
            summary.Balance -= e.Amount
            summary.TotalWithdrawals++
            summary.LastActivity = time.Now()
        }
    }
}

func (p *AccountProjection) GetSummary(accountID string) (*AccountSummary, error) {
    p.mu.RLock()
    defer p.mu.RUnlock()
    summary, ok := p.store[accountID]
    if !ok {
        return nil, errors.New("account not found")
    }
    return summary, nil
}`,
				},
				{
					Title: "Saga Pattern and Distributed Transactions",
					Content: `The Saga pattern manages distributed transactions across multiple services without requiring two-phase commit.

**Saga Types:**

Choreography-based Saga:
  Each service listens for events and publishes events
  No central coordinator
  Decentralized decision making
  
  Example - Order Processing:
    1. Order Service: OrderCreated event
    2. Payment Service: Listens, processes payment -> PaymentCompleted event
    3. Inventory Service: Listens, reserves stock -> StockReserved event
    4. Shipping Service: Listens, creates shipment -> ShipmentCreated event
    
  Compensation (on failure):
    If Payment fails: Order Service cancels order
    If Stock fails: Payment Service refunds, Order Service cancels
    If Shipping fails: Inventory restores stock, Payment refunds, Order cancels

  Pros:
    Simple for few steps
    Loosely coupled services
    No single point of failure
  
  Cons:
    Hard to track overall saga state
    Difficult to add new steps
    Risk of cyclic dependencies
    Harder to understand full flow

Orchestration-based Saga:
  Central orchestrator coordinates the saga steps
  Tells each service what to do
  Centralized decision making
  
  Example - Order Processing:
    Orchestrator (Order Saga):
      Step 1: Tell Payment Service -> ProcessPayment
      Step 2: Tell Inventory Service -> ReserveStock
      Step 3: Tell Shipping Service -> CreateShipment
      Step 4: Tell Notification Service -> SendConfirmation
    
    On failure at any step:
      Execute compensating transactions in reverse order

  Pros:
    Easy to understand and maintain
    Easy to add new steps
    Clear visibility of saga state
    Centralized error handling
  
  Cons:
    Orchestrator can become a bottleneck
    Single point of failure (mitigated by redundancy)
    More coupling to orchestrator

**Compensating Transactions:**

Each saga step must have a compensating action:
  ProcessPayment -> RefundPayment
  ReserveInventory -> ReleaseInventory
  CreateShipment -> CancelShipment
  SendEmail -> SendCancellationEmail

Properties of compensating transactions:
  Must be idempotent (safe to retry)
  Must be retried until successful (or manually handled)
  May not perfectly undo (semantic compensation)
  Should be commutative when possible

**Saga Execution Coordinator (SEC):**
  Persists saga state
  Handles retries and timeouts
  Manages compensation
  Provides observability

Saga States:
  STARTED -> STEP_1_PENDING -> STEP_1_COMPLETED -> 
  STEP_2_PENDING -> STEP_2_COMPLETED -> ... -> COMPLETED
  
  On failure:
  STEP_N_FAILED -> COMPENSATING -> COMPENSATED -> FAILED

**Idempotency:**
  All saga participants must be idempotent
  Same request processed multiple times = same result
  Use idempotency keys (request ID, saga ID + step)
  Store processed request IDs to detect duplicates

**Timeout Handling:**
  Set timeouts for each saga step
  On timeout: Retry or compensate
  Use exponential backoff for retries
  Maximum retry count before compensation
  Dead letter queue for unprocessable messages`,
					CodeExamples: `// Orchestration-based Saga implementation

type SagaStatus string
const (
    SagaStarted      SagaStatus = "started"
    SagaRunning      SagaStatus = "running"
    SagaCompleted    SagaStatus = "completed"
    SagaCompensating SagaStatus = "compensating"
    SagaFailed       SagaStatus = "failed"
)

type SagaStep struct {
    Name       string
    Execute    func(ctx context.Context, data map[string]interface{}) error
    Compensate func(ctx context.Context, data map[string]interface{}) error
}

type SagaState struct {
    ID            string
    Name          string
    Status        SagaStatus
    CurrentStep   int
    Data          map[string]interface{}
    CompletedSteps []string
    Error         string
    CreatedAt     time.Time
    UpdatedAt     time.Time
}

type SagaOrchestrator struct {
    steps    []SagaStep
    store    SagaStateStore
    name     string
}

func NewSagaOrchestrator(name string, store SagaStateStore) *SagaOrchestrator {
    return &SagaOrchestrator{name: name, store: store}
}

func (s *SagaOrchestrator) AddStep(step SagaStep) {
    s.steps = append(s.steps, step)
}

func (s *SagaOrchestrator) Execute(ctx context.Context, data map[string]interface{}) error {
    state := &SagaState{
        ID:        generateID(),
        Name:      s.name,
        Status:    SagaStarted,
        Data:      data,
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }

    if err := s.store.Save(ctx, state); err != nil {
        return fmt.Errorf("save saga state: %w", err)
    }

    // Execute steps forward
    for i, step := range s.steps {
        state.CurrentStep = i
        state.Status = SagaRunning
        state.UpdatedAt = time.Now()
        s.store.Save(ctx, state)

        if err := s.executeWithRetry(ctx, step.Execute, state.Data, 3); err != nil {
            state.Error = err.Error()
            state.Status = SagaCompensating
            s.store.Save(ctx, state)
            
            // Compensate completed steps in reverse
            return s.compensate(ctx, state, i-1)
        }

        state.CompletedSteps = append(state.CompletedSteps, step.Name)
        s.store.Save(ctx, state)
    }

    state.Status = SagaCompleted
    state.UpdatedAt = time.Now()
    return s.store.Save(ctx, state)
}

func (s *SagaOrchestrator) compensate(ctx context.Context, state *SagaState, fromStep int) error {
    for i := fromStep; i >= 0; i-- {
        step := s.steps[i]
        if step.Compensate == nil {
            continue
        }

        if err := s.executeWithRetry(ctx, step.Compensate, state.Data, 5); err != nil {
            state.Status = SagaFailed
            state.Error = fmt.Sprintf("compensation failed at step %s: %v", step.Name, err)
            s.store.Save(ctx, state)
            return fmt.Errorf("compensation failed: %w", err)
        }
    }

    state.Status = SagaFailed
    state.UpdatedAt = time.Now()
    return s.store.Save(ctx, state)
}

func (s *SagaOrchestrator) executeWithRetry(ctx context.Context, fn func(context.Context, map[string]interface{}) error, data map[string]interface{}, maxRetries int) error {
    var lastErr error
    for attempt := 0; attempt <= maxRetries; attempt++ {
        if err := fn(ctx, data); err != nil {
            lastErr = err
            backoff := time.Duration(attempt*attempt) * 100 * time.Millisecond
            select {
            case <-time.After(backoff):
            case <-ctx.Done():
                return ctx.Err()
            }
            continue
        }
        return nil
    }
    return lastErr
}

// Usage: Order Processing Saga
func NewOrderSaga(
    payment PaymentService,
    inventory InventoryService,
    shipping ShippingService,
    notification NotificationService,
    store SagaStateStore,
) *SagaOrchestrator {
    saga := NewSagaOrchestrator("order-processing", store)

    saga.AddStep(SagaStep{
        Name: "process-payment",
        Execute: func(ctx context.Context, data map[string]interface{}) error {
            orderID := data["order_id"].(string)
            amount := data["total"].(int64)
            txnID, err := payment.ProcessPayment(ctx, orderID, amount)
            if err != nil {
                return err
            }
            data["payment_txn_id"] = txnID
            return nil
        },
        Compensate: func(ctx context.Context, data map[string]interface{}) error {
            txnID := data["payment_txn_id"].(string)
            return payment.Refund(ctx, txnID)
        },
    })

    saga.AddStep(SagaStep{
        Name: "reserve-inventory",
        Execute: func(ctx context.Context, data map[string]interface{}) error {
            orderID := data["order_id"].(string)
            items := data["items"].([]OrderItem)
            reservationID, err := inventory.Reserve(ctx, orderID, items)
            if err != nil {
                return err
            }
            data["reservation_id"] = reservationID
            return nil
        },
        Compensate: func(ctx context.Context, data map[string]interface{}) error {
            reservationID := data["reservation_id"].(string)
            return inventory.Release(ctx, reservationID)
        },
    })

    saga.AddStep(SagaStep{
        Name: "create-shipment",
        Execute: func(ctx context.Context, data map[string]interface{}) error {
            orderID := data["order_id"].(string)
            address := data["shipping_address"].(Address)
            shipmentID, err := shipping.CreateShipment(ctx, orderID, address)
            if err != nil {
                return err
            }
            data["shipment_id"] = shipmentID
            return nil
        },
        Compensate: func(ctx context.Context, data map[string]interface{}) error {
            shipmentID := data["shipment_id"].(string)
            return shipping.CancelShipment(ctx, shipmentID)
        },
    })

    saga.AddStep(SagaStep{
        Name: "send-confirmation",
        Execute: func(ctx context.Context, data map[string]interface{}) error {
            orderID := data["order_id"].(string)
            customerID := data["customer_id"].(string)
            return notification.SendOrderConfirmation(ctx, customerID, orderID)
        },
        Compensate: nil, // No compensation needed for notification
    })

    return saga
}`,
				},
			},
		},
		{
			ID:          2315,
			Title:       "Microservices Communication Patterns",
			Description: "Design effective inter-service communication with synchronous and asynchronous patterns, API gateways, service mesh, circuit breakers, and back-pressure handling.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "Synchronous and Asynchronous Communication",
					Content: `Microservices communicate through various patterns, each with different trade-offs for coupling, latency, and reliability.

**Synchronous Communication:**

Request-Response:
  Client sends request, waits for response
  Simple mental model
  Creates temporal coupling (both services must be available)
  
  Protocols:
    REST/HTTP: Resource-based, widely understood
    gRPC: Binary protocol, strongly typed, streaming support
    GraphQL: Flexible queries, schema-based

  REST Best Practices:
    Use meaningful HTTP methods: GET, POST, PUT, PATCH, DELETE
    Return appropriate status codes: 200, 201, 204, 400, 404, 500
    Version APIs: /api/v1/orders
    Use HATEOAS for discoverability
    Implement pagination for collections
    Use ETags for caching

  gRPC Best Practices:
    Define clear proto contracts
    Use streaming for large datasets
    Implement health checks
    Set appropriate deadlines/timeouts
    Use interceptors for cross-cutting concerns

**Asynchronous Communication:**

Message-based:
  Producer sends message, doesn't wait for response
  Temporal decoupling (consumer processes later)
  Requires message broker (Kafka, RabbitMQ, NATS)

  Patterns:
    Point-to-Point (Queue):
      One producer, one consumer
      Message consumed exactly once
      Work distribution / task queue
      Example: Order processing queue

    Publish-Subscribe (Topic):
      One publisher, many subscribers
      Each subscriber gets a copy
      Event notification
      Example: OrderPlaced event consumed by multiple services

    Request-Reply (async):
      Request via queue, reply via correlation ID
      Decoupled but maintains request-reply semantics
      Example: Async validation request

Message Delivery Guarantees:
  At-most-once: Fire and forget, may lose messages
  At-least-once: Retry until acknowledged, may duplicate
  Exactly-once: Complex, requires idempotent consumers + dedup

**API Gateway Pattern:**

Single entry point for all client requests:
  Routing: Forward requests to appropriate services
  Composition: Aggregate responses from multiple services
  Authentication: Verify tokens, enforce policies
  Rate limiting: Protect services from overload
  Caching: Cache frequent responses
  Protocol translation: REST to gRPC, WebSocket handling
  Load balancing: Distribute across service instances

Gateway types:
  Simple proxy: Kong, Nginx, HAProxy
  Full gateway: AWS API Gateway, Apigee
  BFF (Backend for Frontend): Specific gateway per client type

BFF Pattern:
  Web BFF: Optimized for web clients (larger payloads OK)
  Mobile BFF: Optimized for mobile (smaller payloads, less data)
  IoT BFF: Optimized for constrained devices

**Service Mesh:**

Infrastructure layer for service-to-service communication:
  Sidecar proxy handles:
    Load balancing
    Service discovery
    TLS encryption (mTLS)
    Circuit breaking
    Retries and timeouts
    Observability (metrics, tracing, logging)
    Traffic management (canary, blue-green)

  Control plane manages:
    Proxy configuration
    Certificate management
    Policy enforcement
    Service registry

  Implementations: Istio, Linkerd, Consul Connect

  Benefits:
    No code changes in services
    Consistent security policies
    Unified observability
    Traffic control without app changes
  
  Costs:
    Operational complexity
    Latency overhead (proxy hop)
    Resource consumption (sidecar per pod)
    Debugging complexity`,
					CodeExamples: `// Communication patterns implementation

// Circuit Breaker pattern
type CircuitState int
const (
    CircuitClosed CircuitState = iota
    CircuitOpen
    CircuitHalfOpen
)

type CircuitBreaker struct {
    name          string
    maxFailures   int
    timeout       time.Duration
    halfOpenMax   int
    
    mu            sync.Mutex
    state         CircuitState
    failures      int
    successes     int
    lastFailure   time.Time
    halfOpenCalls int
}

func NewCircuitBreaker(name string, maxFailures int, timeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        name:        name,
        maxFailures: maxFailures,
        timeout:     timeout,
        halfOpenMax: 1,
        state:       CircuitClosed,
    }
}

func (cb *CircuitBreaker) Execute(fn func() error) error {
    if !cb.allowRequest() {
        return fmt.Errorf("circuit breaker %s is open", cb.name)
    }

    err := fn()

    cb.mu.Lock()
    defer cb.mu.Unlock()

    if err != nil {
        cb.failures++
        cb.lastFailure = time.Now()
        if cb.state == CircuitHalfOpen || cb.failures >= cb.maxFailures {
            cb.state = CircuitOpen
        }
        return err
    }

    if cb.state == CircuitHalfOpen {
        cb.successes++
        if cb.successes >= cb.halfOpenMax {
            cb.state = CircuitClosed
            cb.failures = 0
            cb.successes = 0
        }
    } else {
        cb.failures = 0
    }

    return nil
}

func (cb *CircuitBreaker) allowRequest() bool {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    switch cb.state {
    case CircuitClosed:
        return true
    case CircuitOpen:
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.state = CircuitHalfOpen
            cb.successes = 0
            cb.halfOpenCalls = 0
            return true
        }
        return false
    case CircuitHalfOpen:
        cb.halfOpenCalls++
        return cb.halfOpenCalls <= cb.halfOpenMax
    }
    return false
}

// Retry with exponential backoff
type RetryConfig struct {
    MaxRetries  int
    InitialWait time.Duration
    MaxWait     time.Duration
    Multiplier  float64
}

func RetryWithBackoff(ctx context.Context, config RetryConfig, fn func() error) error {
    var lastErr error
    wait := config.InitialWait

    for attempt := 0; attempt <= config.MaxRetries; attempt++ {
        if err := fn(); err != nil {
            lastErr = err
            if attempt == config.MaxRetries {
                break
            }
            // Add jitter
            jitter := time.Duration(rand.Int63n(int64(wait) / 2))
            select {
            case <-time.After(wait + jitter):
            case <-ctx.Done():
                return ctx.Err()
            }
            wait = time.Duration(float64(wait) * config.Multiplier)
            if wait > config.MaxWait {
                wait = config.MaxWait
            }
            continue
        }
        return nil
    }
    return fmt.Errorf("max retries exceeded: %w", lastErr)
}

// Bulkhead pattern - limit concurrent requests
type Bulkhead struct {
    name string
    sem  chan struct{}
}

func NewBulkhead(name string, maxConcurrent int) *Bulkhead {
    return &Bulkhead{
        name: name,
        sem:  make(chan struct{}, maxConcurrent),
    }
}

func (b *Bulkhead) Execute(ctx context.Context, fn func() error) error {
    select {
    case b.sem <- struct{}{}:
        defer func() { <-b.sem }()
        return fn()
    case <-ctx.Done():
        return fmt.Errorf("bulkhead %s: %w", b.name, ctx.Err())
    }
}

// Service client with resilience patterns
type ServiceClient struct {
    httpClient     *http.Client
    circuitBreaker *CircuitBreaker
    bulkhead       *Bulkhead
    retryConfig    RetryConfig
    baseURL        string
}

func (c *ServiceClient) Get(ctx context.Context, path string, result interface{}) error {
    return c.bulkhead.Execute(ctx, func() error {
        return c.circuitBreaker.Execute(func() error {
            return RetryWithBackoff(ctx, c.retryConfig, func() error {
                req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+path, nil)
                if err != nil {
                    return err
                }
                resp, err := c.httpClient.Do(req)
                if err != nil {
                    return err
                }
                defer resp.Body.Close()
                if resp.StatusCode >= 500 {
                    return fmt.Errorf("server error: %d", resp.StatusCode)
                }
                if resp.StatusCode >= 400 {
                    return fmt.Errorf("client error: %d", resp.StatusCode)
                }
                return json.NewDecoder(resp.Body).Decode(result)
            })
        })
    })
}

// Timeout propagation
func withTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
    // Respect existing deadline if shorter
    if deadline, ok := ctx.Deadline(); ok {
        remaining := time.Until(deadline)
        if remaining < timeout {
            timeout = remaining
        }
    }
    return context.WithTimeout(ctx, timeout)
}`,
				},
			},
		},
	})
}
