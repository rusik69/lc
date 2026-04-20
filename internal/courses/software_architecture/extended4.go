package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          2316,
			Title:       "Design Patterns in Practice",
			Description: "Apply Gang of Four and modern design patterns including creational, structural, and behavioral patterns with practical refactoring examples.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "Creational and Structural Patterns",
					Content: `Design patterns are reusable solutions to common software design problems. Understanding when and why to apply them is more important than memorizing implementations.

**Creational Patterns:**

Builder Pattern:
  Separates construction of a complex object from its representation.
  Useful when an object has many optional parameters.
  Chain method calls for fluent API.
  
  When to use:
    Object has more than 3-4 constructor parameters
    Many optional parameters
    Object requires complex initialization sequence
    Need to create immutable objects with many fields

Factory Method / Abstract Factory:
  Creates objects without specifying exact class.
  Delegates creation to subclasses or specialized factories.
  
  When to use:
    Don't know exact types in advance
    Want to provide extension point for object creation
    Need to decouple creation from usage
    Configuration-based object creation

Singleton:
  Ensures a class has only one instance.
  USE WITH CAUTION - often an anti-pattern.
  Better alternative: dependency injection with scoped lifetime.
  
  When acceptable:
    Configuration that's loaded once
    Connection pools
    Logger instances
  
  When to avoid:
    When it hides dependencies
    When it makes testing difficult
    When shared mutable state causes issues

Prototype:
  Creates objects by cloning existing instances.
  Useful when creation is expensive.
  Deep copy vs shallow copy considerations.

**Structural Patterns:**

Adapter:
  Converts the interface of a class into another interface clients expect.
  Wraps an existing class with a new interface.
  
  When to use:
    Integrating with third-party libraries
    Legacy system integration
    Conforming to expected interfaces

Decorator:
  Adds behavior to objects dynamically.
  Wraps object with additional functionality.
  More flexible than inheritance for adding behavior.
  
  When to use:
    Adding cross-cutting concerns (logging, caching, auth)
    Combining behaviors dynamically
    Open/Closed principle compliance

Facade:
  Provides a simplified interface to a complex subsystem.
  Reduces coupling between client and subsystem.
  
  When to use:
    Complex library with many entry points
    Layer boundaries
    Simplifying external API

Proxy:
  Controls access to another object.
  Types: virtual proxy (lazy loading), protection proxy (access control),
         remote proxy (network access), caching proxy.

Composite:
  Composes objects into tree structures.
  Treats individual objects and compositions uniformly.
  
  When to use:
    Tree structures (file systems, org charts, UI components)
    Recursive compositions
    Uniform treatment of parts and wholes

Bridge:
  Decouples abstraction from its implementation.
  Both can vary independently.
  
  When to use:
    Multiple dimensions of variation
    Avoiding class explosion from combinations
    Runtime switching of implementations`,
					CodeExamples: `// Design patterns in Go

// Builder Pattern
type ServerConfig struct {
    Host         string
    Port         int
    ReadTimeout  time.Duration
    WriteTimeout time.Duration
    MaxConns     int
    TLS          *TLSConfig
    Middleware    []Middleware
    Logger       Logger
}

type ServerBuilder struct {
    config ServerConfig
}

func NewServerBuilder(host string, port int) *ServerBuilder {
    return &ServerBuilder{
        config: ServerConfig{
            Host:         host,
            Port:         port,
            ReadTimeout:  30 * time.Second,
            WriteTimeout: 30 * time.Second,
            MaxConns:     1000,
        },
    }
}

func (b *ServerBuilder) WithReadTimeout(d time.Duration) *ServerBuilder {
    b.config.ReadTimeout = d
    return b
}

func (b *ServerBuilder) WithWriteTimeout(d time.Duration) *ServerBuilder {
    b.config.WriteTimeout = d
    return b
}

func (b *ServerBuilder) WithMaxConns(n int) *ServerBuilder {
    b.config.MaxConns = n
    return b
}

func (b *ServerBuilder) WithTLS(certFile, keyFile string) *ServerBuilder {
    b.config.TLS = &TLSConfig{CertFile: certFile, KeyFile: keyFile}
    return b
}

func (b *ServerBuilder) WithMiddleware(mw ...Middleware) *ServerBuilder {
    b.config.Middleware = append(b.config.Middleware, mw...)
    return b
}

func (b *ServerBuilder) WithLogger(logger Logger) *ServerBuilder {
    b.config.Logger = logger
    return b
}

func (b *ServerBuilder) Build() (*Server, error) {
    if b.config.Host == "" {
        return nil, errors.New("host is required")
    }
    if b.config.Port <= 0 {
        return nil, errors.New("port must be positive")
    }
    return &Server{config: b.config}, nil
}

// Usage
// server, err := NewServerBuilder("localhost", 8080).
//     WithReadTimeout(10 * time.Second).
//     WithTLS("cert.pem", "key.pem").
//     WithMiddleware(LoggingMiddleware, AuthMiddleware).
//     Build()

// Functional Options Pattern (Go-idiomatic Builder alternative)
type Option func(*ServerConfig)

func WithTimeout(read, write time.Duration) Option {
    return func(c *ServerConfig) {
        c.ReadTimeout = read
        c.WriteTimeout = write
    }
}

func WithMaxConnections(n int) Option {
    return func(c *ServerConfig) { c.MaxConns = n }
}

func NewServer(host string, port int, opts ...Option) *Server {
    config := ServerConfig{
        Host: host, Port: port,
        ReadTimeout: 30 * time.Second, WriteTimeout: 30 * time.Second,
        MaxConns: 1000,
    }
    for _, opt := range opts {
        opt(&config)
    }
    return &Server{config: config}
}

// Decorator Pattern
type HTTPHandler func(http.ResponseWriter, *http.Request)

// Logging decorator
func WithLogging(logger Logger, handler HTTPHandler) HTTPHandler {
    return func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        logger.Info("request started", "method", r.Method, "path", r.URL.Path)
        handler(w, r)
        logger.Info("request completed", "method", r.Method, "path", r.URL.Path, "duration", time.Since(start))
    }
}

// Auth decorator
func WithAuth(auth AuthService, handler HTTPHandler) HTTPHandler {
    return func(w http.ResponseWriter, r *http.Request) {
        token := r.Header.Get("Authorization")
        if !auth.ValidateToken(token) {
            http.Error(w, "unauthorized", http.StatusUnauthorized)
            return
        }
        handler(w, r)
    }
}

// Rate limiting decorator
func WithRateLimit(limiter RateLimiter, handler HTTPHandler) HTTPHandler {
    return func(w http.ResponseWriter, r *http.Request) {
        if !limiter.Allow(r.RemoteAddr) {
            http.Error(w, "too many requests", http.StatusTooManyRequests)
            return
        }
        handler(w, r)
    }
}

// Composing decorators
// handler := WithLogging(logger, WithAuth(auth, WithRateLimit(limiter, myHandler)))

// Adapter Pattern
type OldPaymentGateway struct{}
func (g *OldPaymentGateway) MakePayment(cardNo string, amountCents int, curr string) (string, error) {
    return "txn-123", nil  // old interface
}

// New interface our system expects
type PaymentProcessor interface {
    ProcessPayment(ctx context.Context, payment Payment) (TransactionResult, error)
}

// Adapter
type OldGatewayAdapter struct {
    gateway *OldPaymentGateway
}

func (a *OldGatewayAdapter) ProcessPayment(ctx context.Context, payment Payment) (TransactionResult, error) {
    txnID, err := a.gateway.MakePayment(
        payment.CardNumber,
        int(payment.Amount.Amount),
        payment.Amount.Currency,
    )
    if err != nil {
        return TransactionResult{}, err
    }
    return TransactionResult{
        TransactionID: txnID,
        Status:        "completed",
        ProcessedAt:   time.Now(),
    }, nil
}

// Strategy Pattern
type CompressionStrategy interface {
    Compress(data []byte) ([]byte, error)
    Decompress(data []byte) ([]byte, error)
}

type GzipCompression struct{}
func (g *GzipCompression) Compress(data []byte) ([]byte, error) { /* gzip impl */ return data, nil }
func (g *GzipCompression) Decompress(data []byte) ([]byte, error) { return data, nil }

type ZstdCompression struct{}
func (z *ZstdCompression) Compress(data []byte) ([]byte, error) { /* zstd impl */ return data, nil }
func (z *ZstdCompression) Decompress(data []byte) ([]byte, error) { return data, nil }

type FileStorage struct {
    compression CompressionStrategy
}

func (fs *FileStorage) Store(name string, data []byte) error {
    compressed, err := fs.compression.Compress(data)
    if err != nil {
        return err
    }
    return os.WriteFile(name, compressed, 0644)
}`,
				},
				{
					Title: "Behavioral Patterns and Modern Patterns",
					Content: `Behavioral patterns focus on communication between objects and assignment of responsibilities.

**Observer Pattern:**
  Defines a one-to-many dependency between objects.
  When one object changes state, all dependents are notified.
  Foundation for event-driven systems and reactive programming.
  
  Use when:
    Changes in one object require changing others
    Number of dependent objects unknown at compile time
    Loose coupling between subject and observers

**Strategy Pattern:**
  Defines a family of algorithms and makes them interchangeable.
  Algorithm can be selected at runtime.
  
  Use when:
    Multiple algorithms for a task
    Algorithm selection based on context
    Avoiding conditional logic for algorithm selection

**Command Pattern:**
  Encapsulates a request as an object.
  Enables parameterization, queueing, and undo/redo.
  
  Use when:
    Need to queue or log requests
    Need undo/redo functionality
    Need to decouple sender from handler
    CQRS command handling

**Chain of Responsibility:**
  Passes requests along a chain of handlers.
  Each handler either processes or passes to next.
  
  Use when:
    Multiple objects may handle a request
    Handler not known in advance
    Request should be handled by first capable handler
    Middleware pipelines (HTTP, message processing)

**Template Method:**
  Defines algorithm skeleton with steps overridden by subclasses.
  In Go, use interfaces with default implementations or function fields.
  
  Use when:
    Algorithm structure is fixed but steps vary
    Common behavior across related implementations
    Hook methods for extensibility

**State Pattern:**
  Allows object to change behavior when internal state changes.
  Object appears to change its class.
  
  Use when:
    Object behavior depends on state
    Complex conditional logic based on state
    State transitions are well-defined

**Modern Patterns:**

Repository Pattern:
  Mediates between domain and data mapping layers.
  Provides collection-like interface for domain objects.
  Decouples domain from persistence.

Unit of Work:
  Maintains list of objects affected by a business transaction.
  Coordinates writing out of changes.
  Ensures atomic write operations.

Specification Pattern:
  Encapsulates business rules as composable objects.
  Supports AND, OR, NOT composition.
  Reusable across queries and validations.

Result/Either Pattern:
  Represents success or failure without exceptions.
  Forces handling of both cases.
  Composable and chainable operations.

Middleware/Pipeline Pattern:
  Chain of processing steps.
  Each step can modify input, output, or skip further processing.
  Cross-cutting concerns without decoration nesting.

Outbox Pattern:
  Ensures reliable event publishing alongside database writes.
  Store events in outbox table within same transaction.
  Background process publishes events from outbox.
  Prevents lost events and ensures at-least-once delivery.`,
					CodeExamples: `// Modern behavioral patterns

// Observer Pattern with type safety
type EventBus struct {
    handlers map[string][]interface{}
    mu       sync.RWMutex
}

func NewEventBus() *EventBus {
    return &EventBus{handlers: make(map[string][]interface{})}
}

func Subscribe[T any](bus *EventBus, handler func(T)) func() {
    eventType := fmt.Sprintf("%T", *new(T))
    bus.mu.Lock()
    bus.handlers[eventType] = append(bus.handlers[eventType], handler)
    idx := len(bus.handlers[eventType]) - 1
    bus.mu.Unlock()
    
    return func() {
        bus.mu.Lock()
        defer bus.mu.Unlock()
        bus.handlers[eventType] = append(
            bus.handlers[eventType][:idx],
            bus.handlers[eventType][idx+1:]...,
        )
    }
}

func Publish[T any](bus *EventBus, event T) {
    eventType := fmt.Sprintf("%T", event)
    bus.mu.RLock()
    handlers := bus.handlers[eventType]
    bus.mu.RUnlock()
    
    for _, h := range handlers {
        if handler, ok := h.(func(T)); ok {
            handler(event)
        }
    }
}

// Command Pattern with CQRS
type Command interface {
    CommandName() string
}

type CommandHandler[T Command] interface {
    Handle(ctx context.Context, cmd T) error
}

type CommandBus struct {
    handlers   map[string]interface{}
    middleware []CommandMiddleware
}

type CommandMiddleware func(ctx context.Context, cmd Command, next func(context.Context, Command) error) error

func (b *CommandBus) Register(cmdName string, handler interface{}) {
    b.handlers[cmdName] = handler
}

func (b *CommandBus) Dispatch(ctx context.Context, cmd Command) error {
    handler, ok := b.handlers[cmd.CommandName()]
    if !ok {
        return fmt.Errorf("no handler for command: %s", cmd.CommandName())
    }
    
    // Build middleware chain
    final := func(ctx context.Context, cmd Command) error {
        return reflect.ValueOf(handler).
            MethodByName("Handle").
            Call([]reflect.Value{
                reflect.ValueOf(ctx),
                reflect.ValueOf(cmd),
            })[0].Interface().(error)
    }
    
    chain := final
    for i := len(b.middleware) - 1; i >= 0; i-- {
        mw := b.middleware[i]
        next := chain
        chain = func(ctx context.Context, cmd Command) error {
            return mw(ctx, cmd, next)
        }
    }
    
    return chain(ctx, cmd)
}

// Logging middleware
func LoggingMiddleware(logger Logger) CommandMiddleware {
    return func(ctx context.Context, cmd Command, next func(context.Context, Command) error) error {
        start := time.Now()
        logger.Info("command started", "command", cmd.CommandName())
        err := next(ctx, cmd)
        logger.Info("command completed", "command", cmd.CommandName(), "duration", time.Since(start), "error", err)
        return err
    }
}

// Specification Pattern
type Specification[T any] interface {
    IsSatisfiedBy(entity T) bool
}

type AndSpec[T any] struct {
    specs []Specification[T]
}

func (s AndSpec[T]) IsSatisfiedBy(entity T) bool {
    for _, spec := range s.specs {
        if !spec.IsSatisfiedBy(entity) {
            return false
        }
    }
    return true
}

type OrSpec[T any] struct {
    specs []Specification[T]
}

func (s OrSpec[T]) IsSatisfiedBy(entity T) bool {
    for _, spec := range s.specs {
        if spec.IsSatisfiedBy(entity) {
            return true
        }
    }
    return false
}

type NotSpec[T any] struct {
    spec Specification[T]
}

func (s NotSpec[T]) IsSatisfiedBy(entity T) bool {
    return !s.spec.IsSatisfiedBy(entity)
}

// Example specifications for orders
type OrderAboveAmount struct {
    amount int64
}

func (s OrderAboveAmount) IsSatisfiedBy(order Order) bool {
    return order.Total.Amount > s.amount
}

type OrderInStatus struct {
    status OrderStatus
}

func (s OrderInStatus) IsSatisfiedBy(order Order) bool {
    return order.Status == s.status
}

// Usage:
// highValueActive := AndSpec[Order]{specs: []Specification[Order]{
//     OrderAboveAmount{amount: 10000},
//     OrderInStatus{status: "active"},
// }}

// Outbox Pattern
type OutboxMessage struct {
    ID           string
    AggregateID  string
    EventType    string
    Payload      json.RawMessage
    CreatedAt    time.Time
    PublishedAt  *time.Time
}

type OutboxPublisher struct {
    db        *sql.DB
    publisher EventPublisher
    interval  time.Duration
}

func (p *OutboxPublisher) SaveWithEvents(ctx context.Context, fn func(tx *sql.Tx) ([]OutboxMessage, error)) error {
    tx, err := p.db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }
    defer tx.Rollback()

    messages, err := fn(tx)
    if err != nil {
        return err
    }

    for _, msg := range messages {
        _, err := tx.ExecContext(ctx,
            "INSERT INTO outbox (id, aggregate_id, event_type, payload, created_at) VALUES ($1, $2, $3, $4, $5)",
            msg.ID, msg.AggregateID, msg.EventType, msg.Payload, msg.CreatedAt)
        if err != nil {
            return err
        }
    }

    return tx.Commit()
}

func (p *OutboxPublisher) ProcessOutbox(ctx context.Context) error {
    rows, err := p.db.QueryContext(ctx,
        "SELECT id, event_type, payload FROM outbox WHERE published_at IS NULL ORDER BY created_at LIMIT 100")
    if err != nil {
        return err
    }
    defer rows.Close()

    for rows.Next() {
        var msg OutboxMessage
        if err := rows.Scan(&msg.ID, &msg.EventType, &msg.Payload); err != nil {
            return err
        }
        if err := p.publisher.Publish(ctx, msg.EventType, msg.Payload); err != nil {
            return err
        }
        p.db.ExecContext(ctx, "UPDATE outbox SET published_at = NOW() WHERE id = $1", msg.ID)
    }
    return nil
}`,
				},
			},
		},
	})
}
