package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          2331,
			Title:       "Domain-Specific Architecture Patterns",
			Description: "Apply specialized architecture patterns for e-commerce, real-time systems, multi-tenant SaaS, and IoT platforms.",
			Order:       31,
			Lessons: []problems.Lesson{
				{
					Title: "E-Commerce Architecture",
					Content: `E-commerce systems have unique architectural challenges around inventory management, payment processing, and high-traffic events.

**Core Services:**

Product Catalog Service:
  Product information, categories, attributes
  Search and filtering
  Caching-heavy (products change infrequently)
  Supports faceted search
  Integration with search engine (Elasticsearch)

Inventory Service:
  Real-time stock tracking
  Reservation system for cart items
  Warehouse management integration
  Eventual consistency acceptable for display
  Strong consistency required for purchase

Order Service:
  Order lifecycle management (draft -> placed -> paid -> shipped -> delivered)
  Saga orchestration for distributed transactions
  Order history and tracking
  Integration with payment and shipping

Payment Service:
  Payment gateway integration (Stripe, PayPal)
  Idempotent payment processing
  Refund handling
  PCI DSS compliance
  Never store raw card data

Cart Service:
  Temporary storage (Redis/DynamoDB)
  Session-based for anonymous users
  Persistent for logged-in users
  Inventory reservation on add-to-cart
  Cart expiration and cleanup

Pricing Service:
  Dynamic pricing rules
  Discount and promotion engine
  Tax calculation
  Currency conversion
  Price history for analytics

**Flash Sale / High-Traffic Architecture:**

Pre-sale Preparation:
  Scale infrastructure ahead of time
  Pre-warm caches with sale products
  Queue-based order processing
  Rate limiting per user
  Static asset CDN distribution

Traffic Management:
  Virtual waiting room / queue
  Progressive disclosure (limited users per batch)
  Circuit breakers for downstream services
  Graceful degradation (disable non-critical features)

Inventory Management:
  Pre-allocate inventory to queue positions
  Optimistic locking for stock updates
  Counter-based inventory (Redis DECR)
  Compensating transactions for failures

**Multi-Tenant SaaS Architecture:**

Tenancy Models:

  Shared Everything:
    Single database, discriminator column (tenant_id)
    Cheapest, simplest to operate
    Risk: Noisy neighbor, data isolation
    Best for: Small tenants, similar workloads
    
    SELECT * FROM orders WHERE tenant_id = ?

  Shared Compute, Separate Database:
    Each tenant gets own database/schema
    Better isolation, moderate cost
    Database connection management overhead
    Best for: Medium tenants, compliance requirements

  Dedicated Everything:
    Separate infrastructure per tenant
    Maximum isolation, highest cost
    Independent scaling and deployment
    Best for: Enterprise tenants, strict compliance

Cross-Cutting Concerns:
  Tenant Resolution:
    Subdomain: tenant1.app.com
    Path: app.com/tenant1
    Header: X-Tenant-ID
    Token claim: JWT with tenant_id

  Data Isolation:
    Row-level security (PostgreSQL RLS)
    Separate schemas or databases
    Encryption per tenant
    Backup and restore per tenant

  Rate Limiting:
    Per-tenant rate limits
    Resource quotas (storage, API calls, users)
    Billing based on usage

  Customization:
    Feature flags per tenant
    Configuration overrides
    Custom branding and themes
    Tenant-specific integrations

**IoT Platform Architecture:**

  Device -> Gateway -> Ingestion -> Processing -> Storage -> Analytics/Actions

  Device Layer:
    Sensors and actuators
    Local processing (edge computing)
    Intermittent connectivity
    Resource constrained (memory, CPU, power)

  Communication Layer:
    MQTT: Lightweight pub/sub for constrained devices
    CoAP: REST-like for constrained networks
    HTTP/WebSocket: For richer devices
    LoRaWAN/NB-IoT: For long-range, low-power

  Ingestion Layer:
    Message broker (Kafka, AWS IoT Core)
    Protocol translation
    Device authentication
    Message validation and enrichment
    Buffering for traffic spikes

  Processing Layer:
    Stream processing (Flink, Kafka Streams)
    Rule engine for real-time decisions
    Anomaly detection
    Aggregation and windowing

  Storage Layer:
    Time-series database (InfluxDB, TimescaleDB)
    Object storage for raw data (S3)
    Document store for device metadata
    Data lake for analytics

  Scale Considerations:
    Millions of devices sending telemetry
    Write-heavy workload (time-series data)
    High fan-in (many devices, few endpoints)
    Device state management at scale`,
					CodeExamples: `// E-commerce architecture patterns

// Inventory reservation with optimistic locking
type InventoryService struct {
    db     *sql.DB
    cache  *redis.Client
    events EventPublisher
}

type InventoryReservation struct {
    ID         string
    ProductID  string
    Quantity   int
    OrderID    string
    ExpiresAt  time.Time
    Status     string // active, confirmed, cancelled, expired
}

func (s *InventoryService) Reserve(ctx context.Context, productID string, quantity int, orderID string) (*InventoryReservation, error) {
    tx, err := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
    if err != nil {
        return nil, err
    }
    defer tx.Rollback()
    
    // Check available stock with row-level lock
    var available int
    err = tx.QueryRowContext(ctx,
        "SELECT available_quantity FROM inventory WHERE product_id = $1 FOR UPDATE",
        productID,
    ).Scan(&available)
    if err != nil {
        return nil, fmt.Errorf("product not found: %w", err)
    }
    
    if available < quantity {
        return nil, ErrInsufficientStock
    }
    
    // Decrease available quantity
    _, err = tx.ExecContext(ctx,
        "UPDATE inventory SET available_quantity = available_quantity - $1, updated_at = NOW() WHERE product_id = $2",
        quantity, productID,
    )
    if err != nil {
        return nil, err
    }
    
    // Create reservation
    reservation := &InventoryReservation{
        ID:        generateUUID(),
        ProductID: productID,
        Quantity:  quantity,
        OrderID:   orderID,
        ExpiresAt: time.Now().Add(15 * time.Minute),
        Status:    "active",
    }
    
    _, err = tx.ExecContext(ctx,
        "INSERT INTO reservations (id, product_id, quantity, order_id, expires_at, status) VALUES ($1, $2, $3, $4, $5, $6)",
        reservation.ID, reservation.ProductID, reservation.Quantity,
        reservation.OrderID, reservation.ExpiresAt, reservation.Status,
    )
    if err != nil {
        return nil, err
    }
    
    if err := tx.Commit(); err != nil {
        return nil, err
    }
    
    // Invalidate cache
    s.cache.Del(ctx, "inventory:"+productID)
    
    // Publish event
    s.events.Publish(ctx, "inventory.reserved", InventoryReservedEvent{
        ReservationID: reservation.ID,
        ProductID:     productID,
        Quantity:      quantity,
        OrderID:       orderID,
    })
    
    return reservation, nil
}

func (s *InventoryService) ConfirmReservation(ctx context.Context, reservationID string) error {
    result, err := s.db.ExecContext(ctx,
        "UPDATE reservations SET status = 'confirmed', updated_at = NOW() WHERE id = $1 AND status = 'active'",
        reservationID,
    )
    if err != nil {
        return err
    }
    rows, _ := result.RowsAffected()
    if rows == 0 {
        return ErrReservationNotFound
    }
    return nil
}

func (s *InventoryService) ReleaseExpiredReservations(ctx context.Context) (int, error) {
    rows, err := s.db.QueryContext(ctx,
        "SELECT id, product_id, quantity FROM reservations WHERE status = 'active' AND expires_at < NOW()",
    )
    if err != nil {
        return 0, err
    }
    defer rows.Close()
    
    released := 0
    for rows.Next() {
        var id, productID string
        var quantity int
        if err := rows.Scan(&id, &productID, &quantity); err != nil {
            continue
        }
        
        tx, err := s.db.BeginTx(ctx, nil)
        if err != nil {
            continue
        }
        
        tx.ExecContext(ctx,
            "UPDATE reservations SET status = 'expired' WHERE id = $1", id)
        tx.ExecContext(ctx,
            "UPDATE inventory SET available_quantity = available_quantity + $1 WHERE product_id = $2",
            quantity, productID)
        
        if err := tx.Commit(); err == nil {
            released++
            s.cache.Del(ctx, "inventory:"+productID)
        }
    }
    
    return released, nil
}

// Multi-tenant middleware
type TenantMiddleware struct {
    resolver TenantResolver
    logger   *Logger
}

type TenantResolver interface {
    Resolve(r *http.Request) (string, error)
}

type SubdomainResolver struct {
    baseDomain string
}

func (r *SubdomainResolver) Resolve(req *http.Request) (string, error) {
    host := req.Host
    if idx := strings.Index(host, ":"); idx > 0 {
        host = host[:idx]
    }
    
    if !strings.HasSuffix(host, r.baseDomain) {
        return "", fmt.Errorf("invalid domain: %s", host)
    }
    
    subdomain := strings.TrimSuffix(host, "."+r.baseDomain)
    if subdomain == "" || subdomain == host {
        return "", fmt.Errorf("no tenant subdomain found")
    }
    
    return subdomain, nil
}

func (m *TenantMiddleware) Handle(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        tenantID, err := m.resolver.Resolve(r)
        if err != nil {
            m.logger.Warn("tenant resolution failed", "error", err)
            http.Error(w, "tenant not found", http.StatusNotFound)
            return
        }
        
        ctx := context.WithValue(r.Context(), tenantIDKey, tenantID)
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

// Tenant-aware database connection
type TenantDBRouter struct {
    connections map[string]*sql.DB
    mu          sync.RWMutex
    factory     func(tenantID string) (*sql.DB, error)
}

func (r *TenantDBRouter) GetDB(ctx context.Context) (*sql.DB, error) {
    tenantID, ok := ctx.Value(tenantIDKey).(string)
    if !ok {
        return nil, fmt.Errorf("no tenant in context")
    }
    
    r.mu.RLock()
    db, exists := r.connections[tenantID]
    r.mu.RUnlock()
    
    if exists {
        return db, nil
    }
    
    r.mu.Lock()
    defer r.mu.Unlock()
    
    // Double check after acquiring write lock
    if db, exists = r.connections[tenantID]; exists {
        return db, nil
    }
    
    db, err := r.factory(tenantID)
    if err != nil {
        return nil, fmt.Errorf("failed to create connection for tenant %s: %w", tenantID, err)
    }
    
    r.connections[tenantID] = db
    return db, nil
}

// IoT telemetry ingestion
type TelemetryIngester struct {
    buffer    chan TelemetryMessage
    processor TelemetryProcessor
    batchSize int
    flushInterval time.Duration
    logger    *Logger
}

type TelemetryMessage struct {
    DeviceID  string                 "json:\"device_id\""
    Timestamp time.Time              "json:\"timestamp\""
    Type      string                 "json:\"type\""
    Values    map[string]float64     "json:\"values\""
    Metadata  map[string]string      "json:\"metadata\""
}

type TelemetryProcessor interface {
    ProcessBatch(ctx context.Context, messages []TelemetryMessage) error
}

func NewTelemetryIngester(processor TelemetryProcessor, batchSize int, bufferSize int) *TelemetryIngester {
    return &TelemetryIngester{
        buffer:        make(chan TelemetryMessage, bufferSize),
        processor:     processor,
        batchSize:     batchSize,
        flushInterval: 5 * time.Second,
    }
}

func (i *TelemetryIngester) Ingest(msg TelemetryMessage) error {
    select {
    case i.buffer <- msg:
        return nil
    default:
        return fmt.Errorf("ingestion buffer full")
    }
}

func (i *TelemetryIngester) Start(ctx context.Context) {
    batch := make([]TelemetryMessage, 0, i.batchSize)
    ticker := time.NewTicker(i.flushInterval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            if len(batch) > 0 {
                i.flush(context.Background(), batch)
            }
            return
            
        case msg := <-i.buffer:
            batch = append(batch, msg)
            if len(batch) >= i.batchSize {
                i.flush(ctx, batch)
                batch = make([]TelemetryMessage, 0, i.batchSize)
            }
            
        case <-ticker.C:
            if len(batch) > 0 {
                i.flush(ctx, batch)
                batch = make([]TelemetryMessage, 0, i.batchSize)
            }
        }
    }
}

func (i *TelemetryIngester) flush(ctx context.Context, batch []TelemetryMessage) {
    if err := i.processor.ProcessBatch(ctx, batch); err != nil {
        i.logger.Error("batch processing failed",
            "batch_size", len(batch),
            "error", err,
        )
    } else {
        i.logger.Info("batch processed",
            "batch_size", len(batch),
        )
    }
}

// Virtual waiting room for flash sales
type WaitingRoom struct {
    mu            sync.Mutex
    queue         []WaitingUser
    maxConcurrent int
    activeCount   int
    admitted      map[string]time.Time
    sessionTTL    time.Duration
}

type WaitingUser struct {
    UserID    string
    JoinedAt  time.Time
    Position  int
}

type WaitingRoomStatus struct {
    Position     int    "json:\"position\""
    EstimatedWait string "json:\"estimated_wait\""
    Status       string "json:\"status\"" // waiting, admitted, expired
    Token        string "json:\"token,omitempty\""
}

func NewWaitingRoom(maxConcurrent int, sessionTTL time.Duration) *WaitingRoom {
    return &WaitingRoom{
        maxConcurrent: maxConcurrent,
        admitted:      make(map[string]time.Time),
        sessionTTL:    sessionTTL,
    }
}

func (wr *WaitingRoom) Join(userID string) *WaitingRoomStatus {
    wr.mu.Lock()
    defer wr.mu.Unlock()
    
    // Already admitted?
    if expiry, ok := wr.admitted[userID]; ok {
        if time.Now().Before(expiry) {
            return &WaitingRoomStatus{Status: "admitted", Token: generateToken(userID)}
        }
        delete(wr.admitted, userID)
        wr.activeCount--
    }
    
    // Can admit immediately?
    wr.cleanExpired()
    if wr.activeCount < wr.maxConcurrent {
        wr.activeCount++
        wr.admitted[userID] = time.Now().Add(wr.sessionTTL)
        return &WaitingRoomStatus{Status: "admitted", Token: generateToken(userID)}
    }
    
    // Add to queue
    position := len(wr.queue) + 1
    wr.queue = append(wr.queue, WaitingUser{
        UserID:   userID,
        JoinedAt: time.Now(),
        Position: position,
    })
    
    estimatedWait := time.Duration(position) * 30 * time.Second
    return &WaitingRoomStatus{
        Position:      position,
        EstimatedWait: estimatedWait.String(),
        Status:        "waiting",
    }
}

func (wr *WaitingRoom) cleanExpired() {
    now := time.Now()
    for userID, expiry := range wr.admitted {
        if now.After(expiry) {
            delete(wr.admitted, userID)
            wr.activeCount--
        }
    }
}

func (wr *WaitingRoom) AdmitNext() {
    wr.mu.Lock()
    defer wr.mu.Unlock()
    
    wr.cleanExpired()
    
    for wr.activeCount < wr.maxConcurrent && len(wr.queue) > 0 {
        user := wr.queue[0]
        wr.queue = wr.queue[1:]
        wr.admitted[user.UserID] = time.Now().Add(wr.sessionTTL)
        wr.activeCount++
    }
}`,
				},
			},
		},
	})
}
