package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          2312,
			Title:       "Domain-Driven Design in Depth",
			Description: "Master strategic and tactical DDD patterns including bounded contexts, aggregates, domain events, value objects, and context mapping for complex software systems.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Strategic DDD and Bounded Contexts",
					Content: `Domain-Driven Design provides a framework for tackling complex software by focusing on the core domain and domain logic.

**Strategic Design Patterns:**

Bounded Context:
  A bounded context is a boundary within which a particular domain model is defined and applicable.
  Each bounded context has its own ubiquitous language.
  The same real-world concept may have different representations in different contexts.

  Example - "Customer" in different contexts:
    Sales Context:
      Customer has leads, opportunities, pipeline stage
      Concerned with conversion and revenue
    
    Support Context:
      Customer has tickets, satisfaction score, SLA tier
      Concerned with issue resolution
    
    Billing Context:
      Customer has invoices, payment methods, subscription plan
      Concerned with payment processing
    
    Shipping Context:
      Customer has addresses, delivery preferences
      Concerned with logistics

Context Mapping Patterns:

  Shared Kernel:
    Two bounded contexts share a common subset of the domain model
    Both teams must agree on changes to the shared part
    Use sparingly - creates coupling between teams
    Example: Shared user identity model between Auth and Profile contexts

  Customer-Supplier:
    Upstream context (supplier) provides what downstream (customer) needs
    Downstream team can negotiate with upstream team
    Example: Order context supplies data to Shipping context

  Conformist:
    Downstream context conforms to upstream model
    No negotiation power - must adapt to upstream
    Example: Integrating with a third-party payment API

  Anti-Corruption Layer (ACL):
    Translation layer between bounded contexts
    Protects downstream model from upstream model changes
    Essential when integrating with legacy systems
    
    Implementation:
      Adapter pattern translates external models to internal
      Facade simplifies complex external interfaces
      Translator maps between different ubiquitous languages

  Open Host Service:
    Upstream context publishes a well-defined protocol/API
    Multiple downstream contexts can integrate
    Example: RESTful API or message broker interface

  Published Language:
    Well-documented shared language for integration
    Often combined with Open Host Service
    Example: Industry-standard schemas (FHIR for healthcare)

  Separate Ways:
    No integration between contexts
    Each context implements its own solution
    Appropriate when integration cost exceeds benefit

  Partnership:
    Two teams coordinate planning and development
    Mutual dependency requires collaboration
    Example: Cart and Inventory contexts in e-commerce

Context Map Visualization:

  [Sales Context]---Customer-Supplier--->[Shipping Context]
        |                                       ^
        |                                       |
  Shared Kernel                          Anti-Corruption Layer
        |                                       |
  [Marketing Context]                    [Legacy Warehouse]
        |
  Published Language
        |
  [Analytics Context]

Identifying Bounded Contexts:
  1. Look for different meanings of the same term
  2. Identify organizational boundaries (Conway's Law)
  3. Find areas with different rates of change
  4. Recognize different business capabilities
  5. Detect linguistic boundaries in domain experts' language
  6. Map existing team structures and ownership

Bounded Context Integration:
  Synchronous:
    REST APIs between contexts
    gRPC for internal service communication
    GraphQL for flexible queries across contexts
  
  Asynchronous:
    Domain events via message broker
    Event-carried state transfer
    CQRS across context boundaries`,
					CodeExamples: `// Domain-Driven Design implementation patterns

// Bounded Context with Anti-Corruption Layer
// External payment provider model (their language)
type ExternalPaymentResponse struct {
    TxnID       string  "json:\"txn_id\""
    AmtCents    int     "json:\"amt_cents\""
    CcyCode     string  "json:\"ccy_code\""
    StatusCode  int     "json:\"status_code\""
    MerchantRef string  "json:\"merchant_ref\""
    ProcessedAt string  "json:\"processed_at\""
}

// Our domain model (our language)
type Payment struct {
    ID            PaymentID
    Amount        Money
    Status        PaymentStatus
    OrderID       OrderID
    ProcessedAt   time.Time
}

type Money struct {
    Amount   int64
    Currency Currency
}

type PaymentStatus int
const (
    PaymentPending PaymentStatus = iota
    PaymentCompleted
    PaymentFailed
    PaymentRefunded
)

// Anti-Corruption Layer translates between models
type PaymentACL struct {
    client ExternalPaymentClient
}

func (acl *PaymentACL) ProcessPayment(order Order, amount Money) (Payment, error) {
    // Translate our model to their model
    externalReq := ExternalPaymentRequest{
        AmtCents:    int(amount.Amount),
        CcyCode:     string(amount.Currency),
        MerchantRef: string(order.ID),
    }

    // Call external service
    resp, err := acl.client.Charge(externalReq)
    if err != nil {
        return Payment{}, fmt.Errorf("payment failed: %w", err)
    }

    // Translate their response back to our model
    return acl.translateResponse(resp, order.ID)
}

func (acl *PaymentACL) translateResponse(resp ExternalPaymentResponse, orderID OrderID) (Payment, error) {
    status := acl.translateStatus(resp.StatusCode)
    processedAt, err := time.Parse(time.RFC3339, resp.ProcessedAt)
    if err != nil {
        return Payment{}, fmt.Errorf("invalid timestamp: %w", err)
    }

    return Payment{
        ID:          PaymentID(resp.TxnID),
        Amount:      Money{Amount: int64(resp.AmtCents), Currency: Currency(resp.CcyCode)},
        Status:      status,
        OrderID:     orderID,
        ProcessedAt: processedAt,
    }, nil
}

func (acl *PaymentACL) translateStatus(code int) PaymentStatus {
    switch code {
    case 0:
        return PaymentPending
    case 1:
        return PaymentCompleted
    case -1:
        return PaymentFailed
    default:
        return PaymentPending
    }
}

// Context Map implementation with events
// Sales Context publishes events
type OrderPlaced struct {
    OrderID    string
    CustomerID string
    Items      []OrderItem
    Total      Money
    PlacedAt   time.Time
}

// Shipping Context consumes and translates
type ShipmentEventHandler struct {
    shipmentService ShipmentService
}

func (h *ShipmentEventHandler) HandleOrderPlaced(event OrderPlaced) error {
    // Translate Sales context concept to Shipping context
    shipment := Shipment{
        ID:          NewShipmentID(),
        ReferenceID: event.OrderID,
        RecipientID: event.CustomerID,
        Items:       h.translateItems(event.Items),
        Status:      ShipmentPending,
        CreatedAt:   event.PlacedAt,
    }
    return h.shipmentService.CreateShipment(shipment)
}

func (h *ShipmentEventHandler) translateItems(orderItems []OrderItem) []ShipmentItem {
    items := make([]ShipmentItem, len(orderItems))
    for i, oi := range orderItems {
        items[i] = ShipmentItem{
            ProductID: oi.ProductID,
            Quantity:  oi.Quantity,
            Weight:    oi.Weight,
        }
    }
    return items
}`,
				},
				{
					Title: "Tactical DDD: Aggregates, Entities, and Value Objects",
					Content: `Tactical DDD patterns provide building blocks for implementing domain models within a bounded context.

**Entities:**
  Have a unique identity that persists through state changes.
  Two entities are equal if they have the same identity.
  Identity can be natural (SSN, email) or surrogate (UUID).

  Example: A User entity maintains its identity even when name or email changes.
  Example: An Order entity keeps its ID across status transitions.

**Value Objects:**
  Defined only by their attributes, not by identity.
  Two value objects are equal if all their attributes are equal.
  Should be immutable - create new instances instead of modifying.
  
  Examples:
    Money(amount=100, currency=USD) - defined by amount and currency
    Address(street, city, state, zip) - defined by all fields
    DateRange(start, end) - defined by start and end dates
    EmailAddress(value) - defined by the email string
    Coordinates(lat, lng) - defined by latitude and longitude

**Aggregates:**
  A cluster of domain objects treated as a single unit for data changes.
  Has a root entity (Aggregate Root) that controls access to members.
  External objects can only reference the aggregate root.
  Consistency is enforced within aggregate boundaries.
  
  Rules:
    1. Reference aggregate roots only (not internal entities)
    2. Changes to an aggregate are atomic (transactional)
    3. Aggregates communicate via domain events
    4. Keep aggregates small - one or two entities
    5. Design aggregates around invariants (business rules)

  Example - Order Aggregate:
    Order (Aggregate Root)
    ├── OrderID (Value Object)
    ├── Customer reference (ID only, not the object)
    ├── OrderItems (Entity collection within aggregate)
    │   ├── OrderItem
    │   │   ├── ProductID (reference to Product aggregate)
    │   │   ├── Quantity (Value Object)
    │   │   └── Price (Value Object)
    ├── ShippingAddress (Value Object)
    ├── OrderStatus (Value Object / Enum)
    └── OrderTotal (Value Object, calculated)

  Invariants enforced by Order aggregate:
    - Order must have at least one item
    - Total must match sum of item prices
    - Cannot modify items after order is shipped
    - Quantity must be positive
    - Cannot exceed maximum items per order

**Domain Events:**
  Represent something that happened in the domain.
  Named in past tense: OrderPlaced, PaymentReceived, ItemShipped.
  Immutable records of domain facts.
  Used for communication between aggregates and bounded contexts.

  Event structure:
    - Event ID (unique identifier)
    - Event type (what happened)
    - Aggregate ID (which aggregate)
    - Timestamp (when it happened)
    - Payload (relevant data)
    - Metadata (correlation ID, causation ID)

**Domain Services:**
  Operations that don't naturally belong to any entity or value object.
  Stateless - they operate on domain objects passed to them.
  Named using ubiquitous language verbs.

  Examples:
    TransferMoney(from Account, to Account, amount Money)
    CalculateShipping(order Order, destination Address) ShippingCost
    AuthenticateUser(credentials Credentials) AuthResult

**Repositories:**
  Provide collection-like interface for accessing aggregates.
  Abstract persistence details from the domain.
  One repository per aggregate root.
  
  Interface should use domain language:
    FindByID(id OrderID) (Order, error)
    Save(order Order) error
    FindActiveByCustomer(customerID CustomerID) ([]Order, error)

**Factories:**
  Encapsulate complex object creation logic.
  Ensure aggregates are created in a valid state.
  Can be standalone factory classes or factory methods on aggregates.`,
					CodeExamples: `// Tactical DDD implementation

// Value Objects
type Money struct {
    amount   int64  // stored in smallest unit (cents)
    currency string
}

func NewMoney(amount int64, currency string) (Money, error) {
    if currency == "" {
        return Money{}, errors.New("currency is required")
    }
    return Money{amount: amount, currency: currency}, nil
}

func (m Money) Add(other Money) (Money, error) {
    if m.currency != other.currency {
        return Money{}, fmt.Errorf("cannot add %s to %s", other.currency, m.currency)
    }
    return Money{amount: m.amount + other.amount, currency: m.currency}, nil
}

func (m Money) Multiply(factor int) Money {
    return Money{amount: m.amount * int64(factor), currency: m.currency}
}

func (m Money) IsPositive() bool { return m.amount > 0 }
func (m Money) Equals(other Money) bool {
    return m.amount == other.amount && m.currency == other.currency
}

type EmailAddress struct {
    value string
}

func NewEmailAddress(email string) (EmailAddress, error) {
    if !strings.Contains(email, "@") {
        return EmailAddress{}, errors.New("invalid email address")
    }
    return EmailAddress{value: strings.ToLower(strings.TrimSpace(email))}, nil
}

func (e EmailAddress) String() string { return e.value }
func (e EmailAddress) Domain() string { return strings.Split(e.value, "@")[1] }

// Aggregate Root
type Order struct {
    id              OrderID
    customerID      CustomerID
    items           []OrderItem
    shippingAddress Address
    status          OrderStatus
    placedAt        time.Time
    events          []DomainEvent
}

type OrderItem struct {
    productID ProductID
    name      string
    price     Money
    quantity  int
}

func NewOrder(customerID CustomerID, shippingAddress Address) *Order {
    return &Order{
        id:              NewOrderID(),
        customerID:      customerID,
        shippingAddress: shippingAddress,
        status:          OrderDraft,
        items:           make([]OrderItem, 0),
    }
}

func (o *Order) AddItem(productID ProductID, name string, price Money, quantity int) error {
    if o.status != OrderDraft {
        return errors.New("cannot modify non-draft order")
    }
    if quantity <= 0 {
        return errors.New("quantity must be positive")
    }
    if len(o.items) >= 100 {
        return errors.New("maximum items per order exceeded")
    }

    // Check if item already exists
    for i, item := range o.items {
        if item.productID == productID {
            o.items[i].quantity += quantity
            return nil
        }
    }

    o.items = append(o.items, OrderItem{
        productID: productID,
        name:      name,
        price:     price,
        quantity:  quantity,
    })
    return nil
}

func (o *Order) RemoveItem(productID ProductID) error {
    if o.status != OrderDraft {
        return errors.New("cannot modify non-draft order")
    }
    for i, item := range o.items {
        if item.productID == productID {
            o.items = append(o.items[:i], o.items[i+1:]...)
            return nil
        }
    }
    return errors.New("item not found")
}

func (o *Order) Place() error {
    if o.status != OrderDraft {
        return errors.New("order is not in draft status")
    }
    if len(o.items) == 0 {
        return errors.New("order must have at least one item")
    }
    o.status = OrderPlaced
    o.placedAt = time.Now()
    o.addEvent(OrderPlacedEvent{
        OrderID:    o.id,
        CustomerID: o.customerID,
        Total:      o.Total(),
        PlacedAt:   o.placedAt,
    })
    return nil
}

func (o *Order) Total() Money {
    total := Money{amount: 0, currency: "USD"}
    for _, item := range o.items {
        itemTotal := item.price.Multiply(item.quantity)
        total, _ = total.Add(itemTotal)
    }
    return total
}

func (o *Order) addEvent(event DomainEvent) {
    o.events = append(o.events, event)
}

func (o *Order) PullEvents() []DomainEvent {
    events := o.events
    o.events = nil
    return events
}

// Domain Event
type DomainEvent interface {
    EventType() string
    OccurredAt() time.Time
}

type OrderPlacedEvent struct {
    OrderID    OrderID
    CustomerID CustomerID
    Total      Money
    PlacedAt   time.Time
}

func (e OrderPlacedEvent) EventType() string    { return "order.placed" }
func (e OrderPlacedEvent) OccurredAt() time.Time { return e.PlacedAt }

// Repository interface (domain layer)
type OrderRepository interface {
    FindByID(ctx context.Context, id OrderID) (*Order, error)
    Save(ctx context.Context, order *Order) error
    FindActiveByCustomer(ctx context.Context, customerID CustomerID) ([]*Order, error)
}

// Application Service (orchestrates domain operations)
type PlaceOrderCommand struct {
    CustomerID      string
    ShippingAddress AddressDTO
    Items           []OrderItemDTO
}

type OrderService struct {
    orders    OrderRepository
    products  ProductRepository
    events    EventPublisher
}

func (s *OrderService) PlaceOrder(ctx context.Context, cmd PlaceOrderCommand) (string, error) {
    addr := Address{
        Street: cmd.ShippingAddress.Street,
        City:   cmd.ShippingAddress.City,
        State:  cmd.ShippingAddress.State,
        Zip:    cmd.ShippingAddress.Zip,
    }

    order := NewOrder(CustomerID(cmd.CustomerID), addr)

    for _, item := range cmd.Items {
        product, err := s.products.FindByID(ctx, ProductID(item.ProductID))
        if err != nil {
            return "", fmt.Errorf("product %s not found: %w", item.ProductID, err)
        }
        if err := order.AddItem(product.ID, product.Name, product.Price, item.Quantity); err != nil {
            return "", err
        }
    }

    if err := order.Place(); err != nil {
        return "", err
    }

    if err := s.orders.Save(ctx, order); err != nil {
        return "", fmt.Errorf("failed to save order: %w", err)
    }

    // Publish domain events
    for _, event := range order.PullEvents() {
        if err := s.events.Publish(ctx, event); err != nil {
            return "", fmt.Errorf("failed to publish event: %w", err)
        }
    }

    return string(order.id), nil
}`,
				},
			},
		},
		{
			ID:          2313,
			Title:       "Clean Architecture and Hexagonal Architecture",
			Description: "Design applications with dependency inversion using Clean Architecture, Hexagonal Architecture (Ports and Adapters), and Onion Architecture patterns.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "Clean Architecture Principles",
					Content: `Clean Architecture separates concerns into concentric layers where dependencies point inward toward the domain.

**Layer Structure:**

  Entities (innermost):
    Enterprise business rules
    Domain objects, value objects
    No dependencies on outer layers
    Pure business logic

  Use Cases (Application):
    Application-specific business rules
    Orchestrates data flow between entities
    Defines input/output ports
    Contains application services

  Interface Adapters:
    Converts data between use case format and external format
    Controllers, presenters, gateways
    Repository implementations
    API serialization/deserialization

  Frameworks & Drivers (outermost):
    Database, web framework, UI, external services
    Infrastructure concerns
    Most volatile layer - easy to replace

**The Dependency Rule:**
  Source code dependencies must point inward only.
  Inner layers know nothing about outer layers.
  Data crosses boundaries via simple DTOs or interfaces.

  Allowed: Controller -> Use Case -> Entity
  NOT allowed: Entity -> Database, Use Case -> Web Framework

**Dependency Inversion:**
  High-level modules should not depend on low-level modules.
  Both should depend on abstractions (interfaces).
  
  Example:
    Use Case defines: OrderRepository interface (port)
    Infrastructure implements: PostgresOrderRepository (adapter)
    Use Case depends on interface, not implementation.

Project Structure:

  project/
  ├── domain/           # Entities and value objects
  │   ├── order.go
  │   ├── customer.go
  │   └── money.go
  ├── application/      # Use cases and ports
  │   ├── ports/
  │   │   ├── repositories.go  # Repository interfaces
  │   │   ├── services.go      # External service interfaces
  │   │   └── events.go        # Event publisher interface
  │   ├── commands/
  │   │   ├── place_order.go
  │   │   └── cancel_order.go
  │   └── queries/
  │       ├── get_order.go
  │       └── list_orders.go
  ├── infrastructure/   # Adapters and implementations
  │   ├── persistence/
  │   │   ├── postgres_order_repo.go
  │   │   └── redis_cache.go
  │   ├── messaging/
  │   │   └── kafka_publisher.go
  │   └── external/
  │       └── stripe_payment.go
  └── interfaces/       # Controllers and presenters
      ├── http/
      │   ├── order_handler.go
      │   └── middleware.go
      ├── grpc/
      │   └── order_service.go
      └── cli/
          └── commands.go

Benefits:
  Independent of frameworks - frameworks are tools, not constraints
  Testable - business rules testable without UI, database, or external services
  Independent of UI - easily change from web to CLI
  Independent of database - swap PostgreSQL for MongoDB
  Independent of external agencies - business rules isolated from outside world

Testing Strategy:
  Domain layer: Unit tests with no mocks needed
  Application layer: Unit tests with mocked ports
  Infrastructure layer: Integration tests with real dependencies
  Interface layer: Contract tests and end-to-end tests`,
					CodeExamples: `// Clean Architecture implementation

// === Domain Layer (innermost, no dependencies) ===

// domain/order.go
type OrderStatus string
const (
    OrderStatusDraft     OrderStatus = "draft"
    OrderStatusPlaced    OrderStatus = "placed"
    OrderStatusConfirmed OrderStatus = "confirmed"
    OrderStatusShipped   OrderStatus = "shipped"
    OrderStatusDelivered OrderStatus = "delivered"
    OrderStatusCancelled OrderStatus = "cancelled"
)

type Order struct {
    ID        string
    Customer  CustomerRef
    Items     []LineItem
    Status    OrderStatus
    Total     Money
    CreatedAt time.Time
    UpdatedAt time.Time
}

func (o *Order) Cancel() error {
    if o.Status == OrderStatusShipped || o.Status == OrderStatusDelivered {
        return ErrCannotCancelShippedOrder
    }
    o.Status = OrderStatusCancelled
    o.UpdatedAt = time.Now()
    return nil
}

func (o *Order) CalculateTotal() {
    var total int64
    for _, item := range o.Items {
        total += item.Price.Amount * int64(item.Quantity)
    }
    o.Total = Money{Amount: total, Currency: o.Items[0].Price.Currency}
}

// === Application Layer (depends only on domain) ===

// application/ports/repositories.go
type OrderRepository interface {
    Save(ctx context.Context, order *Order) error
    FindByID(ctx context.Context, id string) (*Order, error)
    FindByCustomer(ctx context.Context, customerID string) ([]*Order, error)
}

type ProductCatalog interface {
    GetProduct(ctx context.Context, id string) (*Product, error)
    CheckAvailability(ctx context.Context, id string, qty int) (bool, error)
}

// application/ports/events.go
type EventPublisher interface {
    Publish(ctx context.Context, event interface{}) error
}

// application/commands/place_order.go
type PlaceOrderInput struct {
    CustomerID string
    Items      []PlaceOrderItem
}

type PlaceOrderItem struct {
    ProductID string
    Quantity  int
}

type PlaceOrderOutput struct {
    OrderID string
    Total   Money
}

type PlaceOrderUseCase struct {
    orders   OrderRepository
    catalog  ProductCatalog
    events   EventPublisher
}

func NewPlaceOrderUseCase(
    orders OrderRepository,
    catalog ProductCatalog,
    events EventPublisher,
) *PlaceOrderUseCase {
    return &PlaceOrderUseCase{
        orders:  orders,
        catalog: catalog,
        events:  events,
    }
}

func (uc *PlaceOrderUseCase) Execute(ctx context.Context, input PlaceOrderInput) (*PlaceOrderOutput, error) {
    if len(input.Items) == 0 {
        return nil, ErrEmptyOrder
    }

    order := &Order{
        ID:        generateID(),
        Customer:  CustomerRef{ID: input.CustomerID},
        Status:    OrderStatusPlaced,
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }

    for _, item := range input.Items {
        product, err := uc.catalog.GetProduct(ctx, item.ProductID)
        if err != nil {
            return nil, fmt.Errorf("product %s: %w", item.ProductID, err)
        }
        available, err := uc.catalog.CheckAvailability(ctx, item.ProductID, item.Quantity)
        if err != nil {
            return nil, fmt.Errorf("availability check: %w", err)
        }
        if !available {
            return nil, fmt.Errorf("product %s: insufficient stock", item.ProductID)
        }
        order.Items = append(order.Items, LineItem{
            ProductID: product.ID,
            Name:      product.Name,
            Price:     product.Price,
            Quantity:  item.Quantity,
        })
    }

    order.CalculateTotal()
    if err := uc.orders.Save(ctx, order); err != nil {
        return nil, fmt.Errorf("save order: %w", err)
    }
    uc.events.Publish(ctx, OrderPlacedEvent{OrderID: order.ID, Total: order.Total})

    return &PlaceOrderOutput{OrderID: order.ID, Total: order.Total}, nil
}

// === Infrastructure Layer (implements ports) ===

// infrastructure/persistence/postgres_order_repo.go
type PostgresOrderRepo struct {
    db *sql.DB
}

func (r *PostgresOrderRepo) Save(ctx context.Context, order *Order) error {
    query := "INSERT INTO orders (id, customer_id, status, total_amount, total_currency, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (id) DO UPDATE SET status=$3, total_amount=$4, updated_at=$7"
    _, err := r.db.ExecContext(ctx, query,
        order.ID, order.Customer.ID, order.Status,
        order.Total.Amount, order.Total.Currency,
        order.CreatedAt, order.UpdatedAt,
    )
    return err
}

func (r *PostgresOrderRepo) FindByID(ctx context.Context, id string) (*Order, error) {
    row := r.db.QueryRowContext(ctx,
        "SELECT id, customer_id, status, total_amount, total_currency, created_at, updated_at FROM orders WHERE id = $1", id)
    order := &Order{}
    var totalAmount int64
    var totalCurrency string
    err := row.Scan(&order.ID, &order.Customer.ID, &order.Status, &totalAmount, &totalCurrency, &order.CreatedAt, &order.UpdatedAt)
    if err != nil {
        return nil, err
    }
    order.Total = Money{Amount: totalAmount, Currency: totalCurrency}
    return order, nil
}

// === Interface Layer (adapts external input to use cases) ===

// interfaces/http/order_handler.go
type OrderHandler struct {
    placeOrder *PlaceOrderUseCase
}

func (h *OrderHandler) HandlePlaceOrder(w http.ResponseWriter, r *http.Request) {
    var req PlaceOrderRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "invalid request", http.StatusBadRequest)
        return
    }
    input := PlaceOrderInput{
        CustomerID: req.CustomerID,
        Items:      mapRequestItems(req.Items),
    }
    output, err := h.placeOrder.Execute(r.Context(), input)
    if err != nil {
        handleError(w, err)
        return
    }
    json.NewEncoder(w).Encode(PlaceOrderResponse{
        OrderID: output.OrderID,
        Total:   output.Total.Amount,
    })
}`,
				},
			},
		},
	})
}
