package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          33,
			Title:       "Clean Architecture & Hexagonal Architecture",
			Description: "Learn Clean Architecture principles, Hexagonal Architecture (Ports and Adapters), and Onion Architecture patterns.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "Clean Architecture",
					Content: `Clean Architecture, introduced by Robert C. Martin (Uncle Bob), organizes code into layers with clear dependencies and boundaries.

**Core Principles:**

**1. Dependency Rule:**
- Dependencies point inward
- Outer layers depend on inner layers
- Inner layers don't know about outer layers
- Business logic is independent of frameworks, UI, and databases

**2. Layers (from outer to inner):**

**Frameworks & Drivers:**
- Web frameworks, UI, databases, external services
- Most volatile layer
- Changes frequently

**Interface Adapters:**
- Controllers, presenters, gateways
- Converts data between layers
- Adapts external interfaces to use cases

**Use Cases (Application Business Rules):**
- Application-specific business logic
- Orchestrates domain entities
- Independent of UI and database

**Entities (Enterprise Business Rules):**
- Core business objects
- Most stable layer
- Independent of application

**Benefits:**
- **Testability**: Business logic can be tested without UI or database
- **Independence**: Business logic independent of frameworks
- **Flexibility**: Easy to change frameworks without affecting business logic
- **Maintainability**: Clear separation of concerns

**Key Concepts:**
- **Dependency Inversion**: Dependencies point inward
- **Interface Segregation**: Small, focused interfaces
- **Single Responsibility**: Each layer has one responsibility
- **Independence**: Business logic doesn't depend on external concerns

**Why Clean Architecture Matters:**

**Problem Without Clean Architecture:**
- Business logic mixed with database code
- Hard to test (requires database)
- Framework changes break business logic
- UI changes affect business rules
- Difficult to reuse business logic

**Solution With Clean Architecture:**
- Business logic isolated and testable
- Framework changes don't affect business logic
- UI can be swapped without changing core
- Business logic reusable across platforms
- Clear boundaries and responsibilities

**Dependency Rule in Detail:**

The dependency rule is the cornerstone of Clean Architecture:

**Rule**: Source code dependencies can only point inward. Nothing in an inner circle can know anything about something in an outer circle.

**Visual Representation:**

Outer Layers -> Inner Layers (Dependencies flow inward)
     ↓              ↓
  Frameworks    Entities
  & Drivers     (Core)
     ↓              ↑
  Interface     Use Cases
  Adapters      (Application)


**Benefits of Dependency Rule:**
- **Testability**: Test business logic without external dependencies
- **Flexibility**: Change frameworks without changing business logic
- **Independence**: Business rules don't depend on UI or database
- **Reusability**: Use cases can be reused across different interfaces

**Real-World Example - E-commerce Order Processing:**

**Without Clean Architecture:**

Controller -> Service -> Database (tightly coupled)
- Can't test without database
- Changing database affects business logic
- Hard to add new interfaces (mobile app)


**With Clean Architecture:**

Controller -> UseCase -> Repository Interface
                          ↑
                    RepositoryImpl (Database)
- Test UseCase with mock repository
- Swap database without changing UseCase
- Add mobile controller easily


**Common Mistakes:**

**1. Violating Dependency Rule:**

// Bad: Entity depends on framework
type User struct {
    gorm.Model  // Framework dependency!
}

// Good: Pure domain entity
type User struct {
    ID    int
    Email string
}


**2. Business Logic in Controllers:**

// Bad: Business logic in controller
func (c *Controller) CreateOrder(r *http.Request) {
    if r.FormValue("total") < 100 { // Business rule!
        // ...
    }
}

// Good: Business logic in use case
func (uc *CreateOrderUseCase) Execute(order Order) error {
    if order.Total < 100 { // Business rule in domain
        return ErrMinimumOrderAmount
    }
}


**3. Framework Types in Domain:**

// Bad: Using framework types in domain
type User struct {
    CreatedAt time.Time // OK
    // But avoid: json tags, db tags in domain layer
}

// Good: Keep domain pure
type User struct {
    ID        int
    Email     string
    CreatedAt time.Time
}


**Key Takeaways:**
- Dependencies point inward only
- Business logic is independent of frameworks
- Use interfaces to invert dependencies
- Test business logic without external dependencies
- Clear separation enables flexibility and maintainability

**Self-Assessment Questions:**
1. Why does the dependency rule point inward?
2. How does Clean Architecture improve testability?
3. What happens if you put business logic in a controller?
4. How do you test use cases without a database?
5. Give an example of violating the dependency rule.`,
					CodeExamples: `Clean Architecture Layers:

┌─────────────────────────────────────┐
│  Frameworks & Drivers                │
│  (Web, UI, DB, External Services)   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Interface Adapters                 │
│  (Controllers, Presenters, Gateways) │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Use Cases                          │
│  (Application Business Logic)       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Entities                           │
│  (Enterprise Business Rules)        │
└─────────────────────────────────────┘

Example: User Registration Use Case

// Entity (Domain Layer)
type User struct {
    ID    int
    Email string
    Name  string
}

func (u *User) Validate() error {
    if u.Email == "" {
        return errors.New("email required")
    }
    return nil
}

// Use Case (Application Layer)
type RegisterUserUseCase struct {
    userRepo UserRepository
    emailService EmailService
}

func (uc *RegisterUserUseCase) Execute(email, name string) error {
    user := &User{Email: email, Name: name}
    
    if err := user.Validate(); err != nil {
        return err
    }
    
    if err := uc.userRepo.Save(user); err != nil {
        return err
    }
    
    return uc.emailService.SendWelcomeEmail(user.Email)
}

// Repository Interface (Application Layer)
type UserRepository interface {
    Save(user *User) error
    FindByEmail(email string) (*User, error)
}

// Gateway (Interface Adapter Layer)
type UserRepositoryImpl struct {
    db *sql.DB
}

func (r *UserRepositoryImpl) Save(user *User) error {
    // Database implementation
    _, err := r.db.Exec("INSERT INTO users ...", user.Email, user.Name)
    return err
}

// Controller (Interface Adapter Layer)
type UserController struct {
    registerUseCase *RegisterUserUseCase
}

func (c *UserController) Register(w http.ResponseWriter, r *http.Request) {
    email := r.FormValue("email")
    name := r.FormValue("name")
    
    if err := c.registerUseCase.Execute(email, name); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    w.WriteHeader(http.StatusCreated)
}

Dependency Rule:
- UseCase depends on Repository interface (abstraction)
- RepositoryImpl implements Repository interface
- Controller depends on UseCase
- Business logic (UseCase, User) doesn't depend on frameworks`,
				},
				{
					Title: "Hexagonal Architecture (Ports and Adapters)",
					Content: `Hexagonal Architecture, also called Ports and Adapters, isolates the core application from external concerns.

**Key Concepts:**

**Ports:**
- **Inbound Ports**: Interfaces for incoming requests (use cases)
- **Outbound Ports**: Interfaces for outgoing calls (repositories, external services)

**Adapters:**
- **Inbound Adapters**: Implement inbound ports (controllers, CLI, message handlers)
- **Outbound Adapters**: Implement outbound ports (database, external APIs, file system)

**Benefits:**
- **Testability**: Core logic can be tested with mock adapters
- **Flexibility**: Easy to swap adapters (e.g., change database)
- **Independence**: Core application doesn't depend on external concerns
- **Multiple Interfaces**: Same core can have multiple interfaces (REST, GraphQL, CLI)

**Architecture:**
- Core application in the center
- Ports define interfaces
- Adapters implement ports
- Adapters are pluggable

**When to Use:**
- Applications with multiple interfaces (REST, GraphQL, CLI)
- Need to swap implementations easily (different databases)
- High testability requirements
- Integration with multiple external systems

**When NOT to Use:**
- Simple CRUD applications
- Applications with single interface
- When overhead outweighs benefits
- Small projects where simplicity is more important

**Real-World Example - E-commerce Platform:**

**Multiple Inbound Adapters:**
- REST API for web frontend
- GraphQL API for mobile apps
- CLI for admin operations
- Message queue handler for async processing

**Multiple Outbound Adapters:**
- PostgreSQL for production database
- MySQL for reporting database
- Redis for caching
- External payment gateway
- Email service provider

**Benefits Realized:**
- Same business logic works for all interfaces
- Easy to add new interfaces (gRPC, WebSocket)
- Can swap database without changing business logic
- Test business logic with mock adapters

**Key Differences from Clean Architecture:**

**Clean Architecture:**
- Focuses on layers (Entities, Use Cases, Interface Adapters)
- Dependency rule: dependencies point inward
- More prescriptive about layer structure

**Hexagonal Architecture:**
- Focuses on ports and adapters
- Core application in center, adapters around it
- More flexible about internal structure
- Emphasizes multiple interfaces

**Both achieve similar goals:**
- Isolate business logic
- Enable testing
- Allow framework swapping
- Support multiple interfaces

**Key Takeaways:**
- Ports define interfaces, adapters implement them
- Core application is independent of external concerns
- Multiple adapters can implement same port
- Easy to test with mock adapters
- Flexible and maintainable architecture

**Self-Assessment Questions:**
1. What is the difference between a port and an adapter?
2. How does Hexagonal Architecture support multiple interfaces?
3. Why is the core application independent of adapters?
4. Give an example of an inbound port and an outbound port.
5. How would you test the core application without real adapters?`,
					CodeExamples: `Hexagonal Architecture:

        ┌─────────────────────┐
        │   Inbound Adapters  │
        │  (REST, GraphQL,    │
        │   CLI, Message)    │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   Inbound Ports     │
        │   (Use Cases)       │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   Core Application  │
        │   (Domain Logic)    │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   Outbound Ports    │
        │   (Interfaces)      │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Outbound Adapters  │
        │  (DB, APIs, Files)  │
        └─────────────────────┘

Example Implementation:

// Inbound Port (Use Case)
type UserService interface {
    RegisterUser(email, name string) error
    GetUser(id int) (*User, error)
}

// Core Application
type userServiceImpl struct {
    userRepo UserRepository  // Outbound Port
    emailService EmailService // Outbound Port
}

func (s *userServiceImpl) RegisterUser(email, name string) error {
    user := NewUser(email, name)
    if err := user.Validate(); err != nil {
        return err
    }
    return s.userRepo.Save(user)
}

// Outbound Port (Repository Interface)
type UserRepository interface {
    Save(user *User) error
    FindByID(id int) (*User, error)
}

// Outbound Adapter (Database Implementation)
type postgresUserRepository struct {
    db *sql.DB
}

func (r *postgresUserRepository) Save(user *User) error {
    // PostgreSQL implementation
    _, err := r.db.Exec("INSERT INTO users ...", user.Email, user.Name)
    return err
}

// Inbound Adapter (REST Controller)
type userController struct {
    userService UserService  // Inbound Port
}

func (c *userController) Register(w http.ResponseWriter, r *http.Request) {
    email := r.FormValue("email")
    name := r.FormValue("name")
    
    if err := c.userService.RegisterUser(email, name); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    w.WriteHeader(http.StatusCreated)
}

// Inbound Adapter (CLI)
type userCLI struct {
    userService UserService
}

func (cli *userCLI) Register(email, name string) {
    if err := cli.userService.RegisterUser(email, name); err != nil {
        fmt.Println("Error:", err)
        return
    }
    fmt.Println("User registered successfully")
}

Multiple Adapters:

Same core application can have:
- REST API adapter
- GraphQL API adapter
- CLI adapter
- Message queue adapter

All use the same inbound port (UserService)`,
				},
				{
					Title: "Onion Architecture",
					Content: `Onion Architecture, introduced by Jeffrey Palermo, organizes application into concentric layers with dependencies pointing inward.

**Layers (from outer to inner):**

**1. Infrastructure Layer:**
- External concerns (database, file system, web services)
- Implements interfaces defined in inner layers
- Most volatile layer

**2. Application Services Layer:**
- Application-specific business logic
- Orchestrates domain objects
- Implements use cases

**3. Domain Services Layer:**
- Domain logic that doesn't belong to entities
- Cross-entity business rules
- Domain-specific operations

**4. Domain Model (Core):**
- Entities and value objects
- Business rules and invariants
- Most stable layer
- No dependencies on outer layers

**Key Principles:**
- **Dependency Inversion**: Outer layers depend on inner layers
- **Domain-Centric**: Domain model is the core
- **Framework Independence**: Domain doesn't depend on frameworks
- **Testability**: Core can be tested without infrastructure

**Differences from Clean Architecture:**
- More emphasis on domain model
- Domain services as separate layer
- Application services orchestrate domain

**Benefits:**
- Domain-focused design
- Testable core
- Framework independence
- Clear separation of concerns

**Visual Representation:**

Onion Architecture Layers:

Infrastructure Layer (Database, External APIs)
    |
Application Services Layer (Use Cases, Orchestration)
    |
Domain Services Layer (Cross-entity Logic)
    |
Domain Model (Core) (Entities, Value Objects)

**Real-World Example - Order Processing:**

**Domain Model (Core):**
- Order entity with business rules
- OrderItem value object
- OrderStatus enum
- Business invariants (order total validation)

**Domain Services:**
- OrderCalculator: Calculates totals across items
- OrderValidator: Validates order business rules
- PricingService: Applies discounts and taxes

**Application Services:**
- CreateOrderUseCase: Orchestrates order creation
- ProcessPaymentUseCase: Handles payment flow
- ShipOrderUseCase: Manages shipping process

**Infrastructure:**
- OrderRepository (PostgreSQL implementation)
- PaymentGatewayAdapter (Stripe integration)
- EmailServiceAdapter (SendGrid integration)

**Key Takeaways:**
- Domain model is the core, most stable layer
- Dependencies point inward
- Domain services handle cross-entity logic
- Application services orchestrate use cases
- Infrastructure implements interfaces from inner layers

**Self-Assessment Questions:**
1. What is the difference between Domain Services and Application Services?
2. Why is the Domain Model the most stable layer?
3. How does Onion Architecture differ from Clean Architecture?
4. Give an example of domain logic vs application logic.
5. How do you test the domain model without infrastructure?`,
					CodeExamples: `Onion Architecture:

┌─────────────────────────────────────┐
│  Infrastructure Layer               │
│  (DB, External APIs, File System) │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Application Services               │
│  (Use Cases, Application Logic)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Domain Services                    │
│  (Domain Logic, Cross-Entity Rules) │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Domain Model (Core)                │
│  (Entities, Value Objects)          │
└─────────────────────────────────────┘

Example:

// Domain Model (Core)
type Order struct {
    ID       int
    Items    []OrderItem
    Customer Customer
    Total    Money
}

func (o *Order) CalculateTotal() {
    total := Money{Amount: 0, Currency: "USD"}
    for _, item := range o.Items {
        total = total.Add(item.Price.Multiply(item.Quantity))
    }
    o.Total = total
}

// Domain Service
type PricingService interface {
    CalculateDiscount(order *Order) Money
}

type pricingServiceImpl struct{}

func (s *pricingServiceImpl) CalculateDiscount(order *Order) Money {
    // Domain logic for discount calculation
    if order.Customer.IsVIP() {
        return order.Total.Multiply(0.1) // 10% discount
    }
    return Money{Amount: 0, Currency: "USD"}
}

// Application Service
type OrderService struct {
    orderRepo OrderRepository
    pricingService PricingService
}

func (s *OrderService) CreateOrder(customerID int, items []OrderItem) (*Order, error) {
    order := &Order{
        Customer: s.getCustomer(customerID),
        Items:    items,
    }
    
    order.CalculateTotal()
    discount := s.pricingService.CalculateDiscount(order)
    order.Total = order.Total.Subtract(discount)
    
    return s.orderRepo.Save(order)
}

// Infrastructure (Repository Implementation)
type orderRepositoryImpl struct {
    db *sql.DB
}

func (r *orderRepositoryImpl) Save(order *Order) error {
    // Database implementation
    return nil
}

Dependency Flow:
- Infrastructure -> Application Services -> Domain Services -> Domain Model
- Domain Model has no dependencies`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          34,
			Title:       "Domain-Driven Design (DDD)",
			Description: "Learn Domain-Driven Design concepts including bounded contexts, ubiquitous language, entities, value objects, aggregates, and domain events.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Strategic Design: Bounded Contexts and Ubiquitous Language",
					Content: `Strategic Design focuses on the big picture: how to organize large systems and teams.

**Bounded Context:**
- Explicit boundary within which a domain model applies
- Each bounded context has its own domain model
- Different contexts can have different models for same concept
- Example: "Customer" in Sales context vs "Customer" in Shipping context

**Ubiquitous Language:**
- Language used by all team members
- Based on domain model
- Used in code, documentation, conversations
- Evolves with domain understanding

**Context Mapping:**
- Maps relationships between bounded contexts
- Patterns: Shared Kernel, Customer-Supplier, Conformist, Anticorruption Layer, etc.

**Benefits:**
- Clear boundaries
- Team autonomy
- Reduced complexity
- Better domain modeling

**Common Patterns:**

**1. Shared Kernel:**
- Shared code between contexts
- Requires coordination
- Use sparingly

**2. Customer-Supplier:**
- Upstream (supplier) provides to downstream (customer)
- Customer has some influence

**3. Conformist:**
- Downstream conforms to upstream
- No influence on upstream
- Simplest integration

**4. Anticorruption Layer:**
- Translates between contexts
- Protects domain model
- Isolates legacy systems

**5. Separate Ways:**
- Contexts are completely independent
- No integration between them
- Used when integration cost exceeds benefit

**6. Open Host Service:**
- Context provides standardized protocol
- Multiple contexts can consume it
- Example: REST API, GraphQL endpoint

**Real-World Example - E-commerce Platform:**

**Bounded Contexts:**
1. **Sales Context**: Handles orders, customers, products
   - Customer: ID, Name, Email, CreditLimit
   - Order: Customer, Items, Total, Status

2. **Shipping Context**: Handles fulfillment, delivery
   - Customer: ID, Name, Address, PreferredCarrier
   - Shipment: OrderID, Address, Carrier, TrackingNumber

3. **Inventory Context**: Manages stock levels
   - Product: SKU, Quantity, Location
   - StockMovement: Product, Quantity, Type

**Ubiquitous Language in Practice:**

**Bad (Technical Language):**
- "OrderEntity"
- "CustomerDTO"
- "OrderLineItem"
- "PaymentTransaction"

**Good (Domain Language):**
- "Order"
- "Customer"
- "OrderItem"
- "Payment"

**Context Mapping Example:**


Sales Context (Upstream)
    ↓ Customer-Supplier
Shipping Context (Downstream)
    ↓ Conformist
Legacy Shipping System
    ↓ Anticorruption Layer
Inventory Context


**Key Takeaways:**
- Bounded contexts define clear boundaries
- Each context has its own domain model
- Ubiquitous language improves communication
- Context mapping shows relationships
- Choose integration patterns based on needs

**Self-Assessment Questions:**
1. Why can "Customer" have different models in different contexts?
2. How does ubiquitous language improve team communication?
3. When would you use an Anticorruption Layer?
4. What's the difference between Customer-Supplier and Conformist patterns?
5. Give an example of how bounded contexts enable team autonomy.`,
					CodeExamples: `Bounded Context Example:

Sales Context:
type Customer struct {
    ID       int
    Name     string
    Email    string
    CreditLimit Money
}

Shipping Context:
type Customer struct {
    ID        int
    Name      string
    Address   Address
    PreferredCarrier string
}

Same concept (Customer), different model in each context

Ubiquitous Language Example:

Domain Terms:
- Order (not "OrderEntity" or "OrderDTO")
- OrderItem (not "LineItem" or "OrderLine")
- Customer (not "Client" or "User")
- Money (not "Decimal" or "Amount")

Code reflects domain language:
type Order struct {
    Customer Customer
    Items    []OrderItem
    Total    Money
}

Context Mapping:

┌──────────────┐      ┌──────────────┐
│   Sales      │──────│  Shipping     │
│   Context    │      │  Context      │
└──────────────┘      └──────────────┘
       │                     │
       │  Customer-Supplier  │
       │  (Sales -> Shipping) │
       │                     │
       ▼                     ▼
┌──────────────┐      ┌──────────────┐
│  Inventory   │      │  Payment    │
│  Context     │      │  Context     │
└──────────────┘      └──────────────┘

Anticorruption Layer Example:

Legacy System (External):
type LegacyCustomer struct {
    CUST_ID    int
    CUST_NAME  string
    CUST_EMAIL string
}

Anticorruption Layer:
type CustomerAdapter struct {
    legacyService LegacyCustomerService
}

func (a *CustomerAdapter) GetCustomer(id int) (*Customer, error) {
    legacy := a.legacyService.GetCustomer(id)
    return &Customer{
        ID:    legacy.CUST_ID,
        Name:  legacy.CUST_NAME,
        Email: legacy.CUST_EMAIL,
    }, nil
}

Protects domain model from legacy system`,
				},
				{
					Title: "Tactical Design: Entities, Value Objects, Aggregates",
					Content: `Tactical Design focuses on building blocks within a bounded context.

**Entities:**
- Objects with unique identity
- Identity remains same even if attributes change
- Example: User (ID: 123) remains same even if name changes
- Compared by identity, not attributes

**Value Objects:**
- Objects defined by their attributes
- Immutable
- No identity
- Example: Money, Address, Email
- Compared by value

**Aggregates:**
- Cluster of entities and value objects
- Aggregate Root: Single entity that controls access
- Consistency boundary
- External references only to aggregate root
- Example: Order (root) contains OrderItems

**Domain Services:**
- Operations that don't belong to entities or value objects
- Stateless operations
- Cross-entity logic
- Example: TransferMoney between accounts

**Repositories:**
- Abstraction for persistence
- Retrieves aggregates by identity
- Saves aggregates
- Hides persistence details

**Factories:**
- Creates complex aggregates
- Encapsulates creation logic
- Ensures valid aggregates

**Entity vs Value Object - Decision Guide:**

**Use Entity When:**
- Object has unique identity
- Identity persists over time
- Object can change attributes
- Need to track object over time
- Example: User, Order, Product

**Use Value Object When:**
- Object defined by its attributes
- Immutability is important
- No need to track identity
- Equality based on values
- Example: Money, Address, Email, DateRange

**Aggregate Design Guidelines:**

**1. Keep Aggregates Small:**
- Large aggregates = contention and performance issues
- Small aggregates = better scalability
- Rule of thumb: 1-2 entities per aggregate

**2. Consistency Boundary:**
- All changes within aggregate are consistent
- Changes across aggregates use eventual consistency
- Example: Order and Payment are separate aggregates

**3. Aggregate Root Responsibilities:**
- Controls access to aggregate members
- Enforces invariants
- Manages aggregate lifecycle
- Only root can be referenced externally

**Real-World Example - E-commerce:**

**Entities:**
- Order (ID: 12345) - identity persists
- Customer (ID: 789) - identity persists
- Product (SKU: "ABC-123") - identity persists

**Value Objects:**
- Money: Amount + Currency (immutable)
- Address: Street, City, Zip (immutable)
- Email: String with validation (immutable)

**Aggregates:**
- Order Aggregate:
  - Root: Order
  - Members: OrderItems (value objects)
  - Invariant: Order total = sum of item totals
  - External references only to Order

**Key Takeaways:**
- Entities have identity, Value Objects have value
- Aggregates define consistency boundaries
- Aggregate root controls access
- Keep aggregates small
- Use domain services for cross-aggregate logic

**Self-Assessment Questions:**
1. What's the difference between an Entity and a Value Object?
2. Why should aggregates be kept small?
3. What is an aggregate root and why is it important?
4. When would you use a Domain Service instead of an Entity method?
5. Give an example of a Value Object and explain why it's immutable.`,
					CodeExamples: `Entity Example:

type User struct {
    ID    int    // Identity
    Name  string // Can change
    Email string // Can change
}

// Same user even if name changes
user1 := User{ID: 1, Name: "John"}
user2 := User{ID: 1, Name: "Johnny"}
// user1 == user2 (same identity)

Value Object Example:

type Money struct {
    Amount   float64
    Currency string
}

func (m Money) Add(other Money) Money {
    if m.Currency != other.Currency {
        panic("Cannot add different currencies")
    }
    return Money{
        Amount:   m.Amount + other.Amount,
        Currency: m.Currency,
    }
}

// Immutable - operations return new instances
money1 := Money{Amount: 100, Currency: "USD"}
money2 := money1.Add(Money{Amount: 50, Currency: "USD"})
// money1 unchanged, money2 is new instance

Aggregate Example:

// Aggregate Root
type Order struct {
    ID       int
    Customer Customer
    Items    []OrderItem // Part of aggregate
    Status   OrderStatus
}

func (o *Order) AddItem(product Product, quantity int) {
    if o.Status != OrderStatusDraft {
        panic("Cannot modify confirmed order")
    }
    item := OrderItem{
        Product:  product,
        Quantity: quantity,
        Price:    product.Price,
    }
    o.Items = append(o.Items, item)
}

func (o *Order) Confirm() {
    if len(o.Items) == 0 {
        panic("Cannot confirm empty order")
    }
    o.Status = OrderStatusConfirmed
}

// OrderItem is part of Order aggregate
type OrderItem struct {
    Product  Product
    Quantity int
    Price    Money
}

// External code references Order, not OrderItem directly
order := orderRepo.FindByID(orderID)
order.AddItem(product, 2) // Access through aggregate root

Domain Service Example:

type TransferService struct {
    accountRepo AccountRepository
}

func (s *TransferService) Transfer(fromID, toID int, amount Money) error {
    fromAccount := s.accountRepo.FindByID(fromID)
    toAccount := s.accountRepo.FindByID(toID)
    
    if fromAccount.Balance.LessThan(amount) {
        return errors.New("Insufficient funds")
    }
    
    fromAccount.Withdraw(amount)
    toAccount.Deposit(amount)
    
    s.accountRepo.Save(fromAccount)
    s.accountRepo.Save(toAccount)
    
    return nil
}

Repository Example:

type OrderRepository interface {
    FindByID(id int) (*Order, error)
    Save(order *Order) error
    FindByCustomerID(customerID int) ([]*Order, error)
}

// Implementation hides persistence details
type orderRepositoryImpl struct {
    db *sql.DB
}

func (r *orderRepositoryImpl) FindByID(id int) (*Order, error) {
    // Database query, map to domain model
    return order, nil
}

func (r *orderRepositoryImpl) Save(order *Order) error {
    // Save aggregate (including items)
    return nil
}`,
				},
				{
					Title: "Domain Events and Event Sourcing",
					Content: `Domain Events represent something important that happened in the domain.

**Domain Events:**
- Immutable facts about what happened
- Named in past tense (OrderPlaced, PaymentReceived)
- Contain relevant data
- Published when domain logic completes

**Benefits:**
- Decouple components
- Enable event-driven architecture
- Audit trail
- Enable eventual consistency

**Event Sourcing:**
- Store events instead of current state
- Rebuild state by replaying events
- Complete audit trail
- Time travel (state at any point)

**CQRS Integration:**
- Command Query Responsibility Segregation
- Separate read and write models
- Write model uses events
- Read model optimized for queries

**Event Store:**
- Append-only log of events
- Events are immutable
- Can replay to rebuild state
- Supports snapshots for performance`,
					CodeExamples: `Domain Event Example:

type OrderPlacedEvent struct {
    OrderID    int
    CustomerID int
    Items      []OrderItem
    Total      Money
    Timestamp  time.Time
}

type Order struct {
    ID       int
    Customer Customer
    Items    []OrderItem
    Status   OrderStatus
    events   []DomainEvent
}

func (o *Order) Confirm() {
    if o.Status != OrderStatusDraft {
        panic("Cannot confirm")
    }
    o.Status = OrderStatusConfirmed
    
    // Publish domain event
    event := OrderPlacedEvent{
        OrderID:    o.ID,
        CustomerID: o.Customer.ID,
        Items:      o.Items,
        Total:      o.CalculateTotal(),
        Timestamp:  time.Now(),
    }
    o.events = append(o.events, event)
}

func (o *Order) GetEvents() []DomainEvent {
    return o.events
}

Event Handler:

type OrderPlacedHandler struct {
    emailService EmailService
    inventoryService InventoryService
}

func (h *OrderPlacedHandler) Handle(event OrderPlacedEvent) {
    // Send confirmation email
    h.emailService.SendOrderConfirmation(event.CustomerID, event.OrderID)
    
    // Reserve inventory
    for _, item := range event.Items {
        h.inventoryService.Reserve(item.Product.ID, item.Quantity)
    }
}

Event Sourcing Example:

// Store events
events := []Event{
    OrderCreatedEvent{OrderID: 1, CustomerID: 123},
    ItemAddedEvent{OrderID: 1, ProductID: 456, Quantity: 2},
    OrderConfirmedEvent{OrderID: 1},
}

// Rebuild state
order := &Order{}
for _, event := range events {
    order.Apply(event)
}

func (o *Order) Apply(event Event) {
    switch e := event.(type) {
    case OrderCreatedEvent:
        o.ID = e.OrderID
        o.CustomerID = e.CustomerID
        o.Status = OrderStatusDraft
    case ItemAddedEvent:
        o.Items = append(o.Items, OrderItem{
            ProductID: e.ProductID,
            Quantity:  e.Quantity,
        })
    case OrderConfirmedEvent:
        o.Status = OrderStatusConfirmed
    }
}

Event Store:

type EventStore interface {
    Append(streamID string, events []Event) error
    GetEvents(streamID string) ([]Event, error)
    GetEventsSince(streamID string, version int) ([]Event, error)
}

CQRS with Event Sourcing:

Write Side:
- Commands create events
- Events stored in event store
- State rebuilt from events

Read Side:
- Read models updated from events
- Optimized for queries
- Can use different database (NoSQL)
- Eventually consistent`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          35,
			Title:       "Microservices Architecture Patterns",
			Description: "Learn microservices patterns including service decomposition, API Gateway, service mesh, and distributed data management.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "Service Decomposition Strategies",
					Content: `Decomposing a monolith into microservices requires careful consideration of boundaries and responsibilities.

**Decomposition Strategies:**

**1. Business Capability:**
- Decompose by business capabilities
- Each service owns a business function
- Example: Order Service, Payment Service, Shipping Service

**2. Domain-Driven Design:**
- Use bounded contexts as service boundaries
- Each service owns a domain
- Example: Customer Service, Product Service, Order Service

**3. Data Ownership:**
- Each service owns its data
- No shared databases
- Data accessed only through service API

**4. Team Structure:**
- Two-pizza team rule
- Each team owns one or few services
- Team autonomy

**Principles:**
- **Single Responsibility**: Each service has one responsibility
- **Autonomy**: Services can be developed and deployed independently
- **Data Ownership**: Each service owns its data
- **API-First**: Services communicate via APIs

**Common Patterns:**

**Database per Service:**
- Each service has its own database
- No shared databases
- Enables independent evolution

**API Gateway:**
- Single entry point for clients
- Routes requests to services
- Handles cross-cutting concerns

**Service Mesh:**
- Infrastructure layer for service communication
- Handles load balancing, retries, circuit breakers
- Transparent to application code`,
					CodeExamples: `Business Capability Decomposition:

Monolith:
┌─────────────────────────────────┐
│  E-commerce Application        │
│  - Order Management            │
│  - Payment Processing          │
│  - Inventory Management        │
│  - Shipping                    │
│  - Customer Management         │
└─────────────────────────────────┘

Microservices:
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Order Service│  │Payment Service│  │Shipping Service│
└──────────────┘  └──────────────┘  └──────────────┘
┌──────────────┐  ┌──────────────┐
│Inventory     │  │Customer      │
│Service       │  │Service       │
└──────────────┘  └──────────────┘

Domain-Driven Decomposition:

Bounded Contexts -> Services:
- Sales Context -> Order Service
- Payment Context -> Payment Service
- Shipping Context -> Shipping Service
- Customer Context -> Customer Service

Database per Service:

Order Service:
┌──────────────┐
│ Order Service│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Order DB     │
│ (PostgreSQL) │
└──────────────┘

Payment Service:
┌──────────────┐
│Payment Service│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Payment DB   │
│ (MySQL)      │
└──────────────┘

Service Communication:

Synchronous:
Order Service -> HTTP/REST -> Payment Service
Order Service -> gRPC -> Inventory Service

Asynchronous:
Order Service -> Message Queue -> Shipping Service
Order Service -> Event Stream -> Analytics Service`,
				},
				{
					Title: "API Gateway and Service Mesh",
					Content: `API Gateway and Service Mesh are key patterns for managing microservices communication.

**API Gateway:**

**Responsibilities:**
- Single entry point for clients
- Request routing to services
- Authentication and authorization
- Rate limiting
- Request/response transformation
- Aggregation of multiple service calls
- Protocol translation

**Benefits:**
- Simplified client code
- Centralized cross-cutting concerns
- Service abstraction
- Load balancing

**Patterns:**
- **Backend for Frontend (BFF)**: Different gateway per client type
- **Aggregation**: Combine multiple service responses
- **Protocol Translation**: REST to gRPC, etc.

**Service Mesh:**

**Components:**
- **Data Plane**: Sidecar proxies (Envoy, Linkerd)
- **Control Plane**: Manages routing, policies, observability

**Features:**
- Load balancing
- Service discovery
- Retries and circuit breakers
- Security (mTLS)
- Observability (metrics, tracing)
- Traffic management

**Benefits:**
- Transparent to application code
- Consistent behavior across services
- Centralized control
- Language-agnostic`,
					CodeExamples: `API Gateway Architecture:

Client
  │
  ▼
┌──────────────┐
│ API Gateway  │
│ - Auth       │
│ - Routing    │
│ - Rate Limit │
└──────┬───────┘
       │
   ┌───┴───┬─────────┬─────────┐
   │       │         │         │
   ▼       ▼         ▼         ▼
┌─────┐ ┌─────┐  ┌─────┐  ┌─────┐
│Order│ │Pay  │  │Ship │  │Inv  │
│Svc  │ │Svc  │  │Svc  │  │Svc  │
└─────┘ └─────┘  └─────┘  └─────┘

API Gateway Example:

type APIGateway struct {
    orderService OrderService
    paymentService PaymentService
    authService AuthService
}

func (gw *APIGateway) CreateOrder(w http.ResponseWriter, r *http.Request) {
    // Authentication
    user, err := gw.authService.Authenticate(r)
    if err != nil {
        http.Error(w, "Unauthorized", http.StatusUnauthorized)
        return
    }
    
    // Rate limiting
    if !gw.rateLimiter.Allow(user.ID) {
        http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
        return
    }
    
    // Route to service
    order, err := gw.orderService.CreateOrder(user.ID, r.Body)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    json.NewEncoder(w).Encode(order)
}

Service Mesh Architecture:

Service A                    Service B
  │                            │
  │                            │
  ▼                            ▼
┌──────────┐                ┌──────────┐
│ Sidecar   │<─────────────>│ Sidecar   │
│ Proxy     │                │ Proxy     │
└──────────┘                └──────────┘
  │                            │
  │                            │
  ▼                            ▼
┌──────────┐                ┌──────────┐
│ Service A│                │ Service B│
└──────────┘                └──────────┘

Control Plane manages:
- Routing rules
- Security policies
- Observability config

Service Mesh Benefits:
- Automatic retries
- Circuit breakers
- Load balancing
- mTLS encryption
- Distributed tracing`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
