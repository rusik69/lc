package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          2309,
			Title:       "SOLID Principles",
			Description: "Master the five SOLID principles of object-oriented design that form the foundation for writing maintainable, flexible, and scalable software.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Single Responsibility Principle (SRP)",
					Content: `The Single Responsibility Principle states: **"A class should have one, and only one, reason to change."** — Robert C. Martin

This is the most fundamental of the SOLID principles. It's not about a class doing "one thing" — it's about a class being responsible to **one actor** (one stakeholder or group of stakeholders).

**Why SRP Matters:**

When a class has multiple responsibilities, changes for one reason can break functionality for another. This creates fragile, hard-to-test code.

**The Classic Violation:**

Imagine an Employee class that:
1. Calculates pay (reports to CFO)
2. Generates reports (reports to COO)
3. Saves to database (reports to CTO)

Three actors, three reasons to change — SRP VIOLATED.

**Before (Violating SRP):**

` + "```" + `
┌──────────────────────────┐
│        Employee          │
├──────────────────────────┤
│ + calculatePay()         │ ← CFO's team changes this
│ + generateReport()       │ ← COO's team changes this
│ + save()                 │ ← CTO's team changes this
└──────────────────────────┘
` + "```" + `

If CFO wants to change how overtime is calculated, you risk breaking report generation or database saving.

**After (Following SRP):**

` + "```" + `
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  PayCalculator │  │ ReportGenerator│  │ EmployeeRepo   │
├────────────────┤  ├────────────────┤  ├────────────────┤
│ + calculate()  │  │ + generate()   │  │ + save()       │
└────────────────┘  └────────────────┘  └────────────────┘
` + "```" + `

Now each class has exactly one reason to change. Changes to pay calculation can't break reports.

**SRP in Go — Practical Example:**

Go naturally encourages SRP through small interfaces and composition. In Go, think of SRP as: **"A package or struct should have one cohesive purpose."**

**Common SRP Violations:**
- A handler function that validates, processes business logic, AND saves to database
- A "God struct" with 30 methods covering unrelated domains
- A package called "utils" that contains everything

**How to Detect SRP Violations:**
1. Look for classes/structs with many unrelated methods
2. Look for methods that change for different reasons
3. Look for "and" in class descriptions ("this class authenticates AND logs AND sends emails")
4. Check if different stakeholders would request changes to the same class

**Best Practices:**
- Extract responsibilities into separate types
- Use composition to combine behaviors
- Keep packages focused on a single domain concept
- Name your types by what they DO, not what they ARE`,
					CodeExamples: `// BAD: Violating SRP — UserService does everything
type UserService struct {
    db *sql.DB
}

func (s *UserService) CreateUser(name, email string) error {
    // Validation logic
    if !strings.Contains(email, "@") {
        return errors.New("invalid email")
    }
    // Business logic
    hashedPassword := hashPassword(generateTempPassword())
    // Database logic
    _, err := s.db.Exec("INSERT INTO users ...", name, email, hashedPassword)
    if err != nil {
        return err
    }
    // Notification logic
    sendWelcomeEmail(email, name)
    return nil
}

// GOOD: Each struct has one responsibility
type EmailValidator struct{}

func (v *EmailValidator) Validate(email string) error {
    if !strings.Contains(email, "@") {
        return errors.New("invalid email")
    }
    return nil
}

type UserRepository struct {
    db *sql.DB
}

func (r *UserRepository) Save(user *User) error {
    _, err := r.db.Exec("INSERT INTO users ...", user.Name, user.Email)
    return err
}

type NotificationService struct {
    mailer Mailer
}

func (n *NotificationService) WelcomeUser(email, name string) error {
    return n.mailer.Send(email, "Welcome!", "Hello "+name)
}

// Orchestrator composes the single-responsibility components
type CreateUserUseCase struct {
    validator    *EmailValidator
    repo         *UserRepository
    notification *NotificationService
}

func (uc *CreateUserUseCase) Execute(name, email string) error {
    if err := uc.validator.Validate(email); err != nil {
        return err
    }
    user := &User{Name: name, Email: email}
    if err := uc.repo.Save(user); err != nil {
        return err
    }
    return uc.notification.WelcomeUser(email, name)
}`,
				},
				{
					Title: "Open/Closed Principle (OCP)",
					Content: `The Open/Closed Principle states: **"Software entities should be open for extension, but closed for modification."** — Bertrand Meyer

**In plain language:** You should be able to add new behavior without changing existing code. New features = new code, not changed code.

**Why This Matters:**
- Existing code is tested and working — don't break it
- Each modification risks introducing bugs
- In large teams, modifying shared code causes merge conflicts
- Closed modules can be compiled, deployed, and tested independently

**The Key Insight:**
OCP is achieved through **abstraction**. Define interfaces (contracts), then add new implementations without changing the interface or its consumers.

**Classic Violation:**

` + "```" + `
func CalculateArea(shape string, dimensions ...float64) float64 {
    switch shape {
    case "circle":
        return math.Pi * dimensions[0] * dimensions[0]
    case "rectangle":
        return dimensions[0] * dimensions[1]
    // Adding triangle? Must MODIFY this function!
    }
}
` + "```" + `

Every new shape requires modifying this function. It's CLOSED for extension and OPEN for modification — the opposite of what we want.

**Following OCP:**

` + "```" + `
type Shape interface {
    Area() float64
}

// To add a new shape, just add a new type.
// No existing code changes.
` + "```" + `

**OCP in Go:**
Go's interface system makes OCP natural. Interfaces are implicitly satisfied — any type that has the right methods automatically implements the interface. This means:
- Define behavior through interfaces
- Add new implementations freely
- Existing code doesn't need to change

**Strategy Pattern (OCP in action):**
Instead of a switch statement that grows with every new case, inject different strategies:

**Common OCP Patterns:**
1. **Strategy Pattern**: Swap algorithms via interfaces
2. **Decorator Pattern**: Wrap to add behavior
3. **Plugin Architecture**: Load new behavior at runtime
4. **Middleware Chains**: Add processing steps without modifying handler

**When OCP Goes Too Far:**
Don't create abstractions prematurely. If you only have one implementation, a concrete type is fine. Add the interface when you actually need a second implementation. As Go proverb says: "Accept interfaces, return structs."`,
					CodeExamples: `// OCP with Go interfaces

// Define the contract (closed for modification)
type NotificationSender interface {
    Send(to, subject, body string) error
}

// Implementation 1: Email
type EmailSender struct{ smtpHost string }
func (e *EmailSender) Send(to, subject, body string) error {
    // send via SMTP
    return nil
}

// Implementation 2: SMS (added later, no existing code changed)
type SMSSender struct{ apiKey string }
func (s *SMSSender) Send(to, subject, body string) error {
    // send via SMS API
    return nil
}

// Implementation 3: Slack (added even later, still no changes)
type SlackSender struct{ webhookURL string }
func (s *SlackSender) Send(to, subject, body string) error {
    // send via Slack webhook
    return nil
}

// Consumer is closed for modification
type OrderService struct {
    notifier NotificationSender // works with ANY sender
}

func (o *OrderService) PlaceOrder(order Order) error {
    // ... process order ...
    return o.notifier.Send(order.CustomerEmail, "Order Placed", "Your order is confirmed!")
}

// Middleware chain example (OCP — add behavior without modifying handler)
type Middleware func(http.Handler) http.Handler

func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        log.Printf("%s %s", r.Method, r.URL.Path)
        next.ServeHTTP(w, r)
    })
}

func AuthMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        token := r.Header.Get("Authorization")
        if token == "" {
            http.Error(w, "unauthorized", 401)
            return
        }
        next.ServeHTTP(w, r)
    })
}

// Usage: chain middlewares without modifying handler
handler := AuthMiddleware(LoggingMiddleware(myHandler))`,
				},
				{
					Title: "Liskov Substitution Principle (LSP)",
					Content: `The Liskov Substitution Principle states: **"Objects of a supertype should be replaceable with objects of a subtype without altering the correctness of the program."** — Barbara Liskov, 1987

**In plain English:** If your code works with a base type, it must also work correctly with any derived type. Subtypes must honor the contract of their parent type.

**Why LSP Matters:**
LSP is the principle that makes polymorphism work. Without it, switching implementations breaks your system. If code depends on an interface, every implementation of that interface must behave consistently.

**Classic Violation — The Rectangle/Square Problem:**

A Square IS-A Rectangle mathematically, but in code:
- Rectangle: SetWidth and SetHeight are independent
- Square: SetWidth must also set height (and vice versa)

If code assumes width and height are independent (valid for Rectangle), Square breaks that assumption. LSP violated.

**LSP in Go:**
Go doesn't have inheritance but has interfaces. LSP applies to interface implementations: every implementation must fulfill the interface's behavioral contract, not just its method signatures.

**Behavioral Contract Includes:**
1. **Preconditions**: Can't be stricter than the interface promises
2. **Postconditions**: Must at least deliver what the interface promises
3. **Invariants**: Must maintain all properties the interface guarantees
4. **No surprises**: Don't throw unexpected errors or produce unexpected side effects

**Real-World Go Violation:**

` + "```" + `
type Storage interface {
    Save(key string, value []byte) error
    Get(key string) ([]byte, error)
}

// FileStorage: works as expected
type FileStorage struct{}
func (f *FileStorage) Save(key string, value []byte) error { /* writes file */ }
func (f *FileStorage) Get(key string) ([]byte, error) { /* reads file */ }

// ReadOnlyStorage: VIOLATES LSP
type ReadOnlyStorage struct{}
func (r *ReadOnlyStorage) Save(key string, value []byte) error {
    return errors.New("storage is read-only") // SURPRISE! Save always fails
}
func (r *ReadOnlyStorage) Get(key string) ([]byte, error) { /* reads */ }
` + "```" + `

Code using Storage expects Save to work. ReadOnlyStorage breaks that contract.

**How To Fix:**
- Split the interface: Reader and Writer
- ReadOnlyStorage only implements Reader
- Code that needs writing takes Writer interface

**LSP Checklist:**
- Can I substitute any implementation without tests failing?
- Does every implementation honor the behavioral contract?
- Are there "special case" implementations that partially work?
- Do subtype methods have unexpected side effects?`,
					CodeExamples: `// LSP-compliant interface design in Go

// BAD: One big interface that not all types can fulfill
type Storage interface {
    Save(key string, value []byte) error
    Get(key string) ([]byte, error)
    Delete(key string) error
    List() ([]string, error)
}

// GOOD: Segregated interfaces (follows LSP + ISP)
type Reader interface {
    Get(key string) ([]byte, error)
}

type Writer interface {
    Save(key string, value []byte) error
}

type Deleter interface {
    Delete(key string) error
}

type Lister interface {
    List() ([]string, error)
}

// Compose interfaces as needed
type ReadWriter interface {
    Reader
    Writer
}

// FileStorage implements all interfaces
type FileStorage struct{ dir string }
func (f *FileStorage) Get(key string) ([]byte, error)          { /* ... */ return nil, nil }
func (f *FileStorage) Save(key string, value []byte) error     { /* ... */ return nil }
func (f *FileStorage) Delete(key string) error                 { /* ... */ return nil }
func (f *FileStorage) List() ([]string, error)                 { /* ... */ return nil, nil }

// CacheStorage only reads — implements Reader only
type CacheStorage struct{ cache map[string][]byte }
func (c *CacheStorage) Get(key string) ([]byte, error) {
    if v, ok := c.cache[key]; ok {
        return v, nil
    }
    return nil, errors.New("not found")
}

// Functions declare what they actually need
func FetchData(r Reader, key string) ([]byte, error) {
    return r.Get(key) // Works with FileStorage AND CacheStorage
}

func SaveData(w Writer, key string, data []byte) error {
    return w.Save(key, data) // Only works with types that can write
}`,
				},
				{
					Title: "Interface Segregation Principle (ISP)",
					Content: `The Interface Segregation Principle states: **"Clients should not be forced to depend on interfaces they do not use."** — Robert C. Martin

**In plain language:** Many small, focused interfaces are better than one large, general-purpose interface.

**Why ISP Matters:**
- Clients only depend on methods they actually need
- Changes to unused methods don't force recompilation/redeployment
- Easier to implement (fewer methods to satisfy)
- Better documentation of intent (interface name tells you what's needed)

**Go is ISP by Default:**
Go's standard library is a masterclass in ISP. Look at the io package:

` + "```" + `
type Reader interface { Read(p []byte) (n int, err error) }
type Writer interface { Write(p []byte) (n int, err error) }
type Closer interface { Close() error }
type ReadWriter interface { Reader; Writer }
type ReadCloser interface { Reader; Closer }
type WriteCloser interface { Writer; Closer }
type ReadWriteCloser interface { Reader; Writer; Closer }
` + "```" + `

Each interface has **one method**. They compose into larger interfaces when needed. A function that only reads takes io.Reader — it doesn't care if the source can also write or close.

**The Fat Interface Problem:**

` + "```" + `
// BAD: Fat interface forces all implementations to have all methods
type UserManager interface {
    CreateUser(u User) error
    DeleteUser(id int) error
    GetUser(id int) (*User, error)
    ListUsers() ([]*User, error)
    ExportUsersCSV() ([]byte, error)
    SendWelcomeEmail(id int) error
    ResetPassword(id int) error
    GenerateReport() ([]byte, error)
}
` + "```" + `

If you only need to read users, you're forced to depend on email-sending and report-generating methods too.

**ISP Applied:**

` + "```" + `
type UserReader interface {
    GetUser(id int) (*User, error)
    ListUsers() ([]*User, error)
}
type UserWriter interface {
    CreateUser(u User) error
    DeleteUser(id int) error
}
type UserNotifier interface {
    SendWelcomeEmail(id int) error
}
` + "```" + `

**Go Proverb:** "The bigger the interface, the weaker the abstraction."

**ISP Guidelines for Go:**
1. Start with no interface — use concrete types
2. When you need a second implementation, extract the minimal interface
3. Define interfaces where they're USED, not where they're implemented
4. Prefer 1-2 method interfaces
5. Compose small interfaces into larger ones when needed`,
					CodeExamples: `// ISP in action: Define interfaces at the consumer

// Package "auth" only needs to read users
package auth

type UserGetter interface {
    GetUser(id int) (*User, error)
}

func Authenticate(repo UserGetter, id int, password string) error {
    user, err := repo.GetUser(id)
    if err != nil {
        return err
    }
    if !checkPassword(user, password) {
        return errors.New("invalid credentials")
    }
    return nil
}

// Package "admin" needs full CRUD
package admin

type UserStore interface {
    CreateUser(u User) error
    GetUser(id int) (*User, error)
    DeleteUser(id int) error
    ListUsers() ([]*User, error)
}

// Package "reports" only needs listing
package reports

type UserLister interface {
    ListUsers() ([]*User, error)
}

func GenerateUserReport(lister UserLister) ([]byte, error) {
    users, err := lister.ListUsers()
    if err != nil {
        return nil, err
    }
    // ... generate report ...
    return report, nil
}

// One concrete type satisfies ALL these interfaces
type PostgresUserRepo struct{ db *sql.DB }

func (r *PostgresUserRepo) CreateUser(u User) error   { /* ... */ return nil }
func (r *PostgresUserRepo) GetUser(id int) (*User, error) { /* ... */ return nil, nil }
func (r *PostgresUserRepo) DeleteUser(id int) error    { /* ... */ return nil }
func (r *PostgresUserRepo) ListUsers() ([]*User, error) { /* ... */ return nil, nil }

// auth.Authenticate(repo, id, pass) -- works! (only uses GetUser)
// admin.ManageUsers(repo)           -- works! (uses full CRUD)
// reports.GenerateUserReport(repo)  -- works! (only uses ListUsers)`,
				},
				{
					Title: "Dependency Inversion Principle (DIP)",
					Content: `The Dependency Inversion Principle states:

1. **"High-level modules should not depend on low-level modules. Both should depend on abstractions."**
2. **"Abstractions should not depend on details. Details should depend on abstractions."**

**In plain language:** Your business logic should not import database packages, HTTP libraries, or external SDKs directly. Instead, it should depend on interfaces that those packages implement.

**Why DIP Matters:**
- Business logic becomes framework-independent
- You can swap databases, APIs, or frameworks without changing core logic
- Testing becomes trivial — inject mocks instead of real dependencies
- Modules can be developed and deployed independently

**The Dependency Direction Problem:**

Without DIP:
` + "```" + `
BusinessLogic → PostgresDatabase
BusinessLogic → StripePayment
BusinessLogic → SendGridEmail
` + "```" + `
Business logic is tightly coupled to specific implementations. Changing Stripe to PayPal means modifying business logic.

With DIP:
` + "```" + `
BusinessLogic → PaymentGateway (interface)
                     ↑
              StripeAdapter
              PayPalAdapter
` + "```" + `
Business logic depends on abstraction. Implementations depend on abstraction. Both point toward the abstraction.

**DIP in Go:**
Go makes DIP natural through implicit interface satisfaction. You don't need to declare "implements" — if the methods match, the type satisfies the interface.

**The Three-Layer Pattern:**
1. **Domain Layer**: Pure business logic, defines interfaces for what it needs
2. **Application Layer**: Orchestrates domain operations
3. **Infrastructure Layer**: Implements interfaces (database, HTTP, etc.)

Dependencies flow: Infrastructure → Application → Domain
But code flow is: Domain defines interfaces ← Infrastructure implements them

**Dependency Injection (DI):**
DI is the mechanism for applying DIP. Instead of creating dependencies inside a struct, inject them from outside:

- **Constructor Injection**: Pass dependencies via constructor (most common in Go)
- **Method Injection**: Pass dependencies as function parameters
- **Interface Injection**: Dependencies set via interface methods

**DIP Anti-Pattern — "New is Glue":**
Every time you call new/make for a dependency inside business logic, you're creating a hard dependency. Let the caller provide it.

**When NOT to use DIP:**
- Simple scripts or CLI tools
- When there's genuinely only one possible implementation
- When the abstraction adds complexity without benefit
- Standard library types (don't wrap fmt.Println in an interface)`,
					CodeExamples: `// WITHOUT DIP: Business logic depends on concrete database
package order

import "database/sql" // Direct dependency on SQL!

type OrderService struct {
    db *sql.DB  // Concrete dependency
}

func (s *OrderService) PlaceOrder(o Order) error {
    // Business logic mixed with database details
    _, err := s.db.Exec("INSERT INTO orders ...", o.ID, o.Total)
    return err
}
// Problem: Can't test without real database
// Problem: Can't switch to MongoDB without changing business logic

// ─────────────────────────────────────────

// WITH DIP: Business logic depends on abstraction

// Domain layer defines what it needs (interface)
type OrderRepository interface {
    Save(order Order) error
    FindByID(id string) (*Order, error)
}

// Business logic depends on interface
type OrderService struct {
    repo OrderRepository  // Abstraction, not concrete type
}

func (s *OrderService) PlaceOrder(o Order) error {
    if o.Total <= 0 {
        return errors.New("order total must be positive")
    }
    return s.repo.Save(o)  // Don't know or care if it's SQL, Mongo, or in-memory
}

// Infrastructure layer implements the interface
type PostgresOrderRepo struct {
    db *sql.DB
}
func (r *PostgresOrderRepo) Save(order Order) error {
    _, err := r.db.Exec("INSERT INTO orders ...", order.ID, order.Total)
    return err
}
func (r *PostgresOrderRepo) FindByID(id string) (*Order, error) {
    // ... SQL query ...
    return &Order{}, nil
}

// Test with mock — no database needed!
type MockOrderRepo struct {
    orders map[string]*Order
}
func (m *MockOrderRepo) Save(order Order) error {
    m.orders[order.ID] = &order
    return nil
}
func (m *MockOrderRepo) FindByID(id string) (*Order, error) {
    if o, ok := m.orders[id]; ok {
        return o, nil
    }
    return nil, errors.New("not found")
}

// Wire it all together in main.go
func main() {
    db, _ := sql.Open("postgres", "...")
    repo := &PostgresOrderRepo{db: db}
    service := &OrderService{repo: repo}  // Inject dependency
    // Test:
    // mockRepo := &MockOrderRepo{orders: make(map[string]*Order)}
    // service := &OrderService{repo: mockRepo}
}`,
				},
			},
		},
		{
			ID:          2310,
			Title:       "Design Patterns (GoF)",
			Description: "Learn the essential Gang of Four design patterns: Creational, Structural, and Behavioral patterns with practical Go examples.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "Creational Patterns: Factory and Builder",
					Content: `Creational patterns deal with object creation mechanisms. They abstract the instantiation process, making systems independent of how objects are created.

**Factory Method Pattern:**

**Intent:** Define an interface for creating an object, but let subclasses/implementations decide which class to instantiate.

**When to Use:**
- You don't know ahead of time which concrete type to create
- You want to centralize creation logic
- You want to return different implementations based on input
- You need to decouple creation from usage

**Real-World Analogy:** A restaurant menu. You order "burger" (abstract request) and the kitchen decides how to make it (concrete creation). You don't go into the kitchen.

**In Go:** Since Go doesn't have classes, the Factory pattern manifests as constructor functions that return interfaces.

**Builder Pattern:**

**Intent:** Separate the construction of a complex object from its representation, allowing the same construction process to create different representations.

**When to Use:**
- Object has many optional parameters
- Construction involves multiple steps
- You want to avoid "telescoping constructors" (functions with 10+ parameters)
- Different representations of the same construction process

**Real-World Analogy:** Ordering a custom pizza. You start with the base, then add toppings one by one. The builder lets you create many different pizzas using the same step-by-step process.

**Functional Options Pattern (Go idiom):**
Go has its own elegant alternative to Builder: the Functional Options pattern. Instead of a builder object, you pass option functions to the constructor.

**Singleton Pattern (Use Sparingly):**

**Intent:** Ensure a class has only one instance and provide global point of access.

**In Go:** Use sync.Once for thread-safe lazy initialization. But prefer dependency injection over singletons — singletons are global state and make testing hard.`,
					CodeExamples: `// FACTORY METHOD PATTERN
// Returns different implementations based on input

type Logger interface {
    Log(message string)
}

type ConsoleLogger struct{}
func (l *ConsoleLogger) Log(msg string) { fmt.Println("[CONSOLE]", msg) }

type FileLogger struct{ file *os.File }
func (l *FileLogger) Log(msg string) { fmt.Fprintln(l.file, "[FILE]", msg) }

type JSONLogger struct{ encoder *json.Encoder }
func (l *JSONLogger) Log(msg string) {
    l.encoder.Encode(map[string]string{"message": msg})
}

// Factory function
func NewLogger(logType string) (Logger, error) {
    switch logType {
    case "console":
        return &ConsoleLogger{}, nil
    case "file":
        f, err := os.OpenFile("app.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
        if err != nil { return nil, err }
        return &FileLogger{file: f}, nil
    case "json":
        return &JSONLogger{encoder: json.NewEncoder(os.Stdout)}, nil
    default:
        return nil, fmt.Errorf("unknown logger type: %s", logType)
    }
}

// ─────────────────────────────────────────

// FUNCTIONAL OPTIONS PATTERN (Go's Builder alternative)
type Server struct {
    host    string
    port    int
    timeout time.Duration
    maxConn int
    tls     bool
}

type ServerOption func(*Server)

func WithPort(port int) ServerOption {
    return func(s *Server) { s.port = port }
}

func WithTimeout(t time.Duration) ServerOption {
    return func(s *Server) { s.timeout = t }
}

func WithMaxConnections(n int) ServerOption {
    return func(s *Server) { s.maxConn = n }
}

func WithTLS(enabled bool) ServerOption {
    return func(s *Server) { s.tls = enabled }
}

func NewServer(host string, opts ...ServerOption) *Server {
    s := &Server{
        host:    host,
        port:    8080,           // sensible defaults
        timeout: 30 * time.Second,
        maxConn: 100,
        tls:     false,
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// Clean, readable construction:
server := NewServer("localhost",
    WithPort(9090),
    WithTimeout(60 * time.Second),
    WithTLS(true),
)

// ─────────────────────────────────────────

// SINGLETON with sync.Once (prefer DI over this)
var (
    dbInstance *Database
    dbOnce     sync.Once
)

func GetDatabase() *Database {
    dbOnce.Do(func() {
        dbInstance = &Database{/* ... */}
    })
    return dbInstance
}`,
				},
				{
					Title: "Structural Patterns: Adapter, Decorator, Facade",
					Content: `Structural patterns deal with object composition — how classes and objects are composed to form larger structures.

**Adapter Pattern:**

**Intent:** Convert the interface of a class into another interface clients expect. Let incompatible interfaces work together.

**When to Use:**
- You need to use an existing class but its interface doesn't match what you need
- You want to create a reusable class that cooperates with unrelated classes
- You're integrating with third-party libraries

**Real-World Analogy:** A power adapter that lets you plug a US device into a European outlet. The device and outlet don't change — the adapter bridges the gap.

**Decorator Pattern:**

**Intent:** Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.

**When to Use:**
- Add behavior to objects without modifying them
- Combine behaviors dynamically (logging + caching + retry)
- When subclassing would create too many combinations

**Real-World Analogy:** Coffee. Start with base coffee, then add milk (decorator), sugar (decorator), whipped cream (decorator). Each addition wraps the previous, adding behavior.

**In Go:** Decorators are incredibly common. The http.Handler middleware pattern IS the decorator pattern. io.Reader wrappers (bufio.NewReader, gzip.NewReader) are decorators.

**Facade Pattern:**

**Intent:** Provide a simplified interface to a complex subsystem.

**When to Use:**
- Complex subsystem with many components
- Clients don't need full subsystem access
- You want to reduce coupling between clients and subsystem
- You want to layer your system

**Real-World Analogy:** A hotel concierge. Instead of dealing with the restaurant, spa, taxi, and tour companies directly, you tell the concierge what you want and they handle it.

**Composite Pattern:**

**Intent:** Compose objects into tree structures. Let clients treat individual objects and compositions uniformly.

**When to Use:**
- Represent part-whole hierarchies (file systems, UI components, org charts)
- Clients should treat individual objects and compositions the same way`,
					CodeExamples: `// ADAPTER PATTERN
// Bridge between your code and a third-party library

// Your application expects this interface
type PaymentProcessor interface {
    Charge(amount float64, currency string) error
}

// Third-party Stripe SDK has different interface
type StripeSDK struct{}
func (s *StripeSDK) CreateCharge(amountCents int64, cur string, desc string) (*StripeCharge, error) {
    // ... Stripe API call ...
    return &StripeCharge{}, nil
}

// Adapter bridges the gap
type StripeAdapter struct {
    sdk *StripeSDK
}

func (a *StripeAdapter) Charge(amount float64, currency string) error {
    cents := int64(amount * 100)
    _, err := a.sdk.CreateCharge(cents, currency, "payment")
    return err
}

// Your code works with the interface — never knows about Stripe details
func ProcessPayment(pp PaymentProcessor, amount float64) error {
    return pp.Charge(amount, "USD")
}

// ─────────────────────────────────────────

// DECORATOR PATTERN (HTTP Middleware in Go)
type Handler func(w http.ResponseWriter, r *http.Request)

// Logging decorator
func WithLogging(h http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        h.ServeHTTP(w, r)
        log.Printf("%s %s took %v", r.Method, r.URL.Path, time.Since(start))
    })
}

// Caching decorator
func WithCaching(h http.Handler, ttl time.Duration) http.Handler {
    cache := make(map[string]cachedResponse)
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if cached, ok := cache[r.URL.Path]; ok && time.Since(cached.time) < ttl {
            w.Write(cached.body)
            return
        }
        // ... capture response, store in cache, pass through
        h.ServeHTTP(w, r)
    })
}

// Stack decorators: Logging wraps Caching wraps Handler
handler := WithLogging(WithCaching(myHandler, 5*time.Minute))

// ─────────────────────────────────────────

// FACADE PATTERN
// Simplify complex subsystem interaction

type OrderFacade struct {
    inventory  *InventoryService
    payment    *PaymentService
    shipping   *ShippingService
    email      *EmailService
}

// One simple method hides complex multi-step process
func (f *OrderFacade) PlaceOrder(order Order) error {
    // Step 1: Check inventory
    if !f.inventory.IsAvailable(order.Items) {
        return errors.New("items not available")
    }
    // Step 2: Process payment
    if err := f.payment.Charge(order.Total); err != nil {
        return fmt.Errorf("payment failed: %w", err)
    }
    // Step 3: Reserve inventory
    f.inventory.Reserve(order.Items)
    // Step 4: Schedule shipping
    tracking := f.shipping.Schedule(order.Address)
    // Step 5: Send confirmation
    f.email.SendConfirmation(order.CustomerEmail, tracking)
    return nil
}

// Client code is simple:
facade.PlaceOrder(order)
// Instead of calling 5 different services manually`,
				},
				{
					Title: "Behavioral Patterns: Strategy, Observer, Chain of Responsibility",
					Content: `Behavioral patterns deal with algorithms and the assignment of responsibilities between objects.

**Strategy Pattern:**

**Intent:** Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

**When to Use:**
- You need different variants of an algorithm
- You want to avoid conditional statements for selecting behavior
- Multiple classes differ only in their behavior

**In Go:** Strategy is just passing a function or interface implementation as a parameter. Go's first-class functions make this pattern trivially easy.

**Observer Pattern:**

**Intent:** Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

**When to Use:**
- When changes to one object require changing others but you don't know how many
- When an object should notify others without making assumptions about who they are
- Event-driven systems, UI updates, data synchronization

**Real-World Analogy:** YouTube subscriptions. When a channel uploads a video, all subscribers are notified. The channel doesn't need to know who the subscribers are.

**Chain of Responsibility Pattern:**

**Intent:** Let more than one object handle a request. Chain the receiving objects and pass the request along the chain until one of them handles it.

**When to Use:**
- Multiple objects can handle a request and the handler isn't known a priori
- You want to issue a request to one of several objects without specifying the receiver
- The set of handlers should be configured dynamically

**In Go:** Chain of Responsibility appears as middleware stacks, validation chains, and command pipelines.

**Template Method Pattern:**

**Intent:** Define the skeleton of an algorithm in a base operation, deferring some steps to subclasses/implementations.

**In Go:** Since there's no inheritance, use function fields or interface composition. Define the overall algorithm, but inject specific steps via functions or interfaces.`,
					CodeExamples: `// STRATEGY PATTERN
type CompressionStrategy interface {
    Compress(data []byte) ([]byte, error)
}

type GzipCompression struct{}
func (g *GzipCompression) Compress(data []byte) ([]byte, error) {
    var buf bytes.Buffer
    w := gzip.NewWriter(&buf)
    w.Write(data)
    w.Close()
    return buf.Bytes(), nil
}

type ZstdCompression struct{}
func (z *ZstdCompression) Compress(data []byte) ([]byte, error) {
    // zstd compression...
    return data, nil
}

type FileUploader struct {
    compression CompressionStrategy
}

func (u *FileUploader) Upload(data []byte) error {
    compressed, err := u.compression.Compress(data)
    if err != nil { return err }
    // upload compressed data...
    _ = compressed
    return nil
}

// Swap strategies at runtime
uploader := &FileUploader{compression: &GzipCompression{}}
uploader.Upload(data)

uploader.compression = &ZstdCompression{} // Switch strategy!
uploader.Upload(data)

// ─────────────────────────────────────────

// OBSERVER PATTERN
type Event struct {
    Type string
    Data interface{}
}

type EventHandler func(Event)

type EventBus struct {
    mu       sync.RWMutex
    handlers map[string][]EventHandler
}

func NewEventBus() *EventBus {
    return &EventBus{handlers: make(map[string][]EventHandler)}
}

func (b *EventBus) Subscribe(eventType string, handler EventHandler) {
    b.mu.Lock()
    defer b.mu.Unlock()
    b.handlers[eventType] = append(b.handlers[eventType], handler)
}

func (b *EventBus) Publish(event Event) {
    b.mu.RLock()
    defer b.mu.RUnlock()
    for _, handler := range b.handlers[event.Type] {
        go handler(event) // Non-blocking notification
    }
}

// Usage
bus := NewEventBus()
bus.Subscribe("order.created", func(e Event) {
    log.Println("Sending confirmation email for", e.Data)
})
bus.Subscribe("order.created", func(e Event) {
    log.Println("Updating inventory for", e.Data)
})
bus.Publish(Event{Type: "order.created", Data: orderID})

// ─────────────────────────────────────────

// CHAIN OF RESPONSIBILITY
type Validator func(value string) error

func Required(value string) error {
    if value == "" { return errors.New("required") }
    return nil
}

func MinLength(n int) Validator {
    return func(value string) error {
        if len(value) < n {
            return fmt.Errorf("minimum length is %d", n)
        }
        return nil
    }
}

func MaxLength(n int) Validator {
    return func(value string) error {
        if len(value) > n {
            return fmt.Errorf("maximum length is %d", n)
        }
        return nil
    }
}

// Chain validators
func Validate(value string, validators ...Validator) error {
    for _, v := range validators {
        if err := v(value); err != nil {
            return err
        }
    }
    return nil
}

// Usage: chain of validators
err := Validate(username, Required, MinLength(3), MaxLength(50))`,
				},
			},
		},
		{
			ID:          2311,
			Title:       "Domain-Driven Design (DDD)",
			Description: "Learn Domain-Driven Design concepts: bounded contexts, aggregates, entities, value objects, and strategic design for complex software systems.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "DDD Strategic Design",
					Content: `Domain-Driven Design (DDD), introduced by Eric Evans in 2003, is an approach to software development that centers the design on the core business domain and its logic.

**Why DDD?**
Most software projects fail not because of technical issues, but because developers don't understand the business domain well enough. DDD bridges this gap by making the domain model the heart of the software.

**Strategic Design — The Big Picture:**

Strategic design is about understanding the domain landscape and how different parts of the business relate to each other.

**Ubiquitous Language:**
The most important concept in DDD. A shared language between developers and domain experts that is used in code, conversations, and documentation.

- **Problem**: Developers say "user record", business says "customer account" — misunderstandings
- **Solution**: Agree on ONE term and use it EVERYWHERE (code, docs, conversations)
- **Example**: If the business calls it "Policy" not "Insurance Contract", the code should have a Policy struct, not an InsuranceContract struct

**Bounded Contexts:**
A bounded context defines the boundary within which a particular model and its ubiquitous language are valid.

` + "```" + `
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Sales Context  │     │ Shipping Context │     │ Billing Context  │
│                  │     │                  │     │                  │
│  Customer =      │     │  Customer =      │     │  Customer =      │
│  name, email,    │     │  name, address,  │     │  name, payment   │
│  preferences     │     │  delivery notes  │     │  method, balance │
└─────────────────┘     └─────────────────┘     └─────────────────┘
` + "```" + `

"Customer" means different things in different contexts — and that's OK! Each context has its own model.

**Context Mapping — How Contexts Relate:**

1. **Shared Kernel**: Two contexts share a common model (tight coupling, use sparingly)
2. **Customer-Supplier**: One context provides data to another (upstream/downstream)
3. **Conformist**: Downstream adopts upstream's model as-is (no translation)
4. **Anti-Corruption Layer (ACL)**: Translate between contexts to protect your model
5. **Open Host Service**: Context provides a well-defined API for others
6. **Published Language**: Shared data format (e.g., JSON schema, protobuf)

**Core, Supporting, and Generic Subdomains:**

Not all parts of your business are equally important:
- **Core Domain**: Your competitive advantage — invest the most here (e.g., Amazon's recommendation engine)
- **Supporting Subdomain**: Necessary but not differentiating (e.g., user management)
- **Generic Subdomain**: Off-the-shelf solutions work fine (e.g., email sending, payments)

**DDD is NOT for everything.** Use it for complex domains where the business logic is the hard part. For simple CRUD apps, DDD adds unnecessary complexity.`,
					CodeExamples: `// Bounded Contexts in Practice
// Each context has its own Customer model

// Sales Context
package sales

type Customer struct {
    ID          string
    Name        string
    Email       string
    Preferences []string
    Segment     string  // "enterprise", "startup", "individual"
}

// Shipping Context
package shipping

type Customer struct {
    ID             string
    Name           string
    Address        Address
    DeliveryNotes  string
    PreferredSlot  TimeSlot
}

// Billing Context
package billing

type Customer struct {
    ID            string
    Name          string
    PaymentMethod PaymentMethod
    Balance       Money
    CreditLimit   Money
}

// Anti-Corruption Layer: Translate between contexts
package shipping

type SalesCustomerAdapter struct {
    salesAPI SalesAPI
}

func (a *SalesCustomerAdapter) GetShippingCustomer(id string) (*Customer, error) {
    // Fetch from Sales context
    salesCustomer, err := a.salesAPI.GetCustomer(id)
    if err != nil {
        return nil, err
    }
    // Translate to Shipping context's model
    return &Customer{
        ID:   salesCustomer.ID,
        Name: salesCustomer.Name,
        // Address comes from a different source
    }, nil
}

// Context Map visualization:
//
// ┌──────────┐   ACL   ┌──────────┐  Open Host  ┌──────────┐
// │  Sales   │ ──────→ │ Shipping │ ←────────── │ Billing  │
// └──────────┘         └──────────┘             └──────────┘
//       ↓ Customer-Supplier
// ┌──────────┐
// │Analytics │
// └──────────┘`,
				},
				{
					Title: "DDD Tactical Design: Entities, Value Objects, Aggregates",
					Content: `Tactical design provides the building blocks for implementing domain models within a bounded context.

**Entities:**
Objects defined by their identity, not their attributes. Two entities with the same attributes but different IDs are different entities.

- **Identity matters**: A User is identified by their ID, not their name
- **Mutable**: Entities change over time (user updates email, order changes status)
- **Lifecycle**: Created, modified, archived, deleted
- **Equality**: Compared by ID, not by attributes

**Value Objects:**
Objects defined by their attributes, not by identity. Two value objects with the same attributes are considered equal.

- **No identity**: An Address of "123 Main St" is the same regardless of where it appears
- **Immutable**: Once created, never changed. Create a new one instead.
- **Equality**: Compared by all attributes
- **Side-effect free**: Methods return new value objects, don't modify state

**Aggregates:**
A cluster of entities and value objects treated as a single unit for data changes. The aggregate root is the only entry point.

` + "```" + `
┌─────────────────────────────────────┐
│          Order (Aggregate Root)      │
│                                      │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ OrderItem │  │ OrderItem        │ │
│  │ (Entity)  │  │ (Entity)         │ │
│  └──────────┘  └──────────────────┘ │
│                                      │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ Money    │  │ ShippingAddress  │ │
│  │ (Value)  │  │ (Value Object)   │ │
│  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────┘
` + "```" + `

**Aggregate Rules:**
1. **One root entity**: The aggregate root is the only entity external objects can reference
2. **Transactional consistency**: All changes within an aggregate are atomic
3. **Reference by ID**: Aggregates reference other aggregates by ID, not by object reference
4. **Small aggregates**: Keep them focused. A common mistake is making aggregates too large
5. **Eventual consistency between aggregates**: Use domain events for cross-aggregate communication

**Domain Services:**
Operations that don't naturally belong to any entity or value object. They represent domain concepts that involve multiple entities.

- **Stateless**: No internal state
- **Named after domain operations**: TransferMoney, CalculateShipping
- **Operate on domain objects**: Take entities/value objects as parameters

**Domain Events:**
Something that happened in the domain that domain experts care about. Used for communication between aggregates and bounded contexts.

- **Past tense**: OrderPlaced, PaymentReceived, InventoryReserved
- **Immutable**: Once published, never changed
- **Carry relevant data**: Include enough data for consumers to act

**Repository Pattern:**
Provides an illusion of an in-memory collection of domain objects. Repositories encapsulate persistence logic, keeping the domain model clean.`,
					CodeExamples: `// ENTITY — identified by ID
type Order struct {
    id        string         // Identity
    items     []OrderItem    // Child entities
    status    OrderStatus    // State
    total     Money          // Value object
    createdAt time.Time
}

// Business logic lives ON the entity
func (o *Order) AddItem(product Product, qty int) error {
    if o.status != OrderStatusDraft {
        return errors.New("can only add items to draft orders")
    }
    item := NewOrderItem(product, qty)
    o.items = append(o.items, item)
    o.recalculateTotal()
    return nil
}

func (o *Order) Submit() error {
    if len(o.items) == 0 {
        return errors.New("cannot submit empty order")
    }
    o.status = OrderStatusSubmitted
    return nil
}

// VALUE OBJECT — identified by attributes, immutable
type Money struct {
    amount   int64  // cents to avoid floating point
    currency string
}

func NewMoney(amount int64, currency string) Money {
    return Money{amount: amount, currency: currency}
}

func (m Money) Add(other Money) (Money, error) {
    if m.currency != other.currency {
        return Money{}, errors.New("currency mismatch")
    }
    return NewMoney(m.amount+other.amount, m.currency), nil
}

func (m Money) Equals(other Money) bool {
    return m.amount == other.amount && m.currency == other.currency
}

// AGGREGATE with repository
type OrderRepository interface {
    Save(order *Order) error
    FindByID(id string) (*Order, error)
    // No method to save individual OrderItems!
    // The aggregate root controls all persistence
}

// DOMAIN EVENT
type OrderPlaced struct {
    OrderID    string
    CustomerID string
    Total      Money
    OccurredAt time.Time
}

// DOMAIN SERVICE
type PricingService struct {
    discountRepo DiscountRepository
}

func (s *PricingService) CalculatePrice(items []OrderItem, customer Customer) (Money, error) {
    subtotal := NewMoney(0, "USD")
    for _, item := range items {
        lineTotal := item.UnitPrice().Multiply(int64(item.Quantity()))
        subtotal, _ = subtotal.Add(lineTotal)
    }
    discount, err := s.discountRepo.FindForCustomer(customer.ID())
    if err != nil {
        return Money{}, err
    }
    return subtotal.Apply(discount), nil
}`,
				},
				{
					Title: "Microservices Patterns",
					Content: `When building distributed systems with microservices, specific patterns help manage the inherent complexity of service-to-service communication, data consistency, and resilience.

**Circuit Breaker Pattern:**

Prevents cascading failures. When a service is failing, stop calling it and fail fast instead of waiting for timeouts.

` + "```" + `
States:
  CLOSED → (failures exceed threshold) → OPEN
  OPEN   → (timeout expires)           → HALF-OPEN
  HALF-OPEN → (test request succeeds) → CLOSED
  HALF-OPEN → (test request fails)    → OPEN
` + "```" + `

**Why:** If Service B is down and Service A keeps calling it, Service A's threads get blocked waiting for timeouts. Eventually Service A runs out of threads and becomes unresponsive too (cascading failure).

**Saga Pattern:**

Manages distributed transactions across multiple services without distributed locks.

` + "```" + `
Order Saga (Choreography):
  OrderService: CreateOrder
       ↓ OrderCreated event
  PaymentService: ProcessPayment
       ↓ PaymentProcessed event
  InventoryService: ReserveItems
       ↓ ItemsReserved event
  ShippingService: ScheduleShipment

Compensation (if payment fails):
  OrderService: CancelOrder ← PaymentFailed event
` + "```" + `

**Two Types:**
1. **Choreography**: Services listen for events and react (decoupled, but hard to track)
2. **Orchestration**: A central coordinator tells services what to do (easier to understand, single point of failure)

**API Gateway Pattern:**

A single entry point for all client requests. Routes requests to appropriate microservices.

**Benefits:**
- Client simplification (one endpoint, not dozens)
- Cross-cutting concerns (auth, rate limiting, logging)
- Protocol translation (REST → gRPC internally)
- Response aggregation (combine multiple service responses)

**Sidecar Pattern:**

Deploy supporting functionality as a separate process (sidecar) alongside the main application.

**Use Cases:**
- Service mesh proxies (Istio/Envoy)
- Log collection
- Configuration management
- TLS termination

**CQRS (Command Query Responsibility Segregation):**

Separate the read and write models of your application.

` + "```" + `
Write Side:                    Read Side:
Commands → Domain Model →     Projections → Read Model → Queries
              ↓
         Domain Events ──────────→ Update Read Model
` + "```" + `

**When to Use CQRS:**
- Read and write patterns are very different
- Need to scale reads and writes independently
- Complex domain with event sourcing
- Different optimization needs for reads vs writes

**Event Sourcing:**

Instead of storing current state, store a sequence of events that led to the current state.

**Benefits:**
- Complete audit trail
- Can reconstruct state at any point in time
- Natural fit with CQRS and message-driven systems
- Enables temporal queries ("what was the balance on March 1?")

**Drawbacks:**
- Complexity: replaying events, handling schema evolution
- Storage: event log grows over time (snapshotting helps)
- Querying: current state requires replaying events (or maintaining projections)`,
					CodeExamples: `// CIRCUIT BREAKER
type CircuitBreaker struct {
    mu           sync.Mutex
    state        string // "closed", "open", "half-open"
    failures     int
    threshold    int
    timeout      time.Duration
    lastFailure  time.Time
}

func NewCircuitBreaker(threshold int, timeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        state:     "closed",
        threshold: threshold,
        timeout:   timeout,
    }
}

func (cb *CircuitBreaker) Execute(fn func() error) error {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    switch cb.state {
    case "open":
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.state = "half-open"
        } else {
            return errors.New("circuit breaker is open")
        }
    }

    err := fn()
    if err != nil {
        cb.failures++
        cb.lastFailure = time.Now()
        if cb.failures >= cb.threshold {
            cb.state = "open"
        }
        return err
    }

    cb.failures = 0
    cb.state = "closed"
    return nil
}

// Usage
breaker := NewCircuitBreaker(5, 30*time.Second)
err := breaker.Execute(func() error {
    return callExternalService()
})

// ─────────────────────────────────────────

// SAGA PATTERN (Orchestration)
type OrderSaga struct {
    orderSvc     OrderService
    paymentSvc   PaymentService
    inventorySvc InventoryService
}

func (s *OrderSaga) Execute(order Order) error {
    // Step 1: Create order
    if err := s.orderSvc.Create(order); err != nil {
        return err
    }
    // Step 2: Process payment
    if err := s.paymentSvc.Charge(order.Total); err != nil {
        // Compensate: cancel order
        s.orderSvc.Cancel(order.ID)
        return fmt.Errorf("payment failed: %w", err)
    }
    // Step 3: Reserve inventory
    if err := s.inventorySvc.Reserve(order.Items); err != nil {
        // Compensate: refund + cancel
        s.paymentSvc.Refund(order.ID)
        s.orderSvc.Cancel(order.ID)
        return fmt.Errorf("inventory failed: %w", err)
    }
    return nil
}

// ─────────────────────────────────────────

// CQRS — Separate read and write models
// Write side (commands)
type CreateOrderCommand struct {
    CustomerID string
    Items      []OrderItemDTO
}

type OrderCommandHandler struct {
    repo     OrderRepository
    eventBus EventBus
}

func (h *OrderCommandHandler) Handle(cmd CreateOrderCommand) error {
    order := NewOrder(cmd.CustomerID, cmd.Items)
    if err := h.repo.Save(order); err != nil {
        return err
    }
    h.eventBus.Publish(OrderCreatedEvent{OrderID: order.ID})
    return nil
}

// Read side (queries)
type OrderReadModel struct {
    ID         string
    Customer   string
    ItemCount  int
    Total      string
    Status     string
}

type OrderQueryHandler struct {
    readDB *sql.DB  // Optimized read database
}

func (h *OrderQueryHandler) GetOrderSummary(id string) (*OrderReadModel, error) {
    // Query optimized read model (denormalized for fast reads)
    return &OrderReadModel{}, nil
}

// Event handler updates read model
func (h *OrderProjection) OnOrderCreated(event OrderCreatedEvent) {
    // Update denormalized read model
    h.readDB.Exec("INSERT INTO order_summaries ...", event.OrderID)
}`,
				},
			},
		},
	})
}
