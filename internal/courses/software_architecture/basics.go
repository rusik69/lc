package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          30,
			Title:       "Introduction to Software Architecture",
			Description: "Learn the fundamentals of software architecture, the role of architects, and key architectural concepts.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "What is Software Architecture?",
					Content: `Software architecture is the fundamental structure of a software system, the discipline of creating such structures, and the documentation of these structures.

**Definition:**
Software architecture represents the significant decisions about the organization of a software system. These decisions include:
- Selection of structural elements and their interfaces
- Behavior as collaboration among those elements
- Composition of structural and behavioral elements into larger subsystems
- Architectural style that guides this organization

**Why This Matters:**
Architecture decisions are hard to change once implemented. They form the foundation that all other development work builds upon. Making the right architectural decisions early can save significant time and cost, while poor decisions can lead to technical debt and system failures.

**Key Characteristics:**
- **Structure**: How components are organized and connected
- **Behavior**: How components interact and collaborate
- **Non-functional Requirements**: Performance, security, scalability, maintainability
- **Trade-offs**: Balancing competing concerns and constraints

**Why Architecture Matters:**
- **Foundation**: Provides the foundation for all development work
- **Communication**: Serves as a communication tool among stakeholders
- **Constraints**: Imposes constraints on implementation
- **Quality**: Directly impacts system quality attributes
- **Evolution**: Enables and constrains system evolution

**Architecture vs Design:**

Understanding the difference helps clarify responsibilities:

**Architecture (High-level):**
- System-wide structure and organization
- Significant decisions that are hard to change
- Focus on quality attributes and constraints
- Example: "System uses microservices architecture"

**Design (Detailed):**
- Component-level structure and implementation
- Decisions that are easier to change
- Focus on functionality and code organization
- Example: "UserService uses Spring Boot with dependency injection"

**Key Insight:** Architecture is a subset of design - the most important design decisions that have system-wide impact.

**Architectural Drivers:**

These are the forces that shape architectural decisions:

**Functional Requirements:**
- What the system must do
- Business capabilities and features
- User stories and use cases

**Quality Attributes:**
- Performance: Response time, throughput
- Security: Authentication, authorization, data protection
- Availability: Uptime, fault tolerance
- Usability: User experience, accessibility
- Scalability: Ability to handle growth
- Maintainability: Ease of modification and extension

**Constraints:**
- Technology: Existing infrastructure, vendor requirements
- Budget: Cost limitations
- Time: Deadlines and schedules
- Regulations: Compliance requirements (GDPR, HIPAA, etc.)
- Team: Skills and expertise available

**Stakeholder Concerns:**
- **Business**: ROI, time-to-market, competitive advantage
- **Users**: Performance, reliability, ease of use
- **Developers**: Maintainability, developer experience
- **Operations**: Deployability, monitoring, support

**Common Misconceptions:**

**Myth 1: Architecture is just diagrams**
- Reality: Architecture includes decisions, rationale, and trade-offs, not just visual representations

**Myth 2: Architecture is only about technology choices**
- Reality: Architecture encompasses structure, behavior, quality attributes, and business alignment

**Myth 3: Architecture is only for large systems**
- Reality: Even small systems benefit from architectural thinking, though the formality may differ

**Myth 4: Architecture is a one-time activity**
- Reality: Architecture evolves with the system and must be continuously evaluated and refined

**Myth 5: Architecture doesn't need to evolve**
- Reality: As requirements change, architecture must adapt to remain effective

**Key Takeaways:**
- Architecture represents significant, system-wide decisions
- It balances functional requirements, quality attributes, constraints, and stakeholder concerns
- Architecture is not just diagrams or technology choices
- It must evolve with the system to remain effective
- Good architecture enables future development and system evolution`,
					CodeExamples: `Architecture Decision Example:

Problem: Need to support 1M concurrent users
Architectural Decision: Use microservices with horizontal scaling
Rationale:
- Allows independent scaling of services
- Better fault isolation
- Enables team autonomy
Trade-offs:
- Increased complexity
- Network latency
- Distributed system challenges

Architecture vs Design - Visual Comparison:

Architecture Level (System-wide):
┌─────────────────────────────────────────┐
│         E-commerce System               │
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────┐  │
│  │   User   │  │ Product │  │Order │  │
│  │ Service  │  │ Service │  │Service│ │
│  └────┬─────┘  └────┬─────┘  └──┬───┘  │
│       │             │            │      │
│       └─────────────┴────────────┘      │
│                    │                    │
│              ┌─────▼─────┐             │
│              │   API      │             │
│              │  Gateway   │             │
│              └────────────┘             │
└─────────────────────────────────────────┘

Design Level (Component details):
┌─────────────────────────────────────────┐
│         User Service (Spring Boot)       │
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────┐  │
│  │Controller│→ │ Service │→ │ Repo │  │
│  └──────────┘  └──────────┘  └──┬───┘  │
│                                  │      │
│                            ┌─────▼─────┐│
│                            │PostgreSQL ││
│                            │Pool: 20   ││
│                            └───────────┘│
└─────────────────────────────────────────┘

Architecture vs Design Examples:

Architecture (High-level):
- System uses microservices architecture
- Services communicate via REST APIs
- Database is sharded by user ID
- Cache layer uses Redis
- API Gateway routes external requests

Design (Detailed):
- UserService uses Spring Boot framework
- Database connection pool size: 20
- Cache TTL: 3600 seconds
- API endpoint: /api/v1/users/{id}
- Uses JWT for authentication tokens
- Implements circuit breaker pattern

Architectural Drivers - Complete Example:

Functional Requirements:
- User registration and authentication
- Product catalog browsing
- Order processing
- Payment processing

Quality Attributes:
- Performance: < 200ms API response time
- Availability: 99.9% uptime (less than 8.76 hours downtime/year)
- Security: PCI-DSS compliance for payment data
- Scalability: Handle 10x traffic growth without redesign
- Maintainability: New features in < 2 weeks

Constraints:
- Must use existing AWS infrastructure
- Budget: $50K/month for infrastructure
- Launch deadline: 6 months
- Team size: 10 developers
- Must comply with GDPR regulations

Stakeholder Concerns:
- Business: Fast time-to-market, competitive advantage
- Users: Fast, reliable experience, data privacy
- Developers: Maintainable codebase, good tooling
- Operations: Easy to deploy and monitor, cost-effective

Decision Framework:

When making architectural decisions, consider:
1. Does it meet functional requirements?
2. Does it satisfy quality attributes?
3. Does it fit within constraints?
4. Does it address stakeholder concerns?
5. What are the trade-offs?
6. Can we evolve it over time?

Self-Assessment Questions:
1. What is the difference between architecture and design?
2. Why do architectural decisions matter more than design decisions?
3. Name three quality attributes that might conflict with each other.
4. How do constraints influence architectural decisions?
5. Give an example of a trade-off in architecture.`,
				},
				{
					Title: "Role of Software Architects",
					Content: `Software architects play a crucial role in software development, bridging the gap between business needs and technical implementation.

**Key Responsibilities:**

**1. Architecture Design:**
- Design system architecture that meets requirements
- Make architectural decisions and document them
- Evaluate and select technologies
- Define architectural patterns and standards

**2. Technical Leadership:**
- Guide development teams on architectural direction
- Review code and architecture for compliance
- Mentor developers on best practices
- Resolve technical conflicts and challenges

**3. Communication:**
- Communicate architecture to stakeholders
- Translate business requirements to technical solutions
- Present architecture to technical and non-technical audiences
- Document architecture decisions and rationale

**4. Risk Management:**
- Identify technical risks early
- Evaluate trade-offs and alternatives
- Plan for scalability and evolution
- Ensure architecture supports business goals

**5. Quality Assurance:**
- Ensure architecture meets quality attributes
- Define and enforce architectural standards
- Review system design for architectural compliance
- Monitor system health and performance

**Architect Types:**

**1. Solution Architect:**
- Focuses on specific solutions or projects
- Designs architecture for particular business problems
- Works closely with development teams
- Ensures solution fits enterprise architecture

**2. Enterprise Architect:**
- Focuses on organization-wide architecture
- Defines standards and guidelines
- Ensures consistency across projects
- Aligns IT strategy with business strategy

**3. Technical Architect:**
- Deep technical expertise in specific domains
- Focuses on technology selection and implementation
- Provides technical guidance to teams
- Evaluates new technologies and patterns

**4. Application Architect:**
- Focuses on application-level architecture
- Designs application structure and components
- Ensures application meets requirements
- Works within application boundaries

**Skills Required:**

**Technical Skills:**
- Deep understanding of software design patterns
- Knowledge of multiple programming languages and frameworks
- Understanding of system design and scalability
- Experience with cloud platforms and infrastructure
- Knowledge of databases, messaging, and integration patterns

**Soft Skills:**
- Strong communication and presentation skills
- Ability to make decisions with incomplete information
- Leadership and mentoring abilities
- Negotiation and conflict resolution
- Strategic thinking and planning

**Common Challenges:**

**1. Balancing Competing Requirements:**
- Performance vs. Cost: Faster systems often cost more
- Security vs. Usability: More security can reduce usability
- Flexibility vs. Simplicity: More flexible systems are often more complex
- **Solution**: Use trade-off analysis and prioritize based on business value

**2. Making Decisions with Limited Information:**
- Requirements may be incomplete or changing
- Future needs are uncertain
- Technology landscape evolves rapidly
- **Solution**: Make decisions that are reversible, document assumptions, plan for evolution

**3. Communicating Complex Technical Concepts:**
- Different stakeholders have different backgrounds
- Technical details can overwhelm non-technical audiences
- **Solution**: Use multiple communication styles (diagrams, code, prose), adapt to audience

**4. Keeping Up with Rapidly Changing Technology:**
- New frameworks and tools emerge constantly
- Best practices evolve
- **Solution**: Focus on principles over specific technologies, continuous learning

**5. Managing Technical Debt:**
- Short-term solutions accumulate over time
- Refactoring requires time and resources
- **Solution**: Balance speed with quality, allocate time for refactoring, document debt

**Key Takeaways:**
- Architects bridge business and technical worlds
- Different architect types focus on different scopes
- Both technical and soft skills are essential
- Effective communication is critical for success
- Architecture is a continuous activity, not a one-time task

**Self-Assessment Questions:**
1. What are the key differences between Solution Architect and Enterprise Architect roles?
2. Why is communication skill important for architects?
3. How do architects balance competing requirements?
4. What strategies help architects make decisions with limited information?
5. Give an example of how an architect might communicate the same concept differently to business vs. technical stakeholders.`,
					CodeExamples: `Architectural Decision Document (ADR) Example:

Title: Use Microservices Architecture
Status: Accepted
Context: Current monolithic application is difficult to scale and maintain.
Decision: Adopt microservices architecture with API Gateway pattern.
Consequences:
- Positive: Independent scaling, technology diversity, team autonomy
- Negative: Increased complexity, network latency, distributed system challenges
Alternatives Considered:
- Monolithic architecture (rejected - scaling issues)
- Modular monolith (rejected - still single deployment)

Architecture Review Checklist:

1. Does architecture meet functional requirements?
2. Does architecture meet quality attributes?
3. Are architectural patterns appropriate?
4. Is technology selection justified?
5. Are risks identified and mitigated?
6. Is architecture documented?
7. Can architecture evolve?
8. Is architecture testable?

Stakeholder Communication Example:

For Business Stakeholders:
- Focus on business value and outcomes
- Use business language, not technical jargon
- Show how architecture supports business goals
- Highlight risks and mitigation strategies

For Technical Teams:
- Provide detailed technical specifications
- Explain design decisions and trade-offs
- Show code examples and patterns
- Discuss implementation challenges

Architecture Documentation Structure:

1. System Overview
2. Architectural Drivers
3. Architectural Decisions
4. System Context Diagram
5. Container Diagram
6. Component Diagrams
7. Deployment Diagram
8. Quality Attributes
9. Risks and Mitigation`,
				},
				{
					Title: "Architecture Quality Attributes",
					Content: `Quality attributes (also called non-functional requirements) define how well a system performs its functions. They are critical architectural concerns.

**Performance:**
- **Response Time**: Time to complete a request
- **Throughput**: Number of requests processed per unit time
- **Resource Utilization**: Efficient use of CPU, memory, disk, network
- **Scalability**: Ability to handle increased load

**Reliability:**
- **Availability**: Percentage of time system is operational
- **Fault Tolerance**: Ability to continue operating despite failures
- **Recoverability**: Ability to recover from failures
- **Mean Time Between Failures (MTBF)**: Average time between failures

**Security:**
- **Confidentiality**: Protection from unauthorized access
- **Integrity**: Protection from unauthorized modification
- **Authentication**: Verifying user identity
- **Authorization**: Controlling access to resources
- **Non-repudiation**: Preventing denial of actions

**Usability:**
- **Learnability**: Easy to learn
- **Efficiency**: Efficient to use
- **Memorability**: Easy to remember
- **Error Prevention**: Prevents user errors
- **Satisfaction**: Pleasant to use

**Maintainability:**
- **Modifiability**: Easy to modify
- **Testability**: Easy to test
- **Analyzability**: Easy to understand
- **Reusability**: Components can be reused
- **Portability**: Easy to move to different environments

**Scalability:**
- **Horizontal Scaling**: Adding more machines (scale out)
- **Vertical Scaling**: Adding more resources to existing machines (scale up)
- **Elasticity**: Ability to scale up and down based on demand

**Why Quality Attributes Matter:**

Quality attributes directly impact user experience, business outcomes, and system success. They often conflict with each other, requiring careful trade-offs:

**Common Conflicts:**
- **Performance vs. Cost**: Faster systems require more resources
- **Security vs. Usability**: More security measures can reduce ease of use
- **Flexibility vs. Simplicity**: More flexible systems are often more complex
- **Scalability vs. Consistency**: Distributed systems face consistency challenges

**Real-World Examples:**

**Performance Example - Instagram Feed:**
- **Challenge**: Load millions of photos quickly
- **Solution**: Pre-compute feed, use CDN, lazy loading
- **Result**: Feed loads in < 1 second for most users

**Reliability Example - AWS S3:**
- **Challenge**: 99.999999999% (11 nines) durability
- **Solution**: Data replication across multiple availability zones
- **Result**: Virtually no data loss

**Security Example - Banking System:**
- **Challenge**: Protect sensitive financial data
- **Solution**: Multi-factor authentication, encryption at rest and in transit, audit logs
- **Result**: Meets regulatory requirements, protects customer data

**Scalability Example - Netflix:**
- **Challenge**: Handle peak traffic (millions of concurrent streams)
- **Solution**: Microservices, auto-scaling, content delivery network
- **Result**: Scales from millions to billions of requests seamlessly

**Quality Attribute Trade-offs:**

When designing architecture, you must make trade-offs. Here's a decision framework:

**Scenario: E-commerce Platform**

**Option 1: Prioritize Performance**
- Use in-memory cache extensively
- Pre-compute expensive queries
- **Trade-off**: Higher memory costs, potential data staleness

**Option 2: Prioritize Cost**
- Use slower, cheaper storage
- Batch processing instead of real-time
- **Trade-off**: Slower response times, reduced user experience

**Option 3: Balanced Approach**
- Cache frequently accessed data
- Use cost-effective storage for less critical data
- **Trade-off**: More complex architecture, requires careful design

**Key Takeaways:**
- Quality attributes define how well a system performs
- They often conflict, requiring trade-offs
- Prioritize based on business value and user needs
- Measure and monitor quality attributes continuously
- Architecture decisions directly impact quality attributes

**Self-Assessment Questions:**
1. What is the difference between horizontal and vertical scaling?
2. Give an example of how performance and cost conflict in architecture.
3. How does security impact usability? Provide a real-world example.
4. What quality attributes are most important for a banking system? Why?
5. Describe a scenario where you would prioritize scalability over consistency.`,
					CodeExamples: `Performance Requirements Example:

Response Time:
- API endpoints: < 200ms (P95)
- Database queries: < 100ms (P95)
- Page load: < 3 seconds

Throughput:
- API: 10,000 requests/second
- Database: 5,000 queries/second
- Message processing: 50,000 messages/second

Resource Utilization:
- CPU: < 70% average
- Memory: < 80% average
- Disk I/O: < 60% average
- Network: < 50% average

Reliability Requirements Example:

Availability:
- Target: 99.9% (three nines)
- Downtime: < 8.76 hours/year
- Planned maintenance: < 4 hours/month

Fault Tolerance:
- System continues operating with single component failure
- Automatic failover within 30 seconds
- Data replication across 3 data centers

Recoverability:
- Recovery Time Objective (RTO): 1 hour
- Recovery Point Objective (RPO): 15 minutes
- Automated backup every 6 hours

Security Requirements Example:

Authentication:
- Multi-factor authentication (MFA) required
- Session timeout: 30 minutes inactivity
- Password complexity requirements

Authorization:
- Role-based access control (RBAC)
- Principle of least privilege
- Audit logging for all access

Data Protection:
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- PII data masking in logs

Maintainability Metrics:

Code Quality:
- Cyclomatic complexity: < 10 per function
- Code coverage: > 80%
- Technical debt ratio: < 5%

Documentation:
- API documentation: 100% coverage
- Architecture documentation: Up to date
- Code comments: For complex logic

Modifiability:
- Feature addition: < 2 days average
- Bug fix: < 4 hours average
- Refactoring: Safe and easy

Quality Attribute Trade-offs:

Example 1: Performance vs Security
- More encryption → Slower performance
- More validation → Higher latency
- Solution: Use caching, optimize algorithms

Example 2: Scalability vs Consistency
- Strong consistency → Lower scalability
- Eventual consistency → Higher scalability
- Solution: Choose based on use case

Example 3: Usability vs Security
- More security steps → Lower usability
- Simpler UX → Potential security risks
- Solution: Balance with smart defaults`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          31,
			Title:       "Architectural Patterns",
			Description: "Explore fundamental architectural patterns including Layered Architecture, MVC, MVP, MVVM, and other structural patterns.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Layered Architecture",
					Content: `Layered Architecture (also called N-tier architecture) organizes software into horizontal layers, each with a specific responsibility.

**Key Characteristics:**
- **Separation of Concerns**: Each layer has distinct responsibilities
- **Dependency Direction**: Layers depend only on layers below
- **Abstraction**: Each layer provides abstraction for layers above
- **Isolation**: Changes in one layer don't affect others directly

**Common Layers:**

**1. Presentation Layer:**
- User interface components
- Handles user input and output
- Formats data for display
- Examples: Web pages, mobile apps, desktop UIs

**2. Application/Business Logic Layer:**
- Core business logic
- Orchestrates workflow
- Validates business rules
- Coordinates between layers

**3. Domain/Service Layer:**
- Domain models and business entities
- Business rules and logic
- Domain services
- Core business concepts

**4. Data Access Layer:**
- Database access logic
- Data persistence
- Data mapping (ORM)
- Query optimization

**5. Infrastructure Layer:**
- External services integration
- File system access
- Network communication
- Cross-cutting concerns (logging, caching)

**Benefits:**
- Clear separation of concerns
- Easy to understand and maintain
- Testable (can test layers independently)
- Reusable components
- Standard structure familiar to developers

**Drawbacks:**
- Can lead to over-engineering
- Performance overhead (layer traversal)
- Can create unnecessary abstractions
- May not fit all use cases

**When to Use:**
- Traditional business applications
- CRUD applications
- Applications with clear layer boundaries
- Teams familiar with layered approach

**When NOT to Use:**
- Simple applications where layers add unnecessary complexity
- Performance-critical applications where layer overhead matters
- Applications requiring complex cross-layer communication
- Microservices (each service may use layers internally, but services themselves aren't layers)

**Common Pitfalls:**
1. **Anemic Domain Model**: Business logic moves to service layer, domain objects become data containers
   - **Solution**: Keep business logic in domain layer where it belongs
2. **Layer Skipping**: Presentation layer directly accessing data access layer
   - **Solution**: Enforce dependency rule, use dependency injection
3. **Over-Engineering**: Creating too many layers for simple applications
   - **Solution**: Start simple, add layers only when needed
4. **Tight Coupling**: Layers know too much about other layers' internals
   - **Solution**: Use interfaces, dependency inversion

**Real-World Example - E-commerce Application:**

**Presentation Layer:**
- Web controllers handle HTTP requests
- Mobile app views display product information
- Admin dashboard for managing orders

**Application Layer:**
- OrderService orchestrates order creation
- PaymentService handles payment processing
- InventoryService manages stock levels

**Domain Layer:**
- Order entity with business rules (minimum order amount, validation)
- Product entity with pricing logic
- Customer entity with account management

**Data Access Layer:**
- OrderRepository persists orders
- ProductRepository queries product catalog
- CustomerRepository manages customer data

**Infrastructure Layer:**
- PostgreSQL database
- Redis cache
- Payment gateway integration
- Email service integration

**Key Takeaways:**
- Layers provide clear separation of concerns
- Dependencies flow downward only
- Each layer has a specific responsibility
- Changes in one layer don't cascade to others
- Testing is easier with isolated layers

**Self-Assessment Questions:**
1. Why should layers only depend on layers below them?
2. What happens if the presentation layer directly accesses the data access layer?
3. Give an example of when layered architecture might be overkill.
4. How does layered architecture help with testing?
5. What is the "anemic domain model" anti-pattern and how do you avoid it?`,
					CodeExamples: `Layered Architecture Example:

┌─────────────────────────────────┐
│   Presentation Layer            │
│   (Controllers, Views, UI)       │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│   Application Layer             │
│   (Use Cases, Services)         │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│   Domain Layer                  │
│   (Entities, Business Logic)    │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│   Data Access Layer             │
│   (Repositories, ORM)           │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│   Infrastructure Layer           │
│   (Database, External APIs)     │
└─────────────────────────────────┘

Example Implementation:

// Presentation Layer
class UserController {
    private UserService userService;
    
    public UserDTO getUser(Long id) {
        User user = userService.getUser(id);
        return mapToDTO(user);
    }
}

// Application Layer
class UserService {
    private UserRepository userRepository;
    
    public User getUser(Long id) {
        return userRepository.findById(id);
    }
}

// Domain Layer
class User {
    private Long id;
    private String email;
    private String name;
    
    public void changeEmail(String newEmail) {
        validateEmail(newEmail);
        this.email = newEmail;
    }
}

// Data Access Layer
class UserRepository {
    public User findById(Long id) {
        // Database query
    }
}

Dependency Rule:
- Upper layers depend on lower layers
- Lower layers don't depend on upper layers
- Domain layer has no dependencies`,
				},
				{
					Title: "MVC, MVP, and MVVM Patterns",
					Content: `These patterns separate presentation logic from business logic, improving maintainability and testability.

**MVC (Model-View-Controller):**

**Components:**
- **Model**: Business logic and data
- **View**: User interface (presentation)
- **Controller**: Handles user input, updates model, selects view

**Flow:**
1. User interacts with View
2. View forwards input to Controller
3. Controller updates Model
4. Model notifies View (Observer pattern)
5. View updates display

**Benefits:**
- Separation of concerns
- Multiple views for same model
- Testable components
- Widely understood pattern

**Use Cases:**
- Web applications (Spring MVC, Ruby on Rails)
- Desktop applications
- Mobile applications

**MVP (Model-View-Presenter):**

**Components:**
- **Model**: Business logic and data
- **View**: Passive UI (no business logic)
- **Presenter**: Contains presentation logic, updates view

**Key Difference from MVC:**
- View is passive (doesn't observe model)
- Presenter updates view directly
- No direct communication between View and Model

**Benefits:**
- Better testability (presenter can be tested without UI)
- Clear separation of presentation logic
- View is completely passive

**Use Cases:**
- Desktop applications
- Android applications
- Applications requiring extensive unit testing

**MVVM (Model-View-ViewModel):**

**Components:**
- **Model**: Business logic and data
- **View**: User interface
- **ViewModel**: Presentation logic, data binding

**Key Features:**
- Data binding between View and ViewModel
- ViewModel exposes data and commands
- View automatically updates when ViewModel changes
- Two-way data binding

**Benefits:**
- Automatic UI updates
- Less code (no manual view updates)
- Better separation of concerns
- Testable ViewModel

**Use Cases:**
- WPF applications (Windows)
- Angular applications
- Vue.js applications
- Xamarin applications

**Pattern Comparison:**

**When to Choose MVC:**
- Web applications with server-side rendering
- Applications where view needs to observe model changes
- Teams familiar with traditional MVC frameworks
- Example: Spring MVC, Ruby on Rails

**When to Choose MVP:**
- Applications requiring extensive unit testing
- Desktop applications with complex UI logic
- When you need complete control over view updates
- Example: Android applications, Swing applications

**When to Choose MVVM:**
- Applications with rich data binding capabilities
- Modern web frameworks (Angular, Vue)
- When automatic UI updates are desired
- Example: Angular, Vue.js, WPF

**Visual Comparison:**

MVC Flow:
┌────────┐    ┌──────────┐    ┌──────┐
│  View │◄───│Controller│───►│Model│
└────┬───┘    └──────────┘    └───┬──┘
     │                            │
     └──────────Observes──────────┘

MVP Flow:
┌────────┐    ┌──────────┐    ┌──────┐
│  View │◄───│Presenter │───►│Model│
└────────┘    └──────────┘    └──────┘
(passive)     (updates view)

MVVM Flow:
┌────────┐    ┌────────────┐    ┌──────┐
│  View │◄──►│ ViewModel  │───►│Model│
└────────┘    └────────────┘    └──────┘
(data binding) (exposes data)

**Key Takeaways:**
- All three patterns separate presentation from business logic
- MVC: View observes Model, Controller coordinates
- MVP: View is passive, Presenter handles all presentation logic
- MVVM: Data binding connects View and ViewModel automatically
- Choose based on framework, testing needs, and team experience

**Self-Assessment Questions:**
1. What is the key difference between MVC and MVP?
2. Why is MVP considered more testable than MVC?
3. How does data binding work in MVVM?
4. When would you choose MVVM over MVP?
5. Explain the flow of data in each pattern.`,
					CodeExamples: `MVC Example:

// Model
class User {
    private String name;
    private String email;
    
    public void setName(String name) {
        this.name = name;
        notifyObservers();
    }
}

// View
class UserView {
    public void displayUser(String name, String email) {
        System.out.println("Name: " + name);
        System.out.println("Email: " + email);
    }
}

// Controller
class UserController {
    private User model;
    private UserView view;
    
    public void updateUser(String name) {
        model.setName(name);
        view.displayUser(model.getName(), model.getEmail());
    }
}

MVP Example:

// Model
class User {
    private String name;
    private String email;
}

// View Interface
interface IUserView {
    void setName(String name);
    void setEmail(String email);
}

// Presenter
class UserPresenter {
    private User model;
    private IUserView view;
    
    public void loadUser() {
        view.setName(model.getName());
        view.setEmail(model.getEmail());
    }
    
    public void updateName(String name) {
        model.setName(name);
        view.setName(name);
    }
}

MVVM Example (Angular):

// Model
export class User {
    name: string;
    email: string;
}

// ViewModel
export class UserViewModel {
    user: User;
    
    constructor() {
        this.user = new User();
    }
    
    updateName(name: string) {
        this.user.name = name;
    }
}

// View (HTML with data binding)
<div>
    <input [(ngModel)]="viewModel.user.name" />
    <p>{{viewModel.user.name}}</p>
</div>

Comparison:

MVC:
- View observes Model
- Controller handles input
- Model notifies View

MVP:
- View is passive
- Presenter updates View
- No View-Model communication

MVVM:
- Data binding
- ViewModel exposes data
- Automatic updates`,
				},
				{
					Title: "Other Architectural Patterns",
					Content: `Beyond layered and MVC patterns, there are many other architectural patterns for different scenarios.

**Client-Server Architecture:**
- **Client**: Requests services
- **Server**: Provides services
- **Communication**: Network protocols (HTTP, TCP/IP)
- **Benefits**: Centralized data, easier maintenance
- **Use Cases**: Web applications, email systems, file sharing

**Peer-to-Peer (P2P) Architecture:**
- **Nodes**: Act as both client and server
- **Decentralized**: No central server
- **Benefits**: Scalability, fault tolerance, no single point of failure
- **Challenges**: Security, coordination, data consistency
- **Use Cases**: File sharing (BitTorrent), blockchain, distributed computing

**Pipe-and-Filter Architecture:**
- **Pipes**: Connect filters (data flow)
- **Filters**: Process data (transform, validate, enrich)
- **Benefits**: Modularity, reusability, parallel processing
- **Use Cases**: Data processing pipelines, ETL systems, compilers

**Event-Driven Architecture:**
- **Events**: Significant occurrences
- **Event Producers**: Generate events
- **Event Consumers**: React to events
- **Event Bus**: Routes events
- **Benefits**: Loose coupling, scalability, responsiveness
- **Use Cases**: Real-time systems, IoT, microservices

**Microkernel Architecture:**
- **Core System**: Minimal functionality
- **Plugins**: Extended functionality
- **Benefits**: Flexibility, extensibility, small core
- **Use Cases**: Operating systems, IDEs, web browsers

**Space-Based Architecture:**
- **Tuple Space**: Shared memory space
- **Processing Units**: Read/write to tuple space
- **Benefits**: Scalability, fault tolerance
- **Use Cases**: High-performance computing, distributed caching

**Service-Oriented Architecture (SOA):**
- **Services**: Self-contained business functions
- **Service Bus**: Communication infrastructure
- **Benefits**: Reusability, interoperability, loose coupling
- **Use Cases**: Enterprise applications, legacy integration

**Pattern Selection Guide:**

**Choose Client-Server When:**
- You need centralized data management
- Security and control are important
- Standard web/mobile applications
- Example: E-commerce website, banking app

**Choose P2P When:**
- You need high scalability without central infrastructure
- Fault tolerance is critical
- Decentralization is a requirement
- Example: Blockchain, distributed file sharing

**Choose Pipe-and-Filter When:**
- Data processing pipeline
- Need to process data in stages
- Want to parallelize processing
- Example: ETL systems, image processing pipelines

**Choose Event-Driven When:**
- Components need loose coupling
- Real-time responsiveness required
- Asynchronous processing needed
- Example: Microservices, IoT systems, real-time dashboards

**Choose Microkernel When:**
- Need extensibility through plugins
- Want to keep core small and stable
- Different features for different users
- Example: IDEs (VS Code, IntelliJ), web browsers

**Visual Diagrams:**

Client-Server Architecture:
┌──────────┐         ┌──────────┐
│  Client  │────────►│  Server  │
│  (Web)   │◄────────│  (API)   │
└──────────┘         └──────────┘

Peer-to-Peer Architecture:
     ┌─────┐
     │Node1│
     └──┬──┘
        │
  ┌─────┼─────┐
  │     │     │
┌─▼─┐ ┌─▼─┐ ┌─▼─┐
│N2 │ │N3 │ │N4 │
└───┘ └───┘ └───┘
  │     │     │
  └─────┴─────┘

Pipe-and-Filter Architecture:
Input → [Filter1] → [Filter2] → [Filter3] → Output
         (validate)  (transform)  (enrich)

Event-Driven Architecture:
┌──────────┐
│Producer1 │──┐
└──────────┘  │
              ▼
┌──────────┐  │   ┌──────────┐
│Producer2 │──┼──►│Event Bus │──►┌──────────┐
└──────────┘  │   └──────────┘   │Consumer1 │
              │                  └──────────┘
┌──────────┐  │                  ┌──────────┐
│Producer3 │──┘                  │Consumer2 │
└──────────┘                      └──────────┘

**Key Takeaways:**
- Different patterns solve different problems
- Pattern selection depends on requirements and constraints
- Patterns can be combined (e.g., microservices with event-driven)
- Understanding trade-offs helps choose the right pattern
- Real-world systems often use multiple patterns

**Self-Assessment Questions:**
1. When would you choose P2P over Client-Server architecture?
2. How does Event-Driven Architecture achieve loose coupling?
3. What are the benefits of Pipe-and-Filter architecture?
4. Give an example of a system that uses Microkernel architecture.
5. How do these patterns differ from Layered Architecture?`,
					CodeExamples: `Client-Server Example:

Client:
class WebClient {
    public void requestData() {
        HttpClient client = new HttpClient();
        String response = client.get("http://server.com/api/data");
        processResponse(response);
    }
}

Server:
class WebServer {
    @GetMapping("/api/data")
    public Data getData() {
        return dataService.getData();
    }
}

P2P Example:

class PeerNode {
    private List<PeerNode> neighbors;
    
    public void shareFile(File file) {
        for (PeerNode neighbor : neighbors) {
            neighbor.receiveFile(file);
        }
    }
    
    public void receiveFile(File file) {
        // Store file
    }
}

Pipe-and-Filter Example:

// Filter interface
interface Filter {
    Data process(Data input);
}

// Pipeline
class Pipeline {
    private List<Filter> filters;
    
    public Data execute(Data input) {
        Data result = input;
        for (Filter filter : filters) {
            result = filter.process(result);
        }
        return result;
    }
}

// Filters
class ValidationFilter implements Filter {
    public Data process(Data input) {
        // Validate data
        return input;
    }
}

class TransformationFilter implements Filter {
    public Data process(Data input) {
        // Transform data
        return transformed;
    }
}

Event-Driven Example:

// Event
class UserCreatedEvent {
    private Long userId;
    private String email;
}

// Event Producer
class UserService {
    private EventBus eventBus;
    
    public void createUser(User user) {
        // Create user
        eventBus.publish(new UserCreatedEvent(user.getId(), user.getEmail()));
    }
}

// Event Consumer
class EmailService {
    @EventListener
    public void handleUserCreated(UserCreatedEvent event) {
        sendWelcomeEmail(event.getEmail());
    }
}

Microkernel Example:

// Core System
class CoreSystem {
    private Map<String, Plugin> plugins;
    
    public void registerPlugin(String name, Plugin plugin) {
        plugins.put(name, plugin);
    }
    
    public void execute(String pluginName) {
        Plugin plugin = plugins.get(pluginName);
        plugin.execute();
    }
}

// Plugin Interface
interface Plugin {
    void execute();
}

// Plugin Implementation
class AuthenticationPlugin implements Plugin {
    public void execute() {
        // Authentication logic
    }
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          32,
			Title:       "Design Principles",
			Description: "Master fundamental design principles including SOLID, DRY, KISS, YAGNI, and composition over inheritance.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "SOLID Principles",
					Content: `SOLID is an acronym for five object-oriented design principles that make software more maintainable, flexible, and understandable.

**S - Single Responsibility Principle (SRP):**
A class should have only one reason to change. It should have only one responsibility.

**Benefits:**
- Easier to understand
- Easier to test
- Easier to maintain
- Reduced coupling

**Example:**
Bad: A class that handles user data AND sends emails
Good: Separate User class and EmailService class

**O - Open/Closed Principle (OCP):**
Software entities should be open for extension but closed for modification.

**Benefits:**
- Stable core code
- Easy to add features
- Reduced risk of breaking changes
- Better testability

**Example:**
Bad: Modify existing class to add new payment method
Good: Use interface/strategy pattern to add new payment methods

**L - Liskov Substitution Principle (LSP):**
Objects of a superclass should be replaceable with objects of its subclasses without breaking the application.

**Benefits:**
- Correct inheritance hierarchies
- Polymorphism works correctly
- Fewer bugs
- Better code reuse

**Example:**
Bad: Square extends Rectangle but changes behavior
Good: Both implement Shape interface correctly

**I - Interface Segregation Principle (ISP):**
Clients should not be forced to depend on interfaces they don't use.

**Benefits:**
- Smaller, focused interfaces
- Less coupling
- Easier to implement
- Better cohesion

**Example:**
Bad: Large interface with many methods
Good: Multiple small, specific interfaces

**D - Dependency Inversion Principle (DIP):**
High-level modules should not depend on low-level modules. Both should depend on abstractions.

**Benefits:**
- Loose coupling
- Easier testing (mock dependencies)
- Flexible design
- Better maintainability

**Example:**
Bad: High-level class directly depends on concrete database class
Good: Both depend on database interface

**SOLID Principles in Practice:**

**Real-World Example - E-commerce Order Processing:**

**Without SOLID:**
- Order class handles validation, persistence, email, and payment
- Adding new payment method requires modifying Order class
- Hard to test, tightly coupled

**With SOLID:**
- Order class: Single responsibility (order data)
- OrderValidator: Validates orders (SRP)
- OrderRepository: Persists orders (SRP)
- PaymentProcessor interface: Payment abstraction (DIP, OCP)
- CreditCardPayment, PayPalPayment: Implementations (OCP, LSP)
- EmailService: Sends emails (SRP, ISP)

**Benefits Realized:**
- Easy to add new payment methods without changing existing code
- Can test each component independently
- Changes in one area don't affect others
- Clear separation of concerns

**Common Violations and Fixes:**

**SRP Violation:**
Bad: User class does too much
class User {
    void save() { /* DB */ }
    void sendEmail() { /* Email */ }
    void validate() { /* Validation */ }
}

Good: Separate concerns
class User { /* Data only */ }
class UserRepository { void save(User u) {} }
class EmailService { void send(User u) {} }
class UserValidator { bool validate(User u) {} }

**OCP Violation:**
Bad: Must modify to extend
class ReportGenerator {
    void generate(String type) {
        if (type == "PDF") { /* PDF */ }
        // Adding Excel requires modification
    }
}

Good: Open for extension
interface ReportGenerator {
    void generate();
}
class PDFReport implements ReportGenerator {}

**LSP Violation:**
Bad: Square breaks Rectangle contract
class Rectangle {
    void setWidth(int w) { width = w; }
    void setHeight(int h) { height = h; }
}
class Square extends Rectangle {
    void setWidth(int w) { width = height = w; } // Breaks contract!
}

Good: Both implement Shape correctly
interface Shape {
    int getArea();
}
class Rectangle implements Shape {}
class Square implements Shape {}

**ISP Violation:**
// Bad: Fat interface
interface Worker {
    void work();
    void eat();
    void sleep();
}
class Robot implements Worker {
    void eat() {} // Forced to implement unused method
    void sleep() {}
}

// Good: Segregated interfaces
interface Workable { void work(); }
interface Eatable { void eat(); }
interface Sleepable { void sleep(); }
class Human implements Workable, Eatable, Sleepable {}
class Robot implements Workable {}


**DIP Violation:**

// Bad: High-level depends on low-level
class OrderService {
    private MySQLDatabase db; // Concrete dependency
    void saveOrder(Order o) {
        db.save(o); // Tight coupling
    }
}

// Good: Both depend on abstraction
interface Database {
    void save(Object o);
}
class OrderService {
    private Database db; // Abstraction
    void saveOrder(Order o) {
        db.save(o); // Loose coupling
    }
}

**Key Takeaways:**
- SOLID principles guide good object-oriented design
- They work together to create maintainable, flexible code
- Violations lead to technical debt and maintenance issues
- Apply principles pragmatically, not dogmatically
- Refactor gradually to improve code quality

**Self-Assessment Questions:**
1. How does SRP help with testing and maintenance?
2. Explain how OCP enables adding features without modifying existing code.
3. Why does the Square-Rectangle example violate LSP?
4. How does ISP reduce coupling between components?
5. Give an example of how DIP enables easier testing.`,
					CodeExamples: `Single Responsibility Principle:

// Bad: Multiple responsibilities
class User {
    private String name;
    private String email;
    
    public void save() {
        // Database logic
    }
    
    public void sendEmail() {
        // Email logic
    }
}

// Good: Single responsibility
class User {
    private String name;
    private String email;
}

class UserRepository {
    public void save(User user) {
        // Database logic
    }
}

class EmailService {
    public void sendEmail(User user) {
        // Email logic
    }
}

Open/Closed Principle:

// Bad: Modify existing code
class PaymentProcessor {
    public void process(String type, double amount) {
        if (type.equals("credit")) {
            // Credit card logic
        } else if (type.equals("paypal")) {
            // PayPal logic - MODIFICATION REQUIRED
        }
    }
}

// Good: Extend without modification
interface PaymentMethod {
    void process(double amount);
}

class CreditCardPayment implements PaymentMethod {
    public void process(double amount) {
        // Credit card logic
    }
}

class PayPalPayment implements PaymentMethod {
    public void process(double amount) {
        // PayPal logic - EXTENSION
    }
}

Liskov Substitution Principle:

// Bad: Violates LSP
class Rectangle {
    protected int width;
    protected int height;
    
    public void setWidth(int w) { width = w; }
    public void setHeight(int h) { height = h; }
}

class Square extends Rectangle {
    public void setWidth(int w) {
        width = w;
        height = w; // Changes behavior!
    }
}

// Good: Correct substitution
interface Shape {
    int getArea();
}

class Rectangle implements Shape {
    private int width;
    private int height;
    public int getArea() { return width * height; }
}

class Square implements Shape {
    private int side;
    public int getArea() { return side * side; }
}

Interface Segregation Principle:

// Bad: Large interface
interface Worker {
    void work();
    void eat();
    void sleep();
}

class Human implements Worker {
    public void work() { }
    public void eat() { }
    public void sleep() { }
}

class Robot implements Worker {
    public void work() { }
    public void eat() { } // Robots don't eat!
    public void sleep() { } // Robots don't sleep!
}

// Good: Segregated interfaces
interface Workable {
    void work();
}

interface Eatable {
    void eat();
}

interface Sleepable {
    void sleep();
}

class Human implements Workable, Eatable, Sleepable {
    // Implements all
}

class Robot implements Workable {
    // Only implements work
}

Dependency Inversion Principle:

// Bad: High-level depends on low-level
class UserService {
    private MySQLDatabase database; // Concrete dependency
    
    public User getUser(Long id) {
        return database.query("SELECT * FROM users WHERE id = ?", id);
    }
}

// Good: Both depend on abstraction
interface Database {
    User query(String sql, Object... params);
}

class UserService {
    private Database database; // Abstract dependency
    
    public User getUser(Long id) {
        return database.query("SELECT * FROM users WHERE id = ?", id);
    }
}

class MySQLDatabase implements Database {
    public User query(String sql, Object... params) {
        // Implementation
    }
}`,
				},
				{
					Title: "DRY, KISS, and YAGNI",
					Content: `These principles focus on simplicity and avoiding unnecessary complexity.

**DRY - Don't Repeat Yourself:**
Every piece of knowledge should have a single, unambiguous representation within a system.

**Benefits:**
- Less code to maintain
- Single source of truth
- Easier to update
- Reduced bugs

**When to Apply:**
- Duplicate code logic
- Repeated business rules
- Similar data structures
- Common functionality

**When NOT to Apply:**
- Don't over-abstract
- Some duplication is acceptable
- Don't create artificial abstractions
- Performance-critical code

**KISS - Keep It Simple, Stupid:**
Simplicity should be a key goal in design. Avoid unnecessary complexity.

**Benefits:**
- Easier to understand
- Easier to maintain
- Fewer bugs
- Faster development

**Principles:**
- Prefer simple solutions
- Avoid over-engineering
- Don't add features "just in case"
- Use familiar patterns

**YAGNI - You Aren't Gonna Need It:**
Don't implement functionality until it's actually needed.

**Benefits:**
- Faster delivery
- Less code to maintain
- Focus on current needs
- Avoid wasted effort

**When to Apply:**
- Don't add features prematurely
- Don't create abstractions "just in case"
- Don't optimize prematurely
- Focus on current requirements

**When NOT to Apply:**
- When requirements are clear and certain
- When it's part of the core architecture
- When it's a standard practice (logging, error handling)
- When it prevents significant future problems

**Balancing the Principles:**

These principles can sometimes conflict. Here's how to balance them:

**DRY vs YAGNI:**
- **DRY says**: Extract common code
- **YAGNI says**: Don't abstract until needed
- **Balance**: Extract duplication when it appears 3+ times, but only if you're certain it's needed

**KISS vs DRY:**
- **KISS says**: Keep it simple
- **DRY says**: Don't repeat yourself
- **Balance**: Extract common code, but don't create overly complex abstractions

**Real-World Examples:**

**DRY Example - Validation Logic:**

// Bad: Repeated validation
class UserService {
    void createUser(User u) {
        if (u.getEmail() == null || !u.getEmail().contains("@")) {
            throw new ValidationException();
        }
        // Create user
    }
    void updateUser(User u) {
        if (u.getEmail() == null || !u.getEmail().contains("@")) {
            throw new ValidationException(); // DUPLICATE
        }
        // Update user
    }
}

// Good: Single source of truth
class EmailValidator {
    void validate(String email) {
        if (email == null || !email.contains("@")) {
            throw new ValidationException();
        }
    }
}


**KISS Example - Over-Engineering:**

// Bad: Unnecessary complexity
class SimpleCalculator {
    private StrategyFactory factory;
    private OperationRegistry registry;
    public int add(int a, int b) {
        Operation op = factory.create(registry.get("add"));
        return op.execute(a, b);
    }
}

// Good: Simple and clear
class SimpleCalculator {
    public int add(int a, int b) {
        return a + b;
    }
}


**YAGNI Example - Premature Abstraction:**

// Bad: Abstracting before need
interface PaymentProcessor {
    void processPayment();
}
class CreditCardProcessor implements PaymentProcessor {}
class PayPalProcessor implements PaymentProcessor {}
class BitcoinProcessor implements PaymentProcessor {} // Not needed yet!

// Good: Add when needed
class PaymentProcessor {
    void processCreditCard() { /* Current need */ }
    // Add PayPal when requirement comes
}


**Key Takeaways:**
- DRY: Eliminate duplication, but don't over-abstract
- KISS: Prefer simple solutions over complex ones
- YAGNI: Don't build what you don't need yet
- Balance principles based on context and requirements
- Apply pragmatically, not dogmatically

**Self-Assessment Questions:**
1. When is it acceptable to have code duplication?
2. How do you balance DRY and YAGNI principles?
3. Give an example of over-engineering that violates KISS.
4. Why is YAGNI important for agile development?
5. How do these principles work together with SOLID?`,
					CodeExamples: `DRY Principle:

// Bad: Repeated code
class OrderService {
    public void validateOrder(Order order) {
        if (order.getTotal() < 0) {
            throw new ValidationException("Total cannot be negative");
        }
        if (order.getItems().isEmpty()) {
            throw new ValidationException("Order must have items");
        }
    }
}

class PaymentService {
    public void validatePayment(Payment payment) {
        if (payment.getAmount() < 0) {
            throw new ValidationException("Amount cannot be negative");
        }
        if (payment.getMethod() == null) {
            throw new ValidationException("Payment method required");
        }
    }
}

// Good: Reusable validation
class Validator {
    public static void validatePositive(double value, String field) {
        if (value < 0) {
            throw new ValidationException(field + " cannot be negative");
        }
    }
    
    public static void validateNotNull(Object value, String field) {
        if (value == null) {
            throw new ValidationException(field + " is required");
        }
    }
}

KISS Principle:

// Bad: Over-engineered
class SimpleCalculator {
    private CalculatorStrategyFactory factory;
    private CalculatorContext context;
    private CalculatorExecutor executor;
    
    public int add(int a, int b) {
        CalculatorStrategy strategy = factory.create(Operation.ADD);
        context.setOperands(a, b);
        return executor.execute(strategy, context);
    }
}

// Good: Simple and clear
class SimpleCalculator {
    public int add(int a, int b) {
        return a + b;
    }
}

YAGNI Principle:

// Bad: Implementing features "just in case"
class UserService {
    // Current requirement: Basic user CRUD
    public void createUser(User user) { }
    public void updateUser(User user) { }
    public void deleteUser(Long id) { }
    
    // YAGNI violation: Not needed yet
    public void bulkCreateUsers(List<User> users) { }
    public void exportUsersToCSV() { }
    public void importUsersFromXML() { }
    public void mergeDuplicateUsers() { }
}

// Good: Only what's needed
class UserService {
    // Current requirement: Basic user CRUD
    public void createUser(User user) { }
    public void updateUser(User user) { }
    public void deleteUser(Long id) { }
    // Add other methods when actually needed
}

Composition over Inheritance:

// Bad: Deep inheritance hierarchy
class Animal {
    void eat() { }
}

class Mammal extends Animal {
    void breathe() { }
}

class Dog extends Mammal {
    void bark() { }
}

class GuardDog extends Dog {
    void guard() { }
}

// Good: Composition
interface Eatable {
    void eat();
}

interface Breatheable {
    void breathe();
}

interface Barkable {
    void bark();
}

interface Guardable {
    void guard();
}

class Dog {
    private Eatable eatingBehavior;
    private Breatheable breathingBehavior;
    private Barkable barkingBehavior;
    
    // Compose behaviors
}

class GuardDog {
    private Dog dog;
    private Guardable guardingBehavior;
    
    // Compose instead of inherit`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
