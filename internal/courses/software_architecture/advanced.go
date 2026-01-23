package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          36,
			Title:       "Enterprise Integration Patterns",
			Description: "Learn enterprise integration patterns for connecting systems, including messaging patterns, routing, and transformation.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "Messaging Patterns",
					Content: `Messaging patterns enable asynchronous communication between systems.

**Point-to-Point:**
- One producer, one consumer
- Message consumed once
- Queue-based
- Example: Task queue

**Publish-Subscribe:**
- One producer, multiple consumers
- Each consumer gets copy
- Topic-based
- Example: Event notifications

**Request-Reply:**
- Request message, reply message
- Synchronous-like over async
- Correlation ID to match requests/replies
- Example: RPC over messaging

**Message Router:**
- Routes messages based on content
- Content-based routing
- Conditional routing
- Example: Route orders by priority

**Message Translator:**
- Transforms message format
- Protocol conversion
- Data transformation
- Example: XML to JSON

**Message Filter:**
- Filters messages based on criteria
- Only relevant messages pass through
- Reduces processing load
- Example: Filter high-priority orders

**Benefits:**
- Loose coupling
- Asynchronous processing
- Scalability
- Reliability

**Real-World Examples:**

**Point-to-Point - Task Queue (Celery/RabbitMQ):**
- Image processing tasks
- Email sending
- Report generation
- One worker processes each task

**Publish-Subscribe - Event Streaming (Kafka):**
- Order events: Email service, Inventory service, Analytics service all receive events
- User activity events: Multiple analytics systems consume
- System events: Monitoring, logging, alerting

**Request-Reply - RPC over Messaging (gRPC, RabbitMQ RPC):**
- Service-to-service calls with async messaging
- Long-running operations with callbacks
- Distributed transactions

**Message Router - Order Processing:**
- Route high-priority orders to premium processing queue
- Route large orders to special handling
- Route standard orders to normal queue

**When to Use Each Pattern:**

**Point-to-Point:**
- Task processing
- Work queues
- Load distribution
- When each message should be processed once

**Publish-Subscribe:**
- Event broadcasting
- Multiple consumers need same data
- Decoupled event processing
- When multiple systems need notifications

**Request-Reply:**
- RPC over messaging
- Async request handling
- When you need responses but want async benefits

**Key Takeaways:**
- Choose pattern based on communication needs
- Point-to-Point: One consumer per message
- Publish-Subscribe: Multiple consumers per message
- Request-Reply: Need response to request
- Patterns can be combined for complex scenarios

**Self-Assessment Questions:**
1. When would you use Publish-Subscribe instead of Point-to-Point?
2. How does Request-Reply pattern work over async messaging?
3. Give an example of when Message Router would be useful.
4. What's the difference between Message Translator and Message Filter?
5. How do messaging patterns enable loose coupling?`,
					CodeExamples: `Point-to-Point Pattern:

Producer → Queue → Consumer

Producer:
queue.send("task_queue", {
    task_id: 123,
    type: "process_image",
    data: image_data
})

Consumer:
message = queue.receive("task_queue")
process_task(message)
queue.acknowledge(message)

Publish-Subscribe Pattern:

Producer → Topic → [Consumer1, Consumer2, Consumer3]

Producer:
pubsub.publish("order_events", {
    event: "order_created",
    order_id: 123,
    customer_id: 456
})

Consumers:
pubsub.subscribe("order_events", email_handler)
pubsub.subscribe("order_events", inventory_handler)
pubsub.subscribe("order_events", analytics_handler)

Request-Reply Pattern:

Client:
correlation_id = generate_id()
request_queue.send({
    correlation_id: correlation_id,
    request: "get_user",
    user_id: 123
})

reply = reply_queue.receive(correlation_id)

Server:
request = request_queue.receive()
result = process_request(request)
reply_queue.send({
    correlation_id: request.correlation_id,
    result: result
})

Message Router:

class OrderRouter {
    route(message) {
        if (message.priority == "high") {
            return "high_priority_queue"
        } else if (message.amount > 1000) {
            return "large_order_queue"
        } else {
            return "standard_queue"
        }
    }
}

Message Translator:

class XMLToJSONTranslator {
    translate(xmlMessage) {
        xmlData = parseXML(xmlMessage)
        jsonData = convertToJSON(xmlData)
        return jsonData
    }
}

Message Filter:

class HighPriorityFilter {
    filter(message) {
        return message.priority == "high"
    }
}`,
				},
				{
					Title: "Integration Styles and Error Handling",
					Content: `Different integration styles suit different scenarios.

**Integration Styles:**

**1. File Transfer:**
- Systems exchange files
- Batch processing
- Simple but slow
- Example: Daily CSV exports

**2. Shared Database:**
- Systems share database
- Direct data access
- Tight coupling
- Example: Legacy integration

**3. Remote Procedure Invocation (RPC):**
- Synchronous calls
- Tight coupling
- Fast but fragile
- Example: gRPC, REST

**4. Messaging:**
- Asynchronous communication
- Loose coupling
- Reliable
- Example: Message queues

**Error Handling Patterns:**

**Retry:**
- Retry failed operations
- Exponential backoff
- Maximum retry attempts
- Idempotent operations

**Dead Letter Queue:**
- Failed messages after max retries
- Manual inspection
- Prevents message loss
- Example: Invalid messages

**Circuit Breaker:**
- Stop calling failing service
- Fail fast
- Allow recovery time
- Example: External API failures

**Compensating Transaction:**
- Undo completed operations
- Saga pattern
- Eventual consistency
- Example: Order cancellation

**Integration Style Selection Guide:**

**File Transfer:**
- Use when: Batch processing, large data volumes, systems can't connect directly
- Pros: Simple, no real-time connection needed
- Cons: Slow, data staleness, file management overhead
- Example: Daily sales reports, bulk data imports

**Shared Database:**
- Use when: Legacy systems, tight integration needed, same team owns systems
- Pros: Fast access, simple queries
- Cons: Tight coupling, scalability issues, data conflicts
- Example: Monolithic applications, legacy ERP integration

**RPC (Remote Procedure Call):**
- Use when: Real-time response needed, synchronous operations, low latency critical
- Pros: Fast, familiar programming model
- Cons: Tight coupling, network failures affect system, cascading failures
- Example: Internal microservices, real-time queries

**Messaging:**
- Use when: Loose coupling needed, async processing, high reliability required
- Pros: Loose coupling, scalable, reliable, fault tolerant
- Cons: Eventual consistency, more complex, message ordering challenges
- Example: Event-driven systems, microservices, distributed systems

**Error Handling Best Practices:**

**Retry Strategy:**
- Exponential backoff: 1s, 2s, 4s, 8s, 16s
- Maximum retries: 3-5 attempts typically
- Idempotency: Operations must be safe to retry
- Jitter: Add randomness to prevent thundering herd

**Dead Letter Queue (DLQ):**
- After max retries, move to DLQ
- Manual inspection and reprocessing
- Alert operations team
- Common causes: Invalid data, downstream service down, bugs

**Circuit Breaker States:**
- Closed: Normal operation, requests pass through
- Open: Service failing, requests fail fast, no calls made
- Half-Open: Testing if service recovered, limited requests allowed

**Compensating Transactions (Saga Pattern):**
- Each step has compensating action
- If step fails, execute compensations in reverse order
- Eventual consistency acceptable
- Example: Order → Payment → Inventory → Shipping
  - If Shipping fails: Cancel Inventory → Refund Payment → Cancel Order

**Real-World Example - E-commerce Order Processing:**

**Scenario:** Process order with payment, inventory, and shipping

**Without Error Handling:**
- Payment succeeds → Inventory fails → Order stuck, payment charged
- User frustrated, manual intervention needed

**With Error Handling:**
- Payment succeeds → Inventory fails → Compensate: Refund payment
- Retry inventory with backoff
- If still fails after retries → Move to DLQ for manual processing
- Circuit breaker prevents overwhelming inventory service

**Key Takeaways:**
- Choose integration style based on requirements
- File Transfer: Batch, simple
- Shared Database: Fast but tightly coupled
- RPC: Real-time, synchronous
- Messaging: Loose coupling, async, reliable
- Error handling is critical for distributed systems
- Retry, DLQ, Circuit Breaker, and Compensating Transactions are essential patterns

**Self-Assessment Questions:**
1. When would you choose File Transfer over Messaging?
2. What are the trade-offs between RPC and Messaging?
3. How does exponential backoff help with retries?
4. When would you use a Dead Letter Queue?
5. Explain how the Saga pattern handles distributed transactions.`,
					CodeExamples: `Integration Styles Comparison:

File Transfer:
System A → CSV File → System B
- Simple
- Slow
- Batch processing

Shared Database:
System A ──┐
           ├──→ Shared Database
System B ──┘
- Fast
- Tight coupling
- Data conflicts

RPC:
System A → RPC Call → System B
- Fast
- Synchronous
- Tight coupling

Messaging:
System A → Message Queue → System B
- Loose coupling
- Asynchronous
- Reliable

Retry Pattern:

class RetryHandler {
    retry(func, maxAttempts = 3) {
        for (attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return func()
            } catch (error) {
                if (attempt == maxAttempts) {
                    throw error
                }
                delay = exponentialBackoff(attempt)
                sleep(delay)
            }
        }
    }
}

Dead Letter Queue:

class MessageProcessor {
    process(message) {
        try {
            handleMessage(message)
        } catch (error) {
            retryCount = message.retryCount || 0
            if (retryCount < MAX_RETRIES) {
                message.retryCount = retryCount + 1
                queue.retry(message)
            } else {
                deadLetterQueue.send(message)
            }
        }
    }
}

Circuit Breaker:

class CircuitBreaker {
    constructor(threshold = 5, timeout = 60) {
        this.failureCount = 0
        this.state = "CLOSED" // CLOSED, OPEN, HALF_OPEN
        this.threshold = threshold
        this.timeout = timeout
    }
    
    call(func) {
        if (this.state == "OPEN") {
            if (timeSinceLastFailure > this.timeout) {
                this.state = "HALF_OPEN"
            } else {
                throw new CircuitBreakerOpenError()
            }
        }
        
        try {
            result = func()
            this.onSuccess()
            return result
        } catch (error) {
            this.onFailure()
            throw error
        }
    }
}

Compensating Transaction:

class OrderSaga {
    createOrder(order) {
        try {
            order = orderService.create(order)
            inventory = inventoryService.reserve(order.items)
            payment = paymentService.charge(order.total)
            shipping = shippingService.ship(order)
        } catch (error) {
            // Compensate
            if (payment) paymentService.refund(payment)
            if (inventory) inventoryService.release(inventory)
            if (order) orderService.cancel(order)
            throw error
        }
    }
}`,
			},
			ProblemIDs: []int{},
		},
		{
			ID:          37,
			Title:       "Architecture Decision Records (ADRs)",
			Description: "Learn how to document architectural decisions using Architecture Decision Records (ADRs).",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Writing Effective ADRs",
					Content: `Architecture Decision Records (ADRs) document important architectural decisions and their rationale.

**Purpose:**
- Capture architectural decisions
- Document context and rationale
- Enable future understanding
- Track decision history

**ADR Structure:**

**1. Title:**
- Descriptive title
- Usually starts with number
- Example: "001-use-microservices-architecture"

**2. Status:**
- Proposed, Accepted, Rejected, Deprecated, Superseded
- Current state of decision

**3. Context:**
- What situation led to this decision?
- What problem are we solving?
- What constraints exist?

**4. Decision:**
- What decision was made?
- Clear and specific
- What was chosen and why

**5. Consequences:**
- Positive consequences
- Negative consequences
- Trade-offs
- Risks

**Best Practices:**
- Write ADRs for significant decisions
- Keep them concise
- Update status as decisions evolve
- Link related ADRs
- Review periodically

**When to Write an ADR:**

**Write ADRs for:**
- Architectural patterns (microservices, monolith, event-driven)
- Technology choices (database, messaging, framework)
- Integration approaches (REST, GraphQL, gRPC)
- Infrastructure decisions (cloud provider, deployment strategy)
- Significant design decisions affecting multiple teams
- Decisions that are hard to reverse

**Don't Write ADRs for:**
- Implementation details
- Routine choices
- Temporary workarounds
- Obvious decisions with no alternatives

**ADR Lifecycle:**

1. **Proposed**: Decision under consideration
2. **Accepted**: Decision approved and implemented
3. **Rejected**: Alternative chosen instead
4. **Deprecated**: Decision no longer recommended
5. **Superseded**: Replaced by new ADR

**Real-World ADR Example:**

**ADR-001: Use Microservices Architecture**
- Context: Monolith scaling issues, team conflicts
- Decision: Adopt microservices with API Gateway
- Consequences: Better scalability, but increased complexity
- Status: Accepted (2023-01-15)

**ADR-002: Use PostgreSQL for Primary Database**
- Context: Need ACID transactions, complex queries
- Decision: PostgreSQL over MySQL or NoSQL
- Consequences: Strong consistency, but scaling challenges
- Status: Accepted (2023-02-01)

**ADR-003: Use Event-Driven Architecture for Order Processing**
- Context: Need loose coupling, async processing
- Decision: Kafka for event streaming
- Consequences: Scalable, but eventual consistency
- Status: Accepted (2023-03-10)

**ADR Template:**

ADR-XXX: [Short Title]

Status:
[Proposed | Accepted | Rejected | Deprecated | Superseded]

Context:
[What situation led to this decision? What problem are we solving?]

Decision:
[What decision was made? Be specific and clear.]

Consequences:

Positive:
- [Benefit 1]
- [Benefit 2]

Negative:
- [Drawback 1]
- [Drawback 2]

Risks:
- [Risk 1]
- [Risk 2]

Alternatives Considered:
- [Alternative 1]: [Why rejected]
- [Alternative 2]: [Why rejected]

References:
- [Link to related ADRs]
- [Link to documentation]

**Key Takeaways:**
- ADRs document important architectural decisions
- Include context, decision, and consequences
- Update status as decisions evolve
- Link related ADRs for traceability
- Review periodically to ensure decisions remain valid
- Use ADRs to onboard new team members

**Self-Assessment Questions:**
1. What information should be included in an ADR?
2. When should you write an ADR vs. just documenting in code?
3. How do ADRs help with onboarding new team members?
4. What's the difference between Deprecated and Superseded status?
5. Give an example of a decision that would NOT need an ADR.`,
					CodeExamples: `ADR Example:

ADR-001: Use Microservices Architecture

Status:
Accepted

Context:
Our monolithic application is becoming difficult to scale and maintain. 
We have multiple teams working on different features, and deployments 
are becoming risky. We need to improve scalability and enable team 
autonomy.

Decision:
We will adopt a microservices architecture with the following principles:
- Each service owns its data (database per service)
- Services communicate via REST APIs
- Use API Gateway for external access
- Implement service mesh for inter-service communication
- Each service can be developed and deployed independently

Consequences:

Positive:
- Independent scaling of services
- Team autonomy and faster development
- Technology diversity (each service can use different tech stack)
- Better fault isolation
- Easier to understand individual services

Negative:
- Increased operational complexity
- Network latency between services
- Distributed system challenges (consistency, transactions)
- More complex testing
- Higher infrastructure costs

Risks:
- Service boundaries may be wrong initially
- Over-engineering for simple use cases
- Distributed system failures
- Data consistency challenges

Alternatives Considered:
1. Modular Monolith: Rejected - still single deployment, doesn't solve scaling
2. Service-Oriented Architecture (SOA): Rejected - too heavyweight for our needs
3. Keep Monolith: Rejected - doesn't solve current problems

ADR Template:

ADR-XXX: [Title]

Status:
[Proposed | Accepted | Rejected | Deprecated | Superseded]

Context:
[Describe the situation and problem]

Decision:
[Describe the decision made]

Consequences:

Positive:
- [Positive consequence 1]
- [Positive consequence 2]

Negative:
- [Negative consequence 1]
- [Negative consequence 2]

Risks:
- [Risk 1]
- [Risk 2]

Alternatives Considered:
1. [Alternative 1]: [Why rejected]
2. [Alternative 2]: [Why rejected]

Related ADRs:
- ADR-XXX: [Related decision]
- ADR-YYY: [Related decision]`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          38,
			Title:       "Architecture Documentation",
			Description: "Learn the C4 model and best practices for documenting software architecture.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "C4 Model for Architecture Documentation",
					Content: `The C4 model provides a hierarchical way to document software architecture at different levels of detail.

**Four Levels:**

**1. Context Diagram (Level 1):**
- Shows system and its relationships with users and other systems
- High-level view
- Audience: Technical and non-technical stakeholders
- Example: E-commerce system and external systems

**2. Container Diagram (Level 2):**
- Shows containers (applications, databases, file systems) within system
- Technology choices visible
- Audience: Technical people
- Example: Web application, API, database, cache

**3. Component Diagram (Level 3):**
- Shows components within a container
- Shows responsibilities and relationships
- Audience: Developers
- Example: Controllers, services, repositories

**4. Code Diagram (Level 4):**
- Shows classes and their relationships
- Optional, usually auto-generated
- Audience: Developers
- Example: Class diagrams

**Benefits:**
- Hierarchical structure
- Different levels for different audiences
- Clear and consistent
- Easy to understand

**Notation:**
- Rectangles for software systems/containers/components
- Arrows for relationships
- Colors for different types
- Notes for additional information

**When to Use Each Level:**

**Context Diagram:**
- Onboarding new team members
- Explaining system to stakeholders
- High-level architecture discussions
- System overview presentations

**Container Diagram:**
- Technology stack discussions
- Infrastructure planning
- Deployment architecture
- Integration planning

**Component Diagram:**
- Detailed design discussions
- Code organization planning
- Developer onboarding
- Refactoring planning

**Code Diagram:**
- Detailed implementation discussions
- Code reviews
- Usually auto-generated from code
- Optional for most cases

**C4 Model Best Practices:**

**1. Start with Context:**
- Always begin with context diagram
- Shows big picture
- Helps stakeholders understand scope

**2. Drill Down as Needed:**
- Not all containers need component diagrams
- Only document what's important
- Focus on complex or critical areas

**3. Keep Diagrams Simple:**
- Don't show everything
- Focus on key relationships
- Use colors and labels effectively

**4. Update Regularly:**
- Keep diagrams in sync with code
- Update when architecture changes
- Version control diagrams

**5. Use Consistent Notation:**
- Standard shapes and colors
- Clear labels
- Consistent arrow styles

**Real-World Example - E-commerce Platform:**

**Context Level:**
- System: E-commerce Platform
- Users: Customers, Admins
- External Systems: Payment Gateway, Shipping Provider, Email Service

**Container Level:**
- Web Application (React)
- API Gateway (Kong)
- Order Service (Node.js)
- Payment Service (Java)
- Inventory Service (Python)
- Database (PostgreSQL)
- Cache (Redis)
- Message Queue (RabbitMQ)

**Component Level (Order Service):**
- Order Controller
- Order Service
- Order Repository
- Payment Client
- Inventory Client
- Event Publisher

**Benefits of C4 Model:**

**For Stakeholders:**
- Easy to understand at context level
- Shows business value and relationships
- Non-technical people can understand

**For Architects:**
- Clear structure for documentation
- Easy to navigate different levels
- Standardized approach

**For Developers:**
- Understand system structure
- See how components fit together
- Plan changes effectively

**Key Takeaways:**
- C4 model provides hierarchical documentation
- Four levels: Context, Container, Component, Code
- Each level serves different audience
- Start with context, drill down as needed
- Keep diagrams simple and up-to-date
- Use consistent notation

**Self-Assessment Questions:**
1. What are the four levels of the C4 model?
2. Who is the audience for each level?
3. Why start with a context diagram?
4. When would you create a component diagram?
5. How does the C4 model help with onboarding?`,
					CodeExamples: `Context Diagram (Level 1):

┌──────────────┐
│   Customer   │
└──────┬───────┘
       │
       │ Uses
       │
       ▼
┌─────────────────────────────┐
│   E-commerce System          │
└──────┬───────────────────────┘
       │
       │ Uses
       │
       ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Payment      │  │ Shipping     │  │ Email        │
│ Gateway      │  │ Provider     │  │ Service      │
└──────────────┘  └──────────────┘  └──────────────┘

Container Diagram (Level 2):

┌─────────────────────────────────────────────┐
│         E-commerce System                    │
│                                               │
│  ┌──────────────┐      ┌──────────────┐      │
│  │ Web App      │──────│ API          │      │
│  │ (React)      │      │ (Node.js)    │      │
│  └──────────────┘      └──────┬───────┘      │
│                                │              │
│                        ┌──────▼───────┐      │
│                        │ Database      │      │
│                        │ (PostgreSQL)  │      │
│                        └───────────────┘      │
│                                               │
│                        ┌───────────────┐     │
│                        │ Cache         │     │
│                        │ (Redis)       │     │
│                        └───────────────┘     │
└───────────────────────────────────────────────┘

Component Diagram (Level 3):

┌─────────────────────────────────────────────┐
│              API Container                  │
│                                              │
│  ┌──────────────┐      ┌──────────────┐     │
│  │ Order        │──────│ Payment      │     │
│  │ Controller   │      │ Controller   │     │
│  └──────┬───────┘      └──────┬───────┘     │
│         │                     │              │
│         ▼                     ▼             │
│  ┌──────────────┐      ┌──────────────┐     │
│  │ Order        │      │ Payment     │     │
│  │ Service      │      │ Service     │     │
│  └──────┬───────┘      └──────┬───────┘     │
│         │                     │              │
│         ▼                     ▼             │
│  ┌──────────────┐      ┌──────────────┐     │
│  │ Order        │      │ Payment     │     │
│  │ Repository   │      │ Repository  │     │
│  └──────────────┘      └──────────────┘     │
└──────────────────────────────────────────────┘

Documentation Structure:

1. System Overview
   - Purpose and scope
   - Key stakeholders
   - Context diagram

2. Architecture Decisions
   - ADRs
   - Key design decisions
   - Rationale

3. Architecture Diagrams
   - Context diagram
   - Container diagrams
   - Component diagrams (key components)

4. Quality Attributes
   - Performance requirements
   - Security requirements
   - Scalability requirements

5. Deployment
   - Deployment architecture
   - Infrastructure requirements
   - Environment configuration

6. Development Guidelines
   - Coding standards
   - Architecture patterns
   - Testing strategies`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          39,
			Title:       "Technology Selection & Trade-offs",
			Description: "Learn frameworks for evaluating technologies and making informed architectural decisions.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Technology Evaluation Framework",
					Content: `Selecting the right technology is crucial for project success. A structured evaluation helps make informed decisions.

**Evaluation Criteria:**

**1. Functional Requirements:**
- Does it meet functional needs?
- Feature completeness
- API quality
- Documentation

**2. Non-Functional Requirements:**
- Performance characteristics
- Scalability
- Reliability
- Security features

**3. Technical Fit:**
- Compatibility with existing stack
- Learning curve
- Community support
- Maturity and stability

**4. Team Factors:**
- Team expertise
- Training requirements
- Hiring availability
- Development velocity

**5. Business Factors:**
- Cost (licensing, hosting, support)
- Vendor support
- Long-term viability
- Migration path

**Evaluation Process:**

**1. Define Requirements:**
- List must-have features
- List nice-to-have features
- Define constraints

**2. Identify Candidates:**
- Research options
- Get recommendations
- Consider alternatives

**3. Evaluate Each Candidate:**
- Score against criteria
- POC (Proof of Concept) if needed
- Get team feedback

**4. Make Decision:**
- Compare scores
- Consider trade-offs
- Document decision (ADR)

**5. Review Periodically:**
- Re-evaluate as needs change
- Monitor technology evolution
- Be ready to pivot if needed

**Common Technology Selection Scenarios:**

**Scenario 1: Database Selection**
- **Requirements**: ACID transactions, complex queries, team knows SQL
- **Candidates**: PostgreSQL, MySQL, Oracle, MongoDB
- **Decision**: PostgreSQL (open source, excellent features, team expertise)
- **Trade-off**: More complex than MySQL, but better features

**Scenario 2: Messaging System**
- **Requirements**: High throughput, event streaming, multiple consumers
- **Candidates**: RabbitMQ, Kafka, AWS SQS, Redis Pub/Sub
- **Decision**: Kafka (high throughput, event streaming, replay capability)
- **Trade-off**: More complex setup, but better for event streaming

**Scenario 3: API Framework**
- **Requirements**: REST API, team knows Java, fast development
- **Candidates**: Spring Boot, Dropwizard, JAX-RS, Node.js
- **Decision**: Spring Boot (team expertise, rapid development, ecosystem)
- **Trade-off**: Heavier than alternatives, but better ecosystem

**Technology Selection Anti-Patterns:**

**1. Resume-Driven Development:**
- Choosing technology to learn new skills
- **Problem**: May not be best fit for project
- **Solution**: Choose based on project needs, not personal goals

**2. Latest and Greatest:**
- Always choosing newest technology
- **Problem**: Unproven, may have issues
- **Solution**: Balance innovation with stability

**3. Vendor Lock-in:**
- Choosing proprietary solutions without exit strategy
- **Problem**: Hard to migrate, expensive
- **Solution**: Prefer open standards, have migration plan

**4. One-Size-Fits-All:**
- Using same technology for everything
- **Problem**: Not optimal for all use cases
- **Solution**: Choose right tool for each job

**5. Analysis Paralysis:**
- Over-analyzing without making decision
- **Problem**: Delays project, opportunity cost
- **Solution**: Set time limit, make decision, iterate

**Real-World Example - Microservices Messaging:**

**Context**: Need messaging for microservices communication

**Requirements:**
- High throughput (100K messages/sec)
- Event streaming capability
- Multiple consumers per message
- At-least-once delivery
- Team has Java expertise

**Candidates Evaluated:**

**RabbitMQ:**
- Pros: Simple, good documentation, team knows it
- Cons: Lower throughput, not designed for streaming
- Score: 6/10

**Apache Kafka:**
- Pros: High throughput, event streaming, replay capability
- Cons: Steeper learning curve, more complex
- Score: 9/10

**AWS SQS:**
- Pros: Managed service, simple
- Cons: Vendor lock-in, lower throughput, more expensive
- Score: 7/10

**Decision**: Apache Kafka
- **Rationale**: Best fit for requirements, high throughput, event streaming
- **Trade-off**: More complex, but worth it for capabilities
- **ADR**: ADR-005: Use Apache Kafka for Event Streaming

**Key Takeaways:**
- Evaluate technologies systematically
- Consider functional, non-functional, team, and business factors
- Use evaluation matrix for comparison
- Avoid common anti-patterns
- Document decisions in ADRs
- Review periodically as needs evolve
- Balance innovation with stability
- Choose right tool for each job

**Self-Assessment Questions:**
1. What criteria should you consider when selecting a technology?
2. How do you balance team expertise with technology fit?
3. What is "resume-driven development" and why is it problematic?
4. When would you choose a managed service vs. self-hosted?
5. How do you handle technology selection when requirements conflict?`,
					CodeExamples: `Technology Evaluation Matrix:

Criteria                Weight  Option A  Option B  Option C
────────────────────────────────────────────────────────────
Functional Fit          30%     8         7         9
Performance            20%     7         9         6
Team Expertise         15%     9         6         7
Community Support      15%     8         9         7
Cost                   10%     7         8         6
Maturity               10%     9         7         8
────────────────────────────────────────────────────────────
Weighted Score                 8.0       7.5       7.4

Example: Database Selection

Requirements:
- Must support ACID transactions
- Must scale to 1M users
- Team knows SQL
- Budget: $10K/month

Candidates:
1. PostgreSQL (Open source)
2. MySQL (Open source)
3. Oracle (Commercial)
4. MongoDB (NoSQL)

Evaluation:

PostgreSQL:
✅ ACID transactions
✅ Scales well
✅ Team knows SQL
✅ Free (open source)
✅ Excellent performance
Score: 9/10

MySQL:
✅ ACID transactions
⚠️ Scaling challenges
✅ Team knows SQL
✅ Free (open source)
⚠️ Performance limitations
Score: 7/10

Oracle:
✅ ACID transactions
✅ Excellent scaling
✅ Team knows SQL
❌ Expensive ($50K+/month)
✅ Excellent performance
Score: 6/10 (cost too high)

MongoDB:
❌ No ACID (eventual consistency)
✅ Excellent scaling
❌ Team doesn't know NoSQL
✅ Free (open source)
✅ Excellent performance
Score: 5/10 (doesn't meet ACID requirement)

Decision: PostgreSQL (best fit)

Trade-off Analysis:

Option: Use Microservices
Pros:
- Independent scaling
- Team autonomy
- Technology diversity

Cons:
- Increased complexity
- Network latency
- Distributed system challenges

Trade-off: Accept complexity for scalability

Risk Assessment:

Technology: New Framework X
Risk: Low maturity, small community
Mitigation: 
- POC to validate
- Have fallback option
- Monitor community growth

Technology: Legacy System Y
Risk: Vendor end-of-life
Mitigation:
- Plan migration path
- Document dependencies
- Set migration timeline`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID: 40,
			Title:       "Real-World Architecture Examples",
			Description: "Explore how major companies and open-source projects apply architectural patterns, design principles, and make architectural decisions in practice.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "Clean Architecture in Practice",
					Content: `Clean Architecture principles are applied in many real-world systems. Let's examine how companies and open-source projects implement these patterns.

**Android Architecture Components (Google):**

Google's Android Architecture Components demonstrate Clean Architecture principles in mobile development:

**Architecture Layers:**
- **UI Layer**: Activities, Fragments, ViewModels (Presentation)
- **Domain Layer**: Use cases, entities (Business Logic)
- **Data Layer**: Repositories, Data Sources (Data Access)

**Key Implementation:**
- ViewModels are framework-independent (can be tested without Android)
- Use cases contain business logic (independent of UI and data)
- Repository pattern abstracts data sources
- Dependency injection enables testing

**Benefits Realized:**
- 80%+ code coverage in business logic
- Easy to swap data sources (REST API → local database)
- UI can be tested independently
- Business logic reusable across platforms

**Spring Boot Applications:**

Many enterprise Spring Boot applications follow Clean Architecture:

**Typical Structure:**
- **Controller Layer**: REST endpoints (Framework)
- **Service Layer**: Business logic (Application)
- **Repository Layer**: Data access (Infrastructure)
- **Domain Models**: Entities (Domain)

**Real-World Example - E-commerce Platform:**
- Controllers handle HTTP requests/responses
- Services contain business rules (pricing, validation)
- Repositories abstract database access
- Domain models are pure Java objects

**Key Patterns:**
- Dependency Injection (Spring IoC)
- Interface-based design (repositories are interfaces)
- Separation of concerns (each layer has single responsibility)

**Open-Source Example - GitHub Projects:**

Popular GitHub projects demonstrating Clean Architecture:

**1. Clean Architecture Sample (by Philipp Lackner):**
- Android app with clear layer separation
- Use cases for each business operation
- Repository pattern for data access
- Dependency injection with Dagger/Hilt

**2. Spring Boot Clean Architecture Template:**
- REST API with layered architecture
- Domain-driven design principles
- Testable business logic
- Framework-independent core

**Migration Case Study - Monolith to Clean Architecture:**

**Before (Monolithic):**
- Business logic mixed with database code
- Hard to test (requires database)
- Tight coupling between layers
- Difficult to change frameworks

**After (Clean Architecture):**
- Business logic isolated in domain layer
- Easy to test (mock dependencies)
- Loose coupling (interfaces)
- Framework changes don't affect business logic

**Key Takeaways:**
- Clean Architecture enables high test coverage
- Business logic becomes framework-independent
- Easier to maintain and evolve
- Migration requires refactoring but pays off long-term`,
					CodeExamples: `Android Architecture Components Example:

// Domain Layer (Business Logic)
class GetUserUseCase(
    private val userRepository: UserRepository
) {
    suspend operator fun invoke(userId: String): Result<User> {
        return userRepository.getUser(userId)
    }
}

// Data Layer (Repository Implementation)
class UserRepositoryImpl(
    private val api: UserApi,
    private val cache: UserCache
) : UserRepository {
    override suspend fun getUser(userId: String): Result<User> {
        return try {
            val cached = cache.getUser(userId)
            if (cached != null) {
                Result.success(cached)
            } else {
                val user = api.getUser(userId)
                cache.saveUser(user)
                Result.success(user)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}

// UI Layer (ViewModel)
class UserViewModel(
    private val getUserUseCase: GetUserUseCase
) : ViewModel() {
    private val _user = MutableStateFlow<User?>(null)
    val user: StateFlow<User?> = _user
    
    fun loadUser(userId: String) {
        viewModelScope.launch {
            getUserUseCase(userId).onSuccess { user ->
                _user.value = user
            }
        }
    }
}

Spring Boot Clean Architecture Example:

// Domain Entity
public class Order {
    private Long id;
    private Customer customer;
    private List<OrderItem> items;
    private Money total;
    
    public void calculateTotal() {
        // Business logic - no framework dependencies
        total = items.stream()
            .map(item -> item.getPrice().multiply(item.getQuantity()))
            .reduce(Money.ZERO, Money::add);
    }
}

// Use Case (Application Layer)
@Service
public class CreateOrderUseCase {
    private final OrderRepository orderRepository;
    private final InventoryService inventoryService;
    
    public Order execute(CreateOrderCommand command) {
        Order order = new Order(command.getCustomer(), command.getItems());
        order.calculateTotal();
        
        // Validate inventory
        inventoryService.reserveItems(order.getItems());
        
        return orderRepository.save(order);
    }
}

// Repository Interface (Application Layer)
public interface OrderRepository {
    Order save(Order order);
    Optional<Order> findById(Long id);
}

// Repository Implementation (Infrastructure Layer)
@Repository
public class JpaOrderRepository implements OrderRepository {
    @Autowired
    private OrderJpaRepository jpaRepository;
    
    @Override
    public Order save(Order order) {
        OrderEntity entity = mapToEntity(order);
        OrderEntity saved = jpaRepository.save(entity);
        return mapToDomain(saved);
    }
}

// Controller (Framework Layer)
@RestController
@RequestMapping("/api/orders")
public class OrderController {
    private final CreateOrderUseCase createOrderUseCase;
    
    @PostMapping
    public ResponseEntity<OrderDTO> createOrder(@RequestBody CreateOrderRequest request) {
        CreateOrderCommand command = mapToCommand(request);
        Order order = createOrderUseCase.execute(command);
        return ResponseEntity.ok(mapToDTO(order));
    }
}

Testing Benefits:

// Test Use Case without database or framework
@Test
void testCreateOrder() {
    OrderRepository mockRepo = mock(OrderRepository.class);
    InventoryService mockInventory = mock(InventoryService.class);
    
    CreateOrderUseCase useCase = new CreateOrderUseCase(mockRepo, mockInventory);
    
    CreateOrderCommand command = new CreateOrderCommand(customer, items);
    Order order = useCase.execute(command);
    
    assertNotNull(order);
    verify(mockInventory).reserveItems(items);
    verify(mockRepo).save(order);
}`,
				},
				{
					Title: "Domain-Driven Design Case Studies",
					Content: `Domain-Driven Design (DDD) is widely used in enterprise applications. Let's examine real-world implementations.

**Amazon - Bounded Contexts in Microservices:**

Amazon's e-commerce platform demonstrates DDD at scale:

**Bounded Contexts:**
- **Catalog Context**: Product information, categories, search
- **Order Context**: Order creation, processing, fulfillment
- **Payment Context**: Payment processing, refunds
- **Shipping Context**: Delivery, tracking, logistics
- **Customer Context**: User accounts, preferences, history

**Key Implementation:**
- Each microservice owns a bounded context
- Services communicate via APIs (no shared database)
- Ubiquitous language within each context
- Domain events for cross-context communication

**Example - Order Context:**
- **Aggregate Root**: Order
- **Entities**: OrderItem, ShippingAddress
- **Value Objects**: Money, OrderStatus
- **Domain Events**: OrderPlaced, OrderShipped, OrderDelivered

**Benefits:**
- Clear service boundaries
- Independent team ownership
- Scalable architecture
- Domain expertise per team

**Uber - Domain Modeling for Ride-Sharing:**

Uber's ride-sharing platform uses DDD to model complex business logic:

**Core Domains:**
- **Ride Domain**: Trip matching, routing, pricing
- **Driver Domain**: Driver availability, earnings
- **Payment Domain**: Fare calculation, processing
- **Matching Domain**: Driver-rider matching algorithm

**Domain Model Example - Ride Aggregate:**
- **Aggregate Root**: Ride
- **Entities**: Driver, Rider, Route
- **Value Objects**: Location, Fare, Rating
- **Domain Services**: RouteCalculator, FareCalculator

**Key Patterns:**
- Event-driven architecture (ride events)
- Saga pattern for distributed transactions
- CQRS for read/write separation
- Event sourcing for audit trail

**E-commerce Platform - Aggregate Design:**

Real-world e-commerce platforms demonstrate aggregate design:

**Order Aggregate:**
- **Root**: Order
- **Entities**: OrderItem, ShippingInfo
- **Invariants**: Order total must match sum of items
- **Boundary**: Order cannot be modified after confirmation

**Product Aggregate:**
- **Root**: Product
- **Value Objects**: Price, Inventory
- **Invariants**: Price must be positive, inventory cannot be negative

**Customer Aggregate:**
- **Root**: Customer
- **Entities**: Address, PaymentMethod
- **Domain Events**: CustomerRegistered, AddressChanged

**Open-Source DDD Examples:**

**1. Event Sourcing Example (EventStore):**
- Domain events stored as immutable facts
- Aggregates rebuilt from events
- Complete audit trail
- Time travel capability

**2. CQRS Implementation:**
- Command side: Write model with domain logic
- Query side: Read-optimized views
- Event bus connects both sides
- Eventual consistency

**Production Patterns:**

**Domain Events in Production:**
- Events published when domain logic completes
- Multiple handlers can react to same event
- Enables loose coupling between bounded contexts
- Supports eventual consistency

**Example Event Flow:**
1. Order aggregate publishes OrderPlaced event
2. Inventory service handles event (reserves items)
3. Payment service handles event (processes payment)
4. Notification service handles event (sends confirmation)

**Key Takeaways:**
- Bounded contexts map naturally to microservices
- Aggregates enforce business invariants
- Domain events enable loose coupling
- DDD scales well for complex domains`,
					CodeExamples: `Amazon Bounded Context Example:

// Order Bounded Context
public class Order {
    private OrderId id;
    private CustomerId customerId;
    private List<OrderItem> items;
    private OrderStatus status;
    private List<DomainEvent> domainEvents;
    
    public void confirm() {
        if (this.status != OrderStatus.DRAFT) {
            throw new IllegalStateException("Order already confirmed");
        }
        
        this.status = OrderStatus.CONFIRMED;
        this.domainEvents.add(new OrderConfirmedEvent(this.id, this.items));
    }
    
    public List<DomainEvent> getDomainEvents() {
        return new ArrayList<>(domainEvents);
    }
    
    public void clearEvents() {
        domainEvents.clear();
    }
}

// Order Service (Application Layer)
@Service
public class OrderService {
    private final OrderRepository orderRepository;
    private final EventPublisher eventPublisher;
    
    public void confirmOrder(OrderId orderId) {
        Order order = orderRepository.findById(orderId)
            .orElseThrow(() -> new OrderNotFoundException(orderId));
        
        order.confirm();
        orderRepository.save(order);
        
        // Publish domain events
        order.getDomainEvents().forEach(eventPublisher::publish);
        order.clearEvents();
    }
}

Uber Ride Domain Model:

// Ride Aggregate Root
public class Ride {
    private RideId id;
    private DriverId driverId;
    private RiderId riderId;
    private Location pickupLocation;
    private Location dropoffLocation;
    private RideStatus status;
    private Fare fare;
    
    public void start(Location currentLocation) {
        if (this.status != RideStatus.ACCEPTED) {
            throw new IllegalStateException("Ride not accepted");
        }
        
        if (!isNearPickupLocation(currentLocation)) {
            throw new InvalidLocationException("Driver not at pickup location");
        }
        
        this.status = RideStatus.IN_PROGRESS;
        this.domainEvents.add(new RideStartedEvent(this.id));
    }
    
    public void complete() {
        if (this.status != RideStatus.IN_PROGRESS) {
            throw new IllegalStateException("Ride not in progress");
        }
        
        this.status = RideStatus.COMPLETED;
        this.fare = calculateFare();
        this.domainEvents.add(new RideCompletedEvent(this.id, this.fare));
    }
    
    private Fare calculateFare() {
        // Domain logic for fare calculation
        Distance distance = calculateDistance(pickupLocation, dropoffLocation);
        return fareCalculator.calculate(distance, timeOfDay);
    }
}

E-commerce Aggregate Example:

// Order Aggregate
public class Order {
    private OrderId id;
    private CustomerId customerId;
    private List<OrderItem> items;
    private Money total;
    private OrderStatus status;
    
    public void addItem(Product product, Quantity quantity) {
        if (this.status != OrderStatus.DRAFT) {
            throw new IllegalStateException("Cannot modify confirmed order");
        }
        
        if (!product.isAvailable(quantity)) {
            throw new InsufficientInventoryException();
        }
        
        OrderItem item = new OrderItem(product.getId(), product.getPrice(), quantity);
        this.items.add(item);
        this.recalculateTotal();
    }
    
    private void recalculateTotal() {
        // Invariant: Total must equal sum of items
        this.total = items.stream()
            .map(item -> item.getPrice().multiply(item.getQuantity()))
            .reduce(Money.ZERO, Money::add);
    }
    
    public void confirm() {
        if (this.items.isEmpty()) {
            throw new EmptyOrderException();
        }
        
        this.status = OrderStatus.CONFIRMED;
        this.domainEvents.add(new OrderConfirmedEvent(this.id, this.items));
    }
}

Domain Events Example:

// Domain Event
public class OrderConfirmedEvent implements DomainEvent {
    private final OrderId orderId;
    private final List<OrderItem> items;
    private final Instant occurredAt;
    
    public OrderConfirmedEvent(OrderId orderId, List<OrderItem> items) {
        this.orderId = orderId;
        this.items = new ArrayList<>(items);
        this.occurredAt = Instant.now();
    }
}

// Event Handler (Inventory Service)
@Component
public class InventoryEventHandler {
    @EventListener
    public void handle(OrderConfirmedEvent event) {
        // Reserve inventory for confirmed order
        event.getItems().forEach(item -> {
            inventoryService.reserve(item.getProductId(), item.getQuantity());
        });
    }
}

// Event Handler (Payment Service)
@Component
public class PaymentEventHandler {
    @EventListener
    public void handle(OrderConfirmedEvent event) {
        // Process payment for confirmed order
        paymentService.charge(event.getOrderId(), calculateTotal(event.getItems()));
    }
}`,
				},
				{
					Title: "Microservices Architecture in Production",
					Content: `Major tech companies have successfully implemented microservices architectures. Let's examine real-world implementations.

**Netflix - Microservices Evolution:**

Netflix is a pioneer in microservices architecture:

**Architecture Overview:**
- 700+ microservices
- Each service handles specific business capability
- Services communicate via REST APIs and messaging
- Independent deployment and scaling

**Key Services:**
- **User Service**: User accounts, profiles, preferences
- **Content Service**: Movie/TV show metadata, catalog
- **Recommendation Service**: ML-based recommendations
- **Playback Service**: Video streaming, quality selection
- **Billing Service**: Subscription management, payments

**Architecture Patterns:**
- **API Gateway**: Zuul gateway routes requests
- **Service Discovery**: Eureka for service registration
- **Circuit Breaker**: Hystrix for fault tolerance
- **Load Balancing**: Ribbon for client-side load balancing
- **Configuration**: Archaius for dynamic configuration

**Evolution Journey:**
- Started as monolith (2008)
- Migrated to microservices (2010-2012)
- Moved to cloud (AWS)
- Implemented chaos engineering
- Continuous evolution

**Key Learnings:**
- Microservices enable rapid innovation
- Independent scaling is crucial
- Fault tolerance is essential
- Monitoring and observability are critical

**Amazon - Service Decomposition:**

Amazon's e-commerce platform demonstrates service decomposition:

**Service Boundaries:**
- Services organized by business capability
- Each service owns its data
- Services communicate via APIs
- No shared databases

**Example Services:**
- **Product Catalog Service**: Product information, search
- **Shopping Cart Service**: Cart management
- **Order Service**: Order processing
- **Payment Service**: Payment processing
- **Shipping Service**: Delivery management
- **Recommendation Service**: Product recommendations

**Decomposition Strategy:**
- **By Business Capability**: Each service handles one business function
- **By Data Ownership**: Each service owns its data
- **By Team**: Two-pizza team rule (small teams)

**API Gateway Pattern:**
- Single entry point for clients
- Routes requests to appropriate services
- Handles authentication, rate limiting
- Aggregates responses when needed

**Spotify - Team Structure and Service Boundaries:**

Spotify's architecture emphasizes team autonomy:

**Squad Model:**
- Small autonomous teams (squads)
- Each squad owns one or more services
- Squads have end-to-end responsibility
- Clear service boundaries

**Service Ownership:**
- Squad owns service development
- Squad owns service operations
- Squad decides technology stack
- Squad handles on-call

**Architecture Principles:**
- **Autonomy**: Teams can deploy independently
- **Alignment**: Shared infrastructure and standards
- **Accountability**: Teams responsible for their services

**Open-Source Microservices Examples:**

**Kubernetes Microservices:**
- Container orchestration for microservices
- Service discovery via DNS
- Load balancing built-in
- Health checks and auto-scaling

**Service Mesh (Istio/Linkerd):**
- Infrastructure layer for service communication
- Handles load balancing, retries, circuit breakers
- mTLS for security
- Observability (metrics, tracing)

**Key Takeaways:**
- Service decomposition by business capability
- API Gateway for external access
- Service mesh for inter-service communication
- Team structure aligns with service boundaries
- Independent deployment and scaling`,
					CodeExamples: `Netflix Microservices Architecture:

// User Service
@RestController
@RequestMapping("/api/users")
public class UserController {
    private final UserService userService;
    
    @GetMapping("/{userId}")
    public User getUser(@PathVariable String userId) {
        return userService.getUser(userId);
    }
}

// Recommendation Service (calls User Service)
@Service
public class RecommendationService {
    private final UserServiceClient userServiceClient;
    private final ContentServiceClient contentServiceClient;
    
    public List<Movie> getRecommendations(String userId) {
        User user = userServiceClient.getUser(userId);
        List<Movie> watchedMovies = userServiceClient.getWatchedMovies(userId);
        
        // ML-based recommendation logic
        return recommendationEngine.recommend(watchedMovies, user.getPreferences());
    }
}

// API Gateway (Zuul)
@EnableZuulProxy
@SpringBootApplication
public class ApiGatewayApplication {
    public static void main(String[] args) {
        SpringApplication.run(ApiGatewayApplication.class, args);
    }
}

// Zuul Routes Configuration
zuul:
  routes:
    user-service:
      path: /api/users/**
      url: http://user-service:8080
    recommendation-service:
      path: /api/recommendations/**
      url: http://recommendation-service:8080

Amazon Service Decomposition:

// Product Catalog Service
@Service
public class ProductCatalogService {
    private final ProductRepository productRepository;
    
    public Product getProduct(String productId) {
        return productRepository.findById(productId);
    }
    
    public List<Product> searchProducts(String query) {
        return productRepository.search(query);
    }
}

// Shopping Cart Service (separate service)
@Service
public class ShoppingCartService {
    private final CartRepository cartRepository;
    private final ProductCatalogClient productCatalogClient;
    
    public void addItem(String cartId, String productId, int quantity) {
        // Verify product exists via Product Catalog Service
        Product product = productCatalogClient.getProduct(productId);
        
        Cart cart = cartRepository.findById(cartId);
        cart.addItem(productId, product.getPrice(), quantity);
        cartRepository.save(cart);
    }
}

// API Gateway Routes
routes:
  - path: /api/products/**
    service: product-catalog-service
  - path: /api/cart/**
    service: shopping-cart-service
  - path: /api/orders/**
    service: order-service

Spotify Squad Model:

// Each squad owns a service
// Example: Playback Squad owns Playback Service

@Service
public class PlaybackService {
    // Owned by Playback Squad
    // Squad decides: technology, deployment, on-call
}

// Service Mesh Configuration (Istio)
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: playback-service
spec:
  hosts:
  - playback-service
  http:
  - route:
    - destination:
        host: playback-service
      weight: 100

Service Discovery Example:

// Eureka Service Discovery (Netflix)
@EnableEurekaClient
@SpringBootApplication
public class UserServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(UserServiceApplication.class, args);
    }
}

// Service Client with Discovery
@Service
public class RecommendationServiceClient {
    @Autowired
    private RestTemplate restTemplate;
    
    @Autowired
    private DiscoveryClient discoveryClient;
    
    public List<Movie> getRecommendations(String userId) {
        // Service discovery
        List<ServiceInstance> instances = discoveryClient.getInstances("recommendation-service");
        String serviceUrl = instances.get(0).getUri().toString();
        
        return restTemplate.getForObject(serviceUrl + "/recommendations/" + userId, List.class);
    }
}

Circuit Breaker Example (Hystrix):

@Service
public class RecommendationServiceClient {
    @HystrixCommand(fallbackMethod = "getDefaultRecommendations")
    public List<Movie> getRecommendations(String userId) {
        return restTemplate.getForObject("http://recommendation-service/recommendations/" + userId, List.class);
    }
    
    public List<Movie> getDefaultRecommendations(String userId) {
        // Fallback when recommendation service is down
        return getPopularMovies();
    }
}`,
				},
				{
					Title: "Real Architectural Decision Records",
					Content: `Examining real ADRs from major open-source projects provides valuable insights into architectural decision-making.

**Kubernetes ADRs:**

Kubernetes project maintains ADRs documenting key architectural decisions:

**Example ADR: API Versioning Strategy**

**Context:**
Kubernetes needed a versioning strategy for its API to support evolution while maintaining compatibility.

**Decision:**
- Use API versioning (v1, v1beta1, etc.)
- Support multiple API versions simultaneously
- Deprecation policy: Support deprecated versions for at least 2 releases
- Graduation path: alpha → beta → stable

**Consequences:**
- ✅ Enables API evolution
- ✅ Maintains backward compatibility
- ✅ Clear deprecation process
- ❌ Increased complexity
- ❌ Multiple code paths to maintain

**Example ADR: Container Runtime Interface (CRI)**

**Context:**
Kubernetes needed to support multiple container runtimes (Docker, containerd, CRI-O).

**Decision:**
- Create Container Runtime Interface (CRI)
- Abstract container runtime operations
- Support pluggable runtimes

**Consequences:**
- ✅ Runtime flexibility
- ✅ Innovation in container runtimes
- ✅ Reduced vendor lock-in
- ❌ Additional abstraction layer
- ❌ More complex code

**Prometheus ADRs:**

Prometheus project documents architectural decisions:

**Example ADR: Time Series Database Design**

**Context:**
Prometheus needed efficient storage for time-series data.

**Decision:**
- Use custom time-series database
- Store data in chunks
- Compression for efficiency
- Retention policies

**Consequences:**
- ✅ Efficient storage
- ✅ Fast queries
- ✅ Scalable
- ❌ Custom solution (more maintenance)
- ❌ Learning curve

**Example ADR: PromQL Query Language**

**Context:**
Prometheus needed a query language for time-series data.

**Decision:**
- Design PromQL (Prometheus Query Language)
- Functional, expression-based language
- Support for aggregation, functions, operators

**Consequences:**
- ✅ Powerful querying
- ✅ Expressive language
- ✅ Well-documented
- ❌ Learning curve
- ❌ Different from SQL

**Docker ADRs:**

Docker project maintains ADRs:

**Example ADR: Client-Server Architecture**

**Context:**
Docker needed to separate client and daemon for security and distribution.

**Decision:**
- Client-server architecture
- Docker CLI as client
- Docker daemon as server
- REST API for communication

**Consequences:**
- ✅ Security (daemon runs with privileges)
- ✅ Remote access capability
- ✅ Separation of concerns
- ❌ Network overhead
- ❌ More complex deployment

**Key Patterns from Real ADRs:**

**1. Clear Context:**
- Describe the problem clearly
- Explain constraints and requirements
- Provide background information

**2. Explicit Decision:**
- State decision clearly
- Explain what was chosen
- Document rationale

**3. Honest Consequences:**
- List both positive and negative
- Acknowledge trade-offs
- Identify risks

**4. Alternatives Considered:**
- Document rejected options
- Explain why they were rejected
- Show decision process

**5. Evolution:**
- ADRs can be superseded
- Decisions evolve over time
- Document changes

**Key Takeaways:**
- ADRs capture important decisions
- Real ADRs show honest trade-offs
- Decisions evolve over time
- Documentation helps future understanding`,
					CodeExamples: `Kubernetes ADR Example:

ADR: ADR-001: API Versioning Strategy

  Status
Accepted

  Context
Kubernetes API needs to evolve while maintaining backward compatibility.
Users depend on stable APIs, but we need to add features and fix issues.

  Decision
We will use API versioning with the following strategy:
- API versions: v1 (stable), v1beta1, v1alpha1 (experimental)
- Support multiple versions simultaneously
- Deprecation policy: Support deprecated versions for at least 2 releases
- Graduation: alpha → beta → stable (requires API review)

  Consequences

    Positive
- Enables API evolution
- Maintains backward compatibility
- Clear deprecation process
- Users can migrate gradually

    Negative
- Increased code complexity
- Multiple code paths to maintain
- More testing required
- Potential confusion for users

    Risks
- Deprecated APIs may be used too long
- Migration burden on users
- Breaking changes if not careful

  Alternatives Considered
1. **No versioning**: Rejected - breaks backward compatibility
2. **Single version**: Rejected - prevents evolution
3. **Semantic versioning**: Rejected - too complex for API

Prometheus ADR Example:

ADR: ADR-002: Time Series Database Design

  Status
Accepted

  Context
Prometheus needs efficient storage for millions of time series.
Standard databases don't optimize for time-series workloads.

  Decision
We will build a custom time-series database with:
- Chunk-based storage
- Compression (Gorilla encoding)
- Retention policies
- Write-optimized design

  Consequences

    Positive
- Highly efficient storage
- Fast queries for time-series data
- Scalable to millions of series
- Optimized for Prometheus use case

    Negative
- Custom solution requires maintenance
- Learning curve for contributors
- Less flexible than general-purpose DB
- Migration complexity if we change

    Risks
- May not scale beyond certain point
- Maintenance burden
- Vendor lock-in (our own solution)

  Alternatives Considered
1. **PostgreSQL**: Rejected - not optimized for time-series
2. **InfluxDB**: Rejected - different data model
3. **Cassandra**: Rejected - too complex for our needs

Docker ADR Example:

ADR: ADR-003: Client-Server Architecture

  Status
Accepted

  Context
Docker needs to separate client and daemon for security and remote access.
Daemon requires root privileges, but client should not.

  Decision
We will use client-server architecture:
- Docker CLI as client (no privileges)
- Docker daemon as server (with privileges)
- REST API for communication
- Unix socket or TCP for transport

  Consequences

    Positive
- Security: Client doesn't need privileges
- Remote access capability
- Separation of concerns
- Can update client independently

    Negative
- Network overhead
- More complex deployment
- Potential security issues if exposed
- Requires daemon to be running

    Risks
- Network security if exposed
- Daemon becomes single point of failure
- API versioning complexity

  Alternatives Considered
1. **Monolithic**: Rejected - security concerns
2. **Library-based**: Rejected - no remote access
3. **RPC instead of REST**: Rejected - REST is more standard

ADR Evolution Example:

ADR: ADR-004: Container Runtime Interface

  Status
Superseded by ADR-005

  Context
[Original context...]

  Decision
[Original decision...]

  Consequences
[Original consequences...]

  Superseded By
ADR-005: Enhanced Container Runtime Interface

  Notes
This ADR was superseded to support additional runtime features.

Key ADR Patterns:

1. **Clear Problem Statement:**
"The system needs X because Y, but faces constraint Z"

2. **Explicit Decision:**
"We will use approach A with principles B, C, D"

3. **Honest Trade-offs:**
- Positive: What we gain
- Negative: What we lose
- Risks: What could go wrong

4. **Alternatives:**
- What else was considered
- Why it was rejected
- What we learned`,
				},
				{
					Title: "Open-Source Architecture Examples",
					Content: `Analyzing successful open-source projects reveals common architectural patterns and design principles.

**Kubernetes - Microservices Orchestration:**

Kubernetes demonstrates microservices orchestration architecture:

**Architecture Overview:**
- **Control Plane**: API server, etcd, scheduler, controller manager
- **Node Components**: kubelet, kube-proxy, container runtime
- **Add-ons**: DNS, networking, monitoring

**Key Patterns:**
- **API-Driven**: Everything through API server
- **Declarative**: Desired state vs. current state
- **Controller Pattern**: Controllers reconcile state
- **Plugin Architecture**: Extensible via plugins

**Design Principles:**
- Separation of concerns (control plane vs. data plane)
- Loose coupling (components communicate via API)
- Extensibility (CRDs, operators, plugins)
- Resilience (self-healing, health checks)

**Docker - Client-Server Architecture:**

Docker demonstrates clean client-server separation:

**Architecture:**
- **Docker CLI**: Client (user interface)
- **Docker Daemon**: Server (container management)
- **REST API**: Communication protocol
- **Containerd**: Runtime abstraction

**Key Patterns:**
- **Client-Server**: Clear separation
- **REST API**: Standard communication
- **Plugin System**: Extensible architecture
- **Layered Storage**: Union filesystem

**Design Principles:**
- Security (daemon has privileges, client doesn't)
- Remote access (can connect to remote daemon)
- Extensibility (plugins for volumes, networks)
- Simplicity (simple CLI commands)

**Git - Distributed Version Control:**

Git demonstrates distributed system architecture:

**Architecture:**
- **Local Repository**: Complete copy of history
- **Remote Repositories**: Distributed copies
- **Object Store**: Content-addressable storage
- **Index**: Staging area

**Key Patterns:**
- **Distributed**: No central server required
- **Content-Addressable**: SHA-1 hashing
- **Snapshot-based**: Full snapshots, not diffs
- **Branching**: Lightweight branches

**Design Principles:**
- Distributed (no single point of failure)
- Efficient (compression, delta encoding)
- Fast (local operations)
- Reliable (content-addressable, checksums)

**Redis - In-Memory Data Structure Store:**

Redis demonstrates high-performance architecture:

**Architecture:**
- **In-Memory Storage**: Fast access
- **Persistence Options**: RDB snapshots, AOF
- **Replication**: Master-replica
- **Clustering**: Distributed mode

**Key Patterns:**
- **Single-Threaded**: Event loop model
- **Data Structures**: Rich data types
- **Persistence**: Optional durability
- **Pub/Sub**: Messaging capability

**Design Principles:**
- Performance (in-memory, single-threaded)
- Simplicity (simple data model)
- Flexibility (multiple data structures)
- Reliability (persistence, replication)

**Common Patterns Across Projects:**

**1. Plugin/Extension Architecture:**
- Kubernetes: CRDs, operators, CNI plugins
- Docker: Volume plugins, network plugins
- Git: Hooks, extensions
- Redis: Modules

**2. API-First Design:**
- Kubernetes: REST API for everything
- Docker: REST API for daemon
- Git: Plumbing commands (API-like)
- Redis: RESP protocol

**3. Layered Architecture:**
- Kubernetes: Control plane, data plane
- Docker: CLI, API, daemon, runtime
- Git: Porcelain, plumbing
- Redis: Network, protocol, storage

**4. Extensibility:**
- All projects support extensions
- Plugin systems
- API-based integration
- Community contributions

**Key Takeaways:**
- Successful projects use clear architectural patterns
- Extensibility is crucial
- API-first design enables integration
- Layered architecture provides separation`,
					CodeExamples: `Kubernetes Architecture:

// API Server (Control Plane)
type APIServer struct {
    storage etcd.Storage
    admissionControllers []AdmissionController
}

func (s *APIServer) CreatePod(pod *Pod) error {
    // Validate
    if err := s.validate(pod); err != nil {
        return err
    }
    
    // Admission control
    for _, controller := range s.admissionControllers {
        if err := controller.Admit(pod); err != nil {
            return err
        }
    }
    
    // Store in etcd
    return s.storage.Create("pods", pod)
}

// Controller Pattern
type ReplicaSetController struct {
    client kubernetes.Interface
}

func (c *ReplicaSetController) Reconcile(rs *ReplicaSet) {
    // Get current pods
    pods := c.client.GetPods(rs.Spec.Selector)
    
    // Desired count
    desired := rs.Spec.Replicas
    
    // Current count
    current := len(pods)
    
    // Reconcile
    if current < desired {
        c.createPods(rs, desired-current)
    } else if current > desired {
        c.deletePods(pods[:current-desired])
    }
}

Docker Architecture:

// Docker Client
type DockerClient struct {
    client *http.Client
    host   string
}

func (c *DockerClient) CreateContainer(config ContainerConfig) (*Container, error) {
    url := c.host + "/containers/create"
    resp, err := c.client.Post(url, "application/json", config)
    // Handle response
}

// Docker Daemon
type DockerDaemon struct {
    containers map[string]*Container
    images     ImageStore
    runtime    Runtime
}

func (d *DockerDaemon) CreateContainer(config ContainerConfig) (*Container, error) {
    // Pull image if needed
    image := d.images.Get(config.Image)
    
    // Create container
    container := d.runtime.CreateContainer(image, config)
    d.containers[container.ID] = container
    
    return container, nil
}

Git Architecture:

// Git Object Store (Content-Addressable)
type ObjectStore struct {
    objects map[string]Object
}

func (s *ObjectStore) Store(obj Object) string {
    // Calculate SHA-1 hash
    hash := sha1.Sum(obj.Content())
    
    // Store by hash
    key := hex.EncodeToString(hash[:])
    s.objects[key] = obj
    
    return key
}

func (s *ObjectStore) Get(hash string) Object {
    return s.objects[hash]
}

// Git Repository
type Repository struct {
    objects ObjectStore
    refs    RefStore
    index   Index
}

func (r *Repository) Commit(message string) string {
    // Create tree from index
    tree := r.index.ToTree()
    
    // Create commit object
    commit := NewCommit(tree, message)
    commitHash := r.objects.Store(commit)
    
    // Update HEAD
    r.refs.Update("HEAD", commitHash)
    
    return commitHash
}

Redis Architecture:

// Redis Server (Single-Threaded Event Loop)
type RedisServer struct {
    db      *Database
    events  EventLoop
    clients map[int]*Client
}

func (s *RedisServer) Start() {
    // Event loop
    for {
        event := s.events.Poll()
        
        switch e := event.(type) {
        case *ClientConnect:
            s.handleConnect(e.Client)
        case *ClientCommand:
            s.handleCommand(e.Client, e.Command)
        case *ClientDisconnect:
            s.handleDisconnect(e.Client)
        }
    }
}

func (s *RedisServer) handleCommand(client *Client, cmd Command) {
    // Execute command (single-threaded)
    result := s.db.Execute(cmd)
    client.Send(result)
}

// Redis Database
type Database struct {
    data map[string]*Object
}

func (db *Database) Set(key string, value *Object) {
    db.data[key] = value
}

func (db *Database) Get(key string) *Object {
    return db.data[key]
}

Common Patterns:

// Plugin Interface (Kubernetes)
type CNIPlugin interface {
    AddNetwork(network *Network) error
    DeleteNetwork(network *Network) error
}

// Plugin Registration
func RegisterCNIPlugin(name string, plugin CNIPlugin) {
    plugins[name] = plugin
}

// API-First Design (Docker)
type ContainerAPI interface {
    CreateContainer(config ContainerConfig) (*Container, error)
    StartContainer(id string) error
    StopContainer(id string) error
    RemoveContainer(id string) error
}

// Layered Architecture (Git)
// Porcelain Layer (User-facing)
func GitCommit(message string) {
    // Plumbing Layer (Internal)
    repository.Commit(message)
}`,
				},
				{
					Title: "Architecture Evolution Case Studies",
					Content: `Examining how major systems evolved their architectures provides valuable lessons for architects.

**Twitter - Monolith to Microservices:**

Twitter's evolution demonstrates a successful migration:

**Phase 1: Monolith (2006-2010)**
- Ruby on Rails monolith
- Single codebase
- Shared database
- Difficult to scale

**Challenges:**
- "Fail whale" (outages)
- Difficult to scale specific features
- Slow deployments
- Team coordination issues

**Phase 2: Service-Oriented (2010-2012)**
- Started extracting services
- Timeline service separated
- User service separated
- Still some shared components

**Phase 3: Microservices (2012-Present)**
- Hundreds of microservices
- Each service owns its data
- Independent deployment
- Service mesh for communication

**Migration Strategy:**
- **Strangler Pattern**: Gradually replace monolith
- **Extract Services**: Move functionality to services
- **Dual Write**: Write to both old and new systems
- **Cutover**: Switch traffic gradually

**Lessons Learned:**
- Start with high-value services
- Use strangler pattern
- Independent data stores
- Service boundaries are critical

**Instagram - Django Monolith Evolution:**

Instagram's evolution shows monolith can scale:

**Phase 1: Django Monolith (2010-2012)**
- Single Django application
- PostgreSQL database
- Vertical scaling initially

**Phase 2: Horizontal Scaling (2012-2014)**
- Multiple application servers
- Database read replicas
- Caching layer (Redis)
- CDN for media

**Phase 3: Microservices (2014-Present)**
- Extracted services gradually
- Feed service separated
- Media service separated
- Notification service separated

**Key Decisions:**
- Kept monolith for core features
- Extracted services for scaling needs
- Used Python across services
- Maintained team structure

**Lessons Learned:**
- Monolith can scale with right patterns
- Extract services when needed
- Don't over-engineer early
- Team structure matters

**Amazon - Service-Oriented Architecture Evolution:**

Amazon's evolution demonstrates service-oriented thinking:

**Phase 1: Monolith (1995-2001)**
- Single application
- Shared database
- All teams work on same codebase

**Phase 2: Service-Oriented (2001-2006)**
- Started extracting services
- "Two-pizza teams" concept
- Services own their data
- API-first approach

**Phase 3: Microservices (2006-Present)**
- Hundreds of services
- Independent deployment
- Service mesh
- Event-driven architecture

**Key Principles:**
- **Service Ownership**: Teams own services end-to-end
- **Data Ownership**: Services own their data
- **API Contracts**: Well-defined interfaces
- **Decentralization**: Teams make decisions

**Migration Patterns:**

**1. Strangler Pattern:**
- Build new system alongside old
- Gradually route traffic to new system
- Eventually retire old system

**2. Database Decomposition:**
- Identify service boundaries
- Extract data to service databases
- Use dual write during transition
- Cutover when ready

**3. Feature Flags:**
- Use flags to control new vs. old
- Gradual rollout
- Easy rollback
- A/B testing

**4. Event Sourcing:**
- Capture events from monolith
- Build new services from events
- Gradually replace functionality

**Common Challenges:**

**1. Data Migration:**
- Moving data between systems
- Maintaining consistency
- Handling large datasets
- Zero-downtime migration

**2. Service Boundaries:**
- Identifying right boundaries
- Avoiding over-decomposition
- Managing dependencies
- Handling shared data

**3. Team Coordination:**
- Coordinating across teams
- Managing dependencies
- Communication overhead
- Knowledge sharing

**4. Operational Complexity:**
- More services to manage
- Distributed system challenges
- Monitoring and debugging
- Deployment complexity

**Key Takeaways:**
- Start simple, evolve as needed
- Use strangler pattern for migration
- Service boundaries are critical
- Team structure aligns with architecture
- Migration is gradual, not big bang`,
					CodeExamples: `Twitter Migration Strategy:

// Phase 1: Monolith
class TwitterMonolith {
    def createTweet(userId, content) {
        // Everything in one place
        user = User.find(userId)
        tweet = Tweet.create(user: user, content: content)
        Timeline.addToFollowers(user, tweet)
        Notification.send(user.followers, tweet)
        return tweet
    }
}

// Phase 2: Extract Timeline Service
class TimelineService {
    def addToTimeline(userId, tweet) {
        // New service
        timeline = Timeline.findOrCreate(userId)
        timeline.add(tweet)
        timeline.save()
    }
}

// Phase 3: Strangler Pattern
class TwitterMonolith {
    def createTweet(userId, content) {
        user = User.find(userId)
        tweet = Tweet.create(user: user, content: content)
        
        // Dual write: old and new
        Timeline.addToFollowers(user, tweet)  // Old
        timelineService.addToTimeline(user.followers, tweet)  // New
        
        Notification.send(user.followers, tweet)
        return tweet
    }
}

// Phase 4: Cutover
class TwitterMonolith {
    def createTweet(userId, content) {
        user = User.find(userId)
        tweet = Tweet.create(user: user, content: content)
        
        // Only new service
        timelineService.addToTimeline(user.followers, tweet)
        
        Notification.send(user.followers, tweet)
        return tweet
    }
}

Instagram Evolution:

// Phase 1: Django Monolith
class Post(models.Model):
    user = models.ForeignKey(User)
    image = models.ImageField()
    caption = models.TextField()
    
    def save(self):
        super().save()
        # Everything in one place
        Feed.addToFollowers(self.user, self)
        Notification.send(self.user.followers, self)

// Phase 2: Extract Feed Service
class FeedService:
    def addToFeed(self, userId, post):
        # New service
        followers = User.objects.filter(following=userId)
        for follower in followers:
            Feed.objects.create(user=follower, post=post)

// Phase 3: Microservices
class PostService:
    def create_post(self, userId, image, caption):
        post = Post.create(userId, image, caption)
        
        # Publish event
        eventBus.publish(PostCreatedEvent(post))
        return post

class FeedService:
    @event_handler(PostCreatedEvent)
    def handle_post_created(self, event):
        # Handle event asynchronously
        self.addToFollowers(event.post.userId, event.post)

Amazon Service Extraction:

// Phase 1: Monolith
class OrderService:
    def create_order(self, customerId, items):
        # Everything together
        customer = Customer.find(customerId)
        order = Order.create(customer, items)
        Payment.charge(customer, order.total)
        Inventory.reserve(items)
        Shipping.ship(order)
        return order

// Phase 2: Extract Services
class OrderService:
    def create_order(self, customerId, items):
        customer = customerService.get(customerId)
        order = Order.create(customer, items)
        
        # Call other services
        paymentService.charge(customer.id, order.total)
        inventoryService.reserve(items)
        shippingService.ship(order.id)
        
        return order

// Phase 3: Event-Driven
class OrderService:
    def create_order(self, customerId, items):
        customer = customerService.get(customerId)
        order = Order.create(customer, items)
        
        # Publish event
        eventBus.publish(OrderCreatedEvent(order))
        return order

class PaymentService:
    @event_handler(OrderCreatedEvent)
    def handle_order_created(self, event):
        self.charge(event.order.customerId, event.order.total)

Database Decomposition:

// Before: Shared Database
class OrderService:
    def create_order(self, customerId, items):
        # Direct database access
        customer = db.query("SELECT * FROM customers WHERE id = ?", customerId)
        order = db.execute("INSERT INTO orders ...")
        return order

// After: Service Database
class OrderService:
    def __init__(self):
        self.orderDb = OrderDatabase()
        self.customerService = CustomerServiceClient()
    
    def create_order(self, customerId, items):
        # Call customer service
        customer = self.customerService.get(customerId)
        
        # Use own database
        order = self.orderDb.create(customerId, items)
        return order

Feature Flags:

class OrderService:
    def create_order(self, customerId, items):
        order = Order.create(customerId, items)
        
        # Feature flag
        if featureFlags.isEnabled("new_timeline_service"):
            timelineService.add(order)
        else:
            Timeline.add(order)  # Old way
        
        return order

Event Sourcing Migration:

// Capture events from monolith
class EventCapture:
    def capture_order_created(self, order):
        event = OrderCreatedEvent(
            orderId=order.id,
            customerId=order.customerId,
            items=order.items,
            timestamp=datetime.now()
        )
        eventStore.append(event)

// Build new service from events
class NewOrderService:
    def rebuild_state(self):
        events = eventStore.get_events("OrderCreatedEvent")
        for event in events:
            # Rebuild state
            order = self.apply_event(event)
            self.orders[order.id] = order`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
