package systemsdesign

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSystemsDesignModules([]problems.CourseModule{
		{
			ID:          15,
			Title:       "Microservices Architecture",
			Description: "Understand microservices patterns, service communication, and distributed system challenges.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "Microservices Fundamentals",
					Content: `Microservices architecture structures an application as a collection of loosely coupled services.

**Characteristics:**
- Services are independently deployable
- Each service has its own database
- Services communicate via APIs
- Organized around business capabilities
- Small, focused teams per service

**Benefits:**
- Independent scaling
- Technology diversity
- Fault isolation
- Faster development
- Easier to understand

**Challenges:**
- Network latency
- Data consistency
- Service coordination
- Testing complexity
- Deployment complexity

**When to Use:**
- Large, complex applications
- Multiple teams
- Need for independent scaling
- Different technology requirements

**When NOT to Use:**
- Small applications
- Simple CRUD apps
- Tight coupling required
- Limited team size`,
					CodeExamples: `Microservices Architecture with API Gateway and Service Mesh:

                       ┌──────────────┐
                       │  External    │
                       │  Clients     │
                       └──────┬───────┘
                              │ (HTTPS / REST / gRPC)
                       ┌──────▼───────┐
                       │ API Gateway  │
                       │ (Auth, Routing, Aggregation) │
                       └──────┬───────┘
                              │
                  ┌───────────┴───────────┐
                  │                       │
          ┌───────▼───────┐       ┌───────▼───────┐
          │  Service A    │       │  Service B    │
          │ (with sidecar │       │ (with sidecar │
          │  proxy)       │       │  proxy)       │
          └───────┬───────┘       └───────┬───────┘
                  │                       │
         (Internal HTTP/gRPC via service mesh)
                  │                       │
        ┌─────────▼─────────┐             │
        │   Kafka Cluster   │◄────────────┘
        │ (Topics, Brokers) │
        └─────────┬─────────┘
                  │
        ┌─────────▼──────────┐
        │ Consumer Services  │
        │ (Notification,    │
        │  Inventory, etc.)  │
        └────────────────────┘

- API Gateway handles external requests (auth, routing)
- Service Mesh manages internal service-to-service communication
- Kafka enables asynchronous event-driven communication`,
				},
				{
					Title: "Service Communication",
					Content: `**Synchronous Communication:**
- HTTP/REST: Request-response pattern
- gRPC: High-performance RPC framework
- GraphQL: Query language for APIs

**Asynchronous Communication:**
- Message queues (RabbitMQ, Kafka)
- Event-driven architecture
- Pub/sub patterns

**API Gateway:**
- Single entry point for clients
- Routes requests to services
- Handles authentication, rate limiting
- Load balancing
- Request/response transformation

**Service Discovery:**
- Services register themselves
- Clients discover service locations
- Handles dynamic IPs
- Examples: Consul, Eureka, Kubernetes DNS

**Service Mesh:**
- Infrastructure layer for service communication
- Handles load balancing, retries, circuit breakers
- Examples: Istio, Linkerd`,
					CodeExamples: `Service Communication Patterns:

Synchronous (Request-Response):
Client → API Gateway → Service A → Service B → Response

Asynchronous (Event-Driven):
Service A → Kafka Topic → [Service B, Service C, Service D] (Consumers)

Service Discovery Flow:
1. Service starts and registers with Service Registry
2. Client queries Service Registry for service location
3. Client connects directly to service
4. Service Registry monitors health and updates registry

Service Mesh Components:
- Data Plane: Sidecar proxies (Envoy) intercept traffic
- Control Plane: Manages routing, security policies, observability
- Features: Load balancing, retries, circuit breakers, mTLS`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          16,
			Title:       "API Design & Rate Limiting",
			Description: "Learn RESTful API design principles, versioning, and rate limiting strategies.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "RESTful API Design",
					Content: `**REST Principles:**
- **Stateless**: Each request contains all information
- **Resource-based**: URLs represent resources
- **HTTP methods**: GET, POST, PUT, DELETE, PATCH
- **Representations**: JSON, XML, etc.

**Best Practices:**
- Use nouns for resources (not verbs)
- Use HTTP status codes correctly
- Version your APIs (/v1/, /v2/)
- Use pagination for large datasets
- Provide filtering and sorting
- Include error details

**API Versioning Strategies:**
- **URL versioning**: /api/v1/users
- **Header versioning**: Accept: application/vnd.api+json;version=1
- **Query parameter**: /api/users?version=1

**Error Handling:**
- Consistent error format
- Meaningful error messages
- Appropriate HTTP status codes
- Include error codes for programmatic handling`,
				},
				{
					Title: "Rate Limiting",
					Content: `Rate limiting controls the number of requests a client can make.

**Why Rate Limiting:**
- Prevent abuse
- Ensure fair usage
- Protect backend services
- Control costs

**Rate Limiting Algorithms:**

**Token Bucket:**
- Tokens added at fixed rate
- Request consumes token
- Reject if no tokens available

**Leaky Bucket:**
- Requests added to bucket
- Processed at fixed rate
- Reject if bucket full

**Fixed Window:**
- Count requests in time window
- Reset counter at window end
- Simple but allows bursts

**Sliding Window:**
- More accurate than fixed window
- Tracks requests in sliding time window
- Better distribution

**Rate Limiting Headers:**
- X-RateLimit-Limit: Request limit
- X-RateLimit-Remaining: Remaining requests
- X-RateLimit-Reset: Reset time`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			Title:       "Monitoring & Observability",
			Description: "Learn about logging, metrics, tracing, and monitoring distributed systems.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Monitoring Fundamentals",
					Content: `**Three Pillars of Observability:**

1. **Logs**: Discrete events with timestamps
   - What happened and when
   - Debugging and auditing
   - Examples: Application logs, access logs

2. **Metrics**: Numerical measurements over time
   - How the system is performing
   - Alerting and dashboards
   - Examples: CPU usage, request rate, error rate

3. **Traces**: Request flow through services
   - How requests flow through system
   - Performance debugging
   - Distributed system visibility

**Key Metrics to Monitor:**
- **Availability**: Uptime percentage
- **Latency**: Response time (p50, p95, p99)
- **Throughput**: Requests per second
- **Error Rate**: Percentage of failed requests
- **Resource Usage**: CPU, memory, disk, network

**Alerting:**
- Set thresholds for critical metrics
- Alert on anomalies
- Avoid alert fatigue
- Use different severity levels`,
				},
				{
					Title: "Distributed Tracing",
					Content: `Distributed tracing tracks requests across multiple services.

**Trace Components:**
- **Trace**: Entire request journey
- **Span**: Single operation within trace
- **Span Context**: Propagates trace information

**Benefits:**
- Understand request flow
- Identify bottlenecks
- Debug distributed issues
- Performance optimization

**Tracing Tools:**
- **Jaeger**: Open-source distributed tracing
- **Zipkin**: Distributed tracing system
- **AWS X-Ray**: AWS tracing service
- **OpenTelemetry**: Vendor-neutral observability

**Instrumentation:**
- Automatic: Framework/library support
- Manual: Add tracing code
- Sampling: Reduce overhead`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          18,
			Title:       "Design Patterns",
			Description: "Learn essential design patterns for distributed systems including Circuit Breaker, Bulkhead, and more.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "Circuit Breaker Pattern",
					Content: `Circuit breaker prevents cascading failures by stopping requests to failing services.

**States:**
- **Closed**: Normal operation, requests pass through
- **Open**: Service failing, requests rejected immediately
- **Half-Open**: Testing if service recovered

**Benefits:**
- Fail fast
- Prevent resource exhaustion
- Allow service recovery time
- Reduce load on failing service

**Implementation:**
- Track failure rate
- Open circuit when threshold exceeded
- After timeout, try half-open
- Close if successful, reopen if fails

**Use Cases:**
- External API calls
- Database connections
- Third-party services
- Any remote service calls`,
				},
				{
					Title: "Bulkhead Pattern",
					Content: `Bulkhead isolates resources to prevent total system failure.

**Concept:**
- Partition resources into isolated groups
- Failure in one partition doesn't affect others
- Like ship bulkheads preventing flooding

**Examples:**
- Separate thread pools per service
- Separate database connections
- Separate connection pools
- Isolated compute resources

**Benefits:**
- Fault isolation
- Better resource management
- Prevent cascading failures
- Independent scaling`,
				},
				{
					Title: "Other Important Patterns",
					Content: `**Retry Pattern:**
- Retry failed operations
- Exponential backoff
- Maximum retry attempts
- Idempotent operations

**Timeout Pattern:**
- Set timeouts for operations
- Fail fast if timeout exceeded
- Prevent hanging requests

**Saga Pattern:**
- Manage distributed transactions
- Sequence of local transactions
- Compensating transactions for rollback

**CQRS (Command Query Responsibility Segregation):**
- Separate read and write models
- Optimize for different use cases
- Better scalability

**Event Sourcing:**
- Store events instead of current state
- Rebuild state by replaying events
- Complete audit trail`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          19,
			Title:       "Real-World System Designs",
			Description: "Study real-world system designs including URL shorteners, chat systems, and social media platforms.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Design a URL Shortener",
					Content: `**Requirements:**
- Shorten long URLs
- Redirect to original URL
- Handle high traffic (100M URLs/day)
- Short URL should be as short as possible

**Design:**
- **Encoding**: Base62 encoding (a-z, A-Z, 0-9)
- **Database**: Store mapping (short → long URL)
- **Cache**: Cache popular URLs in Redis
- **Load Balancer**: Distribute traffic
- **Database Sharding**: Shard by short URL hash

**Key Considerations:**
- Collision handling
- Expiration of URLs
- Analytics tracking
- Custom short URLs`,
				},
				{
					Title: "Design a Chat System",
					Content: `**Requirements:**
- One-on-one messaging
- Group messaging
- Real-time delivery
- Message history
- Online/offline status

**Design:**
- **WebSocket**: Real-time bidirectional communication
- **Message Queue**: Store messages (Kafka)
- **Database**: Store message history
- **Presence Service**: Track online/offline status
- **Load Balancer**: Distribute WebSocket connections

**Challenges:**
- Maintaining WebSocket connections
- Message ordering
- Delivery guarantees
- Scaling WebSocket servers`,
				},
				{
					Title: "Design a Social Media Feed",
					Content: `**Requirements:**
- User timeline (posts from followed users)
- News feed (top posts)
- Handle millions of users
- Real-time updates

**Approaches:**

**Pull Model (Fan-out on Read):**
- User requests feed
- Fetch posts from followed users
- Merge and sort
- Simple but slow for users with many follows

**Push Model (Fan-out on Write):**
- When user posts, push to all followers' feeds
- Fast reads
- Slow writes for users with many followers
- Storage intensive

**Hybrid Approach:**
- Push for active users
- Pull for inactive users
- Best of both worlds`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
