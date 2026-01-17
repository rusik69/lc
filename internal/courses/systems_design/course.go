package systemsdesign

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSystemsDesignModules([]problems.CourseModule{
		{
			ID:          10,
			Title:       "Introduction to Systems Design",
			Description: "Learn the fundamentals of designing scalable, reliable, and maintainable systems.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "What is Systems Design?",
					Content: `Systems design is the process of defining the architecture, components, modules, interfaces, and data for a system to satisfy specified requirements. It's about making trade-offs between different aspects like performance, scalability, reliability, and cost.

**Key Goals:**
- **Scalability**: System can handle growth in users, data, or traffic
- **Reliability**: System works correctly and consistently
- **Availability**: System is accessible when needed (uptime)
- **Performance**: System responds quickly to requests
- **Maintainability**: System is easy to understand, modify, and extend

**Why Systems Design Matters:**
- Design systems that can scale to millions of users
- Make informed architectural decisions
- Understand trade-offs between different approaches
- Prepare for technical interviews at top tech companies
- Build production-ready systems

**Common System Requirements:**
- Functional requirements: What the system should do
- Non-functional requirements: Performance, scalability, availability
- Constraints: Budget, time, technology stack
- Assumptions: Expected usage patterns, growth rates`,
				},
				{
					Title: "Scalability Fundamentals",
					Content: `Scalability is the ability of a system to handle increased load by adding resources. There are two main approaches:

**Vertical Scaling (Scale Up):**
- Add more power to existing machines (CPU, RAM, storage)
- Simpler to implement
- Limited by hardware constraints
- Single point of failure
- Example: Upgrade server from 8GB to 32GB RAM

**Horizontal Scaling (Scale Out):**
- Add more machines to the system
- More complex but more flexible
- Can scale almost infinitely
- Better fault tolerance
- Example: Add 10 more servers to handle load

**Load Types:**
- **Read-heavy**: More reads than writes (e.g., social media feeds)
- **Write-heavy**: More writes than reads (e.g., logging systems)
- **Balanced**: Similar read/write ratio

**Scaling Strategies:**
- **Stateless services**: Easy to scale horizontally
- **Stateful services**: Require careful design (session management, state replication)
- **Database scaling**: Read replicas, sharding, caching`,
					CodeExamples: `Scaling Architecture Evolution:

1. Single Server:
   Client → [Single Server + Database]

2. Vertical Scaling:
   Client → [Larger Server (more CPU/RAM) + Database]

3. Horizontal Scaling with Load Balancer:
   Client → [Load Balancer] → [Server 1, Server 2, Server 3] → Database

4. Horizontal Scaling with Read Replicas:
   Client → [Load Balancer] → [Server 1, Server 2, Server 3]
                                    ↓
                            [Master DB] → [Replica 1, Replica 2]

5. Full Horizontal Scaling (Sharding):
   Client → [Load Balancer] → [Server 1, Server 2, Server 3]
                                    ↓
                            [Shard Router] → [Shard 1, Shard 2, Shard 3]
                                                 ↓
                                         [Each with Replicas]`,
				},
				{
					Title: "Availability & Reliability",
					Content: `**Availability** measures the percentage of time a system is operational and accessible.

**Availability Metrics:**
- 99% availability = 3.65 days downtime/year
- 99.9% (three nines) = 8.76 hours downtime/year
- 99.99% (four nines) = 52.56 minutes downtime/year
- 99.999% (five nines) = 5.26 minutes downtime/year

**Reliability** is the probability that a system will perform correctly for a given time period.

**High Availability Patterns:**
- **Redundancy**: Multiple instances of components
- **Failover**: Automatic switching to backup systems
- **Load balancing**: Distribute load across multiple servers
- **Health checks**: Monitor system health and remove unhealthy instances
- **Circuit breakers**: Prevent cascading failures

**Common Failure Scenarios:**
- Server crashes
- Network partitions
- Database failures
- Third-party service outages
- Data center outages

**Designing for Failure:**
- Assume failures will happen
- Design graceful degradation
- Implement retry mechanisms with exponential backoff
- Use idempotent operations
- Implement proper error handling`,
				},
				{
					Title: "Consistency Models",
					Content: `Consistency refers to how up-to-date and synchronized data appears across all nodes in a distributed system.

**ACID Properties (Strong Consistency):**
- **Atomicity**: All or nothing - transactions complete fully or not at all
- **Consistency**: Database remains in valid state after transaction
- **Isolation**: Concurrent transactions don't interfere
- **Durability**: Committed data persists even after crashes

**CAP Theorem:**
You can only guarantee two out of three:
- **Consistency**: All nodes see same data simultaneously
- **Availability**: System remains operational
- **Partition Tolerance**: System continues despite network failures

**Consistency Levels:**
- **Strong Consistency**: All reads receive most recent write
- **Weak Consistency**: System doesn't guarantee when updates will be reflected
- **Eventual Consistency**: System will become consistent over time

**When to Use Each:**
- **Strong Consistency**: Financial systems, critical data
- **Eventual Consistency**: Social media feeds, DNS, CDN content
- **Weak Consistency**: Real-time analytics, logging`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          11,
			Title:       "Load Balancing & High Availability",
			Description: "Master load balancing strategies and high availability patterns for distributed systems.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Load Balancing Fundamentals",
					Content: `Load balancing distributes incoming network traffic across multiple servers to ensure no single server is overwhelmed.

**Benefits:**
- Distribute traffic evenly
- Improve response times
- Increase availability
- Handle traffic spikes
- Enable zero-downtime deployments

**Load Balancing Algorithms:**

1. **Round Robin**: Distribute requests sequentially
   - Simple and fair
   - Doesn't consider server load

2. **Least Connections**: Send to server with fewest active connections
   - Good for long-lived connections
   - More complex to track

3. **Weighted Round Robin**: Round robin with server capacity weights
   - Accounts for server differences
   - Better resource utilization

4. **IP Hash**: Route based on client IP hash
   - Ensures same client goes to same server
   - Good for session affinity

5. **Geographic**: Route based on client location
   - Reduces latency
   - Better user experience

**Load Balancer Types:**
- **Layer 4 (Transport)**: Routes based on IP and port
- **Layer 7 (Application)**: Routes based on HTTP headers, URLs, cookies`,
					CodeExamples: `Load Balancer Architecture:

             +--------------------+
             |   Clients / Users  |
             +---------+----------+
                       |
                       v
                +--------------+
                | Load Balancer|
                +--------------+
                /      |       \\
               /       |        \\
              v        v         v
   +----------------+   +----------------+   +----------------+
   |  App Server 1  |   |  App Server 2  |   |  App Server 3  |
   +----------------+   +----------------+   +----------------+
          |                  |                   |
          v                  v                   v
     +---------+        +---------+         +---------+
     |  Cache  |        |  Cache  |         |  Cache  |
     +---------+        +---------+         +---------+
          |                  |                   |
          v                  v                   v
     +----------------------------------------------+
     |                 Database Cluster             |
     +----------------------------------------------+

- Clients send requests to the Load Balancer
- LB distributes traffic to multiple App Servers
- App servers may use local/global cache, then access database cluster`,
				},
				{
					Title: "High Availability Patterns",
					Content: `**Active-Passive (Hot Standby):**
- Primary handles all traffic
- Standby ready to take over
- Failover time: seconds to minutes
- Use case: Critical systems

**Active-Active:**
- All servers handle traffic
- Better resource utilization
- Faster failover
- More complex to manage

**Multi-Region Deployment:**
- Deploy in multiple geographic regions
- Route traffic to nearest region
- Failover to backup region if primary fails
- Higher cost but better availability

**Health Checks:**
- **Liveness**: Is the service running?
- **Readiness**: Can the service handle requests?
- **Startup**: Has the service finished initializing?

**Graceful Degradation:**
- System continues with reduced functionality
- Better than complete failure
- Example: Show cached data when database is down`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          12,
			Title:       "Caching Strategies",
			Description: "Learn caching techniques including Redis, Memcached, and CDN to improve system performance.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "Caching Fundamentals",
					Content: `Caching stores frequently accessed data in fast storage to reduce latency and database load.

**Cache Benefits:**
- Reduce latency (memory access vs disk/database)
- Reduce database load
- Improve user experience
- Handle traffic spikes better

**Cache Types:**

1. **Application Cache**: In-memory cache in application
   - Fastest access
   - Limited by server memory
   - Lost on restart

2. **Distributed Cache**: Shared cache across servers (Redis, Memcached)
   - Shared across instances
   - Survives server restarts
   - Network latency overhead

3. **CDN (Content Delivery Network)**: Cache static content at edge locations
   - Closest to users
   - Reduces origin server load
   - Best for static content`,
					CodeExamples: `Multi-Layer CDN + Caching Architecture:

[User / Browser]
     |
     V
[CDN Edge Cache (Global PoPs)]
     — Serves static assets globally (images, CSS, JS, videos)
     |
     V
[Origin / Application Layer]
     |
     V   — Reverse Proxy / Web Server Cache (e.g. Varnish, NGINX)
     |
     V
[In-Memory Cache]
     — Redis or Memcached cluster
     — Used for sessions, hot data, API call caching, DB query results
     |
     V
[Backend Database / Storage]

Layers:
1. CDN Edge Cache: Static files globally distributed
2. Reverse Proxy Cache: Dynamic content caching at origin
3. In-Memory Cache: Fast access to frequently used data
4. Database: Persistent storage`,
				},
				{
					Title: "Cache Patterns",
					Content: `**Cache Patterns:**

**Cache-Aside (Lazy Loading):**
1. Check cache
2. If miss, read from database
3. Write to cache
4. Return data

**Write-Through:**
1. Write to cache
2. Write to database
3. Both updated synchronously

**Write-Behind (Write-Back):**
1. Write to cache
2. Write to database asynchronously
3. Faster writes, risk of data loss

**Cache Invalidation:**
- **TTL (Time To Live)**: Expire after time
- **LRU (Least Recently Used)**: Evict least used items
- **Manual**: Explicitly invalidate on updates`,
				},
				{
					Title: "Redis & Memcached",
					Content: `**Redis (Remote Dictionary Server):**
- In-memory data structure store
- Supports strings, lists, sets, sorted sets, hashes
- Persistence options (RDB snapshots, AOF)
- Built-in replication and clustering
- Rich feature set (pub/sub, transactions, Lua scripting)

**Use Cases:**
- Session storage
- Real-time leaderboards
- Rate limiting
- Pub/sub messaging
- Caching

**Memcached:**
- Simple in-memory key-value store
- No persistence
- Multi-threaded
- Simpler than Redis
- Good for simple caching

**When to Use:**
- **Redis**: Need advanced data structures, persistence, or pub/sub
- **Memcached**: Simple caching, maximum performance

**CDN (Content Delivery Network):**
- Network of geographically distributed servers
- Cache static content (images, CSS, JS, videos)
- Serve content from nearest edge location
- Reduces latency and origin server load

**CDN Providers:**
- CloudFlare
- AWS CloudFront
- Fastly
- Akamai`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          13,
			Title:       "Database Design",
			Description: "Understand SQL vs NoSQL, replication, sharding, and database scaling strategies.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "SQL vs NoSQL",
					Content: `**SQL Databases (Relational):**
- Structured data with relationships
- ACID transactions
- Schema enforcement
- Vertical scaling
- Examples: PostgreSQL, MySQL, Oracle

**When to Use SQL:**
- Complex queries and joins
- ACID compliance required
- Structured data
- Strong consistency needed

**NoSQL Databases:**

**Document Stores** (MongoDB, CouchDB):
- Store documents (JSON-like)
- Flexible schema
- Good for content management

**Key-Value Stores** (Redis, DynamoDB):
- Simple key-value pairs
- Very fast
- Good for caching, sessions

**Column Stores** (Cassandra, HBase):
- Store data in columns
- Excellent for analytics
- Good write performance

**Graph Databases** (Neo4j):
- Store relationships
- Good for social networks, recommendations

**When to Use NoSQL:**
- Large scale
- Flexible schema
- Horizontal scaling needed
- Eventual consistency acceptable`,
				},
				{
					Title: "Database Replication",
					Content: `Replication creates copies of data across multiple database servers.

**Master-Slave (Primary-Replica) Replication:**
- One master handles writes
- Multiple replicas handle reads
- Replicas sync from master
- Read scaling
- Single point of failure for writes

**Master-Master Replication:**
- Multiple masters handle writes
- Better write availability
- Conflict resolution needed
- More complex

**Replication Strategies:**
- **Synchronous**: Wait for all replicas to confirm
- **Asynchronous**: Don't wait (faster, risk of data loss)

**Use Cases:**
- Read scaling
- High availability
- Geographic distribution
- Backup and disaster recovery`,
					CodeExamples: `Master-Slave Replication Architecture:

                +----------------------+
                |        Client        |
                +----------+-----------+
                           |
                           v
                   +---------------+
                   |   Master DB   |
                   +-------+-------+
                           |
              Replication /|\\  (Writes, updates flow to Slaves)
                        |
        +---------------+---------------+
        |                               |
        v                               v
 +---------------+             +----------------+
 |  Slave DB 1   |             |   Slave DB 2   |
 +---------------+             +----------------+
        ^                               ^
        |                               |
    Read traffic                Read traffic
        |                               |
        +----------+--------------+-----+
                   |              |
                   v              v
            +------------+   +------------+
            | Analytics/ |   | Reporting/ |
            | Search/ etc|   | BI tools   |
            +------------+   +------------+

- Write operations go to the Master
- Master asynchronously or synchronously replicates data to Slaves
- Reads can be served from any Slave (or from Master if needed)`,
				},
				{
					Title: "Database Sharding",
					Content: `Sharding splits a database into smaller, more manageable pieces called shards.

**Sharding Strategies:**

**Horizontal Sharding (Partitioning):**
- Split rows across shards
- Each shard has same schema
- Shard by user_id, region, etc.

**Vertical Sharding:**
- Split columns across shards
- Different tables on different servers
- Less common

**Sharding Key Selection:**
- Choose key that distributes evenly
- Avoid hot spots
- Consider query patterns

**Challenges:**
- Cross-shard queries are expensive
- Rebalancing is complex
- Joins across shards difficult
- Transaction management complex

**Sharding Patterns:**
- **Range-based**: Shard by ID ranges
- **Hash-based**: Hash key to determine shard
- **Directory-based**: Lookup table maps keys to shards`,
					CodeExamples: `Sharding (Horizontal Partitioning) Architecture:

                     +----------------------+
                     |        Client        |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | Shard Router / Proxy |
                     +--+---+-----------+---+
                        |   |           |
        Shard Key Range  |   |           |
                        |   |           |
         +--------------+   +------+    +----------------+
         |                     |                |
         v                     v                v
+----------------+   +----------------+   +----------------+
| Shard 1 (DB)   |   | Shard 2 (DB)   |   | Shard N (DB)   |
| key range A-F  |   | key G-M        |   | key N-Z        |
+----------------+   +----------------+   +----------------+
        |                     |                    |
        v                     v                    v
   +--------+           +--------+           +---------+
   | Replica|           | Replica|           | Replica |
   +--------+           +--------+           +---------+

- Clients send requests to a Shard Router which uses the sharding key to pick the proper shard
- Each shard (DB) is responsible for a subset of data (key ranges, hash buckets, etc.)
- Each shard can have one or more Replicas for redundancy and read scaling`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          14,
			Title:       "Message Queues & Event Streaming",
			Description: "Learn about message queues, event-driven architecture, and streaming systems like Kafka.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Message Queues",
					Content: `Message queues enable asynchronous communication between services.

**Benefits:**
- Decouple services
- Handle traffic spikes
- Improve reliability
- Enable async processing
- Better scalability

**Message Queue Patterns:**

**Point-to-Point:**
- One producer, one consumer
- Message consumed once
- Example: Task queue

**Pub/Sub (Publish-Subscribe):**
- One producer, multiple consumers
- Each consumer gets copy
- Example: Event notifications

**Message Queue Features:**
- **Durability**: Messages persist
- **Ordering**: Maintain message order
- **Delivery guarantees**: At-least-once, exactly-once
- **Dead letter queue**: Failed messages

**Popular Message Queues:**
- **RabbitMQ**: Feature-rich, AMQP protocol
- **Apache Kafka**: High throughput, event streaming
- **Amazon SQS**: Managed, simple
- **Redis Pub/Sub**: Simple, fast`,
				},
				{
					Title: "Apache Kafka",
					Content: `Kafka is a distributed event streaming platform.

**Key Concepts:**
- **Topics**: Categories/feeds of messages
- **Partitions**: Topics split into partitions for parallelism
- **Producers**: Write messages to topics
- **Consumers**: Read messages from topics
- **Consumer Groups**: Multiple consumers working together

**Kafka Features:**
- High throughput (millions of messages/second)
- Durability (messages persisted to disk)
- Scalability (distributed, partitions)
- Replay capability (read messages multiple times)
- Ordering within partitions

**Use Cases:**
- Event sourcing
- Log aggregation
- Stream processing
- Activity tracking
- Metrics collection

**Kafka Architecture:**
- Brokers: Kafka servers
- ZooKeeper: Coordination (or KRaft in newer versions)
- Producers → Topics → Partitions → Consumers`,
					CodeExamples: `Kafka Architecture:

        Producer 1          Producer 2          Producer 3
            |                  |                  |
            +------------------+------------------+
                               |
                               v
                    +-------------------+
                    |  Kafka Cluster    |
                    |  (Brokers)        |
                    +-------------------+
                    |                   |
            +-------+-------+   +-------+-------+
            |   Topic A     |   |   Topic B     |
            |               |   |               |
            | Partition 0  |   | Partition 0  |
            | Partition 1  |   | Partition 1  |
            | Partition 2  |   | Partition 2  |
            +-------+-------+   +-------+-------+
                    |                   |
            +-------+-------+   +-------+-------+
            | Consumer Group 1 |   | Consumer Group 2 |
            |  Consumer 1      |   |  Consumer 1      |
            |  Consumer 2      |   |  Consumer 2      |
            +------------------+   +------------------+

Key Components:
- Producers write to topics
- Topics are divided into partitions for parallelism
- Consumers read from partitions
- Consumer groups allow parallel processing
- Brokers store and replicate partitions`,
				},
			},
			ProblemIDs: []int{},
		},
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
			ID:          17,
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
		{
			ID:          20,
			Title:       "Real-World System Examples",
			Description: "Deep dive into how major tech companies built their systems: Twitter, Instagram, YouTube, Netflix, Uber, and Google.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "Twitter/X: Timeline Architecture",
					Content: `**Real-World Architecture:**

Twitter handles 500M+ tweets per day and serves timelines to 450M+ users. Here's how they do it:

**Fan-Out on Write Strategy:**
- When a user tweets, the system immediately pushes the tweet to ALL followers' timelines
- This creates massive write amplification (one tweet → millions of timeline writes)
- But enables instant timeline reads - no computation needed when user refreshes

**Architecture Components:**

1. **Timeline Service:**
   - Precomputed timelines stored in Redis clusters
   - Active users: timelines cached in memory
   - Inactive users: timelines rebuilt on-demand from persistent storage

2. **Graph Database:**
   - Stores follower/following relationships
   - Used to determine where to fan-out tweets
   - Sharded by user ID for scalability

3. **Storage Layers:**
   - Redis: Hot timelines for active users (in-memory)
   - Manhattan (custom KV store): Low-latency timeline storage
   - HDFS: Cold storage for analytics and batch processing
   - Blob storage: Images and videos

4. **Scaling Challenges:**
   - Celebrity problem: Users with millions of followers create write storms
   - Solution: Separate handling for high-follower accounts
   - Write amplification: Accept high write cost for instant reads
   - Replication: Timeline data replicated across data centers

**Trade-offs:**
- ✅ Instant timeline reads (no computation)
- ✅ Simple read path
- ❌ High write amplification
- ❌ Storage intensive (each tweet stored N times for N followers)
- ❌ Slow writes for high-follower accounts

**Key Insight:** Twitter prioritizes read latency over write efficiency, accepting the cost of write amplification to ensure users see their timeline instantly.`,
				},
				{
					Title: "Instagram: Media-Heavy Architecture",
					Content: `**Real-World Architecture:**

Instagram serves billions of photos and videos daily to 2B+ users. Here's their evolution:

**Architecture Evolution:**
- Started as Django monolith (Python)
- Evolved to microservices architecture
- Separate services for: user auth, media storage, feed generation, notifications, recommendations

**Core Components:**

1. **Media Storage & Processing:**
   - Raw uploads stored in object storage (S3-like)
   - Background transcoding pipeline processes videos into multiple formats/resolutions
   - Uses AV1 codec for compression optimization
   - Presigned URLs reduce load on application servers
   - CDN serves media globally (CloudFlare, Fastly)

2. **Dual Database Strategy:**
   - PostgreSQL/MySQL: Transactional data (user accounts, profiles, metadata)
   - Cassandra: High-volume time-series data (feed data, activity logs)
   - Redis/Memcached: Hot data caching (frequently accessed posts, profiles)

3. **Feed Generation:**
   - Hybrid model: Precompute + pull
   - Active users: Precomputed feed cached
   - Inactive users: Feed generated on-demand
   - ML-based ranking for relevance
   - Separate services for: user metadata, timelines, media, recommendations

4. **Scalability Techniques:**
   - Horizontal scaling: Services sharded by user ID or region
   - Media metadata partitioned across shards
   - Asynchronous workflows: Background queues for notifications, media processing
   - Low latency targets: Feed load < 200ms, Stories < 200ms

**Key Challenges:**
- Media processing is CPU-intensive (transcoding)
- Storage costs for billions of images/videos
- Feed freshness vs. performance trade-off
- Global distribution for low latency

**Key Insight:** Instagram separates concerns - transactional data in SQL, high-volume data in NoSQL, and uses extensive caching and CDN for media delivery.`,
				},
				{
					Title: "YouTube: Video Platform Architecture",
					Content: `**Real-World Architecture:**

YouTube serves billions of video views daily. Here's how they handle uploads, streaming, and search:

**Core Components:**

1. **Video Upload & Processing:**
   - Raw video uploads stored in object storage
   - Background transcoding pipeline converts videos to multiple formats/resolutions
   - Adaptive bitrate streaming: Multiple quality tiers (240p, 360p, 480p, 720p, 1080p, 4K)
   - Processing happens asynchronously - user can browse while video processes

2. **Video Delivery (CDN):**
   - Edge servers cache video chunks globally
   - Thumbnails cached at edge locations
   - Reduces latency and origin server load
   - Multiple CDN providers for redundancy

3. **Search & Discovery:**
   - Elasticsearch or similar for text metadata search
   - ML-based recommendation engine
   - Event streaming (Kafka) for user behavior tracking
   - Analytics pipelines for recommendations

4. **API Gateway & Load Balancers:**
   - Routes dynamic requests (search, feed, auth)
   - Handles authentication and authorization
   - Rate limiting and DDoS protection

5. **Data Storage:**
   - SQL databases: User accounts, video metadata, playlists
   - NoSQL: Viewing history, analytics data
   - Object storage: Video files (distributed across regions)
   - Cache: Frequently accessed metadata

**Scaling Strategies:**
- Sharding: Videos and metadata sharded by video ID or user ID
- Horizontal scaling: Add more transcoding workers, API servers
- Caching: Hot videos cached at edge, metadata cached in Redis
- Read replicas: Database replicas for read scaling

**Key Challenges:**
- Massive storage requirements (petabytes of video)
- Transcoding compute costs
- Global distribution for low latency
- Search at scale (billions of videos)

**Key Insight:** YouTube separates upload/processing from streaming, uses extensive CDN caching, and processes videos asynchronously to handle massive scale.`,
				},
				{
					Title: "Netflix: Streaming Architecture",
					Content: `**Real-World Architecture:**

Netflix streams to 250M+ subscribers globally. Here's their architecture:

**Open Connect (Proprietary CDN):**
- Netflix's own CDN deployed in ISP networks
- OCAs (Open Connect Appliances) placed near users
- Stores multiple quality bitrates locally
- 98% edge cache hit rate - most content served from edge
- Reduces backbone egress costs and startup latency

**Microservices Architecture:**
- Hundreds of microservices (recommendation, playback, UI, catalog)
- Zuul: API gateway for routing
- Eureka: Service discovery
- Each service independently deployable and scalable

**Data Layer:**

1. **SQL Databases (MySQL):**
   - Critical transactional data: Billing, account info, subscriptions
   - Strong consistency required
   - Replicated across regions

2. **NoSQL (Cassandra):**
   - High-volume writes: Viewing history, event logs
   - Eventually consistent acceptable
   - Excellent write performance

3. **Distributed Caches:**
   - EVCache, Agar: Reduce read latency
   - Shield origin services from load
   - Cache frequently accessed data

**Messaging & Streaming:**
- Kafka: Ingest user events (plays, pauses, searches)
- Spark, Flink: Analytics and ML workflows
- Real-time recommendations based on viewing patterns

**Geographic Redundancy:**
- Multi-region deployment (AWS regions)
- Data replicated across regions
- Failover to backup regions if primary fails
- Reduces latency and improves availability

**Chaos Engineering:**
- Chaos Monkey: Randomly terminates instances
- Ensures system resilience
- Tests failure scenarios proactively

**Key Challenges:**
- Video storage & caching balance (many quality tiers)
- Microservices complexity (coordination, versioning)
- Global distribution (low latency worldwide)
- Personalization at scale

**Key Insight:** Netflix uses proprietary CDN for cost efficiency, microservices for agility, and extensive caching to handle massive scale while maintaining low latency globally.`,
				},
				{
					Title: "Uber: Real-Time Matching System",
					Content: `**Real-World Architecture:**

Uber matches riders with drivers in real-time across millions of trips daily. Here's how:

**Core Challenge:**
- Real-time location updates from millions of drivers
- Match riders with nearby drivers
- Calculate ETAs dynamically
- Handle surge pricing
- Process payments

**Geo-Spatial Architecture:**

1. **S2 Library (Google):**
   - Divides world into hierarchical cells
   - Quick proximity lookups
   - Spatial indexing for "which drivers are near this rider?"
   - Efficient geographic queries

2. **Supply/Demand Services:**
   - Driver location updates streamed every few seconds
   - Location data flows through Kafka to workers
   - Real-time computation for dispatch logic
   - WebSockets for real-time updates to mobile apps

3. **Dispatch Optimization (DISCO):**
   - Matches riders → drivers minimizing wait time
   - Considers: distance, traffic, driver availability
   - Dynamic routing based on real-time traffic
   - ETA calculations updated continuously

4. **Scalability:**
   - Ringpop: Consistent hashing for sharding
   - Gossip protocols for cluster membership
   - Logical partitioning of geographic cells
   - Horizontal scaling of dispatch workers

**Infrastructure:**

1. **Event Streaming (Kafka):**
   - Massive write/read of location events
   - Ride events, driver status changes
   - Analytics and ML pipelines

2. **Real-Time Communication:**
   - WebSockets: Real-time updates to mobile apps
   - Push notifications for ride status
   - Low latency critical for UX

3. **Data Storage:**
   - SQL: User accounts, ride history, payments
   - NoSQL: Location data, event logs
   - Time-series DB: Location tracking

**Key Challenges:**
- Real-time demands: Low latency critical (delayed updates = bad matches)
- Geographic partitioning: Balancing cell size vs. precision
- Scale: Millions of concurrent location updates
- Consistency: Ensuring accurate ETAs and matches

**Trade-offs:**
- Coarse grid cells: Faster lookup, less precision
- Fine grid cells: More precision, slower lookup
- Solution: Adaptive cell sizing based on density

**Key Insight:** Uber prioritizes real-time performance over perfect consistency, using spatial indexing and event streaming to handle millions of concurrent location updates and matches.`,
				},
				{
					Title: "Google Zanzibar: Authorization System",
					Content: `**Real-World Architecture:**

Google Zanzibar is a global authorization system used across Google services (Drive, Photos, YouTube). It handles billions of authorization checks daily.

**ReBAC Model (Relationship-Based Access Control):**
- Stores authorization as triplets: (subject, object, relation)
- More flexible than traditional RBAC
- Example: (user:alice, file:doc1, relation:viewer)
- Supports dynamic, complex policies

**Architecture:**

1. **Google Spanner Storage:**
   - Strongly consistent, globally replicated database
   - Authorization decisions must reflect up-to-date permissions
   - Cross-region replication for low latency
   - ACID transactions for consistency

2. **Caching Layers:**
   - Multiple levels: Server-local cache, inter-service cache
   - Reduces read latency significantly
   - "Zookies" tokens ensure consistency
   - Requests see snapshot of state consistent with content version

3. **Global Replication:**
   - Authorization data replicated across regions
   - Low latency for global users
   - Handles network partitions gracefully
   - Versioning and snapshots ensure correctness

**Key Features:**

1. **Consistency Guarantees:**
   - Strong consistency via Spanner
   - Caching with consistency tokens
   - Snapshots ensure users see consistent state

2. **Performance:**
   - Multi-level caching reduces latency
   - Parallel evaluation of authorization checks
   - Optimized for high throughput

3. **Scalability:**
   - Handles billions of checks per day
   - Global distribution
   - Horizontal scaling

**Trade-offs:**
- ✅ Strong consistency (required for security)
- ✅ Global scale and low latency
- ❌ Higher overhead than eventual consistency
- ❌ More complex than simple RBAC

**Use Cases:**
- File sharing permissions (Google Drive)
- Photo album access (Google Photos)
- Video privacy settings (YouTube)
- Any fine-grained access control

**Key Insight:** Zanzibar prioritizes correctness and consistency over performance, using Spanner for strong consistency while mitigating latency through multi-level caching and global replication.`,
				},
			},
			ProblemIDs: []int{},
		}, // Last module
	})
}
