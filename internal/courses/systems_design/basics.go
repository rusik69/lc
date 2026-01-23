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
- **Security**: System protects data and prevents unauthorized access
- **Cost Efficiency**: System operates within budget constraints

**Why Systems Design Matters:**

**Professional Impact:**
- Design systems that can scale to millions of users
- Make informed architectural decisions
- Understand trade-offs between different approaches
- Build production-ready systems that don't require constant firefighting
- Reduce technical debt and maintenance costs

**Career Development:**
- Prepare for technical interviews at top tech companies
- Demonstrate senior-level thinking and problem-solving
- Lead architecture discussions and design reviews
- Mentor junior engineers on system design principles

**Business Value:**
- Reduce infrastructure costs through efficient design
- Enable rapid feature development with good architecture
- Minimize downtime and improve user experience
- Support business growth without major rewrites

**Common System Requirements:**

**Functional Requirements:**
- What the system should do (features, capabilities)
- User stories and use cases
- API specifications
- Data models and relationships

**Non-Functional Requirements:**
- **Performance**: Response time, throughput, latency
- **Scalability**: Ability to handle growth
- **Availability**: Uptime percentage (e.g., 99.9%)
- **Reliability**: Error rate, data consistency
- **Security**: Authentication, authorization, encryption
- **Usability**: User experience, accessibility

**Constraints:**
- **Budget**: Infrastructure and development costs
- **Time**: Launch deadlines, development timeline
- **Technology Stack**: Existing systems, team expertise
- **Regulatory**: Compliance requirements (GDPR, HIPAA)
- **Geographic**: Data residency, latency requirements

**Assumptions:**
- Expected usage patterns (read-heavy vs write-heavy)
- Growth rates (users, data, traffic)
- User behavior (peak times, geographic distribution)
- Data access patterns (hot vs cold data)

**Systems Design Process:**

**1. Requirements Gathering:**
- Understand functional requirements
- Identify non-functional requirements
- Clarify constraints and assumptions
- Ask clarifying questions

**2. Capacity Estimation:**
- Estimate traffic (QPS, RPS)
- Estimate storage requirements
- Estimate bandwidth needs
- Plan for growth

**3. System API Design:**
- Define endpoints
- Specify request/response formats
- Consider versioning
- Document error handling

**4. Database Design:**
- Choose database type (SQL vs NoSQL)
- Design schema
- Plan for scaling (sharding, replication)
- Consider indexing strategy

**5. High-Level Design:**
- Identify major components
- Define component interactions
- Plan for scalability
- Consider fault tolerance

**6. Detailed Design:**
- Deep dive into each component
- Design algorithms and data structures
- Plan for edge cases
- Consider security and privacy

**7. Identify Bottlenecks:**
- Analyze potential bottlenecks
- Plan for optimization
- Design for monitoring
- Plan for scaling

**Common Pitfalls:**

**1. Over-engineering:**
- Designing for scale you don't need
- Adding complexity prematurely
- Not considering YAGNI (You Aren't Gonna Need It)
- Solution: Start simple, optimize when needed

**2. Under-engineering:**
- Not planning for growth
- Ignoring non-functional requirements
- Missing critical components
- Solution: Plan for reasonable growth, consider requirements

**3. Ignoring Trade-offs:**
- Not understanding consequences of choices
- Optimizing for wrong metrics
- Not considering costs
- Solution: Explicitly discuss trade-offs, measure what matters

**4. Single Point of Failure:**
- Not planning for component failures
- No redundancy
- No failover mechanisms
- Solution: Design for failure, add redundancy

**5. Poor Scalability Planning:**
- Not considering how to scale
- Tight coupling between components
- Stateful design that's hard to scale
- Solution: Design stateless services, plan scaling strategy

**Best Practices:**

**1. Start with Requirements:**
- Understand the problem before designing solution
- Ask clarifying questions
- Document assumptions
- Validate requirements with stakeholders

**2. Think in Layers:**
- Client layer (mobile, web)
- API layer (REST, GraphQL)
- Application layer (business logic)
- Data layer (databases, caches)
- Infrastructure layer (servers, networks)

**3. Design for Scale:**
- Horizontal scaling over vertical
- Stateless services
- Caching strategies
- Database scaling (read replicas, sharding)

**4. Design for Failure:**
- Assume components will fail
- Add redundancy
- Implement circuit breakers
- Plan graceful degradation

**5. Keep it Simple:**
- Don't over-complicate
- Use proven patterns
- Avoid premature optimization
- Iterate and improve

**Real-World Examples:**

**Twitter-like System:**
- Requirements: Post tweets, follow users, view timeline
- Scale: 300M users, 500M tweets/day
- Key challenges: Timeline generation, real-time updates
- Solutions: Fan-out, caching, read replicas

**URL Shortener (TinyURL):**
- Requirements: Shorten URLs, redirect to original
- Scale: 100M URLs/day, 10:1 read/write ratio
- Key challenges: Unique ID generation, high read volume
- Solutions: Base62 encoding, caching, CDN

**Chat System:**
- Requirements: Send messages, group chats, presence
- Scale: 1B users, 50B messages/day
- Key challenges: Real-time delivery, message ordering
- Solutions: WebSockets, message queues, distributed systems`,
					CodeExamples: `# Example: Requirements Analysis
# Problem: Design a URL shortener

# Functional Requirements:
# 1. Shorten long URLs
# 2. Redirect short URLs to original
# 3. Optional: Analytics, custom URLs, expiration

# Non-Functional Requirements:
# - Scale: 100M URLs/day
# - Read/Write ratio: 100:1 (read-heavy)
# - Availability: 99.9%
# - Latency: < 100ms for redirect

# Constraints:
# - Short URL: 7 characters
# - Must be unique
# - Must be URL-safe

# Assumptions:
# - URLs don't expire (or long expiration)
# - Analytics not required initially
# - Global distribution needed

# Example: API Design
# POST /api/v1/shorten
# Request: {"url": "https://example.com/very/long/url"}
# Response: {"short_url": "https://short.ly/abc1234"}

# GET /abc1234
# Response: 301 Redirect to original URL

# Example: Capacity Estimation
# URLs per day: 100 million
# URLs per second: 100M / 86400 = ~1,200 writes/sec
# Reads per second: 1,200 × 100 = 120,000 reads/sec
# Storage: 100M URLs × 500 bytes = 50 GB/year
# With 5 years retention: 250 GB

# Example: High-Level Design
# Client → Load Balancer → API Servers
#                              ↓
#                    [URL Service] → [Database]
#                              ↓
#                         [Cache (Redis)]
#                              ↓
#                    [Analytics Service] (optional)

# Example: Database Schema
# URLs table:
# - id (bigint, primary key)
# - short_code (varchar(7), unique index)
# - original_url (text)
# - created_at (timestamp)
# - expires_at (timestamp, nullable)
# - user_id (bigint, nullable)

# Example: Scaling Strategy
# Phase 1: Single server, single database
# Phase 2: Multiple app servers, read replicas
# Phase 3: Database sharding by short_code
# Phase 4: CDN for redirects, separate analytics DB`,
				},
				{
					Title: "Scalability Fundamentals",
					Content: `Scalability is the ability of a system to handle increased load by adding resources. There are two main approaches:

**Vertical Scaling (Scale Up):**
- Add more power to existing machines (CPU, RAM, storage)
- Simpler to implement (no code changes usually needed)
- Limited by hardware constraints (maximum CPU/RAM available)
- Single point of failure (one machine)
- Cost increases exponentially (more powerful hardware is expensive)
- Example: Upgrade server from 8GB to 32GB RAM
- **When to use**: Small scale, simple applications, temporary scaling needs
- **Limitations**: Can't scale beyond single machine capacity, downtime for upgrades

**Horizontal Scaling (Scale Out):**
- Add more machines to the system
- More complex but more flexible
- Can scale almost infinitely (add more machines as needed)
- Better fault tolerance (one machine failure doesn't kill system)
- Cost increases linearly (add more standard machines)
- Example: Add 10 more servers to handle load
- **When to use**: Large scale, high availability requirements, cloud deployments
- **Requirements**: Stateless services, load balancing, distributed systems knowledge

**Scaling Comparison:**

**Vertical Scaling:**
- Pros: Simple, no code changes, immediate
- Cons: Limited, expensive, single point of failure
- Best for: Small systems, temporary spikes, development

**Horizontal Scaling:**
- Pros: Unlimited scale, fault tolerant, cost-effective
- Cons: Complex, requires architectural changes, distributed systems challenges
- Best for: Production systems, large scale, high availability

**Load Types:**

**Read-heavy:**
- More reads than writes (e.g., social media feeds, content sites)
- Ratio: 100:1 or higher read/write ratio
- Strategies: Caching, read replicas, CDN
- Examples: Twitter feeds, news sites, product catalogs

**Write-heavy:**
- More writes than reads (e.g., logging systems, analytics)
- Ratio: 10:1 or higher write/read ratio
- Strategies: Write optimization, batching, async processing
- Examples: Logging systems, IoT data collection, analytics pipelines

**Balanced:**
- Similar read/write ratio
- Strategies: Both read and write optimization
- Examples: E-commerce (product updates + browsing), collaboration tools

**Scaling Strategies:**

**Stateless Services:**
- No server-side session state
- Each request is independent
- Easy to scale horizontally (add more servers)
- Load balancer can route to any server
- Example: REST APIs, microservices
- **Best Practice**: Store state in database or cache, not in application server

**Stateful Services:**
- Maintain server-side state (sessions, connections)
- Require careful design for scaling
- Options:
  - Sticky sessions (same client → same server)
  - Shared state store (Redis, database)
  - State replication across servers
- Example: WebSocket servers, gaming servers
- **Challenge**: Harder to scale, requires state management

**Database Scaling:**

**Read Replicas:**
- Master handles writes, replicas handle reads
- Reduces load on master database
- Improves read performance
- Replication lag consideration
- Example: MySQL master-replica, PostgreSQL streaming replication

**Sharding (Partitioning):**
- Split data across multiple databases
- Each shard handles subset of data
- Requires shard key selection
- Complex queries across shards
- Example: User data sharded by user_id

**Caching:**
- Store frequently accessed data in memory
- Reduces database load
- Improves response time
- Cache invalidation challenges
- Example: Redis, Memcached, CDN

**Scaling Patterns:**

**1. Microservices:**
- Break monolith into small services
- Scale each service independently
- Better resource utilization
- More complex operations

**2. CQRS (Command Query Responsibility Segregation):**
- Separate read and write models
- Optimize each independently
- Read models can be heavily cached
- Write models optimized for consistency

**3. Event-Driven Architecture:**
- Services communicate via events
- Decoupled services
- Better scalability
- Eventual consistency

**Common Pitfalls:**

**1. Not Designing for Scale from Start:**
- Hard to retrofit scalability
- Technical debt accumulates
- Solution: Plan for reasonable scale, use scalable patterns

**2. Ignoring Database Bottlenecks:**
- Application scales but database doesn't
- Database becomes bottleneck
- Solution: Plan database scaling (replicas, sharding, caching)

**3. Stateful Design:**
- Storing state in application servers
- Can't scale horizontally easily
- Solution: Make services stateless, use external state store

**4. Tight Coupling:**
- Services depend on each other
- Can't scale independently
- Solution: Loose coupling, async communication, service boundaries

**5. Not Monitoring:**
- Don't know when to scale
- Scale too early or too late
- Solution: Monitor metrics, set up auto-scaling

**Best Practices:**

**1. Design Stateless Services:**
- Store state externally (database, cache)
- Use JWT or session stores
- Enable horizontal scaling

**2. Use Caching Strategically:**
- Cache frequently accessed data
- Set appropriate TTLs
- Handle cache invalidation
- Consider cache warming

**3. Plan Database Scaling:**
- Use read replicas for read-heavy workloads
- Consider sharding for very large datasets
- Use appropriate database types (SQL vs NoSQL)

**4. Implement Auto-scaling:**
- Scale based on metrics (CPU, memory, request rate)
- Scale proactively, not reactively
- Set appropriate thresholds

**5. Load Test:**
- Test scalability assumptions
- Identify bottlenecks before production
- Validate scaling strategies`,
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
					CodeExamples: `Availability Calculation Examples:

Single Server:
- MTBF (Mean Time Between Failures): 720 hours
- MTTR (Mean Time To Repair): 1 hour
- Availability = MTBF / (MTBF + MTTR) = 720 / 721 = 99.86%

Redundant System (2 servers):
- If one server fails, system still operational
- Availability = 1 - (1 - A1) × (1 - A2)
- With 99.86% each: 1 - (0.0014 × 0.0014) = 99.9998%

Redundancy Patterns:

Active-Passive (Hot Standby):
Primary Server (Active)
    |
    | (Heartbeat)
    |
Standby Server (Passive) - Ready to take over
    |
    | (On failure)
    |
Failover → Standby becomes Active

Active-Active:
Server 1 (Active) ←→ Load Balancer ←→ Server 2 (Active)
    |                                          |
    +---------- Shared State Store -----------+
    
Both servers handle traffic simultaneously
Better resource utilization than active-passive

Health Check Implementation:
GET /health
Response: {
  "status": "healthy",
  "timestamp": "2024-01-17T10:00:00Z",
  "checks": {
    "database": "ok",
    "cache": "ok",
    "external_api": "ok"
  }
}

Failover Mechanism:
1. Health check fails (3 consecutive failures)
2. Load balancer removes server from pool
3. Traffic routed to healthy servers
4. Standby server activated (if active-passive)
5. Failed server marked for repair

Graceful Degradation Example:
if database_unavailable:
    serve_cached_data()
    show_banner("Limited functionality")
    log_error("DB unavailable, serving from cache")
else:
    serve_fresh_data()`,
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
					CodeExamples: `ACID Transaction Example:

BEGIN TRANSACTION;
  UPDATE accounts SET balance = balance - 100 WHERE id = 1;
  UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

If any step fails, entire transaction rolls back (Atomicity)
Database remains in valid state (Consistency)
Other transactions don't see partial updates (Isolation)
Changes persist after commit (Durability)

CAP Theorem Scenarios:

CP System (Consistency + Partition Tolerance):
- Example: Traditional SQL databases with replication
- On network partition: Stop accepting writes (sacrifice Availability)
- Ensures all nodes see same data
- Use case: Financial transactions, critical data

AP System (Availability + Partition Tolerance):
- Example: DNS, CDN, NoSQL databases (Cassandra, DynamoDB)
- On network partition: Continue serving requests (sacrifice Consistency)
- May serve stale data temporarily
- Use case: Social media feeds, content delivery

CA System (Consistency + Availability):
- Not possible in distributed systems
- Only works when no network partitions occur
- Single-node systems only

Consistency Level Comparison:

Strong Consistency (Read-after-Write):
Write: User updates profile
Read: Immediately sees updated profile
All replicas updated synchronously
Latency: Higher (waits for all replicas)

Eventual Consistency:
Write: User posts tweet
Read: May see tweet after few seconds
Replicas updated asynchronously
Latency: Lower (doesn't wait for all replicas)

Example: Social Media Feed
- User posts content → Write to primary database
- Content replicated to followers' feeds asynchronously
- User may see their own post immediately (strong consistency)
- Followers may see it after replication delay (eventual consistency)

Weak Consistency Example:
- Real-time analytics dashboard
- Data may be seconds/minutes old
- Acceptable for non-critical metrics
- Very low latency`,
				},
				{
					Title: "Performance Metrics",
					Content: `Understanding and measuring performance metrics is crucial for designing and optimizing systems. These metrics help identify bottlenecks and guide optimization efforts.

**Key Performance Metrics:**

**1. Latency:**
- Time taken for a request to complete
- Measured in milliseconds (ms) or microseconds (μs)
- Types: P50 (median), P95, P99, P99.9
- Example: API response time, database query time

**2. Throughput:**
- Number of requests processed per unit time
- Measured in requests per second (RPS), queries per second (QPS)
- Example: 10,000 requests/second

**3. Bandwidth:**
- Data transfer rate
- Measured in bits per second (bps), bytes per second (Bps)
- Example: 1 Gbps network bandwidth

**4. CPU Utilization:**
- Percentage of CPU time used
- High utilization indicates CPU-bound workload
- Target: 60-80% for headroom

**5. Memory Utilization:**
- Percentage of memory used
- High utilization may cause swapping (slow)
- Monitor for memory leaks

**6. I/O Metrics:**
- Disk I/O: Read/write operations per second (IOPS)
- Network I/O: Packets per second, bytes transferred
- Database: Query execution time, connection pool usage

**7. Error Rate:**
- Percentage of requests that fail
- Target: < 0.1% for production systems
- Monitor: 4xx (client errors), 5xx (server errors)

**Monitoring Best Practices:**
- Set up alerts for critical metrics
- Use percentiles (P95, P99) not just averages
- Track trends over time
- Monitor both application and infrastructure metrics
- Use distributed tracing for request flows

**Performance Targets:**
- API latency: < 200ms (P95)
- Database queries: < 100ms (P95)
- Page load time: < 3 seconds
- Error rate: < 0.1%`,
					CodeExamples: `Performance Monitoring Example:

Latency Percentiles:
- P50 (median): 50ms
- P95: 200ms (95% of requests faster)
- P99: 500ms (99% of requests faster)
- P99.9: 1000ms (99.9% of requests faster)

Throughput Calculation:
- 1 million requests/day = ~11.6 RPS average
- Peak traffic (10x average) = ~116 RPS
- With 3x headroom = ~350 RPS capacity needed

Resource Utilization:
- CPU: 70% average, 90% peak (healthy)
- Memory: 8GB used / 16GB total = 50% (healthy)
- Disk I/O: 1000 IOPS / 5000 max = 20% (healthy)

Error Rate:
- Total requests: 1,000,000
- Errors: 500 (4xx: 300, 5xx: 200)
- Error rate: 0.05% (excellent)`,
				},
				{
					Title: "Capacity Planning",
					Content: `Capacity planning ensures your system can handle expected load with appropriate headroom for growth and traffic spikes.

**Capacity Planning Process:**

**1. Understand Current Load:**
- Measure current traffic patterns
- Identify peak usage times
- Analyze growth trends
- Consider seasonal variations

**2. Forecast Future Load:**
- Project growth based on historical data
- Account for marketing campaigns, product launches
- Consider business goals and targets
- Plan for 6-12 months ahead

**3. Calculate Resource Requirements:**
- Determine resources needed per request
- Account for overhead (OS, monitoring, etc.)
- Add headroom (typically 30-50%)
- Consider redundancy requirements

**4. Plan Scaling Strategy:**
- Vertical scaling limits
- Horizontal scaling capacity
- Auto-scaling thresholds
- Cost optimization

**Key Calculations:**

**Server Capacity:**
- Requests per server = (CPU capacity / CPU per request) × utilization factor
- Example: 4 CPU cores, 0.1 CPU per request, 70% utilization = 28 RPS per server

**Database Capacity:**
- Queries per second (QPS) per database instance
- Connection pool size
- Read replica capacity
- Storage requirements

**Network Capacity:**
- Bandwidth requirements = requests/sec × avg request size
- Account for peak traffic (3-10x average)
- Consider CDN for static content

**Storage Capacity:**
- Current data size
- Growth rate (GB/month)
- Retention policies
- Backup storage needs

**Cost Considerations:**
- Infrastructure costs (servers, databases, storage)
- Network costs (bandwidth, CDN)
- Third-party service costs
- Cost per user/request

**Best Practices:**
- Monitor actual vs. projected usage
- Review and adjust forecasts regularly
- Plan for both gradual growth and sudden spikes
- Consider cloud auto-scaling for flexibility
- Factor in redundancy and failover capacity`,
					CodeExamples: `Capacity Planning Example:

Current State:
- Users: 100,000
- Daily active users: 20,000 (20%)
- Peak RPS: 500
- Average request size: 10 KB
- Storage: 100 GB, growing 10 GB/month

Growth Projection (6 months):
- Users: 200,000 (2x growth)
- Peak RPS: 1,000 (2x growth)
- Storage: 160 GB (100 + 6×10)

Resource Calculation:
- Servers needed: 1,000 RPS / 50 RPS per server = 20 servers
- With 50% headroom: 30 servers
- Network: 1,000 RPS × 10 KB = 10 MB/s = 80 Mbps
- With 3x peak factor: 240 Mbps needed

Database:
- Read QPS: 1,000
- Write QPS: 100 (10:1 read/write ratio)
- Master: Handle writes (100 QPS capacity)
- Read replicas: 1,000 / 500 QPS per replica = 2 replicas

Storage:
- Current: 100 GB
- 6 months: 160 GB
- With 2x redundancy: 320 GB
- Backup (30 days): +30 GB = 350 GB total`,
				},
				{
					Title: "Back-of-Envelope Calculations",
					Content: `Back-of-envelope calculations help quickly estimate system requirements during design discussions. They're essential for making informed decisions without detailed analysis.

**Why Back-of-Envelope?**
- Quick feasibility checks
- Identify potential bottlenecks early
- Guide architectural decisions
- Communicate with stakeholders
- Interview problem-solving

**Key Assumptions:**
- Round numbers for simplicity
- Use powers of 10 when possible
- Make reasonable assumptions
- State assumptions explicitly

**Common Calculations:**

**1. Storage Estimates:**
- Text: ~1 byte per character
- Image: 200 KB average (compressed)
- Video: 1 MB per minute (compressed)
- User data: 1 KB per user (metadata)

**2. Bandwidth Estimates:**
- 1 million users × 10 requests/day × 10 KB = 100 GB/day
- 100 GB/day = ~1.2 MB/s average
- Peak (10x average) = 12 MB/s = 96 Mbps

**3. Database Estimates:**
- Reads: 100:1 read/write ratio common
- Write capacity: 1,000 writes/sec per DB instance
- Read capacity: 10,000 reads/sec per replica
- Storage: Account for indexes (2-3x data size)

**4. Cache Estimates:**
- Hit rate: 80-90% typical
- Cache size: 20% of total data (Pareto principle)
- Memory: 1 GB can store ~1 million small objects

**5. Server Estimates:**
- CPU: 1 core ≈ 1,000-5,000 RPS (depends on workload)
- Memory: 8-16 GB typical per server
- Network: 1 Gbps per server typical

**Calculation Process:**
1. Define the problem clearly
2. List assumptions
3. Break down into smaller calculations
4. Calculate step by step
5. Verify reasonableness
6. Consider edge cases

**Example: Twitter-like System**
- Users: 300 million
- Daily active: 100 million (33%)
- Tweets per user: 5/day average
- Total tweets: 500 million/day
- Peak: 10x average = 5,000 tweets/sec
- Storage: 500M × 280 chars × 1 byte = 140 GB/day
- Reads: 100:1 ratio = 500,000 reads/sec peak
- Cache: 20% of daily data = 28 GB`,
					CodeExamples: `Back-of-Envelope Examples:

Example 1: Image Storage System
- 1 billion images
- Average size: 200 KB
- Total storage: 1B × 200 KB = 200 TB
- With 3x replication: 600 TB
- Cost: 600 TB × $0.023/GB = ~$14,000/month

Example 2: API Throughput
- 10 million requests/day
- Peak: 10x average = 100M requests/day
- Peak RPS: 100M / 86400 = ~1,200 RPS
- Servers needed: 1,200 / 100 RPS = 12 servers
- With 50% headroom: 18 servers

Example 3: Database Capacity
- 100 million users
- 1 KB user data per user = 100 GB
- With indexes (2x): 200 GB
- Reads: 10,000 QPS
- Writes: 100 QPS
- Read replicas: 10,000 / 5,000 = 2 replicas
- Master: Handles 100 QPS (sufficient)

Example 4: Cache Sizing
- 1 million products
- Product data: 10 KB average
- Total data: 10 GB
- Cache 20% (hot data): 2 GB
- Memory needed: 2 GB × 3 replicas = 6 GB
- Can fit in 3 servers with 8 GB RAM each`,
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
					CodeExamples: `Active-Passive Architecture:

Primary Server (Active)
    |
    | (Heartbeat every 5s)
    |
Standby Server (Passive) - Synchronized state
    |
    | (On Primary failure)
    |
Failover → Standby becomes Active (30-60s)

Configuration:
- Heartbeat timeout: 15 seconds
- Failover time: 30-60 seconds
- State synchronization: Continuous replication

Active-Active Architecture:

Load Balancer
    /        \\
Server 1    Server 2
(Active)    (Active)
    |          |
    +----------+
    Shared State (Redis/Database)

Both servers handle traffic simultaneously
No single point of failure
Better resource utilization

Multi-Region Deployment:

Region 1 (US East)          Region 2 (EU)
    |                          |
    +-- Global Load Balancer --+
              |
         [Users]
              |
    Route to nearest region
    Failover if region down

Health Check Configuration:

Liveness Probe:
- Endpoint: /health/live
- Interval: 30s
- Timeout: 5s
- Failure threshold: 3
- Action: Restart container

Readiness Probe:
- Endpoint: /health/ready
- Interval: 10s
- Timeout: 3s
- Failure threshold: 2
- Action: Remove from load balancer

Startup Probe:
- Endpoint: /health/startup
- Interval: 5s
- Timeout: 2s
- Success threshold: 1
- Prevents readiness failures during startup

Graceful Degradation Example:

if database_available:
    data = fetch_from_database()
elif cache_available:
    data = fetch_from_cache()
    show_banner("Showing cached data")
else:
    data = get_default_data()
    show_banner("Limited functionality")`,
				},
				{
					Title: "Consistent Hashing Deep Dive",
					Content: `Consistent hashing is a distributed hashing technique that minimizes reorganization when servers are added or removed, making it essential for distributed caching and load balancing.

**Problem with Traditional Hashing:**
- Hash function: server = hash(key) % num_servers
- When servers added/removed, most keys remap to different servers
- Causes cache misses and data movement

**Consistent Hashing Solution:**
- Map both servers and keys onto a hash ring (circle)
- Key maps to first server encountered clockwise
- When server added/removed, only nearby keys remap
- Minimal disruption (only k/n keys remap, where k=servers, n=total)

**Hash Ring:**
- Imagine a circle with values 0 to 2^32-1
- Servers placed at positions on ring (hash of server ID)
- Keys placed at positions on ring (hash of key)
- Key belongs to first server clockwise

**Virtual Nodes (Vnodes):**
- Each physical server represented by multiple virtual nodes
- Improves load distribution
- Reduces impact of server addition/removal
- Typical: 100-200 virtual nodes per server

**Benefits:**
- Minimal remapping on server changes
- Even load distribution (with virtual nodes)
- Horizontal scaling friendly
- Used in: DynamoDB, Cassandra, Redis Cluster, Load balancers

**Implementation:**
- Use sorted data structure (TreeMap, balanced BST)
- Binary search to find server for key
- O(log n) lookup time

**Use Cases:**
- Distributed caching (Redis Cluster)
- Database sharding
- CDN request routing
- Load balancing with session affinity`,
					CodeExamples: `Consistent Hashing Example:

Hash Ring (0 to 2^32-1):
    0
    |
    |  Server A (hash: 100)
    |  |
    |  |  Key X (hash: 150) → Server B
    |  |
    |  Server B (hash: 200)
    |  |
    |  Server C (hash: 300)
    |
    2^32-1

Adding Server D (hash: 250):
- Only keys between Server B (200) and Server D (250) remap
- Keys between Server D (250) and Server C (300) now map to D
- Minimal disruption

Virtual Nodes Example:
Physical Server 1:
- Virtual Node 1-1 (hash: 100)
- Virtual Node 1-2 (hash: 500)
- Virtual Node 1-3 (hash: 900)

Physical Server 2:
- Virtual Node 2-1 (hash: 200)
- Virtual Node 2-2 (hash: 600)
- Virtual Node 2-3 (hash: 1000)

Better distribution than single hash per server`,
				},
				{
					Title: "Load Balancer Health Checks",
					Content: `Health checks are crucial for load balancers to detect unhealthy servers and route traffic only to healthy instances.

**Health Check Types:**

**1. Liveness Probe:**
- Checks if service is running
- Fails if service crashed or hung
- Action: Restart container/service
- Frequency: Every 10-30 seconds

**2. Readiness Probe:**
- Checks if service can handle requests
- Fails if service is starting up or overloaded
- Action: Remove from load balancer pool
- Frequency: Every 5-10 seconds

**3. Startup Probe:**
- Checks if service has finished initializing
- Used for slow-starting services
- Prevents readiness probe from failing during startup
- Frequency: Every few seconds until ready

**Health Check Methods:**

**1. HTTP/HTTPS:**
- Send GET request to health endpoint
- Check status code (200 = healthy)
- Can check response body
- Most common method

**2. TCP:**
- Check if port is open and accepting connections
- Simpler but less informative
- Faster than HTTP

**3. Command:**
- Execute command in container
- Check exit code
- Most flexible

**Health Check Configuration:**
- **Interval**: How often to check (e.g., 10 seconds)
- **Timeout**: Max time to wait for response (e.g., 5 seconds)
- **Success Threshold**: Consecutive successes needed (e.g., 1)
- **Failure Threshold**: Consecutive failures before marking unhealthy (e.g., 3)

**Best Practices:**
- Health endpoint should be lightweight
- Don't check external dependencies in health check
- Use separate endpoints for liveness and readiness
- Set appropriate timeouts
- Consider graceful shutdown time
- Monitor health check metrics`,
					CodeExamples: `Health Check Configuration Example:

Liveness Probe:
- Endpoint: /health/live
- Method: HTTP GET
- Interval: 30 seconds
- Timeout: 5 seconds
- Success threshold: 1
- Failure threshold: 3
- Action: Restart if fails 3 times

Readiness Probe:
- Endpoint: /health/ready
- Method: HTTP GET
- Interval: 10 seconds
- Timeout: 3 seconds
- Success threshold: 1
- Failure threshold: 2
- Action: Remove from load balancer

Health Check Response:
{
  "status": "healthy",
  "timestamp": "2024-01-17T10:00:00Z",
  "version": "1.2.3"
}

Load Balancer Behavior:
- Healthy servers: Receive traffic
- Unhealthy servers: No traffic, removed from pool
- Recovering servers: Added back after health checks pass`,
				},
				{
					Title: "Geographic Load Balancing",
					Content: `Geographic load balancing routes traffic to the nearest or best-performing data center based on user location, reducing latency and improving user experience.

**Benefits:**
- Reduced latency (route to nearest region)
- Better performance (lower network hops)
- Improved availability (failover to other regions)
- Compliance (data residency requirements)

**Routing Strategies:**

**1. Geographic Proximity:**
- Route based on user's geographic location
- Use DNS to resolve to nearest region
- Simple but may not account for network conditions

**2. Latency-Based:**
- Measure actual latency to each region
- Route to lowest latency region
- More accurate but requires active measurement

**3. Performance-Based:**
- Consider server load, capacity, health
- Route to best-performing region
- Most complex but optimal

**4. Weighted Geographic:**
- Assign weights to regions
- Distribute traffic based on weights
- Useful for capacity planning

**DNS-Based Routing:**
- Use DNS to return different IPs based on location
- Client's DNS resolver location determines response
- Simple but limited by DNS caching

**Anycast:**
- Multiple servers share same IP address
- BGP routes to nearest server automatically
- Transparent to client
- Used by CDNs and DNS services

**Global Load Balancer Architecture:**
- DNS returns IP of regional load balancer
- Regional load balancer routes to servers in region
- Health checks at both levels
- Failover between regions

**Challenges:**
- DNS caching delays failover
- Session affinity across regions
- Data replication and consistency
- Cost of multi-region deployment`,
					CodeExamples: `Geographic Load Balancing Example:

User in US East:
DNS Query → US East IP (50ms latency)
         → US West IP (200ms latency)
         → EU IP (300ms latency)
Result: Route to US East

User in EU:
DNS Query → US East IP (300ms latency)
         → US West IP (350ms latency)
         → EU IP (50ms latency)
Result: Route to EU

Architecture:
[Global DNS]
    |
    +---> [US East Load Balancer] → [Servers]
    |
    +---> [US West Load Balancer] → [Servers]
    |
    +---> [EU Load Balancer] → [Servers]

Failover Scenario:
- US East region fails
- DNS health check detects failure
- DNS returns US West or EU IP
- Traffic routes to healthy region`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          12,
			Title:       "Caching & Performance",
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
					CodeExamples: `Cache-Aside (Lazy Loading) Pattern:

def get_user(user_id):
    # 1. Check cache
    user = cache.get(f"user:{user_id}")
    if user:
        return user
    
    # 2. Cache miss - read from database
    user = database.get_user(user_id)
    
    # 3. Write to cache
    cache.set(f"user:{user_id}", user, ttl=3600)
    
    # 4. Return data
    return user

Pros: Simple, cache only what's accessed
Cons: Cache miss penalty, potential stale data

Write-Through Pattern:

def update_user(user_id, data):
    # 1. Write to cache
    cache.set(f"user:{user_id}", data, ttl=3600)
    
    # 2. Write to database (synchronously)
    database.update_user(user_id, data)
    
    # Both updated together
    return data

Pros: Cache always consistent with DB
Cons: Slower writes (waits for DB)

Write-Behind (Write-Back) Pattern:

def update_user(user_id, data):
    # 1. Write to cache immediately
    cache.set(f"user:{user_id}", data, ttl=3600)
    
    # 2. Queue for async DB write
    async_queue.enqueue(update_db, user_id, data)
    
    return data  # Return immediately

# Background worker processes queue
def background_worker():
    while True:
        task = async_queue.dequeue()
        database.update_user(task.user_id, task.data)

Pros: Fast writes, better performance
Cons: Risk of data loss if cache fails before DB write

Cache Invalidation Strategies:

TTL (Time To Live):
cache.set("key", "value", ttl=3600)  # Expires in 1 hour
# Automatically removed after TTL

LRU (Least Recently Used):
# Evict least recently accessed items when cache full
# Implementation: Use ordered dictionary or Redis with maxmemory-policy=allkeys-lru

Manual Invalidation:
def update_product(product_id, data):
    database.update_product(product_id, data)
    # Explicitly invalidate cache
    cache.delete(f"product:{product_id}")

Cache Warming:
# Pre-populate cache with frequently accessed data
def warm_cache():
    popular_products = database.get_popular_products()
    for product in popular_products:
        cache.set(f"product:{product.id}", product, ttl=3600)`,
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
					CodeExamples: `SQL Database Example (PostgreSQL):

Schema:
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    total DECIMAL(10,2),
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

Complex Query with Joins:
SELECT u.name, o.total, o.status
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.created_at > '2024-01-01'
ORDER BY o.total DESC;

ACID Transaction:
BEGIN;
  UPDATE accounts SET balance = balance - 100 WHERE id = 1;
  UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

NoSQL Document Store Example (MongoDB):

Document Structure:
{
  "_id": ObjectId("..."),
  "email": "user@example.com",
  "name": "John Doe",
  "orders": [
    {
      "order_id": 123,
      "total": 99.99,
      "status": "completed",
      "created_at": ISODate("2024-01-15")
    }
  ],
  "created_at": ISODate("2024-01-01")
}

Query:
db.users.find({
  "orders.total": { $gt: 50 },
  "orders.status": "completed"
})

NoSQL Key-Value Store Example (Redis):

SET user:123:name "John Doe"
SET user:123:email "john@example.com"
GET user:123:name

Hash Structure:
HSET user:123 name "John Doe" email "john@example.com"
HGETALL user:123

NoSQL Column Store Example (Cassandra):

CREATE TABLE user_activity (
    user_id UUID,
    timestamp TIMESTAMP,
    action TEXT,
    details MAP<TEXT, TEXT>,
    PRIMARY KEY (user_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

Query:
SELECT * FROM user_activity 
WHERE user_id = ? 
AND timestamp > ?;

Use Case Comparison:

E-commerce System:
- SQL: User accounts, orders, payments (ACID required)
- NoSQL: Product catalog, reviews, recommendations (flexible schema)

Social Media:
- SQL: User profiles, authentication
- NoSQL: Posts, feeds, activity streams (high volume, flexible)

Analytics Platform:
- SQL: User metadata, configuration
- NoSQL: Time-series data, logs (high write volume)

When to Choose:

Choose SQL when:
- Need complex queries with joins
- ACID transactions required
- Data relationships important
- Strong consistency needed
- Example: Banking, e-commerce orders

Choose NoSQL when:
- High write volume
- Flexible schema needed
- Horizontal scaling required
- Eventual consistency acceptable
- Example: Social feeds, IoT data, logs`,
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
					CodeExamples: `Point-to-Point Pattern:

Producer → Queue → Consumer

Example: Task Queue
Producer sends task:
queue.send("task_queue", {
    "task_id": 123,
    "type": "process_image",
    "image_url": "https://..."
})

Consumer processes task:
task = queue.receive("task_queue")
process_image(task.image_url)
queue.acknowledge(task)  # Remove from queue

Only one consumer processes each message

Pub/Sub Pattern:

Producer → Topic → [Consumer 1, Consumer 2, Consumer 3]

Example: Event Notifications
Producer publishes event:
pubsub.publish("user_events", {
    "event": "user_registered",
    "user_id": 123,
    "timestamp": "2024-01-17T10:00:00Z"
})

Multiple consumers subscribe:
pubsub.subscribe("user_events", email_service_handler)
pubsub.subscribe("user_events", analytics_handler)
pubsub.subscribe("user_events", notification_handler)

Each consumer receives copy of message

RabbitMQ Example:

# Producer
import pika
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='task_queue', durable=True)
channel.basic_publish(
    exchange='',
    routing_key='task_queue',
    body='{"task": "process"}',
    properties=pika.BasicProperties(delivery_mode=2)  # Make message persistent
)

# Consumer
def callback(ch, method, properties, body):
    process_task(body)
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(queue='task_queue', on_message_callback=callback)
channel.start_consuming()

Amazon SQS Example:

# Producer
import boto3
sqs = boto3.client('sqs')
queue_url = sqs.get_queue_url(QueueName='my-queue')['QueueUrl']
sqs.send_message(
    QueueUrl=queue_url,
    MessageBody='{"task": "process"}'
)

# Consumer
response = sqs.receive_message(
    QueueUrl=queue_url,
    MaxNumberOfMessages=10,
    WaitTimeSeconds=20  # Long polling
)
for message in response.get('Messages', []):
    process_task(message['Body'])
    sqs.delete_message(
        QueueUrl=queue_url,
        ReceiptHandle=message['ReceiptHandle']
    )

Redis Pub/Sub Example:

# Publisher
import redis
r = redis.Redis()
r.publish('channel', '{"event": "user_login", "user_id": 123}')

# Subscriber
pubsub = r.pubsub()
pubsub.subscribe('channel')
for message in pubsub.listen():
    if message['type'] == 'message':
        handle_event(message['data'])

Message Queue Architecture:

[Service A] → [Message Queue] → [Service B]
                |
                +→ [Service C]
                |
                +→ [Service D]

Benefits:
- Decoupling: Services don't need to know about each other
- Buffering: Handles traffic spikes
- Reliability: Messages persist if consumer down
- Scalability: Add more consumers easily

Dead Letter Queue (DLQ):
- Failed messages after max retries → DLQ
- Allows manual inspection and reprocessing
- Prevents message loss`,
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
	})
}

