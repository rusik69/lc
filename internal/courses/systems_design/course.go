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
		},
		{
			ID:          21,
			Title:       "Security & Authentication",
			Description: "Learn security best practices, authentication mechanisms, and authorization patterns for secure systems.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "OAuth 2.0",
					Content: `OAuth 2.0 is an authorization framework that allows applications to obtain limited access to user accounts on HTTP services.

**Key Concepts:**
- **Resource Owner**: User who owns the data
- **Client**: Application requesting access
- **Authorization Server**: Issues access tokens
- **Resource Server**: API that holds protected resources

**OAuth 2.0 Flow:**
1. Client redirects user to authorization server
2. User authenticates and authorizes
3. Authorization server returns authorization code
4. Client exchanges code for access token
5. Client uses access token to access resources

**Grant Types:**
- **Authorization Code**: Most secure, for server-side apps
- **Implicit**: Less secure, for client-side apps (deprecated)
- **Client Credentials**: For server-to-server
- **Refresh Token**: Obtain new access tokens

**Security Best Practices:**
- Use HTTPS always
- Store client secrets securely
- Validate redirect URIs
- Use short-lived access tokens
- Implement refresh token rotation
- Use PKCE for mobile apps

**Common Use Cases:**
- Social login (Login with Google/Facebook)
- API access delegation
- Third-party integrations`,
					CodeExamples: `OAuth 2.0 Flow Example:

1. User clicks "Login with Google"
2. Redirect to: https://accounts.google.com/oauth/authorize?
   client_id=YOUR_CLIENT_ID&
   redirect_uri=YOUR_REDIRECT_URI&
   response_type=code&
   scope=email profile

3. User authenticates and authorizes
4. Redirect back with code: 
   YOUR_REDIRECT_URI?code=AUTHORIZATION_CODE

5. Exchange code for token:
   POST https://oauth2.googleapis.com/token
   {
     "code": "AUTHORIZATION_CODE",
     "client_id": "YOUR_CLIENT_ID",
     "client_secret": "YOUR_CLIENT_SECRET",
     "redirect_uri": "YOUR_REDIRECT_URI",
     "grant_type": "authorization_code"
   }

6. Response:
   {
     "access_token": "ACCESS_TOKEN",
     "refresh_token": "REFRESH_TOKEN",
     "expires_in": 3600
   }

7. Use access token:
   GET https://www.googleapis.com/oauth2/v2/userinfo
   Authorization: Bearer ACCESS_TOKEN`,
				},
				{
					Title: "JWT (JSON Web Tokens)",
					Content: `JWT is a compact, URL-safe token format for securely transmitting information between parties as a JSON object.

**JWT Structure:**
- **Header**: Algorithm and token type
- **Payload**: Claims (user data, permissions)
- **Signature**: Verifies token integrity

**Format:** header.payload.signature (base64url encoded)

**Benefits:**
- Stateless (no server-side session storage)
- Self-contained (includes user info)
- Scalable (works across multiple servers)
- Standardized format

**Use Cases:**
- Authentication tokens
- API authorization
- Information exchange
- Single Sign-On (SSO)

**Security Considerations:**
- Use strong signing algorithm (RS256, ES256)
- Keep secrets secure
- Set short expiration times
- Validate signature and expiration
- Use HTTPS for transmission
- Don't store sensitive data in payload

**Token Types:**
- **Access Token**: Short-lived (15-60 minutes)
- **Refresh Token**: Long-lived (days/weeks), stored securely
- **ID Token**: Contains user identity (OpenID Connect)`,
					CodeExamples: `JWT Example:

Header:
{
  "alg": "HS256",
  "typ": "JWT"
}

Payload:
{
  "sub": "1234567890",
  "name": "John Doe",
  "iat": 1516239022,
  "exp": 1516242622
}

Encoded JWT:
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyLCJleHAiOjE1MTYyNDI2MjJ9.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c

Validation Steps:
1. Verify signature
2. Check expiration (exp claim)
3. Validate issuer (iss claim)
4. Check audience (aud claim)
5. Verify not before (nbf claim)`,
				},
				{
					Title: "API Security",
					Content: `API security is critical for protecting backend services and user data from unauthorized access and attacks.

**Security Layers:**

**1. Authentication:**
- Verify user identity
- Methods: API keys, OAuth, JWT, Basic Auth
- Always use HTTPS

**2. Authorization:**
- Control what authenticated users can do
- Role-Based Access Control (RBAC)
- Attribute-Based Access Control (ABAC)

**3. Rate Limiting:**
- Prevent abuse and DoS attacks
- Limit requests per IP/user/API key
- Use token bucket or sliding window
- Return 429 Too Many Requests

**4. Input Validation:**
- Validate and sanitize all inputs
- Prevent injection attacks (SQL, NoSQL, XSS)
- Use parameterized queries
- Validate data types and ranges

**5. Encryption:**
- Use HTTPS/TLS for transport
- Encrypt sensitive data at rest
- Use strong encryption algorithms

**6. API Keys:**
- Unique identifier for API access
- Store securely (hashed)
- Rotate regularly
- Scope permissions

**7. CORS (Cross-Origin Resource Sharing):**
- Control which domains can access API
- Configure allowed origins
- Use credentials carefully

**Best Practices:**
- Use authentication for all endpoints
- Implement least privilege principle
- Log security events
- Monitor for suspicious activity
- Keep dependencies updated
- Use API versioning
- Implement proper error handling (don't leak info)`,
					CodeExamples: `API Security Example:

Rate Limiting:
- 100 requests per minute per API key
- 1000 requests per hour per IP
- Use Redis for distributed rate limiting

API Key Management:
- Generate: random 32-byte key
- Store: Hash with bcrypt/argon2
- Validate: Compare hash on each request
- Rotate: Every 90 days

Input Validation:
// Bad
query = "SELECT * FROM users WHERE id = " + userInput

// Good
query = "SELECT * FROM users WHERE id = ?"
params = [userInput]  // Validated and sanitized

CORS Configuration:
Access-Control-Allow-Origin: https://example.com
Access-Control-Allow-Methods: GET, POST, PUT
Access-Control-Allow-Headers: Authorization, Content-Type
Access-Control-Max-Age: 86400`,
				},
				{
					Title: "Rate Limiting Advanced",
					Content: `Advanced rate limiting techniques for protecting APIs and preventing abuse at scale.

**Rate Limiting Algorithms:**

**1. Token Bucket:**
- Bucket has capacity (tokens)
- Tokens added at fixed rate
- Request consumes token
- If bucket empty, request rejected
- Allows bursts up to bucket size

**2. Leaky Bucket:**
- Requests added to bucket
- Processed at fixed rate
- If bucket full, request rejected
- Smooths out traffic spikes

**3. Fixed Window:**
- Count requests in time window
- Reset counter at window end
- Simple but allows bursts at boundaries

**4. Sliding Window:**
- Track requests in sliding time window
- More accurate than fixed window
- Prevents boundary bursts
- More complex to implement

**5. Distributed Rate Limiting:**
- Use Redis or similar for shared state
- Works across multiple servers
- Consistent limits globally
- Consider Redis atomic operations

**Rate Limiting Strategies:**

**Per User:**
- Limit based on user ID
- Different limits for different user tiers
- Prevents individual abuse

**Per IP:**
- Limit based on client IP
- Prevents IP-based attacks
- May affect users behind NAT

**Per API Key:**
- Limit based on API key
- Different tiers (free, paid, enterprise)
- Easy to manage

**Adaptive Rate Limiting:**
- Adjust limits based on system load
- Reduce limits during high load
- Increase during low load
- Prevents cascading failures

**Implementation Considerations:**
- Use efficient data structures
- Consider memory usage
- Handle race conditions
- Provide clear error messages
- Log rate limit violations`,
					CodeExamples: `Rate Limiting Implementation:

Token Bucket (Redis):
INCR rate_limit:user:123
EXPIRE rate_limit:user:123 60
if count > 100:
    return 429 Too Many Requests

Sliding Window (Redis):
ZADD rate_limit:user:123 timestamp request_id
ZREMRANGEBYSCORE rate_limit:user:123 0 (now - 60)
ZCARD rate_limit:user:123
if count > 100:
    return 429

Distributed Rate Limiting:
- Use Redis with atomic operations
- Lua scripts for atomicity
- Consider Redis Cluster for scale

Rate Limit Headers:
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1609459200
Retry-After: 60`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          22,
			Title:       "Data Processing",
			Description: "Learn batch and stream processing techniques for handling large-scale data processing.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Batch Processing",
					Content: `Batch processing handles large volumes of data by processing them in batches at scheduled intervals.

**Characteristics:**
- Process data in large batches
- Scheduled execution (hourly, daily)
- High throughput
- Latency not critical
- Cost-effective for large volumes

**Use Cases:**
- ETL (Extract, Transform, Load) pipelines
- Data warehousing
- Report generation
- Data aggregation
- Historical data analysis

**Batch Processing Systems:**
- **Hadoop MapReduce**: Distributed batch processing
- **Apache Spark**: Fast batch and stream processing
- **AWS Batch**: Managed batch processing
- **Airflow**: Workflow orchestration

**Batch Processing Patterns:**

**1. Extract, Transform, Load (ETL):**
- Extract data from sources
- Transform (clean, aggregate, enrich)
- Load into data warehouse

**2. MapReduce:**
- Map: Process data in parallel
- Shuffle: Group by key
- Reduce: Aggregate results

**3. Lambda Architecture:**
- Batch layer: Process historical data
- Speed layer: Process real-time data
- Serving layer: Combine results

**Design Considerations:**
- Idempotency (rerun safely)
- Fault tolerance (handle failures)
- Scalability (process large volumes)
- Monitoring (track progress, failures)`,
					CodeExamples: `Batch Processing Example:

Daily ETL Pipeline:
1. Extract: Pull data from production DBs (midnight)
2. Transform: Clean, validate, aggregate
3. Load: Write to data warehouse
4. Generate reports

Spark Batch Job:
val data = spark.read.parquet("s3://data/raw/")
val aggregated = data
  .groupBy("date", "category")
  .agg(sum("amount").as("total"))
aggregated.write.parquet("s3://data/processed/")

Airflow DAG:
extract_task >> transform_task >> load_task`,
				},
				{
					Title: "Stream Processing",
					Content: `Stream processing handles continuous data streams in real-time, processing events as they arrive.

**Characteristics:**
- Process data as it arrives
- Low latency (milliseconds to seconds)
- Continuous processing
- Handle high velocity data
- Real-time insights

**Use Cases:**
- Real-time analytics
- Fraud detection
- Monitoring and alerting
- Event-driven architectures
- Real-time recommendations

**Stream Processing Systems:**
- **Apache Kafka Streams**: Stream processing library
- **Apache Flink**: Distributed stream processing
- **Apache Storm**: Real-time computation
- **AWS Kinesis**: Managed streaming
- **Google Cloud Dataflow**: Unified batch/stream

**Stream Processing Patterns:**

**1. Event Sourcing:**
- Store events as they occur
- Rebuild state from events
- Audit trail built-in

**2. CQRS (Command Query Responsibility Segregation):**
- Separate write and read models
- Optimize each independently
- Event stream as source of truth

**3. Windowing:**
- Process data in time windows
- Tumbling window: Fixed, non-overlapping
- Sliding window: Overlapping windows
- Session window: Activity-based

**4. Stateful Processing:**
- Maintain state across events
- Join streams
- Aggregate over time

**Design Considerations:**
- Exactly-once vs at-least-once processing
- Backpressure handling
- Fault tolerance (checkpointing)
- Scaling (partitioning)`,
					CodeExamples: `Stream Processing Example:

Kafka Streams:
stream
  .filter(record -> record.value() > 100)
  .groupByKey()
  .windowedBy(TimeWindows.of(Duration.ofMinutes(5)))
  .aggregate(...)
  .to("output-topic")

Real-time Fraud Detection:
- Stream: Transaction events
- Process: Check patterns, amounts, locations
- Alert: Flag suspicious transactions
- Latency: < 100ms

Windowing:
- Tumbling: Every 5 minutes, process last 5 min
- Sliding: Every 1 minute, process last 5 min
- Session: Group by user session`,
				},
				{
					Title: "ETL Pipelines",
					Content: `ETL (Extract, Transform, Load) pipelines move and transform data from source systems to destination systems.

**ETL Stages:**

**1. Extract:**
- Read data from source systems
- Sources: Databases, APIs, files, streams
- Handle different formats (CSV, JSON, Parquet)
- Incremental vs full extraction

**2. Transform:**
- Clean data (remove duplicates, handle nulls)
- Validate data (check constraints)
- Enrich data (join with reference data)
- Aggregate data (summarize)
- Format conversion

**3. Load:**
- Write to destination
- Destinations: Data warehouse, data lake, database
- Handle schema changes
- Optimize for query performance

**ETL Patterns:**

**1. Full Load:**
- Extract all data each time
- Simple but inefficient
- Use for small datasets

**2. Incremental Load:**
- Extract only changed data
- Use timestamps or change data capture
- More efficient for large datasets

**3. Change Data Capture (CDC):**
- Capture changes as they occur
- Real-time or near-real-time
- Use database logs or triggers

**ETL Tools:**
- **Apache Airflow**: Workflow orchestration
- **Apache NiFi**: Data flow management
- **Talend**: Enterprise ETL platform
- **AWS Glue**: Managed ETL service
- **dbt**: Transformations in SQL

**Best Practices:**
- Idempotent operations (rerun safely)
- Error handling and retries
- Data quality checks
- Monitoring and alerting
- Documentation
- Version control`,
					CodeExamples: `ETL Pipeline Example:

Extract:
- Source: Production MySQL database
- Method: Read from replica (avoid production load)
- Incremental: WHERE updated_at > last_run

Transform:
- Clean: Remove duplicates, handle nulls
- Validate: Check data types, ranges
- Enrich: Join with user table
- Aggregate: Sum by date, category

Load:
- Destination: Data warehouse (Snowflake)
- Method: Upsert (insert or update)
- Optimize: Partition by date

Airflow DAG:
extract_task = PythonOperator(
    task_id='extract',
    python_callable=extract_from_mysql
)

transform_task = PythonOperator(
    task_id='transform',
    python_callable=transform_data
)

load_task = PythonOperator(
    task_id='load',
    python_callable=load_to_warehouse
)

extract_task >> transform_task >> load_task`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          23,
			Title:       "Search Systems",
			Description: "Learn how to design and implement search systems using Elasticsearch and other search technologies.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "Elasticsearch Fundamentals",
					Content: `Elasticsearch is a distributed search and analytics engine built on Apache Lucene, designed for horizontal scalability and real-time search.

**Key Concepts:**

**1. Index:**
- Collection of documents (like database)
- Can have multiple indices
- Example: products, users, logs

**2. Document:**
- Basic unit of information (like row)
- JSON object
- Stored in index

**3. Mapping:**
- Schema definition (like table schema)
- Defines field types
- Can be dynamic or explicit

**4. Sharding:**
- Split index into shards
- Distributed across nodes
- Enables horizontal scaling

**5. Replication:**
- Copy of shard (replica)
- Provides high availability
- Enables read scaling

**Search Capabilities:**
- Full-text search
- Fuzzy matching
- Faceted search
- Aggregations
- Geospatial queries

**Use Cases:**
- Product search (e-commerce)
- Log analysis
- Full-text search
- Analytics
- Autocomplete

**Architecture:**
- Cluster: Collection of nodes
- Node: Single server instance
- Master node: Manages cluster
- Data node: Stores data
- Coordinating node: Routes requests`,
					CodeExamples: `Elasticsearch Example:

Create Index:
PUT /products
{
  "mappings": {
    "properties": {
      "name": { "type": "text" },
      "price": { "type": "float" },
      "category": { "type": "keyword" }
    }
  }
}

Index Document:
POST /products/_doc
{
  "name": "Laptop",
  "price": 999.99,
  "category": "Electronics"
}

Search:
GET /products/_search
{
  "query": {
    "match": {
      "name": "laptop"
    }
  }
}

Aggregation:
GET /products/_search
{
  "aggs": {
    "avg_price": {
      "avg": { "field": "price" }
    }
  }
}`,
				},
				{
					Title: "Full-Text Search",
					Content: `Full-text search enables searching through text content efficiently, finding documents containing specific words or phrases.

**Search Types:**

**1. Match Query:**
- Analyzes query text
- Finds documents with matching terms
- Handles synonyms, stemming
- Example: "laptop" matches "laptops", "notebook"

**2. Term Query:**
- Exact match (no analysis)
- Case-sensitive
- Use for keywords, IDs

**3. Phrase Query:**
- Matches exact phrase
- Words must appear in order
- Example: "laptop computer"

**4. Fuzzy Query:**
- Handles typos
- Edit distance (Levenshtein)
- Example: "lapto" matches "laptop"

**5. Wildcard Query:**
- Pattern matching
- * matches multiple characters
- ? matches single character
- Example: "lap*" matches "laptop"

**Analysis:**
- **Tokenizer**: Splits text into terms
- **Filters**: Lowercase, remove stop words, stemming
- **Analyzers**: Combination of tokenizer and filters

**Relevance Scoring:**
- TF-IDF: Term frequency × Inverse document frequency
- BM25: Improved scoring algorithm
- Boosting: Increase relevance of certain fields

**Best Practices:**
- Use appropriate analyzers
- Index important fields
- Use filters for exact matches
- Implement autocomplete
- Handle synonyms
- Monitor search performance`,
					CodeExamples: `Full-Text Search Example:

Match Query:
{
  "query": {
    "match": {
      "title": {
        "query": "laptop computer",
        "operator": "and"
      }
    }
  }
}

Multi-Match (search multiple fields):
{
  "query": {
    "multi_match": {
      "query": "laptop",
      "fields": ["title^2", "description"]
    }
  }
}

Fuzzy Query:
{
  "query": {
    "fuzzy": {
      "title": {
        "value": "lapto",
        "fuzziness": "AUTO"
      }
    }
  }
}

Phrase Query:
{
  "query": {
    "match_phrase": {
      "title": "laptop computer"
    }
  }
}`,
				},
				{
					Title: "Ranking Algorithms",
					Content: `Ranking algorithms determine the order of search results, ensuring the most relevant documents appear first.

**Relevance Factors:**

**1. Text Relevance:**
- Term frequency (TF): How often term appears
- Inverse document frequency (IDF): How rare term is
- Field length: Shorter fields more relevant
- BM25: Improved scoring algorithm

**2. Popularity Signals:**
- Click-through rate (CTR)
- Conversion rate
- User ratings
- Sales volume

**3. Freshness:**
- Recent content ranked higher
- Decay function over time
- Balance relevance and freshness

**4. Personalization:**
- User's past behavior
- Preferences
- Location
- Device type

**5. Business Rules:**
- Boost promoted products
- Featured content
- Sponsored results

**Scoring Functions:**
- **BM25**: Default in Elasticsearch
- **Custom Scoring**: Combine multiple factors
- **Learning to Rank**: ML-based ranking

**Ranking Strategies:**

**1. Static Ranking:**
- Pre-computed scores
- Fast but less flexible
- Use for stable content

**2. Dynamic Ranking:**
- Computed at query time
- More flexible
- Slower but more accurate

**3. Hybrid Ranking:**
- Combine static and dynamic
- Best of both worlds
- Common in production

**Optimization:**
- Cache popular queries
- Pre-compute scores
- Use filters when possible
- Monitor ranking quality`,
					CodeExamples: `Ranking Example:

BM25 Scoring:
score = IDF × (TF × (k1 + 1)) / (TF + k1 × (1 - b + b × |d|/avgdl))

Custom Scoring:
{
  "query": {
    "function_score": {
      "query": { "match": { "title": "laptop" } },
      "functions": [
        {
          "filter": { "term": { "featured": true } },
          "weight": 2.0
        },
        {
          "field_value_factor": {
            "field": "rating",
            "factor": 1.2
          }
        }
      ],
      "score_mode": "sum"
    }
  }
}

Learning to Rank:
- Train ML model on click data
- Features: Text relevance, popularity, freshness
- Predict relevance score
- Rank by predicted score`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          24,
			Title:       "Distributed Systems Patterns",
			Description: "Master distributed systems patterns including leader election, distributed locks, and consensus algorithms.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "Leader Election",
					Content: `Leader election ensures only one node acts as leader in a distributed system, coordinating activities and preventing conflicts.

**Why Leader Election?**
- Coordinate distributed tasks
- Prevent split-brain scenarios
- Manage shared resources
- Ensure consistency

**Leader Election Algorithms:**

**1. Bully Algorithm:**
- Highest ID wins
- Node with higher ID challenges current leader
- Simple but can cause multiple elections

**2. Ring Algorithm:**
- Nodes arranged in ring
- Pass token around ring
- First node to receive own token becomes leader
- More efficient but single point of failure

**3. ZooKeeper/etcd:**
- Use distributed coordination service
- Create ephemeral node
- Smallest node becomes leader
- Automatic failover

**4. Raft Consensus:**
- Consensus algorithm with leader
- Leader handles all client requests
- Followers replicate leader's log
- Automatic leader election on failure

**Leader Responsibilities:**
- Coordinate distributed operations
- Make decisions
- Manage shared state
- Handle client requests

**Failure Handling:**
- Detect leader failure (heartbeat timeout)
- Elect new leader
- Transfer leadership gracefully
- Handle split-brain scenarios

**Use Cases:**
- Database primary selection
- Task coordination
- Configuration management
- Distributed locking`,
					CodeExamples: `Leader Election Example:

ZooKeeper Implementation:
1. All nodes try to create /leader node
2. Only one succeeds (becomes leader)
3. Others watch /leader node
4. On leader failure, /leader deleted
5. Remaining nodes compete again

Raft Leader Election:
- Nodes start as followers
- Timeout triggers election
- Candidate requests votes
- Majority vote → leader
- Leader sends heartbeats

etcd Implementation:
etcdctl lock /leader --ttl 60
# Leader holds lock
# On failure, lock expires
# Others can acquire lock`,
				},
				{
					Title: "Distributed Locks",
					Content: `Distributed locks coordinate access to shared resources across multiple nodes, ensuring only one process accesses a resource at a time.

**Use Cases:**
- Prevent concurrent modifications
- Coordinate distributed tasks
- Ensure idempotency
- Manage shared resources

**Lock Properties:**
- **Mutual Exclusion**: Only one holder at a time
- **Deadlock Free**: Locks eventually released
- **Fault Tolerant**: Survives node failures
- **Performance**: Low latency, high throughput

**Distributed Lock Implementations:**

**1. Redis:**
- SET key value NX EX timeout
- Atomic operation
- Simple but single point of failure
- Use Redlock for multi-master

**2. ZooKeeper:**
- Create ephemeral sequential nodes
- Smallest node holds lock
- Automatic release on failure
- More reliable

**3. etcd:**
- Similar to ZooKeeper
- Distributed consensus
- Reliable and fast

**4. Database:**
- Use database locks
- Simple but can be bottleneck
- Good for existing DB infrastructure

**Lock Patterns:**

**1. Simple Lock:**
- Acquire, use resource, release
- Risk: Deadlock if not released

**2. Lock with Timeout:**
- Automatic release after timeout
- Prevents deadlocks
- May release prematurely

**3. Lock with Refresh:**
- Extend lock while holding
- Prevents premature release
- More complex

**Best Practices:**
- Use timeouts
- Implement retry logic
- Handle lock acquisition failures
- Monitor lock contention
- Use idempotent operations`,
					CodeExamples: `Distributed Lock Example:

Redis Lock:
SET lock:resource123 "owner" NX EX 30
# NX = only if not exists
# EX = expire after 30 seconds

Release:
if GET lock:resource123 == "owner":
    DEL lock:resource123

ZooKeeper Lock:
1. Create /locks/resource123/lock-0000000001
2. Check if smallest node
3. If yes, hold lock
4. If no, watch previous node
5. On release, next node acquires

Redlock (Multi-Master):
- Try to acquire lock on majority of Redis instances
- If majority acquired, lock held
- More fault tolerant

Lock with Refresh:
while holding_lock:
    EXPIRE lock:resource123 30
    sleep(10)`,
				},
				{
					Title: "Consensus Algorithms",
					Content: `Consensus algorithms ensure multiple nodes agree on a value or decision, essential for distributed systems consistency.

**Consensus Problem:**
- Multiple nodes must agree on value
- Despite node failures
- Despite network partitions
- Despite message delays

**Consensus Properties:**
- **Safety**: All nodes agree on same value
- **Liveness**: System eventually makes progress
- **Fault Tolerance**: Works despite failures

**Consensus Algorithms:**

**1. Paxos:**
- Classic consensus algorithm
- Complex but proven
- Used in Google Chubby
- Handles minority failures

**2. Raft:**
- Simpler than Paxos
- Easier to understand and implement
- Used in etcd, Consul
- Leader-based approach

**3. PBFT (Practical Byzantine Fault Tolerance):**
- Handles Byzantine failures (malicious nodes)
- More complex
- Used in blockchain systems
- Requires 3f+1 nodes for f failures

**Raft Algorithm:**

**States:**
- **Leader**: Handles client requests
- **Follower**: Replicates leader's log
- **Candidate**: Running for election

**Operations:**
1. Leader receives request
2. Appends to log
3. Replicates to followers
4. Commits when majority acknowledge
5. Applies to state machine

**Leader Election:**
- Timeout triggers election
- Candidate requests votes
- Majority vote → leader
- Leader sends heartbeats

**Use Cases:**
- Distributed databases
- Configuration management
- Service discovery
- Distributed locking`,
					CodeExamples: `Raft Consensus Example:

Leader Election:
1. Follower timeout (no heartbeat)
2. Become candidate, increment term
3. Request votes from all nodes
4. If majority vote → leader
5. Send heartbeats to maintain leadership

Log Replication:
1. Client request → Leader
2. Leader appends to log
3. Leader sends AppendEntries to followers
4. Followers acknowledge
5. Leader commits when majority acknowledge
6. Leader applies to state machine
7. Leader responds to client

Failure Handling:
- Leader fails → election
- Follower fails → continue (majority still works)
- Network partition → majority partition continues`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
