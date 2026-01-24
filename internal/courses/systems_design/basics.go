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
					Content: `Systems design is the process of defining the architecture, components, modules, interfaces, and data for a system to satisfy specified requirements. It's a comprehensive discipline that involves making strategic trade-offs between competing concerns like performance, scalability, reliability, security, and cost to build systems that can handle real-world scale and complexity.

**The Essence of Systems Design:**

**What Systems Design Really Is:**
Systems design is not just about drawing boxes and arrows. It's about understanding requirements deeply, making informed architectural decisions, and designing systems that can evolve and scale. It combines engineering principles, architectural patterns, and practical experience to solve complex problems.

**Key Characteristics:**
- **Holistic Approach**: Considers entire system, not just individual components
- **Trade-off Analysis**: Balances competing requirements and constraints
- **Scalability Focus**: Designs for growth and scale
- **Practical**: Grounded in real-world constraints and requirements
- **Iterative**: Evolves as requirements and understanding deepen

**Why Systems Design Matters:**

**1. Professional Impact:**

**Design Scalable Systems:**
- Build systems that handle millions of users
- Design for growth without major rewrites
- Avoid performance bottlenecks before they occur
- **Real-world example**: Designing Twitter to handle 500M tweets/day

**Make Informed Decisions:**
- Understand trade-offs between different approaches
- Choose appropriate technologies and patterns
- Avoid costly mistakes and rework
- **Real-world example**: Choosing between SQL and NoSQL databases

**Build Production-Ready Systems:**
- Design for reliability and fault tolerance
- Plan for monitoring and observability
- Consider operational requirements
- **Real-world example**: Designing for 99.9% uptime with automatic failover

**Reduce Technical Debt:**
- Make decisions that don't require constant refactoring
- Design for maintainability and extensibility
- Avoid architectural shortcuts that cause problems later
- **Real-world example**: Proper service boundaries prevent tight coupling

**2. Career Development:**

**Technical Interviews:**
- Systems design is a core interview topic at top tech companies
- Demonstrates senior-level thinking and problem-solving
- Shows ability to handle ambiguity and make trade-offs
- **Real-world**: Google, Amazon, Meta all include systems design interviews

**Leadership Opportunities:**
- Lead architecture discussions and design reviews
- Mentor junior engineers on design principles
- Make strategic technical decisions
- **Real-world**: Senior engineers and architects need strong systems design skills

**Career Advancement:**
- Systems design skills are essential for senior roles
- Enables transition from developer to architect
- Opens opportunities in high-scale systems
- **Real-world**: Many tech leaders started with strong systems design skills

**3. Business Value:**

**Cost Optimization:**
- Design efficient systems that minimize infrastructure costs
- Right-size components and resources
- Optimize for cost-effective scaling
- **Real-world example**: Proper caching can reduce database costs by 80%

**Faster Feature Development:**
- Good architecture enables rapid feature development
- Reduces time spent on refactoring and bug fixes
- Enables parallel development by multiple teams
- **Real-world example**: Microservices enable teams to work independently

**Improved User Experience:**
- Fast response times improve user satisfaction
- High availability ensures users can access system
- Scalability prevents degradation during peak usage
- **Real-world example**: Slow systems lead to user churn

**Business Growth Support:**
- Systems that scale support business growth
- Avoid system limitations that constrain business
- Enable new business opportunities
- **Real-world example**: Scalable architecture enables international expansion

**Modern Systems Design Framework: RESHADED (2024-2025 Best Practices):**

The RESHADED framework provides a structured approach to systems design interviews and real-world design problems. It helps translate open-ended problems into specific requirements, constraints, and success metrics.

**RESHADED Framework Components:**

**R - Requirements:**
- Functional requirements (what the system should do)
- Non-functional requirements (performance, scalability, reliability)
- Clarify ambiguous requirements with interviewer/stakeholder

**E - Estimate:**
- Scale estimation (users, requests, data volume)
- Capacity planning (storage, bandwidth, compute)
- Use back-of-envelope calculations

**S - System Interface:**
- Define APIs and interfaces
- Input/output specifications
- Data formats and protocols

**H - High-Level Design:**
- Overall architecture (monolith vs microservices)
- Major components and their interactions
- Data flow and request flow

**A - Algorithm/Data Structure:**
- Core algorithms for critical operations
- Data structures for efficient storage/retrieval
- Optimization strategies

**D - Detailed Design:**
- Deep dive into specific components
- Database schema design
- API design details
- Error handling and edge cases

**E - Evaluation:**
- Identify bottlenecks and limitations
- Discuss trade-offs
- Suggest improvements and optimizations

**D - Discussion:**
- Scalability considerations
- Reliability and fault tolerance
- Security considerations
- Monitoring and observability

**Key Goals of Systems Design:**

**1. Scalability:**
- **Definition**: System's ability to handle growth
- **Types**:
  - **Horizontal Scaling**: Add more instances (preferred, cloud-native)
  - **Vertical Scaling**: Increase instance size (limited, legacy approach)
- **Considerations**: 
  - Current scale vs future scale
  - Read-heavy vs write-heavy workloads
  - Stateless vs stateful components
  - Database scaling strategies (sharding, replication)
- **Patterns**: Load balancing, caching, database partitioning, CDN
- **Real-world example**: Designing for 1M users but planning for 100M

**2. Reliability:**
- **Definition**: System works correctly and consistently
- **Aspects**:
  - **Correctness**: System produces correct results
  - **Consistency**: Data remains consistent
  - **Durability**: Data is not lost
- **Techniques**: 
  - Redundancy, replication, error handling
  - Data validation, transaction management
- **Real-world example**: Financial systems require high reliability

**3. Availability:**
- **Definition**: System is accessible when needed
- **Metrics**: 
  - **Uptime**: Percentage of time system is available
  - **SLA**: Service Level Agreement (e.g., 99.9% = 8.76 hours downtime/year)
- **Techniques**: 
  - Redundancy, failover, health checks
  - Multiple data centers, load balancing
- **Real-world example**: E-commerce sites need high availability during sales

**4. Performance:**
- **Definition**: System responds quickly to requests
- **Metrics**: 
  - **Latency**: Time for single request (e.g., < 200ms)
  - **Throughput**: Requests per second (e.g., 10,000 QPS)
  - **Response Time**: End-to-end time for user request
- **Techniques**: 
  - Caching, CDN, database optimization
  - Load balancing, horizontal scaling
- **Real-world example**: Search engines need sub-second response times

**5. Maintainability:**
- **Definition**: System is easy to understand, modify, and extend
- **Aspects**:
  - **Code Quality**: Clean, well-documented code
  - **Modularity**: Clear separation of concerns
  - **Testability**: Easy to test individual components
  - **Documentation**: Clear architecture and design docs
- **Techniques**: 
  - Microservices for independent deployment
  - API versioning for backward compatibility
  - Comprehensive logging and monitoring
- **Real-world example**: Systems that enable rapid feature development

**6. Cost Efficiency:**
- **Definition**: Optimize infrastructure and operational costs
- **Considerations**:
  - Right-sizing resources (not over-provisioning)
  - Using managed services vs self-hosted
  - Caching to reduce database load
  - Efficient data storage and retrieval
- **Techniques**:
  - Auto-scaling to match demand
  - Reserved instances for predictable workloads
  - Data compression and archiving
  - Cost monitoring and optimization
- **Real-world example**: Reducing cloud costs by 40% through proper caching

**Modern Systems Design Patterns (2024-2025):**

**Architectural Patterns:**
- **Microservices**: Independent, scalable services
- **Event-Driven Architecture**: Loose coupling via events
- **Serverless**: Function-as-a-Service for event processing
- **Service Mesh**: Advanced traffic management (Istio, Linkerd)
- **API Gateway**: Single entry point for microservices

**Scalability Patterns:**
- **Sharding**: Horizontal database partitioning
- **Replication**: Multiple copies for read scaling
- **Caching**: Multi-layer caching (CDN, application, database)
- **Load Balancing**: Distribute traffic across instances
- **Database Patterns**: Read replicas, write-through caching, CQRS

**Reliability Patterns:**
- **Circuit Breaker**: Prevent cascade failures
- **Retry with Exponential Backoff**: Handle transient failures
- **Bulkhead**: Isolate failures to specific components
- **Health Checks**: Monitor component health
- **Graceful Degradation**: Maintain partial functionality during failures

**Data Consistency Patterns:**
- **ACID**: Strong consistency (traditional databases)
- **BASE**: Eventually consistent (NoSQL, distributed systems)
- **CAP Theorem**: Choose 2 of 3 (Consistency, Availability, Partition tolerance)
- **Eventual Consistency**: Acceptable for many use cases
- **CQRS**: Separate read and write models 
  - **Modularity**: Well-defined components and interfaces
  - **Documentation**: Clear documentation and code comments
  - **Testability**: Easy to test individual components
- **Techniques**: 
  - Clean architecture, SOLID principles
  - Comprehensive testing, good documentation
- **Real-world example**: Maintainable systems reduce development costs

**6. Security:**
- **Definition**: System protects data and prevents unauthorized access
- **Aspects**: 
  - **Authentication**: Verifying user identity
  - **Authorization**: Controlling access to resources
  - **Encryption**: Protecting data at rest and in transit
  - **Compliance**: Meeting regulatory requirements
- **Techniques**: 
  - Encryption, access control, security monitoring
  - Regular security audits, penetration testing
- **Real-world example**: Healthcare systems must protect patient data (HIPAA)

**7. Cost Efficiency:**
- **Definition**: System operates within budget constraints
- **Considerations**: 
  - Infrastructure costs (servers, storage, bandwidth)
  - Development costs (team size, timeline)
  - Operational costs (monitoring, maintenance)
- **Techniques**: 
  - Right-sizing resources, using managed services
  - Optimizing for cost-effective scaling
- **Real-world example**: Startups need cost-efficient designs

**Common System Requirements:**

**Functional Requirements:**
- **Definition**: What the system should do
- **Examples**: 
  - User registration and authentication
  - Product catalog browsing and search
  - Shopping cart and checkout
  - Order processing and tracking
- **Documentation**: User stories, use cases, API specifications
- **Real-world**: E-commerce system must allow users to browse products and make purchases

**Non-Functional Requirements:**

**Performance Requirements:**
- **Response Time**: API response time < 200ms
- **Throughput**: Handle 10,000 requests per second
- **Latency**: Database query latency < 50ms
- **Real-world**: Search results must appear in < 1 second

**Scalability Requirements:**
- **Current Scale**: 1 million users, 100K requests/day
- **Future Scale**: 100 million users, 10M requests/day
- **Growth Rate**: 10x growth expected in 2 years
- **Real-world**: System must scale without major redesign

**Availability Requirements:**
- **Uptime**: 99.9% availability (8.76 hours downtime/year)
- **RTO**: Recovery Time Objective < 1 hour
- **RPO**: Recovery Point Objective < 15 minutes
- **Real-world**: Financial systems need 99.99% uptime

**Security Requirements:**
- **Authentication**: OAuth2, JWT tokens
- **Authorization**: Role-based access control (RBAC)
- **Encryption**: TLS 1.3 for data in transit, AES-256 for data at rest
- **Compliance**: GDPR, HIPAA, PCI-DSS
- **Real-world**: Healthcare systems must comply with HIPAA

**Constraints:**

**Budget Constraints:**
- **Infrastructure**: $50K/month for cloud infrastructure
- **Development**: Team of 10 developers, 6-month timeline
- **Licensing**: Database licensing costs
- **Real-world**: Startups have limited budgets

**Time Constraints:**
- **Launch Deadline**: Must launch in 6 months
- **Market Window**: Launch before competitor
- **Resource Availability**: Key developers available for 3 months
- **Real-world**: Time-to-market is critical for competitive advantage

**Technology Constraints:**
- **Existing Systems**: Must integrate with legacy mainframe
- **Vendor Requirements**: Enterprise agreement requires specific cloud provider
- **Team Expertise**: Team knows Java and Python
- **Real-world**: Organizations have existing technology investments

**Regulatory Constraints:**
- **Data Residency**: EU data must stay in EU regions
- **Compliance**: Must comply with GDPR, HIPAA
- **Audit Requirements**: Comprehensive audit trails required
- **Real-world**: Global companies must comply with multiple regulations

**Assumptions:**

**Usage Patterns:**
- **Read/Write Ratio**: 10:1 read to write ratio
- **Peak Times**: 80% of traffic during business hours
- **Geographic Distribution**: 60% US, 30% Europe, 10% Asia
- **Real-world**: Understanding usage patterns informs design decisions

**Growth Rates:**
- **Users**: 10% month-over-month growth
- **Data**: 1TB new data per month
- **Traffic**: 5% month-over-month growth
- **Real-world**: Planning for growth prevents future problems

**User Behavior:**
- **Session Duration**: Average 15 minutes per session
- **Actions per Session**: Average 10 actions per session
- **Peak Concurrent Users**: 100K concurrent users during peak hours
- **Real-world**: Understanding user behavior helps design better systems

**Systems Design Process:**

**1. Requirements Gathering:**

**Understand the Problem:**
- What problem are we solving?
- Who are the users?
- What are the use cases?
- **Example**: "Design a URL shortener like TinyURL"

**Clarify Requirements:**
- Ask clarifying questions
- Identify edge cases
- Understand constraints
- **Example**: "Do we need analytics? Custom URLs? Expiration?"

**Document Requirements:**
- Functional requirements
- Non-functional requirements
- Constraints and assumptions
- **Example**: Document all requirements before designing

**2. Capacity Estimation:**

**Traffic Estimation:**
- **QPS (Queries Per Second)**: Peak queries per second
- **RPS (Reads Per Second)**: Read operations per second
- **WPS (Writes Per Second)**: Write operations per second
- **Example**: 100M URLs/day = ~1,200 QPS average, ~2,400 QPS peak

**Storage Estimation:**
- **Data Size**: Size of each record
- **Total Records**: Expected number of records
- **Growth Rate**: Expected growth over time
- **Example**: 100M URLs × 500 bytes = 50 GB, 5 years = 500 GB

**Bandwidth Estimation:**
- **Request Size**: Average request size
- **Response Size**: Average response size
- **Traffic**: Requests per second × size
- **Example**: 2,400 QPS × 1 KB = 2.4 MB/s = 19.2 Mbps

**3. System API Design:**

**Define Endpoints:**
- RESTful API design
- Clear endpoint naming
- HTTP methods (GET, POST, PUT, DELETE)
- **Example**: POST /api/v1/urls (create), GET /api/v1/urls/{shortUrl} (redirect)

**Request/Response Formats:**
- JSON for data exchange
- Clear field names and types
- Error response format
- **Example**: { "longUrl": "https://...", "shortUrl": "abc123" }

**Versioning:**
- API versioning strategy
- Backward compatibility
- Deprecation policy
- **Example**: /api/v1/urls, /api/v2/urls

**4. Database Design:**

**Choose Database Type:**
- **SQL**: Relational databases (PostgreSQL, MySQL)
  - Use for: Structured data, ACID transactions, complex queries
- **NoSQL**: Non-relational databases (MongoDB, DynamoDB, Cassandra)
  - Use for: Unstructured data, high scale, flexible schema
- **Example**: URL shortener might use NoSQL for scale

**Schema Design:**
- Tables/collections and fields
- Indexes for query optimization
- Relationships between entities
- **Example**: URLs table: id, longUrl, shortUrl, createdAt, expiresAt

**Scaling Strategy:**
- **Replication**: Read replicas for read scaling
- **Sharding**: Partition data across multiple databases
- **Caching**: Cache frequently accessed data
- **Example**: Shard by shortUrl hash, use Redis cache

**5. High-Level Design:**

**Identify Components:**
- Major system components
- Component responsibilities
- Component interactions
- **Example**: Load Balancer → API Servers → Application Servers → Database

**Define Interactions:**
- How components communicate
- Synchronous vs asynchronous
- Protocols and formats
- **Example**: REST APIs for synchronous, message queue for asynchronous

**Plan for Scalability:**
- Horizontal scaling strategy
- Load balancing approach
- Caching strategy
- **Example**: Stateless API servers behind load balancer

**6. Detailed Design:**

**Component Deep Dive:**
- Internal architecture of each component
- Algorithms and data structures
- Error handling and edge cases
- **Example**: URL shortening algorithm (base62 encoding, distributed ID generation)

**Data Flow:**
- How data flows through system
- Data transformations
- Error handling and retries
- **Example**: User request → Load Balancer → API Server → Cache → Database

**Security Design:**
- Authentication and authorization
- Data encryption
- Input validation
- **Example**: API Gateway handles authentication, services validate input

**7. Identify and Address Bottlenecks:**

**Common Bottlenecks:**
- **Database**: Often the bottleneck (slow queries, connection limits)
- **Network**: Bandwidth limitations, latency
- **CPU**: Computation-intensive operations
- **Memory**: Insufficient RAM, memory leaks
- **Disk I/O**: Slow disk reads/writes

**Optimization Strategies:**
- **Database**: Indexing, query optimization, read replicas, caching
- **Network**: CDN, compression, connection pooling
- **CPU**: Algorithm optimization, parallel processing
- **Memory**: Caching, memory optimization
- **Disk**: SSDs, database optimization

**Monitoring and Scaling:**
- **Metrics**: Monitor key metrics (latency, throughput, error rate)
- **Alerts**: Set up alerts for anomalies
- **Auto Scaling**: Automatically scale based on load
- **Example**: CloudWatch metrics trigger Auto Scaling

**Common Pitfalls:**

**1. Over-engineering:**
- **Problem**: Designing for scale you don't need
- **Symptoms**: Unnecessary complexity, high costs, slow development
- **Solution**: Start simple, optimize when needed
- **Example**: Using microservices for a small application

**2. Under-engineering:**
- **Problem**: Not planning for growth
- **Symptoms**: System breaks under load, requires major rewrites
- **Solution**: Plan for reasonable growth, consider requirements
- **Example**: Monolithic design that can't scale

**3. Ignoring Trade-offs:**
- **Problem**: Not understanding consequences of choices
- **Symptoms**: Optimizing for wrong metrics, unexpected costs
- **Solution**: Explicitly discuss trade-offs, measure what matters
- **Example**: Choosing consistency over availability when availability matters more

**4. Single Point of Failure:**
- **Problem**: Component failure brings down entire system
- **Symptoms**: System outages, data loss
- **Solution**: Design for failure, add redundancy
- **Example**: Single database server with no backup

**5. Poor Scalability Planning:**
- **Problem**: System can't scale when needed
- **Symptoms**: Performance degradation, high costs
- **Solution**: Design stateless services, plan scaling strategy
- **Example**: Stateful services that can't scale horizontally

**6. Not Considering Costs:**
- **Problem**: Design is too expensive to operate
- **Symptoms**: High infrastructure costs, budget overruns
- **Solution**: Consider costs in design decisions, optimize for cost
- **Example**: Over-provisioning resources "just in case"

**Best Practices:**

**1. Start with Requirements:**
- Understand the problem deeply before designing solution
- Ask clarifying questions to fill gaps
- Document assumptions explicitly
- Validate requirements with stakeholders
- **Example**: "Do we need real-time updates or is eventual consistency OK?"

**2. Think in Layers:**
- **Client Layer**: Mobile apps, web browsers
- **API Layer**: REST APIs, GraphQL, gRPC
- **Application Layer**: Business logic, services
- **Data Layer**: Databases, caches, message queues
- **Infrastructure Layer**: Servers, networks, load balancers
- **Example**: Clear separation enables independent scaling

**3. Design for Scale:**
- **Horizontal Scaling**: Prefer adding instances over increasing size
- **Stateless Services**: Enable horizontal scaling
- **Caching**: Reduce load on databases and services
- **Database Scaling**: Read replicas, sharding, partitioning
- **Example**: Stateless API servers scale horizontally easily

**4. Design for Failure:**
- **Assume Failures**: Components will fail, plan for it
- **Add Redundancy**: Multiple instances, multiple data centers
- **Circuit Breakers**: Prevent cascading failures
- **Graceful Degradation**: System continues with reduced functionality
- **Example**: If database is down, serve cached data with warning

**5. Keep it Simple:**
- **Avoid Over-Complication**: Don't add complexity prematurely
- **Use Proven Patterns**: Leverage well-known patterns and practices
- **Avoid Premature Optimization**: Optimize when you have data
- **Iterate and Improve**: Start simple, evolve based on needs
- **Example**: Start with monolith, evolve to microservices if needed

**6. Measure and Monitor:**
- **Key Metrics**: Latency, throughput, error rate, availability
- **Monitoring**: Comprehensive monitoring and alerting
- **Logging**: Structured logging for debugging
- **Observability**: Understand system behavior
- **Example**: CloudWatch, Datadog, Prometheus for monitoring

**Real-World System Design Examples:**

**1. Twitter-like System:**
- **Requirements**: Post tweets, follow users, view timeline
- **Scale**: 300M users, 500M tweets/day, 200K tweets/second peak
- **Key Challenges**: 
  - Timeline generation (fan-out problem)
  - Real-time updates
  - High read volume
- **Solutions**: 
  - Fan-out on write (push model) for active users
  - Fan-out on read (pull model) for inactive users
  - Caching (Redis) for hot data
  - Read replicas for database scaling
- **Architecture**: Microservices, message queues, caching layers

**2. URL Shortener (TinyURL):**
- **Requirements**: Shorten URLs, redirect to original URL
- **Scale**: 100M URLs/day, 10:1 read/write ratio, 1,200 QPS average
- **Key Challenges**: 
  - Unique ID generation at scale
  - High read volume (redirects)
  - URL expiration and cleanup
- **Solutions**: 
  - Base62 encoding for short URLs
  - Distributed ID generation (Snowflake algorithm)
  - Caching (Redis) for hot URLs
  - CDN for global distribution
- **Architecture**: Load balancer, API servers, cache, database

**3. Chat System (WhatsApp-like):**
- **Requirements**: Send messages, group chats, presence, delivery status
- **Scale**: 2B users, 100B messages/day, real-time delivery
- **Key Challenges**: 
  - Real-time message delivery
  - Group chat scalability
  - Message ordering and consistency
  - Offline message delivery
- **Solutions**: 
  - WebSocket connections for real-time
  - Message queues for reliable delivery
  - Fan-out for group messages
  - Database for message persistence
- **Architecture**: WebSocket servers, message queues, databases, presence servers

**4. Video Streaming (YouTube-like):**
- **Requirements**: Upload videos, stream videos, recommendations
- **Scale**: 2B users, 500 hours video uploaded/minute, billions of views/day
- **Key Challenges**: 
  - Large file storage and transfer
  - Video encoding and processing
  - Global content delivery
  - Recommendation algorithm
- **Solutions**: 
  - Object storage (S3) for video files
  - CDN for global distribution
  - Transcoding pipeline for multiple formats
  - Distributed processing for recommendations
- **Architecture**: Upload service, transcoding pipeline, CDN, recommendation service

**Conclusion:**

Systems design is a critical skill for building scalable, reliable, and maintainable systems. It requires understanding requirements deeply, making informed trade-offs, and designing systems that can evolve and scale. Whether you're designing a small web application or a large-scale distributed system, the principles of systems design apply.

Key principles:
- **Start with requirements** - understand the problem first
- **Design for scale** - plan for growth from the beginning
- **Design for failure** - assume components will fail
- **Keep it simple** - avoid unnecessary complexity
- **Measure everything** - monitor and optimize based on data
- **Make trade-offs explicit** - understand consequences of decisions

Remember: There's no perfect system design. Every design involves trade-offs. The goal is to make informed decisions that balance competing concerns and meet your specific requirements. Good systems design enables you to build systems that scale, perform well, and support business goals while remaining maintainable and cost-effective.`,
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

