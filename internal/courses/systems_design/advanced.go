package systemsdesign

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSystemsDesignModules([]problems.CourseModule{
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
