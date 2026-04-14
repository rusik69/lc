package systemsdesign

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSystemsDesignModules([]problems.CourseModule{
		{
			ID:          2410,
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
			ID:          2411,
			Title:       "Security & Authentication",
			Description: "Learn security best practices, authentication mechanisms, and authorization patterns for secure systems.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "OAuth 2.0",
					Content: `**1. What Is OAuth 2.0 and Why Does It Exist?**

OAuth 2.0 is the industry-standard authorization framework that allows third-party applications to obtain limited access to user accounts on HTTP services without ever seeing the user's password. Think of it like a valet key for your car: you hand the valet a special key that can start the engine and drive a short distance, but cannot open the trunk or the glove compartment. OAuth 2.0 works the same way — it lets an application access specific pieces of your data on another service, without giving that application your full credentials.

Before OAuth existed, if you wanted a third-party app to access your email contacts, you would literally hand over your email password. This was dangerous: the app could read all your emails, change your password, or do anything it wanted. OAuth was invented to solve this trust problem by introducing a token-based delegation model. The user authenticates directly with the service they trust (like Google), and the service issues a limited-scope token to the third-party app.

**2. The Four Key Roles in OAuth 2.0**

Understanding OAuth requires knowing the four actors involved. The **Resource Owner** is the end user who owns the data — for example, a person with a Google account. The **Client** is the application requesting access to that data, such as a calendar app that wants to read your Google Calendar events. The **Authorization Server** is the service that authenticates the user and issues access tokens — in our example, this is Google's OAuth server at accounts.google.com. Finally, the **Resource Server** is the API that actually holds the protected data — Google Calendar's API endpoint. In many deployments the Authorization Server and Resource Server are run by the same organization, but architecturally they are distinct roles.

**3. The Authorization Code Flow — Step by Step**

The most secure and widely used OAuth flow is the Authorization Code grant. It works in five stages. First, the client application redirects the user's browser to the authorization server's login page, including its client_id, a redirect_uri, the requested scopes, and a random state parameter for CSRF protection. Second, the user authenticates (enters their username and password directly on the authorization server, never on the third-party app) and reviews what permissions the app is requesting. Third, upon approval, the authorization server redirects the browser back to the client's redirect_uri with a short-lived authorization code appended as a query parameter. Fourth, the client's backend server exchanges this authorization code — along with its client_secret — for an access token by making a server-to-server POST request to the authorization server's token endpoint. Fifth, the client uses the access token in the Authorization header of API requests to the resource server to access the user's data.

This flow is designed so that the access token is never exposed in the browser's URL bar or history. The authorization code that does appear in the URL is short-lived (typically 60 seconds) and can only be used once.

**4. Grant Types and When to Use Each**

OAuth 2.0 defines several grant types for different scenarios. The **Authorization Code** grant (described above) is the gold standard for server-side web applications because the client secret stays on the server. The **Authorization Code with PKCE** (Proof Key for Code Exchange) extends this for mobile and single-page apps that cannot securely store a client secret; instead, they use a cryptographic code verifier and challenge to prove they initiated the flow. The **Client Credentials** grant is used for machine-to-machine communication where no user is involved — for example, a backend service calling another backend service's API. The **Refresh Token** grant allows a client to obtain a new access token when the current one expires, without forcing the user to log in again. The older **Implicit** grant, which returned tokens directly in the URL fragment, is now deprecated because it exposed tokens to interception and replay attacks.

**5. Security Best Practices and Common Pitfalls**

Always use HTTPS for every OAuth interaction — tokens transmitted over plain HTTP can be intercepted trivially. Store client secrets in secure vaults (such as AWS Secrets Manager or HashiCorp Vault), never in source code or client-side JavaScript. Validate redirect URIs strictly against a whitelist; open redirectors are one of the most common OAuth vulnerabilities, allowing attackers to intercept authorization codes. Use short-lived access tokens (15 minutes is typical) paired with refresh tokens that are rotated on each use to limit the damage window if a token is stolen. For mobile and single-page applications, always use PKCE to prevent authorization code interception attacks. Finally, request the minimum scopes necessary — a calendar app should not request access to the user's email, contacts, and drive.

**6. Real-World Use Cases**

OAuth 2.0 powers the "Login with Google/Facebook/GitHub" buttons seen across the web, enabling social login without sites needing to manage passwords. It enables API access delegation, allowing services like Zapier to connect your Slack, Gmail, and Trello without knowing any of your passwords. It underpins third-party integrations in platforms like Salesforce, where marketplace apps need limited access to customer data. In microservice architectures, the Client Credentials grant is used for secure service-to-service authentication, ensuring that only authorized internal services can call sensitive APIs.`,
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
					Content: `**1. What Is a JWT and Why Is It So Popular?**

A JSON Web Token (JWT, pronounced "jot") is a compact, URL-safe token format for securely transmitting claims between two parties. Think of a JWT like a tamper-evident envelope: you can see what is written on it (the claims are base64-encoded, not encrypted), but any attempt to alter the contents will break the cryptographic seal. JWTs have become the de facto standard for authentication in modern web applications and APIs because they are stateless — the server does not need to store session data in a database or in-memory store. This is a massive advantage for horizontally scaled systems: any server in the cluster can validate the token independently, without making a database call.

Before JWTs, most web applications used server-side sessions: the server stored session data (user ID, roles, preferences) in memory or a database, and gave the client a session ID cookie. Every request required a database lookup. When you add more servers behind a load balancer, you need sticky sessions or a shared session store (like Redis), adding complexity. JWTs eliminate this entirely — the token itself carries the session data.

**2. The Three-Part Structure of a JWT**

A JWT consists of three parts separated by dots: header.payload.signature, each base64url-encoded. The **Header** specifies the signing algorithm (such as HS256 for HMAC-SHA256 or RS256 for RSA-SHA256) and the token type ("JWT"). The **Payload** contains the claims — key-value pairs that carry the actual data. Standard claims include "sub" (subject, typically user ID), "iat" (issued at timestamp), "exp" (expiration timestamp), "iss" (issuer), and "aud" (audience). You can also add custom claims like roles or permissions. The **Signature** is computed by taking the encoded header and payload, concatenating them with a dot, and signing with the specified algorithm using a secret key (for HMAC) or a private key (for RSA/ECDSA). This signature is what makes the token tamper-proof: if anyone modifies the header or payload, the signature will not match during verification.

**3. Why Statelessness Matters for Scale**

The stateless nature of JWTs is their superpower in distributed systems. Imagine you have 50 API servers behind a load balancer. With traditional sessions, every server needs access to the session store, creating a shared dependency and potential bottleneck. With JWTs, each server has the signing key (or the public key for asymmetric algorithms) and can verify any token independently in microseconds. This makes JWTs ideal for microservice architectures where a token issued by the authentication service needs to be validated by dozens of downstream services without any of them calling back to the auth service.

However, statelessness also has a downside: you cannot easily revoke a JWT before it expires. If a user's account is compromised, you cannot "invalidate" their token the way you would delete a server-side session. Common mitigations include keeping access token lifetimes very short (15 minutes) and maintaining a small revocation list for emergency cases.

**4. Token Types in a Modern Auth System**

A well-designed authentication system uses three token types working together. **Access Tokens** are short-lived JWTs (typically 15-60 minutes) that are sent with every API request in the Authorization header. Their short lifetime limits the window of exposure if stolen. **Refresh Tokens** are long-lived (days or weeks) and are stored securely (in an HTTP-only cookie or secure storage on mobile). When the access token expires, the client sends the refresh token to the authorization server to obtain a new access token without requiring the user to log in again. Refresh tokens should be rotated on each use and stored server-side to enable revocation. **ID Tokens** are a concept from OpenID Connect (an identity layer built on OAuth 2.0); they contain verified identity claims about the user and are used by the client application itself, not sent to APIs.

**5. Security Best Practices**

Always use strong asymmetric signing algorithms (RS256 or ES256) for production systems, because the public key can be distributed to all verifying services without exposing the private signing key. Never store sensitive data (passwords, SSNs, credit cards) in the JWT payload — remember, the payload is only encoded, not encrypted, and anyone can decode it with a simple base64 decoder. Always validate the signature, expiration ("exp"), issuer ("iss"), and audience ("aud") claims on every request. Transmit JWTs only over HTTPS to prevent interception. Be aware of the "alg: none" attack — some libraries historically accepted tokens with no signature if the algorithm header was set to "none," so always explicitly specify allowed algorithms during verification.

**6. Common Use Cases**

JWTs power authentication flows across the modern web. They are the token format used in OAuth 2.0 and OpenID Connect for API authorization and social login. They enable Single Sign-On (SSO) across multiple applications in an enterprise — a user logs into the identity provider once, and the JWT is accepted by all participating applications. They are used for secure information exchange between microservices, carrying claims about the request context (user identity, tenant, permissions) through the call chain. They also appear in passwordless authentication flows (magic links) and in mobile app authentication where maintaining server-side sessions is impractical.`,
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
					Content: `**1. Why API Security Is a First-Class Architectural Concern**

API security is not an afterthought you bolt on before launch — it is a foundational design concern that shapes how you build, deploy, and operate backend services. Modern applications expose rich APIs that mobile apps, single-page applications, third-party partners, and internal microservices all consume. Every one of those API endpoints is an attack surface. A single unprotected endpoint can lead to massive data breaches (as companies like Facebook, T-Mobile, and Optus have painfully demonstrated). API security is about building defense in depth: multiple layers of protection so that if one layer fails, others still protect your users and data.

**2. Authentication — Verifying Who Is Calling**

Authentication answers the question "who are you?" Every API request must be authenticated — there should be no anonymous access to any endpoint that returns or modifies user data. The most common authentication mechanisms include API keys (simple but limited — they identify the application, not the user), OAuth 2.0 access tokens (the standard for delegated user authorization), JWTs (stateless tokens that carry identity claims), and mutual TLS (mTLS) for service-to-service communication in zero-trust networks. Basic Auth (username:password in a header) should only be used over HTTPS and is generally reserved for simple internal tools. The cardinal rule is: always use HTTPS. Transmitting any credential over plain HTTP is equivalent to shouting your password in a crowded room.

**3. Authorization — Controlling What Authenticated Users Can Do**

Authentication tells you who the caller is; authorization tells you what they are allowed to do. Role-Based Access Control (RBAC) assigns users to roles (admin, editor, viewer) and grants permissions to roles — it is simple and works well for most applications. Attribute-Based Access Control (ABAC) makes decisions based on attributes of the user, the resource, and the environment (e.g., "managers can approve expenses under $10,000 during business hours") — it is more flexible but more complex. Relationship-Based Access Control (ReBAC), as used in Google Zanzibar, models permissions as relationships between subjects and objects. Whichever model you choose, always implement the principle of least privilege: every user and service should have the minimum permissions necessary to perform their function, and no more.

**4. Rate Limiting — The Bouncer at the Door**

Rate limiting prevents abuse, protects against denial-of-service attacks, and ensures fair usage across all clients. Without rate limiting, a single misbehaving client (or attacker) can overwhelm your servers and degrade service for everyone. Rate limits are typically enforced per IP address, per user, or per API key, using algorithms like token bucket or sliding window (covered in depth in the Rate Limiting Advanced lesson). When a client exceeds the limit, the API returns a 429 Too Many Requests response with a Retry-After header telling the client when to try again. Different tiers of service (free, paid, enterprise) can have different limits, creating a natural monetization lever.

**5. Input Validation — Never Trust the Client**

Every piece of data that arrives in an API request — URL parameters, query strings, headers, request bodies — must be treated as potentially malicious until validated. SQL injection, NoSQL injection, cross-site scripting (XSS), and command injection attacks all exploit insufficient input validation. The golden rule is to use parameterized queries (prepared statements) for all database access — never concatenate user input into SQL strings. Validate data types, lengths, ranges, and formats on the server side (client-side validation is a UX convenience, not a security measure). Use allowlists rather than denylists when possible: instead of trying to block all dangerous characters, define exactly what is allowed.

**6. Encryption — Protecting Data in Motion and at Rest**

Use HTTPS (TLS 1.2 or 1.3) for all API traffic without exception. This encrypts data in transit, preventing eavesdropping and man-in-the-middle attacks. For sensitive data at rest (passwords, payment information, personal data), use strong encryption algorithms (AES-256) and proper key management. Passwords should never be encrypted — they should be hashed using a slow, salted hashing algorithm like bcrypt or Argon2, so that even if the database is stolen, passwords cannot be recovered.

**7. API Keys, CORS, and Operational Security**

API keys are unique identifiers for API access. Store them hashed (like passwords) in your database, rotate them regularly (every 90 days is common), and scope their permissions narrowly. CORS (Cross-Origin Resource Sharing) controls which browser-based origins can call your API — configure allowed origins explicitly and never use the wildcard (*) with credentials. Beyond these specifics, operational security practices are essential: log all security-relevant events (authentication failures, authorization denials, rate limit hits) for audit and incident response. Monitor for suspicious patterns (credential stuffing, enumeration attacks). Keep all dependencies updated to patch known vulnerabilities. Use API versioning so you can deprecate insecure endpoints. And implement proper error handling that returns useful messages to legitimate developers without leaking internal details (stack traces, database schemas, file paths) to attackers.`,
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
					Content: `**1. Why Rate Limiting Is More Than Just Abuse Prevention**

Rate limiting is a critical mechanism for protecting APIs, ensuring fair resource allocation, and maintaining system stability under load. At its core, rate limiting controls how many requests a client can make within a given time period. But it serves purposes far beyond blocking abusive bots. It prevents accidental denial-of-service from buggy client code (an infinite retry loop can be just as devastating as a deliberate attack). It enables tiered pricing models (free users get 100 requests/minute, paid users get 10,000). It protects downstream dependencies — your API might handle 10,000 requests/second, but if each request queries a database that can only handle 5,000, the rate limiter at the API layer protects the database from being overwhelmed. Think of rate limiting as a pressure valve in a plumbing system: it ensures that no single source of demand can create pressure that bursts the pipes.

**2. Token Bucket Algorithm — The Industry Favorite**

The Token Bucket algorithm is the most widely used rate limiting algorithm, employed by AWS, Stripe, and most major API providers. Imagine a bucket that holds a fixed number of tokens (say, 100). Tokens are added to the bucket at a steady rate (say, 10 per second). Each incoming request must consume one token from the bucket to proceed. If the bucket is empty, the request is rejected (or queued). The key insight is that the bucket can accumulate tokens up to its capacity, which naturally allows short bursts of traffic — if a client has been idle for 10 seconds, the bucket will be full at 100 tokens, allowing a burst of 100 requests. This burst tolerance makes Token Bucket ideal for real-world traffic patterns, which are inherently bursty. The algorithm is elegant because it is defined by just two parameters: the bucket capacity (maximum burst size) and the refill rate (sustained throughput).

**3. Leaky Bucket Algorithm — Smoothing Out Traffic**

The Leaky Bucket algorithm takes the opposite approach to bursts. Imagine a bucket with a small hole in the bottom: requests flow in from the top, and the bucket "leaks" (processes requests) at a fixed, constant rate from the bottom. If requests arrive faster than the leak rate, the bucket fills up; once full, additional requests overflow and are rejected. The critical difference from Token Bucket is that Leaky Bucket enforces a perfectly smooth output rate regardless of how bursty the input is. This makes it ideal for scenarios where downstream systems need a steady, predictable flow of requests — for example, a payment processing gateway that can handle exactly 50 transactions per second and would fail if hit with 200 at once. The trade-off is that it does not allow bursts at all, which can feel restrictive for legitimate users.

**4. Fixed Window vs. Sliding Window Counters**

The Fixed Window algorithm is the simplest: divide time into fixed windows (e.g., one-minute intervals), count requests in each window, and reject when the count exceeds the limit. It is trivial to implement with a simple counter and a reset timestamp. However, it has a well-known boundary problem: a client can send 100 requests at 11:59:59 and another 100 at 12:00:01, effectively getting 200 requests in a 2-second span even though the limit is 100 per minute. The Sliding Window algorithm solves this by considering a rolling time period. One common implementation is the sliding window log, which stores the timestamp of each request and counts how many fall within the trailing window. A more memory-efficient approach is the sliding window counter, which uses weighted averages of the current and previous fixed windows. For example, if you are 30 seconds into the current minute, the effective count is (previous window count * 0.5) + (current window count). This eliminates the boundary burst problem with minimal additional complexity.

**5. Distributed Rate Limiting — Consistency Across a Fleet**

When your API runs on dozens or hundreds of servers behind a load balancer, per-server rate limiting is insufficient — a client could send 100 requests/minute to each of 50 servers, effectively getting 5,000 requests/minute while each server thinks the limit is being respected. Distributed rate limiting solves this by using a shared state store, most commonly Redis. Redis is ideal because it supports atomic increment operations (INCR) and key expiration (EXPIRE), and its in-memory architecture provides sub-millisecond latency. For even stronger atomicity, you can use Redis Lua scripts to perform check-and-increment as a single atomic operation, eliminating race conditions entirely. For high-availability setups, the Redlock algorithm provides distributed locking across multiple Redis masters, though it introduces additional complexity and latency.

**6. Rate Limiting Strategies — Per User, Per IP, Per API Key**

The right rate limiting strategy depends on what you are trying to protect. **Per-User** limiting (keyed by authenticated user ID) is the most fair and granular — it prevents individual abuse while allowing different limits for different subscription tiers. **Per-IP** limiting is useful for unauthenticated endpoints (like login pages) but has a significant weakness: many users behind corporate NATs or VPNs share a single IP address, so aggressive per-IP limits can block legitimate users. **Per-API-Key** limiting is common for developer platforms, where each registered application gets an API key with a specific rate limit based on its pricing tier (free, pro, enterprise). In practice, most systems use a combination: per-IP limits on unauthenticated endpoints, per-user limits on authenticated endpoints, and per-API-key limits for partner integrations.

**7. Adaptive Rate Limiting — Responding to System Load**

Static rate limits work well under normal conditions, but what happens during traffic spikes or when downstream services degrade? Adaptive rate limiting dynamically adjusts limits based on current system health. When CPU utilization, memory pressure, or response latency exceeds thresholds, the rate limiter tightens limits to shed load and prevent cascading failures. When the system recovers, limits are relaxed back to normal. This is conceptually similar to TCP congestion control: back off when congestion is detected, and gradually increase throughput when the path is clear. Netflix's concurrency limiter library implements this pattern, using measured response latencies to calculate an optimal concurrency limit in real-time. Adaptive rate limiting is essential for building truly resilient systems that gracefully degrade rather than catastrophically fail under pressure.

**8. Implementation Considerations**

When building rate limiters, use efficient data structures — sorted sets for sliding window logs, simple counters for fixed windows. Be mindful of memory usage, especially for sliding window logs that store individual request timestamps. Handle race conditions carefully with atomic operations or Lua scripts. Provide clear, helpful error responses: a 429 status code with X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset headers so clients can implement intelligent backoff. Log rate limit violations for security monitoring and abuse detection, and alert on sudden spikes that might indicate an attack.`,
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
					Content: `**1. What Is Batch Processing and When Should You Use It?**

Batch processing is the practice of collecting data over a period of time and then processing it all at once as a single "batch," rather than handling each record individually in real-time. Think of it like doing laundry: you do not wash each piece of clothing the moment it gets dirty — you collect clothes in a hamper throughout the week and then run one big load. Batch processing follows the same principle, and it is remarkably efficient for large-scale data work. It is the backbone of data warehousing, business intelligence reporting, machine learning model training, and any workload where throughput matters more than latency.

The key characteristics that define batch processing are: data is accumulated and processed on a schedule (hourly, nightly, weekly), throughput is optimized over latency (processing a million records efficiently is more important than processing one record instantly), the results are not needed in real-time, and the compute resources can be provisioned temporarily and released when the job completes, making it cost-effective for massive datasets.

**2. Classic Use Cases for Batch Processing**

Batch processing appears everywhere in production systems. ETL (Extract, Transform, Load) pipelines pull data from operational databases nightly, transform and clean it, and load it into a data warehouse for analysts to query. Report generation jobs run at the end of each day or month, crunching millions of transactions into summary reports for finance teams. Data aggregation jobs compute metrics like daily active users, revenue by region, or average response times from raw event logs. Machine learning training pipelines process billions of training examples to produce updated models. Historical data analysis (backfills) reprocesses years of data when business logic changes. In all these cases, the data already exists — we are not reacting to events in real-time, we are processing accumulated data in bulk.

**3. Batch Processing Systems — From MapReduce to Spark**

The foundational batch processing paradigm is **MapReduce**, introduced by Google and popularized by Hadoop. MapReduce works in three phases: the Map phase reads input data and emits key-value pairs in parallel across many machines, the Shuffle phase groups all values for the same key together, and the Reduce phase aggregates the grouped values into final results. For example, to count words in a billion documents, the Map phase emits (word, 1) for each word, the Shuffle groups all counts for the same word, and the Reduce sums them. MapReduce is powerful but verbose and disk-heavy — every intermediate result is written to disk.

**Apache Spark** revolutionized batch processing by keeping intermediate data in memory (Resilient Distributed Datasets, or RDDs), achieving 10-100x speedups over MapReduce for iterative algorithms. Spark provides a rich API for SQL queries, machine learning (MLlib), graph processing (GraphX), and streaming. **Apache Airflow** is a workflow orchestration tool that does not process data itself but schedules and monitors the DAGs (Directed Acyclic Graphs) of batch jobs, handling dependencies, retries, and alerting. **AWS Batch** and **Google Cloud Dataflow** are managed services that handle infrastructure provisioning automatically, letting you focus on the processing logic.

**4. Batch Processing Patterns**

The **ETL pattern** is the most common: extract data from source systems (databases, APIs, files), transform it (clean, deduplicate, enrich, aggregate), and load it into a destination (data warehouse, data lake). The **MapReduce pattern** parallelizes computation by splitting it into independent map tasks and aggregating reduce tasks, making it naturally suited to distributed execution across a cluster. The **Lambda Architecture** combines batch and real-time processing: a batch layer processes the complete historical dataset for accuracy, a speed layer processes recent data for low latency, and a serving layer merges results from both to serve queries. While Lambda Architecture is powerful, it introduces the complexity of maintaining two codepaths (batch and streaming) that must produce consistent results — the **Kappa Architecture** simplifies this by using a single stream processing system for both real-time and historical reprocessing.

**5. Design Considerations for Reliable Batch Systems**

Building reliable batch pipelines requires careful attention to several principles. **Idempotency** means that running the same job twice produces the same result — this is critical because jobs will fail and need to be restarted, and you must not produce duplicate records. **Fault tolerance** means the system can recover from partial failures (a node crashing mid-job) without losing progress; Spark achieves this through RDD lineage, which allows recomputation of lost partitions. **Scalability** means the system can process 10x more data by adding 10x more machines, without rewriting the job logic. **Monitoring and observability** are essential — you need to track job progress, detect failures quickly, alert on-call engineers, and provide enough logging to diagnose problems. Set up dashboards showing job duration trends, success/failure rates, and data quality metrics. A batch job that silently produces incorrect results is worse than one that fails loudly.`,
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
					Content: `**1. What Is Stream Processing and How Does It Differ from Batch?**

Stream processing is the continuous, real-time processing of data records as they arrive, in contrast to batch processing which accumulates data and processes it periodically. If batch processing is like doing laundry once a week, stream processing is like a conveyor belt in a factory — each item is inspected, transformed, and routed the moment it appears, never stopping. The fundamental shift is from "process all the data we have collected" to "process each event the moment it happens."

This paradigm is essential for use cases where timeliness is critical. When a credit card transaction occurs, the fraud detection system must decide within milliseconds whether to approve or flag it — waiting for a nightly batch job is not an option. When a server's CPU spikes to 100%, the monitoring system must fire an alert immediately, not in tomorrow's daily report. Stream processing enables systems to react to the world as it happens, providing low-latency insights (milliseconds to seconds) from high-velocity data sources.

**2. Real-World Use Cases That Demand Streaming**

Real-time fraud detection is the poster child of stream processing: every credit card swipe generates an event that must be checked against patterns (unusual location, atypical amount, rapid successive transactions) and approved or flagged in under 100 milliseconds. Real-time monitoring and alerting systems process infrastructure metrics and application logs as they are generated, triggering alerts when error rates spike or latency exceeds thresholds. Real-time recommendation engines on platforms like Netflix and Spotify update suggestions based on what you are watching or listening to right now, not what you did yesterday. Event-driven architectures use stream processing to decouple microservices: when a user places an order, an event is published, and the inventory service, payment service, shipping service, and notification service all consume and process it independently and concurrently.

**3. Stream Processing Systems — Kafka, Flink, and Beyond**

**Apache Kafka** is both a distributed event streaming platform (a durable, high-throughput message bus) and a stream processing library (Kafka Streams). Kafka Streams is a lightweight Java library that processes data directly from Kafka topics, requiring no separate cluster infrastructure. **Apache Flink** is a full-featured distributed stream processing engine that excels at stateful computations, complex event processing, and exactly-once semantics. It can handle millions of events per second with millisecond latency and provides sophisticated windowing operations. **Apache Storm** was one of the earliest distributed real-time computation systems, now largely superseded by Flink. **AWS Kinesis** and **Google Cloud Dataflow** are managed cloud services that abstract away cluster management, autoscale based on throughput, and integrate tightly with their respective cloud ecosystems. Dataflow is particularly notable because it implements the Apache Beam programming model, which provides a unified API for both batch and stream processing.

**4. Key Streaming Patterns — Event Sourcing, CQRS, and Windowing**

**Event Sourcing** is a pattern where instead of storing the current state of an entity (e.g., account balance = $500), you store every event that led to that state (deposit $1000, withdraw $300, deposit $100, withdraw $300). The current state is derived by replaying all events. This provides a complete audit trail, enables temporal queries ("what was the balance on March 15th?"), and allows you to rebuild read models or fix bugs by replaying events with corrected logic. **CQRS (Command Query Responsibility Segregation)** separates the write model (optimized for recording commands/events) from the read model (optimized for serving queries). The event stream serves as the source of truth, and one or more read models are materialized from it. For example, an e-commerce system might write order events to Kafka, and have separate read models for the customer-facing order status page, the warehouse fulfillment dashboard, and the finance analytics system.

**Windowing** is how stream processors group unbounded data into finite chunks for aggregation. A **tumbling window** divides time into fixed, non-overlapping intervals (e.g., count events every 5 minutes) — simple but events near the boundary can be split. A **sliding window** creates overlapping intervals (e.g., every minute, count the last 5 minutes of events) — provides smoother results but processes each event multiple times. A **session window** groups events by activity — a session starts with the first event and extends as long as events keep arriving within a gap threshold (e.g., 30 minutes of inactivity closes the session), making it ideal for user behavior analysis.

**5. Stateful Processing and Exactly-Once Semantics**

Many stream processing tasks require maintaining state across events — for example, counting unique visitors, computing running averages, or joining two streams. **Stateful processing** stores intermediate results (like counters or aggregations) that are updated as each event arrives. The challenge is fault tolerance: if a processor crashes, its in-memory state is lost. Flink and Kafka Streams solve this with **checkpointing** — periodically saving the processor's state to durable storage (like S3 or HDFS) so that on failure, the processor can resume from the last checkpoint without data loss. This enables **exactly-once processing semantics**, meaning each event affects the output exactly once, even in the presence of failures. Without exactly-once semantics, financial systems might double-count transactions or monitoring systems might fire duplicate alerts.

**6. Design Considerations for Production Streaming Systems**

**Backpressure handling** is critical: what happens when events arrive faster than the processor can handle them? Without backpressure, queues grow unbounded, memory is exhausted, and the system crashes. Well-designed systems signal upstream to slow down (TCP-style backpressure) or spill to disk. **Scaling** in stream processing is achieved through partitioning: Kafka topics are divided into partitions, and each partition is processed by a single consumer instance, allowing horizontal scaling by adding more consumers. **Late-arriving data** is an inherent challenge — events may arrive out of order due to network delays, and the system must decide how long to wait before finalizing windowed aggregations (this is configurable via watermarks in Flink and Dataflow). Finally, choose your processing guarantee carefully: at-least-once is simpler and sufficient for most analytics workloads, while exactly-once is essential for financial transactions and other cases where duplicates are unacceptable.`,
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
					Content: `**1. What Are ETL Pipelines and Why Are They the Backbone of Data Infrastructure?**

ETL stands for Extract, Transform, Load — a three-stage process that moves data from operational source systems into analytical destination systems. Virtually every company that makes data-driven decisions relies on ETL pipelines, even if they do not call them that. The fundamental problem ETL solves is this: your production databases are optimized for serving application requests (fast reads and writes for individual records), but they are terrible for analytical queries ("what was our revenue by region for the last 12 months?"). ETL pipelines bridge this gap by extracting data from production systems, transforming it into an analysis-friendly format, and loading it into a data warehouse or data lake where analysts and data scientists can query it without impacting production performance. Think of ETL as a nightly freight train that moves goods from factories (source systems) to distribution centers (warehouses) where they can be efficiently sorted and delivered to consumers (analysts).

**2. Extract — Getting Data Out of Source Systems**

The Extract phase reads data from one or more source systems, which can include relational databases (PostgreSQL, MySQL), NoSQL databases (MongoDB, DynamoDB), REST APIs, flat files (CSV, JSON), message queues (Kafka), or SaaS applications (Salesforce, Stripe). The key architectural decision is whether to perform a **full extraction** (read all data every time) or an **incremental extraction** (read only what has changed since the last run). Full extraction is simple but wasteful for large datasets — if you have 100 million rows and only 10,000 changed today, reading all 100 million is unnecessarily expensive. Incremental extraction typically uses an "updated_at" timestamp column to identify changed rows, or more sophisticated **Change Data Capture (CDC)** techniques that read the database's transaction log (binlog in MySQL, WAL in PostgreSQL) to capture inserts, updates, and deletes as they happen. CDC provides near-real-time data freshness and captures deletions that timestamp-based approaches miss. Always extract from read replicas rather than the primary database to avoid adding load to your production system.

**3. Transform — Cleaning, Enriching, and Shaping the Data**

The Transform phase is where the real intelligence of the pipeline lives. Raw data from source systems is messy: it contains duplicates, null values, inconsistent formats, and missing relationships. Transformation includes **cleaning** (removing duplicate records, handling null values with defaults or filters, standardizing formats like dates and phone numbers), **validation** (checking that values fall within expected ranges, that required fields are present, that referential integrity holds), **enrichment** (joining with reference data — for example, adding country names to country codes, or adding product categories to product IDs), **aggregation** (summarizing detailed records into useful metrics — total sales by day, average order value by customer segment), and **format conversion** (converting between data formats, renaming columns to match the warehouse schema, casting data types). A modern trend is ELT (Extract, Load, Transform), where raw data is loaded first into a cloud data warehouse like Snowflake or BigQuery, and transformations are performed using SQL inside the warehouse using tools like dbt. ELT leverages the warehouse's massive compute power and avoids building custom transformation code.

**4. Load — Writing Data to the Destination**

The Load phase writes the transformed data into the destination system — typically a data warehouse (Snowflake, BigQuery, Redshift), a data lake (S3 + Parquet/Delta Lake), or an operational database. The loading strategy matters for both performance and correctness. **Full load** (truncate and reload) is simple but causes a window of unavailability. **Incremental load** (upsert — insert new rows, update existing ones) is more efficient but requires a reliable way to identify which rows are new versus updated. **Append-only** loading writes all extracted data as new rows with timestamps, preserving full history (common in data lakes and event stores). Schema management is a critical concern: what happens when a source system adds a new column, renames a field, or changes a data type? Robust pipelines handle schema evolution gracefully, either through schema-on-read approaches (data lakes) or automated migration scripts.

**5. ETL Tools and the Modern Data Stack**

**Apache Airflow** is the most popular open-source workflow orchestrator: you define your pipeline as a DAG (Directed Acyclic Graph) of Python tasks, and Airflow handles scheduling, dependency management, retries, and alerting. It does not process data itself — it orchestrates calls to other systems. **Apache NiFi** provides a visual, drag-and-drop interface for building data flows and excels at handling diverse data formats and routing logic. **AWS Glue** is a fully managed ETL service that auto-discovers schemas, generates ETL code, and runs it on a serverless Spark infrastructure. **dbt (data build tool)** has become the centerpiece of the modern ELT stack: it lets analytics engineers write transformations as SQL SELECT statements, with built-in testing, documentation, and version control. **Fivetran** and **Airbyte** focus on the Extract and Load phases, providing pre-built connectors to hundreds of data sources that sync data automatically into your warehouse, leaving the Transform phase to dbt.

**6. Best Practices for Production ETL Pipelines**

**Idempotency** is the most important property of a reliable pipeline: running the same job twice with the same input must produce the same result. This means using upserts instead of inserts, using deterministic processing logic, and designing loads so that reprocessing does not create duplicates. **Error handling** should include automatic retries with exponential backoff for transient failures (network timeouts, API rate limits), dead-letter queues for records that consistently fail transformation, and clear alerting so the data team knows when a pipeline breaks. **Data quality checks** should be built into every pipeline: validate row counts (did we extract roughly as many rows as expected?), check for null values in critical columns, verify that aggregations sum correctly, and compare today's results to yesterday's to catch anomalies. **Monitoring** should track pipeline duration trends (is it getting slower?), success/failure rates, data freshness (when was the warehouse last updated?), and data volume trends. Finally, treat your pipeline code like production application code: use version control, code review, automated testing, and CI/CD deployment.`,
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
			ID:          2412,
			Title:       "Search Systems",
			Description: "Learn how to design and implement search systems using Elasticsearch and other search technologies.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "Elasticsearch Fundamentals",
					Content: `**1. What Is Elasticsearch and Why Has It Become the Default Search Engine?**

Elasticsearch is a distributed, RESTful search and analytics engine built on top of Apache Lucene — the most powerful open-source full-text search library in existence. While Lucene provides the low-level indexing and search algorithms, Elasticsearch wraps it in a distributed system that handles clustering, replication, sharding, and a simple JSON-over-HTTP API. This combination has made Elasticsearch the dominant search engine for applications ranging from e-commerce product search (used by eBay, Walmart, and Etsy) to log analysis (the "E" in the ELK stack — Elasticsearch, Logstash, Kibana) to real-time analytics dashboards. Think of Elasticsearch as what you would get if you combined a full-text search engine with a distributed database and an analytics platform — it excels at finding needles in haystacks of unstructured text, and it does so in near real-time across petabytes of data.

**2. Core Concepts — Indices, Documents, and Mappings**

Elasticsearch organizes data using three fundamental concepts. An **Index** is a collection of related documents, analogous to a database table (though far more flexible). You might have separate indices for "products," "users," and "application-logs." Each **Document** is a JSON object stored within an index — it is the basic unit of information, analogous to a row in a relational database. A product document might contain fields like name, description, price, category, and rating. A **Mapping** defines the schema for an index: which fields exist, what data type each field has (text, keyword, integer, float, date, geo_point), and how each field should be indexed and analyzed. Mappings can be explicit (you define them upfront) or dynamic (Elasticsearch infers types from the first document it sees). Explicit mappings are strongly recommended for production use because they give you control over how text is analyzed and searched.

**3. Sharding — How Elasticsearch Scales Horizontally**

A single Lucene index has physical limits on how much data it can hold and how fast it can search. Elasticsearch overcomes this through **sharding**: each index is divided into one or more shards, and each shard is a self-contained Lucene index that can be hosted on any node in the cluster. When you create an index with 5 primary shards and your cluster has 5 data nodes, each node holds one shard, and search queries execute in parallel across all 5 shards simultaneously — this is how Elasticsearch achieves near-linear horizontal scaling. The number of primary shards is set at index creation time and cannot be changed later (you would need to reindex), so choosing the right shard count is an important capacity planning decision. A common rule of thumb is to keep each shard between 10-50 GB for optimal performance.

**4. Replication — High Availability and Read Scaling**

Each primary shard can have one or more **replica shards** — exact copies maintained on different nodes. Replicas serve two purposes: high availability (if a node holding a primary shard fails, a replica is promoted to primary automatically, with no downtime) and read scaling (search queries can be served by either the primary or any replica, distributing the read load). In a production cluster, you typically configure at least one replica per shard, meaning your data is stored twice. For read-heavy workloads, increasing the replica count further can dramatically improve search throughput, at the cost of additional storage and indexing overhead (every document must be indexed on the primary and all replicas).

**5. Cluster Architecture — Nodes and Their Roles**

An Elasticsearch cluster is a collection of nodes (server instances) working together. Different nodes play different roles. **Master-eligible nodes** manage cluster state: they track which nodes are in the cluster, which shards are on which nodes, and coordinate shard allocation. You should dedicate 3 master-eligible nodes for fault tolerance (allowing one to fail while maintaining quorum). **Data nodes** store the actual shard data and execute search and indexing operations — they need fast disks and plenty of RAM. **Coordinating nodes** (also called client nodes) receive search requests from clients, scatter them to the relevant data nodes, gather the results, and return them to the client — they act as smart load balancers. In smaller clusters, a single node can play all roles, but production clusters separate these roles for stability and performance.

**6. Search Capabilities and Real-World Use Cases**

Elasticsearch's search capabilities go far beyond simple text matching. It supports full-text search with relevance scoring (finding the best matches, not just any match), fuzzy matching (handling typos — "laptp" still finds "laptop"), phrase matching, wildcard queries, faceted search (showing filter counts like "Electronics: 234, Clothing: 89"), geospatial queries (finding stores within 10 miles), and powerful aggregations that let you compute statistics, histograms, and nested breakdowns over your data in real-time. These capabilities power product search on e-commerce sites (with faceted filtering by brand, price range, and rating), log analysis platforms (searching through billions of log lines with sub-second latency), application performance monitoring (aggregating request latencies by endpoint), and autocomplete/typeahead features (suggesting results as the user types). The combination of speed, flexibility, and scalability is what makes Elasticsearch the go-to choice whenever an application needs to search through large amounts of text or structured data.`,
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
					Content: `**1. What Is Full-Text Search and Why Can't You Just Use SQL LIKE?**

Full-text search is the ability to search through large volumes of natural language text and find the most relevant documents, ranked by how well they match the query. You might wonder why this requires specialized technology — after all, SQL has the LIKE operator. The answer becomes clear at scale and sophistication. A query like "SELECT * FROM products WHERE name LIKE '%laptop%'" performs a full table scan on every row, cannot handle typos or synonyms (searching for "notebook" would not find "laptop"), returns results in no particular order of relevance, and becomes painfully slow on millions of rows. Full-text search engines solve all of these problems by building an **inverted index** — a data structure that maps every unique word to the list of documents containing it, much like the index in the back of a textbook. When you search for "laptop," the engine looks up "laptop" in the inverted index and instantly retrieves all matching documents, regardless of table size. This lookup is O(1) rather than O(n), making it millions of times faster than LIKE for large datasets.

**2. Query Types — From Simple Matches to Fuzzy Typo Correction**

Full-text search engines provide a rich vocabulary of query types for different needs. The **Match Query** is the workhorse: it analyzes the query text using the same analyzer applied at index time, then finds documents containing the resulting terms. This means searching for "running shoes" also finds documents containing "run" and "shoe" because the analyzer stems both the query and the indexed text. The **Term Query** performs an exact, unanalyzed match — it is used for structured fields like status codes, user IDs, or enum values where you want precise matching, not linguistic analysis. The **Phrase Query** requires words to appear in the exact order specified, with no intervening words — useful for searching for specific phrases like "machine learning" where the individual words have different meanings. The **Fuzzy Query** handles typos by finding terms within a specified edit distance (Levenshtein distance) — searching for "laptp" with fuzziness of 1 finds "laptop" because they differ by one character. The **Wildcard Query** supports pattern matching with * (any characters) and ? (single character), though it should be used sparingly because it can be slow on large indices.

**3. Text Analysis — The Secret Sauce of Search Quality**

The quality of full-text search depends entirely on how text is analyzed at index time and query time. An **Analyzer** is a pipeline of three components: a **Character Filter** (optional, performs character-level transformations like stripping HTML tags), a **Tokenizer** (splits text into individual terms — the standard tokenizer splits on whitespace and punctuation), and **Token Filters** (transform individual tokens). The most important token filters are: **Lowercase** (so that "Laptop" and "laptop" match), **Stop Words** removal (eliminating common words like "the," "is," "and" that add noise), **Stemming** (reducing words to their root form — "running," "runs," "ran" all become "run"), and **Synonym expansion** (so that "laptop" also matches "notebook" or "portable computer").

Choosing the right analyzer is critical and depends on your domain. An e-commerce product search might use aggressive stemming and synonym expansion to maximize recall (finding all relevant products). A legal document search might use minimal analysis to preserve exact terminology. A multilingual application needs language-specific analyzers that understand the morphology of each language (German compound words, Japanese without spaces, Arabic right-to-left text). You can even create custom analyzers that chain specific tokenizers and filters for your exact use case.

**4. Relevance Scoring — Why Order Matters More Than Matching**

Finding documents that match a query is only half the battle — ranking them by relevance is what makes search useful. If a user searches for "laptop" and gets 50,000 results, the difference between a good and bad search experience is whether the most relevant results appear on the first page. The classic relevance scoring formula is **TF-IDF** (Term Frequency times Inverse Document Frequency): a term is more relevant to a document if it appears frequently in that document (TF) and is rare across all documents (IDF). A document mentioning "laptop" 10 times is more relevant than one mentioning it once, and the word "laptop" is more discriminating than the word "the" because it is rarer.

Modern search engines use **BM25**, an improved version of TF-IDF that adds two important refinements: it includes a saturation function (mentioning "laptop" 100 times is not 10x more relevant than mentioning it 10 times — there are diminishing returns) and a document length normalization (a short product title matching the query is more relevant than a long blog post where the term appears incidentally). BM25 is the default scoring algorithm in Elasticsearch and works well out of the box for most use cases. For more sophisticated ranking, you can use **field boosting** (matches in the title are worth 3x more than matches in the description), **function scoring** (incorporate popularity, recency, or business rules into the score), or **Learning to Rank** (train a machine learning model on click-through data to predict the optimal ranking).

**5. Best Practices for Production Search Systems**

Design your search experience thoughtfully. Use explicit mappings with appropriate analyzers for each field type — do not rely on dynamic mapping in production. Index the fields that users actually search on, and use "keyword" type (not "text") for fields used for filtering and aggregation. Implement autocomplete using edge n-gram tokenizers or completion suggesters for a responsive typeahead experience. Build a synonym dictionary specific to your domain and update it based on search analytics (what are users searching for that returns no results?). Monitor search performance metrics: query latency (p50, p95, p99), zero-result rate, click-through rate on search results, and position of clicked results. A high zero-result rate means users are searching for things your index does not cover; a low click-through rate on top results means your ranking needs improvement.`,
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
					Content: `**1. Why Ranking Is the Heart of Search — And the Hardest Part to Get Right**

Ranking algorithms determine the order in which search results are presented to the user, and this order is arguably more important than the matching itself. Consider this: a search for "laptop" on an e-commerce site might return 50,000 matching products. The user will look at the first 10-20 results and then either find what they want or give up. If the best results are buried on page 47, the search is effectively broken, even though it "found" the right products. Google's entire multi-trillion-dollar business is built on ranking web pages better than anyone else. Amazon reportedly generates 35% of its revenue from its recommendation and search ranking algorithms. Getting ranking right is the difference between a search experience that feels magical and one that feels useless.

**2. Text Relevance — The Foundation of Ranking**

The most fundamental ranking signal is text relevance: how well does the document's text match the query? The classic approach is **TF-IDF** (Term Frequency times Inverse Document Frequency). Term Frequency measures how often the search term appears in a document — a product description mentioning "laptop" five times is likely more about laptops than one mentioning it once. Inverse Document Frequency measures how rare the term is across all documents — the word "the" appears everywhere and is not discriminating, while "laptop" is more specific and therefore more valuable. Modern search engines use **BM25**, which refines TF-IDF with two key improvements. First, it adds a saturation curve for term frequency: the 1st mention of "laptop" boosts relevance a lot, the 5th mention adds a little more, but the 50th mention adds almost nothing (diminishing returns). Second, it normalizes for document length: a short product title that matches the query is typically more relevant than a 10,000-word blog post where the term appears incidentally.

Field length and field importance also matter. In most search applications, a match in the title is far more relevant than a match in the body text. This is implemented through **field boosting**: you might weight title matches 3x, brand matches 2x, and description matches 1x. Getting these weights right requires experimentation and is often informed by analyzing click-through data.

**3. Beyond Text — Popularity, Freshness, and Behavioral Signals**

Text relevance alone is insufficient for a great search experience. Two products might have equally relevant titles, but one has 4.8 stars from 10,000 reviews while the other has 2.1 stars from 3 reviews — the ranking should reflect this. **Popularity signals** include click-through rate (CTR) — what percentage of users who see this result in the search listing actually click on it, conversion rate — what percentage of clicks lead to a purchase or desired action, user ratings and review count, and sales volume or view count. These signals encode the collective wisdom of millions of users about which results are actually useful.

**Freshness** is critical for time-sensitive content: news articles, job listings, real estate listings, and social media posts should prioritize recent content. This is implemented through decay functions that reduce the relevance score over time — a news article published today gets a full freshness boost, one from last week gets a partial boost, and one from last year gets none. The challenge is balancing freshness against quality: a mediocre article published today should not outrank a seminal article published last year for an evergreen query.

**4. Personalization — Tailoring Results to the Individual**

Personalization uses information about the individual user to adjust ranking. A search for "python" should return programming results for a software engineer and pet care results for a reptile enthusiast. Personalization signals include the user's past search and browsing history, purchase history, stated preferences, geographic location (local restaurants rank higher for "pizza"), device type (mobile users might prefer different content formats), and language preferences. Personalization dramatically improves user satisfaction but introduces complexity and privacy concerns. It also creates "filter bubbles" where users only see content that reinforces their existing preferences. Most systems implement personalization as a re-ranking layer on top of the base relevance score, with configurable weight so it can be tuned or disabled.

**5. Business Rules and Sponsored Results**

In commercial search systems, ranking is not purely about relevance — business objectives also play a role. Sponsored or promoted results are explicitly boosted in the ranking (clearly labeled to maintain user trust). Featured content or curated collections may receive a ranking boost during seasonal events or campaigns. Suppression rules may demote out-of-stock products, age-restricted content, or items with quality issues. These business rules are typically implemented as score modifiers applied on top of the relevance score, and the challenge is ensuring they improve the business without degrading the user's search experience.

**6. Static, Dynamic, and Hybrid Ranking Strategies**

**Static ranking** pre-computes a quality or popularity score for each document offline (e.g., using PageRank for web pages or average rating for products) and uses it as a baseline during search. It is fast because no computation happens at query time, but it cannot adapt to the query context. **Dynamic ranking** computes scores at query time based on the specific query, user, and context — it is more accurate but slower because it requires real-time computation. **Hybrid ranking** combines both: static scores provide a quality baseline, and dynamic scoring adjusts for query relevance and personalization. Most production search systems use a multi-stage ranking pipeline: a fast first stage retrieves candidates using an inverted index (BM25), a second stage re-ranks using more expensive features (popularity, personalization), and an optional third stage applies business rules and diversity filters.

**7. Learning to Rank — The ML-Powered Approach**

**Learning to Rank (LTR)** replaces hand-tuned scoring formulas with a machine learning model trained on user interaction data. The model takes features (text relevance scores, popularity metrics, freshness, personalization signals) and predicts the optimal ranking. Training data comes from implicit feedback: clicks, purchases, dwell time, and bounce rates. LTR approaches include pointwise (predict relevance score for each document independently), pairwise (predict which of two documents should rank higher), and listwise (optimize the entire ranking list). Companies like Google, Amazon, LinkedIn, and Airbnb use LTR extensively because it can discover complex feature interactions that human engineers would miss, and it continuously improves as more interaction data is collected. The trade-off is interpretability: it becomes harder to explain why a specific result ranked where it did, which can be problematic for debugging and for applications with regulatory requirements.

**8. Optimizing Ranking Performance**

Ranking at scale requires careful performance optimization. Cache the results of popular queries (the top 1% of queries often account for 30%+ of traffic). Pre-compute static scores offline so they are ready at query time. Use filters (exact match on category, price range, availability) before scoring to reduce the number of documents that need expensive relevance computation. Monitor ranking quality continuously using metrics like NDCG (Normalized Discounted Cumulative Gain), MRR (Mean Reciprocal Rank), and click-through rate. Run A/B tests when changing ranking algorithms to measure the impact on user behavior before rolling out changes broadly.`,
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
			ID:          2413,
			Title:       "Distributed Systems Patterns",
			Description: "Master distributed systems patterns including leader election, distributed locks, and consensus algorithms.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "Leader Election",
					Content: `**1. Why Does a Distributed System Need a Leader?**

In a distributed system with multiple nodes, many operations require coordination: someone needs to decide which node processes a particular request, someone needs to manage the order of writes to ensure consistency, and someone needs to detect failures and initiate recovery. Without a designated leader, every node would need to coordinate with every other node for every decision, leading to quadratic communication overhead and the constant risk of conflicting actions. Leader election solves this by ensuring exactly one node acts as the "leader" (or "primary" or "master") at any given time, while the remaining nodes are "followers" (or "replicas" or "secondaries") that defer to the leader's decisions.

Think of it like a meeting: if everyone talks simultaneously, nothing gets decided. You need a facilitator who manages the agenda and ensures decisions are made. In distributed systems, the leader is that facilitator. The critical challenge is: what happens when the facilitator leaves the room (crashes)? The remaining participants must quickly and unambiguously elect a new facilitator without ending up with two people claiming to lead (the dreaded "split-brain" scenario, where two nodes both believe they are the leader and issue conflicting commands).

**2. The Bully Algorithm — Simple but Chatty**

The Bully Algorithm is one of the oldest and simplest leader election algorithms. Every node has a unique numeric ID, and the rule is straightforward: the node with the highest ID becomes the leader. When a node detects that the current leader has failed (typically through a missed heartbeat), it initiates an election by sending an "election" message to all nodes with higher IDs. If any higher-ID node responds (saying "I'm alive, back off"), the initiating node steps aside and waits for the higher-ID node to complete the election. If no higher-ID node responds, the initiating node declares itself leader and broadcasts a "victory" message to all other nodes. The algorithm is called "bully" because the highest-ID node always wins, effectively bullying lower-ID nodes into submission.

The Bully Algorithm is easy to understand and implement, but it has significant drawbacks: it generates a lot of network traffic (O(n^2) messages in the worst case), it can cause cascading elections if multiple nodes detect the failure simultaneously, and it always elects the highest-ID node even if that node has the worst network connectivity or the least capacity. It is mainly of academic interest today, having been replaced by more sophisticated approaches in production systems.

**3. ZooKeeper and etcd — The Practical Production Approach**

In practice, most distributed systems delegate leader election to a dedicated coordination service like **Apache ZooKeeper** or **etcd**. These services provide primitives that make leader election straightforward and reliable. In ZooKeeper, leader election works by having all candidate nodes create ephemeral sequential znodes (special nodes that are automatically deleted when the creating client disconnects) under a designated path, such as /election/. Each node gets a sequentially numbered znode (lock-0000000001, lock-0000000002, etc.). The node with the lowest sequence number becomes the leader. All other nodes set a watch on the znode immediately preceding their own. When the leader crashes, its ephemeral znode is automatically deleted, the next node in sequence receives a notification, and it becomes the new leader. This cascade is efficient (each node watches only one other node, not all of them) and handles failures gracefully.

etcd provides similar functionality through its lease and election APIs. A node acquires a lease (a time-limited lock) and campaigns for leadership. The node that wins the campaign becomes leader and must periodically renew its lease. If it fails to renew (because it crashed or lost network connectivity), the lease expires and other nodes can campaign for the now-vacant leadership position. Both ZooKeeper and etcd are built on consensus algorithms internally (ZAB and Raft, respectively), which guarantees that even in the face of network partitions, at most one leader exists at any time.

**4. Raft — Consensus with Built-In Leader Election**

The Raft consensus algorithm (used in etcd, Consul, and CockroachDB) has leader election built into its core design. Every node starts as a follower. Followers expect regular heartbeat messages from the leader. If a follower does not receive a heartbeat within a randomized timeout period (typically 150-300ms), it assumes the leader has failed and transitions to the "candidate" state. The candidate increments its term number (a logical clock) and sends RequestVote RPCs to all other nodes. Each node votes for at most one candidate per term, and a candidate that receives votes from a majority of nodes becomes the new leader. The randomized timeout is crucial: it ensures that in most cases, only one node times out and starts an election, avoiding vote-splitting. If two candidates start simultaneously and neither wins a majority, both back off with new random timeouts and try again.

Raft's elegance is that leader election, log replication, and safety are all tightly integrated. The leader handles all client writes, appends them to its log, and replicates the log to followers. This simplifies reasoning about consistency because all decisions flow through a single node. The trade-off is that the leader can become a bottleneck for write-heavy workloads, and leader transitions (though typically fast — a few hundred milliseconds) cause brief periods of unavailability.

**5. Failure Handling and the Split-Brain Problem**

The most dangerous failure mode in leader election is **split-brain**: a network partition divides the cluster into two groups, and each group elects its own leader. Now two leaders are issuing conflicting commands, and data consistency is violated. Consensus-based systems like Raft prevent split-brain by requiring a majority (quorum) to elect a leader. In a 5-node cluster, a leader needs at least 3 votes. If the network splits into groups of 2 and 3, only the group of 3 can elect a leader — the group of 2 cannot form a quorum and becomes read-only (or unavailable) until the partition heals. This is why clusters always have an odd number of nodes: 3 nodes tolerate 1 failure, 5 tolerate 2, and 7 tolerate 3.

Graceful leadership transfer is another important consideration. When you need to take the leader node offline for maintenance, a graceful transfer (the leader tells a follower to take over before shutting down) avoids the delay and uncertainty of a failure-triggered election. Production systems also implement fencing tokens — monotonically increasing tokens issued to each new leader — to prevent a "zombie leader" (a node that was partitioned away, thinks it is still leader, and tries to issue commands after recovering) from causing damage.

**6. Real-World Use Cases**

Leader election is ubiquitous in distributed systems. Database primary selection (PostgreSQL streaming replication, MongoDB replica sets, MySQL Group Replication) uses leader election to designate which node accepts writes. Distributed task schedulers (like Airflow or Kubernetes controllers) use a leader to assign work to workers without duplication. Configuration management systems use a leader to serialize configuration changes and distribute them consistently. Distributed locking services (covered in the next lesson) often build on leader election to provide mutual exclusion primitives. Kafka uses leader election to designate a leader for each partition, ensuring that producers write to a single broker for ordering guarantees.`,
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
					Content: `**1. Why Are Distributed Locks So Much Harder Than Local Locks?**

In a single-process application, mutual exclusion is straightforward: a mutex or semaphore in memory ensures that only one thread accesses a shared resource at a time. But in a distributed system where multiple processes on multiple machines need to coordinate access to a shared resource (a database record, a file, an external API with a rate limit, a scheduled job that must run on only one node), you cannot use an in-memory lock because processes do not share memory. You need a distributed lock — a mechanism that provides mutual exclusion across network boundaries.

The challenge is that distributed locks must work correctly despite the fundamental unreliability of networks. The process holding the lock might crash without releasing it (deadlock). The network between the lock holder and the lock service might partition, making it impossible to determine whether the holder is still alive. The lock service itself might fail. Clocks on different machines might disagree, making timeout-based reasoning unreliable. These challenges make distributed locking one of the most subtle problems in distributed systems, and getting it wrong can lead to data corruption, duplicate processing, or system deadlocks. As Martin Kleppmann (author of "Designing Data-Intensive Applications") has noted, distributed locks are often used when they shouldn't be — and when they are needed, they are often implemented incorrectly.

**2. Essential Properties of a Correct Distributed Lock**

A distributed lock must provide several properties to be useful and safe. **Mutual Exclusion** (safety) is the fundamental requirement: at most one client can hold the lock at any given time. If two clients both believe they hold the lock simultaneously, the lock is broken and data corruption is likely. **Deadlock Freedom** (liveness) means that even if the client holding the lock crashes or becomes unreachable, the lock will eventually be released so other clients can acquire it — this is typically achieved through automatic expiration (TTL). **Fault Tolerance** means the lock service continues to function despite the failure of individual nodes. **Performance** matters in practice: acquiring and releasing a lock should take milliseconds, not seconds, and the system should handle thousands of lock operations per second without becoming a bottleneck.

**3. Redis-Based Distributed Locks — Simple but Subtle**

The simplest distributed lock uses a single Redis instance and the SET command with NX (only set if Not eXists) and EX (EXpire after N seconds): "SET lock:resource123 owner-id NX EX 30". If the SET succeeds, the client holds the lock for 30 seconds. If it fails (because another client already set the key), the client does not hold the lock and must retry or back off. To release the lock, the client must verify it still owns the lock (by checking the value matches its owner-id) and then delete the key — critically, this check-and-delete must be atomic, which is achieved using a Lua script executed on the Redis server.

The single-Redis approach is simple and fast but has a fundamental weakness: if the Redis instance crashes, the lock is lost, and two clients might simultaneously believe they hold it. To address this, Salvatore Sanfilippo (Redis creator) proposed the **Redlock** algorithm: the client attempts to acquire the lock on a majority (e.g., 3 out of 5) of independent Redis instances. If it succeeds on the majority within a time threshold, the lock is considered held. This provides tolerance for individual Redis instance failures. However, Redlock has been controversial: Kleppmann argued that it relies on timing assumptions that can be violated in real systems (GC pauses, clock skew), and recommended using consensus-based systems like ZooKeeper for safety-critical locks.

**4. ZooKeeper and etcd — Consensus-Backed Locks**

ZooKeeper provides a more reliable (though higher-latency) distributed lock through its ephemeral sequential znode mechanism. To acquire a lock, a client creates an ephemeral sequential znode under a path like /locks/resource123/ — the znode gets a name like lock-0000000042. The client then checks if its znode has the lowest sequence number. If yes, it holds the lock. If not, it sets a watch on the znode with the next-lower sequence number and waits for a notification. When the lock holder finishes and deletes its znode (or crashes, causing the ephemeral znode to be automatically deleted), the next client in sequence is notified and acquires the lock. This creates a fair, ordered queue of lock waiters — no starvation, no thundering herd (only one waiter is notified at a time).

etcd provides similar lock semantics through its lease API and concurrency primitives. A client creates a lease (a TTL-based session) and creates a key associated with that lease. If the client crashes, the lease expires and the key is deleted, releasing the lock. etcd's advantage over ZooKeeper is its simpler operational model (a single binary, no separate ensemble management) and its use of the well-understood Raft consensus protocol. Both ZooKeeper and etcd provide stronger safety guarantees than Redis because they use consensus internally — as long as a majority of nodes are healthy, the lock state is consistent.

**5. Lock Patterns — Timeout, Refresh, and Fencing**

The **Simple Lock** pattern (acquire, use resource, release) works but is fragile: if the holder crashes between acquisition and release, the lock is held forever (deadlock). The **Lock with Timeout** pattern (automatic expiration after N seconds) prevents deadlocks but introduces a new risk: if the holder's operation takes longer than the timeout, the lock expires while the holder is still working, and another client acquires it — now two clients are operating on the resource simultaneously. The **Lock with Refresh** pattern mitigates this by having the holder periodically extend the lock's TTL while it is still working. This is implemented as a background goroutine or thread that sends a "refresh" command every TTL/3 seconds. If the holder crashes, the refresh stops, and the lock expires naturally.

The most robust pattern uses **fencing tokens**: each time a lock is acquired, the lock service issues a monotonically increasing token number. The holder includes this token in all requests to the protected resource, and the resource rejects any request with a token lower than the highest it has seen. This prevents a "zombie" holder (one whose lock expired but which is still running due to a GC pause or network delay) from corrupting data — its stale fencing token will be rejected. Fencing tokens provide end-to-end safety even when the lock mechanism itself has timing-related edge cases.

**6. Best Practices and When Not to Use Distributed Locks**

Always set a timeout on every lock to prevent deadlocks. Implement retry logic with exponential backoff and jitter to avoid thundering herd problems when the lock is released. Handle lock acquisition failures gracefully — your application should have a clear strategy for what to do when it cannot acquire the lock (queue the work, return an error, try a different resource). Monitor lock contention metrics: high contention suggests the locked section is too long or the architecture should be redesigned to reduce shared state. Make operations within the lock idempotent whenever possible — if a lock expires prematurely and another client takes over, idempotent operations ensure that duplicate processing does not cause data corruption.

Most importantly, consider whether you actually need a distributed lock. Many use cases can be solved more simply with optimistic concurrency control (version numbers or ETags), database-level constraints (unique indexes, serializable transactions), or idempotent operation design. Distributed locks add complexity, latency, and potential failure modes. Use them when you truly need mutual exclusion across distributed processes, and prefer simpler alternatives when they suffice.`,
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
					Content: `**1. The Fundamental Problem — Getting Distributed Nodes to Agree**

Consensus is the foundational problem in distributed computing: how do you get multiple nodes (which communicate only by sending messages over an unreliable network) to agree on a single value or decision? This sounds simple until you consider the failure modes: nodes can crash at any moment, network messages can be lost, delayed, reordered, or duplicated, and clocks on different machines can drift. Despite all of this, the system must ensure that all nodes agree on the same value (safety) and that the system eventually makes a decision rather than deadlocking forever (liveness). The impossibility result known as FLP (Fischer, Lynch, Paterson, 1985) proved that in a fully asynchronous system where even one node can crash, no deterministic algorithm can guarantee both safety and liveness. This means every practical consensus algorithm makes trade-offs — typically by introducing timeouts and randomization to ensure liveness in practice while maintaining safety absolutely.

Why does this matter? Because consensus is the building block underneath almost every reliable distributed system. When a database replicates writes to multiple nodes and guarantees consistency, it uses consensus. When etcd or ZooKeeper provides a reliable configuration store, they use consensus internally. When Kafka ensures that all consumers of a partition see messages in the same order, it relies on consensus through its replication protocol. Without consensus, distributed systems would be limited to eventual consistency and could not provide the strong guarantees that many applications require.

**2. Paxos — The Original (and Notoriously Difficult) Algorithm**

Paxos, invented by Leslie Lamport in 1989 (published in 1998), was the first proven consensus algorithm and remains one of the most important results in distributed computing. Paxos works in two phases. In **Phase 1 (Prepare)**, a proposer selects a unique proposal number and sends a Prepare request to a majority of nodes (acceptors). Each acceptor promises not to accept any proposal with a lower number and returns any value it has already accepted. In **Phase 2 (Accept)**, if the proposer receives promises from a majority, it sends an Accept request with the proposal number and a value (either the highest-numbered value returned by any acceptor, or the proposer's own value if no values were returned). If a majority of acceptors accept this proposal, the value is chosen and consensus is achieved.

Paxos is provably correct and can tolerate the failure of up to (n-1)/2 nodes in a cluster of n. However, it is notoriously difficult to understand, implement, and optimize. Lamport himself presented it using an analogy about a fictional Greek parliament, which many readers found more confusing than helpful. The basic (single-decree) Paxos reaches consensus on a single value, but real systems need to agree on a sequence of values (a replicated log). The extension to Multi-Paxos, which elects a stable leader to streamline the protocol, is underspecified in the original paper, leading to many incompatible implementations. Google used Paxos in their Chubby lock service and Spanner database, but Googlers famously described building a production Paxos implementation as extremely challenging.

**3. Raft — Consensus Made Understandable**

Raft was designed by Diego Ongaro and John Ousterhout in 2013 with one explicit goal: to be as understandable as Paxos is confusing, while providing equivalent safety guarantees. Raft achieves this by decomposing consensus into three cleanly separated sub-problems: leader election, log replication, and safety.

Every Raft node is in one of three states: **Follower** (passive, responds to requests from leader and candidates), **Candidate** (actively seeking to become leader), or **Leader** (handles all client requests and replicates log entries). Time is divided into terms (logical clock periods). Each term begins with an election: when a follower does not receive a heartbeat from the leader within a randomized timeout (150-300ms), it transitions to candidate, increments the term, and requests votes from all other nodes. A candidate that receives votes from a majority becomes leader for that term and begins sending periodic heartbeats to maintain authority.

**Log replication** is the core mechanism for achieving consensus on a sequence of operations. When the leader receives a client request, it appends the request as a new entry to its log and sends AppendEntries RPCs to all followers. Each follower appends the entry to its own log and acknowledges. When the leader has received acknowledgments from a majority, it considers the entry "committed" — meaning it is durable and will not be lost. The leader then applies the committed entry to its state machine and responds to the client. Followers apply committed entries to their own state machines in the same order, ensuring all nodes maintain identical state.

**Safety** is guaranteed by several invariants: a leader's log always contains all committed entries from all previous terms (the Election Restriction ensures this by only electing candidates whose log is at least as up-to-date as the voter's), and entries are committed only when a majority of nodes have them. These invariants ensure that committed entries are never lost, even through leader changes and node failures.

**4. PBFT — Handling Malicious Nodes (Byzantine Faults)**

Paxos and Raft assume that nodes are honest — they may crash or become unreachable, but they never send false or malicious messages. **Byzantine Fault Tolerance (BFT)** addresses a harder problem: what if some nodes are actively malicious, sending incorrect data, lying about their state, or colluding to corrupt the system? This is essential in environments where you cannot trust all participants — blockchain networks, multi-party computation, and any system where nodes are operated by different organizations with potentially competing interests.

**PBFT (Practical Byzantine Fault Tolerance)**, proposed by Miguel Castro and Barbara Liskov in 1999, was the first consensus algorithm practical enough for real systems that tolerates Byzantine faults. It requires at least 3f+1 nodes to tolerate f Byzantine (malicious) nodes — for example, 7 nodes to tolerate 2 malicious ones. The protocol works in three phases: Pre-Prepare (the leader proposes a value), Prepare (nodes broadcast their agreement), and Commit (nodes confirm the decision). A value is committed only when 2f+1 nodes agree, ensuring that the honest majority always outweighs the malicious minority. PBFT's communication complexity is O(n^2) per decision (every node talks to every other node), which limits its scalability. Modern blockchain systems have developed variations (Tendermint, HotStuff) that reduce communication overhead while maintaining Byzantine tolerance.

**5. Comparing the Algorithms — When to Use What**

The choice between consensus algorithms depends on your trust model and performance requirements. **Raft** is the right choice for the vast majority of production distributed systems: it is well-understood, has excellent implementations (etcd, Consul, CockroachDB, TiKV), and provides strong consistency with good performance. Use Raft (or a Raft-based system) for replicated databases, configuration stores, service discovery, and distributed locking. **Paxos** is mathematically equivalent to Raft in safety and fault tolerance, but Raft is preferred for new implementations because of its clarity. You will encounter Paxos in existing systems (Google Spanner, AWS internally) and academic literature. **PBFT and its variants** are necessary only when Byzantine fault tolerance is required — blockchain networks, multi-tenant systems where operators are mutually distrusting, or safety-critical systems where hardware faults could cause nodes to send corrupted data. The overhead of Byzantine tolerance (3f+1 nodes instead of 2f+1, and O(n^2) communication) means you should not use it unless you actually need to tolerate malicious behavior.

**6. Real-World Use Cases and the Consensus Landscape**

Consensus algorithms underpin the most critical infrastructure in modern computing. **etcd** (Raft) is the backbone of Kubernetes, storing all cluster state and configuration. **CockroachDB** and **TiKV** (Raft) use consensus for strongly consistent distributed SQL storage. **Apache Kafka** uses a Raft-based controller quorum (KRaft) for metadata management, replacing its earlier ZooKeeper dependency. **Google Spanner** (Paxos) provides globally consistent transactions across continents. **Consul** (Raft) provides service discovery and configuration management. In the blockchain world, Bitcoin uses Proof of Work (a probabilistic alternative to classical consensus), Ethereum moved to Proof of Stake, and permissioned blockchains like Hyperledger Fabric use PBFT-derived protocols. Understanding consensus is not just academic — it is the key to understanding how reliability and consistency are achieved in every distributed system you will ever build or operate.`,
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
