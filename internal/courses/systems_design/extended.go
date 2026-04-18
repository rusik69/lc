package systemsdesign

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSystemsDesignModules([]problems.CourseModule{
		{
			ID:          2415,
			Title:       "CDN and Content Delivery",
			Description: "Learn how Content Delivery Networks work, including edge caching, origin servers, cache invalidation, and global content distribution strategies.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "CDN Fundamentals",
					Content: `A Content Delivery Network (CDN) is a geographically distributed network of proxy servers that cache content close to end users to reduce latency and improve performance.

**Why CDNs Matter:**
Without a CDN, a user in Tokyo requesting content from a server in New York experiences ~200ms round-trip latency just from the speed of light through fiber optics. A CDN with an edge server in Tokyo reduces this to ~5ms.

**How CDNs Work:**

` + "```" + `
           Without CDN:
User (Tokyo) ────────────────────────────→ Origin (New York)
              ~200ms RTT round trip

           With CDN:
User (Tokyo) ───→ CDN Edge (Tokyo) ───→ Origin (New York)
               ~5ms RTT    (only on cache MISS)
` + "```" + `

**CDN Architecture:**

` + "```" + `
┌──────────┐     ┌───────────────┐     ┌──────────┐
│  User    │────→│  CDN Edge     │────→│  Origin  │
│ (Browser)│     │  (PoP)        │     │  Server  │
└──────────┘     │               │     └──────────┘
                 │ Cache HIT?    │
                 │ Yes → Return  │
                 │ No  → Fetch   │
                 │        from   │
                 │        origin │
                 └───────────────┘
` + "```" + `

**Key CDN Concepts:**

**1. Point of Presence (PoP):**
- Physical location with CDN edge servers
- Major CDNs have 200+ PoPs worldwide
- Each PoP serves nearby users
- Examples: Cloudflare (275+ cities), AWS CloudFront (400+ PoPs)

**2. Edge Server:**
- Server at a PoP that caches and serves content
- Handles TLS termination (reducing latency)
- May compress, optimize, or transform content
- Runs application logic at the edge (edge computing)

**3. Origin Server:**
- Your actual web server or storage (S3, etc.)
- CDN fetches content from origin on cache miss
- Can be any HTTP server
- Origin shield: intermediate cache layer to reduce origin load

**Cache Control:**

**Cache-Control Headers:**
- ` + "`" + `Cache-Control: public, max-age=31536000` + "`" + ` — Cache for 1 year (static assets)
- ` + "`" + `Cache-Control: private, no-cache` + "`" + ` — Don't cache (user-specific data)
- ` + "`" + `Cache-Control: s-maxage=3600` + "`" + ` — CDN caches for 1 hour (different from browser)
- ` + "`" + `Vary: Accept-Encoding` + "`" + ` — Cache different versions based on encoding

**Cache Invalidation Strategies:**
1. **TTL-based**: Content expires after set time (simple, eventual consistency)
2. **Purge/Invalidate API**: Actively remove content from cache (immediate, more complex)
3. **Versioned URLs**: ` + "`" + `/style.v2.css` + "`" + ` or ` + "`" + `/style.css?v=abc123` + "`" + ` (recommended for static assets)
4. **Stale-while-revalidate**: Serve stale content while fetching fresh version in background

**CDN Use Cases:**
- **Static assets**: CSS, JS, images, fonts (most common)
- **Video streaming**: HLS/DASH segment delivery (Netflix, YouTube)
- **API acceleration**: Cache API responses at edge
- **DDoS protection**: Absorb attack traffic across distributed PoPs
- **Edge computing**: Run logic at the edge (Cloudflare Workers, Lambda@Edge)

**Choosing a CDN:**

| Feature | Cloudflare | AWS CloudFront | Akamai | Fastly |
|---------|-----------|----------------|--------|--------|
| PoPs | 275+ | 400+ | 4000+ | 70+ |
| Edge compute | Workers | Lambda@Edge | EdgeWorkers | Compute@Edge |
| DDoS | Excellent | Good | Excellent | Good |
| Pricing | Free tier | Pay per use | Enterprise | Pay per use |
| Best for | Most sites | AWS users | Enterprise | Real-time purge |`,
					CodeExamples: `# CDN Configuration Examples

# Cloudflare Page Rules (declarative)
# Cache everything under /static/ for 1 year
URL: example.com/static/*
Cache Level: Cache Everything
Edge Cache TTL: 31536000

# Nginx origin server cache headers
location /static/ {
    add_header Cache-Control "public, max-age=31536000, immutable";
    add_header Vary "Accept-Encoding";
}

location /api/ {
    add_header Cache-Control "public, s-maxage=60, max-age=0";
    # CDN caches for 60s, browser always revalidates
}

location /user/ {
    add_header Cache-Control "private, no-store";
    # Never cache user-specific data
}

# AWS CloudFront distribution (Terraform)
resource "aws_cloudfront_distribution" "cdn" {
  origin {
    domain_name = aws_s3_bucket.static.bucket_domain_name
    origin_id   = "S3-static"
  }

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-static"

    forwarded_values {
      query_string = false
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 86400    # 1 day
    max_ttl                = 31536000  # 1 year
  }
}

# Cache invalidation via API
aws cloudfront create-invalidation \
  --distribution-id EXAMPLE123 \
  --paths "/index.html" "/api/*"

# Versioned URLs (best practice for static assets)
# Instead of: /main.css
# Use:        /main.abc123.css
# Build tools generate unique hashes per content change
# Can set very long cache TTL since URL changes = new content`,
				},
				{
					Title: "DNS and Domain Name System",
					Content: `DNS (Domain Name System) is the phone book of the internet. It translates human-readable domain names (google.com) into IP addresses (142.250.80.46) that computers use to route traffic.

**DNS Resolution Process:**

` + "```" + `
Browser → Local Cache → OS Cache → Resolver → Root NS → TLD NS → Authoritative NS
                                    (ISP)      (.com)             (google.com)

Step by step:
1. Browser checks its cache
2. OS checks /etc/hosts and its cache
3. Recursive resolver (ISP/8.8.8.8) checks its cache
4. If not cached, asks Root nameserver: "Where is .com?"
5. Root says: "Ask the .com TLD nameserver at X.X.X.X"
6. Resolver asks .com TLD: "Where is google.com?"
7. TLD says: "Ask Google's authoritative NS at Y.Y.Y.Y"
8. Resolver asks authoritative NS: "What's the IP for google.com?"
9. Gets answer: 142.250.80.46, caches it
10. Returns IP to browser
` + "```" + `

**DNS Record Types:**

| Record | Purpose | Example |
|--------|---------|---------|
| A | Maps domain to IPv4 | google.com → 142.250.80.46 |
| AAAA | Maps domain to IPv6 | google.com → 2607:f8b0:4004:800::200e |
| CNAME | Alias to another domain | www.example.com → example.com |
| MX | Mail server | example.com → mail.example.com (priority 10) |
| TXT | Text records | SPF, DKIM, domain verification |
| NS | Nameserver | example.com → ns1.provider.com |
| SRV | Service location | _sip._tcp.example.com → sipserver:5060 |
| CAA | Certificate authority | example.com → letsencrypt.org |

**DNS for Systems Design:**

**Global Server Load Balancing (GSLB):**
DNS can route users to the nearest data center:
- User in Europe → eu.example.com (52.X.X.X)
- User in Asia → ap.example.com (13.X.X.X)
- Uses GeoDNS or latency-based routing

**DNS TTL Trade-offs:**
- **Low TTL (60s)**: Fast failover, more DNS queries, higher load on nameservers
- **High TTL (86400s)**: Fewer queries, slower failover, better caching
- **Compromise**: 300s (5 min) for most services, lower during migrations

**DNS Failover:**
Health check + automatic DNS record update when primary server fails. AWS Route 53, Cloudflare, and similar services provide this.

**DNS Security:**
- **DNSSEC**: Cryptographic signatures verify DNS responses aren't tampered with
- **DNS over HTTPS (DoH)**: Encrypts DNS queries (privacy)
- **DNS over TLS (DoT)**: Similar encryption, different protocol

**Common DNS Anti-Patterns:**
- Using DNS as a load balancer (client caching makes it unreliable)
- Very low TTLs (increases load, not always respected by clients)
- Single point of failure (always have multiple nameservers)`,
					CodeExamples: `# DNS Lookup Examples

# Basic lookup
dig google.com
# or
nslookup google.com

# Trace full resolution path
dig +trace google.com

# Check specific record types
dig MX example.com
dig TXT example.com
dig AAAA example.com
dig NS example.com

# Check with specific DNS server
dig @8.8.8.8 example.com

# AWS Route 53 Configuration (Terraform)
resource "aws_route53_record" "www" {
  zone_id = aws_route53_zone.primary.zone_id
  name    = "www.example.com"
  type    = "A"

  alias {
    name                   = aws_lb.main.dns_name
    zone_id                = aws_lb.main.zone_id
    evaluate_target_health = true
  }
}

# Latency-based routing (serve from nearest region)
resource "aws_route53_record" "api_us" {
  zone_id        = aws_route53_zone.primary.zone_id
  name           = "api.example.com"
  type           = "A"
  set_identifier = "us-east-1"

  latency_routing_policy {
    region = "us-east-1"
  }

  alias {
    name    = aws_lb.us_east.dns_name
    zone_id = aws_lb.us_east.zone_id
  }
}

resource "aws_route53_record" "api_eu" {
  zone_id        = aws_route53_zone.primary.zone_id
  name           = "api.example.com"
  type           = "A"
  set_identifier = "eu-west-1"

  latency_routing_policy {
    region = "eu-west-1"
  }

  alias {
    name    = aws_lb.eu_west.dns_name
    zone_id = aws_lb.eu_west.zone_id
  }
}

# Health check + failover
resource "aws_route53_health_check" "primary" {
  fqdn              = "primary.example.com"
  port              = 443
  type              = "HTTPS"
  request_interval  = 30
  failure_threshold = 3
}`,
				},
				{
					Title: "Proxy Servers and Reverse Proxies",
					Content: `Proxies are intermediaries between clients and servers. Understanding forward and reverse proxies is essential for systems design.

**Forward Proxy:**
Sits between clients and the internet. Acts on behalf of clients.

` + "```" + `
Client → Forward Proxy → Internet → Server
` + "```" + `

**Use Cases:**
- Corporate networks (filter/monitor traffic)
- VPNs (hide client IP)
- Content filtering (block sites)
- Caching (reduce bandwidth)

**Reverse Proxy:**
Sits in front of servers. Acts on behalf of servers.

` + "```" + `
Client → Internet → Reverse Proxy → Server(s)
` + "```" + `

**Use Cases:**
- **Load balancing**: Distribute requests across servers
- **SSL termination**: Handle TLS encryption/decryption
- **Caching**: Cache responses to reduce server load
- **Compression**: Compress responses before sending to client
- **Security**: Hide server details, WAF, rate limiting
- **Request routing**: Route to different backends based on path/header

**Popular Reverse Proxies:**

| Software | Best For | Key Feature |
|----------|----------|-------------|
| Nginx | General purpose | High performance, low memory |
| HAProxy | Load balancing | Advanced health checks, TCP/HTTP |
| Envoy | Service mesh | gRPC support, observability |
| Traefik | Containers | Auto-discovery, Let's Encrypt |
| Caddy | Simple setup | Automatic HTTPS |

**Load Balancing Algorithms:**

1. **Round Robin**: Requests distributed sequentially (simple, no affinity)
2. **Weighted Round Robin**: Higher-capacity servers get more requests
3. **Least Connections**: Send to server with fewest active connections
4. **IP Hash**: Client IP determines server (session affinity)
5. **Random**: Random server selection (surprisingly effective at scale)
6. **Consistent Hashing**: Minimize redistribution when servers change

**Layer 4 vs Layer 7 Load Balancing:**

**Layer 4 (Transport):**
- Routes based on IP and port
- Faster (no payload inspection)
- Can't make decisions based on content
- Good for: TCP/UDP services, database connections

**Layer 7 (Application):**
- Routes based on HTTP headers, path, cookies, etc.
- More flexible
- Can do content-based routing
- Good for: Web applications, API routing, A/B testing

**Health Checks:**
- **Active**: Proxy periodically pings servers
- **Passive**: Proxy monitors real traffic for failures
- **Types**: TCP connect, HTTP status, custom script

**SSL/TLS Termination:**
Reverse proxy handles TLS encryption, so backend servers receive plain HTTP. Benefits:
- Centralized certificate management
- Reduced CPU load on app servers
- Easier debugging (can inspect traffic between proxy and backend)`,
					CodeExamples: `# Nginx Reverse Proxy Configuration

# Basic reverse proxy
upstream backend {
    server 10.0.0.1:8080 weight=3;
    server 10.0.0.2:8080 weight=2;
    server 10.0.0.3:8080 weight=1;
    server 10.0.0.4:8080 backup;  # Only used if others fail
}

server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate     /etc/ssl/cert.pem;
    ssl_certificate_key /etc/ssl/key.pem;

    # Path-based routing
    location /api/v1/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /static/ {
        # Serve static files directly
        root /var/www/html;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "OK";
    }
}

# HAProxy Configuration
frontend http_front
    bind *:80
    bind *:443 ssl crt /etc/ssl/cert.pem
    default_backend app_servers

backend app_servers
    balance roundrobin
    option httpchk GET /health
    server app1 10.0.0.1:8080 check inter 5s fall 3 rise 2
    server app2 10.0.0.2:8080 check inter 5s fall 3 rise 2
    server app3 10.0.0.3:8080 check inter 5s fall 3 rise 2`,
				},
			},
		},
		{
			ID:          2416,
			Title:       "Storage and Object Storage",
			Description: "Learn about different storage solutions including block storage, file storage, object storage, and when to use each in system design.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Storage Types for Systems Design",
					Content: `Choosing the right storage solution is a critical system design decision. Each type has different performance characteristics, scalability limits, and cost profiles.

**Block Storage:**
Raw storage volumes that can be formatted with any file system. Think of it as a virtual hard drive.

- **Examples**: AWS EBS, Azure Managed Disks, GCP Persistent Disks
- **Use cases**: Databases, OS boot volumes, high-IOPS applications
- **Performance**: Lowest latency, highest IOPS
- **Scaling**: Vertical (larger volumes), limited horizontal
- **Access**: Single instance at a time (usually)

**File Storage (NAS):**
Shared file system accessible by multiple instances simultaneously.

- **Examples**: AWS EFS, Azure Files, GCP Filestore, NFS
- **Use cases**: Shared configuration, CMS assets, legacy apps needing file system
- **Performance**: Good throughput, higher latency than block
- **Scaling**: Automatic (managed services), multi-attach
- **Access**: Multiple instances via NFS/SMB

**Object Storage:**
Flat namespace of objects (files) accessed via HTTP API. The backbone of modern cloud storage.

- **Examples**: AWS S3, Azure Blob Storage, GCP Cloud Storage, MinIO
- **Use cases**: Static assets, backups, data lake, log storage
- **Performance**: Higher latency (~50-100ms), massive throughput
- **Scaling**: Virtually unlimited (S3 handles exabytes)
- **Access**: HTTP API (PUT, GET, DELETE)
- **Cost**: Cheapest per GB ($0.023/GB/month for S3 Standard)

**Comparison Table:**

| Feature | Block | File | Object |
|---------|-------|------|--------|
| Latency | <1ms | 1-10ms | 50-200ms |
| IOPS | 10K-100K+ | 1K-10K | Low |
| Throughput | High | High | Very High |
| Max Size | 16 TB | Unlimited | 5 TB per object |
| Scaling | Vertical | Horizontal | Unlimited |
| Access | Single instance | Multi-instance | HTTP API |
| Cost/GB | $$$ | $$ | $ |
| Best For | Databases | Shared files | Static content, backups |

**Object Storage Patterns:**

**1. Data Lake:**
- Store raw data in object storage (cheapest per GB)
- Process with Spark, Athena, BigQuery
- Schema-on-read (no predefined schema)

**2. Static Website Hosting:**
- HTML/CSS/JS on S3 + CloudFront CDN
- Serverless, infinitely scalable, pennies per month

**3. Backup and Archival:**
- Lifecycle policies: S3 Standard → Infrequent Access → Glacier
- Cost optimization through storage tiers
- Cross-region replication for disaster recovery

**4. Media Storage:**
- Upload images/videos to S3
- Generate presigned URLs for secure downloads
- Trigger Lambda for processing (thumbnails, transcoding)

**S3 Storage Classes:**

| Class | Use Case | Cost/GB/mo | Retrieval |
|-------|----------|-----------|-----------|
| Standard | Frequently accessed | $0.023 | Immediate |
| IA | Infrequent access | $0.0125 | Immediate |
| Glacier Instant | Archive, instant access | $0.004 | Immediate |
| Glacier Flexible | Archive, hours | $0.0036 | 3-12 hours |
| Glacier Deep | Long-term archive | $0.00099 | 12-48 hours |`,
					CodeExamples: `# S3 Operations (AWS CLI)

# Create bucket
aws s3 mb s3://my-app-data

# Upload with storage class
aws s3 cp backup.tar.gz s3://my-app-data/backups/ \
  --storage-class STANDARD_IA

# Presigned URL (secure temporary access)
aws s3 presign s3://my-app-data/files/report.pdf \
  --expires-in 3600  # 1 hour

# Lifecycle policy (Terraform)
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "archive-old-data"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

# Go: S3 upload with presigned URL
func uploadToS3(bucket, key string, data []byte) (string, error) {
    cfg, _ := config.LoadDefaultConfig(context.TODO())
    client := s3.NewFromConfig(cfg)

    _, err := client.PutObject(context.TODO(), &s3.PutObjectInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(key),
        Body:   bytes.NewReader(data),
    })
    if err != nil {
        return "", err
    }

    // Generate presigned URL
    presigner := s3.NewPresignClient(client)
    req, err := presigner.PresignGetObject(context.TODO(),
        &s3.GetObjectInput{
            Bucket: aws.String(bucket),
            Key:    aws.String(key),
        },
        s3.WithPresignExpires(1*time.Hour),
    )
    return req.URL, err
}`,
				},
				{
					Title: "Database Selection Guide",
					Content: `Choosing the right database is one of the most consequential system design decisions. Different databases excel at different workloads.

**Relational Databases (SQL):**

**When to Use:**
- Structured data with well-defined schema
- ACID transactions required
- Complex queries with JOINs
- Data integrity is critical (financial, healthcare)

**Examples:** PostgreSQL, MySQL, SQL Server, Oracle

**PostgreSQL vs MySQL:**
- **PostgreSQL**: Better for complex queries, JSON support, extensions, geospatial
- **MySQL**: Better for simple reads, replication, ecosystem (WordPress, etc.)

**NoSQL Categories:**

**1. Document Stores:**
- **MongoDB, CouchDB, Firestore**
- Schema-flexible JSON/BSON documents
- Good for: CMS, user profiles, catalogs, rapid prototyping
- Bad for: Complex joins, strict consistency

**2. Key-Value Stores:**
- **Redis, DynamoDB, Memcached**
- Simple key → value mapping, fastest lookups
- Good for: Caching, sessions, feature flags, rate limiting
- Bad for: Complex queries, relationships

**3. Wide-Column Stores:**
- **Cassandra, HBase, ScyllaDB**
- Rows with dynamic columns, optimized for writes
- Good for: Time-series data, IoT, logging, analytics
- Bad for: Ad-hoc queries, transactions

**4. Graph Databases:**
- **Neo4j, Amazon Neptune, JanusGraph**
- Nodes and edges, optimized for relationship queries
- Good for: Social networks, recommendation engines, fraud detection
- Bad for: Bulk data processing, simple CRUD

**5. Time-Series Databases:**
- **InfluxDB, TimescaleDB, Prometheus**
- Optimized for time-stamped data
- Good for: Monitoring, IoT sensor data, financial tick data
- Bad for: General-purpose storage

**Decision Framework:**

` + "```" + `
Start
  │
  ├─ Need ACID transactions? → PostgreSQL/MySQL
  │
  ├─ High-throughput writes? → Cassandra/ScyllaDB
  │
  ├─ Flexible schema, documents? → MongoDB
  │
  ├─ Sub-millisecond lookups? → Redis/DynamoDB
  │
  ├─ Complex relationships? → Neo4j
  │
  ├─ Time-series data? → InfluxDB/TimescaleDB
  │
  └─ Full-text search? → Elasticsearch
` + "```" + `

**Polyglot Persistence:**
Modern systems use multiple databases, each for what it does best:
- PostgreSQL for orders (ACID)
- Redis for sessions (speed)
- Elasticsearch for search (full-text)
- S3 for media (cost)
- ClickHouse for analytics (OLAP)

**CAP Theorem Applied:**

| Database | CAP | Notes |
|----------|-----|-------|
| PostgreSQL | CP | Strong consistency, partition handling via failover |
| MongoDB | CP | Configurable, default is strong consistency |
| Cassandra | AP | Tunable consistency (ONE/QUORUM/ALL) |
| DynamoDB | AP/CP | Configurable per operation |
| Redis | AP | Single-node consistent, cluster can lose writes |`,
					CodeExamples: `# PostgreSQL: Great for complex queries with ACID
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(id),
    total DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Complex query with JOIN, aggregation
SELECT c.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE o.created_at > NOW() - INTERVAL '30 days'
GROUP BY c.name
HAVING SUM(o.total) > 1000
ORDER BY total_spent DESC;

# MongoDB: Great for flexible documents
db.products.insertOne({
    name: "Laptop Pro",
    price: 1299.99,
    specs: {
        cpu: "M2 Pro",
        ram: "16GB",
        storage: "512GB SSD"
    },
    tags: ["electronics", "computers", "apple"],
    reviews: [
        { user: "alice", rating: 5, text: "Amazing!" },
        { user: "bob", rating: 4, text: "Good value" }
    ]
})

# Redis: Great for caching and real-time data
SET session:user123 '{"name":"alice","cart":["item1","item2"]}' EX 3600
GET session:user123

# Rate limiting with Redis
INCR ratelimit:user123:minute
EXPIRE ratelimit:user123:minute 60
# Returns count — if > threshold, reject

# Cassandra: Great for high-throughput time-series writes
CREATE TABLE events (
    device_id UUID,
    event_time TIMESTAMP,
    event_type TEXT,
    data TEXT,
    PRIMARY KEY ((device_id), event_time)
) WITH CLUSTERING ORDER BY (event_time DESC);

-- Fast write (no locking, distributed)
INSERT INTO events (device_id, event_time, event_type, data)
VALUES (uuid(), toTimestamp(now()), 'temperature', '{"value": 72.5}');

-- Fast read for one device's recent events
SELECT * FROM events
WHERE device_id = ?
AND event_time > '2024-01-01'
LIMIT 100;`,
				},
			},
		},
		{
			ID:          2417,
			Title:       "Distributed Systems Fundamentals",
			Description: "Learn essential distributed systems concepts: CAP theorem deep dive, consistency models, clock synchronization, and partition handling.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "CAP Theorem Deep Dive",
					Content: `The CAP theorem (Brewer's theorem, 2000) states that a distributed data store can provide at most two of three guarantees simultaneously:

**C - Consistency:** Every read receives the most recent write or an error. All nodes see the same data at the same time.

**A - Availability:** Every request receives a response (not an error), without guarantee that it contains the most recent write.

**P - Partition Tolerance:** The system continues to operate despite network partitions (messages being dropped/delayed between nodes).

**The Reality:**
In distributed systems, **network partitions WILL happen** (cables break, switches fail, cloud regions disconnect). So you must choose between C and A during a partition:

- **CP (Consistency + Partition Tolerance)**: During partition, reject requests rather than return stale data
  - Examples: PostgreSQL (via failover), MongoDB, HBase, Zookeeper
  - Use when: Financial transactions, inventory counts, anything where stale data = bad

- **AP (Availability + Partition Tolerance)**: During partition, return data even if it might be stale
  - Examples: Cassandra, DynamoDB (default), CouchDB, DNS
  - Use when: Social media feeds, shopping carts, read-heavy workloads

**Beyond CAP — PACELC:**

The PACELC theorem extends CAP: "If there's a Partition, choose between Availability and Consistency. Else (normal operation), choose between Latency and Consistency."

` + "```" + `
PACELC:
  Partition?
    Yes → A or C?
    No  → L or C?

Examples:
  DynamoDB:  PA/EL (Available during partition, Low latency normally)
  PostgreSQL: PC/EC (Consistent always, slower due to consensus)
  Cassandra:  PA/EL (Tunable — can choose per query)
  MongoDB:    PC/EC (Consistent by default)
` + "```" + `

**Consistency Models Spectrum:**

From strongest to weakest:
1. **Linearizability**: Every operation appears to take effect instantaneously (strongest)
2. **Sequential consistency**: All nodes see operations in same order
3. **Causal consistency**: Causally related operations seen in order
4. **Eventual consistency**: Given enough time, all nodes converge (weakest)

**Real-World Trade-offs:**

**Banking (CP):**
- Can't show wrong balance (consistency critical)
- Better to be temporarily unavailable than show incorrect data
- Use: PostgreSQL with synchronous replication

**Social Media (AP):**
- Showing a slightly stale feed is acceptable
- Being unavailable is worse than stale data
- Use: Cassandra with eventual consistency

**Shopping Cart (AP → CP for checkout):**
- Cart browsing: eventual consistency OK (AP)
- Checkout/payment: strong consistency required (CP)
- Use: DynamoDB for cart, PostgreSQL for orders`,
					CodeExamples: `# Consistency in Practice

# PostgreSQL: Strong Consistency (CP)
# Synchronous replication — write waits for standby acknowledgment
synchronous_standby_names = 'first 1 (standby1, standby2)'
synchronous_commit = on

# Client sees consistent reads after write completes
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;  -- Only succeeds if standby confirms

# Cassandra: Tunable Consistency (AP default)
# Write with different consistency levels:

# ONE: Write to one replica, return (fast, might lose data)
INSERT INTO users (id, name) VALUES (1, 'alice')
USING CONSISTENCY ONE;

# QUORUM: Write to majority of replicas (balanced)
INSERT INTO users (id, name) VALUES (1, 'alice')
USING CONSISTENCY QUORUM;

# ALL: Write to all replicas (slowest, most consistent)
INSERT INTO users (id, name) VALUES (1, 'alice')
USING CONSISTENCY ALL;

# Read with QUORUM + Write with QUORUM = Strong Consistency
# (Read quorum + Write quorum > Replication factor)

# DynamoDB: Configurable per read
# Eventually consistent read (default, cheaper)
response = table.get_item(
    Key={'id': '123'},
    ConsistentRead=False
)

# Strongly consistent read (more expensive, higher latency)
response = table.get_item(
    Key={'id': '123'},
    ConsistentRead=True
)`,
				},
				{
					Title: "Distributed Consensus and Replication",
					Content: `Distributed consensus is the process of getting multiple nodes to agree on a single value or sequence of operations. It's the foundation of reliable distributed systems.

**Why Consensus Matters:**
- Leader election (who processes writes?)
- Transaction commit (should this transaction commit or abort?)
- State machine replication (all replicas process same operations in same order)
- Configuration management (all nodes see same config)

**Raft Consensus Algorithm:**

Raft (2014) is the most understandable consensus algorithm. Used by etcd, CockroachDB, TiKV, Consul.

**Three Roles:**
1. **Leader**: Handles all client requests, replicates to followers
2. **Follower**: Passive, replicates leader's log
3. **Candidate**: Trying to become leader (during election)

**How Raft Works:**

` + "```" + `
Normal Operation:
Client → Leader → Replicate to Followers → Majority ACK → Commit → Reply

Leader Election:
Follower timeout → Becomes Candidate → Requests votes → Majority votes → Becomes Leader

Log Replication:
Leader appends entry → Sends to all followers → Followers append → ACK to leader
→ Once majority ACK → Leader commits → Followers commit
` + "```" + `

**Key Properties:**
- **Safety**: At most one leader per term
- **Liveness**: System eventually elects a leader
- **Log matching**: If two logs have entry with same index and term, all preceding entries are identical

**Replication Strategies:**

**1. Single-Leader (Primary-Secondary):**
- One node handles all writes
- Replicates to read replicas
- Simple, but leader is bottleneck
- Examples: PostgreSQL streaming replication, MySQL replication

**2. Multi-Leader:**
- Multiple nodes accept writes
- Conflict resolution needed (last-write-wins, CRDTs)
- Good for: multi-datacenter setups
- Examples: CockroachDB, MySQL Group Replication

**3. Leaderless (Quorum-based):**
- Any node accepts reads and writes
- Quorum determines success (W + R > N)
- Good for: high availability, no single point of failure
- Examples: Cassandra, DynamoDB, Riak

**Quorum Math:**
- N = total replicas
- W = write quorum (nodes that must ACK write)
- R = read quorum (nodes that must respond to read)
- **W + R > N** guarantees reading latest write
- **Common config**: N=3, W=2, R=2 (fault tolerance of 1 node)

**Conflict Resolution:**
When writes conflict (especially in multi-leader or leaderless):
1. **Last-Write-Wins (LWW)**: Timestamp determines winner (simple, data loss)
2. **Version Vectors**: Track causal history, detect conflicts
3. **CRDTs**: Data structures that automatically merge without conflicts
4. **Application-level**: Present conflicts to user (Google Docs, Git)`,
					CodeExamples: `# Replication Configuration Examples

# PostgreSQL: Single-Leader with streaming replication
# On primary:
wal_level = replica
max_wal_senders = 5
synchronous_standby_names = 'standby1'

# On standby:
primary_conninfo = 'host=primary port=5432 user=replicator'
hot_standby = on

# etcd: Raft consensus (used by Kubernetes)
# 3-node cluster for fault tolerance
etcd --name node1 \
  --initial-cluster "node1=http://10.0.0.1:2380,node2=http://10.0.0.2:2380,node3=http://10.0.0.3:2380" \
  --listen-peer-urls http://10.0.0.1:2380 \
  --listen-client-urls http://10.0.0.1:2379

# Check cluster health:
etcdctl endpoint health --cluster
# +-------------------+--------+-------+
# |    ENDPOINT       | HEALTH | ERROR |
# +-------------------+--------+-------+
# | 10.0.0.1:2379     | true   |       |
# | 10.0.0.2:2379     | true   |       |
# | 10.0.0.3:2379     | true   |       |
# +-------------------+--------+-------+

# Cassandra: Leaderless with tunable consistency
# Replication factor = 3
CREATE KEYSPACE my_app WITH replication = {
    'class': 'NetworkTopologyStrategy',
    'dc1': 3,  -- 3 replicas in datacenter 1
    'dc2': 3   -- 3 replicas in datacenter 2
};

# Quorum examples:
# N=3, W=2, R=2: Strong consistency (W+R=4 > N=3)
# N=3, W=1, R=1: Eventual consistency (W+R=2 < N=3)
# N=3, W=3, R=1: Fast reads, slow writes
# N=3, W=1, R=3: Fast writes, slow reads`,
				},
			},
		},
	})
}
