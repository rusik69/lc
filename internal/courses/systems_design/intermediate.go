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
					CodeExamples: `RESTful API Endpoint Examples:

GET /api/v1/users
Response: 200 OK
[
  {"id": 1, "name": "John", "email": "john@example.com"},
  {"id": 2, "name": "Jane", "email": "jane@example.com"}
]

GET /api/v1/users/123
Response: 200 OK
{"id": 123, "name": "John", "email": "john@example.com"}

POST /api/v1/users
Request: {"name": "Bob", "email": "bob@example.com"}
Response: 201 Created
{"id": 124, "name": "Bob", "email": "bob@example.com"}

PUT /api/v1/users/123
Request: {"name": "John Updated", "email": "john@example.com"}
Response: 200 OK
{"id": 123, "name": "John Updated", "email": "john@example.com"}

PATCH /api/v1/users/123
Request: {"name": "John Updated"}
Response: 200 OK
{"id": 123, "name": "John Updated", "email": "john@example.com"}

DELETE /api/v1/users/123
Response: 204 No Content

Pagination Example:

GET /api/v1/users?page=1&limit=10
Response: 200 OK
{
  "data": [...],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 100,
    "total_pages": 10,
    "has_next": true,
    "has_prev": false
  }
}

Filtering and Sorting:

GET /api/v1/users?status=active&sort=name&order=asc
GET /api/v1/users?created_after=2024-01-01&role=admin

API Versioning Examples:

URL Versioning:
GET /api/v1/users
GET /api/v2/users

Header Versioning:
GET /api/users
Headers: Accept: application/vnd.api+json;version=1

Query Parameter:
GET /api/users?version=1

Error Response Format:

400 Bad Request:
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input",
    "details": {
      "email": "Invalid email format"
    }
  }
}

404 Not Found:
{
  "error": {
    "code": "NOT_FOUND",
    "message": "User not found",
    "resource": "user",
    "id": 123
  }
}

500 Internal Server Error:
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An error occurred",
    "request_id": "abc123"
  }
}

HTTP Status Codes:

2xx Success:
- 200 OK: Successful GET, PUT, PATCH
- 201 Created: Successful POST
- 204 No Content: Successful DELETE

4xx Client Error:
- 400 Bad Request: Invalid input
- 401 Unauthorized: Authentication required
- 403 Forbidden: Not authorized
- 404 Not Found: Resource doesn't exist
- 409 Conflict: Resource conflict
- 429 Too Many Requests: Rate limit exceeded

5xx Server Error:
- 500 Internal Server Error: Server error
- 503 Service Unavailable: Service down`,
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
					CodeExamples: `Token Bucket Implementation:

class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity  # Max tokens
        self.tokens = capacity
        self.refill_rate = refill_rate  # Tokens per second
        self.last_refill = time.time()
    
    def allow_request(self):
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
    
    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now

Usage:
bucket = TokenBucket(capacity=100, refill_rate=10)  # 10 tokens/sec
if bucket.allow_request():
    process_request()
else:
    return 429 Too Many Requests

Fixed Window Example (Redis):

def check_rate_limit(user_id):
    key = f"rate_limit:{user_id}:{int(time.time() / 60)}"
    count = redis.incr(key)
    redis.expire(key, 60)
    
    if count == 1:
        # First request in window
        return True
    elif count > 100:  # Limit: 100 requests per minute
        return False
    return True

Sliding Window Example:

def check_sliding_window(user_id, limit=100, window=60):
    now = time.time()
    key = f"rate_limit:{user_id}"
    
    # Remove old entries
    redis.zremrangebyscore(key, 0, now - window)
    
    # Count current requests
    count = redis.zcard(key)
    
    if count < limit:
        # Add current request
        redis.zadd(key, {str(now): now})
        redis.expire(key, window)
        return True
    
    return False

Rate Limit Headers in Response:

200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1609459200

429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1609459200
Retry-After: 60

Leaky Bucket Implementation:

class LeakyBucket:
    def __init__(self, capacity, leak_rate):
        self.capacity = capacity
        self.queue = []
        self.leak_rate = leak_rate  # Requests per second
        self.last_leak = time.time()
    
    def allow_request(self):
        self._leak()
        if len(self.queue) < self.capacity:
            self.queue.append(time.time())
            return True
        return False
    
    def _leak(self):
        now = time.time()
        elapsed = now - self.last_leak
        leak_count = int(elapsed * self.leak_rate)
        
        if leak_count > 0:
            self.queue = self.queue[leak_count:]
            self.last_leak = now`,
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
					CodeExamples: `Logging Example:

Structured Logging (JSON):
{
  "timestamp": "2024-01-17T10:00:00Z",
  "level": "INFO",
  "service": "user-service",
  "request_id": "abc123",
  "message": "User created",
  "user_id": 123,
  "duration_ms": 45
}

Log Levels:
- DEBUG: Detailed debugging information
- INFO: General informational messages
- WARN: Warning messages
- ERROR: Error events
- FATAL: Critical failures

Metrics Collection Example:

Prometheus Metrics:
# Counter: Total requests
http_requests_total{method="GET", status="200"} 1500

# Gauge: Current connections
active_connections 42

# Histogram: Request duration
http_request_duration_seconds_bucket{le="0.1"} 1200
http_request_duration_seconds_bucket{le="0.5"} 1400
http_request_duration_seconds_bucket{le="1.0"} 1500

# Summary: Response size
http_response_size_bytes{quantile="0.5"} 1024
http_response_size_bytes{quantile="0.95"} 4096
http_response_size_bytes{quantile="0.99"} 8192

Key Metrics Dashboard:

Availability:
uptime_percentage = (total_time - downtime) / total_time * 100
Target: 99.9% (three nines)

Latency Percentiles:
p50 = 50ms   # Median response time
p95 = 200ms  # 95% of requests faster
p99 = 500ms  # 99% of requests faster

Throughput:
requests_per_second = total_requests / time_window
Current: 1000 RPS

Error Rate:
error_rate = (failed_requests / total_requests) * 100
Current: 0.1% (acceptable)

Resource Usage:
cpu_usage = 70%  # Target: < 80%
memory_usage = 8GB / 16GB = 50%  # Target: < 80%
disk_usage = 200GB / 500GB = 40%  # Target: < 85%

Alerting Configuration:

Critical Alerts:
- Error rate > 1% for 5 minutes
- Latency p99 > 1000ms for 5 minutes
- CPU usage > 90% for 10 minutes
- Disk usage > 95%
- Service down

Warning Alerts:
- Error rate > 0.5% for 10 minutes
- Latency p95 > 500ms for 10 minutes
- CPU usage > 80% for 15 minutes
- Memory usage > 85%

Alert Example:
{
  "alert_name": "high_error_rate",
  "severity": "critical",
  "condition": "error_rate > 1% for 5m",
  "notification": ["slack", "pagerduty"],
  "runbook": "https://wiki/runbooks/high-error-rate"
}

Dashboard Example (Grafana):

Panel 1: Request Rate
Query: rate(http_requests_total[5m])
Visualization: Line graph

Panel 2: Error Rate
Query: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])
Visualization: Gauge

Panel 3: Latency Percentiles
Query: histogram_quantile(0.99, http_request_duration_seconds)
Visualization: Line graph

Panel 4: Resource Usage
Query: cpu_usage, memory_usage, disk_usage
Visualization: Stacked area chart`,
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
					CodeExamples: `Trace Structure:

Trace (Request Journey):
├── Span 1: API Gateway (10ms)
│   ├── Span 2: Auth Service (5ms)
│   └── Span 3: User Service (20ms)
│       ├── Span 4: Database Query (15ms)
│       └── Span 5: Cache Lookup (2ms)
└── Span 6: Response (3ms)

Total Trace Duration: 10ms + 20ms + 3ms = 33ms

Span Example:

{
  "trace_id": "abc123",
  "span_id": "def456",
  "parent_span_id": "xyz789",
  "operation_name": "get_user",
  "service_name": "user-service",
  "start_time": "2024-01-17T10:00:00.000Z",
  "duration_ms": 25,
  "tags": {
    "http.method": "GET",
    "http.url": "/api/users/123",
    "http.status_code": 200
  },
  "logs": [
    {
      "timestamp": "2024-01-17T10:00:00.010Z",
      "fields": {"event": "cache_miss"}
    }
  ]
}

Trace Context Propagation:

HTTP Headers:
X-Trace-Id: abc123
X-Span-Id: def456
X-Parent-Span-Id: xyz789

gRPC Metadata:
trace-id: abc123
span-id: def456
parent-span-id: xyz789

Manual Instrumentation Example:

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

tracer = trace.get_tracer(__name__)

def get_user(user_id):
    with tracer.start_as_current_span("get_user") as span:
        span.set_attribute("user_id", user_id)
        
        # Database query
        with tracer.start_as_current_span("db_query") as db_span:
            user = database.get_user(user_id)
            db_span.set_attribute("db.query", "SELECT * FROM users")
            db_span.set_attribute("db.duration_ms", 15)
        
        span.set_attribute("http.status_code", 200)
        return user

Distributed Trace Flow:

Client Request
    |
    v
[API Gateway] ← Trace ID: abc123, Span: 1
    |
    | HTTP Header: X-Trace-Id: abc123, X-Span-Id: 1
    v
[Auth Service] ← Trace ID: abc123, Span: 2, Parent: 1
    |
    | HTTP Header: X-Trace-Id: abc123, X-Span-Id: 2
    v
[User Service] ← Trace ID: abc123, Span: 3, Parent: 2
    |
    | gRPC Metadata: trace-id: abc123, span-id: 3
    v
[Database] ← Trace ID: abc123, Span: 4, Parent: 3

All spans share same trace_id, enabling end-to-end visibility

Sampling Configuration:

# Sample 100% of traces in development
sampling_rate = 1.0

# Sample 10% of traces in production
sampling_rate = 0.1

# Adaptive sampling: Sample all errors, 10% of successful requests
if status_code >= 500:
    sample = True
else:
    sample = random() < 0.1

Jaeger Query Example:

Find traces with high latency:
operation="get_user" AND duration > 1000ms

Find error traces:
tags.http.status_code=500

Find traces for specific user:
tags.user_id=123

Trace Visualization:

Timeline View:
[==========API Gateway==========] 10ms
  [==Auth==] 5ms
  [==========User Service==========] 20ms
    [====DB====] 15ms
    [=Cache=] 2ms
[==Response==] 3ms

Shows where time is spent in distributed request`,
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
					CodeExamples: `Circuit Breaker State Machine:

CLOSED (Normal)
    |
    | (Failure rate > threshold)
    |
OPEN (Failing)
    |
    | (After timeout)
    |
HALF-OPEN (Testing)
    |
    | (Success)    | (Failure)
    |              |
CLOSED          OPEN

Circuit Breaker Implementation:

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None
    
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

Usage Example:

breaker = CircuitBreaker(failure_threshold=5, timeout=60)

def call_external_api():
    return breaker.call(
        requests.get,
        "https://api.example.com/data",
        timeout=5
    )

Configuration Example:

{
  "circuit_breaker": {
    "failure_threshold": 5,
    "timeout_seconds": 60,
    "half_open_max_calls": 3,
    "success_threshold": 2
  }
}

Behavior:
- CLOSED: Normal operation, track failures
- OPEN: Reject all requests immediately
- HALF_OPEN: Allow limited requests to test recovery
- If HALF_OPEN succeeds → CLOSED
- If HALF_OPEN fails → OPEN

Failure Handling:

try:
    result = call_external_api()
except CircuitBreakerOpenError:
    # Return cached data or default value
    return get_cached_data() or get_default_value()
except Exception as e:
    # Handle other errors
    log_error(e)
    raise`,
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
					CodeExamples: `Bulkhead Architecture:

Without Bulkhead (Single Pool):
[All Services] → [Shared Thread Pool] → [All Resources]
                    |
                    | (One service overloads)
                    |
                All services affected

With Bulkhead (Isolated Pools):
[Service A] → [Thread Pool A] → [Resources A]
[Service B] → [Thread Pool B] → [Resources B]
[Service C] → [Thread Pool C] → [Resources C]

Failure in Service A doesn't affect B or C

Thread Pool Isolation Example:

from concurrent.futures import ThreadPoolExecutor

# Separate thread pools per service
payment_pool = ThreadPoolExecutor(max_workers=10, thread_name_prefix="payment")
notification_pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix="notification")
analytics_pool = ThreadPoolExecutor(max_workers=20, thread_name_prefix="analytics")

def process_payment(order):
    payment_pool.submit(handle_payment, order)

def send_notification(user_id):
    notification_pool.submit(send_email, user_id)

def log_analytics(event):
    analytics_pool.submit(store_event, event)

# If payment service is slow, notification and analytics still work

Connection Pool Isolation:

# Separate connection pools per service
payment_db_pool = ConnectionPool(
    max_connections=10,
    database="payment_db"
)

user_db_pool = ConnectionPool(
    max_connections=20,
    database="user_db"
)

# Payment DB overload doesn't affect user DB queries

Resource Partitioning:

Service A: 30% CPU, 2GB RAM
Service B: 30% CPU, 2GB RAM
Service C: 40% CPU, 4GB RAM

If Service A fails or overloads:
- Service B and C continue operating
- Resources isolated per service

Kubernetes Resource Limits:

apiVersion: v1
kind: Pod
spec:
  containers:
  - name: payment-service
    resources:
      requests:
        cpu: "500m"
        memory: "512Mi"
      limits:
        cpu: "1000m"
        memory: "1Gi"
  - name: notification-service
    resources:
      requests:
        cpu: "200m"
        memory: "256Mi"
      limits:
        cpu: "500m"
        memory: "512Mi"

Each service has isolated resources

Bulkhead Benefits Example:

Scenario: Payment service is slow (5s response time)

Without Bulkhead:
- All thread pool threads blocked
- User service can't process requests
- Notification service can't send emails
- System-wide degradation

With Bulkhead:
- Payment service threads blocked (isolated)
- User service continues normally
- Notification service continues normally
- Only payment service affected`,
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
					CodeExamples: `Retry Pattern with Exponential Backoff:

def retry_with_backoff(func, max_retries=3, initial_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            delay = initial_delay * (2 ** attempt)  # Exponential backoff
            time.sleep(delay)
    
    raise Exception("Max retries exceeded")

Usage:
result = retry_with_backoff(
    lambda: call_external_api(),
    max_retries=3,
    initial_delay=1
)

Retry attempts: 1s, 2s, 4s delays

Timeout Pattern:

import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def call_with_timeout(func, timeout_seconds=5):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func()
        signal.alarm(0)  # Cancel alarm
        return result
    except TimeoutError:
        raise TimeoutError(f"Operation exceeded {timeout_seconds}s")

Saga Pattern Example:

Order Saga:
1. Create Order (Order Service)
2. Reserve Inventory (Inventory Service)
3. Process Payment (Payment Service)
4. Ship Order (Shipping Service)

If Payment fails:
- Compensate: Release Inventory
- Compensate: Cancel Order

def create_order(order_data):
    order = order_service.create(order_data)
    try:
        inventory_service.reserve(order.items)
        payment_service.charge(order.total)
        shipping_service.ship(order)
    except PaymentError:
        inventory_service.release(order.items)  # Compensate
        order_service.cancel(order.id)  # Compensate
        raise

CQRS Pattern:

Write Model (Commands):
POST /api/users
{
  "name": "John",
  "email": "john@example.com"
}
→ Write to normalized database
→ Optimized for consistency

Read Model (Queries):
GET /api/users/123
→ Read from denormalized view
→ Optimized for read performance
→ Can be cached, replicated

Write Side:
- Normalized schema
- Strong consistency
- ACID transactions

Read Side:
- Denormalized views
- Eventual consistency
- Optimized queries
- Can use different DB (NoSQL)

Event Sourcing Example:

Instead of storing current state:
User: {id: 123, balance: 100}

Store events:
1. UserCreated {id: 123, initial_balance: 0}
2. Deposit {amount: 50, balance: 50}
3. Deposit {amount: 50, balance: 100}

Rebuild state:
balance = 0 + 50 + 50 = 100

Benefits:
- Complete audit trail
- Time travel (state at any point)
- Replay events for debugging

Event Store:
{
  "aggregate_id": "user_123",
  "event_type": "Deposit",
  "event_data": {"amount": 50},
  "timestamp": "2024-01-17T10:00:00Z",
  "version": 3
}

Rebuild State:
events = event_store.get_events("user_123")
state = apply_events(events)  # Rebuild from events`,
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
					CodeExamples: `URL Shortener API Design:

POST /api/v1/shorten
Request:
{
  "url": "https://example.com/very/long/url/path",
  "custom_code": "mylink"  # Optional
}

Response:
{
  "short_url": "https://short.ly/abc1234",
  "original_url": "https://example.com/very/long/url/path",
  "expires_at": "2025-01-17T10:00:00Z"
}

GET /abc1234
Response: 301 Redirect
Location: https://example.com/very/long/url/path

Base62 Encoding:

def base62_encode(num):
    chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = []
    while num > 0:
        result.append(chars[num % 62])
        num //= 62
    return ''.join(reversed(result))

# Counter: 1234567890
# Encoded: "1LY7VK"
# Short URL: https://short.ly/1LY7VK

Database Schema:

CREATE TABLE urls (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    short_code VARCHAR(7) UNIQUE NOT NULL,
    original_url TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    user_id BIGINT,
    click_count BIGINT DEFAULT 0,
    INDEX idx_short_code (short_code),
    INDEX idx_expires_at (expires_at)
);

Architecture:

Client → Load Balancer → API Servers
                              |
                              v
                    [URL Service] → [Database Shards]
                              |
                              v
                         [Redis Cache]
                              |
                              v
                    [Analytics Service]

Shortening Flow:
1. Generate unique ID (counter or UUID)
2. Encode to Base62 (7 characters)
3. Check for collision (rare)
4. Store in database
5. Cache in Redis
6. Return short URL

Redirect Flow:
1. Extract short code from URL
2. Check Redis cache (hot path)
3. If miss, query database
4. Update click count
5. Return 301 redirect

Sharding Strategy:
- Shard by short_code hash
- 10 shards: shard_id = hash(short_code) % 10
- Each shard handles subset of URLs

Capacity Estimation:
- 100M URLs/day = ~1,200 writes/sec
- 100:1 read/write ratio = 120,000 reads/sec
- Storage: 100M × 500 bytes = 50 GB/year
- Cache: 20% hot data = 10 GB`,
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
					CodeExamples: `Chat System Architecture:

Client (WebSocket) → Load Balancer → WebSocket Servers
                                              |
                                              v
                                    [Message Queue (Kafka)]
                                              |
                    +-------------------------+-------------------------+
                    |                         |                         |
                    v                         v                         v
            [Message Service]        [Presence Service]        [Notification Service]
                    |                         |                         |
                    v                         v                         v
            [Database]                [Redis Cache]            [Push Service]

WebSocket Connection:

Client connects:
ws://chat.example.com/ws?user_id=123&token=abc

Server maintains connection:
- Heartbeat every 30 seconds
- Reconnect on disconnect
- Load balancer sticky sessions

Message Flow:

1. User A sends message to User B
2. Message → WebSocket Server A
3. WebSocket Server A → Kafka Topic "messages"
4. Kafka → WebSocket Server B (if User B online)
5. WebSocket Server B → User B (real-time)
6. Kafka → Message Service (persist)
7. Message Service → Database

Message Format:

{
  "message_id": "msg_123",
  "from_user_id": 123,
  "to_user_id": 456,
  "content": "Hello!",
  "timestamp": "2024-01-17T10:00:00Z",
  "type": "text",  # text, image, file
  "room_id": "room_789"  # For group chats
}

Database Schema:

CREATE TABLE messages (
    id BIGINT PRIMARY KEY,
    from_user_id BIGINT NOT NULL,
    to_user_id BIGINT,  # NULL for group chats
    room_id BIGINT,
    content TEXT NOT NULL,
    message_type VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_to_user (to_user_id, created_at),
    INDEX idx_room (room_id, created_at)
);

CREATE TABLE rooms (
    id BIGINT PRIMARY KEY,
    name VARCHAR(255),
    type VARCHAR(20),  # direct, group
    created_at TIMESTAMP
);

Presence Service:

Redis stores online users:
SET user:123:online true EX 60
SET user:123:server "ws-server-1"

Check if user online:
GET user:123:online

Group Chat:

1. User sends message to room
2. Message → Kafka Topic "room_messages"
3. Kafka → All online members in room
4. Each member receives via WebSocket
5. Message persisted to database

Scaling WebSocket Servers:

Challenge: Sticky sessions needed
Solution: Use consistent hashing
- User ID → Hash → WebSocket Server
- Same user always connects to same server
- Server maintains user's connection

Message Ordering:

- Use sequence numbers per conversation
- Client sends: {seq: 1, content: "Hello"}
- Server assigns: {seq: 2, content: "Hi"}
- Deliver in sequence order`,
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
					CodeExamples: `Social Media Feed API:

GET /api/v1/feed?user_id=123&limit=20
Response:
{
  "posts": [
    {
      "post_id": 456,
      "user_id": 789,
      "content": "Hello world!",
      "created_at": "2024-01-17T10:00:00Z",
      "likes": 42,
      "comments": 5
    }
  ],
  "next_cursor": "abc123"
}

Pull Model (Fan-out on Read):

def get_feed(user_id):
    # 1. Get list of followed users
    followed_users = get_followed_users(user_id)
    
    # 2. Fetch recent posts from each followed user
    posts = []
    for followed_user_id in followed_users:
        user_posts = get_recent_posts(followed_user_id, limit=10)
        posts.extend(user_posts)
    
    # 3. Merge and sort by timestamp
    posts.sort(key=lambda x: x['created_at'], reverse=True)
    
    # 4. Return top N posts
    return posts[:20]

Pros: Simple, no write amplification
Cons: Slow for users following many people
Query: O(follows) database queries

Push Model (Fan-out on Write):

def create_post(user_id, content):
    # 1. Create post
    post = create_post_in_db(user_id, content)
    
    # 2. Get all followers
    followers = get_followers(user_id)
    
    # 3. Push to each follower's feed
    for follower_id in followers:
        add_to_feed(follower_id, post)
    
    return post

def get_feed(user_id):
    # Simply read pre-computed feed
    return get_user_feed(user_id)

Pros: Fast reads (O(1) query)
Cons: Slow writes for users with many followers
Storage: O(followers) storage per post

Database Schema:

CREATE TABLE posts (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    content TEXT,
    created_at TIMESTAMP,
    INDEX idx_user_created (user_id, created_at)
);

CREATE TABLE feeds (
    user_id BIGINT,
    post_id BIGINT,
    created_at TIMESTAMP,
    PRIMARY KEY (user_id, post_id),
    INDEX idx_user_created (user_id, created_at)
);

Hybrid Approach:

def create_post(user_id, content):
    post = create_post_in_db(user_id, content)
    followers = get_followers(user_id)
    
    # Push to active users (logged in last 30 days)
    active_followers = filter_active_users(followers)
    for follower_id in active_followers:
        add_to_feed(follower_id, post)
    
    # Inactive users: pull on-demand
    return post

def get_feed(user_id):
    if is_active_user(user_id):
        # Read pre-computed feed
        return get_user_feed(user_id)
    else:
        # Pull model for inactive users
        return pull_feed(user_id)

Benefits:
- Active users: Fast reads (push model)
- Inactive users: No storage waste (pull model)
- Celebrity problem: Use pull for high-follower accounts

Architecture:

[Post Service] → [Feed Service]
                      |
                      +→ [Active User Feeds] (Redis/DB)
                      |
                      +→ [Inactive Users] (Pull on-demand)

Scaling Considerations:

- Cache hot feeds in Redis
- Shard feeds by user_id
- Use message queue for async fan-out
- Batch writes for high-follower accounts

Capacity:
- 1B users, 500M posts/day
- Average 200 followers per user
- Push model: 500M × 200 = 100B feed writes/day
- Storage: 100B × 500 bytes = 50 TB/day`,
				},
				{
					Title: "Design a URL Shortener - Interview Walkthrough",
					Content: `**Complete Interview Walkthrough Using RESHADED Framework**

**Problem Statement:**
Design a URL shortener like TinyURL that can shorten long URLs and redirect users to the original URL when they click the short link.

**R - Requirements Clarification:**

**Questions to Ask:**
1. "What's the expected scale? How many URLs per day?"
2. "What's the read/write ratio?"
3. "Do we need custom short URLs or just auto-generated?"
4. "Do URLs expire or are they permanent?"
5. "Do we need analytics (click tracking)?"
6. "What's the acceptable latency for redirects?"

**Assumptions (After Clarification):**
- 100M URLs/day
- 100:1 read/write ratio (read-heavy)
- Auto-generated 7-character short URLs
- URLs don't expire (or very long expiration)
- Analytics not required initially
- < 100ms latency for redirects
- 99.9% availability

**E - Estimate (Capacity Planning):**

**Traffic Estimation:**
- Writes: 100M URLs/day = 100M / 86400 = ~1,200 writes/sec average
- Peak writes: 10x average = ~12,000 writes/sec
- Reads: 1,200 × 100 = 120,000 reads/sec average
- Peak reads: 12,000 × 100 = 1,200,000 reads/sec

**Storage Estimation:**
- Each URL record: ~500 bytes (short_code, original_url, timestamps, metadata)
- 100M URLs/day × 500 bytes = 50 GB/day
- 5 years retention: 50 GB × 365 × 5 = ~91 TB
- With indexes (2x): ~182 TB

**Bandwidth Estimation:**
- Write: 12,000 writes/sec × 500 bytes = 6 MB/s = 48 Mbps
- Read: 1,200,000 reads/sec × 100 bytes (redirect response) = 120 MB/s = 960 Mbps

**S - System Interface (API Design):**

**Endpoints:**

POST /api/v1/shorten
Request:
{
  "url": "https://example.com/very/long/url/path",
  "custom_code": "mylink"  // Optional
}

Response:
{
  "short_url": "https://short.ly/abc1234",
  "original_url": "https://example.com/very/long/url/path",
  "expires_at": null
}

GET /{shortCode}
Response: 301 Redirect
Location: https://example.com/very/long/url/path

**H - High-Level Design:**

**Architecture:**
    Client → Load Balancer → API Servers → [Cache Layer] → Database
                                  ↓
                            [ID Generator Service]

**Components:**
1. **Load Balancer**: Distributes traffic across API servers
2. **API Servers**: Stateless servers handling requests
3. **ID Generator**: Generates unique IDs (counter-based or distributed)
4. **Encoding Service**: Converts ID to base62 short code
5. **Cache Layer**: Redis for hot URLs (80% cache hit rate target)
6. **Database**: Stores URL mappings (sharded for scale)

**A - Algorithm/Data Structure:**

**Short URL Generation:**
- **Option 1: Counter-based**
  - Use distributed counter (Zookeeper, etcd)
  - Encode counter value to base62
  - Pros: Short URLs, sequential
  - Cons: Single point of failure, needs coordination

- **Option 2: Hash-based**
  - Hash original URL (MD5/SHA256)
  - Take first 7 characters
  - Handle collisions (append counter)
  - Pros: No coordination needed
  - Cons: Longer URLs, collision handling

- **Option 3: Distributed ID (Recommended)**
  - Use Snowflake algorithm or similar
  - Generates unique 64-bit IDs
  - Encode to base62 (7 characters)
  - Pros: Distributed, no collisions, scalable
  - Cons: Slightly more complex

**Base62 Encoding:**
- Characters: 0-9, a-z, A-Z (62 characters)
- 7 characters = 62^7 = ~3.5 trillion possible URLs
- Algorithm: Repeatedly divide by 62, map remainder to character

**D - Detailed Design:**

**Database Schema:**

CREATE TABLE urls (
    id BIGINT PRIMARY KEY,
    short_code VARCHAR(7) UNIQUE NOT NULL,
    original_url TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NULL,
    user_id BIGINT NULL,
    click_count BIGINT DEFAULT 0,
    INDEX idx_short_code (short_code),
    INDEX idx_expires_at (expires_at)
);

**Shortening Flow:**
1. Client sends POST /api/v1/shorten with long URL
2. API server receives request
3. Generate unique ID (distributed ID generator)
4. Encode ID to base62 (7 characters)
5. Check for collision (very rare, but handle it)
6. Store mapping in database: short_code → original_url
7. Cache in Redis (optional, for frequently shortened URLs)
8. Return short URL

**Redirect Flow:**
1. Client requests GET /abc1234
2. Load balancer routes to API server
3. Check Redis cache first (hot path)
4. If cache hit: Return 301 redirect immediately
5. If cache miss: Query database
6. If found: Cache in Redis, return 301 redirect
7. If not found: Return 404

**E - Evaluation (Bottlenecks & Optimizations):**

**Potential Bottlenecks:**
1. **Database writes**: 12,000 writes/sec
   - Solution: Database can handle this, but may need sharding later
   
2. **Database reads**: 1,200,000 reads/sec
   - Solution: Read replicas (10-20 replicas), Redis cache
   - Cache hit rate 80%: 960K reads from cache, 240K from DB
   - Each replica handles ~24K reads/sec (manageable)

3. **Cache**: 960K reads/sec from cache
   - Solution: Redis cluster, consistent hashing
   - Multiple Redis instances

4. **ID Generation**: Single point of failure
   - Solution: Distributed ID generator (Snowflake), multiple instances

**Optimizations:**
- **Caching**: Cache popular URLs (80% hit rate reduces DB load by 80%)
- **CDN**: Use CDN for redirects (static redirects can be cached)
- **Database Sharding**: Shard by short_code hash when needed
- **Connection Pooling**: Reuse database connections
- **Async Processing**: Analytics can be async (if added later)

**D - Discussion (Trade-offs, Scaling, Reliability):**

**Trade-offs:**

1. **SQL vs NoSQL:**
   - SQL: ACID guarantees, easier queries, but harder to scale
   - NoSQL: Better horizontal scaling, but eventual consistency
   - **Choice**: Start with SQL, migrate to NoSQL if needed

2. **Cache Strategy:**
   - Write-through: Always consistent, slower writes
   - Cache-aside: Faster, but potential stale data (acceptable for URLs)
   - **Choice**: Cache-aside (URLs rarely change)

3. **Short URL Length:**
   - 6 chars: 62^6 = ~56 billion (may not be enough long-term)
   - 7 chars: 62^7 = ~3.5 trillion (sufficient)
   - **Choice**: 7 characters

**Scaling Strategy:**

**Phase 1 (Current):**
- Single database, read replicas
- Redis cache
- Multiple API servers

**Phase 2 (10x scale):**
- Database sharding by short_code hash
- Redis cluster
- CDN for redirects

**Phase 3 (100x scale):**
- Multiple database clusters
- Global CDN
- Separate analytics database

**Reliability:**
- Database replication (master + replicas)
- Redis replication
- Multiple API servers (no single point of failure)
- Health checks and auto-failover
- Data backups

**Common Follow-up Questions:**

1. "How would you handle custom short URLs?"
   - Check if custom_code exists in database
   - If exists, suggest alternatives
   - Store with user_id for ownership

2. "What if we need analytics?"
   - Separate analytics database
   - Async logging to message queue (Kafka)
   - Aggregate in batch jobs
   - Don't block redirect path

3. "How would you scale to 10x?"
   - Database sharding
   - More read replicas
   - Redis cluster
   - CDN for redirects

4. "How do you ensure URL uniqueness?"
   - Database unique constraint on short_code
   - Handle collision in application (retry with new ID)
   - Very rare with distributed ID generator

5. "What about URL expiration?"
   - Add expires_at field
   - Background job to clean expired URLs
   - Check expiration on redirect (or pre-filter in cache)`,
					CodeExamples: `Base62 Encoding Implementation:

def base62_encode(num):
    chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if num == 0:
        return chars[0]
    
    result = []
    base = len(chars)
    while num > 0:
        result.append(chars[num % base])
        num //= base
    
    return ''.join(reversed(result)).zfill(7)  # Pad to 7 chars

# Example: 1234567890 → "1LY7VK"

Distributed ID Generator (Snowflake-like):

class IDGenerator:
    def __init__(self, datacenter_id, worker_id):
        self.datacenter_id = datacenter_id  # 5 bits
        self.worker_id = worker_id          # 5 bits
        self.sequence = 0                    # 12 bits
        self.last_timestamp = -1
        self.epoch = 1609459200000  # Custom epoch
    
    def generate_id(self):
        timestamp = int(time.time() * 1000)
        
        if timestamp < self.last_timestamp:
            raise Exception("Clock moved backwards")
        
        if timestamp == self.last_timestamp:
            self.sequence = (self.sequence + 1) & 0xFFF
            if self.sequence == 0:
                timestamp = self._wait_next_millis(self.last_timestamp)
        else:
            self.sequence = 0
        
        self.last_timestamp = timestamp
        
        # 64-bit ID: timestamp (41 bits) + datacenter (5) + worker (5) + sequence (12)
        id = ((timestamp - self.epoch) << 22) | (self.datacenter_id << 17) | (self.worker_id << 12) | self.sequence
        return id

Shortening API Implementation:

def shorten_url(long_url):
    # 1. Generate unique ID
    unique_id = id_generator.generate_id()
    
    # 2. Encode to base62
    short_code = base62_encode(unique_id)
    
    # 3. Check collision (very rare)
    if db.url_exists(short_code):
        # Retry with new ID
        return shorten_url(long_url)
    
    # 4. Store in database
    db.insert_url(short_code, long_url)
    
    # 5. Return short URL
    return f"https://short.ly/{short_code}"

Redirect API Implementation:

def redirect(short_code):
    # 1. Check cache first
    original_url = cache.get(f"url:{short_code}")
    if original_url:
        return redirect_response(original_url)
    
    # 2. Query database
    original_url = db.get_url(short_code)
    if not original_url:
        return 404_response()
    
    # 3. Cache for future requests
    cache.set(f"url:{short_code}", original_url, ttl=3600)
    
    # 4. Return redirect
    return redirect_response(original_url)

Database Sharding:

def get_shard(short_code):
    # Hash short_code to determine shard
    hash_value = hash(short_code)
    shard_id = hash_value % NUM_SHARDS
    return f"shard_{shard_id}"

# Shard routing
def get_url_from_shard(short_code):
    shard = get_shard(short_code)
    return db_clusters[shard].get_url(short_code)

Architecture Diagram:

[Client]
    |
    v
[Load Balancer] (Round-robin or consistent hashing)
    |
    +---> [API Server 1] ---+
    |                       |
    +---> [API Server 2] ---+---> [Redis Cache Cluster]
    |                       |         |
    +---> [API Server N] ---+         |
                              |         |
                              v         v
                    [ID Generator]  [Database Cluster]
                    (Snowflake)         |
                                        +---> [Master DB]
                                        |
                                        +---> [Replica 1]
                                        |
                                        +---> [Replica N]

Capacity Calculation Summary:

Writes: 12,000/sec
- Single DB can handle ~10K writes/sec
- May need sharding at 10x scale

Reads: 1,200,000/sec
- Cache (80% hit): 960K/sec from Redis
- DB (20% miss): 240K/sec from DB
- 10 replicas: 24K reads/sec per replica (manageable)

Storage: 182 TB for 5 years
- Can partition by date
- Archive old data

Cache Size:
- 20% of URLs are hot (Pareto principle)
- 20M hot URLs × 500 bytes = 10 GB
- Redis cluster with 3 nodes, 8GB each`,
				},
				{
					Title: "Design a Chat System - Interview Walkthrough",
					Content: `**Complete Interview Walkthrough Using RESHADED Framework**

**Problem Statement:**
Design a real-time chat system like WhatsApp that supports one-on-one messaging, group chats, message delivery status, and online/offline presence.

**R - Requirements Clarification:**

**Questions to Ask:**
1. "What's the scale? How many users and messages per day?"
2. "Do we need to support group chats? What's the max group size?"
3. "Do we need message history? How long to retain?"
4. "Do we need read receipts and delivery status?"
5. "Do we need to support media (images, videos, files)?"
6. "Do we need end-to-end encryption?"
7. "What's the acceptable message delivery latency?"

**Assumptions (After Clarification):**
- 2B users, 100B messages/day
- Support group chats (max 256 members)
- Message history: 1 year retention
- Delivery status: Sent, Delivered, Read
- Support media (images, videos up to 100MB)
- End-to-end encryption not required initially
- < 100ms latency for message delivery
- 99.9% availability

**E - Estimate (Capacity Planning):**

**Traffic Estimation:**
- Messages: 100B/day = ~1.16M messages/sec average
- Peak: 10x average = ~11.6M messages/sec
- Group messages: Assume 20% are group messages
- Group message fan-out: Average 10 members per group
- Effective writes: 1.16M × (0.8 + 0.2 × 10) = ~3.5M writes/sec

**Storage Estimation:**
- Text message: ~100 bytes (message_id, from, to, content, timestamp)
- Media message: ~10 KB metadata + file storage
- Assume 80% text, 20% media
- Text: 100B × 0.8 × 100 bytes = 8 TB/day
- Media metadata: 100B × 0.2 × 10 KB = 200 TB/day
- Media files: 100B × 0.2 × 1 MB average = 20 PB/day
- 1 year retention: 8 TB × 365 = 2.9 PB (text), 73 PB (media metadata), 7.3 EB (media files)

**Bandwidth Estimation:**
- Outbound: 11.6M messages/sec × 100 bytes = 1.16 GB/s = 9.3 Gbps
- Inbound: Similar (bidirectional)
- Media: Much higher (use CDN)

**S - System Interface (API Design):**

**WebSocket Endpoints:**
ws://chat.example.com/ws?user_id=123&token=abc

**Message Types:**

Send Message:
{
  "type": "send_message",
  "to_user_id": 456,
  "content": "Hello!",
  "message_type": "text"
}

Receive Message:
{
  "type": "message",
  "message_id": "msg_123",
  "from_user_id": 456,
  "content": "Hi there!",
  "timestamp": "2024-01-17T10:00:00Z",
  "status": "sent"
}

Delivery Status:
{
  "type": "status_update",
  "message_id": "msg_123",
  "status": "delivered"  // or "read"
}

**REST API (for history, media):**

GET /api/v1/messages?user_id=123&conversation_id=456&limit=50&cursor=abc
POST /api/v1/media/upload
GET /api/v1/media/{media_id}

**H - High-Level Design:**

**Architecture:**
    Mobile App → WebSocket Servers → Message Queue (Kafka) → [Message Service, Presence Service, Notification Service]
                                          ↓
                                     [Database Cluster]
                                          ↓
                                     [Media Storage (S3)]

**Components:**
1. **WebSocket Servers**: Maintain persistent connections, handle real-time messaging
2. **Message Queue (Kafka)**: Reliable message delivery, fan-out for group chats
3. **Message Service**: Persist messages, handle history
4. **Presence Service**: Track online/offline status
5. **Notification Service**: Push notifications for offline users
6. **Media Service**: Handle media uploads and storage
7. **Database**: Store messages, user data, group info
8. **Object Storage**: Store media files (S3-like)

**A - Algorithm/Data Structure:**

**Message Ordering:**
- Use sequence numbers per conversation
- Client sends sequence number with message
- Server assigns sequence numbers
- Deliver messages in sequence order

**Presence Tracking:**
- Redis: user_id → {status: "online", server: "ws-server-1", last_seen: timestamp}
- Heartbeat every 30 seconds
- Mark offline after 60 seconds of no heartbeat

**Group Chat Fan-out:**
- When user sends group message:
  1. Store message once
  2. Get all group members
  3. Fan-out to online members via WebSocket
  4. Queue notifications for offline members
  5. Update delivery status per member

**D - Detailed Design:**

**Database Schema:**

CREATE TABLE messages (
    id BIGINT PRIMARY KEY,
    from_user_id BIGINT NOT NULL,
    to_user_id BIGINT,  -- NULL for group messages
    room_id BIGINT,     -- For group chats
    content TEXT,
    message_type VARCHAR(20),  -- text, image, video, file
    media_url VARCHAR(500),
    sequence_number BIGINT,
    created_at TIMESTAMP,
    INDEX idx_conversation (from_user_id, to_user_id, created_at),
    INDEX idx_room (room_id, created_at)
);

CREATE TABLE rooms (
    id BIGINT PRIMARY KEY,
    name VARCHAR(255),
    type VARCHAR(20),  -- direct, group
    created_at TIMESTAMP
);

CREATE TABLE room_members (
    room_id BIGINT,
    user_id BIGINT,
    joined_at TIMESTAMP,
    PRIMARY KEY (room_id, user_id)
);

CREATE TABLE message_status (
    message_id BIGINT,
    user_id BIGINT,
    status VARCHAR(20),  -- sent, delivered, read
    updated_at TIMESTAMP,
    PRIMARY KEY (message_id, user_id)
);

**Message Flow (One-on-One):**

1. User A sends message to User B
2. Mobile app → WebSocket Server A
3. WebSocket Server A → Kafka Topic "messages"
4. Kafka → Message Service (persist to DB)
5. Kafka → WebSocket Server B (if User B online)
6. WebSocket Server B → User B's mobile app (real-time)
7. If User B offline: Kafka → Notification Service → Push notification

**Message Flow (Group Chat):**

1. User sends message to group
2. WebSocket Server → Kafka Topic "group_messages"
3. Kafka → Message Service (store once)
4. Kafka → Fan-out Service:
   - Get all group members
   - For each online member: Send via their WebSocket server
   - For each offline member: Queue notification
5. Update delivery status per member

**Presence Service:**

Redis Structure:
- Key: "presence:user:{user_id}"
- Value: JSON {status: "online", server: "ws-1", last_seen: timestamp}
- TTL: 60 seconds (auto-expire if no heartbeat)

**E - Evaluation (Bottlenecks & Optimizations):**

**Potential Bottlenecks:**

1. **WebSocket Connections**: 2B users, but not all online simultaneously
   - Assume 10% online = 200M concurrent connections
   - Each WebSocket server: ~100K connections
   - Need: 2000 WebSocket servers
   - Solution: Horizontal scaling, load balancing with sticky sessions

2. **Message Fan-out**: Group messages create write amplification
   - 1 message → N writes (for N group members)
   - Large groups (256 members) = 256x amplification
   - Solution: Async processing, batch writes, separate storage for group messages

3. **Database Writes**: 3.5M writes/sec
   - Solution: Database sharding by user_id or room_id
   - Write to multiple shards in parallel
   - Use message queue to buffer writes

4. **Media Storage**: 20 PB/day
   - Solution: Object storage (S3), CDN for delivery
   - Compress media, use different quality tiers
   - Archive old media to cold storage

5. **Message History Queries**: Slow for long conversations
   - Solution: Pagination, cursor-based pagination
   - Cache recent messages
   - Archive old messages

**Optimizations:**
- **Message Batching**: Batch multiple messages in single WebSocket frame
- **Read Replicas**: Use read replicas for message history queries
- **Caching**: Cache recent messages, user presence, group members
- **CDN**: Use CDN for media delivery
- **Compression**: Compress WebSocket messages
- **Connection Pooling**: Reuse database connections

**D - Discussion (Trade-offs, Scaling, Reliability):**

**Trade-offs:**

1. **Push vs Pull for Group Messages:**
   - Push (fan-out on write): Fast reads, but high write amplification
   - Pull (fan-out on read): Lower writes, but slower reads
   - **Choice**: Hybrid - push for small groups (<50), pull for large groups

2. **Message Storage:**
   - Store per user: Fast queries, but high storage (duplicate messages)
   - Store once, index by users: Lower storage, but complex queries
   - **Choice**: Store once, index by conversation/room

3. **Consistency:**
   - Strong consistency: Slower, but guaranteed order
   - Eventual consistency: Faster, but may have ordering issues
   - **Choice**: Strong consistency for message ordering (critical)

4. **WebSocket Server Affinity:**
   - Sticky sessions: Same user always on same server (simpler)
   - No affinity: More flexible, but need message routing
   - **Choice**: Sticky sessions with consistent hashing

**Scaling Strategy:**

**Phase 1 (Current):**
- WebSocket servers with sticky sessions
- Single Kafka cluster
- Database with read replicas

**Phase 2 (10x scale):**
- WebSocket server clusters per region
- Kafka cluster with more partitions
- Database sharding by user_id
- Redis cluster for presence

**Phase 3 (100x scale):**
- Global WebSocket server network
- Multiple Kafka clusters (regional)
- Multi-region database replication
- CDN for media globally

**Reliability:**
- WebSocket reconnection logic
- Message queue ensures delivery (at-least-once)
- Database replication
- Health checks and auto-failover
- Message deduplication (idempotent operations)

**Common Follow-up Questions:**

1. "How do you ensure message ordering?"
   - Sequence numbers per conversation
   - Server assigns sequence numbers
   - Client buffers out-of-order messages
   - Database queries ordered by sequence_number

2. "What about offline message delivery?"
   - Store messages in database
   - When user comes online, fetch unread messages
   - Push notifications for important messages
   - Sync on reconnection

3. "How do you handle large group chats (1000+ members)?"
   - Use pull model instead of push
   - Cache group messages
   - Paginate message history
   - Consider separate storage for large groups

4. "How would you implement read receipts?"
   - Store read status in message_status table
   - Update when user views conversation
   - Broadcast status update to sender
   - Use WebSocket for real-time updates

5. "What about media compression?"
   - Compress images (JPEG quality, WebP format)
   - Transcode videos (multiple quality tiers)
   - Progressive loading (thumbnails first)
   - CDN caching`,
					CodeExamples: `WebSocket Message Handler:

class WebSocketHandler:
    def on_message(self, ws, message):
        data = json.loads(message)
        
        if data['type'] == 'send_message':
            self.handle_send_message(ws, data)
        elif data['type'] == 'status_update':
            self.handle_status_update(ws, data)
    
    def handle_send_message(self, ws, data):
        # Generate message ID
        message_id = generate_id()
        
        # Assign sequence number
        sequence = self.get_next_sequence(data['conversation_id'])
        
        # Create message object
        message = {
            'message_id': message_id,
            'from_user_id': ws.user_id,
            'to_user_id': data['to_user_id'],
            'content': data['content'],
            'sequence_number': sequence,
            'timestamp': time.now()
        }
        
        # Send to Kafka
        kafka.produce('messages', message)
        
        # If recipient is online on this server, send immediately
        if self.is_user_online(data['to_user_id']):
            self.send_to_user(data['to_user_id'], message)
        
        # Acknowledge to sender
        ws.send(json.dumps({
            'type': 'message_sent',
            'message_id': message_id
        }))

Group Message Fan-out:

def fan_out_group_message(message, room_id):
    # Get all group members
    members = db.get_room_members(room_id)
    
    # Store message once
    db.insert_message(message, room_id=room_id)
    
    # Fan-out to online members
    online_members = []
    for member_id in members:
        if presence_service.is_online(member_id):
            ws_server = presence_service.get_server(member_id)
            send_via_websocket(ws_server, member_id, message)
            online_members.append(member_id)
        else:
            # Queue notification
            notification_service.queue_notification(member_id, message)
    
    # Update delivery status
    for member_id in online_members:
        db.update_message_status(message['id'], member_id, 'delivered')

Presence Service:

class PresenceService:
    def mark_online(self, user_id, server_id):
        key = f"presence:user:{user_id}"
        value = {
            'status': 'online',
            'server': server_id,
            'last_seen': time.now()
        }
        redis.setex(key, 60, json.dumps(value))
    
    def mark_offline(self, user_id):
        key = f"presence:user:{user_id}"
        redis.delete(key)
    
    def is_online(self, user_id):
        key = f"presence:user:{user_id}"
        return redis.exists(key)
    
    def get_server(self, user_id):
        key = f"presence:user:{user_id}"
        data = json.loads(redis.get(key))
        return data['server']

Message History API:

def get_message_history(user_id, conversation_id, limit=50, cursor=None):
    # Query messages for this conversation
    query = """
        SELECT * FROM messages
        WHERE (from_user_id = ? AND to_user_id = ?)
           OR (from_user_id = ? AND to_user_id = ?)
        ORDER BY sequence_number DESC
        LIMIT ?
    """
    
    if cursor:
        query += " AND sequence_number < ?"
        params = (user_id, conversation_id, conversation_id, user_id, limit, cursor)
    else:
        params = (user_id, conversation_id, conversation_id, user_id, limit)
    
    messages = db.execute(query, params)
    return {
        'messages': messages,
        'next_cursor': messages[-1]['sequence_number'] if messages else None
    }

Architecture Diagram:

[Mobile App] ←→ [WebSocket Servers] (2000 servers, 100K connections each)
                        |
                        v
                [Load Balancer] (Sticky sessions)
                        |
                        v
                [Kafka Cluster] (Message queue)
                        |
        +---------------+---------------+
        |               |               |
        v               v               v
[Message Service] [Presence Service] [Notification Service]
        |               |               |
        v               v               v
[Database Cluster] [Redis Cluster] [Push Service]
(Sharded)        (Presence)        (FCM/APNS)
        |
        v
[Media Storage] (S3-like)
        |
        v
[CDN] (Media delivery)

Capacity Summary:

WebSocket Connections: 200M concurrent
- 2000 servers × 100K connections each

Messages: 11.6M/sec
- Kafka: 11.6M messages/sec (with fan-out: 3.5M effective writes/sec)
- Database: 3.5M writes/sec (sharded across 100 shards = 35K writes/sec per shard)

Storage: 7.3 EB/year (mostly media)
- Text: 2.9 PB/year
- Media: 7.3 EB/year (use compression, CDN, archiving)

Presence: 200M online users
- Redis: 200M keys × 100 bytes = 20 GB (manageable)`,
				},
				{
					Title: "Design a Social Media Feed - Interview Walkthrough",
					Content: `**Complete Interview Walkthrough Using RESHADED Framework**

**Problem Statement:**
Design a social media feed system like Twitter/X that shows users a timeline of posts from people they follow, with support for real-time updates and trending content.

**R - Requirements Clarification:**

**Questions to Ask:**
1. "What's the scale? How many users and posts per day?"
2. "What's the average number of followers per user?"
3. "Do we need to support both user timelines and trending/explore feeds?"
4. "Do we need real-time updates or is eventual consistency OK?"
5. "What's the acceptable feed load latency?"
6. "Do we need to support media (images, videos)?"
7. "Do we need personalized ranking or just chronological?"

**Assumptions (After Clarification):**
- 300M users, 500M posts/day
- Average 200 followers per user
- Support both user timelines and trending feeds
- Real-time updates preferred (< 1 second delay)
- < 200ms latency for feed load
- Support media (images, videos)
- Chronological feed initially, can add ranking later

**E - Estimate (Capacity Planning):**

**Traffic Estimation:**
- Posts: 500M/day = ~5,800 posts/sec average
- Peak: 10x average = ~58,000 posts/sec
- Feed reads: 300M users × 10 reads/day = 3B reads/day = ~35,000 reads/sec average
- Peak reads: 10x average = ~350,000 reads/sec

**Storage Estimation:**
- Post: ~500 bytes (post_id, user_id, content, timestamp, media_urls)
- With fan-out: 500M posts × 200 followers = 100B feed entries/day
- Storage: 100B × 500 bytes = 50 TB/day
- 1 year retention: 50 TB × 365 = 18.25 PB

**Bandwidth Estimation:**
- Write: 58,000 posts/sec × 500 bytes = 29 MB/s = 232 Mbps
- Read: 350,000 reads/sec × 10 KB (feed response) = 3.5 GB/s = 28 Gbps

**S - System Interface (API Design):**

**Endpoints:**

POST /api/v1/posts
Request:
{
  "content": "Hello world!",
  "media_urls": ["https://..."],
  "reply_to_post_id": 123  // Optional
}

Response:
{
  "post_id": 456,
  "user_id": 789,
  "content": "Hello world!",
  "created_at": "2024-01-17T10:00:00Z"
}

GET /api/v1/feed?user_id=123&limit=20&cursor=abc
Response:
{
  "posts": [
    {
      "post_id": 456,
      "user_id": 789,
      "content": "Hello world!",
      "created_at": "2024-01-17T10:00:00Z",
      "likes": 42,
      "replies": 5
    }
  ],
  "next_cursor": "def"
}

GET /api/v1/trending?limit=20
Response: Similar structure

**H - High-Level Design:**

**Architecture:**
    User → API Gateway → [Post Service, Feed Service, Timeline Service]
                              ↓
                        [Message Queue (Kafka)]
                              ↓
                  [Fan-out Service, Ranking Service]
                              ↓
                        [Database Cluster, Cache]

**Components:**
1. **Post Service**: Handle post creation
2. **Fan-out Service**: Push posts to followers' timelines
3. **Timeline Service**: Serve user feeds
4. **Ranking Service**: Rank posts (for trending/explore)
5. **Database**: Store posts, timelines, user relationships
6. **Cache**: Hot timelines, trending posts
7. **Message Queue**: Async fan-out processing

**A - Algorithm/Data Structure:**

**Feed Generation Approaches:**

**Option 1: Push Model (Fan-out on Write)**
- When user posts, push to all followers' timelines
- Pros: Fast reads (O(1) query), real-time
- Cons: High write amplification, storage intensive
- Use for: Active users, small-medium follower counts

**Option 2: Pull Model (Fan-out on Read)**
- When user requests feed, fetch posts from followed users
- Pros: Lower storage, simpler writes
- Cons: Slow reads (O(follows) queries), not real-time
- Use for: Inactive users, very large follower counts

**Option 3: Hybrid Approach (Recommended)**
- Push for active users (< 30 days since last login)
- Pull for inactive users
- Push for users with < 1000 followers
- Pull for users with > 1000 followers (celebrity problem)

**Ranking Algorithm (for Trending):**
- Factors: Likes, replies, time decay, user engagement
- Score = (likes × 2 + replies × 3) / (time_decay_factor)
- Pre-compute trending posts every 5 minutes
- Cache top 1000 trending posts

**D - Detailed Design:**

**Database Schema:**

CREATE TABLE posts (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    content TEXT,
    media_urls JSON,
    reply_to_post_id BIGINT,
    created_at TIMESTAMP,
    likes_count BIGINT DEFAULT 0,
    replies_count BIGINT DEFAULT 0,
    INDEX idx_user_created (user_id, created_at),
    INDEX idx_created (created_at)
);

CREATE TABLE timelines (
    user_id BIGINT,
    post_id BIGINT,
    created_at TIMESTAMP,
    PRIMARY KEY (user_id, post_id),
    INDEX idx_user_created (user_id, created_at DESC)
);

CREATE TABLE follows (
    follower_id BIGINT,
    followee_id BIGINT,
    created_at TIMESTAMP,
    PRIMARY KEY (follower_id, followee_id),
    INDEX idx_followee (followee_id)
);

CREATE TABLE trending_posts (
    post_id BIGINT PRIMARY KEY,
    score DECIMAL(10,2),
    updated_at TIMESTAMP,
    INDEX idx_score (score DESC)
);

**Post Creation Flow:**

1. User creates post
2. Post Service → Store in posts table
3. Post Service → Kafka Topic "new_posts"
4. Kafka → Fan-out Service:
   - Get all followers
   - For active followers: Push to their timeline (timelines table)
   - For inactive followers: Skip (will pull on-demand)
   - For high-follower accounts: Use pull model
5. Update post counts (likes, replies)

**Feed Retrieval Flow (Push Model):**

1. User requests feed
2. Timeline Service → Check cache (Redis)
3. If cache hit: Return cached feed
4. If cache miss: Query timelines table
5. Join with posts table to get full post data
6. Cache result
7. Return feed

**Feed Retrieval Flow (Pull Model):**

1. User requests feed
2. Timeline Service → Get list of followed users
3. Query posts table for recent posts from followed users
4. Merge and sort by timestamp
5. Return top N posts

**Trending Feed Flow:**

1. Background job runs every 5 minutes
2. Calculate scores for all recent posts (last 24 hours)
3. Update trending_posts table
4. Cache top 1000 posts
5. User requests trending feed → Return from cache

**E - Evaluation (Bottlenecks & Optimizations):**

**Potential Bottlenecks:**

1. **Fan-out Write Amplification**: 500M posts × 200 followers = 100B writes/day
   - Solution: 
     - Hybrid approach (push for active, pull for inactive)
     - Batch writes
     - Async processing via message queue
     - Separate handling for high-follower accounts

2. **Timeline Storage**: 100B entries/day = 18.25 PB/year
   - Solution:
     - Only store for active users (push model)
     - Archive old timelines
     - Use compression
     - Consider pull model for inactive users

3. **Feed Read Queries**: 350K reads/sec
   - Solution:
     - Caching (80% hit rate reduces DB load by 80%)
     - Read replicas
     - Pre-computed timelines
     - CDN for static content

4. **Celebrity Problem**: Users with millions of followers
   - Solution:
     - Use pull model for high-follower accounts
     - Separate fan-out queue with rate limiting
     - Batch processing
     - Don't store in individual timelines (too expensive)

5. **Trending Calculation**: Expensive to compute
   - Solution:
     - Pre-compute every 5 minutes (not real-time)
     - Only consider recent posts (last 24 hours)
     - Use approximate algorithms
     - Cache results

**Optimizations:**
- **Caching**: Cache hot timelines (80% hit rate)
- **Read Replicas**: Use for feed queries
- **Database Sharding**: Shard by user_id
- **CDN**: For media content
- **Pagination**: Cursor-based pagination
- **Lazy Loading**: Load media on-demand

**D - Discussion (Trade-offs, Scaling, Reliability):**

**Trade-offs:**

1. **Push vs Pull Model:**
   - Push: Fast reads, high storage/writes
   - Pull: Lower storage, slower reads
   - **Choice**: Hybrid - push for active users, pull for inactive/high-follower

2. **Consistency:**
   - Strong consistency: Slower, but guaranteed order
   - Eventual consistency: Faster, acceptable for feeds
   - **Choice**: Eventual consistency (feeds can be slightly stale)

3. **Storage vs Compute:**
   - Pre-compute timelines: High storage, fast reads
   - Compute on-demand: Lower storage, slower reads
   - **Choice**: Pre-compute for active users, compute for inactive

4. **Real-time vs Batch:**
   - Real-time fan-out: Immediate updates, high cost
   - Batch fan-out: Delayed updates, lower cost
   - **Choice**: Real-time for active users, batch for others

**Scaling Strategy:**

**Phase 1 (Current):**
- Push model for all users
- Single database cluster
- Redis cache

**Phase 2 (10x scale):**
- Hybrid push/pull model
- Database sharding
- Redis cluster
- Separate queues for high-follower accounts

**Phase 3 (100x scale):**
- Optimized hybrid model
- Multi-region deployment
- CDN globally
- Advanced caching strategies

**Reliability:**
- Message queue ensures delivery
- Database replication
- Cache replication
- Health checks and auto-failover
- Graceful degradation (show cached feed if DB down)

**Common Follow-up Questions:**

1. "How do you handle the celebrity problem?"
   - Use pull model for accounts with > 1000 followers
   - Don't fan-out to individual timelines
   - Use separate ranking/trending algorithm
   - Cache their recent posts separately

2. "How do you ensure feed freshness?"
   - Real-time fan-out for active users
   - Background job to refresh stale timelines
   - TTL on cached feeds
   - WebSocket for real-time updates

3. "What about personalized ranking?"
   - ML model to score posts based on user engagement
   - Factors: user interests, past interactions, post content
   - Pre-compute scores, store in timelines table
   - Update scores periodically

4. "How would you add real-time updates?"
   - WebSocket connection per user
   - Push new posts to followers in real-time
   - Update cached timeline
   - Use message queue for reliable delivery

5. "What about search functionality?"
   - Use Elasticsearch for full-text search
   - Index posts content, user names, hashtags
   - Separate search service
   - Cache popular searches`,
					CodeExamples: `Fan-out Service Implementation:

def fan_out_post(post):
    user_id = post['user_id']
    
    # Get all followers
    followers = db.get_followers(user_id)
    
    # Separate active and inactive followers
    active_followers = []
    inactive_followers = []
    
    for follower_id in followers:
        if is_active_user(follower_id):
            active_followers.append(follower_id)
        else:
            inactive_followers.append(follower_id)
    
    # Push to active followers' timelines
    if len(active_followers) < 1000:  # Not a celebrity
        batch_insert_timelines(active_followers, post['id'])
    else:
        # Celebrity - use pull model, don't store
        pass
    
    # Queue for inactive followers (will pull on-demand)
    if inactive_followers:
        queue_pull_feed_update(inactive_followers)

def is_active_user(user_id):
    # Check if user logged in last 30 days
    last_login = db.get_last_login(user_id)
    return (time.now() - last_login).days < 30

Timeline Retrieval (Push Model):

def get_timeline(user_id, limit=20, cursor=None):
    # Check cache first
    cache_key = f"timeline:{user_id}:{cursor or 'latest'}"
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Query database
    query = """
        SELECT p.* FROM timelines t
        JOIN posts p ON t.post_id = p.id
        WHERE t.user_id = ?
        ORDER BY t.created_at DESC
        LIMIT ?
    """
    
    if cursor:
        query += " AND t.created_at < ?"
        params = (user_id, limit, cursor)
    else:
        params = (user_id, limit)
    
    posts = db.execute(query, params)
    
    # Cache result
    redis.setex(cache_key, 300, json.dumps(posts))  # 5 min TTL
    
    return posts

Timeline Retrieval (Pull Model):

def get_timeline_pull(user_id, limit=20, cursor=None):
    # Get followed users
    followed_users = db.get_followed_users(user_id)
    
    # Query recent posts from followed users
    query = """
        SELECT * FROM posts
        WHERE user_id IN (?)
        ORDER BY created_at DESC
        LIMIT ?
    """
    
    if cursor:
        query += " AND created_at < ?"
        params = (followed_users, limit, cursor)
    else:
        params = (followed_users, limit)
    
    posts = db.execute(query, params)
    return posts

Trending Calculation:

def calculate_trending():
    # Get posts from last 24 hours
    recent_posts = db.get_recent_posts(hours=24)
    
    trending = []
    for post in recent_posts:
        # Calculate score
        score = calculate_score(post)
        trending.append({
            'post_id': post['id'],
            'score': score
        })
    
    # Sort by score
    trending.sort(key=lambda x: x['score'], reverse=True)
    
    # Update trending_posts table
    db.truncate_table('trending_posts')
    db.batch_insert('trending_posts', trending[:1000])
    
    # Cache top 100
    redis.set('trending:top100', json.dumps(trending[:100]), ex=300)

def calculate_score(post):
    # Score = (likes × 2 + replies × 3) / time_decay
    likes = post['likes_count']
    replies = post['replies_count']
    age_hours = (time.now() - post['created_at']).total_seconds() / 3600
    
    # Time decay: score halves every 2 hours
    time_decay = 2 ** (age_hours / 2)
    
    score = (likes * 2 + replies * 3) / time_decay
    return score

Architecture Diagram:

[User] → [API Gateway] → [Post Service]
                            |
                            v
                    [Kafka: new_posts]
                            |
                            v
                    [Fan-out Service]
                            |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
[Active Users]      [Inactive Users]    [Celebrity Accounts]
(Push to timeline)  (Skip, pull later)  (Pull model)
        |                   |                   |
        v                   v                   v
[Timelines Table]   [Pull on-demand]   [Ranking Service]
        |                                       |
        v                                       v
[Timeline Service] ←→ [Cache (Redis)]    [Trending Feed]
        |
        v
[Database Cluster] (Sharded)

Capacity Summary:

Posts: 58K/sec
- Fan-out: 58K × 200 = 11.6M timeline writes/sec (with hybrid: ~3M effective)

Feed Reads: 350K/sec
- Cache (80% hit): 280K/sec from Redis
- DB (20% miss): 70K/sec from DB
- 10 replicas: 7K reads/sec per replica

Storage: 18.25 PB/year (with hybrid: ~5 PB/year for active users)
- Push model: Only for active users (~30% of users)
- Pull model: No storage, compute on-demand

Celebrity Handling:
- Accounts with > 1000 followers: Use pull model
- Reduces storage by ~70%`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          25,
			Title:       "Common System Design Interview Questions",
			Description: "Practice with frequently asked system design interview questions from top tech companies.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "Design Twitter/X",
					Content: `**Complete Interview Walkthrough Using RESHADED Framework**

**Problem Statement:**
Design Twitter/X - a social media platform where users can post tweets, follow other users, and see a timeline of tweets from people they follow.

**R - Requirements Clarification:**

**Questions to Ask:**
1. "What's the expected scale? How many users and tweets per day?"
2. "What's the average number of followers per user?"
3. "Do we need to support media (images, videos) in tweets?"
4. "Do we need real-time timeline updates or is eventual consistency OK?"
5. "What's the acceptable latency for timeline loads?"
6. "Do we need to support trending topics or just user timelines?"
7. "What about likes, retweets, replies - are these required?"

**Assumptions (After Clarification):**
- 300M users, 500M tweets/day
- Average 200 followers per user
- Support images and videos (via CDN)
- Real-time updates preferred (< 1 second delay)
- < 200ms latency for timeline load
- Support user timelines and trending feed
- Support likes, retweets, replies

**E - Estimate (Capacity Planning):**

**Traffic Estimation:**
- Tweets: 500M/day = ~5,800 tweets/sec average
- Peak: 10x average = ~58,000 tweets/sec
- Timeline reads: 300M users × 20 reads/day = 6B reads/day = ~70,000 reads/sec average
- Peak reads: 10x average = ~700,000 reads/sec

**Storage Estimation:**
- Tweet: ~500 bytes (tweet_id, user_id, content, timestamp, media_urls)
- With fan-out: 500M tweets × 200 followers = 100B timeline entries/day
- Storage: 100B × 500 bytes = 50 TB/day
- 1 year retention: 50 TB × 365 = 18.25 PB
- Media: Assume 20% tweets have media, average 2 MB per media
- Media storage: 500M × 0.2 × 2 MB = 200 TB/day

**Bandwidth Estimation:**
- Write: 58,000 tweets/sec × 500 bytes = 29 MB/s = 232 Mbps
- Read: 700,000 reads/sec × 10 KB (timeline response) = 7 GB/s = 56 Gbps
- Media: Much higher (use CDN)

**S - System Interface (API Design):**

**Endpoints:**

POST /api/v1/tweets
Request:
{
  "content": "Hello world!",
  "media_urls": ["https://..."],
  "reply_to_tweet_id": 123  // Optional
}

Response:
{
  "tweet_id": 456,
  "user_id": 789,
  "content": "Hello world!",
  "created_at": "2024-01-17T10:00:00Z"
}

GET /api/v1/timeline?user_id=123&limit=20&cursor=abc
Response:
{
  "tweets": [
    {
      "tweet_id": 456,
      "user_id": 789,
      "content": "Hello world!",
      "created_at": "2024-01-17T10:00:00Z",
      "likes": 42,
      "retweets": 5,
      "replies": 3
    }
  ],
  "next_cursor": "def"
}

POST /api/v1/tweets/{tweet_id}/like
POST /api/v1/tweets/{tweet_id}/retweet
POST /api/v1/users/{user_id}/follow

**H - High-Level Design:**

**Architecture:**
    Client → API Gateway → [Post Service, Timeline Service, Graph Service]
                              ↓
                        [Message Queue (Kafka)]
                              ↓
                    [Fan-out Service, Ranking Service]
                              ↓
                    [Database Cluster, Cache, CDN]

**Components:**
1. **Post Service**: Handle tweet creation
2. **Fan-out Service**: Push tweets to followers' timelines
3. **Timeline Service**: Serve user timelines
4. **Graph Service**: Manage follower relationships
5. **Ranking Service**: Rank tweets for trending feed
6. **Database**: Store tweets, timelines, user data
7. **Cache**: Hot timelines (Redis/Manhattan)
8. **CDN**: Media content delivery

**A - Algorithm/Data Structure:**

**Timeline Generation Approaches:**

**Push Model (Fan-out on Write):**
- When user posts tweet, push to all followers' timelines
- Pros: Fast reads (O(1) query), real-time
- Cons: High write amplification, storage intensive
- Use for: Active users, users with < 1000 followers

**Pull Model (Fan-out on Read):**
- When user requests timeline, fetch from followed users
- Pros: Lower storage, simpler writes
- Cons: Slow reads (O(follows) queries)
- Use for: Inactive users, celebrity accounts (> 1000 followers)

**Hybrid Approach (Recommended):**
- Push for active users (< 30 days since last login)
- Pull for inactive users
- Push for users with < 1000 followers
- Pull for users with > 1000 followers (celebrity problem)

**D - Detailed Design:**

**Database Schema:**

CREATE TABLE tweets (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    content VARCHAR(280),
    media_urls JSON,
    reply_to_tweet_id BIGINT,
    created_at TIMESTAMP,
    likes_count BIGINT DEFAULT 0,
    retweets_count BIGINT DEFAULT 0,
    replies_count BIGINT DEFAULT 0,
    INDEX idx_user_created (user_id, created_at),
    INDEX idx_created (created_at)
);

CREATE TABLE timelines (
    user_id BIGINT,
    tweet_id BIGINT,
    created_at TIMESTAMP,
    PRIMARY KEY (user_id, tweet_id),
    INDEX idx_user_created (user_id, created_at DESC)
);

CREATE TABLE follows (
    follower_id BIGINT,
    followee_id BIGINT,
    created_at TIMESTAMP,
    PRIMARY KEY (follower_id, followee_id),
    INDEX idx_followee (followee_id)
);

CREATE TABLE likes (
    user_id BIGINT,
    tweet_id BIGINT,
    created_at TIMESTAMP,
    PRIMARY KEY (user_id, tweet_id)
);

**Tweet Creation Flow:**

1. User creates tweet
2. Post Service → Store in tweets table
3. Post Service → Kafka Topic "new_tweets"
4. Kafka → Fan-out Service:
   - Get all followers from Graph Service
   - For active followers (< 1000 followers): Push to timeline
   - For inactive followers: Skip (rebuild on-demand)
   - For celebrity accounts: Use pull model
5. Update tweet counts

**Timeline Retrieval Flow (Push Model):**

1. User requests timeline
2. Timeline Service → Check Redis cache (hot timeline)
3. If cache hit: Return cached timeline
4. If cache miss: Query timelines table
5. Join with tweets table for full tweet data
6. Cache result
7. Return timeline

**Timeline Retrieval Flow (Pull Model):**

1. User requests timeline
2. Timeline Service → Get list of followed users
3. Query tweets table for recent tweets from followed users
4. Merge and sort by timestamp
5. Return top N tweets

**E - Evaluation (Bottlenecks & Optimizations):**

**Potential Bottlenecks:**

1. **Fan-out Write Amplification**: 500M tweets × 200 followers = 100B writes/day
   - Solution: Hybrid approach (push for active, pull for inactive)
   - Effective writes: ~30B/day (only active users)
   - Batch writes, async processing

2. **Timeline Storage**: 100B entries/day = 18.25 PB/year
   - Solution: Only store for active users (push model)
   - Archive old timelines
   - Use compression

3. **Timeline Read Queries**: 700K reads/sec
   - Solution: Caching (80% hit rate reduces DB load by 80%)
   - Read replicas
   - Pre-computed timelines

4. **Celebrity Problem**: Users with millions of followers
   - Solution: Use pull model for high-follower accounts
   - Separate queue with rate limiting
   - Don't store in individual timelines

5. **Graph Queries**: Getting followers for fan-out
   - Solution: Cache follower lists
   - Graph database (Neo4j) or optimized SQL
   - Shard by user_id

**Optimizations:**
- **Caching**: Cache hot timelines (80% hit rate)
- **Read Replicas**: Use for timeline queries
- **Database Sharding**: Shard by user_id
- **CDN**: For media content
- **Pagination**: Cursor-based pagination
- **Lazy Loading**: Load media on-demand

**D - Discussion (Trade-offs, Scaling, Reliability):**

**Trade-offs:**

1. **Push vs Pull Model:**
   - Push: Fast reads, high storage/writes
   - Pull: Lower storage, slower reads
   - **Choice**: Hybrid - push for active users, pull for inactive/celebrities

2. **Consistency:**
   - Strong consistency: Slower, but guaranteed order
   - Eventual consistency: Faster, acceptable for feeds
   - **Choice**: Eventual consistency (feeds can be slightly stale)

3. **Storage vs Compute:**
   - Pre-compute timelines: High storage, fast reads
   - Compute on-demand: Lower storage, slower reads
   - **Choice**: Pre-compute for active users, compute for inactive

4. **Real-time vs Batch:**
   - Real-time fan-out: Immediate updates, high cost
   - Batch fan-out: Delayed updates, lower cost
   - **Choice**: Real-time for active users, batch for others

**Scaling Strategy:**

**Phase 1 (Current):**
- Push model for all users
- Single database cluster
- Redis cache

**Phase 2 (10x scale):**
- Hybrid push/pull model
- Database sharding by user_id
- Redis cluster
- Separate queues for celebrities

**Phase 3 (100x scale):**
- Optimized hybrid model
- Multi-region deployment
- CDN globally
- Advanced caching strategies

**Reliability:**
- Message queue ensures delivery
- Database replication
- Cache replication
- Health checks and auto-failover
- Graceful degradation (show cached timeline if DB down)

**Common Follow-up Questions:**

1. "How do you handle the celebrity problem?"
   - Use pull model for accounts with > 1000 followers
   - Don't fan-out to individual timelines
   - Use separate ranking/trending algorithm
   - Cache their recent tweets separately

2. "How do you ensure timeline freshness?"
   - Real-time fan-out for active users
   - Background job to refresh stale timelines
   - TTL on cached timelines
   - WebSocket for real-time updates

3. "What about trending topics?"
   - Track hashtag mentions in real-time
   - Calculate trending scores (mentions, time decay)
   - Pre-compute trending topics every 5 minutes
   - Cache top trending topics

4. "How would you add search functionality?"
   - Use Elasticsearch for full-text search
   - Index tweet content, user names, hashtags
   - Separate search service
   - Cache popular searches

5. "What about media storage and delivery?"
   - Store media in object storage (S3)
   - Use CDN for global delivery
   - Generate multiple sizes/thumbnails
   - Lazy load media on timeline`,
					CodeExamples: `Fan-out Service Implementation:

def fan_out_tweet(tweet):
    user_id = tweet['user_id']
    
    # Get all followers
    followers = graph_service.get_followers(user_id)
    
    # Separate by type
    active_followers = []
    inactive_followers = []
    celebrity_followers = []
    
    for follower_id in followers:
        if is_celebrity_account(follower_id):  # > 1000 followers
            celebrity_followers.append(follower_id)
        elif is_active_user(follower_id):
            active_followers.append(follower_id)
        else:
            inactive_followers.append(follower_id)
    
    # Push to active followers' timelines
    if len(active_followers) < 1000:  # Not a celebrity tweet
        batch_insert_timelines(active_followers, tweet['id'])
    
    # Skip inactive followers (will pull on-demand)
    # Skip celebrity followers (use pull model)

Timeline Retrieval (Push Model):

def get_timeline(user_id, limit=20, cursor=None):
    # Check cache first
    cache_key = f"timeline:{user_id}:{cursor or 'latest'}"
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Query database
    query = """
        SELECT t.* FROM timelines tl
        JOIN tweets t ON tl.tweet_id = t.id
        WHERE tl.user_id = ?
        ORDER BY tl.created_at DESC
        LIMIT ?
    """
    
    if cursor:
        query += " AND tl.created_at < ?"
        params = (user_id, limit, cursor)
    else:
        params = (user_id, limit)
    
    tweets = db.execute(query, params)
    
    # Cache result
    redis.setex(cache_key, 300, json.dumps(tweets))  # 5 min TTL
    
    return tweets

Timeline Retrieval (Pull Model):

def get_timeline_pull(user_id, limit=20, cursor=None):
    # Get followed users
    followed_users = graph_service.get_followed_users(user_id)
    
    # Query recent tweets from followed users
    query = """
        SELECT * FROM tweets
        WHERE user_id IN (?)
        ORDER BY created_at DESC
        LIMIT ?
    """
    
    if cursor:
        query += " AND created_at < ?"
        params = (followed_users, limit, cursor)
    else:
        params = (followed_users, limit)
    
    tweets = db.execute(query, params)
    return tweets

Architecture Diagram:

[Client] → [API Gateway] → [Post Service]
                              |
                              v
                    [Kafka: new_tweets]
                              |
                              v
                    [Fan-out Service]
                              |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
[Active Users]      [Inactive Users]    [Celebrity Accounts]
(Push to timeline)  (Skip, pull later)  (Pull model)
        |                   |                   |
        v                   v                   v
[Timelines Table]   [Pull on-demand]   [Ranking Service]
        |                                       |
        v                                       v
[Timeline Service] ←→ [Cache (Redis)]    [Trending Feed]
        |
        v
[Database Cluster] (Sharded)

Capacity Summary:

Tweets: 58K/sec
- Fan-out: 58K × 200 = 11.6M timeline writes/sec (with hybrid: ~3M effective)

Timeline Reads: 700K/sec
- Cache (80% hit): 560K/sec from Redis
- DB (20% miss): 140K/sec from DB
- 10 replicas: 14K reads/sec per replica

Storage: 18.25 PB/year (with hybrid: ~5 PB/year for active users)
- Push model: Only for active users (~30% of users)
- Pull model: No storage, compute on-demand

Celebrity Handling:
- Accounts with > 1000 followers: Use pull model
- Reduces storage by ~70%`,
				},
				{
					Title: "Design Instagram",
					Content: `**Complete Interview Walkthrough Using RESHADED Framework**

**Problem Statement:**
Design Instagram - a photo and video sharing platform where users can upload media, follow others, and see a feed of posts from people they follow.

**R - Requirements Clarification:**

**Questions to Ask:**
1. "What's the expected scale? How many users and posts per day?"
2. "What's the average number of followers per user?"
3. "What media formats do we need to support? (photos, videos, stories?)"
4. "What's the acceptable feed load latency?"
5. "Do we need real-time updates or is eventual consistency OK?"
6. "Do we need to support Stories (24-hour expiring content)?"
7. "What about video processing and transcoding?"

**Assumptions (After Clarification):**
- 2B users, 500M posts/day
- Average 200 followers per user
- Support photos (JPEG, PNG) and videos (MP4)
- Support Stories (expiring content)
- Feed load < 200ms
- Eventual consistency acceptable (< 1 second delay)
- Videos need transcoding for multiple quality tiers

**E - Estimate (Capacity Planning):**

**Traffic Estimation:**
- Posts: 500M/day = ~5,800 posts/sec average
- Peak: 10x average = ~58,000 posts/sec
- Feed reads: 2B users × 10 reads/day = 20B reads/day = ~230,000 reads/sec average
- Peak reads: 10x average = ~2,300,000 reads/sec

**Storage Estimation:**
- Photo: Average 200 KB (compressed)
- Video: Average 5 MB (compressed, 1 minute)
- Assume 70% photos, 30% videos
- Photo storage: 500M × 0.7 × 200 KB = 70 TB/day
- Video storage: 500M × 0.3 × 5 MB = 750 TB/day
- Total media: 820 TB/day
- With 3x replication: 2.46 PB/day
- 1 year retention: 2.46 PB × 365 = 898 PB
- Metadata: 500M posts × 1 KB = 500 GB/day

**Bandwidth Estimation:**
- Upload: 58,000 posts/sec × 2 MB average = 116 GB/s = 928 Gbps
- Download: 2.3M reads/sec × 500 KB average = 1.15 TB/s = 9.2 Tbps
- Use CDN for downloads (reduces origin load by 95%)

**S - System Interface (API Design):**

**Endpoints:**

POST /api/v1/posts
Request: multipart/form-data
- photo/video file
- caption (text)
- location (optional)

Response:
{
  "post_id": 456,
  "user_id": 789,
  "media_url": "https://cdn.instagram.com/...",
  "thumbnail_url": "https://cdn.instagram.com/...",
  "caption": "Beautiful sunset!",
  "created_at": "2024-01-17T10:00:00Z"
}

GET /api/v1/feed?user_id=123&limit=20&cursor=abc
Response:
{
  "posts": [
    {
      "post_id": 456,
      "user_id": 789,
      "media_url": "https://...",
      "caption": "Beautiful sunset!",
      "likes": 42000,
      "comments": 500,
      "created_at": "2024-01-17T10:00:00Z"
    }
  ],
  "next_cursor": "def"
}

POST /api/v1/posts/{post_id}/like
POST /api/v1/posts/{post_id}/comment
GET /api/v1/stories?user_id=123

**H - High-Level Design:**

**Architecture:**
    Client → API Gateway → [Upload Service, Feed Service, Media Service]
                              ↓
                        [Message Queue (Kafka)]
                              ↓
                    [Transcoding Pipeline, Fan-out Service]
                              ↓
                    [Object Storage, Database, Cache, CDN]

**Components:**
1. **Upload Service**: Handle media uploads
2. **Media Service**: Process and serve media
3. **Transcoding Pipeline**: Convert videos to multiple formats
4. **Feed Service**: Generate and serve feeds
5. **Fan-out Service**: Push posts to followers' feeds
6. **Object Storage**: Store media files (S3-like)
7. **CDN**: Global media delivery
8. **Database**: Store metadata (PostgreSQL), feeds (Cassandra)
9. **Cache**: Hot feeds, media metadata (Redis)

**A - Algorithm/Data Structure:**

**Feed Generation:**
- **Hybrid Model**: Push for active users, pull for inactive
- **ML Ranking**: Rank posts by relevance (user interests, engagement)
- **Pre-compute**: Active users get pre-computed feeds
- **On-demand**: Inactive users generate feeds on request

**Media Processing:**
- **Photos**: Generate thumbnails (multiple sizes)
- **Videos**: Transcode to multiple quality tiers (240p, 360p, 720p, 1080p)
- **Format Conversion**: Convert to web-optimized formats (WebP for photos, H.264 for videos)

**D - Detailed Design:**

**Database Schema:**

CREATE TABLE posts (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    media_type VARCHAR(20),  -- photo, video
    media_url TEXT,
    thumbnail_url TEXT,
    caption TEXT,
    location VARCHAR(255),
    created_at TIMESTAMP,
    likes_count BIGINT DEFAULT 0,
    comments_count BIGINT DEFAULT 0,
    INDEX idx_user_created (user_id, created_at),
    INDEX idx_created (created_at)
);

CREATE TABLE feeds (
    user_id BIGINT,
    post_id BIGINT,
    score DECIMAL(10,2),  -- ML ranking score
    created_at TIMESTAMP,
    PRIMARY KEY (user_id, post_id),
    INDEX idx_user_score (user_id, score DESC)
);

CREATE TABLE stories (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    media_url TEXT,
    expires_at TIMESTAMP,
    created_at TIMESTAMP,
    INDEX idx_user_expires (user_id, expires_at)
);

**Photo Upload Flow:**

1. User uploads photo
2. Upload Service → Store raw photo in object storage (S3)
3. Upload Service → Queue processing job (Kafka)
4. Processing Worker:
   - Generate thumbnails (150x150, 320x320, 640x640)
   - Convert to WebP format
   - Store processed images in CDN
5. Store metadata in PostgreSQL
6. Fan-out to followers' feeds (Cassandra)

**Video Upload Flow:**

1. User uploads video
2. Upload Service → Store raw video in object storage
3. Upload Service → Queue transcoding job
4. Transcoding Pipeline:
   - Extract metadata (duration, resolution)
   - Transcode to multiple qualities (240p, 360p, 720p, 1080p)
   - Generate thumbnails
   - Store transcoded videos in CDN
5. Store metadata in PostgreSQL
6. Fan-out to followers' feeds

**Feed Retrieval Flow:**

1. User requests feed
2. Feed Service → Check Redis cache
3. If cache hit: Return cached feed
4. If cache miss:
   - If active user: Query pre-computed feed (Cassandra)
   - If inactive user: Generate feed on-demand (pull model)
5. Rank posts by ML score
6. Join with posts table for full data
7. Cache result
8. Return feed

**E - Evaluation (Bottlenecks & Optimizations):**

**Potential Bottlenecks:**

1. **Media Storage**: 820 TB/day = 898 PB/year
   - Solution: Object storage (S3), compression, CDN caching
   - Archive old media to cold storage
   - Use efficient formats (WebP, H.264)

2. **Video Transcoding**: CPU-intensive, slow
   - Solution: Distributed transcoding workers
   - Async processing (user can browse while video processes)
   - Pre-transcode common formats
   - Use GPU acceleration

3. **Feed Generation**: 2.3M reads/sec
   - Solution: Pre-compute feeds for active users
   - Caching (80% hit rate)
   - Read replicas
   - ML ranking pre-computed

4. **CDN Bandwidth**: 9.2 Tbps
   - Solution: Multiple CDN providers
   - Edge caching (95% cache hit rate)
   - Adaptive quality based on bandwidth

5. **Fan-out Writes**: 500M posts × 200 followers = 100B writes/day
   - Solution: Hybrid model (only active users)
   - Batch writes
   - Async processing

**Optimizations:**
- **CDN**: 95% of media served from CDN
- **Caching**: Cache hot feeds (80% hit rate)
- **Compression**: Compress media files
- **Lazy Loading**: Load media on-demand
- **Progressive Loading**: Show thumbnails first, full media on click
- **Database Sharding**: Shard by user_id

**D - Discussion (Trade-offs, Scaling, Reliability):**

**Trade-offs:**

1. **SQL vs NoSQL:**
   - SQL (PostgreSQL): ACID guarantees, complex queries
   - NoSQL (Cassandra): Better write performance, horizontal scaling
   - **Choice**: SQL for metadata, NoSQL for feeds

2. **Push vs Pull Model:**
   - Push: Fast reads, high storage
   - Pull: Lower storage, slower reads
   - **Choice**: Hybrid - push for active, pull for inactive

3. **Media Quality:**
   - High quality: Better UX, higher storage/bandwidth
   - Lower quality: Lower costs, acceptable UX
   - **Choice**: Multiple quality tiers, adaptive streaming

4. **Transcoding:**
   - Pre-transcode: Faster playback, higher storage
   - On-demand: Lower storage, slower first playback
   - **Choice**: Pre-transcode common formats

**Scaling Strategy:**

**Phase 1 (Current):**
- Single object storage
- Basic transcoding
- Push model for all users

**Phase 2 (10x scale):**
- Multi-region object storage
- Distributed transcoding
- Hybrid push/pull model
- CDN globally

**Phase 3 (100x scale):**
- Advanced compression
- GPU-accelerated transcoding
- ML-based quality optimization
- Edge computing for processing

**Reliability:**
- Object storage replication (3x)
- Database replication
- CDN redundancy (multiple providers)
- Health checks and auto-failover
- Graceful degradation (show cached feed if DB down)

**Common Follow-up Questions:**

1. "How do you handle video transcoding at scale?"
   - Distributed worker pool
   - Queue-based processing
   - GPU acceleration
   - Pre-transcode common formats
   - Adaptive quality based on device/bandwidth

2. "What about Stories (24-hour expiring content)?"
   - Separate storage for stories
   - TTL-based expiration
   - Background job to delete expired stories
   - Optimized for fast upload/view

3. "How do you optimize media delivery?"
   - CDN with edge caching
   - Multiple quality tiers
   - Adaptive bitrate streaming
   - Progressive loading
   - Compression (WebP, H.264)

4. "How would you add search functionality?"
   - Elasticsearch for full-text search
   - Index captions, hashtags, locations
   - Image search using ML (visual similarity)
   - Separate search service

5. "What about real-time features (live videos)?"
   - WebRTC for live streaming
   - Separate infrastructure for live content
   - CDN with live streaming support
   - Lower latency requirements`,
					CodeExamples: `Photo Upload Implementation:

def upload_photo(user_id, photo_file, caption):
    # 1. Upload raw photo to object storage
    raw_url = s3.upload(photo_file, f"raw/{user_id}/{uuid()}")
    
    # 2. Queue processing job
    kafka.produce('photo_processing', {
        'raw_url': raw_url,
        'user_id': user_id,
        'caption': caption
    })
    
    # 3. Return immediately (async processing)
    return {'status': 'processing', 'post_id': generate_id()}

def process_photo(job):
    # Generate thumbnails
    thumbnails = {
        '150x150': resize_image(job['raw_url'], 150, 150),
        '320x320': resize_image(job['raw_url'], 320, 320),
        '640x640': resize_image(job['raw_url'], 640, 640)
    }
    
    # Convert to WebP
    webp_urls = {}
    for size, img in thumbnails.items():
        webp_urls[size] = convert_to_webp(img)
    
    # Upload to CDN
    cdn_urls = {}
    for size, url in webp_urls.items():
        cdn_urls[size] = cdn.upload(url, f"photos/{size}/{uuid()}")
    
    # Store metadata
    post = {
        'id': generate_id(),
        'user_id': job['user_id'],
        'media_url': cdn_urls['640x640'],
        'thumbnail_url': cdn_urls['150x150'],
        'caption': job['caption']
    }
    db.insert_post(post)
    
    # Fan-out to followers
    fan_out_service.fan_out_post(post)

Video Transcoding Pipeline:

def transcode_video(video_url, video_id):
    # Extract metadata
    metadata = ffmpeg.probe(video_url)
    duration = metadata['duration']
    resolution = metadata['streams'][0]['resolution']
    
    # Transcode to multiple qualities
    qualities = ['240p', '360p', '720p', '1080p']
    transcoded = {}
    
    for quality in qualities:
        output_url = transcode_to_quality(video_url, quality)
        transcoded[quality] = cdn.upload(output_url, f"videos/{quality}/{video_id}")
    
    # Generate thumbnails
    thumbnails = generate_video_thumbnails(video_url, count=3)
    
    # Update metadata
    db.update_video_metadata(video_id, {
        'transcoded_urls': transcoded,
        'thumbnails': thumbnails,
        'duration': duration
    })

Feed Generation with ML Ranking:

def get_feed(user_id, limit=20):
    # Check cache
    cache_key = f"feed:{user_id}"
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Get pre-computed feed (if active user)
    if is_active_user(user_id):
        feed = cassandra.get_feed(user_id, limit=limit)
    else:
        # Generate on-demand (pull model)
        feed = generate_feed_pull(user_id, limit=limit)
    
    # Rank by ML score (already included in feed)
    feed.sort(key=lambda x: x['score'], reverse=True)
    
    # Join with posts table for full data
    post_ids = [p['post_id'] for p in feed]
    posts = postgres.get_posts(post_ids)
    
    # Cache result
    redis.setex(cache_key, 300, json.dumps(posts))
    
    return posts

Architecture Diagram:

[Client] → [API Gateway] → [Upload Service]
                              |
                              v
                    [Object Storage (S3)]
                              |
                              v
                    [Kafka: processing_queue]
                              |
                              v
                    [Transcoding Workers]
                              |
                              v
                    [CDN] ←→ [Media Delivery]
                              |
                              v
                    [Feed Service] → [Cache] → [Database]
                                         |
                                         v
                                    [Cassandra Feeds]

Capacity Summary:

Media Storage: 898 PB/year
- Photos: 70 TB/day × 365 = 25.5 PB/year
- Videos: 750 TB/day × 365 = 273.75 PB/year
- With replication: 898 PB/year

Feed Reads: 2.3M/sec
- Cache (80% hit): 1.84M/sec from Redis
- DB (20% miss): 460K/sec from DB
- 20 replicas: 23K reads/sec per replica

Transcoding: 58K posts/sec × 30% videos = 17.4K videos/sec
- Distributed workers: 1000 workers × 20 videos/min = 333 videos/sec per worker
- Need: ~52 workers (with 3x headroom: 156 workers)`,
				},
				{
					Title: "Design YouTube",
					Content: `**Complete Interview Walkthrough Using RESHADED Framework**

**Problem Statement:**
Design YouTube - a video sharing platform where users can upload, watch, and interact with videos.

**R - Requirements Clarification:**

**Questions to Ask:**
1. "What's the expected scale? How many users, videos uploaded, and views per day?"
2. "What video formats and qualities do we need to support?"
3. "Do we need live streaming or just pre-recorded videos?"
4. "What's the acceptable video startup latency?"
5. "Do we need search functionality?"
6. "What about recommendations and personalized feeds?"
7. "Do we need to support comments, likes, subscriptions?"

**Assumptions (After Clarification):**
- 2B users, 500 hours uploaded/minute
- Billions of views/day
- Support multiple quality tiers (240p to 4K)
- Support live streaming
- < 2 seconds video startup latency
- Full-text search required
- ML-based recommendations
- Support comments, likes, subscriptions

**E - Estimate (Capacity Planning):**

**Traffic Estimation:**
- Uploads: 500 hours/min = 30,000 hours/hour = 720,000 hours/day
- Average video: 10 minutes, 100 MB (compressed)
- Videos uploaded: 720,000 hours / 10 min = 4.32M videos/day = ~50 videos/sec
- Peak uploads: 10x average = ~500 videos/sec
- Views: 5B views/day = ~58,000 views/sec average
- Peak views: 10x average = ~580,000 views/sec

**Storage Estimation:**
- Video storage: 4.32M videos/day × 100 MB = 432 TB/day
- With multiple quality tiers (5 tiers): 432 TB × 5 = 2.16 PB/day
- With 3x replication: 6.48 PB/day
- 1 year retention: 6.48 PB × 365 = 2,365 PB
- Metadata: 4.32M × 1 KB = 4.32 GB/day
- Thumbnails: 4.32M × 200 KB = 864 GB/day

**Bandwidth Estimation:**
- Upload: 500 videos/sec × 100 MB = 50 GB/s = 400 Gbps
- Download: 580,000 views/sec × 5 MB average = 2.9 TB/s = 23.2 Tbps
- Use CDN (95% cache hit): Origin load = 2.9 TB/s × 0.05 = 145 GB/s

**S - System Interface (API Design):**

**Endpoints:**

POST /api/v1/videos/upload
Request: multipart/form-data
- video file
- title, description
- category, tags

Response:
{
  "video_id": "abc123",
  "status": "processing",
  "upload_url": "https://..."
}

GET /api/v1/videos/{video_id}
Response:
{
  "video_id": "abc123",
  "title": "Amazing Video",
  "description": "...",
  "stream_urls": {
    "240p": "https://cdn.youtube.com/...",
    "360p": "https://cdn.youtube.com/...",
    "720p": "https://cdn.youtube.com/...",
    "1080p": "https://cdn.youtube.com/..."
  },
  "thumbnail_url": "https://...",
  "views": 1000000,
  "likes": 50000,
  "created_at": "2024-01-17T10:00:00Z"
}

GET /api/v1/videos/{video_id}/stream?quality=720p
Response: Video stream (chunked)

GET /api/v1/search?q=query&limit=20
POST /api/v1/videos/{video_id}/like
POST /api/v1/videos/{video_id}/comment
GET /api/v1/recommendations?user_id=123

**H - High-Level Design:**

**Architecture:**
    Client → API Gateway → [Upload Service, Video Service, Search Service]
                              ↓
                        [Message Queue (Kafka)]
                              ↓
                    [Transcoding Pipeline, Analytics Service]
                              ↓
                    [Object Storage, CDN, Database, Search Index]

**Components:**
1. **Upload Service**: Handle video uploads
2. **Transcoding Pipeline**: Convert videos to multiple formats/qualities
3. **Video Service**: Serve video streams
4. **CDN**: Global video delivery
5. **Search Service**: Full-text search (Elasticsearch)
6. **Recommendation Service**: ML-based recommendations
7. **Analytics Service**: Track views, engagement
8. **Database**: Store metadata (SQL), viewing history (NoSQL)
9. **Object Storage**: Store video files

**A - Algorithm/Data Structure:**

**Video Transcoding:**
- **Multiple Formats**: H.264, VP9, AV1
- **Quality Tiers**: 240p, 360p, 480p, 720p, 1080p, 1440p, 4K
- **Adaptive Bitrate**: Adjust quality based on bandwidth
- **Chunking**: Split videos into chunks for streaming

**Search Algorithm:**
- **Inverted Index**: Index video titles, descriptions, tags
- **Ranking**: Relevance score (text match, views, likes, recency)
- **ML Ranking**: Personalization based on user history

**Recommendation Algorithm:**
- **Collaborative Filtering**: Users who watched X also watched Y
- **Content-Based**: Similar videos based on metadata
- **Deep Learning**: Neural networks for embeddings
- **Real-time**: Update recommendations based on recent views

**D - Detailed Design:**

**Database Schema:**

CREATE TABLE videos (
    id VARCHAR(50) PRIMARY KEY,
    user_id BIGINT NOT NULL,
    title VARCHAR(255),
    description TEXT,
    category VARCHAR(50),
    tags JSON,
    duration INT,  -- seconds
    thumbnail_url TEXT,
    status VARCHAR(20),  -- processing, ready, failed
    views_count BIGINT DEFAULT 0,
    likes_count BIGINT DEFAULT 0,
    comments_count BIGINT DEFAULT 0,
    created_at TIMESTAMP,
    INDEX idx_user_created (user_id, created_at),
    INDEX idx_created (created_at),
    FULLTEXT INDEX idx_search (title, description)
);

CREATE TABLE video_qualities (
    video_id VARCHAR(50),
    quality VARCHAR(20),  -- 240p, 360p, etc.
    format VARCHAR(20),  -- h264, vp9, av1
    cdn_url TEXT,
    file_size BIGINT,
    bitrate INT,
    PRIMARY KEY (video_id, quality)
);

CREATE TABLE viewing_history (
    user_id BIGINT,
    video_id VARCHAR(50),
    watched_seconds INT,
    completed BOOLEAN,
    created_at TIMESTAMP,
    PRIMARY KEY (user_id, video_id, created_at),
    INDEX idx_user_created (user_id, created_at DESC)
);

**Video Upload Flow:**

1. User uploads video
2. Upload Service → Store raw video in object storage (S3)
3. Upload Service → Store metadata in database (status: "processing")
4. Upload Service → Queue transcoding job (Kafka)
5. Transcoding Pipeline:
   - Extract metadata (duration, resolution, codec)
   - Transcode to multiple formats/qualities
   - Generate thumbnails
   - Upload transcoded videos to CDN
   - Update database (status: "ready")
6. Index video in search engine (Elasticsearch)

**Video Streaming Flow:**

1. User requests video
2. Video Service → Get video metadata from database
3. Video Service → Get available quality URLs
4. Client → Request video chunks from CDN
5. CDN → Serve video chunks (adaptive bitrate)
6. Analytics Service → Track viewing progress
7. Update viewing history

**Search Flow:**

1. User searches for query
2. Search Service → Query Elasticsearch
3. Elasticsearch → Return ranked results
4. Search Service → Join with video metadata
5. Apply personalization (if logged in)
6. Return results

**E - Evaluation (Bottlenecks & Optimizations):**

**Potential Bottlenecks:**

1. **Video Storage**: 2,365 PB/year
   - Solution: Object storage (S3), compression, CDN caching
   - Archive old videos to cold storage
   - Use efficient codecs (AV1, VP9)

2. **Video Transcoding**: CPU-intensive, slow
   - Solution: Distributed transcoding workers
   - GPU acceleration
   - Pre-transcode common formats
   - Async processing (user can browse while processing)

3. **CDN Bandwidth**: 23.2 Tbps
   - Solution: Multiple CDN providers
   - Edge caching (95% cache hit rate)
   - Adaptive bitrate streaming
   - Pre-position popular videos

4. **Search Queries**: Millions of queries/day
   - Solution: Elasticsearch cluster
   - Cache popular searches
   - Shard index by terms
   - Pre-compute trending searches

5. **Recommendation Computation**: Expensive ML models
   - Solution: Pre-compute recommendations
   - Update periodically (every few hours)
   - Use approximate algorithms
   - Cache recommendations

**Optimizations:**
- **CDN**: 95% of videos served from CDN
- **Adaptive Bitrate**: Adjust quality based on bandwidth
- **Chunking**: Stream videos in chunks
- **Pre-loading**: Pre-load next chunk
- **Compression**: Use efficient codecs (AV1)
- **Caching**: Cache popular videos at edge
- **Database Sharding**: Shard by video_id or user_id

**D - Discussion (Trade-offs, Scaling, Reliability):**

**Trade-offs:**

1. **Storage vs Quality:**
   - High quality: Better UX, higher storage
   - Lower quality: Lower costs, acceptable UX
   - **Choice**: Multiple quality tiers, adaptive streaming

2. **Transcoding:**
   - Pre-transcode: Faster playback, higher storage
   - On-demand: Lower storage, slower first playback
   - **Choice**: Pre-transcode common formats, on-demand for rare formats

3. **CDN vs Origin:**
   - CDN: Lower latency, higher cost
   - Origin: Lower cost, higher latency
   - **Choice**: CDN for delivery, origin for uploads

4. **Search Freshness:**
   - Real-time indexing: Up-to-date, higher cost
   - Batch indexing: Delayed, lower cost
   - **Choice**: Near-real-time (few minutes delay)

**Scaling Strategy:**

**Phase 1 (Current):**
- Single object storage
- Basic transcoding
- Single CDN provider

**Phase 2 (10x scale):**
- Multi-region object storage
- Distributed transcoding
- Multiple CDN providers
- Elasticsearch cluster

**Phase 3 (100x scale):**
- Advanced compression (AV1)
- GPU-accelerated transcoding
- Edge computing for processing
- ML-based quality optimization

**Reliability:**
- Object storage replication (3x)
- CDN redundancy (multiple providers)
- Database replication
- Health checks and auto-failover
- Graceful degradation (serve lower quality if CDN down)

**Common Follow-up Questions:**

1. "How do you handle video transcoding at scale?"
   - Distributed worker pool (thousands of workers)
   - Queue-based processing (Kafka)
   - GPU acceleration
   - Pre-transcode common formats
   - Priority queue for popular videos

2. "How do you optimize video delivery?"
   - CDN with edge caching
   - Adaptive bitrate streaming
   - Chunking for progressive loading
   - Pre-position popular videos
   - Multiple CDN providers for redundancy

3. "What about live streaming?"
   - WebRTC or RTMP for ingestion
   - Real-time transcoding
   - CDN with live streaming support
   - Lower latency requirements
   - Separate infrastructure

4. "How would you implement recommendations?"
   - Collaborative filtering
   - Content-based filtering
   - Deep learning models
   - Pre-compute recommendations
   - Update based on recent views

5. "What about video analytics?"
   - Track views, watch time, engagement
   - Store in time-series database
   - Real-time analytics pipeline (Kafka → Spark)
   - Dashboards for creators`,
					CodeExamples: `Video Transcoding Pipeline:

def transcode_video(video_id, raw_video_url):
    # Extract metadata
    metadata = ffmpeg.probe(raw_video_url)
    duration = metadata['duration']
    resolution = metadata['streams'][0]['resolution']
    
    # Transcode to multiple qualities
    qualities = [
        {'name': '240p', 'height': 240, 'bitrate': '400k'},
        {'name': '360p', 'height': 360, 'bitrate': '800k'},
        {'name': '720p', 'height': 720, 'bitrate': '2500k'},
        {'name': '1080p', 'height': 1080, 'bitrate': '5000k'},
        {'name': '4K', 'height': 2160, 'bitrate': '20000k'}
    ]
    
    transcoded = {}
    for quality in qualities:
        output_url = ffmpeg.transcode(
            raw_video_url,
            height=quality['height'],
            bitrate=quality['bitrate'],
            codec='h264'
        )
        cdn_url = cdn.upload(output_url, f"videos/{video_id}/{quality['name']}")
        transcoded[quality['name']] = cdn_url
    
    # Generate thumbnails
    thumbnails = generate_thumbnails(raw_video_url, count=3)
    
    # Update database
    db.update_video(video_id, {
        'status': 'ready',
        'duration': duration,
        'qualities': transcoded,
        'thumbnails': thumbnails
    })
    
    # Index in search
    search_service.index_video(video_id, metadata)

Adaptive Bitrate Streaming:

def get_video_stream(video_id, user_bandwidth):
    # Get available qualities
    video = db.get_video(video_id)
    qualities = video['qualities']
    
    # Select quality based on bandwidth
    if user_bandwidth > 5000:  # 5 Mbps
        quality = '1080p'
    elif user_bandwidth > 2500:  # 2.5 Mbps
        quality = '720p'
    elif user_bandwidth > 800:  # 800 Kbps
        quality = '360p'
    else:
        quality = '240p'
    
    # Return stream URL
    return {
        'stream_url': qualities[quality],
        'quality': quality,
        'chunk_size': 10  # seconds per chunk
    }

Search Implementation:

def search_videos(query, user_id=None, limit=20):
    # Query Elasticsearch
    results = elasticsearch.search({
        'query': {
            'multi_match': {
                'query': query,
                'fields': ['title^3', 'description', 'tags']
            }
        },
        'size': limit
    })
    
    # Get video IDs
    video_ids = [r['_id'] for r in results['hits']['hits']]
    
    # Get full video data
    videos = db.get_videos(video_ids)
    
    # Apply personalization (if logged in)
    if user_id:
        videos = personalize_results(videos, user_id)
    
    return videos

Recommendation Algorithm:

def get_recommendations(user_id, limit=20):
    # Get user viewing history
    history = db.get_viewing_history(user_id, limit=100)
    watched_video_ids = [h['video_id'] for h in history]
    
    # Collaborative filtering
    similar_users = find_similar_users(user_id, watched_video_ids)
    recommended_videos = get_videos_from_users(similar_users, exclude=watched_video_ids)
    
    # Content-based filtering
    content_based = get_similar_videos(watched_video_ids, exclude=watched_video_ids)
    
    # ML-based recommendations
    ml_recommendations = ml_model.predict(user_id, limit=limit)
    
    # Combine and rank
    all_recommendations = combine_recommendations(
        collaborative=recommended_videos,
        content_based=content_based,
        ml=ml_recommendations
    )
    
    return rank_recommendations(all_recommendations, limit=limit)

Architecture Diagram:

[Client] → [API Gateway] → [Upload Service]
                              |
                              v
                    [Object Storage (S3)]
                              |
                              v
                    [Kafka: transcoding_queue]
                              |
                              v
                    [Transcoding Workers]
                              |
                              v
                    [CDN] ←→ [Video Delivery]
                              |
                              v
                    [Video Service] → [Database]
                              |
                              v
                    [Search Service] → [Elasticsearch]
                              |
                              v
                    [Recommendation Service] → [ML Models]

Capacity Summary:

Video Storage: 2,365 PB/year
- Raw videos: 432 TB/day
- Transcoded (5 tiers): 2.16 PB/day
- With replication: 6.48 PB/day
- Yearly: 2,365 PB

CDN Bandwidth: 23.2 Tbps
- Views: 580K/sec × 5 MB = 2.9 TB/s
- 95% cache hit: Origin load = 145 GB/s

Transcoding: 500 videos/sec
- Average video: 10 minutes
- Transcoding time: 5x real-time
- Need: 500 × 10 × 5 = 25,000 minutes/sec = 417 workers
- With 3x headroom: ~1,250 workers`,
				},
				{
					Title: "Design Netflix",
					Content: `**Complete Interview Walkthrough Using RESHADED Framework**

**Problem Statement:**
Design Netflix - a video streaming platform where subscribers can stream movies and TV shows with personalized recommendations.

**R - Requirements Clarification:**

**Questions to Ask:**
1. "What's the expected scale? How many subscribers and concurrent streams?"
2. "What video qualities do we need to support?"
3. "Do we need global distribution?"
4. "What's the acceptable video startup latency?"
5. "How important are personalized recommendations?"
6. "Do we need to support downloads for offline viewing?"
7. "What about multiple user profiles per account?"

**Assumptions (After Clarification):**
- 250M+ subscribers globally
- Peak concurrent streams: 100M
- Support multiple quality tiers (SD, HD, 4K)
- Global distribution required
- < 2 seconds startup latency
- Highly personalized recommendations
- Support offline downloads
- Multiple profiles per account

**E - Estimate (Capacity Planning):**

**Traffic Estimation:**
- Subscribers: 250M
- Peak concurrent: 100M streams
- Average stream: 5 Mbps
- Peak bandwidth: 100M × 5 Mbps = 500 Tbps
- With CDN (98% cache hit): Origin load = 500 Tbps × 0.02 = 10 Tbps

**Storage Estimation:**
- Content library: 10,000 titles
- Average title: 2 hours, 5 GB (HD)
- Total storage: 10,000 × 5 GB = 50 TB
- With multiple qualities (3 tiers): 50 TB × 3 = 150 TB
- With replication (3x): 450 TB
- Metadata: 10,000 × 10 KB = 100 MB

**Bandwidth Estimation:**
- Peak streaming: 500 Tbps (mostly from CDN)
- Origin bandwidth: 10 Tbps
- Upload (content ingestion): Minimal (batch uploads)

**S - System Interface (API Design):**

**Endpoints:**

GET /api/v1/catalog
Response:
{
  "titles": [
    {
      "title_id": "abc123",
      "name": "Stranger Things",
      "type": "series",
      "genres": ["Sci-Fi", "Horror"],
      "thumbnail_url": "https://...",
      "stream_url": "https://..."
    }
  ]
}

GET /api/v1/titles/{title_id}/stream?quality=hd
Response: Video stream (chunked)

GET /api/v1/recommendations?user_id=123&profile_id=456
Response:
{
  "recommendations": [
    {
      "title_id": "abc123",
      "score": 0.95,
      "reason": "Because you watched..."
    }
  ]
}

POST /api/v1/profiles/{profile_id}/watch_history
GET /api/v1/profiles/{profile_id}/watchlist

**H - High-Level Design:**

**Architecture:**
    Client → [Open Connect CDN] → Origin Servers
              |
              v
        [API Gateway (Zuul)] → [Microservices]
              |
              v
        [Service Discovery (Eureka)]
              |
              v
        [Data Layer: MySQL + Cassandra + Caches]
              |
              v
        [Analytics: Kafka → Spark/Flink]

**Components:**
1. **Open Connect CDN**: Proprietary CDN in ISP networks
2. **API Gateway (Zuul)**: Route requests to microservices
3. **Playback Service**: Handle video streaming
4. **Recommendation Service**: Generate personalized recommendations
5. **Catalog Service**: Manage content catalog
6. **User Service**: Manage accounts and profiles
7. **Analytics Service**: Track viewing behavior
8. **Database**: MySQL (critical data), Cassandra (high-volume)
9. **Cache**: EVCache for frequently accessed data

**A - Algorithm/Data Structure:**

**CDN Strategy:**
- **Open Connect**: Deploy servers in ISP networks
- **Pre-positioning**: Pre-load popular content
- **98% Cache Hit**: Most content served from edge
- **Adaptive Bitrate**: Adjust quality based on bandwidth

**Recommendation Algorithm:**
- **Collaborative Filtering**: Users with similar tastes
- **Content-Based**: Similar titles based on metadata
- **Deep Learning**: Neural networks for embeddings
- **Real-time Updates**: Update based on recent views
- **A/B Testing**: Test different algorithms

**D - Detailed Design:**

**Database Schema:**

CREATE TABLE titles (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255),
    type VARCHAR(20),  -- movie, series
    genres JSON,
    description TEXT,
    release_date DATE,
    rating VARCHAR(10),
    thumbnail_url TEXT,
    created_at TIMESTAMP,
    INDEX idx_genres (genres),
    INDEX idx_release (release_date)
);

CREATE TABLE profiles (
    id BIGINT PRIMARY KEY,
    account_id BIGINT NOT NULL,
    name VARCHAR(255),
    avatar_url TEXT,
    created_at TIMESTAMP,
    INDEX idx_account (account_id)
);

CREATE TABLE watch_history (
    profile_id BIGINT,
    title_id VARCHAR(50),
    watched_seconds INT,
    completed BOOLEAN,
    rating INT,  -- 1-5 stars
    created_at TIMESTAMP,
    PRIMARY KEY (profile_id, title_id),
    INDEX idx_profile_created (profile_id, created_at DESC)
);

**Video Streaming Flow:**

1. User selects title
2. Client → Request stream from CDN
3. CDN → Check cache (98% hit rate)
4. If cache hit: Serve from edge
5. If cache miss: Fetch from origin, cache in CDN
6. Adaptive bitrate: Adjust quality based on bandwidth
7. Analytics → Track viewing progress

**Recommendation Flow:**

1. User requests recommendations
2. Recommendation Service → Get user viewing history
3. Recommendation Service → Get similar users (collaborative filtering)
4. Recommendation Service → Get similar titles (content-based)
5. Recommendation Service → ML model prediction
6. Combine and rank recommendations
7. Return personalized list

**E - Evaluation (Bottlenecks & Optimizations):**

**Potential Bottlenecks:**

1. **CDN Bandwidth**: 500 Tbps peak
   - Solution: Open Connect CDN (98% cache hit)
   - Pre-position popular content
   - Multiple CDN providers for redundancy

2. **Origin Bandwidth**: 10 Tbps (2% of traffic)
   - Solution: Optimize origin servers
   - Use efficient codecs
   - Compress content

3. **Recommendation Computation**: Expensive ML models
   - Solution: Pre-compute recommendations
   - Update periodically (every few hours)
   - Cache recommendations
   - Use approximate algorithms

4. **Database Queries**: High read volume
   - Solution: Read replicas
   - Caching (EVCache)
   - Database sharding
   - Use Cassandra for high-volume data

5. **Microservices Coordination**: Network latency
   - Solution: Service mesh (Istio)
   - Circuit breakers
   - Retry logic
   - Caching at service level

**Optimizations:**
- **CDN**: 98% cache hit rate reduces origin load
- **Caching**: EVCache for frequently accessed data
- **Pre-computation**: Pre-compute recommendations
- **Adaptive Bitrate**: Optimize bandwidth usage
- **Compression**: Use efficient codecs (AV1, H.265)
- **Database Sharding**: Shard by region or content type

**D - Discussion (Trade-offs, Scaling, Reliability):**

**Trade-offs:**

1. **CDN vs Origin:**
   - CDN: Lower latency, higher cost
   - Origin: Lower cost, higher latency
   - **Choice**: CDN for delivery (Open Connect)

2. **Microservices vs Monolith:**
   - Microservices: Independent scaling, complexity
   - Monolith: Simpler, harder to scale
   - **Choice**: Microservices for Netflix scale

3. **Recommendation Freshness:**
   - Real-time: Up-to-date, expensive
   - Batch: Delayed, cheaper
   - **Choice**: Near-real-time (update every few hours)

4. **Storage vs Compute:**
   - Pre-compute: Higher storage, faster queries
   - On-demand: Lower storage, slower queries
   - **Choice**: Pre-compute recommendations

**Scaling Strategy:**

**Phase 1 (Current):**
- Open Connect CDN
- Microservices architecture
- Single database cluster

**Phase 2 (10x scale):**
- Expand CDN footprint
- More microservices
- Database sharding
- Advanced caching

**Phase 3 (100x scale):**
- Global CDN network
- Edge computing
- Advanced ML models
- Real-time recommendations

**Reliability:**
- CDN redundancy (multiple providers)
- Database replication
- Microservices health checks
- Circuit breakers
- Graceful degradation

**Common Follow-up Questions:**

1. "How does Open Connect CDN work?"
   - Deploy servers in ISP networks
   - Pre-position popular content
   - 98% cache hit rate
   - Reduces backbone egress costs
   - Lower latency for users

2. "How do you handle microservices at scale?"
   - Service discovery (Eureka)
   - API Gateway (Zuul)
   - Service mesh (Istio)
   - Circuit breakers
   - Health checks and auto-scaling

3. "How do you ensure recommendations are accurate?"
   - A/B testing different algorithms
   - Track user engagement
   - Update models based on feedback
   - Use ensemble methods
   - Personalize per profile

4. "What about content delivery in different regions?"
   - Regional CDN deployments
   - Pre-position content per region
   - Respect content licensing
   - Optimize for local bandwidth

5. "How do you handle peak traffic (new releases)?"
   - Pre-position content before release
   - Scale CDN capacity
   - Use multiple CDN providers
   - Throttle if needed`,
					CodeExamples: `CDN Request Flow:

def get_video_stream(title_id, quality, user_location):
    # Determine CDN server based on user location
    cdn_server = get_nearest_cdn_server(user_location)
    
    # Check cache
    cache_key = f"{title_id}:{quality}"
    if cdn_server.has_cache(cache_key):
        return cdn_server.get_stream(cache_key)
    
    # Cache miss - fetch from origin
    origin_url = get_origin_url(title_id, quality)
    stream = origin.fetch(origin_url)
    
    # Cache in CDN
    cdn_server.cache(cache_key, stream)
    
    return stream

Recommendation Service:

def get_recommendations(profile_id, limit=20):
    # Get viewing history
    history = cassandra.get_watch_history(profile_id, limit=100)
    watched_titles = [h['title_id'] for h in history]
    
    # Collaborative filtering
    similar_profiles = find_similar_profiles(profile_id, watched_titles)
    collaborative_recs = get_titles_from_profiles(similar_profiles, exclude=watched_titles)
    
    # Content-based filtering
    content_recs = get_similar_titles(watched_titles, exclude=watched_titles)
    
    # ML-based recommendations
    ml_recs = ml_model.predict(profile_id, limit=limit)
    
    # Combine and rank
    all_recs = combine_recommendations(
        collaborative=collaborative_recs,
        content_based=content_recs,
        ml=ml_recs
    )
    
    return rank_recommendations(all_recs, limit=limit)

Microservices Communication:

# Service Discovery
def get_service_url(service_name):
    instances = eureka.get_instances(service_name)
    # Load balance
    instance = load_balancer.select(instances)
    return instance.url

# Circuit Breaker
def call_service(service_name, endpoint):
    breaker = circuit_breaker.get(service_name)
    try:
        return breaker.call(lambda: http.get(get_service_url(service_name) + endpoint))
    except CircuitBreakerOpenError:
        # Fallback
        return get_cached_response(endpoint)

Architecture Diagram:

[Client] → [Open Connect CDN] (98% cache hit)
              |
              v (2% miss)
        [Origin Servers]
              |
              v
        [API Gateway (Zuul)]
              |
              v
        [Microservices]
        - Playback Service
        - Recommendation Service
        - Catalog Service
        - User Service
              |
              v
        [Service Discovery (Eureka)]
              |
              v
        [Data Layer]
        - MySQL (critical data)
        - Cassandra (high-volume)
        - EVCache (frequently accessed)
              |
              v
        [Analytics]
        - Kafka → Spark/Flink → ML Models

Capacity Summary:

Peak Bandwidth: 500 Tbps
- Concurrent streams: 100M × 5 Mbps = 500 Tbps
- CDN (98% hit): 490 Tbps from CDN
- Origin (2% miss): 10 Tbps from origin

Storage: 450 TB
- Content library: 150 TB (3 quality tiers)
- With replication: 450 TB

Recommendations: Pre-computed
- Update every 6 hours
- Cache per profile
- ~1 KB per recommendation list`,
				},
				{
					Title: "Design Uber",
					Content: `**Complete Interview Walkthrough Using RESHADED Framework**

**Problem Statement:**
Design Uber - a ride-sharing platform that matches riders with nearby drivers in real-time, tracks locations, calculates ETAs, and processes payments.

**R - Requirements Clarification:**

**Questions to Ask:**
1. "What's the expected scale? How many riders, drivers, and rides per day?"
2. "How frequently do we need location updates?"
3. "What's the acceptable matching latency?"
4. "Do we need to support multiple ride types (UberX, UberXL, etc.)?"
5. "How do we handle surge pricing?"
6. "What about ride sharing (UberPool)?"
7. "Do we need real-time tracking during the ride?"

**Assumptions (After Clarification):**
- 100M riders, 5M drivers
- 10M rides/day
- Location updates every 5 seconds
- < 5 seconds matching latency
- Support multiple ride types
- Dynamic surge pricing
- Real-time tracking during rides

**E - Estimate (Capacity Planning):**

**Traffic Estimation:**
- Rides: 10M/day = ~116 rides/sec average
- Peak: 10x average = ~1,160 rides/sec
- Location updates: 5M drivers × 12 updates/min = 60M updates/min = 1M updates/sec
- Peak updates: 2M updates/sec

**Storage Estimation:**
- Ride data: 10M rides/day × 1 KB = 10 GB/day
- Location data: 1M updates/sec × 100 bytes = 100 MB/sec = 8.64 TB/day
- 1 year retention: 10 GB × 365 = 3.65 TB (rides), 8.64 TB × 365 = 3.15 PB (locations)
- Location data can be archived after 30 days

**Bandwidth Estimation:**
- Location updates: 2M updates/sec × 100 bytes = 200 MB/s = 1.6 Gbps
- WebSocket connections: 5M drivers × 1 KB heartbeat = 5 GB/s = 40 Gbps
- API requests: 1,160 rides/sec × 10 KB = 11.6 MB/s = 93 Mbps

**S - System Interface (API Design):**

**Endpoints:**

POST /api/v1/rides/request
Request:
{
  "rider_id": 123,
  "pickup_location": {"lat": 37.7749, "lng": -122.4194},
  "dropoff_location": {"lat": 37.7849, "lng": -122.4094},
  "ride_type": "uberx"
}

Response:
{
  "ride_id": "abc123",
  "driver_id": 456,
  "driver_location": {"lat": 37.7750, "lng": -122.4195},
  "eta": 5,  // minutes
  "price": 25.50
}

POST /api/v1/drivers/{driver_id}/location
Request:
{
  "location": {"lat": 37.7749, "lng": -122.4194},
  "status": "available",  // available, on_trip, offline
  "timestamp": "2024-01-17T10:00:00Z"
}

WebSocket: ws://api.uber.com/ws?driver_id=456
- Real-time location updates
- Ride requests
- Trip updates

**H - High-Level Design:**

**Architecture:**
    Mobile App → API Gateway → [Ride Service, Matching Service, Location Service]
                              ↓
                        [Message Queue (Kafka)]
                              ↓
                    [DISCO Service, ETA Service, Payment Service]
                              ↓
                    [Database, Time-Series DB, WebSocket Servers]

**Components:**
1. **Ride Service**: Handle ride requests
2. **Matching Service (DISCO)**: Match riders with drivers
3. **Location Service**: Track and update locations
4. **ETA Service**: Calculate estimated arrival times
5. **Surge Service**: Calculate surge pricing
6. **Payment Service**: Process payments
7. **WebSocket Servers**: Real-time communication
8. **Database**: Store rides, users (SQL)
9. **Time-Series DB**: Store location data (InfluxDB, TimescaleDB)

**A - Algorithm/Data Structure:**

**Geo-spatial Indexing (S2):**
- **S2 Cells**: Divide world into hierarchical cells
- **Level 13**: ~100m × 100m cells (good for matching)
- **Level 15**: ~10m × 10m cells (good for tracking)
- **Query**: Find drivers in same or adjacent cells

**Matching Algorithm:**
- **Proximity**: Find drivers within radius (e.g., 5 km)
- **Availability**: Only available drivers
- **Ride Type**: Match ride type with driver capability
- **ETA**: Consider traffic, distance
- **Optimization**: Minimize total wait time

**Surge Pricing:**
- **Demand/Supply Ratio**: Calculate ratio per cell
- **Thresholds**: Surge when ratio > 1.5
- **Multiplier**: 1.0x to 3.0x based on ratio
- **Update Frequency**: Every 5 minutes

**D - Detailed Design:**

**Database Schema:**

CREATE TABLE rides (
    id VARCHAR(50) PRIMARY KEY,
    rider_id BIGINT NOT NULL,
    driver_id BIGINT,
    pickup_location POINT,
    dropoff_location POINT,
    ride_type VARCHAR(20),
    status VARCHAR(20),  -- requested, matched, in_progress, completed, cancelled
    price DECIMAL(10,2),
    surge_multiplier DECIMAL(3,2) DEFAULT 1.0,
    requested_at TIMESTAMP,
    matched_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    INDEX idx_rider_status (rider_id, status),
    INDEX idx_driver_status (driver_id, status)
);

CREATE TABLE drivers (
    id BIGINT PRIMARY KEY,
    name VARCHAR(255),
    vehicle_type VARCHAR(50),
    current_location POINT,
    status VARCHAR(20),  -- available, on_trip, offline
    s2_cell_id BIGINT,  -- Level 13 cell
    updated_at TIMESTAMP,
    SPATIAL INDEX idx_location (current_location),
    INDEX idx_cell_status (s2_cell_id, status)
);

CREATE TABLE location_updates (
    driver_id BIGINT,
    location POINT,
    timestamp TIMESTAMP,
    PRIMARY KEY (driver_id, timestamp),
    INDEX idx_timestamp (timestamp)
) PARTITION BY RANGE (timestamp);

**Ride Request Flow:**

1. Rider requests ride
2. Ride Service → Get rider location
3. Ride Service → Get S2 cell for location
4. Matching Service → Query nearby drivers (same/adjacent cells)
5. Matching Service → Filter by availability and ride type
6. Matching Service → Calculate ETAs for each driver
7. Matching Service → Select best driver (minimize ETA)
8. Matching Service → Send match to driver (WebSocket)
9. Driver accepts → Update ride status
10. Send confirmation to rider

**Location Update Flow:**

1. Driver app sends location update (every 5 seconds)
2. Location Service → Receive update
3. Location Service → Update driver's current_location
4. Location Service → Update S2 cell if changed
5. Location Service → Store in time-series DB
6. Location Service → Publish to Kafka
7. ETA Service → Recalculate ETAs if needed

**E - Evaluation (Bottlenecks & Optimizations):**

**Potential Bottlenecks:**

1. **Location Updates**: 2M updates/sec
   - Solution: Time-series database (InfluxDB, TimescaleDB)
   - Batch writes
   - Partition by time
   - Archive old data

2. **Matching Queries**: 1,160 rides/sec
   - Solution: Geo-spatial index (S2)
   - Cache driver locations
   - Pre-filter by cell
   - Parallel queries

3. **WebSocket Connections**: 5M concurrent
   - Solution: WebSocket server clusters
   - Load balancing with sticky sessions
   - Horizontal scaling
   - Connection pooling

4. **ETA Calculation**: Expensive (traffic data)
   - Solution: Cache ETAs
   - Pre-compute common routes
   - Use approximate algorithms
   - Update periodically

5. **Surge Calculation**: Need real-time demand/supply
   - Solution: Pre-compute per cell
   - Update every 5 minutes
   - Cache surge multipliers

**Optimizations:**
- **Geo-spatial Index**: S2 cells for fast queries
- **Caching**: Cache driver locations, ETAs
- **Batch Processing**: Batch location updates
- **Pre-computation**: Pre-compute ETAs, surge
- **Partitioning**: Partition by geographic region
- **Database Sharding**: Shard by region

**D - Discussion (Trade-offs, Scaling, Reliability):**

**Trade-offs:**

1. **Precision vs Performance:**
   - Fine-grained cells: More precision, slower queries
   - Coarse cells: Less precision, faster queries
   - **Choice**: Level 13 for matching (good balance)

2. **Consistency vs Availability:**
   - Strong consistency: Accurate locations, slower
   - Eventual consistency: Slightly stale, faster
   - **Choice**: Eventual consistency (acceptable for locations)

3. **Real-time vs Batch:**
   - Real-time matching: Lower latency, higher cost
   - Batch matching: Higher latency, lower cost
   - **Choice**: Real-time matching (critical for UX)

4. **Storage vs Compute:**
   - Store all locations: High storage, fast queries
   - Compute on-demand: Lower storage, slower queries
   - **Choice**: Store recent locations (30 days), archive older

**Scaling Strategy:**

**Phase 1 (Current):**
- Single database
- Basic geo-spatial queries
- Single WebSocket cluster

**Phase 2 (10x scale):**
- Database sharding by region
- Optimized S2 queries
- WebSocket clusters per region
- Time-series DB for locations

**Phase 3 (100x scale):**
- Multi-region deployment
- Edge computing for matching
- Advanced traffic prediction
- ML-based matching

**Reliability:**
- Database replication
- WebSocket reconnection logic
- Health checks and auto-failover
- Graceful degradation (fallback matching)
- Circuit breakers for external services

**Common Follow-up Questions:**

1. "How do you handle the matching algorithm?"
   - Use S2 cells for proximity queries
   - Filter by availability and ride type
   - Calculate ETAs considering traffic
   - Select driver minimizing total wait time
   - Consider driver preferences

2. "What about surge pricing?"
   - Calculate demand/supply ratio per cell
   - Update every 5 minutes
   - Apply multiplier (1.0x to 3.0x)
   - Show to riders before request
   - Dynamic adjustment

3. "How do you ensure low matching latency?"
   - Pre-index drivers by S2 cell
   - Cache driver locations
   - Parallel queries
   - Optimize database queries
   - Use in-memory data structures

4. "What about ride sharing (UberPool)?"
   - Match multiple riders going same direction
   - Optimize route for all riders
   - Split fare among riders
   - More complex matching algorithm
   - Real-time route updates

5. "How do you handle driver availability?"
   - Track driver status (available, on_trip, offline)
   - Update in real-time
   - Filter unavailable drivers from matching
   - Handle driver going offline during match`,
					CodeExamples: `S2 Cell Implementation:

def get_s2_cell(lat, lng, level=13):
    # Convert lat/lng to S2 cell
    point = s2.LatLng(lat, lng)
    cell_id = s2.CellId.from_lat_lng(point).parent(level)
    return cell_id.id()

def find_nearby_drivers(pickup_location, radius_km=5):
    # Get S2 cell for pickup location
    cell_id = get_s2_cell(pickup_location['lat'], pickup_location['lng'])
    
    # Get adjacent cells (for radius coverage)
    adjacent_cells = s2.get_adjacent_cells(cell_id)
    all_cells = [cell_id] + adjacent_cells
    
    # Query drivers in these cells
    drivers = db.query("""
        SELECT * FROM drivers
        WHERE s2_cell_id IN (?)
        AND status = 'available'
    """, all_cells)
    
    # Filter by actual distance (S2 cells are approximate)
    nearby_drivers = []
    for driver in drivers:
        distance = calculate_distance(pickup_location, driver['current_location'])
        if distance <= radius_km * 1000:  # Convert to meters
            nearby_drivers.append(driver)
    
    return nearby_drivers

Matching Algorithm:

def match_rider_to_driver(ride_request):
    # Get nearby drivers
    drivers = find_nearby_drivers(ride_request['pickup_location'])
    
    if not drivers:
        return None
    
    # Calculate ETAs for each driver
    driver_etas = []
    for driver in drivers:
        eta = calculate_eta(
            driver['current_location'],
            ride_request['pickup_location'],
            ride_request['dropoff_location']
        )
        driver_etas.append({
            'driver': driver,
            'eta': eta
        })
    
    # Select driver with minimum ETA
    best_driver = min(driver_etas, key=lambda x: x['eta'])
    
    # Calculate price
    price = calculate_price(
        ride_request['pickup_location'],
        ride_request['dropoff_location'],
        ride_request['ride_type'],
        surge_multiplier=get_surge_multiplier(ride_request['pickup_location'])
    )
    
    # Create ride
    ride = create_ride(ride_request, best_driver['driver']['id'], price)
    
    # Send match to driver via WebSocket
    websocket.send(best_driver['driver']['id'], {
        'type': 'ride_match',
        'ride_id': ride['id'],
        'rider_location': ride_request['pickup_location']
    })
    
    return ride

Surge Pricing:

def calculate_surge_multiplier(location):
    # Get S2 cell
    cell_id = get_s2_cell(location['lat'], location['lng'])
    
    # Get demand (ride requests in last 5 minutes)
    demand = db.query("""
        SELECT COUNT(*) FROM rides
        WHERE pickup_location_cell = ?
        AND requested_at > NOW() - INTERVAL 5 MINUTE
        AND status IN ('requested', 'matched')
    """, cell_id)
    
    # Get supply (available drivers)
    supply = db.query("""
        SELECT COUNT(*) FROM drivers
        WHERE s2_cell_id = ?
        AND status = 'available'
    """, cell_id)
    
    # Calculate ratio
    if supply == 0:
        ratio = 999  # No drivers available
    else:
        ratio = demand / supply
    
    # Calculate multiplier
    if ratio <= 1.0:
        multiplier = 1.0
    elif ratio <= 1.5:
        multiplier = 1.2
    elif ratio <= 2.0:
        multiplier = 1.5
    elif ratio <= 3.0:
        multiplier = 2.0
    else:
        multiplier = 3.0
    
    # Cache multiplier
    redis.setex(f"surge:{cell_id}", 300, multiplier)  # 5 min TTL
    
    return multiplier

Architecture Diagram:

[Mobile App] → [API Gateway] → [Ride Service]
                              |
                              v
                    [Kafka: location_updates]
                              |
                              v
                    [Location Service] → [Time-Series DB]
                              |
                              v
                    [Matching Service (DISCO)]
                              |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
[ETA Service]      [Surge Service]    [WebSocket Servers]
        |                   |                   |
        v                   v                   v
[Traffic API]      [Redis Cache]      [Driver/Rider Apps]
        |
        v
[Database] (Sharded by region)

Capacity Summary:

Rides: 1,160/sec
- Matching queries: 1,160/sec
- With S2 index: ~10ms per query
- Need: ~12 matching servers

Location Updates: 2M/sec
- Time-series DB: 2M writes/sec
- Partition by time: 100 partitions = 20K writes/sec per partition
- Need: ~100 time-series DB nodes

WebSocket Connections: 5M concurrent
- 5M connections × 1 KB heartbeat = 5 GB/s
- 1000 servers × 5K connections = 5M connections
- Need: ~1000 WebSocket servers`,
				},
				{
					Title: "Design a Search Engine",
					Content: `**Complete Interview Walkthrough Using RESHADED Framework**

**Problem Statement:**
Design a search engine like Google that can crawl billions of web pages, index them, and return ranked search results in sub-second time.

**R - Requirements Clarification:**

**Questions to Ask:**
1. "What's the expected scale? How many pages to crawl and queries per day?"
2. "What's the acceptable search latency?"
3. "How fresh should the index be? (real-time vs daily updates)"
4. "Do we need to support different content types? (HTML, PDF, images)"
5. "What about image search or other specialized searches?"
6. "Do we need to handle different languages?"
7. "What's the storage budget?"

**Assumptions (After Clarification):**
- Crawl 50B web pages
- 5B queries/day
- < 200ms search latency
- Daily index updates acceptable
- Support HTML, PDF, images
- Multi-language support
- Petabyte-scale storage

**E - Estimate (Capacity Planning):**

**Traffic Estimation:**
- Queries: 5B/day = ~58,000 queries/sec average
- Peak: 10x average = ~580,000 queries/sec
- Crawl rate: 50B pages, refresh monthly = 50B / 30 = 1.67B pages/day
- Crawl rate: 1.67B / 86400 = ~19,000 pages/sec

**Storage Estimation:**
- Average page: 50 KB (HTML + extracted text)
- Content storage: 50B × 50 KB = 2.5 PB
- Index size: ~30% of content = 750 TB
- With replication (3x): 2.5 PB × 3 = 7.5 PB content, 2.25 PB index
- Total: ~10 PB

**Bandwidth Estimation:**
- Crawling: 19,000 pages/sec × 50 KB = 950 MB/s = 7.6 Gbps
- Search: 580K queries/sec × 10 KB response = 5.8 GB/s = 46.4 Gbps
- Use CDN for search results (cache popular queries)

**S - System Interface (API Design):**

**Endpoints:**

GET /api/v1/search?q=query&limit=10&offset=0
Response:
{
  "query": "machine learning",
  "results": [
    {
      "url": "https://example.com/ml",
      "title": "Introduction to Machine Learning",
      "snippet": "Machine learning is a subset of...",
      "rank": 1,
      "score": 0.95
    }
  ],
  "total_results": 1000000,
  "search_time_ms": 150
}

POST /api/v1/admin/crawl
Request:
{
  "urls": ["https://example.com", "https://example2.com"],
  "priority": "high"
}

**H - High-Level Design:**

**Architecture:**
    [Web Crawler] → [URL Frontier] → [Content Store]
                              |
                              v
                    [Indexer] → [Inverted Index] → [Index Servers]
                              |
                              v
                    [Query Service] → [Ranking Service] → [Results]
                              |
                              v
                    [Cache Layer] → [CDN]

**Components:**
1. **Web Crawler**: Fetch web pages
2. **URL Frontier**: Queue of URLs to crawl
3. **Content Store**: Store crawled content
4. **Indexer**: Build inverted index
5. **Inverted Index**: Term → Document mappings
6. **Query Service**: Handle search queries
7. **Ranking Service**: Rank search results
8. **Cache**: Cache popular queries

**A - Algorithm/Data Structure:**

**Inverted Index:**
- **Structure**: term → [doc_id1, doc_id2, ...]
- **Postings List**: List of documents containing term
- **Term Frequency**: How often term appears in document
- **Document Frequency**: How many documents contain term
- **TF-IDF**: Term frequency × Inverse document frequency

**PageRank Algorithm:**
- **Concept**: Importance based on links
- **Formula**: PR(A) = (1-d) + d × Σ(PR(T)/C(T))
- **Iterative**: Calculate until convergence
- **Signals**: Hundreds of ranking factors

**Ranking Factors:**
- **Relevance**: Text match, term positions
- **Authority**: PageRank, domain authority
- **Freshness**: Recency of content
- **User Signals**: Click-through rate, dwell time
- **Content Quality**: Spam detection, readability

**D - Detailed Design:**

**Database Schema:**

CREATE TABLE pages (
    id BIGINT PRIMARY KEY,
    url VARCHAR(2048) UNIQUE,
    title VARCHAR(500),
    content TEXT,
    content_hash VARCHAR(64),  -- For deduplication
    crawled_at TIMESTAMP,
    updated_at TIMESTAMP,
    page_rank DECIMAL(10,6),
    INDEX idx_crawled (crawled_at),
    INDEX idx_updated (updated_at)
);

CREATE TABLE inverted_index (
    term VARCHAR(255),
    doc_id BIGINT,
    term_frequency INT,
    positions JSON,  -- Positions of term in document
    PRIMARY KEY (term, doc_id),
    INDEX idx_term (term),
    INDEX idx_doc (doc_id)
);

CREATE TABLE links (
    from_page_id BIGINT,
    to_page_id BIGINT,
    anchor_text VARCHAR(500),
    PRIMARY KEY (from_page_id, to_page_id),
    INDEX idx_to (to_page_id)
);

**Crawling Flow:**

1. URL Frontier → Get next URL to crawl
2. Crawler → Check robots.txt (cached)
3. Crawler → Fetch page (respect rate limits)
4. Crawler → Parse HTML, extract content
5. Crawler → Extract links
6. Crawler → Store content in Content Store
7. Crawler → Add new URLs to Frontier
8. Crawler → Check for duplicates (content hash)

**Indexing Flow:**

1. Indexer → Read crawled content
2. Indexer → Extract terms (tokenization)
3. Indexer → Build inverted index
4. Indexer → Calculate term frequencies
5. Indexer → Store in Index Servers
6. Indexer → Update PageRank (periodic)

**Search Flow:**

1. User submits query
2. Query Service → Parse query (extract terms)
3. Query Service → Lookup terms in inverted index
4. Query Service → Get candidate documents (intersection)
5. Ranking Service → Calculate scores (TF-IDF, PageRank, signals)
6. Ranking Service → Rank documents
7. Query Service → Return top N results
8. Cache result (if popular query)

**E - Evaluation (Bottlenecks & Optimizations):**

**Potential Bottlenecks:**

1. **Crawling**: 19,000 pages/sec
   - Solution: Distributed crawlers
   - Politeness policies (rate limiting per domain)
   - Parallel crawling
   - Efficient parsing

2. **Indexing**: Large index size (750 TB)
   - Solution: Distributed index servers
   - Shard index by terms
   - Compression
   - Incremental updates

3. **Search Queries**: 580K queries/sec
   - Solution: Distributed index servers
   - Caching (80% hit rate for popular queries)
   - Parallel queries across shards
   - Result aggregation

4. **Ranking Computation**: Expensive
   - Solution: Pre-compute PageRank
   - Cache ranking signals
   - Approximate algorithms
   - Parallel ranking

5. **Storage**: 10 PB
   - Solution: Distributed storage
   - Compression
   - Archive old content
   - Deduplication

**Optimizations:**
- **Caching**: Cache popular queries (80% hit rate)
- **Index Sharding**: Shard by terms (alphabetically)
- **Compression**: Compress index and content
- **Deduplication**: Detect duplicate content
- **Incremental Updates**: Update index incrementally
- **CDN**: Cache search results

**D - Discussion (Trade-offs, Scaling, Reliability):**

**Trade-offs:**

1. **Freshness vs Coverage:**
   - Frequent crawls: Fresh content, higher cost
   - Less frequent: Lower cost, stale content
   - **Choice**: Daily updates for most, real-time for important sites

2. **Precision vs Recall:**
   - High precision: Fewer but relevant results
   - High recall: More results, some irrelevant
   - **Choice**: Balance (aim for high precision in top results)

3. **Storage vs Compute:**
   - Store full content: Higher storage, faster queries
   - Store only index: Lower storage, slower queries
   - **Choice**: Store full content (needed for snippets)

4. **Crawl Politeness:**
   - Aggressive crawling: Faster indexing, may overload servers
   - Polite crawling: Slower indexing, better relationships
   - **Choice**: Polite crawling (respect robots.txt, rate limits)

**Scaling Strategy:**

**Phase 1 (Current):**
- Single crawler cluster
- Single index server
- Basic ranking

**Phase 2 (10x scale):**
- Distributed crawlers
- Sharded index servers
- Advanced ranking (PageRank)

**Phase 3 (100x scale):**
- Global crawler network
- Distributed index clusters
- ML-based ranking
- Real-time updates for important sites

**Reliability:**
- Crawler fault tolerance (retry failed pages)
- Index replication
- Health checks and auto-failover
- Graceful degradation (serve cached results if index down)

**Common Follow-up Questions:**

1. "How do you handle duplicate content?"
   - Content hash (SHA-256) for deduplication
   - Canonical URLs
   - Detect near-duplicates (shingle-based)
   - Prefer original source

2. "How do you rank results?"
   - TF-IDF for relevance
   - PageRank for authority
   - Hundreds of signals (freshness, user engagement, etc.)
   - ML models for personalization

3. "How do you handle different languages?"
   - Language detection
   - Separate indexes per language
   - Translation for cross-language search
   - Language-specific ranking

4. "What about image search?"
   - Extract image metadata (alt text, surrounding text)
   - Image analysis (ML for visual similarity)
   - Separate image index
   - Reverse image search

5. "How do you ensure crawl politeness?"
   - Respect robots.txt
   - Rate limiting per domain
   - Crawl-delay directives
   - Monitor server load`,
					CodeExamples: `Inverted Index Construction:

def build_inverted_index(document):
    # Tokenize document
    terms = tokenize(document['content'])
    
    # Build term → document mapping
    term_frequencies = {}
    term_positions = {}
    
    for i, term in enumerate(terms):
        if term not in term_frequencies:
            term_frequencies[term] = 0
            term_positions[term] = []
        
        term_frequencies[term] += 1
        term_positions[term].append(i)
    
    # Store in inverted index
    for term, freq in term_frequencies.items():
        db.insert_inverted_index({
            'term': term,
            'doc_id': document['id'],
            'term_frequency': freq,
            'positions': term_positions[term]
        })

Search Query Processing:

def search(query, limit=10):
    # Parse query
    terms = tokenize(query)
    
    # Get postings lists for each term
    postings_lists = []
    for term in terms:
        postings = db.get_postings_list(term)
        postings_lists.append(set(postings))
    
    # Intersect postings lists (documents containing all terms)
    if not postings_lists:
        return []
    
    candidate_docs = postings_lists[0]
    for postings in postings_lists[1:]:
        candidate_docs = candidate_docs.intersection(postings)
    
    # Calculate scores for each document
    scored_docs = []
    for doc_id in candidate_docs:
        score = calculate_score(doc_id, terms)
        scored_docs.append({'doc_id': doc_id, 'score': score})
    
    # Rank by score
    scored_docs.sort(key=lambda x: x['score'], reverse=True)
    
    # Get top N documents
    top_docs = [d['doc_id'] for d in scored_docs[:limit]]
    
    # Get full document data
    results = db.get_documents(top_docs)
    
    return results

def calculate_score(doc_id, query_terms):
    # TF-IDF score
    tf_idf_score = 0
    for term in query_terms:
        tf = get_term_frequency(doc_id, term)
        idf = get_inverse_document_frequency(term)
        tf_idf_score += tf * idf
    
    # PageRank score
    page_rank = get_page_rank(doc_id)
    
    # Combine scores
    score = 0.7 * tf_idf_score + 0.3 * page_rank
    
    return score

PageRank Calculation:

def calculate_pagerank(pages, iterations=100, damping=0.85):
    # Initialize PageRank
    pr = {page_id: 1.0 / len(pages) for page_id in pages}
    
    # Iterate
    for _ in range(iterations):
        new_pr = {}
        for page_id in pages:
            # Sum of PR from incoming links
            incoming_sum = 0
            for link in get_incoming_links(page_id):
                from_page = link['from_page_id']
                out_links_count = len(get_outgoing_links(from_page))
                if out_links_count > 0:
                    incoming_sum += pr[from_page] / out_links_count
            
            # PageRank formula
            new_pr[page_id] = (1 - damping) / len(pages) + damping * incoming_sum
        
        pr = new_pr
    
    return pr

Crawler with Politeness:

class PoliteCrawler:
    def __init__(self):
        self.domain_delays = {}  # Track last crawl time per domain
        self.robots_cache = {}  # Cache robots.txt
    
    def crawl_url(self, url):
        domain = get_domain(url)
        
        # Check robots.txt
        if not self.can_crawl(url, domain):
            return None
        
        # Check crawl delay
        if not self.check_crawl_delay(domain):
            return None
        
        # Fetch page
        page = fetch_page(url)
        
        # Update last crawl time
        self.domain_delays[domain] = time.now()
        
        return page
    
    def can_crawl(self, url, domain):
        # Get robots.txt (cached)
        if domain not in self.robots_cache:
            robots = fetch_robots_txt(domain)
            self.robots_cache[domain] = parse_robots_txt(robots)
        
        return self.robots_cache[domain].can_fetch('*', url)
    
    def check_crawl_delay(self, domain):
        if domain not in self.domain_delays:
            return True
        
        delay = self.robots_cache[domain].get_crawl_delay('*')
        if delay is None:
            delay = 1  # Default 1 second
        
        elapsed = time.now() - self.domain_delays[domain]
        return elapsed >= delay

Architecture Diagram:

[URL Frontier] → [Crawler Workers] → [Content Store]
                              |
                              v
                    [Indexer] → [Inverted Index]
                              |
                              v
                    [Index Servers] (Sharded by terms)
                              |
                              v
                    [Query Service] → [Ranking Service]
                              |
                              v
                    [Cache] → [Results]

Capacity Summary:

Crawling: 19K pages/sec
- Distributed crawlers: 1000 crawlers × 19 pages/sec = 19K pages/sec
- Storage: 19K × 50 KB = 950 MB/s

Indexing: 750 TB index
- Sharded by terms: 100 shards × 7.5 TB = 750 TB
- With replication: 2.25 PB

Search: 580K queries/sec
- Cache (80% hit): 464K/sec from cache
- Index (20% miss): 116K/sec from index
- 100 index servers: 1.16K queries/sec per server`,
				},
				{
					Title: "Design a Distributed Cache",
					Content: `**Complete Interview Walkthrough Using RESHADED Framework**

**Problem Statement:**
Design a distributed cache system like Redis that can handle millions of operations per second, distribute data across multiple servers, and handle server failures gracefully.

**R - Requirements Clarification:**

**Questions to Ask:**
1. "What's the expected scale? How many operations per second?"
2. "What's the total cache size needed?"
3. "What data types do we need to support? (strings, lists, sets, etc.)"
4. "Do we need persistence or is in-memory only OK?"
5. "What's the acceptable latency for get/set operations?"
6. "How do we handle consistency? (strong vs eventual)"
7. "Do we need to support transactions?"

**Assumptions (After Clarification):**
- 10M operations/sec
- 100 GB total cache size
- Support strings, lists, sets, hashes
- Optional persistence (RDB snapshots, AOF)
- < 1ms latency for get/set
- Eventual consistency acceptable
- Support basic transactions

**E - Estimate (Capacity Planning):**

**Traffic Estimation:**
- Operations: 10M ops/sec
- Reads: 80% = 8M reads/sec
- Writes: 20% = 2M writes/sec
- Average key size: 1 KB
- Average value size: 10 KB

**Storage Estimation:**
- Total cache: 100 GB
- With replication (3x): 300 GB
- Per server (10 servers): 10 GB per server
- With overhead (30%): 13 GB per server

**Bandwidth Estimation:**
- Reads: 8M reads/sec × 10 KB = 80 GB/s = 640 Gbps
- Writes: 2M writes/sec × 10 KB = 20 GB/s = 160 Gbps
- Total: 100 GB/s = 800 Gbps

**S - System Interface (API Design):**

**Endpoints:**

GET /cache/{key}
Response:
{
  "key": "user:123",
  "value": "{\"name\": \"John\"}",
  "ttl": 3600
}

SET /cache/{key}
Request:
{
  "value": "{\"name\": \"John\"}",
  "ttl": 3600  // Optional
}

DELETE /cache/{key}

**Redis-like Commands:**
- GET key
- SET key value [EX seconds]
- DEL key
- EXISTS key
- TTL key
- INCR key
- LPUSH/RPUSH key value
- SADD key member
- HSET key field value

**H - High-Level Design:**

**Architecture:**
    Client → [Consistent Hashing Router] → [Cache Nodes]
              |
              v
        [Replication Layer] → [Replica Nodes]
              |
              v
        [Persistence Layer] → [Disk Storage]

**Components:**
1. **Cache Nodes**: Store data in memory
2. **Consistent Hashing Router**: Route keys to nodes
3. **Replication Layer**: Replicate data to replicas
4. **Persistence Layer**: Optional disk persistence
5. **Eviction Manager**: Handle memory limits (LRU, LFU)
6. **Cluster Manager**: Handle node addition/removal

**A - Algorithm/Data Structure:**

**Consistent Hashing:**
- **Hash Ring**: Map nodes and keys to circle (0 to 2^32-1)
- **Virtual Nodes**: Each physical node has multiple virtual nodes
- **Key Routing**: Key belongs to first node clockwise
- **Node Addition**: Only nearby keys remap
- **Node Removal**: Keys remap to next node

**Eviction Policies:**
- **LRU (Least Recently Used)**: Evict least recently accessed
- **LFU (Least Frequently Used)**: Evict least frequently accessed
- **TTL-based**: Evict expired keys
- **Random**: Evict random keys (simple but less optimal)

**Replication:**
- **Master-Replica**: Master handles writes, replicas handle reads
- **Synchronous**: Wait for replica confirmation (stronger consistency)
- **Asynchronous**: Don't wait (better performance)

**D - Detailed Design:**

**Data Structures:**

**In-Memory Storage:**
- **Hash Table**: O(1) get/set operations
- **Expiry Index**: Sorted set for TTL management
- **LRU Index**: Doubly linked list for LRU eviction

**Consistent Hashing:**

class ConsistentHash:
    def __init__(self, virtual_nodes=150):
        self.virtual_nodes = virtual_nodes
        self.ring = {}  # hash → node_id
        self.nodes = {}  # node_id → [virtual_node_hashes]
    
    def add_node(self, node_id):
        # Add virtual nodes
        for i in range(self.virtual_nodes):
            virtual_hash = hash(f"{node_id}:{i}") % (2**32)
            self.ring[virtual_hash] = node_id
            if node_id not in self.nodes:
                self.nodes[node_id] = []
            self.nodes[node_id].append(virtual_hash)
        
        # Sort ring for binary search
        self.sorted_hashes = sorted(self.ring.keys())
    
    def get_node(self, key):
        # Hash key
        key_hash = hash(key) % (2**32)
        
        # Find first node >= key_hash (clockwise)
        for hash_val in self.sorted_hashes:
            if hash_val >= key_hash:
                return self.ring[hash_val]
        
        # Wrap around (first node)
        return self.ring[self.sorted_hashes[0]]

**Cache Node Implementation:**

class CacheNode:
    def __init__(self, max_memory=10*1024*1024*1024):  # 10 GB
        self.data = {}  # key → (value, timestamp, access_count)
        self.max_memory = max_memory
        self.current_memory = 0
        self.lru_list = DoublyLinkedList()
    
    def get(self, key):
        if key not in self.data:
            return None
        
        # Update LRU
        self.lru_list.move_to_front(key)
        
        # Update access count
        value, timestamp, count = self.data[key]
        self.data[key] = (value, timestamp, count + 1)
        
        # Check TTL
        if self.is_expired(key):
            self.delete(key)
            return None
        
        return value
    
    def set(self, key, value, ttl=None):
        # Check memory limit
        if self.current_memory + len(value) > self.max_memory:
            self.evict_lru()
        
        # Store
        self.data[key] = (value, time.now(), 0)
        self.lru_list.add_to_front(key)
        self.current_memory += len(value)
        
        # Set TTL
        if ttl:
            self.set_ttl(key, ttl)
    
    def evict_lru(self):
        # Remove least recently used
        lru_key = self.lru_list.remove_from_back()
        if lru_key:
            value, _, _ = self.data[lru_key]
            del self.data[lru_key]
            self.current_memory -= len(value)

**E - Evaluation (Bottlenecks & Optimizations):**

**Potential Bottlenecks:**

1. **Memory Limits**: 100 GB total
   - Solution: Eviction policies (LRU, LFU)
   - Distributed across nodes
   - Monitor memory usage

2. **Network Bandwidth**: 800 Gbps
   - Solution: Multiple network interfaces
   - Compression for large values
   - Local caching at client

3. **Consistent Hashing Overhead**: Hash computation
   - Solution: Cache hash results
   - Efficient hash functions
   - Virtual nodes for better distribution

4. **Replication Lag**: Asynchronous replication
   - Solution: Acceptable for cache (eventual consistency)
   - Monitor replication lag
   - Synchronous for critical data

5. **Node Failures**: Data loss risk
   - Solution: Replication (3x)
   - Automatic failover
   - Health checks

**Optimizations:**
- **Virtual Nodes**: Better key distribution
- **Connection Pooling**: Reuse connections
- **Pipelining**: Batch operations
- **Compression**: Compress large values
- **Local Caching**: Cache at client side
- **Sharding**: Distribute load across nodes

**D - Discussion (Trade-offs, Scaling, Reliability):**

**Trade-offs:**

1. **Consistency vs Availability:**
   - Strong consistency: Slower, but guaranteed
   - Eventual consistency: Faster, acceptable for cache
   - **Choice**: Eventual consistency (cache can be stale)

2. **Memory vs Performance:**
   - More memory: Better hit rate, higher cost
   - Less memory: Lower hit rate, lower cost
   - **Choice**: Balance (evict when full)

3. **Replication:**
   - Synchronous: Stronger consistency, slower
   - Asynchronous: Better performance, eventual consistency
   - **Choice**: Asynchronous (acceptable for cache)

4. **Eviction Policy:**
   - LRU: Good for temporal locality
   - LFU: Good for frequency-based access
   - **Choice**: LRU (simpler, works well)

**Scaling Strategy:**

**Phase 1 (Current):**
- Single cache node
- Basic operations

**Phase 2 (10x scale):**
- Multiple nodes with consistent hashing
- Replication
- Eviction policies

**Phase 3 (100x scale):**
- Virtual nodes
- Advanced eviction (LFU, adaptive)
- Persistence options
- Cluster management

**Reliability:**
- Replication (3x)
- Automatic failover
- Health checks
- Data persistence (optional)
- Graceful degradation

**Common Follow-up Questions:**

1. "How do you handle hot keys?"
   - Detect hot keys (high access frequency)
   - Replicate hot keys to multiple nodes
   - Use local cache at client
   - Shard hot keys

2. "What about cache invalidation?"
   - TTL-based expiration
   - Manual invalidation (DELETE)
   - Event-based invalidation
   - Version-based keys

3. "How do you ensure data consistency?"
   - Eventual consistency (acceptable for cache)
   - Version numbers for conflict resolution
   - Last-write-wins
   - Strong consistency for critical data

4. "What about persistence?"
   - RDB snapshots (periodic)
   - AOF (append-only file)
   - Hybrid approach
   - Trade-off: Performance vs durability

5. "How do you handle node failures?"
   - Replication (data on multiple nodes)
   - Automatic failover
   - Health checks
   - Rebalance keys after failure`,
					CodeExamples: `Consistent Hashing Implementation:

class ConsistentHashRing:
    def __init__(self, nodes=None, virtual_nodes=150):
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []
        
        if nodes:
            for node in nodes:
                self.add_node(node)
    
    def add_node(self, node):
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_val = self._hash(virtual_key)
            self.ring[hash_val] = node
            self.sorted_keys.append(hash_val)
        
        self.sorted_keys.sort()
    
    def remove_node(self, node):
        keys_to_remove = []
        for hash_val, node_id in self.ring.items():
            if node_id == node:
                keys_to_remove.append(hash_val)
        
        for hash_val in keys_to_remove:
            del self.ring[hash_val]
            self.sorted_keys.remove(hash_val)
    
    def get_node(self, key):
        if not self.ring:
            return None
        
        hash_val = self._hash(key)
        
        # Binary search for first node >= hash_val
        idx = bisect.bisect_left(self.sorted_keys, hash_val)
        
        if idx == len(self.sorted_keys):
            idx = 0  # Wrap around
        
        return self.ring[self.sorted_keys[idx]]
    
    def _hash(self, key):
        return hash(key) % (2**32)

LRU Eviction:

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def get(self, key):
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        self._move_to_front(node)
        return node.value
    
    def set(self, key, value):
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                self._evict_lru()
            
            node = Node(key, value)
            self.cache[key] = node
            self._add_to_front(node)
    
    def _move_to_front(self, node):
        self._remove_node(node)
        self._add_to_front(node)
    
    def _evict_lru(self):
        lru = self.tail.prev
        self._remove_node(lru)
        del self.cache[lru.key]

Replication:

class ReplicatedCache:
    def __init__(self, nodes, replication_factor=3):
        self.nodes = nodes
        self.replication_factor = replication_factor
        self.hash_ring = ConsistentHashRing(nodes)
    
    def set(self, key, value):
        # Get primary node
        primary_node = self.hash_ring.get_node(key)
        
        # Get replica nodes
        replica_nodes = self._get_replicas(key, primary_node)
        
        # Write to primary and replicas (async)
        primary_node.set(key, value)
        for replica in replica_nodes:
            replica.set_async(key, value)  # Async replication
    
    def get(self, key):
        # Get primary node
        primary_node = self.hash_ring.get_node(key)
        
        # Try primary first
        value = primary_node.get(key)
        if value:
            return value
        
        # Try replicas
        replica_nodes = self._get_replicas(key, primary_node)
        for replica in replica_nodes:
            value = replica.get(key)
            if value:
                return value
        
        return None
    
    def _get_replicas(self, key, primary_node):
        # Get next N nodes for replication
        replicas = []
        current_hash = self.hash_ring._hash(key)
        sorted_nodes = sorted(self.hash_ring.sorted_keys)
        
        idx = sorted_nodes.index(current_hash)
        for i in range(1, self.replication_factor):
            next_idx = (idx + i) % len(sorted_nodes)
            replica_node = self.hash_ring.ring[sorted_nodes[next_idx]]
            if replica_node != primary_node:
                replicas.append(replica_node)
        
        return replicas

Architecture Diagram:

[Client] → [Consistent Hashing Router]
              |
              v
        [Cache Node 1] ←→ [Replica 1-1, Replica 1-2]
        [Cache Node 2] ←→ [Replica 2-1, Replica 2-2]
        [Cache Node 3] ←→ [Replica 3-1, Replica 3-2]
              |
              v
        [Persistence Layer] → [Disk Storage]

Capacity Summary:

Operations: 10M/sec
- Reads: 8M/sec
- Writes: 2M/sec
- 10 nodes: 1M ops/sec per node

Storage: 100 GB total
- 10 nodes: 10 GB per node
- With replication (3x): 30 GB per node
- With overhead: ~40 GB per node

Bandwidth: 800 Gbps
- 10 nodes: 80 Gbps per node
- Multiple network interfaces per node`,
				},
				{
					Title: "Design a Rate Limiter",
					Content: `**Complete Interview Walkthrough Using RESHADED Framework**

**Problem Statement:**
Design a rate limiter that can limit the number of requests per user/IP/API key, support different rate limits, work across multiple servers, and handle millions of requests per second with low latency.

**R - Requirements Clarification:**

**Questions to Ask:**
1. "What's the expected scale? How many requests per second?"
2. "What rate limits do we need? (per minute, per hour, per day)"
3. "Do we need different limits for different users/tiers?"
4. "What's the acceptable latency overhead?"
5. "Do we need distributed rate limiting (across multiple servers)?"
6. "What happens when limit is exceeded? (429 error, queue, throttle)"
7. "Do we need to support burst traffic?"

**Assumptions (After Clarification):**
- 10M requests/sec
- Rate limits: 100/min, 1000/hour per user
- Different tiers (free: 100/min, paid: 1000/min)
- < 1ms latency overhead
- Distributed across multiple servers
- Return 429 Too Many Requests when exceeded
- Support burst traffic (token bucket)

**E - Estimate (Capacity Planning):**

**Traffic Estimation:**
- Requests: 10M requests/sec
- Rate limit checks: 10M checks/sec
- Average key size: 50 bytes (user:123, ip:1.2.3.4)
- Average value size: 8 bytes (counter)

**Storage Estimation:**
- Active users: 10M (checking limits)
- Per key: 50 bytes key + 8 bytes value = 58 bytes
- Total: 10M × 58 bytes = 580 MB
- With TTL expiration: Most keys expire, actual storage ~100 MB
- Redis memory: ~200 MB (with overhead)

**Bandwidth Estimation:**
- Redis operations: 10M ops/sec × 100 bytes = 1 GB/s = 8 Gbps
- Response headers: 10M × 200 bytes = 2 GB/s = 16 Gbps

**S - System Interface (API Design):**

**Middleware Integration:**

def rate_limit_middleware(request):
    # Extract identifier (user_id, IP, API key)
    identifier = get_identifier(request)
    
    # Check rate limit
    allowed, remaining, reset_time = rate_limiter.check_limit(
        identifier=identifier,
        limit=100,  # per minute
        window=60
    )
    
    if not allowed:
        return Response(
            status=429,
            headers={
                'X-RateLimit-Limit': '100',
                'X-RateLimit-Remaining': '0',
                'X-RateLimit-Reset': str(reset_time),
                'Retry-After': '60'
            }
        )
    
    # Process request
    response = process_request(request)
    
    # Add rate limit headers
    response.headers['X-RateLimit-Limit'] = '100'
    response.headers['X-RateLimit-Remaining'] = str(remaining)
    response.headers['X-RateLimit-Reset'] = str(reset_time)
    
    return response

**H - High-Level Design:**

**Architecture:**
    Request → [Rate Limiter Middleware] → [Redis Cluster]
              |
              v
        [Local Cache] (Optional)
              |
              v
        [Allow/Deny] → [Response with Headers]

**Components:**
1. **Rate Limiter Middleware**: Intercept requests
2. **Redis Cluster**: Store rate limit counters
3. **Local Cache**: Cache rate limit status (optional)
4. **Algorithm**: Token bucket or sliding window

**A - Algorithm/Data Structure:**

**Token Bucket Algorithm:**
- **Bucket**: Has capacity (max tokens)
- **Refill Rate**: Tokens added at fixed rate
- **Request**: Consumes token
- **Allow**: If tokens available
- **Deny**: If bucket empty
- **Burst**: Allows bursts up to bucket capacity

**Sliding Window Algorithm:**
- **Window**: Track requests in time window
- **Count**: Number of requests in window
- **Allow**: If count < limit
- **Deny**: If count >= limit
- **Accuracy**: More accurate than fixed window

**Fixed Window Algorithm:**
- **Window**: Fixed time window (e.g., 1 minute)
- **Counter**: Count requests in current window
- **Reset**: Counter resets at window boundary
- **Issue**: Allows bursts at boundaries

**D - Detailed Design:**

**Token Bucket Implementation:**

class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity  # Max tokens
        self.refill_rate = refill_rate  # Tokens per second
        self.tokens = capacity
        self.last_refill = time.time()
    
    def allow_request(self, tokens=1):
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

**Redis Implementation (Token Bucket):**

def check_rate_limit_redis(identifier, limit, window):
    key = f"rate_limit:{identifier}"
    
    # Lua script for atomic operation
    lua_script = """
    local key = KEYS[1]
    local limit = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    
    -- Get current count
    local count = redis.call('GET', key)
    if count == false then
        count = 0
    else
        count = tonumber(count)
    end
    
    -- Check if limit exceeded
    if count >= limit then
        return {0, count, now + window}
    end
    
    -- Increment counter
    redis.call('INCR', key)
    redis.call('EXPIRE', key, window)
    
    return {1, limit - count - 1, now + window}
    """
    
    result = redis.eval(lua_script, 1, key, limit, window, int(time.time()))
    allowed, remaining, reset_time = result
    
    return allowed == 1, remaining, reset_time

**Sliding Window Implementation (Redis):**

def check_rate_limit_sliding_window(identifier, limit, window):
    key = f"rate_limit:sw:{identifier}"
    now = time.time()
    window_start = now - window
    
    # Lua script for atomic operation
    lua_script = """
    local key = KEYS[1]
    local limit = tonumber(ARGV[1])
    local window_start = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    
    -- Remove old entries
    redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
    
    -- Count current requests
    local count = redis.call('ZCARD', key)
    
    if count >= limit then
        return {0, count, now + window}
    end
    
    -- Add current request
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, window)
    
    return {1, limit - count - 1, now + window}
    """
    
    result = redis.eval(lua_script, 1, key, limit, window_start, now)
    allowed, remaining, reset_time = result
    
    return allowed == 1, remaining, reset_time

**E - Evaluation (Bottlenecks & Optimizations):**

**Potential Bottlenecks:**

1. **Redis Operations**: 10M ops/sec
   - Solution: Redis cluster (shard by identifier)
   - Local cache for frequently checked limits
   - Connection pooling
   - Pipelining

2. **Latency**: < 1ms overhead
   - Solution: Local cache (hit rate ~80%)
   - Efficient Redis operations
   - Lua scripts for atomicity
   - Minimize network round-trips

3. **Memory Usage**: Rate limit keys
   - Solution: TTL expiration
   - Cleanup expired keys
   - Use sliding window (more memory efficient than fixed window)

4. **Accuracy**: Fixed window allows bursts
   - Solution: Use sliding window
   - Or token bucket (allows controlled bursts)

5. **Distributed Consistency**: Multiple servers
   - Solution: Redis for shared state
   - Atomic operations (Lua scripts)
   - Acceptable slight inconsistency (rate limits are approximate)

**Optimizations:**
- **Local Cache**: Cache rate limit status (80% hit rate)
- **Redis Cluster**: Shard by identifier hash
- **Lua Scripts**: Atomic operations
- **Connection Pooling**: Reuse Redis connections
- **Pipelining**: Batch operations when possible

**D - Discussion (Trade-offs, Scaling, Reliability):**

**Trade-offs:**

1. **Token Bucket vs Sliding Window:**
   - Token Bucket: Allows bursts, simpler
   - Sliding Window: More accurate, no boundary bursts
   - **Choice**: Token bucket for API limits, sliding window for strict limits

2. **Accuracy vs Performance:**
   - More accurate: Higher memory/CPU
   - Less accurate: Better performance
   - **Choice**: Balance (sliding window is good compromise)

3. **Local Cache vs Redis:**
   - Local cache: Faster, but inconsistent
   - Redis: Slower, but consistent
   - **Choice**: Hybrid (local cache with TTL, fallback to Redis)

4. **Memory vs Accuracy:**
   - Store all requests: High memory, accurate
   - Approximate: Lower memory, less accurate
   - **Choice**: Sliding window (good balance)

**Scaling Strategy:**

**Phase 1 (Current):**
- Single Redis instance
- Fixed window algorithm
- Basic rate limiting

**Phase 2 (10x scale):**
- Redis cluster
- Sliding window algorithm
- Local caching

**Phase 3 (100x scale):**
- Advanced algorithms
- Adaptive rate limiting
- ML-based anomaly detection

**Reliability:**
- Redis replication
- Health checks
- Fallback to allow (fail open) or deny (fail closed)
- Circuit breakers

**Common Follow-up Questions:**

1. "How do you handle distributed rate limiting?"
   - Use Redis for shared state
   - Shard by identifier hash
   - Atomic operations (Lua scripts)
   - Accept slight inconsistency

2. "What about different rate limits for different tiers?"
   - Store limit in user profile
   - Lookup limit before checking
   - Cache limit per user
   - Support multiple limits (per minute, per hour)

3. "How do you handle burst traffic?"
   - Token bucket allows bursts
   - Set appropriate capacity
   - Monitor burst patterns
   - Adjust limits dynamically

4. "What if Redis is down?"
   - Fallback strategy: Allow (fail open) or deny (fail closed)
   - Local cache as backup
   - Circuit breaker pattern
   - Health checks

5. "How do you prevent abuse?"
   - Rate limiting per user/IP
   - Detect patterns (rapid requests)
   - Blacklist abusive users
   - Graduated response (slow down instead of deny)`,
					CodeExamples: `Token Bucket (Redis):

def check_rate_limit_token_bucket(identifier, capacity, refill_rate, window):
    key = f"rate_limit:tb:{identifier}"
    now = time.time()
    
    lua_script = """
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local window = tonumber(ARGV[3])
    local now = tonumber(ARGV[4])
    
    local data = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(data[1]) or capacity
    local last_refill = tonumber(data[2]) or now
    
    -- Refill tokens
    local elapsed = now - last_refill
    local tokens_to_add = elapsed * refill_rate
    tokens = math.min(capacity, tokens + tokens_to_add)
    
    -- Check if can allow request
    if tokens >= 1 then
        tokens = tokens - 1
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, window)
        return {1, math.floor(tokens), now + window}
    else
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, window)
        return {0, 0, now + window}
    end
    """
    
    result = redis.eval(lua_script, 1, key, capacity, refill_rate, window, now)
    allowed, remaining, reset_time = result
    
    return allowed == 1, remaining, reset_time

Distributed Rate Limiter:

class DistributedRateLimiter:
    def __init__(self, redis_cluster, local_cache=None):
        self.redis = redis_cluster
        self.local_cache = local_cache or {}
        self.cache_ttl = 10  # seconds
    
    def check_limit(self, identifier, limit, window):
        # Check local cache first
        cache_key = f"{identifier}:{limit}:{window}"
        if cache_key in self.local_cache:
            cached_data, cached_time = self.local_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
        
        # Check Redis
        allowed, remaining, reset_time = check_rate_limit_sliding_window(
            identifier, limit, window
        )
        
        # Update local cache
        self.local_cache[cache_key] = ((allowed, remaining, reset_time), time.time())
        
        return allowed, remaining, reset_time

Rate Limiter Middleware:

class RateLimiterMiddleware:
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter
    
    def __call__(self, request):
        # Get identifier
        identifier = self.get_identifier(request)
        
        # Get rate limit config
        limit, window = self.get_rate_limit_config(request)
        
        # Check rate limit
        allowed, remaining, reset_time = self.rate_limiter.check_limit(
            identifier, limit, window
        )
        
        if not allowed:
            return self.rate_limit_exceeded(remaining, reset_time)
        
        # Process request
        response = self.process_request(request)
        
        # Add rate limit headers
        response.headers['X-RateLimit-Limit'] = str(limit)
        response.headers['X-RateLimit-Remaining'] = str(remaining)
        response.headers['X-RateLimit-Reset'] = str(reset_time)
        
        return response
    
    def get_identifier(self, request):
        # Priority: API key > user_id > IP
        if 'X-API-Key' in request.headers:
            return f"api_key:{request.headers['X-API-Key']}"
        elif 'user_id' in request.session:
            return f"user:{request.session['user_id']}"
        else:
            return f"ip:{request.remote_addr}"
    
    def get_rate_limit_config(self, request):
        # Get from user tier or default
        if 'user_id' in request.session:
            tier = get_user_tier(request.session['user_id'])
            if tier == 'premium':
                return 1000, 60  # 1000 per minute
            elif tier == 'paid':
                return 500, 60   # 500 per minute
        
        return 100, 60  # Default: 100 per minute

Architecture Diagram:

[Request] → [Rate Limiter Middleware]
              |
              v
        [Local Cache] (80% hit)
              |
              v (20% miss)
        [Redis Cluster] (Sharded by identifier)
              |
              v
        [Allow/Deny] → [Response with Headers]

Capacity Summary:

Requests: 10M/sec
- Rate limit checks: 10M/sec
- Local cache (80% hit): 8M/sec from cache
- Redis (20% miss): 2M ops/sec
- Redis cluster: 10 nodes × 200K ops/sec = 2M ops/sec

Storage: ~200 MB
- Active rate limit keys: 10M
- Per key: 58 bytes
- With TTL expiration: ~100 MB actual
- Redis overhead: ~200 MB total`,
				},
				{
					Title: "Design a Notification System",
					Content: `**Complete Interview Walkthrough Using RESHADED Framework**

**Problem Statement:**
Design a notification system that can send push notifications, emails, and SMS to millions of users, handle different notification types, track delivery status, and respect user preferences.

**R - Requirements Clarification:**

**Questions to Ask:**
1. "What's the expected scale? How many notifications per day?"
2. "What channels do we need? (push, email, SMS, in-app)"
3. "What's the acceptable delivery latency?"
4. "Do we need to support different notification types? (transactional, marketing)"
5. "How important is delivery guarantee? (at-least-once vs exactly-once)"
6. "Do we need to support scheduling? (send at specific time)"
7. "What about user preferences and opt-outs?"

**Assumptions (After Clarification):**
- 100M notifications/day
- Support push (FCM, APNS), email (SendGrid), SMS (Twilio)
- < 5 seconds delivery latency for push
- < 1 minute for email/SMS
- Support transactional and marketing notifications
- At-least-once delivery guarantee
- Support scheduling
- Respect user preferences

**E - Estimate (Capacity Planning):**

**Traffic Estimation:**
- Notifications: 100M/day = ~1,160 notifications/sec average
- Peak: 10x average = ~11,600 notifications/sec
- Push: 60% = 6,960 push/sec
- Email: 30% = 3,480 email/sec
- SMS: 10% = 1,160 SMS/sec

**Storage Estimation:**
- Notification record: ~500 bytes (id, user_id, type, content, status, timestamps)
- 100M notifications/day × 500 bytes = 50 GB/day
- 1 year retention: 50 GB × 365 = 18.25 TB
- User preferences: 100M users × 100 bytes = 10 GB

**Bandwidth Estimation:**
- Push: 6,960/sec × 1 KB = 6.96 MB/s = 56 Mbps
- Email: 3,480/sec × 10 KB = 34.8 MB/s = 278 Mbps
- SMS: 1,160/sec × 140 bytes = 162 KB/s = 1.3 Mbps

**S - System Interface (API Design):**

**Endpoints:**

POST /api/v1/notifications
Request:
{
  "user_id": 123,
  "type": "push",  // push, email, sms
  "title": "New message",
  "body": "You have a new message",
  "data": {"message_id": 456},
  "priority": "high",  // high, normal, low
  "scheduled_at": "2024-01-17T10:00:00Z"  // Optional
}

Response:
{
  "notification_id": "abc123",
  "status": "queued"
}

GET /api/v1/notifications/{notification_id}/status
Response:
{
  "notification_id": "abc123",
  "status": "delivered",  // queued, sent, delivered, failed
  "delivered_at": "2024-01-17T10:00:01Z",
  "failure_reason": null
}

GET /api/v1/users/{user_id}/preferences
POST /api/v1/users/{user_id}/preferences

**H - High-Level Design:**

**Architecture:**
    [Notification Service] → [Kafka Topics] → [Worker Pools]
                                      |
                                      v
                            [Provider APIs]
                            - FCM (Android)
                            - APNS (iOS)
                            - SendGrid (Email)
                            - Twilio (SMS)
                                      |
                                      v
                            [Delivery Tracking] → [Database]

**Components:**
1. **Notification Service**: Create and queue notifications
2. **Kafka**: Reliable message queue
3. **Worker Pools**: Process notifications per channel
4. **Provider Clients**: Integrate with external providers
5. **Delivery Tracker**: Track delivery status
6. **Preference Service**: Manage user preferences
7. **Scheduler**: Handle scheduled notifications

**A - Algorithm/Data Structure:**

**Notification Routing:**
- **User Preferences**: Check user's preferred channels
- **Notification Type**: Route based on type (transactional → all channels, marketing → opt-in only)
- **Priority**: High priority notifications processed first
- **Batching**: Batch notifications when possible (especially email)

**Retry Logic:**
- **Exponential Backoff**: Retry with increasing delays (1s, 2s, 4s, 8s)
- **Max Retries**: 3-5 retries depending on provider
- **Dead Letter Queue**: Failed notifications after max retries

**Delivery Tracking:**
- **Status**: queued, sent, delivered, failed
- **Timestamps**: Created, sent, delivered
- **Failure Reason**: Error codes, messages
- **Webhooks**: Provider callbacks for delivery status

**D - Detailed Design:**

**Database Schema:**

CREATE TABLE notifications (
    id VARCHAR(50) PRIMARY KEY,
    user_id BIGINT NOT NULL,
    type VARCHAR(20),  -- push, email, sms
    title VARCHAR(255),
    body TEXT,
    data JSON,
    priority VARCHAR(20),  -- high, normal, low
    status VARCHAR(20),  -- queued, sent, delivered, failed
    scheduled_at TIMESTAMP,
    created_at TIMESTAMP,
    sent_at TIMESTAMP,
    delivered_at TIMESTAMP,
    failure_reason TEXT,
    retry_count INT DEFAULT 0,
    INDEX idx_user_created (user_id, created_at),
    INDEX idx_status (status),
    INDEX idx_scheduled (scheduled_at)
);

CREATE TABLE user_preferences (
    user_id BIGINT PRIMARY KEY,
    push_enabled BOOLEAN DEFAULT true,
    email_enabled BOOLEAN DEFAULT true,
    sms_enabled BOOLEAN DEFAULT false,
    marketing_push BOOLEAN DEFAULT false,
    marketing_email BOOLEAN DEFAULT false,
    updated_at TIMESTAMP
);

CREATE TABLE device_tokens (
    user_id BIGINT,
    device_id VARCHAR(255),
    platform VARCHAR(20),  -- android, ios
    token VARCHAR(500),
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (user_id, device_id),
    INDEX idx_token (token)
);

**Notification Flow:**

1. Service creates notification
2. Notification Service → Check user preferences
3. Notification Service → Create notification record
4. Notification Service → Send to Kafka topic (per channel)
5. Worker consumes from Kafka
6. Worker → Get user device tokens (for push)
7. Worker → Send via provider API
8. Worker → Update status (sent)
9. Provider → Webhook callback (delivered/failed)
10. Delivery Tracker → Update status

**Retry Flow:**

1. Worker sends notification
2. Provider returns error
3. Worker → Check retry count
4. If retry count < max:
   - Calculate backoff delay
   - Requeue notification with delay
5. If retry count >= max:
   - Move to dead letter queue
   - Alert for manual review

**E - Evaluation (Bottlenecks & Optimizations):**

**Potential Bottlenecks:**

1. **Provider Rate Limits**: External APIs have limits
   - Solution: Rate limiting per provider
   - Queue management
   - Multiple provider accounts
   - Circuit breakers

2. **Worker Throughput**: Limited by provider APIs
   - Solution: Horizontal scaling of workers
   - Parallel processing
   - Batching (especially email)
   - Connection pooling

3. **Kafka Throughput**: 11,600 notifications/sec
   - Solution: Kafka partitions (10-20 partitions)
   - Multiple consumer groups
   - Efficient serialization

4. **Delivery Tracking**: High write volume
   - Solution: Batch updates
   - Async tracking
   - Separate database for tracking
   - Time-series database for analytics

5. **User Preferences**: High read volume
   - Solution: Cache preferences (Redis)
   - Invalidate on update
   - Default preferences

**Optimizations:**
- **Batching**: Batch email notifications (reduce API calls)
- **Caching**: Cache user preferences, device tokens
- **Parallel Processing**: Process multiple notifications concurrently
- **Connection Pooling**: Reuse provider connections
- **Circuit Breakers**: Prevent cascade failures
- **Dead Letter Queue**: Handle failures gracefully

**D - Discussion (Trade-offs, Scaling, Reliability):**

**Trade-offs:**

1. **Latency vs Throughput:**
   - Low latency: Process immediately, lower throughput
   - High throughput: Batch processing, higher latency
   - **Choice**: Balance (priority queue for low latency, batch for high throughput)

2. **Reliability vs Performance:**
   - Strong guarantees: Slower, more complex
   - Best effort: Faster, simpler
   - **Choice**: At-least-once delivery (acceptable for notifications)

3. **Batching:**
   - Batch: Higher throughput, lower latency per notification
   - No batch: Lower latency, higher API calls
   - **Choice**: Batch for email, no batch for push

4. **Retry Strategy:**
   - Aggressive retry: Higher delivery rate, more load
   - Conservative retry: Lower delivery rate, less load
   - **Choice**: Exponential backoff with max retries

**Scaling Strategy:**

**Phase 1 (Current):**
- Single Kafka topic
- Basic worker pool
- Single provider account

**Phase 2 (10x scale):**
- Multiple Kafka partitions
- Worker pools per channel
- Multiple provider accounts
- Rate limiting

**Phase 3 (100x scale):**
- Advanced batching
- ML-based routing
- Predictive scaling
- Multi-region deployment

**Reliability:**
- Kafka ensures delivery (at-least-once)
- Retry logic for failures
- Dead letter queue for manual review
- Health checks and auto-scaling
- Circuit breakers for provider failures

**Common Follow-up Questions:**

1. "How do you ensure delivery?"
   - Kafka guarantees at-least-once delivery
   - Retry logic with exponential backoff
   - Dead letter queue for persistent failures
   - Webhook callbacks for delivery confirmation

2. "What about duplicate notifications?"
   - Idempotent notification IDs
   - Deduplication at provider level
   - Track sent notifications
   - Acceptable for notifications (better than missing)

3. "How do you handle provider failures?"
   - Circuit breakers (stop sending if provider down)
   - Fallback providers (if available)
   - Queue notifications (retry when provider recovers)
   - Alerting for extended outages

4. "What about scheduled notifications?"
   - Store scheduled_at timestamp
   - Background job to process scheduled notifications
   - Query notifications where scheduled_at <= now
   - Process in priority order

5. "How do you optimize costs?"
   - Batch email notifications
   - Use cheaper channels when possible (push vs SMS)
   - Respect user preferences (don't send unwanted)
   - Rate limiting to prevent abuse`,
					CodeExamples: `Notification Service:

def send_notification(user_id, notification_data):
    # Check user preferences
    preferences = get_user_preferences(user_id)
    
    # Determine channels based on preferences and type
    channels = []
    if notification_data['type'] == 'transactional':
        if preferences['push_enabled']:
            channels.append('push')
        if preferences['email_enabled']:
            channels.append('email')
        if preferences['sms_enabled']:
            channels.append('sms')
    elif notification_data['type'] == 'marketing':
        if preferences['marketing_push']:
            channels.append('push')
        if preferences['marketing_email']:
            channels.append('email')
    
    # Create notification for each channel
    for channel in channels:
        notification = {
            'id': generate_id(),
            'user_id': user_id,
            'channel': channel,
            'title': notification_data['title'],
            'body': notification_data['body'],
            'data': notification_data.get('data'),
            'priority': notification_data.get('priority', 'normal'),
            'scheduled_at': notification_data.get('scheduled_at'),
            'status': 'queued',
            'created_at': time.now()
        }
        
        # Store in database
        db.insert_notification(notification)
        
        # Send to Kafka
        if notification['scheduled_at'] is None or notification['scheduled_at'] <= time.now():
            kafka.produce(f'notifications_{channel}', notification)

Push Notification Worker:

def process_push_notification(notification):
    try:
        # Get user device tokens
        device_tokens = db.get_device_tokens(notification['user_id'])
        
        if not device_tokens:
            update_status(notification['id'], 'failed', 'No device tokens')
            return
        
        # Send to each device
        for device_token in device_tokens:
            if device_token['platform'] == 'android':
                send_fcm_notification(device_token['token'], notification)
            elif device_token['platform'] == 'ios':
                send_apns_notification(device_token['token'], notification)
        
        # Update status
        update_status(notification['id'], 'sent', None)
        
    except Exception as e:
        # Retry logic
        retry_count = notification.get('retry_count', 0)
        if retry_count < MAX_RETRIES:
            backoff = 2 ** retry_count  # Exponential backoff
            requeue_notification(notification, backoff)
            update_retry_count(notification['id'], retry_count + 1)
        else:
            move_to_dlq(notification)
            update_status(notification['id'], 'failed', str(e))

Email Batching:

def batch_email_notifications():
    # Get queued email notifications (batch of 100)
    notifications = db.get_queued_notifications('email', limit=100)
    
    if not notifications:
        return
    
    # Group by user (for batching)
    user_notifications = {}
    for notification in notifications:
        user_id = notification['user_id']
        if user_id not in user_notifications:
            user_notifications[user_id] = []
        user_notifications[user_id].append(notification)
    
    # Send batched emails
    for user_id, user_notifs in user_notifications.items():
        # Get user email
        user_email = db.get_user_email(user_id)
        
        # Create batched email
        email_content = create_batched_email(user_notifs)
        
        # Send via SendGrid
        sendgrid.send_email(
            to=user_email,
            subject='You have new notifications',
            html=email_content
        )
        
        # Update status for all notifications
        for notification in user_notifs:
            update_status(notification['id'], 'sent', None)

Retry Logic with Exponential Backoff:

def retry_notification(notification):
    retry_count = notification.get('retry_count', 0)
    
    if retry_count >= MAX_RETRIES:
        move_to_dlq(notification)
        return
    
    # Calculate backoff delay
    backoff_seconds = 2 ** retry_count  # 1s, 2s, 4s, 8s, 16s
    
    # Schedule retry
    retry_at = time.now() + backoff_seconds
    
    # Update notification
    db.update_notification(notification['id'], {
        'retry_count': retry_count + 1,
        'scheduled_at': retry_at,
        'status': 'queued'
    })
    
    # Requeue
    kafka.produce('notifications_retry', notification, delay=backoff_seconds)

Architecture Diagram:

[Notification Service] → [Kafka Topics]
                            - notifications_push
                            - notifications_email
                            - notifications_sms
                            |
                            v
                    [Worker Pools]
                    - Push Workers (FCM, APNS)
                    - Email Workers (SendGrid)
                    - SMS Workers (Twilio)
                            |
                            v
                    [Provider APIs]
                            |
                            v
                    [Delivery Tracking] → [Database]
                            |
                            v
                    [Webhook Handlers] → [Status Updates]

Capacity Summary:

Notifications: 11.6K/sec
- Push: 6.96K/sec
- Email: 3.48K/sec
- SMS: 1.16K/sec

Workers Needed:
- Push: 6.96K/sec ÷ 100/sec per worker = ~70 workers
- Email: 3.48K/sec ÷ 50/sec per worker = ~70 workers (with batching)
- SMS: 1.16K/sec ÷ 10/sec per worker = ~116 workers

Storage: 18.25 TB/year
- Notifications: 50 GB/day × 365 = 18.25 TB/year
- User preferences: 10 GB`,
				},
				{
					Title: "Design a Web Crawler",
					Content: `**Complete Interview Walkthrough Using RESHADED Framework**

**Problem Statement:**
Design a web crawler that can crawl billions of web pages, respect robots.txt and rate limits, extract content, avoid duplicates, and scale to crawl the entire web.

**R - Requirements Clarification:**

**Questions to Ask:**
1. "What's the expected scale? How many pages to crawl?"
2. "How fresh should the crawl be? (daily, weekly, monthly)"
3. "What content types do we need? (HTML, PDF, images)"
4. "Do we need to respect robots.txt?"
5. "What's the acceptable crawl rate? (pages per second)"
6. "Do we need to handle JavaScript-rendered pages?"
7. "What about handling different languages and encodings?"

**Assumptions (After Clarification):**
- Crawl 50B web pages
- Refresh monthly (50B / 30 = 1.67B pages/day)
- Support HTML, PDF, images
- Respect robots.txt and rate limits
- 20,000 pages/sec crawl rate
- Support JavaScript rendering (headless browser)
- Handle multiple languages/encodings

**E - Estimate (Capacity Planning):**

**Traffic Estimation:**
- Pages to crawl: 1.67B pages/day
- Crawl rate: 1.67B / 86400 = ~19,300 pages/sec
- With politeness (1 second delay per domain): Effective rate depends on domain distribution
- Assume 100K unique domains: 19,300 / 100K = ~0.2 pages/sec per domain average

**Storage Estimation:**
- Average page: 50 KB (HTML + extracted text)
- Content storage: 1.67B × 50 KB = 83.5 TB/day
- 1 month retention: 83.5 TB × 30 = 2.5 PB
- With replication (3x): 7.5 PB
- URL frontier: 50B URLs × 100 bytes = 5 TB
- Deduplication index: 50B × 16 bytes (hash) = 800 GB

**Bandwidth Estimation:**
- Crawling: 19,300 pages/sec × 50 KB = 965 MB/s = 7.7 Gbps
- With politeness: Actual bandwidth lower (delays between requests)

**S - System Interface (API Design):**

**Endpoints:**

POST /api/v1/crawl/start
Request:
{
  "seed_urls": ["https://example.com"],
  "max_depth": 10,
  "respect_robots": true
}

Response:
{
  "crawl_id": "abc123",
  "status": "started"
}

GET /api/v1/crawl/{crawl_id}/status
Response:
{
  "crawl_id": "abc123",
  "status": "running",
  "pages_crawled": 1000000,
  "pages_remaining": 50000000,
  "errors": 100
}

GET /api/v1/content/{url_hash}
Response:
{
  "url": "https://example.com/page",
  "content": "...",
  "title": "Example Page",
  "links": ["https://example.com/link1"],
  "crawled_at": "2024-01-17T10:00:00Z"
}

**H - High-Level Design:**

**Architecture:**
    [URL Frontier] → [Crawler Workers] → [Content Store]
              |
              v
        [Robots.txt Cache] → [Politeness Module]
              |
              v
        [Deduplication Service] → [Bloom Filter]
              |
              v
        [Content Parser] → [Link Extractor] → [URL Frontier]

**Components:**
1. **URL Frontier**: Queue of URLs to crawl (priority queue)
2. **Crawler Workers**: Fetch web pages
3. **Robots.txt Cache**: Cache and respect robots.txt
4. **Politeness Module**: Rate limiting per domain
5. **Deduplication Service**: Detect duplicate URLs/content
6. **Content Parser**: Extract content, links, metadata
7. **Content Store**: Store crawled content
8. **Link Extractor**: Extract URLs from pages

**A - Algorithm/Data Structure:**

**URL Frontier:**
- **Priority Queue**: Prioritize important URLs
- **Sharding**: Shard by domain or URL hash
- **Deduplication**: Check before adding
- **Politeness**: Queue per domain with delays

**Deduplication:**
- **URL Normalization**: Normalize URLs (remove fragments, sort params)
- **Content Hash**: SHA-256 hash of content
- **Bloom Filter**: Probabilistic data structure for URL deduplication
- **Exact Match**: Database lookup for content hash

**Politeness:**
- **Per-Domain Delays**: Track last crawl time per domain
- **Robots.txt**: Respect crawl-delay directive
- **Default Delay**: 1 second between requests to same domain
- **Rate Limiting**: Limit concurrent requests per domain

**D - Detailed Design:**

**Database Schema:**

CREATE TABLE urls (
    id BIGINT PRIMARY KEY,
    url VARCHAR(2048) UNIQUE,
    normalized_url VARCHAR(2048),
    domain VARCHAR(255),
    status VARCHAR(20),  -- pending, crawled, failed
    priority INT DEFAULT 0,
    depth INT DEFAULT 0,
    created_at TIMESTAMP,
    crawled_at TIMESTAMP,
    retry_count INT DEFAULT 0,
    INDEX idx_status_priority (status, priority DESC),
    INDEX idx_domain (domain),
    INDEX idx_normalized (normalized_url)
);

CREATE TABLE pages (
    id BIGINT PRIMARY KEY,
    url_id BIGINT REFERENCES urls(id),
    content_hash VARCHAR(64),  -- SHA-256
    content TEXT,
    title VARCHAR(500),
    content_type VARCHAR(100),
    content_size INT,
    crawled_at TIMESTAMP,
    INDEX idx_hash (content_hash),
    INDEX idx_url (url_id)
);

CREATE TABLE robots_cache (
    domain VARCHAR(255) PRIMARY KEY,
    robots_txt TEXT,
    crawl_delay INT,  -- seconds
    disallowed_paths JSON,
    cached_at TIMESTAMP,
    expires_at TIMESTAMP
);

**Crawling Flow:**

1. URL Frontier → Get next URL (priority queue)
2. Crawler Worker → Check robots.txt (cached)
3. Crawler Worker → Check politeness (domain delay)
4. Crawler Worker → Check deduplication (Bloom filter)
5. Crawler Worker → Fetch page (respect rate limits)
6. Crawler Worker → Parse content
7. Crawler Worker → Extract links
8. Crawler Worker → Store content
9. Crawler Worker → Add new URLs to frontier
10. Crawler Worker → Mark URL as crawled

**Deduplication Flow:**

1. Normalize URL (remove fragment, sort params, lowercase)
2. Check Bloom filter (fast, probabilistic)
3. If Bloom filter says "maybe seen":
   - Check database (exact match)
   - If found: Skip URL
   - If not found: Add to Bloom filter, crawl
4. If Bloom filter says "not seen":
   - Add to Bloom filter
   - Crawl URL

**E - Evaluation (Bottlenecks & Optimizations):**

**Potential Bottlenecks:**

1. **Crawling Rate**: 19,300 pages/sec
   - Solution: Distributed crawlers (1000+ workers)
   - Parallel crawling across domains
   - Efficient HTTP clients
   - Connection pooling

2. **Robots.txt Lookups**: High frequency
   - Solution: Cache robots.txt (24 hour TTL)
   - Batch lookups
   - Default rules for common cases

3. **Deduplication**: 50B URLs
   - Solution: Bloom filter (probabilistic, memory-efficient)
   - Database lookup only for Bloom filter positives
   - Content hash for near-duplicate detection

4. **Storage**: 7.5 PB
   - Solution: Distributed storage
   - Compression
   - Archive old content
   - Deduplication (content hash)

5. **Politeness**: Slows down crawling
   - Solution: Parallel crawling across domains
   - Respect delays but crawl multiple domains simultaneously
   - Priority queue (important URLs first)

**Optimizations:**
- **Bloom Filter**: Memory-efficient deduplication
- **Caching**: Cache robots.txt, DNS lookups
- **Parallel Crawling**: Crawl multiple domains simultaneously
- **Compression**: Compress stored content
- **Incremental Crawling**: Only crawl changed pages
- **Priority Queue**: Crawl important URLs first

**D - Discussion (Trade-offs, Scaling, Reliability):**

**Trade-offs:**

1. **Freshness vs Politeness:**
   - Frequent crawls: Fresh content, may overload servers
   - Polite crawls: Better relationships, slower indexing
   - **Choice**: Balance (respect robots.txt, reasonable delays)

2. **Coverage vs Depth:**
   - Wide crawl: More domains, shallow depth
   - Deep crawl: Fewer domains, deeper depth
   - **Choice**: Hybrid (wide for discovery, deep for important sites)

3. **Storage vs Compute:**
   - Store full content: Higher storage, faster queries
   - Store only index: Lower storage, slower queries
   - **Choice**: Store full content (needed for search)

4. **Deduplication Accuracy:**
   - Exact match: 100% accurate, slower
   - Bloom filter: Fast, small false positives
   - **Choice**: Bloom filter + database lookup for positives

**Scaling Strategy:**

**Phase 1 (Current):**
- Single crawler
- Basic deduplication
- Simple politeness

**Phase 2 (10x scale):**
- Distributed crawlers
- Bloom filter deduplication
- Advanced politeness
- Priority queue

**Phase 3 (100x scale):**
- Global crawler network
- Incremental crawling
- ML-based prioritization
- Advanced deduplication

**Reliability:**
- Retry logic for failed pages
- Health checks for crawlers
- Graceful handling of malformed pages
- Monitoring and alerting
- Fault tolerance (crawler failures don't stop system)

**Common Follow-up Questions:**

1. "How do you handle JavaScript-rendered pages?"
   - Use headless browser (Puppeteer, Selenium)
   - Render page, extract content
   - More expensive (CPU, memory)
   - Use for important pages only

2. "How do you prioritize URLs?"
   - PageRank or similar authority score
   - Sitemap priority
   - User signals (clicks, engagement)
   - Recency (fresher content first)
   - Domain authority

3. "What about handling different content types?"
   - HTML: Parse with HTML parser
   - PDF: Extract text with PDF library
   - Images: Extract metadata, OCR if needed
   - Different parsers per content type

4. "How do you handle sitemaps?"
   - Fetch sitemap.xml
   - Extract URLs from sitemap
   - Respect sitemap priority
   - Use sitemap for discovery

5. "What about handling cookies and sessions?"
   - Maintain session per domain
   - Handle cookies
   - Respect session requirements
   - Use for authenticated content (if needed)`,
					CodeExamples: `URL Frontier Implementation:

class URLFrontier:
    def __init__(self):
        self.pending_urls = PriorityQueue()  # Priority queue
        self.crawled_urls = set()  # Set of crawled URLs
        self.domain_queues = {}  # Queue per domain
    
    def add_url(self, url, priority=0, depth=0):
        # Normalize URL
        normalized = normalize_url(url)
        
        # Check if already crawled
        if normalized in self.crawled_urls:
            return False
        
        # Add to priority queue
        self.pending_urls.put((priority, depth, normalized))
        
        # Add to domain queue
        domain = get_domain(url)
        if domain not in self.domain_queues:
            self.domain_queues[domain] = []
        self.domain_queues[domain].append((priority, depth, normalized))
        
        return True
    
    def get_next_url(self):
        # Get URL from priority queue
        if self.pending_urls.empty():
            return None
        
        priority, depth, url = self.pending_urls.get()
        return url

Bloom Filter Deduplication:

class BloomFilter:
    def __init__(self, capacity, error_rate=0.01):
        # Calculate bit array size and hash functions
        import math
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_array_size = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self.num_hashes = int(self.bit_array_size * math.log(2) / capacity)
        self.bit_array = [0] * self.bit_array_size
    
    def add(self, item):
        for i in range(self.num_hashes):
            hash_val = hash(f"{item}:{i}") % self.bit_array_size
            self.bit_array[hash_val] = 1
    
    def contains(self, item):
        for i in range(self.num_hashes):
            hash_val = hash(f"{item}:{i}") % self.bit_array_size
            if self.bit_array[hash_val] == 0:
                return False
        return True  # Maybe (could be false positive)

Crawler Worker:

class CrawlerWorker:
    def __init__(self, frontier, content_store, robots_cache, bloom_filter):
        self.frontier = frontier
        self.content_store = content_store
        self.robots_cache = robots_cache
        self.bloom_filter = bloom_filter
        self.domain_delays = {}  # Track last crawl time per domain
    
    def crawl(self):
        while True:
            # Get next URL
            url = self.frontier.get_next_url()
            if not url:
                time.sleep(1)
                continue
            
            domain = get_domain(url)
            
            # Check robots.txt
            if not self.robots_cache.can_crawl(url, domain):
                continue
            
            # Check politeness
            if not self.check_politeness(domain):
                # Requeue with delay
                self.frontier.add_url(url, delay=self.get_crawl_delay(domain))
                continue
            
            # Check deduplication
            normalized_url = normalize_url(url)
            if self.bloom_filter.contains(normalized_url):
                # Check database for exact match
                if self.content_store.url_exists(normalized_url):
                    continue
            else:
                self.bloom_filter.add(normalized_url)
            
            # Fetch page
            try:
                page = self.fetch_page(url)
            except Exception as e:
                # Retry logic
                self.handle_fetch_error(url, e)
                continue
            
            # Parse content
            content = self.parse_content(page)
            links = self.extract_links(page, url)
            
            # Store content
            self.content_store.store(url, content, links)
            
            # Add new URLs to frontier
            for link in links:
                self.frontier.add_url(link, depth=content['depth'] + 1)
            
            # Mark as crawled
            self.frontier.mark_crawled(url)
            self.domain_delays[domain] = time.now()
    
    def check_politeness(self, domain):
        if domain not in self.domain_delays:
            return True
        
        crawl_delay = self.robots_cache.get_crawl_delay(domain)
        if crawl_delay is None:
            crawl_delay = 1  # Default 1 second
        
        elapsed = time.now() - self.domain_delays[domain]
        return elapsed >= crawl_delay

Robots.txt Parser:

class RobotsCache:
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 24 * 3600  # 24 hours
    
    def can_crawl(self, url, domain):
        # Get robots.txt (cached)
        if domain not in self.cache:
            robots_txt = self.fetch_robots_txt(domain)
            self.cache[domain] = self.parse_robots_txt(robots_txt)
            self.cache[domain]['cached_at'] = time.now()
        
        # Check if cache expired
        if time.now() - self.cache[domain]['cached_at'] > self.cache_ttl:
            robots_txt = self.fetch_robots_txt(domain)
            self.cache[domain] = self.parse_robots_txt(robots_txt)
            self.cache[domain]['cached_at'] = time.now()
        
        # Check if URL is allowed
        robots = self.cache[domain]
        for disallowed_path in robots['disallowed']:
            if url.startswith(disallowed_path):
                return False
        
        return True
    
    def get_crawl_delay(self, domain):
        if domain in self.cache:
            return self.cache[domain].get('crawl_delay', 1)
        return 1

Content Parser:

def parse_content(page, url):
    content_type = page.headers.get('Content-Type', '')
    
    if 'text/html' in content_type:
        return parse_html(page.content, url)
    elif 'application/pdf' in content_type:
        return parse_pdf(page.content)
    elif 'image/' in content_type:
        return parse_image(page.content)
    else:
        return {'content': '', 'type': 'unknown'}

def parse_html(html_content, url):
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract title
    title = soup.title.string if soup.title else ''
    
    # Extract text content
    text_content = soup.get_text()
    
    # Extract links
    links = []
    for link in soup.find_all('a', href=True):
        absolute_url = urljoin(url, link['href'])
        links.append(absolute_url)
    
    return {
        'title': title,
        'content': text_content,
        'links': links,
        'type': 'html'
    }

Architecture Diagram:

[Seed URLs] → [URL Frontier] (Priority Queue)
                    |
                    v
            [Crawler Workers] (1000+ workers)
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
[Robots Cache] [Bloom Filter] [Politeness]
        |           |           |
        +-----------+-----------+
                    |
                    v
            [Content Parser]
                    |
                    v
            [Content Store]
                    |
                    v
            [Link Extractor] → [URL Frontier]

Capacity Summary:

Crawling Rate: 19.3K pages/sec
- 1000 workers × 19.3 pages/sec = 19.3K pages/sec
- With politeness: Effective rate depends on domain distribution

Storage: 7.5 PB (with replication)
- Content: 2.5 PB/month
- With 3x replication: 7.5 PB

Deduplication:
- Bloom filter: 50B URLs × 10 bits = 62.5 GB
- Database: 800 GB for exact matches`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
