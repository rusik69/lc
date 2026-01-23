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
			},
			ProblemIDs: []int{},
		},
	})
}
