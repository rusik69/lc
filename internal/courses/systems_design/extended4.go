package systemsdesign

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSystemsDesignModules([]problems.CourseModule{
		{
			ID:          2422,
			Title:       "Microservices Architecture and Service Mesh",
			Description: "Master microservices patterns, service discovery, API gateways, service mesh, observability, and deployment strategies.",
			Order:       23,
			Lessons: []problems.Lesson{
				{
					Title: "Microservices Patterns Service Discovery and Observability",
					Content: `Microservices architecture decomposes applications into small, independently deployable services. Understanding the patterns and infrastructure is essential for building reliable distributed systems.

**Microservices vs Monolith:**

Monolith:
  Single deployable unit
  Shared database
  In-process communication
  Simple to develop initially
  Harder to scale independently
  Technology lock-in

Microservices:
  Multiple independently deployable services
  Each owns its data
  Network communication (HTTP, gRPC, messaging)
  Complex to develop and operate
  Independent scaling per service
  Polyglot technology choices

When to use microservices:
  Team size > 20 engineers
  Need independent deployments
  Different scaling requirements per component
  Multiple technology requirements
  Organizational boundaries (Conway's Law)

When NOT to use:
  Small team (< 10)
  New/unclear domain boundaries
  Low traffic
  Tight deadlines (monolith is faster initially)

**Service Decomposition:**

Strategies:
  By business capability: Payment, Shipping, Inventory
  By subdomain (DDD): Core, Supporting, Generic
  By data ownership: Each service owns its data
  Strangler fig: Gradually extract from monolith

Bounded Context (DDD):
  Each service has its own domain model
  Shared concepts have different representations
  Anti-corruption layer translates between contexts
  Context mapping shows relationships

Service Granularity:
  Too coarse: Mini-monolith, coupled teams
  Too fine: Network overhead, operational complexity
  Right size: Single team can own, coherent business capability

**Communication Patterns:**

Synchronous:
  REST/HTTP: Simple, cacheable, widely supported
  gRPC: Binary protocol, fast, streaming, code generation
  GraphQL: Flexible queries, client-driven
  
  Pros: Simple mental model, immediate feedback
  Cons: Temporal coupling, cascading failures

Asynchronous:
  Message queue: Point-to-point, task distribution
  Event bus: Pub/sub, event-driven
  
  Pros: Decoupled, resilient, buffering
  Cons: Complex debugging, eventual consistency

Patterns:
  Request-Response: Synchronous call and wait
  Fire-and-Forget: Send message, don't wait
  Publish-Subscribe: Broadcast events
  Event-Carried State Transfer: Events contain data needed
  Orchestration: Central coordinator directs workflow
  Choreography: Services react to events independently

**Service Discovery:**

Client-side discovery:
  Client queries service registry
  Client selects instance (load balancing)
  Examples: Netflix Eureka + Ribbon
  
  Pros: Client controls load balancing
  Cons: Client library per language

Server-side discovery:
  Client sends to load balancer
  Load balancer queries registry
  Examples: AWS ALB, Kubernetes Service
  
  Pros: Language agnostic, simpler clients
  Cons: Additional network hop

Service registry:
  Stores service instances and locations
  Health checking
  Examples: Consul, etcd, ZooKeeper, Kubernetes DNS

Registration patterns:
  Self-registration: Service registers itself
  Third-party registration: External registrar (Kubernetes)

**API Gateway:**

Responsibilities:
  Request routing: Route to correct service
  Composition: Aggregate data from multiple services
  Protocol translation: REST → gRPC, HTTP → WebSocket
  Authentication: Verify tokens, API keys
  Rate limiting: Protect services from overload
  Caching: Cache responses
  Load balancing: Distribute across instances
  SSL termination: Handle HTTPS

Patterns:
  BFF (Backend for Frontend): Gateway per client type
    Mobile BFF: Optimized payloads
    Web BFF: Full responses
    Third-party BFF: Stable API versions

Examples:
  Kong, Ambassador, AWS API Gateway, Istio Ingress

**Service Mesh:**

Architecture:
  Data plane: Sidecar proxies (Envoy, Linkerd-proxy)
  Control plane: Configuration and management (Istiod)
  
  Sidecar proxy intercepts all network traffic
  Transparent to application code

Features:
  Traffic management:
    Load balancing (round-robin, least connections)
    Circuit breaking
    Retries with budget
    Timeouts
    Traffic splitting (canary, A/B)
    
  Security:
    Mutual TLS (mTLS) between services
    Authorization policies
    Certificate rotation
    
  Observability:
    Distributed tracing (Jaeger, Zipkin)
    Metrics collection (Prometheus)
    Access logging

Examples: Istio, Linkerd, Consul Connect

**Observability:**

Three Pillars:

Metrics:
  Numeric measurements over time
  RED method: Rate, Errors, Duration
  USE method: Utilization, Saturation, Errors
  
  Key metrics:
    Request rate (RPS)
    Error rate (%)
    Latency (p50, p95, p99)
    Saturation (queue depth, CPU, memory)
    
  Tools: Prometheus, Grafana, Datadog

Logging:
  Structured logging (JSON)
  Correlation IDs across services
  Log levels: DEBUG, INFO, WARN, ERROR
  Centralized: ELK stack, Loki, CloudWatch
  
  Best practices:
    Include request ID, user ID, service name
    Log at service boundaries
    Avoid logging sensitive data
    Use structured format for querying

Distributed Tracing:
  Follow request across services
  Span: Unit of work in one service
  Trace: Collection of spans for one request
  Context propagation: Headers carry trace context
  Tools: Jaeger, Zipkin, AWS X-Ray, OpenTelemetry

Alerting:
  Symptom-based (user impact) over cause-based
  SLO-based alerting: Alert when error budget consumed
  Actionable: Every alert should have runbook
  Severity levels: Page (P1), Ticket (P2), Info (P3)

**Deployment Strategies:**

Blue-Green:
  Two identical environments
  Switch traffic atomically
  Easy rollback
  Requires double infrastructure

Canary:
  Route small percentage to new version
  Gradually increase if healthy
  Automated rollback on errors
  Requires traffic splitting

Rolling Update:
  Replace instances one at a time
  No extra infrastructure
  Mixed versions during rollout
  Kubernetes default

Shadow/Dark Launch:
  Fork traffic to new version
  Don't return shadow results to users
  Compare results for validation

Feature Flags:
  Deploy code without activating
  Toggle features per user/group
  Gradual rollout
  A/B testing

**Resilience Patterns:**

Retry:
  Exponential backoff with jitter
  Max retry limit
  Retry budget (% of total requests)
  Only retry transient failures

Circuit Breaker:
  Closed → Open → Half-Open
  Prevent cascading failures
  Fail fast instead of waiting

Bulkhead:
  Isolate resources per service/endpoint
  Thread pools or connection pools
  Prevent one service consuming all resources

Timeout:
  Set timeouts on all external calls
  Propagate deadline across services
  Budget timeout: deadline - elapsed

Rate Limiting:
  Protect services from overload
  Token bucket or sliding window
  Return 429 with Retry-After header

Fallback:
  Default response on failure
  Cached data (stale better than nothing)
  Degraded functionality`,
					CodeExamples: `# Microservices Architecture Implementation Examples

import time
import hashlib
import random
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from abc import ABC, abstractmethod

# ============================================================
# Service Registry and Discovery
# ============================================================

@dataclass
class ServiceInstance:
    service_name: str
    instance_id: str
    host: str
    port: int
    metadata: Dict[str, str] = field(default_factory=dict)
    healthy: bool = True
    registered_at: float = 0
    last_heartbeat: float = 0
    weight: int = 1


class ServiceRegistry:
    """Service registry with health checking."""
    
    def __init__(self, heartbeat_timeout: float = 30.0):
        self.heartbeat_timeout = heartbeat_timeout
        self._instances: Dict[str, Dict[str, ServiceInstance]] = defaultdict(dict)
        self._lock = threading.Lock()
    
    def register(self, service_name: str, host: str, port: int,
                 metadata: Dict[str, str] = None) -> str:
        instance_id = f"{service_name}-{host}:{port}"
        now = time.time()
        
        instance = ServiceInstance(
            service_name=service_name,
            instance_id=instance_id,
            host=host,
            port=port,
            metadata=metadata or {},
            registered_at=now,
            last_heartbeat=now,
        )
        
        with self._lock:
            self._instances[service_name][instance_id] = instance
        
        return instance_id
    
    def deregister(self, service_name: str, instance_id: str):
        with self._lock:
            self._instances[service_name].pop(instance_id, None)
    
    def heartbeat(self, service_name: str, instance_id: str):
        with self._lock:
            instance = self._instances[service_name].get(instance_id)
            if instance:
                instance.last_heartbeat = time.time()
                instance.healthy = True
    
    def get_instances(self, service_name: str,
                      healthy_only: bool = True) -> List[ServiceInstance]:
        with self._lock:
            instances = list(self._instances.get(service_name, {}).values())
            
            if healthy_only:
                now = time.time()
                healthy = []
                for inst in instances:
                    if now - inst.last_heartbeat > self.heartbeat_timeout:
                        inst.healthy = False
                    if inst.healthy:
                        healthy.append(inst)
                return healthy
            
            return instances
    
    def get_all_services(self) -> Dict[str, int]:
        with self._lock:
            return {name: len(instances)
                    for name, instances in self._instances.items()}


# ============================================================
# API Gateway
# ============================================================

@dataclass
class Route:
    path_prefix: str
    service_name: str
    strip_prefix: bool = True
    rate_limit: Optional[int] = None
    timeout: float = 30.0
    auth_required: bool = True
    methods: Set[str] = field(default_factory=lambda: {"GET", "POST", "PUT", "DELETE"})


@dataclass
class GatewayRequest:
    method: str
    path: str
    headers: Dict[str, str]
    body: Optional[str] = None
    query_params: Dict[str, str] = field(default_factory=dict)


@dataclass
class GatewayResponse:
    status_code: int
    headers: Dict[str, str]
    body: str


class APIGateway:
    """API Gateway with routing, auth, and rate limiting."""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.routes: List[Route] = []
        self._rate_limits: Dict[str, deque] = defaultdict(deque)
        self._auth_validator: Optional[Callable] = None
        self._middleware: List[Callable] = []
    
    def add_route(self, route: Route):
        self.routes.append(route)
        self.routes.sort(key=lambda r: len(r.path_prefix), reverse=True)
    
    def set_auth_validator(self, validator: Callable[[str], Optional[Dict]]):
        self._auth_validator = validator
    
    def add_middleware(self, middleware: Callable):
        self._middleware.append(middleware)
    
    def handle_request(self, request: GatewayRequest) -> GatewayResponse:
        # Find matching route
        route = self._find_route(request.path, request.method)
        if route is None:
            return GatewayResponse(404, {}, "Not Found")
        
        # Authentication
        if route.auth_required:
            token = request.headers.get("Authorization", "")
            if self._auth_validator:
                user = self._auth_validator(token)
                if user is None:
                    return GatewayResponse(401, {}, "Unauthorized")
                request.headers["X-User-ID"] = str(user.get("id", ""))
        
        # Rate limiting
        if route.rate_limit:
            client_id = request.headers.get("X-Client-ID", "anonymous")
            if not self._check_rate_limit(client_id, route.rate_limit):
                return GatewayResponse(429, {
                    "Retry-After": "60"
                }, "Rate Limit Exceeded")
        
        # Route to service
        instances = self.registry.get_instances(route.service_name)
        if not instances:
            return GatewayResponse(503, {}, "Service Unavailable")
        
        # Select instance (round-robin)
        instance = random.choice(instances)
        
        # Forward request
        return self._forward_request(request, instance, route)
    
    def _find_route(self, path: str, method: str) -> Optional[Route]:
        for route in self.routes:
            if path.startswith(route.path_prefix):
                if method in route.methods:
                    return route
        return None
    
    def _check_rate_limit(self, client_id: str, limit: int) -> bool:
        now = time.time()
        window = self._rate_limits[client_id]
        
        while window and window[0] < now - 60:
            window.popleft()
        
        if len(window) >= limit:
            return False
        
        window.append(now)
        return True
    
    def _forward_request(self, request: GatewayRequest,
                        instance: ServiceInstance,
                        route: Route) -> GatewayResponse:
        # Simulate forwarding
        request.headers["X-Forwarded-For"] = "gateway"
        request.headers["X-Request-ID"] = str(uuid.uuid4())
        
        return GatewayResponse(
            200,
            {"X-Served-By": instance.instance_id},
            f"Response from {instance.instance_id}"
        )


# ============================================================
# Distributed Tracing
# ============================================================

@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    service_name: str
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    
    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """Distributed tracing system."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._spans: Dict[str, List[Span]] = defaultdict(list)
        self._active_span: Optional[Span] = None
    
    def start_trace(self, operation: str) -> Span:
        trace_id = str(uuid.uuid4()).replace('-', '')[:16]
        return self.start_span(trace_id, operation)
    
    def start_span(self, trace_id: str, operation: str,
                   parent_span_id: str = None) -> Span:
        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()).replace('-', '')[:16],
            parent_span_id=parent_span_id or (
                self._active_span.span_id if self._active_span else None),
            service_name=self.service_name,
            operation_name=operation,
            start_time=time.time(),
        )
        
        self._active_span = span
        return span
    
    def finish_span(self, span: Span, status: str = "ok"):
        span.end_time = time.time()
        span.status = status
        self._spans[span.trace_id].append(span)
        
        if self._active_span == span:
            self._active_span = None
    
    def add_tag(self, span: Span, key: str, value: str):
        span.tags[key] = value
    
    def add_log(self, span: Span, message: str, **kwargs):
        span.logs.append({
            "timestamp": time.time(),
            "message": message,
            **kwargs,
        })
    
    def get_trace(self, trace_id: str) -> List[Span]:
        return self._spans.get(trace_id, [])
    
    def inject_context(self, span: Span) -> Dict[str, str]:
        return {
            "X-Trace-ID": span.trace_id,
            "X-Span-ID": span.span_id,
            "X-Parent-Span-ID": span.parent_span_id or "",
        }
    
    def extract_context(self, headers: Dict[str, str]) -> Tuple[str, str]:
        return (
            headers.get("X-Trace-ID", ""),
            headers.get("X-Span-ID", ""),
        )


# ============================================================
# Health Check Aggregator
# ============================================================

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0
    checked_at: float = 0


class HealthChecker:
    """Aggregate health checks from dependencies."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._checks: Dict[str, Callable[[], HealthCheck]] = {}
    
    def register_check(self, name: str,
                       check_fn: Callable[[], HealthCheck]):
        self._checks[name] = check_fn
    
    def check(self) -> Dict[str, Any]:
        results = {}
        overall = HealthStatus.HEALTHY
        
        for name, check_fn in self._checks.items():
            start = time.time()
            try:
                result = check_fn()
                result.latency_ms = (time.time() - start) * 1000
                result.checked_at = time.time()
            except Exception as e:
                result = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                    latency_ms=(time.time() - start) * 1000,
                    checked_at=time.time(),
                )
            
            results[name] = {
                "status": result.status.value,
                "message": result.message,
                "latency_ms": round(result.latency_ms, 2),
            }
            
            if result.status == HealthStatus.UNHEALTHY:
                overall = HealthStatus.UNHEALTHY
            elif (result.status == HealthStatus.DEGRADED and
                  overall != HealthStatus.UNHEALTHY):
                overall = HealthStatus.DEGRADED
        
        return {
            "service": self.service_name,
            "status": overall.value,
            "checks": results,
            "timestamp": time.time(),
        }


# ============================================================
# Sidecar Proxy (Service Mesh Data Plane)
# ============================================================

@dataclass
class ProxyConfig:
    service_name: str
    listen_port: int
    retry_attempts: int = 3
    timeout: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    rate_limit: Optional[int] = None
    mtls_enabled: bool = True


class SidecarProxy:
    """Simplified service mesh sidecar proxy."""
    
    def __init__(self, config: ProxyConfig, registry: ServiceRegistry):
        self.config = config
        self.registry = registry
        self._circuit_breakers: Dict[str, Dict] = defaultdict(
            lambda: {"failures": 0, "state": "closed", "last_failure": 0})
        self._metrics: Dict[str, Dict] = defaultdict(
            lambda: {"requests": 0, "errors": 0, "total_latency": 0})
    
    def proxy_request(self, target_service: str,
                     request: Dict[str, Any]) -> Dict[str, Any]:
        # Check circuit breaker
        cb = self._circuit_breakers[target_service]
        if cb["state"] == "open":
            if time.time() - cb["last_failure"] > self.config.circuit_breaker_timeout:
                cb["state"] = "half-open"
            else:
                return {"error": "Circuit breaker open", "status": 503}
        
        # Rate limiting
        if self.config.rate_limit:
            metrics = self._metrics[target_service]
            # Simple check
            if metrics["requests"] > self.config.rate_limit:
                return {"error": "Rate limited", "status": 429}
        
        # Get instances
        instances = self.registry.get_instances(target_service)
        if not instances:
            return {"error": "No instances available", "status": 503}
        
        # Retry loop
        last_error = None
        for attempt in range(self.config.retry_attempts):
            instance = instances[attempt % len(instances)]
            
            start = time.time()
            try:
                # Simulate request forwarding
                result = self._forward(instance, request)
                latency = time.time() - start
                
                self._record_success(target_service, latency)
                
                if cb["state"] == "half-open":
                    cb["state"] = "closed"
                    cb["failures"] = 0
                
                return result
                
            except Exception as e:
                last_error = e
                self._record_failure(target_service)
        
        return {"error": str(last_error), "status": 502}
    
    def _forward(self, instance: ServiceInstance,
                request: Dict[str, Any]) -> Dict[str, Any]:
        # Add tracing headers
        request.setdefault("headers", {})
        request["headers"]["X-Proxy"] = self.config.service_name
        
        if self.config.mtls_enabled:
            request["headers"]["X-mTLS"] = "verified"
        
        return {
            "status": 200,
            "body": f"Response from {instance.instance_id}",
            "headers": {"X-Served-By": instance.instance_id},
        }
    
    def _record_success(self, service: str, latency: float):
        metrics = self._metrics[service]
        metrics["requests"] += 1
        metrics["total_latency"] += latency
    
    def _record_failure(self, service: str):
        metrics = self._metrics[service]
        metrics["requests"] += 1
        metrics["errors"] += 1
        
        cb = self._circuit_breakers[service]
        cb["failures"] += 1
        cb["last_failure"] = time.time()
        
        if cb["failures"] >= self.config.circuit_breaker_threshold:
            cb["state"] = "open"
    
    def get_metrics(self, service: str) -> Dict[str, Any]:
        m = self._metrics[service]
        total = m["requests"]
        return {
            "requests": total,
            "errors": m["errors"],
            "error_rate": m["errors"] / max(total, 1),
            "avg_latency_ms": (m["total_latency"] / max(total, 1)) * 1000,
        }`,
				},
			},
		},
	})
}
