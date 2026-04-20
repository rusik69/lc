package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1732,
			Title:       "Service Mesh and Advanced Traffic Management",
			Description: "Understand service mesh architecture with Istio and Linkerd, advanced traffic routing, mTLS, observability, and traffic management patterns.",
			Order:       32,
			Lessons: []problems.Lesson{
				{
					Title: "Service Mesh Fundamentals",
					Content: `A service mesh provides infrastructure-level capabilities for service-to-service communication: encryption, observability, traffic control, and resilience — without modifying application code.

**Why Service Mesh:**
` + "```" + `
Problems solved:
  - Mutual TLS between all services (zero-trust networking)
  - Distributed tracing across services
  - Fine-grained traffic routing (canary, A/B, mirroring)
  - Retry, timeout, and circuit breaking policies
  - Service-to-service authorization policies
  - Request-level metrics and observability

Without service mesh:
  Each service implements its own:
    - TLS configuration
    - Retry logic
    - Circuit breakers
    - Metrics collection
    - Tracing headers propagation
  → Inconsistent, error-prone, language-dependent

With service mesh:
  Sidecar proxy handles all of the above
    - Transparent to application
    - Consistent across all services
    - Language-agnostic
    - Centrally configured
` + "```" + `

**Architecture:**
` + "```" + `
Service Mesh Components:

  Data Plane (sidecars):
    - Envoy proxy (Istio) or linkerd2-proxy (Linkerd)
    - Injected alongside each pod
    - Intercepts all inbound/outbound traffic
    - Handles: TLS, routing, retries, metrics, tracing
    
  Control Plane:
    - Istiod (Istio) or linkerd-destination/identity (Linkerd)
    - Manages proxy configuration
    - Certificate authority for mTLS
    - Service discovery
    - Policy distribution

Traffic flow with sidecar:
  Client Pod
    → Client sidecar proxy (outbound)
      → Network
        → Server sidecar proxy (inbound)
          → Server Pod

Sidecar injection:
  Automatic (namespace label):
    kubectl label namespace myapp istio-injection=enabled
    # or for Linkerd:
    kubectl annotate namespace myapp linkerd.io/inject=enabled
  
  Manual:
    istioctl kube-inject -f deployment.yaml | kubectl apply -f -
    # or:
    kubectl get deploy myapp -o yaml | linkerd inject - | kubectl apply -f -
` + "```" + `

**Istio Core Concepts:**
` + "```" + `
Istio Custom Resources:

  VirtualService:
    - How requests are routed to a service
    - Traffic splitting, header-based routing
    - Retries, timeouts, fault injection
  
  DestinationRule:
    - Policies applied after routing
    - Load balancing algorithm
    - Connection pool settings
    - TLS settings for upstream
    - Outlier detection (circuit breaker)
  
  Gateway:
    - Configures ingress/egress at mesh boundary
    - Like Ingress but more powerful
    - Supports TCP, HTTP, gRPC, WebSocket
  
  ServiceEntry:
    - Add external services to the mesh
    - Apply mesh policies to external traffic
  
  PeerAuthentication:
    - mTLS mode: STRICT, PERMISSIVE, DISABLE
    - Per-namespace or mesh-wide
  
  AuthorizationPolicy:
    - Service-to-service access control
    - Allow/deny based on source, operation, conditions
    - Request-level: JWT claims, headers
  
  Sidecar:
    - Configure sidecar proxy behavior
    - Limit services the proxy knows about
    - Reduce memory footprint
` + "```" + `

**Istio vs Linkerd:**
` + "```" + `
Feature comparison:

  Istio:
    + More features (rich traffic management)
    + Larger ecosystem and community
    + Envoy proxy (highly configurable)
    + Multi-cluster, multi-mesh
    - More complex to operate
    - Higher resource overhead
    - Steeper learning curve
    
  Linkerd:
    + Simpler, easier to operate
    + Lower resource overhead
    + Rust-based proxy (fast, safe)
    + Simpler mental model
    + Graduated CNCF project
    - Fewer traffic management features
    - Smaller ecosystem
    - Less configurability
    
  Choose Istio when:
    - Complex traffic routing needed
    - Large-scale multi-cluster
    - Need Envoy filter chains
    - Team has Istio expertise
    
  Choose Linkerd when:
    - Simplicity is priority
    - Resource-constrained environments
    - Quick time to value
    - Team is new to service mesh
` + "```" + `

**mTLS Deep Dive:**
` + "```" + `
Mutual TLS (mTLS):
  - Both client and server present certificates
  - Verifies identity of BOTH sides
  - Encrypts all traffic in transit
  - Service mesh handles certificate lifecycle

Certificate lifecycle:
  1. Identity: each pod gets a unique certificate
     - Based on service account (SPIFFE identity)
     - spiffe://cluster.local/ns/<namespace>/sa/<service-account>
  2. Issuance: CA in control plane signs certificates
  3. Rotation: automatic, short-lived (default: 24h Istio)
  4. Validation: sidecar verifies peer certificate

Migration to strict mTLS:
  Phase 1: PERMISSIVE mode (accepts both plain and mTLS)
    - Install mesh with permissive mode
    - All sidecars injected
    - Services communicate normally
    
  Phase 2: Monitor and verify
    - Check that all traffic is using mTLS
    - Identify services without sidecars
    - Grafana dashboard: % encrypted traffic
    
  Phase 3: STRICT mode (requires mTLS)
    - Set PeerAuthentication to STRICT
    - Non-mesh clients are rejected
    - Full zero-trust networking achieved
` + "```" + ``,
					CodeExamples: `# Service Mesh Configuration Examples

# 1. Istio installation profile
# istioctl install --set profile=production
# Or with IstioOperator:
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: production-mesh
  namespace: istio-system
spec:
  profile: default
  meshConfig:
    accessLogFile: /dev/stdout
    accessLogFormat: |
      [%START_TIME%] "%REQ(:METHOD)% %REQ(X-ENVOY-ORIGINAL-PATH?:PATH)% %PROTOCOL%"
      %RESPONSE_CODE% %RESPONSE_FLAGS% %BYTES_RECEIVED% %BYTES_SENT%
      %DURATION% "%REQ(X-FORWARDED-FOR)%" "%REQ(USER-AGENT)%"
      "%REQ(X-REQUEST-ID)%" "%REQ(:AUTHORITY)%"
    enableTracing: true
    defaultConfig:
      tracing:
        sampling: 10.0  # 10% sampling in production
      holdApplicationUntilProxyStarts: true
      proxyMetadata:
        ISTIO_META_DNS_CAPTURE: "true"
        ISTIO_META_DNS_AUTO_ALLOCATE: "true"
  components:
    pilot:
      k8s:
        resources:
          requests:
            cpu: 500m
            memory: 2Gi
          limits:
            memory: 2Gi
        hpaSpec:
          minReplicas: 2
          maxReplicas: 5
    ingressGateways:
    - name: istio-ingressgateway
      enabled: true
      k8s:
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
        hpaSpec:
          minReplicas: 2
          maxReplicas: 10
        service:
          type: LoadBalancer

---
# 2. Enable strict mTLS mesh-wide
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: istio-system  # Mesh-wide
spec:
  mtls:
    mode: STRICT

---
# Per-namespace permissive (during migration)
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: legacy-apps
spec:
  mtls:
    mode: PERMISSIVE

---
# 3. Authorization policies
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: frontend-to-backend
  namespace: production
spec:
  selector:
    matchLabels:
      app: backend-api
  action: ALLOW
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/production/sa/frontend"]
    to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/api/v1/*"]

---
# Deny all by default (zero trust)
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: deny-all
  namespace: production
spec:
  {}  # Empty spec = deny all

---
# 4. Gateway configuration
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: main-gateway
  namespace: production
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: main-tls-cert
    hosts:
    - "api.example.com"
    - "app.example.com"
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*.example.com"
    tls:
      httpsRedirect: true

---
# 5. Sidecar resource limits
apiVersion: networking.istio.io/v1beta1
kind: Sidecar
metadata:
  name: default
  namespace: production
spec:
  egress:
  - hosts:
    - "./*"                    # Same namespace
    - "istio-system/*"         # Istio components
    - "monitoring/*"           # Monitoring stack
  outboundTrafficPolicy:
    mode: REGISTRY_ONLY  # Only allow known services`,
				},
				{
					Title: "Advanced Traffic Management",
					Content: `Service mesh enables sophisticated traffic management patterns that are essential for safe deployments and resilience.

**Traffic Routing:**
` + "```" + `
VirtualService routes traffic based on:
  - URI path
  - HTTP headers
  - Query parameters
  - Source labels
  - Percentage-based splitting

Routing precedence:
  1. Most specific match wins
  2. Exact > prefix > regex
  3. First match in list for same specificity
  4. Default route (no match conditions)

URI matching:
  exact:  /api/v1/users      → only this path
  prefix: /api/v1/            → everything under /api/v1/
  regex:  /api/v[12]/users/.* → pattern match
` + "```" + `

**Canary Deployments with Istio:**
` + "```" + `yaml
# Route 95% to v1, 5% to v2
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
  namespace: production
spec:
  hosts:
  - reviews
  http:
  - route:
    - destination:
        host: reviews
        subset: v1
      weight: 95
    - destination:
        host: reviews
        subset: v2
      weight: 5

---
# Define subsets (versions)
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: reviews
  namespace: production
spec:
  host: reviews
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        h2UpgradePolicy: DEFAULT
        http1MaxPendingRequests: 100
        http2MaxRequests: 100
    outlierDetection:
      consecutive5xxErrors: 3
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
` + "```" + `

**Header-Based Routing:**
` + "```" + `yaml
# Route beta users to v2
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
  namespace: production
spec:
  hosts:
  - reviews
  http:
  # Beta users get v2
  - match:
    - headers:
        x-user-group:
          exact: beta
    route:
    - destination:
        host: reviews
        subset: v2
  # Internal testing
  - match:
    - headers:
        x-debug:
          exact: "true"
    route:
    - destination:
        host: reviews
        subset: v2
  # Everyone else gets v1
  - route:
    - destination:
        host: reviews
        subset: v1
` + "```" + `

**Traffic Mirroring (Shadow Traffic):**
` + "```" + `yaml
# Mirror production traffic to v2 for testing
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
  namespace: production
spec:
  hosts:
  - reviews
  http:
  - route:
    - destination:
        host: reviews
        subset: v1
    mirror:
      host: reviews
      subset: v2
    mirrorPercentage:
      value: 100.0
  # Mirror traffic:
  # - Fire-and-forget (response discarded)
  # - Does NOT affect real traffic
  # - Great for testing new version with real traffic patterns
  # - Compare: latency, errors, resource usage
` + "```" + `

**Resilience Patterns:**
` + "```" + `
Retries:
  - Automatic retry on transient failures
  - Configure: attempts, per-try timeout, retry conditions
  - Exponential backoff with jitter
  
  http:
  - route:
    - destination:
        host: reviews
    retries:
      attempts: 3
      perTryTimeout: 2s
      retryOn: "5xx,reset,connect-failure,retriable-4xx"
      retryRemoteLocalities: true

Timeouts:
  - Prevent hanging requests
  - Should be end-to-end meaningful
  - Shorter than client timeout

  http:
  - route:
    - destination:
        host: reviews
    timeout: 10s

Circuit breaking:
  - Prevent cascading failures
  - Stop sending requests to unhealthy instances
  - Configured via DestinationRule outlierDetection
  
  trafficPolicy:
    outlierDetection:
      consecutive5xxErrors: 5     # Errors before ejection
      interval: 10s               # Analysis interval
      baseEjectionTime: 30s       # Minimum ejection time
      maxEjectionPercent: 50      # Max % of hosts ejected
      minHealthPercent: 30        # Don't eject if too few healthy

Connection pool:
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100       # Max TCP connections
        connectTimeout: 5s
      http:
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
        maxRequestsPerConnection: 10
        maxRetries: 3
` + "```" + `

**Fault Injection for Testing:**
` + "```" + `yaml
# Test resilience by injecting faults
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
  namespace: staging
spec:
  hosts:
  - reviews
  http:
  - fault:
      delay:
        percentage:
          value: 10.0    # 10% of requests
        fixedDelay: 5s   # 5 second delay
      abort:
        percentage:
          value: 5.0     # 5% of requests
        httpStatus: 503  # Return 503
    route:
    - destination:
        host: reviews
        subset: v1

# Use cases:
# - Test timeout handling
# - Verify circuit breaker triggers
# - Test retry behavior
# - Validate fallback mechanisms
# - Chaos engineering
` + "```" + `

**Rate Limiting:**
` + "```" + `yaml
# Istio rate limiting with EnvoyFilter
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: ratelimit-filter
  namespace: istio-system
spec:
  workloadSelector:
    labels:
      istio: ingressgateway
  configPatches:
  - applyTo: HTTP_FILTER
    match:
      context: GATEWAY
      listener:
        filterChain:
          filter:
            name: envoy.filters.network.http_connection_manager
    patch:
      operation: INSERT_BEFORE
      value:
        name: envoy.filters.http.ratelimit
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.http.ratelimit.v3.RateLimit
          domain: production-ratelimit
          failure_mode_deny: false
          rate_limit_service:
            grpc_service:
              envoy_grpc:
                cluster_name: rate_limit_cluster
            transport_api_version: V3
` + "```" + `

**Observability with Service Mesh:**
` + "```" + `
Built-in metrics (automatic, no code changes):
  Request volume:    istio_requests_total
  Request duration:  istio_request_duration_milliseconds
  Request size:      istio_request_bytes
  Response size:     istio_response_bytes
  TCP connections:   istio_tcp_connections_opened_total

Labels on all metrics:
  source_workload, destination_workload
  source_namespace, destination_namespace
  request_protocol, response_code
  connection_security_policy (mutual_tls/none)

Grafana dashboards (built-in):
  - Mesh overview: total requests, errors, latency
  - Service dashboard: per-service metrics
  - Workload dashboard: per-pod metrics

Kiali (service mesh console):
  - Service topology visualization
  - Traffic flow animation
  - Health status
  - Configuration validation
  - Distributed tracing integration

Distributed tracing:
  - Jaeger or Zipkin integration
  - Automatic span creation at sidecar
  - Application must propagate trace headers:
    x-request-id
    x-b3-traceid
    x-b3-spanid
    x-b3-parentspanid
    x-b3-sampled
    x-b3-flags
    traceparent (W3C)
    tracestate (W3C)
` + "```" + ``,
					CodeExamples: `# Advanced Traffic Management Configuration

# 1. Complete canary deployment with Flagger (Istio)
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: reviews
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: reviews
  progressDeadlineSeconds: 600
  service:
    port: 9080
    targetPort: 9080
    gateways:
    - main-gateway
    hosts:
    - reviews.example.com
    trafficPolicy:
      tls:
        mode: ISTIO_MUTUAL
  analysis:
    # Schedule
    interval: 1m
    threshold: 5          # Max failed checks
    maxWeight: 50         # Max canary traffic %
    stepWeight: 10        # Increment per step
    # Metrics checks
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99           # 99% success required
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 500          # p99 < 500ms
      interval: 1m
    # Webhooks for custom checks
    webhooks:
    - name: smoke-test
      type: pre-rollout
      url: http://flagger-loadtester.test/
      timeout: 30s
      metadata:
        type: bash
        cmd: "curl -s http://reviews-canary.production:9080/health"
    - name: load-test
      url: http://flagger-loadtester.test/
      timeout: 60s
      metadata:
        type: cmd
        cmd: "hey -z 1m -q 10 -c 2 http://reviews-canary.production:9080/"

---
# 2. Multi-destination routing with retries and timeouts
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: api-routing
  namespace: production
spec:
  hosts:
  - api.example.com
  gateways:
  - main-gateway
  http:
  # User service routes
  - match:
    - uri:
        prefix: /api/v1/users
    route:
    - destination:
        host: user-service.production.svc.cluster.local
        port:
          number: 8080
    timeout: 10s
    retries:
      attempts: 3
      perTryTimeout: 3s
      retryOn: "5xx,reset,connect-failure"
    corsPolicy:
      allowOrigins:
      - exact: "https://app.example.com"
      allowMethods:
      - GET
      - POST
      - PUT
      - DELETE
      allowHeaders:
      - authorization
      - content-type
      maxAge: "24h"
  # Order service routes  
  - match:
    - uri:
        prefix: /api/v1/orders
    route:
    - destination:
        host: order-service.production.svc.cluster.local
        port:
          number: 8080
    timeout: 30s  # Longer timeout for order processing
    retries:
      attempts: 2
      perTryTimeout: 10s
      retryOn: "5xx,reset"
  # Default: 404
  - match:
    - uri:
        prefix: /
    directResponse:
      status: 404
      body:
        string: '{"error": "Not Found"}'

---
# 3. Destination rules with locality-aware load balancing
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: user-service
  namespace: production
spec:
  host: user-service.production.svc.cluster.local
  trafficPolicy:
    loadBalancer:
      localityLbSetting:
        enabled: true
        failover:
        - from: us-east-1
          to: us-west-2
      simple: LEAST_REQUEST
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 5s
        tcpKeepalive:
          time: 7200s
          interval: 75s
      http:
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
        maxRequestsPerConnection: 100
        maxRetries: 3
    outlierDetection:
      consecutive5xxErrors: 3
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 30
      minHealthPercent: 50
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
    trafficPolicy:
      connectionPool:
        http:
          http2MaxRequests: 500  # Lower limits during canary

---
# 4. Request authentication with JWT
apiVersion: security.istio.io/v1beta1
kind: RequestAuthentication
metadata:
  name: jwt-auth
  namespace: production
spec:
  selector:
    matchLabels:
      app: api-gateway
  jwtRules:
  - issuer: "https://auth.example.com/"
    jwksUri: "https://auth.example.com/.well-known/jwks.json"
    forwardOriginalToken: true
    outputPayloadToHeader: "x-jwt-payload"

---
# Only allow authenticated requests
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: require-jwt
  namespace: production
spec:
  selector:
    matchLabels:
      app: api-gateway
  action: ALLOW
  rules:
  - from:
    - source:
        requestPrincipals: ["https://auth.example.com/*"]
    when:
    - key: request.auth.claims[groups]
      values: ["admin", "user"]

---
# 5. Linkerd service profile (for comparison)
apiVersion: linkerd.io/v1alpha2
kind: ServiceProfile
metadata:
  name: user-service.production.svc.cluster.local
  namespace: production
spec:
  routes:
  - name: GET /api/v1/users
    condition:
      method: GET
      pathRegex: /api/v1/users
    timeout: 10s
    isRetryable: true
  - name: POST /api/v1/users
    condition:
      method: POST
      pathRegex: /api/v1/users
    timeout: 30s
    isRetryable: false  # Don't retry writes
  retryBudget:
    retryRatio: 0.2     # Max 20% extra load from retries
    minRetriesPerSecond: 10
    ttl: 10s`,
				},
			},
		},
	})
}
