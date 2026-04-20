package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          1458,
			Title:       "Observability and Monitoring at Scale",
			Description: "Build comprehensive observability with metrics, logs, and traces using Prometheus, Grafana, OpenTelemetry, and centralized logging stacks.",
			Order:       58,
			Lessons: []problems.Lesson{
				{
					Title: "Metrics and Alerting with Prometheus",
					Content: `Prometheus provides a powerful metrics collection and alerting system for cloud-native environments.

**Prometheus Architecture:**
` + "```" + `
Components:
  Prometheus Server:
    - Pull-based metric scraping
    - Time-series database (TSDB)
    - PromQL query language
    - Alert evaluation
  
  Pushgateway:
    - For short-lived batch jobs
    - Push metrics (exception to pull model)
  
  Alertmanager:
    - Alert deduplication
    - Grouping and routing
    - Silencing and inhibition
    - Multi-channel notification
  
  Client libraries:
    Go, Java, Python, Ruby, .NET
    Instrument your application code

Metric types:
  Counter:
    Monotonically increasing value.
    http_requests_total, errors_total
    
    from prometheus_client import Counter
    REQUEST_COUNT = Counter('http_requests_total', 'Total requests',
                           ['method', 'endpoint', 'status'])
    REQUEST_COUNT.labels(method='GET', endpoint='/api', status='200').inc()
  
  Gauge:
    Value that goes up and down.
    temperature, memory_usage, active_connections
    
    ACTIVE_REQUESTS = Gauge('active_requests', 'Currently active requests')
    ACTIVE_REQUESTS.inc()
    ACTIVE_REQUESTS.dec()
  
  Histogram:
    Observations distributed into buckets.
    request_duration_seconds, response_size_bytes
    
    REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration',
                                buckets=[0.01, 0.05, 0.1, 0.5, 1, 5])
    REQUEST_DURATION.observe(0.42)
  
  Summary:
    Like histogram but calculates quantiles client-side.
    Less accurate for aggregation across instances.

Scrape configuration:
  # prometheus.yml
  global:
    scrape_interval: 15s
    evaluation_interval: 15s
  
  scrape_configs:
    - job_name: 'prometheus'
      static_configs:
        - targets: ['localhost:9090']
    
    - job_name: 'node-exporter'
      static_configs:
        - targets: ['node1:9100', 'node2:9100']
    
    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
        - role: pod
      relabel_configs:
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
          action: keep
          regex: true
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
          action: replace
          target_label: __address__
          regex: (.+)
          replacement: ${1}:$${2}
    
    - job_name: 'kubernetes-services'
      kubernetes_sd_configs:
        - role: service
      metrics_path: /metrics
      relabel_configs:
        - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scrape]
          action: keep
          regex: true
` + "```" + `

**PromQL:**
` + "```" + `
Basic queries:
  # Instant vector
  http_requests_total
  http_requests_total{method="GET", status="200"}
  
  # Range vector
  http_requests_total[5m]
  
  # Rate (per-second increase)
  rate(http_requests_total[5m])
  
  # Increase (total increase)
  increase(http_requests_total[1h])

Aggregations:
  # Sum by label
  sum(rate(http_requests_total[5m])) by (method)
  
  # Average
  avg(node_memory_MemAvailable_bytes) by (instance)
  
  # Top K
  topk(5, rate(http_requests_total[5m]))
  
  # Count
  count(up == 1) by (job)
  
  # Quantile
  histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
  
  # P99 latency
  histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

Alert rules:
  groups:
    - name: application
      rules:
        - alert: HighErrorRate
          expr: |
            sum(rate(http_requests_total{status=~"5.."}[5m])) 
            / sum(rate(http_requests_total[5m])) > 0.05
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "High error rate ({{ $value | humanizePercentage }})"
        
        - alert: HighLatency
          expr: |
            histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
            > 1.0
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: "P95 latency above 1s ({{ $value }}s)"
        
        - alert: InstanceDown
          expr: up == 0
          for: 3m
          labels:
            severity: critical
          annotations:
            summary: "Instance {{ $labels.instance }} is down"

Recording rules (pre-compute expensive queries):
  groups:
    - name: recording
      rules:
        - record: job:http_requests_total:rate5m
          expr: sum(rate(http_requests_total[5m])) by (job)
        
        - record: job:http_request_duration_seconds:p95
          expr: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (job, le))

Alertmanager configuration:
  route:
    receiver: default
    group_by: [alertname, cluster]
    group_wait: 30s
    group_interval: 5m
    repeat_interval: 4h
    routes:
      - match:
          severity: critical
        receiver: pagerduty
        continue: true
      - match:
          severity: warning
        receiver: slack
  
  receivers:
    - name: default
      email_configs:
        - to: 'team@example.com'
    
    - name: pagerduty
      pagerduty_configs:
        - service_key: '<key>'
    
    - name: slack
      slack_configs:
        - api_url: 'https://hooks.slack.com/...'
          channel: '#alerts'
          title: '{{ .GroupLabels.alertname }}'
          text: '{{ .CommonAnnotations.summary }}'
  
  inhibit_rules:
    - source_match:
        severity: critical
      target_match:
        severity: warning
      equal: [alertname, cluster]
` + "```" + ``,
					CodeExamples: `# Monitoring scripts

# 1. Prometheus health check
#!/bin/bash
echo "=== Prometheus Stack Health ==="

PROM_URL="${PROMETHEUS_URL:-http://localhost:9090}"
AM_URL="${ALERTMANAGER_URL:-http://localhost:9093}"

# Prometheus targets
echo "--- Scrape Targets ---"
curl -s "$PROM_URL/api/v1/targets" 2>/dev/null | \
    jq -r '.data.activeTargets[] | "\(.labels.job): \(.health) (\(.lastScrape))"' 2>/dev/null | head -15

# Unhealthy targets
echo ""
echo "--- Unhealthy Targets ---"
curl -s "$PROM_URL/api/v1/targets" 2>/dev/null | \
    jq -r '.data.activeTargets[] | select(.health != "up") | "\(.labels.job)/\(.labels.instance): \(.health)"' 2>/dev/null

# Active alerts
echo ""
echo "--- Active Alerts ---"
curl -s "$PROM_URL/api/v1/alerts" 2>/dev/null | \
    jq -r '.data.alerts[] | select(.state == "firing") | "\(.labels.alertname): \(.annotations.summary)"' 2>/dev/null | head -10

# Alertmanager
echo ""
echo "--- Alertmanager Status ---"
curl -s "$AM_URL/api/v2/status" 2>/dev/null | jq '.cluster' 2>/dev/null

# Silences
echo ""
echo "--- Active Silences ---"
curl -s "$AM_URL/api/v2/silences" 2>/dev/null | \
    jq -r '.[] | select(.status.state == "active") | "\(.createdBy): \(.comment)"' 2>/dev/null

# 2. SLO checker
#!/bin/bash
echo "=== SLO Status ==="

PROM_URL="${PROMETHEUS_URL:-http://localhost:9090}"

# Error budget for 99.9% availability
QUERY='1 - (sum(rate(http_requests_total{status=~"5.."}[30d])) / sum(rate(http_requests_total[30d])))'
AVAILABILITY=$(curl -s "$PROM_URL/api/v1/query?query=$(echo "$QUERY" | jq -sRr @uri)" 2>/dev/null | \
    jq -r '.data.result[0].value[1]' 2>/dev/null)

if [ -n "$AVAILABILITY" ] && [ "$AVAILABILITY" != "null" ]; then
    AVAIL_PCT=$(echo "scale=4; $AVAILABILITY * 100" | bc 2>/dev/null)
    echo "  Availability (30d): ${AVAIL_PCT}%"
    echo "  SLO Target: 99.9%"
    
    BUDGET=$(echo "scale=4; ($AVAILABILITY - 0.999) * 100" | bc 2>/dev/null)
    echo "  Error Budget Remaining: ${BUDGET}%"
fi

# Latency SLO (P99 < 500ms)
QUERY='histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[30d])) by (le))'
P99_LATENCY=$(curl -s "$PROM_URL/api/v1/query?query=$(echo "$QUERY" | jq -sRr @uri)" 2>/dev/null | \
    jq -r '.data.result[0].value[1]' 2>/dev/null)

if [ -n "$P99_LATENCY" ] && [ "$P99_LATENCY" != "null" ]; then
    P99_MS=$(echo "scale=1; $P99_LATENCY * 1000" | bc 2>/dev/null)
    echo "  P99 Latency (30d): ${P99_MS}ms"
    echo "  SLO Target: <500ms"
fi

# 3. Alert rule validator
#!/bin/bash
echo "=== Alert Rule Validation ==="

PROM_URL="${PROMETHEUS_URL:-http://localhost:9090}"

# Check all rules
echo "--- Rule Groups ---"
curl -s "$PROM_URL/api/v1/rules" 2>/dev/null | \
    jq -r '.data.groups[] | "\(.name): \(.rules | length) rules"' 2>/dev/null

# Inactive alerts (rules that never fire)
echo ""
echo "--- Alerting Rules ---"
curl -s "$PROM_URL/api/v1/rules?type=alert" 2>/dev/null | \
    jq -r '.data.groups[].rules[] | "\(.name): \(.state) (evals: \(.evaluationTime)s)"' 2>/dev/null | head -15`,
				},
				{
					Title: "Distributed Tracing and OpenTelemetry",
					Content: `OpenTelemetry provides a unified standard for collecting metrics, logs, and traces across distributed systems.

**OpenTelemetry Fundamentals:**
` + "```" + `
OpenTelemetry (OTel):
  Vendor-neutral observability framework.
  
  Signals:
    Traces:  Request flow across services
    Metrics: Numeric measurements
    Logs:    Structured log events
  
  Components:
    API:         Instrumentation interface
    SDK:         Implementation of API
    Collector:   Receive, process, export telemetry
    Exporters:   Send data to backends (Jaeger, Prometheus, etc.)

Trace anatomy:
  Trace (end-to-end request):
    └── Span: API Gateway (root span)
        ├── Span: Auth Service
        │   └── Span: Redis lookup
        ├── Span: Order Service
        │   ├── Span: Database query
        │   └── Span: Payment Service (HTTP call)
        │       └── Span: Stripe API call
        └── Span: Notification Service
            └── Span: Send email (async)
  
  Span attributes:
    Trace ID:     Unique across entire request
    Span ID:      Unique for this span
    Parent Span:  Parent span ID
    Name:         Operation name
    Start/End:    Timestamps
    Status:       OK, Error, Unset
    Attributes:   Key-value metadata
    Events:       Timestamped annotations
    Links:        Related spans (batch processing)

Context propagation:
  W3C Trace Context (standard):
    traceparent: 00-<trace-id>-<span-id>-<flags>
    tracestate: vendor1=value1,vendor2=value2
  
  B3 (Zipkin):
    X-B3-TraceId, X-B3-SpanId, X-B3-ParentSpanId, X-B3-Sampled
  
  Propagation injects context into:
    HTTP headers
    gRPC metadata
    Message queue headers
    Kafka record headers

Sampling strategies:
  AlwaysOn:        100% sampling (expensive)
  AlwaysOff:       No sampling
  TraceIdRatio:    Sample N% of traces
  ParentBased:     Follow parent span's decision
  
  Head-based: Decide at start of trace
  Tail-based: Decide after trace completes (Collector)
    Keep: errors, slow requests, specific attributes
    Drop: healthy, fast requests
` + "```" + `

**OpenTelemetry Collector:**
` + "```" + `
Collector pipeline:
  Receivers → Processors → Exporters
  
  Configuration:
  # otel-collector-config.yaml
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317
        http:
          endpoint: 0.0.0.0:4318
    
    prometheus:
      config:
        scrape_configs:
          - job_name: 'app'
            scrape_interval: 15s
            static_configs:
              - targets: ['app:8080']
    
    filelog:
      include: [/var/log/app/*.log]
      operators:
        - type: json_parser
          timestamp:
            parse_from: attributes.timestamp
            layout: '%Y-%m-%dT%H:%M:%S.%LZ'
  
  processors:
    batch:
      timeout: 5s
      send_batch_size: 1024
    
    memory_limiter:
      limit_mib: 512
      spike_limit_mib: 128
    
    attributes:
      actions:
        - key: environment
          value: production
          action: upsert
    
    filter:
      error_mode: ignore
      traces:
        span:
          - 'attributes["http.target"] == "/health"'
    
    tail_sampling:
      policies:
        - name: errors
          type: status_code
          status_code: {status_codes: [ERROR]}
        - name: slow
          type: latency
          latency: {threshold_ms: 5000}
        - name: percentage
          type: probabilistic
          probabilistic: {sampling_percentage: 10}
  
  exporters:
    otlp:
      endpoint: tempo:4317
      tls:
        insecure: true
    
    prometheus:
      endpoint: 0.0.0.0:8889
    
    loki:
      endpoint: http://loki:3100/loki/api/v1/push
    
    debug:
      verbosity: detailed
  
  service:
    pipelines:
      traces:
        receivers: [otlp]
        processors: [memory_limiter, batch, tail_sampling]
        exporters: [otlp]
      metrics:
        receivers: [otlp, prometheus]
        processors: [memory_limiter, batch]
        exporters: [prometheus]
      logs:
        receivers: [otlp, filelog]
        processors: [memory_limiter, batch, attributes]
        exporters: [loki]

Deployment patterns:
  Agent (per-node):
    DaemonSet on each node
    Collects from local pods
    Forwards to central collector
  
  Gateway (centralized):
    Deployment with multiple replicas
    Receives from agents or directly
    Applies global processing
    Exports to backends
  
  Sidecar:
    Per-pod collector
    Most isolation
    Highest resource usage
` + "```" + `

**Centralized Logging:**
` + "```" + `
Logging stacks:
  ELK (Elastic):
    Elasticsearch: Storage and search
    Logstash:      Processing pipeline
    Kibana:        Visualization
    Filebeat:      Log shipping agent
  
  PLG (Grafana):
    Promtail:  Log collection agent
    Loki:      Log aggregation (like Prometheus for logs)
    Grafana:   Visualization
  
  EFK:
    Elasticsearch: Storage
    Fluentd/Fluent Bit: Collection
    Kibana: Visualization

Loki configuration:
  Promtail (agent):
    server:
      http_listen_port: 9080
    positions:
      filename: /tmp/positions.yaml
    clients:
      - url: http://loki:3100/loki/api/v1/push
    scrape_configs:
      - job_name: kubernetes
        kubernetes_sd_configs:
          - role: pod
        pipeline_stages:
          - docker: {}
          - json:
              expressions:
                level: level
                msg: msg
          - labels:
              level:
  
  LogQL queries:
    # Search logs
    {namespace="production"} |= "error"
    
    # JSON parsing
    {app="api"} | json | level="error"
    
    # Rate of errors
    rate({app="api"} | json | level="error" [5m])
    
    # Top error messages
    topk(10, sum by (msg) (count_over_time({app="api"} | json | level="error" [1h])))

Structured logging best practices:
  DO:
    - Use JSON format
    - Include trace_id, span_id
    - Include request_id
    - Use consistent field names
    - Log at appropriate levels
    - Include context (user, request)
  
  DON'T:
    - Log sensitive data (PII, secrets)
    - Log at DEBUG in production
    - Use unstructured text
    - Log in hot loops
    - Include large payloads
  
  Example structured log:
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "level": "error",
      "message": "Failed to process order",
      "service": "order-service",
      "trace_id": "abc123def456",
      "span_id": "789xyz",
      "request_id": "req-456",
      "user_id": "user-789",
      "order_id": "order-123",
      "error": "payment declined",
      "duration_ms": 1523
    }
` + "```" + ``,
					CodeExamples: `# Observability stack management

# 1. Observability stack health
#!/bin/bash
echo "=== Observability Stack Health ==="

# Prometheus
echo "--- Prometheus ---"
PROM_URL="${PROMETHEUS_URL:-http://localhost:9090}"
PROM_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$PROM_URL/-/healthy" 2>/dev/null)
echo "  Status: HTTP $PROM_STATUS"
TARGETS=$(curl -s "$PROM_URL/api/v1/targets" 2>/dev/null | jq '.data.activeTargets | length' 2>/dev/null)
echo "  Active targets: $TARGETS"

# Grafana
echo ""
echo "--- Grafana ---"
GRAF_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAF_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$GRAF_URL/api/health" 2>/dev/null)
echo "  Status: HTTP $GRAF_STATUS"

# Loki
echo ""
echo "--- Loki ---"
LOKI_URL="${LOKI_URL:-http://localhost:3100}"
LOKI_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$LOKI_URL/ready" 2>/dev/null)
echo "  Status: HTTP $LOKI_STATUS"

# Tempo/Jaeger (tracing)
echo ""
echo "--- Tracing Backend ---"
JAEGER_URL="${JAEGER_URL:-http://localhost:16686}"
TRACE_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$JAEGER_URL/" 2>/dev/null)
echo "  Status: HTTP $TRACE_STATUS"

# OTel Collector
echo ""
echo "--- OTel Collector ---"
OTEL_URL="${OTEL_URL:-http://localhost:13133}"
OTEL_STATUS=$(curl -s "$OTEL_URL" 2>/dev/null | jq '.status' 2>/dev/null)
echo "  Status: $OTEL_STATUS"

# 2. Log analysis
#!/bin/bash
echo "=== Log Analysis ==="

LOKI_URL="${LOKI_URL:-http://localhost:3100}"

# Error count by service (last 1h)
echo "--- Errors by Service (1h) ---"
curl -s "$LOKI_URL/loki/api/v1/query" \
    --data-urlencode 'query=sum by (app) (count_over_time({level="error"}[1h]))' \
    2>/dev/null | jq -r '.data.result[] | "\(.metric.app): \(.value[1])"' 2>/dev/null

# Recent errors
echo ""
echo "--- Recent Errors ---"
curl -s "$LOKI_URL/loki/api/v1/query_range" \
    --data-urlencode 'query={level="error"}' \
    --data-urlencode "start=$(date -d '10 minutes ago' +%s 2>/dev/null || date -v-10M +%s)000000000" \
    --data-urlencode "end=$(date +%s)000000000" \
    --data-urlencode 'limit=5' \
    2>/dev/null | jq -r '.data.result[].values[][1]' 2>/dev/null | head -5

# 3. Trace analyzer
#!/bin/bash
echo "=== Trace Analysis ==="

JAEGER_URL="${JAEGER_URL:-http://localhost:16686}"

# Services
echo "--- Services ---"
curl -s "$JAEGER_URL/api/services" 2>/dev/null | jq -r '.data[]' 2>/dev/null

# Slow traces (>1s)
echo ""
echo "--- Slowest Recent Traces ---"
for service in $(curl -s "$JAEGER_URL/api/services" 2>/dev/null | jq -r '.data[]' 2>/dev/null | head -5); do
    echo "Service: $service"
    curl -s "$JAEGER_URL/api/traces?service=$service&limit=3&minDuration=1s" 2>/dev/null | \
        jq -r '.data[] | "  TraceID: \(.traceID) Duration: \(.spans[0].duration/1000)ms"' 2>/dev/null
done`,
				},
			},
		},
	})
}
