package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1722,
			Title:       "Observability and Monitoring",
			Description: "Master Kubernetes observability: Prometheus, Grafana, metrics-server, logging stacks, and alerting.",
			Order:       22,
			Lessons: []problems.Lesson{
				{
					Title: "Prometheus and Metrics",
					Content: `Prometheus is the standard monitoring system for Kubernetes. It uses a pull-based model to scrape metrics from targets.

**Architecture:**
` + "```" + `
Prometheus ecosystem:
  Prometheus Server   → scrape, store, query metrics
  Alertmanager        → route and deduplicate alerts
  Grafana             → visualization dashboards
  Node Exporter       → host-level metrics
  kube-state-metrics  → Kubernetes object metrics
  metrics-server      → resource metrics API (HPA)
  
Metrics flow:
  App → /metrics endpoint → Prometheus scrapes → stores in TSDB
  Prometheus → PromQL queries → Grafana dashboards
  Prometheus → alert rules → Alertmanager → Slack/PagerDuty/Email

Metric types:
  Counter   → monotonically increasing (requests_total)
  Gauge     → can go up/down (temperature, queue_size)
  Histogram → bucketed distributions (request_duration_seconds)
  Summary   → quantile distributions (less common)
` + "```" + `

**Prometheus Operator / kube-prometheus-stack:**
` + "```" + `yaml
# Install via Helm
# helm install monitoring prometheus-community/kube-prometheus-stack

# ServiceMonitor — tell Prometheus to scrape your app
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: myapp
  namespace: production
  labels:
    release: monitoring  # Must match Prometheus selector
spec:
  selector:
    matchLabels:
      app: myapp
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
    scrapeTimeout: 10s
  namespaceSelector:
    matchNames:
    - production

---
# PodMonitor — for pods without services
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: batch-jobs
  namespace: production
spec:
  selector:
    matchLabels:
      app: batch-processor
  podMetricsEndpoints:
  - port: metrics
    interval: 30s
` + "```" + `

**PromQL Basics:**
` + "```" + `
# Instant vector (current values)
http_requests_total{method="GET", status="200"}

# Range vector (over time)
http_requests_total{method="GET"}[5m]

# Rate (per-second over range)
rate(http_requests_total[5m])

# Increase (total increase over range)
increase(http_requests_total[1h])

# Aggregations
sum(rate(http_requests_total[5m])) by (service)
avg(rate(http_requests_total[5m])) by (service)
topk(5, rate(http_requests_total[5m]))

# Histogram quantiles (p50, p95, p99)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

# Kubernetes-specific queries
# CPU usage per pod
sum(rate(container_cpu_usage_seconds_total[5m])) by (pod)

# Memory usage per namespace
sum(container_memory_working_set_bytes) by (namespace)

# Pod restart count
increase(kube_pod_container_status_restarts_total[1h]) > 3

# Node CPU capacity vs usage
1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance)

# Pods not ready
kube_pod_status_ready{condition="false"} == 1
` + "```" + `

**Alerting Rules:**
` + "```" + `yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: myapp-alerts
  namespace: production
  labels:
    release: monitoring
spec:
  groups:
  - name: myapp.rules
    rules:
    # High error rate
    - alert: HighErrorRate
      expr: |
        sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
        /
        sum(rate(http_requests_total[5m])) by (service)
        > 0.05
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate on {{ $labels.service }}"
        description: "Error rate is {{ $value | humanizePercentage }} (>5%)"

    # High latency
    - alert: HighLatency
      expr: |
        histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))
        > 1.0
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "High p95 latency on {{ $labels.service }}"
        description: "p95 latency is {{ $value }}s (>1s)"

    # Pod crash looping
    - alert: PodCrashLooping
      expr: increase(kube_pod_container_status_restarts_total[1h]) > 5
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Pod {{ $labels.namespace }}/{{ $labels.pod }} is crash looping"

    # High memory usage
    - alert: HighMemoryUsage
      expr: |
        container_memory_working_set_bytes / container_spec_memory_limit_bytes > 0.9
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Container {{ $labels.container }} memory >90%"

    # Node disk pressure
    - alert: NodeDiskPressure
      expr: |
        (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) < 0.1
      for: 10m
      labels:
        severity: critical
      annotations:
        summary: "Node {{ $labels.instance }} disk <10%"
` + "```" + ``,
					CodeExamples: `# Prometheus and Grafana Setup

# 1. Application metrics endpoint (Go example pattern)
# Expose metrics at /metrics endpoint:
#   http_requests_total{method, path, status}
#   http_request_duration_seconds{method, path}
#   app_goroutines gauge
#   app_connections_active gauge

---
# 2. ServiceMonitor for comprehensive scraping
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: myapp-metrics
  namespace: production
  labels:
    release: monitoring
    team: platform
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: myapp
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
    scrapeTimeout: 10s
    metricRelabelings:
    # Drop high-cardinality metrics
    - sourceLabels: [__name__]
      regex: "go_gc_.*"
      action: drop
    # Rename metric
    - sourceLabels: [__name__]
      regex: "myapp_http_requests_total"
      targetLabel: __name__
      replacement: "http_requests_total"
  namespaceSelector:
    matchNames:
    - production
    - staging

---
# 3. PrometheusRule with SLO-based alerts
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: slo-alerts
  namespace: monitoring
  labels:
    release: monitoring
spec:
  groups:
  - name: slo.rules
    # Recording rules for efficiency
    rules:
    - record: slo:http_request_availability:ratio_rate5m
      expr: |
        1 - (
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
          /
          sum(rate(http_requests_total[5m])) by (service)
        )
    - record: slo:http_request_latency:p99_5m
      expr: |
        histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))

  - name: slo.alerts
    rules:
    # SLO: 99.9% availability (error budget burn rate)
    - alert: SLOAvailabilityBudgetBurn
      expr: |
        slo:http_request_availability:ratio_rate5m < 0.999
      for: 5m
      labels:
        severity: page
        slo: availability
      annotations:
        summary: "{{ $labels.service }} availability below 99.9%"
        description: "Current availability: {{ $value | humanizePercentage }}"

    # SLO: p99 latency < 500ms
    - alert: SLOLatencyBudgetBurn
      expr: |
        slo:http_request_latency:p99_5m > 0.5
      for: 10m
      labels:
        severity: warning
        slo: latency
      annotations:
        summary: "{{ $labels.service }} p99 latency above 500ms"
        description: "Current p99: {{ $value }}s"

---
# 4. Alertmanager configuration
apiVersion: monitoring.coreos.com/v1alpha1
kind: AlertmanagerConfig
metadata:
  name: myapp-alerts
  namespace: production
  labels:
    release: monitoring
spec:
  route:
    groupBy: ['alertname', 'service']
    groupWait: 30s
    groupInterval: 5m
    repeatInterval: 4h
    receiver: default
    routes:
    - matchers:
      - name: severity
        value: critical
      receiver: pagerduty
      repeatInterval: 1h
    - matchers:
      - name: severity
        value: warning
      receiver: slack
  receivers:
  - name: default
    slackConfigs:
    - apiURL:
        name: slack-webhook
        key: url
      channel: '#alerts'
      sendResolved: true
  - name: pagerduty
    pagerdutyConfigs:
    - routingKey:
        name: pagerduty-key
        key: token
      severity: critical
  - name: slack
    slackConfigs:
    - apiURL:
        name: slack-webhook
        key: url
      channel: '#alerts-warning'
      sendResolved: true

---
# 5. Grafana Dashboard ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: myapp-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  myapp.json: |
    {
      "dashboard": {
        "title": "MyApp Overview",
        "panels": [
          {
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {"expr": "sum(rate(http_requests_total{service=\"myapp\"}[5m])) by (status)"}
            ]
          },
          {
            "title": "Latency (p50/p95/p99)",
            "type": "graph",
            "targets": [
              {"expr": "histogram_quantile(0.5, sum(rate(http_request_duration_seconds_bucket{service=\"myapp\"}[5m])) by (le))", "legendFormat": "p50"},
              {"expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{service=\"myapp\"}[5m])) by (le))", "legendFormat": "p95"},
              {"expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{service=\"myapp\"}[5m])) by (le))", "legendFormat": "p99"}
            ]
          },
          {
            "title": "Error Rate",
            "type": "singlestat",
            "targets": [
              {"expr": "sum(rate(http_requests_total{service=\"myapp\",status=~\"5..\"}[5m])) / sum(rate(http_requests_total{service=\"myapp\"}[5m]))"}
            ]
          }
        ]
      }
    }`,
				},
				{
					Title: "Logging and Tracing",
					Content: `Centralized logging and distributed tracing are essential for debugging Kubernetes workloads. Together with metrics, they form the three pillars of observability.

**Logging Architecture:**
` + "```" + `
Logging patterns in Kubernetes:
  1. Node-level agent (DaemonSet) — most common
     Fluentd/Fluent Bit → collects from /var/log/containers/*
     → ships to: Elasticsearch, Loki, CloudWatch, etc.
  
  2. Sidecar container
     App writes to file → sidecar reads → ships to backend
     Use when: app can't log to stdout, needs processing
  
  3. Direct push (less common)
     App → SDK → logging backend directly
     Use when: needs immediate delivery, structured logs

Best practices:
  ✓ Log to stdout/stderr (Kubernetes captures them)
  ✓ Use structured logging (JSON)
  ✓ Include: timestamp, level, message, request_id, trace_id
  ✓ Don't log sensitive data (passwords, tokens, PII)
  ✓ Use log levels consistently (debug/info/warn/error)
  ✓ Set resource limits on logging agents
` + "```" + `

**Loki Stack (Lightweight Logging):**
` + "```" + `yaml
# Loki is a log aggregation system by Grafana Labs.
# Like Prometheus but for logs. Indexes labels, not content.

# Install: helm install loki grafana/loki-stack

# Promtail DaemonSet scrapes container logs
# Loki stores and indexes by labels
# Grafana queries via LogQL

# LogQL examples:
{namespace="production", app="myapp"}
{namespace="production"} |= "error"
{app="myapp"} | json | level="error"
{app="myapp"} | json | duration > 1s
{app="myapp"} | json | status >= 500

# Aggregations
sum(rate({app="myapp"} |= "error" [5m])) by (pod)
count_over_time({app="myapp"} | json | level="error" [1h])

# Loki vs Elasticsearch:
#   Loki: index labels only, cheaper storage, simpler
#   Elasticsearch: full-text index, powerful search, more resources
` + "```" + `

**Distributed Tracing with OpenTelemetry:**
` + "```" + `
OpenTelemetry (OTel) is the standard for distributed tracing:

Components:
  SDK           → instrument your application
  Collector     → receive, process, export telemetry
  Backend       → Jaeger, Tempo, Zipkin, Datadog

Trace structure:
  Trace    → end-to-end request journey
  ├─ Span  → single operation (e.g., HTTP handler)
  │  ├─ Span → database query
  │  └─ Span → cache lookup
  └─ Span  → downstream service call
     └─ Span → processing

Context propagation:
  W3C Trace Context headers:
    traceparent: 00-<trace-id>-<span-id>-01
    tracestate: vendor=value
  
  Propagated automatically through HTTP headers
  Each service creates child spans linked to parent
` + "```" + `

**OpenTelemetry Collector:**
` + "```" + `yaml
# OTel Collector deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-collector
  namespace: observability
spec:
  replicas: 2
  selector:
    matchLabels:
      app: otel-collector
  template:
    metadata:
      labels:
        app: otel-collector
    spec:
      containers:
      - name: collector
        image: otel/opentelemetry-collector-contrib:0.92.0
        ports:
        - containerPort: 4317   # gRPC OTLP
        - containerPort: 4318   # HTTP OTLP
        - containerPort: 8888   # Collector metrics
        volumeMounts:
        - name: config
          mountPath: /etc/otelcol
      volumes:
      - name: config
        configMap:
          name: otel-collector-config

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
  namespace: observability
data:
  config.yaml: |
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
          http:
            endpoint: 0.0.0.0:4318
    
    processors:
      batch:
        timeout: 5s
        send_batch_size: 1000
      memory_limiter:
        check_interval: 1s
        limit_mib: 512
      resource:
        attributes:
        - key: environment
          value: production
          action: upsert
    
    exporters:
      otlp/tempo:
        endpoint: tempo.observability:4317
        tls:
          insecure: true
      prometheus:
        endpoint: 0.0.0.0:8889
      loki:
        endpoint: http://loki.observability:3100/loki/api/v1/push
    
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [memory_limiter, batch, resource]
          exporters: [otlp/tempo]
        metrics:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [prometheus]
        logs:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [loki]
` + "```" + `

**Application Instrumentation:**
` + "```" + `
Go example with OTel SDK:

  import (
      "go.opentelemetry.io/otel"
      "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
      sdktrace "go.opentelemetry.io/otel/sdk/trace"
  )
  
  // Initialize tracer
  exporter, _ := otlptracegrpc.New(ctx,
      otlptracegrpc.WithEndpoint("otel-collector:4317"),
      otlptracegrpc.WithInsecure(),
  )
  tp := sdktrace.NewTracerProvider(
      sdktrace.WithBatcher(exporter),
      sdktrace.WithResource(resource.NewWithAttributes(
          semconv.SchemaURL,
          semconv.ServiceName("myapp"),
          semconv.ServiceVersion("1.0.0"),
      )),
  )
  otel.SetTracerProvider(tp)
  
  // Create spans
  tracer := otel.Tracer("myapp")
  ctx, span := tracer.Start(ctx, "handleRequest")
  defer span.End()
  
  // Add attributes
  span.SetAttributes(
      attribute.String("user.id", userID),
      attribute.Int("items.count", len(items)),
  )
  
  // Record errors
  if err != nil {
      span.RecordError(err)
      span.SetStatus(codes.Error, err.Error())
  }
` + "```" + ``,
					CodeExamples: `# Complete Observability Stack

# 1. Fluent Bit DaemonSet for log collection
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluent-bit
  namespace: observability
  labels:
    app: fluent-bit
spec:
  selector:
    matchLabels:
      app: fluent-bit
  template:
    metadata:
      labels:
        app: fluent-bit
    spec:
      serviceAccountName: fluent-bit
      tolerations:
      - key: node-role.kubernetes.io/control-plane
        operator: Exists
        effect: NoSchedule
      containers:
      - name: fluent-bit
        image: fluent/fluent-bit:2.2
        resources:
          limits:
            cpu: 200m
            memory: 256Mi
          requests:
            cpu: 50m
            memory: 64Mi
        volumeMounts:
        - name: varlog
          mountPath: /var/log
          readOnly: true
        - name: config
          mountPath: /fluent-bit/etc/
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: config
        configMap:
          name: fluent-bit-config

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
  namespace: observability
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         5
        Log_Level     info
        Parsers_File  parsers.conf

    [INPUT]
        Name              tail
        Tag               kube.*
        Path              /var/log/containers/*.log
        Parser            cri
        Refresh_Interval  10
        Mem_Buf_Limit     5MB
        Skip_Long_Lines   On

    [FILTER]
        Name                kubernetes
        Match               kube.*
        Kube_URL            https://kubernetes.default.svc:443
        Kube_CA_File        /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        Kube_Token_File     /var/run/secrets/kubernetes.io/serviceaccount/token
        Merge_Log           On

    [FILTER]
        Name    grep
        Match   kube.*
        Exclude log health check

    [OUTPUT]
        Name            loki
        Match           kube.*
        Host            loki.observability
        Port            3100
        Labels          job=fluent-bit
        Auto_Kubernetes_Labels on

  parsers.conf: |
    [PARSER]
        Name        cri
        Format      regex
        Regex       ^(?<time>[^ ]+) (?<stream>stdout|stderr) (?<logtag>[^ ]*) (?<log>.*)$
        Time_Key    time
        Time_Format %Y-%m-%dT%H:%M:%S.%L%z

---
# 2. Grafana Tempo for distributed tracing
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tempo
  namespace: observability
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tempo
  template:
    metadata:
      labels:
        app: tempo
    spec:
      containers:
      - name: tempo
        image: grafana/tempo:2.3.1
        ports:
        - containerPort: 3200  # HTTP
        - containerPort: 4317  # OTLP gRPC
        - containerPort: 4318  # OTLP HTTP
        args:
        - -config.file=/etc/tempo/tempo.yaml
        volumeMounts:
        - name: config
          mountPath: /etc/tempo
        - name: data
          mountPath: /tmp/tempo
      volumes:
      - name: config
        configMap:
          name: tempo-config
      - name: data
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: tempo
  namespace: observability
spec:
  selector:
    app: tempo
  ports:
  - name: http
    port: 3200
  - name: otlp-grpc
    port: 4317
  - name: otlp-http
    port: 4318

---
# 3. Grafana datasource configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-datasources
  namespace: monitoring
  labels:
    grafana_datasource: "1"
data:
  datasources.yaml: |
    apiVersion: 1
    datasources:
    - name: Prometheus
      type: prometheus
      url: http://prometheus-operated:9090
      isDefault: true
      access: proxy
    - name: Loki
      type: loki
      url: http://loki.observability:3100
      access: proxy
      jsonData:
        derivedFields:
        - name: TraceID
          matcherRegex: "trace_id=(\\w+)"
          url: "$${__value.raw}"
          datasourceUid: tempo
    - name: Tempo
      type: tempo
      uid: tempo
      url: http://tempo.observability:3200
      access: proxy
      jsonData:
        tracesToLogs:
          datasourceUid: loki
          filterByTraceID: true
        tracesToMetrics:
          datasourceUid: prometheus
          queries:
          - name: "Request rate"
            query: "sum(rate(http_requests_total{$$__tags}[5m]))"`,
				},
			},
		},
	})
}
