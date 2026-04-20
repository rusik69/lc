package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          1456,
			Title:       "Container Orchestration and Service Mesh",
			Description: "Advanced container orchestration patterns including Kubernetes operators, service mesh with Istio and Linkerd, and container security practices.",
			Order:       56,
			Lessons: []problems.Lesson{
				{
					Title: "Kubernetes Advanced Operations",
					Content: `Advanced Kubernetes patterns for production workloads including operators, autoscaling, and resource management.

**Kubernetes Operators:**
` + "```" + `
Operator pattern:
  Custom Resource + Custom Controller = Operator
  
  Extends Kubernetes API to manage complex applications.
  
  Operator Maturity Model:
    Level 1: Basic install (Helm chart equivalent)
    Level 2: Seamless upgrades (handle version migrations)
    Level 3: Full lifecycle (backup, restore, scaling)
    Level 4: Deep insights (metrics, alerts, dashboards)
    Level 5: Auto-pilot (auto-tune, auto-scale, self-heal)

Operator SDK:
  # Initialize Go operator
  operator-sdk init \
    --domain example.com \
    --repo github.com/example/memcached-operator
  
  # Create API (CRD + controller)
  operator-sdk create api \
    --group cache --version v1alpha1 --kind Memcached \
    --resource --controller
  
  # Build and deploy
  make docker-build docker-push IMG=myregistry/operator:v0.1.0
  make deploy IMG=myregistry/operator:v0.1.0

CRD example:
  apiVersion: apiextensions.k8s.io/v1
  kind: CustomResourceDefinition
  metadata:
    name: memcacheds.cache.example.com
  spec:
    group: cache.example.com
    names:
      kind: Memcached
      listKind: MemcachedList
      plural: memcacheds
      singular: memcached
    scope: Namespaced
    versions:
      - name: v1alpha1
        served: true
        storage: true
        schema:
          openAPIV3Schema:
            type: object
            properties:
              spec:
                type: object
                properties:
                  size:
                    type: integer
                    minimum: 1
                    maximum: 10
                  version:
                    type: string

Custom Resource:
  apiVersion: cache.example.com/v1alpha1
  kind: Memcached
  metadata:
    name: memcached-sample
  spec:
    size: 3
    version: "1.6.17"

Popular operators:
  Databases:
    - PostgreSQL: CloudNativePG, Zalando, CrunchyData
    - MySQL: Oracle MySQL Operator, Vitess
    - MongoDB: MongoDB Community Operator
    - Redis: Redis Enterprise, Spotahome
  
  Messaging:
    - Kafka: Strimzi
    - RabbitMQ: RabbitMQ Cluster Operator
    - NATS: NATS Operator
  
  Monitoring:
    - Prometheus: Prometheus Operator (kube-prometheus-stack)
    - Grafana: Grafana Operator
    - ElasticSearch: Elastic Cloud on Kubernetes (ECK)
  
  Cert Management:
    - cert-manager: Let's Encrypt certificates
  
  GitOps:
    - ArgoCD: Application delivery
    - Flux: GitOps toolkit
` + "```" + `

**Autoscaling:**
` + "```" + `
Horizontal Pod Autoscaler (HPA):
  apiVersion: autoscaling/v2
  kind: HorizontalPodAutoscaler
  metadata:
    name: web-hpa
  spec:
    scaleTargetRef:
      apiVersion: apps/v1
      kind: Deployment
      name: web
    minReplicas: 2
    maxReplicas: 20
    metrics:
      - type: Resource
        resource:
          name: cpu
          target:
            type: Utilization
            averageUtilization: 70
      - type: Resource
        resource:
          name: memory
          target:
            type: Utilization
            averageUtilization: 80
      - type: Pods
        pods:
          metric:
            name: http_requests_per_second
          target:
            type: AverageValue
            averageValue: 1000
    behavior:
      scaleUp:
        stabilizationWindowSeconds: 60
        policies:
          - type: Percent
            value: 100
            periodSeconds: 60
      scaleDown:
        stabilizationWindowSeconds: 300
        policies:
          - type: Percent
            value: 10
            periodSeconds: 60

Vertical Pod Autoscaler (VPA):
  apiVersion: autoscaling.k8s.io/v1
  kind: VerticalPodAutoscaler
  metadata:
    name: web-vpa
  spec:
    targetRef:
      apiVersion: apps/v1
      kind: Deployment
      name: web
    updatePolicy:
      updateMode: "Auto"  # Off, Initial, Recreate, Auto
    resourcePolicy:
      containerPolicies:
        - containerName: web
          minAllowed:
            cpu: 100m
            memory: 128Mi
          maxAllowed:
            cpu: 4
            memory: 8Gi

KEDA (Event-driven autoscaling):
  apiVersion: keda.sh/v1alpha1
  kind: ScaledObject
  metadata:
    name: worker-scaler
  spec:
    scaleTargetRef:
      name: worker
    pollingInterval: 30
    cooldownPeriod: 300
    minReplicaCount: 0
    maxReplicaCount: 50
    triggers:
      - type: rabbitmq
        metadata:
          host: amqp://rabbitmq:5672
          queueName: tasks
          queueLength: "10"
      - type: prometheus
        metadata:
          serverAddress: http://prometheus:9090
          metricName: http_requests_total
          query: sum(rate(http_requests_total[1m]))
          threshold: "100"

Cluster Autoscaler:
  # Auto-scale node pool based on pending pods
  Supported providers:
    AWS (ASG), GCP (MIG), Azure (VMSS)
  
  Key settings:
    --scale-down-delay-after-add=10m
    --scale-down-unneeded-time=10m
    --max-node-provision-time=15m
    --max-graceful-termination-sec=600
    --balance-similar-node-groups=true
    --expander=least-waste

Karpenter (AWS):
  # Next-gen node autoscaler
  apiVersion: karpenter.sh/v1beta1
  kind: NodePool
  metadata:
    name: default
  spec:
    template:
      spec:
        requirements:
          - key: kubernetes.io/arch
            operator: In
            values: ["amd64", "arm64"]
          - key: karpenter.k8s.aws/instance-category
            operator: In
            values: ["c", "m", "r"]
          - key: karpenter.k8s.aws/instance-generation
            operator: Gt
            values: ["5"]
        nodeClassRef:
          name: default
    limits:
      cpu: "1000"
      memory: 4000Gi
    disruption:
      consolidationPolicy: WhenUnderutilized
      expireAfter: 720h
` + "```" + ``,
					CodeExamples: `# Kubernetes advanced operations scripts

# 1. Cluster capacity analyzer
#!/bin/bash
echo "=== Cluster Capacity Analysis ==="

# Node resources
echo "--- Node Resources ---"
kubectl top nodes 2>/dev/null

echo ""
echo "--- Resource Requests vs Limits ---"
kubectl get pods --all-namespaces -o json 2>/dev/null | jq -r '
  .items[] | 
  .metadata.namespace + "/" + .metadata.name + " " +
  (.spec.containers[0].resources.requests.cpu // "none") + "/" +
  (.spec.containers[0].resources.limits.cpu // "none") + " CPU, " +
  (.spec.containers[0].resources.requests.memory // "none") + "/" +
  (.spec.containers[0].resources.limits.memory // "none") + " MEM"
' 2>/dev/null | head -20

# Pods without resource requests
echo ""
echo "--- Pods Without Resource Requests ---"
kubectl get pods --all-namespaces -o json 2>/dev/null | jq -r '
  .items[] | select(.spec.containers[0].resources.requests == null) |
  .metadata.namespace + "/" + .metadata.name
' 2>/dev/null | head -10

# HPA status
echo ""
echo "--- HPA Status ---"
kubectl get hpa --all-namespaces 2>/dev/null

# VPA recommendations
echo ""
echo "--- VPA Recommendations ---"
kubectl get vpa --all-namespaces -o json 2>/dev/null | jq -r '
  .items[] |
  .metadata.name + ": " +
  (.status.recommendation.containerRecommendations[0].target.cpu // "N/A") + " CPU, " +
  (.status.recommendation.containerRecommendations[0].target.memory // "N/A") + " MEM"
' 2>/dev/null

# 2. Operator inventory
#!/bin/bash
echo "=== Operator Inventory ==="

# CRDs
echo "--- Custom Resource Definitions ---"
kubectl get crds -o custom-columns='NAME:.metadata.name,GROUP:.spec.group,CREATED:.metadata.creationTimestamp' 2>/dev/null | head -20

# Operator pods
echo ""
echo "--- Operator Pods ---"
kubectl get pods --all-namespaces -l 'app.kubernetes.io/managed-by in (olm,operator-sdk)' 2>/dev/null
kubectl get pods --all-namespaces 2>/dev/null | grep -i "operator\|controller" | head -15

# Custom resources
echo ""
echo "--- Custom Resources ---"
for crd in $(kubectl get crds -o name 2>/dev/null | head -10); do
    KIND=$(kubectl get "$crd" -o jsonpath='{.spec.names.kind}')
    COUNT=$(kubectl get "$KIND" --all-namespaces --no-headers 2>/dev/null | wc -l | tr -d ' ')
    if [ "$COUNT" -gt 0 ]; then
        echo "  $KIND: $COUNT instances"
    fi
done

# 3. Pod disruption budget check
#!/bin/bash
echo "=== PDB Analysis ==="

echo "--- Pod Disruption Budgets ---"
kubectl get pdb --all-namespaces 2>/dev/null

echo ""
echo "--- Deployments Without PDB ---"
for deploy in $(kubectl get deployments --all-namespaces -o json 2>/dev/null | \
    jq -r '.items[] | .metadata.namespace + "/" + .metadata.name'); do
    
    NS=$(echo "$deploy" | cut -d'/' -f1)
    NAME=$(echo "$deploy" | cut -d'/' -f2)
    
    LABELS=$(kubectl get deployment "$NAME" -n "$NS" -o jsonpath='{.spec.selector.matchLabels}' 2>/dev/null)
    
    HAS_PDB=$(kubectl get pdb -n "$NS" -o json 2>/dev/null | jq -r --arg name "$NAME" '
        .items[] | select(.spec.selector.matchLabels | to_entries[] | .value == $name) | .metadata.name
    ' 2>/dev/null)
    
    if [ -z "$HAS_PDB" ]; then
        REPLICAS=$(kubectl get deployment "$NAME" -n "$NS" -o jsonpath='{.spec.replicas}' 2>/dev/null)
        if [ "$REPLICAS" -gt 1 ] 2>/dev/null; then
            echo "  $NS/$NAME (replicas: $REPLICAS)"
        fi
    fi
done`,
				},
				{
					Title: "Service Mesh Architecture",
					Content: `Service meshes provide infrastructure-level networking features for microservices including traffic management, security, and observability.

**Service Mesh Concepts:**
` + "```" + `
What a service mesh provides:
  Traffic management:
    - Load balancing (round-robin, least-conn, random)
    - Traffic splitting (canary, A/B)
    - Circuit breaking
    - Retries and timeouts
    - Rate limiting
    - Fault injection
  
  Security:
    - Mutual TLS (mTLS) between services
    - Certificate rotation
    - Authorization policies
    - Identity-based access control
  
  Observability:
    - Distributed tracing
    - Metrics (RED: Rate, Error, Duration)
    - Access logging
    - Service topology map

Architecture:
  Data plane:
    - Sidecar proxies (Envoy)
    - Injected alongside each pod
    - Intercept all network traffic
    - Apply policies transparently
  
  Control plane:
    - Configuration management
    - Certificate authority
    - Service discovery
    - Policy distribution

Service mesh options:
  Istio:
    - Most feature-rich
    - Envoy-based data plane
    - Complex but powerful
    - Large community
  
  Linkerd:
    - Lightweight, simple
    - Rust-based data plane (linkerd2-proxy)
    - Lower resource overhead
    - Easier to operate
  
  Cilium Service Mesh:
    - eBPF-based (kernel-level)
    - No sidecar needed
    - Lower overhead
    - Network policies included
  
  Consul Connect:
    - HashiCorp ecosystem
    - Multi-platform (K8s, VMs, ECS)
    - Built-in service discovery
` + "```" + `

**Istio Configuration:**
` + "```" + `
Installation:
  istioctl install --set profile=demo
  kubectl label namespace default istio-injection=enabled
  
  Profiles:
    default:    Production, moderate resources
    demo:       All features, testing
    minimal:    Control plane only
    ambient:    Sidecar-less mode (ztunnel)
    remote:     Multi-cluster secondary

Traffic management:
  # VirtualService (routing rules)
  apiVersion: networking.istio.io/v1beta1
  kind: VirtualService
  metadata:
    name: reviews
  spec:
    hosts:
      - reviews
    http:
      - match:
          - headers:
              end-user:
                exact: jason
        route:
          - destination:
              host: reviews
              subset: v2
      - route:
          - destination:
              host: reviews
              subset: v1
            weight: 90
          - destination:
              host: reviews
              subset: v2
            weight: 10
  
  # DestinationRule (load balancing, circuit breaking)
  apiVersion: networking.istio.io/v1beta1
  kind: DestinationRule
  metadata:
    name: reviews
  spec:
    host: reviews
    trafficPolicy:
      connectionPool:
        tcp:
          maxConnections: 100
        http:
          h2UpgradePolicy: DEFAULT
          http1MaxPendingRequests: 100
          http2MaxRequests: 1000
      outlierDetection:
        consecutive5xxErrors: 5
        interval: 30s
        baseEjectionTime: 30s
        maxEjectionPercent: 50
      loadBalancer:
        simple: LEAST_REQUEST
    subsets:
      - name: v1
        labels:
          version: v1
      - name: v2
        labels:
          version: v2

Security:
  # PeerAuthentication (mTLS)
  apiVersion: security.istio.io/v1beta1
  kind: PeerAuthentication
  metadata:
    name: default
    namespace: istio-system
  spec:
    mtls:
      mode: STRICT  # STRICT, PERMISSIVE, DISABLE
  
  # AuthorizationPolicy
  apiVersion: security.istio.io/v1beta1
  kind: AuthorizationPolicy
  metadata:
    name: allow-frontend
    namespace: default
  spec:
    selector:
      matchLabels:
        app: backend
    rules:
      - from:
          - source:
              principals: ["cluster.local/ns/default/sa/frontend"]
        to:
          - operation:
              methods: ["GET", "POST"]
              paths: ["/api/*"]

Fault injection:
  apiVersion: networking.istio.io/v1beta1
  kind: VirtualService
  metadata:
    name: ratings
  spec:
    hosts:
      - ratings
    http:
      - fault:
          delay:
            percentage:
              value: 10
            fixedDelay: 5s
          abort:
            percentage:
              value: 5
            httpStatus: 500
        route:
          - destination:
              host: ratings

Observability:
  # Kiali: Service mesh dashboard
  kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/kiali.yaml
  
  # Jaeger: Distributed tracing
  kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/jaeger.yaml
  
  # Prometheus: Metrics
  kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/prometheus.yaml
  
  # Grafana: Dashboards
  kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/grafana.yaml
` + "```" + ``,
					CodeExamples: `# Service mesh management scripts

# 1. Istio health check
#!/bin/bash
echo "=== Istio Service Mesh Health ==="

# Control plane
echo "--- Control Plane ---"
istioctl version 2>/dev/null
echo ""
kubectl get pods -n istio-system 2>/dev/null

# Proxy status
echo ""
echo "--- Proxy Status ---"
istioctl proxy-status 2>/dev/null | head -20

# Configuration validation
echo ""
echo "--- Configuration Validation ---"
istioctl analyze --all-namespaces 2>/dev/null | head -15

# mTLS status
echo ""
echo "--- mTLS Status ---"
for ns in $(kubectl get ns -o name 2>/dev/null | cut -d'/' -f2); do
    MTLS=$(kubectl get peerauthentication -n "$ns" -o jsonpath='{.items[0].spec.mtls.mode}' 2>/dev/null)
    if [ -n "$MTLS" ]; then
        echo "  $ns: $MTLS"
    fi
done

# 2. Service mesh traffic analysis
#!/bin/bash
echo "=== Traffic Analysis ==="

# Virtual services
echo "--- Virtual Services ---"
kubectl get virtualservices --all-namespaces 2>/dev/null

# Destination rules
echo ""
echo "--- Destination Rules ---"
kubectl get destinationrules --all-namespaces 2>/dev/null

# Authorization policies
echo ""
echo "--- Authorization Policies ---"
kubectl get authorizationpolicies --all-namespaces 2>/dev/null

# Service entries (external services)
echo ""
echo "--- Service Entries ---"
kubectl get serviceentries --all-namespaces 2>/dev/null

# Gateways
echo ""
echo "--- Gateways ---"
kubectl get gateways --all-namespaces 2>/dev/null

# 3. Sidecar resource usage
#!/bin/bash
echo "=== Sidecar Resource Usage ==="

echo "--- Envoy Proxy CPU/Memory ---"
kubectl get pods --all-namespaces -o json 2>/dev/null | jq -r '
  .items[] | select(.spec.containers[].name == "istio-proxy") |
  .metadata.namespace + "/" + .metadata.name
' 2>/dev/null | while read -r pod; do
    NS=$(echo "$pod" | cut -d'/' -f1)
    NAME=$(echo "$pod" | cut -d'/' -f2)
    
    CPU=$(kubectl top pod "$NAME" -n "$NS" --containers 2>/dev/null | grep istio-proxy | awk '{print $3}')
    MEM=$(kubectl top pod "$NAME" -n "$NS" --containers 2>/dev/null | grep istio-proxy | awk '{print $4}')
    
    if [ -n "$CPU" ]; then
        echo "  $pod: CPU=$CPU MEM=$MEM"
    fi
done | head -15`,
				},
			},
		},
	})
}
