package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1727,
			Title:       "Resource Management and Autoscaling",
			Description: "Master Kubernetes resource management: requests/limits, QoS classes, LimitRanges, ResourceQuotas, HPA, VPA, KEDA, and cluster autoscaling.",
			Order:       27,
			Lessons: []problems.Lesson{
				{
					Title: "Resource Requests, Limits, and QoS",
					Content: `Resource management is critical for cluster stability and efficiency. Poorly configured resources lead to OOMKills, CPU throttling, or wasted capacity.

**Requests vs Limits:**
` + "```" + `
Requests:
  - Guaranteed minimum resources for a container
  - Used by scheduler for pod placement
  - Determines which node has enough capacity
  - Affects QoS class

Limits:
  - Maximum resources a container can use
  - CPU: throttled when exceeded (not killed)
  - Memory: OOMKilled when exceeded
  - Sets cgroup constraints

Best practices:
  ✓ Always set requests (for proper scheduling)
  ✓ Set memory limits = memory requests (avoid OOMKill surprises)
  ✓ CPU limits are debatable:
      With limits:    predictable but may throttle unnecessarily
      Without limits: better utilization but noisy neighbor risk
  ✓ Use metrics (VPA recommendations) to right-size

resources:
  requests:
    cpu: 100m      # 0.1 CPU core (millicores)
    memory: 256Mi  # 256 MiB
  limits:
    cpu: "1"       # 1 CPU core
    memory: 512Mi  # 512 MiB (OOMKill if exceeded)
` + "```" + `

**QoS Classes:**
` + "```" + `
Kubernetes assigns Quality of Service classes based on resource config:

Guaranteed (highest priority):
  - Every container has requests = limits for both CPU and memory
  - Last to be evicted
  containers:
  - resources:
      requests: { cpu: "1", memory: 1Gi }
      limits:   { cpu: "1", memory: 1Gi }

Burstable:
  - At least one container has request or limit set
  - Middle eviction priority
  containers:
  - resources:
      requests: { cpu: 100m, memory: 256Mi }
      limits:   { cpu: "2", memory: 1Gi }

BestEffort (lowest priority):
  - No requests or limits set on any container
  - First to be evicted under pressure
  containers:
  - resources: {}  # Nothing set

Eviction order under memory pressure:
  1. BestEffort pods exceeding nothing
  2. Burstable pods exceeding requests
  3. Guaranteed pods (only if system is critically low)
` + "```" + `

**CPU Throttling Deep Dive:**
` + "```" + `
CPU is a compressible resource:
  - Container is NOT killed when exceeding CPU limit
  - Instead, it's throttled (slowed down)
  - CFS (Completely Fair Scheduler) enforces limits
  - 100ms periods → if limit is 200m, container gets 20ms per 100ms

Throttling symptoms:
  - High latency without high CPU utilization
  - container_cpu_cfs_throttled_periods_total increasing
  - Application response times spiking periodically

Detection:
  # PromQL: throttling ratio
  rate(container_cpu_cfs_throttled_periods_total[5m])
  /
  rate(container_cpu_cfs_periods_total[5m])
  > 0.25  # More than 25% throttled

Solutions:
  1. Increase CPU limit
  2. Remove CPU limit (use requests only)
  3. Increase CPU requests for scheduling
  4. Profile application for CPU optimization
` + "```" + `

**Memory Management:**
` + "```" + `
Memory is incompressible:
  - Once allocated, can't be reclaimed without killing
  - OOMKill when container exceeds limit
  - Container restart policy determines what happens

Memory metrics:
  container_memory_working_set_bytes  → what K8s uses for limits
  container_memory_rss               → resident set size
  container_memory_cache              → page cache (reclaimable)
  
  OOMKill happens when:
    working_set_bytes >= memory limit

Diagnosing OOMKills:
  kubectl describe pod <name>
  # Look for: Last State: Terminated, Reason: OOMKilled
  
  kubectl get events --field-selector reason=OOMKilling
  
  dmesg | grep -i oom  # On the node

Right-sizing:
  1. Run workload with generous limits
  2. Observe actual usage via Prometheus
  3. Set request = p99 usage + 20% buffer
  4. Set limit = request (for Guaranteed QoS) or limit = 2x request
` + "```" + `

**LimitRange and ResourceQuota:**
` + "```" + `yaml
# LimitRange — per-container defaults and constraints
apiVersion: v1
kind: LimitRange
metadata:
  name: container-limits
  namespace: production
spec:
  limits:
  - type: Container
    default:            # Default limits if not specified
      cpu: 500m
      memory: 512Mi
    defaultRequest:     # Default requests if not specified
      cpu: 100m
      memory: 128Mi
    max:                # Maximum allowed
      cpu: "4"
      memory: 8Gi
    min:                # Minimum allowed
      cpu: 50m
      memory: 64Mi
    maxLimitRequestRatio:
      cpu: "10"         # Limit can't be more than 10x request
  - type: PersistentVolumeClaim
    max:
      storage: 500Gi
    min:
      storage: 1Gi

---
# ResourceQuota — namespace-level totals
apiVersion: v1
kind: ResourceQuota
metadata:
  name: compute-quota
  namespace: production
spec:
  hard:
    requests.cpu: "100"
    requests.memory: 200Gi
    limits.cpu: "200"
    limits.memory: 400Gi
    pods: "200"
    persistentvolumeclaims: "100"
    services: "50"
    services.loadbalancers: "5"
    services.nodeports: "10"
    secrets: "200"
    configmaps: "200"
    replicationcontrollers: "50"
    count/deployments.apps: "50"
    count/statefulsets.apps: "20"
    count/jobs.batch: "100"
  scopeSelector:
    matchExpressions:
    - scopeName: PriorityClass
      operator: In
      values: ["high"]
` + "```" + ``,
					CodeExamples: `# Resource Management Examples

# 1. Well-configured production deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: production
spec:
  replicas: 5
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
    spec:
      containers:
      - name: api
        image: myregistry/api:v2.0.0
        resources:
          # Guaranteed QoS: requests = limits
          requests:
            cpu: "1"
            memory: 1Gi
          limits:
            cpu: "1"
            memory: 1Gi
        ports:
        - containerPort: 8080
      # Sidecar with Burstable QoS
      - name: envoy-proxy
        image: envoyproxy/envoy:v1.28
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 256Mi
        ports:
        - containerPort: 9901

---
# 2. Batch job with appropriate resources
apiVersion: batch/v1
kind: Job
metadata:
  name: data-import
  namespace: production
spec:
  parallelism: 4
  completions: 100
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: importer
        image: myregistry/importer:v1
        resources:
          requests:
            cpu: "2"
            memory: 4Gi
            ephemeral-storage: 10Gi
          limits:
            memory: 8Gi
            ephemeral-storage: 20Gi
        # No CPU limit — allow bursting when available

---
# 3. Multi-tier ResourceQuotas
apiVersion: v1
kind: ResourceQuota
metadata:
  name: critical-quota
  namespace: production
spec:
  hard:
    requests.cpu: "50"
    requests.memory: 100Gi
    limits.cpu: "100"
    limits.memory: 200Gi
  scopeSelector:
    matchExpressions:
    - scopeName: PriorityClass
      operator: In
      values: ["critical-production"]
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: standard-quota
  namespace: production
spec:
  hard:
    requests.cpu: "100"
    requests.memory: 200Gi
    pods: "200"
  scopeSelector:
    matchExpressions:
    - scopeName: PriorityClass
      operator: In
      values: ["standard-production"]

---
# 4. Node allocatable and reserved resources
# kubelet configuration (on each node)
# apiVersion: kubelet.config.k8s.io/v1beta1
# kind: KubeletConfiguration
# systemReserved:
#   cpu: 500m
#   memory: 1Gi
#   ephemeral-storage: 5Gi
# kubeReserved:
#   cpu: 500m
#   memory: 1Gi
#   ephemeral-storage: 5Gi
# evictionHard:
#   memory.available: 100Mi
#   nodefs.available: 10%
#   imagefs.available: 15%

# Example: Node with 16 CPU, 64Gi memory
# Allocatable = Capacity - system-reserved - kube-reserved - eviction-threshold
# CPU: 16 - 0.5 - 0.5 = 15 allocatable
# Memory: 64Gi - 1Gi - 1Gi - 100Mi ≈ 62Gi allocatable

---
# 5. Topology-aware resource allocation
apiVersion: v1
kind: Pod
metadata:
  name: gpu-workload
  namespace: ml
spec:
  containers:
  - name: training
    image: myregistry/ml-trainer:v1
    resources:
      requests:
        cpu: "8"
        memory: 32Gi
        nvidia.com/gpu: 2
      limits:
        cpu: "16"
        memory: 64Gi
        nvidia.com/gpu: 2
  nodeSelector:
    accelerator: nvidia-a100
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule`,
				},
				{
					Title: "Horizontal and Vertical Pod Autoscaling",
					Content: `Autoscaling automatically adjusts resources based on demand. Kubernetes offers HPA (replicas), VPA (container resources), and cluster autoscaling (nodes).

**Horizontal Pod Autoscaler (HPA):**
` + "```" + `
HPA adjusts replica count based on metrics:

API versions:
  autoscaling/v2 → current (resource + custom + external metrics)
  autoscaling/v1 → legacy (CPU only)

Metrics types:
  Resource:  CPU, memory utilization (from metrics-server)
  Pods:      Custom per-pod metrics (from Prometheus adapter)
  Object:    Metrics from any K8s object
  External:  Metrics from external system (cloud metrics)

Algorithm:
  desiredReplicas = ceil(currentReplicas * (currentMetric / targetMetric))
  
  Example:
    Current: 5 replicas, CPU at 80%
    Target: 50%
    Desired: ceil(5 * 80/50) = ceil(8) = 8 replicas

Scaling behavior:
  - Scale up: fast (respond to load quickly)
  - Scale down: slow (avoid flapping)
  - Stabilization window: prevent thrashing
` + "```" + `

**HPA v2 Configuration:**
` + "```" + `yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: myapp
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  minReplicas: 3
  maxReplicas: 50
  metrics:
  # CPU utilization
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # Memory utilization
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  # Custom metric: requests per second
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  # External metric: SQS queue depth
  - type: External
    external:
      metric:
        name: sqs_queue_length
        selector:
          matchLabels:
            queue: orders
      target:
        type: AverageValue
        averageValue: "5"
  # Scaling behavior
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0    # Immediate scale up
      policies:
      - type: Percent
        value: 100                     # Double replicas at most
        periodSeconds: 60
      - type: Pods
        value: 10                      # Add max 10 pods at once
        periodSeconds: 60
      selectPolicy: Max                # Use whichever allows more scaling
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scale down
      policies:
      - type: Percent
        value: 10                      # Remove max 10% at a time
        periodSeconds: 60
      selectPolicy: Min                # Use whichever is more conservative
` + "```" + `

**Vertical Pod Autoscaler (VPA):**
` + "```" + `yaml
# VPA adjusts container resource requests/limits

apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: myapp-vpa
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  updatePolicy:
    updateMode: "Auto"       # Off, Initial, Recreate, Auto
    # Off:      recommendations only (no changes)
    # Initial:  set on pod creation only
    # Recreate: evict and recreate pods to apply
    # Auto:     gradually update (prefer in-place if supported)
  resourcePolicy:
    containerPolicies:
    - containerName: app
      minAllowed:
        cpu: 100m
        memory: 128Mi
      maxAllowed:
        cpu: "4"
        memory: 8Gi
      controlledResources: ["cpu", "memory"]
      controlledValues: RequestsAndLimits
    - containerName: sidecar
      mode: "Off"  # Don't touch sidecar resources

# Check VPA recommendations:
# kubectl describe vpa myapp-vpa
# Recommendations:
#   Target:     cpu: 250m, memory: 512Mi
#   Lower:      cpu: 100m, memory: 256Mi  (10th percentile)
#   Upper:      cpu: 1000m, memory: 2Gi   (95th percentile)
#   Uncapped:   cpu: 800m, memory: 1.5Gi  (no min/max applied)

# VPA + HPA:
#   Don't use VPA (Auto) + HPA on same CPU metric!
#   VPA changes requests → HPA sees utilization change → conflict
#   
#   Safe combinations:
#   ✓ VPA adjusts memory, HPA scales on CPU
#   ✓ VPA in "Off" mode + HPA (use VPA for recommendations)
#   ✓ VPA + HPA on custom metrics (not CPU/memory)
` + "```" + `

**KEDA (Kubernetes Event-Driven Autoscaling):**
` + "```" + `yaml
# KEDA extends HPA with 60+ event sources

# ScaledObject — scales Deployment based on external trigger
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: order-processor
  namespace: production
spec:
  scaleTargetRef:
    name: order-processor
  minReplicaCount: 1
  maxReplicaCount: 50
  cooldownPeriod: 300
  pollingInterval: 15
  triggers:
  # Scale based on RabbitMQ queue depth
  - type: rabbitmq
    metadata:
      queueName: orders
      host: amqp://rabbitmq.production:5672
      queueLength: "5"
  # Scale based on Prometheus metric
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      query: sum(rate(http_requests_total{service="order-api"}[2m]))
      threshold: "100"

---
# ScaledJob — for one-time batch processing
apiVersion: keda.sh/v1alpha1
kind: ScaledJob
metadata:
  name: email-sender
  namespace: production
spec:
  jobTargetRef:
    parallelism: 1
    completions: 1
    template:
      spec:
        restartPolicy: Never
        containers:
        - name: sender
          image: myregistry/email-sender:v1
  minReplicaCount: 0
  maxReplicaCount: 20
  pollingInterval: 30
  successfulJobsHistoryLimit: 5
  failedJobsHistoryLimit: 5
  triggers:
  - type: aws-sqs-queue
    metadata:
      queueURL: https://sqs.us-east-1.amazonaws.com/123456789/emails
      queueLength: "1"
      awsRegion: us-east-1
` + "```" + `

**Cluster Autoscaler:**
` + "```" + `
Cluster Autoscaler adds/removes nodes based on pod scheduling needs:

Scale up:
  1. Pod can't be scheduled (Pending due to insufficient resources)
  2. CA evaluates which node group can fit the pod
  3. CA increases node group size
  4. Cloud provider creates new node
  5. Pod gets scheduled

Scale down:
  1. Node utilization < threshold (default 50%)
  2. All pods can be rescheduled on other nodes
  3. No PodDisruptionBudget violations
  4. Wait for --scale-down-unneeded-time (default 10m)
  5. Node drained and removed

Configuration flags:
  --scale-down-utilization-threshold=0.5
  --scale-down-unneeded-time=10m
  --scale-down-delay-after-add=10m
  --max-node-provision-time=15m
  --balance-similar-node-groups=true
  --expander=least-waste  # or priority, random, most-pods

Karpenter (AWS):
  - Alternative to Cluster Autoscaler
  - Faster scaling (provisions in seconds)
  - Bin-packing, spot instance support
  - Consolidation (replaces underutilized nodes)
` + "```" + ``,
					CodeExamples: `# Autoscaling Configuration Examples

# 1. Comprehensive HPA with behavior tuning
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-frontend
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-frontend
  minReplicas: 5
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 65
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 75
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 50
        periodSeconds: 30
      - type: Pods
        value: 20
        periodSeconds: 30
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 600
      policies:
      - type: Percent
        value: 5
        periodSeconds: 60
      selectPolicy: Min

---
# 2. HPA with Prometheus Adapter custom metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway
  minReplicas: 3
  maxReplicas: 30
  metrics:
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "500"
  - type: Pods
    pods:
      metric:
        name: http_request_duration_seconds_p99
      target:
        type: AverageValue
        averageValue: "200m"
  - type: Object
    object:
      metric:
        name: queue_messages_ready
      describedObject:
        apiVersion: v1
        kind: Service
        name: rabbitmq
      target:
        type: Value
        value: "100"

---
# 3. Prometheus Adapter configuration for custom metrics
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-adapter-config
  namespace: monitoring
data:
  config.yaml: |
    rules:
    - seriesQuery: 'http_requests_total{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)_total$"
        as: "${1}_per_second"
      metricsQuery: 'sum(rate(<<.Series>>{<<.LabelMatchers>>}[2m])) by (<<.GroupBy>>)'
    - seriesQuery: 'http_request_duration_seconds_bucket{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        as: "http_request_duration_seconds_p99"
      metricsQuery: 'histogram_quantile(0.99, sum(rate(<<.Series>>{<<.LabelMatchers>>}[2m])) by (le, <<.GroupBy>>))'

---
# 4. Karpenter NodePool (AWS)
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: default
spec:
  template:
    spec:
      nodeClassRef:
        name: default
      requirements:
      - key: kubernetes.io/arch
        operator: In
        values: ["amd64"]
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["on-demand", "spot"]
      - key: karpenter.k8s.aws/instance-category
        operator: In
        values: ["c", "m", "r"]
      - key: karpenter.k8s.aws/instance-generation
        operator: Gt
        values: ["5"]
  limits:
    cpu: "1000"
    memory: 2000Gi
  disruption:
    consolidationPolicy: WhenUnderutilized
    consolidateAfter: 30s
    expireAfter: 720h  # 30 days

---
apiVersion: karpenter.k8s.aws/v1beta1
kind: EC2NodeClass
metadata:
  name: default
spec:
  amiFamily: AL2
  role: KarpenterNodeRole
  subnetSelectorTerms:
  - tags:
      karpenter.sh/discovery: production
  securityGroupSelectorTerms:
  - tags:
      karpenter.sh/discovery: production
  blockDeviceMappings:
  - deviceName: /dev/xvda
    ebs:
      volumeSize: 100Gi
      volumeType: gp3
      encrypted: true

---
# 5. VPA recommendation only mode
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: api-vpa-recommender
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  updatePolicy:
    updateMode: "Off"
  resourcePolicy:
    containerPolicies:
    - containerName: api
      controlledResources: ["cpu", "memory"]`,
				},
			},
		},
	})
}
