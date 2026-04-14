package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1716,
			Title:       "Scheduling & Affinity",
			Description: "Master advanced pod scheduling: taints, tolerations, node affinity, pod affinity/anti-affinity, topology spread constraints, and priority classes.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "Taints and Tolerations",
					Content: `Taints and tolerations work together to ensure that pods are not scheduled onto inappropriate nodes. Taints are applied to nodes; tolerations are applied to pods.

**1. How Taints Work:**
*   A taint on a node "repels" pods that do not tolerate it.
*   Syntax: ` + "`" + `kubectl taint nodes node1 key=value:effect` + "`" + `.
*   Effects:
    *   ` + "`" + `NoSchedule` + "`" + ` -- New pods without a matching toleration will NOT be scheduled on this node. Existing pods stay.
    *   ` + "`" + `PreferNoSchedule` + "`" + ` -- Soft version. The scheduler tries to avoid placing pods here, but will if necessary.
    *   ` + "`" + `NoExecute` + "`" + ` -- New pods are NOT scheduled AND existing pods without the toleration are evicted.

**2. Tolerations:**
*   Pods must explicitly declare tolerations to be scheduled on tainted nodes.
*   ` + "`" + `operator: Equal` + "`" + ` matches a specific key-value pair.
*   ` + "`" + `operator: Exists` + "`" + ` matches any value for the key (or all taints if key is empty).
*   ` + "`" + `tolerationSeconds` + "`" + ` with ` + "`" + `NoExecute` + "`" + ` -- pod stays for N seconds before eviction.

**3. Common Use Cases:**
*   **Dedicated nodes:** Taint GPU nodes so only ML workloads run there.
*   **Spot/preemptible nodes:** Taint them so only fault-tolerant workloads use them.
*   **Control plane isolation:** Master nodes are tainted with ` + "`" + `node-role.kubernetes.io/control-plane:NoSchedule` + "`" + `.
*   **Maintenance:** Taint nodes before draining to stop new pods scheduling.

**4. Built-in Taints:**
*   Kubernetes automatically adds taints for node conditions:
    *   ` + "`" + `node.kubernetes.io/not-ready` + "`" + `
    *   ` + "`" + `node.kubernetes.io/unreachable` + "`" + `
    *   ` + "`" + `node.kubernetes.io/memory-pressure` + "`" + `
    *   ` + "`" + `node.kubernetes.io/disk-pressure` + "`" + ``,
					CodeExamples: `# Taint a node for GPU workloads only
kubectl taint nodes gpu-node-1 nvidia.com/gpu=true:NoSchedule

# Pod that tolerates the GPU taint
apiVersion: v1
kind: Pod
metadata:
  name: ml-training
spec:
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
  containers:
  - name: trainer
    image: pytorch/pytorch:latest
    resources:
      limits:
        nvidia.com/gpu: 1

---
# Tolerate ALL taints (useful for DaemonSets like monitoring)
tolerations:
- operator: "Exists"

---
# Temporary toleration (stay 300 seconds then evict)
tolerations:
- key: "node.kubernetes.io/not-ready"
  operator: "Exists"
  effect: "NoExecute"
  tolerationSeconds: 300

# Remove a taint
kubectl taint nodes gpu-node-1 nvidia.com/gpu=true:NoSchedule-`,
				},
				{
					Title: "Node Affinity",
					Content: `Node affinity constrains which nodes your pods can be scheduled on based on node labels. It is more expressive than ` + "`" + `nodeSelector` + "`" + ` and supports soft/hard preferences.

**1. Types:**
*   ` + "`" + `requiredDuringSchedulingIgnoredDuringExecution` + "`" + ` -- **Hard requirement.** The pod MUST be scheduled on a node matching the rules. If no node matches, the pod stays Pending.
*   ` + "`" + `preferredDuringSchedulingIgnoredDuringExecution` + "`" + ` -- **Soft preference.** The scheduler tries to place the pod on a matching node but will schedule anywhere if needed. Includes a weight (1-100) for prioritization.

**2. Operators:**
*   ` + "`" + `In` + "`" + ` -- Label value must be in the specified list.
*   ` + "`" + `NotIn` + "`" + ` -- Label value must NOT be in the list (anti-affinity for nodes).
*   ` + "`" + `Exists` + "`" + ` -- Label key must exist (any value).
*   ` + "`" + `DoesNotExist` + "`" + ` -- Label key must NOT exist.
*   ` + "`" + `Gt` + "`" + ` / ` + "`" + `Lt` + "`" + ` -- Greater/less than (for numeric labels).

**3. Common Labels:**
*   ` + "`" + `kubernetes.io/os` + "`" + ` -- ` + "`" + `linux` + "`" + ` or ` + "`" + `windows` + "`" + `.
*   ` + "`" + `kubernetes.io/arch` + "`" + ` -- ` + "`" + `amd64` + "`" + ` or ` + "`" + `arm64` + "`" + `.
*   ` + "`" + `topology.kubernetes.io/zone` + "`" + ` -- Cloud availability zone.
*   ` + "`" + `topology.kubernetes.io/region` + "`" + ` -- Cloud region.
*   ` + "`" + `node.kubernetes.io/instance-type` + "`" + ` -- Cloud instance type.

**4. IgnoredDuringExecution:**
*   All current affinity rules are "ignored during execution" -- they only apply at scheduling time. If node labels change after scheduling, existing pods are NOT evicted. Future Kubernetes versions may add ` + "`" + `RequiredDuringExecution` + "`" + `.`,
					CodeExamples: `# Hard requirement: must be on an amd64 Linux node in us-east-1
apiVersion: v1
kind: Pod
metadata:
  name: web-server
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: kubernetes.io/arch
            operator: In
            values: ["amd64"]
          - key: topology.kubernetes.io/region
            operator: In
            values: ["us-east-1"]
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 80
        preference:
          matchExpressions:
          - key: node.kubernetes.io/instance-type
            operator: In
            values: ["m5.xlarge", "m5.2xlarge"]
      - weight: 20
        preference:
          matchExpressions:
          - key: topology.kubernetes.io/zone
            operator: In
            values: ["us-east-1a"]
  containers:
  - name: web
    image: nginx:latest`,
				},
				{
					Title: "Pod Affinity and Anti-Affinity",
					Content: `Pod affinity and anti-affinity schedule pods based on the labels of OTHER pods already running on nodes. This controls pod co-location and distribution.

**1. Pod Affinity:**
*   "Schedule this pod on a node that already has pods matching label X."
*   Use case: Co-locate a web server with its cache for low latency.
*   ` + "`" + `topologyKey` + "`" + ` defines the scope: ` + "`" + `kubernetes.io/hostname` + "`" + ` (same node), ` + "`" + `topology.kubernetes.io/zone` + "`" + ` (same AZ).

**2. Pod Anti-Affinity:**
*   "Do NOT schedule this pod on a node that already has pods matching label X."
*   Use case: Spread replicas of the same service across nodes for high availability.
*   **Required** anti-affinity on hostname ensures no two replicas land on the same node.

**3. Hard vs Soft:**
*   ` + "`" + `requiredDuringSchedulingIgnoredDuringExecution` + "`" + ` -- Strict. Pod will be Pending if no valid node exists.
*   ` + "`" + `preferredDuringSchedulingIgnoredDuringExecution` + "`" + ` -- Best-effort. Scheduler tries but will compromise.

**4. Performance Warning:**
*   Pod affinity/anti-affinity is computationally expensive. The scheduler must scan all pods on all nodes.
*   In large clusters (1000+ nodes), use ` + "`" + `namespaceSelector` + "`" + ` to limit the scope.
*   Topology key ` + "`" + `kubernetes.io/hostname` + "`" + ` is the most expensive because every node is a separate domain.`,
					CodeExamples: `# Spread web replicas across nodes (HA)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      affinity:
        # ANTI-AFFINITY: Don't put 2 web pods on same node
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values: ["web"]
            topologyKey: kubernetes.io/hostname
        # AFFINITY: Prefer to be in same zone as cache
        podAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values: ["redis-cache"]
              topologyKey: topology.kubernetes.io/zone
      containers:
      - name: web
        image: nginx:latest`,
				},
				{
					Title: "Topology Spread Constraints",
					Content: `Topology Spread Constraints provide fine-grained control over how pods are distributed across failure domains. They are more flexible than pod anti-affinity for even distribution.

**1. The Problem:**
*   Pod anti-affinity only says "don't co-locate." It doesn't ensure EVEN distribution.
*   Example: With 3 zones and 6 replicas, anti-affinity might give 4-1-1 distribution. Topology spread constraints can enforce 2-2-2.

**2. Key Fields:**
*   ` + "`" + `maxSkew` + "`" + ` -- Maximum allowed difference in pod count between any two topology domains. ` + "`" + `maxSkew: 1` + "`" + ` means all zones must have within 1 pod of each other.
*   ` + "`" + `topologyKey` + "`" + ` -- The node label that defines the topology domain (e.g., ` + "`" + `topology.kubernetes.io/zone` + "`" + `).
*   ` + "`" + `whenUnsatisfiable` + "`" + `:
    *   ` + "`" + `DoNotSchedule` + "`" + ` -- Hard. Pod stays Pending if constraint can't be met.
    *   ` + "`" + `ScheduleAnyway` + "`" + ` -- Soft. Scheduler minimizes skew but allows violation.
*   ` + "`" + `labelSelector` + "`" + ` -- Select which pods to count when computing skew.

**3. Multiple Constraints:**
*   You can specify multiple constraints (e.g., spread across zones AND across nodes).
*   All constraints must be satisfied (logical AND).

**4. Cluster-Level Defaults (v1.24+):**
*   Administrators can set default topology spread constraints at the cluster level using ` + "`" + `--default-topology-spread-constraints` + "`" + ` in the scheduler config.`,
					CodeExamples: `# Evenly spread across zones AND nodes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: balanced-app
spec:
  replicas: 6
  selector:
    matchLabels:
      app: balanced-app
  template:
    metadata:
      labels:
        app: balanced-app
    spec:
      topologySpreadConstraints:
      # Spread evenly across zones (maxSkew 1 = within 1 pod difference)
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: balanced-app
      # Also spread across nodes within each zone
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: ScheduleAnyway
        labelSelector:
          matchLabels:
            app: balanced-app
      containers:
      - name: app
        image: myapp:latest
        resources:
          requests:
            cpu: 100m
            memory: 128Mi`,
				},
				{
					Title: "Priority Classes and Preemption",
					Content: `Priority classes determine the relative importance of pods. When cluster resources are exhausted, higher-priority pods can preempt (evict) lower-priority ones.

**1. PriorityClass Resource:**
*   Defines a priority level with a numeric value (higher = more important).
*   ` + "`" + `value` + "`" + ` ranges from -2,147,483,648 to 1,000,000,000. System pods use values above 1 billion.
*   ` + "`" + `globalDefault: true` + "`" + ` -- This priority applies to all pods that don't specify one (only ONE globalDefault allowed).
*   ` + "`" + `preemptionPolicy` + "`" + `: ` + "`" + `PreemptLowerPriority` + "`" + ` (default) or ` + "`" + `Never` + "`" + ` (high priority but no eviction).

**2. Preemption Process:**
1. Pod X cannot be scheduled due to insufficient resources.
2. Scheduler identifies nodes where evicting lower-priority pods would make room.
3. Scheduler selects the node with the least disruption (fewest evictions, least priority loss).
4. Lower-priority pods are given a graceful termination period, then evicted.
5. Pod X is scheduled.

**3. Built-in Priority Classes:**
*   ` + "`" + `system-cluster-critical` + "`" + ` (2,000,000,000) -- For cluster-level critical pods (e.g., kube-dns).
*   ` + "`" + `system-node-critical` + "`" + ` (2,000,001,000) -- For node-level critical pods (e.g., kube-proxy).

**4. Best Practices:**
*   Define 3-5 priority tiers: critical (production), high (staging), normal (default), low (batch jobs), idle (can be preempted anytime).
*   Set resource requests accurately. Preemption is based on requests, not actual usage.
*   Use Pod Disruption Budgets to protect critical workloads from excessive preemption.`,
					CodeExamples: `# Define priority classes
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: critical-production
value: 1000000
globalDefault: false
preemptionPolicy: PreemptLowerPriority
description: "For critical production services"

---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: batch-low
value: 100
preemptionPolicy: Never  # High priority but don't evict others
description: "For low-priority batch jobs"

---
# Use in a Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payment-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: payment
  template:
    metadata:
      labels:
        app: payment
    spec:
      priorityClassName: critical-production
      containers:
      - name: payment
        image: payment:latest
        resources:
          requests:
            cpu: 500m
            memory: 512Mi`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1717,
			Title:       "Kustomize & Configuration Management",
			Description: "Learn Kustomize for template-free Kubernetes configuration management: bases, overlays, patches, generators, and integration with kubectl.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Kustomize Fundamentals",
					Content: `Kustomize is a built-in Kubernetes configuration management tool (integrated into ` + "`" + `kubectl` + "`" + ` since v1.14). Unlike Helm, it works without templates -- it customizes YAML using overlays and patches.

**1. Philosophy:**
*   **Template-free.** Base manifests are valid Kubernetes YAML. No ` + "`" + `{{ .Values }}` + "`" + ` placeholders.
*   **Overlay-based.** Start with a base, then apply environment-specific modifications.
*   **Declarative.** The ` + "`" + `kustomization.yaml` + "`" + ` file describes WHAT to customize, not HOW.

**2. Directory Structure:**
` + "```" + `
├── base/
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   └── service.yaml
├── overlays/
│   ├── dev/
│   │   └── kustomization.yaml
│   ├── staging/
│   │   └── kustomization.yaml
│   └── prod/
│       └── kustomization.yaml
` + "```" + `

**3. The kustomization.yaml File:**
*   ` + "`" + `resources` + "`" + ` -- List of YAML files or directories to include.
*   ` + "`" + `namePrefix` + "`" + ` / ` + "`" + `nameSuffix` + "`" + ` -- Add prefix/suffix to all resource names.
*   ` + "`" + `commonLabels` + "`" + ` -- Add labels to ALL resources and selectors.
*   ` + "`" + `commonAnnotations` + "`" + ` -- Add annotations to all resources.
*   ` + "`" + `namespace` + "`" + ` -- Override namespace for all resources.

**4. Using Kustomize:**
*   ` + "`" + `kubectl apply -k overlays/prod/` + "`" + ` -- Apply directly.
*   ` + "`" + `kubectl kustomize overlays/prod/` + "`" + ` -- Preview the generated YAML.
*   ` + "`" + `kustomize build overlays/prod/ | kubectl apply -f -` + "`" + ` -- Standalone Kustomize.`,
					CodeExamples: `# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
- deployment.yaml
- service.yaml

# base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 1
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:latest
        ports:
        - containerPort: 8080

# overlays/prod/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
- ../../base
namePrefix: prod-
namespace: production
commonLabels:
  env: production
replicas:
- name: myapp
  count: 5`,
				},
				{
					Title: "Patches and Transformers",
					Content: `Patches allow you to modify specific parts of resources without replacing the entire file. Kustomize supports two patching strategies.

**1. Strategic Merge Patch:**
*   Merges your patch with the base resource using Kubernetes-aware merge logic.
*   For maps: new keys are added, existing keys are updated.
*   For lists: behavior depends on the field (some merge by key, some replace entirely).
*   Simpler to write. Use for most modifications.

**2. JSON6902 Patch:**
*   Precise, operation-based patching following RFC 6902.
*   Operations: ` + "`" + `add` + "`" + `, ` + "`" + `remove` + "`" + `, ` + "`" + `replace` + "`" + `, ` + "`" + `move` + "`" + `, ` + "`" + `copy` + "`" + `, ` + "`" + `test` + "`" + `.
*   More verbose but gives exact control over what changes.
*   Required when you need to modify arrays by index or remove specific elements.

**3. Inline Patches:**
*   Patches can be defined inline in ` + "`" + `kustomization.yaml` + "`" + ` or in separate files.
*   ` + "`" + `patches` + "`" + ` field supports both strategic merge and JSON6902 with target selectors.

**4. Components (v1.25+):**
*   Reusable groups of resources and patches that can be included in multiple overlays.
*   Example: A "monitoring" component that adds sidecar containers and configmaps.
*   Defined with ` + "`" + `kind: Component` + "`" + ` in the kustomization.yaml.`,
					CodeExamples: `# Strategic Merge Patch (overlays/prod/increase-resources.yaml)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  template:
    spec:
      containers:
      - name: myapp
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: "2"
            memory: 1Gi

# JSON6902 Patch -- add a sidecar container
# overlays/prod/add-sidecar.yaml
- op: add
  path: /spec/template/spec/containers/-
  value:
    name: log-shipper
    image: fluentbit:latest

# kustomization.yaml using both patch types
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
- ../../base
patches:
- path: increase-resources.yaml
- target:
    kind: Deployment
    name: myapp
  patch: |-
    - op: replace
      path: /spec/replicas
      value: 10`,
				},
				{
					Title: "ConfigMap and Secret Generators",
					Content: `Kustomize can generate ConfigMaps and Secrets automatically from files, literals, or .env files. A key feature is content-based hashing for rolling updates.

**1. ConfigMapGenerator:**
*   Creates ConfigMaps from files, directories, or literal key-value pairs.
*   Appends a content hash to the name (e.g., ` + "`" + `myapp-config-7h8f2k` + "`" + `).
*   When config content changes, the hash changes, triggering a rolling update of pods referencing it.

**2. SecretGenerator:**
*   Same as ConfigMapGenerator but creates Secrets.
*   Supports ` + "`" + `type` + "`" + ` field for specialized secrets (` + "`" + `kubernetes.io/tls` + "`" + `, ` + "`" + `kubernetes.io/dockerconfigjson` + "`" + `).

**3. Why Hashing Matters:**
*   Without hashing: You update a ConfigMap, but existing pods keep using the cached version. Pods must be manually restarted.
*   With hashing: The new hash creates a new ConfigMap name. The Deployment spec references the new name, triggering an automatic rolling update. Old ConfigMaps are garbage collected.

**4. Disabling Hashing:**
*   Set ` + "`" + `generatorOptions.disableNameSuffixHash: true` + "`" + ` if you need stable names (e.g., for StatefulSets or external references).`,
					CodeExamples: `# kustomization.yaml with generators
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
- deployment.yaml

configMapGenerator:
# From literal values
- name: app-config
  literals:
  - LOG_LEVEL=info
  - DB_HOST=postgres.default.svc
  - CACHE_TTL=300

# From a file
- name: nginx-config
  files:
  - nginx.conf

# From an env file
- name: env-config
  envs:
  - config.env

secretGenerator:
- name: db-credentials
  literals:
  - DB_USER=admin
  - DB_PASSWORD=supersecret
  type: Opaque

- name: tls-cert
  files:
  - tls.crt=cert.pem
  - tls.key=key.pem
  type: kubernetes.io/tls

generatorOptions:
  labels:
    generated-by: kustomize`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1718,
			Title:       "Service Mesh & Gateway API",
			Description: "Understand service mesh architecture, Istio/Linkerd basics, the Kubernetes Gateway API, mTLS, traffic management, and observability in mesh architectures.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "Service Mesh Architecture",
					Content: `A service mesh is a dedicated infrastructure layer for managing service-to-service communication. It uses sidecar proxies to provide networking features without application code changes.

**1. Why Service Mesh?**
*   As microservices grow from 10 to 100+ services, networking complexity explodes.
*   challenges: service discovery, load balancing, retries, timeouts, circuit breaking, mTLS, observability.
*   Without a mesh: Each service implements these features (or doesn't). Inconsistent, error-prone.
*   With a mesh: All networking logic is handled by the sidecar proxy. Apps just talk to localhost.

**2. Architecture:**
*   **Data Plane:** Sidecar proxies (typically Envoy) injected alongside every pod. They intercept all network traffic.
*   **Control Plane:** Manages and configures the proxy fleet. Distributes certificates, policies, and routing rules.

**3. Major Implementations:**
*   **Istio:** Most feature-rich. Uses Envoy proxies. Complex but powerful. Strong community.
*   **Linkerd:** Lightweight, built for simplicity. Uses its own Rust-based proxy (linkerd2-proxy). Lower resource overhead.
*   **Cilium Service Mesh:** eBPF-based (no sidecars for some features). Fast but newer.

**4. Tradeoffs:**
*   **Latency:** Each hop adds 1-5ms of proxy overhead.
*   **Resource usage:** Sidecar per pod means extra CPU/memory (50-100MB per sidecar).
*   **Complexity:** Significant operational overhead. Only adopt when the benefits outweigh the cost.
*   **Recommendation:** Start with Linkerd for simplicity. Move to Istio if you need advanced traffic management (fault injection, traffic shifting, virtual services).`,
					CodeExamples: `# Install Istio (minimal profile)
istioctl install --set profile=minimal

# Enable automatic sidecar injection for a namespace
kubectl label namespace default istio-injection=enabled

# Verify sidecar injection
kubectl get pods -o json | jq '.items[].spec.containers[].name'
# Should show "istio-proxy" alongside your app containers

# Install Linkerd (simpler alternative)
linkerd install --crds | kubectl apply -f -
linkerd install | kubectl apply -f -

# Inject Linkerd proxy into a deployment
kubectl get deploy myapp -o yaml | linkerd inject - | kubectl apply -f -

# Check mesh status
linkerd viz dashboard  # Opens web dashboard
istioctl analyze       # Istio configuration analysis`,
				},
				{
					Title: "mTLS and Traffic Policies",
					Content: `Mutual TLS (mTLS) encrypts all service-to-service traffic and verifies both client and server identities. Service meshes make mTLS transparent to applications.

**1. How mTLS Works in a Mesh:**
1.  The control plane acts as a Certificate Authority (CA) and issues X.509 certificates to each sidecar.
2.  Certificates are short-lived (typically 24h) and automatically rotated.
3.  When Service A calls Service B, both sidecars present their certificates.
4.  Traffic is encrypted in transit. No application code changes needed.

**2. Istio mTLS Modes:**
*   ` + "`" + `STRICT` + "`" + ` -- Only mTLS connections accepted. Non-mesh clients are rejected.
*   ` + "`" + `PERMISSIVE` + "`" + ` -- Accept both mTLS and plain text. Use during migration to mesh.
*   ` + "`" + `DISABLE` + "`" + ` -- No mTLS encryption.

**3. Traffic Management:**
*   **Traffic Splitting:** Route percentage of traffic to different versions (canary).
*   **Fault Injection:** Inject delays or errors for chaos testing.
*   **Circuit Breaking:** Limit concurrent connections to prevent cascade failures.
*   **Retries:** Automatic retry with backoff for transient failures.
*   **Timeouts:** Per-route timeout configuration.

**4. Authorization Policies:**
*   Define which services can communicate with which.
*   Layer 7 policies: Allow/deny based on HTTP method, path, headers.
*   Works with mTLS identity: "Only service A (verified by cert) can call service B on path /api."`,
					CodeExamples: `# Istio PeerAuthentication -- enforce mTLS
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: production
spec:
  mtls:
    mode: STRICT

---
# Traffic splitting (canary deployment)
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: myapp
spec:
  hosts:
  - myapp
  http:
  - route:
    - destination:
        host: myapp
        subset: v1
      weight: 90
    - destination:
        host: myapp
        subset: v2
      weight: 10

---
# Authorization policy
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-frontend-only
  namespace: production
spec:
  selector:
    matchLabels:
      app: backend-api
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/production/sa/frontend"]
    to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/api/*"]`,
				},
				{
					Title: "Gateway API",
					Content: `The Gateway API is the next-generation Kubernetes networking API, replacing Ingress. It provides more expressive, role-oriented, and portable traffic routing. It reached GA (v1.0) in October 2023.

**1. Why Replace Ingress?**
*   Ingress is limited: only HTTP/HTTPS, basic path routing, no header-based routing.
*   Vendor-specific annotations for every advanced feature (different for nginx, traefik, ALB).
*   No standard for TCP/UDP routing, traffic splitting, or request transformation.

**2. Key Resources:**
*   **GatewayClass:** Defines the controller (similar to IngressClass). Managed by infra admins.
*   **Gateway:** The actual load balancer instance. Defines listeners (ports, protocols, TLS). Managed by cluster operators.
*   **HTTPRoute:** Routing rules for HTTP traffic. Defines matches (path, header, method) and actions (forward, redirect, rewrite). Managed by app developers.
*   **TCPRoute / UDPRoute / GRPCRoute / TLSRoute:** Protocol-specific routes.

**3. Role-Oriented Design:**
*   Infrastructure provider: Manages GatewayClass.
*   Cluster operator: Creates Gateways, grants route attachment permissions.
*   Application developer: Creates HTTPRoutes and attaches them to Gateways.
*   This separation of concerns is a major improvement over Ingress.

**4. Advanced Features (Standard):**
*   Header-based routing.
*   Traffic splitting with weights (canary).
*   Request/response header modification.
*   URL rewriting and redirects.
*   Cross-namespace route attachment.`,
					CodeExamples: `# GatewayClass (installed by infra team)
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: istio
spec:
  controllerName: istio.io/gateway-controller

---
# Gateway (created by cluster operator)
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: production-gateway
  namespace: gateway-infra
spec:
  gatewayClassName: istio
  listeners:
  - name: https
    protocol: HTTPS
    port: 443
    tls:
      mode: Terminate
      certificateRefs:
      - name: wildcard-cert
    allowedRoutes:
      namespaces:
        from: Selector
        selector:
          matchLabels:
            gateway-access: "true"

---
# HTTPRoute (created by app developer)
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: myapp-route
  namespace: production
spec:
  parentRefs:
  - name: production-gateway
    namespace: gateway-infra
  hostnames:
  - "myapp.example.com"
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /api/v2
    backendRefs:
    - name: myapp-v2
      port: 8080
      weight: 90
    - name: myapp-v3
      port: 8080
      weight: 10
  - matches:
    - path:
        type: PathPrefix
        value: /api/v1
    backendRefs:
    - name: myapp-v1
      port: 8080`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1719,
			Title:       "Cluster Operations & Lifecycle",
			Description: "Manage Kubernetes cluster lifecycle: upgrades, etcd backup/restore, node management, debugging, and disaster recovery procedures.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "Cluster Upgrades",
					Content: `Kubernetes releases a new minor version every ~4 months. Keeping clusters up to date is critical for security patches, bug fixes, and new features.

**1. Version Skew Policy:**
*   ` + "`" + `kube-apiserver` + "`" + ` must be the newest component.
*   ` + "`" + `kubelet` + "`" + ` can be up to 3 minor versions behind apiserver (e.g., apiserver 1.29, kubelet 1.26).
*   ` + "`" + `kube-controller-manager` + "`" + ` and ` + "`" + `kube-scheduler` + "`" + ` can be 1 minor version behind.
*   ` + "`" + `kubectl` + "`" + ` can be 1 minor version ahead or behind apiserver.
*   **Never skip minor versions during upgrade** (1.27 → 1.28 → 1.29, NOT 1.27 → 1.29).

**2. kubeadm Upgrade Process:**
1. Read the changelog for breaking changes and deprecations.
2. Upgrade the control plane nodes first (one at a time if HA).
3. ` + "`" + `kubeadm upgrade plan` + "`" + ` -- shows available versions and preflight checks.
4. ` + "`" + `kubeadm upgrade apply v1.29.0` + "`" + ` -- upgrades control plane components.
5. Upgrade ` + "`" + `kubelet` + "`" + ` and ` + "`" + `kubectl` + "`" + ` on the control plane node.
6. Upgrade worker nodes one at a time (drain → upgrade → uncordon).

**3. Managed Kubernetes (EKS/GKE/AKS):**
*   Control plane upgrades are managed by the cloud provider.
*   Node group/pool upgrades are semi-automatic (rolling replacement of nodes).
*   Always test in staging first. Some add-ons may break across versions.

**4. Pre-Upgrade Checklist:**
*   Back up etcd.
*   Check API deprecations with ` + "`" + `kubectl get --raw /metrics | grep apiserver_requested_deprecated_apis` + "`" + `.
*   Verify all nodes are healthy.
*   Review PodDisruptionBudgets (ensure they allow node drain).
*   Test in a staging cluster first.`,
					CodeExamples: `# Check current cluster version
kubectl version --short

# View available upgrade
sudo kubeadm upgrade plan

# Upgrade control plane (on first control plane node)
sudo apt-get update
sudo apt-get install -y kubeadm=1.29.0-1.1
sudo kubeadm upgrade apply v1.29.0

# Upgrade kubelet and kubectl
sudo apt-get install -y kubelet=1.29.0-1.1 kubectl=1.29.0-1.1
sudo systemctl daemon-reload
sudo systemctl restart kubelet

# Upgrade worker nodes (one at a time)
# 1. Drain the node
kubectl drain node-1 --ignore-daemonsets --delete-emptydir-data

# 2. SSH to the node and upgrade
sudo apt-get install -y kubeadm=1.29.0-1.1
sudo kubeadm upgrade node
sudo apt-get install -y kubelet=1.29.0-1.1
sudo systemctl daemon-reload
sudo systemctl restart kubelet

# 3. Uncordon the node
kubectl uncordon node-1`,
				},
				{
					Title: "etcd Backup and Disaster Recovery",
					Content: `etcd is the single source of truth for all Kubernetes state. Losing etcd means losing the entire cluster configuration. Regular backups are non-negotiable.

**1. What etcd Stores:**
*   All Kubernetes objects (Pods, Services, Secrets, ConfigMaps, etc.).
*   RBAC rules and service accounts.
*   Custom Resource Definitions and their instances.
*   It does NOT store container images, persistent volume data, or application data.

**2. Backup Methods:**
*   ` + "`" + `etcdctl snapshot save` + "`" + ` -- Creates a point-in-time snapshot of the entire etcd database.
*   Schedule automated backups (every 1-6 hours depending on change frequency).
*   Store backups off-cluster (S3, GCS). Never store ONLY on the cluster itself.
*   Encrypt the backup at rest (it contains Secrets!).

**3. Restore Process:**
1. Stop the kube-apiserver (or all control plane components).
2. ` + "`" + `etcdctl snapshot restore backup.db --data-dir=/var/lib/etcd-restored` + "`" + `.
3. Update etcd configuration to point to the restored data directory.
4. Restart etcd and the control plane.
5. Verify cluster state: ` + "`" + `kubectl get nodes` + "`" + `, ` + "`" + `kubectl get pods --all-namespaces` + "`" + `.

**4. etcd Cluster Health:**
*   Monitor latency: ` + "`" + `etcdctl endpoint health` + "`" + `.
*   Watch database size: ` + "`" + `etcdctl endpoint status --write-out=table` + "`" + `.
*   etcd performs compaction automatically, but defragment periodically for large clusters.
*   Maximum recommended database size: 8 GB. If approaching, investigate what's consuming space.

**5. Managed Clusters:**
*   In EKS, GKE, AKS: etcd backup is handled by the provider. But always maintain separate backups of your manifests (GitOps).`,
					CodeExamples: `# Check etcd cluster health
ETCDCTL_API=3 etcdctl endpoint health \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key

# Create a snapshot backup
ETCDCTL_API=3 etcdctl snapshot save /backup/etcd-$(date +%Y%m%d).db \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key

# Verify backup integrity
ETCDCTL_API=3 etcdctl snapshot status /backup/etcd-20240101.db --write-out=table

# Restore from backup (DESTRUCTIVE -- stops the cluster)
ETCDCTL_API=3 etcdctl snapshot restore /backup/etcd-20240101.db \
  --data-dir=/var/lib/etcd-restored \
  --name=master-0 \
  --initial-cluster=master-0=https://10.0.0.1:2380

# Automated backup CronJob
apiVersion: batch/v1
kind: CronJob
metadata:
  name: etcd-backup
  namespace: kube-system
spec:
  schedule: "0 */6 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: bitnami/etcd:latest
            command: ["/bin/sh", "-c"]
            args:
            - etcdctl snapshot save /backup/etcd-$(date +%Y%m%d-%H%M).db
          restartPolicy: OnFailure`,
				},
				{
					Title: "Debugging Kubernetes Workloads",
					Content: `Debugging in Kubernetes requires a systematic approach. Problems can originate from the pod, the node, the network, or the control plane.

**1. Pod Debugging Hierarchy:**
*   ` + "`" + `kubectl describe pod <name>` + "`" + ` -- Shows events, conditions, and current state. Start here.
*   ` + "`" + `kubectl logs <pod>` + "`" + ` -- Application logs. Add ` + "`" + `-p` + "`" + ` for previous container (after crash).
*   ` + "`" + `kubectl logs <pod> -c <container>` + "`" + ` -- Specific container in multi-container pod.
*   ` + "`" + `kubectl exec -it <pod> -- /bin/sh` + "`" + ` -- Interactive shell for live debugging.

**2. Common Pod Issues:**
*   **CrashLoopBackOff:** Container exits repeatedly. Check logs (` + "`" + `kubectl logs -p` + "`" + `). Common causes: missing config, wrong command, OOM kill.
*   **ImagePullBackOff:** Can't pull the image. Check image name, tag, and registry auth (imagePullSecrets).
*   **Pending:** No node can schedule it. Check resource requests, node affinity, and taints.
*   **OOMKilled:** Container exceeded its memory limit. Increase limits or fix the memory leak.
*   **CreateContainerConfigError:** Missing ConfigMap or Secret referenced in the pod spec.

**3. Ephemeral Debug Containers (v1.25+):**
*   ` + "`" + `kubectl debug -it <pod> --image=busybox --target=<container>` + "`" + `.
*   Attaches a temporary debug container to a running pod.
*   Useful for distroless images that don't have shells or debugging tools.

**4. Network Debugging:**
*   ` + "`" + `kubectl run netshoot --rm -it --image=nicolaka/netshoot -- /bin/bash` + "`" + ` -- Temporary pod with networking tools.
*   Test DNS: ` + "`" + `nslookup <service-name>.<namespace>.svc.cluster.local` + "`" + `.
*   Test connectivity: ` + "`" + `curl <service-name>:<port>` + "`" + `.
*   Check endpoints: ` + "`" + `kubectl get endpoints <service>` + "`" + `.

**5. Node Debugging:**
*   ` + "`" + `kubectl describe node <name>` + "`" + ` -- Shows conditions (MemoryPressure, DiskPressure, PIDPressure).
*   ` + "`" + `kubectl top nodes` + "`" + ` -- Resource usage (requires metrics-server).
*   ` + "`" + `kubectl debug node/<name> -it --image=ubuntu` + "`" + ` -- Debug a node directly.`,
					CodeExamples: `# Debugging workflow for a failing pod
# Step 1: Check pod status and events
kubectl describe pod myapp-7f8b4c5d6-x9z2k

# Step 2: Check logs (current and previous)
kubectl logs myapp-7f8b4c5d6-x9z2k
kubectl logs myapp-7f8b4c5d6-x9z2k -p  # Previous container (after crash)

# Step 3: Check if endpoints exist (for service issues)
kubectl get endpoints myapp-service

# Step 4: Exec into the pod
kubectl exec -it myapp-7f8b4c5d6-x9z2k -- /bin/sh

# Step 5: Debug with ephemeral container (distroless images)
kubectl debug -it myapp-7f8b4c5d6-x9z2k --image=busybox --target=myapp

# Step 6: Network debugging from a temporary pod
kubectl run debug --rm -it --image=nicolaka/netshoot -- bash
# Inside the pod:
nslookup myapp-service.default.svc.cluster.local
curl -v myapp-service:8080/healthz

# Step 7: Debug the node itself
kubectl debug node/worker-1 -it --image=ubuntu
# Inside the debug pod (host filesystem at /host):
chroot /host
journalctl -u kubelet --since "1 hour ago"
crictl ps  # Check container runtime

# Check resource usage across the cluster
kubectl top pods --all-namespaces --sort-by=memory
kubectl top nodes`,
				},
				{
					Title: "Node Management and Maintenance",
					Content: `Proper node management is essential for cluster reliability. Nodes require periodic maintenance: OS patches, kernel upgrades, hardware replacement, and scaling.

**1. Cordon and Drain:**
*   ` + "`" + `kubectl cordon <node>` + "`" + ` -- Mark node as unschedulable. Existing pods keep running, new pods won't be placed here.
*   ` + "`" + `kubectl drain <node>` + "`" + ` -- Cordon + evict all pods (respecting PDBs).
*   Flags for drain:
    *   ` + "`" + `--ignore-daemonsets` + "`" + ` -- DaemonSet pods can't be drained (they exist on every node). Skip them.
    *   ` + "`" + `--delete-emptydir-data` + "`" + ` -- Allow pods with emptyDir volumes to be evicted (data will be lost).
    *   ` + "`" + `--force` + "`" + ` -- Delete pods not managed by a controller (standalone pods with no replica guarantee).
    *   ` + "`" + `--timeout=300s` + "`" + ` -- Maximum time to wait for eviction.
*   ` + "`" + `kubectl uncordon <node>` + "`" + ` -- Mark node as schedulable again.

**2. Node Labels and Taints:**
*   Label nodes for scheduling: ` + "`" + `kubectl label node worker-1 disktype=ssd` + "`" + `.
*   Taint nodes for isolation: ` + "`" + `kubectl taint node gpu-1 gpu=true:NoSchedule` + "`" + `.
*   Remove labels: ` + "`" + `kubectl label node worker-1 disktype-` + "`" + `.

**3. Node Auto-Repair (Managed Clusters):**
*   GKE/EKS/AKS can automatically replace unhealthy nodes.
*   Node auto-repair monitors node conditions and replaces nodes that fail health checks.

**4. Cluster Autoscaler:**
*   Automatically adds/removes nodes based on pending pods and resource utilization.
*   Works with cloud provider node groups/pools.
*   Respects PDBs during scale-down.
*   Key settings: ` + "`" + `--scale-down-utilization-threshold` + "`" + ` (default 0.5), ` + "`" + `--scale-down-unneeded-time` + "`" + ` (default 10m).`,
					CodeExamples: `# Graceful node maintenance workflow
# 1. Cordon the node (stop new pods)
kubectl cordon worker-3

# 2. Drain the node (evict existing pods)
kubectl drain worker-3 \
  --ignore-daemonsets \
  --delete-emptydir-data \
  --timeout=300s

# 3. Perform maintenance (SSH to node)
ssh worker-3
sudo apt-get update && sudo apt-get upgrade -y
sudo reboot

# 4. Verify node is ready
kubectl get node worker-3
# NAME       STATUS                     ROLES    AGE    VERSION
# worker-3   Ready,SchedulingDisabled   <none>   100d   v1.29.0

# 5. Uncordon the node
kubectl uncordon worker-3

# Label management
kubectl label node worker-1 disktype=ssd
kubectl label node worker-1 environment=production
kubectl label node worker-1 disktype-  # Remove label

# Cluster Autoscaler deployment (example)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  template:
    spec:
      containers:
      - name: cluster-autoscaler
        image: registry.k8s.io/autoscaling/cluster-autoscaler:v1.29.0
        command:
        - ./cluster-autoscaler
        - --cloud-provider=aws
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled
        - --scale-down-utilization-threshold=0.5
        - --scale-down-unneeded-time=10m`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
